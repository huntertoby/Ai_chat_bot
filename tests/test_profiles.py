import asyncio
import sqlite3

from tests.conftest import FakeLLM
from yuki.archive import ArchiveDB, ArchivedMessage
from yuki.profile.builder import ProfileBuilder
from yuki.profile.chunk import clean, segments, to_lines
from yuki.profile.store import ProfileDB, ProfileFiles

G = "g"


def row(i, author, name, at, text):
    return (i, author, name, at, text)


def test_clean_strips_emoji_ids_and_urls():
    assert clean("笑死 <:R_:1538152850918481980>") == "笑死 :R_:"
    assert clean("看 https://www.youtube.com/watch?v=abc 這個") == "看 [連結:www.youtube.com] 這個"
    assert clean("a\n\nb") == "a / b"


def test_to_lines_merges_bursts_and_drops_empty():
    rows = [row(1, "1", "A", "2024-01-01 10:00:00", "早"),
            row(2, "1", "A", "2024-01-01 10:01:00", "安"),
            row(3, "1", "A", "2024-01-01 10:01:30", ""),
            row(4, "2", "B", "2024-01-01 10:02:00", "嗨"),
            row(5, "1", "A", "2024-01-01 12:00:00", "午安")]
    lines = to_lines(rows)
    assert [(l.author_name, l.text, l.msg_id) for l in lines] == [("A", "早 / 安", 2), ("B", "嗨", 4), ("A", "午安", 5)]


def test_segments_respect_size_and_prefer_gaps():
    rows = [row(i, str(i % 2), f"U{i % 2}", f"2024-01-01 {10 + i // 10:02d}:{i % 10:02d}:00", "x" * 50)
            for i in range(40)]
    rows.append(row(100, "0", "U0", "2024-01-02 09:00:00", "隔天"))
    segs = segments(to_lines(rows), "c", "聊天", max_chars=1500)
    assert all(s.chars() <= 1500 for s in segs)
    assert segs[-1].lines[-1].text == "隔天"
    assert "--- 2024-01-02 ---" in segs[-1].render()


def make_archive(msgs):
    db = ArchiveDB(":memory:")
    db.add_many([ArchivedMessage(i, G, "c1", "聊天", "", a, n, at, t, None, 0) for i, a, n, at, t in msgs])
    db.commit()
    return db.db


def test_extract_classifies_validates_and_resumes(tmp_path):
    archive = make_archive([(1, "1", "阿明", "2024-03-01 10:00:00", "我考上研究所了"),
                            (2, "2", "Hunter", "2024-03-01 10:01:00", "阿明月薪三百萬 XD")])
    reply = {"items": [
        {"subject": "阿明", "kind": "fact", "text": "阿明 考上研究所", "confidence": "high", "day": "2024-03-01"},
        {"subject": "阿明", "kind": "joke", "text": "阿明 月薪三百萬", "confidence": "low", "day": "1999-01-01"},
        {"subject": "路人", "kind": "fact", "text": "路人 不是成員", "confidence": "high", "day": "2024-03-01"},
        {"subject": "Hunter", "kind": "亂填", "text": "x", "confidence": "high", "day": "2024-03-01"},
    ]}
    llm = FakeLLM(json_reply=reply)
    db = ProfileDB(":memory:")
    b = ProfileBuilder(archive, db, ProfileFiles(tmp_path), llm, G)
    assert asyncio.run(b.extract()) == 1
    facts = [f for _, f in db.pending(G, "1")]
    assert [(f.kind, f.day) for f in facts] == [("fact", "2024-03-01"), ("joke", "2024-03-01")]
    assert db.pending(G, "2") == []
    # 抽過的段落不會再抽
    assert asyncio.run(b.extract()) == 0 and len(llm.calls) == 1
    # 聊天紀錄有送進 prompt，名單有成員
    assert "阿明: 我考上研究所了" in llm.calls[0][1]["content"]


def test_merge_writes_md_and_marks_merged(tmp_path):
    archive = make_archive([(1, "1", "阿明", "2024-03-01 10:00:00", "hi")])
    db = ProfileDB(":memory:")
    files = ProfileFiles(tmp_path)
    from yuki.profile.store import Fact
    db.save_segment(G, "c1", 1, [Fact("1", "阿明", "fact", f"阿明 事實 {i}", "high", "2024-03-01") for i in range(6)])
    llm = FakeLLM(text_reply="好的，以下是檔案：\n## 基本資料\n- 阿明 是研究生\n## 群組梗與綽號\n以下是玩笑，不是事實：")
    b = ProfileBuilder(archive, db, files, llm, G)
    assert asyncio.run(b.merge()) == 1
    md = files.read(G, "1")
    assert md.startswith("# 阿明") and "## 基本資料" in md and "好的，以下是檔案" not in md
    assert db.pending(G, "1") == []
    prompt = files.for_prompt(G, "1", 1000)
    assert prompt.startswith("## 基本資料") and "# 阿明" not in prompt


def test_merge_rejects_bad_output(tmp_path):
    archive = make_archive([(1, "1", "阿明", "2024-03-01 10:00:00", "hi")])
    db = ProfileDB(":memory:")
    from yuki.profile.store import Fact
    db.save_segment(G, "c1", 1, [Fact("1", "阿明", "fact", f"阿明 事實 {i}", "high", "2024-03-01") for i in range(6)])
    b = ProfileBuilder(archive, db, ProfileFiles(tmp_path), FakeLLM(text_reply="我不知道"), G)
    asyncio.run(b.merge())
    assert len(db.pending(G, "1")) == 6 and ProfileFiles(tmp_path).read(G, "1") == ""


def test_forget_and_path_is_safe(tmp_path):
    files = ProfileFiles(tmp_path)
    files.write(G, "../../evil", "x")
    assert (tmp_path / G / "evil.md").exists()
    db = ProfileDB(":memory:")
    from yuki.profile.store import Fact
    db.save_segment(G, "c1", 1, [Fact("1", "A", "fact", "A x", "high", "2024-01-01")])
    assert db.forget(G, "1") == 1 and db.stats(G) == []


def test_reply_system_includes_profiles():
    from yuki.prompts import reply_system
    text = reply_system("Yuki", profiles=[("阿明", "## 基本資料\n- 研究生")])
    assert "【人物檔案：阿明】" in text and "群組梗與綽號" in text


# ---------- 本人更正（開發規格 6.5）----------
def _extract_with(tmp_path, msgs, items):
    archive = make_archive(msgs)
    db = ProfileDB(":memory:")
    b = ProfileBuilder(archive, db, ProfileFiles(tmp_path), FakeLLM(json_reply={"items": items}), G)
    asyncio.run(b.extract())
    return db


def test_correction_kept_only_when_said_by_self(tmp_path):
    msgs = [(1, "1", "阿明", "2024-03-01 10:00:00", "我才沒有在養貓，那是我室友的"),
            (2, "2", "小傑", "2024-03-01 10:01:00", "阿明 根本沒在打工啦，他早就辭了")]
    items = [
        {"subject": "阿明", "kind": "correction", "text": "阿明 說自己沒有養貓，貓是室友的", "confidence": "low",
         "day": "2024-03-01", "quote": "我才沒有在養貓"},
        # 別人替他更正：原句是小傑說的 → 不算
        {"subject": "阿明", "kind": "correction", "text": "阿明 沒在打工", "confidence": "high",
         "day": "2024-03-01", "quote": "阿明 根本沒在打工啦"},
        # 原句不存在（模型編的）→ 不算
        {"subject": "阿明", "kind": "correction", "text": "阿明 不住台北", "confidence": "high",
         "day": "2024-03-01", "quote": "我不住台北"},
    ]
    db = _extract_with(tmp_path, msgs, items)
    got = db.corrections(G, "1")
    assert [f.text for f in got] == ["阿明 說自己沒有養貓，貓是室友的"]
    assert got[0].confidence == "high"


def _md(body="## 基本資料\n- 阿明 是研究生\n## 群組梗與綽號\n以下是玩笑，不是事實："):
    return body


def test_correction_block_written_kept_on_rebuild_and_in_prompt(tmp_path):
    from yuki.profile.store import Fact
    archive = make_archive([(1, "1", "阿明", "2024-03-01 10:00:00", "hi")])
    db = ProfileDB(":memory:")
    files = ProfileFiles(tmp_path)
    db.save_segment(G, "c1", 1, [Fact("1", "阿明", "fact", f"阿明 事實 {i}", "high", "2024-03-01") for i in range(5)]
                    + [Fact("1", "阿明", "correction", "阿明 說自己沒有養貓", "high", "2024-03-02")])
    # 模型自己也寫了一份（改過字的）本人更正：要被程式的版本取代
    llm = FakeLLM(text_reply="## 本人更正\n- 阿明 有養貓\n" + _md())
    b = ProfileBuilder(archive, db, files, llm, G)
    assert asyncio.run(b.merge()) == 1
    md = files.read(G, "1")
    assert "## 本人更正" in md and "- （2024-03-02）阿明 說自己沒有養貓" in md and "阿明 有養貓" not in md
    assert md.index("## 本人更正") < md.index("## 基本資料")
    # 合併 prompt 拿得到更正，而且更正不混在新資訊裡
    user = llm.calls[0][1]["content"]
    assert user.startswith("【本人更正】") and "阿明 說自己沒有養貓" in user
    assert "[correction]" not in user
    # 回覆 prompt：本人更正在最前面
    assert files.for_prompt(G, "1", 1000).startswith("## 本人更正")
    # 重建（reset_merged 再合併，模型這次沒寫更正）：區塊還在
    db.reset_merged(G, "1")
    b.llm = FakeLLM(text_reply=_md())
    asyncio.run(b.merge())
    assert "阿明 說自己沒有養貓" in files.read(G, "1")
    # 整個刪掉重建（remerge）：從資料庫重建區塊
    files.delete(G, "1")
    db.reset_merged(G, "1")
    asyncio.run(b.merge())
    assert "- （2024-03-02）阿明 說自己沒有養貓" in files.read(G, "1")


def test_correction_block_from_old_file_survives_merge(tmp_path):
    from yuki.profile.store import Fact
    archive = make_archive([(1, "1", "阿明", "2024-03-01 10:00:00", "hi")])
    db = ProfileDB(":memory:")
    files = ProfileFiles(tmp_path)
    files.write(G, "1", "# 阿明\n> 自動整理\n\n## 本人更正\n- （2023-01-01）阿明 說自己不是工程師\n\n" + _md())
    db.save_segment(G, "c1", 1, [Fact("1", "阿明", "fact", "阿明 事實", "high", "2024-03-01")])
    b = ProfileBuilder(archive, db, files, FakeLLM(text_reply=_md()), G)
    asyncio.run(b.merge_one("1"))
    assert "- （2023-01-01）阿明 說自己不是工程師" in files.read(G, "1")
    assert "本人說的" in b.llm.calls[0][1]["content"]  # 舊檔的更正有給模型看
    assert "## 本人更正" not in b.llm.calls[0][1]["content"].split("【目前的檔案】")[1]


def test_reply_system_marks_profile_as_impression():
    from yuki.prompts import reply_system
    text = reply_system("Yuki", profiles=[("阿明", "## 基本資料\n- 研究生")])
    assert "【人物檔案：阿明】（你對這個人的印象，可能有錯" in text
