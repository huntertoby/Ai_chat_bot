"""開發規格 6.5：被反駁的舊內容——模型提議、程式裁決、記紀錄、可還原。"""
import asyncio
import json

from tests.conftest import FakeLLM
from tests.test_profiles import G, make_archive
from yuki.profile import revise
from yuki.profile.builder import ProfileBuilder
from yuki.profile.store import Fact, ProfileDB, ProfileFiles

OLD = """## 基本資料
- 阿明 在台北做工程師
- 阿明 養了一隻貓
## 近況
- （2026-10）阿明 在準備轉職
## 經歷
- 2023：阿明 考上研究所
## 群組梗與綽號
以下是玩笑，不是事實：
- 大家叫阿明「富翁」
## 聊天提示
- （還不清楚）"""

FILE = "# 阿明\n> 自動整理（2026-11-01 更新，資料到 2026-10-20）。\n\n## 本人更正\n- （2026-09-01）阿明 說自己不吃辣\n\n" + OLD


def fact(text, day="2026-11-05", kind="fact", conf="high"):
    return Fact("1", "阿明", kind, text, conf, day)


def old_lines():
    return revise.numbered_lines(OLD)


def test_numbered_lines_skip_headings_placeholders_and_corrections():
    lines = revise.numbered_lines(revise.split_corrections(FILE)[1])
    assert [l.text for l in lines] == ["- 阿明 在台北做工程師", "- 阿明 養了一隻貓", "- （2026-10）阿明 在準備轉職",
                                       "- 2023：阿明 考上研究所", "- 大家叫阿明「富翁」"]
    assert lines[4].section == "## 群組梗與綽號"


def test_line_day():
    assert revise.line_day("- （2026-10）阿明 在準備轉職") == "2026-10-01"
    assert revise.line_day("- 2023：考上研究所，2025 搬到台中") == "2025-01-01"
    assert revise.line_day("- 阿明 養了一隻貓") == ""


def test_adjudicate_accepts_valid_and_rejects_bad():
    facts = [fact("阿明 說貓是室友的，他沒養", kind="correction"),        # 1
             fact("阿明 換到台中工作"),                                   # 2
             fact("小傑 說阿明 沒在工作", conf="low"),                    # 3 別人說的
             fact("阿明 月薪三百萬", kind="joke"),                        # 4
             fact("阿明 以前的事", day="2026-10-01")]                      # 5 比檔案舊
    props = [
        {"line": 2, "evidence": [1], "reason": "本人說沒養貓"},           # 准
        {"line": 1, "evidence": [2], "reason": "換工作了"},               # 准（沒日期 → 比資料到新）
        {"line": 99, "evidence": [2], "reason": "不存在"},                # 行不存在
        {"line": 3, "evidence": [7], "reason": "依據不存在"},             # 依據不在這批
        {"line": 3, "evidence": [3], "reason": "別人說"},                 # 不可信
        {"line": 3, "evidence": [4], "reason": "玩笑"},                   # 不可信
        {"line": 1, "evidence": [5], "reason": "舊的"},                   # 重複（第 1 行已准）
        {"line": 4, "evidence": [5], "reason": "比較舊"},                 # 2023 的行，5 是 2026 → 准? 看下面
        {"line": 5, "evidence": [2], "reason": "梗被反駁"},               # 梗只能本人更正刪
    ]
    v = revise.adjudicate(props, old_lines(), facts, "2026-10-20", limit=10)
    assert [x.ok for x in v] == [True, True, False, False, False, False, False, True, False]
    assert v[2].why == "要刪的行不在舊檔裡" and v[3].why == "依據不在這批新資訊裡"
    assert v[4].why.startswith("依據不可信") and v[5].why.startswith("依據不可信")
    assert v[6].why == "重複提議" and v[8].why == "梗只能由本人更正刪除"


def test_adjudicate_requires_newer_evidence():
    old = old_lines()
    # 沒日期的行：依據要不早於「資料到」
    assert not revise.adjudicate([{"line": 1, "evidence": [1], "reason": ""}], old,
                                 [fact("x", day="2026-10-19")], "2026-10-20")[0].ok
    # 有日期的行（2026-10）：依據要晚於 2026-10-01
    assert not revise.adjudicate([{"line": 3, "evidence": [1], "reason": ""}], old,
                                 [fact("x", day="2026-09-30")], "")[0].ok
    assert revise.adjudicate([{"line": 3, "evidence": [1], "reason": ""}], old,
                             [fact("x", day="2026-10-15")], "")[0].ok
    # 沒日期又沒有「資料到」：不刪
    v = revise.adjudicate([{"line": 1, "evidence": [1], "reason": ""}], old, [fact("x")], "")[0]
    assert not v.ok and "沒有日期" in v.why


def test_adjudicate_meme_only_by_self_correction_and_limit():
    old = old_lines()
    v = revise.adjudicate([{"line": 5, "evidence": [1], "reason": ""}], old,
                          [fact("阿明 說別再叫他富翁", kind="correction")], "2026-10-20")
    assert v[0].ok
    props = [{"line": n, "evidence": [1], "reason": ""} for n in (1, 2, 3, 4)]
    v = revise.adjudicate(props, old, [fact("阿明 說都不是真的", kind="correction")], "2026-10-20", limit=3)
    assert [x.ok for x in v] == [True, True, True, False] and v[3].why.startswith("超過")


def test_correction_block_cannot_be_targeted():
    body = revise.split_corrections(FILE)[1]
    assert all("不吃辣" not in l.text for l in revise.numbered_lines(body))
    # 就算模型直接把本人更正的行塞進 numbered 清單（不可能，但保險）：區塊來源也是程式的
    fake = [revise.OldLine(1, revise.CORRECTION_HEAD, "- （2026-09-01）阿明 說自己不吃辣")]
    v = revise.adjudicate([{"line": 1, "evidence": [1], "reason": ""}], fake,
                          [fact("阿明 說自己愛吃辣", kind="correction")], "2026-08-01")
    assert not v[0].ok and v[0].why == "本人更正不能刪"


def test_remove_and_insert_line_roundtrip():
    gone = revise.remove_lines(OLD, [old_lines()[1]])
    assert "養了一隻貓" not in gone and "在台北做工程師" in gone
    back = revise.insert_line(gone, "## 基本資料", "- 阿明 養了一隻貓")
    assert back.index("養了一隻貓") < back.index("## 近況")
    assert "- 阿明 養了" in revise.insert_line(OLD, "## 聊天提示", "- 阿明 養了").split("## 聊天提示")[1]
    assert "（還不清楚）" not in revise.insert_line(OLD, "## 聊天提示", "- 新的")


class TwoStepLLM(FakeLLM):
    """chat_json → 刪除提議；chat → 合併結果（回傳收到的「目前的檔案」，看刪除有沒有先套用）。"""

    async def chat(self, messages, **kw):
        self.calls.append(messages)
        user = messages[1]["content"]
        return user.split("【目前的檔案】\n", 1)[1].split("\n\n【新整理出的資訊】")[0]


def _builder(tmp_path, proposals, facts):
    archive = make_archive([(1, "1", "阿明", "2026-11-05 10:00:00", "hi")])
    db = ProfileDB(":memory:")
    files = ProfileFiles(tmp_path)
    files.write(G, "1", FILE)
    db.save_segment(G, "c1", 1, facts)
    llm = TwoStepLLM(json_reply={"remove": proposals})
    return ProfileBuilder(archive, db, files, llm, G), files, db


def test_merge_applies_approved_removal_and_logs(tmp_path):
    facts = [fact("阿明 說貓是室友的，他沒養", kind="correction"), fact("小傑 說阿明 辭職了", conf="low")]
    b, files, db = _builder(tmp_path, [{"line": 2, "evidence": [1], "reason": "本人說沒養"},
                                       {"line": 1, "evidence": [2], "reason": "別人說的"}], facts)
    assert asyncio.run(b.merge_one("1"))
    md = files.read(G, "1")
    assert "養了一隻貓" not in md and "在台北做工程師" in md     # 第二個提議的依據是別人說的：不刪
    # 本人更正區塊：舊的 + 新的都在，在最前面
    assert md.index("不吃辣") < md.index("## 基本資料") and "貓是室友的" in md.split("## 基本資料")[0]
    recs = files.removed(G, "1")
    assert recs[0]["line"] == "- 阿明 養了一隻貓" and recs[0]["evidence"][0]["kind"] == "correction"
    # 審查的 prompt 有編號的舊條目和新資訊
    review_user = b.llm.calls[0][1]["content"]
    assert "2. 〔基本資料〕阿明 養了一隻貓" in review_user and "#1 [2026-11-05][correction]" in review_user
    assert "不吃辣" not in review_user  # 本人更正不給模型當刪除目標


def test_merge_rejected_proposals_leave_file(tmp_path):
    facts = [fact("小傑 說阿明 沒養貓", conf="low"), fact("阿明 月薪三百萬", kind="joke")]
    b, files, _ = _builder(tmp_path, [{"line": 2, "evidence": [1], "reason": "別人說"},
                                      {"line": 2, "evidence": [2], "reason": "玩笑"},
                                      {"line": 5, "evidence": [9], "reason": "不存在"}], facts)
    asyncio.run(b.merge_one("1"))
    md = files.read(G, "1")
    assert "養了一隻貓" in md and "富翁" in md and files.removed(G, "1") == []


def test_merge_failure_does_not_log_removal(tmp_path):
    facts = [fact("阿明 說貓是室友的", kind="correction")]
    b, files, db = _builder(tmp_path, [{"line": 2, "evidence": [1], "reason": "x"}], facts)

    async def broken(messages, **kw):
        return "格式不對"
    b.llm.chat = broken
    assert not asyncio.run(b.merge_one("1"))
    assert "養了一隻貓" in files.read(G, "1") and files.removed(G, "1") == []
    assert len(db.pending(G, "1")) == 1


def test_review_bad_json_means_no_removal(tmp_path):
    b, files, _ = _builder(tmp_path, None, [fact("阿明 說貓是室友的", kind="correction")])
    b.llm.json_reply = None
    asyncio.run(b.merge_one("1"))
    assert "養了一隻貓" in files.read(G, "1") and files.removed(G, "1") == []


def test_removal_limit_across_batches(tmp_path):
    facts = [fact(f"阿明 說第 {i} 條都不對 " + "字" * 60, kind="correction") for i in range(6)]
    b, files, _ = _builder(tmp_path, [{"line": n, "evidence": [1], "reason": ""} for n in (1, 2, 3, 4, 5)], facts)
    b.merge_batch_chars = 200  # 拆成好幾批
    asyncio.run(b.merge_one("1"))
    assert len(files.removed(G, "1")) == revise.MAX_REMOVALS


def test_delete_removes_log_too(tmp_path):
    files = ProfileFiles(tmp_path)
    files.write(G, "1", FILE)
    files.log_removed(G, "1", [{"line": "- x"}])
    assert files.removed_path(G, "1").exists()
    assert files.delete_everywhere("1") == 1
    assert not files.removed_path(G, "1").exists() and not files.path(G, "1").exists()


def test_restore_cmd_puts_line_back(tmp_path, capsys):
    from types import SimpleNamespace
    from scripts.build_profiles import restore_cmd
    files = ProfileFiles(tmp_path)
    files.write(G, "1", revise.remove_lines(FILE, [old_lines()[1]]))
    files.log_removed(G, "1", [{"at": "t", "section": "## 基本資料", "line": "- 阿明 養了一隻貓", "reason": "r",
                                "evidence": [{"day": "2026-11-05", "kind": "correction", "text": "x"}]}])
    restore_cmd(SimpleNamespace(cmd="removed", only="阿明", index=None), G, files, {"1": "阿明"})
    assert "養了一隻貓" in capsys.readouterr().out
    restore_cmd(SimpleNamespace(cmd="restore", only="阿明", index=1), G, files, {"1": "阿明"})
    assert "養了一隻貓" in files.read(G, "1").split("## 近況")[0]
    assert files.removed(G, "1")[0]["restored"] is True
    assert json.loads(files.removed_path(G, "1").read_text(encoding="utf-8").splitlines()[0])["restored"]


def test_numbered_lines_accepts_star_and_numbered_bullets():
    body = "## 基本資料\n*   阿明 是大學生\n    *   細節\n## 聊天提示\n1. 可以聊羽球\n*   （還不清楚）"
    lines = revise.numbered_lines(body)
    assert [l.text for l in lines] == ["*   阿明 是大學生", "    *   細節", "1. 可以聊羽球"]
    from yuki.profile.prompts import review_user
    assert "1. 〔基本資料〕阿明 是大學生" in review_user("阿明", [(l.no, l.section, l.text) for l in lines], ["- x"], "")
    assert revise.split_corrections("## 本人更正\n*   （2026-01-01）x\n## 基本資料")[0] == ["*   （2026-01-01）x"]
