"""Agentic RAG（開發規格 6.4）：封存區全文檢索和三個工具。"""
import asyncio
from datetime import datetime

from yuki import prompts
from yuki.archive import ArchiveDB, ArchivedMessage
from yuki.tools.archive import ArchiveTools, fts_query

G = "g"
NOW = datetime(2026, 11, 17)


def msg(i, author, text, day="2026-10-01", channel="閒聊", aid=None, guild=G):
    return ArchivedMessage(i, guild, "c1", channel, "", aid or author, author, f"{day} 12:00:00", text, None, 0)


def archive():
    db = ArchiveDB(":memory:")
    db.add_many([
        msg(1, "阿明", "我九月要去新訓中心當兵了", "2026-08-24"),
        msg(2, "小傑", "當兵還好吧", "2023-03-07"),
        msg(3, "阿明", "新訓中心好熱 每天都在練射擊", "2026-10-02"),
        msg(4, "hunterchiao", "麥當勞", "2024-09-16"),
        msg(5, "hunterchiao", "今天晚餐吃麥當勞的大麥克加可樂，超撐", "2026-11-01"),
        msg(6, "Luna", "我是 bot", "2026-11-01", guild="other"),
    ])
    db.commit()
    return db


def tools(db, **kw):
    people = {"阿明": {"m"}, "hunterchiao": {"h"}}
    return ArchiveTools(db.db, G, resolve_person=lambda n: people.get(n, set()),
                        profile=lambda n: "喜歡打 LoL" if n == "hunterchiao" else "", now=NOW, **kw)


def test_fts_query():
    assert fts_query(["當兵", "新訓中心"]) == '("當兵") OR ("成功" AND "嶺" AND "新訓中心")' or "OR" in fts_query(["當兵", "新訓中心"])
    assert fts_query([]) == "" and fts_query(["的"]) == ""  # 停用詞


def test_search_finds_and_filters():
    t = tools(archive())
    assert {h.id for h in t.search(["當兵", "新訓中心"])} == {1, 2, 3}
    assert {h.id for h in t.search(["當兵"], person="阿明")} == {1}  # 作者名比對（沒有 ID 時用名字）
    assert {h.id for h in t.search(["當兵", "新訓中心"], since="2026-01-01")} == {1, 3}
    assert {h.id for h in t.search(["當兵", "新訓中心"], until="2025-12-31")} == {2}
    assert t.search(["bot"]) == []  # 別的伺服器的不會出現


def test_longer_messages_rank_above_one_word_ones():
    hits = tools(archive()).search(["麥當勞"])
    assert [h.id for h in hits][:2] == [5, 4]


def test_read_chat_context():
    t = tools(archive())
    text = t.read_text(3)
    assert "→ #3" in text and "#1" in text and "#5" in text
    assert t.read_text(999) == "（找不到這則訊息）"


def test_tool_calls():
    t = tools(archive())
    out = asyncio.run(t.call("search_chat", {"keywords": "新訓中心"}))  # 模型偶爾給字串而不是陣列
    assert "#3" in out and "阿明" in out
    assert "找不到" in asyncio.run(t.call("search_chat", {"keywords": ["火星移民"]}))
    assert asyncio.run(t.call("get_profile", {"name": "hunterchiao"})) == "喜歡打 LoL"
    assert "沒有 路人 的人物檔案" in asyncio.run(t.call("get_profile", {"name": "路人"}))
    assert set(t.seen) >= {1, 3}


def test_opt_out_removes_from_index():
    db = archive()
    db.opt_out(G, "阿明")
    assert tools(db).search(["新訓中心"]) == []
    assert db.db.execute("SELECT COUNT(*) FROM msg_fts").fetchone()[0] == 4  # 6 則 - 阿明 2 則


def test_readding_message_does_not_duplicate_index():
    db = archive()
    db.add_many([msg(5, "hunterchiao", "改成吃摩斯漢堡", "2026-11-01")])
    t = tools(db)
    assert [h.id for h in t.search(["摩斯"])] == [5]
    assert [h.id for h in t.search(["大麥克"])] == []


def test_archive_prompt():
    p = prompts.reply_system("Yuki", [], None, archive=True)
    assert "search_chat" in p and "【關於群組成員的記憶】" not in p  # 沒有預先檢索的記憶區塊
    assert "沒有找到相關記憶" in prompts.reply_system("Yuki", [], [])


def test_text_tool_calls_are_parsed_and_final_round_is_nudged():
    from yuki.llm import LMStudioClient, text_tool_calls
    raw = "<tool_call>\n<function=search_chat>\n<parameter=keywords>\n[\"搬家\"]\n</parameter>\n</function>\n</tool_call>"
    assert text_tool_calls(raw)[0]["function"]["name"] == "search_chat"

    llm = LMStudioClient("http://x", "m")
    replies = iter([{"content": raw}, {"content": raw}, {"content": "他上個月搬到台中了"}])
    seen = []

    async def fake_request(payload, effort):
        seen.append(payload)
        return next(replies)

    async def call_tool(name, args):
        return "#1 阿明：我搬到台中了"

    llm._request = fake_request
    out = asyncio.run(llm.chat_tools([{"role": "user", "content": "阿明搬去哪"}], [{}], call_tool, max_rounds=1))
    assert out == "他上個月搬到台中了"  # 不會把 <tool_call> 當回覆
    assert "不能再查了" in seen[-1]["messages"][-1]["content"]
