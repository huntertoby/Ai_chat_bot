import asyncio

from tests.conftest import FakeLLM
from yuki.bot import split_message
from yuki.brain import to_messages
from yuki.llm import extract_json, strip_think
from yuki.memory.writer import ChatLine, MemoryWriter
from yuki.rag.decompose import Person
from yuki.rag.retrieve import Retriever

G = "g1"
TOBY = Person("2", "Toby")
BOT = Person("0", "Yuki")


def make_writer(store, embedder, reply, **kw):
    llm = FakeLLM(json_reply=reply)
    return MemoryWriter(store, embedder, llm, Retriever(store, embedder, None), **kw), llm


def add(store, embedder, text, subject=TOBY):
    vec = asyncio.run(embedder.embed_docs([text]))[0]
    return store.add(G, text, vec, subject_id=subject.id, subject_name=subject.name)


def test_writer_rejects_ids_outside_context(store, embedder):
    victim = add(store, embedder, "Toby 喜歡貓")
    writer, _ = make_writer(store, embedder, None)
    data = {"ops": [{"action": "delete", "id": victim.id, "subject": "Toby", "text": ""},
                    {"action": "update", "id": "deadbeef", "subject": "Toby", "text": "亂改"}],
            "whiteboard": ""}
    applied = asyncio.run(writer.apply(G, data, context=[], people=[TOBY]))
    assert applied == [] and store.get(victim.id) is not None


def test_writer_delete_cap(store, embedder):
    mems = [add(store, embedder, f"Toby 事實 {i}") for i in range(4)]
    writer, _ = make_writer(store, embedder, None, max_deletes=2)
    data = {"ops": [{"action": "delete", "id": m.id, "subject": "Toby", "text": ""} for m in mems],
            "whiteboard": ""}
    applied = asyncio.run(writer.apply(G, data, mems, [TOBY]))
    assert len(applied) == 2 and len(store.list(G)) == 2


def test_writer_add_prefixes_subject_and_dedups(store, embedder):
    writer, _ = make_writer(store, embedder, None, dedup_similarity=0.95)
    data = {"ops": [{"action": "add", "id": "", "subject": "toby", "text": "喜歡喝白色 Monster"}],
            "whiteboard": ""}
    first = asyncio.run(writer.apply(G, data, [], [TOBY]))
    assert first[0].action == "add" and first[0].text.startswith("Toby ")
    assert store.get(first[0].memory_id).subject_id == "2"
    # 完全相同的內容再加一次 → 不會產生第二筆
    asyncio.run(writer.apply(G, data, [], [TOBY]))
    assert len(store.list(G)) == 1


def test_writer_process_end_to_end(store, embedder):
    old = add(store, embedder, "Toby 喜歡喝白色 Monster")
    reply = {"ops": [{"action": "update", "id": old.id, "subject": "Toby", "text": "Toby 改喝綠色 Monster"}],
             "whiteboard": "群組在做 Yuki 機器人"}
    writer, llm = make_writer(store, embedder, reply)
    lines = [ChatLine(TOBY, "我最近都改喝綠色 Monster 了"), ChatLine(BOT, "綠色比較好喝嗎？", is_bot=True)]
    applied = asyncio.run(writer.process(G, lines))
    assert [a.action for a in applied] == ["update"]
    assert store.get(old.id).text == "Toby 改喝綠色 Monster"
    assert store.whiteboard(G)[0][1] == "群組在做 Yuki 機器人"
    # 舊記憶有被當成 context 交給 LLM，LLM 才知道該 update 哪個 ID
    assert old.id in llm.calls[0][1]["content"]


def test_writer_ignores_bot_only_batch(store, embedder):
    writer, llm = make_writer(store, embedder, {"ops": [], "whiteboard": ""})
    assert asyncio.run(writer.process(G, [ChatLine(BOT, "嗨", is_bot=True)])) == []
    assert llm.calls == []


def test_whiteboard_compaction_fallback(store, embedder):
    writer, _ = make_writer(store, embedder, None, whiteboard_max_items=3)
    for i in range(4):
        asyncio.run(writer.apply(G, {"ops": [], "whiteboard": f"近況 {i}"}, [], [TOBY]))
    assert len(store.whiteboard(G)) <= 3


def test_extract_json():
    assert extract_json('<think>{"x":1}</think>答案 {"ops": []}') == {"ops": []}
    assert extract_json('```json\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}
    assert extract_json("完全不是 JSON") is None
    assert extract_json("") is None


def test_strip_think():
    assert strip_think("<think>想很久</think>\n你好") == "你好"
    assert strip_think("想很久</think>你好") == "你好"


def test_to_messages_merges_group_chat():
    lines = [ChatLine(BOT, "早", is_bot=True), ChatLine(TOBY, "嗨"), ChatLine(Person("1", "Hunter"), "安"),
             ChatLine(BOT, "你們好", is_bot=True)]
    msgs = to_messages(lines)
    assert msgs[0] == {"role": "user", "content": "[Toby] 嗨\n[Hunter] 安"}
    assert msgs[1] == {"role": "assistant", "content": "你們好"}


def test_split_message():
    chunks = split_message("a" * 4000)
    assert all(len(c) <= 1900 for c in chunks) and "".join(chunks) == "a" * 4000
    assert split_message("短訊息") == ["短訊息"]


def test_is_called_by_bot_managed_role_mention():
    """在輸入框選到 bot 的身分組「@Yuki」也算叫她；別的身分組不算。"""
    import re
    from types import SimpleNamespace as NS
    from yuki.bot import YukiBot
    bot_role = NS(is_bot_managed=lambda: True)
    other = NS(is_bot_managed=lambda: False)
    me = NS(roles=[bot_role, other])
    fake = NS(user=object(), _wake=re.compile(r"(?!x)x"))
    def msg(roles):
        return NS(guild=NS(me=me), mentions=[], role_mentions=roles, content="<@&1> 看這個", reference=None)
    assert YukiBot.is_called(fake, msg([bot_role]))
    assert not YukiBot.is_called(fake, msg([other]))
    assert not YukiBot.is_called(fake, msg([]))
