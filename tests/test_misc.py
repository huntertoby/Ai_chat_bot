"""小工具：JSON 抽取、推理標籤、群聊轉 messages、長訊息切段、bot 身分組的 @。"""
from yuki.bot import split_message
from yuki.brain import to_messages
from yuki.chat import ChatLine, Person
from yuki.llm import extract_json, strip_think

TOBY = Person("2", "Toby")
BOT = Person("0", "Yuki")


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
