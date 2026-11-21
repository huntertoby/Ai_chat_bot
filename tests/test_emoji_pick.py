"""D-37：回話寫好後挑表情（不需要模型）。"""
import asyncio

from yuki import emoji_catalog as cat, emoji_pick as ep

ITEMS = [
    {"id": "11", "name": "happy", "kind": "emoji", "uses": 50, "meaning": "開心", "use": True},
    {"id": "12", "name": "smile", "kind": "emoji", "uses": 40, "meaning": "嘲諷的微笑", "use": True},
    {"id": "13", "name": "off", "kind": "emoji", "uses": 90, "meaning": "停用的", "use": False},
    {"id": "21", "name": "wave", "kind": "sticker", "uses": 0, "looks": "揮手", "use": True},
]
ESET = cat.EmojiSet("1", ITEMS)


class LLM:
    def __init__(self, result=None, *, error=None, delay=0.0):
        self.result, self.error, self.delay = result, error, delay
        self.calls = []

    async def chat_json(self, messages, schema, **kw):
        self.calls.append((messages, kw))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def run(llm, timeout=5.0):
    return asyncio.run(ep.pick(llm, ESET, "阿明：你很爛欸", "對啦我最爛", bot_name="Yuki", timeout=timeout))


def test_name_in_catalog_is_used():
    p = run(LLM({"emoji": "smile"}))
    assert p.entry is not None and p.entry.id == "12" and p.entry.kind == "emoji"
    assert run(LLM({"emoji": ":happy:"})).entry.id == "11"
    s = run(LLM({"emoji": "wave"}))
    assert s.entry.id == "21" and s.entry.kind == "sticker"
    assert run(LLM({"emoji": "[貼圖:wave]"})).entry.id == "21"


def test_unknown_empty_or_disabled_name_means_no_emoji():
    assert run(LLM({"emoji": ""})).entry is None and run(LLM({"emoji": ""})).reason == "none"
    p = run(LLM({"emoji": "not_there"}))
    assert p.entry is None and p.reason == "unknown"
    assert run(LLM({"emoji": "off"})).entry is None   # use=false 不在清單裡


def test_bad_json_means_no_emoji():
    for bad in (None, {}, {"emoji": 3}, "smile"):
        p = run(LLM(bad))
        assert p.entry is None and p.reason == "bad_json"


def test_error_or_timeout_means_no_emoji():
    p = run(LLM(error=RuntimeError("boom")))
    assert p.entry is None and p.reason == "error"
    p = run(LLM({"emoji": "smile"}, delay=2.0), timeout=1.05)
    assert p.entry is None and p.reason == "timeout" and p.ms < 1900


def test_no_call_when_not_enough_time_left():
    llm = LLM({"emoji": "smile"})
    p = run(llm, timeout=ep.PICK_MIN_SEC - 0.01)
    assert p.entry is None and p.reason == "no_time" and llm.calls == []
    assert ep.remaining(100.0, now=103.0) == ep.TEXT_REPLY_BUDGET_SEC - 3.0


def test_prompt_has_catalog_message_and_reply_but_not_disabled_items():
    llm = LLM({"emoji": ""})
    run(llm)
    (system, user), kw = llm.calls[0][0], llm.calls[0][1]
    assert ":happy: 開心" in system["content"] and "[貼圖:wave]" in system["content"]
    assert ":off:" not in system["content"] and "微笑" in system["content"]
    assert "你很爛欸" in user["content"] and "對啦我最爛" in user["content"]
    assert kw["name"] == "emoji_pick" and kw["reasoning_effort"] == "none"
