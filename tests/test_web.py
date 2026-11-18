import asyncio
import json

import pytest

from yuki import prompts
from yuki.llm import LMStudioClient
from yuki.tools.web import BlockedURL, check_public, clip, sample, youtube_id


# ---------- 網址 ----------
@pytest.mark.parametrize("url,vid", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/KdEJUze-ntw?si=abc", "KdEJUze-ntw"),
    ("https://youtube.com/shorts/w1zmqqMWvIc?si=x", "w1zmqqMWvIc"),
    ("https://www.youtube.com/watch?app=desktop&v=fIBjO3RljGw&t=10", "fIBjO3RljGw"),
    ("https://www.bilibili.com/video/BV1Z2sN6kEq1/", None),
])
def test_youtube_id(url, vid):
    assert youtube_id(url) == vid


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:1234/v1/models", "http://localhost/", "http://192.168.0.75:1234/",
    "http://100.87.188.89:1234/", "http://10.0.0.1/", "http://[::1]/", "file:///C:/Windows/win.ini",
    "ftp://example.com/",
])
def test_internal_urls_blocked(url):
    with pytest.raises(BlockedURL):
        check_public(url)


def test_clip_and_sample():
    assert clip("abc", 10) == "abc"
    assert clip("a" * 20, 10).startswith("a" * 10) and "省略" in clip("a" * 20, 10)
    parts = [f"第{i}句" for i in range(1000)]
    out = sample(parts, 600)
    assert "第0句" in out and "第999句" not in out[:100] and len(out) < 700
    assert "第8" in out  # 結尾那段也有抽到（最後一段從第 833 句開始）


# ---------- 工具呼叫流程 ----------
class ScriptedLLM(LMStudioClient):
    """依序回傳預先寫好的 message，記錄每次收到的 payload。"""

    def __init__(self, replies):
        super().__init__("http://x", "m")
        self.replies = list(replies)
        self.payloads = []

    async def _request(self, payload, reasoning_effort):
        self.payloads.append(payload)
        return self.replies.pop(0)


def tool_call(name, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_chat_tools_runs_tool_then_answers():
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [tool_call("web_search", {"query": "世界賽 冠軍"})]},
        {"content": "<think>嗯</think>BLG 拿了啦"},
    ])
    calls = []

    async def run(name, args):
        calls.append((name, args))
        return "BLG 3-1 Gen.G"

    out = asyncio.run(llm.chat_tools([{"role": "user", "content": "誰冠軍"}], [{"t": 1}], run))
    assert out == "BLG 拿了啦" and calls == [("web_search", {"query": "世界賽 冠軍"})]
    second = llm.payloads[1]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "BLG 3-1 Gen.G"}
    assert second[-2]["tool_calls"][0]["function"]["name"] == "web_search"


def test_chat_tools_last_round_has_no_tools():
    looping = {"content": "", "tool_calls": [tool_call("web_search", {"query": "x"})]}
    llm = ScriptedLLM([looping, looping, {"content": "算了不查了"}])

    async def run(name, args):
        return "結果"

    out = asyncio.run(llm.chat_tools([{"role": "user", "content": "q"}], [{"t": 1}], run, max_rounds=2))
    assert out == "算了不查了"
    assert ["tools" in p for p in llm.payloads] == [True, True, False]  # 最後一輪逼它回答


def test_chat_tools_bad_arguments_do_not_crash():
    bad = {"id": "c1", "function": {"name": "open_url", "arguments": "{not json"}}
    llm = ScriptedLLM([{"content": "", "tool_calls": [bad]}, {"content": "打不開"}])
    seen = []

    async def run(name, args):
        seen.append(args)
        return "沒有網址"

    assert asyncio.run(llm.chat_tools([{"role": "user", "content": "q"}], [], run)) == "打不開"
    assert seen == [{}]


def test_web_rules_only_when_enabled():
    assert "web_search" in prompts.reply_system("Yuki", [], [], web=True)
    assert "web_search" not in prompts.reply_system("Yuki", [], [])


def test_chat_tools_retries_without_thinking_when_empty():
    llm = ScriptedLLM([{"content": "", "reasoning_content": "想到一半"}, {"content": "我講了啊 漢堡"}])
    llm.reasoning_effort = "low"

    async def run(name, args):
        return ""

    assert asyncio.run(llm.chat_tools([{"role": "user", "content": "q"}], [{"t": 1}], run)) == "我講了啊 漢堡"
    assert "tools" not in llm.payloads[1]


def test_to_traditional():
    from yuki.brain import to_traditional
    assert to_traditional("对啊 我在回答你啊 不是汉堡是什么") == "對啊 我在回答你啊 不是漢堡是什麼"
    assert to_traditional("幹 乾你屁事 頭髮 發現") == "幹 乾你屁事 頭髮 發現"


def test_strip_name_prefix():
    from yuki.brain import _NAME_PREFIX
    assert _NAME_PREFIX.sub("", "[小傑] 幹 汪") == "幹 汪"
    assert _NAME_PREFIX.sub("", "汪 [小傑] 你叫") == "汪 [小傑] 你叫"


def test_thinking_off_also_sets_chat_template_kwargs():
    import httpx

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}, "finish_reason": "stop"}]})

    llm = LMStudioClient("http://x", "m", reasoning_effort="none")
    llm._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert asyncio.run(llm.chat([{"role": "user", "content": "q"}])) == "好"
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    llm2 = LMStudioClient("http://x", "m", reasoning_effort="low")
    llm2._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    asyncio.run(llm2.chat([{"role": "user", "content": "q"}]))
    assert "chat_template_kwargs" not in sent[1]


def _empty_llm(replies, effort="low"):
    from yuki.llm import LMStudioClient
    llm = LMStudioClient("http://x", "m", reasoning_effort=effort)
    seen = []
    it = iter(replies)

    async def fake_request(payload, eff):
        seen.append((payload, eff))
        return next(it)

    llm._request = fake_request
    return llm, seen


def test_chat_tools_empty_after_retry_asks_again_with_nudge():
    call = {"content": "", "tool_calls": [tool_call("web_search", {"query": "x"})]}
    llm, seen = _empty_llm([call, {"content": ""}, {"content": ""}, {"content": "查不到耶"}])

    async def run(name, args):
        return "（沒有結果）"

    out = asyncio.run(llm.chat_tools([{"role": "user", "content": "q"}], [{"t": 1}], run))
    assert out == "查不到耶"
    assert len(seen) == 4
    assert "tools" not in seen[3][0] and seen[3][1] == "none"
    assert "查不到" in seen[3][0]["messages"][-1]["content"]


def test_chat_tools_empty_gives_up_only_after_nudge():
    llm, seen = _empty_llm([{"content": ""}] * 4)

    async def run(name, args):
        return ""

    assert asyncio.run(llm.chat_tools([{"role": "user", "content": "q"}], [{"t": 1}], run)) == ""
    assert len(seen) == 4  # 原本、不思考、加提醒、攤平成一般訊息


def test_chat_tools_empty_with_effort_none_still_nudges():
    llm, seen = _empty_llm([{"content": ""}, {"content": "不知道欸"}], effort="none")

    async def run(name, args):
        return ""

    assert asyncio.run(llm.chat_tools([{"role": "user", "content": "q"}], [{"t": 1}], run)) == "不知道欸"
    assert len(seen) == 2


def test_chat_tools_flattens_results_when_model_keeps_calling_tools():
    call = {"content": "", "tool_calls": [tool_call("web_search", {"query": "x"})]}
    again = {"content": "<tool_call>\n<function=web_search>\n<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>"}
    llm, seen = _empty_llm([call, again, again, {"content": "沒查到欸"}], effort="none")

    async def run(name, args):
        return "這是查到的內容"

    out = asyncio.run(llm.chat_tools([{"role": "user", "content": "q"}], [{"t": 1}], run, max_rounds=1))
    assert out == "沒查到欸"
    last = seen[-1][0]["messages"]
    assert not any(m.get("role") == "tool" or m.get("tool_calls") for m in last)
    assert "這是查到的內容" in last[-1]["content"] and last[0] == {"role": "user", "content": "q"}
