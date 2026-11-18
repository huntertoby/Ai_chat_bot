"""AI 追蹤紀錄（yuki/trace.py）：不需要網路、模型、Discord。"""
import asyncio
import json
import logging
from datetime import date, datetime, timedelta

import pytest

from yuki import trace
from yuki.llm import LLMError, LMStudioClient


@pytest.fixture(autouse=True)
def _reset():
    trace.reset()
    yield
    trace.reset()


def lines(tmp_path):
    out = []
    for p in sorted(tmp_path.glob("ai-*.jsonl")):
        out += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    return out


def run(coro):
    return asyncio.run(coro)


def test_normal_trace_writes_one_line_with_all_fields(tmp_path):
    trace.configure(tmp_path)

    async def go():
        async with trace.start("reply", guild="g1", channel="c1", trigger="called", user_ids=["u1"]) as tr:
            tr.add_attempt([{"role": "user", "content": "嗨"}], {"content": "你好"}, 12)
            tr.add_tool("web_search", {"query": "x"}, "結果")
            tr.lat("asr", 30)
            tr.degrade("tts_edge")
            tr.degrade("tts_edge")
            tr.final = "你好"

    run(go())
    (rec,) = lines(tmp_path)
    assert set(rec) == {"trace_id", "ts", "kind", "guild", "channel", "trigger", "outcome", "degraded",
                        "latency_ms", "attempts", "final", "error"}
    assert rec["kind"] == "reply" and rec["outcome"] == "ok" and rec["final"] == "你好"
    assert rec["degraded"] == ["tts_edge"] and len(rec["trace_id"]) == 8
    assert rec["latency_ms"]["asr"] == 30 and rec["latency_ms"]["llm"] == 12 and "total" in rec["latency_ms"]
    assert rec["attempts"][0]["tools"][0] == {"name": "web_search", "args": {"query": "x"}, "result": "結果"}
    assert rec["error"] is None


@pytest.mark.parametrize("exc", [RuntimeError("壞了"), asyncio.CancelledError()])
def test_exception_still_writes_and_propagates(tmp_path, exc):
    trace.configure(tmp_path)

    async def go():
        async with trace.start("voice", guild="g", channel="c", trigger="voice"):
            raise exc

    with pytest.raises(type(exc)):
        run(go())
    (rec,) = lines(tmp_path)
    assert rec["outcome"] == "error" and type(exc).__name__ in rec["error"]


def test_current_trace_is_cleared_after_exit(tmp_path):
    trace.configure(tmp_path)

    async def go():
        assert trace.current() is None
        async with trace.start("reply") as tr:
            assert trace.current() is tr and trace.current_id() == tr.trace_id
        assert trace.current() is None

    run(go())


def test_opted_out_user_not_written(tmp_path):
    trace.configure(tmp_path, opted_out=lambda g, uids: "u2" in uids)

    async def go():
        async with trace.start("reply", guild="g", user_ids=["u1"]):
            pass
        async with trace.start("reply", guild="g", user_ids=["u1", "u2"]):
            pass

    run(go())
    assert len(lines(tmp_path)) == 1


def test_write_failure_does_not_break_reply(tmp_path, caplog):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    trace.configure(blocker)  # 目錄位置被檔案佔住，寫檔一定失敗

    async def go():
        async with trace.start("reply") as tr:
            tr.final = "照樣回"
        return tr.final

    with caplog.at_level(logging.WARNING, logger="yuki.trace"):
        assert run(go()) == "照樣回"
    assert "寫入失敗" in caplog.text


def test_unconfigured_does_not_write(tmp_path):
    async def go():
        async with trace.start("reply"):
            pass

    run(go())
    assert not list(tmp_path.iterdir())


# ---------- 保留 ----------
def touch(d, day, size=10):
    p = d / f"ai-{day.isoformat()}.jsonl"
    p.write_bytes(b"x" * size)
    return p


def test_prune_by_days(tmp_path):
    today = date(2026, 6, 20)
    old, edge, new = (touch(tmp_path, today - timedelta(days=n)) for n in (15, 14, 1))
    other = tmp_path / "notes.txt"
    other.write_text("別的檔案不動")
    removed = trace.prune(tmp_path, max_days=14, max_mb=200, today=today)
    assert removed == [old.name]
    assert edge.exists() and new.exists() and other.exists()


def test_prune_by_size_deletes_oldest_first_and_keeps_today(tmp_path):
    today = date(2026, 6, 20)
    mb = 1024 * 1024
    files = [touch(tmp_path, today - timedelta(days=n), mb) for n in (3, 2, 1, 0)]
    trace.prune(tmp_path, max_days=14, max_mb=2, today=today)
    assert [p.exists() for p in files] == [False, False, True, True]
    trace.prune(tmp_path, max_days=14, max_mb=0, today=today)  # 再小也不刪今天的
    assert [p.exists() for p in files] == [False, False, False, True]


def test_configure_prunes_and_daily_check(tmp_path):
    now = [datetime(2026, 6, 20, 12)]
    old = touch(tmp_path, date(2026, 5, 1))
    trace.configure(tmp_path, max_days=14, clock=lambda: now[0])
    assert not old.exists()
    # 隔天第一次寫入時再清一次
    stale = touch(tmp_path, date(2026, 6, 5))
    now[0] = datetime(2026, 6, 21, 0, 5)

    async def go():
        async with trace.start("reply"):
            pass

    run(go())
    assert not stale.exists() and (tmp_path / "ai-2026-06-21.jsonl").exists()


# ---------- llm.chat_tools 的每一輪 ----------
class WireLLM(LMStudioClient):
    """跳過網路：依序回傳預先寫好的 message（蓋掉 _send，所以 _request 的記錄邏輯是真的）。"""

    def __init__(self, replies):
        super().__init__("http://x", "m")
        self.replies = list(replies)

    async def _send(self, payload, reasoning_effort):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def call(name, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_chat_tools_rounds_are_traced(tmp_path):
    trace.configure(tmp_path)
    llm = WireLLM([{"content": "", "tool_calls": [call("web_search", {"query": "冠軍"})]},
                   {"content": "BLG"}])

    async def go():
        async with trace.start("reply") as tr:
            async def tool(name, args):
                return "BLG 3-1"
            out = await llm.chat_tools([{"role": "user", "content": "誰冠軍"}], [{"t": 1}], tool)
            tr.final = out

    run(go())
    (rec,) = lines(tmp_path)
    a1, a2 = rec["attempts"]
    assert a1["messages"] == [{"role": "user", "content": "誰冠軍"}]
    assert a1["response"]["tool_calls"][0]["function"]["name"] == "web_search"
    assert a1["tools"] == [{"name": "web_search", "args": {"query": "冠軍"}, "result": "BLG 3-1"}]
    assert a2["response"]["content"] == "BLG" and a2["tools"] == []
    assert a2["messages"][-1]["role"] == "tool"  # 第二輪看得到工具結果
    assert rec["degraded"] == [] and rec["final"] == "BLG"


def test_tool_budget_marks_degraded(tmp_path):
    trace.configure(tmp_path)
    looping = {"content": "", "tool_calls": [call("web_search", {"query": "x"})]}
    wants_more = {"content": "<tool_call><function=web_search><parameter=query>x</parameter></function></tool_call>"}
    llm = WireLLM([looping, wants_more, {"content": "只能憑資料回答"}])

    async def go():
        async with trace.start("reply"):
            async def tool(name, args):
                return "r"
            return await llm.chat_tools([{"role": "user", "content": "q"}], [{"t": 1}], tool, max_rounds=1)

    assert run(go()) == "只能憑資料回答"
    (rec,) = lines(tmp_path)
    assert rec["degraded"] == ["tool_budget"] and len(rec["attempts"]) == 3


def test_llm_failure_is_recorded_as_error_attempt(tmp_path):
    trace.configure(tmp_path)
    llm = WireLLM([LLMError("LM Studio 連線失敗")])

    async def go():
        async with trace.start("reply"):
            await llm.chat([{"role": "user", "content": "嗨"}])

    with pytest.raises(LLMError):
        run(go())
    (rec,) = lines(tmp_path)
    assert rec["outcome"] == "error" and "LLMError" in rec["attempts"][0]["error"]
    assert rec["attempts"][0]["response"] is None


def test_no_trace_means_no_recording():
    llm = WireLLM([{"content": "好"}])
    assert run(llm.chat([{"role": "user", "content": "嗨"}])) == "好"  # 沒有追蹤時照常運作


def test_resolve_model_connection_error_becomes_llmerror():
    llm = LMStudioClient("http://127.0.0.1:9", "")  # 埠 9 沒有服務

    async def go():
        try:
            await llm.resolve_model()
        finally:
            await llm.aclose()

    with pytest.raises(LLMError):
        run(go())
