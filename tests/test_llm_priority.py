"""LM Studio 排隊聊天優先（審查一-5）：背景的人物檔案整理不能把聊天回覆卡住。用假的 HTTP 伺服器，不需要 LM Studio。"""
import asyncio
import json
import time

import httpx
import pytest

from yuki import chime
from yuki.llm import PREEMPTED, LLMError, LLMSlots, LMStudioClient


def fake_server(log, slow=30.0):
    """system 內容是 bg 的請求要跑很久（模擬整理一段人物檔案），其他立刻回。"""
    async def handler(request: httpx.Request) -> httpx.Response:
        who = json.loads(request.content)["messages"][0]["content"]
        log.append(("start", who))
        if who.startswith("bg"):
            await asyncio.sleep(slow)
        log.append(("done", who))
        return httpx.Response(200, json={"choices": [{"message": {"content": f"ok {who}"}, "finish_reason": "stop"}]})
    return httpx.MockTransport(handler)


def clients(log, **kw):
    transport = fake_server(log, **kw)
    chat = LMStudioClient("http://lm", "m", reasoning_effort="")
    bg = LMStudioClient("http://lm", "m", reasoning_effort="", slots=chat._slots, background=True)
    for c in (chat, bg):
        c._http = httpx.AsyncClient(transport=transport)
    return chat, bg


def msgs(who):
    return [{"role": "system", "content": who}]


def test_chat_does_not_wait_for_a_long_background_request():
    log = []

    async def go():
        chat, bg = clients(log)
        bg_task = asyncio.create_task(bg.chat(msgs("bg-1")))
        await asyncio.sleep(0.05)  # 背景請求已經送出去、正在跑
        t0 = time.monotonic()
        reply = await chat.chat(msgs("chat-1"))
        waited = time.monotonic() - t0
        with pytest.raises(LLMError, match=PREEMPTED):
            await bg_task
        return reply, waited

    reply, waited = asyncio.run(go())
    assert reply == "ok chat-1" and waited < 2
    assert ("done", "bg-1") not in log  # 背景那個被中斷了，不是跑完才換聊天


def test_background_waits_while_chat_is_queued_and_chat_jumps_ahead():
    log = []

    async def go():
        chat, bg = clients(log, slow=0.0)
        slots = chat._slots
        first = await slots.acquire()            # 有一則聊天正在跑
        bg_task = asyncio.create_task(bg.chat(msgs("bg-late")))
        await asyncio.sleep(0.01)
        chat_task = asyncio.create_task(chat.chat(msgs("chat-2")))  # 背景先來，但聊天後到也先送
        await asyncio.sleep(0.01)
        assert log == []                          # 位子被佔著：誰都還沒送
        slots.release(first)
        await asyncio.gather(chat_task, bg_task)

    asyncio.run(go())
    assert [w for e, w in log if e == "start"] == ["chat-2", "bg-late"]


def test_background_holding_the_slot_does_not_make_chime_think_llm_is_busy():
    """插話判斷看到「LM Studio 忙」就跳過；被背景佔著不算忙，因為聊天一來背景就讓位。"""
    async def go():
        slots = LLMSlots(1)
        llm = type("L", (), {"_slots": slots})()
        h = await slots.acquire(background=True)
        assert not chime.llm_is_busy(llm)
        slots.release(h)
        c = await slots.acquire()
        assert chime.llm_is_busy(llm)
        slots.release(c)
        assert not chime.llm_is_busy(llm)

    asyncio.run(go())


def test_cancelled_chat_waiter_does_not_leak_the_slot():
    async def go():
        slots = LLMSlots(1)
        h = await slots.acquire()
        waiter = asyncio.create_task(slots.acquire())
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        slots.release(h)
        again = await asyncio.wait_for(slots.acquire(), 1)
        slots.release(again)
        return slots._free

    assert asyncio.run(go()) == 1
