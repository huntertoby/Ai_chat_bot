"""LM Studio 的 async client。全部用 httpx.AsyncClient，不會卡住 Discord 的 event loop。"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

import httpx

from . import trace

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    # 只有結尾標籤（開頭被 chat template 吃掉）時，取最後一個 </think> 之後的內容
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").strip()


_TEXT_CALL = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)
_FUNC = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", re.DOTALL)
_PARAM = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)


def text_tool_calls(text: str) -> list[dict]:
    """模型有時把工具呼叫用文字寫出來（Qwen 的 XML 格式），伺服器沒解析成 tool_calls。
    解析成跟 API 一樣的格式；參數值能當 JSON 就當 JSON（陣列、數字）。"""
    calls = []
    for block in _TEXT_CALL.findall(text or ""):
        for name, body in _FUNC.findall(block):
            args = {}
            for k, v in _PARAM.findall(body):
                try:
                    args[k] = json.loads(v)
                except json.JSONDecodeError:
                    args[k] = v
            calls.append({"id": f"text-{len(calls)}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


def strip_text_calls(text: str) -> str:
    return _TEXT_CALL.sub("", text or "").strip()


def extract_json(text: str) -> dict | None:
    """從模型輸出中取出最後一個 JSON 物件；失敗回傳 None（永遠不會丟例外）。"""
    text = strip_think(text)
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    found: dict | None = None
    i = 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if isinstance(obj, dict):
                found = obj
            i = end
        else:
            i += 1
    return found


class LLMError(RuntimeError):
    pass


PREEMPTED = "背景請求讓位給聊天"


class _Hold:
    def __init__(self, background: bool):
        self.background = background
        self.preempted = False
        self.task: asyncio.Future | None = None

    def preempt(self) -> None:
        self.preempted = True
        if self.task is not None and not self.task.done():
            self.task.cancel()  # 關掉連線，LM Studio 那邊就停下這個請求


class LLMSlots:
    """送給 LM Studio 的排隊，**聊天優先**（審查一-5）：
    - 聊天（回覆、插話判斷）先排；背景工作（人物檔案整理）只在沒有任何聊天在等、而且有空位時才送。
    - 聊天要等的時候，正在跑的背景請求直接中斷讓位（背景那邊拿到 LLMError(PREEMPTED)，下次閒置再做），
      所以聊天不會被一個要跑好幾分鐘的背景請求卡住。
    - 跟 asyncio.Semaphore 一樣可以 `async with slots:`（算聊天）；`locked()` 只看聊天：
      被背景佔著不算忙，因為聊天一來背景就讓位。"""

    def __init__(self, n: int = 1):
        self._free = max(1, n)
        self._chat_waiters: deque[asyncio.Future] = deque()
        self._bg_waiters: deque[asyncio.Future] = deque()
        self._holds: set[_Hold] = set()

    def locked(self) -> bool:
        chat_holds = any(not h.background for h in self._holds)
        return self.chat_waiting() or (self._free <= 0 and chat_holds)

    def chat_waiting(self) -> bool:
        return any(not f.done() for f in self._chat_waiters)

    def _grant(self, background: bool) -> _Hold:
        self._free -= 1
        h = _Hold(background)
        self._holds.add(h)
        return h

    def _wake(self) -> None:
        while self._free > 0:
            q = self._chat_waiters if self._chat_waiters else self._bg_waiters
            if not q:
                return
            fut = q.popleft()
            if fut.done():
                continue
            fut.set_result(self._grant(q is self._bg_waiters))

    async def acquire(self, background: bool = False) -> _Hold:
        if self._free > 0 and not self._chat_waiters and (not background or not self._bg_waiters):
            return self._grant(background)
        fut = asyncio.get_running_loop().create_future()
        (self._bg_waiters if background else self._chat_waiters).append(fut)
        if not background:
            for h in list(self._holds):
                if h.background:
                    h.preempt()
        try:
            return await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release(fut.result())  # 剛好拿到位子又被取消：還回去
            raise

    def release(self, hold: _Hold) -> None:
        if hold in self._holds:
            self._holds.discard(hold)
            self._free += 1
        self._wake()

    @asynccontextmanager
    async def hold(self, background: bool = False):
        h = await self.acquire(background)
        try:
            yield h
        finally:
            self.release(h)

    async def __aenter__(self) -> "LLMSlots":
        self._ctx = self.hold()
        await self._ctx.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._ctx.__aexit__(*exc)


class LMStudioClient:
    def __init__(self, base_url: str, model: str = "", *, temperature: float = 0.7,
                 max_tokens: int = 2048, timeout: float = 180, reasoning_effort: str = "",
                 json_reasoning_effort: str = "", max_concurrency: int = 1,
                 extra_body: dict[str, Any] | None = None, slots: LLMSlots | None = None,
                 background: bool = False):
        """slots：跟另一個 client 共用排隊（例如人物檔案更新跟聊天共用），避免同時送兩個請求給 LM Studio。
        background：這個 client 的請求是背景工作，排在聊天後面，聊天要用時會被中斷（LLMError(PREEMPTED)）。"""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.json_reasoning_effort = json_reasoning_effort
        self.extra_body = extra_body or {}  # 每次請求都附加的欄位（例如 vLLM 的 chat_template_kwargs）
        self._slots = slots or LLMSlots(max_concurrency)
        self.background = background
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def resolve_model(self) -> str:
        if self.model:
            return self.model
        try:
            resp = await self._http.get(f"{self.base_url}/models")
            resp.raise_for_status()
            ids = [m["id"] for m in resp.json().get("data", [])]
        except httpx.TimeoutException as e:
            raise LLMError(f"LM Studio 逾時：{e!r}") from e
        except httpx.HTTPError as e:  # 連不上也走 LLMError，才會跟其他 LM Studio 錯誤一樣被通知
            raise LLMError(f"LM Studio 連線失敗：{e!r}") from e
        chat_ids = [i for i in ids if "embed" not in i.lower() and "rerank" not in i.lower()]
        if not chat_ids:
            raise LLMError(f"LM Studio 沒有可用的聊天模型（看到的模型：{ids}）")
        self.model = chat_ids[0]
        log.info("自動選用 LM Studio 模型：%s", self.model)
        return self.model

    async def _request(self, payload: dict[str, Any], reasoning_effort: str) -> dict[str, Any]:
        """送出 chat/completions，回傳 choices[0].message（含 content、tool_calls）。
        有「目前的追蹤」（yuki.trace）時，把這一輪的 messages、回應、耗時記進去。"""
        tr = trace.current()
        if tr is None:
            return await self._send(payload, reasoning_effort)
        t0 = time.monotonic()
        try:
            message = await self._send(payload, reasoning_effort)
        except BaseException as e:
            tr.add_attempt(payload.get("messages", []), None, (time.monotonic() - t0) * 1000,
                           error=f"{type(e).__name__}: {e}"[:300])
            raise
        tr.add_attempt(payload.get("messages", []), message, (time.monotonic() - t0) * 1000)
        return message

    async def _send(self, payload: dict[str, Any], reasoning_effort: str) -> dict[str, Any]:
        payload = {"model": await self.resolve_model(), "stream": False, **self.extra_body, **payload}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if reasoning_effort == "none":
            # vLLM 不看 reasoning_effort，Qwen3 要用 chat template 參數關掉思考；LM Studio 會忽略這個欄位
            kwargs = dict(payload.get("chat_template_kwargs") or {})
            kwargs.setdefault("enable_thinking", False)
            payload["chat_template_kwargs"] = kwargs
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                async with self._slots.hold(background=self.background) as slot:
                    if slot.preempted:
                        raise LLMError(PREEMPTED)
                    req = asyncio.ensure_future(self._http.post(f"{self.base_url}/chat/completions", json=payload))
                    slot.task = req
                    try:
                        await asyncio.wait({req})
                    except asyncio.CancelledError:
                        req.cancel()
                        raise
                    if req.cancelled():
                        raise LLMError(PREEMPTED)
                    resp = req.result()
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                message = choice.get("message") or {}
                if not message.get("content") and not message.get("tool_calls") \
                        and choice.get("finish_reason") == "length":
                    log.warning("模型把 token 用完在推理上，沒有產生回答；可調低 LLM_REASONING_EFFORT")
                return message
            except httpx.TimeoutException as e:
                # 逾時代表模型還在跑，重試只會讓使用者再等一輪
                raise LLMError(f"LM Studio 逾時：{e!r}") from e
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last_err = e
                detail = e.response.text[:300] if isinstance(e, httpx.HTTPStatusError) else ""
                log.warning("LM Studio 呼叫失敗（第 %d 次）：%r %s", attempt + 1, e, detail)
        raise LLMError(f"LM Studio 連線失敗：{last_err!r}")

    async def _complete(self, payload: dict[str, Any], reasoning_effort: str) -> str:
        return (await self._request(payload, reasoning_effort)).get("content") or ""

    async def _answer_from_results(self, msgs: list[dict], payload: dict[str, Any], text: str,
                                   *, plain_retry: bool = True, n_base: int = 0) -> str:
        """模型回空字串時補救（推理模型有時把答案寫在思考裡、正文是空的）：
        1. 不給工具、不思考再問一次；2. 還是空的，加一句「依照查到的內容回答，查不到就直說」再問；仍空才放棄。"""
        no_tools = {k: v for k, v in payload.items() if k != "tools"}
        if plain_retry and not text and self.reasoning_effort not in ("", "none"):
            retry = await self._request({**no_tools, "messages": msgs}, "none")
            text = strip_text_calls(strip_think(retry.get("content") or ""))
        if not text:
            nudge = msgs + [{"role": "user", "content": "（系統：不能再查了，依照上面查到的內容回答；"
                                                      "查不到就直接說查不到、沒印象。）"}]
            retry = await self._request({**no_tools, "messages": nudge}, "none")
            text = strip_text_calls(strip_think(retry.get("content") or ""))
        if not text:
            # 模型被工具呼叫的格式帶著走，一直想再查（輸出 <tool_call> 被濾掉就變空字串）：
            # 把查到的結果攤平成一則普通訊息，上下文裡就沒有工具呼叫可以學了
            found = [m.get("content") or "" for m in msgs if m.get("role") == "tool"]
            base = msgs[:n_base]
            flat = base + [{"role": "user", "content": (
                "（系統：以下是你剛剛查到的群組聊天紀錄，不能再查了。\n" + "\n---\n".join(found)
                + "\n\n依照這些內容回答上面的問題；紀錄裡找不到問題問的那件事，就直接說查不到、沒印象。）")}]
            retry = await self._request({**no_tools, "messages": flat}, "none")
            text = strip_text_calls(strip_think(retry.get("content") or ""))
            if not text:
                log.warning("查完工具後模型連續回空字串")
        return text

    async def chat_tools(self, messages: list[dict], tools: list[dict],
                         call_tool: Callable[[str, dict], Awaitable[str]], *, max_rounds: int = 3,
                         max_calls: int = 3) -> str:
        """可以用工具的聊天：模型要求呼叫工具 → 執行 → 把結果給它 → 再問一次，最多 max_rounds 輪。
        最後一輪不給工具，逼它用手上的資料回答。"""
        msgs = list(messages)
        tr = trace.current()
        for rnd in range(max_rounds + 1):
            payload: dict[str, Any] = {"messages": msgs, "temperature": self.temperature,
                                       "max_tokens": self.max_tokens}
            if rnd < max_rounds:
                payload["tools"] = tools
            message = await self._request(payload, self.reasoning_effort)
            calls = (message.get("tool_calls") or [])[:max_calls]
            content = strip_think(message.get("content") or "")
            if not calls and "<tool_call>" in content:
                if rnd < max_rounds:
                    calls = text_tool_calls(content)[:max_calls]  # 文字寫的工具呼叫：照樣執行
                    message = {**message, "content": strip_text_calls(content)}
                else:
                    # 工具次數用完了還想查：叫它用手上的資料回答，不然回覆會是一串 <tool_call>
                    log.info("工具次數用完但模型還想查，要求直接回答")
                    if tr:
                        tr.degrade("tool_budget")
                    return await self._answer_from_results(msgs, payload, "", plain_retry=False, n_base=len(messages))
            if not calls:
                text = strip_text_calls(content)
                if not text:
                    log.info("回覆是空的，改用不思考重試")
                    text = await self._answer_from_results(msgs, payload, text, n_base=len(messages))
                return text
            msgs.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                log.info("工具呼叫：%s %s", name, args)
                result = await call_tool(name, args if isinstance(args, dict) else {})
                if tr:
                    tr.add_tool(name, args, result)
                msgs.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name,
                             "content": result})
        if tr:
            tr.degrade("tool_budget")
        return ""

    async def chat(self, messages: list[dict], *, temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        raw = await self._complete({
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }, self.reasoning_effort)
        return strip_think(raw)

    async def chat_json(self, messages: list[dict], schema: dict, *, name: str = "result",
                        temperature: float = 0.2, max_tokens: int | None = None,
                        reasoning_effort: str | None = None) -> dict | None:
        """要求結構化輸出。LM Studio 支援 json_schema（用 grammar 限制輸出），
        但仍以 extract_json 兜底，解析失敗回傳 None。
        reasoning_effort 可覆寫預設值：簡單又限制 token 的任務要用 "none"，
        不然推理模型會把 max_tokens 花在思考上，答案還沒寫就被截斷。"""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        }
        effort = self.json_reasoning_effort if reasoning_effort is None else reasoning_effort
        raw = await self._complete(payload, effort)
        result = extract_json(raw)
        if result is None:
            log.warning("JSON 解析失敗，原始輸出前 200 字：%s", raw[:200])
        return result
