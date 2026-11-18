"""出錯私訊擁有者（開發規格 6.12）。

- `Alerter.notify(kind, where, message)`：同指紋（kind + where）在 dedupe 時間內只送一次；每小時有上限；
  還沒連上 Discord 時先排隊，連上後合成一則補發；送出前遮掉 token / key / 長字串。
- `AlertLogHandler`：ERROR 以上的 log 轉成 notify。送私訊過程自己的 log 不會再觸發通知。
- `health_loop`：背景定期打 LM Studio 的 /models，連不上就通知（沒人講話時掛掉也會知道）。
- 模組層級的 `notify()`：不方便拿到 bot 的地方（工具、TTS）用；沒設定 Alerter 時只寫 log。
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import time
from collections import deque
from datetime import datetime
from typing import Awaitable, Callable, Iterable

from . import trace

log = logging.getLogger(__name__)

_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD)", re.IGNORECASE)
_PATTERNS = [
    re.compile(r"[MNO][A-Za-z\d_-]{23,}\.[\w-]{6,}\.[\w-]{20,}"),  # Discord token
    re.compile(r"Bearer\s+[\w.~+/=-]+", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9+/_=-]{32,}"),  # 長串 base64 / hex
]
_sending: contextvars.ContextVar[bool] = contextvars.ContextVar("yuki_alert_sending", default=False)


def secrets_from_env(environ: dict[str, str] | None = None) -> list[str]:
    """環境變數（含 .env 載入的）裡名字像 key/token/secret 的值。"""
    env = os.environ if environ is None else environ
    return [v.strip() for k, v in env.items() if _SECRET_NAME.search(k) and len(v.strip()) >= 6]


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    for s in sorted(set(secrets), key=len, reverse=True):
        if s:
            text = text.replace(s, "[已遮蔽]")
    for pat in _PATTERNS:
        text = pat.sub("[已遮蔽]", text)
    return text


SendFn = Callable[[str, str], Awaitable[None]]  # (user_id, text)


class Alerter:
    def __init__(self, owner_ids: Iterable[str], *, dedupe_min: int = 30, max_per_hour: int = 10,
                 secrets: Iterable[str] | None = None, clock: Callable[[], float] = time.time):
        self.owner_ids = [str(i) for i in owner_ids]
        self.dedupe_s = dedupe_min * 60
        self.max_per_hour = max_per_hour
        self.secrets = list(secrets) if secrets is not None else secrets_from_env()
        self.clock = clock
        self._last: dict[str, float] = {}
        self._sent: deque[float] = deque()
        self._pending: list[str] = []
        self._send: SendFn | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._flushing = False

    # ---- 通知 ----
    def notify(self, kind: str, where: str, message: str, *, exc: BaseException | str | None = None,
               trace_id: str | None = None) -> bool:
        """回傳這則有沒有被接受（沒被去重、沒超過上限、有設擁有者）。可以從任何執行緒呼叫。"""
        if not self.owner_ids:
            log.info("通知（沒有設 OWNER_USER_IDS，只寫 log）：%s %s %s", kind, where, message)
            return False
        now = self.clock()
        fp = f"{kind}|{where}"
        last = self._last.get(fp)
        if last is not None and now - last < self.dedupe_s:
            return False
        while self._sent and now - self._sent[0] >= 3600:
            self._sent.popleft()
        if len(self._sent) >= self.max_per_hour:
            log.info("通知超過每小時上限 %d 則，丟掉：%s %s", self.max_per_hour, kind, where)
            return False
        self._last[fp] = now
        self._sent.append(now)
        self._pending.append(self._format(message, exc, trace_id if trace_id is not None else trace.current_id()))
        self._wake()
        return True

    def _format(self, message: str, exc: BaseException | str | None, trace_id: str) -> str:
        lines = [f"【Yuki 出狀況】{message}", f"時間：{datetime.fromtimestamp(self.clock()):%Y-%m-%d %H:%M:%S}"]
        if trace_id:
            lines.append(f"trace_id：{trace_id}")
        if exc:
            first = (str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}").strip().splitlines()
            if first:
                lines.append(f"錯誤：{first[0][:300]}")
        return redact("\n".join(lines), self.secrets)

    # ---- 連線與補發 ----
    def connect(self, send: SendFn, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Discord 連上後呼叫：之後的通知直接送，之前排隊的一次補發。"""
        self._send = send
        self._loop = loop or asyncio.get_running_loop()
        self._wake()

    def _wake(self) -> None:
        if self._send is None or self._loop is None or not self._pending:
            return
        try:
            self._loop.call_soon_threadsafe(lambda: self._loop.create_task(self.flush()))
        except RuntimeError:  # loop 已經關了
            pass

    async def flush(self) -> None:
        if self._send is None or self._flushing or not self._pending:
            return
        self._flushing = True
        token = _sending.set(True)
        try:
            items, self._pending = self._pending, []
            text = items[0] if len(items) == 1 else f"（累積 {len(items)} 則）\n\n" + "\n\n".join(items)
            for uid in self.owner_ids:
                try:
                    await self._send(uid, text[:1900])
                except Exception as e:  # 送失敗只寫 log，不重試
                    log.warning("通知私訊送不出去（%s）：%r", uid, e)
        finally:
            _sending.reset(token)
            self._flushing = False
        if self._pending:  # 送的時候又有新的
            await self.flush()


class AlertLogHandler(logging.Handler):
    """ERROR 以上的 log → notify。kind = logger 名稱，where = 檔名:行號。"""

    def __init__(self, alerter: Alerter, level: int = logging.ERROR):
        super().__init__(level)
        self.alerter = alerter

    def emit(self, record: logging.LogRecord) -> None:
        # 送私訊期間、或 alert 自己的 log，不再轉通知（避免無窮迴圈）
        if _sending.get() or record.name == __name__ or record.name.startswith(__name__ + "."):
            return
        try:
            exc = record.exc_info[1] if record.exc_info and record.exc_info[1] else None
            where = f"{os.path.basename(record.pathname)}:{record.lineno}"
            self.alerter.notify(record.name, where, f"{record.name} 出錯：{record.getMessage()}"[:300], exc=exc)
        except Exception:
            pass  # 通知本身絕不能讓 logging 出錯


# ---------- 模組層級的 Alerter（工具、TTS 等拿不到 bot 的地方用）----------
_default: Alerter | None = None


def set_default(alerter: Alerter | None) -> None:
    global _default
    _default = alerter


def notify(kind: str, where: str, message: str, *, exc: BaseException | str | None = None) -> bool:
    if _default is None:
        return False
    try:
        return _default.notify(kind, where, message, exc=exc)
    except Exception:
        return False


# ---------- LM Studio 健康檢查 ----------
async def check_lmstudio(base_url: str, *, timeout: float = 5.0, client=None) -> str | None:
    """連得上回 None，連不上回原因。"""
    import httpx
    own = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        resp = await client.get(f"{base_url.rstrip('/')}/models")
        resp.raise_for_status()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    finally:
        if own:
            await client.aclose()


async def health_loop(base_url: str, interval: float, alerter: Alerter | None = None, *,
                      checker=check_lmstudio) -> None:
    """啟動後馬上檢查一次，之後每 interval 秒一次。恢復時只寫一行 info。"""
    alerter = alerter or _default
    down = False
    while True:
        reason = await checker(base_url)
        if reason:
            down = True
            log.warning("LM Studio 健康檢查失敗：%s", reason)
            if alerter:
                alerter.notify("lmstudio", "LM Studio 連線", f"連不上 LM Studio（{base_url}）", exc=reason)
        elif down:
            down = False
            log.info("LM Studio 恢復連線")
        await asyncio.sleep(interval)
