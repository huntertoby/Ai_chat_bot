"""AI 追蹤紀錄（開發規格 6.12）：每次動用 AI 在本機留一行 JSONL，事後查得到她為什麼這樣回、卡在哪。

用法：
    async with trace.start("reply", guild=gid, channel=cid, trigger="called", user_ids=[uid]) as tr:
        ...
        tr.final = reply
`start` 會把追蹤放進 contextvar，`llm.py` 不用改函式參數就能記錄每一輪 LLM。
寫檔在 `__aexit__`（出錯、逾時、被取消也會執行）；寫檔失敗只寫 warning，不影響回覆。
沒有呼叫過 `configure()`（例如獨立腳本）時，追蹤照常運作但不寫檔。
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from . import vision

log = logging.getLogger(__name__)

KINDS = ("reply", "voice", "chime", "profile")  # chime 留給 B2（插話判斷），目前沒有接點
_FILE_RE = re.compile(r"^ai-(\d{4}-\d{2}-\d{2})\.jsonl$")

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("yuki_trace", default=None)


@dataclass
class _Config:
    log_dir: Path
    max_days: int = 14
    max_mb: int = 200
    # (guild_id, user_ids) -> 其中有人退出紀錄就回 True，整筆不寫
    opted_out: Callable[[str, list[str]], bool] | None = None
    clock: Callable[[], datetime] = datetime.now
    last_prune: date | None = None


_cfg: _Config | None = None


def configure(log_dir: Path, *, max_days: int = 14, max_mb: int = 200,
              opted_out: Callable[[str, list[str]], bool] | None = None,
              clock: Callable[[], datetime] = datetime.now) -> None:
    """啟動時呼叫一次；同時清一次舊檔。"""
    global _cfg
    _cfg = _Config(Path(log_dir), max_days, max_mb, opted_out, clock)
    _maybe_prune()


def reset() -> None:
    """測試用：取消設定。"""
    global _cfg
    _cfg = None


def current() -> "Trace | None":
    return _current.get()


def current_id() -> str:
    t = _current.get()
    return t.trace_id if t else ""


def start(kind: str, *, guild: str = "", channel: str = "", trigger: str = "",
          user_ids: Iterable[str] = ()) -> "Trace":
    return Trace(kind, guild=str(guild), channel=str(channel), trigger=trigger,
                 user_ids=[str(u) for u in user_ids])


def _jsonable(o: Any) -> Any:
    return str(o)


class Trace:
    def __init__(self, kind: str, *, guild: str = "", channel: str = "", trigger: str = "",
                 user_ids: list[str] | None = None):
        self.trace_id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.guild = guild
        self.channel = channel
        self.trigger = trigger
        self.user_ids = user_ids or []
        self.outcome = "ok"  # ok / skipped / error
        self.degraded: list[str] = []
        self.latency_ms: dict[str, int] = {}
        self.attempts: list[dict] = []
        self.images: list[dict] = []  # 附了哪些圖：只記來源訊息 ID，不記內容
        self.final: str | None = None
        self.error: str | None = None
        self._t0 = time.monotonic()
        self._ts = datetime.now()
        self._token: contextvars.Token | None = None

    # ---- 記錄 ----
    def add_attempt(self, messages: list, response: Any, ms: float, error: str | None = None) -> None:
        self.attempts.append({"messages": vision.redact(messages), "response": response, "ms": round(ms),
                              "tools": [], **({"error": error} if error else {})})

    def add_tool(self, name: str, args: Any, result: str) -> None:
        if not self.attempts:
            self.attempts.append({"messages": [], "response": None, "ms": 0, "tools": []})
        self.attempts[-1]["tools"].append({"name": name, "args": args, "result": result})

    def degrade(self, tag: str) -> None:
        if tag not in self.degraded:
            self.degraded.append(tag)

    def lat(self, key: str, ms: float) -> None:
        self.latency_ms[key] = round(ms)

    # ---- 結束 ----
    def to_record(self) -> dict:
        lat = dict(self.latency_ms)
        lat.setdefault("llm", sum(a["ms"] for a in self.attempts))
        lat["total"] = round((time.monotonic() - self._t0) * 1000)
        rec = {"trace_id": self.trace_id, "ts": self._ts.isoformat(timespec="seconds"), "kind": self.kind,
                "guild": self.guild, "channel": self.channel, "trigger": self.trigger,
                "outcome": self.outcome, "degraded": self.degraded, "latency_ms": lat,
                "attempts": self.attempts, "final": self.final, "error": self.error}
        if self.images:  # 有附圖才多這兩欄：張數和來源訊息 ID（不記圖片內容）
            rec["images"] = len(self.images)
            rec["image_sources"] = self.images
        return rec

    async def __aenter__(self) -> "Trace":
        self._token = _current.set(self)
        log.info("trace_id=%s kind=%s 開始", self.trace_id, self.kind)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.outcome = "error"
            self.error = f"{type(exc).__name__}: {exc}"[:500]
        if self._token is not None:
            try:
                _current.reset(self._token)
            except ValueError:  # 在別的 context 結束（被取消的 task）：照樣寫檔
                _current.set(None)
        try:
            self._write()
        except Exception as e:  # 寫檔失敗不能讓回覆失敗
            log.warning("trace_id=%s 追蹤紀錄寫入失敗：%r", self.trace_id, e)
        log.info("trace_id=%s kind=%s outcome=%s", self.trace_id, self.kind, self.outcome)
        return False

    def _write(self) -> None:
        cfg = _cfg
        if cfg is None:
            return
        if cfg.opted_out is not None and self.user_ids and cfg.opted_out(self.guild, self.user_ids):
            log.info("trace_id=%s 有人退出紀錄，整筆不寫", self.trace_id)
            return
        _maybe_prune()
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.to_record(), ensure_ascii=False, default=_jsonable)
        path = cfg.log_dir / f"ai-{cfg.clock().date().isoformat()}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------- 保留 ----------
def prune(log_dir: Path, *, max_days: int, max_mb: int, today: date) -> list[str]:
    """刪掉超過 max_days 天的檔；剩下的總量超過 max_mb 就從最舊的刪起（今天的檔不刪）。回傳被刪的檔名。"""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    files: list[tuple[date, Path]] = []
    for p in log_dir.iterdir():
        m = _FILE_RE.match(p.name)
        if m:
            try:
                files.append((date.fromisoformat(m.group(1)), p))
            except ValueError:
                continue
    files.sort(key=lambda x: x[0])
    removed: list[str] = []

    def drop(p: Path) -> None:
        try:
            p.unlink()
            removed.append(p.name)
        except OSError as e:
            log.warning("刪除舊追蹤紀錄失敗 %s：%r", p.name, e)

    keep = []
    for d, p in files:
        if (today - d).days > max_days:
            drop(p)
        else:
            keep.append((d, p))
    limit = max_mb * 1024 * 1024
    total = sum(p.stat().st_size for _, p in keep)
    for d, p in keep:
        if total <= limit:
            break
        if d >= today:
            continue
        size = p.stat().st_size
        drop(p)
        total -= size
    if removed:
        log.info("清掉舊追蹤紀錄：%s", removed)
    return removed


def _maybe_prune() -> None:
    cfg = _cfg
    if cfg is None:
        return
    today = cfg.clock().date()
    if cfg.last_prune == today:
        return
    cfg.last_prune = today
    try:
        prune(cfg.log_dir, max_days=cfg.max_days, max_mb=cfg.max_mb, today=today)
    except Exception as e:
        log.warning("清理追蹤紀錄失敗：%r", e)
