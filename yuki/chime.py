"""B2 插話判斷（開發規格 6.1）：沒被叫到的訊息，Yuki 要不要開口、按表情、還是安靜。

不碰 Discord：`decide()` 是純函式介面（給 bot.py 和評估腳本共用）；
`ChimeLimits` 是程式端的硬限制（頻道次數、同人同類一天一次、被叫閉嘴後安靜）；
`fill_replies()` 補抓回覆關係。
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Awaitable, Callable, Iterable

from . import prompts
from .llm import LLMError
from .memory.writer import REPLY_SNIPPET, ChatLine, ReplyRef

log = logging.getLogger(__name__)

KINDS = ("congrats", "comfort", "answer", "addon", "about_me", "other")
ACTIONS = ("reply", "react", "none")
CONTEXT_LINES = 15

# CHIME_IN 等級 → (門檻, 同頻道主動插話的最短間隔秒數)；0 = 只有被叫才回
LEVELS: dict[int, tuple[float, float]] = {1: (0.7, 600.0), 2: (0.6, 300.0), 3: (0.5, 120.0)}
# 被叫到的情況，這些類別不受「同人同類一天一次」限制（對話延續，不是主動搭話）
KIND_EXEMPT_IN_ATTENTION = ("about_me", "answer")

# 對她說「沒問你」「閉嘴」這類話
_SHUSH = re.compile(r"沒問你|沒在問你|沒有問你|閉嘴|乾你屁事|干你屁事|關你屁事|管你屁事|不關你的事|不干你的事|你很吵|吵死了")


def is_shush(text: str) -> bool:
    return bool(_SHUSH.search(text or ""))


# ---------- 判斷結果 ----------
@dataclass
class Decision:
    participate: float = 0.0
    action: str = "none"           # reply / react / none
    emoji: str = ""
    kind: str = "other"
    reason: str = ""
    need_background: list[str] = field(default_factory=list)
    ms: int = 0                    # 這次判斷花的時間（毫秒）
    error: str = ""                # 有值 = 判斷失敗（LLM 錯誤或解析失敗），一律當作不插話
    raw: dict | None = None        # 模型回的原始 JSON

    def should_reply(self, threshold: float) -> bool:
        return not self.error and self.action == "reply" and self.participate >= threshold

    def should_react(self, threshold: float) -> bool:
        return not self.error and self.action == "react" and bool(self.emoji) and self.participate >= threshold

    def to_dict(self) -> dict:
        return {"participate": self.participate, "action": self.action, "emoji": self.emoji, "kind": self.kind,
                "reason": self.reason, "need_background": self.need_background, "ms": self.ms,
                **({"error": self.error} if self.error else {})}


def parse_decision(raw: dict | None) -> Decision:
    """把模型回的 JSON 整理成 Decision：欄位缺漏、型別不對、不在選項裡的值都改成最安靜的預設。"""
    if not isinstance(raw, dict):
        return Decision(error="no_json")
    try:
        p = float(raw.get("participate", 0))
    except (TypeError, ValueError):
        p = 0.0
    action = raw.get("action") if raw.get("action") in ACTIONS else "none"
    kind = raw.get("kind") if raw.get("kind") in KINDS else "other"
    bg = raw.get("need_background")
    need = [str(x).strip()[:60] for x in bg if str(x).strip()][:3] if isinstance(bg, list) else []
    return Decision(participate=min(1.0, max(0.0, p)), action=action, emoji=str(raw.get("emoji") or "").strip(),
                    kind=kind, reason=str(raw.get("reason") or "").strip(), need_background=need, raw=raw)


def build_messages(recent: list[ChatLine], latest: ChatLine, bot_name: str, images: Iterable[str] = (),
                   attention: bool = False, context_lines: int = CONTEXT_LINES) -> list[dict]:
    """[你是誰] 放 system；[最近的對話]、[最新訊息]（和最多幾張縮小的圖）放 user。"""
    shown = [_line_text(l, bot_name) for l in recent[-context_lines:]]
    images = list(images)
    text = prompts.chime_user(shown, _line_text(latest, bot_name), attention=attention, images=len(images))
    content: Any = text
    if images:
        content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in images]
    return [{"role": "system", "content": prompts.CHIME_SYSTEM.format(bot_name=bot_name)},
            {"role": "user", "content": content}]


def _line_text(line: ChatLine, bot_name: str) -> str:
    if line.is_bot:
        return f"[{bot_name}（你）] {line.text}"
    return line.render_chat()


async def decide(llm, *, recent: list[ChatLine], latest: ChatLine, bot_name: str, images: Iterable[str] = (),
                 attention: bool = False, context_lines: int = CONTEXT_LINES, max_tokens: int = 160) -> Decision:
    """插話判斷（一次短的 LLM 呼叫，JSON schema、不思考）。

    recent：最新那則「之前」的對話（舊到新，內部只取最後 context_lines 則）；latest：要判斷的那則；
    images：最近幾則裡的圖（data URL，低解析度，最多幾張由呼叫端決定）；attention：她剛講過話，提示模型這則比較可能是在接她的話。
    LLM 連不上或回不出 JSON 時回傳帶 error 的 Decision（action=none），不丟例外。
    門檻和次數限制不在這裡：呼叫端用 `Decision.should_reply(threshold)` 和 `ChimeLimits` 決定。"""
    messages = build_messages(recent, latest, bot_name, images, attention, context_lines)
    t0 = time.monotonic()
    try:
        raw = await llm.chat_json(messages, prompts.CHIME_SCHEMA, name="chime", temperature=0.0,
                                  max_tokens=max_tokens, reasoning_effort="none")
    except LLMError as e:
        d = Decision(error=f"llm: {e}"[:200])
    else:
        d = parse_decision(raw)
    d.ms = round((time.monotonic() - t0) * 1000)
    return d


def llm_is_busy(llm) -> bool:
    """LM Studio 一次只處理一個請求：`_slots` 被佔滿就是忙。"""
    slots = getattr(llm, "_slots", None)
    return bool(slots is not None and slots.locked())


# ---------- 硬限制 ----------
class ChimeLimits:
    """程式端的限制，不靠模型：等級（門檻、間隔）、同人同類一天一次、閉嘴後安靜、按表情另外計。
    時間用傳入的 clock（秒）和 today（日期），方便測試。"""

    def __init__(self, level: int = 1, *, threshold: float | None = None, mute_sec: float = 1800.0,
                 react_max: int = 3, react_window_sec: float = 600.0,
                 clock: Callable[[], float] = time.monotonic, today: Callable[[], date] = date.today):
        self.level = level
        self._threshold = threshold
        self.mute_sec = mute_sec
        self.react_max = react_max
        self.react_window_sec = react_window_sec
        self._clock = clock
        self._today = today
        self._last_chime: dict[Any, float] = {}
        self._reacts: dict[Any, deque[float]] = defaultdict(deque)
        self._muted_until: dict[Any, float] = {}
        self._kind_day: dict[tuple[Any, str], date] = {}

    @property
    def enabled(self) -> bool:
        return self.level in LEVELS

    @property
    def threshold(self) -> float:
        base = LEVELS.get(self.level, (1.0, 0.0))[0]
        return self._threshold if self._threshold is not None else base

    @property
    def interval_sec(self) -> float:
        return LEVELS.get(self.level, (1.0, 0.0))[1]

    # ---- 閉嘴 ----
    def mute(self, channel: Any) -> None:
        self._muted_until[channel] = self._clock() + self.mute_sec

    def is_muted(self, channel: Any) -> bool:
        return self._muted_until.get(channel, 0) > self._clock()

    # ---- 開口前的檢查：回傳不能插話的原因，None = 可以 ----
    def gate(self, channel: Any, *, attention: bool = False) -> str | None:
        """頻道層級的限制。注意模式（她剛講過話）不受次數間隔限制，但閉嘴仍然有效。"""
        if not self.enabled:
            return "chime_off"
        if self.is_muted(channel):
            return "muted"
        if not attention:
            last = self._last_chime.get(channel)
            if last is not None and self._clock() - last < self.interval_sec:
                return "channel_limit"
        return None

    def react_gate(self, channel: Any) -> str | None:
        if not self.enabled:
            return "chime_off"
        if self.is_muted(channel):
            return "muted"
        q = self._reacts[channel]
        now = self._clock()
        while q and now - q[0] > self.react_window_sec:
            q.popleft()
        if len(q) >= self.react_max:
            return "react_limit"
        return None

    def kind_ok(self, user: Any, kind: str, *, attention: bool = False) -> bool:
        """同一個人、同一類事一天最多一次（注意模式下的 about_me／answer 是對話延續，不算）。"""
        if attention and kind in KIND_EXEMPT_IN_ATTENTION:
            return True
        return self._kind_day.get((user, kind)) != self._today()

    # ---- 記錄實際做了什麼 ----
    def record_reply(self, channel: Any, user: Any, kind: str, *, attention: bool = False) -> None:
        if not attention:  # 注意模式的回覆不佔用「每 10 分鐘一次」的名額
            self._last_chime[channel] = self._clock()
        if not (attention and kind in KIND_EXEMPT_IN_ATTENTION):
            self._kind_day[(user, kind)] = self._today()

    def record_react(self, channel: Any, user: Any, kind: str) -> None:
        self._reacts[channel].append(self._clock())
        self._kind_day[(user, kind)] = self._today()


# ---------- 按表情 ----------
def resolve_emoji(emoji: str, guild_emojis: Iterable[Any] = ()) -> Any | None:
    """模型給的表情 → 可以拿去 add_reaction 的東西：unicode 表情原樣回傳；
    `:名稱:` 或名稱在群組表情裡找得到就回那個表情物件；找不到回 None（不按）。"""
    e = (emoji or "").strip()
    if not e:
        return None
    m = re.fullmatch(r"<a?:(\w+):\d+>", e)
    name = m.group(1) if m else e.strip(":")
    for ge in guild_emojis:
        if getattr(ge, "name", "").lower() == name.lower() and getattr(ge, "available", True):
            return ge
    if re.search(r"[A-Za-z0-9_]", e) or len(e) > 8:
        return None  # 是名稱不是表情符號，群組裡又沒有這張
    return e


# ---------- 回覆關係 ----------
def make_ref(name: str, is_bot: bool, text: str) -> ReplyRef:
    return ReplyRef(name=name, is_bot=is_bot, snippet=" ".join((text or "").split())[:REPLY_SNIPPET])


async def fill_replies(lines: list[ChatLine], fetch: Callable[[str], Awaitable[ReplyRef | None]], *,
                       limit: int = 3) -> int:
    """對話紀錄裡有 reply_id 卻還沒有 reply_to 的訊息：先從對話紀錄裡找被回覆的那則（不花 API），
    找不到才用 `fetch(message_id)` 補抓，每次最多 `limit` 則（新的優先）；失敗就略過，並清掉 reply_id 不再重試。
    回傳補抓（API）的次數。"""
    by_id = {l.msg_id: l for l in lines if l.msg_id}
    fetched = 0
    for line in reversed(lines):
        if line.reply_to is not None or not line.reply_id:
            continue
        src = by_id.get(line.reply_id)
        if src is not None:
            line.reply_to = make_ref(src.author.name, src.is_bot, src.text)
            continue
        if fetched >= limit:
            continue
        fetched += 1
        try:
            ref = await fetch(line.reply_id)
        except Exception as e:  # 訊息被刪、沒權限、網路問題都只是少一個回覆標記
            log.info("補抓回覆對象失敗：%r", e)
            ref = None
        if ref is not None:
            line.reply_to = ref
        else:
            line.reply_id = ""
    return fetched
