"""B3 防洗版與節流（開發規格 6.2）：純邏輯，不碰 Discord，方便測試。

- 雜訊過濾：指令、沒內容的短反應（`is_noise_text`、`is_command`）
- 大量重複：同字元連續很多次、同一人重複 @（`is_flood`）
- `Throttle`：每頻道同時只處理一則、被叫到的排隊合併、頻道冷卻、tag 洗版冷卻
- `is_repeat`：對話紀錄去重
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

# ---------- 雜訊過濾 ----------
# 沒內容的短反應：去掉表情和標點後剛好是這些（小寫比對）
NOISE_WORDS = frozenset({
    "ok", "okay", "好", "好喔", "好啊", "好的", "嗯", "嗯嗯", "喔", "哦", "喔喔", "啊", "欸", "對", "對啊", "對喔",
    "笑死", "笑死我", "笑爛", "我去", "我靠", "幹", "靠", "靠北", "屌", "讚", "推", "真假", "真的假的", "傻眼",
    "lol", "lmao", "wtf", "gg", "orz", "666", "草", "睡著了", "晚安", "早安", "早", "確實", "沒錯", "也是",
})
_CUSTOM_EMOJI = re.compile(r"<a?:\w+:\d+>|:\w+:")
_MENTION = re.compile(r"<[@#][!&]?\d+>|@\S+")
_FORMAT_CHARS = "️‍​"
_PREFIX = "/!."
_PREFIX_NOT_COMMAND = " \t\n.!/。…~、"


def strip_decor(text: str) -> str:
    """拿掉表情（unicode 和群組自訂）、@ 提及、標點、空白，只留下有內容的字。"""
    text = _MENTION.sub("", _CUSTOM_EMOJI.sub("", text or ""))
    out = []
    for ch in text:
        if ch in _FORMAT_CHARS:
            continue
        if unicodedata.category(ch)[0] in "PSZC":
            continue
        out.append(ch)
    return "".join(out)


def is_noise_text(text: str, max_chars: int = 2) -> bool:
    """沒內容的反應：空的、去掉表情標點後 ≤ max_chars 字、同一個字重複（哈哈哈哈）、或在小清單裡。"""
    core = strip_decor(text).lower()
    if len(core) <= max_chars:
        return True
    if len(set(core)) == 1:
        return True
    return core in NOISE_WORDS


def is_command(text: str) -> bool:
    """斜線或前綴指令：以 / ! . 開頭、後面緊接著字（「...」「!!」不算）。"""
    t = (text or "").lstrip()
    return len(t) >= 2 and t[0] in _PREFIX and t[1] not in _PREFIX_NOT_COMMAND


_MENTION_ID = re.compile(r"<@[!&]?\d+>")


_LAUGH_CHARS = frozenset("哈呵嘻wWhH草")  # 笑聲、草、www：重複是正常說話


def _is_char_flood(content: str, repeat_chars: int) -> bool:
    """同一個字連續 ≥ repeat_chars 次，而且拿掉這些重複段之後幾乎沒剩內容（≤ 2 字）才算洗版。
    標點、符號、數字、笑聲（哈、w）的重複不算；「我考上了！！！！！！！！！！」「10000000000」都不是洗版。"""
    runs = list(re.finditer(r"(\S)\1{%d,}" % (repeat_chars - 1), content))
    if not runs:
        return False
    if all(unicodedata.category(m.group(1))[0] in "PSNZC" or m.group(1) in _LAUGH_CHARS for m in runs):
        return False
    rest = re.sub(r"(\S)\1{%d,}" % (repeat_chars - 1), "", _MENTION_ID.sub("", content))
    return len(strip_decor(rest)) <= 2


def is_flood(content: str, *, repeat_chars: int = 10, repeat_mentions: int = 3) -> bool:
    """大量重複：同一個非空白字元連續 ≥ repeat_chars 次，或同一個人／身分組在同一則裡被 @ ≥ repeat_mentions 次。
    `content` 要是原始內容（含 <@id>），不是 clean_content。"""
    if repeat_chars > 1 and _is_char_flood(content or "", repeat_chars):
        return True
    if repeat_mentions > 1:
        counts = Counter(_MENTION_ID.findall(content or ""))
        if counts and max(counts.values()) >= repeat_mentions:
            return True
    return False


def is_repeat(prev: Any, line: Any) -> bool:
    """對話紀錄去重：跟上一則同人、同內容（有圖的話圖也一樣）才算重複；Yuki 自己的話不去重。"""
    if prev is None or getattr(line, "is_bot", False) or getattr(prev, "is_bot", False):
        return False
    return (prev.author.id == line.author.id and prev.text.strip() == line.text.strip()
            and list(prev.images) == list(line.images))


# ---------- 節流狀態 ----------
@dataclass
class ThrottleConfig:
    cooldown_sec: float = 5.0        # 回完後同頻道多久不主動插話
    tag_limit: int = 3               # 60 秒內叫她超過這麼多次就算洗版
    tag_window_sec: float = 60.0
    tag_cooldown_sec: float = 120.0  # 洗版的人冷卻多久
    queue_max: int = 8               # 每頻道排隊上限（超過丟最舊的）


class Throttle:
    def __init__(self, cfg: ThrottleConfig | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg or ThrottleConfig()
        self._clock = clock
        self._busy: set[Any] = set()
        self._queue: dict[Any, list[Any]] = defaultdict(list)
        self._last_done: dict[Any, float] = {}
        self._tags: dict[Any, deque[float]] = defaultdict(deque)
        self._tag_until: dict[Any, float] = {}

    # ---- 每頻道同時只處理一則 ----
    def try_start(self, channel: Any) -> bool:
        if channel in self._busy:
            return False
        self._busy.add(channel)
        return True

    def finish(self, channel: Any) -> None:
        """回覆結束（不管成功失敗）：解除忙碌、開始冷卻。"""
        self._busy.discard(channel)
        self._last_done[channel] = self._clock()

    def is_busy(self, channel: Any) -> bool:
        return channel in self._busy

    # ---- 被叫到的排隊：等目前這則回完，合成一次回覆 ----
    def enqueue(self, channel: Any, item: Any) -> None:
        q = self._queue[channel]
        q.append(item)
        del q[:-self.cfg.queue_max]

    def drain(self, channel: Any) -> list[Any]:
        return self._queue.pop(channel, [])

    def drop_queued(self, channel: Any, pred: Callable[[Any], bool]) -> int:
        """把排隊中符合條件的丟掉（tag 洗版的人）；回傳丟了幾個。"""
        q = self._queue.get(channel, [])
        keep = [x for x in q if not pred(x)]
        dropped = len(q) - len(keep)
        if dropped:
            self._queue[channel] = keep
        return dropped

    # ---- 頻道冷卻（只擋主動插話）----
    def in_cooldown(self, channel: Any) -> bool:
        last = self._last_done.get(channel)
        return last is not None and self._clock() - last < self.cfg.cooldown_sec

    # ---- tag 洗版 ----
    def note_tag(self, user: Any) -> bool:
        """有人叫她一次。回傳 True = 照常回；False = 這個人正在冷卻、或這一次剛好超過上限（開始冷卻）。"""
        now = self._clock()
        until = self._tag_until.get(user)
        if until is not None:
            if now < until:
                return False
            del self._tag_until[user]
        q = self._tags[user]
        q.append(now)
        while q and now - q[0] > self.cfg.tag_window_sec:
            q.popleft()
        if len(q) > self.cfg.tag_limit:
            self._tag_until[user] = now + self.cfg.tag_cooldown_sec
            q.clear()
            return False
        return True

    def tag_cooling(self, user: Any) -> bool:
        return self._tag_until.get(user, 0) > self._clock()
