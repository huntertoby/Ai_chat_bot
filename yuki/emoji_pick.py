"""D-37：先回話、再依回話挑表情（開發規格 6.14）。
回話寫好之後，用一次不帶工具的短呼叫，依「她這句回話的語氣＋原訊息」從圖鑑清單挑一張表情或貼圖，或不配。
- 只在有 reviewed 圖鑑、EmojiPacing 允許時才挑（bot.py 判斷）；語音回覆完全不走這裡。
- 名稱不在圖鑑、JSON 壞掉、超時、出錯：一律當不配，回話照送。
- D-38：整則文字回話（含挑表情）以 TEXT_REPLY_BUDGET_SEC 為上限；挑表情只用剩下的時間，剩不到 PICK_MIN_SEC 就不挑。"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

from . import prompts
from .emoji_catalog import Entry, EmojiSet

log = logging.getLogger(__name__)

TEXT_REPLY_BUDGET_SEC = 10.0   # D-38：文字回話（含挑表情）總共最多等多久
PICK_MIN_SEC = 1.0             # 剩下的時間不到這麼多就不挑
PICK_MAX_TOKENS = 40
MSG_CHARS = 300                # 原訊息、回話各最多放幾個字

_TAG = re.compile(r"^[\[［]\s*貼圖\s*[:：]\s*(.+?)\s*[\]］]$")
_FULL = re.compile(r"^<a?:(\w+):\d+>$")


@dataclass
class Pick:
    entry: Entry | None = None   # 挑到的（None＝不配）
    ms: int = 0                  # 挑表情花的毫秒
    reason: str = ""             # 不配的原因：none（模型選不配）/ unknown（名稱不在圖鑑）/ bad_json / timeout / error / no_time
    raw: str = ""                # 模型寫的名稱（追蹤紀錄用）

    def record(self) -> dict:
        out = {"ms": self.ms, "raw": self.raw}
        if self.entry is not None:
            out.update(name=self.entry.name, kind=self.entry.kind)
        else:
            out["reason"] = self.reason
        return out


def catalog_lines(eset: EmojiSet) -> str:
    """跟以前回覆 prompt 一樣的挑法和數量（最常用的表情＋幾張貼圖）。"""
    return eset.prompt_block()


def build_messages(eset: EmojiSet, said: str, reply: str, bot_name: str) -> list[dict]:
    system = prompts.EMOJI_PICK_SYSTEM.format(bot_name=bot_name, items=catalog_lines(eset))
    user = f"對方說：{(said or '').strip()[:MSG_CHARS]}\n你的回話：{(reply or '').strip()[:MSG_CHARS]}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def resolve(eset: EmojiSet, name: str) -> Entry | None:
    """模型寫的名稱 → 圖鑑項目（名稱或別名；接受 :名稱:、[貼圖:名稱]、<:名稱:ID>）。找不到回 None。"""
    n = (name or "").strip()
    if not n:
        return None
    if m := _TAG.match(n):
        return eset._by_key.get(("sticker", m.group(1).strip().lower()))
    if m := _FULL.match(n):
        n = m.group(1)
    key = n.strip(":：").strip().lower()
    return eset._by_key.get(("emoji", key)) or eset._by_key.get(("sticker", key))


async def pick(llm, eset: EmojiSet, said: str, reply: str, *, bot_name: str, timeout: float) -> Pick:
    """挑一張（或不配）。不會丟例外：出任何錯都回 Pick(entry=None, reason=...)。"""
    if timeout < PICK_MIN_SEC:
        return Pick(reason="no_time")
    t0 = time.monotonic()
    out = Pick()
    try:
        raw = await asyncio.wait_for(
            llm.chat_json(build_messages(eset, said, reply, bot_name), prompts.EMOJI_PICK_SCHEMA,
                          name="emoji_pick", temperature=0.2, max_tokens=PICK_MAX_TOKENS,
                          reasoning_effort="none"),
            timeout)
    except asyncio.TimeoutError:
        out.reason = "timeout"
    except Exception as e:  # LLMError 或其他：不配、照送
        log.info("挑表情失敗：%r", e)
        out.reason = "error"
    else:
        if not isinstance(raw, dict) or not isinstance(raw.get("emoji", None), str):
            out.reason = "bad_json"
        else:
            out.raw = raw["emoji"].strip()[:60]
            if not out.raw:
                out.reason = "none"
            elif (entry := resolve(eset, out.raw)) is None:
                out.reason = "unknown"
            else:
                out.entry = entry
    out.ms = round((time.monotonic() - t0) * 1000)
    return out


def remaining(started: float, now: float | None = None) -> float:
    """文字回話從 started（time.monotonic）開始算，還剩多少秒可以挑表情。"""
    return TEXT_REPLY_BUDGET_SEC - ((time.monotonic() if now is None else now) - started)
