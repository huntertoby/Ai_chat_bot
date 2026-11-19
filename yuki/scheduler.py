"""每個頻道的回覆排程（從 bot.py 抽出來，不碰 Discord，方便測試、評估腳本也用同一套）。

管的事：
- 誰佔住頻道：同一個頻道同時只回一則（`Throttle.try_start` / `finish` 只在這裡呼叫）。
- 被叫到：單人洗版限制、頻道熱鬧模式（收集起來，離上次回覆滿 gap 秒再合成一次回）、
  回覆中被叫的排隊合併。
- 插話：去抖動（同頻道最後一則後幾秒沒人再講才判斷）、判斷前後的程式端限制（`chime_gate`、`plan_chime`）。
- 注意時間：被叫到之後的注意模式、她自己插話之後的注意時間。

真正「怎麼回」（叫 LLM、送訊息）由呼叫端給的 `respond(turn)` 做；「要不要插話」的 LLM 判斷由呼叫端給的
`judge()` 做，這裡只負責什麼時候呼叫、呼叫前後的限制。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Awaitable, Callable

from . import throttle
from .chime import ChimeLimits, Decision
from .config import Settings
from .chat import Person

log = logging.getLogger(__name__)


@dataclass
class Turn:
    """一次要回覆的事：回覆哪則訊息、誰說的。被叫到排隊的、自己決定插話的都用這個。"""
    message: Any
    author: Person
    text: str
    mentions: list[Person]
    trigger: str = "called"          # called / attention / chime / chime_attention
    background: list[str] = field(default_factory=list)  # 插話判斷的 need_background（B4）
    chimed: bool = False

    @staticmethod
    def merge(turns: list["Turn"]) -> "Turn | None":
        """排隊期間被叫到的訊息合成一次回覆：回覆最後一則，@ 到的人聯集（對話紀錄裡本來就有全部的訊息）。"""
        if not turns:
            return None
        last = turns[-1]
        people: dict[str, Person] = {}
        for t in turns:
            for m in t.mentions:
                people.setdefault(m.id, m)
        # 有任何一則是真的叫她才算 called（會延長注意時間）；全部是注意模式接話的就維持原本的 trigger
        trigger = "called" if any(t.trigger == "called" for t in turns) else last.trigger
        return Turn(last.message, last.author, last.text, list(people.values()), trigger=trigger)


def throttle_config(s: Settings) -> throttle.ThrottleConfig:
    return throttle.ThrottleConfig(
        cooldown_sec=s.throttle_cooldown_sec, tag_limit=s.throttle_tag_limit,
        tag_window_sec=s.throttle_tag_window_sec, tag_cooldown_sec=s.throttle_tag_cooldown_sec,
        channel_call_burst=s.channel_call_burst, channel_call_window_sec=s.channel_call_window_sec,
        channel_burst_mode_sec=s.channel_burst_mode_sec, channel_call_gap_sec=s.channel_call_gap_sec)


def decide_options(s: Settings) -> dict:
    """正式設定給 chime.decide() 的參數（bot 和評估腳本共用，兩邊才不會漂移）。"""
    return {"context_lines": s.chime_context_lines, "max_tokens": s.chime_max_tokens,
            "think_on_mention": s.chime_think_on_mention,
            "think_max_tokens": s.chime_think_max_tokens, "aliases": list(s.wake_words)}


def chime_limits(s: Settings, *, clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], date] = date.today) -> ChimeLimits:
    return ChimeLimits(s.chime_in, threshold=s.chime_threshold, mute_sec=s.chime_mute_minutes * 60,
                       react_max=s.chime_react_max, react_window_sec=s.chime_react_window_min * 60,
                       clock=clock, today=today)


class ReplyScheduler:
    """respond(turn)：真的去回一則（例外會被記下來、不影響排隊的）。clock / sleep 可以換成假的（測試、評估）。"""

    def __init__(self, settings: Settings, respond: Callable[[Turn], Awaitable[Any]], *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
                 today: Callable[[], date] = date.today):
        self.settings = settings
        self._respond = respond
        self._clock = clock
        self._sleep = sleep
        self.throttle = throttle.Throttle(throttle_config(settings), clock=clock)
        self.limits = chime_limits(settings, clock=clock, today=today)
        self._active_until: dict[Any, float] = {}   # 被叫到之後的注意模式
        self._chime_until: dict[Any, float] = {}    # 她自己插話之後（只影響閉嘴判斷和插話的 attention 提示）
        self._flush_tasks: dict[Any, asyncio.Task] = {}
        self._debounce: dict[Any, asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()

    # ---------- 注意時間 ----------
    def attention(self, cid: Any) -> tuple[bool, bool]:
        """(注意模式中, 剛自己插過話)。"""
        now = self._clock()
        return self._active_until.get(cid, 0) > now, self._chime_until.get(cid, 0) > now

    def extend_attention(self, cid: Any) -> None:
        """真的被叫到才呼叫（注意模式接話、插話都不延長）。"""
        self._active_until[cid] = self._clock() + self.settings.attention_timeout

    def note_chimed(self, cid: Any) -> None:
        """自己插話成功：開一段注意時間，對方接話（沒用回覆功能）也算接她的話。"""
        self._chime_until[cid] = self._clock() + self.settings.attention_timeout

    def end_attention(self, cid: Any) -> None:
        """叫她走：注意模式、插話後的注意時間都結束，正在等的插話判斷也取消。"""
        self.cancel_debounce(cid)
        self._active_until.pop(cid, None)
        self._chime_until.pop(cid, None)

    # ---------- 被叫到 ----------
    def admit_call(self, cid: Any, author_id: str, *, count_tag: bool = True) -> str | None:
        """被叫到一次，先過頻道和個人的限制。count_tag：算不算這個人「叫她」的次數（注意模式接話不算，
        但頻道熱鬧模式照算）。回傳 None＝這個人在洗版，不回（他排隊中的也丟掉）；
        "hold"＝熱鬧模式中，先收集起來；"now"＝可以處理。"""
        gate_ok = self.throttle.note_call(cid)  # 頻道層級：每次都算，包含被單人限制擋掉的
        if count_tag and not self.throttle.note_tag(author_id):
            dropped = self.throttle.drop_queued(cid, lambda t: t.author.id == author_id)
            log.info("同一個人一直叫她（洗版），不回（排隊中的 %d 則也丟掉）", dropped)
            return None
        return "now" if gate_ok else "hold"

    def submit(self, cid: Any, turn: Turn, admitted: str) -> Turn | None:
        """把通過 admit_call 的這則排進去。回傳現在就要回的 Turn（已經佔住頻道），或 None（排隊等）。"""
        if admitted == "hold":  # 熱鬧模式：先收集，離上次回覆滿 gap 秒再合成一次回
            self.throttle.enqueue(cid, turn)
            if not self.throttle.is_busy(cid):
                self._schedule_flush(cid)
            return None
        if self.throttle.is_busy(cid):  # 正在回覆：等它回完，排隊的合成一次回
            self.throttle.enqueue(cid, turn)
            return None
        if self.throttle.ready_in(cid) is not None:
            # 還有之前收集的在等：跟這則一起合成，不讓新的一則插隊、舊的再多等一輪
            self.throttle.enqueue(cid, turn)
            return self.take_queued(cid)
        self.throttle.try_start(cid)
        return turn

    async def call(self, cid: Any, turn: Turn, admitted: str) -> None:
        if (now := self.submit(cid, turn, admitted)) is not None:
            await self.run_turns(cid, now)

    def take_queued(self, cid: Any) -> Turn | None:
        """佔住頻道，把排隊的全部合成一則。頻道正在忙就回 None。"""
        if not self.throttle.try_start(cid):
            return None
        turn = Turn.merge(self.throttle.drain(cid))
        if turn is None:
            self.throttle.finish(cid)
        return turn

    def next_turn(self, cid: Any) -> Turn | None:
        """回完一則之後：排隊中已經可以回的合成下一則（熱鬧模式中還沒到時間的留著）。"""
        return Turn.merge(self.throttle.drain_ready(cid))

    async def run_turns(self, cid: Any, turn: Turn | None) -> None:
        """turn 必須是已經佔住頻道的（submit / take_queued / plan_chime 給的）。
        回完一則後，把排隊期間被叫的訊息合成一次回覆；最後一定放開頻道。"""
        try:
            while turn is not None:
                try:
                    await self._respond(turn)
                except Exception:  # 例如 reply 時原訊息被刪：記下來，排隊的那幾則照樣處理
                    log.exception("回覆失敗，繼續處理排隊的訊息")
                turn = self.next_turn(cid)
        finally:
            self.throttle.finish(cid)
        if self.throttle.ready_in(cid) is not None:
            self._schedule_flush(cid)  # 熱鬧模式還有收集到的：等到下一次可回的時間

    def _schedule_flush(self, cid: Any) -> None:
        if cid in self._flush_tasks:
            return
        task = asyncio.create_task(self._flush_later(cid))
        self._flush_tasks[cid] = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _flush_later(self, cid: Any) -> None:
        turn = None
        try:
            while True:
                wait = self.throttle.ready_in(cid)
                if wait is None or self.throttle.is_busy(cid):
                    return  # 已經沒有排隊的，或正在回（回完 run_turns 會接手）
                if wait > 0:
                    await self._sleep(wait)
                    continue
                turn = self.take_queued(cid)
                break
        finally:
            self._flush_tasks.pop(cid, None)
        if turn is not None:
            await self.run_turns(cid, turn)

    # ---------- 插話 ----------
    def cancel_debounce(self, cid: Any) -> None:
        task = self._debounce.pop(cid, None)
        if task is not None:
            task.cancel()

    def schedule_chime(self, cid: Any, judge: Callable[[], Awaitable[Turn | None]]) -> None:
        """去抖動：同頻道最後一則後 CHIME_DEBOUNCE_SEC 秒沒人再講，才呼叫 judge()（判斷最新那則）。
        judge 回 Turn（已經由 plan_chime 佔住頻道）就接著回覆。"""
        self.cancel_debounce(cid)
        task = asyncio.create_task(self._chime_later(cid, judge))
        self._debounce[cid] = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _chime_later(self, cid: Any, judge: Callable[[], Awaitable[Turn | None]]) -> None:
        try:
            await self._sleep(self.settings.chime_debounce_sec)
        except asyncio.CancelledError:
            return  # 又有人講話了，由新的那則重新計時
        if self._debounce.get(cid) is asyncio.current_task():
            del self._debounce[cid]  # 開始判斷之後不再被取消
        try:
            turn = await judge()
            if turn is not None:
                await self.run_turns(cid, turn)
        except Exception:
            log.exception("插話判斷失敗")

    async def wait_until(self, ready: Callable[[], bool], timeout: float, step: float = 0.5) -> bool:
        """等到 ready() 為真（D-34：注意模式的判斷遇到 LLM 忙就等）。超過 timeout 秒還沒好回 False。"""
        deadline = self._clock() + timeout
        while not ready():
            if self._clock() >= deadline:
                return False
            await self._sleep(step)
        return True

    def chime_gate(self, cid: Any, *, attention: bool, llm_busy: bool, named: bool = False) -> str | None:
        """呼叫模型判斷之前的限制。回傳不判斷的原因，None＝可以判斷。
        named（D-32）：提到她名字的訊息不吃頻道名額（閉嘴、忙碌、冷卻、LLM 忙照舊）。"""
        return (self.limits.gate(cid, attention=attention, named=named)
                or ("busy" if self.throttle.is_busy(cid) else None)
                or ("cooldown" if self.throttle.in_cooldown(cid) else None)
                or ("llm_busy" if llm_busy else None))

    def plan_chime(self, d: Decision, cid: Any, uid: str, *, attention: bool) -> str:
        """模型判斷完之後的程式端限制。回傳：
        "reply"＝要插話（已經佔住頻道、記下次數，接著用 run_turns 回）；
        "react"＝可以按表情（按成功後呼叫 record_react）；"none"＝不動；"error"＝判斷失敗；
        "skipped:<原因>"＝模型想動但被限制擋下。"""
        if d.error:
            return "error"
        if d.should_reply(self.limits.threshold):
            if not self.limits.kind_ok(uid, d.kind, attention=attention):
                return "skipped:kind_daily_limit"
            if not self.throttle.try_start(cid):
                return "skipped:busy"
            self.limits.record_reply(cid, uid, d.kind, attention=attention)
            return "reply"
        if d.should_react(0.0):
            # 按表情不看門檻（模型已經選了 react），但有自己的次數限制，也受同人同類一天一次限制
            if gate := self.limits.react_gate(cid):
                return f"skipped:{gate}"
            if not self.limits.kind_ok(uid, d.kind, attention=attention):
                return "skipped:kind_daily_limit"
            return "react"
        return "none"

    def record_react(self, cid: Any, uid: str, kind: str) -> None:
        self.limits.record_react(cid, uid, kind)

    def mute(self, cid: Any) -> None:
        self.limits.mute(cid)

    # ---------- 關閉 ----------
    async def close(self) -> None:
        tasks = list(self._tasks)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._debounce.clear()
        self._flush_tasks.clear()
