"""一個語音頻道裡的對話（開發規格 6.0 B）：收音 → 切句 → ASR → 要不要回 → LLM → TTS → 播放、插話、沒人就走。"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

import discord
from discord.ext import voice_recv

from .. import alert, trace
from ..brain import to_traditional
from ..llm import LLMError
from ..memory.writer import ChatLine
from ..rag.decompose import Person
from .asr import asr_context
from .names import NicknameCache
from .dave import decrypt_error_count, patch_reader
from .segmenter import Segmenter, Utterance, to_16k
from .tts import sentences

if TYPE_CHECKING:
    from ..bot import YukiBot

log = logging.getLogger(__name__)
DECRYPT_CHECK_SEC = 30  # 多久看一次收音解密失敗的計數
BARGE_IN_MS = 500  # Yuki 講話時，名單上的人連續講超過這麼久就停下來
ASR_NAMES_MAX_CHARS = 400  # 給 ASR 的名字提示最多這麼多字


def should_reply(text: str, *, humans: int, wake: re.Pattern, attention_until: float, now: float) -> bool:
    """頻道裡只有一個真人（就是講話的人）→ 每句都回；有其他人在 → 叫她名字、或叫完後的注意時間內才回。"""
    if not text.strip():
        return False
    if humans <= 1:
        return True
    return bool(wake.search(text)) or now < attention_until


class ListenSink(voice_recv.AudioSink):
    """只收名單上的人：其他人的聲音在這裡就丟掉，不切句、不轉文字、不存。"""

    def __init__(self, segmenter: Segmenter, allowed: set[str]):
        super().__init__()
        self.segmenter = segmenter
        self.allowed = allowed

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data) -> None:
        if user is None or getattr(user, "bot", False) or str(user.id) not in self.allowed:
            return
        self.segmenter.feed(str(user.id), data.pcm, time.monotonic())

    def cleanup(self) -> None:
        pass


class _Reply:
    """正在處理的一句：還沒開始念之前，同一個人的下一句到了就取消它、接在一起重回。"""

    def __init__(self, user_id: str, text: str):
        self.user_id = user_id
        self.text = text
        self.task: asyncio.Task | None = None
        self.line: ChatLine | None = None  # 已經放進對話紀錄的那一行（合併時要拿掉）
        self.playing = False  # 已經開始念了：不能再取消，之後的話走插話邏輯
        self.merged = False  # 被下一句取代了


class VoiceSession:
    def __init__(self, bot: "YukiBot", vc: voice_recv.VoiceRecvClient, segmenter: Segmenter):
        self.bot = bot
        self.vc = vc
        self.s = bot.settings
        self.segmenter = segmenter
        self.allowed = set(self.s.voice_listen_users)
        self.queue: asyncio.Queue = asyncio.Queue()  # 轉好的文字：(Utterance, member, text, ASR 完成時間)
        self._asr_tasks: set[asyncio.Task] = set()
        self.attention_until = 0.0
        self.speaking = False
        self._interrupt = asyncio.Event()
        self._alone_since: float | None = None
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._decrypt_errors = 0
        self._decrypt_checked = time.monotonic()
        self._cur: _Reply | None = None
        self._nicknames = NicknameCache(bot.brain.profiles)

    @property
    def channel(self) -> discord.VoiceChannel:
        return self.vc.channel

    @property
    def channel_key(self) -> str:
        return f"voice-{self.channel.id}"

    def start(self) -> None:
        self.vc.listen(ListenSink(self.segmenter, self.allowed))
        patch_reader(self.vc)
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._watch()), loop.create_task(self._work())]
        log.info("語音：進入 #%s，只聽 %s", self.channel.name, sorted(self.allowed))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for t in [*self._tasks, *self._asr_tasks]:
            if t is not asyncio.current_task():
                t.cancel()
        try:
            if self.vc.is_listening():
                self.vc.stop_listening()
            await self.vc.disconnect()
        except Exception:
            log.exception("離開語音頻道失敗")
        log.info("語音：離開 #%s", self.channel.name)

    # ---------- 背景：切句、插話、沒人就走 ----------
    async def _watch(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(0.1)
                if not self.vc.is_connected():
                    break
                now = time.monotonic()
                if now - self._decrypt_checked >= DECRYPT_CHECK_SEC:
                    self._decrypt_checked = now
                    self._check_decrypt()
                for utt in self.segmenter.tick(now):
                    # 馬上轉文字，不用等 Yuki 講完或想完上一句（ASR 本身一次處理一句）
                    task = asyncio.create_task(self._transcribe(utt))
                    self._asr_tasks.add(task)
                    task.add_done_callback(self._asr_tasks.discard)
                if self.speaking and any(self.segmenter.talking_ms(u) >= BARGE_IN_MS for u in self.allowed):
                    log.info("語音：被插話，停止播放")
                    self.interrupt()
                humans = [m for m in self.channel.members if not m.bot]
                if humans:
                    self._alone_since = None
                elif self._alone_since is None:
                    self._alone_since = now
                elif now - self._alone_since >= self.s.voice_alone_leave_seconds:
                    log.info("語音：#%s 沒人了，離開", self.channel.name)
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("語音監看失敗")
        await self.bot.end_voice(self.channel.guild.id)

    def _check_decrypt(self) -> None:
        """每 30 秒看一次解密失敗計數；有新增就寫一行 warning 並通知（不是每個封包都通知）。"""
        total = decrypt_error_count(getattr(self.vc, "_dave_stats", None))
        new = total - self._decrypt_errors
        if new <= 0:
            return
        self._decrypt_errors = total
        log.warning("語音：最近 %d 秒有 %d 個封包解密失敗（累計 %d），已用靜音取代", DECRYPT_CHECK_SEC, new, total)
        alert.notify("voice_decrypt", "dave.py:e2ee_decrypt",
                     f"語音收音解密失敗（#{self.channel.name} 最近有 {new} 個封包聽不到）")

    def interrupt(self) -> None:
        self._interrupt.set()
        if self.vc.is_playing():
            self.vc.stop()

    # ---------- 一句一句處理 ----------
    async def _transcribe(self, utt: Utterance) -> None:
        member = self._member(utt.user_id)
        if member is None:
            return
        t0 = time.monotonic()
        context = asr_context(self.asr_names(), self.s.voice_asr_hotwords, ASR_NAMES_MAX_CHARS)
        try:
            text = to_traditional(await self.bot.voice_asr.transcribe(to_16k(utt.audio48), context))
        except Exception:
            log.exception("語音辨識失敗")
            return
        t = time.monotonic()
        log.info("語音：%s（%.1f 秒）→「%s」ASR %.2f 秒（講完後 %.2f 秒）", member.display_name, utt.seconds,
                 text, t - t0, t - utt.ended)
        if text:
            self._enqueue((utt, member, text, t, (t - t0) * 1000))

    def asr_names(self) -> list[str]:
        """給 ASR 提示的名字，越前面越重要：語音頻道裡的人 → Yuki → 伺服器所有成員的名稱 → 人物檔案裡的綽號。
        測試伺服器的成員和人物檔案借主伺服器的（跟回覆一樣）。只有名字，沒有對話。"""
        guild = self.channel.guild
        names = [m.display_name for m in self.channel.members if not m.bot] + [self.s.bot_name]
        try:
            names += [p.name for p in self.bot.members_of(guild)]
            for gid in dict.fromkeys([str(guild.id), self.bot.archive_guild(guild)]):
                if gid:
                    names += self._nicknames.get(gid)
        except Exception:
            log.warning("語音：整理 ASR 名字提示失敗，只用頻道裡的名字", exc_info=True)
        return [n for n in dict.fromkeys(names) if n and (len(n) >= 2 or n == self.s.bot_name)]

    def _enqueue(self, item: tuple) -> None:
        """轉好的一句進排隊。同一個人上一句的回覆還沒開始念（還在想或還在合成）→ 取消那個回覆，
        兩句接成一句重新回；已經開始念的不動（那是插話，由 _watch 的插話邏輯處理）。"""
        utt, member, text, t_asr, asr_ms = item
        cur = self._cur
        if cur and cur.task and not cur.task.done() and not cur.playing and cur.user_id == str(member.id):
            cur.merged = True
            cur.task.cancel()  # 取消 LLM／TTS 的 await：LM Studio 的連線會斷、排隊的位置會放掉
            h = self.bot.brain.history[self.channel_key]
            if cur.line is not None and h and h[-1] is cur.line:
                h.pop()  # 上一句還在對話紀錄裡：拿掉，免得合併後同一句話出現兩次
            log.info("語音：回覆還沒開始念，下一句到了，兩句合併重回")
            self.queue.put_nowait((utt, member, f"{cur.text} {text}", t_asr, asr_ms, text))
            return
        self.queue.put_nowait(item)
    async def _work(self) -> None:
        while not self._closed:
            item = await self.queue.get()
            cur = self._cur = _Reply(str(item[1].id), item[2])
            cur.task = asyncio.ensure_future(self.handle(*item))
            try:
                await asyncio.wait({cur.task})  # 不用 await task 本身：task 被合併取消時這裡不能跟著被取消
            except asyncio.CancelledError:
                cur.task.cancel()
                raise
            finally:
                self._cur = None
            if cur.task.cancelled():
                continue
            try:
                cur.task.result()
            except LLMError:
                log.warning("語音：LM Studio 失敗，這句沒回", exc_info=True)  # 已通知；用 warning 避免重複通知
            except Exception:
                log.exception("語音：處理一句話失敗")

    def _member(self, user_id: str) -> discord.Member | None:
        return self.channel.guild.get_member(int(user_id))

    async def handle(self, utt: Utterance, member: discord.Member, text: str, t_asr: float,
                     asr_ms: float | None = None, new_text: str | None = None) -> None:
        """new_text：合併時只有新的那一句（紀錄頻道只貼新的，前一句已經貼過）。"""
        await self._post_log(f"🎙️ **{member.display_name}**：{new_text or text}")

        author = Person(str(member.id), member.display_name)
        line = ChatLine(author, text)
        self.bot.brain.observe(self.channel_key, line)
        if self._cur is not None and self._cur.task is asyncio.current_task():
            self._cur.line = line
        self.bot._last_activity = time.monotonic()
        now = time.monotonic()
        humans = sum(1 for m in self.channel.members if not m.bot)
        if not should_reply(text, humans=humans, wake=self.bot._wake, attention_until=self.attention_until, now=now):
            return
        if humans > 1:
            self.attention_until = now + self.s.voice_attention_seconds
        if not self.queue.empty():
            return  # 後面還有話在排隊：等最後一句再一起回，不要一句一句搶著講

        gid = str(self.channel.guild.id)
        async with trace.start("voice", guild=gid, channel=self.channel.id, trigger="voice",
                               user_ids=[author.id]) as tr:
            if asr_ms is not None:
                tr.lat("asr", asr_ms)
            try:
                await self._reply_and_speak(gid, author, text, utt, t_asr, tr)
            except asyncio.CancelledError:
                if self._cur is None or not self._cur.merged:
                    raise  # 真的被取消（關閉語音）
                # 被下一句取代：這筆記成略過，下一筆帶著兩句接起來的文字重新回
                tr.outcome = "skipped"
                tr.degrade("merged")

    async def _reply_and_speak(self, gid: str, author: Person, text: str, utt: Utterance, t_asr: float, tr) -> None:
        self.bot._replying += 1
        t_start = time.monotonic()
        try:
            voice_now = self.bot.voice.describe(gid, datetime.now())
            reply = await self.bot.brain.reply(gid, self.channel_key, author, text,
                                               members=self.bot.members_of(self.channel.guild),
                                               voice=voice_now, spoken=True,
                                               owner=self.bot.owner_name(self.channel.guild),
                                               archive_guild=self.bot.archive_guild(self.channel.guild))
        except LLMError as e:
            alert.notify("lmstudio", "LM Studio 連線", "語音回覆時連不上 LM Studio", exc=e)
            raise
        finally:
            self.bot._replying -= 1
            tr.lat("llm", (time.monotonic() - t_start) * 1000)
        t_llm = time.monotonic()
        log.info("語音：Yuki 回「%s」LLM %.2f 秒", reply, t_llm - t_asr)
        tr.final = reply or None
        if not reply:
            tr.outcome = "skipped"
            return
        self.bot.brain.observe(self.channel_key, ChatLine(Person(str(self.bot.user.id), self.s.bot_name),
                                                          reply, is_bot=True))
        await self._post_log(f"🔊 **{self.s.bot_name}**：{reply}")
        await self.speak(reply, started=utt.ended)

    async def speak(self, text: str, *, started: float | None = None) -> None:
        parts = sentences(text)[:3]  # 模型偶爾講太長：最多念 3 句
        if not parts:
            return
        self._interrupt.clear()
        tr, t_speak = trace.current(), time.monotonic()
        jobs: list[asyncio.Task] = []
        self.speaking = True
        try:
            # 第一句單獨先合成：BreezyVoice 一次只做一段，同時送好幾段第一句會排在後面
            jobs.append(asyncio.ensure_future(self.bot.voice_tts.synth(parts[0])))
            mp3 = await jobs[0]
            if tr:
                tr.lat("tts", (time.monotonic() - t_speak) * 1000)  # 第一句合成完成的時間
            # 第一句好了才送後面的，一句接一句（念第一句的時候合成第二句）
            for p in parts[1:]:
                jobs.append(asyncio.ensure_future(self._synth_after(jobs[-1], p)))
            for i in range(len(parts)):
                if i:
                    mp3 = await jobs[i]
                if self._interrupt.is_set() or not self.vc.is_connected():
                    break
                if i == 0 and started is not None:
                    log.info("語音：講完到開口 %.2f 秒", time.monotonic() - started)
                if self._cur is not None:
                    self._cur.playing = True  # 開始念了：之後來的話不再合併
                await self._play(mp3)
                if self._interrupt.is_set():
                    break
        finally:
            self.speaking = False
            for job in jobs:
                job.cancel()

    async def _synth_after(self, prev: asyncio.Task, text: str) -> bytes:
        await asyncio.wait({prev})  # 前一段合成完才送這段（前一段失敗也照送，失敗由念到那段時才處理）
        return await self.bot.voice_tts.synth(text)

    async def _play(self, mp3: bytes) -> None:
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        source = discord.FFmpegPCMAudio(io.BytesIO(mp3), pipe=True)
        self.vc.play(source, after=lambda e: loop.call_soon_threadsafe(done.set))
        await done.wait()

    async def _post_log(self, text: str) -> None:
        cid = self.s.voice_log_channel
        if not cid:
            return
        ch = self.bot.get_channel(int(cid))
        if ch is None:
            return
        try:
            await ch.send(text, allowed_mentions=discord.AllowedMentions.none())
        except discord.HTTPException:
            log.exception("語音紀錄貼不上去")
