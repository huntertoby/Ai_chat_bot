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
from .dave import decrypt_error_count, patch_reader
from .segmenter import Segmenter, Utterance, to_16k
from .tts import sentences

if TYPE_CHECKING:
    from ..bot import YukiBot

log = logging.getLogger(__name__)
DECRYPT_CHECK_SEC = 30  # 多久看一次收音解密失敗的計數
BARGE_IN_MS = 500  # Yuki 講話時，名單上的人連續講超過這麼久就停下來


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
        names = [m.display_name for m in self.channel.members] + [self.s.bot_name]
        context = asr_context(names, self.s.voice_asr_hotwords)
        try:
            text = to_traditional(await self.bot.voice_asr.transcribe(to_16k(utt.audio48), context))
        except Exception:
            log.exception("語音辨識失敗")
            return
        t = time.monotonic()
        log.info("語音：%s（%.1f 秒）→「%s」ASR %.2f 秒（講完後 %.2f 秒）", member.display_name, utt.seconds,
                 text, t - t0, t - utt.ended)
        if text:
            self.queue.put_nowait((utt, member, text, t, (t - t0) * 1000))

    async def _work(self) -> None:
        while not self._closed:
            item = await self.queue.get()
            try:
                await self.handle(*item)
            except asyncio.CancelledError:
                raise
            except LLMError:
                log.warning("語音：LM Studio 失敗，這句沒回", exc_info=True)  # 已通知；用 warning 避免重複通知
            except Exception:
                log.exception("語音：處理一句話失敗")

    def _member(self, user_id: str) -> discord.Member | None:
        return self.channel.guild.get_member(int(user_id))

    async def handle(self, utt: Utterance, member: discord.Member, text: str, t_asr: float,
                     asr_ms: float | None = None) -> None:
        await self._post_log(f"🎙️ **{member.display_name}**：{text}")

        author = Person(str(member.id), member.display_name)
        self.bot.brain.observe(self.channel_key, ChatLine(author, text))
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
        # 每句同時開始合成，念第一句的時候後面的已經在準備
        jobs = [asyncio.create_task(self.bot.voice_tts.synth(p)) for p in parts]
        self.speaking = True
        tr, t_speak = trace.current(), time.monotonic()
        try:
            for i, job in enumerate(jobs):
                mp3 = await job
                if i == 0 and tr:
                    tr.lat("tts", (time.monotonic() - t_speak) * 1000)  # 第一句合成完成的時間
                if self._interrupt.is_set() or not self.vc.is_connected():
                    break
                if i == 0 and started is not None:
                    log.info("語音：講完到開口 %.2f 秒", time.monotonic() - started)
                await self._play(mp3)
                if self._interrupt.is_set():
                    break
        finally:
            self.speaking = False
            for job in jobs:
                job.cancel()

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
