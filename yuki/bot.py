"""Discord 端：判斷要不要回、注意力機制、指令。實際的思考都交給 Brain。"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from . import alert, chime, emoji_catalog, throttle, trace, vision
from .archive import ArchiveDB
from .archive_sync import ArchiveSync, is_public
from .brain import Brain
from .config import ROOT, Settings, load_settings
from .llm import LLMError, LMStudioClient
from .memory.text_index import add_words
from .memory.writer import ChatLine, ReplyRef
from .optout import forget_everywhere
from .profile.auto import ProfileUpdater
from .profile.builder import ProfileBuilder
from .profile.store import ProfileDB
from .rag.decompose import Person, decompose
from .voice_presence import VoicePresence

if TYPE_CHECKING:
    from .voice.session import VoiceSession

log = logging.getLogger(__name__)
ARCHIVE_PATH = ROOT / "data" / "archive.db"
NO_PING = discord.AllowedMentions.none()  # LLM 輸出 @everyone 也不會真的 tag 到人


def split_message(text: str, limit: int = 1900) -> list[str]:
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(buf) + len(line) > limit:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf.strip():
        chunks.append(buf)
    return chunks or ["…"]


def guild_key(guild: discord.Guild | None, user: discord.abc.User) -> str:
    """記憶以伺服器隔離；私訊則每個人各自一份。"""
    return str(guild.id) if guild else f"dm-{user.id}"


def person(member: discord.abc.User) -> Person:
    return Person(str(member.id), getattr(member, "display_name", None) or member.name)


@dataclass
class Turn:
    """一次要回覆的事：回覆哪則訊息、誰說的。被叫到排隊的、自己決定插話的都用這個。"""
    message: discord.Message
    author: Person
    text: str
    mentions: list[Person]
    trigger: str = "called"          # called / chime / chime_attention
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
        return Turn(last.message, last.author, last.text, list(people.values()), trigger="called")


class YukiBot(commands.Bot):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        # 只用斜線指令（bot.tree），文字指令前綴用不到
        super().__init__(command_prefix=commands.when_mentioned, intents=intents,
                         allowed_mentions=NO_PING)
        self.settings = settings
        # 維運（開發規格 6.12）：出錯私訊擁有者、AI 追蹤紀錄
        self.alerter = alert.Alerter(settings.owner_user_ids, dedupe_min=settings.alert_dedupe_min,
                                     max_per_hour=settings.alert_max_per_hour)
        alert.set_default(self.alerter)
        logging.getLogger().addHandler(alert.AlertLogHandler(self.alerter))
        self._health_task: asyncio.Task | None = None
        self.emoji_sets = emoji_catalog.load_all()  # 只有 reviewed 的圖鑑（C1-2）
        self.emoji_pacing = emoji_catalog.EmojiPacing()
        self.brain = Brain.build(settings)
        self.active_until: dict[int, float] = {}   # 被叫到之後的注意模式
        self.chime_until: dict[int, float] = {}    # 她自己插話之後的注意時間（只影響閉嘴判斷和插話的 attention 提示）
        self._members: dict[int, list[Person]] = {}
        self._wake = re.compile("|".join(re.escape(w) for w in settings.wake_words), re.IGNORECASE)
        self._synced = False
        # B3 節流、B2 插話判斷的硬限制
        self.throttle = throttle.Throttle(throttle.ThrottleConfig(
            cooldown_sec=settings.throttle_cooldown_sec, tag_limit=settings.throttle_tag_limit,
            tag_window_sec=settings.throttle_tag_window_sec, tag_cooldown_sec=settings.throttle_tag_cooldown_sec,
            channel_call_burst=settings.channel_call_burst, channel_call_window_sec=settings.channel_call_window_sec,
            channel_burst_mode_sec=settings.channel_burst_mode_sec, channel_call_gap_sec=settings.channel_call_gap_sec))
        self._flush_tasks: dict[int, asyncio.Task] = {}
        self.limits = chime.ChimeLimits(settings.chime_in, threshold=settings.chime_threshold,
                                        mute_sec=settings.chime_mute_minutes * 60,
                                        react_max=settings.chime_react_max,
                                        react_window_sec=settings.chime_react_window_min * 60)
        self._debounce: dict[int, asyncio.Task] = {}
        self._chime_tasks: set[asyncio.Task] = set()
        self.archive = ArchiveDB(ARCHIVE_PATH)
        self.sync = ArchiveSync(self.archive, settings.archive_role, set(settings.archive_exclude))
        trace.configure(settings.trace_dir, max_days=settings.trace_max_days, max_mb=settings.trace_max_mb,
                        opted_out=self._trace_opted_out)
        self.voice = VoicePresence(self.archive.db)
        self.brain.archive_db = self.archive.db  # agentic 檢索（開發規格 6.4）
        self.brain.opted_out = self._all_optouts
        self.brain.optout_names = self._optout_names
        self._catchup_tasks: set = set()
        # 閒置判斷：最後一則真人訊息的時間、正在回覆的數量、是否正在補抓
        self._last_activity = time.monotonic()
        self._replying = 0
        self._catching_up = 0
        self._profile_task: asyncio.Task | None = None
        self._profile_llm: LMStudioClient | None = None
        # 語音對話（開發規格 6.0）：模型第一次 /來語音 才載入
        self.voice_sessions: dict[int, "VoiceSession"] = {}
        self.voice_asr = None
        self.voice_tts = None
        self._voice_lock = asyncio.Lock()
        register_commands(self)

    async def setup_hook(self) -> None:
        if self.settings.voice_enabled:
            from .voice.dave import install
            install()  # 要在連語音之前：補上 DAVE 收音解密
        self.loop.create_task(self._warmup())

    async def sync_commands(self, guild: discord.Guild) -> None:
        # 同步成伺服器指令：立即生效（全域指令可能要等很久才出現）
        self.tree.copy_global_to(guild=guild)
        synced = await self.tree.sync(guild=guild)
        log.info("斜線指令已同步到 %s：%s", guild.name, [c.name for c in synced])

    async def _warmup(self) -> None:
        try:
            await self.brain.warmup()
        except Exception:
            # 只寫 warning：LM Studio 沒開時由健康檢查通知，避免同一件事收到兩則私訊
            log.warning("暖機失敗（LM Studio 沒開？），第一次對話時會再試", exc_info=True)

    def _trace_opted_out(self, guild_id: str, user_ids: list[str]) -> bool:
        """`/不要記我` 的人（任何伺服器）參與的觸發，追蹤紀錄整筆不寫。"""
        marks = ",".join("?" * len(user_ids))
        row = self.archive.db.execute(f"SELECT 1 FROM optout WHERE user_id IN ({marks}) LIMIT 1",
                                      list(user_ids)).fetchone()
        return row is not None

    def _all_optouts(self) -> set[str]:
        """打過 /不要記我 的人（不分在哪個伺服器打的）。"""
        return self.archive.opted_out()

    def _optout_names(self) -> set[str]:
        """退出的人在 Yuki 所在各伺服器的名字（暱稱、帳號名、全域名稱），上網工具不准送出這些字。"""
        names: set[str] = set()
        for uid in self._all_optouts():
            if not uid.isdigit():
                continue
            users = [g.get_member(int(uid)) for g in self.guilds] + [self.get_user(int(uid))]
            for u in filter(None, users):
                names |= {n for n in (getattr(u, "display_name", None), u.name, getattr(u, "global_name", None)) if n}
        return names

    async def _dm_owner(self, user_id: str, text: str) -> None:
        user = await self.fetch_user(int(user_id))
        await user.send(text, allowed_mentions=NO_PING)

    async def on_error(self, event_method: str, /, *args, **kwargs) -> None:
        # discord.py 預設只印到 stderr；改成 log，擁有者才收得到通知
        log.exception("事件 %s 發生未捕捉的例外", event_method)

    async def close(self) -> None:
        for task in list(self._chime_tasks):
            task.cancel()
        if self._health_task:
            self._health_task.cancel()
        for gid in list(self.voice_sessions):
            await self.end_voice(gid)
        if self._profile_task:
            self._profile_task.cancel()
        if self._profile_llm:
            await self._profile_llm.aclose()
        await self.brain.aclose()
        await super().close()

    async def on_ready(self) -> None:
        log.info("已登入為 %s，伺服器數：%d", self.user, len(self.guilds))
        self.alerter.connect(self._dm_owner, self.loop)  # 連上之前排隊的通知在這裡一次補發
        if self._health_task is None:
            self._health_task = self.loop.create_task(
                alert.health_loop(self.settings.lmstudio_base_url, self.settings.alert_health_sec, self.alerter))
        if not self._synced:  # on_ready 斷線重連時會再觸發，只同步一次
            self._synced = True
            for guild in self.guilds:
                try:
                    await self.sync_commands(guild)
                except discord.HTTPException:
                    log.exception("同步斜線指令到 %s 失敗", guild.name)
        for guild in self.guilds:
            self._snapshot_voice(guild)
        self._sync_emoji_sets()
        # 每次 on_ready 都補抓：第一次是補關機期間，之後是斷線又無法恢復連線時補缺口
        if self.settings.catchup_enabled:
            task = self.loop.create_task(self._catch_up_all())
            self._catchup_tasks.add(task)
            task.add_done_callback(self._catchup_tasks.discard)
        self._start_profile_updater()

    # ---------- 語音頻道在場狀態 ----------
    def _voice_room(self, guild: discord.Guild, channel) -> tuple[str, str] | None:
        """公開的語音頻道才算（私人頻道裡有誰不講出去）。"""
        if channel is None or not is_public(channel, self.sync.role(guild), self.sync.exclude):
            return None
        return str(channel.id), channel.name

    def _snapshot_voice(self, guild: discord.Guild) -> None:
        """上線（或重連）時，從 Discord 快取重建誰在語音裡。"""
        gid, now = str(guild.id), datetime.now()
        self.voice.reset_guild(gid, now)
        skip = self.sync.skip(gid)
        n = 0
        for ch in list(guild.voice_channels) + list(guild.stage_channels):
            room = self._voice_room(guild, ch)
            if room is None:
                continue
            for m in ch.members:
                if not m.bot:
                    self.voice.present(gid, str(m.id), m.display_name, *room, now, record=str(m.id) not in skip)
                    n += 1
        log.info("%s：語音頻道現在有 %d 人", guild.name, n)

    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState,
                                    after: discord.VoiceState) -> None:
        if member.bot:
            return
        gid = str(member.guild.id)
        ev = self.voice.update(gid, str(member.id), member.display_name,
                               self._voice_room(member.guild, before.channel),
                               self._voice_room(member.guild, after.channel), datetime.now(),
                               record=str(member.id) not in self.sync.skip(gid))
        if ev:
            log.info("語音：%s %s %s%s", ev.name, ev.kind, ev.channel, f" → {ev.to_channel}" if ev.to_channel else "")

    def archive_guild(self, guild: discord.Guild | None) -> str:
        """agentic 檢索查哪個伺服器的封存區：測試伺服器沒有封存，借主伺服器的。"""
        main = self.settings.profile_fallback_guild
        if self.is_test_guild(guild) and main:
            return main
        return str(guild.id) if guild else ""

    def owner_name(self, guild: discord.Guild | None) -> str:
        """主人在這個伺服器的顯示名稱（多個就用「、」接起來），給 prompt 用。"""
        names = []
        for uid in self.settings.owner_user_ids:
            m = guild.get_member(int(uid)) if guild else None
            u = m or self.get_user(int(uid))
            if u:
                names.append(getattr(u, "display_name", None) or u.name)
        return "、".join(names)

    # ---------- 語音對話 ----------
    def voice_allowed(self, guild: discord.Guild | None) -> bool:
        s = self.settings
        return bool(s.voice_enabled and guild and str(guild.id) in s.voice_guild_ids)

    async def _voice_models(self) -> None:
        if self.voice_asr is None:
            from .voice.asr import QwenASR
            from .voice.tts import BreezyTTS, EdgeTTS
            asr = QwenASR(self.settings.voice_asr_model)
            await asr.warmup()
            tts = EdgeTTS(self.settings.voice_tts_voice)
            if self.settings.voice_tts == "breezy":
                tts = BreezyTTS(self.settings.voice_tts_url, fallback=tts)
            self.voice_asr, self.voice_tts = asr, tts

    async def start_voice(self, channel: discord.VoiceChannel) -> "VoiceSession":
        from discord.ext import voice_recv

        from .voice.segmenter import Segmenter, silero_factory
        from .voice.session import VoiceSession
        async with self._voice_lock:
            await self.end_voice(channel.guild.id)
            await self._voice_models()
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
            seg = Segmenter(silero_factory(), silence_ms=self.settings.voice_silence_ms)
            session = VoiceSession(self, vc, seg)
            self.voice_sessions[channel.guild.id] = session
            session.start()
            return session

    async def end_voice(self, guild_id: int) -> bool:
        session = self.voice_sessions.pop(guild_id, None)
        if session is None:
            return False
        await session.close()
        return True

    # ---------- 人物檔案：閒置時自動增量更新 ----------
    def is_idle(self) -> bool:
        s = self.settings
        quiet = time.monotonic() - self._last_activity >= s.profile_idle_minutes * 60
        return quiet and self._replying == 0 and self._catching_up == 0

    def _start_profile_updater(self) -> None:
        s = self.settings
        if self._profile_task or not (s.profile_enabled and s.profile_auto_update):
            return
        # 跟聊天共用同一個排隊，LM Studio 不會同時收到兩個請求
        self._profile_llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, timeout=max(s.llm_timeout, 600),
                                           reasoning_effort=s.profile_reasoning_effort,
                                           json_reasoning_effort=s.profile_reasoning_effort,
                                           slots=self.brain.llm._slots)
        db = ProfileDB(s.profile_db_path)

        def make_builder(guild_id: str) -> ProfileBuilder:
            return ProfileBuilder(self.archive.db, db, self.brain.profiles, self._profile_llm, guild_id,
                                  segment_chars=s.profile_segment_chars, profile_limit=s.profile_max_chars)

        updater = ProfileUpdater(make_builder, db, self.brain.profiles, self.archive.db, self.is_idle,
                                 self.archive.opted_out, min_new=s.profile_min_new,
                                 skip_guilds=set(s.test_guild_ids))
        self._profile_task = self.loop.create_task(updater.run_forever())
        log.info("人物檔案自動更新：閒置 %d 分鐘、新訊息滿 %d 則時才跑", s.profile_idle_minutes, s.profile_min_new)

    async def _catch_up_all(self) -> None:
        self._catching_up += 1
        try:
            await self._catch_up_guilds()
        finally:
            self._catching_up -= 1

    def chime_allowed(self, guild: discord.Guild | None) -> bool:
        """哪些伺服器開插話判斷：CHIME_GUILD_IDS；留空＝只有測試伺服器（主伺服器等階段 D 再開）。"""
        if guild is None:
            return False
        ids = self.settings.chime_guild_ids or self.settings.test_guild_ids
        return str(guild.id) in ids

    def emoji_set(self, guild: discord.Guild | None) -> emoji_catalog.EmojiSet | None:
        """這個伺服器可以用的表情圖鑑：EMOJI_GUILD_IDS（留空＝只有測試伺服器）、reviewed，而且是**這個伺服器自己**的
        圖鑑（不會借別的伺服器的，PROFILE_FALLBACK_GUILD 的圖鑑不會出現在測試伺服器）。"""
        if guild is None:
            return None
        ids = self.settings.emoji_guild_ids or self.settings.test_guild_ids
        gid = str(guild.id)
        return self.emoji_sets.get(gid) if gid in ids else None

    def _sync_emoji_sets(self) -> None:
        """啟動時比對 guild.emojis：圖鑑有、伺服器已刪的不用；新增或改名的記在 log（自動補建之後再做）。"""
        for guild in self.guilds:
            eset = self.emoji_sets.get(str(guild.id))
            if eset is None:
                continue
            rep = eset.sync(guild.emojis, guild.stickers)
            log.info("表情圖鑑比對（%s）：已刪 %d、改名 %d、新增（圖鑑沒有）%d，可用 %d 張",
                     guild.id, len(rep["removed"]), len(rep["renamed"]), len(rep["new"]), len(eset))

    def is_test_guild(self, guild: discord.Guild | None) -> bool:
        """測試伺服器：不封存、不在背景整理記憶（私人伺服器的頻道可能有密碼之類的東西）。"""
        return guild is not None and str(guild.id) in self.settings.test_guild_ids

    async def _catch_up_guilds(self) -> None:
        for guild in list(self.guilds):
            if self.is_test_guild(guild):
                continue
            try:
                new = await self.sync.catch_up(guild)
            except Exception:
                log.exception("%s：補抓失敗", guild.name)
                continue
            if not new:
                continue
            total = sum(len(rows) for rows in new.values())
            log.info("%s：補抓到 %d 則，開始整理成記憶（%d 個頻道）", guild.name, total, len(new))
            chunks = 0
            for channel_id, rows in new.items():
                lines = [(ChatLine(Person(r.author_id, r.author_name), r.content), r.created_at)
                         for r in rows if r.content.strip()]
                try:
                    chunks += await self.brain.catch_up(str(guild.id), channel_id, lines)
                except Exception:
                    log.exception("整理補抓的訊息失敗（頻道 %s）", channel_id)
            log.info("%s：補抓的訊息整理完成，共 %d 段", guild.name, chunks)

    async def on_guild_join(self, guild: discord.Guild) -> None:
        await self.sync_commands(guild)

    # ---------- 成員名單（給查詢分解找人名用）----------
    def members_of(self, guild: discord.Guild | None) -> list[Person]:
        if guild is None:
            return []
        if guild.id not in self._members:
            people = []
            for m in guild.members:
                if m.bot:
                    continue
                for alias in {m.display_name, m.name, getattr(m, "global_name", None)}:
                    if alias:
                        people.append(Person(str(m.id), alias))
            # 測試伺服器：也認得主伺服器的人，講到「阿明」才找得到他的人物檔案（pick_profiles 會借主伺服器的）
            main = self.settings.profile_fallback_guild
            if self.is_test_guild(guild) and main and (g := self.get_guild(int(main))):
                seen = {p.id for p in people}
                for m in g.members:
                    if m.bot or str(m.id) in seen:
                        continue
                    for alias in {m.display_name, m.name, getattr(m, "global_name", None)}:
                        if alias:
                            people.append(Person(str(m.id), alias))
            # 記憶裡有、但已不在伺服器的人（退出的成員、匯入的測試資料）也要認得出名字
            known = {p.id for p in people}
            people += [Person(sid, name) for sid, name in self.brain.store.subjects(str(guild.id))
                       if sid not in known]
            add_words([p.name for p in people])
            self._members[guild.id] = people
        return self._members[guild.id]

    async def on_member_join(self, member: discord.Member) -> None:
        self._members.pop(member.guild.id, None)

    async def on_member_update(self, before: discord.Member, after: discord.Member) -> None:
        if before.display_name != after.display_name:
            self._members.pop(after.guild.id, None)

    # ---------- 訊息 ----------
    def clean(self, message: discord.Message) -> str:
        text = message.clean_content
        me = message.guild.me if message.guild else self.user
        text = text.replace(f"@{me.display_name}", " ").replace(f"@{self.user.name}", " ")
        text = self._wake.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip(" ,，")

    def is_called(self, message: discord.Message) -> bool:
        if message.guild is None or self.user in message.mentions:
            return True
        me = message.guild.me
        # 在輸入框選到的「@Yuki」可能是 Discord 幫 bot 建的身分組，不是 bot 本人
        if me and any(r.is_bot_managed() and r in me.roles for r in message.role_mentions):
            return True
        if self._wake.search(message.content):
            return True
        ref = message.reference.resolved if message.reference else None
        return isinstance(ref, discord.Message) and ref.author == self.user

    # ---------- 回覆關係 ----------
    def reply_ref(self, message: discord.Message) -> tuple[ReplyRef | None, str]:
        """這則訊息回覆的是誰：能直接從快取拿到就拿（不花 API），拿不到只留訊息 ID，之後補抓。"""
        ref = message.reference
        if ref is None or not ref.message_id:
            return None, ""
        resolved = ref.resolved
        if isinstance(resolved, discord.Message):
            return self._ref_of(resolved), ""
        return None, str(ref.message_id)

    def _ref_of(self, m: discord.Message) -> ReplyRef:
        text = m.clean_content or (vision.IMAGE_PLACEHOLDER if m.attachments or m.embeds else "")
        name = getattr(m.author, "display_name", None) or m.author.name
        return chime.make_ref(name, m.author == self.user, text)

    async def _fill_reply_refs(self, channel) -> None:
        """對話紀錄裡還沒補到回覆對象的訊息：先從紀錄裡找，找不到才問 Discord（每次最多 3 則）。"""
        async def fetch(message_id: str) -> ReplyRef | None:
            return self._ref_of(await channel.fetch_message(int(message_id)))

        lines = list(self.brain.history[str(channel.id)])
        if any(l.reply_id for l in lines):
            await chime.fill_replies(lines, fetch, limit=3)

    async def on_message(self, message: discord.Message) -> None:
        s = self.settings
        if message.author.bot or message.webhook_id:  # bot（包括 Luna）一律不理，C4 才做互動
            return
        self._last_activity = time.monotonic()
        test = self.is_test_guild(message.guild)
        if not test:
            try:
                self.sync.record(message)
            except Exception:
                log.exception("訊息存進封存區失敗")
        if throttle.is_flood(message.content, repeat_chars=s.flood_repeat_chars,
                             repeat_mentions=s.flood_repeat_mentions):
            log.info("大量重複的訊息，不回、不進記憶（封存照常）")
            return
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return  # 加入、置頂、加成之類的系統訊息
        if throttle.is_command(message.content):
            return
        text = self.clean(message)
        gid, cid = guild_key(message.guild, message.author), message.channel.id
        channel_key = str(cid)
        author = person(message.author)
        mentions = [person(u) for u in message.mentions if u != self.user and not u.bot]
        now = time.monotonic()
        active = self.active_until.get(cid, 0) > now
        chiming = self.chime_until.get(cid, 0) > now  # 她剛自己插過話，大家可能在回她
        called = self.is_called(message)

        # 對她說「閉嘴」「沒問你」：這個頻道一陣子不主動插話（被叫仍回）
        if (called or active or chiming) and chime.is_shush(message.content):
            self.limits.mute(cid)
            log.info("被叫閉嘴，頻道 %s 在 %.0f 分鐘內不主動插話", cid, s.chime_mute_minutes)

        if active and any(w in message.content for w in s.exit_words):
            self.active_until.pop(cid, None)
            await message.reply("好的，我先去旁邊休息囉！有事再叫我～", mention_author=False)
            return

        if s.vision_enabled and (called or active):
            await vision.wait_for_embeds(message)  # 連結預覽是 Discord 事後補上的（訊息編輯），等一下
        images = vision.image_urls(message) if s.vision_enabled else []
        if images and not text:
            text = "（貼了圖片）"  # 只有圖片沒有文字：讓這則進對話紀錄，圖才帶得到
        line = None
        if text:
            reply_to, reply_id = self.reply_ref(message)
            line = ChatLine(author, text, images=images, msg_id=str(message.id), reply_to=reply_to,
                            reply_id=reply_id)
            if self.brain.observe(channel_key, line) and (called or active or (s.passive_enabled and not test)):
                self.brain.buffer(gid, channel_key, line, mentions)

        if called:
            self._cancel_debounce(cid)  # 有人叫她，不用再判斷要不要插話
            await self._on_called(message, author, text, mentions)
            return
        if message.guild is not None and not self.chime_allowed(message.guild):
            # 還沒開插話的伺服器（預設＝測試伺服器以外，例如主伺服器）：維持舊行為，注意模式內照樣回
            if (active and line is not None and not throttle.is_noise_text(text, s.noise_max_chars)
                    and not self.throttle.tag_cooling(author.id)):  # 洗版冷卻中的人，注意模式也不回
                await self._on_called(message, author, text, mentions, count_tag=False)
            return
        # 沒被叫到：先過雜訊過濾，再排進插話判斷（注意模式只是多一個提示，不是必回）
        if (line is None or not self.limits.enabled or message.guild is None
                or throttle.is_noise_text(text, s.noise_max_chars)
                or self.throttle.tag_cooling(author.id)):
            return
        self._schedule_chime(message, line, attention=active or chiming)

    # ---------- 被叫到：節流、排隊、回覆 ----------
    async def _on_called(self, message: discord.Message, author: Person, text: str, mentions: list[Person],
                         *, count_tag: bool = True) -> None:
        cid = message.channel.id
        gate_ok = self.throttle.note_call(cid) if count_tag else True  # 頻道層級：每次被叫都算，包含被單人限制擋掉的
        if count_tag and not self.throttle.note_tag(author.id):
            dropped = self.throttle.drop_queued(cid, lambda t: t.author.id == author.id)
            log.info("%s 一直叫她（洗版），不回（排隊中的 %d 則也丟掉）", author.name, dropped)
            return
        if not text:
            await message.reply("找我有事嗎？", mention_author=False)
            return
        turn = Turn(message, author, text, mentions, trigger="called")
        if not gate_ok:  # 熱鬧模式：先收集，離上次回覆滿 gap 秒再合成一次回
            self.throttle.enqueue(cid, turn)
            if not self.throttle.is_busy(cid):
                self._schedule_flush(cid)
            return
        if not self.throttle.try_start(cid):
            self.throttle.enqueue(cid, turn)  # 這個頻道正在回覆：等它回完，排隊的合成一次回
            return
        await self._run_turns(cid, turn)

    async def _run_turns(self, cid: int, turn: Turn | None) -> None:
        """呼叫端已經 try_start 成功。回完一則後，把排隊期間被叫的訊息合成一次回覆。"""
        try:
            while turn is not None:
                try:
                    await self._respond(turn)
                except Exception:  # 例如 reply 時原訊息被刪：記下來，排隊的那幾則照樣處理
                    log.exception("回覆失敗，繼續處理排隊的訊息")
                turn = Turn.merge(self.throttle.drain_ready(cid))
        finally:
            self.throttle.finish(cid)
        if self.throttle.ready_in(cid) is not None:
            self._schedule_flush(cid)  # 熱鬧模式還有收集到的叫她：等到下一次可回的時間

    def _schedule_flush(self, cid: int) -> None:
        if cid in self._flush_tasks:
            return
        self._flush_tasks[cid] = self.loop.create_task(self._flush_later(cid))

    async def _flush_later(self, cid: int) -> None:
        try:
            delay = self.throttle.ready_in(cid)
            if delay:
                await asyncio.sleep(delay)
        finally:
            self._flush_tasks.pop(cid, None)
        if self.throttle.ready_in(cid) != 0.0 or not self.throttle.try_start(cid):
            if self.throttle.ready_in(cid) is not None and not self.throttle.is_busy(cid):
                self._schedule_flush(cid)
            return  # 忙碌中：回完之後 _run_turns 會處理
        await self._run_turns(cid, Turn.merge(self.throttle.drain(cid)))

    def _optout_among(self, ids) -> set[str]:
        ids = sorted({str(i) for i in ids})
        if not ids:
            return set()
        marks = ",".join("?" * len(ids))
        return {r[0] for r in self.archive.db.execute(f"SELECT user_id FROM optout WHERE user_id IN ({marks})", ids)}

    def _hide_optouts(self, tr, channel_key: str, people: list[Person]) -> None:
        """追蹤紀錄寫檔前，把 /不要記我 的人的訊息（和別人引用他們的片段、名字）拿掉。"""
        def compute() -> list[str]:
            lines = list(self.brain.history.get(channel_key, ()))
            ids = {l.author.id for l in lines if not l.is_bot} | {p.id for p in people}
            return trace.hidden_strings(lines, self._optout_among(ids), people)
        tr.hide_fn = compute

    @staticmethod
    def _replied_person(message) -> list[Person]:
        ref = getattr(getattr(message, "reference", None), "resolved", None)
        a = getattr(ref, "author", None)
        return [person(a)] if a is not None and not getattr(a, "bot", False) else []

    async def _respond(self, turn: Turn) -> bool:
        message, author, mentions = turn.message, turn.author, turn.mentions
        s = self.settings
        gid, cid = guild_key(message.guild, message.author), message.channel.id
        channel_key = str(cid)
        if turn.trigger == "called":
            self.active_until[cid] = time.monotonic() + s.attention_timeout
        self._replying += 1
        tr = trace.start("reply", guild=message.guild.id if message.guild else "", channel=cid,
                         trigger=turn.trigger, user_ids=[author.id, *(m.id for m in mentions)])
        self._hide_optouts(tr, channel_key, self._replied_person(message))
        eset = self.emoji_set(message.guild)
        emoji_ok = eset is not None and self.emoji_pacing.allow(cid)  # 剛用過就這則不放清單
        try:
            async with tr:
                async with message.channel.typing():
                    await self._fill_reply_refs(message.channel)
                    voice = self.voice.describe(str(message.guild.id), datetime.now()) if message.guild else ""
                    reply = await self.brain.reply(gid, channel_key, author, turn.text, mentions=mentions,
                                                   members=self.members_of(message.guild), voice=voice,
                                                   owner=self.owner_name(message.guild),
                                                   archive_guild=self.archive_guild(message.guild),
                                                   background=turn.background, chimed=turn.chimed,
                                                   emoji=eset.prompt_block() if emoji_ok else "")
                tr.final = reply
        except LLMError as e:
            # 用 warning：ERROR 會被 AlertLogHandler 當成另一種錯誤再通知一次
            log.warning("回覆失敗 trace_id=%s", tr.trace_id, exc_info=True)
            if not turn.chimed:  # 自己決定插話的不用道歉，安靜地算了
                self.alerter.notify("lmstudio", "LM Studio 連線", "回覆時連不上 LM Studio", exc=e,
                                    trace_id=tr.trace_id)
                await message.reply("我跟 LM Studio 的連線好像斷了…等等再試一次？", mention_author=False)
            return False
        finally:
            self._replying -= 1
        # 群組表情：`:名稱:` 換成 <:名稱:ID>、壞名稱刪掉、一則最多一個；貼圖另外附（沒有圖鑑也會把 :xxx: 清掉）
        reply, sticker = emoji_catalog.process_reply(reply, eset, allow=emoji_ok)
        if eset is not None:
            self.emoji_pacing.record(cid, bool(sticker) or bool(emoji_catalog.CUSTOM_EMOJI.search(reply)))
        if not reply and not sticker:
            if turn.chimed:
                return False
            reply = "……我腦袋突然一片空白，再說一次？"

        stickers = [discord.Object(id=int(sticker))] if sticker else None
        chunks = split_message(reply) if reply else [None]
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            kw = {"stickers": stickers} if last and stickers else {}
            if i == 0:
                await message.reply(chunk, mention_author=False, **kw)
            else:
                await message.channel.send(chunk, **kw)
        if turn.chimed:  # 自己插話成功：開一段注意時間，對方接話（沒用回覆功能）也算接她的話
            self.chime_until[cid] = time.monotonic() + s.attention_timeout
        bot_line = ChatLine(person(self.user), reply or "[貼圖]", is_bot=True)
        self.brain.observe(channel_key, bot_line)
        self.brain.buffer(gid, channel_key, bot_line)
        return True

    # ---------- 沒被叫到：插話判斷（B2）----------
    def _cancel_debounce(self, cid: int) -> None:
        task = self._debounce.pop(cid, None)
        if task is not None:
            task.cancel()

    def _schedule_chime(self, message: discord.Message, line: ChatLine, *, attention: bool) -> None:
        """去抖動：同頻道最後一則後 CHIME_DEBOUNCE_SEC 秒沒人再講，才判斷最新那則。"""
        cid = message.channel.id
        self._cancel_debounce(cid)
        task = self.loop.create_task(self._chime_later(message, line, attention))
        self._debounce[cid] = task
        self._chime_tasks.add(task)
        task.add_done_callback(self._chime_tasks.discard)

    async def _chime_later(self, message: discord.Message, line: ChatLine, attention: bool) -> None:
        cid = message.channel.id
        try:
            await asyncio.sleep(self.settings.chime_debounce_sec)
        except asyncio.CancelledError:
            return  # 又有人講話了，由新的那則重新計時
        if self._debounce.get(cid) is asyncio.current_task():
            del self._debounce[cid]  # 開始判斷之後不再被取消
        try:
            turn = await self._chime(message, line, attention)
            if turn is not None:
                await self._run_turns(cid, turn)
        except Exception:
            log.exception("插話判斷失敗")

    async def _chime(self, message: discord.Message, line: ChatLine, attention: bool) -> Turn | None:
        """判斷一次。決定要回覆就回傳 Turn（已經佔住這個頻道，由呼叫端 _run_turns 接手）；
        按表情在這裡做完；其餘回 None。每次都寫一筆 chime 追蹤紀錄（跳過的也寫，註明原因）。"""
        s = self.settings
        cid, uid = message.channel.id, str(message.author.id)
        channel_key = str(cid)
        info: dict = {"attention": attention}
        turn: Turn | None = None
        tr = trace.start("chime", guild=message.guild.id if message.guild else "", channel=cid,
                         trigger="attention" if attention else "idle", user_ids=[uid],
                         keep_prompts=False)  # 插話判斷每則都會寫一筆：只記判斷結果，不記帶著別人對話的 messages
        self._hide_optouts(tr, channel_key, self._replied_person(message))
        async with tr:
            def skip(reason: str) -> None:
                tr.outcome = "skipped"
                info["action"] = f"skipped:{reason}"

            reason = (self.limits.gate(cid, attention=attention)
                      or ("busy" if self.throttle.is_busy(cid) else None)
                      or ("cooldown" if self.throttle.in_cooldown(cid) else None)
                      or ("llm_busy" if self._replying > 0 or chime.llm_is_busy(self.brain.llm) else None))
            if reason:
                skip(reason)
            else:
                hist = list(self.brain.history[channel_key])
                try:
                    await self._fill_reply_refs(message.channel)
                    new_images = vision.image_urls(message) if s.vision_enabled else []
                    if new_images and not line.images:
                        line.images = new_images  # 連結預覽的圖是事後補上的，去抖動這幾秒夠它出現
                    images = (await vision.glance_images(hist[-s.vision_recent_lines:], s.chime_max_images,
                                                         long_side=s.vision_glance_long_side,
                                                         recent=s.vision_recent_lines,
                                                         client=self.brain._img_http)
                              if s.vision_enabled and s.chime_max_images > 0 else [])
                except Exception:
                    log.warning("插話判斷取圖失敗，只用文字", exc_info=True)
                    images = []
                if self._replying > 0 or chime.llm_is_busy(self.brain.llm):
                    # 補抓回覆對象、下載圖片的這段時間，LM Studio 可能已經被別人佔住：再檢查一次
                    skip("llm_busy")
                    tr.final = json.dumps(info, ensure_ascii=False)
                    return None
                idx = next((i for i, l in enumerate(hist) if l is line), len(hist))
                eset = self.emoji_set(message.guild)  # 有 reviewed 的圖鑑才放表情清單、才限制只能按圖鑑裡的
                d = await chime.decide(self.brain.llm, recent=hist[:idx], latest=line, bot_name=s.bot_name,
                                       images=images, attention=attention, context_lines=s.chime_context_lines,
                                       max_tokens=s.chime_max_tokens, use_logprobs=s.chime_use_logprobs,
                                       emoji_block=eset.react_block() if eset and s.chime_emoji_list else "")
                tr.lat("decide", d.ms)
                info.update(decision=d.to_dict(), threshold=self.limits.threshold, prob=round(d.participate, 3),
                            prob_source=d.prob_source, images=len(images))
                info["action"] = "none"
                if d.error:
                    tr.degrade("chime_failed")
                elif d.should_reply(self.limits.threshold):
                    if not self.limits.kind_ok(uid, d.kind, attention=attention):
                        skip("kind_daily_limit")
                    elif not self.throttle.try_start(cid):
                        skip("busy")
                    else:
                        self.limits.record_reply(cid, uid, d.kind, attention=attention)
                        info["action"] = "reply"
                        mentions = [person(u) for u in message.mentions if u != self.user and not u.bot]
                        turn = Turn(message, person(message.author), line.text, mentions,
                                    trigger="chime_attention" if attention else "chime",
                                    background=d.need_background, chimed=True)
                elif d.should_react(0.0):
                    # 按表情不看門檻（模型已經選了 react），但有自己的次數限制，也受同人同類一天一次限制
                    gate = self.limits.react_gate(cid)
                    wanted = d.emoji
                    if eset is not None:  # 名稱（含別名）換成真名；圖鑑裡沒有的群組表情不按
                        is_name, wanted = eset.react_name(d.emoji)
                        if is_name and not wanted:
                            wanted = ""
                    emoji = chime.resolve_emoji(wanted, message.guild.emojis if message.guild else ())
                    if gate:
                        skip(gate)
                    elif not self.limits.kind_ok(uid, d.kind, attention=attention):
                        skip("kind_daily_limit")
                    elif emoji is None:
                        skip("emoji_not_found")
                    else:
                        try:
                            await message.add_reaction(emoji)
                            self.limits.record_react(cid, uid, d.kind)
                            info["action"] = "react"
                        except discord.HTTPException:
                            log.info("按表情失敗：%s", d.emoji, exc_info=True)
                            skip("react_failed")
            tr.final = json.dumps(info, ensure_ascii=False)
        return turn


def register_commands(bot: YukiBot) -> None:
    """斜線指令。只註冊到 bot.tree，同步在 on_ready 時對每個伺服器做（伺服器指令立即生效）。"""
    store = bot.brain.store

    async def send(interaction: discord.Interaction, header: str, lines: list[str], *,
                   ephemeral: bool = False) -> None:
        for chunk in split_message("\n".join([header, *lines])):
            await interaction.followup.send(chunk, ephemeral=ephemeral, allowed_mentions=NO_PING)

    @bot.tree.command(name="記憶", description="看自己或某人的記憶；填關鍵字則用完整檢索流程搜尋")
    @app_commands.describe(對象="要看誰的記憶（不填就是自己）", 關鍵字="語意搜尋，例如：大熊 昨天 頸椎")
    async def memories(interaction: discord.Interaction, 對象: discord.Member | None = None,
                       關鍵字: str | None = None) -> None:
        await interaction.response.defer(thinking=True)  # 檢索 + LLM 重排可能超過 3 秒
        gid = guild_key(interaction.guild, interaction.user)
        text = (關鍵字 or "").strip()
        if not text:
            who = 對象 or interaction.user
            mems = store.list(gid, str(who.id))[:30]
            if not mems:
                await interaction.followup.send(f"我還沒有記住關於 {who.display_name} 的事喔！")
                return
            await send(interaction, f"**關於 {who.display_name} 的記憶（{len(mems)} 筆）**",
                       [f"`{m.id}` ({m.created_at[:10]}) {m.text}" for m in mems])
            return
        if bot.brain.agentic:
            # 封存區全文檢索（跟 Yuki 的 search_chat 同一套）
            tools = bot.brain.archive_tools(gid, bot.archive_guild(interaction.guild),
                                            bot.members_of(interaction.guild))
            hits = tools.search(text.split(), person=對象.display_name if 對象 else "")
            await send(interaction, f"**聊天紀錄裡的「{text}」**",
                       [h.line() for h in sorted(hits, key=lambda h: h.id)] or ["（找不到）"])
            return
        mentions = [person(對象)] if 對象 and not 對象.bot else []
        subs = await decompose(text, person(interaction.user), mentions, bot.members_of(interaction.guild),
                               wake_words=bot.settings.wake_words)
        groups = await bot.brain.retriever.search(gid, subs)
        lines = []
        for sq, hits in zip(subs, groups):
            label = sq.subject.name if sq.subject else "全體"
            when = f"（時間：{sq.time.phrase} {sq.time.start}～{sq.time.end}）" if sq.time else ""
            lines.append(f"__{label}：{sq.text}__{when}")
            for h in hits[:5]:
                score = f" ({h.score:.2f})" if h.score is not None else ""
                lines.append(f"`{h.memory.id}` ({h.memory.created_at[:10]}) {h.memory.text}{score}")
            if not hits:
                lines.append("（沒有相關記憶）")
        await send(interaction, f"**「{text}」的搜尋結果**", lines)

    @bot.tree.command(name="忘記", description="刪除一筆記憶（只能刪關於自己的，管理員除外）")
    @app_commands.describe(記憶id="記憶前面的 8 碼 ID，用 /記憶 查")
    async def forget(interaction: discord.Interaction, 記憶id: str) -> None:
        mem = store.get(記憶id.strip("` "))
        if mem is None or mem.guild_id != guild_key(interaction.guild, interaction.user):
            await interaction.response.send_message("找不到這筆記憶。", ephemeral=True)
            return
        uid = str(interaction.user.id)
        is_admin = interaction.guild is not None and interaction.user.guild_permissions.manage_guild
        if uid not in (mem.subject_id, mem.author_id) and not is_admin:
            await interaction.response.send_message("只能刪除關於自己的記憶喔（管理員除外）。", ephemeral=True)
            return
        store.delete(mem.id)
        await interaction.response.send_message(f"已忘記：{mem.text}", allowed_mentions=NO_PING)

    @bot.tree.command(name="檔案", description="看 Yuki 整理的人物檔案（只有你看得到）")
    @app_commands.describe(對象="要看誰的檔案（不填就是自己）")
    async def profile(interaction: discord.Interaction, 對象: discord.Member | None = None) -> None:
        who = 對象 or interaction.user
        text = bot.brain.profiles.read(guild_key(interaction.guild, interaction.user), str(who.id))
        if not text:
            await interaction.response.send_message(f"我還沒有整理 {who.display_name} 的檔案喔！", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        await send(interaction, text, [], ephemeral=True)

    @bot.tree.command(name="不要記我", description="刪除你在所有伺服器的人物檔案、記憶與封存的聊天紀錄，之後也不再整理你")
    async def opt_out(interaction: discord.Interaction) -> None:
        gid = guild_key(interaction.guild, interaction.user)
        uid = str(interaction.user.id)
        await interaction.response.defer(ephemeral=True)
        f = forget_everywhere(uid, gid, archive=bot.archive, sync=bot.sync, voice=bot.voice,
                              profile_db=ProfileDB(bot.settings.profile_db_path), profiles=bot.brain.profiles,
                              memories=bot.brain.store)
        await interaction.followup.send(
            f"好，所有伺服器裡關於你的資料都刪掉了：人物檔案 {f.profiles} 份、整理過的資訊 {f.facts} 條、"
            f"記憶 {f.memories} 筆、封存訊息 {f.messages} 則、語音紀錄 {f.voice} 筆。之後也不會再記你、整理你。",
            ephemeral=True)

    @bot.tree.command(name="來語音", description="叫 Yuki 進你所在的語音頻道")
    async def join_voice(interaction: discord.Interaction) -> None:
        if not bot.voice_allowed(interaction.guild):
            await interaction.response.send_message("這裡還沒開語音。", ephemeral=True)
            return
        state = getattr(interaction.user, "voice", None)
        if state is None or state.channel is None:
            await interaction.response.send_message("你先進語音頻道再叫我。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)  # 第一次要載入語音辨識模型
        try:
            await bot.start_voice(state.channel)
        except Exception as e:
            log.error("進語音失敗（語音辨識模型 %s）", bot.settings.voice_asr_model, exc_info=True)
            await interaction.followup.send(f"我進不了語音：{type(e).__name__}（詳情看 log）。", ephemeral=True)
            return
        listening = [m.display_name for uid in bot.settings.voice_listen_users
                     if (m := interaction.guild.get_member(int(uid)))]
        note = f"（目前只聽得到：{'、'.join(listening)}）" if listening else "（還沒設定要聽誰，VOICE_LISTEN_USERS 是空的）"
        await interaction.followup.send(f"來了～ #{state.channel.name} {note}", allowed_mentions=NO_PING)

    @bot.tree.command(name="離開語音", description="叫 Yuki 離開語音頻道")
    async def leave_voice(interaction: discord.Interaction) -> None:
        ok = interaction.guild is not None and await bot.end_voice(interaction.guild.id)
        await interaction.response.send_message("好，我先下去了～" if ok else "我本來就不在語音裡啊。")

    @bot.tree.command(name="白板", description="顯示群組近況白板")
    async def whiteboard(interaction: discord.Interaction) -> None:
        items = store.whiteboard(guild_key(interaction.guild, interaction.user))
        if not items:
            await interaction.response.send_message("白板目前是空的。")
            return
        await interaction.response.defer()
        await send(interaction, "**群組近況白板**", [f"- ({ts[:10]}) {t}" for _, t, ts in items])

    @bot.tree.error
    async def on_app_command_error(interaction: discord.Interaction,
                                   error: app_commands.AppCommandError) -> None:
        log.error("指令錯誤", exc_info=error)
        msg = "指令執行失敗了，請看 log。"
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)


def _install_excepthooks() -> None:
    """未捕捉的例外（主執行緒、其他執行緒）寫成 ERROR log，讓 AlertLogHandler 通知擁有者。"""
    prev_hook, prev_thread = sys.excepthook, threading.excepthook

    def hook(exc_type, exc, tb):
        logging.getLogger("yuki.uncaught").error("未捕捉的例外", exc_info=(exc_type, exc, tb))
        prev_hook(exc_type, exc, tb)

    def thread_hook(args):
        logging.getLogger("yuki.uncaught").error(
            "執行緒 %s 未捕捉的例外", getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        prev_thread(args)

    sys.excepthook, threading.excepthook = hook, thread_hook


def run() -> None:
    settings = load_settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not settings.discord_token:
        raise SystemExit("找不到 DISCORD_TOKEN，請檢查 .env")
    _install_excepthooks()
    YukiBot(settings).run(settings.discord_token, log_handler=None)
