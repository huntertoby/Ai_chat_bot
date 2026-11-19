"""Discord 端：判斷要不要回、注意力機制、指令。實際的思考都交給 Brain。"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import threading
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
from .chat import ChatLine, Person, ReplyRef
from .optout import forget_everywhere
from .profile.auto import ProfileUpdater
from .profile.builder import ProfileBuilder
from .profile.store import ProfileDB
from .scheduler import ReplyScheduler, Turn, decide_options
from .text_index import add_words
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


def wake_pattern(words: list[str]) -> re.Pattern:
    """喚醒詞：英數字的詞要整個字相符（前後不是英文字母或數字），中文照字比對；不分大小寫。"""
    parts = []
    for w in words:
        if not w:
            continue
        if w.isascii() and any(c.isalnum() for c in w):
            parts.append(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])")
        else:
            parts.append(re.escape(w))
    return re.compile("|".join(parts) or r"(?!x)x", re.IGNORECASE)


_NOT_WORD = re.compile(r"[\W_]+")  # 標點、空白、表情符號


def is_exit(text: str, exit_words: list[str]) -> bool:
    """叫她離開：整句（去掉喚醒詞、標點、表情後）就是結束字之一才算；句子裡夾著結束字不算。"""
    core = _NOT_WORD.sub("", text or "").lower()
    return bool(core) and core in {_NOT_WORD.sub("", w).lower() for w in exit_words}


EXIT_REPLY = "好的，我先去旁邊休息囉！有事再叫我～"


class YukiBot(commands.Bot):
    def __init__(self, settings: Settings, *, archive_path=ARCHIVE_PATH, emoji_sets=None):
        """archive_path、emoji_sets：測試時換成暫存的封存區、指定的表情圖鑑（預設讀 data/）。"""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        # 只用斜線指令（bot.tree）；commands.Bot 一定要給前綴，給「@她」等於沒有文字指令
        super().__init__(command_prefix=commands.when_mentioned, intents=intents,
                         allowed_mentions=NO_PING)
        self.settings = settings
        # 維運（開發規格 6.12）：出錯私訊擁有者、AI 追蹤紀錄
        self.alerter = alert.Alerter(settings.owner_user_ids, dedupe_min=settings.alert_dedupe_min,
                                     max_per_hour=settings.alert_max_per_hour)
        alert.set_default(self.alerter)
        logging.getLogger().addHandler(alert.AlertLogHandler(self.alerter))
        self._health_task: asyncio.Task | None = None
        # 只有 reviewed 的圖鑑（C1-2）
        self.emoji_sets = emoji_catalog.load_all() if emoji_sets is None else emoji_sets
        self.emoji_pacing = emoji_catalog.EmojiPacing()
        self.brain = Brain.build(settings)
        self._members: dict[int, list[Person]] = {}
        self._wake = wake_pattern(settings.wake_words)
        self._synced = False
        # 每頻道的回覆排程：節流（B3）、熱鬧模式、排隊合併、插話去抖動與硬限制（B2）、注意時間
        self.replies = ReplyScheduler(settings, self._respond)
        self.archive = ArchiveDB(archive_path)
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
        await self.replies.close()  # 等待中的插話判斷、熱鬧模式的合併回覆
        for task in list(self._catchup_tasks):
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
        self.archive.db.close()

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
        # 跟聊天共用同一個排隊（LM Studio 不會同時收到兩個請求），但是背景優先度：
        # 有聊天在等就不送，正在跑的也會被聊天中斷讓位（llm.LLMSlots），所以單次逾時可以放長
        self._profile_llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, timeout=max(s.llm_timeout, 600),
                                           reasoning_effort=s.profile_reasoning_effort,
                                           json_reasoning_effort=s.profile_reasoning_effort,
                                           slots=self.brain.llm._slots, background=True)
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
            log.info("%s：補抓到 %d 則（%d 個頻道），補進對話紀錄", guild.name, total, len(new))
            for channel_id, rows in new.items():
                self.brain.catch_up(channel_id, [ChatLine(Person(r.author_id, r.author_name), r.content)
                                                 for r in rows if r.content.strip()])

    async def on_guild_join(self, guild: discord.Guild) -> None:
        await self.sync_commands(guild)

    # ---------- 成員名單（查聊天紀錄時把名字對到人、上網時認出退出的人）----------
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
            # 封存區裡講過話、但已不在伺服器的人（退出的成員、匯入的資料）也要認得出名字
            known = {p.id for p in people}
            people += [Person(uid, name) for uid, name in self.archive.authors(self.archive_guild(guild))
                       if uid not in known and name]
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
            log.info("大量重複的訊息，不回、不進對話紀錄（封存照常）")
            return
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return  # 加入、置頂、加成之類的系統訊息
        if throttle.is_command(message.content):
            return
        text = self.clean(message)
        cid = message.channel.id
        channel_key = str(cid)
        author = person(message.author)
        mentions = [person(u) for u in message.mentions if u != self.user and not u.bot]
        active, chiming = self.replies.attention(cid)  # chiming：她剛自己插過話，大家可能在回她
        called = self.is_called(message)

        # 對她說「閉嘴」「沒問你」：這個頻道一陣子不主動插話（被叫仍回）
        if (called or active or chiming) and chime.is_shush(message.content):
            self.replies.mute(cid)
            log.info("被叫閉嘴，頻道 %s 在 %.0f 分鐘內不主動插話", cid, s.chime_mute_minutes)

        # 叫她走：只看跟她講的話（叫她、回覆她），整句就是結束字才算
        leaving = called and (active or chiming) and is_exit(text, s.exit_words)

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
            self.brain.observe(channel_key, line)

        if leaving:  # 那句已經進對話紀錄了，她的道別也記進去
            self.replies.end_attention(cid)
            await message.reply(EXIT_REPLY, mention_author=False)
            self.brain.observe(channel_key, ChatLine(Person(str(self.user.id), s.bot_name), EXIT_REPLY, is_bot=True))
            return

        if called:
            self.replies.cancel_debounce(cid)  # 有人叫她，不用再判斷要不要插話
            await self._on_called(message, author, text, mentions)
            return
        if message.guild is not None and not self.chime_allowed(message.guild):
            # 還沒開插話的伺服器（預設＝測試伺服器以外，例如主伺服器）：注意模式內接著回，
            # 但跟被叫一樣走熱鬧模式（頻道很吵時收集起來合併回），而且這種回覆不會再延長注意時間
            if (active and line is not None and not throttle.is_noise_text(text, s.noise_max_chars)
                    and not self.replies.throttle.tag_cooling(author.id)):  # 洗版冷卻中的人，注意模式也不回
                await self._on_called(message, author, text, mentions, count_tag=False, trigger="attention")
            return
        # 沒被叫到：先過雜訊過濾，再排進插話判斷（注意模式只是多一個提示，不是必回）
        if (line is None or not self.replies.limits.enabled or message.guild is None
                or throttle.is_noise_text(text, s.noise_max_chars)
                or self.replies.throttle.tag_cooling(author.id)):
            return
        attention = active or chiming
        self.replies.schedule_chime(cid, lambda: self._chime(message, line, attention))

    # ---------- 被叫到：交給排程器（節流、熱鬧模式、排隊合併）----------
    async def _on_called(self, message: discord.Message, author: Person, text: str, mentions: list[Person],
                         *, count_tag: bool = True, trigger: str = "called") -> None:
        """count_tag：算不算這個人「叫她」的次數（單人洗版限制）。注意模式接話不算，但頻道熱鬧模式照算。"""
        cid = message.channel.id
        admitted = self.replies.admit_call(cid, author.id, count_tag=count_tag)
        if admitted is None:
            return
        if not text:
            await message.reply("找我有事嗎？", mention_author=False)
            return
        await self.replies.call(cid, Turn(message, author, text, mentions, trigger=trigger), admitted)

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
        """真的回一則（排程器佔住頻道之後才會呼叫）。回傳有沒有送出訊息。"""
        message, author, mentions = turn.message, turn.author, turn.mentions
        gid, cid = guild_key(message.guild, message.author), message.channel.id
        channel_key = str(cid)
        if turn.trigger == "called":  # 只有真的被叫才延長注意時間（注意模式接話、插話不算）
            self.replies.extend_attention(cid)
        self._replying += 1
        try:  # 計數一定要減回來：中間任何一步丟例外都不能讓 _replying 卡在 > 0
            tr = trace.start("reply", guild=message.guild.id if message.guild else "", channel=cid,
                             trigger=turn.trigger, user_ids=[author.id, *(m.id for m in mentions)])
            self._hide_optouts(tr, channel_key, self._replied_person(message))
            eset = self.emoji_set(message.guild)
            emoji_ok = eset is not None and self.emoji_pacing.allow(cid)  # 剛用過就這則不放清單
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
        # 群組表情：`:名稱:` 換成 <:名稱:ID>、壞名稱刪掉、一則最多一個；貼圖另外附。
        # 沒有圖鑑時只清掉 `:名稱:` 和 [貼圖:…]，模型照抄的 <:名稱:ID> 原樣留著
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
        if turn.chimed:
            self.replies.note_chimed(cid)
        bot_line = ChatLine(person(self.user), reply or "[貼圖]", is_bot=True)
        self.brain.observe(channel_key, bot_line)
        return True

    # ---------- 沒被叫到：插話判斷（B2）----------
    def _llm_busy(self) -> bool:
        return self._replying > 0 or chime.llm_is_busy(self.brain.llm)

    async def _chime(self, message: discord.Message, line: ChatLine, attention: bool) -> Turn | None:
        """去抖動之後判斷一次（排程器呼叫）。決定要回覆就回傳 Turn（排程器已經佔住這個頻道，接著回覆）；
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

            d = None
            if reason := self.replies.chime_gate(cid, attention=attention, llm_busy=self._llm_busy()):
                skip(reason)
            else:
                d, images, eset = await self._decide(message, line, attention)
                if d is None:  # 準備資料的這段時間，LM Studio 被別人佔住了
                    skip("llm_busy")
            if d is not None:
                tr.lat("decide", d.ms)
                info.update(decision=d.to_dict(), threshold=self.replies.limits.threshold,
                            prob=round(d.participate, 3), prob_source=d.prob_source, images=len(images))
                plan = self.replies.plan_chime(d, cid, uid, attention=attention)
                info["action"] = "none" if plan == "error" else plan
                if plan == "error":
                    tr.degrade("chime_failed")
                elif plan.startswith("skipped:"):
                    skip(plan.split(":", 1)[1])
                elif plan == "reply":
                    mentions = [person(u) for u in message.mentions if u != self.user and not u.bot]
                    turn = Turn(message, person(message.author), line.text, mentions,
                                trigger="chime_attention" if attention else "chime",
                                background=d.need_background, chimed=True)
                elif plan == "react":
                    if (why := await self._react(message, d, eset)) is not None:
                        skip(why)
            tr.final = json.dumps(info, ensure_ascii=False)
        return turn

    async def _decide(self, message: discord.Message, line: ChatLine, attention: bool):
        """準備對話紀錄和圖片、問模型。回傳 (Decision, 圖片, 表情圖鑑)；準備完 LM Studio 已經被佔住就回 (None, …)。"""
        s = self.settings
        hist = list(self.brain.history[str(message.channel.id)])
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
        eset = self.emoji_set(message.guild)  # 有 reviewed 的圖鑑才放表情清單、才限制只能按圖鑑裡的
        if self._llm_busy():  # 補抓回覆對象、下載圖片的這段時間，LM Studio 可能已經被別人佔住：再檢查一次
            return None, images, eset
        idx = next((i for i, l in enumerate(hist) if l is line), len(hist))
        d = await chime.decide(self.brain.llm, recent=hist[:idx], latest=line, bot_name=s.bot_name,
                               images=images, attention=attention, **decide_options(s),
                               emoji_block=eset.react_block() if eset and s.chime_emoji_list else "")
        return d, images, eset

    async def _react(self, message: discord.Message, d, eset) -> str | None:
        """按表情。回傳沒按成的原因，None＝按好了。"""
        wanted = d.emoji
        if eset is not None:  # 名稱（含別名）換成真名；圖鑑裡沒有的群組表情不按（回空字串）
            _, wanted = eset.react_name(d.emoji)
        emoji = chime.resolve_emoji(wanted, message.guild.emojis if message.guild else ())
        if emoji is None:
            return "emoji_not_found"
        try:
            await message.add_reaction(emoji)
        except discord.HTTPException:
            log.info("按表情失敗：%s", d.emoji, exc_info=True)
            return "react_failed"
        self.replies.record_react(message.channel.id, str(message.author.id), d.kind)
        return None


def register_commands(bot: YukiBot) -> None:
    """斜線指令。只註冊到 bot.tree，同步在 on_ready 時對每個伺服器做（伺服器指令立即生效）。"""
    async def send(interaction: discord.Interaction, header: str, lines: list[str], *,
                   ephemeral: bool = False) -> None:
        for chunk in split_message("\n".join([header, *lines])):
            await interaction.followup.send(chunk, ephemeral=ephemeral, allowed_mentions=NO_PING)

    @bot.tree.command(name="記憶", description="Yuki 在聊天紀錄裡找得到的：某人最近說的話，或含關鍵字的訊息（只有你看得到）")
    @app_commands.describe(對象="看誰說的（不填：有關鍵字就不限人，沒有關鍵字就是自己）",
                           關鍵字="用空白分開，任一個符合就算")
    async def memories(interaction: discord.Interaction, 對象: discord.Member | None = None,
                       關鍵字: str | None = None) -> None:
        # 只有本人看得到：結果可能含別人的事、或別的伺服器（測試伺服器借主伺服器）的聊天紀錄
        await interaction.response.defer(thinking=True, ephemeral=True)
        gid = guild_key(interaction.guild, interaction.user)
        # 跟 Yuki 的 search_chat 同一套（封存區全文檢索，/不要記我 的人查不到）
        tools = bot.brain.archive_tools(gid, bot.archive_guild(interaction.guild), bot.members_of(interaction.guild))
        text = (關鍵字 or "").strip()
        if text:
            hits = tools.search(text.split(), person=對象.display_name if 對象 else "")
            await send(interaction, f"**聊天紀錄裡的「{text}」**",
                       [h.line() for h in sorted(hits, key=lambda h: h.id)] or ["（找不到）"], ephemeral=True)
            return
        who = 對象 or interaction.user
        hits = tools.recent(str(who.id))
        if not hits:
            await interaction.followup.send(f"聊天紀錄裡還沒有 {who.display_name} 說的話喔！", ephemeral=True,
                                            allowed_mentions=NO_PING)
            return
        await send(interaction, f"**{who.display_name} 最近說的話（{len(hits)} 則）**", [h.line() for h in hits],
                   ephemeral=True)

    # 舊版 RAG 的事實記憶和白板：D-04 之後就不再更新、也不再給 Yuki 看，R1 移除了程式。
    # 指令先留著回一句說明（不默默壞掉），要不要拿掉等使用者決定（R1 報告）。
    retired = "這個功能已經停用了：Yuki 現在直接查聊天紀錄，不再另外記一條條的記憶或白板。想看她查得到什麼，用 /記憶。"

    @bot.tree.command(name="忘記", description="（已停用）舊版的記憶刪除")
    @app_commands.describe(記憶id="舊版記憶的 ID")
    async def forget(interaction: discord.Interaction, 記憶id: str) -> None:
        await interaction.response.send_message(retired, ephemeral=True)

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
        profile_db = ProfileDB(bot.settings.profile_db_path)
        try:
            f = forget_everywhere(uid, gid, archive=bot.archive, sync=bot.sync, voice=bot.voice,
                                  profile_db=profile_db, profiles=bot.brain.profiles,
                                  legacy_db=bot.settings.db_path)
        finally:
            profile_db.db.close()
        await interaction.followup.send(
            f"好，所有伺服器裡關於你的資料都刪掉了：人物檔案 {f.profiles} 份、整理過的資訊 {f.facts} 條、"
            f"舊的記憶 {f.memories} 筆、封存訊息 {f.messages} 則、語音紀錄 {f.voice} 筆。之後也不會再記你、整理你。",
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

    @bot.tree.command(name="白板", description="（已停用）舊版的群組近況白板")
    async def whiteboard(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(retired, ephemeral=True)

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
