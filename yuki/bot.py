"""Discord 端：判斷要不要回、注意力機制、指令。實際的思考都交給 Brain。"""
from __future__ import annotations

import asyncio
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

from . import alert, trace, vision
from .archive import ArchiveDB
from .archive_sync import ArchiveSync, is_public
from .brain import Brain
from .config import ROOT, Settings, load_settings
from .llm import LLMError, LMStudioClient
from .memory.text_index import add_words
from .memory.writer import ChatLine
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
        self.brain = Brain.build(settings)
        self.active_until: dict[int, float] = {}
        self._members: dict[int, list[Person]] = {}
        self._wake = re.compile("|".join(re.escape(w) for w in settings.wake_words), re.IGNORECASE)
        self._synced = False
        self.archive = ArchiveDB(ARCHIVE_PATH)
        self.sync = ArchiveSync(self.archive, settings.archive_role, set(settings.archive_exclude))
        trace.configure(settings.trace_dir, max_days=settings.trace_max_days, max_mb=settings.trace_max_mb,
                        opted_out=self._trace_opted_out)
        self.voice = VoicePresence(self.archive.db)
        self.brain.archive_db = self.archive.db  # agentic 檢索（開發規格 6.4）
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

    async def _dm_owner(self, user_id: str, text: str) -> None:
        user = await self.fetch_user(int(user_id))
        await user.send(text, allowed_mentions=NO_PING)

    async def on_error(self, event_method: str, /, *args, **kwargs) -> None:
        # discord.py 預設只印到 stderr；改成 log，擁有者才收得到通知
        log.exception("事件 %s 發生未捕捉的例外", event_method)

    async def close(self) -> None:
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

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        self._last_activity = time.monotonic()
        test = self.is_test_guild(message.guild)
        if not test:
            try:
                self.sync.record(message)
            except Exception:
                log.exception("訊息存進封存區失敗")
        text = self.clean(message)
        gid, cid = guild_key(message.guild, message.author), message.channel.id
        channel_key = str(cid)
        author = person(message.author)
        mentions = [person(u) for u in message.mentions if u != self.user and not u.bot]
        now = time.monotonic()
        active = self.active_until.get(cid, 0) > now
        called = self.is_called(message)

        if active and any(w in message.content for w in self.settings.exit_words):
            self.active_until.pop(cid, None)
            await message.reply("好的，我先去旁邊休息囉！有事再叫我～", mention_author=False)
            return

        images = vision.image_urls(message) if self.settings.vision_enabled else []
        if images and not text:
            text = "（貼了圖片）"  # 只有圖片沒有文字：讓這則進對話紀錄，圖才帶得到
        if text:
            line = ChatLine(author, text, images=images, msg_id=str(message.id))
            self.brain.observe(channel_key, line)
            if called or active or (self.settings.passive_enabled and not test):
                self.brain.buffer(gid, channel_key, line, mentions)

        if not (called or active):
            return
        if not text:
            await message.reply("找我有事嗎？", mention_author=False)
            return

        self.active_until[cid] = now + self.settings.attention_timeout
        self._replying += 1
        tr = trace.start("reply", guild=message.guild.id if message.guild else "", channel=cid,
                         trigger="called" if called else "active",
                         user_ids=[author.id, *(m.id for m in mentions)])
        try:
            async with tr:
                async with message.channel.typing():
                    voice = self.voice.describe(str(message.guild.id), datetime.now()) if message.guild else ""
                    reply = await self.brain.reply(gid, channel_key, author, text, mentions=mentions,
                                                   members=self.members_of(message.guild), voice=voice,
                                                   owner=self.owner_name(message.guild),
                                                   archive_guild=self.archive_guild(message.guild))
                tr.final = reply
        except LLMError as e:
            # 用 warning：ERROR 會被 AlertLogHandler 當成另一種錯誤再通知一次
            log.warning("回覆失敗 trace_id=%s", tr.trace_id, exc_info=True)
            self.alerter.notify("lmstudio", "LM Studio 連線", "回覆時連不上 LM Studio", exc=e,
                                trace_id=tr.trace_id)
            await message.reply("我跟 LM Studio 的連線好像斷了…等等再試一次？", mention_author=False)
            return
        finally:
            self._replying -= 1
        if not reply:
            reply = "……我腦袋突然一片空白，再說一次？"

        for i, chunk in enumerate(split_message(reply)):
            if i == 0:
                await message.reply(chunk, mention_author=False)
            else:
                await message.channel.send(chunk)
        bot_line = ChatLine(person(self.user), reply, is_bot=True)
        self.brain.observe(channel_key, bot_line)
        self.brain.buffer(gid, channel_key, bot_line)


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

    @bot.tree.command(name="不要記我", description="刪除你的人物檔案與封存的聊天紀錄，之後也不再整理你")
    async def opt_out(interaction: discord.Interaction) -> None:
        gid = guild_key(interaction.guild, interaction.user)
        uid = str(interaction.user.id)
        await interaction.response.defer(ephemeral=True)
        n = bot.archive.opt_out(gid, uid)
        bot.sync.forget_skip(gid)
        bot.voice.forget_user(gid, uid)
        facts = ProfileDB(bot.settings.profile_db_path).forget(gid, uid)
        had = bot.brain.profiles.delete(gid, uid)
        await interaction.followup.send(
            f"好，已經刪掉你的檔案{'（有）' if had else '（本來就沒有）'}、{facts} 條整理過的資訊、"
            f"{n} 則封存訊息，之後也不會再整理你。（對話中的事實記憶請用 /記憶 和 /忘記 管理）",
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
