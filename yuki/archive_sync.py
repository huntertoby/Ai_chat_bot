"""把 Discord 公開頻道的訊息同步進封存區（data/archive.db）。

- scripts/archive_fetch.py 用 public_channels / fetch_channel 抓完整歷史。
- Yuki 上線時用 ArchiveSync.catch_up 補抓「上次關機到現在」的訊息，上線期間用 record 即時存。

「公開」= 某個身分組看得到的頻道（預設 @everyone），以及其中的公開討論串、論壇貼文。私人討論串一律不收。
機器人訊息、已退出封存的人不收。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

import discord

from .archive import ArchiveDB, ArchivedMessage

log = logging.getLogger(__name__)

_CHANNEL_TYPES = (discord.TextChannel, discord.VoiceChannel, discord.StageChannel)


async def fetch_channels(guild: discord.Guild) -> list:
    """guild.fetch_channels()，但跳過 discord.py 還不認得的頻道類型。

    Discord 推出新類型（例如 type 20）時，guild.fetch_channels() 會整個丟 InvalidData。
    """
    try:
        return list(await guild.fetch_channels())
    except discord.InvalidData:
        from discord.channel import _guild_channel_factory
        out = []
        for d in await guild._state.http.get_all_guild_channels(guild.id):
            factory, _ = _guild_channel_factory(d["type"])
            if factory is None:
                log.info("略過不認得的頻道類型 %s（%s）", d["type"], d.get("name"))
                continue
            out.append(factory(guild=guild, state=guild._state, data=d))
        return out


async def public_channels(guild: discord.Guild, me: discord.Member, role: discord.Role,
                          exclude: set[str], *, archived_since: datetime | None = None
                          ) -> list[tuple[object, str]]:
    """[(頻道或討論串, 所屬頻道名稱)]：role 看得到、Yuki 也讀得到、不在排除名單的。

    archived_since：只看這個時間之後才封存的討論串（之前封存的不會有新訊息）；None = 全部。
    """
    channels = await fetch_channels(guild)
    by_id = {c.id: c for c in channels}

    def public(ch) -> bool:
        return ch.name not in exclude and ch.permissions_for(role).view_channel

    def readable(ch) -> bool:
        p = ch.permissions_for(me)
        return p.view_channel and p.read_message_history

    out: list[tuple[object, str]] = []
    for ch in channels:
        if isinstance(ch, _CHANNEL_TYPES):
            if public(ch) and readable(ch):
                out.append((ch, ""))

    threads = {t.id: t for t in await guild.active_threads()}
    for ch in channels:
        if isinstance(ch, (discord.TextChannel, discord.ForumChannel)) and public(ch) and readable(ch):
            async for t in ch.archived_threads(limit=None):  # 只要公開的封存討論串，新封存的在前
                if archived_since and t.archive_timestamp < archived_since:
                    break
                threads[t.id] = t
    for t in threads.values():
        parent = by_id.get(t.parent_id)
        if t.type == discord.ChannelType.private_thread or parent is None:
            continue
        if public(parent) and readable(parent):
            out.append((t, parent.name))
    return out


def is_public(channel, role: discord.Role, exclude: set[str]) -> bool:
    """即時訊息用：這個頻道（或討論串所屬的頻道）是不是公開的。"""
    if isinstance(channel, discord.Thread):
        if channel.type == discord.ChannelType.private_thread or channel.parent is None:
            return False
        channel = channel.parent
    if not isinstance(channel, (*_CHANNEL_TYPES, discord.ForumChannel)):
        return False
    return channel.name not in exclude and channel.permissions_for(role).view_channel


def parent_name(channel) -> str:
    return channel.parent.name if isinstance(channel, discord.Thread) and channel.parent else ""


def to_row(m: discord.Message, guild_id: str, parent: str) -> ArchivedMessage:
    return ArchivedMessage(
        id=m.id, guild_id=guild_id, channel_id=str(m.channel.id), channel_name=m.channel.name,
        parent_name=parent, author_id=str(m.author.id), author_name=m.author.display_name,
        created_at=m.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        content=m.clean_content, reply_to=m.reference.message_id if m.reference else None,
        attachments=len(m.attachments))


async def fetch_channel(ch, parent: str, guild_id: str, db: ArchiveDB, skip: set[str],
                        stats: dict | None = None, *, before: int | None = None,
                        collect: list[ArchivedMessage] | None = None) -> None:
    """從這個頻道上次抓到的地方往後抓（沒抓過就從頭），before 之後的不抓。抓到的也放進 collect。"""
    stats = stats if stats is not None else {"seen": 0, "saved": 0, "t0": time.perf_counter()}
    after = db.last_id(str(ch.id))
    batch: list[ArchivedMessage] = []
    last = after

    def save() -> None:
        db.add_many(batch)
        if collect is not None:
            collect.extend(batch)
        stats["saved"] += len(batch)
        batch.clear()

    async for m in ch.history(limit=None, oldest_first=True,
                              after=discord.Object(after) if after else None,
                              before=discord.Object(before) if before else None):
        last = m.id
        stats["seen"] += 1
        if not m.author.bot and str(m.author.id) not in skip:
            batch.append(to_row(m, guild_id, parent))
        if len(batch) >= 1000:
            save()
            db.set_last_id(str(ch.id), last)
            db.commit()
            rate = stats["seen"] / (time.perf_counter() - stats["t0"])
            log.info("已存 %d 則（%.0f 則/秒），目前 #%s", stats["saved"], rate, ch.name)
    if batch:
        save()
    if last and last != after:
        db.set_last_id(str(ch.id), last)
    db.commit()


class ArchiveSync:
    """Yuki 上線時的封存同步。

    進度（fetch_progress.last_id）的規則：補抓期間，即時訊息只存不推進進度，
    等那個頻道補抓完才推進，這樣斷線期間的缺口一定會被補抓到。
    """

    def __init__(self, db: ArchiveDB, role_name: str = "", exclude: set[str] = frozenset(), *,
                 concurrency: int = 3, optout: bool = True):
        """optout=False：/不要記我 停用（D-28），不查退出名單，每個人都照常封存。"""
        self.db = db
        self.optout = optout
        self.role_name = role_name
        self.exclude = set(exclude)
        self.concurrency = concurrency
        self._skip: dict[str, set[str]] = {}
        self._busy: set[str] = set()                 # 正在補抓的伺服器
        self._done: dict[str, set[int]] = {}         # 這次補抓已完成的頻道
        self._failed: dict[str, set[int]] = {}       # 這次補抓失敗的頻道（進度不推進，下次再補）
        self._lock = asyncio.Lock()

    def role(self, guild: discord.Guild) -> discord.Role:
        if self.role_name:
            role = discord.utils.get(guild.roles, name=self.role_name)
            if role is not None:
                return role
            log.warning("找不到身分組「%s」，改用 @everyone 判斷公開頻道", self.role_name)
        return guild.default_role

    def skip(self, guild_id: str) -> set[str]:
        if not self.optout:
            return set()
        if guild_id not in self._skip:
            self._skip[guild_id] = self.db.opted_out(guild_id)
        return self._skip[guild_id]

    def forget_skip(self, guild_id: str | None = None) -> None:
        """退出名單變了：不給 guild_id 就全部重讀（退出以人為單位，每個伺服器都要更新）。"""
        if guild_id is None:
            self._skip.clear()
        else:
            self._skip.pop(guild_id, None)

    def _may_advance(self, guild_id: str, channel_id: int) -> bool:
        if channel_id in self._failed.get(guild_id, ()):
            return False
        return guild_id not in self._busy or channel_id in self._done.get(guild_id, ())

    def record(self, message: discord.Message) -> bool:
        """上線期間收到的訊息：公開頻道的就存進封存區。"""
        guild = message.guild
        if guild is None or message.author.bot:
            return False
        gid = str(guild.id)
        if not is_public(message.channel, self.role(guild), self.exclude):
            return False
        if str(message.author.id) in self.skip(gid):
            return False
        self.db.add_many([to_row(message, gid, parent_name(message.channel))])
        if self._may_advance(gid, message.channel.id):
            self.db.advance_last_id(str(message.channel.id), message.id)
        self.db.commit()
        return True

    async def catch_up(self, guild: discord.Guild) -> dict[str, list[ArchivedMessage]]:
        """補抓上次關機到現在的訊息。回傳 {頻道 ID: 補抓到的訊息（舊到新）}。"""
        async with self._lock:
            gid = str(guild.id)
            latest = self.db.latest_id(gid)
            if latest is None:
                log.info("%s：封存區還沒有這個伺服器的資料，不回溯歷史（要的話跑 scripts.archive_fetch），"
                         "從現在開始記", guild.name)
                return {}
            self._busy.add(gid)
            done = self._done[gid] = set()
            failed = self._failed[gid] = set()
            try:
                return await self._catch_up(guild, gid, latest, done, failed)
            finally:
                self._busy.discard(gid)

    async def _catch_up(self, guild: discord.Guild, gid: str, latest: int, done: set[int],
                        failed: set[int]) -> dict[str, list[ArchivedMessage]]:
        t0 = time.perf_counter()
        since = discord.utils.snowflake_time(latest)
        boundary = discord.utils.time_snowflake(discord.utils.utcnow())
        targets = await public_channels(guild, guild.me, self.role(guild), self.exclude,
                                        archived_since=since)
        skip = self.skip(gid)
        stats = {"seen": 0, "saved": 0, "t0": t0}
        new: dict[str, list[ArchivedMessage]] = {}
        sem = asyncio.Semaphore(self.concurrency)

        async def one(ch, parent: str) -> None:
            cid = str(ch.id)
            if self.db.last_id(cid) is None and ch.id < latest:
                # 早就存在、但從沒抓過的頻道：不在啟動時回溯全部歷史，從現在開始記
                self.db.advance_last_id(cid, boundary)
                done.add(ch.id)
                return
            rows: list[ArchivedMessage] = []
            async with sem:
                try:
                    await fetch_channel(ch, parent, gid, self.db, skip, stats, before=boundary, collect=rows)
                except discord.Forbidden:
                    log.warning("#%s：403，略過", ch.name)
                    failed.add(ch.id)
                    return
                except discord.HTTPException as e:
                    log.warning("#%s 補抓失敗：%s（下次啟動會再補）", ch.name, e)
                    failed.add(ch.id)
                    return
            # before=boundary 之前的都抓完了；補抓期間即時存進來的訊息也算進度
            self.db.advance_last_id(cid, max(boundary, self.db.max_id(cid) or 0))
            self.db.commit()
            done.add(ch.id)
            if rows:
                new[cid] = rows

        await asyncio.gather(*(one(ch, p) for ch, p in targets))
        log.info("%s：補抓完成，%d 個頻道／討論串、新存 %d 則（看過 %d 則，%.0f 秒）%s",
                 guild.name, len(targets), stats["saved"], stats["seen"], time.perf_counter() - t0,
                 f"，失敗 {len(failed)} 個" if failed else "")
        return new
