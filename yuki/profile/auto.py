"""人物檔案自動增量更新：群組閒置時才跑，有人講話就停。

跟 `python -m scripts.build_profiles all` 做一樣的事（抽取新訊息 → 合併進 .md），差別在：
- 只在所有頻道閒置一段時間後才開始，每處理完一段／一個人就再檢查一次，有人講話就停下來，下次閒置再接著做。
- 新訊息累積不夠多就不跑（太少的資訊抽不出東西，白白佔 LLM）。
- 跟聊天共用同一個 LLM 排隊（semaphore），不會同時送兩個請求給 LM Studio。

最壞情況：剛好在整理一段（約 10～25 秒）時有人叫 Yuki，要等這段整理完才輪到回覆。
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Callable

from ..llm import LLMError
from .builder import ProfileBuilder
from .store import ProfileDB, ProfileFiles

log = logging.getLogger(__name__)


def new_message_count(archive: sqlite3.Connection, db: ProfileDB, guild_id: str) -> int:
    """封存區裡還沒被抽取過的訊息數（依每個頻道的抽取進度）。"""
    total = 0
    for (channel_id,) in archive.execute("SELECT DISTINCT channel_id FROM messages WHERE guild_id=?", (guild_id,)):
        after = db.last_id(channel_id) or 0
        total += archive.execute("SELECT COUNT(*) FROM messages WHERE guild_id=? AND channel_id=? AND id>?",
                                 (guild_id, channel_id, after)).fetchone()[0]
    return total


class ProfileUpdater:
    def __init__(self, make_builder: Callable[[str], ProfileBuilder], db: ProfileDB, files: ProfileFiles,
                 archive: sqlite3.Connection, is_idle: Callable[[], bool], opted_out: Callable[[str], set[str]], *,
                 min_new: int = 200, min_facts: int = 5, check_seconds: float = 60,
                 skip_guilds: set[str] = frozenset()):
        self.make_builder = make_builder
        self.db = db
        self.files = files
        self.archive = archive
        self.is_idle = is_idle
        self.opted_out = opted_out
        self.min_new = min_new
        self.min_facts = min_facts
        self.check_seconds = check_seconds
        self.skip_guilds = set(skip_guilds)  # 測試伺服器
        self._lock = asyncio.Lock()

    def guilds(self) -> list[str]:
        return [r[0] for r in self.archive.execute("SELECT DISTINCT guild_id FROM messages")
                if r[0] not in self.skip_guilds]

    def mergeable(self, guild_id: str) -> list[tuple[str, str]]:
        """有新資訊、而且夠格整理的人：(sid, name)。資訊太少又還沒有檔案的人不算（不然每分鐘都會被當成還沒合併）。"""
        return [(sid, name) for sid, name, total, pending in self.db.stats(guild_id)
                if pending and (total >= self.min_facts or self.files.read(guild_id, sid))]

    def has_unmerged(self, guild_id: str) -> bool:
        return bool(self.mergeable(guild_id))

    async def run_forever(self) -> None:
        while True:
            await asyncio.sleep(self.check_seconds)
            if not self.is_idle():
                continue
            try:
                for gid in self.guilds():
                    await self.run_once(gid)
            except Exception:
                log.exception("人物檔案自動更新失敗，下次閒置再試")

    async def run_once(self, guild_id: str) -> str:
        """回傳結果：busy（有人在講話，停下）、skip（新訊息不夠）、done（做完）、error。"""
        async with self._lock:
            gone = self.opted_out(guild_id)
            for uid in gone:  # 退出的人：資訊和檔案都清掉
                self.db.forget(guild_id, uid)
                self.files.delete(guild_id, uid)

            new = new_message_count(self.archive, self.db, guild_id)
            if new < self.min_new and not self.has_unmerged(guild_id):
                return "skip"
            builder = self.make_builder(guild_id)  # 每次重建：成員名單會變

            if new >= self.min_new:
                todo = [seg for cid, cname, _ in builder.channels() for seg in builder.pending_segments(cid, cname)]
                log.info("人物檔案：閒置中，開始抽取 %d 則新訊息（%d 段）", new, len(todo))
                for i, seg in enumerate(todo, 1):
                    if not self.is_idle():
                        log.info("人物檔案：有人在講話，抽取先停在 %d/%d 段", i - 1, len(todo))
                        return "busy"
                    try:
                        facts = await builder.extract_segment(seg)
                    except LLMError as e:
                        log.warning("人物檔案：#%s 抽取失敗，下次再試：%s", seg.channel_name, e)
                        return "error"  # 不能跳過這段：進度依頻道往後推，跳過就永遠不會處理
                    self.db.save_segment(guild_id, seg.channel_id, seg.last_id, facts)

            merged = 0
            for sid, name in self.mergeable(guild_id):
                if sid in gone:  # 別人講到退出的人，抽出來的資訊也不整理成檔案
                    self.db.forget(guild_id, sid)
                    continue
                if not self.is_idle():
                    log.info("人物檔案：有人在講話，整理先停在 %s 之前", name)
                    return "busy"
                await builder.merge_one(sid)
                merged += 1
            if new >= self.min_new or merged:
                log.info("人物檔案：更新完成（整理 %d 人）", merged)
            return "done"
