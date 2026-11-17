"""人物檔案的兩個階段：

1. extract：封存區的聊天 → 段落 → LLM 抽出一條條資訊（分 fact / status / opinion / relation / meme / joke）。
2. merge：每個人尚未整理的資訊 → 跟目前的 .md 一起交給 LLM → 更新後的 .md。

兩階段都可以中斷後重跑，只會處理新的東西。
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from datetime import date

from ..llm import LLMError, LMStudioClient
from . import prompts
from .chunk import Segment, segments, to_lines
from .store import KINDS, Fact, ProfileDB, ProfileFiles

log = logging.getLogger(__name__)
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def member_names(archive: sqlite3.Connection, guild_id: str) -> dict[str, str]:
    """{author_id: 最新的名字}，依發言數由多到少。"""
    rows = archive.execute(
        "SELECT author_id, COUNT(*) FROM messages WHERE guild_id=? GROUP BY author_id ORDER BY 2 DESC",
        (guild_id,)).fetchall()
    names = {}
    for author_id, _ in rows:
        names[author_id] = archive.execute(
            "SELECT author_name FROM messages WHERE guild_id=? AND author_id=? ORDER BY id DESC LIMIT 1",
            (guild_id, author_id)).fetchone()[0]
    return names


def resolve(name: str, by_name: dict[str, str]) -> str | None:
    key = (name or "").strip().lstrip("@＠").lower()
    return by_name.get(key)


def parse_items(data: dict | None, seg: Segment, by_name: dict[str, str],
                names: dict[str, str]) -> list[Fact]:
    facts: list[Fact] = []
    for item in (data or {}).get("items") or []:
        if not isinstance(item, dict):
            continue
        sid = resolve(str(item.get("subject", "")), by_name)
        text = str(item.get("text") or "").strip()
        kind = item.get("kind")
        if sid is None or not text or kind not in KINDS:
            continue
        day = str(item.get("day") or "").strip()
        if not _DAY.match(day) or not (seg.first_day <= day <= seg.last_day):
            day = seg.last_day  # 模型亂填日期就用段落日期
        conf = "high" if item.get("confidence") == "high" else "low"
        facts.append(Fact(sid, names[sid], kind, text[:200], conf, day))
    return facts


class ProfileBuilder:
    def __init__(self, archive: sqlite3.Connection, db: ProfileDB, files: ProfileFiles, llm: LMStudioClient,
                 guild_id: str, *, segment_chars: int = 9000, merge_batch_chars: int = 9000,
                 profile_limit: int = 2500, max_tokens: int = 8192):
        self.archive = archive
        self.db = db
        self.files = files
        self.llm = llm
        self.guild_id = guild_id
        self.segment_chars = segment_chars
        self.merge_batch_chars = merge_batch_chars
        self.profile_limit = profile_limit
        self.max_tokens = max_tokens
        self.names = member_names(archive, guild_id)
        self.by_name = {n.lower(): i for i, n in self.names.items()}

    # ---------- 抽取 ----------
    def channels(self) -> list[tuple[str, str, int]]:
        return self.archive.execute(
            "SELECT channel_id, MAX(channel_name), COUNT(*) FROM messages WHERE guild_id=?"
            " GROUP BY channel_id ORDER BY 3 DESC", (self.guild_id,)).fetchall()

    def pending_segments(self, channel_id: str, channel_name: str) -> list[Segment]:
        after = self.db.last_id(channel_id) or 0
        rows = self.archive.execute(
            "SELECT id, author_id, author_name, created_at, content FROM messages"
            " WHERE guild_id=? AND channel_id=? AND id>? ORDER BY id", (self.guild_id, channel_id, after)).fetchall()
        return segments(to_lines(rows), channel_id, channel_name, max_chars=self.segment_chars)

    async def extract_segment(self, seg: Segment) -> list[Fact]:
        speakers = seg.speakers()
        others = [n for i, n in list(self.names.items())[:40] if i not in speakers]
        messages = [
            {"role": "system", "content": prompts.EXTRACT_SYSTEM},
            {"role": "user", "content": prompts.extract_user(seg.channel_name, list(speakers.values()), others,
                                                             seg.render())},
        ]
        data = await self.llm.chat_json(messages, prompts.EXTRACT_SCHEMA, name="profile_items",
                                        max_tokens=self.max_tokens)
        if data is None:
            # 沒拿到 JSON（多半是推理把 token 用完）不能當成「這段沒東西」，不然進度往前推就再也不會處理
            raise LLMError("沒有拿到 JSON 結果")
        return parse_items(data, seg, self.by_name, self.names)

    async def extract(self, *, limit: int | None = None, channel: str | None = None) -> int:
        """回傳這次處理的段落數。limit：最多處理幾段（試跑用）。"""
        todo = []
        for cid, cname, _ in self.channels():
            if channel and channel not in (cid, cname):
                continue
            todo += self.pending_segments(cid, cname)
        if limit is not None:
            todo = todo[:limit]
        total = len(todo)
        log.info("待抽取 %d 段（約 %d 字）", total, sum(s.chars() for s in todo))
        t0, done = time.perf_counter(), 0
        for seg in todo:
            try:
                facts = await self.extract_segment(seg)
            except LLMError as e:
                log.warning("#%s %s 抽取失敗，下次重跑會再試：%s", seg.channel_name, seg.first_day, e)
                break  # 不能跳過：進度是依頻道往後推的，跳過這段就永遠不會再處理
            self.db.save_segment(self.guild_id, seg.channel_id, seg.last_id, facts)
            done += 1
            eta = (time.perf_counter() - t0) / done * (total - done) / 60
            kinds = ",".join(sorted({f.kind for f in facts}))
            log.info("[%d/%d] #%s %s～%s → %d 條 %s（剩約 %.0f 分）", done, total, seg.channel_name,
                     seg.first_day, seg.last_day, len(facts), kinds, eta)
        return done

    # ---------- 整理 ----------
    def _batches(self, pending: list[tuple[int, Fact]]) -> list[list[tuple[int, Fact]]]:
        out, cur, size = [], [], 0
        for item in pending:
            n = len(item[1].line()) + 1
            if cur and size + n > self.merge_batch_chars:
                out.append(cur)
                cur, size = [], 0
            cur.append(item)
            size += n
        if cur:
            out.append(cur)
        return out

    async def merge_one(self, subject_id: str) -> bool:
        pending = self.db.pending(self.guild_id, subject_id)
        if not pending:
            return False
        name = self.names.get(subject_id) or pending[-1][1].subject_name
        current = self._body(self.files.read(self.guild_id, subject_id))
        last_day = ""
        for batch in self._batches(pending):
            messages = [
                {"role": "system", "content": prompts.MERGE_SYSTEM.format(name=name, limit=self.profile_limit)},
                {"role": "user", "content": prompts.merge_user(name, current, [f.line() for _, f in batch])},
            ]
            try:
                out = await self.llm.chat(messages, temperature=0.3, max_tokens=self.max_tokens)
            except LLMError as e:
                log.warning("%s 整理失敗：%s", name, e)
                return False
            body = self._body(out)
            if "## 基本資料" not in body:
                log.warning("%s 整理結果格式不對，略過這批：%s", name, out[:200])
                return False
            current, last_day = body, batch[-1][1].day
            self.files.write(self.guild_id, subject_id, self._header(name, last_day) + current)
            self.db.mark_merged([i for i, _ in batch])
            log.info("%s：整理了 %d 條（資料到 %s）", name, len(batch), last_day)
        return True

    async def merge(self, *, min_facts: int = 5, only: str | None = None) -> int:
        n = 0
        for sid, name, total, pending in self.db.stats(self.guild_id):
            if only and only not in (sid, name, self.names.get(sid)):
                continue
            if not pending or (total < min_facts and not self.files.read(self.guild_id, sid)):
                continue
            if await self.merge_one(sid):
                n += 1
        return n

    @staticmethod
    def _body(text: str) -> str:
        """去掉標題與說明行，從「## 基本資料」開始。"""
        i = text.find("## 基本資料")
        return text[i:].strip() if i >= 0 else text.strip()

    @staticmethod
    def _header(name: str, last_day: str) -> str:
        return f"# {name}\n> 自動整理（{date.today().isoformat()} 更新，資料到 {last_day}）。玩笑只放在「群組梗與綽號」。\n\n"
