"""人物檔案的兩個階段：

1. extract：封存區的聊天 → 段落 → LLM 抽出一條條資訊（分 fact / status / opinion / relation / meme / joke）。
2. merge：每個人尚未整理的資訊 → 跟目前的 .md 一起交給 LLM → 更新後的 .md。

兩階段都可以中斷後重跑，只會處理新的東西。
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import date, datetime

from .. import trace
from ..llm import LLMError, LMStudioClient
from . import prompts, revise
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


_PUNCT = re.compile(r"[\s\W_]+")


def _norm(text: str) -> str:
    return _PUNCT.sub("", text or "").lower()


def said_by(seg: Segment, author_id: str, quote: str) -> bool:
    """quote 是不是 author_id 本人在這段裡說的（去掉空白標點後，是他某一行的一部分）。"""
    q = _norm(quote)
    return len(q) >= 2 and any(l.author_id == author_id and q in _norm(l.text) for l in seg.lines)


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
        # 本人更正只認本人：模型給的原句要真的是 subject 自己說的（別人替他更正、模型張冠李戴都不算）
        if kind == "correction" and not said_by(seg, sid, str(item.get("quote") or "")):
            continue
        day = str(item.get("day") or "").strip()
        if not _DAY.match(day) or not (seg.first_day <= day <= seg.last_day):
            day = seg.last_day  # 模型亂填日期就用段落日期
        conf = "high" if item.get("confidence") == "high" or kind == "correction" else "low"
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
        async with trace.start("profile", guild=self.guild_id, channel=seg.channel_id,
                               trigger="extract") as tr:
            data = await self.llm.chat_json(messages, prompts.EXTRACT_SCHEMA, name="profile_items",
                                            max_tokens=self.max_tokens)
            tr.final = None if data is None else json.dumps(data, ensure_ascii=False)
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
        existing = self.files.read(self.guild_id, subject_id)
        old_corr, rest = revise.split_corrections(existing)
        current = self._body(rest)
        last_day = revise.header_day(existing)  # 舊檔的「資料到」：判斷新資訊是不是比舊內容新
        removed_n = 0
        for batch in self._batches(pending):
            # 被反駁的舊內容：模型提議、程式裁決（revise.adjudicate），准的先從舊檔拿掉再合併
            removed: list[dict] = []
            if current and removed_n < revise.MAX_REMOVALS:
                current, removed = await self.review(name, current, [f for _, f in batch], last_day,
                                                     revise.MAX_REMOVALS - removed_n)
            # 「本人更正」由程式保存：資料庫裡所有的 correction + 舊檔區塊，不讓模型改寫
            corrections = revise.merge_corrections(
                old_corr, [revise.correction_line(f) for f in self.db.corrections(self.guild_id, subject_id)])
            messages = [
                {"role": "system", "content": prompts.MERGE_SYSTEM.format(name=name, limit=self.profile_limit)},
                {"role": "user", "content": prompts.merge_user(
                    name, current, [f.line() for _, f in batch if f.kind != "correction"], corrections)},
            ]
            async with trace.start("profile", guild=self.guild_id, trigger="merge") as tr:
                try:
                    out = await self.llm.chat(messages, temperature=0.3, max_tokens=self.max_tokens)
                except LLMError as e:
                    tr.outcome, tr.error = "error", str(e)[:500]
                    log.warning("%s 整理失敗：%s", name, e)
                    return False
                tr.final = out
                if "## 基本資料" not in self._body(out):
                    tr.outcome, tr.error = "error", "整理結果格式不對"
            body = self._body(revise.split_corrections(out)[1])  # 模型照抄了「本人更正」也拿掉，用程式的
            if "## 基本資料" not in body:
                log.warning("%s 整理結果格式不對，略過這批：%s", name, out[:200])
                return False
            current, last_day = body, batch[-1][1].day
            self.files.write(self.guild_id, subject_id,
                             self._header(name, last_day) + revise.render_corrections(corrections) + current)
            self.db.mark_merged([i for i, _ in batch])
            self.files.log_removed(self.guild_id, subject_id, removed)  # 合併成功才記（失敗時舊檔沒動）
            removed_n += len(removed)
            log.info("%s：整理了 %d 條（資料到 %s），刪掉 %d 條被反駁的", name, len(batch), last_day, len(removed))
        return True

    async def review(self, name: str, current: str, facts: list[Fact], file_day: str,
                     limit: int) -> tuple[str, list[dict]]:
        """問模型哪些舊條目被這批新資訊推翻了，程式裁決後回傳 (拿掉准刪條目的檔案, 刪除紀錄)。出錯就不刪。"""
        old = revise.numbered_lines(current)
        if not old or not facts:
            return current, []
        lines = [f.line() for f in facts]
        messages = [
            {"role": "system", "content": prompts.REVIEW_SYSTEM.format(name=name)},
            {"role": "user", "content": prompts.review_user(name, [(l.no, l.section, l.text) for l in old],
                                                            lines, file_day)},
        ]
        async with trace.start("profile", guild=self.guild_id, trigger="review") as tr:
            try:
                data = await self.llm.chat_json(messages, prompts.REVIEW_SCHEMA, name="profile_review",
                                                max_tokens=self.max_tokens)
            except LLMError as e:
                tr.outcome, tr.error = "error", str(e)[:500]
                log.warning("%s 檢查被反駁的內容失敗（這次不刪）：%s", name, e)
                return current, []
            tr.final = None if data is None else json.dumps(data, ensure_ascii=False)
        verdicts = revise.adjudicate((data or {}).get("remove") or [], old, facts, file_day, limit=limit)
        now = datetime.now().isoformat(timespec="seconds")
        records = []
        for v in verdicts:
            if not v.ok:
                log.info("%s：不刪（%s）：%s", name, v.why, v.raw)
                continue
            records.append({"at": now, "section": v.line.section, "line": v.line.text, "reason": v.reason,
                            "evidence": [{"day": f.day, "kind": f.kind, "text": f.text} for f in v.evidence]})
        return revise.remove_lines(current, [v.line for v in verdicts if v.ok]), records

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
