"""Agentic RAG（開發規格 6.4）：Yuki 自己用工具查封存區的聊天紀錄和人物檔案，不用 embedding / reranker。

search_chat：SQLite FTS5（jieba 斷詞）找訊息；read_chat：讀某則前後文；get_profile：人物檔案。
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from ..archive import bigram_text, fts_text

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_chat",
        "description": "搜尋群組以前的聊天紀錄（誰說過什麼、某件事是什麼時候、某人最近在幹嘛）。"
                       "用關鍵字比對訊息內容，不是語意搜尋：關鍵字用短的名詞，同一件事換 2～3 種說法一起給。",
        "parameters": {"type": "object", "properties": {
            "keywords": {"type": "array", "items": {"type": "string"},
                         "description": "關鍵字：短的名詞，同一件事給幾種說法；任一個符合就算"},
            "person": {"type": "string", "description": "只找這個人說的話（名字），不限就留空"},
            "since": {"type": "string", "description": "從這天開始 YYYY-MM-DD，不限就留空"},
            "until": {"type": "string", "description": "到這天為止 YYYY-MM-DD，不限就留空"},
        }, "required": ["keywords"]}}},
    {"type": "function", "function": {
        "name": "read_chat",
        "description": "讀某則訊息前後的對話，搞清楚上下文。message_id 用 search_chat 結果裡的 #編號。",
        "parameters": {"type": "object", "properties": {
            "message_id": {"type": "integer"},
        }, "required": ["message_id"]}}},
    {"type": "function", "function": {
        "name": "get_profile",
        "description": "看某個群組成員的人物檔案（你對他的整理和印象，可能有錯）。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
        }, "required": ["name"]}}},
]

NAMES = {t["function"]["name"] for t in TOOL_SPECS}


def bigram_query(keywords: list[str]) -> str:
    """備案查詢：每個關鍵字切成二字組，全部都要出現（AND），不同關鍵字 OR。"""
    groups = []
    for kw in keywords:
        grams = bigram_text(str(kw)).split()
        if grams:
            groups.append("(" + " AND ".join('"' + g.replace('"', "") + '"' for g in grams) + ")")
    return " OR ".join(groups)


def fts_query(keywords: list[str]) -> str:
    """每個關鍵字的詞都要出現（AND），不同關鍵字任一符合（OR）。"""
    groups = []
    for kw in keywords:
        toks = fts_text(str(kw)).split()
        if toks:
            groups.append("(" + " AND ".join('"' + t.replace('"', "") + '"' for t in toks) + ")")
    return " OR ".join(groups)


@dataclass
class Hit:
    id: int
    created_at: str
    channel: str
    author: str
    content: str
    score: float

    def line(self, max_chars: int = 150) -> str:
        text = self.content if len(self.content) <= max_chars else self.content[:max_chars] + "…"
        return f"#{self.id} {self.created_at[:16]} #{self.channel} {self.author}：{text}"


def rank(bm25: float, content: str, created_at: str, now: datetime) -> float:
    """bm25 偏好很短的訊息（只打「麥當勞」三個字的那則分最高），但短訊息沒什麼資訊：
    依長度打折，再稍微偏好新的。"""
    rel = -bm25
    length = min(1.0, 0.1 + len(content) / 25)
    try:
        age = (now - datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")).days
    except ValueError:
        age = 365
    recency = 0.6 + 0.4 * math.exp(-max(age, 0) / 180)
    return rel * length * recency


class ArchiveTools:
    """guild_id：查哪個伺服器的封存區。resolve_person(name) → user ID 集合（名字比對成員名單）。
    profile(name) → 檔案內容或空字串。"""

    def __init__(self, db: sqlite3.Connection, guild_id: str, *, resolve_person: Callable[[str], set[str]],
                 profile: Callable[[str], str], skip_authors: set[str] = frozenset(), limit: int = 10,
                 now: datetime | None = None):
        self.db = db
        self.guild_id = guild_id
        self.resolve_person = resolve_person
        self.profile = profile
        self.skip = set(skip_authors)
        self.limit = limit
        self.now = now
        self.seen: list[int] = []  # 這次回覆查到過的訊息 ID（評估用）

    async def call(self, name: str, args: dict) -> str:
        if name == "search_chat":
            kws = args.get("keywords") or []
            if isinstance(kws, str):
                kws = [kws]
            return self.search_text(kws, args.get("person") or "", args.get("since") or "", args.get("until") or "")
        if name == "read_chat":
            return self.read_text(int(args.get("message_id") or 0))
        if name == "get_profile":
            name_ = str(args.get("name") or "").strip()
            text = self.profile(name_) if name_ else ""
            return text or f"（沒有 {name_} 的人物檔案）"
        return f"（沒有 {name} 這個工具）"

    # ---------- search_chat ----------
    def search(self, keywords: list[str], person: str = "", since: str = "", until: str = "") -> list[Hit]:
        # 關鍵字是成員名字時：名字通常是「誰說的」而不是訊息內容。拿掉它（不然 OR 起來會被
        # 一堆提到他的訊息淹沒），改成「他說的訊息加分」。只剩名字的話照常當關鍵字搜。
        boost: set[str] = set()
        rest = []
        for kw in keywords:
            ids = self.resolve_person(str(kw)) if len(str(kw).strip()) >= 2 else set()
            if ids:
                boost |= ids
            else:
                rest.append(kw)
        if rest:
            keywords = rest
        hits = self._query("msg_fts", fts_query(keywords), boost, person, since, until)
        if len(hits) < self.limit // 2:
            # 斷詞對不上（「那麼晚送」vs「晚」、繁簡、黏在一起的詞）：用二字組再找一次補上
            seen = {h.id for h in hits}
            hits += [h for h in self._query("msg_bi", bigram_query(keywords), boost, person, since, until)
                     if h.id not in seen]
        hits.sort(key=lambda h: h.score, reverse=True)
        out = hits[: self.limit]
        self.seen += [h.id for h in out]
        return out

    def _query(self, table: str, q: str, boost: set[str], person: str, since: str, until: str) -> list[Hit]:
        if not q:
            return []
        sql = ("SELECT m.id, m.created_at, CASE WHEN m.parent_name != '' THEN m.parent_name ELSE m.channel_name END,"
               f" m.author_id, m.author_name, m.content, bm25({table}) FROM {table} JOIN messages m ON m.id = {table}.rowid"
               f" WHERE {table} MATCH ? AND m.guild_id = ?")
        params: list = [q, self.guild_id]
        if person:
            ids = self.resolve_person(person)
            sql += f" AND (m.author_id IN ({','.join('?' * len(ids))}) OR m.author_name = ? COLLATE NOCASE)" if ids \
                else " AND m.author_name = ? COLLATE NOCASE"
            params += [*ids, person]
        if since:
            sql += " AND m.created_at >= ?"
            params.append(since)
        if until:
            sql += " AND m.created_at <= ?"
            params.append(until + " 23:59:59" if len(until) == 10 else until)
        sql += f" ORDER BY bm25({table}) LIMIT 200"
        now = self.now or datetime.now()
        return [Hit(i, ts, ch, au, c, rank(b, c, ts, now) * (1.6 if aid in boost else 1.0))
                for i, ts, ch, aid, au, c, b in self.db.execute(sql, params) if aid not in self.skip]

    def followups(self, hit: Hit, n: int = 2) -> list[Hit]:
        """命中那則之後的 n 則（同頻道）：答案常在下一則（「要用什麼 VPN？」→ 下一則「用 cloudflare」）。"""
        rows = self.db.execute(
            "SELECT id, created_at, author_id, author_name, content FROM messages"
            " WHERE channel_id = (SELECT channel_id FROM messages WHERE id=?) AND id > ? ORDER BY id LIMIT ?",
            (hit.id, hit.id, n + 2)).fetchall()
        out = [Hit(i, ts, hit.channel, au, c, 0.0) for i, ts, aid, au, c in rows if aid not in self.skip and c.strip()]
        return out[:n]

    def search_text(self, keywords, person="", since="", until="") -> str:
        hits = self.search(keywords, person, since, until)
        note = ""
        if not hits and (since or until):
            # 模型算日期常常太窄（「上週」只查 7 天）：時間內沒有就自動拿掉時間再找，省一輪
            hits = self.search(keywords, person)
            note = "（指定的時間內找不到，以下是不限時間的結果，注意日期）\n" if hits else ""
        if not hits and person:
            hits = self.search(keywords)
            note = f"（{person} 說的話裡找不到，以下是所有人的結果）\n" if hits else ""
        if not hits:
            return "（聊天紀錄裡找不到。可以換別的說法再試一次；真的沒有就說不知道。）"
        hits.sort(key=lambda h: h.id)  # 照時間排，比較好讀
        lines, shown = [], {h.id for h in hits}
        for h in hits:
            lines.append(h.line())
            for f in self.followups(h):  # 附上接著的兩則，縮排表示是後文
                if f.id not in shown:
                    shown.add(f.id)
                    self.seen.append(f.id)
                    lines.append("    ↳ " + f.line(100))
        return note + "\n".join(lines)

    def recent(self, user_id: str, n: int = 20) -> list[Hit]:
        """這個人最近說的 n 則（舊到新）。/不要記我 的人回空的。"""
        if user_id in self.skip:
            return []
        rows = self.db.execute(
            "SELECT id, created_at, CASE WHEN parent_name != '' THEN parent_name ELSE channel_name END,"
            " author_name, content FROM messages WHERE guild_id=? AND author_id=? AND content != ''"
            " ORDER BY id DESC LIMIT ?", (self.guild_id, user_id, n)).fetchall()
        return [Hit(i, ts, ch, au, c, 0.0) for i, ts, ch, au, c in reversed(rows)]

    # ---------- read_chat ----------
    def context(self, message_id: int, before: int = 8, after: int = 8) -> list[Hit]:
        row = self.db.execute("SELECT channel_id FROM messages WHERE id=? AND guild_id=?",
                              (message_id, self.guild_id)).fetchone()
        if not row:
            return []
        cid = row[0]
        cols = ("SELECT id, created_at, CASE WHEN parent_name != '' THEN parent_name ELSE channel_name END,"
                " author_id, author_name, content FROM messages WHERE channel_id=? AND guild_id=?")
        prev = self.db.execute(cols + " AND id < ? ORDER BY id DESC LIMIT ?",
                               (cid, self.guild_id, message_id, before)).fetchall()
        rest = self.db.execute(cols + " AND id >= ? ORDER BY id LIMIT ?",
                               (cid, self.guild_id, message_id, after + 1)).fetchall()
        rows = list(reversed(prev)) + rest
        out = [Hit(i, ts, ch, au, c, 0.0) for i, ts, ch, aid, au, c in rows if aid not in self.skip and c.strip()]
        self.seen += [h.id for h in out]
        return out

    def read_text(self, message_id: int) -> str:
        hits = self.context(message_id)
        if not hits:
            return "（找不到這則訊息）"
        return "\n".join(("→ " if h.id == message_id else "  ") + h.line(300) for h in hits)
