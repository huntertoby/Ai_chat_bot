"""聊天封存區：原始訊息（之後切段、算向量給檢索用）。跟「事實記憶」分開放在 data/archive.db。

只收公開頻道（@everyone 看得到的）：私人頻道的內容一旦被 Yuki 在公開頻道講出來就等於外洩。
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,          -- Discord 訊息 ID（snowflake，本身就依時間遞增）
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,        -- 討論串的話是討論串 ID
    channel_name TEXT NOT NULL,
    parent_name TEXT NOT NULL DEFAULT '',  -- 討論串／論壇貼文所屬的頻道
    author_id TEXT NOT NULL,
    author_name TEXT NOT NULL,
    created_at TEXT NOT NULL,        -- 本地時間 YYYY-MM-DD HH:MM:SS
    content TEXT NOT NULL,           -- clean_content：提及已換成名字
    reply_to INTEGER,
    attachments INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_msg_channel ON messages(guild_id, channel_id, id);
CREATE INDEX IF NOT EXISTS idx_msg_author ON messages(guild_id, author_id);
CREATE TABLE IF NOT EXISTS fetch_progress (
    channel_id TEXT PRIMARY KEY,
    last_id INTEGER NOT NULL         -- 這個頻道已抓到的最新訊息 ID，重跑從這之後接著抓
);
CREATE TABLE IF NOT EXISTS optout (
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    PRIMARY KEY (guild_id, user_id)
);
-- 全文檢索（開發規格 6.4）：rowid = 訊息 ID，tok = jieba 斷詞後用空白隔開的內容
CREATE VIRTUAL TABLE IF NOT EXISTS msg_fts USING fts5(tok, tokenize='unicode61');
-- 備案：中文每兩個字一組（「那麼晚送」→「那麼 麼晚 晚送」）、英數整個字。jieba 切法跟查詢對不上時用
CREATE VIRTUAL TABLE IF NOT EXISTS msg_bi USING fts5(bi, tokenize='unicode61');
"""


_T2S = None


def to_simplified(text: str) -> str:
    global _T2S
    if _T2S is None:
        from opencc import OpenCC
        _T2S = OpenCC("t2s")
    return _T2S.convert(text)


def fts_text(content: str) -> str:
    """斷詞後用空白接起來。先轉簡體再斷：jieba 的詞典以簡體為主，繁體會被切爛
    （「麥當勞」→「吃麥」「當勞」；轉簡體後是「麦当劳」）。索引和查詢都走這裡，所以一致。"""
    from .text_index import tokenize
    return " ".join(tokenize(to_simplified(content)))


_CJK_RUN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+|[a-z0-9]+")


def bigram_text(content: str) -> str:
    """中文連續字切成兩兩一組；英數字保留整個字；單一個中文字保留單字。"""
    out = []
    for run in _CJK_RUN.findall(to_simplified(content).lower()):
        if run[0].isascii():
            out.append(run)
        elif len(run) == 1:
            out.append(run)
        else:
            out += [run[i:i + 2] for i in range(len(run) - 1)]
    return " ".join(out)


@dataclass
class ArchivedMessage:
    id: int
    guild_id: str
    channel_id: str
    channel_name: str
    parent_name: str
    author_id: str
    author_name: str
    created_at: str
    content: str
    reply_to: int | None
    attachments: int


class ArchiveDB:
    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(_SCHEMA)

    def add_many(self, msgs: list[ArchivedMessage]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(m.id, m.guild_id, m.channel_id, m.channel_name, m.parent_name, m.author_id, m.author_name,
              m.created_at, m.content, m.reply_to, m.attachments) for m in msgs])
        self.index([(m.id, m.content) for m in msgs])

    def index(self, rows: list[tuple[int, str]]) -> None:
        """把訊息加進全文檢索（同一則重複加會覆蓋）。"""
        rows = [(i, c) for i, c in rows if c and c.strip()]
        ids = [(i,) for i, _ in rows]
        self.db.executemany("DELETE FROM msg_fts WHERE rowid=?", ids)
        self.db.executemany("DELETE FROM msg_bi WHERE rowid=?", ids)
        tok = [(i, fts_text(c)) for i, c in rows]
        self.db.executemany("INSERT INTO msg_fts(rowid, tok) VALUES (?, ?)", [r for r in tok if r[1]])
        bi = [(i, bigram_text(c)) for i, c in rows]
        self.db.executemany("INSERT INTO msg_bi(rowid, bi) VALUES (?, ?)", [r for r in bi if r[1]])

    def last_id(self, channel_id: str) -> int | None:
        row = self.db.execute("SELECT last_id FROM fetch_progress WHERE channel_id=?", (channel_id,)).fetchone()
        return row[0] if row else None

    def set_last_id(self, channel_id: str, last_id: int) -> None:
        self.db.execute("INSERT OR REPLACE INTO fetch_progress VALUES (?, ?)", (channel_id, last_id))

    def advance_last_id(self, channel_id: str, last_id: int) -> None:
        """進度只往前推，不會倒退。"""
        self.db.execute(
            "INSERT INTO fetch_progress VALUES (?, ?)"
            " ON CONFLICT(channel_id) DO UPDATE SET last_id = MAX(last_id, excluded.last_id)",
            (channel_id, last_id))

    def latest_id(self, guild_id: str) -> int | None:
        """這個伺服器封存到的最新一則訊息（大約就是上次關機的時間）。"""
        return self.db.execute("SELECT MAX(id) FROM messages WHERE guild_id=?", (guild_id,)).fetchone()[0]

    def max_id(self, channel_id: str) -> int | None:
        return self.db.execute("SELECT MAX(id) FROM messages WHERE channel_id=?", (channel_id,)).fetchone()[0]

    def commit(self) -> None:
        self.db.commit()

    def authors(self, guild_id: str) -> list[tuple[str, str]]:
        """這個伺服器封存區裡講過話的人：[(user ID, 最近用的名字)]。"""
        return self.db.execute(
            "SELECT author_id, author_name FROM messages WHERE guild_id=? AND id IN "
            "(SELECT MAX(id) FROM messages WHERE guild_id=? GROUP BY author_id)", (guild_id, guild_id)).fetchall()

    def opted_out(self, guild_id: str | None = None) -> set[str]:
        """打過 /不要記我 的人。以人為單位：在任何伺服器（或私訊）打的都算，guild_id 不影響結果
        （參數留著給舊的呼叫端）。"""
        return {r[0] for r in self.db.execute("SELECT DISTINCT user_id FROM optout")}

    def opt_out(self, guild_id: str, user_id: str) -> int:
        """退出封存：刪掉這個人在**所有伺服器**的訊息，之後也不再收。guild_id 只記錄是在哪裡打的。回傳刪除筆數。"""
        self.db.execute("INSERT OR IGNORE INTO optout VALUES (?, ?)", (guild_id, user_id))
        for table in ("msg_fts", "msg_bi"):
            self.db.execute(f"DELETE FROM {table} WHERE rowid IN (SELECT id FROM messages WHERE author_id=?)",
                            (user_id,))
        n = self.db.execute("DELETE FROM messages WHERE author_id=?", (user_id,)).rowcount
        self.db.commit()
        return n

    def counts(self, guild_id: str) -> list[tuple[str, int, str, str]]:
        """[(頻道名稱, 則數, 最早, 最新)]，由多到少。"""
        return self.db.execute(
            "SELECT CASE WHEN parent_name != '' THEN parent_name || ' › ' || channel_name ELSE channel_name END,"
            " COUNT(*), MIN(created_at), MAX(created_at) FROM messages WHERE guild_id=?"
            " GROUP BY channel_id ORDER BY 2 DESC", (guild_id,)).fetchall()
