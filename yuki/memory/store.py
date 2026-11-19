"""記憶庫：SQLite 存資料與向量，記憶體內維護每個 guild 的向量矩陣與 BM25 索引。

為什麼不用 Chroma：記憶量小（幾百～幾千筆），dense 用 numpy 暴力算就夠快；
而 BM25 本來就要另外建索引，資料放一處反而不會兩邊不同步。
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .text_index import BM25, add_words, tokenize


@dataclass
class Memory:
    id: str
    guild_id: str
    subject_id: str | None   # 記憶是「關於誰」，群組事件為 None
    subject_name: str
    text: str
    author_id: str | None    # 誰的發言產生了這筆記憶
    created_at: str
    updated_at: str

    def line(self) -> str:
        return f"[{self.id}] ({self.updated_at[:10]}) {self.text}"


class EmbeddingMismatch(RuntimeError):
    pass


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    subject_id TEXT,
    subject_name TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    author_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_guild ON memories(guild_id, subject_id);
CREATE TABLE IF NOT EXISTS whiteboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_COLS = "id, guild_id, subject_id, subject_name, text, author_id, created_at, updated_at"


class _GuildIndex:
    def __init__(self, rows: list[tuple], tokens_of):
        self.memories = [Memory(*r[:8]) for r in rows]
        self.ids = [m.id for m in self.memories]
        self.subjects = np.array([m.subject_id or "" for m in self.memories], dtype=object)
        self.created = np.array([m.created_at[:10] for m in self.memories], dtype=object)
        self.updated = np.array([m.updated_at[:10] for m in self.memories], dtype=object)
        if rows:
            self.matrix = np.stack([np.frombuffer(r[8], dtype=np.float32) for r in rows])
        else:
            self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25([tokens_of(m) for m in self.memories])

    def mask(self, subject_id: str | None, since: str | None = None, until: str | None = None) -> np.ndarray:
        keep = np.ones(len(self.ids), dtype=bool)
        if subject_id is not None:
            keep &= self.subjects == subject_id
        if since or until:
            # 發生在這段期間，或在這段期間被更新過（舊記憶合併了新消息）都算
            lo, hi = since or "0000-00-00", until or "9999-99-99"
            keep &= ((self.created >= lo) & (self.created <= hi)) | ((self.updated >= lo) & (self.updated <= hi))
        return keep


class MemoryStore:
    def __init__(self, path: Path | str, embed_model: str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(_SCHEMA)
        self.embed_model = embed_model
        self._check_model()
        self._cache: dict[str, _GuildIndex] = {}
        # jieba 斷詞是重建索引最慢的部分（上萬筆要好幾秒），依 (id, text) 快取，只斷新的或改過的
        self._tokens: dict[str, tuple[str, list[str]]] = {}
        add_words(self.subject_names())

    # ---------- 基本 ----------
    def _check_model(self) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
        has_rows = self.db.execute("SELECT 1 FROM memories LIMIT 1").fetchone()
        if row is None or not has_rows:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('embed_model', ?)", (self.embed_model,))
            self.db.commit()
        elif row[0] != self.embed_model:
            raise EmbeddingMismatch(
                f"資料庫裡的向量是用 {row[0]} 算的，現在設定是 {self.embed_model}。"
                f"請執行 python -m scripts.reindex 重新計算向量。")

    def set_embed_model(self, model: str) -> None:
        self.embed_model = model
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('embed_model', ?)", (model,))
        self.db.commit()

    def _index(self, guild_id: str) -> _GuildIndex:
        if guild_id not in self._cache:
            rows = self.db.execute(
                f"SELECT {_COLS}, embedding FROM memories WHERE guild_id=? ORDER BY created_at",
                (guild_id,)).fetchall()
            self._cache[guild_id] = _GuildIndex(rows, self._tokens_of)
        return self._cache[guild_id]

    def _tokens_of(self, mem: Memory) -> list[str]:
        cached = self._tokens.get(mem.id)
        if cached is None or cached[0] != mem.text:
            cached = (mem.text, tokenize(mem.text))
            self._tokens[mem.id] = cached
        return cached[1]

    def _dirty(self, guild_id: str) -> None:
        self._cache.pop(guild_id, None)

    def _new_id(self) -> str:
        while True:
            mem_id = uuid.uuid4().hex[:8]  # 短 ID 讓 LLM 好抄
            if not self.db.execute("SELECT 1 FROM memories WHERE id=?", (mem_id,)).fetchone():
                return mem_id

    # ---------- CRUD ----------
    def add(self, guild_id: str, text: str, embedding: np.ndarray, *, subject_id: str | None = None,
            subject_name: str = "", author_id: str | None = None,
            created_at: str | None = None) -> Memory:
        ts = created_at or now()
        mem = Memory(self._new_id(), guild_id, subject_id, subject_name, text, author_id, ts, ts)
        self.db.execute(
            f"INSERT INTO memories ({_COLS}, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
            (mem.id, guild_id, subject_id, subject_name, text, author_id, ts, ts,
             np.asarray(embedding, dtype=np.float32).tobytes()))
        self.db.commit()
        if subject_name:
            add_words([subject_name])
        self._dirty(guild_id)
        return mem

    def update(self, mem_id: str, text: str, embedding: np.ndarray) -> Memory | None:
        mem = self.get(mem_id)
        if mem is None:
            return None
        self.db.execute("UPDATE memories SET text=?, embedding=?, updated_at=? WHERE id=?",
                        (text, np.asarray(embedding, dtype=np.float32).tobytes(), now(), mem_id))
        self.db.commit()
        self._dirty(mem.guild_id)
        return self.get(mem_id)

    def delete(self, mem_id: str) -> bool:
        mem = self.get(mem_id)
        if mem is None:
            return False
        self.db.execute("DELETE FROM memories WHERE id=?", (mem_id,))
        self.db.commit()
        self._dirty(mem.guild_id)
        return True

    def forget_person(self, user_id: str) -> int:
        """/不要記我：刪掉所有伺服器裡關於這個人、或由他的發言產生的記憶。回傳刪除筆數。"""
        guilds = [r[0] for r in self.db.execute(
            "SELECT DISTINCT guild_id FROM memories WHERE subject_id=? OR author_id=?", (user_id, user_id))]
        n = self.db.execute("DELETE FROM memories WHERE subject_id=? OR author_id=?", (user_id, user_id)).rowcount
        self.db.commit()
        for g in guilds:
            self._dirty(g)
        return n

    def get(self, mem_id: str) -> Memory | None:
        row = self.db.execute(f"SELECT {_COLS} FROM memories WHERE id=?", (mem_id,)).fetchone()
        return Memory(*row) if row else None

    def get_many(self, ids: list[str]) -> list[Memory]:
        found = {m.id: m for m in (self.get(i) for i in ids) if m}
        return [found[i] for i in ids if i in found]

    def list(self, guild_id: str, subject_id: str | None = None) -> list[Memory]:
        idx = self._index(guild_id)
        mask = idx.mask(subject_id)
        mems = [m for m, keep in zip(idx.memories, mask) if keep]
        return sorted(mems, key=lambda m: m.updated_at, reverse=True)

    def all_rows(self) -> list[Memory]:
        return [Memory(*r) for r in self.db.execute(f"SELECT {_COLS} FROM memories")]

    def subjects(self, guild_id: str) -> list[tuple[str, str]]:
        """這個 guild 的記憶涉及的人：[(subject_id, subject_name)]。"""
        return self.db.execute(
            "SELECT DISTINCT subject_id, subject_name FROM memories "
            "WHERE guild_id=? AND subject_id IS NOT NULL AND subject_name != ''", (guild_id,)).fetchall()

    def subject_names(self) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT subject_name FROM memories WHERE subject_name != ''")]

    # ---------- 檢索 ----------
    def dense_search(self, guild_id: str, query_vec: np.ndarray, n: int, subject_id: str | None = None,
                     since: str | None = None, until: str | None = None) -> list[tuple[str, float]]:
        idx = self._index(guild_id)
        if not idx.ids:
            return []
        sims = idx.matrix @ np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = np.where(idx.mask(subject_id, since, until), sims, -np.inf)
        order = np.argsort(-sims)[:n]
        return [(idx.ids[i], float(sims[i])) for i in order if np.isfinite(sims[i])]

    def sparse_search(self, guild_id: str, query: str, n: int, subject_id: str | None = None,
                      since: str | None = None, until: str | None = None) -> list[tuple[str, float]]:
        idx = self._index(guild_id)
        if not idx.ids:
            return []
        scores = np.array(idx.bm25.scores(tokenize(query)))
        scores = np.where(idx.mask(subject_id, since, until), scores, 0.0)
        order = np.argsort(-scores)[:n]
        return [(idx.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    # ---------- 白板 ----------
    def whiteboard(self, guild_id: str) -> list[tuple[int, str, str]]:
        return self.db.execute(
            "SELECT id, text, created_at FROM whiteboard WHERE guild_id=? ORDER BY id",
            (guild_id,)).fetchall()

    def whiteboard_add(self, guild_id: str, text: str, created_at: str | None = None) -> None:
        self.db.execute("INSERT INTO whiteboard (guild_id, text, created_at) VALUES (?,?,?)",
                        (guild_id, text, created_at or now()))
        self.db.commit()

    def whiteboard_replace(self, guild_id: str, items: list[str]) -> None:
        self.db.execute("DELETE FROM whiteboard WHERE guild_id=?", (guild_id,))
        ts = now()
        self.db.executemany("INSERT INTO whiteboard (guild_id, text, created_at) VALUES (?,?,?)",
                            [(guild_id, t, ts) for t in items])
        self.db.commit()
