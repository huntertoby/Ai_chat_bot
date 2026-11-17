"""兩層資料：

- ProfileDB（data/profiles.db）：從聊天段落抽出來的一條條資訊（fact / joke / meme …），以及抽到哪裡的進度。
- ProfileFiles（data/profiles/<guild>/<user>.md）：每個人整理好的檔案，給人看也給 Yuki 讀。
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

KINDS = ("fact", "status", "opinion", "relation", "meme", "joke")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    confidence TEXT NOT NULL,
    day TEXT NOT NULL,
    source TEXT NOT NULL,            -- channel_id:最後一則訊息 ID，可回頭查原文
    merged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_fact_subject ON facts(guild_id, subject_id, merged);
CREATE TABLE IF NOT EXISTS progress (
    channel_id TEXT PRIMARY KEY,
    last_id INTEGER NOT NULL         -- 這個頻道已抽取到的最後一則訊息 ID
);
"""


@dataclass
class Fact:
    subject_id: str
    subject_name: str
    kind: str
    text: str
    confidence: str
    day: str

    def line(self) -> str:
        mark = "" if self.confidence == "high" else "（不確定）"
        return f"- [{self.day}][{self.kind}] {self.text}{mark}"


class ProfileDB:
    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(_SCHEMA)

    def last_id(self, channel_id: str) -> int | None:
        row = self.db.execute("SELECT last_id FROM progress WHERE channel_id=?", (channel_id,)).fetchone()
        return row[0] if row else None

    def save_segment(self, guild_id: str, channel_id: str, last_id: int, facts: list[Fact]) -> None:
        """一個段落的結果和進度一起 commit：中斷後重跑不會重複，也不會漏。"""
        source = f"{channel_id}:{last_id}"
        self.db.executemany(
            "INSERT INTO facts (guild_id, subject_id, subject_name, kind, text, confidence, day, source)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [(guild_id, f.subject_id, f.subject_name, f.kind, f.text, f.confidence, f.day, source)
             for f in facts])
        self.db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?)", (channel_id, last_id))
        self.db.commit()

    def pending(self, guild_id: str, subject_id: str) -> list[tuple[int, Fact]]:
        rows = self.db.execute(
            "SELECT id, subject_id, subject_name, kind, text, confidence, day FROM facts"
            " WHERE guild_id=? AND subject_id=? AND merged=0 ORDER BY day, id", (guild_id, subject_id))
        return [(r[0], Fact(*r[1:])) for r in rows]

    def mark_merged(self, ids: list[int]) -> None:
        self.db.executemany("UPDATE facts SET merged=1 WHERE id=?", [(i,) for i in ids])
        self.db.commit()

    def reset_merged(self, guild_id: str, subject_id: str | None = None) -> None:
        """重新整理 .md 用：把資訊標回未整理。"""
        if subject_id:
            self.db.execute("UPDATE facts SET merged=0 WHERE guild_id=? AND subject_id=?", (guild_id, subject_id))
        else:
            self.db.execute("UPDATE facts SET merged=0 WHERE guild_id=?", (guild_id,))
        self.db.commit()

    def forget(self, guild_id: str, subject_id: str) -> int:
        n = self.db.execute("DELETE FROM facts WHERE guild_id=? AND subject_id=?", (guild_id, subject_id)).rowcount
        self.db.commit()
        return n

    def stats(self, guild_id: str) -> list[tuple[str, str, int, int]]:
        """[(subject_id, 名字, 全部筆數, 未整理筆數)]，由多到少。"""
        return self.db.execute(
            "SELECT subject_id, MAX(subject_name), COUNT(*), SUM(merged=0) FROM facts WHERE guild_id=?"
            " GROUP BY subject_id ORDER BY 3 DESC", (guild_id,)).fetchall()

    def kind_counts(self, guild_id: str) -> dict[str, int]:
        return dict(self.db.execute("SELECT kind, COUNT(*) FROM facts WHERE guild_id=? GROUP BY kind", (guild_id,)))


_SAFE = re.compile(r"[^0-9A-Za-z_-]")


class ProfileFiles:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, guild_id: str, user_id: str) -> Path:
        return self.root / _SAFE.sub("", guild_id) / f"{_SAFE.sub('', user_id)}.md"

    def read(self, guild_id: str, user_id: str) -> str:
        p = self.path(guild_id, user_id)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def write(self, guild_id: str, user_id: str, text: str) -> None:
        p = self.path(guild_id, user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(text.strip() + "\n", encoding="utf-8")
        tmp.replace(p)  # 寫到一半中斷也不會留下半份檔案

    def delete(self, guild_id: str, user_id: str) -> bool:
        p = self.path(guild_id, user_id)
        if p.exists():
            p.unlink()
            return True
        return False

    def for_prompt(self, guild_id: str, user_id: str, max_chars: int) -> str:
        """給回覆 prompt 用：拿掉標題行，太長就截斷（各節的重點寫在前面）。"""
        text = self.read(guild_id, user_id)
        text = "\n".join(l for l in text.splitlines() if not l.startswith("# ") and not l.startswith("> "))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text if len(text) <= max_chars else text[:max_chars].rsplit("\n", 1)[0] + "\n…"
