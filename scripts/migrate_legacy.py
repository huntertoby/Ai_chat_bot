"""把舊版 Chroma 記憶與 memory.md 白板搬進新資料庫，並用目前的 embedding 模型重算向量。

舊資料沒有記錄是哪個伺服器，所以要指定 --guild-id（Discord：右鍵伺服器 → 複製伺服器 ID，
需先在設定裡開啟開發者模式）。

    python -m scripts.migrate_legacy --guild-id 123456789012345678
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sqlite3
from pathlib import Path

from yuki.config import ROOT, load_settings
from yuki.embedding import build_embedder
from yuki.memory.store import MemoryStore

LEGACY = ROOT / "legacy"


def read_chroma(path: Path) -> list[dict]:
    db = sqlite3.connect(path)
    rows = db.execute(
        "SELECT e.embedding_id, m.key, m.string_value FROM embeddings e "
        "JOIN embedding_metadata m ON m.id = e.id").fetchall()
    items: dict[str, dict] = {}
    for emb_id, key, value in rows:
        items.setdefault(emb_id, {})[key] = value
    return [{"text": v["chroma:document"], "user_id": v.get("user_id"), "timestamp": v.get("timestamp")}
            for v in items.values() if v.get("chroma:document")]


def read_whiteboard(path: Path) -> list[tuple[str, str | None]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"-\s*\[(.+?)\]\s*(.+)", line.strip())
        if m:
            out.append((m.group(2).strip(), m.group(1)))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guild-id", required=True)
    ap.add_argument("--chroma", type=Path, default=LEGACY / "chroma_memory" / "chroma.sqlite3")
    ap.add_argument("--whiteboard", type=Path, default=LEGACY / "memory.md")
    ap.add_argument("--db", type=Path, default=None, help="目標資料庫，預設用 .env 的 DB_PATH")
    args = ap.parse_args()

    settings = load_settings()
    embedder = build_embedder(settings)
    store = MemoryStore(args.db or settings.db_path, embedder.model_name)
    existing = {(m.guild_id, m.text) for m in store.all_rows()}

    items = read_chroma(args.chroma) if args.chroma.exists() else []
    todo = [it for it in items if (args.guild_id, it["text"]) not in existing]
    if todo:
        vecs = await embedder.embed_docs([it["text"] for it in todo])
        for it, vec in zip(todo, vecs):
            # 舊版的 user_id 是「發言者」，大多數記憶也是關於發言者本人，所以當成 subject
            store.add(args.guild_id, it["text"], vec, subject_id=it["user_id"],
                      author_id=it["user_id"], created_at=it["timestamp"])
    print(f"記憶：舊資料 {len(items)} 筆，新匯入 {len(todo)} 筆（已存在的略過）")

    if args.whiteboard.exists():
        current = {t for _, t, _ in store.whiteboard(args.guild_id)}
        board = [(t, ts) for t, ts in read_whiteboard(args.whiteboard) if t not in current]
        for text, ts in board:
            store.whiteboard_add(args.guild_id, text, ts)
        print(f"白板：新匯入 {len(board)} 條")


if __name__ == "__main__":
    asyncio.run(main())
