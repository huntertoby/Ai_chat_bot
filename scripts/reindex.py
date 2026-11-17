"""換了 EMBED_MODEL 之後，用新模型重算所有記憶的向量。

    python -m scripts.reindex
"""
from __future__ import annotations

import asyncio
import sqlite3

from yuki.config import load_settings
from yuki.embedding import build_embedder


async def main() -> None:
    settings = load_settings()
    embedder = build_embedder(settings)
    db = sqlite3.connect(settings.db_path)
    rows = db.execute("SELECT id, text FROM memories").fetchall()
    for i in range(0, len(rows), 64):
        batch = rows[i:i + 64]
        vecs = await embedder.embed_docs([t for _, t in batch])
        db.executemany("UPDATE memories SET embedding=? WHERE id=?",
                       [(v.tobytes(), mem_id) for (mem_id, _), v in zip(batch, vecs)])
    db.execute("INSERT OR REPLACE INTO meta VALUES ('embed_model', ?)", (embedder.model_name,))
    db.commit()
    print(f"已用 {embedder.model_name} 重算 {len(rows)} 筆記憶的向量")


if __name__ == "__main__":
    asyncio.run(main())
