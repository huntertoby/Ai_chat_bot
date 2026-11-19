"""舊版 RAG 的事實記憶庫（DB_PATH，預設 data/yuki.db）。

D-04 改成 Yuki 自己查封存區之後，這份資料已經不再寫入、也不再放進 prompt；程式碼（embedding、reranker、
記憶整理）在 R1 移除了，資料留在硬碟上沒動。只剩 /不要記我 還會把那個人的舊記憶一起刪掉。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def forget_person(path: Path | str, user_id: str) -> int:
    """刪掉所有伺服器裡關於這個人、或由他的發言產生的舊記憶。檔案或資料表不存在就是 0（不會建新檔）。"""
    path = Path(path)
    if not path.exists():
        return 0
    db = sqlite3.connect(str(path))
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone():
            return 0
        n = db.execute("DELETE FROM memories WHERE subject_id=? OR author_id=?", (user_id, user_id)).rowcount
        db.commit()
        return n
    finally:
        db.close()
