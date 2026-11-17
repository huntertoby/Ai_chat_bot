"""封存區全文檢索：一次幫舊訊息建索引（開發規格 6.4）。之後新訊息會在封存時自動加入。

    python -m scripts.build_fts            # 只補還沒建的
    python -m scripts.build_fts --rebuild  # 全部重建（改了斷詞方式時用）
"""
from __future__ import annotations

import argparse
import time

from yuki.archive import ArchiveDB, to_simplified
from yuki.bot import ARCHIVE_PATH
from yuki.memory.text_index import add_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    db = ArchiveDB(ARCHIVE_PATH)
    if args.rebuild:
        db.db.execute("DELETE FROM msg_fts")
        db.db.execute("DELETE FROM msg_bi")
    # 成員名字先加進 jieba 詞典，不然「hunterchiao」「阿明」可能被切碎
    names = [r[0] for r in db.db.execute("SELECT DISTINCT author_name FROM messages")]
    add_words(names + [to_simplified(n) for n in names])  # 斷詞前會轉簡體，名字也要加簡體版
    todo = db.db.execute("SELECT id, content FROM messages WHERE length(trim(content)) > 0 AND ("
                         " id NOT IN (SELECT rowid FROM msg_fts) OR id NOT IN (SELECT rowid FROM msg_bi))"
                         " ORDER BY id").fetchall()
    t = time.time()
    for i in range(0, len(todo), 5000):
        db.index(todo[i:i + 5000])
        db.commit()
        print(f"{min(i + 5000, len(todo))}/{len(todo)}（{time.time() - t:.0f} 秒）", flush=True)
    db.db.execute("INSERT INTO msg_fts(msg_fts) VALUES ('optimize')")
    db.db.execute("INSERT INTO msg_bi(msg_bi) VALUES ('optimize')")
    db.commit()
    print("完成，索引筆數：", db.db.execute("SELECT COUNT(*) FROM msg_fts").fetchone()[0])


if __name__ == "__main__":
    main()
