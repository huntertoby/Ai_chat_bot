"""把 5 年假記憶複製成某個伺服器的測試資料庫，讓 bot 在 Discord 上直接用假資料測試。

正式記憶庫（data/yuki.db）完全不會動到；測試時用環境變數把 bot 指到測試資料庫。

    python -m scripts.use_synthetic                        # 列出 bot 所在的伺服器
    python -m scripts.use_synthetic --guild-id 1234567890  # 產生 data/synthetic_test.db

然後用測試資料庫啟動 bot（bash）：
    DB_PATH=data/synthetic_test.db python main.py
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3

import httpx

from yuki.config import ROOT, load_settings

SOURCE = ROOT / "data" / "synthetic.db"
TARGET = ROOT / "data" / "synthetic_test.db"


def list_guilds() -> None:
    s = load_settings()
    resp = httpx.get("https://discord.com/api/v10/users/@me/guilds",
                     headers={"Authorization": f"Bot {s.discord_token}"}, timeout=15)
    resp.raise_for_status()
    for g in resp.json():
        print(f"{g['id']}  {g['name']}")


def build(guild_id: str) -> None:
    if not SOURCE.exists():
        raise SystemExit(f"找不到 {SOURCE}，請先跑 python -m scripts.gen_synthetic import")
    shutil.copyfile(SOURCE, TARGET)
    db = sqlite3.connect(TARGET)
    n = db.execute("UPDATE memories SET guild_id=? WHERE guild_id='synthetic'", (guild_id,)).rowcount
    db.commit()
    db.close()
    print(f"已建立 {TARGET}：{n} 筆假記憶，伺服器 {guild_id}")
    print("啟動：DB_PATH=data/synthetic_test.db python main.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guild-id", help="要放假記憶的伺服器 ID；省略則列出 bot 所在的伺服器")
    args = ap.parse_args()
    if args.guild_id:
        build(args.guild_id)
    else:
        list_guilds()


if __name__ == "__main__":
    main()
