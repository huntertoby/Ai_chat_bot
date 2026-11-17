"""把伺服器公開頻道的歷史訊息抓進封存區（data/archive.db）。

「公開」= 某個身分組看得到的頻道（預設 @everyone；很多伺服器的 @everyone 什麼都看不到，
這時用 --role 指定一般成員的身分組），以及其中的公開討論串、論壇貼文。私人討論串一律不收。
機器人訊息、已退出封存（/封存退出）的人不收。

只用 REST 登入（不連 gateway），可以跟正在跑的 Yuki 同時執行。可以隨時中斷，重跑會從每個頻道
上次抓到的地方接著抓。

    python -m scripts.archive_fetch --guild-id <主伺服器ID> --role <身分組名稱> --dry-run
    python -m scripts.archive_fetch --guild-id <主伺服器ID> --role <身分組名稱> --exclude data
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time

import discord

from yuki.archive import ArchiveDB
from yuki.archive_sync import fetch_channel, public_channels
from yuki.config import ROOT, load_settings

log = logging.getLogger("archive_fetch")
DB_PATH = ROOT / "data" / "archive.db"


async def run(guild_id: int, role_name: str | None, exclude: set[str], dry_run: bool,
              concurrency: int) -> None:
    s = load_settings()
    client = discord.Client(intents=discord.Intents.none())
    await client.login(s.discord_token)
    db = ArchiveDB(DB_PATH)
    gid = str(guild_id)
    try:
        guild = await client.fetch_guild(guild_id)
        me = await guild.fetch_member(client.user.id)
        role = guild.default_role
        if role_name:
            role = discord.utils.get(guild.roles, name=role_name)
            if role is None:
                raise SystemExit(f"找不到身分組「{role_name}」，有：{[r.name for r in guild.roles]}")
        targets = await public_channels(guild, me, role, exclude)
        print(f"{guild.name}：{role.name} 看得到的頻道／討論串 {len(targets)} 個（排除 {sorted(exclude) or '無'}）")
        for ch, parent in targets:
            print(f"  {parent + ' › ' if parent else ''}#{ch.name}")
        if dry_run:
            return

        skip = db.opted_out(gid)
        stats = {"seen": 0, "saved": 0, "t0": time.perf_counter()}
        sem = asyncio.Semaphore(concurrency)

        async def one(ch, parent):
            async with sem:
                try:
                    await fetch_channel(ch, parent, gid, db, skip, stats)
                except discord.Forbidden:
                    log.warning("#%s：403，略過（不重試）", ch.name)
                except discord.HTTPException as e:
                    log.warning("#%s 失敗：%s（重跑會接著抓）", ch.name, e)

        await asyncio.gather(*(one(ch, p) for ch, p in targets))
        elapsed = time.perf_counter() - stats["t0"]
    finally:
        await client.close()

    print(f"\n完成：新存 {stats['saved']:,} 則（看過 {stats['seen']:,} 則，{elapsed:.0f} 秒）→ {DB_PATH}")
    rows = db.counts(gid)
    print(f"封存區目前共 {sum(r[1] for r in rows):,} 則：")
    for name, n, first, last in rows[:15]:
        print(f"  {n:>8,}  #{name}  ({first[:10]} ～ {last[:10]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guild-id", type=int, required=True)
    ap.add_argument("--role", help="「公開」= 這個身分組看得到；預設 @everyone")
    ap.add_argument("--exclude", nargs="*", default=[], help="不收的頻道名稱（討論串則看所屬頻道）")
    ap.add_argument("--dry-run", action="store_true", help="只列出會抓哪些頻道")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("discord").setLevel(logging.WARNING)
    asyncio.run(run(args.guild_id, args.role, set(args.exclude), args.dry_run, args.concurrency))


if __name__ == "__main__":
    main()
