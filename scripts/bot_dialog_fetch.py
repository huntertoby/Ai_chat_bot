"""抓某個機器人（例如 Luna）加入後，公開頻道裡的所有訊息（含機器人），拿來分析大家怎麼跟它互動。

封存區（archive.db）不收機器人訊息，所以另外存成 data/bots/<名字>.jsonl。只用 REST，可以跟 Yuki 同時跑。

    python -m scripts.bot_dialog_fetch --guild-id <主伺服器ID> --bot luna
    python -m scripts.bot_dialog_fetch --guild-id <主伺服器ID> --bot luna --since 2026-10-01
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone

import discord

from yuki.archive import ArchiveDB
from yuki.archive_sync import public_channels
from yuki.config import ROOT, load_settings

log = logging.getLogger("bot_dialog_fetch")


async def find_bot(guild: discord.Guild, name: str) -> discord.Member:
    bots = [m async for m in guild.fetch_members(limit=None) if m.bot]
    key = name.lower()
    found = [m for m in bots if key in m.name.lower() or key in m.display_name.lower()]
    if not found:
        raise SystemExit(f"找不到名字含「{name}」的機器人，伺服器裡的機器人：{[m.display_name for m in bots]}")
    if len(found) > 1:
        print("找到多個：", [(m.display_name, m.id) for m in found], "，用第一個")
    return found[0]


def row(m: discord.Message, parent: str) -> dict:
    return {
        "id": m.id, "channel_id": m.channel.id, "channel": m.channel.name, "parent": parent,
        "author_id": m.author.id, "author": m.author.display_name, "bot": m.author.bot,
        "time": m.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "content": m.clean_content, "reply_to": m.reference.message_id if m.reference else None,
        "mentions": [u.id for u in m.mentions], "attachments": len(m.attachments),
        "embeds": [e.description or e.title or "" for e in m.embeds][:3],
    }


async def run(guild_id: int, bot_name: str, since: str | None, concurrency: int) -> None:
    s = load_settings()
    intents = discord.Intents.none()
    intents.members = True  # fetch_members 要求（REST 也會檢查）
    client = discord.Client(intents=intents)
    await client.login(s.discord_token)
    out_dir = ROOT / "data" / "bots"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        guild = await client.fetch_guild(guild_id)
        me = await guild.fetch_member(client.user.id)
        target = await find_bot(guild, bot_name)
        start = (datetime.fromisoformat(since).astimezone(timezone.utc) if since
                 else target.joined_at)
        print(f"目標：{target.display_name}（{target.id}），加入於 {target.joined_at:%Y-%m-%d}，從 {start:%Y-%m-%d %H:%M} 開始抓")
        role = guild.default_role
        if s.archive_role:
            role = discord.utils.get(guild.roles, name=s.archive_role) or role
        targets = await public_channels(guild, me, role, set(s.archive_exclude), archived_since=start)
        skip = ArchiveDB(ROOT / "data" / "archive.db").opted_out(str(guild_id))
        sem = asyncio.Semaphore(concurrency)
        rows: list[dict] = []

        async def one(ch, parent):
            async with sem:
                try:
                    async for m in ch.history(limit=None, after=start, oldest_first=True):
                        if str(m.author.id) not in skip:
                            rows.append(row(m, parent))
                except discord.HTTPException as e:
                    log.warning("#%s 失敗：%s", ch.name, e)

        await asyncio.gather(*(one(ch, p) for ch, p in targets))
    finally:
        await client.close()

    rows.sort(key=lambda r: r["id"])
    path = out_dir / f"{bot_name.lower()}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"bot_id": target.id, "bot_name": target.display_name,
                                      "since": start.isoformat()}}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_bot = sum(r["author_id"] == target.id for r in rows)
    print(f"完成：{len(rows):,} 則（{target.display_name} 說了 {n_bot:,} 則）→ {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guild-id", type=int, required=True)
    ap.add_argument("--bot", required=True, help="機器人名字（部分即可）")
    ap.add_argument("--since", help="從這天開始抓（YYYY-MM-DD）；預設＝機器人加入伺服器的時間")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("discord").setLevel(logging.WARNING)
    asyncio.run(run(args.guild_id, args.bot, args.since, args.concurrency))


if __name__ == "__main__":
    main()
