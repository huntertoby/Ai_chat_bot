"""統計伺服器的聊天量，決定要不要／要抓哪些頻道進封存區。不儲存任何訊息內容。

只用 REST 登入（不連 gateway），可以跟正在跑的 Yuki 同時執行，不會收到重複事件。
沒有「讀取訊息歷史」權限的頻道直接跳過、不重試——大量 403 會讓 Discord 暫時封鎖 IP。

    python -m scripts.archive_scan --guild-id <主伺服器ID>            # 快速：頻道、權限、時間跨度
    python -m scripts.archive_scan --guild-id <主伺服器ID> --full     # 完整：逐則數數（要很久）

報告寫到 data/archive_scan_<guild>.json：只有數量、日期、作者 ID 與名稱，沒有訊息內容。
Discord 沒有「頻道有幾則訊息」的 API，所以 --full 實際上要把每則訊息抓一遍（只數不存），
花的時間就等於之後真正抓取的時間，順便量出這台機器的實際速度。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import discord

from yuki.config import ROOT, load_settings

log = logging.getLogger("archive_scan")

READABLE = (discord.TextChannel, discord.VoiceChannel, discord.StageChannel, discord.Thread)


@dataclass
class ChannelStat:
    id: int
    name: str
    kind: str
    parent: str = ""
    readable: bool = True
    skip_reason: str = ""
    first: str = ""            # 最早一則的日期
    last: str = ""             # 最新一則的日期
    messages: int = 0          # --full 才有：人類訊息數
    bot_messages: int = 0
    seconds: float = 0.0
    authors: Counter = field(default_factory=Counter)


def day(dt: datetime | None) -> str:
    return dt.astimezone().strftime("%Y-%m-%d") if dt else ""


async def list_channels(client: discord.Client, guild: discord.Guild, me: discord.Member):
    """所有可能有訊息的地方：文字／語音／舞台頻道、進行中與已封存的討論串（含論壇貼文）。"""
    channels = await guild.fetch_channels()
    by_id = {c.id: c for c in channels}
    out: list[tuple[discord.abc.Messageable, ChannelStat]] = []

    def perms_of(ch) -> discord.Permissions:
        base = by_id.get(ch.parent_id) if isinstance(ch, discord.Thread) else ch
        return base.permissions_for(me) if base else discord.Permissions.none()

    def add(ch, kind: str, parent: str = ""):
        p = perms_of(ch)
        st = ChannelStat(ch.id, ch.name, kind, parent)
        if not (p.view_channel and p.read_message_history):
            st.readable, st.skip_reason = False, "沒有查看頻道或讀取訊息歷史權限"
        out.append((ch, st))

    for ch in channels:
        if isinstance(ch, (discord.TextChannel, discord.VoiceChannel, discord.StageChannel)):
            add(ch, type(ch).__name__.replace("Channel", "").lower())

    threads = {t.id: t for t in await guild.active_threads()}
    for ch in channels:
        if not isinstance(ch, (discord.TextChannel, discord.ForumChannel)):
            continue
        p = ch.permissions_for(me)
        if not (p.view_channel and p.read_message_history):
            continue  # 父頻道看不到，就不去要它的封存討論串（避免 403）
        try:
            async for t in ch.archived_threads(limit=None):
                threads[t.id] = t
            if p.manage_threads and isinstance(ch, discord.TextChannel):
                async for t in ch.archived_threads(limit=None, private=True):
                    threads[t.id] = t
        except discord.Forbidden:
            log.info("略過 #%s 的封存討論串（沒有權限）", ch.name)
    for t in threads.values():
        parent = by_id.get(t.parent_id)
        kind = "forum_post" if isinstance(parent, discord.ForumChannel) else "thread"
        add(t, kind, parent.name if parent else str(t.parent_id))
    return out


async def quick(ch, st: ChannelStat) -> None:
    """只抓最早與最新各一則，得到時間跨度（2 次請求）。"""
    async for m in ch.history(limit=1, oldest_first=True):
        st.first = day(m.created_at)
    async for m in ch.history(limit=1):
        st.last = day(m.created_at)


async def full(ch, st: ChannelStat, progress: dict) -> None:
    t0 = time.perf_counter()
    async for m in ch.history(limit=None, oldest_first=True):
        if not st.first:
            st.first = day(m.created_at)
        st.last = day(m.created_at)
        if m.author.bot:
            st.bot_messages += 1
        else:
            st.messages += 1
            st.authors[f"{m.author.id}|{m.author.display_name}"] += 1
        progress["n"] += 1
        if progress["n"] % 10000 == 0:
            rate = progress["n"] / (time.perf_counter() - progress["t0"])
            log.info("已數 %d 則（%.0f 則/秒），目前 #%s", progress["n"], rate, st.name)
    st.seconds = time.perf_counter() - t0


async def scan(guild_id: int, do_full: bool, concurrency: int) -> None:
    s = load_settings()
    client = discord.Client(intents=discord.Intents.none())
    await client.login(s.discord_token)
    try:
        guild = await client.fetch_guild(guild_id)
        me = await guild.fetch_member(client.user.id)
        targets = await list_channels(client, guild, me)
        log.info("%s：%d 個頻道／討論串，其中 %d 個可讀", guild.name, len(targets),
                 sum(st.readable for _, st in targets))

        sem = asyncio.Semaphore(concurrency)
        progress = {"n": 0, "t0": time.perf_counter()}

        async def one(ch, st: ChannelStat):
            if not st.readable:
                return
            async with sem:
                try:
                    await (full(ch, st, progress) if do_full else quick(ch, st))
                except discord.Forbidden:
                    st.readable, st.skip_reason = False, "403（權限不足）"
                except discord.HTTPException as e:
                    st.skip_reason = f"HTTP {e.status}"
                    log.warning("#%s 失敗：%s", st.name, e)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(ch, st) for ch, st in targets))
        elapsed = time.perf_counter() - t0
    finally:
        await client.close()

    stats = [st for _, st in targets]
    report(guild.name, guild_id, stats, do_full, elapsed)


def report(name: str, guild_id: int, stats: list[ChannelStat], do_full: bool, elapsed: float) -> None:
    readable = [st for st in stats if st.readable and st.first]
    authors: Counter = Counter()
    for st in stats:
        authors.update(st.authors)
    total = sum(st.messages for st in stats)
    bots = sum(st.bot_messages for st in stats)
    firsts = [st.first for st in readable]

    print(f"\n===== {name}（{'完整統計' if do_full else '快速掃描'}，{elapsed:.0f} 秒）=====")
    print(f"頻道／討論串 {len(stats)} 個：有訊息且可讀 {len(readable)}、"
          f"沒權限 {sum(not st.readable for st in stats)}、空的 {sum(st.readable and not st.first for st in stats)}")
    if firsts:
        print(f"時間跨度：{min(firsts)} ～ {max(st.last for st in readable)}")
    if do_full:
        rate = (total + bots) / elapsed if elapsed else 0
        print(f"人類訊息 {total:,} 則、機器人訊息 {bots:,} 則；抓取速度約 {rate:,.0f} 則/秒")
        print("\n訊息最多的頻道：")
        for st in sorted(readable, key=lambda x: -x.messages)[:15]:
            where = f"{st.parent} › " if st.parent else ""
            print(f"  {st.messages:>9,}  {where}#{st.name}  ({st.first} ～ {st.last})")
        print(f"\n發言最多的人（共 {len(authors)} 人）：")
        for key, n in authors.most_common(15):
            print(f"  {n:>9,}  {key.split('|', 1)[1]}")
    else:
        print("\n最近仍有活動的頻道（依最新訊息日期）：")
        for st in sorted(readable, key=lambda x: x.last, reverse=True)[:20]:
            where = f"{st.parent} › " if st.parent else ""
            print(f"  {st.last}  {where}#{st.name}（{st.kind}，始於 {st.first}）")
    skipped = [st for st in stats if not st.readable]
    if skipped:
        print(f"\n略過 {len(skipped)} 個（{skipped[0].skip_reason} 等）："
              + "、".join(f"#{st.name}" for st in skipped[:20]) + (" …" if len(skipped) > 20 else ""))

    out = ROOT / "data" / f"archive_scan_{guild_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "guild": name, "full": do_full, "seconds": round(elapsed, 1),
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channels": [{**asdict(st), "authors": None} for st in stats],
        "authors": [{"id": k.split("|", 1)[0], "name": k.split("|", 1)[1], "messages": n}
                    for k, n in authors.most_common()],
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n報告（無訊息內容）：{out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guild-id", type=int, required=True)
    ap.add_argument("--full", action="store_true", help="逐則數數（時間等於完整抓取）")
    ap.add_argument("--concurrency", type=int, default=3, help="同時掃幾個頻道（太多會拖慢 Yuki）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("discord").setLevel(logging.WARNING)
    asyncio.run(scan(args.guild_id, args.full, args.concurrency))


if __name__ == "__main__":
    main()
