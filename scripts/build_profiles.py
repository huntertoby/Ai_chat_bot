"""從封存區（data/archive.db）整理每個成員的人物檔案（data/profiles/<guild>/<user>.md）。

    python -m scripts.build_profiles extract --limit 3      # 先試跑 3 段看品質
    python -m scripts.build_profiles extract                # 抽取全部（可中斷，重跑接著做）
    python -m scripts.build_profiles merge                  # 把抽出的資訊整理成 .md
    python -m scripts.build_profiles merge --only 阿明      # 只整理一個人
    python -m scripts.build_profiles all                    # extract + merge（之後定期跑就是增量更新）
    python -m scripts.build_profiles stats                  # 看目前抽了多少
    python -m scripts.build_profiles remerge                # .md 全部重寫（調整 prompt 後用）
    python -m scripts.build_profiles removed --only 阿明    # 看被反駁刪掉的內容（開發規格 6.5）
    python -m scripts.build_profiles restore --only 阿明 --index 2   # 把第 2 筆放回檔案

用的是 .env 裡的 LM Studio（LMSTUDIO_BASE_URL / LLM_MODEL）；PROFILE_REASONING_EFFORT 可另外指定推理強度。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3

from yuki.archive import ArchiveDB
from yuki.config import ROOT, load_settings
from yuki.llm import LMStudioClient
from yuki.profile import revise
from yuki.profile.builder import ProfileBuilder, member_names
from yuki.profile.store import ProfileDB, ProfileFiles

ARCHIVE = ROOT / "data" / "archive.db"


def guild_of(archive: sqlite3.Connection, guild_id: str | None) -> str:
    ids = [r[0] for r in archive.execute("SELECT DISTINCT guild_id FROM messages")]
    if guild_id:
        return guild_id
    if len(ids) != 1:
        raise SystemExit(f"封存區有多個伺服器，請用 --guild-id 指定：{ids}")
    return ids[0]


async def run(args) -> None:
    s = load_settings()
    archive = sqlite3.connect(str(ARCHIVE))
    gid = guild_of(archive, args.guild_id)
    db = ProfileDB(s.profile_db_path)
    files = ProfileFiles(s.profile_dir)

    # 退出封存的人：連資訊帶檔案一起清掉，也不會再被當成 subject
    for uid in ArchiveDB(ARCHIVE).opted_out(gid):
        db.forget(gid, uid)
        files.delete(gid, uid)

    if args.cmd == "stats":
        kinds = db.kind_counts(gid)
        print("分類：", ", ".join(f"{k} {n}" for k, n in sorted(kinds.items(), key=lambda x: -x[1])) or "（還沒有）")
        for sid, name, total, pending in db.stats(gid):
            has = "✓" if files.read(gid, sid) else " "
            print(f"  {has} {name:<16} {total:>5} 條（未整理 {pending}）")
        return

    if args.cmd in ("removed", "restore"):
        restore_cmd(args, gid, files, member_names(archive, gid))
        return

    llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, timeout=max(s.llm_timeout, 600),
                         reasoning_effort=s.profile_reasoning_effort,
                         json_reasoning_effort=s.profile_reasoning_effort,
                         max_concurrency=s.llm_max_concurrency)
    builder = ProfileBuilder(archive, db, files, llm, gid, segment_chars=s.profile_segment_chars,
                             profile_limit=s.profile_max_chars)
    try:
        print(f"模型：{await llm.resolve_model()}")
        if args.cmd == "remerge":
            db.reset_merged(gid)
            for sid in builder.names:
                files.delete(gid, sid)
        if args.cmd in ("extract", "all"):
            n = await builder.extract(limit=args.limit, channel=args.channel)
            print(f"抽取完成：{n} 段")
        if args.cmd in ("merge", "all", "remerge"):
            n = await builder.merge(min_facts=args.min_facts, only=args.only)
            print(f"整理完成：{n} 人 → {files.root / gid}")
    finally:
        await llm.aclose()


def restore_cmd(args, gid: str, files: ProfileFiles, names: dict[str, str]) -> None:
    if not args.only:
        raise SystemExit("要用 --only 指定是誰（名字或 ID）")
    sid = args.only if args.only in names else next((i for i, n in names.items() if n == args.only), args.only)
    records = files.removed(gid, sid)
    if args.cmd == "removed":
        for i, r in enumerate(records, 1):
            mark = "（已還原）" if r.get("restored") else ""
            print(f"{i}. {r['at']} {r['section']} {r['line'].strip()}{mark}\n   理由：{r['reason']}")
            for e in r["evidence"]:
                print(f"   依據：[{e['day']}][{e['kind']}] {e['text']}")
        print(f"共 {len(records)} 筆")
        return
    if not args.index or not 1 <= args.index <= len(records):
        raise SystemExit(f"--index 要在 1～{len(records)} 之間")
    r = records[args.index - 1]
    if r.get("restored"):
        raise SystemExit("這筆已經還原過了")
    files.write(gid, sid, revise.insert_line(files.read(gid, sid), r["section"], r["line"]))
    r["restored"] = True
    files.rewrite_removed(gid, sid, records)
    print(f"已放回「{r['section'].lstrip('# ')}」：{r['line'].strip()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "merge", "all", "stats", "remerge", "removed", "restore"])
    ap.add_argument("--guild-id")
    ap.add_argument("--limit", type=int, help="extract 最多處理幾段（試跑用）")
    ap.add_argument("--channel", help="extract 只處理這個頻道（名稱或 ID）")
    ap.add_argument("--only", help="merge 只整理這個人；removed／restore 看這個人（名字或 ID）")
    ap.add_argument("--index", type=int, help="restore 要放回第幾筆（removed 列出的編號）")
    ap.add_argument("--min-facts", type=int, default=5, help="資訊少於這個數量的人不建檔（資料太少只會亂猜）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
