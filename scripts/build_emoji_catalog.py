"""建立群組自訂表情／貼圖的圖鑑（開發規格 6.14 第 1 點）。一次性工具，Yuki 本體不用。

    python -m scripts.build_emoji_catalog --limit 5        # 先試 5 張
    python -m scripts.build_emoji_catalog                  # 全部
    python -m scripts.build_emoji_catalog --no-llm         # 只抓清單＋算次數
    python -m scripts.build_emoji_catalog --only-missing   # 已有 meaning/looks 的不重做

輸出 data/emoji/<伺服器ID>.json 與 data/emoji/<伺服器ID>-檢查.html（data/ 不進 git）。
Discord 只用 REST（login + fetch_*），不連 gateway；LM Studio 一次一個請求、每張之間停一下。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import discord
import httpx

from yuki import emoji_catalog as cat
from yuki.archive import ArchiveDB
from yuki.config import ROOT, load_settings
from yuki.llm import LMStudioClient, LLMError
from yuki.vision import fetch_image

log = logging.getLogger("emoji_catalog")
ARCHIVE = ROOT / "data" / "archive.db"
OUT_DIR = ROOT / "data" / "emoji"
YUKI_LOG = ROOT / "logs" / "yuki.log"
PAUSE = 2.0

MEANING_SYSTEM = (
    "你會看到某個自訂表情在聊天群組裡被使用的幾段對話，標「>>」的那一行是用了該表情的訊息，"
    "表情在對話中以冒號包住的名稱出現。請歸納這個表情在這個群組裡實際是在什麼情況下用的，"
    "只寫一句話（30 字以內），用台灣繁體中文，描述情境或語氣，不要引用對話原文，不要提到人名，"
    "不要解釋。如果看不出規律，就寫最常見的那種情境。"
)
LOOKS_PROMPT = "用一句話（30 字以內，台灣繁體中文）描述這張圖長什麼樣子：主體、表情、動作。只回答描述本身。"


def clean(text: str) -> str:
    text = (text or "").strip().replace("\n", " ").strip("「」\"' ")
    return text[:120]


def yuki_busy(window: float = 20.0) -> bool:
    """Yuki 的 log 最近有送聊天請求（有人在跟她聊）就先讓路。"""
    try:
        with open(YUKI_LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            lines = f.read().decode("utf-8", "ignore").splitlines()[-40:]
    except OSError:
        return False
    now = datetime.now()
    for ln in lines:
        if "chat/completions" in ln:
            try:
                if (now - datetime.strptime(ln[:19], "%Y-%m-%d %H:%M:%S")).total_seconds() < window:
                    return True
            except ValueError:
                pass
    return False


async def wait_idle(enabled: bool) -> None:
    waited = 0
    while enabled and yuki_busy() and waited < 300:
        if waited == 0:
            log.info("Yuki 正在回話，先等一下")
        await asyncio.sleep(10)
        waited += 10


async def fetch_items(token: str, guild_id: int) -> list[dict]:
    client = discord.Client(intents=discord.Intents.none())
    try:
        await client.login(token)  # 只用 REST，不連 gateway
        guild = await client.fetch_guild(guild_id)
        emojis = await guild.fetch_emojis()
        stickers = await guild.fetch_stickers()
    finally:
        await client.close()
    items = [{"id": str(e.id), "name": e.name, "animated": bool(e.animated), "kind": "emoji",
              "url": f"https://cdn.discordapp.com/emojis/{e.id}.{'gif' if e.animated else 'png'}?size=96"}
             for e in emojis]
    for s in stickers:
        items.append({"id": str(s.id), "name": s.name, "animated": s.format.name in ("apng", "gif"),
                      "kind": "sticker", "format": s.format.name, "url": str(s.url)})
    return items


async def describe_meaning(llm: LMStudioClient, samples) -> str:
    text = cat.samples_text(samples)
    return clean(await llm.chat([{"role": "system", "content": MEANING_SYSTEM},
                                 {"role": "user", "content": text}], temperature=0.2, max_tokens=200))


async def describe_looks(llm: LMStudioClient, http: httpx.AsyncClient, item: dict) -> str:
    if item.get("format") == "lottie":
        return ""
    url = await fetch_image(item["url"], 256, client=http)
    if not url:
        return ""
    msg = [{"role": "user", "content": [{"type": "text", "text": LOOKS_PROMPT},
                                        {"type": "image_url", "image_url": {"url": url}}]}]
    return clean(await llm.chat(msg, temperature=0.2, max_tokens=200))


def save(items: list[dict], path: Path) -> None:
    items = sorted(items, key=cat.sort_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    path.with_name(path.stem + "-檢查.html").write_text(cat.render_html(items), encoding="utf-8")


async def run(args) -> None:
    s = load_settings()
    gid = args.guild or s.profile_fallback_guild
    if not gid:
        raise SystemExit("請用 --guild 指定伺服器 ID")
    path = OUT_DIR / f"{gid}.json"
    old = {i["id"]: i for i in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}

    items = await fetch_items(s.discord_token, int(gid))
    log.info("抓到 %d 個表情、%d 張貼圖", sum(i["kind"] == "emoji" for i in items),
             sum(i["kind"] == "sticker" for i in items))

    arch = ArchiveDB(ARCHIVE)
    opted = arch.opted_out(gid)
    usage = cat.count_uses(cat.usage_rows(arch.db, gid, opted))
    for it in items:
        u = usage.get(it["id"], {"uses": 0, "last_used": "", "msgs": []})
        it["uses"], it["last_used"], it["msgs"] = u["uses"], u["last_used"], u["msgs"]
        it.update(meaning="", looks="", samples=0)
    items.sort(key=cat.sort_key)  # 常用的先做，--limit 試跑看得到有意義的
    if args.limit:
        items = items[:args.limit]

    done: dict[str, dict] = {}
    llm = http = None
    if not args.no_llm:
        llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, timeout=max(s.llm_timeout, 300),
                             reasoning_effort="none", max_concurrency=1)
        http = httpx.AsyncClient()
    t0 = time.monotonic()
    try:
        for n, it in enumerate(items, 1):
            msgs = it.pop("msgs")
            item = cat.merge_item(it, old.get(it["id"]), only_missing=args.only_missing)
            if llm:
                if not item["meaning"] and msgs:
                    samples = cat.build_samples(arch.db, msgs, opted)
                    item["samples"] = len(samples)
                    if samples:
                        await wait_idle(not args.no_yield)
                        try:
                            item["meaning"] = await describe_meaning(llm, samples)
                        except LLMError as e:
                            log.warning("meaning 失敗（第 %d 張）：%s", n, e)
                        await asyncio.sleep(PAUSE)
                if not item["looks"]:
                    await wait_idle(not args.no_yield)
                    try:
                        item["looks"] = await describe_looks(llm, http, item)
                    except LLMError as e:
                        log.warning("looks 失敗（第 %d 張）：%s", n, e)
                    await asyncio.sleep(PAUSE)
            done[item["id"]] = item
            # 沒處理到的沿用舊資料，中斷時檔案也完整
            save(list(done.values()) + [o for i, o in old.items() if i not in done], path)
            if n % 10 == 0 or n == len(items):
                log.info("進度 %d/%d，已 %.0f 秒", n, len(items), time.monotonic() - t0)
    finally:
        if llm:
            await llm.aclose()
            await http.aclose()
    log.info("完成：%s（耗時 %.0f 秒）", path.name, time.monotonic() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guild")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--no-yield", action="store_true", help="不要在 Yuki 回話時暫停")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
