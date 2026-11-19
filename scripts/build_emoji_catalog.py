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
from yuki.brain import to_traditional
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
    "你會看到某個自訂表情的外觀描述，以及它在聊天群組裡被使用的幾段對話，標「>>」的那一行是用了該表情的訊息，"
    "表情在對話中以冒號包住的名稱出現。請歸納群組成員用「這一張」的具體情境與語氣：它替訊息加了什麼"
    "（是哪一種情緒、針對誰或什麼事、是認真還是玩笑），要能跟其他表情區分。"
    "只寫一句短話（25 字以內），台灣繁體中文，直接寫情境，不要用「用於」「表示」開頭，"
    "不要引用對話原文，不要提到人名。不要寫成泛稱的無奈、吐槽、無語這類萬用說法，除非這張真的只有這個用途。"
)
REVISE_SYSTEM = (
    "下面是一批自訂表情，每個有編號、外觀、初步的用法說明、幾段使用時的短對話片段。"
    "初步說明很多太像，分不出差別。請逐一修訂，讓用法不同的表情說明彼此看得出差異："
    "根據外觀與片段寫出這張特有的情境、對象或語氣，每個 25 字以內，台灣繁體中文，不要用「用於」開頭。"
    "如果兩張真的用法一樣，就在說明裡直接寫出「與第 N 號相同用法」。"
    "只輸出 JSON，items 陣列裡每個編號都要有，欄位 i 是編號、meaning 是說明。"
)
LOOKS_PROMPT = "用一句話（30 字以內，台灣繁體中文）描述這張圖長什麼樣子：主體、表情、動作。只回答描述本身。"


def clean(text: str) -> str:
    text = to_traditional(text or "").strip().replace("\n", " ").strip("「」\"' ")
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


async def describe_meaning(llm: LMStudioClient, samples, looks: str = "") -> str:
    text = (f"外觀：{looks}\n\n" if looks else "") + cat.samples_text(samples)
    return clean(await llm.chat([{"role": "system", "content": MEANING_SYSTEM},
                                 {"role": "user", "content": text}], temperature=0.2, max_tokens=200))


async def revise_batch(llm: LMStudioClient, entries: list[dict]) -> dict[int, str]:
    """entries 見 cat.revise_prompt；回傳 {編號: 新說明}。"""
    res = await llm.chat_json([{"role": "system", "content": REVISE_SYSTEM},
                               {"role": "user", "content": cat.revise_prompt(entries)}],
                              cat.REVISE_SCHEMA, name="revise", temperature=0.2, max_tokens=3000,
                              reasoning_effort="none")
    out = {}
    for r in (res or {}).get("items", []):
        if isinstance(r, dict) and isinstance(r.get("i"), int) and r.get("meaning"):
            out[r["i"]] = clean(str(r["meaning"]))
    return out


async def describe_looks(llm: LMStudioClient, http: httpx.AsyncClient, item: dict) -> str:
    if item.get("format") == "lottie":
        return ""
    url = await fetch_image(item["url"], 256, client=http)
    if not url:
        return ""
    msg = [{"role": "user", "content": [{"type": "text", "text": LOOKS_PROMPT},
                                        {"type": "image_url", "image_url": {"url": url}}]}]
    return clean(await llm.chat(msg, temperature=0.2, max_tokens=200))


async def revise_top(llm, arch, opted, items: list[dict], all_msgs, args) -> None:
    """前 N 名（依使用次數）一起看過：說明太像的請模型改到分得出來。"""
    top = sorted((i for i in items if i["uses"] > 0 and i["meaning"]), key=cat.sort_key)[:args.revise_top]
    for start in range(0, len(top), args.batch):
        chunk = top[start:start + args.batch]
        entries = []
        for k, it in enumerate(chunk, 1):
            snippets = cat.snippets(cat.build_samples(arch.db, all_msgs.get(it["id"], []), opted, 3))
            entries.append({"i": k, "looks": it["looks"], "meaning": it["meaning"], "snippets": snippets})
        await wait_idle(not args.no_yield)
        try:
            new = await revise_batch(llm, entries)
        except LLMError as e:
            log.warning("修訂失敗：%s", e)
            continue
        for k, it in enumerate(chunk, 1):
            if k in new:
                it["meaning"] = new[k]
        log.info("修訂 %d 張，收到 %d 則", len(chunk), len(new))
        await asyncio.sleep(PAUSE)


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
    all_msgs: dict[str, list[int]] = {}
    llm = http = None
    if not args.no_llm:
        llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, timeout=max(s.llm_timeout, 300),
                             reasoning_effort="none", max_concurrency=1)
        http = httpx.AsyncClient()
    t0 = time.monotonic()
    try:
        for n, it in enumerate(items, 1):
            msgs = it.pop("msgs")
            all_msgs[it["id"]] = msgs
            item = cat.merge_item(it, old.get(it["id"]), only_missing=args.only_missing or args.redo_meaning)
            if args.redo_meaning:
                item["meaning"] = ""
            item["looks"] = to_traditional(item["looks"])
            if llm:
                if not item["looks"]:
                    await wait_idle(not args.no_yield)
                    try:
                        item["looks"] = await describe_looks(llm, http, item)
                    except LLMError as e:
                        log.warning("looks 失敗（第 %d 張）：%s", n, e)
                    await asyncio.sleep(PAUSE)
                if not item["meaning"] and msgs:
                    samples = cat.build_samples(arch.db, msgs, opted)
                    item["samples"] = len(samples)
                    if samples:
                        await wait_idle(not args.no_yield)
                        try:
                            item["meaning"] = await describe_meaning(llm, samples, item["looks"])
                        except LLMError as e:
                            log.warning("meaning 失敗（第 %d 張）：%s", n, e)
                        await asyncio.sleep(PAUSE)
            done[item["id"]] = item
            # 沒處理到的沿用舊資料，中斷時檔案也完整
            save(list(done.values()) + [o for i, o in old.items() if i not in done], path)
            if n % 10 == 0 or n == len(items):
                log.info("進度 %d/%d，已 %.0f 秒", n, len(items), time.monotonic() - t0)
        if llm and args.revise_top:
            await revise_top(llm, arch, opted, list(done.values()), all_msgs, args)
            save(list(done.values()) + [o for i, o in old.items() if i not in done], path)
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
    ap.add_argument("--redo-meaning", action="store_true", help="重做所有 meaning（looks 沿用，只轉繁體）")
    ap.add_argument("--revise-top", type=int, default=0, help="前 N 名再一起修訂一次")
    ap.add_argument("--batch", type=int, default=30, help="修訂時每次送幾張")
    ap.add_argument("--no-yield", action="store_true", help="不要在 Yuki 回話時暫停")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
