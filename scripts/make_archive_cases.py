"""從真實封存區出題（開發規格 6.4 驗收）：隨機挑有內容的訊息，叫 LLM 出一題「答案就在這則」的問題。

    python -m scripts.make_archive_cases --n 80      # → eval/archive_cases.jsonl
    python -m scripts.eval_agentic --cases eval/archive_cases.jsonl --real

一半「直接問」（會用到原句的詞），一半「換句話說」（刻意不用原句的關鍵詞），測全文檢索的弱點。
題目裡有群組成員的真實聊天內容：eval/archive_cases.jsonl 不要外流（已在 .gitignore）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re

_NOISE = re.compile(r"<a?:\w+:\d+>|@\S+|https?://\S+")


def informative(text: str) -> bool:
    """拿掉 @、表情、網址後還有 20 個字以上的內容，才拿來出題（只有 @ 一串人的訊息沒有答案）。"""
    return len(_NOISE.sub("", text).strip()) >= 20

from yuki.archive import ArchiveDB
from yuki.bot import ARCHIVE_PATH
from yuki.config import ROOT, load_settings
from yuki.llm import LMStudioClient

SYSTEM = """你幫忙出測驗題。給你一段 Discord 群組的聊天，其中標 → 的那則是「答案」。
出一個群組成員可能會問聊天機器人 Yuki 的問題，答案要靠那則訊息（可以搭配前後文）才答得出來。
- 問題像真人在 Discord 問：口語、短、繁體中文，可以叫 Yuki。
- 問的是事實（誰、什麼、什麼時候、為什麼），不要問感想。
- **答案必須寫在 → 那則訊息裡**（前後文只是幫你看懂），答案在別則的話就 skip。
- 問題要能獨立看懂：要講出是誰、在講什麼事，不能用「那個」「為什麼不用加」這種要看上下文才懂的問法。
- 那則訊息如果只是沒內容的閒聊、梗、單字、表情，或是一般常識（不用看聊天紀錄也答得出來），就把 skip 設成 true。
- style 是「直接」：可以用原句的詞。style 是「換句話說」：刻意不用原句的關鍵詞，改用同義詞或描述。
- answer 用一句話寫出正確答案。"""

SCHEMA = {"type": "object", "properties": {
    "skip": {"type": "boolean"}, "question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["skip", "question", "answer"], "additionalProperties": False}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    s = load_settings()
    llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, temperature=0.7, timeout=s.llm_timeout,
                         reasoning_effort="none", json_reasoning_effort="none")
    db = ArchiveDB(ARCHIVE_PATH).db
    gid = s.profile_fallback_guild or db.execute("SELECT guild_id FROM messages LIMIT 1").fetchone()[0]
    pool = [(i, c) for i, c, t in db.execute(
        "SELECT id, channel_id, content FROM messages WHERE guild_id=? AND length(content) >= 20"
        " AND content NOT LIKE '%http%'", (gid,)) if informative(t)]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    members = [{"id": a, "name": n} for a, n in db.execute(
        "SELECT author_id, author_name FROM messages WHERE guild_id=? GROUP BY author_id", (gid,))]
    out = ROOT / "eval" / "archive_cases.jsonl"
    made = 0
    with out.open("w", encoding="utf-8") as f:
        for mid, cid in pool:
            if made >= args.n:
                break
            rows = db.execute("SELECT id, created_at, author_name, content FROM messages WHERE channel_id=?"
                              " AND id BETWEEN ? AND ? ORDER BY id", (cid, mid - 10**12, mid + 10**12)).fetchall()
            i = next(k for k, r in enumerate(rows) if r[0] == mid)
            ctx = rows[max(0, i - 4): i + 3]
            target = rows[i]
            style = "直接" if made % 2 == 0 else "換句話說"
            chat = "\n".join(("→ " if r[0] == mid else "  ") + f"{r[1][:16]} {r[2]}：{r[3][:200]}" for r in ctx)
            try:
                r = await llm.chat_json([{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": f"style：{style}\n\n{chat}"}], SCHEMA)
            except Exception as e:  # noqa: BLE001
                print("出題失敗", e)
                continue
            if r.get("skip") or not r.get("question"):
                continue
            made += 1
            case = {"guild_id": gid, "author": {"id": "eval", "name": "hunterchiao"}, "members": members,
                    "query": r["question"], "expected": [str(mid)], "memory": f"{target[2]}（{target[1][:10]}）：{target[3]}；答案：{r['answer']}",
                    "now": "2026-11-17", "style": style}
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{made}/{args.n}] {style}｜{r['question']}", flush=True)
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
