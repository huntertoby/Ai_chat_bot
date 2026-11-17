"""Agentic RAG 評估（開發規格 6.4）：讓 Yuki 用 search_chat / read_chat 自己查，看有沒有查到正確那筆。

    python -m scripts.eval_agentic                       # 舊的 200 題（合成記憶，跟 eval_retrieval 同一份）
    python -m scripts.eval_agentic --limit 20            # 先跑 20 題看看
    python -m scripts.eval_agentic --cases eval/archive_cases.jsonl --real   # 真實封存區的題目

合成記憶會轉成一個暫存的「封存區」（每筆記憶 = 一則訊息，發言者 = 記憶的主角），用同一套全文檢索。
命中 = 正確那筆出現在 Yuki 這次查到的結果裡（search_chat 或 read_chat）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

from yuki import prompts
from yuki.archive import ArchiveDB, ArchivedMessage
from yuki.config import ROOT, load_settings
from yuki.llm import LMStudioClient
from yuki.memory.text_index import add_words
from yuki.tools.archive import TOOL_SPECS, ArchiveTools
from yuki.bot import ARCHIVE_PATH

SYNTH_FTS = ROOT / "data" / "synthetic_fts.db"


def synthetic_archive() -> tuple[ArchiveDB, dict[str, int]]:
    """合成記憶 → 暫存封存區。回傳 (db, 記憶 ID → 訊息 ID)。"""
    src = sqlite3.connect(ROOT / "data" / "synthetic.db")
    rows = src.execute("SELECT id, subject_id, subject_name, text, created_at FROM memories ORDER BY created_at").fetchall()
    ids = {mid: i for i, (mid, *_rest) in enumerate(rows, start=1)}
    fresh = not SYNTH_FTS.exists()
    db = ArchiveDB(SYNTH_FTS)
    if fresh:
        add_words(list({r[2] for r in rows}))
        db.add_many([ArchivedMessage(ids[mid], "synthetic", "c1", "閒聊", "", sid or "", name or "", ts, text, None, 0)
                     for mid, sid, name, text, ts in rows])
        db.commit()
    return db, ids


async def run_case(llm, db, case: dict, guild: str, *, rounds: int) -> dict:
    today = date.fromisoformat(case.get("now") or date.today().isoformat())
    members = case.get("members") or []
    add_words([m["name"] for m in members])
    people = {m["name"].lower(): m["id"] for m in members}

    def resolve(name: str) -> set[str]:
        n = name.strip().lower()
        return {people[n]} if n in people else {v for k, v in people.items() if n and n in k}

    tools = ArchiveTools(db.db, guild, resolve_person=resolve, profile=lambda n: "",
                         now=datetime.combine(today, datetime.min.time()))
    calls: list[tuple[str, dict]] = []

    async def call(name, args):
        calls.append((name, args))
        return await tools.call(name, args)

    system = prompts.reply_system("Yuki", [], None, today, archive=True)
    author = case["author"]["name"]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"[{author}] {case['query']}"}]
    t = time.time()
    reply = await llm.chat_tools(messages, TOOL_SPECS, call, max_rounds=rounds)
    return {"query": case["query"], "expected": case["expected"], "seen": tools.seen, "calls": calls,
            "reply": reply, "seconds": round(time.time() - t, 2)}


JUDGE = """你是評分員。判斷「回覆」有沒有正確講到「參考資料」的重點（人、事、大致內容對得上就算，不用一字不差）。
只回一個字：是 或 否。"""


async def judge(llm, memory: str, reply: str) -> bool:
    out = await llm.chat([{"role": "system", "content": JUDGE},
                          {"role": "user", "content": f"參考資料：{memory}\n\n回覆：{reply}"}])
    return out.strip().startswith("是")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=ROOT / "eval" / "synthetic_cases.jsonl")
    ap.add_argument("--real", action="store_true", help="題目用真實封存區（expected 是訊息 ID）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-judge", action="store_true", help="不用 LLM 判斷回覆內容對不對")
    args = ap.parse_args()

    s = load_settings()
    llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, temperature=0.3, max_tokens=s.llm_max_tokens,
                         timeout=s.llm_timeout, reasoning_effort=s.llm_reasoning_effort, max_concurrency=1)
    cases = [json.loads(l) for l in args.cases.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        cases = cases[: args.limit]
    if args.real:
        db, idmap, guild = ArchiveDB(ARCHIVE_PATH), None, None
    else:
        db, idmap = synthetic_archive()
        guild = "synthetic"

    out_path = args.out or ROOT / "data" / "synthetic" / f"eval_agentic_{args.cases.stem}.jsonl"
    results = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            g = guild or case["guild_id"]
            if not g.isdigit():  # 題目檔裡的 <主伺服器ID> 之類的佔位：用 .env 的主伺服器
                g = s.profile_fallback_guild
            r = await run_case(llm, db, case, g, rounds=args.rounds)
            if args.real:
                # 找到同一段對話就算（答案常在前後幾則）：正解前後各 3 則
                want = set()
                for e in case["expected"]:
                    row = db.db.execute("SELECT channel_id FROM messages WHERE id=?", (int(e),)).fetchone()
                    if row:
                        want |= {r[0] for r in db.db.execute(
                            "SELECT id FROM (SELECT id FROM messages WHERE channel_id=? AND id<=? ORDER BY id DESC LIMIT 4)"
                            " UNION SELECT id FROM (SELECT id FROM messages WHERE channel_id=? AND id>? ORDER BY id LIMIT 3)",
                            (row[0], int(e), row[0], int(e)))}
            else:
                want = {idmap[e] for e in case["expected"] if e in idmap}
            r["hit"] = bool(want & set(r["seen"]))
            r["searched"] = bool(r["calls"])
            if not args.no_judge and case.get("memory"):
                r["correct"] = await judge(llm, case["memory"], r["reply"])
            results.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            hits = sum(x["hit"] for x in results)
            ok = sum(x.get("correct", False) for x in results)
            print(f"[{i}/{len(cases)}] 命中{'✓' if r['hit'] else '✗'} 答對{'✓' if r.get('correct') else '✗'}"
                  f" 查了 {len(r['calls'])} 次 {r['seconds']}s｜累計命中 {hits / i:.0%} 答對 {ok / i:.0%}", flush=True)

    n = len(results)
    searched = [r for r in results if r["searched"]]
    if not args.no_judge:
        ok = sum(r.get("correct", False) for r in results)
        print(f"\n答對 {ok}/{n} = {ok / n:.1%}（有查的題目裡答對 "
              f"{sum(r.get('correct', False) for r in searched) / max(len(searched), 1):.1%}）")
        print("沒查的題目（多半是問句在叫別人，Yuki 判斷不是在問她）：",
              [r["query"][:30] for r in results if not r["searched"]][:10])
    print(f"\n命中 {sum(r['hit'] for r in results)}/{n} = {sum(r['hit'] for r in results) / n:.1%}")
    print(f"有查 {sum(r['searched'] for r in results) / n:.0%}；平均查 {sum(len(r['calls']) for r in results) / n:.1f} 次；"
          f"平均 {sum(r['seconds'] for r in results) / n:.1f} 秒；結果：{out_path}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
