"""檢索評估：每改一次檢索就跑一次，用數字確認有沒有進步。

eval/cases.jsonl 每行一題：
{"guild_id": "...", "author": {"id": "...", "name": "Toby"}, "query": "Hunter 生日是哪天",
 "expected": ["e9acedaf"], "now": "2024-03-13"}

"now" 是發問當天（解析「昨天」「上個月」用），可省略，省略就用今天。

    python -m scripts.eval_retrieval                 # 完整管線
    python -m scripts.eval_retrieval --no-rerank     # 比較：不 rerank
    python -m scripts.eval_retrieval --no-time       # 比較：忽略時間說法
    python -m scripts.eval_retrieval --llm-rerank always            # LLM 重排：time（預設）/ always / off
    python -m scripts.eval_retrieval --no-focus --no-fallback       # 比較：rerank 用原句、不保底
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import date
from pathlib import Path

from yuki.config import ROOT, load_settings
from yuki.embedding import build_embedder
from yuki.memory.store import MemoryStore
from yuki.llm import LMStudioClient
from yuki.rag.decompose import Person, decompose
from yuki.rag.refine import recall
from yuki.rag.rerank import build_reranker
from yuki.rag.retrieve import Retriever


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=ROOT / "eval" / "cases.jsonl")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-time", action="store_true", help="忽略問題裡的時間說法")
    ap.add_argument("--no-rewrite", action="store_true", help="不用 LLM 改寫查詢")
    ap.add_argument("--no-llm-rerank", action="store_true", help="不用 LLM 重排（同 --llm-rerank off）")
    ap.add_argument("--llm-rerank", choices=["time", "always", "off"], help="覆寫 .env 的 LLM_RERANK")
    ap.add_argument("--no-focus", action="store_true", help="rerank 用原句，不拿掉時間與問法")
    ap.add_argument("--no-fallback", action="store_true", help="全部低於門檻時不保底")
    ap.add_argument("--db", type=Path, default=None, help="預設用 .env 的 DB_PATH")
    args = ap.parse_args()

    s = load_settings()
    embedder = build_embedder(s)
    store = MemoryStore(args.db or s.db_path, embedder.model_name)
    retriever = Retriever(store, embedder, None if args.no_rerank else build_reranker(s),
                          top_n=s.retrieve_top_n, final_k=args.k or s.retrieve_final_k,
                          rrf_k=s.rrf_k, rerank_min_score=s.rerank_min_score, time_boost=s.time_boost,
                          use_focus=s.rerank_focus and not args.no_focus, order_top_n=s.order_top_n,
                          fallback_k=0 if args.no_fallback else s.rerank_fallback)

    rewrite = s.query_rewrite and not args.no_rewrite
    llm_rerank = "off" if args.no_llm_rerank else (args.llm_rerank or s.llm_rerank)
    llm = None
    if rewrite or llm_rerank != "off":
        llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, timeout=s.llm_timeout,
                             json_reasoning_effort=s.llm_json_reasoning_effort,
                             max_concurrency=s.llm_max_concurrency)
    print(f"改寫={rewrite} LLM重排={llm_rerank} rerank={not args.no_rerank} 時間={not args.no_time} "
          f"focus={retriever.use_focus} 保底={retriever.fallback_k}")

    cases = [json.loads(l) for l in args.cases.read_text(encoding="utf-8").splitlines() if l.strip()]
    hit = recall_sum = 0.0
    timed = timed_hit = 0
    latencies = []
    for c in cases:
        author = Person(**c["author"])
        members = [Person(**m) for m in c.get("members", [])]
        today = date.fromisoformat(c["now"]) if c.get("now") else None
        t0 = time.perf_counter()
        subs = await decompose(c["query"], author, [], members, wake_words=s.wake_words, today=today)
        parsed = subs[0].time if subs else None
        has_time = parsed is not None
        if args.no_time:
            for sq in subs:
                sq.time = None
        got = [m.id for m in await recall(retriever, c["guild_id"], subs, llm=llm, rewrite=rewrite,
                                          rerank=llm_rerank, rerank_top=s.llm_rerank_top, today=today)]
        latencies.append((time.perf_counter() - t0) * 1000)
        expected = set(c["expected"])
        found = expected & set(got)
        hit += bool(found)
        recall_sum += len(found) / len(expected) if expected else 1
        timed += has_time
        timed_hit += has_time and bool(found)
        mark = "OK " if found == expected else "MISS"
        when = f" [{parsed.phrase}]" if has_time else ""
        print(f"{mark} {c['query']}{when}  expected={sorted(expected)} got={got}")
    n = len(cases) or 1
    print(f"\n{len(cases)} 題  Hit@{retriever.final_k}={hit / n:.0%}  Recall@{retriever.final_k}={recall_sum / n:.0%}")
    if timed:
        rest = len(cases) - timed
        print(f"  有時間說法 {timed} 題 Hit={timed_hit / timed:.0%}；"
              f"沒有 {rest} 題 Hit={(hit - timed_hit) / max(rest, 1):.0%}")
    latencies.sort()
    if latencies:
        print(f"  延遲：中位數 {latencies[len(latencies) // 2]:.0f} ms，P90 {latencies[int(len(latencies) * 0.9)]:.0f} ms")
    if llm:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
