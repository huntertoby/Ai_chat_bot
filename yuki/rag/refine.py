"""用聊天 LLM 補強檢索的兩個步驟（都可以關掉）：

1. 查詢改寫：口語問題 →「答案那筆記憶可能的寫法」，多幾組問法一起做 dense + sparse 檢索，
   解決「問題太口語，候選池根本撈不到」。
2. LLM 重排：cross-encoder 排好的前幾筆，連同日期交給 LLM 挑。cross-encoder 看不懂日期，
   同一個人每天做很像的事時分不出是哪一天；LLM 對照今天日期就分得出來。

任何一步失敗都退回原本的結果，不會讓回覆失敗。
"""
from __future__ import annotations

import logging
from datetime import date

from .. import prompts
from ..llm import LLMError, LMStudioClient
from ..memory.store import Memory
from .decompose import SubQuery
from .retrieve import Hit, Retriever, round_robin

log = logging.getLogger(__name__)


async def rewrite_queries(llm: LMStudioClient, subs: list[SubQuery], per_query: int = 2) -> None:
    """把改寫結果加進每個子查詢的 variants（原問法保留在最前面）。"""
    if not subs:
        return
    messages = [{"role": "system", "content": prompts.REWRITE_SYSTEM},
                {"role": "user", "content": prompts.rewrite_user([sq.text for sq in subs])}]
    try:
        data = await llm.chat_json(messages, prompts.REWRITE_SCHEMA, name="rewrite", max_tokens=400,
                                   reasoning_effort="none")
    except LLMError as e:
        log.warning("查詢改寫失敗，用原問法：%s", e)
        return
    items = (data or {}).get("items", [])
    if len(subs) == 1:
        items = [{**it, "index": 1} for it in items[:1]]  # 只有一題時編號寫錯也無妨
    for item in items:
        i = item.get("index")
        if not isinstance(i, int) or not 1 <= i <= len(subs):
            continue
        i -= 1  # prompt 的編號從 1 開始（模型習慣如此，從 0 開始會被寫成 1）
        extra = [str(t).strip() for t in item.get("rewrites", [])[:per_query] if str(t).strip()]
        subs[i].variants = list(dict.fromkeys(subs[i].variants + extra))


async def llm_rerank(llm: LMStudioClient, sq: SubQuery, hits: list[Hit], today: date,
                     top_m: int = 10) -> list[Hit]:
    """LLM 挑出的排前面；沒被挑到的照原順序接在後面（LLM 漏看也不會把正解丟掉）。"""
    head, tail = hits[:top_m], hits[top_m:]
    if len(head) <= 1:
        return hits
    candidates = [(h.memory.created_at[:10], h.memory.text) for h in head]
    messages = [{"role": "system", "content": prompts.LLM_RERANK_SYSTEM},
                {"role": "user", "content": prompts.llm_rerank_user(sq.text, today.isoformat(), candidates)}]
    try:
        data = await llm.chat_json(messages, prompts.LLM_RERANK_SCHEMA, name="rerank", max_tokens=200,
                                   reasoning_effort="none")
    except LLMError as e:
        log.warning("LLM 重排失敗，用 cross-encoder 順序：%s", e)
        return hits
    # prompt 的編號從 1 開始
    order = [i - 1 for i in dict.fromkeys((data or {}).get("relevant", []))
             if isinstance(i, int) and 1 <= i <= len(head)]
    if not order:
        return hits
    picked = [head[i] for i in order]
    rest = [h for i, h in enumerate(head) if i not in set(order)]
    return picked + rest + tail


def wants_llm_rerank(mode: str, sq: SubQuery) -> bool:
    """always：每次都重排；time：只在問題有時間說法時重排（評估中 LLM 重排只改善這類題目，
    其餘題目重排前後一樣卻多花約 1 秒）；off：不重排。依日期排序的「最早」類問題不重排。"""
    if sq.order or mode == "off":
        return False
    return mode == "always" or (mode == "time" and sq.time is not None)


async def recall(retriever: Retriever, guild_id: str, subs: list[SubQuery], *,
                 llm: LMStudioClient | None = None, rewrite: bool = False, rerank: str | bool = "off",
                 rerank_top: int = 10, today: date | None = None, k: int | None = None) -> list[Memory]:
    """回覆用的完整檢索：（改寫）→ 混合檢索 + rerank →（LLM 重排）→ 每組輪流取。"""
    mode = {True: "always", False: "off"}.get(rerank, rerank)
    if llm is not None and rewrite:
        await rewrite_queries(llm, subs)
    groups = await retriever.search(guild_id, subs)
    if llm is not None and mode != "off":
        today = today or date.today()
        groups = [await llm_rerank(llm, sq, g, today, rerank_top) if wants_llm_rerank(mode, sq) else g
                  for sq, g in zip(subs, groups)]
    return round_robin([[h.memory for h in g] for g in groups], k or retriever.final_k)
