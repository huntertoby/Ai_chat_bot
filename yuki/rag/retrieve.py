"""每個子查詢各自一組：dense + sparse 各 top-N → RRF → rerank；最後輪流取，每組都有名額。"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from ..embedding import Embedder
from ..memory.store import Memory, MemoryStore
from .decompose import SubQuery
from .rerank import LocalReranker

log = logging.getLogger(__name__)


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion：score = Σ 1 / (k + 名次)，只看名次，不同分數尺度也能合併。"""
    score: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            score[doc_id] += 1.0 / (k + rank)
    return sorted(score, key=lambda d: score[d], reverse=True)


def round_robin(groups: list[list[Memory]], total: int) -> list[Memory]:
    """每組輪流拿一條，避免某一個人的記憶把名額占滿。"""
    picked: list[Memory] = []
    seen: set[str] = set()
    depth = max((len(g) for g in groups), default=0)
    for i in range(depth):
        for g in groups:
            if i < len(g) and g[i].id not in seen:
                picked.append(g[i])
                seen.add(g[i].id)
                if len(picked) >= total:
                    return picked
    return picked


@dataclass
class Hit:
    memory: Memory
    score: float | None  # rerank 分數；沒開 rerank 時為 None


class Retriever:
    def __init__(self, store: MemoryStore, embedder: Embedder, reranker: LocalReranker | None, *,
                 top_n: int = 20, final_k: int = 6, rrf_k: int = 60, rerank_min_score: float = 0.05,
                 time_boost: float = 0.1, use_focus: bool = True, order_top_n: int = 60, fallback_k: int = 2):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.top_n = top_n
        self.final_k = final_k
        self.rrf_k = rrf_k
        self.rerank_min_score = rerank_min_score
        self.time_boost = time_boost
        self.use_focus = use_focus      # rerank 用拿掉時間與問法的主題，而不是原句
        self.order_top_n = order_top_n  # 「最早／最後一次」要看更多候選，答案不一定在最相關的前 20
        self.fallback_k = fallback_k    # 指定了人卻全部低於門檻時，仍保留分數最高的幾筆

    def _lists(self, guild_id: str, sq: SubQuery, vecs, subject_id: str | None,
               window: tuple[str, str] | None, n: int) -> list[list[str]]:
        since, until = window or (None, None)
        lists = []
        for variant, vec in zip(sq.variants, vecs):
            lists.append([i for i, _ in self.store.dense_search(guild_id, vec, n, subject_id, since, until)])
            lists.append([i for i, _ in self.store.sparse_search(guild_id, variant, n, subject_id, since, until)])
        return lists

    def _candidates(self, guild_id: str, sq: SubQuery, vecs, subject_id: str | None) -> list[str]:
        n = self.order_top_n if sq.order else self.top_n
        lists = self._lists(guild_id, sq, vecs, subject_id, None, n)
        # 有時間說法 → 多加幾份「只搜那段期間」的排名一起 RRF：期間內的記憶會被往前推，
        # 但期間外很相關的記憶不會被丟掉（人講「前天」「上個月」常常不精確）
        for window in sq.time.windows() if sq.time else []:
            lists += self._lists(guild_id, sq, vecs, subject_id, window, n)
        return rrf(lists, self.rrf_k)[:n]

    def _time_bonus(self, sq: SubQuery, mem: Memory) -> float:
        if not sq.time:
            return 0.0
        days = (mem.created_at[:10], mem.updated_at[:10])
        for i, (lo, hi) in enumerate(sq.time.windows()):
            if any(lo <= d <= hi for d in days):
                return self.time_boost / (i + 1)
        return 0.0

    async def search_group(self, guild_id: str, sq: SubQuery, vecs) -> list[Hit]:
        subject_id = sq.subject.id if sq.subject else None
        ids = self._candidates(guild_id, sq, vecs, subject_id)
        if not ids and subject_id is not None:
            # 這個人名下沒有記憶（例如舊資料存在發言者底下）→ 放寬成全 guild 搜尋
            ids = self._candidates(guild_id, sq, vecs, None)
        mems = self.store.get_many(ids)
        if not mems or self.reranker is None:
            return [Hit(m, None) for m in mems]
        scores = await self.reranker.score(sq.focus if self.use_focus else sq.text, [m.text for m in mems])
        scored = sorted((Hit(m, s) for m, s in zip(mems, scores)), key=lambda h: -h.score)
        # 門檻看 rerank 原始分數
        hits = [h for h in scored if h.score >= self.rerank_min_score]
        if not hits and sq.subject is not None:
            # 已經限定是這個人的記憶，全部被門檻刷掉多半是問法讓 reranker 誤判，留最像的幾筆給 LLM 判斷
            hits = scored[: self.fallback_k]
        if sq.order:
            # 「最早／最後一次」：答案在日期，相關的記憶依日期排序
            return sorted(hits, key=lambda h: h.memory.created_at, reverse=sq.order == "latest")
        # 排序再加上時間吻合的加分（reranker 看不懂日期）
        return sorted(hits, key=lambda h: -(h.score + self._time_bonus(sq, h.memory)))

    async def search(self, guild_id: str, subqueries: list[SubQuery]) -> list[list[Hit]]:
        all_variants = [v for sq in subqueries for v in sq.variants]
        if not all_variants:
            return []
        vecs = await self.embedder.embed_queries(all_variants)
        groups, offset = [], 0
        for sq in subqueries:
            n = len(sq.variants)
            groups.append(await self.search_group(guild_id, sq, vecs[offset: offset + n]))
            offset += n
        for sq, hits in zip(subqueries, groups):
            log.debug("子查詢「%s」(%s) → %s", sq.text, sq.subject.name if sq.subject else "全體",
                      [(h.memory.id, round(h.score, 3) if h.score is not None else None) for h in hits[:5]])
        return groups

    async def retrieve(self, guild_id: str, subqueries: list[SubQuery], k: int | None = None) -> list[Memory]:
        groups = await self.search(guild_id, subqueries)
        return round_robin([[h.memory for h in g] for g in groups], k or self.final_k)
