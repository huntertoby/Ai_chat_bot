import asyncio
from datetime import date

from conftest import FakeLLM
from yuki.llm import LLMError
from yuki.memory.store import Memory
from yuki.rag.decompose import Person, SubQuery
from yuki.rag.refine import llm_rerank, recall, rewrite_queries
from yuki.rag.retrieve import Hit, Retriever

BEAR = Person("7", "大熊")
TODAY = date(2024, 3, 13)


def hits(*texts):
    return [Hit(Memory(str(i), "g", "7", "大熊", t, None, f"2024-03-0{i + 1} 21:00:00",
                       f"2024-03-0{i + 1} 21:00:00"), 0.5) for i, t in enumerate(texts)]


class FailingLLM:
    async def chat_json(self, *a, **kw):
        raise LLMError("down")


def test_rewrite_adds_variants_and_keeps_original():
    sq = SubQuery("深夜泡麵王最近搬到哪", BEAR)
    llm = FakeLLM({"items": [{"index": 0, "rewrites": ["深夜泡麵王 搬家到新住處", "深夜泡麵王 新家地址"]},
                             {"index": 9, "rewrites": ["越界的要忽略"]}]})
    asyncio.run(rewrite_queries(llm, [sq]))
    assert sq.variants == ["深夜泡麵王最近搬到哪", "深夜泡麵王 搬家到新住處", "深夜泡麵王 新家地址"]


def test_rewrite_failure_leaves_query_untouched():
    sq = SubQuery("大熊在幹嘛", BEAR)
    asyncio.run(rewrite_queries(FailingLLM(), [sq]))
    assert sq.variants == ["大熊在幹嘛"]


def test_llm_rerank_moves_picks_first_and_keeps_the_rest():
    h = hits("a", "b", "c", "d")
    got = asyncio.run(llm_rerank(FakeLLM({"relevant": [3, 3, 9, 0, 1]}), SubQuery("q", BEAR), h, TODAY, top_m=3))
    # 編號從 1 開始：挑到 3、1 排前面（重複、越界、0 都忽略），沒挑到的照原順序，top_m 外的接在最後
    assert [x.memory.text for x in got] == ["c", "a", "b", "d"]


def test_llm_rerank_shows_dates_to_the_llm():
    llm = FakeLLM({"relevant": [2]})
    asyncio.run(llm_rerank(llm, SubQuery("大熊昨天去哪", BEAR), hits("x", "y"), TODAY))
    prompt = llm.calls[0][1]["content"]
    assert "今天是 2024-03-13" in prompt and "[2] (2024-03-02) y" in prompt


def test_llm_rerank_failure_keeps_order():
    h = hits("a", "b")
    assert asyncio.run(llm_rerank(FailingLLM(), SubQuery("q", BEAR), h, TODAY)) == h


def test_recall_end_to_end(store, embedder):
    async def seed():
        texts = ["大熊 搬家到台北信義區", "大熊 喜歡吃拉麵"]
        for t, v in zip(texts, await embedder.embed_docs(texts)):
            store.add("g", t, v, subject_id="7", subject_name="大熊")
    asyncio.run(seed())
    r = Retriever(store, embedder, None, final_k=1)

    class TwoStepLLM:
        async def chat_json(self, messages, schema, **kw):
            if "rewrite" == kw.get("name"):
                return {"items": [{"index": 1, "rewrites": ["大熊 搬家 新住處"]}]}
            return {"relevant": [2, 1]}  # 故意挑第二名，確認 LLM 重排有生效

    sq = SubQuery("大熊住哪", BEAR)
    got = asyncio.run(recall(r, "g", [sq], llm=TwoStepLLM(), rewrite=True, rerank=True, today=TODAY))
    assert sq.variants == ["大熊住哪", "大熊 搬家 新住處"]
    ranked = asyncio.run(r.search("g", [sq]))[0]
    assert [m.id for m in got] == [ranked[1].memory.id]


def test_rewrite_maps_numbered_items_for_multiple_queries():
    a, b = SubQuery("Hunter 在忙什麼", BEAR), SubQuery("Toby 在忙什麼", BEAR)
    llm = FakeLLM({"items": [{"index": 2, "rewrites": ["Toby 最近在準備考試"]},
                             {"index": 5, "rewrites": ["越界的要忽略"]}]})
    asyncio.run(rewrite_queries(llm, [a, b]))
    assert a.variants == ["Hunter 在忙什麼"] and b.variants == ["Toby 在忙什麼", "Toby 最近在準備考試"]
