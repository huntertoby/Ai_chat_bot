import asyncio
from datetime import date

import pytest

from conftest import FakeLLM
from yuki.rag.decompose import Person, SubQuery, decompose, focus_text
from yuki.rag.refine import recall, wants_llm_rerank
from yuki.rag.retrieve import Retriever
from yuki.rag.timeparse import parse_order, parse_time

G = "g1"
LAOK = Person("5", "老K")
TODAY = date(2026, 10, 30)


@pytest.mark.parametrize("text, order", [
    ("老K 頸椎病最早是什麼時候", "earliest"),
    ("Momo 第一次去日本是哪一年", "earliest"),
    ("大熊最後一次打牌是什麼時候", "latest"),
    ("老K 最近頸椎還好嗎", None),          # 「最近」是時間說法，不是排序
    ("你上次說的那家店在哪", None),        # 「上次」只是指涉，不是問時間
])
def test_parse_order(text, order):
    assert parse_order(text) == order


def test_focus_drops_time_and_when_words():
    focus = focus_text("老K 頸椎病最早是發生在什麼時候", None, [LAOK])
    assert "頸椎病" in focus and "什麼時候" not in focus and "最早" not in focus
    t = parse_time("昨天", TODAY)
    assert focus_text("大熊昨天在幹嘛", t).replace(" ", "") == "大熊在幹嘛"


def test_focus_keeps_original_when_nothing_left():
    # 拿掉之後只剩人名，rerank 沒東西可比，就用原句
    assert focus_text("老K 什麼時候", None, [LAOK]) == "老K 什麼時候"


def test_decompose_sets_order_and_focus():
    subs = asyncio.run(decompose("老K 頸椎病最早是什麼時候", Person("1", "Toby"), [], [LAOK],
                                 wake_words=["yuki"], today=TODAY))
    assert subs[0].order == "earliest" and "什麼時候" not in subs[0].focus


def _seed(store, embedder, items):
    async def run():
        vecs = await embedder.embed_docs([t for _, t in items])
        for (d, t), v in zip(items, vecs):
            store.add(G, t, v, subject_id=LAOK.id, subject_name=LAOK.name, created_at=f"{d} 21:00:00")
    asyncio.run(run())


class FocusReranker:
    """只有主題（沒有「什麼時候」）才給高分，模擬 cross-encoder 對問時間的句子打低分。"""
    def __init__(self):
        self.queries = []

    async def score(self, query, texts):
        self.queries.append(query)
        low = "什麼時候" in query
        return [0.01 if low else (0.9 if "頸椎" in t else 0.02) for t in texts]


def test_earliest_question_returns_oldest_relevant_first(store, embedder):
    _seed(store, embedder, [("2024-05-21", "老K 提醒大家注意頸椎健康"), ("2022-04-25", "老K 頸椎不適，諮詢物理治療師"),
                            ("2026-06-26", "老K 第四次物理治療，頸椎改善"), ("2023-01-01", "老K 去打高爾夫")])
    rr = FocusReranker()
    r = Retriever(store, embedder, rr, final_k=3)
    subs = asyncio.run(decompose("老K 頸椎最早是什麼時候", Person("1", "Toby"), [], [LAOK],
                                 wake_words=["yuki"], today=TODAY))
    got = asyncio.run(r.retrieve(G, subs))
    assert got[0].created_at.startswith("2022-04-25")
    assert all("頸椎" in m.text for m in got)
    assert "什麼時候" not in rr.queries[0]


def test_no_focus_reproduces_zero_result_and_fallback_rescues(store, embedder):
    _seed(store, embedder, [("2022-04-25", "老K 頸椎不適"), ("2024-05-21", "老K 注意頸椎健康")])
    sq = SubQuery("老K 頸椎最早是什麼時候", LAOK)
    no_help = Retriever(store, embedder, FocusReranker(), use_focus=False, fallback_k=0)
    assert asyncio.run(no_help.retrieve(G, [sq])) == []
    rescued = Retriever(store, embedder, FocusReranker(), use_focus=False, fallback_k=2)
    assert len(asyncio.run(rescued.retrieve(G, [sq]))) == 2


def test_fallback_only_when_person_is_known(store, embedder):
    _seed(store, embedder, [("2022-04-25", "老K 頸椎不適")])
    sq = SubQuery("頸椎什麼時候", None)  # 沒指定人：低分就是真的不相關，不保底
    r = Retriever(store, embedder, FocusReranker(), use_focus=False, fallback_k=2)
    assert asyncio.run(r.retrieve(G, [sq])) == []


def test_llm_rerank_mode():
    timed = SubQuery("q", LAOK, time=parse_time("昨天", TODAY))
    plain = SubQuery("q", LAOK)
    ordered = SubQuery("q", LAOK, time=parse_time("昨天", TODAY), order="earliest")
    assert [wants_llm_rerank("time", s) for s in (timed, plain, ordered)] == [True, False, False]
    assert [wants_llm_rerank("always", s) for s in (timed, plain, ordered)] == [True, True, False]
    assert not wants_llm_rerank("off", timed)


def test_recall_time_mode_skips_llm_without_time_phrase(store, embedder):
    _seed(store, embedder, [("2026-10-29", "老K 整理審稿紀錄"), ("2026-10-20", "老K 去跑步")])
    r = Retriever(store, embedder, None)
    llm = FakeLLM({"relevant": [2, 1]})
    asyncio.run(recall(r, G, [SubQuery("老K 在忙什麼", LAOK)], llm=llm, rerank="time", today=TODAY))
    assert llm.calls == []
    asyncio.run(recall(r, G, [SubQuery("老K 昨天在忙什麼", LAOK, time=parse_time("昨天", TODAY))],
                       llm=llm, rerank="time", today=TODAY))
    assert len(llm.calls) == 1


def test_reranker_quantize_config_without_loading():
    from yuki.rag.rerank import LocalReranker
    q = LocalReranker("Qwen/Qwen3-Reranker-4B", quantize="4bit")._model_kwargs()
    assert "quantization_config" in q and q["quantization_config"].load_in_4bit
    assert LocalReranker("BAAI/bge-reranker-v2-m3", quantize="none")._model_kwargs() == {"torch_dtype": "auto"}
