import asyncio

import pytest

from yuki.memory.store import EmbeddingMismatch, MemoryStore
from yuki.rag.decompose import Person, SubQuery, decompose, rule_decompose
from yuki.rag.retrieve import Retriever, round_robin, rrf

G = "g1"
HUNTER = Person("1", "Hunter")
TOBY = Person("2", "Toby")


def seed(store, embedder):
    async def run():
        data = [
            (HUNTER, "Hunter 喜歡玩 APEX Legends"),
            (HUNTER, "Hunter 喜歡玩 Valorant"),
            (HUNTER, "Hunter 也很愛玩 APEX 排位"),
            (HUNTER, "Hunter 喜歡玩 Minecraft"),
            (TOBY, "Toby 喜歡玩原神"),
            (TOBY, "Toby 喜歡喝白色 Monster"),
        ]
        vecs = await embedder.embed_docs([t for _, t in data])
        return [store.add(G, t, v, subject_id=p.id, subject_name=p.name) for (p, t), v in zip(data, vecs)]
    return asyncio.run(run())


def test_rrf_rewards_agreement():
    # y 在兩個清單都有出現，勝過只在一個清單排第一的 x
    assert rrf([["x", "y"], ["y", "z"]])[0] == "y"
    assert set(rrf([["a", "b", "c"], ["b", "a", "d"]])[:2]) == {"a", "b"}


def test_round_robin_gives_each_group_a_quota(store, embedder):
    mems = seed(store, embedder)
    hunter = [m for m in mems if m.subject_id == "1"]
    toby = [m for m in mems if m.subject_id == "2"]
    picked = round_robin([hunter, toby], 4)
    assert [m.subject_id for m in picked] == ["1", "2", "1", "2"]


def test_round_robin_dedups():
    class M:
        def __init__(self, i): self.id = i
    a, b = M("a"), M("b")
    assert [m.id for m in round_robin([[a, b], [a]], 5)] == ["a", "b"]


def test_subject_filter_and_sparse(store, embedder):
    seed(store, embedder)
    hits = store.sparse_search(G, "APEX", 10)
    assert hits and all("APEX" in store.get(i).text for i, _ in hits)
    hits = store.sparse_search(G, "喜歡玩", 10, subject_id="2")
    assert {store.get(i).subject_id for i, _ in hits} == {"2"}


def test_update_delete_invalidate_cache(store, embedder):
    mems = seed(store, embedder)
    vec = asyncio.run(embedder.embed_docs(["Toby 改喝綠色 Monster"]))[0]
    store.update(mems[-1].id, "Toby 改喝綠色 Monster", vec)
    assert store.sparse_search(G, "綠色", 5)[0][0] == mems[-1].id
    store.delete(mems[-1].id)
    assert store.sparse_search(G, "綠色", 5) == []


def test_embed_model_mismatch(tmp_path, embedder):
    path = tmp_path / "m.db"
    s = MemoryStore(path, "model-a")
    s.add(G, "x", asyncio.run(embedder.embed_docs(["x"]))[0])
    s.db.close()
    with pytest.raises(EmbeddingMismatch):
        MemoryStore(path, "model-b")


def test_rule_decompose_multi_person():
    subs = rule_decompose("Hunter 跟 Toby 誰比較愛玩遊戲", TOBY, [HUNTER, TOBY], ["yuki"])
    assert [s.subject for s in subs] == [HUNTER, TOBY]
    assert all("遊戲" in s.text and "跟" not in s.text for s in subs)
    subs = rule_decompose("Hunter跟Toby都在忙什麼", TOBY, [HUNTER, TOBY], ["yuki"])
    assert [s.text for s in subs] == ["Hunter 都在忙什麼", "Toby 都在忙什麼"]


def test_rule_decompose_self_reference():
    subs = asyncio.run(decompose("yuki 我喜歡喝什麼", TOBY, [], [HUNTER, TOBY], wake_words=["yuki"]))
    assert len(subs) == 1 and subs[0].subject == TOBY
    assert "Toby" in subs[0].text and "我" not in subs[0].text


def test_rule_decompose_no_person():
    subs = asyncio.run(decompose("今天好熱", TOBY, [], [HUNTER], wake_words=["yuki"]))
    assert len(subs) == 1 and subs[0].subject is None


def test_retriever_keeps_quota_for_each_person(store, embedder):
    seed(store, embedder)
    r = Retriever(store, embedder, None, top_n=20, final_k=4)
    subs = [SubQuery("Hunter 喜歡玩的遊戲", HUNTER), SubQuery("Toby 喜歡玩的遊戲", TOBY)]
    got = asyncio.run(r.retrieve(G, subs))
    # Hunter 有 4 筆遊戲記憶，若混在一起排序 Toby 很容易被擠掉；輪流取保證兩人都在
    assert {m.subject_id for m in got} == {"1", "2"}
    assert len(got) == 4


def test_retriever_falls_back_when_subject_has_no_memories(store, embedder):
    seed(store, embedder)
    r = Retriever(store, embedder, None)
    got = asyncio.run(r.retrieve(G, [SubQuery("Toby 喜歡 APEX", Person("99", "Nobody"))]))
    assert got


class FakeReranker:
    async def score(self, query, texts):
        return [1.0 if "APEX" in t else 0.0 for t in texts]


def test_rerank_drops_low_scores(store, embedder):
    seed(store, embedder)
    r = Retriever(store, embedder, FakeReranker(), rerank_min_score=0.5)
    got = asyncio.run(r.retrieve(G, [SubQuery("Hunter 玩什麼", HUNTER)]))
    assert got and all("APEX" in m.text for m in got)
