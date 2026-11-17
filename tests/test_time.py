import asyncio
from datetime import date

import pytest

from yuki.rag.decompose import Person, SubQuery, decompose
from yuki.rag.retrieve import Retriever
from yuki.rag.timeparse import TimeRange, cn_int, parse_time

TODAY = date(2024, 3, 13)  # 星期三
G = "g1"
BEAR = Person("7", "大熊")


@pytest.mark.parametrize("text, start, end", [
    ("大熊昨天在幹嘛", "2024-03-12", "2024-03-12"),
    ("前天晚上吃什麼", "2024-03-11", "2024-03-11"),
    ("大前天", "2024-03-10", "2024-03-10"),
    ("三天前去哪", "2024-03-09", "2024-03-11"),
    ("上週去了哪裡", "2024-03-04", "2024-03-10"),
    ("上禮拜五", "2024-03-08", "2024-03-08"),
    ("這禮拜", "2024-03-11", "2024-03-13"),
    ("上個月的旅行", "2024-02-01", "2024-02-29"),
    ("上上個月", "2024-01-01", "2024-01-31"),
    ("兩個月前", "2024-01-01", "2024-01-31"),
    ("這個月", "2024-03-01", "2024-03-13"),
    ("去年夏天", "2023-01-01", "2023-12-31"),
    ("今年", "2024-01-01", "2024-03-13"),
    ("2022年5月", "2022-05-01", "2022-05-31"),
    ("2021年的事", "2021-01-01", "2021-12-31"),
    ("十二月的時候", "2023-12-01", "2023-12-31"),   # 還沒到的月份 → 去年
    ("2月那次", "2024-02-01", "2024-02-29"),
    ("3月5號", "2024-03-05", "2024-03-05"),
])
def test_parse_time(text, start, end):
    t = parse_time(text, TODAY)
    assert t and (t.start.isoformat(), t.end.isoformat()) == (start, end), t


@pytest.mark.parametrize("text", ["大熊上次說的那家店", "你之前提過的書", "一個月都在加班嗎", "每個月存多少"])
def test_no_time(text):
    assert parse_time(text, TODAY) is None


def test_soft_time_widens():
    t = parse_time("大熊最近在忙什麼", TODAY)
    assert t.soft and len(t.windows()) == 2


def test_cn_int():
    assert [cn_int(s) for s in ["3", "十", "十二", "二十", "兩", "二十五"]] == [3, 10, 12, 20, 2, 25]


def test_decompose_attaches_time_to_every_subquery():
    a, b = Person("1", "Hunter"), Person("2", "Toby")
    subs = asyncio.run(decompose("Hunter 跟 Toby 上週在幹嘛", a, [], [a, b], wake_words=["yuki"], today=TODAY))
    assert len(subs) == 2 and all(s.time and s.time.phrase.startswith("上") for s in subs)


def _seed(store, embedder, items):
    async def run():
        vecs = await embedder.embed_docs([t for _, t in items])
        for (d, t), v in zip(items, vecs):
            store.add(G, t, v, subject_id=BEAR.id, subject_name=BEAR.name, created_at=f"{d} 21:00:00")
    asyncio.run(run())


def test_time_filter_picks_the_right_day(store, embedder):
    # 每天都在做幾乎一樣的事：沒有時間過濾就分不出是哪一天
    _seed(store, embedder, [(f"2024-03-{d:02d}", f"大熊 去復健科做頸椎復健 第{d}次") for d in range(1, 13)])
    r = Retriever(store, embedder, None, final_k=1)
    sq = SubQuery("大熊 頸椎復健", BEAR, time=parse_time("昨天", TODAY))
    got = asyncio.run(r.retrieve(G, [sq]))
    assert got[0].created_at.startswith("2024-03-12")


def test_time_filter_falls_back_when_period_is_empty(store, embedder):
    _seed(store, embedder, [("2023-06-01", "大熊 喜歡吃拉麵")])
    r = Retriever(store, embedder, None)
    sq = SubQuery("大熊 喜歡吃什麼", BEAR, time=TimeRange(date(2024, 3, 12), date(2024, 3, 12), "昨天"))
    assert asyncio.run(r.retrieve(G, [sq]))


def test_updated_memory_counts_for_recent_period(store, embedder):
    _seed(store, embedder, [("2022-01-01", "大熊 住在台中")])
    mem = store.list(G)[0]
    store.db.execute("UPDATE memories SET updated_at='2024-03-12 10:00:00' WHERE id=?", (mem.id,))
    store._dirty(G)
    ids = [i for i, _ in store.sparse_search(G, "大熊 住", 5, since="2024-03-01", until="2024-03-13")]
    assert ids == [mem.id]


class KeywordReranker:
    async def score(self, query, texts):
        return [0.9 if "拉麵" in t else 0.3 for t in texts]


def test_wrong_time_phrase_does_not_hide_the_answer(store, embedder):
    # 使用者說「昨天」，但其實是五天前的事：期間內的記憶只加分，不會把期間外的正解排除
    _seed(store, embedder, [("2024-03-07", "大熊 去吃了一蘭拉麵"), ("2024-03-12", "大熊 在家打掃房間")])
    r = Retriever(store, embedder, KeywordReranker(), final_k=1, time_boost=0.1)
    sq = SubQuery("大熊 吃了什麼拉麵", BEAR, time=parse_time("昨天", TODAY))
    assert "拉麵" in asyncio.run(r.retrieve(G, [sq]))[0].text


def test_time_boost_breaks_ties_between_similar_days(store, embedder):
    _seed(store, embedder, [("2024-03-05", "大熊 去健身房練腿"), ("2024-03-12", "大熊 去健身房練背")])
    r = Retriever(store, embedder, KeywordReranker(), final_k=1, time_boost=0.1)
    sq = SubQuery("大熊 去健身房", BEAR, time=parse_time("昨天", TODAY))
    assert asyncio.run(r.retrieve(G, [sq]))[0].created_at.startswith("2024-03-12")


def test_reply_prompt_shows_memory_age():
    from yuki.memory.store import Memory
    from yuki.prompts import age, reply_system
    today = date(2026, 10, 30)
    assert [age(d, today) for d in ["2026-10-30", "2026-10-29", "2026-10-25", "2026-09-30", "2026-06-26",
                                    "2024-10-30"]] == ["今天", "昨天", "5 天前", "約 4 週前", "約 4 個月前",
                                                       "約 2 年前"]
    m = Memory("x", "g", "1", "老K", "老K 做物理治療", None, "2026-06-26 21:00:00", "2026-06-26 21:00:00")
    assert "(2026-06-26，約 4 個月前) 老K 做物理治療" in reply_system("Yuki", [], [m], today)
