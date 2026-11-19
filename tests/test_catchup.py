import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import discord

import yuki.archive_sync as sync_mod
from tests.conftest import FakeLLM
from yuki.archive import ArchiveDB, ArchivedMessage
from yuki.archive_sync import ArchiveSync
from yuki.brain import Brain
from yuki.config import Settings
from yuki.chat import ChatLine, Person

TOBY = Person("2", "Toby")
T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def sf(minutes: float) -> int:
    """T0 之後幾分鐘的 snowflake。"""
    return discord.utils.time_snowflake(T0 + timedelta(minutes=minutes))


def row(i: int, channel: str, author: str = "2") -> ArchivedMessage:
    return ArchivedMessage(i, "g", channel, "聊天", "", author, "Toby", "2026-11-01 12:00:00", f"hi {i}", None, 0)


# ---------- ArchiveDB ----------
def test_advance_last_id_never_goes_back():
    db = ArchiveDB(":memory:")
    db.advance_last_id("c1", 10)
    db.advance_last_id("c1", 5)
    assert db.last_id("c1") == 10
    db.advance_last_id("c1", 20)
    assert db.last_id("c1") == 20


def test_latest_and_max_id():
    db = ArchiveDB(":memory:")
    assert db.latest_id("g") is None
    db.add_many([row(1, "c1"), row(7, "c2"), row(3, "c1")])
    assert db.latest_id("g") == 7 and db.max_id("c1") == 3 and db.max_id("c9") is None


# ---------- 補抓 ----------
def test_catch_up_fetches_gap_and_skips_old_unfetched_channels(monkeypatch):
    db = ArchiveDB(":memory:")
    known = SimpleNamespace(id=sf(-1000), name="聊天")      # 抓過的頻道
    never = SimpleNamespace(id=sf(-2000), name="閒置")      # 早就存在、但從沒抓過
    fresh = SimpleNamespace(id=sf(30), name="新討論串")     # 關機期間才開的討論串
    db.add_many([row(sf(10), str(known.id))])
    db.set_last_id(str(known.id), sf(10))
    db.commit()

    async def fake_channels(*a, **kw):
        assert kw["archived_since"] == discord.utils.snowflake_time(sf(10))
        return [(known, ""), (never, ""), (fresh, "聊天")]

    fetched = {}

    async def fake_fetch(ch, parent, gid, db_, skip, stats, *, before, collect):
        fetched[ch.id] = db_.last_id(str(ch.id))
        new = [row(sf(20), str(ch.id))] if ch is known else [row(sf(40), str(ch.id))]
        db_.add_many(new)
        collect.extend(new)

    monkeypatch.setattr(sync_mod, "public_channels", fake_channels)
    monkeypatch.setattr(sync_mod, "fetch_channel", fake_fetch)
    guild = SimpleNamespace(id="g", name="g", me=None, roles=[], default_role=None)
    sync = ArchiveSync(db)
    new = asyncio.run(sync.catch_up(guild))

    assert fetched == {known.id: sf(10), fresh.id: None}  # 抓過的從上次接著抓；新討論串從頭抓
    assert set(new) == {str(known.id), str(fresh.id)}
    assert db.last_id(str(never.id)) is not None          # 沒回溯，但從現在開始有進度
    for ch in (known, never, fresh):                       # 進度都推到補抓開始的時間之後
        assert db.last_id(str(ch.id)) > sf(40)


def test_catch_up_without_archive_does_nothing():
    sync = ArchiveSync(ArchiveDB(":memory:"))
    guild = SimpleNamespace(id="g", name="g")
    assert asyncio.run(sync.catch_up(guild)) == {}


def test_live_messages_do_not_advance_progress_while_catching_up():
    sync = ArchiveSync(ArchiveDB(":memory:"))
    assert sync._may_advance("g", 1)                       # 沒在補抓：可以推進
    sync._busy.add("g")
    sync._done["g"] = {2}
    sync._failed["g"] = {3}
    assert not sync._may_advance("g", 1)                   # 還沒補抓完的頻道不推進，免得跳過缺口
    assert sync._may_advance("g", 2)
    sync._busy.discard("g")
    assert not sync._may_advance("g", 3)                   # 補抓失敗的頻道留給下次


# ---------- 補進對話紀錄 ----------
def test_brain_catch_up_seeds_history_before_live_messages():
    brain = Brain(replace(Settings(), history_messages=3), FakeLLM())
    brain.observe("c1", ChatLine(TOBY, "即時訊息"))  # 補抓期間已經收到的
    brain.catch_up("c1", [ChatLine(TOBY, f"msg {m}") for m in range(3)])
    assert [l.text for l in brain.history["c1"]] == ["msg 1", "msg 2", "即時訊息"]
