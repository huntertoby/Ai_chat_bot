"""/不要記我 以人為單位（審查一-3）：在哪裡打都一樣，所有伺服器的資料都刪，借別的伺服器資料時也看退出名單。"""
import asyncio
import sqlite3
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from yuki.archive import ArchiveDB, ArchivedMessage
from yuki.archive_sync import ArchiveSync
from yuki.brain import Brain
from yuki.config import Settings
from yuki.memory.store import MemoryStore
from yuki.memory.writer import ChatLine
from yuki.optout import forget_everywhere
from yuki.profile.auto import ProfileUpdater
from yuki.profile.store import Fact, ProfileDB, ProfileFiles
from yuki.rag.decompose import Person
from yuki.voice_presence import VoicePresence

MAIN, TEST, OTHER = "main", "test", "other"
XIAOMEI = Person("3", "小美")
AMING = Person("1", "阿明")


def msg(i, guild, author_id, name, text):
    return ArchivedMessage(i, guild, f"c-{guild}", "閒聊", "", author_id, name, "2026-11-01 12:00:00", text, None, 0)


def archive():
    db = ArchiveDB(":memory:")
    db.add_many([msg(1, MAIN, "3", "小美", "我下週要考期末考"),
                 msg(2, OTHER, "3", "小美", "期末考好難"),
                 msg(3, MAIN, "1", "阿明", "期末考加油")])
    db.commit()
    return db


def stores(tmp_path):
    arch = archive()
    sync = ArchiveSync(arch)
    voice = VoicePresence(sqlite3.connect(":memory:"))
    pdb = ProfileDB(":memory:")
    files = ProfileFiles(tmp_path / "profiles")
    mem = MemoryStore(":memory:", "fake")
    return SimpleNamespace(archive=arch, sync=sync, voice=voice, profile_db=pdb, profiles=files, memories=mem)


def test_opting_out_in_a_test_server_deletes_main_server_data_too(tmp_path):
    s = stores(tmp_path)
    for g in (MAIN, OTHER, "dm-3"):
        s.profiles.write(g, "3", "# 小美\n喜歡貓")
    s.profiles.write(MAIN, "1", "# 阿明\n愛打球")
    s.profile_db.save_segment(MAIN, "c-main", 1, [Fact("3", "小美", "fact", "要考期末考", "high", "2026-11-01")])
    s.profile_db.save_segment(OTHER, "c-other", 2, [Fact("3", "小美", "fact", "覺得考試難", "high", "2026-11-01")])
    vec = np.ones(4, dtype=np.float32)
    s.memories.add(MAIN, "小美要考期末考", vec, subject_id="3", subject_name="小美")
    s.memories.add(OTHER, "阿明說要幫忙", vec, subject_id="1", subject_name="阿明", author_id="3")  # 她講的話產生的
    s.memories.add(MAIN, "阿明在打球", vec, subject_id="1", subject_name="阿明", author_id="1")
    s.voice.update(MAIN, "3", "小美", None, ("v", "語音"), datetime(2026, 11, 1, 20))
    s.voice.update(OTHER, "3", "小美", None, ("v2", "語音"), datetime(2026, 11, 1, 20))
    assert "3" not in s.sync.skip(MAIN)  # 先快取一份舊的退出名單

    f = forget_everywhere("3", TEST, **vars(s))

    assert (f.messages, f.profiles, f.facts, f.memories, f.voice) == (2, 3, 2, 2, 2)
    left = s.archive.db.execute("SELECT author_id FROM messages").fetchall()
    assert left == [("1",)]
    assert [m.text for m in s.memories.list(MAIN)] == ["阿明在打球"] and s.memories.list(OTHER) == []
    assert s.profiles.read(MAIN, "1")  # 別人的檔案不受影響
    # 之後每個伺服器都不再收她的訊息（包括退出前就快取過名單的伺服器）
    assert "3" in s.sync.skip(MAIN) and "3" in s.sync.skip(OTHER)
    assert s.archive.opted_out() == {"3"} and s.archive.opted_out(MAIN) == {"3"}


def test_archive_opt_out_removes_messages_from_search_index_in_every_server():
    db = archive()
    db.opt_out(TEST, "3")
    hits = db.db.execute("SELECT rowid FROM msg_fts").fetchall()
    assert sorted(r[0] for r in hits) == [3]


def make_brain(tmp_path, opted_out=()):
    settings = replace(Settings(), retrieval="agentic", web_enabled=False, profile_enabled=True,
                       profile_dir=tmp_path / "profiles", profile_fallback_guild=MAIN, profile_max_people=4)
    llm = SimpleNamespace(_slots=asyncio.Semaphore(1))
    brain = Brain(settings, llm, MemoryStore(":memory:", "fake"), None, None)
    brain.opted_out = lambda: set(opted_out)
    brain.profiles.write(MAIN, "3", "# 小美\n喜歡貓")
    brain.profiles.write(MAIN, "1", "# 阿明\n愛打球")
    return brain


def test_test_server_does_not_borrow_main_server_profile_of_opted_out_person(tmp_path):
    brain = make_brain(tmp_path, opted_out={"3"})
    names = [n for n, _ in brain.pick_profiles(TEST, "c", XIAOMEI, [AMING], recent=False)]
    assert names == ["阿明"]
    assert brain.profile_text(TEST, "3") == "" and "愛打球" in brain.profile_text(TEST, "1")
    # 沒退出的話照常借
    assert "喜歡貓" in make_brain(tmp_path).profile_text(TEST, "3")


def test_chat_search_never_returns_messages_from_opted_out_people(tmp_path):
    """封存區裡還留著（例如之後由別的程式補抓進來）也查不到。"""
    brain = make_brain(tmp_path, opted_out={"3"})
    brain.archive_db = archive().db
    tools = brain.archive_tools(TEST, MAIN, [XIAOMEI, AMING])
    out = asyncio.run(tools.call("search_chat", {"keywords": ["期末考"]}))
    assert "阿明" in out and "我下週要考期末考" not in out
    assert "沒有" in asyncio.run(tools.call("get_profile", {"name": "小美"}))


def test_old_memory_writer_never_buffers_opted_out_peoples_lines(tmp_path):
    brain = make_brain(tmp_path, opted_out={"3"})
    brain.settings = replace(brain.settings, retrieval="embedding", passive_batch=99)

    async def go():
        brain.buffer(MAIN, "c", ChatLine(XIAOMEI, "我搬家了"))
        brain.buffer(MAIN, "c", ChatLine(AMING, "恭喜"))
        lines = [l.author.name for l in brain._pending["c"]]
        for t in brain._idle_timers.values():
            t.cancel()
        return lines

    assert asyncio.run(go()) == ["阿明"]


def test_profile_updater_never_builds_a_file_for_someone_who_opted_out(tmp_path):
    db = ProfileDB(":memory:")
    files = ProfileFiles(tmp_path)
    merged = []
    builder = SimpleNamespace(merge_one=lambda sid: merged.append(sid) or asyncio.sleep(0))
    arch = ArchiveDB(":memory:")
    up = ProfileUpdater(lambda gid: builder, db, files, arch.db, lambda: True, lambda gid: {"3"},
                        min_new=10_000, min_facts=1)
    # 別人講到她，抽出來的資訊在退出之後才進來
    real_forget = db.forget
    calls = {"n": 0}

    def forget_then_reappear(gid, sid):
        n = real_forget(gid, sid)
        if calls["n"] == 0:
            db.save_segment(MAIN, "c", 1, [Fact("3", "小美", "fact", "x", "high", "2026-11-01"),
                                           Fact("1", "阿明", "fact", "y", "high", "2026-11-01")])
        calls["n"] += 1
        return n

    db.forget = forget_then_reappear
    asyncio.run(up.run_once(MAIN))
    assert merged == ["1"] and not files.read(MAIN, "3")
    assert [r[0] for r in db.stats(MAIN)] == ["1"]
