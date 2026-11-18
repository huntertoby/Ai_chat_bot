import asyncio
from types import SimpleNamespace

from yuki.archive import ArchiveDB, ArchivedMessage
from yuki.profile.auto import ProfileUpdater, new_message_count
from yuki.profile.store import Fact, ProfileDB, ProfileFiles

G = "g"


def archive_with(n: int, channel: str = "c1") -> ArchiveDB:
    db = ArchiveDB(":memory:")
    db.add_many([ArchivedMessage(i, G, channel, "聊天", "", "2", "Toby", "2026-11-01 12:00:00", f"hi {i}", None, 0)
                 for i in range(1, n + 1)])
    db.commit()
    return db


class FakeBuilder:
    """每 10 則一段；抽取時每段產生一條關於 Toby 的資訊。"""

    def __init__(self, archive, on_extract=None):
        self.archive = archive
        self.extracted, self.merged = [], []
        self.on_extract = on_extract

    def channels(self):
        return [("c1", "聊天", 0)]

    def pending_segments(self, cid, cname, db=None):
        after = self.db.last_id(cid) or 0
        ids = [r[0] for r in self.archive.execute("SELECT id FROM messages WHERE channel_id=? AND id>? ORDER BY id",
                                                  (cid, after))]
        return [SimpleNamespace(channel_id=cid, channel_name=cname, last_id=ids[min(i + 9, len(ids) - 1)])
                for i in range(0, len(ids), 10)]

    async def extract_segment(self, seg):
        self.extracted.append(seg.last_id)
        if self.on_extract:
            self.on_extract(len(self.extracted))
        return [Fact("2", "Toby", "興趣", f"第 {seg.last_id} 則附近的事", "high", "2026-11-01")]

    async def merge_one(self, sid):
        self.merged.append(sid)
        self.db.mark_merged([i for i, _ in self.db.pending(G, sid)])
        return True


def make(n_messages, *, idle=lambda: True, min_new=20, on_extract=None, tmp_path=None):
    archive = archive_with(n_messages)
    db = ProfileDB(":memory:")
    builder = FakeBuilder(archive.db, on_extract)
    builder.db = db
    files = ProfileFiles(tmp_path)
    up = ProfileUpdater(lambda gid: builder, db, files, archive.db, idle, lambda gid: set(),
                        min_new=min_new, min_facts=1)
    return up, builder, db


def test_new_message_count_follows_progress():
    archive = archive_with(30)
    db = ProfileDB(":memory:")
    assert new_message_count(archive.db, db, G) == 30
    db.save_segment(G, "c1", 25, [])
    assert new_message_count(archive.db, db, G) == 5


def test_skips_when_too_few_new_messages(tmp_path):
    up, builder, _ = make(10, min_new=20, tmp_path=tmp_path)
    assert asyncio.run(up.run_once(G)) == "skip"
    assert builder.extracted == []


def test_extracts_then_merges_when_idle(tmp_path):
    up, builder, db = make(30, tmp_path=tmp_path)
    assert asyncio.run(up.run_once(G)) == "done"
    assert builder.extracted == [10, 20, 30]
    assert db.last_id("c1") == 30 and builder.merged == ["2"]
    assert new_message_count(up.archive, db, G) == 0


def test_stops_when_someone_starts_talking(tmp_path):
    state = {"idle": True}
    up, builder, db = make(30, idle=lambda: state["idle"], tmp_path=tmp_path,
                           on_extract=lambda n: state.update(idle=n < 2))  # 第 2 段做完時有人講話
    assert asyncio.run(up.run_once(G)) == "busy"
    assert builder.extracted == [10, 20] and db.last_id("c1") == 20  # 做完的段落進度有存
    assert builder.merged == []
    state["idle"] = True
    builder.on_extract = None
    up.min_new = 1
    assert asyncio.run(up.run_once(G)) == "done"  # 下次閒置從第 3 段接著做
    assert builder.extracted == [10, 20, 30]


def test_merges_leftover_facts_even_without_new_messages(tmp_path):
    up, builder, db = make(5, min_new=20, tmp_path=tmp_path)
    db.save_segment(G, "c1", 5, [Fact("2", "Toby", "興趣", "愛打 LoL", "high", "2026-11-01")])
    assert asyncio.run(up.run_once(G)) == "done"
    assert builder.extracted == [] and builder.merged == ["2"]


def test_opted_out_people_are_cleaned(tmp_path):
    up, builder, db = make(5, min_new=20, tmp_path=tmp_path)
    db.save_segment(G, "c1", 5, [Fact("9", "退出的人", "興趣", "x", "high", "2026-11-01")])
    up.files.write(G, "9", "# 退出的人\n## 基本資料\n")
    up.opted_out = lambda gid: {"9"}
    asyncio.run(up.run_once(G))
    assert db.stats(G) == [] and not up.files.read(G, "9")


def test_few_facts_without_file_do_not_keep_it_busy(tmp_path, caplog):
    """資訊太少、又還沒有檔案的人不整理；之後每次閒置檢查都直接 skip，不再寫「更新完成」。"""
    archive = archive_with(20)
    db = ProfileDB(":memory:")
    builder = FakeBuilder(archive.db)
    builder.db = db
    up = ProfileUpdater(lambda gid: builder, db, ProfileFiles(tmp_path), archive.db, lambda: True,
                        lambda gid: set(), min_new=20, min_facts=5)
    assert asyncio.run(up.run_once(G)) == "done"   # 抽了 2 段 → 2 條資訊，不夠 5 條
    assert builder.merged == []
    assert not up.has_unmerged(G)
    caplog.clear()
    with caplog.at_level("INFO"):
        assert asyncio.run(up.run_once(G)) == "skip"
    assert "更新完成" not in caplog.text
