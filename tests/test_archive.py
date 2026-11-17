from yuki.archive import ArchiveDB, ArchivedMessage


def msg(i, author="1", channel="c1"):
    return ArchivedMessage(i, "g", channel, "聊天", "", author, f"user{author}", "2024-01-01 12:00:00",
                           f"hi {i}", None, 0)


def test_resume_point_and_idempotent_insert():
    db = ArchiveDB(":memory:")
    db.add_many([msg(1), msg(2)])
    db.set_last_id("c1", 2)
    db.add_many([msg(2)])  # 重跑時同一則再存一次不會重複
    db.commit()
    assert db.last_id("c1") == 2 and db.last_id("c2") is None
    assert db.counts("g")[0][1] == 2


def test_opt_out_deletes_and_remembers():
    db = ArchiveDB(":memory:")
    db.add_many([msg(1, "1"), msg(2, "2"), msg(3, "1")])
    assert db.opt_out("g", "1") == 2
    assert db.opted_out("g") == {"1"} and db.counts("g")[0][1] == 1
