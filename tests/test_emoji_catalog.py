from yuki import emoji_catalog as cat
from yuki.archive import ArchiveDB, ArchivedMessage


def _msg(i, content, author="u1", ch="c1", day="2026-01-01", name="甲"):
    return ArchivedMessage(id=i, guild_id="g", channel_id=ch, channel_name="x", parent_name="",
                           author_id=author, author_name=name, created_at=f"{day} 10:00:00",
                           content=content, reply_to=None, attachments=0)


def _db(msgs):
    db = ArchiveDB(":memory:")
    db.add_many(msgs)
    db.commit()
    return db


def test_count_uses_by_id_with_rename_and_animated():
    rows = [(1, "hi <:old:111> <:old:111>", "2026-01-01 10:00:00"),
            (2, "<:renamed:111> and <a:dance:222>", "2026-01-03 10:00:00"),
            (3, "no emoji :smile:", "2026-01-04 10:00:00")]
    r = cat.count_uses(rows)
    assert r["111"]["uses"] == 3 and r["111"]["last_used"] == "2026-01-03 10:00:00"
    assert r["111"]["msgs"] == [1, 2]
    assert r["222"]["uses"] == 1 and "333" not in r


def test_pick_messages_limit_recent_and_spread():
    created = {i: f"2026-01-{1 + i // 3:02d} 10:00:00" for i in range(30)}
    picked = cat.pick_messages(list(range(30)), created, 5)
    assert len(picked) == 5
    assert len({created[p][:10] for p in picked}) == 5  # 每天一則
    assert picked[0] == 29  # 新的優先
    assert len(cat.pick_messages(list(range(30)), created, 15)) == 15
    assert cat.pick_messages([1, 2], {1: "2026-01-01 a", 2: "2026-01-01 a"}, 15) == [2, 1]


def test_context_window_and_optout_excluded():
    msgs = [_msg(i, f"line{i}", author="u2" if i == 8 else "u1") for i in range(1, 15)]
    msgs[9] = _msg(10, "hit <:e:1>")
    db = _db(msgs)
    before, hit, after = cat.context_for(db.db, 10, {"u2"})
    assert len(before) == 3 and len(after) == 3
    assert "line8" not in "".join(before) and before[-1].endswith("line9")
    assert ":e:" in hit and "<" not in hit
    assert cat.context_for(db.db, 8, {"u2"}) is None


def test_usage_rows_skip_optout_and_other_guild():
    db = _db([_msg(1, "<:e:1>"), _msg(2, "<:e:1>", author="u2")])
    rows = list(cat.usage_rows(db.db, "g", {"u2"}))
    assert [r[0] for r in rows] == [1]
    assert not list(cat.usage_rows(db.db, "other"))


def test_merge_keeps_use():
    new = {"id": "1", "meaning": "", "looks": ""}
    assert cat.merge_item(new, {"use": False, "meaning": "舊"})["use"] is False
    assert cat.merge_item(new, None)["use"] is True
    assert cat.merge_item(new, {"meaning": "舊", "looks": ""}, only_missing=True)["meaning"] == "舊"
    assert cat.merge_item(new, {"meaning": "舊"})["meaning"] == ""


def test_render_html_sorted_top_and_escaped():
    items = [{"id": "1", "name": "a<b>", "kind": "emoji", "uses": 1, "url": "u1", "meaning": "m&"},
             {"id": "2", "name": "zz", "kind": "sticker", "uses": 0, "use": False},
             {"id": "3", "name": "hot", "kind": "emoji", "uses": 9, "animated": True}]
    h = cat.render_html(items)
    assert h.index("hot") < h.index("a&lt;b&gt;") < h.index("zz")
    assert "m&amp;" in h and "停用" in h and 'class="top"' in h


def test_revise_prompt_has_numbers_no_names():
    sn = cat.snippets([(["甲：before"], "乙：hit :secret:", ["x"])])
    assert sn == ["甲：before / 乙：hit :secret:"]
    p = cat.revise_prompt([{"i": 1, "looks": "貓", "meaning": "草", "snippets": sn},
                           {"i": 2, "looks": "狗", "meaning": "草", "snippets": []}])
    assert "第 1 號" in p and "第 2 號" in p and "外觀：狗" in p
