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


# ---------------- C1-2：使用圖鑑 ----------------
import json as _json

ITEMS = [
    {"id": "11", "name": "happy", "animated": False, "kind": "emoji", "uses": 50, "meaning": "開心的時候", "use": True},
    {"id": "12", "name": "dance", "animated": True, "kind": "emoji", "uses": 40, "meaning": "", "looks": "一隻貓在跳舞", "use": True},
    {"id": "13", "name": "e6fe004e190c4f83a2866f8631078c32", "animated": False, "kind": "emoji", "uses": 30,
     "meaning": "傻眼無言", "use": True},
    {"id": "14", "name": "off", "animated": False, "kind": "emoji", "uses": 99, "meaning": "不想被用", "use": False},
    {"id": "15", "name": "blank", "animated": False, "kind": "emoji", "uses": 5, "meaning": "", "looks": "", "use": True},
    {"id": "21", "name": "wave", "animated": False, "kind": "sticker", "uses": 0, "looks": "揮手的角色", "use": True},
]


def _set():
    return cat.EmojiSet("g1", ITEMS)


def test_only_use_true_with_description():
    s = _set()
    assert {e.name for e in s.entries} == {"happy", "dance", "e6fe004e190c4f83a2866f8631078c32", "wave"}
    block = s.prompt_block()
    assert ":happy: 開心的時候" in block and ":dance: 一隻貓在跳舞" in block and "[貼圖:wave]" in block
    assert "off" not in block and "blank" not in block


def test_hard_names_get_alias():
    s = _set()
    block = s.prompt_block()
    assert "e6fe004e" not in block and ":e1: 傻眼無言" in block
    text, st = s.process("好喔 :e1:")
    assert text == "好喔 <:e6fe004e190c4f83a2866f8631078c32:13>" and st is None


def test_process_replace_animated_and_delete_unknown():
    s = _set()
    assert s.process("哈 :dance:") == ("哈 <a:dance:12>", None)
    assert s.process("哈 :nothing: 好") == ("哈 好", None)          # 壞名稱刪掉、空白收乾淨
    assert s.process("現在 12:30:45 開會") == ("現在 12:30:45 開會", None)  # 時間不是表情
    assert s.process(":off:") == ("", None)                           # use=false 當不存在


def test_process_one_per_message():
    s = _set()
    text, st = s.process(":happy: 然後 :dance: 還有 :happy:")
    assert text == "<:happy:11> 然後 還有" and st is None
    text, st = s.process("好 [貼圖:wave]")  # 貼圖拿掉、另外回 ID
    assert (text, st) == ("好", "21")
    text, st = s.process(":happy: 好 [貼圖:wave]")  # 表情已經用掉，貼圖就不發
    assert (text, st) == ("<:happy:11> 好", None)
    text, st = s.process("好 [貼圖:wave] [貼圖:wave]")
    assert (text, st) == ("好", "21")
    assert s.process("[貼圖:不存在]") == ("", None)
    assert s.process("<:fake:999> 好") == ("好", None)      # 不在圖鑑裡的完整寫法也刪
    assert s.process("<:happy:11>") == ("<:happy:11>", None)


def test_process_without_catalog_cleans_tags():
    assert cat.process_reply("好 :smile: 啊 [貼圖:abc]", None) == ("好 啊", None)
    assert cat.process_reply("看 <:a:1> 這個", None) == ("看 <:a:1> 這個", None)  # 本來就有的不動
    assert cat.process_reply("普通文字 :) 沒事", None) == ("普通文字 :) 沒事", None)


def test_duplicate_sticker_names_get_alias():
    s = cat.EmojiSet("g", [{"id": "1", "name": "x", "kind": "sticker", "uses": 0, "looks": "甲"},
                           {"id": "2", "name": "x", "kind": "sticker", "uses": 0, "looks": "乙"}])
    assert [e.label for e in s.entries] == ["x", "s1"]


def test_react_name():
    s = _set()
    assert s.react_name(":e1:") == (True, "e6fe004e190c4f83a2866f8631078c32")
    assert s.react_name("happy") == (True, "happy")
    assert s.react_name("<:happy:11>") == (True, "happy")
    assert s.react_name("nothing") == (True, "")
    assert s.react_name("👍") == (False, "👍")
    assert "dance" in s.react_block() and "wave" not in s.react_block()


def test_sync_removed_renamed_new():
    from types import SimpleNamespace as NS
    s = _set()
    rep = s.sync([NS(id=11, name="happy2"), NS(id=12, name="dance"), NS(id=99, name="brand_new")])
    assert rep["renamed"] == [("happy", "happy2")] and rep["new"] == ["brand_new"]
    assert set(rep["removed"]) == {"e6fe004e190c4f83a2866f8631078c32", "wave"}
    assert s.process(":happy2: :happy:") == ("<:happy2:11>", None)


def test_reviewed_false_not_used(tmp_path):
    p = tmp_path / "g1.json"
    p.write_text(_json.dumps(ITEMS), encoding="utf-8")
    assert cat.load_all(tmp_path) == {}                        # 沒有 meta 檔＝沒看過
    cat.ensure_meta(p)
    assert _json.loads(cat.meta_path(p).read_text()) == {"reviewed": False}
    assert cat.load_all(tmp_path) == {}
    cat.meta_path(p).write_text('{"reviewed": true}')
    cat.ensure_meta(p)                                          # 重建圖鑑不會把 true 改回去
    assert list(cat.load_all(tmp_path)) == ["g1"]
    assert cat.load_set(p, require_reviewed=False) is not None


def test_load_dict_format_and_broken(tmp_path):
    (tmp_path / "g2.json").write_text(_json.dumps({"reviewed": True, "items": ITEMS}), encoding="utf-8")
    (tmp_path / "g3.json").write_text("{壞掉", encoding="utf-8")
    assert list(cat.load_all(tmp_path)) == ["g2"]
