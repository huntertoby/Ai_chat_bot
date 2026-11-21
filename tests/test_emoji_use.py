"""C1-2：回覆與按反應用群組表情和貼圖（不需要模型或 Discord）。"""
from types import SimpleNamespace as NS

from yuki import chime, emoji_catalog as cat, prompts
from yuki.brain import _NAME_PREFIX
from yuki.chat import ChatLine, Person
from yuki.voice.tts import sentences, speakable

ITEMS = [
    {"id": "11", "name": "happy", "kind": "emoji", "uses": 50, "meaning": "開心", "use": True},
    {"id": "12", "name": "dance", "kind": "emoji", "animated": True, "uses": 40, "looks": "貓在跳舞", "use": True},
    {"id": "21", "name": "wave", "kind": "sticker", "uses": 0, "looks": "揮手", "use": True},
]


def test_unreviewed_catalog_not_loaded(tmp_path):
    (tmp_path / "1.json").write_text('[{"id":"1","name":"a","kind":"emoji","meaning":"x"}]', encoding="utf-8")
    assert cat.load_all(tmp_path) == {}


def test_reply_system_has_no_emoji_list():
    """D-37：回覆 prompt 不放表情清單（回話寫好後另外挑）。"""
    assert not hasattr(prompts, "EMOJI_RULES")
    for kw in ({}, {"spoken": True}, {"chimed": True, "web": True, "archive": True}):
        p = prompts.reply_system("Yuki", **kw)
        assert "群組自己的表情與貼圖" not in p and "[貼圖:" not in p


def test_leading_sticker_tag_not_eaten_as_name_prefix():
    assert _NAME_PREFIX.sub("", "[貼圖:wave]") == "[貼圖:wave]"
    assert _NAME_PREFIX.sub("", "[阿明] 好") == "好"


def test_speakable_strips_emoji_and_sticker_markup():
    t = "好喔 <:happy:11> 哈 <a:dance:12> 欸 :happy: 對 [貼圖:wave]"
    out = speakable(t)
    assert "<" not in out and ":" not in out and "貼圖" not in out and "11" not in out
    assert "好喔" in out and "對" in out
    assert all("<:" not in s for s in sentences(t))
    assert speakable("12:30:45 開會") == "12:30:45 開會"


def test_chime_prompt_has_react_list_only_when_given():
    me = ChatLine(Person("a", "阿明"), "哈哈")
    plain = chime.build_messages([], me, "Yuki")[0]["content"]
    with_list = chime.build_messages([], me, "Yuki", emoji_block=":happy: 開心")[0]["content"]
    assert "可以按的群組表情" not in plain and ":happy: 開心" in with_list and "可以按的群組表情" in with_list
    assert with_list.startswith(plain)


def test_react_target_only_catalog_names_when_catalog():
    s = cat.EmojiSet("1", ITEMS)
    assert s.react_name("happy") == (True, "happy")
    assert s.react_name("not_there") == (True, "")
    assert s.react_name("🎉") == (False, "🎉")


# ---------- 整條流程（假 Discord，真的 bot）----------
import pytest  # noqa: E402

from botkit import TEST_GUILD, FakeLLM, Kit, User  # noqa: E402

GOOD = {"reason": "好笑", "kind": "other", "action": "react", "participate": 0.3, "emoji": "", "need_background": []}
PEOPLE = [User(1, "阿明"), User(2, "小傑"), User(3, "小美"), User(4, "阿凱")]


@pytest.fixture
def emoji_kit(tmp_path):
    made = []

    def make(eset, reply="好喔", decision=None, **kw):
        sets = {str(TEST_GUILD): eset} if eset is not None else {}
        k = Kit(tmp_path / f"k{len(made)}", llm=FakeLLM(reply, decision), emoji_sets=sets,
                emoji_guild_ids=[str(TEST_GUILD)], **kw)
        made.append(k)
        return k

    yield make
    for k in made:
        k.close()


def ask(kit, who=0):
    """叫她一次，回傳 (送出的 (內容, 參數))、這次回覆的 system prompt。"""
    n = len(kit.sent)
    kit.run(kit.say(PEOPLE[who], "yuki 你看"))
    sent = [(c, kw) for _, c, kw in kit.sent[n:]]
    return sent, kit.llm.reply_calls[-1][0]["content"]


def test_reply_still_turns_emoji_names_into_group_emoji_without_the_catalog_in_prompt(emoji_kit):
    """安全網：模型自己寫了 :名稱: 照舊換成群組表情，一則最多一個；回覆 system 不放清單。"""
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "好喔 :happy: :dance:")
    sent, system = ask(kit)
    assert sent == [("好喔 <:happy:11>", {"mention_author": False})]  # 一則最多一個
    assert ":happy: 開心" not in system


def test_reply_sticker_is_attached_and_the_tag_removed(emoji_kit):
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "晚安 [貼圖:wave]")
    (content, kw), = ask(kit)[0]
    assert content == "晚安" and [o.id for o in kw["stickers"]] == [21]
    only = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "[貼圖:wave]")  # 只有貼圖
    (content, kw), = ask(only)[0]
    assert content is None and [o.id for o in kw["stickers"]] == [21]


def test_without_catalog_made_up_emoji_and_stickers_are_cleaned(emoji_kit):
    kit = emoji_kit(None, "哈 :xxx: 好 [貼圖:abc]")
    sent, system = ask(kit)
    assert sent == [("哈 好", {"mention_author": False})] and "群組自己的表情與貼圖" not in system


def test_after_using_an_emoji_the_next_replies_get_no_emoji(emoji_kit):
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "好喔 :happy:")
    sent, _ = ask(kit, 0)
    assert "<:happy:11>" in sent[0][0]
    for who in (1, 2):  # 接下來兩則：模型就算寫了也刪掉
        sent, _ = ask(kit, who)
        assert sent[0][0] == "好喔"
    sent, _ = ask(kit, 3)
    assert "<:happy:11>" in sent[0][0]


def test_chime_reaction_uses_catalog_names_and_aliases_only(emoji_kit):
    items = ITEMS + [{"id": "13", "name": "e6fe004e190c4f83a2866f8631078c32", "kind": "emoji", "uses": 30,
                      "meaning": "傻眼", "use": True}]
    guild_emojis = [NS(name="happy", available=True, id=11), NS(name="outside", available=True, id=77),
                    NS(name="e6fe004e190c4f83a2866f8631078c32", available=True, id=13)]

    def react_with(emoji):
        kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), items), decision={**GOOD, "emoji": emoji},
                        chime_emoji_list=True)  # 預設關（CHIME_EMOJI_LIST）；這裡測開的時候
        kit.guild.emojis = guild_emojis
        msg = kit.say(PEOPLE[0], "這也太扯")
        kit.run(msg)
        return msg.reactions, kit.llm.chime_calls[0][0]["content"]

    got, system = react_with("e1")                 # 別名 → 真名 → 群組表情物件
    assert [g.id for g in got] == [13]
    assert ":happy: 開心" in system and ":e1: 傻眼" in system
    assert [g.id for g in react_with(":happy:")[0]] == [11]
    assert react_with("outside")[0] == []          # 群組有、但圖鑑沒有 → 不按
    assert react_with("🎉")[0] == ["🎉"]            # unicode 照舊


def test_chime_emoji_list_off_by_default_still_allows_unicode(emoji_kit):
    """CHIME_EMOJI_LIST 預設關：插話判斷不帶表情清單，但一般表情（unicode）照樣能按。"""
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), decision={**GOOD, "emoji": "😂"})
    assert kit.settings.chime_emoji_list is False
    msg = kit.say(PEOPLE[0], "這也太扯")
    kit.run(msg)
    assert ":happy:" not in kit.llm.chime_calls[0][0]["content"] and msg.reactions == ["😂"]


def test_pacing_skips_replies_after_use():
    p = cat.EmojiPacing(skip=2)
    seq = []
    for used_if_allowed in [True] * 8:
        ok = p.allow("c")
        seq.append(ok)
        p.record("c", ok and used_if_allowed)
    assert seq == [True, False, False, True, False, False, True, False]
    assert p.allow("other")                                    # 別的頻道各算各的


def test_process_allow_false_strips_everything():
    s = cat.EmojiSet("1", ITEMS)
    assert s.process("好 :happy: [貼圖:wave]") == ("好 <:happy:11>", None)
    assert cat.process_reply("好 :happy: [貼圖:wave]", s, allow=False) == ("好", None)


# ---------- D-37：回話寫好後挑表情 ----------
from yuki import emoji_pick  # noqa: E402


def test_picked_emoji_goes_after_the_reply_text(emoji_kit):
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "對啦我最爛")
    kit.llm.pick = {"emoji": "happy"}
    sent, system = ask(kit)
    assert sent == [("對啦我最爛 <:happy:11>", {"mention_author": False})]
    assert len(kit.llm.pick_calls) == 1 and ":happy: 開心" not in system
    pick_system, pick_user = kit.llm.pick_calls[0]
    assert ":happy: 開心" in pick_system["content"] and "對啦我最爛" in pick_user["content"]
    assert kit.bot.brain.history[str(kit.channel.id)][-1].text == "對啦我最爛 <:happy:11>"  # 照實際送出的記


def test_picked_sticker_is_sent_with_stickers_kwarg(emoji_kit):
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "晚安")
    kit.llm.pick = {"emoji": "wave"}
    (content, kw), = ask(kit)[0]
    assert content == "晚安" and [o.id for o in kw["stickers"]] == [21]


def test_pick_unknown_or_none_sends_reply_as_is(emoji_kit):
    for result in ({"emoji": ""}, {"emoji": "not_there"}, None):
        kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "好喔")
        kit.llm.pick = result if result is not None else {}
        assert ask(kit)[0] == [("好喔", {"mention_author": False})]


def test_pick_error_or_timeout_sends_reply_as_is(emoji_kit, monkeypatch):
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "好喔", )
    kit.llm.pick, kit.llm.pick_error = {"emoji": "happy"}, RuntimeError("boom")
    assert ask(kit)[0] == [("好喔", {"mention_author": False})]
    monkeypatch.setattr(emoji_pick, "TEXT_REPLY_BUDGET_SEC", 1.3)  # 測試不等 10 秒
    slow = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "好喔")
    slow.llm.pick, slow.llm.pick_delay = {"emoji": "happy"}, 3.0
    assert ask(slow)[0] == [("好喔", {"mention_author": False})]


def test_no_pick_without_catalog_or_when_pacing_says_no(emoji_kit):
    kit = emoji_kit(None, "好喔")
    ask(kit)
    assert kit.llm.pick_calls == []
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "好喔")
    kit.llm.pick = {"emoji": "happy"}
    ask(kit, 0)
    ask(kit, 1)
    ask(kit, 2)                                   # 剛用過：接下來兩則不挑
    assert len(kit.llm.pick_calls) == 1
    sent, _ = ask(kit, 3)
    assert len(kit.llm.pick_calls) == 2 and sent[0][0] == "好喔 <:happy:11>"


def test_no_pick_when_the_model_already_wrote_one(emoji_kit):
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "好喔 :dance:")
    kit.llm.pick = {"emoji": "happy"}
    assert ask(kit)[0] == [("好喔 <a:dance:12>", {"mention_author": False})]
    assert kit.llm.pick_calls == []


def test_chimed_reply_also_picks(emoji_kit):
    decision = {"reason": "好笑", "kind": "other", "action": "reply", "participate": 0.95, "emoji": "",
                "need_background": []}
    kit = emoji_kit(cat.EmojiSet(str(TEST_GUILD), ITEMS), "笑死", decision=decision)
    kit.llm.pick = {"emoji": "happy"}
    n = len(kit.sent)
    kit.run(kit.say(PEOPLE[0], "我剛剛跌倒把手機丟進水溝"))
    sent = [c for _, c, _ in kit.sent[n:]]
    assert sent == ["笑死 <:happy:11>"] and len(kit.llm.pick_calls) == 1


def test_voice_path_never_picks_emoji():
    """語音回覆（voice/session.py）完全不呼叫挑表情。"""
    from pathlib import Path
    src = (Path(emoji_pick.__file__).parent / "voice" / "session.py").read_text(encoding="utf-8")
    assert "emoji_pick" not in src and "_pick_emoji" not in src and "emoji=" not in src
