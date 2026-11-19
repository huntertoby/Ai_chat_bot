"""C1-2：回覆與按反應用群組表情和貼圖（不需要模型或 Discord）。"""
from types import SimpleNamespace as NS

from yuki import chime, emoji_catalog as cat, prompts
from yuki.bot import YukiBot
from yuki.brain import _NAME_PREFIX
from yuki.memory.writer import ChatLine
from yuki.rag.decompose import Person
from yuki.voice.tts import sentences, speakable

ITEMS = [
    {"id": "11", "name": "happy", "kind": "emoji", "uses": 50, "meaning": "開心", "use": True},
    {"id": "12", "name": "dance", "kind": "emoji", "animated": True, "uses": 40, "looks": "貓在跳舞", "use": True},
    {"id": "21", "name": "wave", "kind": "sticker", "uses": 0, "looks": "揮手", "use": True},
]


def _bot(emoji_sets, emoji_ids=(), test_ids=("1",)):
    return NS(settings=NS(emoji_guild_ids=list(emoji_ids), test_guild_ids=list(test_ids)), emoji_sets=emoji_sets)


def test_emoji_set_only_own_guild_and_allowed_guilds():
    sets = {"1": cat.EmojiSet("1", ITEMS), "9": cat.EmojiSet("9", ITEMS)}
    b = _bot(sets)
    assert YukiBot.emoji_set(b, NS(id=1)) is sets["1"]
    assert YukiBot.emoji_set(b, NS(id=9)) is None           # 圖鑑有，但 EMOJI_GUILD_IDS 留空＝只有測試伺服器
    assert YukiBot.emoji_set(b, None) is None
    b.settings.emoji_guild_ids = ["9"]
    assert YukiBot.emoji_set(b, NS(id=9)) is sets["9"] and YukiBot.emoji_set(b, NS(id=1)) is None


def test_test_guild_without_own_catalog_does_not_borrow_main():
    """測試伺服器（1）沒有自己的圖鑑，主伺服器（9）有：不借。"""
    b = _bot({"9": cat.EmojiSet("9", ITEMS)}, test_ids=("1",))
    assert YukiBot.emoji_set(b, NS(id=1)) is None


def test_unreviewed_catalog_not_loaded(tmp_path):
    (tmp_path / "1.json").write_text('[{"id":"1","name":"a","kind":"emoji","meaning":"x"}]', encoding="utf-8")
    assert cat.load_all(tmp_path) == {}


def test_prompt_has_emoji_block_only_when_given():
    block = cat.EmojiSet("1", ITEMS).prompt_block()
    p = prompts.reply_system("Yuki", [], None, emoji=block)
    assert ":happy: 開心" in p and "[貼圖:wave]" in p and "群組自己的表情與貼圖" in p
    assert "群組自己的表情與貼圖" not in prompts.reply_system("Yuki", [], None)
    assert "群組自己的表情與貼圖" not in prompts.reply_system("Yuki", [], None, emoji=block, spoken=True)


def test_persona_wording_no_sample_sentences():
    assert "少用表情符號和 XD，偶爾用一下就好" not in prompts.PERSONA
    assert "不要每句都用" in prompts.PERSONA
    assert "名稱」" not in prompts.EMOJI_RULES


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


# ---------- bot 的流程（假物件）----------
import asyncio

from tests.test_chime import AMING, fake_bot, fake_message, JsonLLM  # noqa: E402
from yuki import bot as botmod  # noqa: E402
from yuki.bot import Turn  # noqa: E402


def _respond_with(reply_text, eset, pacing=None):
    class Typing:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    sent, seen = [], {}

    async def reply_fn(*a, **k):
        seen.update(k)
        return reply_text

    async def msg_reply(content=None, **k):
        sent.append((content, k))

    async def send(content=None, **k):
        sent.append((content, k))

    me, _ = fake_bot(JsonLLM({}))
    me.chime_until, me.active_until, me._replying = {}, {}, 0
    me.voice = NS(describe=lambda g, n: "")
    me.members_of = lambda g: []
    me.owner_name = lambda g: ""
    me.archive_guild = lambda g: ""
    me.user = NS(id=99, name="Yuki", display_name="Yuki")
    me.emoji_set = lambda g: eset
    me.emoji_pacing = pacing or cat.EmojiPacing()
    me.brain.reply = reply_fn
    me.brain.observe = lambda k, l: True
    me.brain.buffer = lambda *a, **k: None
    msg = NS(channel=NS(id=5, typing=lambda: Typing(), send=send), guild=NS(id=7), reply=msg_reply,
             author=NS(id=1, name="阿明", display_name="阿明"), reference=None)
    ok = asyncio.run(botmod.YukiBot._respond(me, Turn(msg, AMING, "x", [], trigger="called")))
    return ok, sent, seen


def test_respond_replaces_emoji_and_passes_catalog_block():
    eset = cat.EmojiSet("7", ITEMS)
    ok, sent, seen = _respond_with("好喔 :happy: :dance:", eset)
    assert ok and sent == [("好喔 <:happy:11>", {"mention_author": False})]
    assert ":happy:" in seen["emoji"]


def test_respond_sticker_attached_and_text_cleaned():
    eset = cat.EmojiSet("7", ITEMS)
    ok, sent, _ = _respond_with("晚安 [貼圖:wave]", eset)
    (content, kw), = sent
    assert content == "晚安" and [o.id for o in kw["stickers"]] == [21]
    ok, sent, _ = _respond_with("[貼圖:wave]", eset)  # 只有貼圖
    (content, kw), = sent
    assert content is None and [o.id for o in kw["stickers"]] == [21]


def test_respond_without_catalog_cleans_and_sends_no_block():
    ok, sent, seen = _respond_with("哈 :xxx: 好 [貼圖:abc]", None)
    assert sent == [("哈 好", {"mention_author": False})] and seen["emoji"] == ""


def test_bot_chime_react_with_catalog_alias_and_restriction(tmp_path):
    from tests.test_chime import GOOD, records
    from yuki import trace
    trace.reset()
    trace.configure(tmp_path)
    items = ITEMS + [{"id": "13", "name": "e6fe004e190c4f83a2866f8631078c32", "kind": "emoji", "uses": 30,
                      "meaning": "傻眼", "use": True}]
    eset = cat.EmojiSet("7", items)
    line = ChatLine(Person("1", "阿明"), "這也太扯")
    guild_emojis = [NS(name="happy", available=True, id=11), NS(name="outside", available=True, id=77),
                    NS(name="e6fe004e190c4f83a2866f8631078c32", available=True, id=13)]

    def run_with(emoji):
        llm = JsonLLM({**GOOD, "action": "react", "emoji": emoji, "kind": "other"})
        me, _ = fake_bot(llm, history=[line])
        me.emoji_set = lambda g: eset
        got = []
        asyncio.run(me.chime(fake_message(reactions=got, emojis=guild_emojis), line, False))
        return got, llm

    got, llm = run_with("e1")                      # 別名 → 真名 → 群組表情物件
    assert [g.id for g in got] == [13]
    assert ":happy: 開心" in llm.calls[0][0]["content"] and ":e1: 傻眼" in llm.calls[0][0]["content"]
    got, _ = run_with(":happy:")
    assert [g.id for g in got] == [11]
    got, _ = run_with("outside")                  # 群組有、但圖鑑沒有 → 不按
    assert got == []
    got, _ = run_with("🎉")                        # unicode 照舊
    assert got == ["🎉"]
    assert records(tmp_path)


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


def test_respond_pacing_next_replies_have_no_list_and_no_emoji():
    eset = cat.EmojiSet("7", ITEMS)
    pacing = cat.EmojiPacing(skip=2)
    ok, sent, seen = _respond_with("好喔 :happy:", eset, pacing)
    assert seen["emoji"] and "<:happy:11>" in sent[0][0]
    for _ in range(2):  # 接下來兩則：不放清單，模型就算寫了也刪掉
        ok, sent, seen = _respond_with("好喔 :happy:", eset, pacing)
        assert seen["emoji"] == "" and sent[0][0] == "好喔"
    ok, sent, seen = _respond_with("好喔 :happy:", eset, pacing)
    assert seen["emoji"] and "<:happy:11>" in sent[0][0]
