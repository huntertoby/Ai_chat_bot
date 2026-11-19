"""B2 插話判斷、B4 先查再接話（yuki/chime.py、prompts、brain、bot 的插話流程）：不需要網路、模型、Discord。"""
import asyncio
import json
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace as NS

import pytest

from yuki import chime, prompts, trace
from yuki.brain import Brain, to_messages
from yuki.chime import ChimeLimits, Decision, decide, parse_decision
from yuki.config import Settings
from yuki.llm import LLMError
from yuki.chat import ChatLine, Person, ReplyRef

AMING = Person("1", "阿明")
XIAOJIE = Person("2", "小傑")
XIAOMEI = Person("3", "小美")
YUKI = Person("9", "Yuki")


def run(coro):
    return asyncio.run(coro)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class JsonLLM:
    """chat_json 回固定內容（或丟例外），記下收到的 messages 和參數。"""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls, self.kw = reply, error, [], []

    async def chat_json(self, messages, schema, **kw):
        self.calls.append(messages)
        self.kw.append((schema, kw))
        if self.error:
            raise self.error
        return self.reply


GOOD = {"reason": "好消息", "kind": "congrats", "action": "reply", "participate": 0.9, "emoji": "",
        "need_background": []}


# ---------- 回覆關係 ----------
def test_render_chat_with_reply_to_member_and_to_yuki():
    a = ChatLine(XIAOJIE, "不是啦", reply_to=ReplyRef("阿明", False, "你昨天幾點睡"))
    assert a.render_chat() == "[小傑 回覆 阿明「你昨天幾點睡」] 不是啦"
    b = ChatLine(XIAOJIE, "我在問你", reply_to=ReplyRef("Yuki", True, "我也不確定"))
    assert b.render_chat() == "[小傑 回覆你「我也不確定」] 我在問你"
    assert ChatLine(XIAOJIE, "嗨").render_chat() == "[小傑] 嗨"
    assert ChatLine(XIAOJIE, "嗨", reply_to=ReplyRef("阿明", False, "")).render_chat() == "[小傑 回覆 阿明] 嗨"


def test_reply_snippet_truncated_to_40_chars_and_flattened():
    ref = chime.make_ref("阿明", False, "一" * 60 + "\n二")
    assert ref.snippet == "一" * 40
    assert chime.make_ref("阿明", False, "a\nb  c").snippet == "a b c"


def test_to_messages_uses_reply_format_and_keeps_plain_render_for_memory():
    lines = [ChatLine(AMING, "在嗎"), ChatLine(XIAOJIE, "在", reply_to=ReplyRef("阿明", False, "在嗎"))]
    msgs = to_messages(lines)
    assert msgs == [{"role": "user", "content": "[阿明] 在嗎\n[小傑 回覆 阿明「在嗎」] 在"}]
    assert lines[1].render() == "[小傑] 在"  # 記憶整理用的格式不變


def test_fill_replies_uses_history_first_then_fetches_up_to_limit():
    target = ChatLine(AMING, "原本那句", msg_id="100")
    in_hist = ChatLine(XIAOJIE, "接話", reply_id="100", msg_id="101")
    outside = [ChatLine(XIAOMEI, f"回{i}", reply_id=str(i), msg_id=str(200 + i)) for i in range(5)]
    fetched = []

    async def fetch(mid):
        fetched.append(mid)
        return None if mid == "4" else ReplyRef(f"人{mid}", False, f"內容{mid}")

    n = run(chime.fill_replies([target, in_hist, *outside], fetch, limit=3))
    assert in_hist.reply_to == ReplyRef("阿明", False, "原本那句")
    assert n == 3 and fetched == ["4", "3", "2"]  # 新的優先，最多 3 則
    assert outside[4].reply_to is None and outside[4].reply_id == ""  # 失敗：不再重試
    assert outside[3].reply_to.name == "人3"
    assert outside[0].reply_to is None and outside[0].reply_id == "0"  # 沒輪到，下次再補


def test_fill_replies_swallows_fetch_errors():
    line = ChatLine(XIAOJIE, "x", reply_id="5")

    async def boom(mid):
        raise RuntimeError("404")

    assert run(chime.fill_replies([line], boom)) == 1
    assert line.reply_to is None and line.reply_id == ""


def test_fill_replies_marks_yuki_lines():
    mine = ChatLine(YUKI, "我說的", is_bot=True, msg_id="7")
    line = ChatLine(XIAOJIE, "對啊", reply_id="7")
    run(chime.fill_replies([mine, line], None))
    assert line.reply_to.is_bot and line.render_chat().startswith("[小傑 回覆你")


# ---------- 判斷結果整理 ----------
def test_parse_decision_clamps_and_defaults():
    d = parse_decision({"participate": 1.7, "action": "reply", "kind": "congrats", "emoji": " 🎉 ", "reason": "x",
                        "need_background": ["a", "", "b", "c", "d"]})
    assert d.participate == 1.0 and d.emoji == "🎉" and d.need_background == ["a", "b", "c"]
    d = parse_decision({"participate": "abc", "action": "shout", "kind": "???", "need_background": "x"})
    assert (d.participate, d.action, d.kind, d.need_background) == (0.0, "none", "other", [])
    assert parse_decision(None).error == "no_json"
    assert parse_decision({"participate": -1}).participate == 0.0


def test_decision_threshold_rules():
    d = Decision(participate=0.7, action="reply")
    assert d.should_reply(0.7) and not d.should_reply(0.71)
    assert not Decision(participate=0.9, action="react", emoji="").should_react(0.0)
    assert Decision(participate=0.1, action="react", emoji="🎉").should_react(0.0)
    assert not Decision(participate=0.9, action="reply", error="llm").should_reply(0.5)
    assert not Decision(participate=0.9, action="none").should_reply(0.5)


# ---------- decide：純函式介面 ----------
def test_decide_builds_three_sections_and_uses_short_json_call():
    llm = JsonLLM(GOOD)
    recent = [ChatLine(AMING, "今天好累"), ChatLine(YUKI, "辛苦了", is_bot=True),
              ChatLine(XIAOJIE, "對啊", reply_to=ReplyRef("Yuki", True, "辛苦了"))]
    d = run(decide(llm, recent=recent, latest=ChatLine(AMING, "我考上了"), bot_name="Yuki"))
    assert d.action == "reply" and d.kind == "congrats" and d.participate == 0.9 and d.ms >= 0 and not d.error
    system, user = llm.calls[0]
    assert system["role"] == "system" and "[你是誰]" in system["content"] and "Yuki" in system["content"]
    text = user["content"]
    assert "[最近的對話]" in text and "[最新訊息]" in text
    assert "[Yuki（你）] 辛苦了" in text and "[小傑 回覆你「辛苦了」] 對啊" in text
    assert text.rstrip().endswith("[阿明] 我考上了")
    schema, kw = llm.kw[0]
    assert kw["reasoning_effort"] == "none" and kw["max_tokens"] <= 300 and schema is prompts.CHIME_SCHEMA


def test_decide_keeps_only_last_15_recent_lines():
    recent = [ChatLine(AMING, f"第{i}則") for i in range(30)]
    llm = JsonLLM(GOOD)
    run(decide(llm, recent=recent, latest=ChatLine(XIAOJIE, "最新"), bot_name="Yuki"))
    text = llm.calls[0][1]["content"]
    assert "第14則" not in text and "第15則" in text and "第29則" in text


def test_decide_attaches_images_and_attention_hint():
    llm = JsonLLM(GOOD)
    run(decide(llm, recent=[], latest=ChatLine(AMING, "（貼了圖片）"), bot_name="Yuki",
               images=["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"], attention=True))
    content = llm.calls[0][1]["content"]
    assert isinstance(content, list) and [p["type"] for p in content] == ["text", "image_url", "image_url"]
    assert "2 張圖" in content[0]["text"] and prompts.CHIME_ATTENTION in content[0]["text"]
    llm2 = JsonLLM(GOOD)
    run(decide(llm2, recent=[], latest=ChatLine(AMING, "嗨"), bot_name="Yuki"))
    assert isinstance(llm2.calls[0][1]["content"], str) and prompts.CHIME_ATTENTION not in llm2.calls[0][1]["content"]


def test_decide_never_raises_on_llm_error_or_bad_json():
    d = run(decide(JsonLLM(error=LLMError("斷線")), recent=[], latest=ChatLine(AMING, "嗨"), bot_name="Yuki"))
    assert d.error.startswith("llm") and d.action == "none" and not d.should_reply(0.0)
    d = run(decide(JsonLLM(None), recent=[], latest=ChatLine(AMING, "嗨"), bot_name="Yuki"))
    assert d.error == "no_json" and d.action == "none"


# ---------- D-29：提到她才思考（兩段式）----------
def test_mentions_bot_heuristic():
    names = ["Yuki", "yuki"]
    yuki_said = [ChatLine(AMING, "嗨"), ChatLine(YUKI, "幹嘛", is_bot=True)]
    m = lambda text, recent=(), **kw: chime.mentions_bot(ChatLine(XIAOJIE, text, **kw), list(recent), names)
    assert m("Yuki 好爛") and m("@Yuki 在嗎") and m("yuki才是正品")
    assert not m("yukio 是誰")                                   # 英文名字要整個字相符
    assert m("這機器人蠻有趣的") and m("那個 AI 又在亂講") and m("bot 掛了")
    assert not m("我在 aim 練槍") and not m("他今天沒來")           # 她沒在最近講話：「他」是在講別人
    assert m("他不敢回答", yuki_said)                              # 她剛講完：「他」可能是她
    assert not m("你怎麼看", yuki_said)                            # 單純「你」不開思考（不思考就判得對）
    assert m("對啊", reply_to=ReplyRef("Yuki", True, "幹嘛"))       # 回覆她
    far = [ChatLine(YUKI, "幹嘛", is_bot=True)] + [ChatLine(AMING, f"{i}") for i in range(6)]
    assert not m("他不敢回答", far)                                # 她是好幾則以前講的


def test_decide_thinks_only_when_latest_mentions_her():
    llm = JsonLLM(GOOD)
    d = run(decide(llm, recent=[], latest=ChatLine(AMING, "這機器人好吵"), bot_name="Yuki", think_on_mention=True,
                   think_max_tokens=3000))
    _, kw = llm.kw[0]
    assert d.path == "think" and kw["reasoning_effort"] == "low" and kw["max_tokens"] == 3000
    assert d.to_dict()["path"] == "think"
    llm2 = JsonLLM(GOOD)
    d2 = run(decide(llm2, recent=[], latest=ChatLine(AMING, "我考上了"), bot_name="Yuki", think_on_mention=True))
    assert d2.path == "fast" and llm2.kw[0][1]["reasoning_effort"] == "none"
    llm3 = JsonLLM(GOOD)  # 預設關：提到她也不思考
    d3 = run(decide(llm3, recent=[], latest=ChatLine(AMING, "這機器人好吵"), bot_name="Yuki"))
    assert d3.path == "fast" and llm3.kw[0][1]["reasoning_effort"] == "none"


def test_d29_kinds_split_to_me_and_about_me():
    assert {"to_me", "about_me"} <= set(chime.KINDS)
    assert parse_decision({**GOOD, "kind": "to_me"}).kind == "to_me"
    assert set(prompts.CHIME_SCHEMA["properties"]["kind"]["enum"]) == set(chime.KINDS)
    assert {"to_me", "about_me"} <= set(chime.KIND_EXEMPT_IN_ATTENTION)
    order = prompts.CHIME_SCHEMA["required"]
    assert order.index("action") < order.index("reason")  # 先決定再寫理由


def test_llm_is_busy_follows_slots():
    sem = asyncio.Semaphore(1)
    llm = NS(_slots=sem)
    assert not chime.llm_is_busy(llm)

    async def go():
        async with sem:
            return chime.llm_is_busy(llm)

    assert run(go()) is True
    assert chime.llm_is_busy(NS()) is False


# ---------- 硬限制 ----------
def test_levels_threshold_and_interval():
    assert not ChimeLimits(0).enabled and ChimeLimits(0).gate("c") == "chime_off"
    assert (ChimeLimits(1).threshold, ChimeLimits(1).interval_sec) == (0.7, 600)
    assert (ChimeLimits(2).threshold, ChimeLimits(2).interval_sec) == (0.6, 300)
    assert (ChimeLimits(3).threshold, ChimeLimits(3).interval_sec) == (0.5, 120)
    assert ChimeLimits(1, threshold=0.8).threshold == 0.8


def test_channel_limit_once_per_interval_and_attention_is_exempt():
    clk = Clock()
    lim = ChimeLimits(1, clock=clk)
    assert lim.gate("c1") is None
    lim.record_reply("c1", "u1", "congrats")
    clk.t += 599
    assert lim.gate("c1") == "channel_limit"
    assert lim.gate("c1", attention=True) is None  # 注意模式不受 10 分鐘限制
    assert lim.gate("c2") is None
    clk.t += 2
    assert lim.gate("c1") is None


def test_attention_replies_do_not_use_up_the_channel_quota():
    lim = ChimeLimits(1, clock=Clock())
    lim.record_reply("c1", "u1", "about_me", attention=True)
    assert lim.gate("c1") is None


def test_kind_once_per_person_per_day():
    day = [date(2026, 11, 18)]
    lim = ChimeLimits(1, today=lambda: day[0])
    assert lim.kind_ok("u1", "congrats")
    lim.record_reply("c1", "u1", "congrats")
    assert not lim.kind_ok("u1", "congrats")
    assert lim.kind_ok("u1", "comfort") and lim.kind_ok("u2", "congrats")
    day[0] += timedelta(days=1)
    assert lim.kind_ok("u1", "congrats")


def test_kind_limit_exempts_conversation_kinds_only_in_attention():
    lim = ChimeLimits(1)
    lim.record_reply("c1", "u1", "about_me")
    assert not lim.kind_ok("u1", "about_me")  # 沒在注意模式：照樣限制
    assert lim.kind_ok("u1", "about_me", attention=True)
    assert lim.kind_ok("u1", "congrats")  # 沒記過 congrats
    lim.record_reply("c1", "u1", "addon", attention=True)
    assert not lim.kind_ok("u1", "addon", attention=True)  # 其他類別注意模式也限制


def test_mute_blocks_unsolicited_for_30_minutes_even_in_attention():
    clk = Clock()
    lim = ChimeLimits(1, mute_sec=1800, clock=clk)
    lim.mute("c1")
    assert lim.gate("c1") == "muted" and lim.gate("c1", attention=True) == "muted"
    assert lim.react_gate("c1") == "muted"
    assert lim.gate("c2") is None
    clk.t += 1801
    assert lim.gate("c1") is None


def test_reactions_have_their_own_quota():
    clk = Clock()
    lim = ChimeLimits(1, react_max=3, react_window_sec=600, clock=clk)
    for i in range(3):
        assert lim.react_gate("c1") is None
        lim.record_react("c1", f"u{i}", "other")
    assert lim.react_gate("c1") == "react_limit"
    assert lim.gate("c1") is None  # 按表情不佔插話的名額
    lim.record_reply("c1", "u9", "congrats")
    assert lim.react_gate("c1") == "react_limit"
    clk.t += 601
    assert lim.react_gate("c1") is None


def test_is_shush():
    for t in ["沒問你", "閉嘴啦", "乾你屁事", "yuki 閉嘴"]:
        assert chime.is_shush(t), t
    for t in ["我在問你", "謝謝", ""]:
        assert not chime.is_shush(t), t


def test_resolve_emoji():
    guild = [NS(name="cat_smirk", available=True), NS(name="gone", available=False)]
    assert chime.resolve_emoji("🎉", guild) == "🎉"
    assert chime.resolve_emoji(":cat_smirk:", guild) is guild[0]
    assert chime.resolve_emoji("<:Cat_Smirk:123>", guild) is guild[0]
    assert chime.resolve_emoji("cat_smirk", guild) is guild[0]
    assert chime.resolve_emoji(":nope:", guild) is None
    assert chime.resolve_emoji("gone", guild) is None
    assert chime.resolve_emoji("", guild) is None


# ---------- prompt：自我指稱、先查再接話（B4 第 1、2 層）----------
def test_persona_has_self_reference_and_reply_format_rules():
    sys = prompts.reply_system("Yuki")
    assert "第一人稱" in sys and "回覆你" in sys and "你」就是你本人" in sys


def test_background_rules_only_when_tools_exist():
    assert "先查再接話" not in prompts.reply_system("Yuki")
    for kw in ({"web": True}, {"archive": True}):
        sys = prompts.reply_system("Yuki", **kw)
        assert "先查再接話" in sys and "帶年份" in sys
        # D-27：成員的事先翻聊天紀錄，但不再禁止上網
        assert "先查群組聊天紀錄" in sys and "也可以上網" in sys and "絕對不上網" not in sys


def test_background_line_goes_into_system_only_with_items_and_tools():
    sys = prompts.reply_system("Yuki", web=True, archive=True, background=["某隊 奪冠", " ", "新改版"])
    assert "回覆前先查：某隊 奪冠、新改版" in sys
    assert "回覆前先查" not in prompts.reply_system("Yuki", web=True, background=[])
    assert "回覆前先查" not in prompts.reply_system("Yuki", background=["x"])  # 沒工具就不叫她查


def test_chimed_hint_only_when_chimed():
    assert prompts.CHIMED_RULES in prompts.reply_system("Yuki", chimed=True)
    assert prompts.CHIMED_RULES not in prompts.reply_system("Yuki")


# ---------- brain：去重、need_background 交給回覆、成員名字不上網 ----------
def make_brain(llm, **kw):
    settings = replace(Settings(), web_enabled=True, **kw)
    return Brain(settings, llm)


class ToolLLM:
    """記下 system、並假裝模型叫了一次搜尋。"""

    def __init__(self, query):
        self.query, self.system, self.tool_results = query, "", []
        self._slots = asyncio.Semaphore(1)

    async def chat_tools(self, messages, tools, call_tool, **kw):
        self.system = messages[0]["content"]
        self.tool_results.append(await call_tool("web_search", {"query": self.query}))
        return "好喔"

    async def chat(self, messages, **kw):
        return "好喔"

    async def aclose(self):
        pass


def test_observe_drops_consecutive_duplicates():
    brain = make_brain(ToolLLM("x"))
    assert brain.observe("c", ChatLine(AMING, "在嗎"))
    assert not brain.observe("c", ChatLine(AMING, "在嗎"))
    assert brain.observe("c", ChatLine(XIAOJIE, "在嗎"))
    assert brain.observe("c", ChatLine(AMING, "在嗎"))  # 中間隔了別人，不算連續
    assert [l.author.name for l in brain.history["c"]] == ["阿明", "小傑", "阿明"]


def test_need_background_reaches_the_agentic_system_prompt():
    llm = ToolLLM("某隊 2026 奪冠")
    brain = make_brain(llm)
    brain.observe("c", ChatLine(AMING, "聽說某隊奪冠了"))
    out = run(brain.reply("g", "c", AMING, "聽說某隊奪冠了", members=[AMING, XIAOJIE],
                          background=["某隊 奪冠"], chimed=True))
    assert out == "好喔"
    assert "回覆前先查：某隊 奪冠" in llm.system and prompts.CHIMED_RULES in llm.system
    # 沒有 need_background 就沒有那一行
    llm2 = ToolLLM("天氣")
    brain2 = make_brain(llm2)
    brain2.observe("c", ChatLine(AMING, "嗨"))
    run(brain2.reply("g", "c", AMING, "嗨", members=[AMING]))
    assert "回覆前先查" not in llm2.system


def test_member_names_can_be_searched_on_the_web():
    """D-27：可以拿成員的名字上網查（取消舊規則）。"""
    llm = ToolLLM("小傑 就讀 大學")
    brain = make_brain(llm)
    called = []

    async def fake_call(name, args):
        called.append(args)
        return "網路結果"

    brain.web.call = fake_call
    brain.observe("c", ChatLine(AMING, "小傑最近怎樣"))
    run(brain.reply("g", "c", AMING, "小傑最近怎樣", members=[AMING, XIAOJIE]))
    assert called == [{"query": "小傑 就讀 大學"}] and llm.tool_results == ["網路結果"]


# ---------- 追蹤紀錄（其他測試也用）----------
def records(tmp_path):
    out = []
    for p in sorted(tmp_path.glob("ai-*.jsonl")):
        out += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    return out


@pytest.fixture
def traces(tmp_path):
    trace.reset()
    trace.configure(tmp_path)
    yield tmp_path
    trace.reset()


# 插話判斷整條流程（去抖動、限制、按表情、追蹤紀錄）在 test_bot_flow.py（整合）和 test_scheduler.py


# ---------- D-32：logprobs 路徑已移除，一律用模型寫的 participate＋門檻 ----------
def test_decide_uses_written_participate_and_threshold_is_applied_outside():
    d = run(decide(JsonLLM({**GOOD, "participate": 0.65}), recent=[], latest=ChatLine(AMING, "我考上了"),
                   bot_name="Yuki"))
    assert d.participate == pytest.approx(0.65) and d.should_reply(0.6) and not d.should_reply(0.7)
    assert "prob_source" not in d.to_dict()


def test_decide_llm_error_returns_error_decision():
    d = run(decide(JsonLLM(None, error=LLMError("down")), recent=[], latest=ChatLine(AMING, "x"), bot_name="Yuki"))
    assert d.error and not d.should_reply(0.0)


def test_chime_threshold_setting_prefers_prob_threshold_over_old_name(monkeypatch):
    from yuki.config import load_settings
    monkeypatch.setenv("CHIME_THRESHOLD", "0.5")
    monkeypatch.delenv("CHIME_PROB_THRESHOLD", raising=False)
    assert Settings().chime_threshold == 0.5
    monkeypatch.setenv("CHIME_PROB_THRESHOLD", "0.8")
    assert Settings().chime_threshold == 0.8
    assert ChimeLimits(1, threshold=Settings().chime_threshold).threshold == 0.8
    assert ChimeLimits(1, threshold=None).threshold == 0.7  # 沒設就照 CHIME_IN 等級


def test_logprobs_setting_is_gone():
    assert not hasattr(Settings(), "chime_use_logprobs")
    assert not hasattr(chime, "action_prob") and not hasattr(chime, "action_dist")


def test_named_message_skips_channel_interval_but_not_mute():
    """D-32：提到她名字的訊息不受頻道「每 10 分鐘一次」限制；閉嘴照樣擋。"""
    t = [0.0]
    lim = ChimeLimits(1, clock=lambda: t[0])
    lim.record_reply("c", "u", "congrats")
    t[0] = 60
    assert lim.gate("c") == "channel_limit" and lim.gate("c", named=True) is None
    lim.mute("c")
    assert lim.gate("c", named=True) == "muted"
