"""階段 B 程式審查修正（9 項）的單元測試：用假物件，不需要網路、模型、Discord。"""
import json
import time
from dataclasses import replace
from types import SimpleNamespace as NS

import discord
import pytest

from test_chime import (AMING, GOOD, XIAOJIE, XIAOMEI, YUKI, Clock, JsonLLM, fake_bot, fake_message, records, run,
                        traces)  # noqa: F401  (traces 是 fixture)
from yuki import bot as botmod, prompts, throttle, trace, vision
from yuki.bot import Turn
from yuki.chime import ChimeLimits
from yuki.config import Settings
from yuki.memory.writer import ChatLine, ReplyRef
from yuki.throttle import is_flood


# ---------- on_message 的假物件 ----------
def fake_on_message(*, active=False, chiming=False, called=False, chime_ok=True, test_guild=True, level=1,
                    **settings):
    s = replace(Settings(), **{"vision_enabled": False, "chime_in": level, **settings})
    clk = Clock()
    ev = NS(scheduled=[], called=[], recorded=[], buffered=[], observed=[], triggers=[])
    me = NS(settings=s, _last_activity=0.0, is_test_guild=lambda g: test_guild, user=NS(id=99),
            sync=NS(record=lambda m: ev.recorded.append(m)), clean=lambda m: m.content,
            is_called=lambda m: called, active_until={}, chime_until={},
            limits=ChimeLimits(level, clock=clk), throttle=throttle.Throttle(clock=clk),
            brain=NS(observe=lambda k, l: ev.observed.append(l) or True,
                     buffer=lambda *a, **k: ev.buffered.append(a)),
            reply_ref=lambda m: (None, ""), _cancel_debounce=lambda cid: None,
            chime_allowed=lambda g: chime_ok,
            _schedule_chime=lambda m, l, attention: ev.scheduled.append(attention))

    async def on_called(message, author, text, mentions, *, count_tag=True, trigger="called"):
        ev.called.append((text, count_tag))
        ev.triggers.append(trigger)

    me._on_called = on_called
    if active:
        me.active_until[5] = time.monotonic() + 100
    if chiming:
        me.chime_until[5] = time.monotonic() + 100
    return me, ev


def fake_incoming(content, author=AMING):
    async def reply(*a, **k):
        return None

    return NS(author=NS(id=int(author.id), display_name=author.name, name=author.name, bot=False),
              webhook_id=None, type=discord.MessageType.default, content=content, id=1, mentions=[],
              guild=NS(id=7, me=NS(display_name="Yuki", roles=[])), channel=NS(id=5), reply=reply,
              reference=None, attachments=[], embeds=[], role_mentions=[])


def on_message(me, msg):
    return run(botmod.YukiBot.on_message(me, msg))


# ---------- 1、2：她自己插話後的注意時間 ----------
def test_fix1_shush_after_her_own_chime_mutes_without_mention():
    me, _ = fake_on_message(chiming=True)
    on_message(me, fake_incoming("沒問你"))
    assert me.limits.is_muted(5)
    me2, _ = fake_on_message()  # 她沒講過話、沒人叫她：不理
    on_message(me2, fake_incoming("沒問你"))
    assert not me2.limits.is_muted(5)


def test_fix2_follow_up_after_her_chime_is_attention():
    me, ev = fake_on_message(chiming=True)
    on_message(me, fake_incoming("對啊那場真的很精彩"))
    assert ev.scheduled == [True]
    me2, ev2 = fake_on_message()
    on_message(me2, fake_incoming("對啊那場真的很精彩"))
    assert ev2.scheduled == [False]


def test_fix1_2_respond_sets_chime_until_only_for_chimed_turns():
    class Typing:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    async def reply_fn(*a, **k):
        return "恭喜"

    async def msg_reply(*a, **k):
        return None

    async def send(*a, **k):
        return None

    def make():
        me, _ = fake_bot(JsonLLM(GOOD))
        me.chime_until, me.active_until, me._replying = {}, {}, 0
        me.voice = NS(describe=lambda g, n: "")
        me.members_of = lambda g: []
        me.owner_name = lambda g: ""
        me.archive_guild = lambda g: ""
        me.user = NS(id=99, name="Yuki", display_name="Yuki")
        me.brain.reply = reply_fn
        me.brain.observe = lambda k, l: True
        me.brain.buffer = lambda *a, **k: None
        return me

    msg = NS(channel=NS(id=5, typing=lambda: Typing(), send=send), guild=NS(id=7), reply=msg_reply,
             author=NS(id=1, name="阿明", display_name="阿明"), reference=None)
    me = make()
    assert run(botmod.YukiBot._respond(me, Turn(msg, AMING, "x", [], trigger="chime", chimed=True)))
    assert me.chime_until[5] > 0 and 5 not in me.active_until
    me2 = make()
    run(botmod.YukiBot._respond(me2, Turn(msg, AMING, "x", [], trigger="called")))
    assert 5 not in me2.chime_until and 5 in me2.active_until


# ---------- 3：洗版判斷 ----------
def test_fix3_flood_judgement():
    assert not is_flood("我考上了！！！！！！！！！！")
    assert not is_flood("哈哈哈哈哈哈哈哈哈哈")
    assert not is_flood("10000000000")
    assert not is_flood("----------")
    assert not is_flood("wwwwwwwwwwwww")
    assert not is_flood("今天好開心" + "哈" * 20 + "笑死")
    assert is_flood("a" * 10)
    assert is_flood("阿" * 15)
    assert is_flood("嗚" * 12 + "好")  # 剩 1 字
    assert not is_flood("嗚" * 12 + "我今天真的很難過")  # 還有內容


def test_fix3_flood_message_is_archived_but_not_replied_or_remembered():
    me, ev = fake_on_message(test_guild=False, called=True)
    on_message(me, fake_incoming("a" * 20))
    assert len(ev.recorded) == 1  # 封存照常
    assert ev.observed == [] and ev.called == [] and ev.scheduled == []
    me2, ev2 = fake_on_message(test_guild=False, called=True)
    on_message(me2, fake_incoming("我考上了！！！！！！！！！！"))
    assert len(ev2.recorded) == 1 and len(ev2.called) == 1  # 不是洗版：照常處理


# ---------- 5：_run_turns ----------
def test_fix5_run_turns_survives_unexpected_error_and_keeps_queue():
    me, _ = fake_bot(JsonLLM(GOOD))
    done = []

    async def respond(turn):
        done.append(turn.text)
        if turn.text == "第一則":
            me.throttle.enqueue(5, Turn(NS(), AMING, "排隊的", [], trigger="called"))
            raise RuntimeError("訊息被刪了")
        return True

    me._respond = respond
    me.throttle.try_start(5)
    run(botmod.YukiBot._run_turns(me, 5, Turn(NS(), AMING, "第一則", [], trigger="called")))
    assert done == ["第一則", "排隊的"]
    assert not me.throttle.is_busy(5)


# ---------- 6：LLM 忙碌再檢查 ----------
def test_fix6_busy_rechecked_after_slow_prep(traces):
    line = ChatLine(AMING, "我考上了")
    llm = JsonLLM(GOOD)
    me, _ = fake_bot(llm, history=[line])

    async def slow_fill(channel):
        me._replying = 1  # 補抓回覆對象的時候，別人的回覆開始了

    me._fill_reply_refs = slow_fill
    assert run(me.chime(fake_message(), line, False)) is None
    assert llm.calls == []
    assert json.loads(records(traces)[0]["final"])["action"] == "skipped:llm_busy"


# ---------- 7：tag 洗版冷卻 ----------
def test_fix7_tag_cooling_blocks_attention_paths():
    me, ev = fake_on_message(active=True, chime_ok=False)  # 主伺服器注意模式
    on_message(me, fake_incoming("接下來呢"))
    assert len(ev.called) == 1 and ev.called[0][1] is False
    me2, ev2 = fake_on_message(active=True, chime_ok=False)
    for _ in range(4):
        me2.throttle.note_tag("1")  # 超過 3 次：洗版冷卻
    assert me2.throttle.tag_cooling("1")
    on_message(me2, fake_incoming("接下來呢"))
    assert ev2.called == []
    me3, ev3 = fake_on_message()  # 插話判斷
    for _ in range(4):
        me3.throttle.note_tag("1")
    on_message(me3, fake_incoming("今天考試考得不錯"))
    assert ev3.scheduled == []
    me4, ev4 = fake_on_message()
    on_message(me4, fake_incoming("今天考試考得不錯", author=XIAOJIE))
    assert ev4.scheduled == [False]


# ---------- 8：prompt ----------
def test_fix8_prompts():
    assert "要看聊天紀錄的事" not in prompts.CHIME_SYSTEM
    assert not hasattr(prompts, "ATTENTION_RULES")


# ---------- 9：主伺服器注意模式 ----------
def test_fix9_main_guild_attention_waits_for_embeds_and_buffers(monkeypatch):
    waited = []

    async def fake_wait(message):
        waited.append(1)

    monkeypatch.setattr(vision, "wait_for_embeds", fake_wait)
    monkeypatch.setattr(vision, "image_urls", lambda m: [])
    me, ev = fake_on_message(active=True, chime_ok=False, passive_enabled=False, vision_enabled=True)
    on_message(me, fake_incoming("接下來呢"))
    assert waited == [1] and len(ev.called) == 1
    assert len(ev.buffered) == 1  # 注意模式內的訊息也進記憶 buffer
    me2, ev2 = fake_on_message(chime_ok=False, passive_enabled=False, vision_enabled=True)
    on_message(me2, fake_incoming("沒人叫她"))
    assert waited == [1] and ev2.buffered == []


# ---------- 4：追蹤紀錄的隱私 ----------
def test_fix4_chime_trace_keeps_no_prompts(traces):
    async def go():
        async with trace.start("chime", keep_prompts=False) as tr:
            tr.add_attempt([{"role": "user", "content": "[阿明] 私密內容"}], {"x": 1}, 5)
            tr.final = "{}"
        async with trace.start("reply") as tr:
            tr.add_attempt([{"role": "user", "content": "[阿明] 私密內容"}], {"x": 1}, 5)
    run(go())
    a, b = records(traces)
    assert a["attempts"][0]["messages"] == [] and a["attempts"][0]["response"] == {"x": 1}
    assert b["attempts"][0]["messages"][0]["content"] == "[阿明] 私密內容"


def optout(me, *ids):
    for i in ids:
        me.archive.db.execute("INSERT INTO optout VALUES ('g', ?)", (i,))


def all_text(tmp_path):
    return "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("ai-*.jsonl"))


def test_fix4_optout_people_scrubbed_from_trace_context(traces):
    hist = [ChatLine(AMING, "我的秘密計畫是搬家"),
            ChatLine(XIAOJIE, "真的假的", reply_to=ReplyRef("阿明", False, "我的秘密計畫是搬家")),
            ChatLine(XIAOMEI, "小美說了普通的話"),
            ChatLine(YUKI, "好喔", is_bot=True)]
    me, _ = fake_bot(JsonLLM(GOOD), history=hist)
    optout(me, "1")  # 阿明用過 /不要記我

    async def go():
        async with trace.start("reply", guild="7", user_ids=["2"]) as tr:
            me._hide_optouts(tr, "5", [])
            msgs = [{"role": "system", "content": "在場的人：阿明、小傑"},
                    {"role": "user", "content": "\n".join(l.render_chat() for l in hist if not l.is_bot)}]
            tr.add_attempt(msgs, {"content": "阿明的事我不知道"}, 3)
            tr.final = "阿明的事我不知道"
    run(go())
    text = all_text(traces)
    assert "阿明" not in text and "秘密計畫" not in text
    assert "真的假的" in text and "小美說了普通的話" in text  # 其他人照常


def test_fix4_optout_replied_to_person_not_in_history_is_scrubbed(traces):
    hist = [ChatLine(XIAOJIE, "你說的對", reply_to=ReplyRef("阿明", False, "早就過期的那句話"))]
    me, _ = fake_bot(JsonLLM(GOOD), history=hist)
    optout(me, "1")

    async def go():
        async with trace.start("chime", guild="7", user_ids=["2"]) as tr:
            me._hide_optouts(tr, "5", [AMING])
            tr.add_attempt([{"role": "user", "content": hist[0].render_chat()}], None, 1)
    run(go())
    text = all_text(traces)
    assert "阿明" not in text and "早就過期" not in text and "你說的對" in text


def test_fix4_chime_through_bot_hides_optout_in_decision_and_final(traces):
    line = ChatLine(XIAOJIE, "我考上了")
    hist = [ChatLine(AMING, "我偷偷報名了比賽"), line]
    llm = JsonLLM({**GOOD, "reason": "阿明也在場"})
    me, _ = fake_bot(llm, history=hist)
    optout(me, "1")
    run(me.chime(fake_message(author=XIAOJIE), line, False))
    text = all_text(traces)
    assert "阿明" not in text and "偷偷報名" not in text


def test_fix4_trigger_optout_still_skips_whole_record(tmp_path):
    trace.reset()
    trace.configure(tmp_path, opted_out=lambda g, uids: "1" in uids)

    async def go():
        async with trace.start("chime", guild="7", user_ids=["1"]):
            pass
    run(go())
    trace.reset()
    assert list(tmp_path.glob("ai-*.jsonl")) == []
