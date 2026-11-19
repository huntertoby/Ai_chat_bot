"""每頻道回覆排程器（yuki/scheduler.py）：不碰 Discord，用假的時鐘和假的回覆函式。"""
import asyncio
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from yuki.chime import Decision
from yuki.config import Settings
from yuki.chat import Person
from yuki.scheduler import ReplyScheduler, Turn

AMING, XIAOJIE, XIAOMEI = Person("1", "阿明"), Person("2", "小傑"), Person("3", "小美")
REPLY = Decision(participate=0.9, action="reply", kind="congrats")
REACT = Decision(participate=0.2, action="react", emoji="🎉", kind="other")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(level=1, **kw):
    clk = Clock()
    done: list[Turn] = []

    async def respond(turn):
        done.append(turn)

    s = replace(Settings(), chime_in=level, **kw)
    return ReplyScheduler(s, respond, clock=clk), clk, done


def turn(who=AMING, text="x", trigger="called"):
    return Turn(NS(), who, text, [], trigger=trigger)


# ---------- 合併 ----------
def test_merged_turn_replies_to_the_last_message_and_keeps_everyone_mentioned():
    m1, m2 = NS(id=1), NS(id=2)
    t = Turn.merge([Turn(m1, AMING, "a", [XIAOJIE]), Turn(m2, XIAOMEI, "b", [XIAOJIE, AMING])])
    assert t.message is m2 and t.text == "b" and [p.name for p in t.mentions] == ["小傑", "阿明"]
    assert t.trigger == "called" and not t.chimed
    assert Turn.merge([]) is None


def test_merged_turn_counts_as_called_only_if_someone_really_called_her():
    only_attention = [turn(AMING, "a", "attention"), turn(XIAOJIE, "b", "attention")]
    assert Turn.merge(only_attention).trigger == "attention"
    assert Turn.merge(only_attention + [turn(AMING, "Yuki 你說呢")]).trigger == "called"


# ---------- 被叫到 ----------
def test_call_while_replying_is_queued_and_answered_right_after_in_one_merged_reply():
    sch, _, done = make()

    async def scenario():
        sch.throttle.try_start(5)  # 正在回小美
        await sch.call(5, turn(AMING, "第一"), "now")
        await sch.call(5, turn(XIAOJIE, "第二"), "now")
        assert done == []
        await sch.run_turns(5, turn(XIAOMEI, "正在回的"))
    asyncio.run(scenario())
    assert [t.text for t in done] == ["正在回的", "第二"] and not sch.throttle.is_busy(5)


def test_a_failed_reply_does_not_lose_the_queued_ones_or_keep_the_channel_busy():
    sch, _, _ = make()
    done = []

    async def respond(t):
        done.append(t.text)
        if t.text == "第一則":
            sch.throttle.enqueue(5, turn(text="排隊的"))
            raise RuntimeError("訊息被刪了")

    sch._respond = respond
    sch.throttle.try_start(5)
    asyncio.run(sch.run_turns(5, turn(text="第一則")))
    assert done == ["第一則", "排隊的"] and not sch.throttle.is_busy(5)


def test_flooder_is_dropped_with_his_queued_calls_but_others_stay():
    sch, _, _ = make()
    sch.throttle.try_start(5)
    for i in range(3):
        assert sch.admit_call(5, "1") == "now"
        sch.submit(5, turn(AMING, f"阿明{i}"), "now")
    sch.submit(5, turn(XIAOJIE, "小傑"), "now")
    assert sch.admit_call(5, "1") is None  # 60 秒內第 4 次
    assert [t.text for t in sch.throttle.drain(5)] == ["小傑"]


def test_attention_follow_ups_do_not_count_against_the_person_but_do_count_for_the_channel():
    sch, clk, _ = make()
    for i in range(6):
        assert sch.admit_call(5, "1", count_tag=False) is not None  # 同一個人接話 6 次也不算洗版
        clk.t += 1
    assert sch.throttle.in_burst(5)  # 頻道被叫很多次：熱鬧模式


def test_busy_main_server_attention_switches_to_burst_mode_instead_of_replying_every_line():
    sch, clk, done = make()

    async def scenario():
        for i, who in enumerate([AMING, XIAOJIE] * 3):
            admitted = sch.admit_call(5, who.id, count_tag=False)
            await sch.call(5, turn(who, f"第 {i} 句", "attention"), admitted)
            clk.t += 5
    asyncio.run(scenario())
    # 第 4 句讓頻道進入熱鬧模式；之後的不再一句一句回，收集起來等間隔到了合併回一次
    assert len(done) == 4 and all(t.trigger == "attention" for t in done)
    assert [t.text for t in sch.throttle.drain(5)] == ["第 4 句", "第 5 句"]


# ---------- 注意時間 ----------
def test_attention_windows_and_leaving():
    sch, clk, _ = make()
    sch.extend_attention(5)
    assert sch.attention(5) == (True, False)
    sch.note_chimed(5)
    assert sch.attention(5) == (True, True)
    clk.t += 181
    assert sch.attention(5) == (False, False)
    sch.extend_attention(5)
    sch.end_attention(5)
    assert sch.attention(5) == (False, False)


# ---------- 插話的限制 ----------
@pytest.mark.parametrize("setup,reason", [
    ("mute", "muted"), ("busy", "busy"), ("cooldown", "cooldown"), ("llm_busy", "llm_busy"), ("off", "chime_off"),
])
def test_chime_gate_reasons(setup, reason):
    sch, _, _ = make(level=0 if setup == "off" else 1)
    if setup == "mute":
        sch.mute(5)
    elif setup == "busy":
        sch.throttle.try_start(5)
    elif setup == "cooldown":
        sch.throttle.try_start(5)
        sch.throttle.finish(5)
    assert sch.chime_gate(5, attention=False, llm_busy=setup == "llm_busy") == reason


def test_after_her_own_words_the_interval_limit_is_lifted_but_mute_is_not():
    sch, clk, _ = make()
    assert sch.plan_chime(REPLY, 5, "1", attention=False) == "reply"
    sch.throttle.finish(5)
    clk.t += 10
    assert sch.chime_gate(5, attention=False, llm_busy=False) == "channel_limit"
    assert sch.chime_gate(5, attention=True, llm_busy=False) is None
    sch.mute(5)
    assert sch.chime_gate(5, attention=True, llm_busy=False) == "muted"


def test_same_person_same_kind_only_once_a_day():
    sch, clk, _ = make(level=3)
    assert sch.plan_chime(REPLY, 5, "1", attention=False) == "reply"
    sch.throttle.finish(5)
    clk.t += 1000
    assert sch.plan_chime(REPLY, 5, "1", attention=False) == "skipped:kind_daily_limit"
    assert sch.plan_chime(REPLY, 5, "2", attention=False) == "reply"  # 別人照常


def test_chime_reply_takes_the_channel_and_is_refused_while_busy():
    sch, _, _ = make()
    sch.throttle.try_start(5)
    assert sch.plan_chime(REPLY, 5, "1", attention=False) == "skipped:busy"
    sch.throttle.finish(5)
    assert sch.plan_chime(REPLY, 5, "1", attention=False) == "reply" and sch.throttle.is_busy(5)


def test_reactions_have_their_own_limit():
    sch, _, _ = make(chime_react_max=2)
    for uid in ("1", "2"):
        assert sch.plan_chime(REACT, 5, uid, attention=False) == "react"
        sch.record_react(5, uid, "other")
    assert sch.plan_chime(REACT, 5, "3", attention=False) == "skipped:react_limit"


def test_failed_or_quiet_decisions():
    sch, _, _ = make()
    assert sch.plan_chime(Decision(error="no_json"), 5, "1", attention=False) == "error"
    assert sch.plan_chime(Decision(participate=0.69, action="reply"), 5, "1", attention=False) == "none"
    assert not sch.throttle.is_busy(5)


def test_debounce_only_judges_the_last_one_and_closing_cancels_waiting_work():
    sch, _, done = make(chime_debounce_sec=0.01)
    judged = []

    def judge(tag):
        async def go():
            judged.append(tag)
            return None
        return go

    async def scenario():
        sch.schedule_chime(5, judge("a"))
        sch.schedule_chime(5, judge("b"))
        await asyncio.sleep(0.05)
        sch.schedule_chime(5, judge("c"))
        await sch.close()
    asyncio.run(scenario())
    assert judged == ["b"]
