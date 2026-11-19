"""整合測試：假的 Discord 訊息從 on_message 走完整個流程（排程器、Brain、追蹤紀錄都是真的，只換掉 Discord 和 LLM）。"""
import asyncio
import json

import pytest

from botkit import MAIN_GUILD, YUKI, Clock, FakeLLM, Kit, User
from yuki import bot as botmod
from yuki.llm import LLMError

AMING, XIAOJIE, XIAOMEI, AKAI = User(1, "阿明"), User(2, "小傑"), User(3, "小美"), User(4, "阿凱")
GOOD = {"reason": "好消息", "kind": "congrats", "action": "reply", "participate": 0.9, "emoji": "",
        "need_background": []}
# D-34：主伺服器注意模式內，模型判斷「在跟她講」
TALKING = {**GOOD, "kind": "to_me", "reason": "在跟我講"}


@pytest.fixture
def kits(tmp_path):
    made = []

    def make(**kw):
        k = Kit(tmp_path / f"k{len(made)}", **kw)
        made.append(k)
        return k

    yield make
    for k in made:
        k.close()


def last_user_text(messages) -> str:
    return messages[-1]["content"]


# ---------- 被叫到 ----------
def test_called_by_name_gets_a_reply_to_that_message(kits):
    kit = kits(llm=FakeLLM("在啊，怎麼了"))
    msg = kit.say(AMING, "yuki，在嗎")
    kit.run(msg)
    assert kit.replies == [(msg.id, "在啊，怎麼了")]
    assert "[阿明] 在嗎" in last_user_text(kit.llm.reply_calls[0])  # 喚醒詞拿掉、照對話紀錄格式給模型


def test_mentioning_her_counts_as_calling(kits):
    kit = kits()
    msg = kit.say(AMING, "@Yuki 你覺得呢", mentions=[YUKI])
    kit.run(msg)
    assert [m for m, _ in kit.replies] == [msg.id]


def test_only_her_name_with_nothing_else_gets_a_short_question_back(kits):
    kit = kits()
    kit.run(kit.say(AMING, "yuki"))
    assert [c for _, c in kit.replies] == ["找我有事嗎？"] and kit.llm.reply_calls == []


def test_same_person_calling_too_often_is_ignored_for_a_while(kits):
    kit = kits()
    kit.run(*[kit.say(AMING, f"yuki，第 {i} 次") for i in range(5)])
    assert len(kit.replies) == 3  # 60 秒內第 4 次起不回（D-24）
    kit.run(kit.say(XIAOJIE, "yuki，換我"))
    assert len(kit.replies) == 4  # 別人照常


def test_reply_failure_apologises_and_frees_the_channel(kits):
    kit = kits(llm=FakeLLM(reply_error=LLMError("斷線")))
    kit.run(kit.say(AMING, "yuki，在嗎"))
    assert "連線好像斷了" in kit.replies[0][1]
    assert not kit.bot.replies.throttle.is_busy(5) and kit.bot._replying == 0


def test_flood_message_is_not_answered_even_when_calling_her(kits):
    kit = kits()
    kit.run(kit.say(AMING, "a" * 20, mentions=[YUKI]))
    assert kit.replies == [] and list(kit.bot.brain.history["5"]) == []


def test_slash_or_prefix_commands_are_ignored(kits):
    kit = kits()
    kit.run(kit.say(AMING, "!yuki play"))
    assert kit.replies == []


# ---------- 熱鬧模式（很多人輪流叫她）----------
def test_busy_channel_collects_calls_and_answers_them_together_after_the_gap(kits):
    clk = Clock()
    kit = kits(clock=clk)
    people = [AMING, XIAOJIE, XIAOMEI, AKAI]
    for p in people:  # 60 秒內第 4 次被叫：進入熱鬧模式（第 4 次照回）
        kit.run(kit.say(p, f"yuki，我是{p.name}"))
        clk.t += 1
    assert len(kit.replies) == 4
    late1, late2 = kit.say(AMING, "yuki，還有我"), kit.say(XIAOJIE, "yuki，我也要")

    async def both():
        await kit.bot.on_message(late1)
        await kit.bot.on_message(late2)
    kit.run(both())  # 收集起來，等 90 秒間隔到了合成一次回
    assert len(kit.replies) == 5 and kit.replies[-1][0] == late2.id
    history = last_user_text(kit.llm.reply_calls[-1])
    assert "還有我" in history and "我也要" in history


def test_busy_channel_new_call_does_not_jump_ahead_of_collected_ones(kits):
    clk = Clock()
    kit = kits(clock=clk)
    for p in [AMING, XIAOJIE, XIAOMEI, AKAI]:
        kit.run(kit.say(p, f"yuki，我是{p.name}"))
        clk.t += 1

    held = kit.say(AMING, "yuki，先排隊的")
    newer = kit.say(XIAOJIE, "yuki，後來的")

    async def scenario():
        await kit.bot.on_message(held)   # 收集起來（合併回覆還在等間隔）
        clk.t += 200                     # 間隔已經過了，合併回覆還沒跑到之前，有人又叫她
        await kit.bot.on_message(newer)
    kit.run(scenario())
    # 新的這則跟排隊的那則合成一次回，不會先回新的、舊的再多等一輪
    assert kit.replies[4:] == [(newer.id, "好喔")]
    assert "先排隊的" in last_user_text(kit.llm.reply_calls[-1])


# ---------- 注意模式與結束字 ----------
def test_main_server_follow_ups_during_attention_are_answered_without_calling_her_again(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING))
    kit.run(kit.say(AMING, "yuki 今天打什麼"), kit.say(AMING, "那你覺得哪隊會贏"))
    assert len(kit.replies) == 2
    other = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING))
    other.run(other.say(AMING, "那你覺得哪隊會贏"))  # 沒叫過她：主伺服器不插話
    assert other.replies == []


def test_attention_follow_up_does_not_extend_attention_but_a_real_call_does(kits):
    clk = Clock()
    kit = kits(guild=MAIN_GUILD, clock=clk, llm=FakeLLM("好喔", TALKING))
    kit.run(kit.say(AMING, "yuki，在嗎"))
    clk.t += 170  # 注意時間 180 秒
    kit.run(kit.say(AMING, "那你覺得呢"))
    clk.t += 20   # 第一次被叫之後 190 秒：接話沒有延長，注意時間已經過了
    kit.run(kit.say(AMING, "還在嗎"))
    assert len(kit.replies) == 2


def test_saying_bye_to_her_ends_attention_and_both_lines_stay_in_history(kits):
    kit = kits(guild=MAIN_GUILD)
    kit.run(kit.say(AMING, "yuki，在嗎"), kit.say(AMING, "yuki 掰掰～"))
    assert kit.replies[-1][1] == botmod.EXIT_REPLY
    texts = [l.text for l in kit.bot.brain.history["5"]]
    assert texts[-2:] == ["掰掰～", botmod.EXIT_REPLY]
    kit.run(kit.say(AMING, "那你覺得呢"))  # 注意模式結束了：不再接話
    assert len(kit.replies) == 2


def test_exit_word_inside_a_longer_sentence_is_a_normal_message(kits):
    kit = kits(guild=MAIN_GUILD)
    kit.run(kit.say(AMING, "yuki，在嗎"), kit.say(AMING, "yuki 我先去吃飯掰掰"))
    assert kit.replies[-1][1] == "好喔"


def test_exit_word_said_to_someone_else_does_not_end_her_attention(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING))
    kit.run(kit.say(AMING, "yuki，在嗎"), kit.say(XIAOJIE, "掰掰"), kit.say(AMING, "你還在吧"))
    assert [c for _, c in kit.replies] == ["好喔", "好喔"]  # 跟別人說掰掰（雜訊）不回，也不結束


def test_tag_flooder_is_not_answered_through_attention_either(kits):
    kit = kits(guild=MAIN_GUILD)
    kit.run(*[kit.say(AMING, f"yuki，{i}") for i in range(4)])  # 第 4 次：開始冷卻
    kit.run(kit.say(AMING, "接下來呢"))
    assert len(kit.replies) == 3


# ---------- 插話（測試伺服器）----------
def test_chime_reply_after_debounce_and_follow_up_counts_as_attention(kits):
    clk = Clock()
    kit = kits(llm=FakeLLM("恭喜！", GOOD), clock=clk)
    msg = kit.say(AMING, "我考上研究所了")
    kit.run(msg)
    clk.t += 10  # 回完 5 秒內同頻道不插話
    assert kit.replies == [(msg.id, "恭喜！")]
    rec = [r for r in kit.records() if r["kind"] == "chime"][0]
    assert json.loads(rec["final"])["action"] == "reply"
    kit.run(kit.say(XIAOJIE, "對啊他超強的"))  # 她剛插過話：下一則判斷時提示她可能在接她的話
    assert "提示" in last_user_text(kit.llm.chime_calls[-1])


def test_chime_debounce_judges_only_the_latest_line(kits):
    kit = kits(llm=FakeLLM("恭喜！", {**GOOD, "participate": 0.1}))

    async def scenario():
        await kit.bot.on_message(kit.say(AMING, "今天去面試"))
        await kit.bot.on_message(kit.say(AMING, "結果錄取了"))
    kit.run(scenario())
    assert len(kit.llm.chime_calls) == 1 and "結果錄取了" in last_user_text(kit.llm.chime_calls[0])


def test_called_message_cancels_pending_chime_judgement(kits):
    kit = kits(llm=FakeLLM("好喔", GOOD))

    async def scenario():
        await kit.bot.on_message(kit.say(AMING, "今天天氣很好欸"))
        await kit.bot.on_message(kit.say(XIAOJIE, "yuki 你覺得呢"))
    kit.run(scenario())
    assert kit.llm.chime_calls == [] and len(kit.replies) == 1


def test_low_confidence_chime_stays_quiet(kits):
    kit = kits(llm=FakeLLM("恭喜", {**GOOD, "participate": 0.69}))
    kit.run(kit.say(AMING, "我考上研究所了"))
    assert kit.replies == []
    lvl2 = kits(llm=FakeLLM("恭喜", {**GOOD, "participate": 0.69}), chime_in=2)  # 等級 2 門檻 0.6
    lvl2.run(lvl2.say(AMING, "我考上研究所了"))
    assert len(lvl2.replies) == 1


def test_chime_at_most_once_per_interval_per_channel(kits):
    clk = Clock()
    kit = kits(llm=FakeLLM("恭喜", GOOD), clock=clk)
    kit.run(kit.say(AMING, "我考上研究所了"))
    clk.t += 200  # 過了插話後的注意時間（3 分鐘），還在 10 分鐘一次的限制內
    kit.run(kit.say(XIAOJIE, "我也找到工作了"))
    assert len(kit.replies) == 1
    rec = [r for r in kit.records() if r["kind"] == "chime"][-1]
    assert json.loads(rec["final"])["action"] == "skipped:channel_limit" and rec["outcome"] == "skipped"


def test_shush_after_her_chime_mutes_further_chimes(kits):
    kit = kits(llm=FakeLLM("恭喜", GOOD), chime_in=3)
    kit.run(kit.say(AMING, "我考上研究所了"), kit.say(XIAOJIE, "沒問你"))
    assert kit.bot.replies.limits.is_muted(5)
    quiet = kits(llm=FakeLLM("恭喜", GOOD))
    quiet.run(quiet.say(XIAOJIE, "沒問你"))  # 她沒講過話、沒人叫她：不算對她說
    assert not quiet.bot.replies.limits.is_muted(5)


def test_chime_react_presses_an_emoji_without_replying(kits):
    kit = kits(llm=FakeLLM("x", {**GOOD, "action": "react", "emoji": "🎉", "participate": 0.3, "kind": "other"}))
    msg = kit.say(AMING, "今天運氣好好")
    kit.run(msg)
    assert msg.reactions == ["🎉"] and kit.replies == []
    bad = kits(llm=FakeLLM("x", {**GOOD, "action": "react", "emoji": ":nope:", "kind": "comfort"}))
    m2 = bad.say(AMING, "今天運氣好好")
    bad.run(m2)
    assert m2.reactions == []
    rec = [r for r in bad.records() if r["kind"] == "chime"][-1]
    assert json.loads(rec["final"])["action"] == "skipped:emoji_not_found"


def test_chime_judgement_error_stays_quiet_and_is_marked_degraded(kits):
    kit = kits(llm=FakeLLM("x", decide_error=LLMError("斷線")))
    kit.run(kit.say(AMING, "我考上研究所了"))
    rec = [r for r in kit.records() if r["kind"] == "chime"][0]
    assert kit.replies == [] and "chime_failed" in rec["degraded"]
    assert not kit.bot.replies.throttle.is_busy(5)


def test_chime_off_never_asks_the_model(kits):
    kit = kits(llm=FakeLLM("x", GOOD), chime_in=0)
    kit.run(kit.say(AMING, "我考上研究所了"))
    assert kit.llm.chime_calls == [] and kit.replies == []


def test_chime_skipped_while_her_reply_is_still_being_prepared(kits):
    kit = kits(llm=FakeLLM("x", GOOD))
    kit.bot._replying = 1  # 別的頻道正在回覆（LM Studio 一次只處理一個）
    kit.run(kit.say(AMING, "我考上研究所了"))
    rec = [r for r in kit.records() if r["kind"] == "chime"][0]
    assert kit.llm.chime_calls == [] and json.loads(rec["final"])["action"] == "skipped:llm_busy"


# ---------- /不要記我 的人 ----------
def test_opted_out_person_is_answered_but_never_traced_or_searched_on_the_web(kits):
    kit = kits(llm=FakeLLM("不知道欸", tool=("web_search", {"query": "阿明 2026 比賽"})), web_enabled=True,
               members=[AMING, XIAOJIE], optout_enabled=True)
    kit.bot.archive.opt_out(str(MAIN_GUILD), "1")  # 在別的伺服器打的也算
    kit.run(kit.say(AMING, "yuki，幫我查我自己"))
    assert len(kit.replies) == 1
    assert kit.llm.tool_results and "不要記我" in kit.llm.tool_results[0]  # 名字沒送出去
    assert [r for r in kit.records() if r["kind"] == "reply"] == []  # 他觸發的回覆整筆不寫追蹤紀錄


def test_opted_out_person_is_scrubbed_from_someone_elses_chime_trace(kits):
    kit = kits(llm=FakeLLM("恭喜", {**GOOD, "reason": "阿明也在"}), optout_enabled=True)
    kit.bot.archive.opt_out(str(MAIN_GUILD), "1")
    kit.run(kit.say(AMING, "我偷偷報名了比賽"), kit.say(XIAOJIE, "我考上研究所了"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in kit.trace_dir.glob("ai-*.jsonl"))
    assert text and "阿明" not in text and "偷偷報名" not in text


# ---------- 其他邊界 ----------
def test_flood_is_still_archived_on_the_main_server(kits):
    kit = kits(guild=MAIN_GUILD)
    archived = []
    kit.bot.sync.record = archived.append
    kit.run(kit.say(AMING, "a" * 20, mentions=[YUKI]), kit.say(AMING, "我考上了！！！！！！！！！！", mentions=[YUKI]))
    assert len(archived) == 2 and len(kit.replies) == 1  # 洗版的不回，但封存照常；驚嘆號很多不算洗版


def test_bye_after_her_own_chime_ends_the_follow_up_window(kits):
    kit = kits(llm=FakeLLM("恭喜", GOOD))
    kit.run(kit.say(AMING, "我考上研究所了"), kit.say(AMING, "yuki 掰掰"))
    assert kit.replies[-1][1] == botmod.EXIT_REPLY and kit.bot.replies.attention(5) == (False, False)


def test_chime_rechecks_that_lm_studio_is_free_after_slow_preparation(kits):
    kit = kits(llm=FakeLLM("x", GOOD))

    async def slow_fill(channel):
        kit.bot._replying = 1  # 補抓回覆對象的時候，別的回覆開始了

    kit.bot._fill_reply_refs = slow_fill
    kit.run(kit.say(AMING, "我考上研究所了"))
    rec = [r for r in kit.records() if r["kind"] == "chime"][0]
    assert kit.llm.chime_calls == [] and json.loads(rec["final"])["action"] == "skipped:llm_busy"


def test_tag_flooder_is_not_judged_for_chiming_but_others_are(kits):
    kit = kits(llm=FakeLLM("x", {**GOOD, "participate": 0.1}))
    kit.run(*[kit.say(AMING, f"yuki，{i}") for i in range(4)])
    kit.bot.replies.throttle._last_done.clear()
    kit.run(kit.say(AMING, "今天考試考得不錯"))
    assert kit.llm.chime_calls == []
    kit.run(kit.say(XIAOJIE, "今天考試考得不錯"))
    assert len(kit.llm.chime_calls) == 1


def test_reply_counter_recovers_when_preparing_the_reply_fails(kits):
    """回覆前的準備（追蹤紀錄、表情清單）丟例外：計數不能卡在 > 0，不然插話和人物檔案更新會永遠以為她在忙。"""
    kit = kits()

    def broken(guild):
        raise RuntimeError("圖鑑壞了")

    kit.bot.emoji_set = broken
    kit.run(kit.say(AMING, "yuki，在嗎"))
    assert kit.bot._replying == 0 and not kit.bot.replies.throttle.is_busy(5)


def test_main_server_attention_waits_for_link_previews(kits, monkeypatch):
    from yuki import vision
    waited = []

    async def fake_wait(message):
        waited.append(message.content)

    monkeypatch.setattr(vision, "wait_for_embeds", fake_wait)
    monkeypatch.setattr(vision, "image_urls", lambda m: [])
    quiet = kits(guild=MAIN_GUILD, vision_enabled=True)
    quiet.run(quiet.say(XIAOJIE, "沒人叫她"))  # 沒叫她、也不在注意模式：不等
    kit = kits(guild=MAIN_GUILD, vision_enabled=True, llm=FakeLLM("好喔", TALKING))
    kit.run(kit.say(AMING, "yuki 看這個"), kit.say(AMING, "接下來呢"))
    assert waited == ["yuki 看這個", "接下來呢"] and len(kit.replies) == 2


# ---------- D-28：/不要記我 暫時停用（OPTOUT_ENABLED，預設關）----------
def test_optout_off_by_default_no_command_and_nobody_is_blocked(kits):
    kit = kits(llm=FakeLLM("查到了", tool=("web_search", {"query": "阿明 2026 比賽"})), web_enabled=True,
               members=[AMING])
    assert kit.settings.optout_enabled is False
    assert kit.bot.tree.get_command("不要記我") is None
    kit.bot.archive.opt_out(str(MAIN_GUILD), "1")  # 以前留下的退出紀錄：資料留著，但不再生效

    async def fake_web(name, args):
        return "網路結果"
    kit.bot.brain.web.call = fake_web
    kit.run(kit.say(AMING, "yuki，幫我查我自己"))
    assert kit.llm.tool_results == ["網路結果"]                       # 名字照常送出去查
    assert [r for r in kit.records() if r["kind"] == "reply"]        # 追蹤紀錄照常寫
    assert kit.bot.archive.opted_out() == {"1"}                      # 退出紀錄沒被刪
    assert kit.bot.sync.skip(str(MAIN_GUILD)) == set()               # 封存照常收他的訊息


def test_optout_enabled_registers_the_command(kits):
    kit = kits(optout_enabled=True)
    assert kit.bot.tree.get_command("不要記我") is not None


# ---------- 哪些伺服器開什麼 ----------
def test_chime_runs_only_in_test_servers_unless_listed(kits):
    main = kits(llm=FakeLLM("恭喜", GOOD), guild=MAIN_GUILD)
    main.run(main.say(AMING, "我考上研究所了"))
    assert main.llm.chime_calls == []                       # 預設＝只有測試伺服器
    listed = kits(llm=FakeLLM("恭喜", GOOD), guild=MAIN_GUILD, chime_guild_ids=[str(MAIN_GUILD)])
    listed.run(listed.say(AMING, "我考上研究所了"))
    assert len(listed.llm.chime_calls) == 1
    test = kits(llm=FakeLLM("恭喜", GOOD), chime_guild_ids=[str(MAIN_GUILD)])
    test.run(test.say(AMING, "我考上研究所了"))
    assert test.llm.chime_calls == []                       # 有填就照填的，測試伺服器也不例外


def test_each_server_uses_only_its_own_emoji_catalog(kits):
    from botkit import TEST_GUILD
    from yuki import emoji_catalog as cat
    item = [{"id": "11", "name": "happy", "kind": "emoji", "uses": 50, "meaning": "開心", "use": True}]

    def system_for(guild, sets, **kw):
        kit = kits(guild=guild, emoji_sets=sets, **kw)
        kit.run(kit.say(AMING, "yuki，在嗎"))
        return kit.llm.reply_calls[0][0]["content"]

    both = {str(TEST_GUILD): cat.EmojiSet(str(TEST_GUILD), item), str(MAIN_GUILD): cat.EmojiSet(str(MAIN_GUILD), item)}
    assert ":happy:" in system_for(TEST_GUILD, both)        # EMOJI_GUILD_IDS 留空＝只有測試伺服器
    assert ":happy:" not in system_for(MAIN_GUILD, both)
    assert ":happy:" in system_for(MAIN_GUILD, both, emoji_guild_ids=[str(MAIN_GUILD)])
    only_main = {str(MAIN_GUILD): cat.EmojiSet(str(MAIN_GUILD), item)}
    assert ":happy:" not in system_for(TEST_GUILD, only_main)  # 測試伺服器不借主伺服器的圖鑑


def test_mentioning_her_bot_role_counts_as_calling_but_other_roles_do_not(kits):
    from types import SimpleNamespace as NS
    bot_role, other = NS(is_bot_managed=lambda: True), NS(is_bot_managed=lambda: False)

    def mention(role):
        kit = kits(guild=MAIN_GUILD)
        kit.guild.me.roles = [bot_role, other]
        msg = kit.say(AMING, "<@&1> 看這個")
        msg.role_mentions = [role]
        kit.run(msg)
        return kit.replies

    assert len(mention(bot_role)) == 1   # 在輸入框選到 bot 的身分組「@Yuki」也算叫她
    assert mention(other) == []


# ---------- D-32：提到名字不一定是叫她 ----------
TO_ME = {**GOOD, "kind": "about_me", "reason": "在講我"}


def test_name_inside_a_sentence_goes_to_chime_judgement_in_a_chime_server(kits):
    kit = kits(llm=FakeLLM("謝啦", TO_ME))
    msg = kit.say(AMING, "我覺得 yuki 回得蠻好的")
    kit.run(msg)
    assert len(kit.llm.chime_calls) == 1                      # 走插話判斷，不是直接回
    assert "yuki 回得蠻好" in last_user_text(kit.llm.chime_calls[0])  # 送進判斷的那則保留名字
    assert [l.text for l in kit.bot.brain.history["5"]][0] == "我覺得 yuki 回得蠻好的"  # 對話紀錄只記一份
    assert kit.replies == [(msg.id, "謝啦")]                   # 判斷要回才回
    rec = [r for r in kit.records() if r["kind"] == "chime"][0]
    assert rec["trigger"] == "named" and json.loads(rec["final"])["named"] is True


def test_sentence_start_comment_is_a_mention_not_a_call(kits):
    kit = kits(llm=FakeLLM("哼", {**TO_ME, "action": "none", "participate": 0.1}))
    kit.run(kit.say(AMING, "yuki 根本是複讀機"))
    assert len(kit.llm.chime_calls) == 1 and kit.replies == [] and kit.llm.reply_calls == []


def test_main_server_still_treats_any_mention_of_her_name_as_a_call(kits):
    kit = kits(llm=FakeLLM("在啦", TO_ME), guild=MAIN_GUILD)
    msg = kit.say(AMING, "我覺得 yuki 回得蠻好的")
    kit.run(msg)
    assert kit.llm.chime_calls == [] and kit.replies == [(msg.id, "在啦")]
    assert "[阿明] 我覺得 回得蠻好的" in last_user_text(kit.llm.reply_calls[0])  # 照舊拿掉名字


def test_direct_call_by_name_skips_chime_judgement(kits):
    kit = kits(llm=FakeLLM("怎麼了", TO_ME))
    msg = kit.say(AMING, "yuki 你看這個")
    kit.run(msg)
    assert kit.llm.chime_calls == [] and kit.replies == [(msg.id, "怎麼了")]
    assert "[阿明] 你看這個" in last_user_text(kit.llm.reply_calls[0])


def test_telling_her_to_shut_up_by_name_still_mutes_the_channel(kits):
    kit = kits(llm=FakeLLM("好啦", TO_ME))
    kit.run(kit.say(AMING, "Yuki 閉嘴"))
    assert kit.bot.replies.limits.is_muted(5)
    other = kits(llm=FakeLLM("好啦", TO_ME))
    other.run(other.say(AMING, "可以叫 yuki 閉嘴嗎"))  # 只是提到她：照舊算跟她講
    assert other.bot.replies.limits.is_muted(5)


def test_named_chime_is_not_blocked_by_channel_limit_but_is_by_mute(kits):
    clk = Clock()
    kit = kits(llm=FakeLLM("恭喜", GOOD), clock=clk)
    kit.run(kit.say(AMING, "我考上研究所了"))
    clk.t += 200  # 過了插話後的注意時間，還在 10 分鐘一次的限制內
    kit.run(kit.say(XIAOJIE, "我也找到工作了"))
    assert len(kit.replies) == 1                              # 一般插話被頻道名額擋
    kit.run(kit.say(XIAOJIE, "我覺得 yuki 剛剛講得很對"))
    assert len(kit.replies) == 2                              # 提到她的不吃頻道名額
    kit.bot.replies.mute(5)
    clk.t += 10
    kit.run(kit.say(XIAOMEI, "大家覺得 yuki 怎樣"))
    assert len(kit.replies) == 2
    rec = [r for r in kit.records() if r["kind"] == "chime"][-1]
    assert json.loads(rec["final"])["action"] == "skipped:muted" and rec["trigger"] == "named"


def test_chime_off_server_level_zero_keeps_old_name_behaviour(kits):
    kit = kits(llm=FakeLLM("在", TO_ME), chime_in=0)  # 測試伺服器但插話關掉：沒判斷可走，有名字就回
    kit.run(kit.say(AMING, "我覺得 yuki 回得蠻好的"))
    assert kit.llm.chime_calls == [] and len(kit.replies) == 1


# ---------- D-34：程式端先擋（@別人、叫別的 bot）----------
NOVA = User(77, "Nova", bot=True)


def _reply_to(kit, msg, target):
    from types import SimpleNamespace as NS
    msg.reference = NS(resolved=target, message_id=target.id)
    return msg


def test_chime_server_tagging_someone_else_is_blocked_without_asking_the_model(kits):
    kit = kits(llm=FakeLLM("恭喜", GOOD), members=[NOVA])
    kit.run(kit.say(AMING, f"<@{XIAOJIE.id}> 我考上研究所了", mentions=[XIAOJIE]))
    assert kit.llm.chime_calls == [] and kit.replies == []
    rec = [r for r in kit.records() if r["kind"] == "chime"][0]
    assert json.loads(rec["final"])["action"] == "skipped:at_other" and rec["outcome"] == "skipped"
    kit.run(kit.say(AMING, "Nova 幫我查明天天氣"))  # 句首叫別的 bot
    assert kit.llm.chime_calls == []
    kit.run(kit.say(AMING, "我覺得 Nova 比較會查天氣"))  # 句中提到：交給模型
    assert len(kit.llm.chime_calls) == 1


def test_chime_server_reply_to_someone_else_still_goes_to_the_model(kits):
    kit = kits(llm=FakeLLM("恭喜", {**GOOD, "participate": 0.1}))  # 不插話：不佔頻道名額
    first = kit.say(XIAOJIE, "你今天幹嘛")
    kit.run(first)
    n = len(kit.llm.chime_calls)
    msg = _reply_to(kit, kit.say(AMING, "我考上研究所了", mentions=[XIAOJIE]), first)  # 回覆時自動 ping
    kit.run(msg)
    assert len(kit.llm.chime_calls) == n + 1


# ---------- D-34：主伺服器注意模式先判斷是不是在跟她講 ----------
def _chime_recs(kit):
    return [r for r in kit.records() if r["kind"] == "chime"]


def test_main_server_attention_replies_only_when_judged_talking_to_her(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING))
    follow = kit.say(AMING, "那你覺得哪隊會贏")
    kit.run(kit.say(AMING, "yuki 今天打什麼"), follow)
    assert len(kit.llm.chime_calls) == 1 and kit.replies[-1] == (follow.id, "好喔")
    rec = _chime_recs(kit)[0]
    final = json.loads(rec["final"])
    assert rec["trigger"] == "attention_gate" and final["action"] == "reply" and final["replied"] is True
    assert "提示" in last_user_text(kit.llm.chime_calls[0])  # attention=True
    reply_recs = [r for r in kit.records() if r["kind"] == "reply"]
    assert reply_recs[-1]["trigger"] == "attention"


@pytest.mark.parametrize("decision", [
    {**TALKING, "action": "none", "participate": 0.1},
    {**TALKING, "action": "reply", "participate": 0.3},                 # 低於門檻
    {**TALKING, "action": "react", "emoji": "😂", "participate": 0.9},  # 主伺服器不按表情
])
def test_main_server_attention_stays_quiet_when_judged_not_for_her(kits, decision):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", decision))
    msg = kit.say(XIAOJIE, "晚上要不要去吃火鍋")
    kit.run(kit.say(AMING, "yuki 今天打什麼"), msg)
    assert len(kit.replies) == 1 and len(kit.llm.chime_calls) == 1 and msg.reactions == []
    final = json.loads(_chime_recs(kit)[0]["final"])
    assert final["action"] == "none" and final["replied"] is False


def test_main_server_attention_tagging_someone_else_is_not_judged(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING), members=[NOVA])
    kit.run(kit.say(AMING, "yuki 今天打什麼"),
            kit.say(XIAOJIE, f"<@{AKAI.id}> 你上線了沒", mentions=[AKAI]),
            kit.say(XIAOJIE, f"<@{NOVA.id}> 幫我查天氣", mentions=[NOVA]),
            kit.say(XIAOJIE, "nova 幫我查天氣"))
    assert kit.llm.chime_calls == [] and len(kit.replies) == 1
    finals = [json.loads(r["final"]) for r in _chime_recs(kit)]
    assert [f["action"] for f in finals] == ["skipped:at_other", "skipped:at_other", "skipped:other_bot"]
    assert all(r["trigger"] == "attention_gate" for r in _chime_recs(kit))


def test_main_server_attention_tag_plus_her_name_is_a_call(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING))
    kit.run(kit.say(AMING, "yuki 今天打什麼"), kit.say(XIAOJIE, f"<@{AKAI.id}> 你問 yuki 啦", mentions=[AKAI]))
    assert len(kit.replies) == 2 and kit.llm.chime_calls == []  # 主伺服器有名字就算叫她（D-32 照舊）


def test_main_server_outside_attention_is_not_judged(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING))
    kit.run(kit.say(AMING, "那你覺得哪隊會贏"))
    assert kit.llm.chime_calls == [] and kit.replies == [] and _chime_recs(kit) == []


def test_main_server_direct_call_during_attention_skips_the_judgement(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", {**TALKING, "action": "none", "participate": 0.0}))
    kit.run(kit.say(AMING, "yuki 今天打什麼"), kit.say(XIAOJIE, "yuki，你也來"))
    assert len(kit.replies) == 2 and kit.llm.chime_calls == []


def test_main_server_attention_is_not_limited_by_chime_quotas(kits):
    clk = Clock()
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("恭喜", GOOD), clock=clk)  # congrats：插話時同人同類一天一次
    kit.run(kit.say(AMING, "yuki，在嗎"))
    for t in ("我考上研究所了", "我也找到工作了", "我還中了發票"):
        clk.t += 10
        kit.run(kit.say(AMING, t))
    assert len(kit.replies) == 4


def test_main_server_attention_waits_while_lm_studio_is_busy(kits):
    clk = Clock()
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING), clock=clk)
    kit.run(kit.say(AMING, "yuki，在嗎"))
    sleep = kit.bot.replies._sleep
    naps = []

    async def busy_for_a_while(sec):
        naps.append(sec)
        if len(naps) >= 6:
            kit.bot._replying = 0  # 上一則回完了
        await sleep(sec)

    kit.bot.replies._sleep = busy_for_a_while
    kit.bot._replying = 1  # 她正在回別的（LM Studio 忙）
    follow = kit.say(AMING, "那你覺得呢")
    kit.run(follow)
    assert len(kit.llm.chime_calls) == 1 and kit.replies[-1] == (follow.id, "好喔")
    assert json.loads(_chime_recs(kit)[-1]["final"])["waited_llm"] is True


def test_main_server_attention_gives_up_after_waiting_too_long(kits):
    clk = Clock()
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING), clock=clk)
    kit.run(kit.say(AMING, "yuki，在嗎"))
    kit.bot._replying = 1  # 一直忙
    t0 = clk.t
    kit.run(kit.say(AMING, "那你覺得呢"))
    final = json.loads(_chime_recs(kit)[-1]["final"])
    assert final["action"] == "skipped:llm_busy_timeout" and kit.llm.chime_calls == []
    assert clk.t - t0 >= botmod.ATTENTION_GATE_WAIT_SEC and len(kit.replies) == 1
    kit.bot._replying = 0


def test_main_server_attention_judges_only_the_newest_line_after_waiting(kits):
    clk = Clock()
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING), clock=clk)
    kit.run(kit.say(AMING, "yuki，在嗎"))
    kit.bot._replying = 1
    older, newer = kit.say(AMING, "那你覺得呢"), kit.say(XIAOJIE, "我也想知道")

    async def scenario():
        await kit.bot.on_message(older)
        await asyncio.sleep(0)
        await asyncio.sleep(0)            # older 開始判斷、在等 LLM
        await kit.bot.on_message(newer)   # 等的時候又有人講
        kit.bot._replying = 0
    kit.run(scenario())
    assert len(kit.llm.chime_calls) == 1 and "我也想知道" in last_user_text(kit.llm.chime_calls[0])
    assert kit.replies[-1][0] == newer.id
    actions = [json.loads(r["final"])["action"] for r in _chime_recs(kit)]
    assert "skipped:superseded" in actions


def test_main_server_attention_respects_mute(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING))
    kit.run(kit.say(AMING, "yuki，在嗎"))
    kit.bot.replies.mute(5)
    kit.run(kit.say(AMING, "那你覺得呢"))
    assert len(kit.replies) == 1 and json.loads(_chime_recs(kit)[-1]["final"])["action"] == "skipped:muted"


# ---------- D-34：「沒叫你」結束注意模式 ----------
def test_saying_i_did_not_call_you_ends_attention_without_a_reply(kits):
    kit = kits(guild=MAIN_GUILD, llm=FakeLLM("好喔", TALKING))
    kit.run(kit.say(AMING, "yuki 今天打什麼"), kit.say(XIAOJIE, "我剛剛沒叫你啦"))
    assert len(kit.replies) == 1 and kit.llm.chime_calls == []
    assert kit.bot.replies.attention(5) == (False, False) and kit.bot.replies.limits.is_muted(5)
    kit.run(kit.say(AMING, "那你覺得呢"))  # 注意模式結束了：不再判斷、不回
    assert len(kit.replies) == 1 and kit.llm.chime_calls == []


def test_saying_i_did_not_call_you_after_her_chime_ends_the_follow_up_window(kits):
    kit = kits(llm=FakeLLM("恭喜", GOOD))
    kit.run(kit.say(AMING, "我考上研究所了"))
    n = len(kit.llm.chime_calls)
    kit.run(kit.say(XIAOJIE, "不是在跟你講啦"))
    assert kit.bot.replies.attention(5) == (False, False) and len(kit.replies) == 1
    assert len(kit.llm.chime_calls) == n
