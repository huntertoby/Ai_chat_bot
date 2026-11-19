"""R1 第 1 段的行為修正（審查一-4、一-6、一-7）：結束字、喚醒詞、主伺服器注意模式、回覆計數、指令隱私、個資掃描。"""
import asyncio
import subprocess
import time
from types import SimpleNamespace as NS

import pytest

from test_chime import AMING, XIAOJIE, Clock, JsonLLM, fake_bot, run
from test_review_fixes import fake_incoming, fake_on_message, on_message
from yuki import bot as botmod, throttle
from yuki.bot import Turn, is_exit, wake_pattern


# ---------- 喚醒詞 ----------
def test_wake_word_needs_whole_word_for_english_names():
    w = wake_pattern(["yuki"])
    for text in ("Yuki 在嗎", "yuki的看法呢", "問一下YUKI", "@Yuki", "yuki?"):
        assert w.search(text), text
    for text in ("Yukihira 好帥", "miyuki 來了", "yuki2 是誰"):
        assert not w.search(text), text


def test_chinese_wake_word_still_matches_inside_a_sentence():
    assert wake_pattern(["小雪"]).search("欸小雪你覺得呢")


# ---------- 結束字 ----------
def test_exit_word_must_be_the_whole_sentence():
    words = ["掰掰", "再見", "去睡覺"]
    assert is_exit("掰掰", words) and is_exit("掰掰～！", words) and is_exit(" 去睡覺 。", words)
    assert not is_exit("我先去吃飯掰掰", words)
    assert not is_exit("再見面再說", words)
    assert not is_exit("", words)


def leaving_bot(**kw):
    me, ev = fake_on_message(**kw)
    replies = []

    async def reply(content=None, **k):
        replies.append(content)

    return me, ev, replies, reply


def test_saying_bye_to_her_ends_attention_and_is_kept_in_history():
    me, ev, replies, reply = leaving_bot(active=True, called=True)
    msg = fake_incoming("掰掰")
    msg.reply = reply
    on_message(me, msg)
    assert 5 not in me.active_until
    assert replies == [botmod.EXIT_REPLY]
    assert [l.text for l in ev.observed] == ["掰掰", botmod.EXIT_REPLY]  # 那句和她的道別都進對話紀錄
    assert ev.called == []  # 不另外再回一次


def test_exit_word_inside_a_longer_sentence_to_her_is_a_normal_message():
    me, ev, replies, reply = leaving_bot(active=True, called=True)
    msg = fake_incoming("我先去吃飯掰掰")
    msg.reply = reply
    on_message(me, msg)
    assert 5 in me.active_until and replies == []
    assert ev.called == [("我先去吃飯掰掰", True)]


def test_exit_word_said_to_someone_else_does_not_end_her_attention():
    me, ev, replies, reply = leaving_bot(active=True, called=False, chime_ok=False)
    msg = fake_incoming("掰掰")
    msg.reply = reply
    on_message(me, msg)
    assert 5 in me.active_until and replies == []


def test_bye_after_her_own_chime_also_ends_the_follow_up_window():
    me, ev, replies, reply = leaving_bot(chiming=True, called=True)
    msg = fake_incoming("再見")
    msg.reply = reply
    on_message(me, msg)
    assert 5 not in me.chime_until and replies == [botmod.EXIT_REPLY]


# ---------- 主伺服器（沒開插話）的注意模式 ----------
def test_main_server_attention_follow_ups_are_not_counted_as_calls():
    me, ev = fake_on_message(active=True, called=False, chime_ok=False)
    on_message(me, fake_incoming("那你覺得哪隊會贏"))
    assert ev.called == [("那你覺得哪隊會贏", False)] and ev.triggers == ["attention"]


def attention_bot():
    clk = Clock()
    ran, flushed = [], []
    me = NS(throttle=throttle.Throttle(clock=clk))

    async def run_turns(cid, turn):
        ran.append(turn)
        me.throttle.finish(cid)

    me._run_turns = run_turns
    me._schedule_flush = lambda cid: flushed.append(cid)
    return me, clk, ran, flushed


def test_busy_main_server_attention_switches_to_burst_mode_instead_of_replying_every_line():
    me, clk, ran, flushed = attention_bot()
    people = [AMING, XIAOJIE]
    for i in range(6):
        msg = NS(channel=NS(id=5))
        run(botmod.YukiBot._on_called(me, msg, people[i % 2], f"第 {i} 句", [], count_tag=False,
                                      trigger="attention"))
        clk.t += 5
    # 第 4 句讓頻道進入熱鬧模式；之後的不再一句一句回，收集起來等間隔到了合併回一次
    assert len(ran) == 4 and all(t.trigger == "attention" for t in ran)
    assert flushed and [t.text for t in me.throttle.drain(5)] == ["第 4 句", "第 5 句"]


def test_merged_turn_counts_as_called_only_if_someone_really_called_her():
    msg = NS(channel=NS(id=5))
    only_attention = [Turn(msg, AMING, "a", [], trigger="attention"), Turn(msg, XIAOJIE, "b", [], trigger="attention")]
    assert Turn.merge(only_attention).trigger == "attention"
    mixed = only_attention + [Turn(msg, AMING, "Yuki 你說呢", [], trigger="called")]
    assert Turn.merge(mixed).trigger == "called"


class Typing:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def respond_bot(reply_text="好喔"):
    me, _ = fake_bot(JsonLLM({}))
    me.chime_until, me.active_until, me._replying = {}, {}, 0
    me.voice = NS(describe=lambda g, n: "")
    me.members_of = lambda g: []
    me.owner_name = lambda g: ""
    me.archive_guild = lambda g: ""
    me.user = NS(id=99, name="Yuki", display_name="Yuki")
    me.emoji_pacing = NS(allow=lambda cid: True, record=lambda cid, used: None)

    async def reply_fn(*a, **k):
        return reply_text

    async def noop(*a, **k):
        return None

    me.brain.reply = reply_fn
    me.brain.observe = lambda k, l: True
    me.brain.buffer = lambda *a, **k: None
    msg = NS(channel=NS(id=5, typing=lambda: Typing(), send=noop), guild=NS(id=7), reply=noop,
             author=NS(id=1, name="阿明", display_name="阿明"), reference=None)
    return me, msg


def test_attention_reply_does_not_extend_attention_but_a_real_call_does():
    me, msg = respond_bot()
    run(botmod.YukiBot._respond(me, Turn(msg, AMING, "x", [], trigger="attention")))
    assert 5 not in me.active_until
    run(botmod.YukiBot._respond(me, Turn(msg, AMING, "x", [], trigger="called")))
    assert me.active_until[5] > time.monotonic()


def test_reply_counter_recovers_when_preparing_the_reply_fails():
    """回覆前的準備（追蹤紀錄、表情清單）丟例外：計數不能卡在 > 0，不然插話和人物檔案更新會永遠以為她在忙。"""
    me, msg = respond_bot()

    def broken(guild):
        raise RuntimeError("圖鑑壞了")

    me.emoji_set = broken
    with pytest.raises(RuntimeError):
        run(botmod.YukiBot._respond(me, Turn(msg, AMING, "x", [], trigger="called")))
    assert me._replying == 0


# ---------- 指令：/忘記 不重述、/記憶 只有本人看得到 ----------
class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, name, description):
        def deco(fn):
            self.commands[name] = fn
            return fn
        return deco

    def error(self, fn):
        return fn


class FakeInteraction:
    def __init__(self, user_id="1", guild=None):
        self.user = NS(id=int(user_id), display_name="阿明", guild_permissions=NS(manage_guild=False))
        self.guild = guild
        self.sent = []
        outer = self

        class Resp:
            async def send_message(self, content, **k):
                outer.sent.append((content, k))

            async def defer(self, **k):
                outer.sent.append(("<defer>", k))

            def is_done(self):
                return True

        class Follow:
            async def send(self, content, **k):
                outer.sent.append((content, k))

        self.response, self.followup = Resp(), Follow()


def commands_for(store):
    me = NS(tree=FakeTree(), brain=NS(store=store, agentic=False), settings=NS(wake_words=["yuki"]))
    botmod.register_commands(me)
    return me.tree.commands


def test_forget_does_not_repeat_the_forgotten_text(store):
    import numpy as np
    mem = store.add("dm-1", "阿明的祕密是怕黑", np.ones(4, dtype=np.float32), subject_id="1", subject_name="阿明")
    it = FakeInteraction()
    run(commands_for(store)["忘記"](it, mem.id))
    (content, kw), = it.sent
    assert "怕黑" not in content and kw.get("ephemeral") is True
    assert store.get(mem.id) is None


def test_memories_listing_is_only_visible_to_the_caller(store):
    import numpy as np
    store.add("dm-1", "阿明喜歡打球", np.ones(4, dtype=np.float32), subject_id="1", subject_name="阿明")
    it = FakeInteraction()
    run(commands_for(store)["記憶"](it))
    assert it.sent and all(kw.get("ephemeral") is True for _, kw in it.sent)
    it2 = FakeInteraction(user_id="2")  # 沒有記憶的人：那句「還沒記住」也只有自己看得到
    run(commands_for(store)["記憶"](it2))
    assert all(kw.get("ephemeral") is True for _, kw in it2.sent)


# ---------- 個資掃描：每個要 push 的 commit 都看 ----------
def git(repo, *args):
    subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "-c", "user.name=t", "-c", "user.email=t@t",
                    *args], cwd=repo, check=True, capture_output=True)


def test_private_word_added_then_removed_in_unpushed_commits_is_still_caught(tmp_path, capsys):
    from scripts import check_private
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "tag", "base")
    (tmp_path / "a.txt").write_text("小美的學校", encoding="utf-8")
    git(tmp_path, "commit", "-q", "-am", "oops")
    (tmp_path / "a.txt").write_text("hello again", encoding="utf-8")
    git(tmp_path, "commit", "-q", "-am", "remove it")
    words = {"小美的學校"}
    assert check_private.main([], root=tmp_path, words=words) == 0          # 只看 HEAD：乾淨（舊行為的漏洞）
    assert check_private.main(["base..HEAD"], root=tmp_path, words=words) == 1
    assert "小美的學校" not in capsys.readouterr().out                       # 不把整個字印出來


def test_private_word_in_a_commit_message_is_caught(tmp_path):
    from scripts import check_private
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "tag", "base")
    (tmp_path / "a.txt").write_text("hello 2", encoding="utf-8")
    git(tmp_path, "commit", "-q", "-am", "fix for 阿凱本名")
    assert check_private.main(["base..HEAD"], root=tmp_path, words={"阿凱本名"}) == 1
