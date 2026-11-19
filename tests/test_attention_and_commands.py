"""喚醒詞與結束字的比對、斜線指令的隱私、個資掃描。（結束字、注意模式的整條流程在 test_bot_flow.py）"""
import asyncio
import subprocess
from types import SimpleNamespace as NS

from yuki import bot as botmod
from yuki.bot import is_exit, wake_pattern


def run(coro):
    return asyncio.run(coro)


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
