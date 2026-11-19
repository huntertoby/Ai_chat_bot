"""喚醒詞與結束字的比對、斜線指令的隱私、個資掃描。（結束字、注意模式的整條流程在 test_bot_flow.py）"""
import asyncio
import subprocess
from types import SimpleNamespace as NS

import pytest

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


@pytest.fixture
def cmd_kit(tmp_path):
    from botkit import TEST_GUILD, Kit, User
    from yuki.archive import ArchivedMessage
    kit = Kit(tmp_path, members=[User(1, "阿明"), User(2, "小傑")])
    kit.bot.archive.add_many([
        ArchivedMessage(i, str(TEST_GUILD), "c1", "閒聊", "", uid, name, f"2026-11-0{i} 12:00:00", text, None, 0)
        for i, (uid, name, text) in enumerate([("1", "阿明", "今天去打球"), ("2", "小傑", "我也想打球"),
                                               ("1", "阿明", "晚上吃拉麵")], start=1)])
    kit.bot.archive.commit()
    yield kit
    kit.close()


def call(kit, name, user_id="1", **kw):
    it = FakeInteraction(user_id=user_id, guild=kit.guild)
    run(kit.bot.tree.get_command(name).callback(it, **kw))
    return it


def test_memories_show_what_she_can_find_in_the_chat_log_only_to_the_caller(cmd_kit):
    it = call(cmd_kit, "記憶")
    text = "\n".join(c for c, _ in it.sent if c != "<defer>")
    assert "今天去打球" in text and "晚上吃拉麵" in text and "我也想打球" not in text  # 不填對象＝自己說的
    assert all(kw.get("ephemeral") is True for _, kw in it.sent)
    nobody = call(cmd_kit, "記憶", user_id="9")  # 聊天紀錄裡沒有的人：那句「還沒有」也只有自己看得到
    assert all(kw.get("ephemeral") is True for _, kw in nobody.sent)


def test_memories_keyword_search_uses_the_chat_log(cmd_kit):
    it = call(cmd_kit, "記憶", 關鍵字="打球")
    text = "\n".join(c for c, _ in it.sent if c != "<defer>")
    assert "今天去打球" in text and "我也想打球" in text and "拉麵" not in text
    assert all(kw.get("ephemeral") is True for _, kw in it.sent)


def test_retired_memory_commands_explain_instead_of_failing(cmd_kit):
    for it in (call(cmd_kit, "忘記", 記憶id="abcd1234"), call(cmd_kit, "白板")):
        (content, kw), = it.sent
        assert "停用" in content and kw.get("ephemeral") is True


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
