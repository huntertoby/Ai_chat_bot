"""整合測試用的假 Discord：組一個真的 YukiBot（真的排程器、Brain、封存區、追蹤紀錄），
只把 Discord 物件和 LLM 換成假的，從 on_message 一路走到「送出訊息」。不連網路、不載模型。

    kit = Kit(tmp_path)                     # 預設在測試伺服器（有開插話判斷）
    msg = kit.say(AMING, "yuki 在嗎")
    kit.run(kit.bot.on_message(msg))        # 跑完會等排程器的背景工作（去抖動、熱鬧模式的合併回覆）
    kit.replies                             # [(回覆哪則訊息的 id, 內容)]
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import replace
from types import SimpleNamespace as NS

import discord

from yuki import trace
from yuki.bot import YukiBot
from yuki.config import Settings
from yuki.llm import LLMError
from yuki.scheduler import ReplyScheduler

TEST_GUILD, MAIN_GUILD, CHANNEL = 111, 222, 5
_ids = itertools.count(1000)


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class User:
    def __init__(self, uid: int, name: str, *, bot: bool = False):
        self.id, self.name, self.display_name, self.global_name, self.bot = uid, name, name, None, bot


YUKI = User(99, "Yuki", bot=True)


class FakeLLM:
    """插話判斷（chat_json）回 decision；回覆（chat_tools / chat）回 reply。
    tool=(名稱, 參數)：回覆時先呼叫一次這個工具（看程式端有沒有擋）。"""

    def __init__(self, reply: str = "好喔", decision: dict | None = None, *, tool=None,
                 reply_error: Exception | None = None, decide_error: Exception | None = None,
                 pick: dict | None = None, pick_error: Exception | None = None, pick_delay: float = 0.0):
        self.reply, self.decision, self.tool = reply, decision, tool
        self.reply_error, self.decide_error = reply_error, decide_error
        # D-37 挑表情（chat_json name="emoji_pick"）：回 pick（預設不配），呼叫記在 pick_calls，不算插話判斷
        self.pick, self.pick_error, self.pick_delay = pick or {"emoji": ""}, pick_error, pick_delay
        self.pick_calls: list[list[dict]] = []
        self.chime_calls: list[list[dict]] = []
        self.reply_calls: list[list[dict]] = []
        self.tool_results: list[str] = []

    async def chat_json(self, messages, schema, **kw):
        if kw.get("name") == "emoji_pick":
            self.pick_calls.append(messages)
            if self.pick_delay:
                await asyncio.sleep(self.pick_delay)
            if self.pick_error:
                raise self.pick_error
            return self.pick
        self.chime_calls.append(messages)
        if self.decide_error:
            raise self.decide_error
        return self.decision

    async def chat_tools(self, messages, specs, call_tool, max_rounds=3):
        self.reply_calls.append(messages)
        if self.tool:
            self.tool_results.append(await call_tool(*self.tool))
        if self.reply_error:
            raise self.reply_error
        return self.reply

    async def chat(self, messages, **kw):
        return await self.chat_tools(messages, [], None)

    async def resolve_model(self):
        return "fake"

    async def aclose(self):
        return None


class Typing:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class Channel:
    def __init__(self, kit: "Kit", cid: int = CHANNEL):
        self.id, self.name, self.kit = cid, "閒聊", kit

    def typing(self):
        return Typing()

    async def send(self, content=None, **kw):
        self.kit.sent.append((None, content, kw))

    async def fetch_message(self, message_id):
        raise discord.NotFound(NS(status=404, reason="x"), "gone")


class Message:
    def __init__(self, kit: "Kit", author: User, content: str, guild, channel: Channel, mentions=()):
        self.id = next(_ids)
        self.kit, self.author, self.content, self.clean_content = kit, author, content, content
        self.guild, self.channel, self.mentions = guild, channel, list(mentions)
        self.webhook_id, self.type = None, discord.MessageType.default
        self.role_mentions, self.attachments, self.embeds, self.reference = [], [], [], None
        self.reactions: list = []

    async def reply(self, content=None, **kw):
        self.kit.sent.append((self.id, content, kw))

    async def add_reaction(self, emoji):
        self.reactions.append(emoji)


def make_guild(gid: int, members: list[User]):
    me = NS(display_name="Yuki", roles=[])
    return NS(id=gid, name=f"g{gid}", me=me, members=list(members), emojis=[], stickers=[], default_role=None,
              roles=[], get_member=lambda uid: next((m for m in members if m.id == uid), None))


class Kit:
    """guild：TEST_GUILD（開插話）或 MAIN_GUILD（沒開插話，只有被叫和注意模式）。
    clock：換成假的時鐘（熱鬧模式要等 90 秒那種），等待也用假的（直接把時鐘往前撥）。"""

    def __init__(self, tmp_path, *, llm: FakeLLM | None = None, guild: int = TEST_GUILD, clock: Clock | None = None,
                 members: list[User] = (), emoji_sets: dict | None = None, **settings):
        tmp_path.mkdir(parents=True, exist_ok=True)
        s = replace(Settings(), **{
            "vision_enabled": False, "web_enabled": False, "voice_enabled": False, "catchup_enabled": False,
            "profile_auto_update": False, "profile_dir": tmp_path / "profiles",
            "profile_db_path": tmp_path / "profiles.db", "trace_dir": tmp_path / "traces",
            "db_path": tmp_path / "yuki.db", "test_guild_ids": [str(TEST_GUILD)], "chime_guild_ids": [],
            "emoji_guild_ids": [], "owner_user_ids": [], "profile_fallback_guild": "", "wake_words": ["yuki"],
            "chime_debounce_sec": 0.0, "chime_in": 1, **settings})
        self.settings = s
        root = logging.getLogger()
        before = list(root.handlers)
        self.bot = YukiBot(s, archive_path=tmp_path / "archive.db", emoji_sets=emoji_sets or {})
        for h in list(root.handlers):  # YukiBot 會在 root logger 掛通知用的 handler，測完不留
            if h not in before:
                root.removeHandler(h)
        self.llm = llm or FakeLLM()
        self.bot.brain.llm = self.llm
        self.bot._connection.user = YUKI
        self.clock = clock
        if clock is not None:
            async def fake_sleep(sec: float) -> None:
                clock.t += sec
                await asyncio.sleep(0)
            self.bot.replies = ReplyScheduler(s, self.bot._respond, clock=clock, sleep=fake_sleep)
        self.sent: list[tuple] = []
        self.channel = Channel(self)
        self.guild = make_guild(guild, list(members))
        self.trace_dir = tmp_path / "traces"

    # ---------- 發訊息 ----------
    def say(self, author: User, content: str, *, mentions=()) -> Message:
        return Message(self, author, content, self.guild, self.channel, mentions)

    def run(self, *coros_or_msgs):
        """依序處理（訊息就丟給 on_message），每一則之後都等排程器的背景工作（去抖動後的插話判斷、
        熱鬧模式的合併回覆）做完。要讓幾則訊息「同時」進來，就自己寫一個 coroutine 傳進來。"""
        async def go():
            for c in coros_or_msgs:
                await (self.bot.on_message(c) if isinstance(c, Message) else c)
                await self.settle()
        return asyncio.run(go())

    async def settle(self) -> None:
        for _ in range(50):
            tasks = [t for t in self.bot.replies._tasks if not t.done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def replies(self) -> list[tuple]:
        """[(回覆哪則訊息的 id, 內容)]（不含直接 send 的分段）。"""
        return [(mid, content) for mid, content, _ in self.sent if mid is not None]

    def records(self) -> list[dict]:
        import json
        out = []
        for p in sorted(self.trace_dir.glob("ai-*.jsonl")):
            out += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
        return out

    def close(self) -> None:
        asyncio.run(self.bot.brain.aclose())
        self.bot.archive.db.close()
        trace.reset()


__all__ = ["Kit", "FakeLLM", "User", "YUKI", "Clock", "TEST_GUILD", "MAIN_GUILD", "CHANNEL", "LLMError"]
