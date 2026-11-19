"""R3：回覆壞掉的程式檢查（講出提示規則、照抄對方）和壞掉時重生。"""
import asyncio

import pytest

from yuki import reply_check
from yuki.brain import checked_chat
from yuki.chat import ChatLine, Person
from yuki.reply_check import broken_reason

AMING = Person("1", "阿明")
XIAOMEI = Person("2", "小美")
YUKI = Person("0", "Yuki")


@pytest.mark.parametrize("reply", [
    "阿凱在鬧我，接一下就好。不用查了。\n\n廢話就廢話",          # 規劃怎麼回
    "好，我陪他罵、接住那句廢物就好，不講髒話。",
    "小美在問鑰匙鎖在家裡怎麼辦，這是單純閒聊求助。",
    "（這次回覆：- 換一種開頭）好喔",                              # 提示本身
    "他是被 @ 到的另一個人，所以我不用理",
])
def test_meta_leaks_are_caught(reply):
    assert broken_reason(reply) == "meta"


def test_narrating_the_speaker_then_planning_is_caught_but_plain_mentions_are_not():
    others = [("阿明", "你比 @小傑 還會打吧")]
    assert broken_reason("阿明在問我比小傑還會打吧，查不到小傑。那就回一下", others=others) == "meta"
    assert broken_reason("阿明在問這個喔，我哪會打", others=others) == ""
    assert broken_reason("小傑在問你啦 不是我", others=others) == ""  # 不是說話的人


def test_echo_of_recent_message_or_chat_format_is_caught():
    others = [("小美", "今天被主管念"), ("小美", "你講話好像客服喔 笑死")]
    assert broken_reason("你講話好像客服喔 笑死", others=others) == "echo"
    assert broken_reason("你講話好像客服喔，笑死！", others=others) == "echo"  # 標點不同也算
    assert broken_reason("我新買的鍵盤到了\n[阿明] 紅軸打起來好爽", others=others) == "echo"


@pytest.mark.parametrize("reply", [
    "哈 說得好像你很懂似的，我至少還在混時間",
    "客服？我哪有那麼有禮貌，你吃太鹹啦",
    "你講話好像客服喔？我才不是客服咧，我是專業陪聊，你那碗拉麵鹹到我都隔著螢幕口渴了",  # 引用但自己講更多
    "好喔 [貼圖:哭哭]",
    "幹嘛啦 你又不用查我",  # 「不用查」…是規則片語，見下一個測試
])
def test_normal_replies_pass_except_rule_phrases(reply):
    others = [("小美", "你講話好像客服喔 笑死")]
    expected = "meta" if "不用查" in reply else ""
    assert broken_reason(reply, others=others) == expected


def test_short_messages_are_not_used_for_echo():
    assert reply_check.ECHO_MIN_CHARS >= 4
    assert broken_reason("好喔", others=[("阿明", "好喔")]) == ""


class ScriptedGen:
    """依序回固定的回覆，記下每次收到的 messages。"""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    async def __call__(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


def lines():
    return [ChatLine(AMING, "嗨"), ChatLine(YUKI, "幹 阿明", is_bot=True), ChatLine(AMING, "你到底有什麼用啦")]


def test_checked_chat_passes_good_reply_through_once():
    gen = ScriptedGen("[阿明] 至少我會回你")
    stats = {}
    out = asyncio.run(checked_chat(gen, "SYS", lines(), "Yuki", stats=stats))
    assert out == "至少我會回你" and len(gen.seen) == 1
    assert stats == {"reasons": [], "extra_ms": 0}


def test_checked_chat_retries_then_falls_back_without_note():
    gen = ScriptedGen("這是在嘴我，接住就好", "你到底有什麼用啦", "至少我會回你")
    stats = {}
    out = asyncio.run(checked_chat(gen, "SYS", lines(), "Yuki", stats=stats))
    assert out == "至少我會回你"
    assert stats["reasons"] == ["meta", "echo"] and stats["fallback"] is True
    assert "這次回覆" in gen.seen[0][-1]["content"] and "這次回覆" in gen.seen[1][-1]["content"]
    assert "這次回覆" not in gen.seen[2][-1]["content"]  # 最後一招：不附提示
    assert gen.seen[2][0]["content"].startswith("SYS")


def test_checked_chat_stops_after_good_retry():
    gen = ScriptedGen("你到底有什麼用啦", "至少我會回你")
    stats = {}
    assert asyncio.run(checked_chat(gen, "SYS", lines(), "Yuki", stats=stats)) == "至少我會回你"
    assert stats["reasons"] == ["echo"] and "fallback" not in stats and len(gen.seen) == 2
