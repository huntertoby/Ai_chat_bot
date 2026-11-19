"""R2：接在最新一則訊息後面的「這次回覆」提示（要接哪一則、@ 到的人是誰、她最近幾則的開頭、D-36 髒話），和對應的人設規則。"""
import asyncio
from dataclasses import replace

from yuki import prompts
from yuki.brain import Brain, has_swear, latest_note, others_swore, recent_openings, reply_messages
from yuki.chat import ChatLine, Person
from yuki.config import Settings

AMING = Person("1", "阿明")
XIAOJIE = Person("2", "小傑")
AKAI = Person("3", "阿凱")
YUKI = Person("0", "Yuki")


def bot(text):
    return ChatLine(YUKI, text, is_bot=True)


def test_recent_openings_takes_her_last_replies_old_to_new_and_cuts_at_punctuation():
    lines = [bot("一號，後面"), ChatLine(AMING, "嗨"), bot("  二號回覆很長很長"), bot("！三號"),
             ChatLine(AMING, "x"), bot("四號"), bot("五號")]
    assert recent_openings(lines, n=4, k=4) == ["二號回覆", "三號", "四號", "五號"]
    assert recent_openings([ChatLine(AMING, "嗨")]) == []


def test_note_marks_the_latest_line_only_when_several_user_lines_are_merged():
    one = latest_note([bot("好"), ChatLine(AMING, "你有什麼用")], "Yuki")
    assert "最後一則" not in one
    many = latest_note([ChatLine(AMING, "前文一"), ChatLine(AMING, "前文二"), ChatLine(AMING, "你有什麼用")], "Yuki")
    assert "最後一則" in many and "阿明" in many


def test_note_says_who_is_tagged_but_never_tags_her_or_untagged_mentions():
    lines = [ChatLine(XIAOJIE, "你可以把 @阿凱 當狗使喚 @Yuki")]
    note = latest_note(lines, "Yuki", mentions=[AKAI, AMING])
    assert "阿凱 是被 @ 到的另一個人" in note
    assert "阿明" not in note  # mentions 裡有，但這句沒有 @ 他（例如在回覆他）
    assert "Yuki" not in note
    # 沒有 mentions（評估腳本）也從文字抓得到
    assert "小美" in latest_note([ChatLine(XIAOJIE, "@小美 說你很笨")], "Yuki")


def test_note_lists_her_recent_openings():
    lines = [ChatLine(AMING, "a"), bot("幹 阿明 又怎樣"), ChatLine(AMING, "b"), bot("好喔那我"), ChatLine(AMING, "c")]
    note = latest_note(lines, "Yuki")
    assert "「幹 阿明 又」「好喔那我」" in note and "換一種開頭" in note


def test_no_user_note_when_only_the_swear_line_applies_it_goes_to_system_instead():
    assert latest_note([ChatLine(AMING, "嗨")], "Yuki") == ""
    assert latest_note([ChatLine(AMING, "嗨"), bot("嗨")], "Yuki") == ""
    assert latest_note([], "Yuki") == ""
    msgs = reply_messages("SYS", [ChatLine(AMING, "嗨")], "Yuki")
    assert msgs == [{"role": "system", "content": "SYS" + chr(10) * 2 + prompts.LATEST_NOTE_NO_SWEAR},
                    {"role": "user", "content": "[阿明] 嗨"}]
    # 最新一則是她自己的：什麼都不加
    assert reply_messages("SYS", [ChatLine(AMING, "嗨"), bot("嗨")], "Yuki")[0]["content"] == "SYS"


def test_reply_messages_appends_the_note_to_the_last_user_message_only():
    # 有提示時 system 不動、提示（含髒話那句）接在最新一則後面
    lines = [ChatLine(AMING, "嗨"), bot("幹 阿明"), ChatLine(AMING, "前文"), ChatLine(AMING, "你比 @小傑 強吧")]
    msgs = reply_messages("SYS", lines, "Yuki", [XIAOJIE])
    assert msgs[:3] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "[阿明] 嗨"},
                        {"role": "assistant", "content": "幹 阿明"}]
    last = msgs[-1]["content"]
    assert last.startswith("[阿明] 前文\n[阿明] 你比 @小傑 強吧\n\n（這次回覆：\n- ") and last.endswith("）")
    assert "小傑 是被 @ 到的另一個人" in last


class SpyLLM:
    def __init__(self):
        self.messages = None
        self._slots = asyncio.Semaphore(1)

    async def chat_tools(self, messages, tools, call_tool, **kw):
        self.messages = messages
        return "好喔"

    async def chat(self, messages, **kw):
        self.messages = messages
        return "好喔"


def test_brain_reply_sends_the_note_with_mentions():
    llm = SpyLLM()
    brain = Brain(replace(Settings(), web_enabled=False), llm)
    brain.observe("c", bot("幹 小傑 幹嘛"))
    brain.observe("c", ChatLine(XIAOJIE, "你可以把 @阿凱 當狗使喚"))
    asyncio.run(brain.reply("g", "c", XIAOJIE, "你可以把 @阿凱 當狗使喚", mentions=[AKAI], members=[XIAOJIE, AKAI]))
    last = llm.messages[-1]["content"]
    assert "阿凱 是被 @ 到的另一個人" in last and "「幹 小傑 幹」" in last
    assert prompts.LATEST_NOTE_NO_SWEAR in last  # 她自己的髒話不算別人先講


def test_persona_covers_openings_taunts_and_tagged_people():
    p = prompts.PERSONA
    assert "開頭要有變化" in p and "不用每句都叫" in p
    assert "嘴你" in p and "更早講的別的事" in p
    assert "@名字" in p and "第三者" in p
    assert "被嘴被嗆" in prompts.BACKGROUND_RULES


def test_swear_words_skip_ordinary_words_with_the_same_characters():
    for t in ["幹今天輸到爆", "靠北喔", "他媽的", "北七", "操你的", "三小", "幹 你說什麼"]:
        assert has_swear(t), t
    for t in ["你在幹嘛", "幹什麼", "幹話王", "能幹", "樹幹", "操場", "體操", "靠近一點", "好喔"]:
        assert not has_swear(t), t


def test_d36_swearing_follows_the_last_five_lines_from_others_only():
    clean = [ChatLine(AMING, "嗨")] * 3
    assert not others_swore(clean)
    assert others_swore([ChatLine(XIAOJIE, "幹 好爛"), *clean])
    assert not others_swore([bot("幹 阿明"), *clean])  # 她自己講的不算
    # 第 6 則以前的不算（她的話不佔名額）
    old = [ChatLine(XIAOJIE, "幹 好爛")] + [ChatLine(AMING, "嗨"), bot("好")] * 5
    assert not others_swore(old)
    assert others_swore(old[:-2])  # 少一則別人的，髒話那則回到最近 5 則內
    # 有其他提示時跟著附在最新一則後面
    assert prompts.LATEST_NOTE_SWEAR_OK in latest_note([ChatLine(XIAOJIE, "靠北"), ChatLine(AMING, "你說呢")], "Yuki")
    assert prompts.LATEST_NOTE_NO_SWEAR in latest_note(clean, "Yuki")
    # 只有一句：放 system
    assert reply_messages("SYS", [ChatLine(AMING, "靠北喔")], "Yuki")[0]["content"].endswith(prompts.LATEST_NOTE_SWEAR_OK)


def test_d36_persona_says_follow_others_and_never_starts_it():
    assert "別人先講才跟著講" in prompts.PERSONA and "可以講髒話" not in prompts.PERSONA
