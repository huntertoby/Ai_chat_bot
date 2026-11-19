"""追蹤紀錄的隱私：插話判斷不留 prompt、/不要記我 的人從紀錄裡拿掉（不需要網路、模型、Discord）。"""
import asyncio

import pytest

from botkit import Kit
from yuki import trace
from yuki.chat import ChatLine, Person, ReplyRef

AMING, XIAOJIE, XIAOMEI, YUKI = Person("1", "阿明"), Person("2", "小傑"), Person("3", "小美"), Person("9", "Yuki")


@pytest.fixture
def kit(tmp_path):
    k = Kit(tmp_path)
    k.bot.archive.opt_out("g", "1")  # 阿明用過 /不要記我
    yield k
    k.close()


def all_text(kit) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in kit.trace_dir.glob("ai-*.jsonl"))


def test_chime_trace_keeps_only_the_decision_not_the_conversation(tmp_path):
    trace.reset()
    trace.configure(tmp_path)

    async def go():
        async with trace.start("chime", keep_prompts=False) as tr:
            tr.add_attempt([{"role": "user", "content": "[阿明] 私密內容"}], {"x": 1}, 5)
            tr.final = "{}"
        async with trace.start("reply") as tr:
            tr.add_attempt([{"role": "user", "content": "[阿明] 私密內容"}], {"x": 1}, 5)
    asyncio.run(go())
    trace.reset()
    text = [p.read_text(encoding="utf-8") for p in tmp_path.glob("ai-*.jsonl")][0].splitlines()
    import json
    a, b = (json.loads(x) for x in text)
    assert a["attempts"][0]["messages"] == [] and a["attempts"][0]["response"] == {"x": 1}
    assert b["attempts"][0]["messages"][0]["content"] == "[阿明] 私密內容"


def test_opted_out_people_are_scrubbed_from_someone_elses_trace(kit):
    hist = [ChatLine(AMING, "我的秘密計畫是搬家"),
            ChatLine(XIAOJIE, "真的假的", reply_to=ReplyRef("阿明", False, "我的秘密計畫是搬家")),
            ChatLine(XIAOMEI, "小美說了普通的話"),
            ChatLine(YUKI, "好喔", is_bot=True)]
    kit.bot.brain.history["5"].extend(hist)

    async def go():
        async with trace.start("reply", guild="7", user_ids=["2"]) as tr:
            kit.bot._hide_optouts(tr, "5", [])
            msgs = [{"role": "system", "content": "在場的人：阿明、小傑"},
                    {"role": "user", "content": "\n".join(l.render_chat() for l in hist if not l.is_bot)}]
            tr.add_attempt(msgs, {"content": "阿明的事我不知道"}, 3)
            tr.final = "阿明的事我不知道"
    asyncio.run(go())
    text = all_text(kit)
    assert "阿明" not in text and "秘密計畫" not in text
    assert "真的假的" in text and "小美說了普通的話" in text  # 其他人照常


def test_opted_out_person_quoted_but_no_longer_in_history_is_scrubbed(kit):
    hist = [ChatLine(XIAOJIE, "你說的對", reply_to=ReplyRef("阿明", False, "早就過期的那句話"))]
    kit.bot.brain.history["5"].extend(hist)

    async def go():
        async with trace.start("chime", guild="7", user_ids=["2"]) as tr:
            kit.bot._hide_optouts(tr, "5", [AMING])
            tr.add_attempt([{"role": "user", "content": hist[0].render_chat()}], None, 1)
    asyncio.run(go())
    text = all_text(kit)
    assert "阿明" not in text and "早就過期" not in text and "你說的對" in text


def test_record_triggered_by_an_opted_out_person_is_not_written_at_all(tmp_path):
    trace.reset()
    trace.configure(tmp_path, opted_out=lambda g, uids: "1" in uids)

    async def go():
        async with trace.start("chime", guild="7", user_ids=["1"]):
            pass
    asyncio.run(go())
    trace.reset()
    assert list(tmp_path.glob("ai-*.jsonl")) == []
