"""prompt 的寫法規則（CLAUDE.md）：只描述做法，不放會被照抄的範例句；不同段落不互相矛盾。"""
import json
import re

import pytest

from yuki import prompts
from yuki.tools import archive as archive_tools, web as web_tools

REPLY_PARTS = ["PERSONA", "WEB_RULES", "SPOKEN_RULES", "OWNER_RULES", "ARCHIVE_RULES", "BACKGROUND_RULES",
               "EMOJI_RULES", "CHIMED_RULES", "VISION_RULES",
               "LATEST_NOTE_LAST", "LATEST_NOTE_AT", "LATEST_NOTE_OPENINGS"]
CHIME_PARTS = ["CHIME_SYSTEM", "CHIME_ATTENTION", "CHIME_EMOJI", "CHIME_EMOJI_HINT"]
# 引號裡允許的長片語：段落標題、格式說明、要避開的說法（不是要她講的話）
ALLOWED_QUOTES = {"被回覆的那句", "中華民國才是中國", "順著對方的說法反轉"}


def tool_descriptions() -> str:
    return json.dumps(archive_tools.TOOL_SPECS + web_tools.TOOL_SPECS, ensure_ascii=False)


@pytest.mark.parametrize("name", REPLY_PARTS + CHIME_PARTS + ["tools"])
def test_no_quoted_sentences_she_could_copy(name):
    """引號裡只能是短詞（段落名稱、要避開的用語），不能是整句範例。"""
    text = tool_descriptions() if name == "tools" else getattr(prompts, name)
    for q in re.findall(r"「([^」]*)」", text):
        assert len(q) <= 8 or q in ALLOWED_QUOTES, q


@pytest.mark.parametrize("name", REPLY_PARTS + ["tools"])
def test_reply_prompts_give_no_worked_examples(name):
    """「例如問 X 就給 A、B、C」這種範例會被原封不動拿去用在每一題。"""
    text = tool_descriptions() if name == "tools" else getattr(prompts, name)
    assert not re.search(r"例如[^。\n]{0,15}[「\[\"]", text)


def test_reply_prompt_does_not_point_at_blocks_that_do_not_exist():
    """D-04 起回覆前不預先檢索記憶，也沒有白板：prompt 不能叫她看那些區塊。"""
    p = prompts.reply_system("Yuki", web=True, archive=True)
    assert "記憶」" not in p and "群組近況" not in p


def test_emoji_frequency_is_stated_only_when_there_is_a_catalog():
    """表情比例只寫在 EMOJI_RULES（有圖鑑才出現）。人設只說「有群組表情時偶爾可以用」，不另外講一個比例。"""
    assert not re.search(r"[一二三四五六七八九十\d]成", prompts.PERSONA)
    assert "群組自己的表情與貼圖" not in prompts.reply_system("Yuki")
    assert re.search(r"[一二三四五六七八九十\d]成", prompts.EMOJI_RULES)


def test_small_talk_needs_no_lookup_but_unfamiliar_things_do():
    """閒聊不用查（聊天紀錄、網路都不用）；B4 的「不熟的人事物先查」還在。"""
    text = prompts.reply_system("Yuki", web=True, archive=True)
    assert "閒聊" in prompts.BACKGROUND_RULES and "不用查" in prompts.BACKGROUND_RULES
    assert "先查再接話" in text


def test_background_rules_can_be_turned_off_only_by_argument(monkeypatch):
    monkeypatch.setenv("YUKI_EVAL_NO_BACKGROUND_RULES", "1")  # 舊的環境變數開關不再有作用
    assert prompts.BACKGROUND_RULES in prompts.reply_system("Yuki", web=True)
    assert prompts.BACKGROUND_RULES not in prompts.reply_system("Yuki", web=True, background_rules=False)


def test_chime_reason_is_kept_short():
    assert prompts.CHIME_SCHEMA["properties"]["reason"]["maxLength"] <= 30
