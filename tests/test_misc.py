"""小工具：JSON 抽取、推理標籤、群聊轉 messages、長訊息切段、bot 身分組的 @。"""
from yuki.bot import split_message
from yuki.brain import to_messages
from yuki.chat import ChatLine, Person
from yuki.llm import extract_json, strip_think

TOBY = Person("2", "Toby")
BOT = Person("0", "Yuki")


def test_extract_json():
    assert extract_json('<think>{"x":1}</think>答案 {"ops": []}') == {"ops": []}
    assert extract_json('```json\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}
    assert extract_json("完全不是 JSON") is None
    assert extract_json("") is None


def test_strip_think():
    assert strip_think("<think>想很久</think>\n你好") == "你好"
    assert strip_think("想很久</think>你好") == "你好"


def test_strip_think_old_behavior_unchanged():
    # 沒有思考內容的回話不變（只去頭尾空白，跟原本一樣）
    for s in ["你好", "  哈哈 XD  ", "a < b 而且 c > d", "<b>粗體</b>", "{\"x\": 1}", ""]:
        assert strip_think(s) == s.strip()
    # <think> 的處理照舊（含只有開頭標籤時只拿掉標籤）
    assert strip_think("<think>a</think>b<think>c</think>d") == "bd"
    assert strip_think("<think>沒寫完") == "沒寫完"
    assert strip_think(None) == ""


def test_strip_think_gemma_channel():
    assert strip_think("<|channel>thought\n想一想\n<channel|>你好") == "你好"
    assert strip_think("<|channel>thought\n<channel|>你好") == "你好"  # 關思考時的空思考
    assert strip_think("前<|channel>thought\nx<channel|>後") == "前後"
    # 只有結尾（開頭被 template 吃掉）
    assert strip_think("User wants...\n<channel|>你好") == "你好"
    # 被截斷在思考中：思考一個字都不能留
    assert strip_think("<|channel>thought\n* User asks about") == ""
    assert strip_think("答案<|channel>thought\n* 後面是思考") == "答案"
    # 兩種格式混在一起
    assert strip_think("<|channel>thought\nA<channel|><think>B</think>你好") == "你好"
    assert extract_json('<|channel>thought\n{"x":1}<channel|>{"ok": true}') == {"ok": True}
    assert extract_json('<|channel>thought\n{"x":1}') is None


def test_to_messages_merges_group_chat():
    lines = [ChatLine(BOT, "早", is_bot=True), ChatLine(TOBY, "嗨"), ChatLine(Person("1", "Hunter"), "安"),
             ChatLine(BOT, "你們好", is_bot=True)]
    msgs = to_messages(lines)
    assert msgs[0] == {"role": "user", "content": "[Toby] 嗨\n[Hunter] 安"}
    assert msgs[1] == {"role": "assistant", "content": "你們好"}


def test_split_message():
    chunks = split_message("a" * 4000)
    assert all(len(c) <= 1900 for c in chunks) and "".join(chunks) == "a" * 4000
    assert split_message("短訊息") == ["短訊息"]
