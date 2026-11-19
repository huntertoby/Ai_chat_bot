"""R3：回覆壞掉的程式檢查。兩種壞法：
- meta：把「這次回覆」提示或人設的規則講出來（像在自言自語規劃怎麼回）；
- echo：照抄對方最近講的話（含對話紀錄的 [名字] 格式）。
另外只有刪節號、句點（「...」）的回覆算 empty。只看字面，不呼叫模型。門檻都是常數，誤判（正常回覆被擋）在 R3 報告量過。"""
from __future__ import annotations

import re

# 提示和人設規則裡的片語：正常聊天幾乎不會這樣講，模型在「規劃怎麼回」時才會冒出來
META_PATTERNS = [
    r"這次回覆", r"系統提示", r"前面幾則是前文", r"不講髒話", r"不要講髒話", r"不能講髒話",
    r"不用查", r"單純閒聊", r"直接接", r"接住(?:那句|這句|就好)", r"接一下就好", r"接回去就好",
    r"是在嘴我", r"針對這句", r"針對那句", r"回一句短的", r"自嘲或", r"或輕輕嗆",
    r"換一種開頭", r"一樣的起頭", r"起頭方式", r"第三者", r"被 ?@ ?到",
    r"給點建議就好", r"不用太認真", r"就直接(?:陪他|回他|回她|給他|給她)",
    r"(?:輕鬆|直接|簡單)接(?:一下|話|梗)?就好", r"沒有特別的背景", r"(?:幫|陪|回)他\S{1,8}就好",
]
_META = re.compile("|".join(META_PATTERNS))
# 開頭用第三人稱轉述說話的人在幹嘛（「阿明在問…」「小美把…」後面接規劃），用說話者名字判斷
NARRATE_VERBS = r"(?:在問|在講|在說|在抱怨|在鬧|在嘴|問我|說我|叫我|說|問|又問|講)"
NARRATE_TAIL = re.compile(r"就好|回一下|接話|回嘴|不用|我(?:可以|要|就|來)?接|接這個|接梗")
# 對話紀錄格式：行首的「[名字] 內容」（[貼圖:名稱] 是貼圖語法；markdown 連結 [字](網址)、句中的方括號不算）
_CHAT_TAG = re.compile(r"(?m)^[ \t]*\[(?!貼圖)[^\]\n]{1,32}\][ \t]+\S")
# 照抄查聊天紀錄的結果格式：日期後面直接接「某句話」，或「日期 時:分」
_SEARCH_FORMAT = re.compile(r"\d{4}-\d{2}-\d{2}\s*「[^」]{1,80}」\s*$|\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}")
# 開頭的「我：」：模型學對話紀錄格式替自己加的發言者，直接拿掉（不算壞掉、不重生）
SELF_PREFIX = re.compile(r"^\s*我\s*[：:]\s*")
ECHO_MIN_CHARS = 6     # 對方那句（去掉空白標點）至少幾個字才比對，太短的（「好喔」）不算
ECHO_RATIO = 0.5       # 照抄的部分佔回覆一半以上才算
ECHO_LOOKBACK = 3      # 看最近幾則別人的訊息
_STRIP = re.compile(r"[\s,，.。!！?？~～…、:：;；「」『』()（）\"'`*]+")


_DOTS_ONLY = re.compile(r"^[\s.。…,，、~～·]*$")
_EMOJI = re.compile(r"<a?:\w+:\d+>|:\w+:")  # 群組表情不算照抄（大家本來就會按同一個）


def _norm(text: str) -> str:
    return _STRIP.sub("", _EMOJI.sub("", text or "")).lower()


def broken_reason(reply: str, *, others: list[tuple[str, str]] = ()) -> str:
    """回覆壞掉的原因："meta"（講出提示規則）、"echo"（照抄對方）、"empty"（只有刪節號）；沒壞回 ""。
    others：最近幾則別人的訊息 [(名字, 內容)]，舊到新。"""
    text = (reply or "").strip()
    if not text:
        return ""
    if _DOTS_ONLY.match(text):  # 只有刪節號、句點（「...」），沒有內容；「？？？」這種反應不算
        return "empty"
    if _META.search(text):
        return "meta"
    head = text[:40]
    for name, _ in others:
        if name and re.match(re.escape(name) + NARRATE_VERBS, head) and NARRATE_TAIL.search(text[:120]):
            return "meta"
    if _CHAT_TAG.search(text) or _SEARCH_FORMAT.search(text):
        return "echo"
    for name, _ in others:  # 開頭「別人的名字：」＝替別人講話（模仿對話紀錄格式）
        if name and re.match(re.escape(name) + r"\s*[：:]", text):
            return "echo"
    r = _norm(text)
    for _, msg in list(others)[-ECHO_LOOKBACK:]:
        m = _norm(msg)
        if len(m) >= ECHO_MIN_CHARS and m in r and len(m) >= ECHO_RATIO * len(r):
            return "echo"
    return ""
