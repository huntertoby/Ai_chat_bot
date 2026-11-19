"""喚醒詞：訊息裡有沒有她的名字、是不是「直接叫她」（D-32，開發規格 6.1）。

bot.py 和評估腳本（scripts/eval_chime.py）共用，兩邊才不會漂移。

- 直接叫名字（`direct_call`）：名字在句首（前面只允許空白、標點、@、短的招呼語），而且名字後面
  是標點或結尾、緊接著「你／妳」、或剩下的話是結束字／閉嘴類。→ 一定回。
- 其他有名字的（句中、句尾、句首但接的是評論）＝提到她（`named`）：有開插話的伺服器交給插話判斷。
"""
from __future__ import annotations

import re
from typing import Iterable

from .chime import is_shush


def wake_pattern(words: list[str]) -> re.Pattern:
    """喚醒詞：英數字的詞要整個字相符（前後不是英文字母或數字），中文照字比對；不分大小寫。"""
    parts = []
    for w in words:
        if not w:
            continue
        if w.isascii() and any(c.isalnum() for c in w):
            parts.append(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])")
        else:
            parts.append(re.escape(w))
    return re.compile("|".join(parts) or r"(?!x)x", re.IGNORECASE)


_NOT_WORD = re.compile(r"[\W_]+")  # 標點、空白、表情符號


def is_exit(text: str, exit_words: Iterable[str]) -> bool:
    """叫她離開：整句（去掉喚醒詞、標點、表情後）就是結束字之一才算；句子裡夾著結束字不算。"""
    core = _NOT_WORD.sub("", text or "").lower()
    return bool(core) and core in {_NOT_WORD.sub("", w).lower() for w in exit_words}


# 名字前後可以出現的短招呼語（只算這幾個，不是任何字都行）
GREETINGS = ("欸", "誒", "ㄟ", "嘿", "喂", "嗨", "哈囉", "hey", "hi", "hello", "yo")
# 英文招呼語要整個字（「hiyuki」不算 hi＋yuki）；中文的直接接名字也行（「嘿Yuki」）
_GREET = "|".join(re.escape(g) + (r"(?![A-Za-z0-9])" if g.isascii() else "")
                  for g in sorted(GREETINGS, key=len, reverse=True))
# 名字前面：空白、標點、@、表情（\W）和最多兩個招呼語
_LEAD = re.compile(rf"[\W_]*(?:(?:{_GREET})[\W_]*){{0,2}}", re.IGNORECASE)
# 名字後面只剩標點、表情、招呼語（整則就是叫名字）
_ONLY_GREETING = re.compile(rf"(?:[\W_]|{_GREET})*", re.IGNORECASE)
_YOU = re.compile(r"\s*[你妳]")


def _rest_after_leading_name(text: str, wake: re.Pattern) -> str | None:
    """名字在句首（前面只有空白、標點、@、招呼語）：回傳名字後面的文字；不在句首回 None。"""
    lead = _LEAD.match(text)
    m = wake.match(text, lead.end()) if lead else None
    return text[m.end():] if m else None


def direct_call(text: str, wake: re.Pattern, exit_words: Iterable[str] = ()) -> bool:
    """直接叫名字：名字在句首，後面是標點／結尾、「你／妳」、或結束字／閉嘴類。"""
    rest = _rest_after_leading_name(text or "", wake)
    if rest is None:
        return False
    if _ONLY_GREETING.fullmatch(rest):  # 只叫名字，或名字加招呼、標點
        return True
    head = rest.lstrip()
    if head and _is_punct(head[0]):  # 名字後面接標點：「Yuki，…」「Yuki? …」
        return True
    if _YOU.match(rest):  # 「Yuki 你…」
        return True
    return is_exit(rest, exit_words) or is_shush(rest)


# 名字後面接這些標點才算叫她（「@」「#」「（」之類不算：多半是在講別的）
_PUNCT = set(",，、.。!！?？~～…‥:：;；")


def _is_punct(ch: str) -> bool:
    return ch in _PUNCT


def named(text: str, wake: re.Pattern) -> bool:
    """訊息裡有她的名字（哪裡都算）。"""
    return bool(wake.search(text or ""))
