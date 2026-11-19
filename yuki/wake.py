"""喚醒詞：訊息裡有沒有她的名字、是不是「直接叫她」（D-32，開發規格 6.1）。

bot.py 和評估腳本（scripts/eval_chime.py）共用，兩邊才不會漂移。

- 直接叫名字（`direct_call`）：名字在句首（前面只允許空白、標點、@、短的招呼語），而且名字後面
  是標點或結尾、緊接著「你／妳」或「在嗎」、或剩下的話是結束字／閉嘴類。→ 一定回。
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
# 「Yuki 在嗎」「Yuki 在不在」：問她在不在，也是在叫她
_THERE = re.compile(r"\s*在(?:嗎|嘛|不在|[?？]|\s*$)")


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
    if _YOU.match(rest) or _THERE.match(rest):  # 「Yuki 你…」「Yuki 在嗎」
        return True
    return is_exit(rest, exit_words) or is_shush(rest)


# 名字後面接這些標點才算叫她（「@」「#」「（」之類不算：多半是在講別的）
_PUNCT = set(",，、.。!！?？~～…‥:：;；")


def _is_punct(ch: str) -> bool:
    return ch in _PUNCT


def named(text: str, wake: re.Pattern) -> bool:
    """訊息裡有她的名字（哪裡都算）。"""
    return bool(wake.search(text or ""))


# ---------- D-34：程式端先擋「明顯不是跟她講」的訊息（不呼叫模型）----------
def calls_other_bot(text: str, bots: re.Pattern | None) -> bool:
    """句首是別的 bot 的名字（前面只允許空白、標點、@、短招呼語；英數字的名字要整個字相符，不分大小寫）。
    句中提到別的 bot 不算。bots：`wake_pattern(其他 bot 的名字)`，None＝伺服器裡沒有別的 bot。"""
    return bots is not None and _rest_after_leading_name(text or "", bots) is not None


def addressed_elsewhere(text: str, wake: re.Pattern, *, mentions_others: bool, mentions_her: bool = False,
                        other_bots: re.Pattern | None = None) -> str | None:
    """沒被叫到的訊息，是不是明顯在跟別人（別的 bot）講。回傳擋下的原因，None＝交給模型判斷。

    - @ 了別人（使用者、其他 bot、身分組），而且沒有 @她、沒提到她名字 → "at_other"
    - 句首在叫別的 bot，而且沒提到她 → "other_bot"
    回覆別人的訊息不在這裡擋（可能是在接她的話題）：呼叫端算 mentions_others 時不要把「回覆時自動 ping 對方」算進去。
    bot.py（主伺服器注意模式、測試伺服器插話）和評估腳本共用。"""
    if mentions_her or named(text, wake):
        return None
    if mentions_others:
        return "at_other"
    if calls_other_bot(text, other_bots):
        return "other_bot"
    return None
