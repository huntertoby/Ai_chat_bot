"""中文斷詞（封存區全文檢索用）：jieba，加上成員名字當詞，避免名字被切碎。"""
from __future__ import annotations

import logging
import re
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import jieba  # noqa: E402

jieba.setLogLevel(logging.WARNING)

_KEEP = re.compile(r"[\w一-鿿]")
_STOPWORDS = set("的 了 是 在 我 你 他 她 它 們 和 跟 與 及 或 也 都 就 嗎 呢 吧 啊 喔 哦 欸 什麼 怎麼 哪些 誰 有 沒 不 很 要 會 說 問 一個".split())


def add_words(words: list[str]) -> None:
    """把成員名稱加進 jieba 詞典，避免「Hunterchiao」「小傑傑」被切碎。"""
    from .archive import to_simplified
    for w in words:
        if w and len(w) >= 2:
            jieba.add_word(w, freq=10_000)
            jieba.add_word(to_simplified(w), freq=10_000)  # 封存區全文檢索斷詞前會轉簡體


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in jieba.lcut_for_search(text.lower()):
        tok = tok.strip()
        if tok and _KEEP.search(tok) and tok not in _STOPWORDS:
            tokens.append(tok)
    return tokens
