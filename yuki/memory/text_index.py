"""中文斷詞 + BM25（sparse 檢索）。記憶量在幾千筆以內，自己算比拉一個套件更好控制。"""
from __future__ import annotations

import logging
import math
import re
import warnings
from collections import Counter

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import jieba  # noqa: E402

jieba.setLogLevel(logging.WARNING)

_KEEP = re.compile(r"[\w一-鿿]")
_STOPWORDS = set("的 了 是 在 我 你 他 她 它 們 和 跟 與 及 或 也 都 就 嗎 呢 吧 啊 喔 哦 欸 什麼 怎麼 哪些 誰 有 沒 不 很 要 會 說 問 一個".split())


def add_words(words: list[str]) -> None:
    """把成員名稱加進 jieba 詞典，避免「Hunterchiao」「小傑傑」被切碎。"""
    from ..archive import to_simplified
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


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avgdl = (sum(self.lens) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        # 加 1 的 idf 永遠為正，語料很小時也不會出現負分
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out
