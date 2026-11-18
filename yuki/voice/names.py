"""給 ASR 提示用的名字：人物檔案裡的綽號，加上快取（不要每句都讀檔）。只放名字，不放對話。"""
from __future__ import annotations

import re
import time
from pathlib import Path

CACHE_SECONDS = 600  # 最久 10 分鐘重算一次；檔案更新（修改時間變了）也會重算

_QUOTED = re.compile(r"[「『]([^「」『』\n]{2,8})[」』]")
_INLINE = re.compile(r"綽號[:：是為叫]+\s*([^\n。（）()]+)")
_SPLIT = re.compile(r"[、,，/／\s]+")


def nicknames_from_profile(text: str) -> list[str]:
    """從一份人物檔案抓綽號。規則（解析不到就回空串列，不會報錯）：
    1. 標題含「綽號」的段落（例如「## 群組梗與綽號」，到下一個 `## ` 為止）：抓裡面「」或『』括起來的詞；
    2. 全文任何一行有「綽號：甲、乙」「綽號是甲」這種寫法：抓冒號後面的詞，用頓號、逗號、斜線、空白分開，
       到句號或括號為止，詞外面的「」會拿掉；
    3. 每個詞 2～8 個字、不含標點，才算（太長的是句子不是綽號）。"""
    found: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("#"):
            in_section = line.startswith("##") and "綽號" in line
            continue
        if in_section:
            found += _QUOTED.findall(line)
        for m in _INLINE.finditer(line):
            found += [w.strip("「」『』“”\"'*`") for w in _SPLIT.split(m.group(1))]
    out: list[str] = []
    for w in found:
        w = w.strip()
        if 2 <= len(w) <= 8 and not re.search(r"[，。！？!?：:；;（）()「」]", w) and w not in out:
            out.append(w)
    return out


class NicknameCache:
    """讀 `<root>/<伺服器ID>/*.md` 的綽號。每個伺服器各自快取；超過 10 分鐘或有檔案更新才重算。"""

    def __init__(self, files, ttl: float = CACHE_SECONDS, clock=time.monotonic):
        self.files = files  # ProfileFiles（用它的 path 找到資料夾）
        self.ttl = ttl
        self._clock = clock
        self._cache: dict[str, tuple[float, float, list[str]]] = {}  # guild → (算的時間, 最新修改時間, 綽號)

    def _paths(self, guild_id: str) -> list[Path]:
        folder = self.files.path(guild_id, "x").parent
        try:
            return sorted(folder.glob("*.md"))
        except OSError:
            return []

    def get(self, guild_id: str) -> list[str]:
        now = self._clock()
        paths = self._paths(guild_id)
        try:
            newest = max((p.stat().st_mtime for p in paths), default=0.0)
        except OSError:
            newest = 0.0
        hit = self._cache.get(guild_id)
        if hit and now - hit[0] < self.ttl and hit[1] == newest:
            return hit[2]
        names: list[str] = []
        for p in paths:
            try:
                names += nicknames_from_profile(p.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
        names = list(dict.fromkeys(names))
        self._cache[guild_id] = (now, newest, names)
        return names
