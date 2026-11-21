"""人物檔案的修正（開發規格 6.5）：「本人更正」區塊、被反駁的舊內容刪除裁決。

- 本人更正：抽取時 kind=correction、而且程式確認那句話是本人說的（`builder.parse_items`）。
  .md 裡的「## 本人更正」區塊由程式產生（資料庫裡這個人所有的 correction + 舊檔區塊裡的），
  不交給模型改寫，重建檔案時也一樣在。
- 刪除：模型只能「提議」刪哪一行、依據哪幾條新資訊；這裡的 `adjudicate` 決定准不准。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .store import Fact

CORRECTION_HEAD = "## 本人更正"
CORRECTION_NOTE = "（這一節是他本人親口更正的，跟下面其他內容衝突時以這裡為準）"
MEME_HEAD = "## 群組梗與綽號"
MAX_REMOVALS = 3          # 一次合併（一個人）最多刪幾行
_NOT_EVIDENCE = ("joke", "meme")
_HEADER_DAY = re.compile(r"資料到 (\d{4}-\d{2}-\d{2})")
# 行內的日期：2026-10-05、2026-10、2026/10、2023 年、2023：
_DATES = re.compile(r"(?<!\d)(20\d{2})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?(?!\d)")


# ---------- 本人更正區塊 ----------
def correction_line(f: Fact) -> str:
    return f"- （{f.day}）{f.text}"


def split_corrections(text: str) -> tuple[list[str], str]:
    """拆出「## 本人更正」區塊的條目，回傳 (條目, 其餘內容)。"""
    lines = text.splitlines()
    out, items, inside = [], [], False
    for line in lines:
        if line.strip() == CORRECTION_HEAD:
            inside = True
            continue
        if inside and line.startswith("## "):
            inside = False
        if inside:
            if line.strip().startswith("- "):
                items.append(line.strip())
            continue
        out.append(line)
    return items, "\n".join(out)


def merge_corrections(old: list[str], new: list[str]) -> list[str]:
    """舊檔的條目 + 資料庫的條目，去重，依日期排序。"""
    seen, out = set(), []
    for line in old + new:
        key = re.sub(r"\s+", "", line)
        if key not in seen:
            seen.add(key)
            out.append(line)
    return sorted(out, key=lambda l: (m.group(0) if (m := re.search(r"\d{4}-\d{2}-\d{2}", l)) else ""))


def render_corrections(items: list[str]) -> str:
    return f"{CORRECTION_HEAD}\n{CORRECTION_NOTE}\n" + "\n".join(items) + "\n\n" if items else ""


def corrections_first(text: str) -> str:
    """把「本人更正」移到最前面（回覆 prompt 用；檔案本來就這樣寫，這裡是保險）。"""
    items, rest = split_corrections(text)
    return (render_corrections(items) + rest.strip()).strip() if items else text


def header_day(text: str) -> str:
    m = _HEADER_DAY.search(text)
    return m.group(1) if m else ""


