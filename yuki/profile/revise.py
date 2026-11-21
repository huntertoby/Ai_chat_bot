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
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")  # 條列行：- / * / + / 1.（模型輸出兩種都有）
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
            if _BULLET.match(line):
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


# ---------- 刪除提議與裁決 ----------
@dataclass
class OldLine:
    no: int        # 給模型看的編號（從 1 開始）
    section: str   # 所在的「## 」標題
    text: str      # 整行原文


def numbered_lines(body: str) -> list[OldLine]:
    """舊檔裡可以被提議刪除的條目（條列行；標題、說明、本人更正都不算）。"""
    out, section = [], ""
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s
            continue
        if section == CORRECTION_HEAD or not _BULLET.match(line):
            continue
        if _BULLET.sub("", line).strip() in ("（還不清楚）", "還不清楚"):
            continue
        out.append(OldLine(len(out) + 1, section, line))
    return out


def line_day(text: str) -> str:
    """一行裡最晚出現的日期（取那段期間的開頭：2023 → 2023-01-01，2026-10 → 2026-10-01）；沒有就空字串。"""
    days = []
    for y, m, d in _DATES.findall(text):
        mm = int(m) if m and 1 <= int(m) <= 12 else 1
        dd = int(d) if d and 1 <= int(d) <= 31 else 1
        days.append(f"{y}-{mm:02d}-{dd:02d}")
    return max(days, default="")


@dataclass
class Verdict:
    line: OldLine | None
    evidence: list[Fact]
    reason: str
    ok: bool
    why: str = ""  # 不准的原因（記 log 用）
    raw: dict = field(default_factory=dict)


def adjudicate(proposals: list, old: list[OldLine], facts: list[Fact], file_day: str, *,
               limit: int = MAX_REMOVALS) -> list[Verdict]:
    """AI 只能提議，這裡裁決。facts 是這批新資訊（編號 = 索引 + 1）。規則：
    1. 要刪的行必須在舊檔的編號清單裡（本人更正區塊不在清單裡，所以刪不到）；
    2. 依據必須是這批新資訊裡的編號；能當依據的只有本人更正，或可信度高、不是玩笑／梗的資訊；
    3. 依據要比舊內容新：舊行有寫日期就要晚於那個日期，沒寫就要不早於檔案的「資料到」日期；都沒有日期不刪；
    4. 「群組梗與綽號」裡的只有本人更正能刪（梗本來就不是事實，被反駁不算數）；
    5. 同一行只算一次；准的超過 limit 條，之後的都不刪。
    """
    by_no = {l.no: l for l in old}
    out: list[Verdict] = []
    taken: set[int] = set()
    approved = 0
    for p in proposals or []:
        if not isinstance(p, dict):
            continue
        reason = str(p.get("reason") or "").strip()[:200]
        try:
            no = int(p.get("line"))
        except (TypeError, ValueError):
            no = -1
        ev_nos = []
        for e in p.get("evidence") or []:
            try:
                ev_nos.append(int(e))
            except (TypeError, ValueError):
                pass
        line = by_no.get(no)
        evidence = [facts[i - 1] for i in ev_nos if 1 <= i <= len(facts)]
        v = Verdict(line, evidence, reason, False, raw=p)
        out.append(v)
        if line is None:
            v.why = "要刪的行不在舊檔裡"
            continue
        if no in taken:
            v.why = "重複提議"
            continue
        if not evidence or len(evidence) != len(ev_nos):
            v.why = "依據不在這批新資訊裡"
            continue
        credible = [f for f in evidence
                    if f.kind == "correction" or (f.kind not in _NOT_EVIDENCE and f.confidence == "high")]
        if not credible:
            v.why = "依據不可信（玩笑／梗／別人說的）"
            continue
        old_day = line_day(line.text)
        if old_day:
            newer = [f for f in credible if f.day > old_day]
        elif file_day:
            newer = [f for f in credible if f.day >= file_day]
        else:
            v.why = "舊內容沒有日期，無法確認依據比較新"
            continue
        if not newer:
            v.why = "依據沒有比舊內容新"
            continue
        if line.section == MEME_HEAD and not any(f.kind == "correction" for f in newer):
            v.why = "梗只能由本人更正刪除"
            continue
        if line.section == CORRECTION_HEAD:
            v.why = "本人更正不能刪"
            continue
        if approved >= limit:
            v.why = f"超過一次最多刪 {limit} 條"
            continue
        v.ok, v.evidence = True, newer
        taken.add(no)
        approved += 1
    return out


def remove_lines(body: str, lines: list[OldLine]) -> str:
    """從檔案拿掉這些行（依原文整行比對，只拿掉第一個符合的）。"""
    todo = [l.text for l in lines]
    out = []
    for line in body.splitlines():
        if line in todo:
            todo.remove(line)
            continue
        out.append(line)
    return "\n".join(out)


def insert_line(body: str, section: str, line: str) -> str:
    """還原用：把一行放回原本那一節的最後；那一節不在了就加在檔案最後。"""
    lines = body.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == section)
    except StopIteration:
        return body.rstrip() + f"\n{section}\n{line}\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    # 那一節只有「（還不清楚）」就換掉
    if end == start + 2 and "還不清楚" in lines[start + 1]:
        lines[start + 1] = line
    else:
        lines.insert(end, line)
    return "\n".join(lines)
