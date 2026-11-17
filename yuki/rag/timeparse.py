"""從問題裡解析時間說法（昨天、上週、上個月、去年、3 月……），轉成日期區間給檢索過濾。

規則式、零 LLM 呼叫。只認得出明確的說法；「上次」「之前」這種沒有具體時間的不處理。
「最近」「這陣子」是模糊說法（soft）：先找近兩週，找不到再放寬到兩個月。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

_CN = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUM = r"(\d{1,4}|[零一二兩三四五六七八九十]{1,3})"


def cn_int(s: str) -> int:
    """「12」「十二」「二十」「兩」→ 整數。"""
    if s.isdigit():
        return int(s)
    if "十" in s:
        tens, _, ones = s.partition("十")
        return (_CN.get(tens, 1) if tens else 1) * 10 + (_CN.get(ones, 0) if ones else 0)
    return _CN.get(s, 0)


@dataclass(frozen=True)
class TimeRange:
    start: date               # 含
    end: date                 # 含
    phrase: str               # 對到的說法，除錯用
    soft: bool = False        # 模糊說法：找不到可以放寬

    def windows(self) -> list[tuple[str, str]]:
        """依序嘗試的 (since, until)；soft 會多一個放寬的區間。"""
        wins = [(self.start.isoformat(), self.end.isoformat())]
        if self.soft:
            wins.append(((self.end - timedelta(days=60)).isoformat(), self.end.isoformat()))
        return wins


_EARLIEST = re.compile(r"最早|第一次|頭一次|最初|一開始|什麼時候開始")
_LATEST = re.compile(r"最後一次|最近一次|上一次是|上次是什麼時候")


def parse_order(text: str) -> str | None:
    """「最早／第一次」→ earliest，「最後一次」→ latest。答案在記憶的日期裡，要依日期排序而不是相關度。"""
    if _EARLIEST.search(text):
        return "earliest"
    if _LATEST.search(text):
        return "latest"
    return None


def _month_range(y: int, m: int) -> tuple[date, date]:
    while m < 1:
        y, m = y - 1, m + 12
    start = date(y, m, 1)
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return start, nxt - timedelta(days=1)


def _past_month(today: date, m: int) -> int:
    """只說「3 月」時指最近一次過去的 3 月。"""
    return today.year if m <= today.month else today.year - 1


def parse_time(text: str, today: date) -> TimeRange | None:
    def day(d: date, phrase: str) -> TimeRange:
        return TimeRange(d, d, phrase)

    # ---- 明確日期 ----
    if m := re.search(r"(\d{4})\s*[年/-]\s*(\d{1,2})\s*[月/-]\s*(\d{1,2})\s*[日號]?", text):
        try:
            return day(date(int(m[1]), int(m[2]), int(m[3])), m[0])
        except ValueError:
            pass
    if m := re.search(rf"(?<!個){_NUM}\s*月\s*{_NUM}\s*[日號]", text):
        mo, d = cn_int(m[1]), cn_int(m[2])
        if 1 <= mo <= 12:
            try:
                return day(date(_past_month(today, mo), mo, d), m[0])
            except ValueError:
                pass
    if m := re.search(r"(\d{4})\s*年\s*" + _NUM + r"\s*月", text):
        mo = cn_int(m[2])
        if 1 <= mo <= 12:
            return TimeRange(*_month_range(int(m[1]), mo), m[0])
    if m := re.search(r"(\d{4})\s*年", text):
        return TimeRange(date(int(m[1]), 1, 1), date(int(m[1]), 12, 31), m[0])

    # ---- 相對的天 ----
    for word, delta in (("大前天", 3), ("前天", 2), ("昨天", 1), ("昨日", 1), ("昨晚", 1),
                        ("今天", 0), ("今日", 0), ("今晚", 0)):
        if word in text:
            return day(today - timedelta(days=delta), word)
    if m := re.search(_NUM + r"\s*天前", text):
        d = today - timedelta(days=cn_int(m[1]))
        return TimeRange(d - timedelta(days=1), d + timedelta(days=1), m[0])

    # ---- 週 ----
    monday = today - timedelta(days=today.weekday())
    if m := re.search(r"上(?:個)?(?:週|周|禮拜|星期)([一二三四五六日天])", text):
        weekday = 6 if m[1] in "日天" else "一二三四五六".index(m[1])
        return day(monday - timedelta(days=7 - weekday), m[0])
    if re.search(r"上上(?:個)?(?:週|周|禮拜|星期)", text):
        return TimeRange(monday - timedelta(days=14), monday - timedelta(days=8), "上上週")
    if m := re.search(r"上(?:個)?(?:週|周|禮拜|星期)(?![一二三四五六日天])", text):
        return TimeRange(monday - timedelta(days=7), monday - timedelta(days=1), m[0])
    if m := re.search(r"(?:這|本)(?:個)?(?:週|周|禮拜|星期)", text):
        return TimeRange(monday, today, m[0])
    if m := re.search(_NUM + r"\s*(?:個)?(?:週|周|禮拜|星期)前", text):
        end = today - timedelta(days=7 * cn_int(m[1]))
        return TimeRange(end - timedelta(days=6), end + timedelta(days=3), m[0])

    # ---- 月 ----
    if re.search(r"上上(?:個)?月", text):
        return TimeRange(*_month_range(today.year, today.month - 2), "上上個月")
    if m := re.search(r"上(?:個)?月", text):
        return TimeRange(*_month_range(today.year, today.month - 1), m[0])
    if m := re.search(r"(?:這|本)(?:個)?月", text):
        return TimeRange(date(today.year, today.month, 1), today, m[0])
    if m := re.search(_NUM + r"\s*個月前", text):
        return TimeRange(*_month_range(today.year, today.month - cn_int(m[1])), m[0])

    # ---- 年 ----
    for word, delta in (("大前年", 3), ("前年", 2), ("去年", 1), ("今年", 0)):
        if word in text:
            y = today.year - delta
            return TimeRange(date(y, 1, 1), date(y, 12, 31) if delta else today, word)
    if m := re.search(_NUM + r"\s*年前", text):
        y = today.year - cn_int(m[1])
        return TimeRange(date(y, 1, 1), date(y, 12, 31), m[0])

    # ---- 只說月份：「3 月」「十二月」 ----
    if m := re.search(rf"(?<![個上本這\d]){_NUM}\s*月(?:份)?(?![前底初中])", text):
        mo = cn_int(m[1])
        if 1 <= mo <= 12:
            return TimeRange(*_month_range(_past_month(today, mo), mo), m[0])

    # ---- 模糊 ----
    if m := re.search(r"這幾天|這兩天|前幾天|前兩天|幾天前", text):
        return TimeRange(today - timedelta(days=7), today, m[0], soft=True)
    if m := re.search(r"最近|近期|近來|這陣子|這段時間|前陣子|最近幾天", text):
        return TimeRange(today - timedelta(days=14), today, m[0], soft=True)
    return None
