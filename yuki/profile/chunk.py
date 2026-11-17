"""把封存區的原始訊息整理成給 LLM 讀的對話段落。

- 清掉自訂表情的 ID、網址只留網域，省 token；
- 同一個人短時間內連發的訊息合併成一行；
- 段落以字數為上限切，夠長之後遇到長時間沒人講話就提早切，盡量不把一段對話切成兩半。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

_EMOJI = re.compile(r"<a?:(\w+):\d+>")
_URL = re.compile(r"https?://([^/\s]+)\S*")
_SPACES = re.compile(r"[ \t]+")


def clean(text: str, max_len: int = 300) -> str:
    text = _EMOJI.sub(r":\1:", text or "")
    text = _URL.sub(r"[連結:\1]", text)
    text = _SPACES.sub(" ", text.replace("\r", "")).strip()
    text = re.sub(r"\n+", " / ", text)
    return text if len(text) <= max_len else text[:max_len] + "…"


@dataclass
class Line:
    msg_id: int            # 合併後最後一則的訊息 ID
    author_id: str
    author_name: str
    at: datetime
    text: str

    def render(self) -> str:
        return f"[{self.at:%H:%M}] {self.author_name}: {self.text}"


@dataclass
class Segment:
    channel_id: str
    channel_name: str
    lines: list[Line] = field(default_factory=list)

    @property
    def last_id(self) -> int:
        return self.lines[-1].msg_id

    @property
    def first_day(self) -> str:
        return f"{self.lines[0].at:%Y-%m-%d}"

    @property
    def last_day(self) -> str:
        return f"{self.lines[-1].at:%Y-%m-%d}"

    def speakers(self) -> dict[str, str]:
        """{author_id: 名字}，依發言順序。"""
        return {l.author_id: l.author_name for l in self.lines}

    def render(self) -> str:
        out, day = [], None
        for line in self.lines:
            d = f"{line.at:%Y-%m-%d}"
            if d != day:
                out.append(f"--- {d} ---")
                day = d
            out.append(line.render())
        return "\n".join(out)

    def chars(self) -> int:
        return sum(len(l.text) + len(l.author_name) + 10 for l in self.lines)


def to_lines(rows: list[tuple], merge_seconds: int = 180) -> list[Line]:
    """rows: (id, author_id, author_name, created_at, content)，依 id 排好。"""
    lines: list[Line] = []
    for msg_id, author_id, author_name, created_at, content in rows:
        text = clean(content)
        if not text:
            continue
        at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        prev = lines[-1] if lines else None
        if prev and prev.author_id == author_id and (at - prev.at).total_seconds() <= merge_seconds \
                and len(prev.text) < 600:
            prev.text += " / " + text
            prev.msg_id, prev.at = msg_id, at
        else:
            lines.append(Line(msg_id, author_id, author_name, at, text))
    return lines


def segments(lines: list[Line], channel_id: str, channel_name: str, *, max_chars: int = 8000,
             gap_hours: float = 3.0) -> list[Segment]:
    out: list[Segment] = []
    cur = Segment(channel_id, channel_name)
    size = 0
    for line in lines:
        n = len(line.text) + len(line.author_name) + 10
        if cur.lines:
            gap = (line.at - cur.lines[-1].at).total_seconds() / 3600
            if size + n > max_chars or (size >= max_chars * 0.4 and gap >= gap_hours):
                out.append(cur)
                cur, size = Segment(channel_id, channel_name), 0
        cur.lines.append(line)
        size += n
    if cur.lines:
        out.append(cur)
    return out
