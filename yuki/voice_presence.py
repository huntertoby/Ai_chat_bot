"""語音頻道的在場狀態：誰在哪個語音頻道、待多久、最近誰進來誰出去，以及今天大家在語音待了多久。

不需要收音：Discord 的 voice state 事件本來就會告訴 bot 誰進出語音頻道（大家在 Discord 介面上也看得到）。

- 即時狀態只放在記憶體；上線時從 Discord 快取重建（關機期間的進出不知道）。
- 進出紀錄存在 data/archive.db 的 voice_sessions 表，用來回答「今天誰在語音」；
  /不要記我 的人不存紀錄。
- 只看公開的語音頻道（跟封存區同一套判斷），私人頻道裡有誰不會被講出去。

與 Discord 無關，方便測試；bot.py 負責把 discord 物件轉成這裡的參數。
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    channel_name TEXT NOT NULL,
    joined_at TEXT NOT NULL,          -- 本地時間 YYYY-MM-DD HH:MM:SS
    left_at TEXT,                     -- NULL = 還在裡面
    partial INTEGER NOT NULL DEFAULT 0 -- 1 = 開始或結束時間不確定（bot 上線前就在、或 bot 關機時還在）
);
CREATE INDEX IF NOT EXISTS idx_voice_guild_time ON voice_sessions(guild_id, joined_at);
"""
_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class Here:
    name: str
    since: datetime
    known_start: bool  # False = bot 上線時他就在了，不知道實際待多久


@dataclass
class VoiceEvent:
    at: datetime
    name: str
    kind: str  # join / leave / move
    channel: str
    to_channel: str = ""


def ago(delta: timedelta) -> str:
    m = int(delta.total_seconds() // 60)
    if m < 1:
        return "剛剛"
    if m < 60:
        return f"{m} 分鐘前"
    return f"{m // 60} 小時前"


def duration(delta: timedelta) -> str:
    m = int(delta.total_seconds() // 60)
    if m < 60:
        return f"{max(m, 1)} 分鐘"
    return f"{m // 60} 小時" + (f" {m % 60} 分" if m % 60 else "")


class VoicePresence:
    def __init__(self, db: sqlite3.Connection, *, max_events: int = 30):
        self.db = db
        self.db.executescript(_SCHEMA)
        # guild → channel_id → (channel_name, {user_id: Here})
        self.rooms: dict[str, dict[str, tuple[str, dict[str, Here]]]] = defaultdict(dict)
        self.events: dict[str, deque[VoiceEvent]] = defaultdict(lambda: deque(maxlen=max_events))

    # ---------- 狀態更新 ----------
    def reset_guild(self, guild_id: str, now: datetime) -> None:
        """上線時：上次關機時還開著的紀錄先關掉（不知道實際何時離開），即時狀態清空等重建。"""
        self.db.execute("UPDATE voice_sessions SET left_at=?, partial=1 WHERE guild_id=? AND left_at IS NULL",
                        (now.strftime(_FMT), guild_id))
        self.db.commit()
        self.rooms[guild_id] = {}

    def present(self, guild_id: str, user_id: str, name: str, channel_id: str, channel_name: str,
                now: datetime, *, record: bool = True) -> None:
        """上線時已經在語音裡的人（不知道什麼時候進來的）。"""
        self._add(guild_id, user_id, name, channel_id, channel_name, now, known_start=False)
        if record:
            self._open(guild_id, user_id, name, channel_id, channel_name, now, partial=True)

    def update(self, guild_id: str, user_id: str, name: str, before: tuple[str, str] | None,
               after: tuple[str, str] | None, now: datetime, *, record: bool = True) -> VoiceEvent | None:
        """before / after：(頻道 ID, 頻道名稱)，None = 不在（或是不公開的頻道）。"""
        if before == after or (before and after and before[0] == after[0]):
            return None  # 靜音、開關鏡頭等，不是進出
        if before:
            self._remove(guild_id, user_id, before[0])
            if record:
                self._close(guild_id, user_id, now)
        if after:
            self._add(guild_id, user_id, name, after[0], after[1], now, known_start=True)
            if record:
                self._open(guild_id, user_id, name, after[0], after[1], now)
        if before and after:
            ev = VoiceEvent(now, name, "move", before[1], after[1])
        elif after:
            ev = VoiceEvent(now, name, "join", after[1])
        else:
            ev = VoiceEvent(now, name, "leave", before[1])
        self.events[guild_id].append(ev)
        return ev

    def forget_user(self, guild_id: str | None, user_id: str) -> int:
        """/不要記我：刪掉這個人的語音紀錄（guild_id 是 None 就刪所有伺服器的）。"""
        if guild_id is None:
            n = self.db.execute("DELETE FROM voice_sessions WHERE user_id=?", (user_id,)).rowcount
        else:
            n = self.db.execute("DELETE FROM voice_sessions WHERE guild_id=? AND user_id=?",
                                (guild_id, user_id)).rowcount
        self.db.commit()
        return n

    def _add(self, gid, uid, name, cid, cname, now, *, known_start):
        people = self.rooms[gid].get(cid, (cname, {}))[1]
        people[uid] = Here(name, now, known_start)
        self.rooms[gid][cid] = (cname, people)

    def _remove(self, gid, uid, cid):
        room = self.rooms[gid].get(cid)
        if room:
            room[1].pop(uid, None)
            if not room[1]:
                del self.rooms[gid][cid]

    def _open(self, gid, uid, name, cid, cname, now, *, partial=False):
        self.db.execute("INSERT INTO voice_sessions (guild_id, user_id, user_name, channel_id, channel_name,"
                        " joined_at, partial) VALUES (?,?,?,?,?,?,?)",
                        (gid, uid, name, cid, cname, now.strftime(_FMT), int(partial)))
        self.db.commit()

    def _close(self, gid, uid, now):
        self.db.execute("UPDATE voice_sessions SET left_at=? WHERE guild_id=? AND user_id=? AND left_at IS NULL",
                        (now.strftime(_FMT), gid, uid))
        self.db.commit()

    # ---------- 給 prompt 的描述 ----------
    def today_totals(self, guild_id: str, now: datetime) -> list[tuple[str, timedelta]]:
        """今天（本地時間 0 點起）每個人在語音待了多久，多到少。"""
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        totals: dict[str, tuple[str, timedelta]] = {}
        for uid, name, joined, left in self.db.execute(
                "SELECT user_id, user_name, joined_at, left_at FROM voice_sessions"
                " WHERE guild_id=? AND (left_at IS NULL OR left_at >= ?)", (guild_id, start.strftime(_FMT))):
            a = max(datetime.strptime(joined, _FMT), start)
            b = datetime.strptime(left, _FMT) if left else now
            if b > a:
                prev = totals.get(uid, (name, timedelta()))[1]
                totals[uid] = (name, prev + (b - a))
        return sorted(totals.values(), key=lambda x: -x[1])

    def describe(self, guild_id: str, now: datetime, *, recent_minutes: int = 30, max_events: int = 8) -> str:
        lines = []
        rooms = self.rooms.get(guild_id, {})
        if rooms:
            for cname, people in rooms.values():
                who = []
                for h in people.values():
                    stay = f"已待 {duration(now - h.since)}" if h.known_start else f"Yuki 上線前就在（{duration(now - h.since)}以上）"
                    who.append(f"{h.name}（{stay}）")
                lines.append(f"- #{cname}：" + "、".join(who))
        else:
            lines.append("- 現在沒有人在語音頻道")
        recent = [e for e in self.events.get(guild_id, ()) if now - e.at <= timedelta(minutes=recent_minutes)]
        if recent:
            lines.append(f"最近 {recent_minutes} 分鐘：")
            for e in recent[-max_events:]:
                what = {"join": f"進了 #{e.channel}", "leave": f"離開 #{e.channel}",
                        "move": f"從 #{e.channel} 換到 #{e.to_channel}"}[e.kind]
                lines.append(f"- {ago(now - e.at)} {e.name} {what}")
        totals = [(n, d) for n, d in self.today_totals(guild_id, now) if d >= timedelta(minutes=5)]
        if totals:
            lines.append("今天在語音待過：" + "、".join(f"{n} {duration(d)}" for n, d in totals[:10]))
        return "\n".join(lines)
