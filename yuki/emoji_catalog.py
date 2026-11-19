"""表情圖鑑的邏輯（開發規格 6.14 第 1 點）：算使用次數、抽樣對話、合併舊資料、產生檢查頁。
一次性工具 scripts/build_emoji_catalog.py 用它；Yuki 本體不 import。"""
from __future__ import annotations

import html
import re
import sqlite3
from typing import Iterable

CUSTOM_EMOJI = re.compile(r"<(a?):(\w+):(\d+)>")
MAX_SAMPLES = 15
CONTEXT = 3
LINE_CHARS = 120
TOP_N = 30


def count_uses(rows: Iterable[tuple[int, str, str]]) -> dict[str, dict]:
    """rows = (訊息ID, 內容, 時間)。依表情 ID 統計（改過名也算同一張，動圖一樣）。
    回傳 {表情ID: {"uses": 次數, "last_used": 最後時間, "msgs": [訊息ID...]}}；一則訊息用同一張多次算多次。"""
    out: dict[str, dict] = {}
    for mid, content, created in rows:
        for _animated, _name, eid in CUSTOM_EMOJI.findall(content or ""):
            e = out.setdefault(eid, {"uses": 0, "last_used": "", "msgs": []})
            e["uses"] += 1
            if created > e["last_used"]:
                e["last_used"] = created
            if mid not in e["msgs"]:
                e["msgs"].append(mid)
    return out


def usage_rows(db: sqlite3.Connection, guild_id: str, opted_out: set[str] = frozenset()):
    for mid, content, created, author in db.execute(
            "SELECT id, content, created_at, author_id FROM messages WHERE guild_id=? AND content LIKE '%<%:%>%'",
            (guild_id,)):
        if author in opted_out:
            continue
        yield mid, content, created


def pick_messages(msgs: list[int], created: dict[int, str], limit: int = MAX_SAMPLES) -> list[int]:
    """最多 limit 則：新的優先、盡量分散在不同天（每天先取最新的一則，天數不夠再補同一天的）。"""
    by_day: dict[str, list[int]] = {}
    for m in sorted(msgs, reverse=True):
        by_day.setdefault(created[m][:10], []).append(m)
    days = sorted(by_day, reverse=True)
    picked: list[int] = []
    depth = 0
    while len(picked) < limit and any(len(by_day[d]) > depth for d in days):
        for d in days:
            if len(by_day[d]) > depth and len(picked) < limit:
                picked.append(by_day[d][depth])
        depth += 1
    return picked


def _line(content: str, author: str) -> str:
    text = " ".join((content or "").split())
    text = CUSTOM_EMOJI.sub(lambda m: f":{m.group(2)}:", text)
    return f"{author}：{text[:LINE_CHARS]}"


def context_for(db: sqlite3.Connection, message_id: int, opted_out: set[str] = frozenset(),
                n: int = CONTEXT) -> tuple[list[str], str, list[str]] | None:
    """(前 n 則, 那則本身, 後 n 則)；同一個頻道內。optout 的人的訊息不出現；找不到回 None。"""
    row = db.execute("SELECT channel_id, content, author_name, author_id FROM messages WHERE id=?",
                     (message_id,)).fetchone()
    if not row or row[3] in opted_out:
        return None
    ch, content, author, _ = row
    before = db.execute("SELECT content, author_name, author_id FROM messages WHERE channel_id=? AND id<?"
                        " ORDER BY id DESC LIMIT ?", (ch, message_id, n * 3)).fetchall()
    after = db.execute("SELECT content, author_name, author_id FROM messages WHERE channel_id=? AND id>?"
                       " ORDER BY id LIMIT ?", (ch, message_id, n * 3)).fetchall()

    def keep(rows):
        return [_line(c, a) for c, a, u in rows if u not in opted_out][:n]

    return list(reversed(keep(before))), _line(content, author), keep(after)


def build_samples(db: sqlite3.Connection, msgs: list[int], opted_out: set[str] = frozenset(),
                  limit: int = MAX_SAMPLES) -> list[tuple[list[str], str, list[str]]]:
    if not msgs:
        return []
    marks = ",".join("?" * len(msgs))
    created = {i: c for i, c in db.execute(f"SELECT id, created_at FROM messages WHERE id IN ({marks})", msgs)}
    out = []
    for m in pick_messages([m for m in msgs if m in created], created, limit):
        ctx = context_for(db, m, opted_out)
        if ctx:
            out.append(ctx)
    return out


def samples_text(samples: list[tuple[list[str], str, list[str]]]) -> str:
    blocks = []
    for i, (before, hit, after) in enumerate(samples, 1):
        lines = [*before, f">> {hit}", *after]
        blocks.append(f"[{i}]\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def merge_item(new: dict, old: dict | None, *, only_missing: bool = False) -> dict:
    """新抓到的資料合併舊的：`use` 一律保留舊的（人工改過）；only_missing 時已有的 meaning/looks 保留。"""
    item = dict(new)
    if old:
        item["use"] = old.get("use", True)
        if only_missing:
            for k in ("meaning", "looks"):
                if old.get(k):
                    item[k] = old[k]
    item.setdefault("use", True)
    return item


def sort_key(item: dict):
    return (-item.get("uses", 0), item.get("name", ""))


def render_html(items: list[dict]) -> str:
    rows = sorted(items, key=sort_key)
    esc = html.escape
    body = []
    for rank, it in enumerate(rows, 1):
        top = rank <= TOP_N and it.get("uses", 0) > 0
        img = f'<img src="{esc(it["url"])}" width="48" height="48" loading="lazy">' if it.get("url") else ""
        tags = esc(it.get("kind", "")) + (" 動圖" if it.get("animated") else "") + ("" if it.get("use", True) else " 停用")
        body.append(
            f'<tr class="{"top" if top else ""}"><td>{rank}</td><td>{img}</td>'
            f'<td>{esc(it.get("name", ""))}<br><small>{tags}</small></td>'
            f'<td class="n">{it.get("uses", 0)}</td><td>{esc(it.get("last_used", "") or "")}</td>'
            f'<td>{esc(it.get("meaning", "") or "")}</td><td>{esc(it.get("looks", "") or "")}</td></tr>')
    rows_html = "\n".join(body)
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>表情圖鑑檢查</title>
<style>
body{{font-family:sans-serif;margin:1.5em}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:middle}} th{{background:#eee;position:sticky;top:0}}
tr.top{{background:#fff7d6}} td.n{{text-align:right}} small{{color:#666}}
</style></head><body>
<h1>表情圖鑑檢查</h1>
<p>共 {len(rows)} 張，依使用次數排序；黃底是前 {TOP_N} 名（請看這些的「用法」準不準，不想讓她用的告訴我）。</p>
<table><thead><tr><th>#</th><th>圖</th><th>名稱</th><th>次數</th><th>最後使用</th><th>用法 (meaning)</th><th>長相 (looks)</th></tr></thead>
<tbody>
{rows_html}
</tbody></table></body></html>
"""
