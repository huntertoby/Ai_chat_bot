"""表情圖鑑的邏輯（開發規格 6.14）：
- 建圖鑑（scripts/build_emoji_catalog.py 用）：算使用次數、抽樣對話、合併舊資料、產生檢查頁。
- 使用圖鑑（Yuki 本體用，C1-2）：讀 data/emoji/*.json、產生 prompt 用的清單、把回覆裡的 `:名稱:`／`[貼圖:名稱]` 換成 Discord 格式。

`reviewed`：圖鑑檔本身維持最上層是 list（C1-1 的格式不變），是否看過放在旁邊的 `<伺服器ID>.meta.json`
（`{"reviewed": false}`）。沒有 meta 檔＝沒看過；**只有 reviewed 是 true 的圖鑑 Yuki 才會用**。
（載入時也接受 `{"reviewed": ..., "items": [...]}` 這種單一檔案的寫法。）"""
from __future__ import annotations

import html
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

EMOJI_DIR = Path(__file__).resolve().parent.parent / "data" / "emoji"
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


REVISE_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "meaning": {"type": "string"}},
        "required": ["i", "meaning"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False,
}


def snippets(samples: list[tuple[list[str], str, list[str]]], chars: int = 60) -> list[str]:
    """每個樣本取前一則加那則本身，縮短，給修訂用。"""
    out = []
    for before, hit, _after in samples:
        line = (before[-1][:chars] + " / " if before else "") + hit[:chars * 2]
        out.append(line)
    return out


def revise_prompt(entries: list[dict]) -> str:
    """entries 每個有 i、looks、meaning、snippets；不放名稱，只有編號。"""
    blocks = []
    for e in entries:
        sn = "\n".join(f"  - {s}" for s in e.get("snippets", []))
        blocks.append(f"第 {e['i']} 號\n外觀：{e.get('looks', '')}\n初步說明：{e.get('meaning', '')}\n片段：\n{sn}")
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


# ====================== 使用圖鑑（C1-2）======================
META_SUFFIX = ".meta.json"
DESC_CHARS = 60
PROMPT_EMOJI = 22      # reply prompt 放幾張表情（最常用的）
PROMPT_STICKERS = 4    # 另外最多放幾張貼圖（貼圖沒有使用次數，用 looks/meaning 判斷）
REACT_N = 10           # 插話判斷放幾張

_SHORT = re.compile(r"(?<![\w:]):(\w{1,40}):(?![\w:])")
_STICKER_TAG = re.compile(r"[\[［]\s*貼圖\s*[:：]\s*([^\]］\n]{1,40}?)\s*[\]］]")
_ANY = re.compile(r"<(a?):(\w+):(\d+)>|" + _SHORT.pattern + "|" + _STICKER_TAG.pattern)
_HEXISH = re.compile(r"[0-9a-fA-F]{12,}")
_PLAIN = re.compile(r"[A-Za-z0-9_一-鿿]+")


def meta_path(path: Path) -> Path:
    return path.with_name(path.stem + META_SUFFIX)


def read_reviewed(path: Path) -> bool:
    try:
        return json.loads(meta_path(path).read_text(encoding="utf-8")).get("reviewed") is True
    except (OSError, ValueError, AttributeError):
        return False


def ensure_meta(path: Path) -> None:
    """建圖鑑時呼叫：沒有 meta 檔就建一個 reviewed=false；已經有的（含人工改成 true 的）不動。"""
    mp = meta_path(path)
    if not mp.exists():
        mp.write_text(json.dumps({"reviewed": False}, indent=1), encoding="utf-8")


def hard_to_say(name: str) -> bool:
    """名稱太難念：非中英數底線、太長、一長串十六進位或數字（隨機字串、相機檔名之類）。"""
    return (not _PLAIN.fullmatch(name) or len(name) > 20 or bool(_HEXISH.search(name))
            or bool(re.search(r"\d{6,}", name)))


def describe(item: dict) -> str:
    text = " ".join((item.get("meaning") or item.get("looks") or "").split())
    return text[:DESC_CHARS]


@dataclass
class Entry:
    id: str
    name: str
    kind: str          # emoji / sticker
    animated: bool
    uses: int
    desc: str
    alias: str = ""    # 名稱難念時給模型看的別名

    @property
    def label(self) -> str:
        return self.alias or self.name

    def render(self) -> str:
        return f"<{'a' if self.animated else ''}:{self.name}:{self.id}>"


class EmojiSet:
    """某一個伺服器自己的、可以用的表情與貼圖（use=true 且有 meaning 或 looks）。"""

    def __init__(self, guild_id: str, items: Iterable[dict]):
        self.guild_id = str(guild_id)
        entries = [Entry(str(i["id"]), str(i["name"]), i.get("kind", "emoji"), bool(i.get("animated")),
                         int(i.get("uses", 0) or 0), describe(i))
                   for i in items if i.get("use", True) and describe(i) and i.get("id") and i.get("name")]
        entries.sort(key=lambda e: (-e.uses, e.name))
        taken = {e.name.lower() for e in entries}
        counters = {"emoji": 0, "sticker": 0}
        seen: set[tuple[str, str]] = set()
        for e in entries:
            dup = (e.kind, e.name.lower()) in seen  # 同名（貼圖常見）：後面的給別名，不然分不出來
            seen.add((e.kind, e.name.lower()))
            if hard_to_say(e.name) or dup:
                prefix = "e" if e.kind == "emoji" else "s"
                while True:
                    counters[e.kind] += 1
                    cand = f"{prefix}{counters[e.kind]}"
                    if cand not in taken:
                        break
                e.alias = cand
                taken.add(cand)
        self.entries = entries
        self._index()

    def _index(self) -> None:
        self._by_key: dict[tuple[str, str], Entry] = {}
        self._by_id: dict[str, Entry] = {}
        for e in self.entries:
            for key in {e.name.lower(), e.alias.lower()} - {""}:
                self._by_key[(e.kind, key)] = e
            self._by_id[e.id] = e

    def __len__(self) -> int:
        return len(self.entries)

    def _top(self, kind: str, n: int) -> list[Entry]:
        return [e for e in self.entries if e.kind == kind][:n]

    def lines(self, entries: list[Entry]) -> str:
        return "\n".join(f":{e.label}: {e.desc}" if e.kind == "emoji" else f"[貼圖:{e.label}] {e.desc}"
                         for e in entries)

    def prompt_block(self) -> str:
        """回覆 system 用的清單（空字串＝沒有可用的）：最常用的表情，加上最多幾張貼圖。"""
        return self.lines(self._top("emoji", PROMPT_EMOJI) + self._top("sticker", PROMPT_STICKERS))

    def react_block(self, n: int = REACT_N) -> str:
        """插話判斷用：最常用的 n 張表情，一行一張。"""
        return "\n".join(f":{e.label}: {e.desc}" for e in self._top("emoji", n))

    def react_name(self, emoji: str) -> tuple[bool, str]:
        """模型在 emoji 欄寫的東西 → (是不是群組表情的名稱, 真正的名稱)。
        名稱（含別名、`:名稱:`、`<:名稱:ID>`）找得到回 (True, 真名)；看起來是名稱但圖鑑裡沒有回 (True, "")；
        其他（unicode 表情）回 (False, 原文)。"""
        e = (emoji or "").strip()
        m = re.fullmatch(r"<a?:(\w+):\d+>", e)
        name = (m.group(1) if m else e.strip(":")).strip()
        hit = self._by_key.get(("emoji", name.lower()))
        if hit:
            return True, hit.name
        if m or re.search(r"[A-Za-z0-9_]", e) or len(e) > 8:
            return True, ""
        return False, e

    def process(self, text: str) -> tuple[str, str | None]:
        return process_reply(text, self)

    def sync(self, emojis: Iterable, stickers: Iterable = ()) -> dict:
        """跟伺服器現況比對：圖鑑有、伺服器已刪的不再使用；改名的換成新名字；伺服器新增的只記下來（這次不補建）。
        emojis／stickers 是有 id、name 的物件（guild.emojis、guild.stickers）。"""
        live = {str(o.id): o for o in [*emojis, *stickers]}
        report = {"removed": [], "renamed": [], "new": [str(o.name) for i, o in live.items() if i not in self._by_id]}
        keep = []
        for e in self.entries:
            o = live.get(e.id)
            if o is None:
                report["removed"].append(e.name)
                continue
            if str(o.name) != e.name:
                report["renamed"].append((e.name, str(o.name)))
                e.name = str(o.name)
            keep.append(e)
        self.entries = keep
        self._index()
        return report


class EmojiPacing:
    """「不要每句都用」的程式端保證：同一個頻道用過表情或貼圖之後，接下來 `skip` 則回覆不用
    （那幾則的 prompt 也不放清單）。模型自己的傾向再高，最終比例也不會超過 1/(skip+1)。"""

    def __init__(self, skip: int = 2):
        self.skip = skip
        self._since: dict = {}

    def allow(self, channel) -> bool:
        return self._since.get(channel, self.skip) >= self.skip

    def record(self, channel, used: bool) -> None:
        self._since[channel] = 0 if used else self._since.get(channel, self.skip) + 1


def process_reply(text: str, eset: "EmojiSet | None", *, allow: bool = True) -> tuple[str, str | None]:
    """回覆後處理：`:名稱:`→`<:名稱:ID>`；不在圖鑑裡的 `:xxx:`、`[貼圖:xxx]` 刪掉；
    一則最多 1 個表情或 1 張貼圖（照出現順序，第一個有效的留下）。回傳 (文字, 貼圖 ID 或 None)。
    eset=None（沒有圖鑑、沒 reviewed、或這個伺服器不開放）：`:xxx:`、`[貼圖:…]` 一律清掉，其他不動。
    allow=False（這則不該用，見 EmojiPacing）：圖鑑裡的也一律刪掉。"""
    sticker: str | None = None
    used = not allow  # 不允許＝當作已經用掉名額，後面的有效表情全部刪
    changed = False

    def sub(m: re.Match) -> str:
        nonlocal sticker, used, changed
        raw_id, short, tag = m.group(3), m.group(4), m.group(5)
        if raw_id is not None:  # 模型自己寫（或學對話紀錄）的 <:名稱:ID>
            if eset is None:
                return m.group(0)
            e = eset._by_id.get(raw_id)
            kind, entry = "emoji", (e if e and e.kind == "emoji" else None)
        elif short is not None:
            entry = eset._by_key.get(("emoji", short.lower())) if eset else None
            kind = "emoji"
        else:
            entry = eset._by_key.get(("sticker", tag.strip().lower())) if eset else None
            kind = "sticker"
        changed = True
        if entry is None or used:
            return ""
        used = True
        if kind == "sticker":
            sticker = entry.id
            return ""
        return entry.render()

    out = _ANY.sub(sub, text or "")
    if changed:
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r" +\n", "\n", out).strip()
    return out, sticker


def _items_of(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("items", []), data.get("reviewed") is True
    return data, read_reviewed(path)


def load_set(path: Path, *, require_reviewed: bool = True) -> EmojiSet | None:
    """讀一份圖鑑；沒 reviewed（且 require_reviewed）、檔案壞掉、沒有可用的都回 None。評估腳本用 require_reviewed=False 直接指到檔案。"""
    try:
        items, reviewed = _items_of(path)
    except (OSError, ValueError):
        log.warning("表情圖鑑讀不了：%s", path.name, exc_info=True)
        return None
    if require_reviewed and not reviewed:
        return None
    eset = EmojiSet(path.stem, items)
    return eset if len(eset) else None


def load_all(directory: Path = EMOJI_DIR) -> dict[str, EmojiSet]:
    """啟動時讀 `*.json`（不含 .meta.json）；只留 reviewed 的。回傳 {伺服器ID: EmojiSet}。"""
    out: dict[str, EmojiSet] = {}
    for p in sorted(directory.glob("*.json")):
        if p.name.endswith(META_SUFFIX):
            continue
        eset = load_set(p)
        if eset is None:
            log.info("表情圖鑑 %s 沒有 reviewed 或沒有可用的項目，不使用", p.name)
            continue
        out[p.stem] = eset
        log.info("表情圖鑑 %s：可用 %d 張", p.name, len(eset))
    return out
