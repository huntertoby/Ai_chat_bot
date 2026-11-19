"""用 LM Studio 產生「虛構群組」的 5 年每日記憶，拿來壓力測試與評估檢索。

資料寫到獨立的 data/synthetic/ 與 data/synthetic.db，guild_id 固定為 "synthetic"，不會碰到真的記憶庫。

產生方式（為了讓 5 年的資料前後一致，而不是 1 萬句互不相干的句子）：
  1. personas：10 個虛構成員的人設（身份、興趣、口味、朋友關係）
  2. arcs：每人每年 6~10 個生活主線（換工作、搬家、交往、改變口味…），每月記憶都依照主線發展
  3. days：每人每月一次呼叫，產生當月每一天的記憶（含前一個月的結尾當上下文）
  4. import：寫進 data/synthetic.db 的 memories 表（created_at = 記憶的日期；scripts.eval_agentic 會轉成暫存封存區）
  5. eval：抽樣記憶請 LLM 出題，產生 eval/synthetic_cases.jsonl 給 scripts.eval_agentic 用

每個階段都會存檔，中斷後重跑會從斷點繼續。

    python -m scripts.gen_synthetic all                  # 全部跑完（約數小時，看 GPU）
    python -m scripts.gen_synthetic days --users 2 --months 2   # 先小量試跑
    python -m scripts.gen_synthetic import
    python -m scripts.gen_synthetic eval --cases 200
"""
from __future__ import annotations

import argparse
import asyncio
import calendar
import json
import logging
import random
import sqlite3
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

from yuki.config import ROOT, load_settings
from yuki.llm import LLMError, LMStudioClient

log = logging.getLogger("gen_synthetic")

OUT = ROOT / "data" / "synthetic"
PERSONAS = OUT / "personas.json"
ARCS = OUT / "arcs.json"
DAYS = OUT / "memories.jsonl"
DB = ROOT / "data" / "synthetic.db"
CASES = ROOT / "eval" / "synthetic_cases.jsonl"
GUILD = "synthetic"

END = date(2026, 10, 29)
START = date(END.year - 5, END.month, END.day)


def months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def days_in(y: int, m: int) -> list[int]:
    last = calendar.monthrange(y, m)[1]
    return [d for d in range(1, last + 1) if START <= date(y, m, d) <= END]


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


STR = {"type": "string"}

# ================= 1. personas =================
PERSONA_SCHEMA = obj({"members": {"type": "array", "items": obj({
    "name": STR, "age": {"type": "integer"}, "identity": STR, "city": STR,
    "personality": STR, "hobbies": {"type": "array", "items": STR},
    "food": STR, "games": {"type": "array", "items": STR}, "friends": STR,
})}})


async def gen_personas(llm: LMStudioClient, n: int) -> list[dict]:
    if PERSONAS.exists():
        return json.loads(PERSONAS.read_text(encoding="utf-8"))
    prompt = (f"幫一個台灣的 Discord 朋友群組設計 {n} 個虛構成員（{START.year} 年時的狀態）。"
              "要多元：學生、上班族、研究生、自由業都有，年齡 18~35，興趣、口味、玩的遊戲都要不同，"
              "name 用 Discord 暱稱（中文或英文都可以，不能重複，至少 2 個字）。"
              "friends 寫他跟群組裡哪幾位比較熟、什麼關係。全部用繁體中文。")
    data = await llm.chat_json([{"role": "user", "content": prompt}], PERSONA_SCHEMA,
                               name="personas", temperature=0.9, max_tokens=6000)
    members = (data or {}).get("members", [])[:n]
    if len(members) < n:
        raise SystemExit(f"人設只產生了 {len(members)} 個，請重跑")
    for i, p in enumerate(members, 1):
        p["id"] = f"synth-{i:02d}"
    OUT.mkdir(parents=True, exist_ok=True)
    PERSONAS.write_text(json.dumps(members, ensure_ascii=False, indent=2), encoding="utf-8")
    return members


def persona_text(p: dict) -> str:
    return (f"{p['name']}（{p['age']} 歲，{p['identity']}，住{p['city']}）個性：{p['personality']}；"
            f"興趣：{'、'.join(p['hobbies'])}；口味：{p['food']}；遊戲：{'、'.join(p['games'])}；"
            f"朋友：{p['friends']}")


# ================= 2. arcs =================
ARC_SCHEMA = obj({"arcs": {"type": "array", "items": obj({"month": {"type": "integer"}, "event": STR})}})


async def gen_arcs(llm, personas, sem) -> dict:
    arcs = json.loads(ARCS.read_text(encoding="utf-8")) if ARCS.exists() else {}
    roster = "\n".join(f"- {persona_text(p)}" for p in personas)

    async def one_user(p):
        prev = ""
        for year in range(START.year, END.year + 1):
            key = f"{p['id']}:{year}"
            if key not in arcs:
                prompt = (f"群組成員：\n{roster}\n\n請為 {p['name']} 規劃 {year} 年的 6~10 個生活主線事件，"
                          f"要延續前幾年的發展：{prev or '（第一年）'}\n"
                          "包含：學業/工作變化、搬家、感情、健康、旅行、新興趣、口味或喜好改變（例如不再喝某飲料）、"
                          "跟其他成員一起做的事。month 是 1~12，event 一句話，用繁體中文。")
                try:
                    async with sem:
                        data = await llm.chat_json([{"role": "user", "content": prompt}], ARC_SCHEMA,
                                                   name="arcs", temperature=0.9, max_tokens=2000)
                except LLMError as e:
                    log.warning("arcs %s %d 失敗（之後重跑會補）：%s", p["name"], year, e)
                    continue
                arcs[key] = sorted((data or {}).get("arcs", []), key=lambda a: a.get("month", 0))
                ARCS.write_text(json.dumps(arcs, ensure_ascii=False, indent=2), encoding="utf-8")
                log.info("arcs %s %d：%d 個事件", p["name"], year, len(arcs[key]))
            prev = "；".join(f"{year}/{a['month']} {a['event']}" for a in arcs[key])[-600:]

    await asyncio.gather(*(one_user(p) for p in personas))
    return arcs


# ================= 3. 每日記憶 =================
DAY_SCHEMA = obj({"memories": {"type": "array", "items": obj({"day": {"type": "integer"}, "text": STR})}})

DAY_SYSTEM = """你在替 Discord 聊天機器人產生「測試用的長期記憶」。每筆記憶是機器人從群組聊天中記下的一個事實。
規則：
- 每一天正好一筆，主詞必須是成員的名字，第三人稱，一句話，繁體中文，15~60 字。
- 約六成是當天的具體小事（吃了什麼、玩了什麼、去哪裡、遇到什麼），帶具體名稱（店名、遊戲名、地名、數字）。
- 約四成是比較長期的事實：喜好、計畫、狀態、跟其他成員的互動。
- 必須符合人設與本年度主線；主線事件發生的那個月要有對應記憶，喜好改變後之後的記憶要一致。
- 不要每天重複同樣的事，內容要多樣。"""


def load_done() -> dict[tuple[str, int, int], int]:
    done: dict = {}
    if DAYS.exists():
        for line in DAYS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            y, m, _ = map(int, r["date"].split("-"))
            done[(r["user_id"], y, m)] = done.get((r["user_id"], y, m), 0) + 1
    return done


async def gen_days(llm, personas, arcs, sem, month_list, stats) -> None:
    done = load_done()
    lock = asyncio.Lock()
    by_name = {p["name"]: p for p in personas}

    async def one_user(p):
        tail: list[str] = []
        for y, m in month_list:
            days = days_in(y, m)
            if not days:
                continue
            if (p["id"], y, m) in done:
                continue
            arc = "；".join(f"{a['month']}月 {a['event']}" for a in arcs.get(f"{p['id']}:{y}", []))
            others = "、".join(n for n in by_name if n != p["name"])
            prompt = (f"成員：{persona_text(p)}\n其他成員：{others}\n{y} 年主線：{arc}\n"
                      f"上個月最後幾筆記憶：{' / '.join(tail[-4:]) or '（無）'}\n\n"
                      f"請產生 {p['name']} 在 {y} 年 {m} 月的記憶，day 從 {days[0]} 到 {days[-1]}，每天一筆。")
            msgs = [{"role": "system", "content": DAY_SYSTEM}, {"role": "user", "content": prompt}]
            rows = []
            for attempt in range(2):
                try:
                    async with sem:
                        t = time.time()  # 只算模型實際花的時間，不含排隊等其他人
                        data = await llm.chat_json(msgs, DAY_SCHEMA, name="memories", temperature=0.85,
                                                   max_tokens=len(days) * 90 + 300)
                        elapsed = time.time() - t
                except LLMError as e:
                    log.warning("%s %d-%02d 失敗：%s", p["name"], y, m, e)
                    continue
                seen, rows = set(), []
                for it in (data or {}).get("memories", []):
                    d, text = it.get("day"), str(it.get("text", "")).strip()
                    if d in days and d not in seen and text:
                        if p["name"] not in text:
                            text = f"{p['name']} {text}"
                        seen.add(d)
                        rows.append({"user_id": p["id"], "name": p["name"],
                                     "date": f"{y}-{m:02d}-{d:02d}", "text": text})
                stats["calls"] += 1
                stats["secs"] += elapsed
                if len(rows) >= len(days) * 0.8:
                    break
                log.warning("%s %d-%02d 只拿到 %d/%d 天，重試", p["name"], y, m, len(rows), len(days))
            if not rows:
                continue
            async with lock:
                with DAYS.open("a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats["rows"] += len(rows)
            tail = [r["text"] for r in rows[-4:]]
            log.info("%s %d-%02d：%d/%d 天（累計 %d 筆，平均 %.1fs/次）", p["name"], y, m, len(rows),
                     len(days), stats["rows"], stats["secs"] / max(stats["calls"], 1))

    await asyncio.gather(*(one_user(p) for p in personas))


# ================= 4. import =================
def synth_db() -> sqlite3.Connection:
    """合成記憶的資料表（欄位跟舊版記憶庫一樣；embedding 已經不用，留空）。"""
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB))
    db.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, guild_id TEXT NOT NULL, subject_id TEXT,"
               " subject_name TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, author_id TEXT, created_at TEXT NOT NULL,"
               " updated_at TEXT NOT NULL, embedding BLOB NOT NULL)")
    return db


async def do_import(settings) -> None:
    rows = [json.loads(l) for l in DAYS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["date"])
    db = synth_db()
    existing = {(sid, ts[:10], text) for sid, ts, text in
                db.execute("SELECT subject_id, created_at, text FROM memories")}
    todo = [r for r in rows if (r["user_id"], r["date"], r["text"]) not in existing]
    t = time.time()
    for i in range(0, len(todo), 256):
        batch = todo[i:i + 256]
        db.executemany("INSERT INTO memories (id, guild_id, subject_id, subject_name, text, author_id, created_at,"
                       " updated_at, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
                       [(uuid.uuid4().hex[:8], GUILD, r["user_id"], r["name"], r["text"], r["user_id"],
                         f"{r['date']} 21:00:00", f"{r['date']} 21:00:00", b"") for r in batch])
        db.commit()
        print(f"\r匯入 {min(i + 256, len(todo))}/{len(todo)}", end="", flush=True)
    print(f"\n完成：{DB}（新增 {len(todo)} 筆，總共 {len(existing) + len(todo)} 筆，{time.time() - t:.0f}s）")


# ================= 5. eval cases =================
CASE_SCHEMA = obj({"question": STR})


async def gen_cases(llm, personas, sem, n: int, seed: int) -> None:
    mems = [SimpleNamespace(id=i, subject_id=sid, text=text, created_at=ts) for i, sid, text, ts in synth_db().execute(
        "SELECT id, subject_id, text, created_at FROM memories WHERE guild_id=? ORDER BY created_at", (GUILD,))]
    rng = random.Random(seed)
    sample = rng.sample(mems, min(n, len(mems)))
    members = [{"id": p["id"], "name": p["name"]} for p in personas]
    done = set()
    if CASES.exists():
        done = {json.loads(l)["expected"][0] for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()}

    async def one(mem):
        if mem.id in done:
            return
        asker = rng.choice([p for p in personas if p["id"] != mem.subject_id])
        # 問的時間點：隔天、幾天後、幾週後、幾個月後都有，讓「昨天」「上個月」這類說法跟日期對得上
        happened = date.fromisoformat(mem.created_at[:10])
        lag = rng.choice([1, 1, 1, 2, 3, 5, 6, 9, 14, 20, 30, 45, 60, 120, 300])
        asked = happened + timedelta(days=lag)
        prompt = (f"今天是 {asked.isoformat()}（星期{'一二三四五六日'[asked.weekday()]}）。\n"
                  f"記憶：「{mem.text}」（發生在 {happened.isoformat()}，{lag} 天前）\n"
                  f"假設 {asker['name']} 今天在群組裡問聊天機器人一個問題，而這筆記憶就是回答需要的資料。"
                  "請寫出這個口語問題：要提到被問的人的名字，不要直接照抄記憶的用字，不要包含答案。"
                  "如果自然的話可以用時間說法（昨天、上週、上個月、X 月……），但必須跟今天的日期一致。")
        try:
            async with sem:
                data = await llm.chat_json([{"role": "user", "content": prompt}], CASE_SCHEMA,
                                           name="case", temperature=0.7, max_tokens=200)
        except LLMError as e:
            log.warning("出題失敗：%s", e)
            return
        q = (data or {}).get("question", "").strip()
        if q:
            case = {"guild_id": GUILD, "author": {"id": asker["id"], "name": asker["name"]},
                    "members": members, "query": q, "expected": [mem.id], "memory": mem.text,
                    "now": asked.isoformat()}
            with CASES.open("a", encoding="utf-8") as f:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    CASES.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*(one(m) for m in sample))
    print(f"評估題目：{CASES}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["personas", "arcs", "days", "import", "eval", "all"])
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--users", type=int, default=None, help="只跑前 N 個成員（試跑用）")
    ap.add_argument("--months", type=int, default=None, help="只跑前 N 個月（試跑用）")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="同時送出的請求數（LM Studio 的 MTP 模型只能用 1；vLLM 可用到 MAX_SEQS）")
    ap.add_argument("--vllm", nargs="?", const="http://localhost:8001/v1", default=None, metavar="URL",
                    help="改用 F:/WSL 的 vLLM 產生（預設 http://localhost:8001/v1，模型名 swift）")
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    s = load_settings()
    if args.vllm:
        # vLLM 跑 Qwen3 不認 reasoning_effort，要用 chat template 參數關掉思考
        llm = LMStudioClient(args.vllm, "swift", timeout=600, max_concurrency=args.concurrency,
                             extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    else:
        llm = LMStudioClient(s.lmstudio_base_url, s.llm_model, timeout=600,
                             json_reasoning_effort="none",  # 產生資料不需要推理，速度差很多
                             max_concurrency=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"calls": 0, "secs": 0.0, "rows": 0}
    try:
        personas = arcs = None
        if args.stage in ("personas", "arcs", "days", "eval", "all"):
            personas = await gen_personas(llm, args.members)
            log.info("成員：%s", "、".join(p["name"] for p in personas))
        run_on = personas[: args.users] if personas and args.users else personas
        if args.stage in ("arcs", "days", "all"):
            arcs = await gen_arcs(llm, run_on, sem)
        if args.stage in ("days", "all"):
            month_list = months(START, END)[: args.months] if args.months else months(START, END)
            await gen_days(llm, run_on, arcs, sem, month_list, stats)
        if args.stage in ("import", "all"):
            await do_import(s)
        if args.stage in ("eval", "all"):
            await gen_cases(llm, personas, sem, args.cases, args.seed)
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
