"""D-38 第 2 點：文字回話的總等待（訊息送出 → 她的回話出現），兩段式插話判斷（CHIME_THINK_ON_MENTION）關 vs 開。
題目、追蹤紀錄、結果都在 data/（不進 repo）。

    python -m scripts.eval_text_wait normal --off d38_2_c66route_off_1 ... --on d38_2_c66route_on_1 ...
    python -m scripts.eval_text_wait busy --n 10

normal（不呼叫模型）：總等待 = 插話去抖動（CHIME_DEBOUNCE_SEC）+ 插話判斷 + 回覆 + 挑表情，各段用實測分布的所有組合算
中位數、P90、最大、超過 10 秒的比例。
- 插話判斷：eval_chime 結果檔（data/eval_b/results/<名稱>.json）裡有問模型的題（path fast／think）的 Decision.ms；
  另外只算「判斷要回」的題（使用者只有在她回話時才感覺得到等待）。
- 回覆：正式環境追蹤紀錄（data/logs/ai/*.jsonl，kind=reply）的 latency_ms.total（含工具、R3 檢查重生）。
- 挑表情：D-37 評估（data/eval_c1/results/d37_pick_final_*.json）的 pick_ms；照 bot 的規則，回話＋挑表情以
  TEXT_REPLY_BUDGET_SEC 為上限，剩不到 PICK_MIN_SEC 就不挑。
- 被直接叫的訊息不走去抖動和插話判斷：只有回覆 + 挑表情，另外列。

busy（呼叫模型，LM Studio 只給評估用）：她正在回上一句時又有人講話（主伺服器注意模式 `_attention_gate` 的情況）。
每題：背景開始回上一句（brain.reply + 挑表情）的同時，新的一句進來 → 去抖動 → LLM 忙就每 0.5 秒檢查一次
（跟 scheduler.wait_until 一樣，最多 ATTENTION_GATE_WAIT_SEC 秒，超過就放棄）→ 插話判斷（關／開）→ 回這一句 + 挑表情。
另外量「思考判斷進行中被直接叫」：判斷（開）開始 0.5 秒後有人叫她，這則回覆要多等多久。
題目：66 題、D-32、D-34 三組裡最新一句「提到她」（chime.mentions_bot）、不算直接叫她、預期要回的題。
測試伺服器的插話（`_chime`）遇到 LLM 忙是直接跳過（chime_gate 的 llm_busy），不會等，所以忙時只量注意模式這條。
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import itertools
import json
import time
from dataclasses import replace

from yuki import chime, emoji_catalog as cat, emoji_pick
from yuki.bot import ATTENTION_GATE_WAIT_SEC
from yuki.brain import Brain
from yuki.chat import Person
from yuki.config import ROOT, load_settings
from yuki.scheduler import decide_options
from yuki.wake import wake_pattern

from ._eval_b_common import BOT_NAME, EVAL_DIR, RESULT_DIR, load_jsonl, make_llm, pct, to_line
from .eval_chime import route_called

BUSY_FILES = ("chime_cases.jsonl", "chime_cases_d32.jsonl", "attention_cases_d34.jsonl")


def stats(ms: list[float], limit_ms: float = 10000) -> dict:
    if not ms:
        return {"n": 0}
    xs = sorted(ms)
    return {"n": len(xs), "median_s": round(pct(xs, 0.5) / 1000, 2), "p90_s": round(pct(xs, 0.9) / 1000, 2),
            "max_s": round(xs[-1] / 1000, 2), "over_10s": round(sum(x > limit_ms for x in xs) / len(xs), 3)}


def with_pick(reply_ms: float, pick_ms: float) -> float:
    """bot._pick_emoji：挑表情只用 10 秒裡剩下的時間；剩不到 PICK_MIN_SEC 不挑。"""
    left = emoji_pick.TEXT_REPLY_BUDGET_SEC * 1000 - reply_ms
    if left < emoji_pick.PICK_MIN_SEC * 1000:
        return reply_ms
    return reply_ms + min(pick_ms, left)


# ---------------------------------------------------------------- normal
def trace_replies() -> list[float]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "data" / "logs" / "ai" / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("kind") == "reply" and (r.get("latency_ms") or {}).get("total"):
                out.append(r["latency_ms"]["total"])
    return out


def pick_samples() -> list[float]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "data" / "eval_c1" / "results" / "d37_pick_final_*.json"))):
        out += [r["pick_ms"] for r in json.load(open(f, encoding="utf-8"))["rows"] if r.get("pick_ms")]
    return out


def decide_samples(names: list[str]) -> tuple[list[dict], list[float], list[float]]:
    rows = []
    for n in names:
        res = json.loads((RESULT_DIR / f"{n}.json").read_text(encoding="utf-8"))
        rows += [r for r in res["chime"]["rows"] if r["decision"].get("path") in ("fast", "think")]
    return rows, [r["decision"]["ms"] for r in rows], [r["decision"]["ms"] for r in rows if r["pred"]]


def combine(decide: list[float], replies: list[float], picks: list[float], debounce_ms: float) -> list[float]:
    rp = [with_pick(r, p) for r, p in itertools.product(replies, picks)]
    return [debounce_ms + d + x for d, x in itertools.product(decide, rp)]


def normal(args, s) -> dict:
    replies, picks = trace_replies(), pick_samples()
    debounce = s.chime_debounce_sec * 1000
    out = {"debounce_s": s.chime_debounce_sec, "reply_trace": stats(replies), "pick": stats(picks),
           "called": stats([with_pick(r, p) for r, p in itertools.product(replies, picks)])}
    for mode, names in (("off", args.off), ("on", args.on)):
        rows, all_ms, yes_ms = decide_samples(names)
        think = [r["decision"]["ms"] for r in rows if r["decision"]["path"] == "think"]
        out[mode] = {"files": names, "decide": stats(all_ms), "decide_think": stats(think),
                     "decide_when_reply": stats(yes_ms), "think_share": round(len(think) / max(len(rows), 1), 3),
                     "total_all": stats(combine(all_ms, replies, picks, debounce)),
                     "total_when_reply": stats(combine(yes_ms, replies, picks, debounce))}
    return out


# ---------------------------------------------------------------- busy
def busy_cases(s, n: int) -> list[dict]:
    wake = wake_pattern(s.wake_words)
    names = [BOT_NAME, *s.wake_words]
    picked = []
    for f in BUSY_FILES:
        for c in load_jsonl(EVAL_DIR / f):
            recent = [to_line(m, images=False) for m in c["recent"]]
            latest = to_line(c["latest"], images=False)
            if (not c.get("expect_participate") or route_called(c["latest"], s, wake)
                    or not chime.mentions_bot(latest, recent[-s.chime_context_lines:], names)):
                continue
            prev = max((i for i, l in enumerate(recent) if not l.is_bot), default=None)
            if prev is None:
                continue
            picked.append({"id": c["id"], "file": f, "recent": recent, "latest": latest, "prev": prev})
    step = max(len(picked) // n, 1)
    return picked[::step][:n]


async def reply_and_pick(brain: Brain, eset, lines, author, text) -> tuple[str, float, float]:
    brain.history.clear()
    for l in lines:
        brain.history["busy"].append(l)
    started = time.monotonic()
    try:
        reply = await brain.reply("eval", "busy", author, text)
    except Exception as e:  # noqa: BLE001
        print(f"  回覆失敗：{e!r}", flush=True)
        reply = ""
    reply_ms = (time.monotonic() - started) * 1000
    if eset is not None and reply:
        await emoji_pick.pick(brain.llm, eset, text, reply, bot_name=BOT_NAME,
                              timeout=emoji_pick.remaining(started))
    return reply, reply_ms, (time.monotonic() - started) * 1000


async def busy(args, s) -> dict:
    llm = make_llm(s)
    brain = Brain(replace(s, web_enabled=True), llm)
    eset = cat.load_set(ROOT / "data" / "emoji" / f"{s.profile_fallback_guild}.json", require_reviewed=False)
    cases = busy_cases(s, args.n)
    print(f"忙時題目 {len(cases)} 題；表情圖鑑 {'有' if eset else '沒有'}", flush=True)
    rows = []
    try:
        for mode in ("off", "on"):
            opts = {**decide_options(s), "think_on_mention": mode == "on"}
            for i, c in enumerate(cases, 1):
                prev = c["recent"][c["prev"]]
                lock = asyncio.Lock()  # 跟 bot._replying 一樣：背景在回上一句（含挑表情）就算 LLM 忙
                async def background():
                    async with lock:
                        return await reply_and_pick(brain, eset, c["recent"][:c["prev"] + 1], prev.author, prev.text)
                t0 = time.monotonic()
                bg = asyncio.create_task(background())
                await asyncio.sleep(0)  # 讓背景先拿到鎖
                await asyncio.sleep(s.chime_debounce_sec)
                waited_from = time.monotonic()
                gave_up = False
                while lock.locked() or chime.llm_is_busy(llm):
                    if time.monotonic() - waited_from >= ATTENTION_GATE_WAIT_SEC:
                        gave_up = True
                        break
                    await asyncio.sleep(0.5)
                wait_ms = (time.monotonic() - waited_from) * 1000
                _, bg_reply_ms, bg_total_ms = await bg if not gave_up else (None, 0, 0)
                row = {"mode": mode, "id": c["id"], "file": c["file"], "bg_reply_ms": round(bg_reply_ms),
                       "bg_total_ms": round(bg_total_ms), "wait_llm_ms": round(wait_ms), "gave_up": gave_up}
                if gave_up:
                    await bg
                else:
                    d = await chime.decide(llm, recent=c["recent"], latest=c["latest"], bot_name=BOT_NAME,
                                           attention=True, **opts)
                    _, r_ms, rp_ms = await reply_and_pick(brain, eset, [*c["recent"], c["latest"]],
                                                          c["latest"].author, c["latest"].text)
                    row |= {"decide_ms": d.ms, "path": d.path, "action": d.action, "reply_ms": round(r_ms),
                            "reply_pick_ms": round(rp_ms), "total_ms": round((time.monotonic() - t0) * 1000)}
                rows.append(row)
                print(f"[busy {mode} {i}/{len(cases)}] {c['id']} 背景 {bg_total_ms / 1000:.1f}s 等 {wait_ms / 1000:.1f}s "
                      f"判斷 {row.get('decide_ms', 0) / 1000:.1f}s({row.get('path')}) 總 {row.get('total_ms', 0) / 1000:.1f}s",
                      flush=True)
        # 思考判斷進行中被直接叫：判斷開始 0.5 秒後送一則回覆，看回覆（含挑表情）多久才好
        opts = {**decide_options(s), "think_on_mention": True}
        called = []
        for i, c in enumerate(cases, 1):
            async def called_reply(c=c):
                await asyncio.sleep(0.5)
                t = time.monotonic()
                _, _, total = await reply_and_pick(brain, eset, [*c["recent"], c["latest"]],
                                                   c["latest"].author, c["latest"].text)
                return (time.monotonic() - t) * 1000, total
            dtask = asyncio.create_task(chime.decide(llm, recent=c["recent"], latest=c["latest"], bot_name=BOT_NAME,
                                                     attention=True, **opts))
            wall, own = await called_reply()
            d = await dtask
            called.append({"id": c["id"], "decide_ms": d.ms, "path": d.path, "called_wall_ms": round(wall),
                           "called_own_ms": round(own)})
            print(f"[called-during-think {i}/{len(cases)}] {c['id']} 判斷 {d.ms / 1000:.1f}s({d.path}) "
                  f"被叫的回覆 {wall / 1000:.1f}s", flush=True)
    finally:
        await llm.aclose()
    out = {"n_cases": len(cases), "rows": rows, "called_during_think": called}
    for mode in ("off", "on"):
        rs = [r for r in rows if r["mode"] == mode]
        ok = [r for r in rs if not r["gave_up"]]
        out[mode] = {"gave_up": sum(r["gave_up"] for r in rs), "wait_llm": stats([r["wait_llm_ms"] for r in rs]),
                     "decide": stats([r["decide_ms"] for r in ok]), "think_n": sum(r.get("path") == "think" for r in ok),
                     "reply_pick": stats([r["reply_pick_ms"] for r in ok]), "total": stats([r["total_ms"] for r in ok])}
    out["called_during_think_stats"] = {"wall": stats([r["called_wall_ms"] for r in called]),
                                        "think_n": sum(r["path"] == "think" for r in called)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("normal")
    a.add_argument("--off", nargs="+", required=True, help="兩段式關的 eval_chime 結果檔名（不含 .json）")
    a.add_argument("--on", nargs="+", required=True, help="兩段式開的 eval_chime 結果檔名")
    b = sub.add_parser("busy")
    b.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = load_settings()
    res = normal(args, s) if args.cmd == "normal" else asyncio.run(busy(args, s))
    out = RESULT_DIR / f"{args.out or 'd38_2_wait_' + args.cmd}.json"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "called_during_think")},
                     ensure_ascii=False, indent=1))
    print("結果：", out)


if __name__ == "__main__":
    main()
