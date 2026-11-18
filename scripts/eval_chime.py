"""階段 B 評估：插話判斷（B-規格「評估」）。題目在 data/eval_b/（不進 repo），結果寫到 data/eval_b/results/。

    python -m scripts.eval_chime                 # 全部：chime、milestone、flood
    python -m scripts.eval_chime --only chime    # chime / milestone / flood（可重複）
    python -m scripts.eval_chime --only flood --reply-sec 10 --no-real-replies
    python -m scripts.eval_chime --tag shut_up   # 只跑 chime 題目裡帶這個 tag 的

- chime：每題 decide()；插話 = should_reply(0.7)（react 另外統計，不算插話）。talk_to_me 類在正式流程是被叫到直接回，
  題目多半沒叫名字（靠「你」），所以用 attention=True（她剛講完）；其他類 attention = 前一則是不是她講的。
- milestone：里程碑應插話、轉貼不應；圖縮成 512px 傳入。
- 速度：每題 Decision.ms 的中位數與 P90（標準 <= 0.6 秒）。
- flood：用模擬時間重播人類訊息（Throttle、ChimeLimits、雜訊過濾、4 秒去抖動、decide），算「會回幾則」；
  回覆佔用頻道 --reply-sec 秒（忙碌期間被叫的排隊合併）。另外挑 10 個被叫到的點實際跑 brain.reply，
  檢查有沒有第三人稱稱呼自己（規則 + LLM 判斷）。重播的對話紀錄只有人類訊息（不含 Yuki 的回覆）。
LM Studio 一次只處理一個請求；依序跑。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from dataclasses import replace
from datetime import date, datetime

from yuki import chime, prompts, throttle
from yuki.brain import Brain
from yuki.config import load_settings
from yuki.memory.store import MemoryStore
from yuki.memory.writer import ChatLine
from yuki.rag.decompose import Person

from ._eval_b_common import BOT_NAME, EVAL_DIR, RESULT_DIR, load_jsonl, make_llm, speed, to_line

THRESHOLD = 0.7
SWEEP = (0.5, 0.6, 0.7, 0.8)  # D-22：離線掃門檻（同一次跑記下機率，門檻在程式外套用）
USE_LOGPROBS = True

# D-22 規則變體（評估用；只描述標準，不放範例句）。贏了才會寫進 yuki/prompts.py。
VARIANTS = {
    "none": "",
    # 在「不要開口」那段加一條
    "explain_flow": "- 兩個人正在一來一往解釋事情、話還沒講完：那是他們之間的對話，等講完再說，不要插話。
",
}


def _images_of(lines: list[ChatLine], n: int = 3) -> list[str]:
    out = [u for l in lines for u in l.images]
    return out[-n:]


async def run_decide(llm, case: dict, *, attention: bool | None = None, s=None):
    recent = [to_line(m) for m in case["recent"]]
    latest = to_line(case["latest"])
    if attention is None:
        attention = case.get("label") == "talk_to_me" or bool(recent and recent[-1].is_bot)
    d = await chime.decide(llm, recent=recent, latest=latest, bot_name=BOT_NAME,
                           images=_images_of(recent + [latest]), attention=attention,
                           context_lines=s.chime_context_lines if s else 15,
                           max_tokens=s.chime_max_tokens if s else 160, use_logprobs=USE_LOGPROBS)
    return d, attention


# ---------------------------------------------------------------- chime 50 題
async def eval_chime(llm, s, tag: str | None) -> dict:
    cases = load_jsonl(EVAL_DIR / "chime_cases.jsonl")
    if tag:
        cases = [c for c in cases if tag in c.get("tags", [])]
    rows = []
    for i, c in enumerate(cases, 1):
        d, att = await run_decide(llm, c, s=s)
        pred = d.should_reply(THRESHOLD)
        row = {"id": c["id"], "label": c["label"], "tags": c.get("tags", []), "expect": c["expect_participate"],
               "pred": pred, "react": d.should_react(THRESHOLD), "attention": att,
               "ok": pred == c["expect_participate"], "decision": d.to_dict()}
        rows.append(row)
        print(f"[chime {i}/{len(cases)}] {c['id']} {c['label']} 預期 {c['expect_participate']} 實際 {pred} "
              f"({d.action} {d.participate:.2f}) {d.ms}ms{' ERR ' + d.error if d.error else ''}", flush=True)
    n = len(rows)
    by = {}
    for lab in ("talk_to_me", "about_me", "others"):
        rs = [r for r in rows if r["label"] == lab]
        by[lab] = {"n": len(rs), "acc": round(sum(r["ok"] for r in rs) / max(len(rs), 1), 3)}
    pro = [r for r in rows if "pronoun" in r["tags"]]
    loose = sum(((r["pred"] or r["react"]) == r["expect"]) for r in rows) / max(n, 1)
    kinds = Counter((r["expect"], r["pred"]) for r in rows)
    return {"sweep": sweep_chime(rows), "prob_source": dict(Counter(r["decision"].get("prob_source") for r in rows)),
            "n": n, "accuracy": round(sum(r["ok"] for r in rows) / max(n, 1), 3),
            "accuracy_react_counts": round(loose, 3), "by_label": by,
            "pronoun": {"n": len(pro), "correct": sum(r["ok"] for r in pro), "all_correct": all(r["ok"] for r in pro)},
            "false_positive": kinds[(False, True)], "false_negative": kinds[(True, False)],
            "react_count": sum(r["react"] for r in rows),
            "errors": sum(bool(r["decision"].get("error")) for r in rows),
            "speed": speed([r["decision"]["ms"] for r in rows]),
            "failed": [{k: r[k] for k in ("id", "label", "expect", "pred", "react", "attention")}
                       | {"participate": r["decision"]["participate"], "action": r["decision"]["action"],
                          "kind": r["decision"]["kind"]} for r in rows if not r["ok"]],
            "rows": rows}


def sweep_chime(rows: list[dict], thresholds=SWEEP) -> dict:
    """用記下的機率離線套門檻：不用重跑模型。should_reply 的規則跟 Decision.should_reply 一樣。"""
    out = {}
    for th in thresholds:
        def pred(r, th=th):
            return chime.Decision(**{k: v for k, v in r["decision"].items()
                                     if k in ("participate", "action", "prob_source", "error")}).should_reply(th)
        by = {}
        for lab in ("talk_to_me", "about_me", "others"):
            rs = [r for r in rows if r["label"] == lab]
            by[lab] = round(sum(pred(r) == r["expect"] for r in rs) / max(len(rs), 1), 3)
        pro = [r for r in rows if "pronoun" in r["tags"]]
        out[str(th)] = {"accuracy": round(sum(pred(r) == r["expect"] for r in rows) / max(len(rows), 1), 3), "by_label": by,
                        "pronoun": f"{sum(pred(r) == r['expect'] for r in pro)}/{len(pro)}",
                        "false_positive": sum(pred(r) and not r["expect"] for r in rows),
                        "false_negative": sum((not pred(r)) and r["expect"] for r in rows)}
    return out


# ---------------------------------------------------------------- milestone 40 題
async def eval_milestone(llm, s) -> dict:
    cases = load_jsonl(EVAL_DIR / "milestone_cases.jsonl")
    rows = []
    for i, c in enumerate(cases, 1):
        d, _ = await run_decide(llm, c, attention=False, s=s)
        pred = d.should_reply(THRESHOLD)
        has_img = bool(c["latest"].get("images"))
        rows.append({"id": c["id"], "group": c["group"], "pred": pred, "react": d.should_react(THRESHOLD),
                     "image_only": has_img, "decision": d.to_dict()})
        print(f"[milestone {i}/{len(cases)}] {c['id']} {c['group']}{' (圖)' if has_img else ''} 插話 {pred} "
              f"({d.action} {d.participate:.2f}) {d.ms}ms", flush=True)
    ms = [r for r in rows if r["group"] == "milestone"]
    rp = [r for r in rows if r["group"] == "repost"]
    mi = [r for r in ms if r["image_only"]]
    sweep = {}
    for th in SWEEP:
        def pr(r, th=th):
            dd = r["decision"]
            return chime.Decision(participate=dd["participate"], action=dd["action"], prob_source=dd["prob_source"],
                                  error=dd.get("error", "")).should_reply(th)
        sweep[str(th)] = {"milestone_rate": round(sum(pr(r) for r in ms) / max(len(ms), 1), 3),
                          "repost_rate": round(sum(pr(r) for r in rp) / max(len(rp), 1), 3)}
    return {"sweep": sweep, "milestone_rate": round(sum(r["pred"] for r in ms) / max(len(ms), 1), 3), "milestone_n": len(ms),
            "milestone_img_rate": round(sum(r["pred"] for r in mi) / max(len(mi), 1), 3), "milestone_img_n": len(mi),
            "milestone_react_or_reply": round(sum(r["pred"] or r["react"] for r in ms) / max(len(ms), 1), 3),
            "repost_rate": round(sum(r["pred"] for r in rp) / max(len(rp), 1), 3), "repost_n": len(rp),
            "repost_react": sum(r["react"] for r in rp),
            "missed_milestones": [r["id"] for r in ms if not r["pred"]],
            "false_reposts": [r["id"] for r in rp if r["pred"]],
            "speed": speed([r["decision"]["ms"] for r in rows]),
            "speed_text_only": speed([r["decision"]["ms"] for r in rows if not r["image_only"]]),
            "speed_with_image": speed([r["decision"]["ms"] for r in rows if r["image_only"]]),
            "errors": sum(bool(r["decision"].get("error")) for r in rows), "rows": rows}


# ---------------------------------------------------------------- 洗版重播
class SimClock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def _ts(m: dict) -> float:
    return datetime.strptime(m["time"], "%Y-%m-%d %H:%M:%S").timestamp()


def strip_call(text: str, wake: re.Pattern) -> str:
    """跟 bot.clean 一樣：拿掉 @Yuki 和喚醒詞。"""
    text = text.replace("@Yuki", " ")
    return re.sub(r"\s+", " ", wake.sub(" ", text)).strip(" ,，")


async def replay(llm, s, *, reply_sec: float, debounce: float):
    msgs = load_jsonl(EVAL_DIR / "flood_replay.jsonl")
    meta = json.loads((EVAL_DIR / "flood_meta.json").read_text(encoding="utf-8"))
    humans = [m for m in msgs if not m["is_yuki"]]
    clock = SimClock()
    cfg = throttle.ThrottleConfig(cooldown_sec=s.throttle_cooldown_sec, tag_limit=s.throttle_tag_limit,
                                  tag_window_sec=s.throttle_tag_window_sec,
                                  tag_cooldown_sec=s.throttle_tag_cooldown_sec,
                                  channel_call_burst=s.channel_call_burst,
                                  channel_call_window_sec=s.channel_call_window_sec,
                                  channel_burst_mode_sec=s.channel_burst_mode_sec,
                                  channel_call_gap_sec=s.channel_call_gap_sec)
    th = throttle.Throttle(cfg, clock=clock)
    limits = chime.ChimeLimits(s.chime_in, threshold=s.chime_threshold, mute_sec=s.chime_mute_minutes * 60,
                               react_max=s.chime_react_max, react_window_sec=s.chime_react_window_min * 60,
                               clock=clock, today=lambda: date.fromtimestamp(clock.t + 86400 * 365 * 50))
    wake = re.compile("|".join(re.escape(w) for w in s.wake_words), re.I)
    ch = "chan"
    hist: list[ChatLine] = []
    st: Counter = Counter()
    called_points: list[dict] = []
    state = {"busy_until": 0.0, "active_until": 0.0}
    pending: dict | None = None

    def start_reply(kind: str, at: float, msg: dict | None) -> None:
        st["replies"] += 1
        st[f"replies_{kind}"] += 1
        state["busy_until"] = at + reply_sec
        if kind != "chime":
            state["active_until"] = at + s.attention_timeout
        if msg is not None:
            called_points.append({"id": msg["id"], "idx": msg["_idx"]})

    def advance(to: float) -> None:
        """時間走到 to：回覆結束就解除忙碌；排隊的合成一次回覆（熱鬧模式中要等到下一次可回的時間）。"""
        while True:
            if th.is_busy(ch):
                if state["busy_until"] > to:
                    break
                clock.t = max(clock.t, state["busy_until"])
                th.finish(ch)
                q = th.drain_ready(ch)
            else:
                wait = th.ready_in(ch)
                if wait is None or clock.t + wait > to:
                    break
                clock.t += wait
                q = th.drain(ch)
            if q:
                st["merged_from"] += len(q)
                th.try_start(ch)
                start_reply("queued", clock.t, q[-1])
        clock.t = max(clock.t, to)

    async def fire_chime(p: dict) -> None:
        at = p["fire"]
        advance(at)
        att = state["active_until"] > at
        m = p["msg"]
        reason = (limits.gate(ch, attention=att) or ("busy" if th.is_busy(ch) else None)
                  or ("cooldown" if th.in_cooldown(ch) else None))
        if reason:
            st[f"chime_skip_{reason}"] += 1
            return
        d = await chime.decide(llm, recent=p["recent"], latest=p["line"], bot_name=BOT_NAME, images=[],
                               attention=att, context_lines=s.chime_context_lines, max_tokens=s.chime_max_tokens,
                               use_logprobs=USE_LOGPROBS)
        st["decide_calls"] += 1
        if d.error:
            st["chime_error"] += 1
        elif d.should_reply(limits.threshold):
            if not limits.kind_ok(m["author"], d.kind, attention=att):
                st["chime_skip_kind_daily_limit"] += 1
            elif not th.try_start(ch):
                st["chime_skip_busy"] += 1
            else:
                limits.record_reply(ch, m["author"], d.kind, attention=att)
                start_reply("chime", at, None)
        elif d.should_react(0.0):
            g = limits.react_gate(ch)
            if g:
                st[f"react_skip_{g}"] += 1
            elif not limits.kind_ok(m["author"], d.kind, attention=att):
                st["chime_skip_kind_daily_limit"] += 1
            else:
                limits.record_react(ch, m["author"], d.kind)
                st["reacts"] += 1
        else:
            st["chime_none"] += 1

    t0 = _ts(humans[0])
    for m in humans:
        t = _ts(m) - t0
        if pending is not None and pending["fire"] <= t:
            p, pending = pending, None
            await fire_chime(p)
        advance(t)
        st["human_msgs"] += 1
        raw = m["content"].replace("@Yuki", "<@100>")
        if throttle.is_flood(raw, repeat_chars=s.flood_repeat_chars, repeat_mentions=s.flood_repeat_mentions):
            st["flood_dropped"] += 1
            continue
        if throttle.is_command(m["content"]):
            st["command_skipped"] += 1
            continue
        text = strip_call(m["content"], wake)
        uid = m["author"]
        called = bool(m["mentions_yuki"] or wake.search(m["content"])
                      or (m.get("reply_to") and m["reply_to"].get("is_yuki")))
        active = state["active_until"] > t
        if (called or active) and chime.is_shush(m["content"]):
            limits.mute(ch)
            st["shush"] += 1
        line = to_line(m, images=False)
        line.text = text or "（貼了圖片）"
        if text and not (hist and throttle.is_repeat(hist[-1], line)):
            hist.append(line)
        m["_idx"] = len(hist) - 1
        if called:
            pending = None
            st["called"] += 1
            gate_ok = th.note_call(ch)
            if not th.note_tag(uid):
                st["tag_blocked"] += 1
                st["queue_dropped"] += th.drop_queued(ch, lambda x: x["author"] == uid)
                continue
            if not text:
                st["replies"] += 1
                st["replies_empty_call"] += 1
                continue
            if not gate_ok:  # 熱鬧模式：收集起來，之後合成一次回
                th.enqueue(ch, m)
                st["held_burst"] += 1
                continue
            if not th.try_start(ch):
                th.enqueue(ch, m)
                st["queued"] += 1
                continue
            start_reply("called", t, m)
            continue
        if not text or throttle.is_noise_text(text, s.noise_max_chars):
            st["noise_skipped"] += 1
            continue
        st["chime_candidates"] += 1
        if pending is not None:
            st["debounced"] += 1
        pending = {"fire": t + debounce, "msg": m, "line": line,
                   "recent": list(hist[:-1])[-s.chime_context_lines:]}
        if st["chime_candidates"] % 10 == 0:
            print(f"[flood] {st['human_msgs']}/{len(humans)} 回覆 {st['replies']}", flush=True)
    if pending is not None:
        await fire_chime(pending)
    advance(1e12)
    limit = meta["yuki_reply_limit"]
    out = {"human_msgs": st["human_msgs"], "reply_limit": limit, "replies": st["replies"],
           "pass": st["replies"] <= limit, "reply_sec_model": reply_sec, "counters": dict(st)}
    return out, called_points, hist


SELF_REF = re.compile(r"yuki|她自己", re.I)
JUDGE = """你是評分員。下面是聊天機器人 Yuki 自己寫的一則回覆。判斷她有沒有「用第三人稱稱呼自己」：
也就是回覆裡把自己當成別人在講，例如叫自己「Yuki」或用「她」指自己；用「我」自稱、或只是回覆對方提到的名字不算。
只回一個字：有 或 沒有。"""


async def real_replies(llm, s, points: list[dict], hist: list[ChatLine], n: int) -> dict:
    if not points:
        return {"n": 0}
    step = max(len(points) // n, 1)
    chosen = points[::step][:n]
    settings = replace(s, retrieval="agentic", web_enabled=True)
    brain = Brain(settings, llm, MemoryStore(":memory:", "fake"), None, None)
    uniq = {l.author.id: Person(l.author.id, l.author.name) for l in hist if not l.is_bot}
    out = []
    for i, pt in enumerate(chosen, 1):
        idx = pt["idx"]
        if idx < 0 or idx >= len(hist):
            continue
        brain.history.clear()
        for l in hist[max(0, idx - 14): idx + 1]:
            brain.history["flood"].append(l)
        line = hist[idx]
        try:
            reply = await brain.reply("eval", "flood", line.author, line.text, members=list(uniq.values()))
        except Exception as e:  # noqa: BLE001
            reply = ""
            print(f"[reply {i}] 失敗：{e!r}", flush=True)
        j = (await llm.chat([{"role": "system", "content": JUDGE},
                             {"role": "user", "content": f"回覆：{reply}"}])).strip()
        row = {"id": pt["id"], "input": line.text, "reply": reply, "rule_flag": bool(SELF_REF.search(reply)),
               "judge_third_person": j.startswith("有")}
        out.append(row)
        print(f"[reply {i}/{len(chosen)}] 規則 {row['rule_flag']} 判斷 {row['judge_third_person']}｜{reply[:60]}",
              flush=True)
    return {"n": len(out), "third_person_judge": sum(r["judge_third_person"] for r in out),
            "rule_flag": sum(r["rule_flag"] for r in out), "rows": out}


async def eval_flood(llm, s, args) -> dict:
    res, points, hist = await replay(llm, s, reply_sec=args.reply_sec, debounce=s.chime_debounce_sec)
    res["called_reply_points"] = len(points)
    if not args.no_real_replies:
        res["real_replies"] = await real_replies(llm, s, points, hist, 10)
    return res


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", choices=["chime", "milestone", "flood"])
    ap.add_argument("--tag", default=None, help="只跑 chime 題目裡帶這個 tag 的")
    ap.add_argument("--reply-sec", type=float, default=10.0, help="洗版重播：一次回覆佔用頻道的秒數")
    ap.add_argument("--no-real-replies", action="store_true", help="洗版重播不實際產生 10 則回覆")
    ap.add_argument("--out", default="eval_chime")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="主結果用的門檻（掃描表另外列 0.5~0.8）")
    ap.add_argument("--no-logprobs", action="store_true", help="不用 logprobs，用模型寫的 participate（舊做法）")
    ap.add_argument("--variant", choices=list(VARIANTS), default="none", help="D-22 規則變體（只影響本次評估）")
    args = ap.parse_args()
    global THRESHOLD, USE_LOGPROBS
    THRESHOLD, USE_LOGPROBS = args.threshold, not args.no_logprobs
    if VARIANTS[args.variant]:
        marker = "- 沒內容的反應"
        assert marker in prompts.CHIME_SYSTEM
        prompts.CHIME_SYSTEM = prompts.CHIME_SYSTEM.replace(marker, VARIANTS[args.variant] + marker, 1)
    which = args.only or ["chime", "milestone", "flood"]
    s = load_settings()
    llm = make_llm(s)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    try:
        for name in which:
            if name == "chime":
                result["chime"] = await eval_chime(llm, s, args.tag)
            elif name == "milestone":
                result["milestone"] = await eval_milestone(llm, s)
            else:
                result["flood"] = await eval_flood(llm, s, args)
            (RESULT_DIR / f"{args.out}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
    finally:
        await llm.aclose()
    brief = {k: {kk: vv for kk, vv in v.items() if kk not in ("rows", "failed")} for k, v in result.items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print("結果：", RESULT_DIR / f"{args.out}.json")


if __name__ == "__main__":
    asyncio.run(main())
