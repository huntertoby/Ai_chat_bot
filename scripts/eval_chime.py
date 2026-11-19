"""階段 B 評估：插話判斷（B-規格「評估」）。題目在 data/eval_b/（不進 repo），結果寫到 data/eval_b/results/。

    python -m scripts.eval_chime                 # 全部：chime、milestone、flood
    python -m scripts.eval_chime --only chime    # chime / milestone / flood（可重複）
    python -m scripts.eval_chime --only flood --reply-sec 10 --no-real-replies
    python -m scripts.eval_chime --tag shut_up   # 只跑 chime 題目裡帶這個 tag 的

- chime：每題 decide()（參數跟正式設定一樣，scheduler.decide_options）；插話 = should_reply(門檻)，
  門檻預設＝正式設定（CHIME_IN 等級或 CHIME_PROB_THRESHOLD），react 另外統計，不算插話。talk_to_me 類在正式流程是被叫到直接回，
  題目多半沒叫名字（靠「你」），所以用 attention=True（她剛講完）；其他類 attention = 前一則是不是她講的。
- milestone：里程碑應插話、轉貼不應；圖縮成 512px 傳入。
- 速度：每題 Decision.ms 的中位數與 P90（標準 <= 0.6 秒）；兩段式（CHIME_THINK_ON_MENTION）另外列不思考那條路的速度。
- flood：用模擬時間重播人類訊息，算「會回幾則」。節流、熱鬧模式、排隊合併、插話前後的限制、注意時間
  都是正式的 yuki/scheduler.py（ReplyScheduler），只有時間是模擬的；回覆佔用頻道 --reply-sec 秒。另外挑 10 個被叫到的點實際跑 brain.reply，
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
from pathlib import Path

from yuki import chime, prompts, throttle
from yuki.bot import wake_pattern
from yuki.brain import Brain
from yuki.config import load_settings
from yuki.chat import ChatLine, Person
from yuki.scheduler import ReplyScheduler, Turn, chime_limits, decide_options

from ._eval_b_common import BOT_NAME, EVAL_DIR, RESULT_DIR, load_jsonl, make_llm, speed, to_line

THRESHOLD: float | None = None  # None＝正式設定的門檻
SWEEP = (0.5, 0.6, 0.7, 0.8)  # D-22：離線掃門檻（同一次跑記下機率，門檻在程式外套用）
USE_LOGPROBS: bool | None = None  # None＝正式設定（CHIME_USE_LOGPROBS）
THINK_ON_MENTION: bool | None = None  # None＝正式設定（CHIME_THINK_ON_MENTION，D-29 兩段式）
EMOJI_BLOCK = ""  # C1-2：--emoji 指到圖鑑時，插話判斷帶 10 張群組表情清單（跟上線時一樣）

# D-22 規則變體（評估用；只描述標準，不放範例句）。贏了才會寫進 yuki/prompts.py。
VARIANTS = {
    "none": "",
    # 在「不要開口」那段加一條
    "explain_flow": "- 兩個人正在一來一往解釋事情、話還沒講完：那是他們之間的對話，等講完再說，不要插話。\n",
}


def _images_of(lines: list[ChatLine], n: int = 3) -> list[str]:
    out = [u for l in lines for u in l.images]
    return out[-n:]


def decide_opts(s) -> dict:
    """正式設定給 decide() 的參數；--logprobs / --no-logprobs 才蓋過。"""
    opts = decide_options(s)
    if USE_LOGPROBS is not None:
        opts["use_logprobs"] = USE_LOGPROBS
    if THINK_ON_MENTION is not None:
        opts["think_on_mention"] = THINK_ON_MENTION
    return {**opts, "emoji_block": EMOJI_BLOCK}


def threshold(s) -> float:
    return THRESHOLD if THRESHOLD is not None else chime_limits(s).threshold


async def run_decide(llm, case: dict, *, attention: bool | None = None, s=None):
    recent = [to_line(m) for m in case["recent"]]
    latest = to_line(case["latest"])
    if attention is None:
        attention = case.get("label") == "talk_to_me" or bool(recent and recent[-1].is_bot)
    d = await chime.decide(llm, recent=recent, latest=latest, bot_name=BOT_NAME,
                           images=_images_of(recent + [latest]), attention=attention, **decide_opts(s))
    return d, attention


# ---------------------------------------------------------------- chime 50 題
async def eval_chime(llm, s, tag: str | None) -> dict:
    cases = load_jsonl(EVAL_DIR / "chime_cases.jsonl")
    if tag:
        cases = [c for c in cases if tag in c.get("tags", [])]
    rows = []
    for i, c in enumerate(cases, 1):
        d, att = await run_decide(llm, c, s=s)
        pred = d.should_reply(threshold(s))
        row = {"id": c["id"], "label": c["label"], "tags": c.get("tags", []), "expect": c["expect_participate"],
               "pred": pred, "react": d.should_react(threshold(s)), "attention": att,
               "ok": pred == c["expect_participate"], "decision": d.to_dict()}
        rows.append(row)
        print(f"[chime {i}/{len(cases)}] {c['id']} {c['label']} 預期 {c['expect_participate']} 實際 {pred} "
              f"({d.action} {d.participate:.2f} {d.kind} {d.path}) {d.ms}ms{' ERR ' + d.error if d.error else ''}", flush=True)
    n = len(rows)
    by = {}
    for lab in ("talk_to_me", "about_me", "others"):
        rs = [r for r in rows if r["label"] == lab]
        by[lab] = {"n": len(rs), "acc": round(sum(r["ok"] for r in rs) / max(len(rs), 1), 3)}
    pro = [r for r in rows if "pronoun" in r["tags"]]
    by_tag = {}
    for t in ("mocked", "commentary", "praise", "question_about", "to_her", "has_material", "d29_new"):
        rs = [r for r in rows if t in r["tags"]]
        if rs:
            by_tag[t] = f"{sum(r['ok'] for r in rs)}/{len(rs)}"
    by_path = {}
    for pth in ("fast", "think"):
        rs = [r for r in rows if r["decision"].get("path", "fast") == pth]
        if rs:
            by_path[pth] = {"n": len(rs), "correct": sum(r["ok"] for r in rs),
                            "speed": speed([r["decision"]["ms"] for r in rs])}
    loose = sum(((r["pred"] or r["react"]) == r["expect"]) for r in rows) / max(n, 1)
    kinds = Counter((r["expect"], r["pred"]) for r in rows)
    return {"sweep": sweep_chime(rows), "prob_source": dict(Counter(r["decision"].get("prob_source") for r in rows)),
            "n": n, "accuracy": round(sum(r["ok"] for r in rows) / max(n, 1), 3),
            "accuracy_react_counts": round(loose, 3), "by_label": by, "by_tag": by_tag, "by_path": by_path,
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
        pred = d.should_reply(threshold(s))
        has_img = bool(c["latest"].get("images"))
        rows.append({"id": c["id"], "group": c["group"], "pred": pred, "react": d.should_react(threshold(s)),
                     "path": d.path,
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
            "think_n": sum(r["path"] == "think" for r in rows),
            "speed_text_only": speed([r["decision"]["ms"] for r in rows if not r["image_only"]]),
            "speed_with_image": speed([r["decision"]["ms"] for r in rows if r["image_only"]]),
            "errors": sum(bool(r["decision"].get("error")) for r in rows), "rows": rows}


# ---------------------------------------------------------------- 洗版重播
class SimClock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


class SimScheduler(ReplyScheduler):
    """正式的排程器（同一套節流、熱鬧模式、排隊合併、插話限制、注意時間），只是不開背景計時：
    「回覆花多久、熱鬧模式等多久」由重播的模擬時間（advance）推進。"""

    def _schedule_flush(self, cid) -> None:
        return None


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

    async def no_respond(turn):  # 重播不真的回覆；回覆佔用頻道的時間用 reply_sec 模擬
        return None

    sch = SimScheduler(s, no_respond, clock=clock,
                       today=lambda: date.fromtimestamp(clock.t + 86400 * 365 * 50))
    th = sch.throttle
    wake = wake_pattern(s.wake_words)
    ch = "chan"
    hist: list[ChatLine] = []
    st: Counter = Counter()
    called_points: list[dict] = []
    state = {"busy_until": 0.0}
    pending: dict | None = None

    def start_reply(kind: str, at: float, turn: Turn | None) -> None:
        """頻道已經佔住（submit / take_queued / plan_chime 給的），開始一則回覆。"""
        st["replies"] += 1
        st[f"replies_{kind}"] += 1
        state["busy_until"] = at + reply_sec
        if kind == "chime":
            sch.note_chimed(ch)
        elif turn is not None and turn.trigger == "called":
            sch.extend_attention(ch)
        if turn is not None and kind != "chime":
            called_points.append({"id": turn.message["id"], "idx": turn.message["_idx"]})

    def advance(to: float) -> None:
        """時間走到 to：照 ReplyScheduler.run_turns / _flush_later 的順序處理回覆結束和排隊的合併回覆。"""
        while True:
            if th.is_busy(ch):
                if state["busy_until"] > to:
                    break
                clock.t = max(clock.t, state["busy_until"])
                n = th.queued(ch)
                nxt = sch.next_turn(ch)  # 回完一則：排隊中已經可以回的合成下一則
                if nxt is None:
                    th.finish(ch)
                    continue
                st["merged_from"] += n - th.queued(ch)
                start_reply("queued", clock.t, nxt)
            else:
                wait = th.ready_in(ch)
                if wait is None or clock.t + wait > to:
                    break
                clock.t += wait
                st["merged_from"] += th.queued(ch)
                start_reply("queued", clock.t, sch.take_queued(ch))
        clock.t = max(clock.t, to)

    async def fire_chime(p: dict) -> None:
        at = p["fire"]
        advance(at)
        att = any(sch.attention(ch))
        m = p["msg"]
        reason = sch.chime_gate(ch, attention=att, llm_busy=False)
        if reason:
            st[f"chime_skip_{reason}"] += 1
            return
        d = await chime.decide(llm, recent=p["recent"], latest=p["line"], bot_name=BOT_NAME, images=[],
                               attention=att, **decide_opts(s))
        st["decide_calls"] += 1
        plan = sch.plan_chime(d, ch, m["author"], attention=att)
        if plan == "reply":
            start_reply("chime", at, None)
        elif plan == "react":
            sch.record_react(ch, m["author"], d.kind)
            st["reacts"] += 1
        elif plan == "error":
            st["chime_error"] += 1
        elif plan == "none":
            st["chime_none"] += 1
        else:
            st["chime_" + plan.replace(":", "_")] += 1

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
        active, chiming = sch.attention(ch)
        if (called or active or chiming) and chime.is_shush(m["content"]):
            sch.mute(ch)
            st["shush"] += 1
        line = to_line(m, images=False)
        line.text = text or "（貼了圖片）"
        if text and not (hist and throttle.is_repeat(hist[-1], line)):
            hist.append(line)
        m["_idx"] = len(hist) - 1
        if called:
            pending = None
            st["called"] += 1
            before = th.queued(ch)
            admitted = sch.admit_call(ch, uid)
            if admitted is None:
                st["tag_blocked"] += 1
                st["queue_dropped"] += before - th.queued(ch)
                continue
            if not text:
                st["replies"] += 1
                st["replies_empty_call"] += 1
                continue
            now = sch.submit(ch, Turn(m, Person(uid, line.author.name), text, []), admitted)
            if now is None:
                st["held_burst" if admitted == "hold" else "queued"] += 1
            else:
                if before:
                    st["merged_from"] += before + 1
                start_reply("called", t, now)
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
    brain = Brain(replace(s, web_enabled=True), llm)
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
    global THRESHOLD, USE_LOGPROBS, EMOJI_BLOCK, THINK_ON_MENTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", choices=["chime", "milestone", "flood"])
    ap.add_argument("--tag", default=None, help="只跑 chime 題目裡帶這個 tag 的")
    ap.add_argument("--reply-sec", type=float, default=10.0, help="洗版重播：一次回覆佔用頻道的秒數")
    ap.add_argument("--no-real-replies", action="store_true", help="洗版重播不實際產生 10 則回覆")
    ap.add_argument("--out", default="eval_chime")
    ap.add_argument("--threshold", type=float, default=None, help="主結果用的門檻（預設＝正式設定；掃描表另外列 0.5~0.8）")
    lp = ap.add_mutually_exclusive_group()
    lp.add_argument("--logprobs", dest="logprobs", action="store_true", default=None,
                    help="開口機率從 logprobs 算（預設照 CHIME_USE_LOGPROBS）")
    lp.add_argument("--no-logprobs", dest="logprobs", action="store_false", help="用模型寫的 participate")
    th = ap.add_mutually_exclusive_group()
    th.add_argument("--think-on-mention", dest="think", action="store_true", default=None,
                    help="D-29 兩段式：提到她才思考（預設照 CHIME_THINK_ON_MENTION）")
    th.add_argument("--no-think", dest="think", action="store_false", help="全部不思考")
    ap.add_argument("--variant", choices=list(VARIANTS), default="none", help="D-22 規則變體（只影響本次評估）")
    ap.add_argument("--action-first", action="store_true", help="D-22 實驗：schema 把 action 放最前面（機率不受 reason 影響）")
    ap.add_argument("--emoji", type=Path, default=None, help="C1-2：圖鑑檔（不看 reviewed），帶表情清單跑插話判斷")
    ap.add_argument("--emoji-n", type=int, default=0, help="表情清單放幾張（預設 emoji_catalog.REACT_N）")
    args = ap.parse_args()
    THRESHOLD, USE_LOGPROBS, THINK_ON_MENTION = args.threshold, args.logprobs, args.think
    if args.emoji:
        from yuki import emoji_catalog
        eset = emoji_catalog.load_set(args.emoji, require_reviewed=False)
        EMOJI_BLOCK = (eset.react_block(args.emoji_n or emoji_catalog.REACT_N) if eset else "")
    if args.action_first:
        order = ["action", "kind", "reason", "participate", "emoji", "need_background"]
        sc = prompts.CHIME_SCHEMA
        prompts.CHIME_SCHEMA = {**sc, "properties": {k: sc["properties"][k] for k in order}, "required": order}
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
