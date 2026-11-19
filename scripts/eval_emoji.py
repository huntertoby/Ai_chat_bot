"""C1-2 評估：群組表情與貼圖（開發規格 6.14 驗收）。不需要 Discord，只用 LM Studio。
題目在 data/eval_c1/（不進 repo，裡面是真實聊天），結果寫到 data/eval_c1/results/。

    python -m scripts.eval_emoji --prepare                 # 從封存區挑題目（只需要做一次）
    python -m scripts.eval_emoji                           # 全部：reply、react、voice
    python -m scripts.eval_emoji --only react --out react_1
    python -m scripts.eval_emoji --catalog data/emoji/<ID>.json

--catalog 直接指到圖鑑檔，**不看 reviewed**（評估用；Yuki 本體仍然只用 reviewed 的圖鑑）。預設是 PROFILE_FALLBACK_GUILD 的圖鑑。

- reply：50 則真實對話片段當前文，用跟 Yuki 一樣的 reply system（有群組表情清單，沒有檢索工具）產生回覆，
  後處理後統計：用到表情（或貼圖）的比例（標準 10%～30%）、壞掉的 `:名稱:`（標準 0）、一則兩個以上（標準 0）。
  另外記後處理前的原始狀況（模型自己寫錯名稱、寫了兩個以上的次數），看後處理救了多少。
- react：20 則「值得理一下」的訊息（封存區裡接著有人笑或恭喜的），跑插話判斷（帶 10 張表情清單），
  選 react 而且表情合適的比例（標準 >= 70%）；合適由 LLM 依 meaning 判，另外列 5 則給人工看。
- voice：reply 的回覆（含後處理前的原始文字）和幾句固定的字串，speakable() 之後不能有 `<:`、`<a:`、`[貼圖`。
LM Studio 一次只處理一個請求；依序跑。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
from datetime import date
from pathlib import Path

from yuki import chime, emoji_catalog as cat, prompts
from yuki.archive import ArchiveDB
from yuki.brain import _NAME_PREFIX, to_messages, to_traditional
from yuki.config import ROOT, load_settings
from yuki.voice.tts import speakable

from ._eval_b_common import BOT_NAME, make_llm, speed, to_line

DIR = ROOT / "data" / "eval_c1"
RESULTS = DIR / "results"
ARCHIVE = ROOT / "data" / "archive.db"
CONTEXT = 15

LAUGH = re.compile(r"笑死|哈哈|XDD|www|ww|讚|屌")
CONGRATS = re.compile(r"恭喜|太強|屌|賀|讚|讚啦|厲害|威|猛")
MILESTONE = re.compile(r"考上|上鑽|上分|畢業|結訓|拿到|終於|生日|及格|通過|過了|升|贏了|抽到")
GRIPE = re.compile(r"氣死|爛|靠北|崩潰|倒楣|歪|雷|慘|煩")
URL = re.compile(r"https?://|<a?:\w+:\d+>")


# ---------------------------------------------------------------- 題目
def _load_rows(db, guild: str, opted: set[str]):
    rows = db.execute("SELECT id, channel_id, author_id, author_name, created_at, content, attachments "
                      "FROM messages WHERE guild_id=? AND created_at>='2026-01-01' ORDER BY channel_id, id",
                      (guild,)).fetchall()
    return [r for r in rows if r[2] not in opted and "yuki" not in r[3].lower()]


def _case(rows, i, label, n=CONTEXT):
    mid, ch, _a, name, created, content, _att = rows[i]
    recent = []
    for j in range(max(0, i - n), i):
        r = rows[j]
        if r[1] == ch:
            recent.append({"author": r[3], "content": r[5][:200]})
    return {"id": f"{label}_{mid}", "label": label, "recent": recent,
            "latest": {"author": name, "content": content[:200]}, "created": created}


def prepare(seed: int = 7) -> None:
    s = load_settings()
    guild = s.profile_fallback_guild
    arch = ArchiveDB(ARCHIVE)
    rows = _load_rows(arch.db, guild, arch.opted_out(guild))
    rng = random.Random(seed)

    def ok_text(r, lo=6, hi=60):
        return (lo <= len(r[5]) <= hi and not URL.search(r[5]) and r[6] == 0 and not r[5].startswith("@")
                and re.search(r"[一-鿿]", r[5]) is not None)

    def following(i, pat, need=1):
        ch, author = rows[i][1], rows[i][3]
        nxt = [r for r in rows[i + 1:i + 7] if r[1] == ch and r[3] != author]
        return sum(bool(pat.search(r[5])) for r in nxt[:3]) >= need

    def has_context(i):
        return i >= CONTEXT and all(rows[k][1] == rows[i][1] for k in (i - 1, i - 5))

    # 回覆題：隨機 50 則（不同天、有前文、是一般的一句話）
    cand = [i for i in range(len(rows)) if ok_text(rows[i], 4, 80) and has_context(i)]
    rng.shuffle(cand)
    reply, days = [], set()
    for i in cand:
        d = rows[i][4][:10]
        if d in days:
            continue
        days.add(d)
        reply.append(_case(rows, i, "reply"))
        if len(reply) == 50:
            break
    # 反應題：好笑（後面有人笑）12 + 里程碑（後面有人恭喜）8
    def pick(pat_msg, pat_next, n, label, need=1):
        c = [i for i in range(len(rows)) if ok_text(rows[i], 6, 50) and has_context(i)
             and (pat_msg is None or pat_msg.search(rows[i][5])) and following(i, pat_next, need)]
        rng.shuffle(c)
        out, dd = [], {}
        for i in c:
            d = rows[i][4][:10]
            if dd.get(d, 0) >= 2:
                continue
            dd[d] = dd.get(d, 0) + 1
            out.append(_case(rows, i, label))
            if len(out) == n:
                break
        return out

    react = (pick(MILESTONE, CONGRATS, 8, "milestone") + pick(GRIPE, LAUGH, 6, "gripe", need=1)
             + pick(None, LAUGH, 20, "funny", need=2))
    react = react[:20]
    DIR.mkdir(parents=True, exist_ok=True)
    for name, cases in (("reply_cases", reply), ("react_cases", react)):
        (DIR / f"{name}.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
                                           encoding="utf-8")
        print(name, len(cases))


def load_cases(name: str) -> list[dict]:
    return [json.loads(l) for l in (DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- reply
SHORTCODE = re.compile(r"(?<![\w:]):\w{1,40}:(?![\w:])")
CUSTOM = re.compile(r"<a?:\w+:\d+>")
TAG = re.compile(r"[\[［]\s*貼圖")


def raw_stats(text: str, eset: cat.EmojiSet) -> dict:
    """後處理之前模型寫了什麼：總共幾個表情／貼圖記號、有幾個名稱不在圖鑑裡。"""
    toks = [m for m in cat._ANY.finditer(text)]
    bad = 0
    for m in toks:
        if m.group(3) is not None:
            bad += eset._by_id.get(m.group(3)) is None
        elif m.group(4) is not None:
            bad += ("emoji", m.group(4).lower()) not in eset._by_key
        else:
            bad += ("sticker", m.group(5).strip().lower()) not in eset._by_key
    return {"marks": len(toks), "bad": bad}


async def eval_reply(llm, eset: cat.EmojiSet, s) -> dict:
    cases = load_cases("reply_cases")
    block = eset.prompt_block()
    pacing = cat.EmojiPacing()  # 跟 bot 一樣：用過之後接下來 2 則不放清單、不用（50 題當成同一個頻道依序回）
    rows = []
    for k, c in enumerate(cases, 1):
        allow = pacing.allow("eval")
        system = prompts.reply_system(BOT_NAME, [], None, date.today(), [], emoji=block if allow else "")
        lines = [to_line(m, images=False) for m in c["recent"]] + [to_line(c["latest"], images=False)]
        messages = [{"role": "system", "content": system}] + to_messages(lines)
        raw = to_traditional(_NAME_PREFIX.sub("", await llm.chat(messages)))
        text, sticker = cat.process_reply(raw, eset, allow=allow)
        rs = raw_stats(raw, eset)
        used = bool(CUSTOM.search(text)) or bool(sticker)
        pacing.record("eval", used)
        n_final = len(CUSTOM.findall(text)) + (1 if sticker else 0)
        broken = bool(SHORTCODE.search(text)) or bool(TAG.search(text))
        rows.append({"id": c["id"], "listed": allow, "latest": c["latest"]["content"], "raw": raw, "final": text, "sticker": sticker,
                     "used": used, "n_final": n_final, "broken": broken, **{f"raw_{a}": b for a, b in rs.items()}})
        print(f"[reply {k}/{len(cases)}] used={used} raw_marks={rs['marks']} bad={rs['bad']} | {text[:50]!r}", flush=True)
    n = len(rows)
    used = sum(r["used"] for r in rows)
    listed = [r for r in rows if r["listed"]]
    return {"n": n, "used": used, "used_rate": round(used / n, 3),
            "listed": len(listed), "used_rate_when_listed": round(sum(r["used"] for r in listed) / max(len(listed), 1), 3),
            "broken": sum(r["broken"] for r in rows), "multi_final": sum(r["n_final"] > 1 for r in rows),
            "raw_with_marks": sum(r["raw_marks"] > 0 for r in rows),
            "raw_multi": sum(r["raw_marks"] > 1 for r in rows),
            "raw_bad_names": sum(r["raw_bad"] > 0 for r in rows),
            "stickers": sum(bool(r["sticker"]) for r in rows),
            "used_ids": sorted({m for r in rows for m in re.findall(r"<a?:\w+:(\d+)>", r["final"])}),
            "rows": rows}


# ---------------------------------------------------------------- react
JUDGE_SYSTEM = ("你在檢查一個聊天機器人按的表情反應合不合適。給你一段群組對話、最後一則訊息、她按的表情和那個表情的用法說明。"
                "合適 = 按這個表情，大家看了會覺得貼切、自然（符合訊息的情緒和情境，也符合表情說明的用法）；"
                "表情意思跟訊息對不上、語氣不對、或會讓人不舒服就是不合適。回 JSON：fit，和不超過 30 字的 reason。")
JUDGE_SCHEMA = {"type": "object", "properties": {"fit": {"type": "boolean"}, "reason": {"type": "string", "maxLength": 30}},
                "required": ["fit", "reason"], "additionalProperties": False}


async def judge_fit(llm, case: dict, shown: str, usage: str) -> dict:
    ctx = "\n".join(f"[{m['author']}] {m['content']}" for m in case["recent"][-4:])
    user = (f"對話：\n{ctx}\n\n最後一則：[{case['latest']['author']}] {case['latest']['content']}\n\n"
            f"她按的表情：{shown}\n表情的用法說明：{usage}")
    out = await llm.chat_json([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                              JUDGE_SCHEMA, name="judge", temperature=0.0, max_tokens=300, reasoning_effort="none")
    return out or {"fit": False, "reason": "judge_failed"}


async def eval_react(llm, eset: cat.EmojiSet, s) -> dict:
    cases = load_cases("react_cases")
    block = eset.react_block()
    rows = []
    for k, c in enumerate(cases, 1):
        recent = [to_line(m, images=False) for m in c["recent"]]
        latest = to_line(c["latest"], images=False)
        d = await chime.decide(llm, recent=recent, latest=latest, bot_name=BOT_NAME, attention=False,
                               context_lines=s.chime_context_lines, max_tokens=s.chime_max_tokens,
                               use_logprobs=False, emoji_block=block)
        is_name, real = eset.react_name(d.emoji) if d.emoji else (False, "")
        valid = d.should_react(0.0) and bool(real if is_name else d.emoji)
        row = {"id": c["id"], "label": c["label"], "action": d.action, "emoji": d.emoji, "valid": valid,
               "ms": d.ms, "latest": c["latest"]["content"]}
        if valid:
            if is_name:
                ent = eset._by_key[("emoji", real.lower())]
                shown, usage = f":{real}:", ent.desc
            else:
                shown, usage = d.emoji, "一般表情符號"
            j = await judge_fit(llm, c, shown, usage)
            row.update(fit=bool(j.get("fit")), judge_reason=j.get("reason", ""), usage=usage)
        else:
            row["fit"] = False
        rows.append(row)
        print(f"[react {k}/{len(cases)}] {c['label']} {d.action} {d.emoji!r} valid={valid} fit={row['fit']} {d.ms}ms",
              flush=True)
    n = len(rows)
    reacts = [r for r in rows if r["valid"]]
    fit = sum(r["fit"] for r in rows)
    return {"n": n, "react": len(reacts), "fit": fit, "fit_rate": round(fit / n, 3), "fit_of_reacts": round(fit / max(len(reacts), 1), 3),
            "judge_failed": sum(r.get("judge_reason") == "judge_failed" for r in rows),
            "actions": {a: sum(r["action"] == a for r in rows) for a in chime.ACTIONS},
            "group_emoji": sum(1 for r in reacts if eset.react_name(r["emoji"])[0]),
            "speed": speed([r["ms"] for r in rows]),
            "human_check": [{k: r[k] for k in ("id", "latest", "emoji", "usage", "fit", "judge_reason")}
                            for r in reacts[:5]],
            "rows": rows}


# ---------------------------------------------------------------- voice
FIXED = ["好喔 <:name:1111> 笑死", "<a:dance:2222>哈", "對 :name: 啊", "晚安 [貼圖:abc]",
         "[貼圖：abc]", "12:30:45 開會 <:a:1> :b:"]


def eval_voice(reply_rows: list[dict]) -> dict:
    texts = FIXED + [r["raw"] for r in reply_rows] + [r["final"] for r in reply_rows]
    bad = [t for t in texts if re.search(r"<a?:|\[貼圖|［貼圖|<a?:\w+:\d+", speakable(t)) or SHORTCODE.search(speakable(t))]
    return {"n": len(texts), "with_markup_left": len(bad), "examples": bad[:3]}


# ---------------------------------------------------------------- main
async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--only", action="append", choices=["reply", "react", "voice"])
    ap.add_argument("--catalog", type=Path, default=None)
    ap.add_argument("--out", default="eval_emoji")
    args = ap.parse_args()
    if args.prepare:
        prepare()
        return
    s = load_settings()
    path = args.catalog or (ROOT / "data" / "emoji" / f"{s.profile_fallback_guild}.json")
    eset = cat.load_set(path, require_reviewed=False)  # 評估直接指到圖鑑，不看 reviewed
    if eset is None:
        raise SystemExit(f"圖鑑讀不到或沒有可用的項目：{path}")
    print(f"圖鑑 {path.name}：可用 {len(eset)} 張；reply 清單 {len(eset.prompt_block().splitlines())} 行，"
          f"react 清單 {len(eset.react_block().splitlines())} 行")
    which = args.only or ["reply", "react", "voice"]
    llm = make_llm(s)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    try:
        if "reply" in which or "voice" in which:
            result["reply"] = await eval_reply(llm, eset, s)
        if "react" in which:
            result["react"] = await eval_react(llm, eset, s)
        if "voice" in which:
            result["voice"] = eval_voice(result["reply"]["rows"])
    finally:
        await llm.aclose()
        (RESULTS / f"{args.out}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in result.items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print("結果：", RESULTS / f"{args.out}.json")


if __name__ == "__main__":
    asyncio.run(main())
