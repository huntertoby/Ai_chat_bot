"""D-37 評估：先回話、再依回話挑表情（開發規格 6.14）。不需要 Discord，只用 LM Studio。

    python -m scripts.eval_emoji_pick --out pick_1 --html data/emoji/D37-檢查.html
    python -m scripts.eval_emoji_pick --out pick_2

題目＝data/eval_c1/reply_cases.jsonl 的 50 則真實片段（不進 repo）＋下面 SYNTH 的佔位題（被嘴、嘲諷的誘因、認真問題、一般附和）。
每題：照正式流程產生回話（reply system 沒有表情清單、R3 壞掉檢查；不帶檢索工具）→ emoji_pick 挑表情（剩餘時間照 D-38 算）→ 模型評分。
報：
- 配的比例：照 EmojiPacing 模擬同一個頻道依序回（正式流程會送出的比例，目標 10%～30%）；另外報每題都問時模型選配的比例。
- 壞名稱（模型寫了圖鑑沒有的名稱）、一則超過一個（目標都是 0）。
- 挑表情多花的秒數（中位數、最大）、總時間超過 10 秒的比例（回話＋挑表情；正式流程有工具呼叫會更久）。
- 模型評分：回話分類（sarcasm／comeback／agree／serious／other）＋表情合不合適；微笑類配嘲諷、反諷的回話算合適。
--html：使用者自評檢查頁（說話者換成佔位名；data/ 不進 repo）。
--catalog 直接指到圖鑑檔，不看 reviewed（預設 PROFILE_FALLBACK_GUILD 的圖鑑）。
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import time
from datetime import date
from pathlib import Path

from yuki import emoji_catalog as cat, emoji_pick, prompts
from yuki.brain import checked_chat
from yuki.config import ROOT, load_settings

from ._eval_b_common import BOT_NAME, make_llm, speed, to_line
from .eval_emoji import CUSTOM, RESULTS, load_cases

# 佔位題（不是真實聊天）：確保涵蓋被嘴回嘴、嘲諷／反諷的誘因、認真問題、一般附和
SYNTH = [
    # 被嘴（同 R2 小評估的被嘴題）
    ("taunt", "阿明", ["會把公差都幫你們做了", "沒辦法出公差混時間"], "你有什麼用"),
    ("taunt", "小傑", ["今天排位連輸五場", "打野都不來"], "你這種AI連幫我上分都不會 廢物"),
    ("taunt", "小美", ["剛剛去吃了一家新開的拉麵", "湯頭有夠鹹"], "你講話好像客服喔 笑死"),
    ("taunt", "阿凱", ["明天要交報告", "還沒寫"], "問你也沒用啦 你只會說廢話"),
    ("taunt", "阿明", ["我新買的鍵盤到了", "紅軸打起來好爽"], "你是不是又在裝懂"),
    ("taunt", "小傑", ["週末要不要去看電影"], "你的腦袋大概只有512MB吧"),
    ("taunt", "小美", ["今天被主管念", "超煩"], "你回這麼慢是在當機喔"),
    # 嘲諷／反諷的誘因：自誇、明顯的雷操作
    ("bait", "阿明", ["剛剛那場我0殺10死"], "Yuki 你看我是不是很強"),
    ("bait", "小傑", ["我今天又遲到一小時"], "Yuki 我是不是全公司最準時的人"),
    ("bait", "阿凱", ["我把薪水全部拿去抽卡了", "一隻都沒抽到"], "Yuki 你說我理財是不是很厲害"),
    ("bait", "小美", ["我減肥第一天", "剛吃完雞排跟珍奶"], "Yuki 我這樣是不是很自律"),
    ("bait", "阿明", ["我又把鑰匙鎖在家裡了", "這個月第三次"], "Yuki 我是不是很細心"),
    ("bait", "小傑", ["我開大招打到空氣"], "Yuki 我那招是不是神操作"),
    # 認真問題
    ("serious", "小美", [], "Yuki 我最近一直失眠 有什麼方法嗎"),
    ("serious", "阿凱", [], "Yuki 你知道國慶日是幾月幾號嗎"),
    ("serious", "阿明", ["我阿嬤住院了"], "Yuki 我好擔心"),
    ("serious", "小傑", [], "Yuki 筆電跟桌機哪個適合寫程式"),
    # 一般附和、好笑的事
    ("chat", "小美", ["我家的貓剛剛把整杯水打翻在鍵盤上"], "Yuki 你看我家的貓是不是很皮"),
    ("chat", "阿凱", ["終於放假了"], "Yuki 我要睡到中午"),
    ("chat", "阿明", ["我剛剛走路撞到電線桿"], "Yuki 好痛"),
]


def cases() -> list[dict]:
    out = load_cases("reply_cases")
    for k, (label, who, before, latest) in enumerate(SYNTH, 1):
        out.append({"id": f"synth_{label}_{k}", "label": label,
                    "recent": [{"author": who, "content": t} for t in before],
                    "latest": {"author": who, "content": latest}})
    return out


JUDGE_SYSTEM = (
    "你在檢查一個聊天機器人回話後面附的群組表情合不合適。給你：對方的訊息、她的回話、她附的表情（或不配）和那個表情的用法說明。\n"
    "先判斷她這則回話屬於哪一類（category）：sarcasm（嘲諷、反諷、假裝客氣、陰陽怪氣）、comeback（被嘴之後回嘴或自嘲）、"
    "agree（附和、跟著鬧、一起笑）、serious（認真回答、講事實、給建議、關心）、other（其他）。\n"
    "再判斷表情合不合適（fit）：表情要跟她回話的語氣對得上、大家看了覺得貼切自然才算合適；意思對不上、語氣不對、會讓人不舒服就是不合適。"
    "這個群組常把微笑類的表情拿來嘲諷、反諷，所以微笑類配在嘲諷、反諷的回話上算合適。"
    "如果她選不配，fit 填 true。回 JSON：category、fit，和不超過 30 字的 reason。")
JUDGE_SCHEMA = {"type": "object", "properties": {
    "category": {"type": "string", "enum": ["sarcasm", "comeback", "agree", "serious", "other"]},
    "fit": {"type": "boolean"}, "reason": {"type": "string", "maxLength": 30}},
    "required": ["category", "fit", "reason"], "additionalProperties": False}


async def judge(llm, said: str, reply: str, shown: str, usage: str) -> dict:
    user = f"對方的訊息：{said}\n她的回話：{reply}\n她附的表情：{shown}\n表情的用法說明：{usage}"
    out = await llm.chat_json([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                              JUDGE_SCHEMA, name="judge", temperature=0.0, max_tokens=300, reasoning_effort="none")
    return out or {"category": "other", "fit": False, "reason": "judge_failed"}


async def run(llm, eset: cat.EmojiSet) -> dict:
    items = cases()
    pacing = cat.EmojiPacing()  # 跟 bot 一樣：同一個頻道依序回，用過之後接下來 2 則不挑
    rows = []
    for k, c in enumerate(items, 1):
        lines = [to_line(m, images=False) for m in c["recent"]] + [to_line(c["latest"], images=False)]
        system = prompts.reply_system(BOT_NAME, date.today(), [])
        started = time.monotonic()
        raw = await checked_chat(llm.chat, system, lines, BOT_NAME)
        reply_ms = round((time.monotonic() - started) * 1000)
        text, sticker = cat.process_reply(raw, eset)  # 安全網：模型自己在文字裡寫的表情
        already = bool(sticker) or bool(CUSTOM.search(text))
        said = c["latest"]["content"]
        # 每題都問（才能評分）；pacing 不允許的那題正式流程不會挑，送出時不算
        p = await emoji_pick.pick(llm, eset, said, text, bot_name=BOT_NAME, timeout=emoji_pick.remaining(started))
        allow = pacing.allow("eval")
        picked = p.entry is not None and not already
        used = allow and (picked or already)
        pacing.record("eval", used)
        final, fsticker = text, (sticker if allow else None)
        if allow and picked:
            if p.entry.kind == "sticker":
                fsticker = p.entry.id
            else:
                final = f"{text} {p.entry.render()}"
        n_final = len(CUSTOM.findall(final)) + (1 if fsticker else 0)
        shown = ((f"[貼圖:{p.entry.label}]" if p.entry.kind == "sticker" else f":{p.entry.label}:")
                 if p.entry else "不配")
        j = await judge(llm, said, text, shown, p.entry.desc if p.entry else "（沒有附表情）")
        rows.append({"id": c["id"], "label": c["label"], "author": c["latest"]["author"], "latest": said,
                     "reply": text, "raw_reply": raw, "already": already, "pick": p.record(),
                     "picked_id": p.entry.id if p.entry else "", "picked_label": shown,
                     "picked_kind": p.entry.kind if p.entry else "", "allow": allow, "used": used,
                     "final": final, "n_final": n_final, "bad_name": p.reason == "unknown",
                     "reply_ms": reply_ms, "pick_ms": p.ms, "total_ms": reply_ms + p.ms,
                     "category": j.get("category", "other"), "fit": bool(j.get("fit")),
                     "judge_reason": j.get("reason", "")})
        print(f"[pick {k}/{len(items)}] {c['label']} {shown} ({p.reason or 'ok'}) allow={allow} "
              f"cat={j.get('category')} fit={j.get('fit')} reply={reply_ms}ms pick={p.ms}ms", flush=True)
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    picked = [r for r in rows if r["picked_id"]]
    used_picked = [r for r in rows if r["used"] and r["picked_id"]]
    budget = emoji_pick.TEXT_REPLY_BUDGET_SEC * 1000
    pick_ms = [r["pick_ms"] for r in rows if r["pick"].get("reason") != "no_time"]
    return {
        "n": n,
        "used_rate_with_pacing": round(sum(r["used"] for r in rows) / n, 3),
        "picked_rate_every_time": round(len(picked) / n, 3),
        "bad_names": sum(r["bad_name"] for r in rows),
        "multi": sum(r["n_final"] > 1 for r in rows),
        "already_in_reply": sum(r["already"] for r in rows),
        "reasons": {x: sum((r["pick"].get("reason") or "ok") == x for r in rows)
                    for x in ("ok", "none", "unknown", "bad_json", "timeout", "error", "no_time")},
        "fit_of_picked": round(sum(r["fit"] for r in picked) / max(len(picked), 1), 3),
        "fit_of_sent": round(sum(r["fit"] for r in used_picked) / max(len(used_picked), 1), 3),
        "judge_failed": sum(r["judge_reason"] == "judge_failed" for r in rows),
        "by_category": {c: {"n": sum(r["category"] == c for r in rows),
                            "picked": sum(r["category"] == c and bool(r["picked_id"]) for r in rows),
                            "fit": sum(r["category"] == c and bool(r["picked_id"]) and r["fit"] for r in rows)}
                        for c in sorted({r["category"] for r in rows})},
        "by_label": {l: {"n": sum(r["label"] == l for r in rows),
                         "picked": sum(r["label"] == l and bool(r["picked_id"]) for r in rows)}
                     for l in sorted({r["label"] for r in rows})},
        "pick_speed": speed(pick_ms),
        "reply_speed": speed([r["reply_ms"] for r in rows]),
        "total_over_10s": round(sum(r["total_ms"] > budget for r in rows) / n, 3),
        "reply_over_10s": round(sum(r["reply_ms"] > budget for r in rows) / n, 3),
        "rows": rows,
    }


PLACEHOLDERS = ["阿明", "小傑", "小美", "阿凱"]


def check_page(rows: list[dict], urls: dict[str, str]) -> str:
    """使用者自評檢查頁：一題一列，合適／不合適，存 localStorage，可匯出。說話者一律換成佔位名。"""
    esc = html.escape
    names: dict[str, str] = {}

    def alias(a: str) -> str:
        if a in PLACEHOLDERS:
            return a
        if a not in names:
            names[a] = f"成員{len(names) + 1}"
        return names[a]

    body = []
    for i, r in enumerate(rows, 1):
        if r["picked_id"]:
            kind = "貼圖" if r["picked_kind"] == "sticker" else "表情"
            img = f'<img src="{esc(urls.get(r["picked_id"], ""))}" alt="{kind}" height="48"><br><small>{kind}</small>'
        else:
            img = '<span class="none">不配</span>'
        body.append(
            f'<tr data-id="{esc(r["id"])}"><td>{i}</td><td><b>{esc(alias(r["author"]))}</b>：{esc(r["latest"])}</td>'
            f'<td>{esc(r["reply"])}</td><td class="img">{img}</td>'
            f'<td class="btns"><button data-v="ok">合適</button><button data-v="bad">不合適</button></td></tr>')
    rows_html = "\n".join(body)
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>表情挑選檢查</title>
<style>
body{{font-family:sans-serif;margin:1em;background:#fff;color:#222}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:6px 8px;vertical-align:middle}} th{{background:#eee}}
td.img{{text-align:center;width:90px}} .none{{color:#888}} td.btns{{white-space:nowrap}}
button{{padding:4px 10px;margin:2px;border:1px solid #999;background:#f7f7f7;border-radius:4px;cursor:pointer}}
tr.ok button[data-v=ok]{{background:#3a3;color:#fff}} tr.bad button[data-v=bad]{{background:#c33;color:#fff}}
#bar{{position:sticky;top:0;background:#fff;padding:6px 0;z-index:2}} textarea{{width:100%;height:8em}}
</style></head><body>
<h1>表情挑選檢查（D-37）</h1>
<p>每一列是：對方說的話、Yuki 的回話、她挑的表情（或「不配」）。請按「合適」或「不合適」；「不配」那列是在問這句不配表情對不對。
評完按「匯出結果」，把下面出現的文字整段複製給 PM。結果存在這台電腦的瀏覽器裡，關掉再開也還在（再按一次同一個按鈕可以取消）。</p>
<div id="bar"><button id="export">匯出結果</button> <span id="count"></span></div>
<textarea id="out" hidden readonly></textarea>
<table><thead><tr><th>#</th><th>對方說</th><th>Yuki 的回話</th><th>表情</th><th>評分</th></tr></thead>
<tbody>
{rows_html}
</tbody></table>
<script>
const KEY = "d37-check";
let st = {{}};
try {{ st = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ st = {{}}; }}
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}} }}
function paint() {{
  let n = 0;
  const trs = document.querySelectorAll("tbody tr");
  trs.forEach(tr => {{ const v = st[tr.dataset.id]; tr.className = v || ""; if (v) n++; }});
  document.getElementById("count").textContent = "已評 " + n + " / " + trs.length;
}}
document.querySelectorAll("tbody button").forEach(b => b.addEventListener("click", () => {{
  const id = b.closest("tr").dataset.id;
  if (st[id] === b.dataset.v) delete st[id]; else st[id] = b.dataset.v;
  save(); paint();
}}));
document.getElementById("export").addEventListener("click", () => {{
  const out = document.getElementById("out");
  out.hidden = false; out.value = JSON.stringify(st); out.select();
}});
paint();
</script></body></html>
"""


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=None)
    ap.add_argument("--out", default="pick")
    ap.add_argument("--html", type=Path, default=None)
    args = ap.parse_args()
    s = load_settings()
    path = args.catalog or (ROOT / "data" / "emoji" / f"{s.profile_fallback_guild}.json")
    eset = cat.load_set(path, require_reviewed=False)
    if eset is None:
        raise SystemExit(f"圖鑑讀不到或沒有可用的項目：{path}")
    print(f"圖鑑：可用 {len(eset)} 張；挑表情清單 {len(emoji_pick.catalog_lines(eset).splitlines())} 行")
    llm = make_llm(s)
    RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        result = await run(llm, eset)
    finally:
        await llm.aclose()
    (RESULTS / f"{args.out}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.html:
        items, _ = cat._items_of(path)
        urls = {str(i["id"]): i.get("url", "") for i in items}
        args.html.write_text(check_page(result["rows"], urls), encoding="utf-8")
        print("檢查頁：", args.html)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=1))
    print("結果：", RESULTS / f"{args.out}.json")


if __name__ == "__main__":
    asyncio.run(main())
