"""LLM 速度測試：用接近真實大小的 prompt（4 份人物檔案 + 20 則對話，約 5400 token），量單一請求和同時 2／4 個請求。

    python -m scripts.bench_llm                # 用 .env 的模型
    python -m scripts.bench_llm <模型 ID>
"""
import asyncio, time, httpx, glob, os, sys
from datetime import date
from yuki import prompts
from yuki.config import load_settings
_s = load_settings()
URL = _s.lmstudio_base_url.rstrip("/") + "/chat/completions"; M = sys.argv[1] if len(sys.argv) > 1 else _s.llm_model
files = sorted(glob.glob("data/profiles/*/*.md"), key=os.path.getsize, reverse=True)[:4]
profiles = [(f"成員{i}", open(f, encoding="utf-8").read()[:1500]) for i, f in enumerate(files)]
system = prompts.reply_system("Yuki", ["群組最近在打世界賽"] * 5, [], date.today(), profiles, web=False)
hist = [{"role": "user", "content": f"[成員{i%4}] 今天打LoL又輸了 第{i}場 隊友真的很爛 上路一直送"} for i in range(19)]
def msgs(q): return [{"role": "system", "content": system}] + hist + [{"role": "user", "content": f"[hunterchiao] yuki {q}"}]
async def one(c, q, max_tokens=150):
    t = time.perf_counter()
    r = (await c.post(URL, json={"model": M, "messages": msgs(q), "max_tokens": max_tokens, "temperature": 0.7,
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)).json()
    if "usage" not in r: return time.perf_counter() - t, 0, 0, str(r)[:200]
    u = r["usage"]; return time.perf_counter() - t, u["prompt_tokens"], u["completion_tokens"], ""
async def main():
    async with httpx.AsyncClient() as c:
        await one(c, "暖機")
        for label, n in [("單一請求", 1), ("同時 2 個", 2), ("同時 4 個", 4)]:
            t = time.perf_counter()
            res = await asyncio.gather(*(one(c, f"你覺得我是不是很衰 第{i}個問題") for i in range(n)))
            wall = time.perf_counter() - t; toks = sum(r[2] for r in res)
            errs = [r[3] for r in res if r[3]]
            if errs: print(label, "錯誤:", errs[0])
            print(f"{label}: 總耗時 {wall:.1f}s，各自 {[round(r[0],1) for r in res]}s，prompt {res[0][1]} token，生成 {toks} token（{toks/wall:.0f} token/s）")
        d, p, ct, _ = await one(c, "寫一段很長的打LoL心得 至少300字", max_tokens=400)
        print(f"長回覆：{ct} token / {d:.1f}s = {ct/d:.0f} token/s")
asyncio.run(main())
