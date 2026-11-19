"""階段 B 評估腳本共用的小工具（eval_chime.py、eval_background.py）。題目檔在 data/eval_b/（不進 repo）。"""
from __future__ import annotations

import base64
import json
import statistics
from pathlib import Path

from yuki.config import ROOT, Settings, load_settings
from yuki.llm import LMStudioClient
from yuki.chat import ChatLine, Person, ReplyRef
from yuki.vision import _shrink

EVAL_DIR = ROOT / "data" / "eval_b"
RESULT_DIR = EVAL_DIR / "results"
BOT_NAME = "Yuki"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def make_llm(s: Settings | None = None) -> LMStudioClient:
    s = s or load_settings()
    return LMStudioClient(s.lmstudio_base_url, s.llm_model, temperature=s.llm_temperature,
                          max_tokens=s.llm_max_tokens, timeout=s.llm_timeout,
                          reasoning_effort=s.llm_reasoning_effort,
                          json_reasoning_effort=s.llm_json_reasoning_effort, max_concurrency=1)


def image_data_url(rel: str, long_side: int = 512) -> str:
    """題目裡的圖（相對 data/eval_b）縮成長邊 long_side 的 JPEG data URL。"""
    jpeg = _shrink((EVAL_DIR / rel).read_bytes(), long_side)
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def to_line(m: dict, *, images: bool = True) -> ChatLine:
    """題目的訊息 dict → ChatLine（作者名當 ID；Yuki 的是 bot）。images=True 時圖附成 data URL。"""
    name = m["author"]
    is_bot = name == BOT_NAME or bool(m.get("is_yuki"))
    ref = m.get("reply_to")
    reply = ReplyRef(name=ref["author"], is_bot=bool(ref.get("is_yuki")), snippet=ref.get("text", "")) if ref else None
    imgs = [image_data_url(p) for p in m.get("images") or []] if images else []
    return ChatLine(Person(name, name), m["content"], is_bot=is_bot, images=imgs, reply_to=reply)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def speed(ms: list[float]) -> dict:
    return {"n": len(ms), "median_s": round(statistics.median(ms) / 1000, 3) if ms else 0,
            "p90_s": round(pct(ms, 0.9) / 1000, 3), "max_s": round(max(ms) / 1000, 3) if ms else 0}
