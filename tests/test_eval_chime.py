"""scripts/eval_chime.py 的洗版重播跟正式排程器走同一套（不需要模型：插話判斷用假的 LLM）。"""
import asyncio
import json
from dataclasses import replace

from scripts import eval_chime
from yuki.config import Settings


class QuietLLM:
    def __init__(self):
        self.calls = 0

    async def chat_json(self, messages, schema, **kw):
        self.calls += 1
        return {"reason": "x", "kind": "other", "action": "none", "participate": 0.0, "emoji": "",
                "need_background": []}


def write_case(d, msgs):
    d.mkdir(parents=True)
    (d / "flood_replay.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in msgs),
                                          encoding="utf-8")
    (d / "flood_meta.json").write_text('{"yuki_reply_limit": 99}', encoding="utf-8")


def msg(i, author, content, sec, called=False):
    return {"id": str(i), "author": author, "content": content, "is_yuki": False, "mentions_yuki": called,
            "time": f"2026-11-19 12:{sec // 60:02d}:{sec % 60:02d}"}


def test_flood_replay_uses_burst_mode_and_merges_collected_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_chime, "EVAL_DIR", tmp_path / "eval")
    people = ["阿明", "小傑", "小美", "阿凱"]
    msgs = [msg(i, people[i % 4], f"Yuki 問題 {i}", i * 5) for i in range(7)]  # 35 秒內被 7 個人次叫
    msgs.append(msg(99, "阿明", "今天天氣真的很不錯欸", 50))
    write_case(tmp_path / "eval", msgs)
    s = replace(Settings(), wake_words=["yuki"], chime_in=1)
    llm = QuietLLM()
    out, points, _ = asyncio.run(eval_chime.replay(llm, s, reply_sec=1.0, debounce=4.0))
    c = out["counters"]
    assert c["called"] == 7 and c["held_burst"] == 3     # 第 4 次進熱鬧模式，之後收集起來
    assert c["replies_called"] == 4 and c["replies_queued"] == 1 and c["merged_from"] == 3
    assert out["replies"] == 5 and llm.calls == 1         # 最後那句閒聊有判斷一次插話


def test_eval_uses_production_decide_options_unless_overridden(monkeypatch):
    s = replace(Settings(), chime_use_logprobs=False, chime_context_lines=9)
    monkeypatch.setattr(eval_chime, "USE_LOGPROBS", None)
    assert eval_chime.decide_opts(s)["use_logprobs"] is False and eval_chime.decide_opts(s)["context_lines"] == 9
    monkeypatch.setattr(eval_chime, "USE_LOGPROBS", True)
    assert eval_chime.decide_opts(s)["use_logprobs"] is True
    monkeypatch.setattr(eval_chime, "THINK_ON_MENTION", None)  # D-29 兩段式也照正式設定
    assert eval_chime.decide_opts(replace(s, chime_think_on_mention=True))["think_on_mention"] is True
    monkeypatch.setattr(eval_chime, "THINK_ON_MENTION", False)
    assert eval_chime.decide_opts(replace(s, chime_think_on_mention=True))["think_on_mention"] is False
    monkeypatch.setattr(eval_chime, "THRESHOLD", None)
    assert eval_chime.threshold(replace(s, chime_in=2, chime_threshold=None)) == 0.6
