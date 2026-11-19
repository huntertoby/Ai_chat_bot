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
    msgs = [msg(i, people[i % 4], f"Yuki，問題 {i}", i * 5) for i in range(7)]  # 35 秒內被 7 個人次叫
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
    s = replace(Settings(), chime_context_lines=9)
    assert eval_chime.decide_opts(s)["context_lines"] == 9 and "use_logprobs" not in eval_chime.decide_opts(s)
    monkeypatch.setattr(eval_chime, "THINK_ON_MENTION", None)  # D-29 兩段式也照正式設定
    assert eval_chime.decide_opts(replace(s, chime_think_on_mention=True))["think_on_mention"] is True
    monkeypatch.setattr(eval_chime, "THINK_ON_MENTION", False)
    assert eval_chime.decide_opts(replace(s, chime_think_on_mention=True))["think_on_mention"] is False
    monkeypatch.setattr(eval_chime, "THRESHOLD", None)
    assert eval_chime.threshold(replace(s, chime_in=2, chime_threshold=None)) == 0.6


def test_route_mode_answers_direct_calls_without_asking_the_model(tmp_path, monkeypatch):
    """D-32 --route：最新一則是叫她（@、回覆她、直接叫名字）就直接算「回」；只是提到她的才問模型。"""
    monkeypatch.setattr(eval_chime, "EVAL_DIR", tmp_path)
    base = {"recent": [{"author": "阿明", "content": "今天好熱"}], "tags": []}
    cases = [
        {**base, "id": "a", "label": "talk_to_me", "expect_participate": True, "latest": {"author": "小傑", "content": "Yuki，你覺得呢"}},
        {**base, "id": "b", "label": "talk_to_me", "expect_participate": True, "latest": {"author": "小傑", "content": "@Yuki 評一下"}},
        {**base, "id": "c", "label": "talk_to_me", "expect_participate": True,
         "latest": {"author": "小傑", "content": "對啊", "reply_to": {"author": "Yuki", "is_yuki": True, "text": "x"}}},
        {**base, "id": "d", "label": "about_me", "expect_participate": False, "latest": {"author": "小傑", "content": "Yuki 好吵"}},
        {**base, "id": "e", "label": "about_me", "expect_participate": False, "latest": {"author": "小傑", "content": "我覺得 Yuki 很吵"}},
    ]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")
    s = replace(Settings(), wake_words=["yuki"], chime_in=1)
    llm = QuietLLM()
    out = asyncio.run(eval_chime.eval_chime(llm, s, None, cases_file="t.jsonl", route=True))
    assert out["called_ids"] == ["a", "b", "c"] and llm.calls == 2 and out["accuracy"] == 1.0
    llm2 = QuietLLM()
    plain = asyncio.run(eval_chime.eval_chime(llm2, s, None, cases_file="t.jsonl"))
    assert plain["called_ids"] == [] and llm2.calls == 5   # 不帶 --route：每題都只測插話判斷
