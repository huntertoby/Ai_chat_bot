"""scripts/eval_text_wait.py 的總等待算法（不需要模型）。"""
from scripts import eval_text_wait as w


def test_pick_uses_only_remaining_budget():
    assert w.with_pick(2000, 300) == 2300
    assert w.with_pick(9500, 300) == 9500        # 剩不到 1 秒：不挑
    assert w.with_pick(8800, 3000) == 10000      # 挑表情超時：最多用到 10 秒


def test_combine_adds_debounce_and_all_pairs():
    tot = w.combine([500, 5000], [2000], [300], 4000)
    assert sorted(tot) == [6800, 11300]
    s = w.stats(tot)
    assert s["n"] == 2 and s["max_s"] == 11.3 and s["over_10s"] == 0.5
