"""B3 防洗版與節流（yuki/throttle.py）：不需要網路、模型、Discord。"""
from yuki.chat import ChatLine, Person
from yuki.throttle import (Throttle, ThrottleConfig, is_command, is_flood, is_noise_text, is_repeat,
                           strip_decor)

AMING = Person("1", "阿明")
XIAOJIE = Person("2", "小傑")


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# ---- 雜訊 ----
def test_noise_short_reactions_and_emoji_only():
    for t in ["", "   ", "ok", "OK", "笑死", "我去", "哈哈哈哈", "😂😂", "<:smile:123456>", "<a:dance:99> ！！",
              "...", "?", "好", "@阿明 ok", "睡著了", "666"]:
        assert is_noise_text(t), t


def test_noise_keeps_real_content():
    for t in ["今天要不要打 LoL", "我剛考上了", "這個要怎麼設定", "有人知道這家店嗎"]:
        assert not is_noise_text(t), t


def test_noise_max_chars_is_configurable():
    assert is_noise_text("好耶", max_chars=2)
    assert not is_noise_text("好耶", max_chars=1)


def test_strip_decor_removes_emoji_punct_mentions():
    assert strip_decor("<@123> 你好！！😂 <:a:1>") == "你好"


def test_command_detection():
    assert is_command("/來語音")
    assert is_command("!play abc")
    assert is_command(".help")
    assert not is_command("...")
    assert not is_command("..什麼啦")
    assert not is_command("!!")
    assert not is_command("/ 不是指令")
    assert not is_command("一般的話")
    assert not is_command("")


# ---- 大量重複 ----
def test_flood_repeated_chars():
    assert is_flood("a" * 10)
    assert not is_flood("哈" * 12 + "好")  # 笑聲重複不算洗版
    assert not is_flood("哈" * 9)
    assert not is_flood("a b " * 20)  # 不連續
    assert is_flood("x" * 5, repeat_chars=5)


def test_flood_repeated_mentions():
    assert is_flood("<@1> <@1> <@1> 起床")
    assert is_flood("<@!1><@!1><@!1>")
    assert not is_flood("<@1> <@1> 起床")
    assert not is_flood("<@1> <@2> <@3>")


# ---- 對話紀錄去重 ----
def test_is_repeat_same_author_same_text():
    a = ChatLine(AMING, "在嗎")
    assert is_repeat(a, ChatLine(AMING, "在嗎 "))
    assert not is_repeat(a, ChatLine(XIAOJIE, "在嗎"))
    assert not is_repeat(a, ChatLine(AMING, "在啊"))
    assert not is_repeat(None, a)
    assert not is_repeat(ChatLine(AMING, "x", is_bot=True), ChatLine(AMING, "x", is_bot=True))


def test_is_repeat_images_differ():
    a = ChatLine(AMING, "（貼了圖片）", images=["u1"])
    assert is_repeat(a, ChatLine(AMING, "（貼了圖片）", images=["u1"]))
    assert not is_repeat(a, ChatLine(AMING, "（貼了圖片）", images=["u2"]))


# ---- 每頻道同時只處理一則、排隊 ----
def test_one_at_a_time_per_channel():
    th = Throttle(clock=Clock())
    assert th.try_start("c1")
    assert not th.try_start("c1")
    assert th.is_busy("c1")
    assert th.try_start("c2")  # 別的頻道不受影響
    th.finish("c1")
    assert not th.is_busy("c1")
    assert th.try_start("c1")


def test_queue_drain_merges_and_clears():
    th = Throttle(clock=Clock())
    th.enqueue("c1", "m1")
    th.enqueue("c1", "m2")
    assert th.drain("c1") == ["m1", "m2"]
    assert th.drain("c1") == []


def test_queue_cap_drops_oldest():
    th = Throttle(ThrottleConfig(queue_max=2), clock=Clock())
    for i in range(4):
        th.enqueue("c1", i)
    assert th.drain("c1") == [2, 3]


def test_drop_queued_by_predicate():
    th = Throttle(clock=Clock())
    for item in [("a", 1), ("b", 2), ("a", 3)]:
        th.enqueue("c1", item)
    assert th.drop_queued("c1", lambda x: x[0] == "a") == 2
    assert th.drain("c1") == [("b", 2)]


# ---- 頻道冷卻 ----
def test_channel_cooldown_after_finish():
    clk = Clock()
    th = Throttle(ThrottleConfig(cooldown_sec=5), clock=clk)
    assert not th.in_cooldown("c1")  # 從沒回過
    th.try_start("c1")
    th.finish("c1")
    assert th.in_cooldown("c1")
    clk.t += 4.9
    assert th.in_cooldown("c1")
    clk.t += 0.2
    assert not th.in_cooldown("c1")
    assert not th.in_cooldown("c2")


# ---- tag 洗版 ----
def test_tag_spam_trips_on_fourth_then_cools_down():
    clk = Clock()
    th = Throttle(ThrottleConfig(tag_limit=3, tag_window_sec=60, tag_cooldown_sec=120), clock=clk)
    assert [th.note_tag("u1") for _ in range(3)] == [True, True, True]
    assert th.note_tag("u1") is False  # 第 4 次
    assert th.tag_cooling("u1")
    clk.t += 119
    assert th.note_tag("u1") is False  # 冷卻中
    clk.t += 2
    assert th.note_tag("u1") is True  # 冷卻結束
    assert not th.tag_cooling("u1")


def test_tag_window_slides_and_users_are_independent():
    clk = Clock()
    th = Throttle(ThrottleConfig(tag_limit=3, tag_window_sec=60), clock=clk)
    for _ in range(3):
        assert th.note_tag("u1")
        clk.t += 25  # 每 25 秒一次，60 秒窗口內最多 3 次
    assert th.note_tag("u1")
    assert th.note_tag("u2")


# ---- 頻道層級熱鬧模式 ----
def _burst_throttle():
    clock = Clock()
    th = Throttle(ThrottleConfig(channel_call_burst=5, channel_call_window_sec=60, channel_burst_mode_sec=120,
                                 channel_call_gap_sec=30), clock=clock)
    return th, clock


def _call_round_robin(th, clock, n, step=2.0):
    out = []
    for i in range(n):
        out.append(th.note_call("c"))
        clock.t += step
    return out


def test_burst_triggered_by_many_people_taking_turns():
    th, clock = _burst_throttle()
    assert all(_call_round_robin(th, clock, 4))      # 前 4 次照常
    assert not th.in_burst("c")
    assert th.note_call("c") is True                 # 第 5 次觸發模式，但這一則照回
    assert th.in_burst("c")
    th.try_start("c")                                # 回覆開始：記下時間
    th.finish("c")
    clock.t += 2
    assert th.note_call("c") is False                # 模式中、離上次回覆不到 gap


def test_burst_collects_and_merges_then_replies_once_per_gap():
    th, clock = _burst_throttle()
    _call_round_robin(th, clock, 5)
    th.try_start("c"); th.finish("c")
    for i in range(3):
        clock.t += 5
        assert th.note_call("c") is False
        th.enqueue("c", f"m{i}")
    assert th.ready_in("c") > 0
    assert th.drain_ready("c") == []                 # 還沒到時間：留在佇列
    clock.t += 30
    assert th.ready_in("c") == 0
    assert th.drain_ready("c") == ["m0", "m1", "m2"]  # 一次取出合併
    clock.t += 1
    assert th.note_call("c") is False                # 取出算一次回覆，又重新計 gap


def test_burst_mode_expires_and_normal_again():
    th, clock = _burst_throttle()
    _call_round_robin(th, clock, 5)
    th.try_start("c"); th.finish("c")
    clock.t += 5
    assert th.note_call("c") is False                # 這次叫會延長模式
    th.enqueue("c", "x")
    clock.t += 119
    assert th.in_burst("c")                          # 延長了
    clock.t += 5
    assert not th.in_burst("c")
    assert th.ready_in("c") == 0
    assert th.drain_ready("c") == ["x"]
    assert th.note_call("c") is True


def test_burst_not_triggered_by_slow_calls():
    th, clock = _burst_throttle()
    assert all(_call_round_robin(th, clock, 10, step=20))  # 60 秒內不到 5 次
    assert not th.in_burst("c")


def test_flood_needs_almost_nothing_left_besides_the_repeats():
    from yuki.throttle import is_flood
    assert not is_flood("我考上了！！！！！！！！！！")
    assert not is_flood("哈哈哈哈哈哈哈哈哈哈")
    assert not is_flood("10000000000")
    assert not is_flood("----------")
    assert not is_flood("wwwwwwwwwwwww")
    assert not is_flood("今天好開心" + "哈" * 20 + "笑死")
    assert is_flood("a" * 10)
    assert is_flood("阿" * 15)
    assert is_flood("嗚" * 12 + "好")  # 剩 1 字
    assert not is_flood("嗚" * 12 + "我今天真的很難過")  # 還有內容
