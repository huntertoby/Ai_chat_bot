"""D-32：名字在句首「直接叫她」才一定回；其他提到名字的（句中、句尾、句首接評論）只是在講她。"""
import pytest

from yuki.wake import addressed_elsewhere, calls_other_bot, direct_call, named, wake_pattern

W = wake_pattern(["yuki", "小雪"])
EXIT = ["掰掰", "再見", "退下", "去睡覺", "閉嘴"]


@pytest.mark.parametrize("text", [
    "Yuki，今天要下雨嗎",       # 句首＋逗號
    "yuki, 幫我看一下",
    "Yuki? 人呢",
    "Yuki 你覺得呢",            # 句首＋你
    "yuki妳今天怎麼了",
    "yuki",                     # 只叫名字
    "Yuki～",
    "Yuki！！",
    "Yuki 😭",
    "欸 yuki 你看這個",         # 招呼語＋名字
    "嘿Yuki，在不在",
    "hey yuki, what's up",
    "@Yuki 你說呢",             # 打字打出來的 @（不是真的 tag）也算前面的標點
    "  ，Yuki 你在嗎",
    "yuki 掰掰～",              # 後面是結束字
    "Yuki 閉嘴啦",              # 後面是閉嘴類
    "Yuki 欸",                  # 名字加招呼
    "欸小雪你覺得呢",           # 中文別名
    "小雪，幫我查",
])
def test_direct_call(text):
    assert direct_call(text, W, EXIT), text
    assert named(text, W)


@pytest.mark.parametrize("text", [
    "Yuki 才不是正品",          # 句首但接評論
    "Yuki 超廢",
    "Yuki 到底看不看得懂梗圖",
    "Yuki 會不會傳貼圖",
    "Yuki的回覆好長",
    "我覺得 Yuki 很有料",       # 句中
    "不得不說Yuki你蠻強的",     # 句中（後面接「你」也不算：名字不在句首）
    "這個問題問 yuki 就好",
    "你說對吧 yuki",            # 句尾
    "小雪 根本是複讀機",
    "yuki 我先去吃飯掰掰",      # 結束字夾在句子裡不算
    "yuki @阿明 你看",          # @ 不算名字後面的標點
])
def test_named_but_not_called(text):
    assert not direct_call(text, W, EXIT), text
    assert named(text, W)


def test_english_name_must_match_the_whole_word():
    for text in ("yukimura 你好", "miyuki，你好", "hiyuki 你", "Yuki2 你"):
        assert not direct_call(text, W, EXIT) and not named(text, W), text


def test_no_name_is_neither():
    assert not direct_call("你覺得呢", W, EXIT) and not named("你覺得呢", W)
    assert not direct_call("", W, EXIT) and not direct_call(None, W, EXIT)


def test_asking_if_she_is_there_is_a_direct_call():
    wake = wake_pattern(["yuki"])
    assert direct_call("Yuki 在嗎", wake)
    assert direct_call("yuki在不在", wake)
    assert direct_call("Yuki 在？", wake)
    assert not direct_call("Yuki 在打遊戲的時候很吵", wake)


# ---------- D-34：程式端先擋（不是在跟她講）----------
BOTS = wake_pattern(["Nova", "小助手"])


def blocked(text, *, at_other=False, at_her=False, bots=BOTS):
    return addressed_elsewhere(text, W, mentions_others=at_other, mentions_her=at_her, other_bots=bots)


def test_tagging_someone_else_is_not_talking_to_her():
    assert blocked("<@123> 晚上要不要打", at_other=True) == "at_other"
    assert blocked("<@&5> 集合了", at_other=True) == "at_other"            # 身分組也算


def test_tagging_someone_else_but_naming_her_is_left_to_the_model():
    assert blocked("<@123> 你問 yuki 啦", at_other=True) is None
    assert blocked("<@123> 你看", at_other=True, at_her=True) is None


@pytest.mark.parametrize("text", ["Nova 幫我查明天天氣", "nova，在嗎", "欸 Nova 你覺得呢", "@Nova 你好", "小助手 你好"])
def test_leading_other_bot_name_is_calling_that_bot(text):
    assert blocked(text) == "other_bot"


@pytest.mark.parametrize("text", ["我覺得 Nova 比較聰明", "剛剛 nova 說的對嗎", "Novak 今天贏了", "Nova 跟 yuki 誰比較強"])
def test_other_bot_name_in_the_middle_or_with_her_name_is_not_blocked(text):
    assert blocked(text) is None


def test_reply_to_someone_else_is_not_blocked_here():
    # 回覆別人時 Discord 自動 ping 的不算 @（呼叫端不算進 mentions_others）
    assert blocked("哈哈真的", at_other=False) is None


def test_no_other_bots_on_the_server():
    assert not calls_other_bot("Nova 你好", None)
    assert blocked("Nova 你好", bots=None) is None
