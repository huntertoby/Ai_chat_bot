import sqlite3
from datetime import datetime, timedelta

from yuki import prompts
from yuki.voice_presence import VoicePresence

G = "g"
T0 = datetime(2026, 11, 17, 20, 0, 0)
LOL = ("v1", "LoL 開黑")
CHILL = ("v2", "閒聊")


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def vp() -> VoicePresence:
    return VoicePresence(sqlite3.connect(":memory:"))


def test_join_move_leave_events_and_current_state():
    v = vp()
    assert v.update(G, "1", "阿明", None, LOL, at(0)).kind == "join"
    v.update(G, "2", "hunterchiao", None, LOL, at(5))
    ev = v.update(G, "1", "阿明", LOL, CHILL, at(20))
    assert (ev.kind, ev.channel, ev.to_channel) == ("move", "LoL 開黑", "閒聊")
    assert v.update(G, "2", "hunterchiao", LOL, None, at(25)).kind == "leave"
    text = v.describe(G, at(30))
    assert "#閒聊：阿明（已待 10 分鐘）" in text
    assert "LoL 開黑：" not in text  # 沒人的頻道不列
    assert "5 分鐘前 hunterchiao 離開 #LoL 開黑" in text
    assert "10 分鐘前 阿明 從 #LoL 開黑 換到 #閒聊" in text


def test_mute_or_same_channel_is_not_an_event():
    v = vp()
    v.update(G, "1", "阿明", None, LOL, at(0))
    assert v.update(G, "1", "阿明", LOL, LOL, at(1)) is None
    assert v.update(G, "1", "阿明", None, None, at(1)) is None


def test_old_events_drop_out_of_recent_list():
    v = vp()
    v.update(G, "1", "阿明", None, LOL, at(0))
    assert "進了 #LoL 開黑" not in v.describe(G, at(45))  # 超過 30 分鐘
    assert "#LoL 開黑：阿明（已待 45 分鐘）" in v.describe(G, at(45))


def test_today_totals_and_partial_sessions_on_restart():
    v = vp()
    v.update(G, "1", "阿明", None, LOL, at(0))
    v.update(G, "1", "阿明", LOL, None, at(90))
    v.update(G, "2", "小傑", None, LOL, at(60))
    totals = dict(v.today_totals(G, at(120)))
    assert totals == {"阿明": timedelta(minutes=90), "小傑": timedelta(minutes=60)}
    assert "今天在語音待過：阿明 1 小時 30 分、小傑 1 小時" in v.describe(G, at(120))

    # bot 重開：還開著的紀錄先關掉（標成不確定），已經在裡面的人重新登記
    v.reset_guild(G, at(130))
    assert v.describe(G, at(130)).startswith("- 現在沒有人在語音頻道")
    v.present(G, "2", "小傑", *LOL, at(130))
    assert "Yuki 上線前就在" in v.describe(G, at(140))
    assert v.db.execute("SELECT COUNT(*) FROM voice_sessions WHERE partial=1").fetchone()[0] == 2


def test_sessions_before_today_are_clipped_to_midnight():
    v = vp()
    yesterday = T0.replace(hour=23) - timedelta(days=1)
    v.update(G, "1", "阿明", None, LOL, yesterday)
    v.update(G, "1", "阿明", LOL, None, T0.replace(hour=1))
    assert dict(v.today_totals(G, at(0))) == {"阿明": timedelta(hours=1)}


def test_opted_out_users_are_not_recorded_and_can_be_forgotten():
    v = vp()
    v.update(G, "9", "退出的人", None, LOL, at(0), record=False)
    assert v.db.execute("SELECT COUNT(*) FROM voice_sessions").fetchone()[0] == 0
    v.update(G, "1", "阿明", None, LOL, at(0))
    assert v.forget_user(G, "1") == 1


def test_voice_section_only_when_given():
    assert "【語音頻道現況】" in prompts.reply_system("Yuki", voice="- #LoL：阿明")
    assert "【語音頻道現況】" not in prompts.reply_system("Yuki")
