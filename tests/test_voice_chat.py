"""語音對話（開發規格 6.0）：切句、要不要回、念出來的文字、只聽名單上的人、測試伺服器規則。"""
import re
from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np

from yuki import prompts
from yuki.brain import Brain
from yuki.profile.auto import ProfileUpdater
from yuki.profile.store import ProfileFiles
from yuki.rag.decompose import Person
from yuki.voice.asr import asr_context
from yuki.voice.segmenter import Segmenter
from yuki.voice.session import ListenSink, should_reply
from yuki.voice.tts import sentences, speakable

FRAME = 0.02  # Discord 一個封包 20ms


def frame(loud: bool) -> bytes:
    """48kHz 雙聲道 16-bit 的 20ms PCM。"""
    n = 960
    x = (np.sin(np.arange(n) * 2 * np.pi * 220 / 48000) * (0.3 if loud else 0)).astype(np.float32)
    return (np.repeat(x, 2) * 32767).astype(np.int16).tobytes()


def fake_vad():
    return lambda chunk: float(np.sqrt((chunk ** 2).mean()) > 0.05)


class Clock:
    def __init__(self):
        self.t = 100.0


def feed(seg, clock, uid, seconds, loud):
    for _ in range(round(seconds / FRAME)):
        clock.t += FRAME
        seg.feed(uid, frame(loud), clock.t)


def seg_(**kw):
    return Segmenter(fake_vad, silence_ms=700, **kw), Clock()


def test_sentence_ends_after_silence_frames():
    seg, c = seg_()
    feed(seg, c, "1", 1.0, True)
    assert seg.tick(c.t) == []  # 還在講
    feed(seg, c, "1", 1.0, False)
    out = seg.tick(c.t)
    assert len(out) == 1 and out[0].user_id == "1"
    assert 1.0 <= out[0].seconds <= 1.9  # 1 秒人聲 + 0.7 秒內的尾巴（+ 開口前一點點）


def test_sentence_ends_when_packets_stop():
    """Discord 靜音時不送封包：靠 tick 判斷講完。"""
    seg, c = seg_()
    feed(seg, c, "1", 1.0, True)
    assert seg.tick(c.t + 0.5) == []
    assert len(seg.tick(c.t + 0.8)) == 1


def test_short_noise_is_dropped():
    seg, c = seg_()
    feed(seg, c, "1", 0.2, True)
    assert seg.tick(c.t + 1) == []


def test_long_speech_is_cut_at_max_length():
    seg, c = seg_(max_ms=3000)
    feed(seg, c, "1", 7.0, True)
    assert len(seg.tick(c.t)) == 2


def test_two_people_are_segmented_separately():
    seg, c = seg_()
    for _ in range(50):  # 兩人同時講 1 秒
        c.t += FRAME
        seg.feed("1", frame(True), c.t)
        seg.feed("2", frame(True), c.t)
    assert sorted(u.user_id for u in seg.tick(c.t + 1)) == ["1", "2"]


def test_talking_ms_for_barge_in():
    seg, c = seg_()
    assert seg.talking_ms("1") == 0
    feed(seg, c, "1", 0.6, True)
    assert seg.talking_ms("1") >= 500


WAKE = re.compile("yuki", re.I)


def test_should_reply():
    kw = dict(wake=WAKE, attention_until=0, now=10)
    assert should_reply("今天好累", humans=1, **kw)  # 只有你跟她：每句都回
    assert not should_reply("今天好累", humans=3, **kw)  # 有別人：要叫她
    assert should_reply("Yuki 你覺得呢", humans=3, **kw)
    assert should_reply("那你覺得呢", humans=3, wake=WAKE, attention_until=20, now=10)  # 叫完後的注意時間
    assert not should_reply("  ", humans=1, **kw)


def test_speakable_and_sentences():
    assert speakable("好啊XD 😂 看這個 https://x.com/a （小聲）") == "好啊 看這個"
    assert sentences("對啊。我也覺得！你呢？") == ["對啊。我也覺得！你呢？"]  # 太短的句子併起來
    assert sentences("我覺得還好啦。你自己決定就好！") == ["我覺得還好啦。", "你自己決定就好！"]
    assert sentences("😂😂") == []
    # 第一句太長：在逗號切開，先念短的
    assert sentences("知道啊，阿明嘛，某大學資工那個。幹嘛？") == ["知道啊，阿明嘛，", "某大學資工那個。幹嘛？"]


def test_sink_only_hears_listed_users():
    fed = []
    sink = ListenSink(SimpleNamespace(feed=lambda uid, pcm, now: fed.append(uid)), {"1"})
    data = SimpleNamespace(pcm=frame(True))
    sink.write(SimpleNamespace(id=1, bot=False), data)
    sink.write(SimpleNamespace(id=2, bot=False), data)  # 不在名單：丟掉
    sink.write(None, data)  # 還不知道是誰：丟掉
    assert fed == ["1"]


def test_asr_context():
    ctx = asr_context(["阿明", "hunterchiao", "Yuki"], ["LoL", "Yuki"])
    assert "阿明、hunterchiao、Yuki、LoL" in ctx  # 去重


def test_spoken_prompt():
    assert "語音頻道裡用講的" in prompts.reply_system("Yuki", [], [], spoken=True)
    assert "語音頻道裡用講的" not in prompts.reply_system("Yuki", [], [])


def test_profile_falls_back_to_main_guild(tmp_path):
    files = ProfileFiles(tmp_path)
    files.write("main", "2", "# Toby\n喜歡打 LoL")
    s = SimpleNamespace(profile_enabled=True, profile_max_people=4, profile_prompt_chars=500,
                        profile_fallback_guild="main")
    fake = SimpleNamespace(settings=s, profiles=files, history=defaultdict(deque))
    out = Brain.pick_profiles(fake, "test", "c", Person("2", "Toby"))
    assert out == [("Toby", "喜歡打 LoL")]
    s.profile_fallback_guild = ""
    assert Brain.pick_profiles(fake, "test", "c", Person("2", "Toby")) == []


def test_profile_updater_skips_test_guilds():
    import sqlite3
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE messages (guild_id TEXT)")
    db.executemany("INSERT INTO messages VALUES (?)", [("main",), ("test",)])
    up = ProfileUpdater(None, None, None, db, lambda: True, lambda g: set(), skip_guilds={"test"})
    assert up.guilds() == ["main"]


def test_owner_rules_only_when_owner_set():
    p = prompts.reply_system("Yuki", [], [], owner="hunterchiao")
    assert "hunterchiao 是把你弄進群組" in p and "一起噴他" in p
    assert "關於" not in prompts.reply_system("Yuki", [], []).split("規則：")[0]


def test_member_names_in_search_query():
    from yuki.brain import names_in
    names = {"阿明", "hunterchiao", "A"}
    assert names_in("阿明 某大學 資工系", names) == ["阿明"]
    assert names_in("HunterChiao 是誰", names) == ["hunterchiao"]  # 不分大小寫
    assert names_in("台北 天氣", names) == []
    assert names_in("A 牌 顯示卡", names) == []  # 一個字的名字不算，免得誤擋


def test_voice_reply_rules_are_short():
    p = prompts.reply_system("Yuki", [], [], spoken=True, web=True)
    assert "30 個字以內" in p and "不要拿群組成員的名字" in p
