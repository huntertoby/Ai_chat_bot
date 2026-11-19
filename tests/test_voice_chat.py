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
    """測試伺服器沒有這個人的檔案：借主伺服器的；沒設主伺服器就沒有。"""
    from dataclasses import replace

    from yuki.config import Settings
    from yuki.memory.store import MemoryStore

    def brain(fallback):
        st = replace(Settings(), profile_enabled=True, profile_max_people=4, profile_prompt_chars=500,
                     profile_fallback_guild=fallback, profile_dir=tmp_path)
        return Brain(st, SimpleNamespace(), MemoryStore(":memory:", "fake"), None, None)

    ProfileFiles(tmp_path).write("main", "2", "# Toby\n喜歡打 LoL")
    assert brain("main").pick_profiles("test", "c", Person("2", "Toby")) == [("Toby", "喜歡打 LoL")]
    assert brain("").pick_profiles("test", "c", Person("2", "Toby")) == []


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


def test_names_in_finds_names_in_text():
    from yuki.brain import names_in
    names = {"阿明", "hunterchiao", "A"}
    assert names_in("阿明 某大學 資工系", names) == ["阿明"]
    assert names_in("HunterChiao 是誰", names) == ["hunterchiao"]  # 不分大小寫
    assert names_in("台北 天氣", names) == []
    assert names_in("A 牌 顯示卡", names) == []  # 一個字的名字不算，免得誤擋


def test_voice_reply_rules_are_short():
    p = prompts.reply_system("Yuki", [], [], spoken=True, web=True)
    assert "30 個字以內" in p and "查完只講結論" in p


# ---------- A3 修正：切句合併、ASR 名字、TTS 順序 ----------
import asyncio  # noqa: E402
from pathlib import Path  # noqa: E402

from yuki.voice.names import NicknameCache, nicknames_from_profile  # noqa: E402
from yuki.voice.session import VoiceSession  # noqa: E402

PROFILE = """# 小傑
## 基本資料
- 綽號：阿杰仔、傑哥（群組都這樣叫）
## 群組梗與綽號
以下是玩笑，不是事實：
- 被叫「小凱哥」，也有人叫他『明仔』。
- 常說「這一整句話是一個很長很長的幹話不是綽號」
## 說話風格
- 口頭禪「不要」（不該算進來）
"""


def make_session(tmp_path, *, members=("阿明",), reply=None, profiles=None, test_guild=False):
    guild_members = [SimpleNamespace(id=1, bot=False, display_name="阿明", name="aming", global_name=None),
                     SimpleNamespace(id=2, bot=False, display_name="小傑", name="jie", global_name=None),
                     SimpleNamespace(id=3, bot=False, display_name="小美", name="mei", global_name=None),
                     SimpleNamespace(id=4, bot=False, display_name="阿凱", name="kai", global_name=None)]
    in_channel = [m for m in guild_members if m.display_name in members]
    guild = SimpleNamespace(id=100, get_member=lambda i: next((m for m in guild_members if m.id == i), None))
    channel = SimpleNamespace(id=7, name="vc", guild=guild, members=in_channel)
    vc = SimpleNamespace(channel=channel, is_connected=lambda: True, is_playing=lambda: False)
    history = defaultdict(lambda: deque(maxlen=20))
    files = profiles or ProfileFiles(tmp_path)
    calls = []

    async def default_reply(gid, key, author, text, **kw):
        calls.append(text)
        await asyncio.sleep(0.3)
        return "好的"

    brain = SimpleNamespace(history=history, profiles=files, reply=reply or default_reply,
                            observe=lambda key, line: history[key].append(line))
    bot = SimpleNamespace(
        settings=SimpleNamespace(voice_listen_users=["1"], bot_name="Yuki", voice_asr_hotwords=["LoL"],
                                 voice_log_channel="", voice_attention_seconds=60, voice_alone_leave_seconds=9),
        brain=brain, user=SimpleNamespace(id=999), _wake=WAKE, _replying=0, _last_activity=0,
        voice=SimpleNamespace(describe=lambda g, n: ""), owner_name=lambda g: "",
        archive_guild=lambda g: "main" if test_guild else str(g.id),
        members_of=lambda g: [Person(str(m.id), m.display_name) for m in guild_members] + [Person("1", "aming")])
    session = VoiceSession(bot, vc, SimpleNamespace())
    session.calls = calls
    session.guild_members = guild_members
    return session


def item(session, uid, text):
    m = next(x for x in session.guild_members if x.id == uid)
    return (SimpleNamespace(ended=0.0), m, text, 0.0, 10.0)


def quiet(s):
    """不連 Discord、不合成：只留回覆流程。"""
    s.bot.voice_tts = SimpleNamespace(synth=lambda t: asyncio.sleep(0.05, b"x"))
    s._play = lambda mp3: asyncio.sleep(0.05)
    s._post_log = lambda text: asyncio.sleep(0)


def test_second_sentence_merges_before_reply_starts_playing(tmp_path):
    async def run():
        s = make_session(tmp_path)
        quiet(s)
        worker = asyncio.create_task(s._work())
        s._enqueue(item(s, 1, "先幫我查今天的"))
        await asyncio.sleep(0.1)  # 第一句的回覆還在想
        assert s.calls == ["先幫我查今天的"]
        s._enqueue(item(s, 1, "臺積電一塊"))
        await asyncio.sleep(0.8)
        worker.cancel()
        assert s.calls == ["先幫我查今天的", "先幫我查今天的 臺積電一塊"]
        lines = [l.text for l in s.bot.brain.history[s.channel_key]]
        assert lines == ["先幫我查今天的 臺積電一塊", "好的"]  # 對話紀錄裡只有接起來的那一句
        assert s.bot._replying == 0

    asyncio.run(run())


def test_cancelled_reply_really_stops_before_the_new_one_starts(tmp_path):
    """被取代的回覆要真的結束（連線斷掉、排隊位置放掉），新的回覆才開始，不會兩個同時跑。"""
    async def run():
        running, peak, cancelled = [0], [0], []

        async def reply(gid, key, author, text, **kw):
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            try:
                await asyncio.sleep(0.3)
                return "好"
            except asyncio.CancelledError:
                cancelled.append(text)
                raise
            finally:
                running[0] -= 1

        s = make_session(tmp_path, reply=reply)
        quiet(s)
        worker = asyncio.create_task(s._work())
        s._enqueue(item(s, 1, "第一句"))
        await asyncio.sleep(0.1)
        s._enqueue(item(s, 1, "第二句"))
        await asyncio.sleep(0.7)
        worker.cancel()
        assert cancelled == ["第一句"] and peak[0] == 1

    asyncio.run(run())


def test_merge_while_tts_is_synthesizing(tmp_path):
    """回覆已經想好、還在合成第一段（沒開始念）：也取消，重新回。"""
    async def run():
        s = make_session(tmp_path)
        quiet(s)
        s.bot.voice_tts = SimpleNamespace(synth=lambda t: asyncio.sleep(0.4, b"x"))
        played = []
        s._play = lambda mp3: (played.append(1), asyncio.sleep(0))[1]
        worker = asyncio.create_task(s._work())
        s._enqueue(item(s, 1, "第一句"))
        await asyncio.sleep(0.5)  # LLM 好了（0.3），TTS 還在做
        s._enqueue(item(s, 1, "第二句"))
        await asyncio.sleep(1.5)
        worker.cancel()
        assert s.calls == ["第一句", "第一句 第二句"] and len(played) == 1  # 只念了合併後的那一次

    asyncio.run(run())


def test_no_merge_once_playing_or_other_speaker(tmp_path):
    async def run():
        s = make_session(tmp_path)
        quiet(s)
        started = asyncio.Event()

        async def play(mp3):
            started.set()
            await asyncio.sleep(0.3)

        s._play = play
        worker = asyncio.create_task(s._work())
        s._enqueue(item(s, 1, "第一句"))
        await asyncio.wait_for(started.wait(), 2)  # 已經開始念
        s._enqueue(item(s, 1, "第二句"))
        assert s.queue.qsize() == 1 and s.queue._queue[0][2] == "第二句"  # 不合併，排著等
        worker.cancel()

        s2 = make_session(tmp_path)  # 還在想，但是換人講：不合併
        quiet(s2)
        w2 = asyncio.create_task(s2._work())
        s2._enqueue(item(s2, 1, "第一句"))
        await asyncio.sleep(0.1)
        s2._enqueue(item(s2, 2, "別人的話"))
        assert s2.queue._queue[0][2] == "別人的話"
        w2.cancel()

    asyncio.run(run())


def test_asr_names_include_outside_members_and_nicknames(tmp_path):
    files = ProfileFiles(tmp_path)
    files.write("100", "2", PROFILE)
    s = make_session(tmp_path, members=("阿明",), profiles=files)
    names = s.asr_names()
    assert names[0] == "阿明" and names[1] == "Yuki"  # 頻道裡的人排最前面
    assert {"小傑", "小美", "阿凱", "小凱哥", "明仔", "阿杰仔", "傑哥"} <= set(names)  # 頻道外成員 + 綽號
    assert "不要" not in names and not any("幹話" in n for n in names)
    ctx = asr_context(names, ["LoL"])
    assert ctx.index("阿明") < ctx.index("小傑") < ctx.index("小凱哥") and "LoL" in ctx


def test_asr_names_use_fallback_guild_profiles_and_never_raise(tmp_path):
    files = ProfileFiles(tmp_path)
    files.write("main", "2", PROFILE)
    s = make_session(tmp_path, profiles=files, test_guild=True)
    assert "小凱哥" in s.asr_names()
    s.bot.members_of = lambda g: 1 / 0  # 壞掉也不能報錯
    assert s.asr_names()[0] == "阿明"


def test_asr_context_is_capped():
    names = [f"名字{i:03d}" for i in range(300)]
    ctx = asr_context(names, ["LoL"])
    assert "名字000" in ctx and "名字299" not in ctx and "LoL" in ctx
    assert len(ctx) < 520


def test_nickname_parsing_rules():
    assert nicknames_from_profile(PROFILE) == ["阿杰仔", "傑哥", "小凱哥", "明仔"]
    assert nicknames_from_profile("") == [] and nicknames_from_profile("亂七八糟\n## 綽號\n沒有引號") == []


def test_nickname_cache_reloads_on_change_or_expiry(tmp_path):
    import os
    files = ProfileFiles(tmp_path)
    files.write("g", "2", "## 群組梗與綽號\n- 「小凱哥」\n")
    now = [0.0]
    cache = NicknameCache(files, ttl=600, clock=lambda: now[0])
    assert cache.get("g") == ["小凱哥"]
    reads = []
    orig = Path.read_text
    Path.read_text = lambda self, *a, **k: (reads.append(1), orig(self, *a, **k))[1]
    try:
        assert cache.get("g") == ["小凱哥"] and reads == []  # 快取：不讀檔
        now[0] = 601
        cache.get("g")
        assert len(reads) == 1  # 超過 10 分鐘：重讀
        p = files.path("g", "2")
        files.write("g", "2", "## 群組梗與綽號\n- 「新綽號」\n")
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 5))
        assert cache.get("g") == ["新綽號"]  # 檔案改了：馬上重讀
    finally:
        Path.read_text = orig
    assert cache.get("沒有這個伺服器") == []


def test_tts_first_segment_is_short_and_sent_alone_then_in_order(tmp_path):
    assert sentences("我覺得啊，其實還蠻有趣的啦。") == ["我覺得啊，", "其實還蠻有趣的啦。"]
    assert sentences("你說的那個東西我覺得，其實還蠻有趣的啦。") == ["你說的那個東西我覺得，其實還蠻有趣的啦。"]  # 前 8 字內切不到：照原本
    assert sentences("真的假的啊你在開玩笑吧，沒有逗號可切的長句子。")[0].startswith("真的假的啊")
    assert len(sentences("好啊，那就這樣決定吧。我們明天見。")[0]) <= 8

    async def run():
        s = make_session(tmp_path)
        events = []

        async def synth(text):
            events.append(("start", text))
            await asyncio.sleep(0.05)
            events.append(("done", text))
            return text.encode()

        async def play(mp3):
            events.append(("play", mp3.decode()))
            await asyncio.sleep(0.01)

        s.bot.voice_tts = SimpleNamespace(synth=synth)
        s._play = play
        text = "我覺得這樣還不錯啦。你自己決定就好！明天再說吧，好不好呢。"
        await s.speak(text)
        parts = sentences(text)[:3]
        assert len(parts) == 3
        # 第一段單獨先送、做完才送第二段；第二段做完才送第三段；播放順序不變
        assert events.index(("done", parts[0])) < events.index(("start", parts[1]))
        assert events.index(("done", parts[1])) < events.index(("start", parts[2]))
        assert [e[1] for e in events if e[0] == "play"] == parts

    asyncio.run(run())


def test_names_in_short_english_name_needs_whole_word():
    from yuki.brain import names_in
    names = ["al", "Ab", "阿明", "Bob2號"]
    assert names_in("Alcaraz wins final", names) == []          # 英文短名不擋含它字母的長字
    assert names_in("tab key shortcut", names) == []
    assert names_in("al 是誰", names) == ["al"]                 # 獨立出現、後接中文
    assert names_in("查一下AL的事", names) == ["al"]            # 前後接中文
    assert names_in("who is Al?", names) == ["al"]              # 前後標點／空白、不分大小寫
    assert names_in("阿明的學校", names) == ["阿明"]             # 中文照舊子字串
    assert names_in("bob2號的事", names) == ["Bob2號"]          # 中英夾雜照子字串（寧可多擋）
    assert names_in("xbob2號", names) == ["Bob2號"]


def test_name_prefix_strips_reply_format_and_plain_brackets():
    from yuki.brain import _NAME_PREFIX
    cases = {
        "[小傑] 好喔": "好喔",
        "[小傑 回覆 阿明「這是一段很長很長很長的被引用內容，超過三十二個字也要能被去掉，包括 [括號] 在內」] 好喔": "好喔",
        "[小傑 回覆你「…」]\n接話": "接話",
        "阿明 你那句我看到了": "阿明 你那句我看到了",   # 自然稱呼不動
        "好喔 [備註] 內容": "好喔 [備註] 內容",
    }
    for raw, want in cases.items():
        assert _NAME_PREFIX.sub("", raw) == want, raw
