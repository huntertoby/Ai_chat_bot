"""出錯私訊（yuki/alert.py）：不需要網路、模型、Discord。"""
import asyncio
import logging
import threading

from yuki import alert
from yuki.alert import Alerter, AlertLogHandler, redact


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


def make(clock=None, **kw):
    kw.setdefault("secrets", [])
    return Alerter(["<你的使用者ID>"], clock=clock or Clock(), **kw)


async def connected(al):
    sent = []

    async def send(uid, text):
        sent.append((uid, text))

    al.connect(send)
    await asyncio.sleep(0.01)
    return sent


def test_same_fingerprint_once_per_window():
    clock = Clock()
    al = make(clock, dedupe_min=30)
    assert al.notify("lmstudio", "a.py:1", "壞了")
    clock.t += 29 * 60
    assert not al.notify("lmstudio", "a.py:1", "又壞了")
    assert al.notify("lmstudio", "a.py:2", "別的位置")  # 位置不同＝不同指紋
    assert al.notify("searxng", "a.py:1", "別的類型")
    clock.t += 2 * 60  # 距離第一則 31 分鐘
    assert al.notify("lmstudio", "a.py:1", "過了 30 分鐘")


def test_hourly_cap_and_window_slides():
    clock = Clock()
    al = make(clock, max_per_hour=3, dedupe_min=1)
    assert [al.notify("k", f"w{i}", "x") for i in range(5)] == [True, True, True, False, False]
    clock.t += 3601
    assert al.notify("k", "w9", "x")


def test_no_owner_means_log_only():
    al = Alerter([], secrets=[])
    assert not al.notify("k", "w", "x")
    assert al._pending == []


def test_queue_before_connect_then_one_merged_message():
    async def go():
        al = make()
        al.notify("a", "1", "第一個")
        al.notify("b", "2", "第二個")
        assert len(al._pending) == 2
        sent = await connected(al)
        return al, sent

    al, sent = asyncio.run(go())
    assert len(sent) == 1 and sent[0][0] == "<你的使用者ID>"
    assert "第一個" in sent[0][1] and "第二個" in sent[0][1] and al._pending == []


def test_notify_after_connect_sends_directly():
    async def go():
        al = make()
        sent = await connected(al)
        al.notify("a", "1", "即時")
        await asyncio.sleep(0.01)
        return sent

    sent = asyncio.run(go())
    assert len(sent) == 1 and "即時" in sent[0][1]


def test_notify_from_other_thread():
    async def go():
        al = make()
        sent = await connected(al)
        t = threading.Thread(target=lambda: al.notify("a", "1", "執行緒"))
        t.start()
        t.join()
        await asyncio.sleep(0.05)
        return sent

    assert len(asyncio.run(go())) == 1


def test_message_format_has_time_trace_and_first_line_only():
    al = make()
    try:
        raise ValueError("第一行\n第二行 Traceback...")
    except ValueError as e:
        al.notify("k", "w", "SearXNG 連不上", exc=e, trace_id="abc12345")
    text = al._pending[0]
    assert "SearXNG 連不上" in text and "trace_id：abc12345" in text and "時間：" in text
    assert "ValueError: 第一行" in text and "第二行" not in text


def test_redact_secrets():
    token = ".".join(["MTA4NzY1NDMyMTA5ODc2NTQz", "GabcDE", "abcdefghijklmnopqrstuvwxyz1"])  # 假的，分開寫避免被 GitHub 當成真 token
    out = redact(f"token {token} 與 Bearer sk-abc.def_ghi 與 {'a1b2' * 12} 與 我的密碼是hunter2xyz", ["hunter2xyz"])
    assert token not in out and "sk-abc" not in out and "a1b2a1b2" not in out and "hunter2xyz" not in out
    assert out.count("[已遮蔽]") >= 4
    assert redact("LM Studio 連不上 port 1234") == "LM Studio 連不上 port 1234"  # 普通文字不動


def test_secrets_from_env_picks_key_like_names():
    env = {"DISCORD_TOKEN": "tok-123456", "OPENAI_API_KEY": "key-abcdef", "BOT_NAME": "Yuki", "X_SECRET": "ab"}
    assert sorted(alert.secrets_from_env(env)) == ["key-abcdef", "tok-123456"]


def test_notify_redacts_before_sending():
    al = Alerter(["1"], secrets=["hunter2xyz"], clock=Clock())
    al.notify("k", "w", "連線字串 hunter2xyz 失敗")
    assert "hunter2xyz" not in al._pending[0]


def test_send_failure_does_not_trigger_new_alert():
    """送私訊失敗（甚至失敗時別處記了 ERROR log）都不能再產生通知。"""
    al = make()
    handler = AlertLogHandler(al)
    root = logging.getLogger()
    root.addHandler(handler)
    attempts = []

    async def broken(uid, text):
        attempts.append(text)
        logging.getLogger("discord.http").error("Forbidden 50007 無法傳私訊")
        raise RuntimeError("Cannot send messages to this user")

    async def go():
        al.connect(broken)
        al.notify("a", "1", "第一個")
        await asyncio.sleep(0.05)

    try:
        asyncio.run(go())
    finally:
        root.removeHandler(handler)
    assert len(attempts) == 1  # 沒有重試、沒有迴圈
    assert al._pending == []


def test_log_handler_error_becomes_notify_and_warning_does_not():
    al = make()
    handler = AlertLogHandler(al)
    log = logging.getLogger("yuki.test_alert_sample")
    log.addHandler(handler)
    log.propagate = False
    try:
        log.warning("只是警告")
        assert al._pending == []
        try:
            raise KeyError("boom")
        except KeyError:
            log.exception("某處壞了")
    finally:
        log.removeHandler(handler)
    (text,) = al._pending
    assert "某處壞了" in text and "KeyError" in text and "yuki.test_alert_sample" in text


def test_log_handler_ignores_alert_modules_own_logs():
    al = make()
    handler = AlertLogHandler(al)
    rec = logging.LogRecord("yuki.alert", logging.ERROR, "alert.py", 1, "自己的錯誤", None, None)
    handler.emit(rec)
    assert al._pending == []


def test_health_loop_notifies_once_and_logs_recovery(caplog):
    al = make()
    results = [None, "ConnectError: refused", "ConnectError: refused", None]

    async def checker(url):
        return results.pop(0) if results else None

    async def go():
        task = asyncio.create_task(alert.health_loop("http://x/v1", 0.01, al, checker=checker))
        await asyncio.sleep(0.2)
        task.cancel()

    with caplog.at_level(logging.INFO, logger="yuki.alert"):
        asyncio.run(go())
    assert len(al._pending) == 1 and "LM Studio" in al._pending[0]  # 連兩次失敗，只一則
    assert "恢復連線" in caplog.text


def test_check_lmstudio_unreachable_returns_reason():
    reason = asyncio.run(alert.check_lmstudio("http://127.0.0.1:9/v1", timeout=1))
    assert reason  # 連不上就回原因字串


def test_module_notify_without_default_is_noop():
    alert.set_default(None)
    assert alert.notify("k", "w", "x") is False
