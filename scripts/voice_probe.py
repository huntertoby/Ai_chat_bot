"""語音前置驗證（開發規格 6.0）：進語音頻道錄音、每人存一個 wav，再用 Edge TTS 念一句。

python -m scripts.voice_probe <伺服器ID> [秒數]
會進到「有人在的那個語音頻道」（沒人就進第一個）。
"""
from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import time
import wave
from collections import defaultdict
from pathlib import Path

import discord
from discord.ext import voice_recv

from yuki.config import ROOT, load_settings
from yuki.voice.dave import install as install_dave, patch_reader

OUT = ROOT / "data" / "voice_probe"


class CollectSink(voice_recv.AudioSink):
    def __init__(self):
        super().__init__()
        self.pcm: dict[str, bytearray] = defaultdict(bytearray)
        self.packets: dict[str, int] = defaultdict(int)
        self.start: dict[str, float] = {}

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data):
        # 沒講話時 Discord 不送封包：用到達時間補回停頓，不然句子會黏在一起
        name = user.display_name if user else f"ssrc-{data.packet.ssrc}"
        now = time.monotonic()
        if name not in self.start:
            self.start[name] = now
        expected = int((now - self.start[name]) * 48000) * 4
        gap = expected - len(self.pcm[name]) - len(data.pcm)
        if gap > 48000 * 4 // 5:  # 落後超過 0.2 秒才補，避免網路抖動造成雜音
            self.pcm[name] += bytes(gap - gap % 4)
        self.pcm[name] += data.pcm
        self.packets[name] += 1

    def cleanup(self):
        pass


def rms(pcm: bytes) -> float:
    import array
    a = array.array("h", pcm[: len(pcm) // 2 * 2])
    return (sum(x * x for x in a) / len(a)) ** 0.5 if a else 0.0


async def main(guild_id: int, seconds: int) -> None:
    s = load_settings()
    install_dave()
    intents = discord.Intents.default()
    intents.members = True
    client = discord.Client(intents=intents)
    done = asyncio.Event()

    @client.event
    async def on_ready():
        try:
            g = client.get_guild(guild_id)
            chans = [c for c in g.voice_channels if any(not m.bot for m in c.members)] or g.voice_channels
            ch = chans[0]
            print(f"位元率 {ch.bitrate // 1000} kbps", flush=True)
            print(f"進入 #{ch.name}，裡面有：{[m.display_name for m in ch.members]}", flush=True)
            vc: voice_recv.VoiceRecvClient = await ch.connect(cls=voice_recv.VoiceRecvClient)
            st = vc._connection
            print(f"DAVE 版本：{st.dave_protocol_version}，session 就緒：{bool(st.dave_session and st.dave_session.ready)}", flush=True)

            # 1. 播放測試（失敗不影響收音測試）
            try:
                import edge_tts
                mp3 = Path(tempfile.gettempdir()) / "yuki_probe.mp3"
                await edge_tts.Communicate("測試測試，聽得到我嗎？", "zh-TW-HsiaoChenNeural").save(str(mp3))
                t = time.monotonic()
                vc.play(discord.FFmpegPCMAudio(str(mp3)))
                while vc.is_playing():
                    await asyncio.sleep(0.1)
                print(f"播放完成（{time.monotonic() - t:.1f} 秒）", flush=True)
            except Exception:
                logging.exception("播放失敗，繼續做收音測試")

            # 2. 收音測試
            sink = CollectSink()
            vc.listen(sink)
            patch_reader(vc)
            print(f"開始錄音 {seconds} 秒，請講話…", flush=True)
            await asyncio.sleep(seconds)
            vc.stop_listening()
            await vc.disconnect()

            OUT.mkdir(parents=True, exist_ok=True)
            print(f"DAVE 解密：{vc._dave_stats}", flush=True)
            if not sink.pcm:
                print("沒收到任何聲音", flush=True)
            for name, pcm in sink.pcm.items():
                path = OUT / f"{''.join(c if c.isalnum() else '_' for c in name)}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(2)
                    w.setsampwidth(2)
                    w.setframerate(48000)
                    w.writeframes(pcm)
                secs = len(pcm) / 48000 / 4
                print(f"{name}：{sink.packets[name]} 個封包、{secs:.1f} 秒、音量 RMS {rms(pcm):.0f} → {path}", flush=True)
        except Exception:
            logging.exception("probe 失敗")
        finally:
            done.set()
            await client.close()

    await client.start(s.discord_token)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        raise SystemExit("用法：python -m scripts.voice_probe <伺服器ID> [秒數]")
    gid = int(sys.argv[1])
    asyncio.run(main(gid, int(sys.argv[2]) if len(sys.argv) > 2 else 10))
