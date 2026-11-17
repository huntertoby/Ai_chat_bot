"""文字轉語音（開發規格 6.0）。介面固定：之後換本機模型只改這裡。"""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_PAREN = re.compile(r"[（(][^（）()]{0,40}[）)]")
_SENT = re.compile(r"(?<=[。！？!?…~～\n])")


def speakable(text: str) -> str:
    """念不出來或念出來很怪的東西拿掉：網址、表情符號、XD、括號補充、markdown 符號。"""
    text = _URL.sub("", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"(?i)(?<![a-z])x+d+(?![a-z])", "", text)  # 中文旁邊的 XD 也要拿掉（\b 不行）
    text = _PAREN.sub("", text)
    text = re.sub(r"[*_`#>|]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def sentences(text: str, min_chars: int = 6, first_max: int = 12) -> list[str]:
    """依句號、問號、驚嘆號切句；太短的併到下一句，免得一個字一個字念。"""
    out: list[str] = []
    buf = ""
    for part in _SENT.split(speakable(text)):
        buf += part.strip()
        if len(buf) >= min_chars:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf) < min_chars:
            out[-1] += buf
        else:
            out.append(buf)
    out = [s for s in out if re.search(r"\w", s)]
    # 本機 TTS 生成時間約等於句子長度：第一句太長就在第一個逗號切開，先念短的那段，早點開口
    if out and len(out[0]) > first_max:
        m = re.search(r"[，,、]", out[0][4:])
        if m and len(out[0]) - (m.end() + 4) >= 4:
            cut = m.end() + 4
            out[:1] = [out[0][:cut], out[0][cut:]]
    return out


class EdgeTTS:
    """微軟 Edge 的線上 TTS：0 顯存，但 Yuki 要講的話會送到微軟。edge-tts 要 7.3.1 以上（舊版 403）。"""

    def __init__(self, voice: str = "zh-TW-HsiaoChenNeural"):
        self.voice = voice

    async def synth(self, text: str) -> bytes:
        """回傳 mp3。"""
        import edge_tts
        out = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                out += chunk["data"]
        return bytes(out)


class BreezyTTS:
    """BreezyVoice（台灣腔，本機 GPU）：跑在 WSL 的 scripts/breezy_server.py。
    服務沒開或出錯時改用 fallback（Edge），不會讓 Yuki 啞掉。"""

    def __init__(self, url: str = "http://127.0.0.1:8890", fallback=None, timeout: float = 30):
        import httpx
        self.url = url.rstrip("/")
        self.fallback = fallback
        self._http = httpx.AsyncClient(timeout=timeout)

    async def synth(self, text: str) -> bytes:
        """回傳 wav（失敗時 fallback 的 mp3）；FFmpeg 兩種都吃。"""
        import logging
        try:
            r = await self._http.post(self.url + "/tts", json={"text": text})
            r.raise_for_status()
            return r.content
        except Exception as e:
            if self.fallback is None:
                raise
            logging.getLogger(__name__).warning("BreezyVoice 失敗（%s），改用 Edge", e)
            return await self.fallback.synth(text)
