"""把每個人的語音串流切成一句一句（開發規格 6.0 B-2）。

Discord 每 20ms 給一個 48kHz 雙聲道 16-bit 的 PCM 框；沒講話時通常不送封包，
所以「講完」要同時看 VAD（有收到靜音框）和「多久沒收到封包」（tick）。
這個模組不碰 Discord 和 GPU，方便測試。執行緒安全（收音在另一個執行緒）。
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SR_IN = 48000
SR_VAD = 16000
VAD_CHUNK = 512  # silero-vad 在 16kHz 一次要 512 個樣本（32ms）
CHUNK_MS = VAD_CHUNK * 1000 // SR_VAD


def to_mono48(pcm: bytes) -> np.ndarray:
    a = np.frombuffer(pcm[: len(pcm) // 4 * 4], dtype=np.int16).astype(np.float32) / 32768
    return a.reshape(-1, 2).mean(axis=1)


def to_16k(audio48: np.ndarray) -> np.ndarray:
    """整句重取樣（有濾波），給 ASR 用。"""
    from scipy.signal import resample_poly
    return resample_poly(audio48, 1, 3).astype(np.float32)


@dataclass
class Utterance:
    user_id: str
    audio48: np.ndarray  # 48kHz 單聲道 float32
    started: float
    ended: float

    @property
    def seconds(self) -> float:
        return len(self.audio48) / SR_IN


@dataclass
class _State:
    vad: Callable[[np.ndarray], float]
    pending16: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    pending48: list = field(default_factory=list)  # 還沒判斷 VAD 的 48k 片段
    preroll: deque = field(default_factory=lambda: deque(maxlen=8))  # 開口前約 250ms
    audio: list = field(default_factory=list)
    active: bool = False
    speech_ms: int = 0
    run_ms: int = 0  # 這次開口以來連續（容許短暫停頓）講了多久，給插話判斷
    started: float = 0.0
    last_speech: float = 0.0
    last_packet: float = 0.0


class Segmenter:
    """feed()：收到音訊（任何執行緒）；tick()：定時呼叫，找出講完的句子。"""

    def __init__(self, make_vad: Callable[[], Callable[[np.ndarray], float]], *, silence_ms: int = 700,
                 min_speech_ms: int = 400, max_ms: int = 15000, threshold: float = 0.5):
        self.make_vad = make_vad
        self.silence = silence_ms / 1000
        self.min_speech_ms = min_speech_ms
        self.max_s = max_ms / 1000
        self.threshold = threshold
        self._users: dict[str, _State] = {}
        self._done: list[Utterance] = []
        self._lock = threading.Lock()

    def feed(self, user_id: str, pcm: bytes, now: float) -> None:
        a48 = to_mono48(pcm)
        with self._lock:
            st = self._users.get(user_id)
            if st is None:
                st = self._users[user_id] = _State(self.make_vad())
            st.last_packet = now
            st.pending48.append(a48)
            st.pending16 = np.concatenate([st.pending16, a48[::3]])  # VAD 用，粗略降頻就夠
            while len(st.pending16) >= VAD_CHUNK:
                chunk16, st.pending16 = st.pending16[:VAD_CHUNK], st.pending16[VAD_CHUNK:]
                # 跟這 512 個樣本對應的 48k 音訊
                need, piece = VAD_CHUNK * 3, []
                while need > 0 and st.pending48:
                    head = st.pending48[0]
                    if len(head) <= need:
                        piece.append(st.pending48.pop(0))
                        need -= len(head)
                    else:
                        piece.append(head[:need])
                        st.pending48[0] = head[need:]
                        need = 0
                self._step(user_id, st, chunk16, np.concatenate(piece) if piece else np.zeros(0, np.float32), now)

    def _step(self, uid: str, st: _State, chunk16: np.ndarray, chunk48: np.ndarray, now: float) -> None:
        speech = self.vad_prob(st, chunk16) >= self.threshold
        if speech:
            if not st.active:
                st.active, st.started, st.speech_ms, st.run_ms = True, now, 0, 0
                st.audio = list(st.preroll)
            st.speech_ms += CHUNK_MS
            st.run_ms += CHUNK_MS
            st.last_speech = now
        elif st.active and now - st.last_speech > 0.25:
            st.run_ms = 0  # 停頓超過 0.25 秒，插話計時重來
        if st.active:
            st.audio.append(chunk48)
            if now - st.last_speech > self.silence or now - st.started > self.max_s:
                self._finish(uid, st, now)
        else:
            st.preroll.append(chunk48)

    @staticmethod
    def vad_prob(st: _State, chunk16: np.ndarray) -> float:
        return float(st.vad(chunk16))

    def _finish(self, uid: str, st: _State, now: float) -> None:
        if st.speech_ms >= self.min_speech_ms and st.audio:
            self._done.append(Utterance(uid, np.concatenate(st.audio), st.started, now))
        st.active, st.audio, st.speech_ms, st.run_ms = False, [], 0, 0
        st.preroll.clear()

    def tick(self, now: float) -> list[Utterance]:
        """回傳講完的句子。沒收到封包超過 silence 也算講完（Discord 靜音時不送封包）。"""
        with self._lock:
            for uid, st in self._users.items():
                if st.active and now - max(st.last_speech, st.last_packet) > self.silence:
                    self._finish(uid, st, now)
            done, self._done = self._done, []
            return done

    def talking_ms(self, user_id: str) -> int:
        """這個人現在連續講了多久（毫秒），給插話判斷用。"""
        with self._lock:
            st = self._users.get(user_id)
            return st.run_ms if st and st.active else 0

    def forget(self, user_id: str) -> None:
        with self._lock:
            self._users.pop(user_id, None)


def silero_factory() -> Callable[[], Callable[[np.ndarray], float]]:
    """每個人一個 silero-vad（模型有內部狀態，不能共用）。CPU 跑，一次約 0.1ms。"""
    import torch
    from silero_vad import load_silero_vad

    def make():
        model = load_silero_vad()

        def prob(chunk16: np.ndarray) -> float:
            with torch.no_grad():
                return model(torch.from_numpy(chunk16), SR_VAD).item()
        return prob
    return make
