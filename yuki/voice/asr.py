"""語音辨識（開發規格 6.0）。介面固定：之後換模型只改這裡。"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


def asr_context(names: list[str], hotwords: list[str]) -> str:
    """給 ASR 的提示：人名、常用詞。實測人名錯字率 15.9% → 6.7%。
    不放最近的對話：實測 Qwen3-ASR 會把提示裡的對話照抄進轉錄結果。"""
    parts = ["Discord 語音聊天，台灣口語。"]
    words = list(dict.fromkeys([n for n in names if n] + list(hotwords)))
    if words:
        parts.append("常出現的名字和詞：" + "、".join(words) + "。")
    return "".join(parts)


class QwenASR:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from qwen_asr import Qwen3ASRModel
            t = time.monotonic()
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name, dtype=torch.bfloat16, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                max_new_tokens=256)
            log.info("ASR %s 載入完成（%.1f 秒）", self.model_name, time.monotonic() - t)
        return self._model

    def _run(self, audio16: np.ndarray, context: str) -> str:
        with self._lock:
            model = self._load()
            # 一定要指定中文：沒指定時「喂喂喂」會被判成泰文
            return model.transcribe((audio16, 16000), language="Chinese", context=context)[0].text.strip()

    async def warmup(self) -> None:
        await asyncio.to_thread(self._run, np.zeros(16000, np.float32), "")

    async def transcribe(self, audio16: np.ndarray, context: str = "") -> str:
        return await asyncio.to_thread(self._run, audio16, context)
