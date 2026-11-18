"""語音辨識（開發規格 6.0）。介面固定：之後換模型只改這裡。"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


def asr_context(names: list[str], hotwords: list[str], max_chars: int = 400) -> str:
    """給 ASR 的提示：人名、常用詞。實測人名錯字率 15.9% → 6.7%。
    不放最近的對話：實測 Qwen3-ASR 會把提示裡的對話照抄進轉錄結果。
    names 越前面越重要；名字加起來超過 max_chars 字就丟掉後面的（提示太長會拖慢 ASR）。"""
    parts = ["Discord 語音聊天，台灣口語。"]
    kept, used = [], 0
    for n in dict.fromkeys(n for n in names if n):
        if used + len(n) + 1 > max_chars:
            break
        kept.append(n)
        used += len(n) + 1
    words = list(dict.fromkeys(kept + list(hotwords)))
    if words:
        parts.append("常出現的名字和詞：" + "、".join(words) + "。")
    return "".join(parts)


def local_model_path(name: str) -> str:
    """repo id → 本機快取的 snapshot 資料夾。傳 repo id 的話 transformers 載 tokenizer 時還會上網查模型資訊，
    離線模式下直接報錯；傳本機路徑就不會。缺檔時 snapshot_download 會丟 LocalEntryNotFoundError。"""
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download
    return snapshot_download(name, local_files_only=True)


class QwenASR:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            # 只用本機快取：缺檔時馬上報錯，不要在 /來語音 時偷偷上網下載好幾 GB（會卡住、使用者只看到一直思考）
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import torch
            from qwen_asr import Qwen3ASRModel
            t = time.monotonic()
            self._model = Qwen3ASRModel.from_pretrained(
                local_model_path(self.model_name), dtype=torch.bfloat16, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
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
