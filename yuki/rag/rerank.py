"""Cross-encoder rerank：把（子查詢, 記憶）成對打分，比向量相似度準。

支援兩類模型，都用 sentence-transformers 的 CrossEncoder 載入：
- BAAI/bge-reranker-v2-m3：傳統 cross-encoder，輸出一個分數。
- Qwen/Qwen3-Reranker-*：用 LLM 回答「yes / no」的 logit 差當分數，可以加任務說明（instruction），
  可用 bitsandbytes 以 4-bit 載入省顯存（4B 從約 8 GB 降到約 2.5 GB）。
兩者都經過 sigmoid 變成 0~1，RERANK_MIN_SCORE 門檻對兩種模型意義相同（但分數分布不同，換模型後要重新評估）。
預設用 bge：在 1.8 萬筆模擬記憶的評估中，Qwen3-Reranker-4B（4-bit）Hit@6 80%，bge 82%，bge 也更快。
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = ("Given a question asked in a Discord group chat, judge whether the memory record "
                       "contains information that helps answer the question")


class LocalReranker:
    def __init__(self, model_name: str, device: str = "auto", *, quantize: str = "none",
                 instruction: str = DEFAULT_INSTRUCTION, max_length: int = 512, batch_size: int = 16,
                 half: bool = False):
        self.model_name = model_name
        self.half = half
        self.device = None if device == "auto" else device
        self.quantize = quantize.lower()
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._prompted = False  # 模型本身帶 prompts（Qwen3-Reranker）才送 instruction
        self._lock = asyncio.Lock()

    def _model_kwargs(self) -> dict:
        if self.quantize in ("4bit", "8bit"):
            try:
                import bitsandbytes  # noqa: F401
                import torch
                from transformers import BitsAndBytesConfig
            except ImportError:
                log.warning("沒有安裝 bitsandbytes，%s 改用原本精度載入（pip install bitsandbytes）", self.model_name)
            else:
                if self.quantize == "4bit":
                    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16)
                else:
                    q = BitsAndBytesConfig(load_in_8bit=True)
                return {"quantization_config": q}
        if self.half and self.device != "cpu":
            import torch
            if torch.cuda.is_available():
                return {"torch_dtype": torch.float16}  # bge-reranker：2.2GB → 1.1GB 顯存
        # 依模型 config 的精度載入：Qwen3 是 bf16（比 fp32 省一半），bge 維持原本的 fp32
        return {"torch_dtype": "auto"}

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info("載入 rerank 模型 %s（量化：%s）...", self.model_name, self.quantize)
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length,
                                       model_kwargs=self._model_kwargs())
            self._prompted = bool(getattr(self._model, "prompts", None))
        return self._model

    def _predict(self, query: str, texts: list[str]) -> list[float]:
        import torch
        model = self._load()
        kwargs = {"prompt": self.instruction} if self._prompted and self.instruction else {}
        # Qwen3-Reranker 輸出 yes/no 的 logit 差（沒有 activation）；統一套 sigmoid → P(yes)，
        # bge-reranker 預設本來就是 sigmoid，結果不變
        scores = model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                               show_progress_bar=False, activation_fn=torch.nn.Sigmoid(), **kwargs)
        return [float(s) for s in scores]

    async def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._predict, query, texts)


def build_reranker(settings) -> LocalReranker | None:
    if not settings.rerank_enabled:
        return None
    return LocalReranker(settings.rerank_model, settings.device, quantize=settings.rerank_quantize,
                         instruction=settings.rerank_instruction or DEFAULT_INSTRUCTION,
                         half=settings.model_half)
