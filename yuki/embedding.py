"""Embedding：本機 sentence-transformers 或 LM Studio。輸出一律是 L2 正規化的 float32 矩陣。"""
from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import httpx
import numpy as np

log = logging.getLogger(__name__)

# 某些模型的查詢端 / 文件端需要不同前綴，不加會明顯掉分
_PREFIXES = {
    "nomic": ("search_query: ", "search_document: "),
    "qwen3": ("Instruct: Given a chat question, retrieve memories about Discord group members\nQuery: ", ""),
    "e5": ("query: ", "passage: "),
}


def default_prefixes(model: str) -> tuple[str, str]:
    name = model.lower()
    for key, prefixes in _PREFIXES.items():
        if key in name:
            return prefixes
    return "", ""


def normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


class Embedder(Protocol):
    model_name: str

    async def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    async def embed_docs(self, texts: list[str]) -> np.ndarray: ...


class _PrefixedEmbedder:
    model_name: str

    def __init__(self, model_name: str, query_prefix: str | None, doc_prefix: str | None):
        self.model_name = model_name
        dq, dd = default_prefixes(model_name)
        self.query_prefix = dq if query_prefix is None else query_prefix
        self.doc_prefix = dd if doc_prefix is None else doc_prefix

    async def _embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    async def embed_queries(self, texts: list[str]) -> np.ndarray:
        return await self._embed([self.query_prefix + t for t in texts])

    async def embed_docs(self, texts: list[str]) -> np.ndarray:
        return await self._embed([self.doc_prefix + t for t in texts])


class LocalEmbedder(_PrefixedEmbedder):
    def __init__(self, model_name: str, device: str = "auto",
                 query_prefix: str | None = None, doc_prefix: str | None = None, *, half: bool = False):
        super().__init__(model_name, query_prefix, doc_prefix)
        self.device = None if device == "auto" else device
        self.half = half
        self._model = None
        self._lock = asyncio.Lock()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("載入 embedding 模型 %s ...", self.model_name)
            kwargs = {}
            if self.half and self.device != "cpu":
                import torch
                if torch.cuda.is_available():
                    kwargs["model_kwargs"] = {"torch_dtype": torch.float16}  # bge-m3：2.7GB → 1.4GB 顯存
            self._model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    async def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        async with self._lock:  # GPU 上同時跑多個 encode 沒有好處
            vecs = await asyncio.to_thread(self._encode, texts)
        return normalize(vecs)


class LMStudioEmbedder(_PrefixedEmbedder):
    def __init__(self, base_url: str, model_name: str, timeout: float = 60,
                 query_prefix: str | None = None, doc_prefix: str | None = None):
        super().__init__(model_name, query_prefix, doc_prefix)
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        resp = await self._http.post(f"{self.base_url}/embeddings",
                                     json={"model": self.model_name, "input": texts})
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return normalize(np.array([d["embedding"] for d in data]))


def build_embedder(settings) -> Embedder:
    if settings.embed_backend == "lmstudio":
        return LMStudioEmbedder(settings.lmstudio_base_url, settings.embed_model,
                                query_prefix=settings.embed_query_prefix,
                                doc_prefix=settings.embed_doc_prefix)
    return LocalEmbedder(settings.embed_model, settings.device,
                         query_prefix=settings.embed_query_prefix,
                         doc_prefix=settings.embed_doc_prefix, half=settings.model_half)
