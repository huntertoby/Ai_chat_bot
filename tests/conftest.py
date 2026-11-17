import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuki.embedding import normalize  # noqa: E402
from yuki.memory.store import MemoryStore  # noqa: E402


class FakeEmbedder:
    """字元 bigram 雜湊成向量：不用載模型，但相似文字仍有相近的向量。"""
    model_name = "fake"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, dtype=np.float32)
        text = text.lower()
        for a, b in zip(text, text[1:] + " "):
            v[int(hashlib.md5((a + b).encode()).hexdigest(), 16) % 256] += 1
        return v

    async def embed_queries(self, texts):
        return normalize(np.stack([self._vec(t) for t in texts]))

    async def embed_docs(self, texts):
        return normalize(np.stack([self._vec(t) for t in texts]))


class FakeLLM:
    def __init__(self, json_reply=None, text_reply="好喔"):
        self.json_reply = json_reply
        self.text_reply = text_reply
        self.calls = []

    async def chat_json(self, messages, schema, **kw):
        self.calls.append(messages)
        return self.json_reply

    async def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.text_reply


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store():
    return MemoryStore(":memory:", "fake")
