"""Offline test doubles so the suite runs in Jenkins without HF_TOKEN or network access."""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_assistant.config import settings  # noqa: E402
from rag_assistant.hybrid_index import tokenize  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (64 dims)."""

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


class FakeLLM:
    def __init__(self, answer="E07 means battery cell imbalance [1]."):
        self.answer = answer
        self.calls = []

    def chat(self, messages, max_tokens=None, temperature=None):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if "rewrite user questions" in prompt:
            return json.dumps({"standalone": "What does error code E07 mean?",
                               "variants": ["E07 battery cell imbalance fix"]})
        return self.answer


@pytest.fixture
def cfg(tmp_path):
    return replace(settings, index_dir=tmp_path / "index", chunk_size=400, chunk_overlap=50)


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
