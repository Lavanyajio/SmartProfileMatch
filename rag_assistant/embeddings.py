"""Embedding back-ends: Hugging Face Inference API (default) or local sentence-transformers."""
import logging
from typing import List, Optional

import numpy as np
# from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def _to_matrix(raw) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:  # token-level output -> mean pooling
        arr = arr.mean(axis=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


class HFApiEmbedder:
    """Calls the feature-extraction task on the Hugging Face Inference API."""

    # def __init__(self, model: str, token: str, client: Optional[object] = None, batch_size: int = 32):
    #     if client is None and not token:
    #         raise ValueError("HF_TOKEN is required for API embeddings.")
    #     self.model = model
    #     self.batch_size = batch_size
    #     self.client = client or InferenceClient(token=token, provider="hf-inference")

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        try:
            return _to_matrix(self.client.feature_extraction(batch, model=self.model))
        except Exception as exc:
            logger.info("Batch embedding failed (%s); falling back to one-by-one.", exc)
            rows = [_to_matrix(self.client.feature_extraction(t, model=self.model))[0] for t in batch]
            return np.vstack(rows)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        parts = [self._embed_batch(texts[i:i + self.batch_size])
                 for i in range(0, len(texts), self.batch_size)]
        return _normalize(np.vstack(parts))


class LocalEmbedder:
    """Runs a sentence-transformers model on this machine (pip install sentence-transformers)."""

    # def __init__(self, model: str):
    #     from sentence_transformers import SentenceTransformer  # optional dependency
    #     self.model = SentenceTransformer(model)
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
    def embed(self, texts: List[str]) -> np.ndarray:
        return _normalize(np.asarray(self.model.encode(texts), dtype=np.float32))


def build_embedder(backend: str, model: str, token: str):
    if backend == "local":
        return LocalEmbedder(model)
    return HFApiEmbedder(model, token)
