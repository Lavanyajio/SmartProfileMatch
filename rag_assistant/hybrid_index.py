"""Hybrid index: dense vectors (FAISS, cosine) + sparse keywords (BM25), fused with Reciprocal Rank Fusion."""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Plus

from .documents import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    dense_rank: Optional[int]
    sparse_rank: Optional[int]


class HybridIndex:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.file_hashes: Dict[str, str] = {}  # hash -> file name
        self._faiss = None
        self._bm25 = None
        self._token_sets: List[set] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> List[str]:
        return sorted(set(self.file_hashes.values()))

    def has_file(self, digest: str) -> bool:
        return digest in self.file_hashes

    def add(self, chunks: List[Chunk], vectors: np.ndarray, digest: str, file_name: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)
        self.file_hashes[digest] = file_name
        self._rebuild()

    def clear(self) -> None:
        self.__init__()

    def _rebuild(self) -> None:
        self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self._faiss.add(self.vectors)
        tokens = [tokenize(c.text) or ["_"] for c in self.chunks]
        # BM25Plus keeps IDF positive, so rare terms still score on small corpora
        self._bm25 = BM25Plus(tokens)
        self._token_sets = [set(t) for t in tokens]

    def dense_ranking(self, query_vec: np.ndarray, pool: int) -> List[int]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        _, ids = self._faiss.search(q, min(pool, len(self.chunks)))
        return [int(i) for i in ids[0] if i >= 0]

    def sparse_ranking(self, query: str, pool: int) -> List[int]:
        terms = tokenize(query)
        scores = self._bm25.get_scores(terms)
        query_set = set(terms)
        order = np.argsort(scores)[::-1]
        # only documents sharing at least one keyword count as keyword matches
        return [int(i) for i in order if query_set & self._token_sets[i]][:pool]

    def search(self, queries: List[str], query_vecs: np.ndarray, top_k: int,
               pool: int = 20, rrf_k: int = 60) -> List[SearchHit]:
        """Fuse the dense and sparse rankings of every query variant with RRF."""
        if not self.chunks:
            return []
        fused: Dict[int, float] = {}
        best_dense: Dict[int, int] = {}
        best_sparse: Dict[int, int] = {}
        for query, vec in zip(queries, query_vecs):
            for rank, idx in enumerate(self.dense_ranking(vec, pool), start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
                best_dense[idx] = min(rank, best_dense.get(idx, rank))
            for rank, idx in enumerate(self.sparse_ranking(query, pool), start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
                best_sparse[idx] = min(rank, best_sparse.get(idx, rank))
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [SearchHit(self.chunks[i], s, best_dense.get(i), best_sparse.get(i)) for i, s in ordered]

    # ---- persistence -------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.vectors is None:
            for name in ("vectors.npy", "meta.json"):
                (directory / name).unlink(missing_ok=True)
            return
        np.save(directory / "vectors.npy", self.vectors)
        meta = {"chunks": [c.to_dict() for c in self.chunks], "file_hashes": self.file_hashes}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex":
        index = cls()
        directory = Path(directory)
        if not (directory / "meta.json").exists() or not (directory / "vectors.npy").exists():
            return index
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        index.chunks = [Chunk(**c) for c in meta["chunks"]]
        index.file_hashes = meta.get("file_hashes", {})
        index.vectors = np.load(directory / "vectors.npy")
        if index.chunks:
            index._rebuild()
        return index
