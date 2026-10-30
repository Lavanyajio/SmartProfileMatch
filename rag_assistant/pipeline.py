"""The RAG pipeline: ingest -> rewrite query -> hybrid retrieve -> grounded answer with citations."""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import Settings, settings as default_settings
from .documents import chunk_pages, file_hash, load_file
from .hybrid_index import HybridIndex, SearchHit
from .llm import HFChatModel

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You rewrite user questions for a document search engine.
Given the chat history and the latest question:
1. Write ONE standalone question that makes sense without the history.
2. Write up to 2 alternative phrasings that use different keywords or synonyms.
Reply with JSON only, like: {{"standalone": "...", "variants": ["...", "..."]}}

Chat history:
{history}

Latest question: {question}"""

ANSWER_SYSTEM = """You are a careful assistant that answers ONLY from the numbered context passages.
Rules:
- Cite every fact with the passage number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, say "I could not find that in the uploaded documents." and stop.
- Never invent facts, numbers or file names. Be concise and use bullet points for lists."""


@dataclass
class Answer:
    text: str
    hits: List[SearchHit] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _history_text(history: List[dict], max_turns: int = 6) -> str:
    lines = []
    for msg in history[-max_turns:]:
        content = msg.get("content")
        if isinstance(content, str) and not (msg.get("metadata") or {}).get("title"):
            lines.append(f"{msg.get('role', 'user')}: {content[:500]}")
    return "\n".join(lines) or "(none)"


class RAGPipeline:
    def __init__(self, llm=None, embedder=None, cfg: Settings = default_settings,
                 index: Optional[HybridIndex] = None):
        self.cfg = cfg
        self.llm = llm or HFChatModel(cfg.llm_model, cfg.hf_token, cfg.llm_provider,
                                      cfg.max_new_tokens, cfg.temperature)
        self._embedder = embedder
        self.index = index if index is not None else HybridIndex.load(cfg.index_dir)

    @property
    def embedder(self):
        if self._embedder is None:
            from .embeddings import build_embedder
            self._embedder = build_embedder(self.cfg.embedding_backend, self.cfg.embedding_model,
                                            self.cfg.hf_token)
        return self._embedder

    # ---- ingestion ---------------------------------------------------
    def ingest(self, paths: List[Path]) -> str:
        report = []
        for path in paths:
            path = Path(path)
            try:
                digest = file_hash(path)
                if self.index.has_file(digest):
                    report.append(f"⏭️ {path.name}: already indexed")
                    continue
                chunks = chunk_pages(load_file(path), self.cfg.chunk_size, self.cfg.chunk_overlap)
                if not chunks:
                    report.append(f"⚠️ {path.name}: no readable text (scanned PDF?)")
                    continue
                vectors = self.embedder.embed([c.text for c in chunks])
                self.index.add(chunks, vectors, digest, path.name)
                report.append(f"✅ {path.name}: {len(chunks)} chunks")
            except Exception as exc:
                logger.exception("Ingestion failed for %s", path)
                report.append(f"❌ {path.name}: {exc}")
        self.index.save(self.cfg.index_dir)
        return "\n".join(report)

    def reset(self) -> None:
        self.index.clear()
        self.index.save(self.cfg.index_dir)

    # ---- querying ----------------------------------------------------
    def build_queries(self, question: str, history: List[dict]) -> List[str]:
        raw = self.llm.chat(
            [{"role": "user",
              "content": REWRITE_PROMPT.format(history=_history_text(history), question=question)}],
            max_tokens=200, temperature=0.0)
        data = _extract_json(raw) or {}
        queries = [str(data.get("standalone") or question).strip()]
        for variant in data.get("variants") or []:
            variant = str(variant).strip()
            if variant and variant.lower() not in {q.lower() for q in queries}:
                queries.append(variant)
        return queries[:3]

    def retrieve(self, queries: List[str], top_k: int) -> List[SearchHit]:
        vectors = self.embedder.embed(queries)
        return self.index.search(queries, vectors, top_k, self.cfg.candidate_pool, self.cfg.rrf_k)

    def answer(self, question: str, history: Optional[List[dict]] = None,
               top_k: Optional[int] = None) -> Answer:
        history = history or []
        if len(self.index) == 0:
            return Answer("📂 No documents are indexed yet. Upload files or click **Load sample docs** first.")
        queries = self.build_queries(question, history)
        hits = self.retrieve(queries, top_k or self.cfg.top_k)
        if not hits:
            return Answer("I could not find that in the uploaded documents.", [], queries)

        context = "\n\n".join(f"[{i}] (source: {h.chunk.source}, page {h.chunk.page})\n{h.chunk.text}"
                              for i, h in enumerate(hits, start=1))
        messages = [{"role": "system", "content": ANSWER_SYSTEM}]
        messages += [m for m in history[-4:]
                     if isinstance(m.get("content"), str) and not (m.get("metadata") or {}).get("title")]
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {queries[0]}"})
        return Answer(self.llm.chat(messages), hits, queries)


def format_sources(answer: Answer) -> str:
    if not answer.hits:
        return ""
    lines = [f"**Search queries:** {' · '.join(answer.search_queries)}", ""]
    for i, hit in enumerate(answer.hits, start=1):
        dense = f"#{hit.dense_rank}" if hit.dense_rank else "–"
        sparse = f"#{hit.sparse_rank}" if hit.sparse_rank else "–"
        snippet = hit.chunk.text[:220].replace("\n", " ")
        lines.append(f"**[{i}] {hit.chunk.source} · page {hit.chunk.page}** "
                     f"(semantic {dense}, keyword {sparse}, fused {hit.score:.4f})\n> {snippet}…\n")
    return "\n".join(lines)
