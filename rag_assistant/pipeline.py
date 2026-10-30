"""
Smart Profile Matcher Pipeline

Flow:
Ingest → Query Expansion → Hybrid Retrieval →
Candidate Analysis → Ranked Response
"""

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

# -----------------------------------------------------------
# QUERY REWRITER
# -----------------------------------------------------------

REWRITE_PROMPT = """
You are an AI recruiter and talent intelligence assistant.

Given a recruiter's search request:

1. Rewrite into one standalone query.
2. Generate up to five alternative searches using:
   - role synonyms
   - skill synonyms
   - related technologies
   - industry terminology

Examples:

Question:
How many System Admin profiles exist?

Variants:
- System Administrator
- IT Administrator
- Windows Administrator
- Infrastructure Administrator
- Server Administrator

Question:
Find Data Engineers

Variants:
- ETL Developer
- Analytics Engineer
- Big Data Engineer
- Data Platform Engineer
- Pipeline Developer

Return JSON only:

{{
  "standalone": "...",
  "variants": ["...", "..."]
}}
"""

# -----------------------------------------------------------
# RECRUITER SYSTEM PROMPT
# -----------------------------------------------------------

ANSWER_SYSTEM = """
You are Smart Profile Matcher.

You are an AI-powered Talent Intelligence Assistant.

Your responsibility is to:

- Analyze resumes and candidate profiles
- Match profiles to recruiter requirements
- Identify relevant skills
- Recognize role synonyms
- Compare candidates
- Rank candidates based on relevance
- Highlight experience and qualifications

Instructions:

1. Use retrieved profile evidence.
2. Match similar titles and responsibilities.
3. Infer role suitability from skills and experience.
4. Explain WHY candidates are relevant.
5. Cite supporting passages using [1], [2], [3].
6. Never invent candidate information.

If no exact match exists:

- Identify the closest matching candidates.
- Explain why they might still be relevant.
- Suggest alternative searches.

Always respond using this format:

🎯 Match Summary

Profiles Found:
- Candidate / Resume
- Matching skills
- Relevant experience

Evidence:
[1][2]

Recommendation:
...
"""

# -----------------------------------------------------------
# ANSWER OBJECT
# -----------------------------------------------------------

@dataclass
class Answer:
    text: str
    hits: List[SearchHit] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)


# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------

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

        if (
            isinstance(content, str)
            and not (msg.get("metadata") or {}).get("title")
        ):
            lines.append(
                f"{msg.get('role', 'user')}: {content[:500]}"
            )

    return "\n".join(lines) or "(none)"


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------

class RAGPipeline:

    def __init__(
            self,
            llm=None,
            embedder=None,
            cfg: Settings = default_settings,
            index: Optional[HybridIndex] = None
    ):

        self.cfg = cfg

        self.llm = llm or HFChatModel(
            cfg.llm_model,
            cfg.hf_token,
            cfg.llm_provider,
            cfg.max_new_tokens,
            cfg.temperature
        )

        self._embedder = embedder

        self.index = (
            index
            if index is not None
            else HybridIndex.load(cfg.index_dir)
        )

    @property
    def embedder(self):

        if self._embedder is None:

            from .embeddings import build_embedder

            self._embedder = build_embedder(
                self.cfg.embedding_backend,
                self.cfg.embedding_model,
                self.cfg.hf_token
            )

        return self._embedder

    # -------------------------------------------------------
    # INGEST
    # -------------------------------------------------------

    def ingest(self, paths: List[Path]) -> str:

        report = []

        for path in paths:

            path = Path(path)

            try:

                digest = file_hash(path)

                if self.index.has_file(digest):
                    report.append(
                        f"⏭️ {path.name}: already indexed"
                    )
                    continue

                chunks = chunk_pages(
                    load_file(path),
                    self.cfg.chunk_size,
                    self.cfg.chunk_overlap
                )

                if not chunks:
                    report.append(
                        f"⚠️ {path.name}: no readable text detected"
                    )
                    continue

                vectors = self.embedder.embed(
                    [c.text for c in chunks]
                )

                self.index.add(
                    chunks,
                    vectors,
                    digest,
                    path.name
                )

                report.append(
                    f"✅ {path.name}: {len(chunks)} profile chunks"
                )

            except Exception as exc:

                logger.exception(
                    "Ingestion failed for %s",
                    path
                )

                report.append(
                    f"❌ {path.name}: {exc}"
                )

        self.index.save(self.cfg.index_dir)

        return "\n".join(report)

    def reset(self):

        self.index.clear()
        self.index.save(self.cfg.index_dir)

    # -------------------------------------------------------
    # QUERY EXPANSION
    # -------------------------------------------------------

    def build_queries(
            self,
            question: str,
            history: List[dict]
    ) -> List[str]:

        raw = self.llm.chat(
            [
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(
                        history=_history_text(history),
                        question=question
                    )
                }
            ],
            max_tokens=250,
            temperature=0.0
        )

        data = _extract_json(raw) or {}

        queries = [
            str(
                data.get("standalone")
                or question
            ).strip()
        ]

        for variant in data.get("variants") or []:

            variant = str(variant).strip()

            if (
                    variant
                    and variant.lower()
                    not in {
                        q.lower()
                        for q in queries
                    }
            ):
                queries.append(variant)

        return queries[:6]

    # -------------------------------------------------------
    # RETRIEVAL
    # -------------------------------------------------------

    def retrieve(
            self,
            queries: List[str],
            top_k: int
    ) -> List[SearchHit]:

        vectors = self.embedder.embed(queries)

        return self.index.search(
            queries,
            vectors,
            top_k,
            self.cfg.candidate_pool,
            self.cfg.rrf_k
        )

    # -------------------------------------------------------
    # ANSWER
    # -------------------------------------------------------

    def answer(
            self,
            question: str,
            history: Optional[List[dict]] = None,
            top_k: Optional[int] = None
    ) -> Answer:

        history = history or []

        if len(self.index) == 0:
            return Answer(
                "📂 No profiles indexed. Please upload resumes first."
            )

        queries = self.build_queries(
            question,
            history
        )

        hits = self.retrieve(
            queries,
            top_k or self.cfg.top_k
        )

        if not hits:

            return Answer(
                """
🎯 Match Summary

No strong profile matches were found.

Suggestions:

• Search by skills
• Search using broader job titles
• Search by technologies

Examples:

- Python Developer
- Azure Engineer
- Data Analyst
- Power BI Developer
                """,
                [],
                queries
            )

        context = "\n\n".join(
            f"""
[{i}]
Resume: {hit.chunk.source}
Page: {hit.chunk.page}

{hit.chunk.text}
"""
            for i, hit in enumerate(
                hits,
                start=1
            )
        )

        recruiter_prompt = f"""
Candidate Profile Evidence:

{context}

Recruiter Question:

{question}

Provide:

1. Match Summary
2. Likely number of matching profiles
3. Ranked candidates
4. Matching skills
5. Relevant experience
6. Evidence references
7. Recruiter recommendation

Be concise but useful.
"""

        messages = [
            {
                "role": "system",
                "content": ANSWER_SYSTEM
            }
        ]

        messages += [
            m
            for m in history[-4:]
            if (
                    isinstance(m.get("content"), str)
                    and not (m.get("metadata") or {}).get("title")
            )
        ]


        messages.append(
            {
                "role": "user",
                "content": recruiter_prompt
            }
        )

        result = self.llm.chat(messages)

        return Answer(
            result,
            hits,
            queries
        )


# -----------------------------------------------------------
# SOURCE FORMATTER
# -----------------------------------------------------------

def format_sources(answer: Answer) -> str:

    if not answer.hits:
        return ""

    lines = [
        "# 🎯 Profile Matching Evidence",
        "",
        f"**Expanded Searches:** {' • '.join(answer.search_queries)}",
        ""
    ]

    for i, hit in enumerate(answer.hits, start=1):

        dense = (
            f"#{hit.dense_rank}"
            if hit.dense_rank
            else "-"
        )

        sparse = (
            f"#{hit.sparse_rank}"
            if hit.sparse_rank
            else "-"
        )

        snippet = (
            hit.chunk.text[:250]
            .replace("\n", " ")
        )

        lines.append(
            f"""
### Candidate Evidence [{i}]

📄 Resume: {hit.chunk.source}

📑 Page: {hit.chunk.page}

🔍 Semantic Rank: {dense}

🔤 Keyword Rank: {sparse}

⭐ Match Score: {hit.score:.4f}

> {snippet}...

"""
        )

    return "\n".join(lines)