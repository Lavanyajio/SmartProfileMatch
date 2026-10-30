from rag_assistant.documents import Page, chunk_pages, split_text
from rag_assistant.hybrid_index import HybridIndex
from rag_assistant.llm import HFChatModel, LLMError
from rag_assistant.pipeline import RAGPipeline, format_sources

import pytest


def test_split_text_respects_chunk_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 50 + 2 for c in chunks)


def test_chunk_ids_are_unique():
    pages = [Page("alpha\n\nbeta " * 200, "a.md", 1), Page("gamma " * 300, "a.md", 2)]
    chunks = chunk_pages(pages, 200, 20)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_hybrid_search_and_persistence(tmp_path, fake_embedder):
    pages = [Page("Error E07 means battery cell imbalance. Charge to 100 percent.", "guide.md", 1),
             Page("Tyre pressure front 29 psi rear 33 psi.", "guide.md", 2)]
    chunks = chunk_pages(pages, 400, 50)
    index = HybridIndex()
    index.add(chunks, fake_embedder.embed([c.text for c in chunks]), "hash1", "guide.md")

    query = ["what is E07"]
    hits = index.search(query, fake_embedder.embed(query), top_k=2)
    assert hits[0].chunk.page == 1
    assert hits[0].sparse_rank == 1

    index.save(tmp_path)
    reloaded = HybridIndex.load(tmp_path)
    assert len(reloaded) == len(index)
    assert reloaded.has_file("hash1")


def test_pipeline_end_to_end(cfg, fake_llm, fake_embedder):
    rag = RAGPipeline(llm=fake_llm, embedder=fake_embedder, cfg=cfg, index=HybridIndex())
    report = rag.ingest(sorted(cfg.sample_docs_dir.glob("*.md")))
    assert report.count("✅") == 2
    assert "already indexed" in rag.ingest(sorted(cfg.sample_docs_dir.glob("*.md")))

    answer = rag.answer("what does that code mean?", history=[])
    assert "[1]" in answer.text
    assert answer.search_queries[0] == "What does error code E07 mean?"
    assert any("E07" in h.chunk.text for h in answer.hits)
    assert "Sources" not in format_sources(answer)  # title lives in the UI, body lists passages
    assert "[1]" in format_sources(answer)


def test_pipeline_empty_index_message(cfg, fake_llm, fake_embedder):
    rag = RAGPipeline(llm=fake_llm, embedder=fake_embedder, cfg=cfg, index=HybridIndex())
    assert "No documents" in rag.answer("hi").text
    assert fake_llm.calls == []


def test_rewrite_falls_back_on_bad_json(cfg, fake_embedder):
    class BadLLM:
        def chat(self, *a, **k):
            return "not json"
    rag = RAGPipeline(llm=BadLLM(), embedder=fake_embedder, cfg=cfg, index=HybridIndex())
    assert rag.build_queries("original question", []) == ["original question"]


def test_missing_token_raises_friendly_error():
    with pytest.raises(LLMError, match="HF_TOKEN"):
        HFChatModel("some/model", token="").chat([{"role": "user", "content": "hi"}])
