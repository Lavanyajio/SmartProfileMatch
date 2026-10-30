"""Central configuration. Every value can be overridden with an environment variable or a .env file."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env",override=True)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Hugging Face
    hf_token: str = os.getenv("HF_TOKEN", "")
    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    llm_provider: str = os.getenv("LLM_PROVIDER", "")  # empty = Hugging Face auto-routing
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "local")  # "api" or "local"
    max_new_tokens: int = _int("MAX_NEW_TOKENS", 700)
    temperature: float = _float("TEMPERATURE", 0.2)

    # Retrieval
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)
    top_k: int = _int("TOP_K", 5)
    candidate_pool: int = _int("CANDIDATE_POOL", 20)
    rrf_k: int = _int("RRF_K", 60)

    # Paths
    index_dir: Path = Path(os.getenv("INDEX_DIR", BASE_DIR / "storage" / "index"))
    sample_docs_dir: Path = BASE_DIR / "data" / "sample_docs"

    # Server
    host: str = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port: int = _int("GRADIO_SERVER_PORT", 7860)


settings = Settings()
if settings.groq_api_key:
    print("Groq key loaded")
else:
    print("Groq key missing")
    print("HF_TOKEN exists:", bool(settings.hf_token))
    print("Embedding Backend:", settings.embedding_backend)
    print("Embedding Model:", settings.embedding_model)