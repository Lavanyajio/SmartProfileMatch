"""Loading files and splitting them into overlapping, citation-friendly chunks."""
import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class Page:
    text: str
    source: str
    page: int


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int

    def to_dict(self) -> dict:
        return asdict(self)


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_file(path: Path) -> List[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF, TXT or MD.")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [Page(page.extract_text() or "", path.name, i + 1) for i, page in enumerate(reader.pages)]
    return [Page(path.read_text(encoding="utf-8", errors="ignore"), path.name, 1)]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Paragraph-aware packing; very long paragraphs are cut with a sliding window."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        step = max(chunk_size - overlap, 1)
        pieces.extend(para[i:i + chunk_size] for i in range(0, len(para), step))

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages: List[Page], chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page in pages:
        for i, text in enumerate(split_text(_clean(page.text), chunk_size, overlap)):
            chunks.append(Chunk(f"{page.source}:p{page.page}:c{i}", text, page.source, page.page))
    return chunks
