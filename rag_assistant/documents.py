"""Loading files and splitting them into overlapping, citation-friendly chunks."""
import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

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


def load_file(
    path: Path,
    source_name: str = None,
) -> List[Page]:

    path = Path(path)

    if source_name is None:
        source_name = path.name

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Use PDF, TXT or MD."
        )

    # PDF Processing
    if suffix == ".pdf":

        reader = PdfReader(str(path))

        pages = []
        total_chars = 0

        # Standard text extraction
        for i, page in enumerate(reader.pages):

            text = page.extract_text() or ""

            total_chars += len(text)

            pages.append(
                Page(
                    text=text,
                    source=source_name,
                    page=i + 1,
                )
            )

        # OCR fallback
        if total_chars == 0:

            print(f"OCR fallback triggered for {source_name}")

            images = convert_from_path(str(path))

            pages = []

            for i, image in enumerate(images):

                ocr_text = pytesseract.image_to_string(image)

                print(
                    f"OCR Page {i + 1}: "
                    f"{len(ocr_text)} characters"
                )

                pages.append(
                    Page(
                        text=ocr_text,
                        source=source_name,
                        page=i + 1,
                    )
                )

        return pages

    # TXT / MD files
    return [
        Page(
            text=path.read_text(
                encoding="utf-8",
                errors="ignore"
            ),
            source=source_name,
            page=1,
        )
    ]
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
