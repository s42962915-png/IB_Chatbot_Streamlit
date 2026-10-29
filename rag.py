"""Local, dependency-light retrieval for the InBody Knowledge Assistant.

Parses approved documents in ./knowledge_base (PDF, DOCX, PPTX, TXT, MD),
chunks them, and ranks passages with BM25 keyword scoring.

Retrieval is fully local and free - no paid API and no model weights are
required for the search step. The LLM (Groq) only ever sees the retrieved
excerpts, never the open web.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Plus

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    # original tokens (keep "270s", "inbody270" intact)
    orig = re.findall(r"[a-z0-9]+", text)
    # split letter<->digit boundaries so "InBody270" and "InBody 270" match
    split = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    split = re.sub(r"(\d)([a-z])", r"\1 \2", split)
    tokens = orig + re.findall(r"[a-z0-9]+", split)
    # light normalization: strip plural suffixes only from pure-word tokens
    out = []
    seen = set()
    for t in tokens:
        if t.isalpha():
            if len(t) > 4 and t.endswith("ies"):
                t = t[:-3] + "y"
            elif len(t) > 3 and t.endswith("es"):
                t = t[:-2]
            elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _sniff_suffix(path: Path) -> str:
    """Detect a file's real type from magic bytes (some catalogues have no extension)."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return ""
    if head.startswith(b"%PDF"):
        return ".pdf"
    if head.startswith(b"PK"):
        import zipfile

        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except Exception:
            return ""
        if any(n.startswith("ppt/") for n in names):
            return ".pptx"
        if any(n.startswith("word/") for n in names):
            return ".docx"
        if any(n.startswith("xl/") for n in names):
            return ".xlsx"
    return ""


# --- text extraction -------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    try:
        import pymupdf

        doc = pymupdf.open(str(path))
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text_frame:
                txt = "\n".join(
                    p.text for p in shape.text_frame.paragraphs if p.text.strip()
                )
                if txt.strip():
                    parts.append(txt)
    return "\n".join(parts)


def _extract_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    rows_out = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                rows_out.append(" | ".join(cells))
    wb.close()
    return "\n".join(rows_out)


def extract_text(path: Path) -> str:
    suffix = _sniff_suffix(path) or path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".pptx":
        return _extract_pptx(path)
    if suffix == ".xlsx":
        return _extract_xlsx(path)
    if suffix in {".txt", ".md"}:
        return _extract_text_file(path)
    return ""


# --- chunking --------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = text[start:end]
            stops = [window.rfind(p) for p in (". ", "! ", "? ", "\n") if window.rfind(p) >= 0]
            last_stop = max(stops) if stops else -1
            if last_stop > max_chars * 0.5:
                end = start + last_stop + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < n else n
    return chunks


# --- knowledge base --------------------------------------------------------

class KnowledgeBase:
    def __init__(self, kb_dir: str | Path, rebuild: bool = False):
        self.kb_dir = Path(kb_dir)
        self.chunks: list[tuple[str, str]] = []  # (filename, chunk)
        self._bm25: BM25Plus | None = None
        self._cache_path = Path(__file__).parent / "kb_index.json"
        if not rebuild and self._cache_path.exists():
            self._load_cache()
        else:
            self._build()
        if self.chunks:
            self._bm25 = BM25Plus([_tokenize(c[1]) for c in self.chunks])

    def _load_cache(self) -> None:
        data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        self.chunks = [(d.get("f", ""), d.get("c", "")) for d in data]

    def save_cache(self) -> None:
        payload = [{"f": f, "c": c} for f, c in self.chunks]
        self._cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def reset_cache(self) -> None:
        if self._cache_path.exists():
            self._cache_path.unlink()

    def _build(self) -> None:
        for path in sorted(self.kb_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name.endswith(".part"):
                continue
            suffix = _sniff_suffix(path) or path.suffix.lower()
            if suffix not in SUPPORTED:
                continue
            text = extract_text(path)
            for chunk in chunk_text(text):
                self.chunks.append((path.name, chunk))

    @property
    def document_count(self) -> int:
        return len({c[0] for c in self.chunks})

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 6) -> list[tuple[str, str]]:
        if not self._bm25:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Boost passages whose FILENAME matches a model name in the query
        # (e.g. "Compare 270 vs 270S" should prioritise the 270/270S catalogues,
        #  not documents that merely mention "270 kg").
        for i, (fname, _) in enumerate(self.chunks):
            fname_l = fname.lower()
            for tok in tokens:
                if len(tok) >= 3 and tok in fname_l:
                    scores[i] += 4.0
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            if scores[i] <= 0:
                continue
            results.append(self.chunks[i])
            if len(results) >= top_k:
                break
        return results
