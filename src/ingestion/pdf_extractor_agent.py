"""PDF extraction and metadata discovery agent."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

from src.models import Paper, Provenance

logger = logging.getLogger(__name__)

class PDFExtractorAgent:
    """Agent for extracting metadata from local PDF files."""

    def __init__(self, pdf_dir: Optional[Path] = None):
        """Initialize the PDF extractor.

        Args:
            pdf_dir: Directory containing PDFs. Defaults to papers/
        """
        if pdf_dir is None:
            pdf_dir = Path(__file__).parent.parent.parent / "papers"

        self.pdf_dir = Path(pdf_dir)

    def extract_from_file(self, pdf_path: Path) -> Optional[Paper]:
        """Extract metadata from a single PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Paper object if extraction succeeded, None otherwise.
        """
        reader = self._read_pdf(pdf_path)
        if not reader:
            return None

        num_pages = len(reader.pages)
        title, doi = self._extract_from_metadata(reader, pdf_path)
        first_pages_text = self._extract_text_from_first_pages(reader)
        first_pages_metadata = self._extract_metadata_from_first_pages(first_pages_text)

        if title and self._title_looks_like_filename(title):
            candidate = first_pages_metadata.get("title_candidate")
            if candidate and not self._title_looks_like_filename(candidate):
                logger.info(
                    "Discarding suspicious /Title for %s: %r -> using first-pages candidate %r",
                    pdf_path.name,
                    title,
                    candidate,
                )
                title = None

        logger.info(
            "Extracted from PDF: %s | Title: %s | DOI: %s | FirstPagesTitle: %s | FirstPagesDOI: %s | FirstPagesAuthors: %s | FirstPagesKeywords: %s",
            pdf_path.name,
            title,
            doi,
            first_pages_metadata.get("title_candidate"),
            first_pages_metadata.get("doi_candidate"),
            first_pages_metadata.get("authors_candidate"),
            first_pages_metadata.get("keywords_candidate"),
        )

        if not title and first_pages_metadata.get("title_candidate"):
            title = first_pages_metadata["title_candidate"]
        if not doi and first_pages_metadata.get("doi_candidate"):
            doi = self._normalize_doi(first_pages_metadata["doi_candidate"])

        abstract = first_pages_metadata.get("abstract_candidate") or self._extract_from_text(reader)
        keywords = first_pages_metadata.get("keywords_candidate") or []
        authors = first_pages_metadata.get("authors_candidate") or []
        leading_author = authors[0] if authors else None
        year = first_pages_metadata.get("year_candidate")
        venue = first_pages_metadata.get("venue_candidate")

        if not title:
            title = self._infer_title_from_filename(pdf_path)

        if not title:
            return None

        paper_id = self._generate_paper_id(title, doi, pdf_path)

        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            keywords=keywords,
            authors=authors,
            year=year,
            doi=doi,
            source="local_pdf",
            venue=venue,
            url=None,
            citations_count=0,
            provenance=Provenance(
                retrieved_from=["local_pdf"],
                original_query=pdf_path.name,
            ),
            metadata={
                "pdf_path": str(pdf_path),
                "pdf_num_pages": num_pages,
                "extraction_method": "pypdf",
                "first_pages_text_excerpt": first_pages_text[:2000],
                "title_candidate": first_pages_metadata.get("title_candidate"),
                "doi_candidate": first_pages_metadata.get("doi_candidate"),
                "abstract_candidate": first_pages_metadata.get("abstract_candidate"),
                "authors_candidate": first_pages_metadata.get("authors_candidate"),
                "first_author_candidate": leading_author,
                "keywords_candidate": first_pages_metadata.get("keywords_candidate"),
                "year_candidate": first_pages_metadata.get("year_candidate"),
                "venue_candidate": first_pages_metadata.get("venue_candidate"),
            },
        )

    def _read_pdf(self, pdf_path: Path) -> Optional[PdfReader]:
        """Read a PDF file and return a PdfReader object."""
        if not pdf_path.exists():
            return None

        try:
            return PdfReader(pdf_path)
        except Exception as e:
            logger.warning(f"Error reading PDF {pdf_path}: {e}")
            return None

    def extract_from_directory(self, max_files: Optional[int] = None) -> List[Paper]:
        """Extract papers from all PDFs in the directory.

        Args:
            max_files: Maximum number of files to process. None for all.

        Returns:
            List of extracted Paper objects.
        """
        pdf_files = sorted(self.pdf_dir.rglob("*.pdf"))
        if max_files:
            pdf_files = pdf_files[:max_files]

        papers = []
        for pdf_path in pdf_files:
            paper = self.extract_from_file(pdf_path)
            if paper:
                papers.append(paper)

        return papers

    @staticmethod
    def _title_looks_like_filename(title: str) -> bool:
        """Heuristic: True if ``title`` resembles an internal publisher filename.

        Catches strings like 'BFJ-02-2023-0132_proof 436..461' or
        'degruyter_jci_jci-2022-0078 1..12 ++' where the publisher embeds a
        non-human-readable identifier in the PDF's /Title metadata.
        """
        if not title:
            return True
        t = title.strip()
        if len(t) < 8:
            return True
        if re.search(r"\+{2,}|_{2,}|\\n", t):
            return True
        if re.search(r"\b\d{1,4}\.\.\d{1,4}\b", t):
            return True
        if re.search(r"_proof\b", t, re.IGNORECASE):
            return True
        if sum(c.isdigit() for c in t) / max(1, len(t)) > 0.2:
            return True
        return False

    def _extract_from_metadata(self, reader: PdfReader, pdf_path: Path) -> tuple[Optional[str], Optional[str]]:
        """Extract title and DOI from PDF metadata."""
        metadata = reader.metadata
        if not metadata:
            return None, None

        title = metadata.get("/Title")
        if title:
            title = str(title).strip()

        doi = metadata.get("/DOI")
        if not doi:
            subject = metadata.get("/Subject", "")
            doi = self._extract_doi_from_text(subject)

        if doi:
            doi = self._normalize_doi(doi)

        return title, doi

    def _extract_text_from_first_pages(self, reader: PdfReader, max_pages: int = 2) -> str:
        """Extract text from the first pages for title/DOI heuristics."""
        texts = []
        for index, page in enumerate(reader.pages):
            if index >= max_pages:
                break
            try:
                page_text = page.extract_text() or ""
                page_text = re.sub(r"\n?\s*\d+\s*/\s*\d+\s*\n?", "\n", page_text)
                texts.append(page_text)
            except Exception:
                texts.append("")
        return "\n".join(texts)

    def _extract_title_from_first_pages_text(self, text: str) -> Optional[str]:
        """Heuristically extract a title from the first pages text."""
        if not text:
            return None

        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return None

        header_patterns = (
            r"^published as",
            r"^available online",
            r"^conference paper",
            r"^copyright",
            r"^preprint",
            r"^arxiv",
            r"^journal homepage",
        )
        stop_patterns = (
            r"^abstract$",
            r"^keywords?$",
            r"^index\s+terms?$",
            r"^introduction$",
            r"^contents lists available",
            r"^received ",
            r"^accepted ",
            r"^doi:",
            r"^https?://doi\.org/",
        )

        def is_header(line: str) -> bool:
            lowered = line.lower()
            return any(re.search(pattern, lowered) for pattern in header_patterns)

        def is_stop_line(line: str) -> bool:
            lowered = line.lower()
            if any(re.search(pattern, lowered) for pattern in stop_patterns):
                return True
            if "@" in line:
                return True
            if re.search(r"\b(university|department|school|institute|college|laboratory|faculty|centre|center)\b", lowered):
                return True
            if self._looks_like_author_line(line):
                return True
            return False

        title_lines = []
        started = False
        for line in lines[:40]:
            if is_header(line) and not started:
                continue
            if is_stop_line(line):
                if started:
                    break
                continue

            if len(line) < 4:
                continue
            alpha_count = sum(char.isalpha() for char in line)
            if alpha_count < 6:
                continue
            if re.search(r"\b(university|department|institute|laboratory|center|centre|school|college)\b", line, re.IGNORECASE):
                continue

            if not started:
                started = True
            if re.match(r"^\d+$", line):
                break

            title_lines.append(line)

        if not title_lines:
            return None

        candidate = " ".join(title_lines)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        candidate = self._clean_title_candidate(candidate)
        if not candidate or len(candidate) < 8:
            return None
        return candidate

    def _clean_title_candidate(self, title: Optional[str]) -> Optional[str]:
        """Remove obvious author, affiliation, and section noise from a title candidate."""
        if not title or not isinstance(title, str):
            return None

        text = re.sub(r"\s+", " ", title).strip()
        if not text:
            return None

        cut_markers = [
            r"\babstract\b",
            r"\bkeywords?\b",
            r"\bindex\s+terms?\b",
            r"\bintroduction\b",
            r"\breceived\b",
            r"\baccepted\b",
            r"\bavailable online\b",
            r"\bdoi\s*[:=]",
            r"\bhttps?://doi\.org/",
        ]
        marker_positions = []
        for pattern in cut_markers:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                marker_positions.append(match.start())
        if marker_positions:
            text = text[: min(marker_positions)].strip()

        affiliation_match = re.search(
            r"\b(university|department|school|institute|college|laboratory|faculty|centre|center|knowledge verse ai)\b",
            text,
            re.IGNORECASE,
        )
        if affiliation_match and affiliation_match.start() > 0:
            prefix = text[: affiliation_match.start()].strip()
            if len(prefix.split()) >= 4:
                text = prefix

        author_suffix = re.search(
            r",\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+)*\s*$",
            text,
        )
        if author_suffix and author_suffix.start() > 0:
            prefix = text[: author_suffix.start()].strip()
            if len(prefix.split()) >= 4:
                text = prefix

        text = re.sub(r"\s+", " ", text).strip(" ,;:-")
        return text if len(text) >= 8 else None

    @staticmethod
    def _looks_like_author_line(line: str) -> bool:
        """Return True when a line looks like a list of authors rather than a title."""
        if not line:
            return False

        normalized = re.sub(r"\s+", " ", line).strip(" ,;:-")
        if len(normalized.split()) < 4:
            return False

        if not re.search(r"\b(?:and|,|;)\b", normalized, re.IGNORECASE):
            return False

        author_chunks = re.split(r"\band\b|,|;", normalized)
        matches = 0
        for chunk in author_chunks:
            chunk = re.sub(r"\b[a-z]\b", "", chunk)
            chunk = re.sub(r"\b\d+\b", "", chunk)
            chunk = re.sub(r"\s+", " ", chunk).strip(" ,;:-")
            if not chunk:
                continue
            words = chunk.split()
            if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*", word) for word in words):
                matches += 1

        return matches >= 2

    def _extract_metadata_from_first_pages(self, text: str) -> Dict[str, Any]:
        """Extract a compact metadata bundle from the first pages text."""
        metadata: Dict[str, Any] = {
            "title_candidate": None,
            "doi_candidate": None,
            "abstract_candidate": None,
            "authors_candidate": [],
            "keywords_candidate": [],
            "year_candidate": None,
            "venue_candidate": None,
        }

        if not text:
            return metadata

        metadata["title_candidate"] = self._extract_title_from_first_pages_text(text)
        metadata["doi_candidate"] = self._extract_doi_from_text(text)
        metadata["abstract_candidate"] = self._extract_abstract_from_first_pages_text(text)
        metadata["keywords_candidate"] = self._extract_keywords_from_first_pages_text(text)
        metadata["authors_candidate"] = self._extract_authors_from_first_pages_text(text, metadata["title_candidate"])
        metadata["year_candidate"] = self._extract_year_from_first_pages_text(text)
        metadata["venue_candidate"] = self._extract_venue_from_first_pages_text(text)

        return metadata

    def _extract_abstract_from_first_pages_text(self, text: str) -> Optional[str]:
        """Extract an abstract/resumen block from the first pages text."""
        if not text:
            return None

        match = re.search(
            r"(?:abstract|resumen|summary)[:\s]+(.+?)(?=\n\s*(?:keywords?|index\s+terms?|introduction|1\s+introduction|received|accepted)\b|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None

        abstract = re.sub(r"\s+", " ", match.group(1)).strip()
        if len(abstract) < 40:
            return None
        return abstract[:1500]

    def _extract_keywords_from_first_pages_text(self, text: str) -> List[str]:
        """Extract author keywords or index terms from the first pages text."""
        if not text:
            return []

        match = re.search(
            r"(?:keywords?|index\s+terms?)[:\s]+(.+?)(?=\n\s*(?:abstract|introduction|1\s+introduction|received|accepted)\b|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return []

        raw_keywords = re.split(r"[;,\n]", match.group(1))
        keywords = []
        for keyword in raw_keywords:
            cleaned = re.sub(r"\s+", " ", keyword).strip(" .:-")
            if len(cleaned) < 2:
                continue
            if cleaned.lower() in {"and", "or"}:
                continue
            keywords.append(cleaned)
        return keywords

    def _extract_authors_from_first_pages_text(self, text: str, title_candidate: Optional[str] = None) -> List[str]:
        """Extract likely author names from the first pages text."""
        if not text:
            return []

        header_text = self._extract_header_block(text)
        if not header_text:
            return []

        authors: List[str] = []
        lines = [re.sub(r"\s+", " ", line).strip() for line in header_text.splitlines()]
        lines = [line for line in lines if line]

        collecting = False
        for index, line in enumerate(lines[:20]):
            lowered = line.lower()
            if any(marker in lowered for marker in ["abstract", "keywords", "index terms", "introduction", "received", "accepted"]):
                if collecting:
                    break
                continue
            if "@" in line:
                if collecting:
                    break
                continue
            if re.search(r"\b(university|department|school|institute|college|laboratory|faculty|centre|center)\b", lowered):
                if collecting:
                    break
                continue

            if not collecting:
                if self._looks_like_author_line(line) or self._looks_like_single_author_name_line(line, lines, index):
                    collecting = True
                else:
                    continue

            for author in self._extract_authors_from_line(line):
                if author and author not in authors:
                    authors.append(author)

            if len(authors) >= 8:
                break
        return authors

    def _extract_authors_from_line(self, line: str) -> List[str]:
        """Extract one or more author names from a single line."""
        authors: List[str] = []
        for chunk in re.split(r"\band\b|,|;", line):
            cleaned = self._clean_author_candidate(chunk)
            if cleaned and cleaned not in authors:
                authors.append(cleaned)
        return authors

    @staticmethod
    def _looks_like_single_author_name_line(line: str, lines: List[str], index: int) -> bool:
        """Heuristic for a lone author name on its own line."""
        if not line:
            return False

        normalized = re.sub(r"\s+", " ", line).strip(" ,;:-")
        words = normalized.split()
        if len(words) < 2 or len(words) > 4:
            return False
        if any(marker in normalized.lower() for marker in ["abstract", "keywords", "introduction", "received", "accepted"]):
            return False
        if any(marker in normalized.lower() for marker in ["university", "department", "school", "institute", "college", "laboratory", "faculty", "centre", "center"]):
            return False
        if not all(re.fullmatch(r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*", word) for word in words):
            return False

        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        next_lowered = next_line.lower()
        if re.search(r"\b(university|department|school|institute|college|laboratory|faculty|centre|center)\b", next_lowered):
            return True
        if re.search(r"\b(?:and|,|;)\b", next_line, re.IGNORECASE):
            return True
        if "@" in next_line:
            return True

        return False

    def _clean_author_candidate(self, chunk: str) -> Optional[str]:
        """Normalize a name-like chunk and keep it only if it looks like an author."""
        if not chunk:
            return None

        text = re.sub(r"\s+", " ", chunk).strip(" .:-")
        if not text:
            return None

        lowered = text.lower()
        if any(marker in lowered for marker in ["abstract", "keywords", "introduction", "received", "accepted"]):
            return None

        text = re.sub(r"\b[a-z]\b", "", text)
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"\s+", " ", text).strip(" .:-")

        affiliation_markers = [
            "university",
            "institute",
            "school",
            "college",
            "department",
            "laboratory",
            "centre",
            "center",
            "knowledge verse ai",
        ]
        for marker in affiliation_markers:
            if marker in text.lower():
                prefix = re.split(marker, text, flags=re.IGNORECASE)[0].strip(" ,;:-")
                if len(prefix.split()) >= 2:
                    text = prefix
                else:
                    return None
                break

        words = text.split()
        if len(words) < 2:
            return None

        for size in range(min(4, len(words)), 1, -1):
            candidate = " ".join(words[:size]).strip(" .:-")
            if self._looks_like_person_name(candidate):
                return candidate

        return None

    @staticmethod
    def _looks_like_person_name(text: str) -> bool:
        """Return True when a chunk looks like a personal name."""
        if not text:
            return False

        allowed_particles = {"de", "del", "da", "di", "van", "von", "la", "le", "du", "dos", "das"}
        words = text.split()
        if len(words) < 2 or len(words) > 4:
            return False

        for word in words:
            lowered = word.lower().strip(".")
            if lowered in allowed_particles:
                continue
            if re.fullmatch(r"[A-Z]\.?", word):
                continue
            if not re.fullmatch(r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*", word):
                return False
        return True

    def _extract_year_from_first_pages_text(self, text: str) -> Optional[int]:
        """Extract a likely publication year from the first pages text."""
        if not text:
            return None

        explicit_year = re.search(
            r"\b(?:received|accepted|published(?: online)?|available online)\b[^0-9]{0,40}((?:19|20)\d{2})",
            text,
            re.IGNORECASE,
        )
        if explicit_year:
            return int(explicit_year.group(1))

        header_block = self._extract_header_block(text)
        if not header_block:
            return None

        generic_year = re.search(r"\b((?:19|20)\d{2})\b", header_block)
        if generic_year:
            return int(generic_year.group(1))

        return None

    def _extract_venue_from_first_pages_text(self, text: str) -> Optional[str]:
        """Extract a likely venue/journal/conference name from the first pages text."""
        if not text:
            return None

        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        venue_markers = (
            r"proceedings",
            r"transactions",
            r"journal",
            r"conference",
            r"symposium",
            r"workshop",
            r"letters",
            r"magazine",
            r"review",
        )
        for line in lines[:20]:
            lowered = line.lower()
            if any(marker in lowered for marker in venue_markers):
                if any(marker in lowered for marker in ["abstract", "keywords", "introduction", "received", "accepted"]):
                    continue
                if len(line.split()) >= 3:
                    return line

        return None

    def _extract_header_block(self, text: str) -> str:
        """Return the text before the abstract/keywords section when possible."""
        if not text:
            return ""

        match = re.search(
            r"\n\s*(?:abstract|resumen|summary|keywords?|index\s+terms?|introduction|1\s+introduction|received|accepted)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return text[: match.start()].strip()
        return text.strip()

    @staticmethod
    def _extract_doi_from_text(text: str) -> Optional[str]:
        """Extract DOI from arbitrary text."""
        if not text:
            return None

        match = re.search(
            r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b",
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        doi = match.group(1).rstrip(".,);]")
        return PDFExtractorAgent._normalize_doi(doi)

    def _extract_from_text(self, reader: PdfReader) -> Optional[str]:
        """Extract abstract from first page of PDF."""
        try:
            if len(reader.pages) == 0:
                return None

            first_page = reader.pages[0]
            text = first_page.extract_text() or ""
            abstract = self._extract_abstract_from_first_pages_text(text)
            if abstract:
                return abstract[:500]

            return None
        except Exception:
            return None

    def _infer_title_from_filename(self, pdf_path: Path) -> Optional[str]:
        """Infer title from filename."""
        filename = pdf_path.stem

        filename = re.sub(r'-\d+$', '', filename)
        filename = re.sub(r'v\d+$', '', filename)
        filename = re.sub(r'_', ' ', filename)
        filename = re.sub(r'-', ' ', filename)
        filename = re.sub(r'\s+', ' ', filename)
        filename = filename.strip()

        if len(filename) > 3:
            return filename
        return None

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        """Normalize a DOI string."""
        doi = str(doi).lower()
        doi = doi.replace("https://doi.org/", "")
        doi = doi.replace("http://doi.org/", "")
        return doi.strip(" /")

    _ARXIV_FILENAME_RE = re.compile(r'(?:^|[^\d])(\d{4}\.\d{4,5})(v\d+)?(?=\D|$)')
    _ISBN_FILENAME_RE = re.compile(r'(97[89](?:[-\d]){10,17})')

    @staticmethod
    def _generate_paper_id(
        title: str,
        doi: Optional[str],
        pdf_path: Optional[Path] = None,
    ) -> str:
        """Return a deterministic paper_id, based on a canonical identifier when
        possible. Order of preference:
          1. Normalized DOI.
          2. arXiv id extracted from the PDF filename (e.g. 2308.08155v2).
          3. ISBN extracted from the PDF filename (books / chapters).
          4. SHA1(normalized_title + filename) -> 12 hex.
        """
        if doi:
            return f"p_{doi.replace('.', '_').replace('/', '_')}"

        if pdf_path is not None:
            filename = Path(pdf_path).name
            arxiv_match = PDFExtractorAgent._ARXIV_FILENAME_RE.search(filename)
            if arxiv_match:
                return f"p_arxiv_{arxiv_match.group(1).replace('.', '_')}"
            isbn_match = PDFExtractorAgent._ISBN_FILENAME_RE.search(filename)
            if isbn_match:
                return f"p_isbn_{isbn_match.group(1).replace('-', '')}"
            stem = Path(pdf_path).stem
        else:
            stem = ''

        normalized_title = re.sub(r'\s+', ' ', (title or '').strip().lower())
        digest_input = f"{normalized_title}|{stem.lower()}".encode('utf-8')
        digest = hashlib.sha1(digest_input).hexdigest()[:12]
        return f"p_h_{digest}"
