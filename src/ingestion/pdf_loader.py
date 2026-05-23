"""Load papers from PDF files and convert to Paper objects."""

from pathlib import Path
from typing import List
from pypdf import PdfReader

from src.models.paper import Paper
from src.utils.text import sanitize_unicode, clean_pdf_content


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from all pages of a PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Concatenated text from all pages
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_papers_from_pdf_directory(
    pdf_dir: str,
    min_words: int = 120,
    verbose: bool = False
) -> List[Paper]:
    """
    Load PDF files from directory and convert to Paper objects.
    
    Args:
        pdf_dir: Path to directory containing PDF files
        min_words: Minimum word count to include (skip poorly extracted PDFs)
        verbose: Print progress
        
    Returns:
        List of Paper objects created from PDFs
    """
    pdf_dir = Path(pdf_dir)
    papers = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if verbose:
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    for idx, pdf_file in enumerate(pdf_files):
        if verbose:
            print(f"Processing {idx + 1}/{len(pdf_files)}: {pdf_file.name}")

        try:
            raw_text = extract_pdf_text(pdf_file)
            cleaned_text = clean_pdf_content(raw_text)

            word_count = len(cleaned_text.split())
            if word_count < min_words:
                if verbose:
                    print(f"  ⚠️  Skipped: only {word_count} words (min: {min_words})")
                continue

            paper = Paper(
                paper_id=f"pdf_{pdf_file.stem}",
                title=sanitize_unicode(pdf_file.stem),
                abstract=cleaned_text,
                keywords=[],
                authors=[],
                year=None,
                doi=None,
                source="pdf",
                venue=None,
                url=str(pdf_file),
                citations_count=0,
            )
            papers.append(paper)
            
            if verbose:
                print(f"  ✓ Added: {word_count} words")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error processing {pdf_file.name}: {e}")
            continue

    if verbose:
        print(f"\nLoaded {len(papers)} papers from PDFs")

    return papers
