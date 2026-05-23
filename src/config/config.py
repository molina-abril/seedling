from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IngestionConfig:
    pdf_dir: Optional[Path] = None
    scopus_api_key: Optional[str] = None
    enable_scopus: bool = True
    enable_arxiv: bool = True
    dedup_threshold: float = 0.90
    keywords_synthesis_enabled: bool = True