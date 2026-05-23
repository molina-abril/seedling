"""Text cleaning and preprocessing utilities."""

import re
import unicodedata
from typing import List
import importlib.resources as pkg_resources
import chardet
import stws


def sanitize_unicode(text: str) -> str:
    """
    Remove invalid Unicode characters that break UTF-8 encoding.
    
    Args:
        text: Input text with potential invalid Unicode
        
    Returns:
        Cleaned text with only valid UTF-8 characters
    """
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def remove_accents(text: str) -> str:
    """
    Remove accent marks from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with accents removed
    """
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    return str(text)


def clean_pdf_content(text: str) -> str:
    """
    Clean extracted PDF text.
    
    - Sanitize Unicode
    - Remove null characters
    - Normalize whitespace
    - Remove references section if present
    
    Args:
        text: Raw PDF text
        
    Returns:
        Cleaned text
    """
    text = sanitize_unicode(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    parts = re.split(
        r"\b(references|bibliography)\b",
        text,
        flags=re.IGNORECASE
    )
    text = parts[0]

    return text.strip()


def remove_corpus_artifacts(corpus: List[str], verbose: bool = False) -> List[str]:
    """
    Remove URLs, publisher names, DOIs, and other artifacts from corpus.
    
    Removes:
    - URLs (http://, https://, ftp://, www., etc.)
    - DOIs and ISBNs
    - Publisher/journal names (Wiley, Springer, Elsevier, IEEE, ACM, etc.)
    - Email addresses
    - Truncated words from PDF extraction
    - Reference metadata blocks
    
    Args:
        corpus: List of documents
        verbose: Print progress
        
    Returns:
        Cleaned corpus with artifacts removed
    """
    if verbose:
        print('Removing Corpus Artifacts: Working...')

    publishers = {
        'wiley', 'springer', 'elsevier', 'ieee', 'acm', 'sage', 'taylor',
        'francis', 'oxford', 'cambridge', 'nature', 'science', 'plos',
        'arxiv', 'biorxiv', 'medrxiv', 'ssrn', 'jstor', 'tandfonline',
        'pergamon', 'routledge', 'guild', 'informa', 'gale', 'proquest'
    }
    
    cleaned_corpus = []
    for text in corpus:
        text = re.sub(r'https?://[^\s]+', ' ', text)
        text = re.sub(r'ftp://[^\s]+', ' ', text)
        text = re.sub(r'www\.[^\s]+', ' ', text)
        text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', ' ', text)

        text = re.sub(r'\b(?:doi|isbn|issn)[:\s]?[0-9.\-/]+\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:doi|ISBN|ISSN)\s*[:\-]?\s*[0-9.\-/()a-zA-Z]+\b', ' ', text)

        text = re.sub(r'\b(?:arXiv|bioRxiv|medRxiv):[0-9.]+\b', ' ', text, flags=re.IGNORECASE)

        text = re.sub(r'\bet\s+al\.?,?\s+\d{4}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\(\w+,?\s+\d{4}\)', ' ', text)

        text = re.sub(r'\b(?:pp?|pages?|p\.?)\s*\d+[\-–]\d+\b', ' ', text, flags=re.IGNORECASE)

        text = re.sub(r'\b(?:vol(?:ume)?|no|issue|number)\.?\s+\d+\b', ' ', text, flags=re.IGNORECASE)

        publisher_pattern = r'\b(' + '|'.join(re.escape(p) for p in publishers) + r')\b'
        text = re.sub(publisher_pattern, ' ', text, flags=re.IGNORECASE)

        text = re.sub(r'\b(?:scienti|technolog|organisati|organisat|implementati|administrati|characteristi|strategi|decision|communicat|significan|responsibil)\b', ' ', text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text).strip()

        cleaned_corpus.append(text)
    
    if verbose:
        print('Removing Corpus Artifacts: Done!')
    
    return cleaned_corpus


def clear_text(
    corpus: List[str],
    stop_words: List[str] = None,
    lowercase: bool = True,
    rmv_accents: bool = True,
    rmv_special_chars: bool = True,
    rmv_numbers: bool = True,
    rmv_custom_words: List[str] = None,
    verbose: bool = False
) -> List[str]:
    """
    Preprocess text corpus.
    
    Args:
        corpus: List of documents
        stop_words: Language codes ('en', 'es', etc.) for stopword removal
        lowercase: Convert to lowercase
        rmv_accents: Remove accent marks
        rmv_special_chars: Remove special characters (keep alphanumeric + space)
        rmv_numbers: Remove numbers
        rmv_custom_words: Custom words to remove
        verbose: Print progress
        
    Returns:
        Cleaned corpus
    """
    if stop_words is None:
        stop_words = []
    if rmv_custom_words is None:
        rmv_custom_words = []

    corpus = remove_corpus_artifacts(corpus, verbose=verbose)

    sw_full = []

    if lowercase:
        if verbose:
            print('Lower Case: Working...')
        corpus = [str(x).lower().replace("'", "'") for x in corpus]
        if verbose:
            print('Lower Case: Done!')

    if rmv_special_chars:
        if verbose:
            print('Removing Special Characters: Working...')
        corpus = [re.sub(r"[^a-zA-Z0-9']+", ' ', i) for i in corpus]
        if verbose:
            print('Removing Special Characters: Done!')

    if len(stop_words) > 0:
        for sw_ in stop_words:
            lang_map = {
                'en': 'Stopwords-English.txt',
                'eng': 'Stopwords-English.txt',
                'english': 'Stopwords-English.txt',
            }
            
            name = lang_map.get(sw_.lower(), f'Stopwords-{sw_.capitalize()}.txt')
            
            try:
                with pkg_resources.open_binary(stws, name) as file:
                    raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                with pkg_resources.open_text(stws, name, encoding=encoding) as file:
                    content = file.read().split('\n')
                content = [line.rstrip('\r').rstrip('\n') for line in content]
                sw = list(filter(None, content))
                sw_full.extend(sw)
            except FileNotFoundError:
                if verbose:
                    print(f'Warning: Stopwords file {name} not found, skipping')
                continue
        
        if verbose:
            print('Removing Stopwords: Working...')
        for i in range(0, len(corpus)):
            text = corpus[i].split()
            text = [x.replace(' ', '') for x in text if x.replace(' ', '') not in sw_full]
            corpus[i] = ' '.join(text)
            if verbose:
                print(f'Removing Stopwords: {i + 1} of {len(corpus)}')
        if verbose:
            print('Removing Stopwords: Done!')

    if len(rmv_custom_words) > 0:
        if verbose:
            print('Removing Custom Words: Working...')
        for i in range(0, len(corpus)):
            text = corpus[i].split()
            text = [x.replace(' ', '') for x in text if x.replace(' ', '') not in rmv_custom_words]
            corpus[i] = ' '.join(text)
            if verbose:
                print(f'Removing Custom Words: {i + 1} of {len(corpus)}')
        if verbose:
            print('Removing Custom Words: Done!')

    if rmv_accents:
        if verbose:
            print('Removing Accents: Working...')
        for i in range(0, len(corpus)):
            corpus[i] = remove_accents(corpus[i])
        if verbose:
            print('Removing Accents: Done!')

    if rmv_numbers:
        if verbose:
            print('Removing Numbers: Working...')
        corpus = [re.sub('[0-9]', ' ', i) for i in corpus]
        if verbose:
            print('Removing Numbers: Done!')

    for i in range(0, len(corpus)):
        corpus[i] = ' '.join(corpus[i].split())
    
    return corpus
