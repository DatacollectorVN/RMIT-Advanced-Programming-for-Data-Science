import os
import re
import pandas as pd
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import re
from spellchecker import SpellChecker
from rapidfuzz import fuzz
from wordfreq import word_frequency
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from tqdm.auto import tqdm


TOKENIZER_PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?") # From documentation
BASE_PATH = "./data"
STOPWORDS_PATH = f"{BASE_PATH}/stopwords_en.txt"
ENHANCED_PATH = f"{BASE_PATH}/stopwords_en_enhancement.txt" 
VOCAB_PATH = f"{BASE_PATH}/vocab.txt"
OUTPUT_PATH = f"{BASE_PATH}/processed.csv"
DATA_PATH = f"/Users/nhan.ngo/rmit/RMIT-Advanced-Programming-for-Data-Science/assigments/group/data/cosmetics_beauty_products_reviews.csv"


def load_stopwords(path: str) -> list[str]:
    """Load stopwords from a text file.

    Args:
        path: Path to a stopword file with one word per line.

    Returns:
        List of lowercase stopwords with empty lines removed.
    """
    # Read each non-empty line, strip whitespace and lowercase for consistent lookup
    with open(path, "r", encoding="utf-8") as f:
        words: list[str] = [line.strip().lower() for line in f if line.strip()]
    return words


def save_vocab(processed_docs: list[list[str]], path: str) -> dict:
    """
    Build a vocabulary from a list of tokenized documents.

    Args:
        processed_docs: Tokenized documents after preprocessing.
        path: Output file path for vocabulary entries.

    Returns:
        Dict-like vocabulary mapping represented in the saved file as word:index.
    """
    # Flatten all token lists into a set to deduplicate across all documents
    vocab_set = {t for doc in processed_docs for t in doc}
    # Sort alphabetically so the index assignment is deterministic across runs
    vocab_sorted = sorted(vocab_set)  # A–Z (ASCII-ish; digits would sort before letters if any)
    lines = []
    for i, word in enumerate(vocab_sorted):
        # Format: "word:index" — one entry per line
        lines.append(f"{word}:{i}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Saved] {len(lines)} words → {path}")


def _split_tokens(text: str) -> list[str]:
    """Split raw text into alphabetic tokens.

    Args:
        text: Input text to tokenize.

    Returns:
        List of regex-matched tokens.
    """
    # Apply the compiled regex to extract alphabetic tokens (allows internal hyphens/apostrophes)
    return TOKENIZER_PATTERN.findall(str(text))


def _process_token(t: str, stop_set: set[str]) -> str | None:
    """Apply basic token filtering rules.

    Args:
        t: Candidate token string.
        stop_set: Set of stopwords for fast lookup.

    Returns:
        Normalized token if valid, otherwise None.
    """
    # Reject None / empty strings produced by upstream steps
    if t:
        # Single-character tokens carry no meaning — discard them
        if len(t) >= 2:
            # Keep the token only if it is not a stopword
            if t not in stop_set:
                return t
            else:
                return None
        else:
            return None
    else:
        return None
    

def _is_meaningful(word: str) -> bool:
    """Heuristic check for short-token quality.

    Args:
        word: Candidate token.

    Returns:
        True when token should be kept, else False.
    """
    # 2-char words must be genuinely common (by/as/in pass, ab/ac fail)
    if len(word) == 2 and word_frequency(word, 'en') < 1e-4:
        return False
    return True


def check_and_fix(lemmatizer: WordNetLemmatizer, speller: SpellChecker, word: str) -> str | None:
    """Spell-correct and lemmatize a token when possible.

    Args:
        lemmatizer: NLTK lemmatizer instance.
        speller: SpellChecker instance for candidate correction.
        word: Token to validate and normalize.

    Returns:
        Cleaned lemma token, or None when token should be discarded.
    """
    # Nothing to process for empty/None tokens
    if not word:
        return None

    # Collapse repeated characters: "goood" → "god" (hard) / "good" (light)
    # c_light: only 3+ repeats collapsed to 1;  c_hard: any repeat collapsed to 1
    c_light = re.sub(r'(.)\1{2,}', r'\1', word)
    c_hard  = re.sub(r'(.)\1+',    r'\1', word)

    if speller.known([word]):
        # Word is in the dictionary — drop it if it is too rare / meaningless
        if not _is_meaningful(word):
            return None
        # Accept the spell-checker's suggestion only when it is close enough (≥70 similarity)
        correction = speller.correction(word)
        if correction and _is_meaningful(correction) and fuzz.ratio(word, correction) >= 70:
            return lemmatizer.lemmatize(correction, pos=wordnet.NOUN)
        # Word is already correct — just lemmatize it
        return lemmatizer.lemmatize(word, pos=wordnet.NOUN)

    # Word is unknown — try progressively de-duplicated forms (hard → light → original)
    # dict.fromkeys preserves order and removes duplicates in case forms coincide
    for candidate in dict.fromkeys([c_hard, c_light, word]):
        if not candidate or not _is_meaningful(candidate):
            continue
        correction = speller.correction(candidate)
        if not correction or not _is_meaningful(correction):
            continue
        # Accept correction only when it is visually similar to the candidate
        if fuzz.ratio(candidate, correction) >= 70:
            return lemmatizer.lemmatize(correction, pos=wordnet.NOUN)

    # No valid form found — discard the token
    return None


def _filter_by_frequency(docs: list[list[str]], top_k_df: int = 20, min_tf: int = 2) -> list[list[str]]:
    """Remove low-value tokens using corpus frequencies.

    Args:
        docs: List of tokenized documents.
        top_k_df: Number of most common document-frequency terms to drop.
        min_tf: Minimum corpus term frequency threshold.

    Returns:
        Filtered tokenized documents.
    """
    # Count global term frequency (TF) and document frequency (DF) across the corpus
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for doc in docs:
        tf.update(doc)
        df.update(set(doc))  # set() ensures each word counted once per document

    # Remove tokens that appear fewer than min_tf times in the whole corpus
    docs = [[t for t in doc if tf[t] >= min_tf] for doc in docs]

    # Remove the top_k_df most frequent words — they appear in nearly every doc and carry little signal
    top_df_words: set[str] = {w for w, _ in df.most_common(top_k_df)}
    docs = [
        # If filtering removes all tokens, keep the original to avoid empty documents
        filtered if (filtered := [t for t in doc if t not in top_df_words]) else doc
        for doc in docs
    ]
    return docs


def _setup():
    """Prepare required resources and output directories.

    Returns:
        None.
    """
    # Download required NLTK data files if not already present (wordnet for lemmatization)
    print("Installing nltk packages...")
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # Ensure the output directory exists before any file writes
    print(f"Creating directory: {BASE_PATH}")
    os.makedirs(os.path.dirname(BASE_PATH), exist_ok=True)
    

def _preprocess_chunk(
    chunk: list[str],
    stop_set: set[str],
    worker_id: int,
) -> list[list[str]]:
    """Preprocess one text chunk inside a worker thread.

    Args:
        chunk: Subset of raw review texts.
        stop_set: Stopword set shared across workers.
        worker_id: Worker index for progress display.

    Returns:
        Tokenized documents for this chunk.
    """
    # Each worker owns its own lemmatizer and spell-checker — not thread-safe to share
    lemmatizer = WordNetLemmatizer()
    speller = SpellChecker()
    docs = []
    for text in tqdm(chunk, desc=f"Core {worker_id + 1}", position=worker_id, leave=True):
        tokens = []
        for token in _split_tokens(text):
            t = token.lower()
            # First pass: discard single-char tokens and stopwords
            t = _process_token(t, stop_set)
            # Spell-check, de-duplicate characters, and lemmatize
            t = check_and_fix(lemmatizer, speller, t)
            # Second pass: lemmatization may produce a stopword (e.g. "be") — filter again
            t = _process_token(t, stop_set)
            if t:
                tokens.append(t)
        docs.append(tokens)
    return docs
    

def preprocess(
    texts: list[str],
    stopwords: list[str],
    top_k_df: int = 20,
    min_tf: int = 2,
    n_jobs: int = -1,  # -1 = all available cores
) -> list[list[str]]:
    """Run full text preprocessing with optional multithreading.

    Args:
        texts: Raw text documents.
        stopwords: Stopword list for token filtering.
        top_k_df: Number of highest document-frequency tokens to remove.
        min_tf: Minimum token frequency threshold across corpus.
        n_jobs: Worker count (-1 uses all CPU cores).

    Returns:
        Preprocessed tokenized documents.
    """
    # Build a set for O(1) stopword lookup; normalise case/whitespace upfront
    stop_set: set = set(w.lower().strip() for w in stopwords if w.strip())

    # Resolve worker count: cap at available CPUs; always at least 1
    n_workers: int = os.cpu_count() if n_jobs == -1 else min(n_jobs, os.cpu_count())
    n_workers = max(1, n_workers)

    # Split texts into equal-sized chunks — one chunk per worker thread
    chunk_size: int = ceil(len(texts) / n_workers)
    chunks: list[list[str]] = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    actual_workers: int = len(chunks)  # may be < n_workers for tiny inputs

    # Pre-allocate result list so we can insert by index regardless of completion order
    ordered: list[list[list[str]]] = [None] * actual_workers
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures: dict = {
            executor.submit(_preprocess_chunk, chunk, stop_set, i): i
            for i, chunk in enumerate(chunks)
        }
        # Collect results as they finish; use the stored index to preserve original order
        for future in as_completed(futures):
            idx = futures[future]
            ordered[idx] = future.result()

    # Flatten the list-of-chunks back into a flat list of documents
    docs: list[list[str]] = [doc for chunk_docs in ordered for doc in chunk_docs]

    # Apply corpus-level frequency filtering once across all workers' output.
    return _filter_by_frequency(docs, top_k_df=top_k_df, min_tf=min_tf)


def main():
    """Run the end-to-end task pipeline.

    Returns:
        None.
    """
    print("Setting up...")
    _setup()

    print("Loading data...")
    df_raw: pd.DataFrame = pd.read_csv(DATA_PATH)
    df_raw["review_text"] = df_raw["review_text"].fillna("")
    print(df_raw.head(2))
    print(f"Shape: {df_raw.shape}")

    print("Preprocessing data...")
    stopwords = load_stopwords(ENHANCED_PATH)  # or STOPWORDS_PATH
    reviews = df_raw["review_text"].tolist()
    processed_docs = preprocess(
        reviews,
        stopwords,
        min_tf=1,   # no tf filtering on tiny corpus
        n_jobs=5,
    )

    print("Saving vocabulary...")
    save_vocab(processed_docs, VOCAB_PATH)


if __name__ == "__main__":
    main()




