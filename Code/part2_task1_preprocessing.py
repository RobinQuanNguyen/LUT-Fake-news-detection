import csv
from datetime import datetime
from pathlib import Path
import sys
import time
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from collections import Counter

DATA_PATH = Path("data/news_cleaned_2018_02_13.csv")
OUTPUT_PATH = Path("data/processed_fakenews.csv")
FIRST_CHUNK_CSV_PATH = Path("data/processed_fakenews_first_chunk.csv")

CHUNK_SIZE = 50_000
MAX_WORKERS = 8
STOPWORD_LANGUAGE = "english"

MIN_TOKEN_COUNT = 30
MAX_TOKEN_COUNT = 1200
MIN_UNIQUE_RATIO = 0.28
MAX_TOP_TOKEN_RATIO = 0.20

RAW_BOILERPLATE_PATTERNS = [
    "click thumbnail",
    "click picture",
    "click pictur",
    "photo gallery",
    "photo galleri",
    "view gallery",
    "view galleri",
    "follow us on facebook",
    "follow us facebook",
    "follow us twitter",
    "follow us instagram",
    "reader think story fact",
    "reader think stori fact",
    "add two cent",
    "water cooler open thread",
    "subscrib",
    "share this video",
    "watch the video",
]

STEMMED_BOILERPLATE_PATTERNS = [
    "reader think stori fact",
    "add two cent",
    "water cooler open thread",
    "follow us facebook",
    "follow us twitter",
    "follow us instagram",
    "click thumbnail",
    "photo galleri",
    "view galleri",
]

TEXT_COLUMN = "content"
LABEL_COLUMN = "type"
ID_COLUMN = "id"

def print_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def ensure_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        success = nltk.download(download_name, quiet=False)
        if not success:
            raise RuntimeError(f"Unable to download NLTK resource '{download_name}'.")

def set_max_csv_field_size_limit():
    max_int = sys.maxsize
    while max_int > 0:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int //= 10
    raise RuntimeError("Unable to set csv.field_size_limit().")

def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())

TOKEN_RE = re.compile(r"[A-Za-z]+")

def tokenize_alpha_words(text):
    return TOKEN_RE.findall(text.lower())

# Helper functions
def safe_divide(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator

def normalize_for_phrase_match(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())

def has_boilerplate_phrase(text, patterns):
    return any(p in text for p in patterns)

def token_quality_flags(tokens):
    token_count = len(tokens)
    if token_count < MIN_TOKEN_COUNT:
        return True

    if token_count > MAX_TOKEN_COUNT:
        return True

    counts = Counter(tokens)
    unique_ratio = len(counts) / token_count
    top_token_ratio = max(counts.values()) / token_count if counts else 0.0

    if unique_ratio < MIN_UNIQUE_RATIO:
        return True

    if top_token_ratio > MAX_TOP_TOKEN_RATIO:
        return True

    return False

def is_low_quality_text(raw_cleaned_text, filtered_tokens, stemmed_tokens):
    raw_norm = normalize_for_phrase_match(raw_cleaned_text)
    stemmed_text = " ".join(stemmed_tokens)

    if has_boilerplate_phrase(raw_norm, RAW_BOILERPLATE_PATTERNS):
        return True

    if token_quality_flags(filtered_tokens):
        return True

    if has_boilerplate_phrase(stemmed_text, STEMMED_BOILERPLATE_PATTERNS):
        return True

    return False

WORKER_STOP_WORDS = set()

def init_worker(stop_words):
    global WORKER_STOP_WORDS
    WORKER_STOP_WORDS = set(stop_words)

def split_rows_into_batches(rows, n_batches):
    if not rows:
        return []

    batch_size = max(1, len(rows) // n_batches)
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)] 

    return batches

def process_row_batch(rows):
    stemmer = PorterStemmer()

    output_ids = []
    output_labels = []
    output_texts = []

    vocab_before = set()
    vocab_after_stop = set()
    vocab_after_stem = set()

    tokens_before = 0
    tokens_after_stop = 0
    tokens_after_stem = 0

    for row_id, label, raw_text in rows:
        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            continue

        tokens = tokenize_alpha_words(cleaned_text)
        if not tokens:
            continue

        filtered = [t for t in tokens if t not in WORKER_STOP_WORDS]
        if not filtered:
            continue

        stemmed = [stemmer.stem(t) for t in filtered]
        if is_low_quality_text(cleaned_text, filtered, stemmed):
            continue

        joined_stemmed = " ".join(stemmed)

        vocab_before.update(tokens)
        tokens_before += len(tokens)

        vocab_after_stop.update(filtered)
        tokens_after_stop += len(filtered)

        vocab_after_stem.update(stemmed)
        tokens_after_stem += len(stemmed)

        output_ids.append(row_id)
        output_labels.append(label)
        output_texts.append(joined_stemmed)

    return {
        "output_ids": output_ids,
        "output_labels": output_labels,
        "output_texts": output_texts,
        "vocab_before": vocab_before,
        "vocab_after_stop": vocab_after_stop,
        "vocab_after_stem": vocab_after_stem,
        "tokens_before": tokens_before,
        "tokens_after_stop": tokens_after_stop,
        "tokens_after_stem": tokens_after_stem,
    }

def process_chunk(chunk, executor, max_workers=MAX_WORKERS):
    chunk = chunk.dropna(subset=[TEXT_COLUMN])

    if chunk.empty:
        return {
            "output_df": None,
            "vocab_before": set(),
            "vocab_after_stop": set(),
            "vocab_after_stem": set(),
            "tokens_before": 0,
            "tokens_after_stop": 0,
            "tokens_after_stem": 0,
            "rows_in_chunk": 0,
        }

    rows = list(
        zip(
            chunk[ID_COLUMN].values.tolist(),
            chunk[LABEL_COLUMN].values.tolist(),
            chunk[TEXT_COLUMN].values.tolist(),
        )
    )

    task_count = min(max_workers, len(rows))

    batches = split_rows_into_batches(rows, task_count) 

    output_ids = []
    output_labels = []
    output_texts = []

    vocab_before = set()
    vocab_after_stop = set()
    vocab_after_stem = set()

    tokens_before = 0
    tokens_after_stop = 0
    tokens_after_stem = 0

    for result in executor.map(process_row_batch, batches):
        output_ids.extend(result["output_ids"])
        output_labels.extend(result["output_labels"])
        output_texts.extend(result["output_texts"])

        vocab_before.update(result["vocab_before"])
        vocab_after_stop.update(result["vocab_after_stop"])
        vocab_after_stem.update(result["vocab_after_stem"])

        tokens_before += result["tokens_before"]
        tokens_after_stop += result["tokens_after_stop"]
        tokens_after_stem += result["tokens_after_stem"]

    output_df = None
    if output_ids:
        output_df = pd.DataFrame({
            ID_COLUMN: output_ids,
            LABEL_COLUMN: output_labels,
            "processed_text": output_texts,
        })

    return {
        "output_df": output_df,
        "vocab_before": vocab_before,
        "vocab_after_stop": vocab_after_stop,
        "vocab_after_stem": vocab_after_stem,
        "tokens_before": tokens_before,
        "tokens_after_stop": tokens_after_stop,
        "tokens_after_stem": tokens_after_stem,
        "rows_in_chunk": len(chunk),
    }

def main():
    print_log("Starting pipeline")
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    set_max_csv_field_size_limit()

    stop_words = set(stopwords.words(STOPWORD_LANGUAGE))

    vocab_before = set()
    vocab_after_stop = set()
    vocab_after_stem = set()

    total_rows_seen = 0
    total_rows_written = 0
    total_tokens_before = 0
    total_tokens_after_stop = 0
    total_tokens_after_stem = 0

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    csv_reader = pd.read_csv(
        DATA_PATH,
        usecols=[ID_COLUMN, LABEL_COLUMN, TEXT_COLUMN],
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        encoding_errors="ignore",
        on_bad_lines="skip",
        engine="python",
    )

    output_csv_initialized = False

    worker_count = min(
        MAX_WORKERS,
        max(1, multiprocessing.cpu_count() - 1),
    )

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_worker,
        initargs=(stop_words,),
    ) as executor:
        for chunk_number, chunk in enumerate(csv_reader, start=1):
            t0 = time.perf_counter()
            print_log(f"Chunk {chunk_number}: read {len(chunk):,} rows")

            result = process_chunk(chunk, executor, max_workers=worker_count)

            vocab_before.update(result["vocab_before"])
            vocab_after_stop.update(result["vocab_after_stop"])
            vocab_after_stem.update(result["vocab_after_stem"])

            total_rows_seen += result["rows_in_chunk"]
            total_tokens_before += result["tokens_before"]
            total_tokens_after_stop += result["tokens_after_stop"]
            total_tokens_after_stem += result["tokens_after_stem"]

            output_df = result["output_df"]
            if output_df is not None and not output_df.empty:
                if chunk_number == 1:
                    output_df.to_csv(
                        FIRST_CHUNK_CSV_PATH,
                        index=False,
                        encoding="utf-8",
                    )

                output_df.to_csv(
                    OUTPUT_PATH,
                    mode="a",
                    header=not output_csv_initialized,
                    index=False,
                    encoding="utf-8",
                )
                output_csv_initialized = True
                total_rows_written += len(output_df)

            dt = time.perf_counter() - t0
            print_log(
                f"Chunk {chunk_number}: done in {dt:.1f}s | "
                f"rows_written={total_rows_written:,} | "
                f"vocab_before={len(vocab_before):,} | "
                f"vocab_after_stop={len(vocab_after_stop):,} | "
                f"vocab_after_stem={len(vocab_after_stem):,}"
            )

    stopword_reduction_rate = safe_divide(
        len(vocab_before) - len(vocab_after_stop),
        len(vocab_before),
    ) * 100

    stemming_reduction_rate = safe_divide(
        len(vocab_after_stop) - len(vocab_after_stem),
        len(vocab_after_stop),
    ) * 100

    print_log(f"Rows seen: {total_rows_seen:,}")
    print_log(f"Rows written: {total_rows_written:,}")
    print_log(f"Vocabulary before stopwords: {len(vocab_before):,}")
    print_log(f"Vocabulary after stopwords: {len(vocab_after_stop):,}")
    print_log(f"Stopword reduction rate: {stopword_reduction_rate:.2f}%")
    print_log(f"Vocabulary after stemming: {len(vocab_after_stem):,}")
    print_log(f"Stemming reduction rate: {stemming_reduction_rate:.2f}%")

if __name__ == "__main__":
    main()
