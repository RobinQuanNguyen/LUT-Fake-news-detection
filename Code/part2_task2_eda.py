import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

DATA_PATH = Path("data/processed_fakenews.csv")
OUTPUT_DIR = Path("outputs/processed_fakenews_second_chunk")

TEXT_COLUMN = "content"
LABEL_COLUMN = "type"
ID_COLUMN = "id"
OPTIONAL_COLUMNS = ["title", "domain", "url", "cleaned_text", "processed_text"]

CHUNK_SIZE = 50_000
SAMPLE_FRACTION = 0.10
MAX_ROWS = None  # example: 300_000
RANDOM_SEED = 42

TOP_WORDS_N = 100
TOP_FREQ_PLOT_N = 10_000
TOP_OUTLIERS_TO_SAVE = 30
SAMPLE_ROWS_TO_SAVE = 1_000
STOPWORD_LANGUAGE = "english"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),
    re.compile(
        r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|"
        r"aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
        r"\s+\d{1,2}(?:,\s*\d{4})?\b",
        re.IGNORECASE,
    ),
]
NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
TOKEN_RE = re.compile(r"[A-Za-z]+")


def print_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        ok = nltk.download(download_name, quiet=False)
        if not ok:
            raise RuntimeError(f"Could not download NLTK resource: {download_name}")


def set_max_csv_field_size_limit() -> None:
    max_int = sys.maxsize
    while max_int > 0:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int //= 10
    raise RuntimeError("Unable to set csv.field_size_limit")


def clean_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def tokenize_alpha_words(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def count_dates(text: str) -> int:
    spans = set()
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            spans.add((match.start(), match.end()))
    return len(spans)


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.min_value: Optional[float] = None
        self.max_value: Optional[float] = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    def to_dict(self) -> Dict[str, Optional[float]]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min_value,
            "max": self.max_value,
        }


class TopKRows:
    def __init__(self, k: int) -> None:
        self.k = k
        self.rows: List[Dict[str, object]] = []

    def add(self, score: float, row: Dict[str, object]) -> None:
        row = dict(row)
        row["score"] = score
        self.rows.append(row)
        self.rows.sort(key=lambda x: x["score"], reverse=True)
        if len(self.rows) > self.k:
            self.rows = self.rows[: self.k]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class Explorer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.stop_words = set(stopwords.words(STOPWORD_LANGUAGE))
        self.stemmer = PorterStemmer()

        self.total_rows_seen = 0
        self.total_rows_analyzed = 0
        self.total_missing_content = 0

        self.missing_counts = Counter()
        self.class_counts = Counter()
        self.unique_hashes = set()
        self.duplicate_content_rows = 0

        self.token_stats = RunningStats()
        self.url_stats = RunningStats()
        self.date_stats = RunningStats()
        self.number_stats = RunningStats()

        self.token_stats_by_class = defaultdict(RunningStats)
        self.url_stats_by_class = defaultdict(RunningStats)
        self.date_stats_by_class = defaultdict(RunningStats)
        self.number_stats_by_class = defaultdict(RunningStats)

        self.token_lengths_all: List[int] = []
        self.token_lengths_by_class = defaultdict(list)

        self.word_counter_raw = Counter()
        self.word_counter_no_stop = Counter()
        self.word_counter_stemmed = Counter()

        self.highest_url_rows = TopKRows(TOP_OUTLIERS_TO_SAVE)
        self.highest_date_rows = TopKRows(TOP_OUTLIERS_TO_SAVE)
        self.highest_number_rows = TopKRows(TOP_OUTLIERS_TO_SAVE)
        self.longest_rows = TopKRows(TOP_OUTLIERS_TO_SAVE)

        self.sample_rows: List[Dict[str, object]] = []

    def process_chunk(self, chunk: pd.DataFrame) -> None:
        self.total_rows_seen += len(chunk)

        for column in chunk.columns:
            self.missing_counts[column] += int(chunk[column].isna().sum())

        for _, row in chunk.iterrows():
            content = clean_text(row.get(TEXT_COLUMN, ""))
            if not content:
                self.total_missing_content += 1
                continue

            label = str(row.get(LABEL_COLUMN, "UNKNOWN"))
            row_id = row.get(ID_COLUMN)
            title = clean_text(row.get("title", "")) if "title" in row.index else ""
            domain = clean_text(row.get("domain", "")) if "domain" in row.index else ""
            url_value = clean_text(row.get("url", "")) if "url" in row.index else ""

            tokens_raw = tokenize_alpha_words(content)
            if not tokens_raw:
                continue

            tokens_no_stop = [t for t in tokens_raw if t not in self.stop_words]
            tokens_stemmed = [self.stemmer.stem(t) for t in tokens_no_stop]

            url_count = len(URL_RE.findall(content))
            date_count = count_dates(content)
            number_count = len(NUMBER_RE.findall(content))
            token_count = len(tokens_raw)
            content_hash = stable_hash(content)

            self.total_rows_analyzed += 1
            self.class_counts[label] += 1

            if content_hash in self.unique_hashes:
                self.duplicate_content_rows += 1
            else:
                self.unique_hashes.add(content_hash)

            self.token_stats.add(token_count)
            self.url_stats.add(url_count)
            self.date_stats.add(date_count)
            self.number_stats.add(number_count)

            self.token_stats_by_class[label].add(token_count)
            self.url_stats_by_class[label].add(url_count)
            self.date_stats_by_class[label].add(date_count)
            self.number_stats_by_class[label].add(number_count)

            self.token_lengths_all.append(token_count)
            self.token_lengths_by_class[label].append(token_count)

            self.word_counter_raw.update(tokens_raw)
            self.word_counter_no_stop.update(tokens_no_stop)
            self.word_counter_stemmed.update(tokens_stemmed)

            preview = content[:250]
            base_row = {
                ID_COLUMN: row_id,
                LABEL_COLUMN: label,
                "title": title,
                "domain": domain,
                "url": url_value,
                "preview": preview,
            }
            self.highest_url_rows.add(url_count, {**base_row, "url_count": url_count})
            self.highest_date_rows.add(date_count, {**base_row, "date_count": date_count})
            self.highest_number_rows.add(number_count, {**base_row, "number_count": number_count})
            self.longest_rows.add(token_count, {**base_row, "token_count": token_count})

            if len(self.sample_rows) < SAMPLE_ROWS_TO_SAVE:
                self.sample_rows.append(
                    {
                        ID_COLUMN: row_id,
                        LABEL_COLUMN: label,
                        "title": title,
                        "domain": domain,
                        "url": url_value,
                        "content": content,
                        "token_count": token_count,
                        "url_count": url_count,
                        "date_count": date_count,
                        "number_count": number_count,
                        "processed_text": " ".join(tokens_stemmed),
                    }
                )

    def save_tables(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "rows_seen": self.total_rows_seen,
            "rows_analyzed": self.total_rows_analyzed,
            "rows_missing_content": self.total_missing_content,
            "duplicate_content_rows": self.duplicate_content_rows,
            "duplicate_content_rate": (
                self.duplicate_content_rows / self.total_rows_analyzed
                if self.total_rows_analyzed else 0.0
            ),
            "unique_contents": len(self.unique_hashes),
            "token_stats": self.token_stats.to_dict(),
            "url_stats": self.url_stats.to_dict(),
            "date_stats": self.date_stats.to_dict(),
            "number_stats": self.number_stats.to_dict(),
        }
        (self.output_dir / "summary_stats.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        pd.DataFrame(
            [{"column": k, "missing_count": v} for k, v in self.missing_counts.items()]
        ).sort_values("missing_count", ascending=False).to_csv(
            self.output_dir / "missing_values.csv", index=False, encoding="utf-8"
        )

        class_df = pd.DataFrame(
            [{"label": k, "count": v} for k, v in self.class_counts.items()]
        ).sort_values("count", ascending=False)
        if not class_df.empty:
            class_df["share"] = class_df["count"] / class_df["count"].sum()
        class_df.to_csv(self.output_dir / "class_distribution.csv", index=False, encoding="utf-8")

        by_class_rows = []
        all_labels = sorted(self.class_counts.keys())
        for label in all_labels:
            by_class_rows.append(
                {
                    "label": label,
                    "count": self.class_counts[label],
                    **{f"token_{k}": v for k, v in self.token_stats_by_class[label].to_dict().items()},
                    **{f"url_{k}": v for k, v in self.url_stats_by_class[label].to_dict().items()},
                    **{f"date_{k}": v for k, v in self.date_stats_by_class[label].to_dict().items()},
                    **{f"number_{k}": v for k, v in self.number_stats_by_class[label].to_dict().items()},
                }
            )
        pd.DataFrame(by_class_rows).to_csv(
            self.output_dir / "feature_summary_by_class.csv", index=False, encoding="utf-8"
        )

        self._save_length_quantiles()
        self._save_top_words()
        self.highest_url_rows.to_frame().to_csv(self.output_dir / "rows_with_most_urls.csv", index=False, encoding="utf-8")
        self.highest_date_rows.to_frame().to_csv(self.output_dir / "rows_with_most_dates.csv", index=False, encoding="utf-8")
        self.highest_number_rows.to_frame().to_csv(self.output_dir / "rows_with_most_numbers.csv", index=False, encoding="utf-8")
        self.longest_rows.to_frame().to_csv(self.output_dir / "longest_articles.csv", index=False, encoding="utf-8")
        pd.DataFrame(self.sample_rows).to_csv(self.output_dir / "analysis_sample_rows.csv", index=False, encoding="utf-8")

    def _save_length_quantiles(self) -> None:
        if not self.token_lengths_all:
            return
        quantile_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        overall = pd.Series(self.token_lengths_all).quantile(quantile_values)
        overall_df = overall.reset_index()
        overall_df.columns = ["quantile", "token_count"]
        overall_df.to_csv(self.output_dir / "token_length_quantiles_overall.csv", index=False, encoding="utf-8")

        per_class_rows = []
        for label, values in self.token_lengths_by_class.items():
            if not values:
                continue
            quantiles = pd.Series(values).quantile(quantile_values)
            for q, val in quantiles.items():
                per_class_rows.append({"label": label, "quantile": q, "token_count": val})
        pd.DataFrame(per_class_rows).to_csv(
            self.output_dir / "token_length_quantiles_by_class.csv", index=False, encoding="utf-8"
        )

    def _save_top_words(self) -> None:
        for name, counter in [
            ("raw", self.word_counter_raw),
            ("no_stop", self.word_counter_no_stop),
            ("stemmed", self.word_counter_stemmed),
        ]:
            rows = [{"word": word, "count": count} for word, count in counter.most_common(TOP_WORDS_N)]
            pd.DataFrame(rows).to_csv(
                self.output_dir / f"top_{TOP_WORDS_N}_words_{name}.csv",
                index=False,
                encoding="utf-8",
            )

    def save_plots(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._plot_class_distribution()
        self._plot_token_length_histogram()
        self._plot_feature_means_by_class()
        self._plot_rank_frequency(self.word_counter_raw, "raw")
        self._plot_rank_frequency(self.word_counter_no_stop, "no_stop")
        self._plot_rank_frequency(self.word_counter_stemmed, "stemmed")

    def _plot_class_distribution(self) -> None:
        if not self.class_counts:
            return
        series = pd.Series(self.class_counts).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        series.plot(kind="bar")
        plt.title("Class distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png", dpi=200)
        plt.close()

    def _plot_token_length_histogram(self) -> None:
        if not self.token_lengths_all:
            return
        plt.figure(figsize=(10, 6))
        plt.hist(self.token_lengths_all, bins=60)
        plt.title("Article length distribution (token count)")
        plt.xlabel("Token count")
        plt.ylabel("Number of articles")
        plt.tight_layout()
        plt.savefig(self.output_dir / "token_length_histogram.png", dpi=200)
        plt.close()

    def _plot_feature_means_by_class(self) -> None:
        if not self.class_counts:
            return

        def build_mean_df(stats_map: Dict[str, RunningStats], filename: str, title: str, ylab: str) -> None:
            rows = []
            for label, stats in stats_map.items():
                d = stats.to_dict()
                rows.append({"label": label, "mean": d["mean"]})
            df = pd.DataFrame(rows).sort_values("mean", ascending=False)
            if df.empty:
                return
            plt.figure(figsize=(10, 6))
            plt.bar(df["label"], df["mean"])
            plt.title(title)
            plt.xlabel("Label")
            plt.ylabel(ylab)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(self.output_dir / filename, dpi=200)
            plt.close()

        build_mean_df(self.url_stats_by_class, "avg_urls_by_class.png", "Average URL count per article by class", "Average URLs")
        build_mean_df(self.date_stats_by_class, "avg_dates_by_class.png", "Average date count per article by class", "Average dates")
        build_mean_df(self.number_stats_by_class, "avg_numbers_by_class.png", "Average numeric count per article by class", "Average numeric values")
        build_mean_df(self.token_stats_by_class, "avg_tokens_by_class.png", "Average token count per article by class", "Average tokens")

    def _plot_rank_frequency(self, counter: Counter, name: str) -> None:
        if not counter:
            return
        top_counts = [count for _, count in counter.most_common(TOP_FREQ_PLOT_N)]
        if not top_counts:
            return
        ranks = list(range(1, len(top_counts) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(ranks, top_counts)
        plt.title(f"Rank-frequency plot of top {min(TOP_FREQ_PLOT_N, len(top_counts)):,} words ({name})")
        plt.xlabel("Rank")
        plt.ylabel("Frequency")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"rank_frequency_top_{TOP_FREQ_PLOT_N}_{name}.png", dpi=200)
        plt.close()

    def save_report_template(self) -> None:
        summary = json.loads((self.output_dir / "summary_stats.json").read_text(encoding="utf-8"))
        class_df = pd.read_csv(self.output_dir / "class_distribution.csv") if (self.output_dir / "class_distribution.csv").exists() else pd.DataFrame()
        by_class_df = pd.read_csv(self.output_dir / "feature_summary_by_class.csv") if (self.output_dir / "feature_summary_by_class.csv").exists() else pd.DataFrame()

        obs = []
        if not class_df.empty and len(class_df) >= 2:
            top = class_df.iloc[0]
            bottom = class_df.iloc[-1]
            ratio = top["count"] / bottom["count"] if bottom["count"] else None
            if ratio and ratio >= 1.2:
                obs.append(
                    f"Class distribution is imbalanced: '{top['label']}' appears {ratio:.2f} times as often as '{bottom['label']}'."
                )

        dup_rate = summary.get("duplicate_content_rate", 0.0)
        if dup_rate >= 0.01:
            obs.append(
                f"The dataset contains duplicate or near-duplicate content rows at a rate of {dup_rate:.2%}, which may indicate artefacts or syndicated reposting."
            )

        if not by_class_df.empty and "url_mean" in by_class_df.columns:
            max_url = by_class_df.sort_values("url_mean", ascending=False).iloc[0]
            min_url = by_class_df.sort_values("url_mean", ascending=True).iloc[0]
            if pd.notna(max_url["url_mean"]) and pd.notna(min_url["url_mean"]) and max_url["url_mean"] > min_url["url_mean"]:
                obs.append(
                    f"Metadata-like content differs by class: '{max_url['label']}' has the highest average URL count per article ({max_url['url_mean']:.2f}), while '{min_url['label']}' has the lowest ({min_url['url_mean']:.2f})."
                )

        if not by_class_df.empty and "token_mean" in by_class_df.columns:
            max_len = by_class_df.sort_values("token_mean", ascending=False).iloc[0]
            min_len = by_class_df.sort_values("token_mean", ascending=True).iloc[0]
            if pd.notna(max_len["token_mean"]) and pd.notna(min_len["token_mean"]) and max_len["token_mean"] > min_len["token_mean"]:
                obs.append(
                    f"Article length differs by class: '{max_len['label']}' has the longest average article length ({max_len['token_mean']:.2f} tokens), whereas '{min_len['label']}' has the shortest ({min_len['token_mean']:.2f} tokens)."
                )

        if not obs:
            obs.append("Use the generated CSV files and plots to write three evidence-based observations from your actual run.")

        report_text = f"""# Task 2 report helper

## Dataset representation
I represented the FakeNewsCorpus as Pandas DataFrame chunks while reading the original CSV file. This design is suitable because the corpus is large, and chunked DataFrames let me inspect columns, compute statistics, and process text without loading the whole dataset into memory.

## What this script reports
- missing values by column
- class distribution
- duplicate-content rate
- article-length statistics and outliers
- URL, date, and numeric counts in the content
- top {TOP_WORDS_N} words in three versions: raw, stopwords removed, and stemmed
- rank-frequency plots for the top {TOP_FREQ_PLOT_N:,} words in those same three versions

## Candidate observations from this run
"""
        for idx, item in enumerate(obs, start=1):
            report_text += f"{idx}. {item}\n"

        report_text += f"""

## Inherent data problems to discuss
- Missing values in some columns
- Duplicate or reposted content
- Boilerplate text, URLs, and metadata leaking into article body
- Strong differences in article length between rows/classes

## Files to reference in your notebook/report
- `summary_stats.json`
- `missing_values.csv`
- `class_distribution.csv`
- `feature_summary_by_class.csv`
- `token_length_quantiles_overall.csv`
- `top_{TOP_WORDS_N}_words_raw.csv`
- `top_{TOP_WORDS_N}_words_no_stop.csv`
- `top_{TOP_WORDS_N}_words_stemmed.csv`
- plot PNG files in the same folder
"""
        (self.output_dir / "task2_report_helper.md").write_text(report_text, encoding="utf-8")


def detect_available_columns(data_path: Path) -> List[str]:
    header_df = pd.read_csv(
        data_path,
        nrows=0,
        encoding="utf-8",
        encoding_errors="ignore",
        engine="python",
    )
    available = list(header_df.columns)

    text_candidates = ["content", "cleaned_text", "processed_text", "text", "article"]
    detected_text = next((c for c in text_candidates if c in available), None)

    if ID_COLUMN not in available:
        raise ValueError(f"Missing required column: {ID_COLUMN}")
    if LABEL_COLUMN not in available:
        raise ValueError(f"Missing required column: {LABEL_COLUMN}")
    if detected_text is None:
        raise ValueError(
            f"No usable text column found. Available columns: {available}"
        )

    global TEXT_COLUMN
    TEXT_COLUMN = detected_text

    usecols = [ID_COLUMN, LABEL_COLUMN, TEXT_COLUMN] + [
        c for c in OPTIONAL_COLUMNS if c in available and c != TEXT_COLUMN
    ]
    return usecols


def iter_chunks(data_path: Path, usecols: List[str]) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(
        data_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        encoding_errors="ignore",
        on_bad_lines="skip",
        engine="python",
    )


def main() -> None:
    print_log("Starting Task 2 exploration")
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    set_max_csv_field_size_limit()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    usecols = detect_available_columns(DATA_PATH)
    print_log(f"Using columns: {usecols}")
    print_log(f"Detected text column: {TEXT_COLUMN}")

    if TEXT_COLUMN != "content":
        print_log(
            f"Warning: using '{TEXT_COLUMN}' instead of raw 'content'. "
            "URL/date/number statistics may not fully reflect the original articles."
        )

    explorer = Explorer(OUTPUT_DIR)

    rows_processed = 0
    chunk_index = 0
    for chunk in iter_chunks(DATA_PATH, usecols):
        chunk_index += 1

        if SAMPLE_FRACTION is not None and 0 < SAMPLE_FRACTION < 1:
            chunk = chunk.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED)

        if MAX_ROWS is not None:
            remaining = MAX_ROWS - rows_processed
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        if chunk.empty:
            continue

        print_log(f"Chunk {chunk_index}: analyzing {len(chunk):,} rows")
        explorer.process_chunk(chunk)
        rows_processed += len(chunk)

        if MAX_ROWS is not None and rows_processed >= MAX_ROWS:
            break

    print_log(f"Finished scanning {explorer.total_rows_seen:,} sampled rows")
    print_log(f"Rows analyzed with non-empty text: {explorer.total_rows_analyzed:,}")

    explorer.save_tables()
    explorer.save_plots()
    explorer.save_report_template()

    print_log(f"Outputs written to: {OUTPUT_DIR.resolve()}")
    print_log("Done")


if __name__ == "__main__":
    main()
