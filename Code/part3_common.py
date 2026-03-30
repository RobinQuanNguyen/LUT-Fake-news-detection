from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

LABEL_COLUMN = "type"
ID_COLUMN = "id"
TEXT_CANDIDATES = ["processed_text", "cleaned_text", "content", "text", "article"]
OPTIONAL_METADATA_COLUMNS = ["title", "domain", "url"]

RELIABLE_LABELS = {"reliable", "real", "true"}
FAKE_LABELS = {
    "fake",
    "false",
    "unreliable",
    "clickbait",
    "conspiracy",
    "junksci",
    "hate",
    "rumor",
    "propaganda",
    "fabricated",
}
DROP_LABELS = {
    "bias",
    "political",
    "satire",
    "state",
    "unknown",
    "mixed",
    "partisan",
}


@dataclass
class SplitBundle:
    train: pd.DataFrame
    validate: pd.DataFrame
    test: pd.DataFrame


@dataclass
class LabelMappingResult:
    frame: pd.DataFrame
    kept_counts: Dict[str, int]
    dropped_counts: Dict[str, int]


def print_log(message: str) -> None:
    print(message, flush=True)


def normalize_label(label: object) -> str:
    if pd.isna(label):
        return ""
    return str(label).strip().lower()


def map_binary_label(raw_label: object) -> Optional[int]:
    label = normalize_label(raw_label)
    if not label:
        return None

    if label in RELIABLE_LABELS:
        return 0
    if label in FAKE_LABELS:
        return 1
    if label in DROP_LABELS:
        return None

    reliable_keywords = ["reliable", "credible", "trusted"]
    fake_keywords = [
        "fake",
        "false",
        "unreliable",
        "clickbait",
        "conspiracy",
        "junksci",
        "hate",
        "propaganda",
        "fabricated",
        "rumor",
    ]
    dropped_keywords = ["satire", "political", "bias", "state", "mixed", "unknown"]

    if any(keyword in label for keyword in reliable_keywords):
        return 0
    if any(keyword in label for keyword in fake_keywords):
        return 1
    if any(keyword in label for keyword in dropped_keywords):
        return None
    return None


def read_header(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return list(pd.read_csv(path, nrows=0).columns)


def detect_shared_text_column(paths: Sequence[Path]) -> str:
    header_sets = [set(read_header(path)) for path in paths]
    for candidate in TEXT_CANDIDATES:
        if all(candidate in header for header in header_sets):
            return candidate
    raise ValueError(
        f"No shared text column found across splits. Expected one of: {TEXT_CANDIDATES}"
    )


def detect_shared_metadata_columns(paths: Sequence[Path]) -> List[str]:
    header_sets = [set(read_header(path)) for path in paths]
    return [column for column in OPTIONAL_METADATA_COLUMNS if all(column in header for header in header_sets)]


def read_split(path: Path, text_column: str, metadata_columns: Sequence[str]) -> pd.DataFrame:
    preview = pd.read_csv(path, nrows=0)
    available = set(preview.columns)
    required = {ID_COLUMN, LABEL_COLUMN, text_column}
    missing = required - available
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    usecols = [ID_COLUMN, LABEL_COLUMN, text_column] + [column for column in metadata_columns if column in available]
    return pd.read_csv(path, usecols=usecols)


def clean_split_frame(frame: pd.DataFrame, text_column: str, metadata_columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    result[text_column] = result[text_column].fillna("").astype(str).str.strip()
    result[LABEL_COLUMN] = result[LABEL_COLUMN].fillna("").astype(str).str.strip()
    result = result[result[text_column] != ""].copy()

    for column in metadata_columns:
        if column in result.columns:
            result[column] = result[column].fillna("").astype(str)

    return result.reset_index(drop=True)


def apply_label_mapping(frame: pd.DataFrame) -> LabelMappingResult:
    result = frame.copy()
    result["binary_label"] = result[LABEL_COLUMN].map(map_binary_label)

    kept = result[result["binary_label"].notna()].copy()
    kept["binary_label"] = kept["binary_label"].astype(int)

    dropped = result[result["binary_label"].isna()].copy()

    kept_counts = {
        str(k): int(v) for k, v in kept[LABEL_COLUMN].value_counts(dropna=False).sort_index().items()
    }
    dropped_counts = {
        str(k): int(v) for k, v in dropped[LABEL_COLUMN].value_counts(dropna=False).sort_index().items()
    }

    return LabelMappingResult(frame=kept.reset_index(drop=True), kept_counts=kept_counts, dropped_counts=dropped_counts)


def load_splits(
    train_path: Path,
    validate_path: Path,
    test_path: Path,
    text_column: str,
    metadata_columns: Sequence[str],
) -> SplitBundle:
    train = clean_split_frame(read_split(train_path, text_column, metadata_columns), text_column, metadata_columns)
    validate = clean_split_frame(read_split(validate_path, text_column, metadata_columns), text_column, metadata_columns)
    test = clean_split_frame(read_split(test_path, text_column, metadata_columns), text_column, metadata_columns)
    return SplitBundle(train=train, validate=validate, test=test)


def map_splits_to_binary(splits: SplitBundle) -> Tuple[SplitBundle, Dict[str, Dict[str, Dict[str, int] | int]]]:
    reports: Dict[str, Dict[str, Dict[str, int] | int]] = {}
    mapped_frames = {}

    for split_name, frame in {
        "train": splits.train,
        "validate": splits.validate,
        "test": splits.test,
    }.items():
        mapping_result = apply_label_mapping(frame)
        mapped_frames[split_name] = mapping_result.frame
        reports[split_name] = {
            "kept_counts": mapping_result.kept_counts,
            "dropped_counts": mapping_result.dropped_counts,
            "rows_before": int(len(frame)),
            "rows_after": int(len(mapping_result.frame)),
        }

    return (
        SplitBundle(
            train=mapped_frames["train"],
            validate=mapped_frames["validate"],
            test=mapped_frames["test"],
        ),
        reports,
    )


def available_metadata_columns(splits: SplitBundle, candidates: Optional[Sequence[str]] = None) -> List[str]:
    candidate_columns = list(candidates) if candidates is not None else list(OPTIONAL_METADATA_COLUMNS)
    available: List[str] = []

    for column in candidate_columns:
        present_in_all = all(column in frame.columns for frame in [splits.train, splits.validate, splits.test])
        if not present_in_all:
            continue
        combined = pd.concat(
            [splits.train[column], splits.validate[column], splits.test[column]],
            ignore_index=True,
        )
        non_empty = combined.fillna("").astype(str).str.strip() != ""
        if non_empty.any():
            available.append(column)

    return available


def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_matrix_as_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def describe_binary_distribution(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame["binary_label"].value_counts().to_dict()
    return {
        "reliable_0": int(counts.get(0, 0)),
        "fake_1": int(counts.get(1, 0)),
        "total": int(len(frame)),
    }