from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from part3_common import (
    ID_COLUMN,
    LABEL_COLUMN,
    SplitBundle,
    available_metadata_columns,
    confusion_matrix_as_dict,
    describe_binary_distribution,
    detect_shared_metadata_columns,
    detect_shared_text_column,
    evaluate_predictions,
    load_splits,
    map_splits_to_binary,
    print_log,
    save_json,
    save_text,
)

TRAIN_PATH = Path("data/train.csv")
VALIDATE_PATH = Path("data/validate.csv")
TEST_PATH = Path("data/test.csv")
OUTPUT_DIR = Path("outputs/part3_simple_model")

RUN_METADATA_EXPERIMENT = False
TRAIN_SELECTION_SAMPLE = 80000
VALID_SELECTION_SAMPLE = 20000
USE_HASHING_FALLBACK = True
HASHING_FEATURES = 2**18

TEXT_MODEL_CANDIDATES = [
    {
        "model_name": "multinomial_nb",
        "ngram_range": (1, 1),
        "min_df": 30,
        "max_features": 15000,
        "alpha": 1.0,
    },
    {
        "model_name": "multinomial_nb",
        "ngram_range": (1, 1),
        "min_df": 50,
        "max_features": 25000,
        "alpha": 0.5,
    },
    {
        "model_name": "sgd_logistic",
        "ngram_range": (1, 1),
        "min_df": 30,
        "max_features": 20000,
        "alpha": 1e-5,
    },
    {
        "model_name": "sgd_logistic",
        "ngram_range": (1, 1),
        "min_df": 50,
        "max_features": 30000,
        "alpha": 5e-6,
    },
]

METADATA_MODEL_CANDIDATES = [
    {
        "model_name": "metadata_sgd_logistic",
        "text_ngram_range": (1, 1),
        "text_max_features": 10000,
        "text_min_df": 10,
        "title_max_features": 3000,
        "title_min_df": 5,
        "alpha": 1e-5,
    }
]


def stratified_sample_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()

    label_counts = frame["binary_label"].value_counts()
    if label_counts.min() < 2:
        return frame.sample(n=max_rows, random_state=42)

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_rows, random_state=42)
    y = frame["binary_label"]
    indices, _ = next(splitter.split(frame, y))
    return frame.iloc[indices].reset_index(drop=True)


def make_text_pipeline(config: Dict) -> Pipeline:
    effective_min_df = max(1, int(config.get("effective_min_df", config["min_df"])))

    if USE_HASHING_FALLBACK and config["model_name"] == "sgd_logistic":
        vectorizer = HashingVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 1),
            n_features=HASHING_FEATURES,
            alternate_sign=False,
            norm=None,
        )
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(config["alpha"]),
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )
        return Pipeline([
            ("hashing", vectorizer),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("clf", classifier),
        ])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=config["ngram_range"],
        min_df=effective_min_df,
        max_df=0.95,
        max_features=int(config["max_features"]),
        sublinear_tf=True,
        dtype=np.float32,
    )

    if config["model_name"] == "multinomial_nb":
        classifier = MultinomialNB(alpha=float(config["alpha"]))
    elif config["model_name"] == "sgd_logistic":
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(config["alpha"]),
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported text model: {config['model_name']}")

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def add_numeric_metadata_features(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    result = frame.copy()

    if "title" in result.columns:
        title_series = result["title"].fillna("").astype(str)
        result["title"] = title_series
        result["title_length"] = title_series.str.len().astype(np.float32)
        result["title_word_count"] = title_series.str.split().str.len().fillna(0).astype(np.float32)
    else:
        result["title"] = ""
        result["title_length"] = 0.0
        result["title_word_count"] = 0.0

    if "url" in result.columns:
        url_series = result["url"].fillna("").astype(str)
        result["url"] = url_series
        result["url_length"] = url_series.str.len().astype(np.float32)
        result["has_https"] = url_series.str.startswith("https").astype(np.float32)
        result["url_slash_count"] = url_series.str.count("/").astype(np.float32)
    else:
        result["url"] = ""
        result["url_length"] = 0.0
        result["has_https"] = 0.0
        result["url_slash_count"] = 0.0

    if "domain" in result.columns:
        result["domain"] = result["domain"].fillna("").astype(str)
    else:
        result["domain"] = ""

    text_series = result[text_column].fillna("").astype(str)
    result[text_column] = text_series
    result["text_length"] = text_series.str.len().astype(np.float32)
    result["text_word_count"] = text_series.str.split().str.len().fillna(0).astype(np.float32)

    return result


def make_metadata_pipeline(config: Dict, text_column: str, metadata_columns: List[str]) -> Pipeline:
    numeric_columns = [
        "title_length",
        "title_word_count",
        "url_length",
        "has_https",
        "url_slash_count",
        "text_length",
        "text_word_count",
    ]

    transformers = [
        (
            "text_tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 1),
                min_df=max(1, int(config["text_min_df"])),
                max_df=0.95,
                max_features=min(int(config["text_max_features"]), 10000),
                sublinear_tf=True,
                dtype=np.float32,
            ),
            text_column,
        ),
        ("numeric", StandardScaler(with_mean=False), numeric_columns),
    ]

    if "title" in metadata_columns:
        transformers.append(
            (
                "title_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 1),
                    min_df=max(1, int(config["title_min_df"])),
                    max_df=0.95,
                    max_features=min(int(config["title_max_features"]), 3000),
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
                "title",
            )
        )

    if "domain" in metadata_columns:
        transformers.append(
            (
                "domain_ohe",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                ["domain"],
            )
        )

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(config["alpha"]),
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )

    return Pipeline([
        (
            "metadata_builder",
            FunctionTransformer(lambda df: add_numeric_metadata_features(df, text_column), validate=False),
        ),
        ("features", ColumnTransformer(transformers=transformers, sparse_threshold=0.3)),
        ("clf", classifier),
    ])


def plot_confusion(y_true: List[int], y_pred: List[int], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["reliable", "fake"], ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def fit_text_candidate(config: Dict, train_df: pd.DataFrame, validate_df: pd.DataFrame, text_column: str) -> Tuple[Pipeline, Dict]:
    effective_config = dict(config)
    effective_config["effective_min_df"] = min(
        int(config["min_df"]),
        max(1, len(train_df) // 20),
        len(train_df),
    )

    pipeline = make_text_pipeline(effective_config)
    try:
        pipeline.fit(train_df[text_column], train_df["binary_label"])
    except ValueError:
        effective_config["effective_min_df"] = 1
        pipeline = make_text_pipeline(effective_config)
        pipeline.fit(train_df[text_column], train_df["binary_label"])

    y_pred = pipeline.predict(validate_df[text_column])
    metrics = evaluate_predictions(validate_df["binary_label"], y_pred)
    return pipeline, {**effective_config, **metrics}


def fit_metadata_candidate(
    config: Dict,
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    text_column: str,
    metadata_columns: List[str],
) -> Tuple[Pipeline, Dict]:
    effective_config = dict(config)
    effective_config["text_min_df"] = min(
        int(config["text_min_df"]),
        max(1, len(train_df) // 20),
        len(train_df),
    )
    effective_config["title_min_df"] = min(
        int(config["title_min_df"]),
        max(1, len(train_df) // 20),
        len(train_df),
    )

    pipeline = make_metadata_pipeline(effective_config, text_column, metadata_columns)
    try:
        pipeline.fit(train_df, train_df["binary_label"])
    except ValueError:
        effective_config["text_min_df"] = 1
        effective_config["title_min_df"] = 1
        pipeline = make_metadata_pipeline(effective_config, text_column, metadata_columns)
        pipeline.fit(train_df, train_df["binary_label"])

    y_pred = pipeline.predict(validate_df)
    metrics = evaluate_predictions(validate_df["binary_label"], y_pred)
    return pipeline, {**effective_config, **metrics}


def select_best_text_model(train_df: pd.DataFrame, validate_df: pd.DataFrame, text_column: str) -> Tuple[Pipeline, Dict, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    best_pipeline: Optional[Pipeline] = None
    best_config: Optional[Dict] = None
    best_key = (-1.0, -1.0, -1.0)

    train_sample = stratified_sample_frame(train_df, TRAIN_SELECTION_SAMPLE)
    validate_sample = stratified_sample_frame(validate_df, VALID_SELECTION_SAMPLE)

    print_log("Selecting the best text-only baseline on a stratified validation sample")
    print_log(f"Selection train sample size: {len(train_sample):,}")
    print_log(f"Selection validate sample size: {len(validate_sample):,}")

    for config in TEXT_MODEL_CANDIDATES:
        pipeline, result = fit_text_candidate(config, train_sample, validate_sample, text_column)
        row = {
            **result,
            "selection_train_rows": int(len(train_sample)),
            "selection_validate_rows": int(len(validate_sample)),
        }
        rows.append(row)

        print_log(
            f"{result['model_name']} | ngram={result['ngram_range']} | "
            f"max_features={result.get('max_features', HASHING_FEATURES)} | "
            f"F1={result['f1']:.4f} | Acc={result['accuracy']:.4f} | "
            f"Prec={result['precision']:.4f} | Rec={result['recall']:.4f}"
        )

        current_key = (result["f1"], result["accuracy"], result["recall"])
        if current_key > best_key:
            best_key = current_key
            best_pipeline = pipeline
            best_config = dict(result)

    if best_pipeline is None or best_config is None:
        raise RuntimeError("Failed to select a text-only model")

    results = pd.DataFrame(rows).sort_values(by=["f1", "accuracy", "recall"], ascending=False).reset_index(drop=True)
    return best_pipeline, best_config, results


def select_best_metadata_model(
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    text_column: str,
    metadata_columns: List[str],
) -> Tuple[Optional[Pipeline], Optional[Dict], Optional[pd.DataFrame]]:
    if not metadata_columns:
        return None, None, None

    rows: List[Dict[str, object]] = []
    best_pipeline: Optional[Pipeline] = None
    best_config: Optional[Dict] = None
    best_key = (-1.0, -1.0, -1.0)

    train_sample = stratified_sample_frame(train_df, min(TRAIN_SELECTION_SAMPLE, 80000))
    validate_sample = stratified_sample_frame(validate_df, min(VALID_SELECTION_SAMPLE, 20000))

    print_log("Selecting the best text + metadata baseline on a stratified validation sample")
    print_log(f"Detected metadata columns: {metadata_columns}")

    for config in METADATA_MODEL_CANDIDATES:
        pipeline, result = fit_metadata_candidate(config, train_sample, validate_sample, text_column, metadata_columns)
        row = {
            **result,
            "metadata_columns": ",".join(metadata_columns),
            "selection_train_rows": int(len(train_sample)),
            "selection_validate_rows": int(len(validate_sample)),
        }
        rows.append(row)

        print_log(
            f"{result['model_name']} | text_max_features={result['text_max_features']} | "
            f"F1={result['f1']:.4f} | Acc={result['accuracy']:.4f} | "
            f"Prec={result['precision']:.4f} | Rec={result['recall']:.4f}"
        )

        current_key = (result["f1"], result["accuracy"], result["recall"])
        if current_key > best_key:
            best_key = current_key
            best_pipeline = pipeline
            best_config = dict(result)

    if best_pipeline is None or best_config is None:
        return None, None, None

    results = pd.DataFrame(rows).sort_values(by=["f1", "accuracy", "recall"], ascending=False).reset_index(drop=True)
    return best_pipeline, best_config, results


def refit_best_text_model(config: Dict, train_df: pd.DataFrame, validate_df: pd.DataFrame, text_column: str) -> Pipeline:
    combined = pd.concat([train_df, validate_df], ignore_index=True)
    effective_config = dict(config)

    if "min_df" in effective_config:
        effective_config["effective_min_df"] = min(
            int(effective_config["min_df"]),
            max(1, len(combined) // 20),
            len(combined),
        )

    pipeline = make_text_pipeline(effective_config)
    try:
        pipeline.fit(combined[text_column], combined["binary_label"])
    except ValueError:
        if "effective_min_df" in effective_config:
            effective_config["effective_min_df"] = 1
        pipeline = make_text_pipeline(effective_config)
        pipeline.fit(combined[text_column], combined["binary_label"])

    return pipeline


def refit_best_metadata_model(
    config: Dict,
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    text_column: str,
    metadata_columns: List[str],
) -> Pipeline:
    combined = pd.concat([train_df, validate_df], ignore_index=True)
    effective_config = dict(config)
    effective_config["text_min_df"] = min(
        int(effective_config["text_min_df"]),
        max(1, len(combined) // 20),
        len(combined),
    )
    effective_config["title_min_df"] = min(
        int(effective_config["title_min_df"]),
        max(1, len(combined) // 20),
        len(combined),
    )

    pipeline = make_metadata_pipeline(effective_config, text_column, metadata_columns)
    try:
        pipeline.fit(combined, combined["binary_label"])
    except ValueError:
        effective_config["text_min_df"] = 1
        effective_config["title_min_df"] = 1
        pipeline = make_metadata_pipeline(effective_config, text_column, metadata_columns)
        pipeline.fit(combined, combined["binary_label"])

    return pipeline


def save_predictions_csv(path: Path, frame: pd.DataFrame, y_pred: List[int], text_column: str) -> None:
    output = pd.DataFrame()
    if ID_COLUMN in frame.columns:
        output[ID_COLUMN] = frame[ID_COLUMN]
    output[LABEL_COLUMN] = frame[LABEL_COLUMN]
    output["binary_label"] = frame["binary_label"]
    output["predicted_label"] = y_pred
    output[text_column] = frame[text_column]

    for column in ["title", "domain", "url"]:
        if column in frame.columns:
            output[column] = frame[column]

    output.to_csv(path, index=False)


def evaluate_text_model_on_test(
    model_name: str,
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    text_column: str,
    output_prefix: str,
) -> Dict:
    y_true = test_df["binary_label"].tolist()
    y_pred = pipeline.predict(test_df[text_column]).tolist()

    metrics = evaluate_predictions(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix_as_dict(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=["reliable", "fake"],
        zero_division=0,
    )

    plot_confusion(y_true, y_pred, f"{model_name} on test split", OUTPUT_DIR / f"{output_prefix}_confusion_matrix.png")
    save_predictions_csv(OUTPUT_DIR / f"{output_prefix}_test_predictions.csv", test_df, y_pred, text_column)
    save_text(OUTPUT_DIR / f"{output_prefix}_classification_report.txt", metrics["classification_report"])

    return metrics


def evaluate_metadata_model_on_test(
    model_name: str,
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    text_column: str,
    output_prefix: str,
) -> Dict:
    y_true = test_df["binary_label"].tolist()
    y_pred = pipeline.predict(test_df).tolist()

    metrics = evaluate_predictions(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix_as_dict(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=["reliable", "fake"],
        zero_division=0,
    )

    plot_confusion(y_true, y_pred, f"{model_name} on test split", OUTPUT_DIR / f"{output_prefix}_confusion_matrix.png")
    save_predictions_csv(OUTPUT_DIR / f"{output_prefix}_test_predictions.csv", test_df, y_pred, text_column)
    save_text(OUTPUT_DIR / f"{output_prefix}_classification_report.txt", metrics["classification_report"])

    return metrics


def create_report_helper(
    text_column: str,
    label_reports: Dict,
    binary_distributions: Dict,
    best_text_result: Dict,
    best_text_test_metrics: Dict,
    metadata_columns: List[str],
    best_metadata_result: Optional[Dict],
    best_metadata_test_metrics: Optional[Dict],
) -> str:
    lines = []
    lines.append("Part 3 Report Helper")
    lines.append("")
    lines.append("1. Label grouping")
    lines.append(
        f"The split files were loaded and the main text field used for modeling was '{text_column}'. "
        "Labels were grouped into two classes: reliable (0) and fake (1). "
        "Ambiguous labels such as satire, bias, political, mixed, and unknown were removed."
    )
    lines.append("")
    lines.append("2. Rows kept after label grouping")
    for split_name, report in label_reports.items():
        lines.append(
            f"- {split_name}: before={report['rows_before']}, after={report['rows_after']}"
        )
    lines.append("")
    lines.append("3. Binary distribution after grouping")
    for split_name, distribution in binary_distributions.items():
        lines.append(
            f"- {split_name}: reliable={distribution['reliable_0']}, fake={distribution['fake_1']}, total={distribution['total']}"
        )
    lines.append("")
    lines.append("4. Best text-only baseline")
    lines.append(
        f"The best text-only baseline was {best_text_result['model_name']} with "
        f"ngram_range={best_text_result.get('ngram_range')} and "
        f"max_features={best_text_result.get('max_features', HASHING_FEATURES)}. "
        f"On the test split it achieved accuracy={best_text_test_metrics['accuracy']:.4f}, "
        f"precision={best_text_test_metrics['precision']:.4f}, "
        f"recall={best_text_test_metrics['recall']:.4f}, "
        f"and F1={best_text_test_metrics['f1']:.4f}."
    )
    lines.append("")
    lines.append("5. Metadata discussion")
    if metadata_columns and best_metadata_result and best_metadata_test_metrics:
        lines.append(
            f"Metadata columns detected across all split files: {', '.join(metadata_columns)}. "
            f"The best text+metadata baseline was {best_metadata_result['model_name']}. "
            f"On the test split it achieved accuracy={best_metadata_test_metrics['accuracy']:.4f}, "
            f"precision={best_metadata_test_metrics['precision']:.4f}, "
            f"recall={best_metadata_test_metrics['recall']:.4f}, "
            f"and F1={best_metadata_test_metrics['f1']:.4f}."
        )
    else:
        lines.append(
            "The optional metadata experiment was skipped or no usable metadata columns were available across all split files."
        )
    lines.append("")
    lines.append("6. Useful interpretation")
    lines.append(
        "Model selection was run on a stratified sample to reduce memory usage, and the final chosen baseline was then refit on the full train+validate data. "
        "This keeps the baseline computationally practical while still using the full available labeled training data for the final test evaluation."
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_log("Part 3 - Simple Model")
    text_column = detect_shared_text_column([TRAIN_PATH, VALIDATE_PATH, TEST_PATH])
    shared_metadata_columns = detect_shared_metadata_columns([TRAIN_PATH, VALIDATE_PATH, TEST_PATH])

    print_log(f"Detected shared text column: {text_column}")
    print_log(f"Shared metadata columns from headers: {shared_metadata_columns if shared_metadata_columns else 'None'}")

    splits = load_splits(
        TRAIN_PATH,
        VALIDATE_PATH,
        TEST_PATH,
        text_column=text_column,
        metadata_columns=shared_metadata_columns,
    )

    mapped_splits, label_reports = map_splits_to_binary(splits)
    save_json(OUTPUT_DIR / "label_mapping_report.json", label_reports)

    binary_distributions = {
        "train": describe_binary_distribution(mapped_splits.train),
        "validate": describe_binary_distribution(mapped_splits.validate),
        "test": describe_binary_distribution(mapped_splits.test),
    }

    print_log("Binary label distribution after grouping")
    for split_name, distribution in binary_distributions.items():
        print_log(
            f"{split_name:<8} reliable={distribution['reliable_0']:,} | "
            f"fake={distribution['fake_1']:,} | total={distribution['total']:,}"
        )

    text_best_pipeline, text_best_config, text_results = select_best_text_model(
        mapped_splits.train,
        mapped_splits.validate,
        text_column=text_column,
    )
    text_results.to_csv(OUTPUT_DIR / "text_model_selection_results.csv", index=False)

    final_text_pipeline = refit_best_text_model(
        text_best_config,
        mapped_splits.train,
        mapped_splits.validate,
        text_column=text_column,
    )
    joblib.dump(final_text_pipeline, OUTPUT_DIR / "best_text_model.joblib")

    best_text_test_metrics = evaluate_text_model_on_test(
        model_name=text_best_config["model_name"],
        pipeline=final_text_pipeline,
        test_df=mapped_splits.test,
        text_column=text_column,
        output_prefix="best_text_model",
    )

    metadata_columns = available_metadata_columns(mapped_splits, shared_metadata_columns)
    metadata_best_config = None
    best_metadata_test_metrics = None

    if RUN_METADATA_EXPERIMENT and metadata_columns:
        metadata_best_pipeline, metadata_best_config, metadata_results = select_best_metadata_model(
            mapped_splits.train,
            mapped_splits.validate,
            text_column=text_column,
            metadata_columns=metadata_columns,
        )

        if metadata_results is not None and metadata_best_config is not None and metadata_best_pipeline is not None:
            metadata_results.to_csv(OUTPUT_DIR / "metadata_model_selection_results.csv", index=False)
            final_metadata_pipeline = refit_best_metadata_model(
                metadata_best_config,
                mapped_splits.train,
                mapped_splits.validate,
                text_column=text_column,
                metadata_columns=metadata_columns,
            )
            joblib.dump(final_metadata_pipeline, OUTPUT_DIR / "best_metadata_model.joblib")
            best_metadata_test_metrics = evaluate_metadata_model_on_test(
                model_name=metadata_best_config["model_name"],
                pipeline=final_metadata_pipeline,
                test_df=mapped_splits.test,
                text_column=text_column,
                output_prefix="best_metadata_model",
            )
    else:
        metadata_columns = []

    summary_metrics = {
        "input_paths": {
            "train": str(TRAIN_PATH),
            "validate": str(VALIDATE_PATH),
            "test": str(TEST_PATH),
        },
        "detected_text_column": text_column,
        "selection_train_sample": TRAIN_SELECTION_SAMPLE,
        "selection_validate_sample": VALID_SELECTION_SAMPLE,
        "use_hashing_fallback": USE_HASHING_FALLBACK,
        "label_mapping": label_reports,
        "binary_distribution": binary_distributions,
        "best_text_only_model": {
            "validation_selection_result": text_best_config,
            "test_metrics": best_text_test_metrics,
        },
        "metadata_columns_detected": metadata_columns,
    }

    if metadata_columns and metadata_best_config is not None and best_metadata_test_metrics is not None:
        summary_metrics["best_text_plus_metadata_model"] = {
            "validation_selection_result": metadata_best_config,
            "test_metrics": best_metadata_test_metrics,
        }
    else:
        summary_metrics["best_text_plus_metadata_model"] = {
            "skipped": True,
            "reason": "Metadata experiment disabled or no usable metadata columns were available.",
        }

    save_json(OUTPUT_DIR / "summary_metrics.json", summary_metrics)

    report_helper = create_report_helper(
        text_column=text_column,
        label_reports=label_reports,
        binary_distributions=binary_distributions,
        best_text_result=text_best_config,
        best_text_test_metrics=best_text_test_metrics,
        metadata_columns=metadata_columns,
        best_metadata_result=metadata_best_config,
        best_metadata_test_metrics=best_metadata_test_metrics,
    )
    save_text(OUTPUT_DIR / "part3_report_helper.md", report_helper)

    print_log("Best text-only model on test split")
    print_log(json.dumps(summary_metrics["best_text_only_model"], indent=2))

    if metadata_columns and metadata_best_config is not None and best_metadata_test_metrics is not None:
        print_log("Best text + metadata model on test split")
        print_log(json.dumps(summary_metrics["best_text_plus_metadata_model"], indent=2))
    else:
        print_log("Metadata experiment skipped because it is disabled or no usable metadata columns were available")

    print_log(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()