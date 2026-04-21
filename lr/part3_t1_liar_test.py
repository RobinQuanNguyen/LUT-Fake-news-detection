"""
PART 3 - TF-IDF + SGDClassifier (Train on FakeNewsCorpus, test on LIAR)
Memory-optimized with incremental training.
Binary classification: FAKE (0) vs RELIABLE (1)

Changes for LIAR test set:
- Keep train/validation from FakeNewsCorpus CSV files.
- Allow test set to be loaded from LIAR test.tsv.
- Convert LIAR multi-class labels into binary labels:
    FAKE (0): false, pants-fire, barely-true
    RELIABLE (1): true, mostly-true
    DROPPED: half-true
"""

import json
import time
import gc
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random

# ========== CONFIGURATION ==========
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/part3_text")
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

RANDOM_SEED = 42
BATCH_SIZE = 50_000

# FakeNewsCorpus label mapping
RELIABLE_LABELS = {"reliable"}
FAKE_LABELS = {"fake", "unreliable", "conspiracy", "rumor", "junksci", "clickbait", "hate", "satire"}
DROP_LABELS = {"unknown", "political", "bias"}

# LIAR label mapping (binary conversion)
LIAR_RELIABLE_LABELS = {"true", "mostly-true"}
LIAR_FAKE_LABELS = {"false", "pants-fire", "barely-true"}
LIAR_DROP_LABELS = {"half-true"}

# TF-IDF settings
TEXT_NGRAM_RANGE = (1, 2)
TEXT_MAX_FEATURES = 150_000
TEXT_MIN_DF = 5
TEXT_MAX_DF = 0.9
SUBLINEAR_TF = True
STOP_WORDS = "english"

# SGD settings
SGD_LOSS = "hinge"
SGD_ALPHA = 1e-4

# File paths
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validate.csv"
TEST_CSV_PATH = DATA_DIR / "test.csv"
TEST_LIAR_PATH = DATA_DIR / "test.tsv"


# ========== SETUP ==========
def setup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def print_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ========== TEXT CLEANING ==========
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ========== DATA LOADING ==========
def load_prepare_fakenews_csv(path, split_name):
    """Load FakeNewsCorpus CSV, map labels to binary, drop ambiguous rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print_log(f"Loading {split_name} from FakeNewsCorpus CSV: {path}")
    df = pd.read_csv(path, usecols=["processed_text", "type"], encoding="utf-8")
    print_log(f"Rows loaded: {len(df):,}")

    before = len(df)
    df = df.dropna(subset=["processed_text", "type"])
    if len(df) < before:
        print_log(f"Dropped {before - len(df):,} rows (missing values)")

    df["type"] = df["type"].astype(str).str.strip().str.lower()

    before = len(df)
    df = df[~df["type"].isin(DROP_LABELS)]
    dropped = before - len(df)
    if dropped:
        print_log(f"Dropped {dropped:,} rows (ambiguous labels: {DROP_LABELS})")

    def map_label(label):
        if label in RELIABLE_LABELS:
            return 1
        if label in FAKE_LABELS:
            return 0
        return None

    df["label"] = df["type"].map(map_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["processed_text"] = df["processed_text"].astype(str).map(clean_text)
    df = df[df["processed_text"].str.len() > 0].copy()

    count_reliable = int((df["label"] == 1).sum())
    count_fake = int((df["label"] == 0).sum())
    print_log(f"RELIABLE (1): {count_reliable:,} ({count_reliable / len(df) * 100:.1f}%)")
    print_log(f"FAKE (0): {count_fake:,} ({count_fake / len(df) * 100:.1f}%)")

    return df[["processed_text", "label"]]


def load_prepare_liar_tsv(path, split_name="test"):
    """Load LIAR TSV, convert multi-class labels to binary, and standardize columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print_log(f"Loading {split_name} from LIAR TSV: {path}")

    liar_columns = [
        "id", "label_raw", "statement", "subjects", "speaker", "speaker_job_title",
        "state_info", "party_affiliation", "barely_true_counts", "false_counts",
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
    ]

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=liar_columns,
        encoding="utf-8",
    )
    print_log(f"Rows loaded: {len(df):,}")

    before = len(df)
    df = df.dropna(subset=["statement", "label_raw"])
    if len(df) < before:
        print_log(f"Dropped {before - len(df):,} rows (missing statement/label)")

    df["label_raw"] = df["label_raw"].astype(str).str.strip().str.lower()
    df["processed_text"] = df["statement"].astype(str).map(clean_text)
    df = df[df["processed_text"].str.len() > 0].copy()

    before = len(df)
    df = df[~df["label_raw"].isin(LIAR_DROP_LABELS)].copy()
    dropped = before - len(df)
    if dropped:
        print_log(f"Dropped {dropped:,} rows (ambiguous LIAR labels: {LIAR_DROP_LABELS})")

    def map_liar_label(label):
        if label in LIAR_RELIABLE_LABELS:
            return 1
        if label in LIAR_FAKE_LABELS:
            return 0
        return None

    df["label"] = df["label_raw"].map(map_liar_label)
    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    unmapped = before - len(df)
    if unmapped:
        print_log(f"Dropped {unmapped:,} rows (unmapped LIAR labels)")

    df["label"] = df["label"].astype(int)

    count_reliable = int((df["label"] == 1).sum())
    count_fake = int((df["label"] == 0).sum())
    print_log(f"RELIABLE (1): {count_reliable:,} ({count_reliable / len(df) * 100:.1f}%)")
    print_log(f"FAKE (0): {count_fake:,} ({count_fake / len(df) * 100:.1f}%)")

    return df[["processed_text", "label"]]


def load_test_data():
    """Prefer LIAR test.tsv if present, otherwise use original FakeNewsCorpus test.csv."""
    if TEST_LIAR_PATH.exists():
        return load_prepare_liar_tsv(TEST_LIAR_PATH, "test")
    if TEST_CSV_PATH.exists():
        return load_prepare_fakenews_csv(TEST_CSV_PATH, "test")
    raise FileNotFoundError(
        f"Neither {TEST_LIAR_PATH} nor {TEST_CSV_PATH} exists. Put your test file inside {DATA_DIR}/"
    )


# ========== VECTORIZATION ==========
def fit_vectorizer_on_sample(train_df, sample_size=500_000):
    """Fit vectorizer on a sample to save memory."""
    print_log(f"Fitting vectorizer on {sample_size:,} sample...")

    if len(train_df) > sample_size:
        sample_df = train_df.sample(n=sample_size, random_state=RANDOM_SEED)
    else:
        sample_df = train_df

    vectorizer = TfidfVectorizer(
        ngram_range=TEXT_NGRAM_RANGE,
        max_features=TEXT_MAX_FEATURES,
        min_df=TEXT_MIN_DF,
        max_df=TEXT_MAX_DF,
        sublinear_tf=SUBLINEAR_TF,
        strip_accents="unicode",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
        stop_words=STOP_WORDS,
        use_idf=True,
        smooth_idf=True,
    )

    vectorizer.fit(sample_df["processed_text"].values)
    print_log(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    del sample_df
    gc.collect()

    return vectorizer


def transform_in_batches(vectorizer, df, batch_size=100_000):
    """Transform dataframe in batches."""
    from scipy.sparse import vstack

    texts = df["processed_text"].values
    n_samples = len(texts)
    all_features = []

    print_log(f"Transforming {n_samples:,} samples in batches...")

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_features = vectorizer.transform(texts[start:end])
        all_features.append(batch_features)

        if start % (batch_size * 5) == 0:
            print_log(f"  Progress: {start:,}/{n_samples:,}")

    X = vstack(all_features)
    print_log(f"Transform complete: {X.shape}")
    return X


# ========== TRAINING ==========
def train_incremental(X_train, y_train, n_epochs=3):
    """Train SGDClassifier with incremental learning."""
    print_log(f"Training SGDClassifier ({n_epochs} epochs)...")
    start_time = time.time()

    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    y_for_weights = y_train[: min(len(y_train), 100_000)]
    class_weights = compute_class_weight("balanced", classes=classes, y=y_for_weights)
    class_weight = dict(zip(classes, class_weights))

    model = SGDClassifier(
        loss=SGD_LOSS,
        alpha=SGD_ALPHA,
        max_iter=1,
        tol=None,
        random_state=RANDOM_SEED,
        class_weight=class_weight,
        n_jobs=-1,
        warm_start=True,
    )

    n_samples = X_train.shape[0]

    init_batch = min(BATCH_SIZE, n_samples)
    print_log(f"Initial fit on {init_batch:,} samples...")
    model.partial_fit(X_train[:init_batch], y_train[:init_batch], classes=classes)

    for epoch in range(n_epochs):
        print_log(f"Epoch {epoch + 1}/{n_epochs}...")
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_samples)
            batch_idx = indices[start:end]

            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            model.partial_fit(X_batch, y_batch)

            if start % (BATCH_SIZE * 5) == 0:
                print_log(f"  Progress: {start:,}/{n_samples:,}")

        gc.collect()

    train_time = time.time() - start_time
    print_log(f"Training time: {train_time:.1f}s")

    return model


# ========== EVALUATION ==========
def evaluate_model(model, X, y_true, name):
    """Evaluate model."""
    print_log(f"\nEvaluating {name}...")
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    print_log(f"{name} - Accuracy: {acc:.4f}")
    print_log(f"{name} - Precision (FAKE): {prec:.4f}")
    print_log(f"{name} - Recall (FAKE): {rec:.4f}")
    print_log(f"{name} - F1 (FAKE): {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


# ========== SAVE RESULTS ==========
def plot_confusion_matrix(cm, class_names, save_path, title):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print_log(f"Confusion matrix saved: {save_path}")


def save_results(model, vectorizer, val_metrics, test_metrics, train_size, val_size, test_size, test_source):
    model_name = "SGDClassifier"

    class_names = ["FAKE (0)", "RELIABLE (1)"]
    cm = test_metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(REPORT_DIR / "confusion_matrix.csv")

    plot_confusion_matrix(
        cm,
        class_names,
        REPORT_DIR / "confusion_matrix.png",
        title=f"Confusion Matrix - Test Set ({test_source})",
    )

    summary = {
        "model": model_name,
        "features": "processed_text (TF-IDF)",
        "ngram_range": list(TEXT_NGRAM_RANGE),
        "max_features": TEXT_MAX_FEATURES,
        "min_df": TEXT_MIN_DF,
        "max_df": TEXT_MAX_DF,
        "alpha": SGD_ALPHA,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "test_source": test_source,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1_fake": float(test_metrics["f1"]),
        "test_precision_fake": float(test_metrics["precision"]),
        "test_recall_fake": float(test_metrics["recall"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_f1_fake": float(val_metrics["f1"]),
    }

    txt_output = []
    txt_output.append("=" * 70)
    txt_output.append(f"{model_name} - RESULTS REPORT")
    txt_output.append("=" * 70)
    txt_output.append(f"Model               : {model_name}")
    txt_output.append(f"Max features        : {TEXT_MAX_FEATURES:,}")
    txt_output.append(f"N-gram range        : {TEXT_NGRAM_RANGE}")
    txt_output.append(f"min_df / max_df     : {TEXT_MIN_DF} / {TEXT_MAX_DF}")
    txt_output.append(f"Alpha (reg)         : {SGD_ALPHA}")
    txt_output.append(f"Batch size          : {BATCH_SIZE:,}")
    txt_output.append(f"Test source         : {test_source}")
    txt_output.append("")
    txt_output.append("Dataset sizes:")
    txt_output.append(f"  Train: {train_size:,}")
    txt_output.append(f"  Validation: {val_size:,}")
    txt_output.append(f"  Test: {test_size:,}")
    txt_output.append("")
    txt_output.append("Split          Accuracy     Precision(F)   Recall(F)      F1(F)")
    txt_output.append("-" * 70)
    txt_output.append(f"validation     {val_metrics['accuracy']:.4f}      {val_metrics['precision']:.4f}         {val_metrics['recall']:.4f}         {val_metrics['f1']:.4f}")
    txt_output.append(f"test           {test_metrics['accuracy']:.4f}      {test_metrics['precision']:.4f}         {test_metrics['recall']:.4f}         {test_metrics['f1']:.4f}")
    txt_output.append("")
    txt_output.append("=" * 70)
    txt_output.append("CONFUSION MATRIX (Test Set)")
    txt_output.append("=" * 70)
    txt_output.append(str(cm_df))
    txt_output.append("")

    with open(REPORT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_output))

    with open(REPORT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(MODEL_DIR / "sgd_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print_log(f"\nResults saved to {REPORT_DIR}")
    print_log(f"Model saved to {MODEL_DIR}")

    return summary


# ========== MAIN ==========
def main():
    print_log("=" * 70)
    print_log("PART 3 - TF-IDF + SGDClassifier")
    print_log(f"Batch: {BATCH_SIZE:,}, Features: {TEXT_MAX_FEATURES:,}")
    print_log("Train/Val: FakeNewsCorpus CSV | Test: LIAR TSV if available")
    print_log("=" * 70)

    setup()
    set_random_seed(RANDOM_SEED)

    train_df = load_prepare_fakenews_csv(TRAIN_PATH, "train")
    val_df = load_prepare_fakenews_csv(VAL_PATH, "validate")

    if TEST_LIAR_PATH.exists():
        test_df = load_prepare_liar_tsv(TEST_LIAR_PATH, "test")
        test_source = "LIAR test.tsv"
    else:
        test_df = load_prepare_fakenews_csv(TEST_CSV_PATH, "test")
        test_source = "FakeNewsCorpus test.csv"

    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)

    vectorizer = fit_vectorizer_on_sample(train_df, sample_size=500_000)

    print_log("\nTransforming train data...")
    X_train = transform_in_batches(vectorizer, train_df)
    y_train = train_df["label"].values
    del train_df
    gc.collect()

    print_log("\nTransforming validation data...")
    X_val = transform_in_batches(vectorizer, val_df)
    y_val = val_df["label"].values
    del val_df
    gc.collect()

    print_log("\nTransforming test data...")
    X_test = transform_in_batches(vectorizer, test_df)
    y_test = test_df["label"].values
    del test_df
    gc.collect()

    model = train_incremental(X_train, y_train, n_epochs=3)

    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, f"Test ({test_source})")

    save_results(
        model,
        vectorizer,
        val_metrics,
        test_metrics,
        train_size,
        val_size,
        test_size,
        test_source,
    )

    print_log("\n" + "=" * 70)
    print_log("FINAL RESULTS")
    print_log("=" * 70)
    print_log(f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1 (fake): {val_metrics['f1']:.4f}")
    print_log(f"Test Acc: {test_metrics['accuracy']:.4f}, Test F1 (fake): {test_metrics['f1']:.4f}")
    print_log("=" * 70)


if __name__ == "__main__":
    main()
