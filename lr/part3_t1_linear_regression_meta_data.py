"""
PART 3 - TF-IDF + SGDClassifier (Full Dataset with Metadata)
Sử dụng full dataset, memory optimized với incremental training.
Binary: FAKE=0, RELIABLE=1
"""

import json
import time
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.sparse import hstack, vstack
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random

# ========== CONFIGURATION ==========
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/part3_meta")
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

RANDOM_SEED = 42
BATCH_SIZE = 50_000

# Label mapping
RELIABLE_LABELS = {"reliable"}
FAKE_LABELS = {"fake", "unreliable", "conspiracy", "rumor", "junksci", "clickbait", "hate", "satire"}
DROP_LABELS = {"unknown", "political", "bias"}

# TF-IDF settings
TEXT_NGRAM_RANGE = (1, 2)
TEXT_MAX_FEATURES = 150_000
TEXT_MIN_DF = 5
TEXT_MAX_DF = 0.9
TEXT_STOP_WORDS = 'english'

TITLE_NGRAM_RANGE = (1, 2)
TITLE_MAX_FEATURES = 30_000
TITLE_MIN_DF = 3
TITLE_STOP_WORDS = 'english'

SUBLINEAR_TF = True

# SGD settings
SGD_LOSS = 'hinge'
SGD_ALPHA = 1e-4

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

# ========== DATA LOADING ==========
def load_prepare(path, split_name):
    """Load CSV, map labels to binary, drop ambiguous rows"""
    print_log(f"Loading {split_name} from: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print_log(f"Rows loaded: {len(df):,}")

    # Drop missing values
    before = len(df)
    df = df.dropna(subset=["processed_text", "type"])
    if len(df) < before:
        print_log(f"Dropped {before - len(df):,} rows (missing values)")

    # Ensure text columns are strings
    for col in ["processed_text", "title"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    # Drop ambiguous labels
    before = len(df)
    df = df[~df["type"].isin(DROP_LABELS)]
    dropped = before - len(df)
    if dropped:
        print_log(f"Dropped {dropped:,} rows (ambiguous labels)")

    # Map to binary
    def map_label(label):
        if label in RELIABLE_LABELS:
            return 1
        elif label in FAKE_LABELS:
            return 0
        return None

    df["label"] = df["type"].map(map_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    count_reliable = (df["label"] == 1).sum()
    count_fake = (df["label"] == 0).sum()
    print_log(f"RELIABLE (1): {count_reliable:,} ({count_reliable/len(df)*100:.1f}%)")
    print_log(f"FAKE (0): {count_fake:,} ({count_fake/len(df)*100:.1f}%)")

    return df

# ========== VECTORIZATION ==========
def fit_vectorizers_on_sample(train_df, sample_size=500_000):
    """Fit vectorizers on a sample"""
    print_log(f"Fitting vectorizers on {sample_size:,} sample...")

    if len(train_df) > sample_size:
        sample_df = train_df.sample(n=sample_size, random_state=RANDOM_SEED)
    else:
        sample_df = train_df

    text_vectorizer = TfidfVectorizer(
        ngram_range=TEXT_NGRAM_RANGE,
        max_features=TEXT_MAX_FEATURES,
        min_df=TEXT_MIN_DF,
        max_df=TEXT_MAX_DF,
        sublinear_tf=SUBLINEAR_TF,
        strip_accents="unicode",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
        stop_words=TEXT_STOP_WORDS,
        use_idf=True,
        smooth_idf=True
    )

    title_vectorizer = TfidfVectorizer(
        ngram_range=TITLE_NGRAM_RANGE,
        max_features=TITLE_MAX_FEATURES,
        min_df=TITLE_MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
        strip_accents="unicode",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
        stop_words=TITLE_STOP_WORDS,
        use_idf=True,
        smooth_idf=True
    )

    print_log("  Fitting text TF-IDF...")
    text_vectorizer.fit(sample_df["processed_text"].values)

    print_log("  Fitting title TF-IDF...")
    title_vectorizer.fit(sample_df["title"].values)

    print_log(f"Text vocab: {len(text_vectorizer.vocabulary_):,}")
    print_log(f"Title vocab: {len(title_vectorizer.vocabulary_):,}")

    del sample_df
    gc.collect()

    return text_vectorizer, title_vectorizer

def transform_in_batches(text_vec, title_vec, df, batch_size=100_000):
    """Transform dataframe in batches"""
    texts = df["processed_text"].values
    titles = df["title"].values if "title" in df.columns else [""]
    n_samples = len(texts)
    all_features = []

    print_log(f"Transforming {n_samples:,} samples in batches...")

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        X_text = text_vec.transform(texts[start:end])
        X_title = title_vec.transform(titles[start:end])
        X_combined = hstack([X_text, X_title])
        all_features.append(X_combined)

        if start % (batch_size * 5) == 0:
            print_log(f"  Progress: {start:,}/{n_samples:,}")

    X = vstack(all_features)
    print_log(f"Transform complete: {X.shape}")
    return X

# ========== TRAINING ==========
def train_incremental(X_train, y_train, n_epochs=3):
    """Train SGDClassifier with incremental learning"""
    print_log(f"Training SGDClassifier ({n_epochs} epochs)...")
    start_time = time.time()

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train[:100000])
    class_weight = dict(zip(classes, class_weights))

    model = SGDClassifier(
        loss=SGD_LOSS,
        alpha=SGD_ALPHA,
        max_iter=1,
        tol=None,
        random_state=RANDOM_SEED,
        class_weight=class_weight,
        n_jobs=-1,
        warm_start=True
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
    """Evaluate model"""
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
        "predictions": y_pred
    }

# ========== SAVE RESULTS ==========
def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Fake News Detection (Full Dataset + Metadata)', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print_log(f"Confusion matrix saved: {save_path}")

def save_results(model, vectorizers, val_metrics, test_metrics, train_size, val_size, test_size):
    model_name = "SGDClassifier"
    text_vec, title_vec = vectorizers

    class_names = ['FAKE (0)', 'RELIABLE (1)']
    cm = test_metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(REPORT_DIR / "confusion_matrix.csv")

    plot_confusion_matrix(cm, class_names, REPORT_DIR / "confusion_matrix.png")

    summary = {
        "model": model_name,
        "features": "processed_text + title (TF-IDF - Full Dataset)",
        "text_max_features": TEXT_MAX_FEATURES,
        "title_max_features": TITLE_MAX_FEATURES,
        "alpha": SGD_ALPHA,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1_fake": float(test_metrics["f1"]),
        "test_precision_fake": float(test_metrics["precision"]),
        "test_recall_fake": float(test_metrics["recall"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_f1_fake": float(val_metrics["f1"]),
    }

    txt_output = []
    txt_output.append("=" * 70)
    txt_output.append(f"{model_name} (FULL DATASET + METADATA) - RESULTS REPORT")
    txt_output.append("=" * 70)
    txt_output.append(f"Model               : {model_name}")
    txt_output.append(f"Text max features   : {TEXT_MAX_FEATURES:,}")
    txt_output.append(f"Title max features : {TITLE_MAX_FEATURES:,}")
    txt_output.append(f"Alpha (reg)        : {SGD_ALPHA}")
    txt_output.append(f"Batch size         : {BATCH_SIZE:,}")
    txt_output.append("")
    txt_output.append(f"Dataset sizes:")
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

    with open(REPORT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(MODEL_DIR / "sgd_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODEL_DIR / "tfidf_vectorizers.pkl", "wb") as f:
        pickle.dump(vectorizers, f)

    print_log(f"\nResults saved to {REPORT_DIR}")
    print_log(f"Model saved to {MODEL_DIR}")

    return summary

# ========== MAIN ==========
def main():
    print_log("=" * 70)
    print_log("PART 3 - TF-IDF + SGDClassifier (Full Dataset + Metadata)")
    print_log(f"Batch: {BATCH_SIZE:,}, Text: {TEXT_MAX_FEATURES:,}, Title: {TITLE_MAX_FEATURES:,}")
    print_log("=" * 70)

    setup()
    set_random_seed(RANDOM_SEED)

    # Load data
    train_df = load_prepare(DATA_DIR / "train_meta.csv", "train")
    val_df = load_prepare(DATA_DIR / "validate_meta.csv", "validate")
    test_df = load_prepare(DATA_DIR / "test_meta.csv", "test")

    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)

    # Fit vectorizers on sample
    text_vec, title_vec = fit_vectorizers_on_sample(train_df, sample_size=500_000)
    vectorizers = (text_vec, title_vec)

    # Transform train
    print_log("\nTransforming train data...")
    X_train = transform_in_batches(text_vec, title_vec, train_df)
    y_train = train_df["label"].values
    del train_df
    gc.collect()

    # Transform validation
    print_log("\nTransforming validation data...")
    X_val = transform_in_batches(text_vec, title_vec, val_df)
    y_val = val_df["label"].values
    del val_df
    gc.collect()

    # Transform test
    print_log("\nTransforming test data...")
    X_test = transform_in_batches(text_vec, title_vec, test_df)
    y_test = test_df["label"].values
    del test_df
    gc.collect()

    # Train
    model = train_incremental(X_train, y_train, n_epochs=3)

    # Evaluate
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # Save
    summary = save_results(model, vectorizers, val_metrics, test_metrics,
                         train_size, val_size, test_size)

    print_log("\n" + "=" * 70)
    print_log("FINAL RESULTS")
    print_log("=" * 70)
    print_log(f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1 (fake): {val_metrics['f1']:.4f}")
    print_log(f"Test Acc: {test_metrics['accuracy']:.4f}, Test F1 (fake): {test_metrics['f1']:.4f}")
    print_log("=" * 70)

if __name__ == "__main__":
    main()
