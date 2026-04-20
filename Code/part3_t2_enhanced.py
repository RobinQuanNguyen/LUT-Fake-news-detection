"""
PART 3 - TF-IDF + LinearSVC (Enhanced with Metadata - Binary Classification)
Tối ưu feature engineering cho text + title với binary classification.
Binary: FAKE=0, RELIABLE=1
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.sparse import hstack
import pickle
import random

# ========== CONFIGURATION ==========
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/part3_meta")
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

RANDOM_SEED = 42
SAMPLE_SIZE = 200_000  # Giới hạn sample để tránh MemoryError

# Label mapping (Binary classification)
RELIABLE_LABELS = {"reliable"}
FAKE_LABELS = {"fake", "unreliable", "conspiracy", "rumor", "junksci", "clickbait", "hate", "satire"}
DROP_LABELS = {"unknown", "political", "bias"}  # Drop ambiguous labels

# TF-IDF settings - OPTIMIZED (giảm max_features để tránh MemoryError)
TEXT_NGRAM_RANGE = (1, 3)
TEXT_MAX_FEATURES = 100000  # Giảm từ 150k → 100k
TEXT_MIN_DF = 3
TEXT_MAX_DF = 0.9
TEXT_STOP_WORDS = 'english'

TITLE_NGRAM_RANGE = (1, 2)
TITLE_MAX_FEATURES = 25000  # Giảm từ 30k → 25k
TITLE_MIN_DF = 2
TITLE_STOP_WORDS = 'english'

SUBLINEAR_TF = True

# LinearSVC settings - OPTIMIZED
SVC_C = 1.0
SVC_MAX_ITER = 5000
SVC_DUAL = False
SVC_CLASS_WEIGHT = 'balanced'

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
def load_metadata():
    print_log("Loading metadata...")
    start_time = time.time()

    train_path = DATA_DIR / "train_meta.csv"
    test_path = DATA_DIR / "test_meta.csv"
    validate_path = DATA_DIR / "validate_meta.csv"

    train_df = pd.read_csv(train_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")
    validate_df = pd.read_csv(validate_path, encoding="utf-8")

    print_log(f"Train: {len(train_df):,} rows")
    print_log(f"Validate: {len(validate_df):,} rows")
    print_log(f"Test: {len(test_df):,} rows")

    combined_df = pd.concat([train_df, validate_df, test_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=["processed_text", "type"])

    for col in ["processed_text", "title", "domain"]:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str).fillna("")

    print_log(f"Combined (after dropna): {len(combined_df):,} rows")
    print_log(f"Load time: {time.time() - start_time:.1f}s")

    return combined_df

def map_to_binary(df):
    """Map multi-class labels to binary: FAKE=0, RELIABLE=1, drop ambiguous"""
    print_log("Mapping to binary labels (FAKE=0, RELIABLE=1)...")

    before = len(df)
    # Drop ambiguous labels
    df = df[~df["type"].isin(DROP_LABELS)]
    dropped = before - len(df)
    if dropped:
        print_log(f"Dropped {dropped:,} rows (ambiguous labels: {DROP_LABELS})")

    def map_label(label):
        if label in RELIABLE_LABELS:
            return 1
        elif label in FAKE_LABELS:
            return 0
        else:
            return None

    df["binary_label"] = df["type"].map(map_label)
    df = df.dropna(subset=["binary_label"])
    df["binary_label"] = df["binary_label"].astype(int)

    count_reliable = (df["binary_label"] == 1).sum()
    count_fake = (df["binary_label"] == 0).sum()
    print_log(f"RELIABLE (1): {count_reliable:,} ({count_reliable/len(df)*100:.1f}%)")
    print_log(f"FAKE     (0): {count_fake:,} ({count_fake/len(df)*100:.1f}%)")

    return df

def stratified_sample(df, n_samples, seed=42):
    """Stratified sampling to maintain class distribution and limit memory usage"""
    print_log(f"Sampling {n_samples:,} rows (stratified by binary_label)...")
    start_time = time.time()

    sampled_list = []
    for label in df["binary_label"].unique():
        class_df = df[df["binary_label"] == label]
        # Calculate proportion for this class
        prop = len(class_df) / len(df)
        n_per_class = int(n_samples * prop)
        # Cap at available samples
        n_per_class = min(n_per_class, len(class_df))
        class_sample = class_df.sample(n=n_per_class, random_state=seed)
        sampled_list.append(class_sample)

    sampled_df = pd.concat(sampled_list, ignore_index=True)
    # Shuffle to mix classes
    sampled_df = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print_log(f"Sampled: {len(sampled_df):,} rows")
    print_log(f"Class distribution:")
    for cls, count in sampled_df["binary_label"].value_counts().items():
        label_name = "RELIABLE" if cls == 1 else "FAKE"
        print_log(f"  {label_name}: {count:,} ({count/len(sampled_df)*100:.1f}%)")
    print_log(f"Sample time: {time.time() - start_time:.1f}s")

    return sampled_df

# ========== FEATURE EXTRACTION ==========
def extract_features(train_df, val_df, test_df):
    print_log("Extracting TF-IDF features (binary classification)...")
    start_time = time.time()

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

    train_texts = train_df["processed_text"].values
    val_texts = val_df["processed_text"].values
    test_texts = test_df["processed_text"].values

    train_titles = train_df["title"].values if "title" in train_df.columns else [""] * len(train_df)
    val_titles = val_df["title"].values if "title" in val_df.columns else [""] * len(val_df)
    test_titles = test_df["title"].values if "title" in test_df.columns else [""] * len(test_df)

    print_log("  Fitting text TF-IDF...")
    X_train_text = text_vectorizer.fit_transform(train_texts)
    X_val_text = text_vectorizer.transform(val_texts)
    X_test_text = text_vectorizer.transform(test_texts)

    print_log("  Fitting title TF-IDF...")
    X_train_title = title_vectorizer.fit_transform(train_titles)
    X_val_title = title_vectorizer.transform(val_titles)
    X_test_title = title_vectorizer.transform(test_titles)

    print_log("  Combining features...")
    X_train = hstack([X_train_text, X_train_title])
    X_val = hstack([X_val_text, X_val_title])
    X_test = hstack([X_test_text, X_test_title])

    text_vocab = len(text_vectorizer.vocabulary_)
    title_vocab = len(title_vectorizer.vocabulary_)
    print_log(f"Text features: {X_train_text.shape[1]:,}")
    print_log(f"Title features: {X_train_title.shape[1]:,}")
    print_log(f"Combined features: {X_train.shape[1]:,}")
    print_log(f"Feature extraction time: {time.time() - start_time:.1f}s")

    return (text_vectorizer, title_vectorizer), X_train, X_val, X_test

# ========== MODEL TRAINING ==========
def train_and_evaluate(X_train, y_train, X_val, y_val):
    print_log("Training LinearSVC (binary classification)...")
    start_time = time.time()

    model = LinearSVC(
        C=SVC_C,
        max_iter=SVC_MAX_ITER,
        random_state=RANDOM_SEED,
        dual=SVC_DUAL,
        class_weight=SVC_CLASS_WEIGHT,
        loss='squared_hinge'
    )

    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    # Binary metrics - focus on fake class (0) like NB baseline
    val_f1 = f1_score(y_val, y_val_pred, pos_label=0, zero_division=0)
    val_precision = precision_score(y_val, y_val_pred, pos_label=0, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, pos_label=0, zero_division=0)

    print_log(f"Training time: {train_time:.1f}s")
    print_log(f"Validation - Accuracy: {val_accuracy:.4f}, F1 (fake): {val_f1:.4f}")

    return {
        "model": model,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "train_time": train_time,
        "predictions": y_val_pred
    }

# ========== SAVE RESULTS ==========
def save_results(result, X_test, y_test, vectorizers):
    model = result["model"]
    model_name = "LinearSVC"

    print_log(f"\n{'='*60}")
    print_log(f"TEST RESULTS - {model_name} (Binary + Metadata)")
    print_log(f"{'='*60}")

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=0, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, pos_label=0, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, pos_label=0, zero_division=0)

    print_log(f"Test Accuracy: {test_accuracy:.4f}")
    print_log(f"Test F1 (fake): {test_f1:.4f}")
    print_log(f"Test Precision (fake): {test_precision:.4f}")
    print_log(f"Test Recall (fake): {test_recall:.4f}")

    class_names = ['FAKE (0)', 'RELIABLE (1)']
    report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True, zero_division=0)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(REPORT_DIR / "confusion_matrix.csv")

    summary = {
        "model": model_name,
        "features": "processed_text + title (TF-IDF Enhanced - Binary)",
        "text_ngram_range": list(TEXT_NGRAM_RANGE),
        "text_max_features": TEXT_MAX_FEATURES,
        "text_min_df": TEXT_MIN_DF,
        "text_max_df": TEXT_MAX_DF,
        "title_max_features": TITLE_MAX_FEATURES,
        "C": SVC_C,
        "class_weight": SVC_CLASS_WEIGHT,
        "test_accuracy": float(test_accuracy),
        "test_f1_fake": float(test_f1),
        "test_precision_fake": float(test_precision),
        "test_recall_fake": float(test_recall),
        "val_accuracy": float(result["val_accuracy"]),
        "val_f1_fake": float(result["val_f1"]),
        "train_time": float(result["train_time"]),
    }

    txt_output = []
    txt_output.append("=" * 60)
    txt_output.append(f"{model_name} (ENHANCED BINARY + METADATA) - RESULTS REPORT")
    txt_output.append("=" * 60)
    txt_output.append(f"Model               : {model_name}")
    txt_output.append(f"Text max features   : {TEXT_MAX_FEATURES:,}")
    txt_output.append(f"Text n-gram range   : {TEXT_NGRAM_RANGE}")
    txt_output.append(f"Text min_df / max_df: {TEXT_MIN_DF} / {TEXT_MAX_DF}")
    txt_output.append(f"Text stop words     : {TEXT_STOP_WORDS}")
    txt_output.append(f"Title max features  : {TITLE_MAX_FEATURES:,}")
    txt_output.append(f"Title n-gram range  : {TITLE_NGRAM_RANGE}")
    txt_output.append(f"Title stop words    : {TITLE_STOP_WORDS}")
    txt_output.append(f"Regularization C    : {SVC_C}")
    txt_output.append(f"Class weight        : {SVC_CLASS_WEIGHT}")
    txt_output.append(f"Vectorizer          : TF-IDF (sublinear)")
    txt_output.append(f"Features            : processed_text + title (binary)")
    txt_output.append(f"Labels              : FAKE=0, RELIABLE=1")
    txt_output.append("")
    txt_output.append("Split          Accuracy     Precision(F)   Recall(F)      F1(F)")
    txt_output.append("-" * 62)
    txt_output.append(f"validation     {result['val_accuracy']:.4f}      {result['val_precision']:.4f}         {result['val_recall']:.4f}         {result['val_f1']:.4f}")
    txt_output.append(f"test           {test_accuracy:.4f}      {test_precision:.4f}         {test_recall:.4f}         {test_f1:.4f}")
    txt_output.append("")
    txt_output.append("=" * 60)
    txt_output.append("CLASSIFICATION REPORT (Test Set)")
    txt_output.append("=" * 60)
    txt_output.append(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))
    txt_output.append("")
    txt_output.append("CONFUSION MATRIX")
    txt_output.append("-" * 30)
    txt_output.append(str(cm_df))
    txt_output.append("")
    txt_output.append(f"Training time: {result['train_time']:.2f}s")
    txt_output.append("=" * 60)

    with open(REPORT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_output))

    with open(REPORT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(MODEL_DIR / "linearsvc_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump({"fake": 0, "reliable": 1}, f)

    with open(MODEL_DIR / "tfidf_vectorizers.pkl", "wb") as f:
        pickle.dump(vectorizers, f)

    print_log(f"\nResults saved to {REPORT_DIR}")
    print_log(f"Model saved to {MODEL_DIR}")

    return summary

# ========== MAIN ==========
def main():
    print_log("=" * 60)
    print_log("PART 3 - TF-IDF + LinearSVC (Binary + Metadata)")
    print_log(f"Sample size: {SAMPLE_SIZE:,}, Text: {TEXT_MAX_FEATURES:,}, Title: {TITLE_MAX_FEATURES:,}, C: {SVC_C}")
    print_log("=" * 60)

    setup()
    set_random_seed(RANDOM_SEED)

    # Load and prepare data
    df = load_metadata()
    df = map_to_binary(df)

    # Stratified sampling to limit memory usage
    sampled_df = stratified_sample(df, SAMPLE_SIZE, RANDOM_SEED)

    # Split data - stratify by binary label
    print_log("\nSplitting data...")
    train_val_df, test_df = train_test_split(
        sampled_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=sampled_df["binary_label"]
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=train_val_df["binary_label"]
    )

    print_log(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    print_log(f"Train distribution - FAKE: {(train_df['binary_label']==0).sum():,}, RELIABLE: {(train_df['binary_label']==1).sum():,}")

    vectorizers, X_train, X_val, X_test = extract_features(train_df, val_df, test_df)

    y_train = train_df["binary_label"].values
    y_val = val_df["binary_label"].values
    y_test = test_df["binary_label"].values

    result = train_and_evaluate(X_train, y_train, X_val, y_val)

    summary = save_results(result, X_test, y_test, vectorizers)

    print_log("\n" + "=" * 60)
    print_log("FINAL RESULTS - LinearSVC (Binary + Metadata)")
    print_log("=" * 60)
    print_log(f"Val Acc: {result['val_accuracy']:.4f}, Val F1 (fake): {result['val_f1']:.4f}")
    print_log(f"Test Acc: {summary['test_accuracy']:.4f}, Test F1 (fake): {summary['test_f1_fake']:.4f}")
    print_log("=" * 60)

if __name__ == "__main__":
    main()
