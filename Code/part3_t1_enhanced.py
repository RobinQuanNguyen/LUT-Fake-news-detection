"""
PART 3 - TF-IDF + LinearSVC (Enhanced for Text Only - Binary Classification)
Tối ưu feature engineering để cải thiện accuracy với linear model.
Binary classification: FAKE (0) vs RELIABLE (1)
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
import pickle
import random

# ========== CONFIGURATION ==========
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/part3_text")
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
TEXT_MAX_FEATURES = 100000  # Giảm từ 150k để tránh tràn bộ nhớ
TEXT_MIN_DF = 3
TEXT_MAX_DF = 0.9
SUBLINEAR_TF = True
STOP_WORDS = 'english'

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
def load_data():
    print_log("Loading data...")
    start_time = time.time()

    train_path = DATA_DIR / "train.csv"
    validate_path = DATA_DIR / "validate.csv"
    test_path = DATA_DIR / "test.csv"
    combined_path = DATA_DIR / "processed_fakenews.csv"

    if train_path.exists() and validate_path.exists() and test_path.exists():
        print_log("Loading from split files...")
        train_df = pd.read_csv(train_path, encoding="utf-8")
        validate_df = pd.read_csv(validate_path, encoding="utf-8")
        test_df = pd.read_csv(test_path, encoding="utf-8")
        combined_df = pd.concat([train_df, validate_df, test_df], ignore_index=True)
    elif combined_path.exists():
        print_log("Loading from combined CSV...")
        combined_df = pd.read_csv(combined_path, encoding="utf-8")
    else:
        raise FileNotFoundError("No data file found!")

    # Keep only needed columns
    if "processed_text" in combined_df.columns and "type" in combined_df.columns:
        combined_df = combined_df[["id", "processed_text", "type"]].copy()

    combined_df = combined_df.dropna(subset=["processed_text", "type"])
    combined_df["processed_text"] = combined_df["processed_text"].astype(str)

    print_log(f"Total loaded: {len(combined_df):,} rows")
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
        smooth_idf=True
    )

    train_texts = train_df["processed_text"].values
    val_texts = val_df["processed_text"].values
    test_texts = test_df["processed_text"].values

    print_log("  Fitting TF-IDF vectorizer on training data...")
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    vocab_size = len(vectorizer.vocabulary_)
    print_log(f"Feature shape: {X_train.shape}")
    print_log(f"Vocabulary size: {vocab_size:,}")
    print_log(f"Feature extraction time: {time.time() - start_time:.1f}s")

    return vectorizer, X_train, X_val, X_test

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
def save_results(result, X_test, y_test, vectorizer):
    model = result["model"]
    model_name = "LinearSVC"

    print_log(f"\n{'='*60}")
    print_log(f"TEST RESULTS - {model_name} (Binary Classification)")
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
        "features": "processed_text (TF-IDF Enhanced - Binary)",
        "ngram_range": list(TEXT_NGRAM_RANGE),
        "max_features": TEXT_MAX_FEATURES,
        "min_df": TEXT_MIN_DF,
        "max_df": TEXT_MAX_DF,
        "stop_words": STOP_WORDS,
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
    txt_output.append(f"{model_name} (ENHANCED BINARY) - RESULTS REPORT")
    txt_output.append("=" * 60)
    txt_output.append(f"Model               : {model_name}")
    txt_output.append(f"Max features       : {TEXT_MAX_FEATURES:,}")
    txt_output.append(f"N-gram range       : {TEXT_NGRAM_RANGE}")
    txt_output.append(f"min_df / max_df    : {TEXT_MIN_DF} / {TEXT_MAX_DF}")
    txt_output.append(f"Stop words         : {STOP_WORDS}")
    txt_output.append(f"Regularization C   : {SVC_C}")
    txt_output.append(f"Class weight       : {SVC_CLASS_WEIGHT}")
    txt_output.append(f"Vectorizer         : TF-IDF (sublinear)")
    txt_output.append(f"Features           : processed_text (binary)")
    txt_output.append(f"Labels             : FAKE=0, RELIABLE=1")
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

    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print_log(f"\nResults saved to {REPORT_DIR}")
    print_log(f"Model saved to {MODEL_DIR}")

    return summary

# ========== MAIN ==========
def main():
    print_log("=" * 60)
    print_log("PART 3 - TF-IDF + LinearSVC (Binary Classification)")
    print_log(f"Sample size: {SAMPLE_SIZE:,}, Features: {TEXT_MAX_FEATURES:,}, N-gram: {TEXT_NGRAM_RANGE}, C: {SVC_C}")
    print_log("=" * 60)

    setup()
    set_random_seed(RANDOM_SEED)

    # Load and prepare data
    df = load_data()
    df = map_to_binary(df)

    # Stratified sampling to limit memory usage
    sampled_df = stratified_sample(df, SAMPLE_SIZE, RANDOM_SEED)

    # Split data - stratify by binary label to maintain class balance
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

    vectorizer, X_train, X_val, X_test = extract_features(train_df, val_df, test_df)

    y_train = train_df["binary_label"].values
    y_val = val_df["binary_label"].values
    y_test = test_df["binary_label"].values

    result = train_and_evaluate(X_train, y_train, X_val, y_val)

    summary = save_results(result, X_test, y_test, vectorizer)

    print_log("\n" + "=" * 60)
    print_log("FINAL RESULTS - LinearSVC (Binary)")
    print_log("=" * 60)
    print_log(f"Val Acc: {result['val_accuracy']:.4f}, Val F1 (fake): {result['val_f1']:.4f}")
    print_log(f"Test Acc: {summary['test_accuracy']:.4f}, Test F1 (fake): {summary['test_f1_fake']:.4f}")
    print_log("=" * 60)

if __name__ == "__main__":
    main()
