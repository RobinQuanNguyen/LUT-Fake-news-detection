'''
Part 2 Task 2 - Naive Bayes Baseline with Metadata Features
Approach: Append domain, title, authors to processed_text before TF-IDF
          This lets the model learn from both text and metadata signals.

Input:  data/train_meta.csv
        data/validate_meta.csv
        data/test_meta.csv

Output: outputs/naive_bayes_meta_report.txt

Label mapping:
    RELIABLE (1) : reliable
    FAKE     (0) : fake, unreliable, conspiracy, rumor,
                   junksci, clickbait, hate, satire
    DROPPED      : political, bias, unknown
'''

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np

# Paths
TRAIN_PATH    = Path("data/train_meta.csv")
VAL_PATH      = Path("data/validate_meta.csv")
TEST_PATH     = Path("data/test_meta.csv")
OUTPUT_DIR    = Path("outputs")
REPORT_PATH   = OUTPUT_DIR / "naive_bayes_meta_report.txt"


# Columns
LABEL_COLUMN  = "type"
TEXT_COLUMN   = "processed_text"
META_COLUMNS  = ["domain", "title", "authors"]

# Label mapping
RELIABLE_LABELS = {"reliable"}
FAKE_LABELS = {
    "fake", "unreliable", "conspiracy", "rumor",
    "junksci", "clickbait", "hate", "satire"
}
DROP_LABELS = {"political", "bias", "unknown"}

# TF-IDF parameters
TFIDF_MAX_FEATURES = 10_000
TFIDF_SUBLINEAR_TF = True

# Naive Bayes parameter
NB_ALPHA = 0.1


# ---------- Helpers functions -----------------

'''
Append metadata fields to the processed text.

Purpose: TF-IDF treats all words equally regardless of source.
By prepending domain/title/authors, the model can learn that
certain domains or title words are strong signals for fake vs reliable.
'''
def combine_text_and_meta(df: pd.DataFrame) -> pd.Series:
    
    meta_text = df[META_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)
    return meta_text + " " + df[TEXT_COLUMN].fillna("")

# Load a split CSV, map labels to binary, drop ambiguous rows.
def load_and_prepare(path: Path, split_name: str) -> pd.DataFrame:
    print(f"\nLoading {split_name} from: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"  Rows loaded       : {len(df):>9,}")

    df = df.dropna(subset=[LABEL_COLUMN, TEXT_COLUMN])

    # Drop ambiguous labels
    before = len(df)
    df = df[~df[LABEL_COLUMN].isin(DROP_LABELS)]
    dropped = before - len(df)
    if dropped:
        print(f"  Rows dropped      : {dropped:>9,}  (ambiguous labels)")

    # Map to binary
    def map_label(label):
        if label in RELIABLE_LABELS:
            return 1
        elif label in FAKE_LABELS:
            return 0
        return None

    df["binary_label"] = df[LABEL_COLUMN].map(map_label)
    df = df.dropna(subset=["binary_label"])
    df["binary_label"] = df["binary_label"].astype(int)

    # Combine text + metadata into one column
    df["combined_text"] = combine_text_and_meta(df)

    reliable_count = (df["binary_label"] == 1).sum()
    fake_count     = (df["binary_label"] == 0).sum()
    print(f"  Reliable (1)      : {reliable_count:>9,}  ({reliable_count/len(df)*100:.1f}%)")
    print(f"  Fake     (0)      : {fake_count:>9,}  ({fake_count/len(df)*100:.1f}%)")

    return df

# Vectorize combined text and compute metrics
def evaluate(model, vectorizer, df: pd.DataFrame, split_name: str) -> dict:
    X      = vectorizer.transform(df["combined_text"])
    y_true = df["binary_label"]
    y_pred = model.predict(X)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    print(f"\n  --- {split_name} results ---")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 score  : {f1:.4f}")
    print(f"\n  Full classification report:")
    print(classification_report(y_true, y_pred,
                                target_names=["fake (0)", "reliable (1)"],
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix (rows=true, cols=predicted):")
    print(f"              Pred Fake  Pred Reliable")
    print(f"  True Fake   {cm[0][0]:>9,}  {cm[0][1]:>12,}")
    print(f"  True Rel.   {cm[1][0]:>9,}  {cm[1][1]:>12,}")

    return {
        "split": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Save plain-text report for project report
def write_report(results: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 62,
        "NAIVE BAYES + METADATA — RESULTS REPORT",
        "=" * 62,
        "",
        "Model        : Multinomial Naive Bayes",
        f"Alpha        : {NB_ALPHA}  (Laplace smoothing)",
        f"Vectorizer   : TF-IDF",
        f"Max features : {TFIDF_MAX_FEATURES:,}",
        f"Sublinear TF : {TFIDF_SUBLINEAR_TF}",
        "",
        "Metadata approach : Option A (append domain/title/authors to text)",
        "Extra columns     : domain, title, authors",
        "",
        "Label mapping:",
        "  Reliable (1) : reliable",
        "  Fake     (0) : fake, unreliable, conspiracy, rumor,",
        "                 junksci, clickbait, hate, satire",
        "  Dropped      : political, bias, unknown",
        "",
        f"{'Split':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        "-" * 56,
    ]
    for r in results:
        lines.append(
            f"{r['split']:<12} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f}"
        )

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")
    print("\n" + report)

# ------------------------------------------------------

def main():
    print("=" * 62)
    print("PART 2 TASK 2 — Naive Bayes + Metadata Baseline")
    print("=" * 62)

    # Load and vectorize train only first
    train_df = load_and_prepare(TRAIN_PATH, "train")
    # 12GB RAM on GC so limit to 50% of training data
    train_df = train_df.sample(frac=0.5, random_state=42)

    print("\n[Vectorizing] Fitting TF-IDF on combined train data...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        min_df=5,
        max_df=0.95,
        dtype=np.float32
    )
    X_train = vectorizer.fit_transform(train_df["combined_text"])
    y_train = train_df["binary_label"]
    print(f"  Vocabulary size   : {len(vectorizer.vocabulary_):>9,}")
    print(f"  Training matrix   : {X_train.shape[0]:,} rows x {X_train.shape[1]:,} features")

    # Train
    print("\n[Training] Fitting Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=NB_ALPHA)
    model.fit(X_train, y_train)
    print("  Successfully trained!")

    # Free train data from RAM before loading val/test
    del train_df, X_train, y_train

    # Evaluate on val
    print("\n[Evaluation]")
    results = []
    val_df = load_and_prepare(VAL_PATH, "validation")
    results.append(evaluate(model, vectorizer, val_df, "validation"))
    del val_df  # free RAM before loading test

    # Evaluate on test
    test_df = load_and_prepare(TEST_PATH, "test")
    results.append(evaluate(model, vectorizer, test_df, "test"))
    del test_df

    write_report(results)
    print("\nOMGG it didn't crash this time T^T!")

    

    results_table = pd.DataFrame([
        {
            "Model": "Text only",
            "Metadata": "none",
            "Training data": "100%",
            "Accuracy": 0.8882,
            "Precision": 0.8807,
            "Recall": 0.9363,
            "F1": 0.9076,
        },
        {
            "Model": "Text + metadata",
            "Metadata": "domain only",
            "Training data": "50%",
            "Accuracy": 0.9147,
            "Precision": 0.9006,
            "Recall": 0.9606,
            "F1": 0.9296,
        },
        {
            "Model": "Text + metadata",
            "Metadata": "domain+title+authors",
            "Training data": "50%",
            "Accuracy": 0.9148,
            "Precision": 0.8976,
            "Recall": 0.9649,
            "F1": 0.9300,
        },
    ])

    print(results_table.to_string(index=False))


if __name__ == "__main__":
    main()