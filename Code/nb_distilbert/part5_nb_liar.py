'''
Part 5 Task 2 & 3 - Cross-Domain Evaluation on LIAR Dataset
Input: data/train.csv (FakeNewsCorpus) — to retrain vectorizer + model
       data/liar_test.tsv              — LIAR test set for cross-domain eval

Output: outputs/liar_evaluation_report.txt

Description:
We evaluate our Naive Bayes baseline (trained on FakeNewsCorpus) on the
LIAR dataset to measure cross-domain performance. The model is NOT
retrained on LIAR — this tests how well it generalizes to unseen domains.

Key differences between datasets:
- FakeNewsCorpus : long news articles, NLTK-preprocessed text
- LIAR           : short political statements, raw text
- Label scheme   : LIAR uses 6 labels mapped to binary (fake/reliable)

LIAR binary mapping:
  Reliable (1) : true, mostly-true, half-true
  Fake     (0) : barely-true, false, pants-fire
'''

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ── Paths ────────────────────────────────────────────────────
TRAIN_PATH  = Path("D:/LUT-Fake-news-detection/Code/data/train.csv")
LIAR_PATH   = Path("D:/LUT-Fake-news-detection/Code/data/liar_dataset/test.tsv")
OUTPUT_DIR  = Path("D:/LUT-Fake-news-detection/Code/outputs")
REPORT_PATH = OUTPUT_DIR / "liar_evaluation_report.txt"

# ── Columns ──────────────────────────────────────────────────
LABEL_COLUMN  = 'type'
TEXT_COLUMN   = 'processed_text'

# ── Label mappings ───────────────────────────────────────────
RELIABLE_LABELS = {"reliable"}
FAKE_LABELS     = {"fake", "unreliable", "conspiracy", "rumor",
                   "junksci", "clickbait", "hate", "satire"}
DROP_LABELS     = {"unknown", "political", "bias"}

LIAR_MAP = {
    'true'        : 1,
    'mostly-true' : 1,
    'half-true'   : 1,
    'barely-true' : 0,
    'false'       : 0,
    'pants-fire'  : 0,
}

# ── Model parameters (same as Part 3) ────────────────────────
TFIDF_MAX_FEATURES = 100_000
TFIDF_SUBLINEAR_TF = True
NB_ALPHA           = 0.1

LIAR_COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker',
    'job_title', 'state_info', 'party_affiliation',
    'barely_true_count', 'false_count', 'half_true_count',
    'mostly_true_count', 'pants_fire_count', 'context'
]

# ── Load & prepare FakeNewsCorpus training data ───────────────
def load_fakenews_train(path: Path) -> pd.DataFrame:
    print(f"Loading FakeNewsCorpus train from: {path}")
    df = pd.read_csv(path, usecols=[LABEL_COLUMN, TEXT_COLUMN], encoding="utf-8")
    df = df.dropna(subset=[LABEL_COLUMN, TEXT_COLUMN])
    df = df[~df[LABEL_COLUMN].isin(DROP_LABELS)]

    def map_label(label):
        if label in RELIABLE_LABELS: return 1
        elif label in FAKE_LABELS:   return 0
        else:                        return None

    df['binary_label'] = df[LABEL_COLUMN].map(map_label)
    df = df.dropna(subset=['binary_label'])
    df['binary_label'] = df['binary_label'].astype(int)

    print(f"Rows      : {len(df):,}")
    print(f"Reliable  : {(df['binary_label']==1).sum():,}")
    print(f"Fake      : {(df['binary_label']==0).sum():,}")
    return df

# ── Load & prepare LIAR test data ────────────────────────────
def load_liar(path: Path) -> pd.DataFrame:
    print(f"\nLoading LIAR test from: {path}")
    df = pd.read_csv(path, sep='\t', header=None, names=LIAR_COLUMNS)
    df['binary_label'] = df['label'].map(LIAR_MAP)
    df = df.dropna(subset=['binary_label'])
    df['binary_label'] = df['binary_label'].astype(int)

    print(f"Rows      : {len(df):,}")
    print(f"Reliable  : {(df['binary_label']==1).sum():,}")
    print(f"Fake      : {(df['binary_label']==0).sum():,}")
    return df

# ── Evaluate ─────────────────────────────────────────────────
def evaluate(model, vectorizer, texts, labels, split_name) -> dict:
    X      = vectorizer.transform(texts)
    y_true = labels
    y_pred = model.predict(X)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    print(f"\n--- {split_name} results ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=["Fake (0)", "Reliable (1)"],
                                zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix (rows=true, cols=predicted):")
    print(f"            Pred Fake  Pred Reliable")
    print(f"True Fake   {cm[0][0]:>9,}  {cm[0][1]:>12,}")
    print(f"True Rel.   {cm[1][0]:>9,}  {cm[1][1]:>12,}")

    return {"split": split_name, "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1}

# ── Save report ───────────────────────────────────────────────
def write_report(results: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 62,
        "NAIVE BAYES — LIAR CROSS-DOMAIN EVALUATION REPORT",
        "=" * 62,
        "",
        "Model              : Multinomial Naive Bayes",
        f"Alpha              : {NB_ALPHA}",
        f"Vectorizer         : TF-IDF (max {TFIDF_MAX_FEATURES:,} features)",
        "Trained on         : FakeNewsCorpus",
        "Evaluated on       : LIAR dataset (test split)",
        "",
        "LIAR label mapping :",
        "  Reliable (1)     : true, mostly-true, half-true",
        "  Fake     (0)     : barely-true, false, pants-fire",
        "",
        f"{'Split':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        "-" * 62,
    ]
    for r in results:
        lines.append(
            f"{r['split']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f}"
        )
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")
    print("\n" + report)

# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("PART 5 TASK 2 — Cross-Domain Evaluation on LIAR")
    print("=" * 62)

    # Step 1 — Train on FakeNewsCorpus
    train_df   = load_fakenews_train(TRAIN_PATH)
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF
    )
    X_train = vectorizer.fit_transform(train_df[TEXT_COLUMN])
    y_train = train_df['binary_label']
    print(f"\nVocabulary size : {len(vectorizer.vocabulary_):,}")

    model = MultinomialNB(alpha=NB_ALPHA)
    model.fit(X_train, y_train)
    print("Model trained!")

    # Step 2 — Evaluate on LIAR
    liar_df = load_liar(LIAR_PATH)
    results = []
    results.append(evaluate(
        model, vectorizer,
        liar_df['statement'],
        liar_df['binary_label'],
        "LIAR-test"
    ))

    write_report(results)
    print("\nDone!")

if __name__ == "__main__":
    main()