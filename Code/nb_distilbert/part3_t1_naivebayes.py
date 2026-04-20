'''
Part 3 Task 0 & 1 - Naive Bayes Baseline Classifier
Input: data/train.csv, data/validate.csv, data/test.csv
Output: outputs/naive_bayes_report.txt

Label mapping:
RELIABLE (1) : reliable
FAKE (0) : fake, unreliable, conspiracy, rumor, junksci, clickbait, hate, satire
DROPPED : political, bias, unknown (due to ambiguity)

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
    confusion_matrix
)

# Paths
TRAIN_PATH = Path("data/train.csv")
VAL_PATH = Path("data/validate.csv")
TEST_PATH = Path("data/test.csv")
OUTPUT_DIR = Path("outputs")
REPORT_PATH = OUTPUT_DIR/"naive_bayes_report.txt"

# Columns
LABEL_COLUMN = 'type'
TEXT_COLUMN = 'processed_text'

# Label mapping
# We frame this as binary classification: reliable (1) vs fake (0)
# political, bias, unknown are dropped as they are too ambiguous to label confidently
# Will only cause noise

RELIABLE_LABELS = {"reliable"}

FAKE_LABELS = {
    "fake", "unreliable", "conspiracy", "rumor", "junksci", "clickbait", "hate", "satire"
}

# Labels to drop
DROP_LABELS = {"unknown", "political", "bias"}

# TF-IDF parameters
'''
max_features: only keep the 100,000 most frequent words
Purpose: keeps memory manageable while retaining enough vocabulary

sublinear_tf: apply log to term frequencies
Purpose: reduces the impact of very common words within a document
E.g. a word appearing 100 times shouldn't be 100x more important than one appearing once

'''

TFIDF_MAX_FEATURES = 100_000
TFIDF_SUBLINEAR_TF = True

# Naive Bayes parameter
'''
alpha: Laplace smoothing, prevents 0 probability for unseen words
alpha = 1.0 is the standard default
We use 0.1 which is less smoothing than the default 1.0
Makes the model trust training data more
'''
NB_ALPHA = 0.1

# --------------- Helpers ---------------------------

#Load a split CSV, map labels to binary (0 or 1), drop ambiguous rows
def load_prepare(path: Path, split_name:str) -> pd.DataFrame:
    print(f"\nLoading {split_name} from: {path}")
    df = pd.read_csv(path, usecols=[LABEL_COLUMN, TEXT_COLUMN], encoding="utf-8")
    print(f"Rows loaded : {len(df):>9,}")

    # Drop rows with missing values
    df = df.dropna(subset=[LABEL_COLUMN, TEXT_COLUMN])

    # Drop ambiguous labels (political, bias, unknown)
    before = len(df)
    df = df[~df[LABEL_COLUMN].isin(DROP_LABELS)]
    dropped = before - len(df)
    if dropped:
        print(f"Rows dropped : {dropped:>9,}  (ambiguous labels)")

    # Map to binary
    # reliable = 1,  all fake variants = 0
    def map_label(label):
        if label in RELIABLE_LABELS:
            return 1
        elif label in FAKE_LABELS:
            return 0
        else:
            return None  # safety catch for unexpected labels
        
    df["binary_label"] = df[LABEL_COLUMN].map(map_label)
    df = df.dropna(subset=["binary_label"])
    df["binary_label"] = df["binary_label"].astype(int)

    count_reliable = (df["binary_label"] == 1).sum()
    count_fake = (df["binary_label"] == 0).sum()
    print(f"Reliable (1) : {count_reliable:>9,}  ({count_reliable/len(df)*100:.1f}%)")
    print(f"Fake (0)     : {count_fake:>9,}  ({count_fake/len(df)*100:.1f}%)")
    return df

# Vectorize text & compute accuracy, precision, recall, F1
# Returns a dict (dictionary) of metrics for the report
def evaluate(model, vectorizer, df: pd.DataFrame, split_name:str) -> dict:
    x = vectorizer.transform(df[TEXT_COLUMN])
    y_true = df["binary_label"] # real ans from the dataset
    y_pred = model.predict(x)   # model's guesses

    # pos_label tells sklearn which class we care more about
    # In our case, fake class (0)
    # zero_division prevents crashing when model never predicts 0 & denominator becomes 0
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    print(f"\n  --- {split_name} results ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 score  : {f1:.4f}")
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

# Save a plain-text report summarising all results.
def write_report(results: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 62,
        "NAIVE BAYES BASELINE - RESULTS REPORT",
        "=" * 62,
        "",
        "Model         : Multinomial Naive Bayes",
        f"Alpha        : {NB_ALPHA}  (Laplace smoothing)",
        f"Vectorizer   : TF-IDF",
        f"Max features : {TFIDF_MAX_FEATURES:,}",
        f"Sublinear TF : {TFIDF_SUBLINEAR_TF}",
        "",
        "Label mapping  :",
        "Reliable (1)   : reliable",
        "Fake     (0)   : fake, unreliable, conspiracy, rumor,",
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
    print("PART 3 TASK 1 — Naive Bayes Baseline")
    print("=" * 62)
 
    # Load all three splits
    train_df = load_prepare(TRAIN_PATH, "train")
    val_df = load_prepare(VAL_PATH, "validation")
    test_df = load_prepare(TEST_PATH, "test")
 
    # TF-IDF vectorization
    # fit_transform on train only
    # fitting on test would be data leakage
    print("\n[Vectorizing] Fitting TF-IDF on training data...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )
    X_train = vectorizer.fit_transform(train_df[TEXT_COLUMN])
    y_train = train_df["binary_label"]
    print(f"Vocabulary size : {len(vectorizer.vocabulary_):>9,}")
    print(f"Training matrix : {X_train.shape[0]:,} rows x {X_train.shape[1]:,} features")
 
    # Train Naive Bayes
    print("\n[Training] Fitting Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=NB_ALPHA)
    model.fit(X_train, y_train)
    print("Successfully trained!")
 
    # Evaluate on validation set
    print("\n[Evaluation]")
    results = []
    results.append(evaluate(model, vectorizer, val_df, "validation"))
 
    # Final evaluation on test set
    results.append(evaluate(model, vectorizer, test_df, "test"))
 
    # Save report
    write_report(results)
 
    print("\nAll done!")
 
 
if __name__ == "__main__":
    main()
 