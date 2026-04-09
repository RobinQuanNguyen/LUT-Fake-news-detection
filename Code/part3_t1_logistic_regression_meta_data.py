import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def print_baseline_summary(candidate_C):
    print("\n" + "=" * 62)
    print("BASELINE MODEL SETUP")
    print("=" * 62)
    print("Model type          : TF-IDF + Logistic Regression")
    print("Feature sources     : processed_text + domain + title + authors")
    print("processed_text TFIDF: lowercase=True, stop_words='english',")
    print("                      ngram_range=(1, 2), min_df=2, max_df=0.95")
    print("domain TFIDF        : lowercase=True, ngram_range=(1, 1)")
    print("title TFIDF         : lowercase=True, stop_words='english',")
    print("                      ngram_range=(1, 2), min_df=2, max_df=0.95")
    print("authors TFIDF       : lowercase=True, stop_words='english',")
    print("                      ngram_range=(1, 2), min_df=2, max_df=0.95")
    print("Classifier params   : solver='liblinear', max_iter=1000, random_state=42")
    print(f"Hyperparameter grid : C in {candidate_C}")
    print("Selection metric    : Validation F1-score (binary)")


# -------------------------
# 1. Load data
# -------------------------
train_df = pd.read_csv("data/metadata/train_meta.csv")
val_df = pd.read_csv("data/metadata/validate_meta.csv")
test_df = pd.read_csv("data/metadata/test_meta.csv")


# -------------------------
# 2. Keep only the two classes you decided to use
# -------------------------
fake_labels = {
    "fake", "unreliable", "conspiracy", "rumor",
    "junksci", "clickbait", "hate", "satire"
}
reliable_labels = {"reliable"}


def map_label(label):
    # Convert original class names into binary labels (fake=1, reliable=0).
    label = str(label).strip().lower()
    if label in fake_labels:
        return 1
    elif label in reliable_labels:
        return 0
    else:
        return None


for df in [train_df, val_df, test_df]:
    df["label"] = df["type"].apply(map_label)

# Keep rows that have a valid mapped label
train_df = train_df.dropna(subset=["label"]).copy()
val_df = val_df.dropna(subset=["label"]).copy()
test_df = test_df.dropna(subset=["label"]).copy()

# Fill missing text/metadata fields with empty strings
text_columns = ["processed_text", "domain", "title", "authors"]
for df in [train_df, val_df, test_df]:
    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

# Feature matrix and labels
X_train = train_df[text_columns]
y_train = train_df["label"]

X_val = val_df[text_columns]
y_val = val_df["label"]

X_test = test_df[text_columns]
y_test = test_df["label"]


# -------------------------
# 3. Validation search for a simple baseline
# -------------------------
candidate_C = [0.01, 0.1, 1, 10]

print_baseline_summary(candidate_C)

best_model = None
best_C = None
best_val_f1 = -1

for C in candidate_C:
    model = Pipeline([
        ("features", ColumnTransformer(
            transformers=[
                (
                    "processed_text_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95
                    ),
                    "processed_text"
                ),
                (
                    "domain_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 1)
                    ),
                    "domain"
                ),
                (
                    "title_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95
                    ),
                    "title"
                ),
                (
                    "authors_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95
                    ),
                    "authors"
                )
            ],
            remainder="drop"
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="binary")

    print(f"Validation F1 for C={C}: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = model
        best_C = C

print(f"Best C selected on validation set: {best_C}")
print(f"Best validation F1: {best_val_f1:.4f}")


# -------------------------
# 4. Final evaluation on validation and test
# -------------------------
def evaluate(model, X, y, name):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average="binary", zero_division=0)
    rec = recall_score(y, preds, average="binary", zero_division=0)
    f1 = f1_score(y, preds, average="binary", zero_division=0)

    print(f"\n{name} results")
    print("-" * 30)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y,
        preds,
        target_names=["reliable", "fake"],
        zero_division=0
    ))


evaluate(best_model, X_val, y_val, "Validation")
evaluate(best_model, X_test, y_test, "Test")