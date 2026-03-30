import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def print_baseline_summary(candidate_C):
    # This prints the baseline setup so we can clearly report model settings.
    print("\n" + "=" * 62)
    print("BASELINE MODEL SETUP")
    print("=" * 62)
    print("Model type          : TF-IDF + Logistic Regression")
    print("Feature source      : processed_text only")
    print("Vectorizer params   : lowercase=True, stop_words='english',")
    print("                      ngram_range=(1, 2), min_df=2, max_df=0.95")
    print("Classifier params   : solver='liblinear', max_iter=1000, random_state=42")
    print(f"Hyperparameter grid : C in {candidate_C}")
    print("Selection metric    : Validation F1-score (binary)")


# -------------------------
# 1. Load data
# -------------------------
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validate.csv")
test_df = pd.read_csv("test.csv")

# -------------------------
# 2. Keep only the two classes you decided to use
# -------------------------
fake_labels = {
    "fake", "unreliable", "conspiracy", "rumor",
    "junksci", "clickbait", "hate", "satire"
}
reliable_labels = {"reliable"}

def map_label(label):
    # This converts original class names into binary labels (fake=1, reliable=0).
    label = str(label).strip().lower()
    if label in fake_labels:
        return 1   # fake
    elif label in reliable_labels:
        return 0   # reliable
    else:
        return None

for df in [train_df, val_df, test_df]:
    df["label"] = df["type"].apply(map_label)

train_df = train_df.dropna(subset=["label", "processed_text"]).copy()
val_df = val_df.dropna(subset=["label", "processed_text"]).copy()
test_df = test_df.dropna(subset=["label", "processed_text"]).copy()

train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

X_train, y_train = train_df["processed_text"], train_df["label"]
X_val, y_val = val_df["processed_text"], val_df["label"]
X_test, y_test = test_df["processed_text"], test_df["label"]

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
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
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
    # This computes and prints the main evaluation metrics for one dataset split.
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
    print(classification_report(y, preds, target_names=["reliable", "fake"], zero_division=0))

evaluate(best_model, X_val, y_val, "Validation")
evaluate(best_model, X_test, y_test, "Test")