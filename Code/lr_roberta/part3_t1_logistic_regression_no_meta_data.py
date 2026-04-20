from pathlib import Path
import random
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


TRAIN_PATH = Path("data/no_metadata/train.csv")
VAL_PATH = Path("data/no_metadata/validate.csv")
TEST_PATH = Path("data/no_metadata/test.csv")
LIAR_TEST_PATH = Path("data/evaluate/test.tsv")

MODEL_DIR = Path("models")
RESULT_DIR = Path("result")
PREPROCESSOR_CACHE_PATH = MODEL_DIR / "logreg_no_meta_preprocessor.joblib"
SKLEARN_MODEL_CACHE_PATH = MODEL_DIR / "logreg_no_meta_sklearn.joblib"
TORCH_MODEL_CACHE_PATH = MODEL_DIR / "logreg_no_meta_torch.pt"
RESULT_LOG_PATH = RESULT_DIR / "part3_logistic_regression_no_metadata_gpu_result.txt"

TEXT_COLUMN = "processed_text"
LABEL_COLUMN = "type"
RANDOM_SEED = 42
CANDIDATE_C = [0.01, 0.1, 1, 10]

USE_GPU_IF_AVAILABLE = True
GPU_BATCH_SIZE = 4096
GPU_EPOCHS = 1
GPU_LR = 0.05

LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

fake_labels = {
    "fake", "unreliable", "conspiracy", "rumor",
    "junksci", "clickbait", "hate", "satire",
}
reliable_labels = {"reliable"}


class TeeStdout:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_result_logging():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RESULT_LOG_PATH.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = TeeStdout(original_stdout, log_file)
    return original_stdout, log_file


def print_baseline_summary(device):
    print("\n" + "=" * 68)
    print("BASELINE MODEL SETUP")
    print("=" * 68)
    print("Model type          : TF-IDF + Logistic Regression")
    print("Feature source      : processed_text only")
    print("Vectorizer params   : lowercase=True, stop_words='english',")
    print("                      ngram_range=(1, 2), min_df=2, max_df=0.95")
    print(f"Hyperparameter grid : C in {CANDIDATE_C}")
    print("Selection metric    : Validation F1-score (binary)")
    if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
        print(f"Training backend    : PyTorch sparse logistic regression on {device}")
        print(
            f"GPU train params    : epochs={GPU_EPOCHS}, batch_size={GPU_BATCH_SIZE}, "
            f"lr={GPU_LR}"
        )
    else:
        print("Training backend    : scikit-learn LogisticRegression (CPU fallback)")
        print("Classifier params   : solver='liblinear', max_iter=1000, random_state=42")
    print(f"Random seed         : {RANDOM_SEED}")


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_label(label):
    label = str(label).strip().lower()
    if label in fake_labels:
        return 1
    if label in reliable_labels:
        return 0
    return None


def map_liar_label(label):
    label = str(label).strip().lower()

    liar_fake = {"pants-fire", "false", "barely-true"}
    liar_reliable = {"mostly-true", "true"}

    if label in liar_fake:
        return 1
    if label in liar_reliable:
        return 0
    return None


def load_no_meta_split(path):
    df = pd.read_csv(path)
    df["label"] = df[LABEL_COLUMN].apply(map_label)
    df = df.dropna(subset=["label", TEXT_COLUMN]).copy()
    df["label"] = df["label"].astype(int)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    print(f"\nLoaded {path}")
    print(f"Rows kept: {len(df):,}")
    print("Class counts:")
    print(df["label"].value_counts().sort_index().rename(index={0: "reliable", 1: "fake"}))

    return df[[TEXT_COLUMN, "label"]]


def load_liar_test(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = LIAR_COLUMNS
    df["label"] = df["label"].apply(map_liar_label)
    df = df.dropna(subset=["label", "statement"]).copy()
    df["label"] = df["label"].astype(int)

    liar_df = pd.DataFrame(
        {
            TEXT_COLUMN: df["statement"].astype(str),
            "label": df["label"],
        }
    )

    print(f"\nLoaded {path}")
    print(f"Rows kept after binary mapping: {len(liar_df):,}")
    print("Class counts:")
    print(liar_df["label"].value_counts().sort_index().rename(index={0: "reliable", 1: "fake"}))

    return liar_df


def build_preprocessor():
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )


def load_or_fit_preprocessor(X_train_text):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if PREPROCESSOR_CACHE_PATH.exists():
        print(f"\nLoading cached preprocessor from {PREPROCESSOR_CACHE_PATH}")
        return joblib.load(PREPROCESSOR_CACHE_PATH)

    print(f"\nFitting preprocessor and saving to {PREPROCESSOR_CACHE_PATH}")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_text)
    joblib.dump(preprocessor, PREPROCESSOR_CACHE_PATH)
    return preprocessor


class TorchSparseLogisticRegression(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(n_features, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x_sparse):
        return torch.sparse.mm(x_sparse, self.weight).squeeze(1) + self.bias


def csr_batch_to_torch_sparse(batch_csr, device):
    coo = batch_csr.tocoo()

    indices = torch.tensor(
        np.vstack((coo.row, coo.col)),
        dtype=torch.long,
        device=device,
    )
    values = torch.tensor(
        coo.data,
        dtype=torch.float32,
        device=device,
    )

    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        device=device,
    ).coalesce()


def predict_with_torch_model(model, X_csr, device, batch_size=GPU_BATCH_SIZE):
    preds = []
    model.eval()

    with torch.no_grad():
        for start in range(0, X_csr.shape[0], batch_size):
            stop = min(start + batch_size, X_csr.shape[0])
            batch_x = csr_batch_to_torch_sparse(X_csr[start:stop], device)
            logits = model(batch_x)
            batch_preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            preds.append(batch_preds)

    return np.concatenate(preds, axis=0)


def train_torch_model(X_train, y_train, X_val, y_val, device):
    if TORCH_MODEL_CACHE_PATH.exists():
        print(f"\nLoading cached GPU model from {TORCH_MODEL_CACHE_PATH}")
        checkpoint = torch.load(TORCH_MODEL_CACHE_PATH, map_location=device)
        model = TorchSparseLogisticRegression(checkpoint["n_features"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return {
            "backend": "torch",
            "model": model,
            "best_C": checkpoint["best_C"],
            "best_val_f1": checkpoint["best_val_f1"],
        }

    print(f"\nTraining GPU model and saving to {TORCH_MODEL_CACHE_PATH}")

    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    best_model = None
    best_c = None
    best_val_f1 = -1.0

    for c_value in CANDIDATE_C:
        model = TorchSparseLogisticRegression(X_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=GPU_LR,
            weight_decay=1.0 / max(c_value * X_train.shape[0], 1.0),
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(GPU_EPOCHS):
            for start in range(0, X_train.shape[0], GPU_BATCH_SIZE):
                stop = min(start + GPU_BATCH_SIZE, X_train.shape[0])

                batch_x = csr_batch_to_torch_sparse(X_train[start:stop], device)
                batch_y = torch.tensor(
                    y_train[start:stop],
                    dtype=torch.float32,
                    device=device,
                )

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        val_preds = predict_with_torch_model(model, X_val, device)
        val_f1 = f1_score(y_val, val_preds, average="binary", zero_division=0)
        print(f"Validation F1 for C={c_value}: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_c = c_value

    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "n_features": X_train.shape[1],
            "best_C": best_c,
            "best_val_f1": best_val_f1,
        },
        TORCH_MODEL_CACHE_PATH,
    )

    return {
        "backend": "torch",
        "model": best_model,
        "best_C": best_c,
        "best_val_f1": best_val_f1,
    }


def train_sklearn_model(X_train, y_train, X_val, y_val):
    if SKLEARN_MODEL_CACHE_PATH.exists():
        print(f"\nLoading cached CPU model from {SKLEARN_MODEL_CACHE_PATH}")
        return joblib.load(SKLEARN_MODEL_CACHE_PATH)

    print(f"\nTraining CPU model and saving to {SKLEARN_MODEL_CACHE_PATH}")

    best_model = None
    best_c = None
    best_val_f1 = -1.0

    for c_value in CANDIDATE_C:
        model = LogisticRegression(
            C=c_value,
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="binary", zero_division=0)
        print(f"Validation F1 for C={c_value}: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_c = c_value

    bundle = {
        "backend": "sklearn",
        "model": best_model,
        "best_C": best_c,
        "best_val_f1": best_val_f1,
    }
    joblib.dump(bundle, SKLEARN_MODEL_CACHE_PATH)
    return bundle


def evaluate(model_bundle, X_features, y_true, name, device):
    if model_bundle["backend"] == "torch":
        preds = predict_with_torch_model(model_bundle["model"], X_features, device)
    else:
        preds = model_bundle["model"].predict(X_features)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average="binary", zero_division=0)
    rec = recall_score(y_true, preds, average="binary", zero_division=0)
    f1 = f1_score(y_true, preds, average="binary", zero_division=0)

    print(f"\n{name} results")
    print("-" * 30)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, preds))
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            preds,
            target_names=["reliable", "fake"],
            zero_division=0,
        )
    )


def main():
    original_stdout, log_file = setup_result_logging()

    try:
        print(f"Writing run log to {RESULT_LOG_PATH}")

        set_seed(RANDOM_SEED)
        device = torch.device("cuda" if USE_GPU_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
        print_baseline_summary(device)

        train_df = load_no_meta_split(TRAIN_PATH)
        val_df = load_no_meta_split(VAL_PATH)
        test_df = load_no_meta_split(TEST_PATH)
        liar_df = load_liar_test(LIAR_TEST_PATH)

        X_train_text = train_df[TEXT_COLUMN]
        y_train = train_df["label"].to_numpy()

        X_val_text = val_df[TEXT_COLUMN]
        y_val = val_df["label"].to_numpy()

        X_test_text = test_df[TEXT_COLUMN]
        y_test = test_df["label"].to_numpy()

        X_liar_text = liar_df[TEXT_COLUMN]
        y_liar = liar_df["label"].to_numpy()

        preprocessor = load_or_fit_preprocessor(X_train_text)

        print("\nTransforming datasets into TF-IDF features...")
        X_train = preprocessor.transform(X_train_text)
        X_val = preprocessor.transform(X_val_text)
        X_test = preprocessor.transform(X_test_text)
        X_liar = preprocessor.transform(X_liar_text)

        if not sparse.isspmatrix_csr(X_train):
            X_train = X_train.tocsr()
        if not sparse.isspmatrix_csr(X_val):
            X_val = X_val.tocsr()
        if not sparse.isspmatrix_csr(X_test):
            X_test = X_test.tocsr()
        if not sparse.isspmatrix_csr(X_liar):
            X_liar = X_liar.tocsr()

        if device.type == "cuda":
            model_bundle = train_torch_model(X_train, y_train, X_val, y_val, device)
        else:
            model_bundle = train_sklearn_model(X_train, y_train, X_val, y_val)

        print(f"\nBest C selected on validation set: {model_bundle['best_C']}")
        print(f"Best validation F1: {model_bundle['best_val_f1']:.4f}")

        evaluate(model_bundle, X_val, y_val, "Validation", device)
        evaluate(model_bundle, X_test, y_test, "FakeNewsCorpus test", device)
        evaluate(model_bundle, X_liar, y_liar, "LIAR test", device)
    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()
