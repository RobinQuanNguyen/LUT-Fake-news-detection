import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# -------------------------
# Config
# -------------------------
CHECKPOINT_PATH = "./roberta_no_meta_output/checkpoint-12500"   # change if needed
LIAR_TEST_PATH = "data/evaluate/test.tsv"
MAX_LENGTH = 128
TEXT_COLUMN = "processed_text"


# -------------------------
# 1. Load LIAR test set
# -------------------------
columns = [
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


def map_liar_label(label):
    """
    Map LIAR 6-way labels to binary labels.
    fake = 1, reliable = 0

    We drop half-true because it is ambiguous in a binary setup.
    """
    label = str(label).strip().lower()

    fake_labels = {"pants-fire", "false", "barely-true"}
    reliable_labels = {"mostly-true", "true"}

    if label in fake_labels:
        return 1
    elif label in reliable_labels:
        return 0
    else:
        return None


def load_liar_test(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = columns

    df["label_binary"] = df["label"].apply(map_liar_label)
    df = df.dropna(subset=["label_binary", "statement"]).copy()
    df["label_binary"] = df["label_binary"].astype(int)
    df["statement"] = df["statement"].astype(str)

    eval_df = pd.DataFrame({
        "processed_text": df["statement"],
        "label": df["label_binary"]
    })

    print("\nLoaded LIAR test set")
    print(f"Rows kept after binary mapping: {len(eval_df)}")
    print("Class counts:")
    print(eval_df["label"].value_counts().sort_index().rename(index={0: "reliable", 1: "fake"}))

    return eval_df


# -------------------------
# 2. Tokenization
# -------------------------
def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# -------------------------
# 3. Evaluation
# -------------------------
def evaluate_model(model, tokenizer, df):
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    eval_args = TrainingArguments(
        output_dir="./liar_eval_output",
        per_device_eval_batch_size=16,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=eval_args)

    preds_output = trainer.predict(ds)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)

    print("\nLIAR Test Results")
    print("-" * 30)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(labels, preds))

    print("\nClassification report:")
    print(classification_report(
        labels,
        preds,
        target_names=["reliable", "fake"],
        zero_division=0
    ))


def main():
    print("Loading fine-tuned checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)

    print("Loading LIAR test data...")
    liar_test_df = load_liar_test(LIAR_TEST_PATH)

    print("Evaluating on LIAR...")
    evaluate_model(model, tokenizer, liar_test_df)


if __name__ == "__main__":
    main()