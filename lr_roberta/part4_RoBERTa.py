import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


MODEL_NAME = "roberta-base"
TEXT_COLUMN = "processed_text"
LABEL_SOURCE_COLUMN = "type"
RANDOM_SEED = 42
MAX_LENGTH = 128

# Stratified sampling config
USE_STRATIFIED_TRAIN_SAMPLE = True
TRAIN_SAMPLE_FRAC = 0.5   # half of training data

fake_labels = {
    "fake", "unreliable", "conspiracy", "rumor",
    "junksci", "clickbait", "hate", "satire"
}
reliable_labels = {"reliable"}


def print_model_summary():
    print("\n" + "=" * 68)
    print("FINAL MODEL SETUP")
    print("=" * 68)
    print("Model type          : Fine-tuned RoBERTa-base")
    print(f"Pretrained model    : {MODEL_NAME}")
    print(f"Feature source      : {TEXT_COLUMN} only")
    print("Classification task : Binary (reliable=0, fake=1)")
    print(f"Max sequence length : {MAX_LENGTH}")
    print("Optimizer settings  : learning_rate=2e-5, weight_decay=0.01")
    print("Training settings   : epochs=1, batch_size=8, fp16=True")
    print("Selection metric    : Validation F1-score (binary)")
    print(f"Random seed         : {RANDOM_SEED}")
    print(f"Stratified sampling : {USE_STRATIFIED_TRAIN_SAMPLE}")
    if USE_STRATIFIED_TRAIN_SAMPLE:
        print(f"Train sample frac   : {TRAIN_SAMPLE_FRAC}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_label(label):
    label = str(label).strip().lower()
    if label in fake_labels:
        return 1
    elif label in reliable_labels:
        return 0
    return None


def load_split(path):
    df = pd.read_csv(path)
    df["label"] = df[LABEL_SOURCE_COLUMN].apply(map_label)
    df = df.dropna(subset=["label", TEXT_COLUMN]).copy()
    df["label"] = df["label"].astype(int)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    print(f"\nLoaded {path}")
    print(f"Rows kept: {len(df)}")
    print("Class counts:")
    print(df["label"].value_counts().sort_index().rename(index={0: "reliable", 1: "fake"}))

    return df[[TEXT_COLUMN, "label"]]


def print_class_distribution(df, name):
    counts = df["label"].value_counts().sort_index()
    total = len(df)

    reliable_count = counts.get(0, 0)
    fake_count = counts.get(1, 0)

    print(f"\n{name} class distribution:")
    print(f"reliable: {reliable_count:,} ({reliable_count / total:.2%})")
    print(f"fake    : {fake_count:,} ({fake_count / total:.2%})")


def stratified_sample_df(df, label_col="label", frac=0.5, random_state=42):
    sampled_parts = []

    for _, group in df.groupby(label_col):
        sampled_group = group.sample(frac=frac, random_state=random_state)
        sampled_parts.append(sampled_group)

    sampled = pd.concat(sampled_parts, axis=0)
    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sampled


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def evaluate_split(trainer, dataset, name):
    preds_output = trainer.predict(dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)

    print(f"\n{name} results")
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
    set_seed(RANDOM_SEED)
    print_model_summary()

    train_df = load_split("data/no_metadata/train.csv")
    val_df = load_split("data/no_metadata/validate.csv")
    test_df = load_split("data/no_metadata/test.csv")

    print_class_distribution(train_df, "Original train")
    print_class_distribution(val_df, "Original validation")
    print_class_distribution(test_df, "Original test")

    if USE_STRATIFIED_TRAIN_SAMPLE:
        train_df = stratified_sample_df(
            train_df,
            label_col="label",
            frac=TRAIN_SAMPLE_FRAC,
            random_state=RANDOM_SEED
        )

    print("\nAfter sampling:")
    print(f"Train size     : {len(train_df):,}")
    print(f"Validation size: {len(val_df):,}")
    print(f"Test size      : {len(test_df):,}")

    print_class_distribution(train_df, "Sampled train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    train_ds = train_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)
    test_ds.set_format(type="torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./roberta_no_meta_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=RANDOM_SEED,
        report_to="none",
        fp16=True,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating best checkpoint...")
    evaluate_split(trainer, val_ds, "Validation")
    evaluate_split(trainer, test_ds, "Test")


if __name__ == "__main__":
    main()