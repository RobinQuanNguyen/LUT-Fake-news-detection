"""
PART 4 - DistilBERT Fine-tuning for Fake News Detection (Text + Title)
Uses pretrained DistilBERT with combined text and title input.
"""

import os
import sys
import json
import time
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# ========== CONFIGURATION ==========
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/part4_meta")
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

SAMPLE_SIZE = 10_000  # Smaller for GPU/memory constraints
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Model settings
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ========== DATA LOADING ==========
def load_data():
    print_log("Loading metadata...")
    start_time = time.time()
    
    train_path = DATA_DIR / "train_meta.csv"
    test_path = DATA_DIR / "test_meta.csv"
    validate_path = DATA_DIR / "validate_meta.csv"
    
    train_df = pd.read_csv(train_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")
    validate_df = pd.read_csv(validate_path, encoding="utf-8")
    
    combined_df = pd.concat([train_df, validate_df, test_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=["processed_text", "type"])
    
    for col in ["processed_text", "title"]:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str).fillna("")
    
    print_log(f"Combined: {len(combined_df):,} rows")
    print_log(f"Load time: {time.time() - start_time:.1f}s")
    
    return combined_df

def stratified_sample(df, n_samples, seed=42):
    print_log(f"Sampling {n_samples:,} rows (stratified)...")
    
    min_class_size = df["type"].value_counts().min()
    max_per_class = max(n_samples // len(df["type"].unique()), min_class_size)
    
    sampled_list = []
    for label in df["type"].unique():
        class_df = df[df["type"] == label]
        sample_n = min(len(class_df), max_per_class)
        class_sample = class_df.sample(n=sample_n, random_state=seed)
        sampled_list.append(class_sample)
    
    sampled_df = pd.concat(sampled_list, ignore_index=True)
    
    if len(sampled_df) > n_samples:
        sampled_df = sampled_df.sample(n=n_samples, random_state=seed)
    
    print_log(f"Sampled: {len(sampled_df):,} rows")
    return sampled_df

def prepare_data_for_bert(df, label_encoder):
    """Prepare data for BERT - combine title and text"""
    titles = df["title"].tolist()
    texts = df["processed_text"].tolist()
    
    # Combine title + [SEP] + text
    combined = []
    for title, text in zip(titles, texts):
        combined.append(f"{title} [SEP] {text}")
    
    labels = label_encoder.transform(df["type"].values)
    
    return combined, labels

# ========== TOKENIZATION ==========
def tokenize_function(examples, tokenizer, max_len):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_len
    )

# ========== TRAINING ==========
def train_model(train_texts, train_labels, val_texts, val_labels, num_labels):
    print_log(f"\nLoading DistilBERT tokenizer and model...")
    start_time = time.time()
    
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    
    print_log(f"Model loaded in {time.time() - start_time:.1f}s")
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    
    # Tokenize
    print_log("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LEN),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LEN),
        batched=True
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to="none",
        seed=RANDOM_SEED,
    )
    
    # Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted")
        }
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print_log(f"\nTraining DistilBERT on {len(train_texts):,} samples...")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    print_log(f"Training completed in {train_time:.1f}s")
    
    return model, tokenizer, train_time

def evaluate_model(model, tokenizer, test_texts, test_labels, label_encoder):
    print_log("\nEvaluating on test set...")
    
    # Tokenize test data
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LEN),
        batched=True
    )
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Predict
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    
    # Metrics
    accuracy = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="weighted")
    precision = precision_score(test_labels, preds, average="weighted")
    recall = recall_score(test_labels, preds, average="weighted")
    
    print_log(f"Test Accuracy: {accuracy:.4f}")
    print_log(f"Test F1: {f1:.4f}")
    print_log(f"Test Precision: {precision:.4f}")
    print_log(f"Test Recall: {recall:.4f}")
    
    class_names = label_encoder.classes_
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "predictions": preds,
        "test_labels": test_labels,
        "class_names": class_names
    }

def save_results(results, train_time, label_encoder):
    class_names = results["class_names"]
    model = results["model"]
    tokenizer = results["tokenizer"]
    test_labels = results.get("test_labels", results["predictions"])
    
    # Save model and tokenizer
    model_path = MODEL_DIR / "distilbert_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Classification report
    report = classification_report(
        test_labels, 
        results["predictions"],
        target_names=class_names,
        output_dict=True
    )
    
    # Save report CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_DIR / "classification_report.csv")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, results["predictions"])
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(REPORT_DIR / "confusion_matrix.csv")
    
    # Save results JSON
    summary = {
        "model": "DistilBERT",
        "features": "processed_text + title",
        "test_accuracy": float(results["accuracy"]),
        "test_f1": float(results["f1"]),
        "test_precision": float(results["precision"]),
        "test_recall": float(results["recall"]),
        "train_time": float(train_time),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN
    }
    
    with open(REPORT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save TXT - Baseline style
    txt_output = []
    txt_output.append("=" * 60)
    txt_output.append("DISTILBERT (WITH TITLE) - RESULTS REPORT")
    txt_output.append("=" * 60)
    txt_output.append(f"Model         : DistilBERT")
    txt_output.append(f"Max length    : {MAX_LEN}")
    txt_output.append(f"Batch size    : {BATCH_SIZE}")
    txt_output.append(f"Epochs        : {EPOCHS}")
    txt_output.append(f"Learning rate : {LEARNING_RATE}")
    txt_output.append(f"Features      : processed_text + title")
    txt_output.append("")
    txt_output.append(f"Test Accuracy : {results['accuracy']:.4f}")
    txt_output.append(f"Test F1       : {results['f1']:.4f}")
    txt_output.append(f"Test Precision: {results['precision']:.4f}")
    txt_output.append(f"Test Recall   : {results['recall']:.4f}")
    txt_output.append("")
    txt_output.append(f"Training time : {train_time:.2f}s")
    txt_output.append("=" * 60)
    
    with open(REPORT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_output))
    
    print_log(f"\nResults saved to {REPORT_DIR}")
    print_log(f"Model saved to {model_path}")

# ========== MAIN ==========
def main():
    print_log("=" * 60)
    print_log("PART 4 - DistilBERT Fine-tuning (Text + Title)")
    print_log("=" * 60)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Using device: {device}")
    if torch.cuda.is_available():
        print_log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    setup()
    set_random_seed(RANDOM_SEED)
    
    # Load and prepare data
    df = load_data()
    sampled_df = stratified_sample(df, SAMPLE_SIZE, RANDOM_SEED)
    
    # Split data
    train_val_df, test_df = train_test_split(
        sampled_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=sampled_df["type"]
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_val_df["type"]
    )
    
    print_log(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    # Prepare labels
    label_classes = sorted(sampled_df["type"].unique())
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(label_classes)
    
    print_log(f"Classes: {list(label_encoder.classes_)}")
    num_labels = len(label_encoder.classes_)
    
    # Prepare texts (title + text combined)
    train_texts, train_labels = prepare_data_for_bert(train_df, label_encoder)
    val_texts, val_labels = prepare_data_for_bert(val_df, label_encoder)
    test_texts, test_labels = prepare_data_for_bert(test_df, label_encoder)
    
    # Train model
    model, tokenizer, train_time = train_model(
        train_texts, train_labels, 
        val_texts, val_labels, 
        num_labels
    )
    
    # Evaluate
    results = evaluate_model(model, tokenizer, test_texts, test_labels, label_encoder)
    results["model"] = model
    results["tokenizer"] = tokenizer

    # Save results
    save_results(results, train_time, label_encoder)
    
    print_log("\n" + "=" * 60)
    print_log("FINAL RESULTS - DistilBERT (Text + Title)")
    print_log("=" * 60)
    print_log(f"Test F1: {results['f1']:.4f}, Accuracy: {results['accuracy']:.4f}")
    print_log("=" * 60)

if __name__ == "__main__":
    main()