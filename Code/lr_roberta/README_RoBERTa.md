# Advanced Model - RoBERTa Fake News Classifier
**Files:** `part4_RoBERTa.py`, `part4_evaluate_roberta_on_liar.py`  
**Tasks covered:** Part 4, LIAR cross-dataset evaluation

---

## What was built and why

### Model choice
The advanced model uses `roberta-base` fine-tuned on the `processed_text` field only. RoBERTa was chosen because it can model context, word order, and longer dependencies much better than TF-IDF baselines, while still being a standard and well-documented transformer model for text classification.

Compared with simpler baselines, RoBERTa does not rely on bag-of-words counts alone. Instead, it produces contextual embeddings, so the meaning of a word can change depending on the surrounding sentence. That makes it a reasonable advanced model for fake news detection, where phrasing patterns and context often matter.

---

### Label grouping
The same binary mapping as Part 3 was kept for consistency:

| Group | Labels |
|---|---|
| **Reliable (0)** | reliable |
| **Fake (1)** | fake, unreliable, conspiracy, rumor, junksci, clickbait, hate, satire |
| **Dropped** | political, bias, unknown |

This keeps the advanced-model evaluation directly comparable with the baseline models.

---

## Model configuration

**Training file:** `part4_RoBERTa.py`  
**LIAR evaluation file:** `part4_evaluate_roberta_on_liar.py`

### Training setup
- Pretrained model: `roberta-base`
- Input field: `processed_text`
- Max sequence length: `128`
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Train batch size: `8`
- Eval batch size: `8`
- Epochs: `1`
- Mixed precision: `fp16=True`
- Selection metric: validation F1-score
- Random seed: `42`

### Data usage
- Original training split: `2,155,342` rows
- Stratified train sampling enabled: `True`
- Training fraction used: `0.5`
- Effective training rows: `1,077,671`
- Validation rows: `269,417`
- Test rows: `269,418`

The 50% stratified sample was used to reduce training time and GPU cost while keeping the class distribution stable.

---

## How to run

### Important
Run these commands from the `Code` root folder, not from inside `lr_roberta`, because the scripts expect relative paths such as `data/` and `roberta_no_meta_output/`.

### Requirements
```bash
pip install pandas numpy torch datasets transformers scikit-learn accelerate
```

### Train and evaluate on FakeNewsCorpus
```bash
python lr_roberta/part4_RoBERTa.py
```

**Inputs**
- `data/no_metadata/train.csv`
- `data/no_metadata/validate.csv`
- `data/no_metadata/test.csv`

**Outputs**
- checkpoints under `roberta_no_meta_output/`
- console metrics for validation and FakeNewsCorpus test
- stored report copy: `lr_roberta/outputs/roberta_trained_with_half_database_report.txt`

### Evaluate the fine-tuned checkpoint on LIAR
```bash
python lr_roberta/part4_evaluate_roberta_on_liar.py
```

**Input**
- `data/evaluate/test.tsv`

**Important note**
The evaluation script currently points to:
```python
CHECKPOINT_PATH = "./roberta_no_meta_output/checkpoint-12500"
```
If the best checkpoint is stored under a different folder name, update `CHECKPOINT_PATH` before running.

---

## Results summary

### FakeNewsCorpus test set

| Metric | Value |
|---|---|
| Accuracy | 0.9513 |
| Precision | 0.9669 |
| Recall | 0.9495 |
| F1 | 0.9581 |

**Confusion matrix**
```text
[[106260   5141]
 [  7987 150030]]
```

### Validation set

| Metric | Value |
|---|---|
| Accuracy | 0.9515 |
| Precision | 0.9671 |
| Recall | 0.9497 |
| F1 | 0.9583 |

### LIAR test set

The saved LIAR evaluation used the same binary mapping on LIAR labels:
- fake: `pants-fire`, `false`, `barely-true`
- reliable: `mostly-true`, `true`
- dropped: `half-true`

**Reported LIAR confusion matrix**
```text
[[106 343]
 [109 444]]
```

This shows that performance drops noticeably when moving to a different dataset, which suggests domain shift between FakeNewsCorpus and LIAR.

---

## Main observations

1. RoBERTa is a stronger modeling approach than the simpler bag-of-words methods in terms of representation power, but it did not outperform the metadata-enhanced Logistic Regression result on FakeNewsCorpus.
2. The model still achieved solid in-domain performance with only half of the training split, which shows that pretrained language models can learn useful signals without using the full dataset.
3. The lower LIAR performance suggests that the model learned patterns that transfer only partially across datasets.

---

## Future improvements

1. Train for more than one epoch and compare checkpoints to see whether the model was still improving.
2. Increase `MAX_LENGTH` beyond `128` if GPU memory allows, so more article context can be preserved.
3. Evaluate the best checkpoint on LIAR automatically inside the main training workflow to avoid manual path updates.
4. Try a smaller transformer such as DistilBERT for a speed-performance tradeoff, or a stronger one with more compute if training time is available.
