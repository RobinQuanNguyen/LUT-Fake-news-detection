# Simple Baseline Models - Logistic Regression
**Files:** `part3_t1_logistic_regression_no_meta_data.py`, `part3_t1_logistic_regression_meta_data.py`  
**Tasks covered:** Part 3 Task 0, Task 1, Task 2

---

## What was built and why

### Label grouping
The same binary label mapping was used across all experiments for consistency:

| Group | Labels |
|---|---|
| **Reliable (0)** | reliable |
| **Fake (1)** | fake, unreliable, conspiracy, rumor, junksci, clickbait, hate, satire |
| **Dropped** | political, bias, unknown |

The dropped labels were left out because they are not cleanly fake or reliable. An extra filtering round for labels such as `political` and `bias` was originally considered, but that would require manual rules or additional annotation and would add more complexity than was practical for this task.

---

### Part 3 Task 1 - Text-only baseline (`part3_t1_logistic_regression_no_meta_data.py`)

**Model:** Logistic Regression  
**Features:** TF-IDF vectors from `processed_text` only

**Why Logistic Regression?**
- It is a standard and strong text-classification baseline.
- It handles high-dimensional sparse TF-IDF features well.
- It trains much faster than transformer models while still giving interpretable and competitive results.

**Vectorizer settings:**
- `lowercase=True`
- `stop_words="english"`
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.95`

**Classifier selection:**
- Candidate regularization values: `C in [0.01, 0.1, 1, 10]`
- Model selection metric: validation F1-score
- Random seed: `42`

**Training backend:**
- Preferred path: PyTorch sparse logistic regression on CUDA
- Fallback path: scikit-learn `LogisticRegression` on CPU with `solver="liblinear"` and `max_iter=1000`

**GPU settings:**
- `epochs=1`
- `batch_size=4096`
- `lr=0.05`

The script also evaluates on the LIAR dataset after the FakeNewsCorpus test split to measure cross-dataset generalization.

---

### Part 3 Task 2 - Logistic Regression with metadata (`part3_t1_logistic_regression_meta_data.py`)

**Model:** Logistic Regression  
**Features:** separate TF-IDF representations for:
- `processed_text`
- `domain`
- `title`
- `authors`

Instead of concatenating metadata into one long string, this version uses a `ColumnTransformer` so each field can keep its own vectorizer settings. This is a reasonable extension because fake-news signals often appear not only in article content, but also in source domains, headline wording, and author fields.

**Vectorizer settings by field:**
- `processed_text`: `lowercase=True`, `stop_words="english"`, `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`
- `domain`: `lowercase=True`, `ngram_range=(1, 1)`
- `title`: `lowercase=True`, `stop_words="english"`, `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`
- `authors`: `lowercase=True`, `stop_words="english"`, `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`

**Classifier selection:**
- Candidate regularization values: `C in [0.01, 0.1, 1, 10]`
- Model selection metric: validation F1-score
- Random seed: `42`

**Training backend:**
- Preferred path: PyTorch sparse logistic regression on CUDA
- Fallback path: scikit-learn `LogisticRegression` on CPU with `solver="liblinear"` and `max_iter=1000`

**GPU settings:**
- `epochs=1`
- `batch_size=4096`
- `lr=0.05`

---

## How to run the scripts

### Important
Run the commands from the `Code` root folder, not from inside `lr_roberta`, because the scripts use relative paths such as `data/`, `models/`, and `result/`.

### Requirements
```bash
pip install pandas numpy scipy scikit-learn joblib torch
```

### Text-only Logistic Regression
```bash
python lr_roberta/part3_t1_logistic_regression_no_meta_data.py
```

**Inputs**
- `data/no_metadata/train.csv`
- `data/no_metadata/validate.csv`
- `data/no_metadata/test.csv`
- `data/evaluate/test.tsv`

**Saved artifacts**
- `models/logreg_no_meta_preprocessor.joblib`
- `models/logreg_no_meta_torch.pt` or `models/logreg_no_meta_sklearn.joblib`
- `result/part3_logistic_regression_no_metadata_gpu_result.txt`
- `lr_roberta/outputs/logistic_regression_no_metadata_gpu_report.txt` if a copy is kept there

### Logistic Regression with metadata
```bash
python lr_roberta/part3_t1_logistic_regression_meta_data.py
```

**Inputs**
- `data/metadata/train_meta.csv`
- `data/metadata/validate_meta.csv`
- `data/metadata/test_meta.csv`
- `data/evaluate/test.tsv`

**Saved artifacts**
- `models/logreg_metadata_preprocessor.joblib`
- `models/logreg_metadata_torch.pt` or `models/logreg_metadata_sklearn.joblib`
- console report in `lr_roberta/outputs/logistic_regrerssion_metadata_report.txt` if output is redirected or copied there

### Notes on rerunning
- If cached preprocessors or trained models already exist in `models/`, the scripts load them instead of retraining.
- To force a fresh run, delete the related cache files in `models/`.
- CUDA is used automatically if `torch.cuda.is_available()` returns `True`.

---

## Results summary

### FakeNewsCorpus test set

| Model | Metadata | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | No | 0.9764 | 0.9782 | 0.9816 | 0.9799 |
| Logistic Regression | Yes | 0.9988 | 0.9991 | 0.9988 | 0.9990 |

### LIAR test set

| Model | Metadata | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | No | 0.5419 | 0.5579 | 0.8192 | 0.6637 |
| Logistic Regression | Yes | 0.5659 | 0.5787 | 0.7848 | 0.6662 |

### Main observations
1. Metadata gives a very large boost on FakeNewsCorpus, which suggests that source-related fields contain strong dataset-specific signals.
2. The gain on LIAR is much smaller, which indicates that metadata helps less when the evaluation data comes from a different distribution.
3. Even the text-only version remains a strong baseline, especially considering its much lower training cost than transformer models.

---

## Future improvements

1. Tune the GPU training schedule with more than one epoch to check whether the sparse PyTorch version can improve further.
2. Add probability calibration because very confident predictions can hide domain-shift problems.
3. Compare this setup with L2-only scikit-learn Logistic Regression on the same GPU-selected `C` values for a direct implementation check.
4. Test whether keeping `domain` separate but dropping `authors` improves cross-dataset generalization.
