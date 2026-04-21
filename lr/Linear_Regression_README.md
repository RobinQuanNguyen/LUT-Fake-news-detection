# Simple Baseline Models - SGDClassifier and LinearSVC Reports
**Files:** `part3_t1_linear_regression_no_meta_data.py`, `part3_t1_linear_regression_meta_data.py`, `part3_t1_liar_test.py`, `part3_no_meta.txt`, `part3_meta.txt`, `part3_with_liar.txt`  
**Tasks covered:** Part 3 text-only baseline, Part 3 metadata baseline, and cross-dataset evaluation on LIAR

---

## What was built and why

### Important naming note
The two Python filenames keep the phrase `linear_regression`, but the actual code inside both scripts trains **`SGDClassifier`** with sparse TF-IDF features and incremental `partial_fit` training. The uploaded `.txt` result reports for `part3_no_meta.txt` and `part3_meta.txt` are **LinearSVC** reports, so this README documents them separately instead of treating them as the same experiment.

---

### Label grouping
The main FakeNewsCorpus scripts use the same binary label mapping across experiments for consistency:

| Group | Labels |
|---|---|
| **Reliable (1)** | `reliable` |
| **Fake (0)** | `fake`, `unreliable`, `conspiracy`, `rumor`, `junksci`, `clickbait`, `hate`, `satire` |
| **Dropped** | `unknown`, `political`, `bias` |

These dropped labels were excluded because they are not cleanly binary in the same way as `reliable` versus fake-style categories.

For the LIAR test script, the LIAR labels are converted into a second binary mapping:

| Group | Labels |
|---|---|
| **Reliable (1)** | `true`, `mostly-true` |
| **Fake (0)** | `false`, `pants-fire`, `barely-true` |
| **Dropped** | `half-true` |

This makes the external LIAR evaluation comparable to the FakeNewsCorpus binary setup.

---

### Part 3 - Text-only full-dataset baseline (`part3_t1_linear_regression_no_meta_data.py`)

**Actual model in code:** `SGDClassifier`  
**Features:** TF-IDF from `processed_text` only  
**Training style:** memory-optimized incremental training with `partial_fit`

**Why this approach?**
- It is suitable for very large text datasets.
- It works well with sparse TF-IDF features.
- It is much lighter than transformer models.
- It can be trained in batches instead of loading everything into one dense training pipeline.

**Vectorizer settings:**
- `ngram_range=(1, 2)`
- `max_features=150000`
- `min_df=5`
- `max_df=0.9`
- `sublinear_tf=True`
- `stop_words="english"`
- `token_pattern=r"\b[a-zA-Z]{2,}\b"`

**Classifier settings:**
- `loss="hinge"`
- `alpha=1e-4`
- `class_weight="balanced"` computed from sampled training labels
- `random_state=42`
- `warm_start=True`
- `n_epochs=3`
- `batch_size=50000`

**Output location:**
- `outputs/part3_text/models/`
- `outputs/part3_text/reports/`

---

### Part 3 - Full-dataset baseline with metadata (`part3_t1_linear_regression_meta_data.py`)

**Actual model in code:** `SGDClassifier`  
**Features:** separate TF-IDF vectorizers for `processed_text` and `title`, then combined with sparse `hstack`

This version extends the text-only baseline by adding headline information. Instead of merging everything into one string, it keeps text and title as separate feature spaces and combines them only after vectorization.

**Text vectorizer settings:**
- `ngram_range=(1, 2)`
- `max_features=150000`
- `min_df=5`
- `max_df=0.9`
- `stop_words="english"`

**Title vectorizer settings:**
- `ngram_range=(1, 2)`
- `max_features=30000`
- `min_df=3`
- `stop_words="english"`

**Classifier settings:**
- `loss="hinge"`
- `alpha=1e-4`
- `random_state=42`
- `n_epochs=3`
- `batch_size=50000`

**Output location:**
- `outputs/part3_meta/models/`
- `outputs/part3_meta/reports/`

---

### Part 3 - Train on FakeNewsCorpus, test on LIAR (`part3_t1_liar_test.py`)

**Model:** `SGDClassifier`  
**Features:** TF-IDF from `processed_text`  
**Training data:** FakeNewsCorpus `train.csv` and `validate.csv`  
**Test data:** LIAR `test.tsv` if present, otherwise fallback to FakeNewsCorpus `test.csv`

This script is designed to measure **cross-dataset generalization** rather than only in-domain performance. It adds a text-cleaning step for the LIAR statements by lowercasing text, removing URLs, removing HTML tags, and normalizing whitespace before vectorization.

**Vectorizer settings:**
- `ngram_range=(1, 2)`
- `max_features=150000`
- `min_df=5`
- `max_df=0.9`
- `sublinear_tf=True`
- `stop_words="english"`

**Classifier settings:**
- `loss="hinge"`
- `alpha=1e-4`
- `batch_size=50000`
- `n_epochs=3`

**Important note:**
This script writes to the same base output folder as the text-only full-dataset script:
- `outputs/part3_text/models/`
- `outputs/part3_text/reports/`

That means rerunning one script after the other can overwrite earlier saved artifacts unless the files are renamed or moved.

---

### Uploaded report files
The uploaded `.txt` reports appear to document a **different linear baseline family** from the SGD scripts:

- `part3_no_meta.txt` reports **LinearSVC** with text-only TF-IDF features.
- `part3_meta.txt` reports **LinearSVC** with `processed_text + title` features.
- `part3_with_liar.txt` reports **SGDClassifier** evaluated on LIAR.

Because of that, the README keeps the code description and the uploaded reports clearly separated.

---

## How to run the scripts

### Important
Run the commands from the project root folder, because all scripts use relative paths such as `data/` and `outputs/`.

### Requirements
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### 1) Text-only full-dataset SGD baseline
```bash
python part3_t1_linear_regression_no_meta_data.py
```

**Inputs**
- `data/train.csv`
- `data/validate.csv`
- `data/test.csv`

**Saved artifacts**
- `outputs/part3_text/models/sgd_model.pkl`
- `outputs/part3_text/models/tfidf_vectorizer.pkl`
- `outputs/part3_text/reports/results.txt`
- `outputs/part3_text/reports/results_summary.json`
- `outputs/part3_text/reports/confusion_matrix.csv`
- `outputs/part3_text/reports/confusion_matrix.png`

### 2) Full-dataset SGD baseline with metadata
```bash
python part3_t1_linear_regression_meta_data.py
```

**Inputs**
- `data/train_meta.csv`
- `data/validate_meta.csv`
- `data/test_meta.csv`

**Saved artifacts**
- `outputs/part3_meta/models/sgd_model.pkl`
- `outputs/part3_meta/models/tfidf_vectorizers.pkl`
- `outputs/part3_meta/reports/results.txt`
- `outputs/part3_meta/reports/results_summary.json`
- `outputs/part3_meta/reports/confusion_matrix.csv`
- `outputs/part3_meta/reports/confusion_matrix.png`

### 3) Train on FakeNewsCorpus, test on LIAR
```bash
python part3_t1_liar_test.py
```

**Inputs**
- `data/train.csv`
- `data/validate.csv`
- `data/test.tsv` if testing on LIAR
- optional fallback: `data/test.csv`

**Saved artifacts**
- `outputs/part3_text/models/sgd_model.pkl`
- `outputs/part3_text/models/tfidf_vectorizer.pkl`
- `outputs/part3_text/reports/results.txt`
- `outputs/part3_text/reports/results_summary.json`
- `outputs/part3_text/reports/confusion_matrix.csv`
- `outputs/part3_text/reports/confusion_matrix.png`

### Notes on rerunning
- The current scripts save directly to fixed output paths.
- The two text-only scripts share `outputs/part3_text/`, so their reports can overwrite each other.
- If you want to keep both outputs, rename the report folders or move the saved files after each run.

---

## Results summary

### Uploaded report results
Only the following metrics are included here, because these are the result files that were actually uploaded.

#### FakeNewsCorpus test set

| Report file | Model | Metadata | Accuracy | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| `part3_no_meta.txt` | LinearSVC | No | 0.9563 | 0.9614 | 0.9642 | 0.9628 |
| `part3_meta.txt` | LinearSVC | Yes (`title`) | 0.9607 | 0.9641 | 0.9691 | 0.9666 |

#### LIAR test set

| Report file | Model | Train source | Test source | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| `part3_with_liar.txt` | SGDClassifier | FakeNewsCorpus | LIAR `test.tsv` | 0.5529 | 0.5782 | 0.7016 | 0.6340 |

### Main observations
1. Adding title metadata improved the uploaded LinearSVC result from **0.9563** accuracy and **0.9628** F1 to **0.9607** accuracy and **0.9666** F1 on the FakeNewsCorpus test split.
2. Cross-dataset performance on LIAR is much lower than the in-domain FakeNewsCorpus results, which suggests clear domain shift between the two datasets.
3. The SGD scripts are designed for scale and memory efficiency, while the uploaded LinearSVC reports show another strong sparse linear baseline.
4. The project currently mixes code and report naming conventions, so documenting the true model used in each file is important.

---

## Future improvements

1. Rename the Python files so their names match the real model used in code.
2. Separate output folders for full-dataset testing and LIAR testing to avoid overwriting reports.
3. Save full classification reports for all experiments, not only summary metrics.
4. Add command-line arguments for paths, batch size, epochs, and output folder names.
5. Expand metadata experiments beyond `title`, for example `domain`, `author`, or other source fields if available.
6. Evaluate the same feature settings across both SGDClassifier and LinearSVC for a cleaner comparison.
7. Add probability calibration or threshold analysis if later models need more stable confidence behavior.
