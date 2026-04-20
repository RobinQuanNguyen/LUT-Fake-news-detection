# Simple Baseline Models — Naive Bayes
**Branch:** `simple_model/naiveBayes`  
**Tasks covered:** Part 3 Task 0, Task 1, Task 2

---

## What we built and why

### Part 3 Task 0 — Label grouping
The FakeNewsCorpus has 12 classes. We grouped them into binary labels for fake news detection:

| Group | Labels |
|---|---|
| **Reliable (1)** | reliable |
| **Fake (0)** | fake, unreliable, conspiracy, rumor, junksci, clickbait, hate, satire |
| **Dropped** | political, bias, unknown |

**Why we dropped political, bias, unknown:**
These labels are too ambiguous — a politically biased article is not necessarily fake, and unknown has no ground truth. Including them would introduce noise into the model.

**Limitations of this grouping:**
- `satire` is intentionally fake but not meant to deceive — a human knows this, the model doesn't
- `clickbait` is misleading but not always factually wrong
- Grouping all fake variants together may dilute the signal since conspiracy and hate are very different types of misinformation

---

### Part 3 Task 1 — Text-only Naive Bayes baseline (`part3_t1_naivebayes.py`)

**Model:** Multinomial Naive Bayes  
**Vectorizer:** TF-IDF (Term Frequency — Inverse Document Frequency)  
**Features:** `processed_text` only (stemmed, stopwords removed from Part 2 Task 1 & 2)

**Why Naive Bayes?**
- Fast to train even on millions of rows
- Works well for text classification out of the box
- Simple and interpretable — good baseline to benchmark against
- MultinomialNB is specifically designed for word count/frequency data

**Why TF-IDF over raw word counts?**
- Raw counts treat "the" appearing 100 times as 100x more important than "conspiracy" appearing once
- TF-IDF downweights common words across all articles so rare but meaningful words carry more weight

**Why `pos_label=0` (fake class)?**
The goal is fake news detection. Missing a fake article (false negative) is more harmful than wrongly flagging a real one (false positive). So we report precision/recall/F1 from the perspective of the fake class.

**Key parameters:**
- `max_features=100,000` — top 100K most frequent words
- `sublinear_tf=True` — apply log to term frequencies to reduce impact of very common words
- `alpha=0.1` — Laplace smoothing, less than default (1.0) so model trusts training data more
- `random_state=42` — reproducibility

---

### Part 3 Task 2 — Naive Bayes with metadata (`part3_t1_naivebayes_meta.py`)

**Model:** Multinomial Naive Bayes (same as Task 1)  
**Extra features:** `domain`, `title`, `authors` appended to `processed_text`

**Why metadata?**
Fake news sites have very distinctive domain names (e.g. `endoftheamericandream.com`, `infowars.com`). By prepending domain/title/authors to the article text before TF-IDF, the model can learn these signals.

**Approach — Text concatenation:**
```
combined = domain + " " + title + " " + authors + " " + processed_text
```
Simple but effective as TF-IDF treats all words equally regardless of source.

**Supporting scripts:**
- `merge_metadata.py` — extracts domain/title/authors from raw 27GB file and merges with `processed_fakenews.csv`
- `part2_t3_meta.py` — splits `processed_fakenews_with_meta.csv` into train/val/test (same 80/10/10 strategy as `part1_t3.py`)

---

## How to run the scripts

### Prerequisites
```bash
pip install pandas scikit-learn numpy
```

### Part 3 Task 1 — Text-only baseline
```bash
python part3_t1_naivebayes.py
```
Input: `data/train.csv`, `data/validate.csv`, `data/test.csv`  
Output: `outputs/naive_bayes_report.txt`

### Merge metadata (run once)
```bash
python merge_metadata.py
```
Input: `data/news_cleaned_2018_02_13.csv` (27GB raw file), `data/processed_fakenews.csv`  
Output: `data/processed_fakenews_with_meta.csv`  
[IMPORTANT] Requires 27GB raw file — download from https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0

### Part 2 Task 3 — Split metadata dataset
```bash
python part2_t3_meta.py
```
Input: `data/processed_fakenews_with_meta.csv`  
Output: `data/train_meta.csv`, `data/validate_meta.csv`, `data/test_meta.csv`

### Part 3 Task 2 — Metadata baseline
```bash
python part3_t1_naivebayes_meta.py
```
Input: `data/train_meta.csv`, `data/validate_meta.csv`, `data/test_meta.csv`  
Output: `outputs/naive_bayes_meta_report.txt`

### Running on Google Colab

First upload csv files in your Google Drive. 
Update paths at the top of each script:
```python
TRAIN_PATH = Path("/content/drive/path-to-your-csv-files/train.csv")
# etc.
```
Replace with your actual paths to csv files in your Google Drive. 
And replace `if __name__ == "__main__": main()` with just `main()` at the bottom.

---

## Experiments and results

### Test set results (final evaluation)

| Model | Metadata | Training data | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Text only | none | 100% | 0.8882 | 0.8807 | 0.9363 | 0.9076 |
| Text + metadata | domain only | 50% | 0.9147 | 0.9006 | 0.9606 | 0.9296 |
| Text + metadata | domain+title+authors | 50% | 0.9155 | 0.8984 | 0.9650 | 0.9305 |

All metrics reported on the **fake class (pos_label=0)**.

### Key findings
1. **Metadata helps** — adding domain/title/authors improved F1 by +2.3% even with less training data
2. **Domain is the strongest signal** — domain alone (0.9296 F1) captures almost all the benefit of all three metadata fields (0.9305 F1)
3. **Validation ≈ test scores** — model generalises well, no overfitting
4. **More data helps but plateaus** — going from 30% to 50% training data gave small but consistent gains

### Confusion matrix (text-only, test set)
```
                 Pred FAKE    Pred RELIABLE
True FAKE        147,952          10,065
True RELIABLE     20,051          91,350
```
The model catches 93.6% of fake articles (high recall) at the cost of flagging ~20K reliable articles as fake.

---

## RAM and Colab issues and workarounds

### Problem
The dataset is very large (5.9M rows, processed CSV ~2-3GB). On 8GB RAM laptops and even 12GB Colab sessions, some steps crash.

### Solutions we used

| Problem | Solution |
|---|---|
| `drop_duplicates()` crashes on 8GB RAM | Switched to MD5 hash-based deduplication (see `part2_t3.py`) |
| Loading full train.csv crashes during TF-IDF | Used `del train_df` after fitting to free RAM before loading val/test |
| `combined_text` column doubles memory usage | Limit training data to 50% with `train_df.sample(frac=0.5)` |
| 27GB raw file merge crashes | Used chunked reading + write directly to disk (see `merge_metadata.py`) |
| Colab session disconnects mid-run | Save outputs directly to Google Drive so progress is preserved |

### Recommended setup
- **Minimum:** Google Colab free (12GB RAM) with files on Google Drive
- **Ideal:** Google Colab Pro (51GB RAM) or machine with 32GB+ RAM
- Always restart Colab runtime before running to maximise available RAM
- Close all other browser tabs before running

---

## Future improvements

1. **Use 100% training data** — with more RAM (32GB+), train on full dataset for better results
2. **Option B metadata approach** — use separate TF-IDF vectorizers for text and metadata, combine with `scipy.hstack` for cleaner feature separation
3. **Domain as categorical feature** — encode domain as a one-hot feature instead of text, avoiding it competing with article words in TF-IDF vocabulary
4. **Tune `alpha`** — try a range of Laplace smoothing values (0.01, 0.1, 0.5, 1.0) using the validation set to find optimal
5. **Try Logistic Regression** — often outperforms Naive Bayes on text classification, good next baseline to compare
6. **Authors feature cleaning** — the `authors` column is messy (inconsistent formatting). Cleaning it properly might improve its signal strength
