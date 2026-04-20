# Advanced Model — DistilBERT Fake News Classifier
**File:** `part4_distilbert.ipynb`  
**Tasks covered:** Part 4, Part 5 Task 1 & 2

---

## What we built and why

### Model choice — DistilBERT over alternatives

For the advanced model, we fine-tuned `distilbert-base-uncased` from HuggingFace on the `processed_text` field only. Several alternatives were considered:

| Model | Why not chosen |
|---|---|
| MLP | No understanding of word order or context |
| LSTM | Must learn language structure from scratch, slower to train |
| Full BERT | 2x larger than DistilBERT, not feasible on free-tier GPU |
| RoBERTa | Stronger but even heavier — teammate used this for comparison |
| DistilBERT (chosen)| 40% smaller, 60% faster than BERT, retains ~97% of performance |

Unlike TF-IDF models, DistilBERT uses **contextual embeddings** — the same word gets a different representation depending on surrounding words. This means it can detect subtle linguistic patterns in fake news that bag-of-words models miss entirely.

### Architecture
```
Input text (max 256 tokens)
        ↓
[DistilBERT Tokenizer]    — WordPiece subword tokenization
        ↓
[DistilBERT Base]         — 6 attention layers, 66M parameters
        ↓                    pre-trained on billions of words
[CLS token vector]        — 768-dimensional article representation
        ↓
[Pre-classifier layer]    — 768 -> 768 (Dense + ReLU)
        ↓
[Classifier layer]        — 768 -> 2 (fake / reliable)
        ↓
   Fake (0) / Reliable (1)
```

---

## Label Mapping
Same as Part 3 baseline models for consistency:

| Group | Labels |
|---|---|
| **Reliable (1)** | reliable |
| **Fake (0)** | fake, unreliable, conspiracy, rumor, junksci, clickbait, hate, satire |
| **Dropped** | political, bias, unknown |

---

## Training Parameters

| Parameter | Value | Reason |
|---|---|---|
| Base model | distilbert-base-uncased | Best speed/performance tradeoff |
| Epochs | 3 | Val F1 kept improving each epoch |
| Learning rate | 2e-5 | Standard for transformer fine-tuning |
| Weight decay | 0.01 | L2 regularization via AdamW |
| Batch size | 32 | Balance of GPU memory and speed |
| Max sequence length | 256 | Captures more context than 128 |
| Optimizer | AdamW | Standard for transformers |
| LR scheduler | Linear warmup (10% of steps) | Prevents instability early in training |
| Class weight — Fake | 0.852 | Corrects for class imbalance |
| Class weight — Reliable | 1.209 | Corrects for class imbalance |
| Training sample | 7% stratified (150,874 rows) | GPU time constraints |
| Val sample | 5% stratified (13,471 rows) | Validation was too slow on full 269k rows |
| Random seed | 42 | Reproducibility |

### Why class weights?
The training sample had a 59% fake / 41% reliable split. Without correction, the model would bias toward predicting fake. Class weights scale the loss function so errors on the minority class (reliable) are penalized more.

### Why 7% training sample?
Training DistilBERT on the full 2.1M rows would take ~40+ hours on a free T4 GPU. With 7% (~150k rows), each epoch takes ~65 minutes, making 3 epochs feasible within a 4-hour Colab session.

---

## How to Run

### Requirements
```
torch
transformers
pandas
scikit-learn
seaborn
matplotlib
```

Install with:
```bash
pip install torch transformers pandas scikit-learn seaborn matplotlib
```

### Google Colab setup (T4 GPU required)
1. Open `part4_distilbert.ipynb` in Google Colab
2. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
3. Mount Google Drive and update `BASE_PATH`:
```python
BASE_PATH = "/content/drive/MyDrive/your-folder-name/"
```
4. Run cells in order (Cell 1 → Cell 9)

### Resuming from checkpoint after disconnect
If the session disconnects mid-training, resume without retraining from scratch:
```python
# In Cell 8b, uncomment:
start_epoch, best_val_f1 = load_checkpoint(model, optimizer, scheduler)
print(f"Resuming from epoch {start_epoch}")
```
Then run Cell 8 — training continues from the saved epoch.

### Evaluation only (no retraining needed)
To just run the final test evaluation with saved weights:
1. Start new Colab session with T4 GPU
2. Run setup cells (Cell 1–4, Cell 6 for test_loader only)
3. Load saved weights:
```python
model.load_state_dict(torch.load(
    "/content/drive/MyDrive/LUT-FakeNewsDetection/best_distilbert.pt",
    map_location=device
))
model.eval()
```
4. Run Cell 9 (evaluation) — takes ~30 mins on full 269k test set

---

## Experiments and Results

### Training history
| Epoch | Train Loss | Val F1 | Time |
|---|---|---|---|
| 1 | 0.2089 | 0.9461 | ~1hr 49mins |
| 2 | 0.0928 | 0.9548 | ~1hr 8mins* |
| 3 | 0.0536 | 0.9575 ✅ best | ~1hr 8mins* |

*Epoch 1 was slow because validation ran on the full 269k val set (~70 mins). Epochs 2–3 used a 5% val sample (~13k rows), reducing val time to ~2 mins.

### FakeNewsCorpus Test Set (final)
| Metric | Value |
|---|---|
| Accuracy | 0.9648 |
| Precision | 0.9554 |
| Recall | 0.9596 |
| F1 | 0.9575 |

### LIAR Dataset (Cross-Domain)
| Metric | Value |
|---|---|
| Accuracy | 0.5107 |
| Precision | 0.5867 |
| Recall | 0.4454 |
| F1 | 0.5064 |

### Comparison with Naive Bayes baseline
| Metric | Naive Bayes | DistilBERT | Delta |
|---|---|---|---|
| Accuracy | 0.8882 | 0.9648 | +0.0766 ✔ |
| Precision | 0.8807 | 0.9554 | +0.0747 ✔ |
| Recall | 0.9363 | 0.9596 | +0.0233 ✔ |
| F1 | 0.9076 | 0.9575 | +0.0499 ✔ |

---

## GPU and Colab Issues & Workarounds

| Problem | Cause | Solution |
|---|---|---|
| Validation took 70 mins per epoch | Full 269k val set = 8,420 batches | Subsample val to 5% (~13k rows, 421 batches) |
| `torch.save` crashed with RuntimeError | Space in Drive folder name `LUT-Fake News Detection` | Save to `/content/` first, then copy to Drive with no-space path |
| Session cut off before epoch 1 finished | Free-tier GPU 4hr limit | Added checkpoint saving after every epoch to Drive |
| `save_checkpoint` NameError after crash | Function lost when cell crashed | Inline checkpoint save with `torch.save({...})` directly |
| `stratified_sample` KeyError with groupby | `include_groups=False` excluded label column | Switched to `train_test_split` with `stratify=` parameter |

### Recommended setup
- **Minimum:** Google Colab free tier (T4 GPU, 15.6GB VRAM, 12GB RAM)
- **Ideal:** Google Colab Pro (longer sessions, more RAM) or local machine with 16GB+ GPU
- Keep the Colab tab **open and visible** while training — idle sessions get disconnected
- Save checkpoints to Google Drive after every epoch — never rely on Colab local storage
- Run validation on a subsample during training — use the full test set only for final evaluation

---

## Key Findings

1. **Pre-trained knowledge transfers well** — even with only 7% of training data, DistilBERT achieved F1 0.9575, outperforming Naive Bayes trained on 100% of data
2. **Domain shift is a real problem** — F1 dropped from 0.9575 to 0.5064 on LIAR, showing the model learned FakeNewsCorpus-specific patterns rather than generalizable fake news signals
3. **Validation subsampling is safe** — using 5% of val set gave stable, reliable metrics while reducing validation time by 20x
4. **Loss decreased steadily** — train loss went 0.21 → 0.09 → 0.05 across 3 epochs with no signs of overfitting

---

## Future Improvements

1. **More training data** — training on 50-100% of data (with more GPU time) would likely push F1 above 0.97
2. **Longer sequences** — increasing max_length from 256 to 512 captures more article context at the cost of ~2x memory
3. **More epochs** — val F1 was still improving at epoch 3; 5 epochs might squeeze out more performance
4. **Domain adaptation for LIAR** — fine-tune on a small sample of LIAR training data to improve cross-domain performance
5. **Full BERT or RoBERTa** — with sufficient compute, these stronger models would likely outperform DistilBERT
6. **Ensemble** — combining DistilBERT predictions with Logistic Regression (which performed best overall) might give a stronger final classifier

## AI Usage

AI tool named Claude was used to properly structure this README and to improve overall readability. 