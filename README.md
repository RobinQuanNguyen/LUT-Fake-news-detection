# Fake News Detection Project

This project builds a fake news classifier using the FakeNewsCorpus dataset. The work is organized in stages: data cleaning and preprocessing, train/validation/test splitting, simple baseline models, and more advanced transformer-based models. The main goal is to compare lightweight baselines against stronger deep-learning models on the same binary fake vs reliable classification task.

## Project structure

### Data processing and preparation
- `part2_task1_preprocessing.py`
  Cleans and processes the raw FakeNewsCorpus data, including the text preparation used later by the models.
- `part2_task2_eda.py`
  Runs exploratory data analysis to inspect label distribution, text properties, and dataset characteristics.
- `part2_task3_splitting.py`
  Creates the train, validation, and test splits for the main text-only dataset.
- `merge_metadata.py`
  Extracts and merges metadata fields such as domain, title, and authors into the processed dataset.
- `part2_task3_splitting_meta.py`
  Creates the train, validation, and test splits for the metadata-enhanced dataset.

### Model folders
- [`lr_roberta/`](./lr_roberta)
  Contains the Logistic Regression baseline models and the RoBERTa advanced model.
- [`nb_distilbert/`](./nb_distilbert)
  Contains the Naive Bayes baseline models and the DistilBERT advanced model.
- [`lr/`](./lr)
  Contains the Linear Regression baseline models and the advanced model.

## Quick start

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run all commands from the project root so the relative `data/`, `models/`, `result/`, and output paths work correctly.

## Data-processing workflow

The project can be run in this order:

1. Preprocess the raw text:
```bash
python part2_task1_preprocessing.py
```

2. Run exploratory analysis:
```bash
python part2_task2_eda.py
```

3. Create the main text-only splits:
```bash
python part2_task3_splitting.py
```

4. If metadata is needed, merge metadata into the processed dataset:
```bash
python merge_metadata.py
```

5. Create the metadata-based splits:
```bash
python part2_task3_splitting_meta.py
```

## Model instructions

### Logistic Regression and RoBERTa
The Logistic Regression baseline and the RoBERTa advanced model are documented here:

- [Logistic Regression README](./lr_roberta/README_LogisticRegression.md)
- [RoBERTa README](./lr_roberta/README_RoBERTa.md)

Main scripts in this folder:
- `lr_roberta/part3_t1_logistic_regression_no_meta_data.py`
- `lr_roberta/part3_t1_logistic_regression_meta_data.py`
- `lr_roberta/part4_RoBERTa.py`
- `lr_roberta/part4_evaluate_roberta_on_liar.py`

### Naive Bayes and DistilBERT
The Naive Bayes baseline and the DistilBERT advanced model are documented here:

- [Naive Bayes README](./nb_distilbert/README_NaiveBayes.md)
- [DistilBERT README](./nb_distilbert/README_DistilBERT.md)

Main scripts in this folder:
- `nb_distilbert/part3_t1_naivebayes.py`
- `nb_distilbert/part3_t2_naivebayes_meta.py`
- `nb_distilbert/part4_distilbert.ipynb`
- `nb_distilbert/part5_nb_liar.py`

## Notes

- The binary label mapping used across the project keeps only clearly fake and clearly reliable classes, while ambiguous labels such as `political`, `bias`, and `unknown` are excluded.
- Some model scripts cache preprocessors and trained models under `models/` to avoid retraining every run.
- Reports, logs, and evaluation outputs are stored in folders such as `result/`, `lr_roberta/outputs/`, and `nb_distilbert/outputs/`.

## Summary

The project starts with preprocessing and dataset preparation, then compares simple baselines such as Naive Bayes and Logistic Regression against more advanced transformer models such as DistilBERT and RoBERTa. The linked READMEs inside each model folder contain the detailed settings, commands, and reported results for those experiments.
