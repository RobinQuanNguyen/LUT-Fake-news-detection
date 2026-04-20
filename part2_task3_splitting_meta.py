'''
Part 2 Task 3 - Split the metadata dataset into train, validate and test sets
80% training, 10% validation, 10% test

Input:  data/processed_fakenews_with_meta.csv
        columns: id | type | processed_text | domain | title | authors

Output: data/train_meta.csv
        data/validate_meta.csv
        data/test_meta.csv
        data/split_report_meta.txt

Note: Same split strategy as part1_t3.py (same seed, same ratios)
      but keeps metadata columns (domain, title, authors)
      for use in Part 2 Task 2 metadata baseline comparison only.
      For the rest of the project, use the original splits (train.csv etc.)
'''

import hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
INPUT_PATH      = Path("data/processed_fakenews_with_meta.csv")
TRAIN_PATH      = Path("data/train_meta.csv")
VALIDATE_PATH   = Path("data/validate_meta.csv")
TEST_PATH       = Path("data/test_meta.csv")
REPORT_PATH     = Path("data/split_report_meta.txt")

# Split ratios
TRAIN_RATIO     = 0.8
VALIDATE_RATIO  = 0.1
TEST_RATIO      = 0.1

# Same seed as part1_t3.py for consistency
RANDOM_SEED     = 42

# Column names
LABEL_COLUMN    = "type"
ID_COLUMN       = "id"
TEXT_COLUMN     = "processed_text"
META_COLUMNS    = ["domain", "title", "authors"]


# Load data
def load_data(path: Path) -> pd.DataFrame:
    print(f"\n[1/5] Loading data from: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"      Rows loaded          : {len(df):>10,}")
    print(f"      Columns              : {list(df.columns)}")

    # Drop rows with missing label or text
    before = len(df)
    df = df.dropna(subset=[LABEL_COLUMN, TEXT_COLUMN])
    dropped = before - len(df)
    if dropped:
        print(f"      Rows dropped (NaN)   : {dropped:>10,}")

    print(f"      Rows after NaN drop  : {len(df):>10,}")
    return df


# Deduplicate using MD5 hash on processed_text
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2/5] Deduplicating on '{TEXT_COLUMN}'...")
    before = len(df)
    seen = set()
    mask = []

    for text in df[TEXT_COLUMN]:
        h = hashlib.md5(str(text).encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            mask.append(False)
        else:
            seen.add(h)
            mask.append(True)

    df = df[mask]
    removed = before - len(df)
    rate = removed / before * 100 if before else 0
    print(f"      Duplicates removed   : {removed:>10,}  ({rate:.2f}%)")
    print(f"      Rows after dedup     : {len(df):>10,}")
    return df


# Show class distribution
def show_class_distribution(df: pd.DataFrame, label: str) -> None:
    counts = df[LABEL_COLUMN].value_counts()
    pct = (counts / len(df) * 100).round(2)
    print(f"\n      Class distribution in '{label}'  (total: {len(df):,})")
    for cls, n in counts.items():
        print(f"        {cls:<22} {n:>9,}  ({pct[cls]:.2f}%)")


# Two-step stratified split
def split_data(df: pd.DataFrame):
    print(f"\n[3/5] Splitting  (seed={RANDOM_SEED}) ...")

    # Step A: full dataset -> train (80%) + temp (20%)
    temp_size = 1.0 - TRAIN_RATIO
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=RANDOM_SEED,
        stratify=df[LABEL_COLUMN],
    )

    # Step B: temp -> val (10%) + test (10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df[LABEL_COLUMN],
    )

    n = len(df)
    print(f"      Train    : {len(train_df):>9,}  ({len(train_df)/n*100:.1f}%)")
    print(f"      Validate : {len(val_df):>9,}  ({len(val_df)/n*100:.1f}%)")
    print(f"      Test     : {len(test_df):>9,}  ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df


# Verify splits
def verify_splits(full_df, train_df, val_df, test_df) -> None:
    print(f"\n[4/5] Verifying splits...")

    # Size check
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(full_df), f"Size mismatch: {total} != {len(full_df)}"
    print(f"      Size check    : PASSED  ({total:,} rows accounted for)")

    # Overlap check
    train_ids = set(train_df[ID_COLUMN])
    val_ids   = set(val_df[ID_COLUMN])
    test_ids  = set(test_df[ID_COLUMN])

    assert len(train_ids & val_ids)  == 0, "Overlap found between train and val!"
    assert len(train_ids & test_ids) == 0, "Overlap found between train and test!"
    assert len(val_ids   & test_ids) == 0, "Overlap found between val and test!"
    print(f"      Overlap check : PASSED  (no article in more than one split)")

    # Class distributions
    show_class_distribution(train_df, "train")
    show_class_distribution(val_df,   "validate")
    show_class_distribution(test_df,  "test")


# Save splits
def save_splits(train_df, val_df, test_df) -> None:
    train_df.to_csv(TRAIN_PATH,    index=False, encoding="utf-8")
    val_df.to_csv(VALIDATE_PATH,   index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH,      index=False, encoding="utf-8")
    print(f"\n[5/5] Saved splits:")
    print(f"      {TRAIN_PATH}")
    print(f"      {VALIDATE_PATH}")
    print(f"      {TEST_PATH}")


# Write plain-text report
def write_report(full_df, train_df, val_df, test_df) -> None:
    lines = ["=" * 62, "PART 2 TASK 3 — METADATA SPLIT REPORT", "=" * 62]
    lines += [
        f"Input file         : {INPUT_PATH}",
        f"Total rows (dedup) : {len(full_df):,}",
        f"Training rows      : {len(train_df):,}  ({len(train_df)/len(full_df)*100:.1f}%)",
        f"Validation rows    : {len(val_df):,}  ({len(val_df)/len(full_df)*100:.1f}%)",
        f"Test rows          : {len(test_df):,}  ({len(test_df)/len(full_df)*100:.1f}%)",
        f"Random seed        : {RANDOM_SEED}",
        f"Split strategy     : Stratified random split (scikit-learn)",
        f"Extra columns      : {META_COLUMNS}",
        "",
    ]

    lines.append(f"{'Class':<22} {'Full':>9} {'Train':>9} {'Val':>9} {'Test':>9}")
    lines.append("-" * 62)
    for cls in sorted(full_df[LABEL_COLUMN].unique()):
        n_full  = (full_df[LABEL_COLUMN]  == cls).sum()
        n_train = (train_df[LABEL_COLUMN] == cls).sum()
        n_val   = (val_df[LABEL_COLUMN]   == cls).sum()
        n_test  = (test_df[LABEL_COLUMN]  == cls).sum()
        lines.append(
            f"{cls:<22} {n_full:>9,} {n_train:>9,} {n_val:>9,} {n_test:>9,}"
        )

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n      Report saved : {REPORT_PATH}")
    print("\n" + report)


def main():
    print("=" * 62)
    print("PART 2 TASK 3 — Metadata Split")
    print("=" * 62)

    df = load_data(INPUT_PATH)
    df = deduplicate(df)

    show_class_distribution(df, "full (deduplicated)")

    train_df, val_df, test_df = split_data(df)
    verify_splits(df, train_df, val_df, test_df)

    save_splits(train_df, val_df, test_df)
    write_report(df, train_df, val_df, test_df)

    print("\nTask completed successfully!")


if __name__ == "__main__":
    main()