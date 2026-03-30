'''
Task 3 - Split the data into train, validate and test sets
80% training, 10% validation, 10% test

Input: data/processed_fakenews.csv
columns: id | type | processed_text

Output: data/train.csv
        data/validate.csv
        data/test.csv
        data/split_report.txt

'''

from pathlib import Path
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split

# Paths
INPUT_PATH = Path('data/processed_fakenews.csv')
TRAIN_PATH = Path('data/train.csv')
VALIDATE_PATH = Path('data/validate.csv')
TEST_PATH = Path('data/test.csv')
REPORT_PATH = Path("data/split_report.txt")

# Define the split ratios
TRAIN_RATIO = 0.8
VALIDATE_RATIO = 0.1
TEST_RATION = 0.1

RANDOM_SEED = 42 # For reproducibility

# Column names matching task 1 output
LABEL_COLUMN = 'type'
ID_COLUMN = 'id'
TEXT_COLUMN = 'processed_text'

# Load the data
def load_data(path: Path) -> pd.DataFrame:
    # Only load the three columns from the processed CSV in Task 1.
    print(f"\n[1/5] Loading data from: {path}")
    df = pd.read_csv(path, usecols=[ID_COLUMN, LABEL_COLUMN, TEXT_COLUMN], encoding='utf-8')
    print(f"Rows Loaded: {len(df):>10,}")

    # Drop any rows with missing label or text as we can't use them
    before_drop = len(df)
    df = df.dropna(subset=[LABEL_COLUMN, TEXT_COLUMN])
    dropped_rows = before_drop - len(df)
    if dropped_rows > 0:
        print(f"Rows dropped (NaN)   : {dropped_rows:>10,}")
 
    print(f"Rows after NaN drop  : {len(df):>10,}")
    return df

# Deduplicate the data
def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2/5] Deduplicating data...")
    before_deduplicate = len(df)
    seen = set()
    mask = []

    for text in df[TEXT_COLUMN]:
        h = hashlib.md5(str(text).encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            mask.append(False)
        else:
            seen.add(h)
            mask.append(True)
    
    # keep='first' -> keep the first occurrence, drop the rest
    df = df[mask]
    removed_duplicates = before_deduplicate - len(df)
    rate = removed_duplicates / before_deduplicate * 100 if before_deduplicate > 0 else 0
    print(f"Duplicates removed: {removed_duplicates:>10,} ({rate:.2f}%)")
    print(f"Rows after deduplication: {len(df):>10,}")
    return df

# Class distribution
def show_class_distribution(df: pd.DataFrame, label: str) -> None:
    # Check how many rows belong to each class (in percentage)
    count = df[LABEL_COLUMN].value_counts()
    pct = (count / len(df) * 100).round(2)
    print(f"\nClass distribution in '{label}'  (total: {len(df):,})")
    for cls, n in count.items():
        print(f"{cls:<22} {n:>9,}  ({pct[cls]:.2f}%)")

# Split the data into train, validate and test sets
def split_data(df: pd.DataFrame):
    # Perform two-step stratified split
    # train_test_split() can only split into two sets at once.
    print(f"\n[3/5] Splitting  (seed={RANDOM_SEED}) ...")

    # First split into train and temp
    temp_size = 1.0 - TRAIN_RATIO   # 20% goes to temp
    train_df, temp_df = train_test_split(
        df, 
        test_size=temp_size, 
        random_state=RANDOM_SEED,
        stratify=df[LABEL_COLUMN]   # preserve class ratios
    )

    # Then split temp into validate and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df[LABEL_COLUMN]  # preserve class ratios
    )

    n = len(df)
    print(f"Train  : {len(train_df):>9,}  ({len(train_df)/n*100:.1f}%)")
    print(f"Val    : {len(val_df):>9,}  ({len(val_df)/n*100:.1f}%)")
    print(f"Test   : {len(test_df):>9,}  ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df

# Verify that sizes add up, no ID overlap, right class proportions
def verify_split(full_df, train_df, val_df, test_df) -> None:
    print(f"\n[4/5] Verifying split...")

    # Check total size
    total_size = len(train_df) + len(val_df) + len(test_df)
    assert total_size == len(full_df), (f"Total size mismatch: {total_size} != {len(full_df)}")

    print(f"Size check : PASSED  ({total_size:,} rows accounted for)")

    # Check ID overlap between any pair of splits
    train_ids = set(train_df[ID_COLUMN])
    val_ids   = set(val_df[ID_COLUMN])
    test_ids  = set(test_df[ID_COLUMN])

    assert len(train_ids & val_ids)  == 0, "Overlap found between train and val!"
    assert len(train_ids & test_ids) == 0, "Overlap found between train and test!"
    assert len(val_ids   & test_ids) == 0, "Overlap found between val and test!"
    print(f"Overlap check : PASSED  (no article in more than one split)")

    # Show distribution
    show_class_distribution(train_df, "train")
    show_class_distribution(val_df,   "val")
    show_class_distribution(test_df,  "test")

# Save & report
def save_splits(train_df, val_df, test_df) -> None:
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VALIDATE_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")\
    
    print(f"\n[5/5] Saved splits:")
    print(f"{TRAIN_PATH}")
    print(f"{VALIDATE_PATH}")
    print(f"{TEST_PATH}")

# Plain text report with summary statistics about the split
def write_report(full_df, train_df, val_df, test_df) -> None:
        lines = ["=" * 62, "TASK 3 — DATA SPLIT REPORT", "=" * 62]
        lines += [
        f"Input file         : {INPUT_PATH}",
        f"Total rows (dedup) : {len(full_df):,}",
        f"Training rows      : {len(train_df):,}  ({len(train_df)/len(full_df)*100:.1f}%)",
        f"Validation rows    : {len(val_df):,}  ({len(val_df)/len(full_df)*100:.1f}%)",
        f"Test rows          : {len(test_df):,}  ({len(test_df)/len(full_df)*100:.1f}%)",
        f"Random seed        : {RANDOM_SEED}",
        f"Split strategy     : Stratified random split (scikit-learn)",
        "",
    ]
        
        # Per-class breakdown across all three splits
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
    print("Train / Val / Test Split")

    df = load_data(INPUT_PATH)
    df = deduplicate_data(df)

    show_class_distribution(df, "full (deduplicated)")

    train_df, val_df, test_df = split_data(df)
    verify_split(df, train_df, val_df, test_df)

    save_splits(train_df, val_df, test_df)
    write_report(df, train_df, val_df, test_df)

    print("\nTask completed successfully!")

if __name__ == "__main__":
    main()
