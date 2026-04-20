'''
Script for merging metadta
Purpose: extracts domain, title, authors from the raw CSV 
and merges with existing processed_fakenews.csv on id.

Input:  data/news_cleaned_2018_02_13.csv  (raw 27GB file)
        data/processed_fakenews.csv       (existing processed file)

Output: data/processed_fakenews_with_meta.csv
'''

import csv
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Paths
RAW_PATH = Path("data/news_cleaned_2018_02_13.csv")
PROCESSED_PATH = Path("data/processed_fakenews.csv")
OUTPUT_PATH = Path("data/processed_fakenews_with_meta.csv")

# Metadata columns chosen to extract from raw file
META_COLUMNS = ["id", "domain", "title", "authors"]

# Read only 10,000 rows at a time for the sake of RAM
CHUNK_SIZE = 10_000

def print_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def set_max_csv_field_size_limit():
    max_int = sys.maxsize
    while max_int > 0:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int //= 10

# Read the raw file (27GB) in chunks 
# Only keep id + metadata columns
def extract_metadata(raw_path: Path) -> pd.DataFrame:
    print_log("Extracting metadata from raw file (chunked)...")
    chunks = []
    total = 0

    for chunk in pd.read_csv(
        raw_path,
        usecols=META_COLUMNS,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        encoding_errors="ignore",
        on_bad_lines="skip",
        engine="python",
    ):
        chunks.append(chunk)
        total += len(chunk)
        if total % 500_000 == 0:
            print_log(f"Read {total:,} rows so far...")

    meta_df = pd.concat(chunks, ignore_index=True)
    print_log(f"Total metadata rows extracted: {len(meta_df):,}")

    # Clean up text fields
    for col in ["domain", "title", "authors"]:
        meta_df[col] = meta_df[col].fillna("").astype(str).str.strip()

    # Drop duplicate ids, only keep first occurrence
    meta_df = meta_df.drop_duplicates(subset=["id"], keep="first")
    print_log(f"Unique ids after dedup: {len(meta_df):,}")

    return meta_df

def main():
    print("Script started!")
    print_log("Starting metadata merge")
    set_max_csv_field_size_limit()
 
    #Load existing processed file
    print_log("Step 1/4: Loading processed_fakenews.csv...")
    processed_df = pd.read_csv(
        PROCESSED_PATH,
        encoding="utf-8",
    )
    print_log(f"Rows in processed file: {len(processed_df):,}")

    # Extract metadata from raw file
    print_log("Step 2/4: Extracting metadata from raw file...")
    meta_df = extract_metadata(RAW_PATH)

    # Merge on id
    # left join, keep all rows from processed, add metadata where available
    print_log("Step 3/4: Merging...")
    merged_df = processed_df.merge(meta_df, on="id", how="left")
    print_log(f"Rows after merge: {len(merged_df):,}")

    # Check how many got metadata
    print_log("Step 4/4: Checking metadata coverage...")
    domain_filled = (merged_df["domain"] != "").sum()
    title_filled  = (merged_df["title"]  != "").sum()
    author_filled = (merged_df["authors"] != "").sum()
    print_log(f"Rows with domain  : {domain_filled:,}  ({domain_filled/len(merged_df)*100:.1f}%)")
    print_log(f"Rows with title   : {title_filled:,}  ({title_filled/len(merged_df)*100:.1f}%)")
    print_log(f"Rows with authors : {author_filled:,}  ({author_filled/len(merged_df)*100:.1f}%)")

    # Save
    print_log(f"Saving to {OUTPUT_PATH}...")
    merged_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print_log("Done!")
    print_log(f"Output columns: {list(merged_df.columns)}")

if __name__ == "__main__":
    main()
 