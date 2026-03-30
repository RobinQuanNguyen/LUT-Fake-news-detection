# Task 2 report helper

## Dataset representation
I represented the FakeNewsCorpus as Pandas DataFrame chunks while reading the original CSV file. This design is suitable because the corpus is large, and chunked DataFrames let me inspect columns, compute statistics, and process text without loading the whole dataset into memory.

## What this script reports
- missing values by column
- class distribution
- duplicate-content rate
- article-length statistics and outliers
- URL, date, and numeric counts in the content
- top 100 words in three versions: raw, stopwords removed, and stemmed
- rank-frequency plots for the top 10,000 words in those same three versions

## Candidate observations from this run
1. Class distribution is imbalanced: 'political' appears 22.88 times as often as 'hate'.
2. The dataset contains duplicate or near-duplicate content rows at a rate of 12.59%, which may indicate artefacts or syndicated reposting.
3. Metadata-like content differs by class: 'clickbait' has the highest average URL count per article (0.71), while 'rumor' has the lowest (0.01).
4. Article length differs by class: 'unknown' has the longest average article length (599.91 tokens), whereas 'rumor' has the shortest (308.93 tokens).


## Inherent data problems to discuss
- Missing values in some columns
- Duplicate or reposted content
- Boilerplate text, URLs, and metadata leaking into article body
- Strong differences in article length between rows/classes

## Files to reference in your notebook/report
- `summary_stats.json`
- `missing_values.csv`
- `class_distribution.csv`
- `feature_summary_by_class.csv`
- `token_length_quantiles_overall.csv`
- `top_100_words_raw.csv`
- `top_100_words_no_stop.csv`
- `top_100_words_stemmed.csv`
- plot PNG files in the same folder
