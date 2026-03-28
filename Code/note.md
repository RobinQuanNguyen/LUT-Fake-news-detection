Part 2:
Quan - Task 1
Hung - Task 2
Thet - Task 3

Instruction for downloading the data:
- Go to this repo: https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0
- Download EVERYTHING
- Select news.csv.zip and choose "Open" not unarchive.
- Remember to select a destination for the csv file
- Take about 5 mins for the file to be ready.

Start the process:
Run `pip3 install -r requirements.txt`

Note:
- It is problematic when handling such a big file like this (27GB). Tried with:
    - Dividing the data into smaller files (doesn't work)
    - Take only 20% of the data to process first (doesn't work - the task says we have to clean the whole data first)
    - Pandaralle (fast, strong, and uses all the computer resources but potentially crashes midway)
    - ProcessPoolExecutor (low to maintaince, but weak and slow)
- Currently, try from scraft, go from single-processor, and go to multi-processor, and reduce the chunk size to 50000 rows per chunk. Latest result, ram is about 30-40% and so on the CPU (good result, I guess). The processing time is about 50-70s (previously, in the single processor, it was 250-400s per chunk)
- Blindly tuning is not actually good, for 16 processors available. I achieve the best performance by using 8 (the more workers I use, the lower the performance I get)
- The program runs and does all the tasks well. But the text's quality is a bit low, so I add a filter that detect those nonsense words and remove them.
- With the latest version, the RAM is constantly running at the level of 30/80GB, but the CPU is jumping up and down, which I guess is ok.

Runs for about 5 hours:
[2026-03-20 14:44:53] Chunk 171: done in 45.1s | rows_written=5,917,423 | vocab_before=2,754,697 | vocab_after_stop=2,754,544 | vocab_after_stem=2,393,215
[2026-03-20 14:44:54] Rows seen: 8,529,090
[2026-03-20 14:44:54] Rows written: 5,917,423


===== FINAL STATS =====
Total rows seen: 8,529,090
Total rows counted after pipeline filtering: 5,917,423
Total token count before stopwords: 2,650,426,393
Total token count after stopwords: 1,492,810,996
Total removed stopword tokens: 1,157,615,397

The audit pass over the filtered corpus counted 2.65 billion tokens before stopword removal and 1.49 billion after stopword removal, meaning 1.16 billion stopword tokens were removed, or approximately 43.7% of all tokens. The most frequent tokens before removal were standard English function words such as “the”, “to”, “of”, and “and”, while after removal the distribution shifted toward content-bearing terms such as “said”, “people”, “trump”, and “president”. This indicates the stopword removal stage functioned as intended.

Top 50 frequent tokens BEFORE stopwords:
the     150,477,735
to      72,644,134
of      72,516,047
and     65,503,035
a       56,925,324
in      50,230,099
that    33,040,151
is      30,450,190
s       28,921,184
for     24,812,217
it      21,053,887
on      20,564,758
as      16,255,652
with    15,904,886
are     14,744,116
i       14,644,085
was     13,642,480
this    13,607,040
be      12,870,272
by      12,840,830
at      12,727,531
have    12,537,153
not     12,040,702
from    11,936,913
you     11,675,763
he      11,587,535
they    10,078,089
we      9,913,025
has     9,681,756
an      9,621,119
but     9,368,577
his     9,174,887
or      8,999,842
will    8,130,543
who     7,806,502
t       7,611,451
their   7,329,244
said    7,118,332
more    6,725,236
one     6,470,302
all     6,462,853
about   6,404,935
which   6,110,604
can     5,998,937
if      5,897,942
what    5,562,033
would   5,527,429
people  5,469,876
been    5,415,230
there   5,316,423

Top 50 frequent tokens AFTER stopwords:
said    7,118,332
one     6,470,302
would   5,527,429
people  5,469,876
new     5,087,870
also    4,086,176
like    3,991,674
us      3,880,851
time    3,775,164
year    3,478,026
trump   3,396,741
even    3,203,043
news    3,134,698
president       3,105,712
iran    3,085,909
state   3,012,556
first   2,980,749
many    2,814,182
could   2,800,148
two     2,797,047
government      2,787,357
years   2,721,199
world   2,654,065
may     2,637,540
get     2,503,169
last    2,474,842
obama   2,405,970
u       2,405,950
iranian 2,360,025
states  2,338,574
public  2,332,833
see     2,330,750
right   2,293,289
back    2,266,504
day     2,254,299
well    2,239,454
make    2,159,316
made    2,127,245
know    2,111,568
american        2,084,771
way     2,045,736
use     2,016,873
going   1,993,454
much    1,989,814
health  1,922,440
war     1,905,831
system  1,904,603
com     1,826,055
recs    1,803,027
think   1,801,509

## Task 3

### What was done
Split the cleaned dataset (`processed_fakenews.csv` from Task 1) into
three sets for model training, tuning, and evaluation:
- Training set: 80% → 3,449,718 rows
- Validation set: 10% → 431,215 rows
- Test set: 10% → 431,215 rows

### Steps taken for task 3
**1. Load & drop NaN rows**
Loaded only the three necessary columns (`id`, `type`, `processed_text`).
Dropped 320,760 rows with missing label or text as these can't be used
for training or evaluation.

**2. Deduplication (before splitting)**
Removed 1,284,515 duplicate articles (22.95%) using MD5 hashing instead
of pandas drop_duplicates() to keep memory usage low on 8GB RAM.

**Why before splitting:** if the same article ends up in both train and test,
the model effectively "sees" test data during training, inflating scores
and making results untrustworthy.

**3. Stratified random split**
Used scikit-learn's train_test_split() with stratify=type in two steps:
- Step A: full dataset -> train (80%) + temp (20%)
- Step B: temp -> val (10%) + test (10%)

**Why stratified:** the dataset is heavily imbalanced across 12 classes
(e.g. reliable=25.83%, hate=1.31%). A plain random split could
accidentally under-represent rare classes in the test set. Stratification
guarantees every split mirrors the full dataset's class proportions exactly.

**Why random seed (42):** ensures the exact same splits are reproduced
every time the script is run, making results reproducible for the team.

**4. Verification**
Two checks before saving:
- Size check: all rows accounted for (no rows lost or duplicated)
- Overlap check: no article appears in more than one split
- Both checks PASSED.

### Output files
- `data/train.csv` — model learns from this
- `data/validate.csv` — used to tune and compare models
- `data/test.csv` — final evaluation only, touched once at the end
- `data/split_report.txt` — full per-class breakdown across all splits

### Key results
- Duplicates removed: 1,284,515 (22.95%) confirmed Task 2's finding
  that the corpus contains significant duplicate/syndicated content
- 12 classes all perfectly proportioned across train/val/test
- Final dataset after dedup: 4,312,148 rows