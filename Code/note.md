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
[2026-03-20 14:44:54] Rows seen: 8,528,956
[2026-03-20 14:44:54] Rows written: 5,917,423
[2026-03-20 14:44:54] Vocabulary before stopwords: 2,754,697
[2026-03-20 14:44:54] Vocabulary after stopwords: 2,754,544
[2026-03-20 14:44:54] Stopword reduction rate: 0.01%
[2026-03-20 14:44:54] Vocabulary after stemming: 2,393,215
[2026-03-20 14:44:54] Stemming reduction rate: 13.12%

Note: the data above is a bit off-track. Will update the new one later.