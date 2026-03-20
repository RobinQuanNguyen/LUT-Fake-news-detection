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
- The program runs and does all the tasks well. But the text's quality is a bit low, so I add