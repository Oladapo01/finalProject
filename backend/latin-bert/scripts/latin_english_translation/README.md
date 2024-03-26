---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: la
    dtype: string
  - name: en
    dtype: string
  - name: file
    dtype: string
  splits:
  - name: train
    num_bytes: 39252644
    num_examples: 99343
  - name: test
    num_bytes: 405056
    num_examples: 1014
  - name: valid
    num_bytes: 392886
    num_examples: 1014
  download_size: 25567350
  dataset_size: 40050586
license: mit
task_categories:
- translation
language:
- la
- en
pretty_name: Latin to English Translation Pairs
size_categories:
- 10K<n<100K
---
# Dataset Card for "latin_english_parallel"

101k translation pairs between Latin and English, split 99/1/1 as train/test/val. These have been collected roughly 66% from the Loeb Classical Library and 34% from the Vulgate translation. 

For those that were gathered from the Loeb Classical Library, alignment was performd manually between Source and Target sequences.

Each sample is annotated with the index and file (and therefore author/work) that the sample is from. If you find errors, please feel free to submit a PR to fix them.

![alt text](distribution.png)