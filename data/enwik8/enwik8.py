"""
Prepare the enwik8 dataset for character-level language modeling.
First, downloads the dataset from the URL and unzips it.
Then, it reads the file and converts it to a list of integers.
Finally, it saves the data to train.bin, val.bin, and test.bin.
"""

import os
import pickle
import requests
import zipfile
import numpy as np

from tqdm import tqdm

DATA_URL = "http://mattmahoney.net/dc/enwik8.zip"
VOCAB_SIZE = 256
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05  # Should be 1 - TRAIN_RATIO - VAL_RATIO

file_dir = os.path.dirname(__file__)
input_file_path = os.path.join(file_dir, "enwik8.bin")

# Download and unzip the dataset if not found
if not os.path.exists(input_file_path):
    if not os.path.exists(os.path.join(file_dir, "enwik8.zip")):
        print(f"Dataset not found, downloading from {DATA_URL}")

        r = requests.get(DATA_URL, stream=True)
        total_size = int(r.headers.get("content-length", 0))

        with open(os.path.join(file_dir, "enwik8.zip"), "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    with zipfile.ZipFile(os.path.join(file_dir, "enwik8.zip"), "r") as zip_ref:
        zip_ref.extractall(file_dir)

    os.remove(os.path.join(file_dir, "enwik8.zip"))
    os.rename(os.path.join(file_dir, "enwik8"), input_file_path)


with open(input_file_path, "rb") as f:
    data = f.read()


def encode(s: bytes) -> list[int]:
    return list(s)


def decode(l: list[int]) -> bytes:
    return bytes(l)


# Assert that encode and decode are inverses
assert decode(encode(b"hello")) == b"hello"
assert encode(decode([104, 101, 108, 108, 111])) == [104, 101, 108, 108, 111]

n = len(data)

# Calculate split indices
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

# Split the data
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Encode to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)

print(f"Total data has {n:,} characters")
print(f"Train data has {len(train_ids):,} tokens ({100*len(train_ids)/n:.1f}%)")
print(f"Validation data has {len(val_ids):,} tokens ({100*len(val_ids)/n:.1f}%)")
print(f"Test data has {len(test_ids):,} tokens ({100*len(test_ids)/n:.1f}%)")
print(f"Total tokens after split: {len(train_ids)+len(val_ids)+len(test_ids):,}")

# Convert to numpy arrays and save
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
test_ids.tofile(os.path.join(os.path.dirname(__file__), "test.bin"))

# Save metadata
meta = {"vocab_size": VOCAB_SIZE}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("\nData saved successfully:")
print(f"  - train.bin: {len(train_ids):,} tokens")
print(f"  - val.bin: {len(val_ids):,} tokens")
print(f"  - test.bin: {len(test_ids):,} tokens")