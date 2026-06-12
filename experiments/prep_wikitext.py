"""Prepare WikiText-103 for the PEER teacher: GPT-2 BPE -> packed uint16 memmaps.

Mirrors peer-adaptive-k's src/data.py prepare() but for wikitext-103-raw-v1,
writing to peer-adaptive-k/data_wikitext/{train,val}.bin so the teacher repo's
TokenData can be pointed at it unchanged (data_dir argument).

Run on pop:
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m experiments.prep_wikitext
"""

import os

import numpy as np
import tiktoken
from datasets import load_dataset

OUT = os.path.expanduser("~/Code/HN/peer-adaptive-k/data_wikitext")
EOT = 50256


def main():
    os.makedirs(OUT, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    for split, hf_split in [("train", "train"), ("val", "validation")]:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=hf_split)
        bufs, batch, total = [], [], 0
        for row in ds:
            t = row["text"]
            if t:
                batch.append(t)
            if len(batch) >= 20000:
                ids = enc.encode_ordinary("".join(batch))
                bufs.append(np.array(ids, dtype=np.uint16))
                total += len(ids)
                batch = []
        if batch:
            ids = enc.encode_ordinary("".join(batch))
            bufs.append(np.array(ids, dtype=np.uint16))
            total += len(ids)
        arr = np.concatenate(bufs)
        path = os.path.join(OUT, f"{split}.bin")
        arr.tofile(path)
        print(f"{split}: {total:,} tokens -> {path}", flush=True)


if __name__ == "__main__":
    main()
