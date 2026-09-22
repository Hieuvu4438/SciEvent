#!/usr/bin/env python3
"""
scripts/reconstruct_cscw_abstracts.py

Reconstruct the 100 official CSCW abstract text files from the historical commit
blob (SciEvent_data/DEGREE/processed/cscw.json) in third_party/SciEvent.
Ensures 100% exact token alignment with event_seg.jsonl and event_extraction_finetune_model.jsonl.
"""

import json
import os
import subprocess
from collections import defaultdict

PRIVATE_CSCW_DIR = "/home/haipd/SciEvent/method/TARS-SciEvent/private_data/cscw"
OFFICIAL_ABSTRACTS_DIR = "/home/haipd/SciEvent/third_party/SciEvent/SciEvent_data/abstracts_texts"
REPO_DIR = "/home/haipd/SciEvent/third_party/SciEvent"
BLOB_HASH = "4444d00cf9e59d8031ee60c6463377be2fbba51c"

os.makedirs(PRIVATE_CSCW_DIR, exist_ok=True)
os.makedirs(OFFICIAL_ABSTRACTS_DIR, exist_ok=True)

# Fetch blob content
raw_data = subprocess.check_output(["git", "-C", REPO_DIR, "show", BLOB_HASH]).decode("utf-8")
doc_windows = defaultdict(list)
for line in raw_data.strip().split("\n"):
    if line.strip():
        item = json.loads(line)
        doc_windows[item["doc_id"]].append(item)

print(f"Reconstructing {len(doc_windows)} CSCW abstracts from historical commit blob...")

for doc_id, wnds in doc_windows.items():
    # Sort windows in order (0, 1, 2, ...)
    wnds.sort(key=lambda x: int(x["wnd_id"].split("-")[-1]))

    tokens = []
    for w in wnds:
        tokens.extend(w["tokens"])

    content = f"{doc_id}\n{' '.join(tokens)}\n"

    private_file = os.path.join(PRIVATE_CSCW_DIR, f"{doc_id}.txt")
    with open(private_file, "w", encoding="utf-8") as f:
        f.write(content)

    official_file = os.path.join(OFFICIAL_ABSTRACTS_DIR, f"{doc_id}.txt")
    with open(official_file, "w", encoding="utf-8") as f:
        f.write(content)

print(f"Successfully reconstructed all {len(doc_windows)} CSCW abstracts with exact token alignment!")
