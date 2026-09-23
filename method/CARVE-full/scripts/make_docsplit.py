"""Build a DOCUMENT-disjoint re-split, as a leakage robustness check.

Why this exists
---------------
The released `data_scripts/shared/split_data.py` dedups and splits on `wnd_id`
(window), stratified by event type. The paper describes an 80/10/10 split "by
document". The consequence is that 143 of the 147 test documents (97.3 %) also
have sibling segments in the training split: the *windows* are disjoint, but the
*abstracts* are not.

This is a property of the benchmark, and it applies identically to every baseline
in the paper, so the headline comparison against Table 3 / Table 4 remains
apples-to-apples. But it means the official test number cannot, on its own, tell
us how much of the score comes from having seen the same abstract.

This script rebuilds a split with the same proportions and the same event-type
stratification, but keyed on `doc_id`, so no abstract crosses a split boundary.
Retraining on it gives a lower bound that is free of document overlap.

Output: data/docsplit/{train,dev,test}.oneie.json  (same schema, so
`--set data_dir=...` works unchanged).
"""

import collections
import json
import os
import random
import sys

from carve.paths import default_data_dir, repo_root

OUT = os.path.join(repo_root(), "data", "docsplit")
SEED = 42


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


src = default_data_dir()
pool = []
for sp in ["train", "dev", "test"]:
    pool += load(os.path.join(src, f"{sp}.oneie.json"))
print(f"pooled {len(pool)} windows from the official splits")

by_doc = collections.defaultdict(list)
for w in pool:
    by_doc[w["doc_id"]].append(w)
docs = sorted(by_doc)
print(f"{len(docs)} distinct documents")

# Stratify documents by their dominant event-type signature so the event-type mix
# stays close to the official splits, then assign whole documents.
def sig(d):
    types = sorted({w["event_mentions"][0]["event_type"] for w in by_doc[d]})
    return "|".join(types)


groups = collections.defaultdict(list)
for d in docs:
    groups[sig(d)].append(d)

rng = random.Random(SEED)
train_docs, dev_docs, test_docs = [], [], []
for g in sorted(groups):
    ds = groups[g][:]
    rng.shuffle(ds)
    n = len(ds)
    n_tr, n_dv = int(0.8 * n), int(0.1 * n)
    train_docs += ds[:n_tr]
    dev_docs += ds[n_tr:n_tr + n_dv]
    test_docs += ds[n_tr + n_dv:]

assert not (set(train_docs) & set(dev_docs) & set(test_docs))
assert not set(train_docs) & set(test_docs)
assert not set(dev_docs) & set(test_docs)

os.makedirs(OUT, exist_ok=True)
for name, ds in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
    rows = [w for d in ds for w in by_doc[d]]
    rng.shuffle(rows)
    with open(os.path.join(OUT, f"{name}.oneie.json"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    et = collections.Counter(w["event_mentions"][0]["event_type"] for w in rows)
    dom = collections.Counter(w["sent_id"].split("_")[0] for w in rows)
    print(f"{name:5s}: {len(ds):3d} docs, {len(rows):4d} windows | "
          f"types {dict(sorted(et.items()))}")
    print(f"        domains {dict(sorted(dom.items()))}")

print(f"\nwrote {OUT}")
print("document overlap train->test:",
      len(set(train_docs) & set(test_docs)), "(must be 0)")
