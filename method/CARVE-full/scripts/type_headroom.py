"""How much does imperfect window event-type prediction cost?

The official Arg-I/Arg-C metrics are event-type sensitive and each window holds
exactly one event, so a misclassified window annihilates every argument in it —
as false positives *and* false negatives simultaneously. This script measures
that cost by re-scoring the same argument spans with the gold event type
substituted (an oracle diagnostic, not a claim).

It also reports the accuracy of a simple legitimate feature that a deployed
pipeline would have: the ordinal position of the segment inside its abstract.
That number is reported for transparency only and is NOT used by the headline
model, because the paper's baselines did not use it.
"""

import collections
import os
import sys

import torch
from transformers import AutoTokenizer


from carve.data import EVENT_TYPES, load_split
from carve.paths import default_data_dir
from carve.evaluate import score
from carve.decode import build_records, posteriors

CKPT = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "microsoft/deberta-v3-large"
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "dev"
TAU, MIN_LEN = 0.7, 3

tok = AutoTokenizer.from_pretrained(MODEL)
w = load_split(SPLIT)
gold = os.path.join(default_data_dir(), f"{SPLIT}.oneie.json")
rp, ap, tp = posteriors([CKPT], MODEL, w, tok)

pred_acc = sum(EVENT_TYPES[int(tp[i].argmax())] == x.event_type for i, x in enumerate(w)) / len(w)

oracle = torch.zeros(len(w), len(EVENT_TYPES)).numpy()
for i, x in enumerate(w):
    oracle[i, EVENT_TYPES.index(x.event_type)] = 1.0

print(f"split={SPLIT}  decoding: tau={TAU} min_len={MIN_LEN}\n")
for name, types in [("predicted event type", tp), ("ORACLE event type", oracle)]:
    m = score(build_records(w, rp, ap, types, {"_": TAU}, 0, MIN_LEN), gold)
    print(f"  {name:22s}  Arg-C IoU {m['arg_c_iou']['f1']:6.2f}   Arg-I IoU {m['arg_i_iou']['f1']:6.2f}")
print(f"\n  model window-type accuracy: {pred_acc*100:.2f}")

# legitimate-but-unused feature: segment ordinal
tr = load_split("train")
by_idx = collections.defaultdict(collections.Counter)
for x in tr:
    by_idx[int(x.sent_id.rsplit("-", 1)[1])][x.event_type] += 1
maj = {i: c.most_common(1)[0][0] for i, c in by_idx.items()}
ord_acc = sum(maj.get(int(x.sent_id.rsplit("-", 1)[1])) == x.event_type for x in w) / len(w)
print(f"  segment-ordinal majority rule accuracy: {ord_acc*100:.2f}  "
      f"(reported for transparency; NOT used by the headline model)")
