"""Machine-checkable parts of the audit (see AUDIT.md sections B, C, G, H)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scievent_next.data import EXCLUDED_FROM_SCORING, default_data_dir

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
G = default_data_dir()
tr, dv, te = (load(os.path.join(G, f"{s}.oneie.json")) for s in ["train", "dev", "test"])

w = lambda d: {x["sent_id"] for x in d}
d = lambda x: {y["doc_id"] for y in x}
print("[B] window overlap  train^test=%d  train^dev=%d  dev^test=%d  (all must be 0)"
      % (len(w(tr) & w(te)), len(w(tr) & w(dv)), len(w(dv) & w(te))))
ttr = {" ".join(x["tokens"]) for x in tr}
print("[B] exact window-text duplicates train->test: %d/%d"
      % (sum(1 for x in te if " ".join(x["tokens"]) in ttr), len(te)))
ov = len(d(te) & d(tr))
print("[C] test docs also in train: %d/%d = %.1f%%  <-- benchmark property, affects all baselines"
      % (ov, len(d(te)), 100 * ov / len(d(te))))
print("[H] events per TRAIN window: %s over %d windows (assumption derivable from train alone)"
      % (set(len(x["event_mentions"]) for x in tr), len(tr)))

p = load(os.path.join(os.path.dirname(G), "..", "..", "..", "method", "SciEvent-Next",
                      "artifacts", "preds", "test_preds_s13.jsonl")) \
    if False else load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                    "artifacts", "preds", "test_preds_s13.jsonl"))
sc = lambda r: sum(1 for a in r["event_mentions"][0]["arguments"] if a["role"] not in EXCLUDED_FROM_SCORING)
print("[G] windows pred/gold %d/%d  ids match %s  events/window %s"
      % (len(p), len(te), [x["sent_id"] for x in p] == [x["sent_id"] for x in te],
         set(len(x["event_mentions"]) for x in p)))
print("[G] scored args pred/gold %d/%d  ratio %.3f (<1 => we under-predict)"
      % (sum(map(sc, p)), sum(map(sc, te)), sum(map(sc, p)) / sum(map(sc, te))))
dups = 0
for r in p:
    ent = {e["id"]: (e["start"], e["end"]) for e in r["entity_mentions"]}
    seen = set()
    for a in r["event_mentions"][0]["arguments"]:
        k = (ent[a["entity_id"]], a["role"])
        dups += k in seen
        seen.add(k)
print("[G] duplicate (span,role) predictions: %d (must be 0)" % dups)
