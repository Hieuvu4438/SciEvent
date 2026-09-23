"""Data loading and label construction for CARVE.

Contract (read-only) with the upstream benchmark:
  third_party/SciEvent/SciEvent_data/ONEIE/all_splits/{train,dev,test}.oneie.json
Each line is one *window* (an annotated abstract segment) with:
  sent_id, tokens (whitespace tokens), entity_mentions [{id,start,end}],
  event_mentions [{event_type, trigger:{start,end}, arguments:[{entity_id, role}]}]

Every window carries exactly one event mention (verified on all three splits),
so the event type is a window-level 4-way classification problem.

The official evaluator (baselines/ONEIE/EM_overlap_eval.py) is
  * trigger-insensitive
  * event-type sensitive
  * excludes {Agent, PrimaryObject, SecondaryObject} from Arg-I / Arg-C
so we split the span-labelling problem into two nearly disjoint tagging tasks:
  ROLE head : the 9 scored semantic roles        (97.6% internally non-overlapping)
  AAO  head : Agent / Action / Primary / Secondary Object, used only for the
              trigger ROUGE-L tuple              (99.8% internally non-overlapping)
Cross-group overlap is 5.9%, which is exactly why they cannot share one head.
"""

import json
import os
from dataclasses import dataclass, field

from hone.paths import default_data_dir, split_path

# --- label inventories -------------------------------------------------------

# Scored semantic roles (everything the official Arg-I / Arg-C metrics look at).
ROLE_TYPES = [
    "Context",
    "Method",
    "Results",
    "Challenge",
    "Purpose",
    "Implications",
    "Analysis",
    "Contradictions",
    "Ethical",
]

# The <Agent, Action, Object> trigger tuple. "Action" is the gold trigger span.
AAO_TYPES = ["Agent", "Action", "PrimaryObject", "SecondaryObject"]

EVENT_TYPES = [
    "Background/Introduction",
    "Methods/Approach",
    "Results/Findings",
    "Conclusions/Implications",
]

EXCLUDED_FROM_SCORING = {"Agent", "PrimaryObject", "SecondaryObject"}


def bio_labels(types):
    labels = ["O"]
    for t in types:
        labels += [f"B-{t}", f"I-{t}"]
    return labels


ROLE_LABELS = bio_labels(ROLE_TYPES)
AAO_LABELS = bio_labels(AAO_TYPES)

# Label space for the "single BIO head" ablation: the two groups merged into one
# tagging problem. Because cross-group span overlap is 5.9%, a single layer must
# delete labels on those windows -- which is exactly what the ablation measures.
MERGED_TYPES = ROLE_TYPES + AAO_TYPES
MERGED_LABELS = bio_labels(MERGED_TYPES)
ROLE_LABEL2ID = {l: i for i, l in enumerate(ROLE_LABELS)}
AAO_LABEL2ID = {l: i for i, l in enumerate(AAO_LABELS)}
MERGED_LABEL2ID = {l: i for i, l in enumerate(MERGED_LABELS)}
EVENT_TYPE2ID = {t: i for i, t in enumerate(EVENT_TYPES)}

DOMAINS = ["ACL", "cscw", "bioinfo", "dh", "jmir"]


@dataclass
class Window:
    sent_id: str
    doc_id: str
    tokens: list
    event_type: str
    # list of (start, end, type) in whitespace-token space, end exclusive
    role_spans: list = field(default_factory=list)
    aao_spans: list = field(default_factory=list)

    @property
    def domain(self):
        return self.sent_id.split("_")[0]


def load_split(split, data_dir=None):
    path = split_path(split, data_dir)
    windows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            ent = {e["id"]: (e["start"], e["end"]) for e in raw["entity_mentions"]}
            ev = raw["event_mentions"][0]
            role_spans, aao_spans = [], []
            for arg in ev["arguments"]:
                span = ent.get(arg["entity_id"])
                if span is None:
                    continue
                s, e = span
                if e <= s:
                    continue
                if arg["role"] in EXCLUDED_FROM_SCORING:
                    aao_spans.append((s, e, arg["role"]))
                elif arg["role"] in ROLE_TYPES:
                    role_spans.append((s, e, arg["role"]))
            trig = ev["trigger"]
            if trig["end"] > trig["start"]:
                aao_spans.append((trig["start"], trig["end"], "Action"))
            windows.append(
                Window(
                    sent_id=raw["sent_id"],
                    doc_id=raw["doc_id"],
                    tokens=raw["tokens"],
                    event_type=ev["event_type"],
                    role_spans=role_spans,
                    aao_spans=aao_spans,
                )
            )
    return windows


def spans_to_bio(spans, n_tokens, label2id):
    """Greedy BIO construction. Shorter spans win conflicts (they are the more
    specific annotation); a longer span is truncated to the free tokens it still
    owns, which usually keeps IoU>0.5 against its gold span anyway."""
    labels = ["O"] * n_tokens
    taken = [False] * n_tokens
    for s, e, t in sorted(spans, key=lambda x: (x[1] - x[0], x[0])):
        s = max(0, s)
        e = min(n_tokens, e)
        free = [i for i in range(s, e) if not taken[i]]
        if not free:
            continue
        # keep only the first contiguous free run so BIO stays well-formed
        run = [free[0]]
        for i in free[1:]:
            if i == run[-1] + 1:
                run.append(i)
            else:
                break
        labels[run[0]] = f"B-{t}"
        for i in run[1:]:
            labels[i] = f"I-{t}"
        for i in run:
            taken[i] = True
    return [label2id[l] for l in labels]


def bio_to_spans(label_ids, id2label):
    """Decode BIO ids into (start, end, type). Tolerates I- without B-."""
    spans = []
    cur_type, cur_start = None, None
    for i, lid in enumerate(label_ids):
        lab = id2label[lid]
        if lab == "O":
            if cur_type is not None:
                spans.append((cur_start, i, cur_type))
                cur_type, cur_start = None, None
            continue
        prefix, t = lab.split("-", 1)
        if prefix == "B" or cur_type != t:
            if cur_type is not None:
                spans.append((cur_start, i, cur_type))
            cur_type, cur_start = t, i
    if cur_type is not None:
        spans.append((cur_start, len(label_ids), cur_type))
    return spans
