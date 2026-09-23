"""Stage 3 of HONE: turn verifier probabilities into final predictions.

For every candidate the verifier returns p over {reject, role_1..role_9}.

    keep score  = 1 - p(reject)
    role        = argmax over the 9 roles

A window's output is built greedily: candidates with keep score >= theta (and
length >= min_len) are visited in decreasing keep score and accepted unless they
collide with an already accepted span. Because candidates are pooled from
several proposer seeds, near-duplicates of the same gold span are common; the
collision rule is what keeps one of them. Two collision rules are searched:

    "any"  reject any token overlap  (gold scored spans are non-overlapping in
                                      97.6 % of windows, so this is the natural prior)
    "iou"  reject only IoU > 0.5     (the metric's own notion of "the same span")

Window event type and the AAO spans (used only for the trigger ROUGE-L tuple)
come from the proposer posteriors averaged over seeds. Averaging is safe for
both: neither is thresholded, so the calibration effect that made posterior
ensembling harmful for CARVE's role spans does not apply.
"""

import numpy as np

from hone.candidates import LABELS, iou
from hone.data import AAO_LABEL2ID, EVENT_TYPES, bio_to_spans
from hone.evaluate import score, to_oneie_record

ID2AAO = {i: l for l, i in AAO_LABEL2ID.items()}
THETAS = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
MIN_LENS = [1, 2, 3]
NMS_MODES = ["any", "iou", "wis"]


def _collides(a, b, mode):
    if mode == "any":
        return max(a[0], b[0]) < min(a[1], b[1])
    return iou(a, b) > 0.5


def _wis(cands, keep, theta, min_len):
    """Exact maximum-weight set of pairwise non-overlapping candidates.

    Weight of a candidate = log-odds of its keep score minus log-odds of theta,
    so only candidates the verifier prefers over the threshold carry positive
    weight, and among overlapping alternatives (a whole clause vs its fragments)
    the jointly best set wins instead of whichever single span scores highest.
    Standard weighted-interval-scheduling DP, O(n log n).
    """
    eps = 1e-6
    lo = np.log(theta + eps) - np.log(1 - theta + eps)
    items = []
    for j, c in enumerate(cands):
        if c["e"] - c["s"] < min_len:
            continue
        w = np.log(keep[j] + eps) - np.log(1 - keep[j] + eps) - lo
        if w > 0:
            items.append((c["s"], c["e"], w, j))
    if not items:
        return []
    items.sort(key=lambda x: x[1])
    ends = [it[1] for it in items]
    import bisect
    best = [0.0] * (len(items) + 1)
    take = [False] * len(items)
    prev = []
    for i, (s, e, w, j) in enumerate(items):
        p = bisect.bisect_right(ends, s, 0, i)      # items ending at or before s
        prev.append(p)
        if best[p] + w > best[i]:
            best[i + 1] = best[p] + w
            take[i] = True
        else:
            best[i + 1] = best[i]
    chosen, i = [], len(items) - 1
    while i >= 0:
        if take[i]:
            chosen.append(items[i][3])
            i = prev[i] - 1
        else:
            i -= 1
    return chosen


def decode_window(cands, probs, theta, min_len, nms):
    """cands: candidate dicts with s, e.  probs: [n_cands, 10]."""
    if len(cands) == 0:
        return []
    if nms == "wis":
        keep = 1.0 - probs[:, 0]
        idx = _wis(cands, keep, theta, min_len)
        return sorted((cands[j]["s"], cands[j]["e"], LABELS[1 + int(np.argmax(probs[j, 1:]))]) for j in idx)
    keep = 1.0 - probs[:, 0]
    order = np.argsort(-keep)
    accepted = []
    for j in order:
        if keep[j] < theta:
            break
        s, e = cands[j]["s"], cands[j]["e"]
        if e - s < min_len:
            continue
        if any(_collides((s, e), (a[0], a[1]), nms) for a in accepted):
            continue
        role = LABELS[1 + int(np.argmax(probs[j, 1:]))]
        accepted.append((s, e, role))
    return accepted


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def mix_roles(rows, probs_per_window, alpha):
    """Keep score from the verifier, role from an alpha-mix of the verifier's role
    distribution and the proposer's normalised role mass (alpha = 1: verifier only).

    Measured on dev (H1): the verifier ranks keep/drop better than the proposer
    (AUC 0.890 vs 0.860) but assigns roles worse (77.1 % vs 81.3 % on matched
    candidates), so the two sources of role evidence are complementary.
    """
    if alpha >= 1.0:
        return probs_per_window
    out = []
    for r, pw in zip(rows, probs_per_window):
        m = pw.copy()
        for j, c in enumerate(r["cands"]):
            pm = np.asarray(c["role_mass"], dtype=float)
            pm = pm / max(1e-9, pm.sum())
            vr = pw[j, 1:] / max(1e-9, pw[j, 1:].sum())
            m[j, 1:] = (1 - pw[j, 0]) * (alpha * vr + (1 - alpha) * pm)
        out.append(m)
    return out


def role_given_type(train_windows, alpha=1.0):
    """log P(role | event type) from TRAIN gold, add-alpha smoothed. Used by the
    optional joint type decision (H6). Built from training labels only."""
    from hone.data import ROLE_TYPES
    counts = np.full((len(EVENT_TYPES), len(ROLE_TYPES)), alpha)
    for w in train_windows:
        t = EVENT_TYPES.index(w.event_type)
        for _, _, r in w.role_spans:
            counts[t, ROLE_TYPES.index(r)] += 1
    return np.log(counts / counts.sum(1, keepdims=True))


def joint_type(type_post, spans, log_r_given_t, lam):
    """argmax_t  log p(t) + lam * sum_i log P(role_i | t)   (lam = 0: unchanged)."""
    from hone.data import ROLE_TYPES
    score = np.log(np.asarray(type_post) + 1e-9)
    if lam > 0 and spans:
        idx = [ROLE_TYPES.index(r) for _, _, r in spans]
        score = score + lam * log_r_given_t[:, idx].sum(1)
    return EVENT_TYPES[int(np.argmax(score))]


def records(windows, rows, probs_per_window, theta, min_len, nms, type_lambda=0.0, log_r_given_t=None):
    recs = []
    for w, r, pr in zip(windows, rows, probs_per_window):
        spans = decode_window(r["cands"], pr, theta, min_len, nms)
        if type_lambda > 0 and log_r_given_t is not None:
            etype = joint_type(r["type_post"], spans, log_r_given_t, type_lambda)
        else:
            etype = EVENT_TYPES[int(np.argmax(r["type_post"]))]
        aao = bio_to_spans(np.asarray(r["aao_post"]).argmax(-1).tolist(), ID2AAO)
        n = len(w.tokens)
        trig = next(((s, e) for s, e, t in aao if t == "Action"), (0, min(1, n)))
        aao_out = [(s, e, t) for s, e, t in aao if t != "Action"]
        recs.append(to_oneie_record(w.sent_id, w.tokens, etype, spans, aao_out, trig))
    return recs


def tune(windows, rows, probs_per_window, gold_path):
    """Grid-search the decoding rule on this split (dev only). Returns best."""
    best = None
    for nms in NMS_MODES:
        for ml in MIN_LENS:
            for th in THETAS:
                f = score(records(windows, rows, probs_per_window, th, ml, nms),
                          gold_path, with_rouge=False)["arg_c_iou"]["f1"]
                if best is None or f > best["f1"]:
                    best = {"f1": f, "theta": th, "min_len": ml, "nms": nms}
    return best
