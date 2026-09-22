# Superseded runs

These H0 runs used `k_span=512`, which capped dev candidate span recall at
97.6% — below the STOP_MODEL_02 gate of 98%. They are kept for the audit trail
but are excluded from `artifacts/runs/`, and therefore from `final_report.py`,
so no results table can mix two different proposal capacities.

Replaced by `k_start=k_end=128, k_span=2048` (dev recall 99.40%, which equals
the max_span_width ceiling). Evidence: `artifacts/audits/candidate_recall_sweep.json`.

| run | dev Arg-C IoU F1 | dev candidate recall |
|---|---|---|
| h0_span.seed42 (k_span=512) | 30.75 | 0.9718 |
| h0_span.seed13 (k_span=512) | 33.65 | 0.9759 |
| h0_cand.k96_s1024.seed13    | 30.66 | 0.9899 |
