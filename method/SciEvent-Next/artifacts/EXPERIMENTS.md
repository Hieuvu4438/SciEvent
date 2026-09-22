# Experiment registry

Code state: `5d4d7f7`. Hardware: NVIDIA RTX 5880 Ada (49 GB), CUDA 12.8, torch 2.11.0+cu128, transformers 5.5.4. Split: dev, checkpoint selected on dev Arg-C IoU F1 under a swept confidence threshold.

| run | backbone | seed | ep | lr | llrd | crf | span-role | best ep | Arg-C IoU | Arg-I IoU | Arg-C EM | ROUGE-L | type acc | runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `docsplit_s101` | deberta-v3-large | 101 | 30 | 1e-05/1e-04 | 1.0 | — | — | 16 | 48.51 | 56.25 | 26.49 | 74.5 | 89.2 | 544s |
| `docsplit_s13` | deberta-v3-large | 13 | 30 | 1e-05/1e-04 | 1.0 | — | — | 15 | 48.95 | 57.66 | 30.33 | 75.48 | 91.2 | 544s |
| `docsplit_s42` | deberta-v3-large | 42 | 30 | 1e-05/1e-04 | 1.0 | — | — | 18 | 45.92 | 54.65 | 29.58 | 76.95 | 87.8 | 531s |
| `final_s101` | deberta-v3-large | 101 | 30 | 1e-05/1e-04 | 1.0 | — | — | 26 | 46.87 | 55.78 | 28.42 | 78.68 | 89.2 | 552s |
| `final_s13` | deberta-v3-large | 13 | 30 | 1e-05/1e-04 | 1.0 | — | — | 19 | 48.1 | 57.31 | 28.86 | 78.99 | 88.6 | 549s |
| `final_s42` | deberta-v3-large | 42 | 30 | 1e-05/1e-04 | 1.0 | — | — | 19 | 47.4 | 54.71 | 29.68 | 76.72 | 89.2 | 541s |
| `h1_deberta_s42` | deberta-v3-large | 42 | 40 | 1e-05/1e-04 | 1.0 | — | — | 20 | 38.88 | 47.46 | 24.51 | 79.06 | 89.2 | 763s |
| `h3_llrd_s42` | deberta-v3-large | 42 | 30 | 2e-05/1e-04 | 0.9 | — | — | 19 | 46.25 | 54.15 | 27.67 | 78.34 | 88.0 | 546s |
| `h4_crf_s42` | deberta-v3-large | 42 | 30 | 2e-05/1e-04 | 0.9 | yes | — | 11 | 46.67 | 54.25 | 31.26 | 79.03 | 89.9 | 898s |
| `h8_span_s42` | deberta-v3-large | 42 | 30 | 2e-05/1e-04 | 0.9 | — | yes | 11 | 46.53 | 52.64 | 27.97 | 78.54 | 90.5 | 548s |

Note: the per-run Arg-C IoU above uses each run's own selection-time threshold. The cross-run comparisons in RESEARCH_LOG.md re-score every checkpoint with one identical widened decoding grid, which is the apples-to-apples number.
