"""Build the experiment registry from every run's log.json.

Emits artifacts/EXPERIMENTS.md — one row per run with config, seed, best epoch,
runtime and dev metrics, so the whole campaign is auditable from one file.
"""

import glob
import json
import os
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


rows = []
for path in sorted(glob.glob(os.path.join(HERE, "artifacts", "runs", "*", "log.json"))):
    d = json.load(open(path))
    c, b, log = d["config"], d["best"], d["log"]
    rows.append({
        "run": os.path.basename(os.path.dirname(path)),
        "backbone": c["model_name"].split("/")[-1],
        "seed": c["seed"],
        "epochs": c["epochs"],
        "lr": f'{c["lr_encoder"]:.0e}/{c["lr_head"]:.0e}',
        "llrd": c.get("llrd", 1.0),
        "crf": c.get("use_crf", False),
        "span_role": c.get("use_span_role", False),
        "best_ep": b["epoch"],
        "arg_c_iou": round(b["arg_c_iou_f1"], 2),
        "arg_i_iou": round(b["arg_i_iou_f1"], 2),
        "arg_c_em": round(b["arg_c_em_f1"], 2),
        "rougeL": round(b["trigger_rougeL_f1"], 2),
        "type_acc": round(b["type_acc"], 1),
        "runtime_s": log[-1]["secs"],
    })

out = [
    "# Experiment registry",
    "",
    f"Code state: `{git_sha()}`. Hardware: NVIDIA RTX 5880 Ada (49 GB), CUDA 12.8, "
    "torch 2.11.0+cu128, transformers 5.5.4. Split: dev, checkpoint selected on "
    "dev Arg-C IoU F1 under a swept confidence threshold.",
    "",
    "| run | backbone | seed | ep | lr | llrd | crf | span-role | best ep | Arg-C IoU | Arg-I IoU | Arg-C EM | ROUGE-L | type acc | runtime |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    out.append(
        f"| `{r['run']}` | {r['backbone']} | {r['seed']} | {r['epochs']} | {r['lr']} | "
        f"{r['llrd']} | {'yes' if r['crf'] else '—'} | {'yes' if r['span_role'] else '—'} | "
        f"{r['best_ep']} | {r['arg_c_iou']} | {r['arg_i_iou']} | {r['arg_c_em']} | "
        f"{r['rougeL']} | {r['type_acc']} | {r['runtime_s']}s |"
    )
out += [
    "",
    "Note: the per-run Arg-C IoU above uses each run's own selection-time threshold. "
    "The cross-run comparisons in RESEARCH_LOG.md re-score every checkpoint with one "
    "identical widened decoding grid, which is the apples-to-apples number.",
    "",
]
dest = os.path.join(HERE, "artifacts", "EXPERIMENTS.md")
open(dest, "w").write("\n".join(out))
print("\n".join(out))
print(f"\nwrote {dest}")
