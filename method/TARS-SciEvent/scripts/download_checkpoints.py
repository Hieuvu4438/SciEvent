#!/usr/bin/env python3
"""
scripts/download_checkpoints.py

Download required pretrained models and tokenizers for TARS-SciEvent:
- ModernBERT-large (primary trainable checkpoint)
- BART-large (tokenizer only, for official preprocessing scripts)

Optimized for speed and stability using huggingface_hub and hf_transfer.
Saves immutable provenance to checkpoints/manifest.json.
"""

import argparse
import json
import os
import sys
import time

# Enable hf_transfer for maximum download speed if installed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed. Please run pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


PRIMARY_REPO = "answerdotai/ModernBERT-large"
PREPROCESS_REPO = "facebook/bart-large"


def download_primary(output_dir: str, manifest: dict):
    print(f"\n[1/2] Resolving & Downloading Primary Backbone: {PRIMARY_REPO}...")
    api = HfApi()
    info = api.model_info(PRIMARY_REPO)
    revision_sha = info.sha
    print(f"      Resolved commit SHA: {revision_sha}")

    local_dir = os.path.join(output_dir, "modernbert-large")
    os.makedirs(local_dir, exist_ok=True)

    t0 = time.time()
    snapshot_download(
        repo_id=PRIMARY_REPO,
        revision=revision_sha,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,
    )
    elapsed = time.time() - t0
    print(f"      Downloaded {PRIMARY_REPO} to {local_dir} in {elapsed:.1f}s.")

    manifest["modernbert-large"] = {
        "repo_id": PRIMARY_REPO,
        "revision_sha": revision_sha,
        "local_dir": os.path.relpath(local_dir, start=output_dir),
        "tokenizer_only": False,
        "download_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def download_preprocess(output_dir: str, manifest: dict):
    print(f"\n[2/2] Resolving & Downloading Preprocessing Tokenizer: {PREPROCESS_REPO}...")
    api = HfApi()
    info = api.model_info(PREPROCESS_REPO)
    revision_sha = info.sha
    print(f"      Resolved commit SHA: {revision_sha}")

    local_dir = os.path.join(output_dir, "preprocess-bart-tokenizer")
    os.makedirs(local_dir, exist_ok=True)

    t0 = time.time()
    # Download ONLY tokenizer and config files, NOT the heavy model weights (.bin / .safetensors)
    snapshot_download(
        repo_id=PREPROCESS_REPO,
        revision=revision_sha,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*tokenizer*",
            "*vocab*",
            "*merges*",
            "config.json",
            "special_tokens_map.json",
        ],
        resume_download=True,
        max_workers=4,
    )
    elapsed = time.time() - t0
    print(f"      Downloaded {PREPROCESS_REPO} tokenizer to {local_dir} in {elapsed:.1f}s.")

    manifest["preprocess-bart-tokenizer"] = {
        "repo_id": PREPROCESS_REPO,
        "revision_sha": revision_sha,
        "local_dir": os.path.relpath(local_dir, start=output_dir),
        "tokenizer_only": True,
        "download_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    parser = argparse.ArgumentParser(description="Download pretrained checkpoints and tokenizers for TARS-SciEvent")
    parser.add_argument("--profile", choices=["all", "primary", "preprocess"], default="all",
                        help="Which checkpoint profile to download (default: all)")
    parser.add_argument("--output", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--manifest", default="checkpoints/manifest.json", help="Path to manifest.json")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    manifest = {}
    if os.path.exists(args.manifest):
        try:
            with open(args.manifest, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"Warning: could not read existing manifest {args.manifest}: {e}")

    if args.profile in ("all", "primary"):
        download_primary(args.output, manifest)

    if args.profile in ("all", "preprocess"):
        download_preprocess(args.output, manifest)

    # Write updated manifest
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest successfully written to: {args.manifest}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
