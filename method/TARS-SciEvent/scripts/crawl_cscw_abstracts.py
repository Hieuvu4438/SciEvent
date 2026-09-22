#!/usr/bin/env python3
"""
scripts/crawl_cscw_abstracts.py

Crawl the 100 missing CSCW abstracts from the official Crossref Open REST API
using DOIs provided in SciEvent_metadata.csv.
Formats each abstract according to the official SciEvent specification:
Line 1: doc_id
Line 2: single-line normalized abstract text

Saves to:
- private_data/cscw/<doc_id>.txt
- vendor/SciEvent/SciEvent_data/abstracts_texts/<doc_id>.txt
"""

import html
import json
import os
import re
import sys
import time
import urllib.request
import pandas as pd


METADATA_CSV = "/home/haipd/SciEvent/third_party/SciEvent/SciEvent_data/metadata/SciEvent_metadata.csv"
PRIVATE_CSCW_DIR = "/home/haipd/SciEvent/method/TARS-SciEvent/private_data/cscw"
OFFICIAL_ABSTRACTS_DIR = "/home/haipd/SciEvent/third_party/SciEvent/SciEvent_data/abstracts_texts"


def clean_abstract_xml(abstract_xml: str) -> str:
    # Strip XML tags like <jats:p>, <jats:italic>, <jats:title>
    text = re.sub(r"<[^>]+>", " ", abstract_xml)
    # Unescape HTML/XML entities like &amp;, &quot;, &#39;
    text = html.unescape(text)
    # Normalize multiple whitespace / newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    os.makedirs(PRIVATE_CSCW_DIR, exist_ok=True)
    os.makedirs(OFFICIAL_ABSTRACTS_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_CSV)
    cscw_df = df[df["doc_id"].str.startswith("cscw")].copy()
    total = len(cscw_df)
    print(f"Found {total} CSCW documents in metadata.")

    success_count = 0
    failed = []

    for idx, (_, row) in enumerate(cscw_df.iterrows()):
        doc_id = str(row["doc_id"]).strip()
        doi = str(row["doi"]).strip()

        private_path = os.path.join(PRIVATE_CSCW_DIR, f"{doc_id}.txt")
        official_path = os.path.join(OFFICIAL_ABSTRACTS_DIR, f"{doc_id}.txt")

        url = f"https://api.crossref.org/works/{doi}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SciEventDownloader/1.0 (mailto:haipd@ptit.edu)"}
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.load(resp)
                msg = data.get("message", {})
                abstract_raw = msg.get("abstract", "")
                if not abstract_raw:
                    failed.append((doc_id, doi, "Empty abstract in response"))
                    continue

                clean_text = clean_abstract_xml(abstract_raw)
                # Content must have doc_id as line 1, abstract as line 2
                file_content = f"{doc_id}\n{clean_text}\n"

                with open(private_path, "w", encoding="utf-8") as f:
                    f.write(file_content)

                with open(official_path, "w", encoding="utf-8") as f:
                    f.write(file_content)

                success_count += 1
        except Exception as e:
            failed.append((doc_id, doi, str(e)))

        if (idx + 1) % 20 == 0 or (idx + 1) == total:
            print(f"Progress: [{idx + 1}/{total}] - Successfully downloaded: {success_count}")

        time.sleep(0.05)

    print("\n==========================================")
    print(f"Crawl finished. Success: {success_count}/{total}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All 100 CSCW abstracts successfully saved!")
    print("==========================================")


if __name__ == "__main__":
    main()
