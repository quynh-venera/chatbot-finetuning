"""
merge.py

OLD / SIMPLER merge script. Superseded by processor.py's run_processing(),
which produces a proper train/val/test split plus stats. This file is
kept around for reference only — run_batch.py does NOT call it anymore.

(Historical note: this only ever produced a single train.jsonl with no
val/test split, which is why it was replaced.)
"""

import json
from pathlib import Path

RAW_SYNTHETIC_DIR = Path("data/raw/synthetic")
PROCESSED_DIR = Path("data/processed")


def run_merge() -> dict:
    """Legacy merge: dumps all synthetic pairs into a single train.jsonl.
    No cleaning, no chunking, no dedup, no val/test split."""
    files = sorted(RAW_SYNTHETIC_DIR.glob("synthetic_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No synthetic files found in {RAW_SYNTHETIC_DIR}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "train.jsonl"

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for path in files:
            with path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    pair = json.loads(line)
                    record = {
                        "messages": [
                            {"role": "user", "content": pair.get("question", "")},
                            {"role": "assistant", "content": pair.get("answer", "")},
                        ]
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

    # NOTE: legacy key name 'total_pairs' — this mismatch with the newer
    # 'total' key used by processor.py is what caused run_batch.py's
    # summary log line to crash before the fix.
    return {"total_pairs": count}


if __name__ == "__main__":
    run_merge()
