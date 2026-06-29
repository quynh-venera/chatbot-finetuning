"""
preview_data.py

Quick utility to print the first N records from train.jsonl, val.jsonl,
and test.jsonl in a readable format (not raw JSON dumps).

Usage:
    python preview_data.py
    python preview_data.py --n 10
"""

import argparse
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]


def preview_file(path: Path, n: int):
    if not path.exists():
        print(f"  (file not found: {path})")
        return

    print(f"\n{'=' * 70}")
    print(f"{path.name}")
    print(f"{'=' * 70}")

    with path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    total = len(lines)
    print(f"Total records: {total}\n")

    for i, line in enumerate(lines[:n]):
        record = json.loads(line)
        messages = record.get("messages", [])

        print(f"--- Record {i + 1} ---")
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            # Truncate long content for readability; full text is still in the file.
            preview = content if len(content) <= 300 else content[:300] + " […]"
            print(f"  [{role}] {preview}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Preview train/val/test JSONL files.")
    parser.add_argument("--n", type=int, default=5, help="Number of records to show per file (default: 5)")
    args = parser.parse_args()

    for filename in FILES:
        preview_file(PROCESSED_DIR / filename, args.n)


if __name__ == "__main__":
    main()
