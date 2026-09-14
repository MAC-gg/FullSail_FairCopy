"""
Strips the raw collected data down to the flat, trainable format -- same output shape as other datasets

Input:  training/data/factual_opinion/raw_collected.jsonl
Output: training/data/factual_opinion_prepared.jsonl
  one JSON object per line: {"text": ..., "label": "Factual Reporting" | "Opinion/Editorial"}
"""

import json
from pathlib import Path

RAW_PATH = Path(__file__).parent / "data" / "factual_opinion" / "raw_collected.jsonl"
OUT_PATH = Path(__file__).parent / "data" / "factual_opinion_prepared.jsonl"


def prepare(raw_path: Path, out_path: Path):
    matched, skipped = 0, 0
    with open(raw_path) as f, open(out_path, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "").strip()
            label = record.get("label")
            if not text or not label:
                skipped += 1
                continue
            out.write(json.dumps({"text": text, "label": label}) + "\n")
            matched += 1

    print(f"Wrote {matched} labeled articles to {out_path} ({skipped} skipped -- missing text or label)")


if __name__ == "__main__":
    if not RAW_PATH.exists():
        raise SystemExit(
            f"{RAW_PATH} not found. Run collect_factual_opinion.py first."
        )
    prepare(RAW_PATH, OUT_PATH)