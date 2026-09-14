"""
Parses the Webis Clickbait Corpus 2017 into a flat, trainable format (jsonl)

Expected input: https://zenodo.org/records/5530410
  - unzip into training/data/clickbait/
  - instances.jsonl  -- one JSON object per line, the tweet/teaser text + linked article
  - truth.jsonl      -- matching id -> clickbait truth class + mean score

Output: training/data/clickbait_prepared.jsonl
  one JSON object per line: {"text": ..., "label": "Clickbait" | "Not Clickbait"}
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "clickbait"
OUT_PATH = Path(__file__).parent / "data" / "clickbait_prepared.jsonl"


def load_jsonl(path: Path) -> dict[str, dict]:
    """id -> record, keyed by the corpus's "id" field."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[obj["id"]] = obj
    return records


def build_text(instance: dict) -> str:
    """
    The corpus separates the tweet teaser from the linked article
    this dataset actually measures teaser clickbaitiness, not full-article clickbaitiness
    """
    post_text = " ".join(instance.get("postText", []))
    return post_text.strip()


def prepare(instances_path: Path, truth_path: Path, out_path: Path):
    instances = load_jsonl(instances_path)
    truths = load_jsonl(truth_path)

    matched, skipped = 0, 0
    with open(out_path, "w") as out:
        for record_id, instance in instances.items():
            truth = truths.get(record_id)
            text = build_text(instance)
            if not truth or not text:
                skipped += 1
                continue
            is_clickbait = truth.get("truthClass") == "clickbait"
            record = {
                "text": text,
                "label": "Clickbait" if is_clickbait else "Not Clickbait",
            }
            out.write(json.dumps(record) + "\n")
            matched += 1

    print(f"Wrote {matched} labeled teasers to {out_path} ({skipped} skipped -- missing text or label)")


if __name__ == "__main__":
    instances_path = DATA_DIR / "instances.jsonl"
    truth_path = DATA_DIR / "truth.jsonl"

    if not instances_path.exists() or not truth_path.exists():
        raise SystemExit(
            f"instances.jsonl / truth.jsonl not found in {DATA_DIR}. Download and "
            "unzip the dataset from https://zenodo.org/records/5530410 into that folder first."
        )

    prepare(instances_path, truth_path, OUT_PATH)