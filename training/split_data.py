"""
Splits each *_prepared.jsonl into train/validation sets
- stratified so both splits preserve the source file's original class ratio

Output: <name>_train.jsonl and <name>_val.jsonl alongside each source file.
"""

import json
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
VAL_FRACTION = 0.15
RANDOM_SEED = 42


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(records: list[dict], path: Path):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def split_dataset(name: str):
    source_path = DATA_DIR / f"{name}_prepared.jsonl"
    if not source_path.exists():
        print(f"Skipping {name} -- {source_path} not found.")
        return

    records = load_records(source_path)
    labels = [r["label"] for r in records]

    train_records, val_records = train_test_split(
        records,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=labels,  # preserves class ratio in both splits
    )

    write_records(train_records, DATA_DIR / f"{name}_train.jsonl")
    write_records(val_records, DATA_DIR / f"{name}_val.jsonl")

    print(f"{name}: {len(train_records)} train / {len(val_records)} val")
    print(f"  train distribution: {Counter(r['label'] for r in train_records)}")
    print(f"  val distribution:   {Counter(r['label'] for r in val_records)}")


if __name__ == "__main__":
    split_dataset("hyperpartisan")
    split_dataset("clickbait")
    split_dataset("factual_opinion")