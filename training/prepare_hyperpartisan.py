"""
Parses the SemEval-2019 Task 4 by-article corpus into a flat, trainable format (jsonl)

Expected input: https://zenodo.org/records/5776081
  - unzip into training/data/hyperpartisan/
  - articles-training-byarticle-*.xml       -- article id, title, and body text
  - ground-truth-training-byarticle-*.xml   -- matching id, hyperpartisan true/false

Output: training/data/hyperpartisan_prepared.jsonl
  one JSON object per line: {"text": ..., "label": "Hyperpartisan" | "Not Hyperpartisan"}
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "hyperpartisan"
OUT_PATH = Path(__file__).parent / "data" / "hyperpartisan_prepared.jsonl"


def parse_articles(articles_path: Path) -> dict[str, str]:
    """id -> article text (all <p> text concatenated)."""
    tree = ET.parse(articles_path)
    articles = {}
    for article in tree.getroot().findall("article"):
        article_id = article.get("id")
        text = "".join(article.itertext()).strip()
        articles[article_id] = text
    return articles


def parse_ground_truth(truth_path: Path) -> dict[str, bool]:
    """id -> is hyperpartisan."""
    tree = ET.parse(truth_path)
    labels = {}
    for article in tree.getroot().findall("article"):
        article_id = article.get("id")
        labels[article_id] = article.get("hyperpartisan") == "true"
    return labels


def prepare(articles_path: Path, truth_path: Path, out_path: Path):
    articles = parse_articles(articles_path)
    labels = parse_ground_truth(truth_path)

    matched, skipped = 0, 0
    with open(out_path, "w") as out:
        for article_id, text in articles.items():
            if article_id not in labels or not text.strip():
                skipped += 1
                continue
            record = {
                "text": text,
                "label": "Hyperpartisan" if labels[article_id] else "Not Hyperpartisan",
            }
            out.write(json.dumps(record) + "\n")
            matched += 1

    print(f"Wrote {matched} labeled articles to {out_path} ({skipped} skipped -- missing text or label)")


if __name__ == "__main__":
    articles_files = sorted(DATA_DIR.glob("articles-*.xml"))
    truth_files = sorted(DATA_DIR.glob("ground-truth-*.xml"))

    if not articles_files or not truth_files:
        raise SystemExit(
            f"No source files found in {DATA_DIR}. Download and unzip the dataset "
            "from https://zenodo.org/records/5776081 into that folder first."
        )

    prepare(articles_files[0], truth_files[0], OUT_PATH)