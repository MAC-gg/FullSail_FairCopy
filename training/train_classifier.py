"""
Fine-tunes DistilBERT as a binary classifier for either the Hyperpartisan or Clickbait dataset
Run once per dataset:

    python train_classifier.py --dataset hyperpartisan
    python train_classifier.py --dataset clickbait

Output: Saves the trained model to training/models/<dataset>/
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
BASE_MODEL = "distilbert-base-uncased"

DATASET_CONFIG = {
    "hyperpartisan": {
        "label_map": {"Not Hyperpartisan": 0, "Hyperpartisan": 1},
        "epochs": 4,
    },
    "clickbait": {
        "label_map": {"Not Clickbait": 0, "Clickbait": 1},
        "epochs": 2,
    },
    "factual_opinion": {
        "label_map": {"Factual Reporting": 0, "Opinion/Editorial": 1},
        "epochs": 4, 
    },
}


def load_jsonl(path: Path, label_map: dict) -> Dataset:
    records = [json.loads(line) for line in open(path) if line.strip()]
    texts = [r["text"] for r in records]
    labels = [label_map[r["label"]] for r in records]
    return Dataset.from_dict({"text": texts, "label": labels})


class WeightedTrainer(Trainer):
    """
    add weights to deal with imbalance issue in each dataset
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main(dataset_name: str):
    config = DATASET_CONFIG[dataset_name]
    label_map = config["label_map"]

    train_ds = load_jsonl(DATA_DIR / f"{dataset_name}_train.jsonl", label_map)
    val_ds = load_jsonl(DATA_DIR / f"{dataset_name}_val.jsonl", label_map)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_labels = train_ds["label"]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    output_dir = MODELS_DIR / dataset_name
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    print(f"\n=== Final evaluation on {dataset_name} validation set ===")
    print(trainer.evaluate())

    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved model + tokenizer to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hyperpartisan", "clickbait", "factual_opinion"], required=True)
    args = parser.parse_args()
    main(args.dataset)