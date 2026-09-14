"""
Pushes each fine-tuned model to Hugging Face Hub
Run once per model, after training is finalized
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF_USERNAME = "mackyoop" # my Hugging Face username

MODELS = ["hyperpartisan", "clickbait", "factual_opinion"]

for name in MODELS:
    path = f"./models/{name}/final"
    repo_id = f"{HF_USERNAME}/fair-copy-{name.replace('_', '-')}"

    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"Pushed {name} -> {repo_id}")