"""
Fair Copy - RnD Proof of Concept
Demonstrates the full technology chain end-to-end

Run locally (see README.md)
"""

import re
import uuid
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Fair Copy RnD")

# Allow the local frontend (served separately) to call this API during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predetermined label set (4+), per the pitch/PP4 spec.
LABELS = ["Factual Reporting", "Opinion/Editorial", "Hyperpartisan", "Clickbait"]

# In-memory store for the RnD demo
_SESSIONS: dict[str, dict] = {}

# lazy-load some packages
_classifier = None
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    return _nlp


def get_classifier():
    """
    Lazily loads a zero-shot classification pipeline.

    RnD NOTE: the production version of Fair Copy fine-tunes a transformer on
    labeled datasets (SemEval Hyperpartisan News, Webis Clickbait Corpus).
    That takes real training time and data prep that doesn't fit an RnD proof
    of concept. Zero-shot classification (facebook/bart-large-mnli) is used
    here instead to prove the model-inference link in the chain works end to
    end without requiring a training run first.
    """
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _classifier


def split_sentences(text: str) -> list[str]:
    doc = get_nlp()(text.strip())
    raw = [sent.text.strip() for sent in doc.sents]

    # Drop short fragments (nav links, bare numbers, etc) that aren't real sentences -- a genuine sentence is rarely under ~4 words
    return [s for s in raw if len(s.split()) >= 4]


def strip_markdown_links(text: str) -> str:
    # Jina returns markdown. Images add no classifiable text, so drop them entirely
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)             # ![alt](url) -> removed
    text = re.sub(r'\[([^\]]*)\]\(([^)]*)\)', r'\1', text)  # [text](url) -> text
    return text


def classify_text(text: str) -> dict[str, float]:
    result = get_classifier()(text, LABELS, multi_label=True)
    return dict(zip(result["labels"], result["scores"]))


class ClassifyRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    if not req.text and not req.url:
        raise HTTPException(400, "Provide either 'text' or 'url'.")

    article_text = req.text

    # HEADSUP: URL would overwrite text if both are provided
    if req.url:
        # External parsing API call -- r.jina.ai extracts clean article text from a URL
        try:
            resp = requests.get(f"https://r.jina.ai/{req.url}", timeout=20)
            resp.raise_for_status()
            article_text = resp.text
        except requests.RequestException as e:
            raise HTTPException(502, f"Could not extract article from URL: {e}")

        # Some publishers block scrapers (403/CAPTCHA/anti-bot pages)
        failure_markers = ("Warning: Target URL returned error", "Access to this page has been denied")
        if any(marker in article_text for marker in failure_markers):
            raise HTTPException(
                502,
                "The source site blocked automated access to this article "
                "(bot/CAPTCHA protection). Try pasting the article text directly instead."
            )

        # strip out any markdown that Jina returns
        article_text = strip_markdown_links(article_text)

    if not article_text or len(article_text.strip()) < 20:
        raise HTTPException(400, "Not enough article text to classify.")

    baseline_scores = classify_text(article_text)
    sentences = split_sentences(article_text)

    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "text": article_text,
        "sentences": sentences,
        "baseline": baseline_scores,
    }

    return {
        "id": session_id,
        "labels": [{"name": name, "confidence": round(score, 4)} for name, score in baseline_scores.items()],
        "sentence_count": len(sentences),
    }


@app.get("/api/classify/{session_id}/sentences")
def sentence_impacts(session_id: str, offset: int = 0, limit: int = 10):
    """
    Sentence-ablation loop: for each sentence in the requested batch, removes
    it from the article, re-classifies the remainder, and reports how much
    each label's confidence shifted. Computed on demand (not upfront) so cost
    scales with what the user actually views, not the whole article.
    """
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Unknown session id -- classify an article first.")

    sentences = session["sentences"]
    baseline = session["baseline"]
    batch = sentences[offset: offset + limit]

    results = []
    for i, sentence in enumerate(batch, start=offset):
        remaining = " ".join(s for j, s in enumerate(sentences) if j != i)
        modified_scores = classify_text(remaining) if remaining.strip() else {l: 0.0 for l in LABELS}
        impacts = {
            label: round(baseline[label] - modified_scores.get(label, 0.0), 4)
            for label in LABELS
        }
        results.append({"index": i, "sentence": sentence, "impacts": impacts})

    return {
        "results": results,
        "next_offset": offset + limit if offset + limit < len(sentences) else None,
        "total_sentences": len(sentences),
    }