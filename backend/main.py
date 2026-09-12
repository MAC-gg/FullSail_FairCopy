"""
Fair Copy - RnD Proof of Concept
Demonstrates the full technology chain end-to-end

Run locally (see README.md)
"""

import json
import re
import uuid
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from dotenv import load_dotenv
import trafilatura
from bs4 import BeautifulSoup
from collections import Counter


load_dotenv()
JINA_API_KEY = os.environ.get("JINA_API_KEY")

# API setup
app = FastAPI(title="Fair Copy RnD")

# Allow the local frontend (served separately) to call this API during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predetermined labels
LABELS = ["Factual Reporting", "Opinion/Editorial", "Hyperpartisan", "Clickbait"]

# In-memory store for the RnD demo
_SESSIONS: dict[str, dict] = {}

# lazy-load some packages
_classifier = None
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

def classify_text(text: str) -> dict[str, float]:
    result = get_classifier()(text, LABELS, multi_label=True)
    return {str(label): float(score) for label, score in zip(result["labels"], result["scores"])}

# splitting sentences and words
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    return _nlp

def split_sentences(text: str) -> list[str]:
    all_sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        doc = get_nlp()(paragraph)
        all_sentences.extend(s.text.strip() for s in doc.sents)

    return [s for s in all_sentences if len(s.split()) >= 4]

def word_breakdown(text: str, top_n: int = 15) -> list[dict]:
    nlp = get_nlp()
    doc = nlp(text)
    words = [
        token.text.lower() for token in doc
        if token.is_alpha and not nlp.vocab[token.text.lower()].is_stop and len(token.text) > 2
    ]
    counts = Counter(words)
    return [{"word": word, "count": count} for word, count in counts.most_common(top_n)]

# photo credit setup
PHOTO_CREDIT_PATTERN = re.compile(r'\([A-Za-z][A-Za-z\s]*(?:Photo|Images?)/[^)]*\)')

def extract_photo_credits(html: str) -> list[str]:
    """
    By regex, collects all photo credit to be displayed
    """
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(separator=" ")
    return list(dict.fromkeys(PHOTO_CREDIT_PATTERN.findall(full_text)))


def extract_article(html: str) -> tuple[str | None, str]:
    """
    Runs trafilatura against raw HTML to find the article body algorithmically
    Scores DOM blocks on tag type, link density, and known boilerplate patterns
    rather than filtering by text length that can't tell between a sentence from a linked headline
    """
    extracted = trafilatura.extract(html, with_metadata=True, output_format="json", favor_precision=True, include_comments=False)
    if not extracted:
        return None, ""
    data = json.loads(extracted)
    title = data.get("title")
    body = data.get("text", "")

    # traffy sometimes includes the headline as the body's first line
    # instead strip it out
    if title and body.strip().startswith(title.strip()):
        body = body.strip()[len(title.strip()):].strip()

    return title, body


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
    article_title = None
    photo_credits = []

    # HEADSUP: URL would overwrite text if both are provided
    if req.url:
        # get full HTML
        headers = {"X-Return-Format": "html"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"
        
        # External parsing API call -- r.jina.ai extracts clean article text from a URL
        try:
            resp = requests.get(f"https://r.jina.ai/{req.url}", headers=headers, timeout=20)
        except requests.Timeout:
            raise HTTPException(504, "The article-parsing service timed out. Try again.")
        except requests.RequestException as e:
            raise HTTPException(502, f"Could not reach the article-parsing service: {e}")

        # Error handling for Jina
        if resp.status_code == 401:
            raise HTTPException(502, "Article-parsing API key was rejected -- check JINA_API_KEY in .env.")
        elif resp.status_code == 429:
            raise HTTPException(429, "Article-parsing service rate limit hit. Wait a moment and try again.")
        elif resp.status_code == 403:
            raise HTTPException(502, "The article-parsing service was denied access to this URL.")
        elif not resp.ok:
            raise HTTPException(502, f"Article-parsing service returned an unexpected error ({resp.status_code}).")

        article_text = resp.text

        # collect photo credits
        photo_credits = extract_photo_credits(article_text)

        # run traffy to get the article from the HTML
        article_title, article_text = extract_article(article_text)


    if not article_text or len(article_text.strip()) < 20:
        raise HTTPException(400, "Not enough article text to classify.")

    baseline_scores = classify_text(article_text)
    title_scores = classify_text(article_title) if article_title else None
    sentences = split_sentences(article_text)
    word_counts = word_breakdown(article_text)
    poss_trunc = len(sentences) < 10

    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "text": article_text,
        "sentences": sentences,
        "baseline": baseline_scores,
        "title": article_title,
        "title_scores": title_scores,
        "poss_trunc": poss_trunc,
        "photo_credits": photo_credits,
        "word_counts": word_counts,
    }

    return {
        "id": session_id,
        "title": article_title,
        "labels": [{"name": name, "confidence": round(score, 4)} for name, score in baseline_scores.items()],
        "title_labels": (
            [{"name": name, "confidence": round(score, 4)} for name, score in title_scores.items()]
            if title_scores else None
        ),
        "sentence_count": len(sentences),
        "poss_trunc": poss_trunc,
        "photo_credits": photo_credits,
        "word_counts": word_counts,
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