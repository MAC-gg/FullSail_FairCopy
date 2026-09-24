"""
Fair Copy
Run locally (see README.md)
"""

import json
import re
import uuid
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os
from dotenv import load_dotenv
import trafilatura
from bs4 import BeautifulSoup
from collections import Counter
from pathlib import Path


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
LABELS = ["factual", "opinion", "hyperpartisan", "clickbait"]
MIN_WORDS = 20
MIN_SENTENCES = 5
# chunking settings
MAX_CHUNKS = 3
CHUNK_TOKEN_LIMIT = 500 # has enough headroom to be filled

# memory stuff
_SESSIONS: dict[str, dict] = {}
SERVER_BOOT_ID = str(uuid.uuid4())

# lazy-load models
MODELS_DIR = Path(__file__).parent.parent / "training" / "models"
_hyperpartisan_model = None
_clickbait_model = None
_factual_opinion_model = None


def get_hyperpartisan_classifier():
    global _hyperpartisan_model
    if _hyperpartisan_model is None:
        from transformers import pipeline
        _hyperpartisan_model = pipeline(
            "text-classification", model="mackyoop/fair-copy-hyperpartisan", top_k=None,
        )
    return _hyperpartisan_model


def get_clickbait_classifier():
    global _clickbait_model
    if _clickbait_model is None:
        from transformers import pipeline
        _clickbait_model = pipeline(
            "text-classification", model="mackyoop/fair-copy-clickbait", top_k=None,
        )
    return _clickbait_model


def get_factual_opinion_classifier():
    global _factual_opinion_model
    if _factual_opinion_model is None:
        from transformers import pipeline
        _factual_opinion_model = pipeline(
            "text-classification", model="mackyoop/fair-copy-factual-opinion", top_k=None,
        )
    return _factual_opinion_model


def classify_text(text: str) -> dict[str, float]:
    hp_result = {r["label"]: r["score"] for r in get_hyperpartisan_classifier()(text, truncation=True, max_length=512)[0]}
    cb_result = {r["label"]: r["score"] for r in get_clickbait_classifier()(text, truncation=True, max_length=512)[0]}
    fo_result = {r["label"]: r["score"] for r in get_factual_opinion_classifier()(text, truncation=True, max_length=512)[0]}

    hyperpartisan_score = hp_result.get("LABEL_1", hp_result.get("Hyperpartisan", 0.0))
    clickbait_score = cb_result.get("LABEL_1", cb_result.get("Clickbait", 0.0))
    opinion_score = fo_result.get("LABEL_1", fo_result.get("Opinion/Editorial", 0.0))
    factual_score = 1.0 - opinion_score

    return {
        "factual": factual_score,
        "opinion": opinion_score,
        "hyperpartisan": hyperpartisan_score,
        "clickbait": clickbait_score,
    }

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

    all_words = [token.text for token in doc if token.is_alpha and len(token.text) > 2]

    filtered_words = [
        token.text.lower() for token in doc
        if token.is_alpha and not nlp.vocab[token.text.lower()].is_stop and len(token.text) > 2
    ]
    counts = Counter(filtered_words)

    breakdown = [{"word": word, "count": count} for word, count in counts.most_common(top_n)]
    breakdown.insert(0, {"word": "Words", "count": len(all_words)})
    return breakdown

# WINDOW SENTENCE IMPACT
WINDOW_SIZE = 5
SENTENCE_IMPACT_METHOD = "three_window" # three_window or single_window

def build_window(sentences: list[str], target_index: int, anchor: str) -> list[str]:
    """
    Builds a WINDOW_SIZE-sentence window containing the target sentence,
    positioned at 'start', 'end', or 'center' of the window where possible
    """
    n = len(sentences)
    if anchor == "start":
        start = target_index
    elif anchor == "end":
        start = target_index - WINDOW_SIZE + 1
    else:  # center
        start = target_index - WINDOW_SIZE // 2

    start = max(0, min(start, n - WINDOW_SIZE))
    return sentences[start:start + WINDOW_SIZE]


def impacts_three_window(sentences: list[str], target_index: int, baseline: dict) -> dict:
    """
    Method A: builds 3 windows containing the target sentence (positioned at
    the start, center, and end of a 5-sentence window)
    """
    window_scores = []
    for anchor in ("start", "center", "end"):
        window = build_window(sentences, target_index, anchor)
        window_scores.append(classify_text(" ".join(window)))

    avg_scores = {
        label: sum(ws[label] for ws in window_scores) / len(window_scores)
        for label in LABELS
    }
    return {label: round(avg_scores[label] - baseline[label], 4) for label in LABELS}


def impacts_single_window(sentences: list[str], target_index: int, baseline: dict) -> dict:
    """
    Method B: builds one centered window containing the target sentence
    """
    window = build_window(sentences, target_index, "center")

    # Find the target sentence's position within this window to remove it
    # (target_index is the article-wide index; window is a slice, so we
    # locate it by matching the sentence text itself)
    target_sentence = sentences[target_index]
    window_with = list(window)
    window_without = [s for s in window if s != target_sentence]

    with_scores = classify_text(" ".join(window_with))
    without_scores = classify_text(" ".join(window_without)) if window_without else {l: 0.0 for l in LABELS}

    return {label: round(with_scores[label] - without_scores[label], 4) for label in LABELS}


# CHUNKING
_shared_tokenizer = None
def get_shared_tokenizer():
    global _shared_tokenizer
    if _shared_tokenizer is None:
        from transformers import AutoTokenizer
        _shared_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return _shared_tokenizer


def pack_sentences_into_chunks(sentences: list[str]) -> tuple[list[list[str]], bool]:
    """
    Greedily packs sentences into up to MAX_CHUNKS chunks, each staying under CHUNK_TOKEN_LIMIT tokens.
    
    Returns (chunks, overflowed); overflowed is True if content remained after MAX_CHUNKS chunks were filled.
    """
    tokenizer = get_shared_tokenizer()
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current and current_tokens + sentence_tokens > CHUNK_TOKEN_LIMIT:
            chunks.append(current)
            current, current_tokens = [], 0
            if len(chunks) >= MAX_CHUNKS:
                return chunks, True

        current.append(sentence)
        current_tokens += sentence_tokens
        i += 1

    if current:
        chunks.append(current)

    return chunks, False


# photo credit setup
PHOTO_CREDIT_PATTERN = re.compile(r'\([A-Za-z][A-Za-z\s]*(?:Photo|Images?)/[^)]*\)')

def extract_photo_credits(html: str) -> list[str]:
    """
    By regex, collects all photo credit to be displayed
    """
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(separator=" ")
    return list(dict.fromkeys(PHOTO_CREDIT_PATTERN.findall(full_text)))


def extract_article(html: str) -> dict:
    """
    Run traffy to get article details including title, author, date, and content

    Input: html string from article URL

    Output: article object including title, author, date, content
    """
    extracted = trafilatura.extract(
        html, with_metadata=True, output_format="json",
        favor_precision=True, include_comments=False,
    )
    if not extracted:
        return {"title": None, "author": None, "date": None, "text": ""}

    data = json.loads(extracted)
    title = data.get("title")
    body = data.get("text", "")

    if title and body.strip().startswith(title.strip()):
        body = body.strip()[len(title.strip()):].strip()

    return {"title": title, "author": data.get("author"), "date": data.get("date"), "text": body}


class ClassifyRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "boot_id": SERVER_BOOT_ID}


@app.post("/api/classify/prepare")
def prepare_classification(req: ClassifyRequest):
    if not req.text and not req.url:
        raise HTTPException(400, "Provide either 'text' or 'url'.")

    article_text = req.text
    article_title = None
    article_author = None
    article_date = None
    photo_credits = []

    submitted_type = "URL" if req.url else "Text"
    submitted_content_raw = req.url if req.url else req.text

    if req.url:
        headers = {"X-Return-Format": "html"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"

        try:
            resp = requests.get(f"https://r.jina.ai/{req.url}", headers=headers, timeout=20)
        except requests.Timeout:
            raise HTTPException(504, "The article-parsing service timed out. Try again.")
        except requests.RequestException as e:
            raise HTTPException(502, f"Could not reach the article-parsing service: {e}")

        if resp.status_code == 401:
            raise HTTPException(502, "Article-parsing API key was rejected -- check JINA_API_KEY in .env.")
        elif resp.status_code == 429:
            raise HTTPException(429, "Article-parsing service rate limit hit. Wait a moment and try again.")
        elif resp.status_code == 403:
            raise HTTPException(502, "The article-parsing service was denied access to this URL.")
        elif not resp.ok:
            raise HTTPException(502, f"Article-parsing service returned an unexpected error ({resp.status_code}).")

        raw_html = resp.text
        photo_credits = extract_photo_credits(raw_html)
        article_meta = extract_article(raw_html)
        article_title = article_meta["title"]
        article_author = article_meta["author"]
        article_date = article_meta["date"]
        article_text = article_meta["text"]

    if not article_text or len(article_text.strip()) < 20:
        raise HTTPException(400, "Not enough article text to classify.")

    sentences = split_sentences(article_text)

    if len(sentences) < MIN_SENTENCES:
        raise HTTPException(
            400,
            f"Article has too few real sentences ({len(sentences)}, minimum {MIN_SENTENCES}) for reliable analysis."
        )

    chunks, overflowed = pack_sentences_into_chunks(sentences)
    if overflowed:
        raise HTTPException(
            400,
            f"Article is too long -- it doesn't fit within {MAX_CHUNKS} model passes "
            f"({MAX_CHUNKS * CHUNK_TOKEN_LIMIT} tokens). Try a shorter article."
        )

    word_counts = word_breakdown(article_text, top_n=10)
    possibly_truncated = len(sentences) < 10

    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "text": article_text,
        "sentences": sentences,
        "chunks": chunks,
        "chunk_scores": [None] * len(chunks),
        "baseline": None,
        "title_scores": None,
        "title": article_title,
        "author": article_author,
        "date": article_date,
        "possibly_truncated": possibly_truncated,
        "photo_credits": photo_credits,
        "word_counts": word_counts,
        "submitted_type": submitted_type,
        "submitted_content_raw": submitted_content_raw,
    }

    return {
        "id": session_id,
        "chunk_count": len(chunks),
        "title": article_title,
        "author": article_author,
        "date": article_date,
        "sentence_count": len(sentences),
        "possibly_truncated": possibly_truncated,
        "photo_credits": photo_credits,
        "word_counts": word_counts,
        "submitted_type": submitted_type,
        "submitted_content_raw": submitted_content_raw,
    }


@app.post("/api/classify/{session_id}/chunk/{chunk_index}")
def classify_chunk(session_id: str, chunk_index: int):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Unknown session id -- call /prepare first.")
    if chunk_index < 0 or chunk_index >= len(session["chunks"]):
        raise HTTPException(400, "Invalid chunk index.")

    chunk_text = " ".join(session["chunks"][chunk_index])
    session["chunk_scores"][chunk_index] = classify_text(chunk_text)

    return {"chunk_index": chunk_index, "done": True}


@app.post("/api/classify/{session_id}/finalize")
def finalize_classification(session_id: str):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Unknown session id -- call /prepare first.")
    if any(score is None for score in session["chunk_scores"]):
        raise HTTPException(400, "Not all chunks have been classified yet.")

    chunk_scores = session["chunk_scores"]
    baseline_scores = {
        label: sum(cs[label] for cs in chunk_scores) / len(chunk_scores)
        for label in LABELS
    }
    session["baseline"] = baseline_scores

    title_scores = classify_text(session["title"]) if session["title"] else None
    session["title_scores"] = title_scores

    return {
        "id": session_id,
        "title": session["title"],
        "author": session["author"],
        "date": session["date"],
        "labels": [{"name": name, "confidence": round(score, 4)} for name, score in baseline_scores.items()],
        "title_labels": (
            [{"name": name, "confidence": round(score, 4)} for name, score in title_scores.items()]
            if title_scores else None
        ),
        "sentence_count": len(session["sentences"]),
        "possibly_truncated": session["possibly_truncated"],
        "photo_credits": session["photo_credits"],
        "word_counts": session["word_counts"],
        "submitted_type": session["submitted_type"],
        "submitted_content_raw": session["submitted_content_raw"],
    }


@app.get("/api/classify/{session_id}/sentences")
def sentence_impacts(session_id: str, offset: int = 0, limit: int = 5):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Unknown session id -- classify an article first.")

    sentences = session["sentences"]
    baseline = session["baseline"]
    batch = sentences[offset: offset + limit]

    results = []
    for i, sentence in enumerate(batch, start=offset):
        #### ORIGINAL FULL-ARTICLE ABLATION
        # remaining = " ".join(s for j, s in enumerate(sentences) if j != i)
        # modified_scores = classify_text(remaining) if remaining.strip() else {l: 0.0 for l in LABELS}
        # impacts = {
        #     label: round(baseline[label] - modified_scores.get(label, 0.0), 4)
        #     for label in LABELS
        # }
        
        if SENTENCE_IMPACT_METHOD == "single_window":
            impacts = impacts_single_window(sentences, i, baseline)
        else:
            impacts = impacts_three_window(sentences, i, baseline)

        results.append({"index": i, "sentence": sentence, "impacts": impacts})

    return {
        "results": results,
        "next_offset": offset + limit if offset + limit < len(sentences) else None,
        "total_sentences": len(sentences),
    }


@app.get("/api/classify/{session_id}")
def get_shared_report(session_id: str):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(
            404,
            "This shared report is no longer available -- the server may have restarted since it was created."
        )

    return {
        "id": session_id,
        "title": session["title"],
        "author": session.get("author"),
        "date": session.get("date"),
        "labels": [{"name": name, "confidence": round(score, 4)} for name, score in session["baseline"].items()],
        "title_labels": (
            [{"name": name, "confidence": round(score, 4)} for name, score in session["title_scores"].items()]
            if session.get("title_scores") else None
        ),
        "sentence_count": len(session["sentences"]),
        "possibly_truncated": session.get("possibly_truncated"),
        "photo_credits": session.get("photo_credits", []),
        "word_counts": session.get("word_counts", []),
        "submitted_type": session.get("submitted_type"),
        "submitted_content_raw": session.get("submitted_content_raw"),
    }


# serve frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")