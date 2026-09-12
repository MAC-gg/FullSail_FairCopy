"""
Batch-tests a list of publisher URLs through Jina's reader to gather data on current block status
Results feed into backend/pub_pros.py by hand once patterns are confirmed.
"""
import os
import time
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv("../backend/.env")
JINA_API_KEY = os.environ.get("JINA_API_KEY")

TEST_URLS = [
    # "https://www.apnews.com/article/iran-trump-economy-inflation-unemployment-diesel-720096ea610880ddec66561c29cdc283",
    # "https://www.time.com/article/2026/09/11/25-years-after-9-11-americans-must-remember-what-binds-us/",
    # "https://www.nytimes.com/live/2026/09/11/nyregion/9-11-anniversary-25th",
    # "https://www.theguardian.com/us-news/2026/sep/11/september-11-commemorations",
    # "https://www.washingtonpost.com/politics/2026/09/11/pentagon-trump-edges-toward-embracing-post-911-policies-he-attacked/",
    # "https://www.reuters.com/world/us/americans-mark-25-years-since-september-11-attacks-amid-enduring-grief-2026-09-11/",
    # "https://www.bbc.com/news/articles/clyjqvjzlwno",
    # "https://www.npr.org/2026/09/04/nx-s1-5921394/first-responder-9-11-25-years-anniversary",
    # "https://www.theatlantic.com/ideas/2026/09/bush-trump-mamdani-giuliani-islam/688582/",
    # "https://www.cnn.com/2026/09/12/us/twin-towers-9-11-dust-chemicals",
    # "https://www.foxnews.com/opinion/universes-beautiful-fine-tuning-leaves-atheism-facing-biggest-question",
    "https://www.wsj.com/world/europe/this-spanish-island-wants-to-ditch-drunk-british-tourists-for-rich-american-ones-c1f6c0fd?mod=hp_lead_pos9",
    "https://www.nbcnews.com/health/cancer/-get-cancer-yet-25-years-911-people-are-getting-sick-rcna596778",
    "https://www.cbsnews.com/news/cia-declassified-9-11-files-intelligence-reports/",
    "https://www.huffpost.com/entry/cia-releases-dozens-declassified-documents-related-to-911_n_6aa4775ae4b09fd4319f5f21?origin=home-whats-happening-unit",
    "https://www.politico.com/news/2026/09/11/political-unity-permeates-25th-anniversary-of-9-11-01073118",
    "https://www.abcnews.com/US/911-now-photos-show-world-trade-center-area/story?id=136340214",
    "https://www.bloomberg.com/features/2026-iran-internet/?srnd=homepage-americas",
    "https://www.pbs.org/newshour/show/americans-remember-the-lives-lost-in-the-9-11-attacks-25-years-ago",
    "https://www.newsnationnow.com/us-news/sept11-anniversary/declassified-9-11-intelligence-briefs/",
    "https://www.latimes.com/california/story/2026-09-12/ucla-chancellor-law-school-9-11-event",
    "https://www.usnews.com/news/national-news/articles/2026-09-11/25-years-later-ground-zeros-toxic-legacy-is-still-claiming-lives",
    "https://www.usatoday.com/story/news/politics/2026/09/12/trump-family-melania-don-jr-ireland-convention/91729648007/",
    "https://www.oann.com/commentary/will-radical-islam-again-express-joy-about-9-11/",
    "https://www.newsmax.com/politics/giuliani-reveals-what-he-told-mamdani-in-911-handshake/2026/09/11/id/1269192/",

]

TRUNCATION_SENTENCE_THRESHOLD = 10

_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    return _nlp

def count_sentences(text: str) -> int:
    doc = get_nlp()(text.strip())
    return len([s for s in doc.sents if len(s.text.split()) >= 4])

def test_url(url: str) -> dict:
    headers = {}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    try:
        resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=20)
    except requests.RequestException as e:
        return {"url": url, "status": "error", "sentence_count": None, "truncated": None, "detail": str(e)}

    if resp.status_code != 200:
        return {"url": url, "status": resp.status_code, "sentence_count": None, "truncated": None}

    sentence_count = count_sentences(resp.text)
    truncated = sentence_count < TRUNCATION_SENTENCE_THRESHOLD

    return {
        "url": url,
        "status": resp.status_code,
        "sentence_count": sentence_count,
        "truncated": truncated,
    }


if __name__ == "__main__":
    for url in TEST_URLS:
        result = test_url(url)
        domain = urlparse(url).netloc.replace("www.", "")
        print(f"{domain:25} status={result['status']}  sentences={result.get('sentence_count')}  "
              f"truncated={result.get('truncated')}")
        time.sleep(1)  # be a little polite to the API