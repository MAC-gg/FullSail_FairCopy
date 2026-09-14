"""
WARNING: consumes Jina tokens, so run deliberately and sparingly when new data is needed

Collects article text for the Factual Reporting vs. Opinion/Editorial labels via distant supervision: pulls URLs from each publisher's own RSS feeds (Opinion vs. straight-news sections), fetches each through the same Jina + trafilatura pipeline the app itself uses, and labels by which feed the article came from.

Rate-paced well under Jina's documented 500 RPM (free API key tier) -- default here is conservative, adjustable via REQUESTS_PER_MINUTE.

Checkpointed: writes each result immediately, so an interrupted run can be resumed without re-collecting or losing progress. Tracks the most recent published date seen per feed, so a future run can skip anything already collected (the "avoid overlap" idea) once that logic is added.

Skips content that is too short, but consumes tokens to determine this. Skips content that looks like a video, live coverage, podcast, or interactive feature (which are usually too short to use anyway)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import feedparser
import requests
import trafilatura
from dotenv import load_dotenv

load_dotenv("../backend/.env")
JINA_API_KEY = os.environ.get("JINA_API_KEY")

DATA_DIR = Path(__file__).parent / "data" / "factual_opinion"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = DATA_DIR / "raw_collected.jsonl"
STATE_PATH = DATA_DIR / "scan_state.json"

ARTICLES_PER_FEED = 100
REQUESTS_PER_MINUTE = 60

FEEDS = {
    "Opinion/Editorial": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Opinion.xml",
        "https://feeds.washingtonpost.com/rss/opinions",
        "https://www.theguardian.com/commentisfree/rss",
        "https://www.latimes.com/opinion/rss2.0.xml",
    ],
    "Factual Reporting": [
        "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
        "https://feeds.washingtonpost.com/rss/politics",
        "https://www.theguardian.com/world/rss",
        "https://www.latimes.com/world-nation/rss2.0.xml",
    ],
}

# found these on an initial scan of the feeds, but they are not articles, so skip them
SKIP_URL_PATTERNS = ("/video/", "/live/", "/podcasts/", "/interactive/")

def looks_like_article(url: str) -> bool:
    return not any(pattern in url for pattern in SKIP_URL_PATTERNS)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def extract_via_jina(url: str) -> tuple[str | None, str]:
    # limit Jina's response to avoid spending so many tokens
    headers = {
        "X-Return-Format": "html",
        "X-Remove-Selector": "nav, header, footer, aside, .sidebar, .newsletter, .related, .ad, .advertisement",
    }
    # auth
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    # call
    try:
        resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [skip] fetch failed for {url}: {e}")
        return None, ""
    # just like the main app
    extracted = trafilatura.extract(
        resp.text, with_metadata=True, output_format="json",
        favor_precision=True, include_comments=False,
    )

    if not extracted:
        return None, ""
    data = json.loads(extracted)

    # return both title and body
    return data.get("title"), data.get("text", "")


def collect():
    state = load_state()
    seen_urls = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            seen_urls.add(json.loads(line)["url"])
        print(f"Resuming -- {len(seen_urls)} articles already collected.")

    delay = 60.0 / REQUESTS_PER_MINUTE

    with open(OUT_PATH, "a") as out:
        for label, feed_urls in FEEDS.items():
            for feed_url in feed_urls:
                print(f"\n=== {label}: {feed_url} ===")
                parsed = feedparser.parse(feed_url)

                # feed check for empty or dead feeds
                if not parsed.entries:
                    print("  [warn] no entries found -- feed URL may be dead, verify manually.")
                    continue

                latest_seen = state.get(feed_url)
                collected_this_feed = 0

                for entry in parsed.entries:
                    if collected_this_feed >= ARTICLES_PER_FEED:
                        break
                    url = entry.get("link")

                    # article check - break out of loop if seen or not an article
                    if not url or url in seen_urls:
                        continue
                    if not looks_like_article(url):
                        continue

                    # collect date of article
                    published = entry.get("published", entry.get("updated", None))
                    # extract content and title
                    title, text = extract_via_jina(url)
                    # delay to stay under rate limit
                    time.sleep(delay)

                    # length check - skip if too short
                    if not text or len(text.split()) < 50:
                        print(f"  [skip] too little content: {url}")
                        continue

                    # create record and write to file
                    record = {
                        "text": text,
                        "label": label,
                        "url": url,
                        "title": title,
                        "published": published,
                        "collected_at": datetime.utcnow().isoformat(),
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    seen_urls.add(url)
                    collected_this_feed += 1

                    if latest_seen is None or (published and published > latest_seen):
                        latest_seen = published

                    print(f"  [{collected_this_feed}/{ARTICLES_PER_FEED}] {url}")

                if latest_seen:
                    state[feed_url] = latest_seen
                save_state(state)

    print(f"\nDone. Total collected this run: {len(seen_urls)}")


if __name__ == "__main__":
    collect()