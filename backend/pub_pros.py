"""
Publisher Profiles -- internal registry, not user-facing

Maps a domain to a known-good CSS selector for its article body, where
one has been found and tested. Per Jina's docs, an unmatched selector
silently falls back to full-page content rather than erroring, so entries
here should be periodically re-verified rather than trusted indefinitely.

This file is expected to grow as more publishers are profiled -- kept
separate from main.py so it can scale without cluttering app logic.
"""

# Applied to every request regardless of publisher -- strips common
# boilerplate even on sites with no specific profile entry.
DEFAULT_REMOVE_SELECTOR = "nav, header, footer, aside, .sidebar, .newsletter, .related, .ad, .advertisement"

PROFILES = {
    "apnews.com": {
        "target_selector": ".RichTextStoryBody"
    },
}