"""
fetcher.py — RSS Feed Fetcher for Indian Financial News

PURPOSE:
    Fetches latest financial news from MoneyControl and Economic Times
    using standard RSS/Atom feeds. No web scraping involved.

HOW IT WORKS:
    1. feedparser downloads and parses each RSS feed URL
    2. Each article is cleaned and structured into a dict
    3. Duplicates are removed by (URL hash first, then title hash)
    4. Articles are capped at max_articles for fast response

CHANGES vs. PREVIOUS VERSION:
    - Dual deduplication: by URL first (exact), then by normalised title.
      Previously only title-dedup was used, which missed articles with
      slightly different titles but the same URL.
    - Added feedparser timeout via socket (feedparser itself doesn't
      expose a timeout parameter).
    - parse_date() now also handles feedparser's struct_time objects
      (entry.published_parsed) as a reliable fallback, avoiding the
      brittle manual format list.
    - Per-feed article cap raised to 10 (from 8) to fill 30 slots from
      fewer feeds when MoneyControl RSS is slow/cached.
"""

import feedparser
import re
import socket
import time
from datetime import datetime, timezone


# ============================================================
# RSS Feed URLs — public endpoints, no authentication needed
# ============================================================
RSS_FEEDS = {
    # Economic Times — typically returns today's articles
    "ET - Markets":  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET - Stocks":   "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "ET - Economy":  "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "ET - Industry": "https://economictimes.indiatimes.com/news/industry/rssfeeds/13352306.cms",
    "ET - Banking":  "https://economictimes.indiatimes.com/industry/banking/finance/rssfeeds/13358259.cms",
    # MoneyControl — may return cached/older data
    "MoneyControl - Latest":  "https://www.moneycontrol.com/rss/latestnews.xml",
    "MoneyControl - Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
}

# Per-feed article cap — limits any single feed from dominating the list
_PER_FEED_CAP = 10

# Network timeout in seconds for each RSS fetch
_FETCH_TIMEOUT = 8


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def clean_html(text: str) -> str:
    """
    Strip HTML tags and normalise whitespace from RSS text fields.

    Args:
        text (str): Raw text that may contain HTML markup

    Returns:
        str: Clean plain text
    """
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)          # remove tags
    text = re.sub(r"&[a-z]+;", " ", text)         # remove common HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_date(entry) -> str:
    """
    Extract a human-readable publish date from a feedparser entry.

    Strategy:
        1. Use entry.published_parsed (struct_time) — most reliable.
        2. Fall back to manually parsing entry.published string.
        3. Return "Unknown Date" if both fail.

    Args:
        entry: A feedparser entry object

    Returns:
        str: Formatted date like "17 Apr 2026, 07:32 PM"
    """
    # Preferred path: feedparser already parsed the date for us
    parsed = getattr(entry, "published_parsed", None)
    if parsed:
        try:
            dt = datetime(*parsed[:6], tzinfo=timezone.utc)
            return dt.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            pass

    # Fallback: parse the raw string ourselves
    date_string = getattr(entry, "published", "") or ""
    if not date_string:
        return "Unknown Date"

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            return dt.strftime("%d %b %Y, %I:%M %p")
        except ValueError:
            continue

    return date_string   # last resort: return raw string


# ─────────────────────────────────────────────────────────────
# FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_single_feed(feed_name: str, feed_url: str) -> list:
    """
    Fetch and parse one RSS feed, returning a list of article dicts.

    Each article dict contains:
        title       (str): Headline
        description (str): Short summary
        link        (str): URL to full article
        published   (str): Formatted date string
        source      (str): Feed name

    A socket-level timeout (_FETCH_TIMEOUT seconds) prevents the app
    from hanging on slow/unavailable feeds.

    Args:
        feed_name (str): Human-readable source label
        feed_url  (str): RSS endpoint URL

    Returns:
        list[dict]: Parsed articles (may be empty on failure)
    """
    articles = []
    original_timeout = socket.getdefaulttimeout()

    try:
        socket.setdefaulttimeout(_FETCH_TIMEOUT)
        feed = feedparser.parse(feed_url)

        # feed.bozo == True means the XML was malformed but may still
        # have entries — only bail out if entries are also empty
        if feed.bozo and not feed.entries:
            print(f"⚠️  Feed unavailable: {feed_name}")
            return []

        for entry in feed.entries:
            title       = clean_html(entry.get("title", ""))
            description = clean_html(
                entry.get("description", "") or entry.get("summary", "")
            )
            link        = entry.get("link", "")
            published   = parse_date(entry)

            if not title:    # skip broken / empty entries
                continue

            articles.append({
                "title":       title,
                "description": description,
                "link":        link,
                "published":   published,
                "source":      feed_name,
            })

    except Exception as e:
        print(f"❌ Error fetching '{feed_name}': {e}")
    finally:
        socket.setdefaulttimeout(original_timeout)   # always restore

    return articles


def fetch_all_news(max_articles: int = 30) -> list:
    """
    Fetch financial news from all configured RSS feeds.

    Deduplication is two-stage:
        1. By normalised URL  — catches the same article in multiple feeds
        2. By normalised title — catches reprints with different URLs

    Args:
        max_articles (int): Hard cap on returned articles (default 30)

    Returns:
        list[dict]: Unique articles, newest-available first, up to max_articles
    """
    all_articles = []

    for feed_name, feed_url in RSS_FEEDS.items():
        feed_articles = fetch_single_feed(feed_name, feed_url)
        all_articles.extend(feed_articles[:_PER_FEED_CAP])

    # ── Stage 1: deduplicate by URL ──────────────────────────
    seen_urls    = set()
    after_url    = []
    for art in all_articles:
        url_key = art["link"].strip().rstrip("/").lower()
        if url_key and url_key not in seen_urls:
            seen_urls.add(url_key)
            after_url.append(art)
        elif not url_key:
            after_url.append(art)   # keep articles with no URL (dedup by title below)

    # ── Stage 2: deduplicate by normalised title ─────────────
    seen_titles = set()
    unique      = []
    for art in after_url:
        title_key = re.sub(r"\W+", " ", art["title"]).lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append(art)

    return unique[:max_articles]