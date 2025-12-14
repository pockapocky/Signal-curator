import hashlib
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import pytz
import requests
import yaml
from dateutil import parser as dtparser


def load_config(path="config/topics.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def item_id(title: str, link: str) -> str:
    key = f"{normalize_text(title).lower()}|{normalize_text(link)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def within_lookback(published_dt, cutoff_dt):
    if not published_dt:
        return True  # if no date, keep it (we’ll improve later)
    return published_dt >= cutoff_dt


def parse_published(entry):
    # RSS feeds vary a lot. Try common fields.
    for field in ("published", "updated", "created"):
        if field in entry and entry[field]:
            try:
                return dtparser.parse(entry[field])
            except Exception:
                pass
    return None


def fetch_google_news(query, max_items=10):
    # Google News RSS search endpoint
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        items.append(
            {
                "title": normalize_text(e.get("title")),
                "link": e.get("link"),
                "source": "Google News",
                "published": e.get("published", "") or e.get("updated", ""),
            }
        )
    return items


def fetch_reddit_rss(subreddit, max_items=10):
    url = f"https://www.reddit.com/r/{subreddit}/.rss"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        items.append(
            {
                "title": normalize_text(e.get("title")),
                "link": e.get("link"),
                "source": f"Reddit r/{subreddit}",
                "published": e.get("published", "") or e.get("updated", ""),
            }
        )
    return items


def fetch_arxiv(query, max_items=10):
    # arXiv RSS supports search query
    url = f"https://export.arxiv.org/rss/{quote_plus(query)}"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        items.append(
            {
                "title": normalize_text(e.get("title")),
                "link": e.get("link"),
                "source": "arXiv",
                "published": e.get("published", "") or e.get("updated", ""),
            }
        )
    return items


def fetch_hn(query, max_items=10):
    # Hacker News via Algolia search API (public)
    url = f"https://hn.algolia.com/api/v1/search?query={quote_plus(query)}&tags=story"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = []
    for hit in data.get("hits", [])[:max_items]:
        title = normalize_text(hit.get("title"))
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        created_at = hit.get("created_at", "")
        items.append(
            {
                "title": title,
                "link": link,
                "source": "Hacker News",
                "published": created_at,
            }
        )
    return items


def collect_for_topic(query, sources, per_source=8):
    collected = []

    # Use topic name as the base query unless you later add explicit query strings.
    base_query = query

    for src in sources:
        if src == "google_news":
            collected += fetch_google_news(base_query, max_items=per_source)

        elif src.startswith("reddit:r/"):
            subreddit = src.split("reddit:r/")[1]
            collected += fetch_reddit_rss(subreddit, max_items=per_source)

        elif src == "arxiv":
            collected += fetch_arxiv(base_query, max_items=per_source)

        elif src == "hackernews":
            collected += fetch_hn(base_query, max_items=per_source)

        # placeholders for later:
        # elif src == "trade_rss": ...
        # elif src == "nikkei": ...
        # elif src == "nhk": ...
        # elif src == "reuters": ...
        # elif src == "sec_filings": ...

    return collected


def dedupe(items):
    seen = set()
    out = []
    for it in items:
        key = item_id(it.get("title", ""), it.get("link", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def main():
    cfg = load_config()
    tz = pytz.timezone(cfg.get("time_zone", "Europe/Amsterdam"))
    lookback_days = int(cfg.get("lookback_days", 1))

    now = datetime.now(tz)
    cutoff = now - timedelta(days=lookback_days)

    date_str = now.strftime("%Y-%m-%d")
    print(f"**{date_str} – Headlines for Today**")
    print(f"*(Europe/Amsterdam time — lookback: last {lookback_days} day(s))*")
    print()

    for t in cfg.get("topics", []):
        topic_name = t["name"]
        sources = t.get("sources", [])

        queries = t.get("queries", [topic_name])
items = []
for q in queries:
    items += collect_for_topic(q, sources, per_source=4)

        # Filter by time window where possible
        filtered = []
        for it in items:
            pub_dt = parse_published(it)
            if pub_dt and pub_dt.tzinfo is None:
                # Assume UTC if missing tz (common in APIs)
                pub_dt = pytz.UTC.localize(pub_dt)
            if pub_dt:
                pub_dt_local = pub_dt.astimezone(tz)
            else:
                pub_dt_local = None

            if within_lookback(pub_dt_local, cutoff):
                filtered.append(it)

        filtered = dedupe(filtered)[:10]

        print(f"#### {topic_name}")
        if not filtered:
            print("- (No items found in lookback window)")
            print()
            continue

        for it in filtered:
            title = it["title"]
            link = it["link"]
            source = it["source"]
            published = it.get("published", "")
            print(f"• {title} — ({source}) {link} [{published}]")
        print()

    print("---")
    print(f"**End of {date_str} Update**")


if __name__ == "__main__":
    main()