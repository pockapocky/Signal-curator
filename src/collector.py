"""
SignalCurator — collector.py

Purpose:
- Read config/topics.yaml
- Pull items from the last N days (Europe/Amsterdam clock for display)
- Sources supported:
  - google_news (Google News RSS search)
  - reddit:r/<sub> (Reddit RSS)
  - hackernews (HN Algolia search API)
  - arxiv:<category> (arXiv RSS by category, e.g., arxiv:cs.AI)
- De-duplicate and cap output (10–15 total) while keeping broad coverage
- Print a daily digest with:
  **YYYY-MM-DD – Headlines for Today**
  ...
  **End of YYYY-MM-DD Update**
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import feedparser
import pytz
import requests
import yaml
from dateutil import parser as dtparser


CONFIG_PATH = "config/topics.yaml"


# ----------------------------
# Models
# ----------------------------

@dataclass
class Item:
    title: str
    link: str
    source: str
    published_raw: str = ""
    published_dt: Optional[datetime] = None
    language: str = "EN"   # EN/JA/LT/UNK (simple heuristic)
    tone: str = "0"        # + / – / 0  (light heuristic)
    topic: str = ""        # filled later


# ----------------------------
# Utilities
# ----------------------------

def load_config(path: str = CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def stable_id(title: str, link: str) -> str:
    key = f"{norm(title).lower()}|{norm(link)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return dtparser.parse(s)
    except Exception:
        return None


def detect_language(text: str) -> str:
    # Heuristic: Japanese kana/kanji => JA
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text):
        return "JA"
    # Lithuanian-specific letters => LT (very rough)
    if re.search(r"[ąčęėįšųūžĄČĘĖĮŠŲŪŽ]", text):
        return "LT"
    # Default to EN
    return "EN"


def tone_heuristic(text: str) -> str:
    t = text.lower()
    negative = ["cuts", "layoff", "down", "crisis", "war", "lawsuit", "decline", "ban", "fraud", "hack", "breach"]
    positive = ["record", "surge", "growth", "breakthrough", "wins", "launch", "upgrade", "improves", "expands"]
    if any(w in t for w in negative):
        return "–"
    if any(w in t for w in positive):
        return "+"
    return "0"


def fetch_feed(url: str, timeout: int = 25) -> feedparser.FeedParserDict:
    # Reddit (and sometimes Google News) behaves better with a real User-Agent
    headers = {
        "User-Agent": "signal-curator/1.0 (+https://github.com/; contact: replace-me@example.com)"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return feedparser.parse(r.content)


def within_lookback(dt_obj: Optional[datetime], cutoff: datetime) -> bool:
    # If no date available, keep it (but it may be filtered later by caps)
    if dt_obj is None:
        return True
    return dt_obj >= cutoff


def to_local(dt_obj: Optional[datetime], tz) -> Optional[datetime]:
    if dt_obj is None:
        return None
    if dt_obj.tzinfo is None:
        dt_obj = pytz.UTC.localize(dt_obj)
    return dt_obj.astimezone(tz)


# ----------------------------
# Source collectors
# ----------------------------

def collect_google_news(query: str, per_query: int) -> List[Item]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"
    feed = fetch_feed(url)
    items: List[Item] = []
    for e in feed.entries[:per_query]:
        title = norm(e.get("title", ""))
        link = e.get("link", "")
        published = e.get("published", "") or e.get("updated", "")
        dt_obj = parse_dt(published)
        items.append(
            Item(
                title=title,
                link=link,
                source="Google News",
                published_raw=published,
                published_dt=dt_obj,
                language=detect_language(title),
                tone=tone_heuristic(title),
            )
        )
    return items


def collect_reddit(subreddit: str, per_sub: int) -> List[Item]:
    url = f"https://www.reddit.com/r/{subreddit}/.rss"
    feed = fetch_feed(url)
    items: List[Item] = []
    for e in feed.entries[:per_sub]:
        title = norm(e.get("title", ""))
        link = e.get("link", "")
        published = e.get("published", "") or e.get("updated", "")
        dt_obj = parse_dt(published)
        items.append(
            Item(
                title=title,
                link=link,
                source=f"Reddit r/{subreddit}",
                published_raw=published,
                published_dt=dt_obj,
                language=detect_language(title),
                tone=tone_heuristic(title),
            )
        )
    return items


def collect_hackernews(query: str, per_query: int) -> List[Item]:
    url = f"https://hn.algolia.com/api/v1/search?query={quote_plus(query)}&tags=story"
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    data = r.json()

    items: List[Item] = []
    for hit in data.get("hits", [])[:per_query]:
        title = norm(hit.get("title") or "")
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        published = hit.get("created_at", "")
        dt_obj = parse_dt(published)
        items.append(
            Item(
                title=title,
                link=link,
                source="Hacker News",
                published_raw=published,
                published_dt=dt_obj,
                language=detect_language(title),
                tone=tone_heuristic(title),
            )
        )
    return items


def collect_arxiv(category: str, per_cat: int) -> List[Item]:
    # arXiv RSS is best by category (e.g. cs.AI, cs.LG, cs.CL)
    url = f"https://export.arxiv.org/rss/{category}"
    feed = fetch_feed(url)
    items: List[Item] = []
    for e in feed.entries[:per_cat]:
        title = norm(e.get("title", ""))
        link = e.get("link", "")
        published = e.get("published", "") or e.get("updated", "")
        dt_obj = parse_dt(published)
        items.append(
            Item(
                title=title,
                link=link,
                source=f"arXiv {category}",
                published_raw=published,
                published_dt=dt_obj,
                language="EN",
                tone="0",
            )
        )
    return items


# ----------------------------
# Planning / ranking
# ----------------------------

def dedupe(items: List[Item]) -> List[Item]:
    seen = set()
    out: List[Item] = []
    for it in items:
        key = stable_id(it.title, it.link)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def sort_newest(items: List[Item], tz) -> List[Item]:
    def key(it: Item):
        dt_local = to_local(it.published_dt, tz)
        # None dates go last
        return dt_local or datetime(1970, 1, 1, tzinfo=tz)

    return sorted(items, key=key, reverse=True)


def cap_broad_coverage(
    by_topic: Dict[str, List[Item]],
    tz,
    total_min: int = 10,
    total_max: int = 15,
    per_topic_cap: int = 3,
) -> Dict[str, List[Item]]:
    """
    Keep coverage broad:
    - take up to per_topic_cap per topic first
    - then fill remaining slots with newest across topics
    """
    selected: Dict[str, List[Item]] = {k: [] for k in by_topic.keys()}

    # First pass: up to per_topic_cap per topic
    pool: List[Tuple[str, Item]] = []
    for topic, items in by_topic.items():
        items_sorted = sort_newest(items, tz)
        take = items_sorted[:per_topic_cap]
        selected[topic] = take
        # leftovers go to pool
        for it in items_sorted[per_topic_cap:]:
            pool.append((topic, it))

    # Count
    count = sum(len(v) for v in selected.values())

    # If we’re under minimum, fill from pool (newest first) up to total_max
    pool_sorted = sorted(pool, key=lambda x: (to_local(x[1].published_dt, tz) or datetime(1970, 1, 1, tzinfo=tz)), reverse=True)
    target = max(total_min, min(total_max, total_max))  # keep sane
    while count < target and pool_sorted:
        topic, it = pool_sorted.pop(0)
        selected[topic].append(it)
        count += 1
        if count >= total_max:
            break

    # Trim if over max (rare): trim oldest extras
    if count > total_max:
        # flatten and keep newest total_max, then re-group
        flat: List[Tuple[str, Item]] = []
        for topic, items in selected.items():
            for it in items:
                flat.append((topic, it))
        flat = sorted(flat, key=lambda x: (to_local(x[1].published_dt, tz) or datetime(1970, 1, 1, tzinfo=tz)), reverse=True)[:total_max]
        selected = {k: [] for k in by_topic.keys()}
        for topic, it in flat:
            selected[topic].append(it)

    # Final sort within each topic
    for topic in selected.keys():
        selected[topic] = sort_newest(selected[topic], tz)

    return selected


# ----------------------------
# Main runner
# ----------------------------

def collect_for_topic(topic_cfg: Dict, cutoff: datetime, tz) -> List[Item]:
    name = topic_cfg.get("name", "Untitled Topic")
    queries = topic_cfg.get("queries") or [name]
    sources = topic_cfg.get("sources") or []

    # per-source tuning (keeps it broad but not spammy)
    per_query_google = 4
    per_query_hn = 4
    per_sub_reddit = 6
    per_cat_arxiv = 6

    items: List[Item] = []

    for src in sources:
        try:
            if src == "google_news":
                for q in queries:
                    items += collect_google_news(q, per_query_google)

            elif src.startswith("reddit:r/"):
                sub = src.split("reddit:r/")[1]
                items += collect_reddit(sub, per_sub_reddit)

            elif src == "hackernews":
                for q in queries:
                    items += collect_hackernews(q, per_query_hn)

            elif src.startswith("arxiv:"):
                cat = src.split("arxiv:")[1]
                items += collect_arxiv(cat, per_cat_arxiv)

            else:
                # Unknown source token; ignore (don’t crash)
                pass

        except Exception as e:
            # Don’t let one source kill the whole run
            items.append(
                Item(
                    title=f"[Source error] {src}: {type(e).__name__}",
                    link="",
                    source="SignalCurator",
                    published_raw="",
                    published_dt=None,
                    language="UNK",
                    tone="–",
                    topic=name,
                )
            )

    # Apply topic + lookback filter
    out: List[Item] = []
    for it in items:
        it.topic = name
        dt_local = to_local(it.published_dt, tz)
        if within_lookback(dt_local, cutoff):
            out.append(it)

    return dedupe(out)


import os
from io import StringIO

def print_digest(selected: Dict[str, List[Item]], tz):
    now = datetime.now(tz)
    date_str = now.strftime("%Y-%m-%d")

    buf = StringIO()

    def w(line: str = ""):
        print(line)
        buf.write(line + "\n")

    w(f"**{date_str} – Headlines for Today**")
    w(f"*(Europe/Amsterdam time — generated {now.strftime('%Y-%m-%d %H:%M')} {tz.zone})*")
    w()

    for topic, items in selected.items():
        w(f"#### {topic}")
        if not items:
            w("• (No items found in lookback window)")
            w()
            continue

        for it in items:
            lang = it.language or "UNK"
            tone = it.tone if it.tone in {"+", "–", "0"} else "0"
            pub = it.published_raw or ""
            if it.link:
                w(f"• {it.title} (tone: {tone}) — {it.source} {it.link} [{lang}] [{pub}]")
            else:
                w(f"• {it.title} (tone: {tone}) — {it.source} [{lang}]")
        w()

    w("---")
    w(f"**End of {date_str} Update**")

    # Write to file
    os.makedirs("out", exist_ok=True)
    out_path = f"out/daily_{date_str}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    # Helpful log line for Actions
    print(f"\nWROTE_FILE: {out_path}\n")


def main():
    cfg = load_config()
    tz = pytz.timezone(cfg.get("time_zone", "Europe/Amsterdam"))
    lookback_days = int(cfg.get("lookback_days", 1))
    cutoff = datetime.now(tz) - timedelta(days=lookback_days)

    topics = cfg.get("topics", [])
    by_topic: Dict[str, List[Item]] = {}

    for t in topics:
        name = t.get("name", "Untitled Topic")
        by_topic[name] = collect_for_topic(t, cutoff, tz)

    # Enforce “10–15 total, broad coverage”
    selected = cap_broad_coverage(
        by_topic=by_topic,
        tz=tz,
        total_min=10,
        total_max=15,
        per_topic_cap=3,
    )

    print_digest(selected, tz)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("FATAL ERROR:", repr(e))
        traceback.print_exc()
        raise