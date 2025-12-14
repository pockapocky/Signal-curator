import feedparser

def collect_rss(url):
    feed = feedparser.parse(url)
    return [
        {
            "title": e.title,
            "link": e.link,
            "published": e.get("published", "")
        }
        for e in feed.entries
    ]