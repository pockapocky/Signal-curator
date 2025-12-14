def publish_markdown(items, date):
    lines = [f"# Global + Social Signal Brief — {date}", ""]
    for item in items:
        lines.append(f"- [{item['title']}]({item['link']})")
    return "\n".join(lines)