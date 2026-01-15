#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import yaml
import html
import feedparser
from datetime import datetime, timezone, timedelta
from dateutil import parser as dtparser

ARXIV_API_RSS = "http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    os.makedirs("docs/topics", exist_ok=True)

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def is_blacklisted(title: str, summary: str, blacklist):
    hay = (title + " " + summary).lower()
    for w in blacklist or []:
        if w.lower() in hay:
            return True
    return False

def parse_arxiv_id(link: str) -> str:
    # link is often like http://arxiv.org/abs/XXXX.XXXXvN
    m = re.search(r"arxiv\.org/abs/([^?#]+)", link or "")
    return m.group(1) if m else (link or "")

def entry_date(entry) -> datetime:
    # arXiv RSS uses published/updated
    d = None
    if "published" in entry and entry.published:
        d = dtparser.parse(entry.published)
    elif "updated" in entry and entry.updated:
        d = dtparser.parse(entry.updated)
    else:
        return datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)

def fetch_topic(query: str, max_results: int):
    url = ARXIV_API_RSS.format(query=escape_query(query), max_results=max_results)
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries:
        title = normalize(html.unescape(getattr(e, "title", "")))
        summary = normalize(html.unescape(getattr(e, "summary", "")))
        link = getattr(e, "link", "")
        authors = []
        if hasattr(e, "authors"):
            authors = [a.name for a in e.authors if hasattr(a, "name")]
        published = entry_date(e)

        items.append({
            "title": title,
            "summary": summary,
            "link": link,
            "arxiv_id": parse_arxiv_id(link),
            "authors": authors,
            "published_utc": published,
        })
    return items

def escape_query(q: str) -> str:
    # arXiv wants URL-encoded query. feedparser can handle, but we should encode safely.
    # We avoid urllib.parse.quote to keep this file minimal; do it manually for common chars.
    from urllib.parse import quote
    return quote(q, safe="():\"' ORAND+-_*")

def within_days(dt_utc: datetime, days_back: int) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return dt_utc >= cutoff

def md_paper(item):
    date_str = item["published_utc"].strftime("%Y-%m-%d")
    authors = ", ".join(item["authors"][:10])
    if len(item["authors"]) > 10:
        authors += ", et al."
    return (
        f"- **{item['title']}**  \n"
        f"  {authors}  \n"
        f"  _{date_str}_ · {item['link']}  \n"
        f"  <details><summary>Abstract</summary>\n\n"
        f"  {item['summary']}\n\n"
        f"  </details>\n"
    )

def write_topic_page(topic_name: str, slug: str, items, updated_str: str):
    path = f"docs/topics/{slug}.md"
    lines = []
    lines.append(f"# {topic_name}\n")
    lines.append(f"_Updated: {updated_str}_\n")
    lines.append(f"Total papers shown: **{len(items)}**\n")
    lines.append("\n---\n")
    for it in items:
        lines.append(md_paper(it))
        lines.append("\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_index(site_title: str, description: str, topics_meta, updated_str: str):
    path = "docs/index.md"
    lines = []
    lines.append(f"# {site_title}\n")
    if description:
        lines.append(f"{description}\n")
    lines.append(f"_Updated: {updated_str}_\n")
    lines.append("\n## Topics\n")
    for t in topics_meta:
        lines.append(f"- [{t['name']}](topics/{t['slug']}.html) — **{t['count']}** papers (last {t['days_back']} days)\n")
    lines.append("\n---\n")
    lines.append("Generated automatically from arXiv.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_readme(site_title: str):
    # κρατάμε README απλό: link προς Pages
    text = f"""# {site_title}

This repo auto-updates arXiv papers daily and publishes pages via GitHub Pages.

- Open the site: `Settings → Pages` after enabling Pages (branch: `main`, folder: `/docs`)
- Generated pages live in `docs/`.
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(text)

def main():
    cfg = load_config("config.yaml")
    ensure_dirs()

    site = cfg.get("site", {})
    title = site.get("title", "arXiv Daily")
    desc = site.get("description", "")
    max_per_topic = int(site.get("max_results_per_topic", 40))
    days_back = int(site.get("days_back", 7))

    blacklist = (cfg.get("filters", {}) or {}).get("blacklist", [])

    updated_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    topics_meta = []
    seen_global = set()  # dedup across topics by arXiv id

    for t in cfg.get("topics", []):
        name = t["name"]
        slug = t["slug"]
        query = t["query"]

        raw = fetch_topic(query, max_results=max_per_topic)

        # filter by days_back + blacklist + dedup
        items = []
        for it in raw:
            if not within_days(it["published_utc"], days_back):
                continue
            if is_blacklisted(it["title"], it["summary"], blacklist):
                continue
            if it["arxiv_id"] in seen_global:
                continue
            seen_global.add(it["arxiv_id"])
            items.append(it)

        # sort by date (newest first)
        items.sort(key=lambda x: x["published_utc"], reverse=True)

        write_topic_page(name, slug, items, updated_str)
        topics_meta.append({"name": name, "slug": slug, "count": len(items), "days_back": days_back})

    write_index(title, desc, topics_meta, updated_str)
    write_readme(title)

    print("Done. Pages generated under docs/")

if __name__ == "__main__":
    main()
