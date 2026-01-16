#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import yaml
import html
import feedparser
from datetime import datetime, timezone, timedelta
from dateutil import parser as dtparser
from urllib.parse import urlencode

ARXIV_API_BASE = "http://export.arxiv.org/api/query"


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs():
    os.makedirs("docs/topics", exist_ok=True)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def text_haystack(title: str, summary: str) -> str:
    return (title + " " + summary).lower()


def compile_terms(terms):
    pats = []
    for t in (terms or []):
        t = (t or "").strip()
        if not t:
            continue
        # phrases: simple contains
        if " " in t or "-" in t:
            pats.append(re.compile(re.escape(t), re.IGNORECASE))
        else:
            # single token: word boundary
            pats.append(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
    return pats


def matches_any(text: str, patterns) -> bool:
    return any(p.search(text) for p in patterns)


def is_blacklisted(title: str, summary: str, blacklist):
    hay = text_haystack(title, summary)
    for w in blacklist or []:
        if (w or "").lower() in hay:
            return True
    return False


def parse_arxiv_id(link: str) -> str:
    m = re.search(r"arxiv\.org/abs/([^?#]+)", link or "")
    return m.group(1) if m else (link or "")


def entry_date(entry) -> datetime:
    d = None
    if getattr(entry, "published", None):
        d = dtparser.parse(entry.published)
    elif getattr(entry, "updated", None):
        d = dtparser.parse(entry.updated)
    else:
        return datetime.now(timezone.utc)

    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def fetch_topic(search_query: str, max_results: int, start: int = 0):
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = ARXIV_API_BASE + "?" + urlencode(params)

    feed = feedparser.parse(url)
    if getattr(feed, "bozo", 0):
        raise RuntimeError(
            f"Feed parsing failed: {getattr(feed, 'bozo_exception', 'unknown error')}"
        )

    items = []
    for e in feed.entries:
        title = normalize(html.unescape(getattr(e, "title", "")))
        summary = normalize(html.unescape(getattr(e, "summary", "")))
        link = getattr(e, "link", "")
        authors = []
        if hasattr(e, "authors"):
            authors = [a.name for a in e.authors if hasattr(a, "name")]
        published = entry_date(e)

        items.append(
            {
                "title": title,
                "summary": summary,
                "link": link,
                "arxiv_id": parse_arxiv_id(link),
                "authors": authors,
                "published_utc": published,
            }
        )
    return items


def within_days(dt_utc: datetime, days_back: int) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return dt_utc >= cutoff


def md_paper(item) -> str:
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
        lines.append(
            f"- [{t['name']}](topics/{t['slug']}.md) — **{t['count']}** papers (last {t['days_back']} days)\n"
        )

    lines.append("\n---\n")
    lines.append("Generated automatically from arXiv.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def update_readme_block(updated_str: str, topics_meta):
    start_marker = "<!-- AUTO-GENERATED:START -->"
    end_marker = "<!-- AUTO-GENERATED:END -->"
    updated_date = updated_str.split(" ")[0]

    lines = []
    lines.append("## Latest\n")
    lines.append(f"Updated on: **{updated_date}**\n")
    lines.append("\nGenerated pages are available under `docs/`.\n")
    lines.append("\n## Topic Navigator\n")
    lines.append("| Topic | Papers | Link |")
    lines.append("|------|--------|------|")

    for t in topics_meta:
        link = f"docs/topics/{t['slug']}.md"
        lines.append(f"| {t['name']} | {t['count']} | [{t['name']}]({link}) |")

    block = "\n".join(lines) + "\n"

    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme = f.read()
    except FileNotFoundError:
        readme = f"# arXiv Daily\n\n{start_marker}\n{end_marker}\n"

    if start_marker not in readme or end_marker not in readme:
        readme = readme.rstrip() + f"\n\n{start_marker}\n{end_marker}\n"

    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        flags=re.DOTALL,
    )
    replacement = f"{start_marker}\n\n{block}\n{end_marker}"
    new_readme = pattern.sub(replacement, readme)

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(new_readme)


def main():
    cfg = load_config("config.yaml")
    ensure_dirs()

    site = cfg.get("site", {})
    title = site.get("title", "arXiv Daily")
    desc = site.get("description", "")
    max_per_topic = int(site.get("max_results_per_topic", 40))
    days_back = int(site.get("days_back", 7))

    # Τραβάμε περισσότερα για να μην “χάνονται” papers πριν το keyword filter
    fetch_k = int(site.get("fetch_multiplier", 8))
    fetch_n = min(max_per_topic * fetch_k, 300)

    blacklist = (cfg.get("filters", {}) or {}).get("blacklist", [])
    updated_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    topics_meta = []

    for t in cfg.get("topics", []):
        name = t["name"]
        slug = t["slug"]
        query = t["query"]

        include_any = t.get("include_any", [])
        exclude_any = t.get("exclude_any", [])

        inc_p = compile_terms(include_any)
        exc_p = compile_terms(exclude_any)

        raw = fetch_topic(query, max_results=fetch_n)

        items = []
        seen_in_topic = set()

        for it in raw:
            if not within_days(it["published_utc"], days_back):
                continue

            # topic include/exclude
            hay = text_haystack(it["title"], it["summary"])
            if include_any and not matches_any(hay, inc_p):
                continue
            if exclude_any and matches_any(hay, exc_p):
                continue

            # global blacklist (τελευταίο)
            if is_blacklisted(it["title"], it["summary"], blacklist):
                continue

            if it["arxiv_id"] in seen_in_topic:
                continue
            seen_in_topic.add(it["arxiv_id"])

            items.append(it)

        items.sort(key=lambda x: x["published_utc"], reverse=True)
        items = items[:max_per_topic]  # τελικό cap

        write_topic_page(name, slug, items, updated_str)
        topics_meta.append(
            {"name": name, "slug": slug, "count": len(items), "days_back": days_back}
        )

    write_index(title, desc, topics_meta, updated_str)
    update_readme_block(updated_str, topics_meta)
    print("Done. Pages generated under docs/ and README block updated.")


if __name__ == "__main__":
    main()
