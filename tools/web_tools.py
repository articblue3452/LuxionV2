"""Read-only public-web tools used by execution plans."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree


def web_search(query: str) -> list[dict[str, str]]:
    """Return current web/news results for ``query``.

    Google News RSS is used first because an Instant Answer endpoint is not a
    news index and frequently returns no articles for time-sensitive queries.
    DuckDuckGo remains a read-only fallback for non-news knowledge.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("web_search requires a non-empty query.")

    news_endpoint = "https://news.google.com/rss/search?" + urlencode(
        {"q": query.strip(), "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    )
    request = Request(news_endpoint, headers={"User-Agent": "Luxion/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            root = ElementTree.fromstring(response.read())
        news_results = []
        for item in root.findall("./channel/item")[:10]:
            title = item.findtext("title", default="").strip()
            url = item.findtext("link", default="").strip()
            published = item.findtext("pubDate", default="").strip()
            source = item.findtext("source", default="").strip()
            if title and url:
                news_results.append({"title": title, "snippet": source, "url": url, "published_at": published})
        if news_results:
            return news_results
    except (OSError, ElementTree.ParseError):
        # The non-news fallback below still provides useful results when RSS is
        # unavailable, such as in restricted corporate networks.
        pass

    endpoint = "https://api.duckduckgo.com/?" + urlencode(
        {"q": query.strip(), "format": "json", "no_html": "1", "skip_disambig": "1"}
    )
    request = Request(endpoint, headers={"User-Agent": "Luxion/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except OSError as error:
        raise RuntimeError(f"Web search failed: {error}") from error

    results: list[dict[str, str]] = []
    abstract = payload.get("AbstractText")
    abstract_url = payload.get("AbstractURL")
    if isinstance(abstract, str) and abstract:
        results.append({"title": str(payload.get("Heading") or query), "snippet": abstract, "url": str(abstract_url or "")})

    def collect(items: list[Any]) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            nested = item.get("Topics")
            if isinstance(nested, list):
                collect(nested)
            elif isinstance(item.get("Text"), str):
                results.append(
                    {
                        "title": item["Text"],
                        "snippet": item["Text"],
                        "url": str(item.get("FirstURL") or ""),
                    }
                )

    related = payload.get("RelatedTopics")
    if isinstance(related, list):
        collect(related)
    return results[:5]
