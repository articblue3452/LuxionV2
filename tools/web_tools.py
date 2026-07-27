"""Read-only public-web tools used by execution plans."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def web_search(query: str) -> list[dict[str, str]]:
    """Return concise DuckDuckGo Instant Answer results for ``query``.

    The tool is intentionally read-only and returns structured data so later
    nodes can decide how to present the research result.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("web_search requires a non-empty query.")

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
