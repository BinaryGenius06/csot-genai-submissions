"""
tools/web.py — Week 2 web tools, ported.
==========================================
web_search — Serper (Google) search
web_fetch  — fetch + extract clean text via trafilatura

Carried over from week_2/project/agent.py almost unchanged. Differences:
  - Return Python dicts shaped {"content": ...} / {"error": ...} instead of
    JSON strings — matches the Week 3 file-tool convention (Lesson 2) so every
    tool in the project follows one error-handling pattern.
  - No AlphaXiv / MCP code here. Papers move to tools/papers.py (Lesson 3).
"""

import os
import re

import requests
import trafilatura

SERPER_URL = "https://google.serper.dev/search"
FETCH_CHARS = 8_000   # truncate fetched pages to this many chars
MAX_RESULTS = 5        # default number of search results


def web_search(query: str, num: int = MAX_RESULTS) -> dict:
    """
    Search the web via Serper (Google). Returns:
      {"content": [{"title", "url", "snippet"}, ...]}
      {"error": "..."}
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return {"error": "SERPER_API_KEY not set"}

    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"error": "web_search timed out"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e}"}
    except Exception as exc:
        return {"error": str(exc)}

    results = []
    for item in data.get("organic", [])[:num]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })

    if not results:
        return {"content": [], "note": "No results found."}
    return {"content": results}


def web_fetch(url: str) -> dict:
    """
    Fetch a URL and extract clean readable text via trafilatura.
    Falls back to a crude tag-strip if trafilatura finds nothing.
    Truncates to FETCH_CHARS. Returns {"content": str} or {"error": str}.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research-agent/1.0)"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {"error": "web_fetch timed out"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e}"}
    except Exception as exc:
        return {"error": str(exc)}

    text = trafilatura.extract(resp.text, include_links=False) or ""
    if not text:
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return {"error": "No readable content extracted"}

    truncated = len(text) > FETCH_CHARS
    if truncated:
        text = text[:FETCH_CHARS] + "\n\n[...truncated]"

    return {"content": text, "truncated": truncated}


# ---------------------------------------------------------------------------
# OpenAI tool schemas — used by Agent to register these with the model
# ---------------------------------------------------------------------------
WEB_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using Google via Serper. "
                "Returns a list of {title, url, snippet}. "
                "Use this first to find relevant pages for non-paper questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num": {"type": "integer", "description": "Number of results (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch and extract the full text of a webpage. "
                "Use this on promising URLs from web_search to read the full article."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
]
