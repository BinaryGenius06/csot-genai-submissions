"""
tools/papers.py — Hugging Face Papers API (Lesson 3)
========================================================
Replaces Week 2's AlphaXiv MCP. Hand-written, no MCP server.

  paper_search(query) -> GET /api/papers/search?q=...
  read_paper(arxiv_id) -> GET /api/papers/{id}  (metadata)
                          + GET /papers/{id}.md (markdown content, optional)

IMPORTANT — verify on a machine with real network access:
  This sandbox has no outbound internet, so the exact JSON shape below was
  not confirmed against a live response. Lesson 3 explicitly warns the shape
  varies ("search results may wrap paper data in a 'paper' key or not").
  _extract_paper_fields() below defensively handles both the flat shape and
  the {"paper": {...}} wrapped shape. Run:
      python3 -c "from tools.papers import paper_search; print(paper_search('flashattention'))"
  on your machine and adjust field names in _extract_paper_fields if the
  real response differs.
"""

import os
import re

import requests

HF_BASE = "https://huggingface.co"
SEARCH_URL = f"{HF_BASE}/api/papers/search"
PAPER_URL = f"{HF_BASE}/api/papers/{{id}}"
PAPER_MD_URL = f"{HF_BASE}/papers/{{id}}.md"

CONTENT_CHARS = 12_000  # truncate paper markdown to this many chars
TIMEOUT = 15


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _normalize_arxiv_id(raw: str) -> str:
    """
    Accepts a bare id ('2205.14135'), a versioned id ('2205.14135v2'),
    or a full URL ('https://arxiv.org/abs/2205.14135') and returns the
    bare id with version suffix stripped.
    """
    raw = raw.strip()
    raw = re.sub(r"^(https?://)?(www\.)?arxiv\.org/(abs|pdf)/", "", raw)
    raw = raw.rstrip("/")
    if raw.endswith(".pdf"):
        raw = raw[:-4]
    raw = re.sub(r"v\d+$", "", raw)  # strip version suffix, e.g. v2
    return raw


def _extract_paper_fields(item: dict) -> dict:
    """
    Normalize one search-result record to {title, arxiv_id, snippet}.
    Handles both a flat record and one wrapped under a "paper" key —
    Lesson 3 calls out this variance explicitly.
    """
    paper = item.get("paper", item) if isinstance(item, dict) else {}
    arxiv_id = paper.get("id") or paper.get("arxiv_id") or paper.get("arxivId") or ""
    title = paper.get("title", "")
    snippet = paper.get("summary") or paper.get("abstract") or ""
    if len(snippet) > 300:
        snippet = snippet[:300] + "..."
    return {
        "title": title,
        "arxiv_id": _normalize_arxiv_id(arxiv_id) if arxiv_id else "",
        "snippet": snippet,
    }


def paper_search(query: str, limit: int = 5) -> dict:
    """
    Search papers indexed on huggingface.co/papers (hybrid semantic +
    full-text — NOT all of arXiv). Returns small dicts per paper so the
    model calls read_paper for full text rather than getting it all here.

    Returns {"content": [{"title", "arxiv_id", "snippet"}, ...]} or {"error": ...}
    """
    if not query or not query.strip():
        return {"error": "query must not be empty"}

    try:
        resp = requests.get(
            SEARCH_URL,
            params={"q": query},
            headers=_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"error": "paper_search timed out"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e}"}
    except Exception as exc:
        return {"error": str(exc)}

    # Response may be a bare list or wrapped in a dict — handle both.
    items = data if isinstance(data, list) else data.get("results", data.get("papers", []))
    if not isinstance(items, list):
        return {"error": f"Unexpected response shape from HF Papers API: {type(items)}"}

    results = [_extract_paper_fields(item) for item in items[:limit]]
    results = [r for r in results if r["arxiv_id"]]  # drop unparseable entries

    if not results:
        return {
            "content": [],
            "note": (
                "No papers found on huggingface.co/papers for this query. "
                "This index is a subset of arXiv, not all of it — try web_fetch "
                "on arxiv.org directly if you have a specific paper in mind."
            ),
        }
    return {"content": results}


def read_paper(arxiv_id: str) -> dict:
    """
    Fetch metadata (GET /api/papers/{id}) and markdown content
    (GET /papers/{id}.md) for a paper. Falls back to abstract-only if the
    .md endpoint 404s (not every paper has arXiv HTML available).

    Returns {"content": str, "title": str, "truncated": bool} or {"error": ...}
    """
    pid = _normalize_arxiv_id(arxiv_id)
    if not pid:
        return {"error": "arxiv_id is required, e.g. '2205.14135'"}

    # 1. Metadata
    try:
        meta_resp = requests.get(PAPER_URL.format(id=pid), headers=_headers(), timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        return {"error": "read_paper metadata request timed out"}
    except Exception as exc:
        return {"error": str(exc)}

    if meta_resp.status_code == 404:
        return {
            "error": (
                f"Paper '{pid}' is not indexed on huggingface.co/papers (404). "
                f"Fall back to web_fetch('https://arxiv.org/abs/{pid}')."
            )
        }
    if not meta_resp.ok:
        return {"error": f"HTTP {meta_resp.status_code} fetching paper metadata"}

    meta = meta_resp.json()
    paper_meta = meta.get("paper", meta) if isinstance(meta, dict) else {}
    title = paper_meta.get("title", pid)
    abstract = paper_meta.get("summary") or paper_meta.get("abstract") or ""

    # 2. Markdown content (best-effort — not every paper has it)
    content = abstract
    used_fallback = True
    try:
        md_resp = requests.get(PAPER_MD_URL.format(id=pid), headers=_headers(), timeout=TIMEOUT)
        if md_resp.ok and md_resp.text.strip():
            content = md_resp.text
            used_fallback = False
    except Exception:
        pass  # keep abstract fallback

    if not content:
        return {"error": f"No abstract or markdown content available for '{pid}'"}

    truncated = len(content) > CONTENT_CHARS
    if truncated:
        content = content[:CONTENT_CHARS] + "\n\n[...truncated]"

    return {
        "content": content,
        "title": title,
        "arxiv_id": pid,
        "source": "abstract" if used_fallback else "markdown",
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------
PAPER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "paper_search",
            "description": (
                "Search ML/CS papers indexed on huggingface.co/papers (hybrid semantic "
                "+ full-text search). This is a subset of arXiv, not all of it. "
                "Returns small {title, arxiv_id, snippet} records — call read_paper "
                "with the arxiv_id to get full content. Use this for academic/ML "
                "literature questions, not current events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords or paper title"},
                    "limit": {"type": "integer", "description": "Max results to return (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": (
                "Fetch metadata and full markdown content for a paper by arXiv id "
                "(e.g. '2205.14135', from paper_search results — do not guess ids). "
                "Falls back to abstract-only if full markdown isn't available. "
                "If this returns an error/404, fall back to "
                "web_fetch('https://arxiv.org/abs/{id}')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv id, with or without version suffix or URL prefix",
                    },
                },
                "required": ["arxiv_id"],
            },
        },
    },
]
