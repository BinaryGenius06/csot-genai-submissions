"""
Research Agent — core logic (no TUI)
======================================
Run from CLI to test before wiring into the TUI:
    python project/agent.py "What are the latest advances in protein folding?"

Pipeline per query:
  1. web_search  → Serper → list of URLs + snippets
  2. web_fetch   → requests + trafilatura → clean markdown
  3. discover_papers / get_paper_content → AlphaXiv MCP
  4. synthesise → cited answer
  5. (optional) save_research_note → markdown file in notes/
"""

import asyncio
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import requests
import trafilatura
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL         = "nvidia/nemotron-3-super-120b-a12b:free"
SERPER_URL    = "https://google.serper.dev/search"
ALPHAXIV_MCP  = "https://api.alphaxiv.org/mcp/v1"
NOTES_DIR     = Path(__file__).parent / "notes"
MAX_ITER      = 20          # max agent loop iterations
FETCH_CHARS   = 8_000       # truncate fetched pages to this many chars
MAX_RESULTS   = 5           # web search results to return

NOTES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
def make_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def web_search(query: str, num: int = MAX_RESULTS) -> str:
    """Call Serper and return a JSON string of results."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return json.dumps({"error": "SERPER_API_KEY not set"})
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", [])[:num]:
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def web_fetch(url: str) -> str:
    """Fetch a URL, extract clean text via trafilatura, truncate."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research-agent/1.0)"},
            timeout=15,
        )
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_links=False) or ""
        if not text:
            # fallback: strip tags crudely
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
        if len(text) > FETCH_CHARS:
            text = text[:FETCH_CHARS] + "\n\n[…truncated]"
        return text or "[No readable content extracted]"
    except requests.exceptions.Timeout:
        return "[ERROR] Request timed out"
    except requests.exceptions.HTTPError as e:
        return f"[ERROR] HTTP {e.response.status_code}: {e}"
    except Exception as exc:
        return f"[ERROR] {exc}"


def save_research_note(title: str, content: str) -> str:
    """Save a research note to notes/<slug>.md"""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_]+", "-", slug)[:60]
    ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = NOTES_DIR / f"{ts}-{slug}.md"
    try:
        path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
        return f"Saved to {path}"
    except Exception as exc:
        return f"[ERROR] {exc}"


# ---------------------------------------------------------------------------
# AlphaXiv MCP helpers (async, called inside asyncio.run)
# ---------------------------------------------------------------------------
async def _alphaxiv_call(tool_name: str, arguments: dict) -> str:
    """Connect to AlphaXiv MCP via Streamable HTTP and call one tool."""
    try:
        async with streamablehttp_client(ALPHAXIV_MCP) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                return "\n".join(parts) if parts else "[No content returned]"
    except Exception as exc:
        return f"[ERROR] AlphaXiv MCP: {exc}"


def discover_papers(keywords: list, question: str, difficulty: int = 5) -> str:
    return asyncio.run(_alphaxiv_call("discover_papers", {
        "keywords": keywords,
        "question": question,
        "difficulty": difficulty,
    }))


def get_paper_content(url: str, full_text: bool = False) -> str:
    return asyncio.run(_alphaxiv_call("get_paper_content", {
        "url": url,
        "fullText": full_text,
    }))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using Google via Serper. "
                "Returns a JSON list of {title, url, snippet}. "
                "Use this first to find relevant pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num":   {"type": "integer", "description": "Number of results (default 5, max 10)"},
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
    {
        "type": "function",
        "function": {
            "name": "discover_papers",
            "description": (
                "Search AlphaXiv for academic papers relevant to a topic. "
                "Returns ranked papers with titles, abstracts, and arXiv IDs. "
                "Use this when the question has a scientific or research angle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3-4 concise keyword terms (method names, acronyms, authors, benchmarks)",
                    },
                    "question": {
                        "type": "string",
                        "description": "Detailed semantic description of what papers you need",
                    },
                    "difficulty": {
                        "type": "integer",
                        "description": "Retrieval effort 1-10 (higher = more thorough, slower). Default 5.",
                    },
                },
                "required": ["keywords", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_content",
            "description": (
                "Retrieve the full text or AI-generated report of an academic paper. "
                "Use this after discover_papers to read a specific paper in depth. "
                "Pass the full arXiv URL, e.g. 'https://arxiv.org/abs/2303.08774'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full arXiv or alphaXiv URL, e.g. 'https://arxiv.org/abs/2307.12307'",
                    },
                    "full_text": {
                        "type": "boolean",
                        "description": "If true, return raw extracted text instead of AI report. Default false.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_research_note",
            "description": (
                "Save a research note to a markdown file in the notes/ folder. "
                "Use this to persist key findings for multi-session research. "
                "Call this once you have a complete, well-sourced answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title":   {"type": "string", "description": "Note title (becomes filename)"},
                    "content": {"type": "string", "description": "Full markdown content of the note"},
                },
                "required": ["title", "content"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a rigorous research assistant. When given a question:

1. Do ONE web_search to find relevant sources.
2. Read 2-3 pages with web_fetch (pick the most promising URLs).
3. Do ONE discover_papers search on AlphaXiv.
4. Read ONE paper with get_paper_content if an arxiv ID looks relevant.
5. STOP searching. Synthesise everything into a clear answer with inline citations [1], [2], etc.
6. List all sources under "## Sources".
7. Call save_research_note with your final answer.

Maximum tools: 8. After 8 tool calls, synthesise what you have — do not keep searching.
"""


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def dispatch_tool(name: str, args: dict, callback=None) -> str:
    """
    Call the right function. callback(name, args, result) is called after
    execution — used by the TUI to push updates to the tool panel.
    """
    fn_map = {
        "web_search":        lambda: web_search(args.get("query", ""), args.get("num", MAX_RESULTS)),
        "web_fetch":         lambda: web_fetch(args.get("url", "")),
        "discover_papers":   lambda: discover_papers(args.get("keywords", []), args.get("question", ""), args.get("difficulty", 5)),
        "get_paper_content": lambda: get_paper_content(args.get("url", ""), args.get("full_text", False)),
        "save_research_note":lambda: save_research_note(args.get("title", "note"), args.get("content", "")),
    }
    fn = fn_map.get(name)
    if fn is None:
        result = f"[ERROR] Unknown tool: {name}"
    else:
        result = fn()

    if callback:
        callback(name, args, result)
    return result


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_research_agent(
    query: str,
    client: OpenAI | None = None,
    tool_callback=None,
    token_callback=None,
) -> str:
    """
    Full agent loop. Returns the final answer string.

    tool_callback(name, args, result) — called after each tool execution.
    token_callback(token: str)        — called for each streamed token (TUI use).
    """
    if client is None:
        client = make_client()

    messages = [{"role": "user", "content": query}]

    for iteration in range(MAX_ITER):
        # Retry on rate limit (free tier gets throttled ~30s)
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    tools=LOCAL_TOOLS,
                    tool_choice="auto",
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    wait = 35
                    print(f"  [rate limit] waiting {wait}s… (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    raise
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content or ""

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = dispatch_tool(tc.function.name, args, callback=tool_callback)
                messages.append({
                    "role":        "tool",
                    "tool_call_id": tc.id,
                    "content":     result,
                })
        else:
            return choice.message.content or ""

    return "[ERROR] Agent exceeded max iterations without a final answer."


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Research query: ").strip()
    if not query:
        print("No query provided.")
        sys.exit(1)

    print(f"\nResearching: {query}\n{'─' * 60}")

    def cli_callback(name, args, result):
        truncated = str(result)[:200].replace("\n", " ")
        print(f"  ⚙  {name}({list(args.values())[0] if args else ''})")
        print(f"     → {truncated}{'…' if len(str(result)) > 200 else ''}\n")

    client = make_client()
    answer = run_research_agent(query, client=client, tool_callback=cli_callback)

    print("\n" + "═" * 60)
    print("ANSWER\n" + "═" * 60)
    print(answer)
