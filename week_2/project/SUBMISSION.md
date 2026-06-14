# Week 2 Submission — GenAI/Agentic Track 2026

## Setup Instructions

### Prerequisites
- Python 3.10+
- A free [OpenRouter](https://openrouter.ai) account (API key)
- A free [Serper](https://serper.dev) account (API key, 2500 free queries/month)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment
Copy `.env.example` to `.env` in the `week_2/` folder and fill in your keys:
```
OPENROUTER_API_KEY=sk-or-v1-...
SERPER_API_KEY=...
```

### Run the TUI
From the `week_2/` directory:
```bash
python project/tui.py
```

Or to test the agent without the TUI:
```bash
python project/agent.py "Your research question here"
```

### Model selection
The agent uses `nvidia/nemotron-3-super-120b-a12b:free` by default (free, no credits needed).
To override:
```bash
AGENT_MODEL=meta-llama/llama-3.3-70b-instruct:free python project/tui.py
```

### Notes on AlphaXiv MCP
The AlphaXiv MCP server (`https://api.alphaxiv.org/mcp/v1`) requires OAuth authentication.
The agent attempts to connect and falls back gracefully if unavailable — all other tools
(web search, web fetch) continue to work normally.

---

## What I Built

A terminal-based research tool modelled on Perplexity. You type a question, and an agent:

1. Searches the web via Serper to find relevant pages
2. Fetches and reads 2–3 full pages (not just snippets) using trafilatura
3. Connects to the AlphaXiv MCP server to find and read academic papers
4. Synthesises a cited answer and saves it to a markdown file in `notes/`

Everything runs inside a split-panel Textual TUI: the left panel streams the model's answer token-by-token, and the right panel shows each tool call in real time as the agent works through it.

---

## How the Agent Loop Works

The core loop in `agent.py` is a `while iteration < MAX_ITER` that sends messages to the model with a `tools=` list. The model returns either a final answer (`finish_reason == "stop"`) or a set of tool calls (`finish_reason == "tool_calls"`).

When tool calls come back, I dispatch each one — `web_search`, `web_fetch`, `discover_papers`, `get_paper_content`, or `save_research_note` — append a `role: "tool"` message with the result, and loop again. The model sees all previous tool results and can decide whether it needs more or is ready to answer.

The tricky part is that the loop has to keep going until the model is satisfied, not just until it calls one tool. A question like "what are the latest advances in protein folding?" triggers at least 6–8 tool calls before the model has enough to synthesise an answer.

For the TUI, the agent runs in a background thread. UI updates go through `app.call_from_thread()` — if you call Textual widgets directly from a worker thread, you get race conditions and crashes. Learning this the hard way was the main debugging session of the week.

---

## One Design Decision: Depth Over Breadth in Source Reading

The system prompt instructs the model to read at least 2–3 full pages rather than relying on snippets, and to read one academic paper in full rather than just the abstract. I could have made the agent faster by stopping at snippets, but snippets are often misleading or incomplete.

The `web_fetch` function truncates to 8,000 characters — enough to get the substance of an article without blowing the context window. Trafilatura strips navigation, ads, and boilerplate, so those 8,000 characters are almost entirely useful content.

This came from noticing that early test runs produced confident-sounding but shallow answers whenever the agent only read snippets. After switching to full-page reads, the answers became noticeably more specific and accurate.

---

## What Surprised Me

The AlphaXiv MCP connection is async but the rest of the agent loop is sync. Getting these to coexist cleanly required wrapping every MCP call in `asyncio.run()`, which creates a new event loop per call. It works, but it feels brittle — if you're inside an async context already (like Textual's internal event loop), it deadlocks.

The fix: keep the MCP calls in their own isolated sync wrappers (`discover_papers`, `get_paper_content`) that each spin up and tear down a fresh event loop. Not elegant, but stable.

The other surprise was how much the system prompt matters for tool use. Without explicit instructions about *how many* sources to read and *when* to call `save_research_note`, the model would either stop too early (one web search, done) or go in circles re-searching the same query.

---

## What I'd Improve

**Parallel tool calls.** Right now the agent calls tools sequentially. If it decides it needs 3 web pages, it fetches them one at a time. The OpenAI API supports parallel tool calls in a single response — the model can return multiple `tool_calls` in one turn. I'd restructure the dispatch loop to execute these concurrently with `asyncio.gather` or `ThreadPoolExecutor`.

**Persistent session memory.** Each conversation starts fresh. For research that spans multiple sessions, it'd be useful to load previous notes from the `notes/` folder into the system prompt as context — so the agent builds on prior work rather than re-discovering the same papers.

**Streaming with tools.** The current implementation streams only the final synthesis. Mid-agent reasoning (the model's thinking before a tool call) is not visible to the user. Some providers expose this via streaming tool call deltas — wiring that up would make the right panel even more informative.
