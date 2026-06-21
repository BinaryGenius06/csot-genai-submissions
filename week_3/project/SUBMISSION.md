# Week 3 Submission — Research Desk

## What I made

A research agent called Research Desk. It's an upgrade of my Week 2 agent.
Last week it could search the web and read pages, but it forgot everything
the moment I closed the terminal. This week I gave it memory — it saves
conversations to disk, follows a rules file, and can search academic papers
on top of the web.

## Purpose

The goal was to take a "dumb" agent (one that only responds to whatever's
in front of it right now) and turn it into something closer to a real
assistant — one that remembers past conversations, follows consistent
rules every time it runs, and can read and write its own notes.

## Tech stack

- Python
- OpenRouter API (model: `nvidia/nemotron-3-super-120b-a12b:free`) for the LLM
- Serper API for web search
- Hugging Face Papers API for academic paper search (replaces the AlphaXiv
  MCP server I used in Week 2)
- `requests` + `trafilatura` for fetching and cleaning web pages
- Textual for the terminal UI
- `nanoid` for generating session IDs
- Plain JSON files for storing sessions (no database)

## How it works

I built three classes:

- **Agent** — the actual brain. It holds the conversation, the tool list,
  and the loop that talks to the model and runs tools when asked.
- **REPLAgent** — inherits from Agent, adds a terminal interface (type a
  question, get an answer, or chat back and forth).
- **TUIAgent** — inherits from Agent, adds the full-screen visual interface
  (built with Textual) with a chat panel and a tool activity panel.

The important part is that Agent itself never prints anything or asks for
input directly — it just does the work. REPLAgent and TUIAgent each decide
how to *show* what Agent is doing. That way the same brain runs in three
modes (one question and exit, a back-and-forth chat, or the full UI)
without copy-pasting the logic three times.

The agent has 8 tools total:
- `web_search`, `web_fetch` — search Google and read a page
- `paper_search`, `read_paper` — search and read academic papers
- `read_file`, `write_file`, `list_files`, `edit_file` — read and write its
  own notes folder

## Workflow

1. I run the agent (CLI, REPL, or TUI).
2. It loads `AGENTS.md`, a plain text file of rules I wrote (when to use
   papers vs. web search, how to cite sources, where to save notes). These
   get added to the system prompt every single time, so I can edit the
   rules without touching code.
3. I ask a question.
4. The model decides which tool to call — usually `web_search` or
   `paper_search` first, then it reads a couple of sources, then answers
   with citations.
5. It saves a summary to a `notes/` folder on its own, no prompting needed.
6. The whole conversation (including every tool call) gets saved to a
   session file in `.agent/sessions/`.
7. Next time, I can resume that exact session by its ID and the agent
   remembers everything — what we talked about, what it already searched.

## What worked well

- Session memory actually works. I asked something, closed the program,
  reopened it with the saved session ID, and asked "what did we just
  discuss?" — it answered correctly with zero new searches, just from the
  saved conversation.
- The rules file (`AGENTS.md`) genuinely changes the agent's behavior. For
  an ML question it used paper search first; for a non-ML question (guitar
  amps) it skipped papers completely and went straight to web search — 
  exactly like the rules said to.
- The new paper tools worked first try against the real Hugging Face API
  — search returned real papers with correct titles and IDs, and reading a
  paper pulled the actual full text, not just the abstract.
- All three ways of running it (one-shot, REPL, full UI) worked using the
  exact same underlying code.

## Problem I faced

When I tested the full-screen UI and pressed Ctrl+S to manually save a
note, the whole program crashed with a threading error
(`call_from_thread must run in a different thread from the app`).

## How I fixed it

The crash happened because of how the Textual library handles background
threads. My code always assumed tool results were coming from a background
thread (which is true while the agent is actively answering a question)
and tried to hand the update off using a function called
`call_from_thread`. But Ctrl+S runs directly on the main thread, not a
background one — so that hand-off function had nowhere to "hand off" to,
and Textual threw an error instead of silently doing nothing.

I fixed it by checking which thread the code is currently running on
before deciding what to do: if it's already the main thread, just update
the screen directly; if it's a background thread, use the hand-off
function like before. Tested it again after the fix and Ctrl+S now saves
properly with no crash.

## Bonus features

I went back and did all three bonus challenges:

- **`/sessions` and `/resume <id>` in the REPL** — typing `/sessions` lists
  every saved session (id, last updated, title). Typing `/resume <id>`
  switches the currently running REPL onto that session without restarting
  the program — it swaps in the saved messages and you can keep chatting
  from where that conversation left off.
- **Auto-title** — after the first question in a brand new session, the
  agent quietly asks the model for a short (5-word-ish) title and saves it,
  so sessions stop being called "Untitled" forever. It only fires once per
  session and never overwrites a title that's already set (so resuming an
  old session won't accidentally rename it).
- **OpenCode-inspired detail** — I read OpenCode's architecture docs and
  noticed they make the agent's internal "budget" visible to the user
  instead of hiding it until something silently fails. My loop had a hard
  cap of 20 tool calls per question (`MAX_ITER`), but gave zero warning
  before hitting it — it would just return a generic error at call 21. I
  added an iteration counter that the REPL now prints once it's close to
  the limit (last 2 calls), so if a question is using up an unusually large
  number of tool calls, you see that happening instead of it failing out of
  nowhere. Stays silent for normal short questions.

Everything required by the assignment, plus all three bonus challenges,
is done and tested.
