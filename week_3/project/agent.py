"""
Research Desk — Week 3 Project
===============================
Class hierarchy:
  Agent       — brain: chat(), _run_loop(), dispatch(), sessions
  REPLAgent   — terminal REPL + one-shot CLI
  TUIAgent    — Textual UI (in tui.py)

Agent owns everything except how the user talks to it:
  - messages / session state, saved to .agent/sessions/{id}.json after every turn
  - AGENTS.md loaded into the system prompt (procedural memory, Lesson 1)
  - the tool registry and dispatch() routing (8 tools across web/papers/files)
  - _run_loop(): the OpenRouter tool-calling loop, lifted from Week 2's
    run_research_agent() and generalized to work with any tool module

Agent has no input(), no print(), no Textual imports. REPLAgent and TUIAgent
only add *how the user talks to the agent* — see week_3/2_agent_class.md.

Usage:
  python agent.py                              # REPLAgent.run()
  python agent.py "What is quantum computing?" # REPLAgent.run_once()
  python agent.py --tui                        # TUIAgent.run()
  python agent.py --session abc123 "continue"  # resume a session
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from nanoid import generate as _nanoid_generate
except ImportError:
    _nanoid_generate = None

from tools.web import web_search, web_fetch, WEB_TOOL_SCHEMAS
from tools.papers import paper_search, read_paper, PAPER_TOOL_SCHEMAS
from tools.files import read_file, write_file, list_files, edit_file, FILE_TOOL_SCHEMAS

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
SESSIONS_DIR = PROJECT_ROOT / ".agent" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "notes").mkdir(exist_ok=True)

MODEL = os.environ.get("AGENT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
MAX_ITER = 20  # max tool-calling iterations per chat() call

BASE_PROMPT = (
    "You are Research Desk, a rigorous research assistant. "
    "You search the web and academic papers, read primary sources, "
    "and save findings to notes/ so future sessions can pick up the thread.\n\n"
    "Tool routing:\n"
    "- ML paper / literature questions -> paper_search, then read_paper\n"
    "- Current events, blogs, docs -> web_search, then web_fetch\n"
    "- Save new findings -> write_file('notes/...')\n"
    "- Update an existing note -> read_file then edit_file\n"
    "- Recall past work -> list_files('notes/') then read_file\n"
    "Do ONE web_search or paper_search first. Read 1-3 sources. Then stop "
    "searching and synthesise a cited answer. Save a note if the findings "
    "are worth keeping for later."
)


# ---------------------------------------------------------------------------
# Session I/O (episodic memory — Lesson 1)
# ---------------------------------------------------------------------------
def new_session_id() -> str:
    if _nanoid_generate is not None:
        return _nanoid_generate(size=6)
    return os.urandom(4).hex()[:6]


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def create_session(title: str = "Untitled") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": new_session_id(),
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def save_session(session: dict) -> None:
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    _session_path(session["id"]).write_text(
        json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_session(session_id: str) -> dict | None:
    path = _session_path(session_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions() -> list[dict]:
    out = []
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append({
                "id": data.get("id", path.stem),
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# AGENTS.md — procedural memory (Lesson 1)
# ---------------------------------------------------------------------------
def load_agents_md() -> str | None:
    for rel in ("AGENTS.md", ".agent/AGENTS.md"):
        path = PROJECT_ROOT / rel
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def build_system_prompt() -> str:
    parts = [BASE_PROMPT]
    rules = load_agents_md()
    if rules:
        parts.append(f"## Project rules\n{rules}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool registry — all 8 tools, OpenAI schema + dispatch table
# ---------------------------------------------------------------------------
TOOLS = WEB_TOOL_SCHEMAS + PAPER_TOOL_SCHEMAS + FILE_TOOL_SCHEMAS

_TOOL_FUNCTIONS = {
    "web_search": lambda a: web_search(a.get("query", ""), a.get("num", 5)),
    "web_fetch": lambda a: web_fetch(a.get("url", "")),
    "paper_search": lambda a: paper_search(a.get("query", ""), a.get("limit", 5)),
    "read_paper": lambda a: read_paper(a.get("arxiv_id", "")),
    "read_file": lambda a: read_file(a.get("path", ""), a.get("start_line", 1), a.get("read_lines", 200)),
    "write_file": lambda a: write_file(a.get("path", ""), a.get("content", "")),
    "list_files": lambda a: list_files(a.get("pattern", "**/*"), a.get("path", ".")),
    "edit_file": lambda a: edit_file(
        a.get("path", ""), a.get("operation", ""), a.get("start_line", 1),
        a.get("content"), a.get("end_line"),
    ),
}


# ---------------------------------------------------------------------------
# Agent — the brain. No input(), no print(), no Textual.
# ---------------------------------------------------------------------------
class Agent:
    """
    Owns: messages, session I/O, tool registry, dispatch(), _run_loop(), chat().
    Subclasses (REPLAgent, TUIAgent) only add how the user talks to it.
    """

    def __init__(self, session: dict | None = None):
        self.client = self._make_client()
        self.session = session or create_session()
        # messages excludes the system prompt — that's rebuilt fresh each
        # call so AGENTS.md edits take effect without starting a new session
        self.messages: list[dict] = list(self.session.get("messages", []))

    @staticmethod
    def _make_client() -> OpenAI:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise EnvironmentError("OPENROUTER_API_KEY not set")
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    # ------------------------------------------------------------------
    # Hooks subclasses may override. Defaults are no-ops so Agent works
    # headlessly (e.g. under test) without a UI attached.
    # ------------------------------------------------------------------
    def _emit_tool_start(self, name: str, args: dict) -> None:
        pass

    def _emit_tool_done(self, name: str, args: dict, result: dict) -> None:
        pass

    def _emit_token(self, token: str) -> None:
        pass

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> dict:
        """Route a tool call by name. Always returns a dict, never raises."""
        fn = _TOOL_FUNCTIONS.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = fn(args)
        except Exception as exc:
            result = {"error": f"{name} raised: {exc}"}
        self._emit_tool_done(name, args, result)
        return result

    # ------------------------------------------------------------------
    # The agent loop — generalized from Week 2's run_research_agent()
    # ------------------------------------------------------------------
    def _run_loop(self) -> str:
        system_prompt = build_system_prompt()

        for _ in range(MAX_ITER):
            api_messages = [{"role": "system", "content": system_prompt}] + self.messages

            response = None
            for attempt in range(5):
                try:
                    response = self.client.chat.completions.create(
                        model=MODEL,
                        messages=api_messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 4:
                        time.sleep(35)
                    else:
                        raise

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                self.messages.append(choice.message.model_dump(exclude_none=True))
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    self._emit_tool_start(tc.function.name, args)
                    result = self.dispatch(tc.function.name, args)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                continue

            answer = choice.message.content or ""
            self._emit_token(answer)
            self.messages.append({"role": "assistant", "content": answer})
            return answer

        return "[ERROR] Agent exceeded max iterations without a final answer."

    # ------------------------------------------------------------------
    # chat() — the one method REPLAgent and TUIAgent both call identically
    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        answer = self._run_loop()
        self.session["messages"] = self.messages
        save_session(self.session)
        return answer


# ---------------------------------------------------------------------------
# REPLAgent — terminal: interactive REPL + one-shot CLI.
# Adds only how the user talks to the agent. No loop logic here — chat()
# and _run_loop() are inherited unchanged from Agent.
# ---------------------------------------------------------------------------
class REPLAgent(Agent):

    def _emit_tool_start(self, name: str, args: dict) -> None:
        first_arg = next(iter(args.values()), "") if args else ""
        preview = str(first_arg)[:60]
        print(f"  ⚙  {name}({preview})")

    def _emit_tool_done(self, name: str, args: dict, result: dict) -> None:
        if "error" in result:
            print(f"     → [error] {str(result['error'])[:150]}")
        else:
            preview = str(result.get("content", ""))[:150].replace("\n", " ")
            print(f"     → {preview}{'…' if len(str(result.get('content', ''))) > 150 else ''}")

    def run_once(self, query: str) -> str:
        """Single question, print answer, exit. Used for: python agent.py '<question>'"""
        print(f"\nResearching: {query}\n{'─' * 60}")
        answer = self.chat(query)
        print("\n" + "═" * 60)
        print("ANSWER")
        print("═" * 60)
        print(answer)
        return answer

    def run(self) -> None:
        """Interactive REPL. Used for: python agent.py"""
        print(f"Research Desk [session {self.session['id']}: {self.session['title']}]")
        print("Type your research question, or /quit to exit.\n")
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not query:
                continue
            if query in ("/quit", "/exit"):
                print("Goodbye.")
                break
            answer = self.chat(query)
            print(f"\n{answer}\n")


# ---------------------------------------------------------------------------
# Entry point — dispatches to the right subclass by parsing argv.
# ---------------------------------------------------------------------------
def main() -> None:
    args = sys.argv[1:]

    if "--tui" in args:
        from tui import TUIAgent  # local import: agent.py must not require Textual
        session = None
        if "--session" in args:
            idx = args.index("--session")
            session = load_session(args[idx + 1])
            if session is None:
                print(f"No session found with id '{args[idx + 1]}' — starting fresh.")
        TUIAgent(session=session).run()
        return

    session = None
    if "--session" in args:
        idx = args.index("--session")
        session_id = args[idx + 1]
        session = load_session(session_id)
        if session is None:
            print(f"No session found with id '{session_id}' — starting fresh.")
        # strip --session <id> from args so it doesn't get treated as the query
        args = args[:idx] + args[idx + 2:]

    repl = REPLAgent(session=session)

    # remaining positional args (if any) form a one-shot query
    query = " ".join(a for a in args if not a.startswith("--"))
    if query.strip():
        repl.run_once(query.strip())
    else:
        repl.run()


if __name__ == "__main__":
    main()
