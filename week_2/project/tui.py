"""
Perplexity TUI — Split-panel research terminal
================================================
Left panel  : conversation (streaming tokens)
Right panel : live tool call feed (updates as agent works)
Bottom bar  : input + status

Key bindings:
  Ctrl+L  — clear display
  Ctrl+K  — clear display + history
  Ctrl+S  — save current answer as research note (manual trigger)
  Ctrl+Q  — quit

Run:
    python project/tui.py
"""

import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, RichLog, Static

from agent import (
    LOCAL_TOOLS,
    MAX_ITER,
    NOTES_DIR,
    SYSTEM_PROMPT,
    dispatch_tool,
    make_client,
    save_research_note,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Streaming agent loop (token-by-token)
# ---------------------------------------------------------------------------
def run_streaming_agent(
    query: str,
    client: OpenAI,
    on_token,          # callback(str)
    on_tool_start,     # callback(name, args)
    on_tool_done,      # callback(name, args, result)
    on_done,           # callback(full_answer: str)
    on_error,          # callback(str)
):
    """
    Runs the full agent loop in a background thread.
    Final answer is streamed token-by-token via on_token.
    """
    messages = [{"role": "user", "content": query}]
    full_answer = []

    try:
        for iteration in range(MAX_ITER):
            # Non-streaming for tool-calling iterations; stream only final answer
            for attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model=os.environ.get("AGENT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                        tools=LOCAL_TOOLS,
                        tool_choice="auto",
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 4:
                        on_tool_start("⏳ rate-limit", {"waiting": "35s, retrying…"})
                        time.sleep(35)
                    else:
                        raise
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    on_tool_start(tc.function.name, args)
                    result = dispatch_tool(tc.function.name, args)
                    on_tool_done(tc.function.name, args, result)
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
                continue

            if choice.finish_reason == "stop":
                # Stream the final answer token by token
                final_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
                # Add assistant's last message so it knows what it said
                if choice.message.content:
                    # Already have the answer — stream it character by character
                    # (some providers don't support streaming with tools; simulate)
                    answer = choice.message.content
                    chunk_size = 4
                    for i in range(0, len(answer), chunk_size):
                        on_token(answer[i:i + chunk_size])
                    full_answer.append(answer)
                else:
                    # Try streaming a fresh final call
                    with client.chat.completions.create(
                        model=os.environ.get("AGENT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
                        messages=messages,
                        stream=True,
                    ) as stream:
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content or ""
                            if delta:
                                on_token(delta)
                                full_answer.append(delta)
                break

            # Unexpected finish — surface whatever content we have
            content = choice.message.content or "[No content]"
            on_token(content)
            full_answer.append(content)
            break

        on_done("".join(full_answer))

    except Exception as exc:
        on_error(str(exc))


# ---------------------------------------------------------------------------
# Tool feed entry widget
# ---------------------------------------------------------------------------
TOOL_ICONS = {
    "web_search":         "🔍",
    "web_fetch":          "🌐",
    "discover_papers":    "📄",
    "get_paper_content":  "📖",
    "save_research_note": "💾",
}


# ---------------------------------------------------------------------------
# Main TUI Application
# ---------------------------------------------------------------------------
class PerplexityTUI(App):
    """Split-panel AI research terminal."""

    CSS = """
    Screen {
        background: $surface;
    }

    /* ── Layout ── */
    #main-row {
        height: 1fr;
    }

    #left-panel {
        width: 2fr;
        border: round $primary;
        margin: 0 0 0 1;
        padding: 1;
    }

    #right-panel {
        width: 1fr;
        border: round $accent;
        margin: 0 1 0 0;
        padding: 1;
    }

    #chat-log {
        height: 1fr;
    }

    #tool-log {
        height: 1fr;
    }

    #left-title {
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    #right-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    /* ── Input ── */
    #input-row {
        height: 3;
        margin: 0 1 1 1;
    }

    #input-bar {
        width: 1fr;
        border: round $success;
    }

    #status-label {
        width: 24;
        height: 3;
        content-align: center middle;
        color: $text-muted;
        border: round $surface-darken-1;
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_display",  "Clear display"),
        Binding("ctrl+k", "clear_history",  "Clear history"),
        Binding("ctrl+s", "save_note",      "Save note"),
        Binding("ctrl+q", "quit",           "Quit"),
    ]

    status = reactive("Ready")

    def __init__(self):
        super().__init__()
        self.client        = make_client()
        self.history       = []          # raw Q&A pairs for display
        self.last_answer   = ""
        self.last_question = ""
        self._lock         = threading.Lock()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-row"):
            with Vertical(id="left-panel"):
                yield Label("💬  Research Chat", id="left-title")
                yield RichLog(id="chat-log", wrap=True, markup=True, highlight=True)
            with Vertical(id="right-panel"):
                yield Label("⚙  Live Tool Feed", id="right-title")
                yield RichLog(id="tool-log", wrap=True, markup=True)
        with Horizontal(id="input-row"):
            yield Input(
                placeholder="Ask anything… (Ctrl+L clear, Ctrl+K reset, Ctrl+S save note, Ctrl+Q quit)",
                id="input-bar",
            )
            yield Static(id="status-label")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input-bar", Input).focus()
        self._set_status("Ready")
        chat = self.query_one("#chat-log", RichLog)
        chat.write(
            "[bold cyan]Perplexity Terminal[/bold cyan]\n"
            "[dim]Ask a research question. The agent will search the web, "
            "read pages, and find academic papers.[/dim]\n"
        )
        tool = self.query_one("#tool-log", RichLog)
        tool.write("[dim]Tool calls will appear here as the agent works.[/dim]\n")

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.query_one("#status-label", Static).update(text)

    # ------------------------------------------------------------------
    # Input submitted
    # ------------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        inp = self.query_one("#input-bar", Input)
        inp.clear()
        inp.disabled = True
        self._set_status("⟳ Thinking…")

        self.last_question = query
        chat = self.query_one("#chat-log", RichLog)
        chat.write(f"\n[bold green]You:[/bold green] {query}")
        chat.write("[dim]─[/dim]" * 40)
        chat.write("[bold blue]Assistant:[/bold blue]")

        tool = self.query_one("#tool-log", RichLog)
        tool.write(f"\n[dim]── New query ──[/dim]")

        # Launch in background thread
        thread = threading.Thread(
            target=run_streaming_agent,
            kwargs=dict(
                query=query,
                client=self.client,
                on_token=self._cb_token,
                on_tool_start=self._cb_tool_start,
                on_tool_done=self._cb_tool_done,
                on_done=self._cb_done,
                on_error=self._cb_error,
            ),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Streaming callbacks (called from background thread)
    # ------------------------------------------------------------------
    def _cb_token(self, token: str) -> None:
        with self._lock:
            self._token_buf = getattr(self, "_token_buf", "") + token
        def _update():
            pass  # tokens flushed in _cb_done
        self.app.call_from_thread(_update)

    def _cb_tool_start(self, name: str, args: dict) -> None:
        icon = TOOL_ICONS.get(name, "🔧")
        first_arg = next(iter(args.values()), "") if args else ""
        preview   = str(first_arg)[:60]
        def _update():
            self.query_one("#tool-log", RichLog).write(
                f"{icon} [bold]{name}[/bold]\n"
                f"   [dim]{preview}[/dim]"
            )
            self._set_status(f"⟳ {name[:16]}…")
        self.app.call_from_thread(_update)

    def _cb_tool_done(self, name: str, args: dict, result: str) -> None:
        lines     = result.strip().splitlines()
        preview   = lines[0][:80] if lines else "[empty]"
        n_lines   = len(lines)
        def _update():
            self.query_one("#tool-log", RichLog).write(
                f"   [green]✓[/green] {preview}"
                + (f"  [dim]({n_lines} lines)[/dim]" if n_lines > 1 else "")
                + "\n"
            )
        self.app.call_from_thread(_update)

    def _cb_done(self, answer: str) -> None:
        self.last_answer = answer
        self._token_buf = ""
        def _update():
            chat = self.query_one("#chat-log", RichLog)
            chat.write(answer)
            self.query_one("#input-bar", Input).disabled = False
            self.query_one("#input-bar", Input).focus()
            self._set_status("✓ Done")
            self.query_one("#tool-log", RichLog).write(
                "[green]✓ Agent finished.[/green]\n"
            )
        self.app.call_from_thread(_update)

    def _cb_error(self, error: str) -> None:
        def _update():
            self.query_one("#chat-log", RichLog).write(
                f"\n[bold red]Error:[/bold red] {error}\n"
            )
            self.query_one("#input-bar", Input).disabled = False
            self.query_one("#input-bar", Input).focus()
            self._set_status("✗ Error")
        self.app.call_from_thread(_update)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def action_clear_display(self) -> None:
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#tool-log", RichLog).clear()
        self.query_one("#chat-log", RichLog).write("[dim]Display cleared.[/dim]")
        self._set_status("Cleared")

    def action_clear_history(self) -> None:
        self.last_answer   = ""
        self.last_question = ""
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#tool-log", RichLog).clear()
        self.query_one("#chat-log", RichLog).write(
            "[bold yellow]History cleared.[/bold yellow] Fresh session started.\n"
        )
        self._set_status("Reset")

    def action_save_note(self) -> None:
        if not self.last_answer:
            self.query_one("#chat-log", RichLog).write(
                "[yellow]Nothing to save yet.[/yellow]"
            )
            return
        title   = self.last_question[:60] or "research-note"
        content = f"**Query:** {self.last_question}\n\n{self.last_answer}"
        result  = save_research_note(title, content)
        self.query_one("#chat-log", RichLog).write(
            f"\n[green]💾 {result}[/green]\n"
        )
        self._set_status("Saved ✓")

    def action_quit(self) -> None:
        self.exit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = PerplexityTUI()
    app.run()
