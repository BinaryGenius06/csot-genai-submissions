"""
Build 3 — Textual TUI Chatbot
================================
Week 1 ChatAgent logic wrapped in a full-screen Textual UI.

Key bindings:
  Ctrl+L  — clear display (history unchanged)
  Ctrl+K  — clear display + history (fresh start)
  Ctrl+Q  — quit

Architecture note:
  The API call blocks, so it runs in a background worker thread.
  UI updates from threads must go through app.call_from_thread().
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog

load_dotenv()

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "qwen/qwen3-coder:free"
MAX_TURNS = 10
SYSTEM_PROMPT = "You are a helpful, concise assistant."

# ---------------------------------------------------------------------------
# Core chat logic (reused from Week 1, trimmed for TUI)
# ---------------------------------------------------------------------------
def call_model(client: OpenAI, messages: list[dict]) -> str:
    payload = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = client.chat.completions.create(model=MODEL, messages=payload)
    return response.choices[0].message.content


def trim_history(messages: list[dict], max_turns: int) -> list[dict]:
    """Keep only the last max_turns user+assistant pairs."""
    pairs = []
    for i in range(0, len(messages) - 1, 2):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            pairs.append((messages[i], messages[i + 1]))
    kept = pairs[-max_turns:]
    return [msg for pair in kept for msg in pair]


# ---------------------------------------------------------------------------
# TUI Application
# ---------------------------------------------------------------------------
class ChatTUI(App):
    """Full-screen Textual chatbot."""

    CSS = """
    Screen {
        background: $surface;
    }

    #chat-log {
        border: round $primary;
        height: 1fr;
        margin: 0 1;
        padding: 1;
        scrollbar-gutter: stable;
    }

    #input-bar {
        height: 3;
        margin: 0 1 1 1;
        border: round $accent;
    }

    #input-bar:focus-within {
        border: round $success;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_display", "Clear display"),
        Binding("ctrl+k", "clear_history", "Clear history"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.history: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield RichLog(id="chat-log", wrap=True, markup=True, highlight=True)
            yield Input(placeholder="Type a message… (Ctrl+L clear display, Ctrl+K clear history, Ctrl+Q quit)", id="input-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input-bar", Input).focus()
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold cyan]Chatbot ready.[/bold cyan] Model: " + MODEL)
        log.write("[dim]─" * 60 + "[/dim]")

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        input_widget = self.query_one("#input-bar", Input)
        input_widget.clear()
        input_widget.disabled = True  # block while waiting

        log = self.query_one("#chat-log", RichLog)
        log.write(f"\n[bold green]You:[/bold green] {user_text}")

        self.history.append({"role": "user", "content": user_text})
        self.run_worker(self._get_response, thread=True)

    def _get_response(self) -> None:
        """Runs in background thread — fetches model reply."""
        log = self.query_one("#chat-log", RichLog)

        self.app.call_from_thread(log.write, "[dim]Assistant is thinking…[/dim]")

        try:
            reply = call_model(self.client, self.history)
        except Exception as exc:
            reply = f"[ERROR] {exc}"

        self.history.append({"role": "assistant", "content": reply})
        self.history = trim_history(self.history, MAX_TURNS)

        def update_ui():
            # Remove the "thinking" line by clearing and rewriting is complex;
            # instead just write the reply (thinking line is harmless)
            log.write(f"[bold blue]Assistant:[/bold blue] {reply}\n")
            self.query_one("#input-bar", Input).disabled = False
            self.query_one("#input-bar", Input).focus()

        self.app.call_from_thread(update_ui)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def action_clear_display(self) -> None:
        """Ctrl+L — wipe the visual log, keep history."""
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        log.write("[dim]Display cleared. History intact.[/dim]")

    def action_clear_history(self) -> None:
        """Ctrl+K — wipe display AND conversation history."""
        self.history = []
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        log.write("[bold yellow]History cleared.[/bold yellow] Fresh conversation started.")

    def action_quit(self) -> None:
        self.exit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = ChatTUI()
    app.run()
