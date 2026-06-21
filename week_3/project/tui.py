"""
TUIAgent — full-screen Textual UI, inheriting from Agent.
==============================================================
Reuses the Week 2 split-panel layout (chat log left, tool feed right) but
routes every query through the inherited Agent.chat() / _run_loop() instead
of a standalone run_streaming_agent() function. TUIAgent only overrides the
_emit_*() presentation hooks and adds the Textual App — no tool dispatch or
loop logic lives here (see week_3/2_agent_class.md: "don't over-inherit").

Run:
  python agent.py --tui
  python agent.py --tui --session abc123
"""

import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, RichLog, Static

from agent import Agent, save_session

TOOL_ICONS = {
    "web_search": "🔍",
    "web_fetch": "🌐",
    "paper_search": "📄",
    "read_paper": "📖",
    "read_file": "📂",
    "write_file": "💾",
    "list_files": "🗂",
    "edit_file": "✏️",
}


class TUIAgent(Agent):
    """
    Agent subclass that overrides the _emit_*() hooks to push updates into
    Textual widgets instead of printing. chat() and _run_loop() are
    inherited from Agent, unchanged — this class adds presentation only.
    """

    def __init__(self, session=None, app: "ResearchDeskApp" = None):
        super().__init__(session=session)
        self._app = app  # set by ResearchDeskApp after construction

    def _call_on_main_thread(self, fn, *args) -> None:
        """
        call_from_thread only works when called FROM a background thread —
        it raises RuntimeError if we're already on the app's main thread
        (e.g. dispatch() invoked directly from a key binding like Ctrl+S,
        rather than from the background chat() worker thread). Detect which
        thread we're on and call directly when we're already on the main one.
        """
        if threading.get_ident() == self._app._thread_id:
            fn(*args)
        else:
            self._app.call_from_thread(fn, *args)

    def _emit_tool_start(self, name: str, args: dict) -> None:
        if self._app is None:
            return
        icon = TOOL_ICONS.get(name, "🔧")
        first_arg = next(iter(args.values()), "") if args else ""
        preview = str(first_arg)[:60]
        self._call_on_main_thread(self._app.push_tool_start, icon, name, preview)

    def _emit_tool_done(self, name: str, args: dict, result: dict) -> None:
        if self._app is None:
            return
        if "error" in result:
            preview = f"[error] {str(result['error'])[:120]}"
            ok = False
        else:
            preview = str(result.get("content", ""))[:120].replace("\n", " ")
            ok = True
        self._call_on_main_thread(self._app.push_tool_done, preview, ok)

    def run(self) -> None:
        """Entry point for python agent.py --tui."""
        app = ResearchDeskApp(agent=self)
        self._app = app
        app.run()


# ---------------------------------------------------------------------------
# Textual App — layout, bindings, background worker. Calls self.agent.chat()
# (inherited Agent method) on a background thread; no tool/loop logic here.
# ---------------------------------------------------------------------------
class ResearchDeskApp(App):
    """Split-panel research terminal wrapping a TUIAgent."""

    CSS = """
    Screen { background: $surface; }
    #main-row { height: 1fr; }
    #left-panel {
        width: 2fr; border: round $primary; margin: 0 0 0 1; padding: 1;
    }
    #right-panel {
        width: 1fr; border: round $accent; margin: 0 1 0 0; padding: 1;
    }
    #chat-log { height: 1fr; }
    #tool-log { height: 1fr; }
    #left-title { color: $primary; text-style: bold; margin-bottom: 1; }
    #right-title { color: $accent; text-style: bold; margin-bottom: 1; }
    #input-row { height: 3; margin: 0 1 1 1; }
    #input-bar { width: 1fr; border: round $success; }
    #status-label {
        width: 24; height: 3; content-align: center middle;
        color: $text-muted; border: round $surface-darken-1; margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_display", "Clear display"),
        Binding("ctrl+k", "clear_history", "Clear history"),
        Binding("ctrl+s", "save_note", "Save note"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    status = reactive("Ready")

    def __init__(self, agent: TUIAgent):
        super().__init__()
        self.agent = agent
        self.last_answer = ""
        self.last_question = ""
        self._lock = threading.Lock()

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
            f"[bold cyan]Research Desk[/bold cyan]  "
            f"[dim](session {self.agent.session['id']}: {self.agent.session['title']})[/dim]\n"
            "[dim]Ask a research question. The agent searches the web and papers, "
            "reads sources, and can save notes.[/dim]\n"
        )
        self.query_one("#tool-log", RichLog).write("[dim]Tool calls will appear here as the agent works.[/dim]\n")

    def _set_status(self, text: str) -> None:
        self.query_one("#status-label", Static).update(text)

    # ------------------------------------------------------------------
    # Input submitted -> background worker -> agent.chat() (inherited)
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
        self.query_one("#tool-log", RichLog).write("\n[dim]── New query ──[/dim]")

        thread = threading.Thread(target=self._run_chat, args=(query,), daemon=True)
        thread.start()

    def _run_chat(self, query: str) -> None:
        """Runs on a background thread. Calls the inherited Agent.chat()."""
        try:
            answer = self.agent.chat(query)
            self.call_from_thread(self._on_done, answer)
        except Exception as exc:
            self.call_from_thread(self._on_error, str(exc))

    # ------------------------------------------------------------------
    # Hooks called by TUIAgent._emit_tool_start/_emit_tool_done via
    # call_from_thread — always run on the main thread, safe to touch widgets.
    # ------------------------------------------------------------------
    def push_tool_start(self, icon: str, name: str, preview: str) -> None:
        self.query_one("#tool-log", RichLog).write(f"{icon} [bold]{name}[/bold]\n   [dim]{preview}[/dim]")
        self._set_status(f"⟳ {name[:16]}…")

    def push_tool_done(self, preview: str, ok: bool) -> None:
        color = "green" if ok else "red"
        mark = "✓" if ok else "✗"
        self.query_one("#tool-log", RichLog).write(f"   [{color}]{mark}[/{color}] {preview}\n")

    def _on_done(self, answer: str) -> None:
        self.last_answer = answer
        self.query_one("#chat-log", RichLog).write(answer)
        self.query_one("#input-bar", Input).disabled = False
        self.query_one("#input-bar", Input).focus()
        self._set_status("✓ Done")
        self.query_one("#tool-log", RichLog).write("[green]✓ Agent finished.[/green]\n")

    def _on_error(self, error: str) -> None:
        self.query_one("#chat-log", RichLog).write(f"\n[bold red]Error:[/bold red] {error}\n")
        self.query_one("#input-bar", Input).disabled = False
        self.query_one("#input-bar", Input).focus()
        self._set_status("✗ Error")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def action_clear_display(self) -> None:
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#tool-log", RichLog).clear()
        self.query_one("#chat-log", RichLog).write("[dim]Display cleared.[/dim]")
        self._set_status("Cleared")

    def action_clear_history(self) -> None:
        """Clears the display AND starts a fresh session (new id)."""
        from agent import create_session
        self.last_answer = ""
        self.last_question = ""
        self.agent.session = create_session()
        self.agent.messages = []
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#tool-log", RichLog).clear()
        self.query_one("#chat-log", RichLog).write(
            f"[bold yellow]History cleared.[/bold yellow] New session: {self.agent.session['id']}\n"
        )
        self._set_status("Reset")

    def action_save_note(self) -> None:
        """Manual save — writes the last answer to notes/ via the file tool."""
        if not self.last_answer:
            self.query_one("#chat-log", RichLog).write("[yellow]Nothing to save yet.[/yellow]")
            return
        import re
        from datetime import datetime

        slug = re.sub(r"[^\w\s-]", "", self.last_question.lower())
        slug = re.sub(r"[\s_]+", "-", slug)[:60] or "note"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = f"notes/{ts}-{slug}.md"
        content = f"# {self.last_question}\n\n{self.last_answer}\n"
        result = self.agent.dispatch("write_file", {"path": path, "content": content})

        if "error" in result:
            self.query_one("#chat-log", RichLog).write(f"\n[red]Save failed: {result['error']}[/red]\n")
            self._set_status("Save failed")
        else:
            self.query_one("#chat-log", RichLog).write(f"\n[green]💾 Saved to {path}[/green]\n")
            self._set_status("Saved ✓")

    def action_quit(self) -> None:
        save_session(self.agent.session)
        self.exit()


if __name__ == "__main__":
    TUIAgent().run()
