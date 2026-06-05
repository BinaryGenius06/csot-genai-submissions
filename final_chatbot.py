import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENROUTER_API_KEY from .env

# approximate token counter (exact counts come from response.usage)
_ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(messages) -> int:
    total = 0
    for m in messages:
        total += len(_ENCODER.encode(m["content"])) + 4  # 4 = per-message overhead
    return total


# ANSI colors for terminal output
class C:
    RESET   = "\033[0m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    MAGENTA = "\033[35m"


class ChatAgent:

    def __init__(
        self,
        model: str,
        system_prompt: str = (
            "You are a concise, accurate assistant. Prefer short, well-structured "
            "answers. Avoid filler and excessive emojis. Always remember and use "
            "facts the user shares about themselves (name, studies, preferences, tools)."
        ),
        compact_token_limit: int = 500,  # compact when history exceeds this
        temperature: float = 0.7,
        stream: bool = True,
        auto_compact: bool = True,
    ):
        self.model = model
        self.compact_token_limit = compact_token_limit
        self.temperature = temperature
        self.stream = stream
        self.auto_compact = auto_compact

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],  # never hardcoded
        )

        self.base_system = system_prompt
        self.history = []       # user/assistant turns only
        self.user_facts = ""    # pinned facts, injected into system msg every call

        # token tracking
        self.last_usage = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _build_system(self):
        # attach pinned user facts to system message if any exist
        content = self.base_system
        if self.user_facts:
            content += "\n\nKnown facts about the user (always honor these):\n" + self.user_facts
        return {"role": "system", "content": content}

    def _update_user_facts(self, user_message: str):
        # side-call to extract durable personal facts from the user's message
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You maintain a memory of durable facts about a user "
                        "(name, location, studies, job, tools they use, stated "
                        "preferences and goals).\n"
                        "Given the CURRENT memory and a NEW user message, return "
                        "the UPDATED memory as a short bullet list (one fact per "
                        "line). Keep all still-valid old facts, add any new ones, "
                        "update changed ones. If the message contains no durable "
                        "personal facts and memory is empty, return the single "
                        "word NONE. Output only the list or NONE — no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CURRENT MEMORY:\n{self.user_facts or '(empty)'}\n\n"
                        f"NEW USER MESSAGE:\n{user_message}"
                    ),
                },
            ],
        )
        self._record_usage(getattr(resp, "usage", None))
        raw = resp.choices[0].message.content
        if not raw:
            return
        out = raw.strip()
        normalized = out.lower().strip(" .\n\t-*")
        if normalized == "none" or not out:
            return  # no new facts
        self.user_facts = out

    def _maybe_manage_history(self):
        # auto-compact if prompt size exceeds token budget
        full = [self._build_system()] + self.history
        tokens = count_tokens(full)
        if tokens > self.compact_token_limit and self.auto_compact:
            print(f"{C.DIM}(history ~{tokens} tokens > {self.compact_token_limit} limit -> auto-compacting...){C.RESET}")
            self.compact(silent=True)

    def compact(self, silent: bool = False):
        # summarise old turns; keep last 2 pairs verbatim
        if len(self.history) <= 4:
            if not silent:
                print(f"{C.DIM}(not enough history to compact yet){C.RESET}")
            return

        keep = self.history[-4:]
        to_summarize = self.history[:-4]

        convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)

        summary_resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the dialogue below into a few short bullet "
                        "points covering the topics discussed and any decisions "
                        "made. Preserve numbers and key conclusions. Be terse. "
                        "Invent nothing."
                    ),
                },
                {"role": "user", "content": convo_text},
            ],
        )
        summary = summary_resp.choices[0].message.content
        self._record_usage(getattr(summary_resp, "usage", None))

        # replace old history with summary + recent turns
        self.history = [
            {"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"}
        ] + keep

        if not silent:
            print(f"{C.GREEN}(history compacted; user facts pinned to memory){C.RESET}")

    def call_model(self, user_input: str) -> str:
        self._update_user_facts(user_input)             # pin facts before adding turn
        self.history.append({"role": "user", "content": user_input})
        self._maybe_manage_history()                    # compact if over budget

        messages = [self._build_system()] + self.history  # full prompt every call

        if self.stream:
            reply = self._call_streaming(messages)
        else:
            reply = self._call_blocking(messages)

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _call_blocking(self, messages) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        self._record_usage(resp.usage)
        return resp.choices[0].message.content

    def _call_streaming(self, messages) -> str:
        # print tokens live; collect usage from final chunk
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = []
        for event in stream:
            if event.usage is not None:
                self._record_usage(event.usage)
            if event.choices and event.choices[0].delta.content:
                piece = event.choices[0].delta.content
                print(piece, end="", flush=True)
                chunks.append(piece)
        print()
        return "".join(chunks)

    def _record_usage(self, usage):
        if usage is None:
            return
        self.last_usage = usage
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens

    def reset(self):
        self.history = []
        self.user_facts = ""  # wipe everything including pinned facts

    def print_memory(self):
        if self.user_facts:
            print(f"{C.GREEN}Pinned user facts:{C.RESET}\n{self.user_facts}")
        else:
            print(f"{C.DIM}(no user facts pinned yet){C.RESET}")

    def print_tokens(self):
        est = count_tokens([self._build_system()] + self.history)
        print(f"{C.YELLOW}Current buffer estimate: ~{est} tokens (compaction triggers above {self.compact_token_limit}){C.RESET}")
        if self.last_usage is None:
            print(f"{C.DIM}(no API calls made yet){C.RESET}")
            return
        u = self.last_usage
        total = self.total_prompt_tokens + self.total_completion_tokens
        print(f"{C.YELLOW}Last call  -> prompt: {u.prompt_tokens}, completion: {u.completion_tokens}, total: {u.total_tokens}{C.RESET}")
        print(f"{C.YELLOW}Session    -> prompt: {self.total_prompt_tokens}, completion: {self.total_completion_tokens}, total: {total}{C.RESET}")

    def print_history(self):
        if not self.history:
            print(f"{C.DIM}(history empty){C.RESET}")
            return
        for m in self.history:
            tag = C.CYAN if m["role"] == "user" else C.MAGENTA
            print(f"{tag}{m['role']}:{C.RESET} {m['content'][:80]}")


# ---- model picker ----
def pick_model() -> str:
    models = [
        "openrouter/free",                           # auto-routes to a working free model
        "deepseek/deepseek-v4-flash:free",
        "qwen/qwen3-coder:free",
        "openai/gpt-oss-120b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]
    print(f"{C.BOLD}Pick a model:{C.RESET}")
    for i, m in enumerate(models):
        print(f"  {i}) {m}")
    choice = input("Number > ").strip()
    try:
        return models[int(choice)]
    except (ValueError, IndexError):
        print(f"{C.DIM}Bad input -> defaulting to openrouter/free{C.RESET}")
        return models[0]


HELP_TEXT = f"""{C.BOLD}Commands:{C.RESET}
  /help     show this help
  /tokens   show token usage (buffer estimate + last call + session total)
  /memory   show durable user facts pinned outside the history
  /history  show current conversation buffer
  /compact  manually summarize old history into a short note
  /reset    wipe conversation history (feel the statelessness)
  exit/quit end the session
"""


def run_chatbot():
    model = pick_model()
    agent = ChatAgent(model=model, compact_token_limit=500)

    print(f"\n{C.GREEN}Chat started ({model}). Streaming ON. Compaction triggers above {agent.compact_token_limit} tokens.{C.RESET}")
    print(f"{C.DIM}Type /help for commands, 'exit' to quit.{C.RESET}\n")

    while True:
        user = input(f"{C.CYAN}[YOU] {C.RESET}").strip()
        if not user:
            continue

        if user.lower() in ("exit", "quit"):
            print(f"{C.GREEN}Goodbye!{C.RESET}")
            break
        elif user == "/help":
            print(HELP_TEXT)
        elif user == "/reset":
            agent.reset()
            print(f"{C.YELLOW}(history wiped){C.RESET}")
        elif user == "/tokens":
            agent.print_tokens()
        elif user == "/memory":
            agent.print_memory()
        elif user == "/history":
            agent.print_history()
        elif user == "/compact":
            agent.compact()
        else:
            # normal turn — streaming prints tokens inside call_model
            print(f"{C.MAGENTA}[MODEL] {C.RESET}", end="", flush=True)
            try:
                agent.call_model(user)
            except Exception as e:
                print(f"\n{C.YELLOW}Error: {e}{C.RESET}")
            print()


if __name__ == "__main__":
    run_chatbot()
