import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Available free models on OpenRouter
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = {
    "1": ("deepseek/deepseek-v4-flash:free",    "DeepSeek V4 Flash (default)"),
    "2": ("google/gemini-2.0-flash-exp:free",   "Gemini 2.0 Flash"),
    "3": ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B"),
    "4": ("mistralai/mistral-7b-instruct:free", "Mistral 7B"),
}

SYSTEM_PROMPT = "You are a helpful, concise assistant."


class ChatAgent:
    """
    Multi-turn chatbot with:
    - Configurable model (any OpenRouter model string)
    - Rolling buffer: keeps last N user+assistant pairs
    - Auto-compaction: summarises history when buffer overflows
    - Streaming output
    - Commands: /reset  /compact  /tokens  /exit
    """

    def __init__(
        self,
        model: str = "deepseek/deepseek-v4-flash:free",
        max_turns: int = 10,
        system_prompt: str = SYSTEM_PROMPT,
        stream: bool = True,
    ):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set in environment.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.max_turns = max_turns          # max user+assistant PAIRS to keep
        self.stream = stream
        self.system_prompt = system_prompt

        self._system_msg = {"role": "system", "content": system_prompt}
        self.history: list[dict] = []       # only user/assistant turns
        self._last_usage = None

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------
    def call_model(self, messages: list[dict]) -> str:
        """Send [system] + messages to the API. Returns assistant text."""
        payload = [self._system_msg] + messages

        if self.stream:
            return self._stream_response(payload)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=payload,
            )
            self._last_usage = response.usage
            return response.choices[0].message.content

    def _stream_response(self, payload: list[dict]) -> str:
        """Stream tokens to stdout; return full text."""
        full = []
        print("[MODEL] ", end="", flush=True)
        with self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                full.append(delta)
        print()  # newline after stream
        return "".join(full)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------
    def _pair_count(self) -> int:
        """Number of complete user+assistant pairs in history."""
        return len(self.history) // 2

    def _maybe_compact(self):
        """Auto-compact when pair count exceeds max_turns."""
        if self._pair_count() > self.max_turns:
            self._compact()

    def _compact(self):
        """Summarise current history into a single context message."""
        print("\n[SYSTEM] Compacting history…", flush=True)
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.history
        )
        summary_prompt = (
            "Summarise the following conversation into a single, compact "
            "paragraph that preserves all key facts the assistant needs:\n\n"
            + transcript
        )
        summary = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": summary_prompt}],
        ).choices[0].message.content

        self.history = [
            {"role": "user",      "content": f"[Conversation summary]\n{summary}"},
            {"role": "assistant", "content": "Understood. I have the context."},
        ]
        print("[SYSTEM] Compacted.\n", flush=True)

    # ------------------------------------------------------------------
    # Chat loop
    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        reply = self.call_model(self.history)
        self.history.append({"role": "assistant", "content": reply})
        self._maybe_compact()
        return reply

    def reset(self):
        self.history = []
        print("[SYSTEM] History cleared.\n")

    # ------------------------------------------------------------------
    # Run interactive session
    # ------------------------------------------------------------------
    def run(self):
        print(f"\nModel: {self.model}")
        print(f"Max turns before compact: {self.max_turns}")
        print("Commands: /reset  /compact  /tokens  /exit\n")

        while True:
            try:
                user_input = input("[YOU] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Commands
            if user_input.lower() in ("/exit", "exit", "quit"):
                print("Goodbye!")
                break
            elif user_input == "/reset":
                self.reset()
                continue
            elif user_input == "/compact":
                if self.history:
                    self._compact()
                else:
                    print("[SYSTEM] Nothing to compact.\n")
                continue
            elif user_input == "/tokens":
                if self._last_usage:
                    u = self._last_usage
                    print(
                        f"[TOKENS] prompt={u.prompt_tokens} "
                        f"completion={u.completion_tokens} "
                        f"total={u.total_tokens}\n"
                    )
                else:
                    print("[TOKENS] No usage data yet (streaming suppresses it).\n")
                continue

            # Normal turn — streaming already prints; skip re-print
            if not self.stream:
                reply = self.chat(user_input)
                print(f"[MODEL] {reply}\n")
            else:
                self.chat(user_input)
                print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def pick_model() -> str:
    print("Select a model:")
    for key, (model_id, label) in AVAILABLE_MODELS.items():
        print(f"  {key}. {label}")
    print("  (press Enter for default)\n")
    choice = input("Choice: ").strip()
    model_id, label = AVAILABLE_MODELS.get(choice, list(AVAILABLE_MODELS.values())[0])
    print(f"Using: {label}\n")
    return model_id


if __name__ == "__main__":
    model = pick_model()
    agent = ChatAgent(
        model=model,
        max_turns=10,
        system_prompt=SYSTEM_PROMPT,
        stream=True,
    )
    agent.run()
