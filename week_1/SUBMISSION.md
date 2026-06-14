# Week 1 Submission — GenAI/Agentic Track 2026

## What I Built

A terminal chatbot (`final_chatbot.py`) built around a `ChatAgent` class that:
- Holds multi-turn conversation by manually maintaining a `messages` list
- Lets the user pick from four free OpenRouter models at startup
- Streams tokens as they arrive instead of waiting for a full response
- Auto-compacts history when conversation exceeds `max_turns` pairs
- Supports `/reset`, `/compact`, `/tokens`, `/exit` commands

---

## Key Decisions and Why

### 1. `ChatAgent` as a class, not a script
I wrapped everything in a class so model, buffer size, system prompt, and streaming are all configurable at construction time. Calling `agent.run()` starts the loop; `agent.chat(text)` is usable programmatically. This makes it easy to swap the model or plug the agent into a larger system later.

### 2. Rolling buffer with auto-compaction (not truncation)
The spec offered two options: drop oldest pairs or summarise. I chose summarisation because truncation silently loses facts — the model might contradict itself without knowing why. When `_pair_count() > max_turns`, the agent calls the model again with a summarisation prompt, then replaces the entire history with a two-message "summary block". The cost is one extra API call; the benefit is coherence.

I also added a manual `/compact` command so you can trigger it intentionally before hitting the limit.

### 3. Streaming
Rather than collecting the full response and printing it at once, I used `stream=True` and printed each token delta as it arrived. This feels more responsive and mirrors how production chat UIs work. The trade-off: streaming responses don't return `response.usage` in a single object, so `/tokens` only works after a non-streaming call. I noted this in the output.

### 4. Model selection at startup
OpenRouter gives access to many free models. Rather than hardcoding one, I present a small numbered menu so the user can switch without editing source. The list is easy to extend.

### 5. API key hygiene
Key lives only in `.env`, loaded via `python-dotenv`. `.env` is in `.gitignore`. `.env.example` is committed instead, showing the variable name with a placeholder value — standard practice so collaborators know what to set without seeing real credentials.

---

## What I Learned

**The stateless API is the core insight of the week.** Every call starts from zero. The only "memory" the model has is what you pass in `messages`. Feeling this break when you `/reset` mid-conversation makes it visceral — the model acts like it never met you.

**Token budgets matter.** Even a moderate conversation can grow the prompt quickly. Compaction isn't just a feature; it's a necessity for long sessions. Summarisation is lossy but controlled; truncation is faster but fragile.

**Role alternation is strict.** Sending two consecutive `user` turns causes the API to either error or behave oddly. The `messages` list structure is not just a convention — the model's attention mechanism depends on the alternating pattern.

**Streaming changes the UX completely.** Waiting 3–5 seconds for a full response feels broken. Streaming the same response token-by-token feels alive. Worth the minor implementation complexity.

---

## What Didn't Work (and What I Learned From It)

- First attempt at compaction sent the entire transcript as a `user` message with no system framing, which made the model ramble. Adding an explicit instruction ("Summarise into a single compact paragraph…") fixed the output quality.
- Tried to surface `response.usage` in streaming mode — it's not available on each chunk, only on the final `[DONE]` event which the SDK doesn't expose cleanly. Documented the limitation instead of silently ignoring it.

---

## Checklist

- [x] Coherent multi-turn conversation, uses earlier context
- [x] Model selection before starting
- [x] `/exit` (and `exit`/`quit`) to end loop
- [x] `ChatAgent` class with config options (model, max_turns, system_prompt, stream)
- [x] API key from environment only — never in source
- [x] `.env` in `.gitignore`
- [x] `.env.example` committed
- [x] No external libraries beyond `openai` and `python-dotenv`
- [x] Bonus: auto-compaction + `/compact` command
- [x] Bonus: streaming
