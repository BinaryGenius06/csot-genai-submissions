# Week 1 Submission — GenAI/Agentic Track 2026

## What I Built

A terminal chatbot (`final_chatbot.py`) using the OpenRouter API. It holds a multi-turn conversation, remembers things you tell it, and manages its own memory when the conversation gets too long. Built around a `ChatAgent` class so everything is configurable.

---

## How I Approached It

I started with the basics — just getting an API call to work and printing the response. Then I added the conversation loop, then history management, then kept layering on top.

### Conversation history

The API is stateless — it remembers nothing between calls. So I maintain a `messages` list manually and resend the full history every time. This is the core mechanic everything else builds on.

### ChatAgent class

Wrapped everything in a class so model, token limit, system prompt, and streaming are all set at construction time. Makes it easy to swap models or reuse the agent elsewhere.

### Model selection

At startup the user picks from a list of free OpenRouter models. The class itself doesn't care which model — you pass any string and it works.

### Token-based compaction (bonus)

The spec suggested a rolling buffer of N turns. I went with a token budget instead because the real constraint is the context window, not turn count. One long message costs more than several short ones. I used `tiktoken` to estimate the prompt size before every call — if it crosses `compact_token_limit`, the agent automatically summarises the older turns into a short bullet-point note and keeps the last 2 turns verbatim. You can also trigger this manually with `/compact`.

Tried sending just the raw transcript as a user message at first — the model rambled and added commentary. Fixed it by giving the summariser an explicit system prompt: "summarise into bullet points, preserve numbers, invent nothing."

### Pinned user facts (bonus)

After every user message, a cheap side-call extracts any personal facts (name, studies, tools, preferences) and stores them in `self.user_facts`. This gets injected into the system message on every call — completely outside the compactable history. So even after compaction wipes old turns, the model still knows who you are. This is how production agents separate long-term memory from the disposable conversation buffer.

### Streaming (bonus)

Used `stream=True` and printed each token as it arrived. Also passed `stream_options={"include_usage": True}` so token counts come in on the final chunk — without this, usage data is lost in streaming mode.

### What didn't work

- First compaction prompt was too vague — the model summarised poorly. Adding a strict instruction fixed it.
- `tiktoken`'s encoder doesn't perfectly match every model's tokenizer, so estimates are off by ~10%. Fine for triggering compaction, not for billing.
- After `/reset`, the first message sometimes returned `None` from the fact-extractor side-call, causing a crash. Fixed by null-checking before calling `.strip()`.

---

## Commands

| Command | What it does |
|---|---|
| `/memory` | Show personal facts pinned from conversation |
| `/tokens` | Show token usage — last call + session total |
| `/history` | Show raw message buffer |
| `/compact` | Manually summarise old history |
| `/reset` | Wipe history (demonstrates statelessness) |
| `/help` | List all commands |
| `exit` | Quit |

---

## Checklist

- [x] Coherent multi-turn conversation, uses earlier context
- [x] Model selection before starting
- [x] `exit`/`quit` to end loop
- [x] `ChatAgent` class with config options (model, token limit, system prompt, streaming, auto-compact)
- [x] API key from environment only — never in source
- [x] `.env` in `.gitignore`
- [x] `.env.example` committed
- [x] **Bonus:** Streaming — tokens print live
- [x] **Bonus:** Auto-compaction on token budget overflow
- [x] **Bonus:** Manual `/compact` command
- [x] **Bonus:** Pinned user facts that survive compaction

---

## How to Run

**1. Install dependencies**
```powershell
pip install openai python-dotenv tiktoken
```

**2. Get an API key**

Sign up at https://openrouter.ai → Settings → API Keys → Create key. Free, no credit card.

**3. Create `.env`**
```powershell
cd "D:\week1-chatbot"
copy .env.example .env
notepad .env
```
Replace the placeholder with your actual key:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**4. Run**
```powershell
python final_chatbot.py
```

**5. Pick a model** — type `0` for auto-routed free model, or any number from the list.

**6. Test it**
```
My name is Alex and I study Computer Science
What is my name?
/memory
/tokens
/compact
What did we talk about earlier?
/reset
What is my name?
exit
```