"""
Build 1 — Custom Tool Parser
=============================
The model emits tool calls as <tool_call> XML. We parse them with regex,
dispatch to real Python functions, and inject results as <tool_response>.
No SDK magic — everything is manual so the mechanics are fully visible.

Tools: read_file(path), write_file(path, content)
"""

import json
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "qwen/qwen3-coder:free"

# ---------------------------------------------------------------------------
# System prompt that teaches the model how to emit tool calls
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful file-management assistant.

When you need to call a tool, emit it like this — nothing else on that line:

<tool_call>{"name": "read_file", "arguments": {"path": "example.txt"}}</tool_call>

Available tools:

read_file(path: str) -> str
  Read the contents of a file. Returns the text content or an error message.

write_file(path: str, content: str) -> str
  Write content to a file. Returns "OK" or an error message.

Rules:
- Only emit one tool call at a time.
- Wait for the <tool_response> before continuing.
- After getting the response, continue reasoning normally.
- Never make up file contents — always read first.
"""

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "OK"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Parse a tool call from the model's raw text
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_call(response_text: str) -> dict | None:
    """
    Extract the first <tool_call>...</tool_call> block and parse the JSON.
    Returns {"name": str, "arguments": dict} or None if no tool call found.
    """
    match = TOOL_CALL_RE.search(response_text)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError as e:
        print(f"[WARN] Could not parse tool call JSON: {e}")
        return None


# ---------------------------------------------------------------------------
# Dispatch a parsed tool call to the right function
# ---------------------------------------------------------------------------
def dispatch(name: str, arguments: dict) -> str:
    """Route tool name → function. Returns result as a string."""
    if name == "read_file":
        return read_file(arguments.get("path", ""))
    elif name == "write_file":
        return write_file(arguments.get("path", ""), arguments.get("content", ""))
    else:
        return f"ERROR: Unknown tool '{name}'"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agent(user_message: str) -> str:
    """
    Full back-and-forth loop.
    1. Send user message.
    2. If model emits a tool call → dispatch, inject result, continue.
    3. If model emits normal text → return it.
    """
    messages = [{"role": "user", "content": user_message}]
    max_iterations = 10  # safety cap

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})

        tool_call = parse_tool_call(assistant_text)

        if tool_call is None:
            # No tool call → final answer
            return assistant_text

        # Execute the tool
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})
        print(f"  [TOOL] {name}({args})")
        result = dispatch(name, args)
        print(f"  [RESULT] {result[:120]}{'...' if len(result) > 120 else ''}")

        # Inject result back as a user turn (XML wrapper so model recognises it)
        tool_response = f"<tool_response>{result}</tool_response>"
        messages.append({"role": "user", "content": tool_response})

    return "[ERROR] Max iterations reached without a final answer."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Build 1 — Custom Tool Parser")
    print("=" * 40)

    tests = [
        "Read the file 'test_input.txt' and tell me what's in it.",
        "Write a haiku about recursion to 'haiku.txt', then read it back and confirm it's correct.",
    ]

    # Create a sample file for the read test
    write_file("test_input.txt", "Hello from Build 1!\nThis file was pre-created for testing.\n")

    for msg in tests:
        print(f"\n[USER] {msg}")
        print("-" * 40)
        reply = run_agent(msg)
        # Strip any tool_call tags from the final printed reply
        clean = TOOL_CALL_RE.sub("", reply).strip()
        print(f"[AGENT] {clean}")
        print()
