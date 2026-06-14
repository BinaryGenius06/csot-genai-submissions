"""
Build 2 — SDK Native Tool Calling
===================================
Same concept as Build 1 but using the OpenAI SDK's tools= parameter.
Compare: no XML, no regex, structured function.arguments JSON string.

Tools: get_weather(city, unit), calculate(expression)
"""

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "qwen/qwen3-coder:free"

# ---------------------------------------------------------------------------
# Tool schemas — what the SDK sends to the model
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'London' or 'New Delhi'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Defaults to celsius.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a safe mathematical expression. "
                "Supports +, -, *, /, **, sqrt, abs, round. "
                "Example: '2 ** 10 + sqrt(144)'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def get_weather(city: str, unit: str = "celsius") -> str:
    """Simulated weather — swap in a real API if desired."""
    fake_data = {
        "london":    {"celsius": 14, "fahrenheit": 57,  "condition": "Cloudy"},
        "new delhi": {"celsius": 38, "fahrenheit": 100, "condition": "Sunny"},
        "new york":  {"celsius": 22, "fahrenheit": 72,  "condition": "Partly cloudy"},
        "tokyo":     {"celsius": 26, "fahrenheit": 79,  "condition": "Clear"},
    }
    key = city.lower()
    data = fake_data.get(key, {"celsius": 20, "fahrenheit": 68, "condition": "Unknown"})
    temp = data[unit.lower()] if unit.lower() in data else data["celsius"]
    symbol = "°C" if unit.lower() == "celsius" else "°F"
    return f"{city}: {temp}{symbol}, {data['condition']}"


def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    import math
    allowed = {
        "sqrt": math.sqrt, "abs": abs, "round": round,
        "pow": pow, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Dispatch an SDK tool_call object
# ---------------------------------------------------------------------------
def dispatch(tool_call) -> str:
    """Parse tool_call.function.arguments (JSON string) and call the right function."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return "ERROR: Could not parse tool arguments"

    if name == "get_weather":
        return get_weather(args.get("city", ""), args.get("unit", "celsius"))
    elif name == "calculate":
        return calculate(args.get("expression", ""))
    else:
        return f"ERROR: Unknown tool '{name}'"


# ---------------------------------------------------------------------------
# Agent loop — canonical SDK pattern
# ---------------------------------------------------------------------------
def run_agent(user_message: str) -> str:
    """
    Canonical loop:
    1. Call API with tools.
    2. If finish_reason == "tool_calls" → dispatch all tool calls, append results.
    3. Repeat until finish_reason == "stop".
    """
    messages = [{"role": "user", "content": user_message}]
    max_iterations = 10

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            # Append the assistant message (with tool_calls field)
            messages.append(choice.message)

            # Dispatch each tool call and append results
            for tc in choice.message.tool_calls:
                print(f"  [TOOL] {tc.function.name}({tc.function.arguments})")
                result = dispatch(tc)
                print(f"  [RESULT] {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # Unexpected finish reason — return whatever we have
            return choice.message.content or "[No content]"

    return "[ERROR] Max iterations reached."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Build 2 — SDK Native Tool Calling")
    print("=" * 40)

    tests = [
        "What's the weather like in Tokyo and New Delhi right now?",
        "What is 2 to the power of 20, plus the square root of 1764?",
        "Compare the temperature in London and New York (both in Fahrenheit), "
        "then tell me which is warmer and by how many degrees.",
    ]

    for msg in tests:
        print(f"\n[USER] {msg}")
        print("-" * 40)
        reply = run_agent(msg)
        print(f"[AGENT] {reply}")
