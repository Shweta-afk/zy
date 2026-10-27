"""Check the LLM endpoint and tool calling before a full run.   python scripts/check_llm.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tam_agent.llm import LLMClient  # noqa: E402

c = LLMClient()
print(f"provider={c.provider} model={c.model} tool_mode={c.tool_mode}")
try:
    if c.provider == "openai":
        print("models on this endpoint:", c.list_models()[:20])
except Exception as e:
    print("(could not list models:", e, ")")
print("plain chat:", c.chat([{"role": "user", "content": "Reply with the word OK."}], max_tokens=10))
tool = {"name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
r = c.chat_tools([{"role": "user", "content": "What's the weather in Mumbai? Use the tool."}], "You use tools.", [tool])
print("tool calls:", r["tool_calls"] or "NONE")
if not r["tool_calls"]:
    print("-> Native tool calling did not work. Add TOOL_MODE=json to .env and run this again.")
else:
    print("-> Tool calling works. Next: python run.py --ids TKT-144939")
