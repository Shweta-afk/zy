"""LLM client. Provider is chosen with environment variables (see .env.example).

  LLM_PROVIDER=openai     any OpenAI-compatible endpoint (OpenAI, LiteLLM gateway, vLLM...)
                          OPENAI_BASE_URL, OPENAI_API_KEY, optional LLM_EXTRA_HEADERS (JSON)
  LLM_PROVIDER=anthropic  ANTHROPIC_API_KEY
  LLM_PROVIDER=ollama     local, no key

`chat_tools` is what the agent uses. With TOOL_MODE=native (default for openai) the model
uses native function calling. With TOOL_MODE=json (default for ollama/anthropic here) the model
replies with a JSON object {"tool": ..., "args": {...}} that we parse - this works with any model.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# minimal .env loader (utf-8-sig so a Windows Notepad BOM does not break the first key)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8-sig").splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


class LLMHTTPError(RuntimeError):
    def __init__(self, code: int, message: str, body: str):
        super().__init__(message)
        self.code, self.body = code, body


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", {
            "ollama": "llama3.1:8b", "anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini"}.get(self.provider, ""))
        self.tool_mode = os.getenv("TOOL_MODE", "native" if self.provider == "openai" else "json").lower()
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    # ------------------------------------------------------------------ OpenAI-compatible HTTP
    # Plain urllib instead of the openai SDK: no compiled dependencies (works where security
    # policy blocks DLLs), and urllib uses the Windows system proxy settings automatically.
    def _post_openai(self, payload: dict) -> dict:
        base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        # Explicit User-Agent: Cloudflare-fronted APIs (e.g. Groq) block urllib's default one (error 1010).
        headers = {"Content-Type": "application/json", "User-Agent": "tam-agent/1.0",
                   "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}
        headers.update(json.loads(os.getenv("LLM_EXTRA_HEADERS", "{}") or "{}"))
        req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        timeout = float(os.getenv("LLM_TIMEOUT", "90"))
        max_attempts = int(os.getenv("LLM_MAX_RETRIES", "8"))
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.loads(r.read())
                u = data.get("usage") or {}
                self.tokens_in += u.get("prompt_tokens", 0) or 0
                self.tokens_out += u.get("completion_tokens", 0) or 0
                return data
            except urllib.error.HTTPError as e:
                full_body = e.read().decode(errors="replace")
                body = full_body[:600]
                last = attempt == max_attempts - 1
                if e.code == 429 and not last:
                    # Free tiers limit tokens per minute. Wait as long as the provider asks.
                    wait = _retry_after(e, body)
                    print(f"    rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(wait)
                    continue
                if e.code in (500, 502, 503, 504) and attempt < 2:
                    time.sleep(5)
                    continue
                raise LLMHTTPError(e.code, f"LLM HTTP {e.code} from {base}: {body}", full_body) from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 1:
                    continue
                raise RuntimeError(f"Cannot reach {base}: {getattr(e, 'reason', e)}") from None
        raise RuntimeError("LLM retries exhausted")

    def list_models(self) -> list[str]:
        base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        headers = {"User-Agent": "tam-agent/1.0", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}
        headers.update(json.loads(os.getenv("LLM_EXTRA_HEADERS", "{}") or "{}"))
        with urllib.request.urlopen(urllib.request.Request(f"{base}/models", headers=headers), timeout=30) as r:
            return [m["id"] for m in json.loads(r.read()).get("data", [])]

    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1500,
             temperature: float = 0.0) -> str:
        """Plain text completion. messages: [{"role": "user"|"assistant", "content": str}]."""
        self.calls += 1
        if self.provider == "openai":
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            data = self._post_openai({"model": self.model, "messages": msgs, "max_tokens": max_tokens,
                                      "temperature": temperature})
            return data["choices"][0]["message"].get("content") or ""
        if self.provider == "anthropic":
            import anthropic
            resp = anthropic.Anthropic().messages.create(model=self.model, max_tokens=max_tokens,
                                                         temperature=temperature, system=system or "",
                                                         messages=messages)
            self.tokens_in += resp.usage.input_tokens
            self.tokens_out += resp.usage.output_tokens
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if self.provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            body = json.dumps({"model": self.model, "messages": msgs, "stream": False,
                               "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
            req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"]
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    # ------------------------------------------------------------------ tool calling
    def chat_tools(self, messages: list[dict], system: str, tools: list[dict], force_tool: str | None = None) -> dict:
        """One agent step. Returns {"text": str, "tool_calls": [{"id", "name", "args"}], "raw": assistant_msg}.

        `messages` is kept in OpenAI chat format by the agent; json mode flattens it to plain text.
        """
        if self.provider == "openai" and self.tool_mode == "native":
            self.calls += 1
            choice = {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"
            try:
                data = self._post_openai({
                    "model": self.model, "temperature": 0.0, "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "3000")),
                    "messages": [{"role": "system", "content": system}] +
                                [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages],
                    "tools": [{"type": "function", "function": t} for t in tools], "tool_choice": choice,
                    **({"reasoning_effort": os.getenv("LLM_REASONING_EFFORT")}
                       if os.getenv("LLM_REASONING_EFFORT") else {})})
            except LLMHTTPError as e:
                salvaged = _salvage_tool_call(e, {t["name"] for t in tools}, self.calls)
                if not salvaged:
                    raise
                return salvaged
            m = data["choices"][0]["message"]
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_unparsed": fn.get("arguments")}
                calls.append({"id": tc.get("id"), "name": fn.get("name"), "args": args})
            raw = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                raw["tool_calls"] = [{"id": tc.get("id"), "type": "function",
                                      "function": {"name": tc["function"]["name"],
                                                   "arguments": tc["function"].get("arguments") or "{}"}}
                                     for tc in m["tool_calls"]]
            return {"text": m.get("content") or "", "tool_calls": calls, "raw": raw}

        # JSON protocol: works with any chat model
        tool_doc = "\n".join(f"- {t['name']}({', '.join(t['parameters'].get('properties', {}))}): {t['description']}"
                             for t in tools)
        sys_json = (system + "\n\nTOOLS:\n" + tool_doc +
                    '\n\nReply with ONLY one JSON object: {"thought": "<short reason>", "tool": "<tool name>", '
                    '"args": {...}}. No other text.')
        flat = []
        for m in messages:
            if m["role"] == "tool":
                flat.append({"role": "user", "content": f"TOOL RESULT ({m.get('_name', '')}):\n{m['content']}"})
            elif m["role"] == "assistant":
                flat.append({"role": "assistant", "content": m.get("content") or json.dumps(m.get("_json", {}))})
            else:
                flat.append({"role": "user", "content": m["content"]})
        # merge consecutive same-role messages (Anthropic requires alternation)
        merged = []
        for m in flat:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n\n" + m["content"]
            else:
                merged.append(dict(m))
        text = self.chat(merged, system=sys_json, max_tokens=2000)
        obj = _extract_json(text)
        calls = []
        if obj and obj.get("tool"):
            calls.append({"id": f"j{self.calls}", "name": obj["tool"], "args": obj.get("args") or {}})
        return {"text": (obj or {}).get("thought", text if not obj else ""), "tool_calls": calls,
                "raw": {"role": "assistant", "content": text}}


def _salvage_tool_call(e: "LLMHTTPError", tool_names: set, n: int) -> dict | None:
    """Groq returns HTTP 400 'tool_use_failed' when a tool call does not match the schema exactly
    (e.g. null where a string was declared), but includes the model's attempt as failed_generation.
    Recover that call instead of losing the whole step; our own checker validates it afterwards."""
    if e.code != 400 or "tool_use_failed" not in e.body:
        return None
    try:
        gen = json.loads(e.body).get("error", {}).get("failed_generation")
    except json.JSONDecodeError:
        return None
    obj = gen if isinstance(gen, dict) else _extract_json(gen or "")
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    args = obj.get("arguments", obj.get("parameters"))
    if args is None and "function" in obj:
        args = obj["function"].get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    if not name:
        mm = re.search(r"tool (\w+)", json.loads(e.body).get("error", {}).get("message", ""))
        name, args = (mm.group(1) if mm else None), (args if args is not None else obj)
    if name not in tool_names or not isinstance(args, dict):
        return None
    cid = f"salvaged-{n}"
    raw = {"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function",
           "function": {"name": name, "arguments": json.dumps(args)}}]}
    return {"text": "(recovered a tool call the provider rejected on strict schema check)",
            "tool_calls": [{"id": cid, "name": name, "args": args}], "raw": raw}


def _retry_after(err, body: str) -> float:
    """Seconds to wait after HTTP 429: Retry-After header, else 'try again in 7.5s' in the body, else 15s."""
    h = err.headers.get("retry-after") if getattr(err, "headers", None) else None
    try:
        if h:
            return min(float(h) + 1, 90)
    except ValueError:
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", body)
    if m:
        return min(int(m.group(1) or 0) * 60 + float(m.group(2)) + 1, 90)
    return 15.0


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
