"""
Promptfoo HTTP provider for the /chat endpoint.

This provider formats chat responses in human-readable format instead of JSON.

Environment variables supported:
- PROMPTFOO_RAG_BASE_URL (default http://127.0.0.1:8000)
- PROMPTFOO_RAG_JWT (optional bearer token)
- PROMPTFOO_RAG_FILE_ID (default testid1)
- PROMPTFOO_RAG_ENTITY_ID (default promptfoo-tester)
- PROMPTFOO_RAG_K (default 4)
- PROMPTFOO_RAG_TIMEOUT (default 30)
"""

from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

# -------------------------------------------
# Defaults
# -------------------------------------------

DEFAULT_BASE_URL = os.getenv("PROMPTFOO_RAG_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_FILE_ID = os.getenv("PROMPTFOO_RAG_FILE_ID", "testid1")
DEFAULT_ENTITY_ID = os.getenv("PROMPTFOO_RAG_ENTITY_ID", "promptfoo-tester")
DEFAULT_K = int(os.getenv("PROMPTFOO_RAG_K", "4"))
DEFAULT_TIMEOUT = float(os.getenv("PROMPTFOO_RAG_TIMEOUT", "30"))
JWT_TOKEN = os.getenv("PROMPTFOO_RAG_JWT")


# -------------------------------------------
# Build Payload
# -------------------------------------------

def _build_payload(prompt: str, config: Dict[str, Any], context: Dict[str, Any]):
    vars_ctx = (context or {}).get("vars", {})

    payload = {
        "query": prompt,
        "file_id": vars_ctx.get("file_id")
        or config.get("defaultFileId")
        or DEFAULT_FILE_ID,
        "entity_id": vars_ctx.get("entity_id")
        or config.get("defaultEntityId")
        or DEFAULT_ENTITY_ID,
        "k": vars_ctx.get("k") or config.get("defaultK") or DEFAULT_K,
        "model": vars_ctx.get("model") or config.get("defaultModel") or "gemini",
        "temperature": vars_ctx.get("temperature") or config.get("defaultTemperature") or 0.7,
    }

    # Extra user config
    body_extras = config.get("bodyExtras")
    if isinstance(body_extras, dict):
        payload.update(body_extras)

    return payload


# -------------------------------------------
# SAFE SCRUBBER (removes sensitive data!)
# -------------------------------------------

def _scrub_raw(parsed):
    """
    Remove document content from sources before sending to Promptfoo UI.
    Prevents leaking sensitive information.
    """
    try:
        # Chat returns: { answer: "...", sources: [...], model_used: "..." }
        if isinstance(parsed, dict) and "sources" in parsed:
            for source in parsed["sources"]:
                if isinstance(source, dict) and "content" in source:
                    source["content"] = "(hidden)"

        return parsed

    except Exception:
        return parsed


# -------------------------------------------
# HUMAN-READABLE OUTPUT FOR PROMPTFOO UI
# -------------------------------------------

def _format_output(parsed: Any) -> str:
    """
    Format chat response - show only the answer text.
    """

    if parsed is None:
        return "(empty response)"

    if isinstance(parsed, str):
        return parsed

    if isinstance(parsed, dict):
        # Extract and return only the answer text
        return parsed.get("answer", "(no answer)")

    # Fallback to JSON for unexpected formats
    return json.dumps(parsed, indent=2)


# -------------------------------------------
# Call API
# -------------------------------------------

def call_api(prompt: str, options: Dict[str, Any] | None = None,
             context: Dict[str, Any] | None = None):

    options = options or {}
    config = options.get("config", {})

    base_url = config.get("baseUrl", DEFAULT_BASE_URL).rstrip("/")
    endpoint = config.get("endpoint", "/chat")
    method = config.get("method", "POST").upper()
    url = f"{base_url}{endpoint}"

    payload = _build_payload(prompt, config, context or {})
    data = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if config.get("includeAuth", True) and JWT_TOKEN:
        headers["Authorization"] = f"Bearer {JWT_TOKEN}"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
            raw_body = response.read().decode("utf-8")

            try:
                parsed_json = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed_json = raw_body

            # SCRUB SENSITIVE FIELDS
            scrubbed = _scrub_raw(parsed_json)

            # PRETTY UI OUTPUT
            formatted_output = _format_output(scrubbed)

            return {
                "output": formatted_output,  # Shown in Promptfoo UI
                "raw": scrubbed              # Used for assertions
            }

    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="ignore")
        return {"output": f"HTTP {err.code}: {body}", "raw": None}

    except Exception as exc:
        return {"output": f"Error: {str(exc)}", "raw": None}
