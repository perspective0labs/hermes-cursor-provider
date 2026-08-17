"""Local OpenAI-compatible proxy in front of Cursor's native Agent API.

Hermes core only understands four wire protocols (chat_completions,
codex_responses, anthropic_messages, bedrock_converse) and has no plugin
hook to register a fifth. Cursor's real transport is a custom HTTP/2
"Connect" streaming protocol over protobuf (see cursor_backend/), so this
process translates: it exposes a plain `POST /v1/chat/completions`
(OpenAI Chat Completions shape) on localhost, and internally drives the
real Cursor Agent API over the salvaged HTTP/2 client.

This means the `cursor` ProviderProfile can declare `api_mode:
"chat_completions"` and `base_url: "http://127.0.0.1:<port>/v1"` --
zero core edits, exactly per AGENTS.md policy (`plugins/model-providers/`
is user code; nothing here touches the Hermes repo).

Run standalone for debugging:
    python3 proxy_server.py --port 8765

Auto-started by __init__.py on first use of the `cursor` provider.
"""

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported at module level (not inside create_app()) so FastAPI/pydantic can
# resolve the `Request` type hint on chat_completions() when building the
# OpenAPI schema. Without `from __future__ import annotations` in this file,
# annotations stay as real objects rather than deferred strings, so this
# would work either way -- but keeping the import at module scope is the
# robust fix regardless.
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

logger = logging.getLogger("cursor_plugin.proxy")

DEFAULT_PORT = int(os.getenv("CURSOR_PROXY_PORT", "8765"))
DEFAULT_HOST = "127.0.0.1"


def _load_credentials() -> dict[str, Any]:
    """Read cached Cursor OAuth credentials from the plugin's own store.

    Separate from Hermes' hermes_cli/auth.py auth.json (core file, not
    reachable from a plugin without importing private internals) -- this
    plugin keeps its own small credential file at
    ``$HERMES_HOME/plugins/model-providers/cursor/.cursor_auth.json``.
    """
    from oauth_login import load_cached_credentials

    return load_cached_credentials()


def create_app():
    from cursor_backend.stream_client import run_cursor_agent_turn
    from cursor_backend.model_discovery import fetch_cursor_usable_models
    from cursor_backend.constants import CURSOR_API_URL

    app = FastAPI(title="cursor-plugin-proxy")

    _PLACEHOLDER_TOKEN = "cursor-oauth-managed-by-plugin"

    def _resolve_token() -> str:
        # CURSOR_ACCESS_TOKEN in the environment is honored as a real
        # override IF the user set it themselves. But oauth_login.py also
        # writes a placeholder of this same var into $HERMES_HOME/.env
        # purely so Hermes core's auth_type="api_key" check sees a
        # non-empty value -- that placeholder must never be treated as a
        # real token here, or it would shadow the actual cached OAuth
        # token below.
        env_token = os.getenv("CURSOR_ACCESS_TOKEN", "").strip()
        if env_token and env_token != _PLACEHOLDER_TOKEN:
            return env_token
        creds = _load_credentials()
        token = str(creds.get("access_token", "") or "").strip()
        if not token:
            raise RuntimeError(
                "No Cursor credentials found. Run: python3 oauth_login.py"
            )
        return token

    @app.get("/v1/models")
    def list_models():
        try:
            token = _resolve_token()
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        models = fetch_cursor_usable_models(api_key=token, base_url=CURSOR_API_URL) or [
            "composer-2.5"
        ]
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "owned_by": "cursor"} for m in models
            ],
        }

    def _messages_to_cursor(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
        """Split OpenAI-style messages into (system_prompt, chat messages)."""
        system_parts: list[str] = []
        chat: list[dict] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, list):
                # content-parts array -> flatten to text
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
                content = text
            if role == "system":
                system_parts.append(content or "")
            else:
                chat.append({"role": role, "content": content or "", **{
                    k: v for k, v in m.items() if k not in ("role", "content")
                }})
        system_prompt = "\n\n".join(p for p in system_parts if p) or None
        return system_prompt, chat

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            token = _resolve_token()
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "auth_error"}},
                status_code=401,
            )

        body = await request.json()
        model = body.get("model", "composer-2.5")
        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = bool(body.get("stream", False))
        system_prompt, chat_messages = _messages_to_cursor(messages)
        conversation_id = body.get("cursor_conversation_id") or str(uuid.uuid4())

        try:
            response = run_cursor_agent_turn(
                api_key=token,
                model_id=model,
                messages=chat_messages,
                system_prompt=system_prompt,
                tools=tools,
                conversation_id=conversation_id,
                base_url=CURSOR_API_URL,
            )
        except Exception as exc:
            logger.exception("Cursor turn failed")
            return JSONResponse(
                {"error": {"message": str(exc), "type": "cursor_upstream_error"}},
                status_code=502,
            )

        choice = response.choices[0]
        message = choice.message
        tool_calls = None
        if getattr(message, "tool_calls", None):
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": getattr(response, "model", model) or model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message.content,
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    },
                    "finish_reason": choice.finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            },
        }

        if not stream:
            return JSONResponse(payload)

        # Minimal single-chunk SSE stream (Cursor's turn already ran to
        # completion above; we don't yet forward true token deltas through
        # the proxy hop -- see README "Known limitations").
        def _sse():
            chunk = {
                "id": payload["id"],
                "object": "chat.completion.chunk",
                "created": payload["created"],
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": message.content,
                            **({"tool_calls": tool_calls} if tool_calls else {}),
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            done_chunk = {
                "id": payload["id"],
                "object": "chat.completion.chunk",
                "created": payload["created"],
                "model": payload["model"],
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": choice.finish_reason or "stop"}
                ],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
