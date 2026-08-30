"""Cursor model provider plugin for Hermes Agent.

Lets you route Hermes conversations through your Cursor subscription
(composer, GPT/Claude-family models re-sold via Cursor) using Cursor's
native Agent API.

## Why a local proxy, not a core transport

Hermes core supports four wire protocols (chat_completions,
codex_responses, anthropic_messages, bedrock_converse) registered in
``agent/transports/``. There is currently no plugin hook to register a
fifth wire protocol or to override ``create_openai_client()``'s
provider dispatch (those are core files -- editing them is exactly what
PR #40876 (github.com/NousResearch/hermes-agent/pull/40876) did, and it
was closed under the standing "vendor providers ship as plugins, not
core patches" policy, AGENTS.md ~line 797).

So this plugin runs a tiny local FastAPI proxy (``proxy_server.py``)
that speaks plain OpenAI Chat Completions on localhost and translates
requests to Cursor's real HTTP/2 "Connect" streaming protobuf API
underneath (``cursor_backend/`` -- salvaged from the closed PR, which is
genuinely solid engineering; only the *distribution* mechanism was
rejected, not the code). Hermes then just sees provider=cursor,
api_mode=chat_completions, base_url=http://127.0.0.1:<port>/v1 -- a
completely standard integration requiring zero repo edits.

## Setup

    cd ~/.hermes/plugins/model-providers/cursor
    python3 oauth_login.py          # opens Cursor sign-in; pick "Continue
                                     # with GitHub" for GitHub SSO
    hermes model                    # pick "Cursor" -> starts the proxy
                                     # automatically and lists your models

## Files

- ``__init__.py``           -- this file; registers the ProviderProfile
- ``proxy_server.py``       -- local OpenAI-compatible shim (FastAPI/uvicorn)
- ``oauth_login.py``        -- CLI: `python3 oauth_login.py [--status|--logout]`
- ``cursor_backend_oauth.py`` -- OAuth poll/refresh (from the PR, unmodified logic)
- ``cursor_backend/``       -- HTTP/2 Connect client, protobuf schema, model
                                discovery (from the PR, unmodified logic)
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = int(os.getenv("CURSOR_PROXY_PORT", "8765"))
LOCK_PATH = PLUGIN_DIR / ".proxy_start.lock"

# Both urllib (catalog fetch) and httpx (chat completions) ignore CIDR
# entries in NO_PROXY (e.g. "127.0.0.0/8"), so a request to the localhost
# proxy gets routed through the system HTTP proxy (HTTP_PROXY) and fails
# with 503. Localhost must always be direct -- ensure explicit 127.0.0.1 /
# ::1 entries exist in NO_PROXY. Both casings are updated because
# getproxies() lowercases env keys and the last one in iteration wins.
_NO_PROXY_LOCAL = ("127.0.0.1", "::1")
for _np_key in ("NO_PROXY", "no_proxy"):
    _np_parts = [p.strip() for p in os.environ.get(_np_key, "").split(",") if p.strip()]
    if not all(x in _np_parts for x in _NO_PROXY_LOCAL):
        os.environ[_np_key] = ",".join(
            _np_parts + [x for x in _NO_PROXY_LOCAL if x not in _np_parts]
        )
del _np_key, _np_parts, _NO_PROXY_LOCAL

_proxy_process: subprocess.Popen | None = None


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_proxy_running(port: int = DEFAULT_PORT) -> None:
    """Start the local proxy server as a background subprocess if not already up.

    Idempotent -- safe to call on every request, and safe across
    concurrent processes. Every fresh Hermes invocation (CLI query,
    `hermes doctor`'s parallel connectivity probes, etc.) imports this
    plugin and calls this function; without serialization, N processes
    can all observe "port closed" at once and race to spawn N duplicate
    uvicorn instances, most of which crash with EADDRINUSE -- the
    resulting spawn/crash churn is what produced the "Cursor (timed
    out)" flakiness seen in `hermes doctor`. A cross-process flock on
    ``.proxy_start.lock`` makes the whole check-then-spawn sequence
    atomic across the whole machine, not just this one Python process.
    """
    global _proxy_process
    if _port_open("127.0.0.1", port):
        return

    lock_fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Re-check now that we hold the exclusive lock -- another process
        # may have already started the server while we were waiting.
        if _port_open("127.0.0.1", port):
            return

        python_exe = sys.executable
        log_path = PLUGIN_DIR / "proxy_server.log"
        try:
            log_file = open(log_path, "a")
            _proxy_process = subprocess.Popen(
                [python_exe, str(PLUGIN_DIR / "proxy_server.py"), "--port", str(port)],
                cwd=str(PLUGIN_DIR),
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
            atexit.register(_stop_proxy)
        except Exception as exc:
            logger.warning("Failed to start Cursor proxy server: %s", exc)
            return

        # Wait briefly for the port to open so the very first request
        # doesn't race the server's own startup.
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if _port_open("127.0.0.1", port):
                return
            time.sleep(0.2)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _stop_proxy() -> None:
    global _proxy_process
    if _proxy_process is not None and _proxy_process.poll() is None:
        try:
            _proxy_process.terminate()
        except Exception:
            pass


class CursorProfile(ProviderProfile):
    """Cursor -- routes through a local proxy in front of Cursor's Agent API.

    See module docstring for why this is a proxy rather than a native
    ``api_mode``.
    """

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> list[str] | None:
        _ensure_proxy_running(DEFAULT_PORT)
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


cursor = CursorProfile(
    name="cursor",
    aliases=("cursor-agent", "cursor-subscription"),
    display_name="Cursor",
    description="Cursor subscription via Cursor's native Agent API (local proxy plugin)",
    signup_url="https://cursor.com",
    env_vars=("CURSOR_ACCESS_TOKEN", "CURSOR_PROXY_BASE_URL"),
    base_url=f"http://127.0.0.1:{DEFAULT_PORT}/v1",
    api_mode="chat_completions",
    auth_type="api_key",  # the proxy handles real Cursor OAuth internally;
                           # Hermes only needs *a* non-empty key to route here
    default_aux_model="composer-2.5",
    fallback_models=(
        "composer-2.5",
        "composer-2.5-thinking",
    ),
)

# Kick the proxy off eagerly at import time too, so `hermes doctor`'s
# /models probe (which calls fetch_models via the profile) succeeds even
# on the very first call in a fresh process.
try:
    _ensure_proxy_running(DEFAULT_PORT)
except Exception:
    pass

register_provider(cursor)
