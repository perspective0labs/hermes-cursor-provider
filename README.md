# Cursor Provider Plugin for Hermes Agent

Routes [Hermes Agent](https://github.com/NousResearch/hermes-agent)
conversations through your **Cursor subscription** using Cursor's
native Agent API (the same backend the `cursor-agent` CLI and Cursor
IDE use) — Composer 2.5, Claude Opus/Sonnet, GPT-5.x, Gemini, Grok,
GLM, Kimi, and whatever else your plan exposes.

## Install

Clone directly into your Hermes plugins directory:

```bash
git clone https://github.com/perspective0labs/hermes-cursor-provider.git \
  "${HERMES_HOME:-$HOME/.hermes}/plugins/model-providers/cursor"
```

Then install the plugin's own runtime deps into the same Python
environment Hermes runs under (fastapi/uvicorn/httpx ship with Hermes
already on most installs; `h2` and `protobuf` are the two you'll likely
need to add):

```bash
# Activate whichever venv `hermes` actually runs from, e.g.:
source ~/hermes-agent/venv/bin/activate   # adjust to your install
pip install h2 protobuf
```

Verify Hermes picked it up:

```bash
python3 -c "from providers import get_provider_profile; print(get_provider_profile('cursor'))"
```


## Why this exists

[PR #40876](https://github.com/NousResearch/hermes-agent/pull/40876)
added a full Cursor integration (HTTP/2 "Connect" streaming transport,
OAuth, protobuf schema, ~9,400 lines) directly into Hermes core. It was
closed under the project's standing policy: vendor providers ship as
**plugins** under `~/.hermes/plugins/model-providers/`, not core edits
(see `AGENTS.md` in the hermes-agent repo, "in-tree-provider-integration"
policy). The PR's engineering was solid — this plugin salvages that code
and repackages it as a proper standalone plugin, with zero changes to
the Hermes repo.

## How it works

Hermes core only speaks four wire protocols natively (chat_completions,
codex_responses, anthropic_messages, bedrock_converse) and has no plugin
hook to register a fifth. Cursor's real API is a custom HTTP/2 "Connect"
protobuf streaming protocol, not any of those.

So this plugin runs a small local FastAPI proxy
(`proxy_server.py`, auto-started on 127.0.0.1:8765) that:

1. Exposes plain OpenAI Chat Completions (`POST /v1/chat/completions`,
   `GET /v1/models`) — a completely standard `api_mode: chat_completions`
   endpoint from Hermes' point of view.
2. Internally translates each request into a real Cursor Agent API turn
   over HTTP/2 Connect (`cursor_backend/`, salvaged unmodified from the
   PR) and translates the response back to OpenAI shape.

Result: `provider: cursor` in Hermes config is indistinguishable from
any other `chat_completions` provider. Zero core file edits.

## First-time login

```bash
cd ~/.hermes/plugins/model-providers/cursor
python3 oauth_login.py
```

This opens `cursor.com/loginDeepControl` in your browser. **Choose
"Continue with GitHub"** on the login page for GitHub SSO (this is
Cursor's own login screen — Hermes has no separate code path for it,
the option is just there). Credentials are cached to
`.cursor_auth.json` (0600) in this plugin directory, or under
`$HERMES_HOME/plugins/model-providers/cursor/.cursor_auth.json` if
`HERMES_HOME` is set (keeps multiple profiles isolated). Tokens
auto-refresh on read when close to expiry.

Then:

```bash
hermes model                # pick "Cursor" from the picker
# or
hermes chat -q "hello" --provider cursor -m composer-2.5
```

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Registers the `ProviderProfile`; auto-starts the proxy on import |
| `proxy_server.py` | Local OpenAI-compatible shim (FastAPI + uvicorn) |
| `oauth_login.py` | CLI: `python3 oauth_login.py [--status\|--logout]` |
| `cursor_backend_oauth.py` | OAuth login/poll/refresh (from the PR, logic unmodified) |
| `cursor_backend/` | HTTP/2 Connect client, protobuf schema, exec handlers, model discovery (from the PR, logic unmodified except import paths rewritten from `agent.cursor.*` → `cursor_backend.*`) |

## Known limitations

- **Session-title auxiliary call sometimes fails** (`⚠ Auxiliary title
  generation failed: Connection error`) — cosmetic only, the main turn
  still completes correctly and the session just falls back to using
  the query text as its title. Root cause not yet isolated (didn't show
  up in the proxy's own log, so likely a timing/connect-timeout issue
  on Hermes' aux-client side rather than a proxy bug).
- **Streaming is not true token-by-token** — the proxy runs the whole
  Cursor turn to completion first, then emits it as a single SSE chunk.
  Works fine but you won't see live token streaming in the CLI.
- **Tool/agentic exec calls are stubbed as rejected** by default (the
  `_default_exec_handler` in `cursor_backend/stream_client.py` rejects
  shell/read/write/etc RPCs Cursor's agent side might request) — Hermes
  drives its own tool loop on top of the chat completion, so this
  should rarely matter, but if Cursor's model tries to use its own
  built-in agentic tools mid-turn, those calls get politely rejected
  rather than executed.
- `hermes doctor`'s `/models` health probe works (confirmed — returns
  the full live Cursor catalog), but the probe implicitly starts the
  proxy subprocess as a side effect of the check itself, which is a
  little unusual for a health check. Harmless.

## Debugging

```bash
# Run the proxy in the foreground to see live logs
cd ~/.hermes/plugins/model-providers/cursor
python3 proxy_server.py --port 8765

# Or check the background proxy's log file
cat proxy_server.log

# Check cached credential status
python3 oauth_login.py --status

# Force re-login
python3 oauth_login.py --logout && python3 oauth_login.py
```
