"""Cursor OAuth login for the Hermes cursor provider plugin.

Standalone -- doesn't touch hermes_cli/auth.py (core, off-limits to
plugins per AGENTS.md). Credentials are cached in this plugin's own
directory, not Hermes' central auth.json.

Cursor's hosted login page (cursor.com/loginDeepControl) offers GitHub
SSO as one of the sign-in options in the browser that opens -- pick
"Continue with GitHub" there. This script only drives the
device-code-style poll loop; the identity provider choice happens in
the browser UI Cursor serves, and Hermes has no way (and no need) to
skip that screen.

Usage:
    python3 oauth_login.py            # opens browser, polls, saves token
    python3 oauth_login.py --status   # print cached credential status
    python3 oauth_login.py --logout   # delete cached credentials
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cursor_backend_oauth import login_cursor, refresh_cursor_token, is_cursor_token_expiring_soon  # noqa: E402

CRED_PATH = Path(__file__).resolve().parent / ".cursor_auth.json"


def _hermes_home() -> Path:
    env = os.getenv("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _cred_path() -> Path:
    # Prefer $HERMES_HOME/plugins/model-providers/cursor/.cursor_auth.json
    # so multiple profiles (each with their own HERMES_HOME) don't share
    # credentials, but fall back to the file next to this script when
    # HERMES_HOME can't be resolved (e.g. run standalone for debugging).
    try:
        p = _hermes_home() / "plugins" / "model-providers" / "cursor" / ".cursor_auth.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return CRED_PATH


def load_cached_credentials() -> dict[str, Any]:
    path = _cred_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    token = str(data.get("access_token", "") or "")
    if token and is_cursor_token_expiring_soon(token):
        refresh_token = str(data.get("refresh_token", "") or "")
        if refresh_token:
            try:
                refreshed = refresh_cursor_token(refresh_token)
                data.update(refreshed)
                _save_credentials(data)
            except Exception:
                pass  # fall through with the possibly-stale token; caller handles 401
    return data


def _save_credentials(creds: dict[str, Any]) -> None:
    path = _cred_path()
    path.write_text(json.dumps(creds, indent=2))
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _ensure_env_placeholder() -> None:
    """Write a stable placeholder CURSOR_ACCESS_TOKEN into $HERMES_HOME/.env.

    Hermes core's ``auth_type="api_key"`` credential check just wants a
    non-empty ``CURSOR_ACCESS_TOKEN`` present so `hermes doctor` /
    `hermes model` treat the provider as configured. The REAL, rotating
    Cursor OAuth token never goes through Hermes' own credential store --
    it stays in this plugin's own cache file (0600, plugin-dir-local) and
    is refreshed/read directly by proxy_server.py. This placeholder is
    deliberately not a secret; it's just a presence marker.
    """
    env_path = _hermes_home() / ".env"
    placeholder = "cursor-oauth-managed-by-plugin"
    try:
        existing = env_path.read_text() if env_path.exists() else ""
    except Exception:
        existing = ""
    if "CURSOR_ACCESS_TOKEN=" in existing:
        return
    with open(env_path, "a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(f"CURSOR_ACCESS_TOKEN={placeholder}\n")
    try:
        os.chmod(env_path, 0o600)
    except Exception:
        pass


def do_login() -> None:
    print("Opening Cursor sign-in in your browser...")
    print("On the Cursor login page, choose \"Continue with GitHub\" for GitHub SSO.")

    def on_auth_url(url: str) -> None:
        print(f"\nIf the browser didn't open automatically, visit:\n  {url}\n")

    def on_poll_start() -> None:
        print("Waiting for you to complete sign-in in the browser...")

    creds = login_cursor(on_auth_url=on_auth_url, on_poll_start=on_poll_start)
    _save_credentials(creds)
    _ensure_env_placeholder()
    print("Cursor sign-in complete. Credentials cached at:")
    print(f"  {_cred_path()}")
    print("\nNext: run `hermes model` and pick Cursor, or:")
    print("  hermes -z 'hello' --provider cursor -m composer-2.5")


def do_status() -> None:
    creds = load_cached_credentials()
    if not creds.get("access_token"):
        print("No cached Cursor credentials. Run: python3 oauth_login.py")
        return
    expiring = is_cursor_token_expiring_soon(str(creds["access_token"]))
    print(f"Credential file: {_cred_path()}")
    print(f"Access token present: yes")
    print(f"Expiring soon / needs refresh: {expiring}")


def do_logout() -> None:
    path = _cred_path()
    if path.exists():
        path.unlink()
        print(f"Removed cached credentials: {path}")
    else:
        print("No cached credentials to remove.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--logout", action="store_true")
    args = parser.parse_args()

    if args.status:
        do_status()
    elif args.logout:
        do_logout()
    else:
        do_login()


if __name__ == "__main__":
    main()
