"""dsh (DeepSeek Harness) client integration for the B70 launcher gateway.

dsh runs via ``npx -y @deepseek-ai/dsh@<version>`` and isolates all of its
state (profiles, sessions, storages) under the directory named by the
``DSH_HOME`` environment variable, so a launcher-owned home keeps the user's
``~/.dsh`` untouched.  Model routing is injected through a ``--patch`` loader
overlay (a YAML array of entry-id-targeted config overrides) that registers an
OpenAI-compatible provider pointing at the launcher gateway
(``http://127.0.0.1:<daemon_port>/gw/<client>/<engine_port>/v1``) and makes it
the default model.

dsh 0.2.0-rc.2 ships no dedicated terminal UI profile; its interactive
surface is the ``web`` profile (a browser UI that dsh opens itself), and
one-shot terminal use goes through the ``headless`` profile:
``dsh headless "task"``.  Both accept the same ``--patch`` overlay.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Pinned resolved version found in the npx cache (prefer over @latest for
# reproducible launches; the cache hash dir below is host-specific and only
# used by dsh_installed()).
DSH_VERSION = "0.2.0-rc.2"
DSH_PACKAGE = "@deepseek-ai/dsh@" + DSH_VERSION

PROVIDER_ID = "b70-gateway"
API_KEY_ENV = "B70_GW_KEY"
API_KEY_VALUE = "b70-gateway-key"

_PATCH_FILENAME = "gateway.yml"


def _overlay_yaml(base_url: str, model: str, ctx: int) -> str:
    """Render the --patch overlay as YAML (JSON-escaped scalars, valid YAML)."""
    return (
        "# B70 launcher gateway overlay: dsh --patch layer (generated)\n"
        "- id: llm-pi-ai\n"
        "  config:\n"
        "    providers:\n"
        f"      {PROVIDER_ID}:\n"
        "        displayName: B70 Gateway\n"
        "        api: openai-completions\n"
        f"        apiKeyEnv: {API_KEY_ENV}\n"
        f"        baseURL: {json.dumps(base_url)}\n"
        "        models:\n"
        f"          - id: {json.dumps(model)}\n"
        f"            name: {json.dumps(model)}\n"
        f"            contextWindow: {int(ctx)}\n"
        "- id: agent-default-model\n"
        "  config:\n"
        f"    provider: {PROVIDER_ID}\n"
        f"    model: {json.dumps(model)}\n"
    )


def dsh_prepare(base_url: str, model: str, ctx: int, state_dir: Path) -> tuple[list[str], dict]:
    """Prepare an isolated dsh launch against the gateway.

    Writes only inside ``state_dir``: a launcher-owned ``DSH_HOME`` (``home/``)
    and the provider overlay (``gateway.yml``).  Idempotent: re-running with
    the same arguments rewrites the same overlay and never touches files
    outside ``state_dir``.

    Returns ``(argv, env_additions)`` for the interactive (web) profile. dsh
    0.2 has no terminal UI: the launcher runs this in a terminal so the server
    stays visible, and dsh opens its own tokenized URL in the default browser.
    For one-shot terminal prompts use :func:`dsh_headless_argv` with the same env.
    """
    state_dir = Path(state_dir)
    home = state_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    patch_path = state_dir / _PATCH_FILENAME
    patch_path.write_text(_overlay_yaml(base_url, model, ctx), encoding="utf-8")
    argv = [
        "npx", "-y", DSH_PACKAGE,
        "web",
        "--patch", str(patch_path),
        "--port", "0",
    ]
    env = {
        "DSH_HOME": str(home),
        API_KEY_ENV: API_KEY_VALUE,
    }
    return argv, env


def dsh_headless_argv(state_dir: Path, prompt: str) -> list[str]:
    """argv for a one-shot non-interactive prompt through the same overlay.

    Use with the env returned by :func:`dsh_prepare`.  The prompt is passed as
    a single argv element (no shell involved).
    """
    patch_path = Path(state_dir) / _PATCH_FILENAME
    if not patch_path.is_file():
        raise FileNotFoundError(f"run dsh_prepare() first: missing {patch_path}")
    return [
        "npx", "-y", DSH_PACKAGE,
        "headless",
        "--patch", str(patch_path),
        prompt,
    ]


def _cached_package_dir() -> Path | None:
    cache_root = Path.home() / ".npm" / "_npx"
    if not cache_root.is_dir():
        return None
    for entry in cache_root.glob("*/node_modules/@deepseek-ai/dsh"):
        if entry.is_dir():
            return entry
    return None


def dsh_installed() -> bool:
    """True when npx is present and dsh is cached or fetchable via npm."""
    if shutil.which("npx") is None:
        return False
    return _cached_package_dir() is not None or shutil.which("npm") is not None
