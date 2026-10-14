"""Launch-argv/env builder for the Codex CLI against a local B70 engine.

The B70 launcher serves every engine through a token-metering gateway, e.g.
``http://127.0.0.1:<daemon_port>/gw/codex/<engine_port>/v1``.  Codex 0.159
speaks the native Responses API (``wire_api = "responses"``), so pointing it at
an engine only requires a provider block plus a few model settings -- all of
which can be passed on the command line as ``-c key=value`` overrides, so this
module never writes (or asks Codex to write) ``~/.codex/config.toml``.

Verified against ``codex-cli 0.159.0`` (``/usr/bin/codex``):

* ``-c`` values are parsed as TOML, so string values (model names, URLs) must be
  TOML-quoted -- ``-c model_providers.b70.base_url="http://..."``.
* A custom provider needs no OpenAI login as long as it supplies an API key.
  ``model_providers.b70.env_key = "B70_CODEX_API_KEY"`` makes Codex send
  ``Authorization: Bearer <value of $B70_CODEX_API_KEY>``; any non-empty value
  works, so :func:`codex_env` exports a placeholder.
* ``requires_openai_auth = false`` is the documented default but is stated
  explicitly here so a stale user config cannot flip it.
* OpenAI-only features are switched off: ``web_search = "disabled"`` removes the
  hosted ``web_search`` tool from the request (``tools.web_search = false`` does
  *not*), and ``model_reasoning_summary = "none"`` stops Codex asking for
  reasoning summaries.
* ``model_context_window`` is the live key (integer).  0.159 dropped
  ``model_max_output_tokens`` -- ``-c model_max_output_tokens=<n>`` is silently
  ignored.  The remaining knob is ``model_auto_compact_token_limit``, which is
  set to ``ctx - max_out`` so compaction triggers while there is still room for
  a full-length reply inside the engine's real window.
"""

from __future__ import annotations

#: Provider id used for the ``model_providers.<id>`` overrides.
PROVIDER_ID = "b70"

#: Human-readable provider name shown by Codex.
PROVIDER_NAME = "B70 Local Engine"

#: Environment variable Codex reads the bearer token from (provider ``env_key``).
API_KEY_ENV = "B70_CODEX_API_KEY"

#: Placeholder token value; the gateway/engine ignores the bearer but Codex
#: refuses to start when ``env_key`` points at an unset variable.
API_KEY_PLACEHOLDER = "b70-local"

#: Loopback hosts that must bypass any ambient HTTP(S) proxy.
_LOOPBACK = ("127.0.0.1", "localhost", "::1")

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def toml_str(value: str) -> str:
    """Render *value* as a TOML basic string for the right-hand side of ``-c``."""
    out = ['"']
    for ch in value:
        out.append(_ESCAPES.get(ch, ch))
    out.append('"')
    return "".join(out)


def _provider_overrides(base_url: str) -> list[str]:
    prefix = "model_providers.%s" % PROVIDER_ID
    return [
        "-c", "%s.name=%s" % (prefix, toml_str(PROVIDER_NAME)),
        "-c", "%s.base_url=%s" % (prefix, toml_str(base_url)),
        "-c", '%s.wire_api="responses"' % prefix,
        "-c", "%s.env_key=%s" % (prefix, toml_str(API_KEY_ENV)),
        "-c", "%s.requires_openai_auth=false" % prefix,
    ]


def _auto_compact_limit(ctx: int, max_out: int) -> int:
    """Token threshold at which Codex compacts, leaving *max_out* headroom."""
    return max(1, int(ctx) - int(max_out))


def _model_overrides(model: str, ctx: int, max_out: int) -> list[str]:
    return [
        "-c", "model_provider=%s" % toml_str(PROVIDER_ID),
        "-c", "model_context_window=%d" % int(ctx),
        "-c", "model_auto_compact_token_limit=%d" % _auto_compact_limit(ctx, max_out),
        "-c", 'web_search="disabled"',
        "-c", 'model_reasoning_summary="none"',
        # The TUI otherwise replaces itself with `npm install -g @openai/codex`
        # on startup when it believes a newer release exists.
        "-c", "check_for_update_on_startup=false",
    ]


def codex_argv(base_url, model, ctx, max_out, codex_bin="codex") -> list[str]:
    """argv that opens the Codex interactive TUI against *base_url*.

    ``base_url`` is the engine (or gateway) ``.../v1`` root, ``model`` the model
    id the engine expects (default ``"active"`` in the launcher), ``ctx`` the
    engine context window in tokens and ``max_out`` its maximum reply length.
    """
    return (
        [codex_bin]
        + _provider_overrides(base_url)
        + _model_overrides(model, ctx, max_out)
        + ["-m", str(model)]
    )


def codex_exec_argv(base_url, model, ctx, max_out, codex_bin="codex", prompt=None) -> list[str]:
    """Non-interactive variant of :func:`codex_argv` (``codex exec``).

    Same overrides, plus ``--skip-git-repo-check`` so the smoke check runs in a
    scratch directory.  When *prompt* is given it is appended as the argument.
    """
    argv = (
        [codex_bin, "exec", "--skip-git-repo-check"]
        + _provider_overrides(base_url)
        + _model_overrides(model, ctx, max_out)
        + ["-m", str(model)]
    )
    if prompt is not None:
        argv.append(prompt)
    return argv


def codex_env(base) -> dict:
    """Return *base* plus the environment Codex needs for a loopback engine.

    Adds the placeholder bearer token (unless the caller already provides a real
    one) and makes every loopback host bypass any ambient HTTP proxy, in both
    the upper- and lower-case spellings honoured by curl/libcurl.
    """
    env = dict(base or {})
    if not env.get(API_KEY_ENV):
        env[API_KEY_ENV] = API_KEY_PLACEHOLDER

    existing = [part.strip() for part in env.get("NO_PROXY", "").split(",") if part.strip()]
    for host in _LOOPBACK:
        if host not in existing:
            existing.append(host)
    value = ",".join(existing)
    env["NO_PROXY"] = value
    env["no_proxy"] = value
    return env
