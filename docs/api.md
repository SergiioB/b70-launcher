# B70 Launcher HTTP API

`launcher.py` runs a stdlib `ThreadingHTTPServer` on `127.0.0.1`, default port
`7570` (`settings.json` `port`, override with `--port`). Everything below is
derived from `Handler` in `launcher.py`; there is no OpenAPI spec and no
versioning — the served UI and the API ship together.

## Transport rules

- **All requests** need `Host: 127.0.0.1` or `127.0.0.1:<port>`. Any other
  Host (including `localhost`) gets `403 {"error": "invalid Host"}`.
- **GET** requests additionally need the per-process token
  (`secrets.token_urlsafe(32)`), via any one of:
  - `X-Launcher-Token: <token>` header,
  - the `b70_token` session cookie, or
  - a one-time `?token=<token>` query — the server answers `302` to `/` and sets
    the cookie (`HttpOnly; SameSite=Strict`), so the token never stays in the
    URL, history, or Referer headers.
  The launcher writes the token to `<state>/token` (mode 0600) at startup and
  opens the native window / browser with the `?token=` bootstrap URL.
  Unauthenticated GETs return `403 {"error": "unauthorized — open the UI via b70-launcher"}`.
- **POST** requests need all of:
  - `Host: 127.0.0.1[:port]` (same check as GET),
  - `X-Launcher-Token: <token>` (the cookie alone is not accepted for POSTs),
  - `Origin` absent or exactly `http://127.0.0.1:<port>`,
  - `Content-Type: application/json`,
  - body: a JSON object, at most 64 KiB.
- Response bodies are JSON; `Cache-Control: no-store`. Unknown paths return
  `404 {"error": "not found"}`. POST handler exceptions return
  `500 {"error": "launcher error: ..."}`.

## GET endpoints

### `GET /` and `GET /index.html`

Serves `web/index.html` with `__API_TOKEN__` replaced by the session token.
Sets a strict CSP (`connect-src 'self' http://localhost:* http://127.0.0.1:* ws://...`),
`X-Content-Type-Options: nosniff`, and no-cache headers.

### `GET /api/state`

Full launcher state; called by the UI on load and every ~3.5 s refresh.

```json
{
  "recipes":  "merged catalog: shipped recipes.json + recipes-remote.json overlay + user overrides",
  "settings": "merged settings.json + settings-override.json",
  "running":  ["server entries, see /api/launch"],
  "is_win": false,
  "version": "0.5.0",
  "update":  {"checked": true, "has_update": false, "latest_version": "", "message": "", "download_url": "..."},
  "recipe_notices": {"<model_id>:<engine>": {"model_id", "engine", "model_name", "local_ver", "remote_ver", "note", "recommended", "is_new", "can_apply"}},
  "recipes_remote": {"catalog_ver": "YYYY-MM-DD", "checked": true},
  "preflight": {"devices": [{"render", "pci", "vendor", "device", "label", "card", "accessible", "b70", "vram_total_mib", "vram_used_mib", "power_cap_w"}],
                "blockers": ["..."], "notes": ["..."],
                "profile": {"b70_count": 1, "cards": [{"render", "label", "vram_total_mib", "power_cap_w", "accessible", "b70"}]},
                "default": {"model_id", "engine", "reason"}},
  "scan": {"roots": ["..."], "state": "idle|scanning|done", "ts": 0}
}
```

Preflight is cached for 10 s. A B70 is identified by PCI ID
`8086:e223` + subsystem `8086:1701`.

### `GET /api/scan`

Artifact detection. Returns cached scan results; starts a forced rescan when
the configured roots changed or the last scan is older than 120 s, waiting up
to 5 s for an in-flight scan.

```json
{
  "state": "done", "error": null,
  "roots": ["/home/user/models"],
  "catalog": [{"kind": "gguf|openvino|vllm|exl3", "name", "path", "size_mib", "ctx"}],
  "free_gb": 111.0,
  "matches": {"<model_id>": {"<engine>": {"detected": true, "path": "...", "ctx_native": 131072, "size_mib": 21504.0}}}
}
```

`catalog` is every artifact found under the scan roots (depth 8, 90 s budget,
dot-dirs and common junk skipped, symlinks not followed). `matches` is the
per-recipe detection result. `free_gb` is free space on the `models_dir`
filesystem. `error` carries partial-scan notes (e.g. the 90 s budget hit).

### `GET /api/metrics`

Live telemetry snapshot.

```json
{
  "power": [{"index", "name", "pci", "active", "watts", "cap_w", "temp_c",
             "freq_mhz", "util_pct", "vram_total_gb", "vram_used_gb"}],
  "vram": {"used_mib", "total_mib", "source"},
  "<rid>": {"toks", "engine_mem", "tokens_in", "tokens_out", "requests",
            "tok_s", "metrics_source", "stats": {"cpu", "mem"}}
}
```

- `power`: one entry per `xe` hwmon device. `watts` is computed from
  `energy1_input` deltas between calls (null on first sample); `util_pct` is
  `act_freq / max_freq` — the xe driver exposes no busy counter, so the UI
  labels it "GPU clock". `vram_*_gb` comes from `sudo -n cat
  /sys/kernel/debug/dri/<pci>/tile0/vram_mm` and is null without passwordless
  sudo.
- `vram`: first card's sysfs `mem_info_vram_used`, or an estimate
  (`artifact_mib x 1.1`) when sysfs is unavailable.
- Per-server `tokens_*`/`requests` come from the engine's own Prometheus
  `/metrics` on its API port (summed across label sets), falling back to log
  parsing (`metrics_source` tells which). `tok_s` is a smoothed completion
  rate. `stats` is `docker stats` cpu/mem for container engines.

### `GET /api/usage`

`{"sessions": [last 50 records], "totals": {"tokens_in", "tokens_out",
"requests", "sessions"}}` — read from `usage-history.json` (500 records kept on
disk). A session record: `{id, model, engine, port, started, ended, status,
tokens_in, tokens_out, requests, peak_tok_s, artifact_mib}`.

### `GET /api/logs?id=<rid>`

`{"id", "lines": [...]}` — last 60 lines of the last 16 KiB of that server's
log file under `logs/`. `404 {"error": "no such server"}` for an unknown id.

### `GET /api/downloads`

`{"<model_id>-<engine>": {"id", "model", "engine", "state", "done", "total",
"pct", "speed", "eta", "quant", "repo", "files", "dest", "error"}}` —
`state` is `queued | resolving | downloading | done | cancelled | error`.

### `GET /assets/<file>`

Static files confined to `web/` (svg/png/jpg/jpeg); `404` outside that tree or
for other suffixes.

## POST endpoints

All bodies are JSON objects. `model_id` (or legacy `model`) + `engine` identify
a recipe. Fields accepted by the shared `build()` path: `model_id`, `engine`,
`ctx`, `port`, `slots`, `kv`, `gpus` (list of ints, only `0` and `1`), `power`,
`extra` (extra CLI flags string), `extra_env` (`NAME=value` lines), `mtp`
(bool), `use_docker` (bool), `dry_run`, `models_dir`, `harness`,
`harness_cmd`, and `custom_path` when `model_id` is `"__custom__"`.

### `POST /api/build`

Returns the resolved launch plan without starting anything (the bundled UI
does not call this endpoint; `/api/launch` with `dry_run` is the preview path).

Response: `{cmd, tokens, env, warnings, power_cmd, power, cname, endpoint,
served_name, model_name, engine, ctx, ctx_source, detected, detected_path,
native, artifact_mib, harness_line, download, recipe_notice?}` or
`{error, warnings}`.

`power_cmd` is always the fixed "Not generated…" string — the launcher never
writes power settings; `power` only feeds a warning when it exceeds the card's
current hwmon cap.

Side effects: reads the 10 s preflight cache and the last scan. For an `exl3`
recipe with `dry_run` unset/false and the isolated dockerd down, `build()`
attempts to auto-start it (`sudo` containerd/dockerd) — use `dry_run` or
`/api/launch` dry-run for pure inspection.

### `POST /api/launch`

Builds and (unless `dry_run`) starts the engine. `rid` = `"<model_id>-<engine>-<port>"`.

- If that rid is already running/starting and not `dry_run`:
  `{"id", "already_running": true, "harness_line"}` and the configured client
  is opened instead.
- Errors → `400 {"error"}`: build errors, preflight blockers, artifact not
  detected, selected `gpus` not accessible B70s, duplicate rid, port already
  bound on `127.0.0.1`.
- `dry_run` → `{"id", "harness_line", "cmd", "warnings", "env", "power_cmd",
  "detected", "detected_path", "ctx", "recipe_notice?"}`, no process spawned.
- Success → `{"id", "harness_line", "recipe_notice?"}`.

Real-launch side effects: `docker run -d` (or a native `llama_bin` process when
the recipe/settings `llama_bin` exists and `use_docker` is not set), log file
`logs/<rid>-<epoch>.log`, harness config sync (writes `~/.pi/agent/models.json`,
`~/.omp/agent/models.yml`, `~/.factory/settings.json` when those files exist),
`servers-state.json` persist. Engines always publish on `127.0.0.1` only.

### `POST /api/download`

`{"model_id", "engine"}` → `{"id", "error"}`. Starts a background HF download:
`download.repo` must be in the `HF_APPROVED_REPOS` allow-list (optional
`download.revision`, validated as `[A-Za-z0-9._-]+`). `kind=file` fetches
`name` + `extra_files` into `models_dir`; `kind=snapshot` fetches the whole
repo into `ovms_repo` (openvino) or `models_dir` (others) under the repo
basename. Files stream via `.part` with Range resume (4 attempts); existing
files are reused only when sizes match — mismatched/extra files abort with an
error rather than overwrite. Completion triggers a rescan.

### `POST /api/stop`

`{"id"}` → `{"ok": bool}`. Containers: `docker rm -f` on the recipe's socket
(falls back to the default socket), then waits up to ~15 s for the port to
free. Native processes: `SIGTERM` (Popen handle or adopted PID), `SIGKILL`
after ~5 s. Records usage and persists state.

### `POST /api/test_prompt`

`{"prompt"?, "port"?, "model"?}` → proxies a chat completion to the engine on
`127.0.0.1:<port>` with `max_tokens=512` (120 s timeout). Resolves the served
model id via `GET /v1/models` first.

`{"ok": true, "reply", "latency_s", "tokens", "model"}` — `reply` falls back
to `"Thinking: <reasoning_content>"` (plus a truncation note on
`finish_reason=length`) when a reasoning model emits no `content` before the
cap. Failure: `{"ok": false, "error"}`. `port` outside 1-65535 → `400`.

### `POST /api/harness`

`{"harness", "port", "model"/"model_id", "engine", "harness_cmd"?}` →
`{"line"}` — the shell line that was opened. Spawns a terminal
(kitty / gnome-terminal / xfce4-terminal / x-terminal-emulator / xterm) running
the client with `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL_NAME`; for
`webui` it `xdg-open`s `localhost:3000` (or `:8080` if that answers).
`harness_cmd` overrides the command; unknown `harness` ids look up
`settings.harnesses[]`. Side effects: runs `build(cfg)` internally — including
the exl3 dockerd auto-start caveat above — and always syncs the harness config
files (skipped only when `dry_run` is set).

### `POST /api/recipes/update`

`{"model_id"?, "engine"?}` → applies the published catalog referenced by the
update manifest's `recipes_url` (must be `https://`, or `http://127.0.0.1` /
`http://localhost` for testing; 8 MiB cap).

- With both `model_id` and `engine`: applies that model's complete remote
  entry — every published engine, not just the named one.
- Without them: applies the entire remote catalog.

Response `{"ok": true, "applied", "catalog_ver"}` or `400 {"error"}`
(including `"nothing newer to apply"`). Side effects: persists
`recipes-remote.json` in the state dir, re-applies `recipe_overrides` from
`settings-override.json` (they always win), triggers a rescan. See
`docs/recipe-format.md`.

### `POST /api/settings`

`{"scan_dirs": ["..."]}` → `{"ok": true, "scan_dirs": [...]}` or `400`.
List of at most 8 directories; each must exist; `/` and the home directory are
refused. Persists into `settings-override.json` (other keys preserved) and
starts a rescan. No other settings key is writable through the API.

### `POST /api/shutdown`

`{"stop_engines"?: bool}` → `{"ok": true}` then shuts down asynchronously:
cancels downloads, records usage for every tracked server, optionally
`docker rm -f`s containers and terminates native processes, persists
`servers-state.json`, stops the HTTP server. Same path runs on window close
and SIGINT/SIGTERM (without `stop_engines`).
