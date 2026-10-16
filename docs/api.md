# B70 Launcher HTTP API

`launcher.py` (desktop window/daemon) and `cli.py` (headless `b70` command) both
drive the same local HTTP API, served from `b70launcher/http_api.py` by a stdlib
`ThreadingHTTPServer` on `127.0.0.1`, default port `7570` (`settings.json`
`port`, override with `--port`). The route tables `GET_ROUTES` and `POST_ROUTES`
in that module are the authority; this document mirrors them. There is no
OpenAPI spec and no versioning — the served UI and the API ship together.

## Package layout

The backend lives in `b70launcher/`; `launcher.py` and `cli.py` are entry
points, not implementation:

| module | contents |
|---|---|
| `paths.py` | `VERSION`, install/state/log paths, per-port token read/write, atomic writes |
| `locks.py` | the shared `LOCK`, `STATE_LOCK`, `USAGE_LOCK` |
| `catalog.py` | `recipes.json` + `settings.json` load, user overrides, remote recipe overlay, update check |
| `scan.py` | disk scan, artifact detection/quant pinning, context resolution |
| `hardware.py` | GPU preflight (device list, per-kind blockers, busy attribution), power/VRAM probes |
| `downloads.py` | Hugging Face download plan / confirm / cancel / worker |
| `builders/` | launch command construction: `common.py`, `ovms.py`, `vllm.py`, `llamacpp.py`, `exl3.py` |
| `runtime.py` | server state machine, launch/stop/adopt, snapshots, preflight entry point |
| `telemetry.py` | engine metrics scraping, token accounting, usage history |
| `harness.py` | client config sync and terminal spawning |
| `http_api.py` | the API described here (route tables + auth middleware) |
| `exl3_stack.py` | isolated containerd/dockerd startup for EXL3 recipes |

Importing any module has no side effects; the entry point calls
`paths.init_state(port)` and `catalog.load()` explicitly.

## Transport rules

- **Host header**: `127.0.0.1`, `localhost`, `127.0.0.1:<port>`, or
  `localhost:<port>` for this server's port. Anything else gets
  `403 {"error": "unauthorized request"}`.
- **Token**: `paths.write_token(port)` writes `secrets.token_urlsafe(32)` to
  `<state>/token-<port>` (mode 0600) at startup. `GET /?nonce=<nonce>` and
  `POST /api/window` are the only ways to hand the token to a browser.
- **GET** needs the token, via either
  - `X-Launcher-Token: <token>` header, or
  - the `b70_token_<port>` cookie (`HttpOnly`, `SameSite=Strict`, `Path=/`).
  Otherwise `403 {"error": "unauthorized request"}`.
- **Nonce bootstrap**: `POST /api/window` mints a nonce (`secrets.token_urlsafe(24)`,
  valid 30 s) and returns it as `/?nonce=<nonce>`. A GET of that URL is consumed
  on first use (`BOOTSTRAPS.pop`) and answers `302` to `/` with
  `Set-Cookie: b70_token_<port>=<token>; Path=/; SameSite=Strict; HttpOnly`, so
  the token never appears in the page URL. There is no `?token=` bootstrap.
- **POST** needs all of: valid Host, `X-Launcher-Token` header (the cookie alone
  is not accepted), `Origin` absent or exactly `http://127.0.0.1:<port>` /
  `http://localhost:<port>`, `Content-Type: application/json`, and a JSON object
  body of at most 64 KiB. Otherwise `403`.
- Responses are JSON with `Cache-Control: no-store`,
  `X-Content-Type-Options: nosniff`, and
  `Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`.
- Errors: unknown path `404 {"error": "not found"}`; malformed JSON body
  `400 {"error": "invalid JSON body"}`; a `ValueError`/`OSError` inside a handler
  `400 {"error": "<message>"}`; any other POST exception
  `500 {"error": "Launcher request failed: <exc>"}`. Broken-pipe writes while
  sending a response are swallowed.

## GET endpoints

Served from `GET_ROUTES` plus three special paths (`/`, `/index.html`,
`/api/logs`).

### `GET /` and `GET /index.html`

Serves `web/index.html` with `__API_TOKEN__` replaced by the session token
(`text/html; charset=utf-8`). The page then talks to the API with the
`X-Launcher-Token` header for both GETs and POSTs.

### `GET /api/state`

Full launcher state; the UI polls it every ~2.5 s (power every ~1.5 s).

```json
{
  "recipes":  "merged catalog: recipes.json + recipes-remote.json overlay + recipe_overrides",
  "settings": "merged settings.json + settings-override.json",
  "running":  ["server entries, see POST /api/launch"],
  "version":  "0.6.13",
  "is_win":   false,
  "update":   {"checked": true, "has_update": false, "latest_version": "", "message": "", "download_url": "https://xecores.com/match"},
"recipe_notices": {"<model_id>:<engine>": {"model_id", "engine", "model_name", "local_ver", "remote_ver", "note", "recommended", "is_new", "can_apply"}},
  "recipes_remote": {"checked": true, "catalog_ver": "2026-10-15.2", "recipes_url": "...", "entries": {"<model_id>:<engine>": {"ver", "note", "recommended"}}},
  "clients": {"pi": {"installed", "openable"}, "omp": {...}, "droid": {...}, "codex": {...}, "deepseek": {...},
              "webui": {"installed", "openable", "url", "detail"},
              "none": {"installed": true, "openable": true},
              "terminal": {"installed", "name"}, "display": {"available", "reason"}},
  "custom_templates": {"llamacpp": {...}, "openvino": {...}, "vllm": {...}, "exl3": {...}},
  "preflight": {"devices": [...], "blockers": [...], "blockers_by_kind": {"native": [], "docker": [], "exl3": []},
                "notes": ["..."], "profile": {"b70_count": 1, "cards": [...]},
                "default": {"model_id", "engine", "reason", "gpus": [0]}},
  "scan": {"state": "idle|scanning|done", "ts": 0, "roots": ["..."]}}
 ```

`clients` reports which target applications this machine can open right now
(cached ~10 s): `installed` = binary present (`webui`: a server answered on
localhost:3000), `openable` = a GUI display is reachable. `custom_templates`
is `builders.CUSTOM_TEMPLATES` verbatim, so clients don't duplicate engine
defaults. Server entries in `running` additionally carry the launch-time
recipe snapshot — `quant`, `variant`, `found_name`, `custom`, `recipe_ver`,
`ctx_source`, `detected_path`, `artifact_mib`, `client` (target app), and
`client_open` (timestamp of the last successful app open) — plus `ready`
(true exactly when `status` is `running`, i.e. the endpoint answered
`/v1/models`). These fields never change after launch: later catalog, scan
or settings edits do not rewrite an existing session (`served_name` is the
one exception — the engine reports it at first ready).
Preflight is cached for 10 s (`hardware.PREFLIGHT_CACHE`); `runtime.preflight()`
re-runs `enrich_busy` on top so the `busy`/`busy_by` fields are current. A B70 is
identified by PCI ID `8086:e223` + subsystem `8086:1701`.

Each `preflight.devices[]` entry carries `pcie: {gen, width, gt_s, max_gen,
max_width, gbps, source, detected}`. `gen`/`width` are the link the card
trained at, read from the root port above it (the B70's internal switch
reports a virtual 2.5 GT/s x1 below that and is ignored); `max_gen`/`max_width`
are the card's own maximum from its uplink; `gbps` is the approximate one-way
bandwidth. `source` is `sysfs`, `user` (a `pcie_links` override, with the sysfs
values kept in `detected`), or `unknown` (all null). The UI shows the link on
every GPU choice and warns when a multi-card recipe would run over a narrower
or uneven link; `POST /api/build` adds the same warning.

### `GET /api/scan`

Artifact detection; runs `scan.scan_blocking()` (a forced rescan when the roots
changed or the last scan is older than 120 s, waiting up to 5 s for an in-flight
scan).

```json
{
  "state": "done", "ts": 0, "error": null,
  "roots": ["/home/user/models"],
  "catalog": [{"kind": "gguf|openvino|vllm|exl3", "name", "path", "size_mib", "ctx"}],
  "free_gb": 111.0,
  "matches": {"<model_id>": {"<engine>": {"detected": true, "path": "...", "size_mib": 21504.0, "ctx_native": 131072, "reason": ""}}}
}
```

`catalog` is every artifact found under the scan roots (depth 8, 90 s budget,
dot-dirs and common junk skipped). `free_gb` is free space on the `models_dir`
filesystem. `error` carries partial-scan notes. Each match's `reason` is `""`
when `detected` is true, `"incompatible: …"` when a same-family artifact exists
on disk but no variant satisfies the recipe (wrong format, quantization pin,
sidecar, or a sibling another recipe owns), and `"not downloaded"` when nothing
matching is on disk.

### `GET /api/power`

`{"power": [...]}` — one entry per `xe` hwmon device, empty on Windows. Each
entry: `{index, name, pci, active, watts, cap_w, temp_c, freq_mhz, util_pct,
vram_total_gb, vram_used_gb}`. `watts` comes from `energy1_input` deltas between
calls (null on the first sample); `util_pct` is `act_freq / max_freq`, which the
UI labels "GPU clock" because the xe driver exposes no busy counter.
`vram_*_gb` is null without passwordless sudo (`/sys/kernel/debug/dri/.../vram_mm`).

### `GET /api/servers`

`{"running": [server entries], "scan": {"state", "ts"}}` — the lightweight poll
the UI uses for live status; it refreshes container status, readiness, load
phase, and token counters as a side effect.

### `GET /api/downloads`

`{"<model_id>-<engine>": {"id", "model_id", "model", "engine", "state", "done", "total",
"pct", "speed", "eta", "dest", "quant", "started", "finished", "file",
"files_done", "files_total", "resumed_bytes", "note"?, "error"?}}` — `state` is
`queued | resolving | downloading | done | cancelled | error`. `total` is the
planned byte size and `done` includes bytes already on disk when the transfer
began (`resumed_bytes`). `speed` is bytes/s averaged over the last 5 s across
files (0 until 1 s of samples); `eta` is seconds left or null. `file` is the
file in transfer, `started`/`finished` are epoch seconds. Finished entries stay
listed until the daemon restarts. `note` names a draft/sidecar group when one
is being fetched. A 401/403 from Hugging Face stops at once (no retries, the
`.part` stays) with `error_code` `gated` or `auth` plus `repo_url`. The internal
cancel flag is not exposed.

### `GET /api/metrics`

Full telemetry snapshot for debug callers (the UI polls `/api/power` and
`/api/servers` instead, and `docker stats` only runs while a live server exists).

```json
{
  "power": [ ...same entries as /api/power... ],
  "vram":  {"used_mib", "total_mib", "source"},
  "<rid>": {"toks", "engine_mem", "tokens_in", "tokens_out", "requests",
            "tok_s", "metrics_source", "stats": {"cpu", "mem"}}
}
```

Per-server counters come from the engine's own Prometheus `/metrics`
(`metrics_source: "engine /metrics"`), falling back to log parsing
(`"engine log"`). `tok_s` is a smoothed completion rate; `stats` is
`docker stats` cpu/mem for container engines. `vram` is the first card's sysfs
`mem_info_vram_used` (`"sysfs"`) or an estimate (`artifact_mib x 1.1`).

### `GET /api/usage`

`{"sessions": [last 50 records], "totals": {tokens_in, tokens_out, requests,
sessions}, "clients": {...}, "today": {...}, "by_model": {...}}`.
`sessions`/`totals` come from `usage-history.json` (500 records kept on disk). A
session record: `{id, model, engine, port, started, ended, status, tokens_in,
tokens_out, requests, peak_tok_s, artifact_mib}`.

`clients` (all time) and `today` map an agent id (`pi`, `omp`, `droid`,
`codex`, `deepseek`, `other`) to `{requests, tokens_in, tokens_out, errors,
last_used}`; `by_model` maps a served model name to the same shape. They count
requests that went through the client gateway (below) and are stored in
`client-usage.json` in the state directory (one row per day, one year kept). A
request the agent cancelled counts as a request, not an error; an engine reply
other than HTTP 200 counts as an error and is printed in the daemon output.

## Client gateway `/gw/<client>/<engine_port>/v1/...`

Agents opened from the launcher get `http://127.0.0.1:<daemon port>/gw/<client>/<engine port>/v1`
as their base URL, so the launcher can count tokens per agent. No token is
needed (agents are local CLI processes), but the Host check applies and any
request carrying an `Origin` header (a web page) gets `403`. A port without a
`running` server answers `503`.

- `POST .../v1/chat/completions` and `.../v1/completions`: forwarded. A streamed
  request without `stream_options.include_usage` gets it added, and the extra
  usage-only chunk is removed before the agent sees it.
- `POST .../v1/responses` (Codex): llama.cpp, vLLM and EXL3 serve it natively;
  the gateway first merges `instructions` and developer/system messages into one
  leading system message and drops `store`/`include`. OpenVINO OVMS gets a
  Responses ↔ Chat Completions translation (`b70launcher/responses_translate.py`).
- Anything else (`GET .../v1/models`, …) is passed through unchanged.

Each server entry gains `clients: {"<client>": {requests, tokens_in,
tokens_out, errors, last_used}}` for the current session. Set
`"meter_clients": false` to point agents at the engine port directly.

### `GET /api/logs?id=<rid>&offset=<bytes>`

Byte-offset log tail, used by `b70 logs -f` and the UI log dialog.

- Without `offset`: the last 64 KiB of the log, trimmed to the last 200 lines.
- With `offset`: up to 64 KiB starting at that byte offset, untrimmed; an offset
  past the end is clamped to the file size and a negative one to zero.
- Response: `{"id", "offset", "lines", "eof"}` — pass the returned `offset` back
  on the next poll. `eof` is true when the read reached the current end of file.
- Unknown `id` → `404 {"error": "no such server"}`; a non-numeric `offset` → `400`.
  A missing log file returns an empty tail rather than an error.

### `GET /assets/<path>`

Static files under `web/assets/` (icons and engine/model logos), any file type,
`Content-Type` from `mimetypes`. Paths are resolved and must stay inside that
directory; anything else → `404 {"error": "not found"}`.

## POST endpoints

All bodies are JSON objects. `model_id` (or legacy `model`) + `engine` identify
the recipe. Fields read by the shared `builders.build()` path: `model_id`,
`engine`, `ctx`, `port`, `slots`, `kv`, `prefill` (EXL3: `int8` default or `fp16`), `cache` (`on` or `off`; omitted =
recipe default, which is `on` unless the recipe sets `prefix_caching: false`; any other value warns and uses the
recipe default), `gpus` (list of ints), `power`, `extra`
(raw engine flags string), `extra_env` (`NAME=value` lines), `mtp` (bool),
`use_docker` (bool), `dry_run`, `models_dir`, `harness`, `harness_cmd`,
`no_test`, and `custom_path` when `model_id` is `"__custom__"`.

### `POST /api/window`

`{}` → `{"url": "http://127.0.0.1:<port>/?nonce=<one-time>"}`. The nonce is
valid 30 s and consumed by the first GET that uses it. Used by `launcher.py` to
open the desktop window or browser, and by `b70 open`.

### `POST /api/build` and `POST /api/preview`

Same handler: builds with `dry_run: true` and appends `harness_line`. The
bundled UI calls `/api/build` for the live plan and warning text; `/api/preview`
is kept as an alias.

```json
{"cmd", "tokens", "env", "warnings", "operator_hold", "cname", "docker_sock",
 "endpoint", "served_name", "model_name", "engine", "kind", "rid", "gpus",
 "ctx", "ctx_source", "quant", "variant", "found_name", "custom", "recipe_ver",
 "detected", "detected_path", "native", "effective_binary", "artifact_mib",
 "harness_line"}
```

or `{"error", "warnings"}` (unknown model+engine, unknown recipe kind, or a
rogue `models_dir`). `effective_binary` is the
native binary when one is used, else the recipe image. `ctx_source` is
`"config.json on disk"`, the recipe's `ctx_note`, or the built-in default label
for that recipe kind. A recipe with `operator_hold` set only warns here; a real
launch is refused.

### `POST /api/launch`

Builds and (unless `dry_run`) starts the engine. `id` = `<model_id>-<engine>-<port>`
(sanitized), i.e. `rid`.

- Build error → `400 {"error"}` — including a recipe on `operator_hold`, which
  only warns in the dry-run/plan path.
- `dry_run` → `200 {"id", "harness_line", "cmd", "warnings", ...}` plus the
  build fields above; nothing is spawned, and an occupied port only adds a warning.
- Preflight blockers for this launch kind → `409 {"error"}`; a not-detected
  artifact → `409`; a selected GPU that is not an accessible B70 or that is
  already in use → `409`.
- `runtime.launch()` refusals → `409 {"error", "id"}`: the rid is already live,
  the port is reserved by another launch or already bound, or a selected GPU is
  held by another tracked server.
- Success → `{"id", "status": "starting", "harness_line"}`.

Real-launch side effects: an attached `docker run --rm` whose output streams into
the server log (or a native `llama_bin` process when the recipe/settings
`llama_bin` exists and `use_docker` is not set). The container receives only the
selected cards' render/card nodes and their `/dev/dri/by-path` links. The log file
`logs/<rid>-<epoch>.log` also receives the command line; harness config sync
(writes `~/.pi/agent/models.json`, `~/.omp/agent/models.yml`,
`~/.factory/settings.json` when those files exist), and `servers-state.json`
persistence. EXL3 recipes start their isolated containerd/dockerd first
(`exl3_stack.ensure_exl3_stack`, `sudo -n`, never deletes daemon state). Engines
always publish on `127.0.0.1` only.

### `POST /api/download`

`{"model_id", "engine", "confirm"?}`.

- Without `confirm`: the plan —
  `{"size_bytes", "partial_bytes", "remaining_bytes", "dest_dir", "free_bytes", "enough", "error", "groups"}` with
  `groups: [{"repo", "dest_dir", "files", "size_bytes", "partial_bytes", "label"}]` (an
  `exl3`/`vllm-dflash` recipe can have a `main` and a `draft` group).
  `partial_bytes` counts complete files of the right size plus `.part` files
  already on disk; `enough` compares free space with `remaining_bytes` × 1.05, so
  a paused download can resume on a nearly full disk. Cancelling keeps the
  `.part` files and the next start resumes them. `gated` (`false|"auto"|"manual"`),
  `hf_token` (a Hugging Face token was found: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
  `$HF_HOME/token`, then `~/.cache/huggingface/token`) and `repo_url` describe
  access. A gated repo without a token is refused at plan time with the steps to
  fix it. The token is sent only to `huggingface.co`, never across a redirect to
  another host, and never appears in a response or log.
- With `confirm: true` and a usable plan → `{"id", "error": null}` and a
  background worker; a rejected plan (space, unknown size, operator hold, repo
  not in `HF_APPROVED_REPOS`, pinned filename missing, unsafe paths) → `409`.
- Download start failure after confirm → `400 {"id", "error"}`.

`download.repo` must be in `HF_APPROVED_REPOS` (`b70launcher/downloads.py`);
`download.revision` is validated as `[A-Za-z0-9._-]+`. `kind=file` fetches
`name` + `extra_files` into `models_dir`; `kind=snapshot` fetches the repo into
`ovms_repo` (openvino) or `models_dir` (others), under the repo basename or
`download.dest`. Files stream via `.part` (4 attempts for network errors, none
for 401/403); existing files are reused only when sizes match. Free space must
cover the remaining bytes × 1.05, and an unknown remote size refuses the
download. Completion triggers a rescan.

### `POST /api/download/cancel`

`{"id"}` → `{"ok": bool}`. Cooperative: the worker stops at the next chunk
boundary. `false` for an unknown or already finished id.

### `POST /api/stop`

`{"id"}` → `{"ok": bool}` (`false` for an unknown id). Containers:
`docker stop --time 5` then `docker rm` on the recipe's socket, each with a 15 s
timeout; "No such container" is tolerated. Native processes: `SIGTERM` to the
process group (or the adopted PID), `SIGKILL` after ~5 s. Records usage, persists
state; a Docker refusal raises → `400 {"error": "Stop failed: ..."}`.

### `POST /api/test_prompt`

`{"id"?, "port"?, "model"?, "prompt"?, "max_tokens"?}` → proxies one chat
completion to `127.0.0.1:<port>` with `temperature 0`, `enable_thinking: false`,
and `max_tokens` 512 by default (120 s timeout). The `id` is looked up in the
running table; otherwise the port is used (default 8000). A port outside
1-65535 → `400`. An id/port resolving to a server that is not `running`
returns `ok: false` without dialing the endpoint:
`"Server <id> is <status> (<phase>); the endpoint is not ready yet."`

```json
{"ok", "answer", "reasoning", "finish_reason", "http_status", "elapsed_s",
 "tok_s", "error", "model"?, "tokens"?}
```

Reasoning text is separated from the answer: `reasoning_content`/`reasoning`
plus any `<think>...</think>` blocks are moved to `reasoning`, and `ok` is true
only when `answer` is non-empty. A reasoning-only reply fails with
`"Engine returned reasoning without an answer."`; an engine error is reported in
`error` with `ok: false` (HTTP 200, not a 5xx).

### `POST /api/harness`

`{"id"|"port", "harness"?, "harness_cmd"?}` → `{"line", "opened", "client"}`.
The id (or fallback port) must name a server whose status is `running`, otherwise
`400 {"error": "Choose a ready server to open a client."}`. `line` is the shell
line that was spawned with `OPENAI_BASE_URL` / `OPENAI_API_KEY` /
`OPENAI_MODEL_NAME`. Clients:

- `pi`, `omp`: `<client> --model b70/<served name>`, plus `--no-tools` when the
  engine has no tool parser (`harness.supports_tools`).
- `droid`: `droid --model custom:Desktop-B70-Loaded-Model-0`.
- `codex`: Codex CLI with `-c` overrides only (`b70launcher/codex_client.py`),
  nothing written to `~/.codex`: provider `b70`, `wire_api="responses"`,
  `env_key` + `requires_openai_auth=false`, the context window, web search off,
  no update check.
- `deepseek`: `npx -y @deepseek-ai/dsh@0.2.0-rc.2 web --patch <state>/dsh/gateway.yml --port 0`
  with `DSH_HOME=<state>/dsh/home`, so `~/.dsh` is untouched
  (`b70launcher/dsh_client.py`). dsh 0.2 has no terminal UI; it opens its web UI.
- `webui` (`xdg-open http://localhost:3000`), or `none` (line only, `opened: false`).

Before an agent opens, its config (`~/.pi/agent/models.json`,
`~/.omp/agent/models.yml`, `~/.factory/settings.json`) is pointed at this
server through the gateway; a file is created only inside an agent folder that
already exists. The agent runs in a terminal titled `<Agent> · <model>`, starts
in `settings.agent_dir` (default `~`), and the terminal waits for Enter after it
exits. An agent that is not installed → `400` naming the missing binary.
Headless (no `DISPLAY`/`WAYLAND_DISPLAY`) → `400 {"error", "line"}` with the
command to run by hand. A missing terminal (kitty, gnome-terminal,
xfce4-terminal, x-terminal-emulator, xterm) or a client that exits nonzero
within ~1.2 s → `400` with the exact reason while the server stays ready. A
successful open stamps `client_open` and `client` on the server entry.
`harness_cmd` overrides the command text.

### `POST /api/recipes/update`

`{"model_id"?, "engine"?}` → fetches the catalog referenced by the update
manifest's `recipes_url` (https, or `http://127.0.0.1` / `http://localhost` for
testing; 8 MiB cap).

- With both `model_id` and `engine`: only that model's named recipe is overlaid.
- Without them: the whole published document is applied.

The overlay is field-merged, not wholesale: model fields are limited to
`REMOTE_MODEL_FIELDS`, recipe fields to `REMOTE_RECIPE_FIELDS`
(`b70launcher/catalog.py`) — exec-shaping keys (`kind`, `image`, `llama_bin`,
`docker_sock`, `fixed_flags`, `serve_config`) and host paths (`model_path`,
`gguf`, `source_model`) never come from the wire. Models or engines this app
does not ship are refused, not added.

Response `{"ok": true, "applied": <count>, "catalog_ver": "..."}` or
`400 {"error"}` — including `"nothing newer to apply"` and
`"recipe needs a newer launcher app (update b70-launcher): <keys>"`. Side
effects: persists `recipes-remote.json` in the state dir, re-applies
`recipe_overrides` from `settings-override.json` (they always win), and triggers
a rescan. See `docs/recipe-format.md`.

### `POST /api/settings`

```json
{"scan_dirs"?: ["..."], "models_dir"?: "...", "open_with"?: "pi|omp|droid|codex|deepseek|webui|none",
 "sync_clients"?: true, "meter_clients"?: true,
 "pcie_links"?: {"<pci>": {"gen": 1-6, "width": 1|2|4|8|16} | null} | null}
```

Any other key, or an empty body, → `400`. `scan_dirs`: at most 8 absolute,
existing directories; `/` and the home directory are refused. `models_dir`: one
absolute, existing directory (same refusals). Persists into
`settings-override.json` (other keys preserved) and starts a rescan.
`pcie_links` sets a card's PCIe link by PCI address (`0000:0b:00.0`) when the
detected value is wrong. It is merged onto the stored map, so one card can be
posted at a time; a `null` entry removes that override and a `null` map clears
all of them. Invalid values → `400`. It invalidates the cached preflight and
does not rescan unless other keys are posted with it. No other
settings key is writable here; edit the file for `ovms_repo`, `llama_bin`,
`exl3_data_root`, `agent_dir`, and `recipe_overrides`.

### `POST /api/shutdown`

`{"stop_engines"?: bool}` → `{"ok": true}`, then shuts down asynchronously
(`runtime.graceful_shutdown`): cancels all downloads, optionally stops every
tracked server, persists `servers-state.json`, and stops the HTTP server. The
same path runs on SIGINT/SIGTERM (without `stop_engines`). Closing the app
window runs it only when no engine is starting or running; otherwise the daemon
stays up for the agents using its gateway and exits once no engine is left and
no window has polled `/api/state` for a minute.
