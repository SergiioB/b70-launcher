# b70 — headless CLI

`b70` (`cli.py`) is the command-line face of B70 Launcher. It talks to the
same daemon the windowed app uses (`launcher.py` + the `b70launcher/` package,
`127.0.0.1:7570` by default), so every recipe the UI can launch — engine keys
`openvino`, `vllm`, `llamacpp`, `exl3`, recipe kinds `ovms`, `vllm`, `vllm-mtp`,
`vllm-tp2`, `vllm-arext`, `vllm-autoround`, `vllm-dflash`, `gguf`,
`gguf-tiered`, `exl3` — launches identically from a terminal with no display at
all.

Stdlib only, Python 3.9+, no GUI dependencies. Installed as
`~/.local/bin/b70` by `packaging/install.sh`.

## Mental model

- The daemon owns engines, downloads, scans and usage state. `b70` is a thin
  client over its JSON API (`docs/api.md`).
- If the daemon is down, any command that needs it auto-starts it
  (`launcher.py --no-open`, detached; log at
  `~/.local/state/b70-launcher/logs/daemon.log`). `--no-autostart` disables.
- Engines always publish on `127.0.0.1`. Reaching them from another box is an
  SSH tunnel (`b70 open` prints the line for you).
- Exit codes: `0` ok · `1` operation error · `2` usage error · `3` daemon
  unreachable · `130` Ctrl+C. `--json` emits machine-readable output for
  every command; errors come back as `{"ok": false, "error", "code"}`.
  `NO_COLOR` is honored; colors auto-disable off a TTY. On a narrow terminal
  tables drop their least useful columns (names, kind, load time) instead of
  wrapping, and long notes wrap to the terminal width.

## Commands

| command | does |
|---|---|
| `b70 serve` | start the daemon (idempotent; `--fg` foreground, `--kill` replaces a running daemon, engines keep running) |
| `b70 down` | stop the daemon (`--stop-engines` also stops engines) |
| `b70 list` | every model × engine recipe: kind, validated context, detected artifact, live mark (`-e` filters) |
| `b70 show <model>` | model detail: validated / default / max context, detected path, launch/download hints |
| `b70 launch <model>` | launch a recipe and wait for the OpenAI endpoint |
| `b70 launch` | interactive picker on a TTY (model → engine, detected/recommended marks) |
| `b70 launch <path>` | serve any artifact under a scan root (`__custom__` recipe) |
| `b70 stop [rid]` | stop an engine (`--all` for everything; no arg = the only running one; nothing running = exit 0) |
| `b70 status` | GPUs (draw/cap/temp/busy %/VRAM) + engines (status, load time, tok/s, tokens) |
| `b70 monitor` | live dashboard on a TTY; one snapshot when piped (`-f` streams, `--once` single) |
| `b70 logs <rid>` | engine log tail (`-f` follow by byte offset, `-n` lines, default 60) |
| `b70 test [rid|port] [prompt]` | smoke prompt end-to-end; latency + approx tok/s (`--show-thinking` also prints the reasoning text) |
| `b70 download <model> -e <eng>` | fetch the recipe artifact; size/space check first, then progress to done (`--dry-run` shows the check only, `--yes` skips the question) |
| `b70 open [ui|rid|port]` | open UI or engine endpoint; prints the URL + SSH hint when headless |
| `b70 env [rid]` | `export OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL_NAME` |
| `b70 agent <client> [rid]` | open a coding agent (`pi`, `omp`, `droid`, `codex`, `deepseek`) on a running model |
| `b70 usage` | tokens each agent used: today, all time, and per model |
| `b70 doctor` | render nodes, B70 identification, blockers, scan roots |
| `b70 scan` | detected artifacts + per-recipe matches (`--roots a,b` to set roots) |
| `b70 recipes-update [model]` | apply the published recipe catalog overlay (`-e` limits it to one engine, `--check` only lists what is newer) |
| `b70 version` | CLI / daemon / catalog versions |
| `b70 completion bash` | bash completion script |

Aliases: `ls`, `recipes` → `list`; `ps` → `status`; `ui` → `open ui`;
`tokens` → `usage`.

## Context: the validated value by default

`b70 list` and `b70 show` print the recipe's validated context (`ctx_safe`,
the value the UI preselects), falling back to the recipe `ctx` when no
validated value exists. `b70 show` also lists the recipe default and the
ceiling (`ctx_max`) when they differ. `b70 launch` without `--ctx` sends the
same validated context the UI would, so a CLI launch and a UI launch of the
same recipe start identical engines.

## Agents and token use

```sh
b70 agent pi                 # Pi on the only running model
b70 agent codex qwen38-27b   # Codex on a chosen server (id, model name or port)
b70 usage                    # per-agent tokens: today, all time, per model
b70 usage --json             # the raw GET /api/usage document
```

`b70 agent` asks the daemon (`POST /api/harness`) to open the agent in a new
terminal, wired to the launcher's metering gateway, and prints the command it
ran. With no display it prints the command for you to run yourself. With no
model running it says so and points at `b70 launch`; a model that is still
loading is refused until it is ready.

`b70 usage` reads `GET /api/usage`. Agents only show up after they have sent
traffic through the gateway; until then it prints
`No agent traffic recorded yet.`

## `b70 launch` — every knob maps to the recipe API

```sh
b70 launch qwen38-27b                      # recommended engine (exl3 here), validated ctx
b70 launch qwen36-35b -e vllm              # explicit engine
b70 launch qwen36-35b -e llamacpp --ctx 65536 --port 8080 --kv q5_0/q4_1
b70 launch qwen38-27b-sc5 -e exl3 --prefill fp16   # EXL3 max-quality prefill (slower cold prompts)
b70 launch qwen36-35b -e vllm --cache off  # prefix caching off: testing/benchmarks only
b70 launch qwen38-27b-fp8-tp2              # recipe declares dual-card → gpus 0,1 auto
b70 launch qwen38-flashnext --gpus 0,1     # same, manual
b70 launch qwen36-35b -e vllm --slots 8 --no-mtp --extra "--async-scheduling"
b70 launch /mnt/models/foo.gguf            # custom artifact, engine sniffed (.gguf → llamacpp)
b70 launch /mnt/models/foo-dir -e openvino # custom OpenVINO IR dir
b70 launch qwen38-27b --dry-run            # resolved cmd/env/warnings, nothing started
b70 launch qwen36-35b --no-wait            # fire-and-forget; poll with b70 status
b70 launch nemotron-35 --timeout 1800      # bigger engines need longer than 900s
```

Flags → API fields (`docs/api.md`): `--ctx`→`ctx` · `--port`→`port` ·
`--slots`→`slots` · `--kv`→`kv` · `--prefill`→`prefill` · `--cache`→`cache` · `--gpus`→`gpus` · `--power`→`power` ·
`--mtp/--no-mtp`→`mtp` · `--docker/--native`→`use_docker` · `--extra`→`extra`
· `-E/--env`→`extra_env` (repeatable `NAME=value` lines) · `--path`→
`custom_path` (`model_id=__custom__`).

Prefix caching is on for every engine. `--cache off` turns it off for testing
and benchmarks (vLLM `--no-enable-prefix-caching`, EXL3
`--set vllm.enable_prefix_caching=false`, OVMS `--enable_prefix_caching false`,
llama.cpp `--no-cache-prompt`). A recipe with `prefix_caching: false` defaults
to off and says why in the warnings; `--cache on` forces it back on, with a
warning.

GPU defaulting follows the recipe: `gpus`/`tp`/dual-card kinds send `[0,1]`,
everything else `[0]`. `--gpus` accepts card numbers (`0`, `1`, `0,1`, `2,3` on a
four-card host; 0-7) or `both`. Multi-card launches warn when the cards sit on
different or narrow PCIe links (see `pcie_links` in `docs/api.md`).

A launch waits until the daemon reports the engine's `/v1/models` answering
(the same readiness probe the UI uses), showing the current load phase from
the engine log. `--timeout` bounds it; `--no-wait` skips it. Ctrl+C never
kills the engine — it detaches and prints the follow-up commands.

## Selecting things

Model arguments fuzzy-match: exact id → unique prefix → substring of id or
display name. Ambiguity lists candidates; a typo suggests close ids.
`b70 stop`, `b70 logs`, `b70 test`, `b70 env`, `b70 agent`, `b70 open <target>`
accept a server id (`<model>-<engine>-<port>`), a model name, or a bare port —
and default to the only running engine when there's exactly one.

## Headless / remote workflows

```sh
# same box, no display: everything just works
b70 serve
b70 launch qwen36-35b -e vllm
b70 test 'say ok'          # smoke-check the endpoint

# control from another machine: tunnel, then point the CLI at the local end
ssh -L 7570:127.0.0.1:7570 -L 8000:127.0.0.1:8000 b70host
export B70_API=http://127.0.0.1:7570 B70_TOKEN=<token from b70host:~/.local/state/b70-launcher/token-7570>
b70 status --json | jq .
```

`b70 open` asks the daemon for a single-use nonce UI URL (`POST /api/window`,
valid 30 s) and prints the matching `ssh -L` line when no display is present
(`--print` forces it). `b70 env` output is `eval`-able: `eval "$(b70 env)"` gives
local agents an OpenAI client.

## Environment

| var | meaning |
|---|---|
| `B70_API` | daemon base URL (default `http://127.0.0.1:7570`) |
| `B70_TOKEN` / `B70_TOKEN_FILE` | API token / token file path |
| `B70_LAUNCHER` | launcher.py (or wrapper) used for auto-start |
| `XDG_STATE_HOME` | moves the state dir (`token-<port>`, logs, servers-state) |
| `NO_COLOR` | disable ANSI color |
| `COLUMNS` | table width to fit (defaults to the terminal; unset when piped = no trimming) |

## Notes & edge cases

- `exl3` recipes need their artifact on disk even for `--dry-run` (the plan
  is meaningless without it); the daemon error is passed through verbatim.
- Auto-start finds `launcher.py` next to `cli.py` (installed layout), under
  `~/.local/share/b70-launcher/`, via `B70_LAUNCHER`, or `b70-launcher` on
  PATH.
- Launching a rid that's already up short-circuits with its endpoint — it
  never spawns a second engine.
- Downloads resume via `.part` files; Ctrl+C on `b70 download` detaches —
  the daemon keeps fetching. A repository whose file sizes cannot be read is
  refused before the download starts, because the free-space check would be
  meaningless without them.
- `b70 logs -f` polls with a byte offset, so a rotated or truncated log cannot
  replay lines you already saw.
- `b70 test` reports the answer and the reasoning text separately and exits 1
  when the engine returned reasoning only.
- `b70 stop` understands "starting" engines too: mid-load launches are
  cancelled cleanly.
- `b70 recipes-update <model>` without `-e` updates that model's engines one
  by one; it never applies the whole catalog. `-e` without a model is a usage
  error.
