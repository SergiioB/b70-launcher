# b70 — headless CLI

`b70` (`cli.py`) is the command-line face of B70 Launcher. It talks to the
same daemon the windowed app uses (`launcher.py`, `127.0.0.1:7570` by
default), so every recipe the UI can launch — `ovms`, `vllm`, `vllm-mtp`,
`vllm-tp2`, `vllm-arext`, `vllm-autoround`, `gguf`, `gguf-tiered`, `exl3` —
launches identically from a terminal with no display at all.

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
  every command. `NO_COLOR` is honored; colors auto-disable off a TTY.

## Commands

| command | does |
|---|---|
| `b70 serve` | start the daemon (idempotent; `--fg` foreground, `--kill` free the port) |
| `b70 down` | stop the daemon (`--stop-engines` also stops engines) |
| `b70 list` | every model × engine recipe: kind, ctx, detected artifact, live mark (`-e` filters) |
| `b70 show <model>` | model detail: recipes, ctx ceilings, detected path, launch/download hints |
| `b70 launch <model>` | launch a recipe and wait for the OpenAI endpoint |
| `b70 launch` | interactive picker on a TTY (model → engine, detected/recommended marks) |
| `b70 launch <path>` | serve any artifact under a scan root (`__custom__` recipe) |
| `b70 stop [rid]` | stop an engine (`--all` for everything; no arg = the only running one) |
| `b70 status` | GPUs (draw/cap/temp/clock/VRAM) + engines (status, load time, tok/s, tokens) |
| `b70 monitor` | live dashboard on a TTY; one snapshot when piped (`-f` streams, `--once` single) |
| `b70 logs <rid>` | engine log tail (`-f` follow, `-n` lines) |
| `b70 test [rid|port] [prompt]` | smoke prompt end-to-end; latency + approx tok/s (prompt-only arg works with a single running engine) |
| `b70 download <model> -e <eng>` | fetch the recipe artifact; progress bar to done |
| `b70 open [ui|rid|port]` | open UI or engine endpoint; prints URL + SSH hint when headless |
| `b70 env [rid]` | `export OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL_NAME` |
| `b70 doctor` | render nodes, B70 identification, blockers, scan roots |
| `b70 scan` | detected artifacts + per-recipe matches (`--roots a,b` to set roots) |
| `b70 recipes-update [model]` | apply the published recipe catalog overlay |
| `b70 version` | CLI / daemon / catalog versions |
| `b70 completion bash` | bash completion script |

Aliases: `ls`, `recipes` → `list`; `ps` → `status`; `ui` → `open ui`.

## `b70 launch` — every knob maps to the recipe API

```sh
b70 launch qwen38-27b                      # recommended engine (exl3 here)
b70 launch qwen36-35b -e vllm              # explicit engine
b70 launch qwen36-35b -e llamacpp --ctx 65536 --port 8080 --kv q5_0/q4_1
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
`--slots`→`slots` · `--kv`→`kv` · `--gpus`→`gpus` · `--power`→`power` ·
`--mtp/--no-mtp`→`mtp` · `--docker/--native`→`use_docker` · `--extra`→`extra`
· `-E/--env`→`extra_env` (repeatable `NAME=value` lines) · `--path`→
`custom_path` (`model_id=__custom__`).

GPU defaulting follows the recipe: `gpus`/`tp`/dual-card kinds send `[0,1]`,
everything else `[0]`. `--gpus` accepts `0`, `1`, `0,1`, `both`.

A launch waits until the daemon reports the engine's `/v1/models` answering
(the same readiness probe the UI uses), showing the current load phase from
the engine log. `--timeout` bounds it; `--no-wait` skips it. Ctrl+C never
kills the engine — it detaches and prints the follow-up commands.

## Selecting things

Model arguments fuzzy-match: exact id → unique prefix → substring of id or
display name. Ambiguity lists candidates; a typo suggests close ids.
`b70 stop`, `b70 logs`, `b70 test`, `b70 env`, `b70 open <target>` accept a
server id (`<model>-<engine>-<port>`), a model name, or a bare port — and
default to the only running engine when there's exactly one.

## Headless / remote workflows

```sh
# same box, no display: everything just works
b70 serve
b70 launch qwen36-35b -e vllm
b70 test 'say ok'          # smoke-check the endpoint

# control from another machine: tunnel, then point the CLI at the local end
ssh -L 7570:127.0.0.1:7570 -L 8000:127.0.0.1:8000 b70host
export B70_API=http://127.0.0.1:7570 B70_TOKEN=<token from b70host:~/.local/state/b70-launcher/token>
b70 status --json | jq .
```

`b70 open` prints the tokenized UI URL plus the matching `ssh -L` line when
no display is present (`--print` forces it). `b70 env` output is
`eval`-able: `eval "$(b70 env)"` gives local agents an OpenAI client.

## Environment

| var | meaning |
|---|---|
| `B70_API` | daemon base URL (default `http://127.0.0.1:7570`) |
| `B70_TOKEN` / `B70_TOKEN_FILE` | API token / token file path |
| `B70_LAUNCHER` | launcher.py (or wrapper) used for auto-start |
| `XDG_STATE_HOME` | moves the state dir (token, logs, servers-state) |
| `NO_COLOR` | disable ANSI color |

## Notes & edge cases

- `exl3` recipes need their artifact on disk even for `--dry-run` (the plan
  is meaningless without it); the daemon error is passed through verbatim.
- Auto-start finds `launcher.py` next to `cli.py` (installed layout), under
  `~/.local/share/b70-launcher/`, via `B70_LAUNCHER`, or `b70-launcher` on
  PATH.
- Launching a rid that's already up short-circuits with its endpoint — it
  never spawns a second engine.
- Downloads resume via `.part` files; Ctrl+C on `b70 download` detaches —
  the daemon keeps fetching.
- `b70 stop` understands "starting" engines too: mid-load launches are
  cancelled cleanly.
