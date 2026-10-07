# B70 Launcher 0.4.8 (Linux source)

## What is new in 0.4.8

- **Security hardening.** The session token is no longer embedded in an unauthenticated `GET /` response: it ships in `~/.local/state/b70-launcher/token` (mode 0600), reaches the window once via a `?token=` URL that is immediately swapped for a session cookie, and every GET endpoint now requires it (POSTs still require the `X-Launcher-Token` header). Remote recipe updates can no longer overwrite executable fields (`image`, `llama_bin`, `docker_sock`, `fixed_flags`, paths): overlays are allow-listed to metadata/tuning fields, model IDs and engine names are validated, and manifest URLs are restricted to HTTPS or loopback. Harness commands and generated YAML now quote/validate model names, recipe-supplied environment variables cannot set `LD_*`, `PATH`, `HOME`, `PYTHON*` and similar startup hooks, and state files are written atomically with tighter permissions.
- **Health checks off the lock.** Engine `/v1/models` probes no longer run while holding the global state lock, so a stalled engine cannot freeze Stop, logs, or state polling.
- **Lower idle overhead.** Telemetry polling uses lightweight `/api/power` and `/api/servers` endpoints; `docker stats` only runs while a server is genuinely live, VRAM reads are cached, and the UI reschedules its timers instead of queuing requests. HTTP/1.1 keep-alive removes per-poll reconnects.
- **Accessibility pass.** Real buttons and radio groups for model/engine cards, roving tabindex with arrow-key navigation, modal dialogs with `aria-modal`, focus trapping and restore, live regions for toasts/status/output, and `prefers-reduced-motion` support.
- **GPU selector.** Advanced settings can target card 0, card 1, or both on dual-B70 systems (the backend `gpus` field is now actually sent).
- **Fixed `vllm-arext` on fresh installs.** The 0.4.7 archive omitted `patches/patch_champion_stack_overlay.py`; it is included now. llama.cpp container recipes moved to the published `server-intel` image (`server-sycl` no longer exists upstream).
- **Installer improvements.** Rootless, idempotent installer with prerequisite checks, PATH warning, hicolor icons, a matching `StartupWMClass`, an `uninstall.sh` (`--purge` also removes state), and deterministic release archives with a `--verify` mode.
- **Engine-aware KV cache options.** llama.cpp recipes offer the host-validated `q5_0/q4_1` asymmetric KV pair, plus `q8_0`, `q8_0/q4_1`, and `f16`.

## What is new in 0.4.7

- **Recipe update channel.** The launcher now checks `xecores.com/downloads/recipes-manifest.json` alongside the release check. When a published recipe revision is newer than your local one, the model card gets a badge and the launch step shows a one-click "Get updated recipe" notice — applied to your state directory, never to the installed `recipes.json`, and your own recipe overrides always win. Brand-new catalog models appear under "New in the catalog" with an Add action.
- **Custom artifacts work.** "Browse… / Set Custom Path" now actually drives the launch: the picked file or directory is sniffed (GGUF, OpenVINO IR, HF/exl3 tree), confined to your configured scan roots, matched to a compatible engine, and launched from a generic engine template — with a clear refusal if you pick the wrong engine.
- **Portable defaults.** Shipped settings use `~/models`, `~/Downloads`, and `~/.local/share/b70-exl3`. Machine-specific paths (scan roots, `llama_bin`, per-recipe binaries, EXL3 data root) live in `settings-override.json` under your state directory, so the same archive installs cleanly for any user.
- **Native llama.cpp improvements.** `llama_bin` supports `~`, Intel oneAPI runtime libraries are added to `LD_LIBRARY_PATH` automatically when a `llama-server` binary needs them, and every llama.cpp launch (native or container) gets `--flash-attn on`.
- **Native engines survive restarts.** A running `llama-server` is re-adopted on launcher start (shown as "running (adopted)") and Stop kills it by PID — closing the app no longer orphans a native engine from the UI.
- **Better failure visibility.** Endpoint health-check failures are logged once per server instead of being silently retried forever, and the update check reports why it failed.
- **Reasoning-model test output.** The quick inference test now shows `reasoning_content` when a model thinks without producing visible `content`, uses 512 max tokens, and notes when generation hit the cap.

## What is new in 0.4.4 - 0.4.6

- **EXL3 XPU engine.** A fourth recipe kind runs trellis-compressed EXL3 weights on an isolated dockerd (`/run/b70-exl3-docker.sock`, data root `~/.local/share/b70-exl3`, auto-started on launch — needs passwordless sudo). It is the only single-card route to the full 262K context.
- **Verified download help.** Recipes pinned to an approved repository show copyable `curl` / `hf download` commands plus sidecar files (`extra_files`, e.g. `mmproj`); broken hub exports are never offered.
- **Honest artifact status.** The model-file field now reports the real detected size (GiB) or "Not found on disk" — it no longer claims "Found on disk (180 GB)" unconditionally, and the launch summary matches.
- **Draft sidecars never satisfy a recipe.** DFlash/MTP draft GGUFs are excluded from same-family matching, so a 1 GB draft can no longer be detected as a 20 GB model.
- **Real sizes in model chips.** Placeholder artifact sizes were replaced with measured values, and catalog-questionable throughput claims were aligned to the published benchmark catalog.
- **Recommendation tracks the recipe.** The engine DEFAULT pill, "Recommended for" hint, and launch summary now follow each model's `recommended_engine`, and single-GPU systems stop seeing dual-card copy.
- **Empty states.** First run with zero detected artifacts and an empty model search both render guidance instead of blank lists.
- **Narrow layouts.** Below ~1100 px the three steps stack vertically and the telemetry row wraps; Escape closes every modal.

Inspect local Arc Pro B70 hardware, find cookbook model artifacts, review the exact engine command, and optionally launch an inference server. This is a local desktop UI backed by Python stdlib; it does **not** bundle an engine, model weights, Docker, drivers, or a browser.

B70 Launcher is open source under the [MIT License](LICENSE); the repository lives at [github.com/SergiioB/b70-launcher](https://github.com/SergiioB/b70-launcher). Third-party engine and vendor names/logos shown in the UI remain trademarks of their respective owners. At startup the launcher contacts xecores.com once for the release check and the recipe manifest; the UI itself makes no other network requests until the user selects a model download, applies a recipe update, or a launched engine pulls its configured image/model. Verify the release manifest and checksum before sharing.

## Start a server

1. Choose a model. A model is selected for you; use search or the **All Models** / **On Disk** filters to find another, and **Browse…** to point at a local artifact under a scan root.
2. Check the engine, model-file status, and context size. Open **Advanced settings** only if you need to change recipe defaults.
3. Select **▶ Launch Model**. The button becomes **■ Stop Server** while an engine is tracked; endpoint status, a quick test prompt, and live engine logs sit in the same column.

Unfinished steps stay dimmed and cannot take focus. On wide screens, completed columns remain visible; on smaller screens, they collapse into editable headings. **Preview Command (Dry Run)** — under the ▼ split button — prints the exact command, environment, and warnings without starting an engine. Recipe notes for the selected engine render below the context controls; configured scan roots are not shown in the UI (see `settings-override.json` below).

## What is new in 0.4.0

- **Native app window.** The UI opens in a standalone WebKitGTK window (`webwindow.py`, spawned with the system `python3`) instead of a Chromium tab. Closing the window quits the app cleanly: downloads are cancelled and per-session token usage is stored. `--browser` and `--no-open` keep the old behavior; without `python3-gi` the launcher falls back to a browser app window automatically.
- **Instant first paint.** The server binds and the window opens before any filesystem scan; the model scan runs in the background (depth-limited, time-budgeted `os.scandir` walk) and detection chips update when it lands. Scan roots are portable defaults from `settings.json`, overridable per machine via `settings-override.json` in the state directory or `POST /api/settings` — never written back to the install.
- **Real token accounting.** While an engine runs, the launcher polls the engine's own Prometheus `/metrics` on its API port (`vllm:prompt_tokens_total` / `vllm:generation_tokens_total`, `llamacpp:prompt_tokens_total` / `llamacpp:tokens_predicted_total` — llama.cpp is now launched with `--metrics`), with engine-log parsing as a fallback. The running panel shows prompt tokens, completion tokens, output tok/s, engine cpu/mem, per-card live watts (xe hwmon energy counters) and VRAM.
- **Usage stored on exit.** Session records (model, engine, port, started/ended, prompt/completion tokens, requests, peak tok/s) are appended to `~/.local/state/b70-launcher/usage-history.json` when a server stops and when the app exits. `GET /api/usage` returns lifetime totals plus the last 50 sessions; the current UI does not render them.
- **Engines survive app restarts.** Tracked containers are re-adopted on startup (`servers-state.json` + `docker inspect`), so closing the launcher no longer orphans a running engine from the UI. Container status comes from Docker itself, not the short-lived `docker run -d` client process.
- **Hardware-aware default.** The preflight builds a profile (B70 count, per-card VRAM, power cap) and pre-selects the best default model + engine for the detected hardware, with the reason shown in the UI.

## Install (current user only)

One command (downloads, verifies SHA256, extracts, installs):

```sh
curl -fsSL https://xecores.com/downloads/get.sh | sh
```

Or inspect-first, manually:

1. Verify the archive with the separately supplied SHA256 checksum: `sha256sum -c b70-launcher-0.4.8-linux-source.tar.gz.sha256`.
2. Extract it: `tar -xzf b70-launcher-0.4.8-linux-source.tar.gz`.
3. Inspect `launcher.py`, `webwindow.py`, `recipes.json`, `settings.json`, and `packaging/install.sh`; run `sh b70-launcher-0.4.8-linux-source/packaging/install.sh` if satisfied. Installer copies the inspectable source to `~/.local/share/b70-launcher`, adds `~/.local/bin/b70-launcher` and a desktop entry in `~/.local/share/applications` (XDG_DATA_HOME is honored for the app and entry). No root access or global system changes. Start from your application menu or run `~/.local/bin/b70-launcher`.

Python 3.9+ is required for the UI; `python3-gi` with a WebKitGTK typelib (`gir1.2-webkit2-4.1`, `gir1.2-webkit2-4.0`, or GTK4 `webkit-6.0`) enables the native app window; without it the UI opens in your default browser. To actually launch a GPU engine, install a compatible Linux Intel GPU driver, accessible DRM render nodes, Docker daemon/CLI with Intel GPU support and permission to use it, adequate disk/VRAM, and a valid model artifact. Docker access is effectively root-equivalent; do not grant it to untrusted users. Docker is required by the launch preflight even for the native `llama_bin` path — no Docker CLI and accessible local socket, no launch. The UI shows render-node identification, per-card VRAM/power and basic blockers before launch. A preflight success does **not** certify an engine, quantization, context size, power budget, or card topology. No engine starts at installation or first opening. Container images may be pulled when launching; review recipes and image provenance first. Some recipe images use mutable tags.

The installer does not overwrite system files but does replace an existing user-level B70 Launcher installation. Model downloads write into `~/models` or `~/models/ovms-repo`; inspect repository and model license before requesting them. Automatic download is limited to a fixed allow-list of recipe-pinned Hugging Face repositories (`HF_APPROVED_REPOS` in `launcher.py`) after an explicit in-app confirmation; other recipes require manual artifact acquisition. A pinned repository is **not** a content signature: upstream `main` may change, and equal-sized existing files are reused without cryptographic verification; mismatched/unknown existing artifacts are never overwritten automatically. Do not treat model weights as trusted code; some engines may execute model-defined custom code. No automatic power cap is applied and no generic power-cap command is supplied; identify a board-safe cap and exact GPU node yourself if needed.

## Support boundary

- **B70:** The UI and command templates target Arc Pro B70 on Linux. Recipes are model- and engine-specific; the listed Qwen3.8-27B INT4-OV (GDN8) OpenVINO IR has a validated single-card OVMS recipe. Do not infer support for other OpenVINO exports.
- **B50/B60/B65 and other Intel GPUs:** Not validated by this launcher. The preflight deliberately blocks real launches when it cannot identify accessible B70 render nodes; use dry-run to inspect commands. Do not infer compatibility from shared Intel branding.
- **Card count:** Recipes with an explicit dual-B70 topology can use both identified, accessible B70 cards. Other recipes follow their own engine/device configuration; a dry run previews the exact command without launching it. More than two GPUs are unsupported.
- **Windows:** Python source may open the UI, but GPU launch is intentionally blocked. WSL2/Docker passthrough is not validated; the Linux desktop archive/installer is not a Windows installer.

`settings.json` lists portable defaults; per-machine values live in `${XDG_STATE_HOME:-~/.local/state}/b70-launcher/settings-override.json` — `scan_dirs`, `models_dir`, `ovms_repo`, `llama_bin`, `exl3_data_root`, and `recipe_overrides` (`{"<model_id>": {"<engine>": {field: value}}}`, re-applied over every recipe update). Only `scan_dirs` is writable through the API (`POST /api/settings`, applied on the next scan, no restart); edit the file for the rest. The same state directory holds `servers-state.json` (engine re-adoption), `usage-history.json`, `recipes-remote.json` (applied recipe updates), and `logs/` (newest 20 kept).

The launcher API binds `127.0.0.1:7570` (override with `--port`); launched engines publish on `127.0.0.1` only (default port 8000). POST requests require a per-process `X-Launcher-Token` injected into the served page plus Host/Origin checks; GETs are read-only but tokenless. Do not expose the port through a proxy, do not browse via a `localhost` Host name (the check is literal `127.0.0.1`), and do not share a desktop session with untrusted users. Quit the UI by closing the window (or Ctrl-C when started in a terminal): usage is stored on exit and running engines stay alive and are re-adopted on the next start — use the app's Stop button to remove them.

CLI flags: `--port`, `--no-open`, `--browser` (also `B70_LAUNCHER_WINDOW=browser`), `--kill`/`--force` to free the UI port (also attempted automatically when the bind fails). `B70_UPDATE_URL` overrides the update-check URL for testing — point it at a local `version.json` to exercise the release and recipe-update channel end to end. Endpoint and recipe references for contributors live in `docs/api.md` and `docs/recipe-format.md`.

## Repackage from checked-out sources

Run `python3 packaging/release.py` only when cutting a new version; it overwrites the archive and checksum for its configured version. The source bundle excludes the old PyInstaller `dist/` binary: that ELF uses glibc and bundled Python shared libraries and has no demonstrated portable runtime/ABI support. Its presence in the working tree is not a release. The release is inspectable source, not self-contained executable; Python and runtime dependencies above remain prerequisites.
