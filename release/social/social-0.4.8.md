# B70 Launcher 0.4.8 — Social Drafts (FINAL)

Screenshots (all retaken on the simplified UI):
- `release/screenshots/fresh-first-run.png` — first run: catalog, honest "Needs ~19 GB", compact get-model
- `release/screenshots/step1-models.png` — library with detected artifacts + engine/config/launch
- `release/screenshots/step2-config.png` — config column for a missing model (Muse-Glimmer)
- `release/screenshots/step3-launch.png` — dry-run command modal (exact llama-server command + env)
- `release/screenshots/shot2-notice.png` — recipe-update notice (older UI, notice design unchanged)

Live URLs:
- https://xecores.com/match (one-command install + checksum)
- https://github.com/SergiioB/b70-launcher (source, MIT)
- https://github.com/SergiioB/b70-launcher/releases/tag/v0.4.8 (release + assets)

---

## X (Twitter)

### Main post (attach step1-models.png)

B70 Launcher 0.4.8 is out — open-source local desktop UI for running LLMs on Intel Arc Pro B70.

```
curl -fsSL https://xecores.com/downloads/get.sh | sh
```

Model library → engine pick (vLLM XPU / llama.cpp SYCL / OpenVINO / EXL3) → launch & open in your chat app. MIT: github.com/SergiioB/b70-launcher

### Reply 1 (attach step3-launch.png)

Nothing is hidden: dry-run shows the exact llama-server flags, KV-cache dtypes, SYCL env vars and mounts before anything runs. Native servers survive app restarts — re-adopted, stoppable from the UI.

### Reply 2 (attach fresh-first-run.png)

Rootless install (~/.local only, XDG-aware, zero hardcoded paths). Honest by design: real artifact sizes or "not on disk", never fake numbers. Recipe updates arrive over a channel and apply as state overlays — your catalog is never rewritten.

---

## LinkedIn

I've just released B70 Launcher 0.4.8 — an open-source (MIT) local desktop launcher for LLM inference on the Intel Arc Pro B70. Source: github.com/SergiioB/b70-launcher — install: xecores.com/match

The problem it solves: getting a validated model + engine running on Battlemage hardware normally means juggling Docker flags, SYCL environment variables, KV-cache dtypes and HF repos by hand. The launcher reduces that to three steps — pick a model from the library (honest "on disk / not found" status and real sizes), pick an engine recipe (vLLM XPU, llama.cpp SYCL, OpenVINO OVMS, EXL3), launch, and open it in Pi, Factory Droid, OMP or Open WebUI behind an OpenAI-compatible endpoint.

Install is one command:

curl -fsSL https://xecores.com/downloads/get.sh | sh

It resolves the current release, verifies the SHA256 checksum before extracting, and installs rootless under ~/.local — XDG-aware, no hardcoded paths, state preserved across reinstalls. For the cautious: the script and the Python source (stdlib only, zero pip dependencies) are on GitHub to read first.

What's in 0.4.8:

- Recipe update channel — validated launch recipes are versioned; when a better one ships, installed launchers show a badge and a one-click "Get updated recipe" notice. Applies to your state dir; your own overrides always win.
- Custom artifacts — point it at any GGUF / OpenVINO IR / safetensors tree under your scan roots; it sniffs the format, matches the right engine, refuses honestly if wrong.
- Native llama.cpp lifecycle — oneAPI runtime resolution, correct flags per build, running servers re-adopted after restart instead of orphaned.
- Hardened local API — session token via cookie bootstrap (0600 token file), recipe-overlay field allow-listing, host/origin checks, loopback only.
- Transparency — dry-run shows the exact command + environment before anything launches; per-card watts/VRAM and load timing are live in the UI.

Target platform: dual Intel Arc Pro B70 (2×32GB). No telemetry; the only network calls are the update check and downloads you explicitly request.

[Screenshot: step1-models.png hero; dry-run or fresh-run as second]

#IntelArc #Battlemage #LocalAI #LLM #OpenSource #OpenVINO #llamacpp
