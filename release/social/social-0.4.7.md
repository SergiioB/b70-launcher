# B70 Launcher 0.4.7 — Social Drafts

Screenshots (attach in this order):
- `release/screenshots/step1-models.png` — 3-step flow: model library, honest on-disk sizes, update banner
- `release/screenshots/step2-config.png` — engine config + the "better recipe available" notice
- `release/screenshots/step3-launch.png` — dry-run command preview showing the exact llama-server command + env
- `release/screenshots/shot1-library.png` — alternate hero (1500px)
- `release/screenshots/shot2-notice.png` — alternate notice close-up
- `release/screenshots/shot3-dryrun.png` — alternate dry-run

Live URLs:
- https://xecores.com/match (install + checksum)
- https://xecores.com/shots/launcher/step2-config.png (hotlinkable)

---

## X (Twitter)

### Main post (attach step2-config.png)

B70 Launcher 0.4.7 is out — a local desktop UI for running LLMs on Intel Arc Pro B70.

Model library → engine pick (vLLM XPU / llama.cpp SYCL / OpenVINO / EXL3) → launch & open in your chat app. Recipes update themselves: when a better validated recipe ships, the launcher tells you and applies it without touching your install.

xecores.com/match

### Reply 1 (attach step3-launch.png)

Nothing is hidden: "Preview command" shows the exact llama-server flags, KV cache dtypes, SYCL env vars and mount layout before anything runs. Native llama.cpp servers survive app restarts — re-adopted on relaunch, stoppable from the UI.

### Reply 2 (attach step1-models.png)

Fresh install is rootless: ~/.local only, XDG-aware, no hardcoded paths. Point it at ~/models or let it scan — it reports real artifact sizes or "not found", never fake numbers.

---

## LinkedIn

Releasing B70 Launcher 0.4.7 today — a local desktop launcher for LLM inference on the Intel Arc Pro B70, now live at xecores.com.

The problem it solves: getting a validated model + engine running on Battlemage hardware normally means juggling Docker flags, SYCL environment variables, KV-cache dtypes and HF repos by hand. The launcher turns that into three steps — pick a model from the library (with honest "on disk / not found" status and real sizes), pick an engine (vLLM XPU, llama.cpp SYCL, OpenVINO OVMS, EXL3), launch and open it in Pi, Factory Droid, OMP or Open WebUI behind an OpenAI-compatible endpoint.

What's new in 0.4.7:

- Recipe update channel. Validated launch recipes are versioned on the site. When a better recipe ships — a KV-cache change, a fixed flag, a new model — installed launchers show a badge and a one-click "Get updated recipe" notice. The update applies to your state directory, never modifies the installed catalog, and your own overrides always win.
- Custom artifacts. Point the launcher at any GGUF / OpenVINO IR / safetensors tree under your scan roots; it sniffs the format, matches it to the right engine and refuses honestly if you pick the wrong one.
- Native llama.cpp lifecycle. oneAPI runtime libraries are resolved automatically, new llama.cpp builds get correct flash-attention flags, and a running llama-server is re-adopted after an app restart — visible and stoppable, not orphaned.
- Transparency by default. Dry-run shows the exact command + environment before anything launches. Per-card watts, VRAM and token accounting are live in the UI.

The install is rootless (~/.local only, XDG-aware, no hardcoded paths) and the source is inspectable before you run it — the install page verifies the SHA256 checksum first. No telemetry; the only network calls are the update check and downloads you explicitly request.

Dual Intel Arc Pro B70 (2×32GB) is the target platform. Download + checksum-verified install instructions: xecores.com/match

[Screenshot: step2-config.png — the recipe-update notice; optionally step1 + step3 as a comment]

#IntelArc #Battlemage #LocalAI #LLM #OpenSource
