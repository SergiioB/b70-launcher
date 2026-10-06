# recipes.json format

Reference for editing `recipes.json` and publishing recipe updates. Every field
listed here is observed in the shipped catalog or read by `launcher.py` —
unknown keys are ignored by the backend (the UI may still read some; see
"Inert fields" below).

## Top level

```json
{
  "catalog_ver": "YYYY-MM-DD",
  "engines": [ ... ],
  "models":  [ ... ]
}
```

`catalog_ver` is the fallback `recipe_ver` for any recipe that omits its own
and the baseline the update channel compares against. Keep it a date — version
comparison splits on non-digits (`"2026-10-05"` → `[2026,10,5]`), so any
`YYYY-MM-DD` or plain integer works.

## `engines[]` — engine metadata shown in the UI

```json
{"id": "openvino", "label": "OpenVINO (OVMS)", "tag": "OVMS",
 "tagline": "Easiest · Intel's official server", "note": "..."}
```

`id` must match the recipe keys used under each model. Engine ids in use:
`openvino`, `vllm`, `llamacpp`, `exl3`. `label`/`tagline`/`note` are display
text; `tag` is a short badge.

## `models[]`

```json
{
  "id": "qwen38-27b",
  "name": "Qwen3.8-27B",
  "arch": "Dense 27B",
  "blurb": "one-line pitch",
  "badge": "MAXIMUM INTELLIGENCE · 2× B70 TIERED",
  "tags": [{"t": "Top tier 180B", "c": "violet"}],
  "brand": "qwen",
  "subtitle": "Dense 27B • Balanced performance",
  "chips": ["LLM", "Instruct", "~16 GB"],
  "recommended_engine": "exl3",
  "recommendation": "one-paragraph reason",
  "recipes": {"<engine_id>": { ... }}
}
```

- `id`: stable slug — joins the update manifest, downloads, state, and user
  overrides. Never rename casually.
- `recommended_engine`: must be a key of `recipes`; decides the DEFAULT pill,
  the "Recommended for" hint, and the preselected engine.
- `badge`: truthy marks the model "Recommended" in the library filter.
- `brand`: picks the card icon (`qwen`, `nvidia`, `meta`/`llama`, `mistral`,
  `intel`; anything else gets the launcher logo).
- `chips`: shown under the card; a `~NN GB` chip doubles as the "Needs ~NN GB"
  size hint when no quant string matches.

## Recipe fields — all engines

| field | meaning |
|---|---|
| `kind` | selects the command builder: `ovms`, `vllm`, `vllm-mtp`, `vllm-tp2`, `vllm-arext`, `vllm-autoround`, `gguf`, `gguf-tiered`, `exl3` |
| `image` | container image (ignored when a `llama_bin` native path exists) |
| `ctx` | default context; clamped by `ctx_max` |
| `ctx_max` | hard ceiling; user ctx above it is clamped with a warning |
| `ctx_safe` | UI "Safe" preset; defaults to `ctx` when absent |
| `ctx_note` | shown in Recipe notes and as the ctx source label |
| `power` | watts hint — only used to warn when it exceeds the card cap; never applied |
| `perf` | throughput note shown in Recipe notes |
| `fixed_flags` | argv appended verbatim (for `gguf`: placed before `--cache-type-*` suppression check — include `--cache-type-k` to lock KV) |
| `search_name` | artifact match fallback (see Detection) |
| `model_path` / `source_model` / `gguf` | direct artifact path (supports `~`); checked on disk before scan results. `source_model` also sets the OVMS `--source_model` arg |
| `download` | see below |
| `recipe_ver` | per-recipe version for the update channel (`YYYY-MM-DD` or int) |
| `recipe_note` | copied into the published manifest as the update notice text |
| `recipe_recommended` | truthy marks the manifest entry "recommended" |

### `download`

```json
{"kind": "file|snapshot", "repo": "org/repo", "revision": "abc123",
 "name": "file.gguf", "extra_files": ["mmproj-*.gguf"],
 "search": "...", "match": "qwen3.8-27b" | ["alt1", "alt2"],
 "prefer": ["w4a16", "gptq"], "quant": "GPTQ-INT4 sym G128",
 "verify": ["config.json"], "note": "...", "hf_search": "..."}
```

- `kind=file`: a single file (`name`) + optional `extra_files` sidecars, fetched
  into `models_dir`; matched against GGUF basenames from the scan.
- `kind=snapshot`: a whole repo tree into `ovms_repo` (openvino) or
  `models_dir` (everything else); matched against detected model directories.
- `repo`: automatic download is only offered when it is in the
  `HF_APPROVED_REPOS` allow-list in `launcher.py`; otherwise the UI falls back
  to manual instructions (`hf_search` link, else `search` text).
- `match` / `search` / `name` accept a string or a list of alternates;
  `prefer` narrows same-family hits to the right quant/format.
- `quant`, `note`: display only. `verify` is inert metadata — nothing reads it.
- `doc`/`doc_url`/`docs` (recipe or model level): the UI turns them into an
  "Open recipe docs" link only when neither `repo` nor `hf_search`/`search`
  exists.

### Detection order

1. Direct recipe path (`gguf`, `model_path`, `source_model`) that exists on
   disk wins immediately.
2. Exact `download.name` basename in the scan pool.
3. Same-family match: normalized `match` → `search` → `name` → `search_name` →
   repo basename, filtered by `prefer` and (for snapshots) by detected kind.
   `mmproj*` and `*dflash*`/`*draft*` basenames never satisfy a recipe unless
   the recipe itself asks for them.

Detected `config.json` context (`max_position_embeddings` etc., incl.
`text_config`) overrides `ctx` up to `ctx_max`.

## Per-kind fields

### `kind: ovms` (engine `openvino`)

`source_model` (repo id or on-disk dir), `tool_parser`, `reasoning_parser`,
`cim_long_ctx` (adds `--cache_interval_multiplier N` when ctx > 20480).
Mounts the artifact dir at `/models` (file) or `/models/model` (dir); OVMS
manages KV internally — `ctx` is advisory.

### `kind: vllm` family (engine `vllm`)

Common: `model_path`, `dtype` (default `bfloat16`), `kv` (`"fp8"` adds
`--kv-cache-dtype fp8`), `spec_tokens` (enables MTP `--speculative-config`;
UI sends `mtp:false` to disable), `speculative` (string draft model id, generic
kind only), `gpus`/`tp` (dual-card intent — see Caveats).

- `vllm` (generic): plain `docker run` + `vllm serve` flags; mounts the HF
  cache, `ONEAPI_DEVICE_SELECTOR=level_zero:<sel>`; `--tensor-parallel-size`
  added when `gpus` has two entries.
- `vllm-mtp`: mounts `patches/patch_mtp_nightly.py` + `patch_mtp_boundary.py`,
  runs them before `vllm serve` (GPTQ, fp8 KV, prefix caching on). Requires
  the artifact on disk.
- `vllm-arext`: AutoRound W4A16 path; also mounts
  `patch_champion_stack_overlay.py`; prefix caching intentionally off; requires
  the artifact on disk.
- `vllm-tp2`: dual-card tensor parallel; mounts `patch_vllm_worker_affinity.py`,
  `--cap-add SYS_PTRACE`, `--ipc=host`, oneCCL threshold env; `tp` defaults 2.
- `vllm-autoround`: generic path + `--enforce-eager` via `fixed_flags` and the
  FP16 crash-guard warning.

### `kind: gguf` / `gguf-tiered` (engine `llamacpp`)

`gguf` (default path), `llama_bin` (recipe- or settings-level native binary;
`~` expanded — when it exists and `use_docker` is false, no container is used),
`kv` (`q8_0`, `q5_0/q4_1`, `q8_0/q4_1`, `f16`; UI override maps to
`--cache-type-k/-v`), `tensor_split`, `split_mode`, `offload_tensors` (`-ot`),
`draft_model` (host path or bare basename resolved through the last scan),
`draft_device` (default `SYCL1`), `spec_tokens`, `spec_p_min`.

Native runs get `ONEAPI_DEVICE_SELECTOR`/`ZE_AFFINITY_MASK`/`SYCL_*` env, all
`/opt/intel/oneapi/*/latest/lib` dirs prepended to `LD_LIBRARY_PATH`, and bind
`127.0.0.1`. `gguf-tiered` additionally sets `LLAMA_ATTN_ROT_DISABLE=1` and
disables immediate command lists. Every llama.cpp launch gets `-ngl 99`,
`--flash-attn on`, `--metrics`.

### `kind: exl3` (engine `exl3`)

`model_path` (required on disk), `serve_config` (passed as first arg to the
image entrypoint), `docker_sock` (isolated daemon socket — auto-started via
`sudo` containerd/dockerd when down), `inner_port` (container-side port,
default 8100). Linux only; fp8 KV; `gpu_memory_utilization` 0.90 (<128K ctx)
or 0.94 (≥128K).

## Custom artifacts (`model_id: "__custom__"`)

`Browse… / Set Custom Path` in the UI posts `custom_path` with
`model_id="__custom__"`. Rules:

- The resolved path must be inside a configured scan root and exist.
- Files: only `.gguf` (engine must be `llamacpp`). Directories are sniffed:
  `openvino_language_model.*` or xml+bin → `openvino`; else `vllm`, or `exl3`
  when `quantization_config.json` has `quant_method=exl3`. Picking the wrong
  engine fails the build with a "pick the matching engine" error.
- The recipe is a generic template (image, ctx 32768/20480/65536, power) plus
  `ctx` from the artifact's `config.json` when present. There is no
  verified-recipe tuning — the launch warning says so.

## Recipe update channel (end to end)

1. **Publish.** `packaging/release.py` emits `version.json` (with
   `recipes_manifest_url`), `recipes-manifest.json`, and `recipes.json` into
   `release/`. The manifest is data-only:
   `{catalog_ver, recipes_url, recipes: {"<model_id>:<engine>": {ver, note?, recommended?}}}`
   where `ver` = `recipe_ver` or `catalog_ver`, `note` = `recipe_note`,
   `recommended` = `recipe_recommended`.
2. **Notice.** At startup the launcher fetches `version.json`
   (`B70_UPDATE_URL`, default `https://xecores.com/downloads/version.json`),
   then the manifest (`recipes_manifest_url`, else same directory). Each entry
   newer than the local `recipe_ver`/`catalog_ver` — or naming a model/engine
   absent locally — becomes a notice: card badge, "Get updated recipe" box, and
   a "New in the catalog" strip for unknown models.
3. **Apply.** `POST /api/recipes/update` fetches `recipes_url` (https, or
   localhost for testing) and overlays the document:
   - unknown `id` → whole model appended;
   - known `id` → whitelisted model fields replaced (`recommended_engine`,
     `recommendation`, `badge`, `blurb`, `subtitle`, `tags`, `chips`, `brand`)
     and each published engine recipe replaced **wholesale** — a recipe is
     all-or-nothing, never field-merged;
   - recipes missing `recipe_ver` are stamped with the applied `catalog_ver`;
   - `recipe_overrides` from `settings-override.json` are re-applied last and
     always win.
4. **Persist.** Applied models go to
   `${XDG_STATE_HOME:-~/.local/state}/b70-launcher/recipes-remote.json`
   (`{catalog_ver, applied_at, payload:{models}}`, merged by model id, newer
   `catalog_ver` kept) and are re-loaded at startup before notices are
   computed. The installed `recipes.json` is never modified. A disk rescan
   follows each apply.

## Caveats / inert fields

- The UI never sends `gpus` (it hardcodes `slots:1`, `gpu0:0`), `mtp`,
  `use_docker`, `extra`, `extra_env`, or `models_dir`. Dual-GPU recipes rely on
  recipe-level `tp`/`tensor_split`; for `gguf` kinds the `--device` list and
  `ZE_AFFINITY_MASK` still derive from the single-GPU `cfg.gpus` — exercise the
  dry run before trusting a multi-GPU recipe from the UI.
- `nemotron-35`'s `vllm` recipe carries a `speculative` **dict** and a
  `patches` list — neither is consumed by this `launcher.py` (generic `vllm`
  only reads `speculative` as a string when `spec_tokens` is set). The UI still
  shows "Speculative decoding (MTP) enabled" for it.
- `settings.json` `power_modes`, `vllm_kv_options`, `llamacpp_kv_options`, and
  `terminal_linux` are shipped but unused — the UI hardcodes the power/KV
  choices and `open_harness` probes terminals itself.
