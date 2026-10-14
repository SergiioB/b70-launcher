# recipes.json format

Reference for editing `recipes.json` and publishing recipe updates. Every field
listed here is observed in the shipped catalog or read by the `b70launcher/`
package — unknown keys are ignored by the backend (the UI may still read some;
see "Inert fields" below).

## Add a recipe

1. **Copy an existing entry** in `recipes.json` matching the target model architecture and engine (`vllm`, `llamacpp`, `openvino`, or `exl3`).
2. **Fill fields** according to the schema below (weights, prompt template, context window, arguments, and recommended hardware).
3. **Run recipe tests** to verify catalog integrity:
   ```bash
   python3 -m unittest tests.test_recipes
   ```
4. **Open a PR** with your changes to `recipes.json`.

Alternatively, request a new recipe via the [Recipe Request issue form](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/new?template=recipe-request.yml).

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

`id` must match the recipe keys used under each model. Engine keys in use:
`openvino`, `vllm`, `llamacpp`, `exl3`. `label` is the engine button text; `tag`,
`tagline`, and `note` are shipped with the catalog but are not rendered by the
current UI.

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
- `recommended_engine`: must be a key of `recipes`; the engine the UI preselects
  when that model is chosen.
- `badge`: truthy sorts the model above unbaged models in the library list
  (after on-disk models).
- `chips`: joined into the library search text.
- `recommendation`: shown in Recipe notes when the recipe has no `recipe_note`.
- `brand`: shipped and allow-listed for recipe updates but not read by the
  current UI (the app header uses `web/assets/b70-launcher.svg`; the other
  engine/model logo files in that directory are unused).
- `subtitle` and `tags`: shipped in the catalog and allow-listed for recipe
  updates, but not rendered by the current UI.

## Recipe fields — all engines

| field | meaning |
|---|---|
| `kind` | selects the command builder: `ovms`, `vllm`, `vllm-mtp`, `vllm-tp2`, `vllm-arext`, `vllm-autoround`, `vllm-dflash`, `gguf`, `gguf-tiered`, `exl3` |
| `image` | container image (ignored when a `llama_bin` native path exists) |
| `ctx` | default context; clamped by `ctx_max` |
| `ctx_max` | hard ceiling; user ctx above it is clamped with a warning |
| `ctx_safe` | context the UI preselects ("Safe" preset); falls back to `ctx` when absent |
| `ctx_note` | context hint line under the context field; used as the `ctx_source` label when no on-disk `config.json` value wins |
| `kv_options` | list of KV choices the UI offers for this recipe. The UI falls back to `q8_0/q4_1`, `q8_0`, `q5_0/q4_1`, `f16` for `gguf`, `auto` for `ovms`, and `fp8` otherwise; the llama.cpp/vLLM/EXL3 builders validate a caller-supplied `kv` against the recipe list (or `q8_0/q4_1`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp16`, `bf16`, `auto`, `fp8`) and warn + fall back to the recipe `kv` or the first option when it is not listed. `ovms` does not read `kv` at all |
| `prefill_options` | EXL3 only: prefill modes the UI offers (`int8`, `fp16`). `int8` is the engine default and adds nothing to the command; `fp16` adds `--set env.EXL3_INT8_PREFILL=0`. An unlisted value warns and falls back to `int8`; other engines warn that the option is ignored |
| `operator_hold` | non-empty text blocks the recipe: launches are refused and downloads are not authorized; only a dry run still renders, with a warning |
| `power` | advisory watts hint: when it exceeds the selected card's current hwmon cap the plan warns; never applied, and no power-cap command is generated |
| `perf` | throughput note shown in Recipe notes |
| `fixed_flags` | argv appended verbatim (for `gguf`: placed before the KV flags, so including `--cache-type-k` suppresses the recipe's `--cache-type-k/-v` pair) |
| `search_name` | artifact match fallback (see Detection) |
| `model_path` / `source_model` / `gguf` | direct artifact path (supports `~`); checked on disk before scan results. `source_model` also sets the OVMS `--source_model` arg |
| `download` | see below |
| `draft` | speculative-draft sidecar: `{"path": "<abs dir>"}` or `{"search": "<name fragment>"}` plus an optional `download` block fetched as a second download group |
| `recipe_ver` | per-recipe version for the update channel (`YYYY-MM-DD` or int) |
| `recipe_note` | copied into the published manifest as the update notice text |
| `recipe_recommended` | truthy marks the manifest entry "recommended" |

### `download`

```json
{"kind": "file|snapshot", "repo": "org/repo", "revision": "abc123",
 "name": "file.gguf", "extra_files": ["mmproj-*.gguf"],
 "search": "...", "match": "qwen3.8-27b" | ["alt1", "alt2"],
 "prefer": ["w4a16", "gptq"], "exclude": ["sc_5.00bpw"], "dest": "dir-name",
 "quant": "GPTQ-INT4 sym G128", "quant_method": "gptq",
 "verify": ["config.json"], "note": "..."}
```

- `kind=file`: a single file (`name`) + optional `extra_files` sidecars, fetched
  into `models_dir`; matched against GGUF basenames from the scan.
- `kind=snapshot`: a whole repo tree into `ovms_repo` (openvino) or
  `models_dir` (everything else); matched against detected model directories.
  The directory is the repo basename unless `dest` names another single
  directory; use `dest` when two recipes pin different branches of one repo.
- `quant_method`: pin the quantization format a snapshot may satisfy. A
  same-family directory whose `quantization_config.json` (or root
  `quant_method`) records a different format is rejected, so a GPTQ repo cannot
  satisfy an AutoRound or EXL3 recipe. An artifact that records no format falls
  through to the `prefer`/`exclude` filters. This affects detection only — the
  download plan checks repository, paths, sizes, and free space.
- `repo`: automatic download is only offered when it is in the
  `HF_APPROVED_REPOS` allow-list in `b70launcher/downloads.py`; otherwise the
  download plan is rejected with "Automatic download unavailable …" and the
  user obtains the artifact manually. `search` is a detection hint, not a link.
- `match` / `search` / `name` accept a string or a list of alternates;
  `prefer` narrows same-family hits to the right quant/format; `exclude`
  drops same-family hits whose normalized name contains any listed string
  (e.g. a sibling branch that another recipe owns).
- `quant`, `note`: display only. `verify` is inert metadata — nothing reads it.
- Download guidance shown before a fetch comes from the recipe name/search and
  the daemon's plan text; the current code and UI have no recipe-documentation
  link handling.
- A plan whose remote file sizes cannot be listed is rejected before the
  download starts (`remote file sizes unknown`), and free space must cover
  `size × 1.05`.

### Detection order

1. Direct recipe path (`gguf`, `model_path`, `source_model`) that exists on
   disk wins immediately.
2. Exact `download.name` basename in the scan pool.
3. Same-family match: normalized `match` → `search` → `name` → `search_name` →
   repo basename, filtered by `prefer`, by `download.quant_method`, and (for
   snapshots) by detected kind. `mmproj*` and `*dflash*`/`*draft*` basenames
   never satisfy a recipe unless the recipe itself asks for them.
4. `draft_model` (llama.cpp MTP) is a host path with `~`, or a bare
   `*.gguf` basename resolved through the last scan. `draft.search` /
   `draft.path` (vLLM DFlash) resolve a draft model *directory*: the directory
   must contain `config.json` plus weights, and is looked up under the scan
   roots to depth 4.

The default context is the recipe's tested `ctx`, lowered to the detected
`config.json` limit (`max_position_embeddings` etc., incl. `text_config`) when
that is smaller, then capped at `ctx_max`. `kind: ovms` ignores the IR config
because OVMS manages KV internally. Without a recipe `ctx` the detected value
is used. A recipe that lists two `gpus` refuses a single-card selection.

## Per-kind fields

### `kind: ovms` (engine `openvino`)

`source_model` (repo id or on-disk dir), `tool_parser`, `reasoning_parser`,
`cim_long_ctx` (adds `--cache_interval_multiplier N` when ctx > 20480).
Mounts the artifact dir at `/models` (file) or `/models/model` (dir); OVMS
manages KV internally — `ctx` is advisory.

### `kind: vllm` family (engine `vllm`)

Common: `model_path`, `dtype`, `kv` (`"fp8"` adds `--kv-cache-dtype fp8`),
`kv_options`, `spec_tokens` (enables MTP `--speculative-config`; the caller
sends `mtp:false` to disable), `speculative` (string draft model id, or a dict
for `vllm-dflash`), `gpus`/`tp` (dual-card intent — see Caveats).

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
  `--cap-add SYS_PTRACE`, `--ipc=host`, `--shm-size` (recipe `shm_size`,
  default `16g`), oneCCL threshold env; `tp` defaults 2.
- `vllm-dflash`: `patches[]` (each name must exist under `patches/`, else the
  build fails), `quantization` (default `gptq`), `shm_size`, `draft`
  (directory mount; without a detected draft it serves without speculation and
  warns), `speculative` dict (`method`, `model`, `num_speculative_tokens`,
  default 7), `prefix_caching` (on/off explicitly).
- `vllm-autoround`: generic path + `--enforce-eager` via `fixed_flags` and the
  FP16 crash-guard warning.

### `kind: gguf` / `gguf-tiered` (engine `llamacpp`)

`gguf` (default path), `llama_bin` (recipe- or settings-level native binary;
`~` expanded — when it exists and `use_docker` is false, no container is used),
`kv` (`q8_0`, `q5_0/q4_1`, `q8_0/q4_1`, `f16`; a UI override maps to
`--cache-type-k/-v`), `kv_options`, `tensor_split`, `split_mode`,
`offload_tensors` (`-ot`), `draft_model` (host path or bare basename resolved
through the last scan), `draft_device` (default `SYCL1`), `spec_tokens`,
`spec_p_min`.

Native runs get `ONEAPI_DEVICE_SELECTOR`/`ZE_AFFINITY_MASK`/`SYCL_*` env, all
`/opt/intel/oneapi/*/latest/lib` dirs prepended to `LD_LIBRARY_PATH`, and bind
`127.0.0.1`. `gguf-tiered` additionally sets `LLAMA_ATTN_ROT_DISABLE=1` and
disables immediate command lists. Every llama.cpp launch gets `-ngl 99`,
`--flash-attn on`, `--metrics`.

### `kind: exl3` (engine `exl3`)

`model_path` (required on disk), `serve_config` (passed as first arg to the
image entrypoint), `docker_sock` (isolated daemon socket — `exl3_stack` starts
it via `sudo -n` containerd/dockerd when down and never deletes its state),
`inner_port` (container-side port, default 8100). Linux only; fp8 KV;
`gpu_memory_utilization` 0.90 (<128K ctx) or 0.94 (≥128K). `prefill_options` / caller `prefill`
(`int8` default, `fp16` max quality). Every `extra_env` entry is passed both as `-e` and as
`--set env.NAME="value"`, because the serve config's `env:` block overrides `docker -e`.

## Custom artifacts (`model_id: "__custom__"`)

`Browse… / Set Custom Path` in the UI posts `custom_path` with
`model_id="__custom__"`. Rules:

- The resolved path must be inside a configured scan root and exist.
- Files: only `.gguf` (engine must be `llamacpp`). Directories are sniffed:
  `openvino_language_model.*` or xml+bin → `openvino`; else `vllm`, or `exl3`
  when `quantization_config.json` has `quant_method=exl3`. Picking the wrong
  engine fails the build with a "pick the matching engine" error.
- The recipe is a generic template (image, ctx 32768/20480/65536, power,
  `kv_options`) plus `ctx` from the artifact's `config.json` when present. There
  is no verified-recipe tuning — the launch warning says so.

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
   localhost for testing; 8 MiB cap) and overlays the document in
   `catalog.apply_recipe_doc`:
   - model fields are replaced only from `REMOTE_MODEL_FIELDS`
     (`recommended_engine`, `recommendation`, `badge`, `blurb`, `subtitle`,
     `tags`, `chips`, `brand`);
   - recipe fields are **field-merged** over the local recipe, limited to
     `REMOTE_RECIPE_FIELDS` (`ctx`, `ctx_max`, `ctx_note`, `ctx_safe`, `power`,
     `dtype`, `kv`, `kv_options`, `prefill_options`, `download`, `spec_tokens`, `spec_p_min`,
     `speculative`, `draft_device`, `tensor_split`, `split_mode`,
     `offload_tensors`, `tool_parser`, `reasoning_parser`, `cim_long_ctx`,
     `search_name`, `inner_port`, `tp`, `perf`, `topology`, `recipe_ver`) plus a
     bare `draft_model` filename;
   - exec-shaping keys (`kind`, `image`, `llama_bin`, `docker_sock`,
     `fixed_flags`, `serve_config`) and host paths (`model_path`, `gguf`,
     `source_model`) never come from the wire, so a catalog cannot swap the
     binary/image, pick a socket, append container argv, or remount host paths;
   - a model or engine this app does not ship is **refused**, not added: the
     payload cannot carry its `kind`/`image`, and the response says it needs a
     newer launcher app;
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

- The UI sends `model_id`, `engine`, `port`, `ctx`, `kv`, `prefill` (EXL3), `gpus` (list of ints),
  `harness`, and `no_test`; it never sends `slots`, `mtp`, `use_docker`,
  `power`, `extra`, `extra_env`, or `models_dir`. Dual-GPU recipes rely on
  recipe-level `tp`/`tensor_split`/`gpus`.
- Shipped but not read by the backend: recipe `ver` (the code reads
  `recipe_ver`), `perf_note`, `topology` (allow-listed for updates but not
  consumed), and `download.verify`. The UI shows `perf`, not `perf_note`.
- `settings.json` `power_modes`, `vllm_kv_options`, `llamacpp_kv_options`,
  `terminal_linux`, and `harnesses` are shipped but unused — the UI hardcodes
  the power/KV choices, `harness.harness_line` fixes the client commands, and
  `open_harness` probes terminals itself.
- `POST /api/settings` can write only `scan_dirs`, `models_dir`, `open_with`,
  and `sync_clients`; `settings-override.json` also accepts `ovms_repo`,
  `llama_bin`, `exl3_data_root`, and `recipe_overrides` when edited by hand.
