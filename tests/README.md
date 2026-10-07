# Tests for B70 Launcher

Stdlib-only `unittest` suite — no pytest, no third-party deps, matching the
project's own constraint.

## Run

```bash
cd b70-launcher
python3 -m unittest discover -s tests -v
```

Expected runtime: under 60 seconds (usually under 10).

## Layout

| File               | Covers |
|--------------------|--------|
| `support_env.py`   | Shared fixture: fake HOME / XDG_STATE_HOME, inert update URL, free-port and file helpers, `LauncherStateCase` (snapshots/restores every mutable launcher global). Imported before `launcher` by every test module — this is what keeps the suite off real state. |
| `test_pure.py`     | Pure functions: `safe_artifact`, `_norm`/`_as_list`/`_first_nonempty`, `fmt_bytes`, `_ver_key`, `_stamp_ver`, `pretty`, `read_ctx_from_config`, `container_path`, `resolve_repo`, Prometheus parsing/classification, `scan_roots`, `_walk_root` classification (gguf/openvino/vllm/exl3, skip dirs, depth limit, deadline), `build_catalog`, `_dir_size_mib`, `_exl3_roots`, pid helpers. |
| `test_detect.py`   | `detect()` exact-vs-family matching, `prefer`/`match` filters, mmproj & draft-sidecar exclusion, snapshot kind filtering per engine, direct-path recipe fields with `~` expansion; `resolve_ctx` precedence + `ctx_max` clamp; `_resolve_draft` host-path/scan resolution; `prepare_custom` confinement, engine/kind agreement, format sniffing, ctx sniffing. |
| `test_build.py`    | `build()` command generation: llama.cpp docker & native paths, `kv_map` for every KV option + `fixed_flags` suppression, tiered-memory flags, `draft_model` resolution into `-md`/`--spec-type` tokens, OVMS/vLLM/vLLM-TP2/vLLM-MTP/vLLM-AutoRound shapes, validation errors, `__custom__` flow; `record_usage` deltas, `usage_summary`, `persist_state` filtering; `harness_line` per harness. |
| `test_recipes.py`  | Settings-override merge (whitelist + `recipe_overrides`), remote catalog overlay (`apply_recipe_doc`, `_load_remote_overlay`), user-override precedence over remote recipes, `local_recipe_ver`, `recipe_notices` matrix, `fetch_remote_recipes` URL policy, `apply_recipe_update` with a mocked fetch. |
| `test_http.py`     | Real `launcher.py` subprocess on an ephemeral port: `/api/state` shape (9 models + version), token substitution in `index.html`, POST auth (missing/wrong token, bad Origin, bad Content-Type -> 403), bad Host -> 403, body limits -> 400, `/api/build` + `/api/launch` dry-run, `/api/recipes/update` error path, `/api/settings` + `/api/scan` detection flow, `/api/shutdown` terminates the process cleanly. |

## Isolation guarantees

- `support_env` sets `HOME` and `XDG_STATE_HOME` to a temp dir **before**
  `import launcher`, so `DATA`/`LOGDIR`/override/usage/state files all live
  under `/tmp/b70-launcher-tests-*` (removed at exit). The real
  `~/.local/state/b70-launcher` is never touched.
- `B70_UPDATE_URL` is a `file://` path that fails instantly — the update
  thread opens no socket.
- The HTTP test's subprocess gets `PATH` pointed at an empty directory, so
  `docker`, `sudo`, `pkill`, `fuser`, `lspci`, terminals and browsers cannot
  be invoked from inside it.
- Ports are allocated ephemerally (never 8765, never <1024). `/api/launch`
  is only ever exercised with `dry_run: true`; a non-dry launch is asserted
  to fail preflight (docker unreachable) rather than touching the GPU.
