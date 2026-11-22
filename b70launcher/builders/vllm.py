"""vLLM builders: generic, tp2, arext (AutoRound), mtp, dflash."""
import json
import re
import shlex
from pathlib import Path

from ..paths import IS_WIN, HERE
from . import common


def _docker_run(b, cname):
    return ["docker", "run", "--rm", "--name", cname] + common.labels_for(b.cfg, b.port)


def _patch_steps(tokens, patches_dir, wanted):
    steps = []
    for i, pname in enumerate(wanted):
        tokens += ["-v", f"{patches_dir / pname}:/patch_{i}.py:ro"]
        steps.append(f"python /patch_{i}.py")
    return steps


def _need_patches(wanted, patches_dir, label):
    missing = [p for p in wanted if not (patches_dir / p).is_file()]
    if missing:
        raise common.BuildError(f"bundled {label} patch scripts are missing from this install: "
                                + ", ".join(missing))


def _cache_flag(b):
    return "--enable-prefix-caching" if b.prefix_cache else "--no-enable-prefix-caching"


def _cache_warn(b, spec):
    """Speculative stacks used to ship with the cache off; say plainly it is on."""
    if spec and b.prefix_cache and b.cache_default:
        b.warns.append("prefix caching ON (default); use --cache off for benchmarks")


def _artifact_dir(b, what):
    """Resolve the on-disk model dir for kinds that need a real artifact.

    Real launch without the artifact fails with a clear error; a dry run
    renders with a placeholder mount + warning instead (audit C1)."""
    model_dir = (b.det or {}).get("path") or b.recipe.get("model_path", "")
    if model_dir and Path(str(model_dir)).expanduser().exists():
        return str(Path(str(model_dir)).expanduser()), None
    if not b.cfg.get("dry_run"):
        raise common.BuildError(f"{what} artifact not found on disk — point a model folder at it "
                                "or set the recipe's model_path.")
    placeholder = str(Path(b.cfg.get("models_dir") or "~/models").expanduser())
    return placeholder, (f"{what} artifact not found on disk — preview mounts a placeholder; "
                         "a real launch is refused until the artifact exists.")


_TP2_RECIPE_ENV = re.compile(r"(CCL|FI|VLLM|B70|HF_HUB)_[A-Z0-9_]+")


def build_tp2(b):
    """vLLM TP2 across two cards.

    Default: inline worker-affinity patch, then `vllm serve`. A recipe with
    `entry` instead runs that bundled wrapper (patches/ mounted at /patches),
    which applies its own patches and checks; `patches` lists the files it
    needs. `network: "host"` swaps the port publish for host networking."""
    cfg, model, recipe, warns = b.cfg, b.model, b.recipe, b.warns
    patches_dir = HERE / "patches"
    entry = recipe.get("entry")
    wanted = ([entry] + [Path(p).name for p in (recipe.get("patches") or [])]
              if entry else ["patch_vllm_worker_affinity.py"])
    _need_patches(wanted, patches_dir, "TP2")
    # recipe env: collective/runtime knobs only; local recipes.json only (not a remote field)
    recipe_env = recipe.get("env") or {}
    for k in recipe_env:
        if not _TP2_RECIPE_ENV.fullmatch(k):
            raise common.BuildError(f"recipe env {k} is not a CCL_/FI_/VLLM_/B70_/HF_HUB_ setting")
    tp_size = len(b.gpus)
    if tp_size < recipe.get("tp", 2):
        warns.append(f"TP{recipe.get('tp', 2)} recipe forced onto {tp_size} GPU — "
                     "tensor parallelism reduced; weights+KV may not fit.")
    mount_src = Path(b.det["path"]) if b.det and b.det.get("path") else Path(
        cfg.get("models_dir") or "~/models").expanduser()
    if not b.det:
        warns.append("model not detected on disk — preview mounts the models folder; a real launch needs the artifact.")
    kv = common.resolve_kv(b.kv, recipe, ("fp8",), warns)
    cname = common.cname_for(model["id"], "vllm", b.port)
    host_net = recipe.get("network") == "host"
    tokens = _docker_run(b, cname)
    tokens += common.dri_args(b) + ["--cap-add", "SYS_PTRACE", "--ipc=host", "--workdir", "/"]
    if recipe.get("shm_size"):
        tokens += ["--shm-size", recipe["shm_size"]]
    tokens += ["-v", f"{mount_src}:/model:ro"]
    if entry:
        tokens += ["-v", f"{patches_dir}:/patches:ro"]
    else:
        tokens += ["-v", f"{patches_dir / 'patch_vllm_worker_affinity.py'}:/patch_affinity.py:ro"]
    tokens += ["--network", "host"] if host_net else ["-p", f"127.0.0.1:{b.port}:{b.port}"]
    if not entry:
        tokens += ["-e", "VLLM_TARGET_DEVICE=xpu",
                   "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
                   "-e", "B70_MTP_BF16_DRAFT=1",
                   "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
                   "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True",
                   "-e", "CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296",
                   "-e", "CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296",
                   "-e", "CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296",
                   "-e", "CCL_SYCL_ALLTOALL_TMP_BUF=1"]
    for k, v in recipe_env.items():
        tokens += ["-e", f"{k}={v}"]
    # mask + RELATIVE selector (H4/H5): the mask renumbers cards, so the
    # selector addresses 0..n-1 even when the mask hides card 0
    for k, v in common.container_env(b).items():
        tokens += ["-e", f"{k}={v}"]
    tokens += b.envargs
    serve = ["/model"]
    if host_net:
        serve += ["--host", "127.0.0.1"]
    if recipe.get("quantization", "fp8"):  # null: vLLM reads it from config.json
        serve += ["--quantization", recipe.get("quantization", "fp8")]
    slots = b.slots if cfg.get("slots") is not None else recipe.get("max_num_seqs", b.slots)
    serve += ["--dtype", recipe.get("dtype", "bfloat16"),
              "--tensor-parallel-size", str(tp_size),
              "--max-model-len", str(b.ctx)]
    if recipe.get("async_scheduling", True):
        serve += ["--async-scheduling"]
    serve += ["--gpu-memory-utilization", str(recipe.get("gmu", "0.90")),
              "--kv-cache-dtype", kv,
              "--port", str(b.port),
              "--max-num-seqs", str(slots),
              "--max-num-batched-tokens", str(recipe.get("batched_tokens", 4096)),
              _cache_flag(b),
              "--served-model-name", model["name"]]
    if recipe.get("language_model_only", True):
        serve += ["--language-model-only"]
    use_mtp = bool(recipe.get("spec_tokens") or recipe.get("speculative")) and cfg.get("mtp", True)
    if cfg.get("mtp") and not (recipe.get("spec_tokens") or recipe.get("speculative")):
        warns.append("recipe has no speculative-decoding config — --mtp ignored.")
    if use_mtp:
        serve += ["--speculative-config",
                  json.dumps({"method": "mtp", "num_speculative_tokens": recipe.get("spec_tokens", 8)})]
    serve += recipe.get("fixed_flags", []) + b.extra
    if entry:
        tokens += ["--entrypoint", "bash", recipe["image"], f"/patches/{entry}", "serve"] + serve
    else:
        script = "set -e; python /patch_affinity.py; exec vllm serve " + " ".join(shlex.quote(a) for a in serve)
        tokens += ["--entrypoint", "bash", recipe["image"], "-lc", script]
    if tp_size > 1:
        warns.append("vLLM TP2 dual-card: worker affinity patch + oneCCL thresholds pinned"
                     + (f"; entry {entry} applies {', '.join(wanted[1:]) or 'no extra patches'}" if entry else "")
                     + ".")
    _cache_warn(b, use_mtp)
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": model["name"],
            "native": False, "docker_sock": ""}



def build_mtp(b):
    cfg, model, recipe, warns = b.cfg, b.model, b.recipe, b.warns
    patches_dir = HERE / "patches"
    _need_patches(["patch_mtp_nightly.py", "patch_mtp_boundary.py"], patches_dir, "MTP")
    model_dir, warn = _artifact_dir(b, "MTP")
    if warn:
        warns.append(warn)
    kv = common.resolve_kv(b.kv, recipe, ("fp8",), warns)
    cname = common.cname_for(model["id"], "vllm", b.port)
    tokens = _docker_run(b, cname)
    tokens += common.dri_args(b) + [
               "-v", f"{model_dir}:/model:ro",
               "-v", f"{patches_dir / 'patch_mtp_nightly.py'}:/patch_mtp.py:ro",
               "-v", f"{patches_dir / 'patch_mtp_boundary.py'}:/patch_boundary.py:ro",
               "-p", f"127.0.0.1:{b.port}:{b.port}",
               "-e", "VLLM_TARGET_DEVICE=xpu",
               "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
               "-e", "B70_MTP_BF16_DRAFT=1",
               "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
               "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True"]
    for k, v in common.container_env(b).items():
        tokens += ["-e", f"{k}={v}"]
    tokens += b.envargs
    use_mtp = bool(recipe.get("spec_tokens") or recipe.get("speculative")) and cfg.get("mtp", True)
    serve = ["vllm", "serve", "/model", "--quantization", recipe.get("quantization", "gptq"),
             "--dtype", "float16",
             "--max-model-len", str(b.ctx),
             "--gpu-memory-utilization", "0.88" if use_mtp else "0.90",
             "--kv-cache-dtype", kv,
             "--port", str(b.port),
             "--max-num-seqs", str(b.slots),
             "--max-num-batched-tokens", "8192",
             _cache_flag(b),
             "--served-model-name", model["name"],
             "--language-model-only"]
    if use_mtp:
        serve += ["--speculative-config",
                  json.dumps({"method": "mtp", "num_speculative_tokens": recipe.get("spec_tokens", 4)})]
    serve += recipe.get("fixed_flags", []) + b.extra  # fixed_flags honoured (audit M3)
    script = "set -e; python /patch_mtp.py; python /patch_boundary.py; exec " + \
        " ".join(shlex.quote(a) for a in serve)
    tokens += ["--entrypoint", "bash", recipe["image"], "-lc", script]
    warns.append("cookbook MTP path: BF16 draft + FP8 KV + patched GDN boundary.")
    _cache_warn(b, use_mtp)
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": model["name"],
            "native": False, "docker_sock": ""}


def build_dflash(b):
    cfg, model, recipe, warns = b.cfg, b.model, b.recipe, b.warns
    patches_dir = HERE / "patches"
    wanted = [Path(p).name for p in (recipe.get("patches") or [])]
    _need_patches(wanted, patches_dir, "DFlash")
    from .. import scan
    model_dir = (b.det or {}).get("path")
    if not model_dir:
        warns.append("model not detected on disk — preview mounts the models folder; a real launch needs the artifact.")
    scfg = recipe.get("speculative") or {}
    draft_dir = scan._find_draft_dir(recipe) if scfg.get("model") else None
    spec_on = bool(scfg) and cfg.get("mtp", True)
    if spec_on and not draft_dir:
        wants = (recipe.get("draft") or {}).get("search", "a DFlash draft dir")
        warns.append(f"DFlash draft not detected ({wants}) — serving without "
                     "speculation; drop the draft dir under a model folder (b70 scan --roots).")
        spec_on = False
    kv = common.resolve_kv(b.kv, recipe, common.KV_VLLM, warns)
    cname = common.cname_for(model["id"], "vllm", b.port)
    tokens = _docker_run(b, cname)
    tokens += common.dri_args(b) + [
               "--ipc=host", "--shm-size", recipe.get("shm_size", "16g")]
    if model_dir:
        tokens += ["-v", f"{model_dir}:/model:ro"]
    else:
        mount_src = Path(cfg.get("models_dir") or "~/models").expanduser()
        tokens += ["-v", f"{mount_src}:/models:ro"]
    tokens += ["-p", f"127.0.0.1:{b.port}:{b.port}",
               "-e", "VLLM_TARGET_DEVICE=xpu",
               "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
               "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
               "-e", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0",
               "-e", "SYCL_CACHE_PERSISTENT=0",
               "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True"]
    for k, v in common.container_env(b).items():
        tokens += ["-e", f"{k}={v}"]
    tokens += b.envargs
    steps = _patch_steps(tokens, patches_dir, wanted)
    ssu_dir = patches_dir / "ssu-b70-b8w4"
    if ssu_dir.is_dir():
        tokens += ["-v", f"{ssu_dir}:/ssu:ro"]
        steps.append("for d in /workspace/vllm/vllm /opt/venv/lib/python3.12/site-packages/vllm; "
                     "do [ -d \"$d/model_executor/layers/mamba/ops/configs/selective_state_update\" ] && "
                     "cp /ssu/*.json \"$d/model_executor/layers/mamba/ops/configs/selective_state_update/\"; "
                     "done; true")
    if draft_dir:
        tokens += ["-v", f"{draft_dir}:{scfg['model']}:ro"]
    serve = ["vllm", "serve", "/model" if model_dir else "/models/model",
             "--served-model-name", model["name"],
             "--dtype", recipe.get("dtype", "float16"),
             "--quantization", recipe.get("quantization", "gptq"),
             "--max-model-len", str(b.ctx),
             "--max-num-seqs", str(b.slots),
             "--max-num-batched-tokens", "8192",
             "--gpu-memory-utilization", "0.90",
             "--language-model-only", "--async-scheduling",
             "--host", "0.0.0.0", "--port", str(b.port),
             _cache_flag(b),
             "--kv-cache-dtype", kv]
    if spec_on:
        serve += ["--speculative-config", json.dumps(
            {"method": scfg.get("method", "dflash"),
             "model": scfg["model"],
             "num_speculative_tokens": scfg.get("num_speculative_tokens", 7)})]
    serve += recipe.get("fixed_flags", []) + b.extra
    script = "set -e; " + "; ".join(steps) + ("; " if steps else "") + \
        "exec " + " ".join(shlex.quote(a) for a in serve)
    tokens += ["--entrypoint", "bash", recipe["image"], "-lc", script]
    warns.append("Nemotron DFlash: grouped-topk router patch + SSU B8/W4 tuning"
                 + (f" + draft {Path(draft_dir).name} (n={scfg.get('num_speculative_tokens', 7)})"
                    if draft_dir else "") + ".")
    _cache_warn(b, spec_on)
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": model["name"],
            "native": False, "docker_sock": ""}


def build_generic(b):
    """generic vllm (+ vllm-autoround via fixed_flags): no --privileged, no rw
    HF-cache mount (audit L7) — plain device access like the other builders."""
    cfg, model, recipe, warns = b.cfg, b.model, b.recipe, b.warns
    fallback = recipe.get("model_path", "/models/model")
    if b.det:
        model_path, mount_src, mount_dst = common.container_path(b.det, fallback)
        if mount_src is None:
            mount_src = cfg.get("models_dir") or "~/models"
            mount_dst = "/models"
    else:
        model_path = fallback
        mount_src = cfg.get("models_dir") or "~/models"
        mount_dst = "/models"
        warns.append("model not detected on disk — preview uses the models folder; a real launch needs the artifact.")
    kv = common.resolve_kv(b.kv, recipe, common.KV_VLLM, warns)
    use_mtp = bool(recipe.get("spec_tokens") or recipe.get("speculative")) and cfg.get("mtp", True)
    if cfg.get("mtp") and not (recipe.get("spec_tokens") or recipe.get("speculative")):
        warns.append("recipe has no speculative-decoding config — --mtp ignored.")
    serve_args = ["--model", model_path,
                  "--dtype", recipe.get("dtype", "bfloat16"),
                  "--gpu-memory-utilization", "0.92",
                  "--max-model-len", str(b.ctx),
                  _cache_flag(b),
                  "--port", str(b.port)]
    if use_mtp:
        scfg = recipe.get("speculative") or {}
        if isinstance(scfg, str):  # legacy string draft model id
            scfg = {"model": scfg}
        spec = {"method": scfg.get("method", "mtp"),
                "num_speculative_tokens": scfg.get("num_speculative_tokens",
                                                   recipe.get("spec_tokens", 4))}
        if scfg.get("model"):
            spec["model"] = scfg["model"]
        serve_args += ["--speculative-config", json.dumps(spec)]
    serve_args += ["--kv-cache-dtype", kv]
    serve_args += recipe.get("fixed_flags", [])
    if len(b.gpus) > 1:
        serve_args += ["--tensor-parallel-size", str(len(b.gpus))]
        warns.append("Multi-GPU TP on XPU: verify the level_zero selector string on your driver.")
    if b.slots > 1:
        serve_args += ["--max-num-seqs", str(b.slots)]
    serve_args += b.extra
    patches_dir = HERE / "patches"
    wanted = [Path(p).name for p in (recipe.get("patches") or [])]
    _need_patches(wanted, patches_dir, "vLLM")
    cname = common.cname_for(model["id"], "vllm", b.port)
    tokens = _docker_run(b, cname)
    if not IS_WIN:
        tokens += common.dri_args(b)
    else:
        warns.append("Windows GPU passthrough is unvalidated; real launch blocked by preflight.")
    tokens += ["--ipc=host",
               "-v", f"{Path(str(mount_src)).expanduser()}:{mount_dst}:ro",
               "-p", f"127.0.0.1:{b.port}:{b.port}"]
    for k, v in common.container_env(b).items():
        tokens += ["-e", f"{k}={v}"]
    tokens += b.envargs
    if wanted:
        steps = _patch_steps(tokens, patches_dir, wanted)
        script = "set -e; " + "; ".join(steps) + "; exec vllm serve " + \
            " ".join(shlex.quote(a) for a in serve_args)
        tokens += ["--entrypoint", "bash", recipe["image"], "-lc", script]
    else:
        tokens += [recipe["image"]] + serve_args
    if recipe.get("kind") == "vllm-autoround" and "--enforce-eager" in recipe.get("fixed_flags", []):
        warns.append("recipe pins --enforce-eager (dt_bias crash guard on the BF16 path).")
    _cache_warn(b, use_mtp)
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": model["name"],
            "native": False, "docker_sock": ""}


_NATIVE_RECIPE_ENV = re.compile(r"(INK2|VLLM|TRITON|ZE|ONEAPI|SYCL)_[A-Z0-9_]+")


def build_native(b):
    """Native host-venv vLLM XPU server (for out-of-tree XPU plugins such as
    vllm-inkling2-xpu where no public container image bundles the plugin yet).

    Settings key ``vllm_venv`` (settings-override.json or per-recipe override)
    names an absolute path to the Python venv containing ``bin/vllm``.
    """
    from .. import catalog

    cfg, model, recipe, warns = b.cfg, b.model, b.recipe, b.warns
    if IS_WIN:
        raise common.BuildError("Native vLLM XPU is Linux-only.")

    venv_raw = str(recipe.get("vllm_venv") or catalog.SETTINGS.get("vllm_venv") or "").strip()
    missing = []
    if not venv_raw:
        missing.append('the vLLM XPU venv: set "vllm_venv" in settings-override.json '
                       "(a Python venv with vllm + plugin installed)")
        vllm_bin = Path("~/vllm-xpu-venv/bin/vllm").expanduser()
    else:
        venv_path = Path(venv_raw).expanduser()
        if not venv_path.is_absolute():
            raise common.BuildError(f'setting "vllm_venv" must be an absolute path, not {venv_raw!r}')
        vllm_bin = venv_path / "bin" / "vllm"
        if not vllm_bin.is_file():
            missing.append(f"vLLM executable {vllm_bin}")

    model_dir, art_warn = _artifact_dir(b, "vLLM native")
    if art_warn:
        warns.append(art_warn)

    if missing:
        text = "vLLM native recipe needs " + "; ".join(missing)
        if not cfg.get("dry_run"):
            raise common.BuildError(text + ".")
        warns.append(text + " — preview only; a real launch is refused until they exist.")

    recipe_env = recipe.get("env") or {}
    for k in recipe_env:
        if not _NATIVE_RECIPE_ENV.fullmatch(k):
            raise common.BuildError(f"recipe env {k} is not an allowed INK2_/VLLM_/TRITON_/ZE_/ONEAPI_/SYCL_ setting")

    scfg = recipe.get("speculative") or {}
    use_spec = bool(scfg or recipe.get("spec_tokens")) and cfg.get("mtp", True)
    if cfg.get("mtp") and not (scfg or recipe.get("spec_tokens")):
        warns.append("recipe has no speculative-decoding config — --mtp ignored.")

    # Auto-size SWA --max-num-batched-tokens & gmu for single-B70 context envelopes
    ctx = b.ctx
    batched = int(recipe.get("batched_tokens", 2048))
    gmu = float(recipe.get("gmu", 0.93 if use_spec else 0.95))
    if recipe.get("swa_mbt_profiles"):
        if use_spec:
            if ctx > 163840:
                warns.append("DSpark k=7 max proven context on one B70 is 163,840 (mbt=1024); "
                             "use --no-mtp for 200K-256K no-spec context.")
            if ctx > 106496:
                batched = 1024
                gmu = 0.95
            else:
                batched = 2048
                gmu = 0.93
        else:
            if ctx > 200000:
                batched = 1024
                gmu = 0.95
            elif ctx > 90112:
                batched = 2048
                gmu = 0.95
            else:
                batched = 2048
                gmu = 0.88

    kv = common.resolve_kv(b.kv, recipe, common.KV_VLLM, warns)
    slots = b.slots if cfg.get("slots") is not None else recipe.get("max_num_seqs", 1)
    served_name = recipe.get("served_model_name", model["name"])
    serve = [
        str(vllm_bin), "serve", model_dir,
        "--host", "127.0.0.1",
        "--port", str(b.port),
        "--served-model-name", served_name,
        "--dtype", recipe.get("dtype", "bfloat16"),
        "--max-model-len", str(ctx),
        "--max-num-batched-tokens", str(batched),
        "--max-num-seqs", str(slots),
        "--gpu-memory-utilization", f"{gmu:.2f}",
        _cache_flag(b),
    ]
    if kv and kv != "auto":
        serve += ["--kv-cache-dtype", kv]
    if use_spec:
        spec = dict(scfg) if isinstance(scfg, dict) else {"method": "mtp"}
        spec.setdefault("num_speculative_tokens", recipe.get("spec_tokens", 7))
        serve += ["--speculative-config", json.dumps(spec)]
    serve += recipe.get("fixed_flags", []) + b.extra

    # Keep VLLM_RPC_BASE_PATH short under /tmp to stay below Linux's 107-byte sun_path limit.
    cmd = (
        'RPC_DIR="$(mktemp -d /tmp/b70-vllm-ipc-XXXXXX)" && '
        'trap \'rm -rf "$RPC_DIR"\' EXIT && '
        'export VLLM_RPC_BASE_PATH="$RPC_DIR" && '
        + "exec " + shlex.join(serve)
    )

    env = {
        "ZE_ENABLE_PCI_ID_DEVICE_ORDER": "1",
        "ZE_AFFINITY_MASK": ",".join(str(g) for g in b.gpus),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    for k, v in recipe_env.items():
        env[k] = str(v)
    env.update(b.extra_env)

    _cache_warn(b, use_spec)
    return {
        "tokens": ["bash", "-c", cmd],
        "env": env,
        "cname": "",
        "served_name": served_name,
        "native": True,
        "effective_binary": str(vllm_bin),
        "docker_sock": "",
    }


BUILDERS = {
    "vllm-tp2": build_tp2,
    "vllm-mtp": build_mtp,
    "vllm-dflash": build_dflash,
    "vllm-native": build_native,
}
