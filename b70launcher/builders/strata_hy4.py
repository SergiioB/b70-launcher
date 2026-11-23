"""Strata Hy4-Flash builder: the user's Niko1221/Strata v0.2.20+patch checkout.

The launcher does not build or ship Strata. A ``strata-hy4`` recipe names the
files the verified dual-B70 route uses: the ``strata-hy4-sycl`` engine binary
(``build_hy4_sycl.sh bmg`` output), the server's Python venv and
``engine/sycl/serve/server_intel.py`` — all relative to the checkout
(settings ``strata_hy4_root``, falling back to ``strata_root``). The two
``.strata`` packs (native SQ build + mandatory Engram shard) come from the
model scan like every other ``kind=file`` download. The server and engine run
in one container of the locally built ``strata-sycl-dev`` image; the config is
rendered here and written inside the container at start, the server auto-
generates the tokenizer from the pack's Section 0 into a writable run dir, and
every host mount is read-only.
"""
import json
import shlex
from pathlib import Path

from .. import catalog
from ..downloads import safe_artifact
from ..paths import IS_WIN
from . import common

# container-side layout
ROOT_IN = "/strata"
MODELS_IN = "/models"
RUN_IN = "/tmp/b70-strata"
TOK_IN = f"{RUN_IN}/tokenizer"  # serve auto-generates vocab.json from the pack here

# recipe paths inside the checkout
PATH_DEFAULTS = {"hy4_bin": "build-hy4-sycl-bmg/strata-hy4-sycl",
                 "hy4_python": ".venv-sycl/bin/python",
                 "hy4_server": "engine/sycl/serve/server_intel.py"}
ENGRAM_DEFAULT = "Hy4-Flash-Engram-Q8.strata"

# Verified dual-B70 expert budgets by context rung (cookbook
# docs/hy4-flash/HY4-FLASH-STRATA-SYCL.md capacity ladder): the primary card
# trades expert VRAM for KV as context deepens; the peer always holds 25 GiB.
VRAM_BY_CTX_DEFAULT = {32768: 17, 131072: 15, 262144: 14}
KV_HOST_ABOVE_MIN = 131072
DRAFTS = ("dfly2", "off")
ENV_PREFIXES = ("HY4S_", "STRATA_")
IMAGE_HINT = ("Build it in your Strata v0.2.20 checkout: docker build -t {image} engine/sycl/tools "
              "(Niko1221/Strata v0.2.20 + patches/hy4-flash-strata-sycl/strata-hy4-b70-opt.patch).")


def _rel(recipe, key):
    value = str(recipe.get(key, PATH_DEFAULTS[key]) or "")
    if value and not safe_artifact(value):
        raise common.BuildError(f"recipe {key} must be a relative path inside the Strata checkout, not {value!r}")
    return value


def _basename(recipe, key, default):
    value = str(recipe.get(key) or default).strip()
    if not value or Path(value).name != value:
        raise common.BuildError(f"recipe {key} must be a bare file name, not {value!r}")
    return value


def _num(recipe, key, default, lo, hi, integer=False):
    try:
        v = recipe.get(key, default)
        v = int(v) if integer else float(v)
    except (TypeError, ValueError):
        raise common.BuildError(f"recipe {key} must be a number")
    if not lo <= v <= hi:
        raise common.BuildError(f"recipe {key} out of range")
    return v


def _root():
    for key in ("strata_hy4_root", "strata_root"):
        raw = str(catalog.SETTINGS.get(key) or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute():  # docker -v would take a relative name for a volume
            raise common.BuildError(f'setting "{key}" must be an absolute path, not {raw!r}')
        return p, key
    return None, "strata_hy4_root"


def _vram_experts_gb(recipe, ctx):
    """Primary-card expert budget for the context rung (the verified ladder)."""
    ladder = recipe.get("vram_experts_by_ctx")
    if ladder is None:
        return _num(recipe, "vram_experts_gb", 17, 0, 30)
    if not isinstance(ladder, dict) or not ladder:
        raise common.BuildError("recipe vram_experts_by_ctx must map context ceilings to GiB budgets")
    rungs = []
    for ceil, gb in ladder.items():
        try:
            rungs.append((int(ceil), float(gb)))
        except (TypeError, ValueError):
            raise common.BuildError("recipe vram_experts_by_ctx must map context ceilings to GiB budgets")
    rungs.sort()
    for ceil, gb in rungs:
        if ctx <= ceil:
            return gb
    return rungs[-1][1]


def _gpu_order(b):
    """--gpus device order: primary (compute) first, peers after — a card that
    drives the desktop is always a peer, never the primary. With
    ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 the mapped cards enumerate in host PCI
    order, so the flag names host GPU indices."""
    disp = {g for g, _d in b.display_gpus}
    plain = [g for g in sorted(b.gpus) if g not in disp]
    peers = [g for g in sorted(b.gpus) if g in disp]
    return plain[-1:] + plain[:-1] + peers


def _display_budget_cap(b, host_gpu):
    """GiB ceiling for an engine VRAM budget on a card that drives the desktop
    (the same gmu fraction other engines get), or None when not capped."""
    if b.gmu_cap is None:
        return None
    dev = {g: d for g, d in b.display_gpus}.get(host_gpu)
    if not dev:
        return None
    total = dev.get("vram_total_mib")
    return b.gmu_cap * total / 1024.0 if total else None


def _engine_env(recipe, b):
    """Container env: device + runtime basics, recipe HY4S_*/STRATA_* settings,
    then the caller's extra_env (wins). Recipe env is local only — the remote
    overlay never ships "env" (catalog.REMOTE_RECIPE_FIELDS)."""
    env = {
        # PCI-id order is the recipe's GPU numbering contract (--gpus names
        # primary first); level_zero:gpu selects every mapped card
        "ZE_ENABLE_PCI_ID_DEVICE_ORDER": "1",
        "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
        "SYCL_CACHE_PERSISTENT": "0",  # the persistent JIT cache segfaults on Xe2
        "ZES_ENABLE_SYSMAN": "1",
    }
    for k, v in (recipe.get("env") or {}).items():
        if not any(str(k).startswith(p) for p in ENV_PREFIXES):
            raise common.BuildError(f"recipe env {k} is not a HY4S_* or STRATA_* engine setting")
        if isinstance(v, (dict, list)) or v is None or "\n" in str(v):
            raise common.BuildError(f"recipe env {k} must be a single-line value")
        env[k] = str(v)
    env.update(b.extra_env)
    return env


def build_strata_hy4(b):
    cfg, model, recipe, det, warns = b.cfg, b.model, b.recipe, b.det, b.warns
    if IS_WIN:
        raise common.BuildError("The Hy4-Flash SYCL engine runs on Linux only.")
    paths = {k: _rel(recipe, k) for k in PATH_DEFAULTS}
    engram_name = _basename(recipe, "engram_file", ENGRAM_DEFAULT)
    draft = str(recipe.get("draft") or "dfly2").strip().lower()
    if draft not in DRAFTS:
        raise common.BuildError(f"recipe draft must be one of {', '.join(DRAFTS)}")
    if cfg.get("mtp") is False:
        draft = "off"
        warns.append("DFly2 speculative decoding disabled by --no-mtp.")
    elif cfg.get("mtp") is True and draft != "dfly2":
        draft = "dfly2"
    if b.slots > 1:
        warns.append("the verified Hy4-Flash recipe serves one request at a time — using parallel 1.")
    threads = _num(recipe, "threads", 8, 1, 128, integer=True)
    ram_gb = _num(recipe, "ram_experts_gb", 36, 0, 512)
    peer_gb = _num(recipe, "peer_vram_experts_gb", 25, -1, 30)
    vram_gb = _vram_experts_gb(recipe, b.ctx)
    kv_host_above = max(_num(recipe, "kv_host_above", KV_HOST_ABOVE_MIN, 0, 1048576, integer=True), b.ctx)
    missing = []
    root, root_key = _root()
    if root is None:
        missing.append(f'the Strata v0.2.20 checkout: set "{root_key}" in settings-override.json')
        root = Path("~/Strata").expanduser()  # preview placeholder
    engine = root / paths["hy4_bin"]
    server = root / paths["hy4_server"]
    venv = (root / paths["hy4_python"]).parent.parent
    # an unset checkout is the one thing to say; its placeholder files are not checked
    checks = [] if missing else [
        (server.is_file(), f"{server} (the Strata checkout's SYCL serve front-end)"),
        (engine.is_file(), f"the engine {engine} (engine/src/hy4flash/sycl/build_hy4_sycl.sh bmg)"),
        ((venv / "pyvenv.cfg").is_file(), f"the server's Python environment {venv}, made in the image"),
    ]
    missing += [what for ok, what in checks if not ok]
    # the native .strata pack and the mandatory Engram shard sit side by side
    dl = recipe.get("download") or {}
    if det and det.get("path"):
        first = Path(det["path"])
    else:  # preview: where a launcher download puts it
        models_dir = cfg.get("models_dir") or catalog.SETTINGS.get("models_dir", "~/models")
        first = Path(models_dir).expanduser() / (dl.get("name") or "model.strata")
    if first.suffix.lower() != ".strata":
        raise common.BuildError(f"{first.name} is not a Hy4-Flash .strata pack")
    engram = first.with_name(engram_name)
    if not det:
        missing.append(f"the model file {first.name} (download it or add its folder to the model folders)")
    elif not engram.exists():
        missing.append(f"the Engram shard {engram.name} next to {first.name}")
    if missing:
        text = "Strata Hy4-Flash recipe needs " + "; ".join(missing)
        if not cfg.get("dry_run"):
            raise common.BuildError(text + " (cookbook: docs/hy4-flash/HY4-FLASH-STRATA-SYCL.md).")
        warns.append(text + " — preview only; a real launch is refused until they exist.")
    common.resolve_kv(b.kv, recipe, ("fp16",), warns)  # the engine serves fp16 KV only
    order = _gpu_order(b)
    primary_gpu, peer_gpus = order[0], order[1:]
    cap = _display_budget_cap(b, primary_gpu)
    if cap is not None and vram_gb > int(cap):
        warns.append(f"primary GPU {primary_gpu} drives the desktop: expert budget "
                     f"{vram_gb:g} → {int(cap)} GiB (display reserve).")
        vram_gb = int(cap)
    if peer_gpus and peer_gb >= 0:
        cap = _display_budget_cap(b, peer_gpus[0])
        if cap is not None and peer_gb > int(cap):
            warns.append(f"peer GPU {peer_gpus[0]} drives the desktop: expert budget "
                         f"{peer_gb:g} → {int(cap)} GiB (display reserve).")
            peer_gb = int(cap)
    args = ["--native", f"{MODELS_IN}/{first.name}", "--engram", f"{MODELS_IN}/{engram.name}",
            "--max-context", str(b.ctx), "--session-reuse", "1" if b.prefix_cache else "0",
            "--draft", draft, "--threads", str(threads), "--gpus", ",".join(map(str, order)),
            "--vram-experts-gb", f"{vram_gb:g}", "--ram-experts-gb", f"{ram_gb:g}",
            "--kv-host-above", str(kv_host_above)]
    if peer_gb >= 0:
        args += ["--peer-vram-experts-gb", f"{peer_gb:g}"]
    args += list(recipe.get("fixed_flags", [])) + b.extra
    config = {"arch": "hy4flash", "backend": "sycl", "exe": f"{ROOT_IN}/{paths['hy4_bin']}",
              "cwd": ROOT_IN, "tokenizer": TOK_IN, "model_name": model["name"],
              "gpu": list(range(len(b.gpus))), "gpu_order": "as_given", "parallel": 1,
              "drafting": draft, "log": f"{RUN_IN}/engine.log", "args": args}
    serve = [f"{ROOT_IN}/{paths['hy4_python']}", paths["hy4_server"], "--engine", "strata",
             "--config", f"{RUN_IN}/config.json", "--host", "0.0.0.0", "--port", str(b.port)]
    # the engine log is tailed into the container output, so it lands in the launcher's server log
    script = (f"if [ -f /opt/intel/oneapi/setvars.sh ]; then . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; fi; "
              f"mkdir -p {RUN_IN} {TOK_IN} && printf '%s' {shlex.quote(json.dumps(config))} > {RUN_IN}/config.json "
              f"&& : > {RUN_IN}/engine.log; tail -n +1 -F {RUN_IN}/engine.log & "
              f"cd {ROOT_IN} && exec " + " ".join(shlex.quote(a) for a in serve))
    image = recipe["image"]
    cname = common.cname_for(model["id"], "strata-hy4", b.port)
    tokens = ["docker", "run", "--rm", "--name", cname] + common.labels_for(cfg, b.port)
    tokens += common.dri_args(b)
    if len(b.gpus) > 1:
        # Level Zero IPC between the two cards under Docker's default seccomp (Strata PR #2468)
        tokens += ["--cap-add", "SYS_PTRACE"]
    tokens += ["--ulimit", "memlock=-1", "--oom-score-adj", "1000",
               "-p", f"127.0.0.1:{b.port}:{b.port}",
               "-v", f"{root}:{ROOT_IN}:ro"]
    for pack in (first, engram):
        tokens += ["-v", f"{pack.resolve() if pack.exists() else pack}:{MODELS_IN}/{pack.name}:ro"]
    for k, v in _engine_env(recipe, b).items():
        tokens += ["-e", f"{k}={v}"]
    tokens += ["--entrypoint", "bash", image, "-lc", script]
    warns.append(f"Strata Hy4-Flash runs from your checkout {root} in the {image} image; the launcher "
                 "does not build or ship Strata.")
    if len(b.gpus) > 1:
        warns.append(f"Two-card expert parallel (--gpus {','.join(map(str, order))}): primary on GPU "
                     f"{primary_gpu}, peer experts on GPU {peer_gpus[0]}.")
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": model["name"],
            "native": False, "docker_sock": "", "effective_binary": image,
            "image_hint": IMAGE_HINT.format(image=image)}
