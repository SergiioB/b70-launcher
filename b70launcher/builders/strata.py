"""Strata SYCL builder: the user's own Strata checkout in their strata-sycl-dev image.

The launcher does not build or ship Strata. A strata recipe names the files
Strata's setup produces (the pack, the MTP draft layer) relative to the Strata
data folder, and the engine, the server's Python and the data files relative to
the checkout (settings ``strata_root`` / ``strata_data``). The server
(sycl/serve/server_intel.py) and the engine run in one container: the server
starts the engine binary itself (no nested docker as in Strata's
strata-sycl.sh), the config is rendered here and written inside the container
at start, and every host mount is read-only.
"""
import json
import re
import shlex
from pathlib import Path

from .. import catalog
from ..downloads import safe_artifact
from ..paths import IS_WIN
from . import common

# container-side layout
ROOT_IN = "/strata"
PACK_IN = "/strata-data/pack"
MTP_IN = "/strata-data/mtp"
MODELS_IN = "/models"
RUN_IN = "/tmp/b70-strata"

# recipe paths: inside the checkout (strata_bin, strata_python, expert_profile,
# mtp_draft_vocab) or the data folder (strata_pack, strata_mtp)
PATH_DEFAULTS = {"strata_bin": "build-sycl-aot/strata", "strata_python": ".venv-sycl/bin/python",
                 "expert_profile": "data/expert-profile-sycl.bin", "mtp_draft_vocab": "",
                 "strata_pack": "packs/iq3_xxs", "strata_mtp": "mtp/rt"}
PARALLEL_MAX = 8  # the engine's batch window (serve/server.py PARALLEL_MAX)
SHARD1 = re.compile(r"-00001-of-00002\.gguf$", re.I)
IMAGE_HINT = ("Build it in your Strata checkout: docker build -t {image} sycl/tools "
              "(Strata docs/INTEL.md, \"How to build it\").")


def _rel(recipe, key):
    value = str(recipe.get(key, PATH_DEFAULTS[key]) or "")
    if value and not safe_artifact(value):
        raise common.BuildError(f"recipe {key} must be a relative path inside the Strata folder, not {value!r}")
    return value


def _setting(key):
    raw = str(catalog.SETTINGS.get(key) or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():  # docker -v would take a relative name for a volume
        raise common.BuildError(f'setting "{key}" must be an absolute path, not {raw!r}')
    return p


def _engine_env(recipe, b):
    """Container env: device + runtime basics, recipe STRATA_* settings, then the
    caller's extra_env (wins). Recipe env is local only — the remote overlay never
    ships "env" (catalog.REMOTE_RECIPE_FIELDS)."""
    env = {
        # only the selected cards are mapped into the container, so "*" is
        # exactly them (Strata docs/INTEL.md, expert parallel); the image's own
        # default is level_zero:0
        "ONEAPI_DEVICE_SELECTOR": "level_zero:*",
        "SYCL_CACHE_PERSISTENT": "0",  # the persistent JIT cache segfaults on Xe2
        "ZES_ENABLE_SYSMAN": "1",
    }
    for k, v in (recipe.get("env") or {}).items():
        if not re.fullmatch(r"STRATA_[A-Z0-9_]+", str(k)):
            raise common.BuildError(f"recipe env {k} is not a STRATA_* engine setting")
        if isinstance(v, (dict, list)) or v is None or "\n" in str(v):
            raise common.BuildError(f"recipe env {k} must be a single-line value")
        env[k] = str(v)
    env.update(b.extra_env)
    return env


def build_strata(b):
    cfg, model, recipe, det, warns = b.cfg, b.model, b.recipe, b.det, b.warns
    if IS_WIN:
        raise common.BuildError("Strata's SYCL port runs on Linux only.")
    paths = {k: _rel(recipe, k) for k in PATH_DEFAULTS}
    missing = []
    root = _setting("strata_root")
    if root is None:
        missing.append('the Strata folder: set "strata_root" in settings-override.json')
        root = Path("~/Strata").expanduser()  # preview placeholder
    data = _setting("strata_data") or root.parent / "Strata-data"  # Strata setup's default
    engine, server = root / paths["strata_bin"], root / "sycl" / "serve" / "server_intel.py"
    venv = (root / paths["strata_python"]).parent.parent
    profile, pack, mtp = root / paths["expert_profile"], data / paths["strata_pack"], data / paths["strata_mtp"]
    vocab = paths["mtp_draft_vocab"]
    # spec_tokens / spec_p_min may come from the update channel: numbers only
    try:
        parallel = b.slots if cfg.get("slots") is not None else int(recipe.get("parallel") or 1)
        spec = int(recipe.get("spec_tokens") or 0)
        p_min = float(recipe.get("spec_p_min", 0.5))
    except (TypeError, ValueError):
        raise common.BuildError("recipe parallel, spec_tokens and spec_p_min must be numbers")
    if parallel < 1 or not 0 <= spec <= 16 or not 0 <= p_min <= 1:
        raise common.BuildError("recipe parallel, spec_tokens or spec_p_min out of range")
    use_mtp = spec > 0 and cfg.get("mtp", True)
    if cfg.get("mtp") is False and spec:
        warns.append("MTP speculative decoding disabled by --no-mtp.")
    # an unset strata_root is the one thing to say; its placeholder files are not checked
    checks = [] if missing else [
        (server.is_file(), f"{server} (strata_root must be a Strata checkout)"),
        (engine.is_file(), f"the engine {engine} (Strata docs/INTEL.md, \"How to build it\")"),
        ((venv / "pyvenv.cfg").is_file(), f"the server's Python environment {venv}, made in the image"),
        (profile.is_file(), f"the expert profile {profile}"),
        ((pack / "native_experts.txt").is_file() and (pack / "tokenizer" / "vocab.json").is_file(),
         f"the pack {pack} (Strata's setup writes it)"),
        (not use_mtp or (mtp / "experts.bin").is_file(),
         f"the MTP draft layer {mtp} (Strata's setup writes it; --no-mtp runs without)"),
        (not (use_mtp and vocab) or (root / vocab).is_file(), f"the draft vocabulary {root / vocab}")]
    missing += [what for ok, what in checks if not ok]
    # the GGUF is two shards: 1 holds the model, 2 the per-layer lookup table (--ple-gguf)
    dl = recipe.get("download") or {}
    if det and det.get("path"):
        first = Path(det["path"])
    else:  # preview: where a launcher download puts it
        models_dir = cfg.get("models_dir") or catalog.SETTINGS.get("models_dir", "~/models")
        first = Path(models_dir).expanduser() / (dl.get("name") or "model.gguf")
    if not SHARD1.search(first.name):
        raise common.BuildError(f"{first.name} is not shard 1 of a two-shard Strata GGUF "
                                "(-00001-of-00002.gguf)")
    second = first.with_name(SHARD1.sub("-00002-of-00002.gguf", first.name))
    if not det:
        missing.append(f"the model file {first.name} (download it or add its folder to the model folders)")
    elif not second.exists():
        missing.append(f"the second shard {second.name} next to {first.name}")
    if missing:
        text = "Strata recipe needs " + "; ".join(missing)
        if not cfg.get("dry_run"):
            raise common.BuildError(text + " (README: Strata).")
        warns.append(text + " — preview only; a real launch is refused until they exist.")
    kv = common.resolve_kv(b.kv, recipe, ("int8",), warns)
    if parallel > PARALLEL_MAX:
        warns.append(f"Strata runs at most {PARALLEL_MAX} requests together — using {PARALLEL_MAX}.")
        parallel = PARALLEL_MAX
    ep = len(b.gpus) > 1
    args = ["--pack", PACK_IN, "--native", f"{MODELS_IN}/{first.name}", "--ple-gguf", f"{MODELS_IN}/{second.name}"]
    if ep:
        # card 0 runs every layer, the other card computes its share of each layer's experts
        args += ["--expert-parallel-device", "1"]
    args += ["--expert-profile", f"{ROOT_IN}/{paths['expert_profile']}",
             "--max-context", str(b.ctx), "--kv", kv]
    if use_mtp:
        args += ["--spec", str(spec), "--mtp", MTP_IN, "--spec-min-p", f"{p_min:g}"]
        if vocab:
            args += ["--mtp-draft-vocab", f"{ROOT_IN}/{vocab}"]
    args += list(recipe.get("fixed_flags", []))
    if not b.prefix_cache:
        # the engine keeps the last value of a repeated flag
        args += ["--prompt-cache", "0", "--conversation-cache-mib", "0"]
        warns.append("Strata prefix cache off: --prompt-cache 0 and no conversation parking.")
    args += b.extra
    config = {"backend": "sycl", "exe": f"{ROOT_IN}/{paths['strata_bin']}", "cwd": ROOT_IN,
              "tokenizer": f"{PACK_IN}/tokenizer", "model_name": model["name"],
              "gpu": list(range(len(b.gpus))), "gpu_order": "as_given", "parallel": parallel,
              "log": f"{RUN_IN}/engine.log", "args": args}
    serve = [f"{ROOT_IN}/{paths['strata_python']}", "sycl/serve/server_intel.py", "--engine", "strata",
             "--config", f"{RUN_IN}/config.json", "--host", "0.0.0.0", "--port", str(b.port)]
    # the engine log is tailed into the container output, so it lands in the launcher's server log
    script = (f"if [ -f /opt/intel/oneapi/setvars.sh ]; then . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; fi; "
              f"mkdir -p {RUN_IN} && printf '%s' {shlex.quote(json.dumps(config))} > {RUN_IN}/config.json "
              f"&& : > {RUN_IN}/engine.log; tail -n +1 -F {RUN_IN}/engine.log & "
              f"cd {ROOT_IN} && exec " + " ".join(shlex.quote(a) for a in serve))
    image = recipe["image"]
    cname = common.cname_for(model["id"], "strata", b.port)
    tokens = ["docker", "run", "--rm", "--name", cname] + common.labels_for(cfg, b.port)
    tokens += common.dri_args(b)
    if ep:
        # Level Zero IPC between the two cards under Docker's default seccomp (Strata PR #2468)
        tokens += ["--cap-add", "SYS_PTRACE"]
    tokens += ["--ulimit", "memlock=-1", "--oom-score-adj", "1000",
               "-p", f"127.0.0.1:{b.port}:{b.port}",
               "-v", f"{root}:{ROOT_IN}:ro", "-v", f"{pack}:{PACK_IN}:ro"]
    if use_mtp:
        tokens += ["-v", f"{mtp}:{MTP_IN}:ro"]
    # each shard mounted by its real path: shard 2 is often a link to the copy another size shares
    for shard in (first, second):
        tokens += ["-v", f"{shard.resolve() if shard.exists() else shard}:{MODELS_IN}/{shard.name}:ro"]
    for k, v in _engine_env(recipe, b).items():
        tokens += ["-e", f"{k}={v}"]
    tokens += ["--entrypoint", "bash", image, "-lc", script]
    warns.append(f"Strata runs from your checkout {root} in the {image} image; the launcher does not build "
                 "or ship Strata.")
    if ep:
        warns.append("Two-card expert parallel (--expert-parallel-device 1): every expert in VRAM across both cards.")
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": model["name"],
            "native": False, "docker_sock": "", "effective_binary": image,
            "image_hint": IMAGE_HINT.format(image=image)}
