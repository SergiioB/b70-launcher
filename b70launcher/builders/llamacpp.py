"""llama.cpp SYCL builder (gguf + gguf-tiered), container or native."""
import glob
import os
import re
from pathlib import Path

from .. import catalog, scan
from ..paths import IS_WIN
from . import common

KV_MAP = {"q5_0/q4_1": ["--cache-type-k", "q5_0", "--cache-type-v", "q4_1"],
          "q8_0": ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
          "q8_0/q4_1": ["--cache-type-k", "q8_0", "--cache-type-v", "q4_1"],
          "f16": ["--cache-type-k", "f16", "--cache-type-v", "f16"]}
# cookbook fleet default: q8_0 K + q4_1 V (~half the V memory of q8_0/q8_0)
KV_DEFAULT_ORDER = ("q8_0/q4_1", "q8_0", "q5_0/q4_1", "f16")


def build_llamacpp(b):
    cfg, model, recipe, det, warns = b.cfg, b.model, b.recipe, b.det, b.warns
    fallback = recipe.get("gguf", "/models/model.gguf")
    models_dir = cfg.get("models_dir") or catalog.SETTINGS.get("models_dir", "~/models")
    if det:
        gpath, mount_src, mount_dst = common.container_path(det, fallback)
        if mount_src is None:
            mount_src = models_dir
            mount_dst = "/models"
        host_gpath = det.get("path") or gpath  # native mode needs the host path
    else:
        gpath = fallback
        # a native binary cannot read a container path: assume the artifact
        # lands where a download would put it (recipe-audit #10)
        host_gpath = str(Path(models_dir).expanduser() / Path(fallback).name)
        mount_src = models_dir
        mount_dst = "/models"
        warns.append("model not detected on disk — preview assumes the artifact at "
                     f"{host_gpath}; a real launch needs it there (download it or add its folder to the model folders list).")
    kv = common.resolve_kv(b.kv, recipe, KV_DEFAULT_ORDER, warns)
    if kv not in KV_MAP:
        warns.append(f"--kv '{kv}' isn't a llama.cpp cache type ({', '.join(KV_MAP)}) — using q8_0/q4_1.")
        kv = "q8_0/q4_1"
    if kv == "f16":
        warns.append("f16 KV wastes VRAM with zero quality gain on these workloads. q8_0 K + q4_1 V recommended.")
    llama_bin = (recipe.get("llama_bin") or catalog.SETTINGS.get("llama_bin") or "").strip()
    if llama_bin:
        llama_bin = str(Path(llama_bin).expanduser())
    effective_binary = recipe.get("image", "")
    cname = ""
    # recipe env: engine tuning knobs only (GGML_SYCL_*). Local recipes.json only — the
    # remote overlay never ships "env" (catalog.REMOTE_RECIPE_FIELDS); user extra_env wins.
    recipe_env = recipe.get("env") or {}
    for k in recipe_env:
        if not re.fullmatch(r"GGML_SYCL_[A-Z0-9_]+", k):
            raise common.BuildError(f"recipe env {k} is not a GGML_SYCL_* engine setting")
    if (llama_bin and Path(llama_bin).is_file()) and not cfg.get("use_docker"):
        if not det:
            warns.append("native binary will use the assumed models_dir path — "
                         "point a model folder at the real file or pass --path.")
        tokens = [llama_bin]
        b.native = True
        effective_binary = llama_bin
        if not recipe.get("llama_bin"):
            warns.append(f"local llama_bin from settings overrides the recipe's pinned image "
                         f"({recipe.get('image', '?')}) — running binary: {llama_bin}")
        model_arg = host_gpath
        b.env.update({k: str(v) for k, v in recipe_env.items()})
        b.env.update(b.extra_env)
        # mask renumbers visible devices 0..n-1 -> selector is relative (H5)
        b.env.update(common.device_env(b.gpus))
        b.env.update({
            "ZES_ENABLE_SYSMAN": "1", "SYCL_CACHE_PERSISTENT": "0",
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "0" if b.kind == "gguf-tiered" else "1",
            "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
            "SYCL_DEVICE_FILTER": "level_zero",
            "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
        })
        # oneAPI runtime libs (libsvml, libdnnl, libsycl, …): a fresh user
        # won't have `source setvars.sh` in their shell env, and the SYCL
        # build fails to load without them. The build folder comes first:
        # llama.cpp ships libllama*.so next to the binary and its RUNPATH is
        # absolute, so it breaks when the build folder moves.
        oneapi_dirs = [p for p in sorted(glob.glob("/opt/intel/oneapi/*/latest/lib"))
                       if Path(p).is_dir()]
        if not oneapi_dirs:
            warns.append("Native llama.cpp: oneAPI libs not found under /opt/intel/oneapi — run inside `source /opt/intel/oneapi/setvars.sh` shell if the binary fails to start.")
        ldp = b.env.get("LD_LIBRARY_PATH") or os.environ.get("LD_LIBRARY_PATH") or ""
        b.env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(Path(llama_bin).parent)] + oneapi_dirs + ([ldp] if ldp else []))
        if b.kind == "gguf-tiered":
            b.env["LLAMA_ATTN_ROT_DISABLE"] = "1"
        host = "127.0.0.1"  # native: never expose an unauthenticated server to the LAN
    else:
        host = "0.0.0.0"  # inside the container; only 127.0.0.1 is published
        cname = common.cname_for(model["id"], "sycl", b.port)
        tokens = ["docker", "run", "--rm", "--name", cname] + common.labels_for(cfg, b.port)
        if not IS_WIN:
            tokens += common.dri_args(b)
        tokens += ["--ipc=host", "-v", f"{Path(str(mount_src)).expanduser()}:{mount_dst}:ro",
                   "-p", f"127.0.0.1:{b.port}:{b.port}", "-e", "ZES_ENABLE_SYSMAN=1"] + [
            x for k, v in recipe_env.items() for x in ("-e", f"{k}={v}")] + b.envargs + [
            recipe["image"]]
        model_arg = gpath
    # --alias: /v1/models must report the short name the agents are configured
    # with, not the full .gguf path
    tokens += ["-m", model_arg, "--alias", Path(gpath).stem, "-ngl", "99", "--host", host, "--port", str(b.port),
               "-c", str(b.ctx), "--flash-attn", "on", "--metrics"]
    if not b.prefix_cache:
        tokens += ["--no-cache-prompt"]  # llama-server caches prompts by default
    if "--cache-type-k" not in recipe.get("fixed_flags", []):
        tokens += KV_MAP[kv]
    if len(b.gpus) > 1:
        ts = recipe.get("tensor_split", "49,51" if len(b.gpus) == 2 else ",".join(["1"] * len(b.gpus)))
        # natively the mask renumbers devices 0..n-1; a container sees only the
        # selected cards, which also enumerate as 0..n-1
        devs = ",".join(f"SYCL{i}" for i in range(len(b.gpus)))
        tokens += ["--device", devs,
                   "--tensor-split", ts,
                   "--split-mode", recipe.get("split_mode", "layer")]
        warns.append(f"Dual-GPU tensor split: {ts} (split-mode: {recipe.get('split_mode', 'layer')}).")
    elif recipe.get("tensor_split"):
        warns.append("recipe is tuned for a dual-GPU tensor split — running single-GPU (split disabled).")
    elif not b.native:
        # H4: pin exactly one device (the only one mapped into the container)
        tokens += ["--device", "SYCL0"]
    if recipe.get("offload_tensors"):
        tokens += ["-ot", recipe["offload_tensors"]]
        warns.append("3-Tiered Memory: N-gram embedding & MoE boundary blocks offloaded to CPU host RAM (-ot).")
    draft = scan._resolve_draft(recipe.get("draft_model"))
    if draft and not cfg.get("mtp", True):
        warns.append("MTP speculative draft disabled by --no-mtp.")
        draft = None
    if cfg.get("mtp") and not recipe.get("draft_model"):
        warns.append("recipe has no draft model — --mtp ignored.")
    if draft:
        # M5: on a single GPU the draft shares the serving card, which is
        # SYCL0 both natively (mask) and in a container (only card mapped)
        if len(b.gpus) == 1:
            dd = "SYCL0"
            if dd != recipe.get("draft_device", "SYCL1"):
                warns.append(f"single GPU selected — draft moved onto {dd}.")
        else:
            dd = recipe.get("draft_device", "SYCL1")
        draft_arg = draft if b.native else f"/draft/{Path(draft).name}"
        if not b.native:
            at = tokens.index(recipe["image"]) if recipe["image"] in tokens else len(tokens)
            tokens[at:at] = ["-v", f"{Path(draft).parent}:/draft:ro"]
        tokens += ["-md", draft_arg, "-ngld", "999",
                   "--spec-type", "draft-mtp",
                   "--spec-draft-n-max", str(recipe.get("spec_tokens", 3)),
                   "--spec-draft-p-min", str(recipe.get("spec_p_min", 0.75)),
                   "--spec-draft-backend-sampling",
                   "--spec-draft-device", dd]
        warns.append(f"MTP Speculative Decoding: draft model {Path(draft).name} on {dd}.")
    if b.slots > 1:
        tokens += ["-np", str(b.slots)]
    tokens += recipe.get("fixed_flags", []) + b.extra
    return {"tokens": tokens, "env": b.env, "cname": cname,
            "served_name": Path(gpath).stem, "native": b.native,
            "docker_sock": "", "effective_binary": effective_binary}
