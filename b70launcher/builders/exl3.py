"""EXL3 (exl3xpu, isolated dockerd) builder.

Prefix caching stays ON here — the EXL3 route is verified multi-turn with the
cache enabled (unlike the vLLM-XPU speculative stacks).
"""
import json
from pathlib import Path

from . import common
from ..paths import IS_WIN


def build_exl3(b):
    cfg, model, recipe, warns = b.cfg, b.model, b.recipe, b.warns
    if IS_WIN:
        raise common.BuildError("EXL3 XPU is Linux-only (isolated dockerd on /dev/dri).")
    model_dir = (b.det or {}).get("path") or recipe.get("model_path", "")
    if model_dir and Path(str(model_dir)).expanduser().exists():
        model_dir = str(Path(str(model_dir)).expanduser())
    elif not cfg.get("dry_run"):
        raise common.BuildError("EXL3 artifact not found on disk — point a model folder at it "
                                "or set the recipe's model_path.")
    else:
        # preview always renders (audit C1): placeholder mount + warning
        model_dir = str(Path(cfg.get("models_dir") or "~/models").expanduser())
        warns.append("EXL3 artifact not found on disk — preview mounts a placeholder; "
                     "a real launch is refused until the artifact exists.")
    common.resolve_kv(b.kv, recipe, ("fp8",), warns,
                      label="--kv (EXL3 serves fp8 KV internally)")
    dsock = recipe.get("docker_sock", "")
    docker_cmd = ["docker"] + (["-H", f"unix://{dsock}"] if dsock else [])
    if dsock and cfg.get("dry_run"):
        # previews never touch dockerd — bring-up is a launch-time step only
        warns.append("isolated exl3 dockerd auto-starts on launch (needs passwordless sudo)")
    inner_port = int(recipe.get("inner_port", 8100))
    if recipe.get("gmu"):
        gmu = str(recipe["gmu"])
    elif model.get("id") == "qwen38-27b-sc5":
        gmu = "0.90"
    else:
        gmu = "0.94" if b.ctx >= 131072 else "0.90"
    cname = common.cname_for(model["id"], "exl3", b.port)
    tokens = docker_cmd + ["run", "--rm", "--name", cname] + common.labels_for(cfg, b.port)
    tokens += common.dri_args(b) + [
               "-v", f"{model_dir}:/mnt/exl3model:ro",
               "-p", f"127.0.0.1:{b.port}:{inner_port}",
               # weight upload on the copy engine (BCS) hit "Engine memory CAT
               # error class=bcs" hangs in 2 of 4 loads on 2026-10-13; with
               # copies on the compute engine 4 of 4 loaded and decode matched
               # (70.24 vs 70.3 tok/s greedy p512/g128, RL-198)
               "-e", "UR_L0_USE_COPY_ENGINE=0",
               "-e", "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0"] + b.envargs + [
        recipe["image"],
        recipe.get("serve_config", "models/qwen3.8-27b-exl3-4.00bpw"),
        "--gpu", "0",  # only the selected card is mapped into the container
        "--model-path", "/mnt/exl3model",
        "--set", f"vllm.gpu_memory_utilization={gmu}",
        "--set", f"vllm.max_model_len={b.ctx}",
        "--set", "vllm.enable_prefix_caching=true",
        # serve under the name the launcher hands to clients (default is the config id)
        "--set", f"served_model_name={json.dumps(model['name'])}",
        # same parsers as exl3xpu's published docker recipe: thinking goes to
        # message.reasoning, tool calls are parsed
        "--set", "vllm.reasoning_parser=qwen3",
        "--set", "vllm.enable_auto_tool_choice=true",
        "--set", "vllm.tool_call_parser=qwen3_coder"]
    tokens += recipe.get("fixed_flags", []) + b.extra
    warns.append(f"EXL3 trellis + native MTP3 on the isolated dockerd; fp8 KV only; gmu {gmu} "
                 "(0.90 for SC5 / 0.94 at ≥128K on plain 4bpw); first load ~5-8 min for kernel compile.")
    if model.get("id") == "qwen38-27b-sc5" and b.ctx > 131072:
        warns.append(f"ctx {b.ctx} exceeds the 131,072 validated ceiling for SC5 (tested with ≥2 GiB free VRAM); "
                     "values up to the 168,000 engine pool budget are unverified.")
    if b.slots > 1:
        warns.append("EXL3 manages concurrency internally; slots is advisory here.")
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": model["name"],
            "native": False, "docker_sock": dsock}
