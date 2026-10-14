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
    # Context profile dispatch:
    # 1. Explicit user/operator override in cfg or recipe takes precedence.
    # 2. If recipe defines gmu_safe (e.g. "0.90") and ctx <= ctx_safe (e.g. 131,072),
    #    dispatches to gmu_safe for the >=2 GiB high-reserve cell.
    # 3. Otherwise uses recipe["gmu"] (e.g. "0.94" for long-ctx single-request profile).
    # 4. Fallback when neither is defined: 0.94 if ctx >= 131072 else 0.90.
    # 5. Guard: if ctx > 168000, gmu must be at least 0.94 to avoid container pool aborts.
    user_override = recipe.get("override_gmu") or cfg.get("gmu")
    if user_override:
        gmu = str(user_override)
    elif recipe.get("gmu_safe") and b.ctx <= recipe.get("ctx_safe", 131072):
        gmu = str(recipe["gmu_safe"])
    elif recipe.get("gmu"):
        gmu = str(recipe["gmu"])
    else:
        gmu = "0.94" if b.ctx >= 131072 else "0.90"
    if b.ctx > 168000 and float(gmu) < 0.94:
        gmu = "0.94"
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
    warns.append(f"EXL3 trellis + native MTP3 on the isolated dockerd; fp8 KV only; gmu {gmu}; "
                 "first load ~5-8 min for kernel compile.")
    if recipe.get("gmu_safe"):
        if b.ctx > recipe.get("ctx_safe", 131072):
            warns.append(f"ctx {b.ctx} operates with low VRAM headroom (observation class, ~1.1 GiB free under load); "
                         f"sized for single-client capacity scope. Safe high-reserve alternate is {recipe.get('ctx_safe', 131072)} at gmu {recipe['gmu_safe']}.")
        elif float(gmu) >= 0.94:
            warns.append(f"gmu {gmu} operates with low VRAM headroom (~1.1 GiB free under load, observation class); "
                         f"high-reserve alternate requires gmu {recipe['gmu_safe']}.")
    if recipe.get("ctx_max") and b.ctx > 196608:
        warns.append(f"ctx {b.ctx} exceeds the 196,608 tested profile (unverified budget ceiling up to {recipe['ctx_max']}).")
    if b.slots > 1:
        warns.append("EXL3 manages concurrency internally; slots is advisory here.")
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": model["name"],
            "native": False, "docker_sock": dsock}
