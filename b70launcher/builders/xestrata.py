"""XeStrata (Strata 0.1.45 SYCL) native builder.

Runs Qwen3.8-Flash-Next on 2x Intel Arc Pro B70 with expert parallelism,
layer split, and native MTP 4 speculation.
"""
from pathlib import Path

from . import common
from ..paths import IS_WIN


def build_xestrata(b):
    cfg, model, recipe, warns = b.cfg, b.model, b.recipe, b.warns
    if IS_WIN:
        raise common.BuildError("XeStrata SYCL is Linux-only.")

    py_exe = str(recipe.get("py_exe") or "/data/sata1/engines/xestrata/.venv/bin/python")
    server_py = str(recipe.get("server_script") or "/data/sata1/engines/xestrata/serve/server.py")
    config_file = str(recipe.get("config_file") or Path("~/.config/xestrata/xestrata-iq3_xxs.json").expanduser())
    port = b.port
    host = "127.0.0.1"

    gpus = list(b.gpus)
    gpus_arg = ",".join(map(str, gpus)) if len(gpus) > 1 else str(gpus[0])

    effort = cfg.get("reasoning_effort") or recipe.get("reasoning_effort")
    effort_flag = f" --effort {effort}" if effort else ""

    # Handle vision toggle if explicitly passed
    vision_enabled = cfg.get("vision")
    config_path = Path(config_file)
    if vision_enabled is not None and config_path.exists():
        try:
            import json
            with open(config_path) as f:
                cdata = json.load(f)
            if vision_enabled:
                cdata["vision"] = {
                    "exe": "/data/sata1/engines/xestrata/dist/usr/lib/x86_64-linux-gnu/xestrata/engine/strata-vision-cpu",
                    "mmproj": "/data/sata1/models/qwen38-flashnext-vision/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                    "model": "/data/sata1/engines/strata/models-iq3xxs/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
                }
            else:
                cdata.pop("vision", None)
            with open(config_path, "w") as f:
                json.dump(cdata, f, indent=2)
        except Exception:
            pass

    cmd_str = (
        f"source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && "
        f"exec {py_exe} {server_py} --engine strata --config {config_file} --port {port} --host {host} --gpu {gpus_arg}{effort_flag}"
    )
    tokens = ["bash", "-c", cmd_str]

    tokens += recipe.get("fixed_flags", []) + b.extra

    warns.append(
        "XeStrata SYCL with 2x Arc Pro B70 expert parallelism, K=24 pipelining, and native MTP4 "
        "(55.2–67.5 tok/s code/prose, up to 122 tok/s repetitive; 2,026 tok/s prefill at 32K; full 262K context). "
        "First load takes ~2-3 min to read weights and fill GPU expert cache."
    )

    return {
        "tokens": tokens,
        "env": recipe.get("env", {}),
        "cname": "",
        "served_name": recipe.get("served_model_name", "qwen3.8-flash-next"),
        "native": True,
        "docker_sock": "",
    }
