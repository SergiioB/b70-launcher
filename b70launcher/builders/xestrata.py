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

    port = int(cfg.get("port") or recipe.get("port", 8095))
    host = "127.0.0.1"

    gpus = list(b.gpus)
    gpus_arg = ",".join(map(str, gpus)) if len(gpus) > 1 else str(gpus[0])

    tokens = [
        py_exe, server_py,
        "--engine", "strata",
        "--config", config_file,
        "--port", str(port),
        "--host", host,
        "--gpu", gpus_arg,
    ]

    tokens += recipe.get("fixed_flags", []) + b.extra

    warns.append(
        "XeStrata SYCL with 2x Arc Pro B70 expert parallelism and native MTP4 (122.2 tok/s); "
        "first load takes ~3-4 min to read weights from disk."
    )

    return {
        "tokens": tokens,
        "env": recipe.get("env", {}),
        "cname": "",
        "served_name": recipe.get("served_model_name", "qwen3.8-flash-next"),
        "native": True,
        "docker_sock": "",
    }
