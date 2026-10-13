"""OVMS (OpenVINO Model Server) builder."""
import os
from pathlib import Path

from .. import catalog
from ..paths import IS_WIN
from . import common


def build_ovms(b):
    """kind: ovms — single GPU, KV managed by the server itself."""
    cfg, model, recipe, det, warns = b.cfg, b.model, b.recipe, b.det, b.warns
    fallback_mount = cfg.get("models_dir") or catalog.SETTINGS.get("ovms_repo", "~/models/ovms-repo")
    if det:
        cpath, mount_src, mount_dst = common.container_path(det, fallback_mount)
        source_model = cpath[len("/models/"):] or "model"
    else:
        source_model = recipe.get("source_model", "model")
        mount_src = fallback_mount
        mount_dst = "/models"
        warns.append("model not detected on disk — preview uses default paths; a real launch needs the artifact (download it or add its folder to the model folders list).")
    cname = common.cname_for(model["id"], "ovms", b.port)
    tokens = ["docker", "run", "--rm", "--name", cname] + common.labels_for(cfg, b.port)
    if not IS_WIN:
        dri = common.dri_args(b)
        gid = dri[dri.index("--group-add") + 1] if "--group-add" in dri else str(os.getgid())
        tokens += ["--user", f"{os.getuid()}:{gid}"] + dri
    tokens += ["-p", f"127.0.0.1:{b.port}:{b.port}",
               "-v", f"{Path(mount_src).expanduser()}:{mount_dst}:rw"] + b.envargs + [
        recipe["image"],
        "--rest_port", str(b.port),
        "--model_repository_path", "/models",
        "--source_model", source_model,
        "--task", "text_generation",
        "--target_device", "GPU.0",  # only the selected card is mapped
        "--enable_prefix_caching", "true"]
    if b.slots > 1:
        warns.append("OVMS manages concurrency internally — --slots is advisory.")
    if recipe.get("tool_parser"):
        tokens += ["--tool_parser", recipe["tool_parser"]]
    if recipe.get("reasoning_parser"):
        tokens += ["--reasoning_parser", recipe["reasoning_parser"]]
    if recipe.get("cim_long_ctx") and b.ctx > 20480:
        tokens += ["--cache_interval_multiplier", str(recipe["cim_long_ctx"])]
    tokens += recipe.get("fixed_flags", []) + b.extra
    warns.append("OVMS manages KV internally (GPU INT4 KV supported). ctx is advisory here.")
    return {"tokens": tokens, "env": {}, "cname": cname, "served_name": source_model,
            "native": False, "docker_sock": ""}
