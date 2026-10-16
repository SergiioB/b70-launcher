"""Launch-plan construction: build(cfg) dispatches per recipe kind.

Every builder returns a dict of {tokens, env, cname, served_name, native,
docker_sock, ...}; build() validates config, applies the shared fixes
(port-qualified cnames + ownership labels, single-GPU pinning, relative
selectors) and assembles the JSON plan the API returns.

``power`` in the launch config is ignored — the launcher never changes power
settings; recipes only carry an advisory watts hint (warned against the
card's hwmon cap).
"""
import os
from pathlib import Path

from .. import catalog, scan
from ..paths import IS_WIN
from . import common
from .common import BuildError, Bld
from .ovms import build_ovms
from .vllm import BUILDERS as VLLM_BUILDERS, build_generic
from .llamacpp import build_llamacpp
from .exl3 import build_exl3
7: from .xestrata import build_xestrata
from .strata import build_strata
8:     if kind == "xestrata":
        return build_xestrata
    if kind == "strata":
        return build_strata
CUSTOM_TEMPLATES = {
    "llamacpp": {"kind": "gguf", "image": "ghcr.io/ggml-org/llama.cpp:server-intel-b11429@sha256:3353d968ed8987612b7859828d14d205413567baafed00a90d71d0c23e63d7ba",
                 "ctx": 32768, "power": 150, "kv_options": list(common.KV_LLAMACPP),
                 "download": {"kind": "file", "quant": "GGUF (custom)"}},
    "openvino": {"kind": "ovms", "image": "openvino/model_server:2026.4.1-gpu@sha256:a928903fe429f1e3e27107cc45932d30d95febfd2d1159dc50869c3eac76fc58",
                 "ctx": 20480, "power": 150, "download": {"kind": "snapshot", "quant": "OpenVINO IR (custom)"}},
    "vllm": {"kind": "vllm", "image": "vllm/vllm-openai-xpu:v0.31.0@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7",
             "ctx": 32768, "power": 150, "dtype": "bfloat16", "kv_options": list(common.KV_VLLM),
             "download": {"kind": "snapshot", "quant": "HF/safetensors (custom)"}},
    "exl3": {"kind": "exl3", "image": "ghcr.io/0xsero/exl3xpu:0.1.0@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8",
             "docker_sock": "/run/b70-exl3-docker.sock", "inner_port": 8100,
             "ctx": 65536, "power": 230, "kv_options": ["fp8"], "prefill_options": ["int8", "fp16"],
             "download": {"kind": "snapshot", "quant": "EXL3 trellis (custom)"}},
}
KIND_ENGINE = {"gguf": "llamacpp", "openvino": "openvino", "vllm": "vllm", "exl3": "exl3", "xestrata": "xestrata"}


def prepare_custom(cfg):
    """Resolve an arbitrary on-disk artifact into a synthetic model+recipe so it
    can flow through the normal build path. Only paths under a configured scan
    root are accepted. The engine is inferred from the artifact when omitted."""
    import json
    warns = []
    raw = (cfg.get("custom_path") or "").strip()
    if not raw:
        raise BuildError("custom model path missing")
    engine = cfg.get("engine") or ""
    p = Path(raw).expanduser().resolve()
    roots = scan.scan_roots()
    if not any(p == r or r in p.parents for r in roots):
        raise BuildError("custom model path is not under any configured model folder")
    if not p.exists():
        raise BuildError(f"artifact not found on disk: {raw}")
    quant_method = None
    if p.is_file():
        kind = "gguf" if p.suffix.lower() == ".gguf" else None
    else:
        names = {e.name.lower() for e in os.scandir(p) if not e.is_symlink()}
        if any(n.startswith("openvino_language_model.") for n in names) or \
           any(n.endswith(".xml") and (n[:-4] + ".bin") in names for n in names):
            kind = "openvino"
        else:
            kind = "vllm"
        # the artifact's own quantization record decides the serving family
        quant_method = scan.read_quant_method(p)
        if kind == "vllm" and quant_method == "exl3":
            kind = "exl3"
    if not engine:
        engine = KIND_ENGINE.get(kind or "", "vllm")
    if kind is None or KIND_ENGINE.get(kind) != engine:
        raise BuildError(f"that artifact is {kind or 'of unknown format'}"
                         + (f" (quantization method on disk: {quant_method})" if quant_method else "")
                         + f"; pick the matching engine ({KIND_ENGINE.get(kind, 'llamacpp')})")
    # exl3 is the one custom family with a hard quant requirement: the generic
    # template only serves EXL3 trellis weights, anything else must say exl3
    if engine == "exl3" and quant_method and quant_method != "exl3":
        raise BuildError(f"this artifact records quantization method {quant_method}; the EXL3 "
                         "template only serves EXL3 trellis weights")
    cfg_path = p / "config.json"
    disk_ctx = scan.read_ctx_from_config(cfg_path)
    recipe = dict(CUSTOM_TEMPLATES[engine])
    recipe["ctx"] = disk_ctx or recipe["ctx"]
    if disk_ctx:
        recipe["ctx_note"] = "artifact config.json on disk"
    if engine == "exl3":
        # Stock exl3xpu serves K=4 and K=6 only. Bit widths outside that set
        # (turboderp's SC quants) need the mixed-K image a catalog recipe pins.
        odd = sorted(scan.read_exl3_bits(p) - {4, 6})
        if odd:
            mixed = next((r.get("image") for m in catalog.RECIPES.get("models", [])
                          for r in (m.get("recipes") or {}).values()
                          if r.get("kind") == "exl3" and "exl3xpu-mixedk" in str(r.get("image", ""))), None)
            if not mixed:
                raise BuildError(f"this EXL3 artifact uses {odd}-bit tensors; stock exl3xpu serves "
                                 "only 4 and 6 bits and no mixed-K recipe image is in the catalog")
            recipe["image"] = mixed
            warns.append(f"custom EXL3 with {'/'.join(map(str, odd))}-bit tensors: using the mixed-K image")
    model = {"id": "custom", "name": p.name, "arch": f"{kind} · custom artifact",
             "recipes": {engine: recipe}}
    det = {"path": str(p), "mount_root": str(p.parent if p.is_file() else p),
           "found_name": p.name, "variant": False, "quant_method": quant_method,
           "ctx": disk_ctx}
    warns.append(f"custom artifact: {p.name} — engine flags follow the generic {engine} template, "
                 "not a verified recipe")
    if engine == "llamacpp":
        warns.append("custom GGUF: quantization is read from the file when the engine loads it — "
                     "the launcher does not verify it against a recipe")
    return model, engine, det, warns


def _kind_builder(kind):
    if kind == "ovms":
        return build_ovms
    if kind.startswith("vllm"):
        return VLLM_BUILDERS.get(kind, build_generic)
    if kind in ("gguf", "gguf-tiered"):
        return build_llamacpp
    if kind == "exl3":
        return build_exl3
7: from .xestrata import build_xestrata
from .strata import build_strata
8:     if kind == "xestrata":
        return build_xestrata
    if kind == "strata":
        return build_strata
    return None


def build(cfg):
    warns = []
    try:
        if cfg.get("custom_path") and cfg.get("model_id") not in (None, "", "__custom__"):
            # never silently drop half the request: a custom artifact and a
            # selected recipe are different launch targets
            raise BuildError("custom model path and a selected recipe cannot be combined — "
                             "clear the custom artifact or the recipe selection")
        if cfg.get("model_id") == "__custom__" or (cfg.get("custom_path") and not cfg.get("model_id")):
            model, engine, det, warns = prepare_custom(cfg)
        else:
            model = catalog.find_model(cfg.get("model_id") or cfg.get("model") or "")
            engine = cfg.get("engine", "openvino")
            det = scan.detect(model, engine)
        recipe = (model or {}).get("recipes", {}).get(engine)
        if not model or not recipe:
            return {"error": "no recipe for that model+engine", "warnings": warns}
        hold = str(recipe.get("operator_hold") or "").strip()
        if hold:
            if not cfg.get("dry_run"):
                return {"error": f"recipe is on operator hold: {hold}", "warnings": warns}
            warns.append(f"operator hold: {hold} — real launches are refused until the operator releases the recipe.")
        common.validate_models_dir(cfg)  # BuildError on a rogue models_dir
        ctxres = scan.resolve_ctx(model, engine, det)
        artifact_mib = None
        if det and det.get("path"):
            try:
                p = Path(det["path"])
                artifact_mib = (p.stat().st_size if p.is_file()
                                else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())) / 1048576
            except Exception:
                artifact_mib = None
        dl = recipe.get("download", {}) or {}
        if det and det.get("variant"):
            warns.append(f"using your local \"{det.get('found_name')}\" — recipe default was "
                         f"{dl.get('name') or dl.get('search') or dl.get('dest') or 'the listed quant'}")
        if cfg.get("use_docker") is False and not recipe.get("kind", "").startswith("gguf"):
            warns.append("--native ignored: this engine only runs in a container.")
        b = Bld(cfg, model, engine, recipe, det, warns)
        builder = _kind_builder(recipe.get("kind", ""))
        if builder is None:
            return {"error": f"unknown recipe kind {recipe.get('kind')}", "warnings": warns}
        out = builder(b)
        tokens = out["tokens"]
        return {
            "cmd": common.pretty(tokens), "tokens": tokens,
            "env": out.get("env") or {}, "warnings": warns,
            "operator_hold": hold or None,
            "cname": out.get("cname", ""), "docker_sock": out.get("docker_sock", ""),
            "endpoint": f"http://127.0.0.1:{b.port}/v1", "served_name": out.get("served_name", model["name"]),
            "model_name": model["name"], "engine": engine, "kind": recipe.get("kind", ""),
            "rid": common.rid_for(cfg), "gpus": b.gpus,
            "ctx": b.ctx, "ctx_source": b.ctx_source,
            "quant": dl.get("quant") or "", "variant": bool(det and det.get("variant")),
            "found_name": (det or {}).get("found_name") or "",
            "custom": bool(model.get("id") == "custom"),
            "recipe_ver": catalog.local_recipe_ver(model["id"], engine),
            "detected": bool(det), "detected_path": (det or {}).get("path"),
            "native": bool(out.get("native")),
            "effective_binary": out.get("effective_binary") or recipe.get("image", ""),
            "image_hint": out.get("image_hint") or "",
            "artifact_mib": round(artifact_mib, 1) if artifact_mib else None,
        }
    except BuildError as e:
        return {"error": str(e), "warnings": warns}
