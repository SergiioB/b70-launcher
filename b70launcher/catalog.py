"""Recipe catalog + settings: load, user overrides, remote overlay.

``load()`` populates the module-level RECIPES/SETTINGS dicts (called from
the entry point, and from tests with a controlled environment — importing
this module reads nothing).
"""
import json
import re
import time
import urllib.request

from . import paths
from .locks import LOCK

UPDATE_URL = None  # set by load(); env override honored there
UPDATE_INFO = {"checked": False, "has_update": False, "latest_version": "", "message": "", "download_url": "https://xecores.com/match", "notes": []}
# Remote recipe catalog state: filled by the same background check as
# UPDATE_INFO. entries maps "<model_id>:<engine>" -> {"ver", "note", "recommended"}.
RECIPE_REMOTE = {"checked": False, "catalog_ver": "", "recipes_url": "", "entries": {}}

RECIPES = {}
SETTINGS = {}
USER_RECIPE_OVERRIDES = {}

UA = {"User-Agent": f"b70-launcher/{paths.VERSION}"}

# settings-override keys a user edit may set (everything else ships read-only)
OVERRIDE_KEYS = ("scan_dirs", "models_dir", "ovms_repo", "llama_bin",
                 "exl3_data_root", "open_with", "sync_clients")

# model-level fields a published recipe update may set (never "id"/"recipes")
REMOTE_MODEL_FIELDS = ("recommended_engine", "recommendation", "badge", "blurb",
                       "subtitle", "tags", "chips", "brand")

# recipe fields a published update may tune. Exec-shaping keys (image,
# docker_sock, llama_bin, fixed_flags, serve_config, kind) and host-path keys
# (model_path, gguf, source_model) never come from the wire: they would let a
# catalog swap the binary/image, pick the docker socket, append container argv
# or remount host paths. Everything else merges over the local recipe.
REMOTE_RECIPE_FIELDS = ("ctx", "ctx_max", "ctx_note", "ctx_safe", "power", "dtype",
                        "kv", "kv_options", "download", "spec_tokens", "spec_p_min", "speculative",
                        "draft_device", "tensor_split", "split_mode",
                        "offload_tensors", "tool_parser", "reasoning_parser",
                        "cim_long_ctx", "search_name", "inner_port", "tp",
                        "perf", "topology", "recipe_ver", "gmu", "gmu_safe", "gmu_safe_budget")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_ENGINE_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
_DRAFT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf", re.I)


def _url_ok(url):
    """Remote fetches stay on https; plain http only for localhost test rigs."""
    return url.startswith("https://") or url.startswith("http://127.0.0.1") \
        or url.startswith("http://localhost")


def load():
    """Read recipes.json + settings.json, then layer user overrides and the
    persisted remote overlay. Idempotent."""
    global UPDATE_URL
    import os
    UPDATE_URL = os.environ.get("B70_UPDATE_URL", "https://xecores.com/downloads/version.json")
    RECIPES.clear()
    RECIPES.update(json.loads((paths.HERE / "recipes.json").read_text()))
    SETTINGS.clear()
    SETTINGS.update(json.loads((paths.HERE / "settings.json").read_text()))
    _merge_override()
    _load_remote_overlay()


def find_model(model_id):
    for m in RECIPES.get("models", []):
        if m.get("id") == model_id:
            return m
    return None


# ---------------------------------------------------------------- overrides

def _apply_recipe_overrides(ov_map):
    """Per-recipe user overrides: {"<model_id>": {"<engine>": {field: value}}}.

    Lets a host keep site-specific recipe fields (a native llama_bin path, a
    non-standard artifact location) out of the shipped recipes.json. These
    always win — re-applied after any remote recipe overlay."""
    if not isinstance(ov_map, dict):
        return
    for mid, per_eng in ov_map.items():
        if isinstance(per_eng, dict):
            USER_RECIPE_OVERRIDES.setdefault(mid, {}).update(per_eng)
    for m in RECIPES.get("models", []):
        per = ov_map.get(m.get("id"))
        if not isinstance(per, dict):
            continue
        for eng, fields in per.items():
            if isinstance(fields, dict) and eng in m.get("recipes", {}):
                m["recipes"][eng].update(fields)


def _merge_override():
    """User edits (scan roots etc.) live in the state dir, never in the install."""
    try:
        ov = json.loads(paths.OVERRIDE_PATH.read_text())
    except Exception:
        return
    if isinstance(ov, dict):
        for k, v in ov.items():
            if k in OVERRIDE_KEYS:
                SETTINGS[k] = v
        _apply_recipe_overrides(ov.get("recipe_overrides"))


def save_override(keys):
    ov = {}
    try:
        ov = json.loads(paths.OVERRIDE_PATH.read_text())
    except Exception:
        pass
    if not isinstance(ov, dict):
        ov = {}
    for k in keys:
        ov[k] = SETTINGS.get(k)
    with LOCK:
        paths.atomic_write(paths.OVERRIDE_PATH, json.dumps(ov, indent=2))


# ------------------------------------------------------------ remote recipes

def _sanitize_remote_recipe(r):
    """Strip a published recipe to fields allowed over the wire."""
    out = {k: r[k] for k in REMOTE_RECIPE_FIELDS if k in r}
    dm = r.get("draft_model")  # bare artifact names only — resolved via scan
    if dm is not None and _DRAFT_RE.fullmatch(str(dm)):
        out["draft_model"] = str(dm)
    return out


def _stamp_ver(recipe, fallback):
    if isinstance(recipe, dict) and not recipe.get("recipe_ver") and fallback:
        recipe["recipe_ver"] = fallback


def apply_recipe_doc(doc, catalog_ver=""):
    """Overlay one published recipes document onto the live RECIPES dict.

    Remote recipes are sanitized to REMOTE_RECIPE_FIELDS and merged over the
    local recipe, so the update channel cannot replace exec-shaping fields.
    Recipes for model+engine pairs this app does not ship are NOT added —
    they would arrive without their exec-shaping fields (kind/image were
    stripped) and could never run; the update notice tells the user to
    update the app instead (audit M8). Returns (applied_count, needs_app,
    error): needs_app lists "<model>:<engine>" keys that were refused."""
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, list):
        return 0, [], "remote document has no models list"
    applied = 0
    needs_app = []
    with LOCK:
        by_id = {m.get("id"): m for m in RECIPES.get("models", [])}
        for rm in models:
            if not isinstance(rm, dict) or not _MODEL_ID_RE.fullmatch(str(rm.get("id") or "")):
                continue
            local = by_id.get(rm["id"])
            rrecipes = rm.get("recipes") or {}
            if local is None:
                # unknown model: nothing to merge a sanitized recipe into
                for eng in rrecipes:
                    if isinstance(rrecipes.get(eng), dict):
                        needs_app.append(f"{rm['id']}:{eng}")
                continue
            for f in REMOTE_MODEL_FIELDS:
                if f in rm:
                    local[f] = rm[f]
            for eng, r in rrecipes.items():
                if not isinstance(r, dict) or not _ENGINE_RE.fullmatch(str(eng)):
                    continue
                if eng not in (local.get("recipes") or {}):
                    # unknown engine for a known model: same refusal — the
                    # sanitized payload cannot carry kind/image
                    needs_app.append(f"{rm['id']}:{eng}")
                    continue
                rr = _sanitize_remote_recipe(r)
                _stamp_ver(rr, catalog_ver)
                local_r = local["recipes"][eng]
                r_ver = rr.get("recipe_ver") or catalog_ver
                local_ver = local_r.get("recipe_ver")
                if local_ver and r_ver and _ver_key(local_ver) > _ver_key(r_ver):
                    continue
                local_r.update(rr)
                applied += 1
        _apply_recipe_overrides(USER_RECIPE_OVERRIDES)
    return applied, needs_app, None


def _load_remote_overlay():
    try:
        doc = json.loads(paths.REMOTE_RECIPES_PATH.read_text())
    except Exception:
        return
    cver = doc.get("catalog_ver") or ""
    local_cver = RECIPES.get("catalog_ver") or ""
    if local_cver and cver and _ver_key(local_cver) > _ver_key(cver):
        return
    apply_recipe_doc(doc.get("payload") or doc, cver)


def _ver_key(v):
    """recipe_ver values are YYYY-MM-DD dates (or ints); compare safely."""
    s = str(v or "")
    return [int(p) if p.isdigit() else 0 for p in re.split(r"[^0-9]", s) if p]


def local_recipe_ver(model_id, engine):
    m = find_model(model_id)
    r = (m or {}).get("recipes", {}).get(engine) or {}
    return r.get("recipe_ver") or RECIPES.get("catalog_ver") or ""


def recipe_notices():
    """Remote catalog entries newer than the shipped/applied local recipe."""
    out = {}
    for key, e in RECIPE_REMOTE.get("entries", {}).items():
        parts = key.split(":", 1)
        if len(parts) != 2:
            continue
        mid, eng = parts
        remote_ver = str(e.get("ver") or "")
        if not remote_ver:
            continue
        m = find_model(mid)
        is_new = not m or eng not in (m.get("recipes") or {})
        local_ver = "" if is_new else local_recipe_ver(mid, eng)
        if _ver_key(remote_ver) <= _ver_key(local_ver) and not is_new:
            continue
        out[key] = {
            "model_id": mid, "engine": eng,
            "model_name": (m or {}).get("name") or e.get("model") or mid,
            "local_ver": local_ver, "remote_ver": remote_ver,
            "note": str(e.get("note") or ""),
            "recommended": bool(e.get("recommended")),
            "is_new": is_new,
            # a new model/engine needs app support (kind/image), which the
            # sanitized channel cannot deliver — only a newer app can
            "can_apply": bool(RECIPE_REMOTE.get("recipes_url")) and not is_new,
        }
    return out


def recipe_notice_for(model_id, engine):
    return recipe_notices().get(f"{model_id}:{engine}")


def fetch_remote_recipes():
    """Pull the full published recipes document referenced by the manifest."""
    url = RECIPE_REMOTE.get("recipes_url")
    if not url:
        return None, "no remote recipes_url in the update manifest"
    if not _url_ok(url):
        return None, "remote recipes_url must be https"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(8 << 20)
        doc = json.loads(raw.decode())
    except Exception as e:
        return None, f"remote recipes fetch failed: {e}"
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
        return None, "remote document is not a recipes catalog"
    return doc, None


def apply_recipe_update(model_id=None, engine=None):
    """Fetch the published catalog and overlay it (or one recipe) locally.

    The merged result is persisted in the state dir, so the update survives
    restarts without touching the install. A single model:engine request
    overlays only that engine's recipe — never every engine of the model
    (audit M8)."""
    doc, err = fetch_remote_recipes()
    if err:
        return {"error": err}
    cat_ver = str(doc.get("catalog_ver") or RECIPE_REMOTE.get("catalog_ver") or "")
    if model_id and engine:
        target = None
        for rm in doc["models"]:
            if rm.get("id") == model_id:
                rec = (rm.get("recipes") or {}).get(engine)
                if rec is not None:
                    target = dict(rm, recipes={engine: rec})  # this engine only
                    break
        if target is None:
            return {"error": f"remote catalog has no recipe for {model_id}:{engine}"}
        doc = {"models": [target]}
    applied, needs_app, err = apply_recipe_doc(doc, cat_ver)
    if err:
        return {"error": err}
    if needs_app:
        return {"error": "recipe needs a newer launcher app (update b70-launcher): "
                         + ", ".join(needs_app[:8])}
    if not applied:
        return {"error": "nothing newer to apply"}
    try:
        stored = {"catalog_ver": cat_ver,
                  "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "payload": {"models": list(doc["models"])}}
        if paths.REMOTE_RECIPES_PATH.exists():
            try:
                prev = json.loads(paths.REMOTE_RECIPES_PATH.read_text())
                prev_models = (prev.get("payload") or {}).get("models") or []
                ids = {m.get("id") for m in stored["payload"]["models"]}
                stored["payload"]["models"] = (
                    [m for m in prev_models if m.get("id") not in ids]
                    + stored["payload"]["models"])
                if prev.get("catalog_ver") and _ver_key(prev["catalog_ver"]) > _ver_key(cat_ver):
                    stored["catalog_ver"] = prev["catalog_ver"]
            except Exception:
                pass
        with LOCK:
            paths.atomic_write(paths.REMOTE_RECIPES_PATH, json.dumps(stored, indent=1))
    except Exception as e:
        return {"error": f"applied in memory but failed to persist: {e}"}
    from . import scan
    scan.start_scan_async(force=True)
    return {"ok": True, "applied": applied, "catalog_ver": cat_ver}


# ------------------------------------------------------------ update channel

def check_recipe_manifest(version_doc):
    """Fetch the small per-recipe manifest next to version.json.

    The manifest is data-only (versions + notes); the full recipes document is
    fetched lazily when the user applies an update."""
    manifest_url = version_doc.get("recipes_manifest_url")
    if not manifest_url:
        manifest_url = UPDATE_URL.rsplit("/", 1)[0] + "/recipes-manifest.json"
    if not _url_ok(str(manifest_url)):
        RECIPE_REMOTE["checked"] = True
        return
    try:
        req = urllib.request.Request(str(manifest_url), headers=UA)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read(4 << 20).decode())
        entries = data.get("recipes")
        if isinstance(entries, dict):
            RECIPE_REMOTE.update({
                "catalog_ver": str(data.get("catalog_ver") or ""),
                "recipes_url": str(data.get("recipes_url") or ""),
                "entries": {k: v for k, v in entries.items() if isinstance(v, dict)},
            })
    except Exception:
        pass
    RECIPE_REMOTE["checked"] = True


def _update_notes(raw, limit=8, width=300):
    """Release-note bullets from version.json: plain strings only, at most
    `limit` of them, each cut to `width` characters."""
    if not isinstance(raw, list):
        return []
    notes = []
    for item in raw:
        text = " ".join(item.split()) if isinstance(item, str) else ""
        if text:
            notes.append(text if len(text) <= width else text[:width - 1].rstrip() + "…")
        if len(notes) == limit:
            break
    return notes


def check_for_updates():
    global UPDATE_INFO
    try:
        req = urllib.request.Request(UPDATE_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read(1 << 20).decode())
            latest = data.get("version", "")
            if latest and _ver_key(latest) > _ver_key(paths.VERSION):
                UPDATE_INFO = {
                    "checked": True,
                    "has_update": True,
                    "latest_version": latest,
                    "message": data.get("message", f"B70 Launcher v{latest} is available."),
                    "download_url": data.get("download_url", "https://xecores.com/match"),
                    "notes": _update_notes(data.get("notes")),
                }
            else:
                UPDATE_INFO["checked"] = True
            check_recipe_manifest(data)
    except Exception as exc:
        UPDATE_INFO["checked"] = True
        RECIPE_REMOTE["checked"] = True
        print(f"update check failed ({UPDATE_URL}): {exc}")


def start_update_check():
    """Background update check — started by the entry point, never at import."""
    import threading
    threading.Thread(target=check_for_updates, daemon=True).start()
