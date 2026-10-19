"""Shared builder plumbing: validation, device selection, command rendering."""
import os
import re
import shlex
from pathlib import Path

from .. import catalog, hardware, scan
from ..paths import IS_WIN

# env var names a user-supplied extra_env must not set — they can redirect code
# loading or process startup inside the launched engine/container
BLOCKED_ENV = re.compile(
    r"(?:LD_.*|BASH_ENV|ENV|SHELLOPTS|BASHOPTS|GLOBIGNORE|PROMPT_COMMAND|"
    r"PYTHON.*|PERL5.*|RUBYLIB|NODE_OPTIONS|IFS|PATH|HOME|USER|SHELL|TERM)$")

ENGINES = ("openvino", "vllm", "llamacpp", "exl3", "strata")

# KV dtypes vLLM accepts on this stack
KV_VLLM = ("fp8", "fp8_e4m3", "fp8_e5m2", "fp16", "bf16", "auto")
KV_LLAMACPP = ("q8_0", "q5_0/q4_1", "q8_0/q4_1", "f16")


class BuildError(Exception):
    """Reject the launch with a user-readable message."""


def rid_for(cfg):
    """Tracked-server key; sanitized — it is also used in log filenames and
    container labels. Single authority (runtime + labels share it)."""
    rid = f"{cfg.get('model_id') or cfg.get('model')}-{cfg.get('engine')}-{cfg.get('port') or 8000}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", rid)


def cname_for(model_id, kind, port):
    """Container name, port-qualified: two servers of the same recipe on
    different ports must never collide (audit H3)."""
    base = re.sub(r"[^a-z0-9_.-]", "-", f"b70-{model_id}-{kind}-{int(port)}".lower())
    return base.strip("-")


def pretty(tokens, width=96):
    sep = " ^" if IS_WIN else " \\"
    lines, cur = [], ""
    for tok in tokens:
        quoted = tok if IS_WIN else shlex.quote(tok)
        candidate = (cur + " " + quoted).strip()
        if cur and len(candidate) > width:
            lines.append(cur + sep)
            cur = "    " + quoted
        else:
            cur = candidate
    lines.append(cur)
    return "\n".join(lines)


def by_path_dir(nodes):
    """Directory of /dev/dri/by-path links for the given nodes only.

    oneCCL opens every by-path entry it sees; the host directory also lists
    cards that are not mapped into the container, which fails init_device."""
    from ..paths import DATA
    names = sorted(Path(n).name for n in nodes)
    out = DATA / "dri-by-path" / "-".join(names)
    out.mkdir(parents=True, exist_ok=True)
    host = Path("/dev/dri/by-path")
    for link in (host.iterdir() if host.is_dir() else ()):
        try:
            target = os.readlink(link)
        except OSError:
            continue
        if Path(target).name in names and not (out / link.name).is_symlink():
            (out / link.name).symlink_to(f"../{Path(target).name}")
    return out


def dri_args(b):
    """Docker args exposing only the selected cards (audit: no fd on other cards).

    Maps each selected render node and its card node, plus by-path links for
    those nodes only. Inside the container the selected cards are the only
    devices, so engines address them as 0..n-1 (container_env)."""
    args = []
    nodes = b.render_nodes + b.card_nodes
    for node in nodes:
        args += ["--device", node]
    if nodes:
        args += ["-v", f"{by_path_dir(nodes)}:/dev/dri/by-path:ro"]
    gids = []
    for node in b.render_nodes + b.card_nodes:
        try:
            g = os.stat(node).st_gid
        except OSError:
            continue
        if g not in gids:
            gids.append(g)
            args += ["--group-add", str(g)]
    return args


def container_env(b):
    """Device env for a container that sees only the selected cards."""
    return device_env(list(range(len(b.gpus))))


def container_path(det, fallback):
    """Map an on-disk artifact to its container path and the mount that exposes it.

    GGUF files: the parent directory is mounted at /models, file stays by name.
    Snapshot dirs: the dir itself is mounted AT /models/model so the model always
    has a stable container path regardless of how deep the scan root goes."""
    if not det or not det.get("path"):
        return fallback, None, "/models"
    try:
        p = Path(det["path"])
        if p.is_file():
            src = str(p.parent)
            return "/models/" + p.name, src, "/models"
        return "/models/model", str(p), "/models/model"
    except (ValueError, KeyError, OSError):
        return fallback, None, "/models"


def device_env(gpus):
    """ZE_AFFINITY_MASK + RELATIVE ONEAPI_DEVICE_SELECTOR (audits H4/H5).

    The mask renumbers the visible devices to 0..n-1, so the selector must
    address relative indices — an absolute selector selects nothing once the
    mask hides the leading card (e.g. gpus=[1])."""
    return {
        "ZE_AFFINITY_MASK": ",".join(str(g) for g in gpus),
        "ONEAPI_DEVICE_SELECTOR": "level_zero:" + ",".join(str(i) for i in range(len(gpus))),
    }


def resolve_kv(kv, recipe, default_options, warns, *, label="--kv"):
    """Validate the KV choice against the recipe's kv_options (audit M4).
    Unsupported values warn and fall back to the recipe default."""
    options = tuple(recipe.get("kv_options") or default_options)
    if kv in ("recipe default", "", None):
        kv = recipe.get("kv") or (options[0] if options else default_options[0])
    if options and kv not in options:
        warns.append(f"{label} '{kv}' isn't supported by this recipe "
                     f"(valid: {', '.join(options)}) — using {recipe.get('kv') or options[0]}.")
        kv = recipe.get("kv") or options[0]
    return kv


def labels_for(cfg, port):
    """Ownership labels every launcher container carries (audit H9): the
    launcher re-adopts and stops by these, never by a bare b70- prefix."""
    return ["--label", f"b70.launcher.rid={rid_for(cfg)}",
            "--label", f"b70.launcher.port={int(port)}"]


def validate_models_dir(cfg):
    """A caller-supplied models_dir may only be the configured one or live
    under a scan root — never an arbitrary host dir mounted into containers
    (audit L9)."""
    md = cfg.get("models_dir")
    if not md:
        return None
    p = Path(str(md)).expanduser().resolve()
    allowed = [Path(catalog.SETTINGS.get("models_dir", "~/models")).expanduser().resolve()]
    allowed += scan.scan_roots()
    if not any(p == a or a in p.parents for a in allowed):
        raise BuildError("models_dir must be the configured models directory or live under a scan root.")
    return p


def parse_extra_env(cfg, warns):
    """extra_env lines -> (envargs for docker, dict for native env)."""
    extra_env = {}
    for line in (cfg.get("extra_env") or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise BuildError(f"Invalid environment entry {line!r} — expected NAME=value")
        k, v = line.split("=", 1)
        k = k.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k):
            raise BuildError("Invalid environment variable name")
        if BLOCKED_ENV.match(k):
            raise BuildError(f"Environment variable {k} is not allowed (it can redirect code loading or process startup)")
        extra_env[k] = v.strip()
    envargs = [x for k, v in extra_env.items() for x in ("-e", f"{k}={v}")]
    return envargs, extra_env


def default_gpus(recipe):
    """GPU selection a recipe implies when the caller doesn't send one."""
    g = recipe.get("gpus")
    if isinstance(g, list) and g and all(isinstance(i, int) and i in (0, 1) for i in g):
        return sorted(set(g))
    tp = recipe.get("tp")
    if isinstance(tp, int) and tp > 1:
        return list(range(min(tp, 2)))
    if recipe.get("kind") == "vllm-tp2":
        return [0, 1]
    return [0]


# how many cards each kind can actually use (audit M5: honest messages)
MAX_GPUS = {"ovms": 1, "exl3": 1, "vllm": 2, "vllm-tp2": 2, "vllm-arext": 1,
            "vllm-mtp": 1, "vllm-dflash": 1, "vllm-autoround": 2,
            "gguf": 2, "gguf-tiered": 2, "xestrata": 2, "strata": 2}


class Bld:
    """Resolved, validated launch parameters shared by every builder."""

    def __init__(self, cfg, model, engine, recipe, det, warns):
        self.cfg = cfg
        self.model = model
        self.engine = engine
        self.recipe = recipe
        self.det = det
        self.warns = warns
        ctxres = scan.resolve_ctx(model, engine, det)
        try:
            self.ctx = int(cfg["ctx"]) if cfg.get("ctx") is not None else ctxres["value"]
            self.port = int(cfg["port"]) if cfg.get("port") is not None else 8000
            self.slots = int(cfg["slots"]) if cfg.get("slots") is not None else 1
            self.extra = shlex.split(cfg.get("extra", "") or "")
        except (ValueError, TypeError) as exc:
            raise BuildError(f"Invalid context, port, slots or extra flags: {exc}")
        self.ctx_source = "manual override" if cfg.get("ctx") else ctxres["source"]
        if recipe.get("ctx_max") and self.ctx > recipe["ctx_max"]:
            warns.append(f"ctx {self.ctx} > engine ceiling {recipe['ctx_max']} for this recipe — clamping.")
            self.ctx = recipe["ctx_max"]
        max_ctx = 350000 if recipe.get("kind") == "xestrata" else 262144
        if not 1 <= self.port <= 65535 or not 1 <= self.slots <= 128 or not 512 <= self.ctx <= max_ctx:
            raise BuildError("Port, slots, or context out of range.")
        gpus = cfg.get("gpus")
        if gpus is None:
            gpus = default_gpus(recipe)
        if not isinstance(gpus, list) or not gpus or len(gpus) != len(set(map(str, gpus))) \
                or any(type(g) is not int or not 0 <= g < 8 for g in gpus):
            raise BuildError("Select one or more distinct GPU indices (0-7).")
        needed = recipe.get("gpus") or []
        if len(needed) > 1 and len(gpus) < len(needed):
            raise BuildError(f"This recipe needs {len(needed)} GPUs. "
                             f"Select {len(needed)} cards, or choose a single-card recipe.")
        self.gpus = gpus
        self.kind = recipe.get("kind", "")
        self.max_gpus = MAX_GPUS.get(self.kind, 1)
        if len(gpus) > 1:
            if self.max_gpus < 2:
                warns.append(f"{engine} serves a single GPU — using GPU {gpus[0]}; extra device selections are ignored.")
                self.gpus = gpus[:1]
            elif self.kind.startswith("gguf") or self.kind in ("vllm", "vllm-tp2", "vllm-autoround"):
                warns.append("Dual-card B70 execution enabled: multi-GPU tensor-split or tensor-parallel active.")
                warns.append("Two-card configuration is experimental and requires matching B70 GPUs, verified device order, sufficient PCIe bandwidth, and engine support.")
        self.envargs, self.extra_env = parse_extra_env(cfg, warns)
        self.kv = cfg.get("kv") or "recipe default"
        self.prefill = cfg.get("prefill") or "int8"
        if self.prefill == "fp16" and self.kind != "exl3":
            warns.append("prefill option applies to EXL3 recipes only; ignored")
        # prefix caching: ON unless the recipe opts out (prefix_caching: false
        # + prefix_caching_note) or the caller passes cache "off".
        recipe_cache = recipe.get("prefix_caching", True) is not False
        note = recipe.get("prefix_caching_note") or "recipe sets prefix_caching false"
        cache = cfg.get("cache")
        if isinstance(cache, bool):
            cache = "on" if cache else "off"
        if cache not in (None, "", "on", "off"):
            warns.append(f"cache '{cache}' isn't on or off — using the recipe default.")
            cache = None
        self.prefix_cache = recipe_cache if not cache else cache == "on"
        self.cache_default = not cache or self.prefix_cache == recipe_cache
        if not recipe_cache and self.prefix_cache:
            warns.append(f"prefix caching forced ON over this recipe's default: {note}.")
        elif not recipe_cache:
            warns.append(f"prefix caching OFF for this recipe: {note}. --cache on forces it.")
        elif not self.prefix_cache:
            warns.append("prefix caching OFF (--cache off): for testing and benchmarks; "
                         "every turn re-reads the whole prompt.")
        self.env = {}
        self.native = False
        preflight = hardware.hardware_preflight() if not IS_WIN else {"devices": []}
        self.preflight = preflight
        devices = preflight["devices"]
        if devices and any(g >= len(devices) for g in self.gpus):
            raise BuildError(f"GPU {max(self.gpus)} is not present; this computer has {len(devices)} GPU(s).")
        selected = [devices[g] for g in self.gpus if g < len(devices)]
        links = [d.get("pcie") or {} for d in selected]
        if len(selected) > 1 and all(p.get("gen") for p in links):
            names = [f"GPU {g}: PCIe {p['gen']}.0 x{p['width']}" for g, p in zip(self.gpus, links)]
            if len({(p["gen"], p["width"]) for p in links}) > 1 or any(p["gen"] < 4 or p["width"] < 8 for p in links):
                warns.append("Cards on different or narrow PCIe links (" + ", ".join(names)
                             + "): multi-card traffic runs at the slowest link.")
        self.render_nodes = [d["render"] for d in selected]
        self.card_nodes = [f"/dev/dri/{d['card']}" for d in selected
                           if re.fullmatch(r"card\d+", str(d.get("card") or ""))]
        self.render_node = self.render_nodes[0] if self.render_nodes else "/dev/dri"
        # power is advisory only: never applied, only warned against the cap
        watts = recipe.get("power")
        if watts:
            sel_dev = preflight["devices"][self.gpus[0]] if self.gpus[0] < len(preflight["devices"]) else None
            if sel_dev and sel_dev.get("power_cap_w") and watts > sel_dev["power_cap_w"]:
                warns.append(f"recipe targets {watts}W, above this card's current {sel_dev['power_cap_w']}W hwmon cap — "
                             "the launcher never changes power settings; benchmark claims above the cap will not reproduce.")
