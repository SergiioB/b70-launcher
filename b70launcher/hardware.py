"""Hardware preflight and telemetry probes.

``hardware_preflight`` computes the cached base result (devices, blockers,
profile). Busy-GPU attribution is a separate enrichment step so callers that
know the launcher's own servers can mark owned VRAM; the base cache never
stores ownership. Render-node clients identify non-desktop GPU workloads.
Without a readable client, unowned VRAM above 4 GiB is a conservative
fallback; it does not identify a workload.
"""
import glob
import json
import re
import sys
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import catalog
from .paths import IS_WIN

PREFLIGHT_CACHE = {"data": None, "ts": 0.0}
POWER_PREV = {}   # pci -> (ts, energy1_input uJ) for live watt estimation
VRAM_CACHE = {}   # pci -> (ts, used_gb, total_gb); _vram_mm sudo-forks per GPU

BUSY_VRAM_MIB = 4096  # heuristic floor — proof requires engine attribution
CLIENT_CACHE = {"ts": 0.0, "data": []}


def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def default_docker_sock():
    socket_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/run/user/" + str(os.getuid())) + "/docker.sock")
    if not socket_path.exists():
        socket_path = Path("/var/run/docker.sock")
    return socket_path


def recipe_sockets():
    """Every docker socket the shipped recipes may use (default + isolated)."""
    socks = {""}
    for m in catalog.RECIPES.get("models", []):
        for r in (m.get("recipes") or {}).values():
            s = r.get("docker_sock")
            if isinstance(s, str) and s:
                socks.add(s)
    return socks


def _docker_blockers():
    out = []
    if not shutil.which("docker"):
        out.append("Docker CLI missing; install Docker and a supported GPU runtime.")
    if os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"):
        out.append("Docker remote context/environment configured; only a local Docker socket is supported.")
    socket_path = default_docker_sock()
    try:
        if not socket_path.exists() or not os.access(socket_path, os.R_OK | os.W_OK):
            out.append("Local Docker socket unavailable or inaccessible.")
    except OSError:
        out.append("Local Docker socket unavailable or inaccessible.")
    return out


# ----------------------------------------------------------------- PCIe link

# trained link speed (GT/s) -> generation, and the approximate usable
# one-direction bandwidth per lane (GB/s) at that generation
PCIE_GEN_BY_GT_S = ((2.5, 1), (5.0, 2), (8.0, 3), (16.0, 4), (32.0, 5), (64.0, 6))
PCIE_GT_S_BY_GEN = {gen: gt_s for gt_s, gen in PCIE_GEN_BY_GT_S}
PCIE_GBPS_PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.877}
PCIE_WIDTHS = (1, 2, 4, 8, 16)
PCI_ADDR = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
PCI_DOMAIN_DIR = re.compile(r"^pci[0-9a-fA-F]{4}:[0-9a-fA-F]{2}$")


def _link_gt_s(path):
    """Trained speed from a sysfs ``*link_speed`` file ("16.0 GT/s PCIe")."""
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    try:
        gt_s = float(text.split()[0])
    except (IndexError, ValueError):
        return None
    return gt_s if gt_s > 0 else None


def _link_width(path):
    width = _read_int(path)
    return width if width and width > 0 else None


def _link_gen(gt_s):
    for speed, gen in PCIE_GEN_BY_GT_S:
        if gt_s is not None and abs(gt_s - speed) < 0.01:
            return gen
    return None


def _link_gbps(gen, width):
    """Approximate usable one-direction bandwidth (GB/s) of a gen x width link."""
    if gen in PCIE_GBPS_PER_LANE and width:
        return round(PCIE_GBPS_PER_LANE[gen] * width, 1)
    return None


def _clean_pcie_entry(entry):
    """``{"gen", "width"}`` for one valid override entry, else None."""
    if not isinstance(entry, dict) or set(entry) != {"gen", "width"}:
        return None
    gen, width = entry["gen"], entry["width"]
    if isinstance(gen, bool) or not isinstance(gen, int) or not 1 <= gen <= 6:
        return None
    if isinstance(width, bool) or not isinstance(width, int) or width not in PCIE_WIDTHS:
        return None
    return {"gen": gen, "width": width}


def clean_pcie_links(value, current=None):
    """Validate a posted ``pcie_links`` map and merge it onto ``current``.

    ``{pci: {"gen": 1-6, "width": 1|2|4|8|16}}``; a null entry drops that
    card's override, a null map clears them all. Raises ValueError (-> HTTP
    400) on a malformed address, entry or field."""
    merged = {}
    if isinstance(current, dict):
        for pci, entry in current.items():
            clean = _clean_pcie_entry(entry)
            if clean and isinstance(pci, str) and PCI_ADDR.match(pci):
                merged[pci] = clean
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("PCIe link overrides must map a PCI address to gen and width.")
    for pci, entry in value.items():
        if not isinstance(pci, str) or not PCI_ADDR.match(pci):
            raise ValueError(f"Not a PCI address: {pci if isinstance(pci, str) else type(pci).__name__}.")
        if entry is None:
            merged.pop(pci, None)
            continue
        clean = _clean_pcie_entry(entry)
        if clean is None:
            raise ValueError("Each PCIe link override needs gen 1-6 and width 1, 2, 4, 8 or 16.")
        merged[pci] = clean
    return merged


def _pcie_chain(real_path):
    """``(pci####:## anchor, [nested path, ...])`` of a resolved sysfs device path."""
    parts = Path(real_path).parts
    for idx, part in enumerate(parts):
        if PCI_DOMAIN_DIR.match(part):
            anchor = Path(*parts[:idx + 1])
            nested, current = [], anchor
            for part in parts[idx + 1:]:
                current = current / part
                nested.append(current)
            return anchor, nested
    return None, []


def pcie_link(pci, sysfs_root="/sys", overrides=None):
    """Trained PCIe link of one card, as the launcher UI reads it.

    gen/width come from the ROOT PORT (the ancestor directly under
    ``pci####:##``); max_gen/max_width are the card's own capability, read from
    that port's child (the card's uplink, e.g. its internal switch upstream
    port). Deeper switch links report a virtual 2.5 GT/s x1 on a B70 and are
    ignored. A ``settings["pcie_links"]`` entry for this card wins over sysfs
    (``source`` "user") and keeps the sysfs values in ``detected``."""
    unknown = {"gen": None, "width": None, "gt_s": None, "max_gen": None,
               "max_width": None, "gbps": None, "source": "unknown", "detected": None}
    detected = None
    if not IS_WIN and isinstance(pci, str) and PCI_ADDR.match(pci):
        try:
            real = (Path(sysfs_root) / "bus" / "pci" / "devices" / pci).resolve()
        except OSError:
            real = None
        anchor, chain = _pcie_chain(real) if real else (None, [])
        if anchor is not None and chain:
            port = chain[0]                     # root port
            uplink = chain[1] if len(chain) > 1 else port
            gt_s = _link_gt_s(port / "current_link_speed")
            width = _link_width(port / "current_link_width")
            gen = _link_gen(gt_s)
            if gen and width:
                detected = {"gen": gen, "width": width,
                            "max_gen": _link_gen(_link_gt_s(uplink / "max_link_speed")),
                            "max_width": _link_width(uplink / "max_link_width")}
    link = dict(unknown)
    if detected:
        link.update(gen=detected["gen"], width=detected["width"], gt_s=gt_s,
                    max_gen=detected["max_gen"], max_width=detected["max_width"],
                    gbps=_link_gbps(detected["gen"], detected["width"]),
                    source="sysfs", detected=detected)
    if overrides is None:
        overrides = catalog.SETTINGS.get("pcie_links")
    choice = _clean_pcie_entry(overrides.get(pci)) if isinstance(overrides, dict) else None
    if choice:
        link.update(gen=choice["gen"], width=choice["width"], gt_s=PCIE_GT_S_BY_GEN[choice["gen"]],
                    gbps=_link_gbps(choice["gen"], choice["width"]), source="user",
                    detected=detected)
    return link


def hardware_preflight():
    """Base preflight (cached 10 s): devices, union blockers, profile, default."""
    import stat as stat_m
    now = time.time()
    if PREFLIGHT_CACHE["data"] and now - PREFLIGHT_CACHE["ts"] < 10:
        return PREFLIGHT_CACHE["data"]
    devices = []
    if IS_WIN:
        result = {"devices": devices, "blockers": ["Windows GPU execution is not validated; use a supported Linux host."],
                  "notes": [], "profile": {"b70_count": 0, "cards": []}}
        PREFLIGHT_CACHE.update({"data": result, "ts": now})
        return result
    # map render nodes to their DRM card so sysfs/hwmon telemetry can be attached
    card_of = {}
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        try:
            card_of[card.joinpath("device").resolve()] = card.name
        except OSError:
            continue
    for render in sorted(Path("/sys/class/drm").glob("renderD*"), key=lambda p: int(p.name[7:])):
        dev = render / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
            device = (dev / "device").read_text().strip().lower()
            subsystem_vendor = (dev / "subsystem_vendor").read_text().strip().lower()
            subsystem_device = (dev / "subsystem_device").read_text().strip().lower()
            pci = dev.resolve().name
            b70 = (vendor, device, subsystem_vendor, subsystem_device) == ("0x8086", "0xe223", "0x8086", "0x1701")
            label = f"Arc Pro B70 · {pci[5:]}" if b70 else ""
            if not label and shutil.which("lspci"):
                text = subprocess.run(["lspci", "-s", pci], capture_output=True,
                                      text=True, timeout=3).stdout.strip()
                label = f"{text.partition(': ')[2] or text} · {pci[5:]}"
            vram_total = _read_int(dev / "mem_info_vram_total")
            vram_used = _read_int(dev / "mem_info_vram_used")
            if b70 and not vram_total:
                vram_total = 32656 * 1024 * 1024  # visible VRAM on the 32 GB B70
            cap_w = None
            for hm in Path("/sys/class/hwmon").glob("hwmon*"):
                try:
                    if hm.joinpath("device").resolve() != dev.resolve():
                        continue
                    c = _read_int(hm / "power1_cap")
                    if c:
                        cap_w = round(c / 1_000_000)
                    break
                except OSError:
                    continue
            devices.append({"render": f"/dev/dri/{render.name}", "pci": pci,
                            "vendor": vendor, "device": device, "label": label,
                            "card": card_of.get(dev.resolve()),
                            "accessible": os.access(f"/dev/dri/{render.name}", os.R_OK | os.W_OK),
                            "b70": b70,
                            "vram_total_mib": round(vram_total / 1048576, 1) if vram_total else None,
                            "vram_used_mib": round(vram_used / 1048576, 1) if vram_used else None,
                            "power_cap_w": cap_w,
                            "pcie": pcie_link(pci)})
        except (OSError, subprocess.TimeoutExpired):
            continue
    gpu_blockers = []
    if not devices:
        gpu_blockers.append("No DRM render nodes found; install a working Intel GPU driver.")
    if not any(d["b70"] for d in devices):
        gpu_blockers.append("No Arc Pro B70 identified by Intel PCI subsystem ID 8086:1701; other Intel GPUs are unvalidated.")
    if not any(d["b70"] and d["accessible"] for d in devices):
        gpu_blockers.append("No accessible Arc Pro B70 render node; check the render/video group.")
    docker_blockers = _docker_blockers()
    b70s = [d for d in devices if d["b70"] and d["accessible"]]
    profile = {"b70_count": len(b70s), "cards": [
        {"render": d["render"], "label": d["label"], "vram_total_mib": d["vram_total_mib"],
         "power_cap_w": d["power_cap_w"], "accessible": d["accessible"], "b70": d["b70"]}
        for d in devices]}
    blockers = gpu_blockers + docker_blockers
    result = {"devices": devices, "blockers": blockers,
              "blockers_by_kind": blockers_by_kind(gpu_blockers, docker_blockers),
              "notes": ["GPU container compatibility and model fit are not proven by this preflight."],
              "profile": profile,
              "default": best_default(b70s, devices)}
    PREFLIGHT_CACHE.update({"data": result, "ts": now})
    return result


def blockers_by_kind(gpu_blockers=(), docker_blockers=()):
    """Per-kind launch blockers (audit M9).

    native : GPU only (a llama_bin process needs no docker at all)
    docker : GPU + docker CLI + the default local socket
    exl3   : GPU + docker CLI (its own isolated socket is auto-started on
             launch, so the default socket is irrelevant)"""
    gpu_blockers = list(gpu_blockers)
    docker_blockers = list(docker_blockers)
    cli_missing = [b for b in docker_blockers if b.startswith("Docker CLI missing")]
    default_sock = [b for b in docker_blockers if b.startswith("Local Docker socket")
                    or b.startswith("Docker remote context")]
    return {
        "native": list(gpu_blockers),
        "docker": gpu_blockers + cli_missing + default_sock,
        "exl3": gpu_blockers + cli_missing,
    }


def gpu_clients():
    """Read GPU client PIDs and render nodes without exposing command lines.

    A child that inherited a desktop app's render fd shares that app's DRM
    client id; such entries are the desktop's client, not a new workload."""
    if time.monotonic() - CLIENT_CACHE["ts"] < 10:
        return CLIENT_CACHE["data"]
    code = r'''
import glob,json,os,re
desktop=re.compile(r"^(?:gnome|chrome|chromium|brave|firefox|orca-ide|zcode|antigravity|language_server|slack|code|electron|discord|obsidian|signal-desktop|thunderbird|WebKit|gjs|Xwayland|Xorg|systemd|nautilus|spotify|kitty|tern|codexbar|gpaste|mutter|ibus|pipewire|wireplumber|xdg-desktop-portal|gsd-|at-spi|dbus-daemon)",re.I)
procs=[];desktop_ids=set()
for proc in glob.glob("/proc/[0-9]*"):
 try:
  exe=os.path.basename(os.readlink(proc+"/exe"))
  quiet=bool(desktop.match(exe))
  if exe.startswith("python") and not quiet:
   with open(proc+"/cmdline","rb") as stream: args=stream.read().split(b"\0")
   quiet=len(args)>1 and os.path.basename(args[1]) in (b"webwindow.py",b"appwindow.py")
  clients=set()
  for fd in glob.glob(proc+"/fd/*"):
   try:
    target=os.readlink(fd)
    if not target.startswith("/dev/dri/renderD"): continue
    cid=""
    try:
     with open(proc+"/fdinfo/"+os.path.basename(fd)) as info:
      cid=next((l.split(":",1)[1].strip() for l in info if l.startswith("drm-client-id:")),"")
    except OSError: pass
    clients.add((os.path.basename(target),cid))
   except OSError: pass
  if not clients: continue
  if quiet:
   desktop_ids.update(c for c in clients if c[1]); continue
  procs.append((int(os.path.basename(proc)),exe,clients,open(proc+"/cgroup").read()))
 except OSError: pass
out=[{"pid":pid,"exe":exe,"render":node,"cgroup":cg} for pid,exe,clients,cg in procs
     for node in {n for n,c in clients if not c or (n,c) not in desktop_ids}]
print(json.dumps(out))
'''
    py = shutil.which("python3")
    if not py:
        return []
    argv = [py, "-c", code]
    data = []
    for command in ([["sudo", "-n"] + argv] if shutil.which("sudo") else []) + [argv]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                break
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    CLIENT_CACHE.update(ts=time.monotonic(), data=data)
    return data


_DOCKER_CG_RE = re.compile(r"(?:docker[-/]|containers/)([0-9a-f]{64})")


def _resolve_foreign_containers(clients):
    """Map 64-hex container IDs found in client cgroups to container metadata."""
    cids = {}
    for p in clients:
        m = _DOCKER_CG_RE.search(p.get("cgroup") or "")
        if m:
            cids.setdefault(m.group(1), None)
    if not cids or not shutil.which("docker"):
        return {}
    for sock in recipe_sockets():
        unresolved = [cid for cid, info in cids.items() if info is None]
        if not unresolved:
            break
        try:
            cmd = ["docker"] + (["-H", f"unix://{sock}"] if sock else [])
            res = subprocess.run(
                cmd + ["inspect", "--format", "{{.Id}}\t{{.Name}}\t{{.Config.Image}}", *unresolved],
                capture_output=True, text=True, timeout=3)
            for line in res.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and len(parts[0]) == 64:
                    cname = parts[1].lstrip("/")
                    img = parts[2] if len(parts) > 2 else ""
                    cids[parts[0]] = {"id": cname, "cid": parts[0], "cname": cname,
                                      "image": img, "docker_sock": sock}
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {k: v for k, v in cids.items() if v is not None}


def enrich_busy(preflight, owned_gpus=(), owned_cnames=(), owned_pids=()):
    """Attribute actual clients per card, with a fallback VRAM busy threshold."""
    owned_gpus = {int(g) for g in owned_gpus}
    out = {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
           for k, v in preflight.items()}
    devices = []
    owned_pids = {int(p) for p in owned_pids if p}
    owned_ids = []
    if owned_cnames and shutil.which("docker"):
        for sock in recipe_sockets():
            try:
                cmd = ["docker"] + (["-H", f"unix://{sock}"] if sock else [])
                result = subprocess.run(cmd + ["inspect", "--format", "{{.Id}}", *owned_cnames],
                                        capture_output=True, text=True, timeout=3)
                owned_ids.extend(x for x in result.stdout.splitlines() if len(x) == 64)
            except (OSError, subprocess.TimeoutExpired):
                pass
    clients = [p for p in gpu_clients() if p["pid"] not in owned_pids
               and not any(cid in p.get("cgroup", "") for cid in owned_ids)]
    cmap = _resolve_foreign_containers(clients)
    foreign_map = {}
    for idx, d in enumerate(out["devices"]):
        d = dict(d)
        used = d.get("vram_used_mib")
        if used is None:
            used_gb, _tot = _vram_mm(d.get("pci") or "")
            used = round(used_gb * 1024, 1) if used_gb is not None else None
        d["vram_used_mib"] = used
        foreign = [p for p in clients if p["render"] == Path(d.get("render", "")).name]
        busy = bool(foreign) or idx in owned_gpus or used is not None and used >= BUSY_VRAM_MIB
        d["busy"] = busy
        busy_parts = []
        for p in foreign:
            m = _DOCKER_CG_RE.search(p.get("cgroup") or "")
            cinfo = cmap.get(m.group(1)) if m else None
            if cinfo:
                lbl = f"{cinfo['cname']} ({p['exe']} PID {p['pid']})"
                f_ent = foreign_map.setdefault(
                    cinfo["cname"],
                    {"id": cinfo["cname"], "kind": "container", "cname": cinfo["cname"],
                     "image": cinfo["image"], "docker_sock": cinfo["docker_sock"],
                     "gpus": [], "pids": []})
                if idx not in f_ent["gpus"]:
                    f_ent["gpus"].append(idx)
                if int(p["pid"]) not in f_ent["pids"]:
                    f_ent["pids"].append(int(p["pid"]))
            elif m:
                lbl = f"{p['exe']} (PID {p['pid']})"
            else:
                lbl = f"{p['exe']} (PID {p['pid']})"
                fid = f"pid:{p['pid']}"
                f_ent = foreign_map.setdefault(
                    fid,
                    {"id": fid, "kind": "process", "pid": int(p["pid"]),
                     "exe": p["exe"], "gpus": [], "pids": [int(p["pid"])]})
                if idx not in f_ent["gpus"]:
                    f_ent["gpus"].append(idx)
            if lbl not in busy_parts:
                busy_parts.append(lbl)
        d["busy_by"] = ", ".join(busy_parts)
        if busy and not d["busy_by"]:
            d["busy_by"] = "Launcher server" if idx in owned_gpus else "unknown (VRAM in use)"
        devices.append(d)
    out["devices"] = devices
    out["foreign"] = list(foreign_map.values())
    # default launch card: first free accessible B70 (contract /api/state)
    free = [i for i, d in enumerate(devices) if d.get("b70") and d.get("accessible") and not d.get("busy")]
    dflt = dict(out.get("default") or {})
    if free:
        dflt["gpus"] = [free[0]]
    else:
        dflt["gpus"] = []
    out["default"] = dflt
    return out


def best_default(b70s, devices):
    """Hardware-aware default model + engine for this machine."""
    first = catalog.RECIPES["models"][0] if catalog.RECIPES.get("models") else {"id": "", "recipes": {}}
    if b70s:
        vram = sum((d.get("vram_total_mib") or 0) for d in b70s)
        model = catalog.find_model("qwen36-35b") or first
        return {"model_id": model.get("id", ""), "engine": next(iter(model.get("recipes", {})), ""),
                "reason": f"{len(b70s)}× Arc Pro B70 detected"
                          + (f" · {vram / 1024:.0f} GB total" if vram else "")}
    return {"model_id": first.get("id", ""), "engine": next(iter(first.get("recipes", {})), ""),
            "reason": ("Intel GPU detected but no validated B70 — launch is blocked, dry-run only"
                       if devices else "No GPU detected yet — install the Intel driver, dry-run only")}


def _vram_mm(pci):
    """Real VRAM usage from xe debugfs (same source as the desktop dashboard).

    Each call sudo-forks `cat`, so results are cached briefly — the telemetry
    poller hits this per GPU every couple of seconds."""
    hit = VRAM_CACHE.get(pci)
    now = time.time()
    if hit and now - hit[0] < 10:
        return hit[1], hit[2]
    used_gb = total_gb = None
    try:
        res = subprocess.run(["sudo", "-n", "cat",
                              f"/sys/kernel/debug/dri/{pci}/tile0/vram_mm"],
                             capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            avail = total = 0
            for line in res.stdout.splitlines():
                if line.startswith("visible_avail:"):
                    avail = int(line.split()[1].replace("MiB", ""))
                elif line.startswith("visible_size:"):
                    total = int(line.split()[1].replace("MiB", ""))
            if total:
                used_gb, total_gb = round((total - avail) / 1024, 2), round(total / 1024, 1)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    VRAM_CACHE[pci] = (now, used_gb, total_gb)
    return used_gb, total_gb


def _act_freq(pci):
    for rel in ("tile0/gt0/freq0/act_freq", "tile0/gt0/freq0/cur_freq"):
        v = _read_int(Path(f"/sys/bus/pci/devices/{pci}/{rel}"))
        if v:
            return v
    return None


def power_probe():
    """Live per-card watts from xe hwmon energy deltas + temp + real VRAM + clock-based util."""
    out = []
    now = time.time()
    idx = 0
    labels = {}
    base = PREFLIGHT_CACHE.get("data")
    if base:
        for i, d in enumerate(base.get("devices", [])):
            # real lspci label when present, else the validated B70 name —
            # never label every xe card "Arc Pro B70" (audit L6)
            labels[d.get("pci")] = (d.get("label").split(" (")[0] if d.get("label") else
                                    ("Arc Pro B70" if d.get("b70") else f"Intel GPU {i}"))
    for hm in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (hm / "name").read_text().strip() != "xe":
                continue
            dev = hm.joinpath("device").resolve()
            pci = dev.name
            energy = _read_int(hm / "energy1_input")
            cap = _read_int(hm / "power1_cap")
            temp = _read_int(hm / "temp2_input") or _read_int(hm / "temp1_input")
            used_gb, total_gb = _vram_mm(pci)
            freq = _act_freq(pci)
            maxfreq = None
            for rel in ("tile0/gt0/freq0/max_freq", "tile0/gt0/freq0/rpn_freq"):
                v = _read_int(Path(f"/sys/bus/pci/devices/{pci}/{rel}"))
                if v:
                    maxfreq = v
                    break

            entry = {
                "index": idx,
                "name": f"GPU {idx} {labels.get(pci) or 'Intel GPU'}",
                "pci": pci,
                "active": True,
                "watts": None,
                "cap_w": round(cap / 1_000_000) if cap else 150,
                "temp_c": round(temp / 1000) if temp else None,
                "freq_mhz": freq,
                "util_pct": None,
                "vram_total_gb": total_gb,
                "vram_used_gb": used_gb,
            }
            if freq and maxfreq:
                entry["util_pct"] = max(0, min(100, round(freq / maxfreq * 100)))
            prev = POWER_PREV.get(pci)
            if energy is not None and prev and now > prev[0] and energy >= prev[1]:
                entry["watts"] = round((energy - prev[1]) / (now - prev[0]) / 1_000_000, 1)
            if energy is not None:
                POWER_PREV[pci] = (now, energy)
            out.append(entry)
            idx += 1
        except OSError:
            continue
    return out


def vram_probe():
    """Best real VRAM reading the OS exposes, with its source. None = not exposed."""
    if IS_WIN:
        return None
    for usedp in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")):
        try:
            used = int(Path(usedp).read_text().strip())
            totalp = Path(usedp).parent / "mem_info_vram_total"
            total = int(totalp.read_text().strip()) if totalp.exists() else None
            return {"used_mib": round(used / 1048576, 1),
                    "total_mib": round(total / 1048576, 1) if total else None,
                    "source": "sysfs"}
        except Exception:
            continue
    return None
