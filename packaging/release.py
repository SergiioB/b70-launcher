#!/usr/bin/env python3
"""Package the inspectable Linux source release without invoking build tools.

Build:   python3 packaging/release.py [--out DIR]
Verify:  python3 packaging/release.py --verify [TARBALL]

The tarball is deterministic: sorted member order, fixed mtime, uid/gid 0.

--verify extracts the tarball to a temp dir and checks:
  * every file the release is expected to contain is present and
    byte-identical to the working-tree source (drift is reported)
  * every payload sha256 matches the .manifest.json sidecar, and the
    manifest covers exactly the tarball contents
  * the .sha256 sidecar parses like `sha256sum -c` expects and matches
    the archive bytes (and `sha256sum -c` itself agrees, when available)
  * member metadata stays deterministic (uid/gid 0, mtime 0)
"""
import argparse
import hashlib
import gzip
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

VERSION = "0.5.1"
ROOT = Path(__file__).resolve().parent.parent
NAME = f"b70-launcher-{VERSION}-linux-source"

# Files shipped in the source bundle, sorted for deterministic tar order.
FILES = sorted(
    "launcher.py cli.py webwindow.py appwindow.py recipes.json settings.json README.md LICENSE "
    "ui.png docs/api.md docs/recipe-format.md docs/cli.md "
    "packaging/install.sh packaging/uninstall.sh packaging/b70-launcher.desktop "
    "patches/patch_champion_stack_overlay.py patches/patch_mtp_boundary.py "
    "patches/patch_mtp_nightly.py patches/patch_vllm_worker_affinity.py "
    "patches/patch_xpu_grouped_topk_native_v2.py "
    "web/index.html ".split()
    + [f"web/assets/{p.name}" for p in (ROOT / "web" / "assets").iterdir()]
    + [f"patches/ssu-b70-b8w4/{p.name}" for p in (ROOT / "patches" / "ssu-b70-b8w4").iterdir()]
)

HEX64 = re.compile(r"[0-9a-fA-F]{64}")
SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64}) ([ *])(\S.*)$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_rel(rel: str) -> bool:
    return bool(rel) and not rel.startswith("/") and ".." not in rel.split("/")


def build(out_dir: Path) -> tuple:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{NAME}.tar.gz"
    manifest = {"version": VERSION, "platform": "Linux source (Python 3.9+)", "files": {}}
    with archive.open("wb") as target, gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for rel in FILES:
                if not safe_rel(rel):
                    raise ValueError(f"Unsafe release path: {rel}")
                path = ROOT / rel
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"Missing or symlinked release file: {path}")
                payload = path.read_bytes()
                manifest["files"][rel] = sha256(payload)
                info = tarfile.TarInfo(f"{NAME}/{rel}")
                info.size = len(payload)
                info.mode = 0o755 if rel.startswith("packaging/") and rel.endswith(".sh") else 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(payload))
    (out_dir / f"{archive.name}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = sha256(archive.read_bytes())
    (out_dir / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n")

    # Site-side update channel: version.json, the per-recipe manifest, and the
    # full recipes document. Copy these three files plus the archive set into
    # the site's public/downloads/ on release day.
    recipes_doc = json.loads((ROOT / "recipes.json").read_text())
    entries = {}
    for m in recipes_doc.get("models", []):
        for eng, r in (m.get("recipes") or {}).items():
            e = {"ver": r.get("recipe_ver") or recipes_doc.get("catalog_ver") or ""}
            if r.get("recipe_note"):
                e["note"] = r["recipe_note"]
            if r.get("recipe_recommended"):
                e["recommended"] = True
            entries[f"{m['id']}:{eng}"] = e
    recipe_manifest = {
        "catalog_ver": recipes_doc.get("catalog_ver") or "",
        "recipes_url": "https://xecores.com/downloads/recipes.json",
        "recipes": entries,
    }
    (out_dir / "recipes-manifest.json").write_text(
        json.dumps(recipe_manifest, indent=1, sort_keys=True) + "\n")
    (out_dir / "recipes.json").write_text(json.dumps(recipes_doc, indent=1) + "\n")
    version_doc = {
        "version": VERSION,
        "name": "B70 Launcher",
        "download_url": "https://xecores.com/match",
        "tarball_url": f"https://xecores.com/downloads/{NAME}.tar.gz",
        "sha256": digest,
        "recipes_manifest_url": "https://xecores.com/downloads/recipes-manifest.json",
        "message": f"B70 Launcher v{VERSION} is available.",
    }
    (out_dir / "version.json").write_text(json.dumps(version_doc, indent=1) + "\n")
    return archive, digest


def verify(archive: Path) -> bool:
    errors, warnings = [], []
    err = errors.append
    warn = warnings.append

    if not archive.is_file():
        print(f"ERROR  tarball not found: {archive}")
        return False
    raw = archive.read_bytes()
    actual = sha256(raw)

    # --- .sha256 sidecar: must parse like `sha256sum -c` expects ----------
    sidecar = archive.with_name(archive.name + ".sha256")
    if sidecar.is_file():
        line = sidecar.read_text().strip()
        m = SHA256_LINE.match(line)
        if not m:
            err(f"{sidecar.name}: not in sha256sum format ('<hash>  <name>' / '<hash> *<name>'): {line!r}")
        else:
            recorded, _, fname = m.groups()
            if fname != archive.name:
                warn(f"{sidecar.name}: references {fname!r}; run `sha256sum -c` from {archive.parent}")
            if recorded.lower() != actual:
                err(f"{sidecar.name}: hash mismatch (recorded {recorded}, actual {actual})")
        if shutil.which("sha256sum"):
            rc = subprocess.run(["sha256sum", "-c", sidecar.name], cwd=archive.parent,
                                capture_output=True, text=True)
            if rc.returncode != 0:
                err(f"`sha256sum -c` rejects the sidecar: {(rc.stdout + rc.stderr).strip()}")
    else:
        err(f"missing sidecar: {sidecar.name}")

    # --- manifest sidecar: schema sanity ----------------------------------
    manifest = None
    manifest_files = None
    manifest_path = archive.with_name(archive.name + ".manifest.json")
    if not manifest_path.is_file():
        err(f"missing sidecar: {manifest_path.name}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            err(f"{manifest_path.name}: invalid JSON: {e}")
        if isinstance(manifest, dict):
            if not isinstance(manifest.get("version"), str) or not manifest["version"]:
                err("manifest: 'version' must be a non-empty string")
            if not isinstance(manifest.get("platform"), str):
                err("manifest: 'platform' must be a string")
            mf = manifest.get("files")
            if not isinstance(mf, dict) or not mf:
                err("manifest: 'files' must be a non-empty object of relpath -> sha256")
            else:
                manifest_files = mf
                for rel, h in mf.items():
                    if not isinstance(rel, str) or not safe_rel(rel):
                        err(f"manifest: unsafe path {rel!r}")
                    if not isinstance(h, str) or not HEX64.fullmatch(h):
                        err(f"manifest: bad sha256 for {rel!r}")
        elif manifest is not None:
            err(f"{manifest_path.name}: top level must be a JSON object")

    # --- extract to a temp dir, auditing members on the way ----------------
    try:
        tar = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
        members = tar.getmembers()
    except tarfile.TarError as e:
        print(f"ERROR  unreadable tarball: {e}")
        return False
    tops, payloads = set(), {}
    for m in members:
        if m.name.startswith("/") or ".." in m.name.split("/"):
            err(f"unsafe member name: {m.name}")
            continue
        tops.add(m.name.split("/")[0])
        if m.isdir():
            continue
        if not m.isreg():
            err(f"non-regular member (links/devices are not shipped): {m.name}")
            continue
        if m.mtime != 0 or m.uid != 0 or m.gid != 0:
            warn(f"non-deterministic metadata on {m.name}: mtime={m.mtime} uid={m.uid} gid={m.gid}")
        rel = "/".join(m.name.split("/")[1:])
        if rel:
            payloads[rel] = tar.extractfile(m).read()
    tar.close()
    if len(tops) != 1:
        err(f"tarball should contain exactly one top directory; found {sorted(tops)}")
    elif not tops:
        err("tarball is empty")

    with tempfile.TemporaryDirectory(prefix="b70-verify-") as td:
        base = Path(td)
        for rel, payload in payloads.items():
            dst = base / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(payload)

        # manifest <-> payload agreement
        if manifest_files is not None:
            for rel in sorted(set(manifest_files) - set(payloads)):
                err(f"in manifest but not in tarball: {rel}")
            for rel in sorted(set(payloads) - set(manifest_files)):
                err(f"in tarball but not in manifest: {rel}")
            for rel in sorted(set(manifest_files) & set(payloads)):
                if sha256(payloads[rel]) != manifest_files[rel]:
                    err(f"payload sha256 does not match manifest: {rel}")

        # expected files present + byte-identical to source where applicable
        for rel in FILES:
            if rel not in payloads:
                err(f"expected file missing from tarball: {rel} (present in source tree; release is stale or incomplete)")
                continue
            src = ROOT / rel
            if not src.is_file():
                warn(f"{rel}: no longer exists in the source tree")
            elif src.read_bytes() != payloads[rel]:
                warn(f"{rel}: differs from the working tree (release is stale or the tree is dirty)")
        for rel in sorted(set(payloads) - set(FILES)):
            warn(f"tarball contains {rel}, which is not in the current FILES list")

    # --- adjacent sidecars used by the update channel ----------------------
    vjson = archive.parent / "version.json"
    if vjson.is_file():
        try:
            v = json.loads(vjson.read_text())
            if isinstance(v, dict):
                if v.get("sha256") and v["sha256"] != actual:
                    warn("version.json: 'sha256' does not match this tarball")
                if manifest and v.get("version") and manifest.get("version") != v["version"]:
                    warn(f"version.json: version {v['version']!r} != manifest version {manifest.get('version')!r}")
        except json.JSONDecodeError:
            warn("version.json: not valid JSON")
    else:
        warn("version.json not found next to the tarball (update channel file)")

    for w in warnings:
        print(f"WARN   {w}")
    for e in errors:
        print(f"ERROR  {e}")
    if errors:
        print(f"FAILED {archive.name}: {len(errors)} error(s), {len(warnings)} warning(s)")
        return False
    print(f"OK {archive.name}: {len(payloads)} payload files verified"
          + (f" ({len(warnings)} warning(s))" if warnings else ""))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "release",
                    help="directory to write the release set into (default: release/)")
    ap.add_argument("--verify", nargs="?", const="", metavar="TARBALL",
                    help="do not build; verify TARBALL (default: the archive --out would hold)")
    args = ap.parse_args()
    if args.verify is not None:
        archive = Path(args.verify) if args.verify else args.out / f"{NAME}.tar.gz"
        return 0 if verify(archive) else 1
    archive, digest = build(args.out)
    print(f"{archive}\nSHA256 {digest}\nsidecars: {archive.name}.sha256, "
          f"{archive.name}.manifest.json, version.json, recipes-manifest.json, recipes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
