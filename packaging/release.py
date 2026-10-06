#!/usr/bin/env python3
"""Package the inspectable Linux source release without invoking build tools."""
import hashlib
import gzip
import io
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
version = "0.4.7"
out = root / "release"
out.mkdir(exist_ok=True)
name = f"b70-launcher-{version}-linux-source"
archive = out / f"{name}.tar.gz"
files = [root / p for p in ("launcher.py", "webwindow.py", "appwindow.py", "recipes.json",
                            "settings.json",
                            "README.md", "packaging/install.sh", "packaging/b70-launcher.desktop",
                            "patches/patch_mtp_nightly.py", "patches/patch_mtp_boundary.py", "patches/patch_vllm_worker_affinity.py", "patches/patch_xpu_grouped_topk_native_v2.py",
                            "web/index.html", "web/assets/b70-launcher.svg",
                            "web/assets/b70-launcher-512.png", "web/assets/b70-launcher-256.png",
                            "web/assets/b70-launcher-128.png", "web/assets/b70-launcher-48.png",
                            "web/assets/b70-launcher-32.png", "web/assets/b70-launcher-16.png")]
manifest = {"version": version, "platform": "Linux source (Python 3.9+)", "files": {}}
with archive.open("wb") as target, gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as zipped:
    with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in files:
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or symlinked release file: {path}")
            rel = path.relative_to(root)
            payload = path.read_bytes()
            manifest["files"][rel.as_posix()] = hashlib.sha256(payload).hexdigest()
            info = tarfile.TarInfo(f"{name}/{rel}")
            info.size = len(payload)
            info.mode = 0o755 if rel.as_posix() == "packaging/install.sh" else 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(payload))
(out / f"{archive.name}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(out / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n")

# Site-side update channel: version.json, the per-recipe manifest, and the
# full recipes document. Copy these three files plus the archive set into the
# site's public/downloads/ on release day.
recipes_doc = json.loads((root / "recipes.json").read_text())
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
    "recipes_url": f"https://xecores.com/downloads/recipes.json",
    "recipes": entries,
}
(out / "recipes-manifest.json").write_text(json.dumps(recipe_manifest, indent=1, sort_keys=True) + "\n")
(out / "recipes.json").write_text(json.dumps(recipes_doc, indent=1) + "\n")
version_doc = {
    "version": version,
    "name": "B70 Launcher",
    "download_url": "https://xecores.com/match",
    "tarball_url": f"https://xecores.com/downloads/{name}.tar.gz",
    "sha256": digest,
    "recipes_manifest_url": "https://xecores.com/downloads/recipes-manifest.json",
    "message": f"B70 Launcher v{version} is available.",
}
(out / "version.json").write_text(json.dumps(version_doc, indent=1) + "\n")

print(f"{archive}\nSHA256 {digest}\nsidecars: version.json, recipes-manifest.json, recipes.json")
