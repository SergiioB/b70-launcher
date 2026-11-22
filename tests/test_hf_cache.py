"""Issue #18: detect models already stored in the Hugging Face hub cache.

HF layout: models--<org>--<repo>/{blobs,refs,snapshots/<rev>}/ where snapshot
files are symlinks into the repo's blobs/. Tests build that layout in tmp dirs
and never touch a real cache.
"""
import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import catalog, scan
from b70launcher.builders import common

TMP = support_env.TMPROOT
LauncherStateCase = support_env.LauncherStateCase
write_file = support_env.write_file

REV = "ed40e7da625f594fc9c4990f5f66be24873c22d7"


def hf_snapshot(hub, org, repo, rev=REV, files=None, ref_main=None):
    """Build models--org--repo with a snapshot dir of symlinked blob files.

    ``files`` maps name -> "blob" (link into blobs/), "real" (plain file) or a
    symlink target string (explicit link, e.g. escaping the repo)."""
    repodir = Path(hub) / f"models--{org}--{repo}"
    snap = repodir / "snapshots" / rev
    (repodir / "blobs").mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    for i, (name, kind) in enumerate((files or {}).items()):
        dest = snap / name
        if kind == "blob":
            blob = repodir / "blobs" / f"{i:064x}"
            blob.write_bytes(b"\0" * 1024)
            dest.symlink_to(os.path.relpath(blob, snap))
        elif kind == "real":
            dest.write_bytes(b"local-content")
        else:
            dest.symlink_to(kind)
    if ref_main is not None:
        (repodir / "refs").mkdir(exist_ok=True)
        write_file(repodir / "refs" / "main", ref_main.encode())
    return repodir, snap


VLLM_FILES = {"config.json": "blob", "model.safetensors": "blob",
              "tokenizer.json": "blob"}


class _HfEnvCase(LauncherStateCase):
    """Isolate the HF env vars hf_cache_roots() reads."""
    def setUp(self):
        super().setUp()
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for k in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
            os.environ.pop(k, None)

    def tearDown(self):
        self._env.stop()
        super().tearDown()


class TestHfCacheRoots(_HfEnvCase):
    def test_hf_hub_cache_env_wins(self):
        hub = TMP / "hub-env"
        hub.mkdir(exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(hub)
        self.assertEqual(scan.hf_cache_roots(), [hub.resolve()])

    def test_hf_home_gets_hub_appended(self):
        home = TMP / "hf-home"
        (home / "hub").mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(home)
        self.assertIn((home / "hub").resolve(), scan.hf_cache_roots())

    def test_default_xdg_location(self):
        xdg = TMP / "xdg-cache"
        hub = xdg / "huggingface" / "hub"
        hub.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg)
        self.assertEqual(scan.hf_cache_roots(), [hub.resolve()])

    def test_missing_dirs_are_skipped(self):
        os.environ["HF_HUB_CACHE"] = str(TMP / "no-such-hub")
        os.environ["XDG_CACHE_HOME"] = str(TMP / "empty-xdg")
        self.assertEqual(scan.hf_cache_roots(), [])


class TestHfHubWalk(LauncherStateCase):
    def _walk(self, hub):
        items = {"gguf": {}, "snapshots": {}}
        done = scan._walk_hf_hub(hub, items, time.time() + 30)
        return done, items

    def test_snapshot_detected_as_vllm_artifact(self):
        repodir, snap = hf_snapshot(TMP / "hub1", "thinkingmachines",
                                  "Inkling-2-Mini", files=VLLM_FILES)
        done, items = self._walk(TMP / "hub1")
        self.assertTrue(done)
        ent = items["snapshots"]["inkling-2-mini"]
        self.assertEqual(ent["kind"], "vllm")
        self.assertEqual(ent["path"], str(snap))
        self.assertEqual(ent["repo_hint"], "thinkingmachines/Inkling-2-Mini")
        self.assertEqual(ent["hf_repo_dir"], str(repodir))
        self.assertEqual(ent["size_mib"], round(3 * 1024 / 1048576, 1))

    def test_refs_main_picks_its_revision(self):
        _, snap_main = hf_snapshot(TMP / "hub2", "org", "Model", rev="a" * 40,
                                   files=VLLM_FILES, ref_main="a" * 40)
        hf_snapshot(TMP / "hub2", "org", "Model", rev="b" * 40, files=VLLM_FILES)
        _, items = self._walk(TMP / "hub2")
        self.assertEqual(items["snapshots"]["model"]["path"], str(snap_main))

    def test_escaping_symlink_fails_closed(self):
        outside = TMP / "outside-secret.bin"
        write_file(outside)
        hf_snapshot(TMP / "hub3", "evil", "Model",
                    files={"config.json": "blob", "model.safetensors": str(outside)})
        _, items = self._walk(TMP / "hub3")
        self.assertNotIn("model", items["snapshots"])

    def test_symlinked_repo_dir_outside_hub_is_skipped(self):
        real = TMP / "elsewhere" / "models--org--Model"
        (real / "snapshots" / REV).mkdir(parents=True)
        write_file(real / "blobs" / "x")
        hub = TMP / "hub4"
        hub.mkdir(exist_ok=True)
        (hub / "models--org--Model").symlink_to(real, target_is_directory=True)
        _, items = self._walk(hub)
        self.assertEqual(items["snapshots"], {})

    def test_non_model_and_dataset_dirs_ignored(self):
        hf_snapshot(TMP / "hub5", "org", "Model", files={"README.md": "blob"})
        ds = TMP / "hub5" / "datasets--org--ds" / "snapshots" / REV
        ds.mkdir(parents=True)
        write_file(ds / "config.json")
        write_file(ds / "model.safetensors")
        _, items = self._walk(TMP / "hub5")
        self.assertEqual(items["snapshots"], {})

    def test_openvino_hf_artifact_classification(self):
        files = {"openvino_language_model.xml": "blob",
                 "openvino_language_model.bin": "blob", "config.json": "blob"}
        hf_snapshot(TMP / "hub6", "org", "OV-Model", files=files)
        _, items = self._walk(TMP / "hub6")
        self.assertEqual(items["snapshots"]["ov-model"]["kind"], "openvino")

    def test_ctx_read_through_symlinked_config(self):
        repodir, snap = hf_snapshot(TMP / "hub7", "org", "Ctx-Model",
                                    files={"config.json": "real"})
        (snap / "config.json").write_text(json.dumps({"max_position_embeddings": 262144}))
        write_file(snap / "model.safetensors")
        _, items = self._walk(TMP / "hub7")
        self.assertEqual(items["snapshots"]["ctx-model"]["ctx"], 262144)


class TestHfScanWorkerAndDetect(_HfEnvCase):
    def test_scan_worker_walks_hf_hub_cache(self):
        hub = TMP / "hub-scan"
        hf_snapshot(hub, "thinkingmachines", "Inkling-2-Mini", files=VLLM_FILES)
        root = TMP / "scan-root"
        root.mkdir(exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(hub)
        catalog.SETTINGS["scan_dirs"] = [str(root)]
        scan.scan_worker()
        self.assertIn(str(hub.resolve()), scan.SCAN["roots"])
        self.assertIn("inkling-2-mini", scan.SCAN["items"]["snapshots"])

    def test_hub_inside_scan_root_found_via_models_walk_once(self):
        parent = TMP / "models-parent"
        hf_snapshot(parent / "hub", "org", "Inside-Model", files=VLLM_FILES)
        os.environ["HF_HUB_CACHE"] = str(parent / "hub")
        catalog.SETTINGS["scan_dirs"] = [str(parent)]
        scan.scan_worker()
        self.assertIn("inside-model", scan.SCAN["items"]["snapshots"])
        # the hub is covered by the scan root, not listed as its own root
        self.assertNotIn(str((parent / "hub").resolve()), scan.SCAN["roots"])

    def test_detect_inkling2_mini_from_hf_cache(self):
        repodir, snap = hf_snapshot(TMP / "hub-det", "thinkingmachines",
                                  "Inkling-2-Mini", files=VLLM_FILES, ref_main=REV)
        items = {"gguf": {}, "snapshots": {}}
        scan._walk_hf_hub(TMP / "hub-det", items, time.time() + 30)
        scan.SCAN["items"] = items
        det = scan.detect(catalog.find_model("inkling2-mini"), "vllm")
        self.assertIsNotNone(det)
        self.assertEqual(det["path"], str(snap))
        self.assertEqual(det["hf_repo_dir"], str(repodir))
        self.assertFalse(det["variant"])  # exact identity, not a variant warning
        det2, reason = scan.detect_status(catalog.find_model("inkling2-mini"), "vllm")
        self.assertEqual(reason, "")

    def test_container_mount_keeps_blob_links_alive(self):
        repodir, snap = hf_snapshot(TMP / "hub-mount", "org", "Model", files=VLLM_FILES)
        det = {"path": str(snap), "hf_repo_dir": str(repodir)}
        model_path, src, dst = common.container_path(det, "/models/model")
        self.assertEqual(model_path, f"/models/hf-repo/snapshots/{REV}")
        self.assertEqual((src, dst), (str(repodir), "/models/hf-repo"))
        # ../../blobs/<sha> resolves inside the mount
        inner = Path(dst) / "snapshots" / REV / "../../blobs/x"
        self.assertEqual(os.path.normpath(str(inner)), "/models/hf-repo/blobs/x")

    def test_artifact_mount_non_hf_is_unchanged(self):
        det = {"path": "/data/My-Model"}
        self.assertEqual(common.artifact_mount(det), ("/data/My-Model", ""))
        f = write_file(TMP / "weights.gguf")
        det_f = {"path": str(f)}
        self.assertEqual(common.artifact_mount(det_f),
                         (str(TMP), "weights.gguf"))


if __name__ == "__main__":
    unittest.main()
