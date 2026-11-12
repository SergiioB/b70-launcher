"""Storage manager: sizes, classification, and the delete guards.

No GPU, Docker or network: Docker calls are replaced with stubs, files live in
the isolated temp HOME.
"""
import os
import shutil
import threading
import time
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import scan, storage

TMP = support_env.TMPROOT
write_file = support_env.write_file


class TestSizes(unittest.TestCase):
    def test_parse_size_decimal_units(self):
        self.assertEqual(storage.parse_size("31.8GB"), 31_800_000_000)
        self.assertEqual(storage.parse_size("424MB"), 424_000_000)
        self.assertEqual(storage.parse_size("1.5 TB"), 1_500_000_000_000)
        self.assertEqual(storage.parse_size("12KiB"), 12 * 1024)
        self.assertEqual(storage.parse_size("garbage"), 0)
        self.assertEqual(storage.parse_size(None), 0)

    def test_tree_size_counts_nested_files_and_skips_links(self):
        root = TMP / "st-tree"
        write_file(root / "a.bin", b"x" * 1000)
        write_file(root / "sub" / "b.bin", b"x" * 500)
        os.symlink(root / "a.bin", root / "link.bin")
        size, newest = storage.tree_size(root)
        self.assertGreaterEqual(size, 1500)
        self.assertLess(size, 1500 + 4096 * 4)   # the link adds its own few bytes, not 1000
        self.assertGreater(newest, 0)

    def test_parse_pin(self):
        self.assertEqual(storage._parse_pin("repo/img:1.0@sha256:abc"), ("repo/img", "abc"))
        self.assertEqual(storage._parse_pin("repo/img@sha256:abc"), ("repo/img", "abc"))
        self.assertEqual(storage._parse_pin("local-img"), ("local-img", ""))


class TestDeleteGuards(unittest.TestCase):
    def setUp(self):
        self.root = TMP / "st-models"
        self.root.mkdir(exist_ok=True)
        self.patch = mock.patch.object(scan, "scan_roots", lambda: [self.root])
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_refuses_outside_roots(self):
        outside = TMP / "st-outside" / "model.gguf"
        write_file(outside)
        with self.assertRaisesRegex(ValueError, "outside"):
            storage._delete_model({"path": str(outside)})
        self.assertTrue(outside.exists())

    def test_refuses_a_root_itself(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            storage._delete_model({"path": str(self.root)})
        self.assertTrue(self.root.exists())

    def test_refuses_links_even_inside_a_root(self):
        target = TMP / "st-target-dir"
        write_file(target / "w.bin")
        link = self.root / "link-model"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(target, link)
        # a link that resolves outside every root is refused by the root check first
        with self.assertRaises(ValueError):
            storage._delete_model({"path": str(link)})
        self.assertTrue((target / "w.bin").exists())

    def test_deletes_a_model_dir_inside_a_root(self):
        d = self.root / "m1"
        write_file(d / "config.json")
        write_file(d / "w.safetensors")
        storage._delete_model({"path": str(d)})
        self.assertFalse(d.exists())

    def test_deletes_a_single_file(self):
        f = self.root / "x.gguf"
        write_file(f)
        storage._delete_model({"path": str(f)})
        self.assertFalse(f.exists())

    def test_delete_requires_ids(self):
        for bad in (None, [], "x", [1]):
            with self.assertRaises(ValueError):
                storage.delete(bad)

    def test_unknown_id_is_reported_not_raised(self):
        with mock.patch.object(storage, "snapshot", return_value={"items": []}):
            out = storage.delete(["model:nope"], dry_run=True)
        self.assertFalse(out["results"][0]["ok"])
        self.assertEqual(out["freed_bytes"], 0)

    def test_in_use_and_unmanaged_items_are_refused(self):
        items = [
            {"id": "model:a", "name": "a", "kind": "model", "size_bytes": 10, "deletable": False,
             "why_kept": "a running server is using it"},
            {"id": "image:b", "name": "b", "kind": "image", "size_bytes": 20, "deletable": False,
             "why_kept": "not a pinned launcher image"},
            {"id": "model:c", "name": "c", "kind": "model", "size_bytes": 30, "deletable": True,
             "why_kept": None, "path": str(self.root / "c")},
        ]
        with mock.patch.object(storage, "snapshot", return_value={"items": items}):
            out = storage.delete(["model:a", "image:b", "model:c"], dry_run=True)
        ok = {r["id"]: r["ok"] for r in out["results"]}
        self.assertEqual(ok, {"model:a": False, "image:b": False, "model:c": True})
        self.assertEqual(out["freed_bytes"], 30)

    def test_dry_run_deletes_nothing(self):
        f = self.root / "keep.gguf"
        write_file(f)
        item = {"id": "model:k", "name": "keep", "kind": "model", "size_bytes": 256, "deletable": True,
                "why_kept": None, "path": str(f)}
        with mock.patch.object(storage, "snapshot", return_value={"items": [item]}):
            out = storage.delete(["model:k"], dry_run=True)
        self.assertTrue(f.exists())
        self.assertTrue(out["dry_run"])

    def test_image_removal_never_forces(self):
        calls = []

        def fake_run(argv, timeout=12):
            calls.append(argv)
            return 0, "", ""
        with mock.patch.object(storage, "_run", fake_run):
            storage._delete_image({"socket": "", "image_id": "sha256:abc"})
        self.assertEqual(calls[0][-3:], ["image", "rm", "sha256:abc"])
        self.assertNotIn("-f", calls[0])
        self.assertNotIn("--force", calls[0])

    def test_image_removal_surfaces_dockers_refusal(self):
        with mock.patch.object(storage, "_run", lambda argv, timeout=12: (1, "", "Error: image is being used")):
            with self.assertRaisesRegex(ValueError, "being used"):
                storage._delete_image({"socket": "", "image_id": "sha256:abc"})


class TestTrash(unittest.TestCase):
    """Move-to-trash (freedesktop home trash) instead of a permanent delete."""

    def setUp(self):
        self.root = TMP / "st-trash-root"
        self.root.mkdir(exist_ok=True)
        self.patch = mock.patch.object(scan, "scan_roots", lambda: [self.root])
        self.patch.start()
        self.addCleanup(self.patch.stop)
        files, info = storage._trash_dirs()
        self.trash_files, self.trash_info = files, info

    def test_same_device_trash_moves_and_writes_trashinfo(self):
        name = "st-unique-%s" % os.getpid()
        model = write_file(self.root / name / "model.safetensors")
        method, note = storage._delete_model({"path": str(model.parent)}, mode="trash")
        self.assertEqual(method, "trash")
        self.assertIsNone(note)
        self.assertFalse(model.exists())
        self.assertTrue((self.trash_files / name).is_dir())
        info = self.trash_info / (name + ".trashinfo")
        self.assertTrue(info.is_file())
        body = info.read_text()
        self.assertIn("[Trash Info]", body)
        path_line = urllib.parse.unquote(body.split("Path=")[1].splitlines()[0])
        self.assertEqual(path_line, str(model.parent))
        self.assertRegex(body, r"DeletionDate=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_name_collision_gets_a_suffix(self):
        for i in range(2):
            d = write_file(self.root / f"r{i}" / "Qwen3.8-27B" / "w.bin")
            storage._delete_model({"path": str(d.parent)}, mode="trash")
        self.assertTrue((self.trash_files / "Qwen3.8-27B").exists())
        self.assertTrue((self.trash_files / "Qwen3.8-27B.2").exists())
        self.assertTrue((self.trash_info / "Qwen3.8-27B.2.trashinfo").is_file())

    def test_cross_device_falls_back_to_permanent_with_a_note(self):
        """A model on another filesystem cannot move into the home trash:
        the delete is permanent and says so in the per-item note."""
        model = write_file(self.root / "far-model" / "w.bin")
        with mock.patch.object(storage, "trash_ok", lambda p: False):
            method, note = storage._delete_model({"path": str(model.parent)}, mode="trash")
        self.assertEqual(method, "permanent")
        self.assertIn("no trash", note)
        self.assertFalse(model.exists())

    def test_trash_ok_false_when_trash_cannot_be_created(self):
        with mock.patch.object(storage, "_trash_ready", lambda: False):
            self.assertFalse(storage.trash_ok(str(self.root)))

    def test_delete_reports_the_method_and_refuses_bad_modes(self):
        model = write_file(self.root / "m-report" / "w.bin")
        item = {"id": "model:x", "name": "m-report", "kind": "model", "size_bytes": 8,
                "deletable": True, "why_kept": None, "path": str(model.parent)}
        with mock.patch.object(storage, "snapshot", return_value={"items": [item]}):
            out = storage.delete(["model:x"], mode="trash")
        self.assertTrue(out["results"][0]["ok"])
        self.assertEqual(out["results"][0]["method"], "trash")
        self.assertEqual(out["mode"], "trash")
        with self.assertRaises(ValueError):
            storage.delete(["model:x"], mode="recycle")

    def test_images_ignore_the_trash_mode(self):
        calls = []

        def fake_run(argv, timeout=12):
            calls.append(argv)
            return 0, "", ""

        item = {"id": "image:x", "name": "img", "kind": "image", "size_bytes": 5,
                "deletable": True, "why_kept": None, "socket": "", "image_id": "sha256:abc"}
        with mock.patch.object(storage, "snapshot", return_value={"items": [item]}), \
                mock.patch.object(storage, "_run", fake_run):
            out = storage.delete(["image:x"], mode="trash")
        self.assertEqual(out["results"][0]["method"], "docker")
        self.assertEqual(calls[0][-3:], ["image", "rm", "sha256:abc"])


class TestDeleteJobs(unittest.TestCase):
    """Background deletes: a slow rmtree must not hold the API thread."""

    def setUp(self):
        self.root = TMP / "st-jobs-root"
        self.root.mkdir(exist_ok=True)
        self.catalog = []
        patches = [mock.patch.object(scan, "scan_roots", lambda: [self.root]),
                   mock.patch.object(scan, "scan_blocking",
                                     lambda: {"catalog": self.catalog, "state": "done", "ts": time.time()}),
                   mock.patch.object(storage, "_image_items", lambda *a, **kw: ([], [])),
                   mock.patch.object(scan, "start_scan_async", lambda *a, **kw: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        storage.invalidate()
        self.addCleanup(storage.invalidate)
        with storage._JOBS_LOCK:
            storage.JOBS.clear()

    def entry(self, path):
        return {"kind": "vllm", "name": Path(path).name, "path": str(path), "size_mib": 1, "ctx": None}

    def snap_item(self, path):
        self.catalog = [self.entry(path)]
        storage.invalidate()
        snap = storage.snapshot(force=True)
        return next(i for i in snap["items"] if i["path"] == str(path))

    def wait_done(self, jid, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = storage.job_status(jid)
            if job and job["state"] != "running":
                return job
            time.sleep(0.02)
        self.fail("delete job did not finish in %.0fs" % timeout)

    def test_job_deletes_and_reports_progress(self):
        real_rmtree = shutil.rmtree
        started = threading.Event()

        def slow_rmtree(target, *a, **kw):
            started.set()
            time.sleep(0.15)
            return real_rmtree(target, *a, **kw)

        d = write_file(self.root / "big-model" / "w.bin")
        item = self.snap_item(d.parent)
        with mock.patch.object(storage.shutil, "rmtree", slow_rmtree):
            job = storage.start_delete([item["id"]], mode="trash")
            self.assertEqual(job["state"], "running")
            started.wait(5)
            # the API thread is free while the worker holds the rmtree
            self.assertIsNotNone(storage.job_status(job["id"]))
            done = self.wait_done(job["id"])
        self.assertFalse(d.exists())
        self.assertEqual(done["state"], "done")
        self.assertEqual(done["done"], 1)
        self.assertTrue(done["results"][0]["ok"])
        self.assertEqual(done["results"][0]["method"], "trash")
        self.assertEqual(done["freed_bytes"], item["size_bytes"])

    def test_bad_ids_and_modes_are_refused_before_the_job_starts(self):
        for ids in (None, [], "x", [1]):
            with self.assertRaises(ValueError):
                storage.start_delete(ids)
        with self.assertRaises(ValueError):
            storage.start_delete(["model:x"], mode="recycle")
        self.assertEqual(storage.jobs_status(), [])

    def test_refused_and_unknown_items_are_reported_not_raised(self):
        forged = "model:" + "0" * 10
        d = write_file(self.root / "locked" / "w.bin")
        item = self.snap_item(d.parent)
        locked = dict(item, deletable=False, why_kept="a running server is using it")
        snap2 = {"items": [dict(locked)]}
        with mock.patch.object(storage, "snapshot", return_value=snap2):
            job = storage.start_delete([forged, item["id"]])
            done = self.wait_done(job["id"])
        self.assertEqual(done["state"], "done")
        self.assertEqual([r["ok"] for r in done["results"]], [False, False])
        self.assertIn("running server", done["results"][1]["error"])
        self.assertTrue((self.root / "locked" / "w.bin").exists())

    def test_finished_jobs_are_kept_and_listed_newest_first(self):
        d = write_file(self.root / "j1" / "w.bin")
        iid = self.snap_item(d.parent)["id"]
        job = storage.start_delete([iid])
        self.wait_done(job["id"])
        jobs = storage.jobs_status()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["id"], job["id"])
        self.assertIn("started", jobs[0])


class TestOpenFolder(unittest.TestCase):
    """Open folder resolves ids only — never a raw path from the client."""

    def setUp(self):
        self.root = TMP / "st-open-root"
        self.root.mkdir(exist_ok=True)
        self.catalog = []
        patches = [mock.patch.object(scan, "scan_roots", lambda: [self.root]),
                   mock.patch.object(scan, "scan_blocking",
                                     lambda: {"catalog": self.catalog, "state": "done", "ts": time.time()}),
                   mock.patch.object(storage, "_image_items", lambda *a, **kw: ([], []))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        storage.invalidate()
        self.addCleanup(storage.invalidate)

    def snap_item(self, path):
        self.catalog = [{"kind": "vllm", "name": Path(path).name, "path": str(path),
                         "size_mib": 1, "ctx": None}]
        storage.invalidate()
        snap = storage.snapshot(force=True)
        return next(i for i in snap["items"] if i["path"] == str(path))

    def test_unknown_and_image_ids_are_refused(self):
        with mock.patch.object(storage, "snapshot",
                               return_value={"items": [{"id": "image:x", "kind": "image", "name": "i", "path": "docker · abc"}]}):
            out = storage.open_item_folder("image:x")
        self.assertFalse(out["ok"])
        self.assertIn("docker", out["error"])
        out = storage.open_item_folder("model:nope")
        self.assertFalse(out["ok"])

    def test_opens_the_model_folder_not_a_client_path(self):
        d = write_file(self.root / "openme" / "w.bin")
        iid = self.snap_item(d.parent)["id"]
        seen = []

        def fake_open(folder):
            seen.append(str(folder))
            return True, None

        with mock.patch.object(storage, "open_folder", fake_open):
            out = storage.open_item_folder(iid)
        self.assertTrue(out["ok"])
        self.assertEqual(seen, [str(d.parent)])
        self.assertEqual(out["path"], str(d.parent))

    def test_headless_returns_copy_hint(self):
        d = write_file(self.root / "headless" / "w.bin")
        iid = self.snap_item(d.parent)["id"]
        with mock.patch.object(storage, "open_folder", lambda f: (False, "no display — copy the path instead")):
            out = storage.open_item_folder(iid)
        self.assertFalse(out["ok"])
        self.assertEqual(out["path"], str(d.parent))
        self.assertIn("copy", out["error"])


class TestPartials(unittest.TestCase):
    def test_fresh_partial_is_protected_old_one_is_suggested(self):
        root = TMP / "st-partials"
        old = write_file(root / "old.gguf.part", b"x" * 2048)
        new = write_file(root / "new.gguf.part", b"x" * 2048)
        ago = time.time() - 3 * 86400
        os.utime(old, (ago, ago))
        with mock.patch.object(scan, "scan_roots", lambda: [root]), \
                mock.patch.object(scan, "scan_blocking", lambda: {"catalog": [], "state": "done"}):
            items, _sc = storage._model_items(time.time())
        by = {i["name"]: i for i in items}
        self.assertTrue(by["old.gguf.part"]["suggested"])
        self.assertTrue(by["old.gguf.part"]["deletable"])
        self.assertFalse(by["new.gguf.part"]["suggested"])
        self.assertFalse(by["new.gguf.part"]["deletable"])


if __name__ == "__main__":
    unittest.main()
