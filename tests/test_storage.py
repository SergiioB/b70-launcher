"""Storage manager: sizes, classification, and the delete guards.

No GPU, Docker or network: Docker calls are replaced with stubs, files live in
the isolated temp HOME.
"""
import os
import time
import unittest
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
