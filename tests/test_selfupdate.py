"""In-app update: only a verified tarball from version.json's own folder is installed."""
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env
from b70launcher import catalog, paths, selfupdate


class SelfUpdateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.marker = self.dir / "installed.txt"
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.object(catalog, "UPDATE_URL", f"file://{self.dir}/version.json")
        patch.start()
        self.addCleanup(patch.stop)
        selfupdate.STATE.update(state="idle", step="", version="", error="", log="")

    def _publish(self, version="99.0.0", digest=None, url=None):
        buf = io.BytesIO()
        script = f'echo installed > "{self.marker}"\n'.encode()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(f"b70-launcher-{version}-linux-source/packaging/install.sh")
            info.size = len(script)
            tar.addfile(info, io.BytesIO(script))
        data = buf.getvalue()
        (self.dir / "launcher.tar.gz").write_bytes(data)
        (self.dir / "version.json").write_text(json.dumps({
            "version": version,
            "tarball_url": url or f"file://{self.dir}/launcher.tar.gz",
            "sha256": digest or hashlib.sha256(data).hexdigest()}))

    def run_worker(self):
        selfupdate._worker()
        return selfupdate.STATE

    def test_verified_tarball_runs_its_installer(self):
        self._publish()
        state = self.run_worker()
        self.assertEqual(state["state"], "done", state["error"])
        self.assertEqual(state["version"], "99.0.0")
        self.assertTrue(self.marker.is_file())

    def test_checksum_mismatch_installs_nothing(self):
        self._publish(digest="0" * 64)
        state = self.run_worker()
        self.assertEqual(state["state"], "failed")
        self.assertIn("SHA256 mismatch", state["error"])
        self.assertFalse(self.marker.exists())

    def test_tarball_outside_version_folder_is_refused(self):
        self._publish(url="file:///tmp/elsewhere/launcher.tar.gz")
        state = self.run_worker()
        self.assertEqual(state["state"], "failed")
        self.assertFalse(self.marker.exists())

    def test_same_version_is_not_an_update(self):
        self._publish(version=paths.VERSION)
        state = self.run_worker()
        self.assertEqual(state["state"], "failed")
        self.assertIn("No newer launcher", state["error"])

    def test_source_checkout_refuses_to_update_itself(self):
        with self.assertRaises(ValueError):
            selfupdate.start_install()

    def test_restart_requires_an_installed_update(self):
        with self.assertRaises(ValueError):
            selfupdate.restart()

    def test_update_check_carries_short_release_notes(self):
        self.addCleanup(setattr, catalog, "UPDATE_INFO", catalog.UPDATE_INFO)
        remote = dict(catalog.RECIPE_REMOTE)
        self.addCleanup(lambda: (catalog.RECIPE_REMOTE.clear(), catalog.RECIPE_REMOTE.update(remote)))
        (self.dir / "version.json").write_text(json.dumps({
            "version": "99.0.0",
            "notes": ["Faster  start.\n", 7, "x" * 500] + [f"note {i}" for i in range(10)]}))
        catalog.check_for_updates()
        notes = catalog.UPDATE_INFO["notes"]
        self.assertTrue(catalog.UPDATE_INFO["has_update"])
        self.assertEqual(len(notes), 8)
        self.assertEqual(notes[0], "Faster start.")
        self.assertEqual(len(notes[1]), 300)

    def test_release_notes_come_from_the_readme_section(self):
        spec = importlib.util.spec_from_file_location(
            "b70_release", Path(__file__).resolve().parent.parent / "packaging" / "release.py")
        release = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(release)
        readme = ("## What is new in 1.2.3\n\n"
                  "- **Short lead.** First sentence here. Second one.\n"
                  "- Plain `code` bullet with [a link](https://example.com).\n"
                  "## What is new in 1.2.2\n\n- **Old.** Old text.\n")
        self.assertEqual(release.release_notes(readme, "1.2.3"),
                         ["Short lead. First sentence here.", "Plain code bullet with a link."])
        self.assertEqual(release.release_notes(readme, "9.9.9"), [])


if __name__ == "__main__":
    unittest.main()
