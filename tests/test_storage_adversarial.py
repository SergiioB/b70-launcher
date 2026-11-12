"""Adversarial tests for the storage manager, the usage log and the gateway span.

Every test here is an attack written the way an attacker would write it: a
symlinked model folder, a link swapped in under a root, a ``..`` path in an id,
a forged id, a hostile Docker image id, a model folder an active download is
still writing into, a server whose stop failed, a burst of concurrent refreshes,
a request that drew no measurable energy, a passthrough POST that leaves its span
behind, and hostile file names in the Storage/Usage dialogs.

Isolation is the same as ``tests/test_storage.py``: Docker is stubbed, files live
in the temp HOME, launcher state is restored per test.
"""
import json
import os
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import catalog, downloads, gateway, http_api, paths, requestlog, runtime, scan, storage

TMP = support_env.TMPROOT
write_file = support_env.write_file
LauncherStateCase = support_env.LauncherStateCase

# Docker is never called: `snapshot()` otherwise talks to a real daemon.
NO_DOCKER = mock.patch.object(storage, "_image_items", lambda *a, **kw: ([], []))


class StorageCase(LauncherStateCase):
    """A scan root of our own, a stubbed scanner, and no Docker at all."""

    def setUp(self):
        super().setUp()
        self.root = TMP / ("adv-root-" + self._testMethodName)
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog = []
        self.docker = NO_DOCKER.start()
        self.addCleanup(NO_DOCKER.stop)
        patches = [
            mock.patch.object(scan, "scan_roots", lambda: [self.root]),
            mock.patch.object(scan, "scan_blocking",
                              lambda: {"catalog": self.catalog, "state": "done", "ts": time.time()}),
            mock.patch.object(catalog, "SETTINGS", dict(catalog.SETTINGS, scan_dirs=[str(self.root)])),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        storage.invalidate()
        self.addCleanup(storage.invalidate)

    def model_dir(self, name="Qwen3.8-27B", files=("config.json", "model-00001.safetensors")):
        path = self.root / name
        for name_ in files:
            write_file(path / name_, b"x" * 64)
        return path

    def entry(self, path, kind="vllm"):
        return {"kind": kind, "name": Path(path).name, "path": str(path), "size_mib": 1, "ctx": None}

    def snapshot_with(self, entries):
        self.catalog = entries
        storage.invalidate()
        return storage.snapshot(force=True)


# --------------------------------------------------------------- live servers
class TestLiveServerBlocksDelete(StorageCase):
    """A model file may only go when nothing is reading it."""

    def test_running_and_starting_servers_protect_their_model(self):
        path = self.model_dir()
        for status in ("running", "starting", "stopping"):
            with self.subTest(status=status):
                runtime.RUNNING.clear()
                runtime.RUNNING["rid"] = {"id": "rid", "status": status, "port": 8000,
                                          "detected_path": str(path)}
                item = self.snapshot_with([self.entry(path)])["items"][0]
                self.assertFalse(item["deletable"], status)
                self.assertTrue(item["why_kept"])

    def test_stop_failed_server_protects_its_model(self):
        """A stop that failed leaves the engine alive with the weights mapped.

        ``runtime.launch`` reserves GPUs against ``LIVE_STATES | {"stopping",
        "stop_failed"}``, so storage's own ``LIVE_STATES`` must cover it too:
        deleting weights under a live engine crashes it, and a mmapped GGUF
        faults on the next page.
        """
        path = self.model_dir()
        runtime.RUNNING["rid"] = {"id": "rid", "status": "stop_failed", "port": 8000,
                                  "detected_path": str(path)}
        item = self.snapshot_with([self.entry(path)])["items"][0]
        self.assertFalse(item["deletable"], "a stop_failed server still holds the model")
        self.assertTrue(item["why_kept"])
        out = storage.delete([item["id"]], dry_run=True)
        self.assertFalse(out["results"][0]["ok"])


# ------------------------------------------------------------------ downloads
class TestActiveDownloadBlocksDelete(StorageCase):
    """The destination of a running download is not free space."""

    def download(self, dest, state):
        downloads.DOWNLOADS["m-exl3"] = {
            "id": "m-exl3", "model": "Qwen3.8-27B", "model_id": "m", "engine": "exl3",
            "state": state, "done": 1024, "total": 4096, "pct": 25, "speed": 1, "eta": 3,
            "dest": str(dest), "started": time.time(), "finished": None, "file": "model.bin",
            "files_done": 1, "files_total": 4, "resumed_bytes": 0,
        }

    def test_destination_of_a_running_download_is_not_deletable(self):
        path = self.model_dir()
        for state in ("queued", "resolving", "downloading"):
            with self.subTest(state=state):
                downloads.DOWNLOADS.clear()
                self.download(path, state)
                item = self.snapshot_with([self.entry(path)])["items"][0]
                self.assertFalse(item["deletable"], "download is " + state)
                self.assertTrue(item["why_kept"])
                out = storage.delete([item["id"]], dry_run=True)
                self.assertFalse(out["results"][0]["ok"])
                self.assertTrue(path.exists())

    def test_finished_download_does_not_block_delete(self):
        path = self.model_dir()
        self.download(path, "done")
        item = self.snapshot_with([self.entry(path)])["items"][0]
        self.assertTrue(item["deletable"])


# --------------------------------------------------------------------- paths
class TestDeletePathGuards(StorageCase):
    """Links, ``..`` and roots: what ``_delete_model`` must never follow."""

    def test_rmtree_does_not_follow_a_symlink_inside_the_model_dir(self):
        outside = write_file(TMP / "adv-outside-data" / "precious.bin", b"keep me")
        path = self.model_dir()
        os.symlink(outside.parent, path / "linked-dir")
        os.symlink(outside, path / "linked-file")
        item = self.snapshot_with([self.entry(path)])["items"][0]
        storage._delete_model(item)
        self.assertFalse(path.exists())
        self.assertTrue(outside.exists(), "the link target must survive")

    def test_link_swapped_in_under_a_root_is_refused(self):
        victim = self.model_dir("victim")
        target = self.model_dir("selected")
        os.rename(str(target), str(target) + ".real")
        os.symlink(victim, target)
        item = self.snapshot_with([self.entry(target)])["items"][0]
        with self.assertRaises(ValueError):
            storage._delete_model(item)
        self.assertTrue((victim / "config.json").exists())

    def test_link_pointing_outside_every_root_is_refused(self):
        outside = write_file(TMP / "adv-outside-model" / "model.gguf")
        link = self.root / "link-model"
        os.symlink(outside.parent, link)
        with self.assertRaises(ValueError):
            storage._delete_model({"path": str(link)})
        self.assertTrue(outside.exists())

    def test_dotdot_path_is_refused(self):
        write_file(self.root / "m" / "config.json")
        escape = str(self.root / "m" / ".." / ".." / "adv-outside-model" / "model.gguf")
        with self.assertRaises(ValueError):
            storage._delete_model({"path": escape})

    def test_a_scan_root_that_is_a_symlink_still_bounds_the_delete(self):
        real_root = TMP / "adv-real-root"
        write_file(real_root / "m" / "config.json")
        linked_root = TMP / "adv-linked-root"
        if linked_root.is_symlink() or linked_root.exists():
            linked_root.unlink()
        os.symlink(real_root, linked_root)
        outside = write_file(TMP / "adv-outside-root" / "other.gguf")
        with mock.patch.object(scan, "scan_roots", lambda: [linked_root]):
            with self.assertRaises(ValueError):
                storage._delete_model({"path": str(outside)})
            storage._delete_model({"path": str(real_root / "m")})
        self.assertFalse((real_root / "m").exists())
        self.assertTrue(outside.exists())


# ----------------------------------------------------------------------- ids
class TestDeleteIds(StorageCase):
    """The API takes ids, never paths, and an id must name one live item."""

    def test_forged_and_path_shaped_ids_are_refused(self):
        path = self.model_dir()
        self.snapshot_with([self.entry(path)])
        forged = "model:" + "0" * 10
        out = storage.delete([forged, str(path), "model:../../etc/passwd", "image:deadbeef00"])
        self.assertEqual({r["ok"] for r in out["results"]}, {False})
        self.assertEqual(out["freed_bytes"], 0)
        self.assertTrue(path.exists())

    def test_ids_must_be_non_empty_strings(self):
        for bad in (None, [], "model:x", [1], [None], ["model:x", 3]):
            with self.assertRaises(ValueError):
                storage.delete(bad)

    def test_the_same_id_twice_deletes_once(self):
        path = self.model_dir()
        snap = self.snapshot_with([self.entry(path)])
        iid, size = snap["items"][0]["id"], snap["items"][0]["size_bytes"]
        out = storage.delete([iid, iid])
        oks = [r for r in out["results"] if r["ok"]]
        self.assertEqual(len(oks), 1, out["results"])
        self.assertEqual(out["freed_bytes"], size)
        self.assertFalse(path.exists())

    def test_two_threads_deleting_one_item_free_it_once(self):
        path = self.model_dir()
        item = self.snapshot_with([self.entry(path)])["items"][0]
        results = []
        real_rmtree = storage.shutil.rmtree

        def slow_rmtree(target, *a, **kw):
            time.sleep(0.05)
            return real_rmtree(target, *a, **kw)

        def worker():
            with mock.patch.object(storage.shutil, "rmtree", slow_rmtree):
                results.append(storage.delete([item["id"]]))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        oks = sum(1 for r in results for row in r["results"] if row["ok"])
        self.assertEqual(oks, 1, results)
        self.assertEqual(sum(r["freed_bytes"] for r in results), item["size_bytes"])


# --------------------------------------------------------------- scan storms
class TestSnapshotWorkLimits(StorageCase):
    """``/api/storage/refresh`` runs a full disk walk; N of them must not stack."""

    def test_concurrent_refresh_scans_once(self):
        path = self.model_dir()
        self.catalog = [self.entry(path)]
        calls = []

        def slow_scan():
            calls.append(time.monotonic())
            time.sleep(0.15)
            return {"catalog": self.catalog, "state": "done", "ts": time.time()}

        out = []
        with mock.patch.object(scan, "scan_blocking", slow_scan):
            storage.invalidate()
            threads = [threading.Thread(target=lambda: out.append(storage.snapshot(force=True)))
                       for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
        self.assertEqual(len(out), 5)
        self.assertEqual(len(calls), 1, "%d concurrent scans for one refresh burst" % len(calls))

    def test_plain_snapshot_reuses_the_cache(self):
        path = self.model_dir()
        self.catalog = [self.entry(path)]
        calls = []
        with mock.patch.object(scan, "scan_blocking",
                               lambda: (calls.append(1), {"catalog": self.catalog, "state": "done"})[1]):
            storage.invalidate()
            storage.snapshot()
            storage.snapshot()
        self.assertEqual(len(calls), 1)


# ------------------------------------------------------------------ settings
class TestScanRootScope(LauncherStateCase):
    """A scan root is a delete boundary: it must not be a system-wide folder."""

    def settings(self, cfg):
        with mock.patch.object(scan, "start_scan_async", lambda *a, **kw: None), \
                mock.patch.object(catalog, "save_override", lambda *a, **kw: None):
            return http_api._settings(cfg)

    def hostile_roots(self):
        home = Path.home().resolve()
        return [r for r in (str(home.parent), "/etc", "/usr", "/var") if Path(r).is_dir()]

    def test_scan_dirs_may_not_swallow_home_or_system_folders(self):
        roots = self.hostile_roots()
        self.assertTrue(roots, "no system folder to test with")
        for root in roots:
            with self.subTest(root=root):
                with self.assertRaises(ValueError):
                    self.settings({"scan_dirs": [root]})

    def test_download_folder_may_not_be_home_or_a_system_folder(self):
        for root in self.hostile_roots():
            with self.subTest(root=root):
                with self.assertRaises(ValueError):
                    self.settings({"models_dir": root})

    def test_a_plain_project_folder_is_still_allowed(self):
        folder = TMP / "adv-projects" / "models"
        folder.mkdir(parents=True, exist_ok=True)
        self.assertTrue(self.settings({"scan_dirs": [str(folder)]})["ok"])


# ---------------------------------------------------------------- docker argv
class TestDockerArgv(StorageCase):
    """``docker image rm`` gets one argv element and never a shell."""

    def test_hostile_image_id_stays_one_argv_element(self):
        seen = []

        class Done(object):
            returncode, stdout, stderr = 0, "", ""

        def fake_run(argv, **kw):
            seen.append((argv, kw))
            return Done()

        hostile = "sha256:abc; rm -rf / #"
        with mock.patch.object(storage.subprocess, "run", fake_run):
            storage._delete_image({"socket": "", "image_id": hostile})
        argv, kw = seen[0]
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[-3:], ["image", "rm", hostile])
        self.assertEqual(len(argv), 4)
        self.assertFalse(kw.get("shell"))
        self.assertNotIn("--force", argv)

    def test_an_unparseable_docker_line_does_not_raise(self):
        rows = ['sha256:aa|1.2GB|["repo|a:b"]|[]|2026-01-01',
                "sha256:bb|1.2GB|[]|[]|2026-01-01"]

        def fake_run(argv, timeout=12):
            joined = " ".join(argv)
            if "image ls" in joined and "{{.ID}}" in joined:
                return 0, "sha256:aa|1.2GB\nsha256:bb|1.2GB\n", ""
            if "image ls" in joined:
                return 0, "sha256:aa\nsha256:bb\n", ""
            if "inspect" in joined:
                return 0, "\n".join(rows) + "\n", ""
            return 1, "", "docker not reachable"

        with mock.patch.object(storage, "_run", fake_run):
            items, err = storage._docker_images("")
        self.assertEqual(err, "")
        self.assertIsNotNone(items)
        self.assertTrue(all(i["id"].startswith("sha256:") for i in items))


# ------------------------------------------------------------------- usage log
def usage_record(**kw):
    base = {"ts": "2026-11-12 10:00:00", "client": "pi", "model": "Qwen3.8-27B", "engine": "exl3",
            "port": 8000, "gpus": [0], "ctx": 32768, "ok": True, "stream": True,
            "tokens_in": 100, "tokens_out": 50, "wall_s": 1.5, "ttft_s": 0.2,
            "prefill_tps": 500.0, "decode_tps": 38.0}
    base.update(kw)
    return base


class TestUsageSummaryRobustness(LauncherStateCase):
    """``/api/usage`` must never fail: the Usage dialog dies with it."""

    def write(self, records):
        paths.DATA.mkdir(parents=True, exist_ok=True)
        requestlog.LOG_PATH.write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_a_request_that_drew_no_measurable_energy(self):
        """card_j 0.0 rounds avg_w to 0.0, and the energy mean drops falsy 0.0."""
        self.write([usage_record(card_j=0.0, avg_w=0.0, j_per_ktok=0.0)])
        out = requestlog.summary()
        self.assertEqual(len(out["by_model"]), 1)
        self.assertEqual(out["by_model"][0]["requests"], 1)

    def test_mixed_zero_and_real_energy_rows(self):
        self.write([usage_record(card_j=0.0, avg_w=0.0),
                    usage_record(card_j=12.0, avg_w=200.0),
                    usage_record(card_j=None)])
        row = requestlog.summary()["by_model"][0]
        self.assertEqual(row["requests"], 3)
        self.assertIsNotNone(row.get("avg_w"))

    def test_a_normal_energy_row_still_reports(self):
        self.write([usage_record(card_j=12.0, avg_w=200.0)])
        row = requestlog.summary()["by_model"][0]
        self.assertAlmostEqual(row["avg_w"], 200.0)
        self.assertEqual(row["decode_tps_median"], 38.0)

    def test_no_records_is_an_empty_summary(self):
        self.write([])
        out = requestlog.summary()
        self.assertEqual(out["by_model"], [])
        self.assertEqual(out["total_requests"], 0)


class TestRequestLogRing(LauncherStateCase):
    """The log is a ring on disk and one line per request, even for hostile text."""

    def test_trimming_keeps_one_parseable_line_per_record(self):
        requestlog.LOG_PATH.unlink(missing_ok=True)
        with mock.patch.object(requestlog, "KEEP", 20), mock.patch.object(requestlog, "TRIM_AT", 30):
            for i in range(120):
                requestlog.append({"ts": "2026-11-12 10:00:%02d" % (i % 60), "client": "pi",
                                   "model": "m", "ok": True, "tokens_in": 1, "tokens_out": 1,
                                   "wall_s": 0.1, "pad": "y" * 120})
            records = requestlog.load()
        self.assertLessEqual(len(records), 30)
        self.assertGreaterEqual(len(records), 20)
        self.assertEqual(len(requestlog.LOG_PATH.read_text().splitlines()), len(records))

    def test_a_hostile_client_string_cannot_forge_a_log_line(self):
        requestlog.LOG_PATH.unlink(missing_ok=True)
        evil = 'pi"\n{"ts":"2026-11-12 00:00:00","client":"root","ok":true}\r\t%\x1b[2J'
        requestlog.append({"ts": "2026-11-12 10:00:00", "client": evil, "model": "m", "ok": True,
                           "tokens_in": 0, "tokens_out": 0, "wall_s": 0.1})
        lines = requestlog.LOG_PATH.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(requestlog.load()[-1]["client"], evil)


# ---------------------------------------------------------------------- spans
class FakeHandler(object):
    """Enough of BaseHTTPRequestHandler for gateway.handle to run."""

    def __init__(self, path):
        self.path = path
        self.headers = {"Content-Length": "2"}
        self.rfile = BytesIO(b"{}")
        self.wfile = BytesIO()

    def send_response(self, *a, **kw):
        pass

    def send_header(self, *a, **kw):
        pass

    def end_headers(self):
        pass


class TestGatewaySpanHygiene(LauncherStateCase):
    """One request, one span: a span must not outlive the request that opened it."""

    ENTRY = {"id": "rid", "status": "running", "port": 8000, "engine": "exl3",
             "served_name": "m", "gpus": [], "ctx": 32768}

    def setUp(self):
        super().setUp()
        gateway._CURRENT.span = None

    def test_a_passthrough_post_leaves_no_span_behind(self):
        """``/v1/embeddings`` and friends are forwarded without ``record()``.

        The span set before dispatch must be dropped, or the next ``record()``
        reached on the same keep-alive thread (any accounting path that does not
        go through ``handle``) bills it the earlier request's start time and
        energy baseline.
        """
        with mock.patch.object(gateway, "_find_entry", lambda port: dict(self.ENTRY)), \
                mock.patch.object(gateway, "_passthrough", lambda *a, **kw: None):
            gateway.handle(FakeHandler("/gw/pi/8000/v1/embeddings"), "POST")
        self.assertIsNone(getattr(gateway._CURRENT, "span", None),
                          "the span of an unrecorded request stayed on the thread")

    def test_a_get_never_sets_a_span(self):
        with mock.patch.object(gateway, "_find_entry", lambda port: dict(self.ENTRY)), \
                mock.patch.object(gateway, "_passthrough", lambda *a, **kw: None):
            gateway.handle(FakeHandler("/gw/pi/8000/v1/models"), "GET")
        self.assertIsNone(gateway._CURRENT.span)


# ---------------------------------------------------------------------- dialogs
HOSTILE_FIELDS = ("i.name", "i.path", "i.hint", "i.why_kept", "u.name", "u.engine", "u.role",
                  "i.last_launched", "v.mount", "d.name", "d.error", "r.model", "r.engine",
                  "r.client", "i.kind")

WEB = paths.HERE / "web" / "index.html"


def interpolations(source):
    """Every ``${...}`` expression in a JS blob, with balanced braces."""
    out, i = [], 0
    while True:
        start = source.find("${", i)
        if start < 0:
            return out
        depth, j = 0, start + 1
        while j < len(source):
            if source[j] == "{":
                depth += 1
            elif source[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(source[start + 2:j])
        i = j + 1


class TestDialogEscaping(unittest.TestCase):
    """Hostile file names and model names reach both dialogs through innerHTML."""

    def region(self, start_marker, end_marker):
        text = WEB.read_text(encoding="utf-8")
        start = text.find(start_marker)
        self.assertGreaterEqual(start, 0, start_marker)
        end = text.find(end_marker, start)
        self.assertGreater(end, 0, end_marker)
        return text[start:end]

    def unescaped(self, source):
        bad = []
        for expr in interpolations(source):
            if any(field in expr for field in HOSTILE_FIELDS) and "esc(" not in expr:
                bad.append(" ".join(expr.split())[:90])
        return bad

    def test_storage_dialog_escapes_every_server_field(self):
        bad = self.unescaped(self.region("function stRow", "async function showStorage"))
        self.assertEqual(bad, [], "unescaped server values in the Storage dialog: %s" % bad)

    def test_usage_dialog_escapes_every_server_field(self):
        bad = self.unescaped(self.region("function speedTables", "async function showUsage"))
        self.assertEqual(bad, [], "unescaped server values in the Usage dialog: %s" % bad)

    def test_the_escape_helper_covers_every_html_metacharacter(self):
        text = WEB.read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if ln.startswith("const esc="))
        self.assertIn('[&<>"\']', line, "esc() must cover & < > \" and the apostrophe")
        for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
            self.assertIn(entity, line, entity)


if __name__ == "__main__":
    unittest.main()
