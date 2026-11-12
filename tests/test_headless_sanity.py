"""A machine with no /proc/mounts, no Docker and no hwmon must not crash the
Storage snapshot or the request log — that is every Windows host and most
containers/WSL chroots. Everything here is simulated with mocks; no GPU,
Docker or network is touched.
"""
import unittest
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import hardware, paths, requestlog, scan, storage

TMP = support_env.TMPROOT
write_file = support_env.write_file
LauncherStateCase = support_env.LauncherStateCase


def _no_read_text(cls_self, *args, **kwargs):
    raise OSError("file system entry not available (simulated missing mount table)")


class TestStorageWithoutMountsOrDocker(LauncherStateCase):
    """/proc/mounts unreadable, docker binary absent, nothing else changed."""

    def setUp(self):
        super().setUp()
        self.root = TMP / "headless-root"
        self.root.mkdir(parents=True, exist_ok=True)
        self.model = write_file(self.root / "Qwen3.8-27B" / "model.safetensors")
        self.entry = {"kind": "vllm", "name": "Qwen3.8-27B", "path": str(self.model.parent),
                      "size_mib": 1, "ctx": None}
        patches = [
            mock.patch.object(scan, "scan_roots", lambda: [self.root]),
            mock.patch.object(scan, "scan_blocking",
                              lambda: {"catalog": [self.entry], "state": "done"}),
            # every read_text fails: /proc/mounts AND the usage history are gone
            mock.patch.object(Path, "read_text", _no_read_text),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        storage.invalidate()
        self.addCleanup(storage.invalidate)

    def test_snapshot_survives_without_proc_mounts_and_usage_history(self):
        with mock.patch.object(storage, "_run",
                               lambda argv, timeout=12: (1, "", "docker not reachable")):
            snap = storage.snapshot(force=True)
        self.assertIsInstance(snap["items"], list)
        self.assertTrue(any(i["kind"] == "model" for i in snap["items"]))
        # an unreadable mount table falls back to "/" for every volume mount
        for v in snap["volumes"]:
            self.assertEqual(v["mount"], "/")
        self.assertTrue(all(d["ok"] is False for d in snap["daemons"]))

    def test_snapshot_survives_when_docker_itself_is_missing(self):
        # subprocess.run raises FileNotFoundError (an OSError) when the binary is absent;
        # _run itself must swallow it and report the daemon as unreachable
        def no_binary(argv, **kw):
            raise FileNotFoundError("docker")

        with mock.patch.object(storage.subprocess, "run", no_binary):
            snap = storage.snapshot(force=True)
        self.assertTrue(all(d["ok"] is False for d in snap["daemons"]))
        self.assertTrue(snap["daemons"][0]["error"])

    def test_delete_still_works_with_the_mount_table_gone(self):
        snap = storage.snapshot(force=True)
        item = next(i for i in snap["items"] if i["kind"] == "model")
        out = storage.delete([item["id"]], mode="trash")
        self.assertTrue(out["results"][0]["ok"], out["results"])
        self.assertFalse(self.model.exists())


class TestRequestLogWithoutHwmon(LauncherStateCase):
    """energy_uj cannot read /sys/class/hwmon: the span records speed, not energy."""

    def setUp(self):
        super().setUp()
        requestlog.LOG_PATH.unlink(missing_ok=True)

    def entry(self):
        return {"served_name": "m", "model_id": "m", "engine": "exl3", "port": 8000,
                "gpus": [0], "ctx": 32768}

    def test_span_finishes_without_hwmon(self):
        import time as _time
        with mock.patch.object(hardware, "energy_uj", side_effect=OSError("no /sys/class/hwmon")):
            span = requestlog.Span(self.entry())
            span.mark_first()
            _time.sleep(0.06)          # decode_tps only records past a 50 ms decode window
            rec = span.finish("pi", 100, 50, True)
        self.assertTrue(rec["ok"])
        self.assertNotIn("card_j", rec)
        self.assertIn("decode_tps", rec)
        self.assertIn("overall_tps", rec)

    def test_summary_with_an_unreadable_log_file(self):
        paths.DATA.mkdir(parents=True, exist_ok=True)
        requestlog.LOG_PATH.write_text("{}\n")
        with mock.patch.object(Path, "read_text", _no_read_text):
            self.assertEqual(requestlog.load(), [])
            out = requestlog.summary()
        self.assertEqual(out["by_model"], [])
        self.assertEqual(out["total_requests"], 0)

    def test_load_skips_a_corrupt_line(self):
        paths.DATA.mkdir(parents=True, exist_ok=True)
        requestlog.LOG_PATH.write_text('{"ok": true}\nnot json\n')
        records = requestlog.load()
        self.assertEqual(records, [{"ok": True}])


if __name__ == "__main__":
    unittest.main()
