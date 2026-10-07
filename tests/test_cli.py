"""Tests for cli.py — the `b70` headless client.

Two layers, same isolation rules as the rest of the suite:

  * unit tests import cli and cover pure helpers (model/engine/gpu
    resolution, formatting, rid matching) with a stubbed client;
  * TestCliE2E runs the real cli.py subprocess against a real launcher.py
    subprocess on an ephemeral port, with HOME/XDG_STATE_HOME/PATH masked —
    no GPU, docker, sudo, or network is ever touched.

Run:  cd b70-launcher && python3 -m unittest discover -s tests -v
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import support_env
except ImportError:
    from tests import support_env

ROOT = support_env.ROOT
TMP = support_env.TMPROOT
free_port = support_env.free_port

sys.path.insert(0, str(ROOT))
import cli  # noqa: E402


class TestGpus(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(cli.parse_gpus(None))
        self.assertEqual(cli.parse_gpus("0"), [0])
        self.assertEqual(cli.parse_gpus("1"), [1])
        self.assertEqual(cli.parse_gpus("0,1"), [0, 1])
        self.assertEqual(cli.parse_gpus("both"), [0, 1])
        self.assertEqual(cli.parse_gpus("dual"), [0, 1])

    def test_parse_rejects(self):
        for bad in ("2", "0,2", "0,0", "", "x", "0,1,2"):
            with self.assertRaises(SystemExit):
                cli.parse_gpus(bad)

    def test_default_from_recipe(self):
        self.assertEqual(cli.default_gpus({}), [0])
        self.assertEqual(cli.default_gpus({"kind": "vllm-tp2"}), [0, 1])
        self.assertEqual(cli.default_gpus({"tp": 2}), [0, 1])
        self.assertEqual(cli.default_gpus({"gpus": [0, 1]}), [0, 1])
        # single-device recipes never get a phantom second card
        self.assertEqual(cli.default_gpus({"gpus": [1]}), [1])
        self.assertEqual(cli.default_gpus({"gpus": [0], "tensor_split": "49,51"}), [0])


class TestPickModel(unittest.TestCase):
    RECIPES = {"models": [
        {"id": "qwen38-27b", "name": "Qwen3.8-27B",
         "recommended_engine": "exl3",
         "recipes": {"exl3": {}, "vllm": {}, "openvino": {}, "llamacpp": {}}},
        {"id": "qwen36-35b", "name": "Qwen3.6-35B-A3B",
         "recommended_engine": "openvino",
         "recipes": {"openvino": {}, "vllm": {}, "llamacpp": {}}},
        {"id": "ornith-35b-gguf", "name": "Ornith 35B",
         "recipes": {"llamacpp": {}}},
    ]}

    def test_exact_and_prefix(self):
        m, _ = cli.pick_model(self.RECIPES, "qwen38-27b")
        self.assertEqual(m["id"], "qwen38-27b")
        m, _ = cli.pick_model(self.RECIPES, "ornith")
        self.assertEqual(m["id"], "ornith-35b-gguf")

    def test_name_substring(self):
        m, _ = cli.pick_model(self.RECIPES, "35B-A3B")
        self.assertEqual(m["id"], "qwen36-35b")

    def test_ambiguous_returns_candidates(self):
        m, cands = cli.pick_model(self.RECIPES, "qwen")
        self.assertIsNone(m)
        self.assertTrue(len(cands) >= 2)

    def test_unknown_suggests(self):
        m, cands = cli.pick_model(self.RECIPES, "qwne38")
        self.assertIsNone(m)
        self.assertEqual([c["id"] for c in cands][0], "qwen38-27b")


class TestPickEngine(unittest.TestCase):
    MODEL = {"id": "m", "recommended_engine": "exl3",
             "recipes": {"vllm": {}, "exl3": {}}}

    def test_explicit(self):
        self.assertEqual(cli.pick_engine(self.MODEL, "vllm"), "vllm")

    def test_recommended(self):
        self.assertEqual(cli.pick_engine(self.MODEL, None), "exl3")

    def test_invalid_exits(self):
        with self.assertRaises(SystemExit):
            cli.pick_engine(self.MODEL, "ovms")


class TestFmt(unittest.TestCase):
    def test_units(self):
        self.assertEqual(cli.human_ctx(131072), "128K")
        self.assertEqual(cli.human_ctx(98392), "98392")
        self.assertEqual(cli.human_gb(32768), "32.0 GiB")
        self.assertEqual(cli.fmt_dur(45), "45s")
        self.assertEqual(cli.fmt_dur(125), "2m05s")

    def test_table_strips_ansi(self):
        t = cli.table([[cli.green("ok"), "1"]], ["A", "B"])
        self.assertIn("ok", cli._strip(t))
        self.assertIn("A", t)


class _StubClient:
    def __init__(self, running=None):
        self._state = {"running": running or []}

    def get(self, path, timeout=None):
        return self._state


class TestParser(unittest.TestCase):
    """Global flags must work before AND after the subcommand."""

    def setUp(self):
        self.p = cli.build_parser()

    def test_json_before_and_after(self):
        self.assertTrue(self.p.parse_args(["--json", "list"]).json)
        self.assertTrue(self.p.parse_args(["list", "--json"]).json)
        self.assertFalse(self.p.parse_args(["list"]).json)

    def test_api_and_token_survive_subcommand(self):
        a = self.p.parse_args(["--api", "http://x:1", "--token", "t", "status"])
        self.assertEqual((a.api, a.token), ("http://x:1", "t"))
        a = self.p.parse_args(["status", "--api", "http://y:2"])
        self.assertEqual(a.api, "http://y:2")

    def test_no_autostart_before_subcommand(self):
        # regression: subparser defaults used to clobber top-level flags
        a = self.p.parse_args(["--no-autostart", "status"])
        self.assertTrue(a.no_autostart)

    def test_quiet_and_no_color(self):
        a = self.p.parse_args(["-q", "--no-color", "status"])
        self.assertTrue(a.quiet and a.no_color)


class TestResolveRid(unittest.TestCase):
    RUNNING = [
        {"id": "qwen38-27b-vllm-8000", "status": "running", "port": 8000,
         "model": "Qwen3.8-27B", "model_id": "qwen38-27b"},
        {"id": "ornith-35b-gguf-llamacpp-8001", "status": "starting",
         "port": 8001, "model": "Ornith", "model_id": "ornith-35b-gguf"},
        {"id": "old-x-8002", "status": "stopped", "port": 8002},
        {"id": "dry-x-8003", "status": "dry-run", "port": 8003},
    ]

    def setUp(self):
        self.c = _StubClient(self.RUNNING)

    def test_exact_and_port(self):
        e, _ = cli.resolve_rid(self.c, "qwen38-27b-vllm-8000")
        self.assertEqual(e["port"], 8000)
        e, _ = cli.resolve_rid(self.c, "8001")
        self.assertEqual(e["id"], "ornith-35b-gguf-llamacpp-8001")

    def test_stopped_and_dryrun_invisible(self):
        e, _ = cli.resolve_rid(self.c, "8002")
        self.assertIsNone(e)
        e, _ = cli.resolve_rid(self.c, "dry-x-8003")
        self.assertIsNone(e)

    def test_single_default(self):
        c = _StubClient([self.RUNNING[0]])
        e, _ = cli.resolve_rid(c, None)
        self.assertEqual(e["id"], "qwen38-27b-vllm-8000")

    def test_ambiguous_returns_list(self):
        e, _ = cli.resolve_rid(self.c, "8")
        self.assertIsInstance(e, list)
        self.assertEqual(len(e), 2)


# ── end to end against a real daemon ──────────────────────────────────────


class TestCliE2E(unittest.TestCase):
    """cli.py subprocess vs. launcher.py subprocess on an ephemeral port."""

    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.home = TMP / "cli-home"
        cls.state = TMP / "cli-state"
        cls.home.mkdir(exist_ok=True)
        cls.state.mkdir(exist_ok=True)
        cls.logf = open(TMP / "cli-server.log", "wb")
        env = support_env.server_env(cls.home, cls.state)
        cls.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "launcher.py"),
             "--no-open", "--port", str(cls.port)],
            stdout=cls.logf, stderr=subprocess.STDOUT,
            cwd=str(ROOT), env=env)
        cls._wait_ready()
        cls.token = cls._token()
        cls.artdir = TMP / "cli-artifacts"
        support_env.write_file(
            cls.artdir / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "proc", None) is not None and cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        if getattr(cls, "logf", None) is not None:
            cls.logf.close()

    @classmethod
    def _wait_ready(cls):
        import urllib.request
        import urllib.error
        deadline = __import__("time").time() + 15
        while __import__("time").time() < deadline:
            if cls.proc.poll() is not None:
                cls.logf.close()
                raise RuntimeError("daemon exited:\n"
                                   + (TMP / "cli-server.log").read_text(errors="replace"))
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{cls.port}/api/state", timeout=1).close()
                return
            except urllib.error.HTTPError as exc:
                if exc.code == 403:
                    return
            except Exception:
                pass
            __import__("time").sleep(0.15)
        raise RuntimeError("daemon not ready in 15s")

    @classmethod
    def _token(cls):
        tok_file = cls.state / "b70-launcher" / "token"
        deadline = __import__("time").time() + 10
        while __import__("time").time() < deadline:
            if tok_file.is_file():
                t = tok_file.read_text().strip()
                if t:
                    return t
            __import__("time").sleep(0.1)
        raise RuntimeError("no token file written")

    def b70(self, *argv, extra_env=None):
        """Run the CLI against the test daemon; return (rc, stdout, stderr)."""
        env = {
            "HOME": str(self.home),
            "XDG_STATE_HOME": str(self.state),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "B70_API": f"http://127.0.0.1:{self.port}",
            "B70_TOKEN": self.token,
            "NO_COLOR": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        if extra_env:
            env.update(extra_env)
        r = subprocess.run([sys.executable, str(ROOT / "cli.py"), *argv],
                           capture_output=True, text=True, timeout=60, env=env)
        return r.returncode, r.stdout, r.stderr

    def test_help(self):
        rc, out, _ = self.b70("--help")
        self.assertEqual(rc, 0)
        self.assertIn("launch", out)
        self.assertIn("quick start", out)

    def test_version_json(self):
        rc, out, _ = self.b70("--json", "version")
        self.assertEqual(rc, 0)
        body = json.loads(out)
        self.assertEqual(body["cli"], cli.CLI_VERSION)
        self.assertTrue(body["daemon"])

    def test_list_json(self):
        rc, out, _ = self.b70("--json", "list")
        self.assertEqual(rc, 0)
        body = json.loads(out)
        ids = {m["id"] for m in body["models"]}
        self.assertIn("qwen36-35b", ids)
        self.assertIn("qwen38-27b", ids)

    def test_list_pretty(self):
        rc, out, _ = self.b70("list")
        self.assertEqual(rc, 0)
        self.assertIn("qwen38-27b", out)
        self.assertIn("exl3", out)

    def test_show(self):
        rc, out, _ = self.b70("show", "qwen38-27b")
        self.assertEqual(rc, 0)
        self.assertIn("Qwen3.8-27B", out)
        self.assertIn("b70 launch qwen38-27b -e exl3", out)

    def test_show_unknown_suggests(self):
        rc, _, err = self.b70("show", "qwne")
        self.assertEqual(rc, 1)
        self.assertIn("qwen", err)

    def test_launch_dry_run_gguf(self):
        rc, out, _ = self.b70("--json", "launch", "qwen36-35b", "-e", "llamacpp",
                              "--ctx", "8192", "--port", str(free_port()),
                              "--dry-run")
        self.assertEqual(rc, 0)
        body = json.loads(out)
        self.assertIn("--cache-type-k", body["cmd"])
        self.assertIn("8192", body["cmd"])

    def test_launch_dry_run_all_engines(self):
        """Every recipe kind resolves through the same launch path."""
        for model, eng in (("qwen36-35b", "openvino"), ("qwen36-35b", "vllm"),
                           ("qwen36-35b", "llamacpp"),
                           ("qwen38-27b-fp8-tp2", "vllm")):
            rc, out, err = self.b70(
                "--json", "launch", model, "-e", eng,
                "--port", str(free_port()), "--dry-run")
            body = json.loads(out) if out.strip() else {}
            self.assertEqual(rc, 0, f"{model}:{eng} → {err or body.get('error')}")
            self.assertTrue(body.get("cmd"), f"{model}:{eng} produced no plan")

    def test_launch_dry_run_exl3_missing_artifact(self):
        # exl3 hard-requires its artifact even for a plan — the CLI must
        # surface the daemon's error, not crash or hang
        rc, out, err = self.b70("launch", "qwen38-27b", "-e", "exl3",
                                "--port", str(free_port()), "--dry-run")
        self.assertEqual(rc, 1)
        self.assertIn("artifact", (err + out).lower())

    def test_launch_dry_run_pretty(self):
        rc, out, _ = self.b70("launch", "qwen38-27b", "-e", "llamacpp",
                              "--port", str(free_port()), "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("launch plan", out)
        self.assertIn("llama", out)

    def test_launch_unknown_model(self):
        rc, _, err = self.b70("launch", "no-such-model")
        self.assertEqual(rc, 1)
        self.assertIn("unknown model", err)
        self.assertIn("try:", err)

    def test_launch_bad_engine(self):
        rc, _, err = self.b70("launch", "ornith-35b-gguf", "-e", "vllm")
        self.assertEqual(rc, 1)
        self.assertIn("llamacpp", err)

    def test_launch_real_blocked_preflight(self):
        # no dry-run: daemon preflight fails (PATH masked → no docker) — the
        # CLI must surface the error, not hang
        rc, _, err = self.b70("launch", "qwen36-35b", "-e", "llamacpp",
                              "--port", str(free_port()), "--no-wait")
        self.assertEqual(rc, 1)
        self.assertTrue(err.strip())

    def test_stop_nothing_running(self):
        rc, out, _ = self.b70("stop", "--all")
        self.assertEqual(rc, 0)
        self.assertIn("nothing", out)

    def test_status_json(self):
        rc, out, _ = self.b70("--json", "status")
        self.assertEqual(rc, 0)
        body = json.loads(out)
        self.assertIn("running", body)
        self.assertIn("metrics", body)

    def test_doctor(self):
        rc, out, _ = self.b70("doctor")
        self.assertEqual(rc, 0)
        self.assertIn("doctor", out)
        self.assertIn("Docker", out)  # masked PATH → docker blocker shown

    def test_scan_and_settings(self):
        rc, out, _ = self.b70("scan", "--roots", str(self.artdir))
        self.assertEqual(rc, 0)
        self.assertIn("qwen3.6-35b-a3b-ud-q4_k_xl.gguf", out.lower())

    def test_open_print(self):
        rc, out, _ = self.b70("open", "--print")
        self.assertEqual(rc, 0)
        self.assertIn(f"http://127.0.0.1:{self.port}/?token={self.token}", out)

    def test_env_requires_server(self):
        rc, _, err = self.b70("env")
        self.assertEqual(rc, 1)
        self.assertIn("no engines running", err)

    def test_daemon_down_no_autostart(self):
        rc, _, err = self.b70("--no-autostart", "--api",
                              f"http://127.0.0.1:{free_port()}", "status")
        self.assertEqual(rc, 3)
        self.assertIn("not running", err)

    def test_completion(self):
        rc, out, _ = self.b70("completion", "bash")
        self.assertEqual(rc, 0)
        self.assertIn("_b70", out)


if __name__ == "__main__":
    unittest.main()
