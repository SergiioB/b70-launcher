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
from unittest import mock
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
        self.assertEqual(cli.parse_gpus("2,3"), [2, 3])  # four-card hosts

    def test_parse_rejects(self):
        for bad in ("8", "0,0", "", "x", "-1", "1,1,2"):
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
        self.assertEqual(cli.human_ctx(98392), "98,392")
        self.assertEqual(cli.human_gb(32768), "32.0 GiB")
        self.assertEqual(cli.fmt_dur(45), "45s")
        self.assertEqual(cli.fmt_dur(125), "2m05s")

    def test_table_strips_ansi(self):
        t = cli.table([[cli.green("ok"), "1"]], ["A", "B"])
        self.assertIn("ok", cli._strip(t))
        self.assertIn("A", t)

    def test_table_drops_columns_to_fit_narrow_terminal(self):
        rows = [["qwen38-27b", "Qwen3.8-27B long display name", "exl3", "128K"]]
        heads = ["MODEL", "NAME", "ENGINE", "CTX"]
        with mock.patch.dict(os.environ, {"COLUMNS": "40"}):
            narrow = cli.table(rows, heads, drop=(1,))
        with mock.patch.dict(os.environ, {"COLUMNS": "200"}):
            wide = cli.table(rows, heads, drop=(1,))
        self.assertNotIn("NAME", narrow)
        self.assertIn("128K", narrow)
        self.assertTrue(all(len(line) <= 40 for line in narrow.splitlines()))
        self.assertIn("NAME", wide)


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
        {"id": "crash-x-8003", "status": "exited", "port": 8003, "exitcode": 7},
    ]

    def setUp(self):
        self.c = _StubClient(self.RUNNING)

    def test_exact_and_port(self):
        e, _ = cli.resolve_rid(self.c, "qwen38-27b-vllm-8000")
        self.assertEqual(e["port"], 8000)
        e, _ = cli.resolve_rid(self.c, "8001")
        self.assertEqual(e["id"], "ornith-35b-gguf-llamacpp-8001")

    def test_stopped_invisible_but_exited_resolvable(self):
        e, _ = cli.resolve_rid(self.c, "8002")
        self.assertIsNone(e)
        # an exited server keeps its logs reachable by id
        e, _ = cli.resolve_rid(self.c, "crash-x-8003")
        self.assertEqual(e["exitcode"], 7)

    def test_single_default(self):
        c = _StubClient([self.RUNNING[0]])
        e, _ = cli.resolve_rid(c, None)
        self.assertEqual(e["id"], "qwen38-27b-vllm-8000")

    def test_ambiguous_returns_list(self):
        e, _ = cli.resolve_rid(self.c, "8")
        self.assertIsInstance(e, list)
        self.assertEqual(len(e), 3)


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
        tok_file = cls.state / "b70-launcher" / f"token-{cls.port}"
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

    def test_list_ctx_is_the_validated_context(self):
        """CTX must match what the UI preselects: ctx_safe, else ctx."""
        _, out, _ = self.b70("--json", "list")
        models = json.loads(out)["models"]
        rc, pretty, _ = self.b70("list", "-e", "exl3")
        self.assertEqual(rc, 0)
        checked = 0
        for m in models:
            rec = m["recipes"].get("exl3")
            if not rec:
                continue
            line = next(ln for ln in pretty.splitlines() if f" {m['id']} " in ln)
            want = cli.human_ctx(rec.get("ctx_safe") or rec.get("ctx"))
            self.assertIn(f"  {want}  ", line, f"{m['id']}: {line}")
            checked += rec.get("ctx_safe") is not None
        self.assertTrue(checked, "no exl3 recipe with ctx_safe to check")

    def test_launch_without_ctx_uses_validated_context(self):
        rc, out, err = self.b70("--json", "launch", "qwen38-27b", "-e", "llamacpp",
                                "--port", str(free_port()), "--dry-run")
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out)["ctx"], 131072)

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


    def test_launch_dry_run_pretty(self):
        rc, out, _ = self.b70("launch", "qwen38-27b", "-e", "llamacpp",
                              "--port", str(free_port()), "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("launch plan", out)
        self.assertIn("llama", out)

    def test_launch_prefill_fp16_exl3(self):
        rc, out, err = self.b70("--json", "launch", "qwen38-27b-sc5", "-e", "exl3",
                                "--prefill", "fp16", "--port", str(free_port()),
                                "--dry-run")
        self.assertEqual(rc, 0, err)
        self.assertIn('EXL3_INT8_PREFILL="0"', json.loads(out)["cmd"])

    def test_launch_prefill_default_no_set(self):
        rc, out, err = self.b70("--json", "launch", "qwen38-27b-sc5", "-e", "exl3",
                                "--port", str(free_port()), "--dry-run")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("EXL3_INT8_PREFILL", json.loads(out)["cmd"])

    def test_launch_prefill_bad_choice(self):
        rc, _, _ = self.b70("launch", "qwen38-27b-sc5", "-e", "exl3",
                            "--prefill", "fp32", "--dry-run")
        self.assertEqual(rc, 2)

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
        for argv in (("stop", "--all"), ("stop",)):
            rc, out, _ = self.b70(*argv)
            self.assertEqual(rc, 0, argv)
            self.assertIn("nothing", out)

    def test_agent_without_server_points_at_launch(self):
        rc, out, err = self.b70("agent", "pi")
        self.assertEqual(rc, 1)
        self.assertIn("b70 launch", err)
        self.assertEqual(len(err.strip().splitlines()), 1)

    def test_usage_without_traffic(self):
        rc, out, _ = self.b70("usage")
        self.assertEqual(rc, 0)
        self.assertIn("No agent traffic recorded yet.", out)
        rc, out, _ = self.b70("tokens", "--json")
        self.assertEqual(rc, 0)
        self.assertIn("totals", json.loads(out))

    def test_recipes_update_engine_needs_model(self):
        rc, _, err = self.b70("recipes-update", "-e", "exl3")
        self.assertEqual(rc, 2)
        self.assertIn("needs a model", err)

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

    def test_interactive_picker_hides_held_models_and_stars_only_recommended(self):
        import io
        _, out, _ = self.b70("--json", "list")
        models = json.loads(out)["models"]
        state = {"recipes": {"models": models}, "running": [], "preflight": {"foreign": []}}
        buf = io.StringIO()
        with mock.patch("builtins.input", side_effect=["1", "1"]), \
             mock.patch("sys.stdout", buf):
            mid, eng = cli._interactive_pick(state, {})
        rendered = buf.getvalue()
        self.assertNotIn("qwen38-27b-fp8-tp2", rendered)
        self.assertEqual(mid, "qwen36-35b")
        # Legacy badge does not get a recommendation star
        flash_line = next(ln for ln in rendered.splitlines() if "qwen38-flashnext" in ln)
        self.assertNotIn("★", flash_line)

    def test_status_prints_external_gpu_workload(self):
        import io
        payload = {
            "state": {
                "version": "0.6.17",
                "recipes": {"catalog_ver": "2026-10-15.2"},
                "running": [],
                "preflight": {
                    "foreign": [{
                        "id": "bolt-vllm-tp2",
                        "kind": "container",
                        "cname": "bolt-vllm-tp2",
                        "image": "vllm/vllm-openai-xpu:v0.31.0",
                        "gpus": [0, 1],
                        "pids": [1001, 1002],
                    }]
                }
            },
            "metrics": {"power": []},
            "downloads": {},
        }
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            cli._print_status(payload)
        out = buf.getvalue()
        self.assertIn("bolt-vllm-tp2", out)
        self.assertIn("b70 stop bolt-vllm-tp2", out)
        self.assertNotIn("no engines running", out)

    def test_default_gpus_and_interactive_launch_opts(self):
        import argparse
        import io
        # Single-card recipe auto-selects free GPU 1 when GPU 0 is busy in preflight
        pf = {
            "default": {"gpus": [1]},
            "devices": [
                {"label": "Arc Pro B70 · 0b:00.0", "pci": "0000:0b:00.0", "b70": True,
                 "accessible": True, "busy": True, "busy_by": "forge-g5-roworder (PID 1003510)"},
                {"label": "Arc Pro B70 · 0f:00.0", "pci": "0000:0f:00.0", "b70": True,
                 "accessible": True, "busy": False, "busy_by": ""},
            ],
            "foreign": [{"id": "forge-g5-roworder", "kind": "container", "gpus": [0], "pids": [1003510]}],
        }
        self.assertEqual(cli.default_gpus({"kind": "exl3"}, pf), [1])
        self.assertEqual(cli.default_gpus({"kind": "vllm-tp2"}, pf), [0, 1])
        self.assertEqual(cli.parse_ctx("64K"), 65536)
        self.assertEqual(cli.parse_ctx("128k"), 131072)

        # Pressing Enter on GPU/ctx/port selects free GPU 1, validated ctx 131072, and free port
        state = {"running": [], "preflight": pf}
        model = {"id": "qwen38-27b-sc5", "name": "Qwen3.8-27B SC5"}
        recipe = {"kind": "exl3", "ctx": 131072, "ctx_safe": 131072, "ctx_max": 131072}
        a = argparse.Namespace(gpus=None, ctx=None, port=None)
        buf = io.StringIO()
        with mock.patch("builtins.input", side_effect=["", "64K", "8015"]), \
             mock.patch("sys.stdout", buf):
            cli._interactive_launch_opts(state, model, "exl3", recipe, a)
        self.assertEqual(a.gpus, "1")
        self.assertEqual(a.ctx, 65536)
        self.assertEqual(a.port, 8015)
        self.assertIn("busy (forge-g5-roworder", buf.getvalue())
        self.assertIn("GPU 1", buf.getvalue())



if __name__ == "__main__":
    unittest.main()


