"""Redesign regressions: readiness gating, duplicate starts, and the
session identity/target contract.

Consumer-visible bugs these pin down:
- a session that is not ready must never have its endpoint dialed
  (/api/test_prompt, /api/harness) or its client opened;
- a live session refuses a second launch of the same recipe and holds its
  port, including the mid-launch window a snapshot tick must not re-open;
- what a session IS (recipe identity + target app) is frozen at launch:
  later recipe-file, override or settings drift never rewrites it;
- a recipe file whose pinned artifact is not on disk reports "incompatible",
  not "not downloaded".

No exact wording is asserted, only outcomes and documented state categories.

Run:  cd b70-launcher && python3 -m unittest discover -s tests -v
"""
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import builders, catalog, harness, http_api, paths, runtime, scan

TMP = support_env.TMPROOT
LauncherStateCase = support_env.LauncherStateCase


def _entry(rid, status, port, **extra):
    """Tracked-session fixture in the shape runtime/servers_snapshot expects.
    No `pid` key: that is exactly the mid-launch (registered, not yet
    spawned) state."""
    logf = TMP / f"{rid}.log"
    logf.write_text("")
    entry = {"id": rid, "status": status, "model": "m", "model_id": "m1",
             "model_name": "m", "served_name": "m", "engine": "llamacpp",
             "cfg": {"no_test": True}, "test": {"state": "passed"},
             "cname": "", "endpoint": f"http://127.0.0.1:{port}/v1",
             "port": port, "native": True, "proc": None, "log": str(logf),
             "started": "s", "launch_ts": time.time() - 30, "ready_at": None,
             "phase": "loading weights", "cmd": "c", "artifact_mib": 1,
             "tokens_in": 0, "tokens_out": 0, "requests": 0,
             "sess_in": 0, "sess_out": 0, "sess_reqs": 0}
    entry.update(extra)
    return entry


class TestReadinessGating(LauncherStateCase):
    """Nothing may talk to an endpoint until the session says it is ready."""

    def _recorder(self):
        hits = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                hits.append(("GET", self.path))
                body = json.dumps({"data": [{"id": "served-test-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                hits.append(("POST", self.path))
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = json.dumps({
                    "choices": [{"message": {"content": "B70-OK"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 3}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_port, hits

    def test_unready_session_is_never_dialed(self):
        port, hits = self._recorder()
        ask = http_api.POST_ROUTES["/api/test_prompt"]
        for status in ("starting", "exited", "stopping"):
            rid = f"gate-{status}"
            runtime.RUNNING[rid] = _entry(rid, status, port)
            out = ask({"id": rid})
            self.assertFalse(out["ok"], status)
            self.assertTrue(out["error"], status)
            self.assertIsNone(out.get("http_status"), status)
            self.assertEqual(hits, [], status)
            # the CLI `b70 ask` form resolves by port and gets the same gate
            out = ask({"port": port})
            self.assertFalse(out["ok"], status)
            self.assertEqual(hits, [], status)
            del runtime.RUNNING[rid]

    def test_untracked_port_is_still_probed_honestly(self):
        port, hits = self._recorder()
        out = http_api.POST_ROUTES["/api/test_prompt"]({"port": port})
        self.assertTrue(hits, "a port no session owns must still get a real attempt")
        self.assertTrue(any(m == "POST" for m, _ in hits))
        self.assertTrue(out["ok"])

    def test_client_open_requires_a_ready_session(self):
        ask = http_api.POST_ROUTES["/api/harness"]
        # nothing listens on this port, so no snapshot tick can flip the
        # fixture to ready mid-test
        rid = "harness-gate"
        runtime.RUNNING[rid] = _entry(rid, "starting", support_env.free_port())
        with self.assertRaises(ValueError):
            ask({"id": rid, "harness": "none"})
        self.assertEqual(runtime.RUNNING[rid]["status"], "starting")
        with self.assertRaises(ValueError):
            ask({"id": "no-such-server", "harness": "none"})

    def test_no_client_target_reports_no_open_attempt(self):
        port, _ = self._recorder()
        rid = "harness-none"
        # a running native session is backed by a live pid
        runtime.RUNNING[rid] = _entry(rid, "running", port, pid=os.getpid())
        out = http_api.POST_ROUTES["/api/harness"]({"id": rid, "harness": "none"})
        self.assertFalse(out["opened"])
        self.assertTrue(out["line"])
        self.assertNotIn("client_open", runtime.RUNNING[rid])


class TestDuplicateStarts(LauncherStateCase):
    """One live session per launch target; the guard never lapses."""

    def _plan(self, model_id, port):
        cfg = {"model_id": model_id, "engine": "llamacpp", "port": port,
               "ctx": 8192, "gpus": [0], "no_test": True}
        return cfg, builders.build(dict(cfg, dry_run=True))

    def test_live_session_refuses_relaunch_of_same_recipe(self):
        cfg, plan = self._plan("qwen36-35b", support_env.free_port())
        rid = plan["rid"]
        for status in ("starting", "running", "stopping", "stop_failed"):
            entry = _entry(rid, status, cfg["port"])
            runtime.RUNNING[rid] = entry
            with mock.patch.dict(os.environ, {"PATH": str(support_env.EMPTY_BIN)}):
                with self.assertRaises(ValueError):
                    runtime.launch(cfg, plan)
            self.assertIs(runtime.RUNNING[rid], entry, status)
            self.assertEqual(runtime.RUNNING[rid]["status"], status)

    def test_mid_launch_window_is_not_reopened_by_a_snapshot_tick(self):
        cfg, plan = self._plan("qwen36-35b", support_env.free_port())
        rid = plan["rid"]
        runtime.RUNNING[rid] = _entry(rid, "starting", cfg["port"])
        items = {i["id"]: i for i in runtime.servers_snapshot()}
        self.assertEqual(items[rid]["status"], "starting")
        self.assertFalse(items[rid]["ready"])
        with mock.patch.dict(os.environ, {"PATH": str(support_env.EMPTY_BIN)}):
            with self.assertRaises(ValueError):
                runtime.launch(cfg, plan)

    def test_dead_native_process_is_still_reaped(self):
        port = support_env.free_port()
        rid = "native-dead"
        entry = _entry(rid, "starting", port)
        runtime.RUNNING[rid] = entry
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait(timeout=10)
        entry["pid"] = child.pid
        runtime.servers_snapshot()
        self.assertEqual(runtime.RUNNING[rid]["status"], "exited")

    def test_live_session_reserves_its_port_for_other_launches(self):
        port = support_env.free_port()
        runtime.RUNNING["port-owner"] = _entry("port-owner", "running", port)
        cfg, plan = self._plan("qwen38-27b", port)
        self.assertNotEqual(plan["rid"], "port-owner")
        with mock.patch.dict(os.environ, {"PATH": str(support_env.EMPTY_BIN)}):
            with self.assertRaises(ValueError):
                runtime.launch(cfg, plan)
        self.assertNotIn(plan["rid"], runtime.RUNNING)
        self.assertEqual(runtime.RUNNING["port-owner"]["status"], "running")


class TestSessionIdentity(LauncherStateCase):
    """Recipe identity (model_id + engine + variant facts) and the target app
    are a launch-time snapshot; catalog/settings drift never rewrites them."""

    ID_KEYS = ("quant", "variant", "found_name", "custom", "recipe_ver",
               "ctx_source")

    def test_identity_and_target_are_frozen_at_launch(self):
        catalog.SETTINGS["open_with"] = "omp"
        cfg = {"model_id": "qwen36-35b", "engine": "llamacpp",
               "port": support_env.free_port(), "ctx": 8192, "gpus": [0],
               "no_test": True}
        plan = builders.build(dict(cfg, dry_run=True))
        with mock.patch.dict(os.environ, {"PATH": str(support_env.EMPTY_BIN)}):
            rid = runtime.launch(cfg, plan)
        entry = runtime.RUNNING[rid]
        for key in self.ID_KEYS:
            self.assertEqual(entry[key], plan[key], key)
        self.assertEqual(entry["ctx"], plan["ctx"])
        self.assertEqual(entry["client"], "omp")  # target app from settings

        cfg2 = dict(cfg, port=support_env.free_port(), gpus=[1], harness="webui")
        plan2 = builders.build(dict(cfg2, dry_run=True))
        with mock.patch.dict(os.environ, {"PATH": str(support_env.EMPTY_BIN)}):
            rid2 = runtime.launch(cfg2, plan2)
        self.assertEqual(runtime.RUNNING[rid2]["client"], "webui")  # request wins

        # the recipe file, its overrides and the default client may drift
        # after launch; the session keeps the identity it was started with
        catalog.SETTINGS["open_with"] = "none"
        catalog._apply_recipe_overrides(
            {"qwen36-35b": {"llamacpp": {"ctx": 2048}}})
        catalog.apply_recipe_doc({"models": [{"id": "qwen36-35b",
            "recipes": {"llamacpp": {"ctx": 4096, "download": {"quant": "drifted"}}}}]})
        runtime.servers_snapshot()
        after = runtime.RUNNING[rid]
        self.assertEqual(after["client"], "omp")
        self.assertEqual(after["ctx"], plan["ctx"])
        for key in self.ID_KEYS:
            self.assertEqual(after[key], plan[key], key)
        self.assertEqual(runtime.RUNNING[rid2]["client"], "webui")
        self.assertEqual(runtime.RUNNING[rid2]["ctx"], plan2["ctx"])


class TestRecipeFileMismatch(LauncherStateCase):
    """A recipe whose pinned artifact is not on disk must say so honestly:
    incompatible (something of the family is there) vs not downloaded."""

    def setUp(self):
        super().setUp()
        self.dir = TMP / "session-detect"
        self.dir.mkdir(parents=True, exist_ok=True)
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}

    def _add(self, *names):
        for n in names:
            scan.SCAN["items"]["gguf"][n.lower()] = str(
                support_env.write_file(self.dir / n))

    def test_exact_artifact_reports_no_reason(self):
        self._add("Qwen3.8-27B-UD-Q4_K_M.gguf")
        det, reason = scan.detect_status(catalog.find_model("qwen38-27b"), "llamacpp")
        self.assertIsNotNone(det)
        self.assertEqual(reason, "")

    def test_wrong_quant_of_the_family_is_incompatible(self):
        self._add("Qwen3.8-27B-UD-Q8_0.gguf")
        det, reason = scan.detect_status(catalog.find_model("qwen38-27b"), "llamacpp")
        self.assertIsNone(det)
        self.assertIn("incompatible", reason)
        self.assertNotIn("not downloaded", reason)

    def test_absent_artifact_reports_not_downloaded(self):
        det, reason = scan.detect_status(catalog.find_model("ornith-35b-gguf"), "llamacpp")
        self.assertIsNone(det)
        self.assertIn("not downloaded", reason)


class TestAgentOpenModel(LauncherStateCase):
    """Opening an agent for a ready session starts it on the engine-served
    name — what /v1/models reported at ready time — with nothing to pick:

    - pi/omp take `--model b70/<served_name>` and the synced provider file
      lists exactly that id;
    - droid selects its stable custom id whose synced entry carries the
      served name;
    - codex gets `-m <served_name>` plus the provider overrides;
    - deepseek's --patch overlay pins the served model as the web default.

    A served_name that drifted from the launch-time model_name must win."""

    def _open_line(self, client, entry):
        """The exact line /api/harness would hand to a terminal, without a
        display or binary check — HeadlessError carries it."""
        with mock.patch.object(harness, "_installed", return_value=True), \
                mock.patch.object(harness, "_display_ok", return_value=False):
            try:
                return harness.open_harness({"id": entry["id"], "harness": client},
                                            dict(entry))["line"]
            except harness.HeadlessError as exc:
                return exc.line

    def test_every_agent_opens_on_the_served_name(self):
        catalog.SETTINGS["sync_clients"] = True
        for folder in (".pi/agent", ".omp/agent", ".factory"):
            (support_env.FAKE_HOME / folder).mkdir(parents=True, exist_ok=True)
        entry = _entry("agent-open", "running", support_env.free_port(),
                       pid=os.getpid(), served_name="served-Qwen38",
                       model_name="launch-name", model="launch-name", ctx=8192)
        runtime.RUNNING[entry["id"]] = entry

        for client in ("pi", "omp"):
            line = self._open_line(client, entry)
            self.assertIn("--model b70/served-Qwen38", line, client)
            self.assertNotIn("launch-name", line, client)
        pi_models = json.loads(
            (support_env.FAKE_HOME / ".pi/agent/models.json").read_text())
        self.assertEqual(pi_models["providers"]["b70"]["models"][0]["id"],
                         "served-Qwen38")

        line = self._open_line("droid", entry)
        self.assertIn("--model custom:Desktop-B70-Loaded-Model-0", line)
        droid = json.loads(
            (support_env.FAKE_HOME / ".factory/settings.json").read_text())
        self.assertEqual(droid["customModels"][0]["model"], "served-Qwen38")

        line = self._open_line("codex", entry)
        self.assertIn("-m served-Qwen38", line)
        self.assertIn('model_provider="b70"', line)
        self.assertNotIn("launch-name", line)

        self._open_line("deepseek", entry)
        overlay = (paths.DATA / "dsh" / "gateway.yml").read_text()
        self.assertIn('id: "served-Qwen38"', overlay)
        self.assertIn("- id: agent-default-model", overlay)
        self.assertIn('model: "served-Qwen38"', overlay)
        self.assertIn("provider: b70-gateway", overlay)
        self.assertNotIn("launch-name", overlay)


if __name__ == "__main__":
    unittest.main()
