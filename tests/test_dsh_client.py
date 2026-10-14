import tempfile
import unittest
from pathlib import Path

from b70launcher import dsh_client


class DshPrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "dsh"
        self.base_url = "http://127.0.0.1:7570/gw/dsh/8765/v1"
        self.argv, self.env = dsh_client.dsh_prepare(
            self.base_url, "active", 131072, self.state
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_only_inside_state_dir(self):
        before = {p for p in Path(self.tmp.name).rglob("*")}
        dsh_client.dsh_prepare(self.base_url, "active", 131072, self.state)
        after = {p for p in Path(self.tmp.name).rglob("*")}
        self.assertEqual(before, after)  # idempotent: no new files outside

    def test_overlay_content(self):
        text = (self.state / "gateway.yml").read_text()
        self.assertIn(f"baseURL: \"{self.base_url}\"", text)
        self.assertIn("api: openai-completions", text)
        self.assertIn(f"apiKeyEnv: {dsh_client.API_KEY_ENV}", text)
        self.assertIn('id: "active"', text)
        self.assertIn("contextWindow: 131072", text)
        self.assertIn("provider: b70-gateway", text)
        # patch layer shape: top-level YAML array of two entries
        entries = [ln for ln in text.splitlines() if ln.startswith("- id: ")]
        self.assertEqual(entries, ["- id: llm-pi-ai", "- id: agent-default-model"])

    def test_overlay_escapes_special_chars(self):
        argv, _ = dsh_client.dsh_prepare("http://h/x?a=1&b=2", "model: yes #x", 8, self.state)
        text = (self.state / "gateway.yml").read_text()
        self.assertIn('"http://h/x?a=1&b=2"', text)
        self.assertIn('"model: yes #x"', text)

    def test_ctx_variants_rendered(self):
        for ctx in (4096, 1000000):
            dsh_client.dsh_prepare(self.base_url, "m", ctx, self.state)
            self.assertIn(f"contextWindow: {ctx}", (self.state / "gateway.yml").read_text())

    def test_argv_shape(self):
        self.assertEqual(self.argv[:3], ["npx", "-y", dsh_client.DSH_PACKAGE])
        self.assertIn("web", self.argv)
        self.assertIn("--patch", self.argv)
        self.assertEqual(self.argv[self.argv.index("--patch") + 1], str(self.state / "gateway.yml"))
        self.assertTrue(Path(self.argv[self.argv.index("--patch") + 1]).is_file())
        self.assertEqual(self.argv[self.argv.index("--port") + 1], "0")
        self.assertNotIn("--no-open", self.argv)  # dsh has no terminal UI; it must open its web UI
        # pinned resolved version, not a floating tag
        self.assertNotIn("@latest", self.argv[2])
        self.assertTrue(dsh_client.DSH_VERSION[0].isdigit())

    def test_env(self):
        self.assertEqual(set(self.env), {"DSH_HOME", dsh_client.API_KEY_ENV})
        self.assertEqual(self.env["DSH_HOME"], str(self.state / "home"))
        self.assertTrue(Path(self.env["DSH_HOME"]).is_dir())
        self.assertTrue(self.env["DSH_HOME"].startswith(str(self.state)))
        self.assertTrue(self.env[dsh_client.API_KEY_ENV])

    def test_headless_argv(self):
        argv = dsh_client.dsh_headless_argv(self.state, "run the tests")
        self.assertEqual(argv[:3], ["npx", "-y", dsh_client.DSH_PACKAGE])
        self.assertIn("headless", argv)
        self.assertEqual(argv[-1], "run the tests")
        self.assertIn(str(self.state / "gateway.yml"), argv)

    def test_headless_requires_prepare(self):
        with self.assertRaises(FileNotFoundError):
            dsh_client.dsh_headless_argv(Path(self.tmp.name) / "empty", "x")


class DshInstalledTests(unittest.TestCase):
    def test_returns_bool(self):
        self.assertIsInstance(dsh_client.dsh_installed(), bool)


if __name__ == "__main__":
    unittest.main()
