"""Tests for the Codex CLI launch-argv/env builder.

Everything here is deterministic and offline; the one integration test only runs
``codex debug prompt-input`` (a purely local command that parses the same config
overrides we pass at launch) and is skipped when the binary is absent.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10
    tomllib = None

from b70launcher import codex_client

BASE_URL = "http://127.0.0.1:7570/gw/codex/8765/v1"
CTX = 65536
MAX_OUT = 8192

# The override block that was verified against codex-cli 0.159.0.
EXPECTED_OVERRIDES = [
    "-c", 'model_providers.b70.name="B70 Local Engine"',
    "-c", 'model_providers.b70.base_url="http://127.0.0.1:7570/gw/codex/8765/v1"',
    "-c", 'model_providers.b70.wire_api="responses"',
    "-c", 'model_providers.b70.env_key="B70_CODEX_API_KEY"',
    "-c", 'model_providers.b70.requires_openai_auth=false',
    "-c", 'model_provider="b70"',
    "-c", "model_context_window=65536",
    "-c", "model_auto_compact_token_limit=57344",
    "-c", 'web_search="disabled"',
    "-c", 'model_reasoning_summary="none"',
    "-c", "check_for_update_on_startup=false",
]


class CodexArgvTests(unittest.TestCase):
    def test_exec_argv_exact(self):
        argv = codex_client.codex_exec_argv(BASE_URL, "active", CTX, MAX_OUT, prompt="say hi")
        self.assertEqual(argv[:3], ["codex", "exec", "--skip-git-repo-check"])
        self.assertEqual(argv[3:-3], EXPECTED_OVERRIDES)
        self.assertEqual(argv[-3:], ["-m", "active", "say hi"])

    def test_tui_argv_exact(self):
        argv = codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT)
        self.assertEqual(argv[0], "codex")
        # the TUI must not be given the exec subcommand or the exec-only flag
        self.assertNotIn("exec", argv)
        self.assertNotIn("--skip-git-repo-check", argv)
        self.assertEqual(argv[1:-2], EXPECTED_OVERRIDES)
        self.assertEqual(argv[-2:], ["-m", "active"])

    def test_tui_and_exec_share_override_block(self):
        tui = codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT)
        ex = codex_client.codex_exec_argv(BASE_URL, "active", CTX, MAX_OUT)
        self.assertEqual(tui[1:-2], ex[3:-2])

    def test_exec_prompt_optional(self):
        argv = codex_client.codex_exec_argv(BASE_URL, "active", CTX, MAX_OUT)
        self.assertEqual(argv[-2:], ["-m", "active"])

    def test_custom_binary_is_argv0(self):
        argv = codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT, codex_bin="/opt/codex")
        self.assertEqual(argv[0], "/opt/codex")
        self.assertEqual(argv[1:3], ["-c", 'model_providers.b70.name="B70 Local Engine"'])

    def test_every_override_is_key_equals_value(self):
        for token in codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT):
            if "=" in token and not token.startswith("-"):
                key, _, value = token.partition("=")
                self.assertTrue(key, token)
                self.assertTrue(value, token)  # never emit `-c key=` with no value
                self.assertNotIn(" ", key)  # keys are dotted paths, never quoted

    def test_web_search_and_openai_only_features_disabled(self):
        argv = codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT)
        self.assertIn('-c', argv)
        self.assertIn('web_search="disabled"', argv)
        self.assertIn('model_reasoning_summary="none"', argv)
        self.assertIn("check_for_update_on_startup=false", argv)
        # `tools.web_search = false` was verified to NOT remove the tool; the
        # mode key is what works, so make sure we never emit the dead one.
        self.assertFalse([t for t in argv if t.startswith("tools.web_search")])

    def test_context_window_keys_are_the_live_0159_names(self):
        argv = codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT)
        self.assertIn("model_context_window=65536", argv)
        self.assertIn("model_auto_compact_token_limit=57344", argv)
        # dropped in 0.159 and silently ignored: must never be emitted
        self.assertFalse([t for t in argv if t.startswith("model_max_output_tokens")])

    def test_compact_limit_leaves_max_out_headroom(self):
        argv = codex_client.codex_argv(BASE_URL, "active", 32768, 4096)
        self.assertIn("model_context_window=32768", argv)
        self.assertIn("model_auto_compact_token_limit=28672", argv)

    def test_compact_limit_clamped_when_max_out_exceeds_ctx(self):
        argv = codex_client.codex_argv(BASE_URL, "active", 4096, 8192)
        self.assertIn("model_context_window=4096", argv)
        self.assertIn("model_auto_compact_token_limit=1", argv)

    def test_toml_quoting_of_hostile_values(self):
        argv = codex_client.codex_argv(
            'http://127.0.0.1:1/v1?x="y"\\z', 'weird "model" \\ name\n', CTX, MAX_OUT
        )
        base = [t for t in argv if t.startswith("model_providers.b70.base_url")][0]
        self.assertEqual(
            base,
            'model_providers.b70.base_url="http://127.0.0.1:1/v1?x=\\"y\\"\\\\z"',
        )
        # -m takes the raw model string; the shell/exec layer passes it verbatim
        self.assertEqual(argv[argv.index("-m") + 1], 'weird "model" \\ name\n')

    @unittest.skipIf(tomllib is None, "tomllib unavailable")
    def test_override_values_roundtrip_through_toml(self):
        for model in ["active", 'a "quoted" model', "tab\tand\\slash"]:
            doc = "model = %s\n" % codex_client.toml_str(model)
            self.assertEqual(tomllib.loads(doc)["model"], model)

    def test_provider_id_is_used_consistently(self):
        argv = codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT)
        self.assertIn('model_provider="%s"' % codex_client.PROVIDER_ID, argv)
        for token in argv:
            if token.startswith("model_providers.") and token.endswith((".name=\"B70 Local Engine\"",)):
                self.assertTrue(token.startswith("model_providers.%s." % codex_client.PROVIDER_ID))


class CodexEnvTests(unittest.TestCase):
    def test_adds_placeholder_api_key(self):
        env = codex_client.codex_env({"PATH": "/bin"})
        self.assertEqual(env["PATH"], "/bin")
        self.assertTrue(env[codex_client.API_KEY_ENV])

    def test_keeps_a_real_api_key(self):
        env = codex_client.codex_env({codex_client.API_KEY_ENV: "hermes-agent-key"})
        self.assertEqual(env[codex_client.API_KEY_ENV], "hermes-agent-key")

    def test_loopback_bypasses_proxy_in_both_spellings(self):
        env = codex_client.codex_env({"NO_PROXY": "example.com"})
        for key in ("NO_PROXY", "no_proxy"):
            parts = env[key].split(",")
            self.assertIn("example.com", parts)
            self.assertIn("127.0.0.1", parts)
            self.assertIn("localhost", parts)

    def test_is_idempotent(self):
        once = codex_client.codex_env({})
        twice = codex_client.codex_env(once)
        self.assertEqual(once, twice)

    def test_does_not_mutate_caller_env(self):
        base = {"PATH": "/bin"}
        codex_client.codex_env(base)
        self.assertEqual(base, {"PATH": "/bin"})

    def test_empty_base_ok(self):
        env = codex_client.codex_env({})
        self.assertIn(codex_client.API_KEY_ENV, env)
        self.assertIn("NO_PROXY", env)


class CodexNoWriteTests(unittest.TestCase):
    def test_builders_never_touch_the_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = {str(p) for p in Path(tmp).rglob("*")}
            codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT)
            codex_client.codex_exec_argv(BASE_URL, "active", CTX, MAX_OUT, prompt="hi")
            codex_client.codex_env({})
            after = {str(p) for p in Path(tmp).rglob("*")}
            self.assertEqual(before, after)

    def test_no_config_file_under_a_fake_codex_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / ".codex"
            old = os.environ.get("CODEX_HOME")
            os.environ["CODEX_HOME"] = str(home)
            try:
                codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT)
                codex_client.codex_exec_argv(BASE_URL, "active", CTX, MAX_OUT)
            finally:
                if old is None:
                    os.environ.pop("CODEX_HOME", None)
                else:
                    os.environ["CODEX_HOME"] = old
            self.assertFalse(home.exists())


class CodexOverrideAcceptanceTests(unittest.TestCase):
    """Feed the generated overrides to Codex itself (local config parse only)."""

    def setUp(self):
        if shutil.which("codex") is None:
            self.skipTest("codex CLI not installed")
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_codex_accepts_generated_overrides(self):
        argv = codex_client.codex_argv(BASE_URL, "active", CTX, MAX_OUT)
        # `debug prompt-input` has no -m/positional prompt, so replace the model
        # flag with the equivalent -c override and drop the binary name.
        overrides = [t for t in argv[1:] if t != "exec"]
        overrides = [t for t in overrides if t != "--skip-git-repo-check"]
        model_flag = overrides.index("-m")
        overrides[model_flag : model_flag + 2] = ["-c", 'model="active"']
        env = dict(os.environ)
        env.update(codex_client.codex_env({}))
        env["HOME"] = self.tmp.name
        env["CODEX_HOME"] = self.tmp.name
        proc = subprocess.run(
            ["codex", "debug", "prompt-input"] + overrides,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIsInstance(json.loads(proc.stdout), list)


if __name__ == "__main__":
    unittest.main()
