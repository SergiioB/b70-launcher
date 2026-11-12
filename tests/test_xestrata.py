"""XeStrata builder: settings-rooted paths, prerequisite refusals, portability
of the shipped recipes. Every call is a build only — nothing is executed.

Run:  cd b70-launcher && python3 -m unittest tests.test_xestrata -v
"""
import json
import shlex
import shutil
import unittest
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import builders, catalog, hardware, scan

MID, ENG = "qwen38-flashnext", "xestrata"
SHARD1 = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
VISION_EXE = "dist/usr/lib/x86_64-linux-gnu/xestrata/engine/strata-vision-cpu"
ONE_CARD = {"devices": [
    {"render": "/dev/dri/renderD128", "card": "card0", "b70": True, "accessible": True}],
    "blockers": [], "blockers_by_kind": {}, "default": {"gpus": [0]}}
TWO_CARDS = {"devices": [
    {"render": "/dev/dri/renderD128", "card": "card0", "b70": True, "accessible": True},
    {"render": "/dev/dri/renderD129", "card": "card1", "b70": True, "accessible": True}],
    "blockers": [], "blockers_by_kind": {}, "default": {"gpus": [0, 1]}}


def bash_cmd(plan):
    """The command the plan runs (['bash','-c',cmd]) with the shell prefix removed."""
    assert plan["tokens"][0:2] == ["bash", "-c"], plan["tokens"][:2]
    return plan["tokens"][2]


def cmd_tokens(plan):
    return shlex.split(bash_cmd(plan).split("&& exec ", 1)[1])


class XestrataCase(support_env.LauncherStateCase):
    """A complete XeStrata layout: checkout (venv, server, vision encoder),
    user config under the fake HOME, shard + mmproj in the scan pool."""

    def setUp(self):
        super().setUp()
        self.base = support_env.TMPROOT / f"xestrata-{self.id().rsplit('.', 1)[-1]}"
        self.root = self.base / "xestrata"
        for rel in (".venv/bin/python", "serve/server.py", VISION_EXE):
            support_env.write_file(self.root / rel)
        self.config = support_env.write_file(
            support_env.FAKE_HOME / ".config" / "xestrata" / "xestrata-iq3_xxs.json",
            b'{"args": ["--max-context", "8192"]}')
        support_env.write_file(
            support_env.FAKE_HOME / ".config" / "xestrata" / "xestrata-iq3_s.json",
            b'{"args": ["--max-context", "8192"]}')
        self.shard = support_env.write_file(self.base / "models" / "IQ3_XXS" / SHARD1)
        self.mmproj = support_env.write_file(self.base / "models" / "vision" / MMPROJ)
        self.addCleanup(shutil.rmtree, self.base, True)
        catalog.SETTINGS["xestrata_root"] = str(self.root)
        scan.SCAN["items"] = {"gguf": {MMPROJ.lower(): str(self.mmproj),
                                       SHARD1.lower(): str(self.shard)},
                              "snapshots": {}}
        patcher = mock.patch.object(hardware, "hardware_preflight", return_value=TWO_CARDS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **kw):
        cfg = {"model_id": MID, "engine": ENG, "port": 18199, "dry_run": True}
        cfg.update(kw)
        return builders.build(cfg)

    def build_iq3s(self, **kw):
        cfg = {"model_id": "qwen38-flashnext-iq3s", "engine": ENG, "port": 18199,
               "dry_run": True}
        cfg.update(kw)
        return builders.build(cfg)

    def recipe(self):
        return catalog.find_model(MID)["recipes"][ENG]


class TestXestrataCommand(XestrataCase):

    def test_command_resolves_paths_from_the_checkout(self):
        b = self.build()
        self.assertIsNone(b.get("error"), b.get("error"))
        toks = cmd_tokens(b)
        self.assertEqual(toks[0], str(self.root / ".venv/bin/python"))
        self.assertEqual(toks[toks.index("--server") + 1], str(self.root / "serve/server.py"))
        self.assertEqual(toks[toks.index("--config") + 1], str(self.config))
        self.assertEqual(toks[toks.index("--gpu") + 1], "0,1")
        # the bash -c wrapper still sets up oneAPI first
        self.assertTrue(bash_cmd(b).startswith("source /opt/intel/oneapi/setvars.sh"))

    def test_recipe_relative_paths_reject_escape(self):
        with mock.patch.dict(self.recipe(), {"py_exe": "../outside/python"}):
            b = self.build()
        self.assertIn("error", b)
        self.assertIn("py_exe", b["error"])
        with mock.patch.dict(self.recipe(), {"server_script": "/abs/server.py"}):
            b = self.build()
        self.assertIn("error", b)
        self.assertIn("server_script", b["error"])

    def test_empty_recipe_field_falls_back_to_the_default(self):
        with mock.patch.dict(self.recipe(), {"py_exe": ""}):
            b = self.build()
        self.assertIsNone(b.get("error"), b.get("error"))
        self.assertEqual(cmd_tokens(b)[0], str(self.root / ".venv/bin/python"))

    def test_vision_resolves_mmproj_from_scan_and_exe_from_root(self):
        # the shipped iq3s recipe carries a relative vision_exe + bare mmproj
        b = self.build_iq3s(vision=True)
        self.assertIsNone(b.get("error"), b.get("error"))
        toks = cmd_tokens(b)
        self.assertEqual(toks[toks.index("--vision-exe") + 1], str(self.root / VISION_EXE))
        self.assertEqual(toks[toks.index("--mmproj") + 1], str(self.mmproj))

    def test_vision_mmproj_absolute_path_bypasses_scan(self):
        r = catalog.find_model("qwen38-flashnext-iq3s")["recipes"][ENG]
        with mock.patch.dict(r, {"mmproj": str(self.mmproj)}):
            scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}
            b = self.build_iq3s(vision=True)
        self.assertIsNone(b.get("error"), b.get("error"))
        self.assertEqual(cmd_tokens(b)[cmd_tokens(b).index("--mmproj") + 1], str(self.mmproj))

    def test_vision_mmproj_missing_from_scan_is_a_missing_piece(self):
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}
        b = self.build_iq3s(vision=True)
        self.assertIsNone(b.get("error"))
        self.assertTrue(any("vision projector" in w for w in b["warnings"]), b["warnings"])
        self.assertTrue(any("preview only" in w for w in b["warnings"]))


class TestXestrataPrerequisites(XestrataCase):

    def test_root_unset_names_the_setting(self):
        del catalog.SETTINGS["xestrata_root"]
        b = self.build()
        self.assertIsNone(b.get("error"))
        self.assertTrue(any('set "xestrata_root" in settings-override.json' in w
                            for w in b["warnings"]), b["warnings"])
        real = self.build(dry_run=False)
        self.assertIn("error", real)
        self.assertIn("xestrata_root", real["error"])

    def test_root_must_be_absolute(self):
        catalog.SETTINGS["xestrata_root"] = "relative/xestrata"
        b = self.build()
        self.assertIn("error", b)
        self.assertIn("absolute", b["error"])

    def test_missing_pieces_are_named(self):
        (self.root / ".venv" / "bin" / "python").unlink()
        (self.root / "serve" / "server.py").unlink()
        self.config.unlink()
        b = self.build()
        self.assertIsNone(b.get("error"))
        joined = "\n".join(b["warnings"])
        self.assertIn(str(self.root / ".venv/bin/python"), joined)
        self.assertIn(str(self.root / "serve/server.py"), joined)
        self.assertIn(str(self.config), joined)
        real = self.build(dry_run=False)
        self.assertIn("error", real)
        self.assertIn("XeStrata recipe needs", real["error"])

    def test_missing_recipe_config_field_is_named(self):
        with mock.patch.dict(self.recipe(), {"config_file": ""}):
            b = self.build()
        self.assertIsNone(b.get("error"))
        self.assertTrue(any("config_file" in w for w in b["warnings"]), b["warnings"])

    def test_windows(self):
        from b70launcher.builders.xestrata import build_xestrata
        from b70launcher.builders.common import Bld, BuildError
        model = {"name": "m", "recipes": {ENG: self.recipe()}}
        with mock.patch("b70launcher.builders.xestrata.IS_WIN", True):
            with self.assertRaises(BuildError):
                build_xestrata(Bld({}, model, ENG, self.recipe(), None, []))


class TestXestrataShippedRecipe(XestrataCase):
    """Portability invariants of the shipped catalog entries."""

    def _walk_strings(self, obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from self._walk_strings(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from self._walk_strings(v)

    def test_shipped_xestrata_recipes_carry_no_absolute_engine_paths(self):
        for mid in (MID, "qwen38-flashnext-iq3s"):
            r = catalog.find_model(mid)["recipes"]["xestrata"]
            for field in ("py_exe", "server_script", "vision_exe", "config_file",
                          "mmproj", "model_path"):
                self.assertNotEqual(str(r.get(field, "x")).startswith("/"), True,
                                    f"{mid}.{field} is absolute: {r.get(field)!r}")
        self.assertEqual(self.recipe()["model_path"], "")
        self.assertEqual(catalog.find_model("qwen38-flashnext-iq3s")
                         ["recipes"]["xestrata"]["model_path"], "")

    def test_no_string_in_any_recipe_starts_with_a_machine_path(self):
        # the whole catalog must stay tarball-portable (CONTRIBUTING.md)
        for s in self._walk_strings(json.loads(
                (Path(__file__).resolve().parent.parent / "recipes.json").read_text())):
            self.assertFalse(s.startswith("/data/"), s[:60])
            self.assertFalse(s.startswith("/home/"), s[:60])

    def test_portability_bumped_recipe_vers(self):
        # launch behavior changed with the settings-rooted paths: the update
        # channel needs a newer recipe_ver than the catalog baseline
        self.assertTrue(catalog._ver_key(self.recipe()["recipe_ver"])
                        > catalog._ver_key("2026-10-18"))
        iq3s = catalog.find_model("qwen38-flashnext-iq3s")["recipes"]["xestrata"]
        self.assertTrue(catalog._ver_key(iq3s["recipe_ver"])
                        > catalog._ver_key("2026-10-19"))

    def test_override_key_is_accepted_from_settings_override(self):
        self.assertIn("xestrata_root", catalog.OVERRIDE_KEYS)

    def test_remote_overlay_cannot_add_engine_paths(self):
        for field in ("py_exe", "server_script", "config_file", "vision_exe",
                      "mmproj", "model_path"):
            self.assertNotIn(field, catalog.REMOTE_RECIPE_FIELDS, field)


if __name__ == "__main__":
    unittest.main()
