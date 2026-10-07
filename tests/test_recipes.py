"""Tests for the recipe overlay machinery: settings-override merge,
user recipe_overrides precedence over the remote catalog, remote document
merge semantics, version comparison and per-recipe notices.

Run:  cd b70-launcher && python3 -m unittest discover -s tests -v
"""
import json
import unittest
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

launcher = support_env.launcher
TMP = support_env.TMPROOT
LauncherStateCase = support_env.LauncherStateCase


def recipe_of(mid, eng):
    return launcher.find_model(mid)["recipes"][eng]


class TestMergeOverride(LauncherStateCase):
    """User edits live in DATA/settings-override.json — whitelisted keys only."""

    def test_whitelisted_keys_applied(self):
        launcher.OVERRIDE_PATH.write_text(json.dumps({
            "scan_dirs": [str(TMP / "ovr-scan")],
            "models_dir": str(TMP / "ovr-models"),
            "llama_bin": "/opt/custom/llama-server",
            "not_a_setting": "must-not-land",
            "image": "must-not-land-either"}))
        launcher._merge_override()
        self.assertEqual(launcher.SETTINGS["scan_dirs"], [str(TMP / "ovr-scan")])
        self.assertEqual(launcher.SETTINGS["models_dir"], str(TMP / "ovr-models"))
        self.assertEqual(launcher.SETTINGS["llama_bin"], "/opt/custom/llama-server")
        self.assertNotIn("not_a_setting", launcher.SETTINGS)
        self.assertNotIn("image", launcher.SETTINGS)

    def test_missing_and_corrupt_override_are_noop(self):
        launcher._merge_override()  # file absent: must not raise
        launcher.OVERRIDE_PATH.write_text("{corrupt")
        launcher._merge_override()  # bad JSON: must not raise
        self.assertEqual(launcher.SETTINGS["scan_dirs"],
                         ["~/models", "~/Downloads"])  # shipped defaults intact

    def test_recipe_overrides_land_in_recipes(self):
        launcher.OVERRIDE_PATH.write_text(json.dumps({
            "recipe_overrides": {"qwen36-35b": {"llamacpp": {"power": 111,
                                                            "gguf": "/opt/x.gguf"}},
                                 "qwen36-35b-bad": {"llamacpp": {"power": 1}},
                                 "qwen36-35b-str": "junk"}}))
        launcher._merge_override()
        r = recipe_of("qwen36-35b", "llamacpp")
        self.assertEqual(r["power"], 111)
        self.assertEqual(r["gguf"], "/opt/x.gguf")
        # stored for re-application; never lands on RECIPES (no such model id)
        self.assertEqual(
            launcher.USER_RECIPE_OVERRIDES["qwen36-35b"]["llamacpp"]["power"], 111)
        self.assertIsNone(launcher.find_model("qwen36-35b-bad"))
        # a non-dict override entry is ignored entirely
        self.assertNotIn("qwen36-35b-str", launcher.USER_RECIPE_OVERRIDES)

    def test_save_override_roundtrip(self):
        launcher.SETTINGS["scan_dirs"] = [str(TMP / "rt-scan")]
        launcher.save_override(("scan_dirs",))
        launcher.SETTINGS["scan_dirs"] = []
        launcher._merge_override()
        self.assertEqual(launcher.SETTINGS["scan_dirs"], [str(TMP / "rt-scan")])


class TestApplyRecipeDoc(LauncherStateCase):
    """Remote catalog overlay: new models append, known models merge."""

    def test_new_model_appended_and_stamped(self):
        doc = {"models": [{"id": "zz-new-model", "name": "ZZ",
                           "recipes": {"vllm": {"kind": "vllm", "image": "i"}}}]}
        applied, err = launcher.apply_recipe_doc(doc, "2027-01-01")
        self.assertIsNone(err)
        self.assertEqual(applied, 1)
        m = launcher.find_model("zz-new-model")
        self.assertIsNotNone(m)
        self.assertEqual(m["recipes"]["vllm"]["recipe_ver"], "2027-01-01")

    def test_existing_model_merges_allowlisted_fields(self):
        before_arch = launcher.find_model("qwen36-35b")["arch"]
        doc = {"models": [{"id": "qwen36-35b", "badge": "REMOTE-NEW",
                           "arch": "SHOULD-NOT-LAND",
                           "id": "qwen36-35b",
                           "recipes": {"llamacpp": {"kind": "gguf",
                                                    "recipe_ver": "2027-02-02",
                                                    "power": 222}}}]}
        applied, err = launcher.apply_recipe_doc(doc, "2027-02-02")
        self.assertIsNone(err)
        self.assertEqual(applied, 1)
        m = launcher.find_model("qwen36-35b")
        self.assertEqual(m["badge"], "REMOTE-NEW")          # in REMOTE_MODEL_FIELDS
        self.assertEqual(m["arch"], before_arch)            # not allowlisted
        self.assertEqual(m["id"], "qwen36-35b")             # id never overwritten
        self.assertEqual(m["recipes"]["llamacpp"]["power"], 222)
        self.assertIn("openvino", m["recipes"])             # other engines intact

    def test_bad_documents_rejected(self):
        self.assertEqual(launcher.apply_recipe_doc("not a dict"),
                         (0, "remote document has no models list"))
        self.assertEqual(launcher.apply_recipe_doc({"models": "x"}),
                         (0, "remote document has no models list"))
        # entries without an id are skipped, not counted
        applied, err = launcher.apply_recipe_doc({"models": [{"name": "x"}]})
        self.assertEqual(applied, 0)
        self.assertIsNone(err)

    def test_user_overrides_beat_remote_overlay(self):
        # the ordering guarantee: apply_recipe_doc re-applies user overrides last
        launcher._apply_recipe_overrides(
            {"qwen36-35b": {"llamacpp": {"power": 111}}})
        self.assertEqual(recipe_of("qwen36-35b", "llamacpp")["power"], 111)
        doc = {"models": [{"id": "qwen36-35b",
                           "recipes": {"llamacpp": {"kind": "gguf",
                                                    "power": 222}}}]}
        applied, err = launcher.apply_recipe_doc(doc, "2027-03-03")
        self.assertIsNone(err)
        self.assertEqual(applied, 1)
        # user override wins over the freshly overlaid remote recipe
        self.assertEqual(recipe_of("qwen36-35b", "llamacpp")["power"], 111)


class TestLoadRemoteOverlay(LauncherStateCase):
    def test_overlay_file_loaded_at_startup(self):
        doc = {"catalog_ver": "2027-04-04",
               "payload": {"models": [{"id": "zz-overlay", "name": "OV",
                                       "recipes": {"vllm": {"kind": "vllm"}}}]}}
        launcher.REMOTE_RECIPES_PATH.write_text(json.dumps(doc))
        launcher._load_remote_overlay()
        m = launcher.find_model("zz-overlay")
        self.assertIsNotNone(m)
        self.assertEqual(m["recipes"]["vllm"]["recipe_ver"], "2027-04-04")

    def test_missing_overlay_noop(self):
        launcher._load_remote_overlay()  # no file — must not raise


class TestLocalRecipeVer(LauncherStateCase):
    def test_recipe_ver_preferred(self):
        self.assertEqual(launcher.local_recipe_ver("qwen36-35b", "llamacpp"),
                         recipe_of("qwen36-35b", "llamacpp")["recipe_ver"])

    def test_falls_back_to_catalog_ver(self):
        launcher.find_model("qwen36-35b")["recipes"]["llamacpp"].pop("recipe_ver")
        self.assertEqual(launcher.local_recipe_ver("qwen36-35b", "llamacpp"),
                         launcher.RECIPES["catalog_ver"])

    def test_unknown_model_falls_back_to_catalog(self):
        # recipe_ver falls back to catalog_ver even for unknown model ids
        self.assertEqual(launcher.local_recipe_ver("ghost", "vllm"),
                         launcher.RECIPES["catalog_ver"])


class TestRecipeNotices(LauncherStateCase):
    def _remote(self, entries, url="https://x.test/r.json"):
        launcher.RECIPE_REMOTE.update(
            {"checked": True, "catalog_ver": "2027-01-01",
             "recipes_url": url, "entries": entries})

    def test_newer_remote_produces_notice(self):
        self._remote({"qwen36-35b:llamacpp": {"ver": "2027-02-01",
                                              "note": "better flags",
                                              "recommended": True}})
        n = launcher.recipe_notices()
        self.assertIn("qwen36-35b:llamacpp", n)
        e = n["qwen36-35b:llamacpp"]
        self.assertEqual(e["local_ver"],
                         launcher.local_recipe_ver("qwen36-35b", "llamacpp"))
        self.assertEqual(e["remote_ver"], "2027-02-01")
        self.assertFalse(e["is_new"])
        self.assertTrue(e["can_apply"])
        self.assertTrue(e["recommended"])
        self.assertEqual(e["model_name"], "Qwen3.6-35B-A3B")
        self.assertEqual(
            launcher.recipe_notice_for("qwen36-35b", "llamacpp")["note"],
            "better flags")

    def test_equal_or_older_remote_skipped(self):
        self._remote({"qwen36-35b:llamacpp": {"ver": "2026-10-05"},
                      "qwen36-35b:openvino": {"ver": "2020-01-01"}})
        self.assertEqual(launcher.recipe_notices(), {})

    def test_new_model_always_notices(self):
        # even an "old" version produces a notice for an absent model/engine
        self._remote({"ghost-model:vllm": {"ver": "2020-01-01",
                                            "model": "Ghost 7B"},
                      "qwen36-35b:exl3": {"ver": "2020-01-01"}})
        n = launcher.recipe_notices()
        self.assertTrue(n["ghost-model:vllm"]["is_new"])
        self.assertEqual(n["ghost-model:vllm"]["local_ver"], "")
        self.assertEqual(n["ghost-model:vllm"]["model_name"], "Ghost 7B")
        self.assertTrue(n["qwen36-35b:exl3"]["is_new"])  # new engine on known model

    def test_malformed_keys_skipped(self):
        self._remote({"nocolon": {"ver": "2027-01-01"},
                      "qwen36-35b:llamacpp": {"note": "no ver"}})
        self.assertEqual(launcher.recipe_notices(), {})

    def test_can_apply_requires_recipes_url(self):
        self._remote({"qwen36-35b:llamacpp": {"ver": "2027-02-01"}}, url="")
        self.assertFalse(
            launcher.recipe_notices()["qwen36-35b:llamacpp"]["can_apply"])


class TestRemoteFetchErrors(LauncherStateCase):
    """Only the URL policy is exercised — real fetches go over the network."""

    def test_no_url(self):
        launcher.RECIPE_REMOTE["recipes_url"] = ""
        doc, err = launcher.fetch_remote_recipes()
        self.assertIsNone(doc)
        self.assertIn("no remote recipes_url", err)

    def test_non_https_rejected(self):
        for bad in ("ftp://x/r.json", "http://evil.example/r.json",
                    "file:///etc/passwd"):
            launcher.RECIPE_REMOTE["recipes_url"] = bad
            doc, err = launcher.fetch_remote_recipes()
            self.assertIsNone(doc)
            self.assertIn("must be https", err, bad)

    def test_dead_localhost_url_fails_clean(self):
        launcher.RECIPE_REMOTE["recipes_url"] = \
            f"http://127.0.0.1:{support_env.free_port()}/r.json"
        doc, err = launcher.fetch_remote_recipes()
        self.assertIsNone(doc)
        self.assertIn("fetch failed", err)


class TestApplyRecipeUpdate(LauncherStateCase):
    def test_no_remote_url_errors(self):
        launcher.RECIPE_REMOTE["recipes_url"] = ""
        self.assertIn("error", launcher.apply_recipe_update())

    def test_remote_has_no_such_recipe(self):
        with mock.patch.object(launcher, "fetch_remote_recipes",
                               return_value=({"models": []}, None)):
            res = launcher.apply_recipe_update("qwen36-35b", "llamacpp")
        self.assertIn("no recipe for qwen36-35b:llamacpp", res["error"])

    def test_applies_and_persists(self):
        doc = {"catalog_ver": "2027-05-05",
               "models": [{"id": "qwen36-35b",
                           "recipes": {"llamacpp": {"kind": "gguf",
                                                    "power": 77}}}]}
        with mock.patch.object(launcher, "fetch_remote_recipes",
                               return_value=(doc, None)):
            res = launcher.apply_recipe_update("qwen36-35b", "llamacpp")
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["applied"], 1)
        self.assertEqual(res["catalog_ver"], "2027-05-05")
        self.assertEqual(recipe_of("qwen36-35b", "llamacpp")["power"], 77)
        # persisted overlay is reloadable
        stored = json.loads(launcher.REMOTE_RECIPES_PATH.read_text())
        self.assertEqual(stored["catalog_ver"], "2027-05-05")
        self.assertEqual(stored["payload"]["models"][0]["id"], "qwen36-35b")
        launcher.scan_blocking()  # join the rescan thread apply_recipe_update spawned

    def test_nothing_to_apply(self):
        doc = {"catalog_ver": "2027-05-05", "models": [
            {"id": "qwen36-35b"}]}  # no recipes -> nothing applied
        with mock.patch.object(launcher, "fetch_remote_recipes",
                               return_value=(doc, None)):
            res = launcher.apply_recipe_update()
        self.assertIn("nothing newer", res["error"])


if __name__ == "__main__":
    unittest.main()
