"""Unit tests for the vllm-native builder and Inkling 2 Mini catalog recipe."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import builders, catalog, hardware, scan


def _inkling_fixture(td):
    """Fake vllm_venv + on-disk Inkling artifact for build() calls."""
    venv = Path(td) / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "vllm").write_text("#!/bin/sh\n")
    model_dir = Path(td) / "Inkling-2-Mini-native-ed40e7da"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"max_position_embeddings": 1048576}')
    catalog.SETTINGS["vllm_venv"] = str(venv)
    catalog._apply_recipe_overrides({"inkling2-mini": {"vllm": {"model_path": str(model_dir)}}})


class TestVllmNativeAndInklingRecipe(support_env.LauncherStateCase):
    def test_inkling2_mini_present_in_catalog_and_approved_download(self):
        from b70launcher import downloads
        model = catalog.find_model("inkling2-mini")
        self.assertIsNotNone(model)
        self.assertEqual(model["recommended_engine"], "vllm")
        recipe = model["recipes"]["vllm"]
        self.assertEqual(recipe["kind"], "vllm-native")
        self.assertIs(recipe["prefix_caching"], False)
        self.assertIn("thinkingmachines/Inkling-2-Mini", downloads.HF_APPROVED_REPOS)

    def test_recipe_needs_reports_missing_vllm_venv(self):
        model = catalog.find_model("inkling2-mini")
        catalog.SETTINGS["vllm_venv"] = ""
        needs = scan.recipe_needs(model, "vllm", det={"path": "/tmp/fake"})
        self.assertTrue(any(n["setting"] == "vllm_venv" for n in needs))

    def test_dry_run_warns_and_real_launch_refuses_when_vllm_venv_missing(self):
        catalog.SETTINGS["vllm_venv"] = ""
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "Inkling-2-Mini-native-ed40e7da"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"max_position_embeddings": 1048576}')
            catalog._apply_recipe_overrides({"inkling2-mini": {"vllm": {"model_path": str(model_dir)}}})
            dry = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1], "dry_run": True})
            self.assertNotIn("error", dry)
            self.assertTrue(any("vllm_venv" in w for w in dry["warnings"]))
            real = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1], "dry_run": False})
            self.assertIn("error", real)
            self.assertIn("vllm_venv", real["error"])

    def test_swa_mbt_and_dspark_profiles_render_correctly(self):
        with tempfile.TemporaryDirectory() as td:
            venv = Path(td) / "venv"
            (venv / "bin").mkdir(parents=True)
            vllm_bin = venv / "bin" / "vllm"
            vllm_bin.write_text("#!/bin/sh\n")
            model_dir = Path(td) / "Inkling-2-Mini-native-ed40e7da"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"max_position_embeddings": 1048576}')
            catalog.SETTINGS["vllm_venv"] = str(venv)
            catalog._apply_recipe_overrides({"inkling2-mini": {"vllm": {"model_path": str(model_dir)}}})

            # 1. Default DSpark k=7 @ 106K -> mbt=2048, gmu=0.93, no-enable-prefix-caching
            b1 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1], "port": 8123,
                                 "extra_env": "INK2_CUTLASS_MOE=1"})
            self.assertNotIn("error", b1)
            self.assertTrue(b1["native"])
            self.assertEqual(b1["env"]["ZE_AFFINITY_MASK"], "1")
            self.assertEqual(b1["env"]["INK2_CUTLASS_MOE"], "1")
            cmd1 = b1["tokens"][2]
            self.assertIn("--max-model-len 106496", cmd1)
            self.assertIn("--max-num-batched-tokens 2048", cmd1)
            self.assertIn("--gpu-memory-utilization 0.93", cmd1)
            self.assertIn("--no-enable-prefix-caching", cmd1)
            self.assertIn('"method": "dspark"', cmd1)
            self.assertIn("VLLM_RPC_BASE_PATH", cmd1)

            # 2. Long-context DSpark @ 163,840 -> auto mbt=1024, gmu=0.95
            b2 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1], "ctx": 163840})
            self.assertNotIn("error", b2)
            cmd2 = b2["tokens"][2]
            self.assertIn("--max-model-len 163840", cmd2)
            self.assertIn("--max-num-batched-tokens 1024", cmd2)
            self.assertIn("--gpu-memory-utilization 0.95", cmd2)

            # 3. Max-context no-spec (--no-mtp) @ 262,144 (256K) -> auto mbt=1024, gmu=0.95, no dspark
            b3 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "ctx": 262144, "mtp": False})
            self.assertNotIn("error", b3)
            cmd3 = b3["tokens"][2]
            self.assertIn("--max-model-len 262144", cmd3)
            self.assertIn("--max-num-batched-tokens 1024", cmd3)
            self.assertIn("--gpu-memory-utilization 0.95", cmd3)
            self.assertNotIn("dspark", cmd3)

    def test_fp8_kv_doubles_swa_envelopes(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            # DSpark @ 200K fits on fp8 KV at mbt=2048 (BF16 KV would refuse >163K)
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "kv": "fp8", "ctx": 200000})
            self.assertNotIn("error", b)
            cmd = b["tokens"][2]
            self.assertIn("--kv-cache-dtype fp8", cmd)
            self.assertIn("--max-model-len 200000", cmd)
            self.assertIn("--max-num-batched-tokens 2048", cmd)
            self.assertIn("--gpu-memory-utilization 0.93", cmd)
            # no-spec fp8 @ 400K -> mbt=1024, gmu=0.95 (BF16 ceiling was 262K)
            b2 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "kv": "fp8", "ctx": 400000, "mtp": False})
            self.assertNotIn("error", b2)
            cmd2 = b2["tokens"][2]
            self.assertIn("--max-model-len 400000", cmd2)
            self.assertIn("--max-num-batched-tokens 1024", cmd2)
            self.assertIn("--gpu-memory-utilization 0.95", cmd2)

    def test_vision_flag_enables_mm_and_shrinks_default_ctx(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            # default stays text-only: --language-model-only pinned, 106K ctx
            b0 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            cmd0 = b0["tokens"][2]
            self.assertIn("--language-model-only", cmd0)
            self.assertNotIn("--limit-mm-per-prompt", cmd0)
            # vision:true drops the pin, caps image inputs, BF16 ctx -> 81,920
            b1 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "vision": True})
            self.assertNotIn("error", b1)
            cmd1 = b1["tokens"][2]
            self.assertNotIn("--language-model-only", cmd1)
            self.assertIn("--limit-mm-per-prompt", cmd1)
            self.assertIn("'{\"image\":4}'", cmd1)
            self.assertIn("--max-model-len 81920", cmd1)
            self.assertTrue(any("vision" in w for w in b1["warnings"]))
            # vision + fp8 KV -> 163,840 default
            b2 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "vision": True, "kv": "fp8"})
            self.assertIn("--max-model-len 163840", b2["tokens"][2])
            # an explicit ctx wins over the vision default
            b3 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "vision": True, "ctx": 32768})
            self.assertIn("--max-model-len 32768", b3["tokens"][2])

    def test_display_card_caps_gmu_and_steps_ctx(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            fake_preflight = {"devices": [
                {"render": "/dev/dri/renderD128", "card": "card0", "b70": True,
                 "accessible": True, "vram_total_mib": 32656.0,
                 "is_display": False, "desktop_vram_mib": 0.0},
                {"render": "/dev/dri/renderD129", "card": "card1", "b70": True,
                 "accessible": True, "vram_total_mib": 32656.0,
                 "is_display": True, "desktop_vram_mib": 4000.0}]}
            with mock.patch.object(hardware, "hardware_preflight",
                                   return_value=fake_preflight):
                b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            self.assertNotIn("error", b)
            cmd = b["tokens"][2]
            # cap = 1 - (4000 + 1024) / 32656 = 0.846 -> 0.85
            self.assertIn("--gpu-memory-utilization 0.85", cmd)
            # recipe ctx 106,496 > ctx_safe 90,112 -> stepped down
            self.assertIn("--max-model-len 90112", cmd)
            warn = next(w for w in b["warnings"] if "drives the desktop" in w)
            self.assertIn("4000 MiB", warn)
            self.assertIn("0.85", warn)
            self.assertIn("90,112", warn)
            # an explicit ctx is preserved, only the gmu cap + warning apply
            with mock.patch.object(hardware, "hardware_preflight",
                                   return_value=fake_preflight):
                b2 = builders.build({"model_id": "inkling2-mini", "engine": "vllm",
                                     "gpus": [1], "ctx": 200000})
            cmd2 = b2["tokens"][2]
            self.assertIn("--max-model-len 200000", cmd2)
            self.assertIn("--gpu-memory-utilization 0.85", cmd2)

    def test_remote_overlay_cannot_overwrite_vllm_venv_or_env(self):
        before = dict(catalog.find_model("inkling2-mini")["recipes"]["vllm"])
        doc = {"catalog_ver": "9999-01-01", "models": [{"id": "inkling2-mini", "recipes": {"vllm": {
            "vllm_venv": "/tmp/evil", "env": {"INK2_EVIL": "1"}, "fixed_flags": ["--evil"], "ctx": 65536
        }}}]}
        catalog.apply_recipe_doc(doc)
        after = catalog.find_model("inkling2-mini")["recipes"]["vllm"]
        self.assertEqual(after["ctx"], 65536)
        self.assertNotIn("vllm_venv", after)
        self.assertEqual(after.get("env"), before.get("env"))
        self.assertEqual(after.get("fixed_flags"), before.get("fixed_flags"))


if __name__ == "__main__":
    unittest.main()
