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


def _inkling_plugin(td, hybrid=True, esimd=True):
    """Fake vllm_inkling2_xpu install inside the fixture venv."""
    pkg = (Path(td) / "venv" / "lib" / "python3.14" / "site-packages"
           / "vllm_inkling2_xpu")
    (pkg / "ops").mkdir(parents=True, exist_ok=True)
    (pkg / "attention.py").write_text(
        'import os\n_H = os.environ.get("INK2_HYBRID_PREFIX_SDPA", "0") == "1"\n'
        if hybrid else "import os\n")
    if esimd:
        (pkg / "ops" / "_inkling_esimd_attn_xe2.cpython-314-x86_64-linux-gnu.so"
         ).write_bytes(b"\x7fELF")
    return pkg


def _inkling_ring_patch(td, worker=True, hook=True):
    """Fake the ring-prefix patch inside the fixture venv's site-packages:
    vllm_inkling2_xpu/ring_prefix_worker.py and/or the INK2_RING_PREFIX_CACHE
    hook in vllm/v1/core/kv_cache_manager.py (same site-packages dir)."""
    sp = Path(td) / "venv" / "lib" / "python3.14" / "site-packages"
    if worker:
        pkg = sp / "vllm_inkling2_xpu"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "ring_prefix_worker.py").write_text(
            "class RingPrefixWorker:\n    pass\n")
    if hook:
        kv_mgr = sp / "vllm" / "v1" / "core" / "kv_cache_manager.py"
        kv_mgr.parent.mkdir(parents=True, exist_ok=True)
        kv_mgr.write_text(
            'import os\nRING = os.environ.get("INK2_RING_PREFIX_CACHE", "0") == "1"\n')
    return sp


ACCEL_ENV = ("INK2_HYBRID_PREFIX_SDPA", "INK2_ESIMD_DECODE_ATTN",
             "INK2_ESIMD_SWA_SPLITS", "INK2_ESIMD_FULL_SPLITS")


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

    def test_inkling_accel_env_arms_only_when_plugin_supports_it(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            # stock venv (no vllm_inkling2_xpu): nothing armed, no warning
            b0 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            self.assertNotIn("error", b0)
            for k in ACCEL_ENV:
                self.assertNotIn(k, b0["env"])
            self.assertFalse(any("INK2_ESIMD" in w or "INK2_HYBRID" in w for w in b0["warnings"]))
            # plugin without hybrid patch or esimd .so: all flags stay off, warned
            _inkling_plugin(td, hybrid=False, esimd=False)
            b1 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            for k in ACCEL_ENV:
                self.assertNotIn(k, b1["env"])
            self.assertTrue(any("INK2_ESIMD_DECODE_ATTN" in w and "INK2_HYBRID_PREFIX_SDPA" in w
                                for w in b1["warnings"]))
            # hybrid patch only: sdpa armed, esimd still off + warned
            pkg = _inkling_plugin(td, hybrid=True, esimd=False)
            b2 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            self.assertEqual(b2["env"]["INK2_HYBRID_PREFIX_SDPA"], "1")
            self.assertNotIn("INK2_ESIMD_DECODE_ATTN", b2["env"])
            self.assertTrue(any("INK2_ESIMD_DECODE_ATTN" in w for w in b2["warnings"]))
            # esimd .so installed too: every accelerator armed
            (pkg / "ops" / "_inkling_esimd_attn_xe2.cpython-314-x86_64-linux-gnu.so"
             ).write_bytes(b"\x7fELF")
            b3 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            self.assertEqual(b3["env"]["INK2_HYBRID_PREFIX_SDPA"], "1")
            self.assertEqual(b3["env"]["INK2_ESIMD_DECODE_ATTN"], "1")
            self.assertEqual(b3["env"]["INK2_ESIMD_SWA_SPLITS"], "16")
            self.assertEqual(b3["env"]["INK2_ESIMD_FULL_SPLITS"], "128")
            self.assertFalse(any("opt-in accel" in w for w in b3["warnings"]))

    def test_inkling_accel_env_overrides_beat_detection(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td)
            # caller extra_env can disable or retune an auto-armed flag;
            # disabling ESIMD via =0 does not leave auto-injected split overrides behind
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "extra_env": "INK2_ESIMD_DECODE_ATTN=0\nINK2_ESIMD_FULL_SPLITS=64"})
            self.assertNotIn("error", b)
            self.assertEqual(b["env"]["INK2_ESIMD_DECODE_ATTN"], "0")
            self.assertEqual(b["env"]["INK2_ESIMD_FULL_SPLITS"], "64")
            self.assertNotIn("INK2_ESIMD_SWA_SPLITS", b["env"])
            b0 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "extra_env": "INK2_ESIMD_DECODE_ATTN=0"})
            self.assertNotIn("INK2_ESIMD_SWA_SPLITS", b0["env"])
            self.assertNotIn("INK2_ESIMD_FULL_SPLITS", b0["env"])
            # recipe env (allow-listed INK2_ names) also beats auto-detection
            catalog._apply_recipe_overrides({"inkling2-mini": {"vllm": {"env": {
                "INK2_HYBRID_PREFIX_SDPA": "0", "INK2_ESIMD_SWA_SPLITS": "8"}}}})
            b2 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            self.assertNotIn("error", b2)
            self.assertEqual(b2["env"]["INK2_HYBRID_PREFIX_SDPA"], "0")
            self.assertEqual(b2["env"]["INK2_ESIMD_SWA_SPLITS"], "8")
            self.assertEqual(b2["env"]["INK2_ESIMD_DECODE_ATTN"], "1")
            # extra_env still wins over recipe env
            b3 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "extra_env": "INK2_ESIMD_SWA_SPLITS=32"})
            self.assertEqual(b3["env"]["INK2_ESIMD_SWA_SPLITS"], "32")

    def test_inkling_accel_skips_non_inkling_native_and_explicit_opt_out(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td, hybrid=False, esimd=False)
            # explicit =0 suppresses missing-accel warnings on a stock venv
            b_off = builders.build({
                "model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                "extra_env": "INK2_HYBRID_PREFIX_SDPA=0\nINK2_ESIMD_DECODE_ATTN=0",
            })
            self.assertNotIn("error", b_off)
            self.assertFalse(any("opt-in accel stays off" in w for w in b_off["warnings"]))
            # non-Inkling vllm-native recipe in a patched Inkling venv gets zero INK2_* env or warnings
            _inkling_plugin(td, hybrid=True, esimd=True)
            other = {
                "id": "custom-native-other", "name": "Custom Native Model",
                "recipes": {"vllm": {"kind": "vllm-native", "ctx": 32768, "kv": "auto"}},
            }
            orig_find = catalog.find_model
            catalog.find_model = lambda mid: other if mid == "custom-native-other" else orig_find(mid)
            try:
                b_other = builders.build({"model_id": "custom-native-other", "engine": "vllm", "gpus": [1], "dry_run": True})
            finally:
                catalog.find_model = orig_find
            self.assertNotIn("error", b_other)
            self.assertFalse(any(k.startswith("INK2_") for k in b_other["env"]))
            self.assertFalse(any("vllm-inkling2-xpu" in w for w in b_other["warnings"]))

    def test_inkling_accel_warns_on_editable_install(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            sp = Path(td) / "venv" / "lib" / "python3.14" / "site-packages"
            sp.mkdir(parents=True, exist_ok=True)
            (sp / "__editable__.vllm_inkling2_xpu-0.1.0.pth").write_text("/tmp/src\n")
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            self.assertNotIn("error", b)
            self.assertNotIn("INK2_ESIMD_DECODE_ATTN", b["env"])
            self.assertTrue(any("editable mode" in w for w in b["warnings"]))

    def test_inkling_accel_env_force_arm_on_stock_venv(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            # explicit env flags pass straight through even with no plugin to read
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "extra_env": "INK2_HYBRID_PREFIX_SDPA=1\nINK2_ESIMD_DECODE_ATTN=1"})
            self.assertNotIn("error", b)
            self.assertEqual(b["env"]["INK2_HYBRID_PREFIX_SDPA"], "1")
            self.assertEqual(b["env"]["INK2_ESIMD_DECODE_ATTN"], "1")
            # a recipe env name outside the allow-list still fails the build
            catalog._apply_recipe_overrides({"inkling2-mini": {"vllm": {"env": {"LD_PRELOAD": "/x.so"}}}})
            bad = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            self.assertIn("error", bad)
            self.assertIn("not an allowed", bad["error"])

    def test_inkling_ring_prefix_cache_off_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td)
            _inkling_ring_patch(td)
            # fully patched venv, cache left at the recipe default (off):
            # the flag is never armed and stock argv is unchanged
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1]})
            self.assertNotIn("error", b)
            self.assertNotIn("INK2_RING_PREFIX_CACHE", b["env"])
            self.assertIn("--no-enable-prefix-caching", b["tokens"][2])
            self.assertFalse(any("armed" in w or "53912" in w for w in b["warnings"]))

    def test_inkling_ring_prefix_cache_arms_on_patched_venv(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td)
            _inkling_ring_patch(td)
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "cache": "on"})
            self.assertNotIn("error", b)
            self.assertEqual(b["env"]["INK2_RING_PREFIX_CACHE"], "1")
            self.assertIn(" --enable-prefix-caching ", b["tokens"][2] + " ")
            self.assertNotIn("--no-enable-prefix-caching", b["tokens"][2])
            self.assertTrue(any("INK2_RING_PREFIX_CACHE=1" in w and "armed" in w
                                for w in b["warnings"]))
            self.assertFalse(any("53912" in w for w in b["warnings"]))

    def test_inkling_ring_prefix_cache_unpatched_venv_warns(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td)  # plugin present but no ring files / hook
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "cache": "on"})
            self.assertNotIn("error", b)
            self.assertNotIn("INK2_RING_PREFIX_CACHE", b["env"])
            self.assertIn(" --enable-prefix-caching ", b["tokens"][2] + " ")
            warn = next(w for w in b["warnings"] if "53912" in w)
            self.assertIn("ring_prefix_worker.py", warn)
            self.assertIn("kv_cache_manager.py", warn)
            # worker present but the kv_cache_manager hook missing: still off,
            # warning names only the missing piece
            sp = _inkling_ring_patch(td, worker=True, hook=False)
            b2 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "cache": "on"})
            self.assertNotIn("INK2_RING_PREFIX_CACHE", b2["env"])
            warn2 = next(w for w in b2["warnings"] if "53912" in w)
            self.assertNotIn("ring_prefix_worker.py", warn2)
            self.assertIn("kv_cache_manager.py", warn2)
            # hook present but worker missing: symmetric
            (sp / "vllm_inkling2_xpu" / "ring_prefix_worker.py").unlink()
            _inkling_ring_patch(td, worker=False, hook=True)
            b3 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "cache": "on"})
            self.assertNotIn("INK2_RING_PREFIX_CACHE", b3["env"])
            warn3 = next(w for w in b3["warnings"] if "53912" in w)
            self.assertIn("ring_prefix_worker.py", warn3)

    def test_inkling_ring_prefix_cache_split_site_packages_not_armed(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td)
            # worker in lib/python3.14/site-packages, hook in lib64/python3.14/site-packages
            _inkling_ring_patch(td, worker=True, hook=False)
            other_sp = Path(td) / "venv" / "lib64" / "python3.14" / "site-packages"
            kv_mgr = other_sp / "vllm" / "v1" / "core" / "kv_cache_manager.py"
            kv_mgr.parent.mkdir(parents=True, exist_ok=True)
            kv_mgr.write_text('import os\n_R = os.environ.get("INK2_RING_PREFIX_CACHE") == "1"\n')
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "cache": "on"})
            self.assertNotIn("error", b)
            self.assertNotIn("INK2_RING_PREFIX_CACHE", b["env"])
            warn = next(w for w in b["warnings"] if "53912" in w)
            self.assertIn("split across different site-packages", warn)
            self.assertNotIn("lacks  —", warn)

    def test_inkling_ring_prefix_cache_single_slot_only(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td)
            _inkling_ring_patch(td)
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "cache": "on", "slots": 2})
            self.assertNotIn("error", b)
            self.assertNotIn("INK2_RING_PREFIX_CACHE", b["env"])
            self.assertIn("--max-num-seqs 2", b["tokens"][2])
            warn = next(w for w in b["warnings"] if "53912" in w)
            self.assertIn("single-slot", warn)
            self.assertFalse(any("armed" in w for w in b["warnings"]))

    def test_inkling_ring_prefix_cache_explicit_opt_out_and_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td)
            # caller =0 on an unpatched venv: stays "0" and suppresses the
            # missing-patch warning outright
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "cache": "on", "extra_env": "INK2_RING_PREFIX_CACHE=0"})
            self.assertNotIn("error", b)
            self.assertEqual(b["env"]["INK2_RING_PREFIX_CACHE"], "0")
            self.assertFalse(any("53912" in w or "armed" in w for w in b["warnings"]))
            # recipe env =0 suppresses auto-arm + warning on a patched venv too
            _inkling_ring_patch(td)
            catalog._apply_recipe_overrides({"inkling2-mini": {"vllm": {"env": {
                "INK2_RING_PREFIX_CACHE": "0"}}}})
            b2 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "cache": "on"})
            self.assertEqual(b2["env"]["INK2_RING_PREFIX_CACHE"], "0")
            self.assertFalse(any("53912" in w or "armed" in w for w in b2["warnings"]))
            # caller extra_env wins over recipe env — explicit =1 force-arms
            b3 = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                 "cache": "on", "extra_env": "INK2_RING_PREFIX_CACHE=1"})
            self.assertEqual(b3["env"]["INK2_RING_PREFIX_CACHE"], "1")

    def test_inkling_ring_prefix_cache_no_leakage_to_other_native(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            _inkling_plugin(td)
            _inkling_ring_patch(td)
            other = {
                "id": "custom-native-other", "name": "Custom Native Model",
                "recipes": {"vllm": {"kind": "vllm-native", "ctx": 32768, "kv": "auto"}},
            }
            orig_find = catalog.find_model
            catalog.find_model = lambda mid: other if mid == "custom-native-other" else orig_find(mid)
            try:
                b_other = builders.build({"model_id": "custom-native-other", "engine": "vllm",
                                          "gpus": [1], "cache": "on", "dry_run": True})
            finally:
                catalog.find_model = orig_find
            self.assertNotIn("error", b_other)
            self.assertFalse(any(k.startswith("INK2_") for k in b_other["env"]))
            self.assertFalse(any("53912" in w or "vllm-inkling2-xpu" in w
                                 for w in b_other["warnings"]))

    def test_inkling_ring_prefix_cache_editable_venv_not_armed(self):
        with tempfile.TemporaryDirectory() as td:
            _inkling_fixture(td)
            sp = Path(td) / "venv" / "lib" / "python3.14" / "site-packages"
            sp.mkdir(parents=True, exist_ok=True)
            (sp / "__editable__.vllm_inkling2_xpu-0.1.0.pth").write_text("/tmp/src\n")
            b = builders.build({"model_id": "inkling2-mini", "engine": "vllm", "gpus": [1],
                                "cache": "on"})
            self.assertNotIn("error", b)
            self.assertNotIn("INK2_RING_PREFIX_CACHE", b["env"])
            warn = next(w for w in b["warnings"] if "editable mode" in w)
            self.assertIn("INK2_RING_PREFIX_CACHE", warn)

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
