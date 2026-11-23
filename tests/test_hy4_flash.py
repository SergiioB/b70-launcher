"""Strata Hy4-Flash builder (kind strata-hy4): catalog schema, command
rendering across the verified context rungs, display-card adaptation,
prerequisite refusals and remote-overlay safety. Every call is a build only —
nothing is executed, no GPU is touched.

Run:  cd b70-launcher && python3 -m unittest tests.test_hy4_flash -v
"""
import json
import shlex
import shutil
import unittest
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import builders, catalog, downloads, hardware, scan

MID, ENG = "hy4-flash", "strata"
NATIVE = "Hy4-Flash-SQ-64.strata"
ENGRAM = "Hy4-Flash-Engram-Q8.strata"
COOKBOOK_URL = ("https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook"
                "/blob/master/docs/hy4-flash/HY4-FLASH-STRATA-SYCL.md")
TWO_CARDS = {"devices": [
    {"render": "/dev/dri/renderD128", "card": "card0", "b70": True, "accessible": True,
     "vram_total_mib": 32656},
    {"render": "/dev/dri/renderD129", "card": "card1", "b70": True, "accessible": True,
     "vram_total_mib": 32656}],
    "blockers": [], "blockers_by_kind": {}, "default": {"gpus": [0, 1]}}


def config_of(plan):
    """The server config the container writes at start (printf '%s' <json>)."""
    words = shlex.split(plan["tokens"][-1])
    return json.loads(words[words.index("printf") + 2])


def arg_of(plan, flag):
    a = config_of(plan)["args"]
    return a[a.index(flag) + 1]


def env_of(plan):
    t = plan["tokens"]
    return dict(t[i + 1].split("=", 1) for i, x in enumerate(t) if x == "-e")


def mounts_of(plan):
    t = plan["tokens"]
    return {t[i + 1].rsplit(":", 2)[1]: t[i + 1].rsplit(":", 2)[0] for i, x in enumerate(t) if x == "-v"}


class Hy4Case(support_env.LauncherStateCase):
    """A complete Hy4 layout: patched Strata v0.2.20 checkout + both .strata packs."""

    def setUp(self):
        super().setUp()
        self.base = support_env.TMPROOT / f"hy4-{self.id().rsplit('.', 1)[-1]}"
        self.root = self.base / "Strata"
        for rel in ("engine/sycl/serve/server_intel.py",
                    "build-hy4-sycl-bmg/strata-hy4-sycl", ".venv-sycl/pyvenv.cfg"):
            support_env.write_file(self.root / rel)
        self.native = support_env.write_file(self.base / "models" / NATIVE)
        self.engram = support_env.write_file(self.base / "models" / ENGRAM)
        self.addCleanup(shutil.rmtree, self.base, True)
        catalog.SETTINGS["strata_hy4_root"] = str(self.root)
        catalog.SETTINGS["strata_root"] = ""
        scan.SCAN["items"] = {"gguf": {NATIVE.lower(): str(self.native)}, "snapshots": {}}
        patcher = mock.patch.object(hardware, "hardware_preflight", return_value=dict(TWO_CARDS))
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **kw):
        cfg = {"model_id": MID, "engine": ENG, "port": 18199, "dry_run": True}
        cfg.update(kw)
        return builders.build(cfg)

    def recipe(self):
        return catalog.find_model(MID)["recipes"][ENG]


class TestHy4Command(Hy4Case):

    def test_container_shape(self):
        b = self.build()
        self.assertIsNone(b.get("error"), b.get("error"))
        t = b["tokens"]
        self.assertEqual(t[:3], ["docker", "run", "--rm"])
        self.assertEqual(b["cname"], "b70-hy4-flash-strata-hy4-18199")
        self.assertIn("b70.launcher.rid=hy4-flash-strata-18199", t)
        self.assertIn("127.0.0.1:18199:18199", t)
        self.assertIn("SYS_PTRACE", t)  # Level Zero IPC between the two cards
        self.assertIn("memlock=-1", t)
        self.assertIn("--oom-score-adj", t)
        self.assertNotIn("--privileged", t)
        self.assertNotIn("--network", t)
        mapped = [t[i + 1] for i, x in enumerate(t) if x == "--device"]
        self.assertEqual(mapped, ["/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/card0", "/dev/dri/card1"])
        self.assertEqual(t[t.index("--entrypoint") + 1:t.index("--entrypoint") + 4],
                         ["bash", "strata-sycl-dev", "-lc"])
        self.assertEqual(b["served_name"], "Hy4-Flash SQ-64")
        self.assertEqual(b["effective_binary"], "strata-sycl-dev")
        self.assertIn("strata-sycl-dev", b["image_hint"])
        self.assertFalse(any("recipe needs" in w for w in b["warnings"]))

    def test_every_host_mount_is_read_only(self):
        b = self.build()
        vols = [b["tokens"][i + 1] for i, x in enumerate(b["tokens"]) if x == "-v"]
        self.assertTrue(all(v.endswith(":ro") for v in vols), vols)
        m = mounts_of(b)
        self.assertEqual(m["/strata"], str(self.root))
        self.assertEqual(m[f"/models/{NATIVE}"], str(self.native.resolve()))
        self.assertEqual(m[f"/models/{ENGRAM}"], str(self.engram.resolve()))

    def test_server_config_and_verified_args_default_ctx(self):
        b = self.build()
        c = config_of(b)
        self.assertEqual(c["exe"], "/strata/build-hy4-sycl-bmg/strata-hy4-sycl")
        self.assertEqual((c["arch"], c["backend"], c["cwd"]), ("hy4flash", "sycl", "/strata"))
        self.assertEqual(c["tokenizer"], "/tmp/b70-strata/tokenizer")
        self.assertEqual(c["model_name"], "Hy4-Flash SQ-64")
        self.assertEqual((c["gpu"], c["gpu_order"], c["parallel"]), ([0, 1], "as_given", 1))
        self.assertEqual(c["drafting"], "dfly2")
        a = c["args"]
        for flag, val in (("--native", f"/models/{NATIVE}"), ("--engram", f"/models/{ENGRAM}"),
                          ("--max-context", "131072"), ("--session-reuse", "1"),
                          ("--draft", "dfly2"), ("--threads", "8"), ("--gpus", "1,0"),
                          ("--vram-experts-gb", "15"), ("--peer-vram-experts-gb", "25"),
                          ("--ram-experts-gb", "36"), ("--kv-host-above", "131072")):
            self.assertEqual(a[a.index(flag) + 1], val, flag)
        self.assertNotIn("--kv", a)  # fp16 is the engine's only KV and its default
        script = b["tokens"][-1]
        self.assertIn("exec /strata/.venv-sycl/bin/python engine/sycl/serve/server_intel.py "
                      "--engine strata --config /tmp/b70-strata/config.json --host 0.0.0.0 "
                      "--port 18199", script)
        self.assertIn("tail -n +1 -F /tmp/b70-strata/engine.log &", script)

    def test_context_rungs(self):
        """The verified expert-budget ladder: 32K→17, 128K→15, 256K→14 + deeper KV host gate."""
        for ctx, vram, kvh in ((32768, "17", "131072"), (131072, "15", "131072"),
                               (262144, "14", "262144")):
            with self.subTest(ctx=ctx):
                b = self.build(ctx=ctx)
                self.assertEqual(arg_of(b, "--max-context"), str(ctx))
                self.assertEqual(arg_of(b, "--vram-experts-gb"), vram)
                self.assertEqual(arg_of(b, "--peer-vram-experts-gb"), "25")
                self.assertEqual(arg_of(b, "--kv-host-above"), kvh)

    def test_ctx_above_ceiling_clamps(self):
        b = self.build(ctx=300000)
        self.assertEqual(arg_of(b, "--max-context"), "262144")
        self.assertTrue(any("clamping" in w for w in b["warnings"]))

    def test_draft_toggle(self):
        a = config_of(self.build(mtp=False))["args"]
        self.assertEqual(a[a.index("--draft") + 1], "off")
        b = self.build(mtp=False)
        self.assertEqual(config_of(b)["drafting"], "off")
        self.assertTrue(any("disabled by --no-mtp" in w for w in b["warnings"]))
        saved = self.recipe()["draft"]
        self.recipe()["draft"] = "off"
        try:
            self.assertEqual(arg_of(self.build(), "--draft"), "off")
            self.assertEqual(arg_of(self.build(mtp=True), "--draft"), "dfly2")
        finally:
            self.recipe()["draft"] = saved

    def test_draft_rejects_unknown_values(self):
        saved = self.recipe()["draft"]
        self.recipe()["draft"] = "auto"  # the engine accepts it; the recipe contract does not
        try:
            self.assertIn("draft", self.build()["error"])
        finally:
            self.recipe()["draft"] = saved

    def test_cache_off_maps_to_session_reuse(self):
        self.assertEqual(arg_of(self.build(cache="off"), "--session-reuse"), "0")

    def test_engine_env(self):
        e = env_of(self.build())
        self.assertEqual(e["ZE_ENABLE_PCI_ID_DEVICE_ORDER"], "1")
        self.assertEqual(e["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
        self.assertEqual(e["SYCL_CACHE_PERSISTENT"], "0")
        self.assertEqual(e["ZES_ENABLE_SYSMAN"], "1")
        self.assertNotIn("ZE_AFFINITY_MASK", e)

    def test_recipe_env_and_caller_wins(self):
        self.recipe()["env"] = {"HY4S_VRAM_STATIC": "1"}
        e = env_of(self.build(extra_env="HY4S_VRAM_STATIC=0\nGLIBC_TUNABLES=x=1"))
        self.assertEqual(e["HY4S_VRAM_STATIC"], "0")
        self.assertEqual(e["GLIBC_TUNABLES"], "x=1")
        for env, text in (({"LD_PRELOAD": "/tmp/x.so"}, "LD_PRELOAD"),
                          ({"HY4S_X": "a\nb"}, "single-line"), ({"HY4S_X": {"a": 1}}, "single-line")):
            with self.subTest(env=env):
                self.recipe()["env"] = env
                self.assertIn(text, self.build()["error"])

    def test_extra_flags_append_last(self):
        a = config_of(self.build(extra="--verbose"))["args"]
        self.assertEqual(a[-1], "--verbose")

    def test_one_card_is_refused(self):
        self.assertIn("needs 2 GPUs", self.build(gpus=[0])["error"])

    def test_display_card_is_peer_and_budget_capped(self):
        """The desktop card never holds the primary budget: it is the --gpus peer
        and its expert GiB is capped at the shared gmu fraction."""
        cards = json.loads(json.dumps(TWO_CARDS))
        cards["devices"][0]["is_display"] = True
        cards["devices"][0]["desktop_vram_mib"] = 8192
        with mock.patch.object(hardware, "hardware_preflight", return_value=cards):
            b = self.build()
        # cap = 1 - (8192 + 1024) / 32656 ≈ 0.72 → peer floor = int(0.72 * 32656/1024) = 22
        self.assertEqual(arg_of(b, "--gpus"), "1,0")
        self.assertEqual(arg_of(b, "--peer-vram-experts-gb"), "22")
        self.assertEqual(arg_of(b, "--vram-experts-gb"), "17")  # ctx fell to the safe preset
        self.assertEqual(arg_of(b, "--max-context"), "32768")
        w = " ".join(b["warnings"])
        self.assertIn("display-card safe preset", b["ctx_source"])
        self.assertIn("peer GPU 0 drives the desktop", w)

    def test_display_manual_ctx_keeps_ctx_but_caps_budget(self):
        cards = json.loads(json.dumps(TWO_CARDS))
        cards["devices"][0]["is_display"] = True
        cards["devices"][0]["desktop_vram_mib"] = 8192
        with mock.patch.object(hardware, "hardware_preflight", return_value=cards):
            b = self.build(ctx=131072)
        self.assertEqual(arg_of(b, "--max-context"), "131072")
        self.assertEqual(arg_of(b, "--vram-experts-gb"), "15")
        self.assertEqual(arg_of(b, "--peer-vram-experts-gb"), "22")

    def test_no_private_host_paths(self):
        """The plan must not embed this machine's private layout — every host
        string is either a docker flag or a path the user set themselves."""
        b = self.build()
        text = " ".join(b["tokens"])
        for leak in ("/home/sergio", "/data/samsung980", "/data/nvme", "/data/sata"):
            self.assertNotIn(leak, text)
        # every mount source is under the test's own temp root (the user's
        # configured paths), never a hard-coded host path
        for src in mounts_of(b).values():
            self.assertTrue(str(src).startswith(str(support_env.TMPROOT)), src)


class TestHy4Refusals(Hy4Case):

    def assertRefused(self, text, **kw):
        b = self.build(dry_run=False, **kw)
        self.assertIn(text, b.get("error") or "")
        preview = self.build(**kw)
        self.assertIsNone(preview.get("error"))
        self.assertTrue(any(text in w for w in preview["warnings"]), preview["warnings"])

    def test_real_launch_with_everything_present(self):
        self.assertIsNone(self.build(dry_run=False).get("error"))

    def test_root_unset(self):
        catalog.SETTINGS["strata_hy4_root"] = ""
        catalog.SETTINGS["strata_root"] = ""
        self.assertRefused('set "strata_hy4_root"')

    def test_root_relative(self):
        catalog.SETTINGS["strata_hy4_root"] = "Strata"
        self.assertIn("absolute path", self.build()["error"])

    def test_strata_root_fallback(self):
        catalog.SETTINGS["strata_hy4_root"] = ""
        catalog.SETTINGS["strata_root"] = str(self.root)
        self.assertIsNone(self.build(dry_run=False).get("error"))

    def test_missing_pieces(self):
        for rel, text in (("build-hy4-sycl-bmg/strata-hy4-sycl", "the engine"),
                          (".venv-sycl/pyvenv.cfg", "Python environment"),
                          ("engine/sycl/serve/server_intel.py", "serve front-end")):
            with self.subTest(rel=rel):
                (self.root / rel).rename(self.root / (rel + ".gone"))
                try:
                    self.assertRefused(text)
                finally:
                    (self.root / (rel + ".gone")).rename(self.root / rel)

    def test_missing_packs(self):
        self.engram.unlink()
        self.assertRefused("Engram")
        support_env.write_file(self.engram)
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}
        self.assertRefused("the model file")

    def test_recipe_paths_stay_inside(self):
        for key, value in (("hy4_bin", "../bin/x"), ("hy4_bin", "/etc/x"),
                           ("hy4_python", "a//b"), ("hy4_server", "..\\x"),
                           ("engram_file", "dir/x.strata"), ("engram_file", "/abs")):
            with self.subTest(key=key, value=value):
                saved = self.recipe().get(key)
                self.recipe()[key] = value
                try:
                    self.assertIn(key, self.build()["error"])
                finally:
                    self.recipe()[key] = saved

    def test_numeric_fields(self):
        for key, value in (("threads", "8 --evil"), ("threads", 0),
                           ("ram_experts_gb", "x"), ("peer_vram_experts_gb", 99),
                           ("vram_experts_by_ctx", {"bad": 15})):
            with self.subTest(key=key, value=value):
                saved = self.recipe().get(key)
                self.recipe()[key] = value
                try:
                    self.assertTrue(self.build().get("error"), key)
                finally:
                    self.recipe()[key] = saved

    def test_windows(self):
        with mock.patch("b70launcher.builders.strata_hy4.IS_WIN", True):
            self.assertIn("Linux only", self.build()["error"])


class TestHy4Catalog(support_env.LauncherStateCase):

    def test_shipped_recipe(self):
        m = catalog.find_model(MID)
        self.assertIsNotNone(m)
        self.assertEqual(m["recommended_engine"], "strata")
        r = m["recipes"][ENG]
        self.assertEqual((r["kind"], r["image"], r["gpus"]), ("strata-hy4", "strata-sycl-dev", [0, 1]))
        self.assertEqual((r["ctx"], r["ctx_safe"], r["ctx_max"]), (131072, 32768, 262144))
        self.assertEqual(r["draft"], "dfly2")
        self.assertEqual(r["peer_vram_experts_gb"], 25)
        self.assertEqual(r["vram_experts_by_ctx"], {"32768": 17, "131072": 15, "262144": 14})
        self.assertEqual(r["cookbook_url"], COOKBOOK_URL)
        self.assertTrue(r.get("recipe_ver"))
        dl = r["download"]
        self.assertEqual((dl["kind"], dl["name"]), ("file", NATIVE))
        self.assertEqual(dl["extra_files"], [ENGRAM])
        self.assertEqual(dl["repo"], "strata-engine/Hy4-Flash-Strata")
        self.assertEqual(dl["revision"], "995bac227ab7e0d221772be641e2b5954a9fac06")
        self.assertIn(dl["repo"], downloads.HF_APPROVED_REPOS)
        self.assertIn("strata", [e["id"] for e in catalog.RECIPES["engines"]])
        # exec-shaping fields stay out of the remote allowlist
        for key in ("kind", "image", "env", "fixed_flags", "hy4_bin", "hy4_python",
                    "hy4_server", "engram_file", "draft", "vram_experts_by_ctx",
                    "peer_vram_experts_gb", "ram_experts_gb", "kv_host_above"):
            self.assertNotIn(key, catalog.REMOTE_RECIPE_FIELDS, key)

    def test_remote_overlay_cannot_reshape_the_launch(self):
        original = json.loads(json.dumps(catalog.find_model(MID)["recipes"][ENG]))
        evil = {"kind": "gguf", "image": "evil/image", "env": {"HY4S_EVIL": "1"},
                "fixed_flags": ["--evil"], "hy4_bin": "evil", "hy4_python": "evil",
                "hy4_server": "evil", "engram_file": "evil.strata", "draft": "off",
                "peer_vram_experts_gb": 0, "model_path": "/etc/shadow",
                "ctx": 65536, "cookbook_url": "https://example.com/new"}
        applied, needs_app, error = catalog.apply_recipe_doc(
            {"models": [{"id": MID, "recipes": {ENG: dict(evil, recipe_ver="9999-01-01")}}]})
        self.assertEqual((applied, needs_app, error), (1, [], None))
        r = catalog.find_model(MID)["recipes"][ENG]
        self.assertEqual(r["ctx"], 65536)                      # data fields still merge
        self.assertEqual(r["cookbook_url"], "https://example.com/new")
        for key in ("kind", "image", "env", "fixed_flags", "hy4_bin", "hy4_python",
                    "hy4_server", "engram_file", "draft", "peer_vram_experts_gb", "model_path"):
            self.assertEqual(r.get(key), original.get(key), key)

    def test_recipe_needs_root(self):
        catalog.SETTINGS["strata_hy4_root"] = ""
        catalog.SETTINGS["strata_root"] = ""
        out = {n["setting"] for n in scan.recipe_needs(catalog.find_model(MID), ENG)}
        self.assertIn("strata_hy4_root", out)

    def test_detect_prefers_native_pack_not_engram(self):
        base = support_env.TMPROOT / "hy4-detect"
        native = support_env.write_file(base / NATIVE)
        support_env.write_file(base / ENGRAM)
        support_env.write_file(base / "Hy4-Flash-Vision.strata")
        support_env.write_file(base / "Hy4-Flash-SQ-96.strata")
        self.addCleanup(shutil.rmtree, base, True)
        scan.SCAN["items"] = {"gguf": {n.lower(): str(base / n) for n in
                                       (NATIVE, ENGRAM, "Hy4-Flash-Vision.strata", "Hy4-Flash-SQ-96.strata")},
                              "snapshots": {}}
        det = scan.detect(catalog.find_model(MID), ENG)
        self.assertEqual(det["path"], str(native))
        self.assertFalse(det["variant"])

    def test_tools_supported(self):
        from b70launcher import harness
        self.assertTrue(harness.supports_tools({"engine": "strata", "cmd": "docker run"}))


if __name__ == "__main__":
    unittest.main()
