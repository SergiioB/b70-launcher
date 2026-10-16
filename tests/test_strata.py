"""Strata builder: command rendering, prerequisite refusals, env allow-list,
remote-overlay safety. Every call is a build only — nothing is executed.

Run:  cd b70-launcher && python3 -m unittest tests.test_strata -v
"""
import json
import os
import shlex
import shutil
import unittest
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import builders, catalog, downloads, harness, hardware, runtime, scan

MID, ENG = "qwen38-flashnext-strata", "strata"
SHARD1 = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
SHARD2 = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
TWO_CARDS = {"devices": [
    {"render": "/dev/dri/renderD128", "card": "card0", "b70": True, "accessible": True},
    {"render": "/dev/dri/renderD129", "card": "card1", "b70": True, "accessible": True}],
    "blockers": [], "blockers_by_kind": {}, "default": {"gpus": [0, 1]}}


def config_of(plan):
    """The server config the container writes at start (printf '%s' <json>)."""
    words = shlex.split(plan["tokens"][-1])
    return json.loads(words[words.index("printf") + 2])


def env_of(plan):
    t = plan["tokens"]
    return dict(t[i + 1].split("=", 1) for i, x in enumerate(t) if x == "-e")


def mounts_of(plan):
    t = plan["tokens"]
    return {t[i + 1].rsplit(":", 2)[1]: t[i + 1].rsplit(":", 2)[0] for i, x in enumerate(t) if x == "-v"}


class StrataCase(support_env.LauncherStateCase):
    """A complete Strata layout: checkout, data folder, both shards (shard 2 a link)."""

    def setUp(self):
        super().setUp()
        self.base = support_env.TMPROOT / f"strata-{self.id().rsplit('.', 1)[-1]}"
        self.root, self.data = self.base / "Strata", self.base / "Strata-data"
        for rel in ("sycl/serve/server_intel.py", "build-sycl-aot/strata", ".venv-sycl/pyvenv.cfg",
                    "data/expert-profile-sycl.bin", "data/draft_vocab_en.bin"):
            support_env.write_file(self.root / rel)
        for rel in ("packs/iq3_xxs/native_experts.txt", "packs/iq3_xxs/tokenizer/vocab.json", "mtp/rt/experts.bin"):
            support_env.write_file(self.data / rel)
        self.shard1 = support_env.write_file(self.data / "models" / "IQ3_XXS" / SHARD1)
        self.ple = support_env.write_file(self.base / "shared" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")
        os.symlink(self.ple, self.shard1.parent / SHARD2)
        self.addCleanup(shutil.rmtree, self.base, True)
        catalog.SETTINGS["strata_root"] = str(self.root)
        scan.SCAN["items"] = {"gguf": {SHARD1.lower(): str(self.shard1)}, "snapshots": {}}
        patcher = mock.patch.object(hardware, "hardware_preflight", return_value=TWO_CARDS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **kw):
        cfg = {"model_id": MID, "engine": ENG, "port": 18199, "dry_run": True}
        cfg.update(kw)
        return builders.build(cfg)

    def recipe(self):
        return catalog.find_model(MID)["recipes"][ENG]


class TestStrataCommand(StrataCase):

    def test_container_shape(self):
        b = self.build()
        self.assertIsNone(b.get("error"), b.get("error"))
        t = b["tokens"]
        self.assertEqual(t[:3], ["docker", "run", "--rm"])
        self.assertEqual(b["cname"], "b70-qwen38-flashnext-strata-strata-18199")
        self.assertIn("b70.launcher.rid=qwen38-flashnext-strata-strata-18199", t)
        self.assertIn("127.0.0.1:18199:18199", t)
        self.assertIn("SYS_PTRACE", t)
        self.assertIn("memlock=-1", t)
        self.assertNotIn("--privileged", t)
        self.assertNotIn("--network", t)
        mapped = [t[i + 1] for i, x in enumerate(t) if x == "--device"]
        self.assertEqual(mapped, ["/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/card0", "/dev/dri/card1"])
        self.assertEqual(t[t.index("--entrypoint") + 1:t.index("--entrypoint") + 4], ["bash", "strata-sycl-dev", "-lc"])
        self.assertEqual(b["served_name"], "Qwen3.8-Flash-Next IQ3_XXS")
        self.assertEqual(b["effective_binary"], "strata-sycl-dev")
        self.assertIn("docker build -t strata-sycl-dev sycl/tools", b["image_hint"])
        self.assertFalse(any("Strata recipe needs" in w for w in b["warnings"]))

    def test_every_host_mount_is_read_only_and_shards_resolve(self):
        b = self.build()
        t = b["tokens"]
        vols = [t[i + 1] for i, x in enumerate(t) if x == "-v"]
        self.assertTrue(all(v.endswith(":ro") for v in vols), vols)
        m = mounts_of(b)
        self.assertEqual(m["/strata"], str(self.root))
        self.assertEqual(m["/strata-data/pack"], str(self.data / "packs/iq3_xxs"))
        self.assertEqual(m["/strata-data/mtp"], str(self.data / "mtp/rt"))
        self.assertEqual(m[f"/models/{SHARD1}"], str(self.shard1.resolve()))
        self.assertEqual(m[f"/models/{SHARD2}"], str(self.ple.resolve()))  # the link's target is mounted

    def test_server_config(self):
        b = self.build(ctx=131072)
        c = config_of(b)
        self.assertEqual(c["exe"], "/strata/build-sycl-aot/strata")
        self.assertEqual(c["tokenizer"], "/strata-data/pack/tokenizer")
        self.assertEqual(c["model_name"], "Qwen3.8-Flash-Next IQ3_XXS")
        self.assertEqual((c["gpu"], c["parallel"], c["backend"]), ([0, 1], 2, "sycl"))
        a = c["args"]
        for flag, val in (("--pack", "/strata-data/pack"), ("--native", f"/models/{SHARD1}"),
                          ("--ple-gguf", f"/models/{SHARD2}"), ("--expert-parallel-device", "1"),
                          ("--expert-profile", "/strata/data/expert-profile-sycl.bin"),
                          ("--max-context", "131072"), ("--kv", "int8"), ("--spec", "4"),
                          ("--mtp", "/strata-data/mtp"), ("--spec-min-p", "0"),
                          ("--mtp-draft-vocab", "/strata/data/draft_vocab_en.bin"),
                          ("--conversation-cache-mib", "32768")):
            self.assertEqual(a[a.index(flag) + 1], val, flag)
        self.assertNotIn("--prompt-cache", a)
        script = b["tokens"][-1]
        self.assertIn("exec /strata/.venv-sycl/bin/python sycl/serve/server_intel.py --engine strata "
                      "--config /tmp/b70-strata/config.json --host 0.0.0.0 --port 18199", script)
        self.assertIn("tail -n +1 -F /tmp/b70-strata/engine.log &", script)

    def test_env_recipe_then_caller(self):
        b = self.build(extra_env="STRATA_TRACE=0\nGLIBC_TUNABLES=glibc.malloc.hugetlb=1")
        e = env_of(b)
        self.assertEqual(e["ONEAPI_DEVICE_SELECTOR"], "level_zero:*")
        self.assertEqual(e["SYCL_CACHE_PERSISTENT"], "0")
        self.assertEqual(e["STRATA_VERIFY_NO_HOST"], "1")
        self.assertEqual(e["STRATA_TRACE"], "0")  # the caller's entry wins ...
        self.assertEqual(sum(x == "-e" and b["tokens"][i + 1].startswith("STRATA_TRACE=")
                             for i, x in enumerate(b["tokens"])), 1)  # ... as the only entry
        self.assertEqual(e["GLIBC_TUNABLES"], "glibc.malloc.hugetlb=1")
        self.assertNotIn("ZE_AFFINITY_MASK", e)

    def test_no_mtp(self):
        b = self.build(mtp=False)
        a = config_of(b)["args"]
        for flag in ("--spec", "--mtp", "--spec-min-p", "--mtp-draft-vocab"):
            self.assertNotIn(flag, a)
        self.assertNotIn("/strata-data/mtp", mounts_of(b))
        self.assertTrue(any("disabled by --no-mtp" in w for w in b["warnings"]))
        shutil.rmtree(self.data / "mtp")  # no draft layer needed without MTP
        self.assertIsNone(self.build(mtp=False, dry_run=False).get("error"))

    def test_cache_off_overrides_last(self):
        a = config_of(self.build(cache="off"))["args"]
        self.assertEqual(a[a.index("--prompt-cache") + 1], "0")
        last = len(a) - 1 - a[::-1].index("--conversation-cache-mib")
        self.assertEqual(a[last + 1], "0")

    def test_slots_kv_ctx_and_extra(self):
        b = self.build(slots=12, kv="q8_0", ctx=262144, extra="--verbose")
        c = config_of(b)
        self.assertEqual(c["parallel"], 8)
        self.assertEqual(c["args"][c["args"].index("--kv") + 1], "int8")
        self.assertEqual(c["args"][c["args"].index("--max-context") + 1], "200000")
        self.assertEqual(c["args"][-1], "--verbose")
        w = " ".join(b["warnings"])
        for text in ("at most 8 requests", "--kv 'q8_0'", "clamping"):
            self.assertIn(text, w)

    def test_one_card_is_refused(self):
        b = self.build(gpus=[0])
        self.assertIn("needs 2 GPUs", b["error"])

    def test_strata_data_setting(self):
        moved = self.base / "elsewhere"
        shutil.copytree(self.data / "packs", moved / "packs")
        shutil.copytree(self.data / "mtp", moved / "mtp")
        catalog.SETTINGS["strata_data"] = str(moved)
        self.assertEqual(mounts_of(self.build())["/strata-data/pack"], str(moved / "packs/iq3_xxs"))


class TestStrataRefusals(StrataCase):

    def assertRefused(self, text, **kw):
        b = self.build(dry_run=False, **kw)
        self.assertIn(text, b.get("error") or "")
        preview = self.build(**kw)  # the plan still renders, with the same reason
        self.assertIsNone(preview.get("error"))
        self.assertTrue(any(text in w for w in preview["warnings"]), preview["warnings"])

    def test_real_launch_with_everything_present(self):
        self.assertIsNone(self.build(dry_run=False).get("error"))

    def test_strata_root_unset(self):
        catalog.SETTINGS["strata_root"] = ""
        self.assertRefused('set "strata_root"')

    def test_strata_root_relative(self):
        catalog.SETTINGS["strata_root"] = "Strata"
        self.assertIn("absolute path", self.build()["error"])

    def test_missing_pieces(self):
        for rel, text in (("build-sycl-aot/strata", "the engine"), (".venv-sycl/pyvenv.cfg", "Python environment"),
                          ("sycl/serve/server_intel.py", "Strata checkout"),
                          ("data/expert-profile-sycl.bin", "expert profile"),
                          ("data/draft_vocab_en.bin", "draft vocabulary")):
            with self.subTest(rel=rel):
                (self.root / rel).rename(self.root / (rel + ".gone"))
                try:
                    self.assertRefused(text)
                finally:
                    (self.root / (rel + ".gone")).rename(self.root / rel)
        for rel, text in (("packs/iq3_xxs/native_experts.txt", "the pack"), ("mtp/rt/experts.bin", "MTP draft layer")):
            with self.subTest(rel=rel):
                (self.data / rel).unlink()
                self.assertRefused(text)
                support_env.write_file(self.data / rel)

    def test_shards(self):
        (self.shard1.parent / SHARD2).unlink()
        self.assertRefused("second shard")
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}
        self.assertRefused("the model file")

    def test_recipe_paths_stay_inside(self):
        for key, value in (("strata_bin", "../bin/strata"), ("strata_pack", "/etc"), ("strata_python", "a//b")):
            with self.subTest(key=key):
                saved = self.recipe().get(key)
                self.recipe()[key] = value
                try:
                    self.assertIn(key, self.build()["error"])
                finally:
                    self.recipe()[key] = saved

    def test_recipe_env_allow_list(self):
        for env, text in (({"LD_PRELOAD": "/tmp/x.so"}, "LD_PRELOAD"), ({"GLIBC_TUNABLES": "x"}, "GLIBC_TUNABLES"),
                          ({"STRATA_X": "a\nb"}, "single-line"), ({"STRATA_X": {"a": 1}}, "single-line")):
            with self.subTest(env=env):
                self.recipe()["env"] = env
                self.assertIn(text, self.build()["error"])

    def test_numeric_fields(self):
        for key, value, text in (("spec_tokens", "4 --evil", "must be numbers"), ("spec_p_min", "x", "must be numbers"),
                                 ("parallel", "two", "must be numbers"), ("spec_tokens", 99, "out of range"),
                                 ("spec_p_min", 2, "out of range")):
            with self.subTest(key=key, value=value):
                saved = self.recipe().get(key)
                self.recipe()[key] = value
                try:
                    self.assertIn(text, self.build().get("error") or "")
                finally:
                    self.recipe()[key] = saved

    def test_windows(self):
        with mock.patch("b70launcher.builders.strata.IS_WIN", True):
            self.assertIn("Linux only", self.build()["error"])


class TestStrataCatalog(support_env.LauncherStateCase):

    def test_shipped_recipe(self):
        r = catalog.find_model(MID)["recipes"][ENG]
        self.assertEqual((r["kind"], r["image"], r["gpus"]), ("strata", "strata-sycl-dev", [0, 1]))
        self.assertTrue(r["env"] and all(k.startswith("STRATA_") for k in r["env"]))
        self.assertIn(r["download"]["repo"], downloads.HF_APPROVED_REPOS)
        self.assertTrue(r.get("recipe_ver"))
        self.assertIn("strata", [e["id"] for e in catalog.RECIPES["engines"]])

    def test_remote_overlay_cannot_reshape_the_launch(self):
        original = json.loads(json.dumps(catalog.find_model(MID)["recipes"][ENG]))
        exec_keys = ("kind", "image", "env", "fixed_flags", "strata_bin", "strata_python", "strata_pack",
                     "strata_mtp", "expert_profile", "mtp_draft_vocab", "gguf")
        evil = {"kind": "gguf", "image": "evil/image", "env": {"STRATA_EVIL": "1"}, "fixed_flags": ["--evil"],
                "strata_bin": "evil", "strata_python": "evil", "strata_pack": "evil", "strata_mtp": "evil",
                "expert_profile": "evil", "mtp_draft_vocab": "evil", "gguf": "/etc/shadow", "ctx": 65536}
        applied, needs_app, error = catalog.apply_recipe_doc(
            {"models": [{"id": MID, "recipes": {ENG: dict(evil, recipe_ver="9999-01-01")}}]})
        self.assertEqual((applied, needs_app, error), (1, [], None))
        r = catalog.find_model(MID)["recipes"][ENG]
        self.assertEqual(r["ctx"], 65536)
        for key in exec_keys:
            self.assertNotIn(key, catalog.REMOTE_RECIPE_FIELDS)
            self.assertEqual(r.get(key), original.get(key), key)

    def test_settings_override_paths(self):
        catalog.paths.atomic_write(catalog.paths.OVERRIDE_PATH, json.dumps(
            {"strata_root": "/opt/Strata", "strata_data": "/opt/Strata-data"}))
        catalog._merge_override()
        self.assertEqual((catalog.SETTINGS["strata_root"], catalog.SETTINGS["strata_data"]),
                         ("/opt/Strata", "/opt/Strata-data"))

    def test_tools_and_image_hint(self):
        self.assertTrue(harness.supports_tools({"engine": "strata", "cmd": "docker run"}))
        built = {"native": False, "effective_binary": "strata-sycl-dev", "kind": "strata",
                 "tokens": ["docker", "run"], "cmd": "docker run", "image_hint": "Build it here."}
        entry = {"status": "starting", "docker_sock": "", "log": str(support_env.TMPROOT / "strata-hint.log")}
        with mock.patch.object(runtime.subprocess, "run", return_value=mock.Mock(returncode=1)):
            runtime._start(entry, built)
        self.assertEqual(entry["status"], "exited")
        self.assertIn("not on this machine: strata-sycl-dev. Build it here.", entry["error"])


if __name__ == "__main__":
    unittest.main()
