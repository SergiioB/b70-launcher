"""Tests for build() command generation, KV resolution, draft flags,
usage accounting, state persistence and harness lines.

All calls are dry-builds: build() never executes the generated command.
exl3 is intentionally not exercised — it probes `docker -H unix://...info`
even for dry runs.

Run:  cd b70-launcher && python3 -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

try:
    import support_env
except ImportError:
    from tests import support_env

launcher = support_env.launcher
TMP = support_env.TMPROOT
FAKE_HOME = support_env.FAKE_HOME
LauncherStateCase = support_env.LauncherStateCase
write_file = support_env.write_file
free_port = support_env.free_port

DISK = TMP / "build-disk"


def cfg(**kw):
    base = {"ctx": 8192, "port": free_port(), "dry_run": True}
    base.update(kw)
    return base


class TestBuildLlamaCppDocker(LauncherStateCase):
    def test_docker_command_shape(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp"))
        self.assertNotIn("error", b)
        t = b["tokens"]
        self.assertEqual(t[:3], ["docker", "run", "-d"])
        self.assertIn("b70-qwen36-35b-sycl", t)
        self.assertIn("-m", t)
        self.assertEqual(t[t.index("-m") + 1], "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
        for flag, val in (("-ngl", "99"), ("--host", "0.0.0.0"),
                          ("--flash-attn", "on"), ("-c", "8192")):
            self.assertEqual(t[t.index(flag) + 1], val, flag)
        self.assertIn("--metrics", t)
        self.assertFalse(b["native"])
        self.assertFalse(b["detected"])
        self.assertTrue(any("not detected" in w for w in b["warnings"]))
        self.assertEqual(b["endpoint"], f"http://127.0.0.1:{t[t.index('--port') + 1]}/v1")

    def test_kv_map_all_options(self):
        expect = {
            "q5_0/q4_1": ("q5_0", "q4_1"),
            "q8_0": ("q8_0", "q8_0"),
            "q8_0/q4_1": ("q8_0", "q4_1"),
            "f16": ("f16", "f16"),
        }
        for kv, (k, v) in expect.items():
            b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp", kv=kv))
            t = b["tokens"]
            self.assertEqual(t[t.index("--cache-type-k") + 1], k, kv)
            self.assertEqual(t[t.index("--cache-type-v") + 1], v, kv)

    def test_kv_recipe_default_falls_back_to_q8_0(self):
        # shipped recipes carry no "kv" key; "recipe default" -> q8_0
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp"))
        t = b["tokens"]
        self.assertEqual(t[t.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(t[t.index("--cache-type-v") + 1], "q8_0")

    def test_kv_f16_warns(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp", kv="f16"))
        self.assertTrue(any("f16 KV wastes VRAM" in w for w in b["warnings"]))

    def test_fixed_flags_suppress_kv_map(self):
        # flashnext ships --cache-type-* in fixed_flags: kv_map must not run
        b = launcher.build(cfg(model_id="qwen38-flashnext", engine="llamacpp",
                               kv="f16"))
        t = b["tokens"]
        self.assertEqual(t.count("--cache-type-k"), 1)
        self.assertEqual(t[t.index("--cache-type-k") + 1], "q8_0")
        self.assertNotIn("f16", t)
        self.assertTrue(any("f16 KV wastes VRAM" in w for w in b["warnings"]))

    def test_tiered_memory_flags(self):
        b = launcher.build(cfg(model_id="qwen38-flashnext", engine="llamacpp",
                               gpus=[0, 1]))
        t = b["tokens"]
        self.assertIn("-ot", t)                       # offload_tensors
        self.assertIn("--tensor-split", t)
        self.assertEqual(t[t.index("--tensor-split") + 1], "49,51")
        self.assertIn("--split-mode", t)
        self.assertTrue(any("Dual-GPU tensor split" in w or "3-Tiered" in w
                            for w in b["warnings"]))

    def test_detected_artifact_mounts_parent(self):
        f = write_file(DISK / "det" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
        launcher.SCAN["items"] = {
            "gguf": {"qwen3.6-35b-a3b-ud-q4_k_xl.gguf": str(f)},
            "snapshots": {}}
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp"))
        t = b["tokens"]
        self.assertTrue(b["detected"])
        self.assertIn(f"{f.parent}:/models:ro", t)
        self.assertEqual(t[t.index("-m") + 1], "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")

    def test_variant_detection_warns(self):
        f = write_file(DISK / "det2" / "Alt-Qwen3.6-35B-Q8_0.gguf")
        launcher.SCAN["items"] = {
            "gguf": {"alt-qwen3.6-35b-q8_0.gguf": str(f)}, "snapshots": {}}
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp"))
        self.assertTrue(b["detected"])
        self.assertTrue(any("using your local" in w for w in b["warnings"]))

    def test_extra_flags_and_env(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp",
                               extra="--top-p 0.9", slots=4,
                               extra_env="GOOD=1\n# comment\nX_VAR=two"))
        t = b["tokens"]
        self.assertEqual(t[-2:], ["--top-p", "0.9"])
        self.assertIn("-np", t)
        self.assertEqual(t[t.index("-np") + 1], "4")
        self.assertIn("GOOD=1", t)
        self.assertIn("X_VAR=two", t)


class TestBuildLlamaCppNative(LauncherStateCase):
    def setUp(self):
        super().setUp()
        self.bin = write_file(TMP / "fake-bin" / "llama-server")
        launcher.SETTINGS["llama_bin"] = str(self.bin)

    def test_native_tokens_and_env(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp"))
        t = b["tokens"]
        self.assertTrue(b["native"])
        self.assertEqual(t[0], str(self.bin))
        self.assertNotIn("docker", t)
        self.assertEqual(t[t.index("--host") + 1], "127.0.0.1")
        env = b["env"]
        self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:0")
        self.assertEqual(env["SYCL_DEVICE_FILTER"], "level_zero")
        self.assertEqual(env["ZE_FLAT_DEVICE_HIERARCHY"], "COMPOSITE")
        self.assertEqual(env["SYCL_CACHE_PERSISTENT"], "0")

    def test_tiered_env_flags(self):
        b = launcher.build(cfg(model_id="qwen38-flashnext", engine="llamacpp"))
        self.assertEqual(b["env"]["LLAMA_ATTN_ROT_DISABLE"], "1")
        self.assertEqual(b["env"]["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"], "0")

    def test_use_docker_overrides_binary(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp",
                               use_docker=True))
        self.assertFalse(b["native"])
        self.assertEqual(b["tokens"][0], "docker")

    def test_missing_binary_falls_back_to_docker(self):
        launcher.SETTINGS["llama_bin"] = str(TMP / "no-such-bin")
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp"))
        self.assertFalse(b["native"])
        self.assertEqual(b["tokens"][0], "docker")

    def test_tilde_llama_bin(self):
        f = write_file(FAKE_HOME / "bin" / "llama-server")
        launcher.SETTINGS["llama_bin"] = "~/bin/llama-server"
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp"))
        self.assertTrue(b["native"])
        self.assertEqual(b["tokens"][0], str(f))


class TestBuildDraftModel(LauncherStateCase):
    """draft_model resolves through the disk scan (0.4.7 feature)."""

    def setUp(self):
        super().setUp()
        self.draft = write_file(DISK / "drafts" / "Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf")
        launcher.SCAN["items"] = {
            "gguf": {"qwen3.8-flash-next-mtp-q4_k_m.gguf": str(self.draft)},
            "snapshots": {}}

    def test_draft_flags_docker(self):
        b = launcher.build(cfg(model_id="qwen38-flashnext", engine="llamacpp",
                               gpus=[0, 1]))
        t = b["tokens"]
        self.assertEqual(t[0], "docker")
        self.assertIn(f"{self.draft.parent}:/draft:ro", t)
        self.assertEqual(t[t.index("-md") + 1],
                         "/draft/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf")
        for flag, val in (("-ngld", "999"), ("--spec-type", "draft-mtp"),
                          ("--spec-draft-n-max", "3"),
                          ("--spec-draft-p-min", "0.75"),
                          ("--spec-draft-device", "SYCL1")):
            self.assertEqual(t[t.index(flag) + 1], val, flag)
        self.assertTrue(any("MTP Speculative Decoding" in w
                            for w in b["warnings"]))

    def test_draft_flags_native_host_path(self):
        launcher.SETTINGS["llama_bin"] = str(
            write_file(TMP / "fake-bin" / "llama-server"))
        b = launcher.build(cfg(model_id="qwen38-flashnext", engine="llamacpp"))
        t = b["tokens"]
        self.assertEqual(t[t.index("-md") + 1], str(self.draft))
        self.assertNotIn("/draft/", " ".join(t))

    def test_no_draft_no_flags(self):
        launcher.SCAN["items"] = {"gguf": {}, "snapshots": {}}
        b = launcher.build(cfg(model_id="qwen38-flashnext", engine="llamacpp"))
        self.assertNotIn("-md", b["tokens"])
        self.assertNotIn("--spec-type", b["tokens"])


class TestBuildOtherEngines(LauncherStateCase):
    def test_ovms_command_shape(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="openvino",
                               ctx=131072))
        t = b["tokens"]
        self.assertEqual(t[:3], ["docker", "run", "-d"])
        self.assertIn("b70-qwen36-35b-ovms", t)
        self.assertTrue(any(x.startswith("openvino/model_server:2026.4.1") for x in t))
        port = b["endpoint"].rsplit(":", 1)[-1].split("/")[0]
        self.assertEqual(t[t.index("--rest_port") + 1], port)
        self.assertIn(f"127.0.0.1:{port}:{port}", t)
        self.assertIn("--model_repository_path", t)
        self.assertEqual(t[t.index("--source_model") + 1],
                         "OpenVINO/Qwen3.6-35B-A3B-int4-ov")
        self.assertEqual(t[t.index("--tool_parser") + 1], "qwen3coder")
        self.assertEqual(t[t.index("--reasoning_parser") + 1], "qwen3")
        # cim_long_ctx engages only above 20480
        self.assertEqual(t[t.index("--cache_interval_multiplier") + 1], "64")
        self.assertTrue(any("OVMS manages KV" in w for w in b["warnings"]))

    def test_ovms_no_cim_at_small_ctx(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="openvino",
                               ctx=20480))
        self.assertNotIn("--cache_interval_multiplier", b["tokens"])

    def test_vllm_dflash_command(self):
        # nemotron-35 routes through the vllm-dflash branch: patch mounts +
        # SSU tuning + `vllm serve` under an entrypoint script
        b = launcher.build(cfg(model_id="nemotron-35", engine="vllm",
                               kv="fp8"))
        t = b["tokens"]
        self.assertEqual(t[0], "docker")
        self.assertIn("b70-nemotron-35-vllm", t)
        script = " ".join(t)
        self.assertIn("vllm serve /models/Nemotron-3.5-Lightning-GPTQ-Int4", script)
        self.assertIn("--kv-cache-dtype fp8", script)
        self.assertIn("--max-model-len 8192", script)
        self.assertIn("--no-enable-prefix-caching", script)
        self.assertTrue(any("patch" in x for x in t))
        self.assertTrue(any("/ssu" in x for x in t))

    def test_vllm_recipe_default_kv_has_no_flag(self):
        b = launcher.build(cfg(model_id="nemotron-35", engine="vllm"))
        self.assertNotIn("--kv-cache-dtype", b["tokens"])

    def test_vllm_tp2_command(self):
        # engine flags live inside the `-lc` script token for vllm-* kinds
        b = launcher.build(cfg(model_id="qwen38-27b-fp8-tp2", engine="vllm",
                               gpus=[0, 1]))
        t = b["tokens"]
        script = " ".join(t)
        self.assertIn("--tensor-parallel-size 2", script)
        self.assertIn("--kv-cache-dtype fp8", script)
        self.assertTrue(any("patch_vllm_worker_affinity" in x for x in t))
        self.assertIn("--speculative-config", script)  # spec_tokens 8 + mtp on
        self.assertIn('"num_speculative_tokens": 8', script)
        self.assertTrue(any("Dual" in w or "dual" in w for w in b["warnings"]))

    def test_vllm_mtp_off(self):
        b = launcher.build(cfg(model_id="qwen38-27b-fp8-tp2", engine="vllm",
                               gpus=[0, 1], mtp=False))
        self.assertNotIn("--speculative-config", " ".join(b["tokens"]))

    def test_vllm_arext_requires_artifact(self):
        b = launcher.build(cfg(model_id="qwen38-27b", engine="vllm"))
        self.assertIn("error", b)
        self.assertIn("AutoRound artifact not found", b["error"])

    def test_vllm_arext_command(self):
        mdir = FAKE_HOME / "models" / "Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
        mdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            FAKE_HOME / "models", ignore_errors=True))
        b = launcher.build(cfg(model_id="qwen38-27b", engine="vllm"))
        self.assertNotIn("error", b)
        t = b["tokens"]
        script = " ".join(t)
        self.assertIn("--kv-cache-dtype fp8", script)
        for patch in ("patch_mtp_nightly", "patch_mtp_boundary",
                      "patch_champion_stack_overlay"):
            self.assertTrue(any(patch in x for x in t), patch)
        self.assertIn("--speculative-config", script)
        self.assertIn('"num_speculative_tokens": 4', script)

    def test_vllm_mtp_snapshot(self):
        d = DISK / "snap-mtp" / "Qwen3.6-35B-A3B-GPTQ-Int4"
        d.mkdir(parents=True, exist_ok=True)
        launcher.SCAN["items"] = {"gguf": {}, "snapshots": {
            "qwen3.6-35b-a3b-gptq-int4": {"path": str(d), "kind": "vllm",
                                        "repo_hint": "x", "ctx": None}}}
        b = launcher.build(cfg(model_id="qwen36-35b", engine="vllm"))
        self.assertNotIn("error", b)
        t = b["tokens"]
        script = " ".join(t)
        self.assertIn(f"{d}:/model:ro", t)
        self.assertIn("--speculative-config", script)
        self.assertIn("patch_mtp_nightly", script)

    def test_ctx_clamped_with_warning(self):
        # recipes with ctx_max clamp instead of erroring (qwen38-flashnext: 131072)
        b = launcher.build(cfg(model_id="qwen38-flashnext", engine="llamacpp",
                               ctx=200000))
        self.assertNotIn("error", b)
        self.assertEqual(b["ctx"], 131072)
        self.assertTrue(any("clamping" in w for w in b["warnings"]))
        # without ctx_max, an out-of-range ctx is rejected outright
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp",
                               ctx=999999))
        self.assertIn("error", b)


class TestBuildValidation(LauncherStateCase):
    def test_unknown_model_and_engine(self):
        self.assertIn("error", launcher.build(cfg(model_id="nope", engine="x")))
        self.assertIn("error", launcher.build(cfg(model_id="qwen36-35b",
                                                  engine="nope")))

    def test_port_ctx_slots_ranges(self):
        self.assertIn("error", launcher.build(cfg(model_id="qwen36-35b",
                        engine="llamacpp", port=70000)))
        self.assertIn("error", launcher.build(cfg(model_id="qwen36-35b",
                        engine="llamacpp", ctx=511)))
        self.assertIn("error", launcher.build(cfg(model_id="qwen36-35b",
                        engine="llamacpp", slots=129)))
        self.assertIn("error", launcher.build(cfg(model_id="qwen36-35b",
                        engine="llamacpp", slots=-1)))

    def test_falsy_numeric_fields_rejected(self):
        # explicit 0 is a value, not "unset" — out-of-range port/slots must error
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp",
                               port=0, slots=0, ctx=0))
        self.assertIn("error", b)

    def test_bad_ctx_port_values(self):
        self.assertIn("error", launcher.build(cfg(model_id="qwen36-35b",
                        engine="llamacpp", ctx="abc")))
        self.assertIn("error", launcher.build(cfg(model_id="qwen36-35b",
                        engine="llamacpp", port="xyz")))

    def test_gpu_selection_rules(self):
        base = dict(model_id="qwen36-35b", engine="llamacpp")
        for bad in ([0, 0], [2], [], "0", [0, 1, 1], [-1]):
            self.assertIn("error", launcher.build(cfg(gpus=bad, **base)), bad)
        ok = launcher.build(cfg(gpus=[0], **base))
        self.assertNotIn("error", ok)

    def test_bad_extra_env_name(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp",
                               extra_env="9BAD=1"))
        self.assertIn("Invalid environment variable name", b["error"])

    def test_bad_extra_flags(self):
        b = launcher.build(cfg(model_id="qwen36-35b", engine="llamacpp",
                               extra="--unclosed 'quote"))
        self.assertIn("error", b)

    def test_custom_path_through_build(self):
        root = TMP / "custom-scan"
        f = write_file(root / "own.gguf")
        launcher.SETTINGS["scan_dirs"] = [str(root)]
        b = launcher.build(cfg(model_id="__custom__", custom_path=str(f),
                               engine="llamacpp"))
        self.assertNotIn("error", b)
        self.assertEqual(b["model_name"], "own.gguf")
        self.assertTrue(b["detected"])
        self.assertEqual(b["tokens"][0], "docker")

    def test_custom_outside_root_error(self):
        launcher.SETTINGS["scan_dirs"] = [str(TMP / "custom-scan")]
        b = launcher.build(cfg(model_id="__custom__",
                               custom_path="/etc/hostname", engine="vllm"))
        self.assertIn("error", b)


class TestUsageAndPersist(LauncherStateCase):
    def test_record_usage_deltas(self):
        e = {"id": "r1", "model": "m", "engine": "llamacpp", "port": 8000,
             "started": "2026-01-01 00:00:00",
             "tokens_in": 100, "sess_in": 40,
             "tokens_out": 50, "sess_out": 10,
             "requests": 7, "sess_reqs": 2}
        launcher.record_usage(e, "stopped by user")
        data = json.loads(launcher.USAGE_PATH.read_text())
        self.assertEqual(len(data["sessions"]), 1)
        rec = data["sessions"][0]
        self.assertEqual((rec["tokens_in"], rec["tokens_out"], rec["requests"]),
                         (60, 40, 5))
        self.assertEqual(rec["status"], "stopped by user")
        # sess_* seeds advanced on the SAME entry object: a second record of
        # unchanged counters must write nothing (no double counting)
        launcher.record_usage(e, "stopped by user")
        data = json.loads(launcher.USAGE_PATH.read_text())
        self.assertEqual(len(data["sessions"]), 1)

    def test_usage_summary_totals(self):
        launcher._usage_save({"sessions": [
            {"tokens_in": 5, "tokens_out": 6, "requests": 1},
            {"tokens_in": 7, "tokens_out": 8, "requests": 2}]})
        s = launcher.usage_summary()
        self.assertEqual(s["totals"]["tokens_in"], 12)
        self.assertEqual(s["totals"]["tokens_out"], 14)
        self.assertEqual(s["totals"]["requests"], 3)
        self.assertEqual(s["totals"]["sessions"], 2)

    def test_usage_load_corrupt(self):
        launcher.USAGE_PATH.write_text("{nope")
        self.assertEqual(launcher._usage_load(), {"sessions": []})

    def test_persist_state_filters(self):
        launcher.RUNNING["run1"] = {
            "id": "run1", "status": "running", "model": "m", "model_id": "m1",
            "engine": "llamacpp", "cfg": {"x": 1}, "cname": "c", "endpoint": "e",
            "port": 8000, "native": True, "log": "l", "started": "s",
            "cmd": "c", "artifact_mib": 1, "pid": 1,
            "tokens_in": 1, "tokens_out": 2, "requests": 3,
            "sess_in": 0, "sess_out": 0, "sess_reqs": 0,
            "proc": "not-serializable-anyway"}  # extra keys must be filtered
        launcher.RUNNING["dry1"] = {"id": "dry1", "status": "dry-run"}
        launcher.RUNNING["stopped1"] = {"id": "stopped1", "status": "stopped"}
        launcher.persist_state()
        snap = json.loads(launcher.STATE_PATH.read_text())
        self.assertIn("run1", snap)
        self.assertNotIn("dry1", snap)
        self.assertNotIn("stopped1", snap)
        self.assertNotIn("proc", snap["run1"])
        self.assertEqual(snap["run1"]["tokens_out"], 2)


class TestHarnessLine(LauncherStateCase):
    def _built(self, port):
        return {"endpoint": f"http://127.0.0.1:{port}/v1",
                "model_name": "Qwen3.6-35B-A3B", "engine": "llamacpp"}

    def test_omp_line(self):
        c = cfg(model_id="qwen36-35b", engine="llamacpp", harness="omp")
        line = launcher.harness_line(c, self._built(c["port"]))
        self.assertIn(f"OPENAI_BASE_URL=http://127.0.0.1:{c['port']}/v1", line)
        self.assertIn("OPENAI_API_KEY=local", line)
        self.assertIn("omp --model b70-vllm/", line)
        self.assertIn("--no-tools", line)  # llamacpp engine gets no-tools

    def test_openvino_keeps_tools(self):
        c = cfg(model_id="qwen36-35b", engine="openvino", harness="pi")
        line = launcher.harness_line(c, {"endpoint": "http://127.0.0.1:9/v1",
                                         "model_name": "X", "engine": "openvino"})
        self.assertIn("pi --model b70-vllm/X", line)
        self.assertNotIn("--no-tools", line)

    def test_droid_and_webui_and_custom(self):
        c = cfg(engine="vllm", harness="droid")
        self.assertIn("droid --model custom:Desktop-B70-Loaded-Model-0",
                      launcher.harness_line(c, self._built(c["port"])))
        c2 = cfg(engine="vllm", harness="webui")
        self.assertEqual(launcher.harness_line(c2, self._built(c2["port"])),
                         "xdg-open http://localhost:3000")
        c3 = cfg(engine="vllm", harness="omp", harness_cmd="mytool --flag")
        self.assertTrue(launcher.harness_line(c3, self._built(c3["port"]))
                        .endswith("mytool --flag"))


class TestHarnessSync(LauncherStateCase):
    """sync_harness_configs writes the LAUNCHED model into the client configs —
    omp refuses unknown model ids, so a custom/remote model missing from
    models.yml would make 'Open in OMP' silently unusable."""

    def test_omp_gets_launched_model(self):
        ompdir = FAKE_HOME / ".omp" / "agent"
        ompdir.mkdir(parents=True, exist_ok=True)
        cfgf = ompdir / "models.yml"
        cfgf.write_text(
            "providers:\n"
            "  b70-vllm:\n"
            "    baseUrl: http://127.0.0.1:9999/v1\n"
            "    apiKey: local-b70\n"
            "    api: openai-completions\n"
            "    models:\n"
            "    - id: old-model\n"
            "      name: Old\n")
        launcher.sync_harness_configs(8000, "stub-model-27b", "stub-id", "vllm")
        text = cfgf.read_text()
        self.assertIn("baseUrl: http://127.0.0.1:8000/v1", text)
        self.assertIn("- id: stub-model-27b", text)
        self.assertIn("- id: stub-id", text)

    def test_droid_provider_is_chat_completions(self):
        dcfg = FAKE_HOME / ".factory" / "settings.json"
        dcfg.parent.mkdir(parents=True, exist_ok=True)
        dcfg.write_text("{}")
        launcher.sync_harness_configs(8000, "stub-model-27b", "x", "llamacpp")
        data = json.loads(dcfg.read_text())
        m = data["customModels"][0]
        # /v1/responses is vLLM-only; chat-completions works on every engine
        self.assertEqual(m["provider"], "generic-chat-completion-api")
        self.assertEqual(m["apiKey"], "local-b70")
        self.assertEqual(m["baseUrl"], "http://127.0.0.1:8000/v1")


class TestLoadPhases(LauncherStateCase):
    """Endpoint readiness drives status; the first ready probe stamps load_s,
    and a still-loading engine exposes a phase from its log tail."""

    def _entry(self, rid, port, log_text):
        logf = TMP / f"{rid}.log"
        logf.write_text(log_text)
        return {"id": rid, "status": "running", "model": "m",
                "model_id": "m1", "engine": "llamacpp",
                "cfg": {}, "cname": "", "endpoint": f"http://127.0.0.1:{port}/v1",
                "port": port, "native": True, "log": str(logf),
                "started": "s", "launch_ts": launcher.time.time() - 30,
                "ready_at": None, "proc": None, "pid": __import__("os").getpid(),
                "cmd": "c", "artifact_mib": 1,
                "tokens_in": 0, "tokens_out": 0, "requests": 0,
                "sess_in": 0, "sess_out": 0, "sess_reqs": 0}

    def test_loading_shows_phase_from_log(self):
        port = free_port()  # nothing listens -> probe fails -> still starting
        launcher.RUNNING["loadtest"] = self._entry(
            "loadtest", port,
            "llama_model_load: load_tensors: offloaded 62/62 layers\n"
            "main: Capturing CUDA graph for batch 512\n")
        items = {i["id"]: i for i in launcher.servers_snapshot()}
        self.assertEqual(items["loadtest"]["status"], "starting")
        self.assertEqual(items["loadtest"]["phase"], "graph capture")

    def test_ready_stamps_load_s(self):
        import http.server, threading
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
            def log_message(self, *a):
                pass
        port = free_port()
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        launcher.RUNNING["ready1"] = self._entry("ready1", port, "")
        items = {i["id"]: i for i in launcher.servers_snapshot()}
        e = launcher.RUNNING["ready1"]
        self.assertEqual(items["ready1"]["status"], "running")
        self.assertIsNotNone(e["ready_at"])
        self.assertGreaterEqual(items["ready1"]["load_s"], 29)


if __name__ == "__main__":
    unittest.main()
