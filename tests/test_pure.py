"""Pure-function unit tests for launcher.py — no GPU, docker, or network.

Run:  cd ~/b70-launcher && python3 -m unittest discover -s tests -v
"""
import json
import shlex
import unittest
from pathlib import Path

try:
    import support_env
except ImportError:  # running as tests.test_pure from repo root
    from tests import support_env

launcher = support_env.launcher
TMP = support_env.TMPROOT
FAKE_HOME = support_env.FAKE_HOME
LauncherStateCase = support_env.LauncherStateCase
write_file = support_env.write_file


class TestSafeArtifact(unittest.TestCase):
    """HF artifact names must stay relative — no traversal or drive letters."""

    def test_accepts_plain_and_nested(self):
        for ok in ("model.gguf", "dir/model.gguf", "a/b/c.bin",
                   "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "AP-Q4_K_XL/f.gguf"):
            self.assertTrue(launcher.safe_artifact(ok), ok)

    def test_rejects_traversal_and_absolute(self):
        for bad in ("", "/abs.gguf", "../x.gguf", "a/../b.gguf", "a//b.gguf",
                    ".", "..", "a/./b.gguf", "/a/b", "a\\b.gguf", "C:\\m.gguf",
                    "C:/m.gguf", "x:y.gguf"):
            self.assertFalse(launcher.safe_artifact(bad), bad)


class TestTextHelpers(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(launcher._norm("Qwen3.6-35B-A3B!"), "qwen3635ba3b")
        self.assertEqual(launcher._norm(""), "")
        self.assertEqual(launcher._norm(None), "")

    def test_as_list(self):
        self.assertEqual(launcher._as_list("a"), ["a"])
        self.assertEqual(launcher._as_list(["a", "", None]), ["a"])
        self.assertEqual(launcher._as_list(("x", "y")), ["x", "y"])
        self.assertEqual(launcher._as_list(None), [])
        self.assertEqual(launcher._as_list(0), [])
        self.assertEqual(launcher._as_list(7), ["7"])

    def test_first_nonempty(self):
        self.assertEqual(launcher._first_nonempty(None, "", "x", "y"), "x")
        self.assertEqual(launcher._first_nonempty(None, ""), "")

    def test_fmt_bytes(self):
        self.assertEqual(launcher.fmt_bytes(None), "?")
        self.assertEqual(launcher.fmt_bytes(500), "500.0 B")
        self.assertEqual(launcher.fmt_bytes(2048), "2.0 KB")
        self.assertEqual(launcher.fmt_bytes(5 << 20), "5.0 MB")
        self.assertEqual(launcher.fmt_bytes(3 << 30), "3.0 GB")
        self.assertEqual(launcher.fmt_bytes(2 << 40), "2.0 TB")


class TestVerKey(unittest.TestCase):
    """recipe_ver strings compare as numeric tuples, not lexically."""

    def test_parses_dates_and_ints(self):
        self.assertEqual(launcher._ver_key("2026-10-05"), [2026, 10, 5])
        self.assertEqual(launcher._ver_key(20261005), [20261005])
        self.assertEqual(launcher._ver_key("v1.2.3"), [1, 2, 3])
        self.assertEqual(launcher._ver_key(""), [])
        self.assertEqual(launcher._ver_key(None), [])

    def test_numeric_not_lexical(self):
        # "9" > "10" lexically, but 2026-9 must sort before 2026-10
        self.assertLess(launcher._ver_key("2026-9"), launcher._ver_key("2026-10"))
        self.assertLess(launcher._ver_key("2026-10-05"),
                        launcher._ver_key("2026-10-06"))


class TestStampVer(unittest.TestCase):
    def test_stamps_missing_ver(self):
        r = {}
        launcher._stamp_ver(r, "2027-01-01")
        self.assertEqual(r["recipe_ver"], "2027-01-01")

    def test_keeps_existing_ver(self):
        r = {"recipe_ver": "2020-01-01"}
        launcher._stamp_ver(r, "2027-01-01")
        self.assertEqual(r["recipe_ver"], "2020-01-01")

    def test_no_fallback_no_stamp(self):
        r = {}
        launcher._stamp_ver(r, "")
        self.assertNotIn("recipe_ver", r)


class TestPretty(unittest.TestCase):
    def test_wraps_and_quotes(self):
        toks = ["docker", "run", "--name", "a name", "-m", "/models/x.gguf",
                "--flag", "v" * 120]
        out = launcher.pretty(toks, width=60)
        self.assertIn("\\\n", out)  # line continuation used
        # continuation+indent is display formatting: strip it and tokens survive
        flat = out.replace(" \\\n", " ")
        self.assertEqual(shlex.split(flat), toks)

    def test_short_line_single(self):
        out = launcher.pretty(["a", "b"], width=96)
        self.assertEqual(out, "a b")


class TestReadCtxFromConfig(unittest.TestCase):
    _n = 0

    def _cfg(self, obj):
        type(self)._n += 1
        return write_file(TMP / "cfg" / f"c{type(self)._n}" / "config.json",
                          json.dumps(obj).encode())

    def test_top_level_keys(self):
        for key in ("max_position_embeddings", "context_length",
                    "max_seq_len", "n_ctx"):
            p = self._cfg({key: 131072})
            self.assertEqual(launcher.read_ctx_from_config(p), 131072, key)

    def test_nested_text_config_wins(self):
        p = self._cfg({"text_config": {"max_position_embeddings": 40960},
                       "max_position_embeddings": 8192})
        self.assertEqual(launcher.read_ctx_from_config(p), 40960)

    def test_rejects_small_and_nonint(self):
        self.assertIsNone(launcher.read_ctx_from_config(self._cfg({"n_ctx": 256})))
        self.assertIsNone(launcher.read_ctx_from_config(self._cfg({"n_ctx": "abc"})))
        self.assertIsNone(launcher.read_ctx_from_config(self._cfg({})))

    def test_missing_and_corrupt(self):
        self.assertIsNone(launcher.read_ctx_from_config(TMP / "nope" / "config.json"))
        bad = write_file(TMP / "badcfg" / "config.json", b"{not json")
        self.assertIsNone(launcher.read_ctx_from_config(bad))


class TestContainerPath(unittest.TestCase):
    def test_file_maps_parent_to_models(self):
        f = write_file(TMP / "cp" / "sub" / "m.gguf")
        cp, src, dst = launcher.container_path({"path": str(f)}, "/fb")
        self.assertEqual(cp, "/models/m.gguf")
        self.assertEqual(src, str(f.parent))
        self.assertEqual(dst, "/models")

    def test_dir_mounts_at_models_model(self):
        d = TMP / "cp" / "modeldir"
        d.mkdir(parents=True, exist_ok=True)
        cp, src, dst = launcher.container_path({"path": str(d)}, "/fb")
        self.assertEqual((cp, src, dst), ("/models/model", str(d), "/models/model"))

    def test_none_and_empty_fall_back(self):
        self.assertEqual(launcher.container_path(None, "/fb"), ("/fb", None, "/models"))
        self.assertEqual(launcher.container_path({}, "/fb"), ("/fb", None, "/models"))


class TestResolveRepo(unittest.TestCase):
    def test_approved_repo(self):
        repo, err = launcher.resolve_repo({"repo": "unsloth/Qwen3.8-27B-GGUF"})
        self.assertEqual(repo, "unsloth/Qwen3.8-27B-GGUF")
        self.assertIsNone(err)

    def test_unapproved_repo(self):
        repo, err = launcher.resolve_repo({"repo": "evil/repo"})
        self.assertIsNone(repo)
        self.assertIn("no approved", err)

    def test_revision_validation(self):
        repo, err = launcher.resolve_repo({"repo": "unsloth/Qwen3.8-27B-GGUF",
                                           "revision": "abc-123_X.Y"})
        self.assertIsNone(err)
        repo, err = launcher.resolve_repo({"repo": "unsloth/Qwen3.8-27B-GGUF",
                                           "revision": "bad rev/../x"})
        self.assertIsNone(repo)
        self.assertIn("invalid repository revision", err)


class TestPrometheusParsing(unittest.TestCase):
    SAMPLE = "\n".join([
        "# HELP comment",
        'llamacpp:prompt_tokens_total{model="a"} 10',
        'llamacpp:prompt_tokens_total{model="b"} 5',
        "llamacpp:tokens_predicted_total 42",
        "llamacpp:prompt_seconds_total 99",
        "vllm:prompt_tokens_by_source_total 7",
        "vllm:request_success_total 3",
        "not a metric line",
        "unlabeled_gauge -2.5e1",
        "",
    ])

    def test_parse_sums_label_series(self):
        c = launcher._parse_prom(self.SAMPLE)
        self.assertEqual(c["llamacpp:prompt_tokens_total"], 15.0)
        self.assertEqual(c["llamacpp:tokens_predicted_total"], 42.0)
        self.assertEqual(c["unlabeled_gauge"], -25.0)

    def test_classify_excludes_breakdowns(self):
        tin, tout, reqs = launcher._classify_tokens(launcher._parse_prom(self.SAMPLE))
        self.assertEqual(tin, 15.0)   # by_source variant excluded, seconds excluded
        self.assertEqual(tout, 42.0)
        self.assertEqual(reqs, 3.0)

    def test_pick_counter_prefers_shortest(self):
        c = {"x_prompt_tokens_total": 1.0, "y_input_tokens_total": 2.0}
        # both match the input pattern; shortest name wins deterministically
        v = launcher._pick_counter(c, r"prompt|input")
        self.assertEqual(v, 2.0)

    def test_pick_counter_no_match(self):
        self.assertEqual(launcher._pick_counter({"foo_total": 1.0}, r"token"), 0.0)


class TestScanRoots(LauncherStateCase):
    def test_filters_dangerous_and_missing_roots(self):
        real = TMP / "scanroot"
        real.mkdir()
        launcher.SETTINGS["scan_dirs"] = [
            str(real), str(real) + "/", str(real),  # dedup to one
            "/",                                    # filesystem root refused
            str(FAKE_HOME),                         # home dir refused
            str(TMP / "does-not-exist"),            # missing skipped
        ]
        roots = launcher.scan_roots()
        self.assertEqual(roots, [real.resolve()])

    def test_empty_scan_dirs(self):
        launcher.SETTINGS["scan_dirs"] = []
        self.assertEqual(launcher.scan_roots(), [])


class TestWalkRoot(unittest.TestCase):
    """Filesystem walker: GGUF collection, model-dir classification, pruning."""

    def setUp(self):
        self.root = TMP / "walkroot"
        self.root.mkdir(exist_ok=True)

    def _walk(self):
        items = {"gguf": {}, "snapshots": {}}
        done = launcher._walk_root(self.root, items, launcher.time.time() + 30)
        return done, items

    def test_gguf_and_skip_dirs(self):
        write_file(self.root / "a" / "Model-X.gguf")
        write_file(self.root / "a" / "mmproj-Model-X.gguf")  # collected as gguf entry
        write_file(self.root / ".git" / "hidden.gguf")       # skip dir
        write_file(self.root / ".hidden" / "h.gguf")         # dot dir
        write_file(self.root / "node_modules" / "n.gguf")    # skip dir
        done, items = self._walk()
        self.assertTrue(done)
        self.assertIn("model-x.gguf", items["gguf"])
        self.assertIn("mmproj-model-x.gguf", items["gguf"])
        self.assertNotIn("hidden.gguf", items["gguf"])
        self.assertNotIn("h.gguf", items["gguf"])
        self.assertNotIn("n.gguf", items["gguf"])

    def test_openvino_dir_classification(self):
        d = self.root / "ov-model"
        write_file(d / "openvino_language_model.xml")
        write_file(d / "openvino_language_model.bin")
        write_file(d / "config.json", json.dumps({"max_position_embeddings": 65536}).encode())
        done, items = self._walk()
        self.assertTrue(done)
        snap = items["snapshots"]["ov-model"]
        self.assertEqual(snap["kind"], "openvino")
        self.assertEqual(snap["ctx"], 65536)

    def test_vllm_and_exl3_classification(self):
        v = self.root / "hf-model"
        write_file(v / "config.json", b"{}")
        write_file(v / "model.safetensors")
        e = self.root / "exl3-model"
        write_file(e / "config.json", b"{}")
        write_file(e / "quantization_config.json",
                   json.dumps({"quant_method": "exl3"}).encode())
        done, items = self._walk()
        self.assertTrue(done)
        self.assertEqual(items["snapshots"]["hf-model"]["kind"], "vllm")
        self.assertEqual(items["snapshots"]["exl3-model"]["kind"], "exl3")

    def test_model_dir_not_descended(self):
        d = self.root / "ov2"
        write_file(d / "openvino_language_model.xml")
        write_file(d / "openvino_language_model.bin")
        write_file(d / "nested" / "inner.gguf")  # inside model dir: must NOT be collected
        done, items = self._walk()
        self.assertTrue(done)
        self.assertNotIn("inner.gguf", items["gguf"])

    def test_depth_limit(self):
        deep = self.root
        for i in range(10):
            deep = deep / f"d{i}"
        write_file(deep / "too-deep.gguf")
        done, items = self._walk()
        self.assertTrue(done)
        self.assertNotIn("too-deep.gguf", items["gguf"])

    def test_deadline_returns_false(self):
        items = {"gguf": {}, "snapshots": {}}
        done = launcher._walk_root(self.root, items, launcher.time.time() - 1)
        self.assertFalse(done)


class TestCatalogAndSize(unittest.TestCase):
    def test_dir_size_and_catalog(self):
        d = TMP / "cat" / "mdir"
        write_file(d / "a.bin", b"\0" * 1048576)
        write_file(d / "b.bin", b"\0" * 1048576)
        self.assertEqual(launcher._dir_size_mib(str(d)), 2.0)
        self.assertIsNone(launcher._dir_size_mib(str(TMP / "missing-dir")))

        f = write_file(TMP / "cat" / "one.gguf", b"\0" * 1024)
        items = {"gguf": {"one.gguf": str(f)},
                 "snapshots": {"mdir": {"path": str(d), "kind": "vllm", "ctx": 4096}}}
        cat = launcher.build_catalog(items)
        self.assertEqual(len(cat), 2)
        self.assertEqual(cat[0]["kind"], "gguf")  # sorted by (kind, name)
        self.assertEqual(cat[1]["kind"], "vllm")
        self.assertEqual(cat[1]["size_mib"], 2.0)
        self.assertEqual(cat[1]["ctx"], 4096)


class TestExl3Roots(LauncherStateCase):
    def test_expanduser_and_suffixes(self):
        launcher.SETTINGS["exl3_data_root"] = "~/.local/share/b70-exl3"
        ctr, dk = launcher._exl3_roots()
        self.assertEqual(ctr, str(FAKE_HOME / ".local/share/b70-exl3-containerd"))
        self.assertEqual(dk, str(FAKE_HOME / ".local/share/b70-exl3-docker"))
        self.assertNotIn("~", ctr)

    def test_trailing_slash_stripped(self):
        launcher.SETTINGS["exl3_data_root"] = "/tmp/exl3/"
        ctr, dk = launcher._exl3_roots()
        self.assertEqual((ctr, dk), ("/tmp/exl3-containerd", "/tmp/exl3-docker"))


class TestPidForNative(unittest.TestCase):
    def test_free_port_returns_none(self):
        # scans /proc for --port <p>; a freshly freed port has no owner
        self.assertIsNone(launcher._pid_for_native(support_env.free_port()))

    def test_pid_alive(self):
        self.assertTrue(launcher._pid_alive(__import__("os").getpid()))
        self.assertFalse(launcher._pid_alive(2 ** 22))
        self.assertFalse(launcher._pid_alive("not-a-pid"))


if __name__ == "__main__":
    unittest.main()
