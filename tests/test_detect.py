"""Tests for artifact detection: detect(), prepare_custom(), resolve_ctx(),
_resolve_draft() — all against temp dirs and seeded SCAN state, no GPU/network.

Run:  cd b70-launcher && python3 -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

try:
    import support_env
except ImportError:
    from tests import support_env

import b70launcher.catalog as catalog, b70launcher.scan as scan, b70launcher.hardware as hardware, b70launcher.downloads as downloads, b70launcher.builders as builders, b70launcher.builders.common as common, b70launcher.runtime as runtime, b70launcher.telemetry as telemetry, b70launcher.harness as harness, b70launcher.http_api as http_api, b70launcher.paths as paths, b70launcher.exl3_stack as exl3_stack, b70launcher.locks as locks
TMP = support_env.TMPROOT
FAKE_HOME = support_env.FAKE_HOME
LauncherStateCase = support_env.LauncherStateCase
write_file = support_env.write_file

DISK = TMP / "detect-disk"


def model(mid):
    m = catalog.find_model(mid)
    assert m is not None, mid
    return m


class TestDetectGGUF(LauncherStateCase):
    """GGUF pool: exact name first, then same-family variant matching."""

    def setUp(self):
        super().setUp()
        self.dir = DISK / "gguf"
        self.dir.mkdir(parents=True, exist_ok=True)
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}

    def _add(self, *names):
        for n in names:
            p = write_file(self.dir / n)
            scan.SCAN["items"]["gguf"][n.lower()] = str(p)

    def test_exact_name_beats_family(self):
        self._add("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "Other-Qwen3.6-35B-Q8_0.gguf")
        det = scan.detect(model("qwen36-35b"), "llamacpp")
        self.assertFalse(det["variant"])
        self.assertEqual(Path(det["path"]).name, "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
        self.assertEqual(det["mount_root"], str(self.dir))
        self.assertEqual(det["size_mib"], 0.0)

    def test_family_variant_flagged(self):
        # same family, wrong quant -> detected but flagged variant
        self._add("My-Qwen3.6-35B-A3B-Q8_0.gguf")
        det = scan.detect(model("qwen36-35b"), "llamacpp")
        self.assertTrue(det["variant"])
        self.assertEqual(det["found_name"], "my-qwen3.6-35b-a3b-q8_0.gguf")

    def test_dl_name_subdirectory_basename_match(self):
        # recipe name carries a repo subdir ("AP-Q4_K_XL/file.gguf");
        # the scan pool is keyed by basename only
        self._add("signal-3.8-flash-next-ap-q4_k_xl.gguf")
        det = scan.detect(model("qwen38-flashnext"), "llamacpp")
        self.assertIsNotNone(det)
        self.assertFalse(det["variant"])

    def test_prefer_filters_wrong_quant(self):
        self._add("Qwen3.8-27B-UD-Q8_0.gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf")
        det = scan.detect(model("qwen38-27b"), "llamacpp")
        self.assertEqual(Path(det["path"]).name, "Qwen3.8-27B-UD-Q4_K_M.gguf")

    def test_mmproj_sidecar_never_matches(self):
        self._add("mmproj-Muse-Glimmer-30B-Q4_K_M.gguf")
        self.assertIsNone(scan.detect(model("muse-glimmer"), "llamacpp"))

    def test_draft_sidecar_never_satisfies_model(self):
        self._add("dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
                  "draft-muse-glimmer-30b.gguf")
        self.assertIsNone(scan.detect(model("muse-glimmer"), "llamacpp"))

    def test_no_candidates(self):
        self._add("completely-unrelated.gguf")
        self.assertIsNone(scan.detect(model("qwen36-35b"), "llamacpp"))

    def test_missing_model_and_engine(self):
        self.assertIsNone(scan.detect(None, "llamacpp"))
        self.assertIsNone(scan.detect(model("qwen36-35b"), "not-an-engine"))

    def test_empty_pool(self):
        self.assertIsNone(scan.detect(model("ornith-35b-gguf"), "llamacpp"))


class TestDetectSnapshots(LauncherStateCase):
    """Snapshot pool: engine-kind filtering + prefer + ctx passthrough."""

    def setUp(self):
        super().setUp()
        self.dir = DISK / "snap"
        self.dir.mkdir(parents=True, exist_ok=True)
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}

    def _snap(self, name, kind, ctx=None):
        d = self.dir / name
        d.mkdir(exist_ok=True)
        ent = {"path": str(d), "kind": kind, "repo_hint": "x/" + name, "ctx": ctx}
        scan.SCAN["items"]["snapshots"][name.lower()] = ent
        return d

    def test_engine_kind_filter(self):
        ov = self._snap("Qwen3.6-35B-A3B-int4-ov", "openvino")
        vl = self._snap("Qwen3.6-35B-A3B-GPTQ-Int4", "vllm", ctx=131072)
        # vllm recipe prefers "gptq" and must never see the openvino dir
        det = scan.detect(model("qwen36-35b"), "vllm")
        self.assertEqual(det["path"], str(vl))
        self.assertEqual(det["ctx"], 131072)
        # openvino recipe of qwen38-27b must never see a vllm dir
        self._snap("Qwen3.8-27B-int4-gdn8-ov", "openvino")
        det2 = scan.detect(model("qwen38-27b"), "openvino")
        self.assertEqual(Path(det2["path"]).name, "Qwen3.8-27B-int4-gdn8-ov")

    def test_wrong_kind_pool_excluded(self):
        self._snap("Qwen3.6-35B-A3B-int4-ov", "openvino")
        self.assertIsNone(scan.detect(model("qwen36-35b"), "vllm"))

    def test_dir_detection_and_ctx(self):
        d = self._snap("Nemotron-3.5-Lightning-GPTQ-Int4", "vllm", ctx=262144)
        det = scan.detect(model("nemotron-35"), "vllm")
        self.assertEqual(det["path"], str(d))
        self.assertEqual(det["ctx"], 262144)
        self.assertEqual(det["mount_root"], str(d))

    def test_sibling_branch_dirs_stay_with_their_recipe(self):
        # plain 4.00bpw and SC_5.00bpw_H6 are branches of one HF repo; each
        # recipe must only ever see its own directory (the stock image cannot
        # serve the mixed-K branch)
        sc5 = self._snap("Qwen3.8-27B-exl3-SC_5.00bpw_H6", "exl3")
        self.assertIsNone(scan.detect(model("qwen38-27b"), "exl3"))
        self.assertEqual(scan.detect(model("qwen38-27b-sc5"), "exl3")["path"], str(sc5))
        plain = self._snap("Qwen3.8-27B-exl3", "exl3")
        self.assertEqual(scan.detect(model("qwen38-27b"), "exl3")["path"], str(plain))
        self.assertEqual(scan.detect(model("qwen38-27b-sc5"), "exl3")["path"], str(sc5))


class TestSnapshotDownloadDest(LauncherStateCase):
    """download.dest picks the snapshot directory; default is the repo basename."""

    def _run(self, mid, dest_override=None):
        from unittest import mock
        m = json.loads(json.dumps(model(mid)))
        if dest_override is not None:
            m["recipes"]["exl3"]["download"]["dest"] = dest_override
        did = f"{mid}-exl3"
        downloads.DOWNLOADS[did] = {"id": did, "state": "queued", "done": 0}
        root = TMP / "dl-models"
        fetched = []
        with mock.patch.dict(catalog.SETTINGS, {"models_dir": str(root)}), \
             mock.patch.object(downloads, "resolve_repo", return_value=("turboderp/Qwen3.8-27B-exl3", None)), \
             mock.patch.object(downloads, "hf_files", return_value={"config.json": 0}), \
             mock.patch.object(downloads, "_download_file", side_effect=lambda url, dest, e: (fetched.append((url, dest)), write_file(dest))), \
             mock.patch.object(scan, "start_scan_async"):
            downloads.download_worker(did, m, "exl3")
        return downloads.DOWNLOADS[did], fetched, root

    def test_dest_names_the_directory_and_revision_is_pinned(self):
        entry, fetched, root = self._run("qwen38-27b-sc5")
        self.assertEqual(entry["state"], "done", entry.get("error"))
        self.assertEqual(entry["dest"], str(root / "Qwen3.8-27B-exl3-SC_5.00bpw_H6"))
        self.assertIn("/resolve/dfe2f4fd71bf9a1994051d4d47e6460fa96dd723/", fetched[0][0])
        plain, _, _ = self._run("qwen38-27b")
        self.assertEqual(plain["dest"], str(root / "Qwen3.8-27B-exl3"))

    def test_dest_rejects_paths(self):
        for bad in ("../escape", "a/b", ".."):
            entry, fetched, _ = self._run("qwen38-27b-sc5", dest_override=bad)
            self.assertEqual(entry["state"], "error", bad)
            self.assertEqual(fetched, [])


class TestDetectDirectPath(LauncherStateCase):
    """Recipe fields gguf/model_path/source_model are probed directly,
    with ~ expansion — ahead of any scan result."""

    def test_model_path_expanduser(self):
        target = FAKE_HOME / "models" / "Qwen3.8-27B-exl3"
        target.mkdir(parents=True)
        write_file(target / "model.safetensors")
        self.addCleanup(lambda: __import__("shutil").rmtree(
            FAKE_HOME / "models", ignore_errors=True))
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}
        det = scan.detect(model("qwen38-27b"), "exl3")
        self.assertIsNotNone(det)
        self.assertEqual(det["path"], str(target))
        self.assertFalse(det["variant"])

    def test_missing_direct_path_falls_through(self):
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}
        # ~/models/... does not exist under the fake HOME -> None, not a guess
        self.assertIsNone(scan.detect(model("qwen38-27b"), "vllm"))


class TestResolveCtx(LauncherStateCase):
    def test_recipe_default(self):
        out = scan.resolve_ctx(model("qwen36-35b"), "llamacpp", None)
        self.assertEqual(out["value"], 131072)
        self.assertEqual(out["source"], "128K")  # ctx_note

    def test_disk_config_wins(self):
        out = scan.resolve_ctx(model("qwen36-35b"), "llamacpp",
                                   {"ctx": 65536})
        self.assertEqual(out["value"], 65536)
        self.assertEqual(out["source"], "config.json on disk")

    def test_ctx_max_clamps_disk_value(self):
        out = scan.resolve_ctx(model("qwen38-27b"), "llamacpp",
                                   {"ctx": 999999})
        self.assertEqual(out["value"], 262144)  # ctx_max

    def test_ctx_max_clamps_recipe(self):
        # recipe ctx 262144 == ctx_max; pick a model where recipe ctx > a
        # hypothetical cap is not shipped, so verify cap on a patched recipe
        m = model("qwen36-35b")
        m["recipes"]["llamacpp"]["ctx_max"] = 100000
        out = scan.resolve_ctx(m, "llamacpp", None)
        self.assertEqual(out["value"], 100000)


class TestResolveDraft(LauncherStateCase):
    def test_empty_and_missing(self):
        self.assertIsNone(scan._resolve_draft(None))
        self.assertIsNone(scan._resolve_draft(""))
        self.assertIsNone(scan._resolve_draft("/abs/missing-draft.gguf"))
        self.assertIsNone(scan._resolve_draft("rel/dir/draft.gguf"))
        scan.SCAN["items"] = {"gguf": {}, "snapshots": {}}
        self.assertIsNone(scan._resolve_draft("not-on-disk-zz.gguf"))

    def test_expanduser_existing(self):
        f = write_file(FAKE_HOME / "drafts" / "d.gguf")
        self.assertEqual(scan._resolve_draft("~/drafts/d.gguf"), str(f))

    def test_relative_existing_path(self):
        # a bare existing filename resolves as a host path
        f = write_file(TMP / "cwd-check" / "x.gguf")
        self.assertEqual(scan._resolve_draft(str(f)), str(f))

    def test_scan_lookup_bare_name(self):
        f = write_file(DISK / "drafts" / "DFlash-Draft.gguf")
        scan.SCAN["items"] = {
            "gguf": {"dflash-draft.gguf": str(f)}, "snapshots": {}}
        self.assertEqual(scan._resolve_draft("DFlash-Draft.gguf"), str(f))

    def test_scan_lookup_dict_entry(self):
        f = write_file(DISK / "drafts" / "d2.gguf")
        scan.SCAN["items"] = {
            "gguf": {"d2.gguf": {"path": str(f)}}, "snapshots": {}}
        self.assertEqual(scan._resolve_draft("d2.gguf"), str(f))


class TestPrepareCustom(LauncherStateCase):
    """Confinement to scan roots, engine/kind agreement, format sniffing."""

    def setUp(self):
        super().setUp()
        self.root = TMP / "custom-root"
        self.root.mkdir(exist_ok=True)
        catalog.SETTINGS["scan_dirs"] = [str(self.root)]

    def test_missing_path_and_bad_engine(self):
        with self.assertRaises(common.BuildError):
            builders.prepare_custom({})
        with self.assertRaises(common.BuildError):
            builders.prepare_custom({"custom_path": str(self.root), "engine": "bogus"})

    def test_outside_scan_root_rejected(self):
        with self.assertRaises(common.BuildError):
            builders.prepare_custom({"custom_path": "/etc/hostname", "engine": "vllm"})

    def test_missing_artifact_rejected(self):
        with self.assertRaises(common.BuildError):
            builders.prepare_custom({"custom_path": str(self.root / "no.gguf"), "engine": "llamacpp"})

    def test_gguf_file_ok_for_llamacpp(self):
        f = write_file(self.root / "mine.gguf")
        model_, eng, det, warns = builders.prepare_custom(
            {"custom_path": str(f), "engine": "llamacpp"})
        self.assertEqual(model_["id"], "custom")
        self.assertEqual(eng, "llamacpp")
        self.assertEqual(det["path"], str(f))
        self.assertEqual(det["mount_root"], str(self.root))
        self.assertTrue(warns)  # generic-template warning always present

    def test_wrong_engine_rejected(self):
        f = write_file(self.root / "m2.gguf")
        with self.assertRaises(common.BuildError):
            builders.prepare_custom({"custom_path": str(f), "engine": "vllm"})

    def test_openvino_dir_sniffing(self):
        d = self.root / "ovm"
        write_file(d / "openvino_language_model.xml")
        write_file(d / "openvino_language_model.bin")
        r = builders.prepare_custom({"custom_path": str(d), "engine": "openvino"})
        self.assertEqual(r[1], "openvino")
        with self.assertRaises(common.BuildError):
            builders.prepare_custom({"custom_path": str(d), "engine": "vllm"})

    def test_xml_bin_pair_counts_as_openvino(self):
        d = self.root / "ovm2"
        write_file(d / "model.xml")
        write_file(d / "model.bin")
        r = builders.prepare_custom({"custom_path": str(d), "engine": "openvino"})
        self.assertEqual(r[1], "openvino")

    def test_safetensors_dir_is_vllm(self):
        d = self.root / "hfm"
        write_file(d / "config.json",
                   json.dumps({"max_position_embeddings": 40960}).encode())
        write_file(d / "model.safetensors")
        model_, eng, det, warns = builders.prepare_custom(
            {"custom_path": str(d), "engine": "vllm"})
        self.assertEqual(eng, "vllm")
        self.assertEqual(det["ctx"], 40960)  # ctx sniffed from config.json
        self.assertEqual(model_["recipes"]["vllm"]["ctx"], 40960)

    def test_exl3_dir_via_quantization_config(self):
        d = self.root / "exl3m"
        write_file(d / "config.json", b"{}")
        write_file(d / "quantization_config.json",
                   json.dumps({"quant_method": "exl3"}).encode())
        r = builders.prepare_custom({"custom_path": str(d), "engine": "exl3"})
        self.assertEqual(r[1], "exl3")
        with self.assertRaises(common.BuildError):
            builders.prepare_custom({"custom_path": str(d), "engine": "vllm"})

    def test_unknown_file_type_rejected(self):
        f = write_file(self.root / "readme.txt")
        with self.assertRaises(common.BuildError):
            builders.prepare_custom({"custom_path": str(f), "engine": "vllm"})

    def test_tilde_expansion(self):
        mdir = FAKE_HOME / "models"
        f = write_file(mdir / "sub" / "tilde.gguf")
        catalog.SETTINGS["scan_dirs"] = [str(mdir)]
        r = builders.prepare_custom({"custom_path": "~/models/sub/tilde.gguf",
                                     "engine": "llamacpp"})
        self.assertEqual(r[1], "llamacpp")
        self.assertEqual(r[2]["path"], str(f.resolve()))


if __name__ == "__main__":
    unittest.main()
