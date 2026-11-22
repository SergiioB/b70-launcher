"""Issue #8: AIB Arc Pro B70 detection + KDE/desktop GPU-busy false positives.

Covers hardware_preflight(sysfs_root) device detection on board-partner
subsystem IDs and gpu_clients(proc_root) desktop-client filtering on a fake
/proc tree — no real GPU or /proc state is touched.
"""
import unittest
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import hardware

TMP = support_env.TMPROOT
LauncherStateCase = support_env.LauncherStateCase
write_file = support_env.write_file


def fake_sysfs_drm(root, pci="0000:0a:00.0", render="renderD128", card="card0",
                   vendor="0x8086", device="0xe223",
                   subsystem_vendor="0x1849", subsystem_device="0x6025",
                   vram_total=None):
    """Minimal sysfs tree: class/drm/{card,renderD*} -> device with PCI ids."""
    root = Path(root)
    devdir = root / "devices" / "pci0000:00" / pci
    for name, val in {"vendor": vendor, "device": device,
                      "subsystem_vendor": subsystem_vendor,
                      "subsystem_device": subsystem_device}.items():
        write_file(devdir / name, (val + "\n").encode())
    if vram_total is not None:
        write_file(devdir / "mem_info_vram_total", str(vram_total).encode())
    for node in (card, render):
        nodedir = devdir / "drm" / node
        nodedir.mkdir(parents=True)
        (nodedir / "device").symlink_to(Path("..") / "..")
        (root / "class" / "drm").mkdir(parents=True, exist_ok=True)
        (root / "class" / "drm" / node).symlink_to(nodedir)
    return root


def fake_proc(root, pid, exe, fds, cmdline=None, cgroup="0::/fake.slice\n"):
    """One fake /proc/<pid>: exe symlink, fd symlinks + fdinfo files."""
    proc = Path(root) / str(pid)
    (proc / "fd").mkdir(parents=True)
    (proc / "fdinfo").mkdir()
    (proc / "exe").symlink_to("/nonexistent/bin/" + exe)
    (proc / "cmdline").write_bytes(
        cmdline if cmdline is not None else exe.encode() + b"\0")
    (proc / "cgroup").write_text(cgroup)
    for num, target, fdinfo in fds:
        (proc / "fd" / str(num)).symlink_to(target)
        (proc / "fdinfo" / str(num)).write_text(fdinfo)
    return proc


def fdinfo(cid="", vram_total_kib=None, vram_res_kib=None):
    lines = ["pos:\t0", "flags:\t02100002", "mnt_id:\t29", "ino:\t100"]
    if cid:
        lines.append(f"drm-client-id:\t{cid}")
    lines.append("drm-pdev:\t0000:0a:00.0")
    if vram_total_kib is not None:
        lines.append(f"drm-total-vram0:\t{vram_total_kib}")
    if vram_res_kib is not None:
        lines.append(f"drm-resident-vram0:\t{vram_res_kib}")
    return "\n".join(lines) + "\n"


RENDER_FD = lambda cid="", **kw: (3, "/dev/dri/renderD128", fdinfo(cid, **kw))


class TestB70Detection(LauncherStateCase):
    def _preflight(self, root):
        with mock.patch("os.access", return_value=True):
            return hardware.hardware_preflight(sysfs_root=str(root))

    def test_asrock_aib_b70_is_detected(self):
        root = fake_sysfs_drm(TMP / "sys-aib")
        out = self._preflight(root)
        dev = out["devices"][0]
        self.assertTrue(dev["b70"])
        self.assertEqual(dev["label"], "Arc Pro B70 · 0a:00.0")
        self.assertEqual(dev["subsystem"], "0x1849:0x6025")
        self.assertEqual(out["profile"]["b70_count"], 1)
        self.assertNotIn("subsystem ID 8086:1701", " ".join(out["blockers"]))

    def test_intel_reference_subsystem_still_detected(self):
        root = fake_sysfs_drm(TMP / "sys-ref", subsystem_vendor="0x8086",
                              subsystem_device="0x1701")
        dev = self._preflight(root)["devices"][0]
        self.assertTrue(dev["b70"])

    def test_32gib_vram_fallback_for_aib_card(self):
        """xe does not expose mem_info_vram_total on this card; the B70
        fallback must apply to every 8086:e223 board, not only 8086:1701."""
        root = fake_sysfs_drm(TMP / "sys-novram")
        dev = self._preflight(root)["devices"][0]
        self.assertEqual(dev["vram_total_mib"], 32656.0)

    def test_real_vram_sysfs_value_wins_over_fallback(self):
        root = fake_sysfs_drm(TMP / "sys-vram", vram_total=34_359_738_368)
        dev = self._preflight(root)["devices"][0]
        self.assertEqual(dev["vram_total_mib"], 32768.0)

    def test_other_intel_gpu_is_not_b70(self):
        root = fake_sysfs_drm(TMP / "sys-other", device="0xe202",
                              subsystem_vendor="0x8086", subsystem_device="0x1701")
        out = self._preflight(root)
        self.assertFalse(out["devices"][0]["b70"])
        self.assertIn("8086:e223", " ".join(out["blockers"]))

    def test_non_intel_gpu_is_not_b70(self):
        root = fake_sysfs_drm(TMP / "sys-amd", vendor="0x1002", device="0x744c")
        self.assertFalse(self._preflight(root)["devices"][0]["b70"])

    def test_render_node_maps_to_drm_card(self):
        root = fake_sysfs_drm(TMP / "sys-card")
        dev = self._preflight(root)["devices"][0]
        self.assertEqual(dev["card"], "card0")
        self.assertEqual(dev["render"], "/dev/dri/renderD128")


class TestGpuClients(LauncherStateCase):
    def _clients(self, proc_root):
        return hardware.gpu_clients(proc_root=str(proc_root))

    def test_kde_and_desktop_processes_do_not_count(self):
        root = TMP / "proc-kde"
        for pid, exe in enumerate(
                ("plasmashell", "kwin_wayland", "kded6", "krunner", "ksmserver",
                 "kactivitymanagerd", "kdeconnectd", "konsole", "dolphin",
                 "spectacle", "xdg-desktop-portal", "zen", "librewolf",
                 "telegram-desktop", "claude", "cowork-linux-helper",
                 "disclaimer", "plasma-keyboard"), start=100):
            fake_proc(root, pid, exe, [RENDER_FD("9")])
        fake_proc(root, 999, "real-engine",
                  [RENDER_FD("3", vram_total_kib=8 * 1024 * 1024)])
        out = self._clients(root)
        self.assertEqual({p["exe"] for p in out}, {"real-engine"})

    def test_small_vram_client_is_quiet_but_large_is_busy(self):
        root = TMP / "proc-vram"
        fake_proc(root, 100, "tiny-compute",
                  [RENDER_FD("5", vram_total_kib=200 * 1024)])   # 200 MiB
        fake_proc(root, 200, "vllm-engine",
                  [RENDER_FD("7", vram_total_kib=16 * 1024 * 1024)])  # 16 GiB
        fake_proc(root, 300, "resident-hog",
                  [RENDER_FD("8", vram_total_kib=100 * 1024,
                             vram_res_kib=4 * 1024 * 1024)])    # resident 4 GiB
        out = self._clients(root)
        self.assertEqual({p["exe"] for p in out}, {"vllm-engine", "resident-hog"})

    def test_vram_threshold_boundary(self):
        root = TMP / "proc-boundary"
        fake_proc(root, 100, "just-under",
                  [RENDER_FD("1", vram_total_kib=hardware.CLIENT_VRAM_QUIET_KIB - 1)])
        fake_proc(root, 200, "at-threshold",
                  [RENDER_FD("2", vram_total_kib=hardware.CLIENT_VRAM_QUIET_KIB)])
        out = self._clients(root)
        self.assertEqual({p["exe"] for p in out}, {"at-threshold"})

    def test_missing_vram_fields_stay_conservative(self):
        """No drm-*-vram* fdinfo: an unknown process still marks the card busy."""
        root = TMP / "proc-novram"
        fake_proc(root, 100, "mystery-engine", [RENDER_FD("4")])
        out = self._clients(root)
        self.assertEqual({p["exe"] for p in out}, {"mystery-engine"})

    def test_child_sharing_desktop_client_id_is_filtered(self):
        root = TMP / "proc-share"
        fake_proc(root, 100, "plasmashell", [RENDER_FD("42")])
        fake_proc(root, 200, "helper-child", [RENDER_FD("42")])
        fake_proc(root, 300, "helper-alone",
                  [RENDER_FD("77", vram_total_kib=6 * 1024 * 1024)])
        out = self._clients(root)
        self.assertEqual({p["exe"] for p in out}, {"helper-alone"})

    def test_python_launcher_windows_are_quiet_engines_are_not(self):
        root = TMP / "proc-python"
        fake_proc(root, 100, "python3.12", [RENDER_FD("5")],
                  cmdline=b"python3.12\0/opt/b70/webwindow.py\0")
        fake_proc(root, 200, "python3.12", [RENDER_FD("6", vram_total_kib=20 * 1024 * 1024)],
                  cmdline=b"python3.12\0-m\0vllm.entrypoints.openai.api_server\0")
        out = self._clients(root)
        self.assertEqual({p["exe"] for p in out}, {"python3.12"})
        self.assertEqual(out[0]["pid"], 200)
        self.assertEqual(out[0]["render"], "renderD128")

    def test_chromium_type_helpers_are_quiet(self):
        root = TMP / "proc-chromium"
        fake_proc(root, 100, "customapp", [RENDER_FD("5")],
                  cmdline=b"customapp\0--type=gpu-process\0--foo\0")
        out = self._clients(root)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
