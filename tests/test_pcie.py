"""PCIe link detection against a fake sysfs tree, plus override validation.

The tree mirrors the live B70 host: the card sits behind an internal switch
(root port -> switch upstream port -> downstream port -> endpoint) whose
internal links report a virtual 2.5 GT/s x1 that must be ignored. Links are
symlinked device entries, exactly like /sys/bus/pci/devices.

Run:  cd b70-launcher && python3 -m unittest tests.test_pcie -v
"""
import os
import unittest
from pathlib import Path

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import catalog, hardware

TMP = support_env.TMPROOT
LauncherStateCase = support_env.LauncherStateCase

SYS = TMP / "pcie-sys"
DEVICES = SYS / "devices" / "pci0000:00"
BUS = SYS / "bus" / "pci" / "devices"

UNKNOWN = {"gen": None, "width": None, "gt_s": None, "max_gen": None,
           "max_width": None, "gbps": None, "source": "unknown", "detected": None}


def card(*components):
    """A sysfs device directory below pci0000:00."""
    path = DEVICES
    for part in components:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def link(path, current=None, width=None, max_speed=None, max_width=None):
    """Seed the four link attributes a PCIe bridge exposes."""
    values = {"current_link_speed": current, "current_link_width": width,
              "max_link_speed": max_speed, "max_link_width": max_width}
    for name, value in values.items():
        if value is not None:
            (path / name).write_text(f"{value}\n")
    return path


def attach(pci, target):
    """``/sys/bus/pci/devices/<pci>`` as a relative symlink, like the live host."""
    BUS.mkdir(parents=True, exist_ok=True)
    (BUS / pci).symlink_to(os.path.relpath(target, BUS))


def setUpModule():
    # GPU 0: root port 16.0 GT/s x8 -> switch 32.0 GT/s x16 capable -> 2.5 GT/s x1 internals
    link(card("0000:00:03.1"), "16.0 GT/s PCIe", 8, "16.0 GT/s PCIe", 8)
    link(card("0000:00:03.1", "0000:09:00.0"), "16.0 GT/s PCIe", 8, "32.0 GT/s PCIe", 16)
    link(card("0000:00:03.1", "0000:09:00.0", "0000:0a:01.0"), "2.5 GT/s PCIe", 1, "2.5 GT/s PCIe", 1)
    link(card("0000:00:03.1", "0000:09:00.0", "0000:0a:01.0", "0000:0b:00.0"),
         "2.5 GT/s PCIe", 1, "2.5 GT/s PCIe", 1)
    attach("0000:0b:00.0", SYS / "devices" / "pci0000:00" / "0000:00:03.1"
           / "0000:09:00.0" / "0000:0a:01.0" / "0000:0b:00.0")
    # GPU 1: second root port, same shape
    link(card("0000:00:03.2"), "16.0 GT/s PCIe", 8, "16.0 GT/s PCIe", 8)
    link(card("0000:00:03.2", "0000:0d:00.0"), "16.0 GT/s PCIe", 8, "32.0 GT/s PCIe", 16)
    link(card("0000:00:03.2", "0000:0d:00.0", "0000:0e:01.0"), "2.5 GT/s PCIe", 1, "2.5 GT/s PCIe", 1)
    link(card("0000:00:03.2", "0000:0d:00.0", "0000:0e:01.0", "0000:0f:00.0"),
         "2.5 GT/s PCIe", 1, "2.5 GT/s PCIe", 1)
    attach("0000:0f:00.0", SYS / "devices" / "pci0000:00" / "0000:00:03.2"
           / "0000:0d:00.0" / "0000:0e:01.0" / "0000:0f:00.0")
    # direct-attached card: root port's child IS the card
    link(card("0000:00:1c.0"), "8.0 GT/s PCIe", 4, "16.0 GT/s PCIe", 16)
    link(card("0000:00:1c.0", "0000:0c:00.0"), "8.0 GT/s PCIe", 4, "16.0 GT/s PCIe", 16)
    attach("0000:0c:00.0", SYS / "devices" / "pci0000:00" / "0000:00:1c.0" / "0000:0c:00.0")
    # reachable device whose link files are absent
    card("0000:00:02.0", "0000:0d:00.0")
    attach("0000:0d:00.0", SYS / "devices" / "pci0000:00" / "0000:00:02.0" / "0000:0d:00.0")
    # speed readable, width not
    (card("0000:00:02.0", "0000:0e:00.0") / "current_link_speed").write_text("16.0 GT/s PCIe\n")
    attach("0000:0e:00.0", SYS / "devices" / "pci0000:00" / "0000:00:02.0" / "0000:0e:00.0")


class TestPcieLink(unittest.TestCase):
    def link(self, pci, **kw):
        # the implicit settings["pcie_links"] path is covered by TestSettingsOverride
        kw.setdefault("overrides", {})
        return hardware.pcie_link(pci, sysfs_root=str(SYS), **kw)

    def test_switch_chain_uses_root_port_and_its_child(self):
        detected = {"gen": 4, "width": 8, "max_gen": 5, "max_width": 16}
        self.assertEqual(self.link("0000:0b:00.0"),
                         {"gen": 4, "width": 8, "gt_s": 16.0, "max_gen": 5, "max_width": 16,
                          "gbps": 15.8, "source": "sysfs", "detected": detected})
        self.assertEqual(self.link("0000:0f:00.0")["detected"], detected)

    def test_internal_switch_link_is_ignored(self):
        link = self.link("0000:0b:00.0")
        # 2.5 GT/s x1 sits only on the switch's internal links
        self.assertNotEqual(link["gen"], 1)
        self.assertNotEqual(link["width"], 1)
        self.assertEqual(link["gbps"], 15.8)

    def test_direct_attached_card(self):
        self.assertEqual(self.link("0000:0c:00.0"),
                         {"gen": 3, "width": 4, "gt_s": 8.0, "max_gen": 4, "max_width": 16,
                          "gbps": 3.9, "source": "sysfs",
                          "detected": {"gen": 3, "width": 4, "max_gen": 4, "max_width": 16}})

    def test_missing_links_are_unknown(self):
        for pci in ("0000:0d:00.0",        # link files absent
                    "0000:0e:00.0",        # speed without width
                    "0000:99:00.0",        # no such device
                    "0b:00.0",             # malformed address
                    "0000:0b:00", "0000:0b:00.9", None):
            self.assertEqual(self.link(pci), dict(UNKNOWN), pci)

    def test_missing_sysfs_root_is_unknown(self):
        self.assertEqual(hardware.pcie_link("0000:0b:00.0", sysfs_root=str(TMP / "nope"),
                                            overrides={}), dict(UNKNOWN))

    def test_user_override_wins_and_keeps_detected(self):
        link = self.link("0000:0b:00.0", overrides={"0000:0b:00.0": {"gen": 5, "width": 16}})
        self.assertEqual(link["source"], "user")
        self.assertEqual((link["gen"], link["width"], link["gt_s"], link["gbps"]), (5, 16, 32.0, 63.0))
        self.assertEqual(link["max_gen"], 5)
        self.assertEqual(link["max_width"], 16)
        self.assertEqual(link["detected"],
                         {"gen": 4, "width": 8, "max_gen": 5, "max_width": 16})

    def test_override_without_sysfs_still_reports_the_choice(self):
        link = self.link("0000:99:00.0", overrides={"0000:99:00.0": {"gen": 3, "width": 4}})
        self.assertEqual(link, {"gen": 3, "width": 4, "gt_s": 8.0, "max_gen": None,
                                "max_width": None, "gbps": 3.9, "source": "user",
                                "detected": None})

    def test_invalid_overrides_are_ignored(self):
        for entry in ({"gen": 7, "width": 16},       # gen out of range
                      {"gen": 0, "width": 16},
                      {"gen": 4, "width": 3},        # width not a valid lane count
                      {"gen": 4},                    # incomplete
                      {"gen": 4, "width": 16, "extra": 1},
                      {"gen": True, "width": 8},     # bool is not a generation
                      {"gen": "4", "width": 8},
                      "x", None, []):
            self.assertEqual(self.link("0000:0b:00.0", overrides={"0000:0b:00.0": entry}),
                             self.link("0000:0b:00.0"), entry)
        for overrides in ({"nope": {"gen": 4, "width": 8}},          # bad address
                          {"0000:0b:00.0": {"gen": 4, "width": 8}},  # other device
                          "x", [], 7):
            self.assertEqual(self.link("0000:0c:00.0", overrides=overrides),
                             self.link("0000:0c:00.0"), overrides)


class TestCleanPcieLinks(unittest.TestCase):
    def test_merge_removal_and_clear(self):
        stored = {"0000:0b:00.0": {"gen": 5, "width": 16},
                  "0000:0f:00.0": {"gen": 4, "width": 8}}
        self.assertEqual(hardware.clean_pcie_links({"0000:0f:00.0": None}, stored),
                         {"0000:0b:00.0": {"gen": 5, "width": 16}})
        self.assertEqual(hardware.clean_pcie_links({"0000:0c:00.0": {"gen": 2, "width": 2}}, stored),
                         dict(stored, **{"0000:0c:00.0": {"gen": 2, "width": 2}}))
        self.assertEqual(hardware.clean_pcie_links(None, stored), {})
        self.assertEqual(hardware.clean_pcie_links({"0000:0b:00.0": {"gen": 6, "width": 1}}),
                         {"0000:0b:00.0": {"gen": 6, "width": 1}})

    def test_malformed_values_raise(self):
        for value in ([], "x", 7, {"0b:00.0": {"gen": 4, "width": 8}},
                      {"0000:0b:00.0": {"gen": 7, "width": 8}},
                      {"0000:0b:00.0": {"gen": 4, "width": 3}},
                      {"0000:0b:00.0": {"gen": 4, "width": 32}},
                      {"0000:0b:00.0": {"gen": 4, "width": "8"}},
                      {"0000:0b:00.0": {"gen": 4}}, {"0000:0b:00.0": 5}):
            with self.assertRaises(ValueError, msg=value):
                hardware.clean_pcie_links(value)

    def test_malformed_stored_map_is_dropped(self):
        self.assertEqual(hardware.clean_pcie_links({}, {"0000:0b:00.0": {"gen": 9, "width": 8},
                                                        "junk": {"gen": 4, "width": 8}}), {})


class TestSettingsOverride(LauncherStateCase):
    """pcie_link() picks up settings["pcie_links"] when no explicit map is given."""

    def test_settings_override_applies(self):
        self.assertEqual(hardware.pcie_link("0000:0b:00.0", sysfs_root=str(SYS))["source"], "sysfs")
        catalog.SETTINGS["pcie_links"] = {"0000:0b:00.0": {"gen": 5, "width": 8}}
        link = hardware.pcie_link("0000:0b:00.0", sysfs_root=str(SYS))
        self.assertEqual((link["source"], link["gen"], link["width"], link["gbps"]),
                         ("user", 5, 8, 31.5))
        self.assertEqual(link["detected"]["gen"], 4)
        # an unrelated device keeps its sysfs reading
        self.assertEqual(hardware.pcie_link("0000:0f:00.0", sysfs_root=str(SYS))["source"], "sysfs")


if __name__ == "__main__":
    unittest.main()
