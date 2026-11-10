"""The EXL3 image store must warn before a ~32 GB pull onto a small disk."""
import unittest

try:
    import support_env  # noqa: F401
except ImportError:
    from tests import support_env  # noqa: F401

from b70launcher.builders import exl3


class DataRootTests(unittest.TestCase):
    def test_small_disk_warns_with_setting_name(self):
        w = exl3.data_root_warnings(free_bytes=8 * 1073741824)
        self.assertEqual(len(w), 1)
        self.assertIn("exl3_data_root", w[0])
        self.assertIn("8.0 GiB free", w[0])

    def test_large_disk_is_quiet(self):
        self.assertEqual(exl3.data_root_warnings(free_bytes=200 * 1073741824), [])


if __name__ == "__main__":
    unittest.main()
