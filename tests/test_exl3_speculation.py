"""EXL3's explicit speculation switch must reach the server, without a GPU."""
import unittest

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import builders


class TestExl3Speculation(support_env.LauncherStateCase):
    def plan(self, model="qwen38-27b", **options):
        config = {"model_id": model, "engine": "exl3", "ctx": 8192,
                  "port": support_env.free_port(), "dry_run": True}
        config.update(options)
        plan = builders.build(config)
        self.assertNotIn("error", plan)
        return plan

    def test_off_reaches_both_exl3_recipes(self):
        for model in ("qwen38-27b", "qwen38-27b-sc5"):
            with self.subTest(model=model):
                tokens = self.plan(model, mtp=False)["tokens"]
                self.assertEqual(tokens[-2:], ["--set", "vllm.speculative_config=null"])

    def test_explicit_off_wins_over_conflicting_extra_setting(self):
        plan = self.plan(mtp=False, extra="--set vllm.speculative_config=conflicting")
        self.assertEqual(plan["tokens"][-2:], ["--set", "vllm.speculative_config=null"])
        self.assertTrue(any("disabled (--no-mtp)" in w for w in plan["warnings"]))

    def test_existing_default_does_not_change(self):
        for options in ({}, {"mtp": True}):
            plan = self.plan(**options)
            self.assertNotIn("vllm.speculative_config=null", plan["tokens"])
            self.assertTrue(any("can change the greedy answer" in w for w in plan["warnings"]))


if __name__ == "__main__":
    unittest.main()
