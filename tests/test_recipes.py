"""Catalog update security, persistence and precedence invariants."""
import json
import unittest
from unittest import mock
try:
    import support_env
except ImportError:
    from tests import support_env
from b70launcher import catalog, paths


class TestCatalogUpdates(support_env.LauncherStateCase):
    def test_remote_cannot_replace_executable_or_host_path(self):
        original = dict(catalog.find_model('qwen36-35b')['recipes']['llamacpp'])
        doc = {'models': [{'id': 'qwen36-35b', 'recipes': {'llamacpp': {
            'image': 'malicious/image', 'llama_bin': '/tmp/evil', 'gguf': '/etc/shadow',
            'fixed_flags': ['--evil'], 'ctx': 4096}}}]}
        applied, needs_app, error = catalog.apply_recipe_doc(doc)
        result = catalog.find_model('qwen36-35b')['recipes']['llamacpp']
        self.assertEqual(result['ctx'], 4096)
        for key in ('image', 'llama_bin', 'gguf', 'fixed_flags'):
            self.assertEqual(result.get(key), original.get(key))
        self.assertEqual(needs_app, [])
        self.assertIsNone(error)

    def test_new_model_and_retired_engine_require_app_update(self):
        doc = {'models': [
            {'id': 'unknown', 'recipes': {'vllm': {'kind': 'vllm', 'image': 'x'}}},
            {'id': 'qwen38-27b', 'recipes': {'vllm': {'kind': 'vllm-arext', 'image': 'x'}}}]}
        count, needs_app, error = catalog.apply_recipe_doc(doc)
        self.assertEqual(count, 0)
        self.assertEqual(set(needs_app), {'unknown:vllm', 'qwen38-27b:vllm'})
        self.assertIsNone(catalog.find_model('unknown'))
        self.assertNotIn('vllm', catalog.find_model('qwen38-27b')['recipes'])

    def test_remote_cannot_release_operator_hold(self):
        recipe = catalog.find_model('qwen38-27b-fp8-tp2')['recipes']['vllm']
        before = recipe['operator_hold']
        catalog.apply_recipe_doc({'models': [{'id': 'qwen38-27b-fp8-tp2',
            'recipes': {'vllm': {'operator_hold': '', 'ctx': 4096}}}]})
        self.assertEqual(recipe['operator_hold'], before)

    def test_local_override_precedes_remote_change(self):
        catalog._apply_recipe_overrides({'qwen36-35b': {'llamacpp': {'ctx': 2048}}})
        catalog.apply_recipe_doc({'models': [{'id': 'qwen36-35b', 'recipes': {'llamacpp': {'ctx': 4096}}}]})
        self.assertEqual(catalog.find_model('qwen36-35b')['recipes']['llamacpp']['ctx'], 2048)

    def test_single_engine_update_leaves_other_engine_unchanged(self):
        before = dict(catalog.find_model('qwen36-35b')['recipes']['openvino'])
        doc = {'catalog_ver': '2027-01-01', 'models': [{'id': 'qwen36-35b', 'recipes': {
            'llamacpp': {'ctx': 4096}, 'openvino': {'ctx': 1024}}}]}
        with mock.patch.object(catalog, 'fetch_remote_recipes', return_value=(doc, None)):
            result = catalog.apply_recipe_update('qwen36-35b', 'llamacpp')
        self.assertTrue(result['ok'])
        self.assertEqual(catalog.find_model('qwen36-35b')['recipes']['openvino'], before)
        self.assertEqual(catalog.find_model('qwen36-35b')['recipes']['llamacpp']['ctx'], 4096)
        stored = json.loads(paths.REMOTE_RECIPES_PATH.read_text())
        self.assertEqual(set(stored['payload']['models'][0]['recipes']), {'llamacpp'})

    def test_corrupt_overlay_preserves_shipped_routes(self):
        before = json.dumps(catalog.RECIPES, sort_keys=True)
        paths.REMOTE_RECIPES_PATH.write_text('{bad')
        catalog._load_remote_overlay()
        self.assertEqual(json.dumps(catalog.RECIPES, sort_keys=True), before)

    def test_bad_document_has_no_partial_mutation(self):
        before = json.dumps(catalog.RECIPES, sort_keys=True)
        count, needs_app, error = catalog.apply_recipe_doc({'models': 'not a list'})
        self.assertEqual(count, 0)
        self.assertEqual(needs_app, [])
        self.assertIsNotNone(error)
        self.assertEqual(json.dumps(catalog.RECIPES, sort_keys=True), before)


if __name__ == '__main__':
    unittest.main()
