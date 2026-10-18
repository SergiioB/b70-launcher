"""Consumer-visible lifecycle, log-cursor and answer-check regressions."""
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock
try:
    import support_env
except ImportError:
    from tests import support_env
from b70launcher import builders, catalog, downloads, hardware, runtime


class TestRuntime(support_env.LauncherStateCase):
    def tearDown(self):
        self.doCleanups()
        super().tearDown()

    def _wait(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        self.fail('Expected lifecycle transition was not observed.')

    def test_container_output_exit_and_relaunch(self):
        port = support_env.free_port()
        fake = support_env.TMPROOT / 'docker-bin'
        fake.mkdir(exist_ok=True)
        script = fake / 'docker'
        script.write_text(f'#!{sys.executable}\n' + '''import http.server,json,sys,os,threading
args=sys.argv[1:]
if args[0]=='image':
 sys.exit(0 if os.environ.get('FAKE_IMAGE','1')=='1' else 1)
if args[0]=='inspect':
 print('daemon unavailable',file=sys.stderr);sys.exit(1)
if args[0]=='stop': sys.exit(0)
if args[0]=='rm':
 print('Error response from daemon: removal of container x is already in progress',file=sys.stderr);sys.exit(1)
if args[0]!='run': sys.exit(1)
if '-d' in args:
 print('detached-id');sys.exit(0)
port=int(next(a.split('=',1)[1] for a in args if a.startswith('b70.launcher.port=')))
print('ENGINE WEIGHTS LOADED',flush=True)
class H(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  body=b'{"data":[{"id":"actual-served-model"}]}'
  self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_POST(self):
  self.rfile.read(int(self.headers.get('Content-Length','0')))
  self.send_response(200);self.send_header('Content-Length','2');self.end_headers();self.wfile.write(b'{}')
  threading.Timer(.1,lambda:os._exit(7)).start()
http.server.ThreadingHTTPServer(('127.0.0.1',port),H).serve_forever()
''')
        script.chmod(0o755)
        cfg = {'model_id': 'qwen36-35b', 'engine': 'llamacpp', 'port': port,
               'ctx': 8192, 'gpus': [0], 'no_test': True, 'dry_run': True}
        plan = builders.build(cfg)
        cfg['dry_run'] = False
        def cleanup(server_id):
            with mock.patch.dict(os.environ, {'PATH': str(fake)}):
                runtime.stop_server(server_id)
        with mock.patch.dict(os.environ, {'PATH': str(fake)}):
            rid = runtime.launch(cfg, plan)
            self.addCleanup(cleanup, rid)
            self._wait(lambda: any(e['id'] == rid and e['status'] == 'running' for e in runtime.servers_snapshot()))
            self.assertEqual(runtime.RUNNING[rid]['served_name'], 'actual-served-model')
            first_log = runtime.RUNNING[rid]['log']
            self.assertIn('ENGINE WEIGHTS LOADED', '\n'.join(runtime.read_logs(rid)['lines']))
            other = dict(cfg, port=support_env.free_port())
            other_plan = builders.build(dict(other, dry_run=True))
            self.addCleanup(cleanup, other_plan['rid'])
            with self.assertRaises(ValueError):
                runtime.launch(other, other_plan)
            import urllib.request
            with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/exit', b'{}'), timeout=2):
                pass
            self._wait(lambda: any(e['id'] == rid and e['status'] == 'exited' for e in runtime.servers_snapshot()))
            self.assertEqual(runtime.RUNNING[rid]['exitcode'], 7)
            self.assertEqual(runtime.launch(cfg, plan), rid)
            self._wait(lambda: any(e['id'] == rid and e['status'] == 'running' for e in runtime.servers_snapshot()))
            self.assertNotEqual(runtime.RUNNING[rid]['log'], first_log)
            self.assertTrue(__import__('pathlib').Path(first_log).exists())
            runtime.stop_server(rid)

    def test_log_follow_continues_after_tail_window_is_full(self):
        log = support_env.TMPROOT / 'cursor.log'
        log.write_text(''.join(f'old-{i}\n' for i in range(300)))
        runtime.RUNNING['cursor'] = {'id': 'cursor', 'status': 'exited', 'log': str(log)}
        initial = runtime.read_logs('cursor')
        with log.open('a') as stream:
            stream.write('new-1\nnew-2\n')
        result = runtime.read_logs('cursor', initial['offset'])
        self.assertEqual(result['lines'], ['new-1', 'new-2'])
        self.assertGreater(result['offset'], initial['offset'])
        self.assertEqual(runtime.read_logs('cursor', result['offset'])['lines'], [])

    def test_unreachable_docker_is_not_a_removed_container(self):
        e = {'id': 'adopted', 'status': 'running', 'native': False, 'cname': 'owned',
             'port': support_env.free_port(), 'docker_sock': '/run/example.sock',
             'adopted': True, 'proc': None, 'test': {'state': 'pending'}}
        runtime.RUNNING[e['id']] = e
        failure = subprocess.CompletedProcess([], 1, '', 'permission denied connecting to daemon')
        with mock.patch.object(runtime.subprocess, 'run', return_value=failure):
            result = runtime.servers_snapshot()[0]
        self.assertEqual(result['status'], 'starting')
        self.assertNotEqual(e['status'], 'exited')

    def _engine(self, response, status=200):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                body = b'{"data":[{"id":"model"}]}'
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                body = json.dumps(response).encode()
                self.send_response(status)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_port

    def test_reasoning_without_answer_fails_answer_check(self):
        port = self._engine({'choices': [{'message': {'reasoning_content': 'private reasoning'}, 'finish_reason': 'length'}]})
        result = runtime.test_prompt({'port': port})
        self.assertFalse(result['ok'])
        self.assertEqual(result['answer'], '')
        self.assertEqual(result['reasoning'], 'private reasoning')

    def test_inline_reasoning_is_separate_from_answer(self):
        port = self._engine({'choices': [{'message': {'content': '<think>private</think> B70-OK'}, 'finish_reason': 'stop'}]})
        result = runtime.test_prompt({'port': port})
        self.assertTrue(result['ok'])
        self.assertEqual(result['answer'], 'B70-OK')
        self.assertEqual(result['reasoning'], 'private')

    def test_live_engine_http_error_keeps_status(self):
        port = self._engine({'error': 'model unavailable'}, 404)
        result = runtime.test_prompt({'port': port})
        self.assertFalse(result['ok'])
        self.assertEqual(result['http_status'], 404)
        self.assertIn('model unavailable', result['error'])

    def test_unknown_download_size_cannot_authorize_transfer(self):
        model = catalog.find_model('qwen36-35b')
        with mock.patch.object(downloads, 'hf_files', return_value={'config.json': None}):
            plan = downloads.plan_download(model, 'openvino')
        self.assertFalse(plan['enough'])
        self.assertIsNone(plan['size_bytes'])
        did, error = downloads.start_download(model, 'openvino', plan)
        self.assertIsNone(did)
        self.assertFalse(downloads.DOWNLOADS)

    def test_low_memory_foreign_client_blocks_its_card_only(self):
        base = {'devices': [
            {'render': '/dev/dri/renderD128', 'b70': True, 'accessible': True, 'vram_used_mib': 512},
            {'render': '/dev/dri/renderD129', 'b70': True, 'accessible': True, 'vram_used_mib': 512}],
                'default': {'gpus': [0]}}
        clients = [{'pid': 42, 'exe': 'quant-worker', 'render': 'renderD128', 'cgroup': ''}]
        with mock.patch.object(hardware, 'gpu_clients', return_value=clients):
            result = hardware.enrich_busy(base)
            self.assertTrue(result['devices'][0]['busy'])
            self.assertIn('quant-worker', result['devices'][0]['busy_by'])
            self.assertFalse(result['devices'][1]['busy'])
            self.assertEqual(result['default']['gpus'], [1])
            # An owned server on this card must not hide a second, foreign client.
            self.assertTrue(hardware.enrich_busy(base, owned_gpus=[0])['devices'][0]['busy'])
            owned = hardware.enrich_busy(base, owned_gpus=[0], owned_pids=[42])
            self.assertTrue(owned['devices'][0]['busy'])
            self.assertEqual(owned['devices'][0]['busy_by'], 'Launcher server')

    def test_all_busy_cards_have_no_automatic_launch_selection(self):
        base = {'devices': [
            {'render': '/dev/dri/renderD128', 'b70': True, 'accessible': True, 'vram_used_mib': 8192},
            {'render': '/dev/dri/renderD129', 'b70': True, 'accessible': True, 'vram_used_mib': 8192}],
                'default': {'gpus': [0]}}
        with mock.patch.object(hardware, 'gpu_clients', return_value=[]):
            self.assertEqual(hardware.enrich_busy(base)['default']['gpus'], [])

    def test_created_container_is_still_starting(self):
        """`docker run` creates the container before it starts it; with a big image the
        two can be a second or more apart, and inspect then says Running false,
        ExitCode 0. That is not an exit: the attached run client decides."""
        cname = 'b70-created-race-1'
        proc = mock.Mock()
        proc.poll.return_value = None
        entry = {'id': 'created-race', 'cname': cname, 'docker_sock': '', 'native': False, 'proc': proc,
                 'status': 'starting', 'port': support_env.free_port(), 'log': '', 'test': {'state': 'pending'}}
        runtime.RUNNING['created-race'] = entry
        state = {'running': False, 'status': 'created', 'exitcode': 0, 'object': {}}
        with mock.patch.object(runtime, 'docker_containers_status', return_value={cname: state}), \
                mock.patch('b70launcher.telemetry.record_usage'):
            runtime.servers_snapshot()
            self.assertEqual(entry['status'], 'starting')
            proc.poll.return_value = 125  # the start failed: docker run exits, the container stays "created"
            runtime.servers_snapshot()
            self.assertEqual((entry['status'], entry['exitcode']), ('exited', 125))
            entry.update(status='starting', exitcode=None)
            proc.poll.return_value = None
            state.update(status='exited', exitcode=137)  # a container that ran and stopped is an exit
            runtime.servers_snapshot()
            self.assertEqual((entry['status'], entry['exitcode']), ('exited', 137))

    def test_missing_image_is_refused_without_running_docker(self):
        fake = support_env.TMPROOT / 'docker-missing-image'
        fake.mkdir(exist_ok=True)
        calls = fake / 'calls.txt'
        script = fake / 'docker'
        script.write_text(f'#!{sys.executable}\nimport sys\nopen({str(calls)!r},"a").write(" ".join(sys.argv[1:3])+"\\n")\n'
                          'sys.exit(1 if sys.argv[1]=="image" else 0)\n')
        script.chmod(0o755)
        cfg = {'model_id': 'qwen36-35b', 'engine': 'llamacpp', 'port': support_env.free_port(),
               'ctx': 8192, 'gpus': [1], 'no_test': True, 'dry_run': True}
        plan = builders.build(cfg)
        cfg['dry_run'] = False
        with mock.patch.dict(os.environ, {'PATH': str(fake)}):
            rid = runtime.launch(cfg, plan)
            self._wait(lambda: runtime.RUNNING[rid]['status'] == 'exited')
        self.assertIn('not on this machine', runtime.RUNNING[rid]['error'])
        self.assertNotIn('run --rm', calls.read_text())

    def test_container_sees_only_the_selected_card(self):
        preflight = {'devices': [
            {'render': '/dev/dri/renderD128', 'card': 'card0', 'b70': True, 'accessible': True},
            {'render': '/dev/dri/renderD129', 'card': 'card1', 'b70': True, 'accessible': True}],
                     'blockers': [], 'blockers_by_kind': {}, 'default': {'gpus': [1]}}
        with mock.patch.object(hardware, 'hardware_preflight', return_value=preflight):
            for model, engine in (('qwen38-27b-sc5', 'exl3'), ('qwen36-35b', 'llamacpp'),
                                  ('qwen36-35b', 'openvino'), ('qwen36-35b', 'vllm')):
                plan = builders.build({'model_id': model, 'engine': engine, 'gpus': [1],
                                       'dry_run': True, 'port': support_env.free_port()})
                tokens = plan['tokens']
                mapped = [tokens[i + 1] for i, t in enumerate(tokens) if t == '--device' and tokens[i + 1].startswith('/dev')]
                self.assertEqual(mapped, ['/dev/dri/renderD129', '/dev/dri/card1'], f'{model}:{engine}')
                self.assertNotIn('/dev/dri:/dev/dri:ro', tokens)
                self.assertNotIn('ZE_AFFINITY_MASK=1', tokens)

    def test_foreign_container_attribution_and_stop(self):
        cid = '5bcb610fc1842b9793ccab4409a6be1523c2cbe7f73850193e05c2070ccbbc95'
        base = {'devices': [
            {'render': '/dev/dri/renderD128', 'b70': True, 'accessible': True, 'vram_used_mib': 26000},
            {'render': '/dev/dri/renderD129', 'b70': True, 'accessible': True, 'vram_used_mib': 25000}],
                'default': {'gpus': [0]}}
        clients = [
            {'pid': 1001, 'exe': 'python3.12', 'render': 'renderD128', 'cgroup': f'0::/system.slice/docker-{cid}.scope'},
            {'pid': 1002, 'exe': 'python3.12', 'render': 'renderD129', 'cgroup': f'0::/system.slice/docker-{cid}.scope'},
        ]
        cinfo = {cid: {'id': 'bolt-vllm-tp2', 'cid': cid, 'cname': 'bolt-vllm-tp2',
                       'image': 'vllm/vllm-openai-xpu:v0.31.0', 'docker_sock': ''}}
        with mock.patch.object(hardware, 'gpu_clients', return_value=clients), \
             mock.patch.object(hardware, '_resolve_foreign_containers', return_value=cinfo):
            result = hardware.enrich_busy(base)
            self.assertIn('bolt-vllm-tp2', result['devices'][0]['busy_by'])
            self.assertEqual(len(result['foreign']), 1)
            self.assertEqual(result['foreign'][0]['id'], 'bolt-vllm-tp2')
            self.assertEqual(result['foreign'][0]['gpus'], [0, 1])
            with mock.patch.object(runtime, 'preflight', return_value=result), \
                 mock.patch.object(runtime.subprocess, 'run') as mrun:
                mrun.return_value = mock.Mock(returncode=0, stderr='')
                self.assertTrue(runtime.stop_server('bolt-vllm-tp2'))
                self.assertFalse(runtime.stop_server('nonexistent-container'))


if __name__ == '__main__':
    unittest.main()

