"""Start the isolated stack only on launch; never delete daemon state."""
import os
import subprocess
import threading
import time
from pathlib import Path
from . import catalog, paths

_STACK_LOCK = threading.Lock()


def _checked(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Command failed: {' '.join(argv)}")
    return result


def ensure_exl3_stack(sock):
    from .runtime import docker_cmd, _pid_alive
    with _STACK_LOCK:
        try:
            probe = subprocess.run(docker_cmd(sock) + ["info"], capture_output=True, text=True, timeout=8)
            if probe.returncode == 0:
                return
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("EXL3 Docker daemon did not answer. Its state was left unchanged.") from exc
        pidfile = Path('/run/b70-exl3-docker.pid')
        try:
            live = _pid_alive(int(pidfile.read_text().strip()))
        except (OSError, ValueError):
            live = False
        if live or Path(sock).exists():
            raise RuntimeError("EXL3 Docker socket exists but is unavailable. Check access or restart its owner; no live state was removed.")
        if sock != '/run/b70-exl3-docker.sock':
            raise RuntimeError("Custom Docker socket is unavailable. Start its daemon before launching.")
        root = str(Path(catalog.SETTINGS.get('exl3_data_root') or '~/.local/share/b70-exl3').expanduser())
        bridge = 'b70-exl3-br0'
        found = subprocess.run(['ip', 'link', 'show', 'dev', bridge], capture_output=True, text=True, timeout=10)
        if found.returncode:
            _checked(['sudo', '-n', 'ip', 'link', 'add', bridge, 'type', 'bridge'])
            _checked(['sudo', '-n', 'ip', 'addr', 'add', '172.31.77.1/24', 'dev', bridge])
        _checked(['sudo', '-n', 'ip', 'link', 'set', bridge, 'up'])
        nat = ['sudo', '-n', 'iptables', '-t', 'nat']
        rule = ['POSTROUTING', '-s', '172.31.77.0/24', '!', '-o', bridge, '-j', 'MASQUERADE']
        check = subprocess.run(nat + ['-C'] + rule, capture_output=True, text=True, timeout=10)
        if check.returncode:
            _checked(nat + ['-A'] + rule)
        state = '/run/b70-exl3-containerd'
        cfg_file = paths.DATA / 'exl3-daemon.json'
        try:
            cfg_file.write_text('{}\n')
            os.chmod(cfg_file, 0o644)
        except OSError:
            pass
        with open(paths.LOGDIR / 'exl3-stack.log', 'ab') as log:
            # Existing containerd state is never removed or overwritten.
            if not Path(state + '/containerd.sock').exists():
                child = subprocess.Popen(['sudo', '-n', 'containerd', '--root=' + root + '-containerd',
                                          '--state=' + state, '--address=' + state + '/containerd.sock'],
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                time.sleep(1)
                if child.poll() is not None:
                    raise RuntimeError('containerd startup failed; inspect exl3-stack.log.')
            # overlay2 with the containerd snapshotter off: the image store
            # lives in data-root, so moving data-root moves the images with it.
            # Pass an isolated empty --config-file so a host /etc/docker/daemon.json
            # with its own "data-root" cannot conflict with --data-root.
            daemon = subprocess.Popen(['sudo', '-n', 'dockerd',
                                       '--config-file=' + str(cfg_file),
                                       '--data-root=' + root + '-docker',
                                       '--exec-root=/run/b70-exl3-docker', '--host=unix://' + sock,
                                       '--pidfile=/run/b70-exl3-docker.pid', '--bridge=' + bridge,
                                       '--ip-forward=false', '--ip-masq=false', '--group=docker',
                                       '--storage-driver=overlay2',
                                       '--feature', 'containerd-snapshotter=false',
                                       '--containerd=' + state + '/containerd.sock'],
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if daemon.poll() is not None:
                detail = ''
                try:
                    lines = [ln.strip() for ln in (paths.LOGDIR / 'exl3-stack.log').read_text(errors='replace').splitlines() if ln.strip()]
                    if lines:
                        detail = f' ({lines[-1]})'
                except OSError:
                    pass
                raise RuntimeError(f'EXL3 Docker startup failed; inspect exl3-stack.log.{detail}')
            if Path(sock).exists():
                result = subprocess.run(docker_cmd(sock) + ['info'], capture_output=True, timeout=3)
                if result.returncode == 0:
                    return
            time.sleep(1)
        raise RuntimeError('EXL3 Docker startup timed out; inspect exl3-stack.log. State was retained.')
