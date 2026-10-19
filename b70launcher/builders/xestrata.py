"""Build a native XeStrata command with isolated per-launch settings."""
from pathlib import Path
import shlex

from . import common
from ..paths import IS_WIN


def build_xestrata(b):
    cfg, recipe, warns = b.cfg, b.recipe, b.warns
    if IS_WIN:
        raise common.BuildError('XeStrata SYCL is Linux-only.')
    py_exe = str(recipe.get('py_exe') or '/data/sata1/engines/xestrata/.venv/bin/python')
    server_py = str(recipe.get('server_script') or '/data/sata1/engines/xestrata/serve/server.py')
    config_file = str(Path(recipe.get('config_file') or '~/.config/xestrata/xestrata-iq3_xxs.json').expanduser())
    wrapper = str(Path(__file__).resolve().parents[1] / 'xestrata_server.py')
    tokens = [py_exe, wrapper, '--server', server_py, '--config', config_file, '--ctx', str(b.ctx)]
    effort = cfg.get('reasoning_effort', recipe.get('reasoning_effort'))
    if effort is not None:
        if effort not in ('none', 'low', 'medium', 'high', 'xhigh'):
            raise common.BuildError('Unsupported XeStrata reasoning effort.')
        tokens += ['--effort', effort]
    vision = cfg.get('vision', recipe.get('vision'))
    if vision is not None:
        if not isinstance(vision, bool):
            raise common.BuildError('Vision must be true or false.')
        tokens += ['--vision', 'yes' if vision else 'no']
        if vision:
            for field, flag in (('vision_exe', '--vision-exe'), ('mmproj', '--mmproj')):
                if recipe.get(field):
                    tokens += [flag, recipe[field]]
    tokens += ['--engine', 'strata', '--port', str(b.port), '--host', '127.0.0.1',
               '--gpu', ','.join(map(str, b.gpus))]
    tokens += recipe.get('fixed_flags', []) + b.extra
    cmd = 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && exec ' + shlex.join(tokens)
    warns.append('XeStrata native SYCL: speed depends on quant, prompt, speculation and cache state; see recipe evidence.')
    return {
        'tokens': ['bash', '-c', cmd],
        'env': recipe.get('env', {}),
        'cname': '',
        'served_name': recipe.get('served_model_name', 'qwen3.8-flash-next'),
        'native': True,
        'docker_sock': '',
    }
