"""Build a native XeStrata command with isolated per-launch settings.

The launcher does not build or ship XeStrata. Recipe fields name the engine's
files *relative to the user's checkout* (settings key ``xestrata_root``, an
absolute path in settings-override.json — same pattern as ``strata_root``):
``py_exe``, ``server_script`` and ``vision_exe``. ``config_file`` is the
user-level XeStrata config (``~`` expands). ``mmproj`` is a bare basename
resolved through the last model scan, or an explicit ``~``/absolute path.
``model_path`` ships empty: hosts that keep the GGUF outside the scan roots
pin it via ``recipe_overrides`` in settings-override.json. A missing piece is
refused by a real launch (dry runs warn) with the setting or file named.
"""
from pathlib import Path
import shlex

from . import common
from .. import catalog, scan
from ..downloads import safe_artifact
from ..paths import IS_WIN

# recipe paths inside the XeStrata checkout (xestrata_root)
PATH_DEFAULTS = {"py_exe": ".venv/bin/python", "server_script": "serve/server.py",
                 "vision_exe": ""}


def _root():
    raw = str(catalog.SETTINGS.get("xestrata_root") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():  # a relative root would resolve against the cwd
        raise common.BuildError(f'setting "xestrata_root" must be an absolute path, not {raw!r}')
    return p


def _rel(recipe, key):
    value = str(recipe.get(key) or "").strip() or PATH_DEFAULTS[key]
    if not value:
        return ""  # optional file (vision_exe) the recipe may leave to the config
    if not safe_artifact(value):
        raise common.BuildError(f"recipe {key} must be a relative path inside the XeStrata checkout, not {value!r}")
    return value


def _mmproj(recipe):
    """mmproj: an explicit ~ / absolute path as-is, else a bare basename
    resolved through the last model scan (machine-agnostic recipes). Empty
    when unset or unresolvable — the caller reports it as a missing piece."""
    raw = str(recipe.get("mmproj") or "").strip()
    if not raw:
        return "", ""
    return raw, (scan._resolve_draft(raw) or "")  # same rules as llama.cpp draft_model


def build_xestrata(b):
    cfg, recipe, warns = b.cfg, b.recipe, b.warns
    if IS_WIN:
        raise common.BuildError('XeStrata SYCL is Linux-only.')
    missing = []
    root = _root()
    root_set = root is not None
    if not root_set:
        missing.append('the XeStrata engine folder: set "xestrata_root" in settings-override.json '
                       '(your XeStrata checkout)')
        root = Path('~/xestrata').expanduser()  # preview placeholder
    py_exe = root / _rel(recipe, 'py_exe')
    server_py = root / _rel(recipe, 'server_script')
    config_raw = str(recipe.get('config_file') or '').strip()
    config_file = str(Path(config_raw).expanduser()) if config_raw else ""
    effort = cfg.get('reasoning_effort', recipe.get('reasoning_effort'))
    vision = cfg.get('vision', recipe.get('vision'))
    vision_exe, mmproj, mmproj_raw = "", "", ""
    if vision is not None and not isinstance(vision, bool):
        raise common.BuildError('Vision must be true or false.')
    if vision:
        rel = _rel(recipe, 'vision_exe')
        vision_exe = str(root / rel) if rel else ""
        mmproj_raw, mmproj = _mmproj(recipe)
    # an unset xestrata_root is the one thing to say; its placeholder files
    # are not checked (same rule as the Strata builder)
    checks = [] if not root_set else [
        (py_exe.is_file(), f"the engine's Python {py_exe} (a venv inside the XeStrata checkout)"),
        (server_py.is_file(), f"the XeStrata server script {server_py}"),
        (bool(config_file) and Path(config_file).is_file(),
         f"the XeStrata config {config_file or '(recipe field config_file)'}"
         + (" (XeStrata writes it under ~/.config/xestrata/)" if config_file else "")),
    ]
    if vision:
        if vision_exe:
            checks.append((Path(vision_exe).is_file(),
                           f"the vision encoder {vision_exe} (inside the XeStrata checkout)"))
        if mmproj_raw and not mmproj:
            missing.append(f'the vision projector {mmproj_raw} (download it into a model '
                           'folder, or give recipe field "mmproj" an absolute path)')
        elif mmproj:
            checks.append((Path(mmproj).is_file(), f"the vision projector {mmproj}"))
    missing += [what for ok, what in checks if not ok]
    if missing:
        text = "XeStrata recipe needs " + "; ".join(missing)
        if not cfg.get('dry_run'):
            raise common.BuildError(text + ".")
        warns.append(text + " — preview only; a real launch is refused until they exist.")
    wrapper = str(Path(__file__).resolve().parents[1] / 'xestrata_server.py')
    tokens = [str(py_exe), wrapper, '--server', str(server_py), '--config', config_file,
              '--ctx', str(b.ctx)]
    if effort is not None:
        if effort not in ('none', 'low', 'medium', 'high', 'xhigh'):
            raise common.BuildError('Unsupported XeStrata reasoning effort.')
        tokens += ['--effort', effort]
    if vision is not None:
        tokens += ['--vision', 'yes' if vision else 'no']
        if vision:
            for flag, value in (('--vision-exe', vision_exe), ('--mmproj', mmproj)):
                if value:
                    tokens += [flag, value]
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
