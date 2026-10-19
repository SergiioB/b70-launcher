"""Apply per-launch XeStrata overrides without modifying the user's source config."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ctx', type=int, required=True)
    parser.add_argument('--effort', choices=['none', 'low', 'medium', 'high', 'xhigh'])
    parser.add_argument('--vision', choices=['yes', 'no'])
    parser.add_argument('--vision-exe')
    parser.add_argument('--mmproj')
    args, server_args = parser.parse_known_args()
    config = json.loads(Path(args.config).read_text())
    engine_args = config['args']
    index = engine_args.index('--max-context')
    engine_args[index + 1] = str(args.ctx)
    for flag in ('--rope-scaling', '--rope-scale', '--yarn-orig-ctx'):
        if flag in engine_args:
            index = engine_args.index(flag)
            del engine_args[index:index + 2]
    if args.ctx > 262144:
        engine_args += ['--rope-scaling', 'yarn', '--rope-scale', str(args.ctx / 262144),
                        '--yarn-orig-ctx', '262144']
    if args.effort is not None:
        config['reasoning_effort'] = args.effort
    if args.vision == 'no':
        config.pop('vision', None)
        config['args'] = [flag for flag in engine_args if flag != '--vision']
    elif args.vision == 'yes':
        vision = dict(config.get('vision') or {})
        if args.vision_exe:
            vision['exe'] = args.vision_exe
        if args.mmproj:
            vision['mmproj'] = args.mmproj
        vision['model'] = engine_args[engine_args.index('--native') + 1]
        for key in ('exe', 'mmproj', 'model'):
            if not vision.get(key) or not Path(vision[key]).is_file():
                parser.error('Vision requires an existing ' + key + ' file')
        config['vision'] = vision
    # Keep the original config intact, including when the server exits with an error.
    with tempfile.TemporaryDirectory(prefix='b70-xestrata-') as directory:
        launch_config = Path(directory) / 'config.json'
        launch_config.write_text(json.dumps(config))
        sys.path.insert(0, str(Path(args.server).resolve().parent.parent))
        sys.argv = [args.server, '--config', str(launch_config), *server_args]
        runpy.run_path(args.server, run_name='__main__')


if __name__ == '__main__':
    main()
