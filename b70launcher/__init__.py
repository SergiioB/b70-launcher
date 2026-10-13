"""b70launcher — stdlib backend package for the B70 model launcher.

Split out of the former launcher.py monolith (0.5.x). Importing any module
has NO side effects: no directories are created, no token is written, no
threads start. Entry points call ``b70launcher.paths.init_state`` and
``b70launcher.catalog.load`` explicitly (see launcher.py).

Module map:
  paths      VERSION, install/state paths, token read/write, atomic writes
  locks      the shared LOCK / STATE_LOCK / USAGE_LOCK
  catalog    recipes + settings load, overrides, remote recipe overlay
  scan       disk scan, artifact detection, context resolution
  hardware   GPU preflight (per-kind blockers, busy attribution), telemetry
  downloads  HF downloads: plan / confirm / cancel / worker
  builders   launch command construction (ovms, vllm*, llamacpp, exl3)
  runtime    server state machine, launch/stop/adopt, snapshots, exl3 stack
  telemetry  engine metrics scraping, token accounting, usage history
  harness    client config sync + terminal spawning
  http_api   the localhost JSON API (route table + auth middleware)
"""
from .paths import VERSION

__all__ = ["VERSION"]
