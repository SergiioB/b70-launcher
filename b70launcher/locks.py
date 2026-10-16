"""Shared locks.

One reentrant LOCK guards the mutable catalogs (RECIPES/SETTINGS/SCAN/
RUNNING/DOWNLOADS) exactly like the former module-global RLock; STATE_LOCK
serializes every persisted-file write (audit M12); USAGE_LOCK serializes the
usage-history file.
"""
import threading

LOCK = threading.RLock()  # reentrant: some paths record usage while tracked
STATE_LOCK = threading.Lock()
USAGE_LOCK = threading.Lock()
