"""Restarting the SDK's background threads in a forked child.

Threads do not survive ``fork()``. A client built before a prefork server forks - gunicorn
``--preload``, uWSGI, ``uvicorn --workers`` with a module-level client - would otherwise sit in
every worker with no poll thread and no log sender, serving the snapshot it happened to hold at
fork time for the life of the process.

Every store and buffer that starts a thread registers itself here (weakly, so a client that is
dropped is still collected). After a fork the child rebuilds the locks it inherited - one of them
may have been held by a thread that no longer exists - and restarts what it had running.
"""

from __future__ import annotations

import contextlib
import os
import weakref
from typing import Any

_registry: weakref.WeakSet[Any] = weakref.WeakSet()


def register(target: Any) -> None:
    """Restart ``target._restart_after_fork()`` in a child process after ``fork()``."""
    _registry.add(target)


def _after_fork_in_child() -> None:
    for target in list(_registry):
        # a fork handler may never raise into the app
        with contextlib.suppress(Exception):
            target._restart_after_fork()


if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX only
    os.register_at_fork(after_in_child=_after_fork_in_child)
