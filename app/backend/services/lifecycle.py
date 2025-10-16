"""Lifecycle utilities for managing backend restarts."""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from app.backend.config import get_settings


class BackendLifecycle:
    """Coordinate process-level lifecycle events for the backend."""

    def __init__(self, exit_func: Optional[Callable[[int], None]] = None) -> None:
        self._exit_func = exit_func or os._exit

    def schedule_restart(self) -> Optional[threading.Thread]:
        """Trigger a restart of the backend process after a short grace period."""

        settings = get_settings()
        if not settings.auto_restart_on_purge:
            return None

        delay = max(0.0, settings.restart_grace_seconds)
        return self._schedule(delay, lambda: self._exit_func(0))

    def _schedule(self, delay: float, action: Callable[[], None]) -> threading.Thread:
        thread = threading.Thread(target=self._delayed_action, args=(delay, action), daemon=True)
        thread.start()
        return thread

    @staticmethod
    def _delayed_action(delay: float, action: Callable[[], None]) -> None:
        if delay > 0:
            time.sleep(delay)
        action()


__all__ = ["BackendLifecycle"]
