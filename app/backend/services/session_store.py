"""In-memory session persistence for chat contexts."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable
from uuid import UUID

from app.backend.config import get_settings
from app.backend.models.chat import SessionContext


class SessionStore:
    """Manage ephemeral chat sessions with retention semantics."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._sessions: Dict[UUID, SessionContext] = {}

    def _purge_expired(self) -> None:
        now = datetime.utcnow()
        for session_id, context in list(self._sessions.items()):
            if context.has_expired(now):
                self._sessions.pop(session_id, None)

    def get_or_create(self, session_id: UUID | None = None) -> SessionContext:
        """Return a session context, creating one if needed."""

        self._purge_expired()
        if session_id is not None:
            context = self._sessions.get(session_id)
            if context and not context.has_expired():
                context.touch()
                return context

        context = SessionContext.new(
            session_id=session_id,
            ttl_hours=self._settings.session_retention_hours,
        )
        self._sessions[context.session_id] = context
        return context

    def associate_files(self, session_id: UUID, file_ids: Iterable[UUID]) -> SessionContext:
        """Attach file identifiers to an existing session context."""

        context = self.get_or_create(session_id)
        unique_ids = set(context.active_file_ids)
        unique_ids.update(file_ids)
        context.active_file_ids = sorted(unique_ids)
        context.touch()
        return context

    def clear(self) -> None:
        """Remove all tracked sessions."""

        self._sessions.clear()


__all__ = ["SessionStore"]
