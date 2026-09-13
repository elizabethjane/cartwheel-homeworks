"""Authentication tests for the HW2 session/message endpoints.

Everything here runs offline: no Langfuse, no Docker, no model provider key.
These call the endpoint functions directly (no running uvicorn process),
the same way tests/test_hw_holes.py::test_hw2_create_session_binds_verified_identity
does.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server import app as server_app


def test_create_session_rejects_role_mismatch(world: dict) -> None:
    """A claimed role that doesn't match the user's stored role is a 403,
    not a silently-accepted identity. User 1 is a shopper in the seed data."""
    server_app._SESSIONS.clear()

    with pytest.raises(HTTPException) as exc_info:
        server_app.create_session(
            server_app.SessionCreate(user_id=1, role="merchant")
        )

    assert exc_info.value.status_code == 403


def test_token_cannot_authorize_different_session(world: dict) -> None:
    """A token issued for session A must not authorize requests against
    session B, even though both tokens are validly signed."""
    server_app._SESSIONS.clear()

    session_a = server_app.create_session(
        server_app.SessionCreate(user_id=1, role="shopper")
    )
    session_b = server_app.create_session(
        server_app.SessionCreate(user_id=9002, role="merchant")
    )

    with pytest.raises(HTTPException) as exc_info:
        server_app._authorize(
            session_b["session_id"], f"Bearer {session_a['token']}"
        )

    assert exc_info.value.status_code == 403
