"""Local-only playground for testing the Cartwheel agent as different roles.

A personal dev tool, not a graded HW2 deliverable. It reuses
agent.agent.build_agent directly (no HTTP token/session layer, no OTel
tracing requirement) so it needs neither Docker nor server/app.py running.

Run with:
    uv run uvicorn playground.server:app --port 8020
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import Runner, SQLiteSession
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import db
from agent.agent import build_agent
from agent.auth import AuthContext
from agent.config import REPO_ROOT
from observability.instrument import load_env

load_env()

app = FastAPI(title="Cartwheel prompt playground")

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSIONS_DB = REPO_ROOT / ".playground-sessions.db"

# Matches the exact demo identities used throughout HW1/HW2 testing.
ROLE_USERS = {
    "shopper": {"user_id": 1, "store_id": None},
    "merchant": {"user_id": 9002, "store_id": 2},
    "support": {"user_id": 9501, "store_id": None},
}

# One running conversation per role, so multi-turn requests (e.g. "why?"
# follow-ups, or a refund tool asking for a reason) work naturally.
_sessions: dict[str, SQLiteSession] = {}


def _session_for(role: str) -> SQLiteSession:
    if role not in _sessions:
        _sessions[role] = SQLiteSession(f"playground-{role}", str(SESSIONS_DB))
    return _sessions[role]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class ChatIn(BaseModel):
    role: str
    message: str
    reset: bool = False


@app.post("/api/chat")
async def chat(body: ChatIn) -> dict[str, Any]:
    if body.role not in ROLE_USERS:
        return {"error": f"unknown role: {body.role!r}"}

    if body.reset:
        _sessions.pop(body.role, None)

    user = ROLE_USERS[body.role]
    ctx = AuthContext(user_id=user["user_id"], role=body.role, store_id=user["store_id"])
    agent = build_agent(ctx)
    session = _session_for(body.role)

    result = await Runner.run(agent, body.message, session=session, context=ctx, max_turns=12)
    return {"reply": result.final_output}


@app.post("/api/reset")
def reset(body: dict[str, str]) -> dict[str, Any]:
    _sessions.pop(body.get("role", ""), None)
    return {"ok": True}


def _order_brief(conn: Any, order_id: int) -> dict[str, Any] | None:
    order = db.get_order(conn, order_id)
    if order is None:
        return None
    store = db.get_store(conn, order.store_id)
    product_title = None
    for product in db.list_products(conn, order.store_id):
        if product.id == order.product_id:
            product_title = product.title
            break
    return {
        "order_id": order.id,
        "store": store.name if store else None,
        "product": product_title,
        "status": order.status,
        "total_usd": order.total_usd,
        "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
        "refund_eligible": order.refund_eligible,
    }


@app.get("/api/sample-data")
def sample_data() -> dict[str, Any]:
    """Curated, live-queried reference data for the sidebar. Order/product
    ids are pinned to known demo records so this stays accurate even after
    ``uv run python -m seed.generate`` (the dev seed is deterministic)."""
    with db.connection() as conn:
        store2 = db.get_store(conn, 2)
        return {
            "shopper": {
                "user_id": 1,
                "orders": [
                    order
                    for oid in (4127, 3980, 4455, 1653, 2485, 9161)
                    if (order := _order_brief(conn, oid)) is not None
                ],
            },
            "merchant": {
                "user_id": 9002,
                "store": (
                    {
                        "name": store2.name,
                        "return_window_days_override": store2.return_window_days_override,
                    }
                    if store2
                    else None
                ),
                "products": [
                    {"product_id": p.id, "title": p.title, "price_usd": p.price_usd}
                    for p in db.list_products(conn, 2)[:6]
                ],
                "orders": [
                    order
                    for oid in (6, 32, 40, 82, 88)
                    if (order := _order_brief(conn, oid)) is not None
                ],
            },
            "support": {
                "user_id": 9501,
                "note": "No orders of their own; can look up any order by id, across stores.",
                "example_orders": [
                    order
                    for oid in (4127, 6, 40)
                    if (order := _order_brief(conn, oid)) is not None
                ],
            },
        }
