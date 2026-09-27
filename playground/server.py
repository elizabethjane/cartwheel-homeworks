"""Local-only playground for testing the Cartwheel agent as different roles.

A personal dev tool, not a graded HW2 deliverable. It reuses
agent.agent.build_agent directly (no HTTP token/session layer) so it needs
neither the real server/app.py nor a signed token. It DOES set up the same
Langfuse/OTel tracing as the graded server, though, and tags each turn with
the same cartwheel.* attributes -- so conversations you run here show up
correctly in the trace viewer dashboard, taggable by hw_stage/git_commit
and (optionally) a scenario_id for HW3 testing. Tracing degrades gracefully
when Docker/Langfuse isn't running (setup_tracing() just warns and no-ops).

Run with:
    uv run uvicorn playground.server:app --port 8020
"""

from __future__ import annotations

import ast
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from agents import Runner, RunConfig
from fastapi import FastAPI
from fastapi.responses import FileResponse
from opentelemetry import trace
from pydantic import BaseModel

from agent import db
from agent import tools as hw_tools
from agent.agent import build_agent, prompt_version
from agent.auth import AuthContext
from observability.instrument import (
    CARTWHEEL_GIT_COMMIT,
    CARTWHEEL_HW_STAGE,
    load_env,
    setup_tracing,
)

load_env()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_tracing()  # no-op with a warning if Docker/Langfuse isn't up
    yield


app = FastAPI(title="Cartwheel prompt playground", lifespan=lifespan)
_tracer = trace.get_tracer("cartwheel.playground")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Matches the exact demo identities used throughout HW1/HW2 testing.
ROLE_USERS = {
    "shopper": {"user_id": 1, "store_id": None},
    "merchant": {"user_id": 9002, "store_id": 2},
    "support": {"user_id": 9501, "store_id": None},
}

# Support can view any order, so this is used purely to reuse
# check_return_eligibility's policy/window computation for the sample-data
# sidebar -- it's reference info shown regardless of which role is
# currently selected, not a claim about what that role itself can see.
_SUPPORT_CTX = AuthContext(user_id=9501, role="support")


@app.get("/")
def index() -> FileResponse:
    # No caching: this is a dev tool edited frequently, and a stale cached
    # copy of the page (browser or embedded preview pane) has repeatedly
    # masked real edits during development.
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


class ChatIn(BaseModel):
    role: str
    message: str
    # Optional: tag the resulting trace with a scenario id, e.g. to verify
    # HW3's scenario-runner tagging works before running it for real.
    scenario_id: str | None = None


@app.post("/api/chat")
async def chat(body: ChatIn) -> dict[str, Any]:
    """Every call is a fresh, single-turn conversation -- no session, no
    history. This is deliberate: the playground's job is to let each prompt
    (suggested or free-text) be judged as an isolated test case, the same
    way HW1 Part B always started a fresh CLI session per conversation.
    A prior design kept one running SQLiteSession per role so follow-up
    turns (e.g. a refund tool asking for a reason) worked naturally, but
    that meant unrelated later tests could see stale context from much
    earlier ones -- e.g. asking about order 4127 as a merchant could answer
    "as I mentioned earlier" from a denial several tests ago. Statelessness
    trades that away: a genuine multi-turn flow (ask for a refund, then
    reply with just the reason) won't be remembered across two messages
    anymore -- put the whole request in one message instead (the suggested
    prompts already do this, e.g. "I'd like a refund for order 4455, I
    changed my mind").

    Each call runs inside its own cartwheel.session_message root span,
    tagged the same way server/app.py's post_message tags a real request
    (role, user id, prompt version, hw_stage, git_commit, and scenario_id
    when given) -- so playground activity shows up correctly in the trace
    viewer dashboard. There's no real "session" here (see above), so
    cartwheel.session_id is a fresh id per call, just for trace correlation,
    not a claim of conversation continuity.
    """
    if body.role not in ROLE_USERS:
        return {"error": f"unknown role: {body.role!r}"}

    user = ROLE_USERS[body.role]
    ctx = AuthContext(user_id=user["user_id"], role=body.role, store_id=user["store_id"])
    agent = build_agent(ctx)
    version = prompt_version()
    pseudo_session_id = f"playground-{uuid.uuid4()}"

    with _tracer.start_as_current_span("cartwheel.session_message") as span:
        if span.is_recording():
            span.set_attribute("cartwheel.session_id", pseudo_session_id)
            span.set_attribute("cartwheel.user_role", ctx.role)
            span.set_attribute("cartwheel.user_id", str(ctx.user_id))
            span.set_attribute("cartwheel.prompt_version", version)
            span.set_attribute("cartwheel.hw_stage", CARTWHEEL_HW_STAGE)
            span.set_attribute("cartwheel.git_commit", CARTWHEEL_GIT_COMMIT)
            if body.scenario_id:
                span.set_attribute("cartwheel.scenario_id", body.scenario_id)

        result = await Runner.run(
            agent, body.message, context=ctx, max_turns=12, run_config=RunConfig()
        )

    return {"reply": result.final_output, "tool_calls": _extract_tool_calls(result.new_items)}


def _extract_tool_calls(new_items: list[Any]) -> list[dict[str, Any]]:
    """Same extraction agent/cli.py's --debug flag does (see _print_tool_calls),
    but building a JSON-serializable list instead of printing to stdout."""
    outputs = {
        item.call_id: item.output
        for item in new_items
        if item.type == "tool_call_output_item" and item.call_id is not None
    }
    calls = []
    for item in new_items:
        if item.type != "tool_call_item":
            continue
        raw = item.raw_item
        args = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        result_value = outputs.get(item.call_id)
        if isinstance(result_value, str):
            try:
                # Tool functions return Python dicts; the SDK stores their
                # repr() as a string (single-quoted, not valid JSON).
                result_value = ast.literal_eval(result_value)
            except (ValueError, SyntaxError):
                pass  # leave as the raw string
        calls.append({"name": item.tool_name, "arguments": args, "result": result_value})
    return calls


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

    # Reuse the agent's own eligibility tool so the sidebar always agrees
    # with what the agent would actually say -- including *why*: which
    # policy applies (store override vs. the cw-returns platform default)
    # and the exact window/deadline, not just a bare eligible yes/no.
    eligibility = hw_tools.check_return_eligibility(_SUPPORT_CTX, order_id)

    return {
        "order_id": order.id,
        "store": store.name if store else None,
        "product": product_title,
        "status": order.status,
        "total_usd": order.total_usd,
        "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
        "refund_eligible": order.refund_eligible,
        "policy_id": eligibility.get("policy_id"),
        "return_window_days": eligibility.get("return_window_days"),
        "return_deadline": eligibility.get("deadline"),
        "eligibility_reason": eligibility.get("reason"),
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
