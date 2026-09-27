"""Local, read-only trace viewer for the Cartwheel support agent.

A personal dev tool, not a graded HW deliverable. Reads traces straight from
the self-hosted Langfuse instance's public API (localhost) -- no database of
its own, no writes back to Langfuse, no annotation controls. Built to make
it easy to read a whole trace end to end and scan for failures: the exact
conversation the model saw, every tool call with its full arguments and
result, and a raw observation view for anything the friendly views don't
show.

Run with:
    uv run uvicorn trace_viewer.server:app --port 8030
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from agent.config import REPO_ROOT
from observability.instrument import load_env

load_env()

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3001")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")

app = FastAPI(title="Cartwheel trace viewer")
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Matches a full or short git hash. Deliberately strict: these values come
# from Langfuse trace attributes (which the frontend lets you click), and a
# ref beginning with "-" would otherwise be interpreted by git as a flag
# (argument injection), not a revision -- reject anything that isn't a
# plain hex hash before it ever reaches subprocess.
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{4,40}$")

_client = httpx.Client(
    base_url=LANGFUSE_HOST,
    auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
    timeout=15.0,
)


def _langfuse_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = _client.get(f"/api/public{path}", params=params)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Langfuse API {path} returned {resp.status_code}: {resp.text[:300]}",
        )
    return resp.json()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/traces")
def list_traces(page: int = 1, limit: int = 20) -> dict[str, Any]:
    """Pass-through list of traces, newest first (Langfuse's own default
    order). No enrichment, no filtering -- exactly what Langfuse has."""
    return _langfuse_get("/traces", {"page": page, "limit": limit})


def _richest_generation(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The GENERATION observation whose `input` carries the longest running
    message history -- i.e. the last model call in the agent's tool loop.
    Its `input` array is the fullest record of the turn-by-turn exchange
    (system prompt, user message, each assistant tool call, each tool
    response) short of the very final assistant reply, which Langfuse
    records on the trace's own `output` instead (see agent/agent.py's
    OpenLLMetry integration -- the last GENERATION span's own `output` is
    not populated by this instrumentation)."""
    generations = [o for o in observations if o.get("type") == "GENERATION"]
    if not generations:
        return None
    return max(generations, key=lambda o: len(o.get("input") or []))


def _message_text(messages: Any) -> str | None:
    """The first text part's content out of a Langfuse message or
    single-item message list, or None if there isn't one."""
    if not messages:
        return None
    msg = messages[0] if isinstance(messages, list) else messages
    for part in msg.get("parts") or []:
        if part.get("type") == "text":
            return part.get("content")
    return None


def _build_conversation(trace: dict[str, Any], observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    richest = _richest_generation(observations)
    full_history = list(richest["input"]) if richest and richest.get("input") else []

    # A session in this app can span multiple HTTP requests (server/app.py's
    # SQLiteSession persists per session_id across calls), so a GENERATION's
    # own `input` can carry far more than just this trace's own exchange --
    # every earlier request's turns too. Scope down to where THIS trace's
    # own request actually starts: the last user turn in the full history
    # whose text matches this trace's own recorded input (last, not first,
    # in case the same question was asked in an earlier unrelated request).
    # The system prompt (always first, if present) is kept regardless.
    request_text = _message_text(trace.get("input"))
    start_index = 0
    if request_text is not None:
        for i in range(len(full_history) - 1, -1, -1):
            msg = full_history[i]
            if msg.get("role") == "user" and _message_text([msg]) == request_text:
                start_index = i
                break

    system_prefix = full_history[:1] if full_history and full_history[0].get("role") == "system" else []
    conversation = system_prefix + full_history[start_index:] if start_index > 0 else full_history

    final_output = trace.get("output")
    if final_output:
        # Avoid duplicating the final reply if it's somehow already the
        # last turn (defensive; the instrumentation doesn't populate the
        # last GENERATION's output today, but don't assume that forever).
        already_present = (
            conversation
            and conversation[-1].get("role") == "assistant"
            and conversation[-1] == (final_output[0] if isinstance(final_output, list) else final_output)
        )
        if not already_present:
            conversation.extend(final_output if isinstance(final_output, list) else [final_output])
    return conversation


@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str) -> dict[str, Any]:
    """Full detail for one trace: the raw trace object, every raw
    observation (the "raw observation view"), a reconstructed conversation
    for readable display, and the tool calls pulled out on their own with
    full arguments/results. Nothing here is dropped or defaulted away --
    missing fields come through as null/None exactly as Langfuse has them.
    """
    trace = _langfuse_get(f"/traces/{trace_id}")
    observations = _langfuse_get("/observations", {"traceId": trace_id}).get("data", [])
    observations.sort(key=lambda o: o.get("startTime") or "")

    tool_calls = [o for o in observations if o.get("type") == "TOOL"]

    return {
        "trace": trace,
        "observations": observations,
        "conversation": _build_conversation(trace, observations),
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Git: mapping the cartwheel.hw_stage / cartwheel.git_commit trace tags back
# to actual code changes, so a stage or commit tag in the UI can answer
# "what's actually different here" -- not just "what's the label".
# ---------------------------------------------------------------------------


def _require_commit_ref(commit: str) -> str:
    if not _COMMIT_RE.match(commit):
        raise HTTPException(status_code=400, detail=f"not a valid commit hash: {commit!r}")
    return commit


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.PIPE
        ).decode(errors="replace")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="git is not installed on this machine")
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"git {' '.join(args)} failed: {exc.stderr.decode(errors='replace')[:300]}",
        )


@app.get("/api/git/commit/{commit}")
def commit_detail(commit: str) -> dict[str, Any]:
    """One commit's metadata, file-change stat, and full diff against its
    immediate git parent -- i.e. exactly the code change that produced this
    commit, which is what a cartwheel.git_commit trace tag actually means."""
    commit = _require_commit_ref(commit)
    fmt = "%H%x1f%h%x1f%an%x1f%aI%x1f%B"
    raw = _git(["show", "-s", f"--format={fmt}", commit])
    full_hash, short_hash, author, date, message = raw.strip("\n").split("\x1f", 4)
    stat = _git(["show", "--stat", "--format=", commit]).strip()
    diff = _git(["show", "--format=", commit])
    parent = None
    try:
        parent = _git(["rev-parse", f"{commit}~1"]).strip()
    except HTTPException:
        pass  # this is the repo's first commit; no parent to name
    return {
        "hash": full_hash,
        "short_hash": short_hash,
        "author": author,
        "date": date,
        "message": message.strip(),
        "parent": parent,
        "stat": stat,
        "diff": diff,
    }


@app.get("/api/git/diff")
def git_diff(from_: str = Query(..., alias="from"), to: str = Query(...)) -> dict[str, Any]:
    """Diff between two arbitrary commits -- used to compare two hw_stage
    tags (e.g. the latest commit seen under hw1 vs. the latest under hw2),
    not just a commit against its own immediate parent."""
    from_ref = _require_commit_ref(from_)
    to_ref = _require_commit_ref(to)
    stat = _git(["diff", "--stat", f"{from_ref}..{to_ref}"]).strip()
    diff = _git(["diff", f"{from_ref}..{to_ref}"])
    return {"from": from_ref, "to": to_ref, "stat": stat, "diff": diff}
