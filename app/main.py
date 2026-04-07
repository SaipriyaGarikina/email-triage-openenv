"""
main.py  (v2 — Fixed)
----------------------
FastAPI application with full session-based multi-user support.

FIXES APPLIED:
  [Fix #1] /step unpacks 4-tuple from env.step()
  [Fix #4] /reset returns session_id as structured field
  [Fix #5] /tasks returns action_schema per OpenEnv spec
  [Fix #6] asyncio.get_running_loop() replaces deprecated get_event_loop()

Endpoints:
  POST /reset          → Start new episode (returns session_id)
  POST /step           → Send action (requires session_id)
  GET  /state          → Get episode state (requires session_id)
  GET  /tasks          → List all tasks + action schema
  POST /grader         → Offline grading
  POST /baseline       → Run GPT-4o-mini baseline
  GET  /sessions       → List active sessions
  DELETE /sessions/{id}→ Delete a session
  GET  /health         → Health check
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

from app.env import (
    EmailTriageEnv,
    get_or_create_session,
    delete_session,
    active_sessions,
)
from app.models import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateResponse,
    GraderRequest, GraderResponse,
    BaselineRequest, BaselineResponse,
    TaskInfo, Observation,
    TasksResponse, ActionFieldSchema,
)
from app.tasks import list_tasks, get_task
from app.graders import grade_full_episode

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Triage OpenEnv  v2",
    description=(
        "Production-ready OpenEnv environment simulating AI email triage. "
        "Multi-session, session-safe. "
        "Agent reads incoming emails and classifies, prioritizes, tags, and routes them."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
def reset(
    request: ResetRequest,
    session_id: Optional[str] = Query(default=None, description="Reuse an existing session or leave blank to create new"),
):
    """
    Reset environment and start a new episode.

    Returns a **session_id** — pass this as a query param to all subsequent /step and /state calls.

    task_id options: easy_triage | medium_triage | hard_triage
    """
    try:
        sid, env = get_or_create_session(session_id)
        observation = env.reset(request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task_info = get_task(request.task_id)["info"]
    # FIX #4: session_id is a top-level structured field
    return ResetResponse(
        session_id=sid,
        observation=observation,
        task_info=task_info,
        message=f"Session '{sid}' reset for task '{request.task_id}'. Use session_id in /step and /state.",
    )


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------

@app.post("/step", response_model=StepResponse, tags=["Environment"])
def step(
    request: StepRequest,
    session_id: str = Query(..., description="Session ID returned from /reset"),
):
    """
    Submit one action and receive the next observation + reward.
    Requires a valid session_id from /reset.
    """
    if session_id not in active_sessions():
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /reset first.",
        )

    _, env = get_or_create_session(session_id)

    if env._task_id is None:
        raise HTTPException(status_code=400, detail="Session not initialized. Call /reset first.")

    if env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode already done. Call POST /reset to start a new episode.",
        )

    try:
        # FIX #1: unpack 4-tuple per OpenEnv spec
        next_obs, reward, done, step_info = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=next_obs,
        reward=reward,
        done=done,
        info={"session_id": session_id, **step_info},
    )


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------

@app.get("/state", response_model=StateResponse, tags=["Environment"])
def state(
    session_id: str = Query(..., description="Session ID from /reset"),
):
    """Get the full current episode state for a session."""
    if session_id not in active_sessions():
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    _, env = get_or_create_session(session_id)

    if env._task_id is None:
        raise HTTPException(status_code=400, detail="Session not initialized.")

    episode_state = env.state()
    current_obs = env._build_observation() if not env._done and env._email_queue else None

    return StateResponse(
        episode_state=episode_state,
        current_observation=current_obs,
    )


# ---------------------------------------------------------------------------
# GET /tasks  — FIX #5: returns task list + action schema per spec
# ---------------------------------------------------------------------------

@app.get("/tasks", response_model=TasksResponse, tags=["Tasks"])
def tasks():
    """
    List all available tasks with descriptions and difficulty levels.

    Also returns the **action_schema** — the complete list of fields
    required in every step action (per OpenEnv spec requirement).
    """
    action_schema = [
        ActionFieldSchema(
            name="email_id",
            type="string",
            required=True,
            values=None,
            description="ID of the email being acted upon. Must match the email_id in the current Observation.",
        ),
        ActionFieldSchema(
            name="category",
            type="enum",
            required=True,
            values=["spam", "support", "billing", "sales", "hr", "legal", "general"],
            description="Email category classification.",
        ),
        ActionFieldSchema(
            name="priority",
            type="enum",
            required=True,
            values=["low", "medium", "high", "urgent"],
            description="Assigned priority level. Partial credit given for one-level-off predictions.",
        ),
        ActionFieldSchema(
            name="route_to",
            type="enum",
            required=True,
            values=["sales_team", "tech_team", "billing_team", "hr_team", "legal_team", "trash", "inbox"],
            description="Which team or folder to route the email to.",
        ),
        ActionFieldSchema(
            name="tags",
            type="list[string]",
            required=False,
            values=None,
            description="Topic tags, e.g. ['refund', 'angry_customer']. Scored with F1. Required for hard_triage.",
        ),
        ActionFieldSchema(
            name="notes",
            type="string",
            required=False,
            values=None,
            description="Optional free-text reasoning notes. Not graded — for agent transparency only.",
        ),
    ]
    return TasksResponse(tasks=list_tasks(), action_schema=action_schema)


# ---------------------------------------------------------------------------
# POST /grader
# ---------------------------------------------------------------------------

@app.post("/grader", response_model=GraderResponse, tags=["Grader"])
def grader(request: GraderRequest):
    """
    Run the deterministic grader on a set of actions offline.
    No session needed. Useful for batch evaluation.
    """
    try:
        result = grade_full_episode(request.task_id, request.actions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return GraderResponse(**result)


# ---------------------------------------------------------------------------
# POST /baseline
# ---------------------------------------------------------------------------

@app.post("/baseline", response_model=BaselineResponse, tags=["Baseline"])
async def baseline(request: BaselineRequest):
    """
    Run GPT-4o-mini baseline agent on specified tasks.
    Requires OPENAI_API_KEY environment variable (set as a Space Secret).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set. Add it as a Space Secret in HF Settings.",
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(status_code=500, detail="openai package not installed.")

    client = OpenAI(api_key=api_key)
    results = []

    for task_id in request.task_ids:
        try:
            result = await _run_baseline_task(client, task_id)
            results.append(result)
        except Exception as e:
            results.append({"task_id": task_id, "error": str(e), "total_score": 0.0})

    summary = {r["task_id"]: r.get("total_score", 0.0) for r in results}
    return BaselineResponse(results=results, summary=summary)


async def _run_baseline_task(client: Any, task_id: str) -> dict:
    """Run GPT-4o-mini on one task."""
    from app.models import Action, Category, Priority, RouteTarget

    local_env = EmailTriageEnv()
    obs = local_env.reset(task_id)
    done = False
    all_actions = []

    system_prompt = """You are an expert email triage assistant for a SaaS company.

For each incoming email, analyse it carefully and respond with ONLY valid JSON:
{
  "category": one of [spam, support, billing, sales, hr, legal, general],
  "priority": one of [low, medium, high, urgent],
  "route_to": one of [sales_team, tech_team, billing_team, hr_team, legal_team, trash, inbox],
  "tags": list of 2-4 relevant lowercase keyword tags,
  "notes": short reasoning string
}

IMPORTANT RULES:
- spam emails MUST go to trash
- data breaches and contracts MUST be legal + urgent/high
- harassment complaints MUST go to hr_team
- production outages MUST be urgent
- job applications MUST go to hr_team
- enterprise deals MUST go to sales_team
- Never route spam to any real team
- Think carefully before assigning priority
- A prospect asking for a DPA (Data Processing Agreement) before signing → legal + legal_team + HIGH
- Emails from lookalike domains (e.g. ourcompany-systems.xyz) or asking for credentials via external links → spam + trash + low
- Polite tone does NOT mean low priority — check for billing disputes hidden behind friendly language
"""

    while not done:
        user_prompt = (
            f"Email ID: {obs.email_id}\n"
            f"Subject: {obs.subject}\n"
            f"Sender: {obs.sender}\n"
            f"Body:\n{obs.body}\n\n"
            f"Respond with ONLY valid JSON."
        )

        # FIX #6: use get_running_loop() — get_event_loop() is deprecated in Python 3.10+
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                max_tokens=300,
            )
        )

        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
            action = Action(
                email_id=obs.email_id,
                category=Category(parsed.get("category", "general")),
                priority=Priority(parsed.get("priority", "medium")),
                route_to=RouteTarget(parsed.get("route_to", "inbox")),
                tags=parsed.get("tags", []),
                notes=parsed.get("notes", ""),
            )
        except Exception:
            action = Action(
                email_id=obs.email_id,
                category=Category.GENERAL,
                priority=Priority.MEDIUM,
                route_to=RouteTarget.INBOX,
                tags=[],
                notes="Parse error — defaulting.",
            )

        all_actions.append(action)
        # FIX #1: unpack 4-tuple
        next_obs, reward, done, _ = local_env.step(action)
        obs = next_obs

    result = grade_full_episode(task_id, all_actions)
    result["actions"] = [a.model_dump() for a in all_actions]
    result["model_used"] = "gpt-4o-mini"
    return result


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------

@app.get("/sessions", tags=["Sessions"])
def sessions():
    """List all currently active session IDs."""
    return {"active_sessions": active_sessions(), "count": len(active_sessions())}


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------

@app.delete("/sessions/{session_id}", tags=["Sessions"])
def remove_session(session_id: str):
    """Delete a specific session and free its memory."""
    if session_id not in active_sessions():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    delete_session(session_id)
    return {"message": f"Session '{session_id}' deleted."}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "active_sessions": len(active_sessions()),
    }


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": [
            "POST /reset        → start episode, returns session_id",
            "POST /step         → submit action (requires ?session_id=...)",
            "GET  /state        → get state (requires ?session_id=...)",
            "GET  /tasks        → list tasks + action schema",
            "POST /grader       → offline grading",
            "POST /baseline     → run GPT-4o-mini baseline",
            "GET  /sessions     → list active sessions",
            "DELETE /sessions/{id}",
            "GET  /health",
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        reload=False,
    )
