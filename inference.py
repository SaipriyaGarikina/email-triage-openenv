"""
inference.py
------------
OpenEnv RL Challenge — Email Triage Environment
Inference script that runs an LLM agent through all 3 tasks.
Output format (strict):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
Environment Variables:
  API_BASE_URL  : LLM API base URL   (default: https://api.openai.com/v1)
  MODEL_NAME    : Model identifier   (default: gpt-4o-mini)
  HF_TOKEN      : HuggingFace token  (REQUIRED — no default)
SCORE BOUNDS: All reward values printed are strictly within (0.01, 0.99).
  Never exactly 0.0, 0.00, 1.0, or 1.00 — validator rejects these.
"""

from __future__ import annotations

import os
import sys
import json
import httpx

from openai import OpenAI

# ---------------------------------------------------------------------------
# Score sentinel constants — NEVER print exactly 0.0 or 1.0
# ---------------------------------------------------------------------------
_MIN_SCORE = 0.01
_MAX_SCORE = 0.99


def _clamp(v: float) -> float:
    """Force score strictly inside (0, 1). Never 0.0 or 1.0."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _MIN_SCORE
    if v <= 0.0:
        return _MIN_SCORE
    if v >= 1.0:
        return _MAX_SCORE
    return v


# ---------------------------------------------------------------------------
# Environment variables (with defaults where required)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required but not set.")

# ---------------------------------------------------------------------------
# OpenAI client (using HF_TOKEN as api_key per challenge spec)
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Environment base URL (the running HF Space)
# ---------------------------------------------------------------------------
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://priya8-email-triage-openenv.hf.space"
).rstrip("/")

# ---------------------------------------------------------------------------
# Tasks to run
# ---------------------------------------------------------------------------
TASKS = ["easy_triage", "medium_triage", "hard_triage"]
ENV_NAME = "email-triage-openenv"

# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert email triage agent for a SaaS company.
For each incoming email, analyse it and respond with ONLY valid JSON (no explanation, no markdown):
{
  "category": "spam|support|billing|sales|hr|legal|general",
  "priority": "low|medium|high|urgent",
  "route_to": "sales_team|tech_team|billing_team|hr_team|legal_team|trash|inbox",
  "tags": ["tag1", "tag2"],
  "notes": "brief reasoning"
}
STRICT RULES:
- spam     → always route to trash, priority low
- support  → tech_team. urgent if system down, high if blocking work
- billing  → billing_team. urgent if time-critical
- sales    → sales_team. high if enterprise/large deal
- hr       → hr_team. high if harassment/complaint, low if job application
- legal    → legal_team. urgent if data breach/GDPR, high if contract dispute
- general  → inbox, priority low
- Never route spam to a real team
- Data breaches are LEGAL not support
- Job applications are HR not support
- Contract disputes are LEGAL not sales
"""


# ---------------------------------------------------------------------------
# Helper: call environment API
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    """Call POST /reset and return the full response dict."""
    r = httpx.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(session_id: str, action: dict) -> dict:
    """Call POST /step and return the full response dict."""
    r = httpx.post(
        f"{ENV_BASE_URL}/step",
        params={"session_id": session_id},
        json={"action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Helper: ask the LLM to triage one email
# ---------------------------------------------------------------------------
def llm_triage(email_id: str, subject: str, sender: str, body: str) -> dict:
    """
    Send one email to the LLM and parse the JSON triage decision.
    Returns a dict with category, priority, route_to, tags, notes.
    """
    user_message = (
        f"Email ID : {email_id}\n"
        f"Subject  : {subject}\n"
        f"Sender   : {sender}\n\n"
        f"Body:\n{body}\n\n"
        f"Respond ONLY with valid JSON."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        return {
            "category": parsed.get("category", "general"),
            "priority":  parsed.get("priority",  "medium"),
            "route_to":  parsed.get("route_to",  "inbox"),
            "tags":      parsed.get("tags",      []),
            "notes":     parsed.get("notes",     ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "category": "general",
            "priority":  "medium",
            "route_to":  "inbox",
            "tags":      [],
            "notes":     f"Parse error on raw response: {raw[:100]}",
        }


# ---------------------------------------------------------------------------
# Run one full task episode
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> None:
    """
    Run one complete episode for a task.
    Prints [START], [STEP]s, and [END] to stdout.
    All reward values are strictly within (0.01, 0.99).
    """
    rewards: list[float] = []
    steps_taken: int     = 0
    success: bool        = False
    last_error: str      = "null"

    print(
        f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}",
        flush=True,
    )

    try:
        # ── Reset environment ──────────────────────────────────────────────
        reset_data  = env_reset(task_id)
        session_id  = reset_data["session_id"]
        observation = reset_data["observation"]
        done        = observation.get("done", False)

        # ── Episode loop ───────────────────────────────────────────────────
        while not done:
            steps_taken += 1

            email_id = observation["email_id"]
            subject  = observation.get("subject", "")
            sender   = observation.get("sender",  "")
            body     = observation.get("body",    "")

            decision = llm_triage(email_id, subject, sender, body)

            action = {
                "email_id": email_id,
                "category": decision["category"],
                "priority":  decision["priority"],
                "route_to":  decision["route_to"],
                "tags":      decision["tags"],
                "notes":     decision["notes"],
            }

            action_str = (
                f"triage(email_id={email_id!r},"
                f"category={decision['category']!r},"
                f"priority={decision['priority']!r},"
                f"route_to={decision['route_to']!r})"
            )

            # ── Call environment step ──────────────────────────────────────
            step_data   = env_step(session_id, action)

            # CRITICAL: clamp reward — never print exactly 0.0 or 1.0
            raw_reward  = step_data["reward"]["step_score"]
            reward_val  = _clamp(raw_reward)

            done        = step_data["done"]
            observation = step_data.get("observation") or {}
            last_error  = "null"

            rewards.append(reward_val)

            print(
                f"[STEP] step={steps_taken} "
                f"action={action_str} "
                f"reward={reward_val:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={last_error}",
                flush=True,
            )

        # Episode completed normally
        cumulative = _clamp(sum(rewards) / len(rewards)) if rewards else _MIN_SCORE
        success    = cumulative >= 0.70

    except Exception as exc:
        last_error = str(exc).replace("\n", " ")[:200]
        success    = False

        # Emit a STEP line on crash — use _MIN_SCORE, never 0.00
        steps_taken = max(steps_taken, 1)
        if len(rewards) < steps_taken:
            rewards.append(_MIN_SCORE)

        print(
            f"[STEP] step={steps_taken} "
            f"action=error() "
            f"reward={_MIN_SCORE:.2f} "
            f"done=true "
            f"error={last_error!r}",
            flush=True,
        )

    # ── Print END line ─────────────────────────────────────────────────────
    # CRITICAL: all reward values clamped — never 0.00 or 1.00 in output
    clamped_rewards = [_clamp(r) for r in rewards] if rewards else [_MIN_SCORE]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped_rewards)

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps_taken} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for task_id in TASKS:
        run_task(task_id)
        print("", flush=True)
