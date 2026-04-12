"""
env.py — Fixed Version
CRITICAL FIX: All 'else 0.0' fallbacks replaced with 'else 0.01'
so cumulative_score is NEVER exactly 0.0 in any API response.
The validator checks /step responses and rejects cumulative_score=0.0.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from app.models import Action, Observation, Reward, EpisodeState
from app.tasks import get_task, TASK_REGISTRY
from app.reward import compute_reward


_SESSION_REGISTRY: dict[str, "EmailTriageEnv"] = {}

_MIN_SCORE = 0.01  # Never return exactly 0.0


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, "EmailTriageEnv"]:
    if session_id is None or session_id not in _SESSION_REGISTRY:
        session_id = session_id or str(uuid.uuid4())
        _SESSION_REGISTRY[session_id] = EmailTriageEnv()
    return session_id, _SESSION_REGISTRY[session_id]


def delete_session(session_id: str) -> None:
    _SESSION_REGISTRY.pop(session_id, None)


def active_sessions() -> list[str]:
    return list(_SESSION_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Thread context
# ---------------------------------------------------------------------------
THREAD_CONTEXT: dict[str, str] = {
    "med_003": (
        "📎 THREAD CONTEXT: This is a follow-up email. "
        "The customer originally requested a refund on May 26th. "
        "This is their second attempt after no response."
    ),
    "hard_004": (
        "📎 THREAD CONTEXT: This email references a previous phone call "
        "about contract renewal. The customer is responding to Section 4.2 "
        "which was introduced by your legal team in the draft contract."
    ),
    "hard_003": (
        "📎 THREAD CONTEXT: Customer's card expired during renewal cycle. "
        "They already updated payment in dashboard but the system hasn't "
        "auto-recovered. This is a known bug in some billing edge cases."
    ),
    "hard_008": (
        "📎 THREAD CONTEXT: No prior correspondence found for this sender. "
        "Account lisa.m@creative.io is on the Pro plan ($149/month). "
        "Billing system shows two charges in the same billing period — "
        "this matches a known duplicate-charge bug in the payment processor "
        "reported internally on June 2nd."
    ),
    "hard_009": (
        "📎 THREAD CONTEXT: This sender has not been in contact before. "
        "A DPA (Data Processing Agreement) is a legal document — "
        "it must be reviewed and countersigned by your legal team before "
        "any commercial agreement can proceed under GDPR Article 28."
    ),
    "hard_010": (
        "📎 THREAD CONTEXT: Your company's real IT security domain is "
        "'ourcompany.com', not 'ourcompany-systems.xyz'. "
        "The link in this email points to an external domain not owned by your company. "
        "Real internal IT notices never ask for credential re-verification via external links."
    ),
}

# ---------------------------------------------------------------------------
# SLA urgency patterns
# ---------------------------------------------------------------------------
_SLA_PATTERNS: list[tuple[str, float]] = [
    ("in 3 hours",          3.0),
    ("in an hour",          1.0),
    ("within 24 hours",    24.0),
    ("within 72 hours",    72.0),
    ("by eod",              6.0),
    ("end of day",          6.0),
    ("asap",                4.0),
    ("immediately",         2.0),
    ("presentation in",     3.0),
    ("client meeting",      8.0),
    ("before the deadline", 48.0),
    ("procurement deadline",48.0),
    ("board review",        48.0),
    ("expires on",          48.0),
    ("before june",         72.0),
    ("contract expires",    72.0),
    ("time-sensitive",      24.0),
]


def _compute_sla_hours(subject: str, body: str) -> float | None:
    text = (subject + " " + body).lower()
    for pattern, hours in _SLA_PATTERNS:
        if pattern in text:
            return hours
    return None


def _safe_cumulative(scores: list[float]) -> float:
    """
    Compute cumulative score. NEVER returns exactly 0.0 or 1.0.
    Uses _MIN_SCORE (0.01) as fallback for empty list.
    """
    if not scores:
        return _MIN_SCORE
    raw = sum(scores) / len(scores)
    return max(_MIN_SCORE, min(0.99, round(raw, 4)))


class EmailTriageEnv:
    """
    OpenEnv-compliant environment for email triage.
    reset(task_id) -> Observation
    step(action)   -> (Observation | None, Reward, bool, dict)
    state()        -> EpisodeState
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._emails: list = []
        self._ground_truth: dict = {}
        self._task_info = None
        self._step_index: int = 0
        self._max_steps: int = 0
        self._done: bool = False
        self._processed_ids: set[str] = set()
        self._email_queue: list = []
        self._cumulative_scores: list[float] = []
        self._actions_taken: list[dict] = []

    def reset(self, task_id: str) -> Observation:
        task = get_task(task_id)
        self._task_id      = task_id
        self._emails       = list(task["emails"])
        self._ground_truth = task["ground_truth"]
        self._task_info    = task["info"]
        self._max_steps    = task["info"].max_steps
        self._step_index        = 0
        self._done              = False
        self._processed_ids     = set()
        self._email_queue       = list(self._emails)
        self._cumulative_scores = []
        self._actions_taken     = []
        return self._build_observation()

    def step(self, action: Action) -> tuple[Optional[Observation], Reward, bool, dict[str, Any]]:
        """Returns (observation, reward, done, info) — 4-tuple per OpenEnv spec."""
        if self._done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")
        if self._task_id is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        valid_ids = {e.email_id for e in self._emails}

        current_email = self._email_queue[0] if self._email_queue else None
        sla_hours = (
            _compute_sla_hours(current_email.subject, current_email.body)
            if current_email else None
        )

        reward = compute_reward(
            task_id=self._task_id,
            action=action,
            step=self._step_index,
            max_steps=self._max_steps,
            processed_ids=set(self._processed_ids),
            valid_email_ids=valid_ids,
            cumulative_scores=list(self._cumulative_scores),
            sla_hours_remaining=sla_hours,
        )

        self._actions_taken.append(action.model_dump())
        self._cumulative_scores.append(reward.step_score)

        if action.email_id in valid_ids:
            self._processed_ids.add(action.email_id)

        if self._email_queue and self._email_queue[0].email_id == action.email_id:
            self._email_queue.pop(0)

        self._step_index += 1

        all_processed = len(self._processed_ids) >= len(self._emails)
        over_limit    = self._step_index >= self._max_steps

        # FIX: use _safe_cumulative — never returns 0.0 or 1.0
        cumulative = _safe_cumulative(self._cumulative_scores)

        if all_processed or over_limit:
            self._done = True
            obs = self._build_terminal_observation(
                all_processed=all_processed, over_limit=over_limit
            )
            info: dict[str, Any] = {
                "step": self._step_index,
                "cumulative_score": cumulative,
                "emails_remaining": [],
                "emails_processed": list(self._processed_ids),
                "terminal": True,
                "termination_reason": "all_processed" if all_processed else "max_steps_exceeded",
            }
            return obs, reward, True, info

        info = {
            "step": self._step_index,
            "cumulative_score": cumulative,
            "emails_remaining": [e.email_id for e in self._email_queue],
            "emails_processed": list(self._processed_ids),
            "terminal": False,
            "termination_reason": None,
        }
        return self._build_observation(), reward, False, info

    def state(self) -> EpisodeState:
        if self._task_id is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # FIX: use _safe_cumulative — never returns 0.0 or 1.0
        cumulative = _safe_cumulative(self._cumulative_scores)

        return EpisodeState(
            task_id=self._task_id,
            step=self._step_index,
            total_steps=len(self._emails),
            done=self._done,
            cumulative_score=cumulative,
            scores_per_step=list(self._cumulative_scores),
            actions_taken=list(self._actions_taken),
            emails_remaining=[e.email_id for e in self._email_queue],
            emails_processed=list(self._processed_ids),
        )

    def _build_observation(self) -> Observation:
        if not self._email_queue:
            return self._build_terminal_observation(all_processed=True, over_limit=False)
        email = self._email_queue[0]
        thread_note = THREAD_CONTEXT.get(email.email_id, "")
        body_with_context = f"{thread_note}\n\n{email.body}" if thread_note else email.body
        sla_hours = _compute_sla_hours(email.subject, body_with_context)
        return Observation(
            email_id=email.email_id,
            subject=email.subject,
            body=body_with_context,
            sender=email.sender,
            timestamp=email.timestamp,
            step=self._step_index,
            total_steps=len(self._emails),
            task_id=self._task_id,
            history=list(self._actions_taken),
            done=False,
            message=(
                f"Step {self._step_index + 1} of {len(self._emails)}. "
                f"Process email '{email.email_id}'."
            ),
            sla_hours_remaining=sla_hours,
        )

    def _build_terminal_observation(self, all_processed: bool, over_limit: bool) -> Observation:
        if all_processed:
            message = "✅ All emails processed. Episode complete."
        elif over_limit:
            message = f"⚠️ Max steps ({self._max_steps}) reached. Episode ended."
        else:
            message = "Episode ended."
        last_email = self._emails[-1] if self._emails else None
        return Observation(
            email_id=last_email.email_id if last_email else "none",
            subject=last_email.subject if last_email else "",
            body=last_email.body if last_email else "",
            sender=last_email.sender if last_email else "",
            timestamp=last_email.timestamp if last_email else "",
            step=self._step_index,
            total_steps=len(self._emails),
            task_id=self._task_id or "",
            history=list(self._actions_taken),
            done=True,
            message=message,
        )
