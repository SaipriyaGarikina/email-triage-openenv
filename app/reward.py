"""
reward.py  
--------------------------
Rich, multi-dimensional reward function with sentiment, SLA urgency,
and escalation signals on top of the base grader score.
Base score uses task-specific weights (graders.WEIGHTS — single source of truth):
  easy_triage   : category=0.50, priority=0.25, route=0.25, tags=0.00
  medium_triage : category=0.35, priority=0.35, route=0.20, tags=0.10
  hard_triage   : category=0.25, priority=0.25, route=0.25, tags=0.25
ENHANCEMENTS in v3:
  Sentiment signal   — negative sentiment in email body boosts priority-accuracy reward
  SLA urgency signal — explicit time pressure AND agent correctly assigned urgent/high → +0.03
  Escalation signal  — escalation language AND agent assigned high/urgent priority → +0.03
All bonuses:
  +0.05  speed bonus  (correct in first half of episode)
  +0.05  extra-tag bonus
  +0.03  sentiment-aware priority bonus
  +0.03  SLA-aware priority bonus
  +0.03  escalation-aware priority bonus
All penalties:
  -0.30  invalid email ID
  -0.20  duplicate action
  -0.10  over step limit
  -0.05  empty tags on hard task
  -0.05  contradictory routing
  -0.10  SLA breach
All scores STRICTLY within (0.01, 0.99) — never exactly 0.0 or 1.0.
"""

from __future__ import annotations

from app.models import Action, Category, RouteTarget, Priority, Reward
from app.graders import grade_single_action

# ---------------------------------------------------------------------------
# Sentinel values — strictly inside (0, 1) — matches graders.py
# ---------------------------------------------------------------------------
_MIN_SCORE = 0.01
_MAX_SCORE = 0.99


def _clamp(value: float) -> float:
    """Clamp to strictly (0, 1) — never exactly 0.0 or 1.0."""
    return max(_MIN_SCORE, min(_MAX_SCORE, value))


# ---------------------------------------------------------------------------
# Penalty constants
# ---------------------------------------------------------------------------
PENALTY_INVALID_EMAIL_ID    = 0.30
PENALTY_DUPLICATE_ACTION    = 0.20
PENALTY_OVER_STEPS          = 0.10
PENALTY_EMPTY_TAGS          = 0.05
PENALTY_CONTRADICTORY_ROUTE = 0.05
PENALTY_SLA_BREACH          = 0.10

# ---------------------------------------------------------------------------
# Bonus constants
# ---------------------------------------------------------------------------
BONUS_SPEED          = 0.05
BONUS_EXTRA_TAG      = 0.05
BONUS_SENTIMENT      = 0.03
BONUS_SLA            = 0.03
BONUS_ESCALATION     = 0.03

# ---------------------------------------------------------------------------
# Sentiment signals
# ---------------------------------------------------------------------------
SENTIMENT_NEGATIVE = {
    "frustrated", "angry", "furious", "unacceptable", "disappointed",
    "terrible", "disgusting", "outrageous", "demand", "lawsuit",
    "worst", "horrible", "useless", "ridiculous", "pathetic",
    "getting frustrated", "very upset", "extremely unhappy",
}

# ---------------------------------------------------------------------------
# SLA / urgency signals
# ---------------------------------------------------------------------------
SLA_SIGNALS = {
    "within 24 hours", "within 72 hours", "by eod", "end of day",
    "in 3 hours", "in an hour", "before the deadline", "expires on",
    "procurement deadline", "board review", "before june", "asap",
    "immediately", "right now", "urgent", "time-sensitive",
    "presentation in", "client meeting",
}

# ---------------------------------------------------------------------------
# Escalation signals
# ---------------------------------------------------------------------------
ESCALATION_SIGNALS = {
    "escalate to my bank", "charge back", "chargeback", "contact my lawyer",
    "legal action", "report to", "file a complaint", "consumer protection",
    "better business bureau", "regulatory body", "gdpr complaint",
    "supervisory authority", "no choice but", "forced to",
}

# ---------------------------------------------------------------------------
# Contradictory route pairs
# ---------------------------------------------------------------------------
CONTRADICTORY_ROUTES: set[tuple] = {
    (Category.SPAM,    RouteTarget.BILLING_TEAM),
    (Category.SPAM,    RouteTarget.TECH_TEAM),
    (Category.SPAM,    RouteTarget.SALES_TEAM),
    (Category.SPAM,    RouteTarget.HR_TEAM),
    (Category.SPAM,    RouteTarget.LEGAL_TEAM),
    (Category.HR,      RouteTarget.BILLING_TEAM),
    (Category.HR,      RouteTarget.SALES_TEAM),
    (Category.LEGAL,   RouteTarget.SALES_TEAM),
    (Category.LEGAL,   RouteTarget.BILLING_TEAM),
    (Category.BILLING, RouteTarget.HR_TEAM),
    (Category.BILLING, RouteTarget.LEGAL_TEAM),
}

_HIGH_PRIORITY = {Priority.HIGH, Priority.URGENT}


def _text_contains_any(text: str, signals: set[str]) -> bool:
    """Case-insensitive check: does text contain any signal phrase?"""
    text_lower = text.lower()
    return any(s in text_lower for s in signals)


def compute_reward(
    task_id: str,
    action: Action,
    step: int,
    max_steps: int,
    processed_ids: set[str],
    valid_email_ids: set[str],
    cumulative_scores: list[float],
    sla_hours_remaining: float | None = None,
) -> Reward:
    """
    Compute a rich Reward for a single step action.
    All step_score and cumulative_score values are STRICTLY within (0, 1).
    Never exactly 0.0 or 1.0 — uses _MIN_SCORE=0.01 and _MAX_SCORE=0.99.
    """
    total_penalty = 0.0
    total_bonus   = 0.0
    feedback_parts: list[str] = []

    # ------------------------------------------------------------------
    # PENALTY 1: Invalid email ID — immediate min score, early return
    # ------------------------------------------------------------------
    if action.email_id not in valid_email_ids:
        total_penalty += PENALTY_INVALID_EMAIL_ID
        feedback_parts.append(
            f"❌ PENALTY -{PENALTY_INVALID_EMAIL_ID}: "
            f"Email ID '{action.email_id}' does not exist in task '{task_id}'."
        )
        # FIX: Use _MIN_SCORE (0.01) instead of 0.0 — never return exactly 0.0
        step_score = _MIN_SCORE
        all_scores = cumulative_scores + [step_score]
        cumulative = _clamp(round(sum(all_scores) / len(all_scores), 4))
        return Reward(
            step_score=step_score,
            cumulative_score=cumulative,
            penalty=round(total_penalty, 4),
            breakdown={},
            feedback=" ".join(feedback_parts),
        )

    # ------------------------------------------------------------------
    # PENALTY 2: Duplicate action
    # ------------------------------------------------------------------
    is_duplicate = action.email_id in processed_ids
    if is_duplicate:
        total_penalty += PENALTY_DUPLICATE_ACTION
        feedback_parts.append(
            f"⚠️ PENALTY -{PENALTY_DUPLICATE_ACTION}: "
            f"Email '{action.email_id}' already processed."
        )

    # ------------------------------------------------------------------
    # PENALTY 3: Over step limit
    # ------------------------------------------------------------------
    if step >= max_steps:
        total_penalty += PENALTY_OVER_STEPS
        feedback_parts.append(
            f"⚠️ PENALTY -{PENALTY_OVER_STEPS}: Step {step} exceeds max_steps {max_steps}."
        )

    # ------------------------------------------------------------------
    # PENALTY 4: Contradictory routing
    # ------------------------------------------------------------------
    if (action.category, action.route_to) in CONTRADICTORY_ROUTES:
        total_penalty += PENALTY_CONTRADICTORY_ROUTE
        feedback_parts.append(
            f"⚠️ PENALTY -{PENALTY_CONTRADICTORY_ROUTE}: "
            f"Contradictory — '{action.category}' routed to '{action.route_to}'."
        )

    # ------------------------------------------------------------------
    # PENALTY 5: SLA breach
    # ------------------------------------------------------------------
    _LOW_PRIORITY = {Priority.LOW, Priority.MEDIUM}
    if (
        sla_hours_remaining is not None
        and sla_hours_remaining < 2.0
        and action.priority in _LOW_PRIORITY
        and not is_duplicate
    ):
        total_penalty += PENALTY_SLA_BREACH
        feedback_parts.append(
            f"🚨 PENALTY -{PENALTY_SLA_BREACH}: SLA breach — "
            f"only {sla_hours_remaining:.1f}h remaining but priority is '{action.priority.value}'. "
            f"Should be HIGH or URGENT."
        )

    # ------------------------------------------------------------------
    # GRADE + BONUSES (skip if duplicate)
    # ------------------------------------------------------------------
    if not is_duplicate:
        graded    = grade_single_action(task_id, action)
        raw_score = graded["step_score"]  # already in (0.01, 0.99) from graders.py
        breakdown = graded["breakdown"]
        grader_fb = graded["feedback"]

        # Fetch email body for signal detection
        from app.tasks import TASK_REGISTRY
        task_data  = TASK_REGISTRY.get(task_id, {})
        gt_email   = task_data.get("ground_truth", {}).get(action.email_id, {})
        email_obj  = next(
            (e for e in task_data.get("emails", []) if e.email_id == action.email_id),
            None,
        )
        email_body = (email_obj.body + " " + email_obj.subject) if email_obj else ""

        # PENALTY: Empty tags on hard task
        if task_id == "hard_triage" and not action.tags:
            total_penalty += PENALTY_EMPTY_TAGS
            feedback_parts.append(
                f"⚠️ PENALTY -{PENALTY_EMPTY_TAGS}: Tags required for hard_triage."
            )

        # BONUS 1: Speed bonus
        halfway = max(1, max_steps // 2)
        if step < halfway and raw_score >= 0.7:
            total_bonus += BONUS_SPEED
            feedback_parts.append(
                f"🚀 BONUS +{BONUS_SPEED}: Fast & correct (step {step+1}/{max_steps})."
            )

        # BONUS 2: Extra tag bonus
        expected_tags  = {t.lower().strip() for t in gt_email.get("tags", [])}
        predicted_tags = {t.lower().strip() for t in action.tags}
        if expected_tags and len(predicted_tags & expected_tags) > len(expected_tags) * 0.5:
            if len(predicted_tags) > len(expected_tags):
                total_bonus += BONUS_EXTRA_TAG
                feedback_parts.append(
                    f"🏷️ BONUS +{BONUS_EXTRA_TAG}: Extra relevant tags identified."
                )

        # BONUS 3: Sentiment-aware priority
        gt_priority = gt_email.get("priority")
        if (
            _text_contains_any(email_body, SENTIMENT_NEGATIVE)
            and action.priority in _HIGH_PRIORITY
            and gt_priority in _HIGH_PRIORITY
        ):
            total_bonus += BONUS_SENTIMENT
            feedback_parts.append(
                f"😤 BONUS +{BONUS_SENTIMENT}: Correctly detected negative customer sentiment → high priority."
            )

        # BONUS 4: SLA-aware priority
        if (
            _text_contains_any(email_body, SLA_SIGNALS)
            and action.priority in _HIGH_PRIORITY
            and gt_priority in _HIGH_PRIORITY
        ):
            total_bonus += BONUS_SLA
            feedback_parts.append(
                f"⏰ BONUS +{BONUS_SLA}: Correctly detected SLA/deadline pressure → high priority."
            )

        # BONUS 5: Escalation-aware priority
        if (
            _text_contains_any(email_body, ESCALATION_SIGNALS)
            and action.priority in _HIGH_PRIORITY
            and gt_priority in _HIGH_PRIORITY
        ):
            total_bonus += BONUS_ESCALATION
            feedback_parts.append(
                f"⚡ BONUS +{BONUS_ESCALATION}: Correctly detected escalation threat → high priority."
            )

    else:
        # FIX: Use _MIN_SCORE (0.01) instead of 0.0 for duplicate raw_score
        raw_score = _MIN_SCORE
        breakdown = {}
        grader_fb = "No grading (duplicate action)."

    # ------------------------------------------------------------------
    # Final score: raw - penalties + bonuses
    # FIX: Clamp using _clamp() → strictly (0.01, 0.99), never 0.0 or 1.0
    # ------------------------------------------------------------------
    raw_final  = raw_score - total_penalty + total_bonus
    step_score = _clamp(round(raw_final, 4))

    feedback_parts.insert(0, grader_fb)
    if total_penalty > 0:
        feedback_parts.append(f"| Penalties: -{total_penalty:.2f}")
    if total_bonus > 0:
        feedback_parts.append(f"| Bonuses: +{total_bonus:.2f}")
    feedback_parts.append(f"| Final step score: {step_score:.4f}")

    all_scores = cumulative_scores + [step_score]
    cumulative = _clamp(round(sum(all_scores) / len(all_scores), 4))

    return Reward(
        step_score=step_score,
        cumulative_score=cumulative,
        penalty=round(total_penalty, 4),
        breakdown=breakdown,
        feedback=" ".join(feedback_parts),
    )
