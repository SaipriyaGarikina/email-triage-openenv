"""
reward.py — Final Fixed Version
All scores strictly within (0.01, 0.99). Never 0.0 or 1.0.
_clamp() used everywhere. No raw 0.0 literals.
"""

from __future__ import annotations

from app.models import Action, Category, RouteTarget, Priority, Reward
from app.graders import grade_single_action

_MIN_SCORE = 0.01
_MAX_SCORE = 0.99


def _clamp(value: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _MIN_SCORE
    if f != f:
        return _MIN_SCORE
    return max(_MIN_SCORE, min(_MAX_SCORE, f))


PENALTY_INVALID_EMAIL_ID    = 0.30
PENALTY_DUPLICATE_ACTION    = 0.20
PENALTY_OVER_STEPS          = 0.10
PENALTY_EMPTY_TAGS          = 0.05
PENALTY_CONTRADICTORY_ROUTE = 0.05
PENALTY_SLA_BREACH          = 0.10

BONUS_SPEED      = 0.05
BONUS_EXTRA_TAG  = 0.05
BONUS_SENTIMENT  = 0.03
BONUS_SLA        = 0.03
BONUS_ESCALATION = 0.03

SENTIMENT_NEGATIVE = {
    "frustrated", "angry", "furious", "unacceptable", "disappointed",
    "terrible", "disgusting", "outrageous", "demand", "lawsuit",
    "worst", "horrible", "useless", "ridiculous", "pathetic",
    "getting frustrated", "very upset", "extremely unhappy",
}

SLA_SIGNALS = {
    "within 24 hours", "within 72 hours", "by eod", "end of day",
    "in 3 hours", "in an hour", "before the deadline", "expires on",
    "procurement deadline", "board review", "before june", "asap",
    "immediately", "right now", "urgent", "time-sensitive",
    "presentation in", "client meeting",
}

ESCALATION_SIGNALS = {
    "escalate to my bank", "charge back", "chargeback", "contact my lawyer",
    "legal action", "report to", "file a complaint", "consumer protection",
    "better business bureau", "regulatory body", "gdpr complaint",
    "supervisory authority", "no choice but", "forced to",
}

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
_LOW_PRIORITY  = {Priority.LOW,  Priority.MEDIUM}


def _text_contains_any(text: str, signals: set[str]) -> bool:
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
    Compute reward. All scores strictly within (0.01, 0.99).
    Never returns 0.0 or 1.0 — enforced by _clamp() at every exit point.
    """
    total_penalty  = 0.0
    total_bonus    = 0.0
    feedback_parts: list[str] = []

    # PENALTY 1: Invalid email ID — early return with _MIN_SCORE
    if action.email_id not in valid_email_ids:
        total_penalty += PENALTY_INVALID_EMAIL_ID
        step_score = _MIN_SCORE   # never 0.0
        all_scores = cumulative_scores + [step_score]
        cumulative = _clamp(sum(all_scores) / len(all_scores))
        return Reward(
            step_score=step_score,
            cumulative_score=cumulative,
            penalty=round(total_penalty, 4),
            breakdown={},
            feedback=f"❌ PENALTY: Email '{action.email_id}' not in task '{task_id}'.",
        )

    # PENALTY 2: Duplicate
    is_duplicate = action.email_id in processed_ids
    if is_duplicate:
        total_penalty += PENALTY_DUPLICATE_ACTION
        feedback_parts.append(f"⚠️ PENALTY: Duplicate '{action.email_id}'.")

    # PENALTY 3: Over steps
    if step >= max_steps:
        total_penalty += PENALTY_OVER_STEPS
        feedback_parts.append(f"⚠️ PENALTY: Step {step} > max {max_steps}.")

    # PENALTY 4: Contradictory routing
    if (action.category, action.route_to) in CONTRADICTORY_ROUTES:
        total_penalty += PENALTY_CONTRADICTORY_ROUTE
        feedback_parts.append(f"⚠️ PENALTY: Contradictory route.")

    # PENALTY 5: SLA breach
    if (
        sla_hours_remaining is not None
        and sla_hours_remaining < 2.0
        and action.priority in _LOW_PRIORITY
        and not is_duplicate
    ):
        total_penalty += PENALTY_SLA_BREACH
        feedback_parts.append(f"🚨 PENALTY: SLA breach.")

    # GRADE + BONUSES
    if not is_duplicate:
        graded    = grade_single_action(task_id, action)
        raw_score = _clamp(graded["step_score"])  # always safe
        breakdown = graded["breakdown"]
        feedback_parts.insert(0, graded["feedback"])

        from app.tasks import TASK_REGISTRY
        task_data  = TASK_REGISTRY.get(task_id, {})
        gt_email   = task_data.get("ground_truth", {}).get(action.email_id, {})
        email_obj  = next(
            (e for e in task_data.get("emails", []) if e.email_id == action.email_id), None
        )
        email_body = (email_obj.body + " " + email_obj.subject) if email_obj else ""

        # Empty tags penalty
        if task_id == "hard_triage" and not action.tags:
            total_penalty += PENALTY_EMPTY_TAGS
            feedback_parts.append(f"⚠️ PENALTY: Tags required for hard_triage.")

        # Speed bonus
        halfway = max(1, max_steps // 2)
        if step < halfway and raw_score >= 0.7:
            total_bonus += BONUS_SPEED

        # Extra tag bonus
        expected_tags  = {t.lower().strip() for t in gt_email.get("tags", [])}
        predicted_tags = {t.lower().strip() for t in action.tags}
        if expected_tags and len(predicted_tags & expected_tags) > len(expected_tags) * 0.5:
            if len(predicted_tags) > len(expected_tags):
                total_bonus += BONUS_EXTRA_TAG

        gt_priority = gt_email.get("priority")

        # Sentiment bonus
        if (
            _text_contains_any(email_body, SENTIMENT_NEGATIVE)
            and action.priority in _HIGH_PRIORITY
            and gt_priority in _HIGH_PRIORITY
        ):
            total_bonus += BONUS_SENTIMENT

        # SLA bonus
        if (
            _text_contains_any(email_body, SLA_SIGNALS)
            and action.priority in _HIGH_PRIORITY
            and gt_priority in _HIGH_PRIORITY
        ):
            total_bonus += BONUS_SLA

        # Escalation bonus
        if (
            _text_contains_any(email_body, ESCALATION_SIGNALS)
            and action.priority in _HIGH_PRIORITY
            and gt_priority in _HIGH_PRIORITY
        ):
            total_bonus += BONUS_ESCALATION

    else:
        raw_score = _MIN_SCORE   # never 0.0
        breakdown = {}
        feedback_parts.insert(0, "No grading (duplicate).")

    # Final score — always clamped
    step_score = _clamp(round(raw_score - total_penalty + total_bonus, 4))

    all_scores = cumulative_scores + [step_score]
    cumulative = _clamp(round(sum(all_scores) / len(all_scores), 4))

    if total_penalty > 0:
        feedback_parts.append(f"| Penalties: -{total_penalty:.2f}")
    if total_bonus > 0:
        feedback_parts.append(f"| Bonuses: +{total_bonus:.2f}")
    feedback_parts.append(f"| Score: {step_score:.4f}")

    return Reward(
        step_score=step_score,
        cumulative_score=cumulative,
        penalty=round(total_penalty, 4),
        breakdown=breakdown,
        feedback=" ".join(feedback_parts),
    )
