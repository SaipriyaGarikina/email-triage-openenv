"""
graders.py — Final Fixed Version
All scores strictly within (0.01, 0.99). Double-clamped everywhere.
"""

from __future__ import annotations

from app.models import Action, Category, Priority, RouteTarget
from app.tasks import TASK_REGISTRY

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


WEIGHTS: dict[str, dict[str, float]] = {
    "easy_triage":   {"category": 0.50, "priority": 0.25, "route": 0.25, "tags": 0.00},
    "medium_triage": {"category": 0.35, "priority": 0.35, "route": 0.20, "tags": 0.10},
    "hard_triage":   {"category": 0.25, "priority": 0.25, "route": 0.25, "tags": 0.25},
}

PRIORITY_ORDER: dict[Priority, int] = {
    Priority.LOW: 0, Priority.MEDIUM: 1, Priority.HIGH: 2, Priority.URGENT: 3,
}


def _score_priority(predicted: Priority, expected: Priority) -> float:
    diff = abs(PRIORITY_ORDER[predicted] - PRIORITY_ORDER[expected])
    if diff == 0:   return _MAX_SCORE
    elif diff == 1: return 0.49
    else:           return _MIN_SCORE


def _score_tags(predicted_tags: list[str], expected_tags: list[str]) -> float:
    if not expected_tags:
        return _MAX_SCORE if not predicted_tags else 0.49
    pred_set = {t.lower().strip() for t in predicted_tags}
    exp_set  = {t.lower().strip() for t in expected_tags}
    if not pred_set:
        return _MIN_SCORE
    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set)
    recall    = tp / len(exp_set)
    if precision + recall == 0.0:
        return _MIN_SCORE
    f1 = 2 * precision * recall / (precision + recall)
    return _clamp(round(f1, 4))


def grade_single_action(task_id: str, action: Action) -> dict:
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id: {task_id}")

    ground_truth = task["ground_truth"]
    if action.email_id not in ground_truth:
        return {
            "email_id":   action.email_id,
            "step_score": _MIN_SCORE,
            "breakdown":  {},
            "feedback":   f"Email ID '{action.email_id}' not found in task '{task_id}'.",
        }

    gt      = ground_truth[action.email_id]
    weights = WEIGHTS[task_id]

    cat_score   = _MAX_SCORE if action.category == gt["category"] else _MIN_SCORE
    pri_score   = _score_priority(action.priority, gt["priority"])
    route_score = _MAX_SCORE if action.route_to == gt["route_to"] else _MIN_SCORE
    tag_score   = _score_tags(action.tags, gt.get("tags", []))

    total = (
        weights["category"] * cat_score
        + weights["priority"] * pri_score
        + weights["route"]    * route_score
        + weights["tags"]     * tag_score
    )
    total = _clamp(round(total, 4))

    breakdown = {
        "category": _clamp(cat_score),
        "priority": _clamp(pri_score),
        "route":    _clamp(route_score),
        "tags":     _clamp(tag_score),
    }

    feedback_parts = []
    feedback_parts.append(
        f"✅ Category correct." if cat_score >= _MAX_SCORE
        else f"❌ Category '{action.category}' wrong. Expected: '{gt['category']}'."
    )
    if pri_score >= _MAX_SCORE:
        feedback_parts.append(f"✅ Priority correct.")
    elif pri_score == 0.49:
        feedback_parts.append(f"⚠️ Priority close. Expected: '{gt['priority']}'.")
    else:
        feedback_parts.append(f"❌ Priority '{action.priority}' wrong. Expected: '{gt['priority']}'.")
    feedback_parts.append(
        f"✅ Route correct." if route_score >= _MAX_SCORE
        else f"❌ Route '{action.route_to}' wrong. Expected: '{gt['route_to']}'."
    )
    if weights["tags"] > 0:
        feedback_parts.append(f"Tags F1: {tag_score:.4f}.")

    return {
        "email_id":   action.email_id,
        "step_score": total,
        "breakdown":  breakdown,
        "feedback":   " ".join(feedback_parts),
    }


def grade_full_episode(task_id: str, actions: list[Action]) -> dict:
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id: {task_id}")

    pass_threshold   = task["info"].pass_threshold
    per_email_scores = []
    scores           = []
    acted_ids        = set()

    for action in actions:
        if action.email_id in acted_ids:
            per_email_scores.append({
                "email_id":   action.email_id,
                "step_score": _MIN_SCORE,
                "breakdown":  {},
                "feedback":   f"⚠️ Duplicate action for '{action.email_id}'.",
            })
            scores.append(_MIN_SCORE)
        else:
            result = grade_single_action(task_id, action)
            result["step_score"] = _clamp(result["step_score"])
            per_email_scores.append(result)
            scores.append(result["step_score"])
            acted_ids.add(action.email_id)

    expected_ids = set(task["ground_truth"].keys())
    for missing_id in (expected_ids - acted_ids):
        per_email_scores.append({
            "email_id":   missing_id,
            "step_score": _MIN_SCORE,
            "breakdown":  {},
            "feedback":   f"❌ Email '{missing_id}' never processed.",
        })
        scores.append(_MIN_SCORE)

    # FINAL clamp — impossible to return 0.0 or 1.0
    if not scores:
        total_score = _MIN_SCORE
    else:
        total_score = _clamp(round(sum(scores) / len(scores), 4))

    passed = total_score >= pass_threshold

    return {
        "task_id":          task_id,
        "total_score":      total_score,
        "per_email_scores": per_email_scores,
        "passed":           passed,
        "feedback": (
            f"Task '{task_id}' done. Score: {total_score:.4f}. "
            f"Threshold: {pass_threshold}. "
            f"{'✅ PASSED' if passed else '❌ FAILED'}."
        ),
    }
