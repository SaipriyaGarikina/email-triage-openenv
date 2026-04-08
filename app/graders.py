"""
graders.py
----------
Deterministic graders for each email action.
No randomness. Returns a float score 0.0–1.0.

Grading weights per task:
  Easy   : category=0.50, priority=0.25, route=0.25, tags=0.0
  Medium : category=0.35, priority=0.35, route=0.20, tags=0.10
  Hard   : category=0.25, priority=0.25, route=0.25, tags=0.25
"""

from __future__ import annotations

from app.models import Action, Category, Priority, RouteTarget
from app.tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Grading weights by difficulty
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, dict[str, float]] = {
    "easy_triage": {
        "category": 0.50,
        "priority": 0.25,
        "route": 0.25,
        "tags": 0.00,
    },
    "medium_triage": {
        "category": 0.35,
        "priority": 0.35,
        "route": 0.20,
        "tags": 0.10,
    },
    "hard_triage": {
        "category": 0.25,
        "priority": 0.25,
        "route": 0.25,
        "tags": 0.25,
    },
}

# ---------------------------------------------------------------------------
# Priority adjacency — partial credit for close-enough priority
# ---------------------------------------------------------------------------
#   e.g., agent says HIGH but correct is URGENT → partial credit
PRIORITY_ORDER: dict[Priority, int] = {
    Priority.LOW: 0,
    Priority.MEDIUM: 1,
    Priority.HIGH: 2,
    Priority.URGENT: 3,
}


def _score_priority(predicted: Priority, expected: Priority) -> float:
    """
    Returns:
      1.0 — exact match
      0.5 — one level off
      0.0 — two or more levels off
    """
    diff = abs(PRIORITY_ORDER[predicted] - PRIORITY_ORDER[expected])
    if diff == 0:
        return 0.98
    elif diff == 1:
        return 0.49
    else:
        return 0.02
        


def _score_tags(predicted_tags: list[str], expected_tags: list[str]) -> float:
    """
    Compute a precision-recall F1 on tag sets.
    Tags are lowercased and stripped for comparison.
    Returns 0.0 if both lists are empty (no tags expected, none given = perfect).
    """
    if not expected_tags:
        # No tags expected — penalize only if agent gave wrong tags
        if not predicted_tags:
            return 0.98
        else:
            return 0.49  # agent gave extra tags when none needed

    pred_set = {t.lower().strip() for t in predicted_tags}
    exp_set = {t.lower().strip() for t in expected_tags}

    if not pred_set:
        return 0.02

    true_positives = len(pred_set & exp_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(exp_set) if exp_set else 0.0

    if precision + recall == 0:
        return 0.02

    f1 = 2 * precision * recall / (precision + recall)
    f1 = round(f1, 4)
    f1 = max(0.02, min(0.98, f1))
    return f1


def grade_single_action(
    task_id: str,
    action: Action,
) -> dict:
    """
    Grade one action against the ground truth for that email.

    Returns a dict with:
      - step_score   : float (weighted total)
      - breakdown    : dict with individual dimension scores
      - feedback     : human-readable string
    """
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id: {task_id}")

    ground_truth = task["ground_truth"]
    if action.email_id not in ground_truth:
        return {
            "step_score": 0.02,
            "breakdown": {},
            "feedback": f"Email ID '{action.email_id}' not found in task '{task_id}'.",
        }

    gt = ground_truth[action.email_id]
    weights = WEIGHTS[task_id]

    # --- Category score ---
    cat_score = 0.98 if action.category == gt["category"] else 0.02

    # --- Priority score (with partial credit) ---
    pri_score = _score_priority(action.priority, gt["priority"])

    # --- Route score ---
    route_score = 0.98 if action.route_to == gt["route_to"] else 0.02

    # --- Tags score ---
    tag_score = _score_tags(action.tags, gt.get("tags", []))

    # --- Weighted total ---
    total = (
        weights["category"] * cat_score
        + weights["priority"] * pri_score
        + weights["route"] * route_score
        + weights["tags"] * tag_score
    )
    total = round(total, 4)
    total = max(0.02, min(0.98, total))

    breakdown = {
        
        "category": max(0.02, min(0.98, round(cat_score, 4))),
        "priority": max(0.02, min(0.98, round(pri_score, 4))),
        "route": max(0.02, min(0.98, round(route_score, 4))),
        "tags": max(0.02, min(0.98, round(tag_score, 4))),
    }

    # --- Human feedback ---
    feedback_parts = []
    if cat_score >= 0.98:
        feedback_parts.append(f"✅ Category '{action.category}' is correct.")
    else:
        feedback_parts.append(
            f"❌ Category '{action.category}' is wrong. Expected: '{gt['category']}'."
        )

    if pri_score >= 0.98:
        feedback_parts.append(f"✅ Priority '{action.priority}' is correct.")
    elif pri_score == 0.49:
        feedback_parts.append(
            f"⚠️  Priority '{action.priority}' is close (partial credit). "
            f"Expected: '{gt['priority']}'."
        )
    else:
        feedback_parts.append(
            f"❌ Priority '{action.priority}' is wrong. Expected: '{gt['priority']}'."
        )

    if route_score >= 0.98:
        feedback_parts.append(f"✅ Route '{action.route_to}' is correct.")
    else:
        feedback_parts.append(
            f"❌ Route '{action.route_to}' is wrong. Expected: '{gt['route_to']}'."
        )

    if weights["tags"] > 0:
        feedback_parts.append(f"Tags F1 score: {tag_score:.4f}.")

    feedback = " ".join(feedback_parts)

    return {
        "email_id": action.email_id,
        "step_score": total,
        "breakdown": breakdown,
        "feedback": feedback,
    }


def grade_full_episode(
    task_id: str,
    actions: list[Action],
) -> dict:
    """
    Grade all actions for a completed episode.

    Returns:
      - task_id
      - total_score         : mean of all step scores
      - per_email_scores    : list of individual grading dicts
      - passed              : bool (total_score >= pass_threshold)
      - feedback            : summary string
    """
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id: {task_id}")

    pass_threshold = task["info"].pass_threshold
    per_email_scores = []
    scores = []

    # Track which email_ids were acted upon
    acted_ids = set()

    for action in actions:
        if action.email_id in acted_ids:
            # Duplicate action penalty: score 0 for re-processed email
            per_email_scores.append({
                "email_id": action.email_id,
                "step_score": 0.02,
                "breakdown": {},
                "feedback": f"⚠️ Duplicate action for email '{action.email_id}'. Score penalized to 0.",
            })
            scores.append(0.02)
        else:
            result = grade_single_action(task_id, action)
            per_email_scores.append(result)
            scores.append(result["step_score"])
            acted_ids.add(action.email_id)

    # Penalty: if agent did not process all emails
    expected_ids = set(task["ground_truth"].keys())
    missing_ids = expected_ids - acted_ids
    for missing_id in missing_ids:
        per_email_scores.append({
            "email_id": missing_id,
            "step_score": 0.02,
            "breakdown": {},
            "feedback": f"❌ Email '{missing_id}' was never processed. Score: 0.",
        })
        scores.append(0.02)

    total_score = (sum(scores) / len(scores)) if scores else 0.02
    total_score = round(total_score, 4)
    total_score = max(0.02, min(0.98, total_score))
    passed = total_score >= pass_threshold

    feedback = (
        f"Task '{task_id}' completed. "
        f"Total score: {total_score:.4f} / 0.98"
        f"Pass threshold: {pass_threshold}. "
        f"{'✅ PASSED' if passed else '❌ FAILED'}."
    )

    return {
        "task_id": task_id,
        "total_score": total_score,
        "per_email_scores": per_email_scores,
        "passed": passed,
        "feedback": feedback,
    }

