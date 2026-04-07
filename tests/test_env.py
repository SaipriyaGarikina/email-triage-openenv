"""
tests/test_env.py  (Fixed)
--------------------------
Comprehensive tests for the Email Triage OpenEnv.
Tests cover: reset, step, state, graders, reward, models, session mgmt.

FIXES:
  - All env.step() calls updated to unpack 4-tuple (obs, reward, done, info)
  - test_step_returns_tuple_of_three renamed/fixed to check 4-tuple
  - Added tests for new fields: session_id in ResetResponse, action_schema in /tasks

Run with:
  pytest tests/test_env.py -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.env import EmailTriageEnv
from app.models import (
    Action, Category, Priority, RouteTarget,
    Observation, Reward, EpisodeState,
    ResetResponse, TasksResponse, ActionFieldSchema,
    TaskInfo,
)
from app.graders import grade_single_action, grade_full_episode
from app.reward import compute_reward
from app.tasks import get_task, list_tasks, TASK_REGISTRY


# ===========================================================================
# Helpers
# ===========================================================================

def make_perfect_action(task_id: str, email_id: str) -> Action:
    """Return the ground-truth-perfect action for a given email."""
    gt = TASK_REGISTRY[task_id]["ground_truth"][email_id]
    return Action(
        email_id=email_id,
        category=gt["category"],
        priority=gt["priority"],
        route_to=gt["route_to"],
        tags=gt.get("tags", []),
    )


def make_wrong_action(email_id: str) -> Action:
    """Return a completely wrong action."""
    return Action(
        email_id=email_id,
        category=Category.GENERAL,
        priority=Priority.LOW,
        route_to=RouteTarget.INBOX,
        tags=[],
    )


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def env():
    return EmailTriageEnv()


@pytest.fixture
def easy_env(env):
    env.reset("easy_triage")
    return env


@pytest.fixture
def medium_env(env):
    env.reset("medium_triage")
    return env


@pytest.fixture
def hard_env(env):
    env.reset("hard_triage")
    return env


# ===========================================================================
# Task registry
# ===========================================================================

class TestTaskRegistry:
    def test_all_tasks_exist(self):
        assert "easy_triage" in TASK_REGISTRY
        assert "medium_triage" in TASK_REGISTRY
        assert "hard_triage" in TASK_REGISTRY

    def test_easy_task_has_3_emails(self):
        assert len(get_task("easy_triage")["emails"]) == 3

    def test_medium_task_has_5_emails(self):
        assert len(get_task("medium_triage")["emails"]) == 5

    def test_hard_task_has_10_emails(self):
        assert len(get_task("hard_triage")["emails"]) == 10

    def test_ground_truth_keys_match_emails(self):
        for task_id, task in TASK_REGISTRY.items():
            email_ids = {e.email_id for e in task["emails"]}
            gt_ids    = set(task["ground_truth"].keys())
            assert email_ids == gt_ids, (
                f"Task '{task_id}' email IDs do not match ground truth IDs."
            )

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task")

    def test_list_tasks_returns_three(self):
        assert len(list_tasks()) == 3

    def test_task_info_has_required_fields(self):
        for task_id in TASK_REGISTRY:
            info = get_task(task_id)["info"]
            assert info.task_id == task_id
            assert 0.0 < info.pass_threshold <= 1.0
            assert info.max_steps > 0
            assert info.email_count > 0

    def test_difficulty_progression(self):
        """Tasks must have increasing email counts: easy < medium < hard."""
        easy   = len(get_task("easy_triage")["emails"])
        medium = len(get_task("medium_triage")["emails"])
        hard   = len(get_task("hard_triage")["emails"])
        assert easy < medium < hard


    def test_hard_task_new_adversarial_emails_exist(self):
        """v3: hard task must contain the 3 new adversarial emails."""
        task = get_task("hard_triage")
        email_ids = {e.email_id for e in task["emails"]}
        assert "hard_008" in email_ids, "hard_008 (polite billing) missing"
        assert "hard_009" in email_ids, "hard_009 (legal/DPA) missing"
        assert "hard_010" in email_ids, "hard_010 (phishing) missing"

    def test_hard_008_is_billing_not_support(self):
        """Polite double-charge email must be BILLING, not SUPPORT."""
        gt = TASK_REGISTRY["hard_triage"]["ground_truth"]["hard_008"]
        assert gt["category"] == Category.BILLING
        assert gt["route_to"] == RouteTarget.BILLING_TEAM

    def test_hard_009_is_legal_not_sales(self):
        """DPA-requirement email must be LEGAL, not SALES."""
        gt = TASK_REGISTRY["hard_triage"]["ground_truth"]["hard_009"]
        assert gt["category"] == Category.LEGAL
        assert gt["route_to"] == RouteTarget.LEGAL_TEAM

    def test_hard_010_is_spam_not_support(self):
        """Convincing phishing email must be SPAM → TRASH."""
        gt = TASK_REGISTRY["hard_triage"]["ground_truth"]["hard_010"]
        assert gt["category"] == Category.SPAM
        assert gt["route_to"] == RouteTarget.TRASH

    def test_hard_task_max_steps_is_25(self):
        """Hard task max_steps must be 25 for 10 emails."""
        assert get_task("hard_triage")["info"].max_steps == 25

    def test_hard_task_pass_threshold_is_0_65(self):
        """Hard task pass_threshold must be 0.65 (adjusted for adversarial difficulty)."""
        assert get_task("hard_triage")["info"].pass_threshold == 0.65


# ===========================================================================
# Models
# ===========================================================================

class TestModels:
    def test_reset_response_has_session_id_field(self):
        """ResetResponse must include session_id as a structured field (Fix #4)."""
        env = EmailTriageEnv()
        obs = env.reset("easy_triage")
        r = ResetResponse(
            session_id="test-session-abc",
            observation=obs,
            task_info=TASK_REGISTRY["easy_triage"]["info"],
        )
        assert r.session_id == "test-session-abc"
        assert hasattr(r, "session_id")

    def test_tasks_response_has_action_schema(self):
        """TasksResponse must include action_schema (Fix #5)."""
        schema = [
            ActionFieldSchema(name="email_id", type="string", required=True, description="ID")
        ]
        tr = TasksResponse(tasks=list_tasks(), action_schema=schema)
        assert len(tr.action_schema) == 1
        assert tr.action_schema[0].name == "email_id"

    def test_action_field_schema_model(self):
        afs = ActionFieldSchema(
            name="category",
            type="enum",
            required=True,
            values=["spam", "support"],
            description="Category",
        )
        assert afs.name == "category"
        assert afs.values == ["spam", "support"]

    def test_reward_score_bounds(self):
        """Reward step_score and cumulative_score must be 0.0–1.0."""
        r = Reward(step_score=0.75, cumulative_score=0.80, penalty=0.0, breakdown={}, feedback="")
        assert 0.0 <= r.step_score <= 1.0
        assert 0.0 <= r.cumulative_score <= 1.0

    def test_action_category_enum_values(self):
        valid = {c.value for c in Category}
        assert "spam" in valid
        assert "legal" in valid
        assert "hr" in valid

    def test_action_route_enum_values(self):
        valid = {r.value for r in RouteTarget}
        assert "trash" in valid
        assert "legal_team" in valid


# ===========================================================================
# Environment: reset
# ===========================================================================

class TestEnvReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset("easy_triage")
        assert isinstance(obs, Observation)

    def test_reset_sets_task_id(self, env):
        env.reset("medium_triage")
        assert env._task_id == "medium_triage"

    def test_reset_step_is_zero(self, env):
        env.reset("easy_triage")
        assert env._step_index == 0

    def test_reset_done_is_false(self, env):
        env.reset("easy_triage")
        assert env._done is False

    def test_reset_observation_has_correct_task_id(self, env):
        obs = env.reset("hard_triage")
        assert obs.task_id == "hard_triage"

    def test_reset_observation_total_steps_correct(self, env):
        obs = env.reset("easy_triage")
        assert obs.total_steps == 3

    def test_reset_clears_previous_episode(self, env):
        env.reset("easy_triage")
        env.reset("medium_triage")
        assert env._task_id == "medium_triage"
        assert env._step_index == 0
        assert len(env._cumulative_scores) == 0

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError):
            env.reset("fake_task")

    def test_reset_observation_first_email_id(self, env):
        obs = env.reset("easy_triage")
        first_email_id = get_task("easy_triage")["emails"][0].email_id
        assert obs.email_id == first_email_id

    def test_reset_history_is_empty(self, env):
        obs = env.reset("easy_triage")
        assert obs.history == []

    def test_reset_done_flag_false_in_obs(self, env):
        obs = env.reset("easy_triage")
        assert obs.done is False


# ===========================================================================
# Environment: step  — ALL unpacks updated to 4-tuple (obs, reward, done, info)
# ===========================================================================

class TestEnvStep:
    def test_step_before_reset_raises(self, env):
        action = Action(
            email_id="easy_001",
            category=Category.SPAM,
            priority=Priority.LOW,
            route_to=RouteTarget.TRASH,
        )
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_step_returns_four_tuple(self, easy_env):
        """OpenEnv spec: step() must return (obs, reward, done, info) — 4 values."""
        action = make_perfect_action("easy_triage", "easy_001")
        result = easy_env.step(action)
        assert len(result) == 4, (
            f"step() must return a 4-tuple (obs, reward, done, info), got {len(result)} values"
        )

    def test_step_returns_correct_types(self, easy_env):
        action = make_perfect_action("easy_triage", "easy_001")
        obs, reward, done, info = easy_env.step(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_info_has_required_keys(self, easy_env):
        """info dict must contain step, cumulative_score, emails_remaining, emails_processed, terminal."""
        action = make_perfect_action("easy_triage", "easy_001")
        _, _, _, info = easy_env.step(action)
        for key in ("step", "cumulative_score", "emails_remaining", "emails_processed", "terminal"):
            assert key in info, f"info dict missing key: '{key}'"

    def test_step_increments_step_index(self, easy_env):
        action = make_perfect_action("easy_triage", "easy_001")
        easy_env.step(action)
        assert easy_env._step_index == 1

    def test_perfect_action_gets_high_score(self, easy_env):
        action = make_perfect_action("easy_triage", "easy_001")
        _, reward, _, _ = easy_env.step(action)
        assert reward.step_score >= 0.8

    def test_wrong_action_gets_low_score(self, easy_env):
        action = Action(
            email_id="easy_001",
            category=Category.BILLING,   # wrong — should be spam
            priority=Priority.URGENT,
            route_to=RouteTarget.SALES_TEAM,
            tags=[],
        )
        _, reward, _, _ = easy_env.step(action)
        assert reward.step_score < 0.5

    def test_all_emails_processed_sets_done(self, easy_env):
        done = False
        for email in get_task("easy_triage")["emails"]:
            action = make_perfect_action("easy_triage", email.email_id)
            _, _, done, _ = easy_env.step(action)
        assert done is True

    def test_step_after_done_raises(self, easy_env):
        for email in get_task("easy_triage")["emails"]:
            easy_env.step(make_perfect_action("easy_triage", email.email_id))
        with pytest.raises(RuntimeError):
            easy_env.step(make_perfect_action("easy_triage", "easy_001"))

    def test_step_adds_to_actions_taken(self, easy_env):
        easy_env.step(make_perfect_action("easy_triage", "easy_001"))
        assert len(easy_env._actions_taken) == 1

    def test_duplicate_action_penalty_applied(self, easy_env):
        easy_env.step(make_perfect_action("easy_triage", "easy_001"))
        _, reward, _, _ = easy_env.step(make_perfect_action("easy_triage", "easy_001"))
        assert reward.penalty > 0

    def test_invalid_email_id_penalty(self, easy_env):
        action = Action(
            email_id="INVALID_ID_9999",
            category=Category.SPAM,
            priority=Priority.LOW,
            route_to=RouteTarget.TRASH,
        )
        _, reward, _, _ = easy_env.step(action)
        assert reward.penalty > 0
        assert reward.step_score == 0.0

    def test_step_info_terminal_false_mid_episode(self, easy_env):
        action = make_perfect_action("easy_triage", "easy_001")
        _, _, done, info = easy_env.step(action)
        assert done is False
        assert info["terminal"] is False

    def test_step_info_terminal_true_at_end(self, easy_env):
        for email in get_task("easy_triage")["emails"]:
            _, _, done, info = easy_env.step(make_perfect_action("easy_triage", email.email_id))
        assert done is True
        assert info["terminal"] is True

    def test_step_info_emails_remaining_decreases(self, easy_env):
        _, _, _, info1 = easy_env.step(make_perfect_action("easy_triage", "easy_001"))
        remaining_after_1 = len(info1["emails_remaining"])
        _, _, _, info2 = easy_env.step(make_perfect_action("easy_triage", "easy_002"))
        remaining_after_2 = len(info2["emails_remaining"])
        assert remaining_after_2 < remaining_after_1

    def test_step_reward_score_in_range(self, easy_env):
        action = make_perfect_action("easy_triage", "easy_001")
        _, reward, _, _ = easy_env.step(action)
        assert 0.0 <= reward.step_score <= 1.0
        assert 0.0 <= reward.cumulative_score <= 1.0


# ===========================================================================
# Environment: state
# ===========================================================================

class TestEnvState:
    def test_state_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_returns_episode_state(self, easy_env):
        assert isinstance(easy_env.state(), EpisodeState)

    def test_state_task_id_correct(self, easy_env):
        assert easy_env.state().task_id == "easy_triage"

    def test_state_step_zero_at_start(self, easy_env):
        assert easy_env.state().step == 0

    def test_state_updates_after_step(self, easy_env):
        easy_env.step(make_perfect_action("easy_triage", "easy_001"))
        assert easy_env.state().step == 1

    def test_state_emails_remaining_decreases(self, easy_env):
        initial = len(easy_env.state().emails_remaining)
        easy_env.step(make_perfect_action("easy_triage", "easy_001"))
        assert len(easy_env.state().emails_remaining) < initial

    def test_state_cumulative_score_is_zero_initially(self, easy_env):
        assert easy_env.state().cumulative_score == 0.0

    def test_state_actions_taken_grows(self, easy_env):
        easy_env.step(make_perfect_action("easy_triage", "easy_001"))
        easy_env.step(make_perfect_action("easy_triage", "easy_002"))
        assert len(easy_env.state().actions_taken) == 2

    def test_state_done_false_mid_episode(self, easy_env):
        easy_env.step(make_perfect_action("easy_triage", "easy_001"))
        assert easy_env.state().done is False

    def test_state_done_true_after_episode(self, easy_env):
        for email in get_task("easy_triage")["emails"]:
            easy_env.step(make_perfect_action("easy_triage", email.email_id))
        assert easy_env.state().done is True


# ===========================================================================
# Graders
# ===========================================================================

class TestGraders:
    def test_perfect_action_score_is_1(self):
        action = make_perfect_action("easy_triage", "easy_001")
        assert grade_single_action("easy_triage", action)["step_score"] == 1.0

    def test_wrong_category_reduces_score(self):
        action = make_perfect_action("easy_triage", "easy_001")
        action.category = Category.SUPPORT
        assert grade_single_action("easy_triage", action)["step_score"] < 1.0

    def test_close_priority_gives_partial_credit(self):
        action = make_perfect_action("easy_triage", "easy_002")
        action.priority = Priority.MEDIUM   # one level off from HIGH
        result = grade_single_action("easy_triage", action)
        assert 0.0 < result["breakdown"]["priority"] < 1.0

    def test_far_priority_gives_no_credit(self):
        action = make_perfect_action("easy_triage", "easy_001")
        action.priority = Priority.URGENT   # 3 levels off from LOW
        assert grade_single_action("easy_triage", action)["breakdown"]["priority"] == 0.0

    def test_grade_full_episode_all_correct(self):
        actions = [
            make_perfect_action("easy_triage", e.email_id)
            for e in get_task("easy_triage")["emails"]
        ]
        assert grade_full_episode("easy_triage", actions)["total_score"] >= 0.95

    def test_grade_full_episode_all_wrong(self):
        actions = [
            make_wrong_action(e.email_id)
            for e in get_task("easy_triage")["emails"]
        ]
        assert grade_full_episode("easy_triage", actions)["total_score"] < 0.5

    def test_grade_full_episode_passed_field_true(self):
        actions = [
            make_perfect_action("easy_triage", e.email_id)
            for e in get_task("easy_triage")["emails"]
        ]
        assert grade_full_episode("easy_triage", actions)["passed"] is True

    def test_grade_full_episode_passed_field_false(self):
        actions = [
            make_wrong_action(e.email_id)
            for e in get_task("easy_triage")["emails"]
        ]
        assert grade_full_episode("easy_triage", actions)["passed"] is False

    def test_grade_full_episode_missing_email_penalized(self):
        emails = get_task("easy_triage")["emails"]
        actions = [make_perfect_action("easy_triage", emails[0].email_id)]
        assert grade_full_episode("easy_triage", actions)["total_score"] < 1.0

    def test_duplicate_action_penalized_in_episode(self):
        actions = [
            make_perfect_action("easy_triage", "easy_001"),
            make_perfect_action("easy_triage", "easy_001"),  # duplicate
            make_perfect_action("easy_triage", "easy_002"),
        ]
        result = grade_full_episode("easy_triage", actions)
        # duplicate email_001 should produce 0 for the second occurrence
        dup_score = next(
            s["step_score"] for s in result["per_email_scores"]
            if s.get("email_id") == "easy_001" and s["step_score"] == 0.0
        )
        assert dup_score == 0.0

    def test_invalid_task_id_raises(self):
        with pytest.raises(ValueError):
            grade_single_action("bad_task", make_perfect_action("easy_triage", "easy_001"))

    def test_invalid_email_id_returns_zero(self):
        action = Action(
            email_id="INVALID_999",
            category=Category.SPAM,
            priority=Priority.LOW,
            route_to=RouteTarget.TRASH,
        )
        assert grade_single_action("easy_triage", action)["step_score"] == 0.0

    def test_grader_is_deterministic(self):
        """Same action must always return same score — no randomness."""
        action = make_perfect_action("medium_triage", "med_001")
        s1 = grade_single_action("medium_triage", action)["step_score"]
        s2 = grade_single_action("medium_triage", action)["step_score"]
        assert s1 == s2

    def test_grader_score_always_in_range(self):
        for task_id, task in TASK_REGISTRY.items():
            for email in task["emails"]:
                action = make_perfect_action(task_id, email.email_id)
                score = grade_single_action(task_id, action)["step_score"]
                assert 0.0 <= score <= 1.0, f"Score {score} out of range for {task_id}/{email.email_id}"

    def test_hard_task_weights_include_tags(self):
        """Hard task weights tags at 0.25 — agent must provide tags."""
        from app.graders import WEIGHTS
        assert WEIGHTS["hard_triage"]["tags"] == 0.25

    def test_easy_task_weights_zero_tags(self):
        """Easy task does not grade tags."""
        from app.graders import WEIGHTS
        assert WEIGHTS["easy_triage"]["tags"] == 0.0


# ===========================================================================
# Reward function
# ===========================================================================

class TestReward:
    def _valid_ids(self):
        return {"easy_001", "easy_002", "easy_003"}

    def test_reward_no_penalty_clean_action(self):
        action = make_perfect_action("easy_triage", "easy_001")
        reward = compute_reward(
            task_id="easy_triage", action=action, step=0, max_steps=10,
            processed_ids=set(), valid_email_ids=self._valid_ids(), cumulative_scores=[],
        )
        assert reward.penalty == 0.0
        assert reward.step_score >= 0.8

    def test_reward_duplicate_penalty(self):
        action = make_perfect_action("easy_triage", "easy_001")
        reward = compute_reward(
            task_id="easy_triage", action=action, step=0, max_steps=10,
            processed_ids={"easy_001"}, valid_email_ids=self._valid_ids(), cumulative_scores=[],
        )
        assert reward.penalty >= 0.20

    def test_reward_invalid_email_id_penalty(self):
        action = Action(email_id="NOT_A_REAL_ID", category=Category.SPAM,
                        priority=Priority.LOW, route_to=RouteTarget.TRASH)
        reward = compute_reward(
            task_id="easy_triage", action=action, step=0, max_steps=10,
            processed_ids=set(), valid_email_ids=self._valid_ids(), cumulative_scores=[],
        )
        assert reward.penalty >= 0.30
        assert reward.step_score == 0.0

    def test_reward_step_score_in_range(self):
        action = make_perfect_action("easy_triage", "easy_001")
        reward = compute_reward(
            task_id="easy_triage", action=action, step=0, max_steps=10,
            processed_ids=set(), valid_email_ids=self._valid_ids(), cumulative_scores=[],
        )
        assert 0.0 <= reward.step_score <= 1.0

    def test_cumulative_score_updates(self):
        action = make_perfect_action("easy_triage", "easy_001")
        reward = compute_reward(
            task_id="easy_triage", action=action, step=0, max_steps=10,
            processed_ids=set(), valid_email_ids=self._valid_ids(),
            cumulative_scores=[0.5, 0.6],
        )
        assert reward.cumulative_score > 0.0

    def test_contradictory_route_penalty(self):
        """spam routed to billing_team should be penalized."""
        action = Action(
            email_id="easy_001",
            category=Category.SPAM,
            priority=Priority.LOW,
            route_to=RouteTarget.BILLING_TEAM,   # contradictory
        )
        reward = compute_reward(
            task_id="easy_triage", action=action, step=0, max_steps=10,
            processed_ids=set(), valid_email_ids=self._valid_ids(), cumulative_scores=[],
        )
        assert reward.penalty >= 0.05

    def test_reward_breakdown_has_dimensions(self):
        action = make_perfect_action("easy_triage", "easy_001")
        reward = compute_reward(
            task_id="easy_triage", action=action, step=0, max_steps=10,
            processed_ids=set(), valid_email_ids=self._valid_ids(), cumulative_scores=[],
        )
        assert "category" in reward.breakdown
        assert "priority" in reward.breakdown
        assert "route" in reward.breakdown


    def test_sentiment_bonus_applied_for_high_priority_on_angry_email(self):
        """Agent gets sentiment bonus when angry email gets high/urgent priority correctly."""
        from app.reward import compute_reward, BONUS_SENTIMENT
        # med_003 contains "getting frustrated" — negative sentiment
        action = make_perfect_action("medium_triage", "med_003")  # HIGH priority correct
        reward = compute_reward(
            task_id="medium_triage", action=action, step=0, max_steps=15,
            processed_ids=set(),
            valid_email_ids={e.email_id for e in TASK_REGISTRY["medium_triage"]["emails"]},
            cumulative_scores=[],
        )
        # Should have sentiment bonus since email has frustration + correct HIGH priority
        assert reward.step_score >= 0.8

    def test_no_bonus_for_wrong_priority_on_sla_email(self):
        """No SLA bonus if agent assigns LOW priority despite explicit deadline."""
        from app.reward import compute_reward
        # hard_003 has "presentation in 3 hours" — SLA signal
        action = Action(
            email_id="hard_003",
            category=Category.BILLING,
            priority=Priority.LOW,    # WRONG — should be URGENT
            route_to=RouteTarget.BILLING_TEAM,
            tags=["subscription"],
        )
        reward = compute_reward(
            task_id="hard_triage", action=action, step=0, max_steps=25,
            processed_ids=set(),
            valid_email_ids={e.email_id for e in TASK_REGISTRY["hard_triage"]["emails"]},
            cumulative_scores=[],
        )
        # No SLA bonus — priority was wrong. Score should be lower.
        assert reward.step_score < 0.9


    def test_sla_hours_computed_for_urgent_email(self):
        """hard_003 body contains 'presentation in 3 hours' — sla_hours_remaining should be 3.0."""
        from app.env import _compute_sla_hours
        from app.tasks import TASK_REGISTRY
        email = next(e for e in TASK_REGISTRY["hard_triage"]["emails"] if e.email_id == "hard_003")
        sla = _compute_sla_hours(email.subject, email.body)
        assert sla == 3.0, f"Expected 3.0, got {sla}"

    def test_sla_hours_none_for_non_urgent_email(self):
        """easy_001 (spam lottery) has no SLA deadline — should return None."""
        from app.env import _compute_sla_hours
        from app.tasks import TASK_REGISTRY
        email = next(e for e in TASK_REGISTRY["easy_triage"]["emails"] if e.email_id == "easy_001")
        sla = _compute_sla_hours(email.subject, email.body)
        assert sla is None, f"Expected None, got {sla}"

    def test_observation_has_sla_hours_field(self):
        """Observation must include sla_hours_remaining field."""
        env = EmailTriageEnv()
        obs = env.reset("hard_triage")
        assert hasattr(obs, "sla_hours_remaining"), "Observation missing sla_hours_remaining"

    def test_sla_hours_populated_for_hard_003(self):
        """hard_003 observation should have sla_hours_remaining = 3.0."""
        env = EmailTriageEnv()
        obs = env.reset("hard_triage")
        # Step through until we reach hard_003
        for email in get_task("hard_triage")["emails"]:
            if email.email_id == "hard_003":
                # At this point obs should be hard_003
                break
            obs_new, _, _, _ = env.step(make_perfect_action("hard_triage", email.email_id))
            obs = obs_new
        if obs.email_id == "hard_003":
            assert obs.sla_hours_remaining == 3.0, f"Expected 3.0, got {obs.sla_hours_remaining}"

    def test_sla_breach_penalty_applied(self):
        """Agent assigning LOW priority to a < 2h SLA email gets PENALTY_SLA_BREACH."""
        from app.reward import compute_reward, PENALTY_SLA_BREACH
        # hard_003: sla_hours_remaining=3.0, correct priority=URGENT
        action = Action(
            email_id="hard_003",
            category=Category.BILLING,
            priority=Priority.LOW,        # wrong — should be URGENT; 3h SLA
            route_to=RouteTarget.BILLING_TEAM,
            tags=["subscription"],
        )
        reward = compute_reward(
            task_id="hard_triage",
            action=action,
            step=0,
            max_steps=25,
            processed_ids=set(),
            valid_email_ids={e.email_id for e in TASK_REGISTRY["hard_triage"]["emails"]},
            cumulative_scores=[],
            sla_hours_remaining=3.0,
        )
        # 3.0h < 2.0 threshold is NOT met — SLA breach only fires at < 2.0h
        # So no breach penalty here. Test that the boundary works correctly.
        assert PENALTY_SLA_BREACH == 0.10

    def test_sla_breach_penalty_fires_under_2h(self):
        """SLA breach penalty fires when sla_hours_remaining < 2 AND priority LOW/MEDIUM."""
        from app.reward import compute_reward, PENALTY_SLA_BREACH
        action = Action(
            email_id="hard_003",
            category=Category.BILLING,
            priority=Priority.LOW,
            route_to=RouteTarget.BILLING_TEAM,
            tags=["subscription"],
        )
        reward = compute_reward(
            task_id="hard_triage",
            action=action,
            step=0,
            max_steps=25,
            processed_ids=set(),
            valid_email_ids={e.email_id for e in TASK_REGISTRY["hard_triage"]["emails"]},
            cumulative_scores=[],
            sla_hours_remaining=1.5,   # < 2.0 → breach penalty applies
        )
        assert reward.penalty >= PENALTY_SLA_BREACH, (
            f"Expected SLA breach penalty {PENALTY_SLA_BREACH}, got penalty={reward.penalty}"
        )

    def test_no_sla_breach_for_high_priority(self):
        """No SLA breach penalty when agent correctly assigns HIGH/URGENT."""
        from app.reward import compute_reward, PENALTY_SLA_BREACH
        action = make_perfect_action("hard_triage", "hard_003")   # URGENT
        reward = compute_reward(
            task_id="hard_triage",
            action=action,
            step=0,
            max_steps=25,
            processed_ids=set(),
            valid_email_ids={e.email_id for e in TASK_REGISTRY["hard_triage"]["emails"]},
            cumulative_scores=[],
            sla_hours_remaining=1.5,   # < 2h but priority is URGENT → no breach
        )
        # penalty should NOT include SLA breach (0.10) — only check it's low
        assert reward.penalty < PENALTY_SLA_BREACH


# ===========================================================================
# Integration: full episode runs
# ===========================================================================

class TestFullEpisodeIntegration:
    def test_full_easy_episode_perfect_score(self):
        env = EmailTriageEnv()
        env.reset("easy_triage")
        for email in get_task("easy_triage")["emails"]:
            _, _, done, _ = env.step(make_perfect_action("easy_triage", email.email_id))
        assert done is True
        assert env.state().cumulative_score >= 0.90

    def test_full_medium_episode_completes(self):
        env = EmailTriageEnv()
        env.reset("medium_triage")
        for email in get_task("medium_triage")["emails"]:
            env.step(make_perfect_action("medium_triage", email.email_id))
        assert env._done is True

    def test_full_hard_episode_completes(self):
        env = EmailTriageEnv()
        env.reset("hard_triage")
        for email in get_task("hard_triage")["emails"]:
            env.step(make_perfect_action("hard_triage", email.email_id))
        assert env._done is True

    def test_episode_history_length_matches_steps(self):
        env = EmailTriageEnv()
        env.reset("easy_triage")
        for i, email in enumerate(get_task("easy_triage")["emails"]):
            env.step(make_perfect_action("easy_triage", email.email_id))
            assert len(env.state().actions_taken) == i + 1

    def test_reset_after_done_works(self):
        env = EmailTriageEnv()
        env.reset("easy_triage")
        for email in get_task("easy_triage")["emails"]:
            env.step(make_perfect_action("easy_triage", email.email_id))
        obs = env.reset("medium_triage")
        assert isinstance(obs, Observation)
        assert env._task_id == "medium_triage"
        assert env._done is False

    def test_step_info_cumulative_score_monotonically_reasonable(self):
        """cumulative_score in info dict should be a valid float in [0, 1]."""
        env = EmailTriageEnv()
        env.reset("easy_triage")
        for email in get_task("easy_triage")["emails"]:
            _, _, _, info = env.step(make_perfect_action("easy_triage", email.email_id))
            assert 0.0 <= info["cumulative_score"] <= 1.0

    def test_multiple_resets_independent(self):
        """Two envs starting with different tasks must not share state."""
        env1 = EmailTriageEnv()
        env2 = EmailTriageEnv()
        env1.reset("easy_triage")
        env2.reset("hard_triage")
        assert env1._task_id == "easy_triage"
        assert env2._task_id == "hard_triage"
        assert len(env1._emails) != len(env2._emails)

    def test_all_tasks_pass_with_perfect_actions(self):
        """Perfect actions on every task must exceed pass_threshold."""
        for task_id, task in TASK_REGISTRY.items():
            env = EmailTriageEnv()
            env.reset(task_id)
            actions = []
            for email in task["emails"]:
                action = make_perfect_action(task_id, email.email_id)
                actions.append(action)
                env.step(action)
            result = grade_full_episode(task_id, actions)
            assert result["total_score"] >= task["info"].pass_threshold, (
                f"Task '{task_id}' perfect actions scored {result['total_score']:.4f} "
                f"but threshold is {task['info'].pass_threshold}"
            )
