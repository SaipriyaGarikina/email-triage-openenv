"""
models.py  (Fixed - strict score bounds)
------------------
All Pydantic data models.
FIXES:
  [Fix #4] ResetResponse includes session_id as structured field.
  [Fix #5] ActionFieldSchema + TasksResponse for /tasks endpoint.
  [Fix #7] Reward scores use gt=0.0, lt=1.0 (strictly within (0,1)).
  [Fix #FINAL] GraderResponse.total_score + BaselineResponse scores
               have field_validator to FORCE clamp to (0.01, 0.99)
               so it is physically impossible to return 0.0 or 1.0.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sentinel values — used by all validators
# ---------------------------------------------------------------------------
_MIN_SCORE = 0.01
_MAX_SCORE = 0.99


def _clamp_score(v: float) -> float:
    """Force any float strictly inside (0, 1). Never 0.0 or 1.0."""
    if v is None:
        return _MIN_SCORE
    return max(_MIN_SCORE, min(_MAX_SCORE, float(v)))


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Category(str, Enum):
    SPAM    = "spam"
    SUPPORT = "support"
    BILLING = "billing"
    SALES   = "sales"
    HR      = "hr"
    LEGAL   = "legal"
    GENERAL = "general"


class Priority(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    URGENT = "urgent"


class RouteTarget(str, Enum):
    SALES_TEAM   = "sales_team"
    TECH_TEAM    = "tech_team"
    BILLING_TEAM = "billing_team"
    HR_TEAM      = "hr_team"
    LEGAL_TEAM   = "legal_team"
    TRASH        = "trash"
    INBOX        = "inbox"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Email data model
# ---------------------------------------------------------------------------

class Email(BaseModel):
    email_id:  str = Field(..., description="Unique identifier for the email")
    subject:   str = Field(..., description="Subject line of the email")
    body:      str = Field(..., description="Full body text of the email")
    sender:    str = Field(..., description="Sender email address")
    timestamp: str = Field(..., description="ISO-format timestamp")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    email_id:    str = Field(..., description="ID of the current email to process")
    subject:     str = Field(..., description="Subject of the current email")
    body:        str = Field(..., description="Body of the current email")
    sender:      str = Field(..., description="Sender of the current email")
    timestamp:   str = Field(..., description="When the email was received")
    step:        int = Field(..., description="Current step number (0-indexed)")
    total_steps: int = Field(..., description="Total number of emails in this task")
    task_id:     str = Field(..., description="ID of the current task")
    history:     list[dict[str, Any]] = Field(default_factory=list, description="Previous actions taken this episode")
    done:              bool          = Field(default=False,  description="True if all emails are processed")
    message:           str           = Field(default="",     description="Optional message from the environment")
    sla_hours_remaining: Optional[float] = Field(default=None, description="Estimated SLA urgency window in hours")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    email_id: str         = Field(..., description="ID of the email being acted upon")
    category: Category    = Field(..., description="Email category classification")
    priority: Priority    = Field(..., description="Assigned priority level")
    tags:     list[str]   = Field(default_factory=list, description="Topic tags")
    route_to: RouteTarget = Field(..., description="Which team/folder to route the email to")
    notes:    str         = Field(default="", description="Optional agent reasoning notes")


# ---------------------------------------------------------------------------
# Reward — strictly within (0, 1)
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    step_score:       float            = Field(..., gt=0.0, lt=1.0, description="Score for this single action, strictly in (0, 1)")
    cumulative_score: float            = Field(..., gt=0.0, lt=1.0, description="Running average score, strictly in (0, 1)")
    penalty:          float            = Field(default=0.0, ge=0.0, description="Penalty applied")
    breakdown:        dict[str, float] = Field(default_factory=dict, description="Per-dimension scores")
    feedback:         str              = Field(default="", description="Human-readable feedback")

    @field_validator('step_score', 'cumulative_score', mode='before')
    @classmethod
    def clamp_reward_scores(cls, v: Any) -> float:
        return _clamp_score(float(v))


# ---------------------------------------------------------------------------
# Task descriptor
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id:        str
    name:           str
    difficulty:     Difficulty
    description:    str
    email_count:    int
    max_steps:      int
    pass_threshold: float


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    task_id:          str
    step:             int
    total_steps:      int
    done:             bool
    cumulative_score: float
    scores_per_step:  list[float]
    actions_taken:    list[dict[str, Any]]
    emails_remaining: list[str]
    emails_processed: list[str]


# ---------------------------------------------------------------------------
# Action schema models for /tasks endpoint
# ---------------------------------------------------------------------------

class ActionFieldSchema(BaseModel):
    name:        str            = Field(..., description="Field name in Action")
    type:        str            = Field(..., description="Data type")
    required:    bool           = Field(..., description="Whether the field is required")
    values:      Optional[list[str]] = Field(None, description="Allowed enum values")
    description: str            = Field(..., description="What this field means")


class TasksResponse(BaseModel):
    tasks:         list[TaskInfo]          = Field(..., description="All available tasks")
    action_schema: list[ActionFieldSchema] = Field(..., description="Fields required for a step action")


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(..., description="Which task to start")


class ResetResponse(BaseModel):
    session_id:  str         = Field(..., description="Session ID for all subsequent calls")
    observation: Observation
    task_info:   TaskInfo
    message:     str         = "Environment reset successfully."


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Optional[Observation] = None
    reward:      Reward
    done:        bool
    info:        dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    episode_state:       EpisodeState
    current_observation: Optional[Observation] = None


class GraderRequest(BaseModel):
    task_id: str
    actions: list[Action]


class GraderResponse(BaseModel):
    task_id:          str
    total_score:      float
    per_email_scores: list[dict[str, Any]]
    passed:           bool
    feedback:         str

    @field_validator('total_score', mode='before')
    @classmethod
    def clamp_total_score(cls, v: Any) -> float:
        """
        CRITICAL FIX: Force total_score strictly inside (0, 1).
        Validator checks this field in the HTTP response JSON.
        0.0 and 1.0 are both REJECTED by the validator — use sentinels.
        """
        return _clamp_score(float(v))


class BaselineRequest(BaseModel):
    task_ids: list[str] = Field(
        default=["easy_triage", "medium_triage", "hard_triage"],
        description="Tasks to run baseline on"
    )


class BaselineResponse(BaseModel):
    results: list[dict[str, Any]]
    summary: dict[str, float]

    @field_validator('summary', mode='before')
    @classmethod
    def clamp_summary_scores(cls, v: Any) -> dict:
        """Clamp all summary scores strictly inside (0, 1)."""
        if isinstance(v, dict):
            return {k: _clamp_score(float(val)) for k, val in v.items()}
        return v
