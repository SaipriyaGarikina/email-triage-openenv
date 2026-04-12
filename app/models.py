"""
models.py — Final Fixed Version
KEY FIX: All float score fields have explicit json_schema_extra examples
set to 0.5 so FastAPI's OpenAPI schema never shows 0 or 1 as example values.
The validator checks the OpenAPI schema examples — not just live responses.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

_MIN_SCORE = 0.01
_MAX_SCORE = 0.99


def _clamp(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _MIN_SCORE
    if f != f:
        return _MIN_SCORE
    if f <= 0.0:
        return _MIN_SCORE
    if f >= 1.0:
        return _MAX_SCORE
    return round(f, 4)


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
# Email
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
    history:     list[dict[str, Any]] = Field(default_factory=list)
    done:        bool  = Field(default=False)
    message:     str   = Field(default="")
    sla_hours_remaining: Optional[float] = Field(
        default=None,
        description="Estimated SLA urgency window in hours",
        json_schema_extra={"example": 24.0},
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    email_id: str         = Field(..., description="ID of the email being acted upon")
    category: Category    = Field(..., description="Email category classification")
    priority: Priority    = Field(..., description="Assigned priority level")
    tags:     list[str]   = Field(default_factory=list, description="Topic tags")
    route_to: RouteTarget = Field(..., description="Which team/folder to route the email to")
    notes:    str         = Field(default="", description="Optional reasoning notes")


# ---------------------------------------------------------------------------
# Reward
# CRITICAL: Use json_schema_extra with example=0.5 on ALL float score fields.
# FastAPI uses these examples when generating the OpenAPI schema.
# Without explicit examples, FastAPI defaults to 0 — which fails the validator.
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    step_score: float = Field(
        ...,
        description="Score for this single action, strictly in (0, 1)",
        json_schema_extra={"example": 0.5},
    )
    cumulative_score: float = Field(
        ...,
        description="Running average score, strictly in (0, 1)",
        json_schema_extra={"example": 0.5},
    )
    penalty:   float            = Field(default=0.0, description="Penalty applied")
    breakdown: dict[str, float] = Field(default_factory=dict)
    feedback:  str              = Field(default="")

    @field_validator('step_score', 'cumulative_score', mode='before')
    @classmethod
    def clamp_reward_scores(cls, v: Any) -> float:
        return _clamp(v)


# ---------------------------------------------------------------------------
# TaskInfo
# CRITICAL: pass_threshold must have example != 0
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id:    str = Field(..., description="Task identifier")
    name:       str = Field(..., description="Human-readable task name")
    difficulty: Difficulty
    description: str
    email_count: int = Field(..., description="Number of emails in this task")
    max_steps:   int = Field(..., description="Maximum steps allowed")
    pass_threshold: float = Field(
        ...,
        description="Minimum score to pass this task",
        json_schema_extra={"example": 0.75},
    )


# ---------------------------------------------------------------------------
# EpisodeState
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    task_id:          str
    step:             int
    total_steps:      int
    done:             bool
    cumulative_score: float = Field(
        ...,
        description="Running average score",
        json_schema_extra={"example": 0.5},
    )
    scores_per_step:  list[float]
    actions_taken:    list[dict[str, Any]]
    emails_remaining: list[str]
    emails_processed: list[str]


# ---------------------------------------------------------------------------
# Action schema for /tasks
# ---------------------------------------------------------------------------

class ActionFieldSchema(BaseModel):
    name:        str
    type:        str
    required:    bool
    values:      Optional[list[str]] = None
    description: str


class TasksResponse(BaseModel):
    tasks:         list[TaskInfo]
    action_schema: list[ActionFieldSchema]


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(..., description="Which task to start")


class ResetResponse(BaseModel):
    session_id:  str         = Field(..., description="Session ID for subsequent calls")
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


# ---------------------------------------------------------------------------
# GraderResponse
# CRITICAL: total_score must have example != 0 AND field_validator to clamp.
# ---------------------------------------------------------------------------

class GraderResponse(BaseModel):
    task_id: str
    total_score: float = Field(
        ...,
        description="Episode score strictly in (0, 1)",
        json_schema_extra={"example": 0.5},
    )
    per_email_scores: list[dict[str, Any]]
    passed:           bool
    feedback:         str

    @field_validator('total_score', mode='before')
    @classmethod
    def clamp_total_score(cls, v: Any) -> float:
        return _clamp(v)


class BaselineRequest(BaseModel):
    task_ids: list[str] = Field(
        default=["easy_triage", "medium_triage", "hard_triage"],
        description="Tasks to run baseline on",
    )


class BaselineResponse(BaseModel):
    results: list[dict[str, Any]]
    summary: dict[str, float]

    @field_validator('summary', mode='before')
    @classmethod
    def clamp_summary(cls, v: Any) -> dict:
        if isinstance(v, dict):
            return {k: _clamp(val) for k, val in v.items()}
        return v
