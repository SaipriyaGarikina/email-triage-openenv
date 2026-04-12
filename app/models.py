"""
models.py — Final Fixed Version
All score fields clamped strictly within (0.01, 0.99).
NO gt/lt constraints on Reward — those cause ValidationError crashes on 0.0 input.
Instead: field_validator with mode='before' clamps at input silently.
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
    if f != f:          # NaN check
        return _MIN_SCORE
    if f <= 0.0:
        return _MIN_SCORE
    if f >= 1.0:
        return _MAX_SCORE
    return round(f, 4)


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


class Email(BaseModel):
    email_id:  str
    subject:   str
    body:      str
    sender:    str
    timestamp: str


class Observation(BaseModel):
    email_id:            str
    subject:             str
    body:                str
    sender:              str
    timestamp:           str
    step:                int
    total_steps:         int
    task_id:             str
    history:             list[dict[str, Any]] = Field(default_factory=list)
    done:                bool                 = False
    message:             str                  = ""
    sla_hours_remaining: Optional[float]      = None


class Action(BaseModel):
    email_id: str
    category: Category
    priority: Priority
    tags:     list[str]   = Field(default_factory=list)
    route_to: RouteTarget
    notes:    str         = ""


class Reward(BaseModel):
    """
    IMPORTANT: No gt/lt constraints — those raise ValidationError on 0.0 input
    and crash inference.py. Instead, field_validator clamps silently before
    Pydantic stores the value. All scores strictly in (0.01, 0.99).
    """
    step_score:       float            = Field(..., description="Score strictly in (0,1)")
    cumulative_score: float            = Field(..., description="Running avg strictly in (0,1)")
    penalty:          float            = Field(default=0.0)
    breakdown:        dict[str, float] = Field(default_factory=dict)
    feedback:         str              = ""

    @field_validator('step_score', 'cumulative_score', mode='before')
    @classmethod
    def clamp_reward_scores(cls, v: Any) -> float:
        return _clamp(v)


class TaskInfo(BaseModel):
    task_id:        str
    name:           str
    difficulty:     Difficulty
    description:    str
    email_count:    int
    max_steps:      int
    pass_threshold: float


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


class ActionFieldSchema(BaseModel):
    name:        str
    type:        str
    required:    bool
    values:      Optional[list[str]] = None
    description: str


class TasksResponse(BaseModel):
    tasks:         list[TaskInfo]
    action_schema: list[ActionFieldSchema]


class ResetRequest(BaseModel):
    task_id: str


class ResetResponse(BaseModel):
    session_id:  str
    observation: Observation
    task_info:   TaskInfo
    message:     str = "Environment reset successfully."


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
        return _clamp(v)


class BaselineRequest(BaseModel):
    task_ids: list[str] = Field(default=["easy_triage", "medium_triage", "hard_triage"])


class BaselineResponse(BaseModel):
    results: list[dict[str, Any]]
    summary: dict[str, float]

    @field_validator('summary', mode='before')
    @classmethod
    def clamp_summary_scores(cls, v: Any) -> dict:
        if isinstance(v, dict):
            return {k: _clamp(val) for k, val in v.items()}
        return v
