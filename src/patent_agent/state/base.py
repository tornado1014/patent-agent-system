"""
Base state definitions shared across all patent workflows.

All workflow-specific states extend BasePatentState.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Literal, TypedDict


# Type aliases for workflow identification
WorkflowType = Literal[
    "spec_writing",
    "oa_response",
    "prior_art",
    "translation",
    "analysis",
]

ConfidenceLevel = Literal["확실", "가능", "불확실"]  # 🟢, 🟡, 🔴


class Evidence(TypedDict):
    """
    Evidence record for traceability (LAW-7 compliance).

    All claims and analysis must cite evidence with this structure.
    """

    source: str  # Document source (e.g., "출원서", "D1", "OA")
    page: int | None  # Page number
    paragraph: int | None  # Paragraph number
    claim_number: int | None  # Claim number if applicable
    verbatim_text: str  # Original text (no modifications - LAW-6)
    confidence: ConfidenceLevel
    timestamp: str  # ISO-8601 format


class QualityScore(TypedDict):
    """Quality assessment scores."""

    overall: int  # 0-100
    completeness: int
    accuracy: int
    compliance: int
    comments: list[str]


class HumanFeedback(TypedDict):
    """Human-in-the-loop feedback record."""

    checkpoint: str  # e.g., "E4", "P3"
    approved: bool
    comments: str
    timestamp: str
    reviewer: str | None


class ReviewComment(TypedDict):
    """Review comment from Reflexion critique."""

    section: str
    issue: str
    severity: Literal["critical", "high", "medium", "low"]
    suggestion: str


class BasePatentState(TypedDict, total=False):
    """
    Base state for all patent workflows.

    All domain-specific states extend this with their own fields.
    Uses LangGraph's Annotated[list, add] pattern for message accumulation.
    """

    # ─────────────────────────────────────────────────────────────
    # Core Workflow State
    # ─────────────────────────────────────────────────────────────
    messages: Annotated[list, add]  # Accumulated conversation
    current_workflow: WorkflowType
    current_step: str  # Workflow-specific step (E1-E9, P1-P5, etc.)

    # ─────────────────────────────────────────────────────────────
    # Quality Control (Reflexion Loop)
    # ─────────────────────────────────────────────────────────────
    iteration_count: int  # Current Reflexion iteration
    max_iterations: int  # Maximum allowed iterations
    quality_score: QualityScore
    review_comments: list[ReviewComment]  # From critique node

    # ─────────────────────────────────────────────────────────────
    # Human-in-the-Loop
    # ─────────────────────────────────────────────────────────────
    human_approval_required: bool
    human_feedback: list[HumanFeedback]
    pending_approval_checkpoint: str | None

    # ─────────────────────────────────────────────────────────────
    # Evidence Chain (Zero Hallucination)
    # ─────────────────────────────────────────────────────────────
    evidence_chain: list[Evidence]

    # ─────────────────────────────────────────────────────────────
    # Error Handling
    # ─────────────────────────────────────────────────────────────
    error_messages: list[str]
    is_error_state: bool

    # ─────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────
    created_at: str  # ISO-8601
    updated_at: str
    session_id: str


def create_initial_state(
    workflow: WorkflowType,
    session_id: str | None = None,
) -> BasePatentState:
    """Create initial base state for a workflow."""
    now = datetime.now().isoformat()
    return BasePatentState(
        messages=[],
        current_workflow=workflow,
        current_step="",
        iteration_count=0,
        max_iterations=3,
        quality_score=QualityScore(
            overall=0,
            completeness=0,
            accuracy=0,
            compliance=0,
            comments=[],
        ),
        review_comments=[],
        human_approval_required=False,
        human_feedback=[],
        pending_approval_checkpoint=None,
        evidence_chain=[],
        error_messages=[],
        is_error_state=False,
        created_at=now,
        updated_at=now,
        session_id=session_id or f"{workflow}_{now}",
    )
