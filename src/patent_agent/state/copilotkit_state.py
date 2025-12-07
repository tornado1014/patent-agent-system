"""
CopilotKit-compatible state for Patent Agent System.

Extends BasePatentState with CopilotKitState for UI integration.
This enables real-time state synchronization between the LangGraph
backend and the React frontend via CopilotKit.
"""

from typing import Annotated, Any, Literal
from operator import add

from copilotkit import CopilotKitState

from patent_agent.state.base import (
    WorkflowType,
    ConfidenceLevel,
    QualityScore,
    HumanFeedback,
    ReviewComment,
    Evidence,
)


# Interrupt event types for Human-in-the-Loop
InterruptType = Literal[
    "claim_review",      # E4: 청구항 초안 검토
    "spec_review",       # E6: 명세서 본문 검토
    "final_approval",    # E9: 최종 승인
    "amendment_review",  # P4: 보정안 선택
    "report_approval",   # P5: 보고서 승인
    "prior_art_review",  # 선행기술 검토
]


class InterruptEvent:
    """
    Base class for interrupt events sent to frontend.

    These events trigger useLangGraphInterrupt hooks in the React app.
    """

    def __init__(
        self,
        interrupt_type: InterruptType,
        checkpoint: str,
        message: str,
        data: dict[str, Any] | None = None,
    ):
        self.type = interrupt_type
        self.checkpoint = checkpoint
        self.message = message
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LangGraph interrupt()."""
        return {
            "type": self.type,
            "checkpoint": self.checkpoint,
            "message": self.message,
            **self.data,
        }


class ClaimReviewEvent(InterruptEvent):
    """Interrupt event for claim review (E4 checkpoint)."""

    def __init__(
        self,
        checkpoint: str,
        claims: list[dict[str, Any]],
        quality_score: QualityScore,
        message: str = "청구항 초안을 검토하고 승인해주세요.",
    ):
        super().__init__(
            interrupt_type="claim_review",
            checkpoint=checkpoint,
            message=message,
            data={
                "claims": claims,
                "quality_score": quality_score,
            },
        )


class SpecReviewEvent(InterruptEvent):
    """Interrupt event for specification review (E6 checkpoint)."""

    def __init__(
        self,
        checkpoint: str,
        specification: dict[str, str],
        quality_score: QualityScore,
        message: str = "명세서 본문을 검토하고 승인해주세요.",
    ):
        super().__init__(
            interrupt_type="spec_review",
            checkpoint=checkpoint,
            message=message,
            data={
                "specification": specification,
                "quality_score": quality_score,
            },
        )


class AmendmentReviewEvent(InterruptEvent):
    """Interrupt event for amendment review (P4 checkpoint)."""

    def __init__(
        self,
        checkpoint: str,
        amendments: list[dict[str, Any]],
        recommended: str,
        message: str = "보정안을 검토하고 하나를 선택해주세요.",
    ):
        super().__init__(
            interrupt_type="amendment_review",
            checkpoint=checkpoint,
            message=message,
            data={
                "amendments": amendments,
                "recommended": recommended,
            },
        )


class FinalApprovalEvent(InterruptEvent):
    """Interrupt event for final approval (E9, P5 checkpoints)."""

    def __init__(
        self,
        checkpoint: str,
        final_claims: list[dict[str, Any]],
        final_specification: dict[str, str] | None = None,
        abstract: str | None = None,
        quality_score: QualityScore | None = None,
        message: str = "최종 결과물을 검토하고 승인해주세요.",
    ):
        super().__init__(
            interrupt_type="final_approval",
            checkpoint=checkpoint,
            message=message,
            data={
                "final_claims": final_claims,
                "final_specification": final_specification,
                "abstract": abstract,
                "quality_score": quality_score,
            },
        )


class ReportApprovalEvent(InterruptEvent):
    """Interrupt event for report approval (P5 checkpoint in OA response)."""

    def __init__(
        self,
        checkpoint: str,
        report: dict[str, Any],
        selected_amendment: dict[str, Any] | None = None,
        message: str = "OA 대응 보고서를 검토하고 승인해주세요.",
    ):
        super().__init__(
            interrupt_type="report_approval",
            checkpoint=checkpoint,
            message=message,
            data={
                "report": report,
                "selected_amendment": selected_amendment,
            },
        )


class PatentAgentCopilotKitState(CopilotKitState, total=False):
    """
    Combined state for CopilotKit integration.

    Inherits from CopilotKitState which provides:
    - messages: Accumulated conversation messages
    - copilotkit: CopilotKit actions and tools

    Adds Patent Agent specific fields from BasePatentState.
    """

    # ─────────────────────────────────────────────────────────────
    # Core Workflow State
    # ─────────────────────────────────────────────────────────────
    current_workflow: WorkflowType
    current_step: str  # Workflow-specific step (E1-E9, P1-P5, etc.)

    # ─────────────────────────────────────────────────────────────
    # Quality Control (Reflexion Loop)
    # ─────────────────────────────────────────────────────────────
    iteration_count: int  # Current Reflexion iteration
    max_iterations: int  # Maximum allowed iterations (default: 3)
    quality_score: QualityScore
    review_comments: list[ReviewComment]  # From critique node

    # ─────────────────────────────────────────────────────────────
    # Human-in-the-Loop
    # ─────────────────────────────────────────────────────────────
    human_approval_required: bool
    human_feedback: list[HumanFeedback]
    pending_approval_checkpoint: str | None

    # ─────────────────────────────────────────────────────────────
    # Evidence Chain (Zero Hallucination - LAW-6, LAW-7)
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

    # ─────────────────────────────────────────────────────────────
    # Workflow-specific data (dynamic, set by each workflow)
    # ─────────────────────────────────────────────────────────────
    workflow_data: dict[str, Any]

    # ─────────────────────────────────────────────────────────────
    # Draft outputs for Reflexion loop
    # ─────────────────────────────────────────────────────────────
    draft_claims: list[dict[str, Any]] | None  # For spec_writing
    draft_specification: dict[str, str] | None  # For spec_writing
    amendment_options: list[dict[str, Any]] | None  # For oa_response


def create_copilotkit_state(
    workflow: WorkflowType,
    session_id: str,
) -> PatentAgentCopilotKitState:
    """
    Create initial CopilotKit-compatible state for a workflow.

    Args:
        workflow: The workflow type to initialize
        session_id: Unique session identifier

    Returns:
        Initialized PatentAgentCopilotKitState
    """
    from datetime import datetime

    now = datetime.now().isoformat()

    return PatentAgentCopilotKitState(
        # CopilotKitState fields
        messages=[],

        # Workflow state
        current_workflow=workflow,
        current_step="",

        # Quality control
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

        # Human-in-the-Loop
        human_approval_required=False,
        human_feedback=[],
        pending_approval_checkpoint=None,

        # Evidence chain
        evidence_chain=[],

        # Error handling
        error_messages=[],
        is_error_state=False,

        # Metadata
        created_at=now,
        updated_at=now,
        session_id=session_id,

        # Workflow data
        workflow_data={},
        draft_claims=None,
        draft_specification=None,
        amendment_options=None,
    )
