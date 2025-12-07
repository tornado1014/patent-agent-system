"""Patent Specification Writing Workflow Graph.

Implements the 9-step PatentSpec-KR workflow (E1-E9) using LangGraph.

Workflow Steps:
- E1: 발명 개요 수집
- E2: 핵심 정보 상세 질문
- E3: 선행기술 분석
- E4: 청구항 초안 작성 [Human Checkpoint]
- E5: 명세서 구조 설계
- E6: 본문 작성 [Human Checkpoint]
- E7: 도면 생성 및 설명
- E8: 품질 검증
- E9: 최종 완성 [Human Checkpoint]
"""

from datetime import datetime
from typing import Any, Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from patent_agent.agents.spec_writing import (
    ClaimDrafterAgent,
    DrawingGeneratorAgent,
    InventionAnalyzerAgent,
    PriorArtSearcherAgent,
    QualityCheckerAgent,
    SpecWriterAgent,
)
from patent_agent.config import settings
from patent_agent.graphs.base_graph import BasePatentGraph
from patent_agent.state.spec_writing import (
    HUMAN_CHECKPOINTS,
    SpecWritingState,
    SpecWritingStep,
)
from patent_agent.state.copilotkit_state import (
    ClaimReviewEvent,
    FinalApprovalEvent,
    SpecReviewEvent,
)

logger = structlog.get_logger(__name__)


class SpecWritingGraph(BasePatentGraph[SpecWritingState]):
    """LangGraph implementation of the 9-step specification writing workflow.

    This graph orchestrates the following agents:
    - InventionAnalyzerAgent (E1, E2)
    - PriorArtSearcherAgent (E3)
    - ClaimDrafterAgent (E4)
    - SpecWriterAgent (E5, E6)
    - DrawingGeneratorAgent (E7)
    - QualityCheckerAgent (E8)

    Example:
        >>> graph = SpecWritingGraph()
        >>> initial_state = {
        ...     "session_id": "session-123",
        ...     "current_workflow": "spec_writing",
        ...     "invention_disclosure": "발명 신고서 내용...",
        ...     "inventor_name": "홍길동",
        ...     "applicant_name": "테스트 주식회사",
        ... }
        >>> result = await graph.run(initial_state)
    """

    def __init__(
        self,
        checkpointer: MemorySaver | None = None,
    ):
        """Initialize the specification writing workflow graph.

        Args:
            checkpointer: Optional checkpointer for state persistence
        """
        super().__init__(
            name="spec_writing",
            checkpointer=checkpointer,
        )

        # Initialize agents
        self._invention_analyzer = InventionAnalyzerAgent()
        self._prior_art_searcher = PriorArtSearcherAgent()
        self._claim_drafter = ClaimDrafterAgent()
        self._spec_writer = SpecWriterAgent()
        self._drawing_generator = DrawingGeneratorAgent()
        self._quality_checker = QualityCheckerAgent()

    @property
    def state_class(self) -> type[SpecWritingState]:
        return SpecWritingState

    def get_initial_step(self) -> str:
        return "E1"

    def define_nodes(self, builder: StateGraph) -> None:
        """Add workflow-specific nodes to the graph."""
        # Step E1: Invention Overview Collection
        builder.add_node("E1", self._e1_invention_overview)

        # Step E2: Detailed Information Questions
        builder.add_node("E2", self._e2_detailed_questions)

        # Step E3: Prior Art Analysis
        builder.add_node("E3", self._e3_prior_art_analysis)

        # Step E4: Claim Drafting [Human Checkpoint]
        builder.add_node("E4", self._e4_claim_drafting)

        # Step E5: Specification Structure Design
        builder.add_node("E5", self._e5_structure_design)

        # Step E6: Body Writing [Human Checkpoint]
        builder.add_node("E6", self._e6_body_writing)

        # Step E7: Drawing Generation
        builder.add_node("E7", self._e7_drawing_generation)

        # Step E8: Quality Verification
        builder.add_node("E8", self._e8_quality_verification)

        # Step E9: Final Completion [Human Checkpoint]
        builder.add_node("E9", self._e9_final_completion)

        # Revision nodes for Reflexion loop
        builder.add_node("revise_claims", self._revise_claims)
        builder.add_node("revise_specification", self._revise_specification)

    def define_edges(self, builder: StateGraph) -> None:
        """Add workflow-specific edges to the graph."""
        # E1 -> E2 or E3 (depending on whether more info needed)
        builder.add_conditional_edges(
            "E1",
            self._route_after_e1,
            {
                "E2": "E2",
                "E3": "E3",
                "error": "error_handler",
            },
        )

        # E2 -> E3
        builder.add_edge("E2", "E3")

        # E3 -> E4
        builder.add_edge("E3", "E4")

        # E4 -> human_review or E5 (based on approval)
        builder.add_conditional_edges(
            "E4",
            self._route_after_checkpoint,
            {
                "human_review": "human_review",
                "next": "E5",
                "revise": "revise_claims",
            },
        )

        # Human review -> continue or revise
        builder.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "continue": self._get_continue_node,
                "revise_claims": "revise_claims",
                "revise_specification": "revise_specification",
                "end": END,
            },
        )

        # Claim revision -> E4 (re-check)
        builder.add_edge("revise_claims", "E4")

        # E5 -> E6
        builder.add_edge("E5", "E6")

        # E6 -> human_review or E7
        builder.add_conditional_edges(
            "E6",
            self._route_after_checkpoint,
            {
                "human_review": "human_review",
                "next": "E7",
                "revise": "revise_specification",
            },
        )

        # Specification revision -> E6 (re-check)
        builder.add_edge("revise_specification", "E6")

        # E7 -> E8
        builder.add_edge("E7", "E8")

        # E8 -> E9 or back to revision
        builder.add_conditional_edges(
            "E8",
            self._route_after_quality_check,
            {
                "E9": "E9",
                "revise_claims": "revise_claims",
                "revise_specification": "revise_specification",
            },
        )

        # E9 -> human_review or END
        builder.add_conditional_edges(
            "E9",
            self._route_after_checkpoint,
            {
                "human_review": "human_review",
                "next": END,
                "end": END,
            },
        )

        # Error handler -> END
        builder.add_edge("error_handler", END)

    # ─────────────────────────────────────────────────────────────
    # Node Implementations
    # ─────────────────────────────────────────────────────────────

    async def _e1_invention_overview(self, state: SpecWritingState) -> dict[str, Any]:
        """E1: Collect invention overview."""
        self._logger.info("executing_e1")

        result = await self._invention_analyzer.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _e2_detailed_questions(self, state: SpecWritingState) -> dict[str, Any]:
        """E2: Ask detailed questions for missing information."""
        self._logger.info("executing_e2")

        # In a real implementation, this would wait for user response
        # For now, we just move to E3 with existing information
        return {
            "current_step": "E3",
            "updated_at": datetime.now().isoformat(),
        }

    async def _e3_prior_art_analysis(self, state: SpecWritingState) -> dict[str, Any]:
        """E3: Analyze prior art."""
        self._logger.info("executing_e3")

        result = await self._prior_art_searcher.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _e4_claim_drafting(self, state: SpecWritingState) -> dict[str, Any]:
        """E4: Draft patent claims."""
        self._logger.info("executing_e4")

        result = await self._claim_drafter.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _e5_structure_design(self, state: SpecWritingState) -> dict[str, Any]:
        """E5: Design specification structure."""
        self._logger.info("executing_e5")

        # Update state to indicate E5
        state_with_step = dict(state)
        state_with_step["current_step"] = "E5"

        result = await self._spec_writer.process(state_with_step)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _e6_body_writing(self, state: SpecWritingState) -> dict[str, Any]:
        """E6: Write specification body."""
        self._logger.info("executing_e6")

        # Update state to indicate E6
        state_with_step = dict(state)
        state_with_step["current_step"] = "E6"

        result = await self._spec_writer.process(state_with_step)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _e7_drawing_generation(self, state: SpecWritingState) -> dict[str, Any]:
        """E7: Generate drawings and descriptions."""
        self._logger.info("executing_e7")

        result = await self._drawing_generator.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _e8_quality_verification(self, state: SpecWritingState) -> dict[str, Any]:
        """E8: Perform quality verification."""
        self._logger.info("executing_e8")

        result = await self._quality_checker.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _e9_final_completion(self, state: SpecWritingState) -> dict[str, Any]:
        """E9: Final completion and packaging."""
        self._logger.info("executing_e9")

        # Finalize the specification
        return {
            "final_claims": state.get("draft_claims"),
            "final_specification": state.get("draft_specification"),
            "abstract": await self._generate_abstract(state),
            "current_step": "complete",
            "human_approval_required": True,
            "pending_approval_checkpoint": "E9",
            "updated_at": datetime.now().isoformat(),
        }

    async def _revise_claims(self, state: SpecWritingState) -> dict[str, Any]:
        """Revise claims based on feedback."""
        self._logger.info("revising_claims")

        feedback = self._extract_feedback(state, "claims")
        result = await self._claim_drafter.revise_claims(state, feedback)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _revise_specification(self, state: SpecWritingState) -> dict[str, Any]:
        """Revise specification based on feedback."""
        self._logger.info("revising_specification")

        # Re-run specification writing with updated context
        state_copy = dict(state)
        state_copy["current_step"] = "E6"
        state_copy["iteration_count"] = state.get("iteration_count", 0) + 1

        result = await self._spec_writer.process(state_copy)
        result["updated_at"] = datetime.now().isoformat()

        return result

    # ─────────────────────────────────────────────────────────────
    # Routing Functions
    # ─────────────────────────────────────────────────────────────

    def _route_after_e1(self, state: SpecWritingState) -> str:
        """Route after E1 based on whether more information is needed."""
        if state.get("is_error_state"):
            return "error"

        follow_up = state.get("follow_up_questions", [])
        if follow_up:
            return "E2"
        return "E3"

    def _route_after_checkpoint(
        self, state: SpecWritingState
    ) -> Literal["human_review", "next", "revise", "end"]:
        """Route after a checkpoint step."""
        current_step = state.get("current_step", "")

        # Check for errors
        if state.get("is_error_state"):
            return "end"

        # Check if human review is required
        if settings.workflow.enable_human_in_loop:
            if state.get("human_approval_required"):
                return "human_review"

        # Check if revision is needed based on review comments
        review_comments = state.get("review_comments", [])
        critical_issues = [
            c for c in review_comments
            if c.get("severity") in ("critical", "high")
        ]

        if critical_issues:
            # Determine what to revise based on current step
            if current_step in ("E4",):
                return "revise"
            elif current_step in ("E6",):
                return "revise"

        return "next"

    def _route_after_human_review(
        self, state: SpecWritingState
    ) -> Literal["continue", "revise_claims", "revise_specification", "end"]:
        """Route after human review."""
        human_feedback = state.get("human_feedback", [])
        if not human_feedback:
            return "continue"

        latest_feedback = human_feedback[-1] if human_feedback else {}

        if not latest_feedback.get("approved", False):
            # Determine what needs revision
            checkpoint = state.get("pending_approval_checkpoint", "")
            if checkpoint == "E4":
                return "revise_claims"
            elif checkpoint == "E6":
                return "revise_specification"
            return "end"

        return "continue"

    def _get_continue_node(self, state: SpecWritingState) -> str:
        """Get the next node to continue to after human review."""
        checkpoint = state.get("pending_approval_checkpoint", "")

        step_map = {
            "E4": "E5",
            "E6": "E7",
            "E9": END,
        }

        return step_map.get(checkpoint, END)

    def _route_after_quality_check(
        self, state: SpecWritingState
    ) -> Literal["E9", "revise_claims", "revise_specification"]:
        """Route after quality check based on results."""
        quality_score = state.get("quality_score", {})
        overall = quality_score.get("overall", 0)

        if overall >= settings.workflow.quality_threshold:
            return "E9"

        # Check what needs revision
        review_comments = state.get("review_comments", [])

        claim_issues = [c for c in review_comments if c.get("section") == "claims"]
        spec_issues = [c for c in review_comments if c.get("section") != "claims"]

        # Check iteration count
        iteration = state.get("iteration_count", 0)
        if iteration >= settings.workflow.max_iterations:
            return "E9"  # Proceed anyway after max iterations

        # Prioritize claim issues
        if claim_issues and len(claim_issues) > len(spec_issues):
            return "revise_claims"
        elif spec_issues:
            return "revise_specification"

        return "E9"

    def _get_next_step(self, state: SpecWritingState) -> str:
        """Get the next step based on current state."""
        current = state.get("current_step", "E1")

        step_order = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]
        try:
            idx = step_order.index(current)
            if idx < len(step_order) - 1:
                return step_order[idx + 1]
        except ValueError:
            pass

        return END

    # ─────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────

    async def _generate_abstract(self, state: SpecWritingState) -> str:
        """Generate patent abstract."""
        spec = state.get("draft_specification", {})
        claims = state.get("draft_claims", {})

        # Get first independent claim
        first_claim = ""
        for claim in claims.get("claims", []):
            if claim.get("claim_type") == "independent":
                first_claim = claim.get("full_text", "")
                break

        # Generate abstract based on specification and claims
        # In production, would use LLM for this
        effects = spec.get("effects", "")
        means = spec.get("means_to_solve", "")

        if means and effects:
            abstract = f"{means[:200]}... {effects[:100]}"
        else:
            abstract = first_claim[:300] if first_claim else "요약서 생성 필요"

        return abstract

    def _extract_feedback(self, state: SpecWritingState, section: str) -> str:
        """Extract feedback for a specific section."""
        review_comments = state.get("review_comments", [])
        human_feedback = state.get("human_feedback", [])

        feedback_parts = []

        # Add review comments for section
        for comment in review_comments:
            if comment.get("section") == section or section == "all":
                feedback_parts.append(f"- {comment.get('issue', '')}")

        # Add human feedback
        for hf in human_feedback:
            if hf.get("comments"):
                feedback_parts.append(f"- (사용자) {hf['comments']}")

        return "\n".join(feedback_parts) if feedback_parts else "수정 필요"

    def _build_interrupt_event(
        self, state: SpecWritingState, checkpoint: str
    ) -> dict[str, Any]:
        """Build workflow-specific interrupt event for human review.

        Overrides base class to provide appropriate event data for each checkpoint:
        - E4: ClaimReviewEvent with draft claims
        - E6: SpecReviewEvent with draft specification
        - E9: FinalApprovalEvent with final claims, specification, and abstract
        """
        quality_score = state.get("quality_score", {
            "overall": 0,
            "completeness": 0,
            "accuracy": 0,
            "compliance": 0,
            "comments": [],
        })

        if checkpoint == "E4":
            # Claim review checkpoint
            draft_claims = state.get("draft_claims", {})
            claims_list = draft_claims.get("claims", []) if isinstance(draft_claims, dict) else []

            event = ClaimReviewEvent(
                checkpoint=checkpoint,
                claims=claims_list,
                quality_score=quality_score,
                message="청구항 초안을 검토해주세요. 독립항과 종속항의 구조, 기술적 범위, 표현의 정확성을 확인하세요.",
            )
            return event.to_dict()

        elif checkpoint == "E6":
            # Specification review checkpoint
            draft_spec = state.get("draft_specification", {})

            event = SpecReviewEvent(
                checkpoint=checkpoint,
                specification=draft_spec if isinstance(draft_spec, dict) else {},
                quality_score=quality_score,
                message="명세서 본문을 검토해주세요. 발명의 상세한 설명, 실시예, 청구항과의 일관성을 확인하세요.",
            )
            return event.to_dict()

        elif checkpoint == "E9":
            # Final approval checkpoint
            final_claims = state.get("final_claims") or state.get("draft_claims", {})
            claims_list = final_claims.get("claims", []) if isinstance(final_claims, dict) else []
            final_spec = state.get("final_specification") or state.get("draft_specification", {})
            abstract = state.get("abstract", "")

            event = FinalApprovalEvent(
                checkpoint=checkpoint,
                final_claims=claims_list,
                final_specification=final_spec if isinstance(final_spec, dict) else {},
                abstract=abstract,
                quality_score=quality_score,
                message="최종 명세서를 검토해주세요. 모든 구성요소가 완성되었는지 확인하고 출원 승인을 결정하세요.",
            )
            return event.to_dict()

        # Fallback to base implementation for unknown checkpoints
        return super()._build_interrupt_event(state, checkpoint)


def create_spec_writing_graph(
    checkpointer: MemorySaver | None = None,
) -> SpecWritingGraph:
    """Factory function to create a specification writing graph.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Configured SpecWritingGraph instance
    """
    return SpecWritingGraph(checkpointer=checkpointer)
