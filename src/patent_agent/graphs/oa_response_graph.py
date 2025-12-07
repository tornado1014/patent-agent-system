"""OA (Office Action) Response Workflow Graph.

Implements the 5-phase PALLAS-EVIDENCE workflow using LangGraph.

Workflow Phases:
- P1: 문서 수집 및 검증 (DocParserAgent)
- P2: 거절이유 분석 (RejectionAnalyzerAgent)
- P3: 비판적 검토 (RebuttalWriterAgent)
- P4: 보정안 생성 (AmendmentDrafterAgent)
- P5: 최종 보고서 (ReportGeneratorAgent)

Zero Hallucination Laws (LAW-6 to LAW-10) are enforced throughout.
"""

from datetime import datetime
from typing import Any, Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from patent_agent.agents.oa_response import (
    AmendmentDrafterAgent,
    DocParserAgent,
    RebuttalWriterAgent,
    RejectionAnalyzerAgent,
    ReportGeneratorAgent,
)
from patent_agent.config import settings
from patent_agent.graphs.base_graph import BasePatentGraph
from patent_agent.state.copilotkit_state import (
    AmendmentReviewEvent,
    ReportApprovalEvent,
)
from patent_agent.state.oa_response import OAResponseState

logger = structlog.get_logger(__name__)


class OAResponseGraph(BasePatentGraph[OAResponseState]):
    """LangGraph implementation of the 5-phase OA response workflow.

    This graph orchestrates the following agents:
    - DocParserAgent (P1)
    - RejectionAnalyzerAgent (P2)
    - RebuttalWriterAgent (P3)
    - AmendmentDrafterAgent (P4)
    - ReportGeneratorAgent (P5)

    Features:
    - Zero Hallucination enforcement at all phases
    - 3-layer verification at P2
    - Evidence chain tracking at P3
    - Specification support verification at P4
    - Final hallucination check at P5

    Example:
        >>> graph = OAResponseGraph()
        >>> initial_state = {
        ...     "session_id": "session-123",
        ...     "current_workflow": "oa_response",
        ...     "oa_document_text": "OA 문서 내용...",
        ...     "specification_text": "명세서 내용...",
        ... }
        >>> result = await graph.run(initial_state)
    """

    def __init__(
        self,
        checkpointer: MemorySaver | None = None,
    ):
        """Initialize the OA response workflow graph.

        Args:
            checkpointer: Optional checkpointer for state persistence
        """
        super().__init__(
            name="oa_response",
            checkpointer=checkpointer,
        )

        # Initialize agents
        self._doc_parser = DocParserAgent()
        self._rejection_analyzer = RejectionAnalyzerAgent()
        self._rebuttal_writer = RebuttalWriterAgent()
        self._amendment_drafter = AmendmentDrafterAgent()
        self._report_generator = ReportGeneratorAgent()

    @property
    def state_class(self) -> type[OAResponseState]:
        return OAResponseState

    def get_initial_step(self) -> str:
        return "P1"

    def define_nodes(self, builder: StateGraph) -> None:
        """Add workflow-specific nodes to the graph."""
        # Phase 1: Document Collection and Verification
        builder.add_node("P1", self._p1_document_collection)

        # Phase 2: Rejection Analysis
        builder.add_node("P2", self._p2_rejection_analysis)

        # Phase 3: Critical Review
        builder.add_node("P3", self._p3_critical_review)

        # Phase 4: Amendment Generation
        builder.add_node("P4", self._p4_amendment_generation)

        # Phase 5: Final Report
        builder.add_node("P5", self._p5_final_report)

        # Revision nodes for Reflexion loop
        builder.add_node("revise_analysis", self._revise_analysis)
        builder.add_node("revise_rebuttal", self._revise_rebuttal)
        builder.add_node("revise_amendment", self._revise_amendment)

    def define_edges(self, builder: StateGraph) -> None:
        """Add workflow-specific edges to the graph."""
        # P1 -> P2 (or error)
        builder.add_conditional_edges(
            "P1",
            self._route_after_p1,
            {
                "P2": "P2",
                "error": "error_handler",
            },
        )

        # P2 -> P3 (or revise if verification failed)
        builder.add_conditional_edges(
            "P2",
            self._route_after_p2,
            {
                "P3": "P3",
                "revise": "revise_analysis",
                "error": "error_handler",
            },
        )

        # Revise analysis -> P2
        builder.add_edge("revise_analysis", "P2")

        # P3 -> P4
        builder.add_conditional_edges(
            "P3",
            self._route_after_p3,
            {
                "P4": "P4",
                "revise": "revise_rebuttal",
                "error": "error_handler",
            },
        )

        # Revise rebuttal -> P3
        builder.add_edge("revise_rebuttal", "P3")

        # P4 -> P5 (or revise if insufficient amendments)
        builder.add_conditional_edges(
            "P4",
            self._route_after_p4,
            {
                "P5": "P5",
                "revise": "revise_amendment",
                "human_review": "human_review",
            },
        )

        # Revise amendment -> P4
        builder.add_edge("revise_amendment", "P4")

        # P5 -> human_review or END
        builder.add_conditional_edges(
            "P5",
            self._route_after_p5,
            {
                "human_review": "human_review",
                "end": END,
            },
        )

        # Human review -> continue or revise
        builder.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "continue": END,
                "revise_analysis": "revise_analysis",
                "revise_rebuttal": "revise_rebuttal",
                "revise_amendment": "revise_amendment",
                "end": END,
            },
        )

        # Error handler -> END
        builder.add_edge("error_handler", END)

    # ─────────────────────────────────────────────────────────────
    # Node Implementations
    # ─────────────────────────────────────────────────────────────

    async def _p1_document_collection(
        self, state: OAResponseState
    ) -> dict[str, Any]:
        """P1: Document collection and verification."""
        self._logger.info("executing_p1")

        result = await self._doc_parser.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p2_rejection_analysis(
        self, state: OAResponseState
    ) -> dict[str, Any]:
        """P2: Rejection analysis with 3-layer verification."""
        self._logger.info("executing_p2")

        result = await self._rejection_analyzer.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p3_critical_review(
        self, state: OAResponseState
    ) -> dict[str, Any]:
        """P3: Critical review and evidence chain building."""
        self._logger.info("executing_p3")

        result = await self._rebuttal_writer.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p4_amendment_generation(
        self, state: OAResponseState
    ) -> dict[str, Any]:
        """P4: Amendment generation with specification support."""
        self._logger.info("executing_p4")

        result = await self._amendment_drafter.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p5_final_report(
        self, state: OAResponseState
    ) -> dict[str, Any]:
        """P5: Final report generation and hallucination check."""
        self._logger.info("executing_p5")

        result = await self._report_generator.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _revise_analysis(
        self, state: OAResponseState
    ) -> dict[str, Any]:
        """Revise rejection analysis based on verification failures."""
        self._logger.info("revising_analysis")

        # Increment iteration count
        iteration = state.get("iteration_count", 0) + 1

        # Clear previous analysis to retry
        return {
            "rejection_analyses": [],
            "extraction_verified": False,
            "mapping_verified": False,
            "completeness_verified": False,
            "iteration_count": iteration,
            "updated_at": datetime.now().isoformat(),
        }

    async def _revise_rebuttal(
        self, state: OAResponseState
    ) -> dict[str, Any]:
        """Revise rebuttal arguments."""
        self._logger.info("revising_rebuttal")

        iteration = state.get("iteration_count", 0) + 1

        return {
            "rebuttal_points": [],
            "technical_differences": [],
            "logical_gaps": [],
            "iteration_count": iteration,
            "updated_at": datetime.now().isoformat(),
        }

    async def _revise_amendment(
        self, state: OAResponseState
    ) -> dict[str, Any]:
        """Revise amendment options."""
        self._logger.info("revising_amendment")

        iteration = state.get("iteration_count", 0) + 1

        return {
            "amendment_options": [],
            "recommended_amendment": None,
            "iteration_count": iteration,
            "updated_at": datetime.now().isoformat(),
        }

    # ─────────────────────────────────────────────────────────────
    # Routing Functions
    # ─────────────────────────────────────────────────────────────

    def _route_after_p1(
        self, state: OAResponseState
    ) -> Literal["P2", "error"]:
        """Route after P1 based on document verification."""
        if state.get("is_error_state"):
            return "error"

        documents_verified = state.get("documents_verified", {})
        if not documents_verified.get("oa_document"):
            return "error"

        return "P2"

    def _route_after_p2(
        self, state: OAResponseState
    ) -> Literal["P3", "revise", "error"]:
        """Route after P2 based on 3-layer verification."""
        if state.get("is_error_state"):
            return "error"

        # Check 3-layer verification
        extraction_ok = state.get("extraction_verified", False)
        mapping_ok = state.get("mapping_verified", False)
        completeness_ok = state.get("completeness_verified", False)

        # All layers must pass
        if not (extraction_ok and mapping_ok and completeness_ok):
            # Check iteration limit
            if state.get("iteration_count", 0) >= settings.workflow.max_iterations:
                self._logger.warning(
                    "max_iterations_reached",
                    phase="P2",
                )
                return "P3"  # Proceed anyway
            return "revise"

        return "P3"

    def _route_after_p3(
        self, state: OAResponseState
    ) -> Literal["P4", "revise", "error"]:
        """Route after P3 based on rebuttal quality."""
        if state.get("is_error_state"):
            return "error"

        rebuttal_points = state.get("rebuttal_points", [])

        # Must have at least one rebuttal point
        if not rebuttal_points:
            if state.get("iteration_count", 0) >= settings.workflow.max_iterations:
                return "P4"  # Proceed anyway
            return "revise"

        # Check for hallucination issues
        for point in rebuttal_points:
            if point.get("confidence") == "불확실":
                # Allow but log warning
                self._logger.warning(
                    "uncertain_rebuttal",
                    argument=point.get("argument", "")[:50],
                )

        return "P4"

    def _route_after_p4(
        self, state: OAResponseState
    ) -> Literal["P5", "revise", "human_review"]:
        """Route after P4 based on amendment quality."""
        amendment_options = state.get("amendment_options", [])

        # LAW-3: Must have at least 2 amendments
        if len(amendment_options) < 2:
            if state.get("iteration_count", 0) >= settings.workflow.max_iterations:
                return "P5"  # Proceed anyway
            return "revise"

        # Check for high-risk amendments
        high_risk = [
            a for a in amendment_options
            if a.get("new_matter_risk") == "high"
        ]

        if high_risk and settings.workflow.enable_human_in_loop:
            return "human_review"

        return "P5"

    def _route_after_p5(
        self, state: OAResponseState
    ) -> Literal["human_review", "end"]:
        """Route after P5 based on verification stamp."""
        verification = state.get("verification_stamp", {})

        # Check hallucination status
        if verification.get("hallucination_check") == "FAILED":
            self._logger.error(
                "hallucination_check_failed",
                verbatim_accuracy=verification.get("verbatim_accuracy"),
            )
            # Still require human review
            return "human_review"

        # Always require human review for final output
        if settings.workflow.enable_human_in_loop:
            return "human_review"

        return "end"

    def _route_after_human_review(
        self, state: OAResponseState
    ) -> Literal["continue", "revise_analysis", "revise_rebuttal", "revise_amendment", "end"]:
        """Route after human review."""
        human_feedback = state.get("human_feedback", [])

        if not human_feedback:
            return "continue"

        latest = human_feedback[-1] if human_feedback else {}

        if latest.get("approved", False):
            return "continue"

        # Determine what needs revision based on feedback
        feedback_text = latest.get("comments", "").lower()

        if "분석" in feedback_text or "거절이유" in feedback_text:
            return "revise_analysis"
        elif "반박" in feedback_text or "논리" in feedback_text:
            return "revise_rebuttal"
        elif "보정" in feedback_text or "청구항" in feedback_text:
            return "revise_amendment"

        return "end"

    def _get_next_step(self, state: OAResponseState) -> str:
        """Get the next step based on current state."""
        current = state.get("current_step", "P1")

        phase_order = ["P1", "P2", "P3", "P4", "P5"]
        try:
            idx = phase_order.index(current)
            if idx < len(phase_order) - 1:
                return phase_order[idx + 1]
        except ValueError:
            pass

        return END

    def _build_interrupt_event(
        self, state: OAResponseState, checkpoint: str
    ) -> dict[str, Any]:
        """Build workflow-specific interrupt event for human review.

        Overrides base class to provide appropriate event data for each checkpoint:
        - P4: AmendmentReviewEvent with amendment options and recommendation
        - P5: ReportApprovalEvent with final report and selected amendment
        """
        if checkpoint == "P4":
            # Amendment review checkpoint
            amendments = state.get("amendment_options", [])
            recommended = state.get("recommended_amendment")

            # Convert amendment objects to dicts if needed
            amendment_list = []
            for a in amendments:
                if isinstance(a, dict):
                    amendment_list.append(a)
                elif hasattr(a, "__dict__"):
                    amendment_list.append(a.__dict__)
                else:
                    amendment_list.append({"data": str(a)})

            # Get recommended amendment ID
            recommended_id = ""
            if recommended:
                if isinstance(recommended, dict):
                    recommended_id = recommended.get("amendment_id", "A")
                elif hasattr(recommended, "amendment_id"):
                    recommended_id = recommended.amendment_id
                else:
                    recommended_id = "A"
            elif amendment_list:
                recommended_id = amendment_list[0].get("amendment_id", "A")

            event = AmendmentReviewEvent(
                checkpoint=checkpoint,
                amendments=amendment_list,
                recommended=recommended_id,
                message="보정안을 검토해주세요. LAW-3에 따라 최소 2개의 보정안 중 적합한 안을 선택하고, 명세서 지원 근거를 확인하세요.",
            )
            return event.to_dict()

        elif checkpoint == "P5":
            # Report approval checkpoint
            final_report = state.get("final_report", {})
            selected = state.get("selected_amendment") or state.get("recommended_amendment")

            # Convert selected amendment to dict
            selected_dict = None
            if selected:
                if isinstance(selected, dict):
                    selected_dict = selected
                elif hasattr(selected, "__dict__"):
                    selected_dict = selected.__dict__
                else:
                    selected_dict = {"data": str(selected)}

            event = ReportApprovalEvent(
                checkpoint=checkpoint,
                report=final_report if isinstance(final_report, dict) else {},
                selected_amendment=selected_dict,
                message="OA 대응 보고서를 검토해주세요. 거절이유 분석, 보정안, 의견서 초안이 정확한지 확인하고 최종 승인하세요.",
            )
            return event.to_dict()

        # Fallback to base implementation for unknown checkpoints
        return super()._build_interrupt_event(state, checkpoint)


def create_oa_response_graph(
    checkpointer: MemorySaver | None = None,
) -> OAResponseGraph:
    """Factory function to create an OA response graph.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Configured OAResponseGraph instance
    """
    return OAResponseGraph(checkpointer=checkpointer)
