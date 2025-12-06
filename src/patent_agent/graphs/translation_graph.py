"""Patent Translation (EN→KR) Workflow Graph.

Implements the 6-phase translation workflow using LangGraph.

Workflow Phases:
- P1: 문서 접수 및 초기 추론 (DocumentIntakeAgent)
- P2: 구조 분석 및 의존성 매핑 (StructureAnalyzerAgent)
- P3: 용어 일관성 구축 (GlossaryBuilderAgent)
- P4: 섹션별 스마트 번역 (SectionTranslatorAgent)
- P5: 품질 검증 (QualityVerifierAgent)
- P6: 결과물 생성 (OutputGeneratorAgent)

4 Absolute Laws (최고 우선순위):
- LAW-T-1: '상기' 사용 규칙
- LAW-T-2: 청구항 한 문장 원칙
- LAW-T-3: 권리범위 결정 용어
- LAW-T-4: 도면부호 괄호 규칙
"""

from datetime import datetime
from typing import Any, Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from patent_agent.agents.translation import (
    DocumentIntakeAgent,
    GlossaryBuilderAgent,
    OutputGeneratorAgent,
    QualityVerifierAgent,
    SectionTranslatorAgent,
    StructureAnalyzerAgent,
)
from patent_agent.config import get_settings
from patent_agent.graphs.base_graph import BasePatentGraph
from patent_agent.state.translation import TranslationState

logger = structlog.get_logger(__name__)


class TranslationGraph(BasePatentGraph[TranslationState]):
    """LangGraph implementation of the EN→KR patent translation workflow.

    This graph orchestrates the following agents:
    - DocumentIntakeAgent (P1)
    - StructureAnalyzerAgent (P2)
    - GlossaryBuilderAgent (P3)
    - SectionTranslatorAgent (P4)
    - QualityVerifierAgent (P5)
    - OutputGeneratorAgent (P6)

    Features:
    - 4 Absolute Laws enforcement
    - TAC section priority handling
    - 5C quality criteria assessment
    - Human-in-the-loop for glossary conflicts and critical issues
    - Automatic revision for quality failures

    Example:
        >>> graph = TranslationGraph()
        >>> initial_state = {
        ...     "session_id": "session-123",
        ...     "current_workflow": "translation",
        ...     "source_file_path": "/path/to/patent.pdf",
        ...     "source_document": {...},
        ... }
        >>> result = await graph.run(initial_state)
    """

    def __init__(
        self,
        checkpointer: MemorySaver | None = None,
    ):
        """Initialize the translation workflow graph.

        Args:
            checkpointer: Optional checkpointer for state persistence
        """
        super().__init__(
            name="translation",
            checkpointer=checkpointer,
        )

        # Initialize agents
        self._document_intake = DocumentIntakeAgent()
        self._structure_analyzer = StructureAnalyzerAgent()
        self._glossary_builder = GlossaryBuilderAgent()
        self._section_translator = SectionTranslatorAgent()
        self._quality_verifier = QualityVerifierAgent()
        self._output_generator = OutputGeneratorAgent()

    @property
    def state_class(self) -> type[TranslationState]:
        return TranslationState

    def get_initial_step(self) -> str:
        return "P1"

    def define_nodes(self, builder: StateGraph) -> None:
        """Add workflow-specific nodes to the graph."""
        # Phase 1: Document Intake
        builder.add_node("P1", self._p1_intake_step)

        # Phase 2: Structure Analysis
        builder.add_node("P2", self._p2_structure_step)

        # Phase 3: Glossary Building
        builder.add_node("P3", self._p3_glossary_step)

        # Phase 4: Section Translation
        builder.add_node("P4", self._p4_translate_step)

        # Phase 5: Quality Verification
        builder.add_node("P5", self._p5_verify_step)

        # Phase 6: Output Generation
        builder.add_node("P6", self._p6_output_step)

        # Revision node for quality issues
        builder.add_node("revise", self._revise_step)

        # Glossary conflict resolution
        builder.add_node("resolve_glossary", self._resolve_glossary_step)

    def define_edges(self, builder: StateGraph) -> None:
        """Add workflow-specific edges to the graph."""
        settings = get_settings()

        # P1 -> P2 (or error)
        builder.add_conditional_edges(
            "P1",
            self._route_after_p1,
            {
                "P2": "P2",
                "error": "error_handler",
            },
        )

        # P2 -> P3 (or error)
        builder.add_conditional_edges(
            "P2",
            self._route_after_p2,
            {
                "P3": "P3",
                "error": "error_handler",
            },
        )

        # P3 -> P4 or glossary conflict resolution
        builder.add_conditional_edges(
            "P3",
            self._route_after_p3,
            {
                "P4": "P4",
                "resolve": "resolve_glossary",
                "human_review": "human_review",
            },
        )

        # Glossary resolution -> P3 (retry) or P4
        builder.add_conditional_edges(
            "resolve_glossary",
            self._route_after_resolve,
            {
                "P3": "P3",
                "P4": "P4",
                "human_review": "human_review",
            },
        )

        # P4 -> P5
        builder.add_edge("P4", "P5")

        # P5 -> P6 or revise (if quality fails)
        builder.add_conditional_edges(
            "P5",
            self._route_after_p5,
            {
                "P6": "P6",
                "revise": "revise",
                "human_review": "human_review",
            },
        )

        # Revise -> P5 (re-verify)
        builder.add_edge("revise", "P5")

        # P6 -> END or human review
        builder.add_conditional_edges(
            "P6",
            self._route_after_p6,
            {
                "end": END,
                "human_review": "human_review",
            },
        )

        # Human review -> continue or back to specific phase
        builder.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "continue": END,
                "P3": "P3",
                "P4": "P4",
                "end": END,
            },
        )

        # Error handler -> END
        builder.add_edge("error_handler", END)

    # ─────────────────────────────────────────────────────────────
    # Node Implementations
    # ─────────────────────────────────────────────────────────────

    async def _p1_intake_step(
        self, state: TranslationState
    ) -> dict[str, Any]:
        """Phase 1: Document intake and initial analysis."""
        self._logger.info("executing_p1_intake")

        result = await self._document_intake.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p2_structure_step(
        self, state: TranslationState
    ) -> dict[str, Any]:
        """Phase 2: Document structure analysis."""
        self._logger.info("executing_p2_structure")

        result = await self._structure_analyzer.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p3_glossary_step(
        self, state: TranslationState
    ) -> dict[str, Any]:
        """Phase 3: Glossary building."""
        self._logger.info("executing_p3_glossary")

        result = await self._glossary_builder.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p4_translate_step(
        self, state: TranslationState
    ) -> dict[str, Any]:
        """Phase 4: Section-by-section translation."""
        self._logger.info("executing_p4_translate")

        result = await self._section_translator.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p5_verify_step(
        self, state: TranslationState
    ) -> dict[str, Any]:
        """Phase 5: Quality verification."""
        self._logger.info("executing_p5_verify")

        result = await self._quality_verifier.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _p6_output_step(
        self, state: TranslationState
    ) -> dict[str, Any]:
        """Phase 6: Output generation."""
        self._logger.info("executing_p6_output")

        result = await self._output_generator.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _revise_step(
        self, state: TranslationState
    ) -> dict[str, Any]:
        """Revision step for quality failures."""
        self._logger.info("executing_revision")

        iteration = state.get("iteration_count", 0) + 1
        quality_report = state.get("quality_report", {})
        issues = quality_report.get("issues", [])

        # Find sections with law violations
        law_violations = state.get("law_violations", [])
        translated_sections = state.get("translated_sections", {})

        # Revise sections with issues
        for violation in law_violations[:5]:  # Limit to 5 revisions per iteration
            section_type = violation.get("section")
            if section_type and section_type in translated_sections:
                await self._section_translator.revise_section(
                    state,
                    section_type,
                    [violation.get("law", "법칙 위반")],
                )

        return {
            "iteration_count": iteration,
            "current_step": "P5",  # Go back to verification
            "updated_at": datetime.now().isoformat(),
        }

    async def _resolve_glossary_step(
        self, state: TranslationState
    ) -> dict[str, Any]:
        """Resolve glossary conflicts automatically or flag for review."""
        self._logger.info("resolving_glossary_conflicts")

        conflicts = state.get("glossary_conflicts", [])

        # Try to auto-resolve simple conflicts
        auto_resolved = 0
        for conflict in conflicts:
            # Use first option as default
            if len(conflict.get("options", [])) > 0:
                term = conflict["term"]
                selected = conflict["options"][0]

                result = await self._glossary_builder.resolve_conflict(
                    state, term, selected
                )

                if result.get("glossary_finalized"):
                    auto_resolved += 1

        if auto_resolved == len(conflicts):
            return {
                "glossary_finalized": True,
                "glossary_conflicts": [],
                "current_step": "P4",
                "updated_at": datetime.now().isoformat(),
            }

        # Still have unresolved conflicts
        return {
            "human_approval_required": True,
            "pending_approval_checkpoint": "glossary_conflict",
            "updated_at": datetime.now().isoformat(),
        }

    # ─────────────────────────────────────────────────────────────
    # Routing Functions
    # ─────────────────────────────────────────────────────────────

    def _route_after_p1(
        self, state: TranslationState
    ) -> Literal["P2", "error"]:
        """Route after document intake."""
        if state.get("is_error_state"):
            return "error"

        source_doc = state.get("source_document")
        if not source_doc:
            return "error"

        return "P2"

    def _route_after_p2(
        self, state: TranslationState
    ) -> Literal["P3", "error"]:
        """Route after structure analysis."""
        if state.get("is_error_state"):
            return "error"

        doc_structure = state.get("document_structure")
        if not doc_structure:
            return "error"

        # Check for critical missing sections
        missing = state.get("missing_sections", [])
        if "claims" in missing:
            self._logger.warning("claims_section_missing")
            # Could still proceed - maybe it's a different doc type

        return "P3"

    def _route_after_p3(
        self, state: TranslationState
    ) -> Literal["P4", "resolve", "human_review"]:
        """Route after glossary building."""
        settings = get_settings()

        glossary_finalized = state.get("glossary_finalized", False)
        conflicts = state.get("glossary_conflicts", [])

        if glossary_finalized:
            return "P4"

        if conflicts:
            # Try to auto-resolve first
            if len(conflicts) <= 3:  # Few conflicts - try auto-resolve
                return "resolve"

            # Many conflicts - need human review
            if settings.workflow.enable_human_in_loop:
                return "human_review"
            else:
                return "resolve"  # Auto-resolve anyway

        return "P4"

    def _route_after_resolve(
        self, state: TranslationState
    ) -> Literal["P3", "P4", "human_review"]:
        """Route after glossary conflict resolution."""
        settings = get_settings()

        if state.get("glossary_finalized"):
            return "P4"

        conflicts = state.get("glossary_conflicts", [])
        if conflicts and settings.workflow.enable_human_in_loop:
            return "human_review"

        # Retry P3 to rebuild glossary
        return "P3"

    def _route_after_p5(
        self, state: TranslationState
    ) -> Literal["P6", "revise", "human_review"]:
        """Route after quality verification."""
        settings = get_settings()

        quality_report = state.get("quality_report", {})
        overall_score = quality_report.get("overall_score", 0)
        iteration = state.get("iteration_count", 0)

        # Check for critical TAC errors
        risk_flags = state.get("risk_flags", [])
        critical_flags = [f for f in risk_flags if f.get("level") == "Critical"]

        if critical_flags and settings.workflow.enable_human_in_loop:
            return "human_review"

        # Quality check
        if overall_score >= settings.workflow.quality_threshold:
            return "P6"

        # Below threshold - try to revise
        if iteration < settings.workflow.max_iterations:
            return "revise"

        # Max iterations reached - human review or proceed anyway
        if settings.workflow.enable_human_in_loop:
            return "human_review"

        return "P6"  # Proceed with warning

    def _route_after_p6(
        self, state: TranslationState
    ) -> Literal["end", "human_review"]:
        """Route after output generation."""
        settings = get_settings()

        # Check for any unresolved critical risks
        risk_flags = state.get("risk_flags", [])
        unresolved_critical = [
            f for f in risk_flags
            if f.get("level") == "Critical" and f.get("requires_human_review")
        ]

        if unresolved_critical and settings.workflow.enable_human_in_loop:
            return "human_review"

        return "end"

    def _route_after_human_review(
        self, state: TranslationState
    ) -> Literal["continue", "P3", "P4", "end"]:
        """Route after human review."""
        human_feedback = state.get("human_feedback", [])

        if not human_feedback:
            return "continue"

        latest = human_feedback[-1] if human_feedback else {}

        if latest.get("approved", False):
            return "continue"

        # Check for specific requests
        comments = latest.get("comments", "").lower()

        if "용어" in comments or "glossary" in comments:
            return "P3"

        if "번역" in comments or "재번역" in comments or "translate" in comments:
            return "P4"

        return "end"

    def _get_next_step(self, state: TranslationState) -> str:
        """Get the next step based on current state."""
        current = state.get("current_step", "P1")

        step_order = ["P1", "P2", "P3", "P4", "P5", "P6"]
        try:
            idx = step_order.index(current)
            if idx < len(step_order) - 1:
                return step_order[idx + 1]
        except ValueError:
            pass

        return END


def create_translation_graph(
    checkpointer: MemorySaver | None = None,
) -> TranslationGraph:
    """Factory function to create a translation graph.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Configured TranslationGraph instance
    """
    return TranslationGraph(checkpointer=checkpointer)
