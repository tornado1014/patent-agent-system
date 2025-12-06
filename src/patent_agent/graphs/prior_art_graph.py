"""Prior Art Search Workflow Graph.

Implements the prior art search workflow using LangGraph.

Workflow Steps:
- query: 검색 쿼리 설계 (QueryBuilderAgent)
- search: 다중 DB 병렬 검색 (MultiDBSearcherAgent)
- filter: 관련성 필터링 (RelevanceAnalyzerAgent)
- analyze: 청구항 대비 분석 (ClaimComparerAgent)
- report: 최종 보고서 생성 (NoveltyAssessorAgent)
"""

from datetime import datetime
from typing import Any, Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from patent_agent.agents.prior_art import (
    ClaimComparerAgent,
    MultiDBSearcherAgent,
    NoveltyAssessorAgent,
    QueryBuilderAgent,
    RelevanceAnalyzerAgent,
)
from patent_agent.config import settings
from patent_agent.graphs.base_graph import BasePatentGraph
from patent_agent.state.prior_art import PriorArtSearchState

logger = structlog.get_logger(__name__)


class PriorArtSearchGraph(BasePatentGraph[PriorArtSearchState]):
    """LangGraph implementation of the prior art search workflow.

    This graph orchestrates the following agents:
    - QueryBuilderAgent (query)
    - MultiDBSearcherAgent (search)
    - RelevanceAnalyzerAgent (filter)
    - ClaimComparerAgent (analyze)
    - NoveltyAssessorAgent (report)

    Features:
    - Multi-database parallel search
    - Relevance-based filtering
    - Detailed claim comparison
    - Novelty and inventive step assessment

    Example:
        >>> graph = PriorArtSearchGraph()
        >>> initial_state = {
        ...     "session_id": "session-123",
        ...     "current_workflow": "prior_art",
        ...     "target_invention": "발명 요약...",
        ...     "target_claims": ["청구항 1...", "청구항 2..."],
        ...     "search_type": "novelty_search",
        ... }
        >>> result = await graph.run(initial_state)
    """

    def __init__(
        self,
        checkpointer: MemorySaver | None = None,
    ):
        """Initialize the prior art search workflow graph.

        Args:
            checkpointer: Optional checkpointer for state persistence
        """
        super().__init__(
            name="prior_art_search",
            checkpointer=checkpointer,
        )

        # Initialize agents
        self._query_builder = QueryBuilderAgent()
        self._multi_db_searcher = MultiDBSearcherAgent()
        self._relevance_analyzer = RelevanceAnalyzerAgent()
        self._claim_comparer = ClaimComparerAgent()
        self._novelty_assessor = NoveltyAssessorAgent()

    @property
    def state_class(self) -> type[PriorArtSearchState]:
        return PriorArtSearchState

    def get_initial_step(self) -> str:
        return "query"

    def define_nodes(self, builder: StateGraph) -> None:
        """Add workflow-specific nodes to the graph."""
        # Query building
        builder.add_node("query", self._query_step)

        # Multi-database search
        builder.add_node("search", self._search_step)

        # Relevance filtering
        builder.add_node("filter", self._filter_step)

        # Claim comparison
        builder.add_node("analyze", self._analyze_step)

        # Final report
        builder.add_node("report", self._report_step)

        # Revision node for quality issues
        builder.add_node("refine_query", self._refine_query_step)

    def define_edges(self, builder: StateGraph) -> None:
        """Add workflow-specific edges to the graph."""
        # query -> search (or error)
        builder.add_conditional_edges(
            "query",
            self._route_after_query,
            {
                "search": "search",
                "error": "error_handler",
            },
        )

        # search -> filter (or error)
        builder.add_conditional_edges(
            "search",
            self._route_after_search,
            {
                "filter": "filter",
                "error": "error_handler",
            },
        )

        # filter -> analyze (or refine if not enough results)
        builder.add_conditional_edges(
            "filter",
            self._route_after_filter,
            {
                "analyze": "analyze",
                "refine": "refine_query",
                "report": "report",
            },
        )

        # refine_query -> search
        builder.add_edge("refine_query", "search")

        # analyze -> report (or human review)
        builder.add_conditional_edges(
            "analyze",
            self._route_after_analyze,
            {
                "report": "report",
                "human_review": "human_review",
            },
        )

        # report -> end (or human review)
        builder.add_conditional_edges(
            "report",
            self._route_after_report,
            {
                "end": END,
                "human_review": "human_review",
            },
        )

        # Human review -> continue or refine
        builder.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "continue": END,
                "refine": "refine_query",
                "end": END,
            },
        )

        # Error handler -> END
        builder.add_edge("error_handler", END)

    # ─────────────────────────────────────────────────────────────
    # Node Implementations
    # ─────────────────────────────────────────────────────────────

    async def _query_step(
        self, state: PriorArtSearchState
    ) -> dict[str, Any]:
        """Query building step."""
        self._logger.info("executing_query_step")

        result = await self._query_builder.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _search_step(
        self, state: PriorArtSearchState
    ) -> dict[str, Any]:
        """Multi-database search step."""
        self._logger.info("executing_search_step")

        result = await self._multi_db_searcher.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _filter_step(
        self, state: PriorArtSearchState
    ) -> dict[str, Any]:
        """Relevance filtering step."""
        self._logger.info("executing_filter_step")

        result = await self._relevance_analyzer.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _analyze_step(
        self, state: PriorArtSearchState
    ) -> dict[str, Any]:
        """Claim comparison step."""
        self._logger.info("executing_analyze_step")

        result = await self._claim_comparer.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _report_step(
        self, state: PriorArtSearchState
    ) -> dict[str, Any]:
        """Final report generation step."""
        self._logger.info("executing_report_step")

        result = await self._novelty_assessor.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _refine_query_step(
        self, state: PriorArtSearchState
    ) -> dict[str, Any]:
        """Refine search query based on results."""
        self._logger.info("refining_query")

        iteration = state.get("iteration_count", 0) + 1

        # Adjust relevance threshold down if not enough results
        current_threshold = state.get("relevance_threshold", 0.7)
        new_threshold = max(0.3, current_threshold - 0.1)

        # Could also expand keywords here
        expanded = state.get("expanded_keywords", [])

        return {
            "iteration_count": iteration,
            "relevance_threshold": new_threshold,
            "current_step": "search",
            "updated_at": datetime.now().isoformat(),
        }

    # ─────────────────────────────────────────────────────────────
    # Routing Functions
    # ─────────────────────────────────────────────────────────────

    def _route_after_query(
        self, state: PriorArtSearchState
    ) -> Literal["search", "error"]:
        """Route after query building."""
        if state.get("is_error_state"):
            return "error"

        search_queries = state.get("search_queries", {})
        if not search_queries:
            return "error"

        return "search"

    def _route_after_search(
        self, state: PriorArtSearchState
    ) -> Literal["filter", "error"]:
        """Route after database search."""
        if state.get("is_error_state"):
            return "error"

        # Check if all databases failed
        errors = state.get("search_errors", {})
        total_count = state.get("total_results_count", {})

        all_failed = all(err is not None for err in errors.values())
        no_results = sum(total_count.values()) == 0

        if all_failed and no_results:
            return "error"

        return "filter"

    def _route_after_filter(
        self, state: PriorArtSearchState
    ) -> Literal["analyze", "refine", "report"]:
        """Route after relevance filtering."""
        analyzed_refs = state.get("analyzed_references", [])
        target_claims = state.get("target_claims", [])
        iteration = state.get("iteration_count", 0)

        # If no relevant results and under max iterations, refine query
        if not analyzed_refs:
            if iteration < settings.workflow.max_iterations:
                self._logger.info("no_relevant_results_refining")
                return "refine"
            else:
                # No results but max iterations - generate report anyway
                return "report"

        # If no claims to compare, skip to report
        if not target_claims:
            return "report"

        return "analyze"

    def _route_after_analyze(
        self, state: PriorArtSearchState
    ) -> Literal["report", "human_review"]:
        """Route after claim comparison."""
        comparisons = state.get("claim_comparisons", [])

        # Check for critical findings
        identical_found = any(
            c.get("comparison_result") == "identical"
            for c in comparisons
        )

        if identical_found and settings.workflow.enable_human_in_loop:
            return "human_review"

        return "report"

    def _route_after_report(
        self, state: PriorArtSearchState
    ) -> Literal["end", "human_review"]:
        """Route after report generation."""
        # Check overall assessments
        overall_novelty = state.get("overall_novelty", True)
        overall_inventive = state.get("overall_inventive_step", True)

        # If patentability issues found, request human review
        if not overall_novelty or not overall_inventive:
            if settings.workflow.enable_human_in_loop:
                return "human_review"

        return "end"

    def _route_after_human_review(
        self, state: PriorArtSearchState
    ) -> Literal["continue", "refine", "end"]:
        """Route after human review."""
        human_feedback = state.get("human_feedback", [])

        if not human_feedback:
            return "continue"

        latest = human_feedback[-1] if human_feedback else {}

        if latest.get("approved", False):
            return "continue"

        # Check if refinement requested
        comments = latest.get("comments", "").lower()
        if "재검색" in comments or "추가검색" in comments or "refine" in comments:
            return "refine"

        return "end"

    def _get_next_step(self, state: PriorArtSearchState) -> str:
        """Get the next step based on current state."""
        current = state.get("current_step", "query")

        step_order = ["query", "search", "filter", "analyze", "report"]
        try:
            idx = step_order.index(current)
            if idx < len(step_order) - 1:
                return step_order[idx + 1]
        except ValueError:
            pass

        return END


def create_prior_art_search_graph(
    checkpointer: MemorySaver | None = None,
) -> PriorArtSearchGraph:
    """Factory function to create a prior art search graph.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Configured PriorArtSearchGraph instance
    """
    return PriorArtSearchGraph(checkpointer=checkpointer)
