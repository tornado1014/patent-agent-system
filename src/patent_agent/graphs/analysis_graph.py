"""Patent Analysis Workflow Graph.

Implements the 5-type analysis workflow using LangGraph.

Supported Analysis Types:
- portfolio: 특정 기업/기관의 특허 현황
- trend: 특정 기술분야의 출원/등록 트렌드
- competitor: 경쟁사 특허 전략 파악
- infringement: 제품 vs 특허 청구항 대비
- invalidity: 특허 무효화 가능성 검토

Workflow Steps:
- collect: 특허 데이터 수집
- process: 데이터 전처리 (청구항 매핑 등)
- analyze: 분석 수행
- visualize: 시각화 생성
- report: 보고서 작성
"""

from datetime import datetime
from typing import Any, Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from patent_agent.agents.analysis import (
    ClaimMapperAgent,
    CompetitorProfilerAgent,
    DataCollectorAgent,
    InfringementCheckerAgent,
    ReportWriterAgent,
    TrendAnalyzerAgent,
)
from patent_agent.config import get_settings
from patent_agent.graphs.base_graph import BasePatentGraph
from patent_agent.state.analysis import AnalysisType, PatentAnalysisState

logger = structlog.get_logger(__name__)


class PatentAnalysisGraph(BasePatentGraph[PatentAnalysisState]):
    """LangGraph implementation of the patent analysis workflow.

    This graph orchestrates the following agents based on analysis type:
    - DataCollectorAgent (all types)
    - ClaimMapperAgent (infringement, invalidity)
    - TrendAnalyzerAgent (trend, portfolio)
    - CompetitorProfilerAgent (competitor, portfolio)
    - InfringementCheckerAgent (infringement)
    - ReportWriterAgent (all types)

    Features:
    - Multi-type analysis support
    - Dynamic agent routing based on analysis type
    - Human-in-the-loop for critical findings
    - Comprehensive report generation

    Example:
        >>> graph = PatentAnalysisGraph()
        >>> initial_state = {
        ...     "session_id": "session-123",
        ...     "current_workflow": "analysis",
        ...     "analysis_type": "portfolio",
        ...     "target": "삼성전자",
        ...     "scope": {"jurisdictions": ["KR", "US"]},
        ... }
        >>> result = await graph.run(initial_state)
    """

    def __init__(
        self,
        checkpointer: MemorySaver | None = None,
    ):
        """Initialize the analysis workflow graph.

        Args:
            checkpointer: Optional checkpointer for state persistence
        """
        super().__init__(
            name="analysis",
            checkpointer=checkpointer,
        )

        # Initialize agents
        self._data_collector = DataCollectorAgent()
        self._claim_mapper = ClaimMapperAgent()
        self._trend_analyzer = TrendAnalyzerAgent()
        self._competitor_profiler = CompetitorProfilerAgent()
        self._infringement_checker = InfringementCheckerAgent()
        self._report_writer = ReportWriterAgent()

    @property
    def state_class(self) -> type[PatentAnalysisState]:
        return PatentAnalysisState

    def get_initial_step(self) -> str:
        return "collect"

    def define_nodes(self, builder: StateGraph) -> None:
        """Add workflow-specific nodes to the graph."""
        # Step 1: Data Collection (all types)
        builder.add_node("collect", self._collect_step)

        # Step 2: Data Processing (type-dependent)
        builder.add_node("process", self._process_step)

        # Step 3: Analysis (type-dependent)
        builder.add_node("analyze_trend", self._analyze_trend_step)
        builder.add_node("analyze_competitor", self._analyze_competitor_step)
        builder.add_node("analyze_infringement", self._analyze_infringement_step)

        # Step 4: Visualization
        builder.add_node("visualize", self._visualize_step)

        # Step 5: Report Generation
        builder.add_node("report", self._report_step)

    def define_edges(self, builder: StateGraph) -> None:
        """Add workflow-specific edges to the graph."""
        # collect -> process (or error)
        builder.add_conditional_edges(
            "collect",
            self._route_after_collect,
            {
                "process": "process",
                "error": "error_handler",
            },
        )

        # process -> analyze (type-dependent)
        builder.add_conditional_edges(
            "process",
            self._route_after_process,
            {
                "trend": "analyze_trend",
                "competitor": "analyze_competitor",
                "infringement": "analyze_infringement",
                "error": "error_handler",
            },
        )

        # analyze_* -> visualize
        builder.add_edge("analyze_trend", "visualize")
        builder.add_edge("analyze_competitor", "visualize")

        # infringement -> report (or human review for high risk)
        builder.add_conditional_edges(
            "analyze_infringement",
            self._route_after_infringement,
            {
                "report": "report",
                "human_review": "human_review",
            },
        )

        # visualize -> report
        builder.add_edge("visualize", "report")

        # report -> END or human review
        builder.add_conditional_edges(
            "report",
            self._route_after_report,
            {
                "end": END,
                "human_review": "human_review",
            },
        )

        # Human review -> END
        builder.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "continue": END,
                "revise": "analyze_trend",  # Could route back to appropriate analyze step
                "end": END,
            },
        )

        # Error handler -> END
        builder.add_edge("error_handler", END)

    # ─────────────────────────────────────────────────────────────
    # Node Implementations
    # ─────────────────────────────────────────────────────────────

    async def _collect_step(
        self, state: PatentAnalysisState
    ) -> dict[str, Any]:
        """Step 1: Collect patent data from multiple databases."""
        self._logger.info("executing_collect_step")

        result = await self._data_collector.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _process_step(
        self, state: PatentAnalysisState
    ) -> dict[str, Any]:
        """Step 2: Process data (claim mapping for infringement/invalidity)."""
        self._logger.info("executing_process_step")

        analysis_type = state.get("analysis_type", "portfolio")

        # Claim mapping is only needed for infringement/invalidity
        if analysis_type in ("infringement", "invalidity"):
            result = await self._claim_mapper.process(state)
        else:
            result = {
                "claim_mappings": {},
                "current_step": "analyze",
            }

        result["updated_at"] = datetime.now().isoformat()
        return result

    async def _analyze_trend_step(
        self, state: PatentAnalysisState
    ) -> dict[str, Any]:
        """Step 3a: Analyze trends (for trend and portfolio types)."""
        self._logger.info("executing_analyze_trend_step")

        result = await self._trend_analyzer.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _analyze_competitor_step(
        self, state: PatentAnalysisState
    ) -> dict[str, Any]:
        """Step 3b: Analyze competitors."""
        self._logger.info("executing_analyze_competitor_step")

        result = await self._competitor_profiler.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _analyze_infringement_step(
        self, state: PatentAnalysisState
    ) -> dict[str, Any]:
        """Step 3c: Analyze infringement (for infringement/invalidity types)."""
        self._logger.info("executing_analyze_infringement_step")

        result = await self._infringement_checker.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    async def _visualize_step(
        self, state: PatentAnalysisState
    ) -> dict[str, Any]:
        """Step 4: Generate visualizations."""
        self._logger.info("executing_visualize_step")

        # Visualizations are generated during analysis steps
        # This step can add additional visualizations or format existing ones
        visualizations = state.get("visualizations", [])

        return {
            "visualizations": visualizations,
            "current_step": "report",
            "updated_at": datetime.now().isoformat(),
        }

    async def _report_step(
        self, state: PatentAnalysisState
    ) -> dict[str, Any]:
        """Step 5: Generate analysis report."""
        self._logger.info("executing_report_step")

        result = await self._report_writer.process(state)
        result["updated_at"] = datetime.now().isoformat()

        return result

    # ─────────────────────────────────────────────────────────────
    # Routing Functions
    # ─────────────────────────────────────────────────────────────

    def _route_after_collect(
        self, state: PatentAnalysisState
    ) -> Literal["process", "error"]:
        """Route after data collection."""
        if state.get("is_error_state"):
            return "error"

        patent_data = state.get("patent_data", [])
        if not patent_data:
            self._logger.warning("no_patent_data_collected")
            return "error"

        return "process"

    def _route_after_process(
        self, state: PatentAnalysisState
    ) -> Literal["trend", "competitor", "infringement", "error"]:
        """Route to appropriate analysis step based on type."""
        if state.get("is_error_state"):
            return "error"

        analysis_type = state.get("analysis_type", "portfolio")

        # Map analysis type to analysis step
        type_to_step = {
            "portfolio": "trend",  # Portfolio uses trend analysis
            "trend": "trend",
            "competitor": "competitor",
            "infringement": "infringement",
            "invalidity": "infringement",  # Invalidity uses similar flow
        }

        return type_to_step.get(analysis_type, "trend")

    def _route_after_infringement(
        self, state: PatentAnalysisState
    ) -> Literal["report", "human_review"]:
        """Route after infringement analysis."""
        settings = get_settings()

        infringement = state.get("infringement_analysis", {})
        risk_level = infringement.get("overall_risk", "none")

        # High risk requires human review
        if risk_level == "high" and settings.workflow.enable_human_in_loop:
            return "human_review"

        return "report"

    def _route_after_report(
        self, state: PatentAnalysisState
    ) -> Literal["end", "human_review"]:
        """Route after report generation."""
        settings = get_settings()

        # Check quality score
        quality_score = state.get("quality_score", {})
        overall = quality_score.get("overall", 100)

        if overall < settings.workflow.quality_threshold:
            if settings.workflow.enable_human_in_loop:
                return "human_review"

        return "end"

    def _route_after_human_review(
        self, state: PatentAnalysisState
    ) -> Literal["continue", "revise", "end"]:
        """Route after human review."""
        human_feedback = state.get("human_feedback", [])

        if not human_feedback:
            return "continue"

        latest = human_feedback[-1] if human_feedback else {}

        if latest.get("approved", False):
            return "continue"

        # Check for revision requests
        comments = latest.get("comments", "").lower()

        if any(word in comments for word in ["재분석", "다시", "revise", "redo"]):
            return "revise"

        return "end"

    def _get_next_step(self, state: PatentAnalysisState) -> str:
        """Get the next step based on current state."""
        current = state.get("current_step", "collect")

        step_order = ["collect", "process", "analyze", "visualize", "report"]
        try:
            idx = step_order.index(current)
            if idx < len(step_order) - 1:
                return step_order[idx + 1]
        except ValueError:
            pass

        return END


def create_analysis_graph(
    checkpointer: MemorySaver | None = None,
) -> PatentAnalysisGraph:
    """Factory function to create an analysis graph.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Configured PatentAnalysisGraph instance
    """
    return PatentAnalysisGraph(checkpointer=checkpointer)
