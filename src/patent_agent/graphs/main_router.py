"""
Main Router for dispatching tasks to appropriate workflow graphs.

Routes user requests to one of 5 domain workflows:
- spec_writing: 명세서 작성
- oa_response: OA 대응
- prior_art: 선행기술조사
- translation: 특허번역
- analysis: 특허분석
"""

from datetime import datetime
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from patent_agent.config import settings
from patent_agent.state.base import BasePatentState, WorkflowType, create_initial_state


logger = structlog.get_logger()


# Workflow detection keywords (Korean + English)
WORKFLOW_KEYWORDS = {
    "spec_writing": [
        "명세서",
        "청구항",
        "발명",
        "특허 작성",
        "특허출원",
        "specification",
        "claim",
        "draft patent",
        "write patent",
        "발명신고서",
    ],
    "oa_response": [
        "OA",
        "의견제출",
        "거절이유",
        "보정",
        "office action",
        "rejection",
        "response",
        "amendment",
        "심사관",
        "인용문헌",
    ],
    "prior_art": [
        "선행기술",
        "신규성",
        "진보성",
        "검색",
        "prior art",
        "novelty",
        "inventive step",
        "search",
        "KIPRIS",
        "USPTO",
        "FTO",
    ],
    "translation": [
        "번역",
        "영한",
        "한영",
        "translate",
        "translation",
        "영문",
        "한국어",
        "KIPO",
    ],
    "analysis": [
        "분석",
        "포트폴리오",
        "경쟁사",
        "침해",
        "무효",
        "analysis",
        "portfolio",
        "competitor",
        "infringement",
        "invalidity",
        "트렌드",
        "landscape",
    ],
}


class MainRouter:
    """
    Main task router for the patent agent system.

    Analyzes user input and routes to the appropriate workflow.
    """

    def __init__(self, llm=None):
        """
        Initialize the router.

        Args:
            llm: Optional LLM for advanced intent classification
        """
        self.llm = llm
        self._logger = logger.bind(component="main_router")

    def detect_workflow(self, user_input: str) -> WorkflowType:
        """
        Detect the appropriate workflow from user input.

        Uses keyword matching first, falls back to LLM if available.
        """
        user_input_lower = user_input.lower()

        # Score each workflow by keyword matches
        scores: dict[WorkflowType, int] = {
            "spec_writing": 0,
            "oa_response": 0,
            "prior_art": 0,
            "translation": 0,
            "analysis": 0,
        }

        for workflow, keywords in WORKFLOW_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in user_input_lower:
                    scores[workflow] += 1

        # Get workflow with highest score
        max_score = max(scores.values())

        if max_score > 0:
            for workflow, score in scores.items():
                if score == max_score:
                    self._logger.info(
                        "workflow_detected",
                        workflow=workflow,
                        score=score,
                        method="keyword",
                    )
                    return workflow

        # Default to analysis if no clear match
        self._logger.warning("workflow_unclear", defaulting_to="analysis")
        return "analysis"

    async def detect_workflow_with_llm(self, user_input: str) -> WorkflowType:
        """
        Use LLM for more accurate workflow detection.

        Falls back to keyword-based detection if LLM unavailable.
        """
        if not self.llm:
            return self.detect_workflow(user_input)

        system_prompt = """You are a patent workflow classifier. Given user input, classify it into one of these workflows:

1. spec_writing - 특허명세서 작성 (Patent specification writing)
   - Writing patent applications, claims, descriptions
   - From invention disclosures or technical documents

2. oa_response - OA 대응 (Office Action response)
   - Responding to patent office rejections
   - Analyzing rejection reasons, preparing amendments

3. prior_art - 선행기술조사 (Prior art search)
   - Searching for existing patents and publications
   - Novelty/inventive step assessment

4. translation - 특허번역 (Patent translation)
   - Translating patents (usually English to Korean)
   - Technical term standardization

5. analysis - 특허분석 (Patent analysis)
   - Portfolio analysis, competitor analysis
   - Infringement analysis, invalidity analysis
   - Trend analysis

Respond with ONLY the workflow key (spec_writing, oa_response, prior_art, translation, or analysis)."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            detected = response.content.strip().lower()

            if detected in WORKFLOW_KEYWORDS:
                self._logger.info(
                    "workflow_detected",
                    workflow=detected,
                    method="llm",
                )
                return detected
        except Exception as e:
            self._logger.warning("llm_detection_failed", error=str(e))

        # Fallback to keyword detection
        return self.detect_workflow(user_input)


def create_router_node(router: MainRouter):
    """Create a router node for the main graph."""

    def route_task(state: BasePatentState) -> dict[str, Any]:
        """Route incoming task to appropriate workflow."""
        messages = state.get("messages", [])
        if not messages:
            return {"error_messages": ["No input message provided"]}

        # Get the latest user message
        last_message = messages[-1]
        user_input = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        # Detect workflow
        workflow = router.detect_workflow(user_input)

        return {
            "current_workflow": workflow,
            "updated_at": datetime.now().isoformat(),
        }

    return route_task


def create_workflow_dispatcher():
    """Create a dispatcher that routes to workflow subgraphs."""

    def dispatch(state: BasePatentState) -> str:
        """Return the name of the workflow subgraph to execute."""
        workflow = state.get("current_workflow")

        if not workflow:
            return "error_handler"

        # Map to subgraph names
        workflow_map = {
            "spec_writing": "spec_writing_graph",
            "oa_response": "oa_response_graph",
            "prior_art": "prior_art_graph",
            "translation": "translation_graph",
            "analysis": "analysis_graph",
        }

        return workflow_map.get(workflow, "error_handler")

    return dispatch


def create_main_graph(
    router: MainRouter | None = None,
    checkpointer: MemorySaver | None = None,
) -> CompiledStateGraph:
    """
    Create the main orchestration graph.

    This graph:
    1. Receives user input
    2. Routes to appropriate workflow
    3. Executes the workflow
    4. Returns results

    Args:
        router: MainRouter instance (created if not provided)
        checkpointer: LangGraph checkpointer for state persistence

    Returns:
        Compiled StateGraph
    """
    router = router or MainRouter()
    checkpointer = checkpointer or MemorySaver()

    builder = StateGraph(BasePatentState)

    # ─────────────────────────────────────────────────────────────
    # Nodes
    # ─────────────────────────────────────────────────────────────

    # Router node
    builder.add_node("router", create_router_node(router))

    # Placeholder workflow nodes (to be replaced with actual subgraphs)
    def create_placeholder_workflow(name: str):
        def placeholder(state: BasePatentState) -> dict[str, Any]:
            logger.info(f"Executing {name} workflow (placeholder)")
            return {
                "current_step": "completed",
                "updated_at": datetime.now().isoformat(),
            }

        return placeholder

    builder.add_node("spec_writing_graph", create_placeholder_workflow("spec_writing"))
    builder.add_node("oa_response_graph", create_placeholder_workflow("oa_response"))
    builder.add_node("prior_art_graph", create_placeholder_workflow("prior_art"))
    builder.add_node("translation_graph", create_placeholder_workflow("translation"))
    builder.add_node("analysis_graph", create_placeholder_workflow("analysis"))

    # Error handler
    def error_handler(state: BasePatentState) -> dict[str, Any]:
        return {
            "is_error_state": True,
            "error_messages": state.get("error_messages", [])
            + ["Workflow routing failed"],
        }

    builder.add_node("error_handler", error_handler)

    # ─────────────────────────────────────────────────────────────
    # Edges
    # ─────────────────────────────────────────────────────────────

    # Entry point
    builder.set_entry_point("router")

    # Conditional routing based on detected workflow
    builder.add_conditional_edges(
        "router",
        create_workflow_dispatcher(),
        {
            "spec_writing_graph": "spec_writing_graph",
            "oa_response_graph": "oa_response_graph",
            "prior_art_graph": "prior_art_graph",
            "translation_graph": "translation_graph",
            "analysis_graph": "analysis_graph",
            "error_handler": "error_handler",
        },
    )

    # All workflows end at END
    builder.add_edge("spec_writing_graph", END)
    builder.add_edge("oa_response_graph", END)
    builder.add_edge("prior_art_graph", END)
    builder.add_edge("translation_graph", END)
    builder.add_edge("analysis_graph", END)
    builder.add_edge("error_handler", END)

    return builder.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════


async def run_patent_task(
    user_input: str,
    session_id: str | None = None,
    llm=None,
) -> BasePatentState:
    """
    Convenience function to run a patent task.

    Args:
        user_input: User's task description
        session_id: Optional session ID for state persistence
        llm: Optional LLM for advanced routing

    Returns:
        Final state after workflow execution
    """
    router = MainRouter(llm=llm)
    graph = create_main_graph(router)

    # Create initial state
    workflow = router.detect_workflow(user_input)
    initial_state = create_initial_state(workflow, session_id)
    initial_state["messages"] = [HumanMessage(content=user_input)]

    # Execute
    config = {"configurable": {"thread_id": initial_state["session_id"]}}
    result = await graph.ainvoke(initial_state, config)

    return result
