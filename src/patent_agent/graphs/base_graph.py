"""
Base graph class for all patent workflows.

Provides common patterns:
- State management
- Error handling
- Human-in-the-Loop integration
- Logging and tracing
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from patent_agent.config import settings
from patent_agent.state.base import BasePatentState, HumanFeedback, ReviewComment


logger = structlog.get_logger()

# Generic state type
StateT = TypeVar("StateT", bound=BasePatentState)


class BasePatentGraph(ABC, Generic[StateT]):
    """
    Abstract base class for patent workflow graphs.

    Subclasses must implement:
    - define_nodes(): Add workflow-specific nodes
    - define_edges(): Add workflow-specific edges
    - get_initial_step(): Return the first step name
    """

    def __init__(
        self,
        name: str,
        checkpointer: MemorySaver | None = None,
    ):
        self.name = name
        self.checkpointer = checkpointer or MemorySaver()
        self._graph: CompiledStateGraph | None = None
        self._logger = logger.bind(graph=name)

    @property
    def graph(self) -> CompiledStateGraph:
        """Lazily build and return the compiled graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> CompiledStateGraph:
        """Build the complete workflow graph."""
        builder = StateGraph(self.state_class)

        # Add common nodes
        self._add_common_nodes(builder)

        # Add workflow-specific nodes
        self.define_nodes(builder)

        # Set entry point
        builder.set_entry_point(self.get_initial_step())

        # Add workflow-specific edges
        self.define_edges(builder)

        # Compile with checkpointer
        return builder.compile(checkpointer=self.checkpointer)

    def _add_common_nodes(self, builder: StateGraph) -> None:
        """Add nodes common to all workflows."""
        builder.add_node("error_handler", self._error_handler_node)
        builder.add_node("human_review", self._human_review_node)

    # ─────────────────────────────────────────────────────────────
    # Abstract Methods (must be implemented by subclasses)
    # ─────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def state_class(self) -> type[StateT]:
        """Return the state class for this workflow."""
        ...

    @abstractmethod
    def define_nodes(self, builder: StateGraph) -> None:
        """Add workflow-specific nodes to the graph."""
        ...

    @abstractmethod
    def define_edges(self, builder: StateGraph) -> None:
        """Add workflow-specific edges to the graph."""
        ...

    @abstractmethod
    def get_initial_step(self) -> str:
        """Return the name of the initial step."""
        ...

    # ─────────────────────────────────────────────────────────────
    # Common Node Implementations
    # ─────────────────────────────────────────────────────────────

    def _error_handler_node(self, state: StateT) -> dict[str, Any]:
        """Handle errors and potentially recover."""
        self._logger.error(
            "workflow_error",
            errors=state.get("error_messages", []),
            step=state.get("current_step"),
        )
        return {
            "is_error_state": True,
            "updated_at": datetime.now().isoformat(),
        }

    def _human_review_node(self, state: StateT) -> dict[str, Any]:
        """
        Human-in-the-Loop checkpoint using CopilotKit interrupt pattern.

        This node:
        1. Pauses execution using LangGraph interrupt()
        2. Sends interrupt event to frontend via CopilotKit
        3. Waits for user approval/feedback
        4. Resumes with updated state based on user response
        """
        checkpoint = state.get("pending_approval_checkpoint", "unknown")
        self._logger.info(
            "human_review_requested",
            checkpoint=checkpoint,
            iteration=state.get("iteration_count", 0),
        )

        # Auto-approve if human_in_loop is disabled
        if not settings.workflow.enable_human_in_loop:
            return {
                "human_approval_required": False,
                "human_feedback": state.get("human_feedback", [])
                + [
                    HumanFeedback(
                        checkpoint=checkpoint,
                        approved=True,
                        comments="Auto-approved (human_in_loop disabled)",
                        timestamp=datetime.now().isoformat(),
                        reviewer=None,
                    )
                ],
                "pending_approval_checkpoint": None,
                "updated_at": datetime.now().isoformat(),
            }

        # Build interrupt event based on checkpoint type
        interrupt_event = self._build_interrupt_event(state, checkpoint)

        # Use LangGraph interrupt() to pause and wait for user response
        # This triggers useLangGraphInterrupt hook in React frontend
        human_response = interrupt(interrupt_event)

        # Process the response from frontend
        return self._process_human_response(state, checkpoint, human_response)

    def _build_interrupt_event(
        self, state: StateT, checkpoint: str
    ) -> dict[str, Any]:
        """
        Build interrupt event payload for CopilotKit frontend.

        Override in subclasses for workflow-specific interrupt events.
        """
        quality_score = state.get("quality_score", {})

        # Default interrupt event
        return {
            "type": "approval_request",
            "checkpoint": checkpoint,
            "message": f"체크포인트 {checkpoint}에서 검토가 필요합니다.",
            "quality_score": quality_score,
            "iteration_count": state.get("iteration_count", 0),
            "workflow": state.get("current_workflow"),
            "current_step": state.get("current_step"),
        }

    def _process_human_response(
        self,
        state: StateT,
        checkpoint: str,
        human_response: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        """
        Process the response from human review.

        Args:
            state: Current workflow state
            checkpoint: The checkpoint identifier
            human_response: Response from frontend (dict, string, or None)
        """
        # Handle different response formats
        if human_response is None:
            # No response yet, keep waiting
            return {}

        if isinstance(human_response, str):
            # String response (e.g., "APPROVED" or "REJECTED")
            approved = human_response.upper() in ("APPROVED", "SEND", "YES", "OK")
            comments = human_response if not approved else ""
            response_data = {"approved": approved, "comments": comments}
        else:
            # Dict response with structured data
            response_data = human_response

        approved = response_data.get("approved", False)
        comments = response_data.get("comments", "")
        reviewer = response_data.get("reviewer")

        self._logger.info(
            "human_review_completed",
            checkpoint=checkpoint,
            approved=approved,
            has_comments=bool(comments),
        )

        result = {
            "human_approval_required": False,
            "human_feedback": state.get("human_feedback", [])
            + [
                HumanFeedback(
                    checkpoint=checkpoint,
                    approved=approved,
                    comments=comments,
                    timestamp=datetime.now().isoformat(),
                    reviewer=reviewer,
                )
            ],
            "pending_approval_checkpoint": None,
            "updated_at": datetime.now().isoformat(),
        }

        # If rejected, add revision comments
        if not approved and response_data.get("revision_comments"):
            result["review_comments"] = response_data["revision_comments"]

        return result

    # ─────────────────────────────────────────────────────────────
    # Common Routing Functions
    # ─────────────────────────────────────────────────────────────

    def should_continue_reflexion(self, state: StateT) -> str:
        """
        Determine if Reflexion loop should continue.

        Returns:
        - "revise": Continue with revision
        - "approve": Move to human approval
        - "complete": Exit loop successfully
        - "error": Exit with error
        """
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", settings.workflow.max_iterations)
        review_comments = state.get("review_comments", [])
        quality_score = state.get("quality_score", {}).get("overall", 0)

        # Check for errors
        if state.get("is_error_state"):
            return "error"

        # Max iterations reached
        if iteration >= max_iter:
            self._logger.warning(
                "max_iterations_reached",
                iteration=iteration,
                quality_score=quality_score,
            )
            return "approve"

        # No review comments = quality passed
        if not review_comments:
            return "complete"

        # Check if only low-severity issues remain
        critical_issues = [
            c for c in review_comments if c.get("severity") in ("critical", "high")
        ]
        if not critical_issues and quality_score >= settings.workflow.quality_threshold:
            return "complete"

        return "revise"

    def should_request_human_review(self, state: StateT) -> str:
        """
        Check if human review is required at current checkpoint.

        Returns node name: "human_review" or next step name.
        """
        if state.get("human_approval_required"):
            return "human_review"
        return self._get_next_step(state)

    @abstractmethod
    def _get_next_step(self, state: StateT) -> str:
        """Get the next step based on current state."""
        ...

    # ─────────────────────────────────────────────────────────────
    # Execution Methods
    # ─────────────────────────────────────────────────────────────

    async def run(
        self,
        initial_state: StateT,
        config: dict[str, Any] | None = None,
    ) -> StateT:
        """
        Execute the workflow graph.

        Args:
            initial_state: Initial state to start with
            config: Optional LangGraph config (thread_id, etc.)

        Returns:
            Final state after execution
        """
        config = config or {"configurable": {"thread_id": initial_state.get("session_id")}}

        self._logger.info(
            "workflow_started",
            session_id=initial_state.get("session_id"),
            workflow=initial_state.get("current_workflow"),
        )

        try:
            result = await self.graph.ainvoke(initial_state, config)
            self._logger.info(
                "workflow_completed",
                session_id=result.get("session_id"),
                quality_score=result.get("quality_score", {}).get("overall"),
            )
            return result
        except Exception as e:
            self._logger.exception("workflow_failed", error=str(e))
            raise

    async def stream(
        self,
        initial_state: StateT,
        config: dict[str, Any] | None = None,
    ):
        """
        Stream workflow execution for real-time updates.

        Yields state updates as the graph executes.
        """
        config = config or {"configurable": {"thread_id": initial_state.get("session_id")}}

        async for event in self.graph.astream(initial_state, config):
            yield event

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """Get the current state for a thread."""
        return self.graph.get_state(config)

    def update_state(
        self,
        config: dict[str, Any],
        updates: dict[str, Any],
    ) -> None:
        """Update state for a paused graph (e.g., after human review)."""
        self.graph.update_state(config, updates)


def create_quality_check_node(check_func):
    """
    Factory to create a quality check node.

    Args:
        check_func: Function that takes state and returns list of ReviewComment
    """

    def quality_check(state: BasePatentState) -> dict[str, Any]:
        comments = check_func(state)
        overall_score = 100 - (len(comments) * 10)  # Simple scoring
        overall_score = max(0, min(100, overall_score))

        return {
            "review_comments": comments,
            "quality_score": {
                "overall": overall_score,
                "completeness": overall_score,
                "accuracy": overall_score,
                "compliance": overall_score,
                "comments": [c["issue"] for c in comments],
            },
            "updated_at": datetime.now().isoformat(),
        }

    return quality_check
