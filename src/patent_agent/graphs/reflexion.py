"""
Reflexion Loop pattern for quality assurance.

Implements the Draft → Critique → Revise cycle used across all workflows.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import structlog
from langgraph.graph import END, StateGraph

from patent_agent.config import settings
from patent_agent.state.base import BasePatentState, ReviewComment


logger = structlog.get_logger()


@dataclass
class ReflexionConfig:
    """Configuration for Reflexion loop."""

    max_iterations: int = 3
    quality_threshold: int = 80
    require_human_approval: bool = True
    human_checkpoint_name: str = "human_review"


def create_reflexion_subgraph(
    draft_node: Callable[[BasePatentState], dict[str, Any]],
    critique_node: Callable[[BasePatentState], dict[str, Any]],
    revise_node: Callable[[BasePatentState], dict[str, Any]],
    config: ReflexionConfig | None = None,
    state_class: type[BasePatentState] = BasePatentState,
) -> StateGraph:
    """
    Create a Reflexion loop subgraph.

    The loop follows this pattern:
    Draft → Critique → [Revise → Critique]* → Exit

    Exit conditions:
    1. iteration >= max_iterations
    2. No critical/high review comments
    3. quality_score >= quality_threshold

    Args:
        draft_node: Function that generates initial draft
        critique_node: Function that critiques the draft
        revise_node: Function that revises based on critique
        config: Reflexion configuration
        state_class: State class to use

    Returns:
        Compiled StateGraph for the Reflexion loop
    """
    config = config or ReflexionConfig()
    _logger = logger.bind(component="reflexion")

    # Wrapper nodes with iteration tracking
    def draft_with_tracking(state: BasePatentState) -> dict[str, Any]:
        _logger.info("reflexion_draft", iteration=state.get("iteration_count", 0))
        result = draft_node(state)
        return {
            **result,
            "iteration_count": 0,  # Reset on new draft
            "updated_at": datetime.now().isoformat(),
        }

    def critique_with_tracking(state: BasePatentState) -> dict[str, Any]:
        _logger.info("reflexion_critique", iteration=state.get("iteration_count", 0))
        result = critique_node(state)

        # Log critique results
        comments = result.get("review_comments", [])
        _logger.info(
            "critique_complete",
            num_comments=len(comments),
            severity_counts={
                severity: len([c for c in comments if c.get("severity") == severity])
                for severity in ["critical", "high", "medium", "low"]
            },
        )

        return {
            **result,
            "updated_at": datetime.now().isoformat(),
        }

    def revise_with_tracking(state: BasePatentState) -> dict[str, Any]:
        iteration = state.get("iteration_count", 0) + 1
        _logger.info("reflexion_revise", iteration=iteration)

        result = revise_node(state)
        return {
            **result,
            "iteration_count": iteration,
            "review_comments": [],  # Clear comments after revision
            "updated_at": datetime.now().isoformat(),
        }

    def should_continue(state: BasePatentState) -> str:
        """Routing function for Reflexion loop."""
        iteration = state.get("iteration_count", 0)
        review_comments = state.get("review_comments", [])
        quality_score = state.get("quality_score", {}).get("overall", 0)

        # Check error state
        if state.get("is_error_state"):
            _logger.warning("reflexion_error_exit")
            return "error_exit"

        # Max iterations reached
        if iteration >= config.max_iterations:
            _logger.info(
                "reflexion_max_iterations",
                iteration=iteration,
                quality_score=quality_score,
            )
            return "exit_approve" if config.require_human_approval else "exit_complete"

        # No review comments or quality passed
        if not review_comments:
            _logger.info("reflexion_no_comments", quality_score=quality_score)
            return "exit_complete"

        # Check severity of remaining issues
        critical_high = [
            c for c in review_comments if c.get("severity") in ("critical", "high")
        ]

        if not critical_high and quality_score >= config.quality_threshold:
            _logger.info(
                "reflexion_quality_passed",
                quality_score=quality_score,
                remaining_issues=len(review_comments),
            )
            return "exit_complete"

        _logger.info(
            "reflexion_continue",
            iteration=iteration,
            critical_high_issues=len(critical_high),
        )
        return "revise"

    # Build the graph
    builder = StateGraph(state_class)

    # Add nodes
    builder.add_node("draft", draft_with_tracking)
    builder.add_node("critique", critique_with_tracking)
    builder.add_node("revise", revise_with_tracking)

    # Set entry point
    builder.set_entry_point("draft")

    # Add edges
    builder.add_edge("draft", "critique")

    # Conditional routing after critique
    builder.add_conditional_edges(
        "critique",
        should_continue,
        {
            "revise": "revise",
            "exit_complete": END,
            "exit_approve": END,
            "error_exit": END,
        },
    )

    # Revise loops back to critique
    builder.add_edge("revise", "critique")

    return builder.compile()


class ReflexionLoop:
    """
    Reusable Reflexion Loop component.

    Can be embedded within larger workflow graphs.
    """

    def __init__(
        self,
        name: str,
        draft_fn: Callable,
        critique_fn: Callable,
        revise_fn: Callable,
        config: ReflexionConfig | None = None,
    ):
        self.name = name
        self.draft_fn = draft_fn
        self.critique_fn = critique_fn
        self.revise_fn = revise_fn
        self.config = config or ReflexionConfig()
        self._logger = logger.bind(reflexion_loop=name)

    def add_to_graph(
        self,
        builder: StateGraph,
        entry_node: str,
        exit_node: str,
    ) -> None:
        """
        Add Reflexion loop nodes and edges to an existing graph.

        Args:
            builder: StateGraph builder to add to
            entry_node: Node that leads into the Reflexion loop
            exit_node: Node to transition to after loop completes
        """
        prefix = f"{self.name}_"

        # Add nodes with prefixed names
        builder.add_node(f"{prefix}draft", self._wrap_draft())
        builder.add_node(f"{prefix}critique", self._wrap_critique())
        builder.add_node(f"{prefix}revise", self._wrap_revise())

        # Entry edge
        builder.add_edge(entry_node, f"{prefix}draft")

        # Internal edges
        builder.add_edge(f"{prefix}draft", f"{prefix}critique")
        builder.add_edge(f"{prefix}revise", f"{prefix}critique")

        # Conditional exit
        builder.add_conditional_edges(
            f"{prefix}critique",
            self._should_continue,
            {
                "revise": f"{prefix}revise",
                "exit": exit_node,
            },
        )

    def _wrap_draft(self):
        """Wrap draft function with tracking."""

        def wrapped(state: BasePatentState) -> dict[str, Any]:
            self._logger.info("draft_start")
            result = self.draft_fn(state)
            return {
                **result,
                f"{self.name}_iteration": 0,
                "updated_at": datetime.now().isoformat(),
            }

        return wrapped

    def _wrap_critique(self):
        """Wrap critique function with tracking."""

        def wrapped(state: BasePatentState) -> dict[str, Any]:
            self._logger.info("critique_start")
            return {
                **self.critique_fn(state),
                "updated_at": datetime.now().isoformat(),
            }

        return wrapped

    def _wrap_revise(self):
        """Wrap revise function with tracking."""

        def wrapped(state: BasePatentState) -> dict[str, Any]:
            iteration_key = f"{self.name}_iteration"
            iteration = state.get(iteration_key, 0) + 1
            self._logger.info("revise_start", iteration=iteration)
            result = self.revise_fn(state)
            return {
                **result,
                iteration_key: iteration,
                "review_comments": [],
                "updated_at": datetime.now().isoformat(),
            }

        return wrapped

    def _should_continue(self, state: BasePatentState) -> str:
        """Determine if loop should continue."""
        iteration_key = f"{self.name}_iteration"
        iteration = state.get(iteration_key, 0)
        review_comments = state.get("review_comments", [])
        quality_score = state.get("quality_score", {}).get("overall", 0)

        if iteration >= self.config.max_iterations:
            return "exit"

        if not review_comments:
            return "exit"

        critical_high = [
            c for c in review_comments if c.get("severity") in ("critical", "high")
        ]
        if not critical_high and quality_score >= self.config.quality_threshold:
            return "exit"

        return "revise"


# ═══════════════════════════════════════════════════════════════
# Example Critique Functions
# ═══════════════════════════════════════════════════════════════


def create_llm_critique(
    llm,
    critique_prompt: str,
    parse_fn: Callable[[str], list[ReviewComment]],
):
    """
    Factory to create an LLM-based critique function.

    Args:
        llm: LangChain LLM instance
        critique_prompt: Prompt template for critique
        parse_fn: Function to parse LLM output into ReviewComments
    """

    async def critique(state: BasePatentState) -> dict[str, Any]:
        # Format prompt with state
        prompt = critique_prompt.format(**state)

        # Get LLM critique
        response = await llm.ainvoke(prompt)

        # Parse into structured comments
        comments = parse_fn(response.content)

        # Calculate quality score
        severity_weights = {"critical": 30, "high": 20, "medium": 10, "low": 5}
        penalty = sum(severity_weights.get(c["severity"], 0) for c in comments)
        quality_score = max(0, 100 - penalty)

        return {
            "review_comments": comments,
            "quality_score": {
                "overall": quality_score,
                "completeness": quality_score,
                "accuracy": quality_score,
                "compliance": quality_score,
                "comments": [c["issue"] for c in comments],
            },
        }

    return critique
