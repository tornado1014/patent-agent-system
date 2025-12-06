"""
State definitions for patent agent workflows.

Each workflow has its own state that extends BasePatentState.
"""

from patent_agent.state.base import (
    BasePatentState,
    ConfidenceLevel,
    Evidence,
    HumanFeedback,
    QualityScore,
    WorkflowType,
)
from patent_agent.state.analysis import PatentAnalysisState
from patent_agent.state.oa_response import OAResponseState
from patent_agent.state.prior_art import PriorArtSearchState
from patent_agent.state.spec_writing import SpecWritingState
from patent_agent.state.translation import TranslationState


__all__ = [
    # Base
    "BasePatentState",
    "WorkflowType",
    "ConfidenceLevel",
    "Evidence",
    "QualityScore",
    "HumanFeedback",
    # Domain States
    "SpecWritingState",
    "OAResponseState",
    "PriorArtSearchState",
    "TranslationState",
    "PatentAnalysisState",
]
