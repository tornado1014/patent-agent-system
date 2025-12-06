"""
LangGraph workflow definitions for patent agent system.

Each workflow is implemented as a StateGraph with:
- Domain-specific nodes
- Reflexion Loop for quality assurance
- Human-in-the-Loop checkpoints
"""

from patent_agent.graphs.base_graph import BasePatentGraph
from patent_agent.graphs.reflexion import create_reflexion_subgraph, ReflexionConfig
from patent_agent.graphs.main_router import MainRouter, create_main_graph
from patent_agent.graphs.spec_writing_graph import SpecWritingGraph, create_spec_writing_graph
from patent_agent.graphs.oa_response_graph import OAResponseGraph, create_oa_response_graph
from patent_agent.graphs.prior_art_graph import PriorArtSearchGraph, create_prior_art_search_graph
from patent_agent.graphs.translation_graph import TranslationGraph, create_translation_graph


__all__ = [
    # Base and utilities
    "BasePatentGraph",
    "create_reflexion_subgraph",
    "ReflexionConfig",
    "MainRouter",
    "create_main_graph",
    # Workflow graphs
    "SpecWritingGraph",
    "create_spec_writing_graph",
    "OAResponseGraph",
    "create_oa_response_graph",
    "PriorArtSearchGraph",
    "create_prior_art_search_graph",
    "TranslationGraph",
    "create_translation_graph",
]
