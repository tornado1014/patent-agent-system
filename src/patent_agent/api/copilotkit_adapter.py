"""
Adapter to expose LangGraph workflows as CopilotKit agents.

This module bridges the Patent Agent LangGraph workflows with CopilotKit,
enabling real-time UI integration and Human-in-the-Loop functionality.
"""

from typing import Any

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

logger = structlog.get_logger()


def create_patent_agents() -> dict[str, CompiledStateGraph]:
    """
    Create CopilotKit-compatible agent instances for all workflows.

    Returns:
        Dictionary mapping agent name to compiled LangGraph
    """
    checkpointer = MemorySaver()

    agents = {}

    # Try to import each workflow graph
    try:
        from patent_agent.graphs.spec_writing_graph import create_spec_writing_graph

        spec_graph = create_spec_writing_graph(checkpointer)
        agents["spec_writing"] = spec_graph.graph
        logger.info("agent_registered", agent="spec_writing")
    except ImportError as e:
        logger.warning("agent_import_failed", agent="spec_writing", error=str(e))

    try:
        from patent_agent.graphs.oa_response_graph import create_oa_response_graph

        oa_graph = create_oa_response_graph(checkpointer)
        agents["oa_response"] = oa_graph.graph
        logger.info("agent_registered", agent="oa_response")
    except ImportError as e:
        logger.warning("agent_import_failed", agent="oa_response", error=str(e))

    try:
        from patent_agent.graphs.prior_art_graph import create_prior_art_graph

        prior_art_graph = create_prior_art_graph(checkpointer)
        agents["prior_art"] = prior_art_graph.graph
        logger.info("agent_registered", agent="prior_art")
    except ImportError as e:
        logger.warning("agent_import_failed", agent="prior_art", error=str(e))

    try:
        from patent_agent.graphs.translation_graph import create_translation_graph

        translation_graph = create_translation_graph(checkpointer)
        agents["translation"] = translation_graph.graph
        logger.info("agent_registered", agent="translation")
    except ImportError as e:
        logger.warning("agent_import_failed", agent="translation", error=str(e))

    try:
        from patent_agent.graphs.analysis_graph import create_analysis_graph

        analysis_graph = create_analysis_graph(checkpointer)
        agents["analysis"] = analysis_graph.graph
        logger.info("agent_registered", agent="analysis")
    except ImportError as e:
        logger.warning("agent_import_failed", agent="analysis", error=str(e))

    # Add main router as default agent
    try:
        from patent_agent.graphs.main_router import create_main_graph

        main_graph = create_main_graph(checkpointer=checkpointer)
        agents["patent_agent"] = main_graph
        logger.info("agent_registered", agent="patent_agent (main)")
    except ImportError as e:
        logger.warning("agent_import_failed", agent="patent_agent", error=str(e))

    logger.info("agents_created", count=len(agents), agents=list(agents.keys()))

    return agents


def get_agent_metadata() -> dict[str, dict[str, Any]]:
    """
    Get metadata for all available agents.

    Returns:
        Dictionary mapping agent name to metadata
    """
    return {
        "patent_agent": {
            "name": "Patent Agent",
            "description": "특허 업무 자동화를 위한 메인 에이전트",
            "workflows": ["spec_writing", "oa_response", "prior_art", "translation", "analysis"],
        },
        "spec_writing": {
            "name": "명세서 작성",
            "description": "발명 정보로부터 특허 명세서를 자동 생성합니다.",
            "steps": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
            "hitl_checkpoints": ["E4", "E6", "E9"],
        },
        "oa_response": {
            "name": "OA 대응",
            "description": "거절이유통지서를 분석하고 대응서를 작성합니다.",
            "steps": ["P1", "P2", "P3", "P4", "P5"],
            "hitl_checkpoints": ["P4", "P5"],
        },
        "prior_art": {
            "name": "선행기술조사",
            "description": "다중 데이터베이스에서 선행기술을 검색합니다.",
            "steps": ["S1", "S2", "S3", "S4", "S5"],
            "hitl_checkpoints": ["S5"],
        },
        "translation": {
            "name": "특허번역",
            "description": "영문 특허를 한국어로 번역합니다.",
            "steps": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "hitl_checkpoints": ["T6"],
        },
        "analysis": {
            "name": "특허분석",
            "description": "특허 포트폴리오 및 경쟁사를 분석합니다.",
            "steps": ["A1", "A2", "A3", "A4", "A5"],
            "hitl_checkpoints": ["A5"],
        },
    }
