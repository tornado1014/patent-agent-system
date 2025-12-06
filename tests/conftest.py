"""
Pytest configuration and fixtures for patent-agent-system tests.

Provides:
- Mock LLM for testing without API calls
- Sample states for each workflow
- Common test utilities
"""

import pytest
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage


# ═══════════════════════════════════════════════════════════════
# Fixtures: Mock LLM
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Mock response"))
    llm.invoke = MagicMock(return_value=AIMessage(content="Mock response"))
    return llm


@pytest.fixture
def mock_llm_with_response():
    """Factory fixture to create mock LLM with custom response."""
    def _create_mock(response: str):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content=response))
        llm.invoke = MagicMock(return_value=AIMessage(content=response))
        return llm
    return _create_mock


# ═══════════════════════════════════════════════════════════════
# Fixtures: Sample States
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def base_state():
    """Create a base patent state for testing."""
    from patent_agent.state.base import create_initial_state
    return create_initial_state("spec_writing", "test-session-001")


@pytest.fixture
def spec_writing_state():
    """Create a sample spec writing state."""
    from patent_agent.state.spec_writing import SpecWritingState

    return SpecWritingState(
        messages=[HumanMessage(content="새로운 배터리 기술에 대한 명세서 작성")],
        current_workflow="spec_writing",
        current_step="E1",
        iteration_count=0,
        max_iterations=3,
        quality_score={
            "overall": 0,
            "completeness": 0,
            "accuracy": 0,
            "compliance": 0,
            "comments": [],
        },
        review_comments=[],
        human_approval_required=False,
        human_feedback=[],
        pending_approval_checkpoint=None,
        evidence_chain=[],
        error_messages=[],
        is_error_state=False,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        session_id="test-spec-001",
        # Spec writing specific
        invention_disclosure="본 발명은 리튬이온 배터리의 에너지 밀도를 향상시키는 기술에 관한 것이다.",
        tech_field="H01M (전지)",
        inventor_name="홍길동",
        applicant_name="테스트 주식회사",
    )


@pytest.fixture
def oa_response_state():
    """Create a sample OA response state."""
    from patent_agent.state.oa_response import OAResponseState

    return OAResponseState(
        messages=[HumanMessage(content="OA 거절이유 분석 요청")],
        current_workflow="oa_response",
        current_step="P1",
        iteration_count=0,
        max_iterations=3,
        quality_score={
            "overall": 0,
            "completeness": 0,
            "accuracy": 0,
            "compliance": 0,
            "comments": [],
        },
        review_comments=[],
        human_approval_required=False,
        human_feedback=[],
        pending_approval_checkpoint=None,
        evidence_chain=[],
        error_messages=[],
        is_error_state=False,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        session_id="test-oa-001",
        # OA response specific
        application_info={
            "application_number": "10-2023-0123456",
            "filing_date": "2023-06-15",
            "title": "리튬이온 배터리 장치",
            "applicant": "테스트 주식회사",
            "inventor": "홍길동",
            "agent": "김변리사",
            "response_deadline": "2024-02-15",
        },
    )


@pytest.fixture
def translation_state():
    """Create a sample translation state."""
    from patent_agent.state.translation import TranslationState

    return TranslationState(
        messages=[HumanMessage(content="US patent translation request")],
        current_workflow="translation",
        current_step="P1",
        iteration_count=0,
        max_iterations=3,
        quality_score={
            "overall": 0,
            "completeness": 0,
            "accuracy": 0,
            "compliance": 0,
            "comments": [],
        },
        review_comments=[],
        human_approval_required=False,
        human_feedback=[],
        pending_approval_checkpoint=None,
        evidence_chain=[],
        error_messages=[],
        is_error_state=False,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        session_id="test-trans-001",
        # Translation specific
        source_document={
            "file_path": "/test/patent.pdf",
            "file_type": "pdf",
            "total_pages": 25,
            "requires_ocr": False,
            "patent_number": "US10,123,456",
            "filing_date": "2020-01-15",
        },
    )


# ═══════════════════════════════════════════════════════════════
# Fixtures: Sample Documents
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def sample_claim_en():
    """Sample English claim for testing."""
    return """1. A battery device comprising:
    a cathode comprising a lithium compound;
    an anode comprising graphite; and
    an electrolyte disposed between the cathode and the anode,
    wherein the cathode further comprises a conductive additive."""


@pytest.fixture
def sample_claim_ko():
    """Sample Korean claim for testing."""
    return """1. 리튬 화합물을 포함하는 양극;
그래파이트를 포함하는 음극; 및
상기 양극과 상기 음극 사이에 배치된 전해질을 포함하고,
상기 양극은 도전성 첨가제를 더 포함하는 것을 특징으로 하는 배터리 장치."""


@pytest.fixture
def sample_rejection_reason():
    """Sample OA rejection reason for testing."""
    return """청구항 1은 특허법 제29조 제2항에 의해 거절됩니다.
인용문헌 D1(KR10-2020-0012345)에는 리튬 화합물을 포함하는 양극이 개시되어 있고,
인용문헌 D2(US2019/0123456)에는 도전성 첨가제가 개시되어 있습니다.
따라서 당업자가 D1과 D2를 조합하여 청구항 1의 발명을 용이하게 도출할 수 있습니다."""


# ═══════════════════════════════════════════════════════════════
# Fixtures: Rules Validation
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def translation_laws():
    """Get TranslationLaws instance for testing."""
    from patent_agent.rules.translation_laws import TranslationLaws
    return TranslationLaws()


@pytest.fixture
def oa_laws():
    """Get OAResponseLaws instance for testing."""
    from patent_agent.rules.oa_response_laws import OAResponseLaws
    return OAResponseLaws()


@pytest.fixture
def spec_writing_laws():
    """Get SpecWritingLaws instance for testing."""
    from patent_agent.rules.spec_writing_laws import SpecWritingLaws
    return SpecWritingLaws()


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════


def assert_no_critical_violations(result):
    """Assert that validation result has no critical violations."""
    assert not result.has_critical_violations, (
        f"Critical violations found: "
        f"{[v.to_dict() for v in result.violations if v.severity.value == 'critical']}"
    )


def assert_valid(result):
    """Assert that validation result is valid."""
    assert result.is_valid, (
        f"Validation failed with violations: "
        f"{[v.to_dict() for v in result.violations]}"
    )
