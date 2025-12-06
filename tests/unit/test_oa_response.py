"""Tests for OA Response workflow.

Tests the 5-phase PALLAS-EVIDENCE workflow and Zero Hallucination laws.
"""

from __future__ import annotations

import pytest

from patent_agent.agents.oa_response import (
    AmendmentDrafterAgent,
    DocParserAgent,
    RebuttalWriterAgent,
    RejectionAnalyzerAgent,
    ReportGeneratorAgent,
)
from patent_agent.rules.oa_response_laws import AmendmentGuideline, OAResponseLaws
from patent_agent.state.oa_response import (
    Amendment,
    ApplicationInfo,
    CitedReference,
    OADocument,
    OAResponseState,
    RebuttalPoint,
    RejectionAnalysis,
    VerificationStamp,
)


class TestOAResponseLaws:
    """Tests for OAResponseLaws (Zero Hallucination)."""

    @pytest.fixture
    def laws(self) -> OAResponseLaws:
        """Create laws instance."""
        return OAResponseLaws()

    def test_domain_property(self, laws: OAResponseLaws) -> None:
        """Test domain property."""
        assert laws.domain == "oa_response"

    def test_laws_contain_zero_hallucination(self, laws: OAResponseLaws) -> None:
        """Test that laws include LAW-6 to LAW-10."""
        law_ids = laws.laws.keys()
        assert "LAW-6" in law_ids
        assert "LAW-7" in law_ids
        assert "LAW-8" in law_ids
        assert "LAW-9" in law_ids
        assert "LAW-10" in law_ids

    def test_system_prompt_contains_laws(self, laws: OAResponseLaws) -> None:
        """Test that system prompt includes all laws."""
        prompt = laws.get_system_prompt()
        assert "LAW-6" in prompt
        assert "원문 불가침" in prompt
        assert "LAW-7" in prompt
        assert "근거 추적성" in prompt
        assert "LAW-8" in prompt
        assert "할루시네이션" in prompt

    def test_validate_prohibited_expression(self, laws: OAResponseLaws) -> None:
        """Test that prohibited expressions are detected."""
        # Use exact prohibited expression from PROHIBITED_EXPRESSIONS
        text_with_violation = "이 발명은 효과적인 것~로 보인다."
        result = laws.validate_output(text_with_violation)

        assert not result.is_valid
        assert len(result.violations) > 0
        assert any("~로 보인다" in v.original_text for v in result.violations)

    def test_validate_clean_text(self, laws: OAResponseLaws) -> None:
        """Test that clean text passes validation."""
        clean_text = '청구항 1은 [출원서:p.5:para.3]에 기재된 "프로세서"를 포함한다.'
        result = laws.validate_output(clean_text)

        # Should not have critical violations
        critical = [v for v in result.violations if v.severity.value == "critical"]
        assert len(critical) == 0

    def test_validate_multiple_prohibited_expressions(self, laws: OAResponseLaws) -> None:
        """Test detection of multiple prohibited expressions."""
        text = "일반적으로 이러한 발명은 당연히 효과가 있을 것이다."
        result = laws.validate_output(text)

        assert not result.is_valid
        # Should detect multiple violations
        assert len(result.violations) >= 2


class TestDocParserAgent:
    """Tests for DocParserAgent."""

    @pytest.fixture
    def agent(self) -> DocParserAgent:
        """Create agent for testing."""
        return DocParserAgent()

    def test_agent_properties(self, agent: DocParserAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "DocParser"
        assert "문서" in agent.description
        assert agent.phase == "P1"

    def test_system_prompt_contains_guidelines(self, agent: DocParserAgent) -> None:
        """Test that system prompt includes guidelines."""
        prompt = agent._get_full_system_prompt()
        assert "원문" in prompt
        assert "LAW-6" in prompt

    @pytest.mark.asyncio
    async def test_process_without_oa_document(self, agent: DocParserAgent) -> None:
        """Test processing without OA document."""
        state: OAResponseState = {
            "session_id": "test",
            "current_workflow": "oa_response",
            "current_step": "P1",
            "messages": [],
            "iteration_count": 0,
            "human_approval_required": False,
            "quality_score": {},
            "error_messages": [],
            "is_error_state": False,
        }

        result = await agent.process(state)

        assert result.get("is_error_state") is True
        assert len(result.get("error_messages", [])) > 0


class TestRejectionAnalyzerAgent:
    """Tests for RejectionAnalyzerAgent."""

    @pytest.fixture
    def agent(self) -> RejectionAnalyzerAgent:
        """Create agent for testing."""
        return RejectionAnalyzerAgent()

    def test_agent_properties(self, agent: RejectionAnalyzerAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "RejectionAnalyzer"
        assert "거절이유" in agent.description
        assert agent.phase == "P2"

    def test_classify_novelty_rejection(self, agent: RejectionAnalyzerAgent) -> None:
        """Test novelty rejection classification."""
        text = "청구항 1은 특허법 제29조 제1항에 의해 신규성이 없음"
        rejection_type = agent._classify_rejection_type(text)
        assert rejection_type == "novelty"

    def test_classify_inventive_step_rejection(self, agent: RejectionAnalyzerAgent) -> None:
        """Test inventive step rejection classification."""
        text = "청구항 1은 특허법 제29조 제2항에 의해 진보성이 없음"
        rejection_type = agent._classify_rejection_type(text)
        assert rejection_type == "inventive_step"

    def test_extract_legal_basis(self, agent: RejectionAnalyzerAgent) -> None:
        """Test legal basis extraction."""
        text = "특허법 제29조 제2항에 의하여 거절합니다."
        legal_basis = agent._extract_legal_basis(text)
        assert "제29조" in legal_basis
        assert "제2항" in legal_basis

    def test_find_cited_references(self, agent: RejectionAnalyzerAgent) -> None:
        """Test cited reference detection."""
        text = "청구항 1은 D1과 D2의 결합에 의해 용이하게 도출됨"
        refs = agent._find_cited_references(text, [])
        assert "D1" in refs
        assert "D2" in refs


class TestRebuttalWriterAgent:
    """Tests for RebuttalWriterAgent."""

    @pytest.fixture
    def agent(self) -> RebuttalWriterAgent:
        """Create agent for testing."""
        return RebuttalWriterAgent()

    def test_agent_properties(self, agent: RebuttalWriterAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "RebuttalWriter"
        assert "반박" in agent.description
        assert agent.phase == "P3"

    def test_system_prompt_contains_protocol(self, agent: RebuttalWriterAgent) -> None:
        """Test that system prompt includes rebuttal protocol."""
        prompt = agent._get_full_system_prompt()
        assert "근거 체인" in prompt
        assert "심사관" in prompt
        assert "문서에 없으면" in prompt


class TestAmendmentDrafterAgent:
    """Tests for AmendmentDrafterAgent."""

    @pytest.fixture
    def agent(self) -> AmendmentDrafterAgent:
        """Create agent for testing."""
        return AmendmentDrafterAgent()

    def test_agent_properties(self, agent: AmendmentDrafterAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "AmendmentDrafter"
        assert "보정" in agent.description
        assert agent.phase == "P4"

    def test_uses_amendment_guideline(self, agent: AmendmentDrafterAgent) -> None:
        """Test that agent uses specialized amendment guidelines."""
        assert isinstance(agent._guidelines, AmendmentGuideline)

    def test_system_prompt_contains_law3(self, agent: AmendmentDrafterAgent) -> None:
        """Test that system prompt mentions LAW-3 (2 amendments required)."""
        prompt = agent._get_full_system_prompt()
        assert "2개" in prompt or "최소" in prompt

    def test_assess_new_matter_risk_low(self, agent: AmendmentDrafterAgent) -> None:
        """Test low risk assessment."""
        section = "명세서 명시적 기재: YES\n신규사항 리스크: low"
        risk = agent._assess_new_matter_risk(section)
        assert risk == "low"

    def test_assess_new_matter_risk_high(self, agent: AmendmentDrafterAgent) -> None:
        """Test high risk assessment."""
        section = "⚠️ 신규사항 리스크: HIGH"
        risk = agent._assess_new_matter_risk(section)
        assert risk == "high"


class TestReportGeneratorAgent:
    """Tests for ReportGeneratorAgent."""

    @pytest.fixture
    def agent(self) -> ReportGeneratorAgent:
        """Create agent for testing."""
        return ReportGeneratorAgent()

    def test_agent_properties(self, agent: ReportGeneratorAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "ReportGenerator"
        assert "보고서" in agent.description
        assert agent.phase == "P5"

    def test_system_prompt_contains_templates(self, agent: ReportGeneratorAgent) -> None:
        """Test that system prompt includes report templates."""
        prompt = agent._get_full_system_prompt()
        assert "검토보고서" in prompt
        assert "의견서" in prompt
        assert "검증 스탬프" in prompt

    def test_count_citations(self, agent: ReportGeneratorAgent) -> None:
        """Test citation counting."""
        text = """
        [출원서:p.5:para.3] 기재된 내용
        [D1:p.10:para.2] 인용문헌 내용
        [OA:청구항1] 심사관 의견
        """
        count = agent._count_citations(text)
        assert count == 3

    def test_aggregate_confidence_high(self, agent: ReportGeneratorAgent) -> None:
        """Test confidence aggregation - high."""
        confidences = ["확실", "확실", "가능"]
        result = agent._aggregate_confidence(confidences)
        assert result == "확실"

    def test_aggregate_confidence_medium(self, agent: ReportGeneratorAgent) -> None:
        """Test confidence aggregation - medium."""
        confidences = ["가능", "가능", "불확실"]
        result = agent._aggregate_confidence(confidences)
        assert result == "가능"

    def test_aggregate_confidence_low(self, agent: ReportGeneratorAgent) -> None:
        """Test confidence aggregation - low."""
        confidences = ["불확실", "불확실", "불확실"]
        result = agent._aggregate_confidence(confidences)
        assert result == "불확실"


class TestOAResponseState:
    """Tests for OAResponseState structures."""

    def test_application_info_structure(self) -> None:
        """Test ApplicationInfo TypedDict structure."""
        app_info: ApplicationInfo = {
            "application_number": "10-2024-0001234",
            "filing_date": "2024-01-15",
            "title": "데이터 처리 장치",
            "applicant": "테스트 주식회사",
            "inventor": "홍길동",
            "agent": "대리인 김철수",
            "response_deadline": "2024-05-15",
        }

        assert app_info["application_number"] == "10-2024-0001234"
        assert app_info["agent"] == "대리인 김철수"

    def test_oa_document_structure(self) -> None:
        """Test OADocument TypedDict structure."""
        oa_doc: OADocument = {
            "oa_number": "OA-2024-001",
            "issue_date": "2024-03-15",
            "examiner": "심사관 이영희",
            "rejection_reasons": ["청구항 1은 D1에 의해 신규성이 없음"],
            "cited_claims": [1, 2, 3],
            "total_pages": 5,
        }

        assert len(oa_doc["rejection_reasons"]) == 1
        assert oa_doc["cited_claims"] == [1, 2, 3]

    def test_cited_reference_structure(self) -> None:
        """Test CitedReference TypedDict structure."""
        ref: CitedReference = {
            "reference_id": "D1",
            "document_number": "10-2020-0005678",
            "title": "선행기술 발명",
            "applicant": "선행기술사",
            "filing_date": "2020-01-01",
            "publication_date": "2020-07-01",
            "relevant_passages": ["프로세서와 메모리를 포함하는 장치"],
            "total_pages": 20,
        }

        assert ref["reference_id"] == "D1"
        assert len(ref["relevant_passages"]) == 1

    def test_rejection_analysis_structure(self) -> None:
        """Test RejectionAnalysis TypedDict structure."""
        analysis: RejectionAnalysis = {
            "rejection_type": "inventive_step",
            "legal_basis": "특허법 제29조 제2항",
            "examiner_argument": "청구항 1은 D1과 D2의 결합에 의해 용이하게 도출됨",
            "cited_references": ["D1", "D2"],
            "affected_claims": [1, 2],
            "confidence": "확실",
            "analysis_notes": "진보성 거절이유 분석",
        }

        assert analysis["rejection_type"] == "inventive_step"
        assert len(analysis["cited_references"]) == 2

    def test_rebuttal_point_structure(self) -> None:
        """Test RebuttalPoint TypedDict structure."""
        point: RebuttalPoint = {
            "target_rejection": "특허법 제29조 제2항",
            "argument_type": "technical_difference",
            "argument": "청구항 1의 구성요소 A는 D1에 개시되지 않음",
            "evidence": [
                {
                    "source": "출원서",
                    "page": 5,
                    "paragraph": 3,
                    "verbatim_text": "프로세서와 결합된 메모리",
                    "confidence": "확실",
                }
            ],
            "confidence": "확실",
        }

        assert point["argument_type"] == "technical_difference"
        assert len(point["evidence"]) == 1

    def test_amendment_structure(self) -> None:
        """Test Amendment TypedDict structure."""
        amendment: Amendment = {
            "amendment_id": "A",
            "strategy_name": "최소 한정 보정",
            "original_claim": "청구항 1. 프로세서를 포함하는 장치.",
            "amended_claim": "청구항 1. 메모리와 연결된 프로세서를 포함하는 장치.",
            "added_limitations": ["메모리와 연결된"],
            "specification_support": [],
            "new_matter_risk": "low",
            "effectiveness": "확실",
        }

        assert amendment["amendment_id"] == "A"
        assert amendment["new_matter_risk"] == "low"

    def test_verification_stamp_structure(self) -> None:
        """Test VerificationStamp TypedDict structure."""
        stamp: VerificationStamp = {
            "hallucination_check": "CLEAR",
            "verbatim_accuracy": 100.0,
            "citation_verified": 10,
            "citation_total": 10,
            "spec_support_verified": True,
            "legal_basis_verified": True,
            "timestamp": "2024-03-15T14:30:00",
        }

        assert stamp["hallucination_check"] == "CLEAR"
        assert stamp["verbatim_accuracy"] == 100.0


class TestOAResponseWorkflow:
    """Integration tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_workflow_imports(self) -> None:
        """Test that workflow can be imported."""
        from patent_agent.graphs.oa_response_graph import (
            OAResponseGraph,
            create_oa_response_graph,
        )

        graph = create_oa_response_graph()
        assert graph is not None
        assert graph.name == "oa_response"

    @pytest.mark.asyncio
    async def test_workflow_initial_step(self) -> None:
        """Test that workflow starts at P1."""
        from patent_agent.graphs.oa_response_graph import create_oa_response_graph

        graph = create_oa_response_graph()
        assert graph.get_initial_step() == "P1"


class TestBaseOAResponseAgent:
    """Tests for BaseOAResponseAgent utility methods."""

    @pytest.fixture
    def agent(self) -> DocParserAgent:
        """Create a concrete agent for testing base methods."""
        return DocParserAgent()

    def test_create_evidence(self, agent: DocParserAgent) -> None:
        """Test evidence creation."""
        evidence = agent.create_evidence(
            source="출원서",
            page=5,
            paragraph=3,
            verbatim_text="프로세서를 포함하는",
            confidence="확실",
        )

        assert evidence["source"] == "출원서"
        assert evidence["page"] == 5
        assert evidence["verbatim_text"] == "프로세서를 포함하는"

    def test_format_citation_with_page_para(self, agent: DocParserAgent) -> None:
        """Test citation formatting with page and paragraph."""
        citation = agent.format_citation(
            source="출원서",
            page=5,
            paragraph=3,
        )
        assert citation == "[출원서:p.5:para.3]"

    def test_format_citation_with_claim(self, agent: DocParserAgent) -> None:
        """Test citation formatting with claim number."""
        citation = agent.format_citation(
            source="OA",
            claim_number=1,
        )
        assert citation == "[OA:청구항1]"

    def test_quote_verbatim(self, agent: DocParserAgent) -> None:
        """Test verbatim quoting."""
        quoted = agent.quote_verbatim("프로세서를 포함하는 장치")
        assert quoted == '"프로세서를 포함하는 장치"'

    def test_get_confidence_indicator(self, agent: DocParserAgent) -> None:
        """Test confidence indicator mapping."""
        assert agent.get_confidence_indicator("확실") == "🟢"
        assert agent.get_confidence_indicator("가능") == "🟡"
        assert agent.get_confidence_indicator("불확실") == "🔴"

    def test_format_with_confidence(self, agent: DocParserAgent) -> None:
        """Test formatting text with confidence."""
        formatted = agent.format_with_confidence(
            "청구항 1은 D1과 다름",
            "확실",
        )
        assert formatted.startswith("🟢")
        assert "청구항 1은 D1과 다름" in formatted
