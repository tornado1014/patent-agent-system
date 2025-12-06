"""Tests for specification writing workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from patent_agent.agents.spec_writing import (
    ClaimDrafterAgent,
    DrawingGeneratorAgent,
    InventionAnalyzerAgent,
    PriorArtSearcherAgent,
    QualityCheckerAgent,
    SpecWriterAgent,
)
from patent_agent.state.spec_writing import (
    Claim,
    ClaimSet,
    Drawing,
    PriorArt,
    Specification,
    SpecWritingState,
)


class TestInventionAnalyzerAgent:
    """Tests for InventionAnalyzerAgent."""

    @pytest.fixture
    def agent(self) -> InventionAnalyzerAgent:
        """Create agent for testing."""
        return InventionAnalyzerAgent()

    def test_agent_properties(self, agent: InventionAnalyzerAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "InventionAnalyzer"
        assert "발명 신고서" in agent.description

    def test_system_prompt_contains_guidelines(self, agent: InventionAnalyzerAgent) -> None:
        """Test that system prompt includes guidelines."""
        prompt = agent._get_full_system_prompt()
        assert "특허" in prompt
        assert "발명" in prompt

    @pytest.mark.asyncio
    async def test_process_without_disclosure(self, agent: InventionAnalyzerAgent) -> None:
        """Test processing without invention disclosure."""
        state: SpecWritingState = {
            "session_id": "test",
            "current_workflow": "spec_writing",
            "current_step": "E1",
            "messages": [],
            "iteration_count": 0,
            "human_approval_required": False,
            "quality_score": {},
            "error_messages": [],
            "is_error_state": False,
            "review_comments": [],
        }

        result = await agent.process(state)

        assert result.get("is_error_state") is True
        assert len(result.get("error_messages", [])) > 0


class TestPriorArtSearcherAgent:
    """Tests for PriorArtSearcherAgent."""

    @pytest.fixture
    def agent(self) -> PriorArtSearcherAgent:
        """Create agent for testing."""
        return PriorArtSearcherAgent()

    def test_agent_properties(self, agent: PriorArtSearcherAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "PriorArtSearcher"
        assert "선행기술" in agent.description

    @pytest.mark.asyncio
    async def test_process_without_key_features(self, agent: PriorArtSearcherAgent) -> None:
        """Test processing without key features."""
        state: SpecWritingState = {
            "session_id": "test",
            "current_workflow": "spec_writing",
            "current_step": "E3",
            "messages": [],
            "iteration_count": 0,
            "human_approval_required": False,
            "quality_score": {},
            "error_messages": [],
            "is_error_state": False,
            "review_comments": [],
        }

        result = await agent.process(state)

        assert result.get("is_error_state") is True


class TestClaimDrafterAgent:
    """Tests for ClaimDrafterAgent."""

    @pytest.fixture
    def agent(self) -> ClaimDrafterAgent:
        """Create agent for testing."""
        return ClaimDrafterAgent()

    def test_agent_properties(self, agent: ClaimDrafterAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "ClaimDrafter"
        assert "청구항" in agent.description

    def test_system_prompt_contains_claim_format(self, agent: ClaimDrafterAgent) -> None:
        """Test that system prompt includes claim format rules."""
        prompt = agent._get_full_system_prompt()
        assert "청구항" in prompt
        assert "독립항" in prompt or "종속항" in prompt

    @pytest.mark.asyncio
    async def test_validate_claims_multiple_periods(self, agent: ClaimDrafterAgent) -> None:
        """Test claim validation detects multiple periods."""
        claim_set: ClaimSet = {
            "claims": [
                {
                    "claim_number": 1,
                    "claim_type": "independent",
                    "depends_on": None,
                    "preamble": "장치에 있어서",
                    "body": "구성요소를 포함하는",
                    "full_text": "장치에 있어서. 구성요소를 포함하는 장치.",  # Wrong: 2 periods
                }
            ],
            "total_independent": 1,
            "total_dependent": 0,
        }

        errors = await agent._validate_claims(claim_set)

        assert len(errors) > 0
        assert any("마침표" in err for err in errors)


class TestSpecWriterAgent:
    """Tests for SpecWriterAgent."""

    @pytest.fixture
    def agent(self) -> SpecWriterAgent:
        """Create agent for testing."""
        return SpecWriterAgent()

    def test_agent_properties(self, agent: SpecWriterAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "SpecWriter"
        assert "명세서" in agent.description

    def test_system_prompt_contains_sections(self, agent: SpecWriterAgent) -> None:
        """Test that system prompt includes specification sections."""
        prompt = agent._get_full_system_prompt()
        assert "기술분야" in prompt
        assert "배경기술" in prompt
        assert "발명의 효과" in prompt


class TestDrawingGeneratorAgent:
    """Tests for DrawingGeneratorAgent."""

    @pytest.fixture
    def agent(self) -> DrawingGeneratorAgent:
        """Create agent for testing."""
        return DrawingGeneratorAgent()

    def test_agent_properties(self, agent: DrawingGeneratorAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "DrawingGenerator"
        assert "도면" in agent.description

    def test_generate_block_diagram_svg(self, agent: DrawingGeneratorAgent) -> None:
        """Test block diagram SVG generation."""
        components = ["프로세서", "메모리", "통신부"]
        reference_numerals = {"100": "프로세서", "200": "메모리", "300": "통신부"}

        svg = agent._generate_block_diagram_svg(components, reference_numerals)

        assert "<svg" in svg
        assert "</svg>" in svg
        assert "100" in svg or "rect" in svg

    def test_generate_flowchart_svg(self, agent: DrawingGeneratorAgent) -> None:
        """Test flowchart SVG generation."""
        components = ["단계 1", "단계 2", "단계 3"]
        reference_numerals = {}

        svg = agent._generate_flowchart_svg(components, reference_numerals)

        assert "<svg" in svg
        assert "</svg>" in svg
        assert "시작" in svg
        assert "종료" in svg


class TestQualityCheckerAgent:
    """Tests for QualityCheckerAgent."""

    @pytest.fixture
    def agent(self) -> QualityCheckerAgent:
        """Create agent for testing."""
        return QualityCheckerAgent()

    def test_agent_properties(self, agent: QualityCheckerAgent) -> None:
        """Test agent name and description."""
        assert agent.name == "QualityChecker"
        assert "품질" in agent.description

    def test_calculate_quality_score(self, agent: QualityCheckerAgent) -> None:
        """Test quality score calculation."""
        results = {
            "check1": {"status": "PASS"},
            "check2": {"status": "PASS"},
            "check3": {"status": "WARNING"},
            "check4": {"status": "FAIL"},
        }

        score = agent._calculate_quality_score(results)

        assert score["total_checks"] == 4
        assert score["passed"] == 2
        assert score["warnings"] == 1
        assert score["failed"] == 1
        assert 0 <= score["overall"] <= 100

    def test_calculate_quality_score_all_pass(self, agent: QualityCheckerAgent) -> None:
        """Test quality score with all passing checks."""
        results = {
            "check1": {"status": "PASS"},
            "check2": {"status": "PASS"},
        }

        score = agent._calculate_quality_score(results)

        assert score["overall"] == 100

    def test_calculate_quality_score_all_fail(self, agent: QualityCheckerAgent) -> None:
        """Test quality score with all failing checks."""
        results = {
            "check1": {"status": "FAIL"},
            "check2": {"status": "FAIL"},
        }

        score = agent._calculate_quality_score(results)

        assert score["overall"] == 0


class TestSpecWritingState:
    """Tests for SpecWritingState."""

    def test_claim_structure(self) -> None:
        """Test Claim TypedDict structure."""
        claim: Claim = {
            "claim_number": 1,
            "claim_type": "independent",
            "depends_on": None,
            "preamble": "데이터 처리 장치에 있어서",
            "body": "프로세서를 포함하는",
            "full_text": "데이터 처리 장치에 있어서, 프로세서를 포함하는 데이터 처리 장치.",
        }

        assert claim["claim_number"] == 1
        assert claim["claim_type"] == "independent"

    def test_claim_set_structure(self) -> None:
        """Test ClaimSet TypedDict structure."""
        claim_set: ClaimSet = {
            "claims": [
                {
                    "claim_number": 1,
                    "claim_type": "independent",
                    "depends_on": None,
                    "preamble": "",
                    "body": "",
                    "full_text": "",
                }
            ],
            "total_independent": 1,
            "total_dependent": 0,
        }

        assert len(claim_set["claims"]) == 1
        assert claim_set["total_independent"] == 1

    def test_prior_art_structure(self) -> None:
        """Test PriorArt TypedDict structure."""
        prior_art: PriorArt = {
            "document_number": "10-2020-0001234",
            "title": "선행기술 제목",
            "applicant": "선행기술 출원인",
            "filing_date": "2020-01-01",
            "relevance": "관련성 설명",
            "limitations": "한계점 설명",
        }

        assert prior_art["document_number"] == "10-2020-0001234"

    def test_drawing_structure(self) -> None:
        """Test Drawing TypedDict structure."""
        drawing: Drawing = {
            "figure_number": 1,
            "title": "전체 구성도",
            "description": "본 발명의 전체 구성을 나타내는 블록도",
            "svg_path": "/path/to/drawing.svg",
            "reference_numerals": {"100": "프로세서", "200": "메모리"},
        }

        assert drawing["figure_number"] == 1
        assert len(drawing["reference_numerals"]) == 2

    def test_specification_structure(self) -> None:
        """Test Specification TypedDict structure."""
        spec: Specification = {
            "title": "데이터 처리 장치",
            "technical_field": "본 발명은 데이터 처리 기술에 관한 것이다.",
            "background_art": "종래에는...",
            "problems_to_solve": "본 발명은...",
            "means_to_solve": "상기 목적을 달성하기 위하여...",
            "effects": "본 발명에 따르면...",
            "brief_description_drawings": "도 1은...",
            "detailed_description": "이하, 첨부된 도면을 참조하여...",
            "industrial_applicability": None,
        }

        assert spec["title"] == "데이터 처리 장치"
        assert spec["industrial_applicability"] is None


class TestSpecWritingWorkflow:
    """Integration tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_workflow_imports(self) -> None:
        """Test that workflow can be imported."""
        from patent_agent.graphs.spec_writing_graph import (
            SpecWritingGraph,
            create_spec_writing_graph,
        )

        graph = create_spec_writing_graph()
        assert graph is not None
        assert graph.name == "spec_writing"

    @pytest.mark.asyncio
    async def test_workflow_initial_step(self) -> None:
        """Test that workflow starts at E1."""
        from patent_agent.graphs.spec_writing_graph import create_spec_writing_graph

        graph = create_spec_writing_graph()
        assert graph.get_initial_step() == "E1"

    def test_human_checkpoints(self) -> None:
        """Test human checkpoint configuration."""
        from patent_agent.state.spec_writing import HUMAN_CHECKPOINTS

        assert "E4" in HUMAN_CHECKPOINTS
        assert "E6" in HUMAN_CHECKPOINTS
        assert "E9" in HUMAN_CHECKPOINTS
