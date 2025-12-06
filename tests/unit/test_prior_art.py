"""Tests for Prior Art Search workflow.

Tests the complete prior art search workflow including:
- State definitions
- All 5 agents
- Search graph
- Query building and keyword extraction
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from patent_agent.agents.prior_art import (
    ClaimComparerAgent,
    MultiDBSearcherAgent,
    NoveltyAssessorAgent,
    QueryBuilderAgent,
    RelevanceAnalyzerAgent,
)
from patent_agent.agents.prior_art.base import (
    BasePriorArtAgent,
    IPC_SECTIONS,
    PriorArtGuidelines,
)
from patent_agent.state.prior_art import (
    AnalyzedReference,
    ClaimComparison,
    InventiveStepAssessment,
    NoveltyAssessment,
    PatentResult,
    PriorArtSearchState,
    SearchQuery,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_invention() -> str:
    """Sample invention description for testing."""
    return """
    본 발명은 딥러닝 기반 자연어처리 시스템에 관한 것으로,
    트랜스포머 아키텍처를 활용하여 한국어 특허 문서의
    의미적 유사도를 분석하는 방법에 관한 것이다.
    """


@pytest.fixture
def sample_claims() -> list[str]:
    """Sample claims for testing."""
    return [
        "트랜스포머 인코더를 포함하는 자연어처리 장치",
        "상기 트랜스포머 인코더와 연결된 유사도 분석 모듈",
        "한국어 토큰화 처리부를 포함하는 전처리 모듈",
    ]


@pytest.fixture
def sample_state(sample_invention: str, sample_claims: list[str]) -> PriorArtSearchState:
    """Sample prior art search state."""
    return PriorArtSearchState(
        messages=[],
        current_workflow="prior_art",
        current_step="query",
        iteration_count=0,
        max_iterations=3,
        session_id="test-session-123",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        search_type="novelty_search",
        target_invention=sample_invention,
        target_claims=sample_claims,
        target_filing_date=None,
        search_keywords=[],
        expanded_keywords=[],
        ipc_codes=[],
        cpc_codes=[],
        relevance_threshold=0.7,
    )


@pytest.fixture
def sample_patent_result() -> PatentResult:
    """Sample patent search result."""
    return PatentResult(
        database="KIPRIS",
        document_number="KR10-2020-0123456",
        title="딥러닝 기반 문서 분류 시스템",
        applicant="테스트 출원인",
        inventor=["발명자1", "발명자2"],
        filing_date="2020-01-15",
        publication_date="2020-07-15",
        grant_date=None,
        ipc_codes=["G06N3/08", "G06F18/00"],
        abstract="본 발명은 딥러닝을 활용한 문서 분류 시스템에 관한 것이다.",
        claims_text=None,
        pdf_url=None,
        family_id=None,
    )


@pytest.fixture
def sample_analyzed_reference(sample_patent_result: PatentResult) -> AnalyzedReference:
    """Sample analyzed reference."""
    return AnalyzedReference(
        reference_id="R1",
        patent_result=sample_patent_result,
        relevance_score=0.85,
        matched_claims=[1, 2],
        key_features_matched=["딥러닝", "문서 처리"],
        key_features_missing=["트랜스포머 인코더"],
        differentiation="본원 발명은 트랜스포머 아키텍처를 사용하지만, 선행기술은 일반 CNN 사용",
        confidence="가능",
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM for testing."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
    return mock


# ─────────────────────────────────────────────────────────────
# Guidelines Tests
# ─────────────────────────────────────────────────────────────


class TestPriorArtGuidelines:
    """Tests for PriorArtGuidelines."""

    @pytest.fixture
    def guidelines(self) -> PriorArtGuidelines:
        return PriorArtGuidelines()

    def test_get_system_prompt(self, guidelines: PriorArtGuidelines) -> None:
        """Test that system prompt contains key elements."""
        prompt = guidelines.get_system_prompt()

        assert "검색 원칙" in prompt
        assert "분석 원칙" in prompt
        assert "출력 형식" in prompt

    def test_calculate_weighted_relevance(self, guidelines: PriorArtGuidelines) -> None:
        """Test weighted relevance calculation."""
        # Full scores
        score = guidelines.calculate_weighted_relevance(
            technical_overlap=1.0,
            claim_coverage=1.0,
            temporal_relevance=1.0,
            applicant_relevance=1.0,
        )
        assert score == pytest.approx(1.0)

        # Zero scores
        score = guidelines.calculate_weighted_relevance(
            technical_overlap=0.0,
            claim_coverage=0.0,
            temporal_relevance=0.0,
            applicant_relevance=0.0,
        )
        assert score == pytest.approx(0.0)

        # Weighted average
        score = guidelines.calculate_weighted_relevance(
            technical_overlap=0.5,
            claim_coverage=0.5,
            temporal_relevance=0.5,
            applicant_relevance=0.5,
        )
        assert score == pytest.approx(0.5)

    def test_relevance_scoring_weights(self, guidelines: PriorArtGuidelines) -> None:
        """Test that scoring weights sum to 1.0."""
        weights = guidelines.RELEVANCE_SCORING
        total = sum(weights.values())
        assert total == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────
# Base Agent Tests
# ─────────────────────────────────────────────────────────────


class TestBasePriorArtAgent:
    """Tests for BasePriorArtAgent utility methods."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> QueryBuilderAgent:
        """Create agent instance for testing."""
        agent = QueryBuilderAgent(llm=mock_llm)
        return agent

    def test_get_ipc_section_name(self, agent: QueryBuilderAgent) -> None:
        """Test IPC section name lookup."""
        assert agent.get_ipc_section_name("G06N") == "물리학"
        assert agent.get_ipc_section_name("H04L") == "전기"
        assert agent.get_ipc_section_name("A61K") == "생활필수품"
        assert agent.get_ipc_section_name("") == "알 수 없음"
        assert agent.get_ipc_section_name("Z99") == "알 수 없음"

    def test_suggest_ipc_codes(self, agent: QueryBuilderAgent) -> None:
        """Test IPC code suggestion."""
        ai_codes = agent.suggest_ipc_codes("AI/ML deep learning")
        assert "G06N" in ai_codes

        network_codes = agent.suggest_ipc_codes("networking communication")
        assert "H04L" in network_codes

        # Default for unknown domain
        unknown_codes = agent.suggest_ipc_codes("something random")
        assert len(unknown_codes) > 0

    def test_calculate_relevance_score(self, agent: QueryBuilderAgent) -> None:
        """Test relevance score calculation."""
        score = agent.calculate_relevance_score(
            technical_overlap=0.8,
            claim_coverage=0.6,
            temporal_relevance=1.0,
            applicant_relevance=0.0,
        )
        # 0.8*0.4 + 0.6*0.3 + 1.0*0.15 + 0.0*0.15 = 0.32 + 0.18 + 0.15 = 0.65
        assert score == pytest.approx(0.65)

    def test_relevance_to_confidence(self, agent: QueryBuilderAgent) -> None:
        """Test confidence level mapping."""
        assert agent.relevance_to_confidence(0.9) == "확실"
        assert agent.relevance_to_confidence(0.8) == "확실"
        assert agent.relevance_to_confidence(0.7) == "가능"
        assert agent.relevance_to_confidence(0.5) == "가능"
        assert agent.relevance_to_confidence(0.3) == "불확실"

    def test_get_confidence_indicator(self, agent: QueryBuilderAgent) -> None:
        """Test confidence indicator mapping."""
        assert agent.get_confidence_indicator("확실") == "🟢"
        assert agent.get_confidence_indicator("가능") == "🟡"
        assert agent.get_confidence_indicator("불확실") == "🔴"

    def test_get_available_databases(self, agent: QueryBuilderAgent) -> None:
        """Test available databases list."""
        dbs = agent.get_available_databases()
        assert "KIPRIS" in dbs
        assert "USPTO" in dbs


# ─────────────────────────────────────────────────────────────
# QueryBuilderAgent Tests
# ─────────────────────────────────────────────────────────────


class TestQueryBuilderAgent:
    """Tests for QueryBuilderAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> QueryBuilderAgent:
        return QueryBuilderAgent(llm=mock_llm)

    def test_agent_properties(self, agent: QueryBuilderAgent) -> None:
        """Test agent property values."""
        assert agent.name == "QueryBuilder"
        assert agent.step == "query"
        assert "쿼리" in agent.description or "검색" in agent.description

    def test_get_system_prompt(self, agent: QueryBuilderAgent) -> None:
        """Test system prompt content."""
        prompt = agent.get_system_prompt()
        assert "키워드" in prompt
        assert "IPC" in prompt

    @pytest.mark.asyncio
    async def test_process_missing_invention(
        self, agent: QueryBuilderAgent
    ) -> None:
        """Test error handling when invention is missing."""
        state = PriorArtSearchState(
            current_workflow="prior_art",
            target_invention="",
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True
        assert len(result.get("error_messages", [])) > 0

    @pytest.mark.asyncio
    async def test_process_builds_queries(
        self,
        agent: QueryBuilderAgent,
        sample_state: PriorArtSearchState,
        mock_llm: MagicMock,
    ) -> None:
        """Test that process builds search queries."""
        # Setup mock LLM responses (need 4 calls: keywords, expanded, ipc, cpc)
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                MagicMock(content="- 딥러닝\n- 자연어처리\n- 트랜스포머"),  # keywords
                MagicMock(content="- deep learning\n- NLP"),  # expanded
                MagicMock(content="G06N\nG06F"),  # IPC codes
                MagicMock(content="G06N\nG06F"),  # CPC codes (calls IPC internally again)
            ]
        )

        result = await agent.process(sample_state)

        assert "search_keywords" in result
        assert "expanded_keywords" in result
        assert "ipc_codes" in result
        assert "search_queries" in result
        assert result.get("current_step") == "search"

    def test_build_kipris_query(self, agent: QueryBuilderAgent) -> None:
        """Test KIPRIS query building."""
        query = agent._build_kipris_query(
            keywords=["딥러닝", "deep learning"],
            ipc_codes=["G06N"],
        )

        assert "딥러닝" in query or "deep learning" in query
        assert "G06N" in query


# ─────────────────────────────────────────────────────────────
# MultiDBSearcherAgent Tests
# ─────────────────────────────────────────────────────────────


class TestMultiDBSearcherAgent:
    """Tests for MultiDBSearcherAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> MultiDBSearcherAgent:
        return MultiDBSearcherAgent(llm=mock_llm)

    def test_agent_properties(self, agent: MultiDBSearcherAgent) -> None:
        """Test agent property values."""
        assert agent.name == "MultiDBSearcher"
        assert agent.step == "search"

    @pytest.mark.asyncio
    async def test_process_missing_queries(
        self, agent: MultiDBSearcherAgent
    ) -> None:
        """Test error handling when queries are missing."""
        state = PriorArtSearchState(
            current_workflow="prior_art",
            search_queries={},
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True

    @pytest.mark.asyncio
    async def test_process_executes_parallel_search(
        self, agent: MultiDBSearcherAgent
    ) -> None:
        """Test parallel search execution."""
        # Setup state with queries
        state = PriorArtSearchState(
            current_workflow="prior_art",
            search_queries={
                "KIPRIS": SearchQuery(
                    database="KIPRIS",
                    keywords=["test"],
                    ipc_codes=[],
                    cpc_codes=[],
                    applicant=None,
                    date_range=None,
                    query_string="test query",
                ),
            },
        )

        # Mock KIPRIS client
        with patch.object(agent, "_kipris_client") as mock_client:
            mock_client.search_patents = AsyncMock(return_value=[])

            result = await agent.process(state)

            assert "raw_results" in result
            assert "total_results_count" in result
            assert "search_errors" in result
            assert result.get("current_step") == "filter"


# ─────────────────────────────────────────────────────────────
# RelevanceAnalyzerAgent Tests
# ─────────────────────────────────────────────────────────────


class TestRelevanceAnalyzerAgent:
    """Tests for RelevanceAnalyzerAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> RelevanceAnalyzerAgent:
        return RelevanceAnalyzerAgent(llm=mock_llm)

    def test_agent_properties(self, agent: RelevanceAnalyzerAgent) -> None:
        """Test agent property values."""
        assert agent.name == "RelevanceAnalyzer"
        assert agent.step == "filter"

    @pytest.mark.asyncio
    async def test_process_missing_results(
        self, agent: RelevanceAnalyzerAgent
    ) -> None:
        """Test error handling when results are missing."""
        state = PriorArtSearchState(
            current_workflow="prior_art",
            raw_results={},
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True

    def test_keyword_similarity(self, agent: RelevanceAnalyzerAgent) -> None:
        """Test keyword-based similarity calculation."""
        sim = agent._keyword_similarity(
            "딥러닝 자연어처리 시스템",
            "딥러닝 기반 문서 처리 시스템",
        )

        assert 0.0 <= sim <= 1.0
        assert sim > 0.0  # Should have some overlap

    def test_extract_significant_words(self, agent: RelevanceAnalyzerAgent) -> None:
        """Test significant word extraction."""
        words = agent._extract_significant_words(
            "The 딥러닝 system for 자연어처리"
        )

        assert "딥러닝" in words
        assert "자연어처리" in words
        assert "the" not in words  # Stopword


# ─────────────────────────────────────────────────────────────
# ClaimComparerAgent Tests
# ─────────────────────────────────────────────────────────────


class TestClaimComparerAgent:
    """Tests for ClaimComparerAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> ClaimComparerAgent:
        return ClaimComparerAgent(llm=mock_llm)

    def test_agent_properties(self, agent: ClaimComparerAgent) -> None:
        """Test agent property values."""
        assert agent.name == "ClaimComparer"
        assert agent.step == "analyze"

    @pytest.mark.asyncio
    async def test_process_missing_claims(
        self, agent: ClaimComparerAgent
    ) -> None:
        """Test error handling when claims are missing."""
        state = PriorArtSearchState(
            current_workflow="prior_art",
            target_claims=[],
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True

    def test_generate_markdown_table(
        self,
        agent: ClaimComparerAgent,
        sample_claims: list[str],
        sample_analyzed_reference: AnalyzedReference,
    ) -> None:
        """Test markdown table generation."""
        comparisons = [
            ClaimComparison(
                claim_number=1,
                claim_element="트랜스포머 인코더",
                prior_art_ref="R1 (KR10-2020-0123456)",
                prior_art_disclosure="유사한 인코더 개시",
                comparison_result="similar",
                notes="균등 범위",
            )
        ]

        table = agent._generate_markdown_table(
            comparisons=comparisons,
            claims=sample_claims,
            references=[sample_analyzed_reference],
        )

        assert "# 청구항 대비표" in table
        assert "트랜스포머 인코더" in table or "청구항" in table
        assert "|" in table  # Table format


# ─────────────────────────────────────────────────────────────
# NoveltyAssessorAgent Tests
# ─────────────────────────────────────────────────────────────


class TestNoveltyAssessorAgent:
    """Tests for NoveltyAssessorAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> NoveltyAssessorAgent:
        return NoveltyAssessorAgent(llm=mock_llm)

    def test_agent_properties(self, agent: NoveltyAssessorAgent) -> None:
        """Test agent property values."""
        assert agent.name == "NoveltyAssessor"
        assert agent.step == "report"

    @pytest.mark.asyncio
    async def test_process_missing_claims(
        self, agent: NoveltyAssessorAgent
    ) -> None:
        """Test error handling when claims are missing."""
        state = PriorArtSearchState(
            current_workflow="prior_art",
            target_claims=[],
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True

    def test_generate_executive_summary(
        self,
        agent: NoveltyAssessorAgent,
        sample_analyzed_reference: AnalyzedReference,
    ) -> None:
        """Test executive summary generation."""
        novelty_assessments = [
            NoveltyAssessment(
                claim_number=1,
                is_novel=True,
                blocking_reference=None,
                assessment_reasoning="신규성 있음",
                confidence="확실",
            )
        ]

        inventive_assessments = [
            InventiveStepAssessment(
                claim_number=1,
                has_inventive_step=True,
                combination_references=[],
                combination_motivation="없음",
                technical_difficulty="없음",
                unexpected_effect=None,
                confidence="가능",
            )
        ]

        summary = agent._generate_executive_summary(
            novelty_assessments=novelty_assessments,
            inventive_assessments=inventive_assessments,
            references=[sample_analyzed_reference],
            overall_novelty=True,
            overall_inventive=True,
        )

        assert "선행기술조사" in summary or "요약" in summary
        assert "신규성" in summary
        assert "진보성" in summary

    def test_generate_recommendations(self, agent: NoveltyAssessorAgent) -> None:
        """Test recommendation generation."""
        # Case 1: No issues
        recs = agent._generate_recommendations(
            novelty_assessments=[
                NoveltyAssessment(
                    claim_number=1,
                    is_novel=True,
                    blocking_reference=None,
                    assessment_reasoning="",
                    confidence="확실",
                )
            ],
            inventive_assessments=[
                InventiveStepAssessment(
                    claim_number=1,
                    has_inventive_step=True,
                    combination_references=[],
                    combination_motivation="",
                    technical_difficulty="",
                    unexpected_effect=None,
                    confidence="확실",
                )
            ],
            search_type="novelty_search",
        )
        assert len(recs) > 0
        assert "출원" in recs[0] or "저촉" in recs[0]

        # Case 2: Novelty issue
        recs = agent._generate_recommendations(
            novelty_assessments=[
                NoveltyAssessment(
                    claim_number=1,
                    is_novel=False,
                    blocking_reference="D1",
                    assessment_reasoning="",
                    confidence="확실",
                )
            ],
            inventive_assessments=[],
            search_type="novelty_search",
        )
        assert any("신규성" in r for r in recs)


# ─────────────────────────────────────────────────────────────
# State Definition Tests
# ─────────────────────────────────────────────────────────────


class TestPriorArtSearchState:
    """Tests for PriorArtSearchState."""

    def test_state_creation(self, sample_state: PriorArtSearchState) -> None:
        """Test state creation with required fields."""
        assert sample_state.get("current_workflow") == "prior_art"
        assert sample_state.get("search_type") == "novelty_search"
        assert len(sample_state.get("target_claims", [])) > 0

    def test_patent_result_structure(self, sample_patent_result: PatentResult) -> None:
        """Test PatentResult TypedDict structure."""
        assert sample_patent_result["database"] == "KIPRIS"
        assert sample_patent_result["document_number"].startswith("KR")
        assert len(sample_patent_result["ipc_codes"]) > 0

    def test_analyzed_reference_structure(
        self, sample_analyzed_reference: AnalyzedReference
    ) -> None:
        """Test AnalyzedReference TypedDict structure."""
        assert 0.0 <= sample_analyzed_reference["relevance_score"] <= 1.0
        assert sample_analyzed_reference["confidence"] in ["확실", "가능", "불확실"]

    def test_claim_comparison_structure(self) -> None:
        """Test ClaimComparison TypedDict structure."""
        comp = ClaimComparison(
            claim_number=1,
            claim_element="테스트 요소",
            prior_art_ref="D1",
            prior_art_disclosure="개시 내용",
            comparison_result="similar",
            notes="비고",
        )

        assert comp["comparison_result"] in ["identical", "similar", "different", "not_found"]


# ─────────────────────────────────────────────────────────────
# Graph Tests
# ─────────────────────────────────────────────────────────────


class TestPriorArtSearchGraph:
    """Tests for PriorArtSearchGraph."""

    def test_graph_creation(self) -> None:
        """Test graph can be created."""
        from patent_agent.graphs.prior_art_graph import (
            PriorArtSearchGraph,
            create_prior_art_search_graph,
        )

        graph = create_prior_art_search_graph()
        assert graph is not None
        assert graph.name == "prior_art_search"

    def test_graph_state_class(self) -> None:
        """Test graph uses correct state class."""
        from patent_agent.graphs.prior_art_graph import PriorArtSearchGraph

        graph = PriorArtSearchGraph()
        assert graph.state_class == PriorArtSearchState

    def test_graph_initial_step(self) -> None:
        """Test graph starts at correct step."""
        from patent_agent.graphs.prior_art_graph import PriorArtSearchGraph

        graph = PriorArtSearchGraph()
        assert graph.get_initial_step() == "query"

    def test_graph_builds_successfully(self) -> None:
        """Test graph compiles without errors."""
        from patent_agent.graphs.prior_art_graph import PriorArtSearchGraph

        graph = PriorArtSearchGraph()
        compiled = graph.graph  # Triggers lazy build

        assert compiled is not None


# ─────────────────────────────────────────────────────────────
# IPC Classification Tests
# ─────────────────────────────────────────────────────────────


class TestIPCClassification:
    """Tests for IPC classification utilities."""

    def test_ipc_sections_complete(self) -> None:
        """Test all IPC sections are defined."""
        expected_sections = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for section in expected_sections:
            assert section in IPC_SECTIONS

    def test_ipc_section_names_korean(self) -> None:
        """Test IPC section names are in Korean."""
        for section, name in IPC_SECTIONS.items():
            assert any("\uac00" <= c <= "\ud7af" for c in name), f"{section}: {name}"
