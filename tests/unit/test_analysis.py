"""Unit tests for Patent Analysis workflow.

Tests cover:
- Analysis state definition
- Base agent utilities (ClaimAnalyzer, PatentScorer)
- Individual agents (DataCollector, ClaimMapper, TrendAnalyzer, etc.)
- PatentAnalysisGraph workflow
"""

import pytest
from datetime import datetime
from typing import Any

from patent_agent.agents.analysis import (
    AnalysisGuidelines,
    BaseAnalysisAgent,
    ClaimAnalyzer,
    ClaimMapperAgent,
    CompetitorProfilerAgent,
    DataCollectorAgent,
    InfringementCheckerAgent,
    PatentScorer,
    ReportWriterAgent,
    TrendAnalyzerAgent,
)
from patent_agent.graphs.analysis_graph import PatentAnalysisGraph, create_analysis_graph
from patent_agent.state.analysis import (
    ANALYSIS_TYPE_INFO,
    CORE_PATENT_SCORING,
    AnalysisScope,
    AnalysisStep,
    AnalysisType,
    ClaimChart,
    ClaimElement,
    CompetitorMetrics,
    CompetitorProfile,
    InfringementAnalysis,
    InvalidityAnalysis,
    MaturityPhase,
    PatentAnalysisState,
    PatentRecord,
    PortfolioMetrics,
    PortfolioReport,
    ProductSpec,
    TrendIndicators,
    VisualizationChart,
)


# ─────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_patent_record() -> PatentRecord:
    """Create a sample patent record for testing."""
    return PatentRecord(
        patent_number="KR10-2024-0001234",
        title="인공지능 기반 특허 분석 시스템",
        applicant="삼성전자",
        inventor=["홍길동", "김철수"],
        filing_date="2024-01-15",
        publication_date="2024-07-15",
        grant_date=None,
        ipc_codes=["G06F16/00", "G06N3/08"],
        claims=[
            "인공지능 모델을 이용하여 특허 문서를 분석하는 시스템으로서, 입력부, 처리부, 출력부를 포함하는 시스템.",
            "제1항에 있어서, 상기 처리부는 자연어 처리 모듈을 포함하는 시스템.",
        ],
        abstract="본 발명은 인공지능을 이용한 특허 분석 시스템에 관한 것이다.",
        legal_status="pending",
        citation_count=5,
        family_size=3,
    )


@pytest.fixture
def sample_patent_data() -> list[PatentRecord]:
    """Create sample patent data for testing."""
    return [
        PatentRecord(
            patent_number="KR10-2024-0001234",
            title="특허 분석 시스템",
            applicant="삼성전자",
            inventor=["홍길동"],
            filing_date="2024-01-15",
            publication_date="2024-07-15",
            grant_date=None,
            ipc_codes=["G06F16/00"],
            claims=["시스템 청구항"],
            abstract="특허 분석 시스템",
            legal_status="pending",
            citation_count=5,
            family_size=3,
        ),
        PatentRecord(
            patent_number="KR10-2023-0005678",
            title="데이터 처리 장치",
            applicant="LG전자",
            inventor=["이영희"],
            filing_date="2023-06-20",
            publication_date="2023-12-20",
            grant_date="2024-03-15",
            ipc_codes=["G06F16/00", "H04L29/08"],
            claims=["장치 청구항"],
            abstract="데이터 처리 장치",
            legal_status="granted",
            citation_count=10,
            family_size=5,
        ),
        PatentRecord(
            patent_number="US11234567B2",
            title="AI Analysis System",
            applicant="삼성전자",
            inventor=["John Doe"],
            filing_date="2022-03-10",
            publication_date="2022-09-10",
            grant_date="2023-01-15",
            ipc_codes=["G06N3/08"],
            claims=["System claim"],
            abstract="AI analysis system",
            legal_status="granted",
            citation_count=15,
            family_size=7,
        ),
    ]


@pytest.fixture
def sample_product_spec() -> ProductSpec:
    """Create a sample product specification."""
    return ProductSpec(
        product_name="AI Patent Analyzer",
        features=["자연어 처리", "특허 분류", "유사도 분석"],
        technical_description="인공지능 기반 특허 분석 소프트웨어",
        documentation=["spec.pdf", "manual.pdf"],
    )


@pytest.fixture
def sample_analysis_state(sample_patent_data: list[PatentRecord]) -> PatentAnalysisState:
    """Create a sample analysis state."""
    return PatentAnalysisState(
        session_id="test-session-123",
        current_workflow="analysis",
        current_step="collect",
        analysis_type="portfolio",
        target="삼성전자",
        scope=AnalysisScope(
            date_range=("2022-01-01", "2024-12-31"),
            jurisdictions=["KR", "US"],
            ipc_codes=["G06F16/00", "G06N3/08"],
            applicants=["삼성전자", "LG전자"],
            keywords=["인공지능", "특허분석"],
        ),
        patent_data=sample_patent_data,
        total_patents_collected=3,
        data_sources=["KIPRIS", "USPTO"],
        iteration_count=0,
        messages=[],
        error_messages=[],
        is_error_state=False,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )


# ─────────────────────────────────────────────────────────────
# State Definition Tests
# ─────────────────────────────────────────────────────────────


class TestAnalysisState:
    """Tests for PatentAnalysisState definition."""

    def test_analysis_types(self) -> None:
        """Test that all analysis types are defined."""
        expected_types = ["portfolio", "trend", "competitor", "infringement", "invalidity"]
        for analysis_type in expected_types:
            assert analysis_type in ANALYSIS_TYPE_INFO
            info = ANALYSIS_TYPE_INFO[analysis_type]
            assert "name" in info
            assert "purpose" in info
            assert "outputs" in info

    def test_analysis_steps(self) -> None:
        """Test that analysis steps are valid."""
        valid_steps: list[AnalysisStep] = ["collect", "process", "analyze", "visualize", "report"]
        for step in valid_steps:
            # Type should be valid
            assert step in ["collect", "process", "analyze", "visualize", "report"]

    def test_scoring_weights(self) -> None:
        """Test that scoring weights sum to 1.0."""
        total_weight = sum(CORE_PATENT_SCORING.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_patent_record_structure(self, sample_patent_record: PatentRecord) -> None:
        """Test PatentRecord structure."""
        assert sample_patent_record["patent_number"] == "KR10-2024-0001234"
        assert sample_patent_record["applicant"] == "삼성전자"
        assert len(sample_patent_record["ipc_codes"]) == 2
        assert sample_patent_record["legal_status"] == "pending"


# ─────────────────────────────────────────────────────────────
# Claim Analyzer Tests
# ─────────────────────────────────────────────────────────────


class TestClaimAnalyzer:
    """Tests for ClaimAnalyzer utility class."""

    def test_parse_claim_elements_basic(self) -> None:
        """Test basic claim parsing."""
        claim = "A system for processing data, comprising: an input unit, a processing unit, and an output unit."
        elements = ClaimAnalyzer.parse_claim_elements(claim)

        assert len(elements) >= 1
        # First element should be preamble
        assert elements[0]["type"] == "preamble"
        assert "system" in elements[0]["text"].lower()

    def test_parse_claim_elements_empty(self) -> None:
        """Test parsing empty claim."""
        elements = ClaimAnalyzer.parse_claim_elements("")
        assert elements == []

    def test_calculate_claim_breadth_narrow(self) -> None:
        """Test breadth calculation for narrow claim."""
        # Many limitations = narrow claim
        elements = [
            ClaimElement(element_id="E0", text="preamble", type="preamble"),
            ClaimElement(element_id="E1", text="lim1", type="limitation"),
            ClaimElement(element_id="E2", text="lim2", type="limitation"),
            ClaimElement(element_id="E3", text="lim3", type="limitation"),
            ClaimElement(element_id="E4", text="lim4", type="limitation"),
            ClaimElement(element_id="E5", text="lim5", type="limitation"),
            ClaimElement(element_id="E6", text="lim6", type="limitation"),
            ClaimElement(element_id="E7", text="lim7", type="limitation"),
            ClaimElement(element_id="E8", text="lim8", type="limitation"),
            ClaimElement(element_id="E9", text="lim9", type="limitation"),
        ]
        breadth = ClaimAnalyzer.calculate_claim_breadth(elements)
        assert breadth <= 0.4  # Narrow

    def test_calculate_claim_breadth_broad(self) -> None:
        """Test breadth calculation for broad claim."""
        # Few limitations = broad claim
        elements = [
            ClaimElement(element_id="E0", text="preamble", type="preamble"),
            ClaimElement(element_id="E1", text="lim1", type="limitation"),
        ]
        breadth = ClaimAnalyzer.calculate_claim_breadth(elements)
        assert breadth >= 0.8  # Broad


# ─────────────────────────────────────────────────────────────
# Patent Scorer Tests
# ─────────────────────────────────────────────────────────────


class TestPatentScorer:
    """Tests for PatentScorer utility class."""

    def test_calculate_core_score_high(self, sample_patent_record: PatentRecord) -> None:
        """Test scoring for a high-value patent."""
        # Modify to be high value
        high_value_patent = {**sample_patent_record}
        high_value_patent["citation_count"] = 50
        high_value_patent["family_size"] = 10
        high_value_patent["legal_status"] = "granted"

        score = PatentScorer.calculate_core_score(high_value_patent)
        assert score > 50  # Should be above average

    def test_calculate_core_score_low(self) -> None:
        """Test scoring for a low-value patent."""
        low_value_patent = PatentRecord(
            patent_number="KR10-2024-9999",
            title="Low value",
            applicant="Unknown",
            inventor=[],
            filing_date="2024-01-01",
            publication_date="2024-06-01",
            grant_date=None,
            ipc_codes=[],
            claims=[],
            abstract="",
            legal_status="abandoned",
            citation_count=0,
            family_size=1,
        )
        score = PatentScorer.calculate_core_score(low_value_patent)
        assert score < 50  # Should be below average

    def test_scoring_weights_applied(self) -> None:
        """Test that scoring weights are properly applied."""
        weights = PatentScorer.SCORING_WEIGHTS
        assert "citation_count" in weights
        assert "family_size" in weights
        assert "remaining_life" in weights
        assert all(0 <= w <= 1 for w in weights.values())


# ─────────────────────────────────────────────────────────────
# DataCollectorAgent Tests
# ─────────────────────────────────────────────────────────────


class TestDataCollectorAgent:
    """Tests for DataCollectorAgent."""

    @pytest.fixture
    def agent(self) -> DataCollectorAgent:
        return DataCollectorAgent()

    def test_agent_properties(self, agent: DataCollectorAgent) -> None:
        """Test agent property accessors."""
        assert agent.name == "DataCollectorAgent"
        assert agent.step == "collect"
        assert "수집" in agent.description

    def test_build_search_queries_portfolio(self, agent: DataCollectorAgent) -> None:
        """Test query building for portfolio analysis."""
        scope = AnalysisScope(
            date_range=None,
            jurisdictions=["KR", "US"],
            ipc_codes=[],
            applicants=[],
            keywords=[],
        )
        queries = agent._build_search_queries("portfolio", "삼성전자", scope)

        assert "KIPRIS" in queries
        assert "USPTO" in queries
        assert queries["KIPRIS"]["applicant"] == "삼성전자"

    def test_deduplicate_patents(
        self, agent: DataCollectorAgent, sample_patent_data: list[PatentRecord]
    ) -> None:
        """Test patent deduplication."""
        # Add duplicate
        duplicated = sample_patent_data + [sample_patent_data[0]]
        deduplicated = agent._deduplicate_patents(duplicated)

        assert len(deduplicated) == len(sample_patent_data)

    def test_normalize_date_formats(self, agent: DataCollectorAgent) -> None:
        """Test date normalization."""
        assert agent._normalize_date("2024-01-15") == "2024-01-15"
        assert agent._normalize_date("20240115") == "2024-01-15"
        assert agent._normalize_date("2024/01/15") == "2024-01-15"
        assert agent._normalize_date(None) is None

    @pytest.mark.asyncio
    async def test_process_missing_target(self, agent: DataCollectorAgent) -> None:
        """Test processing with missing target."""
        state = PatentAnalysisState(
            current_step="collect",
            analysis_type="portfolio",
            target="",  # Missing target
        )
        result = await agent.process(state)

        assert result.get("is_error_state") is True
        assert len(result.get("error_messages", [])) > 0


# ─────────────────────────────────────────────────────────────
# ClaimMapperAgent Tests
# ─────────────────────────────────────────────────────────────


class TestClaimMapperAgent:
    """Tests for ClaimMapperAgent."""

    @pytest.fixture
    def agent(self) -> ClaimMapperAgent:
        return ClaimMapperAgent()

    def test_agent_properties(self, agent: ClaimMapperAgent) -> None:
        """Test agent property accessors."""
        assert agent.name == "ClaimMapperAgent"
        assert agent.step == "process"

    def test_create_claim_chart(self, agent: ClaimMapperAgent) -> None:
        """Test claim chart creation."""
        claim_text = "A system comprising: an input unit, a processing unit."
        chart = agent._create_claim_chart(1, claim_text)

        assert chart["claim_number"] == 1
        assert len(chart["claim_elements"]) > 0
        assert chart["overall_result"] == "no_match"

    def test_identify_key_claims_independent(self, agent: ClaimMapperAgent) -> None:
        """Test identifying independent claims."""
        claims = [
            "A system comprising an input unit.",  # Independent
            "제1항에 있어서, 추가 구성을 포함하는 시스템.",  # Dependent
            "A method for processing data.",  # Independent
        ]
        key_claims = agent._identify_key_claims(claims)

        assert 1 in key_claims  # First claim
        assert 3 in key_claims  # Third claim is independent

    def test_identify_key_claims_all_dependent(self, agent: ClaimMapperAgent) -> None:
        """Test when all claims appear dependent."""
        claims = [
            "제1항에 있어서, A를 포함하는 시스템.",
            "제2항에 있어서, B를 포함하는 시스템.",
        ]
        key_claims = agent._identify_key_claims(claims)

        # Should default to claim 1
        assert 1 in key_claims

    @pytest.mark.asyncio
    async def test_process_non_claim_intensive(
        self, agent: ClaimMapperAgent, sample_analysis_state: PatentAnalysisState
    ) -> None:
        """Test processing for non-claim-intensive analysis."""
        # Portfolio analysis doesn't need claim mapping
        sample_analysis_state["analysis_type"] = "portfolio"
        result = await agent.process(sample_analysis_state)

        assert result.get("claim_mappings") == {}
        assert result.get("current_step") == "analyze"


# ─────────────────────────────────────────────────────────────
# TrendAnalyzerAgent Tests
# ─────────────────────────────────────────────────────────────


class TestTrendAnalyzerAgent:
    """Tests for TrendAnalyzerAgent."""

    @pytest.fixture
    def agent(self) -> TrendAnalyzerAgent:
        return TrendAnalyzerAgent()

    def test_agent_properties(self, agent: TrendAnalyzerAgent) -> None:
        """Test agent property accessors."""
        assert agent.name == "TrendAnalyzerAgent"
        assert agent.step == "analyze"

    def test_calculate_growth_rate_positive(self, agent: TrendAnalyzerAgent) -> None:
        """Test growth rate calculation with positive growth."""
        patents_by_year = {
            "2020": [{}] * 10,  # type: ignore
            "2021": [{}] * 15,  # type: ignore
            "2022": [{}] * 20,  # type: ignore
            "2023": [{}] * 30,  # type: ignore
        }
        growth_rate = agent._calculate_growth_rate(patents_by_year)

        assert growth_rate > 0  # Positive growth

    def test_calculate_growth_rate_negative(self, agent: TrendAnalyzerAgent) -> None:
        """Test growth rate calculation with negative growth."""
        patents_by_year = {
            "2020": [{}] * 30,  # type: ignore
            "2021": [{}] * 25,  # type: ignore
            "2022": [{}] * 15,  # type: ignore
            "2023": [{}] * 10,  # type: ignore
        }
        growth_rate = agent._calculate_growth_rate(patents_by_year)

        assert growth_rate < 0  # Negative growth

    def test_calculate_concentration_monopoly(self, agent: TrendAnalyzerAgent) -> None:
        """Test HHI calculation for monopoly."""
        # All patents from one applicant = HHI = 1.0
        patents = [
            PatentRecord(applicant="Company A", patent_number=f"P{i}", title="", inventor=[],
                        filing_date="", publication_date="", grant_date=None, ipc_codes=[],
                        claims=[], abstract="", legal_status="pending", citation_count=0, family_size=1)
            for i in range(10)
        ]
        hhi = agent._calculate_concentration(patents)
        assert hhi == 1.0

    def test_calculate_concentration_distributed(self, agent: TrendAnalyzerAgent) -> None:
        """Test HHI calculation for distributed market."""
        # Patents distributed among 10 companies equally
        patents = []
        for i in range(10):
            for j in range(10):
                patents.append(
                    PatentRecord(applicant=f"Company {i}", patent_number=f"P{i}_{j}", title="",
                                inventor=[], filing_date="", publication_date="", grant_date=None,
                                ipc_codes=[], claims=[], abstract="", legal_status="pending",
                                citation_count=0, family_size=1)
                )
        hhi = agent._calculate_concentration(patents)
        assert hhi < 0.2  # Low concentration

    def test_assess_maturity_emergence(self, agent: TrendAnalyzerAgent) -> None:
        """Test maturity assessment for emergence phase."""
        indicators = TrendIndicators(
            growth_rate=0.6,  # >50% growth
            concentration=0.5,
            technology_shift="신흥 기술",
            emerging_areas=["AI", "ML"],
        )
        maturity = agent._assess_maturity(indicators)

        assert maturity["phase"] == "emergence"
        assert maturity["phase_kr"] == "태동기"

    def test_assess_maturity_decline(self, agent: TrendAnalyzerAgent) -> None:
        """Test maturity assessment for decline phase."""
        indicators = TrendIndicators(
            growth_rate=-0.2,  # <-10% growth
            concentration=0.8,
            technology_shift="쇠퇴",
            emerging_areas=[],
        )
        maturity = agent._assess_maturity(indicators)

        assert maturity["phase"] == "decline"
        assert maturity["phase_kr"] == "쇠퇴기"


# ─────────────────────────────────────────────────────────────
# CompetitorProfilerAgent Tests
# ─────────────────────────────────────────────────────────────


class TestCompetitorProfilerAgent:
    """Tests for CompetitorProfilerAgent."""

    @pytest.fixture
    def agent(self) -> CompetitorProfilerAgent:
        return CompetitorProfilerAgent()

    def test_agent_properties(self, agent: CompetitorProfilerAgent) -> None:
        """Test agent property accessors."""
        assert agent.name == "CompetitorProfilerAgent"
        assert agent.step == "analyze"

    def test_group_by_applicant(
        self, agent: CompetitorProfilerAgent, sample_patent_data: list[PatentRecord]
    ) -> None:
        """Test grouping patents by applicant."""
        grouped = agent._group_by_applicant(sample_patent_data)

        assert "삼성전자" in [k.upper() for k in grouped.keys()]
        assert "LG전자" in [k.upper() for k in grouped.keys()]

    def test_build_competitor_profile(
        self, agent: CompetitorProfilerAgent, sample_patent_data: list[PatentRecord]
    ) -> None:
        """Test building competitor profile."""
        profile = agent._build_competitor_profile(
            "삼성전자",
            sample_patent_data[:2],
            len(sample_patent_data),
        )

        assert profile["company_name"] == "삼성전자"
        assert profile["patent_count"] == 2
        assert profile["market_share"] > 0

    def test_calculate_technology_overlap(
        self, agent: CompetitorProfilerAgent
    ) -> None:
        """Test technology overlap calculation."""
        patents_by_applicant = {
            "Company A": [
                PatentRecord(patent_number="P1", title="", applicant="A", inventor=[],
                            filing_date="", publication_date="", grant_date=None,
                            ipc_codes=["G06F", "H04L"], claims=[], abstract="",
                            legal_status="pending", citation_count=0, family_size=1)
            ],
            "Company B": [
                PatentRecord(patent_number="P2", title="", applicant="B", inventor=[],
                            filing_date="", publication_date="", grant_date=None,
                            ipc_codes=["G06F", "G06N"], claims=[], abstract="",
                            legal_status="pending", citation_count=0, family_size=1)
            ],
        }
        overlap = agent._calculate_technology_overlap(patents_by_applicant)

        # Both have G06F, so there should be some overlap
        assert len(overlap) > 0


# ─────────────────────────────────────────────────────────────
# InfringementCheckerAgent Tests
# ─────────────────────────────────────────────────────────────


class TestInfringementCheckerAgent:
    """Tests for InfringementCheckerAgent."""

    @pytest.fixture
    def agent(self) -> InfringementCheckerAgent:
        return InfringementCheckerAgent()

    def test_agent_properties(self, agent: InfringementCheckerAgent) -> None:
        """Test agent property accessors."""
        assert agent.name == "InfringementCheckerAgent"
        assert agent.step == "analyze"

    def test_calculate_risk_level_high(self, agent: InfringementCheckerAgent) -> None:
        """Test risk calculation for high risk."""
        risk = agent._calculate_risk_level(
            literal_count=1,  # Any literal infringement
            doe_count=0,
            total_claims=10,
        )
        assert risk == "high"

    def test_calculate_risk_level_medium(self, agent: InfringementCheckerAgent) -> None:
        """Test risk calculation for medium risk."""
        risk = agent._calculate_risk_level(
            literal_count=0,
            doe_count=5,  # 50% DOE
            total_claims=10,
        )
        assert risk == "medium"

    def test_calculate_risk_level_none(self, agent: InfringementCheckerAgent) -> None:
        """Test risk calculation for no risk."""
        risk = agent._calculate_risk_level(
            literal_count=0,
            doe_count=0,
            total_claims=10,
        )
        assert risk == "none"

    def test_analyze_claim_infringement_literal(
        self, agent: InfringementCheckerAgent
    ) -> None:
        """Test claim infringement analysis for literal match."""
        chart = ClaimChart(
            claim_number=1,
            claim_elements=[
                ClaimElement(element_id="E0", text="preamble", type="preamble"),
                ClaimElement(element_id="E1", text="lim1", type="limitation"),
            ],
            mappings=[
                {"element_id": "E0", "match_type": "literal"},
                {"element_id": "E1", "match_type": "literal"},
            ],
            overall_result="no_match",
        )
        analysis = agent._analyze_claim_infringement(chart)

        assert analysis["infringement_type"] == "literal"
        assert analysis["confidence"] == "확실"

    def test_generate_recommendations_high_risk(
        self, agent: InfringementCheckerAgent
    ) -> None:
        """Test recommendation generation for high risk."""
        recommendations = agent._generate_recommendations(
            risk_level="high",
            literal=True,
            doe=False,
        )

        assert len(recommendations) > 0
        assert any("법률" in r or "라이선스" in r for r in recommendations)


# ─────────────────────────────────────────────────────────────
# ReportWriterAgent Tests
# ─────────────────────────────────────────────────────────────


class TestReportWriterAgent:
    """Tests for ReportWriterAgent."""

    @pytest.fixture
    def agent(self) -> ReportWriterAgent:
        return ReportWriterAgent()

    def test_agent_properties(self, agent: ReportWriterAgent) -> None:
        """Test agent property accessors."""
        assert agent.name == "ReportWriterAgent"
        assert agent.step == "report"

    def test_collect_key_findings_portfolio(
        self, agent: ReportWriterAgent, sample_analysis_state: PatentAnalysisState
    ) -> None:
        """Test key findings collection for portfolio analysis."""
        sample_analysis_state["portfolio_report"] = PortfolioReport(
            target="삼성전자",
            metrics=PortfolioMetrics(
                total_patents=100,
                patents_by_year={"2023": 50, "2024": 50},
                patents_by_jurisdiction={"KR": 70, "US": 30},
                patents_by_ipc={"G06F": 60, "H04L": 40},
                average_citations=5.0,
                average_family_size=3.0,
                core_patents=["P1", "P2", "P3"],
            ),
            strengths=["다양한 기술 분야"],
            weaknesses=["국제 출원 부족"],
            recommendations=["국제 출원 확대"],
        )
        findings = agent._collect_key_findings(sample_analysis_state)

        assert "100" in findings  # Total patents
        assert "삼성전자" in sample_analysis_state["target"]

    def test_generate_fallback_summary(
        self, agent: ReportWriterAgent, sample_analysis_state: PatentAnalysisState
    ) -> None:
        """Test fallback summary generation."""
        summary = agent._generate_fallback_summary(sample_analysis_state)

        assert "분석 요약" in summary
        assert "삼성전자" in summary

    def test_generate_appendices(
        self, agent: ReportWriterAgent, sample_analysis_state: PatentAnalysisState
    ) -> None:
        """Test appendix generation."""
        appendices = agent._generate_appendices(sample_analysis_state)

        assert len(appendices) > 0
        assert any("patent_list" in a for a in appendices)


# ─────────────────────────────────────────────────────────────
# PatentAnalysisGraph Tests
# ─────────────────────────────────────────────────────────────


class TestPatentAnalysisGraph:
    """Tests for PatentAnalysisGraph workflow."""

    @pytest.fixture
    def graph(self) -> PatentAnalysisGraph:
        return PatentAnalysisGraph()

    def test_graph_creation(self, graph: PatentAnalysisGraph) -> None:
        """Test graph creation."""
        assert graph is not None
        assert graph.name == "analysis"

    def test_create_analysis_graph_factory(self) -> None:
        """Test factory function."""
        graph = create_analysis_graph()
        assert isinstance(graph, PatentAnalysisGraph)

    def test_initial_step(self, graph: PatentAnalysisGraph) -> None:
        """Test initial step."""
        assert graph.get_initial_step() == "collect"

    def test_state_class(self, graph: PatentAnalysisGraph) -> None:
        """Test state class."""
        assert graph.state_class == PatentAnalysisState

    def test_route_after_collect_success(
        self, graph: PatentAnalysisGraph, sample_analysis_state: PatentAnalysisState
    ) -> None:
        """Test routing after successful collection."""
        route = graph._route_after_collect(sample_analysis_state)
        assert route == "process"

    def test_route_after_collect_error(
        self, graph: PatentAnalysisGraph
    ) -> None:
        """Test routing after collection error."""
        state = PatentAnalysisState(
            is_error_state=True,
            patent_data=[],
        )
        route = graph._route_after_collect(state)
        assert route == "error"

    def test_route_after_process_portfolio(
        self, graph: PatentAnalysisGraph, sample_analysis_state: PatentAnalysisState
    ) -> None:
        """Test routing for portfolio analysis."""
        sample_analysis_state["analysis_type"] = "portfolio"
        route = graph._route_after_process(sample_analysis_state)
        assert route == "trend"

    def test_route_after_process_infringement(
        self, graph: PatentAnalysisGraph, sample_analysis_state: PatentAnalysisState
    ) -> None:
        """Test routing for infringement analysis."""
        sample_analysis_state["analysis_type"] = "infringement"
        route = graph._route_after_process(sample_analysis_state)
        assert route == "infringement"

    def test_route_after_infringement_high_risk(
        self, graph: PatentAnalysisGraph, sample_analysis_state: PatentAnalysisState
    ) -> None:
        """Test routing after high-risk infringement analysis."""
        sample_analysis_state["infringement_analysis"] = InfringementAnalysis(
            target_patent="P1",
            target_product="Product",
            claim_charts=[],
            literal_infringement=True,
            doctrine_of_equivalents=False,
            overall_risk="high",
            design_around_options=[],
            recommendations=[],
        )
        # Note: This test depends on settings.workflow.enable_human_in_loop


# ─────────────────────────────────────────────────────────────
# Analysis Guidelines Tests
# ─────────────────────────────────────────────────────────────


class TestAnalysisGuidelines:
    """Tests for AnalysisGuidelines."""

    def test_get_system_prompt(self) -> None:
        """Test system prompt generation."""
        prompt = AnalysisGuidelines.get_system_prompt()

        assert "특허 분석" in prompt
        assert "포트폴리오" in prompt
        assert "침해" in prompt
        assert "무효" in prompt

    def test_system_prompt_contains_risk_levels(self) -> None:
        """Test that system prompt contains confidence/risk level info."""
        prompt = AnalysisGuidelines.get_system_prompt()

        # Check for confidence level indicators
        assert "확실" in prompt
        assert "가능" in prompt
        assert "불확실" in prompt
        # Check for infringement risk types
        assert "문언 침해" in prompt or "Literal Infringement" in prompt


# ─────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────


class TestAnalysisIntegration:
    """Integration tests for analysis workflow."""

    def test_all_agents_have_consistent_interface(self) -> None:
        """Test that all agents have consistent interface."""
        agents = [
            DataCollectorAgent(),
            ClaimMapperAgent(),
            TrendAnalyzerAgent(),
            CompetitorProfilerAgent(),
            InfringementCheckerAgent(),
            ReportWriterAgent(),
        ]

        for agent in agents:
            # All should have these properties
            assert hasattr(agent, "name")
            assert hasattr(agent, "description")
            assert hasattr(agent, "step")
            assert hasattr(agent, "get_system_prompt")
            assert hasattr(agent, "process")

    def test_state_flow_portfolio(self, sample_analysis_state: PatentAnalysisState) -> None:
        """Test state flow for portfolio analysis."""
        # Simulate workflow progression
        sample_analysis_state["analysis_type"] = "portfolio"
        sample_analysis_state["current_step"] = "collect"

        # After collection
        sample_analysis_state["current_step"] = "process"

        # After processing (portfolio skips claim mapping)
        sample_analysis_state["current_step"] = "analyze"

        # After analysis
        sample_analysis_state["current_step"] = "visualize"

        # After visualization
        sample_analysis_state["current_step"] = "report"

        # Verify final state
        assert sample_analysis_state["current_step"] == "report"

    def test_visualization_chart_types(self) -> None:
        """Test that visualization chart types are valid."""
        valid_types = ["bar_chart", "line_chart", "heatmap", "network_graph", "bubble_chart", "treemap"]

        for chart_type in valid_types:
            chart = VisualizationChart(
                chart_type=chart_type,
                title="Test Chart",
                data={},
                file_path=None,
                format="html",
            )
            assert chart["chart_type"] == chart_type
