"""
State definition for Patent Analysis workflow.

Supports 5 analysis types: Portfolio, Trend, Competitor, Infringement, Invalidity.
"""

from typing import Literal, TypedDict

from patent_agent.state.base import BasePatentState, ConfidenceLevel


# Analysis types
AnalysisType = Literal[
    "portfolio",  # 포트폴리오 분석
    "trend",  # 기술 동향 분석
    "competitor",  # 경쟁사 분석
    "infringement",  # 침해 분석
    "invalidity",  # 무효 분석
]

# Workflow steps
AnalysisStep = Literal["collect", "process", "analyze", "visualize", "report"]

# Infringement types
InfringementType = Literal["literal", "doctrine_of_equivalents", "none"]


class AnalysisScope(TypedDict):
    """Scope definition for analysis."""

    date_range: tuple[str, str] | None  # (from_date, to_date)
    jurisdictions: list[str]  # KR, US, EP, etc.
    ipc_codes: list[str]
    applicants: list[str]
    keywords: list[str]


class PatentRecord(TypedDict):
    """Single patent record for analysis."""

    patent_number: str
    title: str
    applicant: str
    inventor: list[str]
    filing_date: str
    publication_date: str
    grant_date: str | None
    ipc_codes: list[str]
    claims: list[str]
    abstract: str
    legal_status: Literal["pending", "granted", "expired", "withdrawn", "abandoned"]
    citation_count: int
    family_size: int


class ClaimElement(TypedDict):
    """Single claim element for mapping."""

    element_id: str
    text: str
    type: Literal["preamble", "body", "limitation"]


class ClaimChart(TypedDict):
    """Claim chart for infringement/invalidity analysis."""

    claim_number: int
    claim_elements: list[ClaimElement]
    mappings: list[dict]  # [{element_id, corresponding_feature, match_type}]
    overall_result: InfringementType | Literal["novelty_destroying", "partial", "no_match"]


class VisualizationChart(TypedDict):
    """Generated visualization."""

    chart_type: Literal[
        "bar_chart",
        "line_chart",
        "heatmap",
        "network_graph",
        "bubble_chart",
        "treemap",
    ]
    title: str
    data: dict
    file_path: str | None
    format: Literal["png", "svg", "html"]


# ═══════════════════════════════════════════════════════════════
# Portfolio Analysis
# ═══════════════════════════════════════════════════════════════


class PortfolioMetrics(TypedDict):
    """Portfolio analysis metrics."""

    total_patents: int
    patents_by_year: dict[str, int]
    patents_by_jurisdiction: dict[str, int]
    patents_by_ipc: dict[str, int]
    average_citations: float
    average_family_size: float
    core_patents: list[str]  # Top N by score


class PortfolioReport(TypedDict):
    """Portfolio analysis report."""

    target: str  # Company/organization
    metrics: PortfolioMetrics
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]


# ═══════════════════════════════════════════════════════════════
# Trend Analysis
# ═══════════════════════════════════════════════════════════════


class TrendIndicators(TypedDict):
    """Technology trend indicators."""

    growth_rate: float  # 출원 증가율
    concentration: float  # 출원인 집중도
    technology_shift: str  # 기술 이동 방향
    emerging_areas: list[str]  # 신흥 기술 영역


class MaturityPhase(TypedDict):
    """Technology maturity assessment."""

    phase: Literal["emergence", "growth", "maturity", "decline"]
    phase_kr: str  # 태동기, 성장기, 성숙기, 쇠퇴기
    characteristics: str
    recommended_strategy: str


# ═══════════════════════════════════════════════════════════════
# Competitor Analysis
# ═══════════════════════════════════════════════════════════════


class CompetitorProfile(TypedDict):
    """Single competitor profile."""

    company_name: str
    patent_count: int
    market_share: float  # Patent share in the field
    key_technologies: list[str]
    core_patents: list[str]
    recent_trends: str


class CompetitorMetrics(TypedDict):
    """Competitor comparison metrics."""

    patent_share: dict[str, float]  # {company: share}
    technology_overlap: dict[str, float]  # {company: overlap_ratio}
    citation_network: dict  # Network structure
    litigation_history: list[dict]


# ═══════════════════════════════════════════════════════════════
# Infringement Analysis
# ═══════════════════════════════════════════════════════════════


class ProductSpec(TypedDict):
    """Product specification for infringement analysis."""

    product_name: str
    features: list[str]
    technical_description: str
    documentation: list[str]  # Paths to product docs


class InfringementAnalysis(TypedDict):
    """Infringement analysis result."""

    target_patent: str
    target_product: str
    claim_charts: list[ClaimChart]
    literal_infringement: bool
    doctrine_of_equivalents: bool
    overall_risk: Literal["high", "medium", "low", "none"]
    design_around_options: list[str]
    recommendations: list[str]


# ═══════════════════════════════════════════════════════════════
# Invalidity Analysis
# ═══════════════════════════════════════════════════════════════


class InvalidityGround(TypedDict):
    """Single invalidity ground."""

    ground_type: Literal["novelty", "inventive_step", "enablement", "written_description"]
    legal_basis: str  # 특허법 조항
    claim_numbers: list[int]
    prior_art_refs: list[str]
    argument: str
    evidence: list[dict]
    success_probability: ConfidenceLevel


class InvalidityAnalysis(TypedDict):
    """Invalidity analysis result."""

    target_patent: str
    grounds: list[InvalidityGround]
    strongest_ground: str | None
    overall_invalidity_likelihood: ConfidenceLevel
    recommendations: list[str]


# ═══════════════════════════════════════════════════════════════
# Main State
# ═══════════════════════════════════════════════════════════════


class PatentAnalysisState(BasePatentState, total=False):
    """
    State for Patent Analysis workflow.

    Analysis types:
    - portfolio: 특정 기업/기관의 특허 현황 파악
    - trend: 특정 기술분야의 출원/등록 트렌드
    - competitor: 경쟁사 특허 전략 파악
    - infringement: 제품 vs 특허 청구항 대비
    - invalidity: 특허 무효화 가능성 검토

    Steps:
    - collect: 특허 데이터 수집
    - process: 데이터 전처리
    - analyze: 분석 수행
    - visualize: 시각화 생성
    - report: 보고서 작성
    """

    # Override current_step with specific type
    current_step: AnalysisStep

    # ─────────────────────────────────────────────────────────────
    # Analysis Configuration
    # ─────────────────────────────────────────────────────────────
    analysis_type: AnalysisType
    target: str  # 분석 대상 (회사명, 기술분야, 특허번호 등)
    scope: AnalysisScope

    # ─────────────────────────────────────────────────────────────
    # Data Collection
    # ─────────────────────────────────────────────────────────────
    patent_data: list[PatentRecord]
    total_patents_collected: int
    data_sources: list[str]  # KIPRIS, USPTO, etc.

    # ─────────────────────────────────────────────────────────────
    # Analysis Results (type-specific)
    # ─────────────────────────────────────────────────────────────
    # Portfolio
    portfolio_report: PortfolioReport | None

    # Trend
    trend_indicators: TrendIndicators | None
    maturity_assessment: MaturityPhase | None

    # Competitor
    competitor_profiles: list[CompetitorProfile] | None
    competitor_metrics: CompetitorMetrics | None

    # Infringement
    product_spec: ProductSpec | None
    infringement_analysis: InfringementAnalysis | None

    # Invalidity
    invalidity_analysis: InvalidityAnalysis | None

    # ─────────────────────────────────────────────────────────────
    # Claim Mapping (for infringement/invalidity)
    # ─────────────────────────────────────────────────────────────
    claim_mappings: dict[int, ClaimChart]  # {claim_num: chart}

    # ─────────────────────────────────────────────────────────────
    # Visualizations
    # ─────────────────────────────────────────────────────────────
    visualizations: list[VisualizationChart]

    # ─────────────────────────────────────────────────────────────
    # Final Report
    # ─────────────────────────────────────────────────────────────
    executive_summary: str
    detailed_report: str
    appendices: list[str]  # Paths to appendix files


# Analysis type descriptions
ANALYSIS_TYPE_INFO = {
    "portfolio": {
        "name": "포트폴리오 분석",
        "purpose": "특정 기업/기관의 특허 현황 파악",
        "outputs": ["보유 현황", "기술 분포", "출원 트렌드", "핵심 특허"],
    },
    "trend": {
        "name": "기술 동향 분석",
        "purpose": "특정 기술분야의 출원/등록 트렌드",
        "outputs": ["시계열 분석", "주요 출원인", "기술 발전 방향"],
    },
    "competitor": {
        "name": "경쟁사 분석",
        "purpose": "경쟁사 특허 전략 파악",
        "outputs": ["특허 포지셔닝", "R&D 방향", "협력/경쟁 관계"],
    },
    "infringement": {
        "name": "침해 분석",
        "purpose": "제품 vs 특허 청구항 대비",
        "outputs": ["침해 가능성", "회피 설계", "라이선스 필요성"],
    },
    "invalidity": {
        "name": "무효 분석",
        "purpose": "특허 무효화 가능성 검토",
        "outputs": ["무효 논거", "선행기술 증거", "성공 가능성"],
    },
}

# Core patent scoring factors
CORE_PATENT_SCORING = {
    "citation_count": 0.25,
    "claim_breadth": 0.20,
    "family_size": 0.15,
    "remaining_life": 0.15,
    "technology_centrality": 0.15,
    "legal_stability": 0.10,
}
