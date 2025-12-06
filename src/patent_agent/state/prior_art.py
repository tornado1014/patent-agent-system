"""
State definition for Prior Art Search workflow.

Multi-database search with relevance analysis and novelty/inventive step assessment.
"""

from typing import Literal, TypedDict

from patent_agent.state.base import BasePatentState, ConfidenceLevel


# Search types
SearchType = Literal[
    "novelty_search",  # 신규성 조사
    "inventive_step_search",  # 진보성 조사
    "freedom_to_operate",  # FTO 조사
    "invalidity_search",  # 무효자료 조사
]

# Workflow steps
PriorArtStep = Literal["query", "search", "filter", "analyze", "report"]

# Database identifiers
DatabaseID = Literal["KIPRIS", "USPTO", "EPO", "WIPO", "Google_Patents"]


class SearchQuery(TypedDict):
    """Search query for a database."""

    database: DatabaseID
    keywords: list[str]
    ipc_codes: list[str]
    cpc_codes: list[str]
    applicant: str | None
    date_range: tuple[str, str] | None  # (from_date, to_date)
    query_string: str  # Database-specific query


class PatentResult(TypedDict):
    """Single patent search result."""

    database: DatabaseID
    document_number: str
    title: str
    applicant: str
    inventor: list[str]
    filing_date: str
    publication_date: str
    grant_date: str | None
    ipc_codes: list[str]
    abstract: str
    claims_text: str | None
    pdf_url: str | None
    family_id: str | None


class AnalyzedReference(TypedDict):
    """Analyzed prior art reference."""

    reference_id: str  # Internal ID
    patent_result: PatentResult
    relevance_score: float  # 0.0 - 1.0
    matched_claims: list[int]  # Which claims of the invention it relates to
    key_features_matched: list[str]
    key_features_missing: list[str]
    differentiation: str  # 차별점 분석
    confidence: ConfidenceLevel


class ClaimComparison(TypedDict):
    """Comparison between invention claim and prior art."""

    claim_number: int
    claim_element: str  # 청구항 구성요소
    prior_art_ref: str  # 선행기술 문헌번호
    prior_art_disclosure: str  # 선행기술 개시 내용
    comparison_result: Literal["identical", "similar", "different", "not_found"]
    notes: str


class NoveltyAssessment(TypedDict):
    """Novelty assessment for a claim."""

    claim_number: int
    is_novel: bool
    blocking_reference: str | None  # 단일 문헌으로 신규성 부정 시
    assessment_reasoning: str
    confidence: ConfidenceLevel


class InventiveStepAssessment(TypedDict):
    """Inventive step assessment for a claim."""

    claim_number: int
    has_inventive_step: bool
    combination_references: list[str]  # 조합 문헌들
    combination_motivation: str  # 결합 동기
    technical_difficulty: str  # 기술적 곤란성
    unexpected_effect: str | None  # 예측 불가 효과
    confidence: ConfidenceLevel


class PriorArtSearchState(BasePatentState, total=False):
    """
    State for Prior Art Search workflow.

    Steps:
    - query: 검색 쿼리 설계
    - search: 다중 DB 병렬 검색
    - filter: 관련성 필터링
    - analyze: 청구항 대비 분석
    - report: 최종 보고서 생성

    Databases:
    - KIPRIS (한국)
    - USPTO (미국)
    - EPO (유럽)
    - WIPO Pearl
    - Google Patents
    """

    # Override current_step with specific type
    current_step: PriorArtStep

    # ─────────────────────────────────────────────────────────────
    # Search Configuration
    # ─────────────────────────────────────────────────────────────
    search_type: SearchType
    target_invention: str  # 조사 대상 발명 요약
    target_claims: list[str]  # 대상 청구항들
    target_filing_date: str | None  # 출원일 (선행기술 기준일)

    # ─────────────────────────────────────────────────────────────
    # Query Building
    # ─────────────────────────────────────────────────────────────
    search_keywords: list[str]
    expanded_keywords: list[str]  # 동의어, 상위/하위 개념 확장
    ipc_codes: list[str]
    cpc_codes: list[str]
    search_queries: dict[DatabaseID, SearchQuery]

    # ─────────────────────────────────────────────────────────────
    # Search Results
    # ─────────────────────────────────────────────────────────────
    raw_results: dict[DatabaseID, list[PatentResult]]
    total_results_count: dict[DatabaseID, int]
    search_errors: dict[DatabaseID, str | None]

    # ─────────────────────────────────────────────────────────────
    # Filtering & Analysis
    # ─────────────────────────────────────────────────────────────
    relevance_threshold: float  # Default 0.7
    filtered_results: list[PatentResult]
    analyzed_references: list[AnalyzedReference]

    # ─────────────────────────────────────────────────────────────
    # Claim Comparison
    # ─────────────────────────────────────────────────────────────
    claim_comparisons: list[ClaimComparison]
    comparison_table: str  # 대비표 (markdown)

    # ─────────────────────────────────────────────────────────────
    # Assessment
    # ─────────────────────────────────────────────────────────────
    novelty_assessments: list[NoveltyAssessment]
    inventive_step_assessments: list[InventiveStepAssessment]
    overall_novelty: bool
    overall_inventive_step: bool

    # ─────────────────────────────────────────────────────────────
    # Final Report
    # ─────────────────────────────────────────────────────────────
    executive_summary: str
    detailed_report: str
    key_references: list[str]  # Top N most relevant references
    recommendations: list[str]


# IPC Classification Guide
IPC_SECTIONS = {
    "A": "생활필수품",
    "B": "처리 조작; 운수",
    "C": "화학; 야금",
    "D": "섬유; 지류",
    "E": "고정 구조물",
    "F": "기계 공학; 조명; 가열; 무기; 폭파",
    "G": "물리학",
    "H": "전기",
}

# Relevance scoring factors
RELEVANCE_SCORING = {
    "technical_overlap": 0.4,  # 기술적 특징 중복도
    "claim_coverage": 0.3,  # 청구항 구성요소 포함도
    "temporal_relevance": 0.15,  # 시간적 관련성
    "applicant_relevance": 0.15,  # 출원인/발명자 관련성
}
