"""
State definition for OA (Office Action) Response workflow.

Based on PALLAS-EVIDENCE v3.0 - 5-phase workflow with Zero Hallucination.
"""

from typing import Literal, TypedDict

from patent_agent.state.base import BasePatentState, ConfidenceLevel, Evidence


# Workflow phases
OAPhase = Literal["P1", "P2", "P3", "P4", "P5"]

# Rejection types
RejectionType = Literal[
    "novelty",  # 신규성 결여 (제29조 제1항)
    "inventive_step",  # 진보성 결여 (제29조 제2항)
    "enablement",  # 실시가능성 위반 (제42조 제3항)
    "written_description",  # 기재불비 (제42조 제4항)
    "claim_clarity",  # 청구항 불명확 (제42조 제4항 제2호)
    "unity",  # 발명의 단일성 위반 (제45조)
    "other",
]


class ApplicationInfo(TypedDict):
    """Patent application information."""

    application_number: str
    filing_date: str
    title: str
    applicant: str
    inventor: str
    agent: str | None
    response_deadline: str  # 의견서 제출기한


class OADocument(TypedDict):
    """Office Action document."""

    oa_number: str
    issue_date: str
    examiner: str
    rejection_reasons: list[str]  # 거절이유 원문
    cited_claims: list[int]  # 거절된 청구항 번호
    total_pages: int


class CitedReference(TypedDict):
    """Cited reference (D1, D2, etc.)."""

    reference_id: str  # D1, D2, etc.
    document_number: str
    title: str
    applicant: str
    filing_date: str
    publication_date: str
    relevant_passages: list[str]  # 인용된 구절 (원문 그대로)
    total_pages: int


class RejectionAnalysis(TypedDict):
    """Analysis of a single rejection reason."""

    rejection_type: RejectionType
    legal_basis: str  # e.g., "특허법 제29조 제2항"
    examiner_argument: str  # 심사관 주장 원문 (LAW-6)
    cited_references: list[str]  # D1, D2 등
    affected_claims: list[int]
    confidence: ConfidenceLevel
    analysis_notes: str


class RebuttalPoint(TypedDict):
    """Rebuttal argument point."""

    target_rejection: str  # Which rejection this rebuts
    argument_type: Literal["technical_difference", "unexpected_effect", "motivation_lacking"]
    argument: str
    evidence: list[Evidence]
    confidence: ConfidenceLevel


class Amendment(TypedDict):
    """Proposed claim amendment."""

    amendment_id: str  # A, B, C, etc.
    strategy_name: str  # 보정 전략 명칭
    original_claim: str  # 원문 (수정 불가 - LAW-6)
    amended_claim: str  # 보정 후
    added_limitations: list[str]  # 추가된 한정사항
    specification_support: list[Evidence]  # 명세서 지원 근거
    new_matter_risk: Literal["low", "medium", "high"]
    effectiveness: ConfidenceLevel  # 거절이유 해소 가능성


class VerificationStamp(TypedDict):
    """Final verification stamp (LAW-10)."""

    hallucination_check: Literal["CLEAR", "WARNING", "FAILED"]
    verbatim_accuracy: float  # 원문 정확도 (%)
    citation_verified: int  # 검증된 인용 수
    citation_total: int  # 총 인용 수
    spec_support_verified: bool
    legal_basis_verified: bool
    timestamp: str  # ISO-8601


class OAResponseState(BasePatentState, total=False):
    """
    State for OA Response workflow.

    5-phase workflow (P1-P5):
    - P1: 문서 수집 및 검증 (원문 보존 추출)
    - P2: 거절이유 분석 (3층 검증)
    - P3: 비판적 검토 (근거 체인 구축)
    - P4: 보정안 생성 (명세서 지원 검증)
    - P5: 최종 보고서 (할루시네이션 체크)

    Zero Hallucination Laws:
    - LAW-6: 원문 불가침
    - LAW-7: 근거 추적성 [Doc:Page:Para]
    - LAW-8: 할루시네이션 차단
    - LAW-9: 신뢰도 명시
    - LAW-10: 검증 프로토콜
    """

    # Override current_step with specific type
    current_step: OAPhase

    # ─────────────────────────────────────────────────────────────
    # Document Collection (P1)
    # ─────────────────────────────────────────────────────────────
    application_info: ApplicationInfo
    oa_document: OADocument
    cited_references: list[CitedReference]  # D1, D2, etc.
    specification_text: str  # 출원 명세서 원문
    original_claims: list[str]  # 원 청구항들 (원문)

    # Document verification
    documents_verified: dict[str, bool]  # {doc_name: verified}
    total_pages_by_doc: dict[str, int]

    # ─────────────────────────────────────────────────────────────
    # Rejection Analysis (P2)
    # ─────────────────────────────────────────────────────────────
    rejection_analyses: list[RejectionAnalysis]
    claim_reference_mapping: dict[int, list[str]]  # {claim_num: [D1, D2]}

    # 3-layer verification
    extraction_verified: bool  # Layer 1
    mapping_verified: bool  # Layer 2
    completeness_verified: bool  # Layer 3

    # ─────────────────────────────────────────────────────────────
    # Critical Review (P3)
    # ─────────────────────────────────────────────────────────────
    rebuttal_points: list[RebuttalPoint]
    technical_differences: list[dict]  # 기술적 차이점
    logical_gaps: list[str]  # 논리적 간극

    # ─────────────────────────────────────────────────────────────
    # Amendment Generation (P4)
    # ─────────────────────────────────────────────────────────────
    amendment_options: list[Amendment]  # 최소 2개 필수 (LAW-3)
    recommended_amendment: str | None  # 권장 보정안 ID

    # ─────────────────────────────────────────────────────────────
    # Final Report (P5)
    # ─────────────────────────────────────────────────────────────
    draft_review_report: str  # 검토보고서 초안
    draft_opinion_letter: str  # 의견서 초안
    verification_stamp: VerificationStamp

    # Confidence scores per section
    confidence_by_section: dict[str, ConfidenceLevel]


# Zero Hallucination prohibited expressions
PROHIBITED_EXPRESSIONS = [
    "~로 보인다",
    "~로 추측된다",
    "일반적으로",
    "통상적으로",
    "당연히",
    "명백히",
    "~일 것이다",
    "~할 수 있다",  # 추측적 용법
]

# Required verification for final output
FINAL_VERIFICATION_CHECKLIST = [
    "모든 청구항 원문 그대로 인용",
    "모든 인용문헌 구절 정확 복사",
    "심사관 의견 원문 보존",
    "창작된 기술 특징 없음",
    "과장된 효과 주장 없음",
    "추측성 표현 제거",
]
