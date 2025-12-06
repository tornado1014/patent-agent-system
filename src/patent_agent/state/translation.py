"""
State definition for Patent Translation (EN→KR) workflow.

Based on 영→한 특허번역 자동화 v3.0 with 4 Absolute Laws.
"""

from typing import Literal, TypedDict

from patent_agent.state.base import BasePatentState


# Translation phases
TranslationPhase = Literal["P1", "P2", "P3", "P4", "P5", "P6"]

# Document sections
SectionType = Literal[
    "title",  # 발명의 명칭
    "field",  # 기술분야
    "background",  # 배경기술
    "summary",  # 발명의 내용
    "brief_description",  # 도면의 간단한 설명
    "detailed_description",  # 발명을 실시하기 위한 구체적인 내용
    "claims",  # 청구범위
    "abstract",  # 요약서
]

# Risk levels for translation decisions
RiskLevel = Literal["Critical", "High", "Medium", "Low"]


class SourceDocument(TypedDict):
    """Source document (English patent)."""

    file_path: str
    file_type: Literal["pdf", "docx", "txt"]
    total_pages: int
    requires_ocr: bool
    patent_number: str | None
    filing_date: str | None


class DocumentStructure(TypedDict):
    """Parsed document structure mapped to KIPO format."""

    sections: dict[SectionType, str]  # Section name -> content
    claims_count: int
    figures_count: int
    reference_numerals: dict[str, str]  # {numeral: description}
    claim_dependencies: dict[int, int | None]  # {claim_num: depends_on}


class GlossaryEntry(TypedDict):
    """Single glossary entry."""

    english_term: str
    korean_term: str
    source: Literal["KIPRIS", "WIPO_Pearl", "user", "auto"]
    domain: str | None
    notes: str | None


class RiskFlag(TypedDict):
    """Risk flag for translation decision."""

    level: RiskLevel
    issue: str
    location: str  # Section and position
    recommendation: str
    requires_human_review: bool


class ReasoningEntry(TypedDict):
    """Translation reasoning log entry."""

    phase: TranslationPhase
    checkpoint: str
    decision: str
    reasoning: str
    applied_laws: list[str]  # LAW-1, LAW-2, etc.
    timestamp: str


class QualityReport(TypedDict):
    """5C Quality assessment report."""

    correctness_score: int  # 0-100
    clarity_score: int
    conciseness_score: int
    consistency_score: int
    compliance_score: int
    overall_score: int
    issues: list[str]
    suggestions: list[str]


class TranslatedSection(TypedDict):
    """Single translated section."""

    section_type: SectionType
    source_text: str
    translated_text: str
    glossary_terms_used: list[str]
    risk_flags: list[RiskFlag]
    law_compliance: dict[str, bool]  # {law_id: compliant}


class TranslationState(BasePatentState, total=False):
    """
    State for Patent Translation (EN→KR) workflow.

    6-phase pipeline (P1-P6):
    - P1: 문서 접수 및 초기 추론
    - P2: 구조 분석 및 의존성 매핑
    - P3: 용어 일관성 구축 (번역 전 필수)
    - P4: 섹션별 스마트 번역
    - P5: 품질 검증 (5C 기준)
    - P6: 결과물 생성

    4 Absolute Laws (최고 우선순위):
    - LAW-1: '상기' 사용 규칙 (청구항: the/said→상기, 청구항 외: 상기 금지)
    - LAW-2: 청구항 한 문장 원칙 (마침표 끝에만)
    - LAW-3: 권리범위 결정 용어 (comprising≠구성된, consisting of≠포함하는)
    - LAW-4: 도면부호 괄호 필수 (10 → (10))
    """

    # Override current_step with specific type
    current_step: TranslationPhase

    # ─────────────────────────────────────────────────────────────
    # Phase 1: Document Intake
    # ─────────────────────────────────────────────────────────────
    source_document: SourceDocument
    ocr_performed: bool
    metadata_extracted: dict[str, str]
    legal_deadline: str | None  # 법정 제출기한

    # ─────────────────────────────────────────────────────────────
    # Phase 2: Structure Analysis
    # ─────────────────────────────────────────────────────────────
    document_structure: DocumentStructure
    missing_sections: list[SectionType]
    section_dependencies: dict[SectionType, list[SectionType]]

    # ─────────────────────────────────────────────────────────────
    # Phase 3: Glossary Building
    # ─────────────────────────────────────────────────────────────
    extracted_terms: list[str]  # 추출된 기술용어
    glossary: list[GlossaryEntry]
    glossary_conflicts: list[dict]  # 충돌 용어
    glossary_finalized: bool  # 용어집 확정 여부 (번역 전 필수)

    # ─────────────────────────────────────────────────────────────
    # Phase 4: Section Translation
    # ─────────────────────────────────────────────────────────────
    translated_sections: dict[SectionType, TranslatedSection]
    current_section: SectionType | None
    law_violations: list[dict]  # 절대법칙 위반 기록

    # ─────────────────────────────────────────────────────────────
    # Phase 5: Quality Verification
    # ─────────────────────────────────────────────────────────────
    quality_report: QualityReport
    terminology_consistency_check: dict[str, list[str]]  # {term: [occurrences]}
    cross_reference_check: bool

    # ─────────────────────────────────────────────────────────────
    # Phase 6: Output Generation
    # ─────────────────────────────────────────────────────────────
    final_korean_document: str
    comparison_document: str  # 영한 대조본
    output_files: dict[str, str]  # {filename: path}

    # ─────────────────────────────────────────────────────────────
    # Reasoning & Risk Tracking
    # ─────────────────────────────────────────────────────────────
    reasoning_log: list[ReasoningEntry]
    risk_flags: list[RiskFlag]
    critical_risks_resolved: bool


# ═══════════════════════════════════════════════════════════════
# ABSOLUTE LAWS (불변 규칙)
# ═══════════════════════════════════════════════════════════════

ABSOLUTE_LAWS = {
    "LAW-1": {
        "name": "'상기' 사용 규칙",
        "claims_rule": "모든 'the/said' → '상기'",
        "non_claims_rule": "'상기' 사용 완전 금지",
        "violation_risk": "권리범위 모호성",
    },
    "LAW-2": {
        "name": "청구항 한 문장 원칙",
        "rule": "모든 청구항은 단일 문장, 마침표는 끝에만",
        "violation_risk": "형식 위반",
    },
    "LAW-3": {
        "name": "권리범위 결정 용어",
        "comprising": "...을 포함하는 (개방형)",
        "consisting_of": "...으로 구성된 (폐쇄형)",
        "consisting_essentially_of": "...을 필수적으로 포함하는 (반개방형)",
        "violation_risk": "권리범위 변경",
    },
    "LAW-4": {
        "name": "도면부호 괄호 규칙",
        "rule": "모든 도면부호는 괄호로 감쌈: 10 → (10)",
        "examples": {"10": "(10)", "10a": "(10a)", "10-20": "(10) 내지 (20)"},
        "violation_risk": "불명확",
    },
}

# Standard legal term mappings
LEGAL_TERM_MAPPINGS = {
    "comprising": "...을 포함하는",
    "consisting of": "...으로 구성된",
    "consisting essentially of": "...을 필수적으로 포함하는",
    "wherein": "...하는",
    "characterized in that": "...것을 특징으로 하는",
    "embodiment": "실시예",
    "aspect": "양태",
    "invention": "발명",
    "prior art": "선행기술",
    "means for": "...하기 위한 수단",
    "configured to": "...하도록 구성된",
    "adapted to": "...에 적합한",
    "operatively connected": "작동 가능하게 연결된",
}

# 5C Quality Criteria
FIVE_C_CRITERIA = {
    "Correctness": "원문 대조 + 추론 근거 검증",
    "Clarity": "명확성 룰 + 절대법칙 준수",
    "Conciseness": "불필요 표현 제거",
    "Consistency": "용어집 100% 일치",
    "Compliance": "법규 준수 + 추론 완전성",
}

# Output file templates
OUTPUT_FILES = {
    "ko_final.docx": "KIPO 제출용 최종본",
    "en_ko_compare.docx": "영한 대조본",
    "qa_report.md": "품질 검증 보고서",
    "reasoning_log.md": "추론 이력 보고서",
    "glossary.xlsx": "사용된 용어집",
    "deadline_reminder.txt": "제출기한 안내",
}
