"""
State definition for Patent Specification Writing workflow.

Based on PatentSpec-KR v2.0 - 9-step workflow (E1-E9).
"""

from typing import Literal, TypedDict

from patent_agent.state.base import BasePatentState


# Workflow steps
SpecWritingStep = Literal["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]

# Human checkpoint steps
HUMAN_CHECKPOINTS: list[SpecWritingStep] = ["E4", "E6", "E9"]


class PriorArt(TypedDict):
    """Prior art reference."""

    document_number: str
    title: str
    applicant: str
    filing_date: str
    relevance: str  # How it relates to the invention
    limitations: str  # What problems it doesn't solve


class Drawing(TypedDict):
    """Drawing/figure information."""

    figure_number: int
    title: str
    description: str
    svg_path: str | None  # Path to generated SVG
    reference_numerals: dict[str, str]  # {numeral: description}


class Claim(TypedDict):
    """Patent claim structure."""

    claim_number: int
    claim_type: Literal["independent", "dependent"]
    depends_on: int | None  # For dependent claims
    preamble: str
    body: str
    full_text: str


class ClaimSet(TypedDict):
    """Complete set of claims."""

    claims: list[Claim]
    total_independent: int
    total_dependent: int


class Specification(TypedDict):
    """Patent specification structure."""

    title: str  # 발명의 명칭
    technical_field: str  # 기술분야
    background_art: str  # 배경기술
    problems_to_solve: str  # 해결하려는 과제
    means_to_solve: str  # 과제의 해결 수단
    effects: str  # 발명의 효과
    brief_description_drawings: str  # 도면의 간단한 설명
    detailed_description: str  # 발명을 실시하기 위한 구체적인 내용
    industrial_applicability: str | None  # 산업상 이용가능성


class SpecWritingState(BasePatentState, total=False):
    """
    State for Patent Specification Writing workflow.

    9-step workflow (E1-E9):
    - E1: 발명 개요 수집
    - E2: 핵심 정보 상세 질문
    - E3: 선행기술 분석
    - E4: 청구항 초안 작성 [Human Checkpoint]
    - E5: 명세서 구조 설계
    - E6: 본문 작성 [Human Checkpoint]
    - E7: 도면 생성 및 설명
    - E8: 품질 검증
    - E9: 최종 완성 [Human Checkpoint]
    """

    # Override current_step with specific type
    current_step: SpecWritingStep

    # ─────────────────────────────────────────────────────────────
    # Input Data (E1-E2)
    # ─────────────────────────────────────────────────────────────
    invention_disclosure: str  # 발명 신고서 원문
    tech_field: str  # 기술분야 (IPC/CPC)
    inventor_name: str
    applicant_name: str

    # Extracted invention details
    invention_title: str
    core_technology: str  # 핵심 기술
    key_features: list[str]  # 주요 특징
    technical_problems: list[str]  # 기술적 과제
    solutions: list[str]  # 해결 수단
    expected_effects: list[str]  # 기대 효과

    # ─────────────────────────────────────────────────────────────
    # Prior Art Analysis (E3)
    # ─────────────────────────────────────────────────────────────
    prior_arts: list[PriorArt]
    differentiation_points: list[str]  # 차별점

    # ─────────────────────────────────────────────────────────────
    # Claims (E4)
    # ─────────────────────────────────────────────────────────────
    draft_claims: ClaimSet
    claim_strategy: str  # 청구항 전략 설명

    # ─────────────────────────────────────────────────────────────
    # Specification (E5-E6)
    # ─────────────────────────────────────────────────────────────
    draft_specification: Specification
    embodiments: list[str]  # 실시예들

    # ─────────────────────────────────────────────────────────────
    # Drawings (E7)
    # ─────────────────────────────────────────────────────────────
    drawings: list[Drawing]
    reference_numeral_table: dict[str, str]  # 통합 도면부호 테이블

    # ─────────────────────────────────────────────────────────────
    # Quality Verification (E8)
    # ─────────────────────────────────────────────────────────────
    law_compliance_check: dict[str, bool]  # 특허법 42조 준수 체크
    terminology_consistency: dict[str, list[str]]  # 용어 일관성 체크

    # ─────────────────────────────────────────────────────────────
    # Final Output (E9)
    # ─────────────────────────────────────────────────────────────
    final_claims: ClaimSet
    final_specification: Specification
    abstract: str  # 요약서


# Law compliance checklist (특허법 제42조)
COMPLIANCE_CHECKLIST = {
    "detailed_description": "발명의 설명에 발명의 실시를 위한 구체적인 내용이 기재되어 있는가?",
    "claim_support": "청구항이 발명의 설명에 의해 뒷받침되는가?",
    "claim_clarity": "청구항이 명확하게 기재되어 있는가?",
    "claim_conciseness": "청구항이 간결하게 기재되어 있는가?",
    "enablement": "당업자가 용이하게 실시할 수 있도록 기재되어 있는가?",
    "best_mode": "최선의 실시 형태가 기재되어 있는가?",
}
