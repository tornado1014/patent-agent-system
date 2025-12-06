"""
Guidelines for OA (Office Action) Response.

Based on PALLAS-EVIDENCE v3.0 - Zero Hallucination laws.
"""

import re
from typing import Any

from patent_agent.rules.base import (
    BaseGuideline,
    GuidelineViolation,
    ValidationResult,
    ViolationSeverity,
)


class OAResponseLaws(BaseGuideline):
    """
    Guidelines for OA response workflow.

    Zero Hallucination Laws:
    - LAW-6: 원문 불가침
    - LAW-7: 근거 추적성
    - LAW-8: 할루시네이션 차단
    - LAW-9: 신뢰도 명시
    - LAW-10: 검증 프로토콜
    """

    # Prohibited expressions that indicate hallucination
    PROHIBITED_EXPRESSIONS = [
        ("~로 보인다", "명확한 근거 제시로 대체"),
        ("~로 추측된다", "명확한 근거 제시로 대체"),
        ("일반적으로", "구체적 출처 명시로 대체"),
        ("통상적으로", "구체적 출처 명시로 대체"),
        ("당연히", "문서 인용으로 대체"),
        ("명백히", "문서 인용으로 대체"),
        ("~일 것이다", "확정적 표현 또는 불확실 명시"),
        ("것으로 예상된다", "확정적 표현 또는 불확실 명시"),
        ("아마도", "확정적 표현 또는 불확실 명시"),
    ]

    @property
    def domain(self) -> str:
        return "oa_response"

    @property
    def laws(self) -> dict[str, dict[str, Any]]:
        return {
            "LAW-6": {
                "name": "원문 불가침",
                "description": "청구항, 인용문헌, OA 의견은 한 글자도 수정 없이 원문 그대로 인용",
                "severity": ViolationSeverity.CRITICAL,
                "applies_to": ["청구항", "인용문헌 구절", "심사관 의견"],
            },
            "LAW-7": {
                "name": "근거 추적성",
                "description": "모든 분석과 주장은 [Doc:Page:Para] 또는 [Doc:청구항N] 형식으로 출처 명시",
                "severity": ViolationSeverity.HIGH,
                "format": "[문서명:p.X:para.Y] 또는 [문서명:청구항N]",
            },
            "LAW-8": {
                "name": "할루시네이션 차단",
                "description": "문서에 없는 기술적 특징, 효과, 해석 절대 창작 금지",
                "severity": ViolationSeverity.CRITICAL,
                "prohibited": self.PROHIBITED_EXPRESSIONS,
            },
            "LAW-9": {
                "name": "신뢰도 명시",
                "description": "모든 분석에 확실성 수준 표시: 🟢확실, 🟡가능, 🔴불확실",
                "severity": ViolationSeverity.MEDIUM,
                "levels": {
                    "🟢": "문서 직접 명시",
                    "🟡": "해석 여지 있음",
                    "🔴": "추가 검토 필요",
                },
            },
            "LAW-10": {
                "name": "검증 프로토콜",
                "description": "각 섹션 작성 후 할루시네이션 체크 및 원문 대조 수행",
                "severity": ViolationSeverity.HIGH,
                "checklist": [
                    "모든 청구항 원문 그대로 인용",
                    "모든 인용문헌 구절 정확 복사",
                    "심사관 의견 원문 보존",
                    "창작된 기술 특징 없음",
                    "과장된 효과 주장 없음",
                    "추측성 표현 제거",
                ],
            },
        }

    def get_system_prompt(self) -> str:
        return """## OA 대응 Zero Hallucination 법칙

### LAW-6: 원문 불가침 (CRITICAL)
- 청구항, 인용문헌, OA 의견은 **한 글자도 수정 없이** 원문 그대로 인용
- 인용 시 반드시 따옴표("")로 감싸서 표시
- 요약이나 의역 금지 - 원문 전체 인용 또는 인용하지 않음

### LAW-7: 근거 추적성 (HIGH)
모든 분석과 주장에는 반드시 출처를 명시:
- 형식: `[문서명:p.페이지:para.단락]` 또는 `[문서명:청구항N]`
- 예시:
  - `[출원서:p.15:para.3]`
  - `[D1:p.7:para.2]`
  - `[OA:청구항1]`

### LAW-8: 할루시네이션 차단 (CRITICAL)
**절대 금지 표현:**
- "~로 보인다" → 명확한 근거 제시로 대체
- "~로 추측된다" → 명확한 근거 제시로 대체
- "일반적으로" → 구체적 출처 명시로 대체
- "통상적으로" → 구체적 출처 명시로 대체
- "당연히" → 문서 인용으로 대체
- "명백히" → 문서 인용으로 대체
- "~일 것이다" → 확정적 표현 또는 불확실 명시

**문서에 없는 내용 창작 금지:**
- 문서에 명시되지 않은 기술적 특징 창작 금지
- 문서에 명시되지 않은 효과 주장 금지
- 문서에 근거 없는 해석 금지

### LAW-9: 신뢰도 명시 (MEDIUM)
모든 분석에 확실성 수준 표시:
- 🟢 **확실** — 문서에 직접 명시된 내용
- 🟡 **가능** — 해석의 여지가 있는 내용
- 🔴 **불확실** — 추가 검토가 필요한 내용

### LAW-10: 검증 프로토콜 (HIGH)
최종 출력 전 반드시 확인:
✓ 모든 청구항 원문 그대로 인용되었는가?
✓ 모든 인용문헌 구절이 정확히 복사되었는가?
✓ 심사관 의견이 원문 그대로 보존되었는가?
✓ 창작된 기술 특징이 없는가?
✓ 과장된 효과 주장이 없는가?
✓ 추측성 표현이 제거되었는가?

---
**경고: LAW-6, LAW-8 위반 시 출력이 즉시 거부됩니다.**
"문서에 없으면 존재하지 않는다" - 이 원칙을 항상 기억하십시오."""

    def validate_output(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        violations = []
        warnings = []
        context = context or {}

        # LAW-8: Check for prohibited expressions
        for expr, suggestion in self.PROHIBITED_EXPRESSIONS:
            if expr in output:
                violations.append(
                    GuidelineViolation(
                        law_id="LAW-8",
                        law_name="할루시네이션 차단",
                        severity=ViolationSeverity.CRITICAL,
                        description=f"금지 표현 '{expr}' 사용 감지",
                        original_text=expr,
                        suggestion=suggestion,
                    )
                )

        # LAW-7: Check for source citations (should have [Doc:...] format)
        if self._requires_citations(output, context):
            if not self._has_proper_citations(output):
                violations.append(
                    GuidelineViolation(
                        law_id="LAW-7",
                        law_name="근거 추적성",
                        severity=ViolationSeverity.HIGH,
                        description="분석 내용에 출처 표시가 없음",
                        suggestion="[문서명:p.X:para.Y] 형식으로 출처를 명시하세요.",
                    )
                )

        # LAW-9: Check for confidence indicators in analysis sections
        if context.get("section_type") == "analysis":
            if not self._has_confidence_indicators(output):
                warnings.append("분석 섹션에 신뢰도 표시(🟢🟡🔴)가 없습니다.")

        # LAW-6: Check quoted text integrity (if original texts provided)
        original_claims = context.get("original_claims", [])
        for claim in original_claims:
            if claim in output:
                # Check if properly quoted
                if f'"{claim}"' not in output and f"'{claim}'" not in output:
                    warnings.append(
                        f"청구항 인용 시 따옴표 사용을 권장합니다."
                    )

        return ValidationResult(
            is_valid=not any(v.severity == ViolationSeverity.CRITICAL for v in violations),
            violations=violations,
            warnings=warnings,
        )

    def _requires_citations(self, output: str, context: dict[str, Any]) -> bool:
        """Check if the output type requires citations."""
        section = context.get("section_type", "")
        # Analysis, rebuttal, and amendment sections require citations
        return section in ["analysis", "rebuttal", "amendment", "comparison"]

    def _has_proper_citations(self, text: str) -> bool:
        """Check if text has proper citation format."""
        # [Doc:p.X:para.Y] or [Doc:청구항N] format
        citation_pattern = r"\[[\w가-힣]+:(p\.\d+:para\.\d+|청구항\d+)\]"
        return bool(re.search(citation_pattern, text))

    def _has_confidence_indicators(self, text: str) -> bool:
        """Check if text has confidence level indicators."""
        indicators = ["🟢", "🟡", "🔴", "확실", "가능", "불확실"]
        return any(ind in text for ind in indicators)


# ═══════════════════════════════════════════════════════════════
# Amendment-specific Guidelines
# ═══════════════════════════════════════════════════════════════


class AmendmentGuideline(OAResponseLaws):
    """Specialized guidelines for amendment drafting."""

    def get_system_prompt(self) -> str:
        base = super().get_system_prompt()
        amendment_specific = """
## 보정안 작성 추가 지침

### 보정안 필수 요건
1. 최소 2개 이상의 보정안 제시 (LAW-3 준수)
2. 모든 추가 한정사항은 명세서에서 지원되어야 함
3. 신규사항 추가 금지

### 보정안 형식
```
━━━━━━━━━━━━━━━━━━━━━━━━
보정안 [A/B/C]: [보정 전략 명칭]
━━━━━━━━━━━━━━━━━━━━━━━━

[현재 청구항]
청구항 N. "[원문 전체 - 수정 불가]"

[보정 후 청구항]
청구항 N. "[보정된 전문]"

[보정 근거 - 명세서 매핑]
추가 한정 1: "[한정사항]"
└─ 근거: [명세서:p.X:para.Y] "[원문 인용]"
└─ 도면: [도면N:부호XXX] "[설명 인용]"

[신규사항 체크]
✅ 명세서 명시적 기재: YES/NO
✅ 도면 지원: YES/NO
⚠️ 신규사항 리스크: LOW/MEDIUM/HIGH

[보정 효과]
- 거절이유 해소: 🟢확실/🟡가능/🔴불확실
- 진보성 논리: "[명세서 기재 효과 인용]"
```

### 금지 사항
- 명세서에 없는 구성요소 추가
- 도면에 없는 특징 창작
- 효과의 과장 또는 창작
- 실시예 없는 조합 제안
"""
        return base + amendment_specific
