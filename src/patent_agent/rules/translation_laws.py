"""
Guidelines for Patent Translation (EN→KR).

Based on 영→한 특허번역 자동화 v3.0 - 4 Absolute Laws.
"""

import re
from typing import Any

from patent_agent.rules.base import (
    BaseGuideline,
    GuidelineViolation,
    ValidationResult,
    ViolationSeverity,
)


class TranslationLaws(BaseGuideline):
    """
    Guidelines for EN→KR patent translation.

    4 Absolute Laws:
    - LAW-T-1: '상기' 사용 규칙
    - LAW-T-2: 청구항 한 문장 원칙
    - LAW-T-3: 권리범위 결정 용어
    - LAW-T-4: 도면부호 괄호 필수
    """

    # Critical term mappings that must never be confused
    CRITICAL_TERMS = {
        "comprising": {
            "correct": "...을 포함하는",
            "wrong": ["구성된", "이루어진", "구성되는"],
            "scope": "개방형 (open-ended)",
        },
        "consisting of": {
            "correct": "...으로 구성된",
            "wrong": ["포함하는", "포함한"],
            "scope": "폐쇄형 (closed)",
        },
        "consisting essentially of": {
            "correct": "...을 필수적으로 포함하는",
            "wrong": ["포함하는", "구성된"],
            "scope": "반개방형",
        },
    }

    @property
    def domain(self) -> str:
        return "translation"

    @property
    def laws(self) -> dict[str, dict[str, Any]]:
        return {
            "LAW-T-1": {
                "name": "'상기' 사용 규칙",
                "description": "청구항: the/said → '상기', 청구항 외: '상기' 사용 금지",
                "severity": ViolationSeverity.CRITICAL,
                "claim_rule": "모든 'the', 'said' → '상기'로 번역",
                "non_claim_rule": "'상기' 완전 금지, '해당', '본', '이러한' 등 사용",
                "violation_risk": "권리범위 모호성",
            },
            "LAW-T-2": {
                "name": "청구항 한 문장 원칙",
                "description": "모든 청구항은 단일 문장, 마침표는 끝에만",
                "severity": ViolationSeverity.CRITICAL,
                "rule": "청구항 전체가 하나의 문장, 마침표 1개만 끝에",
                "violation_risk": "형식 위반",
            },
            "LAW-T-3": {
                "name": "권리범위 결정 용어",
                "description": "comprising ≠ 구성된, consisting of ≠ 포함하는",
                "severity": ViolationSeverity.CRITICAL,
                "terms": self.CRITICAL_TERMS,
                "violation_risk": "권리범위 변경",
            },
            "LAW-T-4": {
                "name": "도면부호 괄호 규칙",
                "description": "모든 도면부호는 괄호로 감쌈: 10 → (10)",
                "severity": ViolationSeverity.HIGH,
                "examples": {
                    "숫자만": "10 → (10)",
                    "문자+숫자": "10a → (10a)",
                    "복수": "10, 20 → (10), (20)",
                    "범위": "10-20 → (10) 내지 (20)",
                },
                "violation_risk": "불명확",
            },
        }

    def get_system_prompt(self) -> str:
        return """## 영→한 특허번역 절대법칙 (최고 우선순위)

### LAW-T-1: '상기' 사용 규칙 (CRITICAL)
**청구항 번역 시:**
- 모든 "the", "said" → "상기"로 번역
- 예: "the processor" → "상기 프로세서"
- 예: "said memory" → "상기 메모리"

**청구항 외 (명세서, 요약서 등) 번역 시:**
- "상기" 사용 **완전 금지**
- 대체어 사용: "해당", "본", "이러한", "상술한"
- 예: "the processor" → "해당 프로세서" 또는 "본 프로세서"

⚠️ 위반 시: 권리범위 모호성 발생

### LAW-T-2: 청구항 한 문장 원칙 (CRITICAL)
- 모든 청구항은 **단일 문장**으로 구성
- 마침표(.)는 청구항 **끝에만** 1회 사용
- 중간에 마침표 사용 금지 (세미콜론, 쉼표로 연결)

예시:
```
❌ 틀림: 프로세서를 포함한다. 상기 프로세서는 연산을 수행한다.
✅ 맞음: 프로세서를 포함하고, 상기 프로세서는 연산을 수행하는 장치.
```

### LAW-T-3: 권리범위 결정 용어 (CRITICAL)
**절대 혼용 금지 - 권리범위 결정:**

| 영문 | 한국어 | 권리범위 |
|------|--------|----------|
| comprising | ...을 포함하는 | 개방형 (추가 구성 허용) |
| consisting of | ...으로 구성된 | 폐쇄형 (기재된 것만) |
| consisting essentially of | ...을 필수적으로 포함하는 | 반개방형 |

⚠️ "comprising"을 "구성된"으로 번역하면 권리범위가 축소됨!
⚠️ "consisting of"를 "포함하는"으로 번역하면 권리범위가 확대됨!

### LAW-T-4: 도면부호 괄호 규칙 (HIGH)
**모든 도면부호는 반드시 괄호()로 감쌈 - 예외 없음**

| 유형 | 영문 | 한국어 |
|------|------|--------|
| 숫자만 | element 10 | 요소(10) |
| 문자+숫자 | part 10a | 부품(10a) |
| 복수 | 10, 20 | (10), (20) |
| 범위 | 10-20 | (10) 내지 (20) |

⚠️ 청구항 포함 모든 섹션에서 예외 없이 적용

---
**위 4가지 법칙은 번역의 모든 결정에서 최우선으로 적용됩니다.**
**CRITICAL 위반 시 번역 출력이 거부됩니다.**"""

    def validate_output(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        violations = []
        warnings = []
        auto_corrections = []
        context = context or {}

        section_type = context.get("section_type", "general")
        is_claim = section_type == "claims"

        # LAW-T-1: '상기' usage validation
        sanggi_violations = self._validate_sanggi_usage(output, is_claim)
        violations.extend(sanggi_violations)

        # LAW-T-2: Single sentence rule for claims
        if is_claim:
            sentence_violations = self._validate_single_sentence(output)
            violations.extend(sentence_violations)

        # LAW-T-3: Critical term validation
        term_violations = self._validate_critical_terms(output, context)
        violations.extend(term_violations)

        # LAW-T-4: Drawing reference format
        ref_violations, ref_corrections = self._validate_drawing_references(output)
        violations.extend(ref_violations)
        auto_corrections.extend(ref_corrections)

        return ValidationResult(
            is_valid=not any(v.severity == ViolationSeverity.CRITICAL for v in violations),
            violations=violations,
            warnings=warnings,
            auto_corrections=auto_corrections,
        )

    def _validate_sanggi_usage(self, text: str, is_claim: bool) -> list[GuidelineViolation]:
        """Validate '상기' usage based on section type."""
        violations = []
        sanggi_count = text.count("상기")

        if is_claim:
            # In claims, should have '상기' for 'the/said'
            # This is harder to validate without the source text
            # Just check that claims are not missing '상기' entirely
            if "the " in text.lower() or "said " in text.lower():
                warnings = []  # Original English still present
        else:
            # In non-claim sections, '상기' is forbidden
            if sanggi_count > 0:
                violations.append(
                    GuidelineViolation(
                        law_id="LAW-T-1",
                        law_name="'상기' 사용 규칙",
                        severity=ViolationSeverity.CRITICAL,
                        description=f"청구항 외 섹션에서 '상기' {sanggi_count}회 사용 감지",
                        suggestion="'해당', '본', '이러한' 등으로 대체하세요.",
                    )
                )

        return violations

    def _validate_single_sentence(self, claim_text: str) -> list[GuidelineViolation]:
        """Validate that claim is a single sentence."""
        violations = []

        # Count periods (should be exactly 1 at the end)
        period_count = claim_text.count(".")

        if period_count > 1:
            violations.append(
                GuidelineViolation(
                    law_id="LAW-T-2",
                    law_name="청구항 한 문장 원칙",
                    severity=ViolationSeverity.CRITICAL,
                    description=f"청구항에 마침표가 {period_count}개 발견됨",
                    suggestion="청구항은 단일 문장이어야 합니다. 세미콜론이나 쉼표로 연결하세요.",
                )
            )
        elif period_count == 1 and not claim_text.strip().endswith("."):
            violations.append(
                GuidelineViolation(
                    law_id="LAW-T-2",
                    law_name="청구항 한 문장 원칙",
                    severity=ViolationSeverity.HIGH,
                    description="마침표가 청구항 중간에 위치",
                    suggestion="마침표는 청구항 끝에만 위치해야 합니다.",
                )
            )

        return violations

    def _validate_critical_terms(
        self, text: str, context: dict[str, Any]
    ) -> list[GuidelineViolation]:
        """Validate critical term translations."""
        violations = []

        # Check for common mistranslations
        # "comprising" mistranslated as "구성된"
        if "comprising" in context.get("source_text", "").lower():
            if "구성된" in text and "포함하는" not in text:
                violations.append(
                    GuidelineViolation(
                        law_id="LAW-T-3",
                        law_name="권리범위 결정 용어",
                        severity=ViolationSeverity.CRITICAL,
                        description="'comprising'이 '구성된'으로 번역됨 (권리범위 축소)",
                        suggestion="'comprising'은 '...을 포함하는'으로 번역해야 합니다.",
                    )
                )

        # "consisting of" mistranslated as "포함하는"
        if "consisting of" in context.get("source_text", "").lower():
            if "포함하는" in text and "구성된" not in text:
                violations.append(
                    GuidelineViolation(
                        law_id="LAW-T-3",
                        law_name="권리범위 결정 용어",
                        severity=ViolationSeverity.CRITICAL,
                        description="'consisting of'가 '포함하는'으로 번역됨 (권리범위 확대)",
                        suggestion="'consisting of'는 '...으로 구성된'으로 번역해야 합니다.",
                    )
                )

        return violations

    def _validate_drawing_references(
        self, text: str
    ) -> tuple[list[GuidelineViolation], list[dict]]:
        """Validate drawing reference format and suggest corrections."""
        violations = []
        corrections = []

        # Find numbers that might be drawing references without parentheses
        # Pattern: Korean word followed by number not in parentheses
        pattern = r"([\uac00-\ud7af]+)\s*(\d+[a-zA-Z]?)(?!\))"
        matches = re.findall(pattern, text)

        for word, num in matches:
            # Skip common non-reference patterns
            if word in ["제", "항", "도", "단계", "약", "년", "월", "일", "페이지"]:
                continue
            # Skip large numbers (probably not references)
            num_value = int(re.match(r"\d+", num).group())
            if num_value > 999:
                continue

            violations.append(
                GuidelineViolation(
                    law_id="LAW-T-4",
                    law_name="도면부호 괄호 규칙",
                    severity=ViolationSeverity.HIGH,
                    description=f"도면부호 '{num}'가 괄호 없이 사용됨",
                    original_text=f"{word} {num}",
                    suggestion=f"{word}({num})",
                )
            )
            corrections.append({
                "original": f"{word} {num}",
                "corrected": f"{word}({num})",
            })

        return violations, corrections


# ═══════════════════════════════════════════════════════════════
# 5C Quality Criteria
# ═══════════════════════════════════════════════════════════════


class FiveCQualityCriteria:
    """5C Quality criteria for translation verification."""

    CRITERIA = {
        "Correctness": {
            "weight": 0.25,
            "checks": ["의미 정확성", "용어 정확성", "수치 정확성"],
            "description": "원문 대조 + 추론 근거 검증",
        },
        "Clarity": {
            "weight": 0.20,
            "checks": ["문장 명확성", "법칙 준수", "모호성 제거"],
            "description": "명확성 룰 + 절대법칙 준수",
        },
        "Conciseness": {
            "weight": 0.15,
            "checks": ["중복 제거", "간결한 표현"],
            "description": "불필요 표현 제거",
        },
        "Consistency": {
            "weight": 0.20,
            "checks": ["용어 일관성", "스타일 일관성"],
            "description": "용어집 100% 일치",
        },
        "Compliance": {
            "weight": 0.20,
            "checks": ["KIPO 형식", "특허법 준수", "추론 검증"],
            "description": "법규 준수 + 추론 완전성",
        },
    }

    @classmethod
    def calculate_score(cls, check_results: dict[str, list[bool]]) -> dict[str, int]:
        """Calculate 5C scores from check results."""
        scores = {}
        for criterion, config in cls.CRITERIA.items():
            if criterion in check_results:
                passed = sum(check_results[criterion])
                total = len(check_results[criterion])
                scores[criterion] = int((passed / total) * 100) if total > 0 else 0
            else:
                scores[criterion] = 0

        # Calculate weighted overall score
        overall = sum(
            scores.get(c, 0) * config["weight"]
            for c, config in cls.CRITERIA.items()
        )
        scores["overall"] = int(overall)

        return scores
