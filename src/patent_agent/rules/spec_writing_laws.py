"""
Guidelines for Patent Specification Writing.

Based on PatentSpec-KR v2.0 and 특허법 제42조.
"""

import re
from typing import Any

from patent_agent.rules.base import (
    BaseGuideline,
    GuidelineViolation,
    ValidationResult,
    ViolationSeverity,
)


class SpecWritingLaws(BaseGuideline):
    """
    Guidelines for patent specification writing.

    Key Laws:
    - LAW-SW-1: 서술형 작성 강제 (청구항 외 목록형 금지)
    - LAW-SW-2: 특허법 제42조 완전 준수
    - LAW-SW-3: 청구항 형식 준수
    - LAW-SW-4: 용어 일관성 강제
    - LAW-SW-5: 도면부호 규칙
    """

    @property
    def domain(self) -> str:
        return "spec_writing"

    @property
    def laws(self) -> dict[str, dict[str, Any]]:
        return {
            "LAW-SW-1": {
                "name": "서술형 작성 강제",
                "description": "청구항을 제외한 모든 섹션은 서술형으로 작성. 불릿 포인트, 번호 목록 금지.",
                "severity": ViolationSeverity.HIGH,
                "applies_to": ["기술분야", "배경기술", "발명의 설명", "효과"],
                "examples": {
                    "wrong": "- 첫 번째 특징\n- 두 번째 특징",
                    "correct": "본 발명의 첫 번째 특징은 ... 이며, 두 번째 특징은 ... 이다.",
                },
            },
            "LAW-SW-2": {
                "name": "특허법 제42조 준수",
                "description": "발명의 설명은 당업자가 용이하게 실시할 수 있도록 명확하고 상세하게 기재",
                "severity": ViolationSeverity.CRITICAL,
                "requirements": [
                    "발명의 목적, 구성, 효과 명확 기재",
                    "실시예를 통한 구체적 설명",
                    "청구항의 명세서 지원",
                ],
            },
            "LAW-SW-3": {
                "name": "청구항 형식 준수",
                "description": "각 청구항은 단일 문장, 명확한 전제부와 특징부 구분",
                "severity": ViolationSeverity.CRITICAL,
                "format": {
                    "independent": "[전제부]에 있어서, [특징부]를 특징으로 하는 [발명의 명칭].",
                    "dependent": "제N항에 있어서, [추가 특징]을 특징으로 하는 [발명의 명칭].",
                },
            },
            "LAW-SW-4": {
                "name": "용어 일관성",
                "description": "동일한 구성요소는 명세서 전체에서 동일한 용어로 지칭",
                "severity": ViolationSeverity.MEDIUM,
                "examples": {
                    "wrong": "프로세서 → CPU → 연산장치 (혼용)",
                    "correct": "프로세서 (일관되게 사용)",
                },
            },
            "LAW-SW-5": {
                "name": "도면부호 규칙",
                "description": "도면부호는 괄호 안에 표기, 최초 등장 시 설명 필수",
                "severity": ViolationSeverity.MEDIUM,
                "format": "구성요소명(100)",
            },
        }

    def get_system_prompt(self) -> str:
        return """## 특허명세서 작성 절대 법칙

### LAW-SW-1: 서술형 작성 강제
- 청구항을 제외한 모든 섹션(기술분야, 배경기술, 발명의 설명, 효과)은 서술형으로 작성
- 불릿 포인트(-, •, *)나 번호 목록(1., 2., 3.) 사용 금지
- 모든 내용은 완전한 문장으로 자연스럽게 연결

### LAW-SW-2: 특허법 제42조 완전 준수
- 발명의 설명은 당업자가 용이하게 실시할 수 있도록 명확하고 상세하게 기재
- 발명의 목적, 구성, 효과를 명확히 기재
- 하나 이상의 실시예를 통해 구체적으로 설명
- 청구항은 반드시 명세서에 의해 뒷받침되어야 함

### LAW-SW-3: 청구항 형식 준수
독립항 형식:
```
[발명이 속하는 기술분야/전제부]에 있어서,
[발명의 구성/특징부]를 포함하는(또는 특징으로 하는)
[발명의 명칭].
```

종속항 형식:
```
제N항에 있어서,
[추가 구성/한정사항]을 포함하는(또는 특징으로 하는)
[발명의 명칭].
```

- 각 청구항은 반드시 단일 문장
- 마침표는 청구항 끝에만 1회

### LAW-SW-4: 용어 일관성 강제
- 동일한 구성요소는 명세서 전체에서 동일한 용어 사용
- 처음 등장 시 정의하고 이후 일관되게 사용
- 약어 사용 시 최초 1회 전체 명칭과 함께 표기

### LAW-SW-5: 도면부호 규칙
- 형식: 구성요소명(도면부호)
- 예: 프로세서(100), 메모리(200)
- 도면부호는 항상 괄호 안에 표기
- 최초 등장 시 해당 구성요소 설명 필수

---
위 법칙 위반 시 출력이 거부될 수 있습니다. 반드시 준수하십시오."""

    def validate_output(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        violations = []
        warnings = []
        context = context or {}

        section_type = context.get("section_type", "general")

        # LAW-SW-1: Check for list formatting in non-claim sections
        if section_type != "claims":
            if self._has_list_formatting(output):
                violations.append(
                    GuidelineViolation(
                        law_id="LAW-SW-1",
                        law_name="서술형 작성 강제",
                        severity=ViolationSeverity.HIGH,
                        description="청구항 외 섹션에서 목록 형식 사용 감지",
                        suggestion="불릿 포인트나 번호 목록 대신 서술형 문장으로 변환하세요.",
                    )
                )

        # LAW-SW-3: Claim format validation
        if section_type == "claims":
            claim_violations = self._validate_claim_format(output)
            violations.extend(claim_violations)

        # LAW-SW-5: Drawing reference format
        ref_violations = self._validate_drawing_references(output)
        if ref_violations:
            violations.extend(ref_violations)

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    def _has_list_formatting(self, text: str) -> bool:
        """Check if text contains list formatting."""
        # Bullet points
        if re.search(r"^[\s]*[-•*]\s", text, re.MULTILINE):
            return True
        # Numbered lists
        if re.search(r"^[\s]*\d+[.)]\s", text, re.MULTILINE):
            return True
        return False

    def _validate_claim_format(self, claim_text: str) -> list[GuidelineViolation]:
        """Validate claim format."""
        violations = []

        # Check for multiple periods (should have only one at the end)
        period_count = claim_text.count(".")
        if period_count > 1:
            violations.append(
                GuidelineViolation(
                    law_id="LAW-SW-3",
                    law_name="청구항 형식 준수",
                    severity=ViolationSeverity.CRITICAL,
                    description=f"청구항에 마침표가 {period_count}개 발견됨 (1개만 허용)",
                    suggestion="청구항은 단일 문장이어야 합니다. 마침표는 끝에만 위치해야 합니다.",
                )
            )

        # Check if claim ends with period
        if claim_text.strip() and not claim_text.strip().endswith("."):
            violations.append(
                GuidelineViolation(
                    law_id="LAW-SW-3",
                    law_name="청구항 형식 준수",
                    severity=ViolationSeverity.HIGH,
                    description="청구항이 마침표로 끝나지 않음",
                    suggestion="청구항은 반드시 마침표로 종결되어야 합니다.",
                )
            )

        return violations

    def _validate_drawing_references(self, text: str) -> list[GuidelineViolation]:
        """Validate drawing reference format."""
        violations = []

        # Find numbers that might be drawing references (not in parentheses)
        # Pattern: word followed by number not in parentheses
        potential_refs = re.findall(r"(\w+)\s+(\d+)(?!\))", text)

        for word, num in potential_refs:
            # Skip common non-reference patterns
            if word.lower() in ["제", "항", "도", "단계", "예", "약", "년"]:
                continue
            if int(num) > 1000:  # Probably not a drawing reference
                continue

            violations.append(
                GuidelineViolation(
                    law_id="LAW-SW-5",
                    law_name="도면부호 규칙",
                    severity=ViolationSeverity.MEDIUM,
                    description=f"도면부호 '{num}'가 괄호 없이 사용됨",
                    original_text=f"{word} {num}",
                    suggestion=f"{word}({num})로 수정",
                )
            )

        return violations


# ═══════════════════════════════════════════════════════════════
# Section-specific Guidelines
# ═══════════════════════════════════════════════════════════════


class ClaimWritingGuideline(SpecWritingLaws):
    """Specialized guidelines for claim writing."""

    def get_system_prompt(self) -> str:
        base = super().get_system_prompt()
        claim_specific = """
## 청구항 작성 추가 지침

### 독립항 작성 원칙
1. 발명의 필수 구성요소만 포함 (과도한 한정 금지)
2. 넓은 권리범위 확보를 위해 상위 개념 사용
3. 전제부에서 발명이 속하는 기술분야 명시
4. 특징부에서 발명의 핵심 구성 명확히 기재

### 종속항 작성 원칙
1. 독립항의 모든 구성요소를 포함
2. 추가적인 한정사항으로 구체화
3. 대안적 실시예 또는 바람직한 실시예 반영
4. 권리범위의 단계적 축소 (fall-back position)

### 청구항 용어 사용
- "포함하는" (comprising): 개방형, 추가 구성 허용
- "구성되는" (consisting of): 폐쇄형, 기재된 구성만
- "필수적으로 포함하는" (consisting essentially of): 반개방형
"""
        return base + claim_specific
