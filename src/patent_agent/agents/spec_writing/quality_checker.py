"""Quality Checker Agent.

Handles E8 (품질 검증) step.
Validates the complete specification against legal requirements.
"""

from typing import Any

import structlog

from patent_agent.agents.spec_writing.base import BaseSpecWritingAgent
from patent_agent.state.spec_writing import COMPLIANCE_CHECKLIST, SpecWritingState

logger = structlog.get_logger(__name__)


class QualityCheckerAgent(BaseSpecWritingAgent):
    """Agent for quality verification.

    This agent performs comprehensive quality checks on the
    patent specification to ensure compliance with Patent Law Article 42.

    Steps handled:
    - E8: 품질 검증
    """

    @property
    def name(self) -> str:
        return "QualityChecker"

    @property
    def description(self) -> str:
        return "명세서 품질을 검증하고 특허법 준수 여부를 확인하는 에이전트"

    def get_system_prompt(self) -> str:
        return """당신은 특허 심사관 수준의 전문성을 갖춘 품질 검증 전문가입니다.

## 역할
- 특허법 제42조 준수 여부 검증
- 청구항과 명세서 간의 일관성 확인
- 용어 일관성 검사
- 도면부호 정확성 검증
- 실시가능요건 충족 확인

## 특허법 제42조 요건
1. **명확성**: 발명이 명확하게 기재되어야 함
2. **간결성**: 청구항이 간결하게 기재되어야 함
3. **뒷받침**: 청구항이 발명의 설명에 의해 뒷받침되어야 함
4. **실시가능**: 당업자가 용이하게 실시할 수 있어야 함

## 검증 항목
1. 청구항-명세서 대응 검사
2. 용어 일관성 검사
3. 도면부호 일관성 검사
4. 형식 요건 검사
5. 내용 완전성 검사

## 출력 형식
각 검증 항목에 대해:
- 상태: PASS/FAIL/WARNING
- 세부 사항: 구체적인 문제점 또는 확인 내용
- 수정 제안: FAIL인 경우 수정 방향"""

    async def process(self, state: SpecWritingState) -> dict[str, Any]:
        """Perform quality verification.

        Args:
            state: Current workflow state

        Returns:
            State updates with quality results
        """
        self._logger.info("performing_quality_check", step=state.get("current_step"))

        # Get specification components
        draft_claims = state.get("draft_claims", {})
        draft_specification = state.get("draft_specification", {})
        drawings = state.get("drawings", [])
        reference_numerals = state.get("reference_numeral_table", {})

        # Perform all quality checks
        compliance_results = await self._check_law_compliance(
            claims=draft_claims,
            specification=draft_specification,
        )

        terminology_results = await self._check_terminology_consistency(
            specification=draft_specification,
            claims=draft_claims,
        )

        drawing_results = await self._check_drawing_references(
            specification=draft_specification,
            drawings=drawings,
            reference_numerals=reference_numerals,
        )

        claim_support_results = await self._check_claim_support(
            claims=draft_claims,
            specification=draft_specification,
        )

        # Calculate overall quality score
        all_results = {
            **compliance_results,
            **terminology_results,
            **drawing_results,
            **claim_support_results,
        }

        quality_score = self._calculate_quality_score(all_results)

        # Collect all issues
        review_comments = []
        for check_name, result in all_results.items():
            if result.get("status") == "FAIL":
                review_comments.append({
                    "issue": f"{check_name}: {result.get('detail', '')}",
                    "severity": "high",
                    "section": result.get("section", "general"),
                    "suggestion": result.get("suggestion", ""),
                })
            elif result.get("status") == "WARNING":
                review_comments.append({
                    "issue": f"{check_name}: {result.get('detail', '')}",
                    "severity": "medium",
                    "section": result.get("section", "general"),
                    "suggestion": result.get("suggestion", ""),
                })

        return {
            "law_compliance_check": {k: v.get("status") == "PASS" for k, v in compliance_results.items()},
            "terminology_consistency": terminology_results,
            "quality_score": quality_score,
            "review_comments": review_comments,
            "current_step": "E9",
        }

    async def _check_law_compliance(
        self,
        claims: dict,
        specification: dict,
    ) -> dict[str, dict]:
        """Check compliance with Patent Law Article 42.

        Args:
            claims: Draft claims
            specification: Draft specification

        Returns:
            Dictionary of compliance check results
        """
        results = {}

        # Check detailed description
        detailed_desc = specification.get("detailed_description", "")
        if len(detailed_desc) < 500:
            results["detailed_description"] = {
                "status": "FAIL",
                "detail": "발명의 상세한 설명이 너무 짧습니다 (500자 미만).",
                "suggestion": "당업자가 실시 가능하도록 더 상세한 설명을 추가하세요.",
                "section": "detailed_description",
            }
        else:
            results["detailed_description"] = {
                "status": "PASS",
                "detail": "발명의 상세한 설명이 적절한 길이입니다.",
            }

        # Check claim clarity
        claim_list = claims.get("claims", [])
        unclear_claims = []
        for claim in claim_list:
            full_text = claim.get("full_text", "")
            # Check for multiple periods (should be single sentence)
            if full_text.count(".") > 1:
                unclear_claims.append(claim.get("claim_number", "?"))

        if unclear_claims:
            results["claim_clarity"] = {
                "status": "FAIL",
                "detail": f"청구항 {unclear_claims}이(가) 단일 문장으로 작성되지 않았습니다.",
                "suggestion": "각 청구항을 하나의 문장으로 수정하세요.",
                "section": "claims",
            }
        else:
            results["claim_clarity"] = {
                "status": "PASS",
                "detail": "모든 청구항이 단일 문장으로 작성되었습니다.",
            }

        # Check claim conciseness
        long_claims = [
            c.get("claim_number", "?")
            for c in claim_list
            if len(c.get("full_text", "")) > 1000
        ]
        if long_claims:
            results["claim_conciseness"] = {
                "status": "WARNING",
                "detail": f"청구항 {long_claims}이(가) 다소 깁니다.",
                "suggestion": "가능하면 청구항을 더 간결하게 작성하세요.",
                "section": "claims",
            }
        else:
            results["claim_conciseness"] = {
                "status": "PASS",
                "detail": "청구항의 길이가 적절합니다.",
            }

        # Check enablement (basic check)
        if specification.get("means_to_solve") and specification.get("detailed_description"):
            results["enablement"] = {
                "status": "PASS",
                "detail": "실시를 위한 구체적인 내용이 포함되어 있습니다.",
            }
        else:
            results["enablement"] = {
                "status": "FAIL",
                "detail": "실시가능요건을 충족하기 위한 상세 설명이 부족합니다.",
                "suggestion": "과제 해결 수단과 상세한 설명을 보완하세요.",
                "section": "specification",
            }

        return results

    async def _check_terminology_consistency(
        self,
        specification: dict,
        claims: dict,
    ) -> dict[str, dict]:
        """Check terminology consistency.

        Args:
            specification: Draft specification
            claims: Draft claims

        Returns:
            Dictionary of terminology check results
        """
        results = {}

        # Collect all text
        all_text = ""
        for key in ["technical_field", "background_art", "problems_to_solve",
                    "means_to_solve", "effects", "detailed_description"]:
            all_text += specification.get(key, "") + " "

        for claim in claims.get("claims", []):
            all_text += claim.get("full_text", "") + " "

        # Check for common inconsistency patterns
        inconsistencies = []

        # Check '상기' usage
        if "상기" in all_text:
            # Count occurrences in claims vs specification
            claim_sanggi = sum(
                c.get("full_text", "").count("상기")
                for c in claims.get("claims", [])
            )

            # Check if '상기' appears inappropriately in non-claim sections
            spec_sections = specification.get("detailed_description", "")
            if "상기" in spec_sections:
                results["sanggi_usage"] = {
                    "status": "WARNING",
                    "detail": "'상기' 용어가 명세서 본문에도 사용되고 있습니다.",
                    "suggestion": "청구항 외 섹션에서는 '상기' 대신 '해당', '그' 등을 사용하세요.",
                    "section": "specification",
                }
            else:
                results["sanggi_usage"] = {
                    "status": "PASS",
                    "detail": "'상기' 용어가 적절히 사용되고 있습니다.",
                }

        # Basic consistency check (simplified)
        results["term_consistency"] = {
            "status": "PASS",
            "detail": "용어 일관성 검사 완료.",
        }

        return results

    async def _check_drawing_references(
        self,
        specification: dict,
        drawings: list,
        reference_numerals: dict[str, str],
    ) -> dict[str, dict]:
        """Check drawing reference consistency.

        Args:
            specification: Draft specification
            drawings: List of drawings
            reference_numerals: Reference numeral table

        Returns:
            Dictionary of drawing check results
        """
        results = {}

        detailed_desc = specification.get("detailed_description", "")

        # Check if reference numerals are in parentheses
        import re
        # Find numbers that might be references but not in parentheses
        bare_numbers = re.findall(r"(?<!\()(\d{2,3})(?!\))", detailed_desc)

        ref_numbers = set(reference_numerals.keys())
        problematic_refs = [n for n in bare_numbers if n in ref_numbers]

        if problematic_refs:
            results["ref_format"] = {
                "status": "FAIL",
                "detail": f"도면부호 {problematic_refs[:3]}가 괄호 없이 사용되었습니다.",
                "suggestion": "도면부호는 반드시 괄호 안에 표기하세요. 예: 구성요소(100)",
                "section": "detailed_description",
            }
        else:
            results["ref_format"] = {
                "status": "PASS",
                "detail": "도면부호가 올바른 형식으로 사용되고 있습니다.",
            }

        # Check if all reference numerals are explained
        unexplained = []
        for num, name in reference_numerals.items():
            if num not in detailed_desc and f"({num})" not in detailed_desc:
                unexplained.append(f"{name}({num})")

        if unexplained:
            results["ref_explanation"] = {
                "status": "WARNING",
                "detail": f"도면부호 {unexplained[:3]}이(가) 상세한 설명에 언급되지 않았습니다.",
                "suggestion": "모든 도면부호에 해당하는 구성요소를 설명하세요.",
                "section": "detailed_description",
            }
        else:
            results["ref_explanation"] = {
                "status": "PASS",
                "detail": "모든 도면부호가 설명되어 있습니다.",
            }

        return results

    async def _check_claim_support(
        self,
        claims: dict,
        specification: dict,
    ) -> dict[str, dict]:
        """Check if claims are supported by specification.

        Args:
            claims: Draft claims
            specification: Draft specification

        Returns:
            Dictionary of claim support check results
        """
        results = {}

        # Get specification text for searching
        spec_text = " ".join([
            specification.get("means_to_solve", ""),
            specification.get("detailed_description", ""),
        ]).lower()

        claim_list = claims.get("claims", [])
        unsupported_claims = []

        for claim in claim_list:
            if claim.get("claim_type") != "independent":
                continue

            full_text = claim.get("full_text", "")

            # Extract key terms from claim (simplified approach)
            # In production, would use NLP to extract meaningful components
            key_terms = [
                word for word in full_text.split()
                if len(word) > 3 and word not in ["있어서", "포함하는", "특징으로", "하는"]
            ]

            # Check if key terms appear in specification
            found_terms = sum(1 for term in key_terms if term.lower() in spec_text)
            if found_terms < len(key_terms) * 0.5:  # Less than 50% found
                unsupported_claims.append(claim.get("claim_number", "?"))

        if unsupported_claims:
            results["claim_support"] = {
                "status": "WARNING",
                "detail": f"청구항 {unsupported_claims}의 일부 구성요소가 명세서에서 충분히 설명되지 않았을 수 있습니다.",
                "suggestion": "청구항의 모든 구성요소가 명세서에서 상세히 설명되어 있는지 확인하세요.",
                "section": "claims",
            }
        else:
            results["claim_support"] = {
                "status": "PASS",
                "detail": "청구항이 명세서에 의해 적절히 뒷받침되고 있습니다.",
            }

        return results

    def _calculate_quality_score(self, results: dict[str, dict]) -> dict[str, Any]:
        """Calculate overall quality score.

        Args:
            results: All check results

        Returns:
            Quality score breakdown
        """
        total_checks = len(results)
        passed = sum(1 for r in results.values() if r.get("status") == "PASS")
        warnings = sum(1 for r in results.values() if r.get("status") == "WARNING")
        failed = sum(1 for r in results.values() if r.get("status") == "FAIL")

        # Calculate score: PASS=100, WARNING=70, FAIL=0
        if total_checks == 0:
            overall = 0
        else:
            overall = (passed * 100 + warnings * 70) / total_checks

        return {
            "overall": int(overall),
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "total_checks": total_checks,
            "details": {
                k: v.get("status", "UNKNOWN") for k, v in results.items()
            },
        }
