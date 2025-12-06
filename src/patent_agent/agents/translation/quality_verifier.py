"""Quality Verifier Agent (P5) for Patent Translation workflow.

Handles quality verification:
- 5C Quality Criteria assessment
- Terminology consistency check
- Cross-reference validation
- Law compliance verification
- TAC section priority enforcement
"""

import re
from typing import Any

from patent_agent.agents.translation.base import (
    BaseTranslationAgent,
    MistranslationPrevention,
)
from patent_agent.state.translation import (
    FIVE_C_CRITERIA,
    GlossaryEntry,
    QualityReport,
    RiskLevel,
    SectionType,
    TranslatedSection,
    TranslationPhase,
    TranslationState,
)


class QualityVerifierAgent(BaseTranslationAgent):
    """Quality verification agent.

    Phase P5: 품질 검증 (5C 기준)

    Responsibilities:
    - Assess translation quality using 5C criteria
    - Check terminology consistency across sections
    - Verify cross-references (claims ↔ description)
    - Validate law compliance
    - Flag issues for revision

    5C Criteria:
    - Correctness: 원문 대조 + 추론 근거 검증
    - Clarity: 명확성 룰 + 절대법칙 준수
    - Conciseness: 불필요 표현 제거
    - Consistency: 용어집 100% 일치
    - Compliance: 법규 준수 + 추론 완전성
    """

    @property
    def name(self) -> str:
        return "QualityVerifierAgent"

    @property
    def description(self) -> str:
        return "5C 품질 기준에 따라 번역 품질을 검증하는 에이전트"

    @property
    def phase(self) -> TranslationPhase:
        return "P5"

    def get_system_prompt(self) -> str:
        return """## Quality Verifier Agent (P5)

당신은 영→한 특허 번역의 품질을 검증하는 에이전트입니다.

### 5C 품질 기준

#### 1. Correctness (정확성) - 25%
- 원문과 번역문의 의미 일치 검증
- 기술적 정확성 확인
- 법률 용어 정확성 확인

#### 2. Clarity (명확성) - 25%
- 문장 구조 명확성
- 절대법칙 준수 여부
- TAC 섹션 특별 검증

#### 3. Conciseness (간결성) - 15%
- 불필요한 반복 제거
- 장황한 표현 개선
- 적절한 문장 길이

#### 4. Consistency (일관성) - 25%
- 용어 일관성 100% 달성
- 문체 일관성
- 도면부호 일관성

#### 5. Compliance (준수) - 10%
- 4대 절대법칙 준수
- KIPO 형식 준수
- 특허법 요건 준수

### 검증 프로세스
1. 각 섹션별 5C 점수 산출
2. 용어 일관성 교차 검증
3. 법칙 위반 사항 식별
4. 수정 필요 항목 플래그
5. 종합 품질 보고서 생성

### 합격 기준
- 전체 점수 80점 이상
- Critical 위반 0건
- TAC 섹션 85점 이상
"""

    async def process(self, state: TranslationState) -> dict[str, Any]:
        """Verify translation quality.

        Args:
            state: Current workflow state

        Returns:
            State updates with quality report
        """
        self._logger.info("starting_quality_verification")

        translated_sections = state.get("translated_sections", {})
        glossary = state.get("glossary", [])
        doc_structure = state.get("document_structure", {})

        if not translated_sections:
            self._logger.error("no_translations_to_verify")
            return {
                "is_error_state": True,
                "error_messages": ["검증할 번역이 없습니다."],
                "current_step": "P4",
            }

        # Step 1: Assess each criterion
        correctness_score = await self._assess_correctness(
            translated_sections, doc_structure
        )
        clarity_score = await self._assess_clarity(translated_sections)
        conciseness_score = await self._assess_conciseness(translated_sections)
        consistency_score, term_consistency = await self._assess_consistency(
            translated_sections, glossary
        )
        compliance_score = await self._assess_compliance(translated_sections)

        # Step 2: Calculate overall score (weighted)
        overall_score = int(
            correctness_score * 0.25
            + clarity_score * 0.25
            + conciseness_score * 0.15
            + consistency_score * 0.25
            + compliance_score * 0.10
        )

        # Step 3: Identify issues and suggestions
        issues, suggestions = await self._generate_feedback(
            translated_sections,
            {
                "correctness": correctness_score,
                "clarity": clarity_score,
                "conciseness": conciseness_score,
                "consistency": consistency_score,
                "compliance": compliance_score,
            },
        )

        # Step 4: Verify cross-references
        cross_ref_check = self._verify_cross_references(
            translated_sections, doc_structure
        )

        # Create quality report
        quality_report: QualityReport = {
            "correctness_score": correctness_score,
            "clarity_score": clarity_score,
            "conciseness_score": conciseness_score,
            "consistency_score": consistency_score,
            "compliance_score": compliance_score,
            "overall_score": overall_score,
            "issues": issues,
            "suggestions": suggestions,
        }

        # Create reasoning entry
        reasoning = self.create_reasoning_entry(
            checkpoint="CP5.1",
            decision="품질 검증 완료" if overall_score >= 80 else "품질 미달 - 수정 필요",
            reasoning=f"5C 점수: 정확성 {correctness_score}, 명확성 {clarity_score}, "
            f"간결성 {conciseness_score}, 일관성 {consistency_score}, "
            f"준수 {compliance_score} / 종합: {overall_score}점",
            applied_laws=["LAW-T-1", "LAW-T-2", "LAW-T-3", "LAW-T-4"],
        )

        # Risk flags for quality issues
        risk_flags = []
        if overall_score < 80:
            risk_flags.append(
                self.create_risk_flag(
                    level="High",
                    issue=f"품질 점수 미달 ({overall_score}/100)",
                    location="전체",
                    recommendation="식별된 문제 수정 후 재검증 필요",
                )
            )

        # TAC sections quality check
        for section_type in ["title", "abstract", "claims"]:
            if section_type in translated_sections:
                section_issues = self._get_section_issues(
                    translated_sections[section_type]
                )
                if section_issues:
                    risk_flags.append(
                        self.create_risk_flag(
                            level="Critical" if section_type == "claims" else "High",
                            issue=f"TAC 섹션 ({section_type}) 품질 문제",
                            location=section_type,
                            recommendation="; ".join(section_issues),
                        )
                    )

        self._logger.info(
            "quality_verification_complete",
            overall_score=overall_score,
            issues=len(issues),
            passed=overall_score >= 80,
        )

        return {
            "quality_report": quality_report,
            "terminology_consistency_check": term_consistency,
            "cross_reference_check": cross_ref_check,
            "current_step": "P6" if overall_score >= 80 else "P4",  # Back to P4 if failed
            "reasoning_log": state.get("reasoning_log", []) + [reasoning],
            "risk_flags": state.get("risk_flags", []) + risk_flags,
        }

    async def _assess_correctness(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
        doc_structure: dict,
    ) -> int:
        """Assess translation correctness.

        Args:
            translated_sections: Translated sections
            doc_structure: Original document structure

        Returns:
            Correctness score (0-100)
        """
        total_score = 0
        section_count = 0

        for section_type, section in translated_sections.items():
            source = section["source_text"]
            translated = section["translated_text"]

            # Use LLM to assess correctness
            prompt = f"""다음 영→한 특허 번역의 정확성을 평가하세요.

원문:
{source[:3000]}

번역문:
{translated[:3000]}

평가 기준:
1. 의미 일치도 (의미가 정확하게 전달되었는가)
2. 기술적 정확성 (기술 용어가 정확하게 번역되었는가)
3. 법률적 정확성 (법률 용어가 정확하게 번역되었는가)
4. 누락/추가 없음 (원문 내용이 빠짐없이 번역되었는가)

0-100 사이의 점수만 출력하세요:"""

            try:
                response = await self._invoke_llm(prompt)
                score = int(re.search(r'\d+', response).group())
                score = min(100, max(0, score))
            except Exception:
                score = 70  # Default score on error

            # Apply TAC multiplier (stricter for TAC sections)
            if self.is_tac_section(section_type):
                score = int(score * 0.95)  # 5% stricter

            total_score += score
            section_count += 1

        return total_score // section_count if section_count > 0 else 0

    async def _assess_clarity(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
    ) -> int:
        """Assess translation clarity.

        Args:
            translated_sections: Translated sections

        Returns:
            Clarity score (0-100)
        """
        scores = []

        for section_type, section in translated_sections.items():
            translated = section["translated_text"]
            law_compliance = section.get("law_compliance", {})

            # Base score
            score = 80

            # Deduct for law violations
            for law_id, is_compliant in law_compliance.items():
                if not is_compliant:
                    if self.is_tac_section(section_type):
                        score -= 15  # Higher penalty for TAC
                    else:
                        score -= 10

            # Check sentence clarity
            sentences = re.split(r'[.。]', translated)
            avg_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0

            # Optimal sentence length is 40-80 characters
            if avg_length > 150:
                score -= 10  # Too long
            elif avg_length < 20:
                score -= 5  # Too short

            scores.append(max(0, score))

        return sum(scores) // len(scores) if scores else 0

    async def _assess_conciseness(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
    ) -> int:
        """Assess translation conciseness.

        Args:
            translated_sections: Translated sections

        Returns:
            Conciseness score (0-100)
        """
        scores = []

        for section_type, section in translated_sections.items():
            source = section["source_text"]
            translated = section["translated_text"]

            # Compare lengths (Korean should be similar or slightly shorter)
            ratio = len(translated) / len(source) if source else 1

            if ratio > 1.3:
                score = 70  # Too verbose
            elif ratio > 1.1:
                score = 85
            else:
                score = 95

            # Check for redundant phrases
            redundant_patterns = [
                r"것이 가능하다",  # Can use simpler form
                r"하는 것이다",
                r"되어지다",  # Passive redundancy
                r"있는 것을",
            ]

            for pattern in redundant_patterns:
                if re.search(pattern, translated):
                    score -= 5

            scores.append(max(0, min(100, score)))

        return sum(scores) // len(scores) if scores else 0

    async def _assess_consistency(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
        glossary: list[GlossaryEntry],
    ) -> tuple[int, dict[str, list[str]]]:
        """Assess terminology consistency.

        Args:
            translated_sections: Translated sections
            glossary: Glossary entries

        Returns:
            Tuple of (consistency score, term consistency map)
        """
        term_consistency: dict[str, list[str]] = {}
        total_terms = 0
        consistent_terms = 0

        # Build glossary map
        glossary_map = {
            entry["english_term"].lower(): entry["korean_term"]
            for entry in glossary
        }

        # Check each section
        for section_type, section in translated_sections.items():
            source = section["source_text"].lower()
            translated = section["translated_text"]

            for eng_term, kor_term in glossary_map.items():
                if eng_term in source:
                    total_terms += 1
                    if kor_term in translated:
                        consistent_terms += 1
                        if eng_term not in term_consistency:
                            term_consistency[eng_term] = []
                        term_consistency[eng_term].append(f"{section_type}: {kor_term}")
                    else:
                        # Term not found - inconsistency
                        if eng_term not in term_consistency:
                            term_consistency[eng_term] = []
                        term_consistency[eng_term].append(
                            f"{section_type}: [MISSING] (expected: {kor_term})"
                        )

        score = (consistent_terms / total_terms * 100) if total_terms > 0 else 100

        return int(score), term_consistency

    async def _assess_compliance(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
    ) -> int:
        """Assess compliance with laws and regulations.

        Args:
            translated_sections: Translated sections

        Returns:
            Compliance score (0-100)
        """
        total_laws = 0
        compliant_laws = 0

        for section_type, section in translated_sections.items():
            law_compliance = section.get("law_compliance", {})

            for law_id, is_compliant in law_compliance.items():
                total_laws += 1
                if is_compliant:
                    compliant_laws += 1

        return int(compliant_laws / total_laws * 100) if total_laws > 0 else 100

    async def _generate_feedback(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
        scores: dict[str, int],
    ) -> tuple[list[str], list[str]]:
        """Generate issues and suggestions based on scores.

        Args:
            translated_sections: Translated sections
            scores: Score dictionary

        Returns:
            Tuple of (issues list, suggestions list)
        """
        issues = []
        suggestions = []

        # Correctness issues
        if scores["correctness"] < 80:
            issues.append("일부 섹션에서 번역 정확성이 부족합니다.")
            suggestions.append("원문과 번역문을 대조하여 의미 일치도를 확인하세요.")

        # Clarity issues
        if scores["clarity"] < 80:
            issues.append("문장 명확성이 부족하거나 절대법칙 위반이 있습니다.")
            suggestions.append("청구항 구조와 '상기' 사용 규칙을 재확인하세요.")

        # Conciseness issues
        if scores["conciseness"] < 80:
            issues.append("번역문이 불필요하게 장황합니다.")
            suggestions.append("'하는 것이다', '되어지다' 등 불필요한 표현을 제거하세요.")

        # Consistency issues
        if scores["consistency"] < 100:
            issues.append("용어 일관성이 100%에 미달합니다.")
            suggestions.append("용어집에 정의된 용어가 모든 섹션에서 일관되게 사용되는지 확인하세요.")

        # Compliance issues
        if scores["compliance"] < 100:
            issues.append("4대 절대법칙 중 일부가 위반되었습니다.")
            suggestions.append("LAW-T-1 ~ LAW-T-4 준수 여부를 다시 확인하세요.")

        # Check for mistranslations
        for section_type, section in translated_sections.items():
            source = section["source_text"]
            translated = section["translated_text"]

            mistranslations = MistranslationPrevention.check_for_mistranslation(
                source, translated
            )
            for m in mistranslations:
                issues.append(
                    f"오역 가능성: '{m['source_term']}' → '{m['wrong_translation']}' (권장: '{m['correct_translation']}')"
                )
                suggestions.append(f"'{m['source_term']}'의 번역을 수정하세요: {m['notes']}")

        return issues, suggestions

    def _verify_cross_references(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
        doc_structure: dict,
    ) -> bool:
        """Verify cross-references between sections.

        Args:
            translated_sections: Translated sections
            doc_structure: Document structure

        Returns:
            True if all cross-references are valid
        """
        # Check claim dependencies
        claim_deps = doc_structure.get("claim_dependencies", {})

        if "claims" in translated_sections:
            claims_text = translated_sections["claims"]["translated_text"]

            for claim_num, depends_on in claim_deps.items():
                if depends_on:
                    # Check that dependent claim reference exists
                    ref_pattern = rf"제{depends_on}항"
                    if not re.search(ref_pattern, claims_text):
                        return False

        # Check reference numerals are consistent
        ref_numerals = doc_structure.get("reference_numerals", {})

        for section_type, section in translated_sections.items():
            for numeral in ref_numerals.keys():
                # If numeral appears in source, should appear with brackets in translation
                if numeral in section["source_text"]:
                    if f"({numeral})" not in section["translated_text"]:
                        return False

        return True

    def _get_section_issues(self, section: TranslatedSection) -> list[str]:
        """Get list of issues for a specific section.

        Args:
            section: Translated section

        Returns:
            List of issue descriptions
        """
        issues = []

        # Check law compliance
        law_compliance = section.get("law_compliance", {})
        for law_id, is_compliant in law_compliance.items():
            if not is_compliant:
                issues.append(f"{law_id} 위반")

        # Check risk flags
        for flag in section.get("risk_flags", []):
            issues.append(flag["issue"])

        return issues
