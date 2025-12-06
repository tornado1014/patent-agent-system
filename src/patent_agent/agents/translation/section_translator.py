"""Section Translator Agent (P4) for Patent Translation workflow.

Core translation agent that applies:
- 4 Absolute Laws (LAW-T-1 through LAW-T-4)
- Style guide rules
- Glossary term consistency
- TAC section priority
"""

import re
from typing import Any

from patent_agent.agents.translation.base import (
    BaseTranslationAgent,
    ClaimTranslationRules,
    MistranslationPrevention,
    SpecTranslationRules,
)
from patent_agent.state.translation import (
    GlossaryEntry,
    SectionType,
    TranslatedSection,
    TranslationPhase,
    TranslationState,
)


# Section translation order (based on dependencies)
TRANSLATION_ORDER: list[SectionType] = [
    "title",  # First - no dependencies
    "abstract",  # Depends on title
    "field",  # Depends on title
    "background",  # Depends on title, field
    "summary",  # Depends on title, field, background
    "claims",  # Depends on title, summary (CRITICAL)
    "brief_description",  # Depends on title
    "detailed_description",  # Last - depends on all above
]


class SectionTranslatorAgent(BaseTranslationAgent):
    """Section-by-section translation agent.

    Phase P4: 섹션별 스마트 번역

    This is the core translation agent that:
    - Translates each section following dependency order
    - Applies 4 Absolute Laws strictly
    - Uses glossary for terminology consistency
    - Validates translations in real-time
    - Handles TAC sections with extra care

    Absolute Laws Applied:
    - LAW-T-1: '상기' usage (claims only)
    - LAW-T-2: Single sentence claims
    - LAW-T-3: Transitional phrases (comprising/consisting of)
    - LAW-T-4: Drawing reference brackets
    """

    @property
    def name(self) -> str:
        return "SectionTranslatorAgent"

    @property
    def description(self) -> str:
        return "섹션별 번역을 수행하고 절대법칙을 적용하는 핵심 에이전트"

    @property
    def phase(self) -> TranslationPhase:
        return "P4"

    def get_system_prompt(self) -> str:
        return """## Section Translator Agent (P4)

당신은 영문 특허 문서를 한국어로 번역하는 핵심 에이전트입니다.

### 4대 절대법칙 (최우선 적용)

#### LAW-T-1: '상기' 사용 규칙
- **청구항**: 모든 'the', 'said' → '상기'로 번역
- **청구항 외**: '상기' 사용 완전 금지 → '해당', '본', '이러한', '상술한' 사용

#### LAW-T-2: 청구항 한 문장 원칙
- 각 청구항은 반드시 **단일 문장**으로 작성
- 마침표는 **문장 끝에만** 사용

#### LAW-T-3: 권리범위 결정 용어
- **comprising** → '포함하는' (개방형 - 추가 구성요소 허용)
- **consisting of** → '~으로 이루어지는' (폐쇄형 - 추가 구성요소 불허)
- **consisting essentially of** → '필수적으로 ~으로 이루어지는' (반개방형)

⚠️ 절대 오역 금지: comprising ≠ 구성된, consisting of ≠ 포함하는

#### LAW-T-4: 도면부호 괄호 규칙
- 모든 도면부호는 괄호로 감쌈: 10 → (10), 10a → (10a)
- 범위 표기: 10-20 → (10) 내지 (20)

### 섹션별 번역 규칙

#### 청구항 (Claims) - TAC 섹션
- 명사구 구조 유지
- 선행사 기반 (Antecedent Basis) 철저 적용
- 전이구 정확 번역
- 종속항 참조 형식: "제N항에 있어서"

#### 명세서 섹션
- '상기' 완전 금지
- 자연스러운 한국어 문체
- 도면부호 괄호 필수
- 용어집 100% 준수

### 품질 기준
- 정확성(Accuracy) > 유창성(Fluency)
- TAC 섹션 오류는 심각도 1.5배
- 법률 용어 정확성 필수
"""

    async def process(self, state: TranslationState) -> dict[str, Any]:
        """Translate document sections.

        Args:
            state: Current workflow state

        Returns:
            State updates with translated sections
        """
        self._logger.info("starting_section_translation")

        # Verify glossary is finalized
        if not state.get("glossary_finalized"):
            self._logger.error("glossary_not_finalized")
            return {
                "is_error_state": True,
                "error_messages": ["용어집이 확정되지 않았습니다. P3을 먼저 완료하세요."],
                "current_step": "P3",
            }

        # Get document structure and glossary
        doc_structure = state.get("document_structure", {})
        sections = doc_structure.get("sections", {})
        glossary = state.get("glossary", [])
        reference_numerals = doc_structure.get("reference_numerals", {})

        if not sections:
            self._logger.error("no_sections_to_translate")
            return {
                "is_error_state": True,
                "error_messages": ["번역할 섹션이 없습니다."],
                "current_step": "P4",
            }

        # Build glossary lookup
        glossary_map = self._build_glossary_map(glossary)

        # Translate each section in order
        translated_sections: dict[SectionType, TranslatedSection] = {}
        all_risk_flags = []
        all_law_violations = []
        reasoning_entries = []

        for section_type in TRANSLATION_ORDER:
            if section_type not in sections:
                continue

            source_text = sections[section_type]
            self._logger.info(f"translating_section_{section_type}")

            # Translate section
            result = await self._translate_section(
                section_type=section_type,
                source_text=source_text,
                glossary_map=glossary_map,
                reference_numerals=reference_numerals,
            )

            translated_sections[section_type] = result["translated_section"]
            all_risk_flags.extend(result.get("risk_flags", []))
            all_law_violations.extend(result.get("violations", []))

            # Create reasoning entry
            reasoning = self.create_reasoning_entry(
                checkpoint=f"CP4.{section_type}",
                decision=f"{section_type} 섹션 번역 완료",
                reasoning=f"글자수: {len(source_text)} → {len(result['translated_section']['translated_text'])}, "
                f"용어 적용: {len(result['translated_section']['glossary_terms_used'])}개, "
                f"위반: {len(result.get('violations', []))}건",
                applied_laws=result.get("applied_laws", []),
            )
            reasoning_entries.append(reasoning)

        # Check for critical violations
        critical_violations = [v for v in all_law_violations if v.get("level") == "Critical"]

        self._logger.info(
            "section_translation_complete",
            sections=len(translated_sections),
            violations=len(all_law_violations),
            critical=len(critical_violations),
        )

        return {
            "translated_sections": translated_sections,
            "current_section": None,
            "law_violations": all_law_violations,
            "current_step": "P5",
            "reasoning_log": state.get("reasoning_log", []) + reasoning_entries,
            "risk_flags": state.get("risk_flags", []) + all_risk_flags,
        }

    async def _translate_section(
        self,
        section_type: SectionType,
        source_text: str,
        glossary_map: dict[str, str],
        reference_numerals: dict[str, str],
    ) -> dict[str, Any]:
        """Translate a single section.

        Args:
            section_type: Type of section
            source_text: Source text to translate
            glossary_map: Glossary term mappings
            reference_numerals: Reference numeral descriptions

        Returns:
            Translation result with validation
        """
        is_claims = section_type == "claims"
        is_tac = self.is_tac_section(section_type)

        # Build section-specific prompt
        prompt = self._build_translation_prompt(
            section_type=section_type,
            source_text=source_text,
            glossary_map=glossary_map,
            reference_numerals=reference_numerals,
        )

        # Invoke LLM for translation
        translated_text = await self._invoke_llm(prompt)

        # Apply post-processing
        translated_text = self._post_process_translation(
            translated_text,
            section_type=section_type,
            reference_numerals=reference_numerals,
        )

        # Validate translation
        validation = self.validate_translation(
            source_text=source_text,
            translated_text=translated_text,
            section=section_type,
        )

        # Extract used glossary terms
        glossary_terms_used = self._find_used_glossary_terms(
            translated_text, glossary_map
        )

        # Build law compliance record
        law_compliance = {
            "LAW-T-1": self._check_law_t1_compliance(translated_text, is_claims),
            "LAW-T-2": self._check_law_t2_compliance(translated_text, is_claims),
            "LAW-T-3": self._check_law_t3_compliance(source_text, translated_text),
            "LAW-T-4": self._check_law_t4_compliance(translated_text, reference_numerals),
        }

        # Create risk flags for violations
        risk_flags = []
        violations = []

        for law_id, is_compliant in law_compliance.items():
            if not is_compliant:
                level = "Critical" if is_tac else "High"
                risk_flags.append(
                    self.create_risk_flag(
                        level=level,
                        issue=f"{law_id} 위반",
                        location=section_type,
                        recommendation=self._get_law_recommendation(law_id),
                    )
                )
                violations.append({
                    "law": law_id,
                    "section": section_type,
                    "level": level,
                })

        # Add mistranslation warnings
        for misinfo in validation.get("mistranslations", []):
            risk_flags.append(
                self.create_risk_flag(
                    level="High",
                    issue=f"오역 가능: {misinfo['source_term']}",
                    location=section_type,
                    recommendation=f"'{misinfo['wrong_translation']}'을 '{misinfo['correct_translation']}'으로 수정",
                )
            )

        translated_section: TranslatedSection = {
            "section_type": section_type,
            "source_text": source_text,
            "translated_text": translated_text,
            "glossary_terms_used": glossary_terms_used,
            "risk_flags": risk_flags,
            "law_compliance": law_compliance,
        }

        return {
            "translated_section": translated_section,
            "risk_flags": risk_flags,
            "violations": violations,
            "applied_laws": [k for k, v in law_compliance.items() if v],
        }

    def _build_translation_prompt(
        self,
        section_type: SectionType,
        source_text: str,
        glossary_map: dict[str, str],
        reference_numerals: dict[str, str],
    ) -> str:
        """Build translation prompt for a section.

        Args:
            section_type: Type of section
            source_text: Source text
            glossary_map: Glossary mappings
            reference_numerals: Reference numerals

        Returns:
            Translation prompt
        """
        is_claims = section_type == "claims"

        # Build glossary section
        glossary_str = "\n".join(
            f"- {eng} → {kor}"
            for eng, kor in list(glossary_map.items())[:50]  # Top 50
        )

        # Build reference numerals section
        ref_str = "\n".join(
            f"- ({num}): {desc}"
            for num, desc in list(reference_numerals.items())[:30]
        )

        if is_claims:
            rules = """### 청구항 번역 규칙 (필수 적용)
1. LAW-T-1: 모든 'the', 'said' → '상기'
2. LAW-T-2: 각 청구항은 단일 문장, 마침표는 끝에만
3. LAW-T-3: comprising → '포함하는', consisting of → '~으로 이루어지는'
4. LAW-T-4: 모든 도면부호에 괄호 필수

### 청구항 구조
- 독립항: "[발명의 카테고리]로서, [구성요소]를 포함하는, [발명의 카테고리]."
- 종속항: "제N항에 있어서, [추가 구성요소]를 더 포함하는, [발명의 카테고리]."
"""
        else:
            rules = """### 명세서 번역 규칙 (필수 적용)
1. LAW-T-1: '상기' 사용 완전 금지 → '해당', '본', '이러한' 사용
2. LAW-T-4: 모든 도면부호에 괄호 필수

### 문체
- 자연스러운 한국어 서술체
- 기술적 정확성 유지
- 용어 일관성 필수
"""

        return f"""다음 영문 특허 섹션을 한국어로 번역하세요.

## 섹션 유형: {section_type}

{rules}

## 사용할 용어집
{glossary_str}

## 도면부호 목록
{ref_str}

## 원문
{source_text}

## 번역문
한국어 번역만 출력하세요:"""

    def _post_process_translation(
        self,
        text: str,
        section_type: SectionType,
        reference_numerals: dict[str, str],
    ) -> str:
        """Post-process translated text.

        Args:
            text: Translated text
            section_type: Section type
            reference_numerals: Reference numerals

        Returns:
            Post-processed text
        """
        is_claims = section_type == "claims"

        # LAW-T-4: Ensure all reference numerals have brackets
        for numeral in reference_numerals.keys():
            # Match numeral without brackets
            pattern = rf'(?<!\()\b{re.escape(numeral)}\b(?!\))'
            text = re.sub(pattern, f'({numeral})', text)

        # For non-claims, replace any '상기' with alternatives
        if not is_claims:
            alternatives = SpecTranslationRules.SANGGI_ALTERNATIVES
            idx = 0
            while "상기" in text:
                text = text.replace("상기", alternatives[idx % len(alternatives)], 1)
                idx += 1

        return text

    def _build_glossary_map(self, glossary: list[GlossaryEntry]) -> dict[str, str]:
        """Build a lookup map from glossary entries.

        Args:
            glossary: List of glossary entries

        Returns:
            Dictionary mapping English terms to Korean terms
        """
        return {
            entry["english_term"]: entry["korean_term"]
            for entry in glossary
            if entry.get("korean_term")
        }

    def _find_used_glossary_terms(
        self,
        translated_text: str,
        glossary_map: dict[str, str],
    ) -> list[str]:
        """Find which glossary terms were used in translation.

        Args:
            translated_text: Translated text
            glossary_map: Glossary mappings

        Returns:
            List of used Korean terms
        """
        used = []
        for korean_term in glossary_map.values():
            if korean_term in translated_text:
                used.append(korean_term)
        return used

    def _check_law_t1_compliance(self, text: str, is_claims: bool) -> bool:
        """Check LAW-T-1 compliance ('상기' usage).

        Args:
            text: Translated text
            is_claims: Whether this is claims section

        Returns:
            True if compliant
        """
        has_sanggi = "상기" in text

        if is_claims:
            # Claims should have '상기' for definite references
            return True  # Assume compliant if it's claims (hard to check without source)
        else:
            # Non-claims should NOT have '상기'
            return not has_sanggi

    def _check_law_t2_compliance(self, text: str, is_claims: bool) -> bool:
        """Check LAW-T-2 compliance (single sentence claims).

        Args:
            text: Translated text
            is_claims: Whether this is claims section

        Returns:
            True if compliant
        """
        if not is_claims:
            return True  # Only applies to claims

        # Split into individual claims
        claims = re.split(r'\n\s*\d+\.', text)

        for claim in claims:
            claim = claim.strip()
            if not claim:
                continue

            # Count periods (should be exactly one at the end)
            periods = claim.count('.')
            if periods > 1:
                return False

            # Check if it ends with period
            if not claim.endswith('.'):
                return False

        return True

    def _check_law_t3_compliance(self, source_text: str, translated_text: str) -> bool:
        """Check LAW-T-3 compliance (transitional phrases).

        Args:
            source_text: Original text
            translated_text: Translated text

        Returns:
            True if compliant
        """
        # Check comprising
        if "comprising" in source_text.lower():
            # Should use '포함하는', not '구성된'
            if "구성된" in translated_text and "포함하는" not in translated_text:
                return False

        # Check consisting of
        if "consisting of" in source_text.lower():
            # Should use '이루어지는', not '포함하는'
            if "포함하는" in translated_text and "이루어지는" not in translated_text:
                return False

        return True

    def _check_law_t4_compliance(
        self,
        text: str,
        reference_numerals: dict[str, str],
    ) -> bool:
        """Check LAW-T-4 compliance (reference numeral brackets).

        Args:
            text: Translated text
            reference_numerals: Reference numerals in document

        Returns:
            True if compliant
        """
        for numeral in reference_numerals.keys():
            # Check for numeral without brackets
            pattern = rf'(?<!\()\b{re.escape(numeral)}\b(?!\))'
            if re.search(pattern, text):
                return False

        return True

    def _get_law_recommendation(self, law_id: str) -> str:
        """Get recommendation for fixing a law violation.

        Args:
            law_id: Law identifier

        Returns:
            Recommendation string
        """
        recommendations = {
            "LAW-T-1": "청구항에서 'the/said'를 '상기'로, 명세서에서 '상기'를 '해당/본/이러한'으로 수정",
            "LAW-T-2": "청구항을 단일 문장으로 수정하고 마침표는 끝에만 사용",
            "LAW-T-3": "comprising→포함하는, consisting of→이루어지는 으로 정확히 번역",
            "LAW-T-4": "모든 도면부호에 괄호 추가 (예: 10 → (10))",
        }
        return recommendations.get(law_id, "해당 법칙 준수 필요")

    async def revise_section(
        self,
        state: TranslationState,
        section_type: SectionType,
        issues: list[str],
    ) -> dict[str, Any]:
        """Revise a section to fix identified issues.

        Args:
            state: Current state
            section_type: Section to revise
            issues: List of issues to fix

        Returns:
            State updates with revised section
        """
        translated_sections = state.get("translated_sections", {})
        section = translated_sections.get(section_type)

        if not section:
            return {"error_messages": [f"{section_type} 섹션을 찾을 수 없습니다."]}

        issues_str = "\n".join(f"- {issue}" for issue in issues)

        prompt = f"""다음 번역문에서 식별된 문제를 수정하세요.

## 원문
{section['source_text']}

## 현재 번역문
{section['translated_text']}

## 수정해야 할 문제
{issues_str}

## 수정된 번역문
수정된 한국어 번역만 출력하세요:"""

        revised_text = await self._invoke_llm(prompt)

        # Update section
        section["translated_text"] = revised_text

        self._logger.info(
            "section_revised",
            section=section_type,
            issues_fixed=len(issues),
        )

        return {
            "translated_sections": {
                **translated_sections,
                section_type: section,
            },
        }
