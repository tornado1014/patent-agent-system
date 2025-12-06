"""Glossary Builder Agent (P3) for Patent Translation workflow.

Handles terminology consistency:
- Extract technical terms from source document
- Query KIPRIS and WIPO Pearl for standard translations
- Detect and resolve terminology conflicts
- Build finalized glossary before translation
"""

import re
from typing import Any

from patent_agent.agents.translation.base import (
    BaseTranslationAgent,
    MistranslationPrevention,
)
from patent_agent.state.translation import (
    GlossaryEntry,
    TranslationPhase,
    TranslationState,
)


# Common technical term patterns in patents
TERM_PATTERNS = [
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Capitalized phrases
    r"\b(?:method|system|device|apparatus|composition|compound)\s+(?:for|of)\s+\w+(?:\s+\w+)*\b",
    r"\b\w+(?:tion|ment|ness|ity|ing)\b",  # Nominalized verbs
    r"\b(?:first|second|third|plurality of)\s+\w+\b",
]

# Standard patent terminology that must be in glossary
REQUIRED_TERMS = [
    "comprising",
    "consisting of",
    "consisting essentially of",
    "wherein",
    "characterized in that",
    "embodiment",
    "aspect",
    "prior art",
    "means for",
    "configured to",
    "adapted to",
]


class GlossaryBuilderAgent(BaseTranslationAgent):
    """Glossary building agent.

    Phase P3: 용어 일관성 구축 (번역 전 필수)

    Responsibilities:
    - Extract technical terms from source
    - Query standard terminology databases (KIPRIS, WIPO Pearl)
    - Detect conflicts and inconsistencies
    - Apply mistranslation prevention rules
    - Finalize glossary (required before P4)
    """

    @property
    def name(self) -> str:
        return "GlossaryBuilderAgent"

    @property
    def description(self) -> str:
        return "용어 추출 및 표준 용어집 구축을 담당하는 에이전트"

    @property
    def phase(self) -> TranslationPhase:
        return "P3"

    def get_system_prompt(self) -> str:
        return """## Glossary Builder Agent (P3)

당신은 특허 번역을 위한 용어집을 구축하는 에이전트입니다.

### 주요 임무
1. 기술용어 추출
   - 영문 특허에서 핵심 기술용어 식별
   - 명사구, 복합어, 약어 추출
   - 도면부호와 연관된 용어 추출

2. 표준 용어 조회
   - KIPRIS 특허 용어 사전
   - WIPO Pearl 다국어 특허 용어집
   - 기술분야별 표준 용어

3. 오역 방지 용어 적용
   - 분야별 주의 용어 매핑
   - 컨텍스트 기반 용어 선택
   - 충돌 용어 플래그

4. 용어집 확정
   - 충돌 해결 (사용자 검토 필요 시 플래그)
   - 최종 용어집 생성
   - 번역 단계로 진행 승인

### 품질 기준
- 용어 일관성 100% 달성 목표
- 권리범위 결정 용어 정확성 필수
- 기술분야 적합성 검증

### 주요 오역 방지 목록
- more than one → 둘 이상 (하나 이상 X)
- substrate → 기판 (화학: 기재)
- cathode → 양극 (이차전지)
- anode → 음극 (이차전지)
- adapted to → ~하도록 구성되는
- ground → 접지 (전기분야)
"""

    async def process(self, state: TranslationState) -> dict[str, Any]:
        """Build terminology glossary.

        Args:
            state: Current workflow state

        Returns:
            State updates with glossary
        """
        self._logger.info("starting_glossary_building")

        # Get document structure from P2
        doc_structure = state.get("document_structure")
        if not doc_structure:
            self._logger.error("no_document_structure")
            return {
                "is_error_state": True,
                "error_messages": ["문서 구조 분석 결과가 없습니다. P2를 먼저 실행하세요."],
                "current_step": "P3",
            }

        sections = doc_structure.get("sections", {})
        reference_numerals = doc_structure.get("reference_numerals", {})

        # Step 1: Extract terms from all sections
        extracted_terms = await self._extract_terms(sections)

        # Step 2: Add required legal terms
        all_terms = list(set(extracted_terms + REQUIRED_TERMS))

        # Step 3: Look up standard translations
        glossary_entries = await self._build_glossary_entries(all_terms, sections)

        # Step 4: Add reference numeral terms
        for numeral, description in reference_numerals.items():
            # Find Korean translation for the description
            korean_desc = await self._translate_term(description)
            glossary_entries.append(
                GlossaryEntry(
                    english_term=description,
                    korean_term=korean_desc,
                    source="auto",
                    domain="reference_numeral",
                    notes=f"도면부호 ({numeral})",
                )
            )

        # Step 5: Detect conflicts
        conflicts = self._detect_conflicts(glossary_entries)

        # Step 6: Determine if finalized
        glossary_finalized = len(conflicts) == 0

        # Create reasoning entry
        reasoning = self.create_reasoning_entry(
            checkpoint="CP3.1",
            decision="용어집 구축" if glossary_finalized else "용어집 충돌 감지",
            reasoning=f"추출 용어 {len(extracted_terms)}개, "
            f"표준 매핑 {len(glossary_entries)}개, "
            f"충돌 {len(conflicts)}개",
            applied_laws=[],
        )

        # Risk flags for conflicts
        risk_flags = []
        if conflicts:
            for conflict in conflicts:
                risk_flags.append(
                    self.create_risk_flag(
                        level="High",
                        issue=f"용어 충돌: {conflict['term']}",
                        location="glossary",
                        recommendation=f"선택: {conflict['options']}",
                    )
                )

        self._logger.info(
            "glossary_building_complete",
            total_terms=len(all_terms),
            entries=len(glossary_entries),
            conflicts=len(conflicts),
            finalized=glossary_finalized,
        )

        return {
            "extracted_terms": all_terms,
            "glossary": glossary_entries,
            "glossary_conflicts": conflicts,
            "glossary_finalized": glossary_finalized,
            "current_step": "P4" if glossary_finalized else "P3",
            "human_approval_required": len(conflicts) > 0,
            "reasoning_log": state.get("reasoning_log", []) + [reasoning],
            "risk_flags": state.get("risk_flags", []) + risk_flags,
        }

    async def _extract_terms(self, sections: dict[str, str]) -> list[str]:
        """Extract technical terms from all sections.

        Args:
            sections: Dictionary of section content

        Returns:
            List of extracted terms
        """
        terms = set()

        full_text = " ".join(sections.values())

        # Use patterns to find technical terms
        for pattern in TERM_PATTERNS:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            terms.update(matches)

        # Extract via LLM for better accuracy
        llm_terms = await self._llm_term_extraction(full_text[:10000])
        terms.update(llm_terms)

        return list(terms)

    async def _llm_term_extraction(self, text: str) -> list[str]:
        """Use LLM to extract technical terms.

        Args:
            text: Text excerpt to analyze

        Returns:
            List of extracted terms
        """
        prompt = f"""다음 영문 특허 텍스트에서 기술용어를 추출하세요.

텍스트:
{text[:8000]}

추출 기준:
1. 핵심 기술 개념 명사 및 명사구
2. 복합 기술용어
3. 약어 및 두문자어
4. 특허 법률 용어 (comprising, wherein 등)
5. 도면부호와 함께 사용된 용어

각 용어를 줄바꿈으로 구분하여 출력하세요.
용어만 출력하고 설명은 포함하지 마세요:"""

        try:
            response = await self._invoke_llm(prompt)
            # Parse response into list
            terms = [
                line.strip()
                for line in response.split("\n")
                if line.strip() and len(line.strip()) > 2
            ]
            return terms[:100]  # Limit to 100 terms

        except Exception as e:
            self._logger.warning("llm_term_extraction_error", error=str(e))
            return []

    async def _build_glossary_entries(
        self, terms: list[str], sections: dict[str, str]
    ) -> list[GlossaryEntry]:
        """Build glossary entries with standard translations.

        Args:
            terms: List of terms to translate
            sections: Document sections for context

        Returns:
            List of glossary entries
        """
        entries = []

        # Detect technical domain from document
        domain = await self._detect_domain(sections)

        for term in terms:
            term_lower = term.lower()

            # Check mistranslation prevention first
            if term_lower in MistranslationPrevention.MAPPINGS:
                wrong, correct, notes = MistranslationPrevention.MAPPINGS[term_lower]
                entries.append(
                    GlossaryEntry(
                        english_term=term,
                        korean_term=correct,
                        source="auto",
                        domain=domain,
                        notes=f"주의: '{wrong}'으로 오역하지 말 것. {notes}",
                    )
                )
                continue  # Skip to next term

            # Standard translation lookup
            korean = await self._lookup_standard_translation(term, domain)
            entries.append(
                GlossaryEntry(
                    english_term=term,
                    korean_term=korean,
                    source="auto" if korean else "user",
                    domain=domain,
                    notes=None,
                )
            )

        return entries

    async def _detect_domain(self, sections: dict[str, str]) -> str:
        """Detect technical domain from document content.

        Args:
            sections: Document sections

        Returns:
            Detected domain string
        """
        # Check for domain indicators
        full_text = " ".join(sections.values()).lower()

        domain_indicators = {
            "chemistry": ["compound", "catalyst", "reaction", "synthesis", "polymer"],
            "electronics": [
                "circuit",
                "semiconductor",
                "transistor",
                "electrode",
                "voltage",
            ],
            "battery": [
                "battery",
                "cathode",
                "anode",
                "electrolyte",
                "lithium",
                "charging",
            ],
            "mechanical": ["gear", "shaft", "bearing", "motor", "actuator", "torque"],
            "biotech": ["protein", "gene", "cell", "antibody", "dna", "rna"],
            "software": ["algorithm", "processor", "memory", "data", "network"],
        }

        scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for ind in indicators if ind in full_text)
            scores[domain] = score

        if max(scores.values()) > 2:
            return max(scores, key=scores.get)
        return "general"

    async def _lookup_standard_translation(
        self, term: str, domain: str
    ) -> str:
        """Look up standard translation from databases.

        Args:
            term: English term
            domain: Technical domain

        Returns:
            Korean translation or empty string
        """
        # In production, this would query KIPRIS and WIPO Pearl APIs
        # For now, use LLM-based translation with domain context

        prompt = f"""다음 영문 기술용어의 한국어 특허 표준 번역을 제공하세요.

용어: {term}
기술분야: {domain}

규칙:
1. 특허 명세서에서 사용되는 공식 용어 사용
2. 불필요한 조사/어미 없이 용어만 출력
3. 해당 분야의 관행적 용어 사용

한국어 용어만 출력:"""

        try:
            response = await self._invoke_llm(prompt)
            return response.strip()
        except Exception:
            return ""

    async def _translate_term(self, term: str) -> str:
        """Translate a single term.

        Args:
            term: English term

        Returns:
            Korean translation
        """
        return await self._lookup_standard_translation(term, "general")

    def _detect_conflicts(
        self, entries: list[GlossaryEntry]
    ) -> list[dict]:
        """Detect conflicting translations.

        Args:
            entries: Glossary entries

        Returns:
            List of conflicts
        """
        conflicts = []

        # Group by English term
        term_groups: dict[str, list[GlossaryEntry]] = {}
        for entry in entries:
            term = entry["english_term"].lower()
            if term not in term_groups:
                term_groups[term] = []
            term_groups[term].append(entry)

        # Find conflicts (same term, different translations)
        for term, group in term_groups.items():
            korean_terms = list(set(e["korean_term"] for e in group))
            if len(korean_terms) > 1:
                conflicts.append(
                    {
                        "term": term,
                        "options": korean_terms,
                        "entries": group,
                        "recommendation": korean_terms[0],  # First one as default
                    }
                )

        return conflicts

    async def resolve_conflict(
        self, state: TranslationState, term: str, selected_korean: str
    ) -> dict[str, Any]:
        """Resolve a terminology conflict.

        Args:
            state: Current state
            term: English term with conflict
            selected_korean: User-selected Korean translation

        Returns:
            State updates
        """
        glossary = state.get("glossary", [])
        conflicts = state.get("glossary_conflicts", [])

        # Update glossary with selected translation
        updated_glossary = []
        for entry in glossary:
            if entry["english_term"].lower() == term.lower():
                entry["korean_term"] = selected_korean
                entry["source"] = "user"
            updated_glossary.append(entry)

        # Remove resolved conflict
        remaining_conflicts = [c for c in conflicts if c["term"].lower() != term.lower()]

        self._logger.info(
            "conflict_resolved",
            term=term,
            selected=selected_korean,
            remaining=len(remaining_conflicts),
        )

        return {
            "glossary": updated_glossary,
            "glossary_conflicts": remaining_conflicts,
            "glossary_finalized": len(remaining_conflicts) == 0,
            "current_step": "P4" if len(remaining_conflicts) == 0 else "P3",
        }
