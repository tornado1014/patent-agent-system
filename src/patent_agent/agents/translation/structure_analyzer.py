"""Structure Analyzer Agent (P2) for Patent Translation workflow.

Handles document structure analysis:
- Section identification and mapping to KIPO format
- Claim dependency analysis
- Reference numeral extraction
- Section dependency mapping
"""

import re
from typing import Any

from patent_agent.agents.translation.base import BaseTranslationAgent
from patent_agent.state.translation import (
    DocumentStructure,
    SectionType,
    TranslationPhase,
    TranslationState,
)


# KIPO section name mappings
KIPO_SECTION_NAMES = {
    "title": "발명의 명칭",
    "field": "기술분야",
    "background": "배경기술",
    "summary": "발명의 내용",
    "brief_description": "도면의 간단한 설명",
    "detailed_description": "발명을 실시하기 위한 구체적인 내용",
    "claims": "청구범위",
    "abstract": "요약서",
}

# English section headers patterns
SECTION_PATTERNS = {
    "title": [
        r"(?i)^title\s*(?:of\s+(?:the\s+)?invention)?[:\s]*",
        r"(?i)invention\s*title[:\s]*",
    ],
    "field": [
        r"(?i)(?:technical\s+)?field\s+of\s+(?:the\s+)?invention[:\s]*",
        r"(?i)technical\s+field[:\s]*",
        r"(?i)field\s+of\s+invention[:\s]*",
    ],
    "background": [
        r"(?i)background\s+(?:of\s+(?:the\s+)?invention)?[:\s]*",
        r"(?i)prior\s+art[:\s]*",
        r"(?i)related\s+art[:\s]*",
    ],
    "summary": [
        r"(?i)summary\s+(?:of\s+(?:the\s+)?invention)?[:\s]*",
        r"(?i)brief\s+summary[:\s]*",
    ],
    "brief_description": [
        r"(?i)brief\s+description\s+of\s+(?:the\s+)?(?:drawings?|figures?)[:\s]*",
        r"(?i)description\s+of\s+(?:the\s+)?(?:drawings?|figures?)[:\s]*",
    ],
    "detailed_description": [
        r"(?i)detailed\s+description[:\s]*",
        r"(?i)description\s+of\s+(?:the\s+)?(?:preferred\s+)?embodiments?[:\s]*",
        r"(?i)modes?\s+for\s+carrying\s+out\s+(?:the\s+)?invention[:\s]*",
    ],
    "claims": [
        r"(?i)^claims?[:\s]*$",
        r"(?i)what\s+is\s+claimed\s+is[:\s]*",
        r"(?i)the\s+claims?\s+defining\s+the\s+invention[:\s]*",
    ],
    "abstract": [
        r"(?i)^abstract[:\s]*$",
        r"(?i)abstract\s+of\s+(?:the\s+)?disclosure[:\s]*",
    ],
}


class StructureAnalyzerAgent(BaseTranslationAgent):
    """Document structure analysis agent.

    Phase P2: 구조 분석 및 의존성 매핑

    Responsibilities:
    - Identify document sections
    - Map to KIPO standard format
    - Extract claim count and dependencies
    - Extract figure/drawing count
    - Identify reference numerals
    - Determine section dependencies
    """

    @property
    def name(self) -> str:
        return "StructureAnalyzerAgent"

    @property
    def description(self) -> str:
        return "문서 구조 분석 및 KIPO 형식 매핑을 담당하는 에이전트"

    @property
    def phase(self) -> TranslationPhase:
        return "P2"

    def get_system_prompt(self) -> str:
        return """## Structure Analyzer Agent (P2)

당신은 영문 특허 문서의 구조를 분석하고 KIPO 형식에 매핑하는 에이전트입니다.

### 주요 임무
1. 문서 섹션 식별
   - 발명의 명칭 (Title)
   - 기술분야 (Field of Invention)
   - 배경기술 (Background)
   - 발명의 내용 (Summary)
   - 도면의 간단한 설명 (Brief Description of Drawings)
   - 발명을 실시하기 위한 구체적인 내용 (Detailed Description)
   - 청구범위 (Claims)
   - 요약서 (Abstract)

2. 청구항 분석
   - 청구항 개수 파악
   - 독립항/종속항 구분
   - 종속항 의존 관계 파악

3. 도면 및 참조번호 분석
   - 도면 개수 파악
   - 도면부호 목록 추출

### KIPO 표준 섹션 순서
1. 발명의 명칭
2. 기술분야
3. 배경기술
4. 발명의 내용
5. 도면의 간단한 설명
6. 발명을 실시하기 위한 구체적인 내용
7. 청구범위
8. 요약서
"""

    async def process(self, state: TranslationState) -> dict[str, Any]:
        """Analyze document structure.

        Args:
            state: Current workflow state

        Returns:
            State updates with document structure
        """
        self._logger.info("starting_structure_analysis")

        # Get source text
        source_text = state.get("source_text_content", "")

        if not source_text:
            self._logger.error("no_source_text")
            return {
                "is_error_state": True,
                "error_messages": ["소스 텍스트가 없습니다."],
                "current_step": "P2",
            }

        # Analyze structure
        sections = await self._identify_sections(source_text)
        missing_sections = self._find_missing_sections(sections)
        claims_count, claim_deps = await self._analyze_claims(sections.get("claims", ""))
        figures_count = self._count_figures(source_text)
        reference_numerals = self._extract_reference_numerals(source_text)
        section_deps = self._determine_section_dependencies()

        # Create document structure
        doc_structure: DocumentStructure = {
            "sections": sections,
            "claims_count": claims_count,
            "figures_count": figures_count,
            "reference_numerals": reference_numerals,
            "claim_dependencies": claim_deps,
        }

        # Create reasoning entry
        reasoning = self.create_reasoning_entry(
            checkpoint="CP2.1",
            decision="구조 분석 완료",
            reasoning=f"섹션 {len(sections)}개 식별, "
            f"청구항 {claims_count}개, "
            f"도면 {figures_count}개, "
            f"누락 섹션: {missing_sections if missing_sections else '없음'}",
            applied_laws=[],
        )

        # Risk flags for missing critical sections
        risk_flags = []
        if "claims" not in sections:
            risk_flags.append(
                self.create_risk_flag(
                    level="Critical",
                    issue="청구범위 섹션 누락",
                    location="document",
                    recommendation="청구범위가 문서에 포함되어 있는지 확인 필요",
                )
            )

        self._logger.info(
            "structure_analysis_complete",
            sections=list(sections.keys()),
            claims_count=claims_count,
            figures_count=figures_count,
        )

        return {
            "document_structure": doc_structure,
            "missing_sections": missing_sections,
            "section_dependencies": section_deps,
            "current_step": "P3",
            "reasoning_log": state.get("reasoning_log", []) + [reasoning],
            "risk_flags": state.get("risk_flags", []) + risk_flags,
        }

    async def _identify_sections(self, text: str) -> dict[SectionType, str]:
        """Identify and extract document sections.

        Args:
            text: Full document text

        Returns:
            Dictionary mapping section types to their content
        """
        sections: dict[SectionType, str] = {}

        # Try pattern matching first
        for section_type, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    # Find the content after the header
                    start = match.end()
                    # Find next section or end
                    next_section = self._find_next_section(text, start)
                    content = text[start:next_section].strip()
                    if content:
                        sections[section_type] = content
                    break

        # If pattern matching didn't work well, use LLM
        if len(sections) < 3:
            sections = await self._llm_section_extraction(text)

        return sections

    def _find_next_section(self, text: str, start: int) -> int:
        """Find the start of the next section.

        Args:
            text: Full document text
            start: Starting position to search from

        Returns:
            Position of next section or end of text
        """
        next_pos = len(text)

        for patterns in SECTION_PATTERNS.values():
            for pattern in patterns:
                match = re.search(pattern, text[start:], re.MULTILINE)
                if match and (start + match.start()) < next_pos:
                    next_pos = start + match.start()

        return next_pos

    async def _llm_section_extraction(self, text: str) -> dict[SectionType, str]:
        """Use LLM to extract sections when pattern matching fails.

        Args:
            text: Full document text

        Returns:
            Dictionary mapping section types to content
        """
        # Truncate if too long
        excerpt = text[:15000] if len(text) > 15000 else text

        prompt = f"""다음 영문 특허 문서에서 각 섹션을 식별하고 추출하세요.

문서 내용:
{excerpt}

다음 섹션들을 JSON 형식으로 추출하세요:
- title: 발명의 명칭
- field: 기술분야
- background: 배경기술
- summary: 발명의 내용/요약
- brief_description: 도면의 간단한 설명
- detailed_description: 상세한 설명
- claims: 청구범위 (청구항 전체)
- abstract: 요약서

각 섹션의 전체 내용을 포함하세요. 해당 섹션이 없으면 빈 문자열로 표시하세요.
JSON만 출력하세요:"""

        try:
            response = await self._invoke_llm(prompt)

            import json

            # Extract JSON
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                return {k: v for k, v in data.items() if v}

        except Exception as e:
            self._logger.warning("llm_section_extraction_error", error=str(e))

        return {}

    def _find_missing_sections(
        self, sections: dict[SectionType, str]
    ) -> list[SectionType]:
        """Find missing sections.

        Args:
            sections: Identified sections

        Returns:
            List of missing section types
        """
        required_sections: list[SectionType] = [
            "title",
            "claims",
            "abstract",
        ]

        missing = [s for s in required_sections if s not in sections]
        return missing

    async def _analyze_claims(
        self, claims_text: str
    ) -> tuple[int, dict[int, int | None]]:
        """Analyze claims section.

        Args:
            claims_text: Claims section text

        Returns:
            Tuple of (claims count, dependency mapping)
        """
        if not claims_text:
            return 0, {}

        # Count claims
        claim_pattern = r"(?:^|\n)\s*(\d+)\s*\."
        claims = re.findall(claim_pattern, claims_text)
        claims_count = len(claims) if claims else 0

        # Find dependencies (e.g., "according to claim 1")
        dependencies: dict[int, int | None] = {}
        dep_pattern = r"(?:claim|claims?)\s*(\d+)\s*,?\s*(?:wherein|where|characterized|according\s+to\s+claim\s+(\d+))"

        for i in range(1, claims_count + 1):
            # Check if this claim depends on another
            claim_text = self._get_claim_text(claims_text, i)
            dep_match = re.search(
                r"(?:according\s+to|as\s+(?:set\s+forth|defined)\s+in)\s+claim\s+(\d+)",
                claim_text,
                re.IGNORECASE,
            )
            if dep_match:
                dependencies[i] = int(dep_match.group(1))
            else:
                dependencies[i] = None  # Independent claim

        return claims_count, dependencies

    def _get_claim_text(self, claims_text: str, claim_num: int) -> str:
        """Extract text for a specific claim.

        Args:
            claims_text: Full claims section
            claim_num: Claim number to extract

        Returns:
            Text of the specific claim
        """
        pattern = rf"(?:^|\n)\s*{claim_num}\s*\.\s*(.*?)(?=(?:\n\s*\d+\s*\.)|$)"
        match = re.search(pattern, claims_text, re.DOTALL)
        return match.group(1) if match else ""

    def _count_figures(self, text: str) -> int:
        """Count the number of figures/drawings.

        Args:
            text: Full document text

        Returns:
            Number of figures
        """
        # Look for figure references
        fig_pattern = r"(?:FIG\.?|Figure|Fig\.?)\s*(\d+)"
        matches = re.findall(fig_pattern, text, re.IGNORECASE)

        if matches:
            return max(int(m) for m in matches)

        return 0

    def _extract_reference_numerals(self, text: str) -> dict[str, str]:
        """Extract reference numerals and their descriptions.

        Args:
            text: Full document text

        Returns:
            Dictionary mapping numerals to descriptions
        """
        numerals: dict[str, str] = {}

        # Pattern: description followed by numeral in parentheses or vice versa
        patterns = [
            r"(\w[\w\s]{2,30})\s*\((\d+[a-z]?)\)",  # "component (10)"
            r"(\d+[a-z]?)\s*[-:]\s*(\w[\w\s]{2,30})",  # "10: component" or "10 - component"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches[:100]:  # Limit to avoid too many
                if pattern.startswith(r"(\d"):
                    numeral, desc = match
                else:
                    desc, numeral = match

                desc = desc.strip()
                if len(desc) > 3 and numeral not in numerals:
                    numerals[numeral] = desc

        return numerals

    def _determine_section_dependencies(
        self,
    ) -> dict[SectionType, list[SectionType]]:
        """Determine translation dependencies between sections.

        Returns:
            Mapping of sections to their dependencies
        """
        # Section translation order matters for term consistency
        return {
            "title": [],
            "abstract": ["title"],
            "field": ["title"],
            "background": ["title", "field"],
            "summary": ["title", "field", "background"],
            "claims": ["title", "summary"],  # Claims should align with summary
            "brief_description": ["title"],
            "detailed_description": ["title", "summary", "claims", "brief_description"],
        }
