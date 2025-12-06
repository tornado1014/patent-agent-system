"""Specification Writer Agent.

Handles E5 (명세서 구조 설계) and E6 (본문 작성) steps.
Writes the patent specification body following KIPO format.
"""

from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate

from patent_agent.agents.spec_writing.base import BaseSpecWritingAgent
from patent_agent.state.spec_writing import Specification, SpecWritingState

logger = structlog.get_logger(__name__)


class SpecWriterAgent(BaseSpecWritingAgent):
    """Agent for writing patent specifications.

    This agent creates the patent specification document following
    Korean patent law requirements and KIPO format guidelines.

    Steps handled:
    - E5: 명세서 구조 설계
    - E6: 본문 작성 [Human Checkpoint]
    """

    @property
    def name(self) -> str:
        return "SpecWriter"

    @property
    def description(self) -> str:
        return "특허 명세서 본문을 작성하는 에이전트"

    def get_system_prompt(self) -> str:
        return """당신은 특허 전문 변리사로서, 특허법 제42조를 준수하는 명세서를 작성하는 역할을 합니다.

## 역할
- KIPO 형식에 맞는 명세서 구조 설계
- 각 섹션별 상세 본문 작성
- 청구항을 뒷받침하는 발명의 설명 작성

## 명세서 구조 (특허법 시행규칙 별지 제15호 서식)

### 1. 기술분야
- 발명이 속하는 기술분야 명시
- IPC/CPC 분류 언급 가능
- 1-2문단, 서술형

### 2. 배경기술
- 종래 기술의 문제점 설명
- 선행기술 문헌 인용 가능
- 서술형으로 자연스럽게 기술

### 3. 해결하려는 과제
- 발명이 해결하고자 하는 기술적 과제
- 종래 기술의 한계를 극복하는 목적
- 서술형

### 4. 과제의 해결 수단
- 발명의 구성 요소 설명
- 청구항의 구성요소와 일치해야 함
- 서술형으로 상세히 기술

### 5. 발명의 효과
- 기술적 효과 위주로 기술
- 과제 해결에 따른 구체적 효과
- 서술형

### 6. 도면의 간단한 설명
- 각 도면의 내용 간략 설명
- 형식: "도 1은 ~를 나타내는 도면이다."

### 7. 발명을 실시하기 위한 구체적인 내용
- 가장 상세한 섹션
- 실시예 포함
- 도면 참조하여 구성요소 설명
- 당업자가 실시 가능하도록 상세히 기술

## 작성 원칙
1. **서술형 필수**: 목록, 불릿 포인트 사용 금지 (청구항 제외)
2. **용어 일관성**: 동일 구성요소는 동일 용어 사용
3. **도면부호 규칙**: 구성요소명(도면부호) 형식
4. **청구항 지원**: 청구항의 모든 구성요소가 명세서에서 설명되어야 함
5. **실시가능 요건**: 당업자가 발명을 재현할 수 있도록 충분히 상세하게"""

    async def process(self, state: SpecWritingState) -> dict[str, Any]:
        """Design specification structure and write body.

        Args:
            state: Current workflow state

        Returns:
            State updates with specification content
        """
        current_step = state.get("current_step", "E5")
        self._logger.info("writing_specification", step=current_step)

        if current_step == "E5":
            return await self._design_structure(state)
        else:  # E6
            return await self._write_body(state)

    async def _design_structure(self, state: SpecWritingState) -> dict[str, Any]:
        """Design the specification structure (E5).

        Args:
            state: Current workflow state

        Returns:
            State updates with structure outline
        """
        invention_title = state.get("invention_title", "")
        key_features = state.get("key_features", [])
        draft_claims = state.get("draft_claims", {})
        drawings = state.get("drawings", [])

        # Generate structure outline
        structure = await self._generate_structure(
            invention_title=invention_title,
            key_features=key_features,
            claims=draft_claims.get("claims", []),
            num_drawings=len(drawings) if drawings else 3,  # Default to 3 drawings
        )

        return {
            "embodiments": structure.get("embodiments", []),
            "reference_numeral_table": structure.get("reference_numerals", {}),
            "current_step": "E6",
        }

    async def _generate_structure(
        self,
        invention_title: str,
        key_features: list[str],
        claims: list,
        num_drawings: int,
    ) -> dict[str, Any]:
        """Generate specification structure.

        Args:
            invention_title: Title of invention
            key_features: Key features
            claims: Draft claims
            num_drawings: Number of planned drawings

        Returns:
            Structure with embodiments and reference numerals
        """
        prompt = f"""다음 발명에 대한 명세서 구조를 설계해주세요.

## 발명의 명칭
{invention_title}

## 핵심 특징
{chr(10).join(f"- {f}" for f in key_features)}

## 청구항 개수
{len(claims)}개

## 도면 개수
{num_drawings}개

## 요청
1. 실시예 구성 (2-3개 실시예 제안)
2. 도면부호 테이블 (주요 구성요소에 번호 할당)

형식:
[실시예]
1. [실시예 1 제목]: [설명]
2. [실시예 2 제목]: [설명]

[도면부호]
100: [구성요소명]
110: [구성요소명]
200: [구성요소명]
..."""

        response = await self._invoke_llm(prompt)

        # Parse response
        embodiments = []
        reference_numerals = {}

        section = None
        for line in response.split("\n"):
            line = line.strip()
            if "[실시예]" in line:
                section = "embodiments"
            elif "[도면부호]" in line:
                section = "numerals"
            elif section == "embodiments" and line:
                if line[0].isdigit() and "." in line:
                    embodiments.append(line.split(".", 1)[1].strip())
            elif section == "numerals" and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    numeral = parts[0].strip()
                    name = parts[1].strip()
                    if numeral.isdigit():
                        reference_numerals[numeral] = name

        return {
            "embodiments": embodiments or ["기본 실시예"],
            "reference_numerals": reference_numerals or {"100": "주요 구성부"},
        }

    async def _write_body(self, state: SpecWritingState) -> dict[str, Any]:
        """Write the specification body (E6).

        Args:
            state: Current workflow state

        Returns:
            State updates with specification content
        """
        # Gather all necessary information
        invention_title = state.get("invention_title", "")
        tech_field = state.get("tech_field", "")
        technical_problems = state.get("technical_problems", [])
        solutions = state.get("solutions", [])
        expected_effects = state.get("expected_effects", [])
        draft_claims = state.get("draft_claims", {})
        embodiments = state.get("embodiments", [])
        reference_numerals = state.get("reference_numeral_table", {})
        drawings = state.get("drawings", [])

        # Write each section
        specification = Specification(
            title=invention_title,
            technical_field=await self._write_technical_field(tech_field, invention_title),
            background_art=await self._write_background(technical_problems),
            problems_to_solve=await self._write_problems(technical_problems),
            means_to_solve=await self._write_means(solutions, draft_claims.get("claims", [])),
            effects=await self._write_effects(expected_effects),
            brief_description_drawings=await self._write_drawing_descriptions(drawings, reference_numerals),
            detailed_description=await self._write_detailed_description(
                invention_title, solutions, embodiments, reference_numerals, draft_claims.get("claims", [])
            ),
            industrial_applicability=None,  # Optional section
        )

        # Validate specification
        validation_errors = await self._validate_specification(specification, draft_claims)

        return {
            "draft_specification": specification,
            "review_comments": [
                {"issue": err, "severity": "medium", "section": "specification"}
                for err in validation_errors
            ],
            "current_step": "E7",
            "human_approval_required": True,  # E6 is a human checkpoint
            "pending_approval_checkpoint": "E6",
        }

    async def _write_technical_field(self, tech_field: str, title: str) -> str:
        """Write the technical field section."""
        prompt = f"""다음 정보로 '기술분야' 섹션을 작성해주세요.

발명의 명칭: {title}
기술 분야: {tech_field}

요구사항:
- 1-2문단
- 서술형 (목록 금지)
- "본 발명은 ~ 에 관한 것이다" 형식으로 시작

기술분야:"""

        response = await self._invoke_llm(prompt)
        return response.strip()

    async def _write_background(self, problems: list[str]) -> str:
        """Write the background art section."""
        prompt = f"""다음 기술적 과제를 바탕으로 '배경기술' 섹션을 작성해주세요.

기술적 과제:
{chr(10).join(f"- {p}" for p in problems)}

요구사항:
- 2-4문단
- 서술형 (목록 금지)
- 종래 기술의 현황과 문제점을 설명
- "종래에는 ~ 하였으나, ~ 의 문제점이 있었다" 흐름

배경기술:"""

        response = await self._invoke_llm(prompt)
        return response.strip()

    async def _write_problems(self, problems: list[str]) -> str:
        """Write the problems to solve section."""
        prompt = f"""다음 기술적 과제로 '해결하려는 과제' 섹션을 작성해주세요.

기술적 과제:
{chr(10).join(f"- {p}" for p in problems)}

요구사항:
- 1-2문단
- 서술형 (목록 금지)
- "본 발명은 상기와 같은 문제점을 해결하기 위해 ~ " 형식

해결하려는 과제:"""

        response = await self._invoke_llm(prompt)
        return response.strip()

    async def _write_means(self, solutions: list[str], claims: list) -> str:
        """Write the means to solve section."""
        claim_texts = [c.get("full_text", "") for c in claims[:2] if c.get("claim_type") == "independent"]

        prompt = f"""다음 해결 수단과 청구항을 바탕으로 '과제의 해결 수단' 섹션을 작성해주세요.

해결 수단:
{chr(10).join(f"- {s}" for s in solutions)}

독립 청구항:
{chr(10).join(claim_texts)}

요구사항:
- 2-4문단
- 서술형 (목록 금지)
- 청구항의 구성요소를 설명
- "상기 목적을 달성하기 위하여 본 발명은 ~ " 형식으로 시작

과제의 해결 수단:"""

        response = await self._invoke_llm(prompt)
        return response.strip()

    async def _write_effects(self, effects: list[str]) -> str:
        """Write the effects section."""
        prompt = f"""다음 효과로 '발명의 효과' 섹션을 작성해주세요.

기대 효과:
{chr(10).join(f"- {e}" for e in effects)}

요구사항:
- 1-3문단
- 서술형 (목록 금지)
- "본 발명에 따르면, ~ 효과가 있다" 형식

발명의 효과:"""

        response = await self._invoke_llm(prompt)
        return response.strip()

    async def _write_drawing_descriptions(
        self,
        drawings: list,
        reference_numerals: dict[str, str],
    ) -> str:
        """Write brief description of drawings."""
        if not drawings and not reference_numerals:
            return "도 1은 본 발명의 일 실시예에 따른 구성을 나타내는 도면이다."

        descriptions = []
        for i, drawing in enumerate(drawings, 1):
            if isinstance(drawing, dict):
                desc = drawing.get("description", f"본 발명의 구성요소를 나타내는 도면")
                descriptions.append(f"도 {i}은 {desc}이다.")
            else:
                descriptions.append(f"도 {i}은 본 발명의 구성을 나타내는 도면이다.")

        # If no drawings provided, create placeholder
        if not descriptions:
            descriptions = [
                "도 1은 본 발명의 일 실시예에 따른 전체 구성을 나타내는 블록도이다.",
                "도 2는 본 발명의 일 실시예에 따른 동작 과정을 나타내는 순서도이다.",
            ]

        return "\n".join(descriptions)

    async def _write_detailed_description(
        self,
        title: str,
        solutions: list[str],
        embodiments: list[str],
        reference_numerals: dict[str, str],
        claims: list,
    ) -> str:
        """Write the detailed description section."""
        # Build reference numeral context
        ref_context = "\n".join(f"- {num}: {name}" for num, name in reference_numerals.items())

        prompt = f"""다음 정보로 '발명을 실시하기 위한 구체적인 내용' 섹션을 작성해주세요.

발명의 명칭: {title}

해결 수단:
{chr(10).join(f"- {s}" for s in solutions)}

실시예:
{chr(10).join(f"- {e}" for e in embodiments)}

도면부호:
{ref_context}

요구사항:
- 10-20문단
- 서술형 (목록 금지)
- 도면부호는 반드시 괄호 안에 표기: 구성요소명(100)
- 상세하게 당업자가 재현할 수 있도록 작성
- 실시예를 통해 구체적으로 설명
- "이하, 첨부된 도면을 참조하여 본 발명의 바람직한 실시예를 상세히 설명한다." 로 시작

발명을 실시하기 위한 구체적인 내용:"""

        response = await self._invoke_llm(prompt)
        return response.strip()

    async def _validate_specification(
        self,
        specification: Specification,
        draft_claims: dict,
    ) -> list[str]:
        """Validate specification against requirements.

        Args:
            specification: Written specification
            draft_claims: Draft claims to check support

        Returns:
            List of validation errors
        """
        errors = []

        # Check each section for list formatting
        sections = [
            ("기술분야", specification.get("technical_field", "")),
            ("배경기술", specification.get("background_art", "")),
            ("해결하려는 과제", specification.get("problems_to_solve", "")),
            ("과제의 해결 수단", specification.get("means_to_solve", "")),
            ("발명의 효과", specification.get("effects", "")),
            ("상세한 설명", specification.get("detailed_description", "")),
        ]

        for section_name, content in sections:
            if content:
                is_valid, violations = self._validate_output(
                    content,
                    {"section_type": "specification"},
                )
                if not is_valid:
                    errors.extend([f"{section_name}: {v}" for v in violations])

        # Check if specification is empty
        if not specification.get("detailed_description"):
            errors.append("상세한 설명이 작성되지 않았습니다.")

        return errors
