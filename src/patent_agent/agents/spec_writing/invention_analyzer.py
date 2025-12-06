"""Invention Analyzer Agent.

Handles E1 (발명 개요 수집) and E2 (핵심 정보 상세 질문) steps.
Analyzes invention disclosure and extracts key information.
"""

from typing import Any

import structlog
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from patent_agent.agents.spec_writing.base import BaseSpecWritingAgent
from patent_agent.state.spec_writing import SpecWritingState

logger = structlog.get_logger(__name__)


class InventionAnalysis(BaseModel):
    """Structured output for invention analysis."""

    invention_title: str = Field(description="발명의 명칭")
    core_technology: str = Field(description="핵심 기술 요약")
    key_features: list[str] = Field(description="주요 기술적 특징 목록")
    technical_problems: list[str] = Field(description="해결하고자 하는 기술적 과제")
    solutions: list[str] = Field(description="기술적 과제의 해결 수단")
    expected_effects: list[str] = Field(description="발명의 기대 효과")
    tech_field: str = Field(description="기술 분야 (IPC 코드 포함)")
    follow_up_questions: list[str] = Field(description="추가로 필요한 정보에 대한 질문")


class InventionAnalyzerAgent(BaseSpecWritingAgent):
    """Agent for analyzing invention disclosures.

    This agent processes invention disclosure documents and extracts
    structured information needed for patent specification writing.

    Steps handled:
    - E1: 발명 개요 수집 - Initial analysis of disclosure
    - E2: 핵심 정보 상세 질문 - Follow-up questions for missing info
    """

    @property
    def name(self) -> str:
        return "InventionAnalyzer"

    @property
    def description(self) -> str:
        return "발명 신고서를 분석하고 핵심 정보를 추출하는 에이전트"

    def get_system_prompt(self) -> str:
        return """당신은 특허 전문 변리사로서, 발명 신고서를 분석하여 특허 명세서 작성에 필요한 핵심 정보를 추출하는 역할을 합니다.

## 역할
- 발명 신고서 분석 및 핵심 기술 파악
- 기술적 과제와 해결 수단 도출
- IPC/CPC 분류 코드 추천
- 누락된 정보에 대한 질문 생성

## 분석 원칙
1. **기술적 특징 중심**: 발명의 핵심 기술적 특징을 명확히 파악
2. **과제-해결-효과 연결**: 기술적 과제와 해결 수단, 효과의 인과관계 명확화
3. **선행기술 대비**: 기존 기술과의 차별점 파악
4. **청구항 지향**: 독립항으로 작성 가능한 핵심 구성요소 식별

## 출력 요구사항
- 모든 분석 결과는 구체적이고 기술적인 용어 사용
- 발명의 범위를 넓게 해석하되, 핵심은 정확히 파악
- 누락된 정보는 구체적인 질문으로 명시"""

    async def process(self, state: SpecWritingState) -> dict[str, Any]:
        """Analyze invention disclosure and extract key information.

        Args:
            state: Current workflow state with invention_disclosure

        Returns:
            State updates with extracted information
        """
        self._logger.info("analyzing_invention", step=state.get("current_step"))

        disclosure = state.get("invention_disclosure", "")
        if not disclosure:
            self._logger.error("no_disclosure_provided")
            return {
                "error_messages": ["발명 신고서가 제공되지 않았습니다."],
                "is_error_state": True,
            }

        # Analyze the disclosure
        analysis = await self._analyze_disclosure(disclosure)

        # Determine if we need more information (E2)
        needs_more_info = len(analysis.follow_up_questions) > 0

        return {
            "invention_title": analysis.invention_title,
            "core_technology": analysis.core_technology,
            "key_features": analysis.key_features,
            "technical_problems": analysis.technical_problems,
            "solutions": analysis.solutions,
            "expected_effects": analysis.expected_effects,
            "tech_field": analysis.tech_field,
            # Store follow-up questions for E2 if needed
            "follow_up_questions": analysis.follow_up_questions if needs_more_info else [],
            "current_step": "E2" if needs_more_info else "E3",
        }

    async def _analyze_disclosure(self, disclosure: str) -> InventionAnalysis:
        """Analyze invention disclosure document.

        Args:
            disclosure: Raw invention disclosure text

        Returns:
            Structured analysis result
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_full_system_prompt()),
            ("human", """다음 발명 신고서를 분석하여 특허 명세서 작성에 필요한 정보를 추출해주세요.

## 발명 신고서
{disclosure}

## 요청 사항
위 발명 신고서를 분석하여 다음 정보를 JSON 형식으로 제공해주세요:
1. invention_title: 발명의 명칭 (간결하고 기술적인 명칭)
2. core_technology: 핵심 기술 요약 (1-2문장)
3. key_features: 주요 기술적 특징 목록 (청구항 작성에 활용)
4. technical_problems: 해결하고자 하는 기술적 과제 목록
5. solutions: 기술적 과제의 해결 수단 목록
6. expected_effects: 발명의 기대 효과 목록
7. tech_field: 기술 분야 (가능하면 IPC 코드 포함)
8. follow_up_questions: 명세서 작성을 위해 추가로 필요한 정보에 대한 질문 목록

JSON 형식으로만 응답해주세요."""),
        ])

        parser = JsonOutputParser(pydantic_object=InventionAnalysis)

        try:
            result = await self._invoke_llm_structured(
                prompt,
                {"disclosure": disclosure},
                parser,
            )

            # Handle both dict and Pydantic model returns
            if isinstance(result, dict):
                return InventionAnalysis(**result)
            return result

        except Exception as e:
            self._logger.error("analysis_failed", error=str(e))
            # Return minimal analysis on failure
            return InventionAnalysis(
                invention_title="분석 실패",
                core_technology="발명 신고서 분석 중 오류가 발생했습니다.",
                key_features=[],
                technical_problems=[],
                solutions=[],
                expected_effects=[],
                tech_field="미분류",
                follow_up_questions=["발명 신고서를 다시 제공해주세요."],
            )

    async def process_e2(
        self,
        state: SpecWritingState,
        additional_info: str,
    ) -> dict[str, Any]:
        """Process additional information provided in E2 step.

        Args:
            state: Current workflow state
            additional_info: Additional information from inventor

        Returns:
            Updated state with refined analysis
        """
        self._logger.info("processing_additional_info")

        # Combine original disclosure with additional info
        original_disclosure = state.get("invention_disclosure", "")
        combined_disclosure = f"""{original_disclosure}

## 추가 정보
{additional_info}"""

        # Re-analyze with additional information
        analysis = await self._analyze_disclosure(combined_disclosure)

        return {
            "invention_title": analysis.invention_title,
            "core_technology": analysis.core_technology,
            "key_features": analysis.key_features,
            "technical_problems": analysis.technical_problems,
            "solutions": analysis.solutions,
            "expected_effects": analysis.expected_effects,
            "tech_field": analysis.tech_field,
            "follow_up_questions": [],  # Clear questions after E2
            "current_step": "E3",
        }
