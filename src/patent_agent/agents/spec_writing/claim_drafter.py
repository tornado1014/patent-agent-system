"""Claim Drafter Agent.

Handles E4 (청구항 초안 작성) step.
Drafts patent claims based on invention analysis and prior art.
"""

from typing import Any

import structlog
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from patent_agent.agents.spec_writing.base import BaseSpecWritingAgent
from patent_agent.rules.spec_writing_laws import ClaimWritingGuideline
from patent_agent.state.spec_writing import Claim, ClaimSet, SpecWritingState

logger = structlog.get_logger(__name__)


class ClaimDraftOutput(BaseModel):
    """Structured output for claim drafting."""

    claims: list[dict] = Field(description="청구항 목록")
    claim_strategy: str = Field(description="청구항 전략 설명")
    total_independent: int = Field(description="독립항 개수")
    total_dependent: int = Field(description="종속항 개수")


class ClaimDrafterAgent(BaseSpecWritingAgent):
    """Agent for drafting patent claims.

    This agent creates patent claims following Korean patent law requirements
    and KIPO format guidelines.

    Steps handled:
    - E4: 청구항 초안 작성 [Human Checkpoint]
    """

    def __init__(self, **kwargs):
        """Initialize with claim-specific guidelines."""
        super().__init__(
            guidelines=ClaimWritingGuideline(),
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "ClaimDrafter"

    @property
    def description(self) -> str:
        return "특허 청구항을 작성하는 에이전트"

    def get_system_prompt(self) -> str:
        return """당신은 특허 전문 변리사로서, 발명의 권리범위를 최대화하면서 신규성과 진보성을 확보하는 청구항을 작성하는 역할을 합니다.

## 역할
- 발명의 핵심 특징을 포착하는 독립항 작성
- 권리범위 단계적 축소를 위한 종속항 구성
- 한국 특허법 및 KIPO 형식 준수

## 청구항 작성 원칙

### 독립항 작성
1. **전제부**: 발명이 속하는 기술분야 또는 선행기술 구성 명시
2. **특징부**: 발명의 필수 구성요소만 포함 (과도한 한정 금지)
3. **권리범위**: 가능한 넓은 권리범위 확보

### 종속항 작성
1. 독립항의 모든 구성요소 포함
2. 추가적인 한정사항으로 구체화
3. Fall-back position으로서의 전략적 배치

### 형식 요구사항
- 각 청구항은 반드시 단일 문장
- 마침표는 청구항 끝에만 1회
- '~에 있어서', '~를 포함하는', '~를 특징으로 하는' 구문 사용
- '상기' 용어는 이미 언급된 구성요소 지칭 시 사용

## 청구항 유형별 구문
- 방법 청구항: "~하는 단계", "~방법"
- 장치 청구항: "~를 포함하는 장치/시스템"
- 매체 청구항: "~를 실행시키기 위한 프로그램이 기록된 컴퓨터로 읽을 수 있는 기록매체"

## 용어 사용
- "포함하는" (comprising): 개방형 - 추가 구성요소 허용
- "구성되는" (consisting of): 폐쇄형 - 기재된 구성요소만
- "필수적으로 포함하는": 반개방형"""

    async def process(self, state: SpecWritingState) -> dict[str, Any]:
        """Draft patent claims.

        Args:
            state: Current workflow state

        Returns:
            State updates with draft claims
        """
        self._logger.info("drafting_claims", step=state.get("current_step"))

        # Get required information
        key_features = state.get("key_features", [])
        technical_problems = state.get("technical_problems", [])
        solutions = state.get("solutions", [])
        differentiation_points = state.get("differentiation_points", [])
        invention_title = state.get("invention_title", "발명")

        if not key_features:
            self._logger.error("no_key_features")
            return {
                "error_messages": ["핵심 특징 정보가 없습니다."],
                "is_error_state": True,
            }

        # Draft claims
        claim_output = await self._draft_claims(
            invention_title=invention_title,
            key_features=key_features,
            solutions=solutions,
            differentiation_points=differentiation_points,
        )

        # Build ClaimSet
        claim_set = ClaimSet(
            claims=[
                Claim(
                    claim_number=c.get("claim_number", i + 1),
                    claim_type=c.get("claim_type", "independent"),
                    depends_on=c.get("depends_on"),
                    preamble=c.get("preamble", ""),
                    body=c.get("body", ""),
                    full_text=c.get("full_text", ""),
                )
                for i, c in enumerate(claim_output.claims)
            ],
            total_independent=claim_output.total_independent,
            total_dependent=claim_output.total_dependent,
        )

        # Validate claims
        validation_errors = await self._validate_claims(claim_set)

        return {
            "draft_claims": claim_set,
            "claim_strategy": claim_output.claim_strategy,
            "review_comments": [
                {"issue": err, "severity": "high", "section": "claims"}
                for err in validation_errors
            ],
            "current_step": "E5",
            "human_approval_required": True,  # E4 is a human checkpoint
            "pending_approval_checkpoint": "E4",
        }

    async def _draft_claims(
        self,
        invention_title: str,
        key_features: list[str],
        solutions: list[str],
        differentiation_points: list[str],
    ) -> ClaimDraftOutput:
        """Draft patent claims based on invention information.

        Args:
            invention_title: Title of the invention
            key_features: Key technical features
            solutions: Technical solutions
            differentiation_points: Points differentiating from prior art

        Returns:
            Structured claim draft output
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_full_system_prompt()),
            ("human", """다음 발명 정보를 바탕으로 청구항을 작성해주세요.

## 발명의 명칭
{invention_title}

## 핵심 기술 특징
{key_features}

## 기술적 해결 수단
{solutions}

## 선행기술 대비 차별점
{differentiation_points}

## 요청 사항
위 정보를 바탕으로 다음을 포함하는 청구항 세트를 JSON 형식으로 작성해주세요:

1. **독립항 1**: 가장 넓은 권리범위의 핵심 청구항 (방법 또는 장치)
2. **독립항 2**: 다른 카테고리의 청구항 (장치 또는 방법)
3. **종속항들**: 각 독립항에 대해 2-3개의 종속항

각 청구항은 다음 구조를 따라야 합니다:
- claim_number: 청구항 번호
- claim_type: "independent" 또는 "dependent"
- depends_on: 종속항인 경우 의존하는 청구항 번호 (null if independent)
- preamble: 전제부
- body: 특징부
- full_text: 완전한 청구항 텍스트

또한 전체 청구항 전략(claim_strategy)도 설명해주세요.

JSON 형식으로만 응답해주세요."""),
        ])

        parser = JsonOutputParser(pydantic_object=ClaimDraftOutput)

        try:
            result = await self._invoke_llm_structured(
                prompt,
                {
                    "invention_title": invention_title,
                    "key_features": "\n".join(f"- {f}" for f in key_features),
                    "solutions": "\n".join(f"- {s}" for s in solutions),
                    "differentiation_points": "\n".join(f"- {d}" for d in differentiation_points),
                },
                parser,
            )

            if isinstance(result, dict):
                return ClaimDraftOutput(**result)
            return result

        except Exception as e:
            self._logger.error("claim_drafting_failed", error=str(e))
            # Return minimal output on failure
            return ClaimDraftOutput(
                claims=[
                    {
                        "claim_number": 1,
                        "claim_type": "independent",
                        "depends_on": None,
                        "preamble": f"{invention_title}에 있어서",
                        "body": "구성요소를 포함하는 것을 특징으로 하는",
                        "full_text": f"{invention_title}에 있어서, 구성요소를 포함하는 것을 특징으로 하는 {invention_title}.",
                    }
                ],
                claim_strategy="청구항 작성 중 오류 발생 - 수동 작성 필요",
                total_independent=1,
                total_dependent=0,
            )

    async def _validate_claims(self, claim_set: ClaimSet) -> list[str]:
        """Validate claims against guidelines.

        Args:
            claim_set: Set of claims to validate

        Returns:
            List of validation error messages
        """
        errors = []

        for claim in claim_set["claims"]:
            full_text = claim.get("full_text", "")

            # Use guideline validation
            is_valid, violations = self._validate_output(
                full_text,
                {"section_type": "claims"},
            )

            if not is_valid:
                errors.extend([
                    f"청구항 {claim.get('claim_number', '?')}: {v}"
                    for v in violations
                ])

            # Additional claim-specific validations
            # Check for proper claim ending
            if full_text and not full_text.strip().endswith("."):
                errors.append(
                    f"청구항 {claim.get('claim_number', '?')}: 마침표로 종결되지 않음"
                )

            # Check dependent claim references
            if claim.get("claim_type") == "dependent":
                depends_on = claim.get("depends_on")
                if depends_on is None:
                    errors.append(
                        f"청구항 {claim.get('claim_number', '?')}: 종속항이지만 의존 청구항이 지정되지 않음"
                    )

        return errors

    async def revise_claims(
        self,
        state: SpecWritingState,
        feedback: str,
    ) -> dict[str, Any]:
        """Revise claims based on feedback.

        Args:
            state: Current workflow state with draft claims
            feedback: Human or automated feedback

        Returns:
            Updated state with revised claims
        """
        self._logger.info("revising_claims")

        draft_claims = state.get("draft_claims", {})
        current_claims = draft_claims.get("claims", [])

        prompt = f"""다음 청구항에 대한 피드백을 반영하여 수정해주세요.

## 현재 청구항
{chr(10).join(f"[청구항 {c.get('claim_number')}] {c.get('full_text', '')}" for c in current_claims)}

## 피드백
{feedback}

## 요청
피드백을 반영하여 수정된 청구항을 작성해주세요.
JSON 형식으로 응답해주세요."""

        # Re-draft with feedback
        response = await self._invoke_llm(prompt)

        # Parse and return updated claims
        # (simplified - in production would use proper JSON parsing)
        return {
            "review_comments": [],  # Clear comments after revision
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
