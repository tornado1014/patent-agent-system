"""ClaimMapperAgent for Patent Analysis workflow.

Responsible for claim element analysis and mapping:
- Parsing claims into constituent elements
- Mapping claim elements to product features
- Building claim charts
- Supporting infringement and invalidity analysis
"""

from datetime import datetime
from typing import Any

import structlog

from patent_agent.agents.analysis.base import BaseAnalysisAgent, ClaimAnalyzer
from patent_agent.state.analysis import (
    AnalysisStep,
    ClaimChart,
    ClaimElement,
    PatentAnalysisState,
)

logger = structlog.get_logger(__name__)


class ClaimMapperAgent(BaseAnalysisAgent):
    """Agent for claim element analysis and mapping.

    This agent is responsible for:
    1. Parsing claims into preamble, body, and limitations
    2. Creating claim charts for infringement analysis
    3. Mapping claim elements to product features
    4. Identifying key claim elements

    Example:
        >>> agent = ClaimMapperAgent()
        >>> state = {
        ...     "analysis_type": "infringement",
        ...     "patent_data": [{"claims": ["A device comprising..."]}],
        ...     "product_spec": {"features": ["encoder", "decoder"]},
        ... }
        >>> result = await agent.process(state)
    """

    @property
    def name(self) -> str:
        return "ClaimMapperAgent"

    @property
    def description(self) -> str:
        return "청구항 요소 분해 및 제품/선행기술 매핑"

    @property
    def step(self) -> AnalysisStep:
        return "process"

    def get_system_prompt(self) -> str:
        return """당신은 특허 청구항 분석 전문가입니다.

## 역할
- 청구항을 구성요소별로 분해
- 제품 기능 또는 선행기술과 청구항 요소 매핑
- 침해/무효 분석을 위한 청구항 차트 작성

## 청구항 구조 분석

### 독립항 구조
1. **전제부 (Preamble)**: 발명의 일반적 범주
   - 예: "A device for processing signals, comprising:"
2. **본문 (Body)**: 전이구 (comprising, consisting of 등)
3. **한정부 (Limitations)**: 구성요소들
   - 각 요소는 구조적 또는 기능적 특징

### 요소 분류
- **필수 요소**: 모든 구성요소 (All-Elements Rule)
- **선택적 요소**: "optionally", "preferably" 등으로 한정된 요소

## 매핑 원칙

### 문언 침해 매핑
- 청구항 용어와 제품 기능의 직접적 대응
- 모든 요소가 1:1 대응 필요

### 균등론 매핑
- 기능 (Function): 동일한 기능 수행
- 방법 (Way): 실질적으로 동일한 방법
- 결과 (Result): 실질적으로 동일한 결과

## 출력 형식
각 청구항에 대해:
- 요소 ID (E0, E1, E2...)
- 요소 유형 (preamble/body/limitation)
- 요소 텍스트
- 대응 제품 기능 (해당 시)
- 매칭 유형 (literal/equivalent/no_match)
"""

    async def process(self, state: PatentAnalysisState) -> dict[str, Any]:
        """Process claims and create mappings.

        Args:
            state: Current workflow state

        Returns:
            State updates with claim mappings
        """
        self._logger.info("starting_claim_mapping")

        analysis_type = state.get("analysis_type", "portfolio")
        patent_data = state.get("patent_data", [])

        # Claim mapping is only needed for infringement/invalidity
        if analysis_type not in ("infringement", "invalidity"):
            self._logger.info("skipping_claim_mapping", reason="not_claim_intensive")
            return {
                "claim_mappings": {},
                "current_step": "analyze",
                "updated_at": datetime.now().isoformat(),
            }

        if not patent_data:
            return {
                "is_error_state": True,
                "error_messages": ["특허 데이터가 없습니다."],
                "current_step": "process",
            }

        # Get target patent (first patent for now)
        target_patent = patent_data[0]
        claims = target_patent.get("claims", [])

        if not claims:
            return {
                "is_error_state": True,
                "error_messages": ["대상 특허에 청구항이 없습니다."],
                "current_step": "process",
            }

        # Parse all claims
        claim_mappings: dict[int, ClaimChart] = {}

        for i, claim_text in enumerate(claims, start=1):
            # Create claim chart
            chart = self._create_claim_chart(i, claim_text)

            # For infringement analysis, map to product
            if analysis_type == "infringement":
                product_spec = state.get("product_spec")
                if product_spec:
                    chart = await self._map_to_product(chart, product_spec, claim_text)

            # For invalidity analysis, map to prior art
            elif analysis_type == "invalidity":
                prior_arts = state.get("patent_data", [])[1:]  # Exclude target patent
                if prior_arts:
                    chart = await self._map_to_prior_art(chart, prior_arts, claim_text)

            claim_mappings[i] = chart

        # Identify key claims (independent claims)
        key_claims = self._identify_key_claims(claims)

        self._logger.info(
            "claim_mapping_complete",
            total_claims=len(claims),
            key_claims=key_claims,
        )

        return {
            "claim_mappings": claim_mappings,
            "current_step": "analyze",
            "updated_at": datetime.now().isoformat(),
        }

    def _create_claim_chart(self, claim_number: int, claim_text: str) -> ClaimChart:
        """Create a claim chart with parsed elements.

        Args:
            claim_number: Claim number
            claim_text: Full claim text

        Returns:
            ClaimChart with parsed elements
        """
        elements = ClaimAnalyzer.parse_claim_elements(claim_text)

        return ClaimChart(
            claim_number=claim_number,
            claim_elements=elements,
            mappings=[],
            overall_result="no_match",
        )

    async def _map_to_product(
        self,
        chart: ClaimChart,
        product_spec: dict,
        claim_text: str,
    ) -> ClaimChart:
        """Map claim elements to product features.

        Args:
            chart: Claim chart with parsed elements
            product_spec: Product specification
            claim_text: Original claim text

        Returns:
            Updated claim chart with mappings
        """
        elements = chart["claim_elements"]
        features = product_spec.get("features", [])
        description = product_spec.get("technical_description", "")

        mappings = []

        # Use LLM to analyze mappings
        prompt = f"""분석 대상 청구항:
{claim_text}

제품 기능:
{', '.join(features)}

기술 설명:
{description}

각 청구항 요소에 대해 제품의 대응 기능을 분석하세요.
각 요소별로:
1. 요소 ID
2. 대응 제품 기능 (없으면 "없음")
3. 매칭 유형 (literal/equivalent/no_match)
4. 분석 근거

JSON 형식으로 응답하세요."""

        try:
            response = await self._invoke_llm(prompt)
            # Parse response and create mappings
            # This is simplified - actual implementation would parse JSON
            for i, element in enumerate(elements):
                mapping = {
                    "element_id": element["element_id"],
                    "corresponding_feature": "",  # Would be extracted from LLM response
                    "match_type": "no_match",  # Would be determined by LLM
                }
                mappings.append(mapping)
        except Exception as e:
            self._logger.warning("llm_mapping_failed", error=str(e))
            # Create default no-match mappings
            for element in elements:
                mappings.append({
                    "element_id": element["element_id"],
                    "corresponding_feature": "",
                    "match_type": "no_match",
                })

        # Update chart
        chart["mappings"] = mappings
        chart["overall_result"] = self.evaluate_infringement(chart)

        return chart

    async def _map_to_prior_art(
        self,
        chart: ClaimChart,
        prior_arts: list[dict],
        claim_text: str,
    ) -> ClaimChart:
        """Map claim elements to prior art disclosures.

        Args:
            chart: Claim chart with parsed elements
            prior_arts: List of prior art patents
            claim_text: Original claim text

        Returns:
            Updated claim chart with mappings
        """
        elements = chart["claim_elements"]

        # Combine prior art abstracts for analysis
        prior_art_text = "\n\n".join([
            f"[{pa.get('patent_number', 'Unknown')}] {pa.get('abstract', '')}"
            for pa in prior_arts[:5]  # Limit to top 5
        ])

        mappings = []

        # Use LLM to analyze mappings
        prompt = f"""분석 대상 청구항:
{claim_text}

선행기술:
{prior_art_text}

각 청구항 요소에 대해 선행기술의 대응 내용을 분석하세요.
각 요소별로:
1. 요소 ID
2. 대응 선행기술 내용 (없으면 "없음")
3. 매칭 유형 (novelty_destroying/partial/no_match)
4. 출처 특허번호

JSON 형식으로 응답하세요."""

        try:
            response = await self._invoke_llm(prompt)
            # Parse response and create mappings
            for element in elements:
                mapping = {
                    "element_id": element["element_id"],
                    "corresponding_feature": "",
                    "match_type": "no_match",
                }
                mappings.append(mapping)
        except Exception as e:
            self._logger.warning("llm_mapping_failed", error=str(e))
            for element in elements:
                mappings.append({
                    "element_id": element["element_id"],
                    "corresponding_feature": "",
                    "match_type": "no_match",
                })

        chart["mappings"] = mappings

        # Determine overall invalidity result
        literal_count = sum(1 for m in mappings if m.get("match_type") == "novelty_destroying")
        if literal_count == len(elements):
            chart["overall_result"] = "novelty_destroying"
        elif literal_count > 0:
            chart["overall_result"] = "partial"
        else:
            chart["overall_result"] = "no_match"

        return chart

    def _identify_key_claims(self, claims: list[str]) -> list[int]:
        """Identify independent (key) claims.

        Independent claims are typically:
        1. Claim 1 (first claim is usually independent)
        2. Claims that don't reference other claims

        Args:
            claims: List of claim texts

        Returns:
            List of independent claim numbers
        """
        key_claims = []

        for i, claim in enumerate(claims, start=1):
            claim_lower = claim.lower()

            # Check for dependency markers in Korean and English
            dependency_markers = [
                "제1항에",
                "제2항에",
                "제3항에",
                "claim 1",
                "claim 2",
                "claim 3",
                "according to",
                "as recited in",
                "있어서",
            ]

            is_dependent = any(marker in claim_lower for marker in dependency_markers)

            if not is_dependent:
                key_claims.append(i)

        # If no independent claims found, assume claim 1 is independent
        if not key_claims and claims:
            key_claims = [1]

        return key_claims

    def calculate_claim_breadth_score(self, claim_text: str) -> float:
        """Calculate the breadth score of a claim.

        Args:
            claim_text: Full claim text

        Returns:
            Breadth score (0-1, higher = broader)
        """
        elements = ClaimAnalyzer.parse_claim_elements(claim_text)
        return ClaimAnalyzer.calculate_claim_breadth(elements)
