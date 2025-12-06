"""InfringementCheckerAgent for Patent Analysis workflow.

Responsible for patent infringement analysis:
- Literal infringement assessment
- Doctrine of equivalents analysis
- Design-around options
- Risk assessment and recommendations
"""

from datetime import datetime
from typing import Any

import structlog

from patent_agent.agents.analysis.base import BaseAnalysisAgent
from patent_agent.state.analysis import (
    AnalysisStep,
    ClaimChart,
    InfringementAnalysis,
    PatentAnalysisState,
    ProductSpec,
)
from patent_agent.state.base import ConfidenceLevel

logger = structlog.get_logger(__name__)


class InfringementCheckerAgent(BaseAnalysisAgent):
    """Agent for patent infringement analysis.

    This agent is responsible for:
    1. Evaluating literal infringement
    2. Assessing doctrine of equivalents
    3. Identifying design-around options
    4. Providing risk assessment and recommendations

    Example:
        >>> agent = InfringementCheckerAgent()
        >>> state = {
        ...     "analysis_type": "infringement",
        ...     "claim_mappings": {...},
        ...     "product_spec": {...},
        ... }
        >>> result = await agent.process(state)
    """

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "high": 0.7,  # >70% of claims infringed
        "medium": 0.3,  # 30-70%
        "low": 0.0,  # <30%
    }

    @property
    def name(self) -> str:
        return "InfringementCheckerAgent"

    @property
    def description(self) -> str:
        return "특허 침해 분석 및 리스크 평가"

    @property
    def step(self) -> AnalysisStep:
        return "analyze"

    def get_system_prompt(self) -> str:
        return """당신은 특허 침해 분석 전문가입니다.

## 역할
- 문언 침해 (Literal Infringement) 평가
- 균등론 (Doctrine of Equivalents) 분석
- 회피 설계 (Design-Around) 옵션 제시
- 침해 리스크 평가 및 권고사항 도출

## 침해 분석 프레임워크

### 1. All-Elements Rule
- 청구항의 모든 구성요소가 대상 제품에 존재해야 문언 침해 성립
- 하나의 요소라도 빠지면 문언 침해 성립 안 함

### 2. 문언 침해 (Literal Infringement)
- 청구항 용어와 제품 기능의 직접적 대응
- 용어의 명확한 1:1 매칭 필요

### 3. 균등론 (Doctrine of Equivalents)
**기능-방법-결과 (FWR) 테스트**:
- Function: 실질적으로 동일한 기능
- Way: 실질적으로 동일한 방법
- Result: 실질적으로 동일한 결과

**제한 원칙**:
- 출원 경과 금반언 (Prosecution History Estoppel)
- 선행 기술 제한 (Prior Art Limitation)
- 공지 기술 제외 (All Limitations Rule)

### 4. 리스크 평가

| 리스크 수준 | 기준 | 권고 조치 |
|------------|------|----------|
| 높음 | 독립항 침해 명확 | 설계 변경 또는 라이선스 |
| 중간 | 일부 청구항 침해 가능 | 상세 분석 후 결정 |
| 낮음 | 침해 가능성 낮음 | 모니터링 |
| 없음 | 침해 요소 없음 | 현상 유지 |

### 5. 회피 설계 고려사항
- 핵심 요소 vs 부수적 요소 구분
- 대체 기술 가용성
- 비용/시간 고려
- 제품 성능 영향

## 출력 형식
1. 침해 평가 요약
2. 청구항별 상세 분석
3. 리스크 수준 및 근거
4. 회피 설계 옵션
5. 권고사항
"""

    async def process(self, state: PatentAnalysisState) -> dict[str, Any]:
        """Perform infringement analysis.

        Args:
            state: Current workflow state

        Returns:
            State updates with infringement analysis
        """
        self._logger.info("starting_infringement_analysis")

        claim_mappings = state.get("claim_mappings", {})
        product_spec = state.get("product_spec")
        patent_data = state.get("patent_data", [])

        if not claim_mappings:
            return {
                "is_error_state": True,
                "error_messages": ["청구항 매핑 데이터가 없습니다."],
                "current_step": "analyze",
            }

        # Get target patent
        target_patent = patent_data[0] if patent_data else {}
        target_patent_number = target_patent.get("patent_number", "Unknown")

        # Analyze each claim
        claim_analyses = []
        literal_infringement_count = 0
        doe_infringement_count = 0

        for claim_num, chart in claim_mappings.items():
            analysis = self._analyze_claim_infringement(chart)
            claim_analyses.append(analysis)

            if analysis["infringement_type"] == "literal":
                literal_infringement_count += 1
            elif analysis["infringement_type"] == "doctrine_of_equivalents":
                doe_infringement_count += 1

        # Determine overall infringement
        total_claims = len(claim_mappings)
        literal_infringement = literal_infringement_count > 0
        doe_infringement = doe_infringement_count > 0 and not literal_infringement

        # Calculate risk level
        overall_risk = self._calculate_risk_level(
            literal_infringement_count,
            doe_infringement_count,
            total_claims,
        )

        # Generate design-around options
        design_around_options = await self._generate_design_around_options(
            claim_mappings, product_spec
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_risk, literal_infringement, doe_infringement
        )

        # Build analysis result
        infringement_analysis = InfringementAnalysis(
            target_patent=target_patent_number,
            target_product=product_spec.get("product_name", "Unknown") if product_spec else "Unknown",
            claim_charts=list(claim_mappings.values()),
            literal_infringement=literal_infringement,
            doctrine_of_equivalents=doe_infringement,
            overall_risk=overall_risk,
            design_around_options=design_around_options,
            recommendations=recommendations,
        )

        self._logger.info(
            "infringement_analysis_complete",
            overall_risk=overall_risk,
            literal_infringement=literal_infringement,
            doe_infringement=doe_infringement,
        )

        return {
            "infringement_analysis": infringement_analysis,
            "current_step": "report",
            "updated_at": datetime.now().isoformat(),
        }

    def _analyze_claim_infringement(self, chart: ClaimChart) -> dict:
        """Analyze infringement for a single claim.

        Args:
            chart: Claim chart with mappings

        Returns:
            Claim analysis result
        """
        elements = chart.get("claim_elements", [])
        mappings = chart.get("mappings", [])

        if not elements:
            return {
                "claim_number": chart.get("claim_number", 0),
                "infringement_type": "none",
                "matched_elements": 0,
                "total_elements": 0,
                "confidence": "불확실",
            }

        # Count matches by type
        literal_matches = 0
        equivalent_matches = 0
        no_matches = 0

        for mapping in mappings:
            match_type = mapping.get("match_type", "no_match")
            if match_type == "literal":
                literal_matches += 1
            elif match_type == "equivalent":
                equivalent_matches += 1
            else:
                no_matches += 1

        total = len(elements)

        # Determine infringement type
        if literal_matches == total:
            infringement_type = "literal"
            confidence = "확실"
        elif literal_matches + equivalent_matches == total:
            infringement_type = "doctrine_of_equivalents"
            confidence = "가능"
        elif literal_matches + equivalent_matches > total * 0.5:
            infringement_type = "partial"
            confidence = "가능"
        else:
            infringement_type = "none"
            confidence = "확실"

        return {
            "claim_number": chart.get("claim_number", 0),
            "infringement_type": infringement_type,
            "literal_matches": literal_matches,
            "equivalent_matches": equivalent_matches,
            "matched_elements": literal_matches + equivalent_matches,
            "total_elements": total,
            "confidence": confidence,
        }

    def _calculate_risk_level(
        self,
        literal_count: int,
        doe_count: int,
        total_claims: int,
    ) -> str:
        """Calculate overall risk level.

        Args:
            literal_count: Number of literally infringed claims
            doe_count: Number of DOE infringed claims
            total_claims: Total number of claims

        Returns:
            Risk level string
        """
        if total_claims == 0:
            return "none"

        # Any literal infringement of independent claims = high risk
        if literal_count > 0:
            return "high"

        # DOE infringement
        if doe_count > 0:
            ratio = doe_count / total_claims
            if ratio >= self.RISK_THRESHOLDS["high"]:
                return "high"
            elif ratio >= self.RISK_THRESHOLDS["medium"]:
                return "medium"
            else:
                return "low"

        return "none"

    async def _generate_design_around_options(
        self,
        claim_mappings: dict[int, ClaimChart],
        product_spec: ProductSpec | None,
    ) -> list[str]:
        """Generate design-around options.

        Args:
            claim_mappings: Claim charts with mappings
            product_spec: Product specification

        Returns:
            List of design-around suggestions
        """
        options = []

        # Identify key elements causing infringement
        problematic_elements = []
        for claim_num, chart in claim_mappings.items():
            for mapping in chart.get("mappings", []):
                if mapping.get("match_type") in ("literal", "equivalent"):
                    element_id = mapping.get("element_id", "")
                    # Find element text
                    for element in chart.get("claim_elements", []):
                        if element.get("element_id") == element_id:
                            problematic_elements.append({
                                "claim": claim_num,
                                "element_id": element_id,
                                "text": element.get("text", ""),
                            })

        if not problematic_elements:
            options.append("침해 요소가 식별되지 않아 설계 변경이 필요하지 않습니다.")
            return options

        # Use LLM to generate design-around suggestions
        element_summary = "\n".join([
            f"- 청구항 {e['claim']}, 요소 {e['element_id']}: {e['text'][:100]}"
            for e in problematic_elements[:5]
        ])

        prompt = f"""다음 특허 청구항 요소들에 대한 회피 설계 옵션을 제안하세요:

{element_summary}

각 요소에 대해:
1. 대체 기술 또는 구현 방법
2. 예상 난이도 (상/중/하)
3. 제품 성능에 미치는 영향

실질적이고 구체적인 제안을 해주세요."""

        try:
            response = await self._invoke_llm(prompt)
            # Parse response into options
            options = [
                line.strip()
                for line in response.split("\n")
                if line.strip() and len(line.strip()) > 10
            ][:5]
        except Exception as e:
            self._logger.warning("llm_design_around_failed", error=str(e))
            options = [
                "문제 요소의 구조적 변경 검토",
                "대체 기술 조사 권장",
                "기능적 등가물 회피 분석 필요",
            ]

        return options

    def _generate_recommendations(
        self,
        risk_level: str,
        literal: bool,
        doe: bool,
    ) -> list[str]:
        """Generate recommendations based on analysis.

        Args:
            risk_level: Overall risk level
            literal: Whether literal infringement found
            doe: Whether DOE infringement found

        Returns:
            List of recommendations
        """
        recommendations = []

        if risk_level == "high":
            if literal:
                recommendations.extend([
                    "🔴 즉각적인 법률 자문 권장",
                    "제품 설계 변경 우선 검토",
                    "라이선스 협상 가능성 탐색",
                    "출시 전 설계 변경 완료 필수",
                ])
            else:
                recommendations.extend([
                    "🟡 법률 자문 권장",
                    "균등론 적용 범위 상세 분석 필요",
                    "설계 변경 옵션 평가",
                ])
        elif risk_level == "medium":
            recommendations.extend([
                "상세 청구항 분석 권장",
                "회피 설계 옵션 평가",
                "특허 유효성 조사 고려",
                "지속적인 모니터링 필요",
            ])
        elif risk_level == "low":
            recommendations.extend([
                "현재 설계 유지 가능",
                "정기적인 특허 동향 모니터링 권장",
                "관련 특허 출원 고려",
            ])
        else:
            recommendations.extend([
                "침해 리스크 없음으로 판단",
                "제품 출시/판매 진행 가능",
                "경쟁 특허 동향 지속 모니터링 권장",
            ])

        return recommendations
