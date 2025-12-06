"""Claim Comparer Agent for prior art search.

Generates detailed comparison tables between target claims and prior art.
"""

from datetime import datetime
from typing import Any

from patent_agent.agents.prior_art.base import BasePriorArtAgent
from patent_agent.state.prior_art import (
    AnalyzedReference,
    ClaimComparison,
    PriorArtSearchState,
    PriorArtStep,
)


class ClaimComparerAgent(BasePriorArtAgent):
    """Agent for generating claim comparison tables.

    Creates detailed element-by-element comparisons between
    target claims and analyzed prior art references.
    """

    @property
    def name(self) -> str:
        return "ClaimComparer"

    @property
    def description(self) -> str:
        return "청구항 vs 선행기술 대비표 생성"

    @property
    def step(self) -> PriorArtStep:
        return "analyze"

    def get_system_prompt(self) -> str:
        return """당신은 특허 청구항 분석 전문가입니다.
청구항의 각 구성요소를 선행기술과 상세히 대비합니다.

## 대비 분석 원칙

### 1. 구성요소 분해
- 각 청구항을 구성요소 단위로 분해
- 독립항과 종속항 구분
- 한정사항(limitation) 명확히 식별

### 2. 대비 결과 분류
- identical: 동일 (완전 일치)
- similar: 유사 (균등 범위)
- different: 상이 (명확한 차이)
- not_found: 미개시 (선행기술에 없음)

### 3. 대비표 형식
| 구성 | 본원 청구항 | 선행기술 | 대비 결과 | 비고 |
|------|-------------|----------|----------|------|
| 1-1  | A를 포함    | A' 개시  | similar  | ... |
| 1-2  | B와 결합    | 미개시   | not_found| ... |

## 분석 기준
- 문언적 동일성 먼저 검토
- 균등론 적용 가능성 검토
- 기술적 의의 차이 분석
"""

    async def process(self, state: PriorArtSearchState) -> dict[str, Any]:
        """Generate claim comparison tables.

        Args:
            state: Current workflow state

        Returns:
            State updates with comparison data
        """
        target_claims = state.get("target_claims", [])
        analyzed_refs = state.get("analyzed_references", [])

        if not target_claims:
            return {
                "error_messages": ["대상 청구항이 없습니다."],
                "is_error_state": True,
            }

        if not analyzed_refs:
            return {
                "error_messages": ["분석된 선행기술이 없습니다."],
                "is_error_state": True,
            }

        self._logger.info(
            "generating_comparison_tables",
            num_claims=len(target_claims),
            num_references=len(analyzed_refs),
        )

        # Limit to top references for detailed comparison
        top_refs = analyzed_refs[:5]

        # Generate comparisons
        comparisons: list[ClaimComparison] = []

        for claim_num, claim in enumerate(target_claims, 1):
            # Break claim into elements
            elements = await self._extract_claim_elements(claim, claim_num)

            for element_id, element_text in elements:
                for ref in top_refs:
                    comparison = await self._compare_element(
                        claim_num=claim_num,
                        element_id=element_id,
                        element_text=element_text,
                        reference=ref,
                    )
                    comparisons.append(comparison)

        # Generate markdown table
        comparison_table = self._generate_markdown_table(
            comparisons,
            target_claims,
            top_refs,
        )

        self._logger.info(
            "comparison_complete",
            num_comparisons=len(comparisons),
        )

        return {
            "claim_comparisons": comparisons,
            "comparison_table": comparison_table,
            "current_step": "report",
            "updated_at": datetime.now().isoformat(),
        }

    async def _extract_claim_elements(
        self,
        claim: str,
        claim_num: int,
    ) -> list[tuple[str, str]]:
        """Extract elements from a claim.

        Args:
            claim: Claim text
            claim_num: Claim number

        Returns:
            List of (element_id, element_text) tuples
        """
        prompt = f"""다음 청구항을 구성요소 단위로 분해하세요.

## 청구항 {claim_num}
{claim}

## 요구사항
- 각 구성요소를 명확히 분리
- 전제부(preamble)와 본문(body) 구분
- 각 한정사항(limitation) 별도 항목으로

## 출력 형식 (각 요소 한 줄)
{claim_num}-1: [요소 내용]
{claim_num}-2: [요소 내용]
...
"""

        try:
            response = await self._invoke_llm(prompt)

            elements = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if ":" in line and line.startswith(str(claim_num)):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        element_id = parts[0].strip()
                        element_text = parts[1].strip()
                        if element_text:
                            elements.append((element_id, element_text))

            if not elements:
                # Fallback: treat whole claim as single element
                elements = [(f"{claim_num}-1", claim)]

            return elements[:10]  # Limit elements

        except Exception:
            return [(f"{claim_num}-1", claim)]

    async def _compare_element(
        self,
        claim_num: int,
        element_id: str,
        element_text: str,
        reference: AnalyzedReference,
    ) -> ClaimComparison:
        """Compare a claim element with a reference.

        Args:
            claim_num: Claim number
            element_id: Element identifier
            element_text: Element text
            reference: Reference to compare against

        Returns:
            ClaimComparison result
        """
        ref_result = reference.get("patent_result", {})
        ref_text = f"{ref_result.get('title', '')} {ref_result.get('abstract', '')}"
        ref_id = reference.get("reference_id", "")
        doc_number = ref_result.get("document_number", "")

        prompt = f"""다음 청구항 구성요소가 선행기술에 개시되어 있는지 분석하세요.

## 청구항 구성요소
{element_text}

## 선행기술 ({doc_number})
{ref_text[:800]}

## 분석 요구사항
다음 중 하나로 결과를 판단하세요:
- identical: 동일하게 개시됨
- similar: 유사하게 개시됨 (균등 범위)
- different: 상이하게 개시됨
- not_found: 개시되지 않음

## 출력 형식
RESULT: [결과]
DISCLOSURE: [선행기술의 관련 개시 내용, 없으면 "없음"]
NOTES: [비고]
"""

        try:
            response = await self._invoke_llm(prompt)

            result = "not_found"
            disclosure = "없음"
            notes = ""

            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("RESULT:"):
                    result_text = line.split(":", 1)[1].strip().lower()
                    if "identical" in result_text:
                        result = "identical"
                    elif "similar" in result_text:
                        result = "similar"
                    elif "different" in result_text:
                        result = "different"
                    else:
                        result = "not_found"
                elif line.startswith("DISCLOSURE:"):
                    disclosure = line.split(":", 1)[1].strip()
                elif line.startswith("NOTES:"):
                    notes = line.split(":", 1)[1].strip()

            return ClaimComparison(
                claim_number=claim_num,
                claim_element=element_text[:200],
                prior_art_ref=f"{ref_id} ({doc_number})",
                prior_art_disclosure=disclosure[:300],
                comparison_result=result,
                notes=notes[:200],
            )

        except Exception as e:
            return ClaimComparison(
                claim_number=claim_num,
                claim_element=element_text[:200],
                prior_art_ref=f"{ref_id} ({doc_number})",
                prior_art_disclosure="분석 오류",
                comparison_result="not_found",
                notes=str(e)[:100],
            )

    def _generate_markdown_table(
        self,
        comparisons: list[ClaimComparison],
        claims: list[str],
        references: list[AnalyzedReference],
    ) -> str:
        """Generate markdown comparison table.

        Args:
            comparisons: List of comparisons
            claims: Original claims
            references: Analyzed references

        Returns:
            Markdown table string
        """
        lines = [
            "# 청구항 대비표\n",
            "## 대상 청구항\n",
        ]

        # List claims
        for i, claim in enumerate(claims, 1):
            lines.append(f"**청구항 {i}:** {claim[:100]}...")
            lines.append("")

        # Reference summary
        lines.append("\n## 주요 선행기술\n")
        for ref in references:
            ref_result = ref.get("patent_result", {})
            score = ref.get("relevance_score", 0)
            confidence = self.get_confidence_indicator(ref.get("confidence", "불확실"))
            lines.append(
                f"- **{ref.get('reference_id')}** ({ref_result.get('document_number', 'N/A')}): "
                f"{ref_result.get('title', 'N/A')[:50]}... {confidence} ({score:.2f})"
            )

        # Comparison table
        lines.append("\n## 구성요소별 대비\n")
        lines.append("| 구성 | 청구항 요소 | 선행기술 | 개시 내용 | 결과 | 비고 |")
        lines.append("|------|-------------|----------|-----------|------|------|")

        for comp in comparisons[:30]:  # Limit rows
            result_indicator = {
                "identical": "🔴 동일",
                "similar": "🟡 유사",
                "different": "🟢 상이",
                "not_found": "🟢 미개시",
            }.get(comp.get("comparison_result", ""), "?")

            lines.append(
                f"| {comp.get('claim_number', '-')}-* | "
                f"{comp.get('claim_element', '')[:30]}... | "
                f"{comp.get('prior_art_ref', '')} | "
                f"{comp.get('prior_art_disclosure', '')[:30]}... | "
                f"{result_indicator} | "
                f"{comp.get('notes', '')[:20]} |"
            )

        # Summary
        identical_count = sum(
            1 for c in comparisons if c.get("comparison_result") == "identical"
        )
        similar_count = sum(
            1 for c in comparisons if c.get("comparison_result") == "similar"
        )
        not_found_count = sum(
            1 for c in comparisons if c.get("comparison_result") == "not_found"
        )

        lines.append("\n## 대비 결과 요약\n")
        lines.append(f"- 🔴 동일 개시: {identical_count}건")
        lines.append(f"- 🟡 유사 개시: {similar_count}건")
        lines.append(f"- 🟢 미개시/상이: {not_found_count}건")

        return "\n".join(lines)
