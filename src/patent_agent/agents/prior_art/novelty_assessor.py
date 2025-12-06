"""Novelty Assessor Agent for prior art search.

Assesses novelty and inventive step of target claims based on prior art analysis.
Generates final search report with recommendations.
"""

from datetime import datetime
from typing import Any

from patent_agent.agents.prior_art.base import BasePriorArtAgent
from patent_agent.state.prior_art import (
    AnalyzedReference,
    ClaimComparison,
    InventiveStepAssessment,
    NoveltyAssessment,
    PriorArtSearchState,
    PriorArtStep,
)


class NoveltyAssessorAgent(BasePriorArtAgent):
    """Agent for assessing novelty and inventive step.

    Evaluates:
    - Novelty (신규성): Is the claim anticipated by any single reference?
    - Inventive Step (진보성): Can the claim be obviously derived from combinations?

    Generates final report with recommendations.
    """

    @property
    def name(self) -> str:
        return "NoveltyAssessor"

    @property
    def description(self) -> str:
        return "신규성/진보성 평가 및 최종 보고서 생성"

    @property
    def step(self) -> PriorArtStep:
        return "report"

    def get_system_prompt(self) -> str:
        return """당신은 특허 심사 전문가입니다.
선행기술 분석 결과를 바탕으로 신규성과 진보성을 평가합니다.

## 신규성 (특허법 제29조 제1항)

### 신규성 결여 기준
- 청구항의 **모든** 구성요소가 **단일** 선행문헌에 개시
- 명시적 개시 또는 내재적 개시 포함
- 당업자가 직접 도출 가능

### 신규성 인정 기준
- **하나 이상**의 구성요소가 어떤 단일 문헌에도 미개시

## 진보성 (특허법 제29조 제2항)

### 진보성 평가 요소
1. **결합 동기**: 복수 문헌 조합의 시사점이 있는가?
2. **기술적 곤란성**: 조합에 기술적 어려움이 있는가?
3. **예측 불가 효과**: 예측할 수 없는 현저한 효과가 있는가?
4. **저해 요인**: Teaching away가 있는가?

### 진보성 결여 기준
- 복수 선행문헌의 조합으로 용이하게 도출 가능
- 조합의 동기 존재
- 특별한 효과 없음

## 신뢰도 표시
- 🟢 확실: 명확한 근거 있음
- 🟡 가능: 해석에 따라 다름
- 🔴 불확실: 추가 검토 필요
"""

    async def process(self, state: PriorArtSearchState) -> dict[str, Any]:
        """Assess novelty and inventive step, generate final report.

        Args:
            state: Current workflow state

        Returns:
            State updates with assessments and report
        """
        target_claims = state.get("target_claims", [])
        analyzed_refs = state.get("analyzed_references", [])
        comparisons = state.get("claim_comparisons", [])
        target_invention = state.get("target_invention", "")
        search_type = state.get("search_type", "novelty_search")

        if not target_claims:
            return {
                "error_messages": ["대상 청구항이 없습니다."],
                "is_error_state": True,
            }

        self._logger.info(
            "assessing_patentability",
            num_claims=len(target_claims),
            num_references=len(analyzed_refs),
            search_type=search_type,
        )

        # Assess each claim
        novelty_assessments: list[NoveltyAssessment] = []
        inventive_step_assessments: list[InventiveStepAssessment] = []

        for claim_num, claim in enumerate(target_claims, 1):
            # Get comparisons for this claim
            claim_comparisons = [
                c for c in comparisons if c.get("claim_number") == claim_num
            ]

            # Assess novelty
            novelty = await self._assess_novelty(
                claim_num=claim_num,
                claim=claim,
                comparisons=claim_comparisons,
                references=analyzed_refs,
            )
            novelty_assessments.append(novelty)

            # Assess inventive step
            inventive = await self._assess_inventive_step(
                claim_num=claim_num,
                claim=claim,
                comparisons=claim_comparisons,
                references=analyzed_refs,
            )
            inventive_step_assessments.append(inventive)

        # Overall assessments
        overall_novelty = all(n.get("is_novel", False) for n in novelty_assessments)
        overall_inventive = all(
            i.get("has_inventive_step", False) for i in inventive_step_assessments
        )

        # Generate reports
        executive_summary = self._generate_executive_summary(
            novelty_assessments,
            inventive_step_assessments,
            analyzed_refs,
            overall_novelty,
            overall_inventive,
        )

        detailed_report = await self._generate_detailed_report(
            state=state,
            novelty_assessments=novelty_assessments,
            inventive_step_assessments=inventive_step_assessments,
        )

        # Key references (top 5 by relevance)
        key_refs = [
            r.get("patent_result", {}).get("document_number", "")
            for r in analyzed_refs[:5]
        ]

        # Recommendations
        recommendations = self._generate_recommendations(
            novelty_assessments,
            inventive_step_assessments,
            search_type,
        )

        self._logger.info(
            "assessment_complete",
            overall_novelty=overall_novelty,
            overall_inventive_step=overall_inventive,
        )

        return {
            "novelty_assessments": novelty_assessments,
            "inventive_step_assessments": inventive_step_assessments,
            "overall_novelty": overall_novelty,
            "overall_inventive_step": overall_inventive,
            "executive_summary": executive_summary,
            "detailed_report": detailed_report,
            "key_references": key_refs,
            "recommendations": recommendations,
            "updated_at": datetime.now().isoformat(),
        }

    async def _assess_novelty(
        self,
        claim_num: int,
        claim: str,
        comparisons: list[ClaimComparison],
        references: list[AnalyzedReference],
    ) -> NoveltyAssessment:
        """Assess novelty of a single claim.

        Args:
            claim_num: Claim number
            claim: Claim text
            comparisons: Comparisons for this claim
            references: Analyzed references

        Returns:
            NoveltyAssessment result
        """
        # Check if any single reference anticipates the claim
        # (all elements found in one reference)

        blocking_reference = None
        is_novel = True

        # Group comparisons by reference
        ref_elements: dict[str, list[str]] = {}
        for comp in comparisons:
            ref_id = comp.get("prior_art_ref", "")
            result = comp.get("comparison_result", "not_found")

            if ref_id not in ref_elements:
                ref_elements[ref_id] = []

            if result in ["identical", "similar"]:
                ref_elements[ref_id].append("found")
            else:
                ref_elements[ref_id].append("not_found")

        # Check if any reference has all elements
        for ref_id, elements in ref_elements.items():
            if elements and all(e == "found" for e in elements):
                blocking_reference = ref_id
                is_novel = False
                break

        # Use LLM for refined assessment
        prompt = f"""다음 청구항의 신규성을 평가하세요.

## 청구항 {claim_num}
{claim}

## 대비 분석 결과
{'신규성 결여 의심: ' + blocking_reference if blocking_reference else '신규성 결여 문헌 미발견'}

## 평가 기준
- 단일 문헌에 모든 구성요소 개시 → 신규성 결여
- 하나라도 미개시 → 신규성 있음

## 출력 형식
IS_NOVEL: [yes/no]
BLOCKING_REF: [문헌번호 또는 none]
REASONING: [2-3문장 근거]
CONFIDENCE: [확실/가능/불확실]
"""

        try:
            response = await self._invoke_llm(prompt)

            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("IS_NOVEL:"):
                    is_novel = "yes" in line.lower()
                elif line.startswith("BLOCKING_REF:"):
                    ref = line.split(":", 1)[1].strip()
                    if ref.lower() != "none":
                        blocking_reference = ref
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    confidence_text = line.split(":", 1)[1].strip()

            confidence = self.relevance_to_confidence(0.8 if is_novel else 0.6)

        except Exception:
            reasoning = "자동 평가 결과"
            confidence = "가능"

        return NoveltyAssessment(
            claim_number=claim_num,
            is_novel=is_novel,
            blocking_reference=blocking_reference,
            assessment_reasoning=reasoning if "reasoning" in dir() else "평가 완료",
            confidence=confidence,
        )

    async def _assess_inventive_step(
        self,
        claim_num: int,
        claim: str,
        comparisons: list[ClaimComparison],
        references: list[AnalyzedReference],
    ) -> InventiveStepAssessment:
        """Assess inventive step of a single claim.

        Args:
            claim_num: Claim number
            claim: Claim text
            comparisons: Comparisons for this claim
            references: Analyzed references

        Returns:
            InventiveStepAssessment result
        """
        # Find which references could be combined
        relevant_refs = []
        for comp in comparisons:
            if comp.get("comparison_result") in ["identical", "similar"]:
                ref_id = comp.get("prior_art_ref", "")
                if ref_id and ref_id not in relevant_refs:
                    relevant_refs.append(ref_id)

        has_inventive_step = True
        combination_motivation = ""
        technical_difficulty = ""
        unexpected_effect = None

        if len(relevant_refs) >= 2:
            # Assess combination obviousness
            prompt = f"""다음 청구항의 진보성을 평가하세요.

## 청구항 {claim_num}
{claim}

## 관련 선행문헌
{', '.join(relevant_refs[:5])}

## 평가 요소
1. 결합 동기: 위 문헌들을 조합할 시사점이 있는가?
2. 기술적 곤란성: 조합에 기술적 어려움이 있는가?
3. 예측 불가 효과: 현저한 효과가 있는가?

## 출력 형식
HAS_INVENTIVE_STEP: [yes/no]
COMBINATION_MOTIVATION: [있음/없음 + 설명]
TECHNICAL_DIFFICULTY: [있음/없음 + 설명]
UNEXPECTED_EFFECT: [있음/없음 + 설명]
CONFIDENCE: [확실/가능/불확실]
"""

            try:
                response = await self._invoke_llm(prompt)

                for line in response.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("HAS_INVENTIVE_STEP:"):
                        has_inventive_step = "yes" in line.lower()
                    elif line.startswith("COMBINATION_MOTIVATION:"):
                        combination_motivation = line.split(":", 1)[1].strip()
                    elif line.startswith("TECHNICAL_DIFFICULTY:"):
                        technical_difficulty = line.split(":", 1)[1].strip()
                    elif line.startswith("UNEXPECTED_EFFECT:"):
                        effect = line.split(":", 1)[1].strip()
                        if "없음" not in effect.lower():
                            unexpected_effect = effect
                    elif line.startswith("CONFIDENCE:"):
                        confidence_text = line.split(":", 1)[1].strip()

            except Exception:
                pass

        confidence = self.relevance_to_confidence(0.7 if has_inventive_step else 0.5)

        return InventiveStepAssessment(
            claim_number=claim_num,
            has_inventive_step=has_inventive_step,
            combination_references=relevant_refs[:5],
            combination_motivation=combination_motivation or "분석 결과 없음",
            technical_difficulty=technical_difficulty or "분석 결과 없음",
            unexpected_effect=unexpected_effect,
            confidence=confidence,
        )

    def _generate_executive_summary(
        self,
        novelty_assessments: list[NoveltyAssessment],
        inventive_assessments: list[InventiveStepAssessment],
        references: list[AnalyzedReference],
        overall_novelty: bool,
        overall_inventive: bool,
    ) -> str:
        """Generate executive summary.

        Args:
            novelty_assessments: Novelty assessment results
            inventive_assessments: Inventive step assessment results
            references: Analyzed references
            overall_novelty: Overall novelty conclusion
            overall_inventive: Overall inventive step conclusion

        Returns:
            Executive summary text
        """
        lines = ["# 선행기술조사 결과 요약\n"]

        # Overall conclusion
        novelty_status = "🟢 신규성 있음" if overall_novelty else "🔴 신규성 결여 의심"
        inventive_status = "🟢 진보성 있음" if overall_inventive else "🟡 진보성 검토 필요"

        lines.append("## 종합 판단\n")
        lines.append(f"- **신규성**: {novelty_status}")
        lines.append(f"- **진보성**: {inventive_status}")
        lines.append("")

        # Claim-by-claim summary
        lines.append("## 청구항별 평가\n")
        lines.append("| 청구항 | 신규성 | 진보성 | 주요 문헌 |")
        lines.append("|--------|--------|--------|----------|")

        for nov, inv in zip(novelty_assessments, inventive_assessments):
            claim_num = nov.get("claim_number", "?")
            nov_status = "🟢" if nov.get("is_novel") else "🔴"
            inv_status = "🟢" if inv.get("has_inventive_step") else "🟡"
            blocking = nov.get("blocking_reference", "-") or "-"
            lines.append(f"| {claim_num} | {nov_status} | {inv_status} | {blocking} |")

        # Key references
        lines.append("\n## 주요 선행기술\n")
        for i, ref in enumerate(references[:5], 1):
            ref_result = ref.get("patent_result", {})
            score = ref.get("relevance_score", 0)
            lines.append(
                f"{i}. **{ref_result.get('document_number', 'N/A')}** "
                f"(관련도: {score:.2f})"
            )
            lines.append(f"   - {ref_result.get('title', 'N/A')[:60]}")

        return "\n".join(lines)

    async def _generate_detailed_report(
        self,
        state: PriorArtSearchState,
        novelty_assessments: list[NoveltyAssessment],
        inventive_step_assessments: list[InventiveStepAssessment],
    ) -> str:
        """Generate detailed report.

        Args:
            state: Current state
            novelty_assessments: Novelty assessment results
            inventive_step_assessments: Inventive step assessment results

        Returns:
            Detailed report text
        """
        lines = [
            "# 선행기술조사 상세 보고서\n",
            f"**조사 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"**조사 유형**: {state.get('search_type', 'N/A')}\n",
        ]

        # Target invention
        lines.append("## 1. 조사 대상 발명\n")
        lines.append(state.get("target_invention", "정보 없음")[:500])
        lines.append("")

        # Search queries
        lines.append("## 2. 검색 전략\n")
        lines.append(f"**키워드**: {', '.join(state.get('search_keywords', []))}")
        lines.append(f"**IPC**: {', '.join(state.get('ipc_codes', []))}")
        lines.append("")

        # Search results
        total_counts = state.get("total_results_count", {})
        lines.append("## 3. 검색 결과\n")
        for db, count in total_counts.items():
            lines.append(f"- {db}: {count}건")
        lines.append("")

        # Analysis
        lines.append("## 4. 관련성 분석\n")
        analyzed = state.get("analyzed_references", [])
        lines.append(f"- 분석 대상: {len(analyzed)}건")
        lines.append(f"- 관련성 임계값: {state.get('relevance_threshold', 0.7)}")
        lines.append("")

        # Comparison table
        lines.append("## 5. 대비표\n")
        lines.append(state.get("comparison_table", "대비표 없음"))
        lines.append("")

        # Assessments
        lines.append("## 6. 신규성 평가\n")
        for nov in novelty_assessments:
            indicator = "🟢" if nov.get("is_novel") else "🔴"
            lines.append(f"### 청구항 {nov.get('claim_number')} {indicator}")
            lines.append(f"- 결론: {'신규성 있음' if nov.get('is_novel') else '신규성 결여 의심'}")
            lines.append(f"- 근거: {nov.get('assessment_reasoning', 'N/A')}")
            if nov.get("blocking_reference"):
                lines.append(f"- 저촉 문헌: {nov.get('blocking_reference')}")
            lines.append("")

        lines.append("## 7. 진보성 평가\n")
        for inv in inventive_step_assessments:
            indicator = "🟢" if inv.get("has_inventive_step") else "🟡"
            lines.append(f"### 청구항 {inv.get('claim_number')} {indicator}")
            lines.append(f"- 결론: {'진보성 있음' if inv.get('has_inventive_step') else '진보성 검토 필요'}")
            lines.append(f"- 결합 동기: {inv.get('combination_motivation', 'N/A')}")
            lines.append(f"- 기술적 곤란성: {inv.get('technical_difficulty', 'N/A')}")
            if inv.get("unexpected_effect"):
                lines.append(f"- 예측 불가 효과: {inv.get('unexpected_effect')}")
            lines.append("")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        novelty_assessments: list[NoveltyAssessment],
        inventive_assessments: list[InventiveStepAssessment],
        search_type: str,
    ) -> list[str]:
        """Generate recommendations based on assessments.

        Args:
            novelty_assessments: Novelty assessment results
            inventive_assessments: Inventive step assessment results
            search_type: Type of search

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for novelty issues
        novelty_issues = [n for n in novelty_assessments if not n.get("is_novel")]
        if novelty_issues:
            claim_nums = [str(n.get("claim_number")) for n in novelty_issues]
            recommendations.append(
                f"청구항 {', '.join(claim_nums)}의 신규성 문제가 의심됩니다. "
                "청구항 보정을 검토하세요."
            )

        # Check for inventive step issues
        inventive_issues = [
            i for i in inventive_assessments if not i.get("has_inventive_step")
        ]
        if inventive_issues:
            claim_nums = [str(i.get("claim_number")) for i in inventive_issues]
            recommendations.append(
                f"청구항 {', '.join(claim_nums)}의 진보성이 의심됩니다. "
                "기술적 효과 강조 또는 구성 한정 추가를 검토하세요."
            )

        # Search type specific recommendations
        if search_type == "freedom_to_operate":
            recommendations.append(
                "FTO 조사 결과, 유효 특허에 대한 회피 설계 또는 라이선스 검토가 필요할 수 있습니다."
            )
        elif search_type == "invalidity_search":
            recommendations.append(
                "무효자료 조사 결과를 바탕으로 무효심판 청구 가능성을 검토하세요."
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "조사 결과 특별한 저촉 문헌이 발견되지 않았습니다. "
                "출원 진행을 권장합니다."
            )

        return recommendations
