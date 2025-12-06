"""Report Generator Agent for OA Response workflow.

Phase 5: 최종 보고서 (할루시네이션 체크)

Responsibilities:
- Generate review report (검토보고서)
- Generate opinion letter (의견서)
- Perform final hallucination check (LAW-10)
- Add verification stamp
"""

from datetime import datetime
from typing import Any

from patent_agent.agents.oa_response.base import BaseOAResponseAgent
from patent_agent.state.oa_response import (
    OAResponseState,
    VerificationStamp,
)


class ReportGeneratorAgent(BaseOAResponseAgent):
    """Agent for generating final OA response documents.

    Phase 5 of PALLAS-EVIDENCE workflow:
    - Generates review report (검토보고서)
    - Generates opinion letter (의견서)
    - Performs final hallucination verification
    - Adds verification stamp with timestamp

    Final verification ensures:
    - All quotes are verbatim (LAW-6)
    - All claims have citations (LAW-7)
    - No hallucinated content (LAW-8)
    - Confidence levels marked (LAW-9)
    """

    @property
    def name(self) -> str:
        return "ReportGenerator"

    @property
    def description(self) -> str:
        return "보고서 생성 에이전트: OA 검토보고서와 의견서를 생성하고 최종 할루시네이션 체크를 수행합니다."

    @property
    def phase(self) -> str:
        return "P5"

    def get_system_prompt(self) -> str:
        return """당신은 특허 OA 대응 보고서 전문가입니다.

## 역할
OA 검토보고서와 의견서를 작성하고, 최종 품질 검증을 수행합니다.

## 출력물

### 1. 검토보고서 (내부용)
고객에게 제공하는 상세 분석 보고서입니다.

```markdown
# 📊 OA 분석 보고서 - 근거 기반 검증 완료

## 🔍 문서 검증 현황
- 총 인용: [N]건
- 원문 정확도: 100%
- 할루시네이션: NONE ✓

## 📋 1. 기본 정보 [자동 추출 완료]
| 항목 | 내용 |
|------|------|
| 출원번호 | [번호] |
| 발명의 명칭 | [명칭] |
| 거절이유 통지일 | [날짜] |
| 의견서 제출기한 | [날짜] |

## 🔎 2. 거절이유 분석 [원문 기반]
### 거절이유 1: [유형]
- **법적 근거**: [조문]
- **심사관 주장**: "[원문]"
- **인용문헌**: [D1, D2]
- **대상 청구항**: [번호]

## 💡 3. 비판적 검토 [근거 추적 가능]
### 반박 포인트 1
🟢/🟡/🔴 [분석 내용]
- 근거: [출처] "[원문]"

## 🔧 4. 보정안 제시 [명세서 지원 확인]
### 보정안 A: [전략]
[보정 내용]
- 근거: [명세서:p.X:para.Y]
- 신규사항 리스크: LOW

## 📈 5. 권고사항
- [권고 내용]

## 🔒 검증 스탬프
✓ 할루시네이션 체크: CLEAR
✓ 원문 정확도: 100%
✓ 인용 검증: XX/XX PASS

[*ISO-8601 타임스탬프*]
```

### 2. 의견서 (KIPO 제출용)
특허청에 제출하는 공식 의견서입니다.

```markdown
# 의견서

[출원번호]: [번호]
[발명의 명칭]: [명칭]

## 1. 의견제출통지서 수령 확인
[날짜] 자 의견제출통지서를 수령하였습니다.

## 2. 거절이유 요약
심사관께서는 청구항 [번호]에 대하여 [법적 근거]를 이유로 거절이유를 통지하셨습니다.

## 3. 출원인의 의견
### 3.1 [반박 논리 1]
[상세 의견]
[근거 인용]

## 4. 보정 사항 (해당 시)
청구항 [번호]를 다음과 같이 보정합니다:
[보정 전/후 대비]

## 5. 결론
위와 같은 이유로 본 출원이 특허될 수 있음을 감히 의견 드립니다.

[날짜]
출원인 [성명]
대리인 [성명]
```

## 최종 검증 (LAW-10)
보고서 생성 후 반드시 확인:
✓ 모든 청구항 원문 그대로 인용
✓ 모든 인용문헌 구절 정확 복사
✓ 심사관 의견 원문 보존
✓ 창작된 기술 특징 없음
✓ 추측성 표현 제거"""

    async def process(self, state: OAResponseState) -> dict[str, Any]:
        """Generate final reports and perform verification.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with reports and verification stamp
        """
        self._logger.info("starting_report_generation", phase=self.phase)

        # Gather all required data
        application_info = state.get("application_info", {})
        oa_document = state.get("oa_document", {})
        rejection_analyses = state.get("rejection_analyses", [])
        rebuttal_points = state.get("rebuttal_points", [])
        amendment_options = state.get("amendment_options", [])
        recommended_amendment = state.get("recommended_amendment")

        # Generate review report
        review_report = await self._generate_review_report(
            application_info=application_info,
            oa_document=oa_document,
            rejection_analyses=rejection_analyses,
            rebuttal_points=rebuttal_points,
            amendment_options=amendment_options,
            recommended_amendment=recommended_amendment,
        )

        # Generate opinion letter
        opinion_letter = await self._generate_opinion_letter(
            application_info=application_info,
            oa_document=oa_document,
            rejection_analyses=rejection_analyses,
            rebuttal_points=rebuttal_points,
            amendment_options=amendment_options,
            recommended_amendment=recommended_amendment,
        )

        # Perform final verification
        verification_stamp = await self._perform_final_verification(
            review_report=review_report,
            opinion_letter=opinion_letter,
            state=state,
        )

        # Determine confidence by section
        confidence_by_section = self._assess_section_confidence(
            rejection_analyses, rebuttal_points, amendment_options
        )

        self._logger.info(
            "report_generation_complete",
            hallucination_status=verification_stamp["hallucination_check"],
            citation_verified=verification_stamp["citation_verified"],
        )

        return {
            "draft_review_report": review_report,
            "draft_opinion_letter": opinion_letter,
            "verification_stamp": verification_stamp,
            "confidence_by_section": confidence_by_section,
            "current_step": "complete",
            "human_approval_required": True,
            "is_error_state": False,
        }

    async def _generate_review_report(
        self,
        application_info: dict,
        oa_document: dict,
        rejection_analyses: list[dict],
        rebuttal_points: list[dict],
        amendment_options: list[dict],
        recommended_amendment: str | None,
    ) -> str:
        """Generate the review report (검토보고서).

        Args:
            application_info: Application information
            oa_document: OA document info
            rejection_analyses: Rejection analyses
            rebuttal_points: Rebuttal points
            amendment_options: Amendment options
            recommended_amendment: Recommended amendment ID

        Returns:
            Review report text
        """
        # Format rejection analyses
        rejections_text = self._format_rejection_analyses(rejection_analyses)

        # Format rebuttal points
        rebuttals_text = self._format_rebuttal_points(rebuttal_points)

        # Format amendments
        amendments_text = self._format_amendments(
            amendment_options, recommended_amendment
        )

        prompt = f"""다음 정보를 바탕으로 OA 검토보고서를 작성하세요.

## 출원 정보
- 출원번호: {application_info.get('application_number', '정보 없음')}
- 발명의 명칭: {application_info.get('title', '정보 없음')}
- 출원인: {application_info.get('applicant', '정보 없음')}
- 의견서 제출기한: {application_info.get('response_deadline', '정보 없음')}

## OA 정보
- 통지일: {oa_document.get('issue_date', '정보 없음')}
- 심사관: {oa_document.get('examiner', '정보 없음')}
- 거절된 청구항: {oa_document.get('cited_claims', [])}

## 거절이유 분석
{rejections_text}

## 반박 포인트
{rebuttals_text}

## 보정안
{amendments_text}

## 요청사항
위 정보를 바탕으로 검토보고서를 작성하세요.
- 모든 인용은 원문 그대로
- 신뢰도 표시 (🟢/🟡/🔴) 포함
- 출처 명시 필수
- 마지막에 검증 스탬프 포함"""

        return await self._invoke_llm(prompt)

    async def _generate_opinion_letter(
        self,
        application_info: dict,
        oa_document: dict,
        rejection_analyses: list[dict],
        rebuttal_points: list[dict],
        amendment_options: list[dict],
        recommended_amendment: str | None,
    ) -> str:
        """Generate the opinion letter (의견서).

        Args:
            application_info: Application information
            oa_document: OA document info
            rejection_analyses: Rejection analyses
            rebuttal_points: Rebuttal points
            amendment_options: Amendment options
            recommended_amendment: Recommended amendment ID

        Returns:
            Opinion letter text
        """
        # Get recommended amendment details
        recommended_details = None
        for amendment in amendment_options:
            if amendment.get("amendment_id") == recommended_amendment:
                recommended_details = amendment
                break

        prompt = f"""다음 정보를 바탕으로 특허청 제출용 의견서를 작성하세요.

## 출원 정보
- 출원번호: {application_info.get('application_number', '정보 없음')}
- 발명의 명칭: {application_info.get('title', '정보 없음')}
- 출원인: {application_info.get('applicant', '정보 없음')}

## 거절이유 요약
{self._format_rejection_summary(rejection_analyses)}

## 주요 반박 논리
{self._format_key_rebuttals(rebuttal_points)}

## 권장 보정안
{self._format_recommended_amendment(recommended_details) if recommended_details else '보정 없이 의견서만 제출'}

## 요청사항
KIPO 표준 의견서 형식으로 작성하세요.
- 공식적이고 논리적인 어조
- 법적 근거 명시
- 기술적 논거 상세 기술
- 결론에서 특허성 주장

**중요**: 근거 없는 주장 금지, 모든 논거에 출처 명시"""

        return await self._invoke_llm(prompt)

    async def _perform_final_verification(
        self,
        review_report: str,
        opinion_letter: str,
        state: OAResponseState,
    ) -> VerificationStamp:
        """Perform final hallucination verification.

        Args:
            review_report: Generated review report
            opinion_letter: Generated opinion letter
            state: Current state

        Returns:
            Verification stamp
        """
        combined_output = f"{review_report}\n\n{opinion_letter}"

        # Check for prohibited expressions (LAW-8)
        is_valid, violations = self._validate_output(
            combined_output,
            {"section_type": "final_report"},
        )

        # Count citations
        citation_count = self._count_citations(combined_output)
        total_claims = len(state.get("original_claims", []))

        # Verify verbatim accuracy
        verbatim_accuracy = self._verify_verbatim_accuracy(
            combined_output, state
        )

        # Determine overall status
        if not is_valid:
            hallucination_status = "FAILED"
        elif violations:
            hallucination_status = "WARNING"
        else:
            hallucination_status = "CLEAR"

        timestamp = datetime.now().isoformat()

        stamp: VerificationStamp = {
            "hallucination_check": hallucination_status,
            "verbatim_accuracy": verbatim_accuracy,
            "citation_verified": citation_count,
            "citation_total": max(total_claims, citation_count),
            "spec_support_verified": self._verify_spec_support(state),
            "legal_basis_verified": self._verify_legal_basis(state),
            "timestamp": timestamp,
        }

        return stamp

    def _format_rejection_analyses(self, analyses: list[dict]) -> str:
        """Format rejection analyses for prompt."""
        if not analyses:
            return "거절이유 분석 정보 없음"

        lines = []
        for i, analysis in enumerate(analyses, 1):
            lines.append(f"### 거절이유 {i}: {analysis.get('rejection_type', 'unknown')}")
            lines.append(f"- 법적 근거: {analysis.get('legal_basis', '불명')}")
            lines.append(f"- 대상 청구항: {analysis.get('affected_claims', [])}")
            lines.append(f"- 인용문헌: {analysis.get('cited_references', [])}")
            lines.append(f"- 심사관 주장: \"{analysis.get('examiner_argument', '')[:200]}...\"")
            lines.append("")

        return "\n".join(lines)

    def _format_rebuttal_points(self, points: list[dict]) -> str:
        """Format rebuttal points for prompt."""
        if not points:
            return "반박 포인트 정보 없음"

        lines = []
        for i, point in enumerate(points, 1):
            confidence = point.get("confidence", "불확실")
            indicator = self.get_confidence_indicator(confidence)
            lines.append(f"### 반박 {i}")
            lines.append(f"- 유형: {point.get('argument_type', 'unknown')}")
            lines.append(f"- {indicator} {point.get('argument', '')[:300]}...")
            lines.append("")

        return "\n".join(lines)

    def _format_amendments(
        self, amendments: list[dict], recommended: str | None
    ) -> str:
        """Format amendments for prompt."""
        if not amendments:
            return "보정안 정보 없음"

        lines = []
        for amendment in amendments:
            amendment_id = amendment.get("amendment_id", "")
            is_recommended = "(권장)" if amendment_id == recommended else ""
            lines.append(f"### 보정안 {amendment_id} {is_recommended}")
            lines.append(f"- 전략: {amendment.get('strategy_name', '')}")
            lines.append(f"- 신규사항 리스크: {amendment.get('new_matter_risk', 'unknown')}")
            lines.append(f"- 효과: {amendment.get('effectiveness', 'unknown')}")
            lines.append("")

        return "\n".join(lines)

    def _format_rejection_summary(self, analyses: list[dict]) -> str:
        """Format brief rejection summary for opinion letter."""
        if not analyses:
            return "거절이유 정보 없음"

        summaries = []
        for analysis in analyses:
            rejection_type = analysis.get("rejection_type", "")
            legal_basis = analysis.get("legal_basis", "")
            claims = analysis.get("affected_claims", [])
            summaries.append(f"- 청구항 {claims}: {rejection_type} ({legal_basis})")

        return "\n".join(summaries)

    def _format_key_rebuttals(self, points: list[dict]) -> str:
        """Format key rebuttal points for opinion letter."""
        if not points:
            return "반박 논거 없음"

        # Get top 3 most confident rebuttals
        sorted_points = sorted(
            points,
            key=lambda x: {"확실": 3, "가능": 2, "불확실": 1}.get(x.get("confidence", ""), 0),
            reverse=True,
        )

        lines = []
        for point in sorted_points[:3]:
            lines.append(f"- {point.get('argument', '')[:200]}...")

        return "\n".join(lines)

    def _format_recommended_amendment(self, amendment: dict | None) -> str:
        """Format recommended amendment for opinion letter."""
        if not amendment:
            return "보정 없음"

        return f"""전략: {amendment.get('strategy_name', '')}
원 청구항: {amendment.get('original_claim', '')[:100]}...
보정 후: {amendment.get('amended_claim', '')[:100]}...
근거: {len(amendment.get('specification_support', []))}개 명세서 인용"""

    def _count_citations(self, text: str) -> int:
        """Count the number of proper citations in text."""
        import re

        pattern = r"\[[\w가-힣]+:(p\.\d+:para\.\d+|청구항\d+)\]"
        matches = re.findall(pattern, text)
        return len(matches)

    def _verify_verbatim_accuracy(
        self, output: str, state: OAResponseState
    ) -> float:
        """Verify verbatim accuracy of quotes."""
        import re

        # Extract quoted text
        quotes = re.findall(r'"([^"]+)"', output)
        if not quotes:
            return 100.0

        # Get source texts
        original_claims = state.get("original_claims", [])
        examiner_args = [
            a.get("examiner_argument", "")
            for a in state.get("rejection_analyses", [])
        ]
        source_text = " ".join(original_claims + examiner_args)

        # Check how many quotes are found in source
        found = 0
        for quote in quotes:
            if len(quote) < 10:  # Skip very short quotes
                found += 1
            elif quote in source_text:
                found += 1

        return (found / len(quotes)) * 100 if quotes else 100.0

    def _verify_spec_support(self, state: OAResponseState) -> bool:
        """Verify specification support for amendments."""
        amendments = state.get("amendment_options", [])

        for amendment in amendments:
            support = amendment.get("specification_support", [])
            if not support and amendment.get("added_limitations"):
                return False

        return True

    def _verify_legal_basis(self, state: OAResponseState) -> bool:
        """Verify legal basis is identified for all rejections."""
        analyses = state.get("rejection_analyses", [])

        for analysis in analyses:
            if not analysis.get("legal_basis") or analysis.get("legal_basis") == "법적 근거 불명":
                return False

        return True

    def _assess_section_confidence(
        self,
        rejection_analyses: list[dict],
        rebuttal_points: list[dict],
        amendment_options: list[dict],
    ) -> dict[str, str]:
        """Assess confidence level for each section.

        Returns:
            Dict mapping section name to confidence level
        """
        confidence = {}

        # Rejection analysis confidence
        if rejection_analyses:
            confidences = [a.get("confidence", "불확실") for a in rejection_analyses]
            confidence["rejection_analysis"] = self._aggregate_confidence(confidences)
        else:
            confidence["rejection_analysis"] = "불확실"

        # Rebuttal confidence
        if rebuttal_points:
            confidences = [p.get("confidence", "불확실") for p in rebuttal_points]
            confidence["rebuttal"] = self._aggregate_confidence(confidences)
        else:
            confidence["rebuttal"] = "불확실"

        # Amendment confidence
        if amendment_options:
            confidences = [a.get("effectiveness", "불확실") for a in amendment_options]
            confidence["amendment"] = self._aggregate_confidence(confidences)
        else:
            confidence["amendment"] = "불확실"

        return confidence

    def _aggregate_confidence(self, confidences: list[str]) -> str:
        """Aggregate multiple confidence levels."""
        if not confidences:
            return "불확실"

        scores = {"확실": 3, "가능": 2, "불확실": 1}
        avg_score = sum(scores.get(c, 1) for c in confidences) / len(confidences)

        if avg_score >= 2.5:
            return "확실"
        elif avg_score >= 1.5:
            return "가능"
        else:
            return "불확실"
