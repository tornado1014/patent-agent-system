"""ReportWriterAgent for Patent Analysis workflow.

Responsible for generating analysis reports:
- Executive summary
- Detailed analysis sections
- Visualizations integration
- Export in multiple formats
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from patent_agent.agents.analysis.base import BaseAnalysisAgent
from patent_agent.state.analysis import (
    ANALYSIS_TYPE_INFO,
    AnalysisStep,
    PatentAnalysisState,
)

logger = structlog.get_logger(__name__)


class ReportWriterAgent(BaseAnalysisAgent):
    """Agent for generating analysis reports.

    This agent is responsible for:
    1. Generating executive summaries
    2. Writing detailed analysis sections
    3. Integrating visualizations
    4. Creating appendices

    Example:
        >>> agent = ReportWriterAgent()
        >>> state = {
        ...     "analysis_type": "portfolio",
        ...     "trend_indicators": {...},
        ...     "visualizations": [...],
        ... }
        >>> result = await agent.process(state)
    """

    @property
    def name(self) -> str:
        return "ReportWriterAgent"

    @property
    def description(self) -> str:
        return "분석 보고서 생성"

    @property
    def step(self) -> AnalysisStep:
        return "report"

    def get_system_prompt(self) -> str:
        return """당신은 특허 분석 보고서 작성 전문가입니다.

## 역할
- 분석 결과를 명확하고 체계적인 보고서로 작성
- 경영진을 위한 요약본 및 상세 분석 보고서 생성
- 시각화 자료 통합 및 설명

## 보고서 구성

### 1. 요약 (Executive Summary)
- 핵심 발견사항 (3-5개 bullet points)
- 주요 권고사항
- 리스크/기회 하이라이트

### 2. 분석 개요
- 분석 목적 및 범위
- 데이터 소스 및 기간
- 분석 방법론

### 3. 주요 발견사항
- 분석 유형별 핵심 결과
- 데이터 기반 인사이트
- 시각화 자료와 함께 설명

### 4. 상세 분석
- 섹션별 심층 분석
- 근거 데이터 제시
- 불확실성 및 제한사항 명시

### 5. 권고사항
- 단기/중기/장기 액션 아이템
- 우선순위 및 타임라인
- 예상 비용/효과

### 6. 부록
- 데이터 출처 상세
- 용어 정의
- 참고 문헌

## 작성 원칙
- 객관적이고 데이터 기반
- 명확하고 간결한 문체
- 비전문가도 이해 가능
- 신뢰도 표시 포함 (🟢/🟡/🔴)
"""

    async def process(self, state: PatentAnalysisState) -> dict[str, Any]:
        """Generate analysis report.

        Args:
            state: Current workflow state

        Returns:
            State updates with generated report
        """
        self._logger.info("starting_report_generation")

        analysis_type = state.get("analysis_type", "portfolio")
        target = state.get("target", "")

        # Generate executive summary
        executive_summary = await self._generate_executive_summary(state)

        # Generate detailed report
        detailed_report = await self._generate_detailed_report(state)

        # Generate appendices
        appendices = self._generate_appendices(state)

        self._logger.info(
            "report_generation_complete",
            analysis_type=analysis_type,
            target=target,
        )

        return {
            "executive_summary": executive_summary,
            "detailed_report": detailed_report,
            "appendices": appendices,
            "current_step": "report",
            "updated_at": datetime.now().isoformat(),
        }

    async def _generate_executive_summary(
        self,
        state: PatentAnalysisState,
    ) -> str:
        """Generate executive summary.

        Args:
            state: Current workflow state

        Returns:
            Executive summary text
        """
        analysis_type = state.get("analysis_type", "portfolio")
        target = state.get("target", "")
        analysis_info = ANALYSIS_TYPE_INFO.get(analysis_type, {})

        # Collect key findings based on analysis type
        key_findings = self._collect_key_findings(state)

        # Build summary prompt
        prompt = f"""다음 특허 분석 결과를 바탕으로 경영진 요약(Executive Summary)을 작성하세요.

분석 유형: {analysis_info.get('name', analysis_type)}
분석 대상: {target}
분석 목적: {analysis_info.get('purpose', '')}

주요 발견사항:
{key_findings}

요약 작성 요건:
1. 3-5개의 핵심 발견사항 (bullet points)
2. 주요 권고사항 2-3개
3. 리스크 또는 기회 하이라이트
4. 전체 길이: 300-500자

비전문가도 이해할 수 있도록 명확하게 작성하세요."""

        try:
            summary = await self._invoke_llm(prompt)
        except Exception as e:
            self._logger.warning("llm_summary_failed", error=str(e))
            summary = self._generate_fallback_summary(state)

        return summary

    def _collect_key_findings(self, state: PatentAnalysisState) -> str:
        """Collect key findings from analysis results.

        Args:
            state: Current workflow state

        Returns:
            Formatted key findings string
        """
        findings = []
        analysis_type = state.get("analysis_type", "portfolio")

        # Portfolio findings
        portfolio_report = state.get("portfolio_report")
        if portfolio_report:
            metrics = portfolio_report.get("metrics", {})
            findings.append(f"총 보유 특허: {metrics.get('total_patents', 0)}건")
            findings.append(f"핵심 특허: {len(metrics.get('core_patents', []))}건")

        # Trend findings
        trend_indicators = state.get("trend_indicators")
        if trend_indicators:
            growth = trend_indicators.get("growth_rate", 0)
            findings.append(f"연평균 성장률: {growth * 100:.1f}%")

            emerging = trend_indicators.get("emerging_areas", [])
            if emerging:
                findings.append(f"신흥 기술 영역: {', '.join(emerging[:3])}")

        # Maturity findings
        maturity = state.get("maturity_assessment")
        if maturity:
            findings.append(f"기술 성숙도: {maturity.get('phase_kr', '')}")
            findings.append(f"권장 전략: {maturity.get('recommended_strategy', '')}")

        # Competitor findings
        competitor_profiles = state.get("competitor_profiles", [])
        if competitor_profiles:
            top_3 = competitor_profiles[:3]
            findings.append(f"상위 경쟁사: {', '.join([p['company_name'][:20] for p in top_3])}")

        # Infringement findings
        infringement = state.get("infringement_analysis")
        if infringement:
            risk = infringement.get("overall_risk", "none")
            findings.append(f"침해 리스크: {risk.upper()}")
            if infringement.get("literal_infringement"):
                findings.append("⚠️ 문언 침해 가능성 있음")

        # Invalidity findings
        invalidity = state.get("invalidity_analysis")
        if invalidity:
            likelihood = invalidity.get("overall_invalidity_likelihood", "불확실")
            findings.append(f"무효 가능성: {likelihood}")

        return "\n".join([f"- {f}" for f in findings]) if findings else "분석 결과 없음"

    def _generate_fallback_summary(self, state: PatentAnalysisState) -> str:
        """Generate fallback summary if LLM fails.

        Args:
            state: Current workflow state

        Returns:
            Basic summary text
        """
        analysis_type = state.get("analysis_type", "portfolio")
        target = state.get("target", "")
        total_patents = state.get("total_patents_collected", 0)

        return f"""# 분석 요약

## 분석 개요
- 유형: {ANALYSIS_TYPE_INFO.get(analysis_type, {}).get('name', analysis_type)}
- 대상: {target}
- 데이터: {total_patents}건의 특허 분석

## 주요 발견사항
{self._collect_key_findings(state)}

## 권고사항
- 상세 보고서를 참조하여 세부 내용 확인
- 필요 시 전문가 검토 권장

---
*본 보고서는 자동 생성되었습니다. 생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    async def _generate_detailed_report(
        self,
        state: PatentAnalysisState,
    ) -> str:
        """Generate detailed analysis report.

        Args:
            state: Current workflow state

        Returns:
            Detailed report text
        """
        analysis_type = state.get("analysis_type", "portfolio")
        analysis_info = ANALYSIS_TYPE_INFO.get(analysis_type, {})

        sections = []

        # Header
        sections.append(f"# 특허 분석 보고서: {analysis_info.get('name', analysis_type)}")
        sections.append(f"\n생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        sections.append(f"\n분석 대상: {state.get('target', '')}")
        sections.append("\n---\n")

        # Analysis overview
        sections.append("## 1. 분석 개요\n")
        sections.append(f"### 분석 목적\n{analysis_info.get('purpose', '')}\n")
        sections.append(f"### 예상 산출물\n")
        for output in analysis_info.get("outputs", []):
            sections.append(f"- {output}")
        sections.append("\n")

        # Data sources
        sections.append("### 데이터 소스\n")
        for source in state.get("data_sources", []):
            sections.append(f"- {source}")
        sections.append(f"\n총 분석 건수: {state.get('total_patents_collected', 0)}건\n")

        # Type-specific sections
        if analysis_type == "portfolio":
            sections.append(self._generate_portfolio_section(state))
        elif analysis_type == "trend":
            sections.append(self._generate_trend_section(state))
        elif analysis_type == "competitor":
            sections.append(self._generate_competitor_section(state))
        elif analysis_type == "infringement":
            sections.append(self._generate_infringement_section(state))
        elif analysis_type == "invalidity":
            sections.append(self._generate_invalidity_section(state))

        # Visualizations
        sections.append("## 시각화 자료\n")
        visualizations = state.get("visualizations", [])
        for i, viz in enumerate(visualizations, 1):
            sections.append(f"### {i}. {viz.get('title', 'Chart')}")
            sections.append(f"- 유형: {viz.get('chart_type', 'unknown')}")
            if viz.get("file_path"):
                sections.append(f"- 파일: {viz.get('file_path')}")
            sections.append("")

        # Recommendations
        sections.append("## 권고사항\n")
        sections.append(self._generate_recommendations_section(state))

        return "\n".join(sections)

    def _generate_portfolio_section(self, state: PatentAnalysisState) -> str:
        """Generate portfolio analysis section.

        Args:
            state: Current workflow state

        Returns:
            Portfolio section text
        """
        section = ["## 2. 포트폴리오 분석 결과\n"]

        portfolio = state.get("portfolio_report")
        if not portfolio:
            section.append("분석 결과 없음\n")
            return "\n".join(section)

        metrics = portfolio.get("metrics", {})

        section.append("### 보유 현황")
        section.append(f"- 총 특허 수: {metrics.get('total_patents', 0)}건")
        section.append(f"- 평균 인용 수: {metrics.get('average_citations', 0):.1f}회")
        section.append(f"- 평균 가족 크기: {metrics.get('average_family_size', 0):.1f}개국")
        section.append("")

        section.append("### 강점")
        for strength in portfolio.get("strengths", []):
            section.append(f"- {strength}")
        section.append("")

        section.append("### 약점")
        for weakness in portfolio.get("weaknesses", []):
            section.append(f"- {weakness}")
        section.append("")

        return "\n".join(section)

    def _generate_trend_section(self, state: PatentAnalysisState) -> str:
        """Generate trend analysis section.

        Args:
            state: Current workflow state

        Returns:
            Trend section text
        """
        section = ["## 2. 기술 동향 분석 결과\n"]

        indicators = state.get("trend_indicators")
        maturity = state.get("maturity_assessment")

        if not indicators:
            section.append("분석 결과 없음\n")
            return "\n".join(section)

        section.append("### 핵심 지표")
        section.append(f"- 성장률: {indicators.get('growth_rate', 0) * 100:.1f}%")
        section.append(f"- 집중도(HHI): {indicators.get('concentration', 0):.4f}")
        section.append(f"- 기술 이동: {indicators.get('technology_shift', '')}")
        section.append("")

        if maturity:
            section.append("### 기술 성숙도 평가")
            section.append(f"- 현재 단계: {maturity.get('phase_kr', '')} ({maturity.get('phase', '')})")
            section.append(f"- 특성: {maturity.get('characteristics', '')}")
            section.append(f"- 권장 전략: {maturity.get('recommended_strategy', '')}")
            section.append("")

        emerging = indicators.get("emerging_areas", [])
        if emerging:
            section.append("### 신흥 기술 영역")
            for area in emerging:
                section.append(f"- {area}")
            section.append("")

        return "\n".join(section)

    def _generate_competitor_section(self, state: PatentAnalysisState) -> str:
        """Generate competitor analysis section.

        Args:
            state: Current workflow state

        Returns:
            Competitor section text
        """
        section = ["## 2. 경쟁사 분석 결과\n"]

        profiles = state.get("competitor_profiles", [])

        if not profiles:
            section.append("분석 결과 없음\n")
            return "\n".join(section)

        section.append("### 상위 경쟁사 프로필")
        for i, profile in enumerate(profiles[:5], 1):
            section.append(f"\n#### {i}. {profile.get('company_name', '')}")
            section.append(f"- 특허 수: {profile.get('patent_count', 0)}건")
            section.append(f"- 점유율: {profile.get('market_share', 0) * 100:.1f}%")
            section.append(f"- 핵심 기술: {', '.join(profile.get('key_technologies', [])[:3])}")
            section.append(f"- 최근 동향: {profile.get('recent_trends', '')}")

        return "\n".join(section)

    def _generate_infringement_section(self, state: PatentAnalysisState) -> str:
        """Generate infringement analysis section.

        Args:
            state: Current workflow state

        Returns:
            Infringement section text
        """
        section = ["## 2. 침해 분석 결과\n"]

        analysis = state.get("infringement_analysis")

        if not analysis:
            section.append("분석 결과 없음\n")
            return "\n".join(section)

        section.append("### 분석 대상")
        section.append(f"- 대상 특허: {analysis.get('target_patent', '')}")
        section.append(f"- 대상 제품: {analysis.get('target_product', '')}")
        section.append("")

        section.append("### 침해 평가")
        risk = analysis.get("overall_risk", "none")
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢", "none": "⚪"}.get(risk, "⚪")
        section.append(f"- 전체 리스크: {risk_emoji} {risk.upper()}")
        section.append(f"- 문언 침해: {'예' if analysis.get('literal_infringement') else '아니오'}")
        section.append(f"- 균등론 침해: {'예' if analysis.get('doctrine_of_equivalents') else '아니오'}")
        section.append("")

        section.append("### 회피 설계 옵션")
        for option in analysis.get("design_around_options", []):
            section.append(f"- {option}")

        return "\n".join(section)

    def _generate_invalidity_section(self, state: PatentAnalysisState) -> str:
        """Generate invalidity analysis section.

        Args:
            state: Current workflow state

        Returns:
            Invalidity section text
        """
        section = ["## 2. 무효 분석 결과\n"]

        analysis = state.get("invalidity_analysis")

        if not analysis:
            section.append("분석 결과 없음\n")
            return "\n".join(section)

        section.append("### 분석 대상")
        section.append(f"- 대상 특허: {analysis.get('target_patent', '')}")
        section.append("")

        section.append("### 무효 가능성 평가")
        likelihood = analysis.get("overall_invalidity_likelihood", "불확실")
        section.append(f"- 전체 평가: {self.get_confidence_indicator(likelihood)} {likelihood}")

        strongest = analysis.get("strongest_ground")
        if strongest:
            section.append(f"- 가장 유력한 무효 사유: {strongest}")
        section.append("")

        section.append("### 무효 논거")
        for ground in analysis.get("grounds", []):
            section.append(f"\n#### {ground.get('ground_type', '')}")
            section.append(f"- 법적 근거: {ground.get('legal_basis', '')}")
            section.append(f"- 대상 청구항: {ground.get('claim_numbers', [])}")
            section.append(f"- 성공 가능성: {ground.get('success_probability', '')}")

        return "\n".join(section)

    def _generate_recommendations_section(self, state: PatentAnalysisState) -> str:
        """Generate recommendations section.

        Args:
            state: Current workflow state

        Returns:
            Recommendations text
        """
        recommendations = []
        analysis_type = state.get("analysis_type", "portfolio")

        # Get type-specific recommendations
        if analysis_type == "portfolio":
            portfolio = state.get("portfolio_report", {})
            recommendations.extend(portfolio.get("recommendations", []))
        elif analysis_type == "infringement":
            infringement = state.get("infringement_analysis", {})
            recommendations.extend(infringement.get("recommendations", []))
        elif analysis_type == "invalidity":
            invalidity = state.get("invalidity_analysis", {})
            recommendations.extend(invalidity.get("recommendations", []))

        # Add general recommendations
        if not recommendations:
            recommendations = [
                "상세 분석 결과 검토 권장",
                "필요 시 전문가 자문 권장",
                "정기적인 모니터링 체계 구축 권장",
            ]

        return "\n".join([f"- {r}" for r in recommendations])

    def _generate_appendices(self, state: PatentAnalysisState) -> list[str]:
        """Generate appendices.

        Args:
            state: Current workflow state

        Returns:
            List of appendix file paths
        """
        # In production, this would create actual files
        # For now, return placeholder paths
        appendices = []

        if state.get("patent_data"):
            appendices.append("appendix_a_patent_list.xlsx")

        if state.get("claim_mappings"):
            appendices.append("appendix_b_claim_charts.xlsx")

        if state.get("visualizations"):
            appendices.append("appendix_c_visualizations.zip")

        appendices.append("appendix_d_methodology.md")
        appendices.append("appendix_e_glossary.md")

        return appendices
