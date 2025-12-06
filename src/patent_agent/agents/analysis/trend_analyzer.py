"""TrendAnalyzerAgent for Patent Analysis workflow.

Responsible for analyzing technology trends:
- Time series analysis of patent filings
- Technology maturity assessment
- Emerging technology identification
- Growth rate calculations
"""

from datetime import datetime
from typing import Any

import structlog

from patent_agent.agents.analysis.base import BaseAnalysisAgent, PatentScorer
from patent_agent.state.analysis import (
    AnalysisStep,
    MaturityPhase,
    PatentAnalysisState,
    PatentRecord,
    TrendIndicators,
)

logger = structlog.get_logger(__name__)


class TrendAnalyzerAgent(BaseAnalysisAgent):
    """Agent for analyzing technology trends in patent data.

    This agent is responsible for:
    1. Analyzing filing trends over time
    2. Assessing technology maturity phase
    3. Identifying emerging technology areas
    4. Calculating growth rates and concentration

    Example:
        >>> agent = TrendAnalyzerAgent()
        >>> state = {
        ...     "analysis_type": "trend",
        ...     "patent_data": [...],
        ... }
        >>> result = await agent.process(state)
    """

    # Maturity phase definitions
    MATURITY_PHASES = {
        "emergence": {
            "phase_kr": "태동기",
            "growth_rate_range": (0.5, float("inf")),  # >50% growth
            "characteristics": "급격한 출원 증가, 적은 출원인 수, 핵심 기술 개발 단계",
            "recommended_strategy": "기초 기술 확보, 핵심 특허 선점, 적극적 R&D 투자",
        },
        "growth": {
            "phase_kr": "성장기",
            "growth_rate_range": (0.1, 0.5),  # 10-50% growth
            "characteristics": "출원 꾸준히 증가, 출원인 수 증가, 응용 기술 개발",
            "recommended_strategy": "응용 기술 확대, 특허 포트폴리오 강화, 라이선스 전략",
        },
        "maturity": {
            "phase_kr": "성숙기",
            "growth_rate_range": (-0.1, 0.1),  # -10% to 10%
            "characteristics": "출원 안정화, 개량 발명 중심, 대기업 지배",
            "recommended_strategy": "개량 발명 집중, 비용 효율화, 방어적 포트폴리오",
        },
        "decline": {
            "phase_kr": "쇠퇴기",
            "growth_rate_range": (float("-inf"), -0.1),  # <-10%
            "characteristics": "출원 감소, 기술 대체 진행, 핵심 특허 만료",
            "recommended_strategy": "기술 전환 준비, 선택적 유지, 새로운 기술 영역 진출",
        },
    }

    @property
    def name(self) -> str:
        return "TrendAnalyzerAgent"

    @property
    def description(self) -> str:
        return "기술 동향 분석 및 성숙도 평가"

    @property
    def step(self) -> AnalysisStep:
        return "analyze"

    def get_system_prompt(self) -> str:
        return """당신은 기술 동향 분석 전문가입니다.

## 역할
- 특허 출원 트렌드 시계열 분석
- 기술 성숙도 평가 (태동기/성장기/성숙기/쇠퇴기)
- 신흥 기술 영역 발굴
- 주요 출원인 동향 파악

## 분석 지표

### 성장률 (Growth Rate)
- 연간 출원 증가율
- 이동 평균 기반 추세 분석
- 가속도 (2차 미분) 분석

### 집중도 (Concentration)
- 허핀달-허쉬만 지수 (HHI)
- 상위 N개 출원인 점유율
- 기술 분야별 집중도

### 기술 이동 (Technology Shift)
- IPC 코드 변화 추적
- 신규 IPC 출현 분석
- 융합 기술 영역 식별

## 성숙도 판단 기준

| 단계 | 성장률 | 특징 |
|------|--------|------|
| 태동기 | >50% | 급격한 증가, 소수 출원인 |
| 성장기 | 10-50% | 지속 증가, 출원인 증가 |
| 성숙기 | ±10% | 안정화, 대기업 지배 |
| 쇠퇴기 | <-10% | 감소, 기술 대체 |

## 출력 형식
1. 트렌드 지표 (성장률, 집중도, 기술 이동)
2. 성숙도 평가 (단계, 특성, 권장 전략)
3. 신흥 기술 영역 목록
4. 시각화 데이터 (시계열, 분포 등)
"""

    async def process(self, state: PatentAnalysisState) -> dict[str, Any]:
        """Analyze technology trends.

        Args:
            state: Current workflow state

        Returns:
            State updates with trend analysis
        """
        self._logger.info("starting_trend_analysis")

        patent_data = state.get("patent_data", [])
        analysis_type = state.get("analysis_type", "trend")

        if not patent_data:
            return {
                "is_error_state": True,
                "error_messages": ["분석할 특허 데이터가 없습니다."],
                "current_step": "analyze",
            }

        # Group patents by year
        patents_by_year = self.group_patents_by_year(patent_data)

        # Calculate trend indicators
        trend_indicators = self._calculate_trend_indicators(patents_by_year, patent_data)

        # Assess maturity phase
        maturity_assessment = self._assess_maturity(trend_indicators)

        # Prepare visualization data
        visualizations = self._prepare_visualizations(patents_by_year, patent_data)

        self._logger.info(
            "trend_analysis_complete",
            growth_rate=trend_indicators.get("growth_rate"),
            maturity_phase=maturity_assessment.get("phase"),
        )

        return {
            "trend_indicators": trend_indicators,
            "maturity_assessment": maturity_assessment,
            "visualizations": visualizations,
            "current_step": "visualize",
            "updated_at": datetime.now().isoformat(),
        }

    def _calculate_trend_indicators(
        self,
        patents_by_year: dict[str, list[PatentRecord]],
        all_patents: list[PatentRecord],
    ) -> TrendIndicators:
        """Calculate trend indicators.

        Args:
            patents_by_year: Patents grouped by year
            all_patents: All patent records

        Returns:
            TrendIndicators dictionary
        """
        # Calculate growth rate
        growth_rate = self._calculate_growth_rate(patents_by_year)

        # Calculate concentration (HHI)
        concentration = self._calculate_concentration(all_patents)

        # Determine technology shift
        technology_shift = self._analyze_technology_shift(all_patents)

        # Identify emerging areas
        emerging_areas = self._identify_emerging_areas(patents_by_year)

        return TrendIndicators(
            growth_rate=growth_rate,
            concentration=concentration,
            technology_shift=technology_shift,
            emerging_areas=emerging_areas,
        )

    def _calculate_growth_rate(
        self,
        patents_by_year: dict[str, list[PatentRecord]],
    ) -> float:
        """Calculate compound annual growth rate.

        Args:
            patents_by_year: Patents grouped by year

        Returns:
            Growth rate as decimal (e.g., 0.15 = 15%)
        """
        if len(patents_by_year) < 2:
            return 0.0

        years = sorted(patents_by_year.keys())
        if len(years) < 2:
            return 0.0

        # Use last 5 years or all available
        recent_years = years[-5:] if len(years) >= 5 else years

        first_year = recent_years[0]
        last_year = recent_years[-1]

        first_count = len(patents_by_year.get(first_year, []))
        last_count = len(patents_by_year.get(last_year, []))

        if first_count == 0:
            return 1.0 if last_count > 0 else 0.0

        # CAGR formula
        years_diff = int(last_year) - int(first_year)
        if years_diff <= 0:
            return 0.0

        try:
            cagr = (last_count / first_count) ** (1 / years_diff) - 1
            return round(cagr, 3)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _calculate_concentration(
        self,
        patents: list[PatentRecord],
    ) -> float:
        """Calculate Herfindahl-Hirschman Index (HHI).

        Args:
            patents: All patent records

        Returns:
            HHI value (0 to 1, higher = more concentrated)
        """
        if not patents:
            return 0.0

        # Count by applicant
        applicant_counts: dict[str, int] = {}
        for patent in patents:
            applicant = patent.get("applicant", "Unknown")
            applicant_counts[applicant] = applicant_counts.get(applicant, 0) + 1

        total = len(patents)
        if total == 0:
            return 0.0

        # Calculate HHI
        hhi = sum((count / total) ** 2 for count in applicant_counts.values())

        return round(hhi, 4)

    def _analyze_technology_shift(
        self,
        patents: list[PatentRecord],
    ) -> str:
        """Analyze technology shift direction.

        Args:
            patents: All patent records

        Returns:
            Description of technology shift
        """
        if not patents:
            return "데이터 부족"

        # Group by IPC and year
        ipc_by_year: dict[str, dict[str, int]] = {}

        for patent in patents:
            year = patent.get("filing_date", "")[:4]
            if not year:
                continue

            for ipc in patent.get("ipc_codes", []):
                ipc_main = ipc[:4] if len(ipc) >= 4 else ipc
                if year not in ipc_by_year:
                    ipc_by_year[year] = {}
                ipc_by_year[year][ipc_main] = ipc_by_year[year].get(ipc_main, 0) + 1

        if len(ipc_by_year) < 2:
            return "분석 기간 부족"

        years = sorted(ipc_by_year.keys())
        early_years = years[: len(years) // 2]
        recent_years = years[len(years) // 2 :]

        # Aggregate IPC counts
        early_ipcs: dict[str, int] = {}
        recent_ipcs: dict[str, int] = {}

        for year in early_years:
            for ipc, count in ipc_by_year[year].items():
                early_ipcs[ipc] = early_ipcs.get(ipc, 0) + count

        for year in recent_years:
            for ipc, count in ipc_by_year[year].items():
                recent_ipcs[ipc] = recent_ipcs.get(ipc, 0) + count

        # Find emerging and declining IPCs
        all_ipcs = set(early_ipcs.keys()) | set(recent_ipcs.keys())
        emerging = []
        declining = []

        for ipc in all_ipcs:
            early = early_ipcs.get(ipc, 0)
            recent = recent_ipcs.get(ipc, 0)

            if recent > early * 1.5:  # 50% increase
                emerging.append(ipc)
            elif recent < early * 0.5:  # 50% decrease
                declining.append(ipc)

        if emerging and declining:
            return f"기술 이동 진행 중: {', '.join(emerging[:3])} 증가, {', '.join(declining[:3])} 감소"
        elif emerging:
            return f"신흥 기술 영역: {', '.join(emerging[:3])}"
        elif declining:
            return f"쇠퇴 기술 영역: {', '.join(declining[:3])}"
        else:
            return "기술 분야 안정적"

    def _identify_emerging_areas(
        self,
        patents_by_year: dict[str, list[PatentRecord]],
    ) -> list[str]:
        """Identify emerging technology areas.

        Args:
            patents_by_year: Patents grouped by year

        Returns:
            List of emerging IPC/technology areas
        """
        if len(patents_by_year) < 3:
            return []

        years = sorted(patents_by_year.keys())
        recent_years = years[-3:]  # Last 3 years

        # Count IPCs in recent years
        recent_ipcs: dict[str, int] = {}
        for year in recent_years:
            for patent in patents_by_year[year]:
                for ipc in patent.get("ipc_codes", []):
                    ipc_main = ipc[:4] if len(ipc) >= 4 else ipc
                    recent_ipcs[ipc_main] = recent_ipcs.get(ipc_main, 0) + 1

        # Find IPCs that are new or rapidly growing
        emerging = []

        # Sort by count and take top ones that weren't prominent before
        sorted_ipcs = sorted(recent_ipcs.items(), key=lambda x: x[1], reverse=True)

        for ipc, count in sorted_ipcs[:10]:
            # Check if this IPC was present in earlier years
            early_count = 0
            early_years = years[:-3] if len(years) > 3 else []
            for year in early_years:
                for patent in patents_by_year.get(year, []):
                    if any(ipc in code for code in patent.get("ipc_codes", [])):
                        early_count += 1

            # Emerging if recent count is significantly higher
            if count > early_count * 2 or (early_count == 0 and count >= 3):
                emerging.append(ipc)

        return emerging[:5]  # Top 5 emerging areas

    def _assess_maturity(
        self,
        indicators: TrendIndicators,
    ) -> MaturityPhase:
        """Assess technology maturity phase.

        Args:
            indicators: Calculated trend indicators

        Returns:
            MaturityPhase assessment
        """
        growth_rate = indicators.get("growth_rate", 0.0)

        # Determine phase based on growth rate
        for phase, info in self.MATURITY_PHASES.items():
            min_rate, max_rate = info["growth_rate_range"]
            if min_rate <= growth_rate < max_rate:
                return MaturityPhase(
                    phase=phase,
                    phase_kr=info["phase_kr"],
                    characteristics=info["characteristics"],
                    recommended_strategy=info["recommended_strategy"],
                )

        # Default to maturity if no match
        return MaturityPhase(
            phase="maturity",
            phase_kr="성숙기",
            characteristics=self.MATURITY_PHASES["maturity"]["characteristics"],
            recommended_strategy=self.MATURITY_PHASES["maturity"]["recommended_strategy"],
        )

    def _prepare_visualizations(
        self,
        patents_by_year: dict[str, list[PatentRecord]],
        all_patents: list[PatentRecord],
    ) -> list[dict]:
        """Prepare visualization data.

        Args:
            patents_by_year: Patents grouped by year
            all_patents: All patent records

        Returns:
            List of visualization configurations
        """
        visualizations = []

        # Timeline chart
        timeline_data = self.prepare_timeline_data(patents_by_year)
        visualizations.append({
            "chart_type": "line_chart",
            "title": "연도별 출원 동향",
            "data": timeline_data,
            "file_path": None,
            "format": "html",
        })

        # Applicant distribution
        applicant_counts: dict[str, int] = {}
        for patent in all_patents:
            applicant = patent.get("applicant", "Unknown")
            applicant_counts[applicant] = applicant_counts.get(applicant, 0) + 1

        distribution_data = self.prepare_distribution_data(applicant_counts)
        visualizations.append({
            "chart_type": "bar_chart",
            "title": "주요 출원인 분포",
            "data": distribution_data,
            "file_path": None,
            "format": "html",
        })

        # IPC distribution
        ipc_counts: dict[str, int] = {}
        for patent in all_patents:
            for ipc in patent.get("ipc_codes", []):
                ipc_main = ipc[:4] if len(ipc) >= 4 else ipc
                ipc_counts[ipc_main] = ipc_counts.get(ipc_main, 0) + 1

        ipc_data = self.prepare_distribution_data(ipc_counts)
        visualizations.append({
            "chart_type": "treemap",
            "title": "기술 분야 분포 (IPC)",
            "data": ipc_data,
            "file_path": None,
            "format": "html",
        })

        return visualizations
