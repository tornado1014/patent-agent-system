"""CompetitorProfilerAgent for Patent Analysis workflow.

Responsible for analyzing competitor patent strategies:
- Building competitor profiles
- Patent portfolio comparison
- Citation network analysis
- Technology overlap assessment
"""

from datetime import datetime
from typing import Any

import structlog

from patent_agent.agents.analysis.base import BaseAnalysisAgent, PatentScorer
from patent_agent.state.analysis import (
    AnalysisStep,
    CompetitorMetrics,
    CompetitorProfile,
    PatentAnalysisState,
    PatentRecord,
)

logger = structlog.get_logger(__name__)


class CompetitorProfilerAgent(BaseAnalysisAgent):
    """Agent for analyzing competitor patent strategies.

    This agent is responsible for:
    1. Building competitor profiles
    2. Calculating market share and positioning
    3. Analyzing citation networks
    4. Identifying technology overlaps

    Example:
        >>> agent = CompetitorProfilerAgent()
        >>> state = {
        ...     "analysis_type": "competitor",
        ...     "target": "삼성전자",
        ...     "patent_data": [...],
        ... }
        >>> result = await agent.process(state)
    """

    @property
    def name(self) -> str:
        return "CompetitorProfilerAgent"

    @property
    def description(self) -> str:
        return "경쟁사 특허 전략 분석 및 프로파일링"

    @property
    def step(self) -> AnalysisStep:
        return "analyze"

    def get_system_prompt(self) -> str:
        return """당신은 경쟁사 특허 전략 분석 전문가입니다.

## 역할
- 경쟁사별 특허 포트폴리오 분석
- 특허 점유율 및 포지셔닝 평가
- 인용 네트워크 분석
- 기술 중첩 영역 식별

## 분석 프레임워크

### 1. 포트폴리오 분석
- 보유 특허 수
- 기술 분포 (IPC/CPC)
- 핵심 특허 식별
- 출원 트렌드

### 2. 시장 포지셔닝
- 특허 점유율 (해당 기술분야 내)
- 기술 리더십 영역
- 방어적/공격적 특허 비율

### 3. 인용 네트워크
- 피인용 빈도 (영향력)
- 자기 인용 비율
- 경쟁사 간 인용 관계

### 4. 기술 중첩
- 동일 IPC 분야 출원
- 유사 청구항 분석
- 잠재적 분쟁 영역

## 경쟁 관계 유형

| 유형 | 특징 | 전략적 시사점 |
|------|------|---------------|
| 직접 경쟁 | 동일 기술/시장 | 차별화, 회피 설계 |
| 인접 경쟁 | 유사 기술/다른 시장 | 협력 또는 진입 |
| 잠재 경쟁 | 다른 기술/유사 시장 | 모니터링 |

## 출력 형식
각 경쟁사에 대해:
1. 기본 프로필 (특허 수, 점유율)
2. 핵심 기술 영역
3. 최근 동향
4. 전략적 시사점
"""

    async def process(self, state: PatentAnalysisState) -> dict[str, Any]:
        """Analyze competitors and build profiles.

        Args:
            state: Current workflow state

        Returns:
            State updates with competitor analysis
        """
        self._logger.info("starting_competitor_analysis")

        patent_data = state.get("patent_data", [])
        target = state.get("target", "")
        scope = state.get("scope", {})

        if not patent_data:
            return {
                "is_error_state": True,
                "error_messages": ["분석할 특허 데이터가 없습니다."],
                "current_step": "analyze",
            }

        # Group patents by applicant
        patents_by_applicant = self._group_by_applicant(patent_data)

        # Build competitor profiles
        competitor_profiles = []
        for applicant, patents in patents_by_applicant.items():
            profile = self._build_competitor_profile(applicant, patents, len(patent_data))
            competitor_profiles.append(profile)

        # Sort by patent count (descending)
        competitor_profiles.sort(key=lambda x: x["patent_count"], reverse=True)

        # Calculate competitor metrics
        competitor_metrics = self._calculate_competitor_metrics(
            patents_by_applicant, patent_data
        )

        # Identify target company position
        target_position = self._get_target_position(target, competitor_profiles)

        # Prepare visualizations
        visualizations = self._prepare_visualizations(competitor_profiles, competitor_metrics)

        self._logger.info(
            "competitor_analysis_complete",
            total_competitors=len(competitor_profiles),
            target_position=target_position,
        )

        return {
            "competitor_profiles": competitor_profiles,
            "competitor_metrics": competitor_metrics,
            "visualizations": visualizations,
            "current_step": "visualize",
            "updated_at": datetime.now().isoformat(),
        }

    def _group_by_applicant(
        self,
        patents: list[PatentRecord],
    ) -> dict[str, list[PatentRecord]]:
        """Group patents by applicant.

        Args:
            patents: All patent records

        Returns:
            Dictionary mapping applicant to patents
        """
        by_applicant: dict[str, list[PatentRecord]] = {}

        for patent in patents:
            applicant = patent.get("applicant", "Unknown")
            # Normalize applicant name (basic normalization)
            applicant = applicant.strip().upper()
            if applicant not in by_applicant:
                by_applicant[applicant] = []
            by_applicant[applicant].append(patent)

        return by_applicant

    def _build_competitor_profile(
        self,
        applicant: str,
        patents: list[PatentRecord],
        total_patents: int,
    ) -> CompetitorProfile:
        """Build a competitor profile.

        Args:
            applicant: Applicant name
            patents: Applicant's patents
            total_patents: Total patents in dataset

        Returns:
            CompetitorProfile
        """
        patent_count = len(patents)
        market_share = patent_count / total_patents if total_patents > 0 else 0

        # Identify key technologies (top IPC codes)
        ipc_counts: dict[str, int] = {}
        for patent in patents:
            for ipc in patent.get("ipc_codes", []):
                ipc_main = ipc[:4] if len(ipc) >= 4 else ipc
                ipc_counts[ipc_main] = ipc_counts.get(ipc_main, 0) + 1

        key_technologies = sorted(ipc_counts, key=ipc_counts.get, reverse=True)[:5]

        # Identify core patents (top by citation)
        sorted_patents = sorted(
            patents,
            key=lambda p: p.get("citation_count", 0),
            reverse=True,
        )
        core_patents = [p.get("patent_number", "") for p in sorted_patents[:5]]

        # Analyze recent trends
        recent_trends = self._analyze_recent_trends(patents)

        return CompetitorProfile(
            company_name=applicant,
            patent_count=patent_count,
            market_share=round(market_share, 4),
            key_technologies=key_technologies,
            core_patents=core_patents,
            recent_trends=recent_trends,
        )

    def _analyze_recent_trends(self, patents: list[PatentRecord]) -> str:
        """Analyze recent filing trends for a competitor.

        Args:
            patents: Competitor's patents

        Returns:
            Trend description
        """
        if not patents:
            return "데이터 없음"

        # Group by year
        by_year: dict[str, int] = {}
        for patent in patents:
            year = patent.get("filing_date", "")[:4]
            if year:
                by_year[year] = by_year.get(year, 0) + 1

        if len(by_year) < 2:
            return "출원 이력 부족"

        years = sorted(by_year.keys())
        recent_years = years[-3:] if len(years) >= 3 else years

        recent_count = sum(by_year.get(y, 0) for y in recent_years)
        avg_recent = recent_count / len(recent_years)

        early_years = years[:-3] if len(years) > 3 else []
        if early_years:
            early_count = sum(by_year.get(y, 0) for y in early_years)
            avg_early = early_count / len(early_years)

            if avg_recent > avg_early * 1.5:
                return "출원 급증 - 적극적 R&D 투자"
            elif avg_recent > avg_early * 1.1:
                return "출원 증가 - 꾸준한 기술 개발"
            elif avg_recent < avg_early * 0.5:
                return "출원 감소 - 기술 전환 또는 축소"
            else:
                return "출원 안정 - 유지 전략"
        else:
            return f"최근 연평균 {avg_recent:.1f}건 출원"

    def _calculate_competitor_metrics(
        self,
        patents_by_applicant: dict[str, list[PatentRecord]],
        all_patents: list[PatentRecord],
    ) -> CompetitorMetrics:
        """Calculate competitor comparison metrics.

        Args:
            patents_by_applicant: Patents grouped by applicant
            all_patents: All patents

        Returns:
            CompetitorMetrics
        """
        total = len(all_patents)

        # Patent share
        patent_share = {
            applicant: len(patents) / total if total > 0 else 0
            for applicant, patents in patents_by_applicant.items()
        }

        # Technology overlap (based on IPC codes)
        technology_overlap = self._calculate_technology_overlap(patents_by_applicant)

        # Citation network (simplified)
        citation_network = self._build_citation_network(all_patents)

        # Litigation history (placeholder - would need external data)
        litigation_history: list[dict] = []

        return CompetitorMetrics(
            patent_share=patent_share,
            technology_overlap=technology_overlap,
            citation_network=citation_network,
            litigation_history=litigation_history,
        )

    def _calculate_technology_overlap(
        self,
        patents_by_applicant: dict[str, list[PatentRecord]],
    ) -> dict[str, float]:
        """Calculate technology overlap between competitors.

        Args:
            patents_by_applicant: Patents grouped by applicant

        Returns:
            Dictionary of applicant pairs to overlap ratio
        """
        # Get IPC sets for each applicant
        applicant_ipcs: dict[str, set] = {}
        for applicant, patents in patents_by_applicant.items():
            ipcs = set()
            for patent in patents:
                for ipc in patent.get("ipc_codes", []):
                    ipc_main = ipc[:4] if len(ipc) >= 4 else ipc
                    ipcs.add(ipc_main)
            applicant_ipcs[applicant] = ipcs

        # Calculate pairwise overlap
        overlap: dict[str, float] = {}
        applicants = list(applicant_ipcs.keys())

        for i, app1 in enumerate(applicants):
            for app2 in applicants[i + 1 :]:
                set1 = applicant_ipcs[app1]
                set2 = applicant_ipcs[app2]

                if not set1 or not set2:
                    continue

                # Jaccard similarity
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                similarity = intersection / union if union > 0 else 0

                key = f"{app1}|{app2}"
                overlap[key] = round(similarity, 4)

        return overlap

    def _build_citation_network(
        self,
        patents: list[PatentRecord],
    ) -> dict:
        """Build citation network structure.

        Args:
            patents: All patents

        Returns:
            Network structure dictionary
        """
        # Simplified network - in production would analyze actual citations
        nodes = []
        edges = []

        # Create nodes for each applicant
        applicant_ids: dict[str, str] = {}
        for patent in patents:
            applicant = patent.get("applicant", "Unknown")
            if applicant not in applicant_ids:
                node_id = f"n{len(applicant_ids)}"
                applicant_ids[applicant] = node_id
                nodes.append({
                    "id": node_id,
                    "label": applicant,
                    "size": 0,
                })

            # Update node size based on patent count
            node_id = applicant_ids[applicant]
            for node in nodes:
                if node["id"] == node_id:
                    node["size"] += 1
                    break

        return {
            "nodes": nodes,
            "edges": edges,  # Would be populated with actual citation data
        }

    def _get_target_position(
        self,
        target: str,
        profiles: list[CompetitorProfile],
    ) -> dict:
        """Get target company's competitive position.

        Args:
            target: Target company name
            profiles: All competitor profiles

        Returns:
            Position information
        """
        target_upper = target.upper().strip()

        for i, profile in enumerate(profiles):
            if target_upper in profile["company_name"].upper():
                return {
                    "rank": i + 1,
                    "total": len(profiles),
                    "patent_count": profile["patent_count"],
                    "market_share": profile["market_share"],
                }

        return {
            "rank": 0,
            "total": len(profiles),
            "note": "대상 기업을 찾을 수 없음",
        }

    def _prepare_visualizations(
        self,
        profiles: list[CompetitorProfile],
        metrics: CompetitorMetrics,
    ) -> list[dict]:
        """Prepare visualization data.

        Args:
            profiles: Competitor profiles
            metrics: Competitor metrics

        Returns:
            List of visualization configurations
        """
        visualizations = []

        # Market share pie chart
        share_data = {
            "labels": [p["company_name"][:20] for p in profiles[:10]],
            "datasets": [{
                "data": [p["market_share"] for p in profiles[:10]],
            }],
        }
        visualizations.append({
            "chart_type": "bar_chart",
            "title": "특허 점유율 (상위 10개사)",
            "data": share_data,
            "file_path": None,
            "format": "html",
        })

        # Technology heatmap (placeholder)
        visualizations.append({
            "chart_type": "heatmap",
            "title": "기술 중첩 매트릭스",
            "data": metrics.get("technology_overlap", {}),
            "file_path": None,
            "format": "html",
        })

        # Citation network
        visualizations.append({
            "chart_type": "network_graph",
            "title": "인용 네트워크",
            "data": metrics.get("citation_network", {}),
            "file_path": None,
            "format": "html",
        })

        return visualizations
