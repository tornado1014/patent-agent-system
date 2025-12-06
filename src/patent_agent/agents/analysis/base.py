"""Base agent class for Patent Analysis agents.

Provides common functionality for analysis workflow agents including:
- Multi-DB data collection (KIPRIS, USPTO, EPO, Google Patents)
- Claim element parsing and mapping
- Visualization generation
- Report generation

Supports 5 analysis types:
- Portfolio: 특정 기업/기관의 특허 현황
- Trend: 특정 기술분야의 출원/등록 트렌드
- Competitor: 경쟁사 특허 전략 파악
- Infringement: 제품 vs 특허 청구항 대비
- Invalidity: 특허 무효화 가능성 검토
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from patent_agent.config import get_settings
from patent_agent.state.analysis import (
    ANALYSIS_TYPE_INFO,
    CORE_PATENT_SCORING,
    AnalysisStep,
    AnalysisType,
    ClaimChart,
    ClaimElement,
    PatentAnalysisState,
    PatentRecord,
)
from patent_agent.state.base import ConfidenceLevel

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS GUIDELINES
# ═══════════════════════════════════════════════════════════════════════════


class AnalysisGuidelines:
    """Guidelines for patent analysis."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the complete analysis guidelines system prompt."""
        return """## 특허 분석 가이드라인

### 1. 분석의 핵심 원칙
- **객관성**: 데이터에 기반한 객관적 분석
- **완전성**: 가능한 모든 관련 특허를 포함
- **명확성**: 분석 결과를 명확하게 제시
- **근거 추적성**: 모든 주장에 근거 명시

### 2. 분석 유형별 접근법

#### 2.1 포트폴리오 분석 (Portfolio)
- 보유 특허 현황 파악
- 기술 분포 분석 (IPC/CPC 기준)
- 출원 트렌드 시계열 분석
- 핵심 특허 식별 (인용수, 청구항 범위 등)

#### 2.2 기술 동향 분석 (Trend)
- 시계열 출원 동향
- 주요 출원인 식별
- 기술 성숙도 평가 (태동기/성장기/성숙기/쇠퇴기)
- 신흥 기술 영역 발굴

#### 2.3 경쟁사 분석 (Competitor)
- 특허 포지셔닝 매핑
- R&D 방향 추론
- 협력/경쟁 관계 분석
- 인용 네트워크 분석

#### 2.4 침해 분석 (Infringement)
- 청구항 요소 분해
- 제품 기능 매핑
- 문언 침해 분석
- 균등론 적용 검토
- 회피 설계 제안

#### 2.5 무효 분석 (Invalidity)
- 선행기술 조사
- 신규성/진보성 분석
- 청구항별 무효 논거 구성
- 성공 가능성 평가

### 3. 핵심 특허 평가 기준

| 요소 | 가중치 | 설명 |
|------|--------|------|
| 인용 횟수 | 25% | 후방 인용 수 |
| 청구항 범위 | 20% | 독립항 범위의 광협 |
| 특허 가족 크기 | 15% | 다국적 출원 수 |
| 잔여 권리 기간 | 15% | 만료까지 남은 기간 |
| 기술 중심성 | 15% | 기술 분야 내 위치 |
| 법적 안정성 | 10% | 무효 심판 이력 등 |

### 4. 신뢰도 표시
- 🟢 확실 (90%+): 명확한 데이터 기반
- 🟡 가능 (60-89%): 해석 여지 있음
- 🔴 불확실 (60% 미만): 추가 검토 필요

### 5. 침해 유형 판단 기준

#### 5.1 문언 침해 (Literal Infringement)
- 청구항의 모든 구성요소가 제품에 존재
- All-Elements Rule 적용

#### 5.2 균등론 (Doctrine of Equivalents)
- 기능-방법-결과 (FWR) 테스트
- 동일한 기능, 실질적으로 동일한 방법, 실질적으로 동일한 결과

### 6. 보고서 구성
1. 요약 (Executive Summary)
2. 분석 범위 및 방법론
3. 주요 발견사항
4. 상세 분석 결과
5. 시각화 자료
6. 권고사항
7. 부록 (데이터 출처, 참고 문헌)
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLAIM ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


class ClaimAnalyzer:
    """Utilities for claim element analysis."""

    @staticmethod
    def parse_claim_elements(claim_text: str) -> list[ClaimElement]:
        """Parse a claim into its constituent elements.

        Args:
            claim_text: Full text of a patent claim

        Returns:
            List of ClaimElement objects
        """
        elements = []

        # Split claim into parts
        parts = claim_text.split(",")
        if not parts:
            return elements

        # First part is typically the preamble
        preamble = parts[0].strip()
        if preamble:
            elements.append(
                ClaimElement(
                    element_id="E0",
                    text=preamble,
                    type="preamble",
                )
            )

        # Remaining parts are body/limitations
        for i, part in enumerate(parts[1:], start=1):
            part = part.strip()
            if not part:
                continue

            # Determine element type
            elem_type = "limitation"
            if any(kw in part.lower() for kw in ["comprising", "including", "having"]):
                elem_type = "body"

            elements.append(
                ClaimElement(
                    element_id=f"E{i}",
                    text=part,
                    type=elem_type,
                )
            )

        return elements

    @staticmethod
    def calculate_claim_breadth(claim_elements: list[ClaimElement]) -> float:
        """Calculate the relative breadth of a claim.

        Fewer limitations = broader claim.

        Args:
            claim_elements: Parsed claim elements

        Returns:
            Score between 0 and 1 (higher = broader)
        """
        limitations = [e for e in claim_elements if e["type"] == "limitation"]
        num_limitations = len(limitations)

        # Typical claims have 3-10 limitations
        # Fewer = broader, more = narrower
        if num_limitations <= 2:
            return 1.0
        elif num_limitations <= 4:
            return 0.8
        elif num_limitations <= 6:
            return 0.6
        elif num_limitations <= 8:
            return 0.4
        else:
            return 0.2


class PatentScorer:
    """Score patents based on multiple factors."""

    SCORING_WEIGHTS = CORE_PATENT_SCORING

    @classmethod
    def calculate_core_score(
        cls,
        patent: PatentRecord,
        field_avg_citations: float = 10.0,
        field_avg_family: float = 3.0,
    ) -> float:
        """Calculate core patent score.

        Args:
            patent: Patent record to score
            field_avg_citations: Average citation count in the field
            field_avg_family: Average family size in the field

        Returns:
            Score between 0 and 100
        """
        score = 0.0

        # Citation score (normalized to field average)
        citation_score = min(100, (patent["citation_count"] / field_avg_citations) * 50)
        score += citation_score * cls.SCORING_WEIGHTS["citation_count"]

        # Family size score
        family_score = min(100, (patent["family_size"] / field_avg_family) * 50)
        score += family_score * cls.SCORING_WEIGHTS["family_size"]

        # Remaining life (assuming 20 year term from filing)
        from datetime import datetime

        try:
            filing = datetime.fromisoformat(patent["filing_date"])
            now = datetime.now()
            years_elapsed = (now - filing).days / 365
            remaining = max(0, 20 - years_elapsed)
            life_score = (remaining / 20) * 100
        except (ValueError, TypeError):
            life_score = 50  # Default if date parsing fails

        score += life_score * cls.SCORING_WEIGHTS["remaining_life"]

        # Claim breadth (estimate based on claim count - simplified)
        claim_count = len(patent.get("claims", []))
        breadth_score = min(100, claim_count * 5)  # More claims often = more coverage
        score += breadth_score * cls.SCORING_WEIGHTS["claim_breadth"]

        # Technology centrality (placeholder - would need network analysis)
        centrality_score = 50  # Default to medium
        score += centrality_score * cls.SCORING_WEIGHTS["technology_centrality"]

        # Legal stability (based on status)
        status = patent.get("legal_status", "pending")
        stability_scores = {
            "granted": 100,
            "pending": 70,
            "expired": 30,
            "withdrawn": 10,
            "abandoned": 10,
        }
        stability_score = stability_scores.get(status, 50)
        score += stability_score * cls.SCORING_WEIGHTS["legal_stability"]

        return min(100, max(0, score))


# ═══════════════════════════════════════════════════════════════════════════
# BASE ANALYSIS AGENT
# ═══════════════════════════════════════════════════════════════════════════


class BaseAnalysisAgent(ABC):
    """Base class for Patent Analysis agents.

    All agents in the analysis workflow inherit from this class.
    It provides:
    - LLM integration with analysis guidelines
    - Multi-analysis type support
    - Patent scoring utilities
    - Claim element analysis
    - Visualization helpers
    """

    # Class-level references
    ANALYSIS_INFO = ANALYSIS_TYPE_INFO
    SCORING_WEIGHTS = CORE_PATENT_SCORING
    CLAIM_ANALYZER = ClaimAnalyzer
    PATENT_SCORER = PatentScorer

    def __init__(
        self,
        llm: BaseChatModel | None = None,
    ):
        """Initialize the agent.

        Args:
            llm: Language model to use. If not provided, creates default from settings.
        """
        self._llm = llm
        self._logger = logger.bind(agent=self.__class__.__name__)

    @property
    def llm(self) -> BaseChatModel:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = self._create_default_llm()
        return self._llm

    def _create_default_llm(self) -> BaseChatModel:
        """Create default LLM from settings."""
        settings = get_settings()

        if settings.llm.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )
        else:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for logging and identification."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description."""
        ...

    @property
    @abstractmethod
    def step(self) -> AnalysisStep:
        """Step this agent handles in the workflow."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        ...

    @abstractmethod
    async def process(self, state: PatentAnalysisState) -> dict[str, Any]:
        """Process the current state and return updates.

        Args:
            state: Current workflow state

        Returns:
            Dictionary of state updates
        """
        ...

    def _get_full_system_prompt(self) -> str:
        """Get complete system prompt including guidelines."""
        base_prompt = self.get_system_prompt()
        guideline_prompt = AnalysisGuidelines.get_system_prompt()

        return f"""{base_prompt}

---

{guideline_prompt}"""

    async def _invoke_llm(
        self,
        user_message: str,
        system_prompt: str | None = None,
    ) -> str:
        """Invoke the LLM with the given message.

        Args:
            user_message: User message to send
            system_prompt: Optional custom system prompt

        Returns:
            LLM response as string
        """
        system = system_prompt or self._get_full_system_prompt()

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_message),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def _invoke_llm_structured(
        self,
        prompt_template: ChatPromptTemplate,
        variables: dict[str, Any],
        output_parser: Any | None = None,
    ) -> Any:
        """Invoke LLM with structured prompt and optional parsing.

        Args:
            prompt_template: Prompt template to use
            variables: Variables to fill in the template
            output_parser: Optional output parser

        Returns:
            Parsed response
        """
        chain = prompt_template | self.llm
        if output_parser:
            chain = chain | output_parser
        else:
            chain = chain | StrOutputParser()

        return await chain.ainvoke(variables)

    # ─────────────────────────────────────────────────────────────
    # Analysis Type Helpers
    # ─────────────────────────────────────────────────────────────

    def get_analysis_info(self, analysis_type: AnalysisType) -> dict:
        """Get information about an analysis type.

        Args:
            analysis_type: Type of analysis

        Returns:
            Analysis type information
        """
        return self.ANALYSIS_INFO.get(analysis_type, {})

    def is_claim_intensive_analysis(self, analysis_type: AnalysisType) -> bool:
        """Check if analysis type requires detailed claim analysis.

        Args:
            analysis_type: Type of analysis

        Returns:
            True if claim-intensive
        """
        return analysis_type in ("infringement", "invalidity")

    # ─────────────────────────────────────────────────────────────
    # Confidence Indicators
    # ─────────────────────────────────────────────────────────────

    def get_confidence_indicator(self, level: ConfidenceLevel) -> str:
        """Get emoji indicator for confidence level.

        Args:
            level: Confidence level

        Returns:
            Emoji indicator
        """
        indicators = {
            "확실": "🟢",
            "가능": "🟡",
            "불확실": "🔴",
        }
        return indicators.get(level, "🔴")

    def score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Convert a score (0-1) to confidence level.

        Args:
            score: Score between 0 and 1

        Returns:
            Confidence level
        """
        if score >= 0.9:
            return "확실"
        elif score >= 0.6:
            return "가능"
        else:
            return "불확실"

    # ─────────────────────────────────────────────────────────────
    # Patent Record Helpers
    # ─────────────────────────────────────────────────────────────

    def filter_patents_by_jurisdiction(
        self,
        patents: list[PatentRecord],
        jurisdictions: list[str],
    ) -> list[PatentRecord]:
        """Filter patents by jurisdiction.

        Args:
            patents: List of patent records
            jurisdictions: List of jurisdiction codes (KR, US, EP, etc.)

        Returns:
            Filtered patent list
        """
        if not jurisdictions:
            return patents

        filtered = []
        for patent in patents:
            # Extract jurisdiction from patent number
            patent_num = patent.get("patent_number", "")
            for jur in jurisdictions:
                if patent_num.upper().startswith(jur.upper()):
                    filtered.append(patent)
                    break

        return filtered

    def group_patents_by_year(
        self,
        patents: list[PatentRecord],
        date_field: str = "filing_date",
    ) -> dict[str, list[PatentRecord]]:
        """Group patents by year.

        Args:
            patents: List of patent records
            date_field: Date field to use for grouping

        Returns:
            Dictionary mapping year to patent list
        """
        by_year: dict[str, list[PatentRecord]] = {}

        for patent in patents:
            date_str = patent.get(date_field, "")
            if date_str:
                try:
                    year = date_str[:4]  # Extract year from ISO date
                    if year not in by_year:
                        by_year[year] = []
                    by_year[year].append(patent)
                except (IndexError, TypeError):
                    continue

        return by_year

    def group_patents_by_ipc(
        self,
        patents: list[PatentRecord],
        level: int = 4,  # Main group level
    ) -> dict[str, list[PatentRecord]]:
        """Group patents by IPC code.

        Args:
            patents: List of patent records
            level: IPC code level (4=main group, 7=subgroup)

        Returns:
            Dictionary mapping IPC code to patent list
        """
        by_ipc: dict[str, list[PatentRecord]] = {}

        for patent in patents:
            ipc_codes = patent.get("ipc_codes", [])
            for ipc in ipc_codes:
                # Truncate to specified level
                ipc_key = ipc[:level] if len(ipc) >= level else ipc
                if ipc_key not in by_ipc:
                    by_ipc[ipc_key] = []
                by_ipc[ipc_key].append(patent)

        return by_ipc

    # ─────────────────────────────────────────────────────────────
    # Claim Chart Helpers
    # ─────────────────────────────────────────────────────────────

    def create_claim_chart(
        self,
        claim_number: int,
        claim_text: str,
    ) -> ClaimChart:
        """Create an empty claim chart for mapping.

        Args:
            claim_number: Claim number
            claim_text: Full claim text

        Returns:
            ClaimChart with parsed elements
        """
        elements = self.CLAIM_ANALYZER.parse_claim_elements(claim_text)

        return ClaimChart(
            claim_number=claim_number,
            claim_elements=elements,
            mappings=[],
            overall_result="no_match",
        )

    def evaluate_infringement(
        self,
        claim_chart: ClaimChart,
    ) -> str:
        """Evaluate infringement based on claim chart mappings.

        Args:
            claim_chart: Completed claim chart

        Returns:
            Infringement type: "literal", "doctrine_of_equivalents", or "none"
        """
        mappings = claim_chart.get("mappings", [])
        elements = claim_chart.get("claim_elements", [])

        if not mappings or not elements:
            return "none"

        # Count matched elements
        literal_matches = 0
        equivalent_matches = 0

        for mapping in mappings:
            match_type = mapping.get("match_type", "none")
            if match_type == "literal":
                literal_matches += 1
            elif match_type == "equivalent":
                equivalent_matches += 1

        # All elements must be matched for infringement
        total_elements = len(elements)

        if literal_matches == total_elements:
            return "literal"
        elif literal_matches + equivalent_matches == total_elements:
            return "doctrine_of_equivalents"
        else:
            return "none"

    # ─────────────────────────────────────────────────────────────
    # Visualization Data Helpers
    # ─────────────────────────────────────────────────────────────

    def prepare_timeline_data(
        self,
        patents_by_year: dict[str, list[PatentRecord]],
    ) -> dict:
        """Prepare data for timeline visualization.

        Args:
            patents_by_year: Patents grouped by year

        Returns:
            Data structure for timeline chart
        """
        years = sorted(patents_by_year.keys())

        return {
            "labels": years,
            "datasets": [
                {
                    "label": "출원 건수",
                    "data": [len(patents_by_year[y]) for y in years],
                }
            ],
        }

    def prepare_distribution_data(
        self,
        distribution: dict[str, int],
        top_n: int = 10,
    ) -> dict:
        """Prepare data for distribution visualization.

        Args:
            distribution: Dictionary of category to count
            top_n: Number of top categories to include

        Returns:
            Data structure for distribution chart
        """
        # Sort by count and take top N
        sorted_items = sorted(
            distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "건수",
                    "data": values,
                }
            ],
        }
