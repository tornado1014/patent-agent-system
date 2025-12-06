"""Prior Art Searcher Agent.

Handles E3 (선행기술 분석) step.
Searches and analyzes prior art to identify differentiation points.
"""

from typing import Any

import structlog

from patent_agent.agents.spec_writing.base import BaseSpecWritingAgent
from patent_agent.state.spec_writing import PriorArt, SpecWritingState

logger = structlog.get_logger(__name__)


class PriorArtSearcherAgent(BaseSpecWritingAgent):
    """Agent for searching and analyzing prior art.

    This agent searches patent databases for relevant prior art
    and analyzes the differentiation points of the invention.

    Steps handled:
    - E3: 선행기술 분석
    """

    @property
    def name(self) -> str:
        return "PriorArtSearcher"

    @property
    def description(self) -> str:
        return "선행기술을 검색하고 분석하여 발명의 차별점을 도출하는 에이전트"

    def get_system_prompt(self) -> str:
        return """당신은 특허 전문 변리사로서, 선행기술 조사 및 분석을 수행하는 역할을 합니다.

## 역할
- 발명 관련 선행기술 검색 및 수집
- 선행기술 대비 발명의 차별점 분석
- 신규성/진보성 예비 검토
- 청구항 작성 방향 제안

## 분석 원칙
1. **관련성 우선**: 발명의 핵심 기술과 가장 관련된 선행기술 집중 분석
2. **객관적 평가**: 선행기술의 장점과 한계를 객관적으로 평가
3. **차별점 명확화**: 발명이 선행기술을 개선하는 구체적인 점 명시
4. **전략적 접근**: 청구항 작성에 활용할 수 있는 차별점 도출

## 출력 요구사항
- 각 선행기술에 대해 문헌번호, 명칭, 출원인, 관련성, 한계점 명시
- 발명의 차별점은 기술적 관점에서 구체적으로 서술
- 특허 등록 가능성에 대한 예비 의견 제시"""

    async def process(self, state: SpecWritingState) -> dict[str, Any]:
        """Search and analyze prior art.

        Args:
            state: Current workflow state with invention analysis

        Returns:
            State updates with prior art analysis
        """
        self._logger.info("searching_prior_art", step=state.get("current_step"))

        # Get invention information from state
        key_features = state.get("key_features", [])
        tech_field = state.get("tech_field", "")
        core_technology = state.get("core_technology", "")

        if not key_features and not core_technology:
            self._logger.warning("insufficient_invention_info")
            return {
                "error_messages": ["발명 분석 정보가 부족합니다. E1/E2 단계를 먼저 완료해주세요."],
                "is_error_state": True,
            }

        # Search prior art using available tools
        prior_arts = await self._search_prior_art(
            key_features=key_features,
            tech_field=tech_field,
            core_technology=core_technology,
        )

        # Analyze differentiation points
        differentiation = await self._analyze_differentiation(
            invention_features=key_features,
            prior_arts=prior_arts,
        )

        return {
            "prior_arts": prior_arts,
            "differentiation_points": differentiation,
            "current_step": "E4",
        }

    async def _search_prior_art(
        self,
        key_features: list[str],
        tech_field: str,
        core_technology: str,
    ) -> list[PriorArt]:
        """Search for prior art using available tools.

        Args:
            key_features: Key technical features of the invention
            tech_field: Technical field (IPC codes)
            core_technology: Core technology summary

        Returns:
            List of relevant prior art references
        """
        prior_arts: list[PriorArt] = []

        # Build search query
        search_terms = " ".join(key_features[:3]) if key_features else core_technology

        # Try KIPRIS search first (Korean patents)
        try:
            from patent_agent.tools.kipris import KIPRISClient, KIPRISSearchType

            kipris_client = KIPRISClient()
            kipris_results = await kipris_client.search_patents(
                query=search_terms,
                search_type=KIPRISSearchType.KEYWORD,
                max_results=10,
            )

            for result in kipris_results[:5]:
                # Analyze relevance using LLM
                relevance, limitations = await self._analyze_single_prior_art(
                    title=result.title,
                    abstract=result.abstract,
                    invention_features=key_features,
                )

                prior_arts.append(
                    PriorArt(
                        document_number=result.application_number,
                        title=result.title,
                        applicant=result.applicant,
                        filing_date=result.filing_date.isoformat() if result.filing_date else "",
                        relevance=relevance,
                        limitations=limitations,
                    )
                )

            await kipris_client.close()

        except Exception as e:
            self._logger.warning("kipris_search_failed", error=str(e))

        # Try USPTO search (US patents)
        try:
            from patent_agent.tools.uspto import USPTOClient, USPTOSearchType

            uspto_client = USPTOClient()
            uspto_results = await uspto_client.search_patents(
                query=search_terms,
                search_type=USPTOSearchType.KEYWORD,
                max_results=5,
            )

            for result in uspto_results[:3]:
                relevance, limitations = await self._analyze_single_prior_art(
                    title=result.title,
                    abstract=result.abstract,
                    invention_features=key_features,
                )

                prior_arts.append(
                    PriorArt(
                        document_number=result.patent_number or result.application_number,
                        title=result.title,
                        applicant=result.applicant,
                        filing_date=result.filing_date.isoformat() if result.filing_date else "",
                        relevance=relevance,
                        limitations=limitations,
                    )
                )

        except Exception as e:
            self._logger.warning("uspto_search_failed", error=str(e))

        # If no results from databases, generate placeholder analysis
        if not prior_arts:
            self._logger.info("no_prior_art_found_from_db")
            prior_arts = await self._generate_hypothetical_analysis(key_features, tech_field)

        return prior_arts

    async def _analyze_single_prior_art(
        self,
        title: str,
        abstract: str,
        invention_features: list[str],
    ) -> tuple[str, str]:
        """Analyze a single prior art document.

        Args:
            title: Prior art title
            abstract: Prior art abstract
            invention_features: Features of the current invention

        Returns:
            Tuple of (relevance analysis, limitations)
        """
        prompt = f"""다음 선행기술 문헌을 분석해주세요.

## 선행기술
- 명칭: {title}
- 요약: {abstract}

## 현재 발명의 특징
{chr(10).join(f"- {f}" for f in invention_features)}

## 요청
1. 관련성: 이 선행기술이 현재 발명과 어떻게 관련되는지 1-2문장으로 설명
2. 한계점: 이 선행기술이 해결하지 못하는 문제점 1-2문장으로 설명

형식:
관련성: [내용]
한계점: [내용]"""

        response = await self._invoke_llm(prompt)

        # Parse response
        relevance = ""
        limitations = ""

        for line in response.split("\n"):
            if line.startswith("관련성:"):
                relevance = line.replace("관련성:", "").strip()
            elif line.startswith("한계점:"):
                limitations = line.replace("한계점:", "").strip()

        return relevance or "분석 필요", limitations or "추가 검토 필요"

    async def _analyze_differentiation(
        self,
        invention_features: list[str],
        prior_arts: list[PriorArt],
    ) -> list[str]:
        """Analyze differentiation points from prior art.

        Args:
            invention_features: Features of the current invention
            prior_arts: List of prior art references

        Returns:
            List of differentiation points
        """
        if not prior_arts:
            return ["선행기술 검색 결과 없음 - 추가 조사 권장"]

        # Build context for analysis
        prior_art_summary = "\n".join(
            f"- {pa['title']}: {pa['limitations']}" for pa in prior_arts
        )

        prompt = f"""다음 정보를 바탕으로 발명의 차별점을 도출해주세요.

## 발명의 특징
{chr(10).join(f"- {f}" for f in invention_features)}

## 선행기술 한계점
{prior_art_summary}

## 요청
발명이 선행기술 대비 갖는 차별점을 3-5개 항목으로 정리해주세요.
각 차별점은 청구항 작성에 활용할 수 있도록 기술적으로 구체적이어야 합니다.

형식:
1. [차별점 1]
2. [차별점 2]
..."""

        response = await self._invoke_llm(prompt)

        # Parse differentiation points
        points = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                point = line.split(".", 1)[1].strip()
                if point:
                    points.append(point)

        return points if points else ["차별점 분석 실패 - 수동 검토 필요"]

    async def _generate_hypothetical_analysis(
        self,
        key_features: list[str],
        tech_field: str,
    ) -> list[PriorArt]:
        """Generate hypothetical prior art analysis when DB search fails.

        Args:
            key_features: Key features of the invention
            tech_field: Technical field

        Returns:
            List of hypothetical prior art analysis
        """
        prompt = f"""다음 발명의 기술 분야에서 예상되는 선행기술을 분석해주세요.

## 발명의 특징
{chr(10).join(f"- {f}" for f in key_features)}

## 기술 분야
{tech_field}

## 요청
이 기술 분야에서 일반적으로 존재할 것으로 예상되는 선행기술 유형과
그 한계점을 2-3개 항목으로 분석해주세요.

형식 (각 항목):
유형: [선행기술 유형]
관련성: [현재 발명과의 관련성]
한계점: [해결하지 못하는 문제]"""

        response = await self._invoke_llm(prompt)

        # Parse into PriorArt structures
        prior_arts: list[PriorArt] = []
        current_item: dict[str, str] = {}

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("유형:"):
                if current_item:
                    prior_arts.append(
                        PriorArt(
                            document_number="가상분석",
                            title=current_item.get("유형", "일반 선행기술"),
                            applicant="N/A",
                            filing_date="",
                            relevance=current_item.get("관련성", ""),
                            limitations=current_item.get("한계점", ""),
                        )
                    )
                current_item = {"유형": line.replace("유형:", "").strip()}
            elif line.startswith("관련성:"):
                current_item["관련성"] = line.replace("관련성:", "").strip()
            elif line.startswith("한계점:"):
                current_item["한계점"] = line.replace("한계점:", "").strip()

        # Don't forget the last item
        if current_item:
            prior_arts.append(
                PriorArt(
                    document_number="가상분석",
                    title=current_item.get("유형", "일반 선행기술"),
                    applicant="N/A",
                    filing_date="",
                    relevance=current_item.get("관련성", ""),
                    limitations=current_item.get("한계점", ""),
                )
            )

        return prior_arts
