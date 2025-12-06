"""Base agent class for Prior Art Search agents.

Provides common functionality for prior art search workflow agents including:
- Multi-database search integration
- Relevance scoring utilities
- IPC/CPC classification helpers
- Evidence tracking for analysis results
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from patent_agent.config import get_settings
from patent_agent.state.base import ConfidenceLevel
from patent_agent.state.prior_art import (
    DatabaseID,
    PriorArtSearchState,
    PriorArtStep,
    SearchType,
)

logger = structlog.get_logger(__name__)


# IPC Classification Guide
IPC_SECTIONS = {
    "A": "생활필수품",
    "B": "처리 조작; 운수",
    "C": "화학; 야금",
    "D": "섬유; 지류",
    "E": "고정 구조물",
    "F": "기계 공학; 조명; 가열; 무기; 폭파",
    "G": "물리학",
    "H": "전기",
}

# Common IPC subclasses for technology domains
COMMON_IPC_CODES = {
    "AI/ML": ["G06N", "G06F18", "G06K9"],
    "networking": ["H04L", "H04W", "H04N"],
    "semiconductor": ["H01L", "H01S"],
    "battery": ["H01M", "H02J"],
    "medical": ["A61B", "A61K", "A61M"],
    "automotive": ["B60W", "B60R", "B60K"],
    "display": ["G09G", "H10K"],
    "robotics": ["B25J", "G05B"],
}


class PriorArtGuidelines:
    """Guidelines for prior art search agents."""

    SEARCH_GUIDELINES = """
## 선행기술조사 원칙

### 검색 원칙
1. 복수 DB 병렬 검색 필수 (KIPRIS, USPTO, Google Patents)
2. IPC/CPC 코드 기반 분류 검색 병행
3. 키워드 확장 (동의어, 상위/하위 개념)
4. 출원일 기준 시간적 범위 설정

### 분석 원칙
1. 관련성 점수 0.7 이상 문헌 상세 분석
2. 청구항별 구성요소 대응 분석 필수
3. 신규성/진보성 별도 평가

### 출력 형식
1. 대비표 형식 필수 (청구항 구성요소 vs 선행기술)
2. 신뢰도 표시: 🟢확실/🟡가능/🔴불확실
3. 명확한 근거 제시
"""

    RELEVANCE_SCORING = {
        "technical_overlap": 0.4,  # 기술적 특징 중복도
        "claim_coverage": 0.3,  # 청구항 구성요소 포함도
        "temporal_relevance": 0.15,  # 시간적 관련성
        "applicant_relevance": 0.15,  # 출원인/발명자 관련성
    }

    def get_system_prompt(self) -> str:
        """Get the guidelines system prompt."""
        return self.SEARCH_GUIDELINES

    def calculate_weighted_relevance(
        self,
        technical_overlap: float,
        claim_coverage: float,
        temporal_relevance: float,
        applicant_relevance: float,
    ) -> float:
        """Calculate weighted relevance score.

        Args:
            technical_overlap: 0-1 score for technical feature overlap
            claim_coverage: 0-1 score for claim element coverage
            temporal_relevance: 0-1 score for temporal relevance
            applicant_relevance: 0-1 score for applicant/inventor relevance

        Returns:
            Weighted relevance score (0-1)
        """
        weights = self.RELEVANCE_SCORING
        return (
            technical_overlap * weights["technical_overlap"]
            + claim_coverage * weights["claim_coverage"]
            + temporal_relevance * weights["temporal_relevance"]
            + applicant_relevance * weights["applicant_relevance"]
        )


class BasePriorArtAgent(ABC):
    """Base class for Prior Art Search agents.

    All agents in the prior art search workflow inherit from this class.
    It provides:
    - LLM integration with search guidelines
    - Multi-database search coordination
    - Relevance scoring utilities
    - IPC/CPC classification helpers
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        guidelines: PriorArtGuidelines | None = None,
    ):
        """Initialize the agent.

        Args:
            llm: Language model to use. If not provided, creates default from settings.
            guidelines: Guidelines to follow. If not provided, uses default.
        """
        self._llm = llm
        self._guidelines = guidelines or PriorArtGuidelines()
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
    def step(self) -> PriorArtStep:
        """Step this agent handles in the workflow."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        ...

    @abstractmethod
    async def process(self, state: PriorArtSearchState) -> dict[str, Any]:
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
        guideline_prompt = self._guidelines.get_system_prompt()

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
    # IPC/CPC Classification Helpers
    # ─────────────────────────────────────────────────────────────

    def get_ipc_section_name(self, ipc_code: str) -> str:
        """Get the section name for an IPC code.

        Args:
            ipc_code: IPC code (e.g., "G06N")

        Returns:
            Section name in Korean
        """
        if not ipc_code:
            return "알 수 없음"
        section = ipc_code[0].upper()
        return IPC_SECTIONS.get(section, "알 수 없음")

    def suggest_ipc_codes(self, tech_domain: str) -> list[str]:
        """Suggest IPC codes for a technology domain.

        Args:
            tech_domain: Technology domain description

        Returns:
            List of suggested IPC codes
        """
        domain_lower = tech_domain.lower()

        for domain_key, codes in COMMON_IPC_CODES.items():
            if domain_key in domain_lower:
                return codes

        # Default to generic computer technology
        return ["G06F", "G06N"]

    # ─────────────────────────────────────────────────────────────
    # Relevance Scoring Utilities
    # ─────────────────────────────────────────────────────────────

    def calculate_relevance_score(
        self,
        technical_overlap: float,
        claim_coverage: float,
        temporal_relevance: float = 1.0,
        applicant_relevance: float = 0.0,
    ) -> float:
        """Calculate weighted relevance score.

        Args:
            technical_overlap: Technical feature overlap (0-1)
            claim_coverage: Claim element coverage (0-1)
            temporal_relevance: Temporal relevance (0-1)
            applicant_relevance: Applicant/inventor relevance (0-1)

        Returns:
            Weighted relevance score (0-1)
        """
        return self._guidelines.calculate_weighted_relevance(
            technical_overlap,
            claim_coverage,
            temporal_relevance,
            applicant_relevance,
        )

    def relevance_to_confidence(self, relevance: float) -> ConfidenceLevel:
        """Convert relevance score to confidence level.

        Args:
            relevance: Relevance score (0-1)

        Returns:
            Confidence level
        """
        if relevance >= 0.8:
            return "확실"
        elif relevance >= 0.5:
            return "가능"
        else:
            return "불확실"

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

    # ─────────────────────────────────────────────────────────────
    # Database Helpers
    # ─────────────────────────────────────────────────────────────

    def get_available_databases(self) -> list[DatabaseID]:
        """Get list of available databases for searching.

        Returns:
            List of database IDs
        """
        return ["KIPRIS", "USPTO", "Google_Patents"]

    def format_database_results_summary(
        self,
        results_count: dict[DatabaseID, int],
        errors: dict[DatabaseID, str | None],
    ) -> str:
        """Format a summary of database search results.

        Args:
            results_count: Count of results per database
            errors: Error messages per database

        Returns:
            Formatted summary string
        """
        lines = ["## 검색 결과 요약\n"]

        for db in self.get_available_databases():
            count = results_count.get(db, 0)
            error = errors.get(db)

            if error:
                lines.append(f"- **{db}**: 오류 - {error}")
            else:
                lines.append(f"- **{db}**: {count}건")

        total = sum(results_count.values())
        lines.append(f"\n**총 검색 결과: {total}건**")

        return "\n".join(lines)
