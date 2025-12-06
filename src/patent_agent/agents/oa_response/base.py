"""Base agent class for OA Response agents.

Implements Zero Hallucination laws (LAW-6 to LAW-10) from PALLAS-EVIDENCE v3.0.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from patent_agent.config import get_settings
from patent_agent.rules.oa_response_laws import OAResponseLaws
from patent_agent.state.base import ConfidenceLevel, Evidence
from patent_agent.state.oa_response import OAResponseState

logger = structlog.get_logger(__name__)


class BaseOAResponseAgent(ABC):
    """Base class for OA Response agents.

    All agents in the OA response workflow inherit from this class.
    It enforces Zero Hallucination laws and provides:
    - LLM integration with PALLAS-EVIDENCE guidelines
    - Evidence chain tracking
    - Confidence level management
    - Output validation against hallucination laws
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        guidelines: OAResponseLaws | None = None,
    ):
        """Initialize the agent.

        Args:
            llm: Language model to use. If not provided, creates default from settings.
            guidelines: Guidelines to follow. If not provided, uses default OAResponseLaws.
        """
        self._llm = llm
        self._guidelines = guidelines or OAResponseLaws()
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
    def phase(self) -> str:
        """Phase this agent handles (P1-P5)."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        ...

    @abstractmethod
    async def process(self, state: OAResponseState) -> dict[str, Any]:
        """Process the current state and return updates.

        Args:
            state: Current workflow state

        Returns:
            Dictionary of state updates
        """
        ...

    def _get_full_system_prompt(self) -> str:
        """Get complete system prompt including Zero Hallucination guidelines."""
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
            system_prompt: Optional custom system prompt (uses default if not provided)

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

    def _validate_output(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate output against Zero Hallucination laws.

        Args:
            output: Generated output to validate
            context: Optional context for validation

        Returns:
            Tuple of (is_valid, list of violation messages)
        """
        result = self._guidelines.validate_output(output, context)
        violations = [v.description for v in result.violations]
        return result.is_valid, violations

    # ─────────────────────────────────────────────────────────────
    # Evidence Chain Management (LAW-7)
    # ─────────────────────────────────────────────────────────────

    def create_evidence(
        self,
        source: str,
        page: int | None,
        paragraph: int | None,
        verbatim_text: str,
        confidence: ConfidenceLevel = "확실",
    ) -> Evidence:
        """Create an evidence object with proper citation format.

        Args:
            source: Document source (출원서, D1, OA, etc.)
            page: Page number
            paragraph: Paragraph number
            verbatim_text: Exact text from document (LAW-6)
            confidence: Confidence level

        Returns:
            Evidence object
        """
        return {
            "source": source,
            "page": page,
            "paragraph": paragraph,
            "verbatim_text": verbatim_text,
            "confidence": confidence,
        }

    def format_citation(
        self,
        source: str,
        page: int | None = None,
        paragraph: int | None = None,
        claim_number: int | None = None,
    ) -> str:
        """Format a citation according to LAW-7.

        Args:
            source: Document source name
            page: Page number
            paragraph: Paragraph number
            claim_number: Claim number (alternative to page/para)

        Returns:
            Formatted citation string
        """
        if claim_number is not None:
            return f"[{source}:청구항{claim_number}]"
        elif page is not None and paragraph is not None:
            return f"[{source}:p.{page}:para.{paragraph}]"
        elif page is not None:
            return f"[{source}:p.{page}]"
        else:
            return f"[{source}]"

    def quote_verbatim(self, text: str) -> str:
        """Quote text verbatim according to LAW-6.

        Args:
            text: Text to quote

        Returns:
            Properly quoted text
        """
        return f'"{text}"'

    # ─────────────────────────────────────────────────────────────
    # Confidence Level Management (LAW-9)
    # ─────────────────────────────────────────────────────────────

    def get_confidence_indicator(self, level: ConfidenceLevel) -> str:
        """Get the emoji indicator for a confidence level.

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

    def format_with_confidence(
        self,
        text: str,
        confidence: ConfidenceLevel,
    ) -> str:
        """Format text with confidence indicator.

        Args:
            text: Text to format
            confidence: Confidence level

        Returns:
            Text with confidence indicator
        """
        indicator = self.get_confidence_indicator(confidence)
        return f"{indicator} {text}"

    # ─────────────────────────────────────────────────────────────
    # Verification (LAW-10)
    # ─────────────────────────────────────────────────────────────

    def verify_no_hallucination(
        self,
        output: str,
        source_documents: dict[str, str],
    ) -> tuple[bool, list[str]]:
        """Verify output contains no hallucinated content.

        Args:
            output: Generated output
            source_documents: Dict of document name -> content

        Returns:
            Tuple of (is_verified, list of issues)
        """
        issues = []

        # Check for prohibited expressions
        is_valid, violations = self._validate_output(output)
        if not is_valid:
            issues.extend(violations)

        # Verify quoted text exists in source
        import re

        quoted_texts = re.findall(r'"([^"]+)"', output)
        for quoted in quoted_texts:
            found = False
            for doc_name, doc_content in source_documents.items():
                if quoted in doc_content:
                    found = True
                    break
            if not found and len(quoted) > 20:  # Ignore short quotes
                issues.append(f"인용문을 원본 문서에서 찾을 수 없음: {quoted[:50]}...")

        return len(issues) == 0, issues
