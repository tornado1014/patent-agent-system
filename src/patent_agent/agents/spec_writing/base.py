"""Base agent class for specification writing agents."""

from abc import ABC, abstractmethod
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from patent_agent.config import get_settings
from patent_agent.rules.spec_writing_laws import SpecWritingLaws
from patent_agent.state.spec_writing import SpecWritingState

logger = structlog.get_logger(__name__)


class BaseSpecWritingAgent(ABC):
    """Base class for specification writing agents.

    All agents in the specification writing workflow inherit from this class.
    It provides:
    - LLM integration
    - Guideline injection
    - Common utility methods
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        guidelines: SpecWritingLaws | None = None,
    ):
        """Initialize the agent.

        Args:
            llm: Language model to use. If not provided, creates default from settings.
            guidelines: Guidelines to follow. If not provided, uses default SpecWritingLaws.
        """
        self._llm = llm
        self._guidelines = guidelines or SpecWritingLaws()
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

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        ...

    @abstractmethod
    async def process(self, state: SpecWritingState) -> dict[str, Any]:
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
        """Validate output against guidelines.

        Args:
            output: Generated output to validate
            context: Optional context for validation

        Returns:
            Tuple of (is_valid, list of violation messages)
        """
        result = self._guidelines.validate_output(output, context)
        violations = [v.description for v in result.violations]
        return result.is_valid, violations
