"""Base agent class for Patent Translation (EN→KR) agents.

Provides common functionality for translation workflow agents including:
- Style guide compliance (영한 특허번역 스타일가이드)
- 4 Absolute Laws (LAW-T-1 through LAW-T-4)
- TAC (Title, Abstract, Claims) section priority
- Mistranslation prevention
- 5C quality validation
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from patent_agent.config import get_settings
from patent_agent.rules.translation_laws import TranslationLaws
from patent_agent.state.base import ConfidenceLevel
from patent_agent.state.translation import (
    RiskLevel,
    SectionType,
    TranslationPhase,
    TranslationState,
)

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# STYLE GUIDE RULES (영한 특허번역 스타일가이드 기반)
# ═══════════════════════════════════════════════════════════════════════════


class StyleGuideRules:
    """Style guide rules from 영한 특허번역 스타일가이드."""

    # 3대 핵심 원칙
    CORE_PRINCIPLES = {
        "faithfulness": "소스에 충실할 것 (Be faithful to the source)",
        "grammar": "한국어 문법에 부합할 것 (Conform to Korean grammar)",
        "convention": "한국 특허 명세서 작성 관행을 따를 것 (Follow Korean patent conventions)",
    }

    # TAC 섹션 (높은 심각도) - Title, Abstract, Claims
    TAC_SECTIONS: list[SectionType] = ["title", "abstract", "claims"]

    # 정확성 > 유창성
    ACCURACY_OVER_FLUENCY = True


class ClaimTranslationRules:
    """Claim-specific translation rules (청구항 번역 규칙)."""

    # 단일 명사구 구조 (마침표 끝에만)
    SINGLE_SENTENCE_RULE = True

    # 선행사 기반 (Antecedent Basis)
    ANTECEDENT_MAPPING = {
        "the": "상기",
        "said": "상기",
    }

    # 전이구 (Transitional Phrases) - 권리범위 결정
    TRANSITIONAL_PHRASES = {
        "comprising": ("포함하는", "open-ended"),
        "consisting of": ("~으로 이루어지는", "closed-ended"),
        "consisting essentially of": ("필수적으로 ~으로 이루어지는", "semi-closed"),
    }

    # 기능적 표현 (Functional Phrases)
    FUNCTIONAL_PHRASES = {
        "adapted to": "~하도록 구성되는",  # 스타일 가이드 권장
        "configured to": "~하도록 구성된",
        "means for": "~하기 위한 수단",
        "operatively connected": "작동 가능하게 연결된",
    }

    # 방법 청구항 표준 구조
    METHOD_CLAIM_STRUCTURE = "~방법으로서, ...단계를 포함하는, 방법."


class SpecTranslationRules:
    """Specification-specific translation rules (명세서 번역 규칙)."""

    # '상기' 사용 완전 금지
    SANGGI_FORBIDDEN = True
    SANGGI_ALTERNATIVES = ["해당", "본", "이러한", "상술한"]

    # 약어 표기 형식
    ABBREVIATION_FORMAT = "한국어 용어 (Full English Name, Acronym)"

    # 학명 표기 형식
    SCIENTIFIC_NAME_FORMAT = "한국어 음역 (*라틴어 학명*)"

    # 숫자 및 단위 규칙
    NUMBER_RULES = {
        "si_units": True,  # 국제단위계
        "space_between": True,  # 숫자와 단위 사이 공백 (100 mm)
        "degree_no_space": True,  # 각도 기호 앞 공백 없음 (45°)
        "ordinal_format": "제{N}",  # 서수: 제1, 제2
    }

    # 조사 사용 (은/는 vs 이/가)
    PARTICLE_USAGE = {
        "은/는": "문장 주제 (Topic) - '왕'",
        "이/가": "문법적 주어 (Subject) - '신하'",
    }


class PunctuationRules:
    """Punctuation rules (문장 부호 규칙)."""

    # 쉼표: 영어보다 훨씬 적게 사용
    COMMA_MINIMIZATION = True

    # 세미콜론 규칙
    SEMICOLON_ALLOWED = "내부 쉼표 포함하는 복잡한 목록"
    SEMICOLON_FORBIDDEN = "독립된 절 연결"


class MistranslationPrevention:
    """Mistranslation prevention mappings (주요 오역 방지).

    Format: source_term -> (wrong_translation, correct_translation, context_notes)
    """

    MAPPINGS = {
        # 수량/범위 관련
        "more than one": ("하나 이상", "둘 이상 또는 하나 초과", "문맥에 따라 명확히 구분"),
        "less than two": ("둘 이하", "하나 이하 또는 둘 미만", "문맥에 따라 명확히 구분"),
        # 분야별 용어
        "substrate": ("기판", "기재", "화학분야에서는 '기재', 전자/반도체는 '기판'"),
        "cathode": ("음극", "양극", "이차 전지 분야에서는 cathode가 양극"),
        "anode": ("양극", "음극", "이차 전지 분야에서는 anode가 음극"),
        "ground": ("지면", "접지", "전기/전자 분야에서는 '접지'"),
        # 기능적 표현
        "adapted to": ("~에 적합한", "~하도록 구성되는", "연방순회항소법원 해석 반영"),
        "generally cylindrical": ("일반적으로 원통형인", "전체적으로 원통형인", "물리적 형상 설명"),
        # 위치 관련
        "distal end": ("말단", "원위 단부", "distal 의미 반드시 포함"),
        "proximal end": ("말단", "근위 단부", "proximal 의미 반드시 포함"),
        "either end of": ("양 단부", "일단", "둘 중 어느 한쪽 끝"),
        # 특허 특유 표현
        "recite": ("암송하다", "기술하다", "특허 문맥에서 '기술하다'"),
        "a person skilled in the art": ("당업자", "통상의 기술자", "원문 의미에 충실"),
        "associated with": ("연관된다", "연결된다", "물리적 연결 관계일 때"),
        # 유체 관련
        "(fluid) communication": ("(유체) 통신", "(유체) 연통", "두 공간의 유체 연결 상태"),
        "intake for fluid": ("흡기구", "흡입구", "기체가 아닐 경우"),
        # 기타
        "one or more surfactants": ("하나 이상의 계면활성제", "1종 이상의 계면활성제", "물질 종류 수"),
        "post transition metal": ("후전이금속", "전이후금속", "'후전이금속'과 구분"),
        "SEQ ID NO:": ("서열 번호:", "서열번호 ", "콜론 없이 붙여 씀"),
        "detach": ("탈착하다", "탈리하다", "'탈착'은 부착+탈착 모두 의미 가능"),
        "incubation": ("배양", "정치", "특정 조건 유지 vs 세포 증식 구분"),
    }

    @classmethod
    def get_correct_translation(
        cls, source_term: str, tech_domain: str | None = None
    ) -> str | None:
        """Get the correct translation for a source term.

        Args:
            source_term: Source term in English
            tech_domain: Optional technology domain for context

        Returns:
            Correct Korean translation or None if not in mappings
        """
        mapping = cls.MAPPINGS.get(source_term.lower())
        if mapping:
            return mapping[1]  # Return correct translation
        return None

    @classmethod
    def check_for_mistranslation(cls, source_text: str, translated_text: str) -> list[dict]:
        """Check translated text for potential mistranslations.

        Args:
            source_text: Original English text
            translated_text: Translated Korean text

        Returns:
            List of potential mistranslation issues
        """
        issues = []
        source_lower = source_text.lower()

        for source_term, (wrong, correct, notes) in cls.MAPPINGS.items():
            if source_term in source_lower:
                if wrong in translated_text and correct not in translated_text:
                    issues.append({
                        "source_term": source_term,
                        "wrong_translation": wrong,
                        "correct_translation": correct,
                        "notes": notes,
                    })

        return issues


# ═══════════════════════════════════════════════════════════════════════════
# TRANSLATION GUIDELINES
# ═══════════════════════════════════════════════════════════════════════════


class TranslationGuidelines:
    """Complete guidelines for EN→KR patent translation."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the complete translation guidelines system prompt."""
        return """## 영→한 특허번역 가이드라인

### 1. 번역의 3대 핵심 원칙
- **소스에 충실할 것**: 원문의 의미, 구조, 법적 기능을 정확하게 전달
- **한국어 문법에 부합할 것**: 자연스럽고 문법적으로 올바른 한국어 문장
- **한국 특허 명세서 작성 관행을 따를 것**: 한국 특허 실무 용어와 문체 준수

### 2. TAC 섹션 우선순위
제목(Title), 초록(Abstract), 청구항(Claims)에서 발생한 오류는
명세서의 동일한 오류보다 심각도가 높게 책정됩니다.

### 3. 정확성 > 유창성
특허 번역에서는 유창성보다 **정확성이 절대적으로 우선**합니다.
문법적 유창성이나 문체적 세련미를 위해 기술적, 법적 정확성을 희생해서는 안 됩니다.

### 4. 청구항 번역 규칙

#### 4.1 기본 구조
- 각 청구항은 문법적으로 완전한 하나의 **명사구**로 번역
- 문장의 끝은 **마침표(.)로 종결**

#### 4.2 선행사 기반 (Antecedent Basis)
- 이전에 언급된 구성요소를 지칭하는 'the' 또는 'said'는 반드시 **'상기'**로 번역
- 예외: 소유격(of)에 의해 한정되는 명사구의 'the'는 '상기'로 번역하지 않음
  - "the structure of the compound" → "화합물의 구조" (상기 없음)

#### 4.3 전이구 (Transitional Phrases)
| 영문 | 한국어 | 권리범위 |
|------|--------|----------|
| comprising | 포함하는 | 개방형 (Open-ended) |
| consisting of | ~으로 이루어지는 | 폐쇄형 (Closed-ended) |
| consisting essentially of | 필수적으로 ~으로 이루어지는 | 반폐쇄형 (Semi-closed) |

#### 4.4 기능적 표현
- **adapted to** → '~하도록 구성되는' (권장)
- configured to → '~하도록 구성된'
- means for → '~하기 위한 수단'

#### 4.5 방법 청구항
표준 구조: `~방법으로서, ...단계를 포함하는, 방법.`

### 5. 명세서 번역 규칙

#### 5.1 '상기' 사용 금지
청구항 외 섹션(명세서, 요약서 등)에서는 **'상기' 사용 완전 금지**
대체어: '해당', '본', '이러한', '상술한'

#### 5.2 약어 표기
`한국어 용어 (Full English Name, Acronym)` 형식

#### 5.3 학명 표기
`한국어 음역 (*라틴어 학명*)` 형식 (라틴어만 이탤릭)

#### 5.4 숫자 및 단위
- 국제단위계(SI) 준수
- 숫자와 단위 사이 공백: `100 mm`
- 각도 기호 앞 공백 없음: `45°`
- 서수: `제1`, `제2`

### 6. 문장 부호 규칙
- **쉼표**: 영어보다 훨씬 적게 사용
- **세미콜론**: 내부 쉼표 포함 복잡한 목록에만 제한적 사용
  - 독립된 절 연결에는 **절대 사용 금지**

### 7. 주요 오역 방지
- more than one → 둘 이상 (하나 이상 아님)
- substrate → 기재 (화학) / 기판 (전자)
- cathode/anode → 양극/음극 (이차전지)
- adapted to → ~하도록 구성되는
- distal/proximal end → 원위/근위 단부
- recite → 기술하다 (특허 문맥)
"""


# ═══════════════════════════════════════════════════════════════════════════
# BASE TRANSLATION AGENT
# ═══════════════════════════════════════════════════════════════════════════


class BaseTranslationAgent(ABC):
    """Base class for Patent Translation (EN→KR) agents.

    All agents in the translation workflow inherit from this class.
    It provides:
    - LLM integration with translation guidelines
    - Style guide compliance checking
    - 4 Absolute Laws validation
    - TAC section priority handling
    - Mistranslation prevention
    """

    # Class-level rule references
    STYLE_GUIDE = StyleGuideRules
    CLAIM_RULES = ClaimTranslationRules
    SPEC_RULES = SpecTranslationRules
    PUNCTUATION_RULES = PunctuationRules
    MISTRANSLATION = MistranslationPrevention

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        laws: TranslationLaws | None = None,
    ):
        """Initialize the agent.

        Args:
            llm: Language model to use. If not provided, creates default from settings.
            laws: Translation laws to enforce. If not provided, uses default.
        """
        self._llm = llm
        self._laws = laws or TranslationLaws()
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
    def phase(self) -> TranslationPhase:
        """Phase this agent handles in the workflow."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        ...

    @abstractmethod
    async def process(self, state: TranslationState) -> dict[str, Any]:
        """Process the current state and return updates.

        Args:
            state: Current workflow state

        Returns:
            Dictionary of state updates
        """
        ...

    def _get_full_system_prompt(self) -> str:
        """Get complete system prompt including guidelines and laws."""
        base_prompt = self.get_system_prompt()
        guideline_prompt = TranslationGuidelines.get_system_prompt()
        law_prompt = self._laws.get_system_prompt()

        return f"""{base_prompt}

---

{guideline_prompt}

---

{law_prompt}"""

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
    # TAC Section Priority
    # ─────────────────────────────────────────────────────────────

    def is_tac_section(self, section: SectionType) -> bool:
        """Check if section is a TAC (Title, Abstract, Claims) section.

        TAC sections have higher severity for errors.

        Args:
            section: Section type to check

        Returns:
            True if TAC section
        """
        return section in self.STYLE_GUIDE.TAC_SECTIONS

    def get_section_severity_multiplier(self, section: SectionType) -> float:
        """Get severity multiplier for a section.

        TAC sections have higher severity (1.5x).

        Args:
            section: Section type

        Returns:
            Severity multiplier
        """
        return 1.5 if self.is_tac_section(section) else 1.0

    # ─────────────────────────────────────────────────────────────
    # Translation Validation
    # ─────────────────────────────────────────────────────────────

    def validate_translation(
        self,
        source_text: str,
        translated_text: str,
        section: SectionType,
    ) -> dict[str, Any]:
        """Validate a translation against laws and style guide.

        Args:
            source_text: Original English text
            translated_text: Translated Korean text
            section: Section type being translated

        Returns:
            Validation result with violations and warnings
        """
        # Run law validation
        law_result = self._laws.validate_output(
            translated_text,
            context={
                "section_type": section,
                "source_text": source_text,
            },
        )

        # Check for mistranslations
        mistranslations = self.MISTRANSLATION.check_for_mistranslation(
            source_text, translated_text
        )

        # Calculate severity with TAC multiplier
        severity_multiplier = self.get_section_severity_multiplier(section)

        return {
            "is_valid": law_result.is_valid and not mistranslations,
            "violations": law_result.violations,
            "warnings": law_result.warnings,
            "auto_corrections": law_result.auto_corrections,
            "mistranslations": mistranslations,
            "severity_multiplier": severity_multiplier,
            "section": section,
        }

    def get_sanggi_alternative(self) -> str:
        """Get a random alternative to '상기' for non-claim sections.

        Returns:
            Alternative word to use instead of '상기'
        """
        import random

        return random.choice(self.SPEC_RULES.SANGGI_ALTERNATIVES)

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
    # Risk Flag Helpers
    # ─────────────────────────────────────────────────────────────

    def create_risk_flag(
        self,
        level: RiskLevel,
        issue: str,
        location: str,
        recommendation: str,
        requires_human_review: bool = False,
    ) -> dict[str, Any]:
        """Create a risk flag dictionary.

        Args:
            level: Risk level (Critical/High/Medium/Low)
            issue: Description of the issue
            location: Location in document
            recommendation: Recommended action
            requires_human_review: Whether human review is needed

        Returns:
            Risk flag dictionary
        """
        # Critical and High always require human review
        if level in ("Critical", "High"):
            requires_human_review = True

        return {
            "level": level,
            "issue": issue,
            "location": location,
            "recommendation": recommendation,
            "requires_human_review": requires_human_review,
        }

    # ─────────────────────────────────────────────────────────────
    # Reasoning Log Helpers
    # ─────────────────────────────────────────────────────────────

    def create_reasoning_entry(
        self,
        checkpoint: str,
        decision: str,
        reasoning: str,
        applied_laws: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a reasoning log entry.

        Args:
            checkpoint: Checkpoint identifier
            decision: Decision made
            reasoning: Reasoning for the decision
            applied_laws: List of laws applied (e.g., ["LAW-T-1", "LAW-T-3"])

        Returns:
            Reasoning entry dictionary
        """
        from datetime import datetime

        return {
            "phase": self.phase,
            "checkpoint": checkpoint,
            "decision": decision,
            "reasoning": reasoning,
            "applied_laws": applied_laws or [],
            "timestamp": datetime.now().isoformat(),
        }
