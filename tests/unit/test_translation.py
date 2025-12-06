"""Tests for Patent Translation (EN→KR) workflow.

Tests the complete translation workflow including:
- State definitions
- 4 Absolute Laws (LAW-T-1 through LAW-T-4)
- Style guide rules
- All 6 phase agents
- Translation graph
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from patent_agent.agents.translation import (
    BaseTranslationAgent,
    ClaimTranslationRules,
    DocumentIntakeAgent,
    GlossaryBuilderAgent,
    MistranslationPrevention,
    OutputGeneratorAgent,
    PunctuationRules,
    QualityVerifierAgent,
    SectionTranslatorAgent,
    SpecTranslationRules,
    StructureAnalyzerAgent,
    StyleGuideRules,
    TranslationGuidelines,
)
from patent_agent.state.translation import (
    ABSOLUTE_LAWS,
    FIVE_C_CRITERIA,
    DocumentStructure,
    GlossaryEntry,
    QualityReport,
    RiskFlag,
    SectionType,
    SourceDocument,
    TranslatedSection,
    TranslationState,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_source_document() -> SourceDocument:
    """Sample source document for testing."""
    return SourceDocument(
        file_path="/path/to/patent.pdf",
        file_type="pdf",
        total_pages=10,
        requires_ocr=False,
        patent_number="US10123456B2",
        filing_date="2020-01-15",
    )


@pytest.fixture
def sample_document_structure() -> DocumentStructure:
    """Sample document structure for testing."""
    return DocumentStructure(
        sections={
            "title": "System for Natural Language Processing",
            "abstract": "A system comprising a transformer encoder...",
            "claims": "1. A system comprising:\na transformer encoder;\na decoder;",
            "detailed_description": "The present invention relates to...",
        },
        claims_count=2,
        figures_count=3,
        reference_numerals={"10": "transformer encoder", "20": "decoder"},
        claim_dependencies={1: None, 2: 1},
    )


@pytest.fixture
def sample_glossary() -> list[GlossaryEntry]:
    """Sample glossary entries for testing."""
    return [
        GlossaryEntry(
            english_term="transformer encoder",
            korean_term="트랜스포머 인코더",
            source="auto",
            domain="AI",
            notes=None,
        ),
        GlossaryEntry(
            english_term="comprising",
            korean_term="포함하는",
            source="auto",
            domain=None,
            notes="Open-ended transitional phrase",
        ),
        GlossaryEntry(
            english_term="decoder",
            korean_term="디코더",
            source="auto",
            domain="AI",
            notes=None,
        ),
    ]


@pytest.fixture
def sample_translated_section() -> TranslatedSection:
    """Sample translated section for testing."""
    return TranslatedSection(
        section_type="claims",
        source_text="1. A system comprising:\na transformer encoder (10);\na decoder (20).",
        translated_text="1. 다음을 포함하는 시스템:\n트랜스포머 인코더 (10);\n디코더 (20).",
        glossary_terms_used=["트랜스포머 인코더", "디코더"],
        risk_flags=[],
        law_compliance={"LAW-T-1": True, "LAW-T-2": True, "LAW-T-3": True, "LAW-T-4": True},
    )


@pytest.fixture
def sample_translation_state(
    sample_source_document: SourceDocument,
    sample_document_structure: DocumentStructure,
    sample_glossary: list[GlossaryEntry],
) -> TranslationState:
    """Sample translation state for testing."""
    return TranslationState(
        messages=[],
        current_workflow="translation",
        current_step="P1",
        iteration_count=0,
        max_iterations=3,
        session_id="test-session-123",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        source_document=sample_source_document,
        document_structure=sample_document_structure,
        glossary=sample_glossary,
        glossary_finalized=True,
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM for testing."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
    return mock


# ─────────────────────────────────────────────────────────────
# Style Guide Rules Tests
# ─────────────────────────────────────────────────────────────


class TestStyleGuideRules:
    """Tests for StyleGuideRules."""

    def test_core_principles(self) -> None:
        """Test core translation principles are defined."""
        principles = StyleGuideRules.CORE_PRINCIPLES
        assert "faithfulness" in principles
        assert "grammar" in principles
        assert "convention" in principles

    def test_tac_sections(self) -> None:
        """Test TAC sections are correctly defined."""
        tac = StyleGuideRules.TAC_SECTIONS
        assert "title" in tac
        assert "abstract" in tac
        assert "claims" in tac
        assert "detailed_description" not in tac

    def test_accuracy_over_fluency(self) -> None:
        """Test accuracy over fluency principle."""
        assert StyleGuideRules.ACCURACY_OVER_FLUENCY is True


class TestClaimTranslationRules:
    """Tests for ClaimTranslationRules."""

    def test_single_sentence_rule(self) -> None:
        """Test single sentence rule is enabled."""
        assert ClaimTranslationRules.SINGLE_SENTENCE_RULE is True

    def test_antecedent_mapping(self) -> None:
        """Test antecedent basis mapping."""
        mapping = ClaimTranslationRules.ANTECEDENT_MAPPING
        assert mapping["the"] == "상기"
        assert mapping["said"] == "상기"

    def test_transitional_phrases(self) -> None:
        """Test transitional phrase mappings."""
        phrases = ClaimTranslationRules.TRANSITIONAL_PHRASES

        # Open-ended
        korean, scope = phrases["comprising"]
        assert korean == "포함하는"
        assert scope == "open-ended"

        # Closed-ended
        korean, scope = phrases["consisting of"]
        assert "이루어지는" in korean
        assert scope == "closed-ended"

    def test_functional_phrases(self) -> None:
        """Test functional phrase mappings."""
        phrases = ClaimTranslationRules.FUNCTIONAL_PHRASES
        assert "구성되는" in phrases["adapted to"]
        assert "구성된" in phrases["configured to"]
        assert "수단" in phrases["means for"]


class TestSpecTranslationRules:
    """Tests for SpecTranslationRules."""

    def test_sanggi_forbidden(self) -> None:
        """Test '상기' is forbidden in spec."""
        assert SpecTranslationRules.SANGGI_FORBIDDEN is True

    def test_sanggi_alternatives(self) -> None:
        """Test alternatives to '상기' are provided."""
        alternatives = SpecTranslationRules.SANGGI_ALTERNATIVES
        assert "해당" in alternatives
        assert "본" in alternatives
        assert "이러한" in alternatives
        assert "상술한" in alternatives

    def test_number_rules(self) -> None:
        """Test number and unit rules."""
        rules = SpecTranslationRules.NUMBER_RULES
        assert rules["si_units"] is True
        assert rules["space_between"] is True
        assert rules["ordinal_format"] == "제{N}"


class TestPunctuationRules:
    """Tests for PunctuationRules."""

    def test_comma_minimization(self) -> None:
        """Test comma minimization rule."""
        assert PunctuationRules.COMMA_MINIMIZATION is True

    def test_semicolon_rules(self) -> None:
        """Test semicolon usage rules."""
        assert "목록" in PunctuationRules.SEMICOLON_ALLOWED
        assert "절 연결" in PunctuationRules.SEMICOLON_FORBIDDEN


# ─────────────────────────────────────────────────────────────
# Mistranslation Prevention Tests
# ─────────────────────────────────────────────────────────────


class TestMistranslationPrevention:
    """Tests for MistranslationPrevention."""

    def test_mappings_exist(self) -> None:
        """Test mistranslation mappings are defined."""
        mappings = MistranslationPrevention.MAPPINGS
        assert len(mappings) > 10  # Should have many entries

    def test_more_than_one_mapping(self) -> None:
        """Test 'more than one' mistranslation mapping."""
        wrong, correct, notes = MistranslationPrevention.MAPPINGS["more than one"]
        assert wrong == "하나 이상"
        assert "둘 이상" in correct

    def test_substrate_mapping(self) -> None:
        """Test 'substrate' domain-specific mapping."""
        wrong, correct, notes = MistranslationPrevention.MAPPINGS["substrate"]
        assert "기재" in correct
        assert "화학" in notes

    def test_cathode_anode_battery_mapping(self) -> None:
        """Test battery domain cathode/anode mapping."""
        # cathode in battery = 양극
        _, correct, notes = MistranslationPrevention.MAPPINGS["cathode"]
        assert "양극" in correct
        assert "전지" in notes or "이차" in notes

        # anode in battery = 음극
        _, correct, notes = MistranslationPrevention.MAPPINGS["anode"]
        assert "음극" in correct

    def test_adapted_to_mapping(self) -> None:
        """Test 'adapted to' functional phrase mapping."""
        wrong, correct, notes = MistranslationPrevention.MAPPINGS["adapted to"]
        assert "구성되는" in correct

    def test_get_correct_translation(self) -> None:
        """Test get_correct_translation method."""
        result = MistranslationPrevention.get_correct_translation("cathode")
        assert result == "양극"

        result = MistranslationPrevention.get_correct_translation("unknown_term")
        assert result is None

    def test_check_for_mistranslation(self) -> None:
        """Test mistranslation detection."""
        # Test case: adapted to -> wrong translation (exact match required)
        source = "The device is adapted to receive signals."
        translated_wrong = "장치는 신호를 수신하는데 ~에 적합한 구조이다."  # Contains wrong phrase

        issues = MistranslationPrevention.check_for_mistranslation(
            source, translated_wrong
        )

        # Should detect that "adapted to" was potentially mistranslated
        assert len(issues) > 0
        adapted_issue = next(
            (i for i in issues if i["source_term"] == "adapted to"), None
        )
        assert adapted_issue is not None
        assert "구성되는" in adapted_issue["correct_translation"]


# ─────────────────────────────────────────────────────────────
# Translation Guidelines Tests
# ─────────────────────────────────────────────────────────────


class TestTranslationGuidelines:
    """Tests for TranslationGuidelines."""

    def test_system_prompt_content(self) -> None:
        """Test system prompt contains key sections."""
        prompt = TranslationGuidelines.get_system_prompt()

        # Core principles
        assert "3대 핵심 원칙" in prompt
        assert "소스에 충실" in prompt

        # TAC priority
        assert "TAC 섹션" in prompt

        # Accuracy over fluency
        assert "정확성" in prompt and "유창성" in prompt

        # Claim rules
        assert "청구항" in prompt
        assert "상기" in prompt

        # Transitional phrases
        assert "comprising" in prompt
        assert "consisting of" in prompt

    def test_mistranslation_examples(self) -> None:
        """Test system prompt includes mistranslation examples."""
        prompt = TranslationGuidelines.get_system_prompt()

        assert "substrate" in prompt
        assert "cathode" in prompt or "anode" in prompt


# ─────────────────────────────────────────────────────────────
# Base Translation Agent Tests
# ─────────────────────────────────────────────────────────────


class TestBaseTranslationAgent:
    """Tests for BaseTranslationAgent utility methods."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> DocumentIntakeAgent:
        """Create agent instance for testing."""
        return DocumentIntakeAgent(llm=mock_llm)

    def test_is_tac_section(self, agent: DocumentIntakeAgent) -> None:
        """Test TAC section detection."""
        assert agent.is_tac_section("title") is True
        assert agent.is_tac_section("abstract") is True
        assert agent.is_tac_section("claims") is True
        assert agent.is_tac_section("detailed_description") is False
        assert agent.is_tac_section("background") is False

    def test_get_section_severity_multiplier(
        self, agent: DocumentIntakeAgent
    ) -> None:
        """Test severity multiplier for TAC sections."""
        # TAC sections have 1.5x severity
        assert agent.get_section_severity_multiplier("claims") == 1.5
        assert agent.get_section_severity_multiplier("title") == 1.5
        assert agent.get_section_severity_multiplier("abstract") == 1.5

        # Non-TAC sections have 1.0x
        assert agent.get_section_severity_multiplier("detailed_description") == 1.0
        assert agent.get_section_severity_multiplier("background") == 1.0

    def test_get_sanggi_alternative(self, agent: DocumentIntakeAgent) -> None:
        """Test getting alternatives to '상기'."""
        alternatives = SpecTranslationRules.SANGGI_ALTERNATIVES
        result = agent.get_sanggi_alternative()
        assert result in alternatives

    def test_score_to_confidence(self, agent: DocumentIntakeAgent) -> None:
        """Test confidence level mapping."""
        assert agent.score_to_confidence(0.95) == "확실"
        assert agent.score_to_confidence(0.90) == "확실"
        assert agent.score_to_confidence(0.75) == "가능"
        assert agent.score_to_confidence(0.60) == "가능"
        assert agent.score_to_confidence(0.50) == "불확실"

    def test_get_confidence_indicator(self, agent: DocumentIntakeAgent) -> None:
        """Test confidence indicator emojis."""
        assert agent.get_confidence_indicator("확실") == "🟢"
        assert agent.get_confidence_indicator("가능") == "🟡"
        assert agent.get_confidence_indicator("불확실") == "🔴"

    def test_create_risk_flag(self, agent: DocumentIntakeAgent) -> None:
        """Test risk flag creation."""
        flag = agent.create_risk_flag(
            level="Critical",
            issue="Test issue",
            location="claims",
            recommendation="Fix it",
        )

        assert flag["level"] == "Critical"
        assert flag["issue"] == "Test issue"
        assert flag["requires_human_review"] is True  # Critical always requires review

        # High also requires human review
        flag = agent.create_risk_flag(
            level="High",
            issue="Test",
            location="test",
            recommendation="Fix",
        )
        assert flag["requires_human_review"] is True

        # Medium doesn't require review by default
        flag = agent.create_risk_flag(
            level="Medium",
            issue="Test",
            location="test",
            recommendation="Fix",
            requires_human_review=False,
        )
        assert flag["requires_human_review"] is False

    def test_create_reasoning_entry(self, agent: DocumentIntakeAgent) -> None:
        """Test reasoning entry creation."""
        entry = agent.create_reasoning_entry(
            checkpoint="CP1.1",
            decision="Test decision",
            reasoning="Test reasoning",
            applied_laws=["LAW-T-1", "LAW-T-2"],
        )

        assert entry["phase"] == "P1"
        assert entry["checkpoint"] == "CP1.1"
        assert entry["decision"] == "Test decision"
        assert "LAW-T-1" in entry["applied_laws"]
        assert "timestamp" in entry


# ─────────────────────────────────────────────────────────────
# DocumentIntakeAgent Tests
# ─────────────────────────────────────────────────────────────


class TestDocumentIntakeAgent:
    """Tests for DocumentIntakeAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> DocumentIntakeAgent:
        return DocumentIntakeAgent(llm=mock_llm)

    def test_agent_properties(self, agent: DocumentIntakeAgent) -> None:
        """Test agent property values."""
        assert agent.name == "DocumentIntakeAgent"
        assert agent.phase == "P1"
        assert "문서" in agent.description or "접수" in agent.description

    def test_get_system_prompt(self, agent: DocumentIntakeAgent) -> None:
        """Test system prompt content."""
        prompt = agent.get_system_prompt()
        assert "P1" in prompt
        assert "문서" in prompt

    @pytest.mark.asyncio
    async def test_process_missing_file(self, agent: DocumentIntakeAgent) -> None:
        """Test error handling when file is missing."""
        state = TranslationState(
            current_workflow="translation",
            source_file_path="",
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True
        assert len(result.get("error_messages", [])) > 0


# ─────────────────────────────────────────────────────────────
# StructureAnalyzerAgent Tests
# ─────────────────────────────────────────────────────────────


class TestStructureAnalyzerAgent:
    """Tests for StructureAnalyzerAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> StructureAnalyzerAgent:
        return StructureAnalyzerAgent(llm=mock_llm)

    def test_agent_properties(self, agent: StructureAnalyzerAgent) -> None:
        """Test agent property values."""
        assert agent.name == "StructureAnalyzerAgent"
        assert agent.phase == "P2"

    @pytest.mark.asyncio
    async def test_process_missing_source_text(
        self, agent: StructureAnalyzerAgent
    ) -> None:
        """Test error handling when source text is missing."""
        state = TranslationState(
            current_workflow="translation",
            source_text_content="",
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True


# ─────────────────────────────────────────────────────────────
# GlossaryBuilderAgent Tests
# ─────────────────────────────────────────────────────────────


class TestGlossaryBuilderAgent:
    """Tests for GlossaryBuilderAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> GlossaryBuilderAgent:
        return GlossaryBuilderAgent(llm=mock_llm)

    def test_agent_properties(self, agent: GlossaryBuilderAgent) -> None:
        """Test agent property values."""
        assert agent.name == "GlossaryBuilderAgent"
        assert agent.phase == "P3"

    @pytest.mark.asyncio
    async def test_process_missing_structure(
        self, agent: GlossaryBuilderAgent
    ) -> None:
        """Test error handling when document structure is missing."""
        state = TranslationState(
            current_workflow="translation",
            document_structure=None,
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True

    @pytest.mark.asyncio
    async def test_resolve_conflict(
        self,
        agent: GlossaryBuilderAgent,
        sample_translation_state: TranslationState,
    ) -> None:
        """Test conflict resolution."""
        # Add a conflict
        sample_translation_state["glossary_conflicts"] = [
            {
                "term": "test",
                "options": ["옵션1", "옵션2"],
            }
        ]

        result = await agent.resolve_conflict(
            sample_translation_state, "test", "옵션1"
        )

        assert len(result.get("glossary_conflicts", [])) == 0


# ─────────────────────────────────────────────────────────────
# SectionTranslatorAgent Tests
# ─────────────────────────────────────────────────────────────


class TestSectionTranslatorAgent:
    """Tests for SectionTranslatorAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> SectionTranslatorAgent:
        return SectionTranslatorAgent(llm=mock_llm)

    def test_agent_properties(self, agent: SectionTranslatorAgent) -> None:
        """Test agent property values."""
        assert agent.name == "SectionTranslatorAgent"
        assert agent.phase == "P4"
        assert "핵심" in agent.description or "번역" in agent.description

    def test_get_system_prompt(self, agent: SectionTranslatorAgent) -> None:
        """Test system prompt contains 4 absolute laws."""
        prompt = agent.get_system_prompt()

        # All 4 laws should be mentioned
        assert "LAW-T-1" in prompt
        assert "LAW-T-2" in prompt
        assert "LAW-T-3" in prompt
        assert "LAW-T-4" in prompt

        # Key rules
        assert "상기" in prompt
        assert "comprising" in prompt
        assert "consisting of" in prompt
        assert "괄호" in prompt

    @pytest.mark.asyncio
    async def test_process_missing_glossary(
        self, agent: SectionTranslatorAgent
    ) -> None:
        """Test error handling when glossary is not finalized."""
        state = TranslationState(
            current_workflow="translation",
            glossary_finalized=False,
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True
        assert "P3" in str(result.get("current_step"))

    def test_check_law_t1_compliance_claims(
        self, agent: SectionTranslatorAgent
    ) -> None:
        """Test LAW-T-1 compliance for claims."""
        # Claims should allow 상기
        text_with_sanggi = "상기 인코더에 연결된 디코더를 포함하는"
        assert agent._check_law_t1_compliance(text_with_sanggi, is_claims=True) is True

    def test_check_law_t1_compliance_spec(
        self, agent: SectionTranslatorAgent
    ) -> None:
        """Test LAW-T-1 compliance for spec (no 상기)."""
        # Spec should NOT have 상기
        text_with_sanggi = "상기 인코더는 데이터를 처리한다."
        assert agent._check_law_t1_compliance(text_with_sanggi, is_claims=False) is False

        text_without_sanggi = "해당 인코더는 데이터를 처리한다."
        assert agent._check_law_t1_compliance(text_without_sanggi, is_claims=False) is True

    def test_check_law_t2_compliance(
        self, agent: SectionTranslatorAgent
    ) -> None:
        """Test LAW-T-2 compliance (single sentence claims)."""
        # Good: proper claim format with multiple claims
        good_claims = """
1. 인코더를 포함하는 시스템.
2. 제1항에 있어서, 디코더를 더 포함하는 시스템."""
        assert agent._check_law_t2_compliance(good_claims, is_claims=True) is True

        # Bad: multiple periods in a single claim
        bad_claims = """
1. 인코더를 포함한다. 디코더도 포함한다."""
        assert agent._check_law_t2_compliance(bad_claims, is_claims=True) is False

        # Non-claims always pass
        assert agent._check_law_t2_compliance(bad_claims, is_claims=False) is True

    def test_check_law_t3_compliance(
        self, agent: SectionTranslatorAgent
    ) -> None:
        """Test LAW-T-3 compliance (transitional phrases)."""
        # Correct: comprising -> 포함하는
        source = "A system comprising an encoder"
        translated_good = "인코더를 포함하는 시스템"
        assert agent._check_law_t3_compliance(source, translated_good) is True

        # Wrong: comprising -> 구성된
        translated_bad = "인코더로 구성된 시스템"
        assert agent._check_law_t3_compliance(source, translated_bad) is False

    def test_check_law_t4_compliance(
        self, agent: SectionTranslatorAgent
    ) -> None:
        """Test LAW-T-4 compliance (reference numeral brackets)."""
        ref_numerals = {"100": "encoder", "200": "decoder"}

        # Good: all numerals have brackets
        good_text = "인코더 (100)와 디코더 (200)를 포함한다."
        assert agent._check_law_t4_compliance(good_text, ref_numerals) is True

        # Bad: numeral without brackets (standalone number)
        bad_text = "The encoder 100 and decoder (200) are included."
        assert agent._check_law_t4_compliance(bad_text, ref_numerals) is False


# ─────────────────────────────────────────────────────────────
# QualityVerifierAgent Tests
# ─────────────────────────────────────────────────────────────


class TestQualityVerifierAgent:
    """Tests for QualityVerifierAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> QualityVerifierAgent:
        return QualityVerifierAgent(llm=mock_llm)

    def test_agent_properties(self, agent: QualityVerifierAgent) -> None:
        """Test agent property values."""
        assert agent.name == "QualityVerifierAgent"
        assert agent.phase == "P5"

    def test_get_system_prompt_5c(self, agent: QualityVerifierAgent) -> None:
        """Test system prompt contains 5C criteria."""
        prompt = agent.get_system_prompt()

        assert "Correctness" in prompt or "정확성" in prompt
        assert "Clarity" in prompt or "명확성" in prompt
        assert "Conciseness" in prompt or "간결성" in prompt
        assert "Consistency" in prompt or "일관성" in prompt
        assert "Compliance" in prompt or "준수" in prompt

    @pytest.mark.asyncio
    async def test_process_missing_translations(
        self, agent: QualityVerifierAgent
    ) -> None:
        """Test error handling when translations are missing."""
        state = TranslationState(
            current_workflow="translation",
            translated_sections={},
        )

        result = await agent.process(state)

        assert result.get("is_error_state") is True


# ─────────────────────────────────────────────────────────────
# OutputGeneratorAgent Tests
# ─────────────────────────────────────────────────────────────


class TestOutputGeneratorAgent:
    """Tests for OutputGeneratorAgent."""

    @pytest.fixture
    def agent(self, mock_llm: MagicMock) -> OutputGeneratorAgent:
        return OutputGeneratorAgent(llm=mock_llm)

    def test_agent_properties(self, agent: OutputGeneratorAgent) -> None:
        """Test agent property values."""
        assert agent.name == "OutputGeneratorAgent"
        assert agent.phase == "P6"

    def test_get_system_prompt_outputs(self, agent: OutputGeneratorAgent) -> None:
        """Test system prompt lists all output files."""
        prompt = agent.get_system_prompt()

        assert "ko_final" in prompt
        assert "en_ko_compare" in prompt
        assert "qa_report" in prompt
        assert "reasoning_log" in prompt
        assert "glossary" in prompt


# ─────────────────────────────────────────────────────────────
# State Definition Tests
# ─────────────────────────────────────────────────────────────


class TestTranslationState:
    """Tests for TranslationState."""

    def test_state_creation(
        self, sample_translation_state: TranslationState
    ) -> None:
        """Test state creation with required fields."""
        assert sample_translation_state.get("current_workflow") == "translation"
        assert sample_translation_state.get("glossary_finalized") is True

    def test_source_document_structure(
        self, sample_source_document: SourceDocument
    ) -> None:
        """Test SourceDocument TypedDict structure."""
        assert sample_source_document["file_type"] in ["pdf", "docx", "txt"]
        assert sample_source_document["total_pages"] > 0

    def test_document_structure_structure(
        self, sample_document_structure: DocumentStructure
    ) -> None:
        """Test DocumentStructure TypedDict structure."""
        assert "title" in sample_document_structure["sections"]
        assert "claims" in sample_document_structure["sections"]
        assert sample_document_structure["claims_count"] > 0

    def test_glossary_entry_structure(
        self, sample_glossary: list[GlossaryEntry]
    ) -> None:
        """Test GlossaryEntry TypedDict structure."""
        entry = sample_glossary[0]
        assert entry["source"] in ["KIPRIS", "WIPO_Pearl", "user", "auto"]

    def test_translated_section_structure(
        self, sample_translated_section: TranslatedSection
    ) -> None:
        """Test TranslatedSection TypedDict structure."""
        assert sample_translated_section["section_type"] == "claims"
        assert len(sample_translated_section["law_compliance"]) == 4


class TestAbsoluteLaws:
    """Tests for ABSOLUTE_LAWS constant."""

    def test_all_laws_defined(self) -> None:
        """Test all 4 absolute laws are defined."""
        assert "LAW-1" in ABSOLUTE_LAWS
        assert "LAW-2" in ABSOLUTE_LAWS
        assert "LAW-3" in ABSOLUTE_LAWS
        assert "LAW-4" in ABSOLUTE_LAWS

    def test_law_1_sanggi_rules(self) -> None:
        """Test LAW-1 has correct sanggi rules."""
        law1 = ABSOLUTE_LAWS["LAW-1"]
        assert "상기" in law1["name"]
        assert "상기" in law1["claims_rule"]
        assert "금지" in law1["non_claims_rule"]

    def test_law_3_transitional_phrases(self) -> None:
        """Test LAW-3 has transitional phrase rules."""
        law3 = ABSOLUTE_LAWS["LAW-3"]
        assert "comprising" in law3
        assert "consisting_of" in law3


class TestFiveCCriteria:
    """Tests for FIVE_C_CRITERIA constant."""

    def test_all_criteria_defined(self) -> None:
        """Test all 5C criteria are defined."""
        assert "Correctness" in FIVE_C_CRITERIA
        assert "Clarity" in FIVE_C_CRITERIA
        assert "Conciseness" in FIVE_C_CRITERIA
        assert "Consistency" in FIVE_C_CRITERIA
        assert "Compliance" in FIVE_C_CRITERIA


# ─────────────────────────────────────────────────────────────
# Graph Tests
# ─────────────────────────────────────────────────────────────


class TestTranslationGraph:
    """Tests for TranslationGraph."""

    def test_graph_creation(self) -> None:
        """Test graph can be created."""
        from patent_agent.graphs.translation_graph import (
            TranslationGraph,
            create_translation_graph,
        )

        graph = create_translation_graph()
        assert graph is not None
        assert graph.name == "translation"

    def test_graph_state_class(self) -> None:
        """Test graph uses correct state class."""
        from patent_agent.graphs.translation_graph import TranslationGraph

        graph = TranslationGraph()
        assert graph.state_class == TranslationState

    def test_graph_initial_step(self) -> None:
        """Test graph starts at P1."""
        from patent_agent.graphs.translation_graph import TranslationGraph

        graph = TranslationGraph()
        assert graph.get_initial_step() == "P1"

    def test_graph_builds_successfully(self) -> None:
        """Test graph compiles without errors."""
        from patent_agent.graphs.translation_graph import TranslationGraph

        graph = TranslationGraph()
        compiled = graph.graph  # Triggers lazy build

        assert compiled is not None
