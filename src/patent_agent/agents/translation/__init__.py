"""Patent Agent - Translation (EN→KR) agents for patent translation workflow."""

from patent_agent.agents.translation.base import (
    BaseTranslationAgent,
    ClaimTranslationRules,
    MistranslationPrevention,
    PunctuationRules,
    SpecTranslationRules,
    StyleGuideRules,
    TranslationGuidelines,
)
from patent_agent.agents.translation.document_intake import DocumentIntakeAgent
from patent_agent.agents.translation.glossary_builder import GlossaryBuilderAgent
from patent_agent.agents.translation.output_generator import OutputGeneratorAgent
from patent_agent.agents.translation.quality_verifier import QualityVerifierAgent
from patent_agent.agents.translation.section_translator import SectionTranslatorAgent
from patent_agent.agents.translation.structure_analyzer import StructureAnalyzerAgent

__all__ = [
    # Base class and rules
    "BaseTranslationAgent",
    "StyleGuideRules",
    "ClaimTranslationRules",
    "SpecTranslationRules",
    "PunctuationRules",
    "MistranslationPrevention",
    "TranslationGuidelines",
    # Phase agents
    "DocumentIntakeAgent",  # P1
    "StructureAnalyzerAgent",  # P2
    "GlossaryBuilderAgent",  # P3
    "SectionTranslatorAgent",  # P4
    "QualityVerifierAgent",  # P5
    "OutputGeneratorAgent",  # P6
]
