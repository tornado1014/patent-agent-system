"""
Rules and Guidelines for patent agent behavior control.

Based on Parlant pattern - runtime behavior rules injected into agent prompts.
"""

from patent_agent.rules.base import (
    BaseGuideline,
    GuidelineViolation,
    ValidationResult,
    create_system_prompt_with_guidelines,
)
from patent_agent.rules.spec_writing_laws import SpecWritingLaws
from patent_agent.rules.oa_response_laws import OAResponseLaws
from patent_agent.rules.translation_laws import TranslationLaws


__all__ = [
    "BaseGuideline",
    "GuidelineViolation",
    "ValidationResult",
    "create_system_prompt_with_guidelines",
    "SpecWritingLaws",
    "OAResponseLaws",
    "TranslationLaws",
]
