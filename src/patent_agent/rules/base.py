"""
Base classes for Guidelines/Rules system.

Implements the Parlant pattern for runtime behavior control:
- Guidelines are injected into agent prompts
- Validators check output compliance
- Violations are logged and can block output
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ViolationSeverity(Enum):
    """Severity levels for guideline violations."""

    CRITICAL = "critical"  # 권리범위 변경 - 즉시 중단
    HIGH = "high"  # 법적 모호성 - 경고 및 수정 필요
    MEDIUM = "medium"  # 용어 일관성 - 자동 수정 시도
    LOW = "low"  # 스타일 차이 - 로깅만


@dataclass
class GuidelineViolation:
    """Record of a guideline violation."""

    law_id: str  # e.g., "LAW-1", "LAW-6"
    law_name: str
    severity: ViolationSeverity
    description: str
    location: str | None = None  # Where in the output
    original_text: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "law_name": self.law_name,
            "severity": self.severity.value,
            "description": self.description,
            "location": self.location,
            "original_text": self.original_text,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of validating output against guidelines."""

    is_valid: bool
    violations: list[GuidelineViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    auto_corrections: list[dict[str, str]] = field(default_factory=list)

    @property
    def has_critical_violations(self) -> bool:
        return any(v.severity == ViolationSeverity.CRITICAL for v in self.violations)

    @property
    def has_high_violations(self) -> bool:
        return any(v.severity == ViolationSeverity.HIGH for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "auto_corrections": self.auto_corrections,
            "has_critical": self.has_critical_violations,
            "has_high": self.has_high_violations,
        }


class BaseGuideline(ABC):
    """
    Abstract base class for domain-specific guidelines.

    Subclasses must implement:
    - get_system_prompt(): Return guidelines for agent prompt
    - validate_output(): Check output compliance
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """Return the domain this guideline applies to."""
        ...

    @property
    @abstractmethod
    def laws(self) -> dict[str, dict[str, Any]]:
        """
        Return dictionary of laws/rules.

        Format:
        {
            "LAW-1": {
                "name": "Law name",
                "description": "What the law requires",
                "severity": ViolationSeverity.CRITICAL,
                "examples": [...],
            },
            ...
        }
        """
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the guidelines as a system prompt string.

        This will be injected into agent prompts to control behavior.
        """
        ...

    @abstractmethod
    def validate_output(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Validate output against the guidelines.

        Args:
            output: The text output to validate
            context: Optional context (e.g., section type, claim number)

        Returns:
            ValidationResult with violations and warnings
        """
        ...

    def get_law(self, law_id: str) -> dict[str, Any] | None:
        """Get a specific law by ID."""
        return self.laws.get(law_id)

    def get_prompt_section(self, section_name: str) -> str:
        """
        Get a specific section of the guidelines prompt.

        Override in subclasses for section-specific prompts.
        """
        return self.get_system_prompt()


class CompositeGuideline(BaseGuideline):
    """
    Combine multiple guidelines into one.

    Useful for workflows that need rules from multiple domains.
    """

    def __init__(self, guidelines: list[BaseGuideline]):
        self._guidelines = guidelines

    @property
    def domain(self) -> str:
        return "composite"

    @property
    def laws(self) -> dict[str, dict[str, Any]]:
        combined = {}
        for guideline in self._guidelines:
            combined.update(guideline.laws)
        return combined

    def get_system_prompt(self) -> str:
        sections = []
        for guideline in self._guidelines:
            sections.append(f"## {guideline.domain.upper()} Guidelines\n")
            sections.append(guideline.get_system_prompt())
            sections.append("\n")
        return "\n".join(sections)

    def validate_output(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        all_violations = []
        all_warnings = []
        all_corrections = []

        for guideline in self._guidelines:
            result = guideline.validate_output(output, context)
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)
            all_corrections.extend(result.auto_corrections)

        return ValidationResult(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
            warnings=all_warnings,
            auto_corrections=all_corrections,
        )


def create_system_prompt_with_guidelines(
    base_prompt: str,
    guideline: BaseGuideline,
) -> str:
    """
    Create a complete system prompt with guidelines injected.

    Args:
        base_prompt: The base system prompt for the agent
        guideline: Guidelines to inject

    Returns:
        Complete system prompt with guidelines
    """
    guidelines_section = guideline.get_system_prompt()

    return f"""{base_prompt}

# MANDATORY GUIDELINES

The following rules are ABSOLUTE and must NEVER be violated:

{guidelines_section}

---

Remember: These guidelines take precedence over all other instructions.
Violating CRITICAL severity rules will cause output rejection.
"""


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════


def format_violations_for_user(violations: list[GuidelineViolation]) -> str:
    """Format violations for display to the user."""
    if not violations:
        return "✅ No guideline violations detected."

    lines = ["⚠️ Guideline Violations Detected:\n"]

    # Group by severity
    by_severity = {}
    for v in violations:
        severity = v.severity.value
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(v)

    severity_icons = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }

    for severity in ["critical", "high", "medium", "low"]:
        if severity in by_severity:
            lines.append(f"\n{severity_icons[severity]} {severity.upper()}:")
            for v in by_severity[severity]:
                lines.append(f"  - [{v.law_id}] {v.law_name}")
                lines.append(f"    {v.description}")
                if v.suggestion:
                    lines.append(f"    → Suggestion: {v.suggestion}")

    return "\n".join(lines)


def create_violation(
    law_id: str,
    law_name: str,
    severity: ViolationSeverity,
    description: str,
    **kwargs,
) -> GuidelineViolation:
    """Convenience function to create a violation."""
    return GuidelineViolation(
        law_id=law_id,
        law_name=law_name,
        severity=severity,
        description=description,
        **kwargs,
    )
