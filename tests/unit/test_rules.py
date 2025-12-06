"""
Unit tests for rules/guidelines system.

Tests validation of:
- Translation laws (LAW-T-1 through LAW-T-4)
- OA response laws (LAW-6 through LAW-10)
- Spec writing laws (LAW-SW-1 through LAW-SW-5)
"""

import pytest

from patent_agent.rules.base import ViolationSeverity


class TestTranslationLaws:
    """Test TranslationLaws validation."""

    def test_sanggi_violation_in_non_claim(self, translation_laws):
        """LAW-T-1: '상기' should not appear in non-claim sections."""
        text = "본 발명의 상기 프로세서는 연산을 수행한다."
        result = translation_laws.validate_output(text, {"section_type": "description"})

        assert not result.is_valid
        assert any(v.law_id == "LAW-T-1" for v in result.violations)
        assert result.has_critical_violations

    def test_sanggi_allowed_in_claims(self, translation_laws):
        """LAW-T-1: '상기' is allowed in claims."""
        text = "상기 프로세서와 연결된 상기 메모리를 포함하는 장치."
        result = translation_laws.validate_output(text, {"section_type": "claims"})

        # Should not have LAW-T-1 violation for '상기' in claims
        sanggi_violations = [v for v in result.violations if v.law_id == "LAW-T-1"]
        assert len(sanggi_violations) == 0

    def test_single_sentence_violation(self, translation_laws):
        """LAW-T-2: Claims must be single sentence."""
        text = "프로세서를 포함한다. 상기 프로세서는 연산을 수행한다."
        result = translation_laws.validate_output(text, {"section_type": "claims"})

        assert not result.is_valid
        assert any(v.law_id == "LAW-T-2" for v in result.violations)

    def test_single_sentence_valid(self, translation_laws):
        """LAW-T-2: Valid single sentence claim."""
        text = "프로세서를 포함하고, 상기 프로세서는 연산을 수행하는 것을 특징으로 하는 장치."
        result = translation_laws.validate_output(text, {"section_type": "claims"})

        law_t2_violations = [v for v in result.violations if v.law_id == "LAW-T-2"]
        assert len(law_t2_violations) == 0

    def test_drawing_reference_without_parentheses(self, translation_laws):
        """LAW-T-4: Drawing references must have parentheses."""
        text = "프로세서 100은 메모리 200과 연결된다."
        result = translation_laws.validate_output(text, {"section_type": "description"})

        # Should detect missing parentheses
        law_t4_violations = [v for v in result.violations if v.law_id == "LAW-T-4"]
        assert len(law_t4_violations) > 0

    def test_drawing_reference_with_parentheses(self, translation_laws):
        """LAW-T-4: Valid drawing references with parentheses."""
        text = "프로세서(100)는 메모리(200)와 연결된다."
        result = translation_laws.validate_output(text, {"section_type": "description"})

        law_t4_violations = [v for v in result.violations if v.law_id == "LAW-T-4"]
        assert len(law_t4_violations) == 0


class TestOAResponseLaws:
    """Test OAResponseLaws validation."""

    def test_prohibited_expression_detection(self, oa_laws):
        """LAW-8: Detect prohibited expressions."""
        text = "이 기술적 특징은 선행기술과 다른 것으로 보인다."
        result = oa_laws.validate_output(text)

        assert not result.is_valid
        assert any(v.law_id == "LAW-8" for v in result.violations)
        assert result.has_critical_violations

    def test_multiple_prohibited_expressions(self, oa_laws):
        """LAW-8: Detect multiple prohibited expressions."""
        text = "일반적으로 이러한 기술은 통상적으로 사용된다."
        result = oa_laws.validate_output(text)

        law_8_violations = [v for v in result.violations if v.law_id == "LAW-8"]
        assert len(law_8_violations) >= 2

    def test_no_prohibited_expressions(self, oa_laws):
        """LAW-8: Text without prohibited expressions."""
        text = "[출원서:p.15:para.3]에 따르면 본 발명은 A와 B를 포함한다."
        result = oa_laws.validate_output(text)

        law_8_violations = [v for v in result.violations if v.law_id == "LAW-8"]
        assert len(law_8_violations) == 0

    def test_citation_format_valid(self, oa_laws):
        """LAW-7: Valid citation format."""
        text = "[D1:p.7:para.2]에 개시된 바와 같이 구성요소 A가 포함된다."
        result = oa_laws.validate_output(text, {"section_type": "analysis"})

        law_7_violations = [v for v in result.violations if v.law_id == "LAW-7"]
        assert len(law_7_violations) == 0

    def test_missing_citation(self, oa_laws):
        """LAW-7: Missing citation in analysis section."""
        text = "청구항 1의 구성요소 A는 선행기술과 구별된다."
        result = oa_laws.validate_output(text, {"section_type": "analysis"})

        law_7_violations = [v for v in result.violations if v.law_id == "LAW-7"]
        assert len(law_7_violations) > 0


class TestSpecWritingLaws:
    """Test SpecWritingLaws validation."""

    def test_list_format_violation(self, spec_writing_laws):
        """LAW-SW-1: List format not allowed in non-claim sections."""
        text = """본 발명의 특징:
- 첫 번째 특징
- 두 번째 특징
- 세 번째 특징"""
        result = spec_writing_laws.validate_output(text, {"section_type": "description"})

        assert any(v.law_id == "LAW-SW-1" for v in result.violations)

    def test_numbered_list_violation(self, spec_writing_laws):
        """LAW-SW-1: Numbered list not allowed in non-claim sections."""
        text = """본 발명의 효과:
1. 첫 번째 효과
2. 두 번째 효과"""
        result = spec_writing_laws.validate_output(text, {"section_type": "effects"})

        assert any(v.law_id == "LAW-SW-1" for v in result.violations)

    def test_narrative_format_valid(self, spec_writing_laws):
        """LAW-SW-1: Narrative format is valid."""
        text = "본 발명의 첫 번째 특징은 A이며, 두 번째 특징은 B이다. 또한 세 번째 특징으로 C를 포함한다."
        result = spec_writing_laws.validate_output(text, {"section_type": "description"})

        law_sw1_violations = [v for v in result.violations if v.law_id == "LAW-SW-1"]
        assert len(law_sw1_violations) == 0

    def test_claim_multiple_periods(self, spec_writing_laws):
        """LAW-SW-3: Claims should have only one period at the end."""
        text = "프로세서를 포함한다. 상기 프로세서는 메모리와 연결된다."
        result = spec_writing_laws.validate_output(text, {"section_type": "claims"})

        assert any(v.law_id == "LAW-SW-3" for v in result.violations)
        assert result.has_critical_violations


class TestGuidelineSystemPrompt:
    """Test system prompt generation."""

    def test_translation_laws_prompt(self, translation_laws):
        """Test TranslationLaws generates valid prompt."""
        prompt = translation_laws.get_system_prompt()

        assert "LAW-T-1" in prompt
        assert "LAW-T-2" in prompt
        assert "LAW-T-3" in prompt
        assert "LAW-T-4" in prompt
        assert "상기" in prompt
        assert "comprising" in prompt

    def test_oa_laws_prompt(self, oa_laws):
        """Test OAResponseLaws generates valid prompt."""
        prompt = oa_laws.get_system_prompt()

        assert "LAW-6" in prompt
        assert "LAW-7" in prompt
        assert "LAW-8" in prompt
        assert "할루시네이션" in prompt or "hallucination" in prompt.lower()

    def test_spec_writing_laws_prompt(self, spec_writing_laws):
        """Test SpecWritingLaws generates valid prompt."""
        prompt = spec_writing_laws.get_system_prompt()

        assert "LAW-SW-1" in prompt
        assert "LAW-SW-2" in prompt
        assert "서술형" in prompt
        assert "특허법" in prompt or "제42조" in prompt
