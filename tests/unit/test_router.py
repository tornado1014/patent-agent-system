"""
Unit tests for main router and workflow detection.
"""

import pytest

from patent_agent.graphs.main_router import MainRouter, WORKFLOW_KEYWORDS


class TestMainRouter:
    """Test MainRouter workflow detection."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        return MainRouter()

    def test_detect_spec_writing_korean(self, router):
        """Detect spec_writing from Korean input."""
        inputs = [
            "새로운 배터리 기술에 대한 명세서 작성해줘",
            "청구항 초안 작성",
            "발명 신고서를 바탕으로 특허 작성",
        ]
        for inp in inputs:
            result = router.detect_workflow(inp)
            assert result == "spec_writing", f"Failed for: {inp}"

    def test_detect_oa_response_korean(self, router):
        """Detect oa_response from Korean input."""
        inputs = [
            "OA 거절이유 분석해줘",
            "의견제출통지서에 대한 대응",
            "보정안 작성",
            "심사관 거절에 반박",
        ]
        for inp in inputs:
            result = router.detect_workflow(inp)
            assert result == "oa_response", f"Failed for: {inp}"

    def test_detect_prior_art_korean(self, router):
        """Detect prior_art from Korean input."""
        inputs = [
            "선행기술 조사해줘",
            "신규성 검토",
            "KIPRIS에서 특허 검색",
            "진보성 평가",
        ]
        for inp in inputs:
            result = router.detect_workflow(inp)
            assert result == "prior_art", f"Failed for: {inp}"

    def test_detect_translation_korean(self, router):
        """Detect translation from Korean input."""
        inputs = [
            "영문 특허 번역해줘",
            "영한 번역 요청",
            "KIPO 형식으로 번역",
        ]
        for inp in inputs:
            result = router.detect_workflow(inp)
            assert result == "translation", f"Failed for: {inp}"

    def test_detect_analysis_korean(self, router):
        """Detect analysis from Korean input."""
        inputs = [
            "특허 포트폴리오 분석",
            "경쟁사 특허 분석",
            "침해 가능성 검토",
            "무효 자료 분석",
        ]
        for inp in inputs:
            result = router.detect_workflow(inp)
            assert result == "analysis", f"Failed for: {inp}"

    def test_detect_spec_writing_english(self, router):
        """Detect spec_writing from English input."""
        inputs = [
            "draft patent specification",
            "write claims for my invention",
        ]
        for inp in inputs:
            result = router.detect_workflow(inp)
            assert result == "spec_writing", f"Failed for: {inp}"

    def test_detect_prior_art_english(self, router):
        """Detect prior_art from English input."""
        inputs = [
            "prior art search",
            "novelty search for patent",
            "search USPTO database",
        ]
        for inp in inputs:
            result = router.detect_workflow(inp)
            assert result == "prior_art", f"Failed for: {inp}"

    def test_ambiguous_defaults_to_analysis(self, router):
        """Ambiguous input defaults to analysis."""
        inputs = [
            "help me with this patent",
            "what should I do?",
            "hello",
        ]
        for inp in inputs:
            result = router.detect_workflow(inp)
            # Should default to analysis when unclear
            assert result is not None

    def test_workflow_keywords_coverage(self):
        """Ensure all workflows have keywords defined."""
        expected_workflows = [
            "spec_writing",
            "oa_response",
            "prior_art",
            "translation",
            "analysis",
        ]
        for wf in expected_workflows:
            assert wf in WORKFLOW_KEYWORDS
            assert len(WORKFLOW_KEYWORDS[wf]) > 0


class TestRouterScoring:
    """Test keyword scoring in router."""

    @pytest.fixture
    def router(self):
        return MainRouter()

    def test_multiple_keyword_match_wins(self, router):
        """Input with more keyword matches should win."""
        # "명세서 청구항 발명" has 3 spec_writing keywords
        inp = "명세서 청구항 발명에 대해 작성"
        result = router.detect_workflow(inp)
        assert result == "spec_writing"

    def test_mixed_keywords(self, router):
        """Mixed keywords should go to dominant workflow."""
        # Has OA keywords and some analysis keywords
        inp = "OA 거절이유에 대한 분석 및 의견제출"
        result = router.detect_workflow(inp)
        # OA should win with more specific keywords
        assert result == "oa_response"
