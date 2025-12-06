"""Rejection Analyzer Agent for OA Response workflow.

Phase 2: 거절이유 분석 (3층 검증)

Responsibilities:
- Classify rejection types (novelty, inventive step, etc.)
- Map claims to cited references
- Perform 3-layer verification
- Build rejection analysis with evidence chain
"""

from typing import Any

from patent_agent.agents.oa_response.base import BaseOAResponseAgent
from patent_agent.state.base import ConfidenceLevel
from patent_agent.state.oa_response import (
    OAResponseState,
    RejectionAnalysis,
    RejectionType,
)


class RejectionAnalyzerAgent(BaseOAResponseAgent):
    """Agent for analyzing rejection reasons in OA.

    Phase 2 of PALLAS-EVIDENCE workflow:
    - Classifies rejection types
    - Maps claims to cited references
    - Performs 3-layer verification
    - Builds structured rejection analysis

    3-Layer Verification:
    - Layer 1: Extraction accuracy (claim numbers, pages, references)
    - Layer 2: Logical mapping (rejection-claim-reference connections)
    - Layer 3: Completeness (all rejections addressed)
    """

    # Rejection type classification rules
    REJECTION_PATTERNS = {
        "novelty": [
            "제29조 제1항",
            "신규성",
            "동일",
            "공지",
        ],
        "inventive_step": [
            "제29조 제2항",
            "진보성",
            "용이하게",
            "결합",
            "당업자",
        ],
        "enablement": [
            "제42조 제3항",
            "실시가능",
            "구체적",
        ],
        "written_description": [
            "제42조 제4항",
            "기재불비",
            "발명의 설명",
        ],
        "claim_clarity": [
            "제42조 제4항 제2호",
            "불명확",
            "명확하게",
        ],
        "unity": [
            "제45조",
            "단일성",
        ],
    }

    @property
    def name(self) -> str:
        return "RejectionAnalyzer"

    @property
    def description(self) -> str:
        return "거절이유 분석 에이전트: 심사관의 거절이유를 분류하고 청구항-인용문헌 매핑을 수행합니다."

    @property
    def phase(self) -> str:
        return "P2"

    def get_system_prompt(self) -> str:
        return """당신은 특허 거절이유 분석 전문가입니다.

## 역할
심사관의 거절이유를 정확하게 분석하고, 청구항과 인용문헌 간의 관계를 매핑합니다.

## 분석 프로세스

### 1단계: 거절이유 분류
각 거절이유를 다음 유형으로 분류:
- **신규성 결여** (제29조 제1항): 동일한 발명이 이미 공지
- **진보성 결여** (제29조 제2항): 당업자가 용이하게 도출 가능
- **실시가능성 위반** (제42조 제3항): 당업자가 실시할 수 없음
- **기재불비** (제42조 제4항): 발명의 설명 미비
- **청구항 불명확** (제42조 제4항 제2호): 청구항 기재 불명확
- **단일성 위반** (제45조): 복수 발명 포함

### 2단계: 청구항-인용문헌 매핑
각 거절된 청구항에 대해:
- 해당 청구항 원문 [LAW-6 준수]
- 인용된 문헌 (D1, D2 등)
- 심사관이 지적한 대응 관계

### 3단계: 3층 검증
1. **추출 정확성**: 청구항 번호, 페이지 번호, 인용 부호 일치
2. **논리적 매핑**: 거절이유-청구항-인용문헌 연결 검증
3. **완전성 확인**: 모든 거절 청구항 포함 확인

## 출력 형식
각 거절이유에 대해:
```
## 거절이유 [N]: [유형]

**법적 근거**: [조문]
**대상 청구항**: [번호 목록]
**인용문헌**: [D1, D2 등]

**심사관 주장** (원문):
"[OA 원문 그대로 인용]"
출처: [OA:p.X:para.Y]

**분석**:
- 🟢/🟡/🔴 [분석 내용]
```"""

    async def process(self, state: OAResponseState) -> dict[str, Any]:
        """Analyze rejection reasons and build mappings.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with rejection analysis results
        """
        self._logger.info("starting_rejection_analysis", phase=self.phase)

        # Check required inputs
        oa_document = state.get("oa_document")
        cited_references = state.get("cited_references", [])
        original_claims = state.get("original_claims", [])

        if not oa_document:
            return {
                "is_error_state": True,
                "error_messages": ["OA 문서 정보가 없습니다. Phase 1을 먼저 실행하세요."],
                "current_step": "P2",
            }

        # Analyze each rejection reason
        rejection_analyses = await self._analyze_rejections(
            oa_document, cited_references, original_claims
        )

        # Build claim-reference mapping
        claim_reference_mapping = self._build_claim_reference_mapping(
            rejection_analyses
        )

        # Perform 3-layer verification
        extraction_verified = self._verify_extraction(rejection_analyses, oa_document)
        mapping_verified = self._verify_mapping(rejection_analyses, claim_reference_mapping)
        completeness_verified = self._verify_completeness(
            rejection_analyses, oa_document.get("cited_claims", [])
        )

        self._logger.info(
            "rejection_analysis_complete",
            rejection_count=len(rejection_analyses),
            extraction_verified=extraction_verified,
            mapping_verified=mapping_verified,
            completeness_verified=completeness_verified,
        )

        return {
            "rejection_analyses": rejection_analyses,
            "claim_reference_mapping": claim_reference_mapping,
            "extraction_verified": extraction_verified,
            "mapping_verified": mapping_verified,
            "completeness_verified": completeness_verified,
            "current_step": "P3",
            "is_error_state": False,
        }

    async def _analyze_rejections(
        self,
        oa_document: dict,
        cited_references: list[dict],
        original_claims: list[str],
    ) -> list[RejectionAnalysis]:
        """Analyze each rejection reason.

        Args:
            oa_document: Parsed OA document
            cited_references: List of cited references
            original_claims: Original claim texts

        Returns:
            List of rejection analyses
        """
        rejection_reasons = oa_document.get("rejection_reasons", [])
        cited_claims = oa_document.get("cited_claims", [])
        analyses = []

        for i, reason in enumerate(rejection_reasons):
            # Classify rejection type
            rejection_type = self._classify_rejection_type(reason)

            # Find legal basis
            legal_basis = self._extract_legal_basis(reason)

            # Find cited references in this rejection
            ref_ids = self._find_cited_references(reason, cited_references)

            # Determine affected claims
            affected = self._determine_affected_claims(reason, cited_claims)

            # Analyze with LLM for detailed understanding
            detailed_analysis = await self._get_detailed_analysis(
                reason, rejection_type, cited_references, original_claims
            )

            # Determine confidence level
            confidence = self._assess_confidence(detailed_analysis)

            analysis: RejectionAnalysis = {
                "rejection_type": rejection_type,
                "legal_basis": legal_basis,
                "examiner_argument": reason,  # Original text (LAW-6)
                "cited_references": ref_ids,
                "affected_claims": affected,
                "confidence": confidence,
                "analysis_notes": detailed_analysis,
            }

            analyses.append(analysis)

        return analyses

    def _classify_rejection_type(self, reason_text: str) -> RejectionType:
        """Classify the rejection type based on text patterns.

        Args:
            reason_text: Rejection reason text

        Returns:
            Classified rejection type
        """
        for rejection_type, patterns in self.REJECTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in reason_text:
                    return rejection_type

        return "other"

    def _extract_legal_basis(self, reason_text: str) -> str:
        """Extract legal basis (law article) from rejection text.

        Args:
            reason_text: Rejection reason text

        Returns:
            Legal basis string
        """
        import re

        # Pattern for Korean patent law articles
        patterns = [
            r"특허법\s*제?(\d+)조(?:\s*제?(\d+)항)?(?:\s*제?(\d+)호)?",
            r"제(\d+)조(?:\s*제?(\d+)항)?(?:\s*제?(\d+)호)?",
        ]

        for pattern in patterns:
            match = re.search(pattern, reason_text)
            if match:
                groups = [g for g in match.groups() if g]
                if len(groups) >= 2:
                    return f"특허법 제{groups[0]}조 제{groups[1]}항"
                elif len(groups) >= 1:
                    return f"특허법 제{groups[0]}조"

        return "법적 근거 불명"

    def _find_cited_references(
        self, reason_text: str, cited_references: list[dict]
    ) -> list[str]:
        """Find which references are cited in the rejection.

        Args:
            reason_text: Rejection reason text
            cited_references: List of cited reference objects

        Returns:
            List of reference IDs (D1, D2, etc.)
        """
        import re

        ref_ids = []

        # Look for D1, D2, etc. patterns
        matches = re.findall(r"[DdⅮ](\d+)", reason_text)
        for match in matches:
            ref_id = f"D{match}"
            if ref_id not in ref_ids:
                ref_ids.append(ref_id)

        # Also check for document numbers
        for ref in cited_references:
            doc_num = ref.get("document_number", "")
            if doc_num and doc_num in reason_text:
                ref_id = ref.get("reference_id", "")
                if ref_id and ref_id not in ref_ids:
                    ref_ids.append(ref_id)

        return sorted(ref_ids)

    def _determine_affected_claims(
        self, reason_text: str, all_cited_claims: list[int]
    ) -> list[int]:
        """Determine which claims are affected by this rejection.

        Args:
            reason_text: Rejection reason text
            all_cited_claims: All cited claim numbers from OA

        Returns:
            List of affected claim numbers
        """
        import re

        # Look for specific claim mentions in this rejection
        affected = []

        # Pattern for "청구항 1, 2, 3" style
        matches = re.findall(r"청구항\s*(\d+(?:\s*,\s*\d+)*)", reason_text)
        for match in matches:
            for num in re.findall(r"\d+", match):
                claim_num = int(num)
                if claim_num not in affected:
                    affected.append(claim_num)

        # Pattern for ranges "청구항 1-5"
        range_matches = re.findall(r"청구항\s*(\d+)\s*[-~]\s*(\d+)", reason_text)
        for start, end in range_matches:
            for num in range(int(start), int(end) + 1):
                if num not in affected:
                    affected.append(num)

        # If no specific claims found, use all cited claims
        if not affected:
            affected = all_cited_claims.copy()

        return sorted(affected)

    async def _get_detailed_analysis(
        self,
        reason_text: str,
        rejection_type: RejectionType,
        cited_references: list[dict],
        original_claims: list[str],
    ) -> str:
        """Get detailed analysis from LLM.

        Args:
            reason_text: Rejection reason text
            rejection_type: Classified rejection type
            cited_references: Cited references
            original_claims: Original claims

        Returns:
            Detailed analysis text
        """
        refs_summary = "\n".join([
            f"- {ref.get('reference_id', 'Unknown')}: {ref.get('title', 'Unknown')}"
            for ref in cited_references
        ])

        claims_text = "\n".join(original_claims[:5])  # First 5 claims

        prompt = f"""다음 거절이유를 분석하세요.

## 거절이유 (원문)
"{reason_text}"

## 거절이유 유형
{rejection_type}

## 인용문헌
{refs_summary}

## 청구항 (일부)
{claims_text}

## 분석 요청
1. 심사관의 핵심 주장을 파악하세요
2. 청구항과 인용문헌 간 대응 관계를 식별하세요
3. 논리적 약점이나 반박 가능 포인트를 찾으세요

**중요**: 원문 인용 시 반드시 그대로 인용하고 출처를 명시하세요.
신뢰도(🟢확실/🟡가능/🔴불확실)를 표시하세요."""

        return await self._invoke_llm(prompt)

    def _assess_confidence(self, analysis: str) -> ConfidenceLevel:
        """Assess confidence level based on analysis.

        Args:
            analysis: Analysis text

        Returns:
            Confidence level
        """
        # Count confidence indicators
        certain_count = analysis.count("🟢") + analysis.count("확실")
        possible_count = analysis.count("🟡") + analysis.count("가능")
        uncertain_count = analysis.count("🔴") + analysis.count("불확실")

        # Determine overall confidence
        if uncertain_count > certain_count:
            return "불확실"
        elif possible_count > certain_count:
            return "가능"
        else:
            return "확실"

    def _build_claim_reference_mapping(
        self, analyses: list[RejectionAnalysis]
    ) -> dict[int, list[str]]:
        """Build mapping of claims to their cited references.

        Args:
            analyses: List of rejection analyses

        Returns:
            Dict mapping claim number to list of reference IDs
        """
        mapping: dict[int, list[str]] = {}

        for analysis in analyses:
            refs = analysis.get("cited_references", [])
            for claim_num in analysis.get("affected_claims", []):
                if claim_num not in mapping:
                    mapping[claim_num] = []
                for ref in refs:
                    if ref not in mapping[claim_num]:
                        mapping[claim_num].append(ref)

        return mapping

    # ─────────────────────────────────────────────────────────────
    # 3-Layer Verification
    # ─────────────────────────────────────────────────────────────

    def _verify_extraction(
        self, analyses: list[RejectionAnalysis], oa_document: dict
    ) -> bool:
        """Layer 1: Verify extraction accuracy.

        Checks:
        - Claim numbers match OA document
        - Legal basis is identified
        - Original text is preserved
        """
        if not analyses:
            return False

        cited_claims = set(oa_document.get("cited_claims", []))

        for analysis in analyses:
            # Check claim numbers exist in OA
            affected = set(analysis.get("affected_claims", []))
            if not affected.intersection(cited_claims):
                return False

            # Check legal basis is identified
            if analysis.get("legal_basis") == "법적 근거 불명":
                return False

            # Check examiner argument is preserved
            if not analysis.get("examiner_argument"):
                return False

        return True

    def _verify_mapping(
        self,
        analyses: list[RejectionAnalysis],
        claim_mapping: dict[int, list[str]],
    ) -> bool:
        """Layer 2: Verify logical mapping.

        Checks:
        - Each rejection has corresponding claims
        - Claims are linked to references
        - Connections are logical
        """
        if not analyses or not claim_mapping:
            return False

        for analysis in analyses:
            affected_claims = analysis.get("affected_claims", [])
            cited_refs = analysis.get("cited_references", [])

            # Each rejection should have at least one claim
            if not affected_claims:
                return False

            # For novelty/inventive step, should have cited references
            if analysis.get("rejection_type") in ["novelty", "inventive_step"]:
                if not cited_refs:
                    return False

        return True

    def _verify_completeness(
        self, analyses: list[RejectionAnalysis], all_cited_claims: list[int]
    ) -> bool:
        """Layer 3: Verify completeness.

        Checks:
        - All cited claims are addressed
        - All rejection reasons are analyzed
        """
        if not analyses:
            return False

        # Collect all analyzed claims
        analyzed_claims = set()
        for analysis in analyses:
            analyzed_claims.update(analysis.get("affected_claims", []))

        # Check all cited claims are covered
        all_cited = set(all_cited_claims)
        if not all_cited.issubset(analyzed_claims):
            return False

        return True
