"""Relevance Analyzer Agent for prior art search.

Analyzes and scores search results for relevance to the target invention.
Filters results based on relevance threshold.
"""

from datetime import datetime
from typing import Any

from patent_agent.agents.prior_art.base import BasePriorArtAgent
from patent_agent.state.prior_art import (
    AnalyzedReference,
    PatentResult,
    PriorArtSearchState,
    PriorArtStep,
)


class RelevanceAnalyzerAgent(BasePriorArtAgent):
    """Agent for analyzing and scoring search result relevance.

    Features:
    - Multi-factor relevance scoring
    - Embedding-based similarity (when available)
    - Threshold-based filtering
    - Confidence assessment
    """

    @property
    def name(self) -> str:
        return "RelevanceAnalyzer"

    @property
    def description(self) -> str:
        return "검색 결과 관련성 분석 및 필터링"

    @property
    def step(self) -> PriorArtStep:
        return "filter"

    def get_system_prompt(self) -> str:
        return """당신은 특허 관련성 분석 전문가입니다.
검색된 선행기술 문헌의 관련성을 평가합니다.

## 관련성 평가 기준

### 1. 기술적 특징 중복도 (40%)
- 동일한 기술적 구성요소 개시 여부
- 유사한 기술적 수단 사용 여부
- 동일한 기술 분야 여부

### 2. 청구항 구성요소 포함도 (30%)
- 청구항 각 구성요소의 개시 여부
- 균등 구성요소 포함 여부

### 3. 시간적 관련성 (15%)
- 출원일 기준 선후관계
- 기술 발전 시기적 적합성

### 4. 출원인/발명자 관련성 (15%)
- 동일 출원인 여부
- 관련 연구그룹 여부

## 관련성 임계값
- 0.7 이상: 상세 분석 대상
- 0.5 이상: 참고 문헌
- 0.5 미만: 제외
"""

    async def process(self, state: PriorArtSearchState) -> dict[str, Any]:
        """Analyze relevance of search results.

        Args:
            state: Current workflow state with raw results

        Returns:
            State updates with analyzed and filtered results
        """
        raw_results = state.get("raw_results", {})
        target_invention = state.get("target_invention", "")
        target_claims = state.get("target_claims", [])
        threshold = state.get("relevance_threshold", 0.7)

        if not raw_results:
            return {
                "error_messages": ["검색 결과가 없습니다."],
                "is_error_state": True,
            }

        self._logger.info(
            "analyzing_relevance",
            total_results=sum(len(v) for v in raw_results.values()),
            threshold=threshold,
        )

        # Flatten results from all databases
        all_results: list[tuple[PatentResult, str]] = []
        for db_id, results in raw_results.items():
            for result in results:
                all_results.append((result, db_id))

        # Analyze each result
        analyzed_references: list[AnalyzedReference] = []
        filtered_results: list[PatentResult] = []

        for i, (result, db_id) in enumerate(all_results):
            analysis = await self._analyze_single_result(
                result=result,
                target_invention=target_invention,
                target_claims=target_claims,
                ref_id=f"R{i+1}",
            )

            if analysis["relevance_score"] >= threshold:
                analyzed_references.append(analysis)
                filtered_results.append(result)

        # Sort by relevance score (descending)
        analyzed_references.sort(
            key=lambda x: x["relevance_score"],
            reverse=True,
        )

        self._logger.info(
            "relevance_analysis_complete",
            total_analyzed=len(all_results),
            above_threshold=len(analyzed_references),
        )

        return {
            "analyzed_references": analyzed_references,
            "filtered_results": filtered_results,
            "current_step": "analyze",
            "updated_at": datetime.now().isoformat(),
        }

    async def _analyze_single_result(
        self,
        result: PatentResult,
        target_invention: str,
        target_claims: list[str],
        ref_id: str,
    ) -> AnalyzedReference:
        """Analyze a single search result for relevance.

        Args:
            result: Patent search result
            target_invention: Target invention description
            target_claims: Target claims
            ref_id: Reference ID

        Returns:
            Analyzed reference with scores
        """
        title = result.get("title", "")
        abstract = result.get("abstract", "")
        ipc_codes = result.get("ipc_codes", [])

        # Calculate component scores
        technical_overlap = await self._calculate_technical_overlap(
            target_invention,
            f"{title} {abstract}",
        )

        claim_coverage = await self._calculate_claim_coverage(
            target_claims,
            f"{title} {abstract}",
        )

        temporal_relevance = self._calculate_temporal_relevance(
            result.get("filing_date", ""),
        )

        applicant_relevance = 0.0  # Would need more context to calculate

        # Calculate weighted relevance score
        relevance_score = self.calculate_relevance_score(
            technical_overlap,
            claim_coverage,
            temporal_relevance,
            applicant_relevance,
        )

        # Identify matched and missing features
        matched_features, missing_features = await self._identify_features(
            target_claims,
            f"{title} {abstract}",
        )

        # Generate differentiation analysis
        differentiation = await self._analyze_differentiation(
            target_invention,
            f"{title} {abstract}",
        )

        confidence = self.relevance_to_confidence(relevance_score)

        return AnalyzedReference(
            reference_id=ref_id,
            patent_result=result,
            relevance_score=relevance_score,
            matched_claims=[],  # Would need detailed claim analysis
            key_features_matched=matched_features,
            key_features_missing=missing_features,
            differentiation=differentiation,
            confidence=confidence,
        )

    async def _calculate_technical_overlap(
        self,
        target: str,
        reference: str,
    ) -> float:
        """Calculate technical feature overlap score.

        Args:
            target: Target invention text
            reference: Reference text

        Returns:
            Overlap score (0-1)
        """
        # Try to use embeddings if available
        try:
            from patent_agent.tools.embeddings import PatentEmbeddings

            embeddings = PatentEmbeddings()
            similarity = await embeddings.similarity(target, reference)
            return min(max(similarity, 0.0), 1.0)
        except Exception:
            pass

        # Fallback to keyword-based similarity
        return self._keyword_similarity(target, reference)

    async def _calculate_claim_coverage(
        self,
        claims: list[str],
        reference: str,
    ) -> float:
        """Calculate claim element coverage score.

        Args:
            claims: Target claims
            reference: Reference text

        Returns:
            Coverage score (0-1)
        """
        if not claims:
            return 0.5  # Neutral score if no claims

        # Simple keyword matching for each claim
        reference_lower = reference.lower()
        matched = 0

        for claim in claims:
            # Extract significant words from claim
            words = self._extract_significant_words(claim)

            # Count matches
            claim_matches = sum(1 for w in words if w.lower() in reference_lower)
            if words and claim_matches / len(words) > 0.3:
                matched += 1

        return matched / len(claims) if claims else 0.0

    def _calculate_temporal_relevance(self, filing_date: str) -> float:
        """Calculate temporal relevance score.

        Args:
            filing_date: Reference filing date

        Returns:
            Temporal relevance score (0-1)
        """
        if not filing_date:
            return 0.5  # Neutral if no date

        from datetime import datetime

        try:
            if isinstance(filing_date, str):
                ref_date = datetime.fromisoformat(filing_date.replace("Z", "+00:00"))
            else:
                ref_date = filing_date

            # Calculate age in years
            age_days = (datetime.now() - ref_date.replace(tzinfo=None)).days
            age_years = age_days / 365

            # Score decreases with age
            # Full score for patents < 5 years
            # Decreasing score up to 20 years
            if age_years <= 5:
                return 1.0
            elif age_years <= 20:
                return 1.0 - (age_years - 5) / 30
            else:
                return 0.5

        except Exception:
            return 0.5

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple keyword-based similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        words1 = set(self._extract_significant_words(text1))
        words2 = set(self._extract_significant_words(text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _extract_significant_words(self, text: str) -> list[str]:
        """Extract significant words from text.

        Args:
            text: Input text

        Returns:
            List of significant words
        """
        import re

        # Extract Korean and English words
        korean = re.findall(r"[\uac00-\ud7af]{2,}", text)
        english = re.findall(r"[a-zA-Z]{3,}", text.lower())

        # Filter common stopwords
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from",
            "are", "was", "were", "been", "have", "has", "had",
            "것", "이", "가", "을", "를", "에", "의", "로", "으로",
        }

        words = [w for w in korean + english if w.lower() not in stopwords]

        return words

    async def _identify_features(
        self,
        claims: list[str],
        reference: str,
    ) -> tuple[list[str], list[str]]:
        """Identify matched and missing features.

        Args:
            claims: Target claims
            reference: Reference text

        Returns:
            Tuple of (matched_features, missing_features)
        """
        if not claims:
            return [], []

        # Use LLM for feature identification
        claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

        prompt = f"""다음 청구항의 구성요소가 선행기술에 개시되어 있는지 분석하세요.

## 청구항
{claims_text}

## 선행기술
{reference[:1000]}

## 요구사항
두 개의 목록을 제공하세요:
1. 선행기술에 개시된 구성요소 (MATCHED:)
2. 선행기술에 없는 구성요소 (MISSING:)

## 출력 형식
MATCHED:
- 구성요소1
- 구성요소2

MISSING:
- 구성요소3
- 구성요소4
"""

        try:
            response = await self._invoke_llm(prompt)

            matched = []
            missing = []
            current_list = None

            for line in response.strip().split("\n"):
                line = line.strip()
                if "MATCHED:" in line.upper():
                    current_list = matched
                elif "MISSING:" in line.upper():
                    current_list = missing
                elif line.startswith("-") and current_list is not None:
                    feature = line[1:].strip()
                    if feature:
                        current_list.append(feature)

            return matched[:5], missing[:5]

        except Exception:
            return [], []

    async def _analyze_differentiation(
        self,
        target: str,
        reference: str,
    ) -> str:
        """Analyze differentiation between target and reference.

        Args:
            target: Target invention
            reference: Reference text

        Returns:
            Differentiation analysis text
        """
        prompt = f"""다음 두 기술 사이의 차별점을 간략히 분석하세요.

## 본원 발명
{target[:500]}

## 선행기술
{reference[:500]}

## 요구사항
- 2-3문장으로 핵심 차이점 설명
- 기술적 특징 중심
"""

        try:
            response = await self._invoke_llm(prompt)
            return response.strip()[:300]  # Limit length
        except Exception:
            return "차별점 분석 불가"
