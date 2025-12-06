"""Query Builder Agent for prior art search.

Builds optimized search queries with:
- Keyword extraction and expansion
- IPC/CPC code determination
- Database-specific query formatting
"""

from datetime import datetime
from typing import Any

from patent_agent.agents.prior_art.base import (
    COMMON_IPC_CODES,
    BasePriorArtAgent,
)
from patent_agent.state.prior_art import (
    DatabaseID,
    PriorArtSearchState,
    PriorArtStep,
    SearchQuery,
)


class QueryBuilderAgent(BasePriorArtAgent):
    """Agent for building optimized search queries.

    This agent analyzes the target invention and generates:
    - Core search keywords
    - Expanded keywords (synonyms, related terms)
    - IPC/CPC classification codes
    - Database-specific query strings
    """

    @property
    def name(self) -> str:
        return "QueryBuilder"

    @property
    def description(self) -> str:
        return "검색 쿼리 최적화 및 IPC/CPC 분류"

    @property
    def step(self) -> PriorArtStep:
        return "query"

    def get_system_prompt(self) -> str:
        return """당신은 특허 선행기술조사 전문가입니다.
발명의 핵심 기술 특징을 분석하여 최적의 검색 전략을 수립합니다.

## 주요 임무
1. 발명 요약에서 핵심 키워드 추출
2. 동의어, 상위/하위 개념으로 키워드 확장
3. 적절한 IPC/CPC 분류 코드 결정
4. 데이터베이스별 최적화된 쿼리 생성

## IPC 섹션 가이드
- A: 생활필수품
- B: 처리 조작; 운수
- C: 화학; 야금
- D: 섬유; 지류
- E: 고정 구조물
- F: 기계 공학; 조명; 가열; 무기; 폭파
- G: 물리학 (컴퓨터, AI 포함)
- H: 전기 (반도체, 통신 포함)

## 키워드 확장 원칙
- 동의어: 같은 의미의 다른 표현
- 상위 개념: 더 넓은 범주
- 하위 개념: 더 구체적인 구현
- 영문/한글 모두 포함
"""

    async def process(self, state: PriorArtSearchState) -> dict[str, Any]:
        """Build search queries for the target invention.

        Args:
            state: Current workflow state with target invention

        Returns:
            State updates with search queries
        """
        self._logger.info(
            "building_search_queries",
            search_type=state.get("search_type"),
        )

        target_invention = state.get("target_invention", "")
        target_claims = state.get("target_claims", [])
        target_filing_date = state.get("target_filing_date")

        if not target_invention:
            return {
                "error_messages": ["발명 요약이 제공되지 않았습니다."],
                "is_error_state": True,
            }

        # Extract keywords using LLM
        keywords = await self._extract_keywords(target_invention, target_claims)

        # Expand keywords
        expanded = await self._expand_keywords(keywords, target_invention)

        # Determine IPC/CPC codes
        ipc_codes = await self._determine_ipc_codes(target_invention, keywords)
        cpc_codes = await self._determine_cpc_codes(target_invention, keywords)

        # Build database-specific queries
        search_queries = self._build_database_queries(
            keywords=keywords,
            expanded_keywords=expanded,
            ipc_codes=ipc_codes,
            cpc_codes=cpc_codes,
            date_cutoff=target_filing_date,
        )

        self._logger.info(
            "queries_built",
            keyword_count=len(keywords),
            expanded_count=len(expanded),
            ipc_codes=ipc_codes,
            databases=list(search_queries.keys()),
        )

        return {
            "search_keywords": keywords,
            "expanded_keywords": expanded,
            "ipc_codes": ipc_codes,
            "cpc_codes": cpc_codes,
            "search_queries": search_queries,
            "current_step": "search",
            "updated_at": datetime.now().isoformat(),
        }

    async def _extract_keywords(
        self,
        invention: str,
        claims: list[str],
    ) -> list[str]:
        """Extract core keywords from invention description.

        Args:
            invention: Invention summary
            claims: List of claims

        Returns:
            List of core keywords
        """
        claims_text = "\n".join(f"- {c}" for c in claims) if claims else "없음"

        prompt = f"""다음 발명에서 선행기술 검색에 사용할 핵심 키워드를 추출하세요.

## 발명 요약
{invention}

## 청구항
{claims_text}

## 요구사항
- 기술적 특징을 나타내는 명사/명사구 추출
- 한국어와 영어 키워드 모두 포함
- 5-15개 키워드 추출
- 각 키워드는 한 줄에 하나씩, '-' 로 시작

## 출력 형식
- 키워드1
- 키워드2
...
"""

        response = await self._invoke_llm(prompt)

        # Parse keywords from response
        keywords = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("-"):
                keyword = line[1:].strip()
                if keyword:
                    keywords.append(keyword)

        return keywords if keywords else self._fallback_keyword_extraction(invention)

    def _fallback_keyword_extraction(self, text: str) -> list[str]:
        """Simple fallback keyword extraction.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        import re

        # Extract Korean words (2+ characters)
        korean_words = re.findall(r"[\uac00-\ud7af]{2,}", text)

        # Extract English words (3+ characters)
        english_words = re.findall(r"[a-zA-Z]{3,}", text)

        # Combine and deduplicate
        all_words = list(set(korean_words + [w.lower() for w in english_words]))

        return all_words[:10]

    async def _expand_keywords(
        self,
        keywords: list[str],
        context: str,
    ) -> list[str]:
        """Expand keywords with synonyms and related terms.

        Args:
            keywords: Core keywords
            context: Invention context

        Returns:
            Expanded keyword list
        """
        keywords_text = ", ".join(keywords)

        prompt = f"""다음 키워드들의 동의어, 상위개념, 하위개념을 제시하세요.

## 핵심 키워드
{keywords_text}

## 발명 맥락
{context[:500]}

## 요구사항
- 각 키워드에 대해 관련 용어 1-3개 추가
- 영문 표현 포함
- 특허 검색에 유용한 용어만 포함

## 출력 형식 (각 용어 한 줄)
- 확장된 용어1
- 확장된 용어2
...
"""

        response = await self._invoke_llm(prompt)

        expanded = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("-"):
                term = line[1:].strip()
                if term and term not in keywords:
                    expanded.append(term)

        return expanded[:20]  # Limit to 20 expanded terms

    async def _determine_ipc_codes(
        self,
        invention: str,
        keywords: list[str],
    ) -> list[str]:
        """Determine relevant IPC classification codes.

        Args:
            invention: Invention summary
            keywords: Extracted keywords

        Returns:
            List of IPC codes
        """
        # First try rule-based suggestion
        suggested = []
        invention_lower = invention.lower()
        keywords_lower = " ".join(keywords).lower()

        for domain, codes in COMMON_IPC_CODES.items():
            if domain in invention_lower or domain in keywords_lower:
                suggested.extend(codes)

        # Use LLM for refinement
        prompt = f"""다음 발명에 적합한 IPC 분류 코드를 제시하세요.

## 발명 요약
{invention[:500]}

## 키워드
{', '.join(keywords[:10])}

## 초기 제안 코드
{', '.join(suggested) if suggested else '없음'}

## 요구사항
- 가장 관련성 높은 IPC 코드 3-5개 선택
- 형식: G06N, H04L 등 (클래스 또는 서브클래스 수준)
- 한 줄에 하나씩

## 출력 (코드만)
"""

        response = await self._invoke_llm(prompt)

        # Parse IPC codes
        import re

        ipc_pattern = r"[A-H]\d{2}[A-Z]?\d*/?[\d]*"
        codes = re.findall(ipc_pattern, response.upper())

        # Combine with suggested, deduplicate
        all_codes = list(dict.fromkeys(suggested + codes))

        return all_codes[:5] if all_codes else ["G06F"]

    async def _determine_cpc_codes(
        self,
        invention: str,
        keywords: list[str],
    ) -> list[str]:
        """Determine relevant CPC classification codes.

        CPC codes are similar to IPC but with more granularity.
        For simplicity, we derive from IPC codes.

        Args:
            invention: Invention summary
            keywords: Extracted keywords

        Returns:
            List of CPC codes
        """
        # CPC often mirrors IPC for main classifications
        ipc_codes = await self._determine_ipc_codes(invention, keywords)

        # CPC codes are same format, just return IPC for now
        # In production, would query CPC classification service
        return ipc_codes

    def _build_database_queries(
        self,
        keywords: list[str],
        expanded_keywords: list[str],
        ipc_codes: list[str],
        cpc_codes: list[str],
        date_cutoff: str | None,
    ) -> dict[DatabaseID, SearchQuery]:
        """Build database-specific search queries.

        Args:
            keywords: Core keywords
            expanded_keywords: Expanded keywords
            ipc_codes: IPC codes
            cpc_codes: CPC codes
            date_cutoff: Date cutoff for search

        Returns:
            Dictionary of database-specific queries
        """
        queries: dict[DatabaseID, SearchQuery] = {}

        # Combine keywords for search
        all_keywords = keywords + expanded_keywords[:10]

        # KIPRIS query (Korean patents)
        queries["KIPRIS"] = SearchQuery(
            database="KIPRIS",
            keywords=keywords,
            ipc_codes=ipc_codes,
            cpc_codes=[],  # KIPRIS uses IPC primarily
            applicant=None,
            date_range=(None, date_cutoff) if date_cutoff else None,
            query_string=self._build_kipris_query(all_keywords, ipc_codes),
        )

        # USPTO query (US patents)
        queries["USPTO"] = SearchQuery(
            database="USPTO",
            keywords=keywords,
            ipc_codes=[],  # USPTO uses CPC primarily
            cpc_codes=cpc_codes,
            applicant=None,
            date_range=(None, date_cutoff) if date_cutoff else None,
            query_string=self._build_uspto_query(all_keywords, cpc_codes),
        )

        # Google Patents query
        queries["Google_Patents"] = SearchQuery(
            database="Google_Patents",
            keywords=all_keywords,
            ipc_codes=ipc_codes,
            cpc_codes=cpc_codes,
            applicant=None,
            date_range=(None, date_cutoff) if date_cutoff else None,
            query_string=self._build_google_query(all_keywords, ipc_codes),
        )

        return queries

    def _build_kipris_query(
        self,
        keywords: list[str],
        ipc_codes: list[str],
    ) -> str:
        """Build KIPRIS-specific query string.

        Args:
            keywords: Search keywords
            ipc_codes: IPC codes

        Returns:
            KIPRIS query string
        """
        # KIPRIS supports Korean keywords directly
        korean_keywords = [k for k in keywords if any("\uac00" <= c <= "\ud7af" for c in k)]
        english_keywords = [k for k in keywords if not any("\uac00" <= c <= "\ud7af" for c in k)]

        parts = []

        if korean_keywords:
            parts.append(" OR ".join(f'"{k}"' for k in korean_keywords[:5]))

        if english_keywords:
            parts.append(" OR ".join(f'"{k}"' for k in english_keywords[:5]))

        query = " OR ".join(parts) if parts else keywords[0] if keywords else ""

        if ipc_codes:
            ipc_part = " OR ".join(f"IPC:{code}" for code in ipc_codes[:3])
            query = f"({query}) AND ({ipc_part})"

        return query

    def _build_uspto_query(
        self,
        keywords: list[str],
        cpc_codes: list[str],
    ) -> str:
        """Build USPTO-specific query string.

        Args:
            keywords: Search keywords
            cpc_codes: CPC codes

        Returns:
            USPTO query string
        """
        # USPTO prefers English keywords
        english_keywords = [
            k for k in keywords if not any("\uac00" <= c <= "\ud7af" for c in k)
        ]

        if not english_keywords:
            english_keywords = keywords[:5]

        keyword_part = " OR ".join(f'"{k}"' for k in english_keywords[:5])

        if cpc_codes:
            cpc_part = " OR ".join(f"CPC:{code}" for code in cpc_codes[:3])
            return f"({keyword_part}) AND ({cpc_part})"

        return keyword_part

    def _build_google_query(
        self,
        keywords: list[str],
        ipc_codes: list[str],
    ) -> str:
        """Build Google Patents query string.

        Args:
            keywords: Search keywords
            ipc_codes: IPC codes

        Returns:
            Google Patents query string
        """
        # Google Patents supports natural language search
        query = " ".join(keywords[:10])

        if ipc_codes:
            query += f" IPC:{ipc_codes[0]}"

        return query
