"""DataCollectorAgent for Patent Analysis workflow.

Responsible for collecting patent data from multiple databases:
- KIPRIS (한국특허정보원)
- USPTO (미국특허청)
- EPO (유럽특허청)
- Google Patents
- WIPO

Handles:
- Multi-DB parallel search
- Data normalization
- Deduplication
- Rate limiting
"""

from datetime import datetime
from typing import Any

import structlog

from patent_agent.agents.analysis.base import BaseAnalysisAgent
from patent_agent.state.analysis import (
    AnalysisScope,
    AnalysisStep,
    PatentAnalysisState,
    PatentRecord,
)

logger = structlog.get_logger(__name__)


class DataCollectorAgent(BaseAnalysisAgent):
    """Agent for collecting patent data from multiple databases.

    This agent is responsible for:
    1. Building search queries based on analysis scope
    2. Executing searches across multiple patent databases
    3. Normalizing results to common PatentRecord format
    4. Deduplicating results
    5. Storing collected data in state

    Example:
        >>> agent = DataCollectorAgent()
        >>> state = {
        ...     "analysis_type": "portfolio",
        ...     "target": "삼성전자",
        ...     "scope": {"jurisdictions": ["KR", "US"]},
        ... }
        >>> result = await agent.process(state)
    """

    @property
    def name(self) -> str:
        return "DataCollectorAgent"

    @property
    def description(self) -> str:
        return "다중 특허 DB에서 데이터를 수집하고 정규화"

    @property
    def step(self) -> AnalysisStep:
        return "collect"

    def get_system_prompt(self) -> str:
        return """당신은 특허 데이터 수집 전문가입니다.

## 역할
- 다중 특허 데이터베이스에서 관련 특허 검색
- 검색 결과를 표준 형식으로 정규화
- 중복 제거 및 데이터 품질 관리

## 지원 데이터베이스
1. KIPRIS (한국): 한국 특허/실용신안/디자인/상표
2. USPTO (미국): 미국 특허/출원
3. EPO (유럽): 유럽 특허
4. Google Patents: 전세계 특허 메타데이터
5. WIPO: PCT 출원

## 검색 전략
- 키워드 검색 + IPC/CPC 코드 조합
- 출원인/발명자 필터링
- 날짜 범위 지정
- 법적 상태 필터링

## 정규화 규칙
- 날짜: ISO-8601 형식 (YYYY-MM-DD)
- 특허번호: 국가코드 + 번호 + 종류코드
- IPC 코드: 최신 버전 기준
"""

    async def process(self, state: PatentAnalysisState) -> dict[str, Any]:
        """Collect patent data based on analysis scope.

        Args:
            state: Current workflow state

        Returns:
            State updates with collected patent data
        """
        self._logger.info("starting_data_collection")

        analysis_type = state.get("analysis_type", "portfolio")
        target = state.get("target", "")
        scope = state.get("scope", {})

        if not target:
            return {
                "is_error_state": True,
                "error_messages": ["분석 대상(target)이 지정되지 않았습니다."],
                "current_step": "collect",
            }

        # Build search queries based on analysis type
        queries = self._build_search_queries(analysis_type, target, scope)

        # Execute searches across databases
        all_results: list[PatentRecord] = []
        data_sources: list[str] = []

        for db_name, query in queries.items():
            self._logger.info("searching_database", database=db_name, query=query)
            try:
                results = await self._search_database(db_name, query, scope)
                all_results.extend(results)
                if results:
                    data_sources.append(db_name)
            except Exception as e:
                self._logger.warning(
                    "database_search_failed",
                    database=db_name,
                    error=str(e),
                )

        # Deduplicate results
        deduplicated = self._deduplicate_patents(all_results)

        # Normalize data
        normalized = [self._normalize_patent(p) for p in deduplicated]

        self._logger.info(
            "data_collection_complete",
            total_collected=len(normalized),
            sources=data_sources,
        )

        return {
            "patent_data": normalized,
            "total_patents_collected": len(normalized),
            "data_sources": data_sources,
            "current_step": "process",
            "updated_at": datetime.now().isoformat(),
        }

    def _build_search_queries(
        self,
        analysis_type: str,
        target: str,
        scope: AnalysisScope,
    ) -> dict[str, dict]:
        """Build search queries for each database.

        Args:
            analysis_type: Type of analysis
            target: Analysis target
            scope: Search scope

        Returns:
            Dictionary of database name to query
        """
        jurisdictions = scope.get("jurisdictions", ["KR", "US"])
        keywords = scope.get("keywords", [])
        ipc_codes = scope.get("ipc_codes", [])
        applicants = scope.get("applicants", [])
        date_range = scope.get("date_range")

        queries = {}

        # Build base query
        base_query = {
            "target": target,
            "keywords": keywords,
            "ipc_codes": ipc_codes,
            "date_range": date_range,
        }

        # Customize by analysis type
        if analysis_type == "portfolio":
            # Portfolio: search by applicant
            base_query["applicant"] = target
        elif analysis_type == "competitor":
            # Competitor: search by multiple applicants
            base_query["applicants"] = applicants or [target]
        elif analysis_type in ("infringement", "invalidity"):
            # These focus on specific patent(s)
            base_query["patent_number"] = target
        else:
            # Trend: keyword-based search
            base_query["keywords"] = keywords or [target]

        # Add queries for each jurisdiction
        if "KR" in jurisdictions:
            queries["KIPRIS"] = {**base_query, "jurisdiction": "KR"}

        if "US" in jurisdictions:
            queries["USPTO"] = {**base_query, "jurisdiction": "US"}

        if "EP" in jurisdictions:
            queries["EPO"] = {**base_query, "jurisdiction": "EP"}

        # Always include Google Patents as backup
        queries["Google Patents"] = {**base_query, "jurisdiction": "all"}

        return queries

    async def _search_database(
        self,
        db_name: str,
        query: dict,
        scope: AnalysisScope,
    ) -> list[PatentRecord]:
        """Search a specific patent database.

        Args:
            db_name: Database name
            query: Search query
            scope: Search scope

        Returns:
            List of patent records
        """
        # This is a placeholder implementation
        # In production, this would call actual API clients:
        # - langchain_kipris_tools for KIPRIS
        # - patent_client for USPTO
        # - EPO OPS API for EPO
        # - google-patents-scraper for Google Patents

        self._logger.info("executing_search", database=db_name)

        # Simulated results for now
        # TODO: Integrate actual API clients
        results: list[PatentRecord] = []

        # Return empty results - actual implementation would query APIs
        return results

    def _deduplicate_patents(
        self,
        patents: list[PatentRecord],
    ) -> list[PatentRecord]:
        """Remove duplicate patents from results.

        Deduplication is based on:
        1. Patent number (primary)
        2. Title + filing date (secondary)

        Args:
            patents: List of patent records

        Returns:
            Deduplicated list
        """
        seen_numbers: set[str] = set()
        seen_title_date: set[str] = set()
        deduplicated: list[PatentRecord] = []

        for patent in patents:
            patent_num = patent.get("patent_number", "")
            title = patent.get("title", "")
            filing_date = patent.get("filing_date", "")

            # Primary check: patent number
            if patent_num:
                if patent_num in seen_numbers:
                    continue
                seen_numbers.add(patent_num)

            # Secondary check: title + date
            title_date_key = f"{title}|{filing_date}"
            if title_date_key in seen_title_date:
                continue
            seen_title_date.add(title_date_key)

            deduplicated.append(patent)

        return deduplicated

    def _normalize_patent(self, patent: PatentRecord) -> PatentRecord:
        """Normalize a patent record to standard format.

        Args:
            patent: Raw patent record

        Returns:
            Normalized patent record
        """
        # Ensure all required fields exist
        normalized = PatentRecord(
            patent_number=patent.get("patent_number", ""),
            title=patent.get("title", ""),
            applicant=patent.get("applicant", ""),
            inventor=patent.get("inventor", []),
            filing_date=self._normalize_date(patent.get("filing_date", "")),
            publication_date=self._normalize_date(patent.get("publication_date", "")),
            grant_date=self._normalize_date(patent.get("grant_date")),
            ipc_codes=patent.get("ipc_codes", []),
            claims=patent.get("claims", []),
            abstract=patent.get("abstract", ""),
            legal_status=patent.get("legal_status", "pending"),
            citation_count=patent.get("citation_count", 0),
            family_size=patent.get("family_size", 1),
        )

        return normalized

    def _normalize_date(self, date_str: str | None) -> str | None:
        """Normalize date to ISO-8601 format.

        Args:
            date_str: Date string in various formats

        Returns:
            ISO-8601 formatted date or None
        """
        if not date_str:
            return None

        # Try various formats
        formats = [
            "%Y-%m-%d",
            "%Y%m%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y.%m.%d",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Return as-is if no format matches
        return date_str
