"""Multi-Database Searcher Agent for prior art search.

Executes parallel searches across multiple patent databases:
- KIPRIS (Korean patents)
- USPTO (US patents)
- Google Patents (global)
"""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from patent_agent.agents.prior_art.base import BasePriorArtAgent
from patent_agent.state.prior_art import (
    DatabaseID,
    PatentResult,
    PriorArtSearchState,
    PriorArtStep,
    SearchQuery,
)
from patent_agent.tools.kipris import KIPRISClient, KIPRISSearchType
from patent_agent.tools.uspto import USPTOClient, USPTOSearchType

logger = structlog.get_logger(__name__)


class MultiDBSearcherAgent(BasePriorArtAgent):
    """Agent for executing parallel searches across multiple patent databases.

    Coordinates searches across:
    - KIPRIS (한국특허정보원)
    - USPTO (미국특허청)
    - Google Patents (전세계)

    Features:
    - Parallel search execution
    - Error handling per database
    - Result normalization
    """

    def __init__(self, **kwargs: Any):
        """Initialize with database clients."""
        super().__init__(**kwargs)
        self._kipris_client: KIPRISClient | None = None
        self._uspto_client: USPTOClient | None = None

    @property
    def name(self) -> str:
        return "MultiDBSearcher"

    @property
    def description(self) -> str:
        return "다중 특허 데이터베이스 병렬 검색"

    @property
    def step(self) -> PriorArtStep:
        return "search"

    def get_system_prompt(self) -> str:
        return """당신은 특허 데이터베이스 검색 전문가입니다.
다중 데이터베이스에서 병렬로 선행기술을 검색합니다.

## 지원 데이터베이스
1. KIPRIS (한국특허정보원): 한국 특허/실용신안
2. USPTO (미국특허청): 미국 특허/출원
3. Google Patents: 전세계 특허

## 검색 원칙
- 모든 데이터베이스 병렬 검색
- 오류 발생 시 해당 DB만 실패 처리
- 결과 표준 형식으로 통일
"""

    async def process(self, state: PriorArtSearchState) -> dict[str, Any]:
        """Execute parallel searches across databases.

        Args:
            state: Current workflow state with search queries

        Returns:
            State updates with search results
        """
        search_queries = state.get("search_queries", {})

        if not search_queries:
            return {
                "error_messages": ["검색 쿼리가 생성되지 않았습니다."],
                "is_error_state": True,
            }

        self._logger.info(
            "executing_parallel_search",
            databases=list(search_queries.keys()),
        )

        # Execute parallel searches
        raw_results: dict[DatabaseID, list[PatentResult]] = {}
        total_counts: dict[DatabaseID, int] = {}
        errors: dict[DatabaseID, str | None] = {}

        # Create search tasks
        tasks = []
        db_order = []

        for db_id, query in search_queries.items():
            db_order.append(db_id)
            if db_id == "KIPRIS":
                tasks.append(self._search_kipris(query))
            elif db_id == "USPTO":
                tasks.append(self._search_uspto(query))
            elif db_id == "Google_Patents":
                tasks.append(self._search_google_patents(query))
            else:
                tasks.append(self._empty_search(db_id))

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, db_id in enumerate(db_order):
            result = results[i]

            if isinstance(result, Exception):
                self._logger.warning(
                    "database_search_error",
                    database=db_id,
                    error=str(result),
                )
                raw_results[db_id] = []
                total_counts[db_id] = 0
                errors[db_id] = str(result)
            else:
                patents, count, error = result
                raw_results[db_id] = patents
                total_counts[db_id] = count
                errors[db_id] = error

        # Calculate totals
        total_found = sum(total_counts.values())
        successful_dbs = [db for db, err in errors.items() if err is None]

        self._logger.info(
            "search_completed",
            total_results=total_found,
            successful_databases=successful_dbs,
        )

        return {
            "raw_results": raw_results,
            "total_results_count": total_counts,
            "search_errors": errors,
            "current_step": "filter",
            "updated_at": datetime.now().isoformat(),
        }

    async def _search_kipris(
        self,
        query: SearchQuery,
    ) -> tuple[list[PatentResult], int, str | None]:
        """Search KIPRIS database.

        Args:
            query: Search query

        Returns:
            Tuple of (results, count, error)
        """
        try:
            if self._kipris_client is None:
                self._kipris_client = KIPRISClient()

            # Execute search
            results = await self._kipris_client.search_patents(
                query=query["query_string"],
                search_type=KIPRISSearchType.KEYWORD,
                max_results=100,
            )

            # Convert to PatentResult format
            patents = []
            for r in results:
                patent = PatentResult(
                    database="KIPRIS",
                    document_number=r.application_number,
                    title=r.title,
                    applicant=r.applicant,
                    inventor=[],
                    filing_date=r.filing_date.isoformat() if r.filing_date else "",
                    publication_date="",
                    grant_date=r.registration_date.isoformat() if r.registration_date else None,
                    ipc_codes=r.ipc_codes,
                    abstract=r.abstract,
                    claims_text=None,
                    pdf_url=None,
                    family_id=None,
                )
                patents.append(patent)

            return patents, len(patents), None

        except Exception as e:
            self._logger.error("kipris_search_error", error=str(e))
            return [], 0, str(e)

    async def _search_uspto(
        self,
        query: SearchQuery,
    ) -> tuple[list[PatentResult], int, str | None]:
        """Search USPTO database.

        Args:
            query: Search query

        Returns:
            Tuple of (results, count, error)
        """
        try:
            if self._uspto_client is None:
                self._uspto_client = USPTOClient()

            # Extract keywords for search
            keywords = query.get("keywords", [])
            search_term = " ".join(keywords[:5]) if keywords else query["query_string"]

            # Execute search
            results = await self._uspto_client.search_patents(
                query=search_term,
                search_type=USPTOSearchType.KEYWORD,
                max_results=100,
            )

            # Convert to PatentResult format
            patents = []
            for r in results:
                patent = PatentResult(
                    database="USPTO",
                    document_number=r.patent_number or r.application_number,
                    title=r.title,
                    applicant=r.applicant,
                    inventor=[],
                    filing_date=r.filing_date.isoformat() if r.filing_date else "",
                    publication_date="",
                    grant_date=r.grant_date.isoformat() if r.grant_date else None,
                    ipc_codes=[],
                    abstract=r.abstract,
                    claims_text=None,
                    pdf_url=None,
                    family_id=None,
                )
                patents.append(patent)

            return patents, len(patents), None

        except Exception as e:
            self._logger.error("uspto_search_error", error=str(e))
            return [], 0, str(e)

    async def _search_google_patents(
        self,
        query: SearchQuery,
    ) -> tuple[list[PatentResult], int, str | None]:
        """Search Google Patents.

        Note: This is a placeholder. In production, would use
        google_patents_scraper or API.

        Args:
            query: Search query

        Returns:
            Tuple of (results, count, error)
        """
        # Google Patents doesn't have an official API
        # Return empty for now - would need web scraping
        self._logger.info(
            "google_patents_search_skipped",
            reason="No official API available",
        )
        return [], 0, "Google Patents 검색은 현재 지원되지 않습니다."

    async def _empty_search(
        self,
        db_id: str,
    ) -> tuple[list[PatentResult], int, str | None]:
        """Return empty results for unsupported database.

        Args:
            db_id: Database identifier

        Returns:
            Empty results with error message
        """
        return [], 0, f"지원되지 않는 데이터베이스: {db_id}"


class SearchResultNormalizer:
    """Utility class for normalizing search results across databases."""

    @staticmethod
    def normalize_date(date_str: str | None) -> str:
        """Normalize date string to ISO format.

        Args:
            date_str: Date string in various formats

        Returns:
            Normalized ISO date string or empty string
        """
        if not date_str:
            return ""

        # Try common formats
        from datetime import datetime

        formats = [
            "%Y-%m-%d",
            "%Y%m%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m/%d/%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return date_str

    @staticmethod
    def normalize_document_number(doc_num: str, database: str) -> str:
        """Normalize document number format.

        Args:
            doc_num: Original document number
            database: Source database

        Returns:
            Normalized document number
        """
        if not doc_num:
            return ""

        # Remove common prefixes
        doc_num = doc_num.strip()
        doc_num = doc_num.replace(" ", "")

        if database == "KIPRIS":
            # Korean format: KR10-2020-0123456
            if not doc_num.startswith("KR"):
                doc_num = f"KR{doc_num}"
        elif database == "USPTO":
            # US format: US12345678 or US2020/0123456
            if not doc_num.startswith("US"):
                doc_num = f"US{doc_num}"

        return doc_num

    @staticmethod
    def deduplicate_results(
        results: list[PatentResult],
    ) -> list[PatentResult]:
        """Remove duplicate results based on document number.

        Args:
            results: List of patent results

        Returns:
            Deduplicated list
        """
        seen: set[str] = set()
        unique: list[PatentResult] = []

        for result in results:
            doc_num = result.get("document_number", "")
            if doc_num and doc_num not in seen:
                seen.add(doc_num)
                unique.append(result)

        return unique
