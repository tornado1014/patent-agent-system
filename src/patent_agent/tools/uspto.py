"""USPTO (United States Patent and Trademark Office) API wrapper.

This module provides access to US patent data using the patent_client library.
It supports searching and retrieving patents, applications, and related data.

Reference: https://github.com/parkerhancock/patent_client
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


class USPTOSearchType(str, Enum):
    """USPTO search types."""

    KEYWORD = "keyword"
    APPLICANT = "applicant"
    INVENTOR = "inventor"
    APPLICATION_NUMBER = "application_number"
    PATENT_NUMBER = "patent_number"
    CPC = "cpc"
    USPC = "uspc"


class USPatentStatus(str, Enum):
    """US patent application status."""

    PENDING = "pending"
    PATENTED = "patented"
    ABANDONED = "abandoned"
    EXPIRED = "expired"


@dataclass
class USPTOSearchResult:
    """Single search result from USPTO."""

    application_number: str
    title: str
    applicant: str
    filing_date: date | None = None
    patent_number: str | None = None
    grant_date: date | None = None
    status: USPatentStatus = USPatentStatus.PENDING
    cpc_codes: list[str] = field(default_factory=list)
    abstract: str = ""
    relevance_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "application_number": self.application_number,
            "title": self.title,
            "applicant": self.applicant,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "patent_number": self.patent_number,
            "grant_date": self.grant_date.isoformat() if self.grant_date else None,
            "status": self.status.value,
            "cpc_codes": self.cpc_codes,
            "abstract": self.abstract,
            "relevance_score": self.relevance_score,
        }


@dataclass
class USPTOPatentDetail:
    """Detailed patent information from USPTO."""

    application_number: str
    title: str
    applicant: str
    inventor: str
    filing_date: date | None = None
    patent_number: str | None = None
    grant_date: date | None = None
    publication_number: str | None = None
    publication_date: date | None = None
    status: USPatentStatus = USPatentStatus.PENDING
    cpc_codes: list[str] = field(default_factory=list)
    uspc_codes: list[str] = field(default_factory=list)
    abstract: str = ""
    claims: list[str] = field(default_factory=list)
    description: str = ""
    drawings_count: int = 0
    priority_claims: list[dict[str, Any]] = field(default_factory=list)
    cited_references: list[str] = field(default_factory=list)
    citing_patents: list[str] = field(default_factory=list)
    continuity_data: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "application_number": self.application_number,
            "title": self.title,
            "applicant": self.applicant,
            "inventor": self.inventor,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "patent_number": self.patent_number,
            "grant_date": self.grant_date.isoformat() if self.grant_date else None,
            "publication_number": self.publication_number,
            "publication_date": (
                self.publication_date.isoformat() if self.publication_date else None
            ),
            "status": self.status.value,
            "cpc_codes": self.cpc_codes,
            "uspc_codes": self.uspc_codes,
            "abstract": self.abstract,
            "claims": self.claims,
            "description": self.description,
            "drawings_count": self.drawings_count,
            "priority_claims": self.priority_claims,
            "cited_references": self.cited_references,
            "citing_patents": self.citing_patents,
            "continuity_data": self.continuity_data,
            "assignments": self.assignments,
        }


class USPTOClient:
    """Client for USPTO patent data.

    This client provides methods to search and retrieve US patent data
    using the patent_client library.

    Example:
        >>> client = USPTOClient()
        >>> results = await client.search_patents(
        ...     query="machine learning natural language",
        ...     search_type=USPTOSearchType.KEYWORD,
        ...     max_results=50
        ... )
        >>> for result in results:
        ...     print(f"{result.patent_number}: {result.title}")
    """

    def __init__(self) -> None:
        """Initialize USPTO client."""
        self._patent_client_available = False

        # Try to import patent_client
        try:
            from patent_client import Patent, USApplication

            self._Patent = Patent
            self._USApplication = USApplication
            self._patent_client_available = True
            logger.info("patent_client loaded successfully")
        except ImportError:
            logger.warning("patent_client not available")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def search_patents(
        self,
        query: str,
        search_type: USPTOSearchType = USPTOSearchType.KEYWORD,
        date_from: date | None = None,
        date_to: date | None = None,
        max_results: int = 100,
    ) -> list[USPTOSearchResult]:
        """Search for patents in USPTO.

        Args:
            query: Search query string
            search_type: Type of search to perform
            date_from: Start date for search range
            date_to: End date for search range
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        logger.info(
            "Searching USPTO",
            query=query,
            search_type=search_type.value,
            max_results=max_results,
        )

        if not self._patent_client_available:
            logger.error("patent_client not available")
            return []

        loop = asyncio.get_event_loop()

        if search_type == USPTOSearchType.PATENT_NUMBER:
            return await loop.run_in_executor(
                None, self._search_by_patent_number, query, max_results
            )
        elif search_type == USPTOSearchType.APPLICATION_NUMBER:
            return await loop.run_in_executor(
                None, self._search_by_application_number, query, max_results
            )
        else:
            return await loop.run_in_executor(
                None,
                self._search_by_keyword,
                query,
                search_type,
                date_from,
                date_to,
                max_results,
            )

    def _search_by_patent_number(
        self, patent_number: str, max_results: int
    ) -> list[USPTOSearchResult]:
        """Search by patent number."""
        results: list[USPTOSearchResult] = []

        try:
            patent = self._Patent.objects.get(patent_number)
            if patent:
                results.append(self._patent_to_result(patent))
        except Exception as e:
            logger.warning(f"Patent not found: {patent_number}, error: {e}")

        return results

    def _search_by_application_number(
        self, app_number: str, max_results: int
    ) -> list[USPTOSearchResult]:
        """Search by application number."""
        results: list[USPTOSearchResult] = []

        try:
            app = self._USApplication.objects.get(app_number)
            if app:
                results.append(self._application_to_result(app))
        except Exception as e:
            logger.warning(f"Application not found: {app_number}, error: {e}")

        return results

    def _search_by_keyword(
        self,
        query: str,
        search_type: USPTOSearchType,
        date_from: date | None,
        date_to: date | None,
        max_results: int,
    ) -> list[USPTOSearchResult]:
        """Search by keyword or other criteria."""
        results: list[USPTOSearchResult] = []

        try:
            # Build search query based on type
            if search_type == USPTOSearchType.CPC:
                patents = self._Patent.objects.filter(cpc_class=query)
            elif search_type == USPTOSearchType.APPLICANT:
                patents = self._USApplication.objects.filter(first_named_applicant=query)
            elif search_type == USPTOSearchType.INVENTOR:
                patents = self._USApplication.objects.filter(inventors=query)
            else:
                # Default keyword search
                patents = self._Patent.objects.filter(title=query)

            # Apply date filters if provided
            # Note: patent_client filtering varies by data source

            count = 0
            for patent in patents:
                if count >= max_results:
                    break
                try:
                    results.append(self._patent_to_result(patent))
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse patent: {e}")
                    continue

        except Exception as e:
            logger.error(f"USPTO search error: {e}")

        logger.info(f"Found {len(results)} results from USPTO")
        return results

    def _patent_to_result(self, patent: Any) -> USPTOSearchResult:
        """Convert patent_client Patent to USPTOSearchResult."""
        grant_date = None
        if hasattr(patent, "grant_date") and patent.grant_date:
            grant_date = patent.grant_date

        filing_date = None
        if hasattr(patent, "filing_date") and patent.filing_date:
            filing_date = patent.filing_date

        return USPTOSearchResult(
            application_number=getattr(patent, "appl_id", ""),
            title=getattr(patent, "title", ""),
            applicant=getattr(patent, "applicant", ""),
            filing_date=filing_date,
            patent_number=getattr(patent, "patent_number", None),
            grant_date=grant_date,
            status=USPatentStatus.PATENTED
            if getattr(patent, "patent_number", None)
            else USPatentStatus.PENDING,
            cpc_codes=getattr(patent, "cpc_classes", []) or [],
            abstract=getattr(patent, "abstract", "") or "",
        )

    def _application_to_result(self, app: Any) -> USPTOSearchResult:
        """Convert patent_client USApplication to USPTOSearchResult."""
        filing_date = None
        if hasattr(app, "filing_date") and app.filing_date:
            filing_date = app.filing_date

        return USPTOSearchResult(
            application_number=getattr(app, "appl_id", ""),
            title=getattr(app, "app_filing_name", "") or getattr(app, "title", ""),
            applicant=getattr(app, "first_named_applicant", ""),
            filing_date=filing_date,
            patent_number=getattr(app, "patent_number", None),
            status=self._parse_status(getattr(app, "app_status", "")),
            cpc_codes=[],
            abstract="",
        )

    def _parse_status(self, status_str: str) -> USPatentStatus:
        """Parse status string to enum."""
        status_lower = status_str.lower()
        if "patent" in status_lower or "issued" in status_lower:
            return USPatentStatus.PATENTED
        elif "abandon" in status_lower:
            return USPatentStatus.ABANDONED
        elif "expired" in status_lower:
            return USPatentStatus.EXPIRED
        return USPatentStatus.PENDING

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def get_patent_detail(
        self, identifier: str, by_patent_number: bool = True
    ) -> USPTOPatentDetail | None:
        """Get detailed patent information.

        Args:
            identifier: Patent number or application number
            by_patent_number: If True, search by patent number; else by application number

        Returns:
            Detailed patent information or None if not found
        """
        logger.info(
            "Fetching USPTO patent detail",
            identifier=identifier,
            by_patent_number=by_patent_number,
        )

        if not self._patent_client_available:
            logger.error("patent_client not available")
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_detail_sync, identifier, by_patent_number
        )

    def _get_detail_sync(
        self, identifier: str, by_patent_number: bool
    ) -> USPTOPatentDetail | None:
        """Synchronous patent detail retrieval."""
        try:
            if by_patent_number:
                patent = self._Patent.objects.get(identifier)
                return self._patent_to_detail(patent)
            else:
                app = self._USApplication.objects.get(identifier)
                return self._application_to_detail(app)
        except Exception as e:
            logger.error(f"Failed to get patent detail: {e}")
            return None

    def _patent_to_detail(self, patent: Any) -> USPTOPatentDetail:
        """Convert patent_client Patent to USPTOPatentDetail."""
        claims = []
        if hasattr(patent, "claims"):
            try:
                claims = [str(c) for c in patent.claims]
            except Exception:
                pass

        return USPTOPatentDetail(
            application_number=getattr(patent, "appl_id", ""),
            title=getattr(patent, "title", ""),
            applicant=getattr(patent, "applicant", ""),
            inventor=", ".join(getattr(patent, "inventors", []) or []),
            filing_date=getattr(patent, "filing_date", None),
            patent_number=getattr(patent, "patent_number", None),
            grant_date=getattr(patent, "grant_date", None),
            publication_number=getattr(patent, "publication_number", None),
            publication_date=getattr(patent, "publication_date", None),
            status=USPatentStatus.PATENTED,
            cpc_codes=getattr(patent, "cpc_classes", []) or [],
            uspc_codes=getattr(patent, "uspc_classes", []) or [],
            abstract=getattr(patent, "abstract", "") or "",
            claims=claims,
            description=getattr(patent, "description", "") or "",
            drawings_count=getattr(patent, "num_drawings", 0) or 0,
            cited_references=getattr(patent, "us_references", []) or [],
        )

    def _application_to_detail(self, app: Any) -> USPTOPatentDetail:
        """Convert patent_client USApplication to USPTOPatentDetail."""
        return USPTOPatentDetail(
            application_number=getattr(app, "appl_id", ""),
            title=getattr(app, "app_filing_name", "") or getattr(app, "title", ""),
            applicant=getattr(app, "first_named_applicant", ""),
            inventor=", ".join(getattr(app, "inventors", []) or []),
            filing_date=getattr(app, "filing_date", None),
            patent_number=getattr(app, "patent_number", None),
            grant_date=getattr(app, "patent_issue_date", None),
            status=self._parse_status(getattr(app, "app_status", "")),
            cpc_codes=[],
            uspc_codes=[],
            abstract="",
            claims=[],
            description="",
        )

    async def get_application_status(self, application_number: str) -> dict[str, Any]:
        """Get application prosecution history/status.

        Args:
            application_number: US application number

        Returns:
            Dictionary with application status and history
        """
        if not self._patent_client_available:
            return {"error": "patent_client not available"}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_application_status_sync, application_number
        )

    def _get_application_status_sync(self, application_number: str) -> dict[str, Any]:
        """Synchronous application status retrieval."""
        try:
            app = self._USApplication.objects.get(application_number)

            return {
                "application_number": getattr(app, "appl_id", ""),
                "status": getattr(app, "app_status", ""),
                "status_date": str(getattr(app, "app_status_date", "")),
                "filing_date": str(getattr(app, "filing_date", "")),
                "patent_number": getattr(app, "patent_number", None),
                "patent_issue_date": str(getattr(app, "patent_issue_date", "")),
                "examiner": getattr(app, "app_examiner_name", ""),
                "group_art_unit": getattr(app, "app_grp_art_unit", ""),
            }
        except Exception as e:
            logger.error(f"Failed to get application status: {e}")
            return {"error": str(e)}

    async def search_by_cpc(
        self, cpc_code: str, max_results: int = 100
    ) -> list[USPTOSearchResult]:
        """Search patents by CPC classification.

        Args:
            cpc_code: CPC classification code (e.g., "G06N")
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        return await self.search_patents(
            query=cpc_code,
            search_type=USPTOSearchType.CPC,
            max_results=max_results,
        )

    async def get_cited_by(self, patent_number: str) -> list[USPTOSearchResult]:
        """Get patents that cite the given patent.

        Args:
            patent_number: US patent number

        Returns:
            List of citing patents
        """
        detail = await self.get_patent_detail(patent_number, by_patent_number=True)
        if not detail or not detail.citing_patents:
            return []

        results: list[USPTOSearchResult] = []
        for citing_num in detail.citing_patents[:20]:  # Limit to 20
            citing_detail = await self.get_patent_detail(
                citing_num, by_patent_number=True
            )
            if citing_detail:
                results.append(
                    USPTOSearchResult(
                        application_number=citing_detail.application_number,
                        title=citing_detail.title,
                        applicant=citing_detail.applicant,
                        filing_date=citing_detail.filing_date,
                        patent_number=citing_detail.patent_number,
                        grant_date=citing_detail.grant_date,
                        status=citing_detail.status,
                        cpc_codes=citing_detail.cpc_codes,
                        abstract=citing_detail.abstract,
                    )
                )

        return results


# LangChain Tool wrappers for agent use
def create_uspto_tools() -> list[Any]:
    """Create LangChain tools for USPTO.

    Returns a list of tools that can be used by LangChain agents.
    """
    from langchain_core.tools import StructuredTool

    client = USPTOClient()

    async def search_us_patents(
        query: str,
        search_type: str = "keyword",
        max_results: int = 50,
    ) -> str:
        """Search US patents in USPTO database.

        Args:
            query: Search query (keywords, patent number, or application number)
            search_type: Type of search - 'keyword', 'patent_number', 'application_number', or 'cpc'
            max_results: Maximum number of results to return

        Returns:
            JSON string with search results
        """
        import json

        search_type_enum = USPTOSearchType(search_type)
        results = await client.search_patents(
            query=query,
            search_type=search_type_enum,
            max_results=max_results,
        )
        return json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2)

    async def get_us_patent_detail(identifier: str, by_patent_number: bool = True) -> str:
        """Get detailed information about a US patent.

        Args:
            identifier: US patent number or application number
            by_patent_number: If True, search by patent number; else by application number

        Returns:
            JSON string with patent details
        """
        import json

        detail = await client.get_patent_detail(identifier, by_patent_number)
        if detail:
            return json.dumps(detail.to_dict(), ensure_ascii=False, indent=2)
        return json.dumps({"error": "Patent not found"})

    async def get_us_application_status(application_number: str) -> str:
        """Get prosecution status of a US patent application.

        Args:
            application_number: US application number

        Returns:
            JSON string with application status
        """
        import json

        status = await client.get_application_status(application_number)
        return json.dumps(status, ensure_ascii=False, indent=2)

    return [
        StructuredTool.from_function(
            coroutine=search_us_patents,
            name="search_us_patents",
            description="Search for US patents in USPTO database by keywords, patent number, application number, or CPC code",
        ),
        StructuredTool.from_function(
            coroutine=get_us_patent_detail,
            name="get_us_patent_detail",
            description="Get detailed information about a specific US patent",
        ),
        StructuredTool.from_function(
            coroutine=get_us_application_status,
            name="get_us_application_status",
            description="Get prosecution status and history of a US patent application",
        ),
    ]
