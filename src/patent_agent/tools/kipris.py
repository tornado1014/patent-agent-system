"""KIPRIS (Korea Intellectual Property Rights Information Service) API wrapper.

KIPRIS provides access to Korean patent, utility model, design, and trademark data.
This module wraps the langchain_kipris_tools library for use in the patent agent system.

API Documentation: https://www.kipris.or.kr/khome/main.jsp
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Literal

import httpx
import structlog
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from patent_agent.config import get_settings

logger = structlog.get_logger(__name__)


class KIPRISSearchType(str, Enum):
    """KIPRIS search types."""

    KEYWORD = "keyword"
    APPLICANT = "applicant"
    INVENTOR = "inventor"
    APPLICATION_NUMBER = "application_number"
    REGISTRATION_NUMBER = "registration_number"
    IPC = "ipc"
    CPC = "cpc"


class PatentStatus(str, Enum):
    """Patent application status."""

    PENDING = "pending"
    REGISTERED = "registered"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    ABANDONED = "abandoned"


@dataclass
class KIPRISSearchResult:
    """Single search result from KIPRIS."""

    application_number: str
    title: str
    applicant: str
    filing_date: date | None = None
    registration_number: str | None = None
    registration_date: date | None = None
    status: PatentStatus = PatentStatus.PENDING
    ipc_codes: list[str] = field(default_factory=list)
    abstract: str = ""
    relevance_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "application_number": self.application_number,
            "title": self.title,
            "applicant": self.applicant,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "registration_number": self.registration_number,
            "registration_date": (
                self.registration_date.isoformat() if self.registration_date else None
            ),
            "status": self.status.value,
            "ipc_codes": self.ipc_codes,
            "abstract": self.abstract,
            "relevance_score": self.relevance_score,
        }


@dataclass
class KIPRISPatentDetail:
    """Detailed patent information from KIPRIS."""

    application_number: str
    title: str
    applicant: str
    inventor: str
    filing_date: date | None = None
    registration_number: str | None = None
    registration_date: date | None = None
    publication_number: str | None = None
    publication_date: date | None = None
    status: PatentStatus = PatentStatus.PENDING
    ipc_codes: list[str] = field(default_factory=list)
    cpc_codes: list[str] = field(default_factory=list)
    abstract: str = ""
    claims: list[str] = field(default_factory=list)
    description: str = ""
    drawings_count: int = 0
    priority_claims: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    family_members: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "application_number": self.application_number,
            "title": self.title,
            "applicant": self.applicant,
            "inventor": self.inventor,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "registration_number": self.registration_number,
            "registration_date": (
                self.registration_date.isoformat() if self.registration_date else None
            ),
            "publication_number": self.publication_number,
            "publication_date": (
                self.publication_date.isoformat() if self.publication_date else None
            ),
            "status": self.status.value,
            "ipc_codes": self.ipc_codes,
            "cpc_codes": self.cpc_codes,
            "abstract": self.abstract,
            "claims": self.claims,
            "description": self.description,
            "drawings_count": self.drawings_count,
            "priority_claims": self.priority_claims,
            "citations": self.citations,
            "family_members": self.family_members,
        }


class KIPRISSearchParams(BaseModel):
    """Parameters for KIPRIS search."""

    query: str = Field(..., description="Search query")
    search_type: KIPRISSearchType = Field(
        default=KIPRISSearchType.KEYWORD, description="Type of search"
    )
    patent_type: Literal["patent", "utility_model", "design", "all"] = Field(
        default="patent", description="Type of IP right to search"
    )
    date_from: date | None = Field(default=None, description="Start date for search")
    date_to: date | None = Field(default=None, description="End date for search")
    max_results: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    sort_by: Literal["relevance", "date_desc", "date_asc"] = Field(
        default="relevance", description="Sort order"
    )


class KIPRISClient:
    """Client for KIPRIS API.

    This client provides methods to search and retrieve Korean patent data.
    It wraps the langchain_kipris_tools library when available, with fallback
    to direct API calls.

    Example:
        >>> client = KIPRISClient()
        >>> results = await client.search_patents(
        ...     query="인공지능 자연어처리",
        ...     search_type=KIPRISSearchType.KEYWORD,
        ...     max_results=50
        ... )
        >>> for result in results:
        ...     print(f"{result.application_number}: {result.title}")
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize KIPRIS client.

        Args:
            api_key: KIPRIS API key. If not provided, reads from settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.kipris_api_key
        self.base_url = "http://plus.kipris.or.kr/openapi/rest"
        self._http_client: httpx.AsyncClient | None = None
        self._langchain_tools: Any = None

        # Try to import langchain_kipris_tools
        try:
            from langchain_kipris import (
                LangChainKiprisForeignTools,
                LangChainKiprisKoreanTools,
            )

            if self.api_key:
                self._langchain_tools = {
                    "korean": LangChainKiprisKoreanTools(api_key=self.api_key),
                    "foreign": LangChainKiprisForeignTools(api_key=self.api_key),
                }
                logger.info("langchain_kipris_tools loaded successfully")
        except ImportError:
            logger.warning(
                "langchain_kipris_tools not available, using direct API calls"
            )

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def search_patents(
        self,
        query: str,
        search_type: KIPRISSearchType = KIPRISSearchType.KEYWORD,
        patent_type: Literal["patent", "utility_model", "design", "all"] = "patent",
        date_from: date | None = None,
        date_to: date | None = None,
        max_results: int = 100,
        sort_by: Literal["relevance", "date_desc", "date_asc"] = "relevance",
    ) -> list[KIPRISSearchResult]:
        """Search for patents in KIPRIS.

        Args:
            query: Search query string
            search_type: Type of search to perform
            patent_type: Type of IP right to search
            date_from: Start date for search range
            date_to: End date for search range
            max_results: Maximum number of results to return
            sort_by: Sort order for results

        Returns:
            List of search results
        """
        logger.info(
            "Searching KIPRIS",
            query=query,
            search_type=search_type.value,
            patent_type=patent_type,
            max_results=max_results,
        )

        # Use langchain_kipris_tools if available
        if self._langchain_tools:
            return await self._search_with_langchain(
                query=query,
                search_type=search_type,
                patent_type=patent_type,
                date_from=date_from,
                date_to=date_to,
                max_results=max_results,
            )

        # Fallback to direct API call
        return await self._search_direct(
            query=query,
            search_type=search_type,
            patent_type=patent_type,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
            sort_by=sort_by,
        )

    async def _search_with_langchain(
        self,
        query: str,
        search_type: KIPRISSearchType,
        patent_type: str,
        date_from: date | None,
        date_to: date | None,
        max_results: int,
    ) -> list[KIPRISSearchResult]:
        """Search using langchain_kipris_tools."""
        results: list[KIPRISSearchResult] = []
        korean_tools = self._langchain_tools["korean"]

        # Map search type to appropriate tool
        tool_map = {
            KIPRISSearchType.KEYWORD: "search_by_keyword",
            KIPRISSearchType.APPLICANT: "search_by_applicant",
            KIPRISSearchType.APPLICATION_NUMBER: "search_by_application_number",
        }

        tool_name = tool_map.get(search_type, "search_by_keyword")

        # Get the appropriate tool
        tool = getattr(korean_tools, tool_name, None)
        if tool is None:
            logger.warning(f"Tool {tool_name} not found, using keyword search")
            tool = korean_tools.search_by_keyword

        # Execute search in thread pool (tools may be sync)
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(
            None,
            lambda: tool.invoke({"query": query, "num_results": max_results}),
        )

        # Parse results
        if isinstance(raw_results, list):
            for item in raw_results[:max_results]:
                result = self._parse_langchain_result(item)
                if result:
                    results.append(result)

        logger.info(f"Found {len(results)} results from langchain_kipris_tools")
        return results

    def _parse_langchain_result(self, item: dict[str, Any]) -> KIPRISSearchResult | None:
        """Parse a single result from langchain_kipris_tools."""
        try:
            filing_date = None
            if item.get("filing_date"):
                try:
                    filing_date = date.fromisoformat(item["filing_date"])
                except (ValueError, TypeError):
                    pass

            return KIPRISSearchResult(
                application_number=item.get("application_number", ""),
                title=item.get("title", ""),
                applicant=item.get("applicant", ""),
                filing_date=filing_date,
                registration_number=item.get("registration_number"),
                status=PatentStatus.REGISTERED
                if item.get("registration_number")
                else PatentStatus.PENDING,
                ipc_codes=item.get("ipc_codes", []),
                abstract=item.get("abstract", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to parse result: {e}")
            return None

    async def _search_direct(
        self,
        query: str,
        search_type: KIPRISSearchType,
        patent_type: str,
        date_from: date | None,
        date_to: date | None,
        max_results: int,
        sort_by: str,
    ) -> list[KIPRISSearchResult]:
        """Search using direct KIPRIS API calls."""
        if not self.api_key:
            logger.error("KIPRIS API key not configured")
            return []

        client = await self._get_http_client()

        # Build API endpoint based on search type
        endpoint_map = {
            KIPRISSearchType.KEYWORD: "/patUtiModInfoSearchSevice/freeSearchInfo",
            KIPRISSearchType.APPLICANT: "/patUtiModInfoSearchSevice/applicantSearchInfo",
            KIPRISSearchType.APPLICATION_NUMBER: "/patUtiModInfoSearchSevice/applicationNumberSearchInfo",
        }

        endpoint = endpoint_map.get(
            search_type, "/patUtiModInfoSearchSevice/freeSearchInfo"
        )

        # Build query parameters
        params = {
            "accessKey": self.api_key,
            "word": query,
            "numOfRows": str(max_results),
            "pageNo": "1",
        }

        if date_from:
            params["startDate"] = date_from.strftime("%Y%m%d")
        if date_to:
            params["endDate"] = date_to.strftime("%Y%m%d")

        try:
            response = await client.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()
            results = self._parse_api_response(data)
            logger.info(f"Found {len(results)} results from direct API")
            return results

        except httpx.HTTPError as e:
            logger.error(f"KIPRIS API error: {e}")
            return []

    def _parse_api_response(self, data: dict[str, Any]) -> list[KIPRISSearchResult]:
        """Parse KIPRIS API response."""
        results: list[KIPRISSearchResult] = []

        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        if not isinstance(items, list):
            items = [items] if items else []

        for item in items:
            try:
                filing_date = None
                if item.get("applicationDate"):
                    try:
                        filing_date = date(
                            int(item["applicationDate"][:4]),
                            int(item["applicationDate"][4:6]),
                            int(item["applicationDate"][6:8]),
                        )
                    except (ValueError, IndexError):
                        pass

                results.append(
                    KIPRISSearchResult(
                        application_number=item.get("applicationNumber", ""),
                        title=item.get("inventionTitle", ""),
                        applicant=item.get("applicantName", ""),
                        filing_date=filing_date,
                        registration_number=item.get("registerNumber"),
                        status=PatentStatus.REGISTERED
                        if item.get("registerNumber")
                        else PatentStatus.PENDING,
                        ipc_codes=[item.get("ipcNumber", "")]
                        if item.get("ipcNumber")
                        else [],
                        abstract=item.get("astrtCont", ""),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse API response item: {e}")
                continue

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def get_patent_detail(
        self, application_number: str
    ) -> KIPRISPatentDetail | None:
        """Get detailed patent information.

        Args:
            application_number: Korean patent application number

        Returns:
            Detailed patent information or None if not found
        """
        logger.info("Fetching patent detail", application_number=application_number)

        if self._langchain_tools:
            return await self._get_detail_with_langchain(application_number)

        return await self._get_detail_direct(application_number)

    async def _get_detail_with_langchain(
        self, application_number: str
    ) -> KIPRISPatentDetail | None:
        """Get patent detail using langchain_kipris_tools."""
        korean_tools = self._langchain_tools["korean"]

        loop = asyncio.get_event_loop()
        raw_result = await loop.run_in_executor(
            None,
            lambda: korean_tools.get_patent_detail.invoke(
                {"application_number": application_number}
            ),
        )

        if not raw_result:
            return None

        return self._parse_detail_result(raw_result)

    def _parse_detail_result(self, item: dict[str, Any]) -> KIPRISPatentDetail | None:
        """Parse detailed patent result."""
        try:
            return KIPRISPatentDetail(
                application_number=item.get("application_number", ""),
                title=item.get("title", ""),
                applicant=item.get("applicant", ""),
                inventor=item.get("inventor", ""),
                filing_date=self._parse_date(item.get("filing_date")),
                registration_number=item.get("registration_number"),
                registration_date=self._parse_date(item.get("registration_date")),
                publication_number=item.get("publication_number"),
                publication_date=self._parse_date(item.get("publication_date")),
                status=PatentStatus.REGISTERED
                if item.get("registration_number")
                else PatentStatus.PENDING,
                ipc_codes=item.get("ipc_codes", []),
                cpc_codes=item.get("cpc_codes", []),
                abstract=item.get("abstract", ""),
                claims=item.get("claims", []),
                description=item.get("description", ""),
                drawings_count=item.get("drawings_count", 0),
                priority_claims=item.get("priority_claims", []),
                citations=item.get("citations", []),
                family_members=item.get("family_members", []),
            )
        except Exception as e:
            logger.warning(f"Failed to parse detail result: {e}")
            return None

    async def _get_detail_direct(
        self, application_number: str
    ) -> KIPRISPatentDetail | None:
        """Get patent detail using direct API calls."""
        if not self.api_key:
            logger.error("KIPRIS API key not configured")
            return None

        client = await self._get_http_client()

        endpoint = "/patUtiModInfoSearchSevice/applicationNumberSearchInfo"
        params = {
            "accessKey": self.api_key,
            "applicationNumber": application_number,
        }

        try:
            response = await client.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()

            data = response.json()
            items = (
                data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            )

            if not items:
                return None

            item = items[0] if isinstance(items, list) else items
            return self._parse_detail_result(item)

        except httpx.HTTPError as e:
            logger.error(f"KIPRIS API error: {e}")
            return None

    def _parse_date(self, date_str: str | None) -> date | None:
        """Parse date string to date object."""
        if not date_str:
            return None
        try:
            if len(date_str) == 8:
                return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            return date.fromisoformat(date_str)
        except (ValueError, TypeError):
            return None

    async def search_by_ipc(
        self,
        ipc_code: str,
        max_results: int = 100,
    ) -> list[KIPRISSearchResult]:
        """Search patents by IPC classification code.

        Args:
            ipc_code: IPC classification code (e.g., "G06F")
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        return await self.search_patents(
            query=ipc_code,
            search_type=KIPRISSearchType.IPC,
            max_results=max_results,
        )

    async def get_patent_family(
        self, application_number: str
    ) -> list[KIPRISSearchResult]:
        """Get patent family members.

        Args:
            application_number: Korean patent application number

        Returns:
            List of family members
        """
        detail = await self.get_patent_detail(application_number)
        if not detail or not detail.family_members:
            return []

        # Search for each family member
        results: list[KIPRISSearchResult] = []
        for family_num in detail.family_members[:10]:  # Limit to 10
            family_detail = await self.get_patent_detail(family_num)
            if family_detail:
                results.append(
                    KIPRISSearchResult(
                        application_number=family_detail.application_number,
                        title=family_detail.title,
                        applicant=family_detail.applicant,
                        filing_date=family_detail.filing_date,
                        registration_number=family_detail.registration_number,
                        status=family_detail.status,
                        ipc_codes=family_detail.ipc_codes,
                        abstract=family_detail.abstract,
                    )
                )

        return results


# LangChain Tool wrappers for agent use
def create_kipris_tools(api_key: str | None = None) -> list[Any]:
    """Create LangChain tools for KIPRIS.

    Returns a list of tools that can be used by LangChain agents.
    """
    from langchain_core.tools import StructuredTool

    client = KIPRISClient(api_key=api_key)

    async def search_korean_patents(
        query: str,
        search_type: str = "keyword",
        max_results: int = 50,
    ) -> str:
        """Search Korean patents in KIPRIS database.

        Args:
            query: Search query (keywords, applicant name, or application number)
            search_type: Type of search - 'keyword', 'applicant', or 'application_number'
            max_results: Maximum number of results to return

        Returns:
            JSON string with search results
        """
        import json

        search_type_enum = KIPRISSearchType(search_type)
        results = await client.search_patents(
            query=query,
            search_type=search_type_enum,
            max_results=max_results,
        )
        return json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2)

    async def get_korean_patent_detail(application_number: str) -> str:
        """Get detailed information about a Korean patent.

        Args:
            application_number: Korean patent application number

        Returns:
            JSON string with patent details
        """
        import json

        detail = await client.get_patent_detail(application_number)
        if detail:
            return json.dumps(detail.to_dict(), ensure_ascii=False, indent=2)
        return json.dumps({"error": "Patent not found"})

    return [
        StructuredTool.from_function(
            coroutine=search_korean_patents,
            name="search_korean_patents",
            description="Search for Korean patents in KIPRIS database by keywords, applicant, or application number",
        ),
        StructuredTool.from_function(
            coroutine=get_korean_patent_detail,
            name="get_korean_patent_detail",
            description="Get detailed information about a specific Korean patent by application number",
        ),
    ]
