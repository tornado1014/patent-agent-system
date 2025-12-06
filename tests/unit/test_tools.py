"""Tests for patent agent tools."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from patent_agent.tools.document_parser import (
    Claim,
    DocumentParser,
    DocumentSection,
    DocumentType,
    Drawing,
    ParsedDocument,
    SectionType,
)
from patent_agent.tools.embeddings import (
    EmbeddingModel,
    EmbeddingResult,
    PatentEmbeddings,
)
from patent_agent.tools.kipris import (
    KIPRISClient,
    KIPRISPatentDetail,
    KIPRISSearchResult,
    KIPRISSearchType,
    PatentStatus,
)
from patent_agent.tools.uspto import (
    USPTOClient,
    USPTOPatentDetail,
    USPTOSearchResult,
    USPTOSearchType,
    USPatentStatus,
)


class TestKIPRISSearchResult:
    """Tests for KIPRISSearchResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = KIPRISSearchResult(
            application_number="10-2024-0001234",
            title="인공지능 기반 자연어 처리 장치",
            applicant="테스트 주식회사",
            filing_date=date(2024, 1, 15),
            registration_number="10-1234567",
            status=PatentStatus.REGISTERED,
            ipc_codes=["G06F", "G06N"],
            abstract="본 발명은 인공지능을 이용한...",
        )

        result_dict = result.to_dict()

        assert result_dict["application_number"] == "10-2024-0001234"
        assert result_dict["title"] == "인공지능 기반 자연어 처리 장치"
        assert result_dict["filing_date"] == "2024-01-15"
        assert result_dict["status"] == "registered"
        assert len(result_dict["ipc_codes"]) == 2


class TestKIPRISClient:
    """Tests for KIPRIS client."""

    @pytest.fixture
    def client(self) -> KIPRISClient:
        """Create KIPRIS client for testing."""
        return KIPRISClient(api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_search_patents_empty_results(self, client: KIPRISClient) -> None:
        """Test search with no results."""
        # Mock HTTP response
        with patch.object(client, "_search_direct", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            results = await client.search_patents(
                query="nonexistent query",
                search_type=KIPRISSearchType.KEYWORD,
                max_results=10,
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_search_patents_with_results(self, client: KIPRISClient) -> None:
        """Test search with results."""
        mock_results = [
            KIPRISSearchResult(
                application_number="10-2024-0001234",
                title="테스트 발명",
                applicant="테스트 회사",
                status=PatentStatus.PENDING,
            ),
        ]

        with patch.object(client, "_search_direct", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results

            results = await client.search_patents(
                query="인공지능",
                search_type=KIPRISSearchType.KEYWORD,
                max_results=10,
            )

            assert len(results) == 1
            assert results[0].application_number == "10-2024-0001234"

    def test_parse_date_valid(self, client: KIPRISClient) -> None:
        """Test date parsing with valid date."""
        result = client._parse_date("20240115")
        assert result == date(2024, 1, 15)

    def test_parse_date_invalid(self, client: KIPRISClient) -> None:
        """Test date parsing with invalid date."""
        result = client._parse_date("invalid")
        assert result is None

    def test_parse_date_none(self, client: KIPRISClient) -> None:
        """Test date parsing with None."""
        result = client._parse_date(None)
        assert result is None


class TestUSPTOSearchResult:
    """Tests for USPTOSearchResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = USPTOSearchResult(
            application_number="17/123,456",
            title="Machine Learning System",
            applicant="Test Corp",
            filing_date=date(2024, 1, 15),
            patent_number="US 12,345,678",
            grant_date=date(2024, 6, 15),
            status=USPatentStatus.PATENTED,
            cpc_codes=["G06N", "G06F"],
            abstract="A system for machine learning...",
        )

        result_dict = result.to_dict()

        assert result_dict["application_number"] == "17/123,456"
        assert result_dict["patent_number"] == "US 12,345,678"
        assert result_dict["status"] == "patented"


class TestUSPTOClient:
    """Tests for USPTO client."""

    @pytest.fixture
    def client(self) -> USPTOClient:
        """Create USPTO client for testing."""
        return USPTOClient()

    def test_parse_status_patented(self, client: USPTOClient) -> None:
        """Test status parsing for patented."""
        assert client._parse_status("Patent Issued") == USPatentStatus.PATENTED
        assert client._parse_status("Issued") == USPatentStatus.PATENTED

    def test_parse_status_abandoned(self, client: USPTOClient) -> None:
        """Test status parsing for abandoned."""
        assert client._parse_status("Abandoned") == USPatentStatus.ABANDONED

    def test_parse_status_pending(self, client: USPTOClient) -> None:
        """Test status parsing for pending."""
        assert client._parse_status("Pending") == USPatentStatus.PENDING
        assert client._parse_status("Unknown Status") == USPatentStatus.PENDING


class TestDocumentSection:
    """Tests for DocumentSection dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        section = DocumentSection(
            section_type=SectionType.CLAIMS,
            title="특허청구범위",
            content="청구항 1. 방법에 있어서...",
            page_start=10,
            page_end=15,
        )

        section_dict = section.to_dict()

        assert section_dict["section_type"] == "claims"
        assert section_dict["title"] == "특허청구범위"
        assert section_dict["page_start"] == 10


class TestClaim:
    """Tests for Claim dataclass."""

    def test_to_dict_independent(self) -> None:
        """Test independent claim to dict."""
        claim = Claim(
            number=1,
            text="인공지능을 이용한 데이터 처리 방법에 있어서...",
            is_independent=True,
            claim_type="method",
        )

        claim_dict = claim.to_dict()

        assert claim_dict["number"] == 1
        assert claim_dict["is_independent"] is True
        assert claim_dict["depends_on"] == []

    def test_to_dict_dependent(self) -> None:
        """Test dependent claim to dict."""
        claim = Claim(
            number=2,
            text="제1항에 있어서, 상기 데이터는...",
            is_independent=False,
            depends_on=[1],
            claim_type="method",
        )

        claim_dict = claim.to_dict()

        assert claim_dict["number"] == 2
        assert claim_dict["is_independent"] is False
        assert claim_dict["depends_on"] == [1]


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    @pytest.fixture
    def sample_document(self) -> ParsedDocument:
        """Create sample parsed document."""
        return ParsedDocument(
            file_path="/path/to/patent.pdf",
            document_type=DocumentType.PDF,
            title="인공지능 기반 자연어 처리 장치",
            abstract="본 발명은 인공지능을 이용한...",
            sections=[
                DocumentSection(
                    section_type=SectionType.CLAIMS,
                    title="특허청구범위",
                    content="청구항 1...",
                ),
                DocumentSection(
                    section_type=SectionType.ABSTRACT,
                    title="요약",
                    content="본 발명은...",
                ),
            ],
            claims=[
                Claim(number=1, text="독립항 내용", is_independent=True),
                Claim(number=2, text="종속항 내용", is_independent=False, depends_on=[1]),
                Claim(number=3, text="종속항 내용", is_independent=False, depends_on=[1]),
            ],
        )

    def test_get_section(self, sample_document: ParsedDocument) -> None:
        """Test getting section by type."""
        claims_section = sample_document.get_section(SectionType.CLAIMS)
        assert claims_section is not None
        assert claims_section.title == "특허청구범위"

    def test_get_section_not_found(self, sample_document: ParsedDocument) -> None:
        """Test getting non-existent section."""
        tech_section = sample_document.get_section(SectionType.TECHNICAL_FIELD)
        assert tech_section is None

    def test_get_independent_claims(self, sample_document: ParsedDocument) -> None:
        """Test getting independent claims."""
        independent = sample_document.get_independent_claims()
        assert len(independent) == 1
        assert independent[0].number == 1

    def test_get_dependent_claims(self, sample_document: ParsedDocument) -> None:
        """Test getting dependent claims."""
        dependent = sample_document.get_dependent_claims(1)
        assert len(dependent) == 2
        assert all(1 in c.depends_on for c in dependent)


class TestDocumentParser:
    """Tests for DocumentParser."""

    @pytest.fixture
    def parser(self) -> DocumentParser:
        """Create document parser for testing."""
        return DocumentParser()

    def test_detect_korean(self, parser: DocumentParser) -> None:
        """Test Korean text detection."""
        korean_text = "본 발명은 인공지능을 이용한 자연어 처리 장치에 관한 것이다."
        english_text = "This invention relates to a natural language processing device."
        mixed_text = "본 발명은 machine learning을 이용한 system이다."

        assert parser._detect_korean(korean_text) is True
        assert parser._detect_korean(english_text) is False
        # Mixed text with >30% Korean
        assert parser._detect_korean(mixed_text) is True

    def test_extract_claims_korean(self, parser: DocumentParser) -> None:
        """Test Korean claim extraction."""
        claims_text = """
        【청구항 1】
        인공지능을 이용한 데이터 처리 방법에 있어서,
        데이터를 수집하는 단계;
        상기 데이터를 처리하는 단계;를 포함하는 방법.

        【청구항 2】
        제1항에 있어서,
        상기 데이터는 텍스트 데이터인 것을 특징으로 하는 방법.
        """

        claims = parser._extract_claims(claims_text, is_korean=True)

        assert len(claims) == 2
        assert claims[0].number == 1
        assert claims[0].is_independent is True
        assert claims[1].number == 2
        assert claims[1].is_independent is False
        assert claims[1].depends_on == [1]

    def test_extract_claims_english(self, parser: DocumentParser) -> None:
        """Test English claim extraction."""
        claims_text = """
        1. A method for processing data using artificial intelligence, comprising:
        collecting data;
        processing the data.

        2. The method of claim 1, wherein the data is text data.
        """

        claims = parser._extract_claims(claims_text, is_korean=False)

        assert len(claims) == 2
        assert claims[0].is_independent is True
        assert claims[1].is_independent is False

    def test_detect_claim_type_korean(self, parser: DocumentParser) -> None:
        """Test Korean claim type detection."""
        method_claim = "데이터를 처리하는 방법에 있어서, 다음 단계를 포함하는 방법."
        apparatus_claim = "데이터 처리 장치에 있어서, 프로세서를 포함하는 장치."
        composition_claim = "약학적 조성물에 있어서, 화합물 A를 포함하는 조성물."

        assert parser._detect_claim_type(method_claim, is_korean=True) == "method"
        assert parser._detect_claim_type(apparatus_claim, is_korean=True) == "apparatus"
        assert parser._detect_claim_type(composition_claim, is_korean=True) == "composition"

    def test_extract_drawings_korean(self, parser: DocumentParser) -> None:
        """Test Korean drawing extraction."""
        drawings_text = """
        도 1은 본 발명의 전체 시스템 구성도(10)이다.
        도 2는 데이터 처리부(20)의 상세 구조도이다.
        """

        drawings = parser._extract_drawings(drawings_text, is_korean=True)

        assert len(drawings) == 2
        assert drawings[0].number == 1
        assert "(10)" in drawings[0].description or "10" in drawings[0].reference_numbers


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_to_numpy(self) -> None:
        """Test conversion to numpy array."""
        result = EmbeddingResult(
            text="test text",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimension=3,
        )

        np_array = result.to_numpy()

        assert np_array.shape == (3,)
        assert np_array[0] == pytest.approx(0.1)

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation."""
        result1 = EmbeddingResult(
            text="text1",
            embedding=[1.0, 0.0, 0.0],
            model="test",
            dimension=3,
        )
        result2 = EmbeddingResult(
            text="text2",
            embedding=[1.0, 0.0, 0.0],
            model="test",
            dimension=3,
        )
        result3 = EmbeddingResult(
            text="text3",
            embedding=[0.0, 1.0, 0.0],
            model="test",
            dimension=3,
        )

        # Same vectors should have similarity 1.0
        assert result1.cosine_similarity(result2) == pytest.approx(1.0)

        # Orthogonal vectors should have similarity 0.0
        assert result1.cosine_similarity(result3) == pytest.approx(0.0)


class TestPatentEmbeddings:
    """Tests for PatentEmbeddings."""

    @pytest.fixture
    def mock_embeddings(self) -> PatentEmbeddings:
        """Create patent embeddings with mocked providers."""
        embeddings = PatentEmbeddings()
        return embeddings

    def test_is_korean_detection(self, mock_embeddings: PatentEmbeddings) -> None:
        """Test Korean text detection."""
        korean_text = "본 발명은 인공지능에 관한 것이다."
        english_text = "This invention relates to artificial intelligence."

        assert mock_embeddings._is_korean(korean_text) is True
        assert mock_embeddings._is_korean(english_text) is False

    def test_available_models(self) -> None:
        """Test getting available models."""
        embeddings = PatentEmbeddings()
        models = embeddings.available_models

        # Should return a list (may be empty if no providers available)
        assert isinstance(models, list)


class TestToolsIntegration:
    """Integration tests for tools module."""

    def test_tools_import(self) -> None:
        """Test that all tools can be imported."""
        from patent_agent.tools import (
            DocumentParser,
            KIPRISClient,
            KIPRISSearchResult,
            ParsedDocument,
            PatentEmbeddings,
            USPTOClient,
            USPTOSearchResult,
        )

        # Verify all classes are importable
        assert KIPRISClient is not None
        assert USPTOClient is not None
        assert DocumentParser is not None
        assert PatentEmbeddings is not None
        assert KIPRISSearchResult is not None
        assert USPTOSearchResult is not None
        assert ParsedDocument is not None

    def test_create_kipris_tools(self) -> None:
        """Test KIPRIS LangChain tools creation."""
        from patent_agent.tools.kipris import create_kipris_tools

        tools = create_kipris_tools(api_key="test_key")

        assert len(tools) == 2
        assert tools[0].name == "search_korean_patents"
        assert tools[1].name == "get_korean_patent_detail"

    def test_create_uspto_tools(self) -> None:
        """Test USPTO LangChain tools creation."""
        from patent_agent.tools.uspto import create_uspto_tools

        tools = create_uspto_tools()

        assert len(tools) == 3
        assert tools[0].name == "search_us_patents"
        assert tools[1].name == "get_us_patent_detail"
        assert tools[2].name == "get_us_application_status"

    def test_create_document_parser_tools(self) -> None:
        """Test document parser LangChain tools creation."""
        from patent_agent.tools.document_parser import create_document_parser_tools

        tools = create_document_parser_tools()

        assert len(tools) == 2
        assert tools[0].name == "parse_patent_document"
        assert tools[1].name == "extract_claims_from_document"
