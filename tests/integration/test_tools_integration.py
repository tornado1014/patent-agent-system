"""Integration tests for patent tools (KIPRIS, USPTO, VectorStore)."""

import asyncio
import pytest
from datetime import date

# Skip if dependencies not installed
pytest.importorskip("chromadb")


class TestToolsImport:
    """Test that all tools can be imported."""

    def test_import_kipris(self):
        """Test KIPRIS client import."""
        from patent_agent.tools.kipris import (
            KIPRISClient,
            KIPRISSearchResult,
            KIPRISSearchType,
        )
        assert KIPRISClient is not None
        assert KIPRISSearchResult is not None

    def test_import_uspto(self):
        """Test USPTO client import."""
        from patent_agent.tools.uspto import (
            USPTOClient,
            USPTOSearchResult,
            USPTOSearchType,
        )
        assert USPTOClient is not None
        assert USPTOSearchResult is not None

    def test_import_embeddings(self):
        """Test embeddings import."""
        from patent_agent.tools.embeddings import (
            PatentEmbeddings,
            EmbeddingResult,
            EmbeddingModel,
        )
        assert PatentEmbeddings is not None
        assert EmbeddingResult is not None

    def test_import_vector_store(self):
        """Test vector store import."""
        from patent_agent.tools.vector_store import (
            PatentVectorStore,
            PatentDocument,
            SearchResult,
        )
        assert PatentVectorStore is not None
        assert PatentDocument is not None

    def test_import_from_package(self):
        """Test imports from package __init__."""
        from patent_agent.tools import (
            KIPRISClient,
            USPTOClient,
            PatentEmbeddings,
            PatentVectorStore,
            PatentDocument,
        )
        assert all([
            KIPRISClient,
            USPTOClient,
            PatentEmbeddings,
            PatentVectorStore,
            PatentDocument,
        ])


class TestKIPRISClient:
    """Test KIPRIS client functionality."""

    def test_client_initialization(self):
        """Test KIPRIS client can be initialized."""
        from patent_agent.tools.kipris import KIPRISClient

        client = KIPRISClient()
        assert client is not None
        # API key may not be set in test environment

    def test_search_result_dataclass(self):
        """Test KIPRISSearchResult dataclass."""
        from patent_agent.tools.kipris import KIPRISSearchResult, PatentStatus

        result = KIPRISSearchResult(
            application_number="10-2024-0001234",
            title="인공지능 기반 특허 분석 시스템",
            applicant="테스트 주식회사",
            filing_date=date(2024, 1, 15),
            status=PatentStatus.PENDING,
            ipc_codes=["G06F", "G06N"],
            abstract="본 발명은 인공지능을 활용한 특허 분석 시스템에 관한 것이다.",
        )

        assert result.application_number == "10-2024-0001234"
        assert result.title == "인공지능 기반 특허 분석 시스템"

        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict["application_number"] == "10-2024-0001234"
        assert result_dict["status"] == "pending"


class TestUSPTOClient:
    """Test USPTO client functionality."""

    def test_client_initialization(self):
        """Test USPTO client can be initialized."""
        from patent_agent.tools.uspto import USPTOClient

        client = USPTOClient()
        assert client is not None

    def test_search_result_dataclass(self):
        """Test USPTOSearchResult dataclass."""
        from patent_agent.tools.uspto import USPTOSearchResult, USPatentStatus

        result = USPTOSearchResult(
            application_number="17/123456",
            title="AI-Based Patent Analysis System",
            applicant="Test Corporation",
            filing_date=date(2024, 1, 15),
            patent_number="US12345678B2",
            status=USPatentStatus.PATENTED,
            cpc_codes=["G06F", "G06N"],
            abstract="A system for analyzing patents using artificial intelligence.",
        )

        assert result.application_number == "17/123456"
        assert result.patent_number == "US12345678B2"

        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict["status"] == "patented"


class TestPatentVectorStore:
    """Test PatentVectorStore functionality."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary vector store for testing."""
        from patent_agent.tools.vector_store import PatentVectorStore

        store = PatentVectorStore(
            collection_name="test_patents",
            persist_directory=tmp_path / "chroma_test",
        )
        yield store

        # Cleanup
        if store.is_available:
            store.reset_collection()

    def test_store_initialization(self, temp_store):
        """Test vector store initialization."""
        # Store may not be available if dependencies missing
        # This is OK for basic import test
        assert temp_store is not None

    def test_patent_document_dataclass(self):
        """Test PatentDocument dataclass."""
        from patent_agent.tools.vector_store import PatentDocument

        doc = PatentDocument(
            id="test_001",
            title="테스트 특허",
            content="본 발명은 테스트에 관한 것이다.",
            document_type="abstract",
            application_number="10-2024-0001234",
            source="kipris",
        )

        assert doc.id == "test_001"
        assert doc.document_type == "abstract"

        doc_dict = doc.to_dict()
        assert doc_dict["source"] == "kipris"

    @pytest.mark.asyncio
    async def test_add_and_search_document(self, temp_store):
        """Test adding and searching documents."""
        if not temp_store.is_available:
            pytest.skip("Vector store not available")

        from patent_agent.tools.vector_store import PatentDocument

        # Add a document
        doc = PatentDocument(
            id="test_doc_001",
            title="인공지능 자연어처리 시스템",
            content="본 발명은 인공지능을 활용한 자연어처리 시스템에 관한 것으로, 딥러닝 기반의 언어 모델을 사용하여 텍스트를 분석한다.",
            document_type="abstract",
            source="test",
        )

        doc_id = await temp_store.add_document(doc)
        assert doc_id == "test_doc_001"

        # Search for similar documents
        results = await temp_store.search("자연어처리 딥러닝", top_k=5)
        assert len(results) > 0
        assert results[0].document.id == "test_doc_001"
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_batch_add_documents(self, temp_store):
        """Test adding multiple documents."""
        if not temp_store.is_available:
            pytest.skip("Vector store not available")

        from patent_agent.tools.vector_store import PatentDocument

        docs = [
            PatentDocument(
                id=f"batch_doc_{i}",
                title=f"테스트 특허 {i}",
                content=f"특허 내용 {i}: 기술 분야에 관한 발명",
                document_type="abstract",
            )
            for i in range(3)
        ]

        ids = await temp_store.add_documents(docs)
        assert len(ids) == 3
        assert temp_store.document_count >= 3


class TestEmbeddings:
    """Test embedding functionality."""

    def test_embedding_result_dataclass(self):
        """Test EmbeddingResult dataclass."""
        from patent_agent.tools.embeddings import EmbeddingResult
        import numpy as np

        result = EmbeddingResult(
            text="테스트 텍스트",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            model="test_model",
            dimension=5,
        )

        assert result.dimension == 5
        assert len(result.embedding) == 5

        # Test numpy conversion
        arr = result.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5,)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from patent_agent.tools.embeddings import EmbeddingResult

        result1 = EmbeddingResult(
            text="A",
            embedding=[1.0, 0.0, 0.0],
            model="test",
            dimension=3,
        )
        result2 = EmbeddingResult(
            text="B",
            embedding=[1.0, 0.0, 0.0],
            model="test",
            dimension=3,
        )
        result3 = EmbeddingResult(
            text="C",
            embedding=[0.0, 1.0, 0.0],
            model="test",
            dimension=3,
        )

        # Same vector should have similarity 1.0
        assert abs(result1.cosine_similarity(result2) - 1.0) < 0.001

        # Orthogonal vectors should have similarity 0.0
        assert abs(result1.cosine_similarity(result3)) < 0.001


class TestToolsIntegration:
    """Integration tests for combined tool usage."""

    @pytest.mark.asyncio
    async def test_kipris_to_vectorstore_flow(self, tmp_path):
        """Test flow from KIPRIS search results to vector store."""
        from patent_agent.tools.kipris import KIPRISSearchResult, PatentStatus
        from patent_agent.tools.vector_store import PatentVectorStore

        # Create mock KIPRIS results
        mock_results = [
            KIPRISSearchResult(
                application_number=f"10-2024-000{i}",
                title=f"테스트 특허 {i}",
                applicant="테스트 출원인",
                filing_date=date(2024, 1, i + 1),
                status=PatentStatus.PENDING,
                abstract=f"테스트 초록 {i}: 인공지능 관련 기술",
            )
            for i in range(3)
        ]

        # Initialize vector store
        store = PatentVectorStore(
            collection_name="kipris_test",
            persist_directory=tmp_path / "chroma_kipris",
        )

        if not store.is_available:
            pytest.skip("Vector store not available")

        # Add KIPRIS results to vector store
        ids = await store.add_from_kipris(mock_results)
        assert len(ids) == 3

        # Search
        results = await store.search("인공지능 기술", top_k=3)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_uspto_to_vectorstore_flow(self, tmp_path):
        """Test flow from USPTO search results to vector store."""
        from patent_agent.tools.uspto import USPTOSearchResult, USPatentStatus
        from patent_agent.tools.vector_store import PatentVectorStore

        # Create mock USPTO results
        mock_results = [
            USPTOSearchResult(
                application_number=f"17/12345{i}",
                title=f"Test Patent {i}",
                applicant="Test Applicant",
                filing_date=date(2024, 1, i + 1),
                patent_number=f"US1234567{i}B2",
                status=USPatentStatus.PATENTED,
                abstract=f"Test abstract {i}: AI related technology",
            )
            for i in range(3)
        ]

        # Initialize vector store
        store = PatentVectorStore(
            collection_name="uspto_test",
            persist_directory=tmp_path / "chroma_uspto",
        )

        if not store.is_available:
            pytest.skip("Vector store not available")

        # Add USPTO results to vector store
        ids = await store.add_from_uspto(mock_results)
        assert len(ids) == 3

        # Search
        results = await store.search("AI technology", top_k=3)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
