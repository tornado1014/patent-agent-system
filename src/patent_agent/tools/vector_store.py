"""Vector store integration for patent documents.

Uses ChromaDB for local vector storage with patent-specific embeddings.
Supports semantic search, similarity matching, and document management.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import structlog

from patent_agent.config import get_settings
from patent_agent.tools.embeddings import (
    EmbeddingModel,
    PatentEmbeddings,
    create_patent_embeddings_langchain,
)

logger = structlog.get_logger(__name__)


@dataclass
class PatentDocument:
    """Patent document for vector storage."""

    id: str
    title: str
    content: str
    document_type: Literal["claim", "abstract", "description", "full"]
    metadata: dict[str, Any] = field(default_factory=dict)
    application_number: str | None = None
    patent_number: str | None = None
    filing_date: str | None = None
    source: str = "unknown"  # kipris, uspto, user_upload

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "document_type": self.document_type,
            "application_number": self.application_number,
            "patent_number": self.patent_number,
            "filing_date": self.filing_date,
            "source": self.source,
            **self.metadata,
        }


@dataclass
class SearchResult:
    """Search result from vector store."""

    document: PatentDocument
    score: float
    distance: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "distance": self.distance,
        }


class PatentVectorStore:
    """Vector store for patent documents using ChromaDB.

    Provides semantic search capabilities for patent documents with support for:
    - Multiple embedding models (KorPatELECTRA, OpenAI, HuggingFace)
    - Document chunking and metadata filtering
    - Similarity search with score thresholds
    - Collection management

    Example:
        >>> store = PatentVectorStore()
        >>> await store.add_document(PatentDocument(
        ...     id="doc1",
        ...     title="AI 특허",
        ...     content="인공지능 기반 자연어처리 시스템...",
        ...     document_type="abstract",
        ... ))
        >>> results = await store.search("자연어처리 기술", top_k=5)
        >>> for result in results:
        ...     print(f"{result.document.title}: {result.score:.3f}")
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize patent vector store.

        Args:
            collection_name: Name of ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_model: Preferred embedding model
        """
        settings = get_settings()
        self._collection_name = collection_name or settings.vector_store.collection_name
        self._persist_directory = Path(
            persist_directory or settings.vector_store.persist_directory
        )
        self._embedding_model = embedding_model

        # Initialize ChromaDB
        self._client: Any = None
        self._collection: Any = None
        self._embeddings: PatentEmbeddings | None = None
        self._available = False

        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            # Ensure persist directory exists
            self._persist_directory.mkdir(parents=True, exist_ok=True)

            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=str(self._persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Initialize embeddings
            self._embeddings = PatentEmbeddings(prefer_model=self._embedding_model)

            self._available = True
            logger.info(
                "PatentVectorStore initialized",
                collection=self._collection_name,
                persist_directory=str(self._persist_directory),
                document_count=self._collection.count(),
            )

        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
        except Exception as e:
            logger.error(f"Failed to initialize PatentVectorStore: {e}")

    @property
    def is_available(self) -> bool:
        """Check if vector store is available."""
        return self._available

    @property
    def document_count(self) -> int:
        """Get number of documents in collection."""
        if not self._available:
            return 0
        return self._collection.count()

    async def add_document(self, document: PatentDocument) -> str:
        """Add a single document to the vector store.

        Args:
            document: Patent document to add

        Returns:
            Document ID
        """
        if not self._available:
            raise RuntimeError("Vector store not available")

        # Generate embedding
        embedding_result = await self._embeddings.embed(document.content)

        # Prepare metadata
        metadata = document.to_dict()
        # Remove content from metadata (stored separately)
        metadata.pop("content", None)
        metadata.pop("id", None)

        # Add timestamp
        metadata["indexed_at"] = datetime.now().isoformat()

        # Add to collection
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                ids=[document.id],
                embeddings=[embedding_result.embedding],
                documents=[document.content],
                metadatas=[metadata],
            ),
        )

        logger.info(
            "Document added to vector store",
            document_id=document.id,
            title=document.title,
        )

        return document.id

    async def add_documents(self, documents: list[PatentDocument]) -> list[str]:
        """Add multiple documents to the vector store.

        Args:
            documents: List of patent documents

        Returns:
            List of document IDs
        """
        if not self._available:
            raise RuntimeError("Vector store not available")

        if not documents:
            return []

        # Generate embeddings in batch
        contents = [doc.content for doc in documents]
        embedding_results = await self._embeddings.embed_batch(contents)

        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        embeddings = [r.embedding for r in embedding_results]

        metadatas = []
        for doc in documents:
            metadata = doc.to_dict()
            metadata.pop("content", None)
            metadata.pop("id", None)
            metadata["indexed_at"] = datetime.now().isoformat()
            metadatas.append(metadata)

        # Add to collection
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            ),
        )

        logger.info(
            "Documents added to vector store",
            count=len(documents),
        )

        return ids

    async def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query text
            top_k: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filter

        Returns:
            List of search results sorted by relevance
        """
        if not self._available:
            raise RuntimeError("Vector store not available")

        # Generate query embedding
        query_embedding = await self._embeddings.embed(query)

        # Build where clause for filtering
        where = None
        if filter_metadata:
            where = filter_metadata

        # Execute search
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=[query_embedding.embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            ),
        )

        # Parse results
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert distance to similarity score (cosine distance)
                score = 1.0 - distance

                if score < score_threshold:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""

                document = PatentDocument(
                    id=doc_id,
                    title=metadata.get("title", ""),
                    content=content,
                    document_type=metadata.get("document_type", "full"),
                    application_number=metadata.get("application_number"),
                    patent_number=metadata.get("patent_number"),
                    filing_date=metadata.get("filing_date"),
                    source=metadata.get("source", "unknown"),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k
                        not in [
                            "title",
                            "document_type",
                            "application_number",
                            "patent_number",
                            "filing_date",
                            "source",
                        ]
                    },
                )

                search_results.append(
                    SearchResult(
                        document=document,
                        score=score,
                        distance=distance,
                    )
                )

        logger.info(
            "Vector search completed",
            query_length=len(query),
            results_count=len(search_results),
        )

        return search_results

    async def get_document(self, document_id: str) -> PatentDocument | None:
        """Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            PatentDocument or None if not found
        """
        if not self._available:
            return None

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._collection.get(
                ids=[document_id],
                include=["documents", "metadatas"],
            ),
        )

        if not result or not result["ids"]:
            return None

        metadata = result["metadatas"][0] if result["metadatas"] else {}
        content = result["documents"][0] if result["documents"] else ""

        return PatentDocument(
            id=document_id,
            title=metadata.get("title", ""),
            content=content,
            document_type=metadata.get("document_type", "full"),
            application_number=metadata.get("application_number"),
            patent_number=metadata.get("patent_number"),
            filing_date=metadata.get("filing_date"),
            source=metadata.get("source", "unknown"),
        )

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID.

        Args:
            document_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        if not self._available:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._collection.delete(ids=[document_id]),
            )
            logger.info("Document deleted", document_id=document_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete document: {e}")
            return False

    async def update_document(self, document: PatentDocument) -> bool:
        """Update an existing document.

        Args:
            document: Updated patent document

        Returns:
            True if updated successfully
        """
        if not self._available:
            return False

        # Delete and re-add
        await self.delete_document(document.id)
        await self.add_document(document)
        return True

    def reset_collection(self) -> None:
        """Reset (delete all documents in) the collection."""
        if not self._available:
            return

        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset", collection=self._collection_name)

    async def add_from_kipris(
        self,
        search_results: list[Any],
    ) -> list[str]:
        """Add documents from KIPRIS search results.

        Args:
            search_results: List of KIPRISSearchResult objects

        Returns:
            List of added document IDs
        """
        documents = []
        for result in search_results:
            doc_id = f"kipris_{result.application_number}"
            documents.append(
                PatentDocument(
                    id=doc_id,
                    title=result.title,
                    content=f"{result.title}\n\n{result.abstract}" if result.abstract else result.title,
                    document_type="abstract",
                    application_number=result.application_number,
                    patent_number=result.registration_number,
                    filing_date=result.filing_date.isoformat() if result.filing_date else None,
                    source="kipris",
                    metadata={
                        "applicant": result.applicant,
                        "ipc_codes": result.ipc_codes,
                        "status": result.status.value if hasattr(result.status, "value") else str(result.status),
                    },
                )
            )

        return await self.add_documents(documents)

    async def add_from_uspto(
        self,
        search_results: list[Any],
    ) -> list[str]:
        """Add documents from USPTO search results.

        Args:
            search_results: List of USPTOSearchResult objects

        Returns:
            List of added document IDs
        """
        documents = []
        for result in search_results:
            doc_id = f"uspto_{result.application_number}"
            documents.append(
                PatentDocument(
                    id=doc_id,
                    title=result.title,
                    content=f"{result.title}\n\n{result.abstract}" if result.abstract else result.title,
                    document_type="abstract",
                    application_number=result.application_number,
                    patent_number=result.patent_number,
                    filing_date=result.filing_date.isoformat() if result.filing_date else None,
                    source="uspto",
                    metadata={
                        "applicant": result.applicant,
                        "cpc_codes": result.cpc_codes,
                        "status": result.status.value if hasattr(result.status, "value") else str(result.status),
                    },
                )
            )

        return await self.add_documents(documents)


# LangChain integration
def create_patent_vectorstore_langchain(
    collection_name: str | None = None,
    persist_directory: Path | str | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> Any:
    """Create LangChain-compatible vector store for patents.

    Args:
        collection_name: ChromaDB collection name
        persist_directory: Directory for persistence
        embedding_model: Preferred embedding model

    Returns:
        LangChain Chroma vector store
    """
    from langchain_community.vectorstores import Chroma

    settings = get_settings()
    collection = collection_name or settings.vector_store.collection_name
    persist_dir = Path(persist_directory or settings.vector_store.persist_directory)

    # Create embeddings
    embeddings = create_patent_embeddings_langchain(model=embedding_model)

    # Ensure directory exists
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
