"""Patent Agent Tools - External API wrappers and utilities."""

from patent_agent.tools.kipris import (
    KIPRISClient,
    KIPRISSearchResult,
    KIPRISPatentDetail,
)
from patent_agent.tools.uspto import (
    USPTOClient,
    USPTOSearchResult,
    USPTOPatentDetail,
)
from patent_agent.tools.document_parser import (
    DocumentParser,
    ParsedDocument,
    DocumentSection,
)
from patent_agent.tools.embeddings import (
    PatentEmbeddings,
    EmbeddingResult,
)
from patent_agent.tools.vector_store import (
    PatentVectorStore,
    PatentDocument,
    SearchResult,
)

__all__ = [
    # KIPRIS
    "KIPRISClient",
    "KIPRISSearchResult",
    "KIPRISPatentDetail",
    # USPTO
    "USPTOClient",
    "USPTOSearchResult",
    "USPTOPatentDetail",
    # Document Parser
    "DocumentParser",
    "ParsedDocument",
    "DocumentSection",
    # Embeddings
    "PatentEmbeddings",
    "EmbeddingResult",
    # Vector Store
    "PatentVectorStore",
    "PatentDocument",
    "SearchResult",
]
