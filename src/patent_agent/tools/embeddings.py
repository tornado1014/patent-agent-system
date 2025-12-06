"""Patent embeddings using KorPatELECTRA and other models.

KorPatELECTRA is a Korean patent-specific language model developed by KIPI.
It provides specialized embeddings for Korean patent text with superior
performance on patent-related NLP tasks:
- NER: 91.01%
- Classification: 73.90%
- MRC: 89.85%

Reference: https://github.com/KIPI-ai/KorPatElectra
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from patent_agent.config import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingModel(str, Enum):
    """Available embedding models."""

    KORPATELECTRA = "korpatelectra"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


@dataclass
class EmbeddingResult:
    """Result of text embedding."""

    text: str
    embedding: list[float]
    model: str
    dimension: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array."""
        return np.array(self.embedding, dtype=np.float32)

    def cosine_similarity(self, other: EmbeddingResult) -> float:
        """Calculate cosine similarity with another embedding."""
        a = self.to_numpy()
        b = other.to_numpy()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        ...


class KorPatELECTRAProvider(BaseEmbeddingProvider):
    """KorPatELECTRA embedding provider.

    Uses the KIPI-ai/KorPatElectra model for Korean patent text embeddings.
    Falls back to mean pooling of token embeddings for sentence-level representations.
    """

    MODEL_ID = "KIPI-ai/KorPatElectra"

    def __init__(self, device: str = "cpu") -> None:
        """Initialize KorPatELECTRA provider.

        Args:
            device: Device to run model on ('cpu' or 'cuda')
        """
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._available = False

        # Try to load model
        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self._model = AutoModel.from_pretrained(self.MODEL_ID)
            self._model.to(device)
            self._model.eval()
            self._available = True
            logger.info("KorPatELECTRA model loaded", device=device)
        except ImportError:
            logger.warning("transformers not available for KorPatELECTRA")
        except Exception as e:
            logger.warning(f"Failed to load KorPatELECTRA: {e}")

    @property
    def dimension(self) -> int:
        """Get embedding dimension (ELECTRA base: 768)."""
        return 768

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.MODEL_ID

    @property
    def is_available(self) -> bool:
        """Check if model is available."""
        return self._available

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single Korean patent text.

        Args:
            text: Korean patent text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        if not self._available:
            raise RuntimeError("KorPatELECTRA model not available")

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed_sync, text)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model_name,
            dimension=self.dimension,
        )

    def _embed_sync(self, text: str) -> list[float]:
        """Synchronous embedding computation."""
        import torch

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling over token embeddings
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Expand attention mask for broadcasting
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Calculate mean
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings[0].cpu().numpy().tolist()

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts.

        Args:
            texts: List of Korean patent texts

        Returns:
            List of EmbeddingResults
        """
        results = []
        for text in texts:
            result = await self.embed_text(text)
            results.append(result)
        return results


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small/large."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (uses settings if not provided)
        """
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key
        self._model = model
        self._client: Any = None
        self._available = False

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
            self._available = True
            logger.info("OpenAI embedding provider initialized", model=model)
        except ImportError:
            logger.warning("openai package not available")

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimensions.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model

    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self._available and self._api_key is not None

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        if not self.is_available:
            raise RuntimeError("OpenAI embedding provider not available")

        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )

        return EmbeddingResult(
            text=text,
            embedding=response.data[0].embedding,
            model=self.model_name,
            dimension=self.dimension,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResults
        """
        if not self.is_available:
            raise RuntimeError("OpenAI embedding provider not available")

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        results = []
        for i, embedding_data in enumerate(response.data):
            results.append(
                EmbeddingResult(
                    text=texts[i],
                    embedding=embedding_data.embedding,
                    model=self.model_name,
                    dimension=self.dimension,
                )
            )

        return results


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers."""

    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
    ) -> None:
        """Initialize HuggingFace embedding provider.

        Args:
            model: HuggingFace model name
            device: Device to run model on
        """
        self._model_name = model
        self._device = device
        self._model: Any = None
        self._available = False

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model, device=device)
            self._available = True
            logger.info("HuggingFace embedding provider initialized", model=model)
        except ImportError:
            logger.warning("sentence-transformers not available")
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model: {e}")

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 384  # Default for MiniLM

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self._available

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        if not self._available:
            raise RuntimeError("HuggingFace embedding provider not available")

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self._model.encode(text).tolist()
        )

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model_name,
            dimension=self.dimension,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResults
        """
        if not self._available:
            raise RuntimeError("HuggingFace embedding provider not available")

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts).tolist()
        )

        return [
            EmbeddingResult(
                text=texts[i],
                embedding=embeddings[i],
                model=self.model_name,
                dimension=self.dimension,
            )
            for i in range(len(texts))
        ]


class PatentEmbeddings:
    """Unified patent embedding interface.

    Provides a single interface for patent text embeddings with automatic
    model selection based on availability and text language.

    For Korean patent text, KorPatELECTRA is preferred.
    For English or mixed text, OpenAI or HuggingFace models are used.

    Example:
        >>> embeddings = PatentEmbeddings()
        >>> result = await embeddings.embed("특허 명세서 텍스트...")
        >>> print(f"Dimension: {result.dimension}")
        >>> print(f"Model: {result.model}")
    """

    def __init__(
        self,
        prefer_model: EmbeddingModel | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize patent embeddings.

        Args:
            prefer_model: Preferred embedding model
            device: Device for local models ('cpu' or 'cuda')
        """
        self._prefer_model = prefer_model
        self._device = device

        # Initialize providers
        self._providers: dict[EmbeddingModel, BaseEmbeddingProvider] = {}

        # KorPatELECTRA for Korean text
        korpatelectra = KorPatELECTRAProvider(device=device)
        if korpatelectra.is_available:
            self._providers[EmbeddingModel.KORPATELECTRA] = korpatelectra

        # OpenAI for general text
        openai_provider = OpenAIEmbeddingProvider()
        if openai_provider.is_available:
            self._providers[EmbeddingModel.OPENAI] = openai_provider

        # HuggingFace as fallback
        hf_provider = HuggingFaceEmbeddingProvider(device=device)
        if hf_provider.is_available:
            self._providers[EmbeddingModel.HUGGINGFACE] = hf_provider

        logger.info(
            "PatentEmbeddings initialized",
            available_models=list(self._providers.keys()),
        )

    def _select_provider(
        self, text: str, model: EmbeddingModel | None = None
    ) -> BaseEmbeddingProvider:
        """Select appropriate provider for text."""
        # Use specified model if available
        if model and model in self._providers:
            return self._providers[model]

        # Use preferred model if available
        if self._prefer_model and self._prefer_model in self._providers:
            return self._providers[self._prefer_model]

        # Auto-select based on text
        if self._is_korean(text) and EmbeddingModel.KORPATELECTRA in self._providers:
            return self._providers[EmbeddingModel.KORPATELECTRA]

        # Prefer OpenAI, then HuggingFace
        if EmbeddingModel.OPENAI in self._providers:
            return self._providers[EmbeddingModel.OPENAI]

        if EmbeddingModel.HUGGINGFACE in self._providers:
            return self._providers[EmbeddingModel.HUGGINGFACE]

        raise RuntimeError("No embedding provider available")

    def _is_korean(self, text: str) -> bool:
        """Check if text is primarily Korean."""
        import re

        korean_chars = len(re.findall(r"[\uac00-\ud7af]", text))
        total_chars = len(text.replace(" ", "").replace("\n", ""))
        if total_chars == 0:
            return False
        return korean_chars / total_chars > 0.3

    async def embed(
        self,
        text: str,
        model: EmbeddingModel | None = None,
    ) -> EmbeddingResult:
        """Embed patent text.

        Args:
            text: Text to embed
            model: Optional specific model to use

        Returns:
            EmbeddingResult with embedding vector
        """
        provider = self._select_provider(text, model)
        return await provider.embed_text(text)

    async def embed_batch(
        self,
        texts: list[str],
        model: EmbeddingModel | None = None,
    ) -> list[EmbeddingResult]:
        """Embed multiple patent texts.

        Args:
            texts: List of texts to embed
            model: Optional specific model to use

        Returns:
            List of EmbeddingResults
        """
        if not texts:
            return []

        provider = self._select_provider(texts[0], model)
        return await provider.embed_batch(texts)

    async def similarity(
        self,
        text1: str,
        text2: str,
        model: EmbeddingModel | None = None,
    ) -> float:
        """Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            model: Optional specific model to use

        Returns:
            Cosine similarity score (0-1)
        """
        results = await self.embed_batch([text1, text2], model)
        return results[0].cosine_similarity(results[1])

    async def find_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
        model: EmbeddingModel | None = None,
    ) -> list[tuple[int, str, float]]:
        """Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            model: Optional specific model to use

        Returns:
            List of (index, text, similarity_score) tuples
        """
        # Embed query and candidates
        all_texts = [query] + candidates
        results = await self.embed_batch(all_texts, model)

        query_result = results[0]
        candidate_results = results[1:]

        # Calculate similarities
        similarities = []
        for i, candidate_result in enumerate(candidate_results):
            similarity = query_result.cosine_similarity(candidate_result)
            similarities.append((i, candidates[i], similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)

        return similarities[:top_k]

    @property
    def available_models(self) -> list[EmbeddingModel]:
        """Get list of available embedding models."""
        return list(self._providers.keys())


# LangChain integration
def create_patent_embeddings_langchain(
    model: EmbeddingModel | None = None,
    device: str = "cpu",
) -> Any:
    """Create LangChain-compatible embeddings for patents.

    Args:
        model: Preferred embedding model
        device: Device for local models

    Returns:
        LangChain Embeddings instance
    """
    from langchain_core.embeddings import Embeddings

    patent_embeddings = PatentEmbeddings(prefer_model=model, device=device)

    class PatentLangChainEmbeddings(Embeddings):
        """LangChain-compatible patent embeddings."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """Embed documents synchronously."""
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(patent_embeddings.embed_batch(texts))
            return [r.embedding for r in results]

        def embed_query(self, text: str) -> list[float]:
            """Embed query synchronously."""
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(patent_embeddings.embed(text))
            return result.embedding

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            """Embed documents asynchronously."""
            results = await patent_embeddings.embed_batch(texts)
            return [r.embedding for r in results]

        async def aembed_query(self, text: str) -> list[float]:
            """Embed query asynchronously."""
            result = await patent_embeddings.embed(text)
            return result.embedding

    return PatentLangChainEmbeddings()
