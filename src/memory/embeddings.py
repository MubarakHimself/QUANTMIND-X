"""
QuantMindX Embedding Providers

Multi-provider embedding support with:
- OpenAI embeddings
- Z.AI embeddings (Anthropic-compatible API)
- Local sentence-transformers fallback
- Batch embedding support
- Automatic retry and error handling
"""

import asyncio
import logging
from typing import List, Optional, Literal, Protocol, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    model: str
    dimension: int
    batch_size: int = 32
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    All embedding providers must implement:
    - embed(): Single text embedding
    - embed_batch(): Batch text embedding
    - get_dimension(): Return embedding dimension
    """
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using the OpenAI API.
    
    Supports models:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    
    Example:
        >>> provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        >>> embedding = await provider.embed("Hello, world!")
        >>> print(len(embedding))  # 1536
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            model: Model name (default: text-embedding-3-small)
            api_key: OpenAI API key (from env if not provided)
            base_url: Custom base URL (optional)
            config: Embedding configuration
        """
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if model not in self.MODELS:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Choose from: {list(self.MODELS.keys())}"
            )
        
        self._dimension = self.MODELS[model]
        self._config = config or EmbeddingConfig(
            model=model,
            dimension=self._dimension,
        )
        
        self._client = None
        logger.info(f"OpenAIEmbeddingProvider initialized: model={model}")
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. "
                    "Install with: pip install openai"
                )
        return self._client
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        client = self._get_client()
        
        try:
            response = await client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        client = self._get_client()
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self._config.batch_size):
            batch = texts[i : i + self._config.batch_size]
            
            try:
                response = await client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                results.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"OpenAI batch embedding failed: {e}")
                # Fallback to individual embeddings
                for text in batch:
                    embedding = await self.embed(text)
                    results.append(embedding)
        
        return results
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model


class ZAIEmbeddingProvider(EmbeddingProvider):
    """
    Z.AI embedding provider using Anthropic-compatible API.
    
    Uses base_url: https://api.z.ai/api/anthropic
    
    Example:
        >>> provider = ZAIEmbeddingProvider(
        ...     api_key="your-api-key",
        ...     model="claude-3-5-sonnet-20241022"
        ... )
        >>> embedding = await provider.embed("Hello, world!")
    """
    
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    DEFAULT_BASE_URL = "https://api.z.ai/api/anthropic"
    
    # Z.AI typically uses OpenAI-compatible embedding endpoints
    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initialize Z.AI embedding provider.
        
        Args:
            model: Model name
            api_key: Z.AI API key (from env if not provided)
            base_url: Custom base URL (default: https://api.z.ai/api/anthropic)
            config: Embedding configuration
        """
        self.model = model
        self._api_key = api_key or os.getenv("ZAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self._base_url = base_url or os.getenv("ZAI_BASE_URL", self.DEFAULT_BASE_URL)
        
        if model not in self.MODELS:
            self._dimension = 1536  # Default
        else:
            self._dimension = self.MODELS[model]
        
        self._config = config or EmbeddingConfig(
            model=model,
            dimension=self._dimension,
        )
        
        self._client = None
        logger.info(f"ZAIEmbeddingProvider initialized: model={model}")
    
    def _get_client(self):
        """Lazy initialization of OpenAI client (Z.AI uses compatible API)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. "
                    "Install with: pip install openai"
                )
        return self._client
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        client = self._get_client()
        
        try:
            response = await client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Z.AI embedding failed: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        client = self._get_client()
        results = []
        
        for i in range(0, len(texts), self._config.batch_size):
            batch = texts[i : i + self._config.batch_size]
            
            try:
                response = await client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                results.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Z.AI batch embedding failed: {e}")
                for text in batch:
                    embedding = await self.embed(text)
                    results.append(embedding)
        
        return results
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    
    Supports any model from the sentence-transformers library:
    - all-MiniLM-L6-v2 (384 dimensions) - Fast, English-only
    - all-mpnet-base-v2 (768 dimensions) - Good quality, English-only
    - paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions) - Multilingual
    
    Example:
        >>> provider = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
        >>> embedding = await provider.embed("Hello, world!")
        >>> print(len(embedding))  # 384
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    # Common models and their dimensions
    MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "distiluse-base-multilingual-cased-v2": 512,
    }
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initialize local embedding provider.
        
        Args:
            model: Model name (default: all-MiniLM-L6-v2)
            device: Device to use (cpu, cuda, etc.) - auto-detect if None
            config: Embedding configuration
        """
        self.model = model
        self._device = device
        self._model = None
        self._config = config or EmbeddingConfig(
            model=model,
            dimension=self.MODELS.get(model, 384),
        )
        
        logger.info(f"LocalEmbeddingProvider initialized: model={model}")
    
    def _get_model(self):
        """Lazy initialization of sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model, device=self._device)
                logger.info(f"Loaded model: {self.model}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        model = self._get_model()
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True)
        )
        
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        model = self._get_model()
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self._config.batch_size):
            batch = texts[i : i + self._config.batch_size]
            
            # Run in executor
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(batch, convert_to_numpy=True)
            )
            
            # Convert to list of lists
            results.extend(embeddings.tolist())
        
        return results
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        # Get dimension from model config
        return self._config.dimension
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing.
    
    Generates deterministic pseudo-random embeddings based on text content.
    """
    
    def __init__(self, dimension: int = 1536):
        """
        Initialize mock embedding provider.
        
        Args:
            dimension: Embedding dimension (default: 1536)
        """
        self._dimension = dimension
        logger.info(f"MockEmbeddingProvider initialized: dim={dimension}")
    
    def _hash_to_vector(self, text: str) -> List[float]:
        """Generate deterministic vector from text hash."""
        import hashlib
        
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Extend hash to desired dimension
        vector = []
        for i in range(self._dimension):
            byte = hash_bytes[i % len(hash_bytes)]
            # Convert to float in [-1, 1]
            value = (byte - 128) / 128.0
            vector.append(value)
        
        return vector
    
    async def embed(self, text: str) -> List[float]:
        """Generate mock embedding for a single text."""
        return self._hash_to_vector(text)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for multiple texts."""
        return [self._hash_to_vector(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return "mock"


def get_embedding_provider(
    provider: Literal["openai", "zai", "local", "mock"] = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.
    
    Args:
        provider: Provider type ("openai", "zai", "local", "mock")
        model: Model name (provider-specific default if not specified)
        api_key: API key (from environment if not specified)
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Configured EmbeddingProvider instance
        
    Example:
        >>> # Use OpenAI
        >>> provider = get_embedding_provider("openai", model="text-embedding-3-small")
        >>> 
        >>> # Use local sentence-transformers
        >>> provider = get_embedding_provider("local", model="all-MiniLM-L6-v2")
        >>> 
        >>> # Use mock for testing
        >>> provider = get_embedding_provider("mock", dimension=384)
    """
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            model=model or OpenAIEmbeddingProvider.DEFAULT_MODEL,
            api_key=api_key,
            **kwargs
        )
    
    elif provider == "zai":
        return ZAIEmbeddingProvider(
            model=model or "text-embedding-3-small",
            api_key=api_key,
            **kwargs
        )
    
    elif provider == "local":
        return LocalEmbeddingProvider(
            model=model or LocalEmbeddingProvider.DEFAULT_MODEL,
            **kwargs
        )
    
    elif provider == "mock":
        dimension = kwargs.get("dimension", 1536)
        return MockEmbeddingProvider(dimension=dimension)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: openai, zai, local, mock"
        )
