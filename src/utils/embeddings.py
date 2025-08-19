import os
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time
from datetime import datetime
import hashlib
import pickle
import numpy as np
import logging
import voyageai


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    cached: bool = False
    generation_time_ms: float = 0


class EmbeddingService:
    """
    Generate semantic embeddings using Voyage AI's embedding models.
    Provides 1024-dimensional vectors for semantic similarity comparison by default.
    
    Part of the AI-driven deduplication system that ensures Ignacio
    never sees the same story twice while preserving important updates.
    
    Voyage AI is Anthropic's recommended embedding provider, offering:
    - Better performance and retrieval quality
    - Higher rate limits (8M tokens/min for voyage-3.5)
    - Flexible dimensions (256, 512, 1024, 2048)
    - Optimized for various domains (general, code, finance, law)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-3.5"):
        """
        Initialize the embedding service with Voyage AI.

        Args:
            api_key: Voyage API key. If None, reads from VOYAGE_API_KEY env var.
            model: Voyage model to use. Options:
                   - "voyage-3.5": Balanced quality and cost (default)
                   - "voyage-3-large": Best quality
                   - "voyage-3.5-lite": Optimized for speed/cost

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key required. Set VOYAGE_API_KEY or pass api_key parameter. Get your key at https://dash.voyageai.com/")
        
        # Initialize Voyage client
        self.client = voyageai.Client(api_key=self.api_key, max_retries=3)
        
        # Model configuration
        self.model = model
        # voyage-3.5 supports flexible dimensions: 256, 512, 1024 (default), 2048
        self.dimensions = 1024  # Default dimension for voyage-3.5
        self.max_batch_size = 128  # Voyage recommends batches of 128
        self.max_retries = 3
        self.timeout = 30  # seconds
        
        # Token limits based on model
        model_token_limits = {
            "voyage-3.5-lite": 1_000_000,  # 1M tokens
            "voyage-3.5": 320_000,  # 320K tokens
            "voyage-3-large": 120_000,  # 120K tokens
            "voyage-code-3": 120_000,  # 120K tokens
            "voyage-finance-2": 120_000,
            "voyage-law-2": 120_000,
        }
        self.max_total_tokens = model_token_limits.get(model, 320_000)
        
        # Simple in-memory cache for the session
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Logger for operations
        self.logger = logging.getLogger(__name__)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for deduplication.
        Combines headline and first 500 chars for better semantic capture.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding

        Raises:
            EmbeddingServiceError: If API call fails after retries
        """
        # Voyage AI handles truncation automatically with truncation=True (default)
        text = text or ""
        
        attempt = 0
        backoff_seconds = 1.0
        last_error: Optional[BaseException] = None
        
        while attempt < self.max_retries:
            attempt += 1
            try:
                # Use synchronous client in async context (Voyage SDK doesn't have async yet)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.embed(
                        texts=[text],
                        model=self.model,
                        input_type="document",  # Optimize for document retrieval
                        truncation=True,  # Auto-truncate if needed
                        output_dimension=self.dimensions,
                    )
                )
                
                # Extract embedding from response
                if response.embeddings and len(response.embeddings) > 0:
                    return response.embeddings[0]
                else:
                    raise EmbeddingServiceError("No embedding returned from Voyage AI")
                    
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds = min(backoff_seconds * 2, 30.0)
                    continue
                break

        raise EmbeddingServiceError(f"Failed to generate embedding after {self.max_retries} attempts: {last_error}")

    async def batch_generate(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with optimized batching.
        
        Voyage AI handles rate limiting well with generous limits:
        - voyage-3.5: 8M tokens/min, 2000 requests/min
        - voyage-3.5-lite: 16M tokens/min, 2000 requests/min
        
        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings in the same order as input texts

        Raises:
            EmbeddingServiceError: If batch generation fails
        """
        if not texts:
            return []

        # Clean texts
        normalized_texts = [t or "" for t in texts]
        
        # Voyage recommends batches of 128 documents
        batch_size = self.max_batch_size
        base_delay = 0.5  # Much shorter delay due to higher rate limits
        
        all_embeddings: List[List[float]] = []
        
        self.logger.info(f"Generating embeddings for {len(normalized_texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(normalized_texts), batch_size):
            batch = normalized_texts[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(normalized_texts) + batch_size - 1)//batch_size
            
            self.logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            # Process this batch
            batch_embeddings = await self._process_embedding_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay between batches (Voyage has much higher rate limits)
            if i + batch_size < len(normalized_texts):
                await asyncio.sleep(base_delay)
        
        self.logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings

    async def _process_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Process a single batch of embedding requests with retries.

        Args:
            texts: List of texts to embed (already batched)

        Returns:
            List of embedding vectors for this batch

        Raises:
            EmbeddingServiceError: If batch processing fails after retries
        """
        attempt = 0
        backoff_seconds = 1.0
        last_error: Optional[BaseException] = None
        
        while attempt < self.max_retries:
            attempt += 1
            try:
                # Use synchronous client in async context
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.embed(
                        texts=texts,
                        model=self.model,
                        input_type="document",  # Optimize for document retrieval
                        truncation=True,  # Auto-truncate if needed
                        output_dimension=self.dimensions,
                    )
                )
                
                # Extract embeddings from response
                if response.embeddings:
                    self.logger.info(f"Successfully processed batch of {len(response.embeddings)} embeddings")
                    return response.embeddings
                else:
                    raise EmbeddingServiceError("No embeddings returned from Voyage AI")
                    
            except Exception as error:
                last_error = error
                error_str = str(error).lower()
                
                # Check if this is a rate limit error (less likely with Voyage's high limits)
                is_rate_error = any(keyword in error_str for keyword in [
                    'rate limit', 'too many requests', '429'
                ])
                
                if is_rate_error:
                    # Longer wait for rate errors
                    wait_time = 10.0  # Voyage has generous rate limits
                    self.logger.warning(f"Rate limit detected. Waiting {wait_time}s before retry: {error}")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Regular retries for other errors
                if attempt < self.max_retries:
                    wait_time = backoff_seconds + (backoff_seconds * 0.1 * attempt)  # Add jitter
                    self.logger.warning(f"Embedding batch failed (attempt {attempt}/{self.max_retries}), retrying in {wait_time:.1f}s: {error}")
                    await asyncio.sleep(wait_time)
                    backoff_seconds = min(backoff_seconds * 2, 30.0)
                    continue
                break

        raise EmbeddingServiceError(f"Failed to process embedding batch after {self.max_retries} attempts: {last_error}")

    async def generate_with_cache(self, text: str) -> 'EmbeddingResult':
        """
        Generate embedding with caching support.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding and metadata
        """
        start = time.perf_counter()
        cache_key = self._generate_cache_key(text)

        if cache_key in self._cache:
            self._cache_hits += 1
            embedding = self._cache[cache_key]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                cached=True,
                generation_time_ms=elapsed_ms,
            )

        self._cache_misses += 1
        embedding = await self.generate_embedding(text)
        # Store in cache
        self._cache[cache_key] = embedding
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model,
            dimensions=len(embedding),
            cached=False,
            generation_time_ms=elapsed_ms,
        )

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Note: Voyage embeddings are normalized to length 1, so dot product equals cosine similarity.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1 (1 = identical)
        """
        vec1 = np.array(embedding1, dtype=np.float32)
        vec2 = np.array(embedding2, dtype=np.float32)
        
        # Voyage embeddings are normalized, so we can use dot product directly
        # But we'll keep the full calculation for compatibility
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denom == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / denom)

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        threshold: float = 0.7,
    ) -> List[tuple]:
        """
        Find embeddings most similar to query.

        Args:
            query_embedding: The embedding to compare against
            candidate_embeddings: List of embeddings to search
            threshold: Minimum similarity score to include

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        results: List[tuple] = []
        for idx, candidate in enumerate(candidate_embeddings):
            score = self.calculate_similarity(query_embedding, candidate)
            if score >= threshold:
                results.append((idx, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _generate_cache_key(self, text: str) -> str:
        """Generate deterministic cache key for text"""
        normalized = (text or "").lower().strip()[:2000]
        return hashlib.md5(normalized.encode()).hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "model": self.model,
            "dimensions": self.dimensions,
        }

    def serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding for database storage"""
        # Preserve Python float precision and exact ordering for round-trip equality
        return pickle.dumps(list(embedding))

    @staticmethod
    def deserialize_embedding(data: bytes) -> List[float]:
        """Deserialize embedding from database"""
        loaded = pickle.loads(data)
        # Ensure we always return a list of floats
        return list(loaded)


class EmbeddingServiceError(Exception):
    """Custom exception for embedding service failures"""
    pass