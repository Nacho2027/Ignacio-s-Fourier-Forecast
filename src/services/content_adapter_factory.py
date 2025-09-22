#!/usr/bin/env python3
"""
Content Adapter Factory

Factory for creating content adapters (AI news adapters) with support for
multiple backends including AINewsContentAdapter and PerplexityService.
Provides a unified interface for the ContentAggregator to use different
content retrieval strategies.
"""

import os
import logging
from typing import Optional, Union
from dataclasses import dataclass

from src.services.ai_news_adapter import AINewsContentAdapter
from src.services.perplexity_newsletter_service import PerplexityNewsletterService
from src.services.perplexity_search_adapter import PerplexitySearchAdapter
from src.services.exa_websets_adapter import ExaWebsetsAdapter


@dataclass
class ContentAdapterConfig:
    """Configuration for content adapter selection and setup."""
    adapter_type: str = "exa_websets"  # "exa_websets" (primary), "perplexity_search" (API), "perplexity" (browser), or "ai_news" (legacy)
    headless: bool = True  # For browser-based adapters
    session_persist: bool = True  # For browser-based adapters
    fallback_enabled: bool = False  # Disable fallback - use primary adapter only


class ContentAdapterFactory:
    """
    Factory for creating content adapters with Exa Websets as primary.

    Supports adapter types:
    - "exa_websets": ExaWebsetsAdapter (Structured API, primary, recommended)
    - "perplexity_search": PerplexitySearchAdapter (API-based, legacy)
    - "perplexity": PerplexityNewsletterService (Browser automation, legacy)
    - "ai_news": AINewsContentAdapter (OpenRouter DeepSeek, legacy only)

    Adapter Selection Priority (create_from_environment):
    1. EXA_API_KEY present → ExaWebsetsAdapter (highest priority)
    2. CONTENT_ADAPTER_TYPE set → Use specified adapter
    3. PERPLEXITY_API_KEY present → PerplexitySearchAdapter
    4. Default → ExaWebsetsAdapter

    Configured to use Exa Websets exclusively without fallback mechanisms.
    """
    
    @staticmethod
    def create_adapter(
        config: Optional[Union[ContentAdapterConfig, str]] = None
    ) -> Union[AINewsContentAdapter, PerplexityNewsletterService, PerplexitySearchAdapter, ExaWebsetsAdapter]:
        """
        Create a content adapter based on configuration.

        Args:
            config: Configuration for adapter selection, or adapter type string

        Returns:
            Content adapter instance

        Raises:
            ValueError: If no suitable adapter can be created
        """
        if config is None:
            config = ContentAdapterConfig()
        elif isinstance(config, str):
            config = ContentAdapterConfig(adapter_type=config)

        logger = logging.getLogger(__name__)

        # Try to create the requested adapter type
        if config.adapter_type == "exa_websets":
            try:
                logger.info("Creating ExaWebsetsAdapter (structured API)")
                return ExaWebsetsAdapter()
            except Exception as e:
                logger.warning(f"Failed to create ExaWebsetsAdapter: {e}")

                if config.fallback_enabled:
                    logger.info("Falling back to PerplexitySearchAdapter (API)")
                    return PerplexitySearchAdapter()
                else:
                    raise ValueError(f"Failed to create ExaWebsetsAdapter and fallback disabled: {e}")

        elif config.adapter_type == "perplexity_search":
            try:
                logger.info("Creating PerplexitySearchAdapter (API-based)")
                return PerplexitySearchAdapter()
            except Exception as e:
                logger.warning(f"Failed to create PerplexitySearchAdapter: {e}")

                if config.fallback_enabled:
                    logger.info("Falling back to PerplexityNewsletterService (browser)")
                    return PerplexityNewsletterService(
                        headless=config.headless,
                        session_persist=config.session_persist
                    )
                else:
                    raise ValueError(f"Failed to create PerplexitySearchAdapter and fallback disabled: {e}")

        elif config.adapter_type == "perplexity":
            try:
                logger.info("Creating PerplexityNewsletterService content adapter (browser)")
                return PerplexityNewsletterService(
                    headless=config.headless,
                    session_persist=config.session_persist
                )
            except Exception as e:
                logger.warning(f"Failed to create PerplexityNewsletterService: {e}")

                if config.fallback_enabled:
                    logger.info("Falling back to AINewsContentAdapter")
                    return ContentAdapterFactory._create_ai_news_adapter()
                else:
                    raise ValueError(f"Failed to create PerplexityNewsletterService and fallback disabled: {e}")

        elif config.adapter_type == "ai_news":
            logger.info("Creating AINewsContentAdapter")
            return ContentAdapterFactory._create_ai_news_adapter()

        else:
            raise ValueError(f"Unknown adapter type: {config.adapter_type}")
    
    @staticmethod
    def _create_ai_news_adapter() -> AINewsContentAdapter:
        """Create AINewsContentAdapter with error handling."""
        try:
            return AINewsContentAdapter()
        except Exception as e:
            raise ValueError(f"Failed to create AINewsContentAdapter: {e}")
    
    @staticmethod
    def create_from_environment() -> Union[AINewsContentAdapter, PerplexityNewsletterService, PerplexitySearchAdapter, ExaWebsetsAdapter]:
        """
        Create adapter based on environment variables.

        Environment variables (priority order):
        1. EXA_API_KEY: If set, use ExaWebsetsAdapter (highest priority)
        2. CONTENT_ADAPTER_TYPE: "exa_websets", "perplexity_search", "perplexity", or "ai_news"
        3. PERPLEXITY_API_KEY: If set and no EXA_API_KEY, use PerplexitySearchAdapter
        4. Default: "exa_websets"

        Other environment variables:
        - BROWSER_HEADLESS: "true" or "false" (default: "true")
        - BROWSER_SESSION_PERSIST: "true" or "false" (default: "true")
        - ADAPTER_FALLBACK_ENABLED: "true" or "false" (default: "false")

        Returns:
            Content adapter instance
        """
        # Priority 1: If EXA_API_KEY is set, use Exa Websets regardless of CONTENT_ADAPTER_TYPE
        if os.getenv("EXA_API_KEY"):
            adapter_type = "exa_websets"
        # Priority 2: Use CONTENT_ADAPTER_TYPE if explicitly set
        elif os.getenv("CONTENT_ADAPTER_TYPE"):
            adapter_type = os.getenv("CONTENT_ADAPTER_TYPE")
        # Priority 3: If PERPLEXITY_API_KEY is set, use Perplexity Search
        elif os.getenv("PERPLEXITY_API_KEY"):
            adapter_type = "perplexity_search"
        # Priority 4: Default to Exa Websets
        else:
            adapter_type = "exa_websets"

        config = ContentAdapterConfig(
            adapter_type=adapter_type,
            headless=os.getenv("BROWSER_HEADLESS", "true").lower() == "true",
            session_persist=os.getenv("BROWSER_SESSION_PERSIST", "true").lower() == "true",
            fallback_enabled=os.getenv("ADAPTER_FALLBACK_ENABLED", "false").lower() == "true"
        )

        return ContentAdapterFactory.create_adapter(config)


class ContentAdapterWrapper:
    """
    Wrapper that provides a unified interface for different content adapters.

    This wrapper ensures that both AINewsContentAdapter and PerplexityNewsletterService
    can be used interchangeably by the ContentAggregator, handling any
    interface differences and providing consistent error handling.
    """

    def __init__(self, adapter: Union[AINewsContentAdapter, PerplexityNewsletterService, PerplexitySearchAdapter, ExaWebsetsAdapter]):
        """
        Initialize wrapper with a content adapter.

        Args:
            adapter: Content adapter instance
        """
        self.adapter = adapter
        self.logger = logging.getLogger(__name__)
        self.adapter_type = type(adapter).__name__

        # Track if this is a browser-based adapter that needs lifecycle management
        # API-based adapters (PerplexitySearchAdapter, ExaWebsetsAdapter) don't need lifecycle management
        # Check for PerplexityService by name or actual type to support mocking
        adapter_name = type(adapter).__name__
        self.needs_lifecycle_management = (
            adapter_name == 'PerplexityService' or
            (hasattr(adapter, 'base_url') and
             hasattr(adapter, 'browser_service') and
             hasattr(adapter, 'initialize') and
             hasattr(adapter, 'cleanup') and
             callable(getattr(adapter, 'initialize', None)) and
             callable(getattr(adapter, 'cleanup', None)) and
             not adapter_name.startswith('Mock') and  # Exclude mock objects
             adapter_name not in ['PerplexitySearchAdapter', 'ExaWebsetsAdapter'])  # Exclude API adapters
        )
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the adapter if needed."""
        if self.needs_lifecycle_management and not self._initialized:
            try:
                await self.adapter.initialize()
                self._initialized = True
                self.logger.info(f"Initialized {self.adapter_type}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {self.adapter_type}: {e}")
                raise
    
    async def cleanup(self) -> None:
        """Clean up adapter resources if needed."""
        if self.needs_lifecycle_management and self._initialized:
            try:
                await self.adapter.cleanup()
                self._initialized = False
                self.logger.info(f"Cleaned up {self.adapter_type}")
            except Exception as e:
                self.logger.warning(f"Error during {self.adapter_type} cleanup: {e}")
    
    async def search_optimized_rate_limited(self, section: str, custom_query: Optional[str] = None):
        """
        Unified search method that works with both adapter types.
        
        Args:
            section: Newsletter section
            custom_query: Optional custom query
            
        Returns:
            Search result with articles
        """
        # Ensure adapter is initialized
        if self.needs_lifecycle_management and not self._initialized:
            await self.initialize()
        
        try:
            result = await self.adapter.search_optimized_rate_limited(section, custom_query)
            self.logger.debug(f"{self.adapter_type} search for '{section}': {len(result.articles)} articles")
            return result
        except Exception as e:
            self.logger.error(f"{self.adapter_type} search failed for section '{section}': {e}")

            # No fallback - Perplexity-only configuration
            # Fallback to AI news adapter has been disabled to ensure exclusive use of Perplexity
            self.logger.error(f"❌ {self.adapter_type} failed for section '{section}' - no fallback configured")
            raise
    
    def get_metrics(self) -> dict:
        """Get adapter metrics."""
        base_metrics = {
            "adapter_type": self.adapter_type,
            "initialized": self._initialized,
            "needs_lifecycle_management": self.needs_lifecycle_management
        }
        
        # Get adapter-specific metrics if available
        if hasattr(self.adapter, 'get_metrics'):
            try:
                adapter_metrics = self.adapter.get_metrics()
                base_metrics.update(adapter_metrics)
            except Exception as e:
                self.logger.warning(f"Failed to get {self.adapter_type} metrics: {e}")
        
        return base_metrics
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
