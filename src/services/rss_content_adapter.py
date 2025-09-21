"""
RSS Content Adapter

This service provides a drop-in replacement for LLMLayer functionality
using RSS feeds. It maintains the same interface as LLMLayerService
but retrieves content from RSS feeds instead of web search.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.services.rss import RSSService


@dataclass 
class RSSAdapterResult:
    """RSS adapter result that mimics LLMLayerResult interface"""
    query: str
    articles: List[Dict[str, Any]]
    search_time_ms: float
    total_results: int
    cached: bool = False
    section: str = ""


class RSSContentAdapter:
    """
    Adapter service that replaces LLMLayer with RSS feed content retrieval.
    Provides the same interface as LLMLayerService for drop-in replacement.
    """
    
    def __init__(self, rss_service: RSSService):
        self.rss_service = rss_service
        self.logger = logging.getLogger(__name__)
        
        # Section-specific configurations
        self.section_configs = self._init_section_configs()
        
        self.logger.info("RSS Content Adapter initialized")
        
    def _init_section_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize section-specific configurations."""
        return {
            "breaking_news": {
                "target_count": 10,
                "max_feeds": 6, 
                "hours_back": 24,
                "quality_threshold": 0.6,
            },
            "business": {
                "target_count": 8,
                "max_feeds": 6,
                "hours_back": 48,
                "quality_threshold": 0.7,
            },
            "tech_science": {
                "target_count": 8,
                "max_feeds": 6,
                "hours_back": 168,  # 1 week
                "quality_threshold": 0.7,
            },
            "politics": {
                "target_count": 6,
                "max_feeds": 5,
                "hours_back": 48,
                "quality_threshold": 0.6,
            },
            "miscellaneous": {
                "target_count": 20,  # Get more for filtering
                "max_feeds": 8,
                "hours_back": 336,  # 2 weeks
                "quality_threshold": 0.8,
            },
            "research_papers": {
                "target_count": 15,
                "max_feeds": 6,
                "hours_back": 720,  # 30 days
                "quality_threshold": 0.8,
            },
        }
    
    async def search_optimized_rate_limited(self, section: str, custom_query: Optional[str] = None) -> RSSAdapterResult:
        """
        Main method that replaces LLMLayerService.search_optimized_rate_limited().
        Retrieves content from RSS feeds for the specified section.
        """
        start_time = asyncio.get_event_loop().time()
        
        # Get configuration for this section
        config = self.section_configs.get(section, {
            "target_count": 10,
            "max_feeds": 5, 
            "hours_back": 168,
            "quality_threshold": 0.7,
        })
        
        self.logger.info(f"Fetching RSS content for section: {section}")
        
        try:
            # Fetch content using RSS service
            articles = await self.rss_service.fetch_section_content(
                section=section,
                target_count=config["target_count"],
                max_feeds=config["max_feeds"],
                hours_back=config["hours_back"]
            )
            
            # Apply additional quality filtering
            filtered_articles = self._apply_quality_filtering(
                articles, 
                config["quality_threshold"]
            )
            
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
            
            result = RSSAdapterResult(
                query=custom_query or f"{section} content",
                articles=filtered_articles,
                search_time_ms=elapsed_ms,
                total_results=len(filtered_articles),
                cached=False,
                section=section
            )
            
            self.logger.info(f"RSS adapter for {section}: {len(filtered_articles)} articles in {elapsed_ms:.1f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"RSS adapter failed for {section}: {e}")
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
            
            # Return empty result on failure
            return RSSAdapterResult(
                query=custom_query or f"{section} content",
                articles=[],
                search_time_ms=elapsed_ms,
                total_results=0,
                cached=False,
                section=section
            )
    
    def _apply_quality_filtering(self, articles: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Apply additional quality filtering to articles."""
        if not articles:
            return articles
            
        filtered = []
        for article in articles:
            # Check priority score against threshold
            priority_score = article.get('priority_score', 0)
            if priority_score >= threshold:
                filtered.append(article)
                
        # Sort by priority score and recency
        filtered.sort(key=lambda x: (
            x.get('priority_score', 0),
            x.get('published', '1970-01-01')
        ), reverse=True)
        
        return filtered
        
    async def search_with_fallback(self, section: str, base_query: str, target_count: int) -> RSSAdapterResult:
        """
        Alternative search method with fallback capabilities.
        Used by some sections that need more flexible content retrieval.
        """
        # For RSS adapter, we'll use the same optimized method but with custom target count
        original_config = self.section_configs.get(section, {})
        
        # Temporarily override target count
        self.section_configs[section] = {
            **original_config,
            "target_count": target_count * 2  # Get extra for selection
        }
        
        try:
            result = await self.search_optimized_rate_limited(section)
            
            # Restore original config
            if original_config:
                self.section_configs[section] = original_config
            
            # Limit to target count
            if len(result.articles) > target_count:
                result.articles = result.articles[:target_count]
                result.total_results = len(result.articles)
                
            return result
            
        except Exception as e:
            # Restore original config on error
            if original_config:
                self.section_configs[section] = original_config
            raise e
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the RSS adapter and underlying services."""
        try:
            # Get RSS service status
            rss_status = await self.rss_service.get_section_feeds_status()
            
            return {
                "status": "healthy",
                "service": "RSS Content Adapter",
                "sections_configured": len(self.section_configs),
                "rss_feeds_status": rss_status,
                "last_check": datetime.now().isoformat(),
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "RSS Content Adapter", 
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }
            
    def get_section_config(self, section: str) -> Dict[str, Any]:
        """Get configuration for a specific section."""
        return self.section_configs.get(section, {})
        
    def update_section_config(self, section: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific section."""
        if section in self.section_configs:
            self.section_configs[section].update(config)
        else:
            self.section_configs[section] = config
            
        self.logger.info(f"Updated configuration for section {section}: {config}")


class RSSAdapterError(Exception):
    """Custom exception for RSS adapter failures"""
    pass