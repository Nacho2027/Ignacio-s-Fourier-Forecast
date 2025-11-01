"""
Exa Websets Content Adapter

Adapter that wraps ExaWebsetsClient to provide a unified interface
compatible with ContentAggregator. This replaces Perplexity Search API
with Exa Websets for structured, high-quality content retrieval.

This adapter implements the same interface as PerplexitySearchAdapter,
AINewsContentAdapter, and RSSContentAdapter for seamless integration.
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.services.exa_websets_client import ExaWebsetsClient
from src.utils.exa_csv_parser import parse_exa_items
from src.services.cache_service import ContentItem


logger = logging.getLogger(__name__)


@dataclass
class ExaAdapterResult:
    """Search result compatible with ContentAggregator expectations."""
    articles: List[Dict[str, Any]]
    total_count: int
    source: str = "exa_websets"
    search_time_ms: float = 0.0
    section: str = ""


class ExaWebsetsAdapter:
    """
    Content adapter using Exa Websets API.
    
    Provides the same interface as other content adapters but uses
    Exa Websets for structured, verified content retrieval.
    
    Features:
    - Structured CSV data format
    - Native verification against criteria
    - Section-specific content gathering
    - Automatic quota fulfillment
    - Compatible with existing ContentAggregator pipeline
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        prompts_path: str = "config/exa_prompts.json",
        settings_path: str = "config/exa_settings.json",
    ):
        """
        Initialize Exa Websets adapter.

        Args:
            api_key: Exa API key (defaults to EXA_API_KEY env var)
            prompts_path: Path to prompts configuration file (full API format with criteria)
            settings_path: Path to settings configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Get API key
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY is required for Exa Websets adapter")

        # Load configuration
        self.prompts = self._load_prompts(prompts_path)
        self.settings = self._load_settings(settings_path)

        # Build section lookup from websets array
        self.section_configs = self._build_section_lookup()

        # Initialize Exa client
        self.client = ExaWebsetsClient(
            api_key=self.api_key,
            timeout_seconds=self.settings.get("search_timeout_seconds", 300),
            max_retries=self.settings.get("max_retries", 3),
            retry_backoff_base=self.settings.get("retry_backoff_base", 2),
        )

        # Concurrent webset limit (Exa API plan limit)
        self.max_concurrent_websets = self.settings.get("max_concurrent_websets", 3)
        self._webset_semaphore = asyncio.Semaphore(self.max_concurrent_websets)

        # Metrics tracking
        self.metrics = {
            'total_searches': 0,
            'total_articles': 0,
            'searches_by_section': {},
            'errors': 0,
            'concurrent_websets_used': 0,
            'max_concurrent_websets_reached': 0,
        }

        self.logger.info(
            f"Exa Websets adapter initialized with max {self.max_concurrent_websets} "
            f"concurrent websets"
        )
    
    def _load_prompts(self, path: str) -> Dict[str, Any]:
        """Load prompts configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                prompts = json.load(f)
            self.logger.info(f"Loaded prompts from {path}")
            return prompts
        except Exception as e:
            self.logger.error(f"Failed to load prompts from {path}: {e}")
            raise
    
    def _load_settings(self, path: str) -> Dict[str, Any]:
        """Load settings configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                settings = json.load(f)
            self.logger.info(f"Loaded settings from {path}")
            return settings
        except Exception as e:
            self.logger.warning(f"Failed to load settings from {path}: {e}, using defaults")
            return {}

    def _build_section_lookup(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a lookup dictionary from websets array to section configs.

        Converts the v2 format (websets array) to a section-keyed dictionary
        for backward compatibility with the adapter interface.

        Returns:
            Dictionary mapping section names to their configurations
        """
        section_lookup = {}

        websets = self.prompts.get("websets", [])
        for webset in websets:
            section_name = webset.get("name")
            if section_name:
                section_lookup[section_name] = {
                    "entity": webset.get("entity", {"type": "article"}),
                    "target_count": webset.get("target_count", 5),
                    "searches": webset.get("searches", [])
                }

        self.logger.info(f"Built section lookup for {len(section_lookup)} sections")
        return section_lookup

    def _inject_absolute_dates(
        self,
        criteria: List[Dict[str, str]],
        query: str,
        section: str
    ) -> tuple[List[Dict[str, str]], str]:
        """
        Inject absolute date criteria and clean temporal references from query.

        Instead of replacing "past 48 hours" with "on or after October 20, 2025",
        we remove temporal phrases from criteria/query and add a separate date criterion.

        This prevents AI hallucination by using structured date constraints.

        Args:
            criteria: List of criteria dicts with description fields
            query: Search query string
            section: Newsletter section (determines lookback period)

        Returns:
            Tuple of (updated_criteria, updated_query)
        """
        import re

        now = datetime.now()

        # Define lookback periods by section (in days)
        # Relaxed slightly to improve item yield while maintaining recency
        section_lookback = {
            "breaking_news": 3,      # 72 hours (was 48)
            "business": 4,           # 96 hours (was 48)
            "tech_science": 5,       # 5 days (was 48 hours)
            "politics": 3,           # 72 hours (was 48)
            "miscellaneous": 14,     # 2 weeks for quality content (was 7 days)
            "research_papers": 45,   # 1.5 months for papers (was 30 days)
        }

        lookback_days = section_lookback.get(section, 3)  # Default 3 days (was 2)
        cutoff_date = now - timedelta(days=lookback_days)
        formatted_date = cutoff_date.strftime("%B %d, %Y")

        # Temporal phrases to remove from criteria/query
        temporal_phrases = [
            "past 24 hours", "last 24 hours",
            "past 48 hours", "last 48 hours",
            "past 3 days", "last 3 days",
            "past week", "last week",
            "past month", "last month",
            "in the last 24 hours", "in the past 24 hours",
            "in the last 48 hours", "in the past 48 hours",
        ]

        # Clean criteria - remove temporal phrases
        updated_criteria = []
        has_temporal_criterion = False

        for criterion in criteria:
            description = criterion.get("description", "")
            original_description = description

            # Check if this is a temporal criterion
            is_temporal = any(phrase in description.lower() for phrase in temporal_phrases)

            if is_temporal:
                has_temporal_criterion = True
                # Skip this criterion - we'll add our own date criterion
                self.logger.debug(f"Removing temporal criterion: '{description}'")
                continue

            updated_criteria.append(criterion)

        # Exa API limit: maximum 5 criteria
        # If we already have 5 criteria, we need to make room for the date criterion
        if len(updated_criteria) >= 5:
            self.logger.warning(
                f"Section '{section}' has {len(updated_criteria)} criteria. "
                f"Exa API allows max 5. Removing last criterion to make room for date criterion."
            )
            updated_criteria = updated_criteria[:4]  # Keep only first 4

        # Add absolute date criterion
        updated_criteria.append({
            "description": f"Article was published on or after {formatted_date}"
        })

        self.logger.debug(
            f"Added date criterion for {section}: published on or after {formatted_date} "
            f"({lookback_days} days lookback). Total criteria: {len(updated_criteria)}"
        )

        # Clean query - remove temporal phrases
        updated_query = query
        for phrase in temporal_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            updated_query = pattern.sub("", updated_query)

        # Clean up extra spaces
        updated_query = re.sub(r'\s+', ' ', updated_query).strip()

        if updated_query != query:
            self.logger.debug(f"Cleaned temporal phrases from query")

        return updated_criteria, updated_query

    async def search_optimized_rate_limited(
        self,
        section: str,
        custom_query: Optional[str] = None
    ) -> ExaAdapterResult:
        """
        Search for content in a specific section.
        
        This method provides the same interface as other adapters for compatibility
        with ContentAggregator.
        
        Args:
            section: Newsletter section (breaking_news, business, tech_science, 
                    research_papers, politics, miscellaneous)
            custom_query: Optional custom query (overrides section-specific queries)
        
        Returns:
            ExaAdapterResult with articles in ContentAggregator-compatible format
        
        Raises:
            ValueError: If section is unknown
            Exception: If search fails
        """
        start_time = asyncio.get_event_loop().time()
        self.logger.info(f"Searching section '{section}' using Exa Websets API")
        
        try:
            # Gather content for section
            content_items = await self._gather_section_content(section, custom_query)
            
            # Convert ContentItems to dict format expected by ContentAggregator
            articles = self._convert_to_article_dicts(content_items)
            
            # Update metrics
            self.metrics['total_searches'] += 1
            self.metrics['total_articles'] += len(articles)
            self.metrics['searches_by_section'][section] = \
                self.metrics['searches_by_section'].get(section, 0) + 1
            
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
            
            self.logger.info(
                f"Section '{section}': Found {len(articles)} articles "
                f"via Exa Websets API in {elapsed_ms:.1f}ms"
            )
            
            return ExaAdapterResult(
                articles=articles,
                total_count=len(articles),
                source="exa_websets",
                search_time_ms=elapsed_ms,
                section=section,
            )
        
        except Exception as e:
            self.metrics['errors'] += 1
            self.logger.error(f"Search failed for section '{section}': {e}")
            raise
    
    async def _gather_section_content(
        self,
        section: str,
        custom_query: Optional[str] = None
    ) -> List[ContentItem]:
        """
        Gather content for a specific section using Exa Websets.

        Args:
            section: Newsletter section
            custom_query: Optional custom query (overrides section queries)

        Returns:
            List of ContentItem objects

        Raises:
            ValueError: If section is unknown
        """
        # Get section configuration from lookup
        section_config = self.section_configs.get(section)
        if not section_config:
            raise ValueError(
                f"Unknown section '{section}'. "
                f"Supported sections: {list(self.section_configs.keys())}"
            )

        target_count = section_config.get("target_count", 5)
        entity = section_config.get("entity", {"type": "article"})
        entity_type = entity.get("type", "article")
        searches = section_config.get("searches", [])

        if custom_query:
            # Use custom query instead of section searches
            searches = [{
                "search": {
                    "query": custom_query,
                    "entity": entity,
                    "count": target_count
                },
                "criteria": []
            }]

        if not searches:
            raise ValueError(f"No searches configured for section '{section}'")
        
        self.logger.info(
            f"🎯 Gathering {target_count} items for section '{section}' "
            f"using {len(searches)} searches"
        )

        # Log search details
        for idx, search_config in enumerate(searches, 1):
            search_query = search_config.get("search", {}).get("query", "")[:100]
            search_count = search_config.get("search", {}).get("count", 0)
            criteria_count = len(search_config.get("criteria", []))
            self.logger.info(
                f"  Search {idx}/{len(searches)}: count={search_count}, "
                f"criteria={criteria_count}, query='{search_query}...'"
            )

        # Execute searches with intelligent concurrent management
        all_items = []

        # CRITICAL: Exa Websets API does NOT support concurrent searches on the same webset
        # From docs: "Can run in parallel with other enrichment operations (not other searches for now)"
        # Therefore, we MUST run searches sequentially within each section
        parallel_search_execution = self.settings.get("parallel_search_execution", False)

        if parallel_search_execution:
            # WARNING: This mode violates Exa's API design and will cause searches to hang!
            # Only enable if Exa updates their API to support concurrent searches per webset
            self.logger.warning(
                f"⚠️  PARALLEL SEARCH MODE ENABLED - This may cause searches to hang! "
                f"Exa does not support concurrent searches on the same webset."
            )
            self.logger.info(
                f"🚀 Executing {len(searches)} searches with max "
                f"{self.max_concurrent_websets} concurrent websets"
            )

            tasks = [
                self._execute_search_with_limit(
                    search_config["search"],
                    search_config.get("criteria", []),
                    section,
                    idx + 1,
                    len(searches)
                )
                for idx, search_config in enumerate(searches)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Search failed: {result}")
                else:
                    all_items.extend(result)
        else:
            # Sequential execution - REQUIRED by Exa API design
            self.logger.info(
                f"🔄 Executing {len(searches)} searches SEQUENTIALLY "
                f"(required by Exa Websets API - concurrent searches per webset not supported)"
            )
            for idx, search_config in enumerate(searches, 1):
                try:
                    self.logger.info(f"  🔹 Starting search {idx}/{len(searches)}...")
                    items = await self._execute_search(
                        search_config["search"],
                        search_config.get("criteria", []),
                        section
                    )
                    all_items.extend(items)
                    self.logger.info(
                        f"  ✅ Search {idx}/{len(searches)} complete: "
                        f"{len(items)} items retrieved"
                    )
                except Exception as e:
                    self.logger.error(
                        f"  ❌ Search {idx}/{len(searches)} failed: {e}",
                        exc_info=True
                    )

        self.logger.info(
            f"Section '{section}': Gathered {len(all_items)} total items "
            f"from {len(searches)} searches"
        )

        return all_items

    async def _execute_search_with_limit(
        self,
        search_config: Dict[str, Any],
        criteria: List[Dict[str, str]],
        section: str,
        search_num: int,
        total_searches: int
    ) -> List[ContentItem]:
        """
        Execute a single Exa search with concurrent webset limit enforcement.

        Uses a semaphore to ensure we never exceed the API's concurrent webset limit.
        When the limit is reached, searches wait in queue until a slot opens.

        Args:
            search_config: Search configuration dict with query, entity, count
            criteria: List of criteria dicts with description fields
            section: Newsletter section
            search_num: Current search number (for logging)
            total_searches: Total number of searches (for logging)

        Returns:
            List of ContentItem objects
        """
        # Wait for available webset slot
        async with self._webset_semaphore:
            # Track concurrent usage
            active_websets = self.max_concurrent_websets - self._webset_semaphore._value
            self.metrics['concurrent_websets_used'] = max(
                self.metrics['concurrent_websets_used'],
                active_websets
            )
            if active_websets == self.max_concurrent_websets:
                self.metrics['max_concurrent_websets_reached'] += 1

            self.logger.info(
                f"  🔹 Search {search_num}/{total_searches} starting "
                f"(active websets: {active_websets}/{self.max_concurrent_websets})"
            )

            # Execute the search
            result = await self._execute_search(search_config, criteria, section)

            self.logger.info(
                f"  ✅ Search {search_num}/{total_searches} complete: "
                f"{len(result)} items (slot freed)"
            )

            return result

    async def _execute_search(
        self,
        search_config: Dict[str, Any],
        criteria: List[Dict[str, str]],
        section: str
    ) -> List[ContentItem]:
        """
        Execute a single Exa search with criteria.

        Args:
            search_config: Search configuration dict with query, entity, count
            criteria: List of criteria dicts with description fields
            section: Newsletter section

        Returns:
            List of ContentItem objects
        """
        webset_id = None

        try:
            # Extract search parameters
            query = search_config.get("query", "")
            count = search_config.get("count", 10)
            entity = search_config.get("entity", {"type": "article"})
            entity_type = entity.get("type", "article")

            # Inject absolute dates into criteria and query
            updated_criteria, updated_query = self._inject_absolute_dates(
                criteria if criteria else [],
                query,
                section
            )

            self.logger.debug(
                f"Original query: {query[:100]}..."
            )
            self.logger.debug(
                f"Updated query: {updated_query[:100]}..."
            )

            # Prepare enrichments for articles (skip for research papers which have abstracts)
            enrichments = None
            if entity_type == "article":
                enrichments = [
                    {
                        "description": (
                            "Extract a comprehensive 200-300 word summary of the article content, "
                            "focusing on key insights, main arguments, important details, and actionable takeaways. "
                            "Prioritize factual accuracy and intellectual depth over sensationalism."
                        ),
                        "format": "text",
                    }
                ]
                self.logger.info(f"🔧 Will create enrichment for article summaries during webset creation")

            # Create search with updated criteria, query, and enrichments
            result = await self.client.create_search(
                query=updated_query,
                count=count,
                entity_type=entity_type,
                criteria=updated_criteria if updated_criteria else None,
                enrichments=enrichments,
            )

            # Store webset_id for cleanup in finally block
            if result and result.webset_id:
                webset_id = result.webset_id

            if not result.success:
                self.logger.warning(
                    f"Search failed: {result.error_message}"
                )
                return []

            # Enrichments are now created during webset creation, no need for separate API call
            self.logger.info(f"✅ Webset {result.webset_id} created with enrichments (if applicable)")

            # Retrieve items (enrichment data will be included if enrichments were created)
            items = await self.client.get_items(result.webset_id)
            self.logger.debug(f"🔍 Exa returned {len(items)} raw items for section '{section}'")

            # Parse to ContentItems
            content_items = parse_exa_items(items, section)
            self.logger.debug(f"📝 Parsed {len(content_items)} ContentItems from {len(items)} raw items")

            return content_items

        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return []

        finally:
            # ALWAYS cleanup webset, even on error - this prevents orphaned websets
            if webset_id and self.settings.get("cleanup_websets_after_fetch", True):
                try:
                    await self.client.cleanup_webset(webset_id)
                    self.logger.debug(f"🗑️  Webset {webset_id} deleted (slot freed)")
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Failed to cleanup webset {webset_id}: {cleanup_error}"
                    )
    
    def _convert_to_article_dicts(
        self,
        content_items: List[ContentItem]
    ) -> List[Dict[str, Any]]:
        """
        Convert ContentItem objects to dict format expected by ContentAggregator.
        
        Args:
            content_items: List of ContentItem objects
        
        Returns:
            List of article dictionaries
        """
        articles = []
        
        for item in content_items:
            # CRITICAL FIX: Extract enrichment_summary from metadata to top-level field
            # This ensures the content aggregator can access it via item.get("enrichment_summary")
            enrichment_summary = item.metadata.get('enrichment_summary') if item.metadata else None

            article = {
                'id': item.id,
                'url': item.url,
                'headline': item.headline,
                'title': item.headline,  # Alias for compatibility
                'summary_text': item.summary_text,
                'content': item.summary_text,  # Alias for compatibility
                'enrichment_summary': enrichment_summary,  # CRITICAL: Top-level field for aggregator
                'abstract': item.metadata.get('abstract') if item.metadata else None,  # Also extract abstract
                'source': item.source,
                'section': item.section,
                'published_date': item.published_date,
                'date': item.published_date.isoformat() if item.published_date else None,
                'metadata': item.metadata or {},
                'source_type': 'exa_websets'
            }

            articles.append(article)
        
        return articles
    
    async def cleanup(self):
        """Clean up all created Websets."""
        deleted = await self.client.cleanup_all()
        self.logger.info(f"Cleaned up {deleted} websets")
        return deleted
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return self.metrics.copy()

