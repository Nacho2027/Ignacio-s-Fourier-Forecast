"""
Adaptive Search Manager for Intelligent Fallback Strategies

This module provides intelligent search orchestration that expands search breadth
rather than compromising on content recency for the daily newsletter.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from src.services.llmlayer import LLMLayerService, LLMLayerResult


class Section:
    """Newsletter sections as string constants (matching content_aggregator)"""
    SCRIPTURE = "scripture"
    BREAKING_NEWS = "breaking_news"
    BUSINESS = "business"
    TECH_SCIENCE = "tech_science"
    RESEARCH_PAPERS = "research_papers"
    STARTUP = "startup"
    POLITICS = "politics"
    LOCAL = "local"
    MISCELLANEOUS = "miscellaneous"
    EXTRA = "extra"


class SearchStrategy(Enum):
    """Search strategy types for progressive fallback"""
    PRIMARY = "primary"          # Original targeted search
    EXPANDED = "expanded"        # Broader keywords, same sources
    CROSS_DOMAIN = "cross_domain"  # Remove domain restrictions
    SEMANTIC = "semantic"        # Alternative keyword variations
    SPILLOVER = "spillover"      # Cross-section content borrowing


@dataclass
class SearchConfig:
    """Configuration for a search strategy"""
    strategy: SearchStrategy
    query_variations: List[str]
    max_results: int
    domains: Optional[List[str]] = None
    recency: str = "day"
    search_type: str = "news"
    quality_threshold: float = 60.0
    description: str = ""


@dataclass
class SearchResult:
    """Result from a search attempt"""
    strategy: SearchStrategy
    success: bool
    articles_found: int
    articles: List[Dict[str, Any]]
    execution_time: float
    error: Optional[str] = None


class AdaptiveSearchManager:
    """
    Orchestrates intelligent search fallback strategies.

    Instead of relaxing temporal constraints, this manager:
    1. Expands search terms and keyword variations
    2. Progressively removes domain restrictions
    3. Implements cross-section content sharing
    4. Uses semantic alternatives for failed searches
    """

    def __init__(self, llmlayer: LLMLayerService):
        self.llmlayer = llmlayer
        self.logger = logging.getLogger(__name__)

        # Section-specific temporal limits (maintain daily newsletter recency)
        self.section_max_age = {
            Section.BREAKING_NEWS: 2,      # Very recent
            Section.BUSINESS: 3,           # Recent business news
            Section.POLITICS: 3,           # Recent political developments
            Section.TECH_SCIENCE: 7,       # Tech can be slightly older
            Section.MISCELLANEOUS: 14,     # Intellectual content has longer shelf life
            Section.RESEARCH_PAPERS: 30,   # Academic publishing cycles
            Section.LOCAL: 7,              # Local news recency
            Section.STARTUP: 7             # Startup ecosystem news
        }

        # Cross-section borrowing relationships
        self.section_spillover = {
            Section.BREAKING_NEWS: [Section.BUSINESS, Section.POLITICS, Section.TECH_SCIENCE],
            Section.BUSINESS: [Section.TECH_SCIENCE, Section.STARTUP],
            Section.TECH_SCIENCE: [Section.RESEARCH_PAPERS, Section.STARTUP],
            Section.POLITICS: [Section.BUSINESS],
            Section.STARTUP: [Section.BUSINESS, Section.TECH_SCIENCE],
        }

        self.logger.info("AdaptiveSearchManager initialized with smart fallback strategies")

    def _generate_search_configs(self, section: Section, base_query: str, target_count: int) -> List[SearchConfig]:
        """Generate progressive search configurations for a section"""
        configs = []
        today = datetime.now().strftime("%B %d, %Y")

        # Get section-specific settings
        max_age = self.section_max_age.get(section, 7)
        recency = "day" if max_age <= 3 else "week"

        if section == Section.BREAKING_NEWS:
            configs = [
                SearchConfig(
                    strategy=SearchStrategy.PRIMARY,
                    query_variations=[
                        f"breaking news today {today} latest urgent developing",
                        f"major news headlines {today} just announced important"
                    ],
                    max_results=target_count * 3,
                    domains=["apnews.com", "reuters.com", "cnn.com", "bbc.com"],
                    recency="day",
                    quality_threshold=70.0,
                    description="Premium breaking news sources"
                ),
                SearchConfig(
                    strategy=SearchStrategy.EXPANDED,
                    query_variations=[
                        f"significant news events {today} worldwide developments",
                        f"important announcements {today} global coverage"
                    ],
                    max_results=target_count * 4,
                    domains=["apnews.com", "reuters.com", "cnn.com", "bbc.com", "npr.org", "pbs.org"],
                    recency="day",
                    quality_threshold=65.0,
                    description="Expanded breaking news with more sources"
                ),
                SearchConfig(
                    strategy=SearchStrategy.CROSS_DOMAIN,
                    query_variations=[
                        f"breaking news {today} latest developments",
                        f"major headlines {today} significant events"
                    ],
                    max_results=target_count * 5,
                    domains=None,  # No domain restrictions
                    recency="day",
                    quality_threshold=60.0,
                    description="Open domain breaking news search"
                )
            ]

        elif section == Section.BUSINESS:
            configs = [
                SearchConfig(
                    strategy=SearchStrategy.PRIMARY,
                    query_variations=[
                        f"business market news {today} stocks economy earnings",
                        f"corporate news {today} financial markets business deals"
                    ],
                    max_results=target_count * 3,
                    domains=["wsj.com", "ft.com", "bloomberg.com", "axios.com"],
                    recency=recency,
                    quality_threshold=70.0,
                    description="Premium financial sources"
                ),
                SearchConfig(
                    strategy=SearchStrategy.EXPANDED,
                    query_variations=[
                        f"business developments {today} market analysis industry",
                        f"economic news {today} corporate strategy business trends"
                    ],
                    max_results=target_count * 4,
                    domains=["wsj.com", "ft.com", "bloomberg.com", "axios.com", "cnbc.com", "marketwatch.com"],
                    recency=recency,
                    quality_threshold=65.0,
                    description="Expanded business sources"
                ),
                SearchConfig(
                    strategy=SearchStrategy.CROSS_DOMAIN,
                    query_variations=[
                        f"business news {today} market economy",
                        f"financial news {today} corporate business"
                    ],
                    max_results=target_count * 5,
                    domains=None,
                    recency=recency,
                    quality_threshold=60.0,
                    description="Open domain business search"
                )
            ]

        elif section == Section.TECH_SCIENCE:
            configs = [
                SearchConfig(
                    strategy=SearchStrategy.PRIMARY,
                    query_variations=[
                        f"technology science breakthroughs {today} AI research",
                        f"scientific discoveries {today} tech innovations breakthrough"
                    ],
                    max_results=target_count * 3,
                    domains=["technologyreview.com", "arstechnica.com", "wired.com", "nature.com", "science.org"],
                    recency=recency,
                    quality_threshold=70.0,
                    description="Premium tech and science sources"
                ),
                SearchConfig(
                    strategy=SearchStrategy.SEMANTIC,
                    query_variations=[
                        f"artificial intelligence machine learning {today} advances",
                        f"quantum computing biotechnology {today} innovation",
                        f"scientific research {today} technological progress"
                    ],
                    max_results=target_count * 4,
                    domains=["technologyreview.com", "arstechnica.com", "wired.com", "nature.com", "science.org", "quantamagazine.org"],
                    recency=recency,
                    quality_threshold=65.0,
                    description="Semantic variations for tech/science"
                ),
                SearchConfig(
                    strategy=SearchStrategy.CROSS_DOMAIN,
                    query_variations=[
                        f"technology science {today} latest research",
                        f"innovation breakthrough {today} scientific"
                    ],
                    max_results=target_count * 5,
                    domains=None,
                    recency=recency,
                    quality_threshold=60.0,
                    description="Open domain tech/science search"
                )
            ]

        elif section == Section.MISCELLANEOUS:
            configs = [
                SearchConfig(
                    strategy=SearchStrategy.PRIMARY,
                    query_variations=[
                        f"intellectual essays philosophy culture {today}",
                        f"thought-provoking analysis {today} cultural commentary"
                    ],
                    max_results=target_count * 4,
                    domains=["theatlantic.com", "newyorker.com", "aeon.co", "harpers.org"],
                    recency="week",  # Intellectual content can be slightly older
                    quality_threshold=70.0,
                    description="Premium intellectual sources"
                ),
                SearchConfig(
                    strategy=SearchStrategy.SEMANTIC,
                    query_variations=[
                        f"psychology sociology anthropology {today} insights",
                        f"literature arts humanities {today} analysis",
                        f"philosophy ethics {today} contemporary thought"
                    ],
                    max_results=target_count * 5,
                    domains=["theatlantic.com", "newyorker.com", "aeon.co", "harpers.org", "lrb.co.uk", "nybooks.com"],
                    recency="week",
                    quality_threshold=65.0,
                    description="Semantic variations for intellectual content"
                ),
                SearchConfig(
                    strategy=SearchStrategy.CROSS_DOMAIN,
                    query_variations=[
                        f"intellectual analysis {today} cultural insights",
                        f"thought-provoking essays {today} humanities"
                    ],
                    max_results=target_count * 6,
                    domains=None,
                    recency="week",
                    quality_threshold=60.0,
                    description="Open domain intellectual search"
                )
            ]

        else:
            # Default configuration for other sections
            configs = [
                SearchConfig(
                    strategy=SearchStrategy.PRIMARY,
                    query_variations=[base_query],
                    max_results=target_count * 3,
                    recency=recency,
                    quality_threshold=65.0,
                    description=f"Primary search for {section}"
                ),
                SearchConfig(
                    strategy=SearchStrategy.CROSS_DOMAIN,
                    query_variations=[base_query.replace(" latest ", " ").replace(" breaking ", " ")],
                    max_results=target_count * 4,
                    domains=None,
                    recency=recency,
                    quality_threshold=60.0,
                    description=f"Fallback search for {section}"
                )
            ]

        return configs

    async def search_with_fallback(
        self,
        section: Section,
        base_query: str,
        target_count: int,
        custom_query: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[SearchResult]]:
        """
        Execute progressive search with intelligent fallback strategies.

        Returns:
            Tuple of (articles_found, search_results_log)
        """
        self.logger.info(f"Starting adaptive search for {section} (target: {target_count})")

        query = custom_query or base_query
        search_configs = self._generate_search_configs(section, query, target_count)
        search_results = []
        all_articles = []
        seen_urls = set()

        for config in search_configs:
            self.logger.info(f"Attempting {config.strategy.value} strategy: {config.description}")

            # Try each query variation in the config
            strategy_articles = []
            strategy_success = False
            start_time = asyncio.get_event_loop().time()

            try:
                for query_variation in config.query_variations:
                    if len(all_articles) >= target_count:
                        break

                    # Execute search
                    result = await self.llmlayer.search(
                        query=query_variation,
                        max_results=config.max_results,
                        domains=config.domains,
                        recency=config.recency,
                        search_type=config.search_type
                    )

                    # Process articles
                    for citation in result.citations:
                        if citation.url not in seen_urls:
                            article = {
                                "headline": citation.title,
                                "url": citation.url,
                                "summary_text": citation.snippet,
                                "source": citation.source_name,
                                "published": citation.published_date.isoformat() if citation.published_date else None,
                                "search_strategy": config.strategy.value,
                                "relevance_score": citation.relevance_score
                            }
                            strategy_articles.append(article)
                            seen_urls.add(citation.url)

                    # Small delay between query variations
                    await asyncio.sleep(0.5)

                strategy_success = len(strategy_articles) > 0
                execution_time = asyncio.get_event_loop().time() - start_time

                search_result = SearchResult(
                    strategy=config.strategy,
                    success=strategy_success,
                    articles_found=len(strategy_articles),
                    articles=strategy_articles,
                    execution_time=execution_time
                )

                self.logger.info(f"{config.strategy.value}: Found {len(strategy_articles)} articles in {execution_time:.1f}s")

            except Exception as e:
                execution_time = asyncio.get_event_loop().time() - start_time
                search_result = SearchResult(
                    strategy=config.strategy,
                    success=False,
                    articles_found=0,
                    articles=[],
                    execution_time=execution_time,
                    error=str(e)
                )
                self.logger.warning(f"{config.strategy.value} failed: {e}")

            search_results.append(search_result)
            all_articles.extend(strategy_articles)

            # Check if we have enough articles
            if len(all_articles) >= target_count:
                self.logger.info(f"Target reached with {config.strategy.value} strategy")
                break

            # Rate limiting between strategies
            await asyncio.sleep(1.0)

        # If still insufficient, try cross-section spillover
        if len(all_articles) < target_count and section in self.section_spillover:
            self.logger.info(f"Attempting cross-section spillover for {section}")
            spillover_articles = await self._attempt_spillover(section, target_count - len(all_articles))
            all_articles.extend(spillover_articles)

        # Sort by relevance score and published date
        all_articles.sort(key=lambda x: (
            x.get('relevance_score', 0.0),
            x.get('published', '1970-01-01')
        ), reverse=True)

        self.logger.info(f"Adaptive search complete: {len(all_articles)} articles found for {section}")
        return all_articles, search_results

    async def _attempt_spillover(self, section: Section, needed_count: int) -> List[Dict[str, Any]]:
        """Attempt to borrow high-quality content from related sections"""
        spillover_sections = self.section_spillover.get(section, [])
        spillover_articles = []

        for spillover_section in spillover_sections:
            if len(spillover_articles) >= needed_count:
                break

            try:
                # Search in the spillover section with broader terms
                today = datetime.now().strftime("%B %d, %Y")
                spillover_query = f"important {spillover_section} news {today} significant developments"

                result = await self.llmlayer.search(
                    query=spillover_query,
                    max_results=needed_count * 2,
                    recency="day",
                    search_type="news"
                )

                for citation in result.citations:
                    if len(spillover_articles) >= needed_count:
                        break

                    article = {
                        "headline": citation.title,
                        "url": citation.url,
                        "summary_text": citation.snippet,
                        "source": citation.source_name,
                        "published": citation.published_date.isoformat() if citation.published_date else None,
                        "search_strategy": "spillover",
                        "spillover_from": spillover_section,
                        "relevance_score": citation.relevance_score * 0.9  # Slight penalty for spillover
                    }
                    spillover_articles.append(article)

                self.logger.info(f"Spillover from {spillover_section}: {len(spillover_articles)} articles")

            except Exception as e:
                self.logger.warning(f"Spillover from {spillover_section} failed: {e}")
                continue

        return spillover_articles

    def get_search_summary(self, search_results: List[SearchResult]) -> Dict[str, Any]:
        """Generate a summary of search performance for monitoring"""
        total_articles = sum(r.articles_found for r in search_results)
        successful_strategies = [r.strategy.value for r in search_results if r.success]
        failed_strategies = [r.strategy.value for r in search_results if not r.success]
        total_time = sum(r.execution_time for r in search_results)

        return {
            "total_articles_found": total_articles,
            "successful_strategies": successful_strategies,
            "failed_strategies": failed_strategies,
            "total_execution_time": total_time,
            "strategies_used": len(search_results),
            "success_rate": len(successful_strategies) / len(search_results) if search_results else 0
        }