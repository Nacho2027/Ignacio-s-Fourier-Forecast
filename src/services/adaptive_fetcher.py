"""
Adaptive Content Fetching Service
Implements multi-tier source fetching with progressive time windows
"""

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class FetchConfig:
    """Configuration for adaptive fetching"""
    section: str
    min_articles: int = 3
    quality_threshold: float = 60.0
    time_windows: List[Tuple[str, float]] = None

    def __post_init__(self):
        if self.time_windows is None:
            # Default progressive time windows with boost factors
            self.time_windows = [
                ('6h', 1.5),   # Last 6 hours, boost factor 1.5
                ('12h', 1.2),  # Last 12 hours, boost factor 1.2
                ('24h', 1.0),  # Last 24 hours, normal scoring
                ('48h', 0.8),  # Last 48 hours, slight penalty
                ('72h', 0.6),  # Last 72 hours, moderate penalty
            ]


class AdaptiveContentFetcher:
    """
    Sophisticated content fetching with multi-tier sources and adaptive time windows
    """

    def __init__(self, llmlayer, source_ranker):
        self.logger = logging.getLogger(__name__)
        self.llmlayer = llmlayer
        self.source_ranker = source_ranker

        # Load source tiers configuration
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'source_tiers.json'
        )
        with open(config_path, 'r') as f:
            self.source_tiers = json.load(f)

        self.logger.info("Adaptive fetcher initialized with multi-tier source pools")

    def calculate_article_score(self, item: Dict[str, Any], hours_old: float, section: str) -> float:
        """
        Calculate sophisticated quality-freshness score for an article
        """
        # Freshness score with different decay curves per section
        if section == 'breaking_news':
            # Steep decay: loses 50% value after 6 hours
            freshness_score = 100 * math.exp(-hours_old / 8)
        elif section in ['startup', 'intellectual']:
            # Gentle decay: loses 50% value after 48 hours
            freshness_score = 100 * math.exp(-hours_old / 64)
        else:
            # Standard decay: loses 50% value after 24 hours
            freshness_score = 100 * math.exp(-hours_old / 32)

        # Source quality score from ranking service
        source_score = self.source_ranker.score_item(item) * 10

        # Content quality indicators
        quality_score = 0

        # Check content length
        content_length = len(item.get('summary_text', ''))
        if content_length > 1000:
            quality_score += 20  # Substantial content
        elif content_length > 500:
            quality_score += 10

        # Check for author attribution
        if item.get('author'):
            quality_score += 10

        # Check source credibility
        source = item.get('source', '').lower()
        if any(premium in source for premium in ['reuters', 'ap', 'wsj', 'bloomberg', 'ft']):
            quality_score += 30
        elif any(quality in source for quality in ['cnbc', 'guardian', 'npr', 'bbc']):
            quality_score += 20

        # Check for original reporting indicators
        if 'exclusive' in item.get('headline', '').lower():
            quality_score += 20

        # Compound score with minimum quality threshold
        if quality_score < 30:  # Below minimum quality
            return 0  # Reject regardless of freshness

        # Weighted combination (prioritizing freshness for daily newsletter)
        final_score = (
            freshness_score * 0.5 +   # 50% weight on freshness
            source_score * 0.3 +      # 30% weight on source reputation
            quality_score * 0.2       # 20% weight on content quality
        )

        return final_score

    def build_time_aware_query(self, section: str, time_window: str) -> str:
        """
        Build a query with explicit time references for better results
        """
        now = datetime.now()
        today = now.strftime("%B %d, %Y")

        # Base queries per section
        base_queries = {
            'breaking_news': f"breaking news today {today} latest just announced developing story",
            'business': f"business market news today {today} stocks economy earnings deals IPO",
            'tech_science': f"technology science breakthrough today {today} AI quantum computing discovery",
            'politics': f"US politics Congress White House Supreme Court today {today} legislation policy",
            'startup': f"startup funding venture capital Series A B C unicorn today {today}",
            'intellectual': f"essays analysis culture philosophy psychology today this week {today}",
            'local': f"Miami Florida Cornell University local news today {today} community"
        }

        query = base_queries.get(section, f"{section} news today {today}")

        # Add time-specific modifiers
        if time_window == '6h':
            query += f" last 6 hours {now.strftime('%I%p')} this morning breaking just in"
        elif time_window == '12h':
            query += " today this morning latest updates just announced"
        elif time_window == '24h':
            query += " today yesterday latest recent developments"
        elif time_window == '48h':
            query += " this week past two days recent important"
        else:
            query += " this week recent significant developments"

        return query

    async def fetch_from_tier(self, section: str, tier: str, time_window: str, max_results: int = 10) -> List[Dict]:
        """
        Fetch articles from a specific source tier
        """
        if section not in self.source_tiers:
            self.logger.warning(f"Section {section} not in source tiers, using breaking_news")
            section = 'breaking_news'

        tier_data = self.source_tiers[section].get(tier, {})
        sources = tier_data.get('sources', [])

        if not sources:
            self.logger.warning(f"No sources for {section}/{tier}")
            return []

        # Build time-aware query
        query = self.build_time_aware_query(section, time_window)

        # Map time windows to recency parameters
        recency_map = {
            '6h': 'hour',
            '12h': 'day',
            '24h': 'day',
            '48h': 'week',
            '72h': 'week'
        }
        recency = recency_map.get(time_window, 'day')

        try:
            # Fetch from all sources in tier
            # Use optimized search if section is configured, otherwise fall back to legacy method
            if section in self.llmlayer.llm_configs:
                self.logger.info(f"Using optimized search for {section}/{tier}")
                result = await self.llmlayer.search_optimized_rate_limited(section, query)
            else:
                # Fallback to legacy search with improved parameters
                result = await self.llmlayer.search(
                    query=query,
                    max_results=max_results,
                    domains=sources,
                    recency=recency,
                    search_type="news"
                )

            articles = []
            for citation in result.citations:
                if citation.published_date:
                    hours_old = (datetime.now() - citation.published_date).total_seconds() / 3600
                else:
                    hours_old = 24  # Default to 24 hours if no date

                article = {
                    'headline': citation.title,
                    'url': citation.url,
                    'summary_text': citation.snippet,
                    'source': citation.source_name,
                    'published': citation.published_date.isoformat() if citation.published_date else None,
                    'hours_old': hours_old,
                    'tier': tier,
                    'tier_score': tier_data.get('score', 5)
                }
                articles.append(article)

            self.logger.info(f"Fetched {len(articles)} articles from {section}/{tier} in {time_window}")
            return articles

        except Exception as e:
            self.logger.error(f"Error fetching from {section}/{tier}: {e}")
            return []

    async def fetch_with_guarantee(self, section: str, config: Optional[FetchConfig] = None) -> List[Dict]:
        """
        Fetch articles with guaranteed minimum count using adaptive strategies
        """
        if config is None:
            config = FetchConfig(section=section)

        all_articles = []
        seen_urls = set()

        # Try each time window progressively
        for time_window, boost_factor in config.time_windows:
            window_articles = []

            # Fetch from all tiers for this time window
            for tier in ['tier_1_premium', 'tier_2_quality', 'tier_3_broad', 'tier_4_fallback']:
                if len(all_articles) >= config.min_articles * 3:  # Get 3x for ranking
                    break

                tier_articles = await self.fetch_from_tier(
                    section=section,
                    tier=tier,
                    time_window=time_window,
                    max_results=15
                )

                # Deduplicate
                for article in tier_articles:
                    if article['url'] not in seen_urls:
                        seen_urls.add(article['url'])

                        # Calculate score with boost factor
                        score = self.calculate_article_score(
                            article,
                            article['hours_old'],
                            section
                        ) * boost_factor

                        article['score'] = score
                        window_articles.append(article)

                # If we have enough quality articles, stop fetching more tiers
                quality_count = len([a for a in window_articles if a['score'] >= config.quality_threshold])
                if quality_count >= config.min_articles:
                    self.logger.info(f"Found {quality_count} quality articles in {tier}, stopping tier expansion")
                    break

            all_articles.extend(window_articles)

            # Check if we have enough quality articles
            quality_articles = [a for a in all_articles if a['score'] >= config.quality_threshold]
            if len(quality_articles) >= config.min_articles:
                self.logger.info(f"Found {len(quality_articles)} quality articles in {time_window}")
                break

            self.logger.warning(f"Only {len(quality_articles)} quality articles in {time_window}, expanding window")

        # Sort by score and return top articles
        all_articles.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Ensure minimum articles even if below quality threshold
        if len(all_articles) < config.min_articles:
            self.logger.error(f"Only found {len(all_articles)} total articles for {section}")

        # Return more than minimum for downstream ranking
        return all_articles[:config.min_articles * 3] if all_articles else []

    async def emergency_fallback(self, section: str, min_articles: int = 3) -> List[Dict]:
        """
        Emergency fallback when normal fetching fails completely
        """
        self.logger.warning(f"Emergency fallback activated for {section}")

        # Use very broad query with top news aggregators
        query = f"top {section} news today {datetime.now().strftime('%B %d, %Y')}"

        try:
            # Use optimized search if section is configured, otherwise fall back to legacy method
            if section in self.llmlayer.llm_configs:
                self.logger.info(f"Using optimized emergency fallback for {section}")
                result = await self.llmlayer.search_optimized_rate_limited(section, query)
            else:
                # Fallback to legacy search
                result = await self.llmlayer.search(
                    query=query,
                    max_results=min_articles * 2,
                    recency="day",
                    search_type="news"
                )

            articles = []
            for citation in result.citations:
                article = {
                    'headline': citation.title,
                    'url': citation.url,
                    'summary_text': citation.snippet,
                    'source': citation.source_name,
                    'published': citation.published_date.isoformat() if citation.published_date else None,
                    'score': 50  # Default medium score for fallback
                }
                articles.append(article)

            return articles[:min_articles]

        except Exception as e:
            self.logger.error(f"Emergency fallback failed: {e}")
            return []