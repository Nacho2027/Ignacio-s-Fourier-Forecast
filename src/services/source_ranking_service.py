"""
Source Ranking Service for quality control and credibility scoring.
This service scores news items based on source authority and filters out low-quality content.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.models.content import NewsItem


class SourceRankingService:
    """
    Service for ranking and filtering news sources based on credibility and quality.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the source ranking service with authority configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Load source authority configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config',
                'sources_authority.json'
            )
        
        self.authority_config = self._load_authority_config(config_path)
        self.source_scores = self._build_source_score_map()
        
        # Configuration settings
        self.min_quality_score = self.authority_config['config']['min_quality_score']
        self.preferred_threshold = self.authority_config['config']['preferred_score_threshold']
        self.max_per_source = self.authority_config['config']['max_items_per_source']
        self.boost_recent_hours = self.authority_config['config']['boost_recent_hours']
        self.recency_weight = self.authority_config['config']['recency_weight']
        
        self.logger.info(f"Source ranking service initialized with {len(self.source_scores)} scored sources")
    
    def _load_authority_config(self, config_path: str) -> Dict:
        """Load the source authority configuration."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded source authority config from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load source authority config: {e}")
            # Return default config if loading fails
            return {
                'tier_1': {'score': 10, 'sources': []},
                'tier_2': {'score': 8, 'sources': []},
                'tier_3': {'score': 6, 'sources': []},
                'blacklist': {'score': 0, 'sources': []},
                'config': {
                    'min_quality_score': 3,
                    'preferred_score_threshold': 5,  # Reduced from 6 to be less restrictive
                    'max_items_per_source': 4,  # Increased from 2 to allow more from good sources
                    'boost_recent_hours': 6,
                    'recency_weight': 1.2
                }
            }
    
    def _build_source_score_map(self) -> Dict[str, int]:
        """Build a map of domain to credibility score."""
        source_scores = {}
        
        for tier_name, tier_data in self.authority_config.items():
            if tier_name == 'config':
                continue
            
            score = tier_data.get('score', 0)
            sources = tier_data.get('sources', [])
            
            for source in sources:
                # Store both with and without www
                source_scores[source] = score
                source_scores[f"www.{source}"] = score
                
                # Also store without .com/.org etc for flexibility
                base = source.split('.')[0]
                if base not in source_scores:
                    source_scores[base] = score
        
        return source_scores
    
    def _extract_domain(self, url: str) -> str:
        """Extract the domain from a URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            
            # Remove www. prefix for consistency
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain.lower()
        except Exception as e:
            self.logger.warning(f"Failed to parse URL {url}: {e}")
            return ""
    
    def _calculate_recency_boost(self, published_date: Optional[datetime]) -> float:
        """Calculate recency boost for recent articles."""
        if not published_date:
            return 1.0
        
        try:
            now = datetime.now()
            if published_date.tzinfo:
                now = now.replace(tzinfo=published_date.tzinfo)
            
            hours_old = (now - published_date).total_seconds() / 3600
            
            if hours_old <= self.boost_recent_hours:
                # Linear boost from 1.0 to recency_weight based on how recent
                boost = 1.0 + (self.recency_weight - 1.0) * (1 - hours_old / self.boost_recent_hours)
                return boost
        except Exception as e:
            self.logger.warning(f"Failed to calculate recency boost: {e}")
        
        return 1.0
    
    def score_item(self, item: NewsItem) -> float:
        """
        Score a single news item based on source credibility and recency.
        
        Returns a score from 0-12 (10 base + up to 2 recency bonus).
        """
        # Extract domain from URL
        domain = self._extract_domain(item.url)
        
        # Get base credibility score
        base_score = self.source_scores.get(domain, 2)  # Default score of 2 for unknown sources
        
        # Check if it's blacklisted
        if base_score == 0:
            self.logger.debug(f"Blacklisted source: {domain}")
            return 0
        
        # Apply recency boost
        recency_boost = self._calculate_recency_boost(item.published_date)
        final_score = base_score * recency_boost
        
        # Log scoring details for high-value items
        if base_score >= self.preferred_threshold:
            self.logger.debug(f"High-quality source {domain}: base={base_score}, recency={recency_boost:.2f}, final={final_score:.2f}")
        
        return final_score
    
    def score_and_rank_items(self, items: List[NewsItem]) -> List[Tuple[NewsItem, float]]:
        """
        Score and rank a list of news items.
        
        Returns a list of (item, score) tuples sorted by score descending.
        """
        scored_items = []
        
        for item in items:
            score = self.score_item(item)
            if score > 0:  # Only include non-blacklisted items
                scored_items.append((item, score))
        
        # Sort by score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Scored {len(items)} items, {len(scored_items)} passed quality filter")
        
        return scored_items
    
    def filter_by_quality(self, items: List[NewsItem], min_score: Optional[float] = None) -> List[NewsItem]:
        """
        Filter items by minimum quality score.
        
        Args:
            items: List of news items to filter
            min_score: Minimum score required (uses config default if not specified)
        
        Returns:
            Filtered list of news items meeting quality threshold
        """
        if min_score is None:
            min_score = self.min_quality_score
        
        scored_items = self.score_and_rank_items(items)
        filtered = [item for item, score in scored_items if score >= min_score]
        
        self.logger.info(f"Filtered {len(items)} items to {len(filtered)} with min_score={min_score}")
        
        return filtered
    
    def apply_diversity_limits(self, items: List[NewsItem]) -> List[NewsItem]:
        """
        Apply diversity limits to prevent too many items from the same source.
        
        Args:
            items: List of news items (should be pre-sorted by score)
        
        Returns:
            List with diversity limits applied
        """
        source_counts = {}
        filtered = []
        
        for item in items:
            domain = self._extract_domain(item.url)
            
            # Track count per source
            count = source_counts.get(domain, 0)
            
            if count < self.max_per_source:
                filtered.append(item)
                source_counts[domain] = count + 1
            else:
                self.logger.debug(f"Skipping item from {domain} (already have {count} items)")
        
        self.logger.info(f"Applied diversity limits: {len(items)} -> {len(filtered)} items")
        
        return filtered
    
    def get_preferred_sources(self, category: Optional[str] = None) -> List[str]:
        """
        Get list of preferred sources for a given category.
        
        Args:
            category: Optional category filter ('news', 'intellectual', 'academic')
        
        Returns:
            List of preferred source domains
        """
        preferred = []
        
        for tier_name, tier_data in self.authority_config.items():
            if tier_name == 'config':
                continue
            
            score = tier_data.get('score', 0)
            if score >= self.preferred_threshold:
                # Filter by category if specified
                if category:
                    if category == 'intellectual' and tier_name == 'intellectual':
                        preferred.extend(tier_data.get('sources', []))
                    elif category == 'academic' and tier_name == 'academic':
                        preferred.extend(tier_data.get('sources', []))
                    elif category == 'news' and tier_name in ['tier_1', 'tier_2']:
                        preferred.extend(tier_data.get('sources', []))
                else:
                    preferred.extend(tier_data.get('sources', []))
        
        return preferred
    
    def get_blacklisted_sources(self) -> List[str]:
        """Get list of blacklisted source domains."""
        return self.authority_config.get('blacklist', {}).get('sources', [])
    
    def process_and_rank(
        self,
        items: List[NewsItem],
        min_score: Optional[float] = None,
        apply_diversity: bool = True,
        max_items: Optional[int] = None
    ) -> List[NewsItem]:
        """
        Complete processing pipeline: score, filter, diversify, and limit.
        
        Args:
            items: Raw list of news items
            min_score: Minimum quality score
            apply_diversity: Whether to apply source diversity limits
            max_items: Maximum number of items to return
        
        Returns:
            Processed and ranked list of news items
        """
        # Score and rank
        scored_items = self.score_and_rank_items(items)
        
        # Filter by quality
        if min_score is None:
            min_score = self.min_quality_score
        
        filtered = [(item, score) for item, score in scored_items if score >= min_score]
        
        # Extract just the items (already sorted by score)
        result = [item for item, score in filtered]
        
        # Apply diversity limits if requested
        if apply_diversity:
            result = self.apply_diversity_limits(result)
        
        # Limit total items if specified
        if max_items and len(result) > max_items:
            result = result[:max_items]
        
        self.logger.info(
            f"Processed {len(items)} items -> {len(result)} final items "
            f"(min_score={min_score}, diversity={apply_diversity})"
        )
        
        return result