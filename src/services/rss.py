import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from urllib.parse import urlparse
import re
import csv
import os
import random
import subprocess

# Try to import external dependencies, fall back to simple implementation
try:
    import aiohttp
    import feedparser
    from bs4 import BeautifulSoup
    from dateutil import parser as dateutil_parser
    from zoneinfo import ZoneInfo
    from src.services.cache_service import ContentItem
    EXTERNAL_DEPS_AVAILABLE = True
except ImportError:
    # Fallback to simple implementation
    EXTERNAL_DEPS_AVAILABLE = False
    from src.services.rss_simple import SimpleRSSService, SimpleRSSAdapter
    ContentItem = None

if EXTERNAL_DEPS_AVAILABLE:
    EASTERN_TZ = ZoneInfo("America/New_York")
else:
    EASTERN_TZ = None


@dataclass
class RSSItem:
    """Structured RSS feed item"""
    title: str
    description: str
    link: str
    published_date: datetime
    guid: str
    source_feed: str
    author: Optional[str] = None
    categories: Optional[List[str]] = None
    content: Optional[str] = None  # Full content if available

    def to_content_item(self, section: str) -> ContentItem:
        """Convert to ContentItem dataclass for pipeline"""
        return ContentItem(
            id=f"rss_{hashlib.md5(self.guid.encode()).hexdigest()[:12]}",
            source=self.source_feed,
            section=section,
            headline=self.title,
            summary_text=self.content or self.description,
            url=self.link,
            published_date=self.published_date,
            metadata={
                "author": self.author,
                "categories": self.categories or [],
                "guid": self.guid,
            },
        )


@dataclass
class DailyReading:
    """Structured Catholic daily reading"""
    date: datetime
    liturgical_day: str  # e.g., "Thursday of the Thirty-second Week in Ordinary Time"
    first_reading: Dict[str, str]  # {reference, text}
    responsorial_psalm: Dict[str, str]  # {reference, text, response}
    second_reading: Optional[Dict[str, str]]  # Sunday/Solemnity only
    gospel: Dict[str, str]  # {reference, text}
    reflection: Optional[str] = None
    saint_of_day: Optional[str] = None


@dataclass
class FeedConfig:
    """Configuration for an RSS feed"""
    name: str
    url: str
    section: str
    check_frequency_hours: int
    parse_full_content: bool = False
    special_parser: Optional[str] = None  # e.g., "usccb"
    priority: int = 1  # Higher numbers = higher priority
    keywords: Optional[List[str]] = None  # Keywords for content filtering
    max_age_hours: int = 168  # Maximum age of articles in hours (default 7 days)


class RSSService:
    """
    Comprehensive RSS feed integration for all newsletter sections.
    Replaces llmlayer with direct RSS feed parsing and intelligent content filtering.
    """

    def __init__(self):
        """Initialize RSS service with comprehensive feed configurations."""
        self.timeout = 30  # seconds
        self.max_retries = 3
        self.logger = logging.getLogger(__name__)

        # Load feed configurations from CSV and hardcoded configs
        self.feeds = self._load_all_feed_configs()
        
        # Section-based feed mappings for organized access
        self.section_feeds = self._organize_feeds_by_section()

        # Cache for the session
        self._cache: Dict[str, List[RSSItem]] = {}
        self._readings_cache: Dict[str, DailyReading] = {}
        
        # Content filtering configuration
        self.quality_keywords = self._init_quality_keywords()
        self.exclude_patterns = self._init_exclude_patterns()
        
        self.logger.info(f"Initialized RSS service with {len(self.feeds)} feeds across {len(self.section_feeds)} sections")

    def _load_all_feed_configs(self) -> Dict[str, FeedConfig]:
        """Load comprehensive feed configurations from CSV and hardcoded configs."""
        configs = {}
        
        # Load spiritual/scripture feeds (maintain existing functionality)
        configs.update(self._init_spiritual_feeds())
        
        # Load newsletter feeds from RSS.csv
        csv_path = "/code/InternalDocs/RSS.csv"
        if os.path.exists(csv_path):
            configs.update(self._load_feeds_from_csv(csv_path))
        else:
            self.logger.warning(f"RSS.csv not found at {csv_path}, using intelligent fallback configuration")
            configs.update(self._init_intelligent_fallback_feeds())
            
        return configs
        
    def _init_spiritual_feeds(self) -> Dict[str, FeedConfig]:
        """Initialize spiritual/scripture feed configurations (existing functionality)."""
        return {
            # Spiritual Content (RSS-based)
            "usccb_daily": FeedConfig(
                name="USCCB Daily Readings",
                url="https://bible.usccb.org/readings.rss",
                section="scripture",
                check_frequency_hours=24,
                parse_full_content=True,
                special_parser="usccb",
                priority=10,
                max_age_hours=48,
            ),
            "catholic_daily_reflections": FeedConfig(
                name="Catholic Daily Reflections",
                url="https://catholic-daily-reflections.com/feed/",
                section="scripture",
                check_frequency_hours=24,
                parse_full_content=True,
                special_parser=None,
                priority=9,
                max_age_hours=48,
            ),
            "vatican_news": FeedConfig(
                name="Vatican News",
                url="https://www.vaticannews.va/en/pope.rss",
                section="scripture",
                check_frequency_hours=12,
                parse_full_content=False,
                priority=8,
                max_age_hours=72,
            ),
            
            # News & Current Affairs (RSS-based for high reliability)
            "ap_news": FeedConfig(
                name="Associated Press",
                url="https://feeds.apnews.com/rss/apf-topnews",
                section="breaking_news",
                check_frequency_hours=1,
                parse_full_content=False,
            ),
            "reuters_world": FeedConfig(
                name="Reuters World News",
                url="https://feeds.reuters.com/reuters/worldNews",
                section="breaking_news",
                check_frequency_hours=1,
                parse_full_content=False,
            ),
            
            # Technology & Science (RSS-based)
            "mit_tech_review": FeedConfig(
                name="MIT Technology Review",
                url="https://www.technologyreview.com/feed/",
                section="tech_science",
                check_frequency_hours=6,
                parse_full_content=False,
            ),
            "ieee_spectrum": FeedConfig(
                name="IEEE Spectrum",
                url="https://spectrum.ieee.org/rss/fulltext",
                section="tech_science", 
                check_frequency_hours=6,
                parse_full_content=False,
            ),
            "nature_news": FeedConfig(
                name="Nature News",
                url="https://www.nature.com/nature.rss",
                section="tech_science",
                check_frequency_hours=12,
                parse_full_content=False,
            ),
            
            # Business & Finance (RSS-based)
            "axios_business": FeedConfig(
                name="Axios Business",
                url="https://api.axios.com/feed/business",
                section="business",
                check_frequency_hours=2,
                parse_full_content=False,
            ),
            
            # Miscellaneous Intellectual Content (RSS-based)
            "aeon": FeedConfig(
                name="Aeon Essays",
                url="https://aeon.co/feed.rss",
                section="miscellaneous",
                check_frequency_hours=12,
                parse_full_content=False,
            ),
            "the_atlantic": FeedConfig(
                name="The Atlantic",
                url="https://www.theatlantic.com/feed/all/",
                section="miscellaneous",
                check_frequency_hours=6,
                parse_full_content=False,
            ),
        }
        
    def _load_feeds_from_csv(self, csv_path: str) -> Dict[str, FeedConfig]:
        """Load feed configurations from RSS.csv file."""
        configs = {}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for i, row in enumerate(reader):
                    section = row.get('Section', '').lower().replace(' ', '_').replace('&', 'and')
                    source = row.get('Source', '').strip()
                    url = row.get('RSS Feed URL', '').strip()
                    description = row.get('Description', '').strip()
                    
                    if not url or not section or not source:
                        continue
                        
                    # Map CSV sections to our internal section names
                    section = self._map_csv_section_to_internal(section)
                    if not section:
                        continue
                        
                    # Generate unique feed ID
                    feed_id = f"{section}_{source.lower().replace(' ', '_').replace('.', '').replace('-', '_')}"
                    
                    # Determine priority based on source authority
                    priority = self._get_source_priority(source, section)
                    
                    # Set max age based on section
                    max_age = self._get_section_max_age(section)
                    
                    # Extract keywords for filtering
                    keywords = self._extract_keywords_for_source(source, section, description)
                    
                    configs[feed_id] = FeedConfig(
                        name=source,
                        url=url,
                        section=section,
                        check_frequency_hours=self._get_check_frequency(section),
                        parse_full_content=False,  # Start with summaries for performance
                        priority=priority,
                        keywords=keywords,
                        max_age_hours=max_age,
                    )
                    
            self.logger.info(f"Loaded {len(configs)} feed configurations from CSV")
            
        except Exception as e:
            self.logger.error(f"Failed to load RSS feeds from CSV: {e}")
            
        return configs
        
    def _map_csv_section_to_internal(self, csv_section: str) -> Optional[str]:
        """Map CSV section names to internal section names."""
        mapping = {
            'breaking_news': 'breaking_news',
            'business': 'business', 
            'tech_and_science': 'tech_science',
            'research_papers': 'research_papers',
            'politics': 'politics',
            'miscellaneous': 'miscellaneous',
        }
        return mapping.get(csv_section)
        
    def _get_source_priority(self, source: str, section: str) -> int:
        """Determine source priority based on authority and section."""
        # High priority sources (premium, authoritative)
        high_priority = {
            'BBC News': 10, 'Reuters Top News': 10, 'Associated Press': 10,
            'Wall Street Journal': 9, 'Financial Times': 9, 'Bloomberg': 9,
            'Nature News': 10, 'Science Magazine': 10, 'MIT Technology Review': 9,
            'The Atlantic': 9, 'The New Yorker': 9, 'Aeon': 8,
        }
        
        # Medium priority sources
        medium_priority = {
            'CNN World': 7, 'Guardian World News': 7, 'NPR News': 7,
            'CNBC': 6, 'MarketWatch': 6, 'Harvard Business Review': 8,
            'TechCrunch': 6, 'Wired': 7, 'The Verge': 6,
            'Politico': 7, 'The Hill': 6,
        }
        
        # Check exact matches first
        if source in high_priority:
            return high_priority[source]
        if source in medium_priority:
            return medium_priority[source]
            
        # Section-based priorities for unlisted sources
        if section == 'research_papers':
            return 8 if 'arxiv' in source.lower() or 'nature' in source.lower() else 5
        elif section == 'breaking_news':
            return 7 if any(term in source.lower() for term in ['news', 'times', 'post']) else 4
        elif section == 'business':
            return 6 if any(term in source.lower() for term in ['business', 'financial', 'market']) else 4
        else:
            return 5  # Default priority
            
    def _get_section_max_age(self, section: str) -> int:
        """Get maximum article age in hours for each section."""
        age_limits = {
            'breaking_news': 24,      # Very recent
            'business': 48,           # Recent business news  
            'politics': 48,           # Recent political developments
            'tech_science': 168,      # Tech can be up to a week old
            'research_papers': 720,   # Research papers can be up to a month old
            'miscellaneous': 336,     # Intellectual content up to 2 weeks
            'scripture': 48,          # Daily readings
        }
        return age_limits.get(section, 168)  # Default 1 week
        
    def _get_check_frequency(self, section: str) -> int:
        """Get check frequency in hours for each section."""
        frequencies = {
            'breaking_news': 1,       # Check hourly
            'business': 2,            # Check every 2 hours
            'politics': 3,            # Check every 3 hours
            'tech_science': 6,        # Check every 6 hours
            'research_papers': 24,    # Check daily
            'miscellaneous': 12,      # Check twice daily
            'scripture': 24,          # Check daily
        }
        return frequencies.get(section, 6)  # Default every 6 hours
        
    def _extract_keywords_for_source(self, source: str, section: str, description: str) -> List[str]:
        """Extract relevant keywords for content filtering based on source and section."""
        keywords = []
        
        # Section-based keywords
        section_keywords = {
            'breaking_news': ['breaking', 'urgent', 'developing', 'alert', 'just in', 'live'],
            'business': ['business', 'market', 'economy', 'finance', 'earnings', 'stock', 'trade'],
            'tech_science': ['technology', 'science', 'research', 'innovation', 'breakthrough', 'study'],
            'research_papers': ['research', 'study', 'paper', 'journal', 'findings', 'analysis'],
            'politics': ['politics', 'government', 'policy', 'congress', 'senate', 'election'],
            'miscellaneous': ['culture', 'arts', 'philosophy', 'society', 'analysis', 'essay'],
        }
        
        keywords.extend(section_keywords.get(section, []))
        
        # Source-specific keywords from description
        if description:
            desc_words = description.lower().split()
            relevant_words = [word for word in desc_words if len(word) > 4 and word.isalpha()]
            keywords.extend(relevant_words[:3])  # Add top 3 relevant words
            
        return keywords
        
    def _init_intelligent_fallback_feeds(self) -> Dict[str, FeedConfig]:
        """Initialize intelligent fallback feed configurations combining CSV approach with our enhancements."""
        return {
            # Breaking News - High reliability sources
            'breaking_news_ap': FeedConfig(
                name='Associated Press', url='https://feeds.apnews.com/rss/apf-topnews',
                section='breaking_news', check_frequency_hours=1, priority=10, max_age_hours=24
            ),
            'breaking_news_reuters': FeedConfig(
                name='Reuters World News', url='https://feeds.reuters.com/reuters/worldNews',
                section='breaking_news', check_frequency_hours=1, priority=10, max_age_hours=24
            ),
            'breaking_news_bbc': FeedConfig(
                name='BBC News', url='https://feeds.bbci.co.uk/news/rss.xml',
                section='breaking_news', check_frequency_hours=1, priority=9, max_age_hours=24
            ),
            
            # Business & Finance - Premium sources
            'business_axios': FeedConfig(
                name='Axios Business', url='https://api.axios.com/feed/business',
                section='business', check_frequency_hours=2, priority=9, max_age_hours=48
            ),
            'business_wsj': FeedConfig(
                name='Wall Street Journal', url='https://feeds.a.dj.com/rss/RSSWorldNews.xml',
                section='business', check_frequency_hours=2, priority=9, max_age_hours=48
            ),
            
            # Technology & Science - Leading sources
            'tech_science_mit': FeedConfig(
                name='MIT Technology Review', url='https://www.technologyreview.com/feed/',
                section='tech_science', check_frequency_hours=6, priority=9, max_age_hours=168
            ),
            'tech_science_ieee': FeedConfig(
                name='IEEE Spectrum', url='https://spectrum.ieee.org/rss/fulltext',
                section='tech_science', check_frequency_hours=6, priority=8, max_age_hours=168
            ),
            'tech_science_nature': FeedConfig(
                name='Nature News', url='https://www.nature.com/nature.rss',
                section='tech_science', check_frequency_hours=12, priority=10, max_age_hours=168
            ),
            'tech_science_wired': FeedConfig(
                name='Wired', url='https://www.wired.com/feed/rss',
                section='tech_science', check_frequency_hours=6, priority=7, max_age_hours=168
            ),
            
            # Miscellaneous - Intellectual sources
            'miscellaneous_aeon': FeedConfig(
                name='Aeon Essays', url='https://aeon.co/feed.rss',
                section='miscellaneous', check_frequency_hours=12, priority=8, max_age_hours=336
            ),
            'miscellaneous_atlantic': FeedConfig(
                name='The Atlantic', url='https://www.theatlantic.com/feed/all/',
                section='miscellaneous', check_frequency_hours=6, priority=9, max_age_hours=336
            ),
        }
        
    def _organize_feeds_by_section(self) -> Dict[str, List[str]]:
        """Organize feed IDs by section for efficient lookup."""
        section_feeds = {}
        for feed_id, config in self.feeds.items():
            section = config.section
            if section not in section_feeds:
                section_feeds[section] = []
            section_feeds[section].append(feed_id)
            
        # Sort feeds within each section by priority (descending)
        for section in section_feeds:
            section_feeds[section].sort(
                key=lambda feed_id: self.feeds[feed_id].priority,
                reverse=True
            )
            
        return section_feeds
        
    def _init_quality_keywords(self) -> Dict[str, List[str]]:
        """Initialize quality keywords for content filtering."""
        return {
            'high_quality': [
                'analysis', 'investigation', 'exclusive', 'report', 'research',
                'study', 'findings', 'breakthrough', 'innovation', 'insight'
            ],
            'low_quality': [
                'listicle', 'clickbait', 'viral', 'trending', 'shocking',
                'you won\'t believe', 'celebrities', 'gossip'
            ]
        }
        
    def _init_exclude_patterns(self) -> List[str]:
        """Initialize patterns for excluding low-quality content."""
        return [
            r'\d+\s+(ways|things|reasons|tips)',  # Listicles
            r'you\s+won\'t\s+believe',              # Clickbait
            r'\d+\s+photos?\s+that',                # Photo galleries
            r'celebrities?\s+(who|that|wearing)',   # Celebrity content
            r'(watch|see)\s+what\s+happens?',       # Clickbait videos
        ]

    async def fetch_feed(self, feed_url: str, max_items: int = 10, filter_quality: bool = True) -> List[RSSItem]:
        """
        Fetch and parse an RSS feed with intelligent filtering.
        """
        cache_key = self._generate_cache_key(feed_url, filter_quality, max_items)
        
        # Don't use cache for daily readings to ensure fresh content
        if "usccb" in feed_url or "catholic" in feed_url:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self.logger.debug(f"Cleared cache for spiritual feed: {feed_url}")
        elif cache_key in self._cache:
            return self._cache[cache_key]

        last_error: Optional[Exception] = None
        content: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                content = await self._fetch_with_retry(feed_url)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                # Exponential backoff
                wait_time = 0.25 * (2 ** attempt)
                await asyncio.sleep(wait_time)
                
        if content is None:
            raise RSSServiceError(str(last_error) if last_error else "Failed to fetch feed")
            
        items = self._parse_feed(content, feed_url)
        
        # Apply quality filtering if requested
        if filter_quality:
            items = self._filter_content_quality(items)
            
        if max_items:
            items = items[:max_items]
            
        self._cache[cache_key] = items
        return items

    def _filter_content_quality(self, items: List[RSSItem]) -> List[RSSItem]:
        """Filter RSS items for quality based on content patterns and keywords."""
        if not items:
            return []
            
        filtered_items = []
        for item in items:
            # Check title and description for quality indicators
            combined_text = f"{item.title} {item.description}".lower()
            
            # Skip low-quality content patterns
            skip_item = False
            for pattern in self.exclude_patterns:
                if re.search(pattern, combined_text):
                    self.logger.debug(f"Excluding item due to pattern '{pattern}': {item.title[:50]}")
                    skip_item = True
                    break
                    
            if skip_item:
                continue
                
            # Check for low-quality keywords
            has_low_quality = any(keyword in combined_text for keyword in self.quality_keywords['low_quality'])
            if has_low_quality:
                self.logger.debug(f"Excluding item due to low-quality keywords: {item.title[:50]}")
                continue
                
            # Prefer items with high-quality keywords
            has_high_quality = any(keyword in combined_text for keyword in self.quality_keywords['high_quality'])
            if has_high_quality:
                item.title = f"⭐ {item.title}"  # Mark high-quality items
                
            filtered_items.append(item)
            
        self.logger.info(f"Quality filtering: {len(items)} -> {len(filtered_items)} items")
        return filtered_items

    def _generate_cache_key(self, feed_url: str, filter_quality: bool = True, max_items: int = 10) -> str:
        """Generate cache key for feed URL with parameters."""
        key_parts = [feed_url, str(filter_quality), str(max_items)]
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()

    async def fetch_configured_feed(self, feed_name: str) -> List[RSSItem]:
        """Fetch a pre-configured feed by name."""
        if feed_name not in self.feeds:
            raise ValueError(f"Feed '{feed_name}' not configured")
        config = self.feeds[feed_name]
        try:
            items = await self.fetch_feed(config.url)
            if not items and config.special_parser == "usccb":
                return await self._fetch_usccb_daily()
            return items
        except RSSServiceError:
            if config.special_parser == "usccb":
                return await self._fetch_usccb_daily()
            raise

    async def get_daily_readings(self, date: Optional[datetime] = None) -> DailyReading:
        """Get Catholic daily readings for a specific date."""
        now_et = datetime.now(EASTERN_TZ) if EASTERN_TZ else datetime.now()
        target_date = (date.astimezone(EASTERN_TZ) if date and date.tzinfo and EASTERN_TZ else (date or now_et)).date()
        
        # Clear stale cache entries (anything not from today)
        today_key = target_date.isoformat()
        keys_to_remove = [k for k in self._readings_cache.keys() if k != today_key]
        for key in keys_to_remove:
            del self._readings_cache[key]
            self.logger.debug(f"Cleared stale readings cache for {key}")
        
        cache_key = today_key
        if cache_key in self._readings_cache:
            self.logger.debug(f"Returning cached readings for {cache_key}")
            return self._readings_cache[cache_key]

        items = await self.fetch_configured_feed("usccb_daily")
        content_html = ""
        if items:
            content_html = items[0].content or items[0].description or ""

        reading = self.parse_usccb_content(content_html)
        # If parsing failed (no references), fallback to direct HTML page
        if not reading.first_reading.get("reference") and not reading.gospel.get("reference"):
            fallback_items = await self._fetch_usccb_daily()
            if fallback_items:
                content_html = fallback_items[0].content or fallback_items[0].description or ""
                reading = self.parse_usccb_content(content_html)

        # Ensure date field in ET midnight
        reading.date = datetime.combine(target_date, datetime.min.time(), tzinfo=EASTERN_TZ if EASTERN_TZ else None)
        self._readings_cache[cache_key] = reading
        return reading

    async def get_todays_spiritual_content(self) -> Dict[str, Any]:
        """Get today's complete spiritual content."""
        readings = await self.get_daily_readings()
        content: Dict[str, Any] = {"readings": readings}
        return content

    async def _fetch_usccb_daily(self) -> List[RSSItem]:
        """Fetch today's USCCB daily reading page and wrap as RSSItem."""
        now_et = datetime.now(EASTERN_TZ) if EASTERN_TZ else datetime.now()
        mm = f"{now_et.month:02d}"
        dd = f"{now_et.day:02d}"
        yy = f"{now_et.year % 100:02d}"
        reading_url = f"https://bible.usccb.org/bible/readings/{mm}{dd}{yy}.cfm"

        html = await self._fetch_with_retry(reading_url)

        title = f"Daily Reading for {now_et.strftime('%A, %B %d, %Y')}"
        item = RSSItem(
            title=title,
            description="USCCB Daily Readings",
            link=reading_url,
            published_date=now_et,
            guid=reading_url,
            source_feed="USCCB",
            categories=["Scripture", "Daily"],
            content=html,
        )
        return [item]

    async def fetch_all_configured_feeds(self) -> Dict[str, List[RSSItem]]:
        """Fetch all configured feeds in parallel."""
        results: Dict[str, List[RSSItem]] = {}
        async def fetch_one(name: str, cfg: FeedConfig) -> None:
            results[name] = await self.fetch_feed(cfg.url)

        tasks = [fetch_one(name, cfg) for name, cfg in self.feeds.items()]
        await asyncio.gather(*tasks)
        return results

    async def fetch_feeds_by_section(self, section: str, max_items_per_feed: int = 10) -> List[RSSItem]:
        """
        Fetch RSS feeds for a specific section with intelligent content selection.
        This replaces hardcoded keyword searches with curated RSS feeds.
        """
        section_feeds = [name for name, config in self.feeds.items() if config.section == section]
        
        if not section_feeds:
            self.logger.warning(f"No RSS feeds configured for section: {section}")
            return []
            
        self.logger.info(f"Fetching {len(section_feeds)} RSS feeds for section: {section}")
        
        all_items: List[RSSItem] = []
        
        # Fetch all feeds for this section in parallel
        async def fetch_section_feed(feed_name: str) -> List[RSSItem]:
            try:
                config = self.feeds[feed_name]
                items = await self.fetch_feed(config.url, max_items_per_feed)
                # Mark items with their source feed for provenance
                for item in items:
                    item.source_feed = config.name
                return items
            except Exception as e:
                self.logger.error(f"Failed to fetch RSS feed {feed_name}: {e}")
                return []
        
        tasks = [fetch_section_feed(feed_name) for feed_name in section_feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all successful results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"RSS feed fetch failed for {section_feeds[i]}: {result}")
            else:
                all_items.extend(result)
        
        # Apply intelligent content filtering based on recency and quality
        filtered_items = self._apply_intelligent_filtering(all_items, section)
        
        self.logger.info(f"Collected {len(all_items)} items from RSS, filtered to {len(filtered_items)} high-quality items for {section}")
        return filtered_items
    
    def _apply_intelligent_filtering(self, items: List[RSSItem], section: str) -> List[RSSItem]:
        """
        Apply intelligent filtering to RSS items instead of relying on hardcoded keywords.
        Uses publication date, content quality indicators, and section-specific criteria.
        """
        if not items:
            return []
        
        # Filter by recency (last 7 days for most sections, 24 hours for breaking news)
        from datetime import timedelta
        now = datetime.now()
        if section == "breaking_news":
            cutoff = now - timedelta(hours=24)
        elif section == "business":
            cutoff = now - timedelta(days=3) 
        elif section in ["tech_science", "miscellaneous"]:
            cutoff = now - timedelta(days=7)
        else:
            cutoff = now - timedelta(days=14)  # Default for other sections
            
        recent_items = [item for item in items if item.published_date >= cutoff]
        
        # Quality filtering based on content characteristics
        quality_items = []
        for item in recent_items:
            # Skip items that are too short (likely not substantial content)
            description_length = len(item.description or "") + len(item.content or "")
            if description_length < 100:  # Minimum content threshold
                continue
                
            # Skip items with common low-quality indicators
            title_lower = item.title.lower()
            if any(skip_phrase in title_lower for skip_phrase in [
                "breaking:", "live updates", "live blog", "watch:", "photos:", 
                "video:", "gallery:", "sponsored", "advertisement"
            ]):
                continue
                
            quality_items.append(item)
        
        # Sort by publication date (most recent first) and limit
        quality_items.sort(key=lambda x: x.published_date, reverse=True)
        
        # Return top items based on section requirements
        section_limits = {
            "breaking_news": 15,  # More items for filtering down to 3
            "business": 12,       # More items for filtering down to 3  
            "tech_science": 12,   # More items for filtering down to 3
            "miscellaneous": 20,  # More items for filtering down to 5
            "spiritual": 10       # Fewer needed for this section
        }
        
        limit = section_limits.get(section, 15)
        return quality_items[:limit]

    def parse_usccb_content(self, content: str) -> DailyReading:
        """Parse USCCB daily readings from HTML content."""
        soup = BeautifulSoup(content or "", "html.parser")

        # Liturgical day
        liturgical_day_tag = soup.find(["h2", "h3"])
        liturgical_day = liturgical_day_tag.get_text(strip=True) if liturgical_day_tag else "Daily Readings"

        def extract_section(label_keywords: List[str]) -> Optional[Dict[str, str]]:
            # Find an h3 that contains any of the keywords
            for h3 in soup.find_all(["h3", "h4"]):
                text = h3.get_text(" ", strip=True)
                if any(keyword.lower() in text.lower() for keyword in label_keywords):
                    # Debug logging to understand current HTML structure
                    self.logger.debug(f"USCCB header found: '{text}' for keywords {label_keywords}")

                    # Improved reference extraction - handle both colon and no-colon formats
                    reference = ""

                    # First try: Extract after colon if present
                    if ":" in text:
                        reference = text.split(":", 1)[1].strip()
                        self.logger.debug(f"Reference after colon split: '{reference}'")
                    else:
                        # No colon - extract using patterns
                        self.logger.debug(f"No colon found in text: '{text}'")

                    # Universal pattern matching for both colon and no-colon cases
                    # Pattern 1: Extract full biblical citation after reading type
                    # Matches: "Reading 1 1 Timothy 6:2c-12" or "Gospel Luke 8:1-3"
                    full_citation_pattern = r'(?:Reading\s+\d+|Reading\s+[IVX]+|First\s+Reading|Second\s+Reading|Gospel|Responsorial\s+Psalm)\s+(.+)$'
                    citation_match = re.search(full_citation_pattern, text, re.IGNORECASE)
                    if citation_match:
                        extracted = citation_match.group(1).strip()
                        # Use this if it's longer/better than what we got from colon split
                        if len(extracted) > len(reference) or re.match(r'^[\d\-\.,;:\s]+$', reference):
                            reference = extracted
                            self.logger.debug(f"Full citation pattern match: '{reference}'")

                    # Pattern 2: If we still have incomplete reference, try to reconstruct
                    if re.match(r'^[\d\-\.,;:\s]+$', reference):
                        # Reference is just numbers/punctuation, look for book name in original text
                        book_pattern = r'((?:\d+\s+)?[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+([\d\-\.,;:\s]+)$'
                        match = re.search(book_pattern, text)
                        if match:
                            book_part = match.group(1).strip()
                            verse_part = match.group(2).strip()
                            # Check if the book part is not a keyword
                            if not any(keyword.lower() in book_part.lower() for keyword in label_keywords):
                                reference = f"{book_part} {verse_part}".strip()
                                self.logger.debug(f"Reconstructed reference from book pattern: '{reference}'")

                    # Fallback: use the original text if nothing else worked
                    if not reference:
                        reference = text
                        self.logger.debug(f"Using original text as fallback: '{reference}'")

                    # Validate and clean the reference
                    reference = self._validate_and_clean_reference(reference, label_keywords)
                    # Gather following paragraphs until next header
                    parts: List[str] = []
                    for sib in h3.find_all_next():
                        if sib.name in ["h2", "h3", "h4"]:
                            break
                        if sib.name == "p":
                            text = sib.get_text(" ", strip=True)
                            # Remove copyright notices
                            text = re.sub(r"Lectionary for Mass.*?©.*?\.", "", text, flags=re.IGNORECASE)
                            text = re.sub(r"Used with permission.*?\.", "", text, flags=re.IGNORECASE)
                            text = re.sub(r"Copyright.*?reserved.*?\.", "", text, flags=re.IGNORECASE)
                            text = re.sub(r"©\s*\d{4}.*?\.", "", text)
                            text = re.sub(r"All rights reserved.*?\.", "", text, flags=re.IGNORECASE)
                            text = re.sub(r"Excerpts from.*?permission.*?\.", "", text, flags=re.IGNORECASE)
                            # Remove the specific "Neither this work" copyright notice
                            text = re.sub(r"Neither this work nor any part.*?copyright owner\.", "", text, flags=re.IGNORECASE | re.DOTALL)
                            text = re.sub(r"Neither this work.*?without permission.*?\.", "", text, flags=re.IGNORECASE | re.DOTALL)
                            # Catch any remaining copyright language
                            text = re.sub(r".*may\s+be\s+reproduced.*?without\s+permission.*?\.", "", text, flags=re.IGNORECASE | re.DOTALL)
                            text = text.strip()
                            if text:  # Only add if there's content left after removing copyright
                                parts.append(text)
                    # Clean the final combined text as well
                    final_text = "\n".join(parts).strip()
                    # Remove any trailing copyright that might span paragraphs
                    final_text = re.sub(r"Lectionary for Mass.*", "", final_text, flags=re.IGNORECASE)
                    final_text = re.sub(r"Copyright.*", "", final_text, flags=re.IGNORECASE)
                    final_text = re.sub(r"Neither this work.*", "", final_text, flags=re.IGNORECASE)
                    return {"reference": reference, "text": final_text.strip()}
            return None

        first_reading = extract_section(["Reading 1", "Reading I", "First Reading"]) or {"reference": "", "text": ""}
        responsorial = extract_section(["Responsorial Psalm"]) or {"reference": "", "text": ""}
        second_reading = extract_section(["Reading 2", "Reading II", "Second Reading"]) or None
        gospel = extract_section(["Gospel"]) or {"reference": "", "text": ""}

        # Try to extract psalm response
        response_match = re.search(r"R\.?\s*(.+)", responsorial.get("text", ""))
        response = response_match.group(1).strip() if response_match else ""
        responsorial_psalm = {
            "reference": responsorial.get("reference", ""),
            "text": responsorial.get("text", ""),
            "response": response,
        }

        return DailyReading(
            date=datetime.now(EASTERN_TZ) if EASTERN_TZ else datetime.now(),
            liturgical_day=liturgical_day,
            first_reading=first_reading,
            responsorial_psalm=responsorial_psalm,
            second_reading=second_reading,
            gospel=gospel,
            reflection=None,
            saint_of_day=None,
        )

    def extract_scripture_wisdom(self, reading: DailyReading) -> str:
        """Extract key wisdom/insight from daily readings."""
        # Simple heuristic for now
        gospel_text = reading.gospel.get("text", "") if reading.gospel else ""
        first_text = reading.first_reading.get("text", "") if reading.first_reading else ""
        combined = f"{gospel_text} {first_text}".strip()
        if not combined:
            return "Reflection unavailable."
        # Return first meaningful sentence
        sentences = re.split(r"(?<=[.!?])\s+", combined)
        return sentences[0][:280]

    async def _fetch_with_retry(self, url: str) -> str:
        """Single attempt fetch; external retry handled by caller."""
        # USCCB URLs need special handling due to anti-bot protection
        # Only curl works reliably, so use it directly
        if 'bible.usccb.org' in url:
            return await self._fetch_with_curl(url)
        
        # Regular fetch for non-USCCB URLs using aiohttp
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        # Disable SSL verification to avoid local cert issues in test environments
        connector = aiohttp.TCPConnector(ssl=False)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": url,
            "Connection": "keep-alive",
        }
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    raise RSSServiceError(f"HTTP {resp.status} for {url}")
                return await resp.text()
    
    async def _fetch_with_curl(self, url: str) -> str:
        """Fetch USCCB content using curl subprocess (only method that works due to anti-bot protection)."""
        cmd = [
            'curl',
            '-s',  # Silent mode
            '-L',  # Follow redirects
            '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            '-H', 'Accept: application/rss+xml,application/xml;q=0.9,text/html;q=0.8,*/*;q=0.7',
            '-H', 'Accept-Language: en-US,en;q=0.9',
            '--compressed',  # Handle gzip/deflate
            '--max-time', str(self.timeout),
            url
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Successfully fetched USCCB content from {url} using curl")
                return stdout.decode('utf-8', errors='ignore')
            else:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else f"curl exited with code {process.returncode}"
                raise RSSServiceError(f"curl failed: {error_msg}")
        except Exception as e:
            raise RSSServiceError(f"Failed to execute curl: {e}")

    def _parse_feed(self, content: str, feed_url: str) -> List[RSSItem]:
        """Parse RSS/Atom feed content."""
        parsed = feedparser.parse(content)
        items: List[RSSItem] = []
        for entry in parsed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            description = getattr(entry, "summary", getattr(entry, "description", "")).strip()
            guid = getattr(entry, "id", getattr(entry, "guid", link or title))
            author = getattr(entry, "author", None)
            categories = [t.term for t in getattr(entry, "tags", [])] if getattr(entry, "tags", None) else []
            content_value: Optional[str] = None
            if getattr(entry, "content", None):
                # entry.content is a list of dicts
                try:
                    content_value = entry.content[0].value
                except Exception:  # noqa: BLE001
                    content_value = None
            elif getattr(entry, "summary", None):
                content_value = entry.summary

            # Published date
            published_raw = getattr(entry, "published", getattr(entry, "updated", ""))
            published_date = self._parse_date(published_raw) if published_raw else datetime.now()

            items.append(
                RSSItem(
                    title=title,
                    description=description,
                    link=link,
                    published_date=published_date,
                    guid=str(guid),
                    source_feed=feed_url,
                    author=author,
                    categories=categories,
                    content=content_value,
                )
            )

        return items

    def _parse_date(self, date_str: str) -> datetime:
        """Parse various RSS date formats."""
        try:
            dt = dateutil_parser.parse(date_str)
            return dt.replace(tzinfo=None)
        except Exception:  # noqa: BLE001
            return datetime.now()

    def _generate_cache_key(self, feed_url: str, filter_quality: bool = False, max_items: int = 10) -> str:
        """Generate cache key for feed URL with parameters"""
        key_parts = [feed_url, str(filter_quality), str(max_items)]
        return hashlib.md5(':'.join(key_parts).encode()).hexdigest()

    def _extract_liturgical_date(self, content: str) -> str:
        """Extract liturgical date from USCCB content."""
        soup = BeautifulSoup(content or "", "html.parser")
        h2 = soup.find("h2")
        return h2.get_text(strip=True) if h2 else "Daily Readings"

    def _validate_and_clean_reference(self, reference: str, label_keywords: List[str]) -> str:
        """Validate and clean a scripture reference, providing fallbacks if needed."""
        if not reference or not reference.strip():
            self.logger.warning(f"Empty reference for keywords {label_keywords}")
            return self._get_fallback_reference(label_keywords)

        # Clean the reference
        cleaned = reference.strip()

        # Remove any remaining label keywords from the reference
        for keyword in label_keywords:
            if keyword.lower() in cleaned.lower():
                # Remove the keyword and common separators
                pattern = rf'\b{re.escape(keyword)}\b[:\s]*'
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        # Check if reference looks valid (contains letters, not just numbers/punctuation)
        if re.match(r'^[\d\-\.,;:\s]+$', cleaned):
            self.logger.warning(f"Reference appears incomplete (numbers only): '{cleaned}' for keywords {label_keywords}")
            # Still return it but add a warning - the expansion function might help
            return cleaned if cleaned else self._get_fallback_reference(label_keywords)

        # Check for minimum length
        if len(cleaned) < 2:
            self.logger.warning(f"Reference too short: '{cleaned}' for keywords {label_keywords}")
            return self._get_fallback_reference(label_keywords)

        self.logger.debug(f"Validated reference: '{cleaned}' for keywords {label_keywords}")
        return cleaned

    def _get_fallback_reference(self, label_keywords: List[str]) -> str:
        """Provide fallback reference names when parsing fails."""
        fallback_map = {
            "Reading 1": "Daily Reading",
            "Reading I": "Daily Reading",
            "First Reading": "Daily Reading",
            "Reading 2": "Second Reading",
            "Reading II": "Second Reading",
            "Second Reading": "Second Reading",
            "Gospel": "Daily Gospel",
            "Responsorial Psalm": "Daily Psalm"
        }

        for keyword in label_keywords:
            if keyword in fallback_map:
                self.logger.info(f"Using fallback reference '{fallback_map[keyword]}' for failed keyword '{keyword}'")
                return fallback_map[keyword]

        # Ultimate fallback
        return "Daily Reading"


    async def fetch_section_content(
        self, 
        section: str, 
        target_count: int = 5, 
        max_feeds: int = 5,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Fetch content for a specific newsletter section using RSS feeds.
        This is the main method that replaces llmlayer functionality.
        
        Args:
            section: Section name (breaking_news, business, tech_science, etc.)
            target_count: Target number of articles to return
            max_feeds: Maximum number of feeds to query
            hours_back: Look back this many hours for articles
            
        Returns:
            List of structured articles ready for the newsletter pipeline
        """
        if section not in self.section_feeds:
            self.logger.warning(f"No feeds configured for section: {section}")
            return []
            
        feed_ids = self.section_feeds[section][:max_feeds]
        all_articles = []
        seen_urls: Set[str] = set()
        
        self.logger.info(f"Fetching content for {section} from {len(feed_ids)} feeds")
        
        # Fetch from feeds in parallel
        fetch_tasks = []
        for feed_id in feed_ids:
            config = self.feeds[feed_id]
            task = self._fetch_feed_with_config(config, target_count * 2)
            fetch_tasks.append(task)
            
        # Wait for all feeds to complete (or timeout)
        feed_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(feed_results):
            if isinstance(result, Exception):
                feed_id = feed_ids[i]
                self.logger.warning(f"Feed {feed_id} failed: {result}")
                continue
                
            # Filter and process articles
            for item in result:
                if self._is_article_suitable(item, section, hours_back, seen_urls):
                    article = self._convert_rss_to_article(item, section)
                    all_articles.append(article)
                    seen_urls.add(item.link)
                    
                    if len(all_articles) >= target_count * 2:  # Get extra for better selection
                        break
                        
        # Sort by quality and recency
        all_articles.sort(key=lambda x: (
            x.get('priority_score', 0),
            x.get('published', '1970-01-01')
        ), reverse=True)
        
        # Return top articles
        selected = all_articles[:target_count]
        self.logger.info(f"Selected {len(selected)} articles for {section} from {len(all_articles)} candidates")
        
        return selected
        
    async def _fetch_feed_with_config(self, config: FeedConfig, max_items: int) -> List[RSSItem]:
        """Fetch a feed with its specific configuration."""
        try:
            return await self.fetch_feed(config.url, max_items=max_items, filter_quality=True)
        except Exception as e:
            self.logger.error(f"Failed to fetch {config.name}: {e}")
            return []
            
    def _is_article_suitable(
        self, 
        item: RSSItem, 
        section: str, 
        hours_back: int, 
        seen_urls: Set[str]
    ) -> bool:
        """Check if an RSS item is suitable for the given section."""
        # Skip duplicates
        if item.link in seen_urls:
            return False
            
        # Check recency
        if item.published_date:
            age_hours = (datetime.now() - item.published_date.replace(tzinfo=None)).total_seconds() / 3600
            if age_hours > hours_back:
                return False
                
        # Check for quality indicators
        if not self._passes_quality_filter(item, section):
            return False
            
        return True
        
    def _passes_quality_filter(self, item: RSSItem, section: str) -> bool:
        """Check if content passes quality filters."""
        title_lower = item.title.lower()
        desc_lower = (item.description or '').lower()
        content_text = f"{title_lower} {desc_lower}"
        
        # Exclude low-quality patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, content_text, re.IGNORECASE):
                return False
                
        # Check for low-quality keywords
        low_quality_count = sum(
            1 for keyword in self.quality_keywords['low_quality']
            if keyword in content_text
        )
        
        # Check for high-quality keywords
        high_quality_count = sum(
            1 for keyword in self.quality_keywords['high_quality']
            if keyword in content_text
        )
        
        # Simple quality scoring
        quality_score = high_quality_count - (low_quality_count * 2)
        
        # Section-specific quality thresholds
        thresholds = {
            'breaking_news': -1,      # More lenient for breaking news
            'business': 0,            # Standard threshold
            'tech_science': 0,        # Standard threshold
            'research_papers': 1,     # Higher threshold for academic content
            'politics': 0,            # Standard threshold
            'miscellaneous': 1,       # Higher threshold for intellectual content
        }
        
        return quality_score >= thresholds.get(section, 0)
        
    def _convert_rss_to_article(self, item: RSSItem, section: str) -> Dict[str, Any]:
        """Convert RSS item to article format expected by newsletter pipeline."""
        # Calculate priority score based on source and content
        source_priority = 0
        for feed_id, config in self.feeds.items():
            if config.url == item.source_feed:
                source_priority = config.priority
                break
                
        content_score = self._calculate_content_score(item, section)
        priority_score = source_priority + content_score
        
        # Extract clean summary
        summary = self._extract_clean_summary(item)
        
        return {
            'headline': item.title,
            'url': item.link,
            'summary_text': summary,
            'source': self._extract_source_name(item.source_feed),
            'published': item.published_date.isoformat() if item.published_date else None,
            'priority_score': priority_score,
            'section': section,
            'metadata': {
                'author': item.author,
                'categories': item.categories or [],
                'guid': item.guid,
                'feed_url': item.source_feed,
            }
        }
        
    def _calculate_content_score(self, item: RSSItem, section: str) -> float:
        """Calculate content quality score based on various factors."""
        score = 0.0
        content_text = f"{item.title} {item.description or ''}".lower()
        
        # High-quality indicators
        for keyword in self.quality_keywords['high_quality']:
            if keyword in content_text:
                score += 0.5
                
        # Section-specific keywords
        if section in self.quality_keywords:
            for keyword in self.quality_keywords[section]:
                if keyword in content_text:
                    score += 0.3
                    
        # Title length (optimal range)
        title_len = len(item.title)
        if 30 <= title_len <= 100:
            score += 0.2
        elif title_len > 150:
            score -= 0.3
            
        # Description quality
        if item.description and len(item.description) > 50:
            score += 0.2
            
        # Recency boost (more recent = higher score)
        if item.published_date:
            age_hours = (datetime.now() - item.published_date.replace(tzinfo=None)).total_seconds() / 3600
            if age_hours < 6:
                score += 0.3
            elif age_hours < 24:
                score += 0.1
                
        return score
        
    def _extract_clean_summary(self, item: RSSItem) -> str:
        """Extract and clean summary text from RSS item."""
        # Prefer full content over description
        text = item.content or item.description or ''
        
        # Remove HTML tags
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text(' ', strip=True)
            
        # Clean up common RSS artifacts
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'Continue reading.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Read more.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        
        # Truncate to reasonable length
        if len(text) > 300:
            text = text[:297] + '...'
            
        return text.strip()
        
    def _extract_source_name(self, feed_url: str) -> str:
        """Extract clean source name from feed URL."""
        # Check if we have a configured name for this URL
        for config in self.feeds.values():
            if config.url == feed_url:
                return config.name
                
        # Extract from URL
        parsed = urlparse(feed_url)
        domain = parsed.netloc.replace('www.', '')
        
        # Clean up common domain suffixes
        if domain.endswith('.com'):
            domain = domain[:-4]
        elif domain.endswith('.org'):
            domain = domain[:-4]
            
        return domain.replace('.', ' ').title()
        
    def _filter_content_quality(self, items: List[RSSItem]) -> List[RSSItem]:
        """Filter RSS items based on quality indicators."""
        filtered_items = []
        
        for item in items:
            # Basic quality checks
            if not item.title or len(item.title) < 10:
                continue
                
            if not item.link or not item.link.startswith('http'):
                continue
                
            # Content quality check
            if self._passes_basic_quality_check(item):
                filtered_items.append(item)
                
        return filtered_items
        
    def _passes_basic_quality_check(self, item: RSSItem) -> bool:
        """Basic quality check for RSS items."""
        title_lower = item.title.lower()
        
        # Skip obvious spam/low-quality content
        spam_indicators = [
            'click here', 'amazing', 'shocking', 'unbelievable',
            'you won\'t believe', 'this one trick', 'doctors hate'
        ]
        
        for indicator in spam_indicators:
            if indicator in title_lower:
                return False
                
        return True
        
    async def get_section_feeds_status(self) -> Dict[str, Any]:
        """Get status information for all section feeds."""
        status = {}
        
        for section, feed_ids in self.section_feeds.items():
            section_status = {
                'total_feeds': len(feed_ids),
                'feeds': [],
                'last_check': datetime.now().isoformat(),
            }
            
            for feed_id in feed_ids:
                config = self.feeds[feed_id]
                feed_status = {
                    'name': config.name,
                    'url': config.url,
                    'priority': config.priority,
                    'max_age_hours': config.max_age_hours,
                    'check_frequency_hours': config.check_frequency_hours,
                }
                section_status['feeds'].append(feed_status)
                
            status[section] = section_status
            
        return status


class RSSServiceError(Exception):
    """Custom exception for RSS service failures"""
    pass


# Factory functions to create the appropriate RSS service
def create_rss_service():
    """Create RSS service using available dependencies."""
    if EXTERNAL_DEPS_AVAILABLE:
        return RSSService()
    else:
        logging.warning("External dependencies not available, using simple RSS implementation")
        return SimpleRSSService()


def create_rss_adapter(rss_service=None):
    """Create RSS adapter using available dependencies.""" 
    if rss_service is None:
        rss_service = create_rss_service()
        
    if EXTERNAL_DEPS_AVAILABLE:
        from src.services.rss_content_adapter import RSSContentAdapter
        return RSSContentAdapter(rss_service)
    else:
        return SimpleRSSAdapter(rss_service)


