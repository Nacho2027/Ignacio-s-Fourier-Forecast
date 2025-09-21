"""
Simplified RSS service using only built-in Python libraries.
This is a fallback implementation when external dependencies are not available.
"""

import asyncio
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import re
import hashlib
import csv
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class SimpleRSSItem:
    """Simple RSS feed item using built-in types only"""
    title: str
    description: str
    link: str
    published_date: datetime
    guid: str
    source_feed: str
    author: Optional[str] = None
    categories: Optional[List[str]] = None


@dataclass
class SimpleFeedConfig:
    """Simple feed configuration"""
    name: str
    url: str
    section: str
    priority: int = 1
    max_age_hours: int = 168


class SimpleRSSService:
    """
    Simple RSS service using only built-in Python libraries.
    Provides basic functionality to replace llmlayer.
    """
    
    def __init__(self):
        """Initialize simple RSS service."""
        self.timeout = 30
        self.logger = logging.getLogger(__name__)
        
        # Load basic feed configurations
        self.feeds = self._init_basic_feeds()
        self.section_feeds = self._organize_feeds_by_section()
        
        self.logger.info(f"Simple RSS service initialized with {len(self.feeds)} feeds")
    
    def _init_basic_feeds(self) -> Dict[str, SimpleFeedConfig]:
        """Initialize basic feed configurations."""
        feeds = {
            # Breaking News
            "bbc_news": SimpleFeedConfig(
                name="BBC News",
                url="http://feeds.bbci.co.uk/news/rss.xml",
                section="breaking_news",
                priority=10,
                max_age_hours=24
            ),
            "reuters_news": SimpleFeedConfig(
                name="Reuters",
                url="http://feeds.reuters.com/reuters/topNews",
                section="breaking_news", 
                priority=9,
                max_age_hours=24
            ),
            
            # Business
            "reuters_business": SimpleFeedConfig(
                name="Reuters Business",
                url="http://feeds.reuters.com/reuters/businessNews",
                section="business",
                priority=8,
                max_age_hours=48
            ),
            
            # Tech & Science
            "techcrunch": SimpleFeedConfig(
                name="TechCrunch",
                url="http://feeds.feedburner.com/TechCrunch/",
                section="tech_science",
                priority=7,
                max_age_hours=168
            ),
            "ars_technica": SimpleFeedConfig(
                name="Ars Technica",
                url="http://feeds.arstechnica.com/arstechnica/index",
                section="tech_science",
                priority=8,
                max_age_hours=168
            ),
            "wired": SimpleFeedConfig(
                name="Wired",
                url="https://www.wired.com/feed/",
                section="tech_science",
                priority=7,
                max_age_hours=168
            ),
            
            # Politics
            "wapo_politics": SimpleFeedConfig(
                name="Washington Post Politics",
                url="https://feeds.washingtonpost.com/rss/politics",
                section="politics",
                priority=8,
                max_age_hours=48
            ),
            "cnn_politics": SimpleFeedConfig(
                name="CNN Politics",
                url="http://rss.cnn.com/rss/cnn_allpolitics.rss",
                section="politics",
                priority=7,
                max_age_hours=48
            ),
            "the_hill": SimpleFeedConfig(
                name="The Hill",
                url="https://thehill.com/news/feed/",
                section="politics",
                priority=7,
                max_age_hours=48
            ),
            
            # Local News
            "local_abc7": SimpleFeedConfig(
                name="ABC7 New York",
                url="https://abc7ny.com/feed/",
                section="local",
                priority=8,
                max_age_hours=48
            ),
            "local_nbc4": SimpleFeedConfig(
                name="NBC New York",
                url="https://www.nbcnewyork.com/feed/",
                section="local", 
                priority=8,
                max_age_hours=48
            ),
            "gothamist": SimpleFeedConfig(
                name="Gothamist",
                url="https://gothamist.com/feed",
                section="local",
                priority=7,
                max_age_hours=48
            ),
            "ny_post_local": SimpleFeedConfig(
                name="NY Post Metro",
                url="https://nypost.com/metro/feed/",
                section="local",
                priority=6,
                max_age_hours=48
            ),
            
            # Miscellaneous (intellectual/cultural content)
            "atlantic": SimpleFeedConfig(
                name="The Atlantic",
                url="https://www.theatlantic.com/feed/all/",
                section="miscellaneous",
                priority=9,
                max_age_hours=336  # 2 weeks
            ),
            "new_yorker": SimpleFeedConfig(
                name="The New Yorker",
                url="https://www.newyorker.com/feed/everything",
                section="miscellaneous",
                priority=9,
                max_age_hours=336
            ),
            "aeon": SimpleFeedConfig(
                name="Aeon Magazine",
                url="https://aeon.co/feed.rss",
                section="miscellaneous",
                priority=8,
                max_age_hours=720  # 30 days for deeper content
            ),
        }
        
        # Try to load from CSV if available
        csv_path = "/code/InternalDocs/RSS.csv"
        if os.path.exists(csv_path):
            feeds.update(self._load_basic_feeds_from_csv(csv_path))
            
        return feeds
    
    def _load_basic_feeds_from_csv(self, csv_path: str) -> Dict[str, SimpleFeedConfig]:
        """Load basic feeds from CSV, using only essential ones."""
        feeds = {}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for i, row in enumerate(reader):
                    if i >= 20:  # Limit to first 20 feeds to avoid overwhelming
                        break
                        
                    section = row.get('Section', '').lower().replace(' ', '_').replace('&', 'and')
                    source = row.get('Source', '').strip()
                    url = row.get('RSS Feed URL', '').strip()
                    
                    if not url or not section or not source:
                        continue
                        
                    # Map sections
                    section_map = {
                        'breaking_news': 'breaking_news',
                        'business': 'business',
                        'tech_and_science': 'tech_science', 
                        'politics': 'politics',
                        'miscellaneous': 'miscellaneous'
                    }
                    
                    internal_section = section_map.get(section)
                    if not internal_section:
                        continue
                        
                    feed_id = f"{internal_section}_{source.lower().replace(' ', '_').replace('.', '')}"
                    
                    feeds[feed_id] = SimpleFeedConfig(
                        name=source,
                        url=url,
                        section=internal_section,
                        priority=5,  # Default priority
                        max_age_hours=168
                    )
                    
        except Exception as e:
            self.logger.error(f"Error loading CSV feeds: {e}")
            
        return feeds
    
    def _organize_feeds_by_section(self) -> Dict[str, List[str]]:
        """Organize feeds by section."""
        section_feeds = {}
        
        for feed_id, config in self.feeds.items():
            section = config.section
            if section not in section_feeds:
                section_feeds[section] = []
            section_feeds[section].append(feed_id)
        
        # Sort by priority
        for section in section_feeds:
            section_feeds[section].sort(
                key=lambda fid: self.feeds[fid].priority,
                reverse=True
            )
            
        return section_feeds
    
    def fetch_feed(self, feed_url: str, max_items: int = 10) -> List[SimpleRSSItem]:
        """Fetch and parse RSS feed using urllib."""
        try:
            # Create request with headers
            req = urllib.request.Request(
                feed_url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'
                }
            )
            
            # Fetch content
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode('utf-8', errors='ignore')
            
            # Parse XML
            items = self._parse_simple_rss(content, feed_url)
            
            return items[:max_items] if max_items else items
            
        except Exception as e:
            self.logger.error(f"Error fetching feed {feed_url}: {e}")
            return []
    
    def _parse_simple_rss(self, content: str, feed_url: str) -> List[SimpleRSSItem]:
        """Parse RSS content using xml.etree."""
        items = []
        
        try:
            root = ET.fromstring(content)
            
            # Handle different RSS formats
            item_elements = []
            
            # Try RSS 2.0 format
            for item in root.findall('.//item'):
                item_elements.append(item)
                
            # Try Atom format
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                item_elements.append(entry)
            
            for item in item_elements:
                try:
                    # Extract basic fields
                    title = self._get_text(item, ['title'])
                    description = self._get_text(item, ['description', 'summary', 'content'])
                    link = self._get_text(item, ['link', 'guid'])
                    
                    # Handle Atom links
                    if not link:
                        link_elem = item.find('.//{http://www.w3.org/2005/Atom}link')
                        if link_elem is not None:
                            link = link_elem.get('href', '')
                    
                    # Parse date
                    date_text = self._get_text(item, ['pubDate', 'published', 'updated'])
                    published_date = self._parse_simple_date(date_text)
                    
                    # Create GUID
                    guid = self._get_text(item, ['guid']) or link or title
                    
                    if title and link:
                        rss_item = SimpleRSSItem(
                            title=title,
                            description=description or '',
                            link=link,
                            published_date=published_date,
                            guid=guid,
                            source_feed=feed_url
                        )
                        items.append(rss_item)
                        
                except Exception as e:
                    self.logger.debug(f"Error parsing RSS item: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error parsing RSS XML: {e}")
            
        return items
    
    def _get_text(self, element, tag_names: List[str]) -> str:
        """Get text from element by trying multiple tag names."""
        for tag_name in tag_names:
            # Try without namespace
            elem = element.find(tag_name)
            if elem is not None and elem.text:
                return elem.text.strip()
                
            # Try with common namespaces
            for ns in ['', '{http://www.w3.org/2005/Atom}']:
                elem = element.find(f'{ns}{tag_name}')
                if elem is not None and elem.text:
                    return elem.text.strip()
                    
        return ''
    
    def _parse_simple_date(self, date_str: str) -> datetime:
        """Parse date string into datetime."""
        if not date_str:
            return datetime.now()
            
        # Remove timezone info for simplicity  
        date_str = re.sub(r'\s*[+-]\d{4}$', '', date_str)
        date_str = re.sub(r'\s*[A-Z]{3,4}$', '', date_str)
        
        # Try common formats
        formats = [
            "%a, %d %b %Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d %b %Y %H:%M:%S",
            "%Y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
                
        return datetime.now()
    
    def fetch_section_content(
        self,
        section: str,
        target_count: int = 5,
        max_feeds: int = 3
    ) -> List[Dict[str, Any]]:
        """Fetch content for a section (synchronous version)."""
        if section not in self.section_feeds:
            return []
            
        feed_ids = self.section_feeds[section][:max_feeds]
        all_articles = []
        seen_urls = set()
        
        for feed_id in feed_ids:
            config = self.feeds[feed_id]
            
            try:
                items = self.fetch_feed(config.url, max_items=target_count * 2)
                
                for item in items:
                    if item.link in seen_urls:
                        continue
                        
                    # Check recency
                    age_hours = (datetime.now() - item.published_date).total_seconds() / 3600
                    if age_hours > config.max_age_hours:
                        continue
                        
                    article = {
                        'headline': item.title,
                        'url': item.link,
                        'summary_text': item.description,
                        'source': config.name,
                        'published': item.published_date.isoformat(),
                        'section': section,
                        'priority_score': config.priority / 10.0,  # Normalize
                        'metadata': {
                            'feed_url': config.url,
                            'guid': item.guid
                        }
                    }
                    
                    all_articles.append(article)
                    seen_urls.add(item.link)
                    
                    if len(all_articles) >= target_count * 2:
                        break
                        
            except Exception as e:
                self.logger.error(f"Error fetching from {config.name}: {e}")
                continue
        
        # Sort by priority and recency
        all_articles.sort(key=lambda x: (
            x.get('priority_score', 0),
            x.get('published', '1970-01-01')
        ), reverse=True)
        
        return all_articles[:target_count]


class SimpleRSSAdapter:
    """Simple RSS adapter that mimics the full RSSContentAdapter interface."""
    
    def __init__(self, rss_service: SimpleRSSService):
        self.rss_service = rss_service
        self.logger = logging.getLogger(__name__)
        
        self.section_configs = {
            "breaking_news": {"target_count": 10, "max_feeds": 3},
            "business": {"target_count": 8, "max_feeds": 3},
            "tech_science": {"target_count": 8, "max_feeds": 3}, 
            "politics": {"target_count": 6, "max_feeds": 2},
            "miscellaneous": {"target_count": 15, "max_feeds": 4}
        }
    
    async def search_optimized_rate_limited(self, section: str, custom_query: Optional[str] = None):
        """Async wrapper for section content fetching."""
        config = self.section_configs.get(section, {"target_count": 5, "max_feeds": 2})
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        articles = await loop.run_in_executor(
            None,
            self.rss_service.fetch_section_content,
            section,
            config["target_count"],
            config["max_feeds"]
        )
        
        # Return result in expected format
        return type('RSSAdapterResult', (), {
            'query': custom_query or f"{section} content",
            'articles': articles,
            'search_time_ms': 100.0,  # Dummy value
            'total_results': len(articles),
            'cached': False,
            'section': section
        })()
    
    def get_section_config(self, section: str) -> Dict[str, Any]:
        """Get section configuration."""
        return self.section_configs.get(section, {})
        
    def update_section_config(self, section: str, config: Dict[str, Any]) -> None:
        """Update section configuration."""
        if section in self.section_configs:
            self.section_configs[section].update(config)