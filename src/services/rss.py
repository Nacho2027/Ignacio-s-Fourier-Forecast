import asyncio
import aiohttp
import feedparser
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from urllib.parse import urlparse
import re
import subprocess
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser
from zoneinfo import ZoneInfo
from src.services.cache_service import ContentItem
import logging

EASTERN_TZ = ZoneInfo("America/New_York")


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


class RSSService:
    """
    RSS feed integration with special support for USCCB daily readings.
    """

    def __init__(self):
        """Initialize RSS service."""
        self.timeout = 30  # seconds
        self.max_retries = 3
        self.logger = logging.getLogger(__name__)

        # Configured feeds
        self.feeds = self._init_feed_configs()

        # Cache for the session
        self._cache: Dict[str, List[RSSItem]] = {}
        self._readings_cache: Dict[str, DailyReading] = {}

    def _init_feed_configs(self) -> Dict[str, FeedConfig]:
        """Initialize feed configurations"""
        return {
            "usccb_daily": FeedConfig(
                name="USCCB Daily Readings",
                url="https://bible.usccb.org/readings.rss",
                section="spiritual",
                check_frequency_hours=24,
                parse_full_content=True,
                special_parser="usccb",
            ),
            "catholic_daily_reflections": FeedConfig(
                name="Catholic Daily Reflections",
                url="https://catholic-daily-reflections.com/feed/",
                section="spiritual",
                check_frequency_hours=24,
                parse_full_content=True,
                special_parser=None,  # Use standard RSS parsing
            ),
            "vatican_news": FeedConfig(
                name="Vatican News",
                url="https://www.vaticannews.va/en/pope.rss",
                section="spiritual",
                check_frequency_hours=12,
                parse_full_content=False,
            ),
        }

    async def fetch_feed(self, feed_url: str, max_items: int = 10) -> List[RSSItem]:
        """
        Fetch and parse an RSS feed.
        """
        cache_key = self._generate_cache_key(feed_url)
        # Don't use cache for daily readings to ensure fresh content
        if "usccb" in feed_url or "catholic" in feed_url:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self.logger.debug(f"Cleared cache for spiritual feed: {feed_url}")
        elif cache_key in self._cache:
            return self._cache[cache_key]

        last_error: Optional[Exception] = None
        content: Optional[str] = None
        for _ in range(self.max_retries):
            try:
                content = await self._fetch_with_retry(feed_url)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(0.25)
        if content is None:
            raise RSSServiceError(str(last_error) if last_error else "Failed to fetch feed")
        items = self._parse_feed(content, feed_url)
        if max_items:
            items = items[:max_items]
        self._cache[cache_key] = items
        return items

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
        now_et = datetime.now(EASTERN_TZ)
        target_date = (date.astimezone(EASTERN_TZ) if date and date.tzinfo else (date or now_et)).date()
        
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
        reading.date = datetime.combine(target_date, datetime.min.time(), tzinfo=EASTERN_TZ)
        self._readings_cache[cache_key] = reading
        return reading

    async def get_todays_spiritual_content(self) -> Dict[str, Any]:
        """Get today's complete spiritual content."""
        readings = await self.get_daily_readings()
        content: Dict[str, Any] = {"readings": readings}
        return content

    async def _fetch_usccb_daily(self) -> List[RSSItem]:
        """Fetch today's USCCB daily reading page and wrap as RSSItem."""
        now_et = datetime.now(EASTERN_TZ)
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
            date=datetime.now(EASTERN_TZ),
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

    def _generate_cache_key(self, feed_url: str) -> str:
        """Generate cache key for feed URL"""
        return hashlib.md5(feed_url.encode()).hexdigest()

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
            return cleaned or self._get_fallback_reference(label_keywords)

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


class RSSServiceError(Exception):
    """Custom exception for RSS service failures"""
    pass


