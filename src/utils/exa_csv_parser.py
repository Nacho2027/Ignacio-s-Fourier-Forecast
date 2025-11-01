"""
Exa CSV Parser

Converts Exa Websets item dictionaries to ContentItem objects for the newsletter pipeline.

Handles:
- Field mapping from Exa format to internal format
- Missing field defaults
- Type conversion and validation
- Source extraction from URLs
- Metadata preservation
"""

import logging
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse

from src.services.cache_service import ContentItem
from src.utils.date_extraction import extract_date_from_url, extract_date_from_content


class ExaCsvParserError(Exception):
    """Exception raised for CSV parsing errors."""
    pass


class ExaCsvParser:
    """
    Parser for converting Exa Websets items to ContentItem objects.

    Provides robust parsing with sensible defaults for missing fields.
    """

    def __init__(self, section: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.section = section  # Track section for section-aware date defaults
        
        # Source domain mapping for authority scoring
        self.source_domains = {
            "reuters.com": "Reuters",
            "ft.com": "Financial Times",
            "bloomberg.com": "Bloomberg",
            "wsj.com": "Wall Street Journal",
            "nytimes.com": "New York Times",
            "washingtonpost.com": "Washington Post",
            "economist.com": "The Economist",
            "nature.com": "Nature",
            "science.org": "Science",
            "arxiv.org": "arXiv",
            "ieee.org": "IEEE",
            "acm.org": "ACM",
            "theinformation.com": "The Information",
            "axios.com": "Axios",
            "politico.com": "Politico",
            "thehill.com": "The Hill",
            "npr.org": "NPR",
            "pbs.org": "PBS",
            "ap.org": "Associated Press",
            "newyorker.com": "The New Yorker",
            "theatlantic.com": "The Atlantic",
            "harpers.org": "Harper's",
            "aeon.co": "Aeon",
            "quantamagazine.org": "Quanta Magazine",
            "spectrum.ieee.org": "IEEE Spectrum",
            "arstechnica.com": "Ars Technica",
        }
    
    def parse_items(
        self,
        items: List[Dict[str, Any]],
        section: str,
    ) -> List[ContentItem]:
        """
        Parse a list of Exa items into ContentItem objects.
        
        Args:
            items: List of item dictionaries from Exa
            section: Newsletter section name
            
        Returns:
            List of ContentItem objects
            
        Raises:
            ExaCsvParserError: If parsing fails critically
        """
        self.logger.info(f"Parsing {len(items)} Exa items for section: {section}")
        
        content_items = []
        parse_errors = 0
        
        for idx, item in enumerate(items):
            try:
                content_item = self._parse_single_item(item, section)
                if content_item:
                    content_items.append(content_item)
                else:
                    parse_errors += 1
                    
            except Exception as e:
                self.logger.warning(
                    f"Failed to parse item {idx + 1}/{len(items)}: {e}"
                )
                parse_errors += 1
        
        self.logger.info(
            f"Parsed {len(content_items)} items successfully, "
            f"{parse_errors} errors for section: {section}"
        )
        
        return content_items
    
    def _parse_single_item(
        self,
        item: Dict[str, Any],
        section: str,
    ) -> Optional[ContentItem]:
        """
        Parse a single Exa item into a ContentItem.
        
        Args:
            item: Item dictionary from Exa
            section: Newsletter section name
            
        Returns:
            ContentItem object or None if parsing fails
        """
        # Extract required fields with defaults
        url = item.get("url", "")
        if not url:
            self.logger.warning("Item missing URL, skipping")
            return None
        
        # Get title (may be in different fields depending on entity type)
        title = (
            item.get("title") or
            item.get("headline") or
            item.get("description", "")[:100]  # Fallback to description
        )
        
        if not title:
            self.logger.warning(f"Item missing title: {url}")
            return None
        
        # Extract description/summary
        # Priority: enrichment_summary > abstract > description > title
        enrichment_summary = item.get("enrichment_summary", "")
        description = item.get("description", "")
        abstract = item.get("abstract", "")
        summary_text = enrichment_summary or abstract or description or title

        # Log if using enrichment
        if enrichment_summary:
            self.logger.debug(f"✨ Using enrichment summary for: {title[:50]}... ({len(enrichment_summary)} chars)")
        
        # Extract published date with multiple fallback strategies
        # Priority: 1) API published_date, 2) URL extraction, 3) Content extraction, 4) Section fallback
        date_raw = item.get("published_date") or item.get("date") or ""
        published_date = self._parse_date(date_raw) if date_raw else None

        # If no date from API, try extracting from URL
        if not published_date:
            url_date = extract_date_from_url(url)
            if url_date:
                published_date = url_date
                self.logger.info(f"✅ Extracted date from URL for: {title[:50]}... -> {url_date.date()}")

        # If still no date, try extracting from content
        if not published_date and enrichment_summary:
            content_date = extract_date_from_content(enrichment_summary)
            if content_date:
                published_date = content_date
                self.logger.info(f"✅ Extracted date from content for: {title[:50]}... -> {content_date.date()}")

        # Log if we still don't have a date (will use section-aware fallback later)
        if not published_date:
            self.logger.debug(
                f"No published_date for item: {title[:50]}... "
                f"(will use section-aware fallback in aggregator)"
            )
        
        # Extract source from URL
        source = self._extract_source(url)
        
        # Generate unique ID
        item_id = self._generate_id(url, title)
        
        # Build metadata
        metadata = {
            "exa_item_id": item.get("id", ""),
            "entity_type": item.get("type", "article"),
            "source_url": url,
            "source_domain": urlparse(url).netloc,
        }
        
        # Add verification/evaluation data if present
        if "evaluations" in item:
            metadata["exa_evaluations"] = item["evaluations"]
        
        # Add author information if present
        if "author" in item:
            metadata["author"] = item["author"]
        elif "authors" in item:
            metadata["authors"] = item["authors"]
        
        # CRITICAL: Store enrichment_summary in metadata so it can be extracted later
        # This ensures enrichment_summary is preserved through the pipeline
        if enrichment_summary:
            metadata['enrichment_summary'] = enrichment_summary
        
        # Create ContentItem
        content_item = ContentItem(
            id=item_id,
            source=source,
            section=section,
            headline=title,
            summary_text=summary_text,
            url=url,
            published_date=published_date,
            metadata=metadata,
        )
        
        return content_item
    
    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse date string to datetime object.

        Args:
            date_str: Date string in various formats

        Returns:
            datetime object (defaults to section-specific lookback if parsing fails)
        """
        if not date_str:
            # Use section-specific lookback periods when date is missing
            # Trust that Exa's date criteria filtered correctly, but we don't have the metadata
            section_lookback = {
                "breaking_news": 2,      # Assume 2 days old for breaking news
                "business": 3,           # Assume 3 days old for business
                "tech_science": 4,       # Assume 4 days old for tech/science
                "politics": 2,           # Assume 2 days old for politics
                "miscellaneous": 7,      # Assume 1 week old for miscellaneous
                "research_papers": 30,   # Assume 1 month old for research papers
            }
            days_ago = section_lookback.get(self.section, 7)  # Default 7 days if section unknown
            self.logger.info(
                f"📅 No date provided for {self.section or 'unknown'} section item, "
                f"using section-aware estimate: {days_ago} days ago (trusting Exa's date filtering)"
            )
            return datetime.now() - timedelta(days=days_ago)

        # Try common date formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",  # With timezone
            "%Y-%m-%dT%H:%M:%S.%f%z",  # With microseconds and timezone
        ]

        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                self.logger.debug(f"✅ Successfully parsed date '{date_str}' using format '{fmt}'")
                return parsed
            except ValueError:
                continue

        # If all formats fail, try ISO format
        try:
            parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            self.logger.debug(f"✅ Successfully parsed date '{date_str}' using ISO format")
            return parsed
        except Exception as e:
            # Use section-specific fallback
            section_lookback = {
                "breaking_news": 2,
                "business": 3,
                "tech_science": 4,
                "politics": 2,
                "miscellaneous": 7,
                "research_papers": 30,
            }
            days_ago = section_lookback.get(self.section, 7)
            self.logger.warning(
                f"⚠️ Failed to parse date: '{date_str}' (error: {e}), "
                f"using section-aware fallback: {days_ago} days ago for {self.section or 'unknown'}"
            )
            return datetime.now() - timedelta(days=days_ago)
    
    def _extract_source(self, url: str) -> str:
        """
        Extract source name from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Source name (e.g., "Reuters", "Bloomberg")
        """
        try:
            domain = urlparse(url).netloc
            
            # Remove www. prefix
            domain = domain.replace("www.", "")
            
            # Check known sources
            for known_domain, source_name in self.source_domains.items():
                if known_domain in domain:
                    return source_name
            
            # Fallback: capitalize domain name
            base_domain = domain.split(".")[0]
            return base_domain.capitalize()
            
        except Exception as e:
            self.logger.warning(f"Failed to extract source from URL {url}: {e}")
            return "Unknown"
    
    def _generate_id(self, url: str, title: str) -> str:
        """
        Generate unique ID for content item.
        
        Args:
            url: Article URL
            title: Article title
            
        Returns:
            Unique ID string
        """
        # Use URL as primary identifier, fallback to title hash
        if url:
            return hashlib.sha256(url.encode()).hexdigest()[:16]
        else:
            return hashlib.sha256(title.encode()).hexdigest()[:16]


def parse_exa_items(
    items: List[Dict[str, Any]],
    section: str,
) -> List[ContentItem]:
    """
    Convenience function to parse Exa items.

    Args:
        items: List of item dictionaries from Exa
        section: Newsletter section name

    Returns:
        List of ContentItem objects
    """
    parser = ExaCsvParser(section=section)  # Pass section for section-aware date defaults
    return parser.parse_items(items, section)

