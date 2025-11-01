"""
Date extraction utilities for extracting publication dates from URLs and content.
"""

import re
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def extract_date_from_url(url: str) -> Optional[datetime]:
    """
    Extract publication date from URL patterns commonly used by news sites.
    
    Supports patterns like:
    - /2025/10/28/article-title
    - /2025-10-28/article-title
    - /article-title-2025-10-28
    - /20251028/article-title
    
    Args:
        url: The article URL to extract date from
        
    Returns:
        datetime object if date found, None otherwise
    """
    if not url:
        return None
    
    try:
        # Pattern 1: /YYYY/MM/DD/ (most common)
        pattern1 = r'/(\d{4})/(\d{1,2})/(\d{1,2})/'
        match = re.search(pattern1, url)
        if match:
            year, month, day = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
                logger.debug(f"Extracted date from URL pattern /YYYY/MM/DD/: {dt.date()}")
                return dt
            except ValueError:
                pass  # Invalid date, try next pattern
        
        # Pattern 2: /YYYY-MM-DD/ or /YYYY-MM-DD-
        pattern2 = r'/(\d{4})-(\d{1,2})-(\d{1,2})[-/]'
        match = re.search(pattern2, url)
        if match:
            year, month, day = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
                logger.debug(f"Extracted date from URL pattern /YYYY-MM-DD/: {dt.date()}")
                return dt
            except ValueError:
                pass
        
        # Pattern 3: -YYYY-MM-DD (at end or middle)
        pattern3 = r'-(\d{4})-(\d{1,2})-(\d{1,2})'
        match = re.search(pattern3, url)
        if match:
            year, month, day = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
                logger.debug(f"Extracted date from URL pattern -YYYY-MM-DD: {dt.date()}")
                return dt
            except ValueError:
                pass
        
        # Pattern 4: /YYYYMMDD/ (compact format)
        pattern4 = r'/(\d{8})/'
        match = re.search(pattern4, url)
        if match:
            date_str = match.group(1)
            try:
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                logger.debug(f"Extracted date from URL pattern /YYYYMMDD/: {dt.date()}")
                return dt
            except (ValueError, IndexError):
                pass
        
        # Pattern 5: /YYYY/MM/ (month-level precision)
        pattern5 = r'/(\d{4})/(\d{1,2})/'
        match = re.search(pattern5, url)
        if match:
            year, month = match.groups()
            try:
                # Use first day of month
                dt = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
                logger.debug(f"Extracted date from URL pattern /YYYY/MM/ (month-level): {dt.date()}")
                return dt
            except ValueError:
                pass
        
        # No date pattern found
        logger.debug(f"No date pattern found in URL: {url[:100]}...")
        return None
        
    except Exception as e:
        logger.warning(f"Error extracting date from URL: {e}")
        return None


def extract_date_from_content(content: str) -> Optional[datetime]:
    """
    Extract publication date from article content.
    
    Looks for patterns like:
    - "Published: October 28, 2025"
    - "Published on 2025-10-28"
    - "October 28, 2025"
    
    Args:
        content: The article content to extract date from
        
    Returns:
        datetime object if date found, None otherwise
    """
    if not content:
        return None
    
    try:
        # Pattern 1: "Published: Month DD, YYYY" or "Published on Month DD, YYYY"
        pattern1 = r'[Pp]ublished[:\s]+(?:on\s+)?([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})'
        match = re.search(pattern1, content[:500])  # Check first 500 chars
        if match:
            month_name, day, year = match.groups()
            try:
                dt = datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
                dt = dt.replace(tzinfo=timezone.utc)
                logger.debug(f"Extracted date from content pattern 'Published: Month DD, YYYY': {dt.date()}")
                return dt
            except ValueError:
                pass
        
        # Pattern 2: "Published: YYYY-MM-DD" or "Published on YYYY-MM-DD"
        pattern2 = r'[Pp]ublished[:\s]+(?:on\s+)?(\d{4})-(\d{1,2})-(\d{1,2})'
        match = re.search(pattern2, content[:500])
        if match:
            year, month, day = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
                logger.debug(f"Extracted date from content pattern 'Published: YYYY-MM-DD': {dt.date()}")
                return dt
            except ValueError:
                pass
        
        # No date pattern found
        logger.debug("No date pattern found in content")
        return None
        
    except Exception as e:
        logger.warning(f"Error extracting date from content: {e}")
        return None


def get_best_date(
    published_date: Optional[datetime],
    url: Optional[str],
    content: Optional[str],
    fallback_days: int = 3
) -> datetime:
    """
    Get the best available publication date using multiple sources.
    
    Priority:
    1. published_date (if provided and valid)
    2. Date extracted from URL
    3. Date extracted from content
    4. Fallback to N days ago (default: 3 days)
    
    Args:
        published_date: The published_date from the data source (may be None)
        url: The article URL
        content: The article content
        fallback_days: Number of days ago to use as fallback (default: 3)
        
    Returns:
        datetime object (always returns a valid date)
    """
    from datetime import timedelta
    
    # Priority 1: Use provided published_date if valid
    if published_date:
        logger.debug(f"Using provided published_date: {published_date}")
        return published_date
    
    # Priority 2: Extract from URL
    url_date = extract_date_from_url(url) if url else None
    if url_date:
        logger.info(f"✅ Extracted date from URL: {url_date.date()}")
        return url_date
    
    # Priority 3: Extract from content
    content_date = extract_date_from_content(content) if content else None
    if content_date:
        logger.info(f"✅ Extracted date from content: {content_date.date()}")
        return content_date
    
    # Priority 4: Fallback to N days ago
    fallback_date = datetime.now(timezone.utc) - timedelta(days=fallback_days)
    logger.debug(f"Using fallback date ({fallback_days} days ago): {fallback_date.date()}")
    return fallback_date

