"""
Content models for the newsletter system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class NewsItem:
    """Represents a news item from any source."""
    
    headline: str
    summary_text: str
    url: str
    source: str
    published_date: Optional[datetime] = None
    source_url: Optional[str] = None
    time_ago: Optional[str] = None
    
    def __hash__(self):
        """Make NewsItem hashable for deduplication."""
        return hash((self.headline, self.url))
    
    def __eq__(self, other):
        """Equality based on headline and URL."""
        if not isinstance(other, NewsItem):
            return False
        return self.headline == other.headline and self.url == other.url