"""
LLMLayer service for enhanced web search with citations.
Provides access to premium news sources with proper citations.
"""

import os
import asyncio
import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
from urllib.parse import urlparse

from llmlayer import LLMLayerClient


@dataclass
class Citation:
    """Structured citation from LLMLayer"""
    url: str
    title: str
    snippet: str
    source_name: str
    relevance_score: float
    published_date: Optional[datetime] = None


@dataclass
class LLMLayerResult:
    """Complete result from LLMLayer search"""
    query: str
    answer: str
    citations: List[Citation]
    search_time_ms: float
    total_results: int
    cached: bool = False


@dataclass
class DomainSearchConfig:
    """Configuration for domain-specific searches"""
    section: str
    query_template: str
    domains: Optional[List[str]]
    max_results: int
    recency: Optional[str] = None  # "day", "week", "month", etc.
    search_type: str = "general"  # "news" or "general"


class LLMLayerServiceError(Exception):
    """Custom exception for LLMLayer service failures"""
    pass


class LLMLayerService:
    """
    LLMLayer integration for Fourier Forecast newsletter.
    
    Uses Claude-4-Sonnet with real web search capabilities.
    Provides access to premium news sources with proper citations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLMLAYER_API_KEY")
        if not self.api_key:
            raise ValueError("LLMLayer API key required. Set LLMLAYER_API_KEY or pass api_key parameter.")
        
        # Initialize LLMLayer client
        self.client = LLMLayerClient(api_key=self.api_key)
        
        # Use Groq OpenAI GPT OSS 20B - cost-effective model for LLM layer operations
        self.model = "groq/openai-gpt-oss-20b"
        
        # Configuration
        self.timeout = 120  # Increased timeout for LLM API calls to prevent failures
        self.max_retries = 3
        
        # Simple in-memory cache for the session
        self._cache: Dict[str, LLMLayerResult] = {}
        
        # Domain search configurations for each section
        self.search_configs = self._init_search_configs()
        
        self.logger = logging.getLogger(__name__)
    
    def _init_search_configs(self) -> Dict[str, DomainSearchConfig]:
        return {
            "breaking_news": DomainSearchConfig(
                section="breaking_news",
                query_template="latest breaking news {query} today headlines urgent important",
                # VISION.txt: AP News and Reuters ONLY for breaking news
                domains=["apnews.com", "reuters.com"],
                max_results=10,
                recency="day",
            ),
            "tech_science": DomainSearchConfig(
                section="tech_science",
                query_template="latest {query} technology science AI breakthrough innovation research",
                # VISION.txt: MIT Tech Review, IEEE Spectrum, Quanta Magazine, TLDR
                domains=["technologyreview.com", "spectrum.ieee.org", "quantamagazine.org"],
                max_results=8,
                recency="week",
            ),
            "business_economy": DomainSearchConfig(
                section="business_economy",
                query_template="{query} Wall Street business economy stock market finance deals earnings",
                # VISION.txt: WSJ and Axios (user has subscriptions)
                domains=["wsj.com", "axios.com"],
                max_results=8,
                recency="day",
            ),
            "startup": DomainSearchConfig(
                section="startup",
                query_template="{query} startup founders advice insights YC First Round",
                # VISION.txt: YC Blog, First Round Review
                domains=["ycombinator.com", "firstround.com", "review.firstround.com"],
                max_results=6,
                recency="week",
            ),
            "politics": DomainSearchConfig(
                section="politics",
                query_template="{query} US politics government policy Congress White House",
                # VISION.txt: Associated Press ONLY for politics
                domains=["apnews.com"],
                max_results=5,
                recency="day",
            ),
            "local": DomainSearchConfig(
                section="local",
                query_template="{query} Miami news Cornell University Ithaca",
                # VISION.txt: Miami Herald and Cornell news
                domains=["miamiherald.com", "news.cornell.edu", "cornellsun.com"],
                max_results=4,
                recency="week",
            ),
            "spiritual": DomainSearchConfig(
                section="spiritual",
                query_template="daily reflection Gospel meditation {query}",
                # VISION.txt: Catholic Daily Reflections
                domains=["catholic-daily-reflections.com", "catholic.org"],
                max_results=5,
                recency="day",
            ),
        }
    
    async def test_connection(self) -> bool:
        """Test that the API connection works."""
        try:
            # Simple test query
            response = await self.asearch("test", max_results=1)
            return True
        except Exception as e:
            self.logger.error(f"LLMLayer connection test failed: {e}")
            return False
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        domains: Optional[List[str]] = None,
        recency: Optional[str] = None,
        search_type: str = "general",
        exclude_domains: Optional[List[str]] = None,
    ) -> LLMLayerResult:
        """
        Perform a web search using LLMLayer with Claude-4-Sonnet.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            domains: List of domains to include/exclude (prefix with - to exclude)
            recency: Time filter - "day", "week", "month", "year"
            search_type: Type of search - "general" or "news"
        
        Returns:
            LLMLayerResult with answer and citations
        """
        cache_key = self._generate_cache_key(
            query, max_results=max_results, domains=tuple(domains or []), recency=recency
        )
        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            cached_result.cached = True
            return cached_result
        
        start = asyncio.get_event_loop().time()
        
        # Retry logic with exponential backoff
        last_error = None
        base_delay = 2.0  # Start with 2 second delay
        
        for attempt in range(self.max_retries):
            try:
                # Prepare LLMLayer search parameters matching API docs
                search_params = {
                    "query": query,
                    "model": self.model,
                    "return_sources": True,  # Get source URLs for citations
                    "citations": True,  # Add inline citations [1] in response
                    "max_tokens": 5000,  # Increased for more comprehensive summaries
                    "temperature": 0.3,  # Lower for more factual responses
                    "search_type": search_type,  # "news" for news articles, "general" for other content
                    "location": "us",
                    "response_language": "en",
                    "answer_type": "markdown",
                    "search_context_size": "high",  # More comprehensive results
                    "max_queries": 2,  # Generate 2 internal search queries for better coverage
                }
                
                # Build domain filter with both inclusions and exclusions
                domain_filter = []
                
                # Add included domains
                if domains:
                    domain_filter.extend(domains)
                
                # Add excluded domains with "-" prefix
                # Default exclusions for all searches
                default_exclusions = [
                    "-twitter.com", "-x.com", "-facebook.com", "-instagram.com", 
                    "-tiktok.com", "-reddit.com", "-pinterest.com", "-linkedin.com",
                    "-medium.com", "-buzzfeed.com", "-businessinsider.com",
                    "-news.google.com", "-news.ycombinator.com", "-forbes.com/sites"
                ]
                domain_filter.extend(default_exclusions)
                
                # Add any additional exclusions
                if exclude_domains:
                    for domain in exclude_domains:
                        if not domain.startswith("-"):
                            domain = f"-{domain}"
                        domain_filter.append(domain)
                
                if domain_filter:
                    search_params["domain_filter"] = domain_filter
                
                # Enhanced system prompt for better article discovery
                search_params["system_prompt"] = (
                    "You are a sophisticated content curator seeking intellectually stimulating articles. "
                    "Focus on finding specific, full-length article URLs (not homepages or category pages). "
                    "Prioritize content that meets these criteria:\n"
                    "1. DEPTH: In-depth, analytical pieces with substantive insights (1000+ words preferred)\n"
                    "2. AUTHORITY: From respected publications, academic sources, or domain experts\n"
                    "3. ORIGINALITY: Novel perspectives, breakthrough findings, or unique analyses\n"
                    "4. DIVERSITY: Ensure variety across different topics and perspectives within the query\n"
                    "5. RECENCY: Recent content but not breaking news unless exceptional\n"
                    "6. INTELLECTUAL MERIT: Content that challenges thinking or offers profound insights\n"
                    "Avoid: listicles, clickbait, press releases, promotional content, or superficial coverage. "
                    "Look for long-form journalism, research summaries, expert analyses, and thoughtful essays."
                )
                
                # Increase search quality parameters
                search_params["search_context_size"] = "high"  # Already set but keep for clarity
                search_params["max_queries"] = 3  # Increase from 2 to 3 for better coverage
            
                # Add recency filter if specified
                if recency:
                    # LLMLayer accepts: "anytime", "hour", "day", "week", "month", "year"
                    search_params["date_filter"] = recency if recency in ["hour", "day", "week", "month", "year"] else "anytime"
                
                # Make the async search call with timeout
                response = await asyncio.wait_for(
                    self.client.asearch(**search_params),
                    timeout=self.timeout
                )
                
                # LOG: Debug the raw API response
                self.logger.info(f"LLMLayer API Response Debug:")
                self.logger.info(f"  Query: {query}")
                self.logger.info(f"  Search params: {search_params}")
                self.logger.info(f"  Response type: {type(response)}")
                self.logger.info(f"  Has sources attr: {hasattr(response, 'sources')}")
                if hasattr(response, 'sources'):
                    self.logger.info(f"  Sources count: {len(response.sources) if response.sources else 0}")
                    if response.sources:
                        self.logger.info(f"  First 3 raw sources:")
                        for i, source in enumerate(response.sources[:3]):
                            self.logger.info(f"    Source {i+1}: {source}")
                
                # Extract citations from sources
                citations = []
                if hasattr(response, 'sources') and response.sources:
                    for idx, source in enumerate(response.sources[:max_results]):  # Limit before fetching
                        self.logger.info(f"Processing source {idx+1}: {source}")
                        citation = self._parse_citation(source)
                        # Log what we extracted
                        self.logger.info(f"  Extracted citation: url={citation.url}, published_date={citation.published_date}, source_name={citation.source_name}")
                        # LLMLayer already provides good snippets, no need to fetch full content
                        citations.append(citation)
                
                elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000.0
                
                result = LLMLayerResult(
                    query=query,
                    answer=str(response.llm_response) if hasattr(response, 'llm_response') else "",
                    citations=citations,
                    search_time_ms=elapsed_ms,
                    total_results=len(citations),
                    cached=False,
                )
                
                self._cache[cache_key] = result
                return result
                
            except asyncio.TimeoutError as e:
                last_error = e
                self.logger.warning(f"LLMLayer search timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Retrying after {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    continue
                    
            except Exception as e:
                last_error = e
                self.logger.error(f"LLMLayer search failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    # Check if it's a rate limit error
                    if "rate" in str(e).lower() or "429" in str(e):
                        delay = base_delay * (2 ** attempt) * 2  # Longer delay for rate limits
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Retrying after {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    continue
        
        # All retries exhausted
        self.logger.error(f"LLMLayer search failed after {self.max_retries} attempts")
        raise LLMLayerServiceError(f"Search failed after {self.max_retries} attempts: {last_error}")
    
    async def asearch(self, *args, **kwargs):
        """Async wrapper for compatibility."""
        return await self.search(*args, **kwargs)
    
    async def search_by_section(self, section: str, base_query: Optional[str] = None) -> LLMLayerResult:
        """Search using predefined section configuration."""
        if section not in self.search_configs:
            raise ValueError(f"Section '{section}' is not configured")
        
        config = self.search_configs[section]
        query = config.query_template.format(query=base_query or "")
        
        return await self.search(
            query=query,
            max_results=config.max_results,
            domains=config.domains,
            recency=config.recency,
            search_type=config.search_type,
            exclude_domains=None,  # Use default exclusions
        )
    
    async def discover_citations_for_topic(self, topic: str, cross_domain: bool = True) -> List[Citation]:
        """Discover citations for a specific topic."""
        domains = None if cross_domain else self.search_configs["tech_science"].domains
        result = await self.search(topic, max_results=10, domains=domains)
        return result.citations
    
    async def batch_search(self, queries: List[Dict[str, Any]]) -> List[LLMLayerResult]:
        """Batch search multiple queries."""
        tasks = [
            self.search(
                q.get("query", ""),
                max_results=q.get("max_results", 10),
                domains=q.get("domains"),
                recency=q.get("recency"),
            )
            for q in queries
        ]
        results = await asyncio.gather(*tasks)
        return results
    
    async def get_trending_topics(self) -> List[str]:
        """Get trending topics across various domains."""
        result = await self.search(
            "trending topics today across technology science business arts",
            max_results=10,
            recency="day"
        )
        
        # Extract topics from the answer
        topics = []
        if result.answer:
            lines = result.answer.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Clean up bullet points and numbers
                    topic = line.lstrip('•-*0123456789. ')
                    if topic:
                        topics.append(topic)
        
        return topics[:10]
    
    def _generate_cache_key(self, query: str, **kwargs) -> str:
        """Generate a cache key for the query."""
        key_parts = [query.lower().strip()]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}:{v}")
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()
    
    def _parse_llmlayer_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats returned by LLMLayer."""
        if not date_str:
            return None
            
        # Handle relative dates like "4 hours ago", "3 days ago" and Spanish versions
        import re
        from datetime import timedelta
        
        # English relative dates
        relative_match = re.match(r'(\d+)\s+(hour|hours|day|days|week|weeks|month|months|minute|minutes)\s+ago', date_str.lower())
        if relative_match:
            number = int(relative_match.group(1))
            unit = relative_match.group(2)
            
            now = datetime.now()
            
            if 'minute' in unit:
                return now - timedelta(minutes=number)
            elif 'hour' in unit:
                return now - timedelta(hours=number)
            elif 'day' in unit:
                return now - timedelta(days=number)
            elif 'week' in unit:
                return now - timedelta(weeks=number)
            elif 'month' in unit:
                return now - timedelta(days=number * 30)  # Approximate
        
        # Log unexpected language in dates for debugging
        if 'hace' in date_str.lower():
            self.logger.warning(f"LLMLayer returned Spanish date despite English config: '{date_str}'")
                
        # Handle common date formats like "Jul 29, 2025", "May 29, 2025"
        try:
            # Try various common formats
            formats = [
                "%b %d, %Y",      # "Jul 29, 2025"
                "%B %d, %Y",      # "July 29, 2025"
                "%Y-%m-%d",       # "2025-07-29"
                "%m/%d/%Y",       # "07/29/2025"
                "%d/%m/%Y",       # "29/07/2025"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
                    
        except Exception:
            pass
            
        # Try ISO format as fallback
        try:
            clean_date_str = date_str.replace("Z", "+00:00")
            return datetime.fromisoformat(clean_date_str)
        except Exception:
            pass
            
        return None
    
    def _parse_citation(self, source: Dict[str, Any]) -> Citation:
        """Parse a source from LLMLayer into our Citation format."""
        # LLMLayer returns 'url' not 'link' according to API docs
        url = source.get("url", "") or source.get("link", "")
        title = source.get("title", "") or source.get("name", "")
        snippet = source.get("snippet", "") or source.get("description", "")
        
        # Log URL validation
        if url:
            self.logger.debug(f"Citation URL: {url}")
            # Check if URL looks like just a domain
            if not any(path_part in url for path_part in ['/article/', '/story/', '/news/', '.html', '.htm', '?']) and url.count('/') <= 3:
                self.logger.warning(f"URL might be just a domain, not a full article: {url}")
        
        # Extract domain name for source
        if url:
            parsed_url = urlparse(url)
            source_name = parsed_url.netloc.replace('www.', '')
        else:
            source_name = source.get("source", "") or source.get("domain", "")
        
        # Parse relevance score if available
        score = float(source.get("relevance", 0.0) or source.get("score", 0.0))
        
        # Parse published date if available
        published_dt = None
        
        # Log all available fields in source for debugging
        self.logger.debug(f"Source fields available: {list(source.keys()) if isinstance(source, dict) else 'Not a dict'}")
        
        # Try multiple possible date field names
        date_fields = ["date", "published_date", "publishedDate", "published", "pubDate", "publication_date", "created_at", "timestamp"]
        date_str = None
        found_field = None
        
        for field in date_fields:
            if source.get(field):
                date_str = source.get(field)
                found_field = field
                break
        
        self.logger.debug(f"Date extraction: found_field='{found_field}', date_str='{date_str}'")
        
        if date_str:
            try:
                # Handle different date formats returned by LLMLayer
                if isinstance(date_str, str):
                    published_dt = self._parse_llmlayer_date(date_str)
                    if published_dt:
                        self.logger.debug(f"Successfully parsed date: {published_dt}")
                    else:
                        self.logger.warning(f"Could not parse date format: '{date_str}'")
                else:
                    self.logger.debug(f"Date string is not a string: {type(date_str)} = {date_str}")
            except Exception as e:
                self.logger.warning(f"Failed to parse date '{date_str}' from field '{found_field}': {e}")
                published_dt = None
        else:
            self.logger.debug("No date field found in source")
        
        return Citation(
            url=url,
            title=title,
            snippet=snippet,
            source_name=source_name,
            relevance_score=score,
            published_date=published_dt,
        )
    
    async def fetch_url_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch full content from a URL using HTTP request and BeautifulSoup parsing.
        """
        import aiohttp
        from bs4 import BeautifulSoup
        import ssl
        import certifi
        
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            # Use a short timeout for fetching article content since it's optional enhancement
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        self.logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
                        
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove scripts and styles
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Try to find main content areas
                    content = ""
                    
                    # Look for article body in common containers
                    article_selectors = [
                        'article', 
                        '[role="main"]',
                        '.article-body',
                        '.story-body',
                        '.entry-content',
                        '.post-content',
                        'main',
                        '.content'
                    ]
                    
                    for selector in article_selectors:
                        element = soup.select_one(selector)
                        if element:
                            content = element.get_text(separator=' ', strip=True)
                            break
                    
                    # Fallback to body if no article found
                    if not content:
                        body = soup.find('body')
                        if body:
                            content = body.get_text(separator=' ', strip=True)
                    
                    # Get title
                    title = ""
                    title_elem = soup.find('title')
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                    
                    # Limit content length to 5000 chars to avoid token limits
                    if len(content) > 5000:
                        content = content[:5000] + "..."
                    
                    return {
                        "title": title,
                        "content": content,
                        "metadata": {"url": url}
                    }
                    
        except Exception as e:
            self.logger.error(f"Error fetching content from {url}: {e}")
            return None