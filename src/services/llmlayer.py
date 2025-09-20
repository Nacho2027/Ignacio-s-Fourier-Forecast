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
class LLMLayerSearchConfig:
    """Comprehensive LLMLayer search configuration per section"""
    section: str
    model: str
    query_template: str
    max_results: int
    temperature: float
    max_tokens: int
    search_type: str  # "news" or "general"
    search_context: str  # "low", "medium", "high"
    max_queries: int  # Number of search queries to generate
    recency: Optional[str] = None  # "hour", "day", "week", "month", "year"
    domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    date_filter: Optional[str] = None
    return_sources: bool = True
    citations: bool = True
    location: str = "us"
    response_language: str = "en"
    answer_type: str = "markdown"

@dataclass
class DomainSearchConfig:
    """Legacy configuration - keeping for backward compatibility"""
    section: str
    query_template: str
    domains: Optional[List[str]]
    max_results: int
    recency: Optional[str] = None  # "day", "week", "month", etc.
    search_type: str = "general"  # "news" or "general"


class LLMLayerServiceError(Exception):
    """Custom exception for LLMLayer service failures"""
    pass


@dataclass
class QueuedRequest:
    """A queued LLMLayer request with its parameters and result future"""
    section: str
    custom_query: Optional[str]
    future: asyncio.Future
    created_at: datetime


class RateLimitedQueue:
    """
    Queue that processes LLMLayer requests at exactly the rate limit.
    50 req/min = 1 request every 1.2 seconds.
    """

    def __init__(self, llmlayer_service):
        self.llmlayer_service = llmlayer_service
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self.rate_limit_interval = 1.2  # seconds between requests (50 req/min)
        self.logger = logging.getLogger(__name__)

    async def enqueue_request(self, section: str, custom_query: Optional[str] = None) -> LLMLayerResult:
        """
        Enqueue a request and return a future that will contain the result.
        """
        future = asyncio.Future()
        request = QueuedRequest(
            section=section,
            custom_query=custom_query,
            future=future,
            created_at=datetime.now()
        )

        await self.queue.put(request)
        self.logger.info(f"Queued LLMLayer request for section: {section}")

        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_queue())

        return await future

    async def _process_queue(self):
        """
        Process queued requests at exactly the rate limit interval.
        """
        if self.processing:
            return

        self.processing = True
        self.logger.info("Starting rate-limited queue processing...")

        last_request_time = 0

        try:
            while True:
                try:
                    # Get next request (wait up to 5 seconds for one)
                    request = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # No more requests, stop processing
                    break

                # Calculate how long to wait before making this request
                current_time = asyncio.get_event_loop().time()
                time_since_last = current_time - last_request_time

                if time_since_last < self.rate_limit_interval:
                    wait_time = self.rate_limit_interval - time_since_last
                    self.logger.info(f"Rate limiting: waiting {wait_time:.1f}s before next request")
                    await asyncio.sleep(wait_time)

                # Make the request
                try:
                    self.logger.info(f"Processing queued request for section: {request.section}")
                    result = await self.llmlayer_service.search_optimized(
                        request.section,
                        request.custom_query
                    )
                    request.future.set_result(result)
                    self.logger.info(f"Completed request for section: {request.section}")

                except Exception as e:
                    self.logger.error(f"Request failed for section {request.section}: {e}")
                    request.future.set_exception(e)

                finally:
                    last_request_time = asyncio.get_event_loop().time()
                    self.queue.task_done()

        finally:
            self.processing = False
            self.logger.info("Rate-limited queue processing completed")


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
        
        # Use cost-effective model with good performance
        # openai/gpt-4o-mini: $0.15 input, $0.60 output - good balance of cost and quality
        self.model = os.getenv("LLMLAYER_MODEL", "openai/gpt-5-mini")
        
        # Configuration
        self.timeout = 300  # 5 minutes to accommodate rate-limited cascading calls (50 req/min limit)
        self.max_retries = 3
        
        # Simple in-memory cache for the session
        self._cache: Dict[str, LLMLayerResult] = {}
        
        # Optimized search configurations for each section
        self.search_configs = self._init_search_configs()

        # New comprehensive configurations with optimal parameters
        self.llm_configs = self._init_llm_configs()

        # Rate-limited queue for managing API calls
        self.rate_limited_queue = RateLimitedQueue(self)

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

    def _init_llm_configs(self) -> Dict[str, LLMLayerSearchConfig]:
        """Initialize optimized LLMLayer configurations for each content section"""
        today = datetime.now().strftime("%B %d, %Y")

        return {
            "breaking_news": LLMLayerSearchConfig(
                section="breaking_news",
                model="openai/gpt-5-mini",  # Better citations for factual news
                query_template=f"breaking news {today} major events urgent developments",
                max_results=15,  # Increased for better filtering
                temperature=0.1,  # Low for factual accuracy
                max_tokens=2000,  # Optimized for news summaries
                search_type="news",
                search_context="high",  # Maximum comprehensive results
                max_queries=1,  # Reduced for better rate compliance
                recency="day",
                exclude_domains=[
                    "twitter.com", "x.com", "facebook.com", "reddit.com",
                    "buzzfeed.com", "businessinsider.com", "forbes.com/sites"
                ]
            ),

            "business": LLMLayerSearchConfig(
                section="business",
                model="openai/gpt-5-mini",  # Better citations for business analysis
                query_template=f"business news {today} markets economy earnings",
                max_results=12,
                temperature=0.2,  # Slightly higher for analysis
                max_tokens=3000,  # Optimized tokens
                search_type="news",
                search_context="high",
                max_queries=1,  # Rate limit optimization
                recency="day",
                domains=["wsj.com", "ft.com", "bloomberg.com", "axios.com", "cnbc.com"]
            ),

            "tech_science": LLMLayerSearchConfig(
                section="tech_science",
                model="openai/gpt-5-mini",  # Better citations for technical content
                query_template=f"technology science news AI discoveries breakthroughs",
                max_results=12,
                temperature=0.3,  # Balance factual + insight
                max_tokens=3500,  # Optimized for technical content
                search_type="general",  # Broader for scientific content
                search_context="high",
                max_queries=1,  # Reduced for rate limiting
                recency="week",  # Allow slightly older tech content
                domains=[
                    "technologyreview.com", "arstechnica.com", "wired.com",
                    "nature.com", "science.org", "quantamagazine.org"
                ]
            ),

            "politics": LLMLayerSearchConfig(
                section="politics",
                model="openai/gpt-5-mini",  # Better citations for political news
                query_template=f"US politics government {today} Congress legislation policy",
                max_results=8,
                temperature=0.1,  # Low for factual political coverage
                max_tokens=2500,
                search_type="news",
                search_context="medium",
                max_queries=1,
                recency="day",
                domains=["apnews.com", "reuters.com", "npr.org", "politico.com"]
            ),

            "miscellaneous": LLMLayerSearchConfig(
                section="miscellaneous",
                model="openai/gpt-5-mini",  # Better citations for intellectual content
                query_template=f"philosophy psychology culture arts literature essays intellectual",
                max_results=20,  # Higher for filtering diverse content
                temperature=0.4,  # Higher for creative/intellectual content
                max_tokens=4500,  # Optimized for essay content
                search_type="general",
                search_context="high",
                max_queries=1,  # Reduced for rate limiting
                recency="week",  # Allow older intellectual content
                domains=[
                    "theatlantic.com", "newyorker.com", "aeon.co", "nautilus.us",
                    "harpers.org", "parisreview.org", "lrb.co.uk", "nybooks.com"
                ]
            ),

            "startup": LLMLayerSearchConfig(
                section="startup",
                model="openai/gpt-5-mini",  # Better citations for startup insights
                query_template=f"startup venture capital entrepreneurship founder insights",
                max_results=10,
                temperature=0.3,
                max_tokens=3500,
                search_type="general",
                search_context="medium",
                max_queries=1,  # Rate limiting optimization
                recency="week",
                domains=[
                    "ycombinator.com", "firstround.com", "a16z.com",
                    "techcrunch.com/startups", "stratechery.com"
                ]
            )
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
        domains: Optional[List[str]] = None,  # DEPRECATED - will be ignored
        recency: Optional[str] = None,
        search_type: str = "general",
        exclude_domains: Optional[List[str]] = None,
        date_context: Optional[str] = None,  # Add date_context parameter
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
            query, max_results=max_results, domains=tuple(domains or []), recency=recency,
            date_context=date_context  # Include date_context in cache key
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
                    "search_context": "high",  # More comprehensive results
                    "max_queries": 2,  # Generate 2 internal search queries for better coverage
                }
                
                # Build domain filter with ONLY exclusions (no inclusions)
                # Including domains restricts search to ONLY those domains, which is too limiting
                domain_filter = []
                
                # Default exclusions for all searches - low quality and social media
                default_exclusions = [
                    "-twitter.com", "-x.com", "-facebook.com", "-instagram.com", 
                    "-tiktok.com", "-reddit.com", "-pinterest.com", "-linkedin.com",
                    "-medium.com", "-buzzfeed.com", "-businessinsider.com",
                    "-news.google.com", "-news.ycombinator.com", "-forbes.com/sites",
                    "-patch.com", "-yahoo.com", "-msn.com",
                    # Add low-quality international sources
                    "-tribune.com.pk", "-timesofindia.indiatimes.com", 
                    "-hindustantimes.com", "-indianexpress.com",
                    "-dailymail.co.uk", "-thesun.co.uk", "-mirror.co.uk",
                    "-rt.com", "-sputniknews.com", "-chinadaily.com.cn"
                ]
                domain_filter.extend(default_exclusions)
                
                # Add any additional exclusions
                if exclude_domains:
                    for domain in exclude_domains:
                        if not domain.startswith("-"):
                            domain = f"-{domain}"
                        domain_filter.append(domain)

                # CRITICAL FIX: Only include domains for TARGETED searches (news sections)
                # For broad intellectual searches (no domains specified), let it search widely
                # This prevents overly restrictive filtering that was causing 0 results
                if domains and search_type == "news":
                    # Only add domain inclusions for news searches that need specific sources
                    for domain in domains:
                        # Clean up the domain
                        domain = domain.strip().lower()
                        # Skip if already an exclusion
                        if domain.startswith("-"):
                            continue
                        # Add as inclusion for targeted news searches
                        domain_filter.append(domain)
                    self.logger.info(f"Using targeted domain search for news: {[d for d in domain_filter if not d.startswith('-')]}")

                if domain_filter:
                    search_params["domain_filter"] = domain_filter
                
                # Enhanced system prompt for better article discovery with source preferences
                search_params["system_prompt"] = (
                    "You are an expert content discovery specialist for a premium daily newsletter. "
                    "Your mission: Find the most compelling, substantive articles from authoritative sources.\n\n"
                    "DISCOVERY PRIORITIES:\n"
                    "1) SUBSTANCE over headlines - Look for in-depth reporting, original analysis, breaking developments\n"
                    "2) AUTHORITY over popularity - Prioritize established publications, expert authors, primary sources\n"
                    "3) NOVELTY over repetition - Fresh insights, exclusive reporting, unique angles on stories\n"
                    "4) DIVERSITY over echo chambers - Multiple perspectives, varied sources, cross-domain insights\n"
                    "5) CLARITY over complexity - Well-written, accessible content that informs and engages\n\n"
                    "SEARCH STRATEGY:\n"
                    "- Cast a wide net initially, then filter by quality\n"
                    "- Include breaking developments AND evergreen insights\n"
                    "- Look beyond obvious headlines for hidden gems\n"
                    "- Prioritize articles that advance understanding\n"
                    "- Return complete article URLs, not homepages or aggregators\n\n"
                    "AVOID: Social media posts, obvious clickbait, press release regurgitation, "
                    "content farms, partisan opinion without substance, outdated information."
                )
                
                # Optimize search quality parameters for rate limiting
                search_params["search_context"] = "high"  # Correct parameter name for comprehensive context
                search_params["max_queries"] = 1  # Single query for optimal rate limit compliance
            
                # Add recency filter if specified
                if recency:
                    # LLMLayer accepts: "anytime", "hour", "day", "week", "month", "year"
                    search_params["date_filter"] = recency if recency in ["hour", "day", "week", "month", "year"] else "anytime"

                # Add date_context to help LLMLayer understand the current date
                if date_context:
                    # Inject date context into the query for better temporal understanding
                    search_params["query"] = f"{query}. Context: {date_context}"
                
                # Make the async search call with timeout
                response = await asyncio.wait_for(
                    self.client.asearch(**search_params),
                    timeout=self.timeout
                )
                
                # Extract citations from sources
                citations = []
                if hasattr(response, 'sources') and response.sources:
                    self.logger.debug(f"Processing {len(response.sources)} sources from LLMLayer")
                    for source in response.sources[:max_results]:  # Limit to requested results
                        citation = self._parse_citation(source)
                        if citation.url:  # Only include sources with valid URLs
                            citations.append(citation)
                    self.logger.info(f"Successfully extracted {len(citations)} citations")
                
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

    async def search_optimized(self, section: str, custom_query: Optional[str] = None) -> LLMLayerResult:
        """
        Optimized search using new comprehensive configuration with intelligent chaining.

        Args:
            section: Section name (breaking_news, business, tech_science, politics, miscellaneous, startup)
            custom_query: Optional custom query to override template

        Returns:
            LLMLayerResult with enhanced article discovery
        """
        if section not in self.llm_configs:
            raise ValueError(f"Section '{section}' not configured. Available: {list(self.llm_configs.keys())}")

        config = self.llm_configs[section]

        # Use custom query or template
        query = custom_query or config.query_template

        # Generate cache key
        cache_key = self._generate_cache_key(
            query, section=section, config_hash=hash(str(config))
        )

        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            cached_result.cached = True
            return cached_result

        start = asyncio.get_event_loop().time()

        # Retry logic with exponential backoff
        last_error = None
        base_delay = 2.0

        for attempt in range(self.max_retries):
            try:
                # Build optimized search parameters
                search_params = {
                    "query": query,
                    "model": config.model,
                    "return_sources": config.return_sources,
                    "citations": config.citations,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "search_type": config.search_type,
                    "search_context": config.search_context,  # Fixed parameter name
                    "max_queries": config.max_queries,
                    "location": config.location,
                    "response_language": config.response_language,
                    "answer_type": config.answer_type,
                }

                # Add optional parameters
                if config.recency:
                    search_params["date_filter"] = config.recency

                # Build domain filter with both inclusions and exclusions
                domain_filter = []

                # Default exclusions for all searches
                default_exclusions = [
                    "-twitter.com", "-x.com", "-facebook.com", "-instagram.com",
                    "-tiktok.com", "-reddit.com", "-pinterest.com", "-linkedin.com",
                    "-medium.com", "-buzzfeed.com", "-businessinsider.com",
                    "-news.google.com", "-news.ycombinator.com", "-forbes.com/sites",
                    "-patch.com", "-yahoo.com", "-msn.com", "-dailymail.co.uk"
                ]
                domain_filter.extend(default_exclusions)

                # Add configured exclusions
                if config.exclude_domains:
                    for domain in config.exclude_domains:
                        if not domain.startswith("-"):
                            domain = f"-{domain}"
                        domain_filter.append(domain)

                # Add domain inclusions if specified and appropriate
                if config.domains and config.search_type == "news":
                    # Only add inclusions for news searches to avoid over-restriction
                    for domain in config.domains:
                        domain = domain.strip().lower()
                        if not domain.startswith("-"):
                            domain_filter.append(domain)
                    self.logger.info(f"Using targeted domain search for {section}: {[d for d in domain_filter if not d.startswith('-')]}")
                elif config.domains and section == "miscellaneous":
                    # For intellectual content, use domains as suggestions in system prompt
                    preferred_sources = ", ".join(config.domains)
                    search_params["system_prompt"] = (
                        f"You are a curator for intellectual content. Prioritize articles from: {preferred_sources}. "
                        f"Focus on essays, analysis, and thought-provoking content. "
                        f"Return specific, full-article URLs. Prefer depth and intellectual rigor over breaking news."
                    )

                if domain_filter:
                    search_params["domain_filter"] = domain_filter

                # Enhanced system prompt for better results
                if "system_prompt" not in search_params:
                    search_params["system_prompt"] = self._generate_system_prompt(section)

                # Make the async search call with timeout
                response = await asyncio.wait_for(
                    self.client.asearch(**search_params),
                    timeout=self.timeout
                )

                # Extract citations from sources
                citations = []
                if hasattr(response, 'sources') and response.sources:
                    self.logger.debug(f"Processing {len(response.sources)} sources from optimized search")
                    for source in response.sources[:config.max_results]:
                        citation = self._parse_citation(source)
                        if citation.url:  # Only include sources with valid URLs
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
                self.logger.info(f"Optimized search for {section}: {len(citations)} articles in {elapsed_ms:.1f}ms")
                return result

            except asyncio.TimeoutError as e:
                last_error = e
                self.logger.warning(f"LLMLayer optimized search timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue

            except Exception as e:
                last_error = e
                self.logger.error(f"LLMLayer optimized search failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    if "rate" in str(e).lower() or "429" in str(e):
                        delay = base_delay * (2 ** attempt) * 2
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue

        # All retries exhausted
        self.logger.error(f"LLMLayer optimized search failed after {self.max_retries} attempts")
        raise LLMLayerServiceError(f"Optimized search failed after {self.max_retries} attempts: {last_error}")

    def _generate_system_prompt(self, section: str) -> str:
        """Generate section-specific system prompts optimized for content discovery"""
        prompts = {
            "breaking_news": (
                "You are an expert breaking news discovery specialist. Your mission: Find the most significant, "
                "verifiable stories that truly matter.\n\n"
                "SEARCH FOCUS: Major events, policy announcements, institutional changes, geopolitical developments, "
                "significant business moves, natural disasters, security incidents, technological breakthroughs with immediate impact.\n\n"
                "DISCOVERY STRATEGY:\n"
                "- Prioritize stories with lasting implications over momentary sensations\n"
                "- Look for exclusive reporting and first-hand accounts\n"
                "- Include international stories that affect global stability\n"
                "- Seek stories that will still matter in 24-48 hours\n"
                "- Find primary source reporting, not aggregated summaries\n\n"
                "AUTHORITY RANKING: AP News, Reuters, BBC, NPR, PBS > CNN, NBC, ABC > Regional quality papers\n"
                "AVOID: Celebrity gossip, sports scores, weather unless catastrophic, speculative analysis."
            ),
            "business": (
                "You are a business intelligence discovery specialist. Your mission: Uncover market-moving "
                "insights and strategic business developments.\n\n"
                "SEARCH FOCUS: Earnings surprises, M&A activity, policy changes affecting markets, "
                "industry disruptions, leadership changes at major companies, economic indicators, "
                "venture capital trends, IPO developments.\n\n"
                "DISCOVERY STRATEGY:\n"
                "- Look beyond obvious earnings reports for strategic insights\n"
                "- Find stories that reveal industry trends and shifts\n"
                "- Prioritize exclusive interviews with business leaders\n"
                "- Seek analysis that connects dots between different business events\n"
                "- Include global business news that affects markets\n\n"
                "AUTHORITY RANKING: WSJ, FT, Bloomberg > Forbes (news), CNBC, Axios > Industry publications\n"
                "AVOID: Generic market updates, obvious press releases, promotional content."
            ),
            "tech_science": (
                "You are a technology and science breakthrough discovery specialist. Your mission: Find "
                "genuinely innovative developments and scientific advances.\n\n"
                "SEARCH FOCUS: Peer-reviewed research, AI/ML advances, quantum computing, biotech breakthroughs, "
                "space exploration, climate technology, cybersecurity developments, academic discoveries.\n\n"
                "DISCOVERY STRATEGY:\n"
                "- Prioritize peer-reviewed studies and research institutions\n"
                "- Look for applications of research to real-world problems\n"
                "- Find interdisciplinary connections and unexpected applications\n"
                "- Seek expert commentary on complex developments\n"
                "- Include both theoretical advances and practical implementations\n\n"
                "AUTHORITY RANKING: Nature, Science, MIT Tech Review > IEEE Spectrum, Ars Technica > Tech blogs\n"
                "AVOID: Product announcements without substance, vendor marketing, unverified claims."
            ),
            "politics": (
                "You are a governance and policy discovery specialist. Your mission: Find substantive "
                "political developments that affect institutions and citizens.\n\n"
                "SEARCH FOCUS: Legislative developments, judicial decisions, regulatory changes, "
                "institutional reforms, policy implementations, electoral integrity, governance innovations.\n\n"
                "DISCOVERY STRATEGY:\n"
                "- Focus on policy substance over political theater\n"
                "- Look for long-term implications of political decisions\n"
                "- Find bipartisan issues and institutional concerns\n"
                "- Prioritize procedural and constitutional matters\n"
                "- Seek expert analysis from governance scholars\n\n"
                "AUTHORITY RANKING: AP News, Reuters, NPR > Politico (news), Washington Post > Partisan sources\n"
                "AVOID: Campaign rhetoric, partisan talking points, speculation without basis."
            ),
            "miscellaneous": (
                "You are an intellectual content discovery specialist. Your mission: Find thought-provoking "
                "essays and cultural insights that broaden perspectives.\n\n"
                "SEARCH FOCUS: Philosophy, psychology, sociology, anthropology, literature, arts, "
                "cultural criticism, historical analysis, interdisciplinary thinking, human condition.\n\n"
                "DISCOVERY STRATEGY:\n"
                "- Prioritize original thinking over current events commentary\n"
                "- Look for essays that challenge conventional wisdom\n"
                "- Find connections between seemingly unrelated domains\n"
                "- Seek timeless insights wrapped in contemporary examples\n"
                "- Include diverse voices and global perspectives\n\n"
                "AUTHORITY RANKING: The Atlantic, The New Yorker, Aeon > Harper's, London Review of Books > Academic journals\n"
                "AVOID: Trending topics without depth, partisan culture war content, shallow lifestyle pieces."
            ),
            "startup": (
                "You are a startup ecosystem discovery specialist. Your mission: Find actionable insights "
                "and wisdom from the entrepreneurial world.\n\n"
                "SEARCH FOCUS: Founder lessons, scaling strategies, venture capital insights, "
                "product-market fit stories, business model innovations, startup failures and recoveries.\n\n"
                "DISCOVERY STRATEGY:\n"
                "- Prioritize first-hand founder experiences over third-party analysis\n"
                "- Look for actionable advice and concrete lessons\n"
                "- Find stories of both success and instructive failure\n"
                "- Seek insights from experienced investors and operators\n"
                "- Include diverse startup ecosystems beyond Silicon Valley\n\n"
                "AUTHORITY RANKING: First Round Review, a16z, YC Blog > TechCrunch (analysis), Stratechery > Generic startup blogs\n"
                "AVOID: Funding announcements without strategy insight, generic startup advice, unverified growth claims."
            )
        }

        return prompts.get(section,
            "You are an expert content discovery specialist. Find high-quality, authoritative articles "
            "that provide substantial value and insights. Focus on depth, accuracy, and unique perspectives."
        )
    
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
        """Parse various date formats returned by LLMLayer with validation."""
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

        parsed_date = None

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
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue

        except Exception:
            pass

        # Try ISO format as fallback
        if not parsed_date:
            try:
                clean_date_str = date_str.replace("Z", "+00:00")
                parsed_date = datetime.fromisoformat(clean_date_str)
            except Exception:
                pass

        # CRITICAL: Validate the parsed date for sanity
        if parsed_date:
            current_year = datetime.now().year

            # Reject dates older than 2 years (definitely not current news)
            if parsed_date.year < current_year - 2:
                self.logger.warning(f"Rejecting ancient date {parsed_date.year} from '{date_str}' - too old for news")
                return None

            # Reject dates more than 7 days in the future (likely parsing error)
            if parsed_date > datetime.now() + timedelta(days=7):
                self.logger.warning(f"Rejecting future date {parsed_date} from '{date_str}' - likely parsing error")
                return None

            return parsed_date

        return None
    
    def _parse_citation(self, source: Dict[str, Any]) -> Citation:
        """Parse a source from LLMLayer into our Citation format.

        LLMLayer provides clean, structured source data, so we can directly use it.
        """
        # LLMLayer returns clean structured data - use it directly
        url = source.get("url", "")  # LLMLayer uses "url" not "link"
        title = source.get("title", "")
        snippet = source.get("snippet", "")

        # Extract domain name for source
        if url:
            parsed_url = urlparse(url)
            source_name = parsed_url.netloc.replace('www.', '')
        else:
            source_name = ""

        # Get relevance score if available
        score = float(source.get("relevance", 0.0))

        # Parse published date if available
        published_dt = None
        date_str = source.get("date")

        if date_str:
            published_dt = self._parse_llmlayer_date(date_str)

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

    async def search_optimized_rate_limited(self, section: str, custom_query: Optional[str] = None) -> LLMLayerResult:
        """
        Perform optimized search using the rate-limited queue.
        This replaces direct calls to search_optimized for rate limit compliance.
        """
        return await self.rate_limited_queue.enqueue_request(section, custom_query)

    async def search_with_json_schema(
        self,
        query: str,
        section: str,
        max_articles: int = 10,
        use_optimized_config: bool = True
    ) -> Dict[str, Any]:
        """
        Search using JSON schema for structured article extraction.

        Args:
            query: Search query
            section: Section name for configuration
            max_articles: Maximum number of articles to return
            use_optimized_config: Whether to use section-specific optimized configuration

        Returns:
            Structured JSON response with articles and section summary
        """
        import json

        # Define the article extraction schema
        article_schema = {
            "type": "object",
            "properties": {
                "articles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "summary": {"type": "string"},
                            "publication": {"type": "string"},
                            "published_date": {"type": "string"},
                            "relevance_score": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["title", "url", "summary", "publication"]
                    },
                    "maxItems": max_articles
                },
                "section_summary": {
                    "type": "string",
                    "description": "Brief summary of the main themes across all articles"
                }
            },
            "required": ["articles", "section_summary"]
        }

        # Get configuration for the section
        if use_optimized_config and section in self.llm_configs:
            config = self.llm_configs[section]
            model = config.model
            temperature = config.temperature
            max_tokens = config.max_tokens
            search_type = config.search_type
            search_context = config.search_context
        else:
            # Use defaults
            model = self.model
            temperature = 0.3
            max_tokens = 4000
            search_type = "general"
            search_context = "high"

        # Prepare search parameters for JSON response
        search_params = {
            "query": query,
            "model": model,
            "return_sources": True,
            "citations": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "search_type": search_type,
            "search_context": search_context,
            "location": "us",
            "response_language": "en",
            "answer_type": "json",
            "json_schema": json.dumps(article_schema),
            "max_queries": 2
        }

        # Add system prompt for better structured extraction
        search_params["system_prompt"] = f"""
        You are an expert content curator for a premium newsletter. Extract {max_articles} high-quality articles
        for the {section} section.

        REQUIREMENTS:
        1. Return ONLY articles with complete, valid URLs (not homepages)
        2. Ensure each article has a clear, informative summary
        3. Include accurate publication names and dates when available
        4. Assign relevance scores based on content quality and recency
        5. Provide a section summary that captures key themes

        Focus on authoritative, well-written content from reputable sources.
        """

        try:
            # Make the API call
            response = await asyncio.wait_for(
                self.client.asearch(**search_params),
                timeout=self.timeout
            )

            # Parse the JSON response
            if hasattr(response, 'llm_response'):
                try:
                    structured_data = json.loads(response.llm_response)
                    self.logger.info(f"Successfully extracted {len(structured_data.get('articles', []))} articles via JSON schema")
                    return structured_data
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response: {e}")
                    # Fallback to regular search
                    return await self._fallback_to_regular_search(query, section, max_articles)
            else:
                self.logger.error("No llm_response found in API response")
                return await self._fallback_to_regular_search(query, section, max_articles)

        except Exception as e:
            self.logger.error(f"JSON schema search failed: {e}")
            return await self._fallback_to_regular_search(query, section, max_articles)

    async def _fallback_to_regular_search(self, query: str, section: str, max_articles: int) -> Dict[str, Any]:
        """Fallback to regular search if JSON schema fails."""
        self.logger.info(f"Falling back to regular search for {section}")

        if section in self.llm_configs:
            result = await self.search_optimized(section, query)
        else:
            result = await self.search(query, max_results=max_articles)

        # Convert to structured format
        articles = []
        for citation in result.citations[:max_articles]:
            articles.append({
                "title": citation.title,
                "url": citation.url,
                "summary": citation.snippet,
                "publication": citation.source_name,
                "published_date": citation.published_date.isoformat() if citation.published_date else "",
                "relevance_score": citation.relevance_score
            })

        return {
            "articles": articles,
            "section_summary": result.answer[:200] + "..." if len(result.answer) > 200 else result.answer
        }