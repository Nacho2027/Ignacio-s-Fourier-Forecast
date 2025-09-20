"""
Semantic Scholar service for fetching academic papers.
Provides access to peer-reviewed papers with rich metadata and citation information.
Uses direct REST API calls instead of the SDK for better control and reliability.
"""

import asyncio
import logging
import os
import ssl
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import aiohttp
import certifi

logger = logging.getLogger(__name__)


@dataclass
class SemanticPaper:
    """Structured representation of a Semantic Scholar paper."""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    citation_count: int
    influential_citation_count: int
    url: str
    pdf_url: Optional[str]
    published_date: Optional[datetime]
    fields_of_study: List[str]
    tldr: Optional[str]  # AI-generated summary if available


class SemanticScholarService:
    """
    Service for fetching papers from Semantic Scholar API.
    
    Uses direct REST API calls for better control over rate limiting,
    error handling, and SSL certificate management.
    
    Focuses on:
    - High-impact peer-reviewed papers
    - Papers with strong citation metrics
    - Cross-disciplinary AI research
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar service.
        
        Args:
            api_key: Optional API key for higher rate limits
                    Without key: 100 requests per 5 minutes
                    With key: 1 request per second
        """
        self.api_key = api_key
        self.logger = logger
        self.enabled = True  # Service works even without API key
        
        # Set up headers with API key if available
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
            self.logger.info("Semantic Scholar service initialized with API key")
        else:
            self.logger.warning(
                "Semantic Scholar service initialized without API key. "
                "Rate limit: 100 requests per 5 minutes. "
                "Consider getting a free API key at https://www.semanticscholar.org/product/api"
            )
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        
        # Retry configuration - more conservative without API key
        self.max_retries = 3 if not self.api_key else 5  # Fewer retries without API key
        self.base_delay = 3.0 if not self.api_key else 2.0  # Longer delays without API key
        
        # Fields to retrieve for optimal performance
        self.paper_fields = [
            'paperId', 'title', 'abstract', 'authors', 'year', 'venue',
            'citationCount', 'influentialCitationCount', 'url', 'openAccessPdf',
            'publicationDate', 'fieldsOfStudy', 's2FieldsOfStudy', 'tldr'
        ]
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp ClientSession with proper SSL configuration."""
        if self.session is None or self.session.closed:
            # Create SSL context using certifi for proper certificate validation
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout,
                connector=connector
            )
        return self.session
    
    async def close_session(self):
        """Close aiohttp session for cleanup."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures session cleanup."""
        await self.close_session()
        return False
    
    def search_papers_sync(
        self,
        queries: List[str],
        max_results: int = 30,
        min_citations: int = 5,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for search_papers.
        
        Note: This creates a new event loop which may not be ideal
        in all contexts. Prefer using the async version directly.
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.search_papers(queries, max_results, min_citations, days_back)
            )
        finally:
            loop.close()
    
    async def search_papers(
        self,
        queries: List[str],
        max_results: int = 30,
        min_citations: int = 5,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for searching Semantic Scholar papers.
        
        Args:
            queries: List of search queries
            max_results: Maximum papers to return
            min_citations: Minimum citation count filter
            days_back: How many days back to search
            
        Returns:
            List of paper dictionaries formatted for the pipeline
        """
        if not self.enabled:
            self.logger.warning("Semantic Scholar service is disabled")
            return []
        
        self.logger.info(f"Searching Semantic Scholar for {len(queries)} queries, max {max_results} results")
        
        # Calculate date range for API parameter
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        # Format date range for publicationDateOrYear parameter
        if days_back > 365:
            # For long ranges, use year format
            date_range = f"{start_date.year}:{end_date.year}"
        else:
            # For shorter ranges, use precise dates
            date_range = f"{start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')}"
        
        session = await self._get_session()
        all_papers: List[Dict[str, Any]] = []
        
        # API limit is 100 results per request
        api_limit = min(max_results, 100)
        
        # Handle citation velocity mode (empty query)
        if len(queries) == 1 and queries[0] == "":
            # Citation velocity mode - get more papers and let velocity sorting work
            self.logger.info("Using citation velocity mode - fetching trending papers")
            query_limit = min(100, max_results * 2)  # Get 2x more to filter by velocity
            task = self._search_single_query_async(
                session, "", date_range, 0, query_limit  # Lower min_citations for velocity mode
            )
            tasks = [task]
        else:
            # Legacy mode - distribute results across queries for diversity
            papers_per_query = max(10, max_results // len(queries))
            
            # Create search tasks for all queries
            tasks = []
            for query in queries:
                query_limit = min(papers_per_query, api_limit)
                task = self._search_single_query_async(
                    session, query, date_range, min_citations, query_limit
                )
                tasks.append(task)
        
        # Execute searches with limited concurrency to respect rate limits
        # More conservative processing without API key
        batch_size = 1 if not self.api_key else 2  # Single queries without API key
        delay_between_batches = 2.0 if not self.api_key else 1.0  # Longer delays without API key

        results_from_tasks = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results_from_tasks.extend(batch_results)

            # Delay between batches to respect rate limits
            if i + batch_size < len(tasks):
                await asyncio.sleep(delay_between_batches)
        
        # Process all gathered results
        for query_results in results_from_tasks:
            if isinstance(query_results, Exception):
                self.logger.warning(f"Search task failed: {query_results}")
                continue
            all_papers.extend(query_results)
        
        # Deduplicate by paperId
        seen_paper_ids = set()
        unique_papers = []
        for paper in all_papers:
            paper_id = paper.get('semantic_scholar_id')
            if paper_id and paper_id not in seen_paper_ids:
                unique_papers.append(paper)
                seen_paper_ids.add(paper_id)
        
        # Sort by quality score and return top results
        unique_papers.sort(key=lambda p: self._calculate_quality_score(p), reverse=True)
        
        final_papers = unique_papers[:max_results]
        self.logger.info(f"Found {len(final_papers)} unique quality papers from Semantic Scholar")
        return final_papers
    
    async def search_papers_async(
        self,
        queries: List[str],
        max_results: int = 30,
        min_citations: int = 5,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Async version of search_papers (alias for backward compatibility).
        """
        return await self.search_papers(queries, max_results, min_citations, days_back)
    
    async def _search_single_query_async(
        self,
        session: aiohttp.ClientSession,
        query: str,
        date_range: str,
        min_citations: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Search using recency and impact, not keywords.
        Uses empty query to get ALL recent papers, then sorts by citation velocity.
        
        Args:
            session: aiohttp ClientSession to use
            query: Search query string (empty for velocity mode)
            date_range: Date range in YYYY-MM-DD:YYYY-MM-DD format
            min_citations: Minimum citation count
            limit: Maximum results to return
            
        Returns:
            List of formatted paper dictionaries sorted by citation velocity
        """
        papers: List[Dict[str, Any]] = []
        
        for attempt in range(self.max_retries):
            try:
                # Add initial delay to avoid hitting rate limits
                if attempt == 0:
                    initial_delay = 1.0 if not self.api_key else 0.5  # Longer initial delay without API key
                    await asyncio.sleep(initial_delay)
                
                # Build query parameters - use broad query for velocity mode
                if query == "" or not query:
                    # Citation velocity mode - use very broad terms to get diverse papers
                    actual_limit = str(min(limit * 3, 100))  # Get 3x more, max 100
                    self.logger.info(f"Using citation velocity mode - fetching {actual_limit} papers")
                    # Use broad academic terms for discovery
                    query = "science OR research OR study OR analysis OR investigation"
                else:
                    # Legacy mode with specific query
                    actual_limit = str(limit)
                
                params = {
                    "query": query,  # Never send empty query - API doesn't support it
                    "fields": ",".join(self.paper_fields),
                    "publicationDateOrYear": date_range,
                    "minCitationCount": str(min_citations),
                    "limit": actual_limit,
                    "fieldsOfStudy": "Computer Science,Mathematics,Engineering,Physics,Biology,Chemistry,Medicine,Economics"
                }
                
                self.logger.debug(
                    f"Attempt {attempt + 1}/{self.max_retries} for query '{query}' "
                    f"with params: {params}"
                )
                
                # Make the API request
                url = f"{self.BASE_URL}/paper/search"
                async with session.get(url, params=params) as response:
                    # Check for rate limiting
                    if response.status == 429:
                        # Handle rate limit with exponential backoff
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                delay = float(retry_after) + random.uniform(0, 1)
                            except ValueError:
                                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 2)
                        else:
                            delay = self.base_delay * (2 ** attempt) + random.uniform(0, 2)
                        
                        self.logger.warning(
                            f"Rate limit hit for query '{query}' (attempt {attempt + 1}/{self.max_retries}). "
                            f"Retrying after {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    
                    # Raise exception for other bad status codes
                    response.raise_for_status()
                    
                    # Parse JSON response
                    data = await response.json()
                    
                    # Extract papers from the data array
                    if data and 'data' in data and isinstance(data['data'], list):
                        for paper_data in data['data']:
                            formatted = self._format_paper(paper_data)
                            if formatted:
                                # Calculate citation velocity for trending detection
                                published_date = formatted.get('published_date')
                                if published_date:
                                    try:
                                        # Ensure published_date is timezone-aware
                                        if published_date.tzinfo is None:
                                            published_date = published_date.replace(tzinfo=timezone.utc)
                                        
                                        days_old = (datetime.now(timezone.utc) - published_date).days
                                        days_old = max(days_old, 1)  # Avoid division by zero
                                        
                                        citations = formatted.get('citation_count', 0)
                                        formatted['citation_velocity'] = citations / days_old
                                        formatted['days_old'] = days_old
                                    except Exception as e:
                                        self.logger.debug(f"Error calculating velocity: {e}")
                                        formatted['citation_velocity'] = 0
                                        formatted['days_old'] = 999
                                else:
                                    formatted['citation_velocity'] = 0
                                    formatted['days_old'] = 999
                                
                                # Add paper if it meets basic quality criteria
                                if self._is_quality_paper(formatted):
                                    papers.append(formatted)
                        
                        # Sort by citation velocity if using empty query
                        if query == "" or not query:
                            papers.sort(key=lambda p: p.get('citation_velocity', 0), reverse=True)
                            
                            # Log top velocities for debugging
                            if papers:
                                top_velocities = [(p.get('citation_velocity', 0), p.get('headline', 'Unknown')) for p in papers[:5]]
                                self.logger.info(f"Top citation velocities: {[(f'{v:.2f}', t[:50]) for v, t in top_velocities]}")
                            
                            # Return only top papers by velocity
                            papers = papers[:limit]
                        
                        if papers:
                            self.logger.info(
                                f"Successfully retrieved {len(papers)} papers for query '{query}'"
                            )
                        return papers
                    else:
                        self.logger.warning(
                            f"No data found in response for query '{query}'. "
                            f"Response structure: {list(data.keys()) if data else 'None'}"
                        )
                        return []
                        
            except aiohttp.ClientResponseError as e:
                if e.status == 400:
                    # Bad request - don't retry
                    self.logger.error(f"Bad request for query '{query}': {e}")
                    return []
                else:
                    # Other HTTP errors - retry with backoff
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.warning(
                        f"HTTP error {e.status} for query '{query}' "
                        f"(attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying after {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                    
            except asyncio.TimeoutError:
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                self.logger.warning(
                    f"Timeout for query '{query}' (attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying after {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                continue
                
            except Exception as e:
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                self.logger.error(
                    f"Unexpected error for query '{query}' "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying after {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                continue
        
        self.logger.error(
            f"Max retries ({self.max_retries}) exceeded for query '{query}'. "
            f"No papers returned."
        )
        return []
    
    def get_recommended_papers(
        self,
        paper_ids: List[str],
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recommended papers based on a set of paper IDs.
        
        Note: This functionality is not directly available in the REST API.
        As a workaround, this returns an empty list. Consider using
        related paper searches or citation networks instead.
        """
        self.logger.warning(
            "Recommendation endpoint not available in REST API. "
            "Consider using citation or reference endpoints instead."
        )
        return []
    
    def _is_quality_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Check if a paper meets quality criteria.
        
        Args:
            paper: Formatted paper dictionary
            
        Returns:
            True if paper meets quality standards
        """
        # Must have title and abstract
        title = paper.get('title') or ''
        abstract = paper.get('abstract') or ''
        
        # Strip whitespace safely
        title = title.strip() if isinstance(title, str) else ''
        abstract = abstract.strip() if isinstance(abstract, str) else ''
        
        if not title or not abstract:
            return False
        
        # Abstract should be substantial
        if len(abstract) < 200:
            return False
        
        # Should have authors
        authors = paper.get('authors', [])
        if not authors:
            return False
        
        # For recent papers, citation requirements are relaxed
        year = paper.get('year')
        current_year = datetime.now().year
        if year and year >= current_year - 1:
            return True  # New papers may not have citations yet
        
        # Older papers should have some citations
        citation_count = paper.get('citation_count', 0)
        if citation_count > 0:
            return True
        
        return False
    
    def _format_paper(self, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format a raw Semantic Scholar API response into pipeline format.
        
        Args:
            paper_data: Raw paper dictionary from API response
            
        Returns:
            Formatted paper dictionary or None if formatting fails
        """
        try:
            # Extract authors
            authors_list = []
            if 'authors' in paper_data and isinstance(paper_data['authors'], list):
                for author_info in paper_data['authors'][:10]:  # Limit to 10 authors
                    if isinstance(author_info, dict) and 'name' in author_info:
                        authors_list.append(author_info['name'])
            
            # Extract PDF URL if available
            pdf_url = None
            if paper_data.get('openAccessPdf') and isinstance(paper_data['openAccessPdf'], dict):
                pdf_url = paper_data['openAccessPdf'].get('url')
            
            # Parse publication date
            published_date = None
            if paper_data.get('publicationDate'):
                try:
                    # API returns date as string in YYYY-MM-DD format
                    published_date = datetime.fromisoformat(paper_data['publicationDate'])
                except (ValueError, TypeError) as e:
                    self.logger.debug(
                        f"Could not parse publicationDate '{paper_data['publicationDate']}': {e}"
                    )
            
            # Fallback to year if precise date is missing
            if not published_date and paper_data.get('year'):
                try:
                    published_date = datetime(paper_data['year'], 1, 1)
                except (ValueError, TypeError):
                    pass
            
            # Final fallback: recent date
            if not published_date:
                published_date = datetime.now(timezone.utc) - timedelta(days=7)
                self.logger.debug(
                    f"No publication date found for paper '{paper_data.get('title', '')[:50]}...', "
                    f"using 7 days ago"
                )
            
            # Extract TLDR if available
            tldr_text = None
            if paper_data.get('tldr') and isinstance(paper_data['tldr'], dict):
                tldr_text = paper_data['tldr'].get('text')
            
            # Extract fields of study
            fields_of_study = paper_data.get('fieldsOfStudy', [])
            if not fields_of_study:
                # Try s2FieldsOfStudy as fallback
                s2_fields = paper_data.get('s2FieldsOfStudy', [])
                if s2_fields and isinstance(s2_fields, list):
                    fields_of_study = [
                        field.get('category', '') 
                        for field in s2_fields 
                        if isinstance(field, dict) and field.get('category')
                    ]
            
            # Build formatted paper
            formatted = {
                'url': paper_data.get('url') or f"https://www.semanticscholar.org/paper/{paper_data.get('paperId', '')}",
                'headline': paper_data.get('title', ''),
                'title': paper_data.get('title', ''),
                'content': paper_data.get('abstract', ''),
                'abstract': paper_data.get('abstract', ''),
                'source': 'Semantic Scholar',
                'source_feed': paper_data.get('venue') or 'Semantic Scholar',
                'published': published_date.isoformat(),
                'published_date': published_date,
                'authors': authors_list,
                'author_names': ', '.join(authors_list[:3]) + (' et al.' if len(authors_list) > 3 else ''),
                'pdf_url': pdf_url,
                'citation_count': paper_data.get('citationCount', 0) or 0,
                'influential_citation_count': paper_data.get('influentialCitationCount', 0) or 0,
                'fields_of_study': fields_of_study,
                'tldr': tldr_text,
                'venue': paper_data.get('venue'),
                'year': paper_data.get('year'),
                'semantic_scholar_id': paper_data.get('paperId'),
                'metadata': {
                    'source': 'semantic_scholar',
                    'citation_count': paper_data.get('citationCount', 0) or 0,
                    'influential_citations': paper_data.get('influentialCitationCount', 0) or 0,
                    'has_pdf': pdf_url is not None,
                    'venue': paper_data.get('venue'),
                    'fields': fields_of_study
                }
            }
            
            return formatted
            
        except Exception as e:
            self.logger.warning(
                f"Error formatting paper: {e}. "
                f"Paper title: {paper_data.get('title', 'Unknown')[:50]}..."
            )
            return None
    
    def _calculate_quality_score(self, paper: Dict[str, Any]) -> float:
        """
        Calculate paper quality score with emphasis on citation velocity.
        
        Prioritizes papers that are gaining citations rapidly (trending).
        
        Args:
            paper: Formatted paper dictionary
            
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # Citation velocity is the PRIMARY factor (0-50 points)
        velocity = paper.get('citation_velocity', 0)
        if velocity > 10:  # >10 citations/day is exceptional
            score += 50
        elif velocity > 5:  # >5 citations/day is excellent
            score += 40
        elif velocity > 2:  # >2 citations/day is very good
            score += 30
        elif velocity > 1:  # >1 citation/day is good
            score += 20
        elif velocity > 0.5:  # >0.5 citations/day is decent
            score += 10
        elif velocity > 0.1:  # >0.1 citations/day is notable
            score += 5
        
        # Recency bonus (0-20 points)
        days_old = paper.get('days_old', 999)
        if days_old <= 3:
            score += 20
        elif days_old <= 7:
            score += 15
        elif days_old <= 14:
            score += 10
        elif days_old <= 30:
            score += 5
        
        # Absolute citation count still matters (0-20 points)
        citations = paper.get('citation_count', 0)
        if citations >= 100:
            score += 20
        elif citations >= 50:
            score += 15
        elif citations >= 20:
            score += 10
        elif citations >= 10:
            score += 5
        
        # Influential citations (0-10 points)
        influential = paper.get('influential_citation_count', 0)
        score += min(influential * 2, 10)
        
        # Content quality (up to 20 points)
        abstract = paper.get('abstract', '')
        if len(abstract) > 1000:
            score += 10
        elif len(abstract) > 500:
            score += 5
        
        if paper.get('tldr'):
            score += 5  # Has AI-generated summary
        
        if paper.get('pdf_url'):
            score += 5  # Has accessible PDF
        
        # Venue prestige (up to 20 points)
        venue = (paper.get('venue') or '').lower()
        prestigious_venues = [
            'nature', 'science', 'cell', 'lancet', 'nejm',  # Top journals
            'neurips', 'icml', 'iclr', 'cvpr', 'acl', 'aaai', 'ijcai', 'kdd',  # Top AI conferences
            'ieee', 'acm', 'pnas', 'prl', 'physical review',  # Other prestigious
            'arxiv'  # Preprint server but valuable for cutting-edge research
        ]
        
        for prestigious in prestigious_venues:
            if prestigious in venue:
                score += 20
                break
        else:
            if venue:  # Any venue is better than none
                score += 5
        
        # Interdisciplinary bonus (up to 5 points)
        fields = paper.get('fields_of_study', [])
        if len(fields) > 2:  # Multiple fields indicate interdisciplinary work
            score += 5
        
        return min(100.0, max(0.0, score))