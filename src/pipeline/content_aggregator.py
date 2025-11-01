import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Services live at src/services/*. Use actual module names in this repo
from src.services.arxiv import ArxivService, ArxivPaper
from src.services.rss import RSSService, DailyReading
from src.services.ai_service import AIService, RankingResult
from src.services.semantic_scholar_service import SemanticScholarService
from src.services.source_ranking_service import SourceRankingService
from src.services.content_extraction_service import ContentExtractionService

from src.services.content_adapter_factory import ContentAdapterFactory, ContentAdapterWrapper


class Section:
    """Newsletter sections as string constants (Enum-like)"""
    SCRIPTURE = "scripture"
    BREAKING_NEWS = "breaking_news"
    BUSINESS = "business"
    TECH_SCIENCE = "tech_science"
    RESEARCH_PAPERS = "research_papers"
    STARTUP = "startup"
    POLITICS = "politics"
    LOCAL = "local"
    MISCELLANEOUS = "miscellaneous"
    EXTRA = "extra"


@dataclass
class RankedItem:
    """Content item with seven-axis Renaissance ranking"""
    id: str
    url: str
    headline: str
    summary_text: str
    source: str
    section: str
    published_date: datetime

    # All 7 Renaissance ranking dimensions
    temporal_impact: float = 0.0
    intellectual_novelty: float = 0.0
    renaissance_breadth: float = 0.0
    actionable_wisdom: float = 0.0
    source_authority: float = 0.0
    signal_clarity: float = 0.0
    transformative_potential: float = 0.0

    # Special handling flags
    preserve_original: bool = False  # If True, skip summarization and preserve original content

    total_score: float = 0.0
    rank: Optional[int] = None

    editorial_note: Optional[str] = None
    angle: Optional[str] = None

    # Backwards-compat constructor support via __init__
    def __init__(
        self,
        id: str,
        url: str,
        headline: str,
        source: str,
        section: str,
        published_date: datetime,
        summary_text: Optional[str] = None,
        content: Optional[str] = None,
        temporal_impact: float = 0.0,
        intellectual_novelty: float = 0.0,
        renaissance_breadth: float = 0.0,
        actionable_wisdom: float = 0.0,
        source_authority: float = 0.0,
        signal_clarity: float = 0.0,
        transformative_potential: float = 0.0,
        total_score: float = 0.0,
        rank: Optional[int] = None,
        editorial_note: Optional[str] = None,
        angle: Optional[str] = None,
        preserve_original: bool = False,
        # legacy aliases
        impact_score: Optional[float] = None,
        delight_score: Optional[float] = None,
        resonance_score: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # Required fields
        self.id = id
        self.url = url
        self.headline = headline
        self.source = source
        self.section = section
        self.published_date = published_date
        # Prefer explicit summary_text; else fall back to legacy 'content'
        self.summary_text = summary_text if summary_text is not None else (content or "")

        # Map legacy axis names if provided
        if impact_score is not None:
            temporal_impact = impact_score
        if delight_score is not None:
            intellectual_novelty = delight_score
        if resonance_score is not None:
            renaissance_breadth = resonance_score

        # Set axis scores
        self.temporal_impact = float(temporal_impact)
        self.intellectual_novelty = float(intellectual_novelty)
        self.renaissance_breadth = float(renaissance_breadth)
        self.actionable_wisdom = float(actionable_wisdom)
        self.source_authority = float(source_authority)
        self.signal_clarity = float(signal_clarity)
        self.transformative_potential = float(transformative_potential)

        # Flags and extra
        # CRITICAL: preserve_original can be passed as a parameter or kwarg
        self.preserve_original = preserve_original or kwargs.get('preserve_original', False)
        self.total_score = float(total_score)
        self.rank = rank
        self.editorial_note = editorial_note
        self.angle = angle

    # Backwards compatibility aliases as properties
    @property
    def impact_score(self) -> float:
        return self.temporal_impact

    @property
    def delight_score(self) -> float:
        return self.intellectual_novelty

    @property
    def resonance_score(self) -> float:
        return self.renaissance_breadth

    def calculate_total_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted total using three-axis defaults for compatibility.
        Defaults to: impact 0.4, delight 0.35, resonance 0.25.
        Accepts custom weights using either canonical keys (temporal_impact, intellectual_novelty, renaissance_breadth)
        or legacy keys (impact, delight, resonance).
        """
        if weights is None:
            weights = {
                "temporal_impact": 0.40,
                "intellectual_novelty": 0.35,
                "renaissance_breadth": 0.25,
                "actionable_wisdom": 0.0,
                "source_authority": 0.0,
                "signal_clarity": 0.0,
                "transformative_potential": 0.0,
            }
        else:
            # Map legacy keys to canonical keys if provided
            if any(k in weights for k in ("impact", "delight", "resonance")):
                weights = {
                    "temporal_impact": weights.get("impact", weights.get("temporal_impact", 0.40)),
                    "intellectual_novelty": weights.get("delight", weights.get("intellectual_novelty", 0.35)),
                    "renaissance_breadth": weights.get("resonance", weights.get("renaissance_breadth", 0.25)),
                    "actionable_wisdom": weights.get("actionable_wisdom", 0.0),
                    "source_authority": weights.get("source_authority", 0.0),
                    "signal_clarity": weights.get("signal_clarity", 0.0),
                    "transformative_potential": weights.get("transformative_potential", 0.0),
                }
        self.total_score = (
            self.temporal_impact * weights.get("temporal_impact", 0.40)
            + self.intellectual_novelty * weights.get("intellectual_novelty", 0.35)
            + self.renaissance_breadth * weights.get("renaissance_breadth", 0.25)
            + self.actionable_wisdom * weights.get("actionable_wisdom", 0.0)
            + self.source_authority * weights.get("source_authority", 0.0)
            + self.signal_clarity * weights.get("signal_clarity", 0.0)
            + self.transformative_potential * weights.get("transformative_potential", 0.0)
        )
        return self.total_score


@dataclass
class FetchResult:
    """Result from a content fetch operation"""
    source: str
    section: str
    items: Any
    fetch_time: float
    error: Optional[str] = None


class AggregationError(Exception):
    """Custom exception for aggregation failures"""
    pass


class ContentAggregator:
    """
    Parallel content fetching and three-axis ranking system.
    """

    def __init__(
        self,
        arxiv: ArxivService,
        rss: RSSService,
        ai: AIService,
        cache_service=None,
        embeddings=None,
        semantic_scholar: Optional[SemanticScholarService] = None,
        source_ranker: Optional[SourceRankingService] = None,
        content_extractor: Optional[ContentExtractionService] = None,
    ) -> None:
        self.arxiv = arxiv
        self.rss = rss
        self.ai = ai
        self.cache_service = cache_service
        self.embeddings = embeddings

        # Initialize source ranking service if not provided
        self.source_ranker = source_ranker or SourceRankingService()
        # Also set as source_ranking_service for backward compatibility
        self.source_ranking_service = self.source_ranker
        # Initialize source_ranking_config from the source_ranker
        self.source_ranking_config = self.source_ranker.authority_config if self.source_ranker else {}
        self.semantic_scholar = semantic_scholar

        # Load per-section guard policies (min_items, max_age_days, fallback queries)
        self.section_policies: Dict[str, Dict[str, Any]] = {}
        try:
            import json
            policy_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config',
                'sections_policy.json'
            )
            if os.path.exists(policy_path):
                with open(policy_path, 'r') as f:
                    self.section_policies = json.load(f)
                logging.getLogger(__name__).info(
                    f"Loaded section guard policies from {policy_path}"
                )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load section policies: {e}")


        # Initialize content extraction service with error handling
        try:
            self.content_extractor = content_extractor or ContentExtractionService()
        except Exception as e:
            self.logger.warning(f"Content extraction service failed to initialize: {e}")
            self.content_extractor = None

        self.parallel_limit = 10
        # Allow configurable timeout for testing
        # Default 1200s (20 minutes) to allow all Exa Websets searches to complete
        # With 18 searches at ~5 min each (count=10) and 3 concurrent slots: 18÷3×5 = 30 minutes needed
        # Phase 4 testing showed all searches complete in ~20 minutes with count=10
        self.fetch_timeout = int(os.getenv('FETCH_TIMEOUT', '1200'))
        # Keep threshold in 30-point scale for initialization expectations/tests
        self.min_score_threshold = 15
        # HARD LIMITS: Exact article counts per section (no ranges!)
        self.items_per_section: Dict[str, Tuple[int, int]] = {
            Section.BREAKING_NEWS: (3, 3),  # Exactly 3
            Section.BUSINESS: (3, 3),       # Exactly 3
            Section.TECH_SCIENCE: (3, 3),   # Exactly 3
            Section.RESEARCH_PAPERS: (5, 5), # Exactly 5
            Section.STARTUP: (2, 2),        # Exactly 2
            Section.SCRIPTURE: (6, 10),     # Keep flexible for Scripture
            Section.POLITICS: (2, 2),       # Exactly 2 per vision
            Section.LOCAL: (2, 2),          # Exactly 2 (1 Miami + 1 Cornell when possible)
            Section.MISCELLANEOUS: (5, 5),  # Exactly 5
            Section.EXTRA: (0, 2),          # 0-2 flexible
        }

        # Initialize content adapter (Perplexity-only configuration)
        try:
            adapter = ContentAdapterFactory.create_from_environment()
            self.news_adapter = ContentAdapterWrapper(adapter)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Content adapter failed to initialize: {e}")
            self.news_adapter = None

        self.logger = logging.getLogger(__name__)

        # Initialize fetch statistics tracking
        self._fetch_stats: List[FetchResult] = []

        # Track initialization state
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the content aggregator and its adapters."""
        if self._initialized:
            return

        try:
            if self.news_adapter:
                await self.news_adapter.initialize()
                self.logger.info("Content aggregator initialized successfully")
            self._initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize content aggregator: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources used by the content aggregator."""
        if not self._initialized:
            return

        try:
            if self.news_adapter:
                await self.news_adapter.cleanup()
                self.logger.info("Content aggregator cleaned up successfully")
            self._initialized = False
        except Exception as e:
            self.logger.warning(f"Error during content aggregator cleanup: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def _enhance_content_with_extraction(self, rss_items: List, section_name: str) -> List:
        """
        Enhance RSS items with extracted web content when beneficial.

        Args:
            rss_items: List of RSS items
            section_name: Newsletter section name for logging

        Returns:
            List of enhanced RSS items with potentially richer content
        """
        # Check if content extraction is available and enabled
        if not self.content_extractor or not getattr(self.content_extractor, 'enabled', False):
            self.logger.debug(f"Content extraction not available for {section_name}, using original items")
            return rss_items

        enhanced_items = []

        try:
            # Process items in batches to avoid overwhelming the extraction service
            batch_size = 5
            for i in range(0, len(rss_items), batch_size):
                batch = rss_items[i:i + batch_size]

                # Process batch concurrently
                extraction_tasks = []
                for rss_item in batch:
                    current_content = rss_item.content or rss_item.description or ""
                    task = self.content_extractor.extract_content_if_needed(
                        url=rss_item.link,
                        headline=rss_item.title,
                        current_content=current_content,
                        source=rss_item.source_feed
                    )
                    extraction_tasks.append(task)

            # Wait for batch to complete
            extracted_contents = await asyncio.gather(*extraction_tasks, return_exceptions=True)


            # Apply extraction results
            for rss_item, extracted in zip(batch, extracted_contents):
                if isinstance(extracted, Exception):
                    self.logger.warning(f"Content extraction error for {rss_item.link}: {extracted}")
                    enhanced_items.append(rss_item)
                elif extracted and extracted.word_count > 0:
                    # Create enhanced RSS item with better content
                    enhanced_item = rss_item
                    enhanced_item.content = extracted.content
                    enhanced_item.author = extracted.author or enhanced_item.author
                    self.logger.info(f"🚀 Enhanced {section_name} content: {rss_item.title[:50]}... "
                                   f"({len(rss_item.description or '')} → {extracted.word_count} words)")
                    enhanced_items.append(enhanced_item)
                else:
                    # No enhancement needed or possible
                    enhanced_items.append(rss_item)

            return enhanced_items

        except Exception as e:
            self.logger.warning(f"Content enhancement failed for {section_name}: {e}")
            # Return original items if enhancement fails
            return rss_items

    async def fetch_all_content_rss_first(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch content prioritizing intelligent RSS feeds over hardcoded searches."""
        tasks: List[asyncio.Task] = []

        # Create tasks for RSS-first intelligent content fetching
        async def fetch_rss_sections_with_timeout():
            start = asyncio.get_event_loop().time()
            try:
                rss_results = await asyncio.wait_for(self._fetch_rss_sections(), timeout=self.fetch_timeout)
                for r in rss_results:
                    if isinstance(r, FetchResult):
                        self._fetch_stats.append(r)
                return rss_results
            except asyncio.TimeoutError:
                self.logger.error(f"RSS sections timed out after {self.fetch_timeout}s - no fallback configured")
                # No fallback - return empty results for failed sections
                return []
            except Exception as e:
                self.logger.error(f"RSS sections failed: {e} - no fallback configured")
                # No fallback - return empty results for failed sections
                return []

        # Research papers (still using existing method)
        async def fetch_research_with_timeout():
            start = asyncio.get_event_loop().time()
            try:
                result = await asyncio.wait_for(self._fetch_research_papers(), timeout=self.fetch_timeout)
                if isinstance(result, FetchResult):
                    self._fetch_stats.append(result)
                return result
            except asyncio.TimeoutError:
                self.logger.warning(f"Research papers timed out after {self.fetch_timeout}s - continuing without them")
                return FetchResult("research", Section.RESEARCH_PAPERS, [], self.fetch_timeout, error="Timeout")
            except Exception as e:
                self.logger.warning(f"Research papers failed: {e} - continuing without them")
                return FetchResult("research", Section.RESEARCH_PAPERS, [], 0.0, error=str(e))

        tasks.append(asyncio.create_task(fetch_rss_sections_with_timeout()))
        tasks.append(asyncio.create_task(fetch_research_with_timeout()))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sections: Dict[str, List[Dict[str, Any]]] = {}
        for res in results:
            if isinstance(res, Exception):
                # Log the exception but continue processing other sections
                self.logger.error(f"Exception in fetch result: {res}")
                continue
            if isinstance(res, list):
                for fr in res:
                    if isinstance(fr, FetchResult):
                        sections.setdefault(fr.section, [])
                        if fr.error is not None:
                            # Include placeholder content for failed sections
                            self.logger.warning(f"Section {fr.section} failed: {fr.error}")

    def _get_guard(self, section_key: str) -> Dict[str, Any]:
        default = {"min_items": 3, "max_age_days": 3, "fallback_query": ""}
        return self.section_policies.get(section_key, default)

    def _ensure_metrics(self) -> None:
        if not hasattr(self, 'run_metrics') or not isinstance(getattr(self, 'run_metrics'), dict):
            self.run_metrics: Dict[str, Dict[str, Any]] = {}

    def _record_section_metrics(
        self,
        section_key: str,
        *,
        raw_count: Optional[int] = None,
        filtered_count: Optional[int] = None,
        final_count: Optional[int] = None,
        fallback_used: Optional[bool] = None,
        alert_flags: Optional[List[str]] = None
    ) -> None:
        self._ensure_metrics()
        m = self.run_metrics.get(section_key, {})
        if raw_count is not None:
            m['raw_count'] = raw_count
        if filtered_count is not None:
            m['filtered_count'] = filtered_count
        if final_count is not None:
            m['final_count'] = final_count
        if fallback_used is not None:
            m['fallback_used'] = fallback_used
        if alert_flags is not None:
            m['alert_flags'] = alert_flags
        self.run_metrics[section_key] = m

    def _emit_run_metrics(self) -> None:
        try:
            import json
            payload = {
                'event': 'run_metrics',
                'timestamp': datetime.now().isoformat(),
                'sections': self.run_metrics,
            }
            self.logger.info(json.dumps(payload))
        except Exception:
            pass

    def _alert_section_empty(self, section_key: str, message: str) -> None:
        # Emit both human-readable and structured alert logs
        self.logger.error(f"ALERT_SECTION_EMPTY: {section_key} - {message}")
        try:
            import json
            self.logger.error(json.dumps({
                'event': 'alert_section_empty',
                'section': section_key,
                'message': message,
                'timestamp': datetime.now().isoformat(),
            }))
        except Exception:
            pass


        # Initialize run metrics container for this aggregation run
        self.run_metrics = {}

        # Research papers - non-critical, can fail gracefully
        async def fetch_research_with_timeout():
            start = asyncio.get_event_loop().time()
            try:
                result = await asyncio.wait_for(self._fetch_research_papers(), timeout=self.fetch_timeout)
                if isinstance(result, FetchResult):
                    self._fetch_stats.append(result)
                return result
            except asyncio.TimeoutError:
                self.logger.warning(f"Research papers timed out after {self.fetch_timeout}s - continuing without them")
                return FetchResult("research", Section.RESEARCH_PAPERS, [], self.fetch_timeout, error="Timeout")
            except Exception as e:
                self.logger.warning(f"Research papers failed: {e} - continuing without them")
                return FetchResult("research", Section.RESEARCH_PAPERS, [], 0.0, error=str(e))

        # Scripture section (special handling)
        async def fetch_scripture_with_timeout():
            start = asyncio.get_event_loop().time()
            try:
                result = await asyncio.wait_for(self._fetch_scripture(), timeout=self.fetch_timeout)
                if isinstance(result, FetchResult):
                    self._fetch_stats.append(result)
                return result
            except asyncio.TimeoutError:
                self.logger.warning(f"Scripture timed out after {self.fetch_timeout}s - continuing without it")
                return FetchResult("rss", Section.SCRIPTURE, [], self.fetch_timeout, error="Timeout")
            except Exception as e:
                self.logger.warning(f"Scripture failed: {e} - continuing without it")
                return FetchResult("rss", Section.SCRIPTURE, [], 0.0, error=str(e))

    async def fetch_all_content(self, sections: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch content from all sources SEQUENTIALLY to respect Exa's 3-webset limit.

        CRITICAL: Exa API only allows 3 concurrent websets. Running all sections in parallel
        causes rate limiting (HTTP 403 "out of credits" errors). We must execute sections
        sequentially to avoid overwhelming the API.

        Args:
            sections: Optional list of section names to fetch. If None, fetches all sections.
                     Valid sections: ['breaking_news', 'business', 'tech_science', 'politics', 'miscellaneous', 'research_papers']

        Returns:
            Dict mapping section names to lists of content items
        """
        # Initialize run metrics container for this aggregation run
        self.run_metrics = {}

        # Define all available sections in priority order
        all_sections = [
            ("breaking_news", self._fetch_breaking_news, Section.BREAKING_NEWS),
            ("business", self._fetch_business_news, Section.BUSINESS),
            ("tech_science", self._fetch_tech_science, Section.TECH_SCIENCE),
            # Removed startup section - not configured in exa_prompts.json
            ("politics", self._fetch_politics, Section.POLITICS),
            ("miscellaneous", self._fetch_miscellaneous, Section.MISCELLANEOUS),
            ("research_papers", self._fetch_research_papers, Section.RESEARCH_PAPERS),
            ("scripture", self._fetch_scripture, Section.SCRIPTURE),
        ]

        # Filter sections based on parameter
        if sections is None:
            sections_to_fetch = all_sections
        else:
            sections_to_fetch = [(name, fn, sec) for name, fn, sec in all_sections if name in sections]
            if not sections_to_fetch:
                self.logger.warning(f"No valid sections found in {sections}. Valid sections: {[name for name, _, _ in all_sections]}")
                return {}

        # Execute sections SEQUENTIALLY to respect Exa's 3-webset limit
        # Each section internally may use up to 3 concurrent websets, which is fine
        results = []
        for name, fn, sec in sections_to_fetch:
            self.logger.info(f"🔄 Fetching section '{name}' (sequential execution to avoid rate limits)...")
            try:
                result = await self._run_fetcher_with_timeout(name, fn, sec)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Section '{name}' failed with exception: {e}")
                results.append(e)

        sections: Dict[str, List[Dict[str, Any]]] = {}
        for res in results:
            if isinstance(res, Exception):
                # Log the exception but continue processing other sections
                self.logger.error(f"Exception in fetch result: {res}")
                continue
            if isinstance(res, FetchResult):
                sections.setdefault(res.section, [])
                if res.error is not None:
                    # Include placeholder content for failed sections
                    self.logger.warning(f"Section {res.section} failed: {res.error}")
                    sections[res.section].append({
                        "headline": f"{res.section.replace('_', ' ').title()} - Content Unavailable",
                        "url": "#",
                        "summary_text": f"We were unable to fetch {res.section.replace('_', ' ')} content at this time. Error: {res.error}",
                        "source": "System Notice",
                        "published": datetime.now().isoformat(),
                        "is_placeholder": True
                    })
                else:
                    # Items may be list or dict depending on section
                    if isinstance(res.items, list):
                        sections[res.section].extend(res.items)
                    else:
                        sections[res.section].append(res.items)

        # Emit structured run metrics for monitoring
        try:
            self._emit_run_metrics()
        except Exception:
            pass

        return sections

    async def _fetch_rss_section(self, section: str) -> FetchResult:
        """Fetch RSS feeds for a specific section."""
        start = asyncio.get_event_loop().time()
        try:
            # Use optimized RSS fetching if available
            if hasattr(self.rss, 'fetch_feeds_by_section_optimized'):
                rss_items = await self.rss.fetch_feeds_by_section_optimized(section, target_items=50, max_feeds_per_section=8)
            else:
                rss_items = await self.rss.fetch_feeds_by_section(section)

            # Convert RSS items to the format expected by the pipeline
            items = []
            for rss_item in rss_items:
                items.append({
                    "headline": rss_item.title,
                    "url": rss_item.link,
                    "content": rss_item.content or rss_item.description,
                    "summary_text": rss_item.description,
                    "source": rss_item.source_feed,
                    "published": rss_item.published_date.isoformat(),
                    "published_date": rss_item.published_date,
                    "guid": rss_item.guid,
                    "author": rss_item.author,
                    "categories": rss_item.categories or [],
                })

            self.logger.info(f"RSS section {section}: fetched {len(items)} items")
            return FetchResult("rss", section, items, asyncio.get_event_loop().time() - start)

        except Exception as e:
            self.logger.error(f"RSS section {section} failed: {e}")
            return FetchResult("rss", section, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_ai_news_sections(self) -> List[FetchResult]:
        """Fetch all AI news sections using rate-limited queue for optimal performance."""
        section_fetchers = [
            ("breaking_news", self._fetch_breaking_news),
            ("business", self._fetch_business_news),
            ("tech_science", self._fetch_tech_science),
            # Removed startup section - not configured in exa_prompts.json
            ("politics", self._fetch_politics),
            ("local", self._fetch_local_news),
            ("miscellaneous", self._fetch_miscellaneous),
            ("catholic", self._fetch_scripture),
        ]

        section_map = {
            "breaking_news": Section.BREAKING_NEWS,
            "business": Section.BUSINESS,
            "tech_science": Section.TECH_SCIENCE,
            "startup": Section.STARTUP,
            "politics": Section.POLITICS,
            "local": Section.LOCAL,
            "miscellaneous": Section.MISCELLANEOUS,
            "catholic": Section.SCRIPTURE,
        }

        # Submit all tasks to run in parallel - the rate limiting is handled by the queue
        tasks = []
        for name, fetcher in section_fetchers:
            task = asyncio.create_task(
                self._run_fetcher_with_timeout(name, fetcher, section_map.get(name, Section.MISCELLANEOUS))
            )
            tasks.append(task)

        self.logger.info(f"Submitting {len(tasks)} sections to rate-limited queue...")

        # Wait for all sections to complete (rate limiting happens in the queue)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: List[FetchResult] = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Section fetch failed with exception: {result}")
                # Add a placeholder failure result
                out.append(FetchResult("rss", Section.MISCELLANEOUS, [], 0.0, error=str(result)))
            elif isinstance(result, FetchResult):
                out.append(result)
            else:
                self.logger.error(f"Unexpected result type: {type(result)}")

        return out

    async def _run_fetcher_with_timeout(self, name: str, fetcher, section_enum) -> FetchResult:
        """Run a single fetcher with timeout and error handling."""
        try:
            self.logger.info(f"Starting {name} section...")
            result = await asyncio.wait_for(fetcher(), timeout=self.fetch_timeout)
            if isinstance(result, FetchResult):
                self.logger.info(f"✓ {name} completed: {len(result.items)} items")
                return result
            else:
                self.logger.error(f"✗ {name} returned invalid result type: {type(result)}")
                return FetchResult("rss", section_enum, [], 0.0, error="Invalid result type")

        except Exception as e:
            self.logger.error(f"✗ AI news section '{name}' failed: {e}")
            return FetchResult("ai_news", section_enum, [], 0.0, error=str(e))

    async def _fetch_breaking_news_rss(self) -> FetchResult:
        """Fetch breaking news from curated RSS feeds with intelligent AI ranking."""
        start = asyncio.get_event_loop().time()
        try:
            self.logger.info("Breaking News: Fetching from curated RSS feeds")

            # Fetch from RSS feeds instead of hardcoded searches
            rss_items = await self.rss.fetch_feeds_by_section("breaking_news")

            # Enhance content with web extraction for high-quality sources
            enhanced_rss_items = await self._enhance_content_with_extraction(rss_items, "breaking_news")

            # Convert RSS items to the expected format
            items = []
            for rss_item in enhanced_rss_items:
                items.append({
                    "id": rss_item.guid or f"rss_{rss_item.title[:20]}",
                    "headline": rss_item.title,
                    "url": rss_item.link,
                    "summary_text": rss_item.content or rss_item.description,
                    "source": rss_item.source_feed,
                    "published": rss_item.published_date.isoformat() if rss_item.published_date else datetime.now().isoformat(),
                })

            # Apply multi-stage pipeline for intelligent ranking and selection
            ranked_items = self._apply_multi_stage_pipeline(items, Section.BREAKING_NEWS, max_age_days=1, min_items=3)

            self.logger.info(f"Breaking News: RSS feeds provided {len(rss_items)} items, selected {len(ranked_items)} after AI ranking")
            return FetchResult("rss_intelligent", Section.BREAKING_NEWS, ranked_items, asyncio.get_event_loop().time() - start)

        except Exception as e:
            self.logger.error(f"Breaking News RSS fetch failed: {e}")
            return FetchResult("rss_intelligent", Section.BREAKING_NEWS, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_breaking_news(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            guard = self._get_guard("breaking_news")
            min_items = int(guard.get("min_items", 3))
            max_age_days = int(guard.get("max_age_days", 2))
            adapter_name = type(self.news_adapter.adapter).__name__ if hasattr(self.news_adapter, 'adapter') else type(self.news_adapter).__name__
            self.logger.info(f"Fetching breaking news using {adapter_name}")
            # Use content adapter for breaking news
            result = await self.news_adapter.search_optimized_rate_limited("breaking_news")
            articles = result.articles
            raw_count = len(articles)

            # Apply filtering and ranking pipeline
            items = self._apply_multi_stage_pipeline(articles, Section.BREAKING_NEWS, max_age_days=max_age_days, min_items=min_items)
            filtered_count = len(items)
            self.logger.info(f"Breaking news: {filtered_count} items after multi-stage pipeline from {raw_count} raw articles")

            fallback_used = False
            alert_flags: List[str] = []

            # Fallback: if too few items, broaden query and relax filters with trusted sources
            if len(items) < min_items:
                self.logger.warning(f"Breaking News: Only {len(items)} items found, trying fallback search")
                fallback_query = guard.get("fallback_query") or "breaking news top headlines Reuters AP BBC Bloomberg FT WSJ CNBC"
                fallback_used = True
                fallback_result = await self.news_adapter.search_optimized_rate_limited("breaking_news", fallback_query)

                # Trusted high-authority sources for breaking news
                trusted_sources = [
                    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
                    "bloomberg.com", "ft.com", "wsj.com", "nytimes.com", "cnbc.com"
                ]

                # Combine and deduplicate by URL
                existing_urls = {it["url"] for it in items}
                combined = list(items)
                for it in (fallback_result.articles or []):
                    if it.get("url") and it["url"] not in existing_urls:
                        combined.append(it)
                        existing_urls.add(it["url"])

                # Validate sources and re-run pipeline with slightly relaxed age
                validated = self._validate_sources(combined, trusted_sources, "Breaking News")
                items = self._apply_multi_stage_pipeline(validated, Section.BREAKING_NEWS, max_age_days=max_age_days + 1, min_items=min_items)

                # Final guard: if still short, take top N from fallback directly (sorted by quality)
                if len(items) < min_items and fallback_result.articles:
                    self.logger.warning("breaking_news: Using direct picks after fallback to reach target")
                    alert_flags.append("used_direct_picks_after_fallback")
                    direct = self._sort_by_quality_indicators(fallback_result.articles)
                    # Dedup again
                    final_urls = {it["url"] for it in items}
                    for it in direct:
                        if it.get("url") and it["url"] not in final_urls:
                            items.append(it)
                            final_urls.add(it["url"])
                        if len(items) >= min_items:
                            break

            # Record metrics for this section
            self._record_section_metrics(
                "breaking_news",
                raw_count=raw_count,
                filtered_count=filtered_count,
                final_count=len(items),
                fallback_used=fallback_used,
                alert_flags=alert_flags or None,
            )

            return FetchResult(
                source="rss",
                section=Section.BREAKING_NEWS,
                items=items,
                fetch_time=asyncio.get_event_loop().time() - start,
            )
        except Exception as e:
            # Breaking news is critical; return error result
            self.logger.error(f"Breaking news RSS search failed: {e}")
            return await self._handle_fetch_failure("rss", e)

    async def _fetch_business_news_rss(self) -> FetchResult:
        """Fetch business news from curated RSS feeds with intelligent AI ranking."""
        start = asyncio.get_event_loop().time()
        try:
            self.logger.info("Business: Fetching from curated RSS feeds")

            # Fetch from RSS feeds instead of hardcoded searches
            rss_items = await self.rss.fetch_feeds_by_section("business")

            # Enhance content with web extraction for high-quality sources
            enhanced_rss_items = await self._enhance_content_with_extraction(rss_items, "business")

            # Convert RSS items to the expected format
            items = []
            for rss_item in enhanced_rss_items:
                items.append({
                    "id": rss_item.guid or f"rss_{rss_item.title[:20]}",
                    "headline": rss_item.title,
                    "url": rss_item.link,
                    "summary_text": rss_item.content or rss_item.description,
                    "source": rss_item.source_feed,
                    "published": rss_item.published_date.isoformat() if rss_item.published_date else datetime.now().isoformat(),
                })

            # Apply multi-stage pipeline for intelligent ranking and selection
            ranked_items = self._apply_multi_stage_pipeline(items, Section.BUSINESS, max_age_days=3, min_items=3)

            self.logger.info(f"Business: RSS feeds provided {len(rss_items)} items, selected {len(ranked_items)} after AI ranking")
            return FetchResult("rss_intelligent", Section.BUSINESS, ranked_items, asyncio.get_event_loop().time() - start)

        except Exception as e:
            self.logger.error(f"Business RSS fetch failed: {e}")
            return FetchResult("rss_intelligent", Section.BUSINESS, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_business_news(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            guard = self._get_guard("business")
            min_items = int(guard.get("min_items", 3))
            max_age_days = int(guard.get("max_age_days", 3))
            adapter_name = type(self.news_adapter.adapter).__name__ if hasattr(self.news_adapter, 'adapter') else type(self.news_adapter).__name__
            self.logger.info(f"Fetching business news using {adapter_name}")
            # Use content adapter for business news
            result = await self.news_adapter.search_optimized_rate_limited("business")
            articles = result.articles
            raw_count = len(articles)

            # Apply filtering and ranking pipeline
            items = self._apply_multi_stage_pipeline(articles, Section.BUSINESS, max_age_days=max_age_days, min_items=min_items)
            filtered_count = len(items)
            self.logger.info(f"Business: {filtered_count} items after multi-stage pipeline from {raw_count} raw articles")

            fallback_used = False
            alert_flags: List[str] = []

            # Fallback: if too few items, broaden query and relax filters with trusted sources
            if len(items) < min_items:
                self.logger.warning("Business: Only %d items found, trying fallback search", len(items))
                fallback_query = guard.get("fallback_query") or "business markets economy stocks finance Reuters Bloomberg FT WSJ CNBC"
                fallback_used = True
                fallback_result = await self.news_adapter.search_optimized_rate_limited("business", fallback_query)

                trusted_sources = [
                    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com", "apnews.com"
                ]

                # Combine and deduplicate
                existing_urls = {it["url"] for it in items}
                combined = list(items)
                for it in (fallback_result.articles or []):
                    if it.get("url") and it["url"] not in existing_urls:
                        combined.append(it)
                        existing_urls.add(it["url"])

                validated = self._validate_sources(combined, trusted_sources, "Business")
                items = self._apply_multi_stage_pipeline(validated, Section.BUSINESS, max_age_days=max_age_days + 1, min_items=min_items)

                # Final guard: if still short, take top N from fallback directly
                if len(items) < min_items and fallback_result.articles:
                    self.logger.warning("business: Using direct picks after fallback to reach target")
                    alert_flags.append("used_direct_picks_after_fallback")
                    direct = self._sort_by_quality_indicators(fallback_result.articles)
                    final_urls = {it["url"] for it in items}
                    for it in direct:
                        if it.get("url") and it["url"] not in final_urls:
                            items.append(it)
                            final_urls.add(it["url"])
                        if len(items) >= min_items:
                            break

            # Record metrics for this section
            self._record_section_metrics(
                "business",
                raw_count=raw_count,
                filtered_count=filtered_count,
                final_count=len(items),
                fallback_used=fallback_used,
                alert_flags=alert_flags or None,
            )

            return FetchResult(
                source="rss",
                section=Section.BUSINESS,
                items=items,
                fetch_time=asyncio.get_event_loop().time() - start,
            )
        except Exception as e:
            self.logger.error(f"Business RSS search failed: {e}")
            return await self._handle_fetch_failure("rss", e)

    async def _fetch_tech_science_rss(self) -> FetchResult:
        """Fetch tech & science content from curated RSS feeds with intelligent AI ranking."""
        start = asyncio.get_event_loop().time()
        try:
            self.logger.info("Tech/Science: Fetching from curated RSS feeds")

            # Fetch from RSS feeds instead of hardcoded searches
            rss_items = await self.rss.fetch_feeds_by_section("tech_science")

            # Enhance content with web extraction for high-quality sources
            enhanced_rss_items = await self._enhance_content_with_extraction(rss_items, "tech_science")

            # Convert RSS items to the expected format
            items = []
            for rss_item in enhanced_rss_items:
                items.append({
                    "id": rss_item.guid or f"rss_{rss_item.title[:20]}",
                    "headline": rss_item.title,
                    "url": rss_item.link,
                    "summary_text": rss_item.content or rss_item.description,
                    "source": rss_item.source_feed,
                    "published": rss_item.published_date.isoformat() if rss_item.published_date else datetime.now().isoformat(),
                })

            # Apply multi-stage pipeline for intelligent ranking and selection
            ranked_items = self._apply_multi_stage_pipeline(items, Section.TECH_SCIENCE, max_age_days=7, min_items=3)

            self.logger.info(f"Tech/Science: RSS feeds provided {len(rss_items)} items, selected {len(ranked_items)} after AI ranking")
            return FetchResult("rss_intelligent", Section.TECH_SCIENCE, ranked_items, asyncio.get_event_loop().time() - start)

        except Exception as e:
            self.logger.error(f"Tech/Science RSS fetch failed: {e}")
            return FetchResult("rss_intelligent", Section.TECH_SCIENCE, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_tech_science(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            guard = self._get_guard("tech_science")
            min_items = int(guard.get("min_items", 3))
            max_age_days = int(guard.get("max_age_days", 14))
            fallback_query = guard.get("fallback_query", "technology science past week")

            # Use AI web search for tech/science content (Perplexity or AI news adapter)
            self.logger.info("Fetching tech/science using AI web search (Perplexity/AI adapter)")
            result = await self.news_adapter.search_optimized_rate_limited("tech_science")
            self.logger.info("Tech/Science: Got %d articles from AI web search", len(result.articles))
            items_raw = result.articles
            # Validate sources - allow any tech news source
            items_validated = self._validate_sources(
                items_raw,
                [],  # Don't restrict to specific domains
                "Tech/Science"
            )
            # Use multi-stage pipeline for better quality control
            items = self._apply_multi_stage_pipeline(items_validated, Section.TECH_SCIENCE, max_age_days=max_age_days, min_items=min_items)
            self.logger.info("Tech/Science: %d items after multi-stage pipeline from %d raw", len(items), len(items_raw))

            # Fallback if under target
            if len(items) < min_items:
                self.logger.warning("Tech/Science: Only %d items after pipeline (need %d), trying fallback query", len(items), min_items)
                fallback_result = await self.news_adapter.search_optimized_rate_limited("tech_science", fallback_query)
                combined = items + [it for it in fallback_result.articles if it not in items]
                items = self._apply_multi_stage_pipeline(combined, Section.TECH_SCIENCE, max_age_days=max_age_days, min_items=min_items)
                if len(items) < min_items and fallback_result.articles:
                    self.logger.warning("tech_science: Using direct picks after fallback to meet target")
                    items = fallback_result.articles[:min_items]

            return FetchResult("rss", Section.TECH_SCIENCE, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            return await self._handle_fetch_failure("ai_news", e)

    async def _fetch_startup_insights(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            guard = self._get_guard("startup")
            min_items = int(guard.get("min_items", 2))
            max_age_days = int(guard.get("max_age_days", 14))
            fallback_query = guard.get("fallback_query", "startup funding launch product update")

            # Use content adapter for startup insights
            adapter_name = type(self.news_adapter.adapter).__name__ if hasattr(self.news_adapter, 'adapter') else type(self.news_adapter).__name__
            self.logger.info(f"Fetching startup insights using {adapter_name}")
            result = await self.news_adapter.search_optimized_rate_limited("startup")
            items_raw = result.articles
            # Use tier-based source validation (no hardcoded domains)
            items_validated = self._validate_sources(
                items_raw,
                [],  # Empty allowed_domains - tier system handles this
                "Startup"
            )
            # Use multi-stage pipeline for better quality control
            items = self._apply_multi_stage_pipeline(items_validated, Section.STARTUP, max_age_days=max_age_days, min_items=min_items)

            # Fallback if under target
            if len(items) < min_items:
                self.logger.warning("Startup: Only %d items after pipeline (need %d), trying fallback query", len(items), min_items)
                fallback_result = await self.news_adapter.search_optimized_rate_limited("startup", fallback_query)
                combined = items + [it for it in fallback_result.articles if it not in items]
                items = self._apply_multi_stage_pipeline(combined, Section.STARTUP, max_age_days=max_age_days, min_items=min_items)
                if len(items) < min_items and fallback_result.articles:
                    self.logger.warning("startup: Using direct picks after fallback to meet target")
                    items = fallback_result.articles[:min_items]

            return FetchResult("rss", Section.STARTUP, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            return await self._handle_fetch_failure("ai_news", e)


    async def _backfill_with_adapter(self, section: str, needed: int, attempt: int = 0) -> List[Dict[str, Any]]:
        """Use AI web adapter with fallback query and progressively relaxed filters to backfill a section."""
        guard = self._get_guard(section)
        base_max_age = int(guard.get("max_age_days", 7))
        fallback_query = guard.get("fallback_query") or section.replace('_', ' ')
        # Slightly broaden window per attempt
        max_age = base_max_age + max(0, attempt) * 7
        self.logger.warning(f"Backfilling {section}: need {needed}, attempt {attempt}, max_age_days={max_age}")
        # Run adapter with fallback query
        result = await self.news_adapter.search_optimized_rate_limited(section, fallback_query)
        articles = result.articles or []
        # Validate sources (section-specific domain lists are handled inside)
        validated = self._validate_sources(articles, [], section)
        # Run multi-stage pipeline with relaxed window and target count
        processed = self._apply_multi_stage_pipeline(validated, section, max_age_days=max_age, min_items=needed)
        # If still short, take direct top picks sorted by quality heuristic
        if len(processed) < needed and articles:
            self.logger.warning(f"{section}: pipeline produced {len(processed)} (<{needed}); topping up with direct picks")
            direct = self._sort_by_quality_indicators(articles)
            # Dedup by URL
            seen = {it.get('url') for it in processed}
            for it in direct:
                if it.get('url') and it['url'] not in seen:
                    processed.append(it)
                    seen.add(it['url'])
                if len(processed) >= needed:
                    break
        return processed[:needed]

    def _shortages_for_strict_sections(self, selected: Dict[str, List[RankedItem]]) -> Dict[str, int]:
        """Compute how many items each strict-count section is short by."""
        shortages: Dict[str, int] = {}
        for section, (min_items, max_items) in self.items_per_section.items():
            if min_items == max_items:
                current = len(selected.get(section, []))
                if current < min_items:
                    shortages[section] = min_items - current
        return shortages

    async def _fetch_politics(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            guard = self._get_guard("politics")
            min_items = int(guard.get("min_items", 2))
            max_age_days = int(guard.get("max_age_days", 3))
            fallback_query = guard.get("fallback_query", "US politics government past 48 hours")

            # Use content adapter for US politics
            adapter_name = type(self.news_adapter.adapter).__name__ if hasattr(self.news_adapter, 'adapter') else type(self.news_adapter).__name__
            self.logger.info(f"Fetching US politics using {adapter_name}")
            result = await self.news_adapter.search_optimized_rate_limited("politics")

            self.logger.info(f"Politics: Got {len(result.articles)} articles from AI web search")
            items_raw = result.articles

            # Validate sources BEFORE other filtering - expanded trusted sources list
            trusted_sources = [
                "apnews.com", "reuters.com", "pbs.org", "npr.org",
                "bbc.com", "bbc.co.uk", "propublica.org", "politico.com"
            ]
            items_validated = self._validate_sources(items_raw, trusted_sources, "Politics")
            # Use multi-stage pipeline for better quality control and to ensure we meet exact target
            items = self._apply_multi_stage_pipeline(items_validated, Section.POLITICS, max_age_days=max_age_days, min_items=min_items)
            self.logger.info("Politics: %d items after multi-stage pipeline from %d raw", len(items), len(items_raw))

            # If still under target, try a fallback search with policy query
            if len(items) < min_items:
                self.logger.warning(f"Politics: Only {len(items)} items found (need {min_items}), trying fallback search")
                fallback_result = await self.news_adapter.search_optimized_rate_limited("politics", fallback_query)

                # Deduplicate by URL
                existing_urls = {item["url"] for item in items}
                for fallback_item in fallback_result.articles:
                    if fallback_item["url"] not in existing_urls:
                        items_raw.append(fallback_item)
                        existing_urls.add(fallback_item["url"])

                # Re-filter combined items
                items_validated = self._validate_sources(items_raw, trusted_sources, "Politics")
                # Use multi-stage pipeline even in fallback
                items = self._apply_multi_stage_pipeline(items_validated, Section.POLITICS, max_age_days=max_age_days, min_items=min_items)
                if len(items) < min_items and fallback_result.articles:
                    self.logger.warning("politics: Using direct picks after fallback to meet target")
                    items = fallback_result.articles[:min_items]

            return FetchResult("rss", Section.POLITICS, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            # Non-critical; return error result
            return FetchResult("rss", Section.POLITICS, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_local_news(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            # VISION.txt specifies: Miami Herald for Miami, Cornell news sources
            self.logger.info("Fetching local news from Miami Herald and Cornell via AI news adapter")
            from datetime import datetime
            this_week = datetime.now().strftime("%B %d, %Y")

            # For local news, we'll try to use RSS feeds but this section may need manual RSS configuration
            self.logger.info("Local news: RSS-based local news not yet fully implemented, returning empty results")
            miami_result = RSSAdapterResult(
                query="Miami news",
                articles=[],
                search_time_ms=0,
                total_results=0
            )
            cornell_result = RSSAdapterResult(
                query="Cornell news",
                articles=[],
                search_time_ms=0,
                total_results=0
            )

            # Log what we got from each source
            self.logger.info(f"Local: Got {len(miami_result.articles)} Miami Herald articles")
            self.logger.info(f"Local: Got {len(cornell_result.articles)} Cornell articles")

            # Process Miami and Cornell separately to ensure 1+1 geographic balance
            miami_items_raw = []
            for article in miami_result.articles:
                item = article.copy()
                item["location"] = "Miami"
                miami_items_raw.append(item)

            cornell_items_raw = []
            for article in cornell_result.articles:
                item = article.copy()
                item["location"] = "Cornell"
                cornell_items_raw.append(item)

            # Validate sources for each location separately - use generic "local" section
            miami_validated = self._validate_sources(miami_items_raw, ["miamiherald.com"], "local")
            cornell_validated = self._validate_sources(cornell_items_raw, ["news.cornell.edu", "cornellsun.com"], "local")

            # Apply pipeline to each location separately and take top 1 from each
            miami_processed = self._apply_multi_stage_pipeline(miami_validated, Section.LOCAL, max_age_days=14, min_items=1)
            cornell_processed = self._apply_multi_stage_pipeline(cornell_validated, Section.LOCAL, max_age_days=14, min_items=1)

            # Combine exactly 1 from each location (target: 1 Miami + 1 Cornell = 2 total)
            items = []
            if miami_processed:
                items.append(miami_processed[0])  # Take best Miami article
                self.logger.info("Local: Selected 1 Miami article")

            if cornell_processed:
                items.append(cornell_processed[0])  # Take best Cornell article
                self.logger.info("Local: Selected 1 Cornell article")

            if not items:
                self.logger.warning("Local: No articles found from either Miami or Cornell")

            total_raw = len(miami_validated) + len(cornell_validated)
            self.logger.info("Local: %d items after multi-stage pipeline from %d raw (%d Miami + %d Cornell)",
                           len(items), total_raw, len(miami_validated), len(cornell_validated))
            return FetchResult("rss", Section.LOCAL, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            return FetchResult("rss", Section.LOCAL, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_miscellaneous_search(self, search_name: str, custom_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Optimized method for miscellaneous searches using new configuration."""
        try:
            adapter_name = type(self.news_adapter.adapter).__name__ if hasattr(self.news_adapter, 'adapter') else type(self.news_adapter).__name__
            self.logger.info(f"Miscellaneous/{search_name}: Starting search using {adapter_name}")
            # Use content adapter for miscellaneous content with custom query
            result = await self.news_adapter.search_optimized_rate_limited("miscellaneous", custom_query)

            items = []
            for article in result.articles:
                item = article.copy()
                item["search_category"] = search_name  # Track which search found this
                items.append(item)

            self.logger.info(f"Miscellaneous/{search_name}: Found {len(items)} items")
            return items

        except Exception as e:
            self.logger.error(f"Miscellaneous/{search_name} search failed: {e}")
            return []

    async def _fetch_miscellaneous_rss(self) -> FetchResult:
        """Fetch miscellaneous intellectual content from curated RSS feeds with intelligent AI ranking."""
        start = asyncio.get_event_loop().time()
        try:
            self.logger.info("Miscellaneous: Fetching from curated RSS feeds")

            # Fetch from RSS feeds instead of hardcoded searches
            rss_items = await self.rss.fetch_feeds_by_section("miscellaneous")

            # Enhance content with web extraction for high-quality sources
            enhanced_rss_items = await self._enhance_content_with_extraction(rss_items, "miscellaneous")

            # Convert RSS items to the expected format
            items = []
            for rss_item in enhanced_rss_items:
                items.append({
                    "id": rss_item.guid or f"rss_{rss_item.title[:20]}",
                    "headline": rss_item.title,
                    "url": rss_item.link,
                    "summary_text": rss_item.content or rss_item.description,
                    "source": rss_item.source_feed,
                    "published": rss_item.published_date.isoformat() if rss_item.published_date else datetime.now().isoformat(),
                })

            # Apply multi-stage pipeline for intelligent ranking and selection
            ranked_items = self._apply_multi_stage_pipeline(items, Section.MISCELLANEOUS, max_age_days=7, min_items=5)

            self.logger.info(f"Miscellaneous: RSS feeds provided {len(rss_items)} items, selected {len(ranked_items)} after AI ranking")
            return FetchResult("rss_intelligent", Section.MISCELLANEOUS, ranked_items, asyncio.get_event_loop().time() - start)

        except Exception as e:
            self.logger.error(f"Miscellaneous RSS fetch failed: {e}")
            return FetchResult("rss_intelligent", Section.MISCELLANEOUS, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_miscellaneous(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            guard = self._get_guard("miscellaneous")
            min_items = int(guard.get("min_items", 5))
            max_age_days = int(guard.get("max_age_days", 14))
            adapter_name = type(self.news_adapter.adapter).__name__ if hasattr(self.news_adapter, 'adapter') else type(self.news_adapter).__name__
            self.logger.info(f"Fetching miscellaneous using {adapter_name}")

            # Use content adapter for miscellaneous content (reads from exa_prompts.json)
            result = await self.news_adapter.search_optimized_rate_limited("miscellaneous")
            articles = result.articles
            raw_count = len(articles)

            # Apply filtering and ranking pipeline
            items = self._apply_multi_stage_pipeline(articles, Section.MISCELLANEOUS, max_age_days=max_age_days, min_items=min_items)
            filtered_count = len(items)
            self.logger.info(f"Miscellaneous: {filtered_count} items after multi-stage pipeline from {raw_count} raw articles")

            fallback_used = False
            alert_flags: List[str] = []

            # Fallback: if too few items, broaden query and relax filters with trusted sources
            if len(items) < min_items:
                self.logger.warning(f"Miscellaneous: Only {len(items)} items found, trying fallback search")
                fallback_query = guard.get("fallback_query") or "intellectual essays philosophy psychology culture history The Atlantic The New Yorker Aeon Harper's"
                fallback_used = True
                fallback_result = await self.news_adapter.search_optimized_rate_limited("miscellaneous", fallback_query)

                # Trusted high-quality intellectual sources
                trusted_sources = [
                    "theatlantic.com", "newyorker.com", "aeon.co", "harpers.org",
                    "lrb.co.uk", "nybooks.com", "theparisreview.org", "nplusonemag.com"
                ]

                fallback_items = self._apply_multi_stage_pipeline(
                    fallback_result.articles,
                    Section.MISCELLANEOUS,
                    max_age_days=max_age_days * 2,  # More lenient for fallback
                    min_items=min_items,
                    trusted_sources=trusted_sources
                )

                # Merge with original items, prioritizing original
                seen_urls = {item.get("url") for item in items}
                for fb_item in fallback_items:
                    if fb_item.get("url") not in seen_urls and len(items) < min_items:
                        items.append(fb_item)
                        seen_urls.add(fb_item.get("url"))

                self.logger.info(f"Miscellaneous: After fallback, have {len(items)} items")

            # Quality check
            if len(items) < min_items:
                alert_flags.append(f"insufficient_items_{len(items)}")
                self.logger.warning(f"Miscellaneous: Still only {len(items)} items after fallback (target: {min_items})")

            return FetchResult("exa_websets", Section.MISCELLANEOUS, items, asyncio.get_event_loop().time() - start)

        except Exception as e:
            self.logger.error(f"Miscellaneous search failed: {e}")
            return await self._handle_fetch_failure("exa_websets", e)

    def _expand_scripture_reference(self, reference: str) -> str:
        """Expand scripture abbreviations to full book names."""
        abbreviations = {
            # Old Testament
            'Gn': 'Genesis', 'Ex': 'Exodus', 'Lv': 'Leviticus', 'Nm': 'Numbers', 'Dt': 'Deuteronomy',
            'Jos': 'Joshua', 'Jgs': 'Judges', 'Ru': 'Ruth', '1 Sm': '1 Samuel', '2 Sm': '2 Samuel',
            '1 Kgs': '1 Kings', '2 Kgs': '2 Kings', '1 Chr': '1 Chronicles', '2 Chr': '2 Chronicles',
            'Ezr': 'Ezra', 'Neh': 'Nehemiah', 'Tb': 'Tobit', 'Jdt': 'Judith', 'Est': 'Esther',
            '1 Mc': '1 Maccabees', '2 Mc': '2 Maccabees', 'Jb': 'Job', 'Ps': 'Psalms', 'Prv': 'Proverbs',
            'Eccl': 'Ecclesiastes', 'Sg': 'Song of Songs', 'Wis': 'Wisdom', 'Sir': 'Sirach',
            'Is': 'Isaiah', 'Jer': 'Jeremiah', 'Lam': 'Lamentations', 'Bar': 'Baruch', 'Ez': 'Ezekiel',
            'Dn': 'Daniel', 'Hos': 'Hosea', 'Jl': 'Joel', 'Am': 'Amos', 'Ob': 'Obadiah',
            'Jon': 'Jonah', 'Mi': 'Micah', 'Na': 'Nahum', 'Hb': 'Habakkuk', 'Zep': 'Zephaniah',
            'Hg': 'Haggai', 'Zec': 'Zechariah', 'Mal': 'Malachi',
            # New Testament
            'Mt': 'Matthew', 'Mk': 'Mark', 'Lk': 'Luke', 'Jn': 'John', 'Acts': 'Acts',
            'Rom': 'Romans', '1 Cor': '1 Corinthians', '2 Cor': '2 Corinthians', 'Gal': 'Galatians',
            'Eph': 'Ephesians', 'Phil': 'Philippians', 'Col': 'Colossians',
            '1 Thes': '1 Thessalonians', '2 Thes': '2 Thessalonians', '1 Tm': '1 Timothy',
            '2 Tm': '2 Timothy', 'Ti': 'Titus', 'Phlm': 'Philemon', 'Heb': 'Hebrews',
            'Jas': 'James', '1 Pt': '1 Peter', '2 Pt': '2 Peter', '1 Jn': '1 John',
            '2 Jn': '2 John', '3 Jn': '3 John', 'Jude': 'Jude', 'Rv': 'Revelation'
        }

        expanded = reference
        for abbr, full in abbreviations.items():
            # Match abbreviation at start of reference or after a space
            import re
            pattern = r'\b' + re.escape(abbr) + r'\b'
            expanded = re.sub(pattern, full, expanded)

        return expanded

    async def _fetch_scripture(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        all_items = []

        # First, get USCCB daily readings from RSS
        try:
            self.logger.info("Scripture: Fetching USCCB daily readings from RSS service")
            content = await self.rss.get_todays_spiritual_content()
            readings = content.get("readings")

            if readings:
                self.logger.info("Scripture: Found daily readings for %s", readings.date if hasattr(readings, 'date') else 'today')
                base_url = "https://bible.usccb.org/daily-bible-reading"
                published = readings.date.isoformat() if readings.date else datetime.now().isoformat()

                # Add First Reading
                if readings.first_reading and readings.first_reading.get('text'):
                    reference = self._expand_scripture_reference(readings.first_reading.get('reference', 'Daily Reading'))
                    all_items.append({
                        "headline": f"First Reading: {reference}",
                        "url": base_url,
                        "summary_text": readings.first_reading.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })

                # Add Second Reading (if it exists - typically on Sundays/Solemnities)
                if readings.second_reading and readings.second_reading.get('text'):
                    reference = self._expand_scripture_reference(readings.second_reading.get('reference', 'Second Reading'))
                    all_items.append({
                        "headline": f"Second Reading: {reference}",
                        "url": base_url,
                        "summary_text": readings.second_reading.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })

                # Add Gospel
                if readings.gospel and readings.gospel.get('text'):
                    reference = self._expand_scripture_reference(readings.gospel.get('reference', 'Daily Gospel'))
                    all_items.append({
                        "headline": f"Gospel: {reference}",
                        "url": base_url,
                        "summary_text": readings.gospel.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })

                # Add Responsorial Psalm if present
                if hasattr(readings, 'responsorial_psalm') and readings.responsorial_psalm and readings.responsorial_psalm.get('text'):
                    psalm_ref = self._expand_scripture_reference(readings.responsorial_psalm.get('reference', 'Psalm'))
                    all_items.append({
                        "headline": f"Responsorial Psalm: {psalm_ref}",
                        "url": base_url,
                        "summary_text": readings.responsorial_psalm.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })
                self.logger.info("Scripture: Successfully fetched %d USCCB readings", len(all_items))
        except Exception as e:
            self.logger.error("Scripture: Failed to fetch USCCB readings: %s", e)

        # Get Catholic Daily Reflections from RSS feed (more reliable than AI search)
        try:
            self.logger.info("Scripture: Fetching daily reflections from Catholic Daily Reflections RSS feed")
            # Fetch from the RSS feed directly
            reflections = await self.rss.fetch_configured_feed("catholic_daily_reflections")

            if reflections:
                # Get today's reflection (RSS feeds are sorted by date, newest first)
                today = datetime.now().date()
                for item in reflections[:3]:  # Check first 3 items for today's reflection
                    # Check if this is today's reflection
                    item_date = item.published_date.date() if item.published_date else None
                    if item_date and abs((item_date - today).days) <= 1:  # Today or yesterday (timezone tolerance)
                        all_items.append({
                            "headline": "Catholic Daily Reflection",  # Consistent title
                            "url": item.link,
                            "summary_text": item.content or item.description,  # Use full content if available
                            "source": "Catholic Daily Reflections",
                            "published": item.published_date.isoformat() if item.published_date else datetime.now().isoformat(),
                            "preserve_original": False,  # This one should be summarized
                        })
                        self.logger.info("Scripture: Found today's reflection from Catholic Daily Reflections RSS")
                        break
                else:
                    # If no today's reflection found, use the most recent one
                    if reflections:
                        item = reflections[0]
                        all_items.append({
                            "headline": "Catholic Daily Reflection",
                            "url": item.link,
                            "summary_text": item.content or item.description,
                            "source": "Catholic Daily Reflections",
                            "published": item.published_date.isoformat() if item.published_date else datetime.now().isoformat(),
                            "preserve_original": False,
                        })
                        self.logger.info("Scripture: Using most recent reflection from Catholic Daily Reflections RSS")
        except Exception as e:
            self.logger.error("Scripture: Failed to fetch Catholic Daily Reflections RSS: %s", e)

        # Note: Previously used AI search fallback, now relies solely on RSS feeds

        self.logger.info("Scripture: Total %d items (USCCB + Reflections)", len(all_items))
        return FetchResult("combined", Section.SCRIPTURE, all_items, asyncio.get_event_loop().time() - start)

    async def _determine_fetch_volume(self) -> int:
        """
        Dynamically determine how many papers to fetch based on cache size.
        More cached items = fetch more papers to maintain quality.
        """
        try:
            cache_count = await self.cache_service.get_active_cache_count('research_papers')

            if cache_count > 100:
                fetch_volume = 60  # Double the normal amount
                self.logger.info(f"Cache has {cache_count} papers, fetching {fetch_volume} new papers")
            elif cache_count > 50:
                fetch_volume = 45  # 1.5x normal
                self.logger.info(f"Cache has {cache_count} papers, fetching {fetch_volume} new papers")
            else:
                fetch_volume = 30  # Normal amount
                self.logger.debug(f"Cache has {cache_count} papers, fetching {fetch_volume} new papers")

            return fetch_volume
        except Exception as e:
            self.logger.warning(f"Could not determine cache size: {e}, using default fetch volume")
            return 30  # Default

    async def _fetch_research_papers(self) -> FetchResult:
        """
        Fetch research papers using Exa Websets API.
        Falls back to ArXiv/Semantic Scholar if Exa fails.
        """
        start = asyncio.get_event_loop().time()

        try:
            # Try Exa first (primary source)
            adapter_name = type(self.news_adapter.adapter).__name__ if hasattr(self.news_adapter, 'adapter') else type(self.news_adapter).__name__
            self.logger.info(f"Fetching research papers using {adapter_name}")

            result = await self.news_adapter.search_optimized_rate_limited("research_papers")
            articles = result.articles

            if len(articles) >= 5:  # Minimum acceptable count
                self.logger.info(f"Research papers: {len(articles)} items from Exa")
                return FetchResult("exa", Section.RESEARCH_PAPERS, articles, asyncio.get_event_loop().time() - start)
            else:
                self.logger.warning(f"Research papers: Only {len(articles)} items from Exa, trying fallback")

        except Exception as e:
            self.logger.warning(f"Research papers Exa fetch failed: {e}, trying fallback")

        # Fallback to ArXiv/Semantic Scholar
        self.logger.info("Research papers: Using ArXiv/Semantic Scholar fallback")
        if self.semantic_scholar:
            return await self._fetch_hybrid_papers()
        else:
            return await self._fetch_arxiv_papers()

    async def _fetch_hybrid_papers(self) -> FetchResult:
        """
        Fetch papers from both ArXiv and Semantic Scholar with smart orchestration.
        60% ArXiv (novelty) + 40% Semantic Scholar (authority).
        Also normalizes fields so downstream ranking/selection always sees
        headline, summary_text, and published.
        """
        start = asyncio.get_event_loop().time()

        # Determine total fetch volume
        fetch_volume = await self._determine_fetch_volume()

        # Check if Semantic Scholar is available
        if not self.semantic_scholar or not getattr(self.semantic_scholar, 'enabled', False):
            self.logger.warning("Semantic Scholar unavailable, falling back to ArXiv only")
            return await self._fetch_arxiv_papers()

        arxiv_count = int(fetch_volume * 0.6)  # 60% from ArXiv
        semantic_count = int(fetch_volume * 0.4)  # 40% from Semantic Scholar

        self.logger.info(f"Fetching {arxiv_count} from ArXiv, {semantic_count} from Semantic Scholar")

        # Use empty query for citation velocity mode - gets ALL trending papers
        # This will fetch papers sorted by how fast they're gaining citations
        search_queries = [""]  # Empty query triggers velocity-based discovery

        # Fetch from both sources in parallel
        tasks = [
            self._fetch_arxiv_subset(arxiv_count),
            self.semantic_scholar.search_papers(  # This is already an async function
                queries=search_queries,
                max_results=semantic_count,
                min_citations=5,
                days_back=30
            )
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Ensure aiohttp session is properly closed to avoid warnings
            try:
                if self.semantic_scholar:
                    await self.semantic_scholar.close_session()
            except Exception as e:
                self.logger.debug(f"Semantic Scholar session close error (ignored): {e}")

        all_papers: List[Dict[str, Any]] = []

        # Process ArXiv results (already normalized by _fetch_arxiv_papers)
        if isinstance(results[0], list):
            all_papers.extend(results[0])
            self.logger.info(f"Got {len(results[0])} papers from ArXiv")
        else:
            self.logger.warning(f"ArXiv fetch failed: {results[0]}")

        # Process Semantic Scholar results (normalize to common shape)
        if isinstance(results[1], list):
            normalized_s2: List[Dict[str, Any]] = []
            for p in results[1]:
                title = p.get('title') or ''
                abstract = p.get('tldr') or p.get('abstract') or ''
                # publicationDate preferred; fallback to Jan 1 of year; else None
                pub_raw = p.get('publicationDate') or None
                if not pub_raw and p.get('year'):
                    try:
                        pub_raw = f"{int(p['year']):04d}-01-01"
                    except Exception:
                        pub_raw = None
                normalized_s2.append({
                    'headline': title,
                    'title': title,
                    'summary_text': abstract,
                    'abstract': abstract,
                    'url': (p.get('openAccessPdf', {}) or {}).get('url') or p.get('url'),
                    'source': p.get('venue') or 'Semantic Scholar',
                    'published': pub_raw,
                })
            all_papers.extend(normalized_s2)
            self.logger.info(f"Got {len(normalized_s2)} papers from Semantic Scholar (normalized)")
        else:
            self.logger.warning(f"Semantic Scholar fetch failed: {results[1]}")

        # Deduplicate by title similarity (some papers may be on both platforms)
        seen_titles: set = set()
        unique_papers: List[Dict[str, Any]] = []

        for paper in all_papers:
            # Normalize title for comparison
            title = (paper.get('headline') or paper.get('title') or '').lower().strip()
            title_key = ''.join(c for c in title if c.isalnum())[:50]  # First 50 alphanumeric chars

            if title_key and title_key not in seen_titles:
                unique_papers.append(paper)
                seen_titles.add(title_key)

        self.logger.info(f"Total {len(unique_papers)} unique papers after deduplication")

        return FetchResult(
            "hybrid",
            Section.RESEARCH_PAPERS,
            unique_papers[:fetch_volume],  # Limit to requested volume
            asyncio.get_event_loop().time() - start
        )

    async def _fetch_arxiv_subset(self, max_results: int) -> List[Dict[str, Any]]:
        """
        Fetch a subset of ArXiv papers for hybrid mode.
        """
        # Use existing ArXiv fetching logic but with limited results
        result = await self._fetch_arxiv_papers()
        if result.items:
            return result.items[:max_results]
        return []

    async def _fetch_arxiv_papers(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            items: List[Dict[str, Any]]
            if hasattr(self.arxiv, "fetch_latest_papers"):
                papers_dicts: List[Dict[str, Any]] = await getattr(self.arxiv, "fetch_latest_papers")()
                items = [
                    {
                        "title": p.get("title"),
                        "headline": p.get("title"),  # Add headline for consistency
                        "url": p.get("url"),
                        "source_url": p.get("url"),  # Add source_url for email template
                        "abstract": p.get("abstract"),
                        "summary_text": p.get("abstract"),  # Add summary_text field
                        "source": "arXiv",  # Add source field
                        "authors": p.get("authors", []),
                        "published": p.get("published"),
                        "categories": p.get("categories", []),
                    }
                    for p in papers_dicts
                ]
            else:
                # Determine how many papers to fetch based on cache size
                fetch_volume = await self._determine_fetch_volume()

                # Use sophisticated ArxivService methods for intelligent paper selection
                # Per VISION.txt: AI, CS, physics, electrical engineering, probability, statistics, math

                # Get a diverse mix of papers using multiple strategies
                tasks = []

                # Adjust fetching strategy based on volume
                breakthrough_count = min(10, fetch_volume // 6)  # ~16% of total
                interdisciplinary_count = min(8, fetch_volume // 7)  # ~14% of total
                latest_count = max(20, fetch_volume // 2)  # ~50% of total

                # 1. Get breakthrough candidates (papers accepted to top venues)
                tasks.append(self.arxiv.search_breakthrough())

                # 2. Get interdisciplinary papers (Renaissance breadth)
                tasks.append(self.arxiv.search_interdisciplinary())

                # 3. Get latest from core interests (AI, physics, math/stats)
                core_categories = [
                    "cs.AI", "cs.LG", "cs.CL",  # AI/ML
                    "stat.ML", "stat.TH", "math.PR", "math.ST",  # Probability/Statistics
                    "physics.app-ph", "eess.SP", "eess.SY",  # Applied Physics/EE
                    "cs.CR", "cs.IT",  # Information theory/Crypto
                    "quant-ph", "cond-mat.stat-mech",  # Quantum/Statistical mechanics
                    "q-fin.CP", "q-fin.PM", "q-fin.RM", "q-fin.ST",  # Quantitative Finance
                    "econ.TH", "econ.EM"  # Economics Theory & Econometrics
                ]
                # Increased from 3 to 7 days for better date diversity
                # This prevents all papers from clustering around "3 days ago"
                tasks.append(self.arxiv.search_latest(core_categories, max_results=latest_count, days_back=7))

                # 4. Get today's highlights (curated by intellectual delight score)
                tasks.append(self.arxiv.get_todays_highlights())

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Combine all papers and deduplicate
                all_papers: List[ArxivPaper] = []
                seen_ids = set()

                # Process breakthrough papers (highest priority)
                if isinstance(results[0], list):
                    for p in results[0][:breakthrough_count]:
                        if p.arxiv_id not in seen_ids:
                            all_papers.append(p)
                            seen_ids.add(p.arxiv_id)

                # Process interdisciplinary papers
                if isinstance(results[1], list):
                    for p in results[1][:interdisciplinary_count]:
                        if p.arxiv_id not in seen_ids:
                            all_papers.append(p)
                            seen_ids.add(p.arxiv_id)

                # Process latest papers from core categories
                if isinstance(results[2], list):
                    for p in results[2][:latest_count]:
                        if p.arxiv_id not in seen_ids:
                            all_papers.append(p)
                            seen_ids.add(p.arxiv_id)

                # Process today's highlights
                if isinstance(results[3], dict):
                    # Prioritize certain categories based on VISION.txt interests
                    priority_categories = ["breaking", "fundamental", "emerging", "breakthrough"]
                    for cat in priority_categories:
                        if cat in results[3]:
                            for p in results[3][cat][:4]:  # Top 2 from each highlight category
                                if p.arxiv_id not in seen_ids and len(all_papers) < 15:
                                    all_papers.append(p)
                                    seen_ids.add(p.arxiv_id)

                # Calculate recency-weighted scores for intelligent ranking
                # Papers should be mostly recent, but allow for some older gems
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc)

                def calculate_paper_score(paper):
                    # Base score from breakthrough status
                    score = 2.0 if self.arxiv._is_breakthrough_candidate(paper) else 1.0

                    # Recency bias: exponential decay over days
                    # Papers from today get full score, older papers decay
                    days_old = (now - paper.published_date).days

                    # Intelligent recency calculation per VISION.txt
                    # "very fresh research with the occasional seminal older paper"
                    if days_old == 0:
                        recency_multiplier = 1.5  # Today's papers get a boost
                    elif days_old <= 2:
                        recency_multiplier = 1.2  # Last 2 days still very relevant
                    elif days_old <= 7:
                        recency_multiplier = 1.0  # Past week is standard
                    elif days_old <= 14:
                        recency_multiplier = 0.8  # 1-2 weeks old, slight penalty
                    elif days_old <= 30:
                        recency_multiplier = 0.6  # Month old, needs to be special
                    else:
                        # Older papers need to be exceptional (e.g., accepted to Nature)
                        # But still allow them if they're truly breakthrough
                        recency_multiplier = 0.4 if self.arxiv._is_breakthrough_candidate(paper) else 0.2

                    # No venue detection - we focus on paper quality, not venue prestige

                    # Bonus for interdisciplinary work (multiple categories)
                    if len(paper.categories) > 2:
                        score *= 1.2

                    # Apply recency multiplier
                    score *= recency_multiplier

                    # Add small random factor to prevent identical scores
                    import random
                    score += random.random() * 0.01

                    return score

                # Sort papers by calculated score
                all_papers.sort(key=calculate_paper_score, reverse=True)

                # Quality assurance: Check if today's papers are weak
                if all_papers:
                    top_scores = [calculate_paper_score(p) for p in all_papers[:5]]
                    avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0

                    # If quality is low, consider bringing back expired high-quality papers
                    if avg_top_score < 2.0:  # Score below 2.0 indicates weak papers
                        self.logger.warning(f"Today's papers are weak (avg score: {avg_top_score:.2f}), checking for expired high-quality papers")
                        try:
                            expired_great_papers = await self.cache_service.get_high_quality_expired_papers('research_papers', min_score=3.0)
                            if expired_great_papers:
                                self.logger.info(f"Found {len(expired_great_papers)} high-quality expired papers that could be reconsidered")
                                # TODO: Re-fetch full content for these papers from ArXiv
                        except Exception as e:
                            self.logger.warning(f"Could not check for expired papers: {e}")

                # Check if it's the first Monday of the month for "Greatest Hits"
                now = datetime.now(timezone.utc)
                if now.day <= 7 and now.weekday() == 0:  # First Monday
                    self.logger.info("First Monday of the month - including Greatest Hits!")
                    try:
                        greatest_hits = await self.cache_service.get_greatest_hits(days_back=30, top_n=5)
                        if greatest_hits:
                            self.logger.info(f"Found {len(greatest_hits)} greatest hits from the past month")
                            # TODO: Re-fetch full content for greatest hits from ArXiv
                    except Exception as e:
                        self.logger.warning(f"Could not get greatest hits: {e}")

                # Check if we need fallback to older uncached papers
                should_fallback = await self.cache_service.should_enable_fallback_mode('research_papers')
                if should_fallback:
                    self.logger.info("Enabling fallback mode for older uncached papers")
                    try:
                        uncached_urls = await self.cache_service.get_uncached_urls_by_timeframe(
                            'research_papers', months_back=3, limit=20
                        )
                        if uncached_urls:
                            self.logger.info(f"Found {len(uncached_urls)} uncached papers from past 3 months")
                            # TODO: Re-fetch papers from uncached URLs via ArXiv service
                            # For now, just log that we would do this
                        else:
                            self.logger.info("No uncached papers found in fallback search")
                    except Exception as e:
                        self.logger.warning(f"Could not get uncached papers for fallback: {e}")

                # Take top papers based on fetch volume
                selected_papers = all_papers[:min(fetch_volume, len(all_papers))]

                self.logger.info(f"ArXiv: Selected {len(selected_papers)} papers from {len(all_papers)} candidates")
                self.logger.info(f"ArXiv: Breakthroughs: {sum(1 for p in selected_papers if self.arxiv._is_breakthrough_candidate(p))}")

                items = [
                    {
                        "title": p.title,
                        "headline": p.title,
                        "url": p.pdf_url,
                        "source_url": p.pdf_url,
                        "abstract": p.abstract,
                        "summary_text": p.abstract,
                        "source": f"arXiv ({', '.join(p.categories[:2])})",  # Include categories in source
                        "authors": p.authors,
                        "published": p.published_date.isoformat(),
                        "categories": p.categories,
                        "journal_ref": p.journal_ref,  # Include if accepted to journal
                        "comment": p.comment,  # Include author comments (e.g., "Accepted to NeurIPS")
                    }
                    for p in selected_papers
                ]
            self.logger.info(f"ArXiv: Returning {len(items)} research papers for ranking")
            return FetchResult("arxiv", Section.RESEARCH_PAPERS, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            # arXiv non-critical; return error result
            self.logger.error(f"ArXiv fetch failed: {e}")
            return FetchResult("arxiv", Section.RESEARCH_PAPERS, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_rss_feeds(self) -> List[FetchResult]:
        start = asyncio.get_event_loop().time()
        try:
            # Prefer a dedicated method if available in tests
            if hasattr(self.rss, "fetch_usccb_readings"):
                data = await getattr(self.rss, "fetch_usccb_readings")()
                self.logger.info("Scripture: Fetched %d items from test method", len(data) if isinstance(data, list) else 1)
                return [
                    FetchResult(
                        source="rss",
                        section=Section.SCRIPTURE,
                        items=data,
                        fetch_time=asyncio.get_event_loop().time() - start,
                    )
                ]

            # Real RSSService path: get today's readings
            self.logger.info("Scripture: Fetching daily readings from RSS service")
            content = await self.rss.get_todays_spiritual_content()
            readings: DailyReading = content.get("readings")  # type: ignore

            # Transform DailyReading into proper list of content items
            readings_items = []
            if readings:
                self.logger.info("Scripture: Found daily readings for %s", readings.date if hasattr(readings, 'date') else 'today')
                base_url = "https://bible.usccb.org/daily-bible-reading"
                published = readings.date.isoformat() if readings.date else datetime.now().isoformat()

                # Add First Reading as an item
                if readings.first_reading and readings.first_reading.get('text'):
                    readings_items.append({
                        "headline": f"First Reading: {readings.first_reading.get('reference', 'Daily Reading')}",
                        "url": base_url,
                        "summary_text": readings.first_reading.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })

                # Add Responsorial Psalm if present
                if hasattr(readings, 'responsorial_psalm') and readings.responsorial_psalm and readings.responsorial_psalm.get('text'):
                    readings_items.append({
                        "headline": f"Responsorial Psalm: {readings.responsorial_psalm.get('reference', 'Psalm')}",
                        "url": base_url,
                        "summary_text": readings.responsorial_psalm.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })

                # Add Gospel as an item
                if readings.gospel and readings.gospel.get('text'):
                    readings_items.append({
                        "headline": f"Gospel: {readings.gospel.get('reference', 'Daily Gospel')}",
                        "url": base_url,
                        "summary_text": readings.gospel.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })

                # Add Reflection if present
                if readings.reflection:
                    readings_items.append({
                        "headline": "Daily Reflection",
                        "url": base_url,
                        "summary_text": readings.reflection,
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })

                # Add Saint of the Day if present
                if hasattr(readings, 'saint_of_day') and readings.saint_of_day:
                    readings_items.append({
                        "headline": "Saint of the Day",
                        "url": base_url,
                        "summary_text": readings.saint_of_day,
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })

            return [
                FetchResult(
                    source="rss",
                    section=Section.SCRIPTURE,
                    items=readings_items,  # Now returning a proper list
                    fetch_time=asyncio.get_event_loop().time() - start,
                )
            ]
        except Exception as e:  # noqa: BLE001
            return [FetchResult("rss", Section.SCRIPTURE, [], asyncio.get_event_loop().time() - start, error=str(e))]

    async def rank_all_content(self, sections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[RankedItem]]:
        ranked: Dict[str, List[RankedItem]] = {}
        tasks: List[asyncio.Task] = []
        failed_sections: List[str] = []

        async def rank_one(section: str, items: List[Dict[str, Any]]):
            try:
                # Scripture section doesn't need ranking - just pass through
                if section == Section.SCRIPTURE:
                    ranked[section] = self._convert_to_ranked_items(items, section)
                else:
                    self.logger.info(f"📊 Ranking {len(items)} items for section '{section}'")
                    ranked[section] = await self._rank_items(items, section)
                    self.logger.info(
                        f"✅ Ranked {len(ranked[section])} items for '{section}' "
                        f"(scores: {ranked[section][0].total_score:.2f} - {ranked[section][-1].total_score:.2f})"
                        if ranked[section] else f"⚠️ No items ranked for '{section}'"
                    )
            except Exception as e:
                # Log the exception instead of silently swallowing it
                self.logger.error(
                    f"🚨 RANKING FAILED for section '{section}': {type(e).__name__}: {e}",
                    exc_info=True
                )
                failed_sections.append(section)

        for section, items in sections.items():
            tasks.append(asyncio.create_task(rank_one(section, items)))

        # Wait for all tasks to complete (exceptions are now handled inside rank_one)
        await asyncio.gather(*tasks, return_exceptions=False)

        # Retry failed sections with simplified ranking (fallback strategy)
        if failed_sections:
            self.logger.warning(
                f"⚠️ {len(failed_sections)} section(s) failed ranking: {failed_sections}. "
                f"Attempting fallback ranking strategy..."
            )

            for section in failed_sections:
                items = sections.get(section, [])
                if not items:
                    self.logger.error(f"🚨 Section '{section}' has no items to retry ranking")
                    continue

                try:
                    # Fallback: Use simple score-based ranking without AI
                    self.logger.info(f"🔄 Retry 1: Simple ranking for '{section}' ({len(items)} items)")
                    ranked[section] = self._fallback_simple_ranking(items, section)
                    self.logger.info(
                        f"✅ Fallback ranking succeeded for '{section}': {len(ranked[section])} items ranked"
                    )
                except Exception as e:
                    self.logger.error(
                        f"🚨 Fallback ranking FAILED for '{section}': {type(e).__name__}: {e}",
                        exc_info=True
                    )
                    # Last resort: Convert items to RankedItems with default scores
                    try:
                        self.logger.warning(f"🔄 Retry 2: Emergency conversion for '{section}'")
                        ranked[section] = self._emergency_ranking(items, section)
                        self.logger.warning(
                            f"⚠️ Emergency ranking used for '{section}': {len(ranked[section])} items with default scores"
                        )
                    except Exception as final_error:
                        self.logger.critical(
                            f"💀 CRITICAL: All ranking strategies failed for '{section}': {final_error}. "
                            f"Section will be MISSING from newsletter!"
                        )

        # After ranking, perform cross-section theme analysis if cache service is available
        if self.cache_service and hasattr(self.cache_service, 'detect_cross_section_themes'):
            try:
                # Convert ranked items back to dict format for theme analysis
                theme_analysis_input = {}
                for section, ranked_items in ranked.items():
                    if section != Section.SCRIPTURE:  # Skip scripture for theme analysis
                        theme_analysis_input[section] = [
                            {
                                'headline': item.headline,
                                'summary_text': item.summary_text,
                                'source': item.source,
                                'url': item.url,
                                'id': item.id
                            }
                            for item in ranked_items
                        ]

                if theme_analysis_input:
                    # Get embedding service from the aggregator (we'll need to pass it)
                    embeddings = getattr(self, 'embeddings', None)
                    if embeddings:
                        theme_analysis = await self.cache_service.detect_cross_section_themes(
                            theme_analysis_input, embeddings, days_back=7
                        )

                        # Store theme analysis for later use in synthesis
                        self._last_theme_analysis = theme_analysis

                        # Log key insights
                        if theme_analysis.get('cross_section_themes'):
                            self.logger.info(f"Detected {len(theme_analysis['cross_section_themes'])} cross-section themes")

                        if theme_analysis.get('theme_overlap_warnings'):
                            self.logger.warning(f"Found {len(theme_analysis['theme_overlap_warnings'])} theme overlap warnings")

                        if theme_analysis.get('diversity_recommendations'):
                            self.logger.info(f"Generated {len(theme_analysis['diversity_recommendations'])} diversity recommendations")

            except Exception as e:
                self.logger.warning(f"Cross-section theme analysis failed: {e}")
                self._last_theme_analysis = None

        return ranked

    def get_theme_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest cross-section theme analysis results.

        Returns:
            Dict containing theme analysis results, or None if not available
        """
        return getattr(self, '_last_theme_analysis', None)

    def _convert_to_ranked_items(self, items: List[Dict[str, Any]], section: str) -> List[RankedItem]:
        """Convert items directly to RankedItems without scoring (for Catholic section)."""
        ranked_items: List[RankedItem] = []
        for idx, item in enumerate(items):
            try:
                import hashlib
                # Generate a unique ID for the item
                content_hash = hashlib.md5((item.get("headline", "") + item.get("url", "")).encode()).hexdigest()[:8]
                item_id = f"{section}_{idx}_{content_hash}"

                # Parse published date if present
                published_date = None
                if item.get("published"):
                    from datetime import datetime
                    try:
                        published_date = datetime.fromisoformat(item["published"].replace("Z", "+00:00"))
                    except:
                        published_date = datetime.now()
                else:
                    from datetime import datetime
                    published_date = datetime.now()

                ranked_item = RankedItem(
                    id=item_id,
                    headline=item.get("headline", "Daily Reading"),
                    url=item.get("url", ""),
                    source=item.get("source", "Catholic Church"),
                    summary_text=item.get("summary_text", ""),
                    section=section,
                    published_date=published_date,
                    # All 7 Renaissance scores - Catholic content gets perfect scores
                    temporal_impact=10.0,
                    intellectual_novelty=10.0,
                    renaissance_breadth=10.0,
                    actionable_wisdom=10.0,
                    source_authority=10.0,
                    signal_clarity=10.0,
                    transformative_potential=10.0,
                    total_score=10.0,  # Perfect weighted total
                    preserve_original=item.get("preserve_original", False),  # Pass through the flag
                    editorial_note="Daily spiritual guidance"
                )
                ranked_items.append(ranked_item)
            except Exception as e:
                self.logger.error(f"Failed to convert Scripture item: {e}")
                continue

        return ranked_items

    def _validate_sources(self, items: List[Dict[str, Any]], allowed_domains: List[str], section_name: str) -> List[Dict[str, Any]]:
        """
        Enrich items with source authority scores for AI ranking.
        No longer filters - just adds source scores that feed into the AI ranking.
        """
        # Load source authority config if available
        if not self.source_ranking_config:
            # Add default source authority score if no config
            for item in items:
                item['source_authority'] = 5.0  # Default middle score
            return items

        # Build source score map from tier configuration
        tier_scores = {}
        blacklist_sources = set()

        for tier_name, tier_data in self.source_ranking_config.items():
            if tier_name == 'config':
                continue
            if tier_name == 'blacklist':
                # Only keep blacklist for truly problematic sources
                blacklist_sources.update(tier_data.get('sources', []))
                continue

            sources = tier_data.get('sources', [])
            score = tier_data.get('score', 5)
            for source in sources:
                tier_scores[source.lower().replace('www.', '')] = float(score)

        enriched_items = []
        blacklisted_count = 0

        for item in items:
            # Extract source domain from URL or source field
            source_domain = ""
            if 'url' in item and item['url']:
                from urllib.parse import urlparse
                try:
                    parsed = urlparse(item['url'])
                    source_domain = parsed.netloc.lower().replace('www.', '')
                except:
                    pass

            # Also check the source field directly
            if not source_domain and 'source' in item and item['source']:
                source_domain = item['source'].lower().replace('www.', '').strip()

            # Only filter out truly blacklisted sources
            is_blacklisted = False
            for black_source in blacklist_sources:
                if black_source.lower() in source_domain or source_domain in black_source.lower():
                    is_blacklisted = True
                    blacklisted_count += 1
                    self.logger.debug(f"Skipping blacklisted source '{source_domain}'")
                    break

            if is_blacklisted:
                continue

            # Find tier score for this source or use default
            source_score = 5.0  # Default middle score for unknown sources
            for tier_source, score in tier_scores.items():
                if tier_source in source_domain or source_domain in tier_source:
                    source_score = score
                    break

            # Add source authority score to item for AI ranking
            item['source_authority'] = source_score
            enriched_items.append(item)

        if blacklisted_count > 0:
            self.logger.info(f"{section_name}: Filtered {blacklisted_count} blacklisted sources")

        self.logger.info(f"{section_name}: Enriched {len(enriched_items)} items with source authority scores")
        return enriched_items

    def _apply_multi_stage_pipeline(
        self,
        items: List[Dict[str, Any]],
        section: str,
        max_age_days: int,
        min_items: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Multi-stage content pipeline with quality assurance.

        Stage 1: Basic filtering (dates, bad domains)
        Stage 2: Source ranking and scoring
        Stage 3: Quality filtering by score
        Stage 4: Diversity limits
        Stage 5: Fallback if needed
        """
        # Stage 1: Adaptive filtering based on content availability
        filtered = self._filter_items_adaptive(items, max_age_days, min_items, section)
        self.logger.info(f"{section}: Stage 1 - {len(items)} -> {len(filtered)} after adaptive filtering")

        # Stage 2: Apply source authority calibration (down-rank marginal outlets early)
        quality_adjustment = self._calculate_quality_adjustment(len(filtered), min_items)
        self.logger.info(f"{section}: Content availability adjustment: {quality_adjustment:.2f} (lower is more lenient)")

        # Score and sort by source quality without dropping items (down-rank only)
        try:
            from src.models.content import NewsItem
            # Preserve original dictionaries with all fields
            orig_items = list(filtered)
            orig_by_key: Dict[str, Dict[str, Any]] = {}
            news_objects: List[Tuple[str, NewsItem]] = []

            for idx, it in enumerate(orig_items):
                key = it.get("url") or f"idx:{idx}"
                orig_by_key[key] = it

                # Parse published date from common fields
                published_date = None
                pub_raw = it.get("published_date") or it.get("published") or it.get("date")
                try:
                    if isinstance(pub_raw, str):
                        from datetime import datetime
                        import dateutil.parser
                        published_date = dateutil.parser.parse(pub_raw)
                    elif isinstance(pub_raw, datetime):
                        published_date = pub_raw
                except Exception:
                    published_date = None

                headline = it.get("headline") or it.get("title") or ""
                summary_text = (
                    it.get("summary_text")
                    or it.get("enrichment_summary")
                    or it.get("abstract")
                    or it.get("description")
                    or ""
                )

                news_objects.append((
                    key,
                    NewsItem(
                        headline=headline,
                        summary_text=summary_text,
                        url=it.get("url", ""),
                        source=it.get("source", ""),
                        published_date=published_date
                    )
                ))

            scored = self.source_ranker.score_and_rank_items([ni for (_k, ni) in news_objects])

            # Reorder originals according to ranked NewsItems; do not drop any
            ranked_items: List[Dict[str, Any]] = []
            used_keys = set()
            for (ni, _score) in scored:
                k = ni.url or None
                if k and k in orig_by_key and k not in used_keys:
                    ranked_items.append(orig_by_key[k])
                    used_keys.add(k)

            # Append any not scored (e.g., unknown/blacklisted) to preserve completeness
            for (k, _ni) in news_objects:
                if k not in used_keys:
                    ranked_items.append(orig_by_key[k])

            self.logger.info(f"{section}: Stage 2 - ranked {len(ranked_items)} items by source authority")
        except Exception as e:
            self.logger.warning(f"{section}: Source authority ranking failed: {e}; passing through unranked")
            ranked_items = list(filtered)

        # Convert back to dict format (preserve all fields, plus ensure content/published_date)
        result: List[Dict[str, Any]] = []
        for it in ranked_items:
            item = dict(it)  # shallow copy
            # Ensure 'content' field for downstream AI steps and tests
            if not item.get("content"):
                item["content"] = (
                    item.get("enrichment_summary")
                    or item.get("abstract")
                    or item.get("summary_text")
                    or item.get("description")
                    or ""
                )
            # Ensure published_date is present for tests
            if not item.get("published_date"):
                pub_raw = item.get("published") or item.get("date")
                if pub_raw:
                    try:
                        from datetime import datetime
                        import dateutil.parser
                        dt = dateutil.parser.parse(pub_raw) if isinstance(pub_raw, str) else pub_raw
                        if isinstance(dt, datetime):
                            item["published_date"] = dt.isoformat()
                    except Exception:
                        # leave as-is if parsing fails
                        pass
            result.append(item)

        # Stage 5: Fallback if insufficient items
        if len(result) < min_items:
            self.logger.warning(f"{section}: Only {len(result)} items after pipeline, need {min_items}")
            # Return what we have - the fetch method will handle retry/fallback

        return result

    def _filter_items_adaptive(self, items: List[Dict[str, Any]], max_age_days: int, target_count: int = 3, section: str = "general") -> List[Dict[str, Any]]:
        """
        Adaptive filtering that adjusts standards based on content availability.
        Uses progressive relaxation to ensure newsletter completeness while maintaining quality.
        """
        from datetime import datetime, timedelta
        import re

        self.logger.info(f"📋 Adaptive filtering {len(items)} items for {section} (target: {target_count}, max_age: {max_age_days} days)")

        # First pass: Apply strict filtering
        strict_filtered = self._filter_items(items, max_age_days)
        if len(items) > 0:
            self.logger.info(f"📊 Strict filtering: {len(items)} → {len(strict_filtered)} ({len(strict_filtered)/len(items)*100:.1f}% kept)")
        else:
            self.logger.info(f"📊 Strict filtering: 0 → {len(strict_filtered)} (no items to filter)")

        # If we have enough high-quality articles, return them
        if len(strict_filtered) >= target_count:
            self.logger.info(f"✅ Target reached with strict filtering: {len(strict_filtered)} articles")
            return strict_filtered

        # Progressive relaxation if we don't have enough articles
        self.logger.warning(f"⚠️ Only {len(strict_filtered)} articles after strict filtering, applying adaptive relaxation")

        # Relaxation strategy 1: Slightly extend temporal window for this section
        if max_age_days < 7:
            relaxed_age = min(max_age_days + 2, 7)  # Add 2 days, cap at 7
            self.logger.info(f"🔄 Relaxation 1: Extending age limit from {max_age_days} to {relaxed_age} days")
            relaxed_filtered = self._filter_items(items, relaxed_age)

            if len(relaxed_filtered) >= target_count:
                self.logger.info(f"✅ Target reached with relaxed age filtering: {len(relaxed_filtered)} articles")
                return relaxed_filtered

        # Relaxation strategy 2: Less aggressive domain filtering
        self.logger.info("🔄 Relaxation 2: Applying lenient domain filtering")
        lenient_filtered = self._filter_items_lenient(items, max_age_days)

        if len(lenient_filtered) >= target_count:
            self.logger.info(f"✅ Target reached with lenient filtering: {len(lenient_filtered)} articles")
            return lenient_filtered

        # Relaxation strategy 3: Accept articles without dates if they're high quality
        if section in ["miscellaneous", "tech_science", "research_papers", "breaking_news", "business"]:
            self.logger.info("🔄 Relaxation 3: Including articles without dates (with source authority checks)")
            no_date_filtered = self._filter_items_allow_no_date(items, max_age_days, section)

            if len(no_date_filtered) >= target_count:
                self.logger.info(f"✅ Target reached with no-date tolerance: {len(no_date_filtered)} articles")
                return no_date_filtered

        # Final fallback: Return what we have, sorted by quality indicators
        final_articles = lenient_filtered or strict_filtered or []
        final_articles = self._sort_by_quality_indicators(final_articles)

        self.logger.warning(f"⚠️ Adaptive filtering complete: {len(final_articles)} articles (target was {target_count})")
        return final_articles

    def _filter_items(self, items: List[Dict[str, Any]], max_age_days: int) -> List[Dict[str, Any]]:
        """
        Quality and freshness filter for AI-derived items.
        Drops:
        - Non-English content (CRITICAL: English-only requirement)
        - Very old items beyond max_age_days
        - Anything older than 30 days regardless of section
        - Anything from before 2024
        - Items without valid dates
        - Aggregator/generic domains (YouTube, Wikipedia, Eventbrite, ScienceDaily, TS2.tech)
        - Non-informative clickbait/generic titles
        - Missing headline or url
        """
        from datetime import datetime, timedelta
        import re
        cutoff = datetime.now() - timedelta(days=max_age_days)
        # Hard cutoff: nothing older than 30 days
        hard_cutoff = datetime.now() - timedelta(days=30)
        bad_domains = {
            # Social media and aggregators (unreliable for authoritative news)
            "youtube.com", "www.youtube.com",
            "reddit.com", "www.reddit.com",
            "twitter.com", "x.com",
            "facebook.com", "www.facebook.com",
            "instagram.com", "www.instagram.com",
            "tiktok.com", "www.tiktok.com",
            # Wikipedia - not a primary source
            "en.wikipedia.org", "wikipedia.org",
            # Event platforms
            "www.eventbrite.com", "eventbrite.com",
            # Known unreliable or low-quality
            "ts2.tech", "www.ts2.tech",
            "buzzfeed.com", "www.buzzfeed.com",
            "medium.com",  # User-generated content
            "substack.com",  # Unless specific trusted authors

            # Foreign/Non-English news sources
            "thehindu.com", "www.thehindu.com",
            "hindustantimes.com", "www.hindustantimes.com",
            "timesofindia.com", "www.timesofindia.com",
            "indianexpress.com", "www.indianexpress.com",
            "asahi.com", "www.asahi.com",
            "chinadaily.com", "www.chinadaily.com.cn",
            "xinhua.net", "xinhuanet.com",
            "scmp.com", "www.scmp.com",  # South China Morning Post
            "japantimes.co.jp", "www.japantimes.co.jp",
            "koreaherald.com", "www.koreaherald.com",
            "bangkokpost.com", "www.bangkokpost.com",

            # Low-quality US sources - only block the worst offenders
            "theintelligencer.net", "www.theintelligencer.net",
            "pressgazette.co.uk", "www.pressgazette.co.uk",

            # Financial spam/low-quality - only block the worst
            "ainvest.com", "www.ainvest.com",
            "markets.financialcontent.com",
            "seekingalpha.com", "www.seekingalpha.com",
            "fool.com", "www.fool.com",  # Motley Fool

            # Regional/African sources
            "cnbcafrica.com", "www.cnbcafrica.com",
            "allafrica.com", "www.allafrica.com",

            # Academic publishers (keep only those that are truly problematic)
        }
        def domain(u: str) -> str:
            try:
                from urllib.parse import urlparse
                return (urlparse(u).netloc or u).lower()
            except Exception:
                return ""
        def too_old(published: Optional[str]) -> bool:
            if not published:
                # No date = reject for strict filtering (this function is for strict filtering only)
                # The adaptive filtering will handle no-date articles separately
                self.logger.debug("❌ No publication date provided - rejecting in strict filter")
                return True  # Reject articles without dates in strict filtering
            try:
                # Try multiple date parsing approaches for better compatibility
                dt = None

                # Try ISO format first
                try:
                    dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except:
                    # Try common date formats
                    import dateutil.parser
                    try:
                        dt = dateutil.parser.parse(published)
                    except:
                        # If all parsing fails, be lenient and assume recent
                        self.logger.debug(f"⚠️ Could not parse date '{published}' - assuming recent for strict filter")
                        return False  # Be lenient with unparseable dates

                if dt is None:
                    return False  # Be lenient if we can't parse

                # Intelligent year check: reject if article is more than 2 years old
                two_years_ago = datetime.now().year - 2
                if dt.year < two_years_ago:
                    self.logger.debug(f"❌ Article from {dt.year} - more than 2 years old")
                    return True
                # Hard cutoff: nothing older than 30 days
                if dt < hard_cutoff:
                    self.logger.debug(f"❌ Article older than 30 days: {published}")
                    return True
                # Section-specific cutoff
                if dt < cutoff:
                    self.logger.debug(f"❌ Article older than {max_age_days} days: {published}")
                    return True
                return False
            except Exception as e:
                # If we can't parse the date, be lenient in strict filtering
                self.logger.debug(f"⚠️ Could not parse date '{published}': {e} - assuming recent")
                return False  # Be lenient with parsing errors

        def is_non_english(text: str) -> bool:
            """
            Detect if text is non-English using simple heuristics.
            Returns True if text appears to be non-English.
            """
            if not text or len(text.strip()) < 10:
                return False

            # Check for non-Latin scripts (CJK, Cyrillic, Arabic, etc.)
            non_latin_chars = 0
            total_chars = 0
            for char in text:
                if char.isalpha():
                    total_chars += 1
                    # Check for non-Latin Unicode ranges
                    code_point = ord(char)
                    # CJK: 0x4E00-0x9FFF, 0x3400-0x4DBF
                    # Cyrillic: 0x0400-0x04FF
                    # Arabic: 0x0600-0x06FF
                    # Thai: 0x0E00-0x0E7F
                    # Korean Hangul: 0xAC00-0xD7AF
                    if (0x4E00 <= code_point <= 0x9FFF or  # CJK Unified
                        0x3400 <= code_point <= 0x4DBF or  # CJK Extension A
                        0x0400 <= code_point <= 0x04FF or  # Cyrillic
                        0x0600 <= code_point <= 0x06FF or  # Arabic
                        0x0E00 <= code_point <= 0x0E7F or  # Thai
                        0xAC00 <= code_point <= 0xD7AF):   # Korean Hangul
                        non_latin_chars += 1

            # If more than 20% of alphabetic characters are non-Latin, it's likely non-English
            if total_chars > 0 and (non_latin_chars / total_chars) > 0.2:
                return True

            # Check for common non-English words/patterns
            text_lower = text.lower()
            non_english_indicators = [
                # Vietnamese
                'của', 'và', 'với', 'trong', 'cho', 'để', 'được', 'có', 'là', 'này',
                # Spanish
                'el ', 'la ', 'los ', 'las ', 'de ', 'del ', 'para ', 'por ', 'con ', 'en ',
                # French
                'le ', 'la ', 'les ', 'de ', 'du ', 'des ', 'pour ', 'avec ', 'dans ',
                # German
                'der ', 'die ', 'das ', 'den ', 'dem ', 'des ', 'für ', 'mit ', 'von ',
                # Portuguese
                'o ', 'a ', 'os ', 'as ', 'do ', 'da ', 'dos ', 'das ', 'para ', 'com ',
                # Italian
                'il ', 'lo ', 'la ', 'i ', 'gli ', 'le ', 'del ', 'della ', 'per ', 'con ',
            ]

            # Count matches
            matches = sum(1 for indicator in non_english_indicators if indicator in text_lower)
            # If we find 3+ non-English indicators, it's likely non-English
            if matches >= 3:
                return True

            return False

        def bad_title(title: str) -> bool:
            if not title:
                return True
            t = title.lower()
            # Filter out junk, clickbait, and non-news content
            patterns = [
                # Clickbait
                r"you won't believe",
                r"this one weird trick",
                r"doctors hate",
                r"^shocking:",
                r"will shock you",
                r"you'll never guess",
                # Astrology/Horoscope garbage
                r"tarot",
                r"horoscope",
                r"zodiac",
                r"astrology",
                r"your daily reading",
                r"star sign",
                r"mercury retrograde",
                # Low quality content
                r"^top \d+ ",  # "Top 10 ..." listicles
                r"^breaking news live",  # Live blogs not actual news
                r"^school assembly",  # Not news
                r"^in memoriam",  # Obituaries (unless specifically requested)
                # Entertainment/lifestyle fluff
                r"reality tv",
                r"celebrity",
                r"kardashian",
                r"bachelor",
                r"bachelorette",
            ]
            return any(re.search(p, t) for p in patterns)

        filtered: List[Dict[str, Any]] = []
        initial_count = len(items)
        self.logger.info(f"📋 Filtering {initial_count} items with max_age_days={max_age_days}")

        for it in items:
            url = it.get("url") or ""
            head = (it.get("headline") or "").strip()
            # FIX: Exa returns "date" or "published_date", not "published"
            published_raw = it.get("date") or it.get("published_date") or it.get("published")

            if not url or not head:
                self.logger.debug(f"❌ Filtered (missing URL/headline): {head[:50] if head else 'NO_HEADLINE'}")
                continue

            url_domain = domain(url)
            if url_domain in bad_domains:
                self.logger.debug(f"❌ Filtered (bad domain): {url_domain} - {head[:50]}")
                continue

            # Check for foreign TLDs
            if any(url_domain.endswith(tld) for tld in ['.in', '.cn', '.jp', '.kr', '.hk', '.tw', '.sg', '.my', '.th', '.id', '.africa']):
                self.logger.debug(f"❌ Filtered (foreign TLD): {url_domain} - {head[:50]}")
                continue

            # CRITICAL: Filter non-English content (English-only requirement)
            if is_non_english(head):
                self.logger.info(f"❌ Filtered (non-English): {head[:80]}")
                continue

            if too_old(published_raw):
                if not published_raw:
                    self.logger.debug(f"❌ Filtered (no date in breaking news): {head[:50]}")
                else:
                    self.logger.debug(f"❌ Filtered (too old): {published_raw} - {head[:50]}")
                continue

            if bad_title(head):
                self.logger.debug(f"❌ Filtered (bad title): {head[:50]}")
                continue

            self.logger.debug(f"✅ Kept: {head[:50]} from {url_domain}")
            filtered.append(it)

        final_count = len(filtered)
        filter_rate = (initial_count - final_count) / initial_count if initial_count > 0 else 0
        self.logger.info(f"📊 Filter results: {initial_count} → {final_count} ({filter_rate:.1%} filtered)")

        return filtered

    def _filter_items_lenient(self, items: List[Dict[str, Any]], max_age_days: int) -> List[Dict[str, Any]]:
        """Lenient filtering with relaxed domain restrictions"""
        from datetime import datetime, timedelta
        import re

        cutoff = datetime.now() - timedelta(days=max_age_days)
        hard_cutoff = datetime.now() - timedelta(days=30)

        # Reduced bad domains list - only block the worst offenders
        bad_domains = {
            "youtube.com", "www.youtube.com",
            "reddit.com", "www.reddit.com",
            "twitter.com", "x.com",
            "facebook.com", "www.facebook.com",
            "buzzfeed.com", "www.buzzfeed.com",
            "ts2.tech", "www.ts2.tech",
        }

        def domain(u: str) -> str:
            try:
                from urllib.parse import urlparse
                return (urlparse(u).netloc or u).lower()
            except Exception:
                return ""

        def too_old_lenient(published: Optional[str]) -> bool:
            if not published:
                return True  # Still reject articles without dates in lenient filter
            try:
                # Try multiple date parsing approaches
                dt = None
                try:
                    dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except:
                    import dateutil.parser
                    try:
                        dt = dateutil.parser.parse(published)
                    except:
                        # If parsing fails, be very lenient and assume recent
                        return False

                if dt is None:
                    return False  # Be lenient if we can't parse

                # Only hard cutoff applies (30 days)
                if dt < hard_cutoff:
                    return True
                return False
            except Exception:
                return False  # Be very lenient with parsing errors

        filtered = []
        for item in items:
            url = item.get("url") or ""
            headline = (item.get("headline") or "").strip()

            if not url or not headline:
                continue

            url_domain = domain(url)
            if url_domain in bad_domains:
                continue

            # FIX: Exa returns "date" or "published_date", not "published"
            if too_old_lenient(item.get("date") or item.get("published_date") or item.get("published")):
                continue

            filtered.append(item)

        return filtered

    def _filter_items_allow_no_date(self, items: List[Dict[str, Any]], max_age_days: int, section: str) -> List[Dict[str, Any]]:
        """Filter allowing articles without dates - much more lenient for section target fulfillment"""
        from datetime import datetime, timedelta
        import re

        cutoff = datetime.now() - timedelta(days=max_age_days)
        hard_cutoff = datetime.now() - timedelta(days=30)

        # Only block the worst offenders - be very lenient to meet section targets
        bad_domains = {
            "youtube.com", "www.youtube.com",
            "reddit.com", "www.reddit.com",
            "buzzfeed.com", "www.buzzfeed.com",
            "ts2.tech", "www.ts2.tech",  # Known spam
        }

        def domain(u: str) -> str:
            try:
                from urllib.parse import urlparse
                return (urlparse(u).netloc or u).lower()
            except Exception:
                return ""

        def is_acceptable_news_source(url_domain: str) -> bool:
            """Check if source is acceptable for news content - much broader than before"""
            # Block only obvious spam/low-quality domains
            spam_indicators = ["ts2.tech", "buzzfeed", "youtube", "reddit", "facebook", "twitter", "x.com"]
            if any(spam in url_domain for spam in spam_indicators):
                return False

            # Accept any domain that looks like a legitimate news source
            # This includes major news outlets, tech sites, business publications, etc.
            legitimate_indicators = [
                # Major news
                "reuters", "apnews", "bbc", "cnn", "npr", "pbs",
                # Business/Finance
                "bloomberg", "wsj", "ft.com", "cnbc", "marketwatch", "fortune",
                # Tech
                "techcrunch", "theverge", "arstechnica", "wired", "technologyreview",
                # Quality publications
                "theatlantic", "newyorker", "economist", "guardian", "nytimes",
                # Academic/Research
                "nature", "science", "arxiv", "scholar", "edu",
                # Government/Official
                ".gov", ".org"
            ]

            # If it matches any legitimate indicator, accept it
            if any(indicator in url_domain for indicator in legitimate_indicators):
                return True

            # For unknown domains, be permissive - accept if it has a reasonable structure
            # This helps with international news sources and smaller publications
            if "." in url_domain and len(url_domain) > 5 and not any(spam in url_domain for spam in spam_indicators):
                return True

            return False

        filtered = []
        for item in items:
            url = item.get("url") or ""
            headline = (item.get("headline") or "").strip()
            # FIX: Exa returns "date" or "published_date", not "published"
            published = item.get("date") or item.get("published_date") or item.get("published")

            if not url or not headline:
                self.logger.debug(f"❌ Filtered (missing URL/headline): {headline[:50] if headline else 'NO_HEADLINE'}")
                continue

            url_domain = domain(url)
            if url_domain in bad_domains:
                self.logger.debug(f"❌ Filtered (bad domain): {url_domain} - {headline[:50]}")
                continue

            # Much more lenient approach - accept articles without dates from any reasonable source
            if not published:
                if is_acceptable_news_source(url_domain):
                    self.logger.debug(f"✅ Accepting no-date article from acceptable source: {url_domain}")
                    filtered.append(item)
                else:
                    self.logger.debug(f"❌ Filtered (unacceptable source, no date): {url_domain} - {headline[:50]}")
                continue

            # For articles with dates, apply lenient age filtering
            try:
                dt = datetime.fromisoformat(published)
                if dt < hard_cutoff:  # Only reject if older than 30 days
                    self.logger.debug(f"❌ Filtered (too old): {published} - {headline[:50]}")
                    continue
                self.logger.debug(f"✅ Kept (with date): {published} - {headline[:50]}")
                filtered.append(item)
            except Exception:
                # If we can't parse date but it's from an acceptable source, include it
                if is_acceptable_news_source(url_domain):
                    self.logger.debug(f"✅ Accepting unparseable-date article from acceptable source: {url_domain}")
                    filtered.append(item)
                else:
                    self.logger.debug(f"❌ Filtered (unacceptable source, bad date): {url_domain} - {headline[:50]}")
                continue

        self.logger.info(f"📊 No-date filter results: {len(items)} → {len(filtered)} ({len(filtered)/len(items)*100:.1f}% kept)" if items else "📊 No-date filter: 0 items to process")
        return filtered

    def _sort_by_quality_indicators(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort articles by quality indicators when quantity is limited"""
        def quality_score(item):
            score = 0
            source = item.get("source", "").lower()
            url = item.get("url", "").lower()
            headline = item.get("headline", "")
            summary = item.get("summary_text", "")

            # Source quality (higher is better)
            premium_sources = ["reuters", "ap news", "bbc", "wsj", "ft", "bloomberg", "nature", "science"]
            if any(premium in source for premium in premium_sources):
                score += 30

            quality_sources = ["npr", "guardian", "economist", "atlantic", "newyorker"]
            if any(quality in source for quality in quality_sources):
                score += 20

            # Content depth indicators
            if len(summary) > 500:
                score += 10
            elif len(summary) > 200:
                score += 5

            # Headline quality (avoid clickbait)
            if len(headline.split()) > 8:  # Longer headlines often more substantive
                score += 5

            # Prefer articles with specific details
            if any(word in headline.lower() for word in ["study", "research", "analysis", "report"]):
                score += 10

            # Domain authority bonus
            authority_domains = [".edu", ".gov", ".org"]
            if any(domain in url for domain in authority_domains):
                score += 15

            return score

        return sorted(items, key=quality_score, reverse=True)

    def _calculate_quality_adjustment(self, available_count: int, target_count: int) -> float:
        """
        Calculate quality adjustment factor based on content availability.
        Returns multiplier for quality thresholds (lower = more lenient)
        """
        if available_count >= target_count * 3:
            # Plenty of content - use strict standards
            adjustment = 1.0
            self.logger.debug(f"📊 Quality adjustment: Abundant content ({available_count} >= {target_count * 3}), strict standards")
        elif available_count >= target_count * 2:
            # Good amount of content - standard quality
            adjustment = 0.9
            self.logger.debug(f"📊 Quality adjustment: Good content availability ({available_count} >= {target_count * 2}), standard quality")
        elif available_count >= target_count:
            # Just enough content - slightly lenient
            adjustment = 0.8
            self.logger.debug(f"📊 Quality adjustment: Adequate content ({available_count} >= {target_count}), slightly lenient")
        elif available_count >= target_count * 0.7:
            # Limited content - more lenient
            adjustment = 0.7
            self.logger.debug(f"📊 Quality adjustment: Limited content ({available_count} < {target_count}), more lenient")
        else:
            # Very limited content - most lenient while maintaining basic standards
            adjustment = 0.6
            self.logger.debug(f"📊 Quality adjustment: Scarce content ({available_count} < {target_count * 0.7}), most lenient")

        return adjustment

    async def _rank_items(self, items: List[Dict[str, Any]], section: str) -> List[RankedItem]:
        ranked_items: List[RankedItem] = []
        # ALWAYS rank the section as a batch; if it fails, fail the pipeline (no silent per-item fallbacks)
        # Attach a synthetic id to each story to ensure stable mapping in AI response
        enriched: List[Dict[str, Any]] = []
        for idx, it in enumerate(items):
            clone = dict(it)
            clone.setdefault("id", str(idx + 1))

            # CRITICAL FIX: Map enrichment_summary/abstract/summary_text to content field for AI ranking
            # The AI ranking prompt expects a 'content' field
            # Priority: enrichment_summary > abstract > summary_text
            if not clone.get("content"):
                clone["content"] = (
                    clone.get("enrichment_summary") or
                    clone.get("abstract") or
                    clone.get("summary_text") or
                    ""
                )

            enriched.append(clone)
        try:
            results: List[RankingResult] = await self.ai.rank_stories(enriched, section=section, cache_service=self.cache_service)  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise AggregationError(f"Ranking failed for section '{section}': {e}")

        # Map results by story_id for score lookup
        id_to_scores: Dict[str, Tuple[float, float, float, float, float, float, float, float, str]] = {}
        for r in results:
            id_to_scores[str(r.story_id)] = (
                float(r.temporal_impact),
                float(r.intellectual_novelty),
                float(r.renaissance_breadth),
                float(r.actionable_wisdom),
                float(r.source_authority),
                float(r.signal_clarity),
                float(r.transformative_potential),
                float(r.total),
                r.one_line_judgment
            )

        for idx, item in enumerate(enriched):
            story_id = str(item.get("id", str(idx + 1)))
            if story_id not in id_to_scores:
                # Try without str conversion in case AI returned an int
                if story_id.isdigit() and int(story_id) in [int(k) if k.isdigit() else -1 for k in id_to_scores.keys()]:
                    # Find the matching key with int conversion
                    for k in id_to_scores.keys():
                        if k.isdigit() and int(k) == int(story_id):
                            story_id = k
                            break
                else:
                    self.logger.warning(f"Ranking results missing story_id {story_id} in section '{section}', skipping item")
                    continue
            temporal, novelty, breadth, wisdom, authority, clarity, transformative, total, _note = id_to_scores[story_id]

            url = item.get("url") or item.get("link") or f"item://{section}/{idx}"
            headline = item.get("headline") or item.get("title") or "Untitled"
            # CRITICAL: Prioritize enrichment_summary from Exa enrichments API
            # Priority: enrichment_summary > abstract > summary_text > description
            content = (
                item.get("enrichment_summary") or
                item.get("abstract") or
                item.get("summary_text") or
                item.get("description") or
                ""
            )
            source = item.get("source") or item.get("source_feed") or ""
            published_raw = item.get("published") or item.get("published_date")
            # Research-specific fallbacks
            if not published_raw and section == Section.RESEARCH_PAPERS:
                published_raw = item.get("publicationDate")
                if not published_raw and item.get("year"):
                    try:
                        published_raw = f"{int(item['year']):04d}-01-01"
                    except Exception:
                        published_raw = None

            # LOG: Track where datetime.now() defaults are happening
            self.logger.debug(f"Date processing for item '{headline[:50]}...': published_raw='{published_raw}' (type: {type(published_raw)})")

            try:
                if isinstance(published_raw, str):
                    published_dt = datetime.fromisoformat(published_raw)
                    self.logger.debug(f"Successfully parsed date string: {published_dt}")
                elif isinstance(published_raw, datetime):
                    published_dt = published_raw
                    self.logger.debug(f"Using datetime object: {published_dt}")
                else:
                    # Adaptive exception: allow no-date for selected sections under shortage-sensitive categories
                    allowed_no_date = {Section.MISCELLANEOUS, Section.TECH_SCIENCE, Section.RESEARCH_PAPERS, Section.BREAKING_NEWS, Section.BUSINESS}
                    if section in allowed_no_date:
                        from datetime import datetime as _dt
                        published_dt = _dt.now()
                        self.logger.warning(f"ALLOWING NO-DATE item in {section}: '{headline[:50]}...' -> defaulting published_date to now()")
                    else:
                        self.logger.warning(f"SKIPPING: No valid date for '{headline[:50]}...', published_raw={published_raw}")
                        continue  # Skip this item entirely
            except Exception as e:
                # Final fallback for allowed sections
                allowed_no_date = {Section.MISCELLANEOUS, Section.TECH_SCIENCE, Section.RESEARCH_PAPERS, Section.BREAKING_NEWS, Section.BUSINESS}
                if section in allowed_no_date:
                    from datetime import datetime as _dt
                    published_dt = _dt.now()
                    self.logger.warning(f"ALLOWING after parse error in {section}: '{headline[:50]}...' -> defaulting published_date to now() ({e})")
                else:
                    self.logger.warning(f"SKIPPING: Date parsing failed for '{headline[:50]}...', published_raw='{published_raw}', error: {e}")
                    continue  # Skip this item entirely

            # CRITICAL: Set preserve_original=True if item has enrichment_summary from Exa Websets
            # This ensures Exa's high-quality enrichments are preserved instead of being re-summarized
            has_enrichment = bool(item.get("enrichment_summary"))
            preserve_original = item.get("preserve_original", False) or has_enrichment
            
            ranked_item = RankedItem(
                id=str(idx + 1),
                url=url,
                headline=headline,
                summary_text=content,
                source=source,
                section=section,
                published_date=published_dt,
                # Store all 7 Renaissance scores for full transparency
                temporal_impact=temporal,
                intellectual_novelty=novelty,
                renaissance_breadth=breadth,
                actionable_wisdom=wisdom,
                source_authority=authority,
                signal_clarity=clarity,
                transformative_potential=transformative,
                # Store the properly calculated weighted total from AI
                total_score=total,
                editorial_note=_note,
                preserve_original=preserve_original
            )
            # Don't recalculate - we already have the proper weighted total
            ranked_items.append(ranked_item)

        ranked_items.sort(key=lambda r: r.total_score, reverse=True)
        for i, r in enumerate(ranked_items, start=1):
            r.rank = i
        return ranked_items

    def _fallback_simple_ranking(self, items: List[Dict[str, Any]], section: str) -> List[RankedItem]:
        """
        Fallback ranking strategy that doesn't use AI.
        Uses heuristics based on source authority, recency, and content quality.
        """
        ranked_items: List[RankedItem] = []

        for idx, item in enumerate(items):
            url = item.get("url", "")
            headline = item.get("headline") or item.get("title", "Unknown")
            # CRITICAL: Prioritize enrichment_summary from Exa enrichments API
            content = (
                item.get("enrichment_summary") or
                item.get("content") or
                item.get("abstract") or
                item.get("summary_text") or
                ""
            )
            source = item.get("source", "Unknown")

            # Parse published date
            published_dt = None
            if item.get("published"):
                try:
                    from datetime import datetime
                    if isinstance(item["published"], str):
                        published_dt = datetime.fromisoformat(item["published"].replace('Z', '+00:00'))
                    elif isinstance(item["published"], datetime):
                        published_dt = item["published"]
                except Exception:
                    pass

            # Calculate simple heuristic scores (0-10 scale)
            # Source authority based on known high-quality sources
            high_authority_sources = ["nature", "science", "arxiv", "financial times", "economist", "wsj", "bloomberg"]
            source_lower = source.lower()
            source_authority = 8.0 if any(auth in source_lower for auth in high_authority_sources) else 5.0

            # Temporal impact based on recency
            temporal_impact = 7.0  # Default
            if published_dt:
                from datetime import datetime, timezone
                # Ensure published_dt is timezone-aware before comparison
                if published_dt.tzinfo is None:
                    # Assume UTC if no timezone info
                    published_dt = published_dt.replace(tzinfo=timezone.utc)

                age_days = (datetime.now(timezone.utc) - published_dt).days
                if age_days <= 1:
                    temporal_impact = 9.0
                elif age_days <= 3:
                    temporal_impact = 7.0
                elif age_days <= 7:
                    temporal_impact = 5.0
                else:
                    temporal_impact = 3.0

            # Signal clarity based on content availability
            signal_clarity = 7.0 if content else 5.0

            # Default scores for other axes
            intellectual_novelty = 6.0
            renaissance_breadth = 5.0
            actionable_wisdom = 6.0
            transformative_potential = 6.0

            # Calculate weighted total (using same weights as AI ranking)
            total = (
                temporal_impact * 0.20 +
                intellectual_novelty * 0.18 +
                signal_clarity * 0.16 +
                source_authority * 0.15 +
                transformative_potential * 0.12 +
                renaissance_breadth * 0.10 +
                actionable_wisdom * 0.09
            )

            # Preserve original if enrichment_summary exists
            has_enrichment = bool(item.get("enrichment_summary"))
            preserve_original = item.get("preserve_original", False) or has_enrichment
            
            ranked_item = RankedItem(
                id=str(idx + 1),
                url=url,
                headline=headline,
                summary_text=content,
                source=source,
                section=section,
                published_date=published_dt,
                temporal_impact=temporal_impact,
                intellectual_novelty=intellectual_novelty,
                renaissance_breadth=renaissance_breadth,
                actionable_wisdom=actionable_wisdom,
                source_authority=source_authority,
                signal_clarity=signal_clarity,
                transformative_potential=transformative_potential,
                total_score=total,
                editorial_note="Ranked using fallback heuristics (AI ranking failed)",
                preserve_original=preserve_original
            )
            ranked_items.append(ranked_item)

        ranked_items.sort(key=lambda r: r.total_score, reverse=True)
        for i, r in enumerate(ranked_items, start=1):
            r.rank = i
        return ranked_items

    def _emergency_ranking(self, items: List[Dict[str, Any]], section: str) -> List[RankedItem]:
        """
        Emergency ranking strategy - converts items to RankedItems with minimal processing.
        Used as absolute last resort when all other ranking strategies fail.
        """
        ranked_items: List[RankedItem] = []

        for idx, item in enumerate(items):
            url = item.get("url", "")
            headline = item.get("headline") or item.get("title", "Unknown")
            # CRITICAL: Prioritize enrichment_summary from Exa enrichments API
            content = (
                item.get("enrichment_summary") or
                item.get("content") or
                item.get("abstract") or
                item.get("summary_text") or
                ""
            )
            source = item.get("source", "Unknown")

            # Use default scores for everything
            default_score = 5.0

            ranked_item = RankedItem(
                id=str(idx + 1),
                url=url,
                headline=headline,
                summary_text=content,
                source=source,
                section=section,
                published_date=None,
                temporal_impact=default_score,
                intellectual_novelty=default_score,
                renaissance_breadth=default_score,
                actionable_wisdom=default_score,
                source_authority=default_score,
                signal_clarity=default_score,
                transformative_potential=default_score,
                total_score=default_score,
                editorial_note="Emergency ranking (all ranking strategies failed)"
            )
            ranked_items.append(ranked_item)

        # Keep original order (no sorting since all scores are the same)
        for i, r in enumerate(ranked_items, start=1):
            r.rank = i
        return ranked_items

    async def _calculate_three_axis_scores(self, item: Dict[str, Any], section: str) -> Tuple[float, float, float]:
        """Legacy method for backward compatibility with tests. Use batch ranking in production."""
        try:
            # For tests, we'll use the AI service's rank_stories method with a single item
            results = await self.ai.rank_stories([item], section=section, cache_service=self.cache_service)
            if results:
                result = results[0]
                # Map the 7-axis scores to the legacy 3-axis format
                impact = (result.temporal_impact + result.transformative_potential) / 2
                delight = (result.intellectual_novelty + result.signal_clarity) / 2
                resonance = (result.renaissance_breadth + result.actionable_wisdom + result.source_authority) / 3
                return (impact, delight, resonance)
        except Exception as e:
            self.logger.warning(f"Per-item ranking failed for test: {e}")

        # Return default scores for tests
        return (5.0, 5.0, 5.0)

    async def select_top_items(self, ranked_sections: Dict[str, List[RankedItem]]) -> Dict[str, List[RankedItem]]:
        """Select final items by section using dynamic thresholds and cross-section balance optimization"""
        return await self._select_top_items_async(ranked_sections)

    async def _select_top_items_async(self, ranked_sections: Dict[str, List[RankedItem]]) -> Dict[str, List[RankedItem]]:
        """Select final items by section using dynamic thresholds and cross-section balance optimization"""
        selected: Dict[str, List[RankedItem]] = {}

        # Get theme analysis for diversity enforcement
        theme_analysis = self.get_theme_analysis()

        # Use fixed quality threshold - no dynamic adjustments
        # We want consistent quality standards across all sections
        fixed_threshold = 4.0  # Fixed threshold on 10-point scale
        self.logger.info(f"Using fixed quality threshold of {fixed_threshold} for all sections")

        # Assess overall content quality for insights
        quality_assessment = None
        try:
            # Now we can properly call async methods since this function is async
            quality_assessment = await self._assess_content_quality_distribution(ranked_sections)
            self._log_quality_assessment(quality_assessment)
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")

        # Phase 1: Initial section selection with dynamic thresholds
        for section, items in ranked_sections.items():
            # Sort by total score descending once
            sorted_items = sorted(items, key=lambda r: r.total_score, reverse=True)

            # Log research papers specifically
            if section == Section.RESEARCH_PAPERS:
                self.logger.info(f"Research Papers: {len(sorted_items)} papers ranked, "
                               f"scores range: {sorted_items[0].total_score if sorted_items else 0:.2f} - "
                               f"{sorted_items[-1].total_score if sorted_items else 0:.2f}")

            # Apply enhanced selection with fixed threshold
            enhanced_selection = self._enhance_section_selection(
                section, sorted_items, theme_analysis, fixed_threshold
            )

            # Log research papers selection result
            if section == Section.RESEARCH_PAPERS:
                self.logger.info(f"Research Papers: {len(enhanced_selection)} papers selected after filtering")

            selected[section] = enhanced_selection

        # Phase 2: Enforce per-section max limits only (do not top-up here)
        for section, items in selected.items():
            min_items, max_items = self.items_per_section.get(section, (0, len(items)))

            if len(items) > max_items:
                items = items[:max_items]
                self.logger.debug(f"Section {section} truncated from {len(selected[section])} to {max_items} items")
                selected[section] = items

            # Log if underfilled; topping up is handled later by aggregate_and_rank
            if len(items) < min_items:
                self.logger.warning(f"Section {section} requires {min_items} items but only has {len(items)}")

        # Phase 3 (removed): Cross-section balance optimization was deemed unnecessary
        # The existing system already handles balance through hard-coded limits and
        # the _enforce_minimums method in aggregate_and_rank

        return selected

    def _enhance_section_selection(self, section: str, ranked_items: List[RankedItem],
                                  theme_analysis: Optional[Dict], dynamic_threshold: Optional[float] = None) -> List[RankedItem]:
        """
        Enhance section selection with narrative flow and diversity enforcement.

        Args:
            section: Section name
            ranked_items: Items ranked by AI scoring
            theme_analysis: Cross-section theme analysis results
            dynamic_threshold: Dynamically calculated quality threshold for this section

        Returns:
            Enhanced selection of items for the section
        """
        if not ranked_items:
            self.logger.warning(f"🚨 Section {section}: No ranked items to select from")
            return []

        # Step 1: Apply dynamic quality threshold
        threshold = dynamic_threshold if dynamic_threshold is not None else self._normalized_threshold()

        self.logger.info(
            f"📊 Section {section}: Starting selection with {len(ranked_items)} ranked items, "
            f"threshold={threshold:.3f}"
        )

        # Special handling for LOCAL section to preserve source diversity
        if section == Section.LOCAL:
            # For local news, use a lower threshold to ensure both sources can be represented
            # We want at least 1 Miami + 1 Cornell if available
            local_threshold = threshold * 0.7  # Lower threshold for local news
            above_threshold = [item for item in ranked_items if item.total_score > local_threshold]

            # Ensure we have enough articles for balancing
            if len(above_threshold) < 4:  # Need at least 4 to have a good chance of both sources
                # Take top 6 articles regardless of threshold for local news
                above_threshold = sorted(ranked_items, key=lambda x: x.total_score, reverse=True)[:6]
        else:
            above_threshold = [item for item in ranked_items if item.total_score > threshold]

            # Log threshold filtering results
            passed = len(above_threshold)
            failed = len(ranked_items) - passed
            self.logger.info(
                f"✅ Section {section}: {passed} items passed threshold, {failed} items filtered out"
            )

            # Log details of filtered items for critical sections
            if section in [Section.BREAKING_NEWS, Section.POLITICS] and failed > 0:
                self.logger.warning(
                    f"⚠️ Section {section}: {failed} items filtered by threshold. "
                    f"Top filtered scores: {[f'{item.total_score:.3f}' for item in sorted(ranked_items, key=lambda x: x.total_score, reverse=True)[passed:passed+3]]}"
                )

        # Fallback: if nothing passes threshold, include all so pipeline can proceed
        if not above_threshold:
            self.logger.warning(
                f"⚠️ Section {section}: NO items passed threshold {threshold:.3f}, "
                f"including all {len(ranked_items)} items as fallback"
            )
            above_threshold = ranked_items

        # Step 2: Apply diversity enforcement if theme analysis is available
        if theme_analysis and self.embeddings and len(above_threshold) > 2:
            diverse_selection = self._enforce_section_diversity(
                section, above_threshold, theme_analysis
            )
        else:
            # Fallback to simple section limits
            diverse_selection = self._apply_section_limits(above_threshold, section)

        self.logger.info(
            f"📋 Section {section}: Selected {len(diverse_selection)} items after diversity/limits"
        )

        # Step 3: Apply narrative flow ordering
        narrative_ordered = self._optimize_narrative_flow(section, diverse_selection)

        return narrative_ordered

    def _enforce_section_diversity(self, section: str, items: List[RankedItem],
                                  theme_analysis: Dict) -> List[RankedItem]:
        """
        Enforce diversity within a section to prevent semantic oversaturation.
        """
        if len(items) <= 2:
            return self._apply_section_limits(items, section)

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            import asyncio

            # Generate embeddings for section items
            item_texts = [f"{item.headline} {item.summary_text[:300]}" for item in items]

            # Create a new event loop for this operation if we're not in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context, but can't await here
                    # Fall back to simple selection
                    return self._apply_section_limits(items, section)
                else:
                    embeddings = loop.run_until_complete(self.embeddings.batch_generate(item_texts))
            except RuntimeError:
                # No event loop, create one
                embeddings = asyncio.run(self.embeddings.batch_generate(item_texts))

            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings_array)

            # Get section limits
            section_limits = self.items_per_section.get(section, (2, 4))
            min_items, max_items = section_limits
            # For exact count sections (min == max), use that exact count
            target_items = max_items if min_items != max_items else min_items

            # Diversity selection algorithm
            selected_items = []
            selected_indices = set()

            # Always include top item
            selected_items.append(items[0])
            selected_indices.add(0)

            # Select remaining items with diversity consideration
            for _ in range(min(target_items - 1, len(items) - 1)):
                best_candidate_idx = None
                best_diversity_score = -1

                for candidate_idx in range(len(items)):
                    if candidate_idx in selected_indices:
                        continue

                    # Calculate diversity score (lower similarity = higher diversity)
                    similarities_to_selected = [
                        similarity_matrix[candidate_idx][selected_idx]
                        for selected_idx in selected_indices
                    ]
                    avg_similarity = np.mean(similarities_to_selected)

                    # Combine diversity with quality score
                    quality_score = items[candidate_idx].total_score / items[0].total_score  # Normalize
                    diversity_bonus = 1 - avg_similarity  # Higher is more diverse

                    # Weighted combination (70% quality, 30% diversity)
                    combined_score = (quality_score * 0.7) + (diversity_bonus * 0.3)

                    if combined_score > best_diversity_score:
                        best_diversity_score = combined_score
                        best_candidate_idx = candidate_idx

                if best_candidate_idx is not None:
                    selected_items.append(items[best_candidate_idx])
                    selected_indices.add(best_candidate_idx)

            # Check against theme analysis warnings
            if theme_analysis.get('theme_overlap_warnings'):
                selected_items = self._filter_oversaturated_themes(
                    selected_items, theme_analysis['theme_overlap_warnings']
                )

            return selected_items

        except Exception as e:
            self.logger.warning(f"Diversity enforcement failed for {section}: {e}")
            return self._apply_section_limits(items, section)

    def _filter_oversaturated_themes(self, items: List[RankedItem],
                                   overlap_warnings: List[Dict]) -> List[RankedItem]:
        """
        Filter out items that contribute to theme oversaturation.
        """
        if not overlap_warnings:
            return items

        filtered_items = []
        warned_themes = {warning['cluster_theme'].lower() for warning in overlap_warnings
                        if warning.get('risk_level') == 'high'}

        for item in items:
            # Check if item contributes to oversaturated themes
            item_text = f"{item.headline} {item.summary_text[:200]}".lower()
            contributes_to_oversaturation = any(
                theme in item_text for theme in warned_themes
            )

            if not contributes_to_oversaturation or len(filtered_items) < 2:
                # Always keep at least 2 items, even if they contribute to oversaturation
                filtered_items.append(item)

        return filtered_items if filtered_items else items[:2]  # Fallback

    def _optimize_narrative_flow(self, section: str, items: List[RankedItem]) -> List[RankedItem]:
        """
        Optimize the ordering of items within a section for better narrative flow.
        """
        if len(items) <= 2:
            return items

        # Section-specific narrative flow patterns
        flow_patterns = {
            Section.BREAKING_NEWS: "impact_desc",      # Highest impact first
            Section.BUSINESS: "impact_desc",           # Major markets/deals first
            Section.TECH_SCIENCE: "novelty_desc",     # Most novel discoveries first
            Section.STARTUP: "actionable_desc",       # Most actionable advice first
            Section.POLITICS: "impact_desc",          # Most significant developments first
            Section.LOCAL: "relevance_desc",          # Most locally relevant first
            Section.MISCELLANEOUS: "novelty_desc",    # Most intellectually novel first
            Section.RESEARCH_PAPERS: "authority_desc" # Highest authority sources first
        }

        pattern = flow_patterns.get(section, "total_desc")

        if pattern == "impact_desc":
            return sorted(items, key=lambda x: x.temporal_impact, reverse=True)
        elif pattern == "novelty_desc":
            return sorted(items, key=lambda x: x.intellectual_novelty, reverse=True)
        elif pattern == "actionable_desc":
            return sorted(items, key=lambda x: x.actionable_wisdom, reverse=True)
        elif pattern == "authority_desc":
            return sorted(items, key=lambda x: x.source_authority, reverse=True)
        elif pattern == "relevance_desc":
            # For local news, combine temporal impact and signal clarity
            return sorted(items, key=lambda x: (x.temporal_impact + x.signal_clarity) / 2, reverse=True)
        else:
            # Default: total score descending
            return sorted(items, key=lambda x: x.total_score, reverse=True)

    def _normalized_threshold(self) -> float:
        """
        Convert the configured 30-point threshold to the 10-point weighted scale
        used by RankedItem.total_score. For example, 15/30 becomes 5.0/10.
        """
        try:
            return float(self.min_score_threshold) / 3.0
        except Exception:
            return 5.0

    # REMOVED: _calculate_dynamic_quality_thresholds and _calculate_section_dynamic_threshold
    # We now use a consistent fixed quality threshold of 4.0 for all sections
    # This ensures consistent quality standards without compromising for content availability
            return base_threshold

        # Analyze current content quality distribution
        scores = [item.total_score for item in items]
        scores.sort(reverse=True)

        # Quality metrics
        max_score = scores[0] if scores else base_threshold
        median_score = scores[len(scores)//2] if scores else base_threshold
        min_score = scores[-1] if scores else base_threshold

        # Calculate quality spread
        quality_spread = max_score - min_score if len(scores) > 1 else 0

        # Get section-specific requirements
        min_items, max_items = self.items_per_section.get(section, (2, 4))

        # Adaptive threshold calculation
        adaptive_threshold = base_threshold

        # Factor 1: Content availability
        # If we have more high-quality content than needed, raise threshold
        # If we barely have enough content, lower threshold
        high_quality_count = sum(1 for score in scores if score >= base_threshold)

        if high_quality_count >= max_items * 1.5:
            # Abundant high-quality content - raise threshold
            adaptive_threshold = base_threshold + (quality_spread * 0.2)
        elif high_quality_count < min_items:
            # Scarce high-quality content - lower threshold to ensure minimum items
            adaptive_threshold = max(median_score * 0.8, base_threshold * 0.7)
        else:
            # Adequate content - minor adjustments based on quality spread
            if quality_spread > 3.0:  # Wide spread indicates mixed quality
                adaptive_threshold = base_threshold + (quality_spread * 0.1)
            else:
                adaptive_threshold = base_threshold

        # Factor 2: Historical performance
        historical_avg = historical_context.get(section, {}).get('avg_score', base_threshold)
        historical_std = historical_context.get(section, {}).get('std_score', 1.0)

        # If current max score is much lower than historical average, be more lenient
        if max_score < historical_avg - historical_std:
            adaptive_threshold *= 0.85  # Lower threshold by 15%
        elif max_score > historical_avg + historical_std:
            adaptive_threshold *= 1.1   # Raise threshold by 10%

        # Factor 3: Section-specific adjustments
        section_multipliers = {
            Section.BREAKING_NEWS: 0.9,    # Lower threshold for breaking news (timeliness matters)
            Section.BUSINESS: 1.0,         # Standard threshold
            Section.TECH_SCIENCE: 1.1,    # Higher threshold for tech/science (quality important)
            Section.RESEARCH_PAPERS: 0.9, # Lower threshold for research (academic papers have different metrics)
            Section.STARTUP: 1.0,         # Standard threshold
            Section.POLITICS: 1.0,        # Standard threshold
            Section.LOCAL: 0.95,          # Slightly lower for local news
            Section.MISCELLANEOUS: 1.05,  # Slightly higher for Renaissance content
            Section.SCRIPTURE: 0.5        # Very low threshold for scripture (always include)
        }

        multiplier = section_multipliers.get(section, 1.0)
        adaptive_threshold *= multiplier

        # Bounds checking
        adaptive_threshold = max(adaptive_threshold, base_threshold * 0.5)  # Never go below 50% of base
        adaptive_threshold = min(adaptive_threshold, base_threshold * 1.5)  # Never exceed 150% of base

        self.logger.debug(f"Dynamic threshold for {section}: {adaptive_threshold:.2f} "
                         f"(base: {base_threshold:.2f}, max_score: {max_score:.2f}, "
                         f"high_quality_count: {high_quality_count})")

        return adaptive_threshold

    async def _get_historical_quality_context(self) -> Dict[str, Dict[str, float]]:
        """
        Get historical quality context from cache service.
        """
        historical_context = {}

        try:
            import aiosqlite
            async with aiosqlite.connect(self.cache_service.db_path) as db:
                # Get average scores by section for last 30 days
                cursor = await db.execute(
                    """
                    SELECT section, AVG(importance_score) as avg_score,
                           COUNT(*) as item_count
                    FROM seen_items
                    WHERE was_included = 1
                      AND importance_score IS NOT NULL
                      AND first_seen_date >= date('now', '-30 days')
                    GROUP BY section
                    HAVING COUNT(*) >= 5
                    """)

                rows = await cursor.fetchall()

                for section, avg_score, count in rows:
                    # Calculate standard deviation for each section
                    cursor2 = await db.execute(
                        """
                        SELECT importance_score
                        FROM seen_items
                        WHERE section = ?
                          AND was_included = 1
                          AND importance_score IS NOT NULL
                          AND first_seen_date >= date('now', '-30 days')
                        """, (section,))

                    scores = [row[0] for row in await cursor2.fetchall()]

                    if len(scores) >= 5:
                        import statistics
                        std_score = statistics.stdev(scores) if len(scores) > 1 else 1.0

                        historical_context[section] = {
                            'avg_score': avg_score,
                            'std_score': std_score,
                            'sample_size': len(scores)
                        }

        except Exception as e:
            self.logger.warning(f"Failed to get historical quality data: {e}")

        return historical_context

    async def _assess_content_quality_distribution(self, sections: Dict[str, List[RankedItem]]) -> Dict[str, Any]:
        """
        Assess the overall quality distribution across all sections.

        Returns:
            Dict containing quality assessment metrics and recommendations
        """
        assessment = {
            'overall_quality': {},
            'section_quality': {},
            'quality_recommendations': [],
            'threshold_adjustments': {}
        }

        all_scores = []
        section_stats = {}

        # Calculate quality statistics
        for section, items in sections.items():
            if not items:
                continue

            scores = [item.total_score for item in items]
            all_scores.extend(scores)

            if scores:
                section_stats[section] = {
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'avg_score': sum(scores) / len(scores),
                    'count': len(scores),
                    'quality_spread': max(scores) - min(scores) if len(scores) > 1 else 0
                }

        # Overall quality metrics
        if all_scores:
            import statistics
            assessment['overall_quality'] = {
                'max_score': max(all_scores),
                'min_score': min(all_scores),
                'avg_score': statistics.mean(all_scores),
                'median_score': statistics.median(all_scores),
                'std_score': statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                'total_items': len(all_scores)
            }

        assessment['section_quality'] = section_stats

        # Generate quality recommendations
        base_threshold = self._normalized_threshold()

        for section, stats in section_stats.items():
            # Check for quality issues
            if stats['max_score'] < base_threshold:
                assessment['quality_recommendations'].append({
                    'section': section,
                    'issue': 'low_max_quality',
                    'max_score': stats['max_score'],
                    'threshold': base_threshold,
                    'recommendation': f'All items in {section} below quality threshold - consider source review'
                })

            if stats['count'] > 0 and stats['avg_score'] < base_threshold * 0.8:
                assessment['quality_recommendations'].append({
                    'section': section,
                    'issue': 'low_avg_quality',
                    'avg_score': stats['avg_score'],
                    'recommendation': f'Average quality in {section} is low - may need better content sources'
                })

            # Check for insufficient content
            min_items, max_items = self.items_per_section.get(section, (2, 4))
            high_quality_count = sum(1 for item in sections[section]
                                   if item.total_score >= base_threshold)

            if high_quality_count < min_items:
                assessment['quality_recommendations'].append({
                    'section': section,
                    'issue': 'insufficient_quality_content',
                    'high_quality_count': high_quality_count,
                    'min_required': min_items,
                    'recommendation': f'Only {high_quality_count} high-quality items in {section}, need {min_items}'
                })

        return assessment

    def _log_quality_assessment(self, assessment: Dict[str, Any]) -> None:
        """
        Log quality assessment results for monitoring and debugging.
        """
        overall = assessment.get('overall_quality', {})
        if overall:
            self.logger.info(f"📊 Quality Assessment: "
                           f"avg={overall.get('avg_score', 0):.2f}, "
                           f"median={overall.get('median_score', 0):.2f}, "
                           f"std={overall.get('std_score', 0):.2f}, "
                           f"items={overall.get('total_items', 0)}")

        # Log section-specific quality
        section_quality = assessment.get('section_quality', {})
        for section, stats in section_quality.items():
            self.logger.debug(f"🎯 {section}: "
                            f"avg={stats.get('avg_score', 0):.2f}, "
                            f"max={stats.get('max_score', 0):.2f}, "
                            f"count={stats.get('count', 0)}")

        # Log quality recommendations
        recommendations = assessment.get('quality_recommendations', [])
        if recommendations:
            self.logger.warning(f"⚠️  Quality issues detected in {len(recommendations)} sections:")
            for rec in recommendations[:3]:  # Limit to first 3 for brevity
                self.logger.warning(f"   {rec['section']}: {rec['recommendation']}")

    def _apply_section_limits(self, items: List[RankedItem], section: str) -> List[RankedItem]:
        limits = self.items_per_section.get(section, (0, len(items) if items else 0))
        min_items, max_items = limits

        self.logger.debug(
            f"🔢 Section {section}: Applying limits (min={min_items}, max={max_items}) to {len(items)} items"
        )

        if not items:
            self.logger.warning(f"⚠️ Section {section}: No items to apply limits to")
            return []

        # Special handling for LOCAL section to ensure source balance
        if section == Section.LOCAL and len(items) >= 2:
            return self._balance_local_sources(items, min_items, max_items)

        items_sorted = sorted(items, key=lambda r: r.total_score, reverse=True)

        # STRICT ENFORCEMENT: For sections with exact counts (min == max), return exactly that many
        if min_items == max_items:
            # Take exactly the required number of items
            exact_count = min_items
            if len(items_sorted) >= exact_count:
                self.logger.info(
                    f"✅ Section {section}: Selecting exactly {exact_count} items (strict quota)"
                )
                return items_sorted[:exact_count]
            else:
                # CRITICAL: Log critical warning - this will trigger backfill in aggregate_and_rank
                self.logger.error(
                    f"🚨 Section {section}: UNDERFILLED - requires {exact_count} items but only has {len(items_sorted)}. "
                    f"This will trigger additional backfill to ensure quota is met."
                )
                return items_sorted  # Return what we have, backfill will add more
        else:
            # For flexible sections (Scripture, Politics, Local, Extra), use the original logic
            if len(items_sorted) > max_items:
                self.logger.debug(
                    f"✂️ Section {section}: Truncating from {len(items_sorted)} to {max_items} items"
                )
                items_sorted = items_sorted[:max_items]

            self.logger.info(
                f"✅ Section {section}: Selected {len(items_sorted)} items (flexible quota)"
            )
            return items_sorted

    def _balance_local_sources(self, items: List[RankedItem], min_items: int, max_items: int) -> List[RankedItem]:
        """
        Balance local news sources to ensure 1 Miami + 1 Cornell article when possible.
        """
        miami_items = []
        cornell_items = []

        # Separate items by location metadata (added during fetch)
        for item in items:
            # Check location metadata first (most reliable)
            location = getattr(item, 'location', None)

            # Log item details for debugging
            self.logger.debug(f"Local item: location='{location}', source='{item.source}', url='{item.url[:50]}...'")

            if location == "Miami":
                miami_items.append(item)
                self.logger.debug(f"  -> Identified as Miami article via location metadata")
            elif location == "Cornell":
                cornell_items.append(item)
                self.logger.debug(f"  -> Identified as Cornell article via location metadata")
            else:
                # Fallback to string matching if no metadata (backward compatibility)
                source_lower = item.source.lower() if item.source else ""
                url_lower = item.url.lower() if item.url else ""

                if ("miami" in source_lower or "miamiherald" in source_lower or
                    "miamiherald.com" in url_lower):
                    miami_items.append(item)
                    self.logger.debug(f"  -> Identified as Miami Herald article via string matching")
                elif ("cornell" in source_lower or "news.cornell" in source_lower or
                      "cornellsun" in source_lower or "cornell.edu" in url_lower or
                      "cornellsun.com" in url_lower):
                    cornell_items.append(item)
                    self.logger.debug(f"  -> Identified as Cornell article via string matching")
                else:
                    self.logger.debug(f"  -> Could not identify source (neither Miami nor Cornell)")

        # Sort each group by score
        miami_items.sort(key=lambda r: r.total_score, reverse=True)
        cornell_items.sort(key=lambda r: r.total_score, reverse=True)

        # Log the source distribution
        self.logger.info(f"Local news source distribution: {len(miami_items)} Miami, {len(cornell_items)} Cornell")

        balanced_items = []

        # If we have both sources and want 2 items, take 1 from each
        if miami_items and cornell_items and max_items >= 2:
            balanced_items.append(miami_items[0])
            balanced_items.append(cornell_items[0])
            self.logger.info(f"✅ Local news balanced: 1 Miami Herald + 1 Cornell article")
            self.logger.info(f"  Miami: {miami_items[0].headline[:60]}...")
            self.logger.info(f"  Cornell: {cornell_items[0].headline[:60]}...")
        # If we only have one source, take the best items up to the limit
        elif miami_items and not cornell_items:
            balanced_items = miami_items[:max_items]
            self.logger.warning(f"⚠️ Local news: Only Miami Herald articles available ({len(miami_items)} found, no Cornell)")
        elif cornell_items and not miami_items:
            balanced_items = cornell_items[:max_items]
            self.logger.warning(f"⚠️ Local news: Only Cornell articles available ({len(cornell_items)} found, no Miami)")
        else:
            # Fallback to regular sorting if no clear source separation
            all_items = sorted(items, key=lambda r: r.total_score, reverse=True)
            balanced_items = all_items[:max_items]
            self.logger.warning(f"⚠️ Local news: Could not identify sources for balancing ({len(items)} total items)")

        # Ensure we meet minimum requirements
        if len(balanced_items) < min_items:
            # Add more items from either source to meet minimum
            remaining_miami = [i for i in miami_items if i not in balanced_items]
            remaining_cornell = [i for i in cornell_items if i not in balanced_items]
            remaining = remaining_miami + remaining_cornell
            remaining.sort(key=lambda r: r.total_score, reverse=True)

            needed = min_items - len(balanced_items)
            balanced_items.extend(remaining[:needed])

        # Enforce maximum
        if len(balanced_items) > max_items:
            balanced_items = balanced_items[:max_items]

        return balanced_items

    async def aggregate_and_rank(self) -> Dict[str, List[RankedItem]]:
        sections_raw = await self.fetch_all_content()
        ranked = await self.rank_all_content(sections_raw)
        selected = await self.select_top_items(ranked)
        # Triage shortages and try to top up from ranked pool first
        selected = self._enforce_minimums(selected, ranked)

        # Strict sections must meet exact targets; attempt controlled backfill before failing
        shortages = self._shortages_for_strict_sections(selected)
        attempt = 0
        while shortages and attempt < 2:
            self.logger.warning(f"Shortages detected: {shortages} (attempt {attempt+1}/2)")
            for section, needed in list(shortages.items()):
                # Compute existing URLs to avoid duplicates
                existing_urls = {ri.url for ri in selected.get(section, []) if getattr(ri, 'url', None)}

                # Backfill per section strategy
                if section == Section.RESEARCH_PAPERS:
                    try:
                        papers_res = await self._fetch_hybrid_papers()
                        # Rank these papers
                        backfill_ranked_map = await self.rank_all_content({Section.RESEARCH_PAPERS: papers_res.items})
                        candidates = backfill_ranked_map.get(Section.RESEARCH_PAPERS, [])
                        # Filter out already selected
                        candidates = [c for c in candidates if c.url and c.url not in existing_urls]
                        # Take needed
                        add = candidates[:needed]
                        if add:
                            selected.setdefault(section, []).extend(add)
                            self.logger.info(f"Added {len(add)} research papers to meet target")
                    except Exception as e:  # noqa: BLE001
                        self.logger.warning(f"Backfill error for research papers: {e}")
                elif section in {Section.BUSINESS, Section.TECH_SCIENCE}:
                    try:
                        backfill_items = await self._backfill_with_adapter(section, needed, attempt)
                        backfill_ranked_map = await self.rank_all_content({section: backfill_items})
                        candidates = backfill_ranked_map.get(section, [])
                        candidates = [c for c in candidates if c.url and c.url not in existing_urls]
                        add = candidates[:needed]
                        if add:
                            selected.setdefault(section, []).extend(add)
                            self.logger.info(f"Added {len(add)} {section} items to meet target")
                    except Exception as e:  # noqa: BLE001
                        self.logger.warning(f"Backfill error for {section}: {e}")
                else:
                    # Generic attempt: use existing ranked pool to top up if any available
                    pool = ranked.get(section, [])
                    pool = [ri for ri in pool if ri.url and ri.url not in existing_urls]
                    add = pool[:needed]
                    if add:
                        selected.setdefault(section, []).extend(add)
                        self.logger.info(f"Generically added {len(add)} items to {section}")

            # Re-enforce per-section max limits and recompute shortages
            for sec, (_, max_items) in self.items_per_section.items():
                if len(selected.get(sec, [])) > max_items:
                    selected[sec] = selected[sec][:max_items]
            shortages = self._shortages_for_strict_sections(selected)
            attempt += 1

        if not self._validate_content_balance(selected):
            raise AggregationError("Content balance validation failed")
        return selected

    def _enforce_minimums(
        self,
        selected: Dict[str, List[RankedItem]],
        ranked: Dict[str, List[RankedItem]],
    ) -> Dict[str, List[RankedItem]]:
        """
        Ensure each section meets its minimum by relaxing threshold when necessary.

        - Operates after select_top_items so it does not affect unit tests that
          assert select_top_items behavior in isolation.
        - Top-up pulls highest-ranked remaining items from the full ranked pool,
          then reapplies max limits if exceeded.
        """
        self.logger.info("🔧 Starting minimum enforcement for all sections")

        out: Dict[str, List[RankedItem]] = {}
        for section, chosen in selected.items():
            min_items, max_items = self.items_per_section.get(section, (0, len(chosen)))

            self.logger.debug(
                f"📋 Section {section}: Checking minimums - has {len(chosen)}, needs {min_items}"
            )

            if len(chosen) >= min_items:
                self.logger.debug(f"✅ Section {section}: Already meets minimum ({len(chosen)} >= {min_items})")
                out[section] = chosen
                continue

            # Need to backfill
            pool = ranked.get(section, [])
            before = len(chosen)

            self.logger.warning(
                f"⚠️ Section {section}: UNDERFILLED - has {before}, needs {min_items}, "
                f"pool has {len(pool)} total items. Attempting backfill..."
            )

            # Fill from top of ranked pool, skipping already selected
            picked_ids = {id(x) for x in chosen}
            backfilled = 0

            for candidate in pool:
                if id(candidate) in picked_ids:
                    continue
                chosen.append(candidate)
                picked_ids.add(id(candidate))
                backfilled += 1
                if len(chosen) >= min_items:
                    break

            self.logger.info(
                f"🔄 Section {section}: Backfilled {backfilled} items from pool "
                f"({before} -> {len(chosen)})"
            )

            # Enforce max after filling
            if len(chosen) > max_items:
                self.logger.debug(f"✂️ Section {section}: Truncating from {len(chosen)} to {max_items}")
                chosen = chosen[:max_items]

            # STRICT ENFORCEMENT: For ALL sections with exact counts (min == max), ensure we have exactly that many
            if min_items == max_items and len(chosen) < min_items:
                # Try harder to get exact count for strict sections (Breaking News, Business, Tech/Science, Research Papers, Startup, Miscellaneous)
                self.logger.error(
                    f"🚨 Section {section}: CRITICAL - requires exactly {min_items} items but only has {len(chosen)} after backfill. "
                    f"Attempting additional backfill from pool..."
                )

                # Try to pull more from the pool if possible
                remaining_pool = [item for item in pool if id(item) not in picked_ids]

                if remaining_pool:
                    additional_needed = min_items - len(chosen)
                    additional_items = remaining_pool[:additional_needed]
                    chosen.extend(additional_items)
                    self.logger.info(
                        f"✅ Section {section}: Added {len(additional_items)} more items "
                        f"to meet exact requirement of {min_items}"
                    )
                else:
                    # CRITICAL: Instead of failing, trigger additional backfill fetch
                    # This ensures quotas are ALWAYS met by fetching more content
                    self.logger.error(
                        f"❌ Section {section}: Cannot meet exact requirement of {min_items} items from pool. "
                        f"Pool size: {len(pool)}, Already picked: {len(picked_ids)}. "
                        f"Will attempt additional backfill in next iteration."
                    )
                    # Shortage will be handled by outer backfill loop in aggregate_and_rank

            after = len(chosen)
            self.logger.info(
                f"📊 Section {section}: Minimum enforcement complete - {before} -> {after} "
                f"(min={min_items}, max={max_items})"
            )
            out[section] = chosen

        # Include sections that had no selected items originally
        for section, pool in ranked.items():
            if section not in out:
                self.logger.warning(
                    f"⚠️ Section {section}: Not in selected items, using empty list from selected"
                )
                out[section] = selected.get(section, [])

        # CRITICAL: Ensure all required sections exist in output, even if empty
        # This prevents sections from completely disappearing from the newsletter
        required_sections = [
            Section.BREAKING_NEWS,
            Section.BUSINESS,
            Section.TECH_SCIENCE,
            Section.RESEARCH_PAPERS,
            Section.POLITICS,
            Section.MISCELLANEOUS,
        ]

        for section in required_sections:
            if section not in out:
                self.logger.critical(
                    f"🚨 CRITICAL: Section '{section}' is MISSING from output! "
                    f"This section is required and must appear in every newsletter."
                )
                # Try to get items from ranked pool
                if section in ranked and ranked[section]:
                    min_items, _ = self.items_per_section.get(section, (0, 0))
                    out[section] = ranked[section][:min_items] if min_items > 0 else ranked[section][:3]
                    self.logger.warning(
                        f"⚠️ Emergency recovery: Added {len(out[section])} items from ranked pool for '{section}'"
                    )
                else:
                    # Section has no ranked items at all - this is a critical failure
                    self.logger.critical(
                        f"💀 CATASTROPHIC: Section '{section}' has NO ranked items! "
                        f"Newsletter will be incomplete. This should never happen."
                    )
                    out[section] = []

        # Final summary
        total_selected = sum(len(items) for items in out.values())
        self.logger.info(
            f"✅ Minimum enforcement complete: {total_selected} total items across {len(out)} sections"
        )

        return out

    def _validate_content_balance(self, sections: Dict[str, List[RankedItem]]) -> bool:
        # Require core sections to be present with at least one item
        required = [Section.BREAKING_NEWS, Section.TECH_SCIENCE, Section.SCRIPTURE]
        for sec in required:
            if not sections.get(sec):
                return False
        return True

    async def _handle_fetch_failure(self, source: str, error: Exception) -> FetchResult:
        msg = f"{type(error).__name__}: {error}"

        # Define which services are truly critical (pipeline cannot continue without them)
        # Currently, we can actually continue without any single service - just with reduced content
        critical_services = set()  # Empty set - no service is truly critical for newsletter delivery

        # Services that are important but not critical
        important_services = {"perplexity", "ai_news", "gemini"}

        # Check if this is a critical failure that should stop the pipeline
        if source == "aggregate":
            # This should only happen if the entire fetch_all_content method fails
            # Not for individual service failures
            self.logger.error("Critical aggregate fetch failure: %s", msg)
            raise AggregationError(msg)
        elif source in critical_services:
            # Critical service failed - pipeline cannot continue
            self.logger.error("Critical service %s failed: %s", source, msg)
            raise AggregationError(f"Critical service {source} failed: {msg}")
        elif source in important_services:
            # Important service failed - log error but continue
            self.logger.error("Important service %s failed: %s (continuing with reduced content)", source, msg)
        else:
            # Non-critical service failed - just log warning
            self.logger.warning("Non-critical service %s failed: %s", source, msg)

        # Return error result with empty items so pipeline can continue
        return FetchResult(source=source, section=Section.MISCELLANEOUS, items=[], fetch_time=0.0, error=msg)

    def get_fetch_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"sources": {}, "total_errors": 0}
        for fr in self._fetch_stats:
            src = fr.source
            src_stats = stats["sources"].setdefault(src, {"count": 0, "time": 0.0, "errors": 0})
            src_stats["count"] += 1
            src_stats["time"] += float(fr.fetch_time)
            if fr.error:
                src_stats["errors"] += 1
                stats["total_errors"] += 1
        return stats


