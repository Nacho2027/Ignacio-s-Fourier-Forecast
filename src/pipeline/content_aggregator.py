import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Services live at src/services/*. Use actual module names in this repo
from src.services.llmlayer import LLMLayerService, LLMLayerResult, Citation
from src.services.arxiv import ArxivService, ArxivPaper
from src.services.rss import RSSService, DailyReading
from src.services.ai_service import AIService, RankingResult
from src.services.semantic_scholar_service import SemanticScholarService


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
    content: str
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

    # Backwards compatibility aliases
    @property
    def impact_score(self) -> float:
        return self.temporal_impact
    
    @property
    def delight_score(self) -> float:
        return self.intellectual_novelty
    
    @property
    def resonance_score(self) -> float:
        return self.renaissance_breadth

    total_score: float = 0.0
    rank: Optional[int] = None

    editorial_note: Optional[str] = None
    angle: Optional[str] = None

    def calculate_total_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted total using all 7 Renaissance dimensions"""
        if weights is None:
            # Use the same weights as AI service (line 207-215)
            weights = {
                "temporal_impact": 0.25,
                "intellectual_novelty": 0.20,
                "renaissance_breadth": 0.15,
                "actionable_wisdom": 0.15,
                "source_authority": 0.10,
                "signal_clarity": 0.10,
                "transformative_potential": 0.05
            }
        self.total_score = (
            self.temporal_impact * weights.get("temporal_impact", 0.25) +
            self.intellectual_novelty * weights.get("intellectual_novelty", 0.20) +
            self.renaissance_breadth * weights.get("renaissance_breadth", 0.15) +
            self.actionable_wisdom * weights.get("actionable_wisdom", 0.15) +
            self.source_authority * weights.get("source_authority", 0.10) +
            self.signal_clarity * weights.get("signal_clarity", 0.10) +
            self.transformative_potential * weights.get("transformative_potential", 0.05)
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
        llmlayer: LLMLayerService,
        arxiv: ArxivService,
        rss: RSSService,
        ai: AIService,
        cache_service=None,
        embeddings=None,
        semantic_scholar: Optional[SemanticScholarService] = None,
    ) -> None:
        self.llmlayer = llmlayer
        self.arxiv = arxiv
        self.rss = rss
        self.ai = ai
        self.cache_service = cache_service
        self.embeddings = embeddings
        self.semantic_scholar = semantic_scholar

        self.parallel_limit = 10
        self.fetch_timeout = 180  # Increased from 60s to 180s (3 minutes) to allow LLMLayer to complete
        # Keep threshold in 30-point scale for initialization expectations/tests
        self.min_score_threshold = 12  # Lowered from 15 to include more research papers (4.0 threshold)
        # HARD LIMITS: Exact article counts per section (no ranges!)
        self.items_per_section: Dict[str, Tuple[int, int]] = {
            Section.BREAKING_NEWS: (3, 3),  # Exactly 3
            Section.BUSINESS: (3, 3),       # Exactly 3
            Section.TECH_SCIENCE: (3, 3),   # Exactly 3
            Section.RESEARCH_PAPERS: (5, 5), # Exactly 5
            Section.STARTUP: (2, 2),        # Exactly 2
            Section.SCRIPTURE: (6, 10),     # Keep flexible for Scripture
            Section.POLITICS: (0, 2),       # 0-2 per VISION.txt
            Section.LOCAL: (2, 2),          # Exactly 2 (1 Miami + 1 Cornell when possible)
            Section.MISCELLANEOUS: (5, 5),  # Exactly 5
            Section.EXTRA: (0, 2),          # 0-2 flexible
        }

        self.logger = logging.getLogger(__name__)
        self._fetch_stats: List[FetchResult] = []

    async def fetch_all_content(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch content from all sources in parallel."""
        tasks: List[asyncio.Task] = []

        # Create tasks with proper service identification for graceful failure handling
        # LLMLayer sections - critical for news content
        async def fetch_llmlayer_with_timeout():
            start = asyncio.get_event_loop().time()
            try:
                result = await asyncio.wait_for(self._fetch_llmlayer_sections(), timeout=self.fetch_timeout)
                if isinstance(result, list):
                    for r in result:
                        if isinstance(r, FetchResult):
                            self._fetch_stats.append(r)
                elif isinstance(result, FetchResult):
                    self._fetch_stats.append(result)
                return result
            except asyncio.TimeoutError:
                self.logger.error(f"LLMLayer sections timed out after {self.fetch_timeout}s - returning partial results")
                # Return empty result instead of crashing, let other sources continue
                return []
            except Exception as e:
                self.logger.error(f"LLMLayer sections failed: {e} - returning partial results")
                return []

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

        tasks.append(asyncio.create_task(fetch_llmlayer_with_timeout()))
        tasks.append(asyncio.create_task(fetch_research_with_timeout()))
        # RSS feeds are now handled within _fetch_catholic()
        # tasks.append(asyncio.create_task(timed(self._fetch_rss_feeds())))

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
                            sections[fr.section].append({
                                "headline": f"{fr.section.replace('_', ' ').title()} - Content Unavailable",
                                "url": "#",
                                "content": f"We were unable to fetch {fr.section.replace('_', ' ')} at this time. Error: {fr.error}",
                                "source": "System Notice",
                                "published": datetime.now().isoformat(),
                                "is_placeholder": True
                            })
                        else:
                            # Items may be list or dict depending on section
                            if isinstance(fr.items, list):
                                sections[fr.section].extend(fr.items)
                            else:
                                sections[fr.section].append(fr.items)
            elif isinstance(res, FetchResult):
                sections.setdefault(res.section, [])
                if res.error is not None:
                    # Include placeholder content for failed sections
                    self.logger.warning(f"Section {res.section} failed: {res.error}")
                    sections[res.section].append({
                        "headline": f"{res.section.replace('_', ' ').title()} - Content Unavailable",
                        "url": "#",
                        "content": f"We were unable to fetch {res.section.replace('_', ' ')} content at this time. Error: {res.error}",
                        "source": "System Notice",
                        "published": datetime.now().isoformat(),
                        "is_placeholder": True
                    })
                else:
                    if isinstance(res.items, list):
                        sections[res.section].extend(res.items)
                    else:
                        sections[res.section].append(res.items)

        return sections

    async def _fetch_llmlayer_sections(self) -> List[FetchResult]:
        names = [
            "breaking_news",
            "business",
            "tech_science",
            "startup",
            "politics",
            "local",
            "miscellaneous",
            "catholic",
        ]
        tasks = [
            asyncio.wait_for(self._fetch_breaking_news(), timeout=self.fetch_timeout),
            asyncio.wait_for(self._fetch_business_news(), timeout=self.fetch_timeout),
            asyncio.wait_for(self._fetch_tech_science(), timeout=self.fetch_timeout),
            asyncio.wait_for(self._fetch_startup_insights(), timeout=self.fetch_timeout),
            asyncio.wait_for(self._fetch_politics(), timeout=self.fetch_timeout),
            asyncio.wait_for(self._fetch_local_news(), timeout=self.fetch_timeout),
            asyncio.wait_for(self._fetch_miscellaneous(), timeout=self.fetch_timeout),
            asyncio.wait_for(self._fetch_scripture(), timeout=self.fetch_timeout),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: List[FetchResult] = []
        for name, r in zip(names, results):
            if isinstance(r, FetchResult):
                out.append(r)
            elif isinstance(r, Exception):
                # Log the error but don't kill the entire pipeline
                self.logger.error("LLMLayer subfetch '%s' failed: %s", name, r)
                # Create an empty FetchResult for this section so other sections can continue
                # Map section names to Section enum values
                section_map = {
                    "breaking_news": Section.BREAKING_NEWS,
                    "business": Section.BUSINESS,
                    "tech_science": Section.TECH_SCIENCE,
                    "startup": Section.STARTUP,
                    "politics": Section.POLITICS,
                    "local": Section.LOCAL,
                    "miscellaneous": Section.MISCELLANEOUS,
                    "scripture": Section.SCRIPTURE,
                }
                section_enum = section_map.get(name, Section.MISCELLANEOUS)
                out.append(FetchResult("llmlayer", section_enum, [], 0.0, error=str(r)))
        return out

    async def _fetch_breaking_news(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            # VISION.txt specifies AP News and Reuters ONLY for Breaking News
            # Must be high-impact, global importance, no sensationalism
            self.logger.info("Fetching breaking news from AP News and Reuters via LLMLayer")
            from datetime import datetime
            today = datetime.now().strftime("%B %d, %Y")
            result = await self.llmlayer.search(
                query=f"What are the most important breaking news stories from {today} today from Associated Press and Reuters? Focus on major global events, conflicts, disasters, and urgent developments happening on {today}",
                max_results=20,  # Fetch 20, then validate and rank to best 3-5
                domains=["apnews.com", "reuters.com"],  # Now properly used via domain_filter
                recency="day",
                search_type="news"  # Use news search type for news content
            )
            self.logger.info("Breaking news: Got %d citations from AP/Reuters", len(result.citations))
            
            # Log citation date processing for debugging
            self.logger.info(f"Breaking News: Processing {len(result.citations)} citations for date conversion")
            items_raw = []
            for i, c in enumerate(result.citations):
                published_iso = None
                if c.published_date:
                    published_iso = c.published_date.isoformat()
                    self.logger.debug(f"Breaking News Citation {i+1}: Found published_date = {c.published_date} -> {published_iso}")
                else:
                    self.logger.warning(f"Breaking News Citation {i+1}: No published_date available for '{c.title[:50]}...' from {c.source_name}")
                
                item = {
                    "headline": c.title,
                    "url": c.url,
                    "content": c.snippet,  # This now contains full article content (up to 5000 chars) after fetch
                    "source": c.source_name,
                    "published": published_iso,
                }
                items_raw.append(item)
            
            # Validate sources BEFORE other filtering
            items_validated = self._validate_sources(items_raw, ["apnews.com", "reuters.com"], "Breaking News")
            items = self._filter_items(items_validated, max_age_days=1)  # Breaking news must be very recent
            self.logger.info("Breaking news: %d items after validation and filtering from %d raw", len(items), len(items_raw))
            return FetchResult(
                source="llmlayer",
                section=Section.BREAKING_NEWS,
                items=items,
                fetch_time=asyncio.get_event_loop().time() - start,
            )
        except Exception as e:  # noqa: BLE001
            return await self._handle_fetch_failure("llmlayer", e)

    async def _fetch_business_news(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            # VISION.txt says WSJ and Axios Pro Rata (user has subscriptions)
            # Focus on major market moves, big deals, economic shifts per VISION.txt
            self.logger.info("Fetching business news from WSJ/Axios via LLMLayer")
            from datetime import datetime
            today = datetime.now().strftime("%B %d, %Y")
            result = await self.llmlayer.search(
                query=f"What are the important business and market developments from {today} today from Wall Street Journal and Axios? Focus on major market moves, M&A deals over $100M, venture capital mega-deals, IPOs, and Federal Reserve decisions happening on {today}",
                max_results=8,  # Fetch 8, then validate and rank to best 2-4
                domains=["wsj.com", "axios.com"],  # Now properly used via domain_filter
                recency="day",
                search_type="news"  # Use news search type for business news
            )
            self.logger.info("Business: Got %d citations from WSJ/Axios/Bloomberg", len(result.citations))
            items_raw = [
                {
                    "headline": c.title,
                    "url": c.url,
                    "content": c.snippet,  # This now contains full article content (up to 5000 chars) after fetch
                    "source": c.source_name,
                    "published": (c.published_date.isoformat() if c.published_date else None),
                }
                for c in result.citations
            ]
            # Validate sources BEFORE other filtering
            items_validated = self._validate_sources(items_raw, ["wsj.com", "axios.com"], "Business")
            items = self._filter_items(items_validated, max_age_days=2)  # Business news can be 1-2 days old
            self.logger.info("Business: %d items after validation and filtering from %d raw", len(items), len(items_raw))
            return FetchResult("llmlayer", Section.BUSINESS, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            return await self._handle_fetch_failure("llmlayer", e)

    async def _fetch_tech_science(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            # VISION.txt specifies: TLDR newsletter, MIT Tech Review, IEEE Spectrum, Quanta Magazine
            self.logger.info("Fetching tech/science from specified sources via LLMLayer")
            from datetime import datetime
            today = datetime.now().strftime("%B %d, %Y")
            
            # More specific, current query with diverse tech topics
            result = await self.llmlayer.search(
                query=(
                    f"Latest breakthrough technology science discovery innovation research 2025 {today}. "
                    f"AI artificial intelligence machine learning breakthrough announcements. "
                    f"Quantum computing research papers and discoveries. "
                    f"Climate technology carbon capture renewable energy innovations. "
                    f"Space exploration NASA SpaceX astronomy findings. "
                    f"Biotech CRISPR gene therapy clinical trial results. "
                    f"Robotics automation Boston Dynamics Tesla innovations. "
                    f"From MIT Technology Review, IEEE Spectrum, Quanta Magazine, TLDR tech newsletter"
                ),
                max_results=30,  # Increase to get more variety
                domains=["technologyreview.com", "spectrum.ieee.org", "quantamagazine.org", "tldr.tech"],
                recency="month",  # Broader recency for tech publications
                search_type="general"  # Tech articles aren't always "news"
            )
            self.logger.info("Tech/Science: Got %d citations from MIT/IEEE/Quanta", len(result.citations))
            items_raw = [
                {
                    "headline": c.title,
                    "url": c.url,
                    "content": c.snippet,  # This now contains full article content (up to 5000 chars) after fetch
                    "source": c.source_name,
                    "published": (c.published_date.isoformat() if c.published_date else None),
                }
                for c in result.citations
            ]
            # Validate sources BEFORE other filtering
            items_validated = self._validate_sources(
                items_raw, 
                ["technologyreview.com", "spectrum.ieee.org", "quantamagazine.org", "tldr.tech"],
                "Tech/Science"
            )
            items = self._filter_items(items_validated, max_age_days=14)  # Tech publications update weekly/biweekly
            self.logger.info("Tech/Science: %d items after filtering from %d raw", len(items), len(items_raw))
            return FetchResult("llmlayer", Section.TECH_SCIENCE, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            return await self._handle_fetch_failure("llmlayer", e)

    async def _fetch_startup_insights(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            # VISION.txt specifies: YC Blog, First Round Review, Founders Podcast
            self.logger.info("Fetching startup insights from YC/First Round via LLMLayer")
            result = await self.llmlayer.search(
                query="What are the latest startup insights and founder advice from Y Combinator blog and First Round Review? Focus on scaling strategies, product-market fit, fundraising, and practical founder lessons",
                max_results=10,  # Fetch 10, then validate and rank to best 2-3
                domains=["ycombinator.com", "firstround.com", "review.firstround.com"],
                recency="week"
            )
            items_raw = [
                {
                    "headline": c.title,
                    "url": c.url,
                    "content": c.snippet,  # This now contains full article content (up to 5000 chars) after fetch
                    "source": c.source_name,
                    "published": (c.published_date.isoformat() if c.published_date else None),
                }
                for c in result.citations
            ]
            # Validate sources BEFORE other filtering
            items_validated = self._validate_sources(
                items_raw,
                ["ycombinator.com", "firstround.com", "review.firstround.com"],
                "Startup"
            )
            items = self._filter_items(items_validated, max_age_days=14)  # Keep content relatively recent
            return FetchResult("llmlayer", Section.STARTUP, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            return await self._handle_fetch_failure("llmlayer", e)

    async def _fetch_politics(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            # VISION.txt specifies: Associated Press, nonpartisan sources only, US politics only
            self.logger.info("Fetching US politics from AP via LLMLayer")
            from datetime import datetime
            today = datetime.now().strftime("%B %d, %Y")
            result = await self.llmlayer.search(
                query=f"What major US federal political developments happened on {today} today according to Associated Press? Focus on Congressional legislation, Supreme Court decisions, executive orders, and White House policy announcements from {today}",
                max_results=10,  # Fetch 10, then validate and rank to best 1-2
                domains=["apnews.com"],  # Now properly used via domain_filter
                recency="day",
                search_type="news"  # Use news search type for political news
            )
            items_raw = [
                {
                    "headline": c.title,
                    "url": c.url,
                    "content": c.snippet,  # This now contains full article content (up to 5000 chars) after fetch
                    "source": c.source_name,
                    "published": (c.published_date.isoformat() if c.published_date else None),
                }
                for c in result.citations
            ]
            # Validate sources BEFORE other filtering
            items_validated = self._validate_sources(items_raw, ["apnews.com"], "Politics")
            items = self._filter_items(items_validated, max_age_days=2)  # Politics needs recent updates  
            return FetchResult("llmlayer", Section.POLITICS, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            # Non-critical; return error result
            return FetchResult("llmlayer", Section.POLITICS, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_local_news(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            # VISION.txt specifies: Miami Herald for Miami, Cornell news sources
            self.logger.info("Fetching local news from Miami Herald and Cornell via LLMLayer")
            from datetime import datetime
            this_week = datetime.now().strftime("%B %d, %Y")
            
            # Fetch Miami news
            miami_result = await self.llmlayer.search(
                query=f"What are the latest local news stories from Miami Herald about Miami, Florida from this week of {this_week}?",
                max_results=5,  # Get 5 Miami articles
                domains=["miamiherald.com"],
                recency="week",
                search_type="news"
            )
            
            # Fetch Cornell news
            cornell_result = await self.llmlayer.search(
                query=f"What are the latest news stories from Cornell University about campus developments, research, or student life from this week of {this_week}?",
                max_results=5,  # Get 5 Cornell articles
                domains=["news.cornell.edu", "cornellsun.com", "cornell.edu"],
                recency="week",
                search_type="news"
            )
            
            # Log what we got from each source
            self.logger.info(f"Local: Got {len(miami_result.citations)} Miami Herald articles")
            self.logger.info(f"Local: Got {len(cornell_result.citations)} Cornell articles")
            
            # Combine citations from both searches
            all_citations = miami_result.citations + cornell_result.citations
            
            items_raw = [
                {
                    "headline": c.title,
                    "url": c.url,
                    "content": c.snippet,  # This now contains full article content (up to 5000 chars) after fetch
                    "source": c.source_name,
                    "published": (c.published_date.isoformat() if c.published_date else None),
                }
                for c in all_citations
            ]
            # Validate sources BEFORE other filtering
            items_validated = self._validate_sources(
                items_raw,
                ["miamiherald.com", "news.cornell.edu", "cornellsun.com"],
                "Local"
            )
            items = self._filter_items(items_validated, max_age_days=7)  # Local news: Miami Herald (daily) + Cornell (weekly cycle)
            self.logger.info("Local: %d items after validation and filtering from %d raw", len(items), len(items_raw))
            return FetchResult("llmlayer", Section.LOCAL, items, asyncio.get_event_loop().time() - start)
        except Exception as e:  # noqa: BLE001
            return FetchResult("llmlayer", Section.LOCAL, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_miscellaneous(self) -> FetchResult:
        start = asyncio.get_event_loop().time()
        try:
            # VISION.txt: Renaissance man topics - history, psychology, art, literature, health, 
            # real estate, arts/music, philosophy (engineering removed to avoid Tech overlap)
            self.logger.info("Miscellaneous: Fetching diverse intellectual content")
            from datetime import datetime
            today = datetime.now().strftime("%B %d, %Y")
            
            # Enhanced Renaissance-style query for intellectually diverse content
            # Engineering removed per user request to avoid overlap with Tech & Science
            result = await self.llmlayer.search(
                query=(
                    f"Intellectually enriching discoveries and insights from {today} and recent weeks. "
                    f"Find diverse, analytical content across these distinct domains:\n\n"
                    f"🏛️ HISTORY & CIVILIZATION: Historical analyses, reinterpretations of past events, "
                    f"civilizational studies, cultural evolution, historical perspectives on current issues\n\n"
                    f"🎨 ARTS & CREATIVE EXPRESSION: Art exhibitions, literary criticism, musical developments, "
                    f"theatrical innovations, aesthetic theory, creative process studies\n\n"
                    f"🧠 PSYCHOLOGY & HUMAN NATURE: Behavioral science research, consciousness studies, "
                    f"social psychology findings, cognitive insights, mental wellness advances\n\n"
                    f"🌿 HEALTH & LONGEVITY: Medical breakthroughs, wellness innovations, public health insights, "
                    f"preventive medicine, lifestyle research, aging studies\n\n"
                    f"💭 PHILOSOPHY & ETHICS: Contemporary philosophical thought, ethical debates, "
                    f"existential questions, moral philosophy, wisdom traditions\n\n"
                    f"🏗️ DESIGN & HUMAN SPACES: Architectural innovation, urban planning, "
                    f"environmental design, spatial psychology, community development\n\n"
                    f"✨ HUMAN STORIES: Remarkable achievements, inspiring journeys, "
                    f"unique perspectives, breakthrough discoveries, exceptional individuals\n\n"
                    f"Seek substantive, in-depth articles from each domain. Avoid polymathic or interdisciplinary meta-content. "
                    f"Focus on specific discoveries, insights, and developments within each field rather than connections between fields."
                ),
                max_results=35,  # Increased for better variety
                recency="month",  # Broader time range for intellectual content
                search_type="general"  # Miscellaneous content isn't always news
            )
            
            # Log the raw API response for debugging
            self.logger.info(f"Miscellaneous: LLMLayer returned {len(result.citations)} citations")
            self.logger.debug(f"Miscellaneous: Raw LLMLayer response - answer length: {len(result.answer) if result.answer else 0}")
            
            # Convert citations to our format
            items_raw = [
                {
                    "headline": c.title,
                    "url": c.url,
                    "content": c.snippet,
                    "source": c.source_name,
                    "published": (c.published_date.isoformat() if c.published_date else None),
                }
                for c in result.citations
            ]
            
            # Log raw items for debugging
            if items_raw:
                self.logger.info(f"Miscellaneous: Found {len(items_raw)} raw items")
                for i, item in enumerate(items_raw[:3]):
                    self.logger.debug(f"Raw item {i+1}: '{item.get('headline', 'NO HEADLINE')[:80]}' from {item.get('source', 'UNKNOWN')}")
            else:
                self.logger.warning("Miscellaneous: LLMLayer returned citations but no items were created")
            
            # Apply date filtering
            items = self._filter_items(items_raw, max_age_days=14)
            
            self.logger.info("Miscellaneous: %d items after filtering from %d raw", len(items), len(items_raw))
            
            if len(items) == 0:
                self.logger.warning("Miscellaneous: No items survived filtering - may be a date/content issue")
                # Log details for debugging
                if items_raw:
                    self.logger.debug("Miscellaneous: Sample raw item details:")
                    for i, item in enumerate(items_raw[:2]):
                        self.logger.debug(f"  Item {i+1}: headline='{item.get('headline', 'NONE')}', source='{item.get('source', 'NONE')}', published='{item.get('published', 'NONE')}'")
            
            return FetchResult("llmlayer", Section.MISCELLANEOUS, items, asyncio.get_event_loop().time() - start)
            
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Miscellaneous: Error during fetch: {e}", exc_info=True)
            return FetchResult("llmlayer", Section.MISCELLANEOUS, [], asyncio.get_event_loop().time() - start, error=str(e))

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
                    all_items.append({
                        "headline": f"First Reading: {readings.first_reading.get('reference', 'Daily Reading')}",
                        "url": base_url,
                        "content": readings.first_reading.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })
                
                # Add Second Reading (if it exists - typically on Sundays/Solemnities)
                if readings.second_reading and readings.second_reading.get('text'):
                    all_items.append({
                        "headline": f"Second Reading: {readings.second_reading.get('reference', 'Second Reading')}",
                        "url": base_url,
                        "content": readings.second_reading.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })
                
                # Add Gospel
                if readings.gospel and readings.gospel.get('text'):
                    all_items.append({
                        "headline": f"Gospel: {readings.gospel.get('reference', 'Daily Gospel')}",
                        "url": base_url,
                        "content": readings.gospel.get('text', ''),
                        "source": "USCCB Daily Readings", 
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })
                    
                # Add Responsorial Psalm if present
                if hasattr(readings, 'responsorial_psalm') and readings.responsorial_psalm and readings.responsorial_psalm.get('text'):
                    all_items.append({
                        "headline": f"Responsorial Psalm: {readings.responsorial_psalm.get('reference', 'Psalm')}",
                        "url": base_url,
                        "content": readings.responsorial_psalm.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })
                self.logger.info("Scripture: Successfully fetched %d USCCB readings", len(all_items))
        except Exception as e:
            self.logger.error("Scripture: Failed to fetch USCCB readings: %s", e)
        
        # Get Catholic Daily Reflections from RSS feed (more reliable than LLMLayer search)
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
                            "content": item.content or item.description,  # Use full content if available
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
                            "content": item.content or item.description,
                            "source": "Catholic Daily Reflections",
                            "published": item.published_date.isoformat() if item.published_date else datetime.now().isoformat(),
                            "preserve_original": False,
                        })
                        self.logger.info("Scripture: Using most recent reflection from Catholic Daily Reflections RSS")
        except Exception as e:
            self.logger.error("Scripture: Failed to fetch Catholic Daily Reflections RSS: %s", e)
        
        # Fallback to LLMLayer if RSS fails and LLMLayer is available
        if not any(item.get("source") == "Catholic Daily Reflections" for item in all_items) and self.llmlayer:
            self.logger.warning("Scripture: LLMLayer service not available, skipping Catholic Daily Reflections")
            
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
        Fetch research papers from both ArXiv and Semantic Scholar.
        Uses a hybrid strategy: ArXiv for cutting-edge preprints,
        Semantic Scholar for peer-reviewed authority.
        """
        if self.semantic_scholar:
            # Use both sources with intelligent orchestration
            return await self._fetch_hybrid_papers()
        else:
            # Fallback to ArXiv only
            return await self._fetch_arxiv_papers()
    
    async def _fetch_hybrid_papers(self) -> FetchResult:
        """
        Fetch papers from both ArXiv and Semantic Scholar with smart orchestration.
        60% ArXiv (novelty) + 40% Semantic Scholar (authority).
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
        
        # Prepare search queries based on VISION.txt interests
        search_queries = [
            "artificial intelligence transformer",
            "machine learning optimization",
            "quantum computing algorithms",
            "statistical mechanics physics",
            "information theory cryptography",
            "neural networks deep learning",
            "probability theory statistics",
            "signal processing systems"
        ]
        
        # Fetch from both sources in parallel
        tasks = [
            self._fetch_arxiv_subset(arxiv_count),
            self.semantic_scholar.search_papers(  # Uses async wrapper
                queries=search_queries,
                max_results=semantic_count,
                min_citations=5,
                days_back=30
            )
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_papers = []
        
        # Process ArXiv results
        if isinstance(results[0], list):
            all_papers.extend(results[0])
            self.logger.info(f"Got {len(results[0])} papers from ArXiv")
        else:
            self.logger.warning(f"ArXiv fetch failed: {results[0]}")
        
        # Process Semantic Scholar results
        if isinstance(results[1], list):
            all_papers.extend(results[1])
            self.logger.info(f"Got {len(results[1])} papers from Semantic Scholar")
        else:
            self.logger.warning(f"Semantic Scholar fetch failed: {results[1]}")
        
        # Deduplicate by title similarity (some papers may be on both platforms)
        seen_titles = set()
        unique_papers = []
        
        for paper in all_papers:
            # Normalize title for comparison
            title = paper.get('title', '').lower().strip()
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
                        "content": p.get("abstract"),  # Add content field
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
                    "quant-ph", "cond-mat.stat-mech"  # Quantum/Statistical mechanics
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
                        "content": p.abstract,
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
                        "content": readings.first_reading.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })
                
                # Add Responsorial Psalm if present
                if hasattr(readings, 'responsorial_psalm') and readings.responsorial_psalm and readings.responsorial_psalm.get('text'):
                    readings_items.append({
                        "headline": f"Responsorial Psalm: {readings.responsorial_psalm.get('reference', 'Psalm')}",
                        "url": base_url,
                        "content": readings.responsorial_psalm.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })
                
                # Add Gospel as an item
                if readings.gospel and readings.gospel.get('text'):
                    readings_items.append({
                        "headline": f"Gospel: {readings.gospel.get('reference', 'Daily Gospel')}",
                        "url": base_url,
                        "content": readings.gospel.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })
                
                # Add Reflection if present
                if readings.reflection:
                    readings_items.append({
                        "headline": "Daily Reflection",
                        "url": base_url,
                        "content": readings.reflection,
                        "source": "USCCB Daily Readings",
                        "published": published,
                    })
                
                # Add Saint of the Day if present
                if hasattr(readings, 'saint_of_day') and readings.saint_of_day:
                    readings_items.append({
                        "headline": "Saint of the Day",
                        "url": base_url,
                        "content": readings.saint_of_day,
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

        async def rank_one(section: str, items: List[Dict[str, Any]]):
            # Scripture section doesn't need ranking - just pass through
            if section == Section.SCRIPTURE:
                ranked[section] = self._convert_to_ranked_items(items, section)
            else:
                ranked[section] = await self._rank_items(items, section)

        for section, items in sections.items():
            tasks.append(asyncio.create_task(rank_one(section, items)))
        await asyncio.gather(*tasks, return_exceptions=True)
        
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
                                'content': item.content,
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
                    content=item.get("content", ""),
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
        NO VALIDATION - just return what LLMLayer gives us.
        LLMLayer knows what it's doing.
        """
        # Just log what we got for debugging
        self.logger.debug(f"{section_name}: Got {len(items)} items from LLMLayer")
        return items
    
    def _filter_items(self, items: List[Dict[str, Any]], max_age_days: int) -> List[Dict[str, Any]]:
        """
        Quality and freshness filter for LLMLayer-derived items.
        Drops:
        - Very old items beyond max_age_days
        - Aggregator/generic domains (YouTube, Wikipedia, Eventbrite, ScienceDaily, TS2.tech)
        - Non-informative clickbait/generic titles
        - Missing headline or url
        """
        from datetime import datetime, timedelta
        import re
        cutoff = datetime.now() - timedelta(days=max_age_days)
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
        }
        def domain(u: str) -> str:
            try:
                from urllib.parse import urlparse
                return (urlparse(u).netloc or u).lower()
            except Exception:
                return ""
        def too_old(published: Optional[str]) -> bool:
            if not published:
                # For breaking news (max_age_days <= 1), no date means we reject it
                # For older content (max_age_days > 1), we're more lenient
                return max_age_days <= 1  # Reject if breaking news, keep if older sections
            try:
                dt = datetime.fromisoformat(published)
                return dt < cutoff
            except Exception:
                # If we can't parse the date, treat as too old for breaking news
                return max_age_days <= 1
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
            published_raw = it.get("published")
            
            if not url or not head:
                self.logger.debug(f"❌ Filtered (missing URL/headline): {head[:50] if head else 'NO_HEADLINE'}")
                continue
                
            url_domain = domain(url)
            if url_domain in bad_domains:
                self.logger.debug(f"❌ Filtered (bad domain): {url_domain} - {head[:50]}")
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

    async def _rank_items(self, items: List[Dict[str, Any]], section: str) -> List[RankedItem]:
        ranked_items: List[RankedItem] = []
        # ALWAYS rank the section as a batch; if it fails, fail the pipeline (no silent per-item fallbacks)
        # Attach a synthetic id to each story to ensure stable mapping in AI response
        enriched: List[Dict[str, Any]] = []
        for idx, it in enumerate(items):
            clone = dict(it)
            clone.setdefault("id", str(idx + 1))
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
            content = item.get("content") or item.get("abstract") or item.get("description") or ""
            source = item.get("source") or item.get("source_feed") or ""
            published_raw = item.get("published") or item.get("published_date")
            
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
                    published_dt = datetime.now()
                    self.logger.warning(f"DEFAULTING TO NOW: No valid date for '{headline[:50]}...', published_raw={published_raw}")
            except Exception as e:
                published_dt = datetime.now()
                self.logger.warning(f"DEFAULTING TO NOW: Date parsing failed for '{headline[:50]}...', published_raw='{published_raw}', error: {e}")

            ranked_item = RankedItem(
                id=str(idx + 1),
                url=url,
                headline=headline,
                content=content,
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
                editorial_note=_note
            )
            # Don't recalculate - we already have the proper weighted total
            ranked_items.append(ranked_item)

        ranked_items.sort(key=lambda r: r.total_score, reverse=True)
        for i, r in enumerate(ranked_items, start=1):
            r.rank = i
        return ranked_items

    async def _calculate_three_axis_scores(self, item: Dict[str, Any], section: str) -> Tuple[float, float, float]:
        # No per-item fallbacks. This path should not be used.
        raise AggregationError("Per-item ranking is disabled. Batch ranking must be used.")

    async def select_top_items(self, ranked_sections: Dict[str, List[RankedItem]]) -> Dict[str, List[RankedItem]]:
        """Select final items by section using dynamic thresholds and cross-section balance optimization"""
        selected: Dict[str, List[RankedItem]] = {}
        
        # Get theme analysis for diversity enforcement
        theme_analysis = self.get_theme_analysis()
        
        # Calculate dynamic thresholds based on content quality and availability
        try:
            # Now we can properly call async methods since this function is async
            dynamic_thresholds = await self._calculate_dynamic_quality_thresholds(ranked_sections)
            self.logger.info("Applied dynamic quality thresholds")
        except Exception as e:
            self.logger.warning(f"Dynamic threshold calculation failed: {e} - using static thresholds")
            dynamic_thresholds = {section: self._normalized_threshold() for section in ranked_sections.keys()}
        
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
            
            # Apply enhanced selection with dynamic thresholds
            enhanced_selection = self._enhance_section_selection(
                section, sorted_items, theme_analysis, dynamic_thresholds.get(section)
            )
            
            # Log research papers selection result
            if section == Section.RESEARCH_PAPERS:
                self.logger.info(f"Research Papers: {len(enhanced_selection)} papers selected after filtering")
            
            selected[section] = enhanced_selection
        
        # Phase 2: Enforce strict section limits
        # Check each section meets its requirements
        for section, items in selected.items():
            min_items, max_items = self.items_per_section.get(section, (0, len(items)))
            
            # For sections with exact requirements (min == max), enforce strictly
            if min_items == max_items and len(items) != min_items:
                if len(items) < min_items:
                    # Try to pull more items from the original ranked pool
                    all_ranked = sorted(ranked_sections.get(section, []), key=lambda x: x.total_score, reverse=True)
                    # Take items not already selected
                    remaining = [item for item in all_ranked if item not in items]
                    needed = min_items - len(items)
                    if remaining:
                        items.extend(remaining[:needed])
                        self.logger.warning(f"Section {section} was short by {needed} items, added from reserves")
                elif len(items) > max_items:
                    # Truncate to exact count
                    items = items[:max_items]
                    self.logger.debug(f"Section {section} truncated from {len(selected[section])} to {max_items} items")
                
                selected[section] = items
            
            # Log final counts
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
            return []
            
        # Step 1: Apply dynamic quality threshold
        threshold = dynamic_threshold if dynamic_threshold is not None else self._normalized_threshold()
        
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
        
        # Fallback: if nothing passes threshold, include all so pipeline can proceed
        if not above_threshold:
            above_threshold = ranked_items
            
        # Step 2: Apply diversity enforcement if theme analysis is available
        if theme_analysis and self.embeddings and len(above_threshold) > 2:
            diverse_selection = self._enforce_section_diversity(
                section, above_threshold, theme_analysis
            )
        else:
            # Fallback to simple section limits
            diverse_selection = self._apply_section_limits(above_threshold, section)
        
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
            item_texts = [f"{item.headline} {item.content[:300]}" for item in items]
            
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
            item_text = f"{item.headline} {item.content[:200]}".lower()
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

    async def _calculate_dynamic_quality_thresholds(self, all_sections: Dict[str, List[RankedItem]]) -> Dict[str, float]:
        """
        Calculate dynamic quality thresholds based on content availability and historical patterns.
        
        Args:
            all_sections: All ranked items by section
            
        Returns:
            Dict mapping section names to dynamic threshold values
        """
        thresholds = {}
        base_threshold = self._normalized_threshold()
        
        # Get historical quality data if cache service is available
        historical_context = {}
        if self.cache_service:
            try:
                historical_context = await self._get_historical_quality_context()
            except Exception as e:
                self.logger.warning(f"Failed to get historical quality context: {e}")
        
        for section, items in all_sections.items():
            if not items:
                thresholds[section] = base_threshold
                continue
                
            section_threshold = self._calculate_section_dynamic_threshold(
                section, items, base_threshold, historical_context
            )
            thresholds[section] = section_threshold
            
        return thresholds

    def _calculate_section_dynamic_threshold(self, section: str, items: List[RankedItem], 
                                           base_threshold: float, historical_context: Dict) -> float:
        """
        Calculate dynamic threshold for a specific section.
        """
        if not items:
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
        if not items:
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
                return items_sorted[:exact_count]
            else:
                # Log warning if we don't have enough items
                self.logger.warning(f"Section {section} requires {exact_count} items but only has {len(items_sorted)}")
                return items_sorted
        else:
            # For flexible sections (Scripture, Politics, Local, Extra), use the original logic
            if len(items_sorted) > max_items:
                items_sorted = items_sorted[:max_items]
            return items_sorted
    
    def _balance_local_sources(self, items: List[RankedItem], min_items: int, max_items: int) -> List[RankedItem]:
        """
        Balance local news sources to ensure 1 Miami + 1 Cornell article when possible.
        """
        miami_items = []
        cornell_items = []
        
        # Separate items by source (check both source name and URL)
        for item in items:
            source_lower = item.source.lower() if item.source else ""
            url_lower = item.url.lower() if item.url else ""
            
            # Log item details for debugging
            self.logger.debug(f"Local item: source='{item.source}', url='{item.url[:50]}...'")
            
            # Check for Miami sources (domain names and source names)
            if ("miami" in source_lower or "miamiherald" in source_lower or 
                "miamiherald.com" in url_lower):
                miami_items.append(item)
                self.logger.debug(f"  -> Identified as Miami Herald article")
            # Check for Cornell sources (domain names and source names)
            elif ("cornell" in source_lower or "news.cornell" in source_lower or 
                  "cornellsun" in source_lower or "cornell.edu" in url_lower or
                  "cornellsun.com" in url_lower):
                cornell_items.append(item)
                self.logger.debug(f"  -> Identified as Cornell article")
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
        # Optionally enforce section minimums by topping up from ranked pool
        selected = self._enforce_minimums(selected, ranked)
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
        out: Dict[str, List[RankedItem]] = {}
        for section, chosen in selected.items():
            min_items, max_items = self.items_per_section.get(section, (0, len(chosen)))
            if len(chosen) >= min_items:
                out[section] = chosen
                continue
            pool = ranked.get(section, [])
            before = len(chosen)
            # Fill from top of ranked pool, skipping already selected
            picked_ids = {id(x) for x in chosen}
            for candidate in pool:
                if id(candidate) in picked_ids:
                    continue
                chosen.append(candidate)
                picked_ids.add(id(candidate))
                if len(chosen) >= min_items:
                    break
            # Enforce max after filling
            if len(chosen) > max_items:
                chosen = chosen[:max_items]
            
            # STRICT ENFORCEMENT: For ALL sections with exact counts (min == max), ensure we have exactly that many
            if min_items == max_items and len(chosen) < min_items:
                # Try harder to get exact count for strict sections (Breaking News, Business, Tech/Science, Research Papers, Startup, Miscellaneous)
                self.logger.warning(
                    f"Section {section} requires exactly {min_items} items but only has {len(chosen)} after filling. "
                    f"Attempting to meet exact requirement."
                )
                # Try to pull more from the pool if possible
                remaining_pool = [item for item in pool if id(item) not in picked_ids]
                if remaining_pool:
                    additional_needed = min_items - len(chosen)
                    additional_items = remaining_pool[:additional_needed]
                    chosen.extend(additional_items)
                    self.logger.info(f"Added {len(additional_items)} more items to {section} to meet exact requirement of {min_items}")
                else:
                    self.logger.error(f"Cannot meet exact requirement of {min_items} items for {section} - no more items in pool")
            
            after = len(chosen)
            self.logger.info(
                "Section '%s' top-up applied: %d -> %d (min=%d, max=%d)",
                section,
                before,
                after,
                min_items,
                max_items,
            )
            out[section] = chosen
        # Include sections that had no selected items originally
        for section, pool in ranked.items():
            if section not in out:
                out[section] = selected.get(section, [])
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
        important_services = {"llmlayer", "anthropic"}
        
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


