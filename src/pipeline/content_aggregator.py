import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.services.arxiv import ArxivService, ArxivPaper
from src.services.rss import RSSService, DailyReading
from src.services.ai_service import AIService, RankingResult
from src.services.semantic_scholar_service import SemanticScholarService
from src.services.source_ranking_service import SourceRankingService


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
    RSS-based content fetching and ranking system for the newsletter.
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
    ) -> None:
        self.arxiv = arxiv
        self.rss = rss
        self.ai = ai
        self.cache_service = cache_service
        self.embeddings = embeddings

        # Initialize source ranking service if not provided
        self.source_ranker = source_ranker or SourceRankingService()
        self.source_ranking_service = self.source_ranker
        self.source_ranking_config = self.source_ranker.authority_config if self.source_ranker else {}
        self.semantic_scholar = semantic_scholar

        self.parallel_limit = 10
        self.fetch_timeout = 300  # 5 minutes for RSS feeds (much faster than llmlayer)
        self.min_score_threshold = 12  # Lowered from 15 to include more research papers

        # Section quotas (min, max)
        self.section_quotas = {
            Section.BREAKING_NEWS: (3, 3),     # Exactly 3
            Section.BUSINESS: (3, 3),          # Exactly 3  
            Section.TECH_SCIENCE: (3, 3),      # Exactly 3
            Section.RESEARCH_PAPERS: (5, 5),   # Exactly 5
            Section.STARTUP: (2, 2),           # Exactly 2
            Section.SCRIPTURE: (6, 10),        # Keep flexible for Scripture
            Section.POLITICS: (2, 2),          # Exactly 2 per vision
            Section.LOCAL: (2, 2),             # Exactly 2 (1 Miami + 1 Cornell when possible)
            Section.MISCELLANEOUS: (5, 5),     # Exactly 5
            Section.EXTRA: (0, 2),             # 0-2 flexible
        }

        self.logger = logging.getLogger(__name__)
        self._fetch_stats: List[FetchResult] = []

    async def fetch_all_content(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch content from all sources in parallel using RSS feeds."""
        tasks: List[asyncio.Task] = []

        # Create tasks for each section
        async def fetch_rss_section_with_timeout(section: str):
            start = asyncio.get_event_loop().time()
            try:
                result = await asyncio.wait_for(self._fetch_rss_section(section), timeout=self.fetch_timeout)
                if isinstance(result, FetchResult):
                    self._fetch_stats.append(result)
                return result
            except asyncio.TimeoutError:
                self.logger.error(f"RSS section {section} timed out after {self.fetch_timeout}s - returning partial results")
                return FetchResult("rss", section, [], self.fetch_timeout, error="Timeout")
            except Exception as e:
                self.logger.error(f"RSS section {section} failed: {e} - returning partial results")
                return FetchResult("rss", section, [], 0.0, error=str(e))

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

        # Add tasks for each section
        tasks.extend([
            asyncio.create_task(fetch_rss_section_with_timeout(Section.BREAKING_NEWS)),
            asyncio.create_task(fetch_rss_section_with_timeout(Section.BUSINESS)),
            asyncio.create_task(fetch_rss_section_with_timeout(Section.TECH_SCIENCE)),
            asyncio.create_task(fetch_rss_section_with_timeout(Section.POLITICS)),
            asyncio.create_task(fetch_rss_section_with_timeout(Section.MISCELLANEOUS)),
            asyncio.create_task(fetch_research_with_timeout()),
            asyncio.create_task(fetch_scripture_with_timeout()),
        ])

        results = await asyncio.gather(*tasks, return_exceptions=True)

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

        return sections

    async def _fetch_rss_section(self, section: str) -> FetchResult:
        """Fetch RSS feeds for a specific section."""
        start = asyncio.get_event_loop().time()
        try:
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

    def _expand_scripture_reference(self, reference: str) -> str:
        """Expand scripture abbreviations to full book names."""
        expansions = {
            "Gn": "Genesis", "Ex": "Exodus", "Lv": "Leviticus", "Nm": "Numbers", "Dt": "Deuteronomy",
            "Jos": "Joshua", "Jgs": "Judges", "Ru": "Ruth", "1 Sm": "1 Samuel", "2 Sm": "2 Samuel",
            "1 Kgs": "1 Kings", "2 Kgs": "2 Kings", "1 Chr": "1 Chronicles", "2 Chr": "2 Chronicles",
            "Ezr": "Ezra", "Neh": "Nehemiah", "Tb": "Tobit", "Jdt": "Judith", "Est": "Esther",
            "1 Mc": "1 Maccabees", "2 Mc": "2 Maccabees", "Jb": "Job", "Ps": "Psalm", "Prv": "Proverbs",
            "Eccl": "Ecclesiastes", "Sg": "Song of Songs", "Wis": "Wisdom", "Sir": "Sirach",
            "Is": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations", "Bar": "Baruch", "Ez": "Ezekiel",
            "Dn": "Daniel", "Hos": "Hosea", "Jl": "Joel", "Am": "Amos", "Ob": "Obadiah",
            "Jon": "Jonah", "Mi": "Micah", "Na": "Nahum", "Hb": "Habakkuk", "Zep": "Zephaniah",
            "Hg": "Haggai", "Zec": "Zechariah", "Mal": "Malachi", "Mt": "Matthew", "Mk": "Mark",
            "Lk": "Luke", "Jn": "John", "Acts": "Acts", "Rom": "Romans", "1 Cor": "1 Corinthians",
            "2 Cor": "2 Corinthians", "Gal": "Galatians", "Eph": "Ephesians", "Phil": "Philippians",
            "Col": "Colossians", "1 Thes": "1 Thessalonians", "2 Thes": "2 Thessalonians",
            "1 Tm": "1 Timothy", "2 Tm": "2 Timothy", "Ti": "Titus", "Phlm": "Philemon",
            "Heb": "Hebrews", "Jas": "James", "1 Pt": "1 Peter", "2 Pt": "2 Peter",
            "1 Jn": "1 John", "2 Jn": "2 John", "3 Jn": "3 John", "Jude": "Jude", "Rv": "Revelation"
        }
        
        for abbrev, full_name in expansions.items():
            if reference.startswith(abbrev + " "):
                return reference.replace(abbrev, full_name, 1)
        return reference

    async def _fetch_scripture(self) -> FetchResult:
        """Fetch Catholic daily readings and reflections"""
        start = asyncio.get_event_loop().time()
        all_items = []
        
        try:
            self.logger.info("Scripture: Fetching USCCB daily readings from RSS service")
            # Get readings from the RSS service
            readings = await self.rss.get_daily_readings()
            
            if readings:
                self.logger.info("Scripture: Found daily readings for %s", readings.date if hasattr(readings, 'date') else 'today')
                published = readings.date.isoformat() if hasattr(readings, 'date') else datetime.now().isoformat()
                base_url = f"https://bible.usccb.org/bible/readings/{readings.date.month:02d}{readings.date.day:02d}{readings.date.year % 100:02d}.cfm" if hasattr(readings, 'date') else "https://bible.usccb.org/"
                
                # Add First Reading
                if hasattr(readings, 'first_reading') and readings.first_reading and readings.first_reading.get('text'):
                    reference = self._expand_scripture_reference(readings.first_reading.get('reference', 'First Reading'))
                    all_items.append({
                        "headline": f"First Reading: {reference}",
                        "url": base_url,
                        "summary_text": readings.first_reading.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })
                
                # Add Second Reading if present (Sundays and Solemnities)
                if hasattr(readings, 'second_reading') and readings.second_reading and readings.second_reading.get('text'):
                    reference = self._expand_scripture_reference(readings.second_reading.get('reference', 'Second Reading'))
                    all_items.append({
                        "headline": f"Second Reading: {reference}",
                        "url": base_url,
                        "summary_text": readings.second_reading.get('text', ''),
                        "source": "USCCB Daily Readings",
                        "published": published,
                        "preserve_original": True,  # Flag to skip summarization
                    })
                    
                # Add Gospel Reading
                if hasattr(readings, 'gospel') and readings.gospel and readings.gospel.get('text'):
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
        
        # Get Catholic Daily Reflections from RSS feed
        try:
            self.logger.info("Scripture: Fetching daily reflections from Catholic Daily Reflections RSS feed")
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
        
        self.logger.info("Scripture: Total %d items (USCCB + Reflections)", len(all_items))
        return FetchResult("combined", Section.SCRIPTURE, all_items, asyncio.get_event_loop().time() - start)

    async def _fetch_research_papers(self) -> FetchResult:
        """Fetch research papers from both ArXiv and Semantic Scholar."""
        if self.semantic_scholar:
            return await self._fetch_hybrid_papers()
        else:
            return await self._fetch_arxiv_papers()
    
    async def _fetch_hybrid_papers(self) -> FetchResult:
        """Fetch papers from both ArXiv and Semantic Scholar with smart orchestration."""
        start = asyncio.get_event_loop().time()
        all_papers = []
        
        try:
            # Fetch from both sources in parallel
            arxiv_task = asyncio.create_task(self._fetch_arxiv_subset(20))  # Get 20 from ArXiv
            ss_task = asyncio.create_task(self._fetch_semantic_scholar_subset(10))  # Get 10 from SS
            
            arxiv_papers, ss_papers = await asyncio.gather(arxiv_task, ss_task, return_exceptions=True)
            
            # Handle results
            if not isinstance(arxiv_papers, Exception) and arxiv_papers:
                all_papers.extend(arxiv_papers)
            if not isinstance(ss_papers, Exception) and ss_papers:
                all_papers.extend(ss_papers)
            
            self.logger.info(f"Hybrid papers: {len(all_papers)} total papers fetched")
            return FetchResult("hybrid", Section.RESEARCH_PAPERS, all_papers, asyncio.get_event_loop().time() - start)
            
        except Exception as e:
            self.logger.error(f"Hybrid papers fetch failed: {e}")
            return FetchResult("hybrid", Section.RESEARCH_PAPERS, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def _fetch_arxiv_subset(self, max_results: int) -> List[Dict[str, Any]]:
        """Fetch a subset of ArXiv papers."""
        try:
            papers = await self.arxiv.fetch_daily_papers(max_results=max_results)
            items = []
            for paper in papers:
                items.append({
                    "headline": paper.title,
                    "url": paper.url,
                    "summary_text": paper.abstract,
                    "source": "arXiv",
                    "published": paper.published_date.isoformat(),
                    "published_date": paper.published_date,
                    "authors": paper.authors,
                    "categories": paper.categories,
                })
            return items
        except Exception as e:
            self.logger.error(f"ArXiv subset fetch failed: {e}")
            return []

    async def _fetch_semantic_scholar_subset(self, max_results: int) -> List[Dict[str, Any]]:
        """Fetch a subset of papers from Semantic Scholar."""
        try:
            if not self.semantic_scholar:
                return []
            papers = await self.semantic_scholar.fetch_recent_papers(limit=max_results)
            items = []
            for paper in papers:
                items.append({
                    "headline": paper.get("title", "Untitled Paper"),
                    "url": paper.get("url", ""),
                    "summary_text": paper.get("abstract", ""),
                    "source": "Semantic Scholar",
                    "published": paper.get("publicationDate", datetime.now().isoformat()),
                    "published_date": datetime.fromisoformat(paper.get("publicationDate", datetime.now().isoformat())),
                    "authors": [author.get("name", "") for author in paper.get("authors", [])],
                    "citation_count": paper.get("citationCount", 0),
                })
            return items
        except Exception as e:
            self.logger.error(f"Semantic Scholar subset fetch failed: {e}")
            return []

    async def _fetch_arxiv_papers(self) -> FetchResult:
        """Fallback to ArXiv only if Semantic Scholar is unavailable."""
        start = asyncio.get_event_loop().time()
        try:
            papers = await self.arxiv.fetch_daily_papers(max_results=30)
            
            items = []
            for paper in papers:
                items.append({
                    "headline": paper.title,
                    "url": paper.url,
                    "summary_text": paper.abstract,
                    "source": "arXiv",
                    "published": paper.published_date.isoformat(),
                    "published_date": paper.published_date,
                    "authors": paper.authors,
                    "categories": paper.categories,
                })
            
            self.logger.info(f"ArXiv: fetched {len(items)} papers")
            return FetchResult("arxiv", Section.RESEARCH_PAPERS, items, asyncio.get_event_loop().time() - start)
            
        except Exception as e:
            self.logger.error(f"ArXiv papers fetch failed: {e}")
            return FetchResult("arxiv", Section.RESEARCH_PAPERS, [], asyncio.get_event_loop().time() - start, error=str(e))

    async def rank_all_content(self, sections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[RankedItem]]:
        """Rank all content using AI service."""
        ranked_sections = {}
        
        async def rank_one(section: str, items: List[Dict[str, Any]]):
            if not items:
                ranked_sections[section] = []
                return
                
            try:
                # Scripture section doesn't need ranking - just pass through
                if section == Section.SCRIPTURE:
                    ranked_sections[section] = self._convert_to_ranked_items(items, section)
                    return
                
                # Use AI service to rank the items
                ranking_result = await self.ai.rank_content(items, section)
                
                if ranking_result and hasattr(ranking_result, 'ranked_items'):
                    ranked_sections[section] = ranking_result.ranked_items
                else:
                    # Fallback to unranked conversion
                    ranked_sections[section] = self._convert_to_ranked_items(items, section)
                    
            except Exception as e:
                self.logger.error(f"Ranking failed for {section}: {e}")
                # Fallback to unranked items
                ranked_sections[section] = self._convert_to_ranked_items(items, section)
        
        tasks = [rank_one(section, items) for section, items in sections.items()]
        await asyncio.gather(*tasks)
        
        return ranked_sections

    def _convert_to_ranked_items(self, items: List[Dict[str, Any]], section: str) -> List[RankedItem]:
        """Convert items directly to RankedItems without scoring."""
        ranked_items = []
        
        for idx, item in enumerate(items):
            try:
                # Parse published date
                published_date = item.get("published_date")
                if isinstance(published_date, str):
                    try:
                        published_date = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                    except:
                        published_date = datetime.now()
                elif not isinstance(published_date, datetime):
                    published_date = datetime.now()
                
                ranked_item = RankedItem(
                    id=f"{section}_{idx}",
                    headline=item.get("headline", "Untitled"),
                    url=item.get("url", ""),
                    source=item.get("source", "Unknown"),
                    summary_text=item.get("summary_text", ""),
                    section=section,
                    published_date=published_date,
                    # Default scores - will be properly set by AI ranking if used
                    temporal_impact=5.0,
                    intellectual_novelty=5.0,
                    renaissance_breadth=5.0,
                    actionable_wisdom=5.0,
                    source_authority=5.0,
                    signal_clarity=5.0,
                    transformative_potential=5.0,
                    total_score=5.0,
                    preserve_original=item.get("preserve_original", False),
                )
                ranked_items.append(ranked_item)
            except Exception as e:
                self.logger.error(f"Failed to convert item in {section}: {e}")
                continue
                
        return ranked_items

    async def select_top_items(self, ranked_sections: Dict[str, List[RankedItem]]) -> Dict[str, List[RankedItem]]:
        """Select top items from each section according to quotas."""
        selected = {}
        
        for section, items in ranked_sections.items():
            if section in self.section_quotas:
                min_quota, max_quota = self.section_quotas[section]
                # For exact quotas (min == max), use that number
                if min_quota == max_quota:
                    selected_count = min_quota
                else:
                    # For flexible quotas, use all available items up to max
                    selected_count = min(len(items), max_quota)
                
                selected[section] = items[:selected_count]
                self.logger.info(f"Selected {len(selected[section])} items from {section} (quota: {min_quota}-{max_quota})")
            else:
                # Default: take all items
                selected[section] = items
                
        return selected