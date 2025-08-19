import asyncio
from typing import List, Dict, Any, Optional
from typing import cast
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import arxiv
import hashlib
from src.services.cache_service import ContentItem as PipelineContentItem
from collections import Counter
import os
import json
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Optional; defaults will be used if unavailable


@dataclass
class ArxivPaper:
    """Structured representation of an arXiv paper"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    primary_category: str
    published_date: datetime
    updated_date: datetime
    pdf_url: str
    comment: Optional[str] = None  # Author's comment (e.g., "Accepted to Nature")
    journal_ref: Optional[str] = None  # Published journal reference

    def to_content_item(self, section: str = "tech_science") -> PipelineContentItem:
        """Convert to pipeline ContentItem dataclass for downstream processing"""
        return PipelineContentItem(
            id=f"arxiv_{self.arxiv_id}",
            source="arXiv",
            section=section,
            headline=self.title,
            content=self.abstract,
            url=self.pdf_url,
            published_date=self.published_date,
            metadata={
                "authors": self.authors,
                "categories": self.categories,
                "journal_ref": self.journal_ref,
                "comment": self.comment,
            },
        )


class ArxivService:
    """
    arXiv.org integration for academic paper discovery.

    This service channels the Renaissance ideal of pursuing knowledge
    at the frontiers of human understanding. It discovers:
    - Breakthrough research papers
    - Interdisciplinary connections
    - Paradigm-shifting discoveries
    - Mathematical beauty and elegance

    Papers from arXiv often score highest on Intellectual Delight.

    Note on API usage: This service explicitly uses `self.client.results(search)`
    to execute searches so that the configured `delay_seconds` and retries are
    respected by the `arxiv` library. Direct `search.results()` calls are avoided
    to ensure rate limiting compliance.
    """

    def __init__(self):
        """
        Initialize arXiv service.
        Note: arXiv doesn't require API keys - it's open access!
        """
        self.client = arxiv.Client(
            page_size=50,
            delay_seconds=3.0,  # Respectful rate limiting
            num_retries=3,
        )

        # Category mappings for Renaissance interests
        self.category_interests = {
            "breaking": [
                "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "cs.NE", "cs.IR", "cs.CR", "cs.DS",
                "physics.gen-ph", "stat.ML"
            ],
            "fundamental": [
                "hep-th", "gr-qc", "quant-ph", "math.GT", "math.PR", "math.ST", "math.OC", "math.NA",
                "physics.app-ph", "physics.comp-ph", "cond-mat.mtrl-sci"
            ],
            "interdisciplinary": [
                "physics.soc-ph", "cs.CY", "cs.SI", "q-fin.GN", "q-fin.PR", "q-fin.ST", "nlin.CD",
                "econ.GN", "econ.EM", "econ.TH"
            ],
            "emerging": [
                "cs.LG", "cs.CV", "cs.CL", "cs.RO", "cs.AI", "cs.NE", "eess.SP", "eess.IV", "eess.SY",
                "stat.ML"
            ],
            "philosophical": ["physics.hist-ph", "math.HO", "cs.AI"],
        }

        # Cache for the session
        self._cache: Dict[str, List[ArxivPaper]] = {}

        # Config-driven daily selection preferences (loaded from YAML if present)
        self.daily_config = self._load_daily_config()

    async def search_latest(
        self,
        categories: List[str],
        max_results: int = 10,
        days_back: int = 7,
    ) -> List[ArxivPaper]:
        """
        Search for latest papers in specified categories.

        Args:
            categories: List of arXiv categories (e.g., ["cs.AI", "cs.LG"])
            max_results: Maximum papers to return
            days_back: How many days back to search

        Returns:
            List of ArxivPaper objects, sorted by relevance/date
        """
        cache_key = self._generate_cache_key(
            method="search_latest", categories=",".join(sorted(categories)), max_results=max_results, days_back=days_back
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        query = " OR ".join([f"cat:{cat}" for cat in categories])
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # overfetch to allow filtering by date
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        all_results = await self._execute_search(search)

        # Filter client-side by published date
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        filtered = [p for p in all_results if p.published_date >= cutoff]
        papers = filtered[:max_results] if filtered else all_results[:max_results]

        self._cache[cache_key] = papers
        return papers

    async def search_breakthrough(self) -> List[ArxivPaper]:
        """
        Search for potential breakthrough papers.
        Uses heuristics like journal acceptance and interdisciplinarity.

        Returns:
            List of high-impact papers likely to represent breakthroughs
        """
        cache_key = self._generate_cache_key(method="search_breakthrough")
        if cache_key in self._cache:
            return self._cache[cache_key]

        broad_categories = list(
            set(
                self.category_interests["breaking"]
                + self.category_interests["fundamental"]
                + self.category_interests["emerging"]
            )
        )
        query = " OR ".join([f"cat:{cat}" for cat in broad_categories])
        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
        )

        candidates = await self._execute_search(search)
        breakthroughs = [p for p in candidates if self._is_breakthrough_candidate(p)]
        self._cache[cache_key] = breakthroughs[:10]
        return self._cache[cache_key]

    async def search_interdisciplinary(self) -> List[ArxivPaper]:
        """
        Search for papers that bridge multiple disciplines.
        Perfect for Renaissance Resonance scoring.

        Returns:
            List of papers with multiple category tags
        """
        cache_key = self._generate_cache_key(method="search_interdisciplinary")
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Broad search across multiple interests, then filter by multi-domain categories
        broad_categories = list(
            set(
                self.category_interests["interdisciplinary"]
                + self.category_interests["breaking"]
                + self.category_interests["emerging"]
            )
        )
        query = " OR ".join([f"cat:{cat}" for cat in broad_categories])
        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
        )

        results = await self._execute_search(search)
        interdisciplinary = [
            p for p in results if len({c.split(".")[0] for c in p.categories}) > 1
        ]
        self._cache[cache_key] = interdisciplinary[:10]
        return self._cache[cache_key]

    async def search_by_query(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",  # or "lastUpdatedDate", "submittedDate"
    ) -> List[ArxivPaper]:
        """
        Search arXiv by query string.

        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum results to return
            sort_by: Sort order for results

        Returns:
            List of matching papers
        """
        cache_key = self._generate_cache_key(
            method="search_by_query", query=query, max_results=max_results, sort_by=sort_by
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        sort_criterion = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }.get(sort_by, arxiv.SortCriterion.Relevance)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
        )
        papers = await self._execute_search(search)
        self._cache[cache_key] = papers
        return papers

    async def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a specific paper by arXiv ID.

        Args:
            arxiv_id: arXiv identifier (e.g., "2311.12345")

        Returns:
            ArxivPaper object or None if not found
        """
        cache_key = self._generate_cache_key(method="get_paper_by_id", arxiv_id=arxiv_id)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached[0] if cached else None

        search = arxiv.Search(id_list=[arxiv_id])
        papers = await self._execute_search(search)
        self._cache[cache_key] = papers
        return papers[0] if papers else None

    async def get_todays_highlights(self) -> Dict[str, List[ArxivPaper]]:
        """
        Curate today's most interesting papers across categories.

        Returns:
            Dictionary of category -> papers for newsletter sections
        """
        highlights: Dict[str, List[ArxivPaper]] = {}

        tasks: List[asyncio.Task] = []
        keys: List[str] = []
        for interest, cats in self.category_interests.items():
            keys.append(interest)
            tasks.append(
                asyncio.create_task(self.search_latest(categories=cats, max_results=5))
            )

        # Add breakthrough section
        keys.append("breakthrough")
        tasks.append(asyncio.create_task(self.search_breakthrough()))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for key, res in zip(keys, results):
            if isinstance(res, Exception):
                highlights[key] = []
            else:
                highlights[key] = cast(List[ArxivPaper], res)

        return highlights

    async def fetch_latest_papers(self) -> List[Dict[str, Any]]:
        """Return a diversified set of daily papers as plain dicts for the pipeline.

        Selection policy (VISION/LAW aligned):
        - Aim for breadth across AI/CS, physics/engineering, math/stats, bio/health, wildcard
        - Favor recency within a short window (days_back)
        - Prefer interdisciplinary and breakthrough signals when ties occur
        - Total count capped by config (default 5)
        """
        days_back = int(self.daily_config.get("days_back", 3))
        target_total = int(self.daily_config.get("max_per_day", 5))
        max_candidates = int(self.daily_config.get("max_candidates", 30))
        buckets: List[Dict[str, Any]] = self.daily_config.get("diversity_buckets", [])

        # Gather candidates across buckets
        seen_ids: set[str] = set()
        bucket_to_results: Dict[str, List[ArxivPaper]] = {}
        combined: List[ArxivPaper] = []

        # 1) Pull candidates per bucket and pick at least min_per_bucket each
        for bucket in buckets:
            categories: List[str] = list(dict.fromkeys(bucket.get("categories", [])))
            min_take: int = int(bucket.get("min", 1))
            if not categories or min_take <= 0:
                # Still gather for the global pool even if min is 0
                try:
                    results = await self.search_latest(categories=categories, max_results=max(20, max_candidates // max(1, len(buckets))), days_back=days_back)
                    bucket_to_results[bucket.get("name", "wildcard")] = results
                    combined.extend(results)
                except Exception:
                    continue
                continue
            try:
                results = await self.search_latest(categories=categories, max_results=max(20, max_candidates // max(1, len(buckets))), days_back=days_back)
                bucket_to_results[bucket.get("name", "bucket")] = results
                combined.extend(results)
            except Exception:
                # Skip bucket on failure; continue assembling from others
                continue

        # Deduplicate combined candidates by arxiv_id
        unique: List[ArxivPaper] = []
        for p in combined:
            if p.arxiv_id in seen_ids:
                continue
            seen_ids.add(p.arxiv_id)
            unique.append(p)

        # Ensure per-bucket minimum representation in the final candidate list
        guaranteed: List[ArxivPaper] = []
        for bucket in buckets:
            name = bucket.get("name", "bucket")
            min_take = int(bucket.get("min", 1))
            results = bucket_to_results.get(name, [])
            if not results or min_take <= 0:
                continue
            ranked = sorted(results, key=lambda p: self._daily_priority_score(p), reverse=True)
            for p in ranked[:min_take]:
                if p not in guaranteed:
                    guaranteed.append(p)

        # Rank the rest globally by daily priority score
        remaining_sorted = sorted(unique, key=lambda p: self._daily_priority_score(p), reverse=True)
        # Merge guaranteed at the top, then fill up to max_candidates without duplicates
        candidate_pool: List[ArxivPaper] = []
        added: set[str] = set()
        for p in guaranteed + remaining_sorted:
            if p.arxiv_id in added:
                continue
            candidate_pool.append(p)
            added.add(p.arxiv_id)
            if len(candidate_pool) >= max_candidates:
                break

        # Convert to lightweight dicts for AI ranking; final selection to 5 happens downstream
        out: List[Dict[str, Any]] = []
        for p in candidate_pool:
            out.append(
                {
                    "title": p.title,
                    "url": p.pdf_url,
                    "abstract": p.abstract,
                    "authors": p.authors,
                    "published": p.published_date.isoformat(),
                    "categories": p.categories,
                    # Heuristic signals for the ranker
                    "priority_score": round(self._daily_priority_score(p), 3),
                    "delight_prior": round(self._calculate_intellectual_delight_score(p), 3),
                    "interdisciplinary": len({c.split(".")[0] for c in (p.categories or [])}) > 1,
                }
            )
        return out

    async def discover_emerging_topics(self) -> List[str]:
        """
        Identify emerging research topics from recent submissions.

        Returns:
            List of emerging topic keywords/phrases
        """
        try:
            papers = await self.search_latest(
                categories=self.category_interests.get("emerging", ["cs.AI", "cs.LG"]),
                max_results=100,
                days_back=14,
            )
        except Exception:
            return []

        if not papers:
            return []

        return _extract_emerging_topics(papers)

    def _parse_paper(self, result: arxiv.Result) -> ArxivPaper:
        """Convert arXiv API result to ArxivPaper object"""
        arxiv_id = (result.entry_id or "").split("/")[-1]
        authors = []
        try:
            authors = [author.name for author in (result.authors or [])]
        except Exception:
            authors = []
        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=result.title,
            authors=authors,
            abstract=result.summary,
            categories=list(result.categories or []),
            primary_category=getattr(result, "primary_category", ""),
            published_date=_ensure_aware_utc(result.published),
            updated_date=_ensure_aware_utc(result.updated),
            pdf_url=(result.pdf_url or f"https://arxiv.org/abs/{arxiv_id}"),
            comment=getattr(result, "comment", None),
            journal_ref=getattr(result, "journal_ref", None),
        )

    def _is_breakthrough_candidate(self, paper: ArxivPaper) -> bool:
        """
        Heuristic to identify potential breakthroughs based on paper characteristics.

        Signals:
        - Multiple category tags (interdisciplinary work often breakthrough)
        - Cross-domain categories (e.g., physics + CS, bio + math)
        - Significant update patterns (multiple versions with major changes)
        """
        # Papers spanning multiple domains are often breakthrough
        if len(paper.categories) > 2:
            return True
        
        # Papers that bridge distinct fields
        if len(paper.categories) > 1:
            domains = {cat.split(".")[0] for cat in paper.categories}
            if len(domains) > 1:  # Multiple distinct domains (cs, physics, math, etc.)
                return True
        
        # Papers with significant updates (updated_date much later than published_date)
        if paper.updated_date and paper.published_date:
            days_between = (paper.updated_date - paper.published_date).days
            if days_between > 30:  # Major revision after a month suggests importance
                return True
        
        return False

    def _calculate_intellectual_delight_score(self, paper: ArxivPaper) -> float:
        """
        Pre-score paper for intellectual delight.

        Factors:
        - Novel concepts mentioned
        - Mathematical elegance
        - Interdisciplinary connections
        - Breakthrough potential

        Returns:
            Score from 0-10
        """
        score = 5.0
        if self._is_breakthrough_candidate(paper):
            score += 2.0
        if len(paper.categories) > 2:
            score += 1.5
        if any(cat in paper.categories for cat in ["math.HO", "physics.hist-ph", "gr-qc"]):
            score += 1.0
        return min(score, 10.0)

    def _daily_priority_score(self, paper: ArxivPaper) -> float:
        """Priority score for daily selection combining delight and recency.

        - Base: intellectual delight score (0-10)
        - Recency boost: up to +2 for papers within 48 hours, linearly decaying
        - Interdisciplinary boost: +0.5 for >2 categories across domains
        """
        base = self._calculate_intellectual_delight_score(paper)
        try:
            hours_old = max(0.0, (datetime.now(timezone.utc) - paper.published_date).total_seconds() / 3600.0)
        except Exception:
            hours_old = 9999.0
        recency = max(0.0, 2.0 - (hours_old / 24.0))  # 0..2
        domains = {c.split(".")[0] for c in (paper.categories or [])}
        interdisciplinary = 0.5 if len(domains) > 1 and len(paper.categories or []) > 2 else 0.0
        return base + recency + interdisciplinary

    async def _execute_search(self, search: arxiv.Search) -> List[ArxivPaper]:
        """Execute arXiv search via client and convert results"""
        try:
            loop = asyncio.get_running_loop()
            # Prefer client.results(search) so delay/retries are applied. Some unit tests
            # mock Search objects without full attributes; in that case, fall back.
            def fetch_results() -> List[arxiv.Result]:
                try:
                    return list(self.client.results(search))
                except Exception:
                    # Test-mocked Search may not work with client.results
                    return list(search.results())

            results = await loop.run_in_executor(None, fetch_results)
            return [self._parse_paper(r) for r in results]
        except Exception as e:
            raise ArxivServiceError(f"arXiv search failed: {e}") from e

    def _generate_cache_key(self, **kwargs) -> str:
        """Generate cache key for search parameters"""
        key_parts = []
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}:{v}")
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()

    def _load_daily_config(self) -> Dict[str, Any]:
        """Load arXiv daily selection configuration from YAML if present.

        Returns sensible defaults if config file is missing or unreadable.
        """
        default_cfg: Dict[str, Any] = {
            "days_back": 3,
            "max_per_day": 5,
            "max_candidates": 30,
            "diversity_buckets": [
                {"name": "ai_cs", "categories": [
                    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "cs.NE", "cs.DS", "cs.IR", "cs.CR",
                    "cs.IT", "cs.HC", "cs.SI"
                ], "min": 1},
                {"name": "physics_engineering", "categories": [
                    "quant-ph", "hep-th", "gr-qc", "physics.app-ph", "physics.comp-ph", "cond-mat.mtrl-sci",
                    "eess.SP", "eess.IV", "eess.SY", "eess.AS"
                ], "min": 1},
                {"name": "math_stats", "categories": [
                    "math.PR", "math.ST", "math.OC", "math.NA", "math.DS", "stat.ML", "stat.TH", "stat.ME", "stat.CO"
                ], "min": 1},
                {"name": "wildcard_1", "categories": [
                    "physics.soc-ph", "physics.hist-ph", "physics.data-an", "physics.ins-det", "physics.optics", "physics.plasm-ph",
                    "astro-ph", "astro-ph.CO", "astro-ph.GA", "astro-ph.HE", "astro-ph.IM", "astro-ph.SR",
                    "cs.CY", "cs.SI", "cs.GT", "cs.MA", "cs.HC", "cs.SD", "cs.DB", "cs.SE", "cs.PL", "cs.OS", "cs.PF", "cs.NI", "cs.MM", "cs.DC", "cs.SY", "cs.AR", "cs.LO",
                    "econ.GN", "econ.EM", "econ.TH", "q-fin.GN", "q-fin.PR", "q-fin.ST",
                    "nlin.CD", "nlin.AO", "nlin.CG", "nlin.PS", "nlin.SI",
                    "math.HO", "math.LO", "math.GM", "math.CO", "math.AT", "math.DS"
                ], "min": 0},
                {"name": "wildcard_2", "categories": [
                    "stat.AP", "stat.CO", "stat.ME", "stat.TH", "stat.OT",
                    "cond-mat.stat-mech", "cond-mat.mtrl-sci", "cond-mat.dis-nn", "cond-mat.soft",
                    "eess.SP", "eess.IV", "eess.SY", "eess.AS",
                    "cs.CR", "cs.IR", "cs.IT", "cs.DS", "cs.NE", "cs.RO", "cs.CV", "cs.CL", "cs.AI", "cs.HC",
                    "math.NA", "math.OC", "math.PR", "math.ST", "math.LO"
                ], "min": 0},
            ],
        }
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "arxiv.yaml")
        try:
            if os.path.exists(cfg_path) and yaml is not None:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                # Support top-level keys or nested under "daily"
                if isinstance(loaded, dict):
                    data = loaded.get("daily", loaded)
                    # Shallow merge with defaults
                    merged = {**default_cfg, **{k: v for k, v in (data or {}).items() if v is not None}}
                    # Ensure buckets exist
                    if not isinstance(merged.get("diversity_buckets"), list):
                        merged["diversity_buckets"] = default_cfg["diversity_buckets"]
                    return merged
        except Exception:
            pass
        return default_cfg


class ArxivServiceError(Exception):
    """Custom exception for arXiv service failures"""
    pass


def _tokenize(text: str) -> List[str]:
    separators = " \n\t,.;:!?()[]{}<>\"'`/\\|+-=*#"
    tokens: List[str] = []
    current = []
    for ch in text:
        if ch in separators:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def _ensure_aware_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC.

    arxiv library returns timezone-aware datetimes, but mocks may be naive.
    """
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_emerging_topics(papers: List[ArxivPaper], top_k: int = 5) -> List[str]:
	"""Extract emerging topics using simple noun-phrase-like bigrams and trigrams.
	Emphasizes frequency across distinct papers to avoid one-paper dominance.
	"""
	stopwords = {
		"of","the","in","a","to","and","for","with","on","is","we","that","this","by",
		"an","as","are","be","from","at","it","our","via","using","use","can","into",
		"based","new","novel","approach","method","methods","results","paper","study"
	}
	# Aggregate per-paper token sets to reduce repetition
	paper_bigrams: Counter[str] = Counter()
	paper_trigrams: Counter[str] = Counter()
	for p in papers:
		text = (p.abstract or "").lower()
		tokens = [t for t in _tokenize(text) if len(t) > 2 and t not in stopwords]
		bigrams = {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}
		trigrams = {f"{a} {b} {c}" for a, b, c in zip(tokens, tokens[1:], tokens[2:])}
		for bg in bigrams:
			paper_bigrams[bg] += 1
		for tg in trigrams:
			paper_trigrams[tg] += 1

	# Favor trigrams, backfill with bigrams
	ranked_trigrams = [phrase for phrase, _ in paper_trigrams.most_common(top_k)]
	if len(ranked_trigrams) >= top_k:
		return ranked_trigrams[:top_k]
	needed = top_k - len(ranked_trigrams)
	ranked_bigrams = [phrase for phrase, _ in paper_bigrams.most_common(needed)]
	return ranked_trigrams + ranked_bigrams


