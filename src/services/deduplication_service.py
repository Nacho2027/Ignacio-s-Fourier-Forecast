import asyncio
import hashlib
import logging
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qsl, urlparse, urlunparse

import numpy as np

from src.services.cache_service import CacheService
from src.services.cache_service import ContentItem as CacheContentItem
from src.services.ai_service import AIService
from src.utils.embeddings import EmbeddingService


@dataclass
class DuplicateCandidate:
    item_id: str
    url: str
    headline: str
    similarity_score: float
    first_seen: datetime
    angle: Optional[str] = None
    editorial_note: Optional[str] = None


@dataclass
class DeduplicationDecision:
    action: str  # 'keep', 'filter', 'follow_up'
    confidence: float
    reasoning: str
    angle: Optional[str] = None
    related_to: Optional[str] = None


@dataclass
class ContentItem:
    id: str
    url: str
    headline: str
    summary_text: str
    source: str
    section: str
    published_date: datetime
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class DeduplicationError(Exception):
    pass


class DeduplicationService:
    """
    Advanced multi-layer deduplication with AI editorial judgment.
    """

    def __init__(
        self,
        cache_service: CacheService,
        embedding_service: EmbeddingService,
        ai_service: AIService,
    ) -> None:
        self.cache = cache_service
        self.embeddings = embedding_service
        self.ai = ai_service

        # Thresholds tuned via research notes in local-research/deduplication-research.md
        self.similarity_threshold = 0.90
        self.high_similarity_threshold = 0.85  # Lowered from 0.95 to catch more cross-section duplicates
        self.follow_up_window = timedelta(hours=48)

        self.stats: Dict[str, int] = {
            "total_processed": 0,
            "url_filtered": 0,
            "title_filtered": 0,
            "ai_filtered": 0,
            "follow_ups_detected": 0,
            "different_angles_kept": 0,
        }
        
        # Track current-run duplicates separately from historical cache
        self.current_run_urls: Set[str] = set()
        self.current_run_titles: Set[str] = set()

        self.logger = logging.getLogger(__name__)

    async def deduplicate_sections(
        self, sections: Dict[str, List[ContentItem]]
    ) -> Dict[str, List[ContentItem]]:
        """
        Deduplicate all sections with multi-layer intelligence and then
        perform a lightweight cross-section pass to remove redundancies.
        """
        self.reset_statistics()

        # Ensure embeddings for all items up front for performance
        all_items: List[ContentItem] = []
        for items in sections.values():
            all_items.extend(items)
        await self._ensure_embeddings(all_items)

        # First, intra-section deduplication
        filtered: Dict[str, List[ContentItem]] = {}
        for section_name, items in sections.items():
            filtered[section_name] = await self.deduplicate_items(items, section_name)

        # Then, cross-section pass (compare kept items across sections)
        filtered = await self.cross_section_deduplication(filtered)
        return filtered

    async def deduplicate_items(self, items: List[ContentItem], section: str) -> List[ContentItem]:
        """
        Deduplicate a list of items within a section with follow-up clustering.
        """
        unique_items: List[ContentItem] = []
        items_to_cache: List[ContentItem] = []  # Store items to add to cache AFTER processing

        # Ensure embeddings first for batch efficiency
        await self._ensure_embeddings(items)
        
        # Debug logging for research papers
        if section == "research_papers":
            self.logger.info(f"=== Deduplicating {len(items)} research papers ===")

        for idx, item in enumerate(items):
            self.stats["total_processed"] += 1
            
            # Debug logging for research papers
            if section == "research_papers":
                self.logger.debug(f"Processing paper {idx+1}/{len(items)}: {item.headline[:50]}...")

            # Layer 1: Check for duplicate within current run first
            if self._check_current_run_duplicate(item):
                if section == "research_papers":
                    self.logger.warning(f"Research paper FILTERED by current-run duplicate: {item.url}")
                self.stats["url_filtered"] += 1
                continue

            # Layer 2: Check historical cache for cross-day deduplication
            is_historical = await self._check_url_duplicate(item) or await self._check_title_duplicate(item)
            if is_historical:
                self.logger.info(f"Item seen in previous days - filtering duplicate: {item.headline[:50]}...")
                # Filter historical duplicates to prevent repetition across days
                if section == "research_papers":
                    self.logger.warning(f"Research paper FILTERED as historical duplicate: {item.url}")
                self.stats["url_filtered"] += 1
                continue

            # Layer 3: Semantic similarity
            similar_items = await self._find_similar_items(item, threshold=self.similarity_threshold)

            if not similar_items:
                if section == "research_papers":
                    self.logger.info(f"Research paper PASSED all filters: {item.headline[:50]}...")
                unique_items.append(item)
                # Don't cache here - only selected items should be cached
                # items_to_cache.append(item)  # REMOVED - we'll cache only selected items
                continue
                
            # Debug similar items for research papers
            if section == "research_papers":
                self.logger.warning(f"Research paper has {len(similar_items)} similar items: {item.headline[:50]}...")

            # Layer 4: AI editorial decision
            decision = await self._make_editorial_decision(item, similar_items)
            if decision.action in ("keep", "follow_up"):
                # Layer 5: Follow-up clustering check
                if decision.action == "follow_up":
                    should_include = await self._should_include_follow_up(item, unique_items, section)
                    if not should_include:
                        if section == "research_papers":
                            self.logger.warning(f"Research paper FILTERED by follow-up clustering: {item.headline[:50]}...")
                        self.stats["ai_filtered"] += 1
                        continue
                    self.stats["follow_ups_detected"] += 1
                
                if decision.action == "keep" and decision.angle:
                    self.stats["different_angles_kept"] += 1
                    
                if section == "research_papers":
                    self.logger.info(f"Research paper KEPT after AI decision ({decision.action}): {item.headline[:50]}...")
                unique_items.append(item)
                # Don't cache here - only selected items should be cached
                # items_to_cache.append(item)  # REMOVED - we'll cache only selected items
            else:
                if section == "research_papers":
                    self.logger.warning(f"Research paper FILTERED by AI editorial ({decision.action}): {item.headline[:50]}...")
                self.stats["ai_filtered"] += 1
        
        # DON'T cache items during deduplication - only cache finally selected items
        # This ensures unselected papers remain available for future newsletters
        # Caching now happens in main.py after selection
        # for item in items_to_cache:
        #     await self._add_to_cache(item)
        
        # Final summary for research papers
        if section == "research_papers":
            self.logger.info(f"=== Research papers deduplication complete: {len(unique_items)}/{len(items)} kept ===")
            if len(unique_items) == 0:
                self.logger.error("WARNING: All research papers were filtered out!")

        return unique_items

    def _check_current_run_duplicate(self, item: ContentItem) -> bool:
        """
        Check if this item is a duplicate within the current run only.
        This prevents duplicates within the same newsletter.
        """
        # Normalize the URL using cache service's method
        normalized_url = self.cache.normalize_url(item.url)
        title_hash = hashlib.sha256(item.headline.lower().encode()).hexdigest()
        
        # Check if we've seen this URL or title in the current run
        if normalized_url in self.current_run_urls:
            self.logger.debug(f"Current-run URL duplicate: {item.url}")
            return True
        if title_hash in self.current_run_titles:
            self.logger.debug(f"Current-run title duplicate: {item.headline[:50]}...")
            return True
            
        # Add to current-run tracking
        self.current_run_urls.add(normalized_url)
        self.current_run_titles.add(title_hash)
        return False

    async def _check_url_duplicate(self, item: ContentItem) -> bool:
        """Layer 1: Check for exact URL match (normalized)."""
        normalized = self.cache.normalize_url(item.url)
        is_dup = await self.cache.is_url_duplicate(normalized, days=30)
        if is_dup:
            self.stats["url_filtered"] += 1
        return is_dup

    async def _check_title_duplicate(self, item: ContentItem) -> bool:
        """Layer 2: Check for title hash match (normalized)."""
        # Don't normalize here - cache.is_title_duplicate will normalize it
        is_dup = await self.cache.is_title_duplicate(item.headline, days=7)
        
        # Debug logging - only warn if we detect duplicates very early in the pipeline
        # This can happen if the database wasn't properly cleared
        if is_dup and self.stats["total_processed"] < 10:
            self.logger.warning(f"⚠️ Early duplicate detection (item #{self.stats['total_processed']}): {item.headline[:50]}...")
            self.logger.debug(f"  This may indicate the cache wasn't properly cleared")
            
        if is_dup:
            self.stats["title_filtered"] += 1
        return is_dup

    async def _find_similar_items(
        self, item: ContentItem, threshold: float = 0.90
    ) -> List[DuplicateCandidate]:
        """
        Layer 3: Find semantically similar items using embeddings stored in cache.
        """
        if item.embedding is None:
            await self._ensure_embeddings([item])

        recent = await self.cache.get_recent_embeddings(item.section, days=7)
        candidates: List[DuplicateCandidate] = []
        for rec_id, rec_headline, emb_blob, rec_url, rec_first_seen in recent:
            if rec_id == item.id:
                continue
            try:
                stored_emb = np.array(pickle.loads(emb_blob))
            except Exception:
                continue
            score = self._calculate_similarity(item.embedding, stored_emb)
            if score >= threshold:
                candidates.append(
                    DuplicateCandidate(
                        item_id=rec_id,
                        url=rec_url or "",
                        headline=rec_headline,
                        similarity_score=float(score),
                        first_seen=self._parse_dt(rec_first_seen) or datetime.now(),
                    )
                )
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return candidates

    async def _make_editorial_decision(
        self, item: ContentItem, similar_items: List[DuplicateCandidate]
    ) -> DeduplicationDecision:
        """
        Layer 4: Use AI to judge whether to keep, filter, or mark as follow-up.
        Enhanced to track follow-up stories properly.
        """
        new_item_dict = {
            "headline": item.headline,
            "source": item.source,
            "content_preview": (item.summary_text or "")[:500],
            "published_date": item.published_date.isoformat() if item.published_date else None,
        }
        now_ts = datetime.now()
        similar_items_dicts = []
        potential_follow_up = None
        
        for s in similar_items:
            try:
                days_ago = (now_ts - s.first_seen).days if isinstance(s.first_seen, datetime) else 0
                hours_ago = (now_ts - s.first_seen).total_seconds() / 3600 if isinstance(s.first_seen, datetime) else 0
            except Exception:
                days_ago = 0
                hours_ago = 0
            
            # Use enhanced follow-up detection
            is_potential_follow_up = await self._is_follow_up(item, s)
            
            # Calculate story significance for context
            significance_score = self._calculate_story_significance(item, s)
            
            # Determine follow-up type for better AI context
            follow_up_type = "none"
            if is_potential_follow_up:
                if significance_score >= 0.8:
                    follow_up_type = "major_development"
                elif significance_score >= 0.6:
                    follow_up_type = "important_update"
                else:
                    follow_up_type = "routine_follow_up"
            
            if is_potential_follow_up and not potential_follow_up:
                potential_follow_up = s.item_id
            
            similar_items_dicts.append(
                {
                    "headline": s.headline,
                    "similarity": round(s.similarity_score, 3),
                    "days_ago": days_ago,
                    "hours_ago": round(hours_ago, 1),
                    "is_potential_follow_up": is_potential_follow_up,
                    "follow_up_type": follow_up_type,
                    "significance_score": round(significance_score, 3),
                }
            )

        ai_decision = await self.ai.editorial_deduplication(
            new_item=new_item_dict, similar_items=similar_items_dicts
        )

        # Enhanced decision with follow-up tracking
        decision = DeduplicationDecision(
            action=ai_decision.decision,
            confidence=float(ai_decision.confidence),
            reasoning=ai_decision.reason,
            angle=ai_decision.editorial_note if hasattr(ai_decision, 'editorial_note') else None,
            related_to=potential_follow_up,
        )
        
        # Track follow-up stories in stats
        if ai_decision.decision == "follow_up" or "follow" in ai_decision.reason.lower():
            self.stats["follow_ups_detected"] += 1
            # Mark the item as a follow-up for cache storage
            item.metadata = item.metadata or {}
            item.metadata["is_follow_up"] = True
            item.metadata["related_to"] = potential_follow_up
        
        return decision

    # URL normalization delegated to CacheService.normalize_url

    def _normalize_headline(self, headline: str) -> str:
        text = re.sub(r"[^\w\s]", "", (headline or "").lower())
        return " ".join(text.split())

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        if embedding1 is None or embedding2 is None:
            return 0.0
        vec1 = np.asarray(embedding1, dtype=np.float32)
        vec2 = np.asarray(embedding2, dtype=np.float32)
        denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denom == 0:
            return 0.0
        return float(np.clip(np.dot(vec1, vec2) / denom, -1.0, 1.0))

    async def _ensure_embeddings(self, items: List[ContentItem]) -> None:
        to_embed: List[ContentItem] = [i for i in items if i.embedding is None]
        if not to_embed:
            return
        texts = [f"{i.headline}\n\n{(i.summary_text or '')[:500]}" for i in to_embed]
        vectors = await self.embeddings.batch_generate(texts)
        for itm, emb in zip(to_embed, vectors):
            itm.embedding = np.array(emb, dtype=np.float32)

    async def _is_follow_up(self, item: ContentItem, original: DuplicateCandidate) -> bool:
        """
        Enhanced follow-up detection with multiple signals and temporal significance.
        A story is a follow-up if:
        1. Within time window (7 days for major stories, 3 days for routine)
        2. High similarity (>= 0.7)
        3. Published after the original
        4. Contains evolution keywords or shows meaningful development
        5. Has temporal significance (major developments get priority)
        """
        # Ensure both datetimes are timezone-aware for comparison
        from datetime import timezone
        
        # Make published_date timezone-aware if it isn't
        if item.published_date.tzinfo is None:
            item_published = item.published_date.replace(tzinfo=timezone.utc)
        else:
            item_published = item.published_date
            
        # Make first_seen timezone-aware if it isn't    
        if original.first_seen.tzinfo is None:
            original_first_seen = original.first_seen.replace(tzinfo=timezone.utc)
        else:
            original_first_seen = original.first_seen
        
        # Must be published after the original
        if item_published <= original_first_seen:
            return False
        
        # Calculate time elapsed since original
        time_elapsed = datetime.now(timezone.utc) - original_first_seen
        hours_elapsed = time_elapsed.total_seconds() / 3600
        
        # Determine story significance for adaptive time windows
        significance_score = self._calculate_story_significance(item, original)
        
        # Adaptive time window based on story significance
        # Major stories (high significance) get longer follow-up windows
        max_follow_up_hours = 72  # 3 days default
        if significance_score >= 0.8:
            max_follow_up_hours = 168  # 7 days for major stories
        elif significance_score >= 0.6:
            max_follow_up_hours = 120  # 5 days for important stories
        
        if hours_elapsed > max_follow_up_hours:
            return False
        
        # Enhanced evolution keywords with categories
        major_development_keywords = [
            "breaking", "major update", "significant", "unprecedented", 
            "escalates dramatically", "major breakthrough", "crisis", "emergency"
        ]
        
        routine_evolution_keywords = [
            "update", "develops", "latest", "new", "responds", "aftermath", 
            "following", "continues", "expands", "revised", "corrected",
            "statement", "announces", "confirms", "denies", "clarifies"
        ]
        
        headline_lower = item.headline.lower()
        content_lower = (item.summary_text or "").lower()
        
        # Check for major development indicators
        has_major_development = any(kw in headline_lower or kw in content_lower[:200] 
                                  for kw in major_development_keywords)
        has_routine_evolution = any(kw in headline_lower 
                                  for kw in routine_evolution_keywords)
        
        # Enhanced scoring for follow-up determination
        follow_up_score = 0.0
        
        # Base similarity contribution
        follow_up_score += min(original.similarity_score, 1.0) * 0.4
        
        # Temporal significance contribution
        follow_up_score += significance_score * 0.3
        
        # Development type contribution
        if has_major_development:
            follow_up_score += 0.25
        elif has_routine_evolution:
            follow_up_score += 0.15
        
        # Timing contribution (fresher developments score higher)
        if hours_elapsed <= 4:
            follow_up_score += 0.1
        elif hours_elapsed <= 24:
            follow_up_score += 0.05
        
        # Follow-up threshold: need at least 0.7 total score
        return follow_up_score >= 0.7

    def _calculate_story_significance(self, item: ContentItem, original: DuplicateCandidate) -> float:
        """
        Calculate temporal significance of a story for adaptive follow-up windows.
        Returns score 0.0-1.0 where 1.0 is maximum significance.
        """
        significance = 0.0
        
        headline_lower = item.headline.lower()
        content_lower = (item.summary_text or "").lower()
        
        # Source authority indicators
        high_authority_sources = [
            "reuters", "associated press", "ap news", "wall street journal", 
            "wsj", "financial times", "bloomberg", "mit technology review"
        ]
        source_lower = item.source.lower()
        if any(source in source_lower for source in high_authority_sources):
            significance += 0.2
        
        # Topic significance indicators
        major_topics = [
            "federal reserve", "central bank", "geopolitical", "breakthrough",
            "pandemic", "crisis", "war", "election", "supreme court",
            "artificial intelligence", "climate change", "quantum"
        ]
        if any(topic in headline_lower or topic in content_lower[:200] 
               for topic in major_topics):
            significance += 0.3
        
        # Impact scope indicators
        global_indicators = [
            "global", "worldwide", "international", "markets", "economy",
            "nato", "united nations", "world health", "climate"
        ]
        if any(indicator in headline_lower for indicator in global_indicators):
            significance += 0.25
        
        # Urgency indicators
        urgency_words = [
            "urgent", "immediate", "emergency", "critical", "unprecedented",
            "historic", "landmark", "major", "significant"
        ]
        if any(word in headline_lower for word in urgency_words):
            significance += 0.15
        
        # Section-based significance adjustment
        if item.section == "breaking_news":
            significance += 0.1
        elif item.section == "research_papers":
            significance += 0.05  # Research moves slower but can be very significant
        
        return min(significance, 1.0)

    async def _should_include_follow_up(self, item: ContentItem, existing_items: List[ContentItem], section: str) -> bool:
        """
        Determine if a follow-up story should be included based on clustering rules and intelligent spacing.
        Prevents oversaturation of follow-ups for the same topic.
        """
        # First check intelligent spacing from cache service
        should_space = await self.cache.should_space_follow_up(item.id, section)
        if not should_space:
            self.logger.debug(f"Follow-up {item.id} filtered due to spacing rules in {section}")
            return False
        
        # Count existing follow-ups in this section
        follow_up_count = sum(1 for existing in existing_items 
                             if hasattr(existing, 'metadata') and existing.metadata 
                             and existing.metadata.get('is_follow_up', False))
        
        # Section-specific follow-up limits
        max_follow_ups = {
            "breaking_news": 2,  # Breaking news can have more follow-ups
            "business": 1,       # Business prefers fewer follow-ups
            "tech_science": 1,   # Tech/science focuses on breakthrough stories
            "startup": 1,        # Startup advice doesn't need many follow-ups
            "politics": 1,       # Politics can get repetitive
            "research_papers": 0, # Research papers are standalone
            "local": 1,          # Local news limited follow-ups
            "miscellaneous": 1,  # Renaissance topics prefer diversity
        }
        
        section_limit = max_follow_ups.get(section, 1)
        
        # If we're at the limit, check if this follow-up is more significant
        if follow_up_count >= section_limit:
            # Calculate significance of new item
            dummy_candidate = DuplicateCandidate(
                item_id="temp", url="", headline="", 
                similarity_score=0.8, first_seen=datetime.now()
            )
            new_significance = self._calculate_story_significance(item, dummy_candidate)
            
            # Only include if this is a major development (significance >= 0.8)
            if new_significance < 0.8:
                return False
            
            # For major developments, allow one extra follow-up
            if follow_up_count >= section_limit + 1:
                return False
        
        # Check for topic clustering to avoid multiple follow-ups on same theme
        item_keywords = self._extract_topic_keywords(item)
        similar_topic_count = 0
        
        for existing in existing_items:
            if hasattr(existing, 'metadata') and existing.metadata and existing.metadata.get('is_follow_up'):
                existing_keywords = self._extract_topic_keywords(existing)
                # Check for keyword overlap
                overlap = len(item_keywords.intersection(existing_keywords))
                if overlap >= 2:  # Significant topic overlap
                    similar_topic_count += 1
        
        # Limit follow-ups on the same topic (max 1 per topic)
        if similar_topic_count >= 1:
            return False
        
        # Check recent story arcs to avoid oversaturating the same story
        recent_arcs = await self.cache.get_recent_story_arcs(section, days=3)
        related_to = item.metadata.get("related_to") if item.metadata else None
        
        if related_to:
            # Check if this story arc already has too many recent follow-ups
            for arc in recent_arcs:
                if arc["original_id"] == related_to and arc["follow_up_count"] >= 3:
                    # Only allow if this is a major development
                    dummy_candidate = DuplicateCandidate(
                        item_id="temp", url="", headline="", 
                        similarity_score=0.8, first_seen=datetime.now()
                    )
                    significance = self._calculate_story_significance(item, dummy_candidate)
                    if significance < 0.8:
                        self.logger.debug(f"Follow-up {item.id} filtered: story arc {related_to} already has {arc['follow_up_count']} follow-ups")
                        return False
        
        return True

    def _extract_topic_keywords(self, item: ContentItem) -> Set[str]:
        """
        Extract key topics from headline and content for clustering analysis.
        """
        text = f"{item.headline} {(item.summary_text or '')[:200]}".lower()
        
        # Key topic indicators for clustering
        topic_patterns = {
            # Technology
            "ai", "artificial intelligence", "machine learning", "chatgpt", "openai",
            "quantum", "blockchain", "cryptocurrency", "bitcoin", "tesla", "apple", "microsoft",
            
            # Economics/Finance
            "federal reserve", "fed", "interest rates", "inflation", "recession", "gdp",
            "stock market", "nasdaq", "dow jones", "crypto", "economy",
            
            # Geopolitics
            "china", "russia", "ukraine", "nato", "eu", "israel", "palestine", "iran",
            "congress", "senate", "white house", "supreme court", "election",
            
            # Health/Science
            "climate change", "covid", "pandemic", "vaccine", "health", "cancer", "drug",
            "fda", "who", "research", "study",
            
            # Business
            "merger", "acquisition", "ipo", "earnings", "ceo", "layoffs", "startup",
            "venture capital", "funding", "valuation",
        }
        
        found_topics = set()
        for topic in topic_patterns:
            if topic in text:
                found_topics.add(topic)
        
        # Also extract company names and proper nouns (simplified)
        words = text.split()
        for word in words:
            if len(word) > 3 and word.isupper():  # Likely acronym
                found_topics.add(word.lower())
            elif len(word) > 4 and word[0].isupper() and word[1:].islower():  # Likely proper noun
                found_topics.add(word.lower())
        
        return found_topics

    def _extract_domain(self, url: str) -> str:
        return urlparse(url).netloc.lower().replace("www.", "")

    async def cross_section_deduplication(
        self, sections: Dict[str, List[ContentItem]]
    ) -> Dict[str, List[ContentItem]]:
        """
        Compare items kept in earlier sections against later ones using embeddings.
        If highly similar (>= high threshold), ask AI whether to keep both.
        """
        ordered_sections = list(sections.keys())
        kept_so_far: List[ContentItem] = []
        for name in ordered_sections:
            new_list: List[ContentItem] = []
            for item in sections[name]:
                # Compare against all previously kept items
                max_sim = 0.0
                most_similar: Optional[ContentItem] = None
                for prev in kept_so_far:
                    if item.embedding is None or prev.embedding is None:
                        continue
                    sim = self._calculate_similarity(item.embedding, prev.embedding)
                    if sim > max_sim:
                        max_sim = sim
                        most_similar = prev

                if max_sim >= self.high_similarity_threshold and most_similar is not None:
                    duplicate_candidate = DuplicateCandidate(
                        item_id=most_similar.id,
                        url=most_similar.url,
                        headline=most_similar.headline,
                        similarity_score=max_sim,
                        first_seen=most_similar.published_date,
                    )
                    decision = await self._make_editorial_decision(item, [duplicate_candidate])
                    if decision.action == "filter":
                        # Drop this cross-duplicate
                        self.stats["ai_filtered"] += 1
                        continue

                new_list.append(item)
                kept_so_far.append(item)

            sections[name] = new_list
        return sections

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self.stats)

    def reset_statistics(self) -> None:
        for k in list(self.stats.keys()):
            self.stats[k] = 0
        # Reset current-run tracking
        self.current_run_urls = set()
        self.current_run_titles = set()

    async def learn_from_decision(
        self, item: ContentItem, decision: DeduplicationDecision, outcome: Optional[str] = None
    ) -> None:
        # Placeholder hook for future feedback loop
        self.logger.debug("Learn from decision: %s on %s", decision.action, item.id)

    async def _add_to_cache(self, item: ContentItem) -> None:
        # Convert to CacheService.ContentItem to satisfy expected fields
        # Enhanced to properly track follow-up stories and create story arcs
        is_follow_up = False
        editorial_note = None
        related_to = None
        
        if item.metadata:
            is_follow_up = item.metadata.get("is_follow_up", False)
            related_to = item.metadata.get("related_to")
            if is_follow_up:
                editorial_note = f"Follow-up story{f' to {related_to}' if related_to else ''}"
        
        cache_item = CacheContentItem(
            id=item.id,
            source=item.source,
            section=item.section,
            headline=item.headline,
            summary_text=item.summary_text,
            url=item.url,
            published_date=item.published_date,
            metadata=item.metadata or {},
            embedding=(item.embedding.tolist() if isinstance(item.embedding, np.ndarray) else item.embedding),
            is_follow_up=is_follow_up,
            editorial_note=editorial_note,
            importance_score=None,
        )
        await self.cache.add_item(cache_item)
        
        # Create story arc if this is a follow-up
        if is_follow_up and related_to:
            try:
                # Determine arc type based on significance and content
                dummy_candidate = DuplicateCandidate(
                    item_id="temp", url="", headline="", 
                    similarity_score=0.8, first_seen=datetime.now()
                )
                significance = self._calculate_story_significance(item, dummy_candidate)
                
                if significance >= 0.8:
                    arc_type = "major_development"
                elif significance >= 0.6:
                    arc_type = "important_update"
                else:
                    arc_type = "routine_follow_up"
                
                await self.cache.create_story_arc(related_to, item.id, arc_type)
                self.logger.info(f"Created story arc: {related_to} -> {item.id} ({arc_type})")
                
            except Exception as e:
                self.logger.warning(f"Failed to create story arc for {item.id}: {e}")

    def _parse_dt(self, value: Optional[str]) -> Optional[datetime]:
        try:
            from dateutil.parser import isoparse
            if value:
                return isoparse(value)
            return None
        except Exception:
            return None


