from __future__ import annotations

import sqlite3
import hashlib
import pickle
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from dateutil.parser import isoparse
import asyncio
import aiosqlite


EASTERN_TZ = ZoneInfo("America/New_York")


@dataclass
class ContentItem:
    """Unified content structure for all sources"""
    id: str
    source: str
    section: str
    headline: str
    summary_text: str
    url: str
    published_date: datetime
    metadata: dict
    # Added fields for deduplication
    embedding: Optional[List[float]] = None
    is_follow_up: bool = False
    editorial_note: Optional[str] = None
    importance_score: Optional[float] = None


class CacheService:
    """
    SQLite-based intelligent deduplication with embeddings.
    Provides 30-day memory for the Renaissance newsletter.
    """

    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        # Note: Database initialization is now async. Call await initialize_db() after constructing.

    def _init_db(self) -> None:
        """Initialize database with all required tables and indexes (sync)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()

            # Tables
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_items (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    normalized_url TEXT,
                    
                    headline TEXT,
                    title_hash TEXT,
                    content_hash TEXT,
                    
                    content_embedding BLOB,
                    
                    section TEXT,
                    importance_score REAL,
                    source TEXT,
                    
                    first_seen_date TEXT,
                    last_seen_date TEXT,
                    times_seen INTEGER DEFAULT 1,
                    
                    editorial_decision TEXT,
                    decision_reason TEXT,
                    decision_confidence REAL,
                    angle_classification TEXT,
                    multi_angle_worthy INTEGER,
                    temporal_relevance TEXT,
                    
                    related_stories_json TEXT,
                    is_follow_up_to TEXT,
                    follow_ups_json TEXT,
                    
                    reader_value_score REAL,
                    was_included INTEGER,
                    reader_engagement TEXT
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS newsletter_manifest (
                    date TEXT PRIMARY KEY,
                    subject TEXT,
                    greeting TEXT,
                    golden_thread TEXT,
                    sections_json TEXT,
                    metrics_json TEXT,
                    total_items INTEGER,
                    total_read_time_minutes REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS dedup_metrics (
                    date TEXT PRIMARY KEY,
                    total_fetched INTEGER,
                    url_duplicates INTEGER,
                    title_duplicates INTEGER,
                    semantic_duplicates INTEGER,
                    follow_ups_allowed INTEGER,
                    final_unique INTEGER
                );
                """
            )

            # Indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_seen_date ON seen_items(last_seen_date);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_normalized_url ON seen_items(normalized_url);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_title_hash ON seen_items(title_hash);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_date ON seen_items(section, last_seen_date);"
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_section_date
                ON seen_items(section, last_seen_date)
                WHERE content_embedding IS NOT NULL;
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_editorial_decisions ON seen_items(editorial_decision, decision_confidence);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships ON seen_items(is_follow_up_to, section);"
            )

            conn.commit()
        finally:
            conn.close()

    async def initialize_db(self) -> None:
        """Async initialization: create tables and indexes using aiosqlite."""
        async with aiosqlite.connect(self.db_path) as db:
            # Tables
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_items (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    normalized_url TEXT,
                    
                    headline TEXT,
                    title_hash TEXT,
                    content_hash TEXT,
                    
                    content_embedding BLOB,
                    
                    section TEXT,
                    importance_score REAL,
                    source TEXT,
                    
                    first_seen_date TEXT,
                    last_seen_date TEXT,
                    times_seen INTEGER DEFAULT 1,
                    
                    editorial_decision TEXT,
                    decision_reason TEXT,
                    decision_confidence REAL,
                    angle_classification TEXT,
                    multi_angle_worthy INTEGER,
                    temporal_relevance TEXT,
                    
                    related_stories_json TEXT,
                    is_follow_up_to TEXT,
                    follow_ups_json TEXT,
                    
                    reader_value_score REAL,
                    was_included INTEGER,
                    reader_engagement TEXT
                );
                """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS newsletter_manifest (
                    date TEXT PRIMARY KEY,
                    subject TEXT,
                    greeting TEXT,
                    golden_thread TEXT,
                    sections_json TEXT,
                    metrics_json TEXT,
                    total_items INTEGER,
                    total_read_time_minutes REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS dedup_metrics (
                    date TEXT PRIMARY KEY,
                    total_fetched INTEGER,
                    url_duplicates INTEGER,
                    title_duplicates INTEGER,
                    semantic_duplicates INTEGER,
                    follow_ups_allowed INTEGER,
                    final_unique INTEGER
                );
                """
            )

            # Indexes
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_seen_date ON seen_items(last_seen_date);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_normalized_url ON seen_items(normalized_url);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_title_hash ON seen_items(title_hash);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_date ON seen_items(section, last_seen_date);"
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_section_date
                ON seen_items(section, last_seen_date)
                WHERE content_embedding IS NOT NULL;
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_editorial_decisions ON seen_items(editorial_decision, decision_confidence);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships ON seen_items(is_follow_up_to, section);"
            )

            await db.commit()

    async def add_selected_item(self, item: ContentItem, score: float) -> None:
        """
        Add a SELECTED item to cache with adaptive expiry based on score.
        High-scoring items get shorter cache periods (can reappear sooner).
        
        Args:
            item: The content item that was selected for the newsletter
            score: The item's quality score (0-10 scale typically)
        
        Cache duration:
        - Score 9-10: 7 days (exceptional content can reappear sooner)
        - Score 7-9: 14 days  
        - Score 5-7: 21 days
        - Score <5: 30 days (lower quality needs longer cooldown)
        """
        # Calculate adaptive expiry based on score
        if score >= 9:
            cache_days = 7
        elif score >= 7:
            cache_days = 14
        elif score >= 5:
            cache_days = 21
        else:
            cache_days = 30
            
        # Add metadata about selection
        if item.metadata is None:
            item.metadata = {}
        item.metadata['selected_for_newsletter'] = True
        item.metadata['selection_score'] = score
        item.metadata['cache_days'] = cache_days
        item.metadata['cache_expiry'] = (datetime.now(EASTERN_TZ) + timedelta(days=cache_days)).isoformat()
        
        # Use regular add_item with the metadata
        await self.add_item(item)
        
        self.logger.info(f"Cached selected item '{item.headline[:50]}...' for {cache_days} days (score: {score:.1f})")
    
    async def add_item(self, item: ContentItem) -> None:
        """
        Store a content item with all metadata and embeddings.
        Behavior expected by tests:
        - Use provided item.id as the primary key when creating new rows
        - If an item exists with the same id, update it and increment times_seen
        - If no id match, but URL or normalized URL matches an existing row, update that row (keeping the original id)
        - Ensure timestamps: first_seen_date is preserved; last_seen_date reflects the latest observation
        """
        normalized_url = self.normalize_url(item.url)
        title_hash = self.generate_hash(self.normalize_text(item.headline))
        content_hash = self.generate_hash(item.summary_text)

        now_et = datetime.now(EASTERN_TZ).isoformat()
        pub_dt = item.published_date
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=EASTERN_TZ)
        pub_iso = pub_dt.astimezone(EASTERN_TZ).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            existing_url: Optional[str] = None
            # 1) Try exact id match
            cur = await db.execute(
                "SELECT id, times_seen, first_seen_date, url FROM seen_items WHERE id = ? LIMIT 1",
                (item.id,),
            )
            row = await cur.fetchone()

            # 2) Else try exact URL match
            if row is None:
                cur = await db.execute(
                    "SELECT id, times_seen, first_seen_date, url FROM seen_items WHERE url = ? LIMIT 1",
                    (item.url,),
                )
                row = await cur.fetchone()

            # 3) Else try normalized URL match
            if row is None:
                cur = await db.execute(
                    "SELECT id, times_seen, first_seen_date, url FROM seen_items WHERE normalized_url = ? LIMIT 1",
                    (normalized_url,),
                )
                row = await cur.fetchone()

            if row is not None:
                target_id = row[0]
                times_seen = (row[1] or 1) + 1
                first_seen_date = row[2] or pub_iso
                last_seen_date = now_et
                existing_url = row[3]
            else:
                target_id = item.id
                times_seen = 1
                first_seen_date = pub_iso
                last_seen_date = pub_iso
                existing_url = None

            # Use existing stored URL when we matched by normalized URL to avoid UNIQUE(url) conflicts
            url_to_store = existing_url or item.url

            await db.execute(
                """
                INSERT INTO seen_items (
                    id, url, normalized_url,
                    headline, title_hash, content_hash,
                    content_embedding,
                    section, importance_score, source,
                    first_seen_date, last_seen_date, times_seen,
                    editorial_decision, decision_reason, decision_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    url = excluded.url,
                    normalized_url = excluded.normalized_url,
                    headline = excluded.headline,
                    title_hash = excluded.title_hash,
                    content_hash = excluded.content_hash,
                    content_embedding = excluded.content_embedding,
                    section = excluded.section,
                    importance_score = excluded.importance_score,
                    source = excluded.source,
                    -- Preserve original first_seen_date when present
                    first_seen_date = COALESCE(first_seen_date, excluded.first_seen_date),
                    last_seen_date = excluded.last_seen_date,
                    times_seen = excluded.times_seen,
                    editorial_decision = excluded.editorial_decision,
                    decision_reason = excluded.decision_reason,
                    decision_confidence = excluded.decision_confidence
                ;
                """,
                (
                    target_id,
                    url_to_store,
                    normalized_url,
                    item.headline,
                    title_hash,
                    content_hash,
                    pickle.dumps(item.embedding) if item.embedding is not None else None,
                    item.section,
                    item.importance_score,
                    item.source,
                    first_seen_date,
                    last_seen_date,
                    times_seen,
                    None,
                    None,
                    None,
                ),
            )
            await db.commit()

    async def get_item(self, item_id: str) -> Optional[ContentItem]:
        """Retrieve a stored item by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM seen_items WHERE id = ?", (item_id,))
            row = await cur.fetchone()
            if row is None:
                return None

            embedding_blob = row["content_embedding"]
            embedding_val: Optional[List[float]]
            if embedding_blob is not None:
                try:
                    embedding_val = pickle.loads(embedding_blob)
                except Exception:
                    embedding_val = None
            else:
                embedding_val = None

            # Note: content and metadata are not stored in schema; default them
            return ContentItem(
                id=row["id"],
                source=row["source"],
                section=row["section"],
                headline=row["headline"],
                summary_text="",
                url=row["url"],
                published_date=isoparse(row["first_seen_date"]).astimezone(EASTERN_TZ) if row["first_seen_date"] else datetime.now(EASTERN_TZ),
                metadata={},
                embedding=embedding_val,
                is_follow_up=False,
                editorial_note=None,
                importance_score=row["importance_score"],
            )

    async def is_url_duplicate(self, url: str, days: int = 30) -> bool:
        """Check if URL was seen in the last N days"""
        normalized = self.normalize_url(url)
        cutoff = (datetime.now(EASTERN_TZ) - timedelta(days=days)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                """
                SELECT 1 FROM seen_items
                WHERE normalized_url = ? AND last_seen_date >= ?
                LIMIT 1
                """,
                (normalized, cutoff),
            )
            return await cur.fetchone() is not None

    async def is_title_duplicate(self, title: str, days: int = 7) -> bool:
        """Check if similar title was seen recently"""
        title_hash = self.generate_hash(self.normalize_text(title))
        cutoff = (datetime.now(EASTERN_TZ) - timedelta(days=days)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                """
                SELECT 1 FROM seen_items
                WHERE title_hash = ? AND last_seen_date >= ?
                LIMIT 1
                """,
                (title_hash, cutoff),
            )
            return await cur.fetchone() is not None

    async def get_recent_embeddings(self, section: str, days: int = 7) -> List[Tuple[str, str, bytes, str, str]]:
        """
        Retrieve recent embeddings for semantic comparison.
        Returns list of (id, headline, embedding, url, first_seen_date_iso) tuples.
        """
        cutoff = (datetime.now(EASTERN_TZ) - timedelta(days=days)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                """
                SELECT id, headline, content_embedding, url, first_seen_date, last_seen_date
                FROM seen_items
                WHERE section = ?
                  AND content_embedding IS NOT NULL
                """,
                (section,),
            )
            rows = await cur.fetchall()
            out: List[Tuple[str, str, bytes, str, str]] = []
            cutoff_dt = isoparse(cutoff)
            for r in rows:
                first_iso = r[4]
                last_iso = r[5]
                try:
                    first_ok = isoparse(first_iso) >= cutoff_dt if first_iso else False
                except Exception:
                    first_ok = False
                try:
                    last_ok = isoparse(last_iso) >= cutoff_dt if last_iso else False
                except Exception:
                    last_ok = False
                if first_ok or last_ok:
                    out.append((r[0], r[1], r[2], r[3], first_iso))
            return out

    async def cleanup(self, days: int = 30) -> int:
        """
        Remove entries older than specified days.
        Returns number of removed entries.
        """
        cutoff = (datetime.now(EASTERN_TZ) - timedelta(days=days)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "DELETE FROM seen_items WHERE last_seen_date < ?",
                (cutoff,),
            )
            await db.commit()
            return cur.rowcount

    async def clear_all_cache(self) -> dict:
        """
        Clear all cache data from all tables.
        Returns statistics about cleared data.
        """
        self.logger.warning("Clearing ALL cache data - this is irreversible!")
        
        stats = {
            "seen_items": 0,
            "newsletter_manifest": 0,
            "dedup_metrics": 0
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            # Count items before deletion
            for table in stats.keys():
                cur = await db.execute(f"SELECT COUNT(*) FROM {table}")
                row = await cur.fetchone()
                stats[table] = row[0] if row else 0
            
            self.logger.info(f"Clearing cache: {stats['seen_items']} seen items, "
                           f"{stats['newsletter_manifest']} newsletters, "
                           f"{stats['dedup_metrics']} metrics")
            
            # Clear all tables
            await db.execute("DELETE FROM seen_items")
            await db.execute("DELETE FROM newsletter_manifest")
            await db.execute("DELETE FROM dedup_metrics")
            await db.commit()
            
        self.logger.info("All cache data cleared successfully")
        return stats

    async def clear_test_data(self, hours: int = 48) -> dict:
        """
        Clear recent test data (last N hours).
        Returns statistics about cleared data.
        """
        cutoff = (datetime.now(EASTERN_TZ) - timedelta(hours=hours)).isoformat()
        self.logger.info(f"Clearing test data from last {hours} hours (cutoff: {cutoff})")
        
        stats = {
            "seen_items_removed": 0,
            "newsletters_removed": 0,
            "metrics_removed": 0,
            "cutoff_time": cutoff
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            # Remove recent seen_items
            cur = await db.execute(
                "DELETE FROM seen_items WHERE last_seen_date >= ?",
                (cutoff,)
            )
            stats["seen_items_removed"] = cur.rowcount
            self.logger.debug(f"Removed {stats['seen_items_removed']} seen items")
            
            # Remove recent newsletter manifests
            recent_date = (datetime.now(EASTERN_TZ) - timedelta(hours=hours)).date().isoformat()
            cur = await db.execute(
                "DELETE FROM newsletter_manifest WHERE date >= ?",
                (recent_date,)
            )
            stats["newsletters_removed"] = cur.rowcount
            self.logger.debug(f"Removed {stats['newsletters_removed']} newsletter manifests")
            
            # Remove recent dedup metrics
            cur = await db.execute(
                "DELETE FROM dedup_metrics WHERE date >= ?",
                (recent_date,)
            )
            stats["metrics_removed"] = cur.rowcount
            self.logger.debug(f"Removed {stats['metrics_removed']} dedup metrics")
            
            await db.commit()
            
        total_removed = stats["seen_items_removed"] + stats["newsletters_removed"] + stats["metrics_removed"]
        self.logger.info(f"Test data cleanup completed: {total_removed} total items removed")
        return stats

    async def get_high_quality_expired_papers(self, section: str, min_score: float = 8.0, days: int = 30) -> List[ContentItem]:
        """
        Get high-quality papers that have expired from cache but could be reconsidered.
        Used when today's paper quality is low.
        
        Args:
            section: Section to search (e.g., 'research_papers')
            min_score: Minimum score to consider (default 8.0)
            days: How far back to look (default 30)
            
        Returns:
            List of high-quality expired papers
        """
        cutoff = (datetime.now(EASTERN_TZ) - timedelta(days=days)).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM seen_items 
                WHERE section = ? 
                AND importance_score >= ?
                AND last_seen_date >= ?
                ORDER BY importance_score DESC
                LIMIT 10
                """,
                (section, min_score, cutoff)
            )
            rows = await cur.fetchall()
            
            papers = []
            for row in rows:
                # Check if this paper's cache has expired
                metadata = {}
                if 'cache_expiry' in row:
                    cache_expiry = row.get('cache_expiry')
                    if cache_expiry and datetime.fromisoformat(cache_expiry) < datetime.now(EASTERN_TZ):
                        # Paper has expired, can be reconsidered
                        papers.append(ContentItem(
                            id=row["id"],
                            source=row["source"],
                            section=row["section"],
                            headline=row["headline"],
                            summary_text="",  # Don't need full content for reconsideration
                            url=row["url"],
                            published_date=datetime.fromisoformat(row["first_seen_date"]),
                            metadata={"importance_score": row["importance_score"]},
                            embedding=None,
                            is_follow_up=False,
                            editorial_note=None,
                            importance_score=row["importance_score"]
                        ))
            
            return papers
    
    async def get_greatest_hits(self, days_back: int = 30, top_n: int = 10) -> List[ContentItem]:
        """
        Get the best papers from the past N days for monthly "Greatest Hits" review.
        
        Args:
            days_back: How many days to look back (default 30)
            top_n: How many top papers to return (default 10)
            
        Returns:
            List of top papers from the period
        """
        cutoff = (datetime.now(EASTERN_TZ) - timedelta(days=days_back)).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM seen_items 
                WHERE section = 'research_papers'
                AND importance_score IS NOT NULL
                AND last_seen_date >= ?
                ORDER BY importance_score DESC
                LIMIT ?
                """,
                (cutoff, top_n)
            )
            rows = await cur.fetchall()
            
            papers = []
            for row in rows:
                papers.append(ContentItem(
                    id=row["id"] + "_greatest_hit",  # Mark as greatest hit
                    source=row["source"],
                    section=row["section"],
                    headline=row["headline"],
                    summary_text="",  # Don't need full content
                    url=row["url"],
                    published_date=datetime.fromisoformat(row["first_seen_date"]),
                    metadata={
                        "importance_score": row["importance_score"],
                        "is_greatest_hit": True
                    },
                    embedding=None,
                    is_follow_up=False,
                    editorial_note="Greatest Hit from Past Month",
                    importance_score=row["importance_score"]
                ))
            
            return papers

    async def get_uncached_urls_by_timeframe(self, section: str, months_back: int = 6, limit: int = 100) -> List[str]:
        """
        Get URLs that have never been cached for a specific section and timeframe.
        This helps find older papers that were never selected but might be valuable
        when fresh content is running low.
        
        Args:
            section: Content section to search
            months_back: How many months back to search (default: 6 months)
            limit: Maximum number of URLs to return
            
        Returns:
            List of URLs that were never cached in the specified timeframe
        """
        cutoff_date = (datetime.now(EASTERN_TZ) - timedelta(days=months_back * 30)).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            # This would require access to historical fetch data
            # For now, return empty list as this requires integration with ArXiv service
            # to maintain historical fetch records
            
            # TODO: Implement historical URL tracking in ArxivService
            # For proper implementation, we need:
            # 1. ArxivService to log all fetched URLs with metadata
            # 2. Cross-reference with cache to find never-selected items
            # 3. Return URLs that can be re-fetched for consideration
            
            self.logger.info(f"Fallback mode: Would search for uncached {section} URLs from {months_back} months back")
            return []

    async def should_enable_fallback_mode(self, section: str, min_fresh_papers: int = 20) -> bool:
        """
        Determine if we should enable fallback mode to search for older uncached papers.
        
        Args:
            section: Content section to check
            min_fresh_papers: Minimum number of fresh papers needed to avoid fallback
            
        Returns:
            True if fallback mode should be enabled
        """
        # Check how many papers are currently cached
        cache_count = await self.get_active_cache_count(section)
        
        # If cache is very full (>80 items), we likely need fallback for diversity
        if cache_count > 80:
            self.logger.info(f"Fallback mode recommended: {section} cache has {cache_count} items (>80)")
            return True
            
        # If cache is moderately full (>50 items), consider fallback
        if cache_count > 50:
            self.logger.info(f"Fallback mode suggested: {section} cache has {cache_count} items (>50)")
            return True
            
        self.logger.debug(f"Fallback mode not needed: {section} cache has {cache_count} items")
        return False
    
    async def get_active_cache_count(self, section: str = None) -> int:
        """
        Get count of actively cached items (not expired).
        
        Args:
            section: Optional section filter (e.g., 'research_papers')
            
        Returns:
            Number of items currently in cache
        """
        now = datetime.now(EASTERN_TZ).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            if section:
                cur = await db.execute(
                    """
                    SELECT COUNT(*) FROM seen_items 
                    WHERE section = ? 
                    AND last_seen_date >= datetime('now', '-30 days')
                    """,
                    (section,)
                )
            else:
                cur = await db.execute(
                    """
                    SELECT COUNT(*) FROM seen_items 
                    WHERE last_seen_date >= datetime('now', '-30 days')
                    """
                )
            row = await cur.fetchone()
            return row[0] if row else 0
    
    async def get_cache_statistics(self) -> dict:
        """
        Get comprehensive cache statistics.
        Returns detailed information about cache contents.
        """
        self.logger.debug("Gathering cache statistics")
        
        stats = {
            "total_items": 0,
            "items_by_section": {},
            "items_by_source": {},
            "date_range": {},
            "deduplication_stats": {},
            "storage_info": {}
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            # Total items
            cur = await db.execute("SELECT COUNT(*) FROM seen_items")
            row = await cur.fetchone()
            stats["total_items"] = row[0] if row else 0
            
            if stats["total_items"] > 0:
                # Items by section
                cur = await db.execute(
                    "SELECT section, COUNT(*) FROM seen_items GROUP BY section ORDER BY COUNT(*) DESC"
                )
                stats["items_by_section"] = {row[0]: row[1] for row in await cur.fetchall()}
                
                # Items by source
                cur = await db.execute(
                    "SELECT source, COUNT(*) FROM seen_items GROUP BY source ORDER BY COUNT(*) DESC"
                )
                stats["items_by_source"] = {row[0]: row[1] for row in await cur.fetchall()}
                
                # Date range
                cur = await db.execute(
                    "SELECT MIN(first_seen_date), MAX(last_seen_date) FROM seen_items"
                )
                row = await cur.fetchone()
                if row and row[0] and row[1]:
                    stats["date_range"] = {
                        "oldest": row[0],
                        "newest": row[1]
                    }
            
            # Newsletter manifests
            cur = await db.execute("SELECT COUNT(*) FROM newsletter_manifest")
            row = await cur.fetchone()
            stats["storage_info"]["newsletters_stored"] = row[0] if row else 0
            
            # Dedup metrics
            cur = await db.execute("SELECT COUNT(*) FROM dedup_metrics")
            row = await cur.fetchone()
            stats["storage_info"]["dedup_records"] = row[0] if row else 0
            
            # Recent deduplication effectiveness
            recent_date = (datetime.now(EASTERN_TZ) - timedelta(days=7)).date().isoformat()
            cur = await db.execute(
                """
                SELECT 
                    AVG(total_fetched) as avg_fetched,
                    AVG(final_unique) as avg_unique,
                    AVG(CAST(final_unique AS FLOAT) / NULLIF(total_fetched, 0) * 100) as avg_uniqueness_pct
                FROM dedup_metrics 
                WHERE date >= ?
                """,
                (recent_date,)
            )
            row = await cur.fetchone()
            if row and row[0]:
                stats["deduplication_stats"] = {
                    "avg_items_fetched": round(row[0], 1),
                    "avg_unique_items": round(row[1], 1),
                    "avg_uniqueness_percentage": round(row[2], 1) if row[2] else 0
                }
        
        return stats

    async def store_newsletter_manifest(self, date: datetime, manifest: dict) -> None:
        """Store the complete newsletter manifest for a given date"""
        date_et = date.astimezone(EASTERN_TZ).date().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO newsletter_manifest (
                    date, subject, greeting, golden_thread, sections_json,
                    metrics_json, total_items, total_read_time_minutes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date_et,
                    manifest.get("subject"),
                    manifest.get("greeting"),
                    manifest.get("golden_thread"),
                    json.dumps(manifest.get("sections")),
                    json.dumps(manifest.get("metrics")),
                    manifest.get("total_items"),
                    manifest.get("total_read_time_minutes"),
                    datetime.now(EASTERN_TZ).isoformat(),
                ),
            )
            await db.commit()

    async def record_dedup_metrics(self, metrics: dict) -> None:
        """Record deduplication metrics for analysis"""
        # Accept either datetime/date under 'date'
        date_val = metrics["date"]
        if isinstance(date_val, datetime):
            date_et = date_val.astimezone(EASTERN_TZ).date().isoformat()
        else:
            # date object
            date_et = date_val.isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO dedup_metrics (
                    date, total_fetched, url_duplicates, title_duplicates,
                    semantic_duplicates, follow_ups_allowed, final_unique
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date_et,
                    metrics.get("total_fetched"),
                    metrics.get("url_duplicates"),
                    metrics.get("title_duplicates"),
                    metrics.get("semantic_duplicates"),
                    metrics.get("follow_ups_allowed"),
                    metrics.get("final_unique"),
                ),
            )
            await db.commit()

    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL for comparison (remove params, fragments) and lightly filter
        out obvious date components while preserving path structure. Keeps behavior
        compatible with existing tests.
        """
        parsed = urlparse(url.lower())
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path.rstrip('/')

        if not path:
            return domain

        parts = [p for p in path.split('/') if p]
        if not parts:
            return domain

        # Detect if path likely contains a full date (year present)
        has_year = any(CacheService._looks_like_year(p) for p in parts)
        # Filter out date-like components only if a year is present, so that single-number paths like '/1'
        # are preserved (as expected by tests)
        filtered_parts = [p for p in parts if not CacheService._is_date_component(p, has_year=has_year)]
        # If everything was date-like, collapse to domain
        if not filtered_parts:
            return domain
        normalized_parts = filtered_parts

        # Try to detect a story slug with hyphens (common unique identifiers)
        slug_candidates = [p for p in normalized_parts if '-' in p and len(p) > 6]
        if slug_candidates:
            # Prefer the last slug-like segment
            return f"{domain}/{slug_candidates[-1]}"

        normalized_path = '/'.join(normalized_parts)
        return f"{domain}/{normalized_path}"

    @staticmethod
    def _is_date_component(component: str, has_year: bool = False) -> bool:
        c = component.strip('/').lower()
        # years
        if c.isdigit() and len(c) == 4:
            try:
                year = int(c)
                if 1990 <= year <= 2035:
                    return True
            except ValueError:
                pass
        # months or days - only treat as date-like when a year exists somewhere in path
        if has_year and c.isdigit() and 1 <= len(c) <= 2:
            try:
                val = int(c)
                if 1 <= val <= 31:
                    return True
            except ValueError:
                pass
        months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        if has_year and c[:3] in months:
            return True
        return False

    @staticmethod
    def _looks_like_year(component: str) -> bool:
        c = component.strip('/')
        if c.isdigit() and len(c) == 4:
            try:
                year = int(c)
                return 1990 <= year <= 2035
            except ValueError:
                return False
        return False

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for hashing (lowercase, remove extra spaces).
        Used for title deduplication.
        """
        import re
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        return ' '.join(normalized.split())

    @staticmethod
    def generate_hash(text: str) -> str:
        """Generate MD5 hash for text comparison"""
        if text is None:
            text = ""
        return hashlib.md5(text.encode()).hexdigest()

    async def create_story_arc(self, original_item_id: str, follow_up_item_id: str, arc_type: str = "follow_up") -> None:
        """
        Create or update a story arc linking related articles.
        
        Args:
            original_item_id: ID of the original story
            follow_up_item_id: ID of the follow-up story
            arc_type: Type of relationship (follow_up, update, development, etc.)
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get existing follow-ups for the original story
            cursor = await db.execute(
                "SELECT follow_ups_json FROM seen_items WHERE id = ?",
                (original_item_id,)
            )
            result = await cursor.fetchone()
            
            if result and result[0]:
                try:
                    follow_ups = json.loads(result[0])
                except json.JSONDecodeError:
                    follow_ups = []
            else:
                follow_ups = []
            
            # Add new follow-up with timestamp and type
            follow_up_entry = {
                "item_id": follow_up_item_id,
                "arc_type": arc_type,
                "created_at": datetime.now().isoformat(),
                "position": len(follow_ups) + 1
            }
            follow_ups.append(follow_up_entry)
            
            # Update original story with new follow-up
            await db.execute(
                "UPDATE seen_items SET follow_ups_json = ? WHERE id = ?",
                (json.dumps(follow_ups), original_item_id)
            )
            
            # Update follow-up story to reference original
            await db.execute(
                "UPDATE seen_items SET is_follow_up_to = ? WHERE id = ?",
                (original_item_id, follow_up_item_id)
            )
            
            await db.commit()
            self.logger.info(f"Created story arc: {original_item_id} -> {follow_up_item_id} ({arc_type})")

    async def get_story_arc(self, item_id: str) -> dict:
        """
        Get the complete story arc for an item (both predecessors and successors).
        
        Returns:
            dict: Story arc information including original, follow-ups, and metadata
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get the item details
            cursor = await db.execute(
                "SELECT id, headline, first_seen_date, is_follow_up_to, follow_ups_json FROM seen_items WHERE id = ?",
                (item_id,)
            )
            item = await cursor.fetchone()
            
            if not item:
                return {"error": "Item not found"}
            
            arc_info = {
                "item_id": item[0],
                "headline": item[1],
                "first_seen": item[2],
                "is_follow_up_to": item[3],
                "follow_ups": [],
                "arc_length": 1,
                "arc_span_hours": 0
            }
            
            # If this is a follow-up, find the original
            original_id = item_id
            if item[3]:  # is_follow_up_to
                original_id = item[3]
                # Get original story details
                cursor = await db.execute(
                    "SELECT id, headline, first_seen_date, follow_ups_json FROM seen_items WHERE id = ?",
                    (original_id,)
                )
                original = await cursor.fetchone()
                if original:
                    arc_info["original"] = {
                        "item_id": original[0],
                        "headline": original[1],
                        "first_seen": original[2]
                    }
            
            # Get all follow-ups for the arc
            cursor = await db.execute(
                "SELECT follow_ups_json FROM seen_items WHERE id = ?",
                (original_id,)
            )
            result = await cursor.fetchone()
            
            if result and result[0]:
                try:
                    follow_ups = json.loads(result[0])
                    arc_info["follow_ups"] = follow_ups
                    arc_info["arc_length"] = len(follow_ups) + 1
                    
                    # Calculate arc span
                    if follow_ups:
                        first_time = datetime.fromisoformat(arc_info.get("original", {}).get("first_seen", item[2]))
                        last_follow_up = max(follow_ups, key=lambda x: x["created_at"])
                        last_time = datetime.fromisoformat(last_follow_up["created_at"])
                        arc_info["arc_span_hours"] = (last_time - first_time).total_seconds() / 3600
                        
                except json.JSONDecodeError:
                    pass
            
            return arc_info

    async def get_recent_story_arcs(self, section: Optional[str] = None, days: int = 7) -> List[dict]:
        """
        Get recent story arcs with multiple follow-ups for analysis.
        
        Args:
            section: Filter by section (optional)
            days: Look back this many days
            
        Returns:
            List of story arcs with metadata
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            if section:
                cursor = await db.execute(
                    """SELECT id, headline, first_seen_date, follow_ups_json, section 
                       FROM seen_items 
                       WHERE section = ? AND first_seen_date >= ? AND follow_ups_json IS NOT NULL""",
                    (section, cutoff_str)
                )
            else:
                cursor = await db.execute(
                    """SELECT id, headline, first_seen_date, follow_ups_json, section 
                       FROM seen_items 
                       WHERE first_seen_date >= ? AND follow_ups_json IS NOT NULL""",
                    (cutoff_str,)
                )
            
            rows = await cursor.fetchall()
            arcs = []
            
            for row in rows:
                try:
                    follow_ups = json.loads(row[3]) if row[3] else []
                    if len(follow_ups) >= 1:  # Only include stories with follow-ups
                        arc_info = {
                            "original_id": row[0],
                            "headline": row[1],
                            "first_seen": row[2],
                            "section": row[4],
                            "follow_up_count": len(follow_ups),
                            "latest_follow_up": max(follow_ups, key=lambda x: x["created_at"]) if follow_ups else None
                        }
                        arcs.append(arc_info)
                except json.JSONDecodeError:
                    continue
            
            # Sort by follow-up count and recency
            arcs.sort(key=lambda x: (x["follow_up_count"], x["first_seen"]), reverse=True)
            return arcs

    async def should_space_follow_up(self, item_id: str, section: str, min_hours: int = 12) -> bool:
        """
        Determine if a follow-up should be spaced out to avoid oversaturation.
        
        Args:
            item_id: ID of the potential follow-up
            section: Newsletter section
            min_hours: Minimum hours between follow-ups in same section
            
        Returns:
            bool: True if follow-up should be included, False if it should be spaced out
        """
        # Get recent follow-ups in this section
        cutoff = datetime.now() - timedelta(hours=min_hours)
        cutoff_str = cutoff.isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT COUNT(*) FROM seen_items 
                   WHERE section = ? AND was_included = 1 AND first_seen_date >= ? 
                   AND is_follow_up_to IS NOT NULL""",
                (section, cutoff_str)
            )
            recent_follow_ups = (await cursor.fetchone())[0]
            
            # Section-specific spacing rules
            max_recent_follow_ups = {
                "breaking_news": 3,  # Breaking news can have more frequent follow-ups
                "business": 1,       # Business prefers less frequent follow-ups
                "tech_science": 1,   # Tech/science focuses on breakthroughs
                "startup": 1,        # Startup advice doesn't need frequent follow-ups
                "politics": 2,       # Politics can have moderate follow-ups
                "research_papers": 0, # Research papers are standalone
                "local": 1,          # Local news limited follow-ups
                "miscellaneous": 1,  # Renaissance topics prefer diversity
            }
            
            section_limit = max_recent_follow_ups.get(section, 1)
            return recent_follow_ups < section_limit

    async def find_historical_connections(self, item_headline: str, item_content: str, 
                                        section: Optional[str] = None, days_back: int = 30) -> List[dict]:
        """
        Find historical connections between new content and previously featured stories.
        
        Args:
            item_headline: Headline of new item
            item_content: Content of new item (first 500 chars used)
            section: Optional section filter
            days_back: How many days to look back for connections
            
        Returns:
            List of related historical items with connection strength
        """
        # Extract key entities and topics from new item
        search_text = f"{item_headline} {item_content[:500]}".lower()
        
        # Key entities to search for (simplified - could be enhanced with NER)
        import re
        
        # Extract potential company names, people, technologies
        entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper names (First Last)
            r'\b[A-Z]{2,}\b',                 # Acronyms (AI, NASA, etc.)
            r'\$[A-Z]{1,5}\b',               # Stock symbols
            r'\b[A-Z][a-z]+(?:\s+Inc\.?|\s+Corp\.?|\s+LLC\.?)?\b'  # Company names
        ]
        
        entities = set()
        for pattern in entity_patterns:
            matches = re.findall(pattern, item_headline + " " + item_content[:200])
            entities.update(match.lower() if isinstance(match, str) else " ".join(match).lower() 
                          for match in matches)
        
        # Key topic keywords
        topic_keywords = [
            "artificial intelligence", "ai", "machine learning", "chatgpt", "openai",
            "federal reserve", "fed", "interest rates", "inflation", "recession",
            "quantum", "blockchain", "cryptocurrency", "bitcoin", "climate change",
            "russia", "ukraine", "china", "congress", "senate", "supreme court"
        ]
        
        found_keywords = [kw for kw in topic_keywords if kw in search_text]
        
        if not entities and not found_keywords:
            return []
        
        # Search in historical cache
        cutoff = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff.isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Build search conditions
            search_conditions = []
            search_params = [cutoff_str]
            
            if section:
                search_conditions.append("section = ?")
                search_params.append(section)
            
            # Search for entity and keyword matches
            entity_conditions = []
            for entity in list(entities)[:10]:  # Limit to top 10 entities
                entity_conditions.append("headline LIKE ?")
                search_params.append(f"%{entity}%")
            
            for keyword in found_keywords:
                entity_conditions.append("headline LIKE ?")
                search_params.append(f"%{keyword}%")
            
            if not entity_conditions:
                return []
            
            base_query = """
                SELECT id, headline, first_seen_date, section, was_included, 
                       '' as content_preview
                FROM seen_items 
                WHERE first_seen_date >= ?
            """
            
            if search_conditions:
                base_query += " AND " + " AND ".join(search_conditions)
            
            base_query += " AND (" + " OR ".join(entity_conditions) + ")"
            base_query += " ORDER BY first_seen_date DESC LIMIT 20"
            
            cursor = await db.execute(base_query, search_params)
            rows = await cursor.fetchall()
            
            connections = []
            for row in rows:
                # Calculate connection strength
                connection_strength = 0.0
                historical_text = f"{row[1]} {row[5]}".lower()  # headline + content_preview
                
                # Entity match scoring
                for entity in entities:
                    if entity in historical_text:
                        connection_strength += 0.3
                
                # Keyword match scoring
                for keyword in found_keywords:
                    if keyword in historical_text:
                        connection_strength += 0.4
                
                # Recency bonus (more recent = stronger connection)
                try:
                    hist_date = datetime.fromisoformat(row[2])
                    days_ago = (datetime.now() - hist_date).days
                    recency_factor = max(0.1, 1.0 - (days_ago / days_back))
                    connection_strength *= recency_factor
                except:
                    pass
                
                # Inclusion bonus (previously featured stories are more significant)
                if row[4]:  # was_included
                    connection_strength *= 1.5
                
                if connection_strength >= 0.2:  # Minimum threshold
                    connections.append({
                        "item_id": row[0],
                        "headline": row[1],
                        "first_seen": row[2],
                        "section": row[3],
                        "was_included": bool(row[4]),
                        "connection_strength": round(connection_strength, 3),
                        "content_preview": row[5]
                    })
            
            # Sort by connection strength
            connections.sort(key=lambda x: x["connection_strength"], reverse=True)
            return connections[:5]  # Return top 5 connections

    async def detect_cross_section_themes(self, current_stories: Dict[str, List[Dict]], 
                                         embedding_service, days_back: int = 7) -> Dict[str, Any]:
        """
        Detect semantic themes across sections using embedding-based clustering.
        
        Args:
            current_stories: Stories being considered {section: [stories]}
            embedding_service: EmbeddingService for generating embeddings
            days_back: How many days back to analyze for historical patterns
            
        Returns:
            Dict containing semantic theme analysis and recommendations
        """
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_similarity
        
        analysis = {
            "semantic_clusters": [],
            "cross_section_themes": [],
            "theme_overlap_warnings": [],
            "diversity_recommendations": [],
            "historical_context": {}
        }
        
        try:
            # Prepare current stories for embedding analysis
            story_data = []
            story_embeddings = []
            
            for section, stories in current_stories.items():
                for idx, story in enumerate(stories):
                    story_text = f"{story.get('headline', '')} {story.get('content', '')[:500]}"
                    if story_text.strip():
                        story_data.append({
                            'id': f"{section}_{idx}",
                            'section': section,
                            'headline': story.get('headline', ''),
                            'text': story_text,
                            'story_data': story
                        })
            
            if not story_data:
                return analysis
            
            # Generate embeddings for all current stories
            texts_to_embed = [item['text'] for item in story_data]
            embeddings = await embedding_service.batch_generate(texts_to_embed)
            
            # Convert to numpy arrays for clustering
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Perform DBSCAN clustering to identify semantic themes
            # eps=0.3 means stories with >70% cosine similarity are clustered together
            # min_samples=2 means need at least 2 stories to form a cluster
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(embeddings_array)
            
            # Analyze clusters
            clusters = {}
            for idx, cluster_id in enumerate(clustering.labels_):
                if cluster_id != -1:  # -1 means noise/outlier
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append({
                        'story_idx': idx,
                        'story_data': story_data[idx],
                        'embedding': embeddings[idx]
                    })
            
            # Generate semantic cluster analysis
            semantic_clusters = []
            for cluster_id, cluster_stories in clusters.items():
                # Calculate cluster centroid
                cluster_embeddings = np.array([item['embedding'] for item in cluster_stories])
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate internal cohesion (average similarity within cluster)
                similarities = cosine_similarity(cluster_embeddings)
                # Get upper triangle excluding diagonal
                upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
                cohesion = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
                
                # Extract sections involved
                sections_involved = list(set(item['story_data']['section'] for item in cluster_stories))
                
                # Generate theme label by finding common entities/concepts
                theme_label = self._generate_cluster_theme_label(cluster_stories)
                
                semantic_clusters.append({
                    'cluster_id': cluster_id,
                    'theme_label': theme_label,
                    'story_count': len(cluster_stories),
                    'sections_involved': sections_involved,
                    'cohesion_score': round(cohesion, 3),
                    'stories': [item['story_data'] for item in cluster_stories],
                    'centroid_embedding': centroid.tolist()
                })
            
            # Sort by cohesion score (most coherent themes first)
            semantic_clusters.sort(key=lambda x: x['cohesion_score'], reverse=True)
            analysis['semantic_clusters'] = semantic_clusters
            
            # Identify cross-section themes (clusters spanning multiple sections)
            cross_section_themes = [
                cluster for cluster in semantic_clusters 
                if len(cluster['sections_involved']) >= 2
            ]
            analysis['cross_section_themes'] = cross_section_themes
            
            # Check for theme overlap with historical newsletters
            historical_overlap = await self._analyze_historical_theme_overlap(
                semantic_clusters, embedding_service, days_back
            )
            analysis['theme_overlap_warnings'] = historical_overlap['warnings']
            analysis['historical_context'] = historical_overlap['context']
            
            # Generate diversity recommendations based on semantic space coverage
            diversity_recs = await self._analyze_semantic_diversity(
                embeddings_array, story_data, embedding_service
            )
            analysis['diversity_recommendations'] = diversity_recs
            
        except Exception as e:
            self.logger.error(f"Error in semantic theme detection: {e}")
            # Return minimal analysis on error
            analysis['error'] = str(e)
        
        return analysis

    def _generate_cluster_theme_label(self, cluster_stories: List[Dict]) -> str:
        """
        Generate a descriptive label for a semantic cluster by finding common elements.
        """
        import re
        from collections import Counter
        
        # Extract potential theme indicators
        all_headlines = [item['story_data']['headline'] for item in cluster_stories]
        all_text = ' '.join(all_headlines).lower()
        
        # Extract meaningful words (excluding common words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'new', 'says'
        }
        
        # Extract significant words (3+ chars, not stop words)
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        meaningful_words = [word for word in words if word not in stop_words]
        
        # Extract potential entities (capitalized words from original headlines)
        entities = []
        for headline in all_headlines:
            words_in_headline = headline.split()
            for word in words_in_headline:
                if len(word) > 2 and word[0].isupper() and word.isalpha():
                    entities.append(word)
        
        # Count frequency and find most common
        word_counts = Counter(meaningful_words)
        entity_counts = Counter(entities)
        
        # Generate label prioritizing entities, then common words
        label_parts = []
        
        # Add most common entity if available
        if entity_counts:
            top_entity = entity_counts.most_common(1)[0][0]
            label_parts.append(top_entity)
        
        # Add most common meaningful word if different from entity
        if word_counts:
            top_word = word_counts.most_common(1)[0][0]
            if not label_parts or top_word.lower() not in [p.lower() for p in label_parts]:
                label_parts.append(top_word)
        
        # Create label
        if label_parts:
            return ' + '.join(label_parts[:2])  # Max 2 components
        else:
            return f"theme_{len(cluster_stories)}_stories"

    async def _analyze_historical_theme_overlap(self, current_clusters: List[Dict], 
                                               embedding_service, days_back: int) -> Dict:
        """
        Analyze overlap between current semantic themes and historical newsletter themes.
        """
        overlap_analysis = {
            'warnings': [],
            'context': {}
        }
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cutoff_date = (datetime.now(EASTERN_TZ) - timedelta(days=days_back)).date().isoformat()
                
                # Get recent newsletter content with embeddings
                cursor = await db.execute(
                    """
                    SELECT headline, '', content_embedding, section, first_seen_date
                    FROM seen_items 
                    WHERE first_seen_date >= ? AND was_included = 1 AND content_embedding IS NOT NULL
                    ORDER BY first_seen_date DESC
                    LIMIT 50
                    """,
                    (cutoff_date,)
                )
                
                historical_rows = await cursor.fetchall()
                if not historical_rows:
                    return overlap_analysis
                
                # Load historical embeddings
                historical_embeddings = []
                historical_data = []
                
                for row in historical_rows:
                    try:
                        import pickle
                        embedding = pickle.loads(row[2])
                        historical_embeddings.append(embedding)
                        historical_data.append({
                            'headline': row[0],
                            'content': row[1],
                            'section': row[3], 
                            'date': row[4]
                        })
                    except Exception:
                        continue
                
                if not historical_embeddings:
                    return overlap_analysis
                
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                
                historical_embeddings_array = np.array(historical_embeddings)
                
                # Check each current cluster against historical content
                for cluster in current_clusters:
                    cluster_centroid = np.array(cluster['centroid_embedding']).reshape(1, -1)
                    
                    # Calculate similarity with all historical content
                    similarities = cosine_similarity(cluster_centroid, historical_embeddings_array)[0]
                    
                    # Find high overlap (>75% similarity)
                    high_overlap_indices = np.where(similarities > 0.75)[0]
                    
                    if len(high_overlap_indices) >= 2:  # Multiple similar historical pieces
                        overlap_stories = [historical_data[i] for i in high_overlap_indices]
                        max_similarity = np.max(similarities)
                        
                        overlap_analysis['warnings'].append({
                            'cluster_theme': cluster['theme_label'],
                            'cluster_sections': cluster['sections_involved'],
                            'max_similarity': round(max_similarity, 3),
                            'overlap_count': len(high_overlap_indices),
                            'historical_examples': overlap_stories[:3],  # Top 3 examples
                            'risk_level': 'high' if max_similarity > 0.85 else 'moderate'
                        })
                
                # Store historical context summary
                overlap_analysis['context'] = {
                    'historical_stories_analyzed': len(historical_data),
                    'analysis_period_days': days_back,
                    'clusters_analyzed': len(current_clusters)
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to analyze historical theme overlap: {e}")
        
        return overlap_analysis

    async def _analyze_semantic_diversity(self, embeddings_array, story_data: List[Dict], 
                                         embedding_service) -> List[Dict]:
        """
        Analyze semantic diversity and suggest improvements.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        recommendations = []
        
        try:
            # Calculate pairwise similarities between all stories
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Get upper triangle (excluding diagonal) for diversity analysis
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            # Calculate diversity metrics
            avg_similarity = np.mean(upper_triangle)
            similarity_std = np.std(upper_triangle)
            
            # Analyze section-level diversity
            section_diversity = {}
            sections = list(set(item['section'] for item in story_data))
            
            for section in sections:
                section_indices = [i for i, item in enumerate(story_data) if item['section'] == section]
                
                if len(section_indices) > 1:
                    section_embeddings = embeddings_array[section_indices]
                    section_similarities = cosine_similarity(section_embeddings)
                    section_upper = section_similarities[np.triu_indices_from(section_similarities, k=1)]
                    section_avg_sim = np.mean(section_upper)
                    
                    section_diversity[section] = {
                        'avg_similarity': round(section_avg_sim, 3),
                        'story_count': len(section_indices),
                        'diversity_score': round(1 - section_avg_sim, 3)  # Higher is more diverse
                    }
            
            # Generate recommendations
            overall_diversity = round(1 - avg_similarity, 3)
            
            if overall_diversity < 0.6:  # Low diversity threshold
                recommendations.append({
                    'type': 'overall_diversity',
                    'current_score': overall_diversity,
                    'recommendation': 'Consider adding stories from more diverse topics to increase newsletter variety',
                    'priority': 'high' if overall_diversity < 0.4 else 'medium'
                })
            
            # Section-specific recommendations
            for section, metrics in section_diversity.items():
                if metrics['diversity_score'] < 0.5 and metrics['story_count'] > 2:
                    recommendations.append({
                        'type': 'section_diversity',
                        'section': section,
                        'current_score': metrics['diversity_score'],
                        'story_count': metrics['story_count'],
                        'recommendation': f'Stories in {section} section are semantically similar - consider more diverse angles',
                        'priority': 'medium'
                    })
            
            # Identify potential semantic gaps by finding underrepresented areas
            if len(story_data) >= 5:  # Only for reasonable story counts
                # Find stories that are most dissimilar to others (potential unique perspectives)
                dissimilarity_scores = 1 - np.mean(similarity_matrix, axis=1)
                most_unique_idx = np.argmax(dissimilarity_scores)
                
                if dissimilarity_scores[most_unique_idx] > 0.7:  # Very unique story
                    unique_story = story_data[most_unique_idx]
                    recommendations.append({
                        'type': 'leverage_uniqueness',
                        'story': unique_story['headline'],
                        'section': unique_story['section'],
                        'uniqueness_score': round(dissimilarity_scores[most_unique_idx], 3),
                        'recommendation': 'This story offers a unique perspective - consider highlighting it for newsletter diversity'
                    })
        
        except Exception as e:
            self.logger.warning(f"Failed to analyze semantic diversity: {e}")
        
        return recommendations

    async def get_newsletter_themes(self, days_back: int = 14) -> dict:
        """
        Analyze recent newsletter themes to understand ongoing narratives.
        
        Args:
            days_back: Days to look back for theme analysis
            
        Returns:
            Dict with theme analysis and trending topics
        """
        cutoff = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff.isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Get recently included stories
            cursor = await db.execute(
                """SELECT headline, section, first_seen_date, 
                          '' as content_preview
                   FROM seen_items 
                   WHERE was_included = 1 AND first_seen_date >= ?
                   ORDER BY first_seen_date DESC""",
                (cutoff_str,)
            )
            rows = await cursor.fetchall()
            
            # Simple theme extraction
            theme_keywords = {}
            section_themes = {}
            
            key_themes = [
                "artificial intelligence", "ai", "machine learning", "chatgpt",
                "federal reserve", "inflation", "interest rates", "recession",
                "quantum", "blockchain", "cryptocurrency", "climate change",
                "russia", "ukraine", "china", "geopolitical", "war",
                "startup", "venture capital", "ipo", "merger", "acquisition",
                "breakthrough", "discovery", "research", "study",
                "congress", "senate", "election", "politics", "supreme court"
            ]
            
            for row in rows:
                text = f"{row[0]} {row[3]}".lower()
                section = row[1]
                
                if section not in section_themes:
                    section_themes[section] = {}
                
                for theme in key_themes:
                    if theme in text:
                        theme_keywords[theme] = theme_keywords.get(theme, 0) + 1
                        section_themes[section][theme] = section_themes[section].get(theme, 0) + 1
            
            # Sort themes by frequency
            trending_themes = sorted(theme_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "trending_themes": trending_themes,
                "section_themes": section_themes,
                "total_stories_analyzed": len(rows),
                "analysis_period_days": days_back
            }


