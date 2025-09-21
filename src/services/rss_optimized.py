"""
Optimized RSS Service with production-ready performance architecture.

Key Improvements:
1. Intelligent feed prioritization and circuit breaking  
2. Concurrent processing with backpressure control
3. Smart caching with TTL and content-based invalidation
4. Performance monitoring and adaptive timeout
5. Feed health scoring and automatic fallbacks
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import sqlite3
from pathlib import Path

# Import from existing RSS service
from src.services.rss import (
    RSSItem, FeedConfig, RSSService as BaseRSSService, 
    RSSServiceError, EXTERNAL_DEPS_AVAILABLE
)

if EXTERNAL_DEPS_AVAILABLE:
    import aiohttp
    from zoneinfo import ZoneInfo
    EASTERN_TZ = ZoneInfo("America/New_York")
else:
    EASTERN_TZ = None


class FeedStatus(Enum):
    """Feed health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    FAILING = "failing"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class FeedPerformanceMetrics:
    """Performance metrics for a feed"""
    feed_id: str
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    circuit_breaker_until: Optional[datetime] = None
    consecutive_failures: int = 0
    total_items_fetched: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate over recent attempts"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def status(self) -> FeedStatus:
        """Determine current feed status"""
        now = datetime.now()
        
        # Check circuit breaker
        if (self.circuit_breaker_until and 
            now < self.circuit_breaker_until):
            return FeedStatus.CIRCUIT_OPEN
            
        # Check failure patterns
        if self.consecutive_failures >= 5:
            return FeedStatus.FAILING
        elif self.consecutive_failures >= 2 or self.success_rate() < 0.7:
            return FeedStatus.DEGRADED
        else:
            return FeedStatus.HEALTHY


@dataclass 
class CachedFeedResult:
    """Cached feed result with metadata"""
    items: List[RSSItem]
    cached_at: datetime
    feed_url: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    
    def is_stale(self, ttl_seconds: int) -> bool:
        """Check if cache entry is stale"""
        age = (datetime.now() - self.cached_at).total_seconds()
        return age > ttl_seconds


class OptimizedRSSService(BaseRSSService):
    """
    Production-optimized RSS service with intelligent concurrency control,
    caching, circuit breaking, and performance monitoring.
    """
    
    def __init__(self, 
                 max_concurrent_feeds: int = 15,
                 default_feed_timeout: int = 10,
                 cache_ttl_seconds: int = 1800,  # 30 minutes
                 performance_db_path: str = "rss_performance.db"):
        """Initialize optimized RSS service"""
        super().__init__()
        
        # Performance configuration
        self.max_concurrent_feeds = max_concurrent_feeds
        self.default_feed_timeout = default_feed_timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_feeds)
        self.active_feeds: Set[str] = set()
        
        # Performance tracking
        self.performance_db_path = performance_db_path
        self.performance_metrics: Dict[str, FeedPerformanceMetrics] = {}
        self.session_stats: Dict[str, Any] = {
            "session_start": datetime.now(),
            "feeds_attempted": 0,
            "feeds_successful": 0,
            "total_items": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        # Enhanced caching
        self.feed_cache: Dict[str, CachedFeedResult] = {}
        
        # Initialize performance database
        self._init_performance_db()
        self._load_performance_metrics()
        
        self.logger.info(
            f"Initialized OptimizedRSSService: "
            f"max_concurrent={max_concurrent_feeds}, "
            f"timeout={default_feed_timeout}s, "
            f"cache_ttl={cache_ttl_seconds}s"
        )

    def _init_performance_db(self):
        """Initialize SQLite database for performance metrics"""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feed_performance (
                        feed_id TEXT PRIMARY KEY,
                        success_count INTEGER DEFAULT 0,
                        failure_count INTEGER DEFAULT 0,
                        avg_response_time REAL DEFAULT 0.0,
                        last_success TIMESTAMP,
                        last_failure TIMESTAMP,
                        consecutive_failures INTEGER DEFAULT 0,
                        total_items_fetched INTEGER DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize performance DB: {e}")

    def _load_performance_metrics(self):
        """Load performance metrics from database"""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM feed_performance")
                for row in cursor:
                    feed_id = row['feed_id']
                    self.performance_metrics[feed_id] = FeedPerformanceMetrics(
                        feed_id=feed_id,
                        success_count=row['success_count'],
                        failure_count=row['failure_count'],
                        avg_response_time=row['avg_response_time'],
                        last_success=datetime.fromisoformat(row['last_success']) if row['last_success'] else None,
                        last_failure=datetime.fromisoformat(row['last_failure']) if row['last_failure'] else None,
                        consecutive_failures=row['consecutive_failures'],
                        total_items_fetched=row['total_items_fetched'],
                    )
        except Exception as e:
            self.logger.error(f"Failed to load performance metrics: {e}")

    def _save_performance_metrics(self):
        """Save performance metrics to database"""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                for feed_id, metrics in self.performance_metrics.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO feed_performance 
                        (feed_id, success_count, failure_count, avg_response_time,
                         last_success, last_failure, consecutive_failures, 
                         total_items_fetched, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        feed_id,
                        metrics.success_count,
                        metrics.failure_count,
                        metrics.avg_response_time,
                        metrics.last_success.isoformat() if metrics.last_success else None,
                        metrics.last_failure.isoformat() if metrics.last_failure else None,
                        metrics.consecutive_failures,
                        metrics.total_items_fetched,
                    ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}")

    async def fetch_feeds_by_section_optimized(self, 
                                             section: str, 
                                             target_items: int = 50,
                                             max_feeds_per_section: int = 10) -> List[RSSItem]:
        """
        Optimized section feed fetching with intelligent prioritization.
        
        Strategy:
        1. Prioritize feeds by health and authority
        2. Fetch concurrently with circuit breaking
        3. Stop early when target items reached
        4. Use cached results when appropriate
        """
        start_time = time.time()
        
        # Get section feeds sorted by priority
        section_feed_configs = self._get_prioritized_section_feeds(section, max_feeds_per_section)
        
        if not section_feed_configs:
            self.logger.warning(f"No healthy feeds available for section: {section}")
            return []
            
        self.logger.info(
            f"Fetching {section} content from {len(section_feed_configs)} prioritized feeds "
            f"(target: {target_items} items)"
        )
        
        # Concurrent fetch with early termination
        all_items = []
        completed_feeds = 0
        
        # Process feeds in priority batches
        batch_size = min(self.max_concurrent_feeds, len(section_feed_configs))
        
        for i in range(0, len(section_feed_configs), batch_size):
            batch = section_feed_configs[i:i + batch_size]
            
            # Execute batch concurrently
            batch_tasks = [
                self._fetch_feed_with_metrics(config, section)
                for config in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                config = batch[j]
                completed_feeds += 1
                
                if isinstance(result, Exception):
                    self.logger.warning(f"Feed {config.name} failed: {result}")
                    continue
                    
                items, metrics = result
                if items:
                    all_items.extend(items)
                    self.logger.debug(f"Got {len(items)} items from {config.name}")
                
                # Early termination check
                if len(all_items) >= target_items:
                    self.logger.info(
                        f"Reached target {target_items} items from {completed_feeds}/{len(section_feed_configs)} feeds"
                    )
                    break
                    
            # Break outer loop if target reached
            if len(all_items) >= target_items:
                break
        
        # Sort by recency and quality
        all_items.sort(key=lambda x: (
            x.published_date or datetime.min,
            1 if "⭐" in x.title else 0
        ), reverse=True)
        
        elapsed = time.time() - start_time
        self.session_stats["feeds_attempted"] += completed_feeds
        self.session_stats["total_items"] += len(all_items)
        
        self.logger.info(
            f"Section {section} completed: {len(all_items)} items from "
            f"{completed_feeds} feeds in {elapsed:.2f}s"
        )
        
        return all_items[:target_items]

    def _get_prioritized_section_feeds(self, 
                                     section: str, 
                                     max_feeds: int) -> List[FeedConfig]:
        """Get section feeds prioritized by health and authority"""
        if section not in self.section_feeds:
            return []
            
        feed_ids = self.section_feeds[section]
        configs = [self.feeds[feed_id] for feed_id in feed_ids]
        
        # Filter out circuit-broken feeds
        healthy_configs = []
        for config in configs:
            feed_id = self._get_feed_id(config)
            metrics = self.performance_metrics.get(feed_id)
            
            if not metrics:
                # New feed, give it a chance
                healthy_configs.append(config)
            elif metrics.status() != FeedStatus.CIRCUIT_OPEN:
                healthy_configs.append(config)
            else:
                self.logger.debug(f"Skipping circuit-open feed: {config.name}")
        
        # Sort by priority score (health + authority)
        def priority_score(config: FeedConfig) -> float:
            feed_id = self._get_feed_id(config)
            metrics = self.performance_metrics.get(feed_id)
            
            base_priority = config.priority
            
            if metrics:
                # Boost for reliable feeds
                reliability_bonus = metrics.success_rate() * 2
                
                # Speed bonus (faster feeds ranked higher)
                speed_bonus = max(0, 2 - metrics.avg_response_time / 5)
                
                # Recent success bonus
                recency_bonus = 0
                if metrics.last_success:
                    hours_since_success = (datetime.now() - metrics.last_success).total_seconds() / 3600
                    if hours_since_success < 1:
                        recency_bonus = 1
                    elif hours_since_success < 6:
                        recency_bonus = 0.5
                        
                return base_priority + reliability_bonus + speed_bonus + recency_bonus
            else:
                return base_priority
        
        healthy_configs.sort(key=priority_score, reverse=True)
        return healthy_configs[:max_feeds]

    async def _fetch_feed_with_metrics(self, 
                                     config: FeedConfig, 
                                     section: str) -> Tuple[List[RSSItem], FeedPerformanceMetrics]:
        """Fetch feed with performance tracking and circuit breaking"""
        feed_id = self._get_feed_id(config)
        
        # Get or create metrics
        if feed_id not in self.performance_metrics:
            self.performance_metrics[feed_id] = FeedPerformanceMetrics(feed_id=feed_id)
        
        metrics = self.performance_metrics[feed_id]
        
        # Check circuit breaker
        if metrics.status() == FeedStatus.CIRCUIT_OPEN:
            self.logger.debug(f"Circuit breaker open for {config.name}")
            return [], metrics
        
        # Check cache first
        cache_key = self._get_cache_key(config.url, section)
        if cache_key in self.feed_cache:
            cached = self.feed_cache[cache_key]
            if not cached.is_stale(self.cache_ttl_seconds):
                # Filter cached items by age to prevent serving stale articles
                fresh_items = []
                for item in cached.items:
                    if item.published_date:
                        item_age_hours = (datetime.now() - item.published_date.replace(tzinfo=None)).total_seconds() / 3600
                        if item_age_hours <= 168:  # 1 week max
                            fresh_items.append(item)

                if fresh_items:
                    self.session_stats["cache_hits"] += 1
                    self.logger.debug(f"Cache hit for {config.name}: {len(fresh_items)} fresh items")
                    return fresh_items, metrics
                else:
                    # All cached items are stale, remove from cache
                    del self.feed_cache[cache_key]
                    self.logger.debug(f"All cached items for {config.name} are stale")
        
        # Fetch with concurrency control
        async with self.semaphore:
            return await self._fetch_feed_with_timeout_and_tracking(config, metrics, cache_key)

    async def _fetch_feed_with_timeout_and_tracking(self, 
                                                   config: FeedConfig,
                                                   metrics: FeedPerformanceMetrics,
                                                   cache_key: str) -> Tuple[List[RSSItem], FeedPerformanceMetrics]:
        """Fetch feed with adaptive timeout and performance tracking"""
        start_time = time.time()
        
        # Adaptive timeout based on historical performance
        timeout = self._calculate_adaptive_timeout(metrics)
        
        try:
            # Fetch with timeout
            items = await asyncio.wait_for(
                self.fetch_feed(config.url, max_items=20, filter_quality=True),
                timeout=timeout
            )
            
            # Success - update metrics
            response_time = time.time() - start_time
            metrics.success_count += 1
            metrics.consecutive_failures = 0
            metrics.last_success = datetime.now()
            metrics.total_items_fetched += len(items)
            
            # Update average response time (exponential moving average)
            if metrics.avg_response_time == 0:
                metrics.avg_response_time = response_time
            else:
                metrics.avg_response_time = (metrics.avg_response_time * 0.7 + response_time * 0.3)
            
            # Cache result
            self.feed_cache[cache_key] = CachedFeedResult(
                items=items,
                cached_at=datetime.now(),
                feed_url=config.url
            )
            self.session_stats["cache_misses"] += 1
            self.session_stats["feeds_successful"] += 1
            
            self.logger.debug(
                f"Successfully fetched {len(items)} items from {config.name} "
                f"in {response_time:.2f}s"
            )
            
            return items, metrics
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching {config.name} ({timeout}s)")
            return await self._handle_fetch_failure(config, metrics, "timeout")
            
        except Exception as e:
            self.logger.warning(f"Error fetching {config.name}: {e}")
            return await self._handle_fetch_failure(config, metrics, str(e))

    async def _handle_fetch_failure(self, 
                                  config: FeedConfig, 
                                  metrics: FeedPerformanceMetrics, 
                                  error: str) -> Tuple[List[RSSItem], FeedPerformanceMetrics]:
        """Handle feed fetch failure with circuit breaking logic"""
        metrics.failure_count += 1
        metrics.consecutive_failures += 1
        metrics.last_failure = datetime.now()
        
        # Circuit breaker logic
        if metrics.consecutive_failures >= 5:
            # Open circuit for 10 minutes
            metrics.circuit_breaker_until = datetime.now() + timedelta(minutes=10)
            self.logger.warning(
                f"Circuit breaker opened for {config.name} "
                f"({metrics.consecutive_failures} consecutive failures)"
            )
        
        return [], metrics

    def _calculate_adaptive_timeout(self, metrics: FeedPerformanceMetrics) -> float:
        """Calculate adaptive timeout based on feed performance history"""
        base_timeout = self.default_feed_timeout
        
        if metrics.avg_response_time > 0:
            # Use 2x historical average, capped at 15 seconds
            adaptive_timeout = min(15, metrics.avg_response_time * 2)
            return max(5, adaptive_timeout)  # Minimum 5 seconds
        
        return base_timeout

    def _get_feed_id(self, config: FeedConfig) -> str:
        """Generate consistent feed ID"""
        return hashlib.md5(f"{config.section}_{config.url}".encode()).hexdigest()[:12]

    def _get_cache_key(self, url: str, section: str) -> str:
        """Generate cache key for feed"""
        return hashlib.md5(f"{url}_{section}".encode()).hexdigest()

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        now = datetime.now()
        session_duration = (now - self.session_stats["session_start"]).total_seconds()
        
        # Calculate feed health distribution
        feed_health = {status.value: 0 for status in FeedStatus}
        for metrics in self.performance_metrics.values():
            feed_health[metrics.status().value] += 1
        
        # Top performing feeds
        top_feeds = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1].success_rate(),
            reverse=True
        )[:10]
        
        return {
            "session_stats": {
                **self.session_stats,
                "session_duration_seconds": session_duration,
                "feeds_per_minute": (self.session_stats["feeds_attempted"] / session_duration * 60) if session_duration > 0 else 0,
                "cache_hit_rate": (self.session_stats["cache_hits"] / 
                                 (self.session_stats["cache_hits"] + self.session_stats["cache_misses"])) 
                                 if (self.session_stats["cache_hits"] + self.session_stats["cache_misses"]) > 0 else 0,
            },
            "feed_health": feed_health,
            "total_feeds_tracked": len(self.performance_metrics),
            "cache_size": len(self.feed_cache),
            "top_performing_feeds": [
                {
                    "feed_id": feed_id,
                    "success_rate": metrics.success_rate(),
                    "avg_response_time": metrics.avg_response_time,
                    "total_items": metrics.total_items_fetched,
                }
                for feed_id, metrics in top_feeds
            ],
        }

    def __del__(self):
        """Save metrics on cleanup"""
        try:
            self._save_performance_metrics()
        except:
            pass


# Factory function for backward compatibility
def create_optimized_rss_service(**kwargs) -> OptimizedRSSService:
    """Create optimized RSS service with custom configuration"""
    return OptimizedRSSService(**kwargs)