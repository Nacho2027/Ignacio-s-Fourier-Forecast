#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
import signal
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from zoneinfo import ZoneInfo

# Services (ensure names match for tests' patching)
from src.services.cache_service import CacheService
from src.services.cache_service import ContentItem as CacheContentItem
from src.utils.embeddings import EmbeddingService
from src.services.ai_service import AIService
from src.services.llmlayer import LLMLayerService
from src.services.arxiv import ArxivService
from src.services.rss import RSSService
from src.services.email_service import EmailService, EmailConfig
from src.services.deduplication_service import (
    DeduplicationService,
    ContentItem as DedupContentItem,
)
from src.services.summarization_service import SummarizationService
from src.services.synthesis_service import SynthesisService
from src.services.semantic_scholar_service import SemanticScholarService

# Pipeline components
from src.pipeline.content_aggregator import ContentAggregator
from src.pipeline.email_compiler import EmailCompiler


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # API Keys
    anthropic_api_key: str  # For Claude AI service
    voyage_api_key: str  # For Voyage AI embeddings service
    llmlayer_api_key: str

    # Email settings
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    recipient_email: str

    # Timing
    execution_time: time  # 5:00 AM ET
    delivery_time: time   # 5:30 AM ET
    max_execution_minutes: int = 20  # Reasonable time for full pipeline execution

    # Paths
    database_path: str = "data/newsletter.db"
    template_dir: str = "templates"
    log_dir: str = "logs"

    # Features
    dry_run: bool = False
    test_mode: bool = False
    send_test_email: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Execution metrics"""
    start_time: datetime
    end_time: Optional[datetime] = None

    # Stage timings
    fetch_time: float = 0.0
    dedup_time: float = 0.0
    rank_time: float = 0.0
    summarize_time: float = 0.0
    synthesis_time: float = 0.0
    compile_time: float = 0.0
    send_time: float = 0.0

    # Counts
    items_fetched: int = 0
    items_after_dedup: int = 0
    items_selected: int = 0
    sections_created: int = 0

    # Success flags
    fetch_success: bool = False
    dedup_success: bool = False
    summary_success: bool = False
    synthesis_success: bool = False
    compile_success: bool = False
    send_success: bool = False

    # Detailed dedup statistics
    url_filtered: int = 0
    title_filtered: int = 0
    ai_filtered: int = 0
    follow_ups_detected: int = 0
    different_angles_kept: int = 0

    def total_time(self) -> float:
        """Calculate total execution time"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class MainPipeline:
    """
    Main orchestrator for daily newsletter generation.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or self._load_config()
        self.metrics: Optional[PipelineMetrics] = None
        self.services: Dict[str, Any] = {}

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Eastern timezone for scheduling (zoneinfo for robust DST handling)
        self.et_timezone = ZoneInfo('America/New_York')

        # Graceful shutdown
        self.shutdown_event = asyncio.Event()
        try:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except Exception:
            pass

    def _load_config(self) -> PipelineConfig:
        """Load configuration from environment and files"""
        # Create default directories
        try:
            Path(self._resolve_path('logs')).mkdir(parents=True, exist_ok=True)
            Path(self._resolve_path('data')).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return PipelineConfig(
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', ''),
            voyage_api_key=os.getenv('VOYAGE_API_KEY', ''),
            llmlayer_api_key=os.getenv('LLMLAYER_API_KEY', ''),
            smtp_host=os.getenv('SMTP_HOST', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            smtp_user=os.getenv('SMTP_USER', ''),
            smtp_password=os.getenv('SMTP_PASSWORD', ''),
            recipient_email=os.getenv('RECIPIENT_EMAIL', 'ignacio@example.com'),
            execution_time=time(5, 0),  # 5:00 AM
            delivery_time=time(5, 30),  # 5:30 AM
            max_execution_minutes=int(os.getenv('MAX_EXECUTION_MINUTES', '120')),  # 120 minutes for full pipeline with LLM calls
            database_path=os.getenv('DATABASE_PATH', 'data/newsletter.db'),
            template_dir=os.getenv('TEMPLATE_DIR', 'templates'),
            log_dir=os.getenv('LOG_DIR', 'logs'),
            dry_run=(os.getenv('DRY_RUN', 'false').lower() == 'true'),
            test_mode=(os.getenv('TEST_MODE', 'false').lower() == 'true'),
            send_test_email=os.getenv('SEND_TEST_EMAIL'),
        )

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = Path(self._resolve_path(self.config.log_dir if self.config else 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{datetime.now():%Y-%m-%d}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def _resolve_path(self, path: str) -> str:
        try:
            return str(Path(path))
        except Exception:
            return path

    async def initialize_services(self) -> Dict[str, Any]:
        """
        Initialize all required services.
        """
        cfg = self.config
        if not cfg.anthropic_api_key:
            raise PipelineError("Missing ANTHROPIC_API_KEY")
        if not cfg.voyage_api_key:
            raise PipelineError("Missing VOYAGE_API_KEY (required for embeddings)")
        if not cfg.llmlayer_api_key:
            raise PipelineError("Missing LLMLAYER_API_KEY")
        if not cfg.smtp_user or not cfg.smtp_password:
            raise PipelineError("Missing SMTP credentials")

        cache = CacheService(db_path=cfg.database_path)
        try:
            if hasattr(cache, 'initialize_db'):
                await cache.initialize_db()
        except Exception:
            pass

        ai = AIService(api_key=cfg.anthropic_api_key)
        llmlayer = LLMLayerService(api_key=cfg.llmlayer_api_key)
        arxiv = ArxivService()
        rss = RSSService()
        # Initialize Semantic Scholar service (optional API key for higher rate limits)
        semantic_scholar = SemanticScholarService(api_key=os.getenv('SEMANTIC_SCHOLAR_API_KEY'))
        # Construct EmailService with explicit config to avoid env dependency in tests
        email_cfg = EmailConfig(
            smtp_host=cfg.smtp_host,
            smtp_port=cfg.smtp_port,
            smtp_user=cfg.smtp_user,
            smtp_password=cfg.smtp_password,
            from_email=cfg.smtp_user,
            to_email=cfg.recipient_email,
            reply_to=None,
            use_tls=True,
        )
        email = EmailService(email_cfg)
        embeddings = EmbeddingService(api_key=cfg.voyage_api_key)

        aggregator = ContentAggregator(llmlayer, arxiv, rss, ai, cache, embeddings, semantic_scholar)
        dedup = DeduplicationService(cache, embeddings, ai)
        summarization = SummarizationService(ai)
        synthesis = SynthesisService(ai)
        compiler = EmailCompiler(template_dir=cfg.template_dir)

        self.services = {
            'cache': cache,
            'embeddings': embeddings,
            'ai': ai,
            'llmlayer': llmlayer,
            'arxiv': arxiv,
            'rss': rss,
            'semantic_scholar': semantic_scholar,
            'email': email,
            'aggregator': aggregator,
            'deduplication': dedup,
            'summarization': summarization,
            'synthesis': synthesis,
            'compiler': compiler,
        }
        return self.services

    async def run_daily_pipeline(self) -> bool:
        """
        Execute complete daily pipeline.
        """
        self.metrics = PipelineMetrics(start_time=datetime.now())
        try:
            await asyncio.wait_for(self._execute_pipeline(), timeout=self.config.max_execution_minutes * 60)
            return True
        finally:
            if self.metrics:
                self.metrics.end_time = datetime.now()

    async def _execute_pipeline(self) -> None:
        if not self.services:
            await self.initialize_services()

        # Clear old cache entries to prevent cross-day duplicates
        # Only keep items from last 3 days for deduplication purposes
        cache_service = self.services.get('cache')
        if cache_service:
            try:
                removed = await cache_service.cleanup(days=3)
                logging.info(f"Cleared {removed} old cache entries before newsletter generation")
            except Exception as e:
                logging.warning(f"Failed to cleanup old cache entries: {e}")

        # Stage 1: Fetch
        start = asyncio.get_event_loop().time()
        raw_content = await self._fetch_content()
        self.metrics.fetch_time = asyncio.get_event_loop().time() - start
        self.metrics.fetch_success = True

        # Stage 2: Deduplicate
        start = asyncio.get_event_loop().time()
        deduped = await self._deduplicate_content(raw_content)
        self.metrics.dedup_time = asyncio.get_event_loop().time() - start
        self.metrics.dedup_success = True
        # Capture detailed dedup stats
        try:
            stats = self.services['deduplication'].get_statistics()  # type: ignore[attr-defined]
            self.metrics.url_filtered = int(stats.get('url_filtered', 0))
            self.metrics.title_filtered = int(stats.get('title_filtered', 0))
            self.metrics.ai_filtered = int(stats.get('ai_filtered', 0))
            self.metrics.follow_ups_detected = int(stats.get('follow_ups_detected', 0))
            self.metrics.different_angles_kept = int(stats.get('different_angles_kept', 0))
        except Exception:
            pass

        # Stage 3: Rank & Select
        start = asyncio.get_event_loop().time()
        selected = await self._rank_and_select(deduped)
        self.metrics.rank_time = asyncio.get_event_loop().time() - start
        self.metrics.items_selected = sum(len(v) for v in selected.values())

        # Stage 4: Summaries
        start = asyncio.get_event_loop().time()
        summaries = await self._generate_summaries(selected)
        self.metrics.summarize_time = asyncio.get_event_loop().time() - start
        self.metrics.summary_success = True

        # Stage 5: Synthesis
        start = asyncio.get_event_loop().time()
        golden_thread, surprise = await self._synthesize_insights(summaries)
        self.metrics.synthesis_time = asyncio.get_event_loop().time() - start
        self.metrics.synthesis_success = True

        # Stage 6: Compile
        start = asyncio.get_event_loop().time()
        compiled = await self._compile_email(summaries, golden_thread, surprise)
        self.metrics.compile_time = asyncio.get_event_loop().time() - start
        self.metrics.compile_success = True

        # Stage 7: Send
        start = asyncio.get_event_loop().time()
        sent = await self._send_newsletter(compiled)
        self.metrics.send_time = asyncio.get_event_loop().time() - start
        self.metrics.send_success = bool(sent)

        # Stage 8: Cache update (non-critical)
        try:
            await self._update_cache(selected)
        except Exception:
            pass

    async def _fetch_content(self) -> Dict[Any, Any]:
        aggregator: ContentAggregator = self.services['aggregator']
        content = await aggregator.fetch_all_content()
        
        # Debug: Log what sections were fetched
        self.logger.info(f"Fetched sections: {list(content.keys())}")
        for section, items in content.items():
            self.logger.info(f"Section {section}: {len(items)} items fetched")
            
        if self.metrics is not None:
            try:
                self.metrics.items_fetched = sum(len(v) for v in content.values())
            except Exception:
                self.metrics.items_fetched = 0
        return content

    async def _deduplicate_content(self, content: Dict[Any, Any]) -> Dict[Any, Any]:
        dedup: DeduplicationService = self.services['deduplication']
        
        # Separate Scripture immediately - it's completely independent
        scripture_items = content.pop('scripture', None)
        
        # Transform dicts to DedupContentItem for non-Scripture sections only
        transformed: Dict[str, List[DedupContentItem]] = {}
        for section, items in content.items():
            as_items: List[DedupContentItem] = []
            for idx, it in enumerate(items or []):
                url = (it.get('url') if isinstance(it, dict) else None) or f"item://{section}/{idx}"
                headline = (it.get('headline') if isinstance(it, dict) else None) or (it.get('title') if isinstance(it, dict) else "Untitled")
                text = (it.get('content') if isinstance(it, dict) else None) or (it.get('abstract') if isinstance(it, dict) else "")
                source = (it.get('source') if isinstance(it, dict) else None) or (it.get('source_feed') if isinstance(it, dict) else "")
                published_raw = (it.get('published') if isinstance(it, dict) else None) or (it.get('published_date') if isinstance(it, dict) else None)
                try:
                    published_dt = datetime.fromisoformat(published_raw) if isinstance(published_raw, str) else (
                        published_raw if isinstance(published_raw, datetime) else datetime.now()
                    )
                except Exception:
                    published_dt = datetime.now()
                as_items.append(
                    DedupContentItem(
                        id=f"{section}:{idx}",
                        url=url,
                        headline=headline,
                        content=text,
                        source=source,
                        section=section,
                        published_date=published_dt,
                        embedding=None,
                        metadata={},
                    )
                )
            transformed[section] = as_items
        
        # Debug: Log what's going into deduplication
        self.logger.info(f"Sections going into dedup: {list(transformed.keys())}")
        for section, items in transformed.items():
            self.logger.info(f"Section {section}: {len(items)} items before dedup")
            
        # Only deduplicate non-Scripture sections
        result = await dedup.deduplicate_sections(transformed) if transformed else {}
        
        # Scripture bypasses deduplication completely - add it back as-is
        if scripture_items:
            result['scripture'] = scripture_items
            self.logger.info(f"Scripture section: {len(scripture_items)} items preserved (independent pipeline)")
        
        # Debug: Log what came out of deduplication
        self.logger.info(f"Sections after dedup: {list(result.keys())}")
        for section, items in result.items():
            self.logger.info(f"Section {section}: {len(items)} items after dedup")
            
        if self.metrics is not None:
            try:
                self.metrics.items_after_dedup = sum(len(v) for v in result.values())
            except Exception:
                self.metrics.items_after_dedup = 0
        return result

    async def _rank_and_select(self, content: Dict[Any, Any]) -> Dict[Any, Any]:
        aggregator: ContentAggregator = self.services['aggregator']
        
        # Separate Scripture - it doesn't need ranking/selection
        scripture_items = content.pop('scripture', None)
        
        # Convert dedup ContentItem -> simple dicts expected by aggregator
        normalized: Dict[str, List[Dict[str, Any]]] = {}
        for section, items in content.items():
            out: List[Dict[str, Any]] = []
            for it in items:
                out.append({
                    'url': getattr(it, 'url', None),
                    'headline': getattr(it, 'headline', None),
                    'title': getattr(it, 'headline', None),
                    'content': getattr(it, 'content', ''),
                    'source': getattr(it, 'source', ''),
                    'published': getattr(it, 'published_date', datetime.now()).isoformat(),
                })
            normalized[section] = out
            # Debug logging for research papers
            if section == 'research_papers':
                self.logger.info(f"Normalized {len(out)} research papers for ranking")
                if out:
                    self.logger.debug(f"  First paper: {out[0].get('headline', 'NO HEADLINE')[:50]}...")
        
        # Rank and select non-Scripture content
        ranked = await aggregator.rank_all_content(normalized)
        selected = await aggregator.select_top_items(ranked)
        
        # Add Scripture back - it bypasses ranking/selection completely
        if scripture_items:
            # Scripture items are already the right structure from fetch
            selected['scripture'] = scripture_items
            self.logger.info(f"Scripture: {len(scripture_items)} items added (bypassed ranking)")
        
        return selected

    async def _generate_summaries(self, content: Dict[Any, Any]) -> Dict[Any, Any]:
        summarizer: SummarizationService = self.services['summarization']
        
        # Ensure we have content to summarize
        if not content:
            self.logger.error("❌ No content to summarize - content dict is empty")
            return {}
        
        # Scripture needs special handling - convert raw dicts to RankedItem-like objects
        if 'scripture' in content:
            scripture_items = content['scripture']
            if scripture_items and isinstance(scripture_items[0], dict):
                # Convert Scripture dicts to RankedItem-compatible format
                from src.pipeline.content_aggregator import RankedItem, Section
                from datetime import datetime
                converted_items = []
                for idx, item in enumerate(scripture_items):
                    try:
                        # Parse published date
                        published_date = None
                        if item.get("published"):
                            try:
                                published_date = datetime.fromisoformat(item["published"].replace("Z", "+00:00"))
                            except:
                                published_date = datetime.now()
                        else:
                            published_date = datetime.now()
                        
                        # Create a RankedItem for Scripture
                        ranked_item = RankedItem(
                            id=f"scripture_{idx}",
                            headline=item.get("headline", "Daily Reading"),
                            url=item.get("url", ""),
                            source=item.get("source", "USCCB Daily Readings"),
                            content=item.get("content", ""),
                            section=Section.SCRIPTURE,
                            published_date=published_date,
                            # Perfect scores for Scripture
                            temporal_impact=10.0,
                            intellectual_novelty=10.0,
                            renaissance_breadth=10.0,
                            actionable_wisdom=10.0,
                            source_authority=10.0,
                            signal_clarity=10.0,
                            transformative_potential=10.0,
                            total_score=10.0,
                            preserve_original=item.get("preserve_original", True),
                            editorial_note="Daily spiritual guidance"
                        )
                        converted_items.append(ranked_item)
                    except Exception as e:
                        self.logger.error(f"Failed to convert Scripture item {idx}: {e}")
                        continue
                
                # Replace with converted items
                content['scripture'] = converted_items
                self.logger.info(f"Converted {len(converted_items)} Scripture items for summarization")
        
        return await summarizer.summarize_all_sections(content)

    async def _synthesize_insights(self, summaries: Dict[Any, Any]) -> Tuple[Any, Any]:
        synthesis: SynthesisService = self.services['synthesis']
        return await synthesis.synthesize_complete(summaries)

    async def _compile_email(self, summaries: Dict[Any, Any], golden_thread: Any, surprise: Any) -> Any:
        compiler: EmailCompiler = self.services['compiler']
        
        # Ensure we have summaries to compile
        if not summaries:
            self.logger.error("❌ No summaries to compile - summaries dict is empty")
            # Create minimal summaries to prevent crash
            from src.services.summarization_service import SectionSummaries
            from src.pipeline.content_aggregator import Section
            summaries = {
                Section.BREAKING_NEWS: SectionSummaries(
                    section=Section.BREAKING_NEWS,
                    title="Breaking News",
                    summaries=[],
                    intro_text="No breaking news available at this time."
                )
            }
        
        compiled = await compiler.compile_newsletter(summaries, golden_thread, surprise)
        if not getattr(compiled, 'is_valid', False):
            self.logger.error(f"Email compilation failed: {getattr(compiled, 'validation_errors', [])}")
            # Log what we tried to compile for debugging
            self.logger.debug(f"Attempted to compile with {len(summaries)} sections")
            for section, summary in summaries.items():
                if hasattr(summary, 'summaries'):
                    self.logger.debug(f"  {section}: {len(summary.summaries)} items")
            raise PipelineError(f"Email compilation failed: {getattr(compiled, 'validation_errors', [])}")
        return compiled

    async def _send_newsletter(self, compiled_email: Any) -> bool:
        if self.config.dry_run:
            self.logger.info("Dry run enabled - skipping send.")
            return True
        email: EmailService = self.services['email']
        subject = getattr(compiled_email, 'subject', "Ignacio's Fourier Forecast")
        html = getattr(compiled_email, 'html_content', '')
        # Ensure we call the mocked method directly on the service instance in tests
        return await self.services['email'].send_newsletter(html, subject)

    async def _update_cache(self, content: Dict[Any, Any]) -> None:
        cache: CacheService = self.services['cache']
        # Only cache SELECTED items with adaptive expiry based on score
        # High-scoring items get shorter cache periods (can reappear sooner)
        for section, items in (content or {}).items():
            for it in items or []:
                try:
                    # Get the item's score for adaptive caching
                    score = getattr(it, 'total_score', 5.0)  # Default to middle score if not available
                    
                    cache_item = CacheContentItem(
                        id=getattr(it, 'id', f'{section}:unknown'),
                        source=getattr(it, 'source', ''),
                        section=getattr(it, 'section', section),
                        headline=getattr(it, 'headline', ''),
                        content=getattr(it, 'content', ''),
                        url=getattr(it, 'url', ''),
                        published_date=getattr(it, 'published_date', datetime.now()),
                        metadata={},
                        embedding=None,
                        is_follow_up=False,
                        editorial_note=getattr(it, 'editorial_note', None),
                        importance_score=score,
                    )
                    # Use adaptive caching for selected items
                    await cache.add_selected_item(cache_item, score)
                    
                    # Log for research papers
                    if section == 'research_papers':
                        self.logger.info(f"Cached selected research paper with score {score:.1f}")
                except Exception as e:
                    # Best-effort; don't fail pipeline on cache write issues
                    self.logger.warning(f"Failed to cache item: {e}")
                    continue

    def _validate_timing(self) -> bool:
        now_et = datetime.now(self.et_timezone)
        scheduled = now_et.replace(
            hour=self.config.execution_time.hour,
            minute=self.config.execution_time.minute,
            second=0,
            microsecond=0,
        )
        window_end = scheduled + timedelta(minutes=self.config.max_execution_minutes)
        return scheduled <= now_et <= window_end

    async def _handle_failure(self, stage: str, error: Exception) -> None:
        self.logger.error("Failure in stage %s: %s", stage, error, exc_info=True)
        email = self.services.get('email')
        if email and hasattr(email, 'send_error_alert'):
            try:
                await email.send_error_alert(f"Stage '{stage}' failed: {error}")
            except Exception:
                pass

    def _handle_shutdown(self, signum, frame) -> None:  # noqa: D401, ANN001, ANN201
        self.shutdown_event.set()

    async def schedule_daily_execution(self) -> None:
        while not self.shutdown_event.is_set():
            await self._wait_until_execution_time()
            if self.shutdown_event.is_set():
                break
            # Run pipeline as background task so we can cancel on shutdown
            pipeline_task = asyncio.create_task(self.run_daily_pipeline())
            shutdown_wait = asyncio.create_task(self.shutdown_event.wait())
            try:
                done, pending = await asyncio.wait({pipeline_task, shutdown_wait}, return_when=asyncio.FIRST_COMPLETED)
                if shutdown_wait in done and not pipeline_task.done():
                    pipeline_task.cancel()
                    await asyncio.gather(pipeline_task, return_exceptions=True)
                    break
                # Pipeline finished
                if pipeline_task in done:
                    try:
                        _ = pipeline_task.result()
                    except Exception as e:  # noqa: BLE001
                        await self._handle_failure("run", e)
            finally:
                for t in (pipeline_task, shutdown_wait):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(pipeline_task, shutdown_wait, return_exceptions=True)
            # Avoid tight loop before scheduling next run
            await asyncio.sleep(60)

    async def _wait_until_execution_time(self) -> None:
        next_run = self._calculate_next_run_time()
        while not self.shutdown_event.is_set():
            now_et = datetime.now(self.et_timezone)
            delay = (next_run - now_et).total_seconds()
            if delay <= 0:
                break
            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=min(delay, 1.0))
            except asyncio.TimeoutError:
                continue

    def _calculate_next_run_time(self) -> datetime:
        now_et = datetime.now(self.et_timezone)
        exec_time = self.config.execution_time
        # Build today's candidate at 5:00 AM ET using zoneinfo to respect DST
        try:
            next_run = now_et.replace(hour=exec_time.hour, minute=exec_time.minute, second=0, microsecond=0)
        except Exception:
            # Fallback: add a day then set time fields
            tmp = now_et + timedelta(days=1)
            next_run = tmp.replace(hour=exec_time.hour, minute=exec_time.minute, second=0, microsecond=0)
        if next_run <= now_et:
            next_run = (now_et + timedelta(days=1)).replace(
                hour=exec_time.hour, minute=exec_time.minute, second=0, microsecond=0
            )
        return next_run

    async def test_pipeline(self) -> bool:
        return await self.run_daily_pipeline()

    def get_metrics(self) -> Optional[PipelineMetrics]:
        return self.metrics

    async def health_check(self) -> Dict[str, bool]:
        if not self.services:
            await self.initialize_services()
        results: Dict[str, bool] = {}
        for name in ['ai', 'llmlayer', 'email']:
            svc = self.services.get(name)
            ok = False
            try:
                if hasattr(svc, 'test_connection'):
                    ok = bool(await svc.test_connection())
            except Exception:
                ok = False
            results[name] = ok
        return results


class PipelineError(Exception):
    """Custom exception for pipeline failures"""
    pass


async def handle_cache_management(args, cache_db_path: str):
    """Handle cache management commands"""
    import json
    from datetime import datetime
    
    cache_service = CacheService(cache_db_path)
    await cache_service.initialize_db()
    
    if args.cache_stats:
        print("📊 Cache Statistics")
        print("=" * 50)
        stats = await cache_service.get_cache_statistics()
        
        print(f"Total items in cache: {stats['total_items']}")
        
        if stats['total_items'] > 0:
            print(f"\nDate range:")
            if stats.get('date_range'):
                print(f"  Oldest: {stats['date_range']['oldest']}")
                print(f"  Newest: {stats['date_range']['newest']}")
            
            print(f"\nItems by section:")
            for section, count in stats['items_by_section'].items():
                print(f"  {section}: {count}")
            
            print(f"\nItems by source:")
            for source, count in stats['items_by_source'].items():
                print(f"  {source}: {count}")
        
        print(f"\nStorage info:")
        print(f"  Newsletter manifests: {stats['storage_info']['newsletters_stored']}")
        print(f"  Deduplication records: {stats['storage_info']['dedup_records']}")
        
        if stats.get('deduplication_stats'):
            print(f"\nDeduplication effectiveness (last 7 days):")
            dedup = stats['deduplication_stats']
            print(f"  Avg items fetched: {dedup['avg_items_fetched']}")
            print(f"  Avg unique items: {dedup['avg_unique_items']}")
            print(f"  Avg uniqueness: {dedup['avg_uniqueness_percentage']}%")
    
    if args.clear_test_data:
        print(f"🧹 Clearing test data from last {args.cache_hours} hours...")
        stats = await cache_service.clear_test_data(hours=args.cache_hours)
        print(f"✅ Cleared {stats['seen_items_removed']} seen items")
        print(f"✅ Cleared {stats['newsletters_removed']} newsletter manifests")
        print(f"✅ Cleared {stats['metrics_removed']} dedup metrics")
        print(f"   (cutoff: {stats['cutoff_time']})")
    
    if args.clear_cache:
        print("🗑️  WARNING: This will clear ALL cache data!")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() == 'yes':
            stats = await cache_service.clear_all_cache()
            print(f"✅ Cleared {stats['seen_items']} seen items")
            print(f"✅ Cleared {stats['newsletter_manifest']} newsletter manifests")
            print(f"✅ Cleared {stats['dedup_metrics']} dedup metrics")
            print("Cache completely cleared!")
        else:
            print("❌ Cache clear cancelled")


async def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Ignacio's Fourier Forecast Pipeline")
    parser.add_argument('--test', action='store_true', help='Run test pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without sending')
    parser.add_argument('--once', action='store_true', help='Run once immediately')
    parser.add_argument('--health', action='store_true', help='Health check only')
    
    # Cache management commands
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cache data')
    parser.add_argument('--clear-test-data', action='store_true', help='Clear recent test data (last 48 hours)')
    parser.add_argument('--cache-stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--cache-hours', type=int, default=48, help='Hours of test data to clear (default: 48)')
    args = parser.parse_args()

    config = PipelineConfig(
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', ''),
        voyage_api_key=os.getenv('VOYAGE_API_KEY', ''),
        llmlayer_api_key=os.getenv('LLMLAYER_API_KEY', ''),
        smtp_host=os.getenv('SMTP_HOST', 'smtp.gmail.com'),
        smtp_port=int(os.getenv('SMTP_PORT', '587')),
        smtp_user=os.getenv('SMTP_USER', ''),
        smtp_password=os.getenv('SMTP_PASSWORD', ''),
        recipient_email=os.getenv('RECIPIENT_EMAIL', 'ignacio@example.com'),
        execution_time=time(5, 0),
        delivery_time=time(5, 30),
        dry_run=args.dry_run,
        test_mode=args.test,
        max_execution_minutes=int(os.getenv('MAX_EXECUTION_MINUTES', '120')),
    )

    pipeline = MainPipeline(config)

    try:
        # Handle cache management commands first
        if args.clear_cache or args.clear_test_data or args.cache_stats:
            cache_db_path = "cache.db"  # Cache is always in project root
            await handle_cache_management(args, cache_db_path)
            return
            
        if args.health:
            health = await pipeline.health_check()
            print("Service Health Status:")
            for service, status in health.items():
                print(f"  {service}: {'✅' if status else '❌'}")
            return
        elif args.test:
            print("Running test pipeline...")
            success = await pipeline.test_pipeline()
            print("✅ Test pipeline completed successfully!" if success else "❌ Test pipeline failed!")
            if not success:
                sys.exit(1)
        elif args.once:
            print("Running pipeline once...")
            success = await pipeline.run_daily_pipeline()
            if success:
                print("✅ Pipeline completed successfully!")
                metrics = pipeline.get_metrics()
                if metrics:
                    print(f"Total time: {metrics.total_time():.2f}s")
                    print(f"Items processed: {metrics.items_selected}")
            else:
                print("❌ Pipeline failed!")
                sys.exit(1)
        else:
            print("Starting daily scheduler...")
            print("Will run daily at 5:00 AM ET")
            await pipeline.schedule_daily_execution()
    except KeyboardInterrupt:
        print("\n⚠️ Shutting down gracefully...")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Fatal error: {e}")
        logging.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


