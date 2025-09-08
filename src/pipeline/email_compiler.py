import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import markdown2
import premailer
import pytz
from jinja2 import Environment, FileSystemLoader, TemplateError

from src.pipeline.content_aggregator import Section
from src.services.summarization_service import SectionSummaries
from src.services.synthesis_service import DelightfulSurprise, GoldenThread
from src.utils.logo_embedder import get_embedded_logo_url


@dataclass
class CompiledEmail:
    """Complete compiled email ready for sending"""
    subject: str
    html_content: str
    plain_text: str
    preview_text: str

    # Metadata
    compile_time: datetime
    word_count: int
    estimated_read_time: int  # minutes
    section_count: int
    total_items: int

    # Validation
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class EmailTemplate:
    """Email template configuration"""
    name: str
    path: str
    variables: Dict[str, Any]
    mobile_optimized: bool = True
    dark_mode_support: bool = True


class CompilationError(Exception):
    """Custom exception for compilation failures"""
    pass


class EmailCompiler:
    """
    Compiles newsletter content into beautiful HTML emails.

    This service is the final touchpoint before delivery, transforming
    all the curated content into a visual experience that:
    - Looks beautiful on every device
    - Loads quickly even on slow connections
    - Guides the eye with elegant typography
    - Creates joy through thoughtful design
    """

    def __init__(self, template_dir: str = "templates") -> None:
        """Initialize email compiler."""
        self.template_dir = template_dir

        # Jinja2 environment
        try:
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True,
            )
        except Exception as e:  # noqa: BLE001
            raise CompilationError(f"Failed to initialize Jinja2 environment: {e}") from e

        # Add custom filters
        self.env.filters["markdown"] = self._markdown_filter
        self.env.filters["smartquotes"] = self._smartquotes_filter
        self.env.filters["timeformat"] = self._timeformat_filter

        # Configuration
        self.max_email_size_kb = 102  # Gmail clipping limit
        self.target_read_time = 7  # minutes
        self.words_per_minute = 250

        # Section display order
        self.section_order: List[str] = [
            Section.SCRIPTURE,
            Section.BREAKING_NEWS,
            Section.BUSINESS,
            Section.TECH_SCIENCE,
            Section.RESEARCH_PAPERS,
            Section.STARTUP,
            Section.POLITICS,
            Section.LOCAL,
            Section.MISCELLANEOUS,
            Section.EXTRA,
        ]

        # Eastern timezone for dates
        self.et_timezone = pytz.timezone("America/New_York")

        self.logger = logging.getLogger(__name__)

    async def compile_newsletter(
        self,
        sections: Dict[Section, SectionSummaries],
        golden_thread: Optional[GoldenThread] = None,
        surprise: Optional[DelightfulSurprise] = None,
        date: Optional[datetime] = None,
    ) -> CompiledEmail:
        """
        Compile complete newsletter from components.
        """
        compile_time = datetime.now(self.et_timezone)
        newsletter_date = date or compile_time

        try:
            # Calculate metrics FIRST (before template rendering)
            # Use preliminary metrics based on sections content
            word_count, read_time, total_items = self._calculate_metrics(sections, "", "")
            
            # Prepare context with accurate metrics
            template_data = await self._prepare_template_data(
                sections=sections,
                golden_thread=golden_thread,
                surprise=surprise,
                date=newsletter_date,
            )
            
            # Override template data with calculated metrics
            template_data["estimated_read_time"] = read_time
            template_data["total_items"] = total_items

            # Render HTML and plain text with correct metrics
            html_raw = self._render_html_template(template_data)
            # Inline CSS for client compatibility (template already responsive)
            html_inlined = self._inline_css(html_raw)
            plain_text = self._render_plain_text(template_data)

            compiled = CompiledEmail(
                subject=str(template_data.get("subject", "Ignacio's Fourier Forecast")),
                html_content=html_inlined,
                plain_text=plain_text,
                preview_text=str(template_data.get("preview_text", "")),
                compile_time=compile_time,
                word_count=word_count,
                estimated_read_time=read_time,
                section_count=len(self._order_sections(sections)),
                total_items=total_items,
            )

            is_valid, errors = self._validate_email(compiled)
            compiled.is_valid = is_valid
            compiled.validation_errors = errors
            if not is_valid:
                self.logger.warning("Email validation errors: %s", errors)

            return compiled
        except Exception as e:  # noqa: BLE001
            self.logger.critical("Newsletter compilation failed: %s", e, exc_info=True)
            raise CompilationError(f"Critical failure in email compilation: {e}") from e

    async def _prepare_template_data(
        self,
        sections: Dict[Section, SectionSummaries],
        golden_thread: Optional[GoldenThread],
        surprise: Optional[DelightfulSurprise],
        date: datetime,
    ) -> Dict[str, Any]:
        """Prepare data for template rendering."""
        ordered: List[Tuple[Section, SectionSummaries]] = self._order_sections(sections)

        # Build sections mapping compatible with existing templates/newsletter.html.j2
        # { section_name: { intro_text: str, items: [ {headline, summary, source, time}, ... ] } }
        sections_map: Dict[str, Dict[str, Any]] = {}
        for section_key, sec in ordered:
            items: List[Dict[str, Any]] = []
            for s in sec.summaries:
                items.append(
                    {
                        "headline": s.headline,
                        "summary": s.summary_text,
                        "source": s.source,
                        "time": s.time_ago,
                        "source_url": s.source_url,
                    }
                )
            sections_map[section_key] = {
                "intro_text": sec.intro_text,
                "items": items
            }

        subject = self._generate_subject_line(date, golden_thread=golden_thread)
        preview_text = self._generate_preview_text(sections, golden_thread)
        greeting = await self._generate_greeting(date, sections, golden_thread)

        # Render-friendly date: keep raw datetime for filters, otherwise string for existing template
        safe_date = date.strftime("%B %d, %Y")
        
        # Get embedded logo URL for email
        logo_url = get_embedded_logo_url()

        data: Dict[str, Any] = {
            "subject": subject,
            "preview_text": preview_text,
            "greeting": greeting,
            "date": safe_date,
            "logo_url": logo_url,  # Add logo URL for template
            # Existing template expects string for golden_thread
            "golden_thread": golden_thread.to_newsletter_text() if golden_thread else None,
            # Existing template expects `delightful_surprise`
            "delightful_surprise": surprise,
            # Sections for HTML and plain text templates in repo
            "sections": sections_map,
            # Metrics placeholders; will be backfilled after calculation
            "estimated_read_time": self.target_read_time,
            "total_items": sum(len(v["items"]) for v in sections_map.values()),
        }

        return data

    def _generate_subject_line(
        self,
        date: datetime,
        golden_thread: Optional[GoldenThread] = None,
        highlight: Optional[str] = None,
    ) -> str:
        """Generate compelling 3-5 word subject line."""
        # If a highlight is provided, use it (but truncate to 3-5 words)
        if highlight:
            words = highlight.split()[:5]
            return " ".join(words)
        
        # If golden thread exists, create subject from it
        if golden_thread:
            # Support 'insight' (newest), 'connection' (recent), or 'theme' (legacy)
            hint = getattr(golden_thread, 'insight', None) or getattr(golden_thread, 'connection', None) or getattr(golden_thread, 'theme', None)
            if hint:
                # Take first 3-5 words of the golden thread
                words = hint.split()[:5]
                if len(words) >= 3:
                    return " ".join(words)
        
        # Default subjects based on common themes (3-5 words)
        default_subjects = [
            "Markets Rally Today",
            "Tech Breakthrough News",
            "Fed Decision Looms",
            "AI News Today",
            "Breaking Global Updates"
        ]
        
        # Use date to pick a default (rotate through them)
        import hashlib
        date_hash = int(hashlib.md5(str(date).encode()).hexdigest()[:8], 16)
        return default_subjects[date_hash % len(default_subjects)]

    def _generate_preview_text(
        self,
        sections: Dict[Section, SectionSummaries],
        golden_thread: Optional[GoldenThread],
    ) -> str:
        """Generate email preview text (shows in inbox). 150 chars max."""
        if golden_thread:
            # Support 'insight' (newest), 'connection' (recent), or 'theme' (legacy)
            text = getattr(golden_thread, 'insight', None) or getattr(golden_thread, 'connection', None) or getattr(golden_thread, 'theme', None)
            if text:
                return text[:150]

        ordered = self._order_sections(sections)
        for _, sec in ordered:
            if sec.summaries:
                # Prefer headline or summary text
                text = sec.summaries[0].headline or sec.summaries[0].summary_text
                if text:
                    return str(text)[:150]
        return "Your daily distillation of signal from noise."[:150]

    async def _generate_greeting(self, date: datetime, sections: Dict[Section, SectionSummaries], golden_thread: Optional[GoldenThread] = None) -> str:
        """Generate AI-powered contextual greeting."""
        # Import AIService here to avoid circular imports
        from src.services.ai_service import AIService
        
        try:
            # Initialize AI service
            ai_service = AIService()
            
            # Build comprehensive newsletter content structure like synthesis service does
            newsletter_content = self._build_newsletter_content_for_greeting(sections)
            
            # Get golden thread text if available
            # Support 'insight' (newest), 'connection' (recent), or 'theme' (legacy)
            golden_thread_text = None
            if golden_thread:
                golden_thread_text = getattr(golden_thread, 'insight', None) or getattr(golden_thread, 'connection', None) or getattr(golden_thread, 'theme', None)
            
            # Generate contextually relevant greeting
            greeting = await ai_service.generate_morning_greeting(
                date=date,
                newsletter_content=newsletter_content,
                golden_thread=golden_thread_text
            )
            
            return greeting
            
        except Exception as e:
            # Fallback to simple greeting if AI service fails
            self.logger.warning(f"AI greeting generation failed: {e}, using fallback")
            date_str = date.strftime("%A, %B %d, %Y")
            return f"Good morning, Ignacio! Ready for today's signal from the noise? — {date_str}"

    def _build_newsletter_content_for_greeting(self, sections: Dict[Section, SectionSummaries]) -> Dict[str, Any]:
        """Build newsletter content structure for AI greeting generation."""
        newsletter_content = {}
        for section_key, sec in sections.items():
            newsletter_content[section_key] = {
                "title": sec.title,
                "intro": sec.intro_text,
                "items": [
                    {
                        "headline": s.headline,
                        "summary": s.summary_text,
                        "source": s.source,
                        "url": s.source_url,
                        "time_ago": s.time_ago,
                    }
                    for s in sec.summaries
                ],
            }
        return newsletter_content

    def _order_sections(
        self, sections: Dict[Section, SectionSummaries]
    ) -> List[Tuple[Section, SectionSummaries]]:
        """Order sections for optimal reading flow, excluding empty ones."""
        ordered: List[Tuple[Section, SectionSummaries]] = []
        for s in self.section_order:
            if s in sections and sections[s].summaries:
                ordered.append((s, sections[s]))
        # Include any sections not in predefined order at the end (only if non-empty)
        for s, data in sections.items():
            if s not in self.section_order and data.summaries:
                ordered.append((s, data))
        return ordered

    def _render_html_template(self, template_data: Dict[str, Any]) -> str:
        """Render HTML email template."""
        try:
            template = self.env.get_template("newsletter.html.j2")
            return template.render(template_data)
        except TemplateError as e:  # noqa: BLE001
            self.logger.error("HTML template rendering failed: %s", e, exc_info=True)
            raise CompilationError(f"HTML template error: {e}") from e

    def _render_plain_text(self, template_data: Dict[str, Any]) -> str:
        """Render plain text version."""
        try:
            template = self.env.get_template("newsletter_plain.j2")
            return template.render(template_data)
        except TemplateError as e:  # noqa: BLE001
            self.logger.error("Plain text template rendering failed: %s", e, exc_info=True)
            raise CompilationError(f"Plain text template error: {e}") from e

    def _inline_css(self, html: str) -> str:
        """Inline CSS for email client compatibility."""
        try:
            # Configure premailer to preserve important declarations and style tags
            return premailer.transform(
                html,
                keep_style_tags=True,  # Keep <style> tags for media queries
                strip_important=False,  # Don't strip !important declarations
                cssutils_logging_level=logging.ERROR  # Reduce CSS warnings
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error("CSS inlining failed: %s", e, exc_info=True)
            raise CompilationError(f"CSS inlining error: {e}") from e

    def _calculate_metrics(
        self,
        sections: Dict[Section, SectionSummaries],
        html: str,
        plain_text: str,
    ) -> Tuple[int, int, int]:
        """Calculate email metrics: (word_count, read_time_minutes, total_items)."""
        # Count words from canonical content (avoid double-counting rendered outputs)
        word_count = 0
        total_items = 0
        
        # Count words from section summaries and prefaces
        for sec in sections.values():
            total_items += len(sec.summaries)
            for s in sec.summaries:
                word_count += len((s.summary_text or "").split())
            # Count section intro/preface text if present
            if sec.intro_text:
                word_count += len(sec.intro_text.split())
        
        # Note: Greeting, golden thread, and delightful surprise are counted
        # separately in the template rendering phase, not here
        
        # Calculate reading time (average reader: 200-250 words per minute)
        read_time = max(1, round(word_count / max(1, self.words_per_minute)))
        
        # Enforce 30-minute maximum reading time per VISION.txt
        if read_time > 30:
            self.logger.warning("Email exceeds 30-minute reading time: %d minutes", read_time)
        
        return word_count, read_time, total_items

    def _validate_email(self, compiled: CompiledEmail) -> Tuple[bool, List[str]]:
        """Validate compiled email."""
        errors: List[str] = []
        if not compiled.subject:
            errors.append("Missing subject line.")
        if not compiled.html_content:
            errors.append("Empty HTML content.")
        if not compiled.plain_text:
            errors.append("Empty plain text content.")
        if compiled.section_count <= 0 or compiled.total_items <= 0:
            errors.append("No content sections or items.")
        if not self._check_email_size(compiled.html_content):
            size_kb = len(compiled.html_content.encode("utf-8")) / 1024
            errors.append(
                f"Email size ({size_kb:.1f}KB) exceeds limit of {self.max_email_size_kb}KB."
            )
        return (len(errors) == 0), errors

    def _check_email_size(self, html: str) -> bool:
        """Check if email size is within limits."""
        size_kb = len((html or "").encode("utf-8")) / 1024
        return size_kb <= self.max_email_size_kb

    # Jinja2 filters
    def _markdown_filter(self, text: str) -> str:
        """Jinja2 filter for markdown conversion."""
        if text is None:
            return ""
        return markdown2.markdown(text, extras=["fenced-code-blocks", "smarty"])

    def _smartquotes_filter(self, text: str) -> str:
        """Jinja2 filter for smart quotes (returns plain text)."""
        if text is None:
            return ""
        html_text = markdown2.markdown(text, extras=["smarty"]).strip()
        # Strip paragraph tags if present
        if html_text.startswith("<p>") and html_text.endswith("</p>"):
            html_text = html_text[3:-4]
        return html_text

    def _timeformat_filter(self, dt: datetime, format: str = "%B %d, %Y") -> str:  # noqa: A002 - keep name per spec
        """Jinja2 filter for date formatting."""
        if not isinstance(dt, datetime):
            return str(dt)
        return dt.strftime(format)

    async def preview_email(self, compiled: CompiledEmail) -> str:
        """Generate preview HTML. For now, return the compiled HTML."""
        return compiled.html_content

    def get_template_list(self) -> List[EmailTemplate]:
        """Get available email templates in the template directory."""
        templates: List[EmailTemplate] = []
        try:
            for name in os.listdir(self.template_dir):
                if not name.endswith(".j2"):
                    continue
                path = os.path.join(self.template_dir, name)
                templates.append(
                    EmailTemplate(
                        name=name,
                        path=path,
                        variables={},
                        mobile_optimized=True,
                        dark_mode_support=True,
                    )
                )
        except FileNotFoundError:
            return []
        return templates


