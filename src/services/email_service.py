import os
import smtplib
import asyncio
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Environment, FileSystemLoader
import pytz
from src.services.ai_service import AIService, AIServiceError


@dataclass
class NewsletterContent:
    """Complete newsletter content ready for compilation"""
    subject: str
    greeting: str
    golden_thread: Optional[str]
    sections: Dict[str, List[Dict[str, str]]]
    delightful_surprise: Optional[Dict[str, str]]
    date: datetime
    total_items: int
    estimated_read_time: int


@dataclass
class EmailConfig:
    """Email service configuration"""
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    from_email: str
    to_email: str
    reply_to: Optional[str] = None
    use_tls: bool = True


class EmailService:
    """
    Email compilation and delivery service for the Renaissance newsletter.
    """

    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config or self._load_config_from_env()

        # Logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # Basic configuration if not already configured by app
            logging.basicConfig(level=logging.INFO)

        if not self.config.smtp_password:
            raise ValueError("SMTP password required. Set SMTP_PASSWORD environment variable.")

        self.template_env = Environment(
            loader=FileSystemLoader('templates'),
            autoescape=True
        )

        self.et_timezone = pytz.timezone('America/New_York')
        self.test_mode = os.getenv("EMAIL_TEST_MODE", "false").lower() == "true"

        # Optional AI-driven subject generation
        self.use_ai_subjects = os.getenv("EMAIL_SUBJECT_VIA_AI", "false").lower() == "true"
        self.ai: Optional[AIService] = None
        if self.use_ai_subjects:
            try:
                self.ai = AIService()
                self.logger.info("AIService initialized for email subject generation")
            except Exception as e:
                # Disable AI subject generation if AIService cannot initialize
                self.ai = None
                self.use_ai_subjects = False
                self.logger.error(f"Disabling AI subject generation: {e}")

    def _load_config_from_env(self) -> EmailConfig:
        """Load email configuration from environment variables"""
        return EmailConfig(
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            from_email=os.getenv("FROM_EMAIL", ""),
            to_email=os.getenv("TO_EMAIL", "ignacio@example.com"),
            reply_to=os.getenv("REPLY_TO_EMAIL"),
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
        )

    async def test_connection(self) -> bool:
        """
        Test SMTP connection and authentication.
        """
        self.logger.info("Testing SMTP connection to %s:%s", self.config.smtp_host, self.config.smtp_port)
        try:
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            try:
                if self.config.use_tls:
                    server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                self.logger.info("SMTP connection and authentication successful")
                return True
            finally:
                try:
                    server.quit()
                except Exception:
                    pass
        except Exception as exc:
            self.logger.error("SMTP connection test failed: %s", exc)
            raise EmailServiceError(f"SMTP connection test failed: {exc}") from exc

    async def compile_newsletter(self, content: NewsletterContent) -> str:
        """Compile newsletter content into beautiful HTML."""
        self.logger.info("Compiling newsletter HTML for %s", content.date)
        template = self.template_env.get_template('newsletter.html.j2')
        html = template.render(
            subject=content.subject,
            greeting=content.greeting,
            golden_thread=content.golden_thread,
            sections=content.sections,
            delightful_surprise=content.delightful_surprise,
            date=content.date.strftime('%B %d, %Y') if isinstance(content.date, datetime) else content.date,
            estimated_read_time=content.estimated_read_time,
        )
        optimized = self._apply_mobile_optimization(html)
        self.logger.info("Newsletter HTML compiled (%d chars)", len(optimized))
        return optimized

    async def send_newsletter(
        self,
        html_content: str,
        subject: str,
        test_recipient: Optional[str] = None,
    ) -> bool:
        """Send the compiled newsletter."""
        self.logger.info("Preparing to send newsletter (subject: %s)", subject)
        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = self.config.from_email or self.config.smtp_user
        recipient = test_recipient or self.config.to_email
        message['To'] = recipient
        if self.config.reply_to:
            message['Reply-To'] = self.config.reply_to

        plain_text = self.create_plain_text_version(html_content)
        message.attach(MIMEText(plain_text, 'plain'))
        message.attach(MIMEText(html_content, 'html'))

        result = await self._send_via_smtp(message)
        self.logger.info("Newsletter sent to %s", recipient)
        return result

    async def schedule_delivery(
        self,
        content: NewsletterContent,
        delivery_time: Optional[datetime] = None,
    ) -> bool:
        """Schedule newsletter delivery for specific time. Default: 5:30 AM ET."""
        now_et = datetime.now(self.et_timezone)

        if delivery_time is None:
            delivery_time = now_et.replace(hour=5, minute=30, second=0, microsecond=0)
            if now_et > delivery_time:
                delivery_time = delivery_time + timedelta(days=1)
        else:
            # Ensure delivery_time is timezone-aware in ET
            if delivery_time.tzinfo is None:
                delivery_time = self.et_timezone.localize(delivery_time)
            else:
                delivery_time = delivery_time.astimezone(self.et_timezone)

        wait_seconds = (delivery_time - now_et).total_seconds()
        if wait_seconds > 0:
            self.logger.info("Scheduling newsletter for %s (wait %.1fs)", delivery_time.isoformat(), wait_seconds)
            await asyncio.sleep(wait_seconds)

        html = await self.compile_newsletter(content)
        subject: str
        if content.subject:
            subject = content.subject
        else:
            subject = await self.generate_subject_line_async(content)
        self.logger.info("Dispatching scheduled newsletter at %s", datetime.now(self.et_timezone).isoformat())
        return await self.send_newsletter(html, subject)

    

    def generate_subject_line(self, date: datetime, special_note: Optional[str] = None) -> str:
        """Generate compelling subject line."""
        date_str = date.strftime('%B %d, %Y') if isinstance(date, datetime) else str(date)
        if special_note:
            return f"{special_note} | Fourier Forecast — {date_str}"
        return f"Fourier Forecast — {date_str}"

    async def generate_subject_line_async(self, content: NewsletterContent) -> str:
        """Optionally generate subject via AI, else fallback to deterministic subject."""
        try:
            if self.use_ai_subjects and self.ai is not None:
                ctx = {
                    "date": content.date.strftime('%B %d, %Y') if isinstance(content.date, datetime) else str(content.date),
                    "golden_thread": content.golden_thread or "",
                    "top_sections": list(content.sections.keys())[:3],
                    "total_items": content.total_items,
                }
                self.logger.info("Generating subject via AI with context: %s", ctx)
                resp = await self.ai.interact_with_gpt("subject_generation", ctx)
                if resp.success and isinstance(resp.content, str) and resp.content.strip():
                    subject = resp.content.strip()
                    self.logger.info("AI-generated subject: %s", subject)
                    return subject
                self.logger.error("AI subject generation failed: %s", resp.error_message)
        except Exception as e:
            self.logger.error("AI subject generation exception: %s", e)
        # Fallback deterministic subject
        return self.generate_subject_line(content.date)

    def create_plain_text_version(self, html_content: str) -> str:
        """Create plain text version from HTML."""
        # Remove script/style content
        text = re.sub(r'<(script|style)[^>]*>[\s\S]*?</\1>', '', html_content, flags=re.IGNORECASE)
        # Replace <br> and </p> with newlines
        text = re.sub(r'<\s*br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p\s*>', '\n', text, flags=re.IGNORECASE)
        # Strip remaining tags
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = text.strip()
        return text

    async def _send_via_smtp(self, message: MIMEMultipart) -> bool:
        """Send email via SMTP."""
        try:
            self.logger.info("Connecting to SMTP server %s:%s", self.config.smtp_host, self.config.smtp_port)
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            try:
                if self.config.use_tls:
                    server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(message)
                self.logger.info("SMTP send_message completed")
                return True
            finally:
                try:
                    server.quit()
                except Exception:
                    pass
        except Exception as exc:
            self.logger.error("SMTP send failed: %s", exc)
            raise EmailServiceError(f"SMTP send failed: {exc}") from exc

    def _apply_mobile_optimization(self, html: str) -> str:
        """Apply mobile-responsive optimizations if missing."""
        if '<head>' in html and 'viewport' not in html.lower():
            return html.replace(
                '<head>',
                '<head>\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">'
            )
        return html

    def _estimate_read_time(self, content: NewsletterContent) -> int:
        """Estimate reading time in minutes."""
        if content.estimated_read_time:
            return int(content.estimated_read_time)
        words = 0
        words += len((content.greeting or '').split())
        words += len((content.golden_thread or '').split())
        if content.delightful_surprise:
            words += len((content.delightful_surprise.get('content', '')).split())
        for items in content.sections.values():
            for item in items:
                words += len(item.get('summary', '').split())
        minutes = max(1, round(words / 200))
        return int(minutes)

    async def send_error_alert(self, error_message: str) -> None:
        """Send error alert if newsletter fails."""
        alert = MIMEMultipart('alternative')
        alert['Subject'] = 'Newsletter Error Alert'
        alert['From'] = self.config.from_email or self.config.smtp_user
        alert['To'] = self.config.to_email or (self.config.from_email or self.config.smtp_user)
        alert.attach(MIMEText(f"A critical error occurred while sending the newsletter:\n\n{error_message}", 'plain'))
        await self._send_via_smtp(alert)


class EmailServiceError(Exception):
    """Custom exception for email service failures"""
    pass


