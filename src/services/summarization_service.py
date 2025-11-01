import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.services.ai_service import AIService
from src.pipeline.content_aggregator import Section, RankedItem


@dataclass
class Summary:
    """A polished summary ready for the newsletter"""
    item_id: str
    headline: str
    summary_text: str
    source: str
    source_url: str
    time_ago: str
    
    
    # Editorial metadata
    editorial_voice: str = ""
    word_count: int = 0
    reading_level: Optional[float] = None


@dataclass
class SectionSummaries:
    """Complete summaries for a newsletter section"""
    section: Section
    title: str
    summaries: List[Summary]
    intro_text: Optional[str] = None
    editorial_note: Optional[str] = None


class EditorialVoice(Enum):
    """Section-specific editorial personas"""
    HISTORICAL = "historical"      # Breaking news: Voice of history in real-time
    MEDICI = "medici"              # Business: Modern Medici understanding commerce as culture
    POET_SCIENTIST = "poet"        # Science: Translator of nature's language into poetry
    APPRENTICE = "apprentice"      # Startup: Learning from masters
    CONTEMPLATIVE = "contemplative" # Scripture: Warm, spiritual reflection
    CIVIC = "civic"                # Politics: Thoughtful civic awareness
    NEIGHBOR = "neighbor"          # Local: Community connection
    WANDERER = "wanderer"          # Miscellaneous: Intellectual serendipity


class SummarizationService:
    """
    Creates brilliant summaries with section-specific editorial voices.
    """
    
    def __init__(self, ai_service: AIService):
        self.ai = ai_service
        
        # Editorial voice mapping
        self.section_voices = {
            Section.BREAKING_NEWS: EditorialVoice.HISTORICAL,
            Section.BUSINESS: EditorialVoice.MEDICI,
            Section.TECH_SCIENCE: EditorialVoice.POET_SCIENTIST,
            Section.STARTUP: EditorialVoice.APPRENTICE,
            Section.SCRIPTURE: EditorialVoice.CONTEMPLATIVE,
            Section.POLITICS: EditorialVoice.CIVIC,
            Section.LOCAL: EditorialVoice.NEIGHBOR,
            Section.MISCELLANEOUS: EditorialVoice.WANDERER,
            Section.RESEARCH_PAPERS: EditorialVoice.POET_SCIENTIST,
            Section.EXTRA: EditorialVoice.WANDERER,
        }
        
        # Configuration
        self.target_words_per_summary = 25  # One brilliant sentence
        self.max_words_per_summary = 40
        self.batch_size = 5  # Summaries per API call
        
        self.logger = logging.getLogger(__name__)

    async def summarize_all_sections(
        self,
        sections: Dict[Section, List[RankedItem]]
    ) -> Dict[Section, SectionSummaries]:
        tasks = [self.summarize_section(items, section) for section, items in sections.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results with error handling
        summaries = {}
        for idx, (section, items) in enumerate(sections.items()):
            result = results[idx]
            if isinstance(result, Exception):
                self.logger.error(f"Failed to summarize section {section}: {result}")
                # Create a minimal fallback summary
                summaries[section] = SectionSummaries(
                    section=section,
                    title=self._get_section_title(section),
                    summaries=[],
                    intro_text=f"Items for {self._get_section_title(section)}"
                )
            else:
                summaries[section] = result
        
        return summaries

    async def summarize_section(
        self,
        items: List[RankedItem],
        section: Section,
        all_items: Optional[List[RankedItem]] = None
    ) -> SectionSummaries:
        voice = self.section_voices.get(section, EditorialVoice.WANDERER)
        summaries: List[Summary] = []
        if not items:
            return SectionSummaries(section=section, title=self._get_section_title(section), summaries=[])
        
        # Batch in chunks
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            chunk_summaries = await self._batch_summarize(chunk, voice)
            summaries.extend(chunk_summaries)
        
        # No validation needed - our prompts are strong enough
        title = self._get_section_title(section)
        # Pass all_items if available, otherwise use items as fallback
        intro = await self.generate_section_intro(section, summaries, all_items or items)
        return SectionSummaries(section=section, title=title, summaries=summaries, intro_text=intro)

    async def _generate_summary(
        self,
        item: RankedItem,
        voice: EditorialVoice
    ) -> Summary:
        # CRITICAL FIX: Check preserve_original flag first (for Exa Websets enrichment_summary)
        # If preserve_original=True, skip AI summarization and preserve the original content
        if item.preserve_original:
            self.logger.info(f"Preserving original content for {item.headline[:50]}... (enrichment_summary or preserve_original flag)")
            summary_text = item.summary_text
        # Check if this is a USCCB reading that should preserve its original content
        # Scripture section + USCCB source = preserve full text
        elif item.section == Section.SCRIPTURE and item.source == "USCCB Daily Readings":
            # For USCCB readings, preserve the full original biblical text
            summary_text = item.summary_text
        else:
            # For Catholic Daily Reflections and other content, generate a summary
            prompt = self._get_editorial_prompt(voice, item.section)
            # Use AIService.generate_summary tool-calling for robust single-sentence output
            generated = await self.ai.generate_summary({
                "headline": item.headline,
                "content": item.summary_text,
                "url": item.url,
                "source": item.source,
                "section": item.section,
                "voice": voice.value,
                "constraints": {
                    "sentence": True,
                    "min_words": self.target_words_per_summary,
                    "max_words": self.max_words_per_summary,
                },
                "prompt": prompt,
            })
            summary_text: str = generated.get("summary", "") if isinstance(generated, dict) else str(generated)
        
        # Only enforce sentence and word limits for non-USCCB content
        if not (item.section == Section.SCRIPTURE and item.source == "USCCB Daily Readings"):
            # Enforce exactly one sentence for summaries
            sentences = self._split_into_sentences(summary_text)
            if len(sentences) > 1:
                # Take the best sentence (usually the first complete one)
                summary_text = sentences[0]
            elif len(sentences) == 0:
                # Fallback if no proper sentence detected
                summary_text = summary_text.rstrip('.') + '.'
            
            # Enforce word count cap for summaries
            if self._count_words(summary_text) > self.max_words_per_summary:
                words = summary_text.split()
                summary_text = " ".join(words[: self.max_words_per_summary - 1]) + '.'  # Ensure it ends with period
        source_name, clean_url = self._format_source_attribution(item.source, item.url)
        s = Summary(
            item_id=item.id,
            headline=item.headline,
            summary_text=summary_text,
            source=source_name,
            source_url=clean_url,
            time_ago=self._calculate_time_ago(item.published_date),
            editorial_voice=voice.value,
            word_count=self._count_words(summary_text),
            reading_level=self._assess_reading_level(summary_text),
        )
        return s

    def _get_editorial_prompt(
        self,
        voice: EditorialVoice,
        section: Section
    ) -> str:
        voice_map: Dict[EditorialVoice, str] = {
            EditorialVoice.HISTORICAL: (
                "Speak with the gravitas of history unfolding in real-time; "
                "capture significance and context in one sentence. "
                "Example: 'China's semiconductor breakthrough marks the end of Western monopoly, "
                "reshaping global tech power as profoundly as the transistor's invention.'"
            ),
            EditorialVoice.MEDICI: (
                "See commerce as culture; illuminate how business shapes "
                "society and strategy in one refined sentence. "
                "Example: 'Apple's $3 trillion valuation reflects not market mechanics but "
                "humanity's migration into digital existence, where devices become identity.'"
            ),
            EditorialVoice.POET_SCIENTIST: (
                "Translate nature's language into poetry; be elegant, precise, "
                "and accessible in one luminous sentence. "
                "Example: 'Quantum computers achieved supremacy by embracing uncertainty itself, "
                "teaching silicon to think in maybes rather than certainties.'"
            ),
            EditorialVoice.APPRENTICE: (
                "Learn from builders; extract practical wisdom in one concise "
                "sentence with humility and clarity. "
                "Example: 'Stripe's founders discovered that accepting payments wasn't the product—"
                "removing the fear of building payment systems was.'"
            ),
            EditorialVoice.CONTEMPLATIVE: (
                "Warm, spiritual reflection; find quiet meaning and hope in one "
                "gentle sentence. "
                "Example: 'Today's Gospel reminds us that mustard seeds of kindness, "
                "planted in ordinary Tuesday soil, grow into shelters for the weary.'"
            ),
            EditorialVoice.CIVIC: (
                "Thoughtful civic awareness; weigh institutions, citizens, and "
                "consequences in one responsible sentence. "
                "Example: 'The Senate's infrastructure vote transcends party lines, proving "
                "democracy still functions when focused on bridges rather than barriers.'"
            ),
            EditorialVoice.NEIGHBOR: (
                "Community connection; human-scale relevance in one welcoming "
                "sentence. "
                "Example: 'Miami's new flood gates protect not just buildings but abuela's "
                "kitchen where three generations still gather for Sunday cafecito.'"
            ),
            EditorialVoice.WANDERER: (
                "Intellectual serendipity; curious, cultured, and concise in "
                "one sentence. "
                "Example: 'Archaeologists discovered that medieval monks invented the @ symbol, "
                "proving that even Twitter's syntax has roots in illuminated manuscripts.'"
            ),
        }
        title = self._get_section_title(section)
        base = (
            f"Editorial Voice: {voice.value}\n"
            f"Section: {title}\n"
            "Write exactly one brilliant sentence (25-40 words)."
        )
        return base + "\n" + voice_map.get(voice, "")

    async def _batch_summarize(
        self,
        items: List[RankedItem],
        voice: EditorialVoice
    ) -> List[Summary]:
        results: List[Summary] = []
        tasks = [self._generate_summary(it, voice) for it in items]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for it, res in zip(items, gathered):
            if isinstance(res, Exception):
                self.logger.error("Summary generation failed for %s: %s. Using fallback.", it.id, res)
                results.append(self._create_fallback_summary(it))
            else:
                results.append(res)
        return results

    def _create_fallback_summary(self, item: RankedItem) -> Summary:
        """
        Deterministic fallback summary when AI generation/validation fails.
        """
        time_ago_str = self._calculate_time_ago(item.published_date)
        base = f"{item.headline}"
        # Include source and timing for minimal context
        if item.source:
            base += f" ({item.source} — {time_ago_str})"
        else:
            base += f" ({time_ago_str})"
        source_name, clean_url = self._format_source_attribution(item.source, item.url)
        return Summary(
            item_id=item.id,
            headline=item.headline,
            summary_text=base,
            source=source_name,
            source_url=clean_url,
            time_ago=time_ago_str,
            editorial_voice="fallback",
            word_count=self._count_words(base),
            reading_level=self._assess_reading_level(base),
        )


    def _calculate_time_ago(
        self,
        published_date: datetime
    ) -> str:
        # Align timezone awareness to avoid naive/aware subtraction errors
        now = datetime.now(published_date.tzinfo) if published_date.tzinfo is not None else datetime.now()
        delta: timedelta = now - published_date

        # Handle negative time deltas (future dates due to timezone issues or bad RSS timestamps)
        if delta.total_seconds() < 0:
            self.logger.warning(f"Article appears to be from the future: {published_date} vs {now}")
            return "just published"

        # Handle very large time deltas (likely bad data)
        if delta.days > 365:
            self.logger.warning(f"Article appears to be very old: {delta.days} days ago")
            return f"{delta.days} days ago"

        minutes = int(delta.total_seconds() // 60)
        hours = int(delta.total_seconds() // 3600)
        days = delta.days

        if minutes < 60:
            return f"{minutes} minutes ago" if minutes > 0 else "just now"
        if hours < 24:
            return f"{hours} hours ago"
        if days == 1:
            return "yesterday"
        return f"{days} days ago"

    def _format_source_attribution(
        self,
        source: str,
        url: str
    ) -> Tuple[str, str]:
        return source, url

    async def generate_section_intro(
        self,
        section: Section,
        summaries: List[Summary],
        all_items: Optional[List[RankedItem]] = None
    ) -> Optional[str]:
        """
        Generate a comprehensive preface for a section based on ALL gathered articles.
        
        Args:
            section: The section to generate intro for
            summaries: The selected/summarized items (for backward compatibility)
            all_items: ALL articles gathered for this section (for comprehensive context)
        
        Returns:
            A paragraph summarizing the overall themes and trends in the section
        """
        # Scripture gets a simple, spiritual intro
        if section == Section.SCRIPTURE:
            return "Today's readings and reflections for spiritual contemplation."
            
        # Get the editorial voice for this section
        voice = self.section_voices.get(section, EditorialVoice.WANDERER)
        section_title = self._get_section_title(section)
        
        try:
            # If we have all items, use them for comprehensive context
            if all_items:
                # Prepare article data for the AI service
                articles_data = []
                for item in all_items:
                    articles_data.append({
                        "headline": item.headline,
                        "summary": item.summary_text[:300] if item.summary_text else "",  # Truncate for API limits
                        "source": item.source
                    })
                
                # Use the new generate_section_preface method
                if hasattr(self.ai, "generate_section_preface"):
                    preface = await self.ai.generate_section_preface(
                        section_name=section_title,
                        all_articles=articles_data,
                        editorial_voice=voice.value
                    )
                    return preface
            
            # Fallback to using just the selected summaries if all_items not available
            if summaries and len(summaries) > 0:
                articles_data = []
                for s in summaries:
                    articles_data.append({
                        "headline": s.headline,
                        "summary": s.summary_text,
                        "source": s.source
                    })
                
                if hasattr(self.ai, "generate_section_preface"):
                    preface = await self.ai.generate_section_preface(
                        section_name=section_title,
                        all_articles=articles_data,
                        editorial_voice=voice.value
                    )
                    return preface
            
            return None
            
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Failed to generate section intro for {section}: {e}")
            return None

    def _get_section_title(
        self,
        section: Section
    ) -> str:
        name_map: Dict[str, str] = {
            Section.BREAKING_NEWS: "Breaking News",
            Section.BUSINESS: "Business",
            Section.TECH_SCIENCE: "Tech & Science",
            Section.RESEARCH_PAPERS: "Research Papers",
            Section.STARTUP: "Startup",
            Section.SCRIPTURE: "Scripture",
            Section.POLITICS: "Politics",
            Section.LOCAL: "Local",
            Section.MISCELLANEOUS: "Miscellaneous",
            Section.EXTRA: "Extra",
        }
        return name_map.get(section, str(section)).title()

    def _count_words(
        self,
        text: str
    ) -> int:
        if not text:
            return 0
        # Simple word count - more reliable than regex
        return len(text.split())
    
    def _split_into_sentences(
        self,
        text: str
    ) -> List[str]:
        """Split text into sentences, handling common abbreviations and middle initials."""
        if not text:
            return []
        
        # CRITICAL FIX: Handle single-letter middle initials (e.g., "Peter E. Gordon")
        # Replace middle initials with placeholder BEFORE splitting
        text = re.sub(r'\b([A-Z])\.\s+([A-Z][a-z])', r'\1<MIDDLE_INITIAL>\2', text)
        
        # Handle common abbreviations that shouldn't end sentences
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s+', r'\1<DOT> ', text)
        text = re.sub(r'\b(Inc|Corp|Ltd|Co)\.\s+', r'\1<DOT> ', text)
        text = re.sub(r'\b(U\.S|U\.K|E\.U)\.\s+', r'\1<DOT> ', text)
        
        # Split on sentence endings (period/question/exclamation followed by space and capital)
        # Only split if followed by capital letter (real sentence boundary)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore middle initials and abbreviations
        sentences = [s.replace('<MIDDLE_INITIAL>', '. ').replace('<DOT>', '.') for s in sentences if s.strip()]
        
        # Filter out empty sentences
        return [s for s in sentences if s.strip()]

    def _assess_reading_level(
        self,
        text: str
    ) -> float:
        # Lightweight heuristic: words per sentence proxy
        if not text:
            return 0.0
        words = self._count_words(text)
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        return round(words / sentences, 2)


    def get_voice_description(
        self,
        voice: EditorialVoice
    ) -> str:
        descriptions: Dict[EditorialVoice, str] = {
            EditorialVoice.HISTORICAL: "Gravitas of history unfolding now.",
            EditorialVoice.MEDICI: "Commerce as culture and strategy.",
            EditorialVoice.POET_SCIENTIST: "Poetic clarity for scientific truths.",
            EditorialVoice.APPRENTICE: "Practical wisdom from builders.",
            EditorialVoice.CONTEMPLATIVE: "Warm spiritual reflection.",
            EditorialVoice.CIVIC: "Thoughtful civic awareness.",
            EditorialVoice.NEIGHBOR: "Community-scale relevance.",
            EditorialVoice.WANDERER: "Curious intellectual serendipity.",
        }
        return descriptions.get(voice, voice.value)


class SummarizationError(Exception):
    """Custom exception for summarization failures"""
    pass


