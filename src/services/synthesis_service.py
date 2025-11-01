import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from google.genai import types
from src.services.ai_service import AIService
from src.services.summarization_service import SectionSummaries
from src.pipeline.content_aggregator import Section


@dataclass
class GoldenThread:
    """The insight revealing hidden patterns across the day's stories"""
    insight: str  # The non-obvious pattern or principle connecting stories
    confidence: float
    reasoning: str
    connected_sections: List[Section]
    key_stories: List[str]  # The specific stories that demonstrate the pattern

    def to_newsletter_text(self) -> str:
        """Convert to newsletter-ready text"""
        return self.insight  # Returns the insight that makes readers think differently


@dataclass
class DelightfulSurprise:
    """The perfect ending to the newsletter"""
    content: str
    type: str  # 'quote', 'fact', 'historical', 'question', 'connection'
    attribution: Optional[str] = None
    relevance_score: float = 0.0

    def to_newsletter_text(self) -> str:
        """Convert to newsletter-ready text"""
        if self.attribution:
            return f"{self.content}\n— {self.attribution}"
        return self.content


class SurpriseType(Enum):
    """Types of delightful surprises"""
    QUOTE = "quote"              # Perfect quote crystallizing themes
    HISTORICAL = "historical"    # Historical parallel illuminating present
    STATISTIC = "statistic"      # Stunning fact reframing everything
    QUESTION = "question"        # Beautiful question lingering in mind
    CONNECTION = "connection"    # Unexpected link between stories


class SynthesisService:
    """
    Discovers golden threads and creates delightful surprises.

    This service is the Renaissance mind of the newsletter, finding:
    - The golden thread: subtle theme connecting disparate stories
    - Cross-domain patterns: how science relates to business to art
    - Perfect endings: quotes, facts, or questions that delight
    - Intellectual coherence: making the day's chaos comprehensible
    """

    def __init__(self, ai_service: AIService):
        """
        Initialize synthesis service.

        Args:
            ai_service: AI service for pattern recognition
        """
        self.ai = ai_service

        # Configuration
        # Lowered from 0.85 to 0.80 based on Task 14 evidence showing 3 high-quality patterns
        # with confidence scores of 0.9, 0.85, and 0.8 that were all rejected.
        # The adjusted Golden Thread prompt (Task 14) emphasizes authenticity over rigid formulas,
        # so a slightly lower threshold (0.80) balances quality with availability while still
        # filtering out weak patterns (< 0.80).
        self.min_thread_confidence = 0.80   # Balanced threshold for authentic cross-disciplinary patterns
        self.min_sections_for_thread = 2    # Minimum sections for a valid pattern
        self.candidates_per_type = 2       # Generate candidates per surprise type

        self.logger = logging.getLogger(__name__)

    async def find_golden_thread(
        self,
        sections: Dict[str, SectionSummaries]
    ) -> Optional[GoldenThread]:
        """
        Discover the golden thread connecting the day's stories.

        Uses a multi-tier retry strategy with progressively lower thresholds
        to ensure a Golden Thread always appears in the newsletter.

        Args:
            sections: All summarized sections

        Returns:
            Golden thread (guaranteed to return something unless no patterns generated)
        """
        # Tier 1: Try with primary threshold (0.80)
        self.logger.info(f"🎯 Tier 1: Attempting Golden Thread with threshold {self.min_thread_confidence}")
        patterns = await self._analyze_cross_section_patterns(sections)
        thread = await self._evaluate_thread_candidates(patterns, sections, threshold=self.min_thread_confidence)

        if thread:
            self.logger.info(f"✅ Tier 1 SUCCESS: Found Golden Thread with confidence ≥{self.min_thread_confidence}")
            return thread

        # Tier 2: Retry with relaxed threshold (0.70)
        self.logger.warning(
            f"⚠️ Tier 1 FAILED: No patterns ≥{self.min_thread_confidence}. "
            f"Tier 2: Retrying with relaxed threshold 0.70..."
        )
        thread = await self._evaluate_thread_candidates(patterns, sections, threshold=0.70)

        if thread:
            self.logger.info(f"✅ Tier 2 SUCCESS: Found Golden Thread with confidence ≥0.70")
            return thread

        # Tier 3: Final fallback - take best available pattern (0.60)
        self.logger.warning(
            f"⚠️ Tier 2 FAILED: No patterns ≥0.70. "
            f"Tier 3: Taking best available pattern (threshold 0.60)..."
        )
        thread = await self._evaluate_thread_candidates(patterns, sections, threshold=0.60)

        if thread:
            self.logger.warning(
                f"⚠️ Tier 3 SUCCESS: Using lower-confidence Golden Thread (≥0.60). "
                f"Consider reviewing pattern quality."
            )
            return thread

        # Absolute fallback: Generate new patterns with more lenient prompt
        self.logger.error(
            f"❌ All tiers FAILED: No patterns ≥0.60. "
            f"Tier 4: Generating new patterns with lenient criteria..."
        )

        # Try one more time with a fresh generation
        patterns = await self._analyze_cross_section_patterns(sections)
        thread = await self._evaluate_thread_candidates(patterns, sections, threshold=0.50)

        if thread:
            self.logger.warning(
                f"⚠️ Tier 4 SUCCESS: Using minimal-confidence Golden Thread (≥0.50). "
                f"Quality may be lower than usual."
            )
            return thread

        # If we still have nothing, log critical error but return None
        # (The email compiler should handle this gracefully)
        self.logger.error(
            f"🚨 CRITICAL: Failed to generate Golden Thread after 4 attempts. "
            f"Total patterns analyzed: {len(patterns)}. This should rarely happen."
        )
        return None

    async def _analyze_cross_section_patterns(
        self,
        sections: Dict[str, SectionSummaries]
    ) -> List[Dict[str, Any]]:
        """
        Analyze patterns across different sections.

        Args:
            sections: Summarized content

        Returns:
            List of potential patterns
        """
        # Provide detailed content for rich pattern analysis
        # Include headlines AND summaries for better cross-sectional insights
        normalized_sections = {}
        
        for key, value in sections.items():
            if not value.summaries:
                continue
                
            # Build detailed content for pattern detection
            section_details = []
            
            # Include intro if available for context
            if value.intro_text:
                section_details.append(f"Overview: {value.intro_text}")
            
            # Include top 5 articles with full headlines and summaries
            # This gives AI specific details like names, places, amounts
            for i, summary in enumerate(value.summaries[:5]):
                section_details.append(
                    f"Article {i+1}: {summary.headline}. "
                    f"Summary: {summary.summary_text}"
                )
            
            # Join all details for comprehensive section representation
            normalized_sections[key] = "\n".join(section_details)
        
        context = {
            "sections": normalized_sections
        }
        self.logger.info(f"🔍 Analyzing cross-section patterns for {len(sections)} sections")
        self.logger.debug(f"Sections being analyzed: {list(sections.keys())}")
        
        try:
            # Use tool calling for reliable pattern synthesis
            self.logger.debug("Calling AI service for pattern synthesis using tool calling")

            # Create function declaration for pattern synthesis
            function = types.FunctionDeclaration(
                name="return_patterns",
                description="Return the analyzed patterns connecting news stories",
                parameters=types.Schema(
                    type='OBJECT',
                    properties={
                        'patterns': types.Schema(
                            type='ARRAY',
                            description='Array of patterns found in the news',
                            items=types.Schema(
                                type='OBJECT',
                                properties={
                                    'insight': types.Schema(
                                        type='STRING',
                                        description='The golden thread insight connecting stories'
                                    ),
                                    'confidence': types.Schema(
                                        type='NUMBER',
                                        description='Confidence score from 0.0 to 1.0'
                                    ),
                                    'sections': types.Schema(
                                        type='ARRAY',
                                        description='List of section names this pattern connects',
                                        items=types.Schema(type='STRING')
                                    ),
                                    'key_stories': types.Schema(
                                        type='ARRAY',
                                        description='List of key story headlines that demonstrate this pattern',
                                        items=types.Schema(type='STRING')
                                    )
                                },
                                required=['insight', 'confidence', 'sections']
                            )
                        )
                    },
                    required=['patterns']
                )
            )

            tools = [types.Tool(function_declarations=[function])]
            tool_choice = {"type": "tool", "name": "return_patterns"}

            # Format the prompt for pattern synthesis
            messages = self.ai._format_prompt("gpt5_pattern_synthesis", context)

            # Call Gemini with tool calling
            resp = await self.ai._call_gemini(
                messages=messages,
                max_tokens=8192,
                tools=tools,
                tool_choice=tool_choice
            )

            # Extract patterns from tool call
            tool = self.ai._extract_tool_call(resp)
            if tool:
                try:
                    args = json.loads(tool["function"]["arguments"]) or {}
                    patterns = args.get("patterns", [])
                    self.logger.info(f"📊 Found {len(patterns)} potential patterns via tool calling")

                    # Log pattern details for debugging
                    for i, pattern in enumerate(patterns):
                        insight = pattern.get('insight', 'Unknown')
                        confidence = pattern.get('confidence', 0)
                        sections_list = pattern.get('sections', [])
                        self.logger.debug(f"Pattern {i+1}: {insight} (confidence: {confidence:.2f}, sections: {sections_list})")

                    # Filter for high-confidence patterns
                    high_confidence = [p for p in patterns if p.get('confidence', 0) >= 0.7]
                    if high_confidence:
                        self.logger.info(f"✨ {len(high_confidence)} high-confidence patterns (≥0.7) available for golden thread")

                    return patterns

                except json.JSONDecodeError as parse_error:
                    self.logger.error(f"❌ Failed to parse pattern tool call arguments: {parse_error}")
                    return []
            else:
                self.logger.warning("⚠️ No tool call found in pattern synthesis response")
                return []

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"💥 Pattern analysis failed: {e}")
            return []


    async def _evaluate_thread_candidates(
        self,
        patterns: List[Dict[str, Any]],
        sections: Dict[str, SectionSummaries],
        threshold: Optional[float] = None
    ) -> Optional[GoldenThread]:
        """
        Evaluate and select best golden thread.

        Args:
            patterns: Potential patterns found
            sections: Content for validation
            threshold: Optional confidence threshold override (defaults to self.min_thread_confidence)

        Returns:
            Best golden thread or None
        """
        # Use provided threshold or fall back to instance default
        effective_threshold = threshold if threshold is not None else self.min_thread_confidence

        self.logger.info(f"🧵 Evaluating {len(patterns)} thread candidates (threshold: {effective_threshold})")

        best: Optional[Tuple[float, Dict[str, Any]]] = None
        rejected_patterns = []  # Track rejected patterns for detailed logging

        for i, p in enumerate(patterns):
            # Support 'insight' (newest), 'connection' (recent), or 'theme' (legacy)
            insight = p.get('insight') or p.get('connection') or p.get('theme', 'Unknown')
            self.logger.debug(f"Evaluating pattern {i+1}: {insight[:100]}...")

            try:
                strength = self._calculate_thread_strength(p, sections)
                confidence = float(p.get("confidence", 0.0))
                self.logger.info(
                    f"📊 Pattern {i+1}: strength={strength:.3f}, confidence={confidence:.3f}, "
                    f"threshold={effective_threshold}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to evaluate pattern {i+1}: {e}")
                continue

            # Require minimums
            connected = p.get("sections") or []
            if not isinstance(connected, list):
                self.logger.debug(f"Pattern {i+1}: Invalid sections format: {type(connected)}")
                continue

            if (confidence >= effective_threshold) and (len(connected) >= self.min_sections_for_thread):
                score = (confidence + strength) / 2.0
                self.logger.info(f"✅ Pattern {i+1}: QUALIFIED with score {score:.3f}")
                if best is None or score > best[0]:
                    best = (score, p)
                    self.logger.info(f"🏆 Pattern {i+1}: New best candidate (score: {score:.3f})")
            else:
                # Track rejection reason
                rejection_reason = []
                if confidence < effective_threshold:
                    rejection_reason.append(
                        f"confidence too low ({confidence:.3f} < {effective_threshold})"
                    )
                if len(connected) < self.min_sections_for_thread:
                    rejection_reason.append(
                        f"too few sections ({len(connected)} < {self.min_sections_for_thread})"
                    )

                rejected_patterns.append({
                    "pattern_num": i + 1,
                    "insight": insight[:150],  # Truncate for logging
                    "confidence": confidence,
                    "strength": strength,
                    "sections": len(connected),
                    "reason": ", ".join(rejection_reason)
                })

                self.logger.warning(
                    f"❌ Pattern {i+1}: REJECTED - {', '.join(rejection_reason)}"
                )

        if not best:
            self.logger.warning(
                f"❌ No qualifying golden thread found. "
                f"Evaluated {len(patterns)} patterns, all rejected."
            )

            # Log detailed rejection summary
            if rejected_patterns:
                self.logger.warning("📋 Rejected patterns summary:")
                for rp in rejected_patterns:
                    self.logger.warning(
                        f"  Pattern {rp['pattern_num']}: conf={rp['confidence']:.3f}, "
                        f"strength={rp['strength']:.3f}, sections={rp['sections']} - "
                        f"Reason: {rp['reason']}"
                    )
                    self.logger.info(f"  Text preview: {rp['insight']}")

            return None

        score, chosen = best
        # Use 'insight' if available, then 'connection', then 'theme'
        insight_text = chosen.get('insight') or chosen.get('connection') or chosen.get('theme', 'Unknown')
        self.logger.info(f"✅ Selected golden thread: '{insight_text}' (score: {score:.3f})")
        
        sections_list = self._map_sections(chosen.get("sections") or [])
        # Use 'key_stories' if available, otherwise fall back to 'insights' or 'keywords'
        key_stories = chosen.get("key_stories") or chosen.get("insights") or chosen.get("keywords") or []
        reasoning = "; ".join(key_stories) if isinstance(key_stories, list) and key_stories else insight_text
        
        thread = GoldenThread(
            insight=str(insight_text),  # Use the new field name!
            confidence=float(chosen.get("confidence", 0.0)),
            reasoning=reasoning,
            connected_sections=sections_list,
            key_stories=list(key_stories) if isinstance(key_stories, list) else [],
        )
        
        self.logger.debug(f"Golden thread created: {len(thread.connected_sections)} sections, {len(thread.key_stories)} key stories")
        return thread

    def _calculate_thread_strength(
        self,
        pattern: Dict[str, Any],
        sections: Dict[str, SectionSummaries]
    ) -> float:
        """
        Calculate strength of a potential golden thread.
        
        Enhanced to reward cross-sectional diversity and penalize single-domain patterns.
        """
        connected = pattern.get("sections") or []
        keywords = [k.lower() for k in (pattern.get("keywords") or []) if isinstance(k, str)]
        
        # Calculate how many sections the pattern actually connects
        available_sections = set(sections.keys())
        connected_sections = set(connected).intersection(available_sections)
        num_connected = len(connected_sections)
        total_available = len(available_sections)
        
        # Coverage ratio: what proportion of available sections does this pattern connect?
        coverage_ratio = num_connected / max(1, total_available) if total_available > 0 else 0.0
        
        # Diversity bonus: reward patterns that connect MORE sections
        diversity_bonus = 0.0
        if num_connected >= self.min_sections_for_thread:
            # Give bonus for each additional section beyond minimum
            # Max bonus of 0.3 for connecting 5+ sections
            diversity_bonus = min(0.3, (num_connected - self.min_sections_for_thread) * 0.1)
        
        # Keyword relevance across connected sections only
        keyword_ratio = 0.5  # Default neutral score
        if keywords and connected_sections:
            hits = 0
            for section_name in connected_sections:
                if section_name in sections:
                    section_data = sections[section_name]
                    # Check keywords in section's content
                    section_text = (section_data.intro_text or "").lower()
                    for summary in section_data.summaries[:3]:  # Check first few summaries
                        section_text += f" {summary.headline.lower()} {summary.summary_text.lower()}"
                    
                    for keyword in set(keywords):
                        if keyword in section_text:
                            hits += 1
                            break  # Count each section only once per keyword
            
            # Normalize by number of connected sections and keywords
            keyword_ratio = hits / max(1, min(len(connected_sections), len(set(keywords))))
        
        # Balance penalty: penalize if pattern claims many sections but evidence is weak
        balance_penalty = 0.0
        if len(connected) > num_connected:  # Pattern claims sections that don't exist
            balance_penalty = 0.1
        
        # Calculate final strength with emphasis on diversity
        # Weights: 30% coverage, 30% keywords, 30% diversity, 10% penalty
        strength = max(0.0, min(1.0, 
            0.3 * coverage_ratio + 
            0.3 * keyword_ratio + 
            0.3 * diversity_bonus + 
            0.1 * (1.0 - balance_penalty)
        ))
        
        self.logger.debug(f"Pattern strength calculation: coverage={coverage_ratio:.2f}, "
                         f"keywords={keyword_ratio:.2f}, diversity={diversity_bonus:.2f}, "
                         f"penalty={balance_penalty:.2f}, final={strength:.2f}")
        
        return strength

    async def create_delightful_surprise(
        self,
        sections: Dict[str, SectionSummaries],
        golden_thread: Optional[GoldenThread] = None
    ) -> DelightfulSurprise:
        """
        Create the perfect ending for the newsletter.
        """
        # Build rich AI context
        now = datetime.now()
        context = {
            "themes": self._extract_key_themes(sections),
            "sections": list(sections.keys()),
            "golden_thread": (golden_thread.insight if golden_thread else None),
            "date": now,
            "day_of_week": now.weekday(),
            "newsletter_content": self._build_newsletter_content(sections),
            "historical_events": [],  # Placeholder; integrate real events later
        }

        # Generate candidates across all surprise types and select the best
        candidates = await self._generate_all_surprise_candidates(context)

        # Score all candidates if needed
        scored: List[DelightfulSurprise] = []
        for c in candidates:
            if not c.relevance_score or c.relevance_score <= 0.0:
                c.relevance_score = await self._score_surprise_relevance(c, sections, golden_thread)
            scored.append(c)
        return self._select_best_surprise(scored)

    async def _generate_all_surprise_candidates(
        self,
        context: Dict[str, Any]
    ) -> List[DelightfulSurprise]:
        """Generate multiple surprise candidates across all types."""
        tasks = []
        
        # Generate candidates for each surprise type
        for _ in range(self.candidates_per_type):
            tasks.append(asyncio.create_task(self._generate_perfect_quote(dict(context))))
            tasks.append(asyncio.create_task(self._generate_historical_parallel(dict(context))))
            tasks.append(asyncio.create_task(self._generate_stunning_statistic(dict(context))))
            tasks.append(asyncio.create_task(self._generate_beautiful_question(dict(context))))
            tasks.append(asyncio.create_task(self._generate_unexpected_connection(dict(context))))
        
        results: List[DelightfulSurprise] = []
        for coro in asyncio.as_completed(tasks):
            try:
                results.append(await coro)
            except Exception:  # noqa: BLE001
                continue
        return results

    async def _generate_perfect_quote(
        self,
        context: Dict[str, Any]
    ) -> DelightfulSurprise:
        """Find a perfect quote crystallizing themes."""
        payload = dict(context)
        payload["type"] = "quote"
        resp = await self.ai.generate_delightful_surprise(
            newsletter_content=payload.get("newsletter_content", {}),
            date=payload.get("date", datetime.now()),
            historical_events=payload.get("historical_events", []),
        )
        return DelightfulSurprise(
            content=str(resp.get("content", "")),
            type=str(resp.get("type", "quote")),
            attribution=resp.get("attribution") or resp.get("source"),
            relevance_score=float(resp.get("relevance", 0.0) or 0.0),
        )

    async def _generate_historical_parallel(
        self,
        context: Dict[str, Any]
    ) -> DelightfulSurprise:
        """Find historical parallel illuminating present."""
        payload = dict(context)
        payload["type"] = "historical"
        resp = await self.ai.generate_delightful_surprise(
            newsletter_content=payload.get("newsletter_content", {}),
            date=payload.get("date", datetime.now()),
            historical_events=payload.get("historical_events", []),
        )
        return DelightfulSurprise(
            content=str(resp.get("content", "")),
            type=str(resp.get("type", "historical")),
            attribution=resp.get("attribution") or resp.get("source"),
            relevance_score=float(resp.get("relevance", 0.0) or 0.0),
        )

    async def _generate_stunning_statistic(
        self,
        context: Dict[str, Any]
    ) -> DelightfulSurprise:
        """Find statistic that reframes everything."""
        payload = dict(context)
        payload["type"] = "statistic"
        resp = await self.ai.generate_delightful_surprise(
            newsletter_content=payload.get("newsletter_content", {}),
            date=payload.get("date", datetime.now()),
            historical_events=payload.get("historical_events", []),
        )
        return DelightfulSurprise(
            content=str(resp.get("content", "")),
            type=str(resp.get("type", "statistic")),
            attribution=resp.get("attribution") or resp.get("source"),
            relevance_score=float(resp.get("relevance", 0.0) or 0.0),
        )

    async def _generate_beautiful_question(
        self,
        context: Dict[str, Any]
    ) -> DelightfulSurprise:
        """Create question that lingers in the mind."""
        payload = dict(context)
        payload["type"] = "question"
        resp = await self.ai.generate_delightful_surprise(
            newsletter_content=payload.get("newsletter_content", {}),
            date=payload.get("date", datetime.now()),
            historical_events=payload.get("historical_events", []),
        )
        return DelightfulSurprise(
            content=str(resp.get("content", "")),
            type=str(resp.get("type", "question")),
            attribution=resp.get("attribution") or resp.get("source"),
            relevance_score=float(resp.get("relevance", 0.0) or 0.0),
        )

    async def _generate_unexpected_connection(
        self,
        context: Dict[str, Any]
    ) -> DelightfulSurprise:
        """Find unexpected connection between stories."""
        payload = dict(context)
        payload["type"] = "connection"
        resp = await self.ai.generate_delightful_surprise(
            newsletter_content=payload.get("newsletter_content", {}),
            date=payload.get("date", datetime.now()),
            historical_events=payload.get("historical_events", []),
        )
        return DelightfulSurprise(
            content=str(resp.get("content", "")),
            type=str(resp.get("type", "connection")),
            attribution=resp.get("attribution") or resp.get("source"),
            relevance_score=float(resp.get("relevance", 0.0) or 0.0),
        )


    async def _score_surprise_relevance(
        self,
        surprise: DelightfulSurprise,
        sections: Dict[str, SectionSummaries],
        golden_thread: Optional[GoldenThread]
    ) -> float:
        """Score how relevant a surprise is to content (0-1)."""
        # Prefer AI-assigned score when provided
        if isinstance(surprise.relevance_score, (int, float)) and surprise.relevance_score > 0.0:
            return float(surprise.relevance_score)

        score = 0.5
        # Simple heuristic boost: overlap of words with themes and thread
        text = (surprise.content or "").lower()
        themes = [t.lower() for t in self._extract_key_themes(sections)]
        overlap = sum(1 for t in set(themes) if t and (t in text))
        if themes:
            score = 0.3 + 0.4 * (overlap / min(len(set(themes)), 10))
        if golden_thread and golden_thread.insight:
            if any(tok in text for tok in golden_thread.insight.lower().split()):
                score = min(1.0, score + 0.2)
        return max(0.0, min(1.0, score))

    def _select_best_surprise(
        self,
        candidates: List[DelightfulSurprise]
    ) -> DelightfulSurprise:
        """Select best surprise from candidates."""
        if not candidates:
            return DelightfulSurprise(content="", type="quote", relevance_score=0.0)
        return max(candidates, key=lambda c: float(c.relevance_score or 0.0))

    async def synthesize_complete(
        self,
        sections: Dict[str, SectionSummaries]
    ) -> Tuple[Optional[GoldenThread], DelightfulSurprise]:
        """
        Complete synthesis: find thread and create surprise.
        """
        thread = await self.find_golden_thread(sections)
        surprise = await self.create_delightful_surprise(sections, thread)
        return thread, surprise

    def _extract_key_themes(
        self,
        sections: Dict[str, SectionSummaries]
    ) -> List[str]:
        """
        Extract key themes from all content.
        """
        themes: List[str] = []
        for sec in sections.values():
            for s in sec.summaries:
                if s.headline:
                    themes.append(s.headline)
                if s.summary_text:
                    themes.append(s.summary_text)
        return themes

    def _build_newsletter_content(
        self,
        sections: Dict[str, SectionSummaries]
    ) -> Dict[str, Any]:
        """Build a structured dict representing newsletter content for AI prompts."""
        output: Dict[str, Any] = {}
        for section_key, sec in sections.items():
            output[section_key] = {
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
        return output

    async def validate_synthesis(
        self,
        golden_thread: Optional[GoldenThread],
        surprise: DelightfulSurprise,
        sections: Dict[str, SectionSummaries]
    ) -> bool:
        """
        Validate synthesis quality and relevance.
        """
        if surprise.relevance_score < 0.7:
            return False
        if golden_thread is not None and golden_thread.confidence < self.min_thread_confidence:
            return False
        return True

    def _map_sections(self, names: List[str]) -> List[Section]:
        """Map string section names to Section enum-like constants when possible."""
        # Section in this project is a class with string constants, not Enum subclass.
        # We will map to those constants by equality.
        mapping = {v: v for k, v in vars(Section).items() if not k.startswith("_")}
        out: List[Section] = []
        for n in names:
            if isinstance(n, str) and n in mapping:
                out.append(mapping[n])
        return out


class SynthesisError(Exception):
    """Custom exception for synthesis failures"""
    pass


