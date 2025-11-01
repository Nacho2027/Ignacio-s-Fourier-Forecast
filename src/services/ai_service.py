import os
import json
import asyncio
import yaml
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import types


@dataclass
class AIResponse:
    content: Any
    prompt_key: str
    model: str
    tokens_used: int
    response_time_ms: float
    temperature: float
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EditorialDecision:
    decision: str
    reason: str
    confidence: float
    reader_value: str
    editorial_note: Optional[str] = None


@dataclass
class RankingResult:
    story_id: str
    temporal_impact: float
    intellectual_novelty: float
    renaissance_breadth: float
    actionable_wisdom: float
    source_authority: float
    signal_clarity: float
    transformative_potential: float
    one_line_judgment: str
    total: float = 0.0  # We calculate this ourselves


class AIServiceError(Exception):
    pass


class AIService:
    """
    Central AI service for Gemini interactions using Google GenAI API.
    References: Google GenAI Python SDK documentation.
    """

    def __init__(self, api_key: Optional[str] = None, prompts_path: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or pass api_key parameter.")

        # Configure Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Prompts
        self.prompts_path = prompts_path or "config/prompts_v2.yaml"
        self.prompts = self._load_prompts()

        # Model and temperatures from config when present
        params = self.prompts.get("parameters", {}) if isinstance(self.prompts, dict) else {}
        model_cfg = params.get("gemini", {}) if isinstance(params, dict) else {}
        # Use Gemini 2.5 Pro as default model
        self.model = os.getenv("GEMINI_MODEL") or model_cfg.get("model", "gemini-2.5-pro")

        temps_cfg = params.get("temperatures", {}) if isinstance(params, dict) else {}
        self.temperatures: Dict[str, float] = {
            "ranking": float(temps_cfg.get("ranking", 0.3)),
            "deduplication": float(temps_cfg.get("deduplication", 0.3)),
            "summary": float(temps_cfg.get("summary", 0.5)),
            "synthesis": float(temps_cfg.get("synthesis", 0.6)),
            "validation": float(temps_cfg.get("validation", 0.0)),
        }

        self.max_retries = 5  # Increased for better reliability
        self.max_tokens = 8192  # Higher default for Gemini 2.5 Pro thinking mode
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML configuration file."""
        try:
            with open(self.prompts_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError as e:
            raise AIServiceError(f"Prompts file not found at {self.prompts_path}") from e
        except yaml.YAMLError as e:
            raise AIServiceError(f"Error parsing YAML at {self.prompts_path}: {e}") from e

    async def test_connection(self) -> bool:
        """Ping the API to validate connectivity and key."""
        try:
            self.logger.info("🔍 Testing AI service connection...")
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=100)
            )
            # Handle response.text access safely
            try:
                response_text = response.text if hasattr(response, 'text') else None
                self.logger.info(f"✅ AI service test response: {response_text if response_text else 'No content'}")
                return bool(response_text)
            except ValueError:
                # response.text throws error when function calls are present
                self.logger.info("✅ AI service test successful (function call response)")
                return True
        except Exception as e:
            self.logger.error(f"❌ AI service test connection failed: {e}")
            # Log more context for debugging
            self.logger.error(f"   API key configured: {'Yes' if self.api_key else 'No'}")
            self.logger.error(f"   API key length: {len(self.api_key) if self.api_key else 0}")
            self.logger.error(f"   Model: {self.model}")
            return False

    async def rank_stories(self, stories: List[Dict], section: str, cache_service=None) -> List[RankingResult]:
        # Debug logging for research papers
        if section == "research_papers":
            self.logger.info(f"AI Service: Ranking {len(stories)} research papers")
            for i, story in enumerate(stories[:3]):  # Log first 3 for debugging
                story_id = story.get('id', f'idx_{i+1}')
                self.logger.debug(f"  Paper {i+1} (ID: {story_id}): {story.get('headline', 'NO HEADLINE')[:50]}...")
                self.logger.debug(f"    Has content: {bool(story.get('content'))}, "
                                f"Content length: {len(story.get('content', ''))}")
        
        # Get section-specific prompt additions
        section_additions = self._get_section_ranking_additions(section)
        
        # Enhance stories with adaptive temporal factors and historical context
        enhanced_stories = []
        temporal_guidance = []
        historical_guidance = []
        
        for story in stories:
            enhanced_story = dict(story)
            published_date = story.get("published") or story.get("published_date") or ""

            # Calculate adaptive temporal factor
            temporal_factor = self._calculate_adaptive_temporal_factor(published_date, section)
            enhanced_story["temporal_relevance_factor"] = round(temporal_factor, 3)

            # Include source authority score if available
            if "source_authority" not in enhanced_story:
                # Default to middle score if not provided
                enhanced_story["source_authority"] = 5.0
            else:
                # Ensure it's a float
                enhanced_story["source_authority"] = float(enhanced_story.get("source_authority", 5.0))
            
            # Add temporal guidance for AI
            if temporal_factor >= 0.8:
                temporal_hint = "PEAK RELEVANCE"
            elif temporal_factor >= 0.6:
                temporal_hint = "HIGH RELEVANCE"
            elif temporal_factor >= 0.4:
                temporal_hint = "MODERATE RELEVANCE"
            elif temporal_factor >= 0.2:
                temporal_hint = "DECLINING RELEVANCE"
            else:
                temporal_hint = "LOW RELEVANCE"
            
            enhanced_story["temporal_hint"] = temporal_hint
            enhanced_stories.append(enhanced_story)
            
            # Track for summary guidance
            temporal_guidance.append(f"Story {story.get('id', 'N/A')}: {temporal_hint} (factor: {temporal_factor:.2f})")
            
            # Add historical context analysis
            enhanced_story["historical_connections"] = []
            enhanced_story["historical_context_strength"] = 0.0
            
            # Populate historical connections if cache service is available
            if cache_service:
                try:
                    headline = enhanced_story.get('headline', '')
                    content = enhanced_story.get('content', '')[:300]
                    historical_connections = await cache_service.find_historical_connections(
                        headline, content, section, days_back=90
                    )
                    
                    if historical_connections:
                        enhanced_story["historical_connections"] = historical_connections[:3]  # Top 3 connections
                        # Calculate historical context strength (0.0-1.0)
                        strength = min(1.0, len(historical_connections) * 0.2 + 
                                     sum(conn.get("relevance_score", 0) for conn in historical_connections[:3]) / 3)
                        enhanced_story["historical_context_strength"] = round(strength, 3)
                        
                        # Add historical guidance for AI
                        if strength >= 0.7:
                            hist_hint = "STRONG HISTORICAL CONTEXT"
                        elif strength >= 0.5:
                            hist_hint = "MODERATE HISTORICAL CONTEXT"
                        elif strength >= 0.3:
                            hist_hint = "WEAK HISTORICAL CONTEXT"
                        else:
                            hist_hint = "NO HISTORICAL CONTEXT"
                        
                        enhanced_story["historical_hint"] = hist_hint
                        historical_guidance.append(
                            f"Story {story.get('id', 'N/A')}: {hist_hint} "
                            f"(strength: {strength:.2f}, connections: {len(historical_connections)})"
                        )
                    else:
                        enhanced_story["historical_hint"] = "NO HISTORICAL CONTEXT"
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get historical connections for story {story.get('id')}: {e}")
                    enhanced_story["historical_hint"] = "NO HISTORICAL CONTEXT"
        
        context = {
            "section": section,
            "stories": enhanced_stories,
            "stories_json": json.dumps(enhanced_stories),
            "temporal_guidance": "\n".join(temporal_guidance),
            "historical_guidance": "\n".join(historical_guidance) if historical_guidance else "No historical context analysis available"
        }
        messages = self._format_prompt("gpt5_editorial_ranking", context)
        
        # Append section-specific instructions and temporal guidance
        if section_additions and len(messages) > 0:
            messages[-1]["content"] += f"\n\n{section_additions}"
            
        # Add temporal and historical scoring guidance
        temporal_instruction = f"""

ADAPTIVE TEMPORAL SCORING GUIDANCE:
Each story includes a temporal_relevance_factor (0.0-1.0) and temporal_hint.
For TEMPORAL IMPACT scoring, consider:
- Stories with PEAK/HIGH relevance: Base score 6-10 depending on lasting significance
- Stories with MODERATE relevance: Base score 4-8, emphasize lasting impact over recency
- Stories with DECLINING/LOW relevance: Must have exceptional lasting significance for scores >6

{context["temporal_guidance"]}

HISTORICAL CONTEXT AWARENESS:
Each story includes historical_context_strength (0.0-1.0) and historical_hint.
For INTELLECTUAL NOVELTY and CROSS-DOMAIN VALUE scoring, consider:
- Stories with STRONG historical context: These connect to previous themes - boost novelty if new angle
- Stories with MODERATE historical context: May be follow-ups or variations - score novelty carefully
- Stories with WEAK/NO historical context: Potentially truly novel topics - boost novelty if substantive

{context["historical_guidance"]}

SOURCE AUTHORITY SCORING:
Each story includes a source_authority score (1-10) based on publication credibility.
For SOURCE AUTHORITY dimension in your scoring:
- Use the provided source_authority value directly
- Scores 8-10: Premium sources (Reuters, AP, BBC, etc.)
- Scores 6-7: Major newspapers and established media
- Scores 4-5: Specialized or regional sources
- Scores 1-3: Lesser-known or questionable sources

Remember: Temporal impact measures lasting significance, not just recency. Historical context helps
identify genuinely novel developments vs. variations on recurring themes."""
        
        if len(messages) > 0:
            messages[-1]["content"] += temporal_instruction
        tools = [
            {
                "type": "custom",
                "name": "return_ranking",
                "description": "Return ranked stories with 7-axis scores.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "story_id": {"type": "string"},
                                    "temporal_impact": {"type": "number"},
                                    "intellectual_novelty": {"type": "number"},
                                    "renaissance_breadth": {"type": "number"},
                                    "actionable_wisdom": {"type": "number"},
                                    "source_authority": {"type": "number"},
                                    "signal_clarity": {"type": "number"},
                                    "transformative_potential": {"type": "number"},
                                    "one_line_judgment": {"type": "string"}
                                },
                                "required": ["story_id", "temporal_impact", "intellectual_novelty", 
                                           "renaissance_breadth", "actionable_wisdom", "source_authority",
                                           "signal_clarity", "transformative_potential", "one_line_judgment"]
                            }
                        }
                    },
                    "required": ["results"]
                }
            }
        ]
        tool_choice = {"type": "tool", "name": "return_ranking"}
        # Increase token limit for ranking many papers (30 papers * ~150 tokens each = ~4500 tokens)
        max_tokens = 8192 if len(enhanced_stories) > 15 else 4096
        resp = await self._call_gemini(messages=messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
        tool = self._extract_tool_call(resp)
        if tool is None:
            # single retry enforcing function again with same token limit
            resp = await self._call_gemini(messages=messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
            tool = self._extract_tool_call(resp)

        data: List[Dict[str, Any]] = []
        if tool is not None:
            try:
                args = json.loads(tool["function"]["arguments"]) or {}
                data = args.get("results") or []
            except Exception as e:
                raise AIServiceError(f"Failed to parse ranking tool arguments: {e}") from e
        else:
            # Fallback: parse JSON array from assistant text (unit-test friendly)
            text_content = self._extract_text_content(resp)
            if not text_content:
                raise AIServiceError("Expected tool call 'return_ranking' not found")
            try:
                parsed = json.loads(text_content)
                if isinstance(parsed, list):
                    data = parsed
                elif isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                    data = parsed["results"]
                else:
                    raise ValueError("Unexpected JSON shape for ranking results")
            except Exception as e:
                raise AIServiceError(f"Failed to parse ranking results JSON: {e}") from e
        try:
            results = []
            for item in data:
                # Fix common typos in field names that Gemini might make
                # Map of typo -> correct field name
                field_corrections = {
                    'intellectual_novelance': 'intellectual_novelty',  # Common typo
                    'intellectual_novely': 'intellectual_novelty',     # Another possible typo
                    'intelectual_novelty': 'intellectual_novelty',     # Missing 'l'
                    'transformative_potencial': 'transformative_potential',  # Spanish-influenced typo
                    'actionable_wisdow': 'actionable_wisdom',          # Typo in wisdom
                    'actionable_wisdon': 'actionable_wisdom',          # Another wisdom typo
                    'source_autorithy': 'source_authority',            # Authority typo
                    'source_autority': 'source_authority',             # Authority typo
                    'signal_clarty': 'signal_clarity',                 # Clarity typo
                    'rennaissance_breadth': 'renaissance_breadth',     # Double n typo
                    'renaisance_breadth': 'renaissance_breadth',       # Missing s typo
                }
                
                # Apply corrections to the item dictionary
                corrected_item = {}
                for key, value in item.items():
                    # Check if this key is a known typo
                    corrected_key = field_corrections.get(key, key)
                    if corrected_key != key:
                        self.logger.warning(f"Correcting typo in AI response: '{key}' -> '{corrected_key}'")
                    corrected_item[corrected_key] = value
                
                result = RankingResult(**corrected_item)
                # Calculate weighted total score based on the 7 axes
                result.total = (
                    result.temporal_impact * 0.25 +
                    result.intellectual_novelty * 0.20 +
                    result.renaissance_breadth * 0.15 +
                    result.actionable_wisdom * 0.15 +
                    result.source_authority * 0.10 +
                    result.signal_clarity * 0.10 +
                    result.transformative_potential * 0.05
                )
                results.append(result)
        except TypeError as e:
            raise AIServiceError(f"Invalid ranking fields: {e}") from e
        results.sort(key=lambda r: r.total, reverse=True)
        
        # Debug logging for research papers to track coverage
        if section == "research_papers":
            expected_ids = [str(i+1) for i in range(len(stories))]
            returned_story_ids = [r.story_id for r in results]
            missing_ids = [id for id in expected_ids if id not in returned_story_ids]
            self.logger.info(f"AI Service: Returned rankings for {len(returned_story_ids)}/{len(stories)} papers")
            if missing_ids:
                self.logger.warning(f"AI Service: Missing rankings for story IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
                # Log token usage info
                self.logger.info(f"AI Service: Used max_tokens={max_tokens} for {len(stories)} papers")
        
        return results

    async def editorial_deduplication(self, new_item: Dict, similar_items: List[Dict]) -> EditorialDecision:
        # Build a readable summary list for templates that aren't loop-aware
        similar_str = "\n".join(
            [
                f"- Headline: {s.get('headline', '')} | Source: {s.get('source', '')} | Similarity: {s.get('similarity', 0)} | Days Ago: {s.get('days_ago', 'N/A')}"
                for s in similar_items
            ]
        )
        context = {
            "new_item": new_item,
            "similar_items": similar_items,
            "similar_items_str": similar_str,
        }
        messages = self._format_prompt("editorial_deduplication", context)
        tools = [
            {
                "type": "custom",
                "name": "return_editorial_decision",
                "description": "Return editorial deduplication decision.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["keep", "filter", "follow_up"]},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reader_value": {"type": "string"},
                        "editorial_note": {"type": ["string", "null"]}
                    },
                    "required": ["decision", "reason", "confidence", "reader_value"]
                }
            }
        ]
        tool_choice = {"type": "tool", "name": "return_editorial_decision"}
        resp = await self._call_gemini(messages=messages, max_tokens=2048, tools=tools, tool_choice=tool_choice)
        tool = self._extract_tool_call(resp)
        if tool is None:
            resp = await self._call_gemini(messages=messages, max_tokens=2048, tools=tools, tool_choice=tool_choice)
            tool = self._extract_tool_call(resp)

        if tool is not None:
            try:
                data = json.loads(tool["function"]["arguments"]) or {}
                return EditorialDecision(**data)
            except Exception as e:
                raise AIServiceError(f"Failed to parse editorial decision: {e}") from e
        else:
            # Fallback: parse assistant text JSON
            text_content = self._extract_text_content(resp)
            if not text_content:
                raise AIServiceError("Expected tool call 'return_editorial_decision' not found")
            try:
                data = json.loads(text_content) or {}
                return EditorialDecision(**data)
            except Exception as e:
                raise AIServiceError(f"Failed to parse editorial decision JSON: {e}") from e

    async def summarize_section(self, section: str, items: List[Dict]) -> str:
        context = {"section": section, "items": items, "items_json": json.dumps(items)}
        messages = self._format_prompt("gpt5_renaissance_summary", context)
        resp = await self._call_gemini(
            messages=messages,
            max_tokens=3000,
        )
        return self._extract_text_content(resp)

    async def find_golden_thread(self, all_content: Dict[str, List]) -> Optional[str]:
        context = {"all_content": all_content, "all_content_json": json.dumps(all_content)}
        messages = self._format_prompt("gpt5_golden_thread", context)
        function = types.FunctionDeclaration(
            name="return_golden_thread",
            description="Return golden thread summary.",
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'thread': types.Schema(
                        type='STRING',
                        description='The golden thread insight connecting stories'
                    ),
                    'confidence': types.Schema(
                        type='NUMBER',
                        description='Confidence score for the thread (0.0-1.0)'
                    )
                },
                required=['thread', 'confidence']
            )
        )
        tools = [types.Tool(function_declarations=[function])]
        tool_choice = {"type": "tool", "name": "return_golden_thread"}
        resp = await self._call_gemini(messages=messages, max_tokens=2048, tools=tools, tool_choice=tool_choice)
        tool = self._extract_tool_call(resp)
        if tool is None:
            resp = await self._call_gemini(messages=messages, max_tokens=2048, tools=tools, tool_choice=tool_choice)
            tool = self._extract_tool_call(resp)
        if tool is not None:
            try:
                data = json.loads(tool["function"]["arguments"]) or {}
                thread = data.get("thread")
                confidence = float(data.get("confidence", 0))

                # CRITICAL: Validate Golden Thread length (must be 150-200 words / 400+ chars)
                if thread:
                    thread_length = len(thread)
                    word_count = len(thread.split())

                    # STRICT REJECTION: Threads shorter than 400 chars are rejected entirely
                    if thread_length < 400:  # ~150 words minimum
                        self.logger.error(
                            f"❌ Golden Thread REJECTED: {thread_length} chars ({word_count} words). "
                            f"Required: 400+ chars (150-200 words). "
                            f"This is too short to provide genuine signal extraction."
                        )
                        return None  # Reject short threads completely
                    elif thread_length < 600:  # ~200 words target
                        self.logger.info(
                            f"✅ Golden Thread length acceptable: {thread_length} chars ({word_count} words)"
                        )
                    else:
                        self.logger.info(
                            f"✅ Golden Thread length excellent: {thread_length} chars ({word_count} words)"
                        )

                return thread if (thread and confidence >= 0.7) else None
            except Exception:
                return None
        # Fallback: parse assistant text JSON
        text_content = self._extract_text_content(resp)
        try:
            data = json.loads(text_content) if text_content else {}
            thread = data.get("thread")
            confidence = float(data.get("confidence", 0))

            # CRITICAL: Validate Golden Thread length (must be 150-200 words / 400+ chars)
            if thread:
                thread_length = len(thread)
                word_count = len(thread.split())

                # STRICT REJECTION: Threads shorter than 400 chars are rejected entirely
                if thread_length < 400:  # ~150 words minimum
                    self.logger.error(
                        f"❌ Golden Thread REJECTED: {thread_length} chars ({word_count} words). "
                        f"Required: 400+ chars (150-200 words). "
                        f"This is too short to provide genuine signal extraction."
                    )
                    return None  # Reject short threads completely
                elif thread_length < 600:  # ~200 words minimum
                    self.logger.info(
                        f"✅ Golden Thread length acceptable: {thread_length} chars ({word_count} words)"
                    )
                else:
                    self.logger.info(
                        f"✅ Golden Thread length excellent: {thread_length} chars ({word_count} words)"
                    )

            # Enforce higher bar to avoid forced, low-quality threads
            return thread if (thread and confidence >= 0.8) else None
        except Exception:
            return None

    async def generate_delightful_surprise(
        self,
        newsletter_content: Dict,
        date: datetime,
        historical_events: List[str],
    ) -> Optional[Dict]:
        context = {
            "newsletter_content": newsletter_content,
            "date": date.isoformat(),
            "historical_events": historical_events,
        }
        messages = self._format_prompt("gpt5_serendipity_engine", context)
        tools = [
            {
                "type": "custom",
                "name": "return_delightful_surprise",
                "description": "Return delightful surprise block.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "content": {"type": "string"},
                        "source": {"type": ["string", "null"]}
                    },
                    "required": ["type", "content"]
                }
            }
        ]
        tool_choice = {"type": "tool", "name": "return_delightful_surprise"}
        resp = await self._call_gemini(messages=messages, max_tokens=4096, tools=tools, tool_choice=tool_choice)
        tool = self._extract_tool_call(resp)
        if tool is None:
            resp = await self._call_gemini(messages=messages, max_tokens=4096, tools=tools, tool_choice=tool_choice)
            tool = self._extract_tool_call(resp)
        if tool is not None:
            try:
                return json.loads(tool["function"]["arguments"]) or {}
            except Exception as e:
                raise AIServiceError(f"Failed to parse delightful surprise JSON: {e}") from e
        # Fallback: parse assistant text JSON (must be strict JSON per tests)
        text_content = self._extract_text_content(resp)
        try:
            return json.loads(text_content)
        except Exception as e:
            raise AIServiceError("Expected tool call or strict JSON for delightful surprise, but got invalid text") from e


    async def generate_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a one-sentence summary with confidence via tool-calling.
        Expects payload containing: headline, content, url, source, section, voice, constraints, prompt
        Returns: {"summary": str, "confidence": float, "content_quality": Dict, "generation_mode": str}
        """
        headline = payload.get("headline", "")
        content = payload.get("content", "")
        source = payload.get("source", "")
        url = payload.get("url", "")
        section = payload.get("section", "")
        voice = payload.get("voice", "")
        prompt_hint = payload.get("prompt", "")

        # Content Quality Assessment
        content_quality = self._assess_content_quality(headline, content, source)

        # Enhanced logging with quality metrics
        self.logger.info(f"📊 Summary generation for: {headline[:50]}... (source: {source})")
        self.logger.info(f"📊 Content Quality: {content_quality['quality_category']} - "
                        f"Content: {len(content)} chars, Headline: {len(headline)} chars, "
                        f"Ratio: {content_quality['content_to_headline_ratio']:.1f}")

        content = content.strip()

        # Determine generation mode and content processing
        generation_mode, processed_content = self._determine_generation_mode(headline, content, content_quality)

        self.logger.info(f"🎯 Generation mode: {generation_mode}")

        # Log content quality statistics for tracking
        self._log_content_quality_stats(content_quality, generation_mode, source, section)

        # Generate prompts based on determined mode
        system_text, user_text = self._generate_mode_specific_prompts(
            generation_mode, headline, processed_content, source, url, section, voice, prompt_hint
        )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        tools = [
            {
                "type": "custom",
                "name": "return_summary",
                "description": "Return a one-sentence summary and confidence",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["summary", "confidence"]
                }
            }
        ]
        tool_choice = {"type": "tool", "name": "return_summary"}

        # Try AI generation with multiple fallback strategies
        for attempt in range(3):  # Up to 3 attempts
            try:
                self.logger.debug(f"Summary generation attempt {attempt + 1} for: {headline[:30]}...")

                resp = await self._call_gemini(messages=messages, max_tokens=1024, tools=tools, tool_choice=tool_choice)

                # Log the response structure for debugging
                self.logger.debug(f"AI response keys: {list(resp.keys()) if isinstance(resp, dict) else 'Not a dict'}")

                # Check if we got a text response instead of tool call (indicates error)
                text_response = self._extract_text_content(resp)
                if text_response and any(error_word in text_response.lower() for error_word in ["cannot", "sorry", "unable", "error"]):
                    self.logger.warning(f"AI returned error message: {text_response[:200]}...")
                    if attempt < 2:  # Try again on first two attempts
                        continue
                    return self._create_fallback_summary(headline, "AI declined to summarize", 0.3)

                tool = self._extract_tool_call(resp)
                if tool is None:
                    self.logger.warning(f"No tool call found in attempt {attempt + 1}")
                    if attempt == 0:
                        # First retry with enhanced prompt
                        enhanced_system = (
                            "You are the Renaissance editor. Write exactly one brilliant, factual sentence (25-40 words). "
                            "No markdown, no bullets, no code. Use clear, vivid language. "
                            "CRITICAL: You MUST call the return_summary function with your summary and confidence."
                        )
                        messages[0]["content"] = enhanced_system
                        continue
                    elif attempt == 1:
                        # Second retry with simpler prompt
                        messages[0]["content"] = "Create a concise one-sentence summary. Use the return_summary function."
                        messages[1]["content"] = f"Summarize: {headline}\n\nContent: {content[:500]}..."
                        continue
                    else:
                        # Final attempt failed
                        self.logger.error("All attempts to get tool call failed")
                        return self._create_fallback_summary(headline, "Tool call extraction failed", 0.2)

                # Successfully got a tool call, try to parse it
                try:
                    data = json.loads(tool["function"]["arguments"]) or {}
                    summary = data.get("summary", "").strip()
                    confidence = float(data.get("confidence", 0.0))

                    # Validate summary quality
                    if not summary:
                        self.logger.warning(f"AI returned empty summary")
                        if attempt < 2:
                            continue
                        return self._create_fallback_summary(headline, "Empty AI response", 0.3)

                    if len(summary.split()) < 3:
                        self.logger.warning(f"AI returned very short summary: '{summary}'")
                        if attempt < 2:
                            continue
                        return self._create_fallback_summary(headline, "AI summary too short", 0.4)

                    # CRITICAL: Ensure summary is complete and not truncated
                    # Check for common truncation patterns
                    if summary.endswith('...') or summary.endswith('…'):
                        self.logger.warning(f"Summary appears truncated: {summary}")
                    
                    # Ensure summary doesn't get cut off mid-sentence
                    if len(summary.split()) < 5:
                        self.logger.warning(f"Summary suspiciously short ({len(summary.split())} words): {summary}")
                    
                    # Success! Return the generated summary with quality metrics
                    # Adjust confidence based on generation mode and content quality
                    final_confidence = self._calculate_final_confidence(
                        confidence, generation_mode, content_quality, len(summary.split())
                    )

                    # Validate summary quality
                    summary_validation = self._validate_summary_quality(summary, headline, generation_mode)

                    self.logger.info(f"✅ Generated {generation_mode} summary: {summary[:100]}... "
                                   f"(confidence: {final_confidence:.2f}, validation: {summary_validation['quality_score']:.2f}, length: {len(summary)} chars)")

                    return self._create_enhanced_response(
                        summary, final_confidence, content_quality, generation_mode, summary_validation
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to parse summary result on attempt {attempt + 1}: {e}")
                    if attempt < 2:
                        continue
                    return self._create_fallback_summary(headline, "Failed to parse AI response", 0.3)

            except Exception as e:
                self.logger.error(f"Error in generate_summary attempt {attempt + 1}: {e}")
                if attempt < 2:
                    continue
                return self._create_fallback_summary(headline, f"AI service error: {str(e)[:50]}", 0.2)

        # Should never reach here, but just in case
        fallback_reason = "All attempts exhausted"
        fallback_confidence = 0.1 if generation_mode == "content_based" else 0.5
        return self._create_fallback_summary(headline, fallback_reason, fallback_confidence)

    def _create_enhanced_response(self, summary: str, confidence: float,
                                content_quality: Dict[str, Any], generation_mode: str,
                                summary_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized enhanced response with all fields."""
        return {
            "summary": summary,                        # Core field - always present
            "confidence": confidence,                  # Core field - always present
            "content_quality": content_quality or {},  # Enhancement - optional
            "generation_mode": generation_mode or "unknown",  # Enhancement - optional
            "summary_validation": summary_validation or {}  # Enhancement - optional
        }

    def _create_fallback_summary(self, headline: str, reason: str, confidence: float,
                                content_quality: Dict[str, Any] = None, generation_mode: str = "fallback") -> Dict[str, Any]:
        """
        Create a fallback summary when AI generation fails.

        Args:
            headline: The article headline
            reason: Why we're falling back (for logging)
            confidence: Confidence score for the fallback
            content_quality: Content quality assessment
            generation_mode: The attempted generation mode

        Returns:
            Dict with summary, confidence, and quality metrics
        """
        # Clean up the headline for use as a summary
        clean_headline = headline.strip()
        if not clean_headline:
            clean_headline = "Article summary unavailable"

        # Create a more informative fallback than just the headline
        # Try to make it sound like a proper summary sentence
        if len(clean_headline.split()) <= 3:
            # Very short headline, add some context
            fallback_summary = f"{clean_headline}."
        elif clean_headline.endswith('.'):
            # Already ends with period
            fallback_summary = clean_headline
        else:
            # Add period to make it a complete sentence
            fallback_summary = f"{clean_headline}."

        self.logger.warning(f"🔄 Created fallback summary: '{fallback_summary}' (reason: {reason})")

        # Create default content quality if not provided (for backward compatibility)
        if content_quality is None:
            content_quality = {
                "headline_length": len(headline),
                "content_length": 0,
                "content_to_headline_ratio": 0.0,
                "quality_category": "no_content",
                "is_content_duplicate": False,
                "source_reliability": "unknown",
                "has_meaningful_content": False
            }

        # Create minimal validation for fallback
        summary_validation = {
            "quality_score": 0.3,  # Low quality for fallbacks
            "is_headline_paraphrase": True,
            "content_utilization": "headline_only",
            "issues": [reason, "fallback_generation"]
        }

        return self._create_enhanced_response(
            fallback_summary, confidence, content_quality, f"fallback_{generation_mode}", summary_validation
        )

    def _calculate_adaptive_temporal_factor(self, published_date: str, section: str) -> float:
        """
        Calculate adaptive temporal decay factor based on section-specific requirements.
        
        Args:
            published_date: ISO format date string
            section: Newsletter section
            
        Returns:
            float: Temporal relevance factor (0.0-1.0)
        """
        try:
            from datetime import datetime
            if not published_date:
                return 0.5  # Default for unknown dates
                
            pub_dt = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            now = datetime.now()
            hours_old = (now - pub_dt).total_seconds() / 3600
            
            # Section-specific temporal decay parameters
            decay_params = {
                "breaking_news": {
                    "half_life_hours": 6,    # Very fast decay for breaking news
                    "min_factor": 0.1,       # Steep penalty for old breaking news
                    "peak_hours": 2,         # Peak relevance within 2 hours
                    "grace_period": 24       # Still relevant within 24 hours
                },
                "business": {
                    "half_life_hours": 48,   # Moderate decay for business news
                    "min_factor": 0.3,       # Business stories age better
                    "peak_hours": 8,         # Peak relevance within 8 hours
                    "grace_period": 72       # Relevant for 3 days
                },
                "tech_science": {
                    "half_life_hours": 168,  # Slow decay for tech/science (1 week)
                    "min_factor": 0.6,       # Tech discoveries stay relevant longer
                    "peak_hours": 24,        # Peak within 24 hours
                    "grace_period": 336      # Relevant for 2 weeks
                },
                "research_papers": {
                    "half_life_hours": 720,  # Very slow decay (1 month)
                    "min_factor": 0.8,       # Research stays relevant
                    "peak_hours": 72,        # Peak within 3 days
                    "grace_period": 2160     # Relevant for 3 months
                },
                "startup": {
                    "half_life_hours": 240,  # Moderate-slow decay (10 days)
                    "min_factor": 0.4,       # Startup advice ages moderately
                    "peak_hours": 48,        # Peak within 2 days  
                    "grace_period": 336      # Relevant for 2 weeks
                },
                "politics": {
                    "half_life_hours": 72,   # Fast decay for politics
                    "min_factor": 0.2,       # Political news becomes stale quickly
                    "peak_hours": 12,        # Peak within 12 hours
                    "grace_period": 120      # Relevant for 5 days
                },
                "local": {
                    "half_life_hours": 96,   # Moderate decay for local news
                    "min_factor": 0.3,       # Local news has lasting community impact
                    "peak_hours": 24,        # Peak within 24 hours
                    "grace_period": 168      # Relevant for 1 week
                },
                "miscellaneous": {
                    "half_life_hours": 336,  # Slow decay for Renaissance topics
                    "min_factor": 0.5,       # Intellectual content stays relevant
                    "peak_hours": 72,        # Peak within 3 days
                    "grace_period": 672      # Relevant for 4 weeks
                }
            }
            
            params = decay_params.get(section, decay_params["miscellaneous"])
            
            # Calculate temporal factor using sophisticated decay model
            if hours_old <= params["peak_hours"]:
                # Peak relevance period - linear increase to 1.0
                factor = 0.8 + (0.2 * (1 - hours_old / params["peak_hours"]))
            elif hours_old <= params["grace_period"]:
                # Grace period - exponential decay but not too steep
                decay_rate = 0.693 / params["half_life_hours"]  # ln(2) / half_life
                factor = max(params["min_factor"], 0.8 * (1 - decay_rate * (hours_old - params["peak_hours"]) / 24))
            else:
                # Beyond grace period - apply minimum factor with slight decay
                factor = max(0.1, params["min_factor"] * (params["grace_period"] / hours_old))
            
            return min(1.0, max(0.1, factor))
            
        except Exception:
            return 0.5  # Fallback for parsing errors

    def _get_section_ranking_additions(self, section: str) -> str:
        """Get section-specific prompt additions for ranking with adaptive temporal guidance."""
        section_prompts = {
            "breaking_news": """
BREAKING NEWS CRITERIA:
Focus on historically significant events:
- Events with long-term global impact
- Major geopolitical developments affecting millions
- Policy decisions with lasting consequences
- Prioritize TEMPORAL IMPACT - lasting significance over momentary attention
- Prioritize SIGNAL CLARITY - verified facts, not speculation
- Reject: routine political processes, minor incidents, sensationalism
Standard: Will this matter in 6 months?""",
            
            "business": """
BUSINESS CRITERIA:
Focus on strategic industry developments:
- Moves that reshape entire industries
- Capital allocation revealing future trends
- Leadership lessons from proven executives
- Prioritize ACTIONABLE WISDOM - what can leaders learn?
- Prioritize TEMPORAL IMPACT - lasting business implications
- Reject: routine earnings, minor market movements, corporate announcements
Standard: Significant strategic value for decision-makers.""",
            
            "tech_science": """
TECH & SCIENCE CRITERIA:
Focus on breakthrough discoveries:
- Advances that expand what's possible
- Research addressing fundamental questions
- Prioritize INTELLECTUAL NOVELTY - genuinely new knowledge
- Prioritize TRANSFORMATIVE POTENTIAL - field-changing potential
- Scientific rigor and reproducible results
- Reject: incremental improvements, vendor announcements, unsubstantiated claims
Standard: Advances that redefine understanding.""",
            
            "startup": """
STARTUP CRITERIA:
Focus on proven wisdom from builders:
- Tested strategies from successful founders
- Counterintuitive insights backed by experience
- Specific, applicable tactics
- Prioritize ACTIONABLE WISDOM - immediate practical value
- Prioritize TRANSFORMATIVE POTENTIAL - scalable principles
- Evidence-based advice, not theory
- Reject: generic motivation, unproven speculation, promotional content
Standard: Practical value for entrepreneurs.""",
            
            "miscellaneous": """
MISCELLANEOUS CRITERIA (RENAISSANCE BREADTH):
Focus on humanities and liberal arts excellence with intellectual diversity:
- Philosophy, history, literature, art, music, architecture insights
- Cultural criticism and anthropological perspectives
- Human nature, consciousness, health, and ethical explorations
- ACTIVELY PRIORITIZE: Intellectual diversity across humanities disciplines
- ACTIVELY PRIORITIZE: Philosophy, history, culture, health, psychology topics
- ACTIVELY PRIORITIZE: Essays that expand understanding of human experience
- BALANCE is key: Seek variety across philosophy, history, arts, culture, health
- Include profound philosophical takes on modern topics (including tech/AI if deeply insightful)
- Focus on substance and depth rather than excluding topics categorically
Standard: Expands horizons through diverse intellectual pursuits and cultural understanding.""",
            
            "politics": """
POLITICS CRITERIA:
Focus on institutional and policy substance:
- Structural changes to governance systems
- Policy decisions with long-term impact
- Prioritize TEMPORAL IMPACT - lasting institutional effects
- Prioritize SIGNAL CLARITY - factual developments, not speculation
- Governance mechanics over political theater
- Reject: campaign coverage, partisan commentary, polling data
Standard: Governance substance over political process.""",
            
            "research_papers": """
RESEARCH PAPERS CRITERIA:
Focus on significant academic contributions:
- Papers opening new research directions
- Prioritize INTELLECTUAL NOVELTY - genuine knowledge advances
- Prioritize SOURCE AUTHORITY - top-tier journals and conferences
- Cross-disciplinary relevance
- Methodological rigor and reproducibility
- Reject: incremental studies, confirmatory research, poor methodology
Standard: Citeable contributions to knowledge.""",
            
            "local": """
LOCAL CRITERIA:
Focus on community impact:
- Developments affecting daily life in Miami/Cornell
- Infrastructure, education, safety improvements
- Community initiatives with broader applications
- Prioritize SIGNAL CLARITY - factual local developments
- Local lessons with universal principles
- Reject: minor administrative news, routine announcements
Standard: Meaningful impact on community life.""",
            
        }
        
        return section_prompts.get(section, "")

    def _convert_json_schema_to_gemini_schema(self, json_schema: Dict[str, Any]) -> types.Schema:
        """
        Convert a JSON Schema to Gemini's Schema format using new SDK types.

        Returns a types.Schema object compatible with the new Google GenAI library.
        """
        # Type mapping from JSON Schema to Gemini type enums
        type_mapping = {
            'string': types.Type.STRING,
            'number': types.Type.NUMBER,
            'integer': types.Type.INTEGER,
            'boolean': types.Type.BOOLEAN,
            'array': types.Type.ARRAY,
            'object': types.Type.OBJECT,
        }

        def convert_type_enum(schema_type):
            """Convert JSON Schema type to Gemini type enum."""
            if isinstance(schema_type, list):
                # Handle union types like ['string', 'null'] - use the primary (non-null) type
                primary_types = [t for t in schema_type if t != 'null']
                if primary_types:
                    return type_mapping.get(primary_types[0], types.Type.STRING)
                else:
                    return types.Type.STRING  # Default fallback
            else:
                return type_mapping.get(schema_type, types.Type.STRING)

        def convert_schema_recursive(schema):
            """Recursively convert schema structure to types.Schema."""
            if not isinstance(schema, dict):
                return schema

            schema_params = {}

            # Convert type to enum
            if 'type' in schema:
                schema_params['type'] = convert_type_enum(schema['type'])

            # Copy safe fields
            if 'description' in schema:
                schema_params['description'] = schema['description']
            if 'enum' in schema:
                schema_params['enum'] = schema['enum']

            # Handle properties for objects
            if 'properties' in schema:
                converted_properties = {}
                for prop_name, prop_schema in schema['properties'].items():
                    converted_properties[prop_name] = convert_schema_recursive(prop_schema)
                schema_params['properties'] = converted_properties

            # Handle required fields
            if 'required' in schema:
                schema_params['required'] = schema['required']

            # Handle array items
            if 'items' in schema:
                schema_params['items'] = convert_schema_recursive(schema['items'])

            return types.Schema(**schema_params)

        try:
            # Convert the schema using the recursive converter
            converted = convert_schema_recursive(json_schema)
            return converted

        except Exception as e:
            # Robust fallback: create a minimal schema that always works
            self.logger.warning(f"Schema conversion failed, using minimal fallback: {e}")
            return types.Schema(
                type=types.Type.OBJECT,
                description='Function parameters',
                properties={}
            )

    async def _call_gemini(
        self,
        messages: List[Dict],
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Call Google Gemini API.
        Uses the Google GenAI Python SDK for proper API interaction.
        """
        try:
            # Convert messages format - Gemini uses system instructions and contents
            system_instruction = None
            contents = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_instruction = msg["content"]
                else:
                    # Gemini expects contents as strings or objects
                    contents.append(msg["content"])

            # Build the configuration using new SDK types
            config_params = {
                "max_output_tokens": max_tokens,
                "temperature": 0.3,  # Default temperature, can be made configurable
            }

            if system_instruction:
                config_params["system_instruction"] = system_instruction

            # Convert tools to Gemini format
            gemini_tools = None
            if tools:
                function_declarations = []
                for tool in tools:
                    # Handle both old dict format and new types.Tool format
                    if isinstance(tool, types.Tool):
                        # Tool is already in the new format, use it directly
                        if not gemini_tools:
                            gemini_tools = []
                        gemini_tools.append(tool)
                    elif isinstance(tool, dict) and tool.get("type") == "custom":
                        # Convert old custom tool to Gemini function declaration format
                        function_declaration = types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool["description"],
                            parameters_json_schema=tool["input_schema"]
                        )
                        function_declarations.append(function_declaration)

                # If we converted any old-format tools, create Tool objects for them
                if function_declarations:
                    if not gemini_tools:
                        gemini_tools = []
                    gemini_tools.append(types.Tool(function_declarations=function_declarations))

                # Set tools and config if we have any tools
                if gemini_tools:
                    config_params["tools"] = gemini_tools

                    # CRITICAL: Disable automatic function calling to get function_calls in response
                    config_params["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                        disable=True
                    )

                    # Handle tool choice (function calling mode)
                    if tool_choice and tool_choice.get("type") == "tool":
                        # Force function calling - use tool_config with ANY mode
                        config_params["tool_config"] = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(mode="ANY")
                        )

            config = types.GenerateContentConfig(**config_params)

            # Call the Gemini API (async) using new client pattern with timeout protection
            try:
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config
                    ),
                    timeout=180.0  # 180 second (3 minute) timeout for Gemini API calls
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Gemini API call timed out after 180 seconds")
                raise AIServiceError("Gemini API call timed out after 180 seconds")

            # Convert response to format expected by our code
            usage_meta = getattr(response, 'usage_metadata', None)
            result = {
                "id": "gemini_response",  # Gemini doesn't provide ID
                "model": self.model,
                "usage": {
                    "input_tokens": getattr(usage_meta, 'prompt_token_count', 0) if usage_meta else 0,
                    "output_tokens": getattr(usage_meta, 'candidates_token_count', 0) if usage_meta else 0,
                    "total_tokens": getattr(usage_meta, 'total_token_count', 0) if usage_meta else 0,
                },
                "stop_reason": "stop",  # Default stop reason
            }

            # Handle content and function calls using new SDK response format
            tool_calls = []
            text_content = ""

            # Check for function calls using the correct new SDK pattern
            # Debug logging for response structure
            self.logger.debug(f"Response has function_calls attribute: {hasattr(response, 'function_calls')}")
            if hasattr(response, 'function_calls'):
                self.logger.debug(f"Function calls found: {response.function_calls}")
                self.logger.debug(f"Function calls type: {type(response.function_calls)}")
                if response.function_calls:
                    self.logger.debug(f"First function call: {response.function_calls[0]}")
                    self.logger.debug(f"First function call type: {type(response.function_calls[0])}")

            # Handle both response.function_calls formats: FunctionCall objects vs wrapped objects
            if hasattr(response, 'function_calls') and response.function_calls:
                self.logger.debug(f"Found {len(response.function_calls)} function calls in response")
                for function_call_part in response.function_calls:
                    try:
                        # Check if this is a direct FunctionCall object or a wrapper
                        if hasattr(function_call_part, 'function_call'):
                            # Wrapper format: response.function_calls[0].function_call.args
                            function_call = function_call_part.function_call
                            function_name = function_call_part.name
                        else:
                            # Direct FunctionCall format: response.function_calls[0].args
                            function_call = function_call_part
                            function_name = function_call_part.name

                        # Extract arguments using the correct pattern from Context7 docs
                        if hasattr(function_call, 'args') and function_call.args:
                            # Convert protobuf struct to dictionary
                            args_dict = dict(function_call.args)
                            arguments_json = json.dumps(args_dict)
                        else:
                            arguments_json = "{}"

                        tool_calls.append({
                            "type": "custom",
                            "function": {
                                "name": function_name,
                                "arguments": arguments_json,
                            }
                        })

                        self.logger.debug(f"Successfully parsed function call: {function_name} with args: {arguments_json}")

                    except Exception as e:
                        self.logger.error(f"Failed to parse function call: {e}")
                        # Log the structure for debugging
                        self.logger.debug(f"Function call part type: {type(function_call_part)}")
                        self.logger.debug(f"Function call part attributes: {dir(function_call_part)[:10]}...")  # Truncate long list

            # Fallback: Check for function calls in candidates (alternative format)
            if not tool_calls and response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and
                    candidate.content and
                    hasattr(candidate.content, 'parts') and
                    candidate.content.parts):

                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            try:
                                func_call = part.function_call
                                # Try to extract arguments from part-level function call
                                if hasattr(func_call, 'args') and func_call.args:
                                    args_dict = dict(func_call.args)
                                    arguments_json = json.dumps(args_dict)
                                else:
                                    arguments_json = "{}"

                                tool_calls.append({
                                    "type": "custom",
                                    "function": {
                                        "name": func_call.name if hasattr(func_call, 'name') else 'unknown_function',
                                        "arguments": arguments_json,
                                    }
                                })

                                self.logger.debug(f"Parsed function call from candidate part: {func_call.name}")

                            except Exception as e:
                                self.logger.error(f"Failed to parse function call from candidate part: {e}")

            # Check for text content - handle error when function call is present
            try:
                if hasattr(response, 'text') and response.text:
                    text_content = response.text
            except ValueError as e:
                # When function calls are present, accessing response.text throws ValueError
                self.logger.debug(f"Could not access response.text (function call present): {e}")
                text_content = ""

            # If no text content from response.text, try candidates
            if not text_content and response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and
                    candidate.content and
                    hasattr(candidate.content, 'parts') and
                    candidate.content.parts):

                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_content = part.text
                            break

            # Add tool calls if present
            if tool_calls:
                result["tool_calls"] = tool_calls

            # Add text content if present
            if text_content:
                result["output_text"] = text_content
                # Also add in choices format for compatibility
                result["choices"] = [{
                    "message": {
                        "content": text_content,
                        "tool_calls": tool_calls if tool_calls else None
                    }
                }]

            return result

        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {e}")
            raise AIServiceError(f"Failed to call Gemini API: {e}")

    def _format_prompt(self, prompt_key: str, context: Dict[str, Any]) -> List[Dict]:
        """Construct messages array from prompt templates and context (plain text content)."""
        cfg = self.prompts.get(prompt_key)
        if not cfg:
            # Fallback minimal prompt for generic interactions/tests
            master_persona = self.prompts.get("master_persona", "")
            system_text = master_persona or "You are a helpful assistant."
            user_text = json.dumps(context, ensure_ascii=False) if context else ""
            return [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]

        master_persona = self.prompts.get("master_persona", "")

        # Choose fields based on prompt type
        system_part = cfg.get("system") or cfg.get("system_base") or ""
        template = cfg.get("template") or cfg.get("analysis_template") or ""

        system_text = (master_persona + "\n" + system_part).strip()

        try:
            user_text = template.format(**context)
        except Exception:
            # If format fails (e.g., complex structures), fall back to JSON injection
            user_text = template + "\n\nContext JSON:\n" + json.dumps(context, ensure_ascii=False)

        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

    def _extract_text_content(self, resp: Dict[str, Any]) -> str:
        """Extract assistant text from Gemini response format."""
        if not isinstance(resp, dict):
            return ""
        
        # Log the full response structure for debugging
        self.logger.debug(f"Full Gemini response structure: {json.dumps(resp, indent=2)[:1000]}...")
        
        # Direct output_text field
        if "output_text" in resp and isinstance(resp["output_text"], str):
            self.logger.debug(f"Found output_text field: {resp['output_text'][:200]}...")
            return resp["output_text"]
        
        # JSON output field
        if "output_json" in resp:
            # If it's already a string, return it
            if isinstance(resp["output_json"], str):
                self.logger.debug(f"Found output_json string: {resp['output_json'][:200]}...")
                return resp["output_json"]
            # If it's a dict/object, stringify it
            elif isinstance(resp["output_json"], (dict, list)):
                json_str = json.dumps(resp["output_json"], ensure_ascii=False)
                self.logger.debug(f"Found output_json object, stringified: {json_str[:200]}...")
                return json_str
        
        # Standard message format: choices[0].message.content
        try:
            content = resp["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            # Some shapes: content may be list of blocks with {type: text, text: ...}
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text_val = content[0].get("text") or content[0].get("content")
                if isinstance(text_val, str):
                    return text_val
        except Exception:
            pass
        
        # Alternative output array format
        try:
            outputs = resp.get("output") or []
            if isinstance(outputs, list) and outputs:
                self.logger.debug(f"Found output array with {len(outputs)} items")
                for i, output_item in enumerate(outputs):
                    if isinstance(output_item, dict):
                        self.logger.debug(f"Output item {i} type: {output_item.get('type')}")
                        # Handle message type outputs
                        if output_item.get("type") == "message":
                            content_arr = output_item.get("content") or []
                            for block in content_arr:
                                if block.get("type") in ("output_text", "text") and isinstance(block.get("text"), str):
                                    self.logger.debug(f"Found text in message block: {block['text'][:200]}...")
                                    return block["text"]
                        # Handle direct content arrays
                        elif output_item.get("content"):
                            content_arr = output_item.get("content") or []
                            for block in content_arr:
                                if block.get("type") in ("output_text", "text") and isinstance(block.get("text"), str):
                                    self.logger.debug(f"Found text in content block: {block['text'][:200]}...")
                                    return block["text"]
                    elif isinstance(output_item, str):
                        self.logger.debug(f"Found direct string in output: {output_item[:200]}...")
                        return output_item
        except Exception as e:
            self.logger.debug(f"Error parsing output array: {e}")
            pass

        # Recursive search for first string value under a 'text' key
        try:
            def find_text(node: Any) -> str:
                if isinstance(node, dict):
                    val = node.get("text")
                    if isinstance(val, str):
                        return val
                    for v in node.values():
                        t = find_text(v)
                        if t:
                            return t
                elif isinstance(node, list):
                    for it in node:
                        t = find_text(it)
                        if t:
                            return t
                return ""
            tval = find_text(resp)
            if isinstance(tval, str) and tval:
                return tval
        except Exception:
            pass
        
        return ""

    def _extract_tool_call(self, resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract tool call from Google GenAI SDK response.
        The _call_gemini method already correctly extracts function calls and puts them in resp["tool_calls"].
        """
        # Log the response for debugging
        self.logger.debug(f"Extracting tool call from response...")

        # Check for tool_calls in our properly structured response from _call_gemini
        if "tool_calls" in resp and isinstance(resp["tool_calls"], list) and resp["tool_calls"]:
            tool_call = resp["tool_calls"][0]
            self.logger.debug(f"Found tool call: {tool_call.get('function', {}).get('name', 'unknown')}")
            return tool_call

        # Fallback: Check choices format (compatibility)
        try:
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    self.logger.debug(f"Found tool call in choices: {calls[0].get('function', {}).get('name', 'unknown')}")
                    return calls[0]
        except Exception as e:
            self.logger.debug(f"Error checking choices format: {e}")

        self.logger.debug("No tool calls found in response")
        return None

    async def generate_subject_line(self, newsletter_content: Dict[str, Any]) -> str:
        """
        Generate a simple one-sentence summary of today's forecast as the email subject line.

        Args:
            newsletter_content: Complete newsletter content (all sections, headlines, summaries)

        Returns:
            A one-sentence summary of today's forecast
        """
        self.logger.info(f"📧 Generating AI subject line from complete newsletter content")

        # Prepare simplified content for the AI
        simplified_content = {}
        if newsletter_content:
            for section_name, section_data in newsletter_content.items():
                if isinstance(section_data, list) and section_data:
                    # Extract just the headlines for each section
                    headlines = [item.get('headline', '') for item in section_data[:3] if item.get('headline')]
                    if headlines:
                        simplified_content[section_name] = headlines

        context = {
            "newsletter_summary": json.dumps(simplified_content, ensure_ascii=False)
        }

        try:
            messages = self._format_prompt("gpt5_subject_line", context)

            # Use tool-calling for reliable extraction
            function = types.FunctionDeclaration(
                name="return_subject",
                description="Return the email subject line summary",
                parameters=types.Schema(
                    type='OBJECT',
                    properties={
                        'subject': types.Schema(
                            type='STRING',
                            description='The email subject line'
                        )
                    },
                    required=['subject']
                )
            )
            tools = [types.Tool(function_declarations=[function])]
            tool_choice = {"type": "tool", "name": "return_subject"}

            resp = await self._call_gemini(messages=messages, max_tokens=1024, tools=tools, tool_choice=tool_choice)

            # Extract subject from tool call
            tool = self._extract_tool_call(resp)
            if tool:
                try:
                    args = json.loads(tool["function"]["arguments"]) or {}
                    subject = args.get("subject", "").strip()

                    # No character limits - just ensure we got something
                    if subject:
                        self.logger.info(f"✅ Generated AI subject line: {subject}")
                        return subject

                except Exception as e:
                    self.logger.error(f"Failed to parse subject tool arguments: {e}")

            # Retry without tool-calling using a simplified generation, then sanitize
            self.logger.warning("⚠️ AI subject tool-call failed; retrying with simplified prompt")
            resp2 = await self._call_gemini(messages=messages, max_tokens=512)
            candidate = self._extract_text_content(resp2)
            if candidate:
                subject = candidate.strip().splitlines()[0].strip().strip('"')
                # Trim to ~80 chars max to keep concise
                if len(subject) > 80:
                    subject = subject[:80].rstrip()
                if subject:
                    self.logger.info(f"✅ Generated AI subject (retry): {subject}")
                    return subject

            # Final fallback
            self.logger.warning("❌ AI subject generation failed, using fallback")
            return "Today's Fourier Forecast - Your Daily News Summary"

        except Exception as e:
            self.logger.error(f"💥 Subject line generation failed: {e}")
            return "Today's Fourier Forecast - Your Daily News Summary"

    async def generate_morning_greeting(self, date: datetime, newsletter_content: Optional[Dict[str, Any]] = None, golden_thread: Optional[str] = None) -> str:
        """
        Generate an AI-powered inspirational morning greeting that's contextually relevant to the day's content.
        
        Args:
            date: Date for the newsletter
            newsletter_content: Complete newsletter content structure (all sections, headlines, summaries)
            golden_thread: The golden thread connecting today's stories
            
        Returns:
            Inspirational greeting string
        """
        date_str = date.strftime("%A, %B %d, %Y")
        
        self.logger.info(f"🌅 Generating morning greeting for {date_str}")
        self.logger.debug(f"Newsletter content provided: {bool(newsletter_content)}")
        self.logger.debug(f"Golden thread provided: {bool(golden_thread)}")
        
        # Build context with simplified newsletter content (similar to subject line)
        simplified_content = {}
        if newsletter_content:
            for section_name, section_data in newsletter_content.items():
                if isinstance(section_data, list) and section_data:
                    # Extract just the headlines for each section (first 2 items to keep context manageable)
                    headlines = [item.get('headline', '') for item in section_data[:2] if item.get('headline')]
                    if headlines:
                        simplified_content[section_name] = headlines

        context = {
            "date_str": date_str,
            "newsletter_content_json": json.dumps(simplified_content, ensure_ascii=False),
            "golden_thread": golden_thread or "",
            "has_content": bool(newsletter_content)
        }
        
        self.logger.debug(f"Context keys: {list(context.keys())}")
        
        try:
            messages = self._format_prompt("gpt5_morning_greeting", context)
            self.logger.debug(f"Formatted prompt for greeting generation: {len(messages)} messages")
            
            # Use tool-calling pattern for more reliable response extraction
            function = types.FunctionDeclaration(
                name="return_greeting",
                description="Return the morning greeting",
                parameters=types.Schema(
                    type='OBJECT',
                    properties={
                        'greeting': types.Schema(
                            type='STRING',
                            description='The morning greeting'
                        )
                    },
                    required=['greeting']
                )
            )
            tools = [types.Tool(function_declarations=[function])]
            tool_choice = {"type": "tool", "name": "return_greeting"}

            resp = await self._call_gemini(messages=messages, max_tokens=1024, tools=tools, tool_choice=tool_choice)
            self.logger.debug(f"GPT-5 response received: {type(resp)}")

            # Debug: Log the actual response structure for greeting
            if 'choices' in resp and resp['choices']:
                choice = resp['choices'][0]
                if 'message' in choice:
                    content = choice['message'].get('content', 'No content')
                    self.logger.debug(f"Greeting response content: {content[:200]}...")

            # Extract greeting from tool call
            tool = self._extract_tool_call(resp)
            greeting = ""
            if tool:
                try:
                    args = json.loads(tool["function"]["arguments"]) or {}
                    greeting = args.get("greeting", "").strip()
                    self.logger.debug(f"Extracted greeting from tool: '{greeting}' (length: {len(greeting)})")
                except Exception as e:
                    self.logger.error(f"Failed to parse greeting tool arguments: {e}")
                    greeting = ""
            else:
                # Retry once if tool extraction failed
                self.logger.warning("No tool call found, retrying...")
                resp = await self._call_gemini(messages=messages, max_tokens=1024, tools=tools, tool_choice=tool_choice)
                tool = self._extract_tool_call(resp)
                if tool:
                    try:
                        args = json.loads(tool["function"]["arguments"]) or {}
                        greeting = args.get("greeting", "").strip()
                        self.logger.debug(f"Extracted greeting from retry: '{greeting}'")
                    except Exception as e:
                        self.logger.error(f"Failed to parse greeting on retry: {e}")
                        greeting = ""
                else:
                    self.logger.error("Failed to extract greeting even after retry")
            
            # Validate the greeting - ensure it starts appropriately and isn't empty
            if greeting and len(greeting) > 10 and any(word in greeting.lower() for word in ['good morning', 'hello', 'rise', 'greetings']):
                # Ensure it's under 200 characters (with safety margin)
                greeting_len = len(greeting)
                if greeting_len <= 200:
                    self.logger.info(f"✅ Generated contextual greeting ({greeting_len} chars): {greeting}")
                    return greeting
                else:
                    # Truncate gracefully at sentence boundary
                    max_len = 200
                    # Look for sentence endings, but not the first "Good morning!" 
                    # Find all periods and exclamation marks after position 20
                    sentence_ends = []
                    for i, char in enumerate(greeting[:max_len]):
                        if i > 20 and char in '.!':  # Skip the initial greeting
                            sentence_ends.append(i)
                    
                    if sentence_ends:
                        # Use the last sentence ending before the limit
                        truncate_at = sentence_ends[-1] + 1
                        truncated = greeting[:truncate_at]
                        self.logger.warning(f"⚠️ Truncated greeting from {greeting_len} to {len(truncated)} chars")
                        return truncated
                    else:
                        # No good sentence boundary found - truncate at word boundary
                        if len(greeting) > max_len:
                            # Find last space before limit
                            last_space = greeting.rfind(' ', 0, max_len - 3)
                            if last_space > 20:  # Ensure we keep more than just "Good morning!"
                                truncated = greeting[:last_space] + "..."
                            else:
                                truncated = greeting[:max_len - 3] + "..."
                            self.logger.warning(f"⚠️ Hard truncated greeting from {greeting_len} to {len(truncated)} chars")
                            return truncated
                        else:
                            return greeting  # Should not happen since we checked length
            else:
                self.logger.warning(f"❌ AI greeting validation failed. Greeting: '{greeting}'")
                        
            # Fallback if AI response is not suitable
            fallback = f"Good morning, Ignacio! Ready to discover today's signal from the noise? — {date_str}"
            self.logger.warning(f"🔄 Using fallback greeting: {fallback}")
            return fallback
            
        except Exception as e:
            # Fallback greeting if AI service fails (following NO FALLBACKS philosophy by logging but providing minimal backup)
            self.logger.error(f"💥 Greeting generation failed: {e}")
            fallback = f"Good morning, Ignacio! — {date_str}"
            self.logger.warning(f"🔄 Using minimal fallback: {fallback}")
            return fallback

    async def generate_section_preface(
        self,
        section_name: str,
        all_articles: List[Dict[str, Any]],
        editorial_voice: str
    ) -> str:
        """
        Generate a comprehensive preface/overview for a newsletter section.
        
        This method analyzes ALL articles gathered for a section (not just the selected ones)
        to provide readers with a broader context of what's happening in that domain.
        
        Args:
            section_name: Name of the section (e.g., "Tech & Science", "Business")
            all_articles: List of ALL articles gathered for this section, each containing:
                - headline: Article title
                - summary: Article summary or content
                - source: Source publication
            editorial_voice: The editorial voice to use (e.g., "poet", "medici", "historical")
            
        Returns:
            A one-paragraph preface (50-75 words) summarizing the overall trends and themes
        """
        try:
            # Build context from all articles
            articles_context = []
            for article in all_articles[:20]:  # Cap at 20 for API limits
                articles_context.append({
                    "headline": article.get("headline", ""),
                    "summary": article.get("summary", "")[:200],  # Truncate long summaries
                    "source": article.get("source", "")
                })
            
            # Define voice-specific instructions
            voice_instructions = {
                "historical": "Write with gravitas, capturing the historical significance of current events.",
                "medici": "View commerce through a cultural lens, seeing business as a force shaping society.",
                "poet": "Translate technical and scientific advances into accessible, poetic language.",
                "apprentice": "Extract practical wisdom with humility, learning from builders and innovators.",
                "civic": "Provide thoughtful civic awareness, weighing institutions and consequences.",
                "neighbor": "Connect at a human scale, making global events relevant to community.",
                "wanderer": "Embrace intellectual serendipity with cultured curiosity."
            }
            
            voice_guide = voice_instructions.get(editorial_voice, 
                "Provide an insightful overview of the section's themes.")
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a brilliant newsletter editor creating section prefaces. "
                        "Your task is to synthesize multiple articles into a cohesive overview "
                        "that gives readers context about what's happening in this domain. "
                        f"{voice_guide}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Section: {section_name}\n"
                        f"Number of articles gathered: {len(all_articles)}\n\n"
                        "Articles overview:\n" +
                        "\n".join([
                            f"- {art['headline']}: {art['summary']}"
                            for art in articles_context
                        ]) +
                        "\n\nWrite a single paragraph (50-75 words) that captures the overall "
                        "themes, trends, and significance of what's happening in this section. "
                        "Don't just list articles—synthesize them into a cohesive narrative "
                        "that provides context and meaning. Focus on the bigger picture and "
                        "common threads across all the stories gathered (not just the ones shown)."
                    )
                }
            ]
            
            # Call Gemini for the preface
            response = await self._call_gemini(
                messages=messages,
                max_tokens=2000
            )
            
            preface = self._extract_text_content(response)
            
            # Strip markdown formatting for plain text output
            import re
            # Remove markdown bold (**text** or __text__)
            preface = re.sub(r'\*\*([^*]+)\*\*', r'\1', preface)
            preface = re.sub(r'__([^_]+)__', r'\1', preface)
            # Remove markdown italic (*text* or _text_)
            preface = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'\1', preface)
            preface = re.sub(r'(?<!_)_(?!_)([^_]+)_(?!_)', r'\1', preface)
            # Remove markdown headers (# Header)
            preface = re.sub(r'^#+\s+', '', preface, flags=re.MULTILINE)
            # Remove markdown links [text](url)
            preface = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', preface)
            # Remove markdown code blocks `code`
            preface = re.sub(r'`([^`]+)`', r'\1', preface)
            
            # Ensure it's roughly the right length
            words = preface.split()
            if len(words) > 85:
                # Truncate to roughly 75 words
                preface = " ".join(words[:75]) + "..."
            elif len(words) < 40:
                # Too short, add a note about the variety
                preface += f" Today's {section_name} section captures {len(all_articles)} stories shaping our understanding."
            
            return preface
            
        except Exception as e:
            self.logger.error(f"Failed to generate section preface for {section_name}: {e}")
            # Fallback preface
            return (
                f"Today's {section_name} section brings together {len(all_articles)} "
                f"carefully curated stories, offering insights into the latest developments "
                f"and emerging trends shaping this dynamic field."
            )

    async def interact_with_gpt(self, prompt_key: str, context: Dict[str, Any], max_tokens: int = 4096) -> AIResponse:
        messages = self._format_prompt(prompt_key, context)

        # Temperature not used with tool-calling flow
        temperature = 0.0

        start = asyncio.get_event_loop().time()
        try:
            resp = await self._call_gemini(messages=messages, max_tokens=max_tokens)
            end = asyncio.get_event_loop().time()
            tokens_used = 0
            try:
                tokens_used = int((resp.get("usage") or {}).get("total_tokens") or 0)
            except Exception:
                tokens_used = 0
            content = self._extract_text_content(resp)
            return AIResponse(
                content=content,
                prompt_key=prompt_key,
                model=self.model,
                tokens_used=tokens_used,
                response_time_ms=(end - start) * 1000.0,
                temperature=temperature,
                success=True,
            )
        except AIServiceError as e:
            end = asyncio.get_event_loop().time()
            return AIResponse(
                content=None,
                prompt_key=prompt_key,
                model=self.model,
                tokens_used=0,
                response_time_ms=(end - start) * 1000.0,
                temperature=temperature,
                success=False,
                error_message=str(e),
            )

    def _assess_content_quality(self, headline: str, content: str, source: str) -> Dict[str, Any]:
        """
        Assess the quality and characteristics of available content for summarization.

        Returns:
            Dict containing quality metrics and categorization
        """
        headline_len = len(headline)
        content_len = len(content.strip())

        # Calculate content-to-headline ratio
        content_to_headline_ratio = content_len / max(headline_len, 1)

        # Determine content quality category
        if content_len == 0:
            quality_category = "no_content"
        elif content_len < 50:
            quality_category = "minimal_content"
        elif content_len < 200:
            quality_category = "short_content"
        elif content_len < 500:
            quality_category = "medium_content"
        else:
            quality_category = "rich_content"

        # Check for content duplication with headline
        content_lower = content.lower().strip()
        headline_lower = headline.lower().strip()
        is_content_duplicate = content_lower == headline_lower or headline_lower in content_lower[:100]

        # Assess source reliability (could be enhanced with source authority scoring)
        source_reliability = "unknown"
        high_quality_sources = ["bbc", "reuters", "ap", "bloomberg", "ft", "wsj", "nature", "science"]
        if any(trusted in source.lower() for trusted in high_quality_sources):
            source_reliability = "high"
        elif any(reliable in source.lower() for reliable in ["cnn", "guardian", "times", "post"]):
            source_reliability = "medium"

        return {
            "headline_length": headline_len,
            "content_length": content_len,
            "content_to_headline_ratio": content_to_headline_ratio,
            "quality_category": quality_category,
            "is_content_duplicate": is_content_duplicate,
            "source_reliability": source_reliability,
            "has_meaningful_content": content_len > 50 and not is_content_duplicate
        }

    def _determine_generation_mode(self, headline: str, content: str, content_quality: Dict[str, Any]) -> Tuple[str, str]:
        """
        Determine the best generation mode and process content accordingly.

        Returns:
            Tuple of (generation_mode, processed_content)
        """
        content = content.strip()

        # Determine generation mode based on content quality
        if content_quality["has_meaningful_content"]:
            if content_quality["content_length"] > 500:
                mode = "content_based_rich"
            elif content_quality["content_length"] > 200:
                mode = "content_based_medium"
            else:
                mode = "content_based_short"
            processed_content = content
        elif content_quality["content_length"] > 0 and not content_quality["is_content_duplicate"]:
            mode = "minimal_content"
            processed_content = content
        else:
            mode = "headline_based"
            processed_content = ""  # Clear content to focus on headline

        return mode, processed_content

    def _generate_mode_specific_prompts(self, generation_mode: str, headline: str, content: str,
                                      source: str, url: str, section: str, voice: str, prompt_hint: str) -> Tuple[str, str]:
        """
        Generate prompts optimized for the specific generation mode.
        """
        if generation_mode == "content_based_rich":
            system_text = (
                f"You are the Renaissance {voice} editor. You have rich article content. "
                f"Write exactly one brilliant, factual sentence (25-40 words) that captures the "
                f"most important insight from this {section} story. Extract the core meaning, "
                f"not just surface details. No markdown, no bullets, no code. "
                f"IMPORTANT: You must use the return_summary tool with a summary and confidence score."
            )
            user_text = (
                f"Section: {section}\nVoice: {voice}\nSource: {source}\nURL: {url}\n"
                f"Headline: {headline}\n\nFull Content:\n{content}\n\n"
                f"Extract the most significant insight from this content. Look beyond the headline. "
                f"Constraints: sentence=true, words=25-40. {prompt_hint}"
            )
        elif generation_mode in ["content_based_medium", "content_based_short"]:
            system_text = (
                f"You are the Renaissance {voice} editor. You have some article content. "
                f"Write exactly one brilliant, factual sentence (25-40 words) that captures "
                f"the essence of this {section} story using the available content. "
                f"No markdown, no bullets, no code. Use clear, vivid language. "
                f"IMPORTANT: You must use the return_summary tool with a summary and confidence score."
            )
            user_text = (
                f"Section: {section}\nVoice: {voice}\nSource: {source}\nURL: {url}\n"
                f"Headline: {headline}\n\nContent:\n{content}\n\n"
                f"Summarize the key insight from this content. "
                f"Constraints: sentence=true, words=25-40. {prompt_hint}"
            )
        elif generation_mode == "minimal_content":
            system_text = (
                f"You are the Renaissance {voice} editor. You have minimal content beyond the headline. "
                f"Write exactly one brilliant sentence (25-40 words) that combines the headline "
                f"with the additional context to explain what this {section} story means. "
                f"IMPORTANT: You must use the return_summary tool with a summary and confidence score."
            )
            user_text = (
                f"Section: {section}\nVoice: {voice}\nSource: {source}\n"
                f"Headline: {headline}\n\nAdditional Context: {content}\n\n"
                f"Combine the headline with this context to create an insightful summary. "
                f"Constraints: sentence=true, words=25-40. {prompt_hint}"
            )
        else:  # headline_based
            system_text = (
                f"You are the Renaissance {voice} editor. From this headline alone, write exactly one brilliant, "
                f"contextual sentence (25-40 words) that expands on what this story likely means or why it matters. "
                f"Be insightful and specific to the {section} domain. No markdown, no bullets. "
                f"IMPORTANT: You must use the return_summary tool with a summary and confidence score."
            )
            user_text = (
                f"Section: {section}\nVoice: {voice}\nHeadline: {headline}\nSource: {source}\n"
                f"Generate an insightful summary from this headline that explains its significance or context. "
                f"Make it engaging and informative as if you understand the story behind the headline. "
                f"Constraints: sentence=true, words=25-40. {prompt_hint}"
            )

        return system_text, user_text

    def _calculate_final_confidence(self, ai_confidence: float, generation_mode: str,
                                  content_quality: Dict[str, Any], summary_word_count: int) -> float:
        """
        Calculate final confidence score based on generation mode and content quality.
        """
        base_confidence = ai_confidence

        # Adjust based on generation mode
        mode_multipliers = {
            "content_based_rich": 1.0,      # Highest confidence
            "content_based_medium": 0.9,    # High confidence
            "content_based_short": 0.8,     # Good confidence
            "minimal_content": 0.7,         # Moderate confidence
            "headline_based": 0.6           # Lower confidence
        }

        mode_adjusted = base_confidence * mode_multipliers.get(generation_mode, 0.5)

        # Boost for high-quality sources
        source_boost = 0.1 if content_quality["source_reliability"] == "high" else 0.0

        # Boost for good summary length (20+ words indicates thoughtful expansion)
        length_boost = 0.1 if summary_word_count >= 20 else 0.0

        # Penalty for content duplication
        duplication_penalty = -0.2 if content_quality["is_content_duplicate"] else 0.0

        final_confidence = min(1.0, max(0.1, mode_adjusted + source_boost + length_boost + duplication_penalty))

        return final_confidence

    def _validate_summary_quality(self, summary: str, headline: str, generation_mode: str) -> Dict[str, Any]:
        """
        Validate the quality of the generated summary and detect potential issues.
        """
        summary_lower = summary.lower().strip()
        headline_lower = headline.lower().strip()

        # Check if summary is just a paraphrase of the headline
        summary_words = set(summary_lower.split())
        headline_words = set(headline_lower.split())

        # Calculate word overlap
        common_words = summary_words & headline_words
        headline_coverage = len(common_words) / max(len(headline_words), 1)

        is_headline_paraphrase = headline_coverage > 0.7 or summary_lower.startswith(headline_lower[:20])

        # Determine content utilization
        if generation_mode.startswith("content_based"):
            content_utilization = "content_based"
        elif generation_mode == "minimal_content":
            content_utilization = "minimal_content"
        else:
            content_utilization = "headline_only"

        # Calculate quality score
        quality_score = 0.8  # Start with good score

        # Deduct for headline paraphrasing
        if is_headline_paraphrase:
            quality_score -= 0.3

        # Deduct for headline-only generation
        if content_utilization == "headline_only":
            quality_score -= 0.2

        # Boost for content-based generation
        if content_utilization == "content_based":
            quality_score += 0.2

        quality_score = max(0.0, min(1.0, quality_score))

        # Identify issues
        issues = []
        if is_headline_paraphrase:
            issues.append("headline_paraphrase")
        if content_utilization == "headline_only":
            issues.append("no_content_used")
        if len(summary.split()) < 15:
            issues.append("too_short")

        return {
            "quality_score": quality_score,
            "is_headline_paraphrase": is_headline_paraphrase,
            "headline_coverage": headline_coverage,
            "content_utilization": content_utilization,
            "issues": issues,
            "word_count": len(summary.split())
        }

    def _log_content_quality_stats(self, content_quality: Dict[str, Any], generation_mode: str,
                                 source: str, section: str) -> None:
        """
        Log content quality statistics for tracking and analysis.
        """
        self.logger.info(f"📈 CONTENT_QUALITY_STATS: "
                        f"source={source[:20]} "
                        f"section={section} "
                        f"mode={generation_mode} "
                        f"category={content_quality['quality_category']} "
                        f"content_len={content_quality['content_length']} "
                        f"ratio={content_quality['content_to_headline_ratio']:.1f} "
                        f"reliable_source={content_quality['source_reliability']} "
                        f"has_content={content_quality['has_meaningful_content']}")


