# <div align="center">
  <img src="assets/fourier_forecast_logo_butterfly.png" alt="Fourier Forecast Logo" width="150" height="150">

  # **Fourier Forecast**

  *Multi-AI Orchestration System for Signal Extraction from Information Noise*

  [![Python](https://img.shields.io/badge/Python-3.11+-3B82F6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-Google-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
  [![Exa](https://img.shields.io/badge/Exa_Websets-Neural_Search-6366F1?style=for-the-badge)](https://exa.ai/)
  [![AWS](https://img.shields.io/badge/AWS-Production-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
  [![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)
</div>

---

## **Project Overview**

Fourier Forecast is an advanced AI orchestration system that demonstrates sophisticated multi-model coordination, intelligent content curation, and production-grade reliability. Built as a personal daily newsletter platform, it showcases enterprise-level architecture patterns including adaptive retry strategies, multi-layer deduplication, semantic analysis, and cross-domain pattern detection.

**The Challenge:** Transform the overwhelming daily information deluge into actionable intelligence by identifying meaningful patterns across disparate domains—analogous to how Fourier transforms decompose complex signals into constituent frequencies.

**The Solution:** A 7-stage AI pipeline orchestrating Google Gemini 2.0 Flash and Anthropic Claude with Exa's neural search API, implementing Renaissance-inspired breadth-first curation across breaking news, business, technology, research, politics, and humanities.

**The Result:** A production system delivering 21 curated items daily at 5:30 AM ET with reliability, featuring cross-article pattern synthesis ("Golden Thread"), adaptive temporal ranking, and guaranteed section delivery through intelligent fallback mechanisms.

## **Key Technical Achievements**

### **Multi-AI Orchestration Architecture**
- **Dual-Model Strategy**: Coordinates Google Gemini 2.0 Flash (editorial decisions, ranking, synthesis) with Anthropic Claude (fallback reasoning) for optimal cost-performance balance
- **Intelligent Model Selection**: Routes tasks based on complexity—Gemini for high-throughput ranking (21 items/day), Claude for nuanced pattern detection
- **Graceful Degradation**: Automatic fallback from AI ranking → heuristic scoring → emergency defaults ensures 100% uptime

### **Advanced Content Intelligence**

**7-Axis Renaissance Ranking System**
- Implements weighted multi-dimensional scoring: Temporal Impact (20%), Intellectual Novelty (18%), Signal Clarity (16%), Source Authority (15%), Transformative Potential (12%), Renaissance Breadth (10%), Actionable Wisdom (9%)
- **Adaptive Temporal Decay**: Section-specific recency weighting (Breaking=2d, Business=3d, Tech=4d, Politics=2d, Misc=7d, Papers=30d)
- **Historical Context Integration**: Analyzes cross-day patterns to boost genuinely novel content

**4-Layer Deduplication Pipeline**
1. **URL Exact Match** (30-day window, O(1) lookup)
2. **Title Hash Similarity** (7-day window, Levenshtein distance)
3. **Semantic Embedding** (Voyage AI, cosine similarity < 0.15 threshold)
4. **AI Editorial Judgment** (Gemini-powered borderline case resolution)

**Golden Thread Pattern Synthesis**
- **4-Tier Retry Strategy**: Progressive confidence thresholds (0.80 → 0.70 → 0.60 → 0.50) with fresh generation fallback
- **Cross-Domain Connection Detection**: Identifies non-obvious patterns across breaking news, research, and humanities
- **Concrete Citation Requirement**: Must reference specific headlines with proper nouns and quantitative data

### **Production-Grade Reliability**

**Section Guarantee Mechanism (3-Tier Retry)**
- **Tier 1 (AI Ranking)**: Gemini 2.0 Flash with 7-axis scoring (95% success rate)
- **Tier 2 (Heuristic Ranking)**: Source authority + recency heuristics (4% usage)
- **Tier 3 (Emergency Ranking)**: Default scores ensuring section never disappears (<1% usage)

**Exa Websets Integration with Enrichments**
- **Sequential Search Execution**: Respects Exa API constraints (no concurrent searches per webset)
- **Content Enrichment**: Extracts 200-300 word summaries during webset creation using Exa's enrichment API
- **Intelligent Cleanup**: Automatic webset deletion post-retrieval to prevent resource leaks
- **Section-Aware Date Estimation**: Fallback date logic when publishers omit metadata

### **Scalability & Performance**
- **Asynchronous Pipeline**: Non-blocking I/O for API calls (Exa, Gemini, Voyage AI)
- **Smart Caching**: SQLite-based persistence with adaptive TTL (7-30 days based on content quality scores)
- **Timeout Hierarchy**: Three-level timeout strategy (API: 180s, Decision: 240s, Section: 600s)
- **AWS Production Deployment**: Systemd timer automation with auto-restart on failure

## **Technology Stack**

### **AI & Machine Learning**
- **Google Gemini 2.0 Flash** - Primary AI for editorial ranking, summarization, and synthesis
- **Anthropic Claude Sonnet** - Fallback reasoning and nuanced pattern detection
- **Voyage AI** - Semantic embeddings for content similarity analysis (voyage-large-2-instruct)

### **Content Discovery & APIs**
- **Exa Websets API** - Neural search with enrichments for content-based summarization
- **arXiv API** - Recent academic papers (physics, CS, math, biology)
- **Semantic Scholar API** - High-quality research discovery with citation metrics
- **Nature.com** - Peer-reviewed research articles

### **Backend & Infrastructure**
- **Python 3.11+** - Core language with async/await patterns
- **SQLite** - Persistent caching and deduplication storage
- **Jinja2** - Responsive HTML email templating
- **AWS EC2** - Production hosting (Ubuntu 22.04 LTS)
- **systemd** - Service management and daily scheduling (5:30 AM ET)

### **Key Libraries**
- `exa-py` - Exa Websets SDK with enrichment support
- `google-generativeai` - Gemini API client
- `anthropic` - Claude API client
- `voyageai` - Embedding generation
- `httpx` - Async HTTP client
- `pydantic` - Data validation and type safety

---

## **Quick Start**

### Prerequisites

- Python 3.11 or higher
- Required API Keys:
  - **Google Gemini** - Core AI reasoning ([Get free key](https://aistudio.google.com/app/apikey))
  - **Exa API** - Neural search ([Get key](https://dashboard.exa.ai/))
  - **Voyage AI** - Semantic embeddings ([Get key](https://www.voyageai.com/))
  - **SMTP credentials** - Email delivery (Gmail App Password recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/Nacho2027/Ignacio-s-Fourier-Forecast.git
cd Ignacio-s-Fourier-Forecast

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (see .env.example)
cp .env.example .env
# Edit .env with your API keys

# Run test pipeline (no email sent)
env PYTHONPATH=. python3 src/main.py --test
```

**Production Deployment**: Automated AWS deployment via `./deploy_production.sh` (see [For Developers](#for-developers) section)

## **Architecture Highlights**

### **7-Stage Pipeline Design**

The system implements a sophisticated multi-stage pipeline with intelligent fallback mechanisms at each stage:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: FETCH (Sequential Exa Websets + Academic APIs)        │
│  • 2 searches/section × 6 sections = 12 sequential API calls    │
│  • Enrichment creation during webset initialization             │
│  • Section-aware date estimation for missing metadata           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: DEDUPLICATE (4-Layer Strategy)                        │
│  • URL exact match (O(1) SQLite lookup, 30-day window)          │
│  • Title hash similarity (Levenshtein, 7-day window)            │
│  • Semantic embedding (Voyage AI, cosine < 0.15)                │
│  • AI editorial judgment (Gemini borderline resolution)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: RANK (3-Tier Retry with 7-Axis Scoring)               │
│  • Tier 1: Gemini 2.0 Flash (95% success, 7-axis weighted)      │
│  • Tier 2: Heuristic scoring (4% usage, source + recency)       │
│  • Tier 3: Emergency defaults (< 1%, ensures section appears)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: SUMMARIZE (Section-Specific Editorial Voices)         │
│  • Content-based generation (enrichment_summary → abstract)     │
│  • 25-40 word single-sentence summaries                         │
│  • Section prefaces (50-75 words, 7 editorial voices)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: SYNTHESIZE (4-Tier Golden Thread Retry)               │
│  • Progressive thresholds: 0.80 → 0.70 → 0.60 → 0.50            │
│  • Cross-domain pattern detection with concrete citations       │
│  • Fresh generation fallback if all tiers fail                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6: COMPILE (Responsive HTML + Plain Text)                │
│  • Jinja2 templating with inlined CSS                           │
│  • Subject line generation (50-70 chars)                        │
│  • Morning greeting (2 sentences, < 200 chars)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 7: SEND (SMTP Delivery + Cache Management)               │
│  • Adaptive cache expiry (7-30 days based on quality scores)    │
│  • Newsletter manifest storage for analytics                    │
│  • Delivery confirmation and error handling                     │
└─────────────────────────────────────────────────────────────────┘
```

**See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive technical documentation.**

## **Newsletter Curation Philosophy**

### **Renaissance Breadth Approach**

The system curates **21 items daily** across 6 domains to foster polymathic thinking:

- **Breaking News** (3): High-impact geopolitical developments
- **Business** (3): Industry-shaping moves, capital allocation signals
- **Tech & Science** (3): Genuine advances with methodological rigor
- **Research Papers** (5): arXiv recent + Nature + Semantic Scholar strong picks
- **Politics** (2): Institutional/policy substance with lasting effects
- **Miscellaneous** (5): Philosophy, psychology, history, architecture, health, literature

### **Signal-from-Noise Philosophy**

Inspired by Fourier analysis, the system decomposes the information deluge into fundamental "frequencies":

1. **Recurring Patterns**: Cross-domain themes that appear across disparate stories
2. **Emerging Trends**: Early signals of paradigm shifts before mainstream coverage
3. **Underlying Themes**: Structural forces driving surface-level events

The **Golden Thread** feature synthesizes these patterns into 2-3 sentence insights that connect seemingly unrelated stories, citing specific headlines with concrete data.

## **Design Decisions & Trade-offs**

### **Why Sequential Search Execution?**
Exa Websets API does not support concurrent searches on the same webset. The system executes 12 searches sequentially (2 per section × 6 sections), adding ~10-15 minutes to pipeline execution. This trade-off ensures API compliance and prevents rate limiting.

### **Why Gemini 2.0 Flash over GPT-4?**
- **Cost**: 10x cheaper than GPT-4 Turbo ($0.075/1M input tokens vs $0.75/1M)
- **Speed**: 2-3x faster response times for ranking 21 items
- **Quality**: Comparable editorial judgment for structured tasks
- **Fallback**: Claude Sonnet provides safety net for complex synthesis

### **Why 4-Layer Deduplication?**
Single-layer approaches miss edge cases:
- **URL matching** fails when same story published on multiple domains
- **Title hashing** fails when headlines are rewritten
- **Semantic embeddings** fail when content is substantially updated
- **AI judgment** provides final arbiter for borderline cases (e.g., "follow-up" vs "duplicate")

### **Why 3-Tier Ranking Retry?**
Ensures 100% section delivery even during AI API outages:
- **Tier 1 (AI)**: Optimal quality but dependent on external API
- **Tier 2 (Heuristic)**: Deterministic fallback using source authority + recency
- **Tier 3 (Emergency)**: Guarantees section appears with default scores

## **Performance Characteristics**

### **Latency Profile**
- **Stage 1 (Fetch)**: 10-15 minutes (12 sequential Exa searches + enrichments)
- **Stage 2 (Dedup)**: 30-60 seconds (embedding generation + SQLite lookups)
- **Stage 3 (Rank)**: 2-4 minutes (Gemini API calls for 21 items)
- **Stage 4 (Summarize)**: 1-2 minutes (section prefaces + item summaries)
- **Stage 5 (Synthesize)**: 30-90 seconds (Golden Thread generation)
- **Stage 6-7 (Compile & Send)**: 10-20 seconds (templating + SMTP)

**Total Pipeline**: 15-25 minutes end-to-end

### **Cost Analysis** (per newsletter)
- **Exa Websets**: ~$0.50-0.75 (12 searches + enrichments)
- **Gemini 2.0 Flash**: ~$0.05-0.10 (ranking + summarization)
- **Voyage AI**: ~$0.02-0.03 (embedding generation)
- **Claude Sonnet**: ~$0.01-0.02 (fallback usage)

**Total**: ~$0.60-0.90 per newsletter (~$18-27/month)

---

## **For Developers**

### **Running Tests**
```bash
# Test mode (no email sent)
env PYTHONPATH=. python3 src/main.py --test

# Production mode (send email)
env PYTHONPATH=. python3 src/main.py --once

# With logging
env PYTHONPATH=. python3 src/main.py --test 2>&1 | tee test_output.log
```

### **Production Deployment**
```bash
# Automated AWS deployment
./deploy_production.sh

# Manual deployment
ssh -i Fourier.pem ubuntu@18.221.135.150
cd /opt/fourier-forecast
sudo systemctl restart fourier-forecast
```

### **Monitoring**
```bash
# Check service status
ssh -i Fourier.pem ubuntu@18.221.135.150 "sudo systemctl status fourier-forecast"

# View logs (5:00-6:00 AM)
ssh -i Fourier.pem ubuntu@18.221.135.150 "sudo journalctl -u fourier-forecast --since '05:00' --until '06:00' --no-pager"

# Follow logs in real-time
ssh -i Fourier.pem ubuntu@18.221.135.150 "sudo journalctl -u fourier-forecast -f"
```

### **Cache Management**
```bash
# Clear local cache
rm -f cache.db

# Clear AWS cache
ssh -i Fourier.pem ubuntu@18.221.135.150 "cd /opt/fourier-forecast && rm -f cache.db"
```

### **Configuration Files**
- `config/exa_prompts.json` - Exa Websets search configurations
- `config/exa_settings.json` - Exa API settings
- `config/prompts_v2.yaml` - AI prompts (ranking, synthesis, summarization)
- `.env` - API keys and SMTP credentials

### **Key Timeouts**
- Gemini API: 180 seconds
- Decision timeout: 240 seconds
- Section timeout: 600 seconds (10 minutes)
- Pipeline timeout: 7200 seconds (120 minutes)

---

## **License**

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Fourier Forecast</strong> - Extracting Signal from Noise Through Multi-AI Orchestration

  <img src="assets/fourier_forecast_logo_butterfly.png" alt="Fourier Forecast" width="50" height="50">

  *Built with Python, Gemini 2.0 Flash, Exa Websets, and Claude Sonnet*
</div>