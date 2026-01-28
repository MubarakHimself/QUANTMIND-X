"""
Prompt templates for Analyst Agent.

This module contains all LangChain ChatPromptTemplate instances used
throughout the Analyst Agent workflow for concept extraction, knowledge
base search, TRD generation, and gap detection.

Author: Analyst Agent v1.0
"""

from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
# 1. EXTRACTION_PROMPT - Extract Trading Concepts
# =============================================================================

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert trading strategy analyst specializing in extracting structured information from unstructured trading content.

Your task is to analyze the provided transcript and extract trading concepts into a structured JSON format.

**Extraction Guidelines:**

1. **Entry Conditions**: Look for specific triggers that initiate trades
   - Price action patterns (breakouts, reversals, bounces)
   - Indicator crossovers or levels
   - Time-based entries
   - Session-specific entries

2. **Exit Conditions**: Identify how trades are closed
   - Take profit strategies (fixed pips, risk:reward ratio, resistance levels)
   - Stop loss placement (fixed pips, ATR-based, support/resistance)
   - Trailing stop methods
   - Time-based exits

3. **Filters**: Identify conditions that filter trades
   - Time filters (trading sessions, days to avoid)
   - Market conditions (trend, range, volatility)
   - Spread limits
   - News event filters

4. **Indicators**: List all technical indicators mentioned
   - Include settings/parameters if specified (e.g., "RSI period 14")
   - Note the purpose (entry signal, exit signal, trend filter)

5. **Position Sizing**: Extract risk management details
   - Risk per trade (percentage or fixed amount)
   - Position sizing method (fixed lots, risk-based, volatility-based)
   - Maximum drawdown limits

**Output Format:**

Return ONLY valid JSON. No markdown formatting, no explanations.

Example output:
```json
{{
  "strategy_name": "ORB Breakout Scalper",
  "overview": "Opening Range Breakout strategy that trades the first 30 minutes of London session",
  "entry_conditions": [
    "Price breaks above opening range high",
    "Breakout occurs with volume increase",
    "RSI is above 50"
  ],
  "exit_conditions": {{
    "take_profit": "1.5x risk (fixed reward ratio)",
    "stop_loss": "Below opening range low (ATR 1.5)",
    "trailing_stop": "Trail at breakeven after 1x risk reached",
    "time_exit": "Close at 11:00 AM London time"
  }},
  "filters": [
    "Trade only London session (8:00-11:00 AM)",
    "Avoid high-impact news events",
    "Minimum ATR of 0.0010 required",
    "Maximum spread 2 pips"
  ],
  "indicators": [
    {{"name": "RSI", "settings": "Period 14", "purpose": "Entry confirmation"}},
    {{"name": "ATR", "settings": "Period 14", "purpose": "Stop loss calculation"}},
    {{"name": "Volume", "settings": "N/A", "purpose": "Breakout confirmation"}}
  ],
  "position_sizing": {{
    "risk_per_trade": "0.5% of account",
    "method": "Risk-based on ATR stop loss",
    "max_drawdown": "Not specified"
  }},
  "mentioned_concepts": [
    "opening range breakout",
    "London session",
    "volume confirmation",
    "ATR-based stops"
  ]
}}
```

**Important Rules:**
- Extract ONLY what is explicitly mentioned in the text
- Do not make up or assume values not present
- If information is missing, use null or empty array
- Be specific with values (e.g., "14 pips" not "small stop loss")
- Preserve the original terminology used in the content"""),
    ("human", """Analyze the following trading content and extract structured information:

**Content Type:** {content_type}
**Keywords:** {keywords}

**Full Content:**
{content}

Extract and return ONLY valid JSON following the specified format.""")
])

# =============================================================================
# 2. MISSING_INFO_PROMPT (GAP_DETECTION_PROMPT) - Identify Missing Information
# =============================================================================

GAP_DETECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical requirements analyst specializing in trading strategy documentation.

Your task is to review extracted trading concepts and identify missing information required for a complete Technical Requirements Document (TRD).

**Required TRD Fields by Category:**

**CRITICAL** (Cannot implement without these):
- Entry trigger: Specific condition that initiates the trade
- Entry confirmation: Additional signal(s) validating the entry
- Take profit strategy: How profits are realized
- Stop loss strategy: How losses are limited
- Position sizing method: How trade size is determined

**IMPORTANT** (Significantly improves implementation):
- Entry timeframe: Chart timeframe for entries (e.g., M1, M5, H1)
- Indicator settings: Specific parameters for all indicators
- Time filters: Trading sessions, days to avoid
- Trailing stop: Method for locking in profits
- Risk per trade: Percentage or amount risked

**OPTIONAL** (Nice to have):
- Market condition filters: Volatility, spread, trend requirements
- Max drawdown limit: Daily or overall loss limits
- Max concurrent trades: Number of trades allowed simultaneously
- Time exit: Specific time to close positions

**Gap Analysis Process:**

1. Review each extracted concept
2. Compare against required fields checklist
3. Identify which fields are missing or incomplete
4. Categorize each gap (CRITICAL, IMPORTANT, OPTIONAL)
5. Provide a clear description and example for each gap

**Output Format:**

Return ONLY valid JSON:

```json
{{
  "has_gaps": true,
  "gap_summary": {{
    "critical": 2,
    "important": 3,
    "optional": 1
  }},
  "gaps": [
    {{
      "field_name": "entry_timeframe",
      "category": "IMPORTANT",
      "description": "The specific chart timeframe for entry signals is not specified",
      "example": "M5 (5-minute chart)",
      "context": "Required to implement entry logic correctly"
    }},
    {{
      "field_name": "trailing_stop",
      "category": "IMPORTANT",
      "description": "Trailing stop method is not mentioned",
      "example": "Trail stop at breakeven after price moves 1x risk in favor",
      "context": "Important for protecting profits once trade is in profit"
    }}
  ],
  "recommendations": [
    "Entry timeframe should be specified for accurate backtesting",
    "Consider adding trailing stop for better risk management"
  ]
}}
```

**Rules:**
- Only flag fields that are genuinely missing or incomplete
- Vague mentions (e.g., "small stop loss") should be flagged as missing
- If field is present and specific, do not flag it
- Prioritize CRITICAL gaps over NICE_TO_HAVE"""),
    ("human", """Review the following extracted trading concepts and identify missing information:

**Extracted Concepts:**
{extracted_concepts}

Analyze the data and return ONLY valid JSON with identified gaps following the specified format.""")
])

# =============================================================================
# 3. TRD_GENERATION_PROMPT - Generate TRD Markdown
# =============================================================================

TRD_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical documentation specialist specializing in trading strategy requirements.

Your task is to generate a complete Technical Requirements Document (TRD) in markdown format from the provided trading strategy information.

**TRD Structure:**

```markdown
---
strategy_name: "{Strategy Name}"
source: "{Source}"
generated_at: "{ISO 8601 Timestamp}"
status: "draft|complete"
version: "1.0"
analyst_version: "1.0"
kb_collection: "analyst_kb"
kb_articles_count: {Number}
---

# Trading Strategy: {Strategy Name}

## Overview

{2-3 paragraph overview of the strategy including: what it does, when it trades, what makes it unique}

**Source:** {Video title or document name}
**Analyst:** Analyst Agent v1.0
**Generated:** {Timestamp}
**Status:** {Draft/Complete}

---

## Entry Logic

### Primary Entry Trigger

- {Entry condition 1}
- {Entry condition 2}
- {Entry condition N}

### Entry Confirmation

- {Confirmation signal if mentioned}
- {Timeframe: e.g., 1-minute, 5-minute}

### Entry Example

```
IF [condition 1] AND [condition 2]
AND time is within [session]
THEN enter [LONG/SHORT] at [price level]
```

---

## Exit Logic

### Take Profit

- {TP strategy: fixed pips, percentage, resistance}
- {TP value if specified}

### Stop Loss

- {SL strategy: fixed pips, ATR-based, support}
- {SL value if specified}

### Trailing Stop

- {Trailing method if mentioned}
- {Trail distance}

### Time Exit

- {Close at specific time if mentioned}

---

## Filters

### Time Filters

- **Trading Session:** {London, NY, Asian, overlap}
- **Time of Day:** {e.g., 8am-12pm GMT}
- **Days to Avoid:** {Friday, news days, etc.}

### Market Condition Filters

- **Volatility:** {ATR threshold, avoid low volatility}
- **Spread:** {Max spread allowed}
- **Trend:** {Trend following or range}
- **News Events:** {Avoid high-impact news}

---

## Indicators & Settings

| Indicator | Settings | Purpose |
|-----------|----------|---------|
| {RSI} | {Period: 14} | {Entry/exit signal} |
| {ATR} | {Period: 14} | {Stop loss calculation} |
| {MA} | {Period: 20, EMA} | {Trend filter} |

---

## Position Sizing & Risk Management

> **Note:** This section will be enhanced in v2.0 with dynamic risk management system.

### Risk Per Trade

- {Mentioned: e.g., 0.5% of account}
- {Or: TODO - specify risk amount}

### Position Sizing

- {Method: fixed lots, risk-based, volatility-based}
- {Or: TODO - implement position sizing}

### Max Drawdown Limit

- {Daily limit if mentioned}
- {Or: TODO - specify drawdown limit}

---

## Knowledge Base References

### Relevant Articles Found

1. **[{Article Title 1}]({URL})**
   - **Relevance:** {why it matters}
   - **Key Insight:** {specific implementation detail}
   - **Category:** {MQL5 category}

2. **[{Article Title 2}]({URL})**
   - **Relevance:** {implementation pattern}
   - **Key Insight:** {technical consideration}

### Implementation Notes

- {Technical considerations from KB articles}
- {Potential pitfalls mentioned in literature}
- {Suggested MQL5 patterns}

---

## Missing Information (Requires Input)

{If gaps exist, list them here. Otherwise omit this section}

The following information was not found in the source and needs user input:

- [ ] [{Field 1}]: {description}
- [ ] [{Field 2}]: {description}

**To complete this TRD, run:**
\`\`\`bash
python tools/analyst_cli.py --complete docs/trds/{this_file}.md
\`\`\`

---

## Next Steps

1. **Review this TRD** - Verify all sections are accurate
2. **Complete missing fields** - Provide input for TODO items
3. **Generate MQL5 code** - Use QuantCode CLI (future)
4. **Backtest** - Validate strategy performance

---

**Generated by:** QuantMindX Analyst Agent v1.0
**Knowledge Base:** analyst_kb ({count} articles)
**LangGraph Version:** 0.2.x
```

**Generation Guidelines:**

1. **Be Specific**: Use exact values from the extracted data
2. **Be Complete**: Fill all sections with available information
3. **Be Clear**: Use trading terminology correctly
4. **Be Organized**: Follow the structure exactly
5. **Cross-Reference**: Incorporate insights from KB articles
6. **Flag Gaps**: Clearly mark missing information

**Markdown Formatting Rules:**
- Use proper markdown syntax (bold, italic, code blocks, tables)
- Include YAML frontmatter at the top
- Use horizontal rules (---) to separate major sections
- Format code examples with proper language tags
- Create proper tables for indicators and settings

**Integration with KB Articles:**
- Reference relevant KB articles in the Knowledge Base References section
- Extract key insights from article previews
- Note any implementation patterns mentioned
- Highlight potential pitfalls"""),
    ("human", """Generate a complete Technical Requirements Document (TRD) from the following information:

**Extracted Concepts:**
{extracted_concepts}

**User Answers (if any):**
{user_answers}

**Knowledge Base Articles:**
{kb_articles}

**Source:** {source}
**Timestamp:** {timestamp}

Generate a well-formatted markdown TRD following the specified structure. Include all sections, incorporate KB references, and clearly flag any missing information.""")
])

# =============================================================================
# 4. KB_SEARCH_PROMPTS (SEARCH_QUERY_PROMPT) - Generate Search Queries
# =============================================================================

SEARCH_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a semantic search expert specializing in trading strategy concepts.

Your task is to generate effective search queries for a knowledge base containing MQL5 trading articles.

**Query Generation Strategy:**

1. **Analyze the extracted concepts** and identify key themes:
   - Strategy type (scalping, swing, breakout, reversal)
   - Technical indicators (RSI, MACD, ATR, moving averages)
   - Entry/exit mechanisms
   - Risk management approaches
   - Time filters (sessions, specific times)
   - Market conditions (volatility, spread, trend)

2. **Generate diverse queries** that cover different aspects:
   - **Implementation queries**: How to code specific components
   - **Best practice queries**: Recommended approaches
   - **Troubleshooting queries**: Common pitfalls and solutions
   - **Indicator-specific queries**: Technical indicator usage
   - **Pattern queries**: Trading pattern implementations

3. **Optimize for semantic search**:
   - Use natural language (not just keywords)
   - Include trading terminology
   - Mention MQL5/Forex context
   - Be specific but not overly narrow
   - Include synonyms and related terms

**Query Examples:**

For an ORB (Opening Range Breakout) strategy:
```
Query 1: "Opening range breakout strategy implementation MQL5"
Query 2: "How to code breakout entry signals with volume confirmation"
Query 3: "ATR based stop loss placement forex expert advisor"
Query 4: "London session time filter Metatrader EA"
Query 5: "Trailing stop implementation break even MQL5"
```

For an RSI oversold strategy:
```
Query 1: "RSI oversold entry signal MQL5 expert advisor"
Query 2: "How to implement trend following with RSI and EMA"
Query 3: "Position sizing based on risk percentage forex"
Query 4: "Take profit risk reward ratio implementation"
Query 5: "Market volatility filter ATR threshold"
```

**Output Format:**

Return ONLY valid JSON:

```json
{{
  "queries": [
    "{query 1}",
    "{query 2}",
    "{query 3}",
    "{query 4}",
    "{query 5}"
  ],
  "query_count": 5,
  "rationale": "{Brief explanation of query selection strategy}"
}}
```

**Rules:**
- Generate 3-5 diverse queries
- Each query should be 5-10 words
- Include MQL5/Forex context in most queries
- Cover different aspects (entry, exit, indicators, risk)
- Avoid overly similar queries
- Focus on implementation and best practices"""),
    ("human", """Generate semantic search queries for the knowledge base based on the following trading strategy concepts:

**Extracted Concepts:**
{extracted_concepts}

**Strategy Keywords:**
{keywords}

Generate 3-5 diverse search queries that will retrieve relevant MQL5 articles for implementing this strategy. Return ONLY valid JSON.""")
])

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "EXTRACTION_PROMPT",
    "GAP_DETECTION_PROMPT",
    "TRD_GENERATION_PROMPT",
    "SEARCH_QUERY_PROMPT",
]
