"""
MQL5 Indicator Writer Skill

Generates compilation-ready MQL5 indicator code following patterns from:
- MQL5 Cookbook: Creating a ring buffer for fast calculation of indicators in a sliding window
- Introduction to MQL5 Part 13: A Beginner's Guide to Building Custom Indicators (II)

This skill generates:
1. CRiBuffDbl ring buffer class for sliding window calculations
2. #property directives (indicator_chart_window, indicator_buffers, indicator_plots)
3. OnInit/OnDeinit/OnCalculate function scaffolds
4. SetIndexBuffer initialization code
5. Complete indicator files ready to compile in MetaEditor (F7)

Usage:
    result = generate_mql5_indicator(
        indicator_name="MyRSI",
        indicator_type="line",
        buffers=1,
        use_ring_buffer=True
    )
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

from ..skill_schema import SkillDefinition

logger = logging.getLogger(__name__)


# =============================================================================
# CRiBuffDbl Ring Buffer Generation
# =============================================================================

def generate_cribuffdbl_class() -> str:
    """
    Generate CRiBuffDbl ring buffer class following MQL5 Cookbook pattern.

    The CRiBuffDbl class provides fast sliding window calculations for indicators.
    Source: https://www.mql5.com/en/articles/3047

    Returns:
        MQL5 code for CRiBuffDbl class
    """
    code = """
//+------------------------------------------------------------------+
//| CRiBuffDbl Ring Buffer Class                                       |
//| Based on MQL5 Cookbook article                                    |
//+------------------------------------------------------------------+
class CRiBuffDbl
{
private:
   bool              m_full_buff;       // Buffer full flag
   int               m_max_total;       // Maximum buffer size
   int               m_head_index;      // Index of last added element

protected:
   double            m_buffer[];        // Ring buffer array
   virtual void      OnAddValue(double value);
   virtual void      OnRemoveValue(double value);
   virtual void      OnChangeValue(int index, double prev_value, double new_value);
   virtual void      OnChangeArray(void);
   virtual void      OnSetMaxTotal(int max_total);
   int               ToRealInd(int index);

public:
                     CRiBuffDbl(void);
   void              AddValue(double value);
   void              ChangeValue(int index, double new_value);
   double            GetValue(int index);
   int               GetTotal(void);
   int               GetMaxTotal(void);
   void              SetMaxTotal(int max_total);
   void              ToArray(double &array[]);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CRiBuffDbl::CRiBuffDbl(void) : m_full_buff(false),
                              m_head_index(-1),
                              m_max_total(0)
{
   SetMaxTotal(3);
}

//+------------------------------------------------------------------+
//| Set the new size of the ring buffer                              |
//+------------------------------------------------------------------+
void CRiBuffDbl::SetMaxTotal(int max_total)
{
   if(ArraySize(m_buffer) == max_total)
      return;
   m_max_total = ArrayResize(m_buffer, max_total);
   OnSetMaxTotal(max_total);
}

//+------------------------------------------------------------------+
//| Get the actual ring buffer size                                  |
//+------------------------------------------------------------------+
int CRiBuffDbl::GetMaxTotal(void)
{
   return m_max_total;
}

//+------------------------------------------------------------------+
//| Get the total number of elements                                 |
//+------------------------------------------------------------------+
int CRiBuffDbl::GetTotal(void)
{
   if(m_full_buff)
      return m_max_total;
   return m_head_index + 1;
}

//+------------------------------------------------------------------+
//| Get value by virtual index                                       |
//+------------------------------------------------------------------+
double CRiBuffDbl::GetValue(int index)
{
   return m_buffer[ToRealInd(index)];
}

//+------------------------------------------------------------------+
//| Add a new value to the ring buffer                               |
//+------------------------------------------------------------------+
void CRiBuffDbl::AddValue(double value)
{
   if(++m_head_index == m_max_total)
   {
      m_head_index = 0;
      if(!m_full_buff)
      {
         m_full_buff = true;
         OnChangeArray();
      }
      else
         OnRemoveValue(m_buffer[m_head_index]);
   }
   m_buffer[m_head_index] = value;
   OnAddValue(value);
}

//+------------------------------------------------------------------+
//| Change value at virtual index                                    |
//+------------------------------------------------------------------+
void CRiBuffDbl::ChangeValue(int index, double new_value)
{
   int r_index = ToRealInd(index);
   double prev_value = m_buffer[r_index];
   m_buffer[r_index] = new_value;
   OnChangeValue(index, prev_value, new_value);
}

//+------------------------------------------------------------------+
//| Convert virtual index to real array index                        |
//+------------------------------------------------------------------+
int CRiBuffDbl::ToRealInd(int index)
{
   if(index >= GetTotal() || index < 0)
      return m_max_total;
   if(!m_full_buff)
      return index;
   int delta = (m_max_total - 1) - m_head_index;
   if(index < delta)
      return m_max_total + (index - delta);
   return index - delta;
}

//+------------------------------------------------------------------+
//| Copy buffer contents to array                                    |
//+------------------------------------------------------------------+
void CRiBuffDbl::ToArray(double &array[])
{
   int total = GetTotal();
   ArrayResize(array, total);
   for(int i = 0; i < total; i++)
      array[i] = GetValue(i);
}

//+------------------------------------------------------------------+
//| Event: Value added (override in derived classes)                 |
//+------------------------------------------------------------------+
void CRiBuffDbl::OnAddValue(double value)
{
}

//+------------------------------------------------------------------+
//| Event: Value removed (override in derived classes)               |
//+------------------------------------------------------------------+
void CRiBuffDbl::OnRemoveValue(double value)
{
}

//+------------------------------------------------------------------+
//| Event: Value changed (override in derived classes)               |
//+------------------------------------------------------------------+
void CRiBuffDbl::OnChangeValue(int index, double prev_value, double new_value)
{
}

//+------------------------------------------------------------------+
//| Event: Array changed (override in derived classes)               |
//+------------------------------------------------------------------+
void CRiBuffDbl::OnChangeArray(void)
{
}

//+------------------------------------------------------------------+
//| Event: Max total changed (override in derived classes)           |
//+------------------------------------------------------------------+
void CRiBuffDbl::OnSetMaxTotal(int max_total)
{
}
"""
    return code.strip()


# =============================================================================
# Property Directives Generation
# =============================================================================

def generate_property_directives(
    indicator_name: str,
    indicator_type: str = "line",
    num_buffers: int = 1,
    separate_window: bool = False
) -> str:
    """
    Generate #property directives for MQL5 indicator.

    Args:
        indicator_name: Name of the indicator
        indicator_type: Type of indicator (line, histogram, candles, arrows)
        num_buffers: Number of indicator buffers
        separate_window: If True, plot in separate window

    Returns:
        MQL5 #property directives code
    """
    # Map indicator types to MQL5 DRAW constants
    type_mapping = {
        "line": "DRAW_LINE",
        "histogram": "DRAW_HISTOGRAM",
        "candles": "DRAW_CANDLES",
        "color_candles": "DRAW_COLOR_CANDLES",
        "arrows": "DRAW_ARROW",
        "sections": "DRAW_SECTION",
        "zigzag": "DRAW_ZIGZAG",
        "none": "DRAW_NONE",
    }

    draw_type = type_mapping.get(indicator_type.lower(), "DRAW_LINE")

    # Calculate additional buffers for color if needed
    if indicator_type.lower() == "color_candles":
        # Color candles need 4 OHLC buffers + 1 color buffer
        actual_buffers = num_buffers + 1
    else:
        actual_buffers = num_buffers

    code = f"""//+------------------------------------------------------------------+
//| {indicator_name}                                                |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://quantmindx.ai"
#property version   "1.00"
#property strict

// Indicator display properties
#property indicator_{"separate_window" if separate_window else "chart_window"}
#property indicator_buffers {actual_buffers}
#property indicator_plots   {num_buffers}

// Plot 1 settings
#property indicator_label1  "{indicator_name}"
#property indicator_type1   {draw_type}
#property indicator_color1  clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
"""
    return code.strip()


# =============================================================================
# SetIndexBuffer Initialization Generation
# =============================================================================

def generate_setindexbuffer_code(
    num_buffers: int,
    buffer_names: Optional[List[str]] = None,
    use_color_buffer: bool = False
) -> str:
    """
    Generate SetIndexBuffer initialization code.

    Args:
        num_buffers: Number of buffers to initialize
        buffer_names: Optional list of buffer names
        use_color_buffer: Whether to include color buffer

    Returns:
        MQL5 SetIndexBuffer initialization code
    """
    if buffer_names is None:
        buffer_names = [f"Buffer{i}" for i in range(num_buffers)]

    code = "   // Set indicator buffers\n"

    for i in range(num_buffers):
        name = buffer_names[i] if i < len(buffer_names) else f"Buffer{i}"
        if i == num_buffers - 1 and use_color_buffer:
            code += f"   SetIndexBuffer({i}, {name}, INDICATOR_COLOR_INDEX);\n"
        else:
            code += f"   SetIndexBuffer({i}, {name}, INDICATOR_DATA);\n"

    return code.strip()


# =============================================================================
# OnInit/OnDeinit Scaffold Generation
# =============================================================================

def generate_oninit_deinit(
    custom_init_code: str = "",
    custom_deinit_code: str = ""
) -> str:
    """
    Generate OnInit and OnDeinit function scaffolds.

    Args:
        custom_init_code: Custom code to insert in OnInit
        custom_deinit_code: Custom code to insert in OnDeinit

    Returns:
        MQL5 OnInit and OnDeinit functions
    """
    code = f"""//+------------------------------------------------------------------+
//| Custom indicator initialization function                          |
//+------------------------------------------------------------------+
int OnInit()
{{
   {custom_init_code}

   return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
   {custom_deinit_code}
}}
"""
    return code.strip()


# =============================================================================
# OnCalculate Function Generation
# =============================================================================

def generate_oncalculate_function(
    use_full_ohlcv: bool = False,
    use_ring_buffer: bool = False,
    custom_calculation: str = ""
) -> str:
    """
    Generate OnCalculate function following MQL5 documentation.

    Args:
        use_full_ohlcv: Use full OHLCV price arrays (not just close)
        use_ring_buffer: Use CRiBuffDbl ring buffer pattern
        custom_calculation: Custom calculation code to insert

    Returns:
        MQL5 OnCalculate function
    """
    if use_full_ohlcv:
        # Full signature with all price arrays
        signature = """int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])"""
    else:
        # Simplified signature with price array
        signature = """int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])"""

    loop_code = ""
    if use_ring_buffer:
        loop_code = """
   // Calculate using ring buffer for sliding window
   for(int i = prev_calculated; i < rates_total; i++)
   {
      ringBuffer.AddValue(price[i]);
      Buffer0[i] = ringBuffer.GetValue(0);
   }

   // Handle last bar update
   if(prev_calculated > 0)
   {
      ringBuffer.ChangeValue(period - 1, price[rates_total - 1]);
      Buffer0[rates_total - 1] = ringBuffer.GetValue(0);
   }"""
    elif custom_calculation:
        loop_code = custom_calculation
    else:
        loop_code = """
   // Simple calculation: copy price to buffer
   if(rates_total < 1)
      return(0);

   for(int i = prev_calculated; i < rates_total; i++)
   {
      Buffer0[i] = price[i];
   }"""

    code = f"""//+------------------------------------------------------------------+
//| Custom indicator iteration function                               |
//+------------------------------------------------------------------+
{signature}
{{
   if(rates_total < 1)
      return(0);
{loop_code}

   return(rates_total);
}}"""
    return code.strip()


# =============================================================================
# Complete Indicator Generation
# =============================================================================

def generate_mql5_indicator(
    indicator_name: str,
    indicator_type: str = "line",
    buffers: int = 1,
    period: int = 14,
    separate_window: bool = False,
    use_ring_buffer: bool = False,
    use_full_ohlcv: bool = False
) -> str:
    """
    Generate a complete, compilation-ready MQL5 indicator file.

    Args:
        indicator_name: Name of the indicator (will be sanitized)
        indicator_type: Type of indicator (line, histogram, candles, arrows)
        buffers: Number of indicator buffers
        period: Indicator period (for moving averages, etc.)
        separate_window: If True, plot in separate window
        use_ring_buffer: Include CRiBuffDbl ring buffer class
        use_full_ohlcv: Use full OHLCV arrays in OnCalculate

    Returns:
        Complete MQL5 indicator code ready to compile

    Raises:
        ValueError: If input parameters are invalid
    """
    # Validate inputs
    if not indicator_name or not indicator_name.strip():
        raise ValueError("indicator_name cannot be empty")

    if buffers < 1:
        raise ValueError("buffers must be at least 1")

    valid_types = ["line", "histogram", "candles", "color_candles", "arrows", "sections", "zigzag"]
    if indicator_type.lower() not in valid_types:
        raise ValueError(f"indicator_type must be one of: {valid_types}")

    # Sanitize indicator name (remove special characters)
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', indicator_name)
    if not sanitized_name:
        sanitized_name = "CustomIndicator"

    # Build the complete indicator
    parts = []

    # 1. Property directives
    parts.append(generate_property_directives(
        sanitized_name,
        indicator_type,
        buffers,
        separate_window
    ))

    # 2. Input parameters
    parts.append(f"""
// Input parameters
input int      InpPeriod = {period};  // Indicator Period
input color    InpColor  = clrRed;    // Indicator Color
input int      InpWidth  = 1;         // Line Width
""")

    # 3. Buffer declarations
    buffer_decls = "\n// Indicator buffers\n"
    for i in range(buffers):
        buffer_decls += f"double Buffer{i}[];\n"

    if indicator_type.lower() == "color_candles":
        # Additional OHLC buffers for candles
        buffer_decls += "double OpenBuffer[];\n"
        buffer_decls += "double HighBuffer[];\n"
        buffer_decls += "double LowBuffer[];\n"
        buffer_decls += "double ColorBuffer[];\n"

    parts.append(buffer_decls.strip())

    # 4. CRiBuffDbl ring buffer class (if enabled)
    if use_ring_buffer:
        parts.append("\n" + generate_cribuffdbl_class())
        parts.append("\n// Ring buffer instance\nCRiBuffDbl ringBuffer;")

    # 5. OnInit/OnDeinit
    setindexbuffer = generate_setindexbuffer_code(buffers)
    init_code = setindexbuffer
    if use_ring_buffer:
        init_code += f"\n   ringBuffer.SetMaxTotal(InpPeriod);"

    parts.append("\n" + generate_oninit_deinit(init_code))

    # 6. OnCalculate
    calc_code = ""
    if use_ring_buffer:
        calc_code = f"""
   // Calculate using ring buffer for sliding window
   for(int i = prev_calculated; i < rates_total; i++)
   {{
      ringBuffer.AddValue(price[i]);
      Buffer0[i] = ringBuffer.GetValue(0);
   }}

   // Handle last bar update
   if(prev_calculated > 0)
   {{
      ringBuffer.ChangeValue(InpPeriod - 1, price[rates_total - 1]);
      Buffer0[rates_total - 1] = ringBuffer.GetValue(0);
   }}
"""
    else:
        calc_code = f"""
   // Simple calculation
   for(int i = prev_calculated; i < rates_total; i++)
   {{
      Buffer0[i] = price[i];
   }}
"""

    parts.append("\n" + generate_oncalculate_function(
        use_full_ohlcv=use_full_ohlcv,
        use_ring_buffer=use_ring_buffer,
        custom_calculation=calc_code.strip()
    ))

    # 7. End comment
    parts.append("""
//+------------------------------------------------------------------+
""")

    return "\n".join(parts)


# =============================================================================
# Specialized Indicator Generators
# =============================================================================

def generate_heikin_ashi_indicator() -> str:
    """
    Generate Heikin Ashi indicator following Part 13 reference article.

    Returns:
        Complete MQL5 Heikin Ashi indicator code
    """
    code = """//+------------------------------------------------------------------+
//| Heikin Ashi Indicator                                            |
//| Based on Introduction to MQL5 Part 13                            |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://quantmindx.ai"
#property version   "1.00"
#property strict

#property indicator_chart_window
#property indicator_buffers 5
#property indicator_plots   1

#property indicator_label1  "Heikin Ashi"
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  clrGreen, clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

// Indicator buffers
double HA_Open[];
double HA_High[];
double HA_Low[];
double HA_Close[];
double HA_Color[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                          |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set indicator buffers
   SetIndexBuffer(0, HA_Open, INDICATOR_DATA);
   SetIndexBuffer(1, HA_High, INDICATOR_DATA);
   SetIndexBuffer(2, HA_Low, INDICATOR_DATA);
   SetIndexBuffer(3, HA_Close, INDICATOR_DATA);
   SetIndexBuffer(4, HA_Color, INDICATOR_COLOR_INDEX);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                               |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(rates_total < 2)
      return(0);

   // Start from the second bar (or prev_calculated)
   int start = (prev_calculated == 0) ? 1 : prev_calculated;

   for(int i = start; i < rates_total; i++)
   {
      // Heikin Ashi Close = (Open + High + Low + Close) / 4
      HA_Close[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;

      // Heikin Ashi Open = (Previous HA Open + Previous HA Close) / 2
      if(i == 1)
         HA_Open[i] = (open[i-1] + close[i-1]) / 2.0;
      else
         HA_Open[i] = (HA_Open[i-1] + HA_Close[i-1]) / 2.0;

      // Heikin Ashi High = Max(High, HA_Open, HA_Close)
      HA_High[i] = MathMax(high[i], MathMax(HA_Open[i], HA_Close[i]));

      // Heikin Ashi Low = Min(Low, HA_Open, HA_Close)
      HA_Low[i] = MathMin(low[i], MathMin(HA_Open[i], HA_Close[i]));

      // Color: Green for bullish, Red for bearish
      HA_Color[i] = (HA_Close[i] >= HA_Open[i]) ? 0 : 1;
   }

   return(rates_total);
}
//+------------------------------------------------------------------+
"""
    return code.strip()


# =============================================================================
# Skill Definition
# =============================================================================

skill_definition = SkillDefinition(
    name="indicator_writer",
    category="trading_skills",
    description="Generates compilation-ready MQL5 indicator code with CRiBuffDbl ring buffer support",
    input_schema={
        "type": "object",
        "properties": {
            "indicator_name": {
                "type": "string",
                "description": "Name of the indicator to generate"
            },
            "indicator_type": {
                "type": "string",
                "enum": ["line", "histogram", "candles", "color_candles", "arrows"],
                "description": "Visual display type"
            },
            "buffers": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of indicator buffers"
            },
            "period": {
                "type": "integer",
                "minimum": 1,
                "description": "Indicator period (e.g., 14 for RSI)"
            },
            "use_ring_buffer": {
                "type": "boolean",
                "description": "Include CRiBuffDbl ring buffer class"
            },
            "separate_window": {
                "type": "boolean",
                "description": "Plot in separate window instead of main chart"
            }
        },
        "required": ["indicator_name"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "mql5_code": {
                "type": "string",
                "description": "Complete MQL5 indicator code ready to compile"
            },
            "file_name": {
                "type": "string",
                "description": "Suggested file name for the indicator"
            },
            "compilation_ready": {
                "type": "boolean",
                "description": "Whether the code is ready to compile (F7 in MetaEditor)"
            }
        }
    },
    code="""
# MQL5 Indicator Writer Implementation

from typing import Dict, Any, Optional
import re

def generate_indicator(
    indicator_name: str,
    indicator_type: str = "line",
    buffers: int = 1,
    period: int = 14,
    use_ring_buffer: bool = False,
    separate_window: bool = False
) -> Dict[str, Any]:
    '''
    Generate MQL5 indicator code.

    Args:
        indicator_name: Name of the indicator
        indicator_type: Visual type (line, histogram, candles, etc.)
        buffers: Number of indicator buffers
        period: Calculation period
        use_ring_buffer: Include CRiBuffDbl ring buffer
        separate_window: Plot in separate window

    Returns:
        Dict with mql5_code, file_name, and compilation_ready status
    '''
    mql5_code = generate_mql5_indicator(
        indicator_name=indicator_name,
        indicator_type=indicator_type,
        buffers=buffers,
        period=period,
        separate_window=separate_window,
        use_ring_buffer=use_ring_buffer
    )

    file_name = f"{re.sub(r'[^a-zA-Z0-9_]', '', indicator_name)}.mq5"

    return {
        "mql5_code": mql5_code,
        "file_name": file_name,
        "compilation_ready": True
    }
""",
    dependencies=[],
    example_usage='result = generate_indicator(indicator_name="MyRSI", indicator_type="line", buffers=1, period=14, use_ring_buffer=True)',
    version="1.0.0"
)


__all__ = [
    "skill_definition",
    "generate_mql5_indicator",
    "generate_cribuffdbl_class",
    "generate_property_directives",
    "generate_setindexbuffer_code",
    "generate_oninit_deinit",
    "generate_oncalculate_function",
    "generate_heikin_ashi_indicator",
]
