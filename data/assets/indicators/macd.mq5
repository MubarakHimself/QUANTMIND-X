//+------------------------------------------------------------------+
//|                                                  macd.mq5        |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| This file implements the Moving Average Convergence Divergence  |
//| (MACD) indicator for trend and momentum analysis.               |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3

//--- plot MACD
#property indicator_label1  "MACD"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- plot Signal
#property indicator_label2  "Signal"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrRed
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1

//--- plot Histogram
#property indicator_label3  "Histogram"
#property indicator_type3   DRAW_HISTOGRAM
#property indicator_color3  clrSilver
#property indicator_style3  STYLE_SOLID
#property indicator_width3  2

//--- input parameters
input int      InpFastEMA     = 12;        // Fast EMA Period
input int      InpSlowEMA     = 26;        // Slow EMA Period
input int      InpSignalSMA   = 9;         // Signal SMA Period
input int      InpAppliedPrice = PRICE_CLOSE; // Applied price

//--- indicator buffers
double         ExtMacdBuffer[];
double         ExtSignalBuffer[];
double         ExtHistogramBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- check input parameters
   if(InpFastEMA <= 0 || InpSlowEMA <= 0 || InpSignalSMA <= 0)
     {
      Print("Wrong input parameters");
      return(INIT_PARAMETERS_INCORRECT);
     }
   
   if(InpFastEMA >= InpSlowEMA)
     {
      Print("Fast EMA period must be less than Slow EMA period");
      return(INIT_PARAMETERS_INCORRECT);
     }
   
   //--- set indicator buffers
   SetIndexBuffer(0, ExtMacdBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, ExtSignalBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, ExtHistogramBuffer, INDICATOR_DATA);
   
   //--- set drawing begin
   SetIndexDrawBegin(0, InpSlowEMA + InpSignalSMA - 1);
   SetIndexDrawBegin(1, InpSlowEMA + InpSignalSMA - 1);
   SetIndexDrawBegin(2, InpSlowEMA + InpSignalSMA - 1);
   
   //--- set indicator short name
   IndicatorShortName("MACD(", InpFastEMA, ",", InpSlowEMA, ",", InpSignalSMA, ")");
   
   //--- initialization done
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
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
   int start = prev_calculated > 0 ? prev_calculated - 1 : 0;
   
   //--- check for minimum bars
   if(rates_total < InpSlowEMA + InpSignalSMA)
      return(0);
   
   //--- calculate MACD
   for(int i = start; i < rates_total && !IsStopped(); i++)
     {
      if(i < InpSlowEMA + InpSignalSMA - 1)
        {
         ExtMacdBuffer[i] = 0.0;
         ExtSignalBuffer[i] = 0.0;
         ExtHistogramBuffer[i] = 0.0;
         continue;
        }
      
      //--- calculate fast and slow EMAs
      double fast_ema = iMA(NULL, 0, InpFastEMA, 0, MODE_EMA, InpAppliedPrice, i);
      double slow_ema = iMA(NULL, 0, InpSlowEMA, 0, MODE_EMA, InpAppliedPrice, i);
      
      //--- calculate MACD line
      ExtMacdBuffer[i] = fast_ema - slow_ema;
      
      //--- calculate signal line (SMA of MACD)
      double signal_sum = 0.0;
      for(int j = 0; j < InpSignalSMA; j++)
         signal_sum += iMA(NULL, 0, InpFastEMA, 0, MODE_EMA, InpAppliedPrice, i + j) - 
                      iMA(NULL, 0, InpSlowEMA, 0, MODE_EMA, InpAppliedPrice, i + j);
      
      ExtSignalBuffer[i] = signal_sum / InpSignalSMA;
      
      //--- calculate histogram
      ExtHistogramBuffer[i] = ExtMacdBuffer[i] - ExtSignalBuffer[i];
     }
   
   //--- return number of calculated bars
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Get current MACD value                                           |
//+------------------------------------------------------------------+
double GetMACDValue()
  {
   return(ExtMacdBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Get current Signal value                                         |
//+------------------------------------------------------------------+
double GetSignalValue()
  {
   return(ExtSignalBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Get current Histogram value                                      |
//+------------------------------------------------------------------+
double GetHistogramValue()
  {
   return(ExtHistogramBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Check for MACD crossover                                         |
//+------------------------------------------------------------------+
int GetCrossoverSignal()
  {
   //--- need at least 2 bars
   if(Bars < 2)
      return(0);
   
   //--- check for bullish crossover (MACD crosses above signal)
   if(ExtMacdBuffer[1] <= ExtSignalBuffer[1] && ExtMacdBuffer[0] > ExtSignalBuffer[0])
      return(1); // Bullish crossover
   
   //--- check for bearish crossover (MACD crosses below signal)
   if(ExtMacdBuffer[1] >= ExtSignalBuffer[1] && ExtMacdBuffer[0] < ExtSignalBuffer[0])
      return(-1); // Bearish crossover
   
   return(0); // No crossover
  }

//+------------------------------------------------------------------+
//| Check for MACD divergence                                        |
//+------------------------------------------------------------------+
int GetDivergenceSignal(int bars_back = 50)
  {
   //--- check minimum bars
   if(Bars < bars_back + 10)
      return(0);
   
   //--- look for bullish divergence
   double price_low_1 = iLow(NULL, 0, 1);
   double price_low_2 = iLow(NULL, 0, iLowest(NULL, 0, MODE_LOW, bars_back, 2));
   double macd_low_1 = ExtMacdBuffer[1];
   double macd_low_2 = ExtMacdBuffer[iLowest(NULL, 0, MODE_LOW, bars_back, 2)];
   
   if(price_low_2 < price_low_1 && macd_low_2 > macd_low_1)
      return(1); // Bullish divergence
   
   //--- look for bearish divergence
   double price_high_1 = iHigh(NULL, 0, 1);
   double price_high_2 = iHigh(NULL, 0, iHighest(NULL, 0, MODE_HIGH, bars_back, 2));
   double macd_high_1 = ExtMacdBuffer[1];
   double macd_high_2 = ExtMacdBuffer[iHighest(NULL, 0, MODE_HIGH, bars_back, 2)];
   
   if(price_high_2 > price_high_1 && macd_high_2 < macd_high_1)
      return(-1); // Bearish divergence
   
   return(0); // No divergence
  }

//+------------------------------------------------------------------+
//| Get MACD trend strength                                          |
//+------------------------------------------------------------------+
double GetTrendStrength()
  {
   //--- calculate trend strength based on histogram magnitude
   double hist_abs = MathAbs(ExtHistogramBuffer[0]);
   
   //--- normalize to 0-100 scale (assuming typical range of -2 to +2)
   double strength = hist_abs * 25.0; // 2 * 25 = 50, so max is 100
   return(MathMin(strength, 100.0));
  }

//+------------------------------------------------------------------+
//| Check if MACD is in bullish territory                            |
//+------------------------------------------------------------------+
bool IsBullish()
  {
   return(ExtMacdBuffer[0] > ExtSignalBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Check if MACD is in bearish territory                            |
//+------------------------------------------------------------------+
bool IsBearish()
  {
   return(ExtMacdBuffer[0] < ExtSignalBuffer[0]);
  }