//+------------------------------------------------------------------+
//|                                             bollinger_bands.mq5  |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| Implementation of Bollinger Bands for volatility and price      |
//| level analysis.                                                 |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict
#property indicator_chart_window
#property indicator_buffers 3
#property indicator_plots   3

//--- plot Upper Band
#property indicator_label1  "Upper Band"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrRed
#property indicator_style1  STYLE_DOT
#property indicator_width1  1

//--- plot Middle Band
#property indicator_label2  "Middle Band"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrYellow
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//--- plot Lower Band
#property indicator_label3  "Lower Band"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrLimeGreen
#property indicator_style3  STYLE_DOT
#property indicator_width3  1

//--- input parameters
input int      InpBBPeriod    = 20;        // BB Period
input double   InpDeviations  = 2.0;       // Deviations
input int      InpAppliedPrice = PRICE_CLOSE; // Applied price
input int      InpMA_Method   = MODE_SMA;  // MA Method

//--- indicator buffers
double         ExtUpperBuffer[];
double         ExtMiddleBuffer[];
double         ExtLowerBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- check input parameters
   if(InpBBPeriod <= 0 || InpDeviations <= 0)
     {
      Print("Wrong input parameters");
      return(INIT_PARAMETERS_INCORRECT);
     }
   
   //--- set indicator buffers
   SetIndexBuffer(0, ExtUpperBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, ExtMiddleBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, ExtLowerBuffer, INDICATOR_DATA);
   
   //--- set drawing begin
   SetIndexDrawBegin(0, InpBBPeriod - 1);
   SetIndexDrawBegin(1, InpBBPeriod - 1);
   SetIndexDrawBegin(2, InpBBPeriod - 1);
   
   //--- set indicator short name
   IndicatorShortName("Bollinger Bands(", InpBBPeriod, ",", DoubleToString(InpDeviations, 1), ")");
   
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
   if(rates_total < InpBBPeriod)
      return(0);
   
   //--- calculate Bollinger Bands
   for(int i = start; i < rates_total && !IsStopped(); i++)
     {
      if(i < InpBBPeriod - 1)
        {
         ExtUpperBuffer[i] = 0.0;
         ExtMiddleBuffer[i] = 0.0;
         ExtLowerBuffer[i] = 0.0;
         continue;
        }
      
      //--- calculate middle band (moving average)
      ExtMiddleBuffer[i] = iMA(NULL, 0, InpBBPeriod, 0, InpMA_Method, InpAppliedPrice, i);
      
      //--- calculate standard deviation
      double sum = 0.0;
      for(int j = 0; j < InpBBPeriod; j++)
        {
         double price = GetPrice(close, open, high, low, i + j, InpAppliedPrice);
         double diff = price - ExtMiddleBuffer[i];
         sum += diff * diff;
        }
      
      double standard_deviation = MathSqrt(sum / InpBBPeriod);
      
      //--- calculate upper and lower bands
      double band_width = InpDeviations * standard_deviation;
      ExtUpperBuffer[i] = ExtMiddleBuffer[i] + band_width;
      ExtLowerBuffer[i] = ExtMiddleBuffer[i] - band_width;
     }
   
   //--- return number of calculated bars
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Get price based on applied price type                            |
//+------------------------------------------------------------------+
double GetPrice(const double &close[],
                const double &open[],
                const double &high[],
                const double &low[],
                int index,
                int price_type)
  {
   switch(price_type)
     {
      case PRICE_CLOSE:  return(close[index]);
      case PRICE_OPEN:   return(open[index]);
      case PRICE_HIGH:   return(high[index]);
      case PRICE_LOW:    return(low[index]);
      case PRICE_MEDIAN: return((high[index] + low[index]) / 2.0);
      case PRICE_TYPICAL: return((high[index] + low[index] + close[index]) / 3.0);
      case PRICE_WEIGHTED: return((high[index] + low[index] + 2.0 * close[index]) / 4.0);
      default:           return(close[index]);
     }
  }

//+------------------------------------------------------------------+
//| Get current upper band value                                     |
//+------------------------------------------------------------------+
double GetUpperBand()
  {
   return(ExtUpperBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Get current middle band value                                    |
//+------------------------------------------------------------------+
double GetMiddleBand()
  {
   return(ExtMiddleBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Get current lower band value                                     |
//+------------------------------------------------------------------+
double GetLowerBand()
  {
   return(ExtLowerBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Get current bandwidth                                            |
//+------------------------------------------------------------------+
double GetBandwidth()
  {
   if(ExtMiddleBuffer[0] == 0)
      return(0);
   
   return((ExtUpperBuffer[0] - ExtLowerBuffer[0]) / ExtMiddleBuffer[0] * 100.0);
  }

//+------------------------------------------------------------------+
//| Get current %B value                                             |
//+------------------------------------------------------------------+
double GetPercentB(const double &price[])
  {
   double current_price = GetPrice(price, NULL, NULL, NULL, 0, InpAppliedPrice);
   
   if(ExtUpperBuffer[0] == ExtLowerBuffer[0])
      return(0);
   
   return((current_price - ExtLowerBuffer[0]) / (ExtUpperBuffer[0] - ExtLowerBuffer[0]));
  }

//+------------------------------------------------------------------+
//| Check for price touching upper band                              |
//+------------------------------------------------------------------+
bool IsTouchingUpperBand(const double &price[], double threshold_percent = 1.0)
  {
   double current_price = GetPrice(price, NULL, NULL, NULL, 0, InpAppliedPrice);
   double upper_band = ExtUpperBuffer[0];
   double threshold = upper_band * (threshold_percent / 100.0);
   
   return(current_price >= upper_band - threshold);
  }

//+------------------------------------------------------------------+
//| Check for price touching lower band                              |
//+------------------------------------------------------------------+
bool IsTouchingLowerBand(const double &price[], double threshold_percent = 1.0)
  {
   double current_price = GetPrice(price, NULL, NULL, NULL, 0, InpAppliedPrice);
   double lower_band = ExtLowerBuffer[0];
   double threshold = lower_band * (threshold_percent / 100.0);
   
   return(current_price <= lower_band + threshold);
  }

//+------------------------------------------------------------------+
//| Check for price breakout above upper band                        |
//+------------------------------------------------------------------+
bool IsBreakoutAbove(const double &price[])
  {
   double current_price = GetPrice(price, NULL, NULL, NULL, 0, InpAppliedPrice);
   double previous_price = GetPrice(price, NULL, NULL, NULL, 1, InpAppliedPrice);
   
   return(previous_price <= ExtUpperBuffer[1] && current_price > ExtUpperBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Check for price breakout below lower band                        |
//+------------------------------------------------------------------+
bool IsBreakoutBelow(const double &price[])
  {
   double current_price = GetPrice(price, NULL, NULL, NULL, 0, InpAppliedPrice);
   double previous_price = GetPrice(price, NULL, NULL, NULL, 1, InpAppliedPrice);
   
   return(previous_price >= ExtLowerBuffer[1] && current_price < ExtLowerBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Get volatility level based on bandwidth                          |
//+------------------------------------------------------------------+
int GetVolatilityLevel()
  {
   double bandwidth = GetBandwidth();
   
   if(bandwidth < 2.0)
      return(1); // Low volatility
   else if(bandwidth < 4.0)
      return(2); // Medium volatility
   else
      return(3); // High volatility
  }

//+------------------------------------------------------------------+
//| Get squeeze signal                                               |
//+------------------------------------------------------------------+
bool IsSqueeze(double squeeze_threshold = 1.0)
  {
   //--- need at least 2 bars for comparison
   if(Bars < 2)
      return(false);
   
   double current_bandwidth = GetBandwidth();
   double previous_bandwidth = ((ExtUpperBuffer[1] - ExtLowerBuffer[1]) / ExtMiddleBuffer[1] * 100.0);
   
   //--- squeeze occurs when bandwidth decreases
   return(current_bandwidth < previous_bandwidth && current_bandwidth < squeeze_threshold);
  }