//+------------------------------------------------------------------+
//|                                                    atr.mq5       |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| Implementation of Average True Range (ATR) for volatility       |
//| measurement and stop-loss calculation.                          |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict
#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1

//--- plot ATR
#property indicator_label1  "ATR"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- input parameters
input int      InpATRPeriod   = 14;        // ATR Period

//--- indicator buffers
double         ExtATRBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- check input parameters
   if(InpATRPeriod <= 0)
     {
      Print("Wrong input parameter ATR Period = ", InpATRPeriod);
      return(INIT_PARAMETERS_INCORRECT);
     }
   
   //--- set indicator buffers
   SetIndexBuffer(0, ExtATRBuffer, INDICATOR_DATA);
   
   //--- set drawing begin
   SetIndexDrawBegin(0, InpATRPeriod);
   
   //--- set indicator short name
   IndicatorShortName("ATR(", InpATRPeriod, ")");
   
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
   if(rates_total < InpATRPeriod + 1)
      return(0);
   
   //--- calculate ATR
   for(int i = start; i < rates_total && !IsStopped(); i++)
     {
      if(i < InpATRPeriod)
        {
         ExtATRBuffer[i] = 0.0;
         continue;
        }
      
      //--- calculate true range for current bar
      double true_range = GetTrueRange(high, low, close, i);
      
      if(i == InpATRPeriod)
        {
         //--- first ATR calculation - simple average of true ranges
         double sum_tr = 0.0;
         for(int j = 0; j <= InpATRPeriod; j++)
            sum_tr += GetTrueRange(high, low, close, i - j);
         
         ExtATRBuffer[i] = sum_tr / (InpATRPeriod + 1);
        }
      else
        {
         //--- subsequent calculations - smoothed average
         ExtATRBuffer[i] = (ExtATRBuffer[i-1] * (InpATRPeriod - 1) + true_range) / InpATRPeriod;
        }
     }
   
   //--- return number of calculated bars
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Calculate True Range                                             |
//+------------------------------------------------------------------+
double GetTrueRange(const double &high[],
                    const double &low[],
                    const double &close[],
                    int index)
  {
   double method1 = high[index] - low[index];                    // High - Low
   double method2 = MathAbs(high[index] - close[index-1]);       // High - Previous Close
   double method3 = MathAbs(low[index] - close[index-1]);        // Low - Previous Close
   
   return(MathMax(MathMax(method1, method2), method3));
  }

//+------------------------------------------------------------------+
//| Get current ATR value                                            |
//+------------------------------------------------------------------+
double GetATRValue()
  {
   return(ExtATRBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Calculate stop loss in points based on ATR                       |
//+------------------------------------------------------------------+
double CalculateStopLossPoints(double atr_multiplier = 1.5)
  {
   return(ExtATRBuffer[0] * atr_multiplier);
  }

//+------------------------------------------------------------------+
//| Calculate position size based on ATR and risk percentage         |
//+------------------------------------------------------------------+
double CalculatePositionSize(double account_balance,
                            double risk_percent,
                            double atr_multiplier = 1.5)
  {
   if(ExtATRBuffer[0] == 0)
      return(0);
   
   double risk_amount = account_balance * (risk_percent / 100.0);
   double stop_loss_points = CalculateStopLossPoints(atr_multiplier);
   
   //--- convert points to monetary value (simplified - assumes 1 point = 1 pip)
   double point_value = MarketInfo(Symbol(), MODE_POINT);
   double stop_loss_value = stop_loss_points * point_value;
   
   if(stop_loss_value == 0)
      return(0);
   
   return(risk_amount / stop_loss_value);
  }

//+------------------------------------------------------------------+
//| Get volatility level based on ATR                                |
//+------------------------------------------------------------------+
int GetVolatilityLevel()
  {
   double current_atr = ExtATRBuffer[0];
   double average_atr = iMAOnArray(ExtATRBuffer, 0, 20, 0, MODE_SMA, 0);
   
   if(current_atr == 0 || average_atr == 0)
      return(0);
   
   double atr_ratio = current_atr / average_atr;
   
   if(atr_ratio < 0.8)
      return(1); // Low volatility
   else if(atr_ratio < 1.2)
      return(2); // Normal volatility
   else
      return(3); // High volatility
  }

//+------------------------------------------------------------------+
//| Check if current volatility is above threshold                   |
//+------------------------------------------------------------------+
bool IsVolatilityHigh(double threshold_multiplier = 1.5)
  {
   double average_atr = iMAOnArray(ExtATRBuffer, 0, 20, 0, MODE_SMA, 0);
   
   if(average_atr == 0)
      return(false);
   
   return(ExtATRBuffer[0] > average_atr * threshold_multiplier);
  }

//+------------------------------------------------------------------+
//| Check if current volatility is below threshold                   |
//+------------------------------------------------------------------+
bool IsVolatilityLow(double threshold_multiplier = 0.7)
  {
   double average_atr = iMAOnArray(ExtATRBuffer, 0, 20, 0, MODE_SMA, 0);
   
   if(average_atr == 0)
      return(false);
   
   return(ExtATRBuffer[0] < average_atr * threshold_multiplier);
  }

//+------------------------------------------------------------------+
//| Get ATR percentile rank                                          |
//+------------------------------------------------------------------+
double GetATRPercentile(int lookback_period = 100)
  {
   if(Bars < lookback_period)
      return(0);
   
   double current_atr = ExtATRBuffer[0];
   int rank = 0;
   int total = 0;
   
   for(int i = 0; i < lookback_period && i < Bars; i++)
     {
      if(ExtATRBuffer[i] <= current_atr)
         rank++;
      total++;
     }
   
   return((double)rank / total * 100.0);
  }

//+------------------------------------------------------------------+
//| Get ATR-based volatility contraction signal                      |
//+------------------------------------------------------------------+
bool IsVolatilityContraction(int contraction_period = 20)
  {
   if(Bars < contraction_period + 5)
      return(false);
   
   double current_atr = ExtATRBuffer[0];
   double average_atr = iMAOnArray(ExtATRBuffer, 0, contraction_period, 0, MODE_SMA, 0);
   
   if(average_atr == 0)
      return(false);
   
   //--- check if ATR is significantly below average
   return(current_atr < average_atr * 0.7);
  }