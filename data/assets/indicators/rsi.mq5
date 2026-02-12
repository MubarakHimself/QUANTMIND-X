//+------------------------------------------------------------------+
//|                                                  rsi.mq5         |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| This file is part of the QuantMindX shared assets library.      |
//| It provides a robust implementation of the Relative Strength    |
//| Index (RSI) indicator for overbought/oversold analysis.         |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3

//--- plot RSI
#property indicator_label1  "RSI"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- plot Overbought
#property indicator_label2  "Overbought"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrRed
#property indicator_style2  STYLE_DOT
#property indicator_width2  1

//--- plot Oversold
#property indicator_label3  "Oversold"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrLimeGreen
#property indicator_style3  STYLE_DOT
#property indicator_width3  1

//--- input parameters
input int      InpRSIPeriod   = 14;        // RSI Period
input int      InpOverbought  = 70;        // Overbought Level
input int      InpOversold    = 30;        // Oversold Level
input int      InpAppliedPrice = PRICE_CLOSE; // Applied price

//--- indicator buffers
double         ExtRSIBuffer[];
double         ExtOverboughtBuffer[];
double         ExtOversoldBuffer[];

//--- global variables
double         ExtPosBuffer[];
double         ExtNegBuffer[];
int            ExtPeriod;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- check input parameters
   if(InpRSIPeriod <= 0)
     {
      Print("Wrong input parameter RSI Period = ", InpRSIPeriod);
      return(INIT_PARAMETERS_INCORRECT);
     }
   
   //--- set indicator buffers
   SetIndexBuffer(0, ExtRSIBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, ExtOverboughtBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, ExtOversoldBuffer, INDICATOR_DATA);
   
   //--- set drawing begin
   SetIndexDrawBegin(0, InpRSIPeriod);
   
   //--- set indicator short name
   IndicatorShortName("RSI(", InpRSIPeriod, ")");
   
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
   if(rates_total < InpRSIPeriod + 1)
      return(0);
   
   //--- calculate RSI
   for(int i = start; i < rates_total && !IsStopped(); i++)
     {
      if(i < InpRSIPeriod)
        {
         ExtRSIBuffer[i] = 0.0;
         ExtOverboughtBuffer[i] = InpOverbought;
         ExtOversoldBuffer[i] = InpOversold;
         continue;
        }
      
      //--- calculate price change
      double price_change = 0.0;
      if(i > 0)
         price_change = GetPrice(close, open, high, low, i, InpAppliedPrice) - 
                       GetPrice(close, open, high, low, i-1, InpAppliedPrice);
      
      //--- separate positive and negative changes
      double pos_change = price_change > 0 ? price_change : 0.0;
      double neg_change = price_change < 0 ? -price_change : 0.0;
      
      //--- calculate average gains and losses
      double avg_gain = 0.0;
      double avg_loss = 0.0;
      
      if(i == InpRSIPeriod)
        {
         //--- first calculation - simple average
         double sum_gain = 0.0;
         double sum_loss = 0.0;
         
         for(int j = 1; j <= InpRSIPeriod; j++)
           {
            double change = GetPrice(close, open, high, low, i-j+1, InpAppliedPrice) - 
                           GetPrice(close, open, high, low, i-j, InpAppliedPrice);
            
            if(change > 0)
               sum_gain += change;
            else if(change < 0)
               sum_loss += (-change);
           }
         
         avg_gain = sum_gain / InpRSIPeriod;
         avg_loss = sum_loss / InpRSIPeriod;
        }
      else
        {
         //--- subsequent calculations - smoothed average
         avg_gain = (ExtPosBuffer[i-1] * (InpRSIPeriod - 1) + pos_change) / InpRSIPeriod;
         avg_loss = (ExtNegBuffer[i-1] * (InpRSIPeriod - 1) + neg_change) / InpRSIPeriod;
        }
      
      //--- store averages for next iteration
      ExtPosBuffer[i] = avg_gain;
      ExtNegBuffer[i] = avg_loss;
      
      //--- calculate RSI
      if(avg_loss == 0.0)
         ExtRSIBuffer[i] = 100.0;
      else
        {
         double rs = avg_gain / avg_loss;
         ExtRSIBuffer[i] = 100.0 - (100.0 / (1.0 + rs));
        }
      
      //--- set overbought/oversold levels
      ExtOverboughtBuffer[i] = InpOverbought;
      ExtOversoldBuffer[i] = InpOversold;
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
//| Get current RSI value                                            |
//+------------------------------------------------------------------+
double GetRSIValue()
  {
   return(ExtRSIBuffer[0]);
  }

//+------------------------------------------------------------------+
//| Check if RSI is overbought                                       |
//+------------------------------------------------------------------+
bool IsOverbought()
  {
   return(ExtRSIBuffer[0] > InpOverbought);
  }

//+------------------------------------------------------------------+
//| Check if RSI is oversold                                         |
//+------------------------------------------------------------------+
bool IsOversold()
  {
   return(ExtRSIBuffer[0] < InpOversold);
  }

//+------------------------------------------------------------------+
//| Get RSI divergence signal                                        |
//+------------------------------------------------------------------+
int GetDivergenceSignal(int bars_back = 50)
  {
   //--- check minimum bars
   if(Bars < bars_back + 10)
      return(0);
   
   //--- look for bullish divergence (price makes lower low, RSI makes higher low)
   double price_low_1 = iLow(NULL, 0, 1);
   double price_low_2 = iLow(NULL, 0, iLowest(NULL, 0, MODE_LOW, bars_back, 2));
   double rsi_low_1 = ExtRSIBuffer[1];
   double rsi_low_2 = ExtRSIBuffer[iLowest(NULL, 0, MODE_LOW, bars_back, 2)];
   
   if(price_low_2 < price_low_1 && rsi_low_2 > rsi_low_1)
      return(1); // Bullish divergence
   
   //--- look for bearish divergence (price makes higher high, RSI makes lower high)
   double price_high_1 = iHigh(NULL, 0, 1);
   double price_high_2 = iHigh(NULL, 0, iHighest(NULL, 0, MODE_HIGH, bars_back, 2));
   double rsi_high_1 = ExtRSIBuffer[1];
   double rsi_high_2 = ExtRSIBuffer[iHighest(NULL, 0, MODE_HIGH, bars_back, 2)];
   
   if(price_high_2 > price_high_1 && rsi_high_2 < rsi_high_1)
      return(-1); // Bearish divergence
   
   return(0); // No divergence
  }