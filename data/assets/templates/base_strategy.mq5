//+------------------------------------------------------------------+
//|                                                BaseStrategy.mq5  |
//|                                        Copyright 2026, QuantMind |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, QuantMind"
#property link      "https://quantmind.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

//--- Input parameters
input double   RiskPercent = 1.0;
input int      MagicNumber = 123456;

//--- Global variables
CTrade         trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(MagicNumber);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Check for entry conditions
   if(PositionsTotal() == 0) {
      // Entry logic here
   }
  }
//+------------------------------------------------------------------+
