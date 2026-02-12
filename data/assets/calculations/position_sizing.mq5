//+------------------------------------------------------------------+
//|                                      position_sizing.mq5         |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| Position sizing calculation methods for risk management         |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict

//+------------------------------------------------------------------+
//| Calculate fixed lot size based on account percentage            |
//+------------------------------------------------------------------+
double CalculateFixedLots(double risk_percent)
  {
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (risk_percent / 100.0);
   
   //--- Convert to standard lots (100,000 units per lot)
   double lot_size = risk_amount / 100000.0;
   
   //--- Apply symbol limits
   double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   
   lot_size = MathMax(lot_size, min_lot);
   lot_size = MathMin(lot_size, max_lot);
   lot_size = MathFloor(lot_size / lot_step) * lot_step;
   
   return(lot_size);
  }

//+------------------------------------------------------------------+
//| Calculate position size based on stop loss in pips              |
//+------------------------------------------------------------------+
double CalculatePositionSizeBySL(double account_balance,
                                double risk_percent,
                                double stop_loss_pips,
                                double point_value = 0)
  {
   if(point_value == 0)
      point_value = MarketInfo(Symbol(), MODE_POINT);
   
   double risk_amount = account_balance * (risk_percent / 100.0);
   double sl_value = stop_loss_pips * point_value;
   
   if(sl_value == 0)
      return(0);
   
   //--- Convert to lot size
   double lot_size = risk_amount / sl_value / 100000.0;
   
   //--- Apply symbol limits
   double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   
   lot_size = MathMax(lot_size, min_lot);
   lot_size = MathMin(lot_size, max_lot);
   lot_size = MathFloor(lot_size / lot_step) * lot_step;
   
   return(lot_size);
  }

//+------------------------------------------------------------------+
//| Calculate position size based on ATR                            |
//+------------------------------------------------------------------+
double CalculateATRPositionSize(double account_balance,
                               double risk_percent,
                               double atr_value,
                               double atr_multiplier = 1.5,
                               double point_value = 0)
  {
   if(atr_value == 0)
      return(0);
   
   if(point_value == 0)
      point_value = MarketInfo(Symbol(), MODE_POINT);
   
   double risk_amount = account_balance * (risk_percent / 100.0);
   double sl_value = atr_value * atr_multiplier * point_value;
   
   if(sl_value == 0)
      return(0);
   
   double lot_size = risk_amount / sl_value / 100000.0;
   
   //--- Apply symbol limits
   double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   
   lot_size = MathMax(lot_size, min_lot);
   lot_size = MathMin(lot_size, max_lot);
   lot_size = MathFloor(lot_size / lot_step) * lot_step;
   
   return(lot_size);
  }

//+------------------------------------------------------------------+
//| Calculate Kelly Criterion position size                         |
//+------------------------------------------------------------------+
double CalculateKellyPositionSize(double account_balance,
                                 double win_rate,
                                 double win_loss_ratio,
                                 double kelly_fraction = 1.0,
                                 double max_risk_percent = 5.0)
  {
   //--- Kelly Criterion formula: f = (bp - q) / b
   // where f = fraction of capital, b = win/loss ratio, p = win rate, q = loss rate
   double b = win_loss_ratio;
   double p = win_rate / 100.0;
   double q = 1.0 - p;
   
   if(b == 0)
      return(0);
   
   double kelly_fraction_full = (b * p - q) / b;
   
   //--- Apply fraction for more conservative approach
   double kelly_fraction_adjusted = kelly_fraction_full * kelly_fraction;
   
   //--- Limit maximum risk
   kelly_fraction_adjusted = MathMin(kelly_fraction_adjusted, max_risk_percent / 100.0);
   kelly_fraction_adjusted = MathMax(kelly_fraction_adjusted, 0);
   
   double lot_size = account_balance * kelly_fraction_adjusted / 100000.0;
   
   //--- Apply symbol limits
   double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   
   lot_size = MathMax(lot_size, min_lot);
   lot_size = MathMin(lot_size, max_lot);
   lot_size = MathFloor(lot_size / lot_step) * lot_step;
   
   return(lot_size);
  }

//+------------------------------------------------------------------+
//| Calculate position size based on volatility                     |
//+------------------------------------------------------------------+
double CalculateVolatilityPositionSize(double account_balance,
                                      double risk_percent,
                                      double volatility_percent,
                                      double max_volatility = 2.0)
  {
   //--- Reduce position size in high volatility
   double volatility_factor = MathMin(volatility_percent / max_volatility, 1.0);
   double adjusted_risk = risk_percent * (1.0 - volatility_factor);
   
   //--- Use fixed lot calculation with adjusted risk
   return(CalculateFixedLots(adjusted_risk));
  }

//+------------------------------------------------------------------+
//| Calculate pyramid position sizing                               |
//+------------------------------------------------------------------+
double CalculatePyramidPositionSize(double account_balance,
                                   double risk_percent,
                                   int pyramid_level,
                                   double base_lot_size)
  {
   //--- Pyramid sizing: reduce position size for additional entries
   double reduction_factor = 1.0 / (pyramid_level + 1);
   double adjusted_risk = risk_percent * reduction_factor;
   
   if(base_lot_size > 0)
     {
      //--- Use base lot size as reference
      return(base_lot_size * reduction_factor);
     }
   else
     {
      //--- Calculate from scratch
      return(CalculateFixedLots(adjusted_risk));
     }
  }

//+------------------------------------------------------------------+
//| Get maximum allowed position size for symbol                    |
//+------------------------------------------------------------------+
double GetMaxPositionSize()
  {
   return(SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX));
  }

//+------------------------------------------------------------------+
//| Get minimum allowed position size for symbol                    |
//+------------------------------------------------------------------+
double GetMinPositionSize()
  {
   return(SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN));
  }

//+------------------------------------------------------------------+
//| Get position size step for symbol                               |
//+------------------------------------------------------------------+
double GetPositionSizeStep()
  {
   return(SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP));
  }

//+------------------------------------------------------------------+
//| Round position size to valid lot step                           |
//+------------------------------------------------------------------+
double RoundToLotStep(double lot_size)
  {
   double lot_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   return(MathFloor(lot_size / lot_step) * lot_step);
  }