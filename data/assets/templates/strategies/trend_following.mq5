//+------------------------------------------------------------------+
//|                                           trend_following.mq5    |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| Trend Following Strategy Template                               |
//| Uses moving average crossovers and momentum confirmation        |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict

//--- Input parameters
input int      InpMAPeriodFast   = 10;     // Fast MA period
input int      InpMAPeriodSlow   = 20;     // Slow MA period
input int      InpMAType         = MODE_EMA; // MA type (0=SMA, 1=EMA, 2=SMMA, 3=LWMA)
input double   InpRiskPercent    = 1.0;     // Risk percentage per trade
input double   InpTakeProfitPips = 50;      // Take profit in pips
input double   InpStopLossPips   = 25;      // Stop loss in pips
input int      InpMaxSpread      = 30;      // Maximum allowed spread in points
input bool     InpUseTrailingStop = true;   // Use trailing stop
input int      InpTrailingStopPips = 20;    // Trailing stop distance in pips

//--- Global variables
int            ExtFastMAHandle   = 0;
int            ExtSlowMAHandle   = 0;
datetime       ExtLastTradeTime  = 0;
double         ExtPointValue     = 0;
int            ExtDigits         = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Initialize market info
   ExtPointValue = MarketInfo(Symbol(), MODE_POINT);
   ExtDigits = (int)MarketInfo(Symbol(), MODE_DIGITS);
   
   //--- Create MA handles
   ExtFastMAHandle = iMA(Symbol(), Period(), InpMAPeriodFast, 0, InpMAType, PRICE_CLOSE);
   ExtSlowMAHandle = iMA(Symbol(), Period(), InpMAPeriodSlow, 0, InpMAType, PRICE_CLOSE);
   
   if(ExtFastMAHandle == INVALID_HANDLE || ExtSlowMAHandle == INVALID_HANDLE)
     {
      Print("Failed to create MA handles");
      return(INIT_FAILED);
     }
   
   //--- Check symbol availability
   if(!SymbolInfoInteger(Symbol(), SYMBOL_TRADE_ALLOWED))
     {
      Print("Trading not allowed for symbol: ", Symbol());
      return(INIT_FAILED);
     }
   
   //--- Set strategy name
   Comment("Trend Following Strategy Active\n",
           "Fast MA: ", InpMAPeriodFast, " ", GetMATypeName(InpMAType), "\n",
           "Slow MA: ", InpMAPeriodSlow, " ", GetMATypeName(InpMAType), "\n",
           "Risk: ", DoubleToString(InpRiskPercent, 2), "%");
   
   Print("Trend Following Strategy initialized successfully");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   //--- Release indicator handles
   if(ExtFastMAHandle != 0)
      IndicatorRelease(ExtFastMAHandle);
   if(ExtSlowMAHandle != 0)
      IndicatorRelease(ExtSlowMAHandle);
   
   Comment("");
   Print("Trend Following Strategy deinitialized");
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- Check if market is closed
   if(IsTradeAllowed() == false)
      return;
   
   //--- Check spread
   if(GetSpread() > InpMaxSpread * ExtPointValue)
      return;
   
   //--- Update trailing stops
   if(InpUseTrailingStop)
      UpdateTrailingStops();
   
   //--- Check for new signals
   CheckForTradingSignals();
  }

//+------------------------------------------------------------------+
//| Check for trading signals                                        |
//+------------------------------------------------------------------+
void CheckForTradingSignals()
  {
   //--- Get current MA values
   double fast_ma = iMA(Symbol(), Period(), InpMAPeriodFast, 0, InpMAType, PRICE_CLOSE, 0);
   double slow_ma = iMA(Symbol(), Period(), InpMAPeriodSlow, 0, InpMAType, PRICE_CLOSE, 0);
   double prev_fast_ma = iMA(Symbol(), Period(), InpMAPeriodFast, 0, InpMAType, PRICE_CLOSE, 1);
   double prev_slow_ma = iMA(Symbol(), Period(), InpMAPeriodSlow, 0, InpMAType, PRICE_CLOSE, 1);
   
   if(fast_ma == 0 || slow_ma == 0 || prev_fast_ma == 0 || prev_slow_ma == 0)
      return;
   
   //--- Check for bullish crossover (fast MA crosses above slow MA)
   if(prev_fast_ma <= prev_slow_ma && fast_ma > slow_ma)
     {
      if(!HasOpenPositions(ORDER_TYPE_BUY))
        {
         CloseOppositePositions(ORDER_TYPE_SELL);
         ExecuteBuyOrder();
        }
     }
   //--- Check for bearish crossover (fast MA crosses below slow MA)
   else if(prev_fast_ma >= prev_slow_ma && fast_ma < slow_ma)
     {
      if(!HasOpenPositions(ORDER_TYPE_SELL))
        {
         CloseOppositePositions(ORDER_TYPE_BUY);
         ExecuteSellOrder();
        }
     }
  }

//+------------------------------------------------------------------+
//| Execute buy order                                                |
//+------------------------------------------------------------------+
void ExecuteBuyOrder()
  {
   //--- Calculate position size
   double lot_size = CalculatePositionSize(InpRiskPercent);
   if(lot_size <= 0)
      return;
   
   //--- Calculate SL and TP
   double stop_loss = NormalizeDouble(Bid - InpStopLossPips * ExtPointValue, ExtDigits);
   double take_profit = NormalizeDouble(Bid + InpTakeProfitPips * ExtPointValue, ExtDigits);
   
   //--- Send order
   MqlTradeRequest request;
   MqlTradeResult result;
   
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = Symbol();
   request.volume = lot_size;
   request.type = ORDER_TYPE_BUY;
   request.price = Ask;
   request.sl = stop_loss;
   request.tp = take_profit;
   request.deviation = 10;
   request.type_filling = ORDER_FILLING_IOC;
   
   if(!OrderSend(request, result))
     {
      Print("OrderSend failed: ", result.retcode, " - ", result.comment);
      return;
     }
   
   ExtLastTradeTime = TimeCurrent();
   Print("Buy order executed. Ticket: ", result.order, ", Lot size: ", DoubleToString(lot_size, 2));
  }

//+------------------------------------------------------------------+
//| Execute sell order                                               |
//+------------------------------------------------------------------+
void ExecuteSellOrder()
  {
   //--- Calculate position size
   double lot_size = CalculatePositionSize(InpRiskPercent);
   if(lot_size <= 0)
      return;
   
   //--- Calculate SL and TP
   double stop_loss = NormalizeDouble(Ask + InpStopLossPips * ExtPointValue, ExtDigits);
   double take_profit = NormalizeDouble(Ask - InpTakeProfitPips * ExtPointValue, ExtDigits);
   
   //--- Send order
   MqlTradeRequest request;
   MqlTradeResult result;
   
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = Symbol();
   request.volume = lot_size;
   request.type = ORDER_TYPE_SELL;
   request.price = Bid;
   request.sl = stop_loss;
   request.tp = take_profit;
   request.deviation = 10;
   request.type_filling = ORDER_FILLING_IOC;
   
   if(!OrderSend(request, result))
     {
      Print("OrderSend failed: ", result.retcode, " - ", result.comment);
      return;
     }
   
   ExtLastTradeTime = TimeCurrent();
   Print("Sell order executed. Ticket: ", result.order, ", Lot size: ", DoubleToString(lot_size, 2));
  }

//+------------------------------------------------------------------+
//| Calculate position size based on risk percentage                 |
//+------------------------------------------------------------------+
double CalculatePositionSize(double risk_percent)
  {
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (risk_percent / 100.0);
   
   //--- Calculate stop loss value in account currency
   double sl_value = InpStopLossPips * ExtPointValue;
   if(sl_value == 0)
      return(0);
   
   //--- Convert to lot size (simplified calculation)
   double lot_size = risk_amount / sl_value / 100000.0; // Assuming standard lot = 100,000 units
   
   //--- Apply lot step and limits
   double lot_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
   
   lot_size = MathFloor(lot_size / lot_step) * lot_step;
   lot_size = MathMax(lot_size, min_lot);
   lot_size = MathMin(lot_size, max_lot);
   
   return(lot_size);
  }

//+------------------------------------------------------------------+
//| Check if there are open positions of specific type               |
//+------------------------------------------------------------------+
bool HasOpenPositions(ENUM_ORDER_TYPE order_type)
  {
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(Symbol() == PositionGetString(i, POSITION_SYMBOL) && 
         order_type == (ENUM_ORDER_TYPE)PositionGetInteger(i, POSITION_TYPE))
         return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Close opposite positions                                         |
//+------------------------------------------------------------------+
void CloseOppositePositions(ENUM_ORDER_TYPE opposite_type)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(Symbol() == PositionGetString(i, POSITION_SYMBOL) && 
         opposite_type == (ENUM_ORDER_TYPE)PositionGetInteger(i, POSITION_TYPE))
        {
         //--- Close position
         MqlTradeRequest request;
         MqlTradeResult result;
         
         ZeroMemory(request);
         ZeroMemory(result);
         
         request.action = TRADE_ACTION_POSITION_CLOSE;
         request.position = PositionGetInteger(i, POSITION_TICKET);
         request.symbol = Symbol();
         request.deviation = 10;
         
         if(!OrderSend(request, result))
           {
            Print("Failed to close position: ", result.retcode, " - ", result.comment);
           }
         else
           {
            Print("Opposite position closed. Ticket: ", result.order);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Update trailing stops for open positions                        |
//+------------------------------------------------------------------+
void UpdateTrailingStops()
  {
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(Symbol() != PositionGetString(i, POSITION_SYMBOL))
         continue;
      
      double current_sl = PositionGetDouble(i, POSITION_SL);
      double price = PositionGetDouble(i, POSITION_PRICE_OPEN);
      ulong ticket = PositionGetInteger(i, POSITION_TICKET);
      ENUM_POSITION_TYPE position_type = (ENUM_POSITION_TYPE)PositionGetInteger(i, POSITION_TYPE);
      
      double new_sl = 0;
      
      if(position_type == POSITION_TYPE_BUY)
        {
         double trail_level = Bid - InpTrailingStopPips * ExtPointValue;
         if(current_sl == 0 || trail_level > current_sl)
            new_sl = trail_level;
        }
      else if(position_type == POSITION_TYPE_SELL)
        {
         double trail_level = Ask + InpTrailingStopPips * ExtPointValue;
         if(current_sl == 0 || trail_level < current_sl)
            new_sl = trail_level;
        }
      
      if(new_sl != 0 && new_sl != current_sl)
        {
         MqlTradeRequest request;
         MqlTradeResult result;
         
         ZeroMemory(request);
         ZeroMemory(result);
         
         request.action = TRADE_ACTION_SLTP;
         request.position = ticket;
         request.symbol = Symbol();
         request.sl = NormalizeDouble(new_sl, ExtDigits);
         
         if(!OrderSend(request, result))
           {
            Print("Failed to update trailing stop: ", result.retcode, " - ", result.comment);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Get spread in points                                             |
//+------------------------------------------------------------------+
double GetSpread()
  {
   return(MarketInfo(Symbol(), MODE_ASK) - MarketInfo(Symbol(), MODE_BID));
  }

//+------------------------------------------------------------------+
//| Get MA type name                                                 |
//+------------------------------------------------------------------+
string GetMATypeName(int ma_type)
  {
   switch(ma_type)
     {
      case MODE_SMA:  return("SMA");
      case MODE_EMA:  return("EMA");
      case MODE_SMMA: return("SMMA");
      case MODE_LWMA: return("LWMA");
      default:        return("Unknown");
     }
  }

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool IsTradeAllowed()
  {
   //--- Check if market is open
   if(!MarketIsOpen(Symbol(), true))
      return(false);
   
   //--- Check for weekends (if applicable)
   int day = TimeDayOfWeek(TimeCurrent());
   if(day == 0 || day == 6) // Sunday or Saturday
      return(false);
   
   return(true);
  }