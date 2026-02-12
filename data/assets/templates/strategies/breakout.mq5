//+------------------------------------------------------------------+
//|                                            breakout.mq5          |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| Breakout Strategy Template                                      |
//| Uses range breakouts with volume confirmation                   |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict

//--- Input parameters
input int      InpRangePeriod    = 20;     // Range period for high/low calculation
input double   InpRiskPercent    = 1.0;     // Risk percentage per trade
input double   InpTakeProfitPips = 60;      // Take profit in pips
input double   InpStopLossPips   = 20;      // Stop loss in pips
input int      InpMinVolume      = 100;     // Minimum volume for confirmation
input double   InpVolumeMultiplier = 1.5;   // Volume multiplier for confirmation
input int      InpMaxSpread      = 30;      // Maximum allowed spread in points
input int      InpCooldownPeriod = 3600;    // Cooldown period in seconds between trades

//--- Global variables
datetime       ExtLastTradeTime  = 0;
double         ExtPointValue     = 0;
int            ExtDigits         = 0;
double         ExtRangeHigh      = 0;
double         ExtRangeLow       = 0;
bool           ExtRangeCalculated = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Initialize market info
   ExtPointValue = MarketInfo(Symbol(), MODE_POINT);
   ExtDigits = (int)MarketInfo(Symbol(), MODE_DIGITS);
   
   //--- Check symbol availability
   if(!SymbolInfoInteger(Symbol(), SYMBOL_TRADE_ALLOWED))
     {
      Print("Trading not allowed for symbol: ", Symbol());
      return(INIT_FAILED);
     }
   
   //--- Calculate initial range
   CalculateRange();
   
   //--- Set strategy name
   Comment("Breakout Strategy Active\n",
           "Range Period: ", InpRangePeriod, " bars\n",
           "Risk: ", DoubleToString(InpRiskPercent, 2), "%\n",
           "Cooldown: ", InpCooldownPeriod, " seconds");
   
   Print("Breakout Strategy initialized successfully");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Comment("");
   Print("Breakout Strategy deinitialized");
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
   
   //--- Check cooldown period
   if(TimeCurrent() - ExtLastTradeTime < InpCooldownPeriod)
      return;
   
   //--- Recalculate range if needed
   if(Bars > InpRangePeriod && !ExtRangeCalculated)
      CalculateRange();
   
   //--- Check for breakout signals
   CheckForBreakoutSignals();
  }

//+------------------------------------------------------------------+
//| Calculate range high/low                                         |
//+------------------------------------------------------------------+
void CalculateRange()
  {
   if(Bars < InpRangePeriod)
      return;
   
   ExtRangeHigh = High[Highest(NULL, 0, MODE_HIGH, InpRangePeriod, 1)];
   ExtRangeLow = Low[Lowest(NULL, 0, MODE_LOW, InpRangePeriod, 1)];
   ExtRangeCalculated = true;
   
   Print("Range calculated - High: ", DoubleToString(ExtRangeHigh, ExtDigits), 
         ", Low: ", DoubleToString(ExtRangeLow, ExtDigits));
  }

//+------------------------------------------------------------------+
//| Check for breakout signals                                       |
//+------------------------------------------------------------------+
void CheckForBreakoutSignals()
  {
   if(!ExtRangeCalculated)
      return;
   
   double current_high = High[0];
   double current_low = Low[0];
   double current_close = Close[0];
   double current_volume = Volume[0];
   
   //--- Check for bullish breakout (price breaks above range high)
   if(current_high > ExtRangeHigh)
     {
      //--- Volume confirmation
      if(IsVolumeConfirmed())
        {
         if(!HasOpenPositions(ORDER_TYPE_BUY))
           {
            CloseOppositePositions(ORDER_TYPE_SELL);
            ExecuteBuyOrder();
            //--- Reset range for next breakout
            CalculateRange();
           }
        }
     }
   //--- Check for bearish breakout (price breaks below range low)
   else if(current_low < ExtRangeLow)
     {
      //--- Volume confirmation
      if(IsVolumeConfirmed())
        {
         if(!HasOpenPositions(ORDER_TYPE_SELL))
           {
            CloseOppositePositions(ORDER_TYPE_BUY);
            ExecuteSellOrder();
            //--- Reset range for next breakout
            CalculateRange();
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Check if volume confirms the breakout                            |
//+------------------------------------------------------------------+
bool IsVolumeConfirmed()
  {
   //--- Simple volume confirmation - current volume vs average
   double avg_volume = iMA(NULL, 0, InpRangePeriod, 0, MODE_SMA, VOLUME_TICK, 0);
   double current_volume = Volume[0];
   
   return(current_volume >= avg_volume * InpVolumeMultiplier);
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
   double stop_loss = NormalizeDouble(ExtRangeHigh - InpStopLossPips * ExtPointValue, ExtDigits);
   double take_profit = NormalizeDouble(Ask + InpTakeProfitPips * ExtPointValue, ExtDigits);
   
   //--- Ensure SL is reasonable
   if(stop_loss >= Bid)
      stop_loss = NormalizeDouble(Bid - InpStopLossPips * ExtPointValue, ExtDigits);
   
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
   double stop_loss = NormalizeDouble(ExtRangeLow + InpStopLossPips * ExtPointValue, ExtDigits);
   double take_profit = NormalizeDouble(Bid - InpTakeProfitPips * ExtPointValue, ExtDigits);
   
   //--- Ensure SL is reasonable
   if(stop_loss <= Ask)
      stop_loss = NormalizeDouble(Ask + InpStopLossPips * ExtPointValue, ExtDigits);
   
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
   
   //--- Convert to lot size
   double lot_size = risk_amount / sl_value / 100000.0;
   
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
//| Get spread in points                                             |
//+------------------------------------------------------------------+
double GetSpread()
  {
   return(MarketInfo(Symbol(), MODE_ASK) - MarketInfo(Symbol(), MODE_BID));
  }

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool IsTradeAllowed()
  {
   //--- Check if market is open
   if(!MarketIsOpen(Symbol(), true))
      return(false);
   
   //--- Check for weekends
   int day = TimeDayOfWeek(TimeCurrent());
   if(day == 0 || day == 6)
      return(false);
   
   return(true);
  }