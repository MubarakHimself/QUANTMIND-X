//+------------------------------------------------------------------+
//|                                            scalping.mq5          |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| Scalping Strategy Template                                      |
//| High-frequency strategy with tight stops and quick profits      |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict

//--- Input parameters
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M1; // Strategy timeframe (should be M1-M5)
input int      InpMAPeriod       = 9;          // Moving average period
input int      InpRSIPeriod      = 7;          // RSI period
input int      InpOverbought     = 80;         // RSI overbought level
input int      InpOversold       = 20;         // RSI oversold level
input double   InpRiskPercent    = 0.5;        // Risk percentage per trade (lower for scalping)
input double   InpPipTarget      = 5;          // Target profit in pips
input double   InpStopLossPips   = 3;          // Stop loss in pips
input int      InpMaxSpread      = 10;         // Maximum allowed spread in points
input int      InpMaxPositions   = 3;          // Maximum concurrent positions
input int      InpCooldownTicks  = 5;          // Cooldown in ticks between trades

//--- Global variables
int            ExtMAHandle       = 0;
int            ExtRSIHandle      = 0;
datetime       ExtLastTradeTime  = 0;
double         ExtPointValue     = 0;
int            ExtDigits         = 0;
int            ExtTickCounter    = 0;
int            ExtConsecutiveWins = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Check timeframe (scalping should use M1-M5)
   if(Period() > PERIOD_M5)
     {
      Print("Warning: Scalping strategy works best on M1-M5 timeframes. Current: ", Period());
     }
   
   //--- Initialize market info
   ExtPointValue = MarketInfo(Symbol(), MODE_POINT);
   ExtDigits = (int)MarketInfo(Symbol(), MODE_DIGITS);
   
   //--- Create indicator handles
   ExtMAHandle = iMA(Symbol(), Period(), InpMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   ExtRSIHandle = iRSI(Symbol(), Period(), InpRSIPeriod, PRICE_CLOSE);
   
   if(ExtMAHandle == INVALID_HANDLE || ExtRSIHandle == INVALID_HANDLE)
     {
      Print("Failed to create indicator handles");
      return(INIT_FAILED);
     }
   
   //--- Check symbol availability
   if(!SymbolInfoInteger(Symbol(), SYMBOL_TRADE_ALLOWED))
     {
      Print("Trading not allowed for symbol: ", Symbol());
      return(INIT_FAILED);
     }
   
   //--- Set strategy name
   Comment("Scalping Strategy Active\n",
           "Timeframe: ", Period(), "\n",
           "Target: ", DoubleToString(InpPipTarget, 1), " pips\n",
           "Risk: ", DoubleToString(InpRiskPercent, 2), "%");
   
   Print("Scalping Strategy initialized successfully");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   //--- Release indicator handles
   if(ExtMAHandle != 0)
      IndicatorRelease(ExtMAHandle);
   if(ExtRSIHandle != 0)
      IndicatorRelease(ExtRSIHandle);
   
   Comment("");
   Print("Scalping Strategy deinitialized");
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   ExtTickCounter++;
   
   //--- Check if market is closed
   if(IsTradeAllowed() == false)
      return;
   
   //--- Check spread
   if(GetSpread() > InpMaxSpread * ExtPointValue)
      return;
   
   //--- Check cooldown
   if(ExtTickCounter < InpCooldownTicks)
      return;
   
   //--- Check maximum positions
   if(PositionsTotal() >= InpMaxPositions)
      return;
   
   //--- Check for new signals
   CheckForScalpingSignals();
  }

//+------------------------------------------------------------------+
//| Check for scalping signals                                       |
//+------------------------------------------------------------------+
void CheckForScalpingSignals()
  {
   //--- Get current indicator values
   double ma_value = iMA(Symbol(), Period(), InpMAPeriod, 0, MODE_EMA, PRICE_CLOSE, 0);
   double rsi_value = iRSI(Symbol(), Period(), InpRSIPeriod, PRICE_CLOSE, 0);
   double close_price = Close[0];
   double open_price = Open[0];
   
   if(ma_value == 0 || rsi_value == 0)
      return;
   
   //--- Bullish scalping setup
   if(close_price > ma_value && rsi_value > InpOversold && rsi_value < 50)
     {
      //--- Price action confirmation (bullish candle)
      if(close_price > open_price)
        {
         if(!HasOpenPositions(ORDER_TYPE_BUY))
           {
            ExecuteBuyOrder();
            ExtTickCounter = 0;
           }
        }
     }
   //--- Bearish scalping setup
   else if(close_price < ma_value && rsi_value < InpOverbought && rsi_value > 50)
     {
      //--- Price action confirmation (bearish candle)
      if(close_price < open_price)
        {
         if(!HasOpenPositions(ORDER_TYPE_SELL))
           {
            ExecuteSellOrder();
            ExtTickCounter = 0;
           }
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
   
   //--- Calculate SL and TP (very tight for scalping)
   double stop_loss = NormalizeDouble(Bid - InpStopLossPips * ExtPointValue, ExtDigits);
   double take_profit = NormalizeDouble(Bid + InpPipTarget * ExtPointValue, ExtDigits);
   
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
   request.deviation = 3;
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
   
   //--- Calculate SL and TP (very tight for scalping)
   double stop_loss = NormalizeDouble(Ask + InpStopLossPips * ExtPointValue, ExtDigits);
   double take_profit = NormalizeDouble(Ask - InpPipTarget * ExtPointValue, ExtDigits);
   
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
   request.deviation = 3;
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
   
   //--- Check for major news (simplified - would need news calendar integration)
   int hour = TimeHour(TimeCurrent());
   if((hour == 13 && TimeMinute(TimeCurrent()) < 30) || // US news
      (hour == 8 && TimeMinute(TimeCurrent()) < 30))   // EU news
      return(false);
   
   return(true);
  }

//+------------------------------------------------------------------+
//| Event handler for position close                                 |
//+------------------------------------------------------------------+
void OnPositionChange()
  {
   //--- Check if any positions were closed
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(Symbol() == PositionGetString(i, POSITION_SYMBOL))
        {
         //--- Check if position was profitable
         double profit = PositionGetDouble(i, POSITION_PROFIT);
         if(profit > 0)
            ExtConsecutiveWins++;
         else
            ExtConsecutiveWins = 0;
         
         //--- Risk management: reduce position size after consecutive losses
         if(ExtConsecutiveWins >= 3)
           {
            Print("3 consecutive wins achieved. Resetting win counter.");
            ExtConsecutiveWins = 0;
           }
         break;
        }
     }
  }