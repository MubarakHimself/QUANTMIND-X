//+------------------------------------------------------------------+
//|                                         mean_reversion.mq5       |
//|                        QuantMindX Shared Assets Library          |
//|                                                                  |
//| Mean Reversion Strategy Template                                |
//| Uses Bollinger Bands for overbought/oversold conditions         |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.0.0"
#property strict

//--- Input parameters
input int      InpBBPeriod       = 20;     // Bollinger Bands period
input double   InpBBDeviations   = 2.0;     // Bollinger Bands deviations
input double   InpRiskPercent    = 1.0;     // Risk percentage per trade
input int      InpOverbought     = 70;      // RSI overbought level
input int      InpOversold       = 30;      // RSI oversold level
input int      InpRSIPeriod      = 14;      // RSI period
input double   InpTakeProfitPips = 30;      // Take profit in pips
input double   InpStopLossPips   = 15;      // Stop loss in pips
input int      InpMaxSpread      = 30;      // Maximum allowed spread in points
input int      InpMinConsecutive = 3;       // Minimum consecutive candles for reversal

//--- Global variables
int            ExtBBHandle       = 0;
int            ExtRSIHandle      = 0;
datetime       ExtLastTradeTime  = 0;
double         ExtPointValue     = 0;
int            ExtDigits         = 0;
int            ExtConsecutiveCount = 0;
ENUM_ORDER_TYPE ExtLastSignal    = ORDER_TYPE_BUY;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Initialize market info
   ExtPointValue = MarketInfo(Symbol(), MODE_POINT);
   ExtDigits = (int)MarketInfo(Symbol(), MODE_DIGITS);
   
   //--- Create indicator handles
   ExtBBHandle = iBands(Symbol(), Period(), InpBBPeriod, InpBBDeviations, 0, PRICE_CLOSE);
   ExtRSIHandle = iRSI(Symbol(), Period(), InpRSIPeriod, PRICE_CLOSE);
   
   if(ExtBBHandle == INVALID_HANDLE || ExtRSIHandle == INVALID_HANDLE)
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
   Comment("Mean Reversion Strategy Active\n",
           "BB Period: ", InpBBPeriod, ", Deviations: ", DoubleToString(InpBBDeviations, 1), "\n",
           "RSI Period: ", InpRSIPeriod, " (", InpOversold, "/", InpOverbought, ")\n",
           "Risk: ", DoubleToString(InpRiskPercent, 2), "%");
   
   Print("Mean Reversion Strategy initialized successfully");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   //--- Release indicator handles
   if(ExtBBHandle != 0)
      IndicatorRelease(ExtBBHandle);
   if(ExtRSIHandle != 0)
      IndicatorRelease(ExtRSIHandle);
   
   Comment("");
   Print("Mean Reversion Strategy deinitialized");
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
   
   //--- Check for new signals
   CheckForTradingSignals();
  }

//+------------------------------------------------------------------+
//| Check for trading signals                                        |
//+------------------------------------------------------------------+
void CheckForTradingSignals()
  {
   //--- Get current indicator values
   double upper_band = iBands(Symbol(), Period(), InpBBPeriod, InpBBDeviations, 0, PRICE_CLOSE, 0, MODE_UPPER);
   double lower_band = iBands(Symbol(), Period(), InpBBPeriod, InpBBDeviations, 0, PRICE_CLOSE, 0, MODE_LOWER);
   double middle_band = iBands(Symbol(), Period(), InpBBPeriod, InpBBDeviations, 0, PRICE_CLOSE, 0, MODE_MAIN);
   double rsi_value = iRSI(Symbol(), Period(), InpRSIPeriod, PRICE_CLOSE, 0);
   double close_price = Close[0];
   
   if(upper_band == 0 || lower_band == 0 || middle_band == 0 || rsi_value == 0)
      return;
   
   //--- Check for oversold condition with price confirmation
   if(close_price <= lower_band && rsi_value <= InpOversold)
     {
      if(ExtLastSignal != ORDER_TYPE_BUY)
        {
         ExtConsecutiveCount = 1;
         ExtLastSignal = ORDER_TYPE_BUY;
        }
      else
        {
         ExtConsecutiveCount++;
        }
      
      //--- Execute buy if we have enough consecutive signals
      if(ExtConsecutiveCount >= InpMinConsecutive && !HasOpenPositions(ORDER_TYPE_BUY))
        {
         CloseOppositePositions(ORDER_TYPE_SELL);
         ExecuteBuyOrder();
         ExtConsecutiveCount = 0;
        }
     }
   //--- Check for overbought condition with price confirmation
   else if(close_price >= upper_band && rsi_value >= InpOverbought)
     {
      if(ExtLastSignal != ORDER_TYPE_SELL)
        {
         ExtConsecutiveCount = 1;
         ExtLastSignal = ORDER_TYPE_SELL;
        }
      else
        {
         ExtConsecutiveCount++;
        }
      
      //--- Execute sell if we have enough consecutive signals
      if(ExtConsecutiveCount >= InpMinConsecutive && !HasOpenPositions(ORDER_TYPE_SELL))
        {
         CloseOppositePositions(ORDER_TYPE_BUY);
         ExecuteSellOrder();
         ExtConsecutiveCount = 0;
        }
     }
   else
     {
      //--- Reset counter when conditions change
      ExtConsecutiveCount = 0;
      ExtLastSignal = (ENUM_ORDER_TYPE)(-1); // Invalid order type
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
   
   //--- Ensure SL is below lower band for mean reversion logic
   double lower_band = iBands(Symbol(), Period(), InpBBPeriod, InpBBDeviations, 0, PRICE_CLOSE, 0, MODE_LOWER);
   if(stop_loss > lower_band)
      stop_loss = NormalizeDouble(lower_band - 5 * ExtPointValue, ExtDigits);
   
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
   
   //--- Ensure SL is above upper band for mean reversion logic
   double upper_band = iBands(Symbol(), Period(), InpBBPeriod, InpBBDeviations, 0, PRICE_CLOSE, 0, MODE_UPPER);
   if(stop_loss < upper_band)
      stop_loss = NormalizeDouble(upper_band + 5 * ExtPointValue, ExtDigits);
   
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