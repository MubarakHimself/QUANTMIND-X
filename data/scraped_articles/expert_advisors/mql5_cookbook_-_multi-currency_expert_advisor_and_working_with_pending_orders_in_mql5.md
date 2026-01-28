---
title: MQL5 Cookbook - Multi-Currency Expert Advisor and Working with Pending Orders in MQL5
url: https://www.mql5.com/en/articles/755
categories: Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:43:31.949502
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ugcyijmifprgabwrksifcmeesjbmgvod&ssn=1769093010489496007&ssn_dr=0&ssn_sr=0&fv_date=1769093010&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F755&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Cookbook%20-%20Multi-Currency%20Expert%20Advisor%20and%20Working%20with%20Pending%20Orders%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909301050883885&fz_uniq=5049332165352728978&sv=2552)

MetaTrader 5 / Examples


### Introduction

This time we are going to create a multi-currency Expert Advisor with a trading algorithm based on work with the pending orders Buy Stop and Sell Stop. The pattern we are going to create will be designed for the intra-day trade/tests. The article considers the following matters:

- Trading in a specified time range. Let's create a feature that will allow us to set up the time of the beginning and the end of trading. For instance, it can be the time of the European or American trading sessions. For sure there will be an opportunity to find the most suitable time range when optimizing parameters of the Expert Advisor.
- Placing/modifying/deleting pending orders.
- Processing of trade events: checking if the last position was closed at Take Profit or Stop Loss and control over the history of the deals for each symbol.

### Expert Advisor Development

We are going to use the code from the article [MQL5 Cookbook: Multi-Currency Expert Advisor - Simple, Neat and Quick Approach](https://www.mql5.com/en/articles/648 "Article MQL5 Cookbook: Multi-Currency Expert Advisor - Simple, Neat and Quick Approach.") as a template. Though the essential structure of the pattern will remain the same, some significant changes will be introduced. The Expert Advisor will be designed for the intra-day trade, however, this mode could be switched off should the necessity arise. Pending orders, in such case, will always be placed immediately (on New Bar event) if a position has been closed.

Let's start with the external parameters of the expert advisor. At first we will create a new enumeration **ENUM\_HOURS** in the include file **Enums.mqh**. The number of identifiers in this enumeration is equal to the number of hours in a day:

```
//--- Hours Enumeration
enum ENUM_HOURS
  {
   h00 = 0,  // 00 : 00
   h01 = 1,  // 01 : 00
   h02 = 2,  // 02 : 00
   h03 = 3,  // 03 : 00
   h04 = 4,  // 04 : 00
   h05 = 5,  // 05 : 00
   h06 = 6,  // 06 : 00
   h07 = 7,  // 07 : 00
   h08 = 8,  // 08 : 00
   h09 = 9,  // 09 : 00
   h10 = 10, // 10 : 00
   h11 = 11, // 11 : 00
   h12 = 12, // 12 : 00
   h13 = 13, // 13 : 00
   h14 = 14, // 14 : 00
   h15 = 15, // 15 : 00
   h16 = 16, // 16 : 00
   h17 = 17, // 17 : 00
   h18 = 18, // 18 : 00
   h19 = 19, // 19 : 00
   h20 = 20, // 20 : 00
   h21 = 21, // 21 : 00
   h22 = 22, // 22 : 00
   h23 = 23  // 23 : 00
  };
```

Then in the list of external parameters we will create four parameters related to trading in a time range:

- **TradeInTimeRange**\- enabling/disabling the mode. As already mentioned, we are going to make work of the Expert Advisor possible not only within a certain time range but also around the clock, that is in a continuous mode.
- **StartTrade**\- the hour when a trading session starts. As soon as the server time is equal to this value, the Expert Advisor will place pending orders, providing that the TradeInTimeRange mode is on.
- **StopOpenOrders** \- the hour of the end of placing orders. When the server time is equal to this value the Expert Advisor will stop placing pending orders if a position is closed.
- **EndTrade**\- the hour when a trading session stops. Once the server time is equal to this value the Expert Advisor stops trading. An open position for the specified symbol will be closed and pending orders will be deleted.

The list of the external parameters will look as shown below. The given example is for two symbols. In the parameter **PendingOrder** we set up a distance from the current price in points.

```
//--- External parameters of the Expert Advisor
sinput long       MagicNumber       = 777;      // Magic number
sinput int        Deviation         = 10;       // Slippage
//---
sinput string delimeter_00=""; // --------------------------------
sinput string     Symbol_01            ="EURUSD";  // Symbol 1
input  bool       TradeInTimeRange_01  =true;      // |     Trading in a time range
input  ENUM_HOURS StartTrade_01        = h10;      // |     The hour of the beginning of a trading session
input  ENUM_HOURS StopOpenOrders_01    = h17;      // |     The hour  of the end of placing orders
input  ENUM_HOURS EndTrade_01          = h22;      // |     The hour of the end of a trading session
input  double     PendingOrder_01      = 50;       // |     Pending order
input  double     TakeProfit_01        = 100;      // |     Take Profit
input  double     StopLoss_01          = 50;       // |     Stop Loss
input  double     TrailingStop_01      = 10;       // |     Trailing Stop
input  bool       Reverse_01           = true;     // |     Position reversal
input  double     Lot_01               = 0.1;      // |     Lot
//---
sinput string delimeter_01=""; // --------------------------------
sinput string     Symbol_02            ="AUDUSD";  // Symbol 2
input  bool       TradeInTimeRange_02  =true;      // |     Trading in a time range
input  ENUM_HOURS StartTrade_02        = h10;      // |     The hour of the beginning of a trading session
input  ENUM_HOURS StopOpenOrders_02    = h17;      // |     The hour  of the end of placing orders
input  ENUM_HOURS EndTrade_02          = h22;      // |     The hour of the end of a trading session
input  double     PendingOrder_02      = 50;       // |     Pending order
input  double     TakeProfit_02        = 100;      // |     Take Profit
input  double     StopLoss_02          = 50;       // |     Stop Loss
input  double     TrailingStop_02      = 10;       // |     Trailing Stop
input  bool       Reverse_02           = true;     // |     Position reversal
input  double     Lot_02               = 0.1;      // |     Lot
```

Also correspondent changes have to be made in the list of arrays which will be filled with the values of external parameters:

```
//--- Arrays for storing external parameters
string     Symbols[NUMBER_OF_SYMBOLS];          // Symbol
bool       TradeInTimeRange[NUMBER_OF_SYMBOLS]; // Trading in a time range
ENUM_HOURS StartTrade[NUMBER_OF_SYMBOLS];       // The hour of the beginning of a trading session
ENUM_HOURS StopOpenOrders[NUMBER_OF_SYMBOLS];   // The hour  of the end of placing orders
ENUM_HOURS EndTrade[NUMBER_OF_SYMBOLS];         // The hour of the end of a trading session
double     PendingOrder[NUMBER_OF_SYMBOLS];     // Pending order
double     TakeProfit[NUMBER_OF_SYMBOLS];       // Take Profit
double     StopLoss[NUMBER_OF_SYMBOLS];         // Stop Loss
double     TrailingStop[NUMBER_OF_SYMBOLS];     // Trailing Stop
bool       Reverse[NUMBER_OF_SYMBOLS];          // Position Reversal
double     Lot[NUMBER_OF_SYMBOLS];              // Lot
```

Now we are going to arrange that in the reversal mode (the **Reverse** parameter value is **true**) the opposite pending order is deleted and placed anew, when one of the pending orders is triggered. We can not change the volume of the pending order as we would do in case of changing its price levels (order price, Stop Loss, Take Profit). We, therefore, have to delete it and place a new pending order with the required volume.

Moreover, if the reversal mode is enabled and Trailing Stop level is set up at the same time, then the pending order will be following the price. If, on top of that, Stop Loss is placed, its price value will be calculated and specified based on the pending order.

On the global scope let's create two string variables for the pending order comments:

```
//--- Pending order comments
string comment_top_order    ="top_order";
string comment_bottom_order ="bottom_order";
```

At the initialization in the function [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) during Expert Advisor loading, we will check the external parameters for correctness. Criteria for the assessment are as follows. When the TradeInTimeRange mode is enabled, the hour of the beginning of a trade session must not be one hour less than the hour of the end of placing pending orders. The hour of the end of placing pending orders, in its turn, must not be one hour less than the hour of the end of a trade session. Let's write the function **CheckInputParameters()** that will carry out such a check:

```
//+------------------------------------------------------------------+
//| Checks external parameters                                       |
//+------------------------------------------------------------------+
bool CheckInputParameters()
  {
//--- Loop through the specified symbols
   for(int s=0; s<NUMBER_OF_SYMBOLS; s++)
     {
      //--- If there is no symbol and the TradeInTimeRange mode is disabled, move on to the following symbol.
      if(Symbols[s]=="" || !TradeInTimeRange[s])
         continue;
      //--- Check the accuracy of the start and the end of a trade session time
      if(StartTrade[s]>=EndTrade[s])
        {
         Print(Symbols[s],
               ": The hour of the beginning of a trade session("+IntegerToString(StartTrade[s])+") "
               "must be less than the hour of the end of a trade session"("+IntegerToString(EndTrade[s])+")!");
         return(false);
        }
      //--- A trading session is to start no later that one hour before the hour of placing pending orders.
      //    Pending orders are to be placed no later than one hour before the hour of the end  of a trading session.
      if(StopOpenOrders[s]>=EndTrade[s] ||
         StopOpenOrders[s]<=StartTrade[s])
        {
         Print(Symbols[s],
               ": The hour of the end of placing orders ("+IntegerToString(StopOpenOrders[s])+") "
               "is to be less than the hour of the end ("+IntegerToString(EndTrade[s])+") and "
               "greater than the hour of the beginning of a trading session  ("+IntegerToString(StartTrade[s])+")!");
         return(false);
        }
     }
//--- Parameters are correct
   return(true);
  }
```

To implement this pattern we will need the functions that will carry out checks for staying within the specified time ranges for trade and placing pending orders. We shall name those functions **IsInTradeTimeRange()** and **IsInOpenOrdersTimeRange()**. They both work the same, the only difference is in the upper limit of the range in check. Further along we shall see where these functions will be used.

```
//+------------------------------------------------------------------+
//| Checks if we are within the time range for trade                 |
//+------------------------------------------------------------------+
bool IsInTradeTimeRange(int symbol_number)
  {
//--- If TradeInTimeRange mode is enabled
   if(TradeInTimeRange[symbol_number])
     {
      //--- Structure of the date and time
      MqlDateTime last_date;
      //--- Get the last value of the date and time data set
      TimeTradeServer(last_date);
      //--- Outside of the allowed time range
      if(last_date.hour<StartTrade[symbol_number] ||
         last_date.hour>=EndTrade[symbol_number])
         return(false);
     }
//--- Within the allowed time range
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks if we are within the time range for placing orders        |
//+------------------------------------------------------------------+
bool IsInOpenOrdersTimeRange(int symbol_number)
  {
//--- If the TradeInTimeRange mode if enabled
   if(TradeInTimeRange[symbol_number])
     {
      //--- Structure of the date and time
      MqlDateTime last_date;
      //--- Get the last value of the date and time data set
      TimeTradeServer(last_date);
      //--- Outside the allowed time range
      if(last_date.hour<StartTrade[symbol_number] ||
         last_date.hour>=StopOpenOrders[symbol_number])
         return(false);
     }
//--- Within the allowed time range
   return(true);
  }
```

Previous articles already considered functions for receiving properties of position, symbol and the history of the deals. In this article we will need a similar function for getting properties of a pending order. In the include file **Enums.mqh** we are going to create an enumeration with properties of a pending order:

```
//--- Enumeration of the properties of a pending order
enum ENUM_ORDER_PROPERTIES
  {
   O_SYMBOL          = 0,
   O_MAGIC           = 1,
   O_COMMENT         = 2,
   O_PRICE_OPEN      = 3,
   O_PRICE_CURRENT   = 4,
   O_PRICE_STOPLIMIT = 5,
   O_VOLUME_INITIAL  = 6,
   O_VOLUME_CURRENT  = 7,
   O_SL              = 8,
   O_TP              = 9,
   O_TIME_SETUP      = 10,
   O_TIME_EXPIRATION = 11,
   O_TIME_SETUP_MSC  = 12,
   O_TYPE_TIME       = 13,
   O_TYPE            = 14,
   O_ALL             = 15
  };
```

Then in the include file **TradeFunctions.mqh** we need to write a structure with the properties of a pending order and then instantiate it:

```
//-- Properties of a pending order
struct pending_order_properties
  {
   string            symbol;          // Symbol
   long              magic;           // Magic number
   string            comment;         // Comment
   double            price_open;      // Price specified in the order
   double            price_current;   // Current price of the order symbol
   double            price_stoplimit; // Limit order price for the Stop Limit order
   double            volume_initial;  // Initial order volume
   double            volume_current;  // Current order volume
   double            sl;              // Stop Loss level
   double            tp;              // Take Profit level
   datetime          time_setup;      // Order placement time
   datetime          time_expiration; // Order expiration time
   datetime          time_setup_msc;  // The time of placing an order for execution in milliseconds since 01.01.1970
   datetime          type_time;       // Order lifetime
   ENUM_ORDER_TYPE   type;            // Position type
  };
//--- Variable of the order features
pending_order_properties ord;
```

To get a property or even all properties of a pending order we are going to write the function **GetPendingOrderProperties()**. After the pending order has been selected, we can use this function for retrieving the properties of the order. The way to do it will be described further down.

```
//+------------------------------------------------------------------+
//| Retrieves the properties of the previously selected pending order|
//+------------------------------------------------------------------+
void GetPendingOrderProperties(ENUM_ORDER_PROPERTIES order_property)
  {
   switch(order_property)
     {
      case O_SYMBOL          : ord.symbol=OrderGetString(ORDER_SYMBOL);                              break;
      case O_MAGIC           : ord.magic=OrderGetInteger(ORDER_MAGIC);                               break;
      case O_COMMENT         : ord.comment=OrderGetString(ORDER_COMMENT);                            break;
      case O_PRICE_OPEN      : ord.price_open=OrderGetDouble(ORDER_PRICE_OPEN);                      break;
      case O_PRICE_CURRENT   : ord.price_current=OrderGetDouble(ORDER_PRICE_CURRENT);                break;
      case O_PRICE_STOPLIMIT : ord.price_stoplimit=OrderGetDouble(ORDER_PRICE_STOPLIMIT);            break;
      case O_VOLUME_INITIAL  : ord.volume_initial=OrderGetDouble(ORDER_VOLUME_INITIAL);              break;
      case O_VOLUME_CURRENT  : ord.volume_current=OrderGetDouble(ORDER_VOLUME_CURRENT);              break;
      case O_SL              : ord.sl=OrderGetDouble(ORDER_SL);                                      break;
      case O_TP              : ord.tp=OrderGetDouble(ORDER_TP);                                      break;
      case O_TIME_SETUP      : ord.time_setup=(datetime)OrderGetInteger(ORDER_TIME_SETUP);           break;
      case O_TIME_EXPIRATION : ord.time_expiration=(datetime)OrderGetInteger(ORDER_TIME_EXPIRATION); break;
      case O_TIME_SETUP_MSC  : ord.time_setup_msc=(datetime)OrderGetInteger(ORDER_TIME_SETUP_MSC);   break;
      case O_TYPE_TIME       : ord.type_time=(datetime)OrderGetInteger(ORDER_TYPE_TIME);             break;
      case O_TYPE            : ord.type=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);                break;
      case O_ALL             :
         ord.symbol=OrderGetString(ORDER_SYMBOL);
         ord.magic=OrderGetInteger(ORDER_MAGIC);
         ord.comment=OrderGetString(ORDER_COMMENT);
         ord.price_open=OrderGetDouble(ORDER_PRICE_OPEN);
         ord.price_current=OrderGetDouble(ORDER_PRICE_CURRENT);
         ord.price_stoplimit=OrderGetDouble(ORDER_PRICE_STOPLIMIT);
         ord.volume_initial=OrderGetDouble(ORDER_VOLUME_INITIAL);
         ord.volume_current=OrderGetDouble(ORDER_VOLUME_CURRENT);
         ord.sl=OrderGetDouble(ORDER_SL);
         ord.tp=OrderGetDouble(ORDER_TP);
         ord.time_setup=(datetime)OrderGetInteger(ORDER_TIME_SETUP);
         ord.time_expiration=(datetime)OrderGetInteger(ORDER_TIME_EXPIRATION);
         ord.time_setup_msc=(datetime)OrderGetInteger(ORDER_TIME_SETUP_MSC);
         ord.type_time=(datetime)OrderGetInteger(ORDER_TYPE_TIME);
         ord.type=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);                                      break;
         //---
     default: Print("Retrieved feature of the pending order was not taken into account in the enumeration "); return;
     }
  }
```

Now we are going to write basic functions for placing, modifying and deleting pending orders. The function **SetPendingOrder()** places a pending order. If the pending order failed to be placed, the mentioned function will make an entry in the journal with an error code and its description:

```
//+------------------------------------------------------------------+
//| Places a pending order                                           |
//+------------------------------------------------------------------+
void SetPendingOrder(int                  symbol_number,   // Symbol number
                     ENUM_ORDER_TYPE      order_type,      // Order type
                     double               lot,             // Volume
                     double               stoplimit_price, // Level of the StopLimit order
                     double               price,           // Price
                     double               sl,              // Stop Loss
                     double               tp,              // Take Profit
                     ENUM_ORDER_TYPE_TIME type_time,       // Order Expiration
                     string               comment)         // Comment
//--- Set magic number in the trade structure
   trade.SetExpertMagicNumber(MagicNumber);
//--- If a pending order failed to be placed, print an error message
   if(!trade.OrderOpen(Symbols[symbol_number],
                       order_type,lot,stoplimit_price,price,sl,tp,type_time,0,comment))
      Print("Error when placing a pending order: ",GetLastError()," - ",ErrorDescription(GetLastError()));
  }
```

The function **ModifyPendingOrder()** modifies a pending order. We are going to arrange so that we can change not only the price of the order but also its volume and pass it as the last parameter of the function. If the passed volume value is greater than zero, it means that the pending order has to be deleted and a new one with a required volume value placed. In all other cases we simply modify the existing order by changing the price value.

```
//+------------------------------------------------------------------+
//| Modifies a pending order                                         |
//+------------------------------------------------------------------+
void ModifyPendingOrder(int                  symbol_number,   //Symbol number
                        ulong                ticket,          // Order ticket
                        ENUM_ORDER_TYPE      type,            // Order type
                        double               price,           // Order price
                        double               sl,              // Stop Loss of the order
                        double               tp,              // Take Profit of the order
                        ENUM_ORDER_TYPE_TIME type_time,       // Order expiration
                        datetime             time_expiration, // Order expiration time
                        double               stoplimit_price, // Price
                        string               comment,         // Comment
                        double               volume)          // Volume
  {
//--- If the passed volume value is non-zero, delete the order and place it again
   if(volume>0)
     {
      //--- If the order failed to be deleted, exit
      if(!DeletePendingOrder(ticket))
         return;
      //--- Place a pending order
      SetPendingOrder(symbol_number,type,volume,0,price,sl,tp,type_time,comment);
      //--- Adjust Stop Loss of position as related to the order
      CorrectStopLossByOrder(symbol_number,price,type);
     }
//--- If the passed volume value is zero, modify the order
   else
     {
      //--- If the pending order failed to be modified, print a relevant message
      if(!trade.OrderModify(ticket,price,sl,tp,type_time,time_expiration,stoplimit_price))
         Print("Error when modifying the pending order price: ",
         GetLastError()," - ",ErrorDescription(GetLastError()));
      //--- Otherwise adjust Stop Loss of position as related to the order
      else
         CorrectStopLossByOrder(symbol_number,price,type);
     }
  }
```

In the code above highlighted are two new functions **DeletePendingOrder()** and **CorrectStopLossByOrder()**. The first one deletes a pending order and the second one adjusts Stop Loss of the position as related to the pending order.

```
//+------------------------------------------------------------------+
//| Deletes a pending order                                          |
//+------------------------------------------------------------------+
bool DeletePendingOrder(ulong ticket)
  {

//--- If a pending order failed to get deleted, print a relevant message
   if(!trade.OrderDelete(ticket))
     {
      Print("Error when deleting a pending order: ",GetLastError()," - ",ErrorDescription(GetLastError()));
      return(false);
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Modifies StopLoss of the position as related to the pending order|
//+------------------------------------------------------------------+
void CorrectStopLossByOrder(int             symbol_number, // Symbol number
                            double          price,         // Order Price
                            ENUM_ORDER_TYPE type)          // Order Type
  {
//--- If Stop Loss disabled, exit
   if(StopLoss[symbol_number]==0)
      return;
//--- If Stop Loss enabled
   double new_sl=0.0; // New Stop Loss value
//--- Get a Point value
   GetSymbolProperties(symbol_number,S_POINT);
//--- Number of decimal places
   GetSymbolProperties(symbol_number,S_DIGITS);
//--- Get Take Profit of position
   GetPositionProperties(symbol_number,P_TP);
//--- Calculate as related to the order type
   switch(type)
     {
      case ORDER_TYPE_BUY_STOP  :
         new_sl=NormalizeDouble(price+CorrectValueBySymbolDigits(StopLoss[symbol_number]*symb.point),symb.digits);
         break;
      case ORDER_TYPE_SELL_STOP :
         new_sl=NormalizeDouble(price-CorrectValueBySymbolDigits(StopLoss[symbol_number]*symb.point),symb.digits);
         break;
     }
//--- Modify the position
   if(!trade.PositionModify(Symbols[symbol_number],new_sl,pos.tp))
      Print("Error when modifying position: ",GetLastError()," - ",ErrorDescription(GetLastError()));
  }
```

Before placing a pending order, it is also necessary to check if a pending order with the same comments already exists. As mentioned in the beginning of this article, we shall place the top Buy Stop order with a comment "top\_order" and the Sell Stop order with a comment "bottom\_order". To facilitate such a check let's write a function named **CheckPendingOrderByComment()**:

```
//+------------------------------------------------------------------+
//| Checks existence of a pending order by a comment                 |
//+------------------------------------------------------------------+
bool CheckPendingOrderByComment(int symbol_number,string comment)
  {
   int    total_orders  =0;  // Total number of pending orders
   string order_symbol  =""; // Order Symbol
   string order_comment =""; // Order Comment
//--- Get the total number of pending orders
   total_orders=OrdersTotal();
//--- Loop through the total orders
   for(int i=total_orders-1; i>=0; i--)
     {
      //---Select the order by the ticket
      if(OrderGetTicket(i)>0)
        {
         //--- Get the symbol name
         order_symbol=OrderGetString(ORDER_SYMBOL);
         //--- If the symbols are equal
         if(order_symbol==Symbols[symbol_number])
           {
            //--- Get the order comment
            order_comment=OrderGetString(ORDER_COMMENT);
            //--- If the comments are equal
            if(order_comment==comment)
               return(true);
           }
        }
     }
//--- Order with a specified comment not found
   return(false);
  }
```

The code above shows that the total number of orders can be obtained using the system function [OrdersTotal()](https://www.mql5.com/en/docs/trading/orderstotal). However, to get the total number of pending orders for a specified symbol we are going to write a user-defined function. We shall name it **OrdersTotalBySymbol()**:

```
//+------------------------------------------------------------------+
//| Returns the total number of orders for the specified symbol      |
//+------------------------------------------------------------------+
int OrdersTotalBySymbol(string symbol)
  {
   int   count        =0; // Order counter
   int   total_orders =0; // Total number of pending orders
//--- Get the total number of pending orders
   total_orders=OrdersTotal();
//--- Loop through the total number of orders
   for(int i=total_orders-1; i>=0; i--)
     {
      //--- If an order has been selected
      if(OrderGetTicket(i)>0)
        {
         //--- Get the order symbol
         GetOrderProperties(O_SYMBOL);
         //--- If the order symbol and the specified symbol are equal
         if(ord.symbol==symbol)
            //--- Increase the counter
            count++;
        }
     }
//--- Return the total number of orders
   return(count);
  }
```

Before placing a pending order it is necessary to calculate a price for it as well as Stop Loss and Take Profit levels if required. If the reversal mode is enabled, we will need separate user-defined functions for recalculating and changing Trailing Stop levels.

To calculate a pending order price let's write the function **CalculatePendingOrder()**:

```
//+------------------------------------------------------------------+
//| Calculates the pending order level(price)                        |
//+------------------------------------------------------------------+
double CalculatePendingOrder(int symbol_number,ENUM_ORDER_TYPE order_type)
  {
//--- For the calculated pending order value
   double price=0.0;
//--- If the value for SELL STOP order is to be calculated
   if(order_type==ORDER_TYPE_SELL_STOP)
     {
      //--- Calculate level
      price=NormalizeDouble(symb.bid-CorrectValueBySymbolDigits(PendingOrder[symbol_number]*symb.point),symb.digits);
      //--- Return calculated value if it is less than the lower limit of Stops level
      //    If the value is equal or greater, return the adjusted value
      return(price<symb.down_level ? price : symb.down_level-symb.offset);
     }
//--- If the value for BUY STOP order is to be calculated
   if(order_type==ORDER_TYPE_BUY_STOP)
     {
      //--- Calculate level
      price=NormalizeDouble(symb.ask+CorrectValueBySymbolDigits(PendingOrder[symbol_number]*symb.point),symb.digits);
      //--- Return the calculated value if it is greater than the upper limit of Stops level
      //    If the value is equal or less, return the adjusted value
      return(price>symb.up_level ? price : symb.up_level+symb.offset);
     }
//---
   return(0.0);
  }
```

Below is the function code for calculating Stop Loss and Take Profit levels in a pending order.

```
//+------------------------------------------------------------------+
//| Calculates Stop Loss level for a pending order                   |
//+------------------------------------------------------------------+
double CalculatePendingOrderStopLoss(int symbol_number,ENUM_ORDER_TYPE order_type,double price)
  {
//--- If Stop Loss is required
   if(StopLoss[symbol_number]>0)
     {
      double sl         =0.0; // For the Stop Loss calculated value
      double up_level   =0.0; // Upper limit of Stop Levels
      double down_level =0.0; // Lower limit of Stop Levels
      //--- If the value for BUY STOP order is to be calculated
      if(order_type==ORDER_TYPE_BUY_STOP)
        {
         //--- Define lower threshold
         down_level=NormalizeDouble(price-symb.stops_level*symb.point,symb.digits);
         //--- Calculate level
         sl=NormalizeDouble(price-CorrectValueBySymbolDigits(StopLoss[symbol_number]*symb.point),symb.digits);
         //--- Return the calculated value if it is less than the lower limit of Stop level
         //    If the value is equal or greater, return the adjusted value
         return(sl<down_level ? sl : NormalizeDouble(down_level-symb.offset,symb.digits));
        }
      //--- If the value for the SELL STOP order is to be calculated
      if(order_type==ORDER_TYPE_SELL_STOP)
        {
         //--- Define the upper threshold
         up_level=NormalizeDouble(price+symb.stops_level*symb.point,symb.digits);
         //--- Calculate the level
         sl=NormalizeDouble(price+CorrectValueBySymbolDigits(StopLoss[symbol_number]*symb.point),symb.digits);
         //--- Return the calculated value if it is greater than the upper limit of the Stops level
         //    If the value is less or equal, return the adjusted value.
         return(sl>up_level ? sl : NormalizeDouble(up_level+symb.offset,symb.digits));
        }
     }
//---
   return(0.0);
  }
//+------------------------------------------------------------------+
//| Calculates the Take Profit level for a pending order             |
//+------------------------------------------------------------------+
double CalculatePendingOrderTakeProfit(int symbol_number,ENUM_ORDER_TYPE order_type,double price)
  {
//--- If Take Profit is required
   if(TakeProfit[symbol_number]>0)
     {
      double tp         =0.0; // For the calculated Take Profit value
      double up_level   =0.0; // Upper limit of Stop Levels
      double down_level =0.0; // Lower limit of Stop Levels
      //--- If the value for SELL STOP order is to be calculated
      if(order_type==ORDER_TYPE_SELL_STOP)
        {
         //--- Define lower threshold
         down_level=NormalizeDouble(price-symb.stops_level*symb.point,symb.digits);
         //--- Calculate the level
         tp=NormalizeDouble(price-CorrectValueBySymbolDigits(TakeProfit[symbol_number]*symb.point),symb.digits);
         //--- Return the calculated value if it is less than the below limit of the Stops level
         //    If the value is greater or equal, return the adjusted value
         return(tp<down_level ? tp : NormalizeDouble(down_level-symb.offset,symb.digits));
        }
      //--- If the value for the BUY STOP order is to be calculated
      if(order_type==ORDER_TYPE_BUY_STOP)
        {
         //--- Define the upper threshold
         up_level=NormalizeDouble(price+symb.stops_level*symb.point,symb.digits);
         //--- Calculate the level
         tp=NormalizeDouble(price+CorrectValueBySymbolDigits(TakeProfit[symbol_number]*symb.point),symb.digits);
         //--- Return the calculated value if it is greater than the upper limit of the Stops level
         //    If the value is less or equal, return the adjusted value
         return(tp>up_level ? tp : NormalizeDouble(up_level+symb.offset,symb.digits));
        }
     }
//---
   return(0.0);
  }
```

To calculate the Stops level (price)of a reversed pending order and pulling it up we are going to write the following functions **CalculateReverseOrderTrailingStop()** and **ModifyPendingOrderTrailingStop()**. You can find the codes of the functions below.

The code of the function **CalculateReverseOrderTrailingStop()**:

```
//+----------------------------------------------------------------------------+
//| Calculates the Trailing Stop level for the reversed order                  |
//+----------------------------------------------------------------------------+
double CalculateReverseOrderTrailingStop(int symbol_number,ENUM_POSITION_TYPE position_type)
  {
//--- Variables for calculation
   double    level       =0.0;
   double    buy_point   =low[symbol_number].value[1];  // Low value for Buy
   double    sell_point  =high[symbol_number].value[1]; // High value for Sell
//--- Calculate the level for the BUY position
   if(position_type==POSITION_TYPE_BUY)
     {
      //--- Bar's low minus the specified number of points
      level=NormalizeDouble(buy_point-CorrectValueBySymbolDigits(PendingOrder[symbol_number]*symb.point),symb.digits);
      //---  If the calculated level is lower than the lower limit of the Stops level,
      //    the calculation is complete, return the current value of the level
      if(level<symb.down_level)
         return(level);
      //--- If it is not lower, try to calculate based on the bid price
      else
        {
         level=NormalizeDouble(symb.bid-CorrectValueBySymbolDigits(PendingOrder[symbol_number]*symb.point),symb.digits);
         //--- If the calculated level is lower than the limit, return the current value of the level
         //    otherwise set the nearest possible value
         return(level<symb.down_level ? level : symb.down_level-symb.offset);
        }
     }
//--- Calculate the level for the SELL position
   if(position_type==POSITION_TYPE_SELL)
     {
      // Bar's high plus the specified number of points
      level=NormalizeDouble(sell_point+CorrectValueBySymbolDigits(PendingOrder[symbol_number]*symb.point),symb.digits);
      //--- If the calculated level is higher than the upper limit of the Stops level,
      //    then the calculation is complete, return the current value of the level
      if(level>symb.up_level)
         return(level);
      //--- If it is not higher, try to calculate based on the ask price
      else
        {
         level=NormalizeDouble(symb.ask+CorrectValueBySymbolDigits(PendingOrder[symbol_number]*symb.point),symb.digits);
         //--- If the calculated level is higher than the limit, return the current value of the level
         //    Otherwise set the nearest possible value
         return(level>symb.up_level ? level : symb.up_level+symb.offset);
        }
     }
//---
   return(0.0);
  }
```

The code of the function **ModifyPendingOrderTrailingStop()**:

```
//+------------------------------------------------------------------+
//| Modifying the Trailing Stop level for a pending order            |
//+------------------------------------------------------------------+
void ModifyPendingOrderTrailingStop(int symbol_number)
  {
//--- Exit, if the reverse position mode is disabled and Trailing Stop is not set
   if(!Reverse[symbol_number] || TrailingStop[symbol_number]==0)
      return;
//---
   double          new_level              =0.0;         // For calculating a new level for a pending order
   bool            condition              =false;       // For checking the modification condition
   int             total_orders           =0;           // Total number of pending orders
   ulong           order_ticket           =0;           // Order ticket
   string          opposite_order_comment ="";          // Opposite order comment
   ENUM_ORDER_TYPE opposite_order_type    =WRONG_VALUE; // Order type

//--- Get the flag of presence/absence of a position
   pos.exists=PositionSelect(Symbols[symbol_number]);
//--- If a position is absent
   if(!pos.exists)
      return;
//--- Get a total number of pending orders
   total_orders=OrdersTotal();
//--- Get the symbol properties
   GetSymbolProperties(symbol_number,S_ALL);
//--- Get the position properties
   GetPositionProperties(symbol_number,P_ALL);
//--- Get the level for Stop Loss
   new_level=CalculateReverseOrderTrailingStop(symbol_number,pos.type);
//--- Loop through the orders from the last to the first one
   for(int i=total_orders-1; i>=0; i--)
     {
      //--- If the order selected
      if((order_ticket=OrderGetTicket(i))>0)
        {
         //--- Get the order symbol
         GetPendingOrderProperties(O_SYMBOL);
         //--- Get the order comment
         GetPendingOrderProperties(O_COMMENT);
         //--- Get the order price
         GetPendingOrderProperties(O_PRICE_OPEN);
         //--- Depending on the position type, check the relevant condition for the Trailing Stop modification
         switch(pos.type)
           {
            case POSITION_TYPE_BUY  :
               //---If the new order value is greater than the current value plus set step then condition fulfilled
               condition=new_level>ord.price_open+CorrectValueBySymbolDigits(TrailingStop[symbol_number]*symb.point);
               //--- Define the type and comment of the reversed pending order for check.
               opposite_order_type    =ORDER_TYPE_SELL_STOP;
               opposite_order_comment =comment_bottom_order;
               break;
            case POSITION_TYPE_SELL :
               //--- If the new value for the order if less than the current value minus a set step then condition fulfilled
               condition=new_level<ord.price_open-CorrectValueBySymbolDigits(TrailingStop[symbol_number]*symb.point);
               //--- Define the type and comment of the reversed pending order for check
               opposite_order_type    =ORDER_TYPE_BUY_STOP;
               opposite_order_comment =comment_top_order;
               break;
           }
         //--- If condition fulfilled, the order symbol and positions are equal
         //    and order comment and the reversed order comment are equal
         if(condition &&
            ord.symbol==Symbols[symbol_number] &&
            ord.comment==opposite_order_comment)
           {
            double sl=0.0; // Stop Loss
            double tp=0.0; // Take Profit
            //--- Get Take Profit and Stop Loss levels
            sl=CalculatePendingOrderStopLoss(symbol_number,opposite_order_type,new_level);
            tp=CalculatePendingOrderTakeProfit(symbol_number,opposite_order_type,new_level);
            //--- Modify order
            ModifyPendingOrder(symbol_number,order_ticket,opposite_order_type,new_level,sl,tp,
                               ORDER_TIME_GTC,ord.time_expiration,ord.price_stoplimit,ord.comment,0);
            return;
           }
        }
     }
  }
```

Sometimes it may be necessary to find out if a position was closed at Stop Loss or Take Profit. In this particular case we are going to come across such a requirement. Therefore let's write functions that will identify this event by the last deal comment. To retrieve the last deal comment for a specified symbol we are going to write a separate function named **GetLastDealComment()**:

```
//+------------------------------------------------------------------+
//| Returns a the last deal comment for a specified symbol           |
//+------------------------------------------------------------------+
string GetLastDealComment(int symbol_number)
  {
   int    total_deals  =0;  // Total number of deals in the selected history
   string deal_symbol  =""; // Deal symbol
   string deal_comment =""; // Deal comment
//--- If the deals history retrieved
   if(HistorySelect(0,TimeCurrent()))
     {
      //--- Receive the number of deals in the retrieved list
      total_deals=HistoryDealsTotal();
      //--- Loop though the total number of deals in the retrieved list from the last deal to the first one.
      for(int i=total_deals-1; i>=0; i--)
        {
         //--- Receive the deal comment
         deal_comment=HistoryDealGetString(HistoryDealGetTicket(i),DEAL_COMMENT);
         //--- Receive the deal symbol
         deal_symbol=HistoryDealGetString(HistoryDealGetTicket(i),DEAL_SYMBOL);
         //--- If the deal symbol and the current symbol are equal, stop the loop
         if(deal_symbol==Symbols[symbol_number])
            break;
        }
     }
//---
   return(deal_comment);
  }
```

Now it is easy to write functions that will determine the reason of closing of the last position for the specified symbol. Below are the codes of the functions **IsClosedByTakeProfit()** and **IsClosedByStopLoss()**:

```
//+------------------------------------------------------------------+
//| Returns the reason for closing position at Take Profit           |
//+------------------------------------------------------------------+
bool IsClosedByTakeProfit(int symbol_number)
  {
   string last_comment="";
//--- Get the last deal comment for the specified symbol
   last_comment=GetLastDealComment(symbol_number);
//--- If the comment contain a string "tp"
   if(StringFind(last_comment,"tp",0)>-1)
      return(true);
//--- If the comment does not contain a string "tp"
   return(false);
  }
//+------------------------------------------------------------------+
//| Returns the reason for closing position at Stop Loss             |
//+------------------------------------------------------------------+
bool IsClosedByStopLoss(int symbol_number)
  {
   string last_comment="";
//--- Get the last deal comment for the specified symbol
   last_comment=GetLastDealComment(symbol_number);
//--- If the comment contains the string "sl"
   if(StringFind(last_comment,"sl",0)>-1)
      return(true);
//--- If the comment does not contain the string "sl"
   return(false);
  }
```

We are going to carry out another check to determine if the last deal in the history is truly a deal for the specified symbol. We want to keep last deal ticket in memory. To achieve that we are going to add an array on the global scope:

```
//--- Array for checking the ticket of the last deal for each symbol.
ulong last_deal_ticket[NUMBER_OF_SYMBOLS];
```

The function **IsLastDealTicket()** for checking the last deal ticket will look as shown in the code below:

```
//+------------------------------------------------------------------+
//| Returns the event of the last deal for the specified symbol      |
//+------------------------------------------------------------------+
bool IsLastDealTicket(int symbol_number)
  {
   int    total_deals =0;  // Total number of deals in the selected history list
   string deal_symbol =""; // Deal symbol
   ulong  deal_ticket =0;  // Deal ticket
//--- If the deal history was received
   if(HistorySelect(0,TimeCurrent()))
     {
      //--- Get the total number of deals in the received list
      total_deals=HistoryDealsTotal();
      //--- Loop through the total number of deals from the last deal to the first one
      for(int i=total_deals-1; i>=0; i--)
        {
         //--- Get deal ticket
         deal_ticket=HistoryDealGetTicket(i);
         //--- Get deal symbol
         deal_symbol=HistoryDealGetString(deal_ticket,DEAL_SYMBOL);
         //--- If deal symbol and the current one are equal, stop the loop
         if(deal_symbol==Symbols[symbol_number])
           {
            //--- If the tickets are equal, exit
            if(deal_ticket==last_deal_ticket[symbol_number])
               return(false);
            //--- If the tickets are not equal report it
            else
              {
               //--- Save the last deal ticket
               last_deal_ticket[symbol_number]=deal_ticket;
               return(true);
              }
           }
        }
     }
//---
   return(false);
  }
```

If the current time is outside the specified trade range, the position will be forced to close no matter whether it is at loss or at profit. Let's write the function **ClosePosition()** for closing a position:

```
//+------------------------------------------------------------------+
//| Closes position                                                  |
//+------------------------------------------------------------------+
void ClosePosition(int symbol_number)
  {
//--- Check if position exists
   pos.exists=PositionSelect(Symbols[symbol_number]);
//--- If there is no position, exit
   if(!pos.exists)
      return;
//--- Set the slippage value in points
   trade.SetDeviationInPoints(CorrectValueBySymbolDigits(Deviation));
//--- If the position was not closed, print the relevant message
   if(!trade.PositionClose(Symbols[symbol_number]))
      Print("Error when closing position: ",GetLastError()," - ",ErrorDescription(GetLastError()));
  }
```

When a position is closed at going outside the trade time range, all pending orders must be deleted. The function **DeleteAllPendingOrders()** that we are about to write, will be deleting all pending orders for the specified symbol:

```
//+------------------------------------------------------------------+
//| Deletes all pending orders                                       |
//+------------------------------------------------------------------+
void DeleteAllPendingOrders(int symbol_number)
  {
   int   total_orders =0; // Total number of pending orders
   ulong order_ticket =0; // Order ticket
//--- Get the total number of pending orders
   total_orders=OrdersTotal();
//--- Loop through the total number of pending orders
   for(int i=total_orders-1; i>=0; i--)
     {
      //--- If the order selected
      if((order_ticket=OrderGetTicket(i))>0)
        {
         //--- Get the order symbol
         GetOrderProperties(O_SYMBOL);
         //--- If the order symbol and the current symbol are equal
         if(ord.symbol==Symbols[symbol_number])
            //--- Delete the order
            DeletePendingOrder(order_ticket);
        }
     }
  }
```

So now we have all functions necessary for the structural scheme. Let's take a look at the familiar function **TradingBlock()**, which has gone through some significant changes and a new one for managing pending orders **ManagePendingOrders()**. Full control over the current situation concerning pending orders will be carried out in it.

The function **TradingBlock()** for the current pattern looks as follows:

```
//+------------------------------------------------------------------+
//| Trade block                                                      |
//+------------------------------------------------------------------+
void TradingBlock(int symbol_number)
  {
   double          tp=0.0;                 // Take Profit
   double          sl=0.0;                 // Stop Loss
   double          lot=0.0;                // Volume for position calculation in case of reversed position
   double          order_price=0.0;        // Price for placing the order
   ENUM_ORDER_TYPE order_type=WRONG_VALUE; // Order type for opening position
//--- If outside of the time range for placing pending orders
   if(!IsInOpenOrdersTimeRange(symbol_number))
      return;
//--- Find out if there is an open position for the symbol
   pos.exists=PositionSelect(Symbols[symbol_number]);
//--- If there is no position
   if(!pos.exists)
     {
      //--- Get symbol properties
      GetSymbolProperties(symbol_number,S_ALL);
      //--- Adjust the volume
      lot=CalculateLot(symbol_number,Lot[symbol_number]);
      //--- If there is no upper pending order
      if(!CheckPendingOrderByComment(symbol_number,comment_top_order))
        {
         //--- Get the price for placing a pending order
         order_price=CalculatePendingOrder(symbol_number,ORDER_TYPE_BUY_STOP);
         //--- Get Take Profit and Stop Loss levels
         sl=CalculatePendingOrderStopLoss(symbol_number,ORDER_TYPE_BUY_STOP,order_price);
         tp=CalculatePendingOrderTakeProfit(symbol_number,ORDER_TYPE_BUY_STOP,order_price);
         //--- Place a pending order
         SetPendingOrder(symbol_number,ORDER_TYPE_BUY_STOP,lot,0,order_price,sl,tp,ORDER_TIME_GTC,comment_top_order);
        }
      //--- If there is no lower pending order
      if(!CheckPendingOrderByComment(symbol_number,comment_bottom_order))
        {
         //--- Get the price for placing the pending order
         order_price=CalculatePendingOrder(symbol_number,ORDER_TYPE_SELL_STOP);
         //--- Get Take Profit and Stop Loss levels
         sl=CalculatePendingOrderStopLoss(symbol_number,ORDER_TYPE_SELL_STOP,order_price);
         tp=CalculatePendingOrderTakeProfit(symbol_number,ORDER_TYPE_SELL_STOP,order_price);
         //--- Place a pending order
         SetPendingOrder(symbol_number,ORDER_TYPE_SELL_STOP,lot,0,order_price,sl,tp,ORDER_TIME_GTC,comment_bottom_order);
        }
     }
  }
```

Code of the function **ManagePendingOrders()** for managing pending orders:

```
//+------------------------------------------------------------------+
//| Manages pending orders                                           |
//+------------------------------------------------------------------+
void ManagePendingOrders()
  {
//--- Loop through the total number of symbols
   for(int s=0; s<NUMBER_OF_SYMBOLS; s++)
     {
      //--- If trading this symbol is forbidden, go to the following one
      if(Symbols[s]=="")
         continue;
      //--- Find out if there is an open position for the symbol
      pos.exists=PositionSelect(Symbols[s]);
      //--- If there is no position
      if(!pos.exists)
        {
         //--- If the last deal on current symbol and
         //    position  was exited on Take Profit or Stop Loss
         if(IsLastDealTicket(s) &&
            (IsClosedByStopLoss(s) || IsClosedByTakeProfit(s)))
            //--- Delete all pending orders for the symbol
            DeleteAllPendingOrders(s);
         //--- Go to the following symbol
         continue;
        }
      //--- If there is a position
      ulong           order_ticket           =0;           // Order ticket
      int             total_orders           =0;           // Total number of pending orders
      int             symbol_total_orders    =0;           // Number of pending orders for the specified symbol
      string          opposite_order_comment ="";          // Opposite order comment
      ENUM_ORDER_TYPE opposite_order_type    =WRONG_VALUE; // Order type
      //--- Get the total number of pending orders
      total_orders=OrdersTotal();
      //--- Get the total number of pending orders for the specified symbol
      symbol_total_orders=OrdersTotalBySymbol(Symbols[s]);
      //--- Get symbol properties
      GetSymbolProperties(s,S_ASK);
      GetSymbolProperties(s,S_BID);
      //--- Get the comment for the selected position
      GetPositionProperties(s,P_COMMENT);
      //--- If the position comment belongs to the upper order,
      //    then the lower order is to be deleted, modified/placed
      if(pos.comment==comment_top_order)
        {
         opposite_order_type    =ORDER_TYPE_SELL_STOP;
         opposite_order_comment =comment_bottom_order;
        }
      //--- If the position comment belongs to the lower order,
      //    then the upper order is to be deleted/modified/placed
      if(pos.comment==comment_bottom_order)
        {
         opposite_order_type    =ORDER_TYPE_BUY_STOP;
         opposite_order_comment =comment_top_order;
        }
      //--- If there are no pending orders for the specified symbol
      if(symbol_total_orders==0)
        {
         //--- If the position reversal is enabled, place a reversed order
         if(Reverse[s])
           {
            double tp=0.0;          // Take Profit
            double sl=0.0;          // Stop Loss
            double lot=0.0;         // Volume for position calculation in case of reversed positio
            double order_price=0.0; // Price for placing the order
            //--- Get the price for placing a pending order
            order_price=CalculatePendingOrder(s,opposite_order_type);
            //---Get Take Profit и Stop Loss levels
            sl=CalculatePendingOrderStopLoss(s,opposite_order_type,order_price);
            tp=CalculatePendingOrderTakeProfit(s,opposite_order_type,order_price);
            //--- Calculate double volume
            lot=CalculateLot(s,pos.volume*2);
            //--- Place the pending order
            SetPendingOrder(s,opposite_order_type,lot,0,order_price,sl,tp,ORDER_TIME_GTC,opposite_order_comment);
            //--- Adjust Stop Loss as related to the order
            CorrectStopLossByOrder(s,order_price,opposite_order_type);
           }
         return;
        }
      //--- If there are pending orders for this symbol, then depending on the circumstances delete or
      //    modify the reversed order
      if(symbol_total_orders>0)
        {
         //--- Loop through the total number of orders from the last one to the first one
         for(int i=total_orders-1; i>=0; i--)
           {
            //--- If the order chosen
            if((order_ticket=OrderGetTicket(i))>0)
              {
               //--- Get the order symbol
               GetPendingOrderProperties(O_SYMBOL);
               //--- Get the order comment
               GetPendingOrderProperties(O_COMMENT);
               //--- If order symbol and position symbol are equal,
               //    and order comment and the reversed order comment are equal
               if(ord.symbol==Symbols[s] &&
                  ord.comment==opposite_order_comment)
                 {
                  //--- If position reversal is disabled
                  if(!Reverse[s])
                     //--- Delete order
                     DeletePendingOrder(order_ticket);
                  //--- If position reversal is enabled
                  else
                    {
                     double lot=0.0;
                     //--- Get the current order properties
                     GetPendingOrderProperties(O_ALL);
                     //--- Get the current position volume
                     GetPositionProperties(s,P_VOLUME);
                     //--- If the order has been modified already, exit the loop.
                     if(ord.volume_initial>pos.volume)
                        break;
                     //--- Calculate double volume
                     lot=CalculateLot(s,pos.volume*2);
                     //--- Modify (delete and place again) the order
                     ModifyPendingOrder(s,order_ticket,opposite_order_type,
                                        ord.price_open,ord.sl,ord.tp,
                                        ORDER_TIME_GTC,ord.time_expiration,
                                        ord.price_stoplimit,opposite_order_comment,lot);
                    }
                 }
              }
           }
        }
     }
  }
```

Now we are only to make minor adjustments in the main program file. We shall add the trade events handler [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade). Assessment of the current situation in relation to the pending orders against the trading event will be carried out in this function.

```
//+------------------------------------------------------------------+
//| Processing of trade events                                       |
//+------------------------------------------------------------------+
void OnTrade()
  {
//--- Check the state of pending orders
   ManagePendingOrders();
  }
```

The function **ManagePendingOrders()** will also be used in the user event handler [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent):

```
//+------------------------------------------------------------------+
//| User events and chart events handler                             |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,         // Event identifier
                  const long &lparam,   // Parameter of long event type
                  const double &dparam, // Parameter of double event type
                  const string &sparam) // Parameter of string event type
  {
//--- If it is a user event
   if(id>=CHARTEVENT_CUSTOM)
     {
      //--- Exit, if trade is prohibited
      if(CheckTradingPermission()>0)
         return;
      //--- If it is a tick event
      if(lparam==CHARTEVENT_TICK)
        {
         //--- Check the state of pending orders
         ManagePendingOrders();
         //--- Check signals and trade according to them
         CheckSignalsAndTrade();
         return;
        }
     }
  }
```

Some changes were made in the function **CheckSignalsAndTrade()** as well. In the code below highlighted are strings featuring new functions considered in this article.

```
//+------------------------------------------------------------------+
//| Checks signals and trades based on New Bar event                 |
//+------------------------------------------------------------------+
void CheckSignalsAndTrade()
  {
//--- Loop through all specified signals
   for(int s=0; s<NUMBER_OF_SYMBOLS; s++)
     {
      //--- If trading this symbol is prohibited, exit
      if(Symbols[s]=="")
         continue;
      //--- If the bar is not new, move on to the following symbol
      if(!CheckNewBar(s))
         continue;
      //--- If there is a new bar
      else
        {
         //--- If outside the time range
         if(!IsInTradeTimeRange(s))
           {
            //--- Close position
            ClosePosition(s);
            //--- Delete all pending orders
            DeleteAllPendingOrders(s);
            //--- Move on to the following symbol
            continue;
           }
         //--- Get bars data
         GetBarsData(s);
         //--- Check conditions and trade
         TradingBlock(s);
         //--- If position reversal if enabled
         if(Reverse[s])
            //--- Pull up Stop Loss for pending order
            ModifyPendingOrderTrailingStop(s);
         //--- If position reversal is disabled
         else
         //--- Pull up Stop Loss
            ModifyTrailingStop(s);
        }
     }

```

Now everything is ready and we can try to optimize parameters of this multi-currency Expert Advisor Let's set up the strategy Tester as shown below:

![Fig. 1 - Tester settings for parameters optimization.](https://c.mql5.com/2/6/0l884edso_cj386o9_clj_80c0qh07yb4_ob2bakrahe.png)

Fig. 1 - Tester settings for parameters optimization.

First we shall optimize parameters for the currency pair **EURUSD**, and then for **AUDUSD**. The screen shot below shows what parameters we shall select for optimization of **EURUSD**:

![Fig. 2 - Setting up parameters for optimization of multi-currency Expert Advisor](https://c.mql5.com/2/11/en_02.png)

Fig. 2 - Setting up parameters for optimization of multi-currency Expert Advisor

After the parameters of the currency pair **EURUSD** have been optimized, the same parameters should be optimized for **AUDUSD**. Below is the result for both symbols tested together. Results were selected by the maximum recovery factor. For the test, the lot value was set to **1** for both symbols.

![Fig. 3 - test result for the two symbols together.](https://c.mql5.com/2/11/en_03.png)

Fig. 3 - test result for the two symbols together.

### Conclusion

That's pretty much about it. With ready functions at hand, you can concentrate on developing the idea of making trade decisions. In this case changes will have to be implemented in the functions **TradingBlock()** and **ManagePendingOrders()**. For those who started learning MQL5 recently, we recommend to practice adding more symbols and change the trade algorithm scheme.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/755](https://www.mql5.com/ru/articles/755)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/755.zip "Download all attachments in the single ZIP archive")

[755.set](https://www.mql5.com/en/articles/download/755/755.set "Download 755.set")(1.59 KB)

[multisymbolpendingorders\_en.zip](https://www.mql5.com/en/articles/download/755/multisymbolpendingorders_en.zip "Download multisymbolpendingorders_en.zip")(27.69 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Magic of time trading intervals with Frames Analyzer tool](https://www.mql5.com/en/articles/11667)
- [The power of ZigZag (part II). Examples of receiving, processing and displaying data](https://www.mql5.com/en/articles/5544)
- [The power of ZigZag (part I). Developing the base class of the indicator](https://www.mql5.com/en/articles/5543)
- [Universal RSI indicator for working in two directions simultaneously](https://www.mql5.com/en/articles/4828)
- [Expert Advisor featuring GUI: Adding functionality (part II)](https://www.mql5.com/en/articles/4727)
- [Expert Advisor featuring GUI: Creating the panel (part I)](https://www.mql5.com/en/articles/4715)
- [Visualizing optimization results using a selected criterion](https://www.mql5.com/en/articles/4636)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/35643)**
(24)


![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
5 Apr 2020 at 20:01

I still found it on a backup from 2017.


![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
5 Apr 2020 at 20:06

Otto ...now you can use it :-)

It makes trades for me.

![Otto Pauser](https://c.mql5.com/avatar/2016/5/574C2261-ACAB.JPG)

**[Otto Pauser](https://www.mql5.com/en/users/kronenchakra)**
\|
5 Apr 2020 at 20:56

That's quite a response. Thank you!

I just wanted to point out that the authors of the articles should take care of them.

All you have to do is

```
CheckTradingPermission()
```

and all the MQL5xxx rubbish and [it will work](https://www.mql5.com/en/articles/180 "Article: Averaging of price series without additional buffers for intermediate calculations");)


![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
5 Apr 2020 at 21:09

**Otto Pauser:**

That's quite a response. Thank you!

I just wanted to point out that the authors of the articles should take care of them.

Mmmhhh ... yes, we know.

And I gave it some expression.

Something like that works, you notice it in other places, even if nobody says anything about it :-)

![Otto Pauser](https://c.mql5.com/avatar/2016/5/574C2261-ACAB.JPG)

**[Otto Pauser](https://www.mql5.com/en/users/kronenchakra)**
\|
6 Apr 2020 at 01:23

**Christian:**

Mmmhhh ... yes, we know.

And I've given it some expression.

Something like that works, you notice it in other places, even if nobody says anything about it :-)

My intention was to reprogramme the MarketOrders into PendigOrders.

Whoever can use it, here is the code how it [works](https://www.mql5.com/en/articles/180 "Article: Averaging of price series without additional buffers for intermediate calculations").

This is not a useful EA but just an example of how to calculate it. I hope it is correct, it works in the tester.

It's also not my real programming style, but kept very simple.

```
#include <Trade\Trade.mqh> CTrade Trade;

//+------------------------------------------------------------------+
input int    inp_Dist     =  120;   // Distance pending order (points)
input int    inp_Stop     =  125;   // SL (points)
input int    inp_Take     =  150;   // TP (points)
input double inp_Volume   = 0.01;   // Volume

//+------------------------------------------------------------------+
double distPend = inp_Dist*_Point;
double distStop = inp_Stop*_Point;
double distTake = inp_Take*_Point;

//+------------------------------------------------------------------+
bool   done     = false;
double ask;
double bid;
double levelSell;
double levelBuy;
double stopSell;
double stopBuy;
double takeSell;
double takeBuy;

void OnTick()
{
   if(done)
      return;

   ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);

   levelBuy =NormalizeDouble(bid-distPend,_Digits);
   levelSell=NormalizeDouble(ask+distPend,_Digits);

   stopBuy  =NormalizeDouble(levelBuy -distStop,_Digits);
   stopSell =NormalizeDouble(levelSell+distStop,_Digits);

   takeBuy  =NormalizeDouble(levelBuy +distTake,_Digits);
   takeSell =NormalizeDouble(levelSell-distTake,_Digits);

   SellLimit();
   BuyLimit();

   done=true;
}

bool BuyLimit()
{
   //return(Trade.BuyLimit (inp_Volume,levelBuy ));
   return(Trade.BuyLimit (inp_Volume,levelBuy ,_Symbol,stopBuy,takeBuy ));
}

bool SellLimit()
{
   //return(Trade.SellLimit(inp_Volume,levelSell));
   return(Trade.SellLimit(inp_Volume,levelSell,_Symbol,stopSell,takeSell));
}
```

![How we developed the MetaTrader Signals service and Social Trading](https://c.mql5.com/2/11/signals_icon.png)[How we developed the MetaTrader Signals service and Social Trading](https://www.mql5.com/en/articles/1100)

We continue to enhance the Signals service, improve the mechanisms, add new functions and fix flaws. The MetaTrader Signals Service of 2012 and the current MetaTrader Signals Service are like two completely different services. Currently, we are implementing A Virtual Hosting Cloud service which consists of a network of servers to support specific versions of the MetaTrader client terminal.

![Tips for an Effective Product Presentation on the Market](https://c.mql5.com/2/11/ava_paint-market2.png)[Tips for an Effective Product Presentation on the Market](https://www.mql5.com/en/articles/999)

Selling programs to traders effectively does not only require writing an efficient and useful product and then publishing it on the Market. It is vital to provide a comprehensive, detailed description and good illustrations. A quality logo and correct screenshots are equally as important as the "true coding". Bear in mind a simple formula: no downloads = no sales.

![How we developed the MetaTrader Signals service and Social Trading](https://c.mql5.com/2/13/1200_3.png)[How we developed the MetaTrader Signals service and Social Trading](https://www.mql5.com/en/articles/1400)

We continue to enhance the Signals service, improve the mechanisms, add new functions and fix flaws. The MetaTrader Signals Service of 2012 and the current MetaTrader Signals Service are like two completely different services. Currently, we are implementing A Virtual Hosting Cloud service which consists of a network of servers to support specific versions of the MetaTrader client terminal. Traders will need to complete only 5 steps in order to rent the virtual copy of their terminal with minimal network latency to their broker's trade server directly from the MetaTrader client terminal.

![Freelance Jobs on MQL5.com - Developer's Favorite Place](https://c.mql5.com/2/10/ava_freelance-mql5.png)[Freelance Jobs on MQL5.com - Developer's Favorite Place](https://www.mql5.com/en/articles/1022)

Developers of trading robots no longer need to market their services to traders that require Expert Advisors - as now they will find you. Already, thousands of traders place orders to MQL5 freelance developers, and pay for work in on MQL5.com. For 4 years, this service facilitated three thousand traders to pay for more than 10 000 jobs performed. And the activity of traders and developers is constantly growing!

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nfljfiqoccpdgmooumymhzbpqalyxtbg&ssn=1769093010489496007&ssn_dr=0&ssn_sr=0&fv_date=1769093010&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F755&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Cookbook%20-%20Multi-Currency%20Expert%20Advisor%20and%20Working%20with%20Pending%20Orders%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909301050847726&fz_uniq=5049332165352728978&sv=2552)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).

![close](https://c.mql5.com/i/close.png)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)