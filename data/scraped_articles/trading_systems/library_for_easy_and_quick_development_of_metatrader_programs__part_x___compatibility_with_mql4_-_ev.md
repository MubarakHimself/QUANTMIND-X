---
title: Library for easy and quick development of MetaTrader programs (part X): Compatibility with MQL4 - Events of opening a position and activating pending orders
url: https://www.mql5.com/en/articles/6767
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:41:10.931656
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/6767&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070482169161258659)

MetaTrader 5 / Examples


### Contents

- [Test EA](https://www.mql5.com/en/articles/6767#node01)
- [Improving the library](https://www.mql5.com/en/articles/6767#node02)
- [Testing](https://www.mql5.com/en/articles/6767#node03)
- [What's next?](https://www.mql5.com/en/articles/6767#node04)


### Test EA

In the [previous article](https://www.mql5.com/en/articles/6651), we removed errors in the library files related to the
differences between MQL4 and MQL5, and introduced a collection of MQL4 historical orders and positions. In this article, we will continue
merging MQL4 and MQL5 in the library and define the events of opening positions and activating pending orders.

The sequence of improvement steps will be reversed. Previously, we introduced the functionality followed by the test EA. Now, in order
to understand what needs to be improved, we need to launch the test EA and see where it works and where it does not. The things that do not work are
the ones to be improved.

To achieve this, let's take the test EA **TestDoEasyPart08.mq5** from the [eighth \\
part](https://www.mql5.com/en/articles/6595) of the library description from the **\\MQL5\\Experts\\TestDoEasy\\Part08** folder and save it under the name **TestDoEasyPart10.mq4**
in the MetaTrader 4 folder **\\MQL4\\Experts\\TestDoEasy\\Part10**.

Let's try to compile it. This eventually leads to 34 compilation errors. Almost all of them are related to the absence of the trading classes in
the MQL4 standard library:

![](https://c.mql5.com/2/36/Errors34.png)

Let's move to the first error indicating the absence of the include file

![](https://c.mql5.com/2/36/Errors1.png)

and fix it —
the file is to be included only for MQL5:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart08.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#ifdef __MQL5__
#include <Trade\Trade.mqh>
#endif
//--- enums
```

Compilation ends in 33 errors. Move to the very first error again, which indicates the missing type when declaring the CTrade trading class object — it
is not present in MQL4.

Let's use the conditional compilation directive as before:

```
//--- global variables
CEngine        engine;
#ifdef __MQL5__
CTrade         trade;
#endif
SDataButt      butt_data[TOTAL_BUTT];
```

Compile. Now, the 'trade' object of the CTrade class has become unknown for MQL4. Fix this in a similar
way:

```
//--- setting trade parameters
#ifdef __MQL5__
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
#endif
//---
   return(INIT_SUCCEEDED);
  }
```

Frame all **trade** object instances into the conditional compilation
directives throughout the EA code using the #else
directive —

the MQL4 code is to be placed there. Let's use the very first error
of the unknown

**trade** type after making previous edits and compilations:

```
      //--- If the BUTT_BUY button is pressed: Open Buy position
      if(button==EnumToString(BUTT_BUY))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY,0,takeprofit);
         //--- Open Buy position
         #ifdef __MQL5__
            trade.Buy(lot,Symbol(),0,sl,tp);
         #else

         #endif
        }
```

After encasing all 'trade' object instances into the conditional compilation directive, we receive another error indicating that the
compiler cannot accurately define what call of an overloaded function should be used due to the lack of parameters:

![](https://c.mql5.com/2/36/Errors2.png)

If we look closely at the code, the reason for the compiler confusion becomes clear:

```
//+------------------------------------------------------------------+
//| Return the flag of a prefixed object presence                    |
//+------------------------------------------------------------------+
bool IsPresentObects(const string object_prefix)
  {
   for(int i=ObjectsTotal(0)-1;i>=0;i--)
      if(StringFind(ObjectName(0,i,0),object_prefix)>WRONG_VALUE)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
```

In [MQL5, the function](https://www.mql5.com/en/docs/objects/objectstotal) only has a single call form:

```
int  ObjectsTotal(
   long  chart_id,           // chart ID
   int   sub_window=-1,      // window index
   int   type=-1             // object type
   );
```

where the first parameter is a chart ID (0 - current),

while in MQL4 the function has two call forms for some time now. The first one is the same as in MQL5:

```
int  ObjectsTotal(
   long  chart_id,           // chart ID
   int   sub_window=-1,      // window index
   int   type=-1             // object type
   );
```

and the second one is outdated and has only one parameter:

```
int  ObjectsTotal(
   int   type=EMPTY         // object type
   );
```

In MQL5, passing 0 as a chart ID to the function (the current chart) does not cause any contradictions and doubts, but in MQL4, the compiler
should use the passed parameters to define the call type. In this case, it cannot accurately define whether we pass the current chart ID (0)
and the first call form should be used (after all, the other two parameters are set to their default values, which means we do not have to pass
them to the function), or we pass a window index (or an object type) and the second call form should be used.

The solution here is simple — pass the subwindow index (0 = main chart
window) as the second parameter:

```
//+------------------------------------------------------------------+
//| Return the flag of a prefixed object presence                    |
//+------------------------------------------------------------------+
bool IsPresentObects(const string object_prefix)
  {
   for(int i=ObjectsTotal(0,0)-1;i>=0;i--)
      if(StringFind(ObjectName(0,i,0),object_prefix)>WRONG_VALUE)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
```

and

```
//+------------------------------------------------------------------+
//| Manage button status                                             |
//+------------------------------------------------------------------+
void PressButtonsControl(void)
  {
   int total=ObjectsTotal(0,0);
   for(int i=0;i<total;i++)
     {
      string obj_name=ObjectName(0,i);
      if(StringFind(obj_name,prefix+"BUTT_")<0)
         continue;
      PressButtonEvents(obj_name);
     }
  }
//+------------------------------------------------------------------+
```

Now all is compiled with no errors. Before launching the test, keep in mind that the EA features no MQL4 trading functions as we excluded them
from the code using the conditional compilation directives, which means we need to add them.

As we write the code for the tester, we are not going to implement any checks required when trading on a real/demo account limiting
ourselves to minimal checks instead.

Since order and position tickets as well as calculated price levels are passed in the function, all we should do is select an
order/position by its ticket and check a close type and time. If the type does not coincide with an order or position type, display the
appropriate message and exit the function with an error. If an order is removed or a position is closed, display the message and exit with an
error. Next, call the opening/closing/modifying function and return its execution result.

At the end of the listing in the **DELib.mqh** file, write all necessary MQL4 tester functions:

```
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| MQL4 temporary functions for the tester                          |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
bool Buy(const double volume,const string symbol=NULL,const ulong magic=0,const double sl=0,const double tp=0,const string comment=NULL,const int deviation=2)
  {
   string sym=(symbol==NULL ? Symbol() : symbol);
   double price=0;
   ResetLastError();
   if(!SymbolInfoDouble(sym,SYMBOL_ASK,price))
     {
      Print(DFUN,TextByLanguage("Не удалось получить цену Ask. Ошибка ","Could not get Ask price. Error "),(string)GetLastError());
      return false;
     }
   if(!OrderSend(sym,ORDER_TYPE_BUY,volume,price,deviation,sl,tp,comment,(int)magic,0,clrBlue))
     {
      Print(DFUN,TextByLanguage("Не удалось открыть позицию Buy. Ошибка ","Failed to open a Buy position. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Set pending BuyLimit order                                       |
//+------------------------------------------------------------------+
bool BuyLimit(const double volume,const double price_set,const string symbol=NULL,const ulong magic=0,const double sl=0,const double tp=0,const string comment=NULL,const int deviation=2)
  {
   string sym=(symbol==NULL ? Symbol() : symbol);
   ResetLastError();
   if(!OrderSend(sym,ORDER_TYPE_BUY_LIMIT,volume,price_set,deviation,sl,tp,comment,(int)magic,0,clrBlue))
     {
      Print(DFUN,TextByLanguage("Не удалось установить ордер BuyLimit. Ошибка ","Could not place order BuyLimit. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Set pending BuyStop order                                        |
//+------------------------------------------------------------------+
bool BuyStop(const double volume,const double price_set,const string symbol=NULL,const ulong magic=0,const double sl=0,const double tp=0,const string comment=NULL,const int deviation=2)
  {
   string sym=(symbol==NULL ? Symbol() : symbol);
   ResetLastError();
   if(!OrderSend(sym,ORDER_TYPE_BUY_STOP,volume,price_set,deviation,sl,tp,comment,(int)magic,0,clrBlue))
     {
      Print(DFUN,TextByLanguage("Не удалось установить ордер BuyStop. Ошибка ","Could not place order BuyStop. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Open Sell position                                               |
//+------------------------------------------------------------------+
bool Sell(const double volume,const string symbol=NULL,const ulong magic=0,const double sl=0,const double tp=0,const string comment=NULL,const int deviation=2)
  {
   string sym=(symbol==NULL ? Symbol() : symbol);
   double price=0;
   ResetLastError();
   if(!SymbolInfoDouble(sym,SYMBOL_BID,price))
     {
      Print(DFUN,TextByLanguage("Не удалось получить цену Bid. Ошибка ","Could not get Bid price. Error "),(string)GetLastError());
      return false;
     }
   if(!OrderSend(sym,ORDER_TYPE_SELL,volume,price,deviation,sl,tp,comment,(int)magic,0,clrRed))
     {
      Print(DFUN,TextByLanguage("Не удалось открыть позицию Sell. Ошибка ","Failed to open a Sell position. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Set pending SellLimit order                                      |
//+------------------------------------------------------------------+
bool SellLimit(const double volume,const double price_set,const string symbol=NULL,const ulong magic=0,const double sl=0,const double tp=0,const string comment=NULL,const int deviation=2)
  {
   string sym=(symbol==NULL ? Symbol() : symbol);
   ResetLastError();
   if(!OrderSend(sym,ORDER_TYPE_SELL_LIMIT,volume,price_set,deviation,sl,tp,comment,(int)magic,0,clrRed))
     {
      Print(DFUN,TextByLanguage("Не удалось установить ордер SellLimit. Ошибка ","Could not place order SellLimit. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Set pending SellStop order                                       |
//+------------------------------------------------------------------+
bool SellStop(const double volume,const double price_set,const string symbol=NULL,const ulong magic=0,const double sl=0,const double tp=0,const string comment=NULL,const int deviation=2)
  {
   string sym=(symbol==NULL ? Symbol() : symbol);
   ResetLastError();
   if(!OrderSend(sym,ORDER_TYPE_SELL_STOP,volume,price_set,deviation,sl,tp,comment,(int)magic,0,clrRed))
     {
      Print(DFUN,TextByLanguage("Не удалось установить ордер SellStop. Ошибка ","Could not place order SellStop. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Close position by ticket                                         |
//+------------------------------------------------------------------+
bool PositionClose(const ulong ticket,const double volume=0,const int deviation=2)
  {
   ResetLastError();
   if(!OrderSelect((int)ticket,SELECT_BY_TICKET))
     {
      Print(DFUN,TextByLanguage("Не удалось выбрать позицию. Ошибка ","Could not select position. Error "),(string)GetLastError());
      return false;
     }
   if(OrderCloseTime()>0)
     {
      Print(DFUN,TextByLanguage("Позиция уже закрыта","Position already closed"));
      return false;
     }
   ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderType();
   if(type>ORDER_TYPE_SELL)
     {
      Print(DFUN,TextByLanguage("Ошибка. Не позиция: ","Error. Not position: "),OrderTypeDescription(type)," #",ticket);
      return false;
     }
   double price=0;
   color  clr=clrNONE;
   if(type==ORDER_TYPE_BUY)
     {
      price=SymbolInfoDouble(OrderSymbol(),SYMBOL_BID);
      clr=clrBlue;
     }
   else
     {
      price=SymbolInfoDouble(OrderSymbol(),SYMBOL_ASK);
      clr=clrRed;
     }
   double vol=(volume==0 || volume>OrderLots() ? OrderLots() : volume);
   ResetLastError();
   if(!OrderClose((int)ticket,vol,price,deviation,clr))
     {
      Print(DFUN,TextByLanguage("Не удалось закрыть позицию. Ошибка ","Could not close position. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Close position by an opposite one                                |
//+------------------------------------------------------------------+
bool PositionCloseBy(const ulong ticket,const ulong ticket_by)
  {
   ResetLastError();
   if(!OrderSelect((int)ticket,SELECT_BY_TICKET))
     {
      Print(DFUN,TextByLanguage("Не удалось выбрать позицию. Ошибка ","Could not select position. Error "),(string)GetLastError());
      return false;
     }
   if(OrderCloseTime()>0)
     {
      Print(DFUN,TextByLanguage("Позиция уже закрыта","Position already closed"));
      return false;
     }
   ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderType();
   if(type>ORDER_TYPE_SELL)
     {
      Print(DFUN,TextByLanguage("Ошибка. Не позиция: ","Error. Not position: "),OrderTypeDescription(type)," #",ticket);
      return false;
     }
   ResetLastError();
   if(!OrderSelect((int)ticket_by,SELECT_BY_TICKET))
     {
      Print(DFUN,TextByLanguage("Не удалось выбрать встречную позицию. Ошибка ","Could not select the opposite position. Error "),(string)GetLastError());
      return false;
     }
   if(OrderCloseTime()>0)
     {
      Print(DFUN,TextByLanguage("Встречная позиция уже закрыта","Opposite position already closed"));
      return false;
     }
   ENUM_ORDER_TYPE type_by=(ENUM_ORDER_TYPE)OrderType();
   if(type_by>ORDER_TYPE_SELL)
     {
      Print(DFUN,TextByLanguage("Ошибка. Встречная позиция не является позицией: ","Error. Opposite position is not a position: "),OrderTypeDescription(type_by)," #",ticket_by);
      return false;
     }
   color clr=(type==ORDER_TYPE_BUY ? clrBlue : clrRed);
   ResetLastError();
   if(!OrderCloseBy((int)ticket,(int)ticket_by,clr))
     {
      Print(DFUN,TextByLanguage("Не удалось закрыть позицию встречной. Ошибка ","Could not close position by opposite position. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Remove a pending order by ticket                                 |
//+------------------------------------------------------------------+
bool PendingOrderDelete(const ulong ticket)
  {
   ResetLastError();
   if(!OrderSelect((int)ticket,SELECT_BY_TICKET))
     {
      Print(DFUN,TextByLanguage("Не удалось выбрать ордер. Ошибка ","Could not select order. Error "),(string)GetLastError());
      return false;
     }
   if(OrderCloseTime()>0)
     {
      Print(DFUN,TextByLanguage("Ордер уже удалён","Order already deleted"));
      return false;
     }
   ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderType();
   if(type<ORDER_TYPE_SELL || type>ORDER_TYPE_SELL_STOP)
     {
      Print(DFUN,TextByLanguage("Ошибка. Не ордер: ","Error. Not order: "),PositionTypeDescription((ENUM_POSITION_TYPE)type)," #",ticket);
      return false;
     }
   color clr=(type<ORDER_TYPE_SELL_LIMIT ? clrBlue : clrRed);
   ResetLastError();
   if(!OrderDelete((int)ticket,clr))
     {
      Print(DFUN,TextByLanguage("Не удалось удалить ордер. Ошибка ","Could not delete order. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Modify position by ticket                                        |
//+------------------------------------------------------------------+
bool PositionModify(const ulong ticket,const double sl,const double tp)
  {
   ResetLastError();
   if(!OrderSelect((int)ticket,SELECT_BY_TICKET))
     {
      Print(DFUN,TextByLanguage("Не удалось выбрать позицию. Ошибка ","Could not select position. Error "),(string)GetLastError());
      return false;
     }
   ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderType();
   if(type>ORDER_TYPE_SELL)
     {
      Print(DFUN,TextByLanguage("Ошибка. Не позиция: ","Error. Not position: "),OrderTypeDescription(type)," #",ticket);
      return false;
     }
   if(OrderCloseTime()>0)
     {
      Print(DFUN,TextByLanguage("Ошибка. Для модификации выбрана закрытая позиция: ","Error. Closed position selected for modification: "),PositionTypeDescription((ENUM_POSITION_TYPE)type)," #",ticket);
      return false;
     }
   color clr=(type==ORDER_TYPE_BUY ? clrBlue : clrRed);
   ResetLastError();
   if(!OrderModify((int)ticket,OrderOpenPrice(),sl,tp,0,clr))
     {
      Print(DFUN,TextByLanguage("Не удалось модифицировать позицию. Ошибка ","Failed to modify position. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Modify pending order by ticket                                   |
//+------------------------------------------------------------------+
bool PendingOrderModify(const ulong ticket,const double price_set,const double sl,const double tp)
  {
   ResetLastError();
   if(!OrderSelect((int)ticket,SELECT_BY_TICKET))
     {
      Print(DFUN,TextByLanguage("Не удалось выбрать ордер. Ошибка ","Could not select order. Error "),(string)GetLastError());
      return false;
     }
   ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderType();
   if(type<ORDER_TYPE_SELL || type>ORDER_TYPE_SELL_STOP)
     {
      Print(DFUN,TextByLanguage("Ошибка. Не ордер: ","Error. Not order: "),PositionTypeDescription((ENUM_POSITION_TYPE)type)," #",ticket);
      return false;
     }
   if(OrderCloseTime()>0)
     {
      Print(DFUN,TextByLanguage("Ошибка. Для модификации выбран удалённый ордер: ","Error. Deleted order selected for modification: "),OrderTypeDescription(type)," #",ticket);
      return false;
     }
   color clr=(type<ORDER_TYPE_SELL_LIMIT ? clrBlue : clrRed);
   ResetLastError();
   if(!OrderModify((int)ticket,price_set,sl,tp,0,clr))
     {
      Print(DFUN,TextByLanguage("Не удалось модифицировать ордер. Ошибка ","Failed to modify order. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
#endif
```

These functions are temporary. Soon we will write the full-fledged trading classes for MQL5 and MQL4 and remove these functions from the
listing.

Now we need to add the call of newly written functions wherever we have left a place in the EA code for calling the MQL4 trading functions. Press
Ctrl+F and enter

**trade** to the search box. Thus, we will quickly find the code passages where the calls of trading MQL4 functions are to be set.

Implement the call of MQL4 trading functions where it is necessary starting with the PressButtonEvents() function for handling button press
events and down to the end of the listing. The code is quite bulky, while selection of the necessary function is unambiguous. Therefore, I
will not display the code here. You can find it in the files attached to the article. We will only have a look at handling the pressing of two
buttons —

the button for opening a Buy position and the
button for placing a BuyLimit pending order:

```
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   //--- Convert the button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
   //--- If the button is pressed
   if(ButtonState(button_name))
     {
      //--- If the BUTT_BUY button is pressed: Open Buy position
      if(button==EnumToString(BUTT_BUY))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY,0,takeprofit);
         //--- Open Buy position
         #ifdef __MQL5__
            trade.Buy(lot,Symbol(),0,sl,tp);
         #else
            Buy(lot,Symbol(),magic_number,sl,tp);
         #endif
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Set BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,takeprofit);
         //--- Set BuyLimit order
         #ifdef __MQL5__
            trade.BuyLimit(lot,price_set,Symbol(),sl,tp);
         #else
            BuyLimit(lot,price_set,Symbol(),magic_number,sl,tp);
         #endif
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
```

When testing the library code, I noticed something strange: events MQL4 sees without improving the code are displayed in the journal only
after a while. After some delving into the matter, I realized that the reason is in the counter of the collection timer working in the
CEngine timer. We have set the minimum delay of 16 milliseconds for the collection timer counter we developed

[in the third part](https://www.mql5.com/en/articles/5687#node02) of the library description when creating the
library basic object. However, since we do not work with the timer in the tester and call the OnTimer() library handler directly from
OnTick() to work by ticks, the delay of 16 milliseconds turns into the delay of 16 ticks. To fix this, I slightly modified the CEngine
class introducing the method returning the tester flag and handling the work in the tester in the OnTimer() handler, which in turn is
called from the EA's OnTick() when working in the tester.

A private class member variable and the
method returning the variable value were created to make changes:

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Collection of events
   CArrayObj            m_list_counters;                 // List of timer counters
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_tester;                     // Flag of working in the tester
   bool                 m_is_market_trade_event;         // Flag of an account trading event
   bool                 m_is_history_trade_event;        // Flag of an account history trading event
   ENUM_TRADE_EVENT     m_acc_trade_event;               // Account trading event
//--- Return counter index by id
```

```
public:
   //--- Return the list of market (1) positions, (2) pending orders and (3) market orders
   CArrayObj*           GetListMarketPosition(void);
   CArrayObj*           GetListMarketPendings(void);
   CArrayObj*           GetListMarketOrders(void);
   //--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals, (4) all position market orders by its id
   CArrayObj*           GetListHistoryOrders(void);
   CArrayObj*           GetListHistoryPendings(void);
   CArrayObj*           GetListDeals(void);
   CArrayObj*           GetListAllOrdersByPosID(const ulong position_id);
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_events.ResetLastTradeEvent(); }
//--- Return the (1) last trading event, (2) hedge account flag, (3) flag of working in the tester
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_acc_trade_event;       }
   bool                 IsHedge(void)                             const { return this.m_is_hedge;              }
   bool                 IsTester(void)                            const { return this.m_is_tester;             }
//--- Create the timer counter
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

The value of this tester flag variable is set in the class constructor:

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),m_acc_trade_event(TRADE_EVENT_NO_EVENT)
  {
   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_COUNTER_ID,COLLECTION_COUNTER_STEP,COLLECTION_PAUSE);
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_is_tester=::MQLInfoInteger(MQL_TESTER);
   ::ResetLastError();
   #ifdef __MQL5__
      if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
         ::Print(DFUN,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
   //---__MQL4__
   #else
      if(!this.IsTester() && !::EventSetMillisecondTimer(TIMER_FREQUENCY))
         ::Print(DFUN,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
   #endif
  }
//+------------------------------------------------------------------+
```

In the OnTimer() handler of the CEngine class, check working in the tester and, depending on whether the work

**is performed** in the tester or not, work either by
the

timer counter or by
tick:

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of historical orders, deals, market orders and positions collections
   int index=this.CounterIndex(COLLECTION_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the collections events
            if(counter.IsTimeDone())
               this.TradeEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
           {
            this.TradeEventsControl();
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

**Compile the EA, launch it in the tester and try the buttons:**

![](https://c.mql5.com/2/36/Test1.gif)

The messages indicate that the library sees some events: setting a pending order and modifying order and position parameters. It cannot see
other events yet.

Let's deal with the errors.

### Improving the library

The first thing we should look at is why the library does not see the removal of a pending order.


All events are tracked in the method of the

**CEventsCollection::Refresh()** event collection class. We are interested in the account history events. Let's pass to the method
and have a look at the code responsible for

tracking changes in the collection of MQL5 historical orders and deals:

```
     }
//--- If the event is in the account history
   if(is_history_event)
     {
      //--- If the number of historical orders increased
      if(new_history_orders>0)
        {
         //--- Receive the list of removed pending orders only
         CArrayObj* list=this.GetListHistoryPendings(list_history);
         if(list!=NULL)
           {
            Print(DFUN);
            //--- Sort the new list by order removal time
            list.Sort(SORT_BY_ORDER_TIME_CLOSE_MSC);
            //--- Take the number of orders equal to the number of newly removed ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_history_orders;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive an order from the list. If this is a removed pending order without a position ID,
               //--- this is an order removal - set a trading event
               COrder* order=list.At(i);
               if(order!=NULL && order.Status()==ORDER_STATUS_HISTORY_PENDING && order.PositionID()==0)
                  this.CreateNewEvent(order,list_history,list_market);
              }
           }
        }
      //--- If the number of deals increased
```

The order property specifying the position ID is not filled in
(equal to zero). After finding the right passage, we can
see that we used that feature for accurate identification of a pending order deletion (rather than activation) in MQL5 (in MQL5, if an order
was activated and led to a deal and a position, the position ID would be equal to the ID of the position opened as a result of the order
activation). In MQL4, this field is immediately filled with the order ticket, which is incorrect.

Go to the abstract order's closed class constructor and find the
order property string containing the position ID:

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
COrder::COrder(ENUM_ORDER_STATUS order_status,const ulong ticket)
  {
//--- Save integer properties
   this.m_ticket=ticket;
   this.m_long_prop[ORDER_PROP_STATUS]                               = order_status;
   this.m_long_prop[ORDER_PROP_MAGIC]                                = this.OrderMagicNumber();
   this.m_long_prop[ORDER_PROP_TICKET]                               = this.OrderTicket();
   this.m_long_prop[ORDER_PROP_TIME_OPEN]                            = (long)(ulong)this.OrderOpenTime();
   this.m_long_prop[ORDER_PROP_TIME_CLOSE]                           = (long)(ulong)this.OrderCloseTime();
   this.m_long_prop[ORDER_PROP_TIME_EXP]                             = (long)(ulong)this.OrderExpiration();
   this.m_long_prop[ORDER_PROP_TYPE]                                 = this.OrderType();
   this.m_long_prop[ORDER_PROP_STATE]                                = this.OrderState();
   this.m_long_prop[ORDER_PROP_DIRECTION]                            = this.OrderTypeByDirection();
   this.m_long_prop[ORDER_PROP_POSITION_ID]                          = this.OrderPositionID();
   this.m_long_prop[ORDER_PROP_REASON]                               = this.OrderReason();
   this.m_long_prop[ORDER_PROP_DEAL_ORDER_TICKET]                    = this.DealOrderTicket();
   this.m_long_prop[ORDER_PROP_DEAL_ENTRY]                           = this.DealEntry();
   this.m_long_prop[ORDER_PROP_POSITION_BY_ID]                       = this.OrderPositionByID();
   this.m_long_prop[ORDER_PROP_TIME_OPEN_MSC]                        = this.OrderOpenTimeMSC();
   this.m_long_prop[ORDER_PROP_TIME_CLOSE_MSC]                       = this.OrderCloseTimeMSC();
   this.m_long_prop[ORDER_PROP_TIME_UPDATE]                          = (long)(ulong)this.PositionTimeUpdate();
   this.m_long_prop[ORDER_PROP_TIME_UPDATE_MSC]                      = (long)(ulong)this.PositionTimeUpdateMSC();

//--- Save real properties
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_OPEN)]         = this.OrderOpenPrice();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_CLOSE)]        = this.OrderClosePrice();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PROFIT)]             = this.OrderProfit();
   this.m_double_prop[this.IndexProp(ORDER_PROP_COMMISSION)]         = this.OrderCommission();
   this.m_double_prop[this.IndexProp(ORDER_PROP_SWAP)]               = this.OrderSwap();
   this.m_double_prop[this.IndexProp(ORDER_PROP_VOLUME)]             = this.OrderVolume();
   this.m_double_prop[this.IndexProp(ORDER_PROP_SL)]                 = this.OrderStopLoss();
   this.m_double_prop[this.IndexProp(ORDER_PROP_TP)]                 = this.OrderTakeProfit();
   this.m_double_prop[this.IndexProp(ORDER_PROP_VOLUME_CURRENT)]     = this.OrderVolumeCurrent();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_STOP_LIMIT)]   = this.OrderPriceStopLimit();

//--- Save string properties
   this.m_string_prop[this.IndexProp(ORDER_PROP_SYMBOL)]             = this.OrderSymbol();
   this.m_string_prop[this.IndexProp(ORDER_PROP_COMMENT)]            = this.OrderComment();
   this.m_string_prop[this.IndexProp(ORDER_PROP_EXT_ID)]             = this.OrderExternalID();

//--- Save additional integer properties
   this.m_long_prop[ORDER_PROP_PROFIT_PT]                            = this.ProfitInPoints();
   this.m_long_prop[ORDER_PROP_TICKET_FROM]                          = this.OrderTicketFrom();
   this.m_long_prop[ORDER_PROP_TICKET_TO]                            = this.OrderTicketTo();
   this.m_long_prop[ORDER_PROP_CLOSE_BY_SL]                          = this.OrderCloseByStopLoss();
   this.m_long_prop[ORDER_PROP_CLOSE_BY_TP]                          = this.OrderCloseByTakeProfit();
   this.m_long_prop[ORDER_PROP_GROUP_ID]                             = 0;

//--- Save additional real properties
   this.m_double_prop[this.IndexProp(ORDER_PROP_PROFIT_FULL)]        = this.ProfitFull();
  }
//+------------------------------------------------------------------+
```

This is done by the **OrderPositionID()** method. As we can see, in MQL4, the
ticket is set as the ID right away:

```
//+------------------------------------------------------------------+
//| Return position ID                                               |
//+------------------------------------------------------------------+
long COrder::OrderPositionID(void) const
  {
#ifdef __MQL4__
   return ::OrderTicket();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_POSITION   : res=::PositionGetInteger(POSITION_IDENTIFIER);            break;
      case ORDER_STATUS_MARKET_ORDER      :
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_POSITION_ID);                 break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_POSITION_ID); break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_POSITION_ID);   break;
      default                             : res=0;                                                    break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
```

Initially, 0 should be set there (no open position when removing the order). This
is what we do:

```
//+------------------------------------------------------------------+
//| Return position ID                                               |
//+------------------------------------------------------------------+
long COrder::OrderPositionID(void) const
  {
#ifdef __MQL4__
   return 0;
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_POSITION   : res=::PositionGetInteger(POSITION_IDENTIFIER);            break;
      case ORDER_STATUS_MARKET_ORDER      :
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_POSITION_ID);                 break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_POSITION_ID); break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_POSITION_ID);   break;
      default                             : res=0;                                                    break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
```

**Compile the EA, launch it in the tester and then set and remove a pending order:**

![](https://c.mql5.com/2/36/Test2.gif)

Now the event of a pending order removal is tracked.

If we wait for the pending order activation, we will see again that this event, just like a simple position opening, is not visible for the
library. Let's define the reasons.

As we remember, all starts from the **OnTimer()** handler of the **CEngine** class:

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of historical orders, deals, market orders and positions collections
   int index=this.CounterIndex(COLLECTION_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the collections events
            if(counter.IsTimeDone())
               this.TradeEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
           {
            this.TradeEventsControl();
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

According to the code, the events are managed in the TradeEventsControl() method. In
case of any event, we call the method for updating the events of the **CEventsCollection::Refresh()**
event collection class:

```
//+------------------------------------------------------------------+
//| Check trading events                                             |
//+------------------------------------------------------------------+
void CEngine::TradeEventsControl(void)
  {
//--- Initialize the trading events code and flags
   this.m_is_market_trade_event=false;
   this.m_is_history_trade_event=false;
//--- Update the lists
   this.m_market.Refresh();
   this.m_history.Refresh();
//--- First launch actions
   if(this.IsFirstStart())
     {
      this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;
      return;
     }
//--- Check the changes in the market status and account history
   this.m_is_market_trade_event=this.m_market.IsTradeEvent();
   this.m_is_history_trade_event=this.m_history.IsTradeEvent();

//--- If there is any event, send the lists, the flags and the number of new orders and deals to the event collection, and update it
   int change_total=0;
   CArrayObj* list_changes=this.m_market.GetListChanges();
   if(list_changes!=NULL)
      change_total=list_changes.Total();
   if(this.m_is_history_trade_event || this.m_is_market_trade_event || change_total>0)
     {
      this.m_events.Refresh(this.m_history.GetList(),this.m_market.GetList(),list_changes,
                            this.m_is_history_trade_event,this.m_is_market_trade_event,
                            this.m_history.NewOrders(),this.m_market.NewPendingOrders(),
                            this.m_market.NewMarketOrders(),this.m_history.NewDeals());
      //--- Get the account's last trading event
      this.m_acc_trade_event=this.m_events.GetLastTradeEvent();
     }
  }
//+------------------------------------------------------------------+
```

Here we send the lists of historical and market collections, flags of changes in the collections, the number of new historical orders and
active market orders and

positions, as well as the number of new deals to the method. But a closer
look reveals that instead of the number of new market positions, the method receives the

number of new market orders we have not used in the library yet. This
is my error. Initially, everything was developed for MQL5, while the number of new positions should be sent for the MQL4 method. In MQL5, new
positions are defined by the number of deals. The error occurred when I filled in the passed data for the MQL4 method. Now it is clear why the
method cannot see the new market positions.

Let's fix this and solve another issue along the way:

Unlike MQL5, MQL4 features no ability to find an order that led to
opening a position. However, we already have a list of control orders for tracking changes of order and position properties. We have not yet
cleared this list of unnecessary data. This list will help us to track an order that led to opening a position and identify the event — a market
order or a pending order activation.

**Add the public method returning the list of control orders**
**to the collection of market orders and positions (CMarketCollection class in the MarketCollection.mqh file):**

```
public:
//--- Return the list (1) of all pending orders and open positions, (2) control orders and positions
   CArrayObj*        GetList(void)                                                                       { return &this.m_list_all_orders;                                       }
   CArrayObj*        GetListChanges(void)                                                                { return &this.m_list_changed;                                          }
   CArrayObj*        GetListControl(void)                                                                { return &this.m_list_control;                                          }
//--- Return the list of orders and positions with an open time from begin_time to end_time
   CArrayObj*        GetListByTime(const datetime begin_time=0,const datetime end_time=0);
//--- Return the list of orders and positions by selected (1) double, (2) integer and (3) string property fitting a compared condition
   CArrayObj*        GetList(ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   CArrayObj*        GetList(ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   CArrayObj*        GetList(ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
//--- Return the number of (1) new market orders, (2) new pending orders, (3) new positions, (4) occurred trading event flag, (5) changed volume
   int               NewMarketOrders(void)                                                         const { return this.m_new_market_orders;                                      }
   int               NewPendingOrders(void)                                                        const { return this.m_new_pendings;                                           }
   int               NewPositions(void)                                                            const { return this.m_new_positions;                                          }
   bool              IsTradeEvent(void)                                                            const { return this.m_is_trade_event;                                         }
   double            ChangedVolumeValue(void)                                                      const { return this.m_change_volume_value;                                    }
//--- Constructor
                     CMarketCollection(void);
//--- Update the list of pending orders and positions
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

To use data from the list, we need to pass it to the Refresh() method of the CEventsCollection class.

**To do this, write all the necessary changes described**
**above:**

```
//+------------------------------------------------------------------+
//| Check trading events                                             |
//+------------------------------------------------------------------+
void CEngine::TradeEventsControl(void)
  {
//--- Initialize the trading events code and flags
   this.m_is_market_trade_event=false;
   this.m_is_history_trade_event=false;
//--- Update the lists
   this.m_market.Refresh();
   this.m_history.Refresh();
//--- First launch actions
   if(this.IsFirstStart())
     {
      this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;
      return;
     }
//--- Check the changes in the market status and account history
   this.m_is_market_trade_event=this.m_market.IsTradeEvent();
   this.m_is_history_trade_event=this.m_history.IsTradeEvent();

//--- If there is any event, send the lists, the flags and the number of new orders and deals to the event collection, and update it
   int change_total=0;
   CArrayObj* list_changes=this.m_market.GetListChanges();
   if(list_changes!=NULL)
      change_total=list_changes.Total();
   if(this.m_is_history_trade_event || this.m_is_market_trade_event || change_total>0)
     {
      this.m_events.Refresh(this.m_history.GetList(),this.m_market.GetList(),list_changes,this.m_market.GetListControl(),
                            this.m_is_history_trade_event,this.m_is_market_trade_event,
                            this.m_history.NewOrders(),this.m_market.NewPendingOrders(),
                            this.m_market.NewPositions(),this.m_history.NewDeals());
      //--- Get the account's last trading event
      this.m_acc_trade_event=this.m_events.GetLastTradeEvent();
     }
  }
//+------------------------------------------------------------------+
```

Here in the **TradeEventsControl()** method of the **CEngine** class, we added passing
yet another list — the list of control orders to the **Refresh()**
method of the **CEventsCollection** class and replaced the erroneous passing of a number of new market orders to the method with passing
a number of new positions.

**Let's make corrections to the defining of the Refresh()**
**method in the CEventsCollection class body:**

```
public:
//--- Select events from the collection with time within the range from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0);
//--- Return the full event collection list "as is"
   CArrayObj        *GetList(void)                                                                       { return &this.m_list_events;                                           }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_EVENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_EVENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_EVENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
//--- Update the list of events
   void              Refresh(CArrayObj* list_history,
                             CArrayObj* list_market,
                             CArrayObj* list_changes,
                             CArrayObj* list_control,
                             const bool is_history_event,
                             const bool is_market_event,
                             const int  new_history_orders,
                             const int  new_market_pendings,
                             const int  new_market_positions,
                             const int  new_deals);
//--- Set the control program chart ID
   void              SetChartID(const long id)        { this.m_chart_id=id;         }
//--- Return the last trading event on the account
   ENUM_TRADE_EVENT  GetLastTradeEvent(void)    const { return this.m_trade_event;  }
//--- Reset the last trading event
   void              ResetLastTradeEvent(void)        { this.m_trade_event=TRADE_EVENT_NO_EVENT;   }
//--- Constructor
                     CEventsCollection(void);
  };
//+------------------------------------------------------------------+
```

and in its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Update the event list                                            |
//+------------------------------------------------------------------+
void CEventsCollection::Refresh(CArrayObj* list_history,
                                CArrayObj* list_market,
                                CArrayObj* list_changes,
                                CArrayObj* list_control,
                                const bool is_history_event,
                                const bool is_market_event,
                                const int  new_history_orders,
                                const int  new_market_pendings,
                                const int  new_market_positions,
                                const int  new_deals)
  {
```

The method for updating the event list of the event collection class events is still missing handling the event of opening a position for MQL4.
We will need a few methods for it.

To get the list of open positions, we should have the method of
obtaining it. Besides, we do not have the method for using the list of
control orders to define a type of the order, which led to opening a position.

We will also need two private class members for
storing the

type of an opening order found in the list of control orders and a position
ID. The type and the ID are to be defined in the code block for handling market position opening events for MQL4.

**Add them to the**
**private class section:**

```
//+------------------------------------------------------------------+
//| Collection of account events                                     |
//+------------------------------------------------------------------+
class CEventsCollection : public CListObj
  {
private:
   CListObj          m_list_events;                   // List of events
   bool              m_is_hedge;                      // Hedge account flag
   long              m_chart_id;                      // Control program chart ID
   int               m_trade_event_code;              // Trading event code
   ENUM_TRADE_EVENT  m_trade_event;                   // Account trading event
   CEvent            m_event_instance;                // Event object for searching by property
   MqlTick           m_tick;                          // Last tick structure
   ulong             m_position_id;                   // Position ID (MQL4)
   ENUM_ORDER_TYPE   m_type_first;                    // Opening order type (MQL4)

//--- Create a trading event depending on the order (1) status and (2) change type
   void              CreateNewEvent(COrder* order,CArrayObj* list_history,CArrayObj* list_market);
   void              CreateNewEvent(COrderControl* order);
//--- Create an event for a (1) hedging account, (2) netting account
   void              NewDealEventHedge(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
   void              NewDealEventNetto(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
//--- Select from the list and return the list of (1) market pending orders, (2) open positions
   CArrayObj*        GetListMarketPendings(CArrayObj* list);
   CArrayObj*        GetListPositions(CArrayObj* list);
//--- Select from the list and return the list of historical (1) removed pending orders, (2) deals, (3) all closing orders
   CArrayObj*        GetListHistoryPendings(CArrayObj* list);
   CArrayObj*        GetListDeals(CArrayObj* list);
   CArrayObj*        GetListCloseByOrders(CArrayObj* list);
//--- Return the list of (1) all position orders by its ID, (2) all deal positions by its ID
//--- (3) all market entry deals by position ID, (4) all market exit deals by position ID,
//--- (5) all position reversal deals by position ID
   CArrayObj*        GetListAllOrdersByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsInByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsOutByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsInOutByPosID(CArrayObj* list,const ulong position_id);
//--- Return the total volume of all deals (1) IN, (2) OUT of the position by its ID
   double            SummaryVolumeDealsInByPosID(CArrayObj* list,const ulong position_id);
   double            SummaryVolumeDealsOutByPosID(CArrayObj* list,const ulong position_id);
//--- Return the (1) first, (2) last and (3) closing order from the list of all position orders,
//--- (4) an order by ticket, (5) market position by ID,
//--- (6) the last and (7) penultimate InOut deal by position ID
   COrder*           GetFirstOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetLastOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetCloseByOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetHistoryOrderByTicket(CArrayObj* list,const ulong order_ticket);
   COrder*           GetPositionByID(CArrayObj* list,const ulong position_id);
//--- Return the type of the opening order by the position ticket (MQL4)
   ENUM_ORDER_TYPE   GetTypeFirst(CArrayObj* list,const ulong ticket);
//--- Return the flag of the event object presence in the event list
   bool              IsPresentEventInList(CEvent* compared_event);
//--- Existing order/position change event handler
   void              OnChangeEvent(CArrayObj* list_changes,const int index);

public:
```

**Implement the method for receiving the list of open positions** **outside the class body:**

```
//+------------------------------------------------------------------+
//| Select only market positions from the list                       |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListPositions(CArrayObj *list)
  {
   if(list.Type()!=COLLECTION_MARKET_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком рыночной коллекции","Error. The list is not a list of the market collection"));
      return NULL;
     }
   CArrayObj* list_positions=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   return list_positions;
  }
//+------------------------------------------------------------------+
```

The full list of market orders and positions is passed to the
method and

sorted out by
"market position" status. The resulting list is returned
to the calling program.

Let's write the **method returning a type of the order, which led to opening a position:**

```
//+------------------------------------------------------------------+
//| Return the type of an opening order by position ticket (MQL4)    |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE CEventsCollection::GetTypeFirst(CArrayObj* list,const ulong ticket)
  {
   if(list==NULL)
      return WRONG_VALUE;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      COrderControl* ctrl=list.At(i);
      if(ctrl==NULL)
         continue;
      if(ctrl.Ticket()==ticket)
         return (ENUM_ORDER_TYPE)ctrl.TypeOrder();
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

The list of control orders and the
ticket of a newly opened position are passed to the method. Next, in
a loop from the beginning of the list (assuming that a pending order was placed before other open positions, so that its ticket comes
up faster),

get the control order from the list and compare its ticket with the
one passed to the function.

If the ticket is found, this order is an opening one for the position
whose ticket has been passed to the method —

return the order type. If no order with such a ticket is found, return
-1.

Now we can improve handling events with positions for MQL4.

**Add handling position opening for MQL4 to the event list**
**update method:**

```
//+------------------------------------------------------------------+
//| Update the event list                                            |
//+------------------------------------------------------------------+
void CEventsCollection::Refresh(CArrayObj* list_history,
                                CArrayObj* list_market,
                                CArrayObj* list_changes,
                                CArrayObj* list_control,
                                const bool is_history_event,
                                const bool is_market_event,
                                const int  new_history_orders,
                                const int  new_market_pendings,
                                const int  new_market_positions,
                                const int  new_deals)
  {
//--- Exit if the lists are empty
   if(list_history==NULL || list_market==NULL)
      return;
//--- If the event is in the market environment
   if(is_market_event)
     {
      //--- if the order properties were changed
      int total_changes=list_changes.Total();
      if(total_changes>0)
        {
         for(int i=total_changes-1;i>=0;i--)
           {
            this.OnChangeEvent(list_changes,i);
           }
        }
      //--- if the number of placed pending orders increased
      if(new_market_pendings>0)
        {
         //--- Receive the list of the newly placed pending orders
         CArrayObj* list=this.GetListMarketPendings(list_market);
         if(list!=NULL)
           {
            //--- Sort the new list by order placement time
            list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
            //--- Take the number of orders equal to the number of newly placed ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_market_pendings;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive an order from the list, if this is a pending order, set a trading event
               COrder* order=list.At(i);
               if(order!=NULL && order.Status()==ORDER_STATUS_MARKET_PENDING)
                  this.CreateNewEvent(order,list_history,list_market);
              }
           }
        }
      #ifdef __MQL4__
      //--- If the number of positions increased
      if(new_market_positions>0)
        {
         //--- Get the list of open positions
         CArrayObj* list=this.GetListPositions(list_market);
         if(list!=NULL)
           {
            //--- Sort the new list by a position open time
            list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
            //--- Take the number of positions equal to the number of newly placed open positions from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_market_positions;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive a position from the list. If this is a position, search for opening order data and set a trading event
               COrder* position=list.At(i);
               if(position!=NULL && position.Status()==ORDER_STATUS_MARKET_POSITION)
                 {
                  //--- Find an order and set (1) a type of an order that led to opening a position and a (2) position ID
                  this.m_type_first=this.GetTypeFirst(list_control,position.Ticket());
                  this.m_position_id=position.Ticket();
                  this.CreateNewEvent(position,list_history,list_market);
                 }
              }
           }
        }
      #endif
     }
//--- If the event is in the account history
   if(is_history_event)
     {
      //--- If the number of historical orders increased
      if(new_history_orders>0)
        {
         //--- Receive the list of removed pending orders only
         CArrayObj* list=this.GetListHistoryPendings(list_history);
         if(list!=NULL)
           {
            //--- Sort the new list by order removal time
            list.Sort(SORT_BY_ORDER_TIME_CLOSE_MSC);
            //--- Take the number of orders equal to the number of newly removed pending ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_history_orders;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive an order from the list. If this is a removed pending order without a position ID,
               //--- this is an order removal - set a trading event
               COrder* order=list.At(i);
               if(order!=NULL && order.Status()==ORDER_STATUS_HISTORY_PENDING && order.PositionID()==0)
                  this.CreateNewEvent(order,list_history,list_market);
              }
           }
        }
      //--- If the number of deals increased
      if(new_deals>0)
        {
         //--- Receive the list of deals only
         CArrayObj* list=this.GetListDeals(list_history);
         if(list!=NULL)
           {
            //--- Sort the new list by deal time
            list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
            //--- Take the number of deals equal to the number of new ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_deals;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive a deal from the list and set a trading event
               COrder* order=list.At(i);
               if(order!=NULL)
                  this.CreateNewEvent(order,list_history,list_market);
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

All actions for handling opening a new position or triggering a pending order for
MQL4 are described in the code comments and do not need
any additional explanation.

Now let's move to the **CEventsCollection::CreateNewEvent()** method for creating a new event and find the code block
responsible for creating a position opening event for MQL4 (the start of the block is

marked in the code comments) and supplement the position opening
event definition and the

reasons for its opening, as well as add data
on the appropriate order and position ID to the open
position data:

```
//--- Position opened (__MQL4__)
   if(status==ORDER_STATUS_MARKET_POSITION)
     {
      //--- Set the "position opened" trading event code
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_OPENED;
      //--- Set the "request executed partially" reason
      ENUM_EVENT_REASON reason=EVENT_REASON_DONE;
      //--- If an opening order is a pending one
      if(this.m_type_first>ORDER_TYPE_SELL && this.m_type_first<ORDER_TYPE_BALANCE)
        {
         //--- set the "pending order activated" reason
         reason=EVENT_REASON_ACTIVATED_PENDING;
         //--- add a pending order activation flag to the event code
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_ACTIVATED;
        }
      CEvent* event=new CEventPositionOpen(this.m_trade_event_code,order.Ticket());
      if(event!=NULL)
        {
         event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpenMSC());                             // Event time
         event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                                        // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,this.m_type_first);                          // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());                           // Event deal ticket
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,this.m_type_first);                         // Type of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,this.m_type_first);                      // Type of an order that triggered a position deal (the first position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());                          // Ticket of an order, based on which a deal event is opened (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());                       // Ticket of an order, based on which a position event is opened (the first position order)
         event.SetProperty(EVENT_PROP_POSITION_ID,this.m_position_id);                             // Position ID
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());                        // Opposite position ID
         event.SetProperty(EVENT_PROP_MAGIC_BY_ID,0);                                              // Opposite position magic number

         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order.TypeOrder());                      // Position order type before direction changed
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order.Ticket());                       // Position order ticket before direction changed
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order.TypeOrder());                     // Current position order type
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order.Ticket());                      // Current position order ticket

         event.SetProperty(EVENT_PROP_PRICE_OPEN_BEFORE,order.PriceOpen());                        // Order price before modification
         event.SetProperty(EVENT_PROP_PRICE_SL_BEFORE,order.StopLoss());                           // StopLoss before modification
         event.SetProperty(EVENT_PROP_PRICE_TP_BEFORE,order.TakeProfit());                         // TakeProfit before modification
         event.SetProperty(EVENT_PROP_PRICE_EVENT_ASK,this.m_tick.ask);                            // Ask price during an event
         event.SetProperty(EVENT_PROP_PRICE_EVENT_BID,this.m_tick.bid);                            // Bid price during an event

         event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                                  // Order/deal/position magic number
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpenMSC());                    // Time of an order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                              // Event price
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());                               // Order/deal/position open price
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());                             // Order/deal/position close price
         event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                                  // StopLoss position price
         event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());                                // TakeProfit position price
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order.Volume());                        // Requested order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,order.Volume()-order.VolumeCurrent()); // Executed order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,order.VolumeCurrent());                 // Remaining (unexecuted) order volume
         event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,order.Volume());                    // Executed position volume
         event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                                      // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                                      // Order symbol
         event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,order.Symbol());                                // Opposite position symbol
         //--- Set control program chart ID, decode the event code and set the event type
         event.SetChartID(this.m_chart_id);
         event.SetTypeEvent();
         //--- Add the event object if it is not in the list
         if(!this.IsPresentEventInList(event))
           {
            this.m_list_events.InsertSort(event);
            //--- Send a message about the event and set the value of the last trading event
            event.SendEvent();
            this.m_trade_event=event.TradeEvent();
           }
         //--- If the event is already present in the list, remove a new event object and display a debugging message
         else
           {
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event is already in the list."));
            delete event;
           }
        }
     }
//--- New deal (__MQL5__)
```

After making all the changes, the library should "see" position opening and activation of MQL4 pending orders.

### Testing

Let's check the applied changes. Compile the TestDoEasyPart10.mq4, launch it in the tester, open and close positions, place pending orders,
wait till one of them is activated and check if stop levels and trailing are activated (modifying positions and pending orders). All events
the library "sees" for MQL4 are to be displayed in the tester journal:

![](https://c.mql5.com/2/36/Test3.gif)

If we carefully observe the tester journal, we can see that the library still cannot see closing positions. When the BuyLimit #3 pending
order is triggered, the journal entry informs that the \[BuyLimit #3\] is activated leading to the Buy #3 position. Now the library sees the
events of pending order activation and knows a source order a position originated from. Besides, we can see a slight omission in the
modification function — the label of the BuyStop #1 pending order modified by trailing becomes red. But the library sees all order and
position modification events.

Add corrections to the tester's trading MQL4 functions in the **DELib.mqh** file. Let's create yet
another function that returns Buy/Sell position type depending on the pending
order type passed to it and replace checking an order type with checking
an order type by direction in the strings for selecting the arrow color:

```
//+------------------------------------------------------------------+
//| Modifying a pending order by ticket                              |
//+------------------------------------------------------------------+
bool PendingOrderModify(const ulong ticket,const double price_set,const double sl,const double tp)
  {
   ResetLastError();
   if(!OrderSelect((int)ticket,SELECT_BY_TICKET))
     {
      Print(DFUN,TextByLanguage("Не удалось выбрать ордер. Ошибка ","Could not select order. Error "),(string)GetLastError());
      return false;
     }
   ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderType();
   if(type<ORDER_TYPE_BUY_LIMIT || type>ORDER_TYPE_SELL_STOP)
     {
      Print(DFUN,TextByLanguage("Ошибка. Не ордер: ","Error. Not order: "),PositionTypeDescription((ENUM_POSITION_TYPE)type)," #",ticket);
      return false;
     }
   if(OrderCloseTime()>0)
     {
      Print(DFUN,TextByLanguage("Ошибка. Для модификации выбран удалённый ордер: ","Error. Deleted order selected for modification: "),OrderTypeDescription(type)," #",ticket);
      return false;
     }
   color clr=(TypeByPendingDirection(type)==ORDER_TYPE_BUY ? clrBlue : clrRed);
   ResetLastError();
   if(!OrderModify((int)ticket,price_set,sl,tp,0,clr))
     {
      Print(DFUN,TextByLanguage("Не удалось модифицировать ордер. Ошибка ","Failed to modify order. Error "),(string)GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Return the type by a pending order direction                     |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE TypeByPendingDirection(const ENUM_ORDER_TYPE type)
  {
   if(type==ORDER_TYPE_BUY_LIMIT  || type==ORDER_TYPE_BUY_STOP)  return ORDER_TYPE_BUY;
   if(type==ORDER_TYPE_SELL_LIMIT || type==ORDER_TYPE_SELL_STOP) return ORDER_TYPE_SELL;
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

### What's next?

In the next article, we will implement tracking position closing and fix errors that may arise in the current version of tracking events for
MQL4. Currently, placing and removing orders are tracked by MQL5 code, and there may be some nuances that should be taken into account when
working under MQL4.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6767#node00)

**Previous articles within the series:**

[Part 1. Concept, data management.](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals.](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search.](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept.](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program.](https://www.mql5.com/en/articles/6211)

[Part \\
6\. Netting account events.](https://www.mql5.com/en/articles/6383)

[Part 7. StopLimit order activation events, preparing \\
the functionality for order and position modification events.](https://www.mql5.com/en/articles/6482)

[Part 8. Order and \\
position modification events.](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 - Preparing data.](https://www.mql5.com/en/articles/6651)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6767](https://www.mql5.com/ru/articles/6767)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6767.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6767/mql5.zip "Download MQL5.zip")(99.2 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/6767/mql4.zip "Download MQL4.zip")(99.2 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/319583)**
(60)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
28 May 2019 at 15:29

**Aleksei Mikhanoshin:**

I'm writing in a different line.

Thank you, Artem, for such a wonderful and informative series of articles

You're welcome, Alexei. We've been on a first-name basis for a long time..... ![](https://c.mql5.com/3/280/neznaet.gif)

![Sam Zabil](https://c.mql5.com/avatar/avatar_na2.png)

**[Sam Zabil](https://www.mql5.com/en/users/willy78pro)**
\|
28 May 2019 at 16:20

Thank you all for the responses. I completely understand the purpose of this panel. I just liked the panel. By the way, I solved my question by sorting position sheets and orders by magic. Honestly, I was too lazy to get into the code right away.


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
28 May 2019 at 16:54

**Sam Zabil:**

Thank you all for the responses. I understand the purpose of this panel perfectly. I just liked the panel. By the way, I solved my question by sorting position sheets and orders by magic. Honestly, I was too lazy to get into the code right away.

Well, that's good.

Indeed, now you can already get all lists, and from them any object. And the lists can be sorted as needed and filtered by any of the values. To do this, CSelect() is used for the obtained list - there are examples in the [library class](https://www.mql5.com/en/articles/138 "Article: How to use trading classes of the Standard Library when writing an Expert Advisor") methods.

As a result, easy access, selection and filtering will be organised. So far only the creation of the necessary database is in progress.

![Sergey Seriy](https://c.mql5.com/avatar/2016/8/57B3CC35-B9E9.jpg)

**[Sergey Seriy](https://www.mql5.com/en/users/serggray)**
\|
25 Dec 2022 at 04:10

The idea is good. But I think it has already been implemented by fxsaber.


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
25 Dec 2022 at 10:30

**Sergey Seriy [#](https://www.mql5.com/ru/forum/314135/page6#comment_43973470):**

The idea is good. But I think it has already been implemented by fxsaber.

You can use the works of fxsaber

![Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://c.mql5.com/2/36/mql5_ea_adviser_grid__1.png)[Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://www.mql5.com/en/articles/7013)

In this article, we will make an attempt to develop the best possible grid-based EA. As usual, this will be a cross-platform EA capable of working both with MetaTrader 4 and MetaTrader 5. The first EA was good enough, except that it could not make a profit over a long period of time. The second EA could work at intervals of more than several years. Unfortunately, it was unable to yield more than 50% of profit per year with a maximum drawdown of less than 50%.

![Extract profit down to the last pip](https://c.mql5.com/2/36/MQL5-avatar-profit_digging__1.png)[Extract profit down to the last pip](https://www.mql5.com/en/articles/7113)

The article describes an attempt to combine theory with practice in the algorithmic trading field. Most of discussions concerning the creation of Trading Systems is connected with the use of historic bars and various indicators applied thereon. This is the most well covered field and thus we will not consider it. Bars represent a very artificial entity; therefore we will work with something closer to proto-data, namely the price ticks.

![Optimization management (Part I): Creating a GUI](https://c.mql5.com/2/36/mql5-avatar-opt_control.png)[Optimization management (Part I): Creating a GUI](https://www.mql5.com/en/articles/7029)

This article describes the process of creating an extension for the MetaTrader terminal. The solution discussed helps to automate the optimization process by running optimizations in other terminals. A few more articles will be written concerning this topic. The extension has been developed using the C# language and design patterns, which additionally demonstrates the ability to expand the terminal capabilities by developing custom modules, as well as the ability to create custom graphical user interfaces using the functionality of a preferred programming language.

![Arranging a mailing campaign by means of Google services](https://c.mql5.com/2/36/logo_Csharp.png)[Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)

A trader may want to arrange a mailing campaign to maintain business relationships with other traders, subscribers, clients or friends. Besides, there may be a necessity to send screenshotas, logs or reports. These may not be the most frequently arising tasks but having such a feature is clearly an advantage. The article deals with using several Google services simultaneously, developing an appropriate assembly on C# and integrating it with MQL tools.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/6767&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070482169161258659)

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