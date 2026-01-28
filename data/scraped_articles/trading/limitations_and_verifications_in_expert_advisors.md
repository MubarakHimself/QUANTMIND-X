---
title: Limitations and Verifications in Expert Advisors
url: https://www.mql5.com/en/articles/22
categories: Trading, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:23:27.293436
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/22&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069398553207440494)

MetaTrader 5 / Examples


### Introduction

When creating an algorithm for automated trading you should be able not only to process the prices for making trade signals, but to be able to get a lot of auxiliary information about limitations imposed on the operation of Expert Advisors. This article will tell you how to:


- Get information about trading sessions;

- Check if you have enough assets to open a position;

- Impose a limitation on the total trading volume by a symbol;

- Impose a limitation on the total number of orders;

- Calculate the potential loss between the entry price and Stop Loss;

- Check if there is a new bar.


### Trading and Quotation Sessions

To receive the information about trading sessions, you should use the [SymbolInfoSessionTrade](https://www.mql5.com/en/docs/marketinformation/symbolinfosessiontrade)() function, for quotation sessions use the corresponding [SymbolInfoSessionQuote](https://www.mql5.com/en/docs/marketinformation/symbolinfosessionquote)() function. Both functions work in the same way: if there is a session with the specified index for the specified day of week (the indexation of sessions starts from zero), then the function returns _true_. The time of start and end of a session is written to the fourth and fifth parameters passed by the link.


```
//--- check if there is a quotation session with the number session_index
bool session_exist=SymbolInfoSessionQuote(symbol,day,session_index,start,finish);
```

To find out all the session of the specified day, call this function in a loop until it returns _false_.

```
//+------------------------------------------------------------------+
//|  Display information about quotation sessions                    |
//+------------------------------------------------------------------+
void PrintInfoForQuoteSessions(string symbol,ENUM_DAY_OF_WEEK day)
  {
//--- start and end of session
   datetime start,finish;
   uint session_index=0;
   bool session_exist=true;

//--- go over all sessions of this day
   while(session_exist)
     {
      //--- check if there is a quotation session with the number session_index
      session_exist=SymbolInfoSessionQuote(symbol,day,session_index,start,finish);

      //--- if there is such session
      if(session_exist)
        {
         //--- display the day of week, the session number and the time of start and end
         Print(DayToString(day),": session index=",session_index,"  start=",
               TimeToString(start,TIME_MINUTES),"    finish=",TimeToString(finish-1,TIME_MINUTES|TIME_SECONDS));
        }
      //--- increase the counter of sessions
      session_index++;
     }
  }
```

The day of week is displayed in the string format using the custom function DayToString() that receives the value of the [ENUM\_DAY\_OF\_WEEK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_day_of_week) enumeration as the parameter.

```
//+------------------------------------------------------------------+
//| Receive the string representation of a day of week               |
//+------------------------------------------------------------------+
string DayToString(ENUM_DAY_OF_WEEK day)
  {
   switch(day)
     {
      case SUNDAY:    return "Sunday";
      case MONDAY:    return "Monday";
      case TUESDAY:   return "Tuesday";
      case WEDNESDAY: return "Wednesday";
      case THURSDAY:  return "Thursday";
      case FRIDAY:    return "Friday";
      case SATURDAY:  return "Saturday";
      default:        return "Unknown day of week";
     }
   return "";
  }
```

The final code of the SymbolInfoSession.mq5 script is attached to the bottom of the article. Let's show here its main part only.

```
void OnStart()
  {
//--- the array where the days of week are stored
   ENUM_DAY_OF_WEEK days[]={SUNDAY,MONDAY,TUESDAY,WEDNESDAY,THURSDAY,FRIDAY,SATURDAY};
   int size=ArraySize(days);

//---
   Print("Quotation sessions");
//--- go over all the days of week
   for(int d=0;d<size;d++)
     {
      PrintInfoForQuoteSessions(Symbol(),days[d]);
     }

//---
   Print("Trading sessions");
//--- go over all the days of week
   for(int d=0;d<size;d++)
     {
      PrintInfoForTradeSessions(Symbol(),days[d]);
     }
  }
```

### Checking Margin

To find out the amount of margin required for opening or increasing a position, you can use the [OrderCalcMargin()](https://www.mql5.com/en/docs/trading/ordercalcmargin) function; the first parameter that is passed to it is a value from the [ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) enumeration. For a buy operation you should call it with the ORDER\_TYPE\_BUY parameter; to sell, useORDER\_TYPE\_SELLparameter. The function returns the amount of margin depending on the number of lots and the open price.

```
void OnStart()
  {
//--- the variable to receive the value of margin
   double margin;
//--- to receive information about the last tick
   MqlTick last_tick;
//--- try to receive the value from the last tick
   if(SymbolInfoTick(Symbol(),last_tick))
     {
      //--- reset the last error code
      ResetLastError();
      //--- calculate margin value
      bool check=OrderCalcMargin(type,Symbol(),lots,last_tick.ask,margin);
      if(check)
        {
         PrintFormat("For the operation %s  %s %.2f lot at %G required margin is %.2f %s",OrderTypeToString(type),
                     Symbol(),lots,last_tick.ask,margin,AccountInfoString(ACCOUNT_CURRENCY));
        }
     }
   else
     {
      Print("Unsuccessful execution of the SymbolInfoTick() function, error ",GetLastError());
     }
  }
```

It should be noted that the OrderCalcMargin() function allows to calculate the value of margin not only for the market orders, but for pending orders as well. You can check the values that are returned for all the types of orders using the Check\_Money.mq5 script.

The OrderCalcMargin() function is intended for calculation of the size of margin for pending orders, since a money backing may be required in some trading systems for pending orders as well. Usually, the margin size for pending orders is calculated through a coefficient to the size of margin for the long and short positions.

|     |     |     |
| --- | --- | --- |
| Identifier | Description | Type of Property |
| SYMBOL\_MARGIN\_LONG | Rate of margin charging on long positions | double |
| SYMBOL\_MARGIN\_SHORT | Rate of margin charging on short positions | double |
| SYMBOL\_MARGIN\_LIMIT | Rate of margin charging on Limit orders | double |
| SYMBOL\_MARGIN\_STOP | Rate of margin charging on Stop order | double |
| SYMBOL\_MARGIN\_STOPLIMIT | Rate of margin charging on Stop Limit orders | double |

You can get the values of those coefficients using the simple code:


```
//--- Calculate the rates of margin charging for different types of orders
   PrintFormat("Rate of margin charging on long positions is equal to %G",SymbolInfoDouble(Symbol(),SYMBOL_MARGIN_LONG));
   PrintFormat("Rate of margin charging on short positions is equal to %G",SymbolInfoDouble(Symbol(),SYMBOL_MARGIN_SHORT));
   PrintFormat("Rate of margin charging on Limit orders is equal to %G",SymbolInfoDouble(Symbol(),SYMBOL_MARGIN_LIMIT));
   PrintFormat("Rate of margin charging on Stop orders is equal to %G",SymbolInfoDouble(Symbol(),SYMBOL_MARGIN_STOP));
   PrintFormat("Rate of margin charging on Stop Limit orders is equal to %G",SymbolInfoDouble(Symbol(),SYMBOL_MARGIN_STOPLIMIT));
```

For Forex symbols, the rates of margin charging for pending orders are usually equal to 0, i.e. there are no margin requirements for the them.

![The results of execution of the Check_Money.mq5 script.](https://c.mql5.com/2/0/1__1.png)

The results of execution of the Check\_Money.mq5 script.

Depending on the way of margin charging, the system of money management may change, as well as the trading system itself may experience some limitations if a margin is required for pending orders. That is why these parameters may also be natural limitations of an Expert Advisor's operation.

### Accounting Possible Profits and Losses

When placing a protecting stop level, you should be ready to its triggering. The risk of potential loss should be taken into account in terms of money; and the [OrderCalcProfit()](https://www.mql5.com/en/docs/trading/ordercalcprofit) is intended for this purpose. It is very similar to the already considered OrderCalcMargin() function, but it requires both open and close prices for calculations.

Specify one of two values of the[ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) enumeration as the first parameter -ORDER\_TYPE\_BUY or ORDER\_TYPE\_SELL; other types of orders will lead to an error. In the last parameter, you should pass a variable using the reference, to which theOrderCalcProfit() function will writethe value of profit/loss in case of successful execution.

The example of using the CalculateProfitOneLot() function that calculates the profit or loss, when closing a long position with specified levels of entering and exiting:

```
//+------------------------------------------------------------------+
//| Calculate potential profit/loss for buying 1 lot                 |
//+------------------------------------------------------------------+
double CalculateProfitOneLot(double entry_price,double exit_price)
  {
//--- receive the value of profit to this variable
   double profit=0;
   if(!OrderCalcProfit(ORDER_TYPE_BUY,Symbol(),1.0,entry_price,exit_price,profit))
     {
      Print(__FUNCTION__,"  Failed to calculate OrderCalcProfit(). Error ",GetLastError());
     }
//---
   return(profit);
  }
```

The result of calculation of this function is shown in the figure.

![](https://c.mql5.com/2/0/CalculateProfit_EA.png)

The example of calculation and displaying the potential loss on the chart using the OrderCalcProfit() function.

The whole code can be found in the attached Expert Advisor CalculateProfit\_EA.mq5.

### Checking If There Is a New Bar

Development of many trading system assumes that the trade signals are calculated only when a new bar appears; and all the trade actions are performed only once. The "Only open prices" mode of the strategy tester in the MetaTrader 5 client terminal is good for checking such automated trading systems.

In the "Open prices only" mode, all the calculations of indicators and the call of the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) function in Expert Advisor are performed only once per bar while testing. It is the fastest mode of trading; and, as a rule, the most fault-tolerant to insignificant price oscillations way of creating trading systems. At the same time, of course, indicators used in an Expert Advisor should be written correctly and shouldn't distort their values when a new bar comes.

The strategy tester in "Open prices only" mode allows to relieve you from caring about the Expert Advisor to be launched only once per bar; and it is very convenient. But while working in the real time mode on a demo or on a real account, a trader should control the activity of their Expert Advisor, to make it perform only one [trade operation](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions) per one received signal. The easiest way for this purpose is tracking of opening of the current unformed bar.

To get the time of opening of the last bar, you should use the [SeriesInfoInteger()](https://www.mql5.com/en/docs/series/seriesinfointeger)function with specified name of symbol, timeframe and the[SERIES\_LASTBAR\_DATE](https://www.mql5.com/en/docs/constants/tradingconstants/enum_series_info_integer) property. By constant comparing of the time of opening of the current bar with the one of the bar stored in a variable, you can easily detect the moment when a new bar appears. It allows to create the custom function isNewBar() that can look as following:

```
//+------------------------------------------------------------------+
//| Return true if a new bar appears for the symbol/period pair      |
//+------------------------------------------------------------------+
bool isNewBar()
  {
//--- remember the time of opening of the last bar in the static variable
   static datetime last_time=0;
//--- current time
   datetime lastbar_time=SeriesInfoInteger(Symbol(),Period(),SERIES_LASTBAR_DATE);

//--- if it is the first call of the function
   if(last_time==0)
     {
      //--- set time and exit
      last_time=lastbar_time;
      return(false);
     }

//--- if the time is different
   if(last_time!=lastbar_time)
     {
      //--- memorize time and return true
      last_time=lastbar_time;
      return(true);
     }
//--- if we pass to this line then the bar is not new, return false
   return(false);
  }
```

An example of using the function is given in the attached Expert Advisor CheckLastBar.mq5.


![The messages of the CheckLastBar Expert Advisor about appearing of new bars on M1 timeframe.](https://c.mql5.com/2/0/4__1.png)

The messages of the CheckLastBar Expert Advisor about appearing of new bars on M1 timeframe.

### Limiting Number of Pending Orders

If you need to limit the number of active pending orders thatcanbe simultaneously placed at an account, you can write your own custom function. Let's name itIsNewOrderAllowed(); it willcheck if it is allowed to place another pending order. Let's write it to comply with the [rules](https://championship.mql5.com/2010/en/rules) of the Automated Trading Championship.

```
//+------------------------------------------------------------------+
//| Checks if it is allowed to place another order                   |
//+------------------------------------------------------------------+
bool IsNewOrderAllowed()
  {
//--- get the allowed number of pending orders on an account
   int max_allowed_orders=(int)AccountInfoInteger(ACCOUNT_LIMIT_ORDERS);

//--- if there is no limitations, return true; you can send an order
   if(max_allowed_orders==0) return(true);

//--- if we pass to this line, then there are limitations; detect how many orders are already active
   int orders=OrdersTotal();

//--- return the result of comparing
   return(orders<max_allowed_orders);
  }
```

The function is simple: get the allowed number of orders to the _max\_allowed\_orders_ variable; and if its value is not equal to zero, compare with the [current number of orders](https://www.mql5.com/en/docs/trading/orderstotal). However, this function doesn't consider another possible limitation - the limitation on the allowed total volume of open positions and pending orders by a specific symbol.

### Limiting Number of Lots by a Specific Symbol

To get the size of open position by a specific symbol, first of all, you need to select a position using the [PositionSelect()](https://www.mql5.com/en/docs/trading/positionselect) function. And only after that, you can request the volume of the open position using the [PositionGetDouble()](https://www.mql5.com/en/docs/trading/positiongetdouble); it returns various [properties of the selected position](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties) that have the double type. Let's write the PostionVolume() function to get the position volume by a given symbol.


```
//+------------------------------------------------------------------+
//| Returns the size of position by a specific symbol                |
//+------------------------------------------------------------------+
double PositionVolume(string symbol)
  {
//--- try to select a positions by a symbol
   bool selected=PositionSelect(symbol);
//--- the position exists
   if(selected)
      //--- return the position volume
      return(PositionGetDouble(POSITION_VOLUME));
   else
     {
      //--- report about the unsuccessful attempt to select the position
      Print(__FUNCTION__," Failed to execute PositionSelect() for the symbol ",
            symbol," Error ",GetLastError());
      return(-1);
     }
  }
```

Before [making a trade request](https://www.mql5.com/en/docs/trading/ordersend) for placing a pending [order](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties) by a symbol, you should check the limitation on the total volume of open position and pending orders by that symbol - SYMBOL\_VOLUME\_LIMIT. If there is no limitation, then the volume of a pending order cannot exceed the maximum allowed volume that can be received using the [SymbolInfoDouble()](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) volume.

```
double max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_LIMIT);
if(max_volume==0) volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
```

However, this approach doesn't consider the volume of current pending orders by the specified symbol. Let's write a function that calculates this value:


```
//+------------------------------------------------------------------+
//| Returns the size of position by a specified symbol               |
//+------------------------------------------------------------------+
double PositionVolume(string symbol)
  {
//--- try to select a position by a symbol
   bool selected=PositionSelect(symbol);
//--- the position exist
   if(selected)
      //--- return the position volume
      return(PositionGetDouble(POSITION_VOLUME));
   else
     {
      //--- return zero if there is no position
      return(0);
     }
  }
```

With the consideration of the volume of open position and the volume of pending orders, the final checking will look as following:


```
//+------------------------------------------------------------------+
//|  Returns maximum allowed volume for an order by a symbol         |
//+------------------------------------------------------------------+
double NewOrderAllowedVolume(string symbol)
  {
   double allowed_volume=0;
//--- get the limitation on the maximum volume of an order
   double symbol_max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
//--- get the limitation of volume by a symbol
   double max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_LIMIT);

//--- get the volume of open position by a symbol
   double opened_volume=PositionVolume(symbol);
   if(opened_volume>=0)
     {
      //--- if we already used available volume
      if(max_volume-opened_volume<=0)
         return(0);

      //--- volume of the open position doen't exceed max_volume
      double orders_volume_on_symbol=PendingsVolume(symbol);
      allowed_volume=max_volume-opened_volume-orders_volume_on_symbol;
      if(allowed_volume>symbol_max_volume) allowed_volume=symbol_max_volume;
     }
   return(allowed_volume);
  }
```

The whole code of the Check\_Order\_And\_Volume\_Limits.mq5 Expert Advisor that contains the functions, mentioned in this section, is attached to the article.


![The example of checking using the Check_Order_And_Volume_Limits Expert Advisor on the account of a participant of the Automated Trading Championship 2010.](https://c.mql5.com/2/0/2.png)

The example of checking using the Check\_Order\_And\_Volume\_Limits Expert Advisor on the account of a participant of the Automated Trading Championship 2010.

### Checking the Correctness of Volume

A significant part of any trading robot is the ability to choose a correct volume for performing a trade operation. Here, we are not going to talk about the systems of money management and risk management, but about the volume to be correct according to the corresponding [properties of a symbol](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants).


|     |     |     |
| --- | --- | --- |
| Identifier | Description | Type of Property |
| SYMBOL\_VOLUME\_MIN | Minimal volume for a deal | double |
| SYMBOL\_VOLUME\_MAX | Maximal volume for a deal | double |
| SYMBOL\_VOLUME\_STEP | Minimal volume change step for deal execution | double |

To perform such verification, we can write the custom function CheckVolumeValue():

```
//+------------------------------------------------------------------+
//|  Check the correctness of volume of an order                     |
//+------------------------------------------------------------------+
bool CheckVolumeValue(double volume,string &description)
  {
//--- Minimum allowed volume for trade operations
   double min_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
   if(volume<min_volume)
     {
      description=StringFormat("Volume is less than the minimum allowed SYMBOL_VOLUME_MIN=%.2f",min_volume);
      return(false);
     }

//--- Maximum allowed volume for trade opertations
   double max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
   if(volume>max_volume)
     {
      description=StringFormat("Volume is greater than the maximum allowed SYMBOL_VOLUME_MAX=%.2f",max_volume);
      return(false);
     }

//--- get the minimal volume change step
   double volume_step=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP);

   int ratio=(int)MathRound(volume/volume_step);
   if(MathAbs(ratio*volume_step-volume)>0.0000001)
     {
      description=StringFormat("Volume is not a multiple of minimal step SYMBOL_VOLUME_STEP=%.2f, the closest correct volume is %.2f",
                               volume_step,ratio*volume_step);
      return(false);
     }
   description="Correct value of volume ";
   return(true);
  }
```

You can check the working of this function using the CheckVolumeValue.mq5 script attached to the article.


![The messages of the CheckVolumeValue.mq5 that checks the volume to be correct.](https://c.mql5.com/2/0/3.png)

The messages of the CheckVolumeValue.mq5 that checks the volume to be correct.

### Conclusion

The article describes the basic verifications for possible limitations on working of an Expert Advisor, that can be faced when creating your own automated trading system. These examples don't cover all the possible conditions that should be checked during the operation of an Expert Advisor on a trade account. But I hope, these examples will help newbies to understand how to implement the most popular verifications in the MQL5 language.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/22](https://www.mql5.com/ru/articles/22)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/22.zip "Download all attachments in the single ZIP archive")

[calculateprofit\_ea.mq5](https://www.mql5.com/en/articles/download/22/calculateprofit_ea.mq5 "Download calculateprofit_ea.mq5")(9.67 KB)

[check\_money.mq5](https://www.mql5.com/en/articles/download/22/check_money.mq5 "Download check_money.mq5")(3.38 KB)

[check\_order\_and\_volume\_limits.mq5](https://www.mql5.com/en/articles/download/22/check_order_and_volume_limits.mq5 "Download check_order_and_volume_limits.mq5")(4.66 KB)

[checklastbar.mq5](https://www.mql5.com/en/articles/download/22/checklastbar.mq5 "Download checklastbar.mq5")(1.78 KB)

[checkvolumevalue.mq5](https://www.mql5.com/en/articles/download/22/checkvolumevalue.mq5 "Download checkvolumevalue.mq5")(2.96 KB)

[symbolinfosession.mq5](https://www.mql5.com/en/articles/download/22/symbolinfosession.mq5 "Download symbolinfosession.mq5")(4.05 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1611)**
(20)


![Максим](https://c.mql5.com/avatar/avatar_na2.png)

**[Максим](https://www.mql5.com/en/users/maxx)**
\|
24 Oct 2010 at 17:54

The following code is used in the CheckVolumeValue  function:

int ratio=(int)MathRound(volume/volume\_step);

if(MathAbs(ratio\*volume\_step-volume)>0.0000001)

{

description=StringFormat("Объем не является кратным минимальной градации SYMBOL\_VOLUME\_STEP=%.2f, ближайший корректный объем %.2f",

volume\_step,ratio\*volume\_step);

return(false);

}

But it is more correct:

int ratio = (int)MathRound((volume-min\_volume)/volume\_step);

if (MathAbs(ratio\*volume\_step+min\_volume-volume)>0.0000001)

{

description=StringFormat("Объем не является кратным минимальной градации SYMBOL\_VOLUME\_STEP=%.2f, ближайший корректный объем %.2f",

volume\_step,ratio\*volume\_step+min\_volume);

return(false);

}

Because the [minimum step of volume change](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double "MQL5 documentation: Information about the tool") must be counted from the minimum value.

![Максим](https://c.mql5.com/avatar/avatar_na2.png)

**[Максим](https://www.mql5.com/en/users/maxx)**
\|
24 Oct 2010 at 19:43

And perhaps the code snippet

//\-\-\- вычислим значение маржи

bool check=OrderCalcMargin(type,Symbol(),lots,last\_tick.ask,margin);

should be replaced with:

//\-\-\- вычислим значение маржи

double price = (type == ORDER\_TYPE\_BUY \|\| type == ORDER\_TYPE\_BUY\_LIMIT \|\| type == ORDER\_TYPE\_BUY\_STOP \|\| type == ORDER\_TYPE\_BUY\_STOP\_LIMIT) ? last\_tick.ask : last\_tick.bid;

bool check=OrderCalcMargin(type,Symbol(),lots,price,margin);

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
23 Dec 2013 at 12:28

[Attached source](https://www.mql5.com/en/articles/24#insert-code "MQL5.community - User Memo: Insert Code") code files and source code insets in HTML code are now completely translated into Portuguese for your convenience.


![Hamed Dehgani](https://c.mql5.com/avatar/2025/10/68f619d9-6752.jpg)

**[Hamed Dehgani](https://www.mql5.com/en/users/eeecad)**
\|
20 Jan 2020 at 14:53

**Rashid Umarov:**

Due to changes in MQL5, now the maximal overall volume allowed for one symbol can be obtained as following:

**Do not use the old variant!** It was like this:

The article has been corrected and the new _Check\_Order\_And\_Volume\_Limits.mq5_ expert code has been attached to it.

Dear Admin

I try to use this [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") but it will return 0 in all cases.

My MT5 build is 2280

![Poker_player](https://c.mql5.com/avatar/2020/12/5FCD05A8-E7B6.jpg)

**[Poker\_player](https://www.mql5.com/en/users/speed_fanat1c)**
\|
27 Dec 2020 at 16:37

```
double orders_volume_on_symbol=PendingsVolume(symbol);
```

gives error, no such [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions")

And after I modify function to not include pending orders, it  gives 0

```
double NewOrderAllowedVolume(string symbol)
  {
   double allowed_volume=0;
//--- get the limitation on the maximum volume of an order
   double symbol_max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
//--- get the limitation of volume by a symbol
   double max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_LIMIT);

//--- get the volume of open position by a symbol
   double opened_volume=PositionVolume(symbol);
   if(opened_volume>=0)
     {
      //--- if we already used available volume
      if(max_volume-opened_volume<=0)
         return(0);

      //--- volume of the open position doen't exceed max_volume
      //double orders_volume_on_symbol=PendingsVolume(symbol);
      //allowed_volume=max_volume-opened_volume-orders_volume_on_symbol;
      allowed_volume=max_volume-opened_volume;
      if(allowed_volume>symbol_max_volume) allowed_volume=symbol_max_volume;
     }
   return(allowed_volume);
  }
```

It is because max\_volume is 0 and opened\_volume is 0;

Why max volume is 0 if there is no opened positions?

Build 2715

![MetaTrader 5 and MATLAB Interaction](https://c.mql5.com/2/0/matlab.png)[MetaTrader 5 and MATLAB Interaction](https://www.mql5.com/en/articles/44)

This article covers the details of interaction between MetaTrader 5 and MatLab mathematical package. It shows the mechanism of data conversion, the process of developing a universal library to interact with MatLab desktop. It also covers the use of DLL generated by MatLab environment. This article is intended for experienced readers, who know C++ and MQL5.

![Creating Multi-Colored Indicators in MQL5](https://c.mql5.com/2/0/paint_indicator_MQL5__1.png)[Creating Multi-Colored Indicators in MQL5](https://www.mql5.com/en/articles/135)

In this article, we will consider how to create multi-colored indicators or convert the existing ones to multi-color. MQL5 allows to represent the information in the convenient form. Now it isn't necessary to look at a dozen of charts with indicators and perform analyses of the RSI or Stochastic levels, it's better just to paint the candles with different colors depending on the values of the indicators.

![How to Create Your Own Trailing Stop](https://c.mql5.com/2/0/Trailing_Stop_MQL5.png)[How to Create Your Own Trailing Stop](https://www.mql5.com/en/articles/134)

The basic rule of trader - let profit to grow, cut off losses! This article considers one of the basic techniques, allowing to follow this rule - moving the protective stop level (Stop loss level) after increasing position profit, i.e. - Trailing Stop level. You'll find the step by step procedure to create a class for trailing stop on SAR and NRTR indicators. Everyone will be able to insert this trailing stop into their experts or use it independently to control positions in their accounts.

![Creating and Publishing of Trade Reports and SMS Notification](https://c.mql5.com/2/0/trade_reports_SMS_MQL5.png)[Creating and Publishing of Trade Reports and SMS Notification](https://www.mql5.com/en/articles/61)

Traders don't always have ability and desire to seat at the trading terminal for hours. Especially, if trading system is more or less formalized and can automatically identify some of the market states. This article describes how to generate a report of trade results (using Expert Advisor, Indicator or Script) as HTML-file and upload it via FTP to WWW-server. We will also consider sending notification of trade events as SMS to mobile phone.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/22&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069398553207440494)

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