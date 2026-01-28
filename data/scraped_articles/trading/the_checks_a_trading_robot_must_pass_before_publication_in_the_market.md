---
title: The checks a trading robot must pass before publication in the Market
url: https://www.mql5.com/en/articles/2555
categories: Trading, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:18:35.633628
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/2555&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069330804393313097)

MetaTrader 5 / Examples


### Why products are checked before they are published in the Market

Before any product is published in the Market, it must undergo compulsory preliminary checks, as a small error in the expert or indicator logic can cause losses on the trading account. That is why we have developed a series of basic checks to ensure the required quality level of the Market products.

- [How to quickly catch and fix errors in trading robots](https://www.mql5.com/en/articles/2555#how_to_check)
- [Insufficient funds to perform trade operation](https://www.mql5.com/en/articles/2555#not_enough_money)
- [Invalid volumes in trade operations](https://www.mql5.com/en/articles/2555#invalid_lot)
- [Limiting Number of Pending Orders](https://www.mql5.com/en/articles/2555#account_limit_pending_orders)
- [Limiting Number of Lots by a Specific Symbol](https://www.mql5.com/en/articles/2555#symbol_limit_lots)
- [Setting the TakeProfit and StopLoss levels within the SYMBOL\_TRADE\_STOPS\_LEVEL minimum level](https://www.mql5.com/en/articles/2555#invalid_SL_TP_for_position)
- [Attempt to modify order or position within the SYMBOL\_TRADE\_FREEZE\_LEVEL freeze level](https://www.mql5.com/en/articles/2555#modify_in_freeze_level_prohibited)
- [Errors that occur when working with symbols which have insufficient quote history](https://www.mql5.com/en/articles/2555#not_enough_quotes_history)
- [Array out of Range](https://www.mql5.com/en/articles/2555#out_of_range)
- [Zero Divide](https://www.mql5.com/en/articles/2555#zero_divide)
- [Sending a request to modify the levels without actually changing them](https://www.mql5.com/en/articles/2555#no_changes_in_modification_request)
- [Attempt to import compiled files (even EX4/EX5) and DLL](https://www.mql5.com/en/articles/2555#dll_and_libraries_prohibited)
- [Calling custom indicators with iCustom()](https://www.mql5.com/en/articles/2555#attempt_to_use_icustom)
- [Passing an invalid parameter to the function (runtime error)](https://www.mql5.com/en/articles/2555#invalid_parameter_runtime)
- [Access violation](https://www.mql5.com/en/articles/2555#access_violation)
- [Consumption of the CPU resources and memory](https://www.mql5.com/en/articles/2555#excessive_load_memory_cpu)
- [Articles for reading](https://www.mql5.com/en/articles/2555#articles_for_reading)

If any errors are identified by the Market moderators in the process of checking your product, you will have to fix all of them. This article considers the most frequent errors made by developers in their trading robots and technical indicators. We also recommend reading the following articles:

- [How to Write a Good Description for a Market Product](https://www.mql5.com/en/articles/557)
- [Tips for an Effective Product Presentation on the Market.](https://www.mql5.com/en/articles/999)

### How to quickly catch and fix errors in trading robots

The [strategy tester](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") integrated into the platform allows not only to backtest trading systems, but also to identify logical and algorithmic errors made at the development stage of the trading robot. During testing, all messages about trade operations and identified errors are output to the tester [Journal](https://www.metatrader5.com/en/terminal/help/algotrading/visualization#toolbox_journal "https://www.metatrader5.com/en/terminal/help/algotrading/visualization#toolbox_journal"). It is convenient to analyze those messages in a special log [Viewer](https://www.metatrader5.com/en/terminal/help/start_advanced/journal#viewer "https://www.metatrader5.com/en/terminal/help/start_advanced/journal#viewer"), which can be called using a context menu command.

![](https://c.mql5.com/2/23/tester_lof_viewer.png)

After testing the EA, open the viewer and enable the "Errors only" mode, as shown in the figure. If your trading robot contains errors, you will see them immediately. If no errors were detected the first time, perform a series of tests with different instruments/timeframes/input parameters and different values of the initial deposit. 99% of the errors can be identified by these simple techniques, and they will be discussed in this article.

For a detailed study of detected errors use the [Debugging on History Data](https://www.metatrader5.com/en/metaeditor/help/development/debug#history "https://www.metatrader5.com/en/metaeditor/help/development/debug#history") in the MetaEditor. This method allows to use the [visual testing](https://www.metatrader5.com/en/terminal/help/algotrading/visualization "https://www.metatrader5.com/en/terminal/help/algotrading/visualization") mode for not only monitoring the price charts and values of indicators in use, but also track the value of each variable of the program at each tick. Thus, you will be able to debug your trading strategy without having to spend weeks in real time.

### Insufficient funds to perform trade operation

Before sending every trade order, it is necessary to check if there are enough funds on the account. Lack of funds to support the future open position or order is considered a blunder.

**Keep in mind** that even placing a [pending order](https://www.metatrader5.com/en/terminal/help/trading/general_concept#pending_order "https://www.metatrader5.com/en/terminal/help/trading/general_concept#pending_order") may require collateral — [margin](https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex "https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex").

![](https://c.mql5.com/2/23/errors.png)

It is recommended to [test](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") a trading robot with a deliberately small size of the initial deposit, for example, 1 USD or 1 Euro.

If a check shows that there are insufficient funds to perform a trade operation, it is necessary to output an error message to the log instead of calling the OrderSend() function. Sample check:

**MQL5**

```
bool CheckMoneyForTrade(string symb,double lots,ENUM_ORDER_TYPE type)
  {
//--- Getting the opening price
   MqlTick mqltick;
   SymbolInfoTick(symb,mqltick);
   double price=mqltick.ask;
   if(type==ORDER_TYPE_SELL)
      price=mqltick.bid;
//--- values of the required and free margin
   double margin,free_margin=AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   //--- call of the checking function
   if(!OrderCalcMargin(type,symb,lots,price,margin))
     {
      //--- something went wrong, report and return false
      Print("Error in ",__FUNCTION__," code=",GetLastError());
      return(false);
     }
   //--- if there are insufficient funds to perform the operation
   if(margin>free_margin)
     {
      //--- report the error and return false
      Print("Not enough money for ",EnumToString(type)," ",lots," ",symb," Error code=",GetLastError());
      return(false);
     }
//--- checking successful
   return(true);
  }
```

**MQL4**

```
bool CheckMoneyForTrade(string symb, double lots,int type)
  {
   double free_margin=AccountFreeMarginCheck(symb,type, lots);
   //-- if there is not enough money
   if(free_margin<0)
     {
      string oper=(type==OP_BUY)? "Buy":"Sell";
      Print("Not enough money for ", oper," ",lots, " ", symb, " Error code=",GetLastError());
      return(false);
     }
   //--- checking successful
   return(true);
  }
```

### Invalid volumes in trade operations

Before sending trade orders, it is also necessary to check the correctness of the volumes specified in the orders. The number of lots that the EA is about to set in the order, must be checked before calling the OrderSend() function. The minimum and maximum allowed trading volumes for the symbols are specified in the [Specification](https://www.metatrader5.com/en/terminal/help/trading/market_watch#specification "https://www.metatrader5.com/en/terminal/help/trading/market_watch#specification"), as well as the volume step. In MQL5, these values can be obtained from the [ENUM\_SYMBOL\_INFO\_DOUBLE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double)enumeration with the help of the [SymbolInfoDouble()](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function.

|     |     |
| --- | --- |
| SYMBOL\_VOLUME\_MIN | Minimal volume for a deal |
| SYMBOL\_VOLUME\_MAX | Maximal volume for a deal |
| SYMBOL\_VOLUME\_STEP | Minimal volume change step for deal execution |

Example of function for checking the correctness of the volume

```
//+------------------------------------------------------------------+
//| Check the correctness of the order volume                        |
//+------------------------------------------------------------------+
bool CheckVolumeValue(double volume,string &description)
  {
//--- minimal allowed volume for trade operations
   double min_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
   if(volume<min_volume)
     {
      description=StringFormat("Volume is less than the minimal allowed SYMBOL_VOLUME_MIN=%.2f",min_volume);
      return(false);
     }

//--- maximal allowed volume of trade operations
   double max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
   if(volume>max_volume)
     {
      description=StringFormat("Volume is greater than the maximal allowed SYMBOL_VOLUME_MAX=%.2f",max_volume);
      return(false);
     }

//--- get minimal step of volume changing
   double volume_step=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP);

   int ratio=(int)MathRound(volume/volume_step);
   if(MathAbs(ratio*volume_step-volume)>0.0000001)
     {
      description=StringFormat("Volume is not a multiple of the minimal step SYMBOL_VOLUME_STEP=%.2f, the closest correct volume is %.2f",
                               volume_step,ratio*volume_step);
      return(false);
     }
   description="Correct volume value";
   return(true);
  }
```

### Limiting Number of Pending Orders

There can also be a limit on the number of active pending orders that can be simultaneously placed at an account. Example of the IsNewOrderAllowed() function, which checks if another pending order can be placed.

```
//+------------------------------------------------------------------+
//| Check if another order can be placed                             |
//+------------------------------------------------------------------+
bool IsNewOrderAllowed()
  {
//--- get the number of pending orders allowed on the account
   int max_allowed_orders=(int)AccountInfoInteger(ACCOUNT_LIMIT_ORDERS);

//--- if there is no limitation, return true; you can send an order
   if(max_allowed_orders==0) return(true);

//--- if we passed to this line, then there is a limitation; find out how many orders are already placed
   int orders=OrdersTotal();

//--- return the result of comparing
   return(orders<max_allowed_orders);
  }
```

The function is simple: get the allowed number of orders to the _max\_allowed\_orders_ variable; and if its value is not equal to zero, compare with the [current number of orders](https://www.mql5.com/en/docs/trading/orderstotal). However, this function does not consider another possible restriction - the limitation on the allowed total volume of open positions and pending orders on a specific symbol.

### Limiting Number of Lots by a Specific Symbol

To get the size of open position by a specific symbol, first of all you need to select a position using the [PositionSelect()](https://www.mql5.com/en/docs/trading/positionselect) function. And only after that you can request the volume of the open position using the [PositionGetDouble()](https://www.mql5.com/en/docs/trading/positiongetdouble), it returns various [properties of the selected position](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties) that have the double type. Let's write the PostionVolume() function to get the position volume by a given symbol.

```
//+------------------------------------------------------------------+
//| Return the size of position on the specified symbol              |
//+------------------------------------------------------------------+
double PositionVolume(string symbol)
  {
//--- try to select position by a symbol
   bool selected=PositionSelect(symbol);
//--- there is a position
   if(selected)
      //--- return volume of the position
      return(PositionGetDouble(POSITION_VOLUME));
   else
     {
      //--- report a failure to select position
      Print(__FUNCTION__," Failed to perform PositionSelect() for symbol ",
            symbol," Error ",GetLastError());
      return(-1);
     }
  }
```

For accounts that support hedging, it is necessary to iterate over all positions on the current instrument.

Before [making a trade request](https://www.mql5.com/en/docs/trading/ordersend) for placing a pending [order](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties) by a symbol, you should check the limitation on the total volume of open position and pending orders on one symbol - SYMBOL\_VOLUME\_LIMIT. If there is no limitation, then the volume of a pending order cannot exceed the maximum allowed volume that can be received using the [SymbolInfoDouble()](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) volume.

```
double max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_LIMIT);
if(max_volume==0) volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
```

However, this approach doesn't consider the volume of current pending orders by the specified symbol. Here is an example of a function that calculates this value:

```
//+------------------------------------------------------------------+
//|  returns the volume of current pending order by a symbol         |
//+------------------------------------------------------------------+
double   PendingsVolume(string symbol)
  {
   double volume_on_symbol=0;
   ulong ticket;
//---  get the number of all currently placed orders by all symbols
   int all_orders=OrdersTotal();

//--- get over all orders in the loop
   for(int i=0;i<all_orders;i++)
     {
      //--- get the ticket of an order by its position in the list
      if(ticket=OrderGetTicket(i))
        {
         //--- if our symbol is specified in the order, add the volume of this order
         if(symbol==OrderGetString(ORDER_SYMBOL))
            volume_on_symbol+=OrderGetDouble(ORDER_VOLUME_INITIAL);
        }
     }
//--- return the total volume of currently placed pending orders for a specified symbol
   return(volume_on_symbol);
  }
```

With the consideration of the open position and volume in pending orders, the final check will look the following way:

```
//+------------------------------------------------------------------+
//| Return the maximum allowed volume for an order on the symbol     |
//+------------------------------------------------------------------+
double NewOrderAllowedVolume(string symbol)
  {
   double allowed_volume=0;
//--- get the limitation on the maximal volume of an order
   double symbol_max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
//--- get the limitation on the volume by a symbol
   double max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_LIMIT);

//--- get the volume of the open position by a symbol
   double opened_volume=PositionVolume(symbol);
   if(opened_volume>=0)
     {
      //--- if we have exhausted the volume
      if(max_volume-opened_volume<=0)
         return(0);

      //--- volume of the open position doesn't exceed max_volume
      double orders_volume_on_symbol=PendingsVolume(symbol);
      allowed_volume=max_volume-opened_volume-orders_volume_on_symbol;
      if(allowed_volume>symbol_max_volume) allowed_volume=symbol_max_volume;
     }
   return(allowed_volume);
  }
```

### Setting the TakeProfit and StopLoss levels within the SYMBOL\_TRADE\_STOPS\_LEVEL minimum level

Many experts trade using the TakeProfit and StopLoss orders with levels calculated dynamically at the moment of performing a buy or a sell. The TakeProfit order serves to close the position when the price moves in a favorable direction, while the StopLoss is used for limiting losses when the price moves in an unfavorable direction.

Therefore, the TakeProfit and StopLoss levels should be compared to the current price for performing the opposite operation:

- Buying is done at the Ask price — the TakeProfit and StopLoss levels should be compared to the Bid price.
- Selling is done at the Bid price — the TakeProfit and StopLoss levels should be compared to the Ask price.

| Buying is done at the Ask price | Selling is done at the Bid price |
| --- | --- |
| TakeProfit >= Bid<br> StopLoss <= Bid | TakeProfit <= Ask<br> StopLoss >= Ask |

![](https://c.mql5.com/2/23/Check_TP_and_SL_en.png)

Financial instrument may have the [SYMBOL\_TRADE\_STOPS\_LEVEL](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer) parameter set in the symbol settings. It determines the number of points for minimum indentation of the StopLoss and TakeProfit levels from the current closing price of the open position. If the value of this property is zero, then the minimum indentation for SL/TP orders has not been set for buying and selling.

In general, checking the TakeProfit and StopLoss levels with the minimum distance of [SYMBOL\_TRADE\_STOPS\_LEVEL](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer) taken into account looks as follows:

- **Buying** is done at the Ask price — the TakeProfit and StopLoss levels must be **at the distance of at least SYMBOL\_TRADE\_STOPS\_LEVEL points from the Bid price.**
- **Selling** is done at the Bid price — the TakeProfit and StopLoss levels must be **at the distance of at least SYMBOL\_TRADE\_STOPS\_LEVEL points from the Ask price.**

| Buying is done at the Ask price | Selling is done at the Bid price |
| --- | --- |
| TakeProfit - Bid >= SYMBOL\_TRADE\_STOPS\_LEVEL<br> Bid - StopLoss >= SYMBOL\_TRADE\_STOPS\_LEVEL | Ask - TakeProfit >= SYMBOL\_TRADE\_STOPS\_LEVEL<br> StopLoss - Ask >= SYMBOL\_TRADE\_STOPS\_LEVEL |

So, we can create a CheckStopLoss\_Takeprofit() check function, which requires the distance from the TakeProfit and StopLoss to the closing price to be at least SYMBOL\_TRADE\_STOPS\_LEVEL points:

```
bool CheckStopLoss_Takeprofit(ENUM_ORDER_TYPE type,double SL,double TP)
  {
//--- get the SYMBOL_TRADE_STOPS_LEVEL level
   int stops_level=(int)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   if(stops_level!=0)
     {
      PrintFormat("SYMBOL_TRADE_STOPS_LEVEL=%d: StopLoss and TakeProfit must"+
                  " not be nearer than %d points from the closing price",stops_level,stops_level);
     }
//---
   bool SL_check=false,TP_check=false;
//--- check only two order types
   switch(type)
     {
      //--- Buy operation
      case  ORDER_TYPE_BUY:
        {
         //--- check the StopLoss
         SL_check=(Bid-SL>stops_level*_Point);
         if(!SL_check)
            PrintFormat("For order %s StopLoss=%.5f must be less than %.5f"+
                        " (Bid=%.5f - SYMBOL_TRADE_STOPS_LEVEL=%d points)",
                        EnumToString(type),SL,Bid-stops_level*_Point,Bid,stops_level);
         //--- check the TakeProfit
         TP_check=(TP-Bid>stops_level*_Point);
         if(!TP_check)
            PrintFormat("For order %s TakeProfit=%.5f must be greater than %.5f"+
                        " (Bid=%.5f + SYMBOL_TRADE_STOPS_LEVEL=%d points)",
                        EnumToString(type),TP,Bid+stops_level*_Point,Bid,stops_level);
         //--- return the result of checking
         return(SL_check&&TP_check);
        }
      //--- Sell operation
      case  ORDER_TYPE_SELL:
        {
         //--- check the StopLoss
         SL_check=(SL-Ask>stops_level*_Point);
         if(!SL_check)
            PrintFormat("For order %s StopLoss=%.5f must be greater than %.5f "+
                        " (Ask=%.5f + SYMBOL_TRADE_STOPS_LEVEL=%d points)",
                        EnumToString(type),SL,Ask+stops_level*_Point,Ask,stops_level);
         //--- check the TakeProfit
         TP_check=(Ask-TP>stops_level*_Point);
         if(!TP_check)
            PrintFormat("For order %s TakeProfit=%.5f must be less than %.5f "+
                        " (Ask=%.5f - SYMBOL_TRADE_STOPS_LEVEL=%d points)",
                        EnumToString(type),TP,Ask-stops_level*_Point,Ask,stops_level);
         //--- return the result of checking
         return(TP_check&&SL_check);
        }
      break;
     }
//--- a slightly different function is required for pending orders
   return false;
  }
```

The check itself can look like this:

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- randomly get the type of operation
   int oper=(int)(GetTickCount()%2); // remainder of division by two is always either 0 or 1
   switch(oper)
     {
      //--- buy
      case  0:
        {
         //--- get the opening price and knowingly set invalid TP/SL
         double price=Ask;
         double SL=NormalizeDouble(Bid+2*_Point,_Digits);
         double TP=NormalizeDouble(Bid-2*_Point,_Digits);
         //--- perform a check
         PrintFormat("Buy at %.5f   SL=%.5f   TP=%.5f  Bid=%.5f",price,SL,TP,Bid);
         if(!CheckStopLoss_Takeprofit(ORDER_TYPE_BUY,SL,TP))
            Print("The StopLoss or TakeProfit level is incorrect!");
         //--- try to buy anyway, in order to see the execution result
         Buy(price,SL,TP);
        }
      break;
      //--- sell
      case  1:
        {
         //--- get the opening price and knowingly set invalid TP/SL
         double price=Bid;
         double SL=NormalizeDouble(Ask-2*_Point,_Digits);
         double TP=NormalizeDouble(Ask+2*_Point,_Digits);
         //--- perform a check
         PrintFormat("Sell at %.5f   SL=%.5f   TP=%.5f  Ask=%.5f",price,SL,TP,Ask);
         if(!CheckStopLoss_Takeprofit(ORDER_TYPE_SELL,SL,TP))
            Print("The StopLoss or TakeProfit level is incorrect!");
         //--- try to sell anyway, in order to see the execution result
         Sell(price,SL,TP);
        }
      break;
      //---
     }
  }
```

The example of the function can be found in the attached scripts _Check\_TP\_and\_SL.mq4_ and _Check\_TP\_and\_SL.mq5_. Example of execution:

```
MQL5
Check_TP_and_SL (EURUSD,H1) Buy at 1.11433   SL=1.11425   TP=1.11421  Bid=1.11423
Check_TP_and_SL (EURUSD,H1) SYMBOL_TRADE_STOPS_LEVEL=30: StopLoss and TakeProfit must not nearer than 30 points from the closing price
Check_TP_and_SL (EURUSD,H1) For order ORDER_TYPE_BUY StopLoss=1.11425 must be less than 1.11393 (Bid=1.11423 - SYMBOL_TRADE_STOPS_LEVEL=30 points)
Check_TP_and_SL (EURUSD,H1) For order ORDER_TYPE_BUY TakeProfit=1.11421 must be greater than 1.11453 (Bid=1.11423 + SYMBOL_TRADE_STOPS_LEVEL=30 points)
Check_TP_and_SL (EURUSD,H1) The StopLoss or TakeProfit level is incorrect!
Check_TP_and_SL (EURUSD,H1) OrderSend error 4756
Check_TP_and_SL (EURUSD,H1) retcode=10016  deal=0  order=0
MQL4
Check_TP_and_SL EURUSD,H1:  Sell at 1.11430   SL=1.11445   TP=1.11449  Ask=1.11447
Check_TP_and_SL EURUSD,H1:  SYMBOL_TRADE_STOPS_LEVEL=1: StopLoss and TakeProfit must not nearer than 1 points from the closing price
Check_TP_and_SL EURUSD,H1:  For order ORDER_TYPE_SELL StopLoss=1.11445 must be greater than 1.11448  (Ask=1.11447 + SYMBOL_TRADE_STOPS_LEVEL=1 points)
Check_TP_and_SL EURUSD,H1:  For order ORDER_TYPE_SELL TakeProfit=1.11449 must be less than 1.11446  (Ask=1.11447 - SYMBOL_TRADE_STOPS_LEVEL=1 points)
Check_TP_and_SL EURUSD,H1:  The StopLoss or TakeProfit level is incorrect!
Check_TP_and_SL EURUSD,H1:  OrderSend error 130
```

To simulate the situation with the invalid TakeProfit and StopLoss values, the _Test\_Wrong\_TakeProfit\_LEVEL.mq5_ and _Test\_Wrong\_StopLoss\_LEVEL.mq5_ experts can be found in the article attachments. They can be run only on a demo account. Study these examples in order to see the conditions when a successful buy operation is possible.

Example of the _Test\_Wrong\_StopLoss\_LEVEL.mq5_ EA execution:

```
Test_Wrong_StopLoss_LEVEL.mq5
Point=0.00001 Digits=5
SYMBOL_TRADE_EXECUTION=SYMBOL_TRADE_EXECUTION_INSTANT
SYMBOL_TRADE_FREEZE_LEVEL=20: order or position modification is not allowed, if there are 20 points to the activation price
SYMBOL_TRADE_STOPS_LEVEL=30: StopLoss and TakeProfit must not nearer than 30 points from the closing price
1. Buy 1.0 EURUSD at 1.11442 SL=1.11404 Bid=1.11430 ( StopLoss-Bid=-26 points ))
CTrade::OrderSend: instant buy 1.00 EURUSD at 1.11442 sl: 1.11404 [invalid stops]
2. Buy 1.0 EURUSD at 1.11442 SL=1.11404 Bid=1.11431 ( StopLoss-Bid=-27 points ))
CTrade::OrderSend: instant buy 1.00 EURUSD at 1.11442 sl: 1.11404 [invalid stops]
3. Buy 1.0 EURUSD at 1.11442 SL=1.11402 Bid=1.11430 ( StopLoss-Bid=-28 points ))
CTrade::OrderSend: instant buy 1.00 EURUSD at 1.11442 sl: 1.11402 [invalid stops]
4. Buy 1.0 EURUSD at 1.11440 SL=1.11399 Bid=1.11428 ( StopLoss-Bid=-29 points ))
CTrade::OrderSend: instant buy 1.00 EURUSD at 1.11440 sl: 1.11399 [invalid stops]
5. Buy 1.0 EURUSD at 1.11439 SL=1.11398 Bid=1.11428 ( StopLoss-Bid=-30 points ))
Buy 1.0 EURUSD done at 1.11439 with StopLoss=41 points (spread=12 + SYMBOL_TRADE_STOPS_LEVEL=30)
```

Example of the _Test\_Wrong\_TakeProfit\_LEVEL.mq5_ EA execution:

```
Test_Wrong_TakeProfit_LEVEL.mq5
Point=0.00001 Digits=5
SYMBOL_TRADE_EXECUTION=SYMBOL_TRADE_EXECUTION_INSTANT
SYMBOL_TRADE_FREEZE_LEVEL=20: order or position modification is not allowed, if there are 20 points to the activation price
SYMBOL_TRADE_STOPS_LEVEL=30: StopLoss and TakeProfit must not nearer than 30 points from the closing price
1. Buy 1.0 EURUSD at 1.11461 TP=1.11478 Bid=1.11452 (TakeProfit-Bid=26 points)
CTrade::OrderSend: instant buy 1.00 EURUSD at 1.11461 tp: 1.11478 [invalid stops]
2. Buy 1.0 EURUSD at 1.11461 TP=1.11479 Bid=1.11452 (TakeProfit-Bid=27 points)
CTrade::OrderSend: instant buy 1.00 EURUSD at 1.11461 tp: 1.11479 [invalid stops]
3. Buy 1.0 EURUSD at 1.11461 TP=1.11480 Bid=1.11452 (TakeProfit-Bid=28 points)
CTrade::OrderSend: instant buy 1.00 EURUSD at 1.11461 tp: 1.11480 [invalid stops]
4. Buy 1.0 EURUSD at 1.11461 TP=1.11481 Bid=1.11452 (TakeProfit-Bid=29 points)
CTrade::OrderSend: instant buy 1.00 EURUSD at 1.11461 tp: 1.11481 [invalid stops]
5. Buy 1.0 EURUSD at 1.11462 TP=1.11482 Bid=1.11452 (TakeProfit-Bid=30 points)
Buy 1.0 EURUSD done at 1.11462 with TakeProfit=20 points (SYMBOL_TRADE_STOPS_LEVEL=30 - spread=10)
```

Checking the StopLoss and TakeProfit levels in pending orders is much simpler, as these levels must be set based on the order opening price. That is, the checking of levels taking into account the [SYMBOL\_TRADE\_STOPS\_LEVEL](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer) minimum distance looks as follows: the TakeProfit and StopLoss levels must be **at the distance of at least SYMBOL\_TRADE\_STOPS\_LEVEL points from the order activation price.**

| BuyLimit and BuyStop | SellLimit and SellStop |
| --- | --- |
| TakeProfit - Open >= SYMBOL\_TRADE\_STOPS\_LEVEL<br> Open - StopLoss >= SYMBOL\_TRADE\_STOPS\_LEVEL | Open - TakeProfit >= SYMBOL\_TRADE\_STOPS\_LEVEL<br> StopLoss - Open >= SYMBOL\_TRADE\_STOPS\_LEVEL |

The _Test\_StopLoss\_Level\_in\_PendingOrders.mq5_ EA makes a series of attempts to place the BuyStop and BuyLimt orders until the operation succeeds. With each successful attempt the StopLoss or TakeProfit level is shifted by 1 point in the right direction. Example of this EA execution:

```
Test_StopLoss_Level_in_PendingOrders.mq5
SYMBOL_TRADE_EXECUTION=SYMBOL_TRADE_EXECUTION_INSTANT
SYMBOL_TRADE_FREEZE_LEVEL=20: order or position modification is not allowed, if there are 20 points to the activation price
SYMBOL_TRADE_STOPS_LEVEL=30: StopLoss and TakeProfit must not nearer than 30 points from the closing price
1. BuyStop 1.0 EURUSD at 1.11019 SL=1.10993 (Open-StopLoss=26 points)
CTrade::OrderSend: buy stop 1.00 EURUSD at 1.11019 sl: 1.10993 [invalid stops]
2. BuyStop 1.0 EURUSD at 1.11019 SL=1.10992 (Open-StopLoss=27 points)
CTrade::OrderSend: buy stop 1.00 EURUSD at 1.11019 sl: 1.10992 [invalid stops]
3. BuyStop 1.0 EURUSD at 1.11020 SL=1.10992 (Open-StopLoss=28 points)
CTrade::OrderSend: buy stop 1.00 EURUSD at 1.11020 sl: 1.10992 [invalid stops]
4. BuyStop 1.0 EURUSD at 1.11021 SL=1.10992 (Open-StopLoss=29 points)
CTrade::OrderSend: buy stop 1.00 EURUSD at 1.11021 sl: 1.10992 [invalid stops]
5. BuyStop 1.0 EURUSD at 1.11021 SL=1.10991 (Open-StopLoss=30 points)
BuyStop 1.0 EURUSD done at 1.11021 with StopLoss=1.10991 (SYMBOL_TRADE_STOPS_LEVEL=30)
 ---------
1. BuyLimit 1.0 EURUSD at 1.10621 TP=1.10647 (TakeProfit-Open=26 points)
CTrade::OrderSend: buy limit 1.00 EURUSD at 1.10621 tp: 1.10647 [invalid stops]
2. BuyLimit 1.0 EURUSD at 1.10621 TP=1.10648 (TakeProfit-Open=27 points)
CTrade::OrderSend: buy limit 1.00 EURUSD at 1.10621 tp: 1.10648 [invalid stops]
3. BuyLimit 1.0 EURUSD at 1.10621 TP=1.10649 (TakeProfit-Open=28 points)
CTrade::OrderSend: buy limit 1.00 EURUSD at 1.10621 tp: 1.10649 [invalid stops]
4. BuyLimit 1.0 EURUSD at 1.10619 TP=1.10648 (TakeProfit-Open=29 points)
CTrade::OrderSend: buy limit 1.00 EURUSD at 1.10619 tp: 1.10648 [invalid stops]
5. BuyLimit 1.0 EURUSD at 1.10619 TP=1.10649 (TakeProfit-Open=30 points)
BuyLimit 1.0 EURUSD done at 1.10619 with TakeProfit=1.10649 (SYMBOL_TRADE_STOPS_LEVEL=30)
```

Examples of checking the TakeProfit and StopLoss levels in pending orders can be found in the attached source files: _Check\_TP\_and\_SL.mq4_ and _Check\_TP\_and\_SL.mq5_.

### Attempt to modify order or position within the SYMBOL\_TRADE\_FREEZE\_LEVEL freeze level

The [SYMBOL\_TRADE\_FREEZE\_LEVEL](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) parameter may be set in the symbol specification. It shows the distance of freezing the trade operations for pending orders and open positions in points. For example, if a trade on financial instrument is redirected for processing to an external trading system, then a BuyLimit pending order may be currently too close to the current Ask price. And, if and a request to modify this order is sent at the moment when the opening price is close enough to the Ask price, it may happen so that the order will have been executed and modification will be impossible.

Therefore, the symbol specifications for pending orders and open positions may have a freeze distance specified, within which they cannot be modified. In general, before attempting to send a modification request, it is necessary to perform a check with consideration of the [SYMBOL\_TRADE\_FREEZE\_LEVEL](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer):

| Type of order/position | Activation price | Check |
| --- | --- | --- |
| Buy Limit order | Ask | Ask-OpenPrice >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Buy Stop order | Ask | OpenPrice-Ask >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Sell Limit order | Bid | OpenPrice-Bid >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Sell Stop order | Bid | Bid-OpenPrice >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Buy position | Bid | TakeProfit-Bid >= SYMBOL\_TRADE\_FREEZE\_LEVEL <br>Bid-StopLoss >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Sell position | Ask | Ask-TakeProfit >= SYMBOL\_TRADE\_FREEZE\_LEVEL<br>StopLoss-Ask >= SYMBOL\_TRADE\_FREEZE\_LEVEL |

The complete examples of functions for checking the SYMBOL\_TRADE\_FREEZE\_LEVEL level of orders and positions can be found in the attached _Check\_FreezeLevel.mq5_ and _Check\_FreezeLevel.mq4_ scripts.

```
//--- check the order type
   switch(type)
     {
      //--- BuyLimit pending order
      case  ORDER_TYPE_BUY_LIMIT:
        {
         //--- check the distance from the opening price to the activation price
         check=((Ask-price)>freeze_level*_Point);
         if(!check)
            PrintFormat("Order %s #%d cannot be modified: Ask-Open=%d points < SYMBOL_TRADE_FREEZE_LEVEL=%d points",
                        EnumToString(type),ticket,(int)((Ask-price)/_Point),freeze_level);
         return(check);
        }
      //--- BuyLimit pending order
      case  ORDER_TYPE_SELL_LIMIT:
        {
         //--- check the distance from the opening price to the activation price
         check=((price-Bid)>freeze_level*_Point);
         if(!check)
            PrintFormat("Order %s #%d cannot be modified: Open-Bid=%d points < SYMBOL_TRADE_FREEZE_LEVEL=%d points",
                        EnumToString(type),ticket,(int)((price-Bid)/_Point),freeze_level);
         return(check);
        }
      break;
      //--- BuyStop pending order
      case  ORDER_TYPE_BUY_STOP:
        {
         //--- check the distance from the opening price to the activation price
         check=((price-Ask)>freeze_level*_Point);
         if(!check)
            PrintFormat("Order %s #%d cannot be modified: Ask-Open=%d points < SYMBOL_TRADE_FREEZE_LEVEL=%d points",
                        EnumToString(type),ticket,(int)((price-Ask)/_Point),freeze_level);
         return(check);
        }
      //--- SellStop pending order
      case  ORDER_TYPE_SELL_STOP:
        {
         //--- check the distance from the opening price to the activation price
         check=((Bid-price)>freeze_level*_Point);
         if(!check)
            PrintFormat("Order %s #%d cannot be modified: Bid-Open=%d points < SYMBOL_TRADE_FREEZE_LEVEL=%d points",
                        EnumToString(type),ticket,(int)((Bid-price)/_Point),freeze_level);
         return(check);
        }
      break;
     }
```

You can simulate a situation where there is an attempt to modify a pending order within the freeze level. To do that, open a demo account with financial instruments that have non-zero SYMBOL\_TRADE\_FREEZE\_LEVEL level, then attach the _Test\_SYMBOL\_TRADE\_FREEZE\_LEVEL.mq5_ ( _Test\_SYMBOL\_TRADE\_FREEZE\_LEVEL.mq4_) EA to the chart and manually place any pending order. The EA will automatically move the order as close to the current price as possible and will start making illegal modification attempts. It will play a sound alert using the PlaySound() function.

### Errors that occur when working with symbols which have insufficient quote history

If an expert or indicator is launched on a chart with insufficient history, then there are two possibilities:

1. the program checks for availability of the required history in all the required depth. If there are less bars than required, the program requests the missing data and completes its operation before the next tick arrives. This path is the most correct and it helps to avoid a lot of mistakes, such as array out of range or zero divide;
2. the program does not make any checks and immediately starts its work, as if all the necessary history on all the required symbols and timeframes was available immediately upon request. This approach is fraught with many unpredictable errors.


You can try simulating this situation yourself. To do that, run the tested indicator or EA on the chart, then close the terminal and delete all history and start the terminal again. If there are no errors in the log after such a restart, then try changing the symbols and timeframes on the charts the program is running on. Many indicators give errors when started on weekly or monthly timeframes, which usually have a limited number of bars. Also, during a sudden change of the chart symbol (for example, from EURUSD to CADJPY), an indicator or an expert running on that chart may encounter an error caused by the absence of history required for its calculation.

### Array out of Range

When working with arrays, the access to their elements is performed by the index number, which cannot be negative and must be less than the array size. The array size can be obtained using the [ArraySize](https://www.mql5.com/en/docs/array/arraysize)() function.

This error can be encountered while working with a [dynamic array](https://www.mql5.com/en/docs/basis/types/dynamic_array) when its size has not been explicitly defined by the [ArrayResize()](https://www.mql5.com/en/docs/array/arrayresize) function, or when using such an array in functions that independently set the size of the dynamic arrays passed to them. For example, the [CopyTicks](https://www.mql5.com/en/docs/series/copyticks)() function tries to store the requested number of ticks to an array, but if there are less ticks than requested, the size of resulting array will be smaller than expected.

Another quite obvious way to get this error is to attempt to access the data of an indicator buffer while its size has not been initialized yet. As a reminder, the indicator buffers are dynamic arrays, and their sizes are defined by the terminal's execution system only after the chart initialization. Therefore, for instance, an attempt to access the data of such a buffer in the OnInit() function causes an "array out of range" error.

![](https://c.mql5.com/2/23/Out_of_range_en.png)

A simple example of an indicator that generates this error can be found in attached Test\_Out\_of\_range.mq5 file.

### Zero Divide

Another critical error is an attempt to divide by zero. In this case, the program execution is terminated immediately, the tester displays the name of the function and line number in the source code where the error occurred in the Journal.

![](https://c.mql5.com/2/23/zerodivide_en.png)

As a rule, division by zero occurs due to a situation unforeseen by the programmer. For example, getting a property or evaluating an expression with "bad" data.

The zero divide can be easily reproduced using a simple TestZeroDivide.mq5 EA, its source code is displayed in the screenshot. Another critical error is the use of incorrect [object pointer](https://www.mql5.com/en/docs/basis/types/object_pointers). The [Debugging on History Data](https://www.metatrader5.com/en/metaeditor/help/development/debug#history "https://www.metatrader5.com/en/metaeditor/help/development/debug#history") is useful for determining the cause of such error.

### Sending a request to modify the levels without actually changing them

If the rules of the trading system require the pending orders or open positions to be modified, then before sending a trade request to perform a transaction, it is necessary to make sure that the requested operation would actually change parameters of the order or position. Sending a trade request which does not make any changes is considered an error. The trade server would respond to such action with a TRADE\_RETCODE\_NO\_CHANGES=10025 [return code](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes) (MQL5) or with an [ERR\_NO\_RESULT=1](https://docs.mql4.com/en/constants/errorswarnings/enum_trade_return_codes "https://docs.mql4.com/en/constants/errorswarnings/enum_trade_return_codes") code (MQL4)

Example of the check in MQL5 is provided in the _Check\_OrderLevels.mq5_ script:

```
//--- class for performing trade operations
#include <Trade\Trade.mqh>
CTrade trade;
#include <Trade\Trade.mqh>
//--- class for working with orders
#include <Trade\OrderInfo.mqh>
COrderInfo orderinfo;
//--- class for working with positions
#include <Trade\PositionInfo.mqh>
CPositionInfo positioninfo;
//+------------------------------------------------------------------+
//| Checking the new values of levels before order modification      |
//+------------------------------------------------------------------+
bool OrderModifyCheck(ulong ticket,double price,double sl,double tp)
  {
//--- select order by ticket
   if(orderinfo.Select(ticket))
     {
      //--- point size and name of the symbol, for which a pending order was placed
      string symbol=orderinfo.Symbol();
      double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
      int digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
      //--- check if there are changes in the Open price
      bool PriceOpenChanged=(MathAbs(orderinfo.PriceOpen()-price)>point);
      //--- check if there are changes in the StopLoss level
      bool StopLossChanged=(MathAbs(orderinfo.StopLoss()-sl)>point);
      //--- check if there are changes in the Takeprofit level
      bool TakeProfitChanged=(MathAbs(orderinfo.TakeProfit()-tp)>point);
      //--- if there are any changes in levels
      if(PriceOpenChanged || StopLossChanged || TakeProfitChanged)
         return(true);  // order can be modified
      //--- there are no changes in the Open, StopLoss and Takeprofit levels
      else
      //--- notify about the error
         PrintFormat("Order #%d already has levels of Open=%.5f SL=%.5f TP=%.5f",
                     ticket,orderinfo.PriceOpen(),orderinfo.StopLoss(),orderinfo.TakeProfit());
     }
//--- came to the end, no changes for the order
   return(false);       // no point in modifying
  }
//+------------------------------------------------------------------+
//| Checking the new values of levels before order modification      |
//+------------------------------------------------------------------+
bool PositionModifyCheck(ulong ticket,double sl,double tp)
  {
//--- select order by ticket
   if(positioninfo.SelectByTicket(ticket))
     {
      //--- point size and name of the symbol, for which a pending order was placed
      string symbol=positioninfo.Symbol();
      double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
      //--- check if there are changes in the StopLoss level
      bool StopLossChanged=(MathAbs(positioninfo.StopLoss()-sl)>point);
      //--- check if there are changes in the Takeprofit level
      bool TakeProfitChanged=(MathAbs(OrderTakeProfit()-tp)>point);
      //--- if there are any changes in levels
      if(StopLossChanged || TakeProfitChanged)
         return(true);  // position can be modified
      //--- there are no changes in the StopLoss and Takeprofit levels
      else
      //--- notify about the error
         PrintFormat("Order #%d already has levels of Open=%.5f SL=%.5f TP=%.5f",
                     ticket,orderinfo.PriceOpen(),orderinfo.StopLoss(),orderinfo.TakeProfit());
     }
//--- came to the end, no changes for the order
   return(false);       // no point in modifying
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- price levels for orders and positions
   double priceopen,stoploss,takeprofit;
//--- ticket of the current order and position
   ulong orderticket,positionticket;
/*
   ... get the order ticket and new StopLoss/Takeprofit/PriceOpen levels
*/
//--- check the levels before modifying the pending order
   if(OrderModifyCheck(orderticket,priceopen,stoploss,takeprofit))
     {
      //--- checking successful
      trade.OrderModify(orderticket,priceopen,stoploss,takeprofit,
                        orderinfo.TypeTime(),orderinfo.TimeExpiration());
     }
/*
   ... get the position ticket and new StopLoss/Takeprofit levels
*/
//--- check the levels before modifying the position
   if(PositionModifyCheck(positionticket,stoploss,takeprofit))
     {
      //--- checking successful
      trade.PositionModify(positionticket,stoploss,takeprofit);
     }
//---
  }
```

Example of the check in the MQL4 language can be found in the _Check\_OrderLevels.mq4_ script:

```
#property strict
//+------------------------------------------------------------------+
//| Checking the new values of levels before order modification      |
//+------------------------------------------------------------------+
bool OrderModifyCheck(int ticket,double price,double sl,double tp)
  {
//--- select order by ticket
   if(OrderSelect(ticket,SELECT_BY_TICKET))
     {
      //--- point size and name of the symbol, for which a pending order was placed
      string symbol=OrderSymbol();
      double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
      //--- check if there are changes in the Open price
      bool PriceOpenChanged=true;
      int type=OrderType();
      if(!(type==OP_BUY || type==OP_SELL))
        {
         PriceOpenChanged=(MathAbs(OrderOpenPrice()-price)>point);
        }
      //--- check if there are changes in the StopLoss level
      bool StopLossChanged=(MathAbs(OrderStopLoss()-sl)>point);
      //--- check if there are changes in the Takeprofit level
      bool TakeProfitChanged=(MathAbs(OrderTakeProfit()-tp)>point);
      //--- if there are any changes in levels
      if(PriceOpenChanged || StopLossChanged || TakeProfitChanged)
         return(true);  // order can be modified
      //--- there are no changes in the Open, StopLoss and Takeprofit levels
      else
      //--- notify about the error
         PrintFormat("Order #%d already has levels of Open=%.5f SL=%.5f TP=%.5f",
                     ticket,OrderOpenPrice(),OrderStopLoss(),OrderTakeProfit());
     }
//--- came to the end, no changes for the order
   return(false);       // no point in modifying
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- price levels for orders and positions
   double priceopen,stoploss,takeprofit;
//--- ticket of the current order
   int orderticket;
/*
   ... get the order ticket and new StopLoss/Takeprofit/PriceOpen levels
*/
//--- check the levels before modifying the order
   if(OrderModifyCheck(orderticket,priceopen,stoploss,takeprofit))
     {
      //--- checking successful
      OrderModify(orderticket,priceopen,stoploss,takeprofit,OrderExpiration());
     }
  }
```

Recommended articles for reading:

1. [How to Make the Detection and Recovery of Errors in an Expert Advisor Code Easier](https://www.mql5.com/en/articles/1473)
2. [How to Develop a Reliable and Safe Trade Robot in MQL 4](https://www.mql5.com/en/articles/1462)

### Attempt to import compiled files (even EX4/EX5) and DLL

The programs distributed via the Market must have a guaranteed safety for users. Therefore, any attempts to use and DLL or functions from compiled EX4/EX5 files are considered mistakes. These products will not be published in the Market.

If your program needs to use additional indicators not present in the standard delivery, use [Resources](https://www.mql5.com/en/docs/runtime/resources).

### Calling custom indicators with iCustom()

If the operation of your program requires calling to the data of a custom indicator, then all the necessary indicators should be placed to the [Resources](https://www.mql5.com/en/docs/runtime/resources). Market products are supposed to be ready to work in any unprepared environment, so they should contain all everything they need within their EX4/EX5 files. Recommended related articles:

- [Use of Resources in MQL5](https://www.mql5.com/en/articles/261)
- [How to create an indicator of non-standard charts for MetaTrader Market](https://www.mql5.com/en/articles/2297)

### Passing an invalid parameter to the function (runtime error)

This type of errors is relatively rare, many of them have [ready codes](https://www.mql5.com/en/docs/constants/errorswarnings/errorcodes), that are designed to help in finding the cause.

| Constant | Value | Description |
| --- | --- | --- |
| ERR\_INTERNAL\_ERROR | 4001 | Unexpected internal error |
| ERR\_WRONG\_INTERNAL\_PARAMETER | 4002 | Wrong parameter in the inner call of the client terminal function |
| ERR\_INVALID\_PARAMETER | 4003 | Wrong parameter when calling the system function |
| ERR\_NOTIFICATION\_WRONG\_PARAMETER | 4516 | Invalid parameter for sending a notification — an empty string or [NULL](https://www.mql5.com/en/docs/basis/types/void) has been passed to the [SendNotification()](https://www.mql5.com/en/docs/network/sendnotification) function |
| ERR\_BUFFERS\_WRONG\_INDEX | 4602 | Wrong indicator buffer index |
| ERR\_INDICATOR\_WRONG\_PARAMETERS | 4808 | Wrong number of parameters when creating an indicator |
| ERR\_INDICATOR\_PARAMETERS\_MISSING | 4809 | No parameters when creating an indicator |
| ERR\_INDICATOR\_CUSTOM\_NAME | 4810 | The first parameter in the array must be the name of the custom indicator |
| ERR\_INDICATOR\_PARAMETER\_TYPE | 4811 | Invalid parameter type in the array when creating an indicator |
| ERR\_NO\_STRING\_DATE | 5030 | No date in the string |
| ERR\_WRONG\_STRING\_DATE | 5031 | Wrong date in the string |
| ERR\_TOO\_MANY\_FORMATTERS | 5038 | Amount of format specifiers more than the parameters |
| ERR\_TOO\_MANY\_PARAMETERS | 5039 | Amount of parameters more than the format specifiers |

The table does not list all the errors that can me encountered during the operation of a program.

### Access violation

This error occurs when trying to address memory, the access to which is denied. In each such case, it is necessary to contact the developers via the Service Desk in your Profile or via the [Contacts](https://www.mql5.com/en/contact) page. A detailed description of the steps to reproduce the error and an attached source code will significantly accelerate the search for the causes of this error and will help to improve the source code compiler.

![](https://c.mql5.com/2/23/Access_violation.png)

### Consumption of the CPU resources and memory

When writing a program, use of time-optimal algorithms is essential, as otherwise the operation of other programs running in the terminal might become hindered or even impossible.

**It is important to remember** that the terminal allocates one common thread for working per each symbol in the [Market watch](https://www.metatrader5.com/en/terminal/help/trading/market_watch "https://www.metatrader5.com/en/terminal/help/trading/market_watch"). All the indicators running and charts opened for that symbol are processed in that thread.

This means that if there are 5 charts opened for EURUSD on different timeframes and there are 15 indicators running on those charts, then all these charts and indicators receive only a single thread for calculating and displaying information on the chart. Therefore, one inefficient resource-consuming indicator running on a chart may slow down the operation of all other indicators or even inhibit rendering of prices on all other charts of the symbol.

You can easily check the time taken by your algorithm using the [GetMicrosecondCount](https://www.mql5.com/en/docs/common/getmicrosecondcount)() function. It is easy to get the execution time in microseconds by measuring the time between two lines of code. To convert this time into milliseconds (ms), it should be divided by 1000 (1 millisecond contains 1000 microseconds). Usually, the runtime bottleneck of indicators is the [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) handler. As a rule, the first calculation of the indicator is heavily dependent on the [Max bars in chart](https://www.metatrader5.com/en/terminal/help/startworking/settings#max_bars "https://www.metatrader5.com/en/terminal/help/startworking/settings#max_bars") parameter. Set it to "Unlimited" and run your indicator on a symbol with history of over 10 years on the M1 timeframe. If the first start takes too much time (for example, more than 100 ms), then code optimization is required.

Here is an example of measuring the execution time of the OnCalculate() handler in the [ROC](https://www.mql5.com/en/code/46) indicator, provided in the standard delivery of the terminal with source code. Insertions are highlighted in yellow:

```
//+------------------------------------------------------------------+
//| Rate of Change                                                   |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,const int prev_calculated,const int begin,const double &price[])
  {
//--- check for rates count
   if(rates_total<ExtRocPeriod)
      return(0);
//--- calculation start time
   ulong start=GetMicrosecondCount();
//--- preliminary calculations
   int pos=prev_calculated-1; // set calc position
   if(pos<ExtRocPeriod)
      pos=ExtRocPeriod;
//--- the main loop of calculations
   for(int i=pos;i<rates_total && !IsStopped();i++)
     {
      if(price[i]==0.0)
         ExtRocBuffer[i]=0.0;
      else
         ExtRocBuffer[i]=(price[i]-price[i-ExtRocPeriod])/price[i]*100;
     }
//--- calculation end time
   ulong finish=GetMicrosecondCount();
   PrintFormat("Function %s in %s took %.1f ms",__FUNCTION__,__FILE__,(finish-start)/1000.);
//--- OnCalculate done. Return new prev_calculated.
   return(rates_total);
  }
```

The memory in use can be measured with the [MQLInfoInteger(MQL\_MEMORY\_USED)](https://www.mql5.com/en/docs/check/mqlinfointeger) function. And, of course, use the code [Profiler](https://www.metatrader5.com/en/metaeditor/help/development/profiling "https://www.metatrader5.com/en/metaeditor/help/development/profiling") to find the costliest functions in your program. We also recommend reading [The Principles of Economic Calculation of Indicators](https://www.mql5.com/en/articles/109) and [Debugging MQL5 Programs](https://www.mql5.com/en/articles/654) articles.

The Experts work in their own threads, but all of the above applies to them as well. Writing optimal code in any types of programs is essential, be it expert, indicator, library or script.

### There cannot be too many checks

All of the above tips on checking the indicators and experts are recommended not only for publishing products in the Market, but also in common practice, when you write for yourself. This article did not cover all the errors that can be encountered during trading on real accounts. It did not consider the rules for handling the trade errors, which occur during loss of connection to the trading server, requotes, transaction rejections and many others that may disrupt the perfect rules of the trading system. Every robot developer has personal tried ready-made recipes for such cases.

Newcomers are recommended to read all the articles about error handling, as well as ask questions on the forum and in the article comments. Other more experienced members of the MQL5.community will help you figure out any unclear points. We hope that the information gathered in the article will help you create more reliable trading robots and in a shorter time.

**Recommended related articles:**

- [Common Errors in MQL4 Programs and How to Avoid Them](https://www.mql5.com/en/articles/1391)
- [Debugging MQL5 Programs](https://www.mql5.com/en/articles/654)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2555](https://www.mql5.com/ru/articles/2555)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2555.zip "Download all attachments in the single ZIP archive")

[2555\_en.zip](https://www.mql5.com/en/articles/download/2555/2555_en.zip "Download 2555_en.zip")(26.47 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/92921)**
(263)


![Aleh Sasonka](https://c.mql5.com/avatar/2016/3/56D58E26-6A47.jpg)

**[Aleh Sasonka](https://www.mql5.com/en/users/sova75)**
\|
18 Jan 2026 at 16:36

**fxsaber [#](https://www.mql5.com/ru/forum/91657/page26#comment_58964458):**

It may be sufficient to do this in Tester only.

_I agree, it is enough for publishing._

**fxsaber [#](https://www.mql5.com/ru/forum/91657/page26#comment_58964458):**

It is enough to do this check only in OnTrade.

_I'm not sure here. It's probably too late to check the margin in OnTrade._

|     |     |
| --- | --- |
| OnTrade | Called in Expert Advisors when the [Trade](https://www.mql5.com/en/docs/runtime/event_fire) event occurs, which is generated when a trade operation is completed on the trade server |

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
18 Jan 2026 at 16:44

**Aleh Sasonka [#](https://www.mql5.com/ru/forum/91657/page26#comment_58965813):**

_I'm not sure about that. It's probably too late to check the margin in OnTrade._

The levels of placed orders/SL/TP are always known. Accordingly, you can calculate the situation on the account, when prices (Ask/Bid) will reach these levels - from the closest to the current state. If the situation shows that the margin will not be enough, delete the corresponding order. This approach allows you to work only in OnTrade.

It is enough to write such a universal public function that can be called for all published Market Advisors in OnTrade. And then, probably, all the problems with Market Expert Advisors will be solved.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
18 Jan 2026 at 20:58

**Aleh Sasonka [#](https://www.mql5.com/ru/forum/91657/page26#comment_58964026):**

_This check is not enough when using pending orders!_

_There is no guarantee to pass the check:_

test on EURUSD,H1 2023.04.28 17:00:38 Tester: [not enough money](https://www.mql5.com/en/market/validation/errors/mt4/134) to buy 0.60 EURUSD at 1.10395 sl: 0.00000 tp: 0.00000 \[2023.04.28 17:00\] 2023.04.28 17:00:38 Tester: PrevBalance: 10272.11, PrevPL: -4308.04, PrevEquity 5964.07, PrevMargin: 6271.62, NewMargin: 6293, FreeMargin: -328.50 2023.04.28 17:00:38 Tester: pending order is deleted \[no enough money\] strategy tester report 360 total trades

_We will have to check the margin on every tick.... And what do we achieve by this? Additional load on the server?_

Well, before setting all orders, do an OrderCheck for real buy/sell on the volume of all orders, as if the orders were immediately executed. Let's write off the error due to price changes for the potential time before triggering, because in any case we need to leave some reserve in the margin.


![Andrea Capuani](https://c.mql5.com/avatar/avatar_na2.png)

**[Andrea Capuani](https://www.mql5.com/en/users/81590031)**
\|
23 Jan 2026 at 11:24

Good morning everyone, the test tells me that there are no operations.


![Vinicius Pereira De Oliveira](https://c.mql5.com/avatar/2025/4/6804f561-0038.png)

**[Vinicius Pereira De Oliveira](https://www.mql5.com/en/users/vinicius-fx)**
\|
23 Jan 2026 at 13:30

**Andrea Capuani [#](https://www.mql5.com/en/forum/92921/page27#comment_59009524):** Good morning everyone, the test tells me that there are no operations.

### [There are no trading operations](https://www.mql5.com/en/blogs/post/686716\#:~:text=it%20vanishes%20%22automatically%22.-,There%20are%20no%20trading%20operations,-This%20error%20is)

This error is specific for expert advisers only. The rule is: expert advisers _must_ trade. If your robot should be used only on a specific symbol timeframe, then here is what MetaQuotes say: "Products can not apply restrictions. All limitations should be marked as recommendations in the product description." If your robot does not trade by design (a helper tool, for example), choose approriate category ("Utilities") in the product properties.

![Graphical Interfaces VII: the Tables Controls (Chapter 1)](https://c.mql5.com/2/23/avatar-vii.png)[Graphical Interfaces VII: the Tables Controls (Chapter 1)](https://www.mql5.com/en/articles/2500)

The seventh part of the series on MetaTrader graphical interfaces deals with three table types: text label, edit box and rendered one. Another important and frequently used controls are tabs allowing you to show/hide groups of other controls and develop space effective interfaces in your MQL applications.

![LifeHack for Traders: Indicators of Balance, Drawdown, Load and Ticks during Testing](https://c.mql5.com/2/23/avac18.png)[LifeHack for Traders: Indicators of Balance, Drawdown, Load and Ticks during Testing](https://www.mql5.com/en/articles/2501)

How to make the testing process more visual? The answer is simple: you need to use one or more indicators in the Strategy Tester, including a tick indicator, an indicator of balance and equity, an indicator of drawdown and deposit load. This solution will help you visually track the nature of ticks, balance and equity changes, as well as drawdown and deposit load.

![Testing trading strategies on real ticks](https://c.mql5.com/2/23/test-real-tick-ava.png)[Testing trading strategies on real ticks](https://www.mql5.com/en/articles/2612)

The article provides the results of testing a simple trading strategy in three modes: "1 minute OHLC", "Every tick" and "Every tick based on real ticks" using actual historical data.

![Universal Expert Advisor: Integration with Standard MetaTrader Modules of Signals (Part 7)](https://c.mql5.com/2/23/zapvwy5wjkj_54w2.png)[Universal Expert Advisor: Integration with Standard MetaTrader Modules of Signals (Part 7)](https://www.mql5.com/en/articles/2540)

This part of the article describes the possibilities of the CStrategy engine integration with the signal modules included into the standard library in MetaTrader. The article describes how to work with signals, as well as how to create custom strategies on their basis.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/2555&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069330804393313097)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).