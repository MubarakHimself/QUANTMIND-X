---
title: An Example of a Trading Strategy Based on Timezone Differences on Different Continents
url: https://www.mql5.com/en/articles/59
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:23:37.095934
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/59&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069400477352789111)

MetaTrader 5 / Examples


### Introduction

I had some free time and it was clear that this time I will be spending studying the markets and researching the divergence of economical cycles and technical indicators. You can see the results of the research in the following article [Creating an Indicator with Graphical Control Options](https://www.mql5.com/en/articles/42). But these were not only findings! I have found a phenomenon of an even larger scale, but to understand it, let's take a look at our world in terms of time zones (Fig. 1).

![](https://c.mql5.com/2/1/timemap.gif)

Figure 1. Timezones

As we can see, the day begins differently in each country and continues with various times, as we see our great vast country expanded through almost 10 timezones, while the Atlantic Ocean only spans over 6 timezones.

What is the pattern here? Let's look at the order of the opening of the markets in the following countries: Japan, Australia, China, Russia. By the time that Russia clerks first arrive at their office and begin trade operations, in Asia it is already evening, and when the European session is opened, they are already closed.

Here is where the fun begins. Once the market is opened, European brokers and hedge fund managers put out their assets on the market, and take speculative or concurrent with their investors' interests, actions. No, it's too early for the most interesting part, which begins with the dawn in Chicago, in connection with the lesser extent of the Atlantic Ocean, there comes a time when the managers of the Chicago stock market open their trading terminals and begin to manage their capital.

Let us pause at this moment, in connection with a lesser extent of the Atlantic Ocean. The European session has not yet ended (Fig. 1). London is separated from Chicago by 8 time zones, if we assume that the working day is 8 hours + put 45-minute for smoke breaks and an hour for, which prolongs the time for another 1,30 - 2 hours (those who have worked in an office know this).

This is the time when we gain the market assets of European and American managers. I'm afraid to list all of the zeros of assets at this point, European and American managers are beginning their struggle for the price of the stock exchanges, and at this point is a phenomenon, as shown in Figure 2.

![](https://c.mql5.com/2/1/92461820.gif)

Figure 2. Market  Pulse

Financial battles fill the market, assets shift their owners every second, but this is not what the market ends on! After this period is over, a quiet movement begins and moves in the direction of the current trend. At this point, the range expands its boundaries, and then narrows them once again and further continues to move in the direction of the main trend.

I'm sure you've noticed that the market price may go either up or down, but it always moves towards the right.

### 2\. Flowcharts as the primary method of developing an algorithm

The first block of the "Start-Stop"  program is demonstrated in Figure 3:

![](https://c.mql5.com/2/1/start-block.jpg)

Figure 3. "Start-Stop" block

This block is used to denote the start of the program, the beginning and the end of a function or other procedure, such as such as an initialization and deinitialization. The next block we will look at is labeled "Data" and is depicted in Figure 4.

![](https://c.mql5.com/2/1/incoming.jpg)

Figure 4. "Data" block

The "Data" block is used to determine the specified parameters at the startup of the program, or the output variables in the case MQL5. This unit also serves the function of the destination of global variables.

Next, we consider a commonly used block (99% of programs in MQL use this method) - it is depicted in two parts, which mark the boundaries of a cycle. Consider the Fig. 5:

![Cycle blocks](https://c.mql5.com/2/1/cyrkle__1__1__1.jpg)

Figure 5. Cycle blocks

Processes such as attribution or accounting, usually take place within these cycles, examples of which are depicted in Figure 6.

![Actions](https://c.mql5.com/2/1/deistvia__1.jpg)

Figure 6. Actions

And we mustn't forget about the logical blocks - in Fig. 7 demonstrates the "solution" block.

![](https://c.mql5.com/2/1/if.jpg)

Figure 7. The "Solution" block

The "solution" block may also have more than two outputs if it is located inside the operator of the "switch which depends on the number of placings" type. This unit will have a corresponding number of outputs.

The next block induces predefined functions, such as [iMACD](https://www.mql5.com/en/docs/indicators/imacd) or [iRSA](https://www.mql5.com/en/docs/indicators/irsi), as well as custom functions, defined elsewhere in the program or in the library (Fig. 8).

![](https://c.mql5.com/2/1/function1.jpg)

Figure 8. Function

And the last two blocks implement solely service functions - such as comments and the breach (Fig. 9).

![Service blocks](https://c.mql5.com/2/1/comment__1.jpg)

Figure 9. Service blocks

These are all of the types of blocks, which can be used to describe any program written for a machine; they are all clear, simple and easy to use In the primary stage of development, they also reveal the system's weak points and devise methods for eliminating them.

You are now acquainted with this method, however, I'm not asking you to act precisely according to these schemes, but simply knowing the initial value of flowcharts, it is easy enough to understand some type of calculation method. This method helps me to quickly formulate an idea, which had quickly shot into the left hemisphere of my brain, ready to escape from the right hemisphere.

### 3\. Construction of the algorithm

And so, let's move on to preparing the Expert Advisor based on the flowchart strategy.

The first block will be requesting the input parameters. As we have determined, it is vital to important to wait through the moment of the main battles, i.e. 2 hours after the opening session of the United States, when the European markets close. We will watch this on the global clock, i.e, on the terminal time, and so we will calculate the opening hour ourselves.

Then we determine the size of the positions and the levels of profit and loss, which in the future have the potential to optimize. Special attention will be paid to the Magic number parameter, as it will be used by our Expert Advisor to determine its order and open trades. Further on, we will make a trailing stop for limiting the risk of our positions, to our observations.

There is another interesting parameter which we will need - a safety level, according to which we will watch the presence of significant economic news at the moment, and whether or not they pose a threat, considering that the main panic on the markets arises within these two hours.

![](https://c.mql5.com/2/1/incoming__1.jpg)

Figure 10. Input parameters

```
//--- input parameters
input int      America=16;
input double   Lots=0.1;
input int      TakeProfit=500;
input long     MagicNumber=665;
input int      Limited=600;
input int      TrailingStop=100;
```

Let's proceed with the next part of strategy formation.

We need to determine whether the session was a cross and whether the orders have been or are being set. (Fig. 11).

![Time to trade](https://c.mql5.com/2/1/kdhd7ur4__1__2__1.jpg)

Figure 11. Time to trade

As you can see, the given algorithm is a closed program with input parameters, completed calculations and output results. Such mini-programs are called functions and are protected from the main program by encapsulation.

Encapsulation is a barrier between the programs or parts of the program, which is separated by methods, such as Get and Set (get and set) to prevent them from going through the territory of other Gets' and Sets'. The essence of this process lies in the fact that the names of variables may be the same within the functions and within the main program, but when the Get method tries to take from a cell with a variable name, it will be faced with encapsulation, which will only give it access to a particular sector of memory cells, allocated for this function or program.

The same applies to the Set method, but unlike Get. It sets the value in the cell's memory to the name of the variable, and if the names of variables within the program and within the function coincide, then encapsulation will not allow the Set method to assign the values of variables inside another program or function.

```
bool time2trade(int TradeHour,int Number)
  {
   MqlDateTime time2trade;
   TimeTradeServer(time2trade);
   if(time2trade.hour!=TradeHour) return(false);
   time2trade.hour= 0;
   time2trade.min = 0;
   time2trade.sec = 1;
   for(int ii=OrdersTotal()-1;ii>=0;ii--)
     {
      OrderGetTicket(ii);
      long ordmagic=OrderGetInteger(ORDER_MAGIC);
      if(Number==ordmagic) return(false);
     }
   HistorySelect(StructToTime(time2trade),TimeTradeServer());
   for(int ii=HistoryOrdersTotal()-1;ii>=0;ii--)
     {
      long HistMagic=HistoryOrderGetInteger(HistoryOrderGetTicket(ii),ORDER_MAGIC);
      if(Number==HistMagic) return(false);
     }
   return(true);
  }
```

We have identified the required session and determined whether or not we have set orders. Let us consider what should be done next.

Previously, we noticed that the major fluctuations occur 2 hours after the opening session of the United States. We obtain 9 fifteen-minute bars after the opening of the American session. We find the maximum range for this period and consider it carefully - if this variation is large enough, then most likely, there is a wide-spread panic in the market, and future trends are difficult to predict. Therefore, we will need some restriction here.

When the market is calm, then the session will increase their volatility. This will give us the opportunity to determine the maximum deviation from the main trend, and to place the order-traps at a safe distance, which will work because the main trend will continue. As we noted earlier, the price may go either up or down, but it always shifts to the right. (Fig. 12).

![Algorithm of order placement](https://c.mql5.com/2/1/0rytcz1s__1__1.jpg)

Figure 12. Algorithm of order set

Shaping the program code, pay attention to the fact that the trading terminal MetaTrader 5 does not allow the setting of an order close to the price of the last transaction. If at this moment the price draws a new minimum or maximum, we will defend our position by stepping back to a minimum distance from the last transaction price, for a reliable determination of an order. Also, we will set the duration period of our orders before the end of the day, as they will later no longer be effective.

```
void OnTick()
  {
//---
   if(time2trade(America+2,MagicNumber))
     {
      int i;
      double Highest = 0;
      double Lowest = 0;
      MqlRates Range[];
      CopyRates(Symbol(),15,0,9,Range);
      Lowest=Range[1].low;
      for(i=0; i<9;i++)
        {
         if(Highest<Range[i].high) Highest=Range[i].high;//MathMax(,Highest);
         if(Lowest>Range[i].low)  Lowest=Range[i].low;
        }
      long StopLevel=SymbolInfoInteger(Symbol(),SYMBOL_TRADE_STOPS_LEVEL);
      Highest=Highest+StopLevel*Point();
      // add to the current prices parameters of the minimum distance possible for the setting of orders
      Lowest=Lowest-StopLevel*Point();
      // to ensure the maximum probability of the acceptance of our order 30>

      if((Higest-Lowest)/Point()<Limited)
        {
         MqlTradeRequest BigDogBuy;
         MqlTradeRequest BigDogSell;
         BigDogBuy.action=TRADE_ACTION_PENDING;
         // Set the pending order
         BigDogBuy.magic = MagicNumber;
         BigDogBuy.symbol=Symbol();
         BigDogBuy.price=Highest;
         //Price by which the order will be set
         BigDogBuy.volume=Lots;
         BigDogBuy.sl=Lowest;
         //if the stop loss is not set, then set by the strategy /s39>
         BigDogBuy.tp=Highest+TakeProfit*Point();
         //set the take profit/s41>
         BigDogBuy.deviation=dev;
         //minimum deviation from the requested price,
         //in other words, by how much the executed price can differ from the specified price
         BigDogBuy.type=ORDER_TYPE_BUY_STOP;
         //order type, which is executed based on the specified price or by a higher than specified price
         //in this case the order is set to a higher or equal amount to the specified price
         //if the order type was buy_limit, then it would be executed
         //by the specified price, or prices lower than the specified price
         BigDogBuy.type_filling=ORDER_FILLING_AON;
         //the given parameter demonstrates how the order acts
         //with partial execution of the scope
         BigDogBuy.expiration=TimeTradeServer()+6*60*60;
         //by the strategy text the order life span only for the current work day
         //since it has been 2 hours since the opening of the American market, and the work day is 8 hours, we have 8-2 = 6
         BigDogSell.action=TRADE_ACTION_PENDING;

         // Set the pending order
         BigDogSell.magic = MagicNumber;
         BigDogSell.symbol=Symbol();
         BigDogSell.price=Lowest;
         //Price, by which the order will be set
         BigDogSell.volume=Lots;
         BigDogSell.sl=Highest;
         //Stop loss set by the strategy
         BigDogSell.tp=Lowest-TakeProfit*Point();
         //set the take profit
         BigDogSell.deviation=dev;
         //Minimum deviation from the requested price,
         //in other words, by how much the executed price can differ from the specified price
         BigDogSell.type=ORDER_TYPE_SELL_STOP;
         //order type, which is executed based on the specified price or by a higher than specified price
         //in this case the order is set to a higher or equal amount to the specified price
         //if the order type was buy_limit, then it would be executed
         //by the specified price, or prices lower than the specified price
         BigDogSell.type_filling=ORDER_FILLING_AON;
         //the given parameter demonstrates how the order acts
         ///with partial execution of the scope
         BigDogSell.expiration=TimeTradeServer()+6*60*60;
         //by the strategy text the order life span only for the current work day
         //since it has been 2 hours since the opening of the American market, and the work day is 8 hours, we have 8-2 = 6
         MqlTradeResult ResultBuy,ResultSell;
         OrderSend(BigDogBuy,ResultBuy);
         OrderSend(BigDogSell,ResultSell);
        }
     }
```

Orders are placed, the traps are set - now it's time to take care of reducing the risks of our positions, let's apply technology of the trailing stop (trailing stop).

To identify our position, we will use the Magic number (MagicNumber), and shift the level of stop-loss, when it reaches a certain level of profit with minimal price changes.(Figure 13).

![Implementation of trailing stop](https://c.mql5.com/2/1/Trall__1__1__2.jpg)

Figure 13. Implementation of the trailing stop

For the different strategies, the trailing stop is implemented using the simplest method, although in some strategies, it is recommended not to use a trailing stop, so as not to prevent the price from achieving its aim, or to use such a mechanism only for the transfer of positions to no-loss. But in this strategy, we apply the classical mechanism for moving the protective stop, in case of price shift, in our direction for a certain number of minimal price changes.

```
//--- trailing implementation
   int PosTotal=PositionsTotal();
   for(int i=PosTotal-1; i>=0; i--)
     {
      //--- go through open positions and see if there are positions created by this Expert Advisor.
      if(PositionGetSymbol(i)==Symbol())
        {
         if(MagicNumber==PositionGetInteger(POSITION_MAGIC))
           {
            MqlTick lasttick;
            SymbolInfoTick(Symbol(),lasttick);
            if(PositionGetInteger(POSITION_TYPE)==0)
              { //--- buy
               if(TrailingStop>0
                  &&(((lasttick.bid-PositionGetDouble(POSITION_PRICE_OPEN))/Point())>TrailingStop)
                  && ((lasttick.bid-PositionGetDouble(POSITION_SL))/Point())>TrailingStop)
                 {
                  MqlTradeRequest BigDogModif;
                  ZeroMemory(BigDogModif);
                  BigDogModif.action= TRADE_ACTION_SLTP;
                  BigDogModif.symbol= Symbol();
                  BigDogModif.sl = lasttick.bid - TrailingStop*Point();
                  BigDogModif.tp = PositionGetDouble(POSITION_TP);
                  BigDogModif.deviation=3;
                  MqlTradeResult BigDogModifResult;
                  ZeroMemory(BigDogModifResult);
                  OrderSend(BigDogModif,BigDogModifResult);
                 }
              }
            if(PositionGetInteger(POSITION_TYPE)==1)
              {//--- sell
               if(TrailingStop>0
                  && ((PositionGetDouble(POSITION_PRICE_OPEN)-lasttick.ask)/Point()>TrailingStop)
                  && (PositionGetDouble(POSITION_SL)==0
                  || (PositionGetDouble(POSITION_SL)-lasttick.ask)/Point()>TrailingStop))
                 {
                  MqlTradeRequest BigDogModif;
                  ZeroMemory(BigDogModif);
                  BigDogModif.action= TRADE_ACTION_SLTP;
                  BigDogModif.symbol= Symbol();
                  BigDogModif.sl = lasttick.ask + TrailingStop*Point();
                  BigDogModif.tp = PositionGetDouble(POSITION_TP);
                  BigDogModif.deviation=3;
                  MqlTradeResult BigDogModifResult;
                  ZeroMemory(BigDogModifResult);
                  OrderSend(BigDogModif,BigDogModifResult);
                 }
              }
           }
        }
     }
  }
```

Next, gather our algorithm (Fig. 14).

![](https://c.mql5.com/2/1/incoming__2.jpg)

![](https://c.mql5.com/2/1/kdhd7ur4__1__3.jpg)

![](https://c.mql5.com/2/1/0rytcz1s__2.jpg)

![](https://c.mql5.com/2/1/Trall__1__2.jpg)

Fig. 14. Assembling an algorithm

```
//+------------------------------------------------------------------+
//|                                          BigDog_By_CoreWinTT.mq5 |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "2010, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
#property version   "1.00"
//--- input parameters
input int      America=16;
input double   Lots=0.1;
input int      TakeProfit=500;
input long     MagicNumber=665;
input int      Limited=600;
input int      TrailingStop=100;
int dev=30;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
  }
//+------------------------------------------------------------------+
bool time2trade(int TradeHour,int Number)
  {
   MqlDateTime time2trade;
   TimeTradeServer(time2trade);
   if(time2trade.hour!=TradeHour) return(false);
   time2trade.hour= 0;
   time2trade.min = 0;
   time2trade.sec = 1;
   for(int ii=OrdersTotal()-1;ii>=0;ii--)
     {
      OrderGetTicket(ii);
      long ordmagic=OrderGetInteger(ORDER_MAGIC);
      if(Number==ordmagic) return(false);
     }
   HistorySelect(StructToTime(time2trade),TimeTradeServer());
   for(int ii=HistoryOrdersTotal()-1;ii>=0;ii--)
     {
      long HistMagic=HistoryOrderGetInteger(HistoryOrderGetTicket(ii),ORDER_MAGIC);
      if(Number==HistMagic) return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(time2trade(America+2,int(MagicNumber)))
     {
      int i;
      double Highest= 0;
      double Lowest = 0;
      MqlRates Range[];
      CopyRates(Symbol(),15,0,9,Range);
      Lowest=Range[1].low;
      for(i=0; i<9;i++)
        {
         if(Highest<Range[i].high) Highest=Range[i].high;
         if(Lowest>Range[i].low) Lowest=Range[i].low;
        }
      long StopLevel=SymbolInfoInteger(Symbol(),SYMBOL_TRADE_STOPS_LEVEL);
      Highest=Highest+StopLevel*Point();
      //--- add to the current prices the parameters of a minimum possible distance for the order set.
      Lowest=Lowest-StopLevel*Point();
      //--- to ensure the maximum probability of the acceptance of our order.

      if((Highest-Lowest)/Point()<Limited)
        {
         MqlTradeRequest BigDogBuy;
         MqlTradeRequest BigDogSell;
         ZeroMemory(BigDogBuy);
         ZeroMemory(BigDogSell);
         BigDogBuy.action=TRADE_ACTION_PENDING;
         //--- set the pending order
         BigDogBuy.magic = MagicNumber;
         BigDogBuy.symbol=Symbol();
         BigDogBuy.price=Highest;
         //--- Price by which the order will be set
         BigDogBuy.volume=Lots;
         BigDogBuy.sl=Lowest;
         //--- if the stop loss is not established, then we set by the strategy
         BigDogBuy.tp=Highest+TakeProfit*Point();
         //--- set the take profit
         BigDogBuy.deviation=dev;
         //--- Minimum deviation from the requested price,
         //--- in other words, by how much the executed price can differ from the specified price
         BigDogBuy.type=ORDER_TYPE_BUY_STOP;
         //--- order type, which is executed based on the specified price or by a higher than specified price
         //--- in this case the order is set to a higher or equal amount to the specified price
         //--- if the order type was buy_limit, then it would be executed
         //--- by the specified price, or prices lower than the specified price
         BigDogBuy.type_filling=ORDER_FILLING_FOK;
         //--- the given parameter demonstrates how the order acts
         //--- with partial execution of the scope
         BigDogBuy.expiration=TimeTradeServer()+6*60*60;
         //--- by the strategy text the order life span only for the current work day
         //--- since it has been 2 hours since the opening of the American market,
         //--- and the work day is 8 hours, we have 8-2 = 6
         BigDogSell.action=TRADE_ACTION_PENDING;

         //-- Set the pending order
         BigDogSell.magic = MagicNumber;
         BigDogSell.symbol=Symbol();
         BigDogSell.price=Lowest;
         //--- Price by which the order will be set
         BigDogSell.volume=Lots;
         BigDogSell.sl=Highest;
         //-- Stop loss set by the strategy
         BigDogSell.tp=Lowest-TakeProfit*Point();
         //--- Set take profit
         BigDogSell.deviation=dev;
         //--- Minimum deviation from the requested price,
         //--- in other words, by how much the executed price can differ from the specified price
         BigDogSell.type=ORDER_TYPE_SELL_STOP;
         //--- order type, which is executed based on the specified price or by a higher than specified price
         //--- in this case the order is set to a higher or equal amount to the specified price
         //--- if the order type was buy_limit, then it would be executed
         //--- by the specified price, or prices lower than the specified price
         BigDogSell.type_filling=ORDER_FILLING_FOK;
         //--- the given parameter demonstrates how the order acts
         //--- with partial execution of the scope
         BigDogSell.expiration=TimeTradeServer()+6*60*60;
         //-- by the strategy text the order life span only for the current work day
         //--- since it has been 2 hours since the opening of the American market,
         //--- and the work day is 8 hours, we have 8-2 = 6
         MqlTradeResult ResultBuy,ResultSell;
         ZeroMemory(ResultBuy);
         ZeroMemory(ResultSell);
         OrderSend(BigDogBuy,ResultBuy);
         OrderSend(BigDogSell,ResultSell);
        }
     }

//--- trailing implementation
   int PosTotal=PositionsTotal();
   for(int i=PosTotal-1; i>=0; i--)
     {
      //--- go through open positions and see if there are positions created by this Expert Advisor.
      if(PositionGetSymbol(i)==Symbol())
        {
         if(MagicNumber==PositionGetInteger(POSITION_MAGIC))
           {
            MqlTick lasttick;
            SymbolInfoTick(Symbol(),lasttick);
            if(PositionGetInteger(POSITION_TYPE)==0)
              { //--- buy
               if(TrailingStop>0
                  &&(((lasttick.bid-PositionGetDouble(POSITION_PRICE_OPEN))/Point())>TrailingStop)
                  && ((lasttick.bid-PositionGetDouble(POSITION_SL))/Point())>TrailingStop)
                 {
                  MqlTradeRequest BigDogModif;
                  ZeroMemory(BigDogModif);
                  BigDogModif.action= TRADE_ACTION_SLTP;
                  BigDogModif.symbol= Symbol();
                  BigDogModif.sl = lasttick.bid - TrailingStop*Point();
                  BigDogModif.tp = PositionGetDouble(POSITION_TP);
                  BigDogModif.deviation=3;
                  MqlTradeResult BigDogModifResult;
                  ZeroMemory(BigDogModifResult);
                  OrderSend(BigDogModif,BigDogModifResult);
                 }
              }
            if(PositionGetInteger(POSITION_TYPE)==1)
              {//--- sell
               if(TrailingStop>0
                  && ((PositionGetDouble(POSITION_PRICE_OPEN)-lasttick.ask)/Point()>TrailingStop)
                  && (PositionGetDouble(POSITION_SL)==0
                  || (PositionGetDouble(POSITION_SL)-lasttick.ask)/Point()>TrailingStop))
                 {
                  MqlTradeRequest BigDogModif;
                  ZeroMemory(BigDogModif);
                  BigDogModif.action= TRADE_ACTION_SLTP;
                  BigDogModif.symbol= Symbol();
                  BigDogModif.sl = lasttick.ask + TrailingStop*Point();
                  BigDogModif.tp = PositionGetDouble(POSITION_TP);
                  BigDogModif.deviation=3;
                  MqlTradeResult BigDogModifResult;
                  ZeroMemory(BigDogModifResult);
                  OrderSend(BigDogModif,BigDogModifResult);
                 }
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

### **Conclusion**

There are many market phenomena that occur periodically and regularly, and when investigated, can grant us some advantage. Also, perhaps, experienced traders noticed some overlap with the well-known strategy of "BigDog", the article does not mention this, and I did that on purpose - so that the reader would think about how they really are prepared.

In the Internet you can acquaint yourself with the different variations of this strategy, this article deals only with the phenomenon, on which this strategy is based.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/59](https://www.mql5.com/ru/articles/59)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/59.zip "Download all attachments in the single ZIP archive")

[bigdog\_by\_corewintt.mq5](https://www.mql5.com/en/articles/download/59/bigdog_by_corewintt.mq5 "Download bigdog_by_corewintt.mq5")(8.67 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Transferring Indicators from MQL4 to MQL5](https://www.mql5.com/en/articles/66)
- [Creating an Indicator with Graphical Control Options](https://www.mql5.com/en/articles/42)

**[Go to discussion](https://www.mql5.com/en/forum/1147)**

![Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners](https://c.mql5.com/2/0/create_EA_step_by_step_MQL5.png)[Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners](https://www.mql5.com/en/articles/100)

The Expert Advisors programming in MQL5 is simple, and you can learn it easy. In this step by step guide, you will see the basic steps required in writing a simple Expert Advisor based on a developed trading strategy. The structure of an Expert Advisor, the use of built-in technical indicators and trading functions, the details of the Debug mode and use of the Strategy Tester are presented.

![Connection of Expert Advisor with ICQ in MQL5](https://c.mql5.com/2/0/icq.png)[Connection of Expert Advisor with ICQ in MQL5](https://www.mql5.com/en/articles/64)

This article describes the method of information exchange between the Expert Advisor and ICQ users, several examples are presented. The provided material will be interesting for those, who wish to receive trading information remotely from a client terminal, through an ICQ client in their mobile phone or PDA.

![Creating a Multi-Currency Indicator, Using a Number of Intermediate Indicator Buffers](https://c.mql5.com/2/0/Multicurrency_Indicator_MQL5.png)[Creating a Multi-Currency Indicator, Using a Number of Intermediate Indicator Buffers](https://www.mql5.com/en/articles/83)

There has been a recent rise of interest in the cluster analyses of the FOREX market. MQL5 opens up new possibilities of researching the trends of the movement of currency pairs. A key feature of MQL5, differentiating it from MQL4, is the possibility of using an unlimited amount of indicator buffers. This article describes an example of the creation of a multi-currency indicator.

![The Magic of Filtration](https://c.mql5.com/2/17/893_81.jpg)[The Magic of Filtration](https://www.mql5.com/en/articles/1577)

Most of the automated trading systems developers use some form of trading signals filtration. In this article, we explore the creation and implementation of bandpass and discrete filters for Expert Advisors, to improve the characteristics of the automated trading system.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=cqpigbkwotkdadbhoqgxsybkxaycqvwj&ssn=1769181815706555787&ssn_dr=0&ssn_sr=0&fv_date=1769181815&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F59&back_ref=https%3A%2F%2Fwww.google.com%2F&title=An%20Example%20of%20a%20Trading%20Strategy%20Based%20on%20Timezone%20Differences%20on%20Different%20Continents%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918181570593212&fz_uniq=5069400477352789111&sv=2552)

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