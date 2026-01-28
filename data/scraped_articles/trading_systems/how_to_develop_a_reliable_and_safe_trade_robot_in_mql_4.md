---
title: How to Develop a Reliable and Safe Trade Robot in MQL 4
url: https://www.mql5.com/en/articles/1462
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:58:05.853460
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=rtsldvlzqxihpsukuoqqjconpfryyocs&ssn=1769252284440688268&ssn_dr=0&ssn_sr=0&fv_date=1769252284&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1462&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Develop%20a%20Reliable%20and%20Safe%20Trade%20Robot%20in%20MQL%204%20-%20MQL4%20Articles&scr_res=1920x1080&ac=17692522848623186&fz_uniq=5083245987331315644&sv=2552)

MetaTrader 4 / Trading systems


### Introduction

In the process of creating any serious program solution the developer faces the fact that his program can contain all possible and impossible errors. The errors cause much trouble at the stage of development, lead to unreliability of the solution and, if it is a trade robot, can show negative result on your deposit. Let us analyze the most common errors, their origin, and the methods of detecting and program processing of errors. In the process of development and use of an Expert Advisor for the client terminal MetaTrader 4 the following errors can occur:

1. syntax – they can be found at the compilation stage and can be easily fixed by the programmer;
2. logical – they are not detected with a compiler. The examples are: a mess with the names of the variables, the wrong function calls, operation of the data of various types and so on;
3. algorithmic – they occur when the brackets are not placed correctly, in case of a mess with the branch statements and so on;
4. critical – these are improbable errors, you should spend some effort to evoke them. Non the less they often occur when you work with dll;
5. trading – these are the errors that occur when you work with the orders. Such kind of errors is a tender spot for trade robots.

First of all we recommend you to study the [documentation](https://docs.mql4.com/runtime/errors) about the execution errors. Once having performed this procedure, you can save much time later. The errors induced by trading operations are described [here](https://docs.mql4.com/constants/errorswarnings/errorcodes).

### Syntax Errors

The errors of this type are induced by misprints of operators, variables, different functions calls. During compilation the code of the program is checked, and all syntax errors are displayed in the “Tools” window of the MetaEditor. In fact almost all errors are detected and can be fixed by the programmer.

The exception is a mess with the brackets, when the wrong placed open/close bracket is detected at the compilation stage, but the placement of the error is displayed in a wrong way. Then you have to double check the code to be able to find the error visually, it, unfortunately, can be unsuccessful. The second approach is the successive shutdown of the blocks of the code using [comments](https://docs.mql4.com/basis/syntax/commentaries). In this case if after the commenting of a new block the error disappears, it is obviously placed in this commented block. It drastically narrows the area of search and helps to find the wrong placement of the brackets quickly.

### Logical, Algorithmic and Critical Errors

The most common errors of this type are the mess in the names and the types of the variables and also the algorithmic errors in the branches of the Expert Advisor. For example, let us study this code:

```
bool Some = false;

void check()
  {
    // Much code
    Some = true;
  }
// Very much code
int start()
  {
    bool Some = false;
    //
    if(Some)
      {
        //sending the order
      }
   return(0);
  }
```

What we can see now? The logical variable "Some", which is common for the whole program and is an important indicator for opening the position, was accidentally set lower. It can lead to the wrong opening of the order and, therefore, to losses. You can set many names for the variables! But for some reason these names accidentally repeat in big programs, and it leads to the problem mentioned above.

This kind of error occurs when the variables are mixed up or the expression of one type is assigned to the expression of another type. For example, in this line

```
int profit = NormalizeDouble(SomeValue*point*2 / 3, digit);
```

we are trying to assign the value expression of the "double" type to the variable of the "int" type, which results in zero value. And we are calculating the takeprofit level! This kind of error leads to wrong trading.

The algorithmic error in the branches of the Expert Advisor means that the brackets are placed not according to the algorithm, or the wrong coverage of 'if" operators by the "else" operators occurs. As a result we have the Expert Advisor, which does not work according to the technical requirement.

Some errors can be so imperceptible, that you can spend several hours, "meditating on the code" to find them. Unfortunately, there is no possibility to trace the variables values in MetaEditor, unlike the environments for the languages of the C++ family. So the only way is to trace the errors through the output of messages by the Print() function.

The function GetLastError() returns the code of the error. The last value is recommended to check after the each potentially vulnerable place of the program. Using the code of the error you can easily find its description in the documentation, and for some errors you can find even the methods of treatment.

We should say that the errors mentioned above most probably will be detected at the stage of testing before using the demo account, so the losses induced by them are improbable.

The main feature of the critical errors is that when they occur, the execution of the program immediately stops. Non the less, the code of the error stays unchanged in the predefined variable "last\_error". It gives us the possibility to learn the code of the error calling the function GetLastError().

### Trading Errors

These errors often lead to losses and non-operability of the Expert Advisor on the demo and, moreover, on the real accounts. They occur when you work with the sending and modification of orders, in other words, while interaction with the trading server.

The simple processing like this one:

```
ticket = OrderSend(Symbol(), OP_SELL, LotsOptimized(), Bid, 3,
         Bid + StopLoss*Point, Bid - TakeProfit*Point, 0, MAGICMA,
         0, Red);
if(ticket > 0)
  {
    err = GetLastError();
    Print("While opening the order the error #  occured", err);
  }
```

will not help. We made sure that the order has nor been sent to the server and learned the code of the error. So what of it? We missed an important entrance to the market, of course, if we have a profit-making Expert Advisor.

The variant with the endless loop:

```
while (true)
  {
    ticket = OrderSend(Symbol(), OP_SELL, Lots, Bid, slippage,
             Bid + StopLoss*Point, Bid - TakeProfit*Point, 0,
             MAGICMA, 0, Red);
    if(ticket > 0)
      {
        err = GetLastError();
        Print("While opening the order the error #  occured", err);
        break;
      }
    Sleep(1000);
    RefleshRates();
  }
```

helps a little bit. The order will more probably reach the server. But we can face some problems:

1. The broker will not like frequent requests;
2. The error can be fatal, in this case the request will not reach the server anyway;
3. The Expert Advisor will not respond for a long time;
4. The server may not accept trading requests at all - it can be a weekend, a holiday, maintenance works and so on.

Almost every error is unique and needs its own treatment. So let us discuss the variant with the [Switch](https://docs.mql4.com/basis/operators/switch) operator and cultivate each error more or less individually. The standard error #146 - 'Trade flow is busy", is processed using the semaphore realized in the TradeContext.mqh library. You can find the library and its detailed description [in this article](https://www.mql5.com/en/articles/1412).

```
//The library for differentiation of work with the trading flow
//written by komposter
#include <TradeContext.mqh>

//parameters for the signals
extern double MACDOpenLevel=3;
extern double MACDCloseLevel=2;
extern double MATrendPeriod=26;

// maximum acceptable slippage
int       slippage = 3;
//total number of transactions
int deals = 0;
//time for the pause after transaction
int TimeForSleep = 10;
//period of request
int time_for_action = 1;
//number of tries of opening/closing the position
int count = 5;
//indicator of operability of the EA
bool Trade = true;
//indicator of availability of funds for opening the position
bool NoOpen = false;
//+------------------------------------------------------------------+
//| Do not ask the server for quotes on weekends                     |
//+------------------------------------------------------------------+
bool ServerWork()
  {
   if(DayOfWeek() == 0 || DayOfWeek() == 6)
       return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Generation of magik                                              |
//+------------------------------------------------------------------+
int GenericMagik()
  {
   return(deals);
  }
//+------------------------------------------------------------------+
//| Closing of transactions                                          |
//+------------------------------------------------------------------+
bool CloseOrder(int magik)
  {
   int ticket,i;
   double Price_close;
   int err;
   int N;
//Function tries to shut the server at count attempts, if it fails,
//it gives an error message to the logfile
   while(N < count)
     {
       for(i = OrdersTotal() - 1; i >= 0; i--)
         {
           if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
               if(OrderSymbol() == Symbol())
                   if(OrderMagicNumber() == magik)
                     {
                       if(OrderType() == OP_BUY)
                           Price_close = NormalizeDouble(Bid, Digits);
                       if(OrderType() == OP_SELL)
                           Price_close = NormalizeDouble(Ask, Digits);
                       if(OrderClose(OrderTicket(), OrderLots(),
                          Price_close,slippage))
                         {
                           //reduce the number of transactions for the EA
                           deals--;
                           //the piece of the margin became available - you can open again
                           NoOpen = false;
                           return(true);
                         }
                         //we have reached this place, it means that the order has not been sent
                       N++;
                       //processing of possible errors
                       err = ErrorBlock();
                       //if the error is seriuos
                       if(err > 1)
                         {
                           Print("Manual closing of the order #  needed",
                                 OrderTicket());
                           return(false);
                         }
                     }
         }
        // taking a pause of 5 seconds and trying to close the transaction again
       Sleep(5000);
       RefreshRates();
     }
    //if we have reached this place, the transaction was not closed at count attempts
   Print("Manual closing of the order #  needed",OrderTicket());
   return(false);
  }
//+-----------------------------------------------------------------------------+
//|Transaction for act 1-buy, 2-sell, the second parameter - the number of lots |
//+-----------------------------------------------------------------------------+
int Deal(int act, double Lot)
  {
   int N = 0;
   int ticket;
   int err;
   double Price_open;
   double Lots;
   int cmd;
   int magik;
   magik = GenericMagik();
   Lots = NormalizeDouble(Lot,1);
   if(act == 1)
     {
       Price_open = NormalizeDouble(Ask, Digits);
       cmd = OP_BUY;
     }
   if(act == 2)
     {
       Price_open = NormalizeDouble(Bid, Digits);
       cmd = OP_SELL;
     }
   //checking the margin for opening the position
   AccountFreeMarginCheck(Symbol(), cmd,Lots);
   err = GetLastError();
   if(err>0)
     {
       Print("No money for new position");
       NoOpen = true;
       return(0);
     }
//Sending the order
   ticket = OrderSend(Symbol(), cmd, Lots, Price_open, slippage,
                      0, 0, 0, magik);
   if(ticket > 0)
     {
       deals++;
       return(ticket);
     }
//If the order has not been sent, we will try to open it 5 times again
   else
     {
       while(N < count)
         {
           N++;
           err = ErrorBlock();
           if(err == 1)
             {
               Sleep(5000);
               RefreshRates();
               if(act == 1)
                   Price_open = NormalizeDouble(Ask, Digits);
               if(act == 2)
                   Price_open = NormalizeDouble(Bid, Digits);
               ticket = OrderSend(Symbol(), cmd, Lots, Price_open,
                                  slippage, 0, 0, 0, magik);
               if(ticket > 0)
                 {
                   deals++;
                   return(ticket);
                 }
             }
           // we have got a serious error
           if(err > 1)
               return(0);
         }
     }
   return(0);
  }
//+------------------------------------------------------------------+
//| 0-no error, 1-need to wait and refresh,                          |
//| 2-transaction rejected, 3-fatal error                            |
//+------------------------------------------------------------------+
//Block of the error control
int ErrorBlock()
  {
   int err = GetLastError();
   switch(err)
     {
       case 0: return(0);
       case 2:
         {
           Print("System failure. Reboot the computer/check the server");
           Trade = false;
           return(3);
         }
       case 3:
         {
           Print("Error of the logic of the EA");
           Trade = false;
           return(3);
         }
       case 4:
         {
           Print("Trading server is busy. Wait for 2 minutes.");
           Sleep(120000);
           return(2);
         }
       case 6:
         {
           bool connect = false;
           int iteration = 0;
           Print("Disconnect ");
           while((!connect) || (iteration > 60))
             {
               Sleep(10000);
               Print("Connection not restored", iteration*10,
                     "  seconds passed");
               connect = IsConnected();
               if(connect)
                 {
                   Print("Connection restored");
                   return(2);
                 }
               iteration++;
             }
           Trade = false;
           Print("Connection problems");
           return(3);
         }
       case 8:
         {
           Print("Frequent requests");
           Trade = false;
           return(3);
         }
       case 64:
         {
           Print("Account is blocked!");
           Trade = false;
           return(3);
         }
       case 65:
         {
           Print("Wrong account number???");
           Trade = false;
           return(3);
         }
       case 128:
         {
           Print("Waiting of transaction timed out");
           return(2);
         }
       case 129:
         {
           Print("Wrong price");
           return(1);
         }
       case 130:
         {
           Print("Wrong stop");
           return(1);
         }
       case 131:
         {
           Print("Wrong calculation of trade volume");
           Trade = false;
           return(3);
         }
       case 132:
         {
           Print("Market closed");
           Trade = false;
           return(2);
         }
       case 134:
         {
           Print("Lack of margin for performing operation");
           Trade = false;
           return(2);
         }
       case 135:
         {
           Print("Prices changed");
           return (1);
         }
       case 136:
         {
           Print("No price!");
           return(2);
         }
       case 138:
         {
           Print("Requote again!");
           return(1);
         }
       case 139:
         {
           Print("The order is in process. Program glitch");
           return(2);
         }
       case 141:
         {
           Print("Too many requests");
           Trade = false;
           return(2);
         }
       case 148:
         {
           Print("Transaction volume too large");
           Trade = false;
           return(2);
         }
     }
   return (0);
  }
//+------------------------------------------------------------------+
//| generation of signals for opening/closing position on Macd       |
//+------------------------------------------------------------------+
int GetAction(int &action, double &lot, int &magik)
   {
   double MacdCurrent, MacdPrevious, SignalCurrent;
   double SignalPrevious, MaCurrent, MaPrevious;
   int cnt,total;

   MacdCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,0);
   MacdPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,1);
   SignalCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,0);
   SignalPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,1);
   MaCurrent=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,0);
   MaPrevious=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,1);

  if(MacdCurrent<0 && MacdCurrent>SignalCurrent && MacdPrevious<SignalPrevious &&
         MathAbs(MacdCurrent)>(MACDOpenLevel*Point) && MaCurrent>MaPrevious)
      {
         action=1;
         lot=1;
         return (0);
      }
  if(MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious &&
         MacdCurrent>(MACDOpenLevel*Point) && MaCurrent<MaPrevious)
      {
         action=2;
         lot=1;
         return (0);
      }
   total=OrdersTotal();
   for(cnt=0;cnt<total;cnt++)
     {
      OrderSelect(cnt, SELECT_BY_POS, MODE_TRADES);
      if(OrderType()<=OP_SELL &&   // check for opened position
         OrderSymbol()==Symbol())  // check for symbol
        {
         if(OrderType()==OP_BUY)   // long position is opened
           {
            // should it be closed?
            if(MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious &&
               MacdCurrent>(MACDCloseLevel*Point))
                {
                 action=3;
                 magik=OrderMagicNumber();
                 return(0); // exit
                }
           }
         else // go to short position
           {
            // should it be closed?
            if(MacdCurrent<0 && MacdCurrent>SignalCurrent &&
               MacdPrevious<SignalPrevious && MathAbs(MacdCurrent)>(MACDCloseLevel*Point))
              {
               action=3;
               magik=OrderMagicNumber();
               return(0);
              }
           }
        }
     }
   }
//+------------------------------------------------------------------+
//| The EA initialization function                                   |
//+------------------------------------------------------------------+
int init()
  {
    if(!IsTradeAllowed())
      {
        Print("Trade not allowed!");
        return(0);
      }
  }
//+------------------------------------------------------------------+
//| The EA deinitialization function                                 |
//+------------------------------------------------------------------+
int deinit()
  {
//Closing all orders
   for(int k = OrdersTotal() - 1; k >= 0 ; k--)
       if(OrderSymbol() == Symbol())
         {
           if(OrderType() == OP_BUY)
              OrderClose(OrderTicket(), OrderLots(),
                         NormalizeDouble(Bid,Digits), 10);
           if(OrderType() == OP_SELL)
               OrderClose(OrderTicket(), OrderLots(),
                          NormalizeDouble(Ask, Digits),10);
         }
  }
//+------------------------------------------------------------------+
//| The EA start function                                            |
//+------------------------------------------------------------------+
int start()
  {
   int action =0;
   double lot = 1;
   int magik = 0;
   while(Trade)
     {
       Sleep(time_for_action*1000);
       RefreshRates();
       /*Logic of the EA where the we calculate the action, position limit and magik for closing the order
       action 1-buy, 2-sell, 3-close
       for example, take the EA on Macd*/
       GetAction(action,lot,magik);
       if(ServerWork())
         {
           if(((action == 1) || (action == 2)) && (!NoOpen))
             {
               if(TradeIsBusy() < 0)
                   return(-1);
               Deal(action, lot);
               Sleep(TimeForSleep*1000);
               TradeIsNotBusy();
             }
           if(action == 3)
             {
               if(TradeIsBusy() < 0)
                 {
                   return(-1);
                   if(!CloseOrder(magik))
                       Print("MANUAL CLOSE OF TRANSATION NEEDED");
                   Sleep(TimeForSleep*1000);
                   TradeIsNotBusy();
                 }
             }
         }
       else
         {
            Print("Weekends");
            if(TradeIsBusy() < 0)
                  return(-1);
            Sleep(1000*3600*48);
            TradeIsNotBusy();
         }
       action = 0;
       lot = 0;
       magik = 0;
     }
   Print("Critical error occured and the work of the EA terminated");
   return(0);
  }
//+------------------------------------------------------------------+
```

This version of the trade robot works in an endless loop. Its demand occurs when the scalping multicurrency Expert Advisor created. The algorithm of operating of the EA is the following:

1. Get the signal from the analytical block GetAction();
2. Make the necessary transaction in the functions Deal() и CloseOrder();
3. Return to the point 1 after a short pause time\_for\_action in case that there were no serious failures.

After getting the signal (buy, sell, close) from the analyzing block the Expert Advisor blocks the trading flow ( [read the article](https://www.mql5.com/en/articles/1412)) and tries to make the transaction, after that it takes a pause for a few seconds and releases the trading flow for other EAs. The Expert Advisor tries to send the order no more than "count" times. It should be enough for the order to pass on the unsteady market where you can get requotes. If while sending the order a serious error occurred, the Expert Advisor stops functioning. If any problem occurs, an error message appears in the "Expert Advisors" folder. The Expert Advisor will continue working if the error is not critical.

The errors are processed in the ErrorBlock() procedure according to the following scheme: the procedure gets the code of the error and gives a short algorithm of processing it. For most of the errors it is just a message in the log. If the error is serious, then the trade indicators Trade and NoOpen change. If it is a connection failure, the processing of the situation is a little bit more complicated. The robot tries to reach the server sixty times with the predefined periodic sequence. If the server is not reached, then most likely it has some serious problems, and you should stop your trading for some time. Depending on the influence of the error on trading, the processing algorithm returns different meanings:

- 0 - no error;
- 1 - the error is associated with the volatility of the market, you can try to send the order once again;
- 2 - a serious error occurred while sending this order, stop opening positions for some time;
- 3 - a serious failure of the EA, connection failure - stop trading until the clarifying of the circumstances.

### Conclusion

Syntax, algorithmic and logical errors occur when you do not pay much attention to the algorithm coding. These errors are fixed by thorough checking and verifying the values of variables in the log. They can be detected at the stage of compilation and testing the Expert Advisor. Such kinds of errors do not exist for a long time, they are usually fixed before using a demo account.

Trading errors occur while sending the orders to the server. They are related with the real trading where you can find requotes, slippage, dealers’ fight with scalping and equipment failures. Such errors cannot be predicted, but they can be and should be processed. You should process them individually each week depending on the logic of the Expert Advisor, the frequency of transactions and modification of orders.

The mistakes which occur during the operation of the Expert Advisor, need processing. It is not a trivial task. It depends on the complexity of the EA and its features. In the article you can find the exemplary pattern of the Expert Advisor which performs this task. It takes much time to create more secure and safe trading system. But the time spent for developing of trouble-free automated trading system will be rewarded several times by the security of your deposit and your quiet sleep.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1462](https://www.mql5.com/ru/articles/1462)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1462.zip "Download all attachments in the single ZIP archive")

[AntiError.mq4](https://www.mql5.com/en/articles/download/1462/AntiError.mq4 "Download AntiError.mq4")(14.86 KB)

[TradeContext.mqh](https://www.mql5.com/en/articles/download/1462/TradeContext.mqh "Download TradeContext.mqh")(11.54 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How Reliable is Night Trading?](https://www.mql5.com/en/articles/1373)
- [Easy Stock Market Trading with MetaTrader](https://www.mql5.com/en/articles/1566)
- [Price Forecasting Using Neural Networks](https://www.mql5.com/en/articles/1482)
- [Automated Choice of Brokerage Company for an Efficient Operation of Expert Advisors](https://www.mql5.com/en/articles/1476)
- [How Not to Fall into Optimization Traps?](https://www.mql5.com/en/articles/1434)
- [Construction of Fractal Lines](https://www.mql5.com/en/articles/1429)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39315)**
(2)


![Jinsong Zhang](https://c.mql5.com/avatar/2010/6/4C2450DB-041A.jpg)

**[Jinsong Zhang](https://www.mql5.com/en/users/song_song)**
\|
8 Feb 2008 at 16:34

mark


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
22 Mar 2011 at 12:22

```
while((!connect) || (iteration > 60))
```

Looks like an infinite loop. Should be &&.

Actually, you don't really need (!connect), because the loop will end once it's true.

And:

```
               Sleep(10000);
               Print("Connection not restored", iteration*10,
                     "  seconds passed")
```

After 10 seconds, the log will say "0 seconds passed". (0\*10=0)

![MQL4 Language for Newbies. Introduction](https://c.mql5.com/2/14/404_19.gif)[MQL4 Language for Newbies. Introduction](https://www.mql5.com/en/articles/1475)

This sequence of articles is intended for traders, who know nothing about programming, but have a desire to learn MQL4 language as quick as possible with minimal time and effort inputs. If you are afraid of such phrases as "object orientation" or "three dimensional arrays", this article is what you need. The lessons are designed for the maximally quick result. Moreover, the information is delivered in a comprehensible manner. We shall not go too deep into the theory, but you will gain the practical benefit already from the first lesson.

![Modelling Requotes in Tester and Expert Advisor Stability Analysis](https://c.mql5.com/2/14/246_2.png)[Modelling Requotes in Tester and Expert Advisor Stability Analysis](https://www.mql5.com/en/articles/1442)

Requote is a scourge for many Expert Advisors, especially for those that have rather sensitive conditions of entering/exiting a trade. In the article, a way to check up the EA for the requotes stability is offered.

![Running MetaTrader 4 Client Terminal on Linux-Desktop ](https://c.mql5.com/i/0.gif)[Running MetaTrader 4 Client Terminal on Linux-Desktop](https://www.mql5.com/en/articles/1433)

Description of a step-by-step Linux-desktop setup using a non-emulator wine for running MetaTrader 4 Client Terminal on it.

![Ten "Errors" of a Newcomer in Trading?](https://c.mql5.com/2/13/193_3.png)[Ten "Errors" of a Newcomer in Trading?](https://www.mql5.com/en/articles/1424)

The article substantiates approach to building a trading system as a sequence of opening and closing the interrelated orders regarding the existing conditions - prices and the current values of each order's profit/loss, not only and not so much the conventional "alerts". We are giving an exemplary realization of such an elementary trading system.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pvsaszfipyqhmafyndboefanoosqjjwi&ssn=1769252284440688268&ssn_dr=0&ssn_sr=0&fv_date=1769252284&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1462&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Develop%20a%20Reliable%20and%20Safe%20Trade%20Robot%20in%20MQL%204%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925228486229177&fz_uniq=5083245987331315644&sv=2552)

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