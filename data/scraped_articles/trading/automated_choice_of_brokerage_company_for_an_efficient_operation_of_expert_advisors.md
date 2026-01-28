---
title: Automated Choice of Brokerage Company for an Efficient Operation of Expert Advisors
url: https://www.mql5.com/en/articles/1476
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:24:15.913235
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kdvrqfcqbmywqhlqmpnuklpnnrroiahx&ssn=1769181854755259023&ssn_dr=0&ssn_sr=0&fv_date=1769181854&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1476&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automated%20Choice%20of%20Brokerage%20Company%20for%20an%20Efficient%20Operation%20of%20Expert%20Advisors%20-%20MQL4%20Articles&scr_res=1920x1080&ac=1769181854920929&fz_uniq=5069409642812998814&sv=2552)

MetaTrader 4 / Trading


### Introduction

Very often we face situations when an Expert Advisor successfully operates with
one brokerage company and is not profitable or even lossmaking on the other one.
The reasons may be different. Different brokerage companies have different settings:

- Quotes. They slightly differ because of two factors - different data feeds and different
filtration that smooths quotes. For some Expert Advisors this may be relevant.
Situations may occur when an EA trades often with one brokerage company and rarely
on another one.
- Slippage. It may differ much in different brokerage companies. This also may lead
to worse characteristics of an EA because of lower expected profit.

- Requotes. In some brokerage companies they are oftener than in others. In this case
an EA will miss successful entering points because of large number of requotes.


So, the operation of some EAs depends greatly on the brokerage company it works
with. If there were no choice, we could put up with quotes, slippage, and requotes.
But now we have choice, and it is getting wider and wider. So, we can compare different
brokerage companies, see their technical characteristics, and using this information
we can choose **the best brokerage companies for our EA**.

### Statistic

Of course we can start several terminals from different brokerage companies, use
them a month or two and then choose one, with which the profit is maximal. But
such testing is low-informative. It would be better to get more information: mean
slippage per a trade, number of requotes that take place at opening of each certain
trade, opening time and so on. In order not to have to analyze logs, a project
_Statistic_ was developed. It is based on the following set of rules:

1. An EA analyzes the market, makes transactions, gets all necessary data about the
trade and passes it into a common module.
2. This module contains all information about the current and closed deals. It also
counts the statistic about all technical characteristics of the brokerage companies.

3. It should be maximally comfortable for operation with large amounts of data, so
that we could see only necessary information, and not all that can be collected
and counted.

Analyzing the statistic (number of requotes, time of a trade execution, slippage)
and viewing all trades we can conclude, with what brokerage company it is better
to work. If statistic on all companies is negative, we should change some EA parameters,
for example, time in the market, frequency of trades. And change them until the
EA starts working with a profit. If on a certain stage an EA stops bringing profit
with one brokerage company being profitable on others, we stop testing this company.

### Theory

We can organize submission of data from an EA into an application ether through
files, or through dll. The variant with files is easier in terms of technical realization,
because it does not require serious system programming. However, it is not very
convenient to work with files, because we do not know beforehand where MetaTrader
4 terminals for different brokerage companies will be, on what currencies they
will be tested, what to do if the files are lost, and so on. If we need everything
to be done dynamically, with maximum security, it is better to organize data passing
through dll.

For different terminals to operate with one dll, it should be located in the system
directory windows\\system32. It is important that EAs from one terminal download
one and the same dll copy, because all they operate within one process, which is
the terminal (terminal.exe), it means the have one and the same address space,
i.e. they operate with the same variables. One terminal with EAs has its own dll
copy for all EAs and the same variables, announced inside dll, another terminal
has another copy with other variables. Thus a terminal does not get access to variables
from another terminal.

We want to create a single, where data from different terminals will be collected.
There are different ways to organize a synchronous operation of different processes
with one data field. This may be implemented through files, but again we will face
the problem of path indicating, as well as the problem with the processing speed
if we have many terminals and EAs. The best solution is a dividable core memory.
Working with it requires higher attention and knowledge of the features of the
used operating system (in our case it is Windows), however, the available possibilities
are infinite. For the organization of a successive access to a certain block of
the shared memory a special mechanism of program semaphores is used.

Theory conclusion: through dll EAs write data into shared memory, and then in the
application, let us call it Monitor, it reads data from the memory, displays it
and conducts necessary statistic calculations. When MetaTrader 4 calls DLL for
the first time, operating system generates a copy of this DLL for each terminal,
because each terminal is a separate process. The operating scheme is in the picture
below.

![](https://c.mql5.com/2/15/sheme.png)

### Practice

#### Expert Advisor

Certainly, data about the current trade should be formed by an Expert Advisor. For
data passing we need to form the interface of the function dll. For the implemented
task we need three functions:

```
bool NewExpert(string  isBrokerName, string isInstrument, int Digit);
```

Create a new Expert Advisor, identifying it by a broker's name and security. For
the calculation of some statistical characteristics we pass the number of figures
after a point in a security price.

```
bool NewDeal(string isInstrument, int Action, int magik,
double PriceOpen, int Slippage, int TimeForOpen, int Requotes);
```

The registration of a new trade is performed the following way. While the terminal-process
is already identified by a broker's name, for a new trade the name of a security
upon which the trade is executed is enough. Other parameters are the trade characteristics.

Table 1. Trade Opening

|     |     |
| --- | --- |
| **Parameter** | **Value** |
| Action | 0 – buy, 1 - sell |
| magik | Magic number |
| PriceOpen | Opening price |
| Slippage | Slippage |
| TimeForOpen | Opening duration |
| Requotes | Number of received requotes |

```
bool CloseDeal(string isInstrument, int magik, double PriceClose,
               int Slippage, int TimeForClose, int Requotes);
```

A trade closing is identified upon a security and the magic. Passed parameters:

Table 2. Trade Closing

|     |     |
| --- | --- |
| **Parameter** | **Value** |
| PriceClose | Closing price |
| Slippage | Slippage |
| TimeForClose | Closing duration |
| Requotes | Number of received requotes |

Due to this interface, initialization and opening and closing functions will look
like this:

_Initialization:_

```
int init()
  {
   int Digit;
   if(IsDllsAllowed() == false)
     {
       Print("Calling from libraries (DLL) is impossible." +
             " EA cannot be executed.");
       return(0);
     }
   if(!IsTradeAllowed())
     {
       Print("Trade is not permitted!");
       return(0);
     }
   Digit = MarketInfo(Symbol(), MODE_DIGITS);
   if((Digit > 0) && (Bid > 0))
     {
       if(!NewExpert(AccountServer(), Symbol(), Digit))
         {
           Print("Creation of a new broker failed");
           return (0);
         }
       Print("A broker is successfully created ");
       return(0);
     }
   Print("No symbol in MarketInfo!");
   return(0);
  }
```

During the initialization after checking the terminal parameters (trade permission
and confirmation of DLL calling) we receive the information about a security's
digits and its current price. If both parameters are more than zero, the security
is adequately presented in the terminal and we can work with it. Each broker differs
in its name that can be received using the function AccountServer(), upon this
name terminals differ from one another in the shared memory. EAs differ in the
name of security they are trading with. That is why if different EAs are attached
to one and the same currency pair, they will download one and the same DLL copy
which may lead to collision.

_Function of opening a new order:_

```
int Deal(int act, double Lot)
  {
   int N = 0;
   int ticket;
   int err;
   double Price_open;
   double Real_price;
   datetime begin_deal;
   double Lots;
   int cmd;
   int magik;
   magik = GenericMagik() + 1;
   Lots = NormalizeDouble(Lot, 1);
// checking margin for a position opening
   AccountFreeMarginCheck(Symbol(), cmd, Lots);
   err = GetLastError();
   if(err > 0)
     {
       Print("No money for new position");
       return(0);
     }
   begin_deal=TimeCurrent();
   while(N < count)
     {
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
       ticket = OrderSend(Symbol(), cmd, Lots, Price_open,
                          slippage, 0, 0, 0, magik);
       if(ticket > 0)
         {
           if(OrderSelect(ticket, SELECT_BY_TICKET) == true)
             {
               Real_price = OrderOpenPrice();
               NewDeal(Symbol(), cmd,magik, Real_price ,
                       MathAbs(Real_price - Price_open),
                       (TimeCurrent() - begin_deal), N);
             }
           return(ticket);
         }
       N++;
       Sleep(5000);
       RefreshRates();
     }
   return(0);
  }
```

An order is opened by the function Deal with two parameters: action (1 - buy, 2
\- sell ) and lot. Each order differs from the previous one in magic - it is incremented.
A position tries to open in **count** attempts. Thу information about the number of attempts together with opening duration, price and slippage is
passed into a shared memory, from where it is read by the monitor.

_Order closing function:_

```
bool CloseOrder(int magik)
  {
   int ticket, i;
   double Price_close;
   int count = 0;
   datetime begin_time;
   double Real_close;
   begin_time = TimeCurrent();
   for(i = OrdersTotal() - 1; i >= 0; i--)
     {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
           if(OrderSymbol() == Symbol())
               if(OrderMagicNumber() == magik)
                 {
                   while(count < 10)
                     {
                       if(OrderType() == OP_BUY)
                           Price_close = NormalizeDouble(Bid, Digits);
                       if(OrderType() == OP_SELL)
                           Price_close = NormalizeDouble(Ask, Digits);
                       if(OrderClose(OrderTicket(), OrderLots(),
                                     Price_close, slippage))
                         {
                           Real_close = OrderClosePrice();
                           CloseDeal(Symbol(), magik, Real_close,
                                     MathAbs(Real_close - Price_close),
                                     (TimeCurrent() - begin_time), count);
                           return(true);
                         }
                       count++;
                       Sleep(5000);
                       RefreshRates();
                     }
                 }
     }
   return(false);
  }
```

Function of closing CloseOrder() has only one input parameter - the magic. An order
tries to close several times and this number of attempts will be passed together
with the time of transaction execution, closing price and slippage into the memory
and then read by the monitor.

The remaining code is the tested EA. So for using Statistic in your own EAs, you
need to import the necessary dll functions; for initialization and opening/closing
positions use functions **Deal** and **CloseOrder**. If you want, you may rewrite these functions, but data on transactions should
be passed in accordance with the interface contained in dll.

Below is the example of the implementation of such an EA using DLL (code of the
above enumerated functions is not included).

```
// Enable dll for operation with monitor
#import "statistik.dll"
  bool NewExpert(string  isBrokerName, string isInstrument,
                 int Digit);   // Create a broker
  bool NewDeal(string isInstrument, int Action, int magik,
               double PriceOpen, int Slippage, int TimeForOpen,
               int Requotes);
  bool CloseDeal(string isInstrument, int magik, double PriceClose,
                 int Slippage, int TimeForClose,
                 int Requotes);
#import
//----
extern int Num_Deals = 3;
extern int TimeInMarket = 4;
// maximally acceptable slippage
int  slippage = 10;
// time for rest after a trade
int TimeForSleep = 10;
// period of request
int time_for_action = 1;
// number of attempts for opening a position
int count = 5;
// Function of a new bar
bool isNewBar()
  {
    static datetime BarTime;
    bool res = false;
    if(BarTime != Time[0])
      {
        BarTime = Time[0];
        res = true;
      }
   return(res);
  }
//+------------------------------------------------------------------+
//| Generation of magic                                              |
//+------------------------------------------------------------------+
int GenericMagic()
  {
   int deals;
//----
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
           if(OrderSymbol() == Symbol())
               if(OrderMagicNumber() != 0)
                   deals++;
     }
   return (deals);
  }
//+------------------------------------------------------------------+
//| forming signals to open/close a position                         |
//+------------------------------------------------------------------+
int GetAction(int &action, double &lot, int &magic)
   {
    int cnt, total;
    if(OrdersTotal() <= Num_Deals)
      {
        if(Close[1] > Close[2])
          {
            action = 1;
            lot = 1;
            return(0);
          }
        if(Close[2] < Close[1])
          {
            action = 2;
            lot = 1;
            return(0);
          }
      }
    total = OrdersTotal();
    for(cnt = total - 1; cnt >= 0; cnt--)
      {
        if(OrderSelect(cnt, SELECT_BY_POS, MODE_TRADES))
            if(OrderSymbol() == Symbol())
                if((TimeCurrent() - OrderOpenTime()) > TimeInMarket*60)
                  {
                    action = 3;
                    magic = OrderMagicNumber();
                    return(0);
                  }
      }
   }
//+------------------------------------------------------------------+
//| expert start function                                            |
//+------------------------------------------------------------------+
int start()
  {
   int action = 0;
   double lot = 1;
   int magic = 0;
   while(!IsStopped())
     {
       Sleep(time_for_action*1000);
       RefreshRates();
       if(isNewBar())
         {
           GetAction(action, lot, magic);
           if(((action == 1) || (action == 2)))
             {
               if(IsTradeAllowed())
                   Deal(action, lot);
               Sleep(TimeForSleep*1000);
             }
           if(action == 3)
             {
               if(IsTradeAllowed())
                   if(!CloseOrder(magik))
                     {
                       Print("MANUAL CLOSING OF A POSITION IS NEEDED");
                       Sleep(TimeForSleep*1000);
                     }
             }
           action = 0;
           lot = 0;
           magik = 0;
         }
     }
   Print("A serious error occurred, the EA stopped operating");
   return(0);
  }
//+------------------------------------------------------------------+
```

The EA's executive block is an infinite cycle on the start function. At a preset
frequency _time\_for\_action_ the EA calls the analytical function _GetAction()_, which by reference returns an action that should be done by the EA, lot with which
a position should be opened, and magic in case if you need to close a position.

The analytical block is elementary here - buy, if the previous bar was more than
the one before it, and sell if vice versa. Positions are closed by time. For testing
your own EAs simply rewrite this block in accordance with their algorithm. You
may make no changes in the executive part.

#### DLL

DLL may be implemented in different environments and in different languages. The
dll necessary for our work was created in Visual C++. The trades will have the
following structure:

```
struct DealRec
  {
    int Index;
    int Magic;
    int Cur;
    int Broker;
    double PriceOpen;
    double PriceClose;
    int SlipOpen;
    int SlipClose;
    int Action;  // 0 = BUY 1 = SELL
    int TimeForOpen;
    int TimeForClose;
    int ReqOpen;
    int ReqClose;
    int Profit;
    bool Checked; // indication that the position is closed
  };
```

They will be fulfilled in two stages - at opening and closing. I.e. one part of
data (opening price, slippage at opening etc.) is passed at opening, another part
(closing price, closing time etc.) is passed at closing. Prototypes of calling
a function in dll

```
__declspec(dllexport) bool __stdcall NewExpert (char *isBrokerName,
                                                char *isInstrument,
                                                int Digit);
__declspec(dllexport) bool __stdcall NewDeal (char *isInstrument,
                                              int Action,
                                              int magic,
                                              double PriceOpen,
                                              int Slippage,
                                              int TimeForOpen,
                                              int Requotes);
__declspec(dllexport) bool __stdcall CloseDeal (char *isInstrument,
                                                int magic,
                                                double PriceClose,
                                                int Slippage,
                                                int TimeForClose,
                                                int Requotes);
```

differ from prototypes in MQL4 only in passing lines. You may look through source
dll, which may help in creation of other projects. For the project recompiling
open the file statistic.dsw using the program Visual C++. The full code dll is
in the files statistic.cpp and statistic.h, the remaining ones are subsidiary.
All the enumerated files are in Statistic.zip.

#### Monitor

An optimal tool for a quick writing of applications with tables and graphical interface
\- solution from Borland. That is Delphi and C++Builder.

Monitor functions: create a shared memory, read data from it and display in tables,
keep the statistics of slippage. There are some more options that make the work
more convenient. So this is the functional of the monitor:

1. Keeping the journal of opened positions;
2. Keeping the journal of closed positions;
3. Statistics of slippage and requotes;
4. Adjustable tables;
5. Saving trades in html-file.

![](https://c.mql5.com/2/15/all_12.jpg)

The implementation is in the attached zip-file Statistic Monitor.zip. For the project recompiling use the program C++Builder. Extension of the project file is \*.bpr. The main code
is in в main.cpp.

### Testing

For testing a special EA was created - it has the simplest conditions of entering
and closing positions by time (the implementation was shown earlier). The EA with
dll and monitor.exe is in the zip-file monitor+dll+expert.zip. When starting, click START, thus creating a shared memory. DLL should be in the
folder system32. After that start several terminals and attach the EA to charts
of currencies, on which it is going to trade. After a number of trades statistics
is accumulated. The data is collected in the monitor/Journal. From time to time
they should be transfered into a file, stored in the form of html-page.

![](https://c.mql5.com/2/15/as1.gif)

The real operation of the application will be the same. It allows traders to compare
the operation of different brokerage companies in terms of their technical characteristics
and choose the best ones for automated trading.

### Conclusion

Enabling dll in MQL4 allows to develop different application programs, which help
not only make decisions about trading, but also collect statistics. The latter
one may be very useful in trafing and in choosing a brokerage company. The created
application should help developers in this difficult search. For analyzing brokers,
attach statistic.dll to an Expert Advisor as described in the example analyzed
in this article. The files necessary for the work are in monitor+dll+expert.zip. For the operation copy statistic.dll into the folder system32, start Statistic.
exe from any location and open terminal with Expert Advisors, which download dll,
start trading and passing their data into the shared memory. Statistic.exe creates
auxiliary files, that is why it is better to start the application from an empty
folder. If the program is interesting to the developers of trading robots, it can
be modified and amended.

It should be noted that not all brokerage companies provide similar conditions for
the automated trading:

1. A broker may prohibit automated trading.
2. A broker may prohibit to indicate at order placing SL or TP [https://www.mql5.com/ru/forum/103341](https://www.mql5.com/ru/forum/103341 "https://www.mql5.com/ru/forum/103341").
3. Nonsymmetrical levels for SL and TP.
4. May have no option of a mutual opening of orders.
5. Restriction on the number of simultaneously opened positions in an account. If the
number of orders (open positions + pending orders) exceeds the restriction, the
function OrderSend will return the error code ERR\_TRADE\_TOO\_MANY\_ORDERS.
6. Other restrictions.

That is why it is strongly recommended to read carefully the regulations of the
brokerage company you are going to work with.

The project Statistic shows, what complexes can be created if you add the possibilities
of other languages and programming environments to MQL4 options. The created program
will be useful for Expert Advisors working with different brokerage companies,
because it helps to analyze their technical characteristics in a convenient form.
If a brokerage companies has slippages, trades are executed by time and requotes
are quite often, then what for do we need such a brokerage company? There are so
many alternatives!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1476](https://www.mql5.com/ru/articles/1476)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1476.zip "Download all attachments in the single ZIP archive")

[monitorudllyexpert.zip](https://www.mql5.com/en/articles/download/1476/monitorudllyexpert.zip "Download monitorudllyexpert.zip")(524.74 KB)

[Statistic\_Monitor.zip](https://www.mql5.com/en/articles/download/1476/Statistic_Monitor.zip "Download Statistic_Monitor.zip")(219.41 KB)

[Statistic.zip](https://www.mql5.com/en/articles/download/1476/Statistic.zip "Download Statistic.zip")(24.28 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How Reliable is Night Trading?](https://www.mql5.com/en/articles/1373)
- [Easy Stock Market Trading with MetaTrader](https://www.mql5.com/en/articles/1566)
- [Price Forecasting Using Neural Networks](https://www.mql5.com/en/articles/1482)
- [How to Develop a Reliable and Safe Trade Robot in MQL 4](https://www.mql5.com/en/articles/1462)
- [How Not to Fall into Optimization Traps?](https://www.mql5.com/en/articles/1434)
- [Construction of Fractal Lines](https://www.mql5.com/en/articles/1429)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39382)**
(2)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
19 Oct 2007 at 21:33

I tried the code, and MQL doesn't recognize GenericMagik().


![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
28 Jun 2009 at 06:52

Write to sharedmemory from multe DLL attached multi MT, and read to moniter exe program.

In NewDeal(....)

before write, check

if((index = FindDeal(-1, -1, -1)) == -1)

{

[MessageBox](https://www.mql5.com/en/docs/constants/io_constants/messbconstants "MQL5 documentation: Constants of the MessageBox Dialog Window")(NULL, "xxxxxxxxxxxxxxxxx?DLL", MB\_OK);

ReleaseMutex(hSem);//

return true;

}

If last one some DLLwrited is not readed out, then no -1,-1,-1 item, then no write this newDeal, so this newDeal write faile FALSE, but also return true.

Then

How to sure multe DLL writing succeed ?

![Trading Strategy Based on Pivot Points Analysis](https://c.mql5.com/2/14/332_13.png)[Trading Strategy Based on Pivot Points Analysis](https://www.mql5.com/en/articles/1465)

Pivot Points (PP) analysis is one of the simplest and most effective strategies for high intraday volatility markets. It was used as early as in the precomputer times, when traders working at stocks could not use any ADP equipment, except for counting frames and arithmometers.

![MT4TerminalSync - System for the Synchronization of MetaTrader 4 Terminals](https://c.mql5.com/2/14/418_30.png)[MT4TerminalSync - System for the Synchronization of MetaTrader 4 Terminals](https://www.mql5.com/en/articles/1488)

This article is devoted to the topic "Widening possibilities of MQL4 programs by using functions of operating systems and other means of program development". The article describes an example of a program system that implements the task of the synchronization of several terminal copies based on a single source template.

![Object Approach in MQL](https://c.mql5.com/2/15/499_6.gif)[Object Approach in MQL](https://www.mql5.com/en/articles/1499)

This article will be interesting first of all for programmers both beginners and professionals working in MQL environment. Also it would be useful if this article were read by MQL environment developers and ideologists, because questions that are analyzed here may become projects for future implementation of MetaTrader and MQL.

![Automated Optimization of a Trading Robot in Real Trading](https://c.mql5.com/2/14/336_2.gif)[Automated Optimization of a Trading Robot in Real Trading](https://www.mql5.com/en/articles/1467)

The articles describes and provides a library of functions that allows a trader to optimize his or her Expert Advisor's inputs by launching optimization directly from the EA.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nwkhfnayfmbyqxjtzxsqljbuvcfmkvin&ssn=1769181854755259023&ssn_dr=0&ssn_sr=0&fv_date=1769181854&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1476&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automated%20Choice%20of%20Brokerage%20Company%20for%20an%20Efficient%20Operation%20of%20Expert%20Advisors%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176918185491966540&fz_uniq=5069409642812998814&sv=2552)

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