---
title: Object Approach in MQL
url: https://www.mql5.com/en/articles/1499
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:57:18.740470
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/1499&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083237105338947469)

MetaTrader 4 / Trading systems


### Introduction

This article will be interesting first of all for programmers both beginners and
professionals working in MQL environment. Also it would be useful if this article
were read by MQL environment developers and ideologists, because questions that
are analyzed here may become projects for future implementation of MetaTrader and
MQL. To some extent alike ideas may be found in articles **[Universal Expert Advisor Template](https://www.mql5.com/en/articles/1495)** and [**Sending Trading Signals in a Universal Expert Advisor**](https://www.mql5.com/en/articles/1436)

### So,

One of the disadvantages of MQL, in my programmer opinion, is the absence of the
object approach in constructing the model of the trading system. MQL developers
offer us two ways out: using the calling of external functions or using the order
parameter MAGIC for the identification of the order belonging.

Actually, if only one system operates on one account, we do not need identification.
But when we have the program option of attaching to one account several automated
trading systems, then we cannot do without MAGIC. Even when calling external functions,
we need to determine them. Of course, we may build an array OrderTicket and identify
the array belonging only to one trading system, but as we know in some brokerage
companies the order ticket changes at swap (actually, one is closed, another one
is opened). That is why we cannot do without using MAGIC.

So, while developers are busy improving MQL language making it more flexible, let
us try to implement already now the object approach in building a trading model.

This is a trading system in accordance with my object model. Of course, it is not
universal, but by now I do not see other approaches.

![](https://c.mql5.com/2/15/1.png)

**So, let us analyze this model.**

**A). Signal System (SS).**

Object of this module process and interpret incoming quotes. Usually the "object"
of the signal system is a set of indicators, for example, moving averages. As a
result of processing quotes and indicator values, "the object" (or **semaphore**) **produces signals** to enter/exit, or order modification etc.

The semaphore forms its signal and sends it to another object from the module **Entry/Exit (EE)**.

Setting the semaphore in MQL is rather easy.

**1\. Define a global identifier using #define.**

It is better to set not consecutive numbers like 1, 2, 3, 4..., but in 5-10, so
that in an Expert Advisor we could use one signal for several processes (see the
second module).

```
//+------------------------------------------------------------------+
//|                                                      Signals.mqh |
//|                                    Copyright © 2007 Сергеев Алексей |
//|                                                los@we.kherson.ua |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2007, Сергеев Алексей "
#property link      "mailto: los@we.kherson.ua"
#property library

#define BLACKSYS   10
#define BORCHAN    20
#define ELDER      80
#define ENVELOP    90
```

**2\. Then in the global function of this module we should enable its processor.**

```
int CheckSignal(bool bEntry, int SignalID)
{
      switch (SignalID)
      {
                  case BLACKSYS:             return (BlackSys(bEntry)); break;
                  case BORCHAN:              return (BorChan(bEntry)); break;
                  case ELDER:                   return (Elder(bEntry)); break;
                  case ENVELOP:              return (Envelop(bEntry)); break;
                  default:                                     return (-1);
      }
}
```

**3\. And the last step is the description of functions.**

Here is an example for processing signals of an object that inherits the features
of the indicator Envelope.

```
int Envelope(bool bEntry)
{
      int MA=21;
      double Deviation=0.6;
      int Mode=MODE_SMA;//0-sma, 1-ema, 2-smma, 3-lwma
      int Price=PRICE_CLOSE;//0-close, 1-open, 2-high, 3-low, 4-median, 5-typic, 6-wieight

      double envH0, envL0, m0;
      double envH1, envL1, m1;
      envH0=iEnvelopes(NULL, 0, MA, Mode, 0, Price, Deviation, MODE_UPPER, 0);
      envL0=iEnvelopes(NULL, 0, MA, Mode, 0, Price, Deviation, MODE_LOWER, 0);
      envH1=iEnvelopes(NULL, 0, MA, Mode, 0, Price, Deviation, MODE_UPPER, 1);
      envL1=iEnvelopes(NULL, 0, MA, Mode, 0, Price, Deviation, MODE_LOWER, 1);

      m0 = (Low[0]+High[0])/2;          m1 = (Low[1]+High[1])/2;
      //----- condition for operation execution
      if (bEntry)   //for opening
      {
                  if (envH0<m0 && envH1<m1) return (OP_SELL);
                  if (envL0>m0 && envL1>m1) return (OP_BUY);
      }
      else //for closing
      {
                  if (envH0<m0 && envH1<m1) return (OP_BUY);
                  if (envL0>m0 && envL1>m1) return (OP_SELL);
      }

   return (-1); //no signal
}
```

**Thus we get a module that will contain different objects-signals.**

**B).** Objects of **the block EE** have minimal tasks:

First, its objects interact with objects-signals - observe them. The life cycle
and interaction is the following:

Checking semaphore -> if there are any positive signals, open/close/modify positions
-\> Passing control to objects in the module PS.

All the objects in the module EE have a prefix Process…, which determine its behavior more specifically.

**For example:**

```
ProcessAvgLim         //  -  the object processes signals with opening pending limit-orders and positions averaging
ProcessTurn           //  -  the object processes signals with position turning
```

Each sample of a trading system class (we all understand this and use in our modules)
must have its **own individual characteristics**, such as profit, stop-loss, its own Money Management, as well as other additional
parameters, implemented in different trailing variants etc.

When implementing these features I tried several variants of approach and the most
suiting in MQL, in my opinion, is creating a two-dimensional array. Here is its
description:

```
double SysPar[nSignal][11];

#define _TP        0 // Profit
#define _NullTP    1 // profit level, after which we set into losslessness
#define _NullTP2   2 // profit level, after which we set into losslessness
#define _TS        3 // distance of the trailing stop
#define _NullSL    4 // level, after achieving which the expected profit is transfered into opening point
#define _SL        5 // level, after achieving which the expected profit is transfered into opening point
#define _dSL       6 // the initial step upon the opening level of the next order in the position support
#define _dStep     7 // The step is increased in .. times upon the level of the next opening
#define _dLot      8 // In how many times (as compared to the last one) we increase the lot on the next one
#define _nLot      9 // In how many times (as compared to the initial one) we increase the lot on the next one

string SysParName[nSignal];
```

where **nSignal** is the identifier of object-signals.

**For example:**

```
SysPar[ENVELOP][_TS] = 40.0;    // distance of the trailing stop
SysPar[ENVELOP][_NullSL] = 20.0;// level, after achieving which the expected profit is transfered into opening point
SysPar[ENVELOP][_SL] = 70;      // changing stop-loss
```

Upon your desire you may increase the number of set parameters of this array-structure.

So, after setting the parameters we call the function of semaphore processing. In
other words we **interact with the signal system**. It is done in my favorite function **_**start()**_**

```
void start()
{
      ProcessAvgLim(ENVELOP, ENVELOP, Green, Red);
… …
```

![](https://c.mql5.com/2/15/2.png)

As it is seen in the scheme, in the trading system we have 4 registered semaphores
and 3 observers. Each semaphore is based on its own variant of quote interpretation.

For example, **Semaphore 1** sends signals analyzing **the indicator MACD**. **Observer 1** in its turn after receiving these signals opens orders in a simple scheme **ProcessSimple**.

Observers 2 and 3 are more difficult. Each one controls signals of two semaphores.
And, consequently, the approach to order opening is different.

So, after we set parameters of the observer and attach a semaphore to it, we need
to control and trail opening positions.

"Responsible" for the state of opened orders are objects of the module
**Position Support (PS)**.

**C). Block PS** is in my opinion the most interesting one, and not less important than semaphores.

Here different trailing variants are implemented, pending orders are opened, position
support and locking, profit and loss controlling is implemented and so on. Such
a PS should adequately react on EE signals about exiting the market in case of
loss positions with minimal losses.

_There is an interesting library of trailings on this site [**Library of Functions and Expert Advisors for trailing / Yury Dzyuban**](https://www.mql5.com/ru/code/7108). All trailing types are easily attached to the system._

For an easier perception, all support objects start from the prefix Trailing…

It has the following scheme:

![](https://c.mql5.com/2/15/3.png)

Calling, control transfer from an observer to trailing is done in the same function
start()

```
void start()
{
      ProcessSimple(MACD, MACD, Black, Plum); TrailingSimple(MACD, Black, Plum);
      ProcessAvgLim(ENVELOPE, ENVELOPE, Green, Red);  TrailingAvgLim(ENVELOPE, Green, Red);
}
```

So this was an example variant of the object approach to system building. Those
who want may use it.

And once again I would like to ask MQL developers to widen the options of the language.
And as an example, here is a variant of implementing object classes written in
the language C++.

```
struct SystemParam
{
    double TP;        // profit
    double NullTP;    // profit level, after which we set into losslessness
    double NullTP2;   // profit level, after which we set into losslessness a set of one-direction orders
    double TS;        // distance of the trailing stop
    double NullSL;    // loss level, at which we transfer the expected profit into losslessness
    double SL;        // stop-loss
    double dSL;       // a step upon the opening level of the next order for the position support
    double dStep;     // In how many times we increase the step upon the opening level of the next order
    double dLot;      // In how many times we increase the lot on the next order
}


class MTS
{
    public:
    string m_NameTS;    // system name (for making comments for the order)
    int m_SignalID;     // identifier of trading signals (for semaphore inquiry)

    long int Tickets[1000];    // array of order tickets, selected upon m_SignalID (MAGIC)

    SystemParam SysPar;    // Trading system parameters
    color ClrBuy;         // color for indicating BUY order
    color ClrSell;        // color for indicating SELL order

    // Initialization
    void MyMTS ();            // standard function that sets initial values of the system
    void MyMTS (int aSignalID, int nProcessMode, int nTrailingMode); // standard function
                                    // that sets initial values of the system


    // Implementation
    int CheckSignal();     //function of checking state of market signals

    // Processing
    int m_nProcessMode;          // identifier of observation mode
    int m_nTrailingMode;         // identifier of trailing mode
    void Process();         // EE function - processing CheckSignal()
    void Trailing();        // PS function - order trailing

    // Special functions
    bool CreatTicketArray(int dir);    // creating an array of tickets, selected upon m_SignalID (MAGIC)
                    // and desired type dir: buy, sell, buylim, buystop, sellim, sellstop
    bool ArrangeOrderBy(int iSort);  // arranging array Tickets upon the parameter (date, profit, price...)

};

…

MTS MyTS; // our trading system
…

int init()
{
…
    MyTS.m_SignalID = SIGNAL_MACD; // our system is based on MACD signals
    MyTS.m_NameTS = "MACD";
    MyTS.SysPar.TP = 500;
    MyTS.SysPar.NullTP = 20;
    MyTS.SysPar.TS = 50;
    MyTS.SysPar.SL = 1000;

    MyTS.SetProcess (MODE_AVGLIM);
    MyTS.SetTrailing (MODE_AVGLIM);
…
}

void start()
{
…
    MyTS.Process ();
    MyTS.Trailing ();
…
}

…

void MTS::Process()
{
…
    int Signal = CheckSignal(true, m_SignalID); //calling the global function of signal processing
    if (Signal == -1) return; // if no signal, do nothing

//----- for buying
    if(Signal == OP_BUY)
    {
    }

    if(Signal == OP_SELL)
    {
    }
…
}

…
// global processor of semaphores

int CheckSignal(bool bEntry, int SignalID)
{
    switch (SignalID)
    {
        case ELDER:    return (Elder(bEntry)); break;
        case ENVELOP:    return (Envelop(bEntry)); break;
        case LAGUER:    return (Laguer(bEntry)); break;
        case MACD:    return (Macd(bEntry)); break;
        …
    }
}

// calling a certain semaphore
int Macd(bool bEntry)
{
    double MACDOpen=3;
    double MACDClose=2;
    double MA=26;
    int MODE_MA    = MODE_EMA; // method of the calculation of averages
    int PRICE_MA   = PRICE_CLOSE; // method of the calculation of averages
    int PERIOD     = PERIOD_H1; // the period to work with

    //parameters of averages
    double MacdCur, MacdPre, SignalCur;
    double SignalPre, MaCur, MaPre;

    //---- get the value
    MacdCur=iMACD(NULL,0,8,17,9,PRICE_MA,MODE_MAIN,0);   MacdPre=iMACD(NULL,0,8,17,9,PRICE_MA,MODE_MAIN,1);
    SignalCur=iMACD(NULL,0,8,17,9,PRICE_MA,MODE_SIGNAL,0);   SignalPre=iMACD(NULL,0,8,17,9,PRICE_MA,MODE_SIGNAL,1);
    MaCur=iMA(NULL,0,MA,0,MODE_MA,PRICE_MA,0);   MaPre=iMA(NULL,0,MA,0,MODE_MA,PRICE_MA,1);

    //----- condition for the operation execution
    if (bEntry)   //for buying bEntry==true
    {
        if(MacdCur<0 && MacdCur>SignalCur && MacdPre<SignalPre && MathAbs(MacdCur)>(MACDOpen*Point) && MaCur>MaPre)
         return (OP_BUY);
        if(MacdCur>0 && MacdCur<SignalCur && MacdPre>SignalPre && MacdCur>(MACDOpen*Point) && MaCur<MaPre)
         return (OP_SELL);
    }
    else //for closing bEntry==false
    {
        if(MacdCur>0 && MacdCur<SignalCur && MacdPre>SignalPre && MacdCur>(MACDClose*Point))
         return (OP_BUY);
        if(MacdCur>0 && MacdCur<SignalCur && MacdPre>SignalPre && MacdCur>(MACDOpen*Point) && MaCur<MaPre)
         return (OP_BUY);

        if(MacdCur<0 && MacdCur>SignalCur && MacdPre<SignalPre && MathAbs(MacdCur)>(MACDClose*Point))
         return (OP_SELL);
        if(MacdCur<0 && MacdCur>SignalCur && MacdPre<SignalPre && MathAbs(MacdCur)>(MACDOpen*Point) && MaCur>MaPre)
         return (OP_SELL);
    }

    return (-1); //no signal
}
```

The system logics in the MQL language will not be much different. All the functions
become global. And for differentiating orders of one trading system from the orders
of another one, we will need to add the parameter SignalID (i.e. MAGIC) into all
functions that work with orders.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1499](https://www.mql5.com/ru/articles/1499)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1499.zip "Download all attachments in the single ZIP archive")

[Signals.mqh](https://www.mql5.com/en/articles/download/1499/Signals.mqh "Download Signals.mqh")(30.61 KB)

[TradeSystem.mq4](https://www.mql5.com/en/articles/download/1499/TradeSystem.mq4 "Download TradeSystem.mq4")(17.78 KB)

[Traling.mqh](https://www.mql5.com/en/articles/download/1499/Traling.mqh "Download Traling.mqh")(21.5 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)
- [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)
- [Getting Rid of Self-Made DLLs](https://www.mql5.com/en/articles/364)
- [Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)
- [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276)
- [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)
- [The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39385)**
(5)


![LesioS](https://c.mql5.com/avatar/avatar_na2.png)

**[LesioS](https://www.mql5.com/en/users/lesios)**
\|
23 Oct 2007 at 14:19

"And once again I would like to ask MQL developers to widen the options of
the language. And as an example, here is a variant of implementing object classes
written in the language C++."

I also hope that MQL5 will have some of this features. If events occur also....

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
6 Jan 2008 at 17:39

Can you attach the LaGuerre.mq4? I download the LaGuerre indicator from this
site. It get a error "'LaGuerre' - [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") is not defined ".


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
1 Mar 2008 at 17:02

I presume that the .zip file here contains the Laquerre file.  However, I can't download it.  A square symbol appears next to the file.  Perhaps this is an indication that the file is protected from downloading.  Any help in resolving this will be appreciated.


![庄继军](https://c.mql5.com/avatar/avatar_na2.png)

**[庄继军](https://www.mql5.com/en/users/puti1)**
\|
8 Jun 2008 at 03:06

Laguerre.mq4    ??????????????????????????


![---](https://c.mql5.com/avatar/avatar_na2.png)

**[\-\-\-](https://www.mql5.com/en/users/sergeev)**
\|
24 Jun 2008 at 10:58

Laguerre.mq4   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

![Terminal Service Client. How to Make Pocket PC a Big Brother's Friend](https://c.mql5.com/2/14/315_5.png)[Terminal Service Client. How to Make Pocket PC a Big Brother's Friend](https://www.mql5.com/en/articles/1458)

The article describes the way of connecting to the remote PC with installed MT4 Client Terminal via a PDA.

![Trading Strategy Based on Pivot Points Analysis](https://c.mql5.com/2/14/332_13.png)[Trading Strategy Based on Pivot Points Analysis](https://www.mql5.com/en/articles/1465)

Pivot Points (PP) analysis is one of the simplest and most effective strategies for high intraday volatility markets. It was used as early as in the precomputer times, when traders working at stocks could not use any ADP equipment, except for counting frames and arithmometers.

![MQL4 Language for Newbies. Custom Indicators (Part 1)](https://c.mql5.com/2/15/516_15.gif)[MQL4 Language for Newbies. Custom Indicators (Part 1)](https://www.mql5.com/en/articles/1500)

This is the fourth article from the series "MQL4 Languages for Newbies". Today we will learn to write custom indicators. We will get acquainted with the classification of indicator features, will see how these features influence the indicator, will learn about new functions and optimization, and, finally, we will write our own indicators. Moreover, at the end of the article you will find advice on the programming style. If this is the first article "for newbies" that you are reading, perhaps it would be better for you to read the previous ones. Besides, make sure that you have understood properly the previous material, because the given article does not explain the basics.

![Automated Choice of Brokerage Company for an Efficient Operation of Expert Advisors](https://c.mql5.com/2/14/367_27.png)[Automated Choice of Brokerage Company for an Efficient Operation of Expert Advisors](https://www.mql5.com/en/articles/1476)

It is not a secret that for an efficient operation of Expert Advisors we need to find a suitable brokerage company. This article describes a system approach to this search. You will get acquainted with the process of creating a program with dll for working with different terminals.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/1499&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083237105338947469)

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