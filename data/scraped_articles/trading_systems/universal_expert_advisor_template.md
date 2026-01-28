---
title: Universal Expert Advisor Template
url: https://www.mql5.com/en/articles/1495
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:03:13.335494
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/1495&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071537339546675680)

MetaTrader 4 / Trading systems


### Introduction

Many traders face the problem of writing their own Expert Advisors. What comes first? How to set in an EA code take-profit, stop-loss or trailing-stop? How to check the functionality of a strategy? In this article we will dwell on the main functions for creating Expert Advisors. Perhaps someone will find the trailing code useful.

### Expert Advisor Variables

So what variables are necessary in every Expert Advisor? What should be done to have a tester go through also the parameters, set by bool type variables? First, I will present a general EA code, which works on all currency pairs; for higher speed it is recommended to adjust them according to your own currency type. Second, 0-disabled, 1-enabled (because we want all the parameters to be checked in an optimizer. If a final code is rarely optimized, it is recommended to change this parameter). Third, variables of levels (stops) are initialized by integers, and in a code, where necessary, are changed into fractional numbers. While in real life we trade on different currencies and here is a template! Forth, if the stop parameters are equal to zero, then stop does not work.

### Defining Variables

Now let us start defining variables. Let us start with those, liable to optimization - external variables.

```
extern double MaxLot;
extern double TakeProfit;
extern double TrailingStop;
extern double StopLoss;
extern double MinProfit;
extern double ProfitPoints;
extern int    Slippage=0;
extern int Condition1=1;
extern double LotSpliter=1.0;
extern int CloseByOtherSideCondition;
```

The variable MaxLot sets the maximal lot, when we want to limit the maximally used lot (the lot is also limited by a server, but this will be discussed later).

TakeProfit, StopLoss and TrailingStop operate in our code, when they are larger than zero.

MinProfit and ProfitPoints operate, when ProfitPoints are higher than zero, according to the principle: a price reaches ProfitPoints and turns back till MinProfit.

ConditionX enables the entry conditions.

LotSpliter is a splitter of lots. It uses only part of available lots, for example 0.1 includes only lots, 10 times smaller than the available rate for the whole deposit.

CloseByOtherSideCondition closes an order at the appearance of an opposite side condition.

Let us set internal variables, which will be discussed along with the EA description.

```
double Lot;
double PP=0;
double slu,sld,a,b;
double tp,sl;
```

### Initialization Code

Now let us see, what can be calculated only by starting an Expert Advisor and used further in a code.

```
int init()
  {
   tp=TakeProfit;
   sl=StopLoss;
   return(0);
  }
```

We take the values of these variables for the case of stop-levels changing. We can also calculate a lot, if we are going to trade all the time with the same volume, and display the excess (later in the article the lot calculation is analyzed). We can also create a displayable comment, containing an EA description or copyright. It is done this way:

```
Comment("cloud trade \n v2.0.11");
```

"\\n" - means shift to the next display line, "carriage return".

### Framework Code

Now let us view a code, when there are no orders:

```
if(OrdersTotal()==0)
   {
      preinit();
      if(U()==1)
      {
         OrderBuy();
         return(0);
      }
      if(U()==2)
      {
         OrderSell();
         return(0);
      }
      return(0);
   }
```

Part of these functions will be analyzed later. Here is the principle: initialize parameters, check if there are entry conditions, enter by a condition.

Now let us view a code, when there is one order:

```
if(OrderType()==OP_BUY)
        {
         if((slu)>PP)
           {
            PP=slu;
           }
         if(((slu)>0.001) && (OrderStopLoss()<(b-TrailingStop))
          && (OrderOpenPrice()<(b-TrailingStop))
           && (OrderProfit()>MathAbs(OrderSwap())))
           {
            if(TrailingStop!=0)
              {
               OrderModify(OrderTicket(), 0, b-TrailingStop, 0, 0, 0);
              }
           }
        }
      if(OrderType()==OP_SELL)
        {
         if((sld)>PP)
           {
            PP=sld;
           }
         if(((sld)>0.001) && (OrderStopLoss()>(a+TrailingStop))
          && (OrderOpenPrice()>(a+TrailingStop)))
           {
            if(TrailingStop!=0)
              {
               OrderModify(OrderTicket(), 0, a+TrailingStop, 0, 0, 0);
              }
           }
        }
      if(ProfitPoints!=0)
        {
         if(OrderType()==OP_BUY && PP>=ProfitPoints && (slu)<=MinProfit)
           {
            CloseOnlyOrder(OrderTicket());
            return(0);
           }
         if(OrderType()==OP_SELL && PP>=ProfitPoints && (sld)<=MinProfit)
           {
            CloseOnlyOrder(OrderTicket());
            return(0);
           }
        }
      if(CloseByOtherSideCondition==1)
        {
         if(OrderType()==OP_BUY && U()==2)
           {
            CloseOnlyOrder(OrderTicket());
            return(0);
           }
         if(OrderType()==OP_SELL && U()==1)
           {
            CloseOnlyOrder(OrderTicket());
            return(0);
           }
        }
```

First we choose the only one order for further actions on it (this code will be analyzed in a separate article part). Then we assign variables to prices, in order not to change an order, for example, at a trailing-stop in the wrong direction or at unprofitable new prices. First we check an order for the probability of a trailing-stop, and at the same time gathering data for the next function - minimal profit, the usage of which was described earlier. Then goes the function of closing an order at the appearance of an opposite side condition and opening an order in the opposite direction.

### Functions, Analyzed in the Article

Now let us view the functions, intended for shortening a code and incorporate into blocks the most frequently used commands, so that later full blocks could be called. Let us try to set such conditions and check them:

```
//+------------------------------------------------------------------+
//|  returns a signal to buy or to sell                              |
//+------------------------------------------------------------------+
int U()
  {
      if((U1()==2 && Condition1==1)
       || (U2()==2 && Condition2==1)){return(2);}
      if((U1()==1 && Condition1==1)
       || (U2()==1 && Condition2==1)){return(1);}
   return(0);
  }
//+------------------------------------------------------------------+
//|  returns a signal based on stochastic values                     |
//+------------------------------------------------------------------+
int U1()
  {
   if(iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing, Method,PriceUsing,MODE_SIGNAL,1)>=80)
     {
      if(iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing,Method,PriceUsing,MODE_SIGNAL,2)
           <=iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing, Method,PriceUsing,MODE_MAIN,2))
        {
         if(iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing,Method,PriceUsing,MODE_SIGNAL,1)
           >=iStochastic(Symbol(),Period(),
              Kperiod,Dperiod,Slowing,Method,PriceUsing,MODE_MAIN,1))
           {
            return(2);
           }
        }
     }
   if(iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing,Method,PriceUsing,MODE_SIGNAL,1)<=20)
     {
      if(iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing,Method,PriceUsing,MODE_SIGNAL,2)
           >=iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing,Method,PriceUsing,MODE_MAIN,2))
        {
         if(iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing, Method,PriceUsing,MODE_SIGNAL,1)
              <=iStochastic(Symbol(),Period(),Kperiod,Dperiod,Slowing,Method,PriceUsing,MODE_MAIN,1))
           {
            return(1);
           }
        }
     }
   return(0);
  }
//+------------------------------------------------------------------+
//| find trend direction using fractals                              |
//+------------------------------------------------------------------+
int U2()
  {
   double fu=0,fd=0;
   int f=0,shift=2;
   while(f<2)
     {
      if(iFractals(Symbol(),Period(),MODE_UPPER,shift)>0)
        {
         fu=fu+1;
         f=f+1;
        }
      if(iFractals(Symbol(),Period(),MODE_LOWER,shift)>0)
        {
         fd=fd+1;
         f=f+1;
        }
      shift=shift+1;
     }
   if(fu==2){return(2);}
   if(fd==2){return(1);}
   return(0);
  }
```

The first function checks conditions, the next two ones set conditions.

Now let us view the function, calculating stop levels, if they are set wrong, and defines a lot value:

```
//+------------------------------------------------------------------+
//| preliminary initialization of variables                          |
//+------------------------------------------------------------------+
int preinit()
  {
   Lot=NormalizeDouble(MathFloor(LotSpliter*AccountBalance()*AccountLeverage()
      /Ask/MathPow(10,Digits+1)*10)/10,1);
   if(MaxLot>0 && Lot>MaxLot){Lot=MaxLot;}
   if(Lot>MarketInfo(Symbol(),MODE_MAXLOT)){Lot=MarketInfo(Symbol(),MODE_MAXLOT);}
   PP=0;
   StopLoss=sl;
   TakeProfit=tp;
   if(TakeProfit!=0 && TakeProfit<(MarketInfo(Symbol(),MODE_STOPLEVEL)))
     {
      TakeProfit=MarketInfo(Symbol(),MODE_STOPLEVEL);
     }
   if(StopLoss!=0 && StopLoss<(MarketInfo(Symbol(),MODE_STOPLEVEL)))
     {
      StopLoss=MarketInfo(Symbol(),MODE_STOPLEVEL);
     }
   return(0);
  }
```

Now set functions, opening orders depending on the preset stop levels:

```
//+------------------------------------------------------------------+
//| returns true in case of a successful opening of Buy              |
//+------------------------------------------------------------------+
bool OrderBuy()
  {
   bool res=false;
   if(StopLoss!=0 && TakeProfit!=0)
     {
      res=OrderSend(Symbol(), 0, NormalizeDouble(Lot,1), Ask, Slippage,
       NormalizeDouble(Ask-StopLoss,4),
        NormalizeDouble(Ask+TakeProfit,4), 0, 0, 0, 0);
      return(res);
     }
   if(StopLoss==0 && TakeProfit!=0)
     {
      res=OrderSend(Symbol(), 0, NormalizeDouble(Lot,1), Ask, Slippage, 0,
       NormalizeDouble(Ask+TakeProfit,4), 0, 0, 0, 0);
      return(res);
     }
   if(StopLoss==0 && TakeProfit==0)
     {
      res=OrderSend(Symbol(), 0, NormalizeDouble(Lot,1), Ask,
       Slippage, 0, 0, 0, 0, 0, 0);
      return(res);
     }
   if(StopLoss!=0 && TakeProfit==0)
     {
      res=OrderSend(Symbol(), 0, NormalizeDouble(Lot,1), Ask, Slippage,
       NormalizeDouble(Ask-StopLoss,4), 0, 0, 0, 0, 0);
      return(res);
     }
   return(res);
  }
//+------------------------------------------------------------------+
//|   returns true in case of a successful opening of Sell           |
//+------------------------------------------------------------------+
bool OrderSell()
  {
   bool res=false;
   if(StopLoss!=0 && TakeProfit!=0)
     {
      res=OrderSend(Symbol(), OP_SELL, NormalizeDouble(Lot,1), Bid, Slippage,
       NormalizeDouble(Bid+StopLoss,4),
        NormalizeDouble(Bid-TakeProfit,4), 0, 0, 0, 0);
      return(res);
     }
   if(StopLoss==0 && TakeProfit!=0)
     {
      res=OrderSend(Symbol(), OP_SELL, NormalizeDouble(Lot,1), Bid, Slippage,
       NormalizeDouble(Bid+StopLoss,4),
        NormalizeDouble(Bid-TakeProfit,4), 0, 0, 0, 0);
      return(res);
     }
   if(StopLoss==0 && TakeProfit==0)
     {
      res=OrderSend(Symbol(), OP_SELL, NormalizeDouble(Lot,1), Bid, Slippage,
       NormalizeDouble(Bid+StopLoss,4),
        NormalizeDouble(Bid-TakeProfit,4), 0, 0, 0, 0);
      return(res);
     }
   if(StopLoss!=0 && TakeProfit==0)
     {
      res=OrderSend(Symbol(), OP_SELL, NormalizeDouble(Lot,1), Bid, Slippage,
       NormalizeDouble(Bid+StopLoss,4),
        NormalizeDouble(Bid-TakeProfit,4), 0, 0, 0, 0);
      return(res);
     }
   return(res);
  }
```

The next function closes an order with an indicated ticket, volume and at an indicated price:

```
//+-------------------------------------------------------------------------+
//|  returns true in case of a successful closing of an order with Ticket   |
//+-------------------------------------------------------------------------+
bool CloseOnlyOrder(int Ticket, double Lots ,double priceClose)
  {
   bool res=false;
   res=OrderClose(Ticket, Lots, priceClose, Slippage, 0);
   return(res);
```

Now let us view the function of choosing an order upon the position number for further operation on them:

```
//+--------------------------------------------------------------------------------+
//| returns true in case of a successful choosing of an order in the position pos  |
//+--------------------------------------------------------------------------------+
bool SelectOnlyOrder(int pos)
  {
   bool res=false;
   res=OrderSelect(pos,SELECT_BY_POS,MODE_TRADES);
   return(res);
  }
//+------------------------------------------------------------------+
```

### Some Coding Recommendations

First, set options like 0 and 1 instead of true and false. This will help you better optimize your Expert Advisor. Second, do not neglect stop-loss for limiting possible losses, when the market moves in the direction, opposite to conditions. Third, do not test experts without stop-loss - it is likely to lead to a quick deposit loss. Forth, use functions and blocks, which help to make the code understanding easier.

### Conclusion

It is easy to create Expert Advisors. And to make this task even easier, the attached file contains the Expert Advisor, analyzed in this article.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1495](https://www.mql5.com/ru/articles/1495)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1495.zip "Download all attachments in the single ZIP archive")

[template.mq4](https://www.mql5.com/en/articles/download/1495/template.mq4 "Download template.mq4")(8.07 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39363)**
(2)


![Mizan Sharif](https://c.mql5.com/avatar/2014/5/537F70ED-82EF.jpg)

**[Mizan Sharif](https://www.mql5.com/en/users/mrs)**
\|
20 Jun 2014 at 02:31

Excellent Template. Not easily found. Expecting some [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") for buy/sell conditions.

Lack of Magic number.

Some comments are in a unknown language do not make any sense to me (greek to me):

1\. âîçâðàùàåò ñèãíàë íà ïîêóïêó èëè ïðîäàæó

2\. âîçâðàùàåò ñèãíàë ïîêóïêè(1) èëè ïðîäàæè (2)

3\. Âîçâðàò ñ óñëîâèåì ïðîäàæè

4\. Âîçâðàò ñ óñëîâèåì ïîêóïêè

Mizan Sharif

![mimja](https://c.mql5.com/avatar/avatar_na2.png)

**[mimja](https://www.mql5.com/en/users/mimja)**
\|
1 Aug 2019 at 11:14

Hi, where i can put my condition for buy and sell and closing them? is it true?

[![](https://c.mql5.com/3/287/image__6.png)](https://c.mql5.com/3/287/image__5.png "https://c.mql5.com/3/287/image__5.png")

![How Not to Fall into Optimization Traps?](https://c.mql5.com/2/14/218_2.png)[How Not to Fall into Optimization Traps?](https://www.mql5.com/en/articles/1434)

The article describes the methods of how to understand the tester optimization results better. It also gives some tips that help to avoid "harmful optimization".

![Principles of Time Transformation in Intraday Trading](https://c.mql5.com/2/14/307_4.gif)[Principles of Time Transformation in Intraday Trading](https://www.mql5.com/en/articles/1455)

This article contains the concept of operation time that allows to receive more even price flow. It also contains the code of the changed moving average with an allowance for this time transformation.

![Filtering by History](https://c.mql5.com/2/14/244_1.png)[Filtering by History](https://www.mql5.com/en/articles/1441)

The article describes the usage of virtual trading as an integral part of trade opening filter.

![How To Implement Your Own Optimization Criteria](https://c.mql5.com/2/14/460_29.jpg)[How To Implement Your Own Optimization Criteria](https://www.mql5.com/en/articles/1498)

In this article an example of optimization by profit/drawdown criterion with results returned into a file is developed for a standard Expert Advisor - Moving Average.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/1495&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071537339546675680)

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