---
title: Comfortable Scalping
url: https://www.mql5.com/en/articles/1509
categories: Trading
relevance_score: 1
scraped_at: 2026-01-23T21:37:09.257720
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/1509&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071968932220318164)

MetaTrader 4 / Trading


### Introduction

This article describes an algorithm of trade opening that allows to make scalping more comfortable. However, this algorithm can also be applied in other trading approaches. Actually, the article offers a method of helping a trader at such a fast trading.

Generally scalping is considered a very nervous type of trading. Very important here is the necessity to indicate lot, TakeProfit and StopLoss levels each time, thus being distracted from a chart.

This article is the continuation of " **[Betting Modeling as Means of Developing "Market Intuition"](https://www.mql5.com/en/articles/1505)**". I recommend reading it before starting to study the present article.

I'd like to remind you of what scalping is. Scalping is a method of quick trading. Usually in such a trading profit is fixed at 1-10 pips (points). Scalping is known for its complexity, nervousness and higher attentiveness required. Someone thinks it not serious, someone considers it a perfect mastery. As for me, I am not going to estimate this type of trading - it is widely discussed and everyone has one's own opinion.

### Concept

Probably every trader ever tried to use a scalping strategy. For some traders scalping is the most convenient type of trading, for others - on the contrary. Some consider scalping the most interesting trading, others - a mere waste of time. However everyone can note about the necessity of a higher attention to market and opened trades in this type of trading.

Many traders decline using scalping just because it requires much effort and nerves. However, there is a method to help a scalper.

Suppose a trader is going to use scalping with a fixed lot and take profit at each trade. Obviously, it is reasonable to eliminate the necessity of indicating these parameters every time for each trade. Because it takes extra time and draws a trader's attention from a chart.

It means we need a tool that will open a trade with a fixed lot and TP/SL levels on a trader's command. The tool's operation should be maximally simple; besides it should minimally distract a trader from a chart.

Such a tool can be easily implemented using MQL4 means.

### Implementation

As the basis we will take a game described in the article " **[Betting Modeling as Means of developing "Market Intuition"](https://www.mql5.com/en/articles/1505)**". We will create a tool, which will help to play this game and trade at the same time.

Short description of the game. Two arrows are drawn on the chart - up and down. A trader deletes an unnecessary arrow, thus making a choice denoting his opinion - whether a security will rise or fall. At the beginning of a new candlestick the EA checks whether a trader's forecast is right or wrong. The correctness of forecasting influences the game score. Moreover, a trader can make his choice within a limited period of time, which can be changed (a trader decides whether to set it or not).

For the implementation we will draw two more arrows one bar back than the current one. The current bar will be still used for the betting. Deletion of an arrow on the previous bar will be a signal for the EA to open a trade in the necessary direction. Besides, limitation of time period for choosing a trade direction will be disabled for trading. There will be the following changeable parameters: TakeProfit and StopLoss levels, lot, acceptable slippage and the magic number. Besides, trading can be disabled using the extern bool variable, thus the EA will be used only for betting.

Besides, at trade opening an arrow named 'buy' or 'sell' will be drawn on a chart, depending on a trade open at the time. This will be done to prevent the EA from opening new trades on this candlestick. This arrow will be drawn 300 points away from a bar opening price, so a user won't probably even notice it.

The EA itself will be divided into two blocks - the game and trade opening. Thus a reader can see what is added to the code.

So, we have the following program code:

```
//+------------------------------------------------------------------+
//|                                                       trener.mq4 |
//|                                       Copyright © 2008, FXRaider |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2008, FXRaider"
extern int gap=2;
extern bool Trading=true;
extern int TP=2;
extern int SL=20;
extern double Lots=0.02;
extern int slippage=1;
extern int MagicNumber=777;
extern int time_limit=30;
int start()
  {
//----
//#################################################################################
//####################################### GAME ####################################
//------------------------------
string solution="none";
int point,
    point_neg,
    point_pos;
//------------------------------
//+---------------------------------------------------------------+
//|                      "up" choice searching                    |
 if(
    ObjectGet("up", OBJPROP_PRICE1)==Open[1]+gap*Point
    &&iBarShift(NULL,0,ObjectGet("up",OBJPROP_TIME1))==1
    &&ObjectFind("down") != 0
    &&ObjectFind("up") == 0
    )
    {
     solution="up";
    }
//|                      "up" choice searching                    |
//+---------------------------------------------------------------+

//+---------------------------------------------------------------+
//|                      "down" choice searching                  |
 if(
    ObjectGet("down", OBJPROP_PRICE1)==Open[1]-gap*Point
    &&iBarShift(NULL,0,ObjectGet("down",OBJPROP_TIME1))==1
    &&ObjectFind("up") != 0
    &&ObjectFind("down") == 0
    )
    {
     solution="down";
    }
//|                      "down" choice searching                  |
//+---------------------------------------------------------------+

//+---------------------------------------------------------------+
//|             counting points at a positive answer              |
    if((solution=="up"&&Open[1]<Close[1])
      ||(solution=="down"&&Open[1]>Close[1]))
    {
     point=1;
     point_pos=1;
     point_neg=0;
    }
//|             counting points at a positive answer              |
//+---------------------------------------------------------------+

//+---------------------------------------------------------------+
//|             counting points at a negative answer              |
    if((solution=="up"&&Open[1]>Close[1])
      ||(solution=="down"&&Open[1]<Close[1]))
    {
     point=-1;
     point_pos=0;
     point_neg=1;
    }
//|             counting points at a negative answer              |
//+---------------------------------------------------------------+

//+----------------------------------------------------------------------------------+
//|                              working with an external file                       |
      int handle;
      double points,     //total score
             points_pos, //score of positive answers
             points_neg; //score of negative answers
       handle=FileOpen("trener_"+Symbol()+"_"+Period()+".csv",
                       FILE_CSV|FILE_WRITE|FILE_READ,";");
       if(handle>0) //if there is a file, read it
       {
        points=NormalizeDouble(StrToDouble(FileReadString(handle)),Digits);
        points_pos=NormalizeDouble(StrToDouble(FileReadString(handle)),Digits);
        points_neg=NormalizeDouble(StrToDouble(FileReadString(handle)),Digits);
        FileClose(handle);
       }

    if(solution!="none") //if a choice has been made made
    {
      handle=FileOpen("trener_"+Symbol()+"_"+Period()+".csv",
                      FILE_CSV|FILE_WRITE|FILE_READ,";");
      FileWrite(handle ,points+point);         //write the total score
      FileWrite(handle ,points_pos+point_pos); //write the score of positive answers
      FileWrite(handle ,points_neg+point_neg); //write the score of negative answers
      FileClose(handle);
    }
//|                              working with an external file                       |
//+----------------------------------------------------------------------------------+

//+------------------------------------------------------------------------------------+
//|                                 working with objects                               |
  if(iBarShift(NULL,0,ObjectGet("down",OBJPROP_TIME1))>0
     ||ObjectGet("down",OBJPROP_PRICE1)!=Open[0]-gap*Point)
    {
     ObjectDelete("down");
    }
 if(iBarShift(NULL,0,ObjectGet("up",OBJPROP_TIME1))>0
    ||ObjectGet("up",OBJPROP_PRICE1)!=Open[0]+gap*Point)
    {
     ObjectDelete("up");
    }

  int sec_lim;
  if(!time_limit)
  {
   sec_lim=0;
  }
  else
  {
   sec_lim=TimeCurrent()-time_limit;
  }
  if(sec_lim>ObjectGet("up",OBJPROP_TIME1)
     &&sec_lim>ObjectGet("down",OBJPROP_TIME1)
     &&ObjectFind("down") == 0&&ObjectFind("up") == 0
     &&iBarShift(NULL,0,ObjectGet("down",OBJPROP_TIME1))==0
     &&iBarShift(NULL,0,ObjectGet("up",OBJPROP_TIME1))==0)
    {
     ObjectDelete("up");
     ObjectDelete("down");
    }

   if((ObjectFind("down") != 0&&ObjectFind("up") != 0) //if no object
      &&sec_lim<Time[0])
   {
     ObjectCreate("down", OBJ_ARROW, 0, Time[0], Open[0]-gap*Point); //draw a down arrow
     ObjectSet("down", OBJPROP_STYLE, STYLE_DOT);
     ObjectSet("down", OBJPROP_ARROWCODE, SYMBOL_ARROWDOWN);
     ObjectSet("down", OBJPROP_COLOR, Red);

     ObjectCreate("up", OBJ_ARROW, 0, Time[0], Open[0]+gap*Point); //draw an up arrow
     ObjectSet("up", OBJPROP_STYLE, STYLE_DOT);
     ObjectSet("up", OBJPROP_ARROWCODE, SYMBOL_ARROWUP);
     ObjectSet("up", OBJPROP_COLOR, Blue);
    }
//|                                 working with objects                               |
//+------------------------------------------------------------------------------------+
//####################################### GAME ####################################
//#################################################################################


//#################################################################################
//#################################### TRADING ####################################
//+------------------------------------------------------------------------------------+
//|                                working with objects I                              |
  if(iBarShift(NULL,0,ObjectGet("down_1",OBJPROP_TIME1))>1
  ||ObjectGet("down_1",OBJPROP_PRICE1)!=Open[0]-gap*Point
  ||!Trading)
  {
   ObjectDelete("down_1");
  }

  if(iBarShift(NULL,0,ObjectGet("up_1",OBJPROP_TIME1))>1
  ||ObjectGet("up_1",OBJPROP_PRICE1)!=Open[0]+gap*Point
  ||!Trading)
  {
   ObjectDelete("up_1");
  }

  if(iBarShift(NULL,0,ObjectGet("sell",OBJPROP_TIME1))>0
  ||ObjectGet("sell",OBJPROP_PRICE1)!=Open[0]-300*Point
  ||!Trading)
  {
   ObjectDelete("sell");
  }
  if(iBarShift(NULL,0,ObjectGet("buy",OBJPROP_TIME1))>0
  ||ObjectGet("buy",OBJPROP_PRICE1)!=Open[0]+300*Point
  ||!Trading)

  {
   ObjectDelete("buy");
  }

   if(ObjectFind("down_1") != 0&&ObjectFind("up_1") != 0 && Trading)
   {
     ObjectCreate("down_1", OBJ_ARROW, 0, Time[1], Open[0]-gap*Point);
     ObjectSet("down_1", OBJPROP_STYLE, STYLE_DOT);
     ObjectSet("down_1", OBJPROP_ARROWCODE, SYMBOL_ARROWDOWN);
     ObjectSet("down_1", OBJPROP_COLOR, Red);

     ObjectCreate("up_1", OBJ_ARROW, 0, Time[1], Open[0]+gap*Point);
     ObjectSet("up_1", OBJPROP_STYLE, STYLE_DOT);
     ObjectSet("up_1", OBJPROP_ARROWCODE, SYMBOL_ARROWUP);
     ObjectSet("up_1", OBJPROP_COLOR, Blue);
    }
//|                                working with objects I                              |
//+------------------------------------------------------------------------------------+
 if(Trading)
 {
//+----------------------------------------------------------------------------------------------+
//|                              searching open orders for a security                            |
    int pos_sell=0, bar_op_buy, bar_op_sell;
  for (int i_op_sell=OrdersTotal()-1; i_op_sell>=0; i_op_sell--)
  {
   if (!OrderSelect(i_op_sell,SELECT_BY_POS,MODE_TRADES)) break;
   if (Symbol()==OrderSymbol()
   &&(OrderType()==OP_SELLSTOP||OrderType()==OP_SELL)
   &&(OrderMagicNumber()==MagicNumber)
   &&iBarShift(NULL,0,OrderOpenTime())==0)
   {
    pos_sell=1; break;
   }
  }

    int pos_buy=0;
  for (int i_op_buy=OrdersTotal()-1; i_op_buy>=0; i_op_buy--)
  {
   if (!OrderSelect(i_op_buy,SELECT_BY_POS,MODE_TRADES)) break;
   if (Symbol()==OrderSymbol()
   &&(OrderType()==OP_BUYSTOP||OrderType()==OP_BUY)
   &&(OrderMagicNumber()==MagicNumber)
   &&iBarShift(NULL,0,OrderOpenTime())==0)
   {
    pos_buy=1; break;
   }
  }
//|                              searching open orders for a security                            |
//+----------------------------------------------------------------------------------------------+

//+------------------------------------------------------------------------------------+
//|                                working with objects II                             |
 if(pos_buy==1)
 {
      ObjectCreate("buy", OBJ_ARROW, 0, Time[0], Open[0]+300*Point);
      ObjectSet("buy", OBJPROP_STYLE, STYLE_DOT);
      ObjectSet("buy", OBJPROP_ARROWCODE, SYMBOL_ARROWUP);
      ObjectSet("buy", OBJPROP_COLOR, Red);
 }

 if(pos_sell==1)
 {
      ObjectCreate("sell", OBJ_ARROW, 0, Time[0], Open[0]-300*Point);
      ObjectSet("sell", OBJPROP_STYLE, STYLE_DOT);
      ObjectSet("sell", OBJPROP_ARROWCODE, SYMBOL_ARROWDOWN);
      ObjectSet("sell", OBJPROP_COLOR, Red);
 }
//|                                working with objects II                             |
//+------------------------------------------------------------------------------------+

//+------------------------------------------------------------------------------------+
//|                                   opening trades                                   |
double sl_buy, sl_sell;
 if(!SL)
 {
  sl_buy=0;
  sl_sell=0;
 }
 else
 {
  sl_buy=Ask-SL*Point;
  sl_sell=Bid+SL*Point;
 }
  if(
     ObjectGet("up_1", OBJPROP_PRICE1)==Open[0]+gap*Point
     &&iBarShift(NULL,0,ObjectGet("up_1",OBJPROP_TIME1))==1
     &&ObjectFind("down_1") != 0
     &&ObjectFind("up_1") == 0
     &&!pos_buy
     &&ObjectFind("buy") != 0
     )
     {
      OrderSend(Symbol(),OP_BUY, Lots,Ask,slippage,sl_buy,Ask+TP*Point,"trener",MagicNumber,0,Blue);
     }
  if(
     ObjectGet("down_1", OBJPROP_PRICE1)==Open[0]-gap*Point
     &&iBarShift(NULL,0,ObjectGet("down_1",OBJPROP_TIME1))==1
     &&ObjectFind("up_1") != 0
     &&ObjectFind("down_1") == 0
     &&!pos_sell
     &&ObjectFind("sell") != 0
     )
     {
      OrderSend(Symbol(),OP_SELL, Lots,Bid,slippage,sl_sell,Bid-TP*Point,"trener",MagicNumber,0,Red);
     }
//|                                   opening trades                                   |
//+------------------------------------------------------------------------------------+
 }
//#################################### TRADING ####################################
//#################################################################################

Comment("Score: ", points," (",points_pos,"/",points_neg,   //displaying score
        ") | Time: ", Hour(),":", Minute(),":", Seconds()); //displaying time (for convenience)

   return(0);
  }
//+------------------------------------------------------------------+
```

The code includes all the necessary comments.

After the Expert Advisor is attached to a chart, we will get the following:

![](https://c.mql5.com/2/21/1__1.gif)

Here the last two arrows are intended for the game, the two arrows before them are used to open orders.

The deletion of an arrow on the previous candlestick will cause the execution of the OrderSend() function and a corresponding order will be opened:

![](https://c.mql5.com/2/21/2__1.gif)

Here is the tab of changing input parameters:

![](https://c.mql5.com/2/21/3__1.jpg)

The "gap" variable is responsible for the number of points equal to the distance between an arrow and the open price of a candlestick. The variable "Trading" denotes the trading function, "TP" - TakeProfit in points, "SL" - StopLoss in points. The "Lots" variable is responsible for the volume of opened positions; "slippage" denotes the admissible slippage in points that we are ready to accept. "MagicNumber" indicates the magic number which is assigned by the EA to opened positions (necessary for the EA to be able to track its "own" orders). The "time\_limit" limit variable sets the number of seconds
within which a user must make his choice. If "0" is indicated, time is not limited, i.e. choice can be made during all the period of candlestick formation.

### Conclusion

As a result we have a security for a comfortable trading using orders with standard parameters (TP, SL, Slippage, lot). This tool can be useful in any trading. However, it is the most efficiently used when a large number of trades is opened within a short period of time. For example, in scalping.

Using this program, a trader does not have to set parameters of an opened order each time. Thus the maximum of his attention is concentrated in a chart. Undoubtedly, this may help to increase the effectiveness of trading.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1509](https://www.mql5.com/ru/articles/1509)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1509.zip "Download all attachments in the single ZIP archive")

[trener.mq4](https://www.mql5.com/en/articles/download/1509/trener.mq4 "Download trener.mq4")(11.66 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Method of Determining Errors in Code by Commenting](https://www.mql5.com/en/articles/1547)
- [Trend-Hunting](https://www.mql5.com/en/articles/1515)
- [Betting Modeling as Means of Developing "Market Intuition"](https://www.mql5.com/en/articles/1505)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39456)**
(7)


![alex dahmen](https://c.mql5.com/avatar/avatar_na2.png)

**[alex dahmen](https://www.mql5.com/en/users/cooltrader)**
\|
29 Jul 2008 at 23:06

you rock !!!!!!! .. can you contact me at skype : ctzulu

i just want to discuss two of my [trading strategies](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies ") and what you think of it

thx alex

![eduardo fiuza](https://c.mql5.com/avatar/2018/2/5A96E044-95BD.jpg)

**[eduardo fiuza](https://www.mql5.com/en/users/edfiuza)**
\|
18 Oct 2008 at 01:36

Hi,

Can you add a feature to close the trades using only the mouse at chart window?

Thanks

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
26 Jan 2009 at 14:32

Hi

I have put this trener.mq4 in experts folder, comiled and attached it to my chart (smiley face appeared), but when I remove one of the candles on the 2nd last bar no trade is placed.... I have got Allow Live Trading [checked](https://www.mql5.com/en/docs/common/GetTickCount64 "MQL5 Documentation: GetTickCount64 function") and also have set the Trading value to true in the EA settings.....

any ideas how to get this to work

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
9 Aug 2010 at 18:07

**gbolla:**

Stop level must be more then 5pips to open price!!! TP=2 pips don't work --> error: invalid stop

Bolla

In some pair, not only 5 pips minimum at [stop level](https://www.metatrader5.com/en/terminal/help/trading/general_concept "User Guide: Types of orders") to make an order, but also you have to set 10 pips at minimum. I want to remove TP from OrderSend() and move it to OrderClose() instead. I hope it worked.


![ROGERIO BORGES](https://c.mql5.com/avatar/avatar_na2.png)

**[ROGERIO BORGES](https://www.mql5.com/en/users/rogeriob26)**
\|
3 Jun 2016 at 18:53

Hello

ThisEAworks withthe latest version ofMT4?

I testedonlythearrowappearsin thegraphbutdoes not openorders.

Thank you.

ROgério

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part  V)](https://c.mql5.com/2/15/600_99.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part V)](https://www.mql5.com/en/articles/1525)

In this article the author offers ways to improve trading systems described in his previous articles. The article will be interesting for traders that already have some experience of writing Expert Advisors.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part IV)](https://c.mql5.com/2/15/595_130.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part IV)](https://www.mql5.com/en/articles/1523)

In this article the author continues to analyze implementation algorithms of simplest trading systems and introduces recording of optimization results in backtesting into one html file in the form of a table. The article will be useful for beginning traders and EA writers.

![Show Must Go On, or Once Again about ZigZag](https://c.mql5.com/2/16/620_30.gif)[Show Must Go On, or Once Again about ZigZag](https://www.mql5.com/en/articles/1531)

About an obvious but still substandard method of ZigZag composition, and what it results in: the Multiframe Fractal ZigZag indicator that represents ZigZags built on three larger ons, on a single working timeframe (TF). In their turn, those larger TFs may be non-standard, too, and range from M5 to MN1.

![A Non-Trading EA Testing Indicators](https://c.mql5.com/2/16/627_11.gif)[A Non-Trading EA Testing Indicators](https://www.mql5.com/en/articles/1534)

All indicators can be divided into two groups: static indicators, the displaying of which, once shown, always remains the same in history and does not change with new incoming quotes, and dynamic indicators that display their status for the current moment only and are fully redrawn when a new price comes. The efficiency of a static indicator is directly visible on the chart. But how can we check whether a dynamic indicator works ok? This is the question the article is devoted to.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/1509&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071968932220318164)

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