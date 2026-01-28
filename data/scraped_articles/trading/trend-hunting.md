---
title: Trend-Hunting
url: https://www.mql5.com/en/articles/1515
categories: Trading
relevance_score: 1
scraped_at: 2026-01-23T21:37:00.198763
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/1515&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6346301693336957386)

MetaTrader 4 / Trading


### Warning

Information contained in this article is solely my point of view. As an author I do not induce you to act according to the algorithm described in this article. Besides I warn you of possible losses that may result from the use of this information.

### Introduction

The article contains a description of a trading method - volume accumulation of profit trades. I suppose only profit trades can be accumulated. The article explains the optimal way to implement this and contains an EA code that helps to execute such trades correctly.

### Concept of the Algorithm

Accumulating the volume of a profitable trade allows to gain maximum profit from the market movement that we come across. But the volume must be accumulated so that it does not result in the increase of risks. One of the algorithms of such volume accumulation is described in this article.

First of all we need a reference point - the first main trade. The main trade should be of volume larger than volume of each auxiliary trade. Suppose the volume of the main trade is 0.2 lot, volume of auxiliary trades will be 0.1 lot. Trailing Stop Loss is used for all trades, for example 50 points. When profit of the main trade reaches +100 points, Stop Loss will be +50. At this moment a trade of 0.1 lot is opened in the same direction with Stop Loss of -50 points. If price goes back, both trades will be closed by Stop Loss order. Acquired profit will be equal to 50 points of 0.2 lot, loss - 50 points of 0.1 lot. Totally profit will be equal to 50 points of 0.1 lot. Thus loss protection is achieved, while the trade volume is increased.

If a trade continues movement in the necessary direction, when profit of the auxiliary trade reaches +50 points, Trailing Stop Loss is enabled. When profit of 200 points is reached in the main trade, and that of 100 points in the auxiliary one, one more auxiliary trade will be opened. SL again is equal to -50. And so on.

This quite a simple method allows gaining good profit from lot accumulation. At the same time risks are minimized. Actually risk here is only losing part of the main trade, i.e. risk that the first auxiliary trade makes loss. But this is not a risk of loss, this is a risk of not getting the full profit.

The standard TrailingStopLoss
allows trailing trades only if there is profit. But if it is done on auxiliary trades before the necessary profit is gained, the profitability of such an approach to lot accumulation can be increased. You can do this as well as optimize opening of auxiliary orders by means of MQL4.

### Implementation

The Expert Advisor written for this purpose is based on the EA described in the article ["Comfortable Scalping"](https://www.mql5.com/en/articles/1509). In this article the EA has combined functions: it plays the role of a trainer-game and a tool for opening trades. For this trade the function of a game was deleted. Thus the Expert Advisor draws two arrows on a chart - up and down. Deletion of one of them is a signal to open trades in a necessary direction. For example, deleting a down arrow an up arrow is left on a chart. For the EA this is a signal to open a Buy order and placing a number of pending Buy Stop orders.

A market order is the main order here. Pending orders have the functions of auxiliary orders that have a smaller lot than the main one. For the calculation of "frequency" of auxiliary trades opening and their quantity, two notions are used. The first one is the ultimate goal, Take Profit. It is equal for all orders (main and auxiliary). The second notion is a step for opening pending orders. Depending on the ultimate goal the EA calculate how many orders it can place in the interval from the current price to the Take Profit level.

For example, if we use Take Profit equal to 400 points and the step for order opening is 100 points (by default), 4 orders will be opened at Buy. The first one is Buy at the Ask price, the main order. The second one is an auxiliary Buy Stop at the price Ask+100 points. The third one is an auxiliary Buy Stop at Ask+200 points. The fourth one - an auxiliary Buy Stop at the price of Ask+300 points. Take profit for all orders will be equal to Ask+400 points, i.e. for the first order it will be 400 points, for the second one 300, for the third one 200 and 100 points for the fourth order.

TrailingStopLossfor the main trade works only if a necessary profit is obtained (by default 50 points). For auxiliary trades it works from the moment of opening (i.e. Stop Loss can be trailed the area of loss). The Trailing Stop Loss level is equal for all trades. Besides Trailing Stop Loss is a Stop Loss level for auxiliary trades.

If the main order is closed, remaining pending orders are deleted. After that two arrows are drawn on the chart again. This indicates the readiness of the EA to open trades again.

All this may seem lengthy but it can be easily implemented in practice. Let's analyze the EA code.

```
//+------------------------------------------------------------------+
//|                                                   take_trend.mq4 |
//|                                       Copyright © 2008, FXRaider |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2008, FXRaider"
extern int gap=20;            //level at which arrows are placed
extern int TP=400;            //Take Profit level
extern int SL=0;              //Stop Loss level
extern double Lots1=0.2;      //lot of a main trade
extern double Lots2=0.1;      //lot of auxiliary trades
extern int slippage=2;        //level of acceptable requote
extern int MagicNumber1=5345; //magic number of the main trade
extern int MagicNumber2=4365; //magic number of auxiliary trades
extern int Open_Step=100;     //step for opening auxiliary trades

extern bool UseTrailing = true; //enabling/disabling T-SL
extern int TrailingStop = 50;   //Trailing Stop Loss level
extern int TrailingStep = 1;    //Trailing Stop Loss step
int start()
  {
//------------------------------
//+----------------------------------------------------------------------------------------------+
//|                              searching open orders for a pair                                |
    int pos_sell=0;
  for (int i_op_sell=OrdersTotal()-1; i_op_sell>=0; i_op_sell--)
  {
   if (!OrderSelect(i_op_sell,SELECT_BY_POS,MODE_TRADES)) break;
   if (Symbol()==OrderSymbol()
   &&OrderMagicNumber()==MagicNumber1
   &&(OrderType()==OP_SELL))
   {
    pos_sell=1; break;
   }
  }

    int pos_buy=0;
  for (int i_op_buy=OrdersTotal()-1; i_op_buy>=0; i_op_buy--)
  {
   if (!OrderSelect(i_op_buy,SELECT_BY_POS,MODE_TRADES)) break;
   if (Symbol()==OrderSymbol()
   &&OrderMagicNumber()==MagicNumber1
   &&(OrderType()==OP_BUY))
   {
    pos_buy=1; break;
   }
  }
//|                              searching open orders for a pair                                |
//+----------------------------------------------------------------------------------------------+
//+------------------------------------------------------------------------------------+
//|                                 working with objects                               |

//+----------------------------------------------------------+
//|                    deleting objects                      |
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
//|                    deleting objects                      |
//+----------------------------------------------------------+

//+----------------------------------------------------------+
//|                   drawing objects                        |
   if((ObjectFind("down") != 0&&ObjectFind("up") != 0) //if no objects
   &&!pos_sell&&!pos_buy)                              //if no open orders
   {
     ObjectCreate("down", OBJ_ARROW, 0, Time[0], Open[0]-gap*Point); //draw a down arrow
     ObjectSet("down", OBJPROP_STYLE, STYLE_DOT);
     ObjectSet("down", OBJPROP_ARROWCODE, 234);
     ObjectSet("down", OBJPROP_COLOR, Red);

     ObjectCreate("up", OBJ_ARROW, 0, Time[0], Open[0]+gap*Point); //draw an up arrow
     ObjectSet("up", OBJPROP_STYLE, STYLE_DOT);
     ObjectSet("up", OBJPROP_ARROWCODE, 233);
     ObjectSet("up", OBJPROP_COLOR, Blue);
   }
//|                   drawing objects                        |
//+----------------------------------------------------------+

//|                                 working with objects                               |
//+------------------------------------------------------------------------------------+




//+----------------------------------------------------------------------------------------------+
//|                                deleting unnecessary orders                                   |
int cnt_del;
if(pos_buy==0)
{
  for (cnt_del=0; cnt_del<OrdersTotal(); cnt_del++)
  {
    if (!(OrderSelect(cnt_del, SELECT_BY_POS, MODE_TRADES))) continue;
    if(OrderSymbol()==Symbol())
    {
     if (OrderType()==OP_BUYSTOP && OrderMagicNumber()==MagicNumber2) OrderDelete(OrderTicket());
    }
  }
 }

if(pos_sell==0)
{
  for (cnt_del=0; cnt_del<OrdersTotal(); cnt_del++)
  {
    if (!(OrderSelect(cnt_del, SELECT_BY_POS, MODE_TRADES))) continue;
    if(OrderSymbol()==Symbol())
    {
    if (OrderType()==OP_SELLSTOP && OrderMagicNumber()==MagicNumber2) OrderDelete(OrderTicket());
    }
  }
 }
//|                                deleting unnecessary orders                                   |
//+----------------------------------------------------------------------------------------------+


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
 int stop_positions=MathFloor(TP/Open_Step-1);
 int i, open_step_2;
  if(
     ObjectGet("up", OBJPROP_PRICE1)==Open[0]+gap*Point
     &&iBarShift(NULL,0,ObjectGet("up",OBJPROP_TIME1))==0
     &&ObjectFind("down") != 0
     &&ObjectFind("up") == 0
     &&!pos_buy
     )
     {
      OrderSend(Symbol(),OP_BUY, Lots1,Ask,slippage,sl_buy,Ask+TP*Point,"take_trend",MagicNumber1,0,Blue);
      for(i=stop_positions;i>=0; i--)
      {
       open_step_2=open_step_2+Open_Step;
       OrderSend(Symbol(),OP_BUYSTOP, Lots2,
       Ask+open_step_2*Point,slippage,
       0,Ask+TP*Point,"take_trend",MagicNumber2,0,Blue);
      }
     }
  if(
     ObjectGet("down", OBJPROP_PRICE1)==Open[0]-gap*Point
     &&iBarShift(NULL,0,ObjectGet("down",OBJPROP_TIME1))==0
     &&ObjectFind("up") != 0
     &&ObjectFind("down") == 0
     &&!pos_sell
     )
     {
      OrderSend(Symbol(),OP_SELL, Lots1,Bid,slippage,sl_sell,Bid-TP*Point,"take_trend",MagicNumber1,0,Red);
      for(i=stop_positions;i>=0; i--)
      {
       open_step_2=open_step_2+Open_Step;
       OrderSend(Symbol(),OP_SELLSTOP, Lots2,
       Bid-open_step_2*Point,slippage,
       0,Bid-TP*Point,"take_trend",MagicNumber2,0,Red);
      }
     }
//|                                   opening trades                                   |
//+------------------------------------------------------------------------------------+


//+-------------------------------------------------------------------------------------------------+
//|                                      trail of open orders                                       |
if (UseTrailing)
{
  for (int trall=0; trall<OrdersTotal(); trall++) {
    if (!(OrderSelect(trall, SELECT_BY_POS, MODE_TRADES))) continue;
    if (OrderSymbol() != Symbol()) continue;

    if (OrderType() == OP_BUY ) {
      if (Bid-OrderOpenPrice() > TrailingStop*Point || OrderMagicNumber()==MagicNumber2) {
        if (OrderStopLoss() < Bid-(TrailingStop+TrailingStep-1)*Point || OrderStopLoss() == 0) {
          OrderModify(OrderTicket(), OrderOpenPrice(), Bid-TrailingStop*Point, OrderTakeProfit(), 0, Blue);
       }
      }
    }

    if (OrderType() == OP_SELL) {
     if (OrderOpenPrice()-Ask > TrailingStop*Point || OrderMagicNumber()==MagicNumber2) {
        if (OrderStopLoss() > Ask+(TrailingStop+TrailingStep-1)*Point || OrderStopLoss() == 0) {
          OrderModify(OrderTicket(), OrderOpenPrice(), Ask+TrailingStop*Point, OrderTakeProfit(), 0, Blue);
        }
     }
    }
  }
 }
//|                                      trail of open orders                                       |
//+-------------------------------------------------------------------------------------------------+

   return(0);
  }
//+------------------------------------------------------------------+
```

The code includes all necessary comments.

Variables:

```
gap – level, on which arrows are placed;
TP – Take Profit level;
SL – Stop Loss level;
Lots1 – lot of a main trade;
Lots2 – lot of auxiliary trades;
slippage – level of acceptable requote;
MagicNumber1 – magic number of the main trade;
MagicNumber2 – magic number of auxiliary trades;
Open_Step – step for opening auxiliary trades;

UseTrailing – enabling/disabling T-SL;
TrailingStop – Trailing Stop Loss level;
TrailingStep – Trailing Stop Loss step.
```

When the Expert Advisor is attached to a chart, it draws two arrows:

![](https://c.mql5.com/2/16/take_trend_1.gif)

The remaining arrow will stay on the chart until the next candlestick:

![](https://c.mql5.com/2/16/take_trend_2.gif)

Immediately after one of the arrows is deleted the EA opens trades:

![](https://c.mql5.com/2/16/take_trend_3.gif)

You see, the remaining arrow is not shown on the screenshot - as it has been written earlier, it is deleted on the next candlestick and arrows do not appear till all trades are closed. As soon as the profit of the first trade reaches a necessary level, T-SL starts working (it works always for auxiliary trades). In this example only the first trade is closed at the rollback. Together with this all other orders are deleted:

![](https://c.mql5.com/2/16/take_trend_4.gif)

### Conclusion

To conclude I would like to add that Trailing Stop Loss level, Take Profit and the level for opening auxiliary trades should be chosen for each pair individually. Also it should be noted that for an efficient use the level of auxiliary trades opening must be larger than the T-SL level. In the main trade it is recommended to use lot size larger than that of auxiliary trades.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1515](https://www.mql5.com/ru/articles/1515)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1515.zip "Download all attachments in the single ZIP archive")

[take\_trend.mq4](https://www.mql5.com/en/articles/download/1515/take_trend.mq4 "Download take_trend.mq4")(8.22 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Method of Determining Errors in Code by Commenting](https://www.mql5.com/en/articles/1547)
- [Comfortable Scalping](https://www.mql5.com/en/articles/1509)
- [Betting Modeling as Means of Developing "Market Intuition"](https://www.mql5.com/en/articles/1505)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39457)**
(12)


![Isiaka Abidoye](https://c.mql5.com/avatar/avatar_na2.png)

**[Isiaka Abidoye](https://www.mql5.com/en/users/sicco)**
\|
1 Jun 2009 at 15:58

Thank u FXRaider (FXPiter.net)

I am making good use of the indicator, but I tried to reach you, your website does not display in English language to see your other services, please send me a link to your english displayed website.

![Isiaka Abidoye](https://c.mql5.com/avatar/avatar_na2.png)

**[Isiaka Abidoye](https://www.mql5.com/en/users/sicco)**
\|
1 Jun 2009 at 16:00

I visited your website but it did not display in English, please, provide me aanother link to your english displayed websites.


![anh trinh](https://c.mql5.com/avatar/2014/12/54876521-3E13.jpg)

**[anh trinh](https://www.mql5.com/en/users/anhtrinh)**
\|
24 Mar 2015 at 12:51

```
I 'm using 1 EA opposite effect .model than 2 fold model than 2 fold sheaves under pip loss
```

![anh trinh](https://c.mql5.com/avatar/2014/12/54876521-3E13.jpg)

**[anh trinh](https://www.mql5.com/en/users/anhtrinh)**
\|
24 Mar 2015 at 12:55

@@


![Obed Ekeocha](https://c.mql5.com/avatar/2021/3/6061D948-45A0.png)

**[Obed Ekeocha](https://www.mql5.com/en/users/obedekeocha)**
\|
29 Mar 2021 at 14:04

Please help I don't know how to trade please help a brother


![All about Automated Trading Championship: Registration](https://c.mql5.com/2/16/698_8.gif)[All about Automated Trading Championship: Registration](https://www.mql5.com/en/articles/1548)

This article comprises useful materials that will help you learn more about the procedure of registration for participation in the Automated Trading Championship.

![Layman's Notes: ZigZag…](https://c.mql5.com/2/16/660_10.gif)[Layman's Notes: ZigZag…](https://www.mql5.com/en/articles/1537)

Surely, a fey thought to trade closely to extremums visited every apprentice trader when he/she saw "enigmatic" polyline for the first time. It's so simple, indeed. Here is the maximum. And there is the minimum. A beautiful picture on the history. And what is in practice? A ray is drawn. It should seem, that is it, the peak! It is time to sell. And now we go down. But hell no! The price is treacherously moving upwards. Haw! It's a trifle, not an indicator. And you throw it out!

![All about Automated Trading Championship: Statistical Reports](https://c.mql5.com/2/16/699_7.gif)[All about Automated Trading Championship: Statistical Reports](https://www.mql5.com/en/articles/1549)

Creating a profitable and stable trading system is always related to statistical data processing. In this article, we pout together statistical reports of the Automated Trading Championships 2006-2007. It may well be that the information they provide will help you find new trading ideas or correct the existing ones. Analyze the data and save your time.

![The Statistic Analysis of Market Movements and Their Prognoses](https://c.mql5.com/2/16/634_10.jpg)[The Statistic Analysis of Market Movements and Their Prognoses](https://www.mql5.com/en/articles/1536)

The present article contemplates the wide opportunities of the statistic approach to marketing. Unfortunately, beginner traders deliberately fail to apply the really mighty science of statistics. Meanwhile, it is the only thing they use subconsciously while analyzing the market. Besides, statistics can give answers to many questions.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/1515&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6346301693336957386)

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