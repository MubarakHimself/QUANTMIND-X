---
title: The Golden Rule of Traders
url: https://www.mql5.com/en/articles/1349
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:55:07.673688
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=qsnolvsjwwxwxvufhpjzjhbqhqtukmjx&ssn=1769252106840821051&ssn_dr=0&ssn_sr=0&fv_date=1769252106&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1349&back_ref=https%3A%2F%2Fwww.google.com%2F&title=The%20Golden%20Rule%20of%20Traders%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925210670916910&fz_uniq=5083212207413532426&sv=2552)

MetaTrader 4 / Trading systems


### Introduction

The main task of a trader is not only to find the right time to enter the market. It is also necessary to find the right moment to exit it. The golden rule of trading says: "Always cut your losses short and let your profits run".

To make profit based on a high mathematical expectation, we must understand three basic principles of good trading.

1. Know your risk when entering the market (that is an initial Stop Loss value);

2. Cut your profits short and allow your profits to run (do not close your position before it is required by your system);

3. Know the mathematical expectation of your system – test and adjust it regularly.

### Step-by-Step Positions Trailing Method Allowing Profits to Run

Many think that it is impossible to make profit, as we do not know where the market will go. But do we really need to know that to trade successfully? Successful trading is mostly based on a
properly designed system considering suitable moments for entering the market. Such considerations are made using the power of expectation and the rules of step-by-step positions trailing that allow profit to run
for the highest possible level.

The market entry moment can be found in many ways, for example, using candlesticks models, wave models, etc. At the same time, the profit factor must be considered (profit/loss ratio).

This method is based on the following requirement: a trader selects the lowest possible Stop Loss value when opening a position. That value can be determined using various methods, for
example, it may be equal to 1.5% of the deposit. When the market reaches a profit equal to Stop Loss value, the half of the lot is closed but Stop Loss is not changed!

Therefore, we create a sort of a safety net, in case the market goes the opposite way. I.e., we reduce our risk by fixing the minimum losses. If the market moved in favorable direction just to turn
back some time later, Stop Loss is triggered (Fig. 1-3).

![](https://c.mql5.com/2/12/op_buy_1_1.jpg)

Fig. 1. Opening a position

![](https://c.mql5.com/2/12/op_buy_2_1.jpg)

Fig. 2. Setting Stop Loss

if the market has turned back:

![](https://c.mql5.com/2/12/stoploss_2.jpg)

Fig. 3. If the market has turned back, you are at break-even level

### Position Trailing Program Code

We offer a program code trailing open positions and actualizing the second golden principle, as it allows profit to run for the highest possible level.

If the market still moves in favorable direction and reaches some predetermined value, for example, 100 pips, Stop Loss is reset to a break-even level. Further resets are made when reaching the
profit in predetermined intervals, for example, 50 pips. We can move Stop Loss at each next bar but brokers do not like frequent resets, especially when the trade is performed on the lower timeframes. The
error file (stdlib.mq4) from the libraries folder even has the error # 8 error="too frequent requests" exactly for that case.

The method for determining each next Stop Loss level is selected by a price position at the time of gaining a profit depending on the Fibonacci levels. Applied Fibonacci levels are built according to Vegas Tunnel method here.

Fibonacci levels calculation is performed by LevelFibo() levels generation function:

```
//+------------------------------------------------------------------+
   void LevelFibo()
   {
   double Fb1,Fb2,Fb3,Fb4,Fb5,Fb6,Fb7,Fb8,Fb9,Fb10,Fb11,Fb12,Fb13;
   // "Vegas" channel
   double Ma144_1 = iMA(NULL,0,144,0,MODE_EMA,PRICE_CLOSE,1);
   double Ma169_1 = iMA(NULL,0,169,0,MODE_EMA,PRICE_CLOSE,1);
   // "Vegas" channel median
   double MedVegas=NormalizeDouble((Ma144_1+Ma169_1)/2,Digits);
   // calculate Fibo levels values using "Vegas" method
   Fb1=MedVegas-377*Point;     Fb12=MedVegas+377*Point;
   Fb2=MedVegas-233*Point;     Fb11=MedVegas+233*Point;
   Fb3=MedVegas-144*Point;     Fb10=MedVegas+144*Point;
   Fb4=MedVegas-89*Point;      Fb9=MedVegas+89*Point;
   Fb5=MedVegas-55*Point;      Fb8=MedVegas+55*Point;
   Fb6=MedVegas-34*Point;      Fb7=MedVegas+34*Point;
   }
//+------------------------------------------------------------------+
```

When calculating Stop Loss for BUY positions, the profit is the difference between Max price of the first bar High\[1\] and position opening level OrderOpenPrice(). Stop Loss level is defined as the
"closest" Fibonacci level relative to the Min value of the first bar Low\[1\] (Fig. 4).

![](https://c.mql5.com/2/12/sl_b_1.jpg)

Fig. 4. Stop Loss calculation for BUY position

When calculating Stop Loss for SELL positions, the profit is the difference between position opening level OrderOpenPrice() and Max price of the first bar High\[1\] (Fig. 5).

![](https://c.mql5.com/2/12/sl_s.jpg)

Fig. 5. Stop Loss calculation for SELL position

For Buy positions, Stop Loss values are based on Fibo levels. Depending on the lowest value of the first candlestick, it is presented as a separate function.

The function code is shown below:

```
//+---------------------------------------------------------------------+
//| Function (table) for specifying Stop Loss values for BUY position   |
//| by Fibo levels according to the lowest value of the first candle    |
//+---------------------------------------------------------------------+
 void StopLevelFiboBuy()
   {
   if(Low[1]>Fb12)                                newSL_B=Fb12-100*Point;
   if(Low[1]<=Fb12 && Low[1]>(Fb12+Fb11)/2)       newSL_B=(Fb12+Fb11)/2;
   if(Low[1]<=(Fb12+Fb11)/2 && Low[1]>Fb11)       newSL_B=Fb11;
   if(Low[1]<=Fb11 && Low[1]>(Fb11+Fb10)/2)       newSL_B=(Fb11+Fb10)/2;
   if(Low[1]<=(Fb10+Fb11)/2 && Low[1]>Fb10)       newSL_B=Fb10;
   if(Low[1]<=Fb10 && Low[1]>(Fb10+Fb9)/2)        newSL_B=Fb9;
   if(Low[1]<=(Fb10+Fb9)/2 && Low[1]>Fb9)         newSL_B=Fb8;
   if(Low[1]<=Fb9  && Low[1]>Fb8)                 newSL_B=Fb7;
   if(Low[1]<=Fb8  && Low[1]>Fb7)                 newSL_B=(Fb7+MedVegas)/2;
   if(Low[1]<=Fb7  && Low[1]>MedVegas)            newSL_B=Fb6;
   if(Low[1]<=MedVegas && Low[1]>(MedVegas+Fb6)/2)newSL_B=Fb6;
   if(Low[1]<=(MedVegas+Fb6)/2 && Low[1]>Fb6)     newSL_B=Fb5;
   if(Low[1]<=Fb6  && Low[1]>Fb5)                 newSL_B=Fb4;
   if(Low[1]<=Fb5  && Low[1]>Fb4)                 newSL_B=(Fb3+Fb4)/2;
   if(Low[1]<=Fb4  && Low[1]>Fb3)                 newSL_B=Fb3;
   if(Low[1]<=Fb3  && Low[1]>(Fb3+Fb2)/2)         newSL_B=(Fb3+Fb2)/2;
   if(Low[1]<=(Fb3+Fb2)/2  && Low[1]>Fb2)         newSL_B=Fb2;
   if(Low[1]<=Fb2  && Low[1]>(Fb2+Fb1)/2)         newSL_B=(Fb1+Fb2)/2;
   if(Low[1]<=(Fb2+Fb1)/2 && Low[1]>Fb1)          newSL_B=Fb1;
   if(Low[1]<=Fb1)                                newSL_B=Fb1-100*Point;
   }
//+------------------------------------------------------------------+
```

The table of Stop Loss values by Fibo levels depending on the maximum value of the first StopLevelFiboSell() function candlestick for Sell positions is represented by the following code:

```
//+----------------------------------------------------------------------+
//| Function (table) for specifying Stop Loss values for SELL position   |
//| by Fibo levels according to the highest value of the first candle    |
//+----------------------------------------------------------------------+
 void StopLevelFiboSell()
   {
   if(High[1]<=Fb12 && High[1]>(Fb12+Fb11)/2)        newSL_S=Fb12+100*Point;
   if(High[1]<=Fb12 && High[1]>Fb11)                 newSL_S=Fb12;
   if(High[1]<=Fb11 && High[1]>Fb11+Fb10)            newSL_S=Fb11;
   if(High[1]<=Fb10 && High[1]>(Fb10+Fb9)/2)         newSL_S=(Fb11+Fb10)/2;
   if(High[1]<=Fb9  && High[1]>Fb8)                  newSL_S=(Fb10+Fb9)/2;
   if(High[1]<=Fb8  && High[1]>Fb7)                  newSL_S=Fb9;
   if(High[1]<=Fb7  && High[1]>MedVegas)             newSL_S=Fb8;
   if(High[1]<=MedVegas && High[1]>MedVegas)         newSL_S=Fb7;
   if(High[1]<=(MedVegas+Fb6)/2 && High[1]>Fb6)      newSL_S=MedVegas;
   if(High[1]<=Fb6  && High[1]>Fb5)                  newSL_S=MedVegas;
   if(High[1]<=Fb5  && High[1]>Fb4)                  newSL_S=Fb6;
   if(High[1]<=Fb4  && High[1]>Fb3)                  newSL_S=Fb5;
   if(High[1]<=Fb3  && High[1]>Fb2)                  newSL_S=Fb4;
   if(High[1]<=Fb2  && High[1]>(Fb2+Fb1)/2)          newSL_S=(Fb2+Fb3)/2;
   if(High[1]<(Fb2+Fb1)/2   && High[1]>Fb1)          newSL_S=Fb2;
   if(High[1]<Fb1)                                   newSL_S=(Fb2+Fb1)/2;
   }
//+------------------------------------------------------------------+
```

It would be appropriate to add LevelFibo Fibonacci levels calculation function to each of the two mentioned dependency functions. That has been done in the attached demo files.

Now, regarding the combination of two functions or rather enabling LevelFibo() levels calculation function in the identification function of Stop Loss levels focused on these levels. This combination makes sense, as the functions work together. Therefore, the number of functions calls during the trailing will be reduced - only one will remain instead of two.

After combining, they will look as follows:

```
//+----------------------------------------------------------------------+
//| Function (table) for specifying Stop Loss values for BUY position    |
//| by Fibo levels according to the lowest value of the first candle     |
//+----------------------------------------------------------------------+
   void StoplevelFiboBuy()
   {
   double Fb1,Fb2,Fb3,Fb4,Fb5,Fb6,Fb7,Fb8,Fb9,Fb10,Fb11,Fb12,Fb13;
   double Ma144_1 = iMA(NULL,0,144,0,MODE_EMA,PRICE_CLOSE,1);
   double Ma169_1 = iMA(NULL,0,169,0,MODE_EMA,PRICE_CLOSE,1);
   double MedVegas=NormalizeDouble((Ma144_1+Ma169_1)/2,Digits);
   Fb1=MedVegas-377*Point;     Fb12=MedVegas+377*Point;
   Fb2=MedVegas-233*Point;     Fb11=MedVegas+233*Point;
   Fb3=MedVegas-144*Point;     Fb10=MedVegas+144*Point;
   Fb4=MedVegas-89*Point;      Fb9=MedVegas+89*Point;
   Fb5=MedVegas-55*Point;      Fb8=MedVegas+55*Point;
   Fb6=MedVegas-34*Point;      Fb7=MedVegas+34*Point;
   if(Low[1]>Fb12)                                newSL_B=Fb12-100*Point;
   if(Low[1]<=Fb12 && Low[1]>(Fb12+Fb11)/2)       newSL_B=(Fb12+Fb11)/2;
   if(Low[1]<=(Fb12+Fb11)/2 && Low[1]>Fb11)       newSL_B=Fb11;
   if(Low[1]<=Fb11 && Low[1]>(Fb11+Fb10)/2)       newSL_B=(Fb11+Fb10)/2;
   if(Low[1]<=(Fb10+Fb11)/2 && Low[1]>Fb10)       newSL_B=Fb10;
   if(Low[1]<=Fb10 && Low[1]>(Fb10+Fb9)/2)        newSL_B=Fb9;
   if(Low[1]<=(Fb10+Fb9)/2 && Low[1]>Fb9)         newSL_B=Fb8;
   if(Low[1]<=Fb9  && Low[1]>Fb8)                 newSL_B=Fb7;
   if(Low[1]<=Fb8  && Low[1]>Fb7)                 newSL_B=(Fb7+MedVegas)/2;
   if(Low[1]<=Fb7  && Low[1]>MedVegas)            newSL_B=Fb6;
   if(Low[1]<=MedVegas && Low[1]>(MedVegas+Fb6)/2)newSL_B=Fb6;
   if(Low[1]<=(MedVegas+Fb6)/2 && Low[1]>Fb6)     newSL_B=Fb5;
   if(Low[1]<=Fb6  && Low[1]>Fb5)                 newSL_B=Fb4;
   if(Low[1]<=Fb5  && Low[1]>Fb4)                 newSL_B=(Fb3+Fb4)/2;
   if(Low[1]<=Fb4  && Low[1]>Fb3)                 newSL_B=Fb3;
   if(Low[1]<=Fb3  && Low[1]>(Fb3+Fb2)/2)         newSL_B=(Fb3+Fb2)/2;
   if(Low[1]<=(Fb3+Fb2)/2  && Low[1]>Fb2)         newSL_B=Fb2;
   if(Low[1]<=Fb2  && Low[1]>(Fb2+Fb1)/2)         newSL_B=(Fb1+Fb2)/2;
   if(Low[1]<=(Fb2+Fb1)/2 && Low[1]>Fb1)          newSL_B=Fb1;
   if(Low[1]<=Fb1)                                newSL_B=Fb1-100*Point;
   }
// ----------------------------------------------------------------------+
```

Main command code of positions trailing is represented as an Expert Advisor fragment with the sequence of actions for execution of the functions mentioned above according to the price
values of the current bars. Step-by-step trailing starts after an appropriate open order has been selected.

A small fragment of trailing code is shown below:

```
//+------------------------------------------------------------------+
//| TRAILING OPEN POSITIONS                                          |
//+------------------------------------------------------------------+
for(int i=OrdersTotal()-1; i>=0; i--)
  {
   if(!OrderSelect(i,SELECT_BY_POS,MODE_TRADES))
     {Print("Order selection error = ",GetLastError());}
   if(OrderSymbol()==Symbol())
     {
      if(OrderType()==OP_BUY)
        {
         if(OrderMagicNumber()==Magic)
           {
            // ----------------------------------------------------------------------+
            if((High[1]-OrderOpenPrice())>=SL_B*Point && OrderLots()==0.2)Close_B_lot();
            // ----------------------------------------------------------------------+
            if((High[1]-OrderOpenPrice())/Point>=100 && OrderLots()==0.1 && OrderStopLoss()<OrderOpenPrice())
              {
               Print(" 1 - StopLoss shift");
               if(!OrderModify(OrderTicket(),OrderOpenPrice(),OrderOpenPrice()+2*Point,OrderTakeProfit(),0,Aqua))
                 {
                  Print(" at Modif.ord.# ",OrderTicket()," Error # ",GetLastError());
                 }
               return;
              }
            // ----------------------------------------------------------------------+
            if((High[1]-OrderOpenPrice())>=120*Point && OrderOpenPrice()>Ma144_1-144*Point)
              {
               StoplevelFiboBuy();
               newSL_B=newSL_B+21*Point;
               if((Bid-newSL_B)/Point<StopLevel)newSL_B=Bid-StopLevel*Point;
               if(newSL_B>OrderStopLoss() && (Bid-newSL_B)/Point>StopLevel)
                 {
                  Print("2nd shift of StopLoss ");
                  Modyf_B_lot();
                 }
              }
            // ----------------------------------------------------------------------+
            if((High[1]-OrderOpenPrice())>=200*Point && (High[1]-OrderOpenPrice())<=250*Point)
              {
               StoplevelFiboBuy();
               if((Bid-newSL_B)/Point<StopLevel)newSL_B=Bid-StopLevel*Point;
               if(newSL_B>OrderStopLoss() && (Bid-newSL_B)/Point>StopLevel)
                 {
                  Print(" 3rd shift by level order # ",OrderTicket());
                  Modyf_B_lot();
                 }
              }
            // ----------------------------------------------------------------------+
            if((High[1]-OrderOpenPrice())>=250*Point && OrderOpenPrice()>Ma144_1-144*Point)
              {
               StoplevelFiboBuy();
               newSL_B=newSL_B+10*Point;
               if((Bid-newSL_B)/Point<StopLevel) newSL_B=Bid-StopLevel*Point;
               if(newSL_B>OrderStopLoss() && (Bid-newSL_B)/Point>StopLevel)
                 {
                  Print(" 4th shift by level order # ",OrderTicket());
                  Modyf_B_lot();
                 }
              }
            // ----------------------------------------------------------------------+
            if((High[1]-OrderOpenPrice())>=300*Point && (High[1]-OrderOpenPrice())<=350*Point)
              {
               StoplevelFiboBuy();
               newSL_B=newSL_B+20*Point;
               if((Bid-newSL_B)/Point<StopLevel) newSL_B=Bid-StopLevel*Point;
               if(newSL_B>OrderStopLoss() && (Bid-newSL_B)/Point>StopLevel)
                 {
                  Print(" 5th shift by level order # ",OrderTicket());
                  Modyf_B_lot();
                 }
              }
            // ----------------------------------------------------------------------+
            ...
           }
        }
     }
  }
```

The sequence of step-by-step trailing consists of 8 steps in the attached version of the demo Expert Advisor.

Below is the screenshot displaying the steps of operation of the demo Expert Advisor for Buy position trailing. The position was opened due to the fact that the "absorption of three
candles" model has been formed. Besides, the "inverted hammer" is among the absorbed candles. That fact reinforces the signal's strength to open a buy position.

![](https://c.mql5.com/2/12/trendup_1.jpg)

Fig. 6. Example of Buy position trailing

"Vegas" indicator should be used to see Fibo lines like on a light screenshot. It can be seen on MQL4 website: [https://www.mql5.com/en/code/7148](https://www.mql5.com/en/code/7148)

The same little stretched screenshot from the "dark" screen:

![](https://c.mql5.com/2/12/result_up_demo.jpg)

Fig. 7. Screenshot from the monitor screen (demo version for Buy)

![](https://c.mql5.com/2/12/result_sell_demo.jpg)

Fig. 8. Example of Sell position trailing

Temporary testing parameters should be installed to display the work of demo Expert Advisors, as shown in Fig. 9-10 below:

![](https://c.mql5.com/2/12/boot_demo_sell.jpg)

Fig. 9. Demo\_trail\_Buy.mql Expert Advisor testing parameters

![](https://c.mql5.com/2/12/boot_demo_up.jpg)

Fig. 10. Demo\_trail\_Sell.mql Expert Advisor testing parameters

Note: Files with demo Expert Advisors are attached to the article, as mini-robots for trailing buy and sell positions. Positions are opened in a double lot here. But when Stop Loss is moved to a break-even level, there is confidence in the future price movement in the right direction and it is possible to add another position.

### Conclusion

Presented examples of orders trailing show that Stop Loss orders moving method focused on Fibo dynamic levels may yield positive results. The described method can be recommended for practical use in trading.

Dynamic Fibo levels identifying functions:

- LevelFiboKgd.mq4
- StopLevelFiboBuy.mq4
- StopLevelFiboSell.mq4
- Demo files:
- Demo\_trail\_Buy\_v1.mq4 – Demo file for Buy
- Demo\_trail\_Sell\_v1.mq4 - Demo file for Sell

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1349](https://www.mql5.com/ru/articles/1349)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1349.zip "Download all attachments in the single ZIP archive")

[Demo\_trail\_Buy\_v1.mq4](https://www.mql5.com/en/articles/download/1349/Demo_trail_Buy_v1.mq4 "Download Demo_trail_Buy_v1.mq4")(11.56 KB)

[Demo\_trail\_Sell\_v1.mq4](https://www.mql5.com/en/articles/download/1349/Demo_trail_Sell_v1.mq4 "Download Demo_trail_Sell_v1.mq4")(9.01 KB)

[LevelFiboKgd.mq4](https://www.mql5.com/en/articles/download/1349/LevelFiboKgd.mq4 "Download LevelFiboKgd.mq4")(1.96 KB)

[StopLevelFiboBuy.mq4](https://www.mql5.com/en/articles/download/1349/StopLevelFiboBuy.mq4 "Download StopLevelFiboBuy.mq4")(2.88 KB)

[StopLevelFiboSell.mq4](https://www.mql5.com/en/articles/download/1349/StopLevelFiboSell.mq4 "Download StopLevelFiboSell.mq4")(2.54 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Mechanical Trading System "Chuvashov's Triangle"](https://www.mql5.com/en/articles/1364)
- [Mechanical Trading System "Chuvashov's Fork"](https://www.mql5.com/en/articles/1352)
- [Expert Advisor for Trading in the Channel](https://www.mql5.com/en/articles/1375)
- [Two-Stage Modification of Opened Positions](https://www.mql5.com/en/articles/1529)
- [A Trader's Assistant Based on Extended MACD Analysis](https://www.mql5.com/en/articles/1519)
- [Trend Lines Indicator Considering T. Demark's Approach](https://www.mql5.com/en/articles/1507)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39111)**
(6)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
20 Sep 2012 at 08:15

Great article you have going here, please continue! I like the clean charts and the detailed openness on trades you're taking/have taken. Hope i can also post up some value-adding charts as time goes by. You can also visit www.ikonfx.com to find the articles which I written about it


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
4 Nov 2012 at 17:09

I just skimmed the article, that's really a great summary of exits.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
27 Dec 2012 at 14:54

I am new on this software. Can anyone guide me that how I have to use these files? I know this would be offending but I am sorry and I appreciate for your help.


![1stepbeyond007](https://c.mql5.com/avatar/avatar_na2.png)

**[1stepbeyond007](https://www.mql5.com/en/users/1stepbeyond007)**
\|
3 Sep 2013 at 21:14

hey how to apply this things..I can't understnad

![Martin Kamogelo](https://c.mql5.com/avatar/2020/10/5F91E35A-7AE4.jpg)

**[Martin Kamogelo](https://www.mql5.com/en/users/martinkamogelo)**
\|
31 Oct 2020 at 16:44

Anyone help me using  this please


![Limitless Opportunities with MetaTrader 5 and MQL5](https://c.mql5.com/2/0/TW_logoMarket_60x60.png)[Limitless Opportunities with MetaTrader 5 and MQL5](https://www.mql5.com/en/articles/392)

In this article, I would like to give an example of what a trader's program can be like as well as what results can be achieved in 9 months, having started to learn MQL5 from scratch. This example will also show how multi-functional and informative such a program can be for a trader while taking minimum space on the price chart. And we will be able to see just how colorful, bright and intuitively clear to the user trade information panels can get. As well as many other features...

![Visualize a Strategy in the MetaTrader 5 Tester](https://c.mql5.com/2/0/trade_robot_in_Backtester.png)[Visualize a Strategy in the MetaTrader 5 Tester](https://www.mql5.com/en/articles/403)

We all know the saying "Better to see once than hear a hundred times". You can read various books about Paris or Venice, but based on the mental images you wouldn't have the same feelings as on the evening walk in these fabulous cities. The advantage of visualization can easily be projected on any aspect of our lives, including work in the market, for example, the analysis of price on charts using indicators, and of course, the visualization of strategy testing. This article contains descriptions of all the visualization features of the MetaTrader 5 Strategy Tester.

![OpenCL: From Naive Towards More Insightful Programming](https://c.mql5.com/2/0/OpenCL_Logo__1.png)[OpenCL: From Naive Towards More Insightful Programming](https://www.mql5.com/en/articles/407)

This article focuses on some optimization capabilities that open up when at least some consideration is given to the underlying hardware on which the OpenCL kernel is executed. The figures obtained are far from being ceiling values but even they suggest that having the existing resources available here and now (OpenCL API as implemented by the developers of the terminal does not allow to control some parameters important for optimization - particularly, the work group size), the performance gain over the host program execution is very substantial.

![Get 200 usd for your algorithmic trading article!](https://c.mql5.com/2/0/new_article_system.png)[Get 200 usd for your algorithmic trading article!](https://www.mql5.com/en/articles/408)

Write an article and contribute to the development of algorithmic trading. Share your experience in trading and programming, and we will pay you $200. Additionally, publishing an article on the popular MQL5.com website offers an excellent opportunity to promote your personal brand in a professional community. Thousands of traders will read your work. You can discuss your ideas with like-minded people, gain new experience, and monetize your knowledge.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dytxothgvfwikasarurgdhiydbrslnjw&ssn=1769252106840821051&ssn_dr=0&ssn_sr=0&fv_date=1769252106&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1349&back_ref=https%3A%2F%2Fwww.google.com%2F&title=The%20Golden%20Rule%20of%20Traders%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925210670956059&fz_uniq=5083212207413532426&sv=2552)

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