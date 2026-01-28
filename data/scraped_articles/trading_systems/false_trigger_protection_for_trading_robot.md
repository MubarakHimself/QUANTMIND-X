---
title: False trigger protection for Trading Robot
url: https://www.mql5.com/en/articles/2110
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:44:55.908169
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/2110&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070529611370010502)

MetaTrader 4 / Trading systems


### Introduction

This article describes different ways to increase the stability of operation of trading robots by removing possible repeated triggers (chatter): both using entry and exit algorithms separately, as well as connecting them.

### Nature of problem

The problem of false triggers is particularly prominent during dips and sharp rises in the market when amplitude of the current candlestick is high, and there are no precautions against chatter provided in the algorithm of a trading robot. It causes consequent multiple opening and closing of positions on the current candlestick.

Financial consequences are various and depend on specific algorithms of market parameters that are set by developers of trading robots. In any case, trader's expenses on spread are increased proportionally to the number of triggers during chatter.

In this article, I will not consider the subject of analyzing financial instruments (of technical and fundamental character) that can affect the stability of operation of trading Expert Advisors and help to avoid scatter (this is a separate subject — I am the author of the impulse equilibrium theory and systems that it is based on). Here, we will focus on measures of software that don't depend directly on methods of analyzing financial markets.

So, let's proceed to solving the problem. As an example, I will use "MACD Sample" Expert Advisor from the standard set available in the МetaТrader 4 client terminal.

This is visual interpretation of the scatter problem shown with an example where EURUSD price spiked on 2nd of October this year (М15 timeframe, "MACD Sample" EA with default settings):

![](https://c.mql5.com/2/21/EURUSDM15.png)

The screenshot clearly shows 8 subsequent triggers (Buy entries) on a single candlestick. Only 1 of them is correct (in terms of normal market logic), remaining 7 are scatter.

The reasons behind false triggers in this specific case are:

- this is a small value of TakeProfit in default settings (exit algorithm), therefore each position is closing quickly;
- and also the entry algorithm of "MACD Sample" EA that is triggered after closing a previous position by letting a double entry, despite the number of entries on this candlestick.

We have already agreed that matters of filtering market fluctuations won't be considered on purpose (since each trader has their own entry and exit algorithm), therefore to solve the problem we will consider the following factors of a more general nature:

- time (length of candlestick),
- number of triggers (counter),
- amplitude of movement (range of candlestick).

### Solution in entry algorithm

The easiest and at the same time the most reliable way to fix the entry point is through the time factor, the reasons are the following:

- Counter of the amount of triggers implies the creation of a loop in the program, which not only complicates the algorithm, but also slows down the speed of a trading Expert Advisor.
- Amplitude and related control levels of price can be repeated because prices return at the candlestick reversal making criteria of price level uneven.
- The time is irreversible, moves only in one direction (increase), therefore, it is the most accurate and even criteria for solving the problem by providing a one-time trigger or scatter removal.

This way, the main factor is the moment of triggering the entry algorithm, to be more specific, the moment of triggering the order for opening the position ( [OrderSend](https://docs.mql4.com/trading/ordersend)), as these two moments may not match, if some special delays of opening order are present in the algorithm.

So, we can remember the moment (current time) when opening a position. But how to use this parameter in the entry algorithm to ban the second and consequent entries on the specified candlestick? We don't know this moment in advance (its absolute value), therefore we cannot enter it in advance in the entry algorithm. The algorithm should consider (include) certain common condition that will solve the first entry on the candlestick, but prohibit all consequent entries on the candlestick without calculating triggers (the option with a counter we have previously declined).

The solution is fairly simple. First, I will write a code with some comments, and then will clarify in more details. This is an auxiliary code (highlighted in yellow), that needs to be placed in the algorithm of the trading Expert Advisor (see **MACD\_Sample\_plus1.mq4**):

```
//+------------------------------------------------------------------+
//|                                                  MACD Sample.mq4 |
//|                   Copyright 2005-2014, MetaQuotes Software Corp. |
//|                                              https://www.mql4.com |
//+------------------------------------------------------------------+
#property copyright   "2005-2014, MetaQuotes Software Corp."
#property link        "https://www.mql4.com"

input double TakeProfit    =50;
input double Lots          =0.1;
input double TrailingStop  =30;
input double MACDOpenLevel =3;
input double MACDCloseLevel=2;
input int    MATrendPeriod =26;
//--- enter new variable (value in seconds for 1 bar of this TF, for М15 equals 60 с х 15 = 900 с)
datetime Time_open=900;
//--- enter new variable (time of opening the bar with 1st entry)
datetime Time_bar = 0;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   double MacdCurrent,MacdPrevious;
   double SignalCurrent,SignalPrevious;
   double MaCurrent,MaPrevious;
   int    cnt,ticket,total;
//---
// initial data checks
// it is important to make sure that the expert works with a normal
// chart and the user did not make any mistakes setting external
// variables (Lots, StopLoss, TakeProfit,
// TrailingStop) in our case, we check TakeProfit
// on a chart of less than 100 bars
//---
   if(Bars<100)
     {
      Print("bars less than 100");
      return;
     }
   if(TakeProfit<10)
     {
      Print("TakeProfit less than 10");
      return;
     }
//--- to simplify the coding and speed up access data are put into internal variables
   MacdCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,0);
   MacdPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,1);
   SignalCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,0);
   SignalPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,1);
   MaCurrent=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,0);
   MaPrevious=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,1);

   total=OrdersTotal();
   if(total<1)
     {
      //--- no opened orders identified
      if(AccountFreeMargin()<(1000*Lots))
        {
         Print("We have no money. Free Margin = ",AccountFreeMargin());
         return;
        }
      //--- check for long position (BUY) possibility

      //--- enter new string (removes ban on repeated entry if a new bar is opened)
      if( (TimeCurrent() - Time_bar) > 900 ) Time_open = 900;

      if(MacdCurrent<0 && MacdCurrent>SignalCurrent && MacdPrevious<SignalPrevious &&
         MathAbs(MacdCurrent)>(MACDOpenLevel*Point) && MaCurrent>MaPrevious &&
         (TimeCurrent()-Time[0])<Time_open) //enter new string into the entry algorithm (performed only once, the condition on this candlestick cannot be completed afterwards)
        {
         ticket=OrderSend(Symbol(),OP_BUY,Lots,Ask,3,0,Ask+TakeProfit*Point,"macd sample",16384,0,Green);
         if(ticket>0)
           {
            if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES))
             {
              Print("BUY order opened : ",OrderOpenPrice());
              Time_open = TimeCurrent()-Time[0]; //enter new string (store interval from the bar opening time with the entry until the exit moment)
              Time_bar = Time[0]; //enter new string (remember opening time of the bar that had 1st entry)
             }
           }
         else
            Print("Error opening BUY order : ",GetLastError());
         return;
        }
```

Read more:

Instead of absolute time (entry moment) we use relative time — time gap from the moment of opening the current candlestick until the entry moment. This value is then compared with a previously set, fairly large time value (length of the entire candlestick), that allows the first entry to be triggered. At the moment of opening the position we change (decrease) value of the _Time\_open_ variable by writing there value of time gap from the start of the candlestick until the actual point of closing. And since in any consequent moment of time, value of (TimeCurrent() - Time\[0\]) will exceed value that we wrote at the entry point, then the (TimeCurrent() - Time\[0\]) < Time\_open condition will remain impossible, which is achieved by blocking the second and consequent entries on this candlestick.

This way, without any counter of the amount of entries and the analysis of price movement amplitude, we have solved the problem of false triggers.

Below is the result of such simple improvement of the initial algorithm of the EA entry ("MACD Sample\_plus1"):

![](https://c.mql5.com/2/21/EURUSDM15_plus1.png)

We see that there is only one entry on the candlestick, there are no false triggers, and scatter is completely eliminated. Settings by default are completely saved, so it is clear that the problem is solved without assistance of changing settings of EA.

Now that a problem of scatter at the entry is solved, we will improve the entry algorithm in order to exclude a possible scatter that appears with quick closing of the position, that in this specific case increases the profit (impulse is good, and exit was quick, premature).

### Solution in the exit algorithm

Since the initial problem involves removing possible scatter of the trading robot instead of increasing profit, then I will not consider issues of analyzing dynamics of financial instruments in relation to this subject, and will restrict myself with fixing the selected parameter without consideration of such dynamics.

Previously we have used a safe parameter and time factor, and we will use it again strictly regulating the moment of closing the position by time, to be specific, at the point of opening the following candlestick (after entry). This moment in the exit algorithm we will show as:

```
if(!OrderSelect(cnt,SELECT_BY_POS,MODE_TRADES))
         continue;
      if(OrderType()<=OP_SELL &&   // check for opened position
         OrderSymbol()==Symbol())  // check for symbol
        {
         //--- long position is opened
         if(OrderType()==OP_BUY)
           {
            //--- should it be closed?
            if(/* MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious && // remove code of triggering exit on MACD, not to disturb a new condition of closing (see further)
               MacdCurrent>(MACDCloseLevel*Point) &&
             */
               Bid > OrderOpenPrice() &&  // enter new string - optional (price in a positive area in regards to the entry level)
               TimeCurrent() == Time[0] ) // enter new string (simple improvement of exit algorithm: exit is strictly at the moment of opening current candlestick)
              {
               //--- close order and exit
               if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,Violet))
                  Print("OrderClose error ",GetLastError());

               return;
              }
```

Such minor modification will allow the entry algorithm to function (position opened, no conditions for closing), the position lasts until the TimeCurrent() == Time\[0\] moment and closes simultaneously at the beginning of a new candlestick after the impulse. Eventually, in addition to the protection from scatter, we also made a good profit (see photo "MACD Sample\_plus2"):

![](https://c.mql5.com/2/21/EURUSDM15_plus2.png)

For this purpose we had to remove the trigger by MACD from the exit algorithm, otherwise necessary condition for exiting wouldn't be able to take place.

Thus, it appears that the scatter problem can be solved separately in the entry and exit algorithms. Now, let's discuss how to solve a problem by connecting these algorithms of opening and closing positions.

### Connecting entry and exit algorithms

Connection implies certain preliminary modeling of the entire process: opening position — managing — closing position. This is also reflected in selecting a shift in indicators and functions used in entry and exit algorithms.

For example, if you use the TimeCurrent() = Time\[0\] condition in the exit algorithm, and the exit point is set strictly by the beginning of the current candlestick, then the entry algorithm should be tested on previous complete bars, so exit condition could be met. Therefore, in order to close the position under the TimeCurrent() = Time\[0\] condition without any additional conditions, it is necessary that the entire algorithm of comparison (at the exit) is performed (finished) on the previous bar. There should be an offset that equals 1 in the settings of indicators that participate in comparison of values. In this case the comparison of values will be correct, and the beginning of the current candlestick will be a logical ending to the exit algorithm.

This way, connecting algorithms of exit and entry is also linked to the time factor.

### Conclusion

The problem of false triggering of Expert Advisors is efficiently solved by using the time factor in the entry algorithm. Additional stability of operation for EA can be achieved by fixing the exit point (for example, by time), and connecting entry and exit algorithms through preliminary modeling of the main logic of triggering and offsets (a bar where indicator or function will be calculated).

Below are the codes of EA: the initial one ( **MACD\_Sample.mq4**), with improved entry ( **MACD\_Sample\_plus1.mq4**), and with improved exit ( **MACD\_Sample\_plus2.mq4**). Only Buy channels are improved, whereas Sell channels remain without changes deliberately to compare the initial and improved algorithms.

And, certainly, all indicated EAs are for demonstration purposes and are not intended for real trading in the financial markets.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2110](https://www.mql5.com/ru/articles/2110)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2110.zip "Download all attachments in the single ZIP archive")

[MACD\_Sample.mq4](https://www.mql5.com/en/articles/download/2110/macd_sample.mq4 "Download MACD_Sample.mq4")(6.1 KB)

[macd\_sample\_plus2.mq4](https://www.mql5.com/en/articles/download/2110/macd_sample_plus2.mq4 "Download macd_sample_plus2.mq4")(6.35 KB)

[macd\_sample\_plus1.mq4](https://www.mql5.com/en/articles/download/2110/macd_sample_plus1.mq4 "Download macd_sample_plus1.mq4")(6.95 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [On Methods to Detect Overbought/Oversold Zones. Part I](https://www.mql5.com/en/articles/7782)
- [How to reduce trader's risks](https://www.mql5.com/en/articles/4233)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/94486)**
(4)


![Zaw Zaw](https://c.mql5.com/avatar/avatar_na2.png)

**[Zaw Zaw](https://www.mql5.com/en/users/zawz84360)**
\|
5 Sep 2016 at 15:27

Good update


![rafyazhari](https://c.mql5.com/avatar/avatar_na2.png)

**[rafyazhari](https://www.mql5.com/en/users/rafyazhari)**
\|
9 Sep 2016 at 08:32

**Zaw Zaw:**

Good update

👍🏻


![Johnathan Froeming](https://c.mql5.com/avatar/2019/1/5C335CA2-E309.png)

**[Johnathan Froeming](https://www.mql5.com/en/users/johnabtc)**
\|
16 Feb 2019 at 22:04

I'll try it. Thanks!


![Yousuf Mesalm](https://c.mql5.com/avatar/2022/5/6288CAC8-33C2.jpg)

**[Yousuf Mesalm](https://www.mql5.com/en/users/20163440)**
\|
15 Feb 2020 at 16:47

Very good , thank you


![Graphical Interfaces X: Updates for Easy And Fast Library (Build 2)](https://c.mql5.com/2/23/Graphic-interface_10.png)[Graphical Interfaces X: Updates for Easy And Fast Library (Build 2)](https://www.mql5.com/en/articles/2634)

Since the publication of the previous article in the series, Easy And Fast library has received some new features. The library structure and code have been partially optimized slightly reducing CPU load. Some recurring methods in many control classes have been moved to the CElement base class.

![Graphical Interfaces IX: The Progress Bar and Line Chart Controls (Chapter 2)](https://c.mql5.com/2/23/IX__1.png)[Graphical Interfaces IX: The Progress Bar and Line Chart Controls (Chapter 2)](https://www.mql5.com/en/articles/2580)

The second chapter of the part nine is dedicated to the progress bar and line chart controls. As always, there will be detailed examples provided to reveal how these controls can be used in custom MQL applications.

![How to copy signals using an EA by your rules?](https://c.mql5.com/2/23/ava__1.png)[How to copy signals using an EA by your rules?](https://www.mql5.com/en/articles/2438)

When you subscribe to signals, such situation may occur: your trade account has a leverage of 1:100, the provider has a leverage of 1:500 and trades using the minimal lot, and your trade balances are virtually equal — but the copy ratio will comprise only 10% to 15%. This article describes how to increase the copy rate in such cases.

![Graphical Interfaces IX: The Color Picker Control (Chapter 1)](https://c.mql5.com/2/23/IX.png)[Graphical Interfaces IX: The Color Picker Control (Chapter 1)](https://www.mql5.com/en/articles/2579)

With this article we begin chapter nine of series of articles dedicated to creating graphical interfaces in MetaTrader trading terminals. It consists of two chapters where new elements of controls and interface, such as color picker, color button, progress bar and line chart are presented.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=otkzvuvbgymrvshsjrzvvdnvuyjdoeoh&ssn=1769186694536525588&ssn_dr=0&ssn_sr=0&fv_date=1769186694&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2110&back_ref=https%3A%2F%2Fwww.google.com%2F&title=False%20trigger%20protection%20for%20Trading%20Robot%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176918669474477006&fz_uniq=5070529611370010502&sv=2552)

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