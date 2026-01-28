---
title: Filtering Signals Based on Statistical Data of Price Correlation
url: https://www.mql5.com/en/articles/269
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:41:37.650271
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/269&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068535204651399840)

MetaTrader 5 / Trading


### How This Began

The idea that led to writing this article appeared after I had read the book by Larry Williams ["Long-Term Secrets to Short-Term Trading"](https://www.mql5.com/go?link=https://www.amazon.com/Long-Term-Secrets-Short-Term-Trading-Williams/dp/0471297224 "http://www.amazon.com/Long-Term-Secrets-Short-Term-Trading-Williams/dp/0471297224"), in which the world record holder in investments (during 1987 he increased his capital by 11,000%) is completely dispelling the myths by "... college professors and other academics, who are rich in theory and poor in knowledge of the market..." about the absence of any correlation between the past behavior of prices and the future trends.

If you toss a coin 100 times, 50 times it will fall up heads and 50 times - tails. With each successive toss, the probability of heads is 50%, the same as of tails. The probability does not change from toss to toss, because this game is random and has no memory. Suppose the markets are behaving like a coin, in a chaotic manner.

Consequently, when a new bar appears, a price has equal opportunity to go up or down, and the previous bars do not affect even the slightest way the current one. Idyll! Create a trading system, set the take profit larger than the stop loss (i.e., set the math. expectation to the positive zone), and the trick is done. Simply breathtaking. However, the problem is that our assumption about the behavior of the market is not quite true. Frankly speaking, it's absurd! And I will prove it.

Let's create an Expert Advisor template using the [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea") and by using simple alphanumeric interventions, present it in a condition suitable for the fulfillment of the task. We will encode an Expert Advisor to simulate buy that follows one, two and three bars closed down. Simulation means that the program will simply remember the parameters of analyzed bars. Sending orders (a more usual way) in this case will not work, because the spreads and swaps are able to question the reliability of the information received.

Here is the code:

```
//+------------------------------------------------------------------+
//|                                                     explorer.mq5 |
//|                        Copyright 2011, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//---Variables---
double profit_percent,open_cur,close_cur;
double profit_trades=0,loss_trades=0,day_cur,hour_cur,min_cur,count;
double open[],close[];
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
/* Calculate percent of closures with increase from the total number */
   profit_percent=NormalizeDouble(profit_trades*100/(profit_trades+loss_trades),2);
   Print("Percent of closures with increase ",profit_percent,"%");   // Enter data to the Journal
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---find out the time---
   MqlDateTime time;                        // Create a structure to store time
   TimeToStruct(TimeCurrent(),time);         // Structuring the data
   day_cur=time.day_of_week;              // Receive the value of the current day
   hour_cur=time.hour;                    // Receive the current hour
   min_cur=time.min;                      // Receive the current minute
//---Find out the prices---
   CopyOpen(NULL,0,0,4,open);ArraySetAsSeries(open,true);
   CopyClose(NULL,0,0,4,close);ArraySetAsSeries(close,true);

   if(close[1]<open[1]/*&&close[2]<open[2]&&close[3]<open[3]*/ && count==0) // If it closed with a loss
     {
      open_cur=open[0];                   // Remember open price of the current bar
      count=1;
     }
   if(open_cur!=open[0] && count==1)      // The current bar has closed
     {
      close_cur=close[1];                 // Remember the close price of the formed bar
      count=0;
      if(close_cur>=open_cur)profit_trades+=1;  // If the close price is higher than open,
      else loss_trades+=1;                      // +1 to closures with profit, otherwise +1 to closures with loss
     }
  }
//+------------------------------------------------------------------+
```

The test will be carried out on EUR/USD, on the interval from January 1, 2000 to December 31, 2010:

> > > > > > > ![Figure 1. The percentage of closures with the increase.](https://c.mql5.com/2/3/w7l91.jpg)

> > > > > > > Figure 1. The percentage of closures with the increase
> > > > > > >
> > > > > > > (The first column shows data for the whole period, the second, third and fourth - after a single, double and triple closing down)

That is what I was talking about! The previous bars have a fairly significant impact on the current one, because the price is always seeking to win losses back.

### Another step forward

Great! Once we are sure that the behavior of prices is not accidental, we should use this amazing fact as soon as possible. Of course, it is not enough for an independent trading system, but it will be a fine tool that can free you from the tedious and often erroneous signals. Let's implement it!

So that's what we need:

1. A self-trading system, showing positive results at least for the last year.
2. Some amusing example that confirms the presence of correlations in the behavior of prices.

I found a lot of useful ideas in the book by L. Williams. I will share one of them with you.

**The TDW (Trade Day Of Week) strategy.** It will allow us to see what will happen if some of the days of the week we will only buy, and the other ones - open only short positions. After all, we can assume that the price within one day grows in a larger percent of cases than within the other. What's the reason for that? The geopolitical situation, macroeconomic statistics, or, as written in the book by A. Elder, Monday and Tuesday are the days of laymen, while Thursdays and Fridays are the time of professionals? Let's try to understand.

First, we will only by each day of the week, and then only sell. At the end of the study we will match the best results, and this will be a filter to our trading system. By the way, I have a couple of words about it. It is a pure classic!

The system is based on two [MAs](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") and [MACDake](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd"). Signals:

1. If the fast moving average crosses the slow one from the bottom up and the MACD histogram is below the zero line, then **BUY**.
2. If the fast moving average crosses the slow one from upside down and MACD is above zero, then **SELL**.

Exit a position using a trailing stop from one point. The lot is fixed - 0,1.

For the purpose of convenience, I've placed the Expert Advisor class in a separate header file:

```
//+------------------------------------------------------------------+
//|                                                       moving.mqh |
//|                        Copyright 2011, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Класс my_expert                                                  |
//+------------------------------------------------------------------+
class my_expert
  {                                                  // Creating a class
   // Closed class members
private:
   int               ma_red_per,ma_yel_per;          // Periods of MAs
   int               ma_red_han,ma_yel_han,macd_han; // Handles
   double            sl,ts;                          // Stop orders
   double            lots;                           // Lot
   double            MA_RED[],MA_YEL[],MACD[];       // Arrays for the indicator values
   MqlTradeRequest   request;                         // Structure of a trade request
   MqlTradeResult    result;                          // Structure of a server response
                                                    // Open class members
public:
   void              ma_expert();                                   // Constructor
   void get_lot(double lot){lots=lot;}                               // Receiving a lot
   void get_periods(int red,int yel){ma_red_per=red;ma_yel_per=yel;} // Receiving the periods of MAs
   void get_stops(double SL,double TS){sl=SL;ts=TS;}                  // Receiving the values of stops
   void              init();                                         // Receiving the indicator values
   bool              check_for_buy();                                // Checking for buy
   bool              check_for_sell();                               // Checking for sell
   void              open_buy();                                     // Open buy
   void              open_sell();                                    // Open sell
   void              position_modify();                              // Position modification
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
/* Function definition */
//---Constructor---
void my_expert::ma_expert(void)
  {
//--- Reset the values of variables
   ZeroMemory(ma_red_han);
   ZeroMemory(ma_yel_han);
   ZeroMemory(macd_han);
  }
//---The function for receiving the indicator values---
void  my_expert::init(void)
  {
   ma_red_han=iMA(_Symbol,_Period,ma_red_per,0,MODE_EMA,PRICE_CLOSE); // Handle of the slow MA
   ma_yel_han=iMA(_Symbol,_Period,ma_yel_per,0,MODE_EMA,PRICE_CLOSE); // Handle of the fast MA
   macd_han=iMACD(_Symbol,_Period,12,26,9,PRICE_CLOSE);               // Handle of MACDaka
//---Copy data into arrays and set indexing like in a time-series---
   CopyBuffer(ma_red_han,0,0,4,MA_RED);
   CopyBuffer(ma_yel_han,0,0,4,MA_YEL);
   CopyBuffer(macd_han,0,0,2,MACD);
   ArraySetAsSeries(MA_RED,true);
   ArraySetAsSeries(MA_YEL,true);
   ArraySetAsSeries(MACD,true);
  }
//---Function to check conditions to open buy---
bool my_expert::check_for_buy(void)
  {
   init();  //Receive values of indicator buffers
/* If the fast MA has crossed the slow MA from bottom up between 2nd and 3rd bars,
   and there was no crossing back. MACD-hist is below zero */
   if(MA_RED[3]>MA_YEL[3] && MA_RED[1]<MA_YEL[1] && MA_RED[0]<MA_YEL[0] && MACD[1]<0)
     {
      return(true);
     }
   return(false);
  }
//----Function to check conditions to open sell---
bool my_expert::check_for_sell(void)
  {
   init();  //Receive values of indicator buffers
/* If the fast MA has crossed the slow MA from up downwards between 2nd and 3rd bars,
  and there was no crossing back. MACD-hist is above zero */
   if(MA_RED[3]<MA_YEL[3] && MA_RED[1]>MA_YEL[1] && MA_RED[0]>MA_YEL[0] && MACD[1]>0)
     {
      return(true);
     }
   return(false);
  }
//---Open buy---
/* Form a standard trade request to buy */
void my_expert::open_buy(void)
  {
   request.action=TRADE_ACTION_DEAL;
   request.symbol=_Symbol;
   request.volume=lots;
   request.price=SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   request.sl=request.price-sl*_Point;
   request.tp=0;
   request.deviation=10;
   request.type=ORDER_TYPE_BUY;
   request.type_filling=ORDER_FILLING_FOK;
   OrderSend(request,result);
   return;
  }
//---Open sell---
/* Form a standard trade request to sell */
void my_expert::open_sell(void)
  {
   request.action=TRADE_ACTION_DEAL;
   request.symbol=_Symbol;
   request.volume=lots;
   request.price=SymbolInfoDouble(Symbol(),SYMBOL_BID);
   request.sl=request.price+sl*_Point;
   request.tp=0;
   request.deviation=10;
   request.type=ORDER_TYPE_SELL;
   request.type_filling=ORDER_FILLING_FOK;
   OrderSend(request,result);
   return;
  }
//---Position modification---
void my_expert::position_modify(void)
  {
   if(PositionGetSymbol(0)==_Symbol)
     {     //If a position is for our symbol
      request.action=TRADE_ACTION_SLTP;
      request.symbol=_Symbol;
      request.deviation=10;
      //---If a buy position---
      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
        {
/* if distance from price to stop loss is more than trailing stop
   and the new stop loss is not less than the previous one */
         if(SymbolInfoDouble(Symbol(),SYMBOL_BID)-PositionGetDouble(POSITION_SL)>_Point*ts)
           {
            if(PositionGetDouble(POSITION_SL)<SymbolInfoDouble(Symbol(),SYMBOL_BID)-_Point*ts)
              {
               request.sl=SymbolInfoDouble(Symbol(),SYMBOL_BID)-_Point*ts;
               request.tp=PositionGetDouble(POSITION_TP);
               OrderSend(request,result);
              }
           }
        }
      //---If it is a sell position---
      else if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
        {
/*  if distance from price to stop loss is more than the trailing stop value
   and the new stop loss is not above the previous one. Or the stop loss from the moment of opening is equal to zero */
         if((PositionGetDouble(POSITION_SL)-SymbolInfoDouble(Symbol(),SYMBOL_ASK))>(_Point*ts))
           {
            if((PositionGetDouble(POSITION_SL)>(SymbolInfoDouble(Symbol(),SYMBOL_ASK)+_Point*ts)) ||
               (PositionGetDouble(POSITION_SL)==0))
              {
               request.sl=SymbolInfoDouble(Symbol(),SYMBOL_ASK)+_Point*ts;
               request.tp=PositionGetDouble(POSITION_TP);
               OrderSend(request,result);
              }
           }
        }
     }
  }
//+------------------------------------------------------------------
```

My humble obeisances to the author of the article **["Writing an Expert Advisor using the MQL5 Object-Oriented Approach](https://www.mql5.com/en/articles/116)**". What would I do without it! I recommend reading this article to anyone who is not very well versed in this evil, but extremely functional [Object-oriented programming.](https://www.mql5.com/en/docs/basis/oop)

Add the file with the class to the main code of the Expert Advisor? crate an object and initialize functions:

```
//+------------------------------------------------------------------+
//|                                                       Moving.mq5 |
//|                        Copyright 2011, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//---Include a file with the class---
#include <moving.mqh>
//---External Variables---
input int MA_RED_PERIOD=7; // The period of a slow MA
input int MA_YEL_PERIOD=2; // The period of a fast MA
input int STOP_LOSS=800;   // Stop loss
input int TRAL_STOP=800;   // Trailing stop
input double LOTS=0.1;     // Lot
//---Create an object---
my_expert expert;
//---Initialize the MqlDataTime structure---
MqlDateTime time;
int day_of_week;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---Initialize the EA
   expert.get_periods(MA_RED_PERIOD,MA_YEL_PERIOD);   // Set the MA periods
   expert.get_lot(LOTS);                              // Set the lot
   expert.get_stops(STOP_LOSS,TRAL_STOP);             // Set stop orders
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   TimeToStruct(TimeCurrent(),time);
   day_of_week=time.day_of_week;
   if(PositionsTotal()<1)
     {
      if(day_of_week==5 && expert.check_for_buy()==true){expert.open_buy();}
      else if(day_of_week==1 && expert.check_for_sell()==true){expert.open_sell();}
     }
   else expert.position_modify();
  }
//+------------------------------------------------------------------+
```

Done! I'd like to note some special features. To identify the days of the week at the software level, I used the [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) structure. First, we transform the current server time into a structured format. We obtain an index of the current day (1-Monday, ..., 5-Friday) and compare it with the value that we've set.

Trying out! In order not to burden you with tedious researches and extra digits, I'm bringing all of the results in the table.

Here it is:

> > > > > > ![Table 1. Summary of Buys on every day of the week](https://c.mql5.com/2/3/table1.png)
> > > > > >
> > > > > > Table 1. Summary of Buys on every day of the week
> > > > > >
> > > > > > ![Table 2. Summary of Sells on every day of the week](https://c.mql5.com/2/3/table2.png)
> > > > > >
> > > > > > Table 2. Summary of Sells on every day of the week

The best results are highlighted in green, the worst ones are orange.

I make a reservation, that after the above-described actions the system must ensure profit in combination with low relative drawdown, a good percentage of winning trades (here, the less trades the better) and a relatively high profit per trade.

Obviously, the most effective system is buying on Friday and selling on Monday. Combine both of these conditions:

```
if(PositionsTotal()<1){
      if(day_of_week==5&&expert.check_for_buy()==true){expert.open_buy();}
      else if(day_of_week==1&&expert.check_for_sell()==true){expert.open_sell();}}
   else expert.position_modify();
```

Now the Expert Advisor opens positions in both directions, but on strictly defined days. For clarity, I'll draw the diagrams obtained without and with the filter:

> > > > ![Figure 2. The results of EA testing without using a filter (EURUSD, H1, 01.01.2010-31.12.2010,)](https://c.mql5.com/2/3/Moving_Test_wo_filter.jpg)

> > > > > Figure 2. The results of EA testing without using a filter (EURUSD, H1, 01.01.2010-31.12.2010,)

> > > > ![Figure 3. The results of EA testing using the filter (EURUSD, H1, 01.01.2010-31.12.2010,)](https://c.mql5.com/2/3/Moving_test_with_filter.jpg)

> > > > > Figure 3. The results of EA testing using the filter (EURUSD, H1, 01.01.2010-31.12.2010)

How do you like the result? By using the filter the trading system became more stable. Before the modifications, the Expert Advisor was mainly increasing the balance in the first half of the testing period, after the "upgrade" it is increasing throughout the whole period.

We compare the reports:

> > > > > > > ![Table 3. Results of testing before and after using the filter](https://c.mql5.com/2/3/table3.png)

> > > > > > > Table 3. Results of testing before and after using the filter

The only distressing factor, which can not be ignored, is the fall of the net profit by almost 1000 USD (26%). But we are reducing the number of trades almost in 3,5 times, i.e. significantly reducing, firstly, the potential to make a negative trade and, secondly, the expenses for the spread (218\*2-62\*2=312 USD and it's only for EUR/USD). Winning percentage is increased to 57%, which is already significant. While profit per trade increases by 14% to 113 USD. As L. Williams would say: "This is the amount, which is worth trading!"

### Conclusion

Prices do not behave randomly - it is a fact. This fact can and should be used. I have given only one example, which is a tiny fraction of the innumerable variations and techniques that can improve the performance of your trading system. However, this diversity hides a vice. Not every filter can be integrated, so it must be choose carefully, thinking through all possible scenarios.

Do not forget that no matter how perfect the filter is, it weeds out profitable trades as well, i.e. your profit... Good luck!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/269](https://www.mql5.com/ru/articles/269)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/269.zip "Download all attachments in the single ZIP archive")

[explorer.mq5](https://www.mql5.com/en/articles/download/269/explorer.mq5 "Download explorer.mq5")(2.84 KB)

[moving.mq5](https://www.mql5.com/en/articles/download/269/moving.mq5 "Download moving.mq5")(2.28 KB)

[moving.mqh](https://www.mql5.com/en/articles/download/269/moving.mqh "Download moving.mqh")(6.98 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3366)**
(16)


![Alex](https://c.mql5.com/avatar/2009/11/4B070E32-CC84.jpg)

**[Alex](https://www.mql5.com/en/users/coaster)**
\|
7 Mar 2011 at 02:20

Suppose markets behave in a chaotic manner, like a coin. <br/ translate="no">

Therefore, when a [new bar](https://www.mql5.com/ru/articles/159 "Article 'Limitations and checks in experts'") appears, the price has an equal opportunity to go up or down, and the previous bars do not affect the current one in the slightest way. Idyll! Create a trading system, set Take Profit higher than Stop Loss (i.e. bring the expectation matrix into the positive zone), and you are done. It's simply breathtaking.

On the quote: I didn't understand anything at all!!! What was that??? Poetry? :) And where is the maths?

And second: It doesn't matter the probability of closing direction, if **more** **probable** small profits will have **less probable** fat losers. Your statistics rather depends on the risk management details omitted from it ;)

Too few examples to analyse in your article. You could say that the example you gave is a fitting to the story.

![snookeredman](https://c.mql5.com/avatar/avatar_na2.png)

**[snookeredman](https://www.mql5.com/en/users/snookeredman)**
\|
18 Mar 2011 at 16:18

Let's assume that markets behave like a coin, in a chaotic way.

Therefore, when a [new bar](https://www.mql5.com/ru/articles/159 "Article 'Limitations and checks in experts'") appears, the price has an equal opportunity to go up or down, and the previous bars do not affect the current one in the slightest way. Create a trading system, set Take Profit higher than Stop Loss (i.e. bring the expectation matrix into the positive zone), and you are done. It's breathtaking.

Author, have you ever thought that a moose can trigger even in every trade, despite the fact that the number of upward closes will be equal to the number of downward closes. So, even if it is known that there will be as many upward closes as downward closes, the strategy you described is not suitable.

![Alireza](https://c.mql5.com/avatar/avatar_na2.png)

**[Alireza](https://www.mql5.com/en/users/ali2e7a)**
\|
3 Apr 2011 at 22:37

Hi Тарачков,

thank you for your fine article,

correct me if i am wrong....

you extracted data from the past and run the filtered expert in the same time?(both in 2010)

if this is the case with all due respect i think there is no point in this filtering. This kind of filtering proves nothing because obviously makes the results better....

I am not a fan of totally random walk market but i think you should have used one period(for example 2005) to extract the data for filtering and run your filtered expert for the next year(2006) and continue till the last year then compare it with the original expert to see if there is any [correlations](https://www.mql5.com/en/docs/matrix/matrix_products/matrix_correlate "MQL5 Documentation: function Correlate") between the past price behavior and its future trends.

![Marco Lermer](https://c.mql5.com/avatar/avatar_na2.png)

**[Marco Lermer](https://www.mql5.com/en/users/tradermarco)**
\|
1 Dec 2013 at 10:45

It´s really a interesting articel, very good.

Thank you for this writing.

![Zeke Yaeger](https://c.mql5.com/avatar/2022/6/629E37C1-8BFC.jpg)

**[Zeke Yaeger](https://www.mql5.com/en/users/ozymandias_vr12)**
\|
12 May 2020 at 07:42

Thank you for your article, I'll test and try to improve your idea on some of my EA's.

Thank you again !!!


![Expert Advisor based on the "New Trading Dimensions" by Bill Williams](https://c.mql5.com/2/0/MQL5_alligator__1.png)[Expert Advisor based on the "New Trading Dimensions" by Bill Williams](https://www.mql5.com/en/articles/139)

In this article I will discuss the development of Expert Advisor, based on the book "New Trading Dimensions: How to Profit from Chaos in Stocks, Bonds, and Commodities" by Bill Williams. The strategy itself is well known and its use is still controversial among traders. The article considers trading signals of the system, the specifics of its implementation, and the results of testing on historical data.

![Implementation of Indicators as Classes by Examples of Zigzag and ATR](https://c.mql5.com/2/0/indicator_boxed.png)[Implementation of Indicators as Classes by Examples of Zigzag and ATR](https://www.mql5.com/en/articles/247)

Debate about an optimal way of calculating indicators is endless. Where should we calculate the indicator values - in the indicator itself or embed the entire logic in a Expert Advisor that uses it? The article describes one of the variants of moving the source code of a custom indicator iCustom right in the code of an Expert Advisor or script with optimization of calculations and modeling the prev\_calculated value.

![Econometric Approach to Analysis of Charts](https://c.mql5.com/2/0/econometrics.png)[Econometric Approach to Analysis of Charts](https://www.mql5.com/en/articles/222)

This article describes the econometric methods of analysis, the autocorrelation analysis and the analysis of conditional variance in particular. What is the benefit of the approach described here? Use of the non-linear GARCH models allows representing the analyzed series formally from the mathematical point of view and creating a forecast for a specified number of steps.

![Parallel Calculations in MetaTrader 5](https://c.mql5.com/2/0/parallel.png)[Parallel Calculations in MetaTrader 5](https://www.mql5.com/en/articles/197)

Time has been a great value throughout the history of mankind, and we strive not to waste it unnecessarily. This article will tell you how to accelerate the work of your Expert Advisor if your computer has a multi-core processor. Moreover, the implementation of the proposed method does not require the knowledge of any other languages besides MQL5.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/269&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068535204651399840)

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