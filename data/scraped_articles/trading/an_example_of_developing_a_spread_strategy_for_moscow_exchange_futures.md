---
title: An Example of Developing a Spread Strategy for Moscow Exchange Futures
url: https://www.mql5.com/en/articles/2739
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:18:14.128109
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=baiscppkfnjaxpouegysdnzgabqhimgn&ssn=1769181492472301660&ssn_dr=0&ssn_sr=0&fv_date=1769181492&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2739&back_ref=https%3A%2F%2Fwww.google.com%2F&title=An%20Example%20of%20Developing%20a%20Spread%20Strategy%20for%20Moscow%20Exchange%20Futures%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918149263793395&fz_uniq=5069325736331903796&sv=2552)

MetaTrader 5 / Examples


The MetaTrader 5 platform allows developing and testing trading robots that simultaneously trade multiple financial instruments. The built-in Strategy Tester automatically downloads required tick history from the broker's server taking into account contract specifications, so the developer does not need to do that manually. This makes it possible to easily and reliably reproduce trading environment conditions, including even millisecond intervals between the arrival of ticks on different symbols. In this article we will demonstrate the development and testing of a spread strategy on two [Moscow Exchange futures](https://www.mql5.com/go?link=http://www.moex.com/en/derivatives/select.aspx "http://moex.com/en/derivatives/select.aspx").

### Negative Correlation of Assets: Si and RTS

Si-M.Y and RTS-M.Y futures are traded on Moscow Exchange. These futures types are tightly correlated. Here M.Y means contract expiration date:

- M — the number of the month
- Y — the last two digits of the year


Si is a futures contract on US dollar/Russian ruble exchange rate, RTS is a futures contract on the RTS index expressed in US dollars. The RTS index includes stocks of Russian companies, the prices of which are expressed in rubles, USD/RUR fluctuations also affect index fluctuations expressed in US dollars. Price charts show that when one asset grows, the second asset usually falls.

![](https://c.mql5.com/2/25/SiRi_charts__1.png)

For a better visualization, we have drawn a standard deviation channel on these charts.

### Calculating Linear Regression between Si and RTS

We can express correlation between the two assets using a linear regression equation Y(X)=A(X)+B. Let's create a script _CalcShowRegression\_script.mq5_, which takes two arrays of close prices, calculates coefficients and shows the distribution diagram with a regression line straight on the chart.

Regression coefficients are calculated using an [ALGLIB](https://www.mql5.com/en/code/1146) function, and the values are drawn using [graphic classes](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) of the standard library.

![](https://c.mql5.com/2/25/CalcRegression__7.png)

### Drawing an indicator of spread between Si and a synthetic sequence

We have received linear regression coefficients and can draw a synthetic chart of type Y(RTS) = A\*RTS+B. Let us call the difference between the source asset and the synthetic sequence "a spread". This difference will vary at each bar from negative to positive values.

In order to visualize the spread, let us create the _TwoSymbolsSpread\_Ind.mql5_ indicator that displays the histogram of spread on the last 500 bars. Positive values are drawn in blue, negative values are yellow.

![](https://c.mql5.com/2/25/Spread_Two_Symbols_2__2.png)

The indicator updates and writes to the [Experts journal](https://www.metatrader5.com/en/terminal/help/start_advanced/journal "https://www.metatrader5.com/en/terminal/help/start_advanced/journal") linear regression coefficients when a new bar is opened. Moreover, it waits till the new candlestick opens on both instruments, including Si and RTS. This way the indicator ensures correctness and accuracy of calculations.

### Creating a linear regression channel on the spread channel over the last 100 bars

The spread indicator shows that the difference between the Si futures and the synthetic symbol changes from time to time. In order to evaluate the current spread, let us create the _SpreadRegression\_Ind.mq5_ indicator (spread with a linear regression on it) that draws a trend line on a spread chart. The line parameters are calculated using linear regression. Let us launch the two indicators on a chart for debugging.

![](https://c.mql5.com/2/25/Spread_N_LR.png)

The slope of the red trend line changes depending on the spread value on the last 100 bars. Now we have a minimum of required data and we can try to build a trading system.

### Strategy \#1: Linear regression slope change on a spread chart

Spread values in the _TwoSymbolsSpread\_Ind.mql5_ indicator are calculated as the difference between Si and Y(RTS)=A\*RTS + B. You can easily check it by running the indicator in the [debugging](https://www.metatrader5.com/en/metaeditor/help/development/debug "https://www.metatrader5.com/en/metaeditor/help/development/debug") mode (F5 key).

![](https://c.mql5.com/2/25/debug_ind__5.png)

Let us create a simple Expert Advisor that would monitor change of slope of the linear regression attached to a spread chart. Line slope is the A coefficient in the equation: Y=A\*X+B. If trend is positive on the spread chart, A>0. If trend is negative, A<0. The linear regression is calculated using the last 100 values of the spread chart. Here is a part of the Expert Advisor code _Strategy1\_AngleChange\_EA.mq5._

#include <Trade\\Trade.mqh>

//+------------------------------------------------------------------+

//\| Spread strategy type                                             \|

//+------------------------------------------------------------------+

enum SPREAD\_STRATEGY

{

    BUY\_AND\_SELL\_ON\_UP,  // Buy 1-st, Sell 2-nd

    SELL\_AND\_BUY\_ON\_UP,  // Sell 1-st, Buy 2-nd

};

//---

inputintLR\_length=100;                     // Number of bars for a regression on spread

inputintSpread\_length=500;                 // number of bars for spread calculation

inputENUM\_TIMEFRAMESperiod=PERIOD\_M5;           // Time-frame

inputstringsymbol1="Si-12.16";                // The first symbol of the pair

inputstringsymbol2="RTS-12.16";               // The second symbol of the pair

inputdoubleprofit\_percent=10;                 // Percent of profit to lock in

input SPREAD\_STRATEGY strategy=SELL\_AND\_BUY\_ON\_UP; // Type of a spread strategy

//\-\-\- Indicator handles

int ind\_spreadLR,ind,ind\_2\_symbols;

//\-\-\- A class for trading operations

CTrade trade;

//+------------------------------------------------------------------+

//\| Expert tick function                                             \|

//+------------------------------------------------------------------+

voidOnTick()

{

//\-\-\- The A coefficient of the linear regression slope on the spread chart Y(X)=A\*X+B

staticdouble Spread\_A\_prev=0;

if(isNewBar())

PrintFormat("New bar %s opened at %s",\_Symbol,TimeToString(TimeCurrent(),TIME\_DATE\|TIME\_SECONDS));

//\-\-\- Wait for indicator data to refresh, because it works on two symbols

if(BarsCalculated(ind\_spreadLR)==Bars(\_Symbol,\_Period))

      {

//\-\-\- Get linear regression values on the spread chart for bars with indices 1 and 2 ("yesterday" and "the day before yesterday")

double LRvalues\[\];

double Spread\_A\_curr;

int copied=CopyBuffer(ind\_spreadLR,1,1,2,LRvalues);

if(copied!=-1)

         {

//\-\-\- Linear regression coefficient on the last completed ("yesterday") bar

          Spread\_A\_curr=LRvalues\[1\]-LRvalues\[0\];

//\-\-\- If the linear regression slope has changed, the product of current and previous value is less than zero

if(Spread\_A\_curr\*Spread\_A\_prev<0)

            {

PrintFormat("Slope of LR changed, Spread\_A\_curr=%.2f, Spread\_A\_prev=%.2f: %s",

                         Spread\_A\_curr,Spread\_A\_prev,TimeToString(TimeCurrent(),TIME\_SECONDS));

//\-\-\- If we have no open positions, enter the market with both symbols

if(PositionsTotal()==0)

                DoTrades(Spread\_A\_curr-Spread\_A\_prev>0,strategy,symbol1,1,symbol2,1);

//\-\-\- If there are open positions, reverse them

else

                ReverseTrades(symbol1,symbol2);

            }

//\-\-\- LR slope has not changed, check the floating profit - isn't it time to close?

else

            {

double profit=AccountInfoDouble(ACCOUNT\_PROFIT);

double balance=AccountInfoDouble(ACCOUNT\_BALANCE);

if(profit/balance\*100>=profit\_percent)

               {

//\-\-\- Required floating profit level reached, take it

                trade.PositionClose(symbol1);

                trade.PositionClose(symbol2);

               }

            }

//\-\-\- Remember trend direction to compare at the opening of a new bar

          Spread\_A\_prev=Spread\_A\_curr;

         }

      }

}

In order to eliminate the necessity to make assumptions on what to buy and what to sell when trend changes, let us add an external parameter that allows reversing trading rules:

input SPREAD\_STRATEGY strategy=SELL\_AND\_BUY\_ON\_UP; // Type of a spread strategy

Now we can start Expert Advisor testing and debugging.

### Testing the trading Strategy \#1

The [visual testing mode](https://www.metatrader5.com/en/terminal/help/algotrading/visualization "https://www.metatrader5.com/en/terminal/help/algotrading/visualization") suits best for debugging. Set the required data using the menu Tools-Settings-Debug:

1. Symbol
2. TimeFrame
3. Testing interval
4. Execution
5. Deposit

6. Tick Generation Mode

The recommended mode for exchange instruments is " **Every tick based on real ticks**". In this case the EA will be tested using recorded history data, and final results will be very close to real trading conditions.

**The MetaTrader 5 trade server automatically collects and stores all ticks received from an exchange** and sends the whole tick history to the terminal upon the first request.

![](https://c.mql5.com/2/26/Testing_settings__2.png)

This debugging mode allows executing the testing process in the visual mode while checking the values of any variables where necessary using breakpoints. Indicators used in the robot will be automatically loaded to a chart, there is no need to attach them manually.

![](https://c.mql5.com/2/26/Break_point_visual__3.png)

Once the EA code is debugged, we can optimize parameters.

### Optimization of trading Strategy \#1

The _Strategy1\_AngleChange\_EA.mq5_ Expert Advisor has several external parameters that can be configured by optimization (highlighted in yellow):

inputintLR\_length=100;                     // Number of bars for a regression on spread

inputintSpread\_length=500;                 // number of bars for spread calculation

inputdoubleprofit\_percent=10;                 // Percent of profit to lock in

input SPREAD\_STRATEGY strategy=SELL\_AND\_BUY\_ON\_UP; // Type of a spread strategy

In this case we will only optimize _profit\_percent_ for two versions of the strategy, in order to understand whether there is a difference between them. In other words, we fix the value of the _strategy_ parameter and optimize based on _profit\_percent_ from 0.2 to 3.0%, in order to see the overall picture for the two methods to trade line slope changes.

For the **BUY\_AND\_SELL\_ON\_UP rule** (buy the first asset, sell the second one), when the line slope changes from negative to positive, optimization does not show good results. In general, this market entry method does not look attractive, we get more losses during the two-month testing.

![](https://c.mql5.com/2/25/fig1__7.gif)

The **SELL\_AND\_BUY\_ON\_UP rule** (sell the first asset, buy the second one) gives better optimization result: 5 of 15 test runs show some profit.

![](https://c.mql5.com/2/25/fig2__5.gif)

Optimization was performed on history data from August 1 to September 30, 2016 (two months interval). In general, both trading variants do not look promising. Perhaps the problem is that the parameter that we used for entries, i.e. the trend line slope over the last 100 bars, is a lagging indicator. Let's try to develop a second version of the strategy.

### Strategy \#2: Spread sign change on a completed bar

In the second strategy, we analyze change of spread sign. We will only analyze values of completed bars, i.e. we will check it at the opening of "today's" bar. If spread on the "day before yesterday"'s bar was negative, and it was positive on the "yesterday"'s bar, we can assume that the spread has turned up. The code still provides for the possibility to trade spread change in any direction. We can change entry direction using the _strategy_ parameter. Here is a block from the _Strategy2\_SpreadSignChange\_EA.mq5_ code:

//+------------------------------------------------------------------+

//\| Expert tick function                                             \|

//+------------------------------------------------------------------+

voidOnTick()

{

//\-\-\- Previous spread value as difference between Symbol1 and Y(Symbol2)=A\*Symbol2+B

staticdouble Spread\_prev=0;

if(isNewBar())

PrintFormat("New bar %s opened at %s",\_Symbol,TimeToString(TimeCurrent(),TIME\_DATE\|TIME\_SECONDS));

//\-\-\- Wait for indicator data to refresh, because it works on two symbols

if(BarsCalculated(ind\_spreadLR)==Bars(\_Symbol,\_Period))

      {

//\-\-\- Get spread values for bars with indices 1 and 2 ("yesterday" and "the day before yesterday")

double SpreadValues\[\];

int copied=CopyBuffer(ind\_spreadLR,0,1,2,SpreadValues);

double Spread\_curr=SpreadValues\[1\];

if(copied!=-1)

         {

//\-\-\- If the spread sign has changed, the product of current and previous value is less than zero

if(Spread\_curr\*Spread\_prev<0)

            {

PrintFormat("Spread sign changed, Spread\_curr=%.2f, Spread\_prev=%.2f: %s",

                         Spread\_curr,Spread\_prev,TimeToString(TimeCurrent(),TIME\_SECONDS));

//\-\-\- If we have no open positions, enter the market with both symbols

if(PositionsTotal()==0)

                DoTrades(Spread\_curr>0,strategy,symbol1,1,symbol2,1);

//\-\-\- There are open positions, reverse them

else

                ReverseTrades(symbol1,symbol2);

            }

//\-\-\- Spread sign has not changed, check the floating profit - isn't it time to close?

else

            {

double profit=AccountInfoDouble(ACCOUNT\_PROFIT);

double balance=AccountInfoDouble(ACCOUNT\_BALANCE);

if(profit/balance\*100>=profit\_percent)

               {

//\-\-\- Required floating profit level reached, take it

                trade.PositionClose(symbol1);

                trade.PositionClose(symbol2);

               }

            }

//\-\-\- Remember spread value to compare at the opening of a new bar

          Spread\_prev=Spread\_curr;

         }

      }

}

First we debug the EA in the visual testing mode, and then run optimization by _profit\_percent_, like we did for the first strategy. Results:

![](https://c.mql5.com/2/25/fig3__7.gif)

![](https://c.mql5.com/2/25/fig4__5.gif)

As you can see, the "sell first and buy second asset" rule applied to the second strategy also gives disappointing testing results. The "Buy first and sell the second asset" gives more losses in all test runs.

Let us try to create the third variant of the strategy.

### Strategy \#3: Spread sign change on the current bar and confirmation over N ticks

Two previous strategies only worked at bar opening, i.e. they only analyzed changes on fully completed bars. Now we will try to work inside the current bar. Let us analyze spread changes on every tick, and if the spread sign on the completed bar and that on the current bar differ, we should assume that the spread direction has changed.

Also, the spread sign change should be stable over the last N ticks, which will help filter false signals. We need to add the external parameter _ticks\_for\_trade_=10 into our Expert Advisor. If the spread sign is negative on the last 10 ticks, and it was positive on the previous bar, the EA should enter the market. Here is the OnTick() function of the _Strategy3\_SpreadSignOnTick\_EA.mq5_ Expert Advisor.

//+------------------------------------------------------------------+

//\| Expert tick function                                             \|

//+------------------------------------------------------------------+

voidOnTick()

{

if(isNewBar())

PrintFormat("New bar %s opened at %s",\_Symbol,TimeToString(TimeCurrent(),TIME\_DATE\|TIME\_SECONDS));

//\-\-\- Wait for indicator data to refresh, because it works on two symbols

if(BarsCalculated(ind\_spreadLR)==Bars(\_Symbol,\_Period))

      {

//\-\-\- Get spread values on the current (today) and previous (yesterday) bar

double SpreadValues\[\];

int copied=CopyBuffer(ind\_spreadLR,0,0,2,SpreadValues);

double Spread\_curr=SpreadValues\[1\]; // spread on the current incomplete bar

double Spread\_prev=SpreadValues\[0\]; // spread on the previous complete bar

if(copied!=-1)

         {

//\-\-\- If the spread sign change is stable on the last ticks\_for\_trade ticks

if(SpreadSignChanged(Spread\_curr,Spread\_prev,ticks\_for\_trade))

            {

PrintFormat("Spread sign changed, Spread\_curr=%.2f, Spread\_prev=%.2f: %s",

                         Spread\_curr,Spread\_prev,TimeToString(TimeCurrent(),TIME\_SECONDS));

//\-\-\- Show on the chart the values of the last ticks\_for\_trade trades of both symbols

             ShowLastTicksComment(ticks\_for\_trade);

//\-\-\- If we have no open positions, enter the market with both symbols

if(PositionsTotal()==0)

                DoTrades(Spread\_curr>0,strategy,symbol1,1,symbol2,1);

//\-\-\- There are open positions, reverse them

else

                ReverseTrades(Spread\_curr>0,positionstype,symbol1,symbol2);

            }

//\-\-\- Spread sign has not changed, check the floating profit - isn't it time to close?

else

            {

double profit=AccountInfoDouble(ACCOUNT\_PROFIT);

double balance=AccountInfoDouble(ACCOUNT\_BALANCE);

if(profit/balance\*100>=profit\_percent)

               {

//\-\-\- Required floating profit level reached, take it

                trade.PositionClose(symbol1);

                trade.PositionClose(symbol2);

                positionstype=0;

               }

            }

         }

      }

    }

In this Expert Advisor we have added the ShowLastTicksComment() function which displays on the chart the values of last N ticks of both symbols once the signal appears. This allows us to visually test the strategy and monitor tick changes with a millisecond precision.

![](https://c.mql5.com/2/25/showticks2__3.png)

Now we start the same optimization options applied in the first two strategies, and receive the following results:

"Buying the first asset and selling the seconds one"

![](https://c.mql5.com/2/25/fig5__7.gif)

"Selling the first asset and buying the seconds one"

![](https://c.mql5.com/2/25/fig6__5.gif)

Results of such a simple optimization are not much improved.

### Strategy 4: Spread reaches a preset percent value

Now let us create the fourth and the last strategy for spread trading. It will be as simple as the three previous strategies: a trade signal appears when spread value riches the specified percent of the first asset price — _spread\_delta_. Tick handler OnInit() has changed slightly, here is how it looks like in _Strategy4\_SpreadDeltaPercent\_EA.mq5_.

//+------------------------------------------------------------------+

//\| Expert tick function                                             \|

//+------------------------------------------------------------------+

voidOnTick()

{

if(isNewBar())

PrintFormat("New bar %s opened at %s",\_Symbol,TimeToString(TimeCurrent(),TIME\_DATE\|TIME\_SECONDS));

//\-\-\- Wait for indicator data to refresh, because it works on two symbols

if(BarsCalculated(ind\_spreadLR)==Bars(\_Symbol,\_Period))

      {

//\-\-\- Get spread value on the current (today) bar

double SpreadValues\[\];

int copied=CopyBuffer(ind\_spreadLR,0,0,1,SpreadValues);

double Spread\_curr=SpreadValues\[0\]; // spread on the current incomplete bar

if(copied!=-1)

         {

MqlTick tick;

SymbolInfoTick(symbol1,tick);

double last=tick.last;

double spread\_percent=Spread\_curr/last\*100;

//\-\-\- If spread % reached the spread\_delta value

if(MathAbs(spread\_percent)>=spread\_delta)

            {

PrintFormat("Spread reached %.1f%% (%G) %s",

                         spread\_percent,TimeToString(TimeCurrent(),TIME\_SECONDS),

                         Spread\_curr);

//\-\-\- If we have no open positions, enter the market with both symbols

if(PositionsTotal()==0)

                DoTrades(Spread\_curr,strategy,symbol1,1,symbol2,1);

//\-\-\- There are open positions, reverse them

else

                ReverseTrades(Spread\_curr,positionstype,symbol1,symbol2);

            }

//\-\-\- Spread is within acceptable range, check the floating profit - isn't it time to close?

else

            {

double profit=AccountInfoDouble(ACCOUNT\_PROFIT);

double balance=AccountInfoDouble(ACCOUNT\_BALANCE);

if(profit/balance\*100>=profit\_percent)

               {

//\-\-\- Required floating profit level reached, take it

                trade.PositionClose(symbol1);

                trade.PositionClose(symbol2);

                positionstype=0;

               }

            }

         }

      }

}

Positions will also be closed when specified profit percent  profit\_percent=2 is reached. It is a fixed value this time. Start optimization using the _spread\_delta_ parameter in the range of 0.1 to 1%.

"Buying the first asset and selling the seconds one"

![](https://c.mql5.com/2/25/fig7__7.gif)

"Selling the first asset and buying the seconds one"

![](https://c.mql5.com/2/25/fig8__5.gif)

This time the first "Buy first and sell the second asset" rule looks much better than the second rule. You can further optimize using other parameters.

### MetaTrader 5 — Trading strategy developing environment

In this article, we have considered 4 simple strategies for spread trading. Testing and optimization results produced by these strategies should not be used as a guide to action, because they were obtained in a limited interval and can be random to some extent. The original purpose of this article is to show how easy and convenient it is to test and debug trading ideas using MetaTrader 5.

The MetaTrader 5 tester provides the following convenient features for the developers of automated trading systems:

- automatic download of tick history of all symbols used in the Expert Advisor
- visual indicator and strategy debugging mode, which includes visualization of trades, trading history and Experts journal
- automatic launch of all indicators used in the EA in the visual testing mode
- testing strategies using real recorded history data and reproduction of real trading environment
- multi-threaded optimization of parameters using a custom target function
- use of thousands of testing agents [for faster optimization](https://www.mql5.com/en/articles/669)

- [visualization of results](https://www.mql5.com/en/articles/403) of optimization in accordance with custom rules

- testing strategies that trade multiple instruments with synchronization of ticks up to a millisecond
- debugging strategies straight during the testing process — you can set breakpoints to check the values of required variables and run a step-by-step testing.


In this article, the Strategy Tester was used as a research tool to find the right direction. This was done as an optimization using one parameter, which allowed to make quick qualitative conclusions. You can add new rules, modify existing ones and run full EA optimization. To speed up calculations, use the [MQL5 Cloud Network](https://cloud.mql5.com/en "https://cloud.mql5.com/en") which is specially designed for the MetaTrader 5 platform.

### Important notes on the Strategies

Normally when searching for symbols for spread calculation, price increment is used instead of absolute price values. It means Delta\[i\]=Close\[i\]-Close\[i-1\] is calculated instead of the Close\[i\] series.

For a balanced trading, you should select volume for each spread symbol. In this article, we only used a 1-lot volume for each symbol.

Current settings in Si and RTS contract specifications are used during testing. It is important to mention that:

- the RTS-12.16 futures is based on the US dollar,

- the price of the RTS-12.16 futures tick is set every day on Moscow Exchange

- the tick value is equal to 0.2 of the [indicative USD/RUB exchange rate](https://www.mql5.com/go?link=http://www.moex.com/en/index/rtsusdcur.aspx "http://moex.com/en/index/rtsusdcur.aspx").

![](https://c.mql5.com/2/25/rts_ticksize__2.png)

Information on index calculation is available on MOEX site at [http://fs.moex.com/files/4856](https://www.mql5.com/go?link=http://fs.moex.com/files/4856 "http://fs.moex.com/files/4856"). Therefore, you should remember that the results of optimization in the Strategy Tester depend on the dollar rate at the time of testing. The article contains screenshots with optimization results as of October 25, 2016.

The code is written for execution under perfect performance conditions: it does not contain handling of order sending results, handling of errors connected with connection loss, and it does not take into account commission and slippage.

The futures liquidity and chart filling are improved by the end to contract expiration. The code does not contain an explicit handling of the situation, when quotes of one symbol are received, and whole bars are missed on the second symbol (no trading on the exchange for any reason). However, indicators used in the EA wait for synchronization of bars of both symbols to calculate the spread value, and write these events into journal.

The article does not contain analysis of statistics of spread deviation from average values, which is required for creating more reliable trading rules.

Market Depth analysis is not used, because the order book is not simulated in the MetaTrader 5 Strategy Tester.

**Attention:** Indicators used in this article dynamically recalculate linear regression coefficients for creating spread charts and the trend line. Therefore, by the end of testing, the appearance of charts and indicator values will differ from those displayed during the testing process.

Run the indicators or EAs attached below in the visual testing mode, to see the process in real time.

**Related articles:**

- [Testing trading strategies on real ticks](https://www.mql5.com/en/articles/2612)
- [How to quickly develop and debug a trading strategy](https://www.mql5.com/en/articles/2661)
- [Creating a trading robot for Moscow Exchange. Where to start?](https://www.mql5.com/en/articles/2513)
- [How to secure yourself and your Expert Advisor while trading on Moscow Exchange](https://www.mql5.com/en/articles/1683)
- [MQL5 vs QLUA - why trading operations in MQL5 are up to 28 times faster?](https://www.mql5.com/en/articles/2635)
- [The checks a trading robot must pass before publication in the Market](https://www.mql5.com/en/articles/2555)

**Programs used in the article:**

| \# | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | CalcShowRegression\_script.mq5 | Script | Calculates linear regression coefficients and draws a point chart with a trend line (using the СGraphic class and an Alglib function) |
| 2 | TwoSymbolsSpread\_Ind.mql5 | Indicator | The indicator draws a spread histogram on two symbols |
| 3 | SpreadRegression\_Ind.mq5 | Indicator | The indicator draws a spread chart and a regression line on it |
| 4 | Strategy1\_AngleChange\_EA.mq5 | Expert Advisor | Strategy #1. Trading based on the sign change of the A linear regression coefficient in the equation Y=A\*X+B. Analysis and entries only at the opening of a new bar |
| 5 | Strategy2\_SpreadSignChange\_EA.mq5 | Expert Advisor | Strategy #2. Trading based on sign change of the spread value. Analysis and entries only at the opening of a new bar |
| 6 | Strategy3\_SpreadSignOnTick\_EA.mq5 | Expert Advisor | Strategy #3. Trading based on sign change of the spread value. Analysis and entries within the current bar, the sign change should be stable on the last N ticks |
| 7 | Strategy4\_SpreadDeltaPercent\_EA.mq5 | Expert Advisor | Strategy #4. Trading based on reaching a certain percent value of spread. Analysis and entries within the current bar on the first received tick |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2739](https://www.mql5.com/ru/articles/2739)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2739.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/2739/mql5.zip "Download MQL5.zip")(22.9 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/169276)**
(44)


![Roman Konopelko](https://c.mql5.com/avatar/2015/8/55DF2310-9E28.png)

**[Roman Konopelko](https://www.mql5.com/en/users/konopelko)**
\|
16 Oct 2017 at 15:10

**Maxim Dmitrievsky:**

How can I get F-statistic and p -value from built reg model through alglib, maybe someone has already done it? There are only AVGerr and RMSerr there

There is a [static method](https://www.mql5.com/en/docs/basis/oop/staticmembers "MQL5 Documentation: Static class members") for F-statistic in CAlglib class:

```
//+------------------------------------------------------------------+
//| Two-sample F-test|
//| This test checks three hypotheses about dispersions of the given |
//| samples. |
//| Input parameters:|
//| X - sample 1. Array whose index goes from 0 to N-1. ||
//| N - sample size.|
//| Y - sample 2. Array whose index goes from 0 to M-1. |
//| M - sample size.|
//| Output parameters:|
//| BothTails - p-value for two-tailed test. ||
//|If BothTails is less than the given |
//| significance level the null hypothesis is |
//| rejected.|
//| LeftTail - p-value for left-tailed test. ||
//|If LeftTail is less than the given |
//| significance level, the null hypothesis is |
//| rejected.|
//| RightTail - p-value for right-tailed test. ||
//|If RightTail is less than the given |
//| significance level the null hypothesis is |
//| rejected.|
//+------------------------------------------------------------------+
static void CAlglib::FTest(const double &x[],const int n,const double &y[],
                           const int m,double &bothTails,double &leftTail,
                           double &rightTail)
```

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
16 Oct 2017 at 16:10

**Roman Konopelko:**

The CAlglib class has a [static method](https://www.mql5.com/en/docs/basis/oop/staticmembers "MQL5 Documentation: Static class members") for the F test:

exactly, thanks.. it's a bit strange that they didn't build different tests into the regression class, for example to evaluate each of the traits, but you can do it yourself

![junhtv](https://c.mql5.com/avatar/2018/12/5C1729FD-9402.jpg)

**[junhtv](https://www.mql5.com/en/users/junhtv)**
\|
16 Dec 2018 at 15:00

[![](https://c.mql5.com/3/258/aaaa__3.jpg)](https://c.mql5.com/3/258/aaaa__2.jpg "https://c.mql5.com/3/258/aaaa__2.jpg")

![junhtv](https://c.mql5.com/avatar/2018/12/5C1729FD-9402.jpg)

**[junhtv](https://www.mql5.com/en/users/junhtv)**
\|
16 Dec 2018 at 15:27

m\_graphic.YAxis().ValuesWith(45);  // Length of values along the Y axis error here

![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
16 Feb 2025 at 11:42

```
double            m_x[];
...
CMatrixDouble xy(size,2);
xy[0].Set(0, m_x[0]);
```

'operator\[\]' - constant variable cannot be passed as reference CalcShowRegression\_script.mq5 103 6

how to cure this?

![Graphical interfaces X: Advanced management of lists and tables. Code optimization (build 7)](https://c.mql5.com/2/25/Graphic-interface_11-2.png)[Graphical interfaces X: Advanced management of lists and tables. Code optimization (build 7)](https://www.mql5.com/en/articles/2943)

The library code needs to be optimized: it should be more regularized, which is — more readable and comprehensible for studying. In addition, we will continue to develop the controls created previously: lists, tables and scrollbars.

![Graphical Interfaces X: Time control, List of checkboxes control and table sorting (build 6)](https://c.mql5.com/2/25/jxd7fn-zcrx8k35mvp-3ii6s7g5j1-II-001.png)[Graphical Interfaces X: Time control, List of checkboxes control and table sorting (build 6)](https://www.mql5.com/en/articles/2897)

Development of the library for creating graphical interfaces continues. The Time and List of checkboxes controls will be covered this time. In addition, the CTable class now provides the ability to sort data in ascending or descending order.

![Auto detection of extreme points based on a specified price variation](https://c.mql5.com/2/25/math_compass.png)[Auto detection of extreme points based on a specified price variation](https://www.mql5.com/en/articles/2817)

Automation of trading strategies involving graphical patterns requires the ability to search for extreme points on the charts for further processing and interpretation. Existing tools do not always provide such an ability. The algorithms described in the article allow finding all extreme points on charts. The tools discussed here are equally efficient both during trends and flat movements. The obtained results are not strongly affected by a selected timeframe and are only defined by a specified scale.

![Patterns available when trading currency baskets](https://c.mql5.com/2/25/WPR_02.png)[Patterns available when trading currency baskets](https://www.mql5.com/en/articles/2816)

Following up our previous article on the currency baskets trading principles, here we are going to analyze the patterns traders can detect. We will also consider the advantages and the drawbacks of each pattern and provide some recommendations on their use. The indicators based on Williams' oscillator will be used as analysis tools.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/2739&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069325736331903796)

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