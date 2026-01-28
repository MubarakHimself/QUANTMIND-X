---
title: Trading with Donchian Channels
url: https://www.mql5.com/en/articles/3146
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:54:03.368359
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/3146&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083195182163171017)

MetaTrader 5 / Tester


### Contents

- [Introduction](https://www.mql5.com/en/articles/3146#intro)
- [Operation Principle and Application](https://www.mql5.com/en/articles/3146#principles)
- [Developing a Trading System](https://www.mql5.com/en/articles/3146#development)
- [Implementing a Trading Strategy](https://www.mql5.com/en/articles/3146#implementation)
- [Testing](https://www.mql5.com/en/articles/3146#tests)
- [Conclusion](https://www.mql5.com/en/articles/3146#end)

### Introduction

The Donchian Channel is a technical indicator developed in early 1970s. First it was called a Moving Channel, and later was renamed after its creator Richard Donchian. This indicator measures the degree of market volatility based on a given price range using recent lows and highs. The Donchian channel is drawn as two lines, between which the price fluctuates. Sell or buy signals are formed when the price breaks the lower or upper channel border respectively. The author recommended to draw the channel using the price range equal to 20 (an average number of business days in a month) and the D1 timeframe.

### Operation Principle and Application of the Donchian Channel

I will not reinvent the wheel and will not create another program implementation of this indicator. I decided to use its [Donchian Channels System](https://www.mql5.com/en/code/2099) modification, which perfectly characterizes the trading style based on this system. In Fig.1 pink and blue candlesticks show the areas where the channel borders are broken, market entry is supposed in this places.

![](https://c.mql5.com/2/26/1_a.png)

Fig1. Operating principles and entry points based on the Donchian Channel.

Pay attention to red areas marked on the chart. They indicate the main disadvantage of the Donchian channel — the so-called false breakouts, after which the price rolls back into its range. Therefore, entering the market by only using the Donchian Channel without additional confirmation would be reckless. In order to have a clearer understanding of the indicator idea, let us analyze the parameters and implementation of this modification:

```
//+----------------------------------------------+
//|  Indicator input parameters                  |
//+----------------------------------------------+
input uint           DonchianPeriod=20;            // Averaging period
input Applied_Extrem Extremes=HIGH_LOW;            // Type of extrema
input uint           Shift=2;                      // Horizontal shift in bars
//+----------------------------------------------+
```

- **Averaging period** is the used price range.
- **Type of extrema** means the type of price values used in calculations. A classical setup is used by default, which means using candlestick high and low values.
- **Horizontal shift in bar** means shifting the channel.

Let us dwell on extrema types, because in this modification not only High and Low can be used. Below are possible options and software implementation:

```
//+----------------------------------------------+
//|  Declaration of enumeration                  |
//+----------------------------------------------+
enum Applied_Extrem //Type of extrema
  {
   HIGH_LOW,
   HIGH_LOW_OPEN,
   HIGH_LOW_CLOSE,
   OPEN_HIGH_LOW,
   CLOSE_HIGH_LOW
  };
```

- **HIGH\_LOW** is the classical application of candlestick Highs and Lows.
- **HIGH\_LOW\_OPEN —** in this interpretation, the upper channel border is drawn based on the average value between the _open price_ and the _candlestick highs_ in the selected price range. Candlestick lows are used for the lower border.
- **HIGH\_LOW\_CLOSE —** the upper channel border is drawn based on the average value between the _close price_ and the _candlestick highs_ in the selected price range. Candlestick lows are used for the lower border.
- **OPEN\_HIGH\_LOW —** the upper border is drawn based on the highest _open prices_ over the selected price range, and lowest ones are used for the lower border.
- **CLOSE\_HIGH\_LOW —** the upper border is drawn based on the highest _close prices_ over the selected price range, and lowest ones are used for the lower border.

The listing of implementations of different type of extreme values is available below:

```
for(bar=first; bar<rates_total && !IsStopped(); bar++)
     {
      switch(Extremes)
        {
         case HIGH_LOW:
            SsMax=high[ArrayMaximum(high,bar,DonchianPeriod)];
            SsMin=low[ArrayMinimum(low,bar,DonchianPeriod)];
            break;

         case HIGH_LOW_OPEN:
            SsMax=(open[ArrayMaximum(open,bar,DonchianPeriod)]+high[ArrayMaximum(high,bar,DonchianPeriod)])/2;
            SsMin=(open[ArrayMinimum(open,bar,DonchianPeriod)]+low[ArrayMinimum(low,bar,DonchianPeriod)])/2;
            break;

         case HIGH_LOW_CLOSE:
            SsMax=(close[ArrayMaximum(close,bar,DonchianPeriod)]+high[ArrayMaximum(high,bar,DonchianPeriod)])/2;
            SsMin=(close[ArrayMinimum(close,bar,DonchianPeriod)]+low[ArrayMinimum(low,bar,DonchianPeriod)])/2;
            break;

         case OPEN_HIGH_LOW:
            SsMax=open[ArrayMaximum(open,bar,DonchianPeriod)];
            SsMin=open[ArrayMinimum(open,bar,DonchianPeriod)];
            break;

         case CLOSE_HIGH_LOW:
            SsMax=close[ArrayMaximum(close,bar,DonchianPeriod)];
            SsMin=close[ArrayMinimum(close,bar,DonchianPeriod)];
            break;
        }
```

### Developing a Trading System

When developing a strategy, we need to take into account not only false breakouts, but also the fact that the Donchian channel is most often used in trend strategies. The entry signal is formed by the channel breakout, therefore, in order to eliminate false exits outside channel borders, we need to use at least one trend indicator for signal confirmation. We also need to determine exact conditions for entry, open position management, exit, and money management. Let us formulate the above conditions.

**1\. A confirmation signal**

The purpose of the article is not only to show examples of trading based on Donchian channels, but also to analyze them in terms of "survivability" in modern markets. Therefore, let us select a few confirmation indicators. Each of these indicators will form a tandem with the Donchian Channel. Thus we obtain several trading strategies based on the analyzed underlying strategy. To create tandems, I selected three confirmation signals from the following indicators:

- Average Directional Movement Index (ADX). A combination with this signal will allow us to evaluate the state and strength of the current trend, and then enter into the market upon the breakout of the channel borders.
- Moving Average Convergence/Divergence (MACD). MACD will monitor the current trend. When the price breaks the channel borders, we will check if this breakout is in the market direction or it is an accidental price spike (a false breakout).
- The third confirmation will be generated by two indicators: [Average Speed](https://www.mql5.com/en/code/1544) (the average price change speed in points/min) and [X4Period\_RSI\_Arrows](https://www.mql5.com/en/code/2237) (a semaphore indicator consisting of four RSIs with different periods).

**2\. Formalizing trading systems**

We need to find common parameters for these three strategies: let us select them so as to provide the maximum possible testing period. Therefore, let us define the parameters that will be controlled during testing:

- **Timeframe**. The timeframe selection option will allow us to test the strategies on different time periods, which can characterize specific market phases, including weak movement and correction, as well as long trends, which may be obvious on higher periods.
- **Money Management**. Several position sizing options depending on trade results will allow us to reveal whether re-investing is efficient or if it is more convenient to trade a fixed lot.
- **Open Position Management**. Several options of managing an open position will help us determine the profit as the percentage of the current favorable movement that we can fix.
- **Indicator Parameters**. Testing of selected strategies in different modes will help us find optimal parameters, with which our system will be efficient, as well as detect parameters that can make our system unprofitable.


Next, we need to formulate entry conditions for our trading strategies:

**#1\. Donchian Channel + ADX.**

Conditions for the system:

- The price breaks the upper or lower border of the Donchian channel.
- The main line of the ADX trend strength must be above the pre-set ADX Level.
- If the price breaks the channel border upwards, the DI+ line must be above DI-. DI- must be above DI+, if downwards.

![](https://c.mql5.com/2/26/2__21.png)

Fig.2. Market entry conditions of the strategy Donchian channel+ADX

**#2\. Donchian Channel + MACD.**

Conditions for the system:

- The price breaks the upper or lower border of the Donchian channel.
- Also, the histogram value is above zero and above the signal line for buying.
- The histogram value is below zero and below the signal line for selling.

![](https://c.mql5.com/2/26/3__20.png)

Fig.3. Market entry conditions of the strategy Donchian channel+MACD

**#3\. Donchian Channel + (Average Speed and X4Period\_RSI\_Arrows).**

Here are the system conditions:

- The price breaks the upper or lower border of the Donchian channel.
- The value of Average Speed must be greater than 1, and the semaphore RSI must mark at least 2 points above the candlestick for selling or below the candlestick for buying.

![](https://c.mql5.com/2/26/4__14.png)

Fig.4. Market entry conditions of the strategy Donchian channel+(Average Speed and X4Period\_RSI\_Arrows)

### Implementing a Trading Strategy

For the convenience of testing and optimization, all this types of strategies will be implemented in one Expert Advisor. The EA parameters will provide a selection of four strategies:

```
//+------------------------------------------------------------------+
//|  Declaration of enumerations of strategy types                   |
//+------------------------------------------------------------------+
enum Strategy
  {
   Donchian=0,
   Donchian_ADX,
   Donchian_MACD,
   Donchian_AvrSpeed_RSI
  };
```

- **Donchian**— a breakout strategy only using the Donchian channel.
- **Donchian\_ADX**— a strategy with the ADX trend strength indicator used as a filter.
- **Donchian\_MACD**— a strategy with the MACD oscillator used as a filter.
- **Donchian\_AvrSpeed\_RSI**— a strategy with the price rate of change and the semaphore RSI with different periods used as the filter.

Then we declare types of extreme values used for the Donchian channel, which we want to test:

```
//+------------------------------------------------------------------+
//| Declaration of enumerations of extreme types                     |
//+------------------------------------------------------------------+
enum Applied_Extrem
  {
   HIGH_LOW,
   HIGH_LOW_OPEN,
   HIGH_LOW_CLOSE,
   OPEN_HIGH_LOW,
   CLOSE_HIGH_LOW
  };
```

In addition, the Expert Advisor will provide for the possibility to use three money management systems:

```
//+------------------------------------------------------------------+
//|  Enumeration for lot calculation types                           |
//+------------------------------------------------------------------+
enum MarginMode
  {
   FREEMARGIN=0,     //MM Free Margin
   BALANCE,          //MM Balance
   LOT               //Constant Lot
  };
```

- **FREEMARGIN**— calculating the basic lot based on the free margin.
- **BALANCE**— lot calculation based on the current balance.
- **LOT —** fixed lot. It is a manually specified basic lot.

Open positions will be managed by a system based on the [Universal Trailing Stop](https://www.mql5.com/en/code/15848) Expert Advisor. On its basis, the _CTrailing_ class was developed, which is located in the **Trailing.mqh** file.

Position managing methods and input parameters are given below:

```
enum   TrallMethod
  {
   b=1,     //Based on candlestick extrema
   c=2,     //Using fractals
   d=3,     //Based on the ATR indicator
   e=4,     //Based on the Parabolic indicator
   f=5,     //Based on the MA indicator
   g=6,     //% of profit
   i=7,     //Using points
  };

//--- Trailing Stop parameters

input bool                 UseTrailing=true;                            //Use of trailing stop
input bool                 VirtualTrailingStop=false;                   //Virtual trailing stop
input TrallMethod          parameters_trailing=7;                       //Trailing method

input ENUM_TIMEFRAMES      TF_Tralling=PERIOD_CURRENT;                  //Indicator timeframe

input int                  StepTrall=50;                                //Trailing step (in points)
input int                  StartTrall=100;                              //Minimum trailing profit (in points)

input int                  period_ATR=14;                               //ATR period (method #3)

input double               step_PSAR=0.02;                              //PSAR step (method #4)
input double               maximum_PSAR=0.2;                            //Maximum PSAR (method #4)

input int                  ma_period=34;                                //MA period (method #5)
input ENUM_MA_METHOD       ma_method=MODE_SMA;                          //Averaging method (method #5)
input ENUM_APPLIED_PRICE   applied_price=PRICE_CLOSE;                   //Price type (method #5)

input double               PercentProfit=50;                            //Percent of profit (method #6)
```

For a more convenient display of selected parameters during visual testing and demo operation, I have added a panel using information provided in the [Graphical Interfaces](https://www.mql5.com/en/articles/2125) series. It provides the main type of selected settings:

- Type of strategy
- Type of trailing stop
- Type of money management
- Take Profit
- Stop Loss

For the display, the _CDonchianUI_ class was developed, which is available in the file **DonchianUI.mqh.** If desired, you can add your own values. The panel is shown in Fig.5.

![](https://c.mql5.com/2/26/2__22.png)

Fig.5. Information panel with the main modes of Donchian Expert.

### Testing

Before launching the testing, we need to determine its conditions. The basic conditions are summarized in the below table:

| Parameter | Value |
| --- | --- |
| Testing interval | 01.03.2016 — 01.03.2017 |
| Market | EURUSD/GOLD-6.17 |
| Execution | Every tick based on real ticks |
| Initial Deposit | 1000 USD |
| Leverage | 1:500 |
| Server | MetaQuotes |

The Expert Advisor is presented in the 4-in-1 format, so there is no need in describing the widest choice of all possible combinations of settings. Let us single out the main blocks of settings, which will be combined during testing, and some individual parameters from these blocks will be optimized. For better understanding, the below table provides the Expert Advisor modes that will be changed during testing:

| Expert Advisor block | Testing Modes |
| --- | --- |
| Money Management | Fixed lot/Balance |
| Working Timeframe | M30 - D1 |
| Position Management | Trailing Stop/None |
| Strategy Type | Donchian/Donchian+ADX/Donchian+MACD/Donchian+Avr.Speed+RSI |

I will explain why only 2 modes are used for Money Management: during testing, the deposit will only be loaded by our Expert Advisor, and only one position will exist at a time, so modes of _Balance_ and _Free margin_ will actually be the same.

**1\. The Donchian Channel trading strategy.**

We start with the strategy using purely Donchian channels.

![](https://c.mql5.com/2/28/006__1.gif)

Fig. 6. The Donchian trading strategy on EURUSD.

During strategy testing and optimization on EURUSD, we made the following conclusions:

- In the given range, the best results were obtained on M30-H1.
- The indicator operation timeframe was set to 10-12.
- In terms of the price used, the best results were obtained in the CLOSE\_HIGH\_LOW mode.
- The efficiency of the trailing stop function was very low compared to the mode without it.
- The general conclusion: A strategy based only on the Donchian channel behaved as a classical trend strategy, having a series of profitable trades during strong market movements.

```
//+------------------------------------------------------------------+
//| Expert Advisor input parameters                                  |
//+------------------------------------------------------------------+
sinput string              Inp_EaComment="Donchian Expert";             //EA comment
input double               Inp_Lot=0.01;                                //Basic lot
input MarginMode           Inp_MMode=LOT;                               //Money Management
input int                  Inp_MagicNum=555;                            //Magic
input int                  Inp_StopLoss=400;                            //Stop Loss (in points)
input int                  Inp_TakeProfit=600;                          //Take Profit (in points)
input int                  Inp_Deviation = 20;                          //Deviation
input ENUM_TIMEFRAMES      InpInd_Timeframe=PERIOD_M30;                 //Working timeframe
input bool                 InfoPanel=false;                             //Display of the information panel
//--- Donchian Channel System indicator parameters

input uint                 DonchianPeriod=10;                           //Channel period
input Applied_Extrem       Extremes=CLOSE_HIGH_LOW;                     //Type of extrema
//--- Selecting the strategy

input Strategy             CurStrategy=Donchian;                        //The selected strategy

//--- Trailing Stop parameters

input bool                 UseTrailing=false;                           //Use of trailing stop
```

As compared with the currency pair, testing on the GOLD-6.17 futures with the same period and timeframes - M30-H1, produced better results.

![](https://c.mql5.com/2/28/007__1.gif)

Fig.7. The Donchian trading strategy on GOLD-6.17.

```
//+------------------------------------------------------------------+
//| Expert Advisor input parameters                                  |
//+------------------------------------------------------------------+
sinput string              Inp_EaComment="Donchian Expert";             //EA comment
input double               Inp_Lot=0.01;                                //Basic lot
input MarginMode           Inp_MMode=LOT;                               //Money Management
input int                  Inp_MagicNum=555;                            //Magic
input int                  Inp_StopLoss=400;                            //Stop Loss (in points)
input int                  Inp_TakeProfit=600;                          //Take Profit (in points)
input int                  Inp_Deviation = 20;                          //Deviation
input ENUM_TIMEFRAMES      InpInd_Timeframe=PERIOD_M30;                 //Working timeframe
input bool                 InfoPanel=false;                             //Display of the information panel
//--- Donchian Channel System indicator parameters

input uint                 DonchianPeriod=12;                           //Channel period
input Applied_Extrem       Extremes=CLOSE_HIGH_LOW;                     //Type of extrema
//--- Selecting the strategy

input Strategy             CurStrategy=Donchian;                        //The selected strategy
....

//--- Trailing Stop parameters

input bool                 UseTrailing=true;                            //Use of trailing stop
input bool                 VirtualTrailingStop=false;                   //Virtual trailing stop
input TrallMethod          parameters_trailing=7;                       //Trailing method

input ENUM_TIMEFRAMES      TF_Tralling=PERIOD_CURRENT;                  //Indicator timeframe

input int                  StepTrall=50;                                //Stop level trailing step (in points)
input int                  StartTrall=100;                              //Minimum trailing profit (in points)
```

**2\. The Donchian Channel and ADX trading strategy.**

After testing the strategy with ADX used to filter the Donchian channel signals, we made the following conclusions:

- The EA executed less trades, which had been expected.
- The best working timeframe was again M30.
- The percentage of profitable trades was higher.
- The effective channel period was 18, and ADX period was 10.
- Again, the most effective price for the channel was CLOSE\_HIGH\_LOW.

![](https://c.mql5.com/2/28/008__1.gif)

Fig.8. Donchian + ADX trading strategy on EURUSD.

```
//+------------------------------------------------------------------+
//| Expert Advisor input parameters                                  |
//+------------------------------------------------------------------+
sinput string              Inp_EaComment="Donchian Expert";             //EA comment
input double               Inp_Lot=0.01;                                //Basic lot
input MarginMode           Inp_MMode=LOT;                               //Money Management
input int                  Inp_MagicNum=555;                            //Magic
input int                  Inp_StopLoss=400;                            //Stop Loss (in points)
input int                  Inp_TakeProfit=600;                          //Take Profit (in points)
input int                  Inp_Deviation = 20;                          //Deviation
input ENUM_TIMEFRAMES      InpInd_Timeframe=PERIOD_M30;                 //Working timeframe
input bool                 InfoPanel=false;                             //Display of the information panel
//--- Donchian Channel System indicator parameters

input uint                 DonchianPeriod=18;                           //Channel period
input Applied_Extrem       Extremes=CLOSE_HIGH_LOW;                     //Type of extrema
//--- Selecting the strategy

input Strategy             CurStrategy=Donchian_ADX;                    //The selected strategy
//--- ADX indicator parameter

input int                  ADX_period=10;
input double               ADX_level=20;
```

Testing on futures showed that on the H1 timeframe, with the Donchian channel period equal to 10 and ADX period equal to 8, the results are similar to the previous strategy. Again, the best results were obtained when drawing the channel based on Close prices.

![](https://c.mql5.com/2/28/009__1.gif)

Fig.9. The Donchian + ADX trading strategy on GOLD-6.17.

```
//+------------------------------------------------------------------+
//| Expert Advisor input parameters                                  |
//+------------------------------------------------------------------+
sinput string              Inp_EaComment="Donchian Expert";             //EA comment
input double               Inp_Lot=0.01;                                //Basic lot
input MarginMode           Inp_MMode=LOT;                               //Money Management
input int                  Inp_MagicNum=555;                            //Magic
input int                  Inp_StopLoss=400;                            //Stop Loss (in points)
input int                  Inp_TakeProfit=600;                          //Take Profit (in points)
input int                  Inp_Deviation = 20;                          //Deviation
input ENUM_TIMEFRAMES      InpInd_Timeframe=PERIOD_H1;                  //Working timeframe
input bool                 InfoPanel=false;                             //Display of the information panel
//--- Donchian Channel System indicator parameters

input uint                 DonchianPeriod=18;                           //Channel period
input Applied_Extrem       Extremes=CLOSE_HIGH_LOW;                     //Type of extrema
//--- Selecting the strategy

input Strategy             CurStrategy=Donchian_ADX;                    //The selected strategy
//--- ADX indicator parameter

input int                  ADX_period=8;
input double               ADX_level=20;

//--- Trailing Stop parameters

input bool                 UseTrailing=true;                            //Use of trailing stop
input bool                 VirtualTrailingStop=false;                   //Virtual trailing stop
input TrallMethod          parameters_trailing=7;                       //Trailing method

input ENUM_TIMEFRAMES      TF_Tralling=PERIOD_CURRENT;                  //Indicator timeframe

input int                  StepTrall=50;                                //Trailing step (in points)
input int                  StartTrall=100;                              //Minimum trailing profit (in points)
```

**3\. The Donchian Channel and MACD trading strategy.**

After testing the strategy with MACD used to filter the Donchian channel signals, we made the following conclusions:

- Once again, the best results were obtained on the H1 timeframe.
- Changing MACD parameters didn't have any serious effect.

![](https://c.mql5.com/2/28/010__1.gif)

Figure 10. Donchian + MACD trading strategy on EURUSD.

```
//+------------------------------------------------------------------+
//| Expert Advisor input parameters                                  |
//+------------------------------------------------------------------+
sinput string              Inp_EaComment="Donchian Expert";             //EA comment
input double               Inp_Lot=0.01;                                //Basic lot
input MarginMode           Inp_MMode=LOT;                               //Money Management
input int                  Inp_MagicNum=555;                            //Magic
input int                  Inp_StopLoss=400;                            //Stop Loss (in points)
input int                  Inp_TakeProfit=600;                          //Take Profit (in points)
input int                  Inp_Deviation = 20;                          //Deviation
input ENUM_TIMEFRAMES      InpInd_Timeframe=PERIOD_H1;                  //Working timeframe
input bool                 InfoPanel=true;                              //Display of the information panel
//--- Donchian Channel System indicator parameters

input uint                 DonchianPeriod=16;                           //Channel period
input Applied_Extrem       Extremes=CLOSE_HIGH_LOW;                     //Type of extrema
//--- Selecting the strategy

input Strategy             CurStrategy=Donchian_MACD;                   //The selected strategy

//--- MACD indicator parameters

input int                  InpFastEMA=12;                               //Fast EMA period
input int                  InpSlowEMA=26;                               //Slow EMA period
input int                  InpSignalSMA=9;                              //Signal SMA period
input ENUM_APPLIED_PRICE   InpAppliedPrice=PRICE_CLOSE;                 //Applied price
```

When tested with futures, the best result was obtained with the Donchian channel period equal to 10, as in the previous strategy. A version with the trailing stop function was more efficient, than without it.

![](https://c.mql5.com/2/28/011__1.gif)

Fig.11. The Donchian + MACD trading strategy on GOLD-6.17.

```
//+------------------------------------------------------------------+
//| Expert Advisor input parameters                                  |
//+------------------------------------------------------------------+
sinput string              Inp_EaComment="Donchian Expert";             //EA comment
input double               Inp_Lot=0.01;                                //Basic lot
input MarginMode           Inp_MMode=LOT;                               //Money Management
input int                  Inp_MagicNum=555;                            //Magic
input int                  Inp_StopLoss=400;                            //Stop Loss (in points)
input int                  Inp_TakeProfit=600;                          //Take Profit (in points)
input int                  Inp_Deviation = 20;                          //Deviation
input ENUM_TIMEFRAMES      InpInd_Timeframe=PERIOD_M30;                 //Working timeframe
input bool                 InfoPanel=true;                              //Display of the information panel
//--- Donchian Channel System indicator parameters

input uint                 DonchianPeriod=10;                           //Channel period
input Applied_Extrem       Extremes=CLOSE_HIGH_LOW;                     //Type of extrema
//--- Selecting the strategy

input Strategy             CurStrategy=Donchian_MACD;                   //The selected strategy

//--- MACD indicator parameters

input int                  InpFastEMA=12;                               //Fast EMA period
input int                  InpSlowEMA=26;                               //Slow EMA period
input int                  InpSignalSMA=9;                              //Signal SMA period
input ENUM_APPLIED_PRICE   InpAppliedPrice=PRICE_CLOSE;                 //Applied price
```

Before proceeding to the next strategy with a complex filter, we should draw an intermediate conclusion based on the above strategies using classical indicators. There was not much difference between the tested strategies on EURUSD. However, the ADX and MACD filters reduced the number of market entries and increased the final result. In connection with the specifics of the futures market and its volatility, which differs much from currency pairs, filtering of Donchian signals didn't provide changes. However, the total results on futures were much better compared to the currency pair.

**4\. The Donchian Channel and Average Speed+X4Period\_RSI\_Arrows.**

By optimizing the strategy of the Donchian channel combined with complex filter, we confirmed the conclusions about the strategy use in different trading conditions. Due to the low volatility of futures, the selectivity of this strategy was too high.

![](https://c.mql5.com/2/28/012__1.gif)

Fig.12. Donchian + Average Speed+X4Period\_RSI\_Arrows on EURUSD.

```
//+------------------------------------------------------------------+
//| Expert Advisor input parameters                                  |
//+------------------------------------------------------------------+
sinput string              Inp_EaComment="Donchian Expert";             //EA comment
input double               Inp_Lot=0.01;                                //Basic lot
input MarginMode           Inp_MMode=LOT;                               //Money Management
input int                  Inp_MagicNum=555;                            //Magic
input int                  Inp_StopLoss=400;                            //Stop Loss (in points)
input int                  Inp_TakeProfit=600;                          //Take Profit (in points)
input int                  Inp_Deviation = 20;                          //Deviation
input ENUM_TIMEFRAMES      InpInd_Timeframe=PERIOD_H1;                  //Working timeframe
input bool                 InfoPanel=true;                              //Display of the information panel
//--- Donchian Channel System indicator parameters

input uint                 DonchianPeriod=12;                           //Channel period
input Applied_Extrem       Extremes=CLOSE_HIGH_LOW;                     //Type of extrema
//--- Selecting the strategy

input Strategy             CurStrategy=Donchian_AvrSpeed_RSI;           //The selected strategy

//--- Average Speed indicator parameters

input int                  Inp_Bars=1;                                  //Number of bars
input ENUM_APPLIED_PRICE   Price=PRICE_CLOSE;                           //Applied price
input double               Trend_lev=2;                                 //Trend level
//--- The x4period_rsi_arrows indicator parameters

input uint                 RSIperiod1=7;                                //Period of RSI_1
input uint                 RSIperiod2=12;                               //Period of RSI_2
input uint                 RSIperiod3=18;                               //Period of RSI_3
input uint                 RSIperiod4=32;                               //Period of RSI_4
input ENUM_APPLIED_PRICE   Applied_price=PRICE_WEIGHTED;                //Applied price
input uint                 rsiUpperTrigger=62;                          //Overbought level
input uint                 rsiLowerTrigger=38;                          //Oversold level
//--- Trailing Stop parameters

input bool                 UseTrailing=false;                           //Use of trailing stop
```

### Summary

After testing the trading system based on the Donchian channel, we made the following conclusions and notes about the strategy features:

- Trading based on the channel behaves as a classical trend following strategy, and it can be improved by using additional filters.
- Testing results were better on the futures market.
- Unlike the classical calculation of the channel borders based on Highs and Lows, the CLOSE\_HIGH\_LOW (close prices) showed better results. This is connected with the testing symbol — the EURUSD market is characterized with price spikes in the form of long shadows. Such spikes distort channel borders, while close prices tend to be more objective.
- Also, the best results were obtained on timeframes М30 — H1, not daily ones.
- And the effective period ranges from 10 to 20 candlesticks.

### Conclusion

The archive attached below contains all described files properly sorted into folders. For a correct operation, you should save the **MQL5** folder to the terminal's root directory. We also used a graphical interface library **EasyAndFastGUI**, which is available in the related [article](https://www.mql5.com/en/articles/3104).

**Programs used in the article:**

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | DonchianExpert.mq5 | Expert Advisor | A complex Expert Advisor combining 4 strategies based on the Donchian Channel. |
| 2 | DonchianUI.mqh | Code Base | The GUI class |
| 3 | TradeFunctions.mqh | Code Base | The class of trade functions. |
| 4 | Trailing.mqh | Code Base | Open position management class. |
| 5 | average\_speed.mq5 | Indicator | Indicator of the average price change speed. |
| 6 | donchian\_channels\_system.mq5 | Indicator | The Donchian Channel displaying candlesticks that break the channel borders. |
| 7 | x4period\_rsi\_arrows.mq5 | Indicator | A semaphore indicator containing four RSIs with different period. |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/3146](https://www.mql5.com/ru/articles/3146)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3146.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/3146/mql5.zip "Download MQL5.zip")(625.3 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [A system of voice notifications for trade events and signals](https://www.mql5.com/en/articles/8111)
- [Quick Manual Trading Toolkit: Working with open positions and pending orders](https://www.mql5.com/en/articles/7981)
- [Quick Manual Trading Toolkit: Basic Functionality](https://www.mql5.com/en/articles/7892)
- [Multicurrency monitoring of trading signals (Part 5): Composite signals](https://www.mql5.com/en/articles/7759)
- [Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)
- [Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://www.mql5.com/en/articles/7600)
- [Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://www.mql5.com/en/articles/7528)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/203084)**
(21)


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
7 Mar 2020 at 18:39

**ruspbtrader:**

The article was written a long time ago - since then, the [MQL5 language](https://www.mql5.com/en/docs "MQL5 Programming Language Reference") has introduced system functions iHigh, iLow ...

Comment these functions in the code:

```
//+------------------------------------------------------------------+
//||
//+------------------------------------------------------------------+
//double iLow(string symbol,ENUM_TIMEFRAMES tf,int index)
// {
// if(index < 0) return(-1);
// double Arr[];
// if(CopyLow(symbol,tf, index, 1, Arr)>0) return(Arr[0]);
// else return(-1);
// }
//+------------------------------------------------------------------+
//||
//+------------------------------------------------------------------+
//double iHigh(string symbol,ENUM_TIMEFRAMES tf,int index)
// {
// if(index < 0) return(-1);
// double Arr[];
// if(CopyHigh(symbol,tf, index, 1, Arr)>0) return(Arr[0]);
// else return(-1);
// }
```

![stasTraider01](https://c.mql5.com/avatar/avatar_na2.png)

**[stasTraider01](https://www.mql5.com/en/users/stastraider01)**
\|
7 Apr 2020 at 11:44

I commented out the lines recommended above and the following errors appeared:

'TrailingStop' - unexpected token, probably type is missing? Trailing.mqh 114 12

'TrailingStop' - [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 Documentation: Predefined Macro Substitutions") already defined and has different type Trailing.mqh 114 12

![Sathish Collection](https://c.mql5.com/avatar/2023/7/64A2F065-FAC5.png)

**[Sathish Collection](https://www.mql5.com/en/users/sathishcollect)**
\|
8 Jul 2023 at 05:20

Hi,

Am getting this below kind of error. it's not fixed. Please share fixed code.

'ENUM\_SORT\_MODE:: [SORT\_ASCENDING](https://www.mql5.com/en/docs/matrix/matrix_types/matrix_enumerations " MQL5 Documentation: Matrix Enumerations")' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh165853

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh165853

Regards,

Sathish

![Sathish Collection](https://c.mql5.com/avatar/2023/7/64A2F065-FAC5.png)

**[Sathish Collection](https://www.mql5.com/en/users/sathishcollect)**
\|
8 Jul 2023 at 05:32

Hi Fexception,

Please fix below error at compiling for MT5.

'DonchianExpert.mq5'DonchianExpert.mq511

'TradeFunctions.mqh'TradeFunctions.mqh11

'PositionInfo.mqh'PositionInfo.mqh11

'Object.mqh'Object.mqh11

'StdLibErr.mqh'StdLibErr.mqh11

'Trailing.mqh'Trailing.mqh11

'DonchianUI.mqh'DonchianUI.mqh11

'WndEvents.mqh'WndEvents.mqh11

'Defines.mqh'Defines.mqh11

'WndContainer.mqh'WndContainer.mqh11

'Window.mqh'Window.mqh11

'ElementBase.mqh'ElementBase.mqh11

'Mouse.mqh'Mouse.mqh11

'Objects.mqh'Objects.mqh11

'Enums.mqh'Enums.mqh11

'Fonts.mqh'Fonts.mqh11

'LineChart.mqh'LineChart.mqh11

'ChartCanvas.mqh'ChartCanvas.mqh11

'CustomCanvas.mqh'CustomCanvas.mqh11

'Colors.mqh'Colors.mqh11

'FileBin.mqh'FileBin.mqh11

'File.mqh'File.mqh11

'Rect.mqh'Rect.mqh11

'ChartObjectsBmpControls.mqh'ChartObjectsBmpControls.mqh11

'ChartObject.mqh'ChartObject.mqh11

'ArrayInt.mqh'ArrayInt.mqh11

'Array.mqh'Array.mqh11

'ArrayDouble.mqh'ArrayDouble.mqh11

'ArrayString.mqh'ArrayString.mqh11

'ArrayObj.mqh'ArrayObj.mqh11

'ChartObjectSubChart.mqh'ChartObjectSubChart.mqh11

'ChartObjectsTxtControls.mqh'ChartObjectsTxtControls.mqh11

'Chart.mqh'Chart.mqh11

'MenuBar.mqh'MenuBar.mqh11

'Element.mqh'Element.mqh11

'MenuItem.mqh'MenuItem.mqh11

'' \- double quotes are neededMenuItem.mqh228119

'ContextMenu.mqh'ContextMenu.mqh11

'SeparateLine.mqh'SeparateLine.mqh11

'SimpleButton.mqh'SimpleButton.mqh11

'IconButton.mqh'IconButton.mqh11

'SplitButton.mqh'SplitButton.mqh11

'ButtonsGroup.mqh'ButtonsGroup.mqh11

'IconButtonsGroup.mqh'IconButtonsGroup.mqh11

'RadioButtons.mqh'RadioButtons.mqh11

'StatusBar.mqh'StatusBar.mqh11

'Tooltip.mqh'Tooltip.mqh11

'ListView.mqh'ListView.mqh11

'Scrolls.mqh'Scrolls.mqh11

'ComboBox.mqh'ComboBox.mqh11

'CheckBox.mqh'CheckBox.mqh11

'SpinEdit.mqh'SpinEdit.mqh11

'CheckBoxEdit.mqh'CheckBoxEdit.mqh11

'CheckComboBox.mqh'CheckComboBox.mqh11

'Slider.mqh'Slider.mqh11

'DualSlider.mqh'DualSlider.mqh11

'LabelsTable.mqh'LabelsTable.mqh11

'Table.mqh'Table.mqh11

'CanvasTable.mqh'CanvasTable.mqh11

'Pointer.mqh'Pointer.mqh11

'Tabs.mqh'Tabs.mqh11

'IconTabs.mqh'IconTabs.mqh11

'Calendar.mqh'Calendar.mqh11

'DateTime.mqh'DateTime.mqh11

'DropCalendar.mqh'DropCalendar.mqh11

'TreeItem.mqh'TreeItem.mqh11

'TreeView.mqh'TreeView.mqh11

'FileNavigator.mqh'FileNavigator.mqh11

'ColorButton.mqh'ColorButton.mqh11

'ColorPicker.mqh'ColorPicker.mqh11

'ProgressBar.mqh'ProgressBar.mqh11

'IndicatorBar.mqh'IndicatorBar.mqh11

'LineGraph.mqh'LineGraph.mqh11

'StandardChart.mqh'StandardChart.mqh11

'TextBox.mqh'TextBox.mqh11

'Keys.mqh'Keys.mqh11

'KeyCodes.mqh'KeyCodes.mqh11

'TimeCounter.mqh'TimeCounter.mqh11

'TextEdit.mqh'TextEdit.mqh11

'TextLabel.mqh'TextLabel.mqh11

'Picture.mqh'Picture.mqh11

'PicturesSlider.mqh'PicturesSlider.mqh11

'TimeEdit.mqh'TimeEdit.mqh11

'CheckBoxList.mqh'CheckBoxList.mqh11

'TrailingStop' - unexpected token, probably type is missing?Trailing.mqh11412

function 'CTrailing::TrailingStop' already defined and has different return typeTrailing.mqh11412

see declaration of function 'CTrailing::TrailingStop'Trailing.mqh8322

'iLow' - override system functionTrailing.mqh2418

'iHigh' - override system functionTrailing.mqh2518

identifier 'ENUM\_SORT\_MODE' already usedEnums.mqh1046

'advisor.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp16\\advisor.bmp"advisor.bmp11

'indicator.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp16\\indicator.bmp"indicator.bmp11

'script.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp16\\script.bmp"script.bmp11

'Close\_red.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\Close\_red.bmp"Close\_red.bmp11

'Close\_black.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\Close\_black.bmp"Close\_black.bmp11

'DropOn\_black.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\DropOn\_black.bmp"DropOn\_black.bmp11

'DropOn\_white.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\DropOn\_white.bmp"DropOn\_white.bmp11

'DropOff\_black.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\DropOff\_black.bmp"DropOff\_black.bmp11

'DropOff\_white.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\DropOff\_white.bmp"DropOff\_white.bmp11

'Help\_dark.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\Help\_dark.bmp"Help\_dark.bmp11

'Help\_light.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\Help\_light.bmp"Help\_light.bmp11

'CheckBoxOn\_min\_gray.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\CheckBoxOn\_min\_gray.bmp"CheckBoxOn\_min\_gray.bmp11

'CheckBoxOn\_min\_white.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\CheckBoxOn\_min\_white.bmp"CheckBoxOn\_min\_white.bmp11

'RArrow.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\RArrow.bmp"RArrow.bmp11

'RArrow\_white.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\RArrow\_white.bmp"RArrow\_white.bmp11

'DropOff.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\DropOff.bmp"DropOff.bmp11

'radio\_button\_on.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\radio\_button\_on.bmp"radio\_button\_on.bmp11

'radio\_button\_off.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\radio\_button\_off.bmp"radio\_button\_off.bmp11

'radio\_button\_on\_locked.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\radio\_button\_on\_locked.bmp"radio\_button\_on\_locked.bmp11

'radio\_button\_off\_locked.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\radio\_button\_off\_locked.bmp"radio\_button\_off\_locked.bmp11

'UArrow\_min.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\UArrow\_min.bmp"UArrow\_min.bmp11

'UArrow\_min\_dark.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\UArrow\_min\_dark.bmp"UArrow\_min\_dark.bmp11

'LArrow\_min.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\LArrow\_min.bmp"LArrow\_min.bmp11

'LArrow\_min\_dark.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\LArrow\_min\_dark.bmp"LArrow\_min\_dark.bmp11

'DArrow\_min.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\DArrow\_min.bmp"DArrow\_min.bmp11

'DArrow\_min\_dark.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\DArrow\_min\_dark.bmp"DArrow\_min\_dark.bmp11

'RArrow\_min.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\RArrow\_min.bmp"RArrow\_min.bmp11

'RArrow\_min\_dark.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\RArrow\_min\_dark.bmp"RArrow\_min\_dark.bmp11

'CheckBoxOn.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\CheckBoxOn.bmp"CheckBoxOn.bmp11

'CheckBoxOff.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\CheckBoxOff.bmp"CheckBoxOff.bmp11

'CheckBoxOn\_locked.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\CheckBoxOn\_locked.bmp"CheckBoxOn\_locked.bmp11

'CheckBoxOff\_locked.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\CheckBoxOff\_locked.bmp"CheckBoxOff\_locked.bmp11

'SpinInc.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\SpinInc.bmp"SpinInc.bmp11

'SpinInc\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\SpinInc\_blue.bmp"SpinInc\_blue.bmp11

'SpinDec.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\SpinDec.bmp"SpinDec.bmp11

'SpinDec\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\SpinDec\_blue.bmp"SpinDec\_blue.bmp11

'SORT\_ASCEND' - improper enumerator cannot be usedTable.mqh22088

'SORT\_ASCEND' - improper enumerator cannot be usedTable.mqh168680

'pointer\_x\_rs.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_x\_rs.bmp"pointer\_x\_rs.bmp11

'pointer\_x\_rs\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_x\_rs\_blue.bmp"pointer\_x\_rs\_blue.bmp11

'pointer\_y\_rs.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_y\_rs.bmp"pointer\_y\_rs.bmp11

'pointer\_y\_rs\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_y\_rs\_blue.bmp"pointer\_y\_rs\_blue.bmp11

'pointer\_xy1\_rs.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_xy1\_rs.bmp"pointer\_xy1\_rs.bmp11

'pointer\_xy1\_rs\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_xy1\_rs\_blue.bmp"pointer\_xy1\_rs\_blue.bmp11

'pointer\_xy2\_rs.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_xy2\_rs.bmp"pointer\_xy2\_rs.bmp11

'pointer\_xy2\_rs\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_xy2\_rs\_blue.bmp"pointer\_xy2\_rs\_blue.bmp11

'pointer\_x\_rs\_rel.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_x\_rs\_rel.bmp"pointer\_x\_rs\_rel.bmp11

'pointer\_y\_rs\_rel.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_y\_rs\_rel.bmp"pointer\_y\_rs\_rel.bmp11

'pointer\_x\_scroll.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_x\_scroll.bmp"pointer\_x\_scroll.bmp11

'pointer\_x\_scroll\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_x\_scroll\_blue.bmp"pointer\_x\_scroll\_blue.bmp11

'pointer\_y\_scroll.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_y\_scroll.bmp"pointer\_y\_scroll.bmp11

'pointer\_y\_scroll\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_y\_scroll\_blue.bmp"pointer\_y\_scroll\_blue.bmp11

'pointer\_text\_select.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\pointer\_text\_select.bmp"pointer\_text\_select.bmp11

'SORT\_ASCEND' - improper enumerator cannot be usedCanvasTable.mqh30988

'SORT\_ASCEND' - improper enumerator cannot be usedCanvasTable.mqh163386

'LeftTransp\_black.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\LeftTransp\_black.bmp"LeftTransp\_black.bmp11

'LeftTransp\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\LeftTransp\_blue.bmp"LeftTransp\_blue.bmp11

'RArrow\_black.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\RArrow\_black.bmp"RArrow\_black.bmp11

'RArrow\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\RArrow\_blue.bmp"RArrow\_blue.bmp11

'calendar\_today.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\calendar\_today.bmp"calendar\_today.bmp11

'calendar\_drop\_on.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\calendar\_drop\_on.bmp"calendar\_drop\_on.bmp11

'calendar\_drop\_off.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\calendar\_drop\_off.bmp"calendar\_drop\_off.bmp11

'calendar\_drop\_locked.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\calendar\_drop\_locked.bmp"calendar\_drop\_locked.bmp11

'RArrow\_rotate\_black.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\RArrow\_rotate\_black.bmp"RArrow\_rotate\_black.bmp11

'RArrow\_rotate\_white.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\RArrow\_rotate\_white.bmp"RArrow\_rotate\_white.bmp11

'folder\_w10.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp16\\folder\_w10.bmp"folder\_w10.bmp11

'text\_file\_w10.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp16\\text\_file\_w10.bmp"text\_file\_w10.bmp11

'arrow\_down.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp16\\arrow\_down.bmp"arrow\_down.bmp11

'arrow\_up.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp16\\arrow\_up.bmp"arrow\_up.bmp11

'stop\_gray.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp16\\stop\_gray.bmp"stop\_gray.bmp11

'no\_image.bmp' as resource "::Images\\EasyAndFastGUI\\Icons\\bmp64\\no\_image.bmp"no\_image.bmp11

'ArrowLeft.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\ArrowLeft.bmp"ArrowLeft.bmp11

'ArrowLeft\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\ArrowLeft\_blue.bmp"ArrowLeft\_blue.bmp11

'ArrowRight.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\ArrowRight.bmp"ArrowRight.bmp11

'ArrowRight\_blue.bmp' as resource "::Images\\EasyAndFastGUI\\Controls\\ArrowRight\_blue.bmp"ArrowRight\_blue.bmp11

'iLow' - ambiguous call to overloaded function with the same parametersTrailing.mqh27836

could be one of 2 function(s)Trailing.mqh27836

built-in: double iLow(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh27836

double iLow(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2418

'iHigh' - ambiguous call to overloaded function with the same parametersTrailing.mqh28736

could be one of 2 function(s)Trailing.mqh28736

built-in: double iHigh(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh28736

double iHigh(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2518

'iLow' - ambiguous call to overloaded function with the same parametersTrailing.mqh29919

could be one of 2 function(s)Trailing.mqh29919

built-in: double iLow(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh29919

double iLow(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2418

'iLow' - ambiguous call to overloaded function with the same parametersTrailing.mqh29950

could be one of 2 function(s)Trailing.mqh29950

built-in: double iLow(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh29950

double iLow(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2418

'iLow' - ambiguous call to overloaded function with the same parametersTrailing.mqh30019

could be one of 2 function(s)Trailing.mqh30019

built-in: double iLow(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh30019

double iLow(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2418

'iLow' - ambiguous call to overloaded function with the same parametersTrailing.mqh30050

could be one of 2 function(s)Trailing.mqh30050

built-in: double iLow(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh30050

double iLow(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2418

'iLow' - ambiguous call to overloaded function with the same parametersTrailing.mqh30119

could be one of 2 function(s)Trailing.mqh30119

built-in: double iLow(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh30119

double iLow(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2418

'iLow' - ambiguous call to overloaded function with the same parametersTrailing.mqh30150

could be one of 2 function(s)Trailing.mqh30150

built-in: double iLow(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh30150

double iLow(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2418

'iLow' - ambiguous call to overloaded function with the same parametersTrailing.mqh30323

could be one of 2 function(s)Trailing.mqh30323

built-in: double iLow(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh30323

double iLow(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2418

'iHigh' - ambiguous call to overloaded function with the same parametersTrailing.mqh31719

could be one of 2 function(s)Trailing.mqh31719

built-in: double iHigh(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh31719

double iHigh(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2518

'iHigh' - ambiguous call to overloaded function with the same parametersTrailing.mqh31751

could be one of 2 function(s)Trailing.mqh31751

built-in: double iHigh(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh31751

double iHigh(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2518

'iHigh' - ambiguous call to overloaded function with the same parametersTrailing.mqh31819

could be one of 2 function(s)Trailing.mqh31819

built-in: double iHigh(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh31819

double iHigh(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2518

'iHigh' - ambiguous call to overloaded function with the same parametersTrailing.mqh31851

could be one of 2 function(s)Trailing.mqh31851

built-in: double iHigh(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh31851

double iHigh(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2518

'iHigh' - ambiguous call to overloaded function with the same parametersTrailing.mqh31919

could be one of 2 function(s)Trailing.mqh31919

built-in: double iHigh(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh31919

double iHigh(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2518

'iHigh' - ambiguous call to overloaded function with the same parametersTrailing.mqh31951

could be one of 2 function(s)Trailing.mqh31951

built-in: double iHigh(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh31951

double iHigh(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2518

'iHigh' - ambiguous call to overloaded function with the same parametersTrailing.mqh32123

could be one of 2 function(s)Trailing.mqh32123

built-in: double iHigh(const string,ENUM\_TIMEFRAMES,int)Trailing.mqh32123

double iHigh(string,ENUM\_TIMEFRAMES,int)Trailing.mqh2518

expression not booleanChartCanvas.mqh12953

'method' - undeclared identifierMenuItem.mqh228110

'method' - some operator expectedMenuItem.mqh228110

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh27546

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh27546

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh27546

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh27546

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh97530

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh97530

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh97530

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh97530

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh1070113

'ENUM\_SORT\_MODE:: [SORT\_DESCENDING](https://www.mql5.com/en/docs/matrix/matrix_types/matrix_enumerations " MQL5 Documentation: Matrix Enumerations")' will be used instead of 'ENUM\_SORT\_MODE::SORT\_DESCEND'Table.mqh1070113

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh107129

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh107129

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh107129

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh107129

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh107329

'ENUM\_SORT\_MODE::SORT\_DESCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_DESCEND'Table.mqh107329

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh107329

'ENUM\_SORT\_MODE::SORT\_DESCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_DESCEND'Table.mqh107329

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh108147

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh108147

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh170353

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh170353

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'Table.mqh171153

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'Table.mqh171153

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh44458

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh44458

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh44458

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh44458

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh105030

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh105030

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh105030

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh105030

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh1360113

'ENUM\_SORT\_MODE::SORT\_DESCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_DESCEND'CanvasTable.mqh1360113

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh136129

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh136129

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh136129

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh136129

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh136329

'ENUM\_SORT\_MODE::SORT\_DESCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_DESCEND'CanvasTable.mqh136329

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh136329

'ENUM\_SORT\_MODE::SORT\_DESCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_DESCEND'CanvasTable.mqh136329

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh165053

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh165053

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh165853

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh165853

implicit conversion from 'enum ENUM\_SORT\_MODE' to 'enum ENUM\_SORT\_MODE'CanvasTable.mqh220444

'ENUM\_SORT\_MODE::SORT\_ASCENDING' will be used instead of 'ENUM\_SORT\_MODE::SORT\_ASCEND'CanvasTable.mqh220444

28 errors, 25 warnings2926

Regards,

Sathish

![Guru Maximux](https://c.mql5.com/avatar/2023/5/6461E4DA-DB63.png)

**[Guru Maximux](https://www.mql5.com/en/users/gurumaximux)**
\|
18 Jun 2024 at 18:29

rename to enum ENUM\_SORT\_MODE\_

![MQL5 Cookbook - Creating a ring buffer for fast calculation of indicators in a sliding window](https://c.mql5.com/2/26/Fon.png)[MQL5 Cookbook - Creating a ring buffer for fast calculation of indicators in a sliding window](https://www.mql5.com/en/articles/3047)

The ring buffer is the simplest and the most efficient way to arrange data when performing calculations in a sliding window. The article describes the algorithm and shows how it simplifies calculations in a sliding window and makes them more efficient.

![Cross-Platform Expert Advisor: Order Manager](https://c.mql5.com/2/28/Expert_Advisor_Introduction__2.png)[Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

This article discusses the creation of an order manager for a cross-platform expert advisor. The order manager is responsible for the entry and exit of orders or positions entered by the expert, as well as for keeping an independent record of such trades that is usable for both versions.

![Cross-Platform Expert Advisor: Signals](https://c.mql5.com/2/28/Cross_Platform_Expert_Advisor.png)[Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)

This article discusses the CSignal and CSignals classes which will be used in cross-platform expert advisors. It examines the differences between MQL4 and MQL5 on how particular data needed for evaluation of trade signals are accessed to ensure that the code written will be compatible with both compilers.

![How Long Is the Trend?](https://c.mql5.com/2/27/MQL5-avatar-TrendTime-001.png)[How Long Is the Trend?](https://www.mql5.com/en/articles/3188)

The article highlights several methods for trend identification aiming to determine the trend duration relative to the flat market. In theory, the trend to flat rate is considered to be 30% to 70%. This is what we'll be checking.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lmoqrohpoxzvksryjmxivjfscikwyjxt&ssn=1769252041850666197&ssn_dr=0&ssn_sr=0&fv_date=1769252041&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F3146&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20with%20Donchian%20Channels%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925204173968776&fz_uniq=5083195182163171017&sv=2552)

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