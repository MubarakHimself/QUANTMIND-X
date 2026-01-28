---
title: Build Self Optimizing Expert Advisors in MQL5  (Part 3): Dynamic Trend Following and Mean Reversion Strategies
url: https://www.mql5.com/en/articles/16856
categories: Trading Systems
relevance_score: 10
scraped_at: 2026-01-22T17:22:13.474218
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/16856&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6369019141164999827)

MetaTrader 5 / Examples


Algorithmic trading strategies based on moving averages stand out from most trading strategies due to their ability to keep our trading applications aligned with the long-term market trend. Unfortunately, when markets are ranging and not following a true long-term trend, our once reliable trend following strategies will do us more harm than good. Understanding how markets switch between trend regimes and range bound regimes can significantly help us make more effective use of our moving-average based strategies.

Typically, traders tend to classify markets as either being currently range bound or currently trending before deciding which strategies are applicable for that particular trading day. There is limited literature aimed at discussing how markets switch between these 2 regimes. Instead of perceiving markets as a static environment that exists in one of 2 states. We want to frame financial markets as dynamic environments, that are constantly alternating between 2 possible states, without truly settling in either one. Our goal, today, is to build a single dynamic trading strategy that can independently detect when the underlying market regime changes from trending to range bound or vice versa.

This single dynamic strategy will hopefully substitute the traditional paradigms of having 1 strategy for each market condition. The proposed strategy relies on calculating a trading channel on a roll forward basis. The underlying idea behind the strategy is the belief that there is a boundary separating trends moves and range bound moves. And by paying attention to where price levels are relative to our boundary, we can make better informed trades. The upper and lower boundary of our channel is calculated by adding (upper boundary) and subtracting (lower boundary) a multiple of the ATR value from the average value of the moving average indicator. We will extensively discuss the strategy in the subsequent sections of the article.

However, the reader should understand that the channel is being dynamically calculated daily using standard technical indicators included in every installation of MetaTrader 5. Our initial strategy was a simple trend following strategy, that would enter long positions whenever price levels closed above the 100 period moving average, and short positions otherwise. After the trades were opened, they were subsequently managed using a fixed stop loss and take profit. Over a 4-year back test on M1 data on the EURUSD, only 52% of the trades placed by our initial trading strategy were profitable. Our proposed dynamic channel-based rules increased our proportion of winning trades to 86% over the same 4-year period on the M1 time frame, without using any curve fitting or AI techniques.

The findings suggest that effort spent learning how to better estimate the boundary, is worth the reward obtained. By relaxing our need to categorically classify markets into well-defined boxes, and instead trying to follow the natural rhythm of the market, we find that our trading application was able to ascend to impressive proportions of winning trades. Additionally, given the simple layout chosen for our coding style, the reader will find it easy to extend the template provided to them with their unique understanding of the market.

### **Overview of The Trading Strategy**

As we explained in the introduction of the article, trading strategies based on moving averages are especially popular among traders because they keep us aligned with the long-term market trend. A rather exceptional example of this principle in action is provided in Fig 1 below. The screenshot is taken from the Daily EURUSD exchange rate, and shows a bullish market trend that began late in Nov 2016 and ran until April 2018. An impressive run by any measure. Note that the moving average indicator also gave confirmation that price levels have been in a long-standing uptrend.

Generally, speaking, we can see that our long positions could have been set up whenever price levels closed above our moving average and our short positions could’ve been set up whenever price levels closed beneath our moving average.

![Trend regime](https://c.mql5.com/2/109/Screenshot_2025-01-07_181812__1.png)

Fig 1: An example of our moving average defined trend following system in action

The example given in Fig 1 shows the main strength of moving average strategies, when employed in markets that are trending. However, when the market is devoid of any long-term trend, such as the market conditions illustrated in Fig 2, these simple trend following strategies will not be effective.

![Range bound market regime](https://c.mql5.com/2/109/Screenshot_2025-01-07_182325__1.png)

Fig 2: An example of an extended unprofitable period for using trend following systems

The strategy proposed in this article can handle both market conditions exceptionally well without the need for any additional indicators, or complex trading logic for detecting when to change strategies, and which strategy to change to for that matter. Let's get started learning what it will take to make our simple moving average trading strategy, dynamic and self-adjusting.

### Getting Started In MQL5

Our trading system is built up of a collection of different parts:

| System Part | Intended Purpose |
| --- | --- |
| System constants | These are constants hidden away from the end user, that are intended to keep our system's behavior consistent across both of our back tests to avoid bias, or unintentional changes that will break the trading logic. |
| Libraries | In our application, we only imported the trade library to help open and manage our positions. |
| Global variables | These variables are meant to store indicator values, price levels and other data in the system that will be shared by different individual parts of the system. |
| System event handlers | Functions such as OnTick() are system-driven event handlers which help us perform our tasks in an organized fashion. |
| Custom functions | These are functions tailored to our specific needs for successfully following the moving average defined trend. |

To get started, we will first define our system constants. Note that we will not change these values in any of the subsequent versions we will build of our trading application.

```
//+------------------------------------------------------------------+
//|                              Dynamic Moving Average Strategy.mq5 |
//|                                               Gamuchirai Ndawana |
//|                    https://www.mql5.com/en/users/gamuchiraindawa |
//+------------------------------------------------------------------+
#property copyright "Gamuchirai Ndawana"
#property link      "https://www.mql5.com/en/users/gamuchiraindawa"
#property version   "1.00"

//+------------------------------------------------------------------+
//| System constants                                                 |
//+------------------------------------------------------------------+
#define MA_PRICE    PRICE_CLOSE           //Moving average applied price
#define MA_MODE    MODE_EMA               //Moving average type
#define MA_SHIFT   0                      //Moving average shift
#define ATR_PERIOD 14                     //Period for our ATR
#define TF_1       PERIOD_D1              //Hihger order timeframe
#define TRADING_MA_PERIOD  100            //Moving average period
#define SL_WIDTH 1.5                      //How wide should the stop loss be?
#define CURRENT_VOL 0.1                   //Trading volume
```

Next we will import one of the most commonly used libraries in the MQL5 API, the Trade Library. It is essential for easily managing our positions.

```
//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;
```

We shall also need to create global variables for storing market prices and technical indicator values.

```
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
int    ma_handler,trading_ma_handler,atr_handler;
double ma[],atr[],trading_ma[];
double ask,bid;
```

The body of our MQL5 application is built up by event handlers. These handlers are called whenever a new event happens in our terminal, such as new prices being offered or the application being removed from the chart. Some events are triggered by our end user, and some events are triggered by the trade server. The OnInit() handler is triggered when our end user launches our trading application. It will call a dedicated function we have designed to initialize our global variables.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   setup();
//---
   return(INIT_SUCCEEDED);
  }
```

When our end user removes the trading application from the chart, the OnDeinit() handler is called. We will use this handler to free up the system memory resources we are no longer consuming.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
IndicatorRelease(atr_handler);
IndicatorRelease(ma_handler);
IndicatorRelease(trading_ma_handler);
  }
```

The OnTick() handler is triggered by the trader server, not the end user. It is called whenever we receive updated price information. We will call dedicated functions to update our technical indicators and store the new price information, and afterward, we if we have no open positions, we will look to open a position.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   update();
   if(PositionsTotal() == 0)
      find_setup();
  }
//+------------------------------------------------------------------+
```

Our rules for opening positions under the trend following strategy are easy to grasp. If price levels are above the 100 Period moving average, we will open long positions with fixed stop losses. Otherwise, if price levels are beneath the moving average, we will open short positions.

```
//+------------------------------------------------------------------+
//| Find setup                                                       |
//+------------------------------------------------------------------+
void find_setup(void)
  {
//Buy on rallies
   if(iClose(Symbol(),PERIOD_CURRENT,0) > trading_ma[0])
      Trade.Buy(CURRENT_VOL,Symbol(),ask,(bid - (atr[0] * SL_WIDTH)),(bid + (atr[0] * SL_WIDTH)),"");

   if(iClose(Symbol(),PERIOD_CURRENT,0) < trading_ma[0])
      Trade.Sell(CURRENT_VOL,Symbol(),bid,(ask + (atr[0] * SL_WIDTH)),(ask - (atr[0] * SL_WIDTH)),"");
  }
```

The function called by OnInit() handler to set up our technical indicators.

```
//+------------------------------------------------------------------+
//| Setup our global variables                                       |
//+------------------------------------------------------------------+
void setup(void)
  {
   atr_handler =  iATR(Symbol(),TF_1,ATR_PERIOD);
   trading_ma_handler  =  iMA(Symbol(),PERIOD_CURRENT,TRADING_MA_PERIOD,MA_SHIFT,MA_MODE,MA_PRICE);
  }
```

Our Update() function will be called by the OnTick() handler to update our global variables.

```
//+------------------------------------------------------------------+
//| Update                                                           |
//+------------------------------------------------------------------+
void update(void)
  {
   static datetime time_stamp;
   datetime current_time = iTime(Symbol(),PERIOD_CURRENT,0);
   ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);

   if(time_stamp != current_time)
     {
      time_stamp = current_time;
      CopyBuffer(atr_handler,0,0,1,atr);
      CopyBuffer(trading_ma_handler,0,0,1,trading_ma);
     }
  }
//+------------------------------------------------------------------+
```

This is the current state of our trading application in its current form.

```
//+------------------------------------------------------------------+
//|                              Dynamic Moving Average Strategy.mq5 |
//|                                               Gamuchirai Ndawana |
//|                    https://www.mql5.com/en/users/gamuchiraindawa |
//+------------------------------------------------------------------+
#property copyright "Gamuchirai Ndawana"
#property link      "https://www.mql5.com/en/users/gamuchiraindawa"
#property version   "1.00"

//+------------------------------------------------------------------+
//| System constants                                                 |
//+------------------------------------------------------------------+
#define MA_PRICE    PRICE_CLOSE           //Moving average applied price
#define MA_MODE    MODE_EMA               //Moving average type
#define MA_SHIFT   0                      //Moving average shift
#define ATR_PERIOD 14                     //Period for our ATR
#define TF_1       PERIOD_D1              //Hihger order timeframe
#define TRADING_MA_PERIOD  100            //Moving average period
#define SL_WIDTH 1.5                      //How wide should the stop loss be?
#define CURRENT_VOL 0.1                   //Trading volume

//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
int    ma_handler,trading_ma_handler,atr_handler;
double ma[],atr[],trading_ma[];
double ask,bid;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   setup();
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
IndicatorRelease(atr_handler);
IndicatorRelease(ma_handler);
IndicatorRelease(trading_ma_handler);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   update();
   if(PositionsTotal() == 0)
      find_setup();
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Find setup                                                       |
//+------------------------------------------------------------------+
void find_setup(void)
  {
//Buy on rallies
   if(iClose(Symbol(),PERIOD_CURRENT,0) > trading_ma[0])
      Trade.Buy(CURRENT_VOL,Symbol(),ask,(bid - (atr[0] * SL_WIDTH)),(bid + (atr[0] * SL_WIDTH)),"");

   if(iClose(Symbol(),PERIOD_CURRENT,0) < trading_ma[0])
      Trade.Sell(CURRENT_VOL,Symbol(),bid,(ask + (atr[0] * SL_WIDTH)),(ask - (atr[0] * SL_WIDTH)));
  }

//+------------------------------------------------------------------+
//| Setup our global variables                                       |
//+------------------------------------------------------------------+
void setup(void)
  {
   atr_handler =  iATR(Symbol(),TF_1,ATR_PERIOD);
   trading_ma_handler  =  iMA(Symbol(),PERIOD_CURRENT,TRADING_MA_PERIOD,MA_SHIFT,MA_MODE,MA_PRICE);
  }

//+------------------------------------------------------------------+
//| Update                                                           |
//+------------------------------------------------------------------+
void update(void)
  {
   static datetime time_stamp;
   datetime current_time = iTime(Symbol(),PERIOD_CURRENT,0);
   ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);

   if(time_stamp != current_time)
     {
      time_stamp = current_time;
      CopyBuffer(atr_handler,0,0,1,atr);
      CopyBuffer(trading_ma_handler,0,0,1,trading_ma);
     }
  }
//+------------------------------------------------------------------+
```

### Establishing A Benchmark

Let's observe how well our trend following application performs from historical data M1 starting on Wednesday 1 January 2020 until Monday 6 January 2025. We will use the EURUSD pair for our back test. Note that we are not tuning any parameters of the strategy, therefore the "Forward" setting of the back test is set to "No".

![](https://c.mql5.com/2/109/5169218364965__1.png)

Fig 3: The dates of our back test

We will additionally set our modelling to be based on real ticks to best emulate actual trading conditions in our tester, additionally, using Random delay gives us an idea of how our system will perform stressed. Recall that the latency experienced when trading in real time may vary, so naturally we want to stay close to the conditions we expect our system to perform under in production.

![](https://c.mql5.com/2/109/2260516996813__1.png)

Fig 4: The conditions of our back test

This is the resulting equity curve obtained by our trading strategy over our 5-year back test. Pay attention to the difference between the maximum balance of the account and the final balance of the account. The results clearly show that our strategy is not stable, and it has a tendency of loosing money almost as easily as it profits. Our winning streaks following this strategy were short-lived and followed by drawdown periods that were sustained for just as long as our winning periods. Those long sustained drawdown periods are likely associated with the periods in which the market was not following a long-term trend, and our naive trading strategy cannot handle such market conditions appropriately.

![](https://c.mql5.com/2/109/5571027189738__1.png)

Fig 5: Our equity curve obtained from the initial version of our strategy

When we examine the detailed results of our back test, we can see that our strategy was profitable over the 5-year period, which encourages us to try and improve the strategy, however the proportion of winning and losing trades is almost 50/50. This is undesirable. We want to filter out the losing trades, so we can have a higher ratio of winning trades. Our average consecutive wins are 2 and our average consecutive losses are also 2, this supports our remark that our system appears just as likely to make profits for us as it is to lose our capital. Such a system cannot be trusted to trade without supervision.

![](https://c.mql5.com/2/109/2785373122757__1.png)

Fig 6: A detailed summary of our trading back test

### Overview of The Proposed Solution

This initial back test now brings us to discuss the proposed system in detail. We know that our system will profit when markets are following a trend, but it will lose money when markets have no trend. So how can we design our system to independently decide if a market is likely to be following a trend or not?

Knowing how markets switch between these 2 modes will help our system make better use of the moving average indicator. On days when our system believes the market is likely to be range bound, it will place its trades against the trend implied by the moving average. Otherwise, it will place its trades in line with the implied trend.

In other words, if our algorithm detects that the Symbol we are trading is likely in a range bound mode, then whenever price levels close above the moving average, we will sell, instead of buying. However, if our algorithm detects that the Symbol is in a trend mode, then when the price levels close above the moving average, we will buy. Essentially, the same market event will be interpreted and handled differently depending on what mode we detect the market is in.

So now the only question left is "How can we identify the mode the market is in?". Our proposed strategy is to divide price levels into 4 discrete zones. The foundational idea behind our strategy is intuitive:

1. Trend Mode: The market can only truly trend in either Zone 1 or Zone 4.
2. Range Mode: The market can only truly be range bound in Zone 2 or Zone 3.

A visualization of the 4 zones is provided below in Fig 7.

![](https://c.mql5.com/2/109/3899153129877__1.png)

Fig 7: A simple illustrated example of our Zone-based trading system

Let us take a look at each zone in turn, starting with Zone 1. When price levels are in Zone 1, we perceive this as bullish market sentiment, and we will look for opportunities to buy whenever price levels close above the 100 period moving average. The width of our take profit and stop loss will be preserved from our initial back test. Note, we will only look for long opportunities to follow the bullish trends so long as we are in Zone 1. We will not take short positions whilst price levels are in Zone 1!

![](https://c.mql5.com/2/109/5129985702132__1.png)

Fig 8: Explaining the significance of Zone 1

If price levels fall from Zone 1 into Zone 2, then our market sentiment changes. We no longer believe that any true trends will be observed for as long as price levels remain in Zone 2. Rather, we believe that in Zone 2, price levels tend to oscillate around the mid-band separating Zone 2 and Zone 3. We will exclusively be looking for opportunities to sell when we are in Zone 2 because we believe that price levels will demonstrate a tendency to fall back to the middle band.

If price levels rise above the 100 period moving average whilst we are in Zone 2, we will sell, the placement of our stop loss will be preserved from our initial trading strategy. However, the positioning of our take profit must be modified. We will place our take profit on the middle band separating Zone 2 and Zone 3 because we suspect this is where price levels will tend to rest so long as we are within Zone 2. Note, we will not take any long positions whilst we are in Zone 2. Each zone only allows us 1 position type.

![](https://c.mql5.com/2/109/2200135730290__1.png)

Fig 9: Understating how our trading strategy evolves as we pass through the 4 market zones we have defined

I hope that at this point, a pattern is forming in the mind of the reader, and the remaining set of rules should be intuitive. Let's play a fun quiz to make sure we are both on the same page. If price levels fall to Zone 3, then what types of positions do you think we are looking to take? I hope you mentally said long positions.

And where will we place our take profit? I'd like to believe the reader now intuitively understands that when we are in Zone 2 or Zone 3, our take profit will be placed on the middle band that separates Zone 2 and Zone 3.

So, in other words, when we are in Zone 3, we will only take long positions when price levels close beneath the 100 period moving average. Our take profit will be placed on the middle band, and our stop loss will be the same size as it was in the original strategy. We will not occupy any short positions whilst we are in Zone 3.

![](https://c.mql5.com/2/109/2559232802332__1.png)

Fig 10: Zone 2 and Zone 3 are believed to be mean reverting zones

Finally, when price levels are in Zone 4, we will only look for opportunities to occupy short positions. We will take our short positions whenever price levels are beneath our 100 period moving average. The width of our take profit and stop loss will be identical to the width used in our benchmark strategy. We will not occupy any long positions so long as price levels remain in Zone 4.

![](https://c.mql5.com/2/109/1661141395811__1.png)

Fig 11: Zone 4 is where we believe bearish trends will be formed. Therefore, we will not take any long positions in Zone 4

To implement the 4 Zone strategy, we will have to make alterations to the original trend following strategy:

| Proposed Change | Intended Purpose |
| --- | --- |
| New system variables | We will need new system variables to give us control over the channel we are going to create. |
| Creation of user inputs | Although the user will not be able to control all aspects of the channel, some aspects such as the width of the channel, and the number of bars used to calculate the channel, will be controlled by the end user. |
| New global variables | New global variables related to the channel will be created to help us improve the trading logic of our application. |
| Alterations to our custom functions | We will make new functions in our trading application and modify some of the existing functions to implement the trading logic we have outlined. |

The details regarding the calculation of the channel will be discussed as we progress.

### Implementing Our Solution In MQL5

We will start by defining system constants that will help us calculate the channel. Calculating the channel requires us to first apply a moving average indicator on a time frame higher than the intended time frame. So in our example, we want to trade the M1. We will apply a 20 period moving average on the Daily Time Frame, and use it to calculate our channel. If we calculated the channel on the same time frame we intend to trade, then we may observe that our channel moves around too much for us to build a stable trading strategy.

```
#define MA_PERIOD  20                     //Moving Average period for calculating the mid point of our range zone
```

Next, we need to define global variables that keep track of the channel for us. In particular, we aim to know where the edge of Zone 2 (upper boundary) and Zone 3 (lower boundary) lie. Additionally, we desire to pinpoint where the boundary between Zone 2 and Zone 3 lies (middle boundary). Zone 1 is defined from the upper boundary and has no upper bound limit. Likewise, Zone 4 starts where Zone 3 ends, and has no lower bound limit.

```
double range_zone_mid,range_zone_ub,range_zone_lb;
int    active_zone;
```

Now we shall define the parameters of the channel that we will allow the end user to change. Such as the period of calculation from historical data, and the width of the channel the user would like. It is important to allow the user to control the width of the channel because the channel width is associated with the amount of risk in the account. For our demonstration, we will simply make the channel width equivalent to 2 ATR reading. Most markets tend to move 1 ATR in a day, and recall that we do not want our channel moving around too much.

```
//+------------------------------------------------------------------+
//| User inputs                                                      |
//+------------------------------------------------------------------+
input group "Technical Analysis"
input double atr_multiple =
2 ;            //ATR Multiple
input int    bars_used = 30;              //How Many Bars should we use to calculate the channel?
```

We need a function that will determine which zone we are currently in. The logic for determining zones has already been explained extensively.

```
//+------------------------------------------------------------------+
//| Get our current active zone                                      |
//+------------------------------------------------------------------+
void get_active_zone(void)
  {
   if(iClose(Symbol(),PERIOD_CURRENT,0) > range_zone_ub)
     {
      active_zone = 1;
      return;
     }

   if((iClose(Symbol(),PERIOD_CURRENT,0) < range_zone_ub) && (iClose(Symbol(),PERIOD_CURRENT,0) > range_zone_mid))
     {
      active_zone = 2;
      return;
     }

   if((iClose(Symbol(),PERIOD_CURRENT,0) < range_zone_mid) && (iClose(Symbol(),PERIOD_CURRENT,0) > range_zone_lb))
     {
      active_zone = 3;
      return;
     }

   if(iClose(Symbol(),PERIOD_CURRENT,0) < range_zone_lb)
     {
      active_zone = 4;
      return;
     }
  }
//+------------------------------------------------------------------+
```

We also need to update our setup function. Let us ignore the parts of the function that have not changed. We are only introducing 1 new technical indicator. The first technical indicator we applied was a 100 period moving average on the M1 time frame. Our new indicator is a 20 period moving average applied to the Daily Time Frame.

```
//+------------------------------------------------------------------+
//| Setup our global variables                                       |
//+------------------------------------------------------------------+
void setup(void)
  {
   //We have omitted parts of the code that have not changed
   ma_handler  =  iMA(Symbol(),TF_1,MA_PERIOD,MA_SHIFT,MA_MODE,MA_PRICE);
  }
```

Next, we need to make adjustments to our update function. We have, intentionally, omitted parts of the function that remained unchanged so we can focus on the new parts of the function. We will begin by initializing a vector with zeros. Then, using our new vector, we copy the number of bars our user has instructed us to use for our calculations from the new moving average we have applied to the daily time frame in our previous step above.

We then take the average value of the 20 period daily moving average and that will be the mid-band separating Zone 2 and Zone 3. The limits of Zone 2 and Zone 3 will be calculated by adding (Zone 2) and subtracting (Zone 3) a multiple of the ATR reading from the mid-point, we calculated using the 20 period moving average.

```
//+------------------------------------------------------------------+
//| Update                                                           |
//+------------------------------------------------------------------+
void update(void)
  {
   static datetime time_stamp;
   datetime current_time = iTime(Symbol(),PERIOD_CURRENT,0);
   ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);

   if(time_stamp != current_time)
     {
      //Omitted parts of the function that remained unchanged
      vector ma_average = vector::Zeros(1);
      ma_average.CopyIndicatorBuffer(ma_handler,0,1,bars_used);
      range_zone_mid = ma_average.Mean();
      range_zone_ub = (range_zone_mid + (atr[0] * atr_multiple));
      range_zone_lb = (range_zone_mid - (atr[0] * atr_multiple));
      get_active_zone();
      Comment("Zone: ",active_zone);
      ObjectDelete(0,"RANGE HIGH");
      ObjectDelete(0,"RANGE LOW");
      ObjectDelete(0,"RANGE MID");
      ObjectCreate(0,"RANGE MID",OBJ_HLINE,0,0,range_zone_mid);
      ObjectCreate(0,"RANGE LOW",OBJ_HLINE,0,0,range_zone_lb);
      ObjectCreate(0,"RANGE HIGH",OBJ_HLINE,0,0,range_zone_ub);
     }
  }
```

The last modification we need to make will apply to how we will find our setups. The trading logic behind our position placement has been discussed extensively, so this segment of code should be straightforward for the reader. To summarize the main idea, we will only follow the trend in Zones 1 and 4. Meaning, if the price closes above the 100 period moving average in Zone 1 we will buy. Otherwise, if we are in Zones 2 or 3, we will go against the trend, meaning if price levels close above the 100 period moving average in Zone 2, we will sell instead of buying like we did in Zone 1.

```
//+------------------------------------------------------------------+
//| Find setup                                                       |
//+------------------------------------------------------------------+
void find_setup(void)
  {

// Follow the trend
   if(active_zone == 1)
     {
      //Buy on rallies
      if(iClose(Symbol(),PERIOD_CURRENT,0) > trading_ma[0])
         Trade.Buy(CURRENT_VOL,Symbol(),ask,(bid - (atr[0] * 1.5)),(bid + (atr[0] * SL_WIDTH)),"");
     }

// Go against the trend
   if(active_zone == 2)
     {
      //Sell on rallies
      if(iClose(Symbol(),PERIOD_CURRENT,0) > trading_ma[0])
         Trade.Sell(CURRENT_VOL,Symbol(),bid,(ask + (atr[0] * 1.5)),range_zone_mid);
     }

// Go against the trend
   if(active_zone == 3)
     {
      //Buy the dip
      if(iClose(Symbol(),PERIOD_CURRENT,0) < trading_ma[0])
         Trade.Buy(CURRENT_VOL,Symbol(),ask,(bid - (atr[0] * 1.5)),range_zone_mid,"");
     }

// Follow the trend
   if(active_zone == 4)
     {
      //Sell the dip
      if(iClose(Symbol(),PERIOD_CURRENT,0) < trading_ma[0])
         Trade.Sell(CURRENT_VOL,Symbol(),bid,(ask + (atr[0] * atr_multiple)),(ask - (atr[0] * SL_WIDTH)));
     }
  }
```

Putting it all together, this is what the revised version of our trading strategy looks like.

```
//+------------------------------------------------------------------+
//|                              Dynamic Moving Average Strategy.mq5 |
//|                                               Gamuchirai Ndawana |
//|                    https://www.mql5.com/en/users/gamuchiraindawa |
//+------------------------------------------------------------------+
#property copyright "Gamuchirai Ndawana"
#property link      "https://www.mql5.com/en/users/gamuchiraindawa"
#property version   "1.00"

//+------------------------------------------------------------------+
//| System constants                                                 |
//+------------------------------------------------------------------+
#define MA_PRICE    PRICE_CLOSE           //Moving average shift
#define MA_MODE    MODE_EMA               //Moving average shift
#define MA_SHIFT   0                      //Moving average shift
#define ATR_PERIOD 14                     //Period for our ATR
#define TF_1       PERIOD_D1              //Hihger order timeframe
#define MA_PERIOD  20                     //Moving Average period for calculating the mid point of our range zone
#define TRADING_MA_PERIOD  100            //Moving average period
#define SL_WIDTH   1.5                    //How wide should the stop loss be?
#define CURRENT_VOL 0.1                   //Trading volume

//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
int    ma_handler,trading_ma_handler,atr_handler;
double ma[],atr[],trading_ma[];
double range_zone_mid,range_zone_ub,range_zone_lb;
double ask,bid;
int    active_zone;

//+------------------------------------------------------------------+
//| User inputs                                                      |
//+------------------------------------------------------------------+
input group "Technical Analysis"
input double atr_multiple = 1;            //ATR Multiple
input int    bars_used = 30;              //How Many Bars should we use to calculate the channel?

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   setup();
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   IndicatorRelease(ma_handler);
   IndicatorRelease(atr_handler);
   IndicatorRelease(trading_ma_handler);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   update();
   if(PositionsTotal() == 0)
      find_setup();
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Find setup                                                       |
//+------------------------------------------------------------------+
void find_setup(void)
  {

// Follow the trend
   if(active_zone == 1)
     {
      //Buy on rallies
      if(iClose(Symbol(),PERIOD_CURRENT,0) > trading_ma[0])
         Trade.Buy(CURRENT_VOL,Symbol(),ask,(bid - (atr[0] * 1.5)),(bid + (atr[0] * SL_WIDTH)),"");
     }

// Go against the trend
   if(active_zone == 2)
     {
      //Sell on rallies
      if(iClose(Symbol(),PERIOD_CURRENT,0) > trading_ma[0])
         Trade.Sell(CURRENT_VOL,Symbol(),bid,(ask + (atr[0] * 1.5)),range_zone_mid);
     }

// Go against the trend
   if(active_zone == 3)
     {
      //Buy the dip
      if(iClose(Symbol(),PERIOD_CURRENT,0) < trading_ma[0])
         Trade.Buy(CURRENT_VOL,Symbol(),ask,(bid - (atr[0] * 1.5)),range_zone_mid,"");
     }

// Follow the trend
   if(active_zone == 4)
     {
      //Sell the dip
      if(iClose(Symbol(),PERIOD_CURRENT,0) < trading_ma[0])
         Trade.Sell(CURRENT_VOL,Symbol(),bid,(ask + (atr[0] * atr_multiple)),(ask - (atr[0] * SL_WIDTH)));
     }
  }

//+------------------------------------------------------------------+
//| Setup our global variables                                       |
//+------------------------------------------------------------------+
void setup(void)
  {
   atr_handler =  iATR(Symbol(),TF_1,ATR_PERIOD);
   trading_ma_handler  =  iMA(Symbol(),PERIOD_CURRENT,TRADING_MA_PERIOD,MA_SHIFT,MA_MODE,MA_PRICE);
   ma_handler  =  iMA(Symbol(),TF_1,MA_PERIOD,MA_SHIFT,MA_MODE,MA_PRICE);
  }

//+------------------------------------------------------------------+
//| Update                                                           |
//+------------------------------------------------------------------+
void update(void)
  {
   static datetime time_stamp;
   datetime current_time = iTime(Symbol(),PERIOD_CURRENT,0);
   ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);

   if(time_stamp != current_time)
     {
      time_stamp = current_time;
      CopyBuffer(ma_handler,0,0,1,ma);
      CopyBuffer(atr_handler,0,0,1,atr);
      CopyBuffer(trading_ma_handler,0,0,1,trading_ma);
      vector ma_average = vector::Zeros(1);
      ma_average.CopyIndicatorBuffer(ma_handler,0,1,bars_used);
      range_zone_mid = ma_average.Mean();
      range_zone_ub = (range_zone_mid + (atr[0] * atr_multiple));
      range_zone_lb = (range_zone_mid - (atr[0] * atr_multiple));
      get_active_zone();
      Comment("Zone: ",active_zone);
      ObjectDelete(0,"RANGE HIGH");
      ObjectDelete(0,"RANGE LOW");
      ObjectDelete(0,"RANGE MID");
      ObjectCreate(0,"RANGE MID",OBJ_HLINE,0,0,range_zone_mid);
      ObjectCreate(0,"RANGE LOW",OBJ_HLINE,0,0,range_zone_lb);
      ObjectCreate(0,"RANGE HIGH",OBJ_HLINE,0,0,range_zone_ub);
     }
  }

//+------------------------------------------------------------------+
//| Get our current active zone                                      |
//+------------------------------------------------------------------+
void get_active_zone(void)
  {
   if(iClose(Symbol(),PERIOD_CURRENT,0) > range_zone_ub)
     {
      active_zone = 1;
      return;
     }

   if((iClose(Symbol(),PERIOD_CURRENT,0) < range_zone_ub) && (iClose(Symbol(),PERIOD_CURRENT,0) > range_zone_mid))
     {
      active_zone = 2;
      return;
     }

   if((iClose(Symbol(),PERIOD_CURRENT,0) < range_zone_mid) && (iClose(Symbol(),PERIOD_CURRENT,0) > range_zone_lb))
     {
      active_zone = 3;
      return;
     }

   if(iClose(Symbol(),PERIOD_CURRENT,0) < range_zone_lb)
     {
      active_zone = 4;
      return;
     }
  }
//+------------------------------------------------------------------+
```

Let's see if these changes will give us the control we desire. We will fix the period over which the back test is performed.

![](https://c.mql5.com/2/109/316254581443__1.png)

Fig 12: The period of our back test matches the initial period we used

Likewise, we will fix the back test conditions so they are consistent with our previous test.

![](https://c.mql5.com/2/109/4527359470981__1.png)

Fig 13: Ensure that both strategies are being tested under the same conditions

Our new strategy has 2 inputs. The first setting controls the width of the channel, and the second controls how many bars are used for calculating the channel.

![](https://c.mql5.com/2/109/1511446201545__1.png)

Fig 14: The controls our end user can adjust

The new equity curve created by our revised strategy has losing periods like any strategy, but notice how quickly it recovers from a loss. It doesn't remain stuck like our old trading strategy did. It loses a trade but quickly realigns itself with the market. This is evident by the fact that our losing periods are followed by a stretch of profitable trading.

![](https://c.mql5.com/2/109/1049329128633__1.png)

Fig 15: The equity curve produced by our new trading strategy

When we analyze the detailed results of our new trading strategy, we can see that the proportion of our winning trades jumped from 52% to 86%, and our losing trades fell from 47% to 13%. Our average profit is smaller than our average loss, but bear in mind our stop loss and take profit are fixed, this problem could be solved by keeping the stop loss fixed if we are losing, and allowing it to trail if we are profiting. Additionally, our average consecutive wins jumped from 2 to 9 while on the other hand or average consecutive losses fell from 2 to 1. Our original strategy placed 300 trades. While our new strategy placed a total of 1301 trades. So our new strategy is placing more trades and winning more often.

![](https://c.mql5.com/2/109/1801516000705__1.png)

Fig 16: A detailed summary of the performance of our new strategy

### Conclusion

In this discussion today, we started with a naive trend following strategy and guided it to make more informed decisions using historical data that was available in our MetaTrader 5 terminal. In theory, we can implement this strategy with pen and paper, but fortunately for us, the MQL5 API streamlines this entire process for us. From quickly calculating statistical measures using vector functions to optimal trade execution, a lot of the heavy lifting is being done for us in the background by our MetaTrader 5 application. In future articles, we will consider taking our proportion of profitable trades even higher and exercising additional control over the size of our losing trades.

| Attached File | Description |
| --- | --- |
| Benchmark Moving Average Strategy | The initial implementation of our trend following strategy |
| Dynamic Moving Average Strategy | Our new proposed dynamic strategy. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16856.zip "Download all attachments in the single ZIP archive")

[Benchmark\_Moving\_Average\_Strategy.mq5](https://www.mql5.com/en/articles/download/16856/benchmark_moving_average_strategy.mq5 "Download Benchmark_Moving_Average_Strategy.mq5")(4.47 KB)

[Dynamic\_Moving\_Average\_Strategy.mq5](https://www.mql5.com/en/articles/download/16856/dynamic_moving_average_strategy.mq5 "Download Dynamic_Moving_Average_Strategy.mq5")(7.24 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Reimagining Classic Strategies (Part 21): Bollinger Bands And RSI Ensemble Strategy Discovery](https://www.mql5.com/en/articles/20933)
- [Reimagining Classic Strategies (Part 20): Modern Stochastic Oscillators](https://www.mql5.com/en/articles/20530)
- [Overcoming The Limitation of Machine Learning (Part 9): Correlation-Based Feature Learning in Self-Supervised Finance](https://www.mql5.com/en/articles/20514)
- [Reimagining Classic Strategies (Part 19): Deep Dive Into Moving Average Crossovers](https://www.mql5.com/en/articles/20488)
- [Overcoming The Limitation of Machine Learning (Part 8): Nonparametric Strategy Selection](https://www.mql5.com/en/articles/20317)
- [Overcoming The Limitation of Machine Learning (Part 7): Automatic Strategy Selection](https://www.mql5.com/en/articles/20256)
- [Self Optimizing Expert Advisors in MQL5 (Part 17): Ensemble Intelligence](https://www.mql5.com/en/articles/20238)

**[Go to discussion](https://www.mql5.com/en/forum/479579)**

![Developing A Swing Entries Monitoring (EA)](https://c.mql5.com/2/109/Developing_A_Swing_Entries_Monitoring___LOGO.png)[Developing A Swing Entries Monitoring (EA)](https://www.mql5.com/en/articles/16563)

As the year approaches its end, long-term traders often reflect on market history to analyze its behavior and trends, aiming to project potential future movements. In this article, we will explore the development of a long-term entry monitoring Expert Advisor (EA) using MQL5. The objective is to address the challenge of missed long-term trading opportunities caused by manual trading and the absence of automated monitoring systems. We'll use one of the most prominently traded pairs as an example to strategize and develop our solution effectively.

![Neural Networks Made Easy (Part 97): Training Models With MSFformer](https://c.mql5.com/2/82/Neural_networks_are_easy_Part_96__LOGO__1.png)[Neural Networks Made Easy (Part 97): Training Models With MSFformer](https://www.mql5.com/en/articles/15171)

When exploring various model architecture designs, we often devote insufficient attention to the process of model training. In this article, I aim to address this gap.

![Developing a Replay System (Part 55): Control Module](https://c.mql5.com/2/83/Desenvolvendo_um_sistema_de_Replay_Parte_55__LOGO.png)[Developing a Replay System (Part 55): Control Module](https://www.mql5.com/en/articles/11988)

In this article, we will implement a control indicator so that it can be integrated into the message system we are developing. Although it is not very difficult, there are some details that need to be understood about the initialization of this module. The material presented here is for educational purposes only. In no way should it be considered as an application for any purpose other than learning and mastering the concepts shown.

![MQL5 Wizard Techniques you should know (Part 52): Accelerator Oscillator](https://c.mql5.com/2/108/MQL5_Wizard_Techniques_you_should_know_Part_52_Accelerator_Oscillator____LOGO.png)[MQL5 Wizard Techniques you should know (Part 52): Accelerator Oscillator](https://www.mql5.com/en/articles/16781)

The Accelerator Oscillator is another Bill Williams Indicator that tracks price momentum's acceleration and not just its pace. Although much like the Awesome oscillator we reviewed in a recent article, it seeks to avoid the lagging effects by focusing more on acceleration as opposed to just speed. We examine as always what patterns we can get from this and also what significance each could have in trading via a wizard assembled Expert Advisor.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/16856&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6369019141164999827)

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