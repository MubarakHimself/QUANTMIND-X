---
title: Reimagining Classic Strategies (Part 13): Minimizing The Lag in Moving Average Cross-Overs
url: https://www.mql5.com/en/articles/16758
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:43:48.909279
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/16758&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069693411302246707)

MetaTrader 5 / Examples


In our previous discussions, we have taken different perspectives on how we can maximize our efficiency when using moving average crossover strategies. To briefly summarize, we have previously conducted an exercise over more than 200 different symbols and observed that our computer appears to learn to predict future moving average values better than its tendency to predict future price levels correctly, I have provided a quick link to the article, [here](https://www.mql5.com/en/articles/16230). We extended this idea and modeled 2 moving averages, with the single goal of predicting cross-overs earlier than casual market participants and adjusting our trades accordingly. I have included a link to the second article as well, readily available at your fingertips, [here](https://www.mql5.com/en/articles/16280).

Our discussion today, we will extend the original strategy yet again, to minimize the lag inherent in the trading strategy. Traditional cross over strategies require a temporal gap to exist between the 2 moving average periods. However, we will break away from this conventional school of thought, and use the same period on both moving averages.

At this point, some readers may be questioning if this can still qualify as a moving average cross over strategy because how will your moving averages cross over, if they both share the same period? The answer is surprisingly simple, we apply one moving average on the open and close price, respectively.

When the opening moving average is above the close moving average, price levels are finishing lower than they started. This is identical to knowing that price levels are falling. And the converse is true when the close moving average is above the open.

A common period on both moving averages is unconventional, however I have selected it for our exercise today to demonstrate to the reader one possible way we can lay to rest any criticisms about the lag associated with moving average cross-overs.

Before we dive into the technical segment of our discussion, this article may be read as an example of a generalized class of trading strategies that can easily be extended to many other technical indicators. For example, the Relative Strength Indicator can also be applied to the open, close, high or low independently, and we will observe the RSI tracking the open price will tend to rise above the RSI tracking the close price when markets are in downtrends.

### Overview of Our Backtest

For us to appreciate the significance of our discussion today, we will first establish a benchmark created by the traditional cross-over strategy. Then we will compare the legacy performance against what we stand to achieve using our reimagined version of the strategy.

I have selected the EURUSD pair for our discussion today. The EURUSD is the most actively traded currency pair in the world. It is significantly more volatile than most currency pairs and is generally not a good choice for simple cross-over-based strategies. As we have already discussed earlier, we will focus on the daily time frame. We will back test our strategy over approximately 4 years of historical data, from the 1st of January 2020 until the 24th of December 2024, the back test period is highlighted below in Fig 1.

![](https://c.mql5.com/2/109/1419632070512.png)

Fig 1: Viewing our 4 year EURUSD back test period on our MetaTrader 5 terminal using the Monthly time frame

Although traditional cross-over strategies are intuitive to grasp and backed by fundamental principles that are reasonably sound, these strategies often require endless optimization to guarantee effective use. Additionally, the “right” periods to use for the slow and fast-moving indicator are not immediately obvious, and can change dramatically.

To recap, the original strategy is based on the intersections created by 2 moving averages both tracking the close price of the same security, but with different periods. When the moving average with the shorter period is on top, we interpret this as a signal that price levels have been in an uptrend and are likely to continue rising. The opposite applies when the moving average with the longer period is on top, we interpret this as a bearish signal, an illustrated example is provided for you in Fig 2 that follows.

![](https://c.mql5.com/2/109/2164418513259.png)

Fig 2: An example of the traditional moving average cross over strategy in action, the yellow line is the fast-moving average and the white is the slow

In Fig 2 above, we have randomly selected a time period that shows the limitations of the classical strategy. To the left of the vertical line on the chart, you will observe that price action was stuck in a range for roughly 2 months. This lethargic price action generates trading signals that are quickly reversed and most likely unprofitable. However, after this period of dismal performance, we finally saw price levels break away in a true trend on the right of the vertical line. Traditional cross-over strategies work best in trending market conditions. However, the strategy proposed in this article handles these problems elegantly.

### Establishing A Benchmark

Our trading application can be conceptualized as 4 main components that will work together, to help us trade:

| Feature | Description |
| --- | --- |
| System Constants | Help us isolate the improvements brought about by the changes we are making to the trading logic of our application. |
| Global Variables | Responsible for keeping track of indicator values, current market prices and potentially more information we need. |
| Event Handlers | Perform various tasks at the appropriate time to meet our goal of trading moving average cross-overs effectively. |
| Customized Functions | Each customized function in our system has a specific task delegated to it, and all together the functions collectively help us achieve our goal. |

The benchmark version of our application will be minimalistic in its implementation. Our first order of business is to set up system constants that will remain fixed across both tests we will perform. Our system constants are important for making fair comparisons between different trading strategies and prevent us from unintentionally changing settings that shouldn't be changed across tests, like the size of our stop loss should be constant in both tests to ensure a fair comparison.

```
//+------------------------------------------------------------------+
//|                                               Channel And MA.mq5 |
//|                                        Gamuchirai Zororo Ndawana |
//|                          https://www.mql5.com/en/gamuchiraindawa |
//+------------------------------------------------------------------+
#property copyright "Gamuchirai Zororo Ndawana"
#property link      "https://www.mql5.com/en/gamuchiraindawa"
#property version   "1.00"

//+------------------------------------------------------------------+
//| System constants                                                 |
//+------------------------------------------------------------------+
#define TF_1           PERIOD_D1 //--- Our main time frame
#define TF_2           PERIOD_H1 //--- Our lower time frame
#define ATR_PERIOD     14        //--- The period for our ATR
#define ATR_MULTIPLE   3         //--- How wide should our stops be?
#define VOL            0.01      //--- Trading volume
```

We will also define a few important global variables that we will use to fetch indicator values and get current market prices.

```
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
int    trade;
int    ma_f_handler,ma_s_handler,atr_handler;
double ma_f[],ma_s[],atr[];
double bid,ask;
double o,h,l,c;
double original_sl;
```

We will use the trade library to manage our positions.

```
//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;
```

In MQL5, expert advisors are built up from event handlers. There are many different types of events that happen in the MetaTrader 5 terminal. These events can be triggered by actions the user has taken, or if new prices are quoted. Each event is paired with an event handler that is called each time the event is triggered. Therefore, I have designed our application such that each event handler has its own designated function that it will call in turn to perform the tasks necessary for a moving average cross-over strategy.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Setup
   setup();
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Release our indicators
   release();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Update system variables
   update();
  }
//+------------------------------------------------------------------+
```

The setup function is called when our system is launched. The OnInit handler will be called when the trading application is first applied onto a chart, and it will pass on the chain of command to our customized setup function that will apply our technical indicators for us.

```
//+------------------------------------------------------------------+
//| Custom functions                                                 |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Setup our system                                                 |
//+------------------------------------------------------------------+
void setup(void)
  {
   atr_handler  = iATR(Symbol(),TF_1,ATR_PERIOD);

   ma_f_handler = iMA(Symbol(),TF_1,10,0,MODE_EMA,PRICE_CLOSE);
   ma_s_handler = iMA(Symbol(),TF_1,60,0,MODE_EMA,PRICE_CLOSE);
  }
```

If we are no longer using our trading application, the on Deinit event handler is called, and it will call our release function that will free up the system resources that were previously being consumed by our technical indicators.

```
//+------------------------------------------------------------------+
//| Release variables we do not need                                 |
//+------------------------------------------------------------------+
void release(void)
  {
   IndicatorRelease(atr_handler);
   IndicatorRelease(ma_f_handler);
   IndicatorRelease(ma_s_handler);
  }
```

Whenever new market prices are quoted, the OnTick handler will be called, and it will in turn call the update function to store the new market information that is available. Afterward, if we have no open positions, we will look for a trading setup. Otherwise, we will manage the positions we have open.

```
/+------------------------------------------------------------------+
//| Update system variables                                          |
//+------------------------------------------------------------------+
void update(void)
  {
//--- Update the system
   static datetime time_stamp;
   datetime current_time = iTime(Symbol(),TF_2,0);
   if(current_time != time_stamp)
     {
      time_stamp = current_time;

      CopyBuffer(atr_handler,0,0,1,atr);
      CopyBuffer(ma_s_handler,0,0,1,ma_s);
      CopyBuffer(ma_f_handler,0,0,1,ma_f);

      o  = iOpen(Symbol(),TF_1,0);
      h  = iHigh(Symbol(),TF_1,0);
      l  = iLow(Symbol(),TF_1,0);
      c  = iClose(Symbol(),TF_1,0);
      bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
      ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);

      if(PositionsTotal() == 0)
         find_position();
      if(PositionsTotal() > 0)
         manage_position();
     }
  }
```

Our rules for entering positions are straightforward and have already been explained in detail above. We enter long positions if our fast-moving average is above the slow-moving average, and if the opposite is true we will occupy short positions.

```
//+------------------------------------------------------------------+
//| Find a position                                                  |
//+------------------------------------------------------------------+
void find_position(void)
  {

   if((ma_s[0] > ma_f[0]))
     {
      Trade.Sell(VOL,Symbol(),bid,0,0,"");
      trade = -1;
     }

   if((ma_s[0] < ma_f[0]))
     {
      Trade.Buy(VOL,Symbol(),ask,0,0,"");
      trade = 1;
     }
  }
```

Finally, our stop loss will be adjusted dynamically using the Average True Range indicator. We will add a fixed multiple of the ATR reading above and beneath our entry price to mark our stop loss and take profit levels. Additionally, we will also add the average ATR reading over the past 90 days (1 business cycle), our intention for doing so is to account for recent volatility levels in the market. Lastly, we will use a ternary operator to adjust the take profit and stop loss levels. Our rule is that the stops should only be updated if the new position will be more profitable than the old position. Ternary operators allow us to express this logic in a compact fashion. Additionally, ternary operators also give us the flexibility to easily adjust the take profit and stop loss independently of each other.

```
//+------------------------------------------------------------------+
//| Manage our positions                                             |
//+------------------------------------------------------------------+
void manage_position(void)
  {
//--- Select the position
   if(PositionSelect(Symbol()))
     {
      //--- Get ready to update the SL/TP
      double initial_sl  = PositionGetDouble(POSITION_SL);
      double initial_tp  = PositionGetDouble(POSITION_TP);
      //--- Calculate the average ATR move
      vector atr_mean;
      atr_mean.CopyIndicatorBuffer(atr_handler,0,0,90);
      double buy_sl      = (ask - ((ATR_MULTIPLE * atr[0]) + atr_mean.Mean()));
      double sell_sl     = (bid + ((ATR_MULTIPLE * atr[0]) + atr_mean.Mean()));
      double buy_tp      = (ask + ((ATR_MULTIPLE * 0.5 * atr[0]) + atr_mean.Mean()));
      double sell_tp     = (bid - ((ATR_MULTIPLE * 0.5 * atr[0]) + atr_mean.Mean()));
      double new_sl      = ((trade == 1) && (initial_sl <  buy_sl)) ? (buy_sl) : ((trade == -1) && (initial_sl > sell_sl)) ? (sell_sl) : (initial_sl);
      double new_tp      = ((trade == 1) && (initial_tp <  buy_tp)) ? (buy_tp) : ((trade == -1) && (initial_tp > sell_tp)) ? (sell_tp) : (initial_tp);

      if(initial_sl == 0 && initial_tp == 0)
        {
         if(trade == 1)
           {
            original_sl = buy_sl;
            Trade.PositionModify(Symbol(),buy_sl,buy_tp);
           }

         if(trade == -1)
           {
            original_sl = sell_sl;
            Trade.PositionModify(Symbol(),sell_sl,sell_tp);
           }

        }
      //--- Update the position
      else
         if((initial_sl * initial_tp) != 0)
           {
            Trade.PositionModify(Symbol(),new_sl,new_tp);
           }
     }
  }
//+------------------------------------------------------------------+
```

When put all together, our current code looks like this so far.

```
//+------------------------------------------------------------------+
//|                                               Channel And MA.mq5 |
//|                                        Gamuchirai Zororo Ndawana |
//|                          https://www.mql5.com/en/gamuchiraindawa |
//+------------------------------------------------------------------+
#property copyright "Gamuchirai Zororo Ndawana"
#property link      "https://www.mql5.com/en/gamuchiraindawa"
#property version   "1.00"

//+------------------------------------------------------------------+
//| This version off the application is mean reverting               |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| System constants                                                 |
//+------------------------------------------------------------------+
#define TF_1           PERIOD_D1 //--- Our main time frame
#define TF_2           PERIOD_H1 //--- Our lower time frame
#define ATR_PERIOD     14        //--- The period for our ATR
#define ATR_MULTIPLE   3         //--- How wide should our stops be?
#define VOL            0.01         //--- Trading volume

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
int    trade;
int    ma_f_handler,ma_s_handler,atr_handler;
double ma_f[],ma_s[],atr[];
double bid,ask;
double o,h,l,c;
double original_sl;

//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Setup
   setup();
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Release our indicators
   release();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Update system variables
   update();
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Custom functions                                                 |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Setup our system                                                 |
//+------------------------------------------------------------------+
void setup(void)
  {
   atr_handler  = iATR(Symbol(),TF_1,ATR_PERIOD);

   ma_f_handler = iMA(Symbol(),TF_1,10,0,MODE_EMA,PRICE_CLOSE);
   ma_s_handler = iMA(Symbol(),TF_1,60,0,MODE_EMA,PRICE_CLOSE);
  }

//+------------------------------------------------------------------+
//| Release variables we do not need                                 |
//+------------------------------------------------------------------+
void release(void)
  {
   IndicatorRelease(atr_handler);
   IndicatorRelease(ma_f_handler);
   IndicatorRelease(ma_s_handler);
  }

//+------------------------------------------------------------------+
//| Update system variables                                          |
//+------------------------------------------------------------------+
void update(void)
  {
//--- Update the system
   static datetime time_stamp;
   datetime current_time = iTime(Symbol(),TF_2,0);
   if(current_time != time_stamp)
     {
      time_stamp = current_time;

      CopyBuffer(atr_handler,0,0,1,atr);
      CopyBuffer(ma_s_handler,0,0,1,ma_s);
      CopyBuffer(ma_f_handler,0,0,1,ma_f);

      o  = iOpen(Symbol(),TF_1,0);
      h  = iHigh(Symbol(),TF_1,0);
      l  = iLow(Symbol(),TF_1,0);
      c  = iClose(Symbol(),TF_1,0);
      bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
      ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);

      if(PositionsTotal() == 0)
         find_position();
      if(PositionsTotal() > 0)
         manage_position();
     }
  }

//+------------------------------------------------------------------+
//| Find a position                                                  |
//+------------------------------------------------------------------+
void find_position(void)
  {

   if((ma_s[0] > ma_f[0]))
     {
      Trade.Sell(VOL,Symbol(),bid,0,0,"");
      trade = -1;
     }

   if((ma_s[0] < ma_f[0]))
     {
      Trade.Buy(VOL,Symbol(),ask,0,0,"");
      trade = 1;
     }
  }

//+------------------------------------------------------------------+
//| Manage our positions                                             |
//+------------------------------------------------------------------+
void manage_position(void)
  {
//--- Select the position
   if(PositionSelect(Symbol()))
     {
      //--- Get ready to update the SL/TP
      double initial_sl  = PositionGetDouble(POSITION_SL);
      double initial_tp  = PositionGetDouble(POSITION_TP);
      //--- Calculate the average ATR move
      vector atr_mean;
      atr_mean.CopyIndicatorBuffer(atr_handler,0,0,90);
      double buy_sl      = (ask - ((ATR_MULTIPLE * atr[0]) + atr_mean.Mean()));
      double sell_sl     = (bid + ((ATR_MULTIPLE * atr[0]) + atr_mean.Mean()));
      double buy_tp      = (ask + ((ATR_MULTIPLE * 0.5 * atr[0]) + atr_mean.Mean()));
      double sell_tp     = (bid - ((ATR_MULTIPLE * 0.5 * atr[0]) + atr_mean.Mean()));
      double new_sl      = ((trade == 1) && (initial_sl <  buy_sl)) ? (buy_sl) : ((trade == -1) && (initial_sl > sell_sl)) ? (sell_sl) : (initial_sl);
      double new_tp      = ((trade == 1) && (initial_tp <  buy_tp)) ? (buy_tp) : ((trade == -1) && (initial_tp > sell_tp)) ? (sell_tp) : (initial_tp);

      if(initial_sl == 0 && initial_tp == 0)
        {
         if(trade == 1)
           {
            original_sl = buy_sl;
            Trade.PositionModify(Symbol(),buy_sl,buy_tp);
           }

         if(trade == -1)
           {
            original_sl = sell_sl;
            Trade.PositionModify(Symbol(),sell_sl,sell_tp);
           }

        }
      //--- Update the position
      else
         if((initial_sl * initial_tp) != 0)
           {
            Trade.PositionModify(Symbol(),new_sl,new_tp);
           }
     }
  }
//+------------------------------------------------------------------+
```

Since our current strategy does not employ AI or any curve fitting techniques, we can perform a simple back test without worrying about overfitting to the data we have available. Additionally, we have no input parameters that need to be adjusted. Therefore, we have set "Forward" to "No" because we do not need to employ the forward testing capabilities of the MetaTrader 5 terminal right now.

![](https://c.mql5.com/2/109/2550483380959.png)

Fig 3: Selecting the dates for our back test

Additionally, it is good practice to test your trading applications under the most stressful environment you can simulate. Therefore, we have selected the "Random delay" mode for our back test today, with our modelling set to use real ticks. When using real ticks, the time needed to download historical data may be longer than when using other modes such as "Open prices only". However, the results are likely to be closer to your true results on live ticks.

![Our second batch of settings](https://c.mql5.com/2/109/2.png)

Fig 4: Our second batch of settings for our EURUSD cross-over back test

When we analyze the results we have obtained using the simple cross-over strategy, we can quickly see potential problems we would run into, if we followed the strategy in its original form. Notice that from the beginning of the back test, until July 2022, our strategy was working hard just to break even. This is a drawdown period of almost half our back test, or 2 years. This is undesirable and not a characteristic of the type of strategy we can trust to trade our money unsupervised.

![](https://c.mql5.com/2/109/5348860639940.png)

Fig 5: Analyzing our profit and loss curve produced by following the MA cross over strategy

Our strategy in its original form is barely profitable and loses close to 61% of all the trades it places. This gives us negative expectations regarding the strategy, and our pessimism is further validated by the fact that our Sharpe ratio is very close to 0. But observe how drastically we can improve our strategy by making a few simple adjustments to the trading logic being employed.

![](https://c.mql5.com/2/109/3273528705086.png)

Fig 6: A detailed summary of our performance using the legacy MA cross-over strategy

### Improving The Original Strategy

In Fig 7 below, I have provided you a visual illustration of our new suggested cross-over strategy. The blue and green line are 5 period moving averages following the close (blue) and open (green) price. Notice that when the blue moving average is above the green moving average, price levels were rising. Additionally, pay attention to how responsive our cross-overs are to changes in price levels. Traditional cross-over strategies will take a variable amount of time to reflect any changes in the market trend. However, when we follow price levels using our new strategy, we can quickly observe changes in the trend, and even periods of consolidation, whereby or 2 moving averages keep crossing each other but make no true progress.

![](https://c.mql5.com/2/109/5833228641686.png)

Fig 7: Visualizing our new cross-over strategy on the EURUSD Daily time-frame

So far, our system has been able to trade profitably, but we can do better. The only change we are going to make to our system, is changing the conditions under which our positions are triggered:

| Change | Description |
| --- | --- |
| Trading<br>Rules | Our traditional rules buy when the fast-moving average is above the slow. Instead, we will now buy when our open moving average is above the close moving average. |

To realize our desired improvement, we have to make a few changes to the original form of our current trading strategy:

| Change | Description |
| --- | --- |
| Additional System Variables | We will need a new system variable responsible for fixing the period of the open and close moving average. |
| New Global Variables | New global variables will be created to keep track of the new information we are paying attention to. |
| Modify Custom Functions | Some of the customized functions we have built so far, need to be extended in light of the new system design we are following. |

For the most part, all other parts of our system will be preserved. We want to isolate the improvements brought about by changing our perspective on moving average cross-overs. Therefore, to realize our goal, we will start off by creating a new system constant to fix the period of our moving averages.

```
//--- Omitted code that has not changed
#define MA_PERIOD      2         //--- The period for our moving average following the close
```

Then we need to define new global variables for the new information we are keeping track of. We are now creating moving average handlers for each of the 4 prices quoted (Open, High, Low & Close).

```
//--- Omitted code that have not changed
int    ma_h_handler,ma_l_handler,ma_c_handler,ma_o_handler;
double ma_h[],ma_l[],ma_c[],ma_o[];
```

When our trading application is being launched, we will need to load a few additional indicators on top of the ones we are already familiar with.

```
//+------------------------------------------------------------------+
//| Setup our system                                                 |
//+------------------------------------------------------------------+
void setup(void)
  {
   atr_handler  = iATR(Symbol(),TF_1,ATR_PERIOD);
   ma_h_handler = iMA(Symbol(),TF_1,MA_PERIOD,0,MODE_EMA,PRICE_HIGH);
   ma_l_handler = iMA(Symbol(),TF_1,MA_PERIOD,0,MODE_EMA,PRICE_LOW);
   ma_c_handler = iMA(Symbol(),TF_1,MA_PERIOD,0,MODE_EMA,PRICE_CLOSE);
   ma_o_handler = iMA(Symbol(),TF_1,MA_PERIOD,0,MODE_EMA,PRICE_OPEN);
  }
```

The same applies for our customized function responsible for removing our expert advisor, it will be extended to accommodate the new indicators we have introduced into the system.

```
//+------------------------------------------------------------------+
//| Release variables we do not need                                 |
//+------------------------------------------------------------------+
void release(void)
  {
   IndicatorRelease(atr_handler);
   IndicatorRelease(ma_h_handler);
   IndicatorRelease(ma_l_handler);
   IndicatorRelease(ma_c_handler);
   IndicatorRelease(ma_o_handler);
  }
```

Lastly, the conditions which we will use to open our trades must be updated so that they are inline with our new trading logic.

```
//+------------------------------------------------------------------+
//| Find a position                                                  |
//+------------------------------------------------------------------+
void find_position(void)
  {

   if((ma_o[0] > ma_c[0]))
     {
      Trade.Sell(VOL,Symbol(),bid,0,0,"");
      trade = -1;
     }

   if((ma_o[0] < ma_c[0]))
     {
      Trade.Buy(VOL,Symbol(),ask,0,0,"");
      trade = 1;
     }
  }
```

Let us now see what difference this makes for our bottom line. We will first set up our new trading application to trade over the same time period we used in our first test.

![](https://c.mql5.com/2/109/6511855244545.png)

Fig 8: Setting up our new and improved trading algorithm to trade over the same time period we used in our first test

Additionally, we will want to perform both back-tests under identical conditions to ensure that our test in unbiased. Otherwise, if 1 strategy is given an unfair advantage, then the integrity of our tests so far would be challenged.

![](https://c.mql5.com/2/109/2514985897217.png)

Fig 9: We want to ensure that the test conditions are identical in both tests to make a fair comparison

We can already see a big improvement over the initial results we obtained. We spent the first 2 years of our initial back-test trying to break-even. However, with our new trading strategy, we broke through that limitation and were profitable across all 4 years.

![](https://c.mql5.com/2/109/6308962909506.png)

Fig 10: Our new trading strategy, broke through the unprofitable period of trading that would've been challenging to solve using traditional cross-over strategies

In our initial back-test, our system made 41 trades in total and in our latest back-test we are have made 42 trades in total. Therefore, our new system is taking on more risk than the old way of trading. Because if we allowed more time to pass, the gap between them may continue growing larger. However, although our new system appears to be placing more trades than our old system, our total profit of from our old system was $59.57, and now our total profit has more than doubled to $125.36. Recall that for now we have limited our trading system to place 1 trade at minimum lot. Additionally, in our first system our gross loss was $410.02 and with our new strategy our gross loss has fallen to $330.74.

When designing systems, we have to think in terms of trade-offs. Our new system is performing better for us. However, we should also take note that the size of our average profit has fallen from $29.35 to $22.81. This is because, sporadically, our new system will miss out on trades our old system profited from. However, this occasional regret may be warranted given the gains in performance we accrue.

Our Sharpe ratio has increased from 0.18 in our first test, to 0.5 in our current test. This is a good sign that indicates we are better utilizing our capital. Additionally, our proportion of losing trades has fallen from 60.98% in the first test, to a new low of 52.38%.

![](https://c.mql5.com/2/109/5318227929906.png)

Fig 11: A detailed review of the performance of our new trading strategy

### Conclusion

Most members of our community tend to be independent developers, working on their projects alone. For our fellow community members in those positions, I believe simple algorithms such as the one we have suggested for you here may be more practical solutions. It is easier to maintain, develop and extend. Managing a complex and large code base as a single developer is no easy task, even for experienced developers. And if you are new to our algorithmic trading community, then this strategy may be especially helpful for you. If you have enjoyed reading this article, then be sure to join us in our next discussion where we will try to outperform the best results we have produced today.

| File Name | Description |
| --- | --- |
| Open & Close MA Cross | This file contains our new reimagined version of the moving average cross-over strategy. |
| Traditional MA Cross | This file contains the classical implementation of moving average cross-overs. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16758.zip "Download all attachments in the single ZIP archive")

[Traditional\_MA\_Cross.mq5](https://www.mql5.com/en/articles/download/16758/traditional_ma_cross.mq5 "Download Traditional_MA_Cross.mq5")(6.97 KB)

[Open\_5\_Close\_MA\_Cross.mq5](https://www.mql5.com/en/articles/download/16758/open_5_close_ma_cross.mq5 "Download Open_5_Close_MA_Cross.mq5")(7.41 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/479523)**
(3)


![linfo2](https://c.mql5.com/avatar/2023/4/6438c14d-e2f0.png)

**[linfo2](https://www.mql5.com/en/users/neilhazelwood)**
\|
24 Jan 2025 at 19:16

Thanks for the code and Idea's , I like how you have structured your code . You have introduced me to the [Vector](https://www.mql5.com/en/docs/matrix/matrix_types " MQL5 Documentation: Matrix and vector types") data type , I had not used this before . very helpful to me else where

```
  vector atr_mean;
      atr_mean.CopyIndicatorBuffer(atr_handler,0,0,90); atr_mean.Mean())
```

![Gamuchirai Zororo Ndawana](https://c.mql5.com/avatar/2024/3/6607f08d-4cae.jpg)

**[Gamuchirai Zororo Ndawana](https://www.mql5.com/en/users/gamuchiraindawa)**
\|
2 Feb 2025 at 03:33

**linfo2 [#](https://www.mql5.com/en/forum/479523#comment_55728034):**

Thanks for the code and Idea's , I like how you have structured your code . You have introduced me to the Vector data type , I had not used this before . very helpful to me else where

Thank you Niel, my aim is to blend simplicity with technical rigour,  it's nice to know that it is paying off.

And yeah man the vector class is a game changer, it actually provides  more functionality than we make use of on a day to day.

I'm really looking forward to learn how to use the Matrix class because it allows us to build linear models in just 1 function call.

![Celestine Nwakaeze](https://c.mql5.com/avatar/2024/9/66F8354D-4964.jpg)

**[Celestine Nwakaeze](https://www.mql5.com/en/users/celestinenwakae)**
\|
13 Sep 2025 at 10:39

Thanks so much for this wonderful way of code structure. As a beginner, this article has taken my learning far. More grease to your elbow. God bless you for me. Thanks.


![Ensemble methods to enhance classification tasks in MQL5](https://c.mql5.com/2/108/Ensemble_methods_to_enhance_classification_tasks_in_MQL5___LOGO.png)[Ensemble methods to enhance classification tasks in MQL5](https://www.mql5.com/en/articles/16838)

In this article, we present the implementation of several ensemble classifiers in MQL5 and discuss their efficacy in varying situations.

![Artificial Electric Field Algorithm (AEFA)](https://c.mql5.com/2/83/Artificial_Electric_Field_Algorithm___LOGO.png)[Artificial Electric Field Algorithm (AEFA)](https://www.mql5.com/en/articles/15162)

The article presents an artificial electric field algorithm (AEFA) inspired by Coulomb's law of electrostatic force. The algorithm simulates electrical phenomena to solve complex optimization problems using charged particles and their interactions. AEFA exhibits unique properties in the context of other algorithms related to laws of nature.

![MQL5 Wizard Techniques you should know (Part 52): Accelerator Oscillator](https://c.mql5.com/2/108/MQL5_Wizard_Techniques_you_should_know_Part_52_Accelerator_Oscillator____LOGO.png)[MQL5 Wizard Techniques you should know (Part 52): Accelerator Oscillator](https://www.mql5.com/en/articles/16781)

The Accelerator Oscillator is another Bill Williams Indicator that tracks price momentum's acceleration and not just its pace. Although much like the Awesome oscillator we reviewed in a recent article, it seeks to avoid the lagging effects by focusing more on acceleration as opposed to just speed. We examine as always what patterns we can get from this and also what significance each could have in trading via a wizard assembled Expert Advisor.

![News Trading Made Easy (Part 6): Performing Trades (III)](https://c.mql5.com/2/108/News_Trading_Made_Easy_oPart_6h_Performing_Trades_zIIIs___LOGO.png)[News Trading Made Easy (Part 6): Performing Trades (III)](https://www.mql5.com/en/articles/16170)

In this article news filtration for individual news events based on their IDs will be implemented. In addition, previous SQL queries will be improved to provide additional information or reduce the query's runtime. Furthermore, the code built in the previous articles will be made functional.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/16758&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069693411302246707)

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