---
title: The Kalman Filter for Forex Mean-Reversion Strategies
url: https://www.mql5.com/en/articles/17273
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T17:57:28.774386
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/17273&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068842694245023319)

MetaTrader 5 / Examples


### Introduction

The Kalman filter is a recursive algorithm used in algorithmic trading to estimate the true state of a financial time series by filtering out noise from price movements. It dynamically updates predictions based on new market data, making it valuable for adaptive strategies like mean reversion. This article first introduces the Kalman filter, covering its calculation and implementation. Next, we apply the filter to a classic mean-reversion forex strategy as an example. Finally, we conduct various statistical analyses by comparing the filter with a moving average across different forex pairs.

### The Kalman Filter

The Kalman filter, introduced by Rudolf E. Kalman in 1960, is an optimal recursive estimator used for tracking and predicting dynamic systems. Originally developed for aerospace and control systems, it has been widely applied in finance, robotics, and signal processing. The filter operates in two steps: a prediction step, where it estimates the system’s next state, and an update step, where it refines the estimate based on new observations while minimizing noise.

In the field of algorithmic trading, one can simply see it as a common regime filter that traders normally use, akin to moving average or linear regression models. The Kalman filter adapts dynamically to new data, reduces noise, and efficiently updates estimates in real-time, making it effective for detecting market regime shifts. However, it assumes linear dynamics, requires careful parameter tuning, may lag in detecting abrupt changes, and is computationally more complex than simpler filters like moving averages.

Some common usages for using the Kalman filter in algorithmic trading:

- Mean Reversion Trading: Using current price compared to estimated price as an entry filter.
- Pairs Trading: Dynamically estimates the spread between correlated assets and adjusts hedge ratios based on changing market conditions.
- Trend Following: Filters short-term noise to detect long-term price trends more accurately.
- Volatility Estimation: Provides adaptive estimates of market volatility for risk management and position sizing.

The formula of calculating the Kalman filter value is as follows:

![Kalman formula](https://c.mql5.com/2/121/Kalmam_formula.png)

To understand the complex formula in a simple way, let us turn to a visualization example.

![Kalman Visualization](https://c.mql5.com/2/120/Kalman_visualization.png)

The Kalman filter works by updating its estimate of the true price based on noisy measurements and predictions. It is usually obtained in three steps:

1. Prediction: The filter starts with an initial guess for the price (predicted price) and its uncertainty (predicted covariance). This is shown by the orange Predicted Zone—the range within which the filter expects the true price to be, considering both the previous estimate and the process noise.

2. Update: When new price data (measured prices) is available, the Kalman filter compares it to the predicted price. It then calculates something called the Kalman Gain (purple line) to decide how much weight to give the new measurement versus the prediction. If the measurement is very noisy, the filter trusts its prediction more.

3. Estimate: The filter updates the predicted price by incorporating the new measurement. The updated price (shown by the blue Estimate Zone) has a reduced uncertainty compared to the prediction. This zone shrinks as the filter refines its estimate.


Here, the prediction uncertainty is defined by covariance between the measurement and prediction. A larger covariance means a less confident estimate, while a smaller covariance means the filter is more certain about its estimate. If the measurement is reliable, the Kalman gain is high, and the filter trusts the new data more (shrinking uncertainty). If the measurement is noisy, the gain is lower, and the filter relies more on the previous prediction.

The nosiness is defined by the variance, namely the measurement variance and process variance. They don't self-adjust like covariance, they are determined from the start based on how smooth you want your Kalman filter to be.

Here are some examples of how the variances affect the smoothness of the curve:

![smoothness differences](https://c.mql5.com/2/120/Smoothness_differences.png)

In general, process variance (Q) represents how much the model expects the true state to change over time, while measurement variance (R) reflects the confidence in the observed data. HigherQ makes the filter more responsive to sudden changes, but increases volatility. HigherR smooths predictions by trusting past estimates more, at the cost of delayed adjustments. Low Q and moderate R yield stable predictions, while high Q and low R make the filter more reactive but noisier.

### Coding the Strategy

Mean-reversion strategies often follow the approach of buying when oversold, selling when overbought, and exiting when the price reverts to the mean. This strategy is based on the assumption that price data is generally stable and doesn't tend to reach extreme levels. When the price is at one extreme, it is expected to eventually return to the equilibrium point. This theory holds especially true for semi-stationary data like forex, where mean-reversion strategies have been profitable over the years.

We will quantify our approach using 100-period Bollinger Bands with a 2.0 deviation. The detailed plan is as follows:

- Buy when the last close price is lower than the lower band.
- Sell when the last close price is higher than the upper band.
- Close the position whenever the price crosses the middle band.
- Only one position at a time to avoid over-trading.
- Set the stop loss gap at 1% of price to avoid [fat tail risk](https://en.wikipedia.org/wiki/Tail_risk#:~:text=Tail%20risk%2C%20sometimes%20called%20%22fat,risk%20of%20a%20normal%20distribution. "https://en.wikipedia.org/wiki/Tail_risk#:~:text=Tail%20risk%2C%20sometimes%20called%20%22fat,risk%20of%20a%20normal%20distribution.") that is common for mean-reversion strategies.

We plan to trade forex pairs on the 15-minute timeframe, a common timeframe that provides enough trades while ensuring decent trade quality.

We always start by defining the necessary functions first, as this makes the trade logic coding easier later on. In this case, we only need to code the buy and sell functions as follows:

```
#include <Trade/Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
//| Buy Function                                                     |
//+------------------------------------------------------------------+
void executeBuy(string symbol) {
       double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
       double lots=0.01;
       double sl = ask*(1-0.01);
       trade.Buy(lots,symbol,ask,sl);
}

//+------------------------------------------------------------------+
//| Sell Function                                                    |
//+------------------------------------------------------------------+
void executeSell(string symbol) {
       double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
       double lots=0.01;
       double sl = bid*(1+0.01);
       trade.Sell(lots,symbol,bid,sl);
}
```

Then, we initialize the global variables and the initializer here. This initializes the magic number for the expert advisor, as well as the Bollinger Band's handle that we will use later.

```
input int Magic = 0;
input int bbPeriod = 100;
input double d = 2.0;

int barsTotal = 0;
int handleMa;

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
{   handleBb = iBands(_Symbol,PERIOD_CURRENT,bbPeriod,0,d,PRICE_CLOSE);
    trade.SetExpertMagicNumber(Magic);
    return INIT_SUCCEEDED;
}
```

Finally, in the OnTick() function, we use this to ensure that we only process the trade logic every bar instead of every tick:

```
int bars = iBars(_Symbol,PERIOD_CURRENT);

if (barsTotal!= bars){
   barsTotal = bars;
```

We obtain the current values of the Bollinger Bands by creating buffer arrays that can store the current values by copying from the handle.

```
double bbLower[], bbUpper[], bbMiddle[];
CopyBuffer(handleBb,UPPER_BAND,1,1,bbUpper);
CopyBuffer(handleBb,LOWER_BAND,1,1,bbLower);
CopyBuffer(handleBb,0,1,1,bbMiddle);
```

This check loops through all the current opened positions in the trading account to check whether the position is opened by this specific EA. If there is already a position opened by this EA, we set the _NotInPosition_ variable to false. For every opened position that have reverted back to the middle band, we close it.

```
bool NotInPosition = true;
for(int i = 0; i<PositionsTotal(); i++){
    ulong pos = PositionGetTicket(i);
    string symboll = PositionGetSymbol(i);
    if(PositionGetInteger(POSITION_MAGIC) == Magic&&symboll== _Symbol){
       NotInPosition = false;
       if((PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY&&price>bbMiddle[0])
       ||(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL&&price<bbMiddle[0]))trade.PositionClose(pos);
  }
}
```

The final trade logic will be carried out like this:

```
if(price<bbLower[0]&&NotInPosition) executeBuy(_Symbol);
if(price>bbUpper[0]&&NotInPosition) executeSell(_Symbol);
```

Compile the EA and head to strategy tester visualizer to see if the EA is working as expected.

A typical trade in the visualizer should look like this:

![trade example](https://c.mql5.com/2/120/trade_example.png)

We then go back to MetaEditor and code the regime filters.

First, for the 500-EMA, we want the mean-reversion strategy to align with the trend when entering, as an additional confirmation. We add these lines to the original EA:

```
int handleMa;
handleMa = iMA(_Symbol,PERIOD_CURRENT,maPeriod,0,MODE_EMA,PRICE_CLOSE);

double ma[];
CopyBuffer(handleMa,0,1,1,ma);

if(price<bbLower[0]&&price>ma[0]&&NotInPosition) executeBuy(_Symbol);
if(price>bbUpper[0]&&price<ma[0]&&NotInPosition) executeSell(_Symbol);
```

After that, we code the function for getting the Kalman filter value:

```
//+------------------------------------------------------------------+
//| Kalman Filter Function                                           |
//+------------------------------------------------------------------+
double KalmanFilter(double price,double measurement_variance,double process_variance)
{
    // Prediction step (state does not change)
    double predicted_state = prev_state;
    double predicted_covariance = prev_covariance + process_variance;

    // Kalman gain calculation
    double kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance);

    // Update step (incorporate new price observation)
    double updated_state = predicted_state + kalman_gain * (price - predicted_state);
    double updated_covariance = (1 - kalman_gain) * predicted_covariance;

    // Store updated values for next iteration
    prev_state = updated_state;
    prev_covariance = updated_covariance;

    return updated_state;
}
```

The function follows the recursive procedure like this [diagram](https://www.mql5.com/go?link=https://www.google.com/url?sa=i%26url=https%253A%252F%252Farshren.medium.com%252Fan-easy-explanation-of-kalman-filter-ec2ccb759c46%26psig=AOvVaw3YrX4RDAY1EhNKIaBvqeUj%26ust=1740141490776000%26source=images%26cd=vfe%26opi=89978449%26ved=0CBcQjhxqFwoTCJilwLSi0osDFQAAAAAdAAAAABAE "https://www.google.com/url?sa=i&url=https%3A%2F%2Farshren.medium.com%2Fan-easy-explanation-of-kalman-filter-ec2ccb759c46&psig=AOvVaw3YrX4RDAY1EhNKIaBvqeUj&ust=1740141490776000&source=images&cd=vfe&opi=89978449&ved=0CBcQjhxqFwoTCJilwLSi0osDFQAAAAAdAAAAABAE"):

![procedure diagram](https://c.mql5.com/2/120/procedure_diagram.png)

To implement the Kalman regime filter in the EA, we add these lines to the OnTick() function:

```
double kalman = KalmanFilter(price,mv,pv);
if(price<bbLower[0]&&price>kalman&&NotInPosition) executeBuy(_Symbol);
if(price>bbUpper[0]&&price<kalman&&NotInPosition) executeSell(_Symbol);
```

The Kalman filter works by continuously updating its estimate of the true price, smoothing out noise, and adapting to price movements over time. It essentially acts as a price predictor. When the price falls below the lower band of the Bollinger Bands, it signals that the market is oversold and is expected to revert to the mean. Here, we use the Kalman filter as a confirmation of the reversal. In the oversold scenario, if the price is higher than the Kalman estimate, it suggests that the price has already shown signs of potential upward movement. In the sell scenario, the reverse is true.

However, while the moving average is also a common regime filter, its purpose is slightly different from that of the Kalman filter. Moving averages serve as a trend indicator, with the position of the price relative to the MA signaling the current trend direction.

The full code is as follows:

```
#include <Trade/Trade.mqh>
CTrade trade;

input double mv = 10;
input double pv = 1.0;
input int Magic = 0;
input int bbPeriod = 100;
input double d = 2.0;
input int maPeriod = 500;

double prev_state;       // Previous estimated price
double prev_covariance = 1;  // Previous covariance (uncertainty)
int barsTotal = 0;
int handleMa;
int handleBb;

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
{   handleMa = iMA(_Symbol,PERIOD_CURRENT,maPeriod,0,MODE_EMA,PRICE_CLOSE);
    handleBb = iBands(_Symbol,PERIOD_CURRENT,bbPeriod,0,d,PRICE_CLOSE);
    prev_state = iClose(_Symbol,PERIOD_CURRENT,1);
    trade.SetExpertMagicNumber(Magic);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Deinitializer function                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }

//+------------------------------------------------------------------+
//| OnTick Function                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
  int bars = iBars(_Symbol,PERIOD_CURRENT);

  if (barsTotal!= bars){
     barsTotal = bars;
     bool NotInPosition = true;
     double price = iClose(_Symbol,PERIOD_CURRENT,1);
     double bbLower[], bbUpper[], bbMiddle[];
     double ma[];
     double kalman = KalmanFilter(price,mv,pv);

     CopyBuffer(handleMa,0,1,1,ma);
     CopyBuffer(handleBb,UPPER_BAND,1,1,bbUpper);
     CopyBuffer(handleBb,LOWER_BAND,1,1,bbLower);
     CopyBuffer(handleBb,0,1,1,bbMiddle);

     for(int i = 0; i<PositionsTotal(); i++){
         ulong pos = PositionGetTicket(i);
         string symboll = PositionGetSymbol(i);
         if(PositionGetInteger(POSITION_MAGIC) == Magic&&symboll== _Symbol){
            NotInPosition = false;
            if((PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY&&price>bbMiddle[0])
            ||(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL&&price<bbMiddle[0]))trade.PositionClose(pos);
      }
    }

     if(price<bbLower[0]&&price>kalman&&NotInPosition) executeBuy(_Symbol);
     if(price>bbUpper[0]&&price<kalman&&NotInPosition) executeSell(_Symbol);
    }
}

//+------------------------------------------------------------------+
//| Kalman Filter Function                                           |
//+------------------------------------------------------------------+
double KalmanFilter(double price,double measurement_variance,double process_variance)
{
    // Prediction step (state does not change)
    double predicted_state = prev_state;
    double predicted_covariance = prev_covariance + process_variance;

    // Kalman gain calculation
    double kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance);

    // Update step (incorporate new price observation)
    double updated_state = predicted_state + kalman_gain * (price - predicted_state);
    double updated_covariance = (1 - kalman_gain) * predicted_covariance;

    // Store updated values for next iteration
    prev_state = updated_state;
    prev_covariance = updated_covariance;

    return updated_state;
}

//+------------------------------------------------------------------+
//| Buy Function                                                     |
//+------------------------------------------------------------------+
void executeBuy(string symbol) {
       double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
       double lots=0.01;
       double sl = ask*(1-0.01);
       trade.Buy(lots,symbol,ask,sl);
}

//+------------------------------------------------------------------+
//| Sell Function                                                    |
//+------------------------------------------------------------------+
void executeSell(string symbol) {
       double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
       double lots=0.01;
       double sl = bid*(1+0.01);
       trade.Sell(lots,symbol,bid,sl);
}
```

To change the regime filter, simply adjust the final buy/sell criteria.

### Statistical Analysis

Compile the EA and go to the MetaTrader 5 terminal, on the top left, click View->Symbols->Forex and select "Show Symbol" for all major pairs and minor pairs. This puts them in your market watch list for later market scanning.

![show symbol](https://c.mql5.com/2/120/show_symbol.png)

Then, we go to the Market Scanner in the Strategy Tester section and backtest the strategy using data from the past 3 years. This will help us get a sense of whether the regime filters improve profitability for the majority of the forex pairs we might be trading.

![market scanner setting](https://c.mql5.com/2/120/market_scanner_setting.png)

![parameters](https://c.mql5.com/2/120/parameters.png)

Of course, the number of filtered trades depends on the parameters of the indicators. In this study, we use commonly applied parameter values that filter a similar number of trades: a 500-period exponential moving average and a measurement variance of 10 and process variance of 1 for the Kalman filter. Readers are encouraged to fine-tune the parameters for the most effective results.

We first test the result with no regime filters involved as a baseline. We expect that, on average, the EA with regime filters should outperform the majority of the baseline results.

The result of the top-performing forex pairs shows something like this:

![baseline performance](https://c.mql5.com/2/120/baseline_performance.png)

![distribution of baseline](https://c.mql5.com/2/120/distribution_baseline.png)

We see that, on average, the strategy executes 800+ trades in the past 3 years for each pair, providing enough samples to imply generality in the conclusions. The distribution is mostly scattered around a 0.8–1.1 profit factor, which is decent, but no pairs exceed the 1.1 profit factor or 1 Sharpe ratio mark. Overall, the raw strategy works across many forex pairs in recent years, but the profitability isn't particularly impressive. Keep this in mind as we compare it with the performance when filters are applied.

Next, we backtest the strategy with the moving average filter involved. Here are the results:

![ma performance](https://c.mql5.com/2/120/ma_performance.png)

![distribution of ma](https://c.mql5.com/2/120/distribution_ma.png)

We see that, by using the moving average filter, we filtered out about 70% of the original trades, leaving approximately 250 trades for each pair. Additionally, the filtered trades are, on average, of higher quality compared to the baseline. Most forex pairs hover between a 0.9 and 1.2 profit factor, with the best-performing pair having a 1.33 profit factor and a 2.34 Sharpe ratio. This suggests that using the moving average as a filter has overall improved the profitability of this classic mean-reversion strategy.

Now, let's address the elephant in the room and see how the strategy performs with the Kalman filter.

![kalman performance](https://c.mql5.com/2/120/kalman_performance.png)

![distribution of kalman](https://c.mql5.com/2/120/distribution_kalman.png)

The Kalman filter filtered out about 60% of the original trades, leaving about 350 trades for each pair. From the distribution, we see that most profit factors stay between 0.85 and 1.2, which is similar to the moving average performance and better than the baseline performance. Moreover, considering the overall number of forex pairs with a 1.0+ profit factor and 1.2+ profit factor, we may conclude that both the moving average and the Kalman filter are similar in terms of improving the average trade quality for this strategy. The Kalman filter is not superior to the moving average in this scenario, suggesting that complexity does not always result in better performance.

We mentioned that the use of the Kalman filter differs slightly from the use of the moving average in terms of filtering logic. However, in this case, they seem to perform similarly in terms of filtering out bad trades. To investigate whether they are filtering similar trades, we will analyze the differences between their filtered trades to determine whether the effect of the Kalman filter is merely the same as the EMA.

For reference, we will choose the AUDUSD forex pair, as it has been the best performing pair across all two conditions mentioned above.

The backtest result of the baseline:

![baseline equity curve](https://c.mql5.com/2/120/baseline_curve.png)

![baseline result](https://c.mql5.com/2/120/baseline_result.png)

The backtest result of the exponential moving average filter:

![moving average equity curve](https://c.mql5.com/2/120/ma_curve.png)

![moving average result](https://c.mql5.com/2/120/ma_result.png)

The result of the Kalman filter:

![Kalman equity curve](https://c.mql5.com/2/120/kalman_curve.png)

![Kalman result](https://c.mql5.com/2/120/kalman_result.png)

The first thing we can notice is that the win rate of the moving average version is significantly higher than the baseline or the Kalman version, while its average profit is lower and its average loss is higher than the others. This already suggests that the moving average version is taking very different trades than the Kalman version. To further investigate, we obtain the backtest excel report by right-clicking on the backtest result page:

![excel report](https://c.mql5.com/2/120/ExcelReport.png)

For each report, we note the row number of the "Deals" sign:

![find row](https://c.mql5.com/2/120/find_row.png)

Next, we go into Python or Jupyter Notebook. We copy and paste the following code, change the _skiprow_ number to each Excel report's "Deals" row number, and we’re done.

```
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
df1 = pd.read_excel("baseline.xlsx", skiprows=1805)
df2 = pd.read_excel("ma.xlsx", skiprows =563 )
df3 = pd.read_excel("kalman.xlsx",skiprows = 751)
df1 = df1[['Time']][1:-1]
df1 = df1[df1.index % 2 == 0]  # Filter for rows with odd indices
df2 = df2[['Time']][1:-1]
df2 = df2[df2.index % 2 == 0]
df3 = df3[['Time']][1:-1]
df3 = df3[df3.index % 2 == 0]

# Convert "Time" columns to datetime
df1['Time'] = pd.to_datetime(df1['Time'])
df2['Time'] = pd.to_datetime(df2['Time'])
df3['Time'] = pd.to_datetime(df3['Time'])

# Find intersections
set1 = set(df1['Time'])
set2 = set(df2['Time'])
set3 = set(df3['Time'])

# Create the Venn diagram
venn_labels = {
    '100': len(set1 - set2 - set3),  # Only in df1
    '010': len(set2 - set1 - set3),  # Only in df2
    '001': len(set3 - set1 - set2),  # Only in df3
    '110': len(set1 & set2 - set3),  # In df1 and df2
    '011': len(set2 & set3 - set1),  # In df2 and df3
    '101': len(set1 & set3 - set2),  # In df1 and df3
    '111': len(set1 & set2 & set3)   # In all three
}

# Plot the Venn diagram
plt.figure(figsize=(8, 8))
venn3(subsets=venn_labels, set_labels=('Baseline', 'EMA', 'Kalman'))
plt.title("Venn Diagram of Time Overlap")
plt.show()
```

The logic of this code is essentially that we store the position exit times into three dataframes for each version by skipping rows and selecting rows with even indices. Then, we store each data frame into a set and obtain the Venn diagram by comparing whether the times overlap. The graph will show the number of trades in each region, separated by different colors. Note that even the baseline won't contain all the trades that the EMA and Kalman versions have because we set the strategy to trade only one at a time, which causes it to miss out on some trades that the other versions have.

Here’s the output Venn diagram:

![overlap](https://c.mql5.com/2/120/Overlap.png)

Observing the area where the Kalman and MA overlap, we find that among the hundreds of trades each version took, only 71 trades were the same. This suggests that the filter effects of these two are very dissimilar, despite filtering out a similar number of trades from the original strategy. This further emphasizes the importance of studying and utilizing the Kalman filter, as it provides a unique filtering option that differs from common trend filters.

### Conclusion

In this article, we introduced an advanced recursive algorithm for algorithmic trading called the Kalman filter. We started by explaining its mechanisms and implementations, providing visualizations and formulas. Then, we walked through the entire process of developing a forex mean-reversion strategy and implementing the Kalman filter in MQL5. Finally, we conducted various statistical analyses, including market scanning, backtesting, and overlapping comparisons, to evaluate its filtering ability in comparison to the moving average and baseline.

In actual trading, the Kalman filter is widely used by top quantitative trading institutions but remains lesser known in the retail trading space. This article aims to provide insights and practical implementation for the MQL5 community, offering an approach to evaluate the filtering ability of the Kalman filter, so it can be better integrated into future strategy development. Readers are encouraged to experiment with this framework themselves and incorporate it into their own trading arsenal.

**File Table**

| File Name | File Usage |
| --- | --- |
| Kalman visualizations.ipynb | The python code for visualizations used in this article |
| MR-Kalman.mq5 | The expert advisor code |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17273.zip "Download all attachments in the single ZIP archive")

[MR-Kalman.mq5](https://www.mql5.com/en/articles/download/17273/mr-kalman.mq5 "Download MR-Kalman.mq5")(4.22 KB)

[Kalman\_visualizations.ipynb](https://www.mql5.com/en/articles/download/17273/kalman_visualizations.ipynb "Download Kalman_visualizations.ipynb")(454.97 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Decoding Opening Range Breakout Intraday Trading Strategies](https://www.mql5.com/en/articles/17745)
- [Day Trading Larry Connors RSI2 Mean-Reversion Strategies](https://www.mql5.com/en/articles/17636)
- [Exploring Advanced Machine Learning Techniques on the Darvas Box Breakout Strategy](https://www.mql5.com/en/articles/17466)
- [Robustness Testing on Expert Advisors](https://www.mql5.com/en/articles/16957)
- [Trend Prediction with LSTM for Trend-Following Strategies](https://www.mql5.com/en/articles/16940)
- [The Inverse Fair Value Gap Trading Strategy](https://www.mql5.com/en/articles/16659)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/482147)**
(4)


![Too Chee Ng](https://c.mql5.com/avatar/2025/6/68446669-e598.png)

**[Too Chee Ng](https://www.mql5.com/en/users/68360626)**
\|
29 Apr 2025 at 09:20

Like your presentation.

Thank you very much. Please keep it up.

![Too Chee Ng](https://c.mql5.com/avatar/2025/6/68446669-e598.png)

**[Too Chee Ng](https://www.mql5.com/en/users/68360626)**
\|
29 Apr 2025 at 10:21

```
Low Q and moderate R yield stable predictions, while high Q and low R make the filter more reactive but noisier.
```

What is your view in optimizing these inputs (Q and R)?

How would you decide their values for the EA?

![Zhuo Kai Chen](https://c.mql5.com/avatar/2024/11/6743e84b-8a3d.jpg)

**[Zhuo Kai Chen](https://www.mql5.com/en/users/sicklemql)**
\|
30 Apr 2025 at 01:20

**Too Chee Ng [#](https://www.mql5.com/en/forum/482147#comment_56574080):**

Like your presentation.

Thank you very much. Please keep it up.

Thank you! I'm will keep improving my article quality as I'm learning more.

![Zhuo Kai Chen](https://c.mql5.com/avatar/2024/11/6743e84b-8a3d.jpg)

**[Zhuo Kai Chen](https://www.mql5.com/en/users/sicklemql)**
\|
30 Apr 2025 at 01:29

**Too Chee Ng [#](https://www.mql5.com/en/forum/482147#comment_56574428):**

What is your view in optimizing these inputs (Q and R)?

How would you decide their values for the EA?

Great question! I would say do not try too hard to optimize the values specifically. Try to select some standard values and optimize the threshold rather than optimize the indicator parameters. I would recommend you choose the measurement variance from 1000, 100, and 10, and choose the process variance from 1, 0.1 and 0.01.

![Automating Trading Strategies in MQL5 (Part 10): Developing the Trend Flat Momentum Strategy](https://c.mql5.com/2/122/Automating_Trading_Strategies_in_MQL5_Part_10__LOGO.png)[Automating Trading Strategies in MQL5 (Part 10): Developing the Trend Flat Momentum Strategy](https://www.mql5.com/en/articles/17247)

In this article, we develop an Expert Advisor in MQL5 for the Trend Flat Momentum Strategy. We combine a two moving averages crossover with RSI and CCI momentum filters to generate trade signals. We also cover backtesting and potential enhancements for real-world performance.

![William Gann methods (Part II): Creating Gann Square indicator](https://c.mql5.com/2/89/logo-midjourney_image_15566_400_3863__3.png)[William Gann methods (Part II): Creating Gann Square indicator](https://www.mql5.com/en/articles/15566)

We will create an indicator based on the Gann's Square of 9, built by squaring time and price. We will prepare the code and test the indicator in the platform on different time intervals.

![Neural Network in Practice: Sketching a Neuron](https://c.mql5.com/2/88/Neural_network_in_practice_Sketching_a_neuron___LOGO.png)[Neural Network in Practice: Sketching a Neuron](https://www.mql5.com/en/articles/13744)

In this article we will build a basic neuron. And although it looks simple, and many may consider this code completely trivial and meaningless, I want you to have fun studying this simple sketch of a neuron. Don't be afraid to modify the code, understanding it fully is the goal.

![From Basic to Intermediate: Operators](https://c.mql5.com/2/88/From_basic_to_intermediate_Operators___LOGO.png)[From Basic to Intermediate: Operators](https://www.mql5.com/en/articles/15305)

In this article we will look at the main operators. Although the topic is simple to understand, there are certain points that are of great importance when it comes to including mathematical expressions in the code format. Without an adequate understanding of these details, programmers with little or no experience eventually give up trying to create their own solutions.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/17273&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068842694245023319)

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