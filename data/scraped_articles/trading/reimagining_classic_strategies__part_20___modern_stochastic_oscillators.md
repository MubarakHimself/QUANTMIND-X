---
title: Reimagining Classic Strategies (Part 20): Modern Stochastic Oscillators
url: https://www.mql5.com/en/articles/20530
categories: Trading, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T17:51:47.916247
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/20530&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068734151831518465)

MetaTrader 5 / Examples


The stochastic oscillator is a well-known technical indicator traditionally used to identify potential market reversals. In its classical form, it measures the momentum of price movements within a defined range. When price action pushes into extreme territories, the market is generally considered overbought or oversold. Under this conventional interpretation, traders look for selling opportunities in overbought conditions and buying opportunities in oversold conditions, based on the assumption that prices will eventually return toward equilibrium.

This approach has been used successfully for many years. However, this article suggests that the stochastic oscillator may have overlooked capabilities—specifically, that it may function more effectively as a trend-following indicator rather than strictly a mean-reversion tool. We show that, with slight adjustments to the interpretive rules, the oscillator can be repurposed into a useful method for identifying dominant market trends.

To support this, we re-examine the indicator’s signals and challenge the classical rules. We introduce an alternative framework that encourages buying in overbought conditions and selling in oversold conditions—an idea that may seem counterintuitive at first. Yet, as we demonstrate, the stochastic oscillator is more versatile than commonly assumed and is not limited to a single mode of interpretation.

In earlier analysis, we also found that the oscillator was more predictable than raw price movements. This insight motivated us to look deeper into the indicator in search of additional, previously untapped value.

To explore this fully, we present five distinct versions of a stochastic-based strategy. Although our final attempt to extract maximal value from the indicator was not successful, this result highlights a key point: four out of the five strategies performed well. This prompts us to reconsider how well we truly understand an indicator that many traders believe they already know.

Because we evaluate multiple versions of the strategy, several parameters must remain consistent across all of them. Each version will execute trades one at a time, ensuring that differences in profitability stem only from changes in the trading rules. All strategies will use the same position size, and every backtest will run over the identical historical window from 2021 through 2025.

![](https://c.mql5.com/2/186/Screenshot_2025-12-14_at_00_15_55__1.png)

Figure 1: The backtest dates we will use for all versions of our strategy

Finally, each version will be tested under randomized execution delays to approximate real-world trading latency.

![](https://c.mql5.com/2/186/Screenshot_2025-12-14_at_00_16_20__1.png)

Figure 2: The test conditions we will test our strategies under emulate real market conditions

### Establishing A Baseline

We will start off by first defining important global variables we need for our discussion.

```
//+------------------------------------------------------------------+
//|                                          Stochastic Strategy.mq5 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
int      stoch_handler,atr_handler;
double   stoch_main_reading[],stoch_signal_reading[],atr_reading[];
double   bid,ask;
```

Next, we import the Trade library to assist with managing our positions.

```
//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;
```

When the trading application is initialized for the first time, we define the technical indicators required for our strategy.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Setup our technical indicators
atr_handler    = iATR(Symbol(),PERIOD_D1,14);
stoch_handler  = iStochastic(Symbol(),PERIOD_D1,5,3,3,MODE_SMA,STO_LOWHIGH);
//---
   return(INIT_SUCCEEDED);
  }
```

When the trading application is no longer in use, we release the technical indicators that were relied upon.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   IndicatorRelease(atr_handler);
   IndicatorRelease(stoch_handler);
  }
```

Each time new price levels are received, the appropriate indicator buffers and global variables are updated. We first implement the traditional interpretation of the stochastic oscillator: buying during oversold conditions and selling during overbought conditions.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Keep track of the time
datetime        current_time   = iTime(Symbol(),PERIOD_D1,0);
static datetime time_stamp;

   if(current_time != time_stamp)
      {
         //--- Update the time
         time_stamp = current_time;

         //--- Update our technical indicators
         CopyBuffer(stoch_handler,0,0,1,stoch_main_reading);
         CopyBuffer(stoch_handler,1,0,1,stoch_signal_reading);
         CopyBuffer(atr_handler,0,0,1,atr_reading);
         ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
         bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);

         //--- Trading rules
         if(PositionsTotal() == 0)
            {
               if(stoch_main_reading[0] < 20) Trade.Buy(0.01,Symbol(),ask,(ask - (atr_reading[0]*2)),(ask + (atr_reading[0]*2)));

               if(stoch_main_reading[0] > 80) Trade.Sell(0.01,Symbol(),bid,(bid + (atr_reading[0]*2)),(bid - (atr_reading[0]*2)));
            }
      }

  }
//+------------------------------------------------------------------+
```

The equity curve produced by this benchmark is volatile and provides little confidence in the integrity of the proposed strategy. Normally, such a strategy would be abandoned, but we propose that all hope is not lost.

![](https://c.mql5.com/2/184/Screenshot_2025-12-05_at_21_46_59.png)

Figure 3: The equity curve produced by the traditional interpretation of the stochastic oscilator appear unreliable

Additionally, the strategy lost trades 49% of the time, resulting in a depreciation of investor equity over the backtest period.

![](https://c.mql5.com/2/184/Screenshot_2025-12-05_at_21_48_11.png)

Figure 4: The detailed statistics of the classical strategy leaves considerable room for improvement

### Improving Beyond The Baseline

In our proposed solution, we reinterpret the indicator as a tool for trend identification. The dominant trend that we identify in any market will be defined by where current market prices fall, with respect to the observed range of the market.We begin by creating vectors to record previously observed highs and lows.

```
vector   high,low;
```

We then calculate the midpoint of the trading range over the previous 90 daily candles. The value of 90 was chosen arbitrarily, as it aligns with typical business cycles of institutional market participants that dominate the forex markets.

```
//--- Calculate the middle of the trading range
high.CopyRates(Symbol(),PERIOD_D1,COPY_RATES_HIGH,0,90);
low.CopyRates(Symbol(),PERIOD_D1,COPY_RATES_LOW,0,90);
double mid = ((high.Mean() + low.Mean())/2);
```

If no positions are open, we first seek to enter long trades during overbought price levels. In addition, we require further confirmation by observing the close price above the midpoint of the observed high–low range.

```
//--- Trading rules
if(PositionsTotal() == 0)
   {
      if((stoch_main_reading[0] > 80) && (iClose(Symbol(),PERIOD_D1,0) > mid)) Trade.Buy(0.01,Symbol(),ask,(ask - (atr_reading[0]*2)),(ask + (atr_reading[0]*2)));

      if((stoch_main_reading[0] < 20) && (iClose(Symbol(),PERIOD_D1,0) < mid)) Trade.Sell(0.01,Symbol(),bid,(bid + (atr_reading[0]*2)),(bid - (atr_reading[0]*2)));
   }
```

The equity curve realised under these conditions now demonstrates material upward growth when compared to the volatile and underperforming benchmark with which we began.

![](https://c.mql5.com/2/184/Screenshot_2025-12-05_at_14_09_07.png)

Figure 5: The equity curve produced by our refined application of the stochastic oscilator

The detailed statistics obtained show significant improvement. Total net profit increased dramatically to $184.35, compared to a benchmark performance of $-26.22. Additionally, the proportion of winning trades rose from 49% in the initial attempt to 55% at present.

![](https://c.mql5.com/2/184/Screenshot_2025-12-05_at_14_10_07.png)

Figure 6: The detailed statistics we have obtained from our revised strategy demonstrate the changes we made were meaningful

### Reaching For Higher Performance Levels

We can still make meaningful improvements to the strategy by carefully performing candlestick analysis on lower time frames. The reasoning is that a trend observed on the daily time frame should also be reflected in the spread between the open and close prices on a lower time frame, helping us identify meaningful entry points.

```
vector  open,close;
```

On the 1-hour timeframe, we will copy the last 12 candles, to determine the dominant trend of the day.

```
//--- Calculate the current trend on the lower time frame
 open.CopyRates(Symbol(),PERIOD_H1,COPY_RATES_OPEN,0,12);
 close.CopyRates(Symbol(),PERIOD_H1,COPY_RATES_CLOSE,0,12);
```

When combined, our revised trading rules form a filter consisting of three requirements. Each requirement reinforces the idea that a single dominant trend may be active in the market.

```
//--- Trading rules
if(PositionsTotal() == 0)
   {
      if((stoch_main_reading[0] > 80) && (iClose(Symbol(),PERIOD_D1,0) > mid) && (open.Mean() < close.Mean())) Trade.Buy(0.01,Symbol(),ask,(ask - (atr_reading[0]*2)),(ask + (atr_reading[0]*2)));

      if((stoch_main_reading[0] < 20) && (iClose(Symbol(),PERIOD_D1,0) < mid) && (open.Mean() > close.Mean())) Trade.Sell(0.01,Symbol(),bid,(bid + (atr_reading[0]*2)),(bid - (atr_reading[0]*2)));
   }
```

The equity curve produced by this refined strategy now grows more smoothly, in contrast to the jagged and volatile equity curve obtained from our initial attempt.

![](https://c.mql5.com/2/184/Screenshot_2025-12-05_at_14_02_46.png)

Figure 7: Our revised strategy is performing well above our previous best score

Total net profit has increased further to $223, while the Sharpe ratio improved to 0.88 from a previous value of 0.69. The total number of trades decreased from 123 to 118. A clear indication of improved efficiency is the ability to achieve the same objective with less effort. The changes implemented appear to have successfully achieved this outcome. Additionally, the percentage of winning trades increased to a new high of 56%.

![](https://c.mql5.com/2/184/Screenshot_2025-12-05_at_14_04_22.png)

Figure 8: The detailed results produced by our third iteration of our stochastic oscillator strategy

### Algorithmic Discovery of Trading Rules From The Stochastic Oscillator

So far in our discussion, we have been manually defining trading rules and market filters to guide trade execution. While this process is a valuable exercise in creative thinking and market reasoning, there is a natural limit to how far human intuition alone can take us.

There may exist meaningful patterns, rules, and decision-making logic within the market data that are not immediately intuitive or easily recognized by human reasoning. To explore this possibility, we now turn our attention to algorithmic methods for discovering additional rules—specifically, rules for interpreting the stochastic oscillator.

To do this, we will write an MQL5 script that retrieves historical EUR/USD market data along with values from the Stochastic Oscillator indicator. This data will then be exported in CSV format. The dataset will include the standard open, high, low, and close (OHLC) price feeds, followed by the stochastic indicator buffers: the %K (main) line and the %D (signal) line.

Finally, we perform manual feature engineering on the stochastic data to enrich the dataset. This includes calculating the midpoint between the %K and %D lines, measuring the distance of the main reading from the 80 and 20 threshold levels, and deriving additional observations that help capture the indicator’s behavior. Together, these features form a high-dimensional representation of the stochastic oscillator, enabling more sophisticated algorithmic analysis.

```
//+------------------------------------------------------------------+
//|                                          Fetch Data Stochastic 2 |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs

//--- File name
string file_name = Symbol() + " Stochastic Strategy.csv";

int stoch_handler = iStochastic(Symbol(),PERIOD_CURRENT,5,3,3,MODE_EMA,STO_LOWHIGH);
double stoch_main[],stoch_signal[];
double stoch_o,stoch_h,stoch_l;

//--- Amount of data requested
input int size = 365;

//+------------------------------------------------------------------+
//| Our script execution                                             |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- Write to file
   int file_handle=FileOpen(file_name,FILE_WRITE|FILE_ANSI|FILE_CSV,",");

//---
   CopyBuffer(stoch_handler,0,0,size,stoch_main);
   stoch_o = stoch_main[0];
   stoch_h = stoch_main[0];
   stoch_l = stoch_main[0];
   ArraySetAsSeries(stoch_main,true);
   CopyBuffer(stoch_handler,1,0,size,stoch_signal);
   ArraySetAsSeries(stoch_signal,true);

   for(int i=size;i>=1;i--)
     {
      if(i == size)
        {

         FileWrite(file_handle,
                  //--- Time
                  "Time",
                   //--- OHLC
                   "Open",
                   "High",
                   "Low",
                   "Close",
                   //--- Stochastic Readings
                   "Stochastic Main",
                   "Stochastic Signal",
                   //--- Feature Engineering Stochastic Oscilator
                   "Stoch Main - Signal",
                   "Stoch M-S Mid",
                   "Stoch - 80",
                   "Stoch - 20",
                   "Stoch O",
                   "Stoch H",
                   "Stoch L",
                   "Stoch O-C",
                   "Stoch H-C",
                   "Stoch L-C"
                  );
        }

      else
        {

        //--- Set features
        stoch_h = (stoch_h < stoch_main[i]) ? stoch_main[i] : stoch_h;
        stoch_l = (stoch_l > stoch_main[i]) ? stoch_main[i] : stoch_l;

         FileWrite(file_handle,
                   iTime(_Symbol,PERIOD_CURRENT,i),
                   //--- OHLC
                   iOpen(_Symbol,PERIOD_CURRENT,i),
                   iHigh(_Symbol,PERIOD_CURRENT,i),
                   iLow(_Symbol,PERIOD_CURRENT,i),
                   iClose(_Symbol,PERIOD_CURRENT,i),
                   //--- Stochastic Readings
                   stoch_main[i],
                   stoch_signal[i],
                   //--- Stochastic Feature Engineering
                   stoch_main[i] - stoch_signal[i],
                   ((stoch_main[i] + stoch_signal[i])/2),
                   (stoch_main[i] - 80),
                   (stoch_main[i] - 20),
                   stoch_o,
                   stoch_h,
                   stoch_l,
                   stoch_o - stoch_main[i],
                   stoch_h - stoch_main[i],
                   stoch_l - stoch_main[i]
                   );
        }
     }
//--- Close the file
   FileClose(file_handle);
  }
//+------------------------------------------------------------------+
```

### Analyzing Our Market Data in Python

Now that we have written out our CSV of historical market data, we are ready to analyze the data in Python. First, import the Python libraries we need.

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

Next, we read in the CSV file we wrote.

```
data = pd.read_csv("./EURUSD Stochastic Strategy.csv")
data
```

Separate the training set of observations, from the test set, which is reserved for backtesting in MetaTrader 5.

```
train = data.iloc[:-(365 * 5),:]
test = data.iloc[-(365 * 5):,:]
```

Load in the machine learning libraries we need.

```
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import cross_val_score,TimeSeriesSplit
```

Define the forecasting horizon.

```
HORIZON = 5
```

Create a time series cross-validation object that will help us to assess the accuracy of each model we consider.

```
tscv = TimeSeriesSplit(n_splits=10,gap=HORIZON)
```

Label our dataset. We are interested in forecasting the stochastic main oscillator reading.

```
data['Target'] = data['Stochastic Main'].shift(-HORIZON)
data = data.iloc[:-HORIZON,:]
```

Let us compare the accuracy obtained by using the classical OHLC columns, the new stochastic features we have engineered, and lastly a combination of both.

```
X_classic = data.iloc[:,1:7].columns
X_new     = data.iloc[:,7:-1].columns
X_all     = data.iloc[:,1:-1].columns
y         = 'Target'
```

We will keep record of the accuracy levels we have observed as we change the inputs we feed our model.

```
scores = []
```

Keep the underlying model the same, to ensure that changes in error are coming from the inputs we have selected.

```
model = LinearRegression()
```

Record the accuracy associated with each possible set of inputs we have available.

```
scores.append(np.mean(np.abs(cross_val_score(model,data.loc[:,X_classic],data.loc[:,y],cv=tscv,scoring='neg_mean_squared_error'))))
scores.append(np.mean(np.abs(cross_val_score(model,data.loc[:,X_new],data.loc[:,y],cv=tscv,scoring='neg_mean_squared_error'))))
scores.append(np.mean(np.abs(cross_val_score(model,data.loc[:,X_all],data.loc[:,y],cv=tscv,scoring='neg_mean_squared_error'))))
```

When we plot the accuracy levels obtained by the different sets of inputs possible, we can clearly observe that the custom stochastic oscillator features we have engineered produced the lowest error levels we found possible.

```
sns.barplot(np.abs(scores),color='black')
plt.axhline(np.min(scores),linestyle=':',color='red')
plt.xticks([0,1,2],['OHLC Features','Custom Features','All Features'])
```

![](https://c.mql5.com/2/187/fig_9.jpg)

Figure 9: The custom features we have generated helped us best forecast the main buffer of the stochastic oscilator

In most dynamic processes, not all recorded variables are equally informative. Identifying which variables carry the most information can guide future feature engineering efforts, allowing us to focus on generating richer and more diverse variations of the most influential features. To quantify this, we use mutual information regression. Mutual information (MI) is a measure of statistical dependence that captures both linear and nonlinear relationships, making it well suited for assessing any form of dependency between two variables.

```
from sklearn.feature_selection import mutual_info_regression
```

Perform the statistical test. The test requires the new stochastic oscillator features we have generated and the present target we have selected.

```
scores = mutual_info_regression(data.loc[:,X_new],data.loc[:,'Target'])
```

It appears that from the 10 custom features we have generated, only 3 appear to have no meaningful relationship whatsoever with the target. This is inferred by the columns of MI scores that are almost 0. Therefore, it appears there is a meaningful relationship we have discovered algorithmically from the data we generated in our MQL5 script.

```
sns.barplot(scores,color='black')
plt.axhline(np.mean(scores),color='red',linestyle=':')
```

![](https://c.mql5.com/2/187/fig_10.jpg)

Figure 10: The MI scores we have observed suggest to us that the dataset we have engineered is meaningful and has a relationship we can learn

### Exporting To ONNX

We are now ready to export our trained statistical model to the Open Neural Network Exchange (ONNX) format. ONNX enables the deployment of machine learning models without requiring the original training framework or programming language, making them portable across platforms and environments. To begin, we load the ONNX libraries required for this process.

```
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
```

Specify the input shape of our model.

```
initial_types = [('float input',FloatTensorType([1,len(X_new)]))]
```

Fit the model on the training data.

```
model = GradientBoostingRegressor()

model.fit(train.loc[:,X_new],train.iloc[:,-1])
```

Save the model as an ONNX prototype.

```
onnx_proto = convert_sklearn(model,initial_types=initial_types,target_opset=12)
```

Save the ONNX prototype to disk, as an ONNX file.

```
onnx.save(onnx_proto,'EURUSD Stochastic GBR AI.onnx')
```

### Imeplenting The Improvements

Let us now modify that trading strategy to include our ONNX file.

```
//+------------------------------------------------------------------+
//|                                               Stochastic AI.mq5  |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#resource "\\Files\\EURUSD Stochastic GBR AI.onnx" as const uchar onnx_proto[];
```

Define global variables we will use to handle our ONNX model in our application.

```
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
vectorf  model_inputs,model_outputs;
long     model;
```

Specify system constants. These constants will specify the number of inputs and outputs our model has.

```
//+------------------------------------------------------------------+
//| System Definitions                                               |
//+------------------------------------------------------------------+
#define MODEL_INPUT_SHAPE  10
#define MODEL_OUTPUT_SHAPE 1
```

When our application is initialized, we will load our technical indicators and also keep track of the custom stochastic oscillator features we generated in the training set, such as the all time high and low readings. Then we will configure our ONNX model and ensure it is set up appropriately and working.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Setup our indicators
   atr_handler     = iATR("EURUSD",PERIOD_D1,14);
   stoch_handler   = iStochastic(Symbol(),PERIOD_CURRENT,5,3,3,MODE_EMA,STO_LOWHIGH);
   stoch_o = 22.69153;
   stoch_h = 98.551023;
   stoch_l = 1.372058;

//--- Setup the ONNX model
   model = OnnxCreateFromBuffer(onnx_proto,ONNX_DATA_TYPE_FLOAT);

//--- Define the model parameter shape
   ulong input_shape[] = {1,MODEL_INPUT_SHAPE};
   ulong output_shape[] = {1,MODEL_OUTPUT_SHAPE};

   if(!OnnxSetInputShape(model,0,input_shape))
     {
      Print("ONNX Model Error: Incorrect Input Shape ",GetLastError());
      return(INIT_FAILED);
     }

   if(!OnnxSetOutputShape(model,0,output_shape))
     {
      Print("ONNX Model Error: Incorrect Output Shape ",GetLastError());
      return(INIT_FAILED);
     }

   model_inputs = vectorf::Zeros(MODEL_INPUT_SHAPE);
   model_outputs = vectorf::Zeros(MODEL_OUTPUT_SHAPE);

   if(model != INVALID_HANDLE)
     {
      return(INIT_SUCCEEDED);
     }

//---
   return(INIT_FAILED);
  }
```

When our trading application is no longer in use, we will release the technical indicators and the ONNX model we are no longer using.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Free up memory we are no longer using when the application is off
   IndicatorRelease(atr_handler);
   IndicatorRelease(stoch_handler);
   OnnxRelease(model);
  }
```

If new price levels are received, we will perform our manual trading rules and then supplement them with the trading signals learned by our statistical model.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- When price levels change

   datetime current_time = iTime("EURUSD",PERIOD_D1,0);
   static datetime  time_stamp;

//--- Update the time
   if(current_time != time_stamp)
     {

      time_stamp = current_time;

      //--- Calculate the middle of the trading range produced over the last business cycle
      high.CopyRates(Symbol(),PERIOD_D1,COPY_RATES_HIGH,0,90);
      low.CopyRates(Symbol(),PERIOD_D1,COPY_RATES_LOW,0,90);
      double mid = ((high.Mean() + low.Mean())/2);

      //--- Calculate the current trend on the lower time frame
      open.CopyRates(Symbol(),PERIOD_H1,COPY_RATES_OPEN,0,12);
      close.CopyRates(Symbol(),PERIOD_H1,COPY_RATES_CLOSE,0,12);

      //--- Fetch indicator current readings
      CopyBuffer(atr_handler,0,0,1,atr_reading);
      CopyBuffer(stoch_handler,0,0,1,stoch_main);
      CopyBuffer(stoch_handler,1,0,1,stoch_signal);

      //--- Setting model inputs
      stoch_h = (stoch_h < stoch_main[0]) ? stoch_main[0] : stoch_h;
      stoch_l = (stoch_l > stoch_main[0]) ? stoch_main[0] : stoch_l;

      model_inputs[0] = (float)(stoch_main[0] - stoch_signal[0]);
      model_inputs[1] = (float)(((stoch_main[0] + stoch_signal[0])/2));
      model_inputs[2] = (float)((stoch_main[0] - 80));
      model_inputs[3] = (float)((stoch_main[0] - 20));
      model_inputs[4] = (float)(stoch_o);
      model_inputs[5] = (float)(stoch_h);
      model_inputs[6] = (float)(stoch_l);
      model_inputs[7] = (float)(stoch_o - stoch_main[0]);
      model_inputs[8] = (float)(stoch_h - stoch_main[0]);
      model_inputs[9] = (float)(stoch_l - stoch_main[0]);

      ask = SymbolInfoDouble("EURUSD",SYMBOL_ASK);
      bid = SymbolInfoDouble("EURUSD",SYMBOL_BID);

      //--- If we have no open positions
      if(PositionsTotal() == 0)
        {

         if(!(OnnxRun(model,ONNX_DATA_TYPE_FLOAT,model_inputs,model_outputs)))
           {
            Comment("Failed to obtain a forecast from our model: ",GetLastError());
           }

         else
           {
            Comment("Forecast: ",model_outputs);

            //--- Trading rules
            if((model_outputs[0] > stoch_main[0]) && (stoch_main[0] > 80) && (iClose(Symbol(),PERIOD_D1,0) > mid) && (open.Mean() < close.Mean()))
              {
               //--- Buy signal
               Trade.Buy(0.01,"EURUSD",ask,ask-(atr_reading[0] * 2),ask+(atr_reading[0] * 2),"");
              }

            else
               if((model_outputs[0] < stoch_main[0]) && (stoch_main[0] < 20) && (iClose(Symbol(),PERIOD_D1,0) < mid) && (open.Mean() > close.Mean()))
                 {
                  //--- Sell signal
                  Trade.Sell(0.01,"EURUSD",bid,bid+(atr_reading[0] * 2),bid-(atr_reading[0] * 2),"");
                 }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

Lastly, undefine all system definitions you make in MQL5; this is good practice for developers.

```
#undef MODEL_INPUT_SHAPE
#undef MODEL_OUTPUT_SHAPE
```

When we observe the equity curve produced by our revised strategy, it appears we have added too much noise into the system. The curve has lost its previously smooth upward trend, and now looks more volatile than we would prefer.

![](https://c.mql5.com/2/186/1000016798.png)

Figure 11: The equity curve we have obtained from our final iteration of our trading strategy appears to have noisy signals

Although the strategy remains profitable, its performance has declined significantly from the previous peak of $223. This does not imply that the stochastic oscillator lacks value as a foundation for statistical trading strategies; rather, it highlights the need for a more careful and rigorous methodology from the practitioner. Additionally, we can observe that the strategy placed no long trades and learned a bias for short trades.

For new readers, these results may appear unexpected. While developing our statistical model, we observed clear improvements in error metrics using custom stochastic features. However, returning readers will recognize this pattern.

As shown in our sister article series, Overcoming the Limitations of Machine Learning (Part 1): Lack of Interoperable Metrics, the statistical metrics used to train models often fail to reflect the real-world objectives of trading. Consequently, improvements in statistical error do not reliably translate into better trading performance.

In practice, modern statistical modeling is often a process akin to trial and error. Therefore, readers who achieve poor trading results despite careful analysis should not interpret this as a reflection of their lack of skill. A link to the related article is provided [here](https://www.mql5.com/en/articles/17906). At this point, it is sound to conclude that excessive noise may have been inadvertently introduced into the trading system, and we therefore revert to version 3 as the best-performing iteration of the application.

![](https://c.mql5.com/2/186/1000016797.png)

Figure 12: The detailed statistical analysis of our final iteration of our stochastic trading strategy

### Conclusion

This article demonstrates how a classical technical indicator can be repurposed beyond its conventional use. Readers gain insight into how familiar strategies can yield new value when viewed through a different analytical lens, and how new paradigms and trading rules can emerge through a thoughtful process of trial and error. Ultimately, every technical indicator in the MetaTrader 5 terminal holds untapped potential—the challenge lies in automatically uncovering the meaningful interpretations that remain hidden from view.

| File Name | File Description |
| --- | --- |
| Stochastic\_Strategy.mq5 | The traditional version of the stochastic oscillator strategy, it proved unprofitable during backtesting. |
| Stochastic\_Strategy\_2.mq5 | Our first iteration of our application that relied on the daily range to set clear trend directions. |
| Stochastic\_Strategy\_3.mq5 | The most profitable version of our trading application that used lower time frame analysis in addition to the previous changes. |
| Stochastic\_AI.mq5 | The second most profitable version of our trading strategy, it appeared to have too much noise in its signals. |
| Stochastic\_Strategy.ipynb | The Jupyter notebook we wrote to analyze our historical EURUSD market data and custom features. |
| Fetch\_Data\_Stochastic\_2.mq5 | The MQL5 script we wrote to fetch OHLC EURUSD data and other custom made observations on the stochastic oscillator. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20530.zip "Download all attachments in the single ZIP archive")

[Stochastic\_Strategy.mq5](https://www.mql5.com/en/articles/download/20530/Stochastic_Strategy.mq5 "Download Stochastic_Strategy.mq5")(3.04 KB)

[Stochastic\_Strategy\_2.mq5](https://www.mql5.com/en/articles/download/20530/Stochastic_Strategy_2.mq5 "Download Stochastic_Strategy_2.mq5")(3.04 KB)

[Stochastic\_Strategy\_3.mq5](https://www.mql5.com/en/articles/download/20530/Stochastic_Strategy_3.mq5 "Download Stochastic_Strategy_3.mq5")(3.39 KB)

[Stochastic\_AI.mq5](https://www.mql5.com/en/articles/download/20530/Stochastic_AI.mq5 "Download Stochastic_AI.mq5")(6.45 KB)

[Stochastic\_Strategy.ipynb](https://www.mql5.com/en/articles/download/20530/Stochastic_Strategy.ipynb "Download Stochastic_Strategy.ipynb")(1090.32 KB)

[Fetch\_Data\_Stochastic\_2.mq5](https://www.mql5.com/en/articles/download/20530/Fetch_Data_Stochastic_2.mq5 "Download Fetch_Data_Stochastic_2.mq5")(3.53 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Reimagining Classic Strategies (Part 21): Bollinger Bands And RSI Ensemble Strategy Discovery](https://www.mql5.com/en/articles/20933)
- [Overcoming The Limitation of Machine Learning (Part 9): Correlation-Based Feature Learning in Self-Supervised Finance](https://www.mql5.com/en/articles/20514)
- [Reimagining Classic Strategies (Part 19): Deep Dive Into Moving Average Crossovers](https://www.mql5.com/en/articles/20488)
- [Overcoming The Limitation of Machine Learning (Part 8): Nonparametric Strategy Selection](https://www.mql5.com/en/articles/20317)
- [Overcoming The Limitation of Machine Learning (Part 7): Automatic Strategy Selection](https://www.mql5.com/en/articles/20256)
- [Self Optimizing Expert Advisors in MQL5 (Part 17): Ensemble Intelligence](https://www.mql5.com/en/articles/20238)

**[Go to discussion](https://www.mql5.com/en/forum/502230)**

![Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://c.mql5.com/2/177/19979-tablici-v-paradigme-mvc-na-logo.png)[Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)

In the article, we will make the table column widths adjustable using the mouse cursor, sort the table by column data, and add a new class to simplify the creation of tables based on any data sets.

![From Novice to Expert: Navigating Market Irregularities](https://c.mql5.com/2/186/20645-from-novice-to-expert-navigating-logo.png)[From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)

Market rules are continuously evolving, and many once-reliable principles gradually lose their effectiveness. What worked in the past no longer works consistently over time. Today’s discussion focuses on probability ranges and how they can be used to navigate market irregularities. We will leverage MQL5 to develop an algorithm capable of trading effectively even in the choppiest market conditions. Join this discussion to find out more.

![Larry Williams Market Secrets (Part 2): Automating a Market Structure Trading System](https://c.mql5.com/2/187/20512-larry-williams-market-secrets-logo.png)[Larry Williams Market Secrets (Part 2): Automating a Market Structure Trading System](https://www.mql5.com/en/articles/20512)

Learn how to automate Larry Williams market structure concepts in MQL5 by building a complete Expert Advisor that reads swing points, generates trade signals, manages risk, and applies a dynamic trailing stop strategy.

![Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://c.mql5.com/2/186/20632-creating-custom-indicators-logo__1.png)[Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

In this article, we develop a gauge-style RSI indicator in MQL5 that visualizes Relative Strength Index values on a circular scale with a dynamic needle, color-coded ranges for overbought and oversold levels, and customizable legends. We utilize the Canvas class to draw elements like arcs, ticks, and pies, ensuring smooth updates on new RSI data.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/20530&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068734151831518465)

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