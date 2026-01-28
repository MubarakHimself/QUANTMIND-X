---
title: Self Optimizing Expert Advisors in MQL5 (Part 16): Supervised Linear System Identification
url: https://www.mql5.com/en/articles/20023
categories: Expert Advisors, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T17:27:49.083050
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=vzgxpaurtjhnggfdoqzwqueyyzkvkagb&ssn=1769178467058398382&ssn_dr=0&ssn_sr=0&fv_date=1769178467&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20023&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Self%20Optimizing%20Expert%20Advisors%20in%20MQL5%20(Part%2016)%3A%20Supervised%20Linear%20System%20Identification%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917846789223571&fz_uniq=5068264385488549731&sv=2552)

MetaTrader 5 / Examples


In our previous discussion on feedback controllers, we learned that these systems can stabilize the performance of trading strategies by first observing their behavior in action. We have provided a quick link to the previous discussion, [here](https://www.mql5.com/en/articles/19891). This application design, allowed us to capture the dominant correlational structures that persisted across both winning and losing trades. In essence, feedback controllers helped our trading application learn how to behave optimally under current market conditions—much like human traders, who focus less on predicting the future and more on responding intelligently to the present.

The reader should note that, up to this point, our focus has been on feedback controllers that correct simple, rule-based strategies. This simple approach allowed the reader to immediately observe the impact the feedback controller made, even if it may have been the reader's first encouter with the subject matter. In Figure 1 below, we have built a schematic diagram of the application configuration to help the reader visualize the changes we are making today.

![](https://c.mql5.com/2/177/5688472147412.png)

Figure 1: Visualizing the design pattern we initially selected for our feedback controller

In this discussion, we push beyond that boundary and ask a deeper question: Can we learn to optimally control a trading strategy that is itself defined by a statistical model of the market? This represents a shift in how we apply machine learning in algorithmic trading. Instead of using models merely for prediction, we explore how statistical models can supervise or correct one another—a potentialy new class of tasks for machine learning systems.

![](https://c.mql5.com/2/177/1134758118092.png)

Figure 2: We will substitute the fixed trading strategy with a statistical model estimated from the market data

Our objective is to determine whether starting with a more sophisticated, data-driven trading strategy provides richer structure for the feedback controller to learn from, and ultimately, better results. To investigate this, we revisited our earlier work on feedback control and linear system identification, where we built a simple moving-average strategy and fitted a feedback controller to establish a baseline. We then replaced the moving-average component with a supervised statistical model of the EUR/USD market and evaluated performance under identical testing conditions. The findings were:

1. Net profit rose from $56 in the baseline system to $170—a nearly 200% improvement.

2. Gross loss fell from $333 to $143, a 57% reduction in downside exposure.

3. Accuracy improved from 52.9% to 72%, a 37% increase in precision.

4. The number of trades fell from 51 to 33, a 35% gain in efficiency, showing that the system filtered out unnecessary trades.
5. The profit factor improved from 1.17 to 2.18, an 86% appreciation in profitability per unit of risk.

Together, these results demonstrate that coupling a feedback controller with a well suited statistical model can lead to material improvements in both efficiency and stability. The synergy between closed-loop control and supervised learning enables a form of intelligent adaptation. Such a system may bring to mind reinforcement learning algorithms, but taken from a supervised perspective.

When all is done, this article, will outline the design choices that shaped this improved system and provide readers with a structured approach for enhancing the performance of their own MetaTrader 5 applications using feedback control principles to support their statistical models.

### Getting Started With Our Analysis in Python

The first step in our Python-based analysis of MetaTrader 5 market data is to import the necessary libraries.

```
#Import the standard python libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
```

Once our dependencies are loaded, we proceed to initialize the MetaTrader 5 terminal.

```
#Check if we have started the terminal
if(mt5.initialize()):
    print("Failed To Startup")

else:
    print("Logged In")
```

Logged In

At this stage, we select the market symbol we’re interested in analyzing.

```
if(mt5.symbol_select("EURUSD")):
    print("Found EURUSD Market")

else:
    print("Failed To Find EURUSD Market")
```

Found EURUSD Market

If you’ve followed along this far, you’re now ready to fetch historical market data directly from your MetaTrader 5 terminal. Be sure to convert timestamps from seconds to a human-readable format, as MetaTrader 5 returns time data in seconds by default

```
#Read in the market data
data = pd.DataFrame(mt5.copy_rates_from_pos("EURUSD",mt5.TIMEFRAME_D1,0,4000))
data['time'] = pd.to_datetime(data['time'],unit='s')
data
```

![](https://c.mql5.com/2/177/4439581499351.png)

Figure 3: The market data we fetched from the MetaTrader 5 terminal

The terminal provides a detailed dataset containing multiple market attributes. However, for this discussion, we will focus only on the four key price levels — Open, High, Low, and Close. Therefore, we drop all other columns from the dataset.

```
#Focus on the major price levels
data = data.iloc[:,:5]
data
```

![](https://c.mql5.com/2/177/335018043886.png)

Figure 4: We will focus on the four fundamanetal price levels for this exercise

Next, we remove any observations that overlap with the backtest period we plan to use. In our previous discussion on linear system identification, we ran a backtest from January 1, 2023 up to October 2025 (the current period at the time of writing). To maintain consistency, we’ll preserve the same backtest window here. It’s good practice to eliminate any data that might leak information from the test period into the training set

```
#Drop off the test period
data = data.iloc[:-(370*2),:]
data
```

![](https://c.mql5.com/2/177/2251758486113.png)

Figure 5: It is good practice to drop off all observations that overlap with the back test we intend to perform

With our dataset cleaned, we now define the forecast horizon — how far into the future our model will attempt to predict — and label the dataset accordingly with the target values.

```
#Define the new horizon
HORIZON = 10
```

Finally, we drop all missing rows to ensure data integrity. Once the dataset is complete and properly formatted, we’re ready to load the machine learning libraries and begin model training.

```
#Label the data
data['Target 1'] = data['close'].shift(-HORIZON)
data['Target 2'] = data['high'].shift(-HORIZON)
data['Target 3'] = data['low'].shift(-HORIZON)
```

Then we drop any rows that have missing data.

```
#Drop missing rows
data.dropna(inplace=True)
```

Let us now get ready to start fitting our machine learning models. Since we are not aware which model will work best, we import a variety of models to get started.

```
#Import cross validation tools
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.svm import LinearSVR
```

Creates new instances of each model.

```
models = [LinearRegression(),\
          Ridge(alpha=10e-3),\
          RandomForestRegressor(random_state=0),\
          GradientBoostingRegressor(random_state=0),\
          KNeighborsRegressor(n_jobs=-1,n_neighbors=5),\
          RadiusNeighborsRegressor(n_jobs=-1),\
          LinearSVR(random_state=0),\
          MLPRegressor(random_state=0,hidden_layer_sizes=(4,10,40,10),solver='lbfgs')]
```

Partition or data into equal halves. One for training and the latter testing.

```
#The big picture of what we want to test
train , test = data.iloc[:data.shape[0]//2,:] , data.iloc[data.shape[0]//2:,:]
```

Now that our dataset is ready, we can define the inputs and targets for our machine learning model.

```
#Define inputs and target
X = data.columns[1:-3]
y = data.columns[-3:]
```

We begin by creating a dedicated function that returns a fresh model instance each time it’s called.

```
#Fetch a new copy of the model
def get_model():
    return(LinearRegression())
```

As we learned in our earlier discussion, not all historical data is always useful for present forecasting. Readers who have not read our prior discussion on market memory, can find a helpful link provided, [here](https://www.mql5.com/en/articles/20010). To determine how much history we actually need, we once again perform cross-validation, this time testing how effectively the first half of our training data can predict the second half. Our findings reveal that only about 60% of the first half is required to accurately predict the remaining half. This means we can safely reduce our training set to focus only on the most coherent partition — the portion of data that appears internally consistent.

```
#Store our performance
error = []

#Define the total number of iterations we wish to perform
ITERATIONS = 10

#Let us perform the line search
for i in np.arange(ITERATIONS):
    #Training fraction
    fraction =((i+1)/10)

    #Partition the data to select the most recent information
    partition_index = train.shape[0] - int(train.shape[0]*fraction)

    train_X_partition = train.loc[partition_index:,X]
    train_y_partition = train.loc[partition_index:,y[0]]

    #Fit a model
    model = get_model()

    #Fit the model
    model.fit(train_X_partition,train_y_partition)

    #Cross validate the model out of sample
    score = root_mean_squared_error(test.loc[:,y[0]],model.predict(test.loc[:,X]))

    #Append the error levels
    error.append(score)

#Plot the results
plt.title('Improvements Made By Historical Data')
plt.plot(error,color='black')
plt.grid()
plt.ylabel('Out of Sample RMSE')
plt.xlabel('Progressivley Fitting On All Historical Data')
plt.scatter(np.argmin(error),np.min(error),color='red')
```

![](https://c.mql5.com/2/177/148174621712.png)

Figure 6: As we learned in our previous discussion on effective memory cross validation, not all the historical data on hand is helpful

Identify the coherent index.

```
#Let us select the partition of interest
partition_index = train.shape[0] - int(train.shape[0]*(0.6))
```

Reshape the training data and remove older, less relevant observations.

```
train = train.loc[partition_index:,:]
train.reset_index(inplace=True,drop=True)
train
```

![](https://c.mql5.com/2/177/5447212948520.png)

Figure 7: We have trimmed our data set down, to only keep observations we believe are best aligned with the present

Earlier in our design process, we defined a list of candidate model types. We will now iterate through each one and assess their performance on the test set. Note that while we evaluate the models on the test data, we never fit them on this test set, since it is reserved for our final backtest.

```
#Store each model's error levels
error = []

#Fit each model
for m in models:
    m.fit(train.loc[:,X],train.loc[:,y[0]])
    #Store our error levels
    error.append(root_mean_squared_error(test.loc[:,y[0]],m.predict(test.loc[:,X])))
```

Next, we visualize each model’s performance using a bar plot. As shown, the Ridge Regression model performs best, though the Deep Neural Network (DNN) closely follows. This suggests the DNN may benefit from parameter tuning.

```
sns.barplot(error,color='black')
plt.axhline(np.min(error),color='red',linestyle=':')
plt.scatter(np.argmin(error),np.min(error),color='red')
plt.ylabel('Out of Sample RMSE')
plt.title('Model Selection For EURUSD Market')
plt.xticks([0,1,2,3,4,5,6,7],['OLS','Ridge','RF','GBR','KNR','RNR','LSVR','DNN'])
```

![](https://c.mql5.com/2/177/2626519537794.png)

Figure 8: We have identified a good benchmark model to outperform

To find optimal parameters for the DNN, we employ time-series cross-validation using scikit-learn.

```
from sklearn.model_selection import RandomizedSearchCV,TimeSeriesSplit
```

We define the number of splits and the temporal gap between each fold, then specify a parameter grid covering all values to explore. We then define a base neural network configuration with fixed parameters to ensure reproducibility. For example, we disable shuffle=True (since time-series data must preserve order) and fix the random state to 0 so that weight initialization remains consistent across runs. We also disable early stopping and set the maximum iterations to 1000.

```
#Define the time series cross validation tool
tscv = TimeSeriesSplit(n_splits=5,gap=HORIZON)

#Define the parameter values we want to search over
dist = dict(
    loss=['squared_error','poisson'],
    activation = ['identity','relu','tanh','logistic'],
    solver=['adam','lbfgs','sgd'],
    learning_rate=['constant','invscaling','adaptive'],
    learning_rate_init=[1,0,10e-1,10e-2,10e-3],
    hidden_layer_sizes=[(4,10,4),(4,4,4,4),(4,1,8,2),(4,2,6,3),(4,2,1,4),(4,2,8,16,2)],
    alpha=[1,0,10e-1,10e-2,10e-3]
)

#Define basic model parameters we want to keep fixed
model = MLPRegressor(shuffle=False,random_state=0,early_stopping=False,max_iter=1000)

#Define the randomized search object
rscv = RandomizedSearchCV(model,cv=tscv,param_distributions=dist,random_state=0,n_iter=50)

#Perform the search
rscv.fit(train.loc[:,X],train.loc[:,y[0]])

#Retreive the best parameters we found
rscv.best_params_
```

{'solver': 'lbfgs',

'loss': 'squared\_error',

'learning\_rate\_init': 0.1,

'learning\_rate': 'adaptive',

'hidden\_layer\_sizes': (4, 2, 1, 4),

'alpha': 0.01,

'activation': 'identity'}

After running the grid search, the model returns the best-performing parameter combination, which we compare against earlier results. Interestingly, our optimized DNN — visible on the far right of the performance chart — still does not outperform the Ridge Regression benchmark.

```
sns.barplot(error,color='black')
plt.scatter(x=np.argmin(error),y=np.min(error),color='red')
plt.axhline(np.min(error),color='red',linestyle=':')
plt.xticks([0,1,2,3,4,5,6,7,8],['OLS','Ridge','RF','GBR','KNR','RNR','LSVR','DNN','ODNN'])
plt.ylabel('Out of Sample RMSE')
plt.title('Final Model Selection For EURUSD 2023-2025 Backtest')
```

![](https://c.mql5.com/2/177/1522797499171.png)

Figure 9: We have successfully outperformed the control level we identified earlier

### Exporting To ONNX

With the optimization complete, we now export the final model to the Open Neural Network Exchange (ONNX) format. ONNX provides a framework-independent interface that allows trained models to be shared and deployed across multiple programming environments without carrying over their original training dependencies.

```
#Fit the baseline model
model = rscv.best_estimator_
```

To begin the export, we define the model, import the necessary ONNX libraries.

```
#Prepare to export to ONNX
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
```

Specify the input shape (1x4, corresponding to the four major price levels) and output shape (1x1, representing the predicted value).

```
#Define ONNX model input and output dimensions
initial_types = [("FLOAT_INPUT",FloatTensorType([1,4]))]
final_types = [("FLOAT_OUTPUT",FloatTensorType([1,1]))]
```

We then generate the ONNX prototype, an intermediary representation of the model.

```
#Convert the model to its ONNX prototype
onnx_proto = convert_sklearn(model,initial_types=initial_types,final_types=final_types,target_opset=12)
```

Finally save it to disk as an ONNX buffer file which we will subsequently import into our MetaTrader 5 application.

```
#Save the ONNX model
onnx.save(onnx_proto,"EURUSD Improved Baseline LR.onnx")
```

### Building Our MQL5 Application

With our ONNX model defined and ready, we now begin constructing the MetaTrader 5 application. The first step is to define the system constants — the fixed parameters that guide our strategy throughout the application. These include the periods of the moving averages, the number of observations required before the feedback controller becomes active, and the number of input and output variables for the ONNX model.

```
//+------------------------------------------------------------------+
//|                                  Feedback Control Benchmark .mq5 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| System constants                                                 |
//+------------------------------------------------------------------+
#define SYMBOL Symbol()
#define MA_PERIOD 42
#define MA_SHIFT 0
#define MA_MODE MODE_EMA
#define MA_APPLIED_PRICE PRICE_CLOSE
#define SYSTEM_TIME_FRAME PERIOD_D1
#define MIN_VOLUME SymbolInfoDouble(SYMBOL,SYMBOL_VOLUME_MIN)
#define OBSERVATIONS 90
#define FEATURES     7
#define MODEL_INPUTS 8
#define TOTAL_MODEL_INPUTS 4
#define TOTAL_MODEL_OUTPUTS 1
```

Once these constants are defined, we load the ONNX model created earlier.

```
//+------------------------------------------------------------------+
//| System resources we need                                         |
//+------------------------------------------------------------------+
#resource "\\Files\\EURUSD Improved Baseline LR.onnx" as const uchar onnx_buffer[];
```

Our application also imports several supporting libraries to simplify common trading operations, such as opening, closing, and modifying positions.

```
//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;
```

Next, we define global variables to maintain shared state across functions — ensuring the same key values are accessible wherever needed.

```
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
int          ma_handler,atr_handler,scenes;
bool         forecast;
long         onnx_model;
double       ma[],atr[];
double       ask,bid,open,high,low,close,padding;
matrix       snapshots,b,X,y,U,S,VT,current_forecast;
vector       s;
vectorf      onnx_inputs,onnx_output;
```

During initialization, we instantiate the ONNX model from the exported buffer and perform an integrity check to confirm that it hasn’t been corrupted. If successful, we define the model’s input and output shapes, which must match those defined in Python. We then load our technical indicators and initialize the global variables with default values.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Create the ONNX model from its buffer
   onnx_model = OnnxCreateFromBuffer(onnx_buffer,ONNX_DATA_TYPE_FLOAT);

//--- Check for errors
   if(onnx_model == INVALID_HANDLE)
     {
      //--- User feedback
      Print("An error occured loading the ONNX model:\n",GetLastError());
      //--- Abort
      return(INIT_FAILED);
     }

//--- Setup the ONNX handler input shape
   else
     {
      //--- Define the I/O shapes
      ulong input_shape[] = {1,4};
      ulong output_shape[] = {1,1};

      //--- Attempt to set input shape
      if(!OnnxSetInputShape(onnx_model,0,input_shape))
        {
         //--- User feedback
         Print("Failed to specify the correct ONNX model input shape:\n",GetLastError());
         //--- Abort
         return(INIT_FAILED);
        }

      //--- Attempt to set output shape
      if(!OnnxSetOutputShape(onnx_model,0,output_shape))
        {
         //--- User feedback
         Print("Failed to specify the correct ONNX model output shape:\n",GetLastError());
         //--- Abort
         return(INIT_FAILED);
        }
     }

//--- Initialize the indicator
   ma_handler = iMA(SYMBOL,SYSTEM_TIME_FRAME,MA_PERIOD,MA_SHIFT,MA_MODE,MA_APPLIED_PRICE);
   atr_handler = iATR(SYMBOL,SYSTEM_TIME_FRAME,14);

//--- Prepare global variables
   forecast = false;
   snapshots = matrix::Zeros(FEATURES,OBSERVATIONS);
   scenes = -1;
   return(INIT_SUCCEEDED);
  }
```

When the program ends, all allocated resources are freed to ensure efficient memory usage.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Release the ONNX model
   OnnxRelease(onnx_model);
//--- Release the indicator
   IndicatorRelease(ma_handler);
   IndicatorRelease(atr_handler);
  }
```

Whenever new price data arrives, the system checks for the formation of a new candle. If a new candle has formed, it updates both the candle count and the total number of “scenes” (episodes) observed by the feedback controller. Once the controller has gathered the required number of observations, it is activated — from that point, its predictions are consulted before any new trades are placed.

If no open positions exist, the application updates its indicators and requests a forecast — either from the feedback controller (if active) or from the ONNX model. The model receives the four major price levels as input and outputs a forecast value. The system then takes a snapshot of key variables such as the price levels, account balance, equity, and indicator readings.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Check if a new candle has formed
   datetime current_time = iTime(Symbol(),SYSTEM_TIME_FRAME,0);
   static datetime time_stamp;

   if(current_time != time_stamp)
     {
      //--- Update the time
      time_stamp = current_time;
      scenes = scenes+1;

      //--- Check how many scenes have elapsed
      if(scenes == (OBSERVATIONS-1))
        {
         forecast   = true;
        }

      //--- If we have no open positions
      if(PositionsTotal()==0)
        {
         //--- Update indicator buffers
         CopyBuffer(ma_handler,0,1,1,ma);
         CopyBuffer(atr_handler,0,0,1,atr);
         padding = atr[0] * 2;

         //--- Prepare a prediction from our model
         onnx_inputs = vectorf::Zeros(TOTAL_MODEL_INPUTS);
         onnx_inputs[0] = (float) iOpen(Symbol(),SYSTEM_TIME_FRAME,0);
         onnx_inputs[1] = (float) iHigh(Symbol(),SYSTEM_TIME_FRAME,0);
         onnx_inputs[2] = (float) iLow(Symbol(),SYSTEM_TIME_FRAME,0);
         onnx_inputs[3] = (float) iClose(Symbol(),SYSTEM_TIME_FRAME,0);

         //--- Also prepare the outputs
         onnx_output = vectorf::Zeros(TOTAL_MODEL_OUTPUTS);

         //--- Fetch current market prices
         ask = SymbolInfoDouble(SYMBOL,SYMBOL_ASK);
         bid = SymbolInfoDouble(SYMBOL,SYMBOL_BID);
         close = iClose(SYMBOL,SYSTEM_TIME_FRAME,1);

         //--- Do we need to forecast?
         if(!forecast)
           {
            //--- Check trading signal
            check_signal();
           }

         //--- We need a forecast
         else
            if(forecast)
              {
               model_forecast();
              }
        }

      //--- Take a snapshot
      if(!forecast)
         take_snapshot();

      //--- Otherwise, we have positions open
      else
        {
         //--- Let the model decide if we should close or hold our position
         if(forecast)
            model_forecast();

         //--- Otherwise record all observations on the performance of the application
         else
            if(!forecast)
               take_snapshot();
        }
     }
  }
//+------------------------------------------------------------------+
```

Trading signals are generated only when no positions are open. If the ONNX model expects prices to rise, a buy signal is registered — but only if the price is already above its moving average. Conversely, a sell signal is registered only if the closing price is below the moving average and the model expects depreciation.

```
//+------------------------------------------------------------------+
//| Check for our trading signal                                     |
//+------------------------------------------------------------------+
void check_signal(void)
  {
   if(PositionsTotal() == 0)
     {
      //--- Fetch a prediction from our model
      if(OnnxRun(onnx_model,ONNX_DATA_TYPE_FLOAT,onnx_inputs,onnx_output))
        {
         if((close > ma[0]) && (onnx_output[0] > iClose(Symbol(),SYSTEM_TIME_FRAME,0)))
           {
            Trade.Buy(MIN_VOLUME,SYMBOL,ask,ask-padding,ask+padding);
           }

         if((close < ma[0]) && (onnx_output[0] < iClose(Symbol(),SYSTEM_TIME_FRAME,0)))
           {
            Trade.Sell(MIN_VOLUME,SYMBOL,bid,ask+padding,ask-padding);
           }
        }
     }
  }
```

The feedback controller’s prediction method begins by copying all previously recorded observations and appending the current one. It then constructs two shifted partitions: one representing the present inputs and another representing the next-step targets (future observations). The target variable in this setup is the future account balance.

Using Singular Value Decomposition (SVD), the controller factorizes the observation matrix into three unitary matrices. Since two of these are orthogonal, their inverses can be obtained simply by taking their transposes — leaving only the diagonal S matrix to invert. This approach greatly reduces computational load.

Once the optimal coefficients are derived, the controller multiplies them by the current input vector to obtain a predicted future balance. If the predicted balance exceeds the current one, trading permission is granted; otherwise, it is withheld. In rare cases where coefficient estimation fails — typically due to a singular diagonal matrix (S containing zeros) — the controller aborts the prediction process.

```
//+------------------------------------------------------------------+
//| Obtain a forecast from our model                                 |
//+------------------------------------------------------------------+
void model_forecast(void)
  {

   Print(scenes);
   Print(snapshots);

//--- Create a copy of the current snapshots
   matrix temp;
   temp.Copy(snapshots);
   snapshots = matrix::Zeros(FEATURES,scenes+1);

   for(int i=0;i<FEATURES;i++)
     {
      snapshots.Row(temp.Row(i),i);
     }

//--- Attach the latest readings to the end
   take_snapshot();

//--- Obtain a forecast for our trading signal
//--- Define the model inputs and outputs

//--- Implement the inputs and outputs
   X = matrix::Zeros(FEATURES+1,scenes);
   y = matrix::Zeros(1,scenes);

//--- The first row is the intercept.
   X.Row(vector::Ones(scenes),0);

//--- Filling in the remaining rows
   for(int i =0; i<scenes;i++)
     {
      //--- Filling in the inputs
      X[1,i] = snapshots[0,i]; //Open
      X[2,i] = snapshots[1,i]; //High
      X[3,i] = snapshots[2,i]; //Low
      X[4,i] = snapshots[3,i]; //Close
      X[5,i] = snapshots[4,i]; //Moving average
      X[6,i] = snapshots[5,i]; //Account equity
      X[7,i] = snapshots[6,i]; //Account balance

      //--- Filling in the target
      y[0,i] = snapshots[6,i+1];//Future account balance
     }

   Print("Finished implementing the inputs and target: ");
   Print("Snapshots:\n",snapshots);
   Print("X:\n",X);
   Print("y:\n",y);

//--- Singular value decomposition
   X.SingularValueDecompositionDC(SVDZ_S,s,U,VT);

//--- Transform s to S, that is the vector to a diagonal matrix
   S = matrix::Zeros(s.Size(),s.Size());
   S.Diag(s,0);

//--- Done
   Print("U");
   Print(U);
   Print("S");
   Print(s);
   Print(S);
   Print("VT");
   Print(VT);

//--- Learn the system's coefficients

//--- Check if S is invertible
   if(S.Rank() != 0)
     {
      //--- Invert S
      matrix S_Inv = S.Inv();
      Print("S Inverse: ",S_Inv);

      //--- Obtain psuedo inverse solution
      b = VT.Transpose().MatMul(S_Inv);
      b = b.MatMul(U.Transpose());
      b = y.MatMul(b);

      //--- Prepare the current inputs
      matrix inputs = matrix::Ones(MODEL_INPUTS,1);
      for(int i=1;i<MODEL_INPUTS;i++)
        {
         inputs[i,0] = snapshots[i-1,scenes];
        }

      //--- Done
      Print("Coefficients:\n",b);
      Print("Inputs:\n",inputs);
      current_forecast = b.MatMul(inputs);
      Print("Forecast:\n",current_forecast[0,0]);

      //--- The next trade may be expected to be profitable
      if(current_forecast[0,0] > AccountInfoDouble(ACCOUNT_BALANCE))
        {
         //--- Feedback
         Print("Next trade expected to be profitable. Checking for trading singals.");
         //--- Check for our trading signal
         check_signal();
        }

      //--- Next trade may be expected to be unprofitable
      else
        {
         Print("Next trade expected to be unprofitable. Waiting for better market conditions");
        }
     }

//--- S is not invertible!
   else
     {
      //--- Error
      Print("[Critical Error] Singular values are not invertible.");
     }
  }
```

The system continuously records snapshots of its state throughout the process of its trading sessions. This method ofrecordings are how we build applications that can learn from their experience of the market.

```
//+------------------------------------------------------------------+
//| Take a snapshot of the market                                    |
//+------------------------------------------------------------------+
void take_snapshot(void)
  {
//--- Record system state
   snapshots[0,scenes]=iOpen(SYMBOL,SYSTEM_TIME_FRAME,1); //Open
   snapshots[1,scenes]=iHigh(SYMBOL,SYSTEM_TIME_FRAME,1); //High
   snapshots[2,scenes]=iLow(SYMBOL,SYSTEM_TIME_FRAME,1);  //Low
   snapshots[3,scenes]=iClose(SYMBOL,SYSTEM_TIME_FRAME,1);//Close
   snapshots[4,scenes]=ma[0];                             //Moving average
   snapshots[5,scenes]=AccountInfoDouble(ACCOUNT_EQUITY); //Equity
   snapshots[6,scenes]=AccountInfoDouble(ACCOUNT_BALANCE);//Balance

   Print("Scene: ",scenes);
   Print(snapshots);
  }
//+------------------------------------------------------------------+
```

When shutting down, it clears all previously defined constants and variables.

```
//+------------------------------------------------------------------+
//| Undefine system constants                                        |
//+------------------------------------------------------------------+
#undef SYMBOL
#undef SYSTEM_TIME_FRAME
#undef MA_APPLIED_PRICE
#undef MA_MODE
#undef MA_SHIFT
#undef MIN_VOLUME
#undef MODEL_INPUTS
#undef FEATURES
#undef OBSERVATIONS
//+------------------------------------------------------------------+
```

We then select our application and test dates.

![](https://c.mql5.com/2/176/389370948801.png)

Figure 10: Selecting our Expert Advisor for its 2 year back test

Ensuring that the evaluation period poses realistic trading challenges is key to emulating real market. This includes enabling random delays in MetaTrader 5’s Strategy Tester to simulate live trading uncertainty.

![](https://c.mql5.com/2/176/2112566811279.png)

Figure 11: Select back test conditions that emulate real market conditions

The performance metrics of our enhanced application speak for themselves. The total net profit increased by more than twofold, and trading accuracy rose to 72%, approaching the 80% range. Key performance ratios — including expected payoff, Sharpe ratio, and recovery factor — all improved over the baseline model.

![](https://c.mql5.com/2/176/2717365909553.png)

Figure 12: Detailed statistics on the performance of our trading application over the 2 year test period

The equity curve of this revised system exhibits a smoother, more consistent upward trend across the same backtest period. Because this backtest window was not part of the training data, we can be confident that the improvements are genuine and not the result of information leakage.

![](https://c.mql5.com/2/176/4686922110619.png)

Figure 13: The equity curve produced by our trading application has a strong upward which is what we desire to observe

Finally, one particularly impressive outcome was the feedback controller’s predictive precision. At the end of the backtest, it forecasted a final balance of $270.28, while the actual result was within 10 cents of that estimate. As we’ve discussed in previous articles, this minor discrepancy is likely due to the inherent difference between the mathematical manifold of model predictions and that of real-world outcomes — meaning perfect alignment is theoretically impossible. Nevertheless, the proximity of this result confirms that our feedback control framework delivers meaningful forecasts.

![](https://c.mql5.com/2/176/4086792578857.png)

Figure 14: The feedback controller also appears to have reasonable expectations of how the strategy affects the balance of the account

### Conclusion

After reading this article, the reader learns a new framework for building self adapting trading applications, that control their behaviour depending on the outcomes of their actions. Linear feedback control algorithms allow us to effeciently identify unwanted behaviour even in a complex non-linear system. Their utility for algorithmic trading cannot be exhausted. These algorithms appear well suited to enhance our classical models of the market. Additionally, this article has taught the reader how to build an ensemble of intelligent systems that coorperate to build trading applications that seek to learn good behaviour in the market. It appears that time series forecasting on its own, is only a component of a bigger solution.

| File Name | File Description |
| --- | --- |
| Feedback\_Control\_Benchmark\_3.mq5 | The MetaTrader 5 trading application we built to rely on a combination of supervised learning and system identification. |
| Supervised Linear System Identification.ipynb | The Jupyter notebook we wrote to analyze the EURUSD market data we fetched from the terminal using the Python integration library. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20023.zip "Download all attachments in the single ZIP archive")

[Feedback\_Control\_Benchmark\_3.mq5](https://www.mql5.com/en/articles/download/20023/Feedback_Control_Benchmark_3.mq5 "Download Feedback_Control_Benchmark_3.mq5")(11.89 KB)

[Supervised\_Linear\_System\_Identification.ipynb](https://www.mql5.com/en/articles/download/20023/Supervised_Linear_System_Identification.ipynb "Download Supervised_Linear_System_Identification.ipynb")(187.93 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/499135)**

![Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (Final Part)](https://c.mql5.com/2/111/Neural_Networks_in_Trading____FinCon____LOGO2__1.png)[Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (Final Part)](https://www.mql5.com/en/articles/16937)

We continue to implement the approaches proposed by the authors of the FinCon framework. FinCon is a multi-agent system based on Large Language Models (LLMs). Today, we will implement the necessary modules and conduct comprehensive testing of the model on real historical data.

![Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (FinCon)](https://c.mql5.com/2/110/Neural_Networks_in_Trading____FinCon____LOGO2.png)[Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (FinCon)](https://www.mql5.com/en/articles/16916)

We invite you to explore the FinCon framework, which is a a Large Language Model (LLM)-based multi-agent system. The framework uses conceptual verbal reinforcement to improve decision making and risk management, enabling effective performance on a variety of financial tasks.

![Price Action Analysis Toolkit Development (Part 48): Multi-Timeframe Harmony Index with Weighted Bias Dashboard](https://c.mql5.com/2/178/20097-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 48): Multi-Timeframe Harmony Index with Weighted Bias Dashboard](https://www.mql5.com/en/articles/20097)

This article introduces the “Multi-Timeframe Harmony Index”—an advanced Expert Advisor for MetaTrader 5 that calculates a weighted bias from multiple timeframes, smooths the readings using EMA, and displays the results in a clean chart panel dashboard. It includes customizable alerts and automatic buy/sell signal plotting when strong bias thresholds are crossed. Suitable for traders who use multi-timeframe analysis to align entries with overall market structure.

![From Novice to Expert: Revealing the Candlestick Shadows (Wicks)](https://c.mql5.com/2/178/19919-from-novice-to-expert-revealing-logo.png)[From Novice to Expert: Revealing the Candlestick Shadows (Wicks)](https://www.mql5.com/en/articles/19919)

In this discussion, we take a step forward to uncover the underlying price action hidden within candlestick wicks. By integrating a wick visualization feature into the Market Periods Synchronizer, we enhance the tool with greater analytical depth and interactivity. This upgraded system allows traders to visualize higher-timeframe price rejections directly on lower-timeframe charts, revealing detailed structures that were once concealed within the shadows.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/20023&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068264385488549731)

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