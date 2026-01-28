---
title: Self Optimizing Expert Advisors in MQL5 (Part 15): Linear System Identification
url: https://www.mql5.com/en/articles/19891
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:33:07.616438
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/19891&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062547844147094593)

MetaTrader 5 / Trading systems


Trading systems are complex applications expected to operate in chaotic, dynamic environments — a challenge for even the most experienced developers. It’s nearly impossible to define every correct action a trading application should take, as market outcomes are virtually infinite. Maintaining control and ensuring consistent profitability under such uncertainty remains one of the greatest challenges in algorithmic trading.

Simple strategies may appear reliable during calm market conditions, yet both simple and complex systems often fail when volatility increases. Despite this, the fields of control theory and signal processing appear underutilized in addressing these challenges. Control theory, devoted to maintaining stability in dynamic and uncertain systems, aligns closely with the problems our community of algorithmic traders face daily.

Classical control theory assumes a first-principles understanding of the system — explicit formulas describing the relationship between inputs and outputs. Modern financial markets, however, defy such clear mathematical structure. This has led to growing interest in integrating control theory with machine learning, which can approximate these relationships directly from data rather than relying on explicit equations.

This concept is powerful: even without knowing the precise control equations, practitioners can still learn to regulate system behavior from data. Control theory and algorithmic trading share the same goal — managing uncertainty while maintaining stability. A feedback controller does not predict prices; it regulates system responses, suppressing overreactions to noise and ensuring steady performance.

Feedback controllers also improve capital efficiency by learning when capital is being deployed effectively and reducing unnecessary trades. When combined with machine learning, these systems gain the ability to adapt autonomously, enhancing precision, control, and reliability. Despite the clear overlap, a significant research gap remains between control theory and algorithmic trading — a gap rich with potential.

In this article, we demonstrate how control theory can rejuvenate even the most basic trading systems. Using a simple moving-average strategy — buying when price breaks above the average and selling when it falls below — we explore how feedback control can restore stability and profitability to a strategy often dismissed as obsolete. While many claim such methods fail because they are “too well known,” such arguments lack empirical rigor. Our approach instead uses feedback control to identify when and why the strategy succeeds or fails.

For our experiment, we implemented the classical moving-average strategy and fixed all parameters. Using two years of historical data (January 2023 – May 2025), we optimized the moving-average period on the first half and tested performance on the second, establishing a benchmark for comparison. Once this benchmark was set, the feedback controller learned entirely from the system’s behavior during backtesting, without any parameter adjustments.

Initially, both the controlled and uncontrolled systems performed identically, as the controller was still observing. Once active, however, the feedback-controlled system produced significant improvements:

- Total loss fell from –$575 to –$333 (a 42% reduction in inefficient use of capital)
- Net profit rose from –$49 to +$57
- The number of trades dropped from 78 to 51 (a 34% increase in efficiency).
- The win rate improved from 44% to 53%
- The profit factor from 0.91 to 1.17 — a 28% gain in profitability.

These results, achieved under identical market conditions and system constraints, demonstrate the stabilizing power of feedback control. Where human intuition and traditional modeling reach their limits, control theory offers a principled path forward — revealing deeper relationships and untapped potential in strategies long considered exhausted.

### Getting Started In MQL5

To begin developing our application, we first define key system constants that remain fixed throughout all exercises. In later versions, the number of constants will grow, but we intend to carry them forward from one version to the next.

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
#define MA_SHIFT 0
#define MA_MODE MODE_EMA
#define MA_APPLIED_PRICE PRICE_CLOSE
#define SYSTEM_TIME_FRAME PERIOD_D1
#define MIN_VOLUME SymbolInfoDouble(SYMBOL,SYMBOL_VOLUME_MIN)
```

The reader should note that the purpose of this benchmark version is to establish a good initial period for our technical indicators. We therefore set a tuning parameter as an input, which we plan to optimize later using a genetic algorithm.

```
//+------------------------------------------------------------------+
//| Tuning parameters                                                |
//+------------------------------------------------------------------+
input group "Technical Indicators"
input int  MA_PERIOD = 10;//Moving average period
```

Next, we load the necessary libraries for this exercise. The Trade library is sufficient.

```
//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;
```

We also define important global variables, such as buffers for the Moving Average and Average True Range (ATR) indicators. The ATR defines our stop-loss and risk levels, which remain the same across all exercises. We also include global variables to track market prices (open, high, low, close) and handles for our technical indicators.

```
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
double ma[],atr[];
double ask,bid,open,high,low,close,padding;
int    ma_handler,atr_handler;
```

When the application loads for the first time, we initialize handlers for the indicators—one for the Moving Average and one for the ATR. The ATR measures market volatility and sets stop-loss and take-profit levels accordingly.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Initialize the indicator
   ma_handler = iMA(SYMBOL,SYSTEM_TIME_FRAME,MA_PERIOD,MA_SHIFT,MA_MODE,MA_APPLIED_PRICE);
   atr_handler = iATR(SYMBOL,SYSTEM_TIME_FRAME,14);
   return(INIT_SUCCEEDED);
  }
```

When the application closes, we deinitialize the indicators and release their resources, following best practices in MQL5.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Release the indicator
   IndicatorRelease(ma_handler);
   IndicatorRelease(atr_handler);
  }
```

Whenever new price levels are received by the terminal, we execute our trading logic. We first check if a new daily candle has formed. If so, we open a position, trading once per day.

Our trading strategy compares price levels against the Moving Average. However, during backtests in MetaTrader 5, the first daily candle forms at midnight, when price readings are often flat and unreliable. To avoid this, our algorithm references the previous day’s candle and indicator readings. In essence, the application decides today’s actions based on what occurred yesterday.

We then execute the trading logic by checking whether the closing price is above or below the Moving Average to decide whether to buy or sell.

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

      //--- If we have no open positions
      if(PositionsTotal()==0)
        {
         //--- Update indicator buffers
         CopyBuffer(ma_handler,0,1,1,ma);
         CopyBuffer(atr_handler,0,0,1,atr);
         padding = atr[0] * 2;

         //--- Fetch current market prices
         ask = SymbolInfoDouble(SYMBOL,SYMBOL_ASK);
         bid = SymbolInfoDouble(SYMBOL,SYMBOL_BID);
         close = iClose(SYMBOL,SYSTEM_TIME_FRAME,0);

         //--- Check trading signal
         if(close > ma[0])
            Trade.Buy(MIN_VOLUME,SYMBOL,ask,ask-padding,ask+padding);

         if(close < ma[0])
            Trade.Sell(MIN_VOLUME,SYMBOL,bid,ask+padding,ask-padding);
        }
     }
  }
//+------------------------------------------------------------------+
```

Once execution is complete, we undefine all previously defined system constants.

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
//+------------------------------------------------------------------+
```

When pieced together, this forms the benchmark version of the application.

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
#define MA_SHIFT 0
#define MA_MODE MODE_EMA
#define MA_APPLIED_PRICE PRICE_CLOSE
#define SYSTEM_TIME_FRAME PERIOD_D1
#define MIN_VOLUME SymbolInfoDouble(SYMBOL,SYMBOL_VOLUME_MIN)

//+------------------------------------------------------------------+
//| Tuning parameters                                                |
//+------------------------------------------------------------------+
input group "Technical Indicators"
input int  MA_PERIOD = 10;//Moving average period

//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
double ma[],atr[];
double ask,bid,open,high,low,close,padding;
int    ma_handler,atr_handler;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Initialize the indicator
   ma_handler = iMA(SYMBOL,SYSTEM_TIME_FRAME,MA_PERIOD,MA_SHIFT,MA_MODE,MA_APPLIED_PRICE);
   atr_handler = iATR(SYMBOL,SYSTEM_TIME_FRAME,14);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Release the indicator
   IndicatorRelease(ma_handler);
   IndicatorRelease(atr_handler);
  }
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

      //--- If we have no open positions
      if(PositionsTotal()==0)
        {
         //--- Update indicator buffers
         CopyBuffer(ma_handler,0,1,1,ma);
         CopyBuffer(atr_handler,0,0,1,atr);
         padding = atr[0] * 2;

         //--- Fetch current market prices
         ask = SymbolInfoDouble(SYMBOL,SYMBOL_ASK);
         bid = SymbolInfoDouble(SYMBOL,SYMBOL_BID);
         close = iClose(SYMBOL,SYSTEM_TIME_FRAME,0);

         //--- Check trading signal
         if(close > ma[0])
            Trade.Buy(MIN_VOLUME,SYMBOL,ask,ask-padding,ask+padding);

         if(close < ma[0])
            Trade.Sell(MIN_VOLUME,SYMBOL,bid,ask+padding,ask-padding);
        }
     }

  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Undefine system constants                                        |
//+------------------------------------------------------------------+
#undef SYMBOL
#undef SYSTEM_TIME_FRAME
#undef MA_APPLIED_PRICE
#undef MA_MODE
#undef MA_SHIFT
#undef MIN_VOLUME
//+------------------------------------------------------------------+
```

### Identifying Good Initial Conditions

We can now select our benchmark trading strategy and define the historical dates for backtesting. For this exercise, we use data from 2023 to 2025 and also perform forward testing. Forward testing, for readers unfamiliar with the term, involves partitioning the backtest period into segments, which may or may not be equal in size. Here, we split the dataset in half, setting the forward parameter to ½. This allows the genetic optimizer to tune parameters on the first half of the data and test them on the second half. The second half of the dataset is hidden from the model and used as a final evaluation set, only revealed to the optimizer after training.

![](https://c.mql5.com/2/175/6166192860083.png)

Figure 1: Selecting the back test days for our optimization procedure

Now that we understand the importance of forward testing, we can define the modeling conditions under which to evaluate our strategy. Due to network limits, I used the “every tick” modeling mode instead of “every tick based on real ticks,” though the latter offers more realistic results and is recommended for those with stable connections. For optimization, we used the fast genetic algorithm for efficiency; a slower, complete version can be used for more thorough searches. Input parameters were defined by their minimum, maximum, and step size to control the optimizer’s range.

![](https://c.mql5.com/2/175/2870970215737.png)

Figure 2: Choose the fast genetic based algorithm to help us identify good initial indicator periods

We began by setting the delay parameter to random delay to simulate real market latency.

![](https://c.mql5.com/2/175/292736855982.png)

Figure 3: The tuning parameters of our trading application are straightforward

Backtesting results were poor—none of the configurations were profitable.

![](https://c.mql5.com/2/175/5197019970231.png)

Figure 4: The backtest results appear unstable and need improvement

Scatter plots confirmed consistent losses across all trials.

![](https://c.mql5.com/2/175/4496551414682.png)

Figure 5: It appears that none of the configurations we tested were profitable in the first half of the optimization procedure

Surprisingly, in the forward test, the strategy turned profitable, particularly when the period was set to 42. While selecting the best-performing result risks overfitting, long evaluation periods reduce this likelihood.

![](https://c.mql5.com/2/175/5359978817030.png)

Figure 6: The forward results were more profitable than the back test results we obtained.

Many top profitable configurations were identified in the forward test, suggesting the strategy can perform well under certain conditions but may be unreliable in its current form. A reliable strategy should perform consistently across time, which ours did not.

![](https://c.mql5.com/2/175/1890839172070.png)

Figure 7: Visualizing the performance of our strategy out of sample over data periods the genetic optimizer did not observe.

### Establishing Our Benchmark

To establish a benchmark level of profitability, we first select the compiled application from our IDE, then specify the backtest dates—the same period used earlier and later applied for forward testing.

![](https://c.mql5.com/2/175/6137618234483.png)

Figure 8: Running a complete historical backtest of our trading application using the best period we have identified

In a full backtest using the 42-period setting, the strategy accrued a total loss of $559 over 78 trades, with a profit factor of 0.97, indicating long-term capital decay rather than growth.

![](https://c.mql5.com/2/175/3292878191543.png)

Figure 9: Inspecting the detailed statistical performance of our control benchmark on historical data

The equity curve from this version of our trading strategy is highly volatile, with the strategy performing poorly overall. Even in the initial backtest—intended to showcase the best results found by the genetic optimizer—the strategy barely broke even after two years of trading.

![](https://c.mql5.com/2/175/297979376678.png)

Figure 10: The equity curve produced by the control setup of our trading application appears very unstable

### Improving Our Initial Results

We are now ready to begin improving upon our benchmark profitability levels. To get started, we define additional system constants, extending those introduced earlier. These new constants determine the parameters our model requires—for instance, how many observations the feedback controller must collect before adjusting the strategy’s behavior.

```
//+------------------------------------------------------------------+
//| System constants                                                 |
//+------------------------------------------------------------------+
#define SYMBOL            Symbol()
#define MA_PERIOD         42
#define MA_SHIFT          0
#define MA_MODE           MODE_EMA
#define MA_APPLIED_PRICE  PRICE_CLOSE
#define SYSTEM_TIME_FRAME PERIOD_D1
#define MIN_VOLUME        SymbolInfoDouble(SYMBOL,SYMBOL_VOLUME_MIN)
#define OBSERVATIONS      90
#define FEATURES          7
#define MODEL_INPUTS      8
```

We also define new global variables to store forecasts from our linear system, along with its inputs, targets, and a matrix of historical observations.

```
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
double ma[],atr[];
double ask,bid,open,high,low,close,padding;
int    ma_handler,atr_handler,scenes;
bool   forecast;
matrix snapshots,b,X,y,U,S,VT,current_forecast;
vector s;
```

The Expert Advisor initialization function is slightly modified to prepare these global variables.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
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

When price levels update, our trading logic checks whether a forecast from the linear system is needed. If the system is still gathering observations, it skips forecasting and executes trades based on the previous logic. Once sufficient data is collected, forecasts are activated. Regardless of open positions, the system records periodic snapshots with each new candle, capturing the state of the model over time.

If positions are open, the model forecast method is called to obtain a linear prediction, which may later help with timing exits. For now, our goal is simply to observe whether this linear feedback system can regulate the strategy’s behavior.

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

The trading logic has been refactored into a separate CheckSignal method, which follows the same rules: if no positions are open, buy when price is above the moving average and sell when below.

```
//+------------------------------------------------------------------+
//| Check for our trading signal                                     |
//+------------------------------------------------------------------+
void check_signal(void)
  {
   if(PositionsTotal() == 0)
     {
      if(close > ma[0])
        {
         Trade.Buy(MIN_VOLUME,SYMBOL,ask,ask-padding,ask+padding);
        }

      if(close < ma[0])
        {
         Trade.Sell(MIN_VOLUME,SYMBOL,bid,ask+padding,ask-padding);
        }
     }
  }
```

Obtaining a forecast involves preparing and updating snapshots. We first copy existing snapshots, then refresh them with take\_snapshots(). The inputs (X) and target (y) of our linear system are then prepared: the first row of X is a vector of ones (the intercept), while remaining rows store system observations. The target is the account balance one step ahead of the snapshots.

We then perform Singular Value Decomposition (SVD)—an unsupervised algorithm that factorizes a matrix into a series of rank-1 components, revealing dominant correlational structures in the data. The algorithm returns one vector and two matrices, which we use to reconstruct our linear system. The vector is converted to a diagonal matrix using the Diag() method, after which we verify its rank. If nonzero, we compute the pseudoinverse solution to estimate the system’s coefficients, stored in b.

Next, we retrieve the current market inputs and multiply them by b to obtain an estimate from our linear system. If the forecasted balance exceeds the current one, the system proceeds to trade; otherwise, it waits for better market conditions. A final check, catches any cases where we may encounter errors during matrix inversion.

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

We also define a method for recording system snapshots. Each snapshot stores values of interest in a matrix (noting that in MQL5, matrices are referenced by row, then column).

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
```

When the application completes execution, all system constants are undefined.

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

When put together, this forms our Feedback Controller version of the trading strategy.

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

//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade Trade;

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
double ma[],atr[];
double ask,bid,open,high,low,close,padding;
int    ma_handler,atr_handler,scenes;
bool   forecast;
matrix snapshots,b,X,y,U,S,VT,current_forecast;
vector s;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Initialize the indicator
   ma_handler = iMA(SYMBOL,SYSTEM_TIME_FRAME,MA_PERIOD,MA_SHIFT,MA_MODE,MA_APPLIED_PRICE);
   atr_handler = iATR(SYMBOL,SYSTEM_TIME_FRAME,14);

//--- Prepare global variables
   forecast = false;
   snapshots = matrix::Zeros(FEATURES,OBSERVATIONS);
   scenes = -1;
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Release the indicator
   IndicatorRelease(ma_handler);
   IndicatorRelease(atr_handler);
  }
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

//+------------------------------------------------------------------+
//| Check for our trading signal                                     |
//+------------------------------------------------------------------+
void check_signal(void)
  {
   if(PositionsTotal() == 0)
     {
      if(close > ma[0])
        {
         Trade.Buy(MIN_VOLUME,SYMBOL,ask,ask-padding,ask+padding);
        }

      if(close < ma[0])
        {
         Trade.Sell(MIN_VOLUME,SYMBOL,bid,ask+padding,ask-padding);
        }
     }
  }

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

Running it over the same backtest window as before reveals a dramatic improvement.

![](https://c.mql5.com/2/175/3681737023442.png)

Figure 11: Establishing an improvement benchmark by using our linear feedback controller

The system shifts from consistent losses to sustained profitability. Accuracy climbs from the mid-40% range to above 50%, while the number of trades declines—indicating greater efficiency. Our Sharpe ratio and recovery factor have appreciated meaningfully. And all these improvements have been realized without us explicitly instructing the application what exactly it should do, to perform better.

![](https://c.mql5.com/2/175/5338899752443.png)

Figure 12: A detailed analysis of the improvements brought about by the linear system we have identified from the observations we recorded

The new equity curve replaces the prior instability with steady growth, and early drawdowns no longer reappear in subsequent periods. Even during losing streaks, the system never falls as deeply as it once did.

![](https://c.mql5.com/2/175/466557519658.png)

Figure 13: Visualizing the equity curve produced by the refined version of our trading application

### Conclusion

We have arrived at the end of our discussion for today. We believe that his article has demonstrated to you how feedback controllers can be augmented with machine learning to manage uncertainty, improve capital efficiency, and stabilize trading systems in volatile or shifting markets using the MQL5 API. Readers gain a clear understanding of why traditional strategies often fail and how to systematically improve them using data-driven methods, while learning practical ways to implement feedback controllers, optimize performance, and manage risk. Ultimately, the concepts presented equip you to transform unstable or underperforming strategies into controlled, profitable systems.

| File Name | File Description |
| --- | --- |
| Feedback Control Benchmark 1.mq5 | The classical version of the strategy that we aimed to outperform by observing its relationship with the market. |
| Feedback Control Benchmark 2.mq5 | The feedback controller we implemented to learn the relationship between our strategy and the current market conditions. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19891.zip "Download all attachments in the single ZIP archive")

[Feedback\_Control\_Benchmark\_1.mq5](https://www.mql5.com/en/articles/download/19891/Feedback_Control_Benchmark_1.mq5 "Download Feedback_Control_Benchmark_1.mq5")(4.18 KB)

[Feedback\_Control\_Benchmark\_2.mq5](https://www.mql5.com/en/articles/download/19891/Feedback_Control_Benchmark_2.mq5 "Download Feedback_Control_Benchmark_2.mq5")(9.54 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/497604)**

![Biological neuron for forecasting financial time series](https://c.mql5.com/2/117/Biological_neuron_for_forecasting_financial_time_series___LOGO.png)[Biological neuron for forecasting financial time series](https://www.mql5.com/en/articles/16979)

We will build a biologically correct system of neurons for time series forecasting. The introduction of a plasma-like environment into the neural network architecture creates a kind of "collective intelligence," where each neuron influences the system's operation not only through direct connections, but also through long-range electromagnetic interactions. Let's see how the neural brain modeling system will perform in the market.

![Risk Management (Part 1): Fundamentals for Building a Risk Management Class](https://c.mql5.com/2/112/Gesti7n_de_Riesgo_Parte_1_LOGO.png)[Risk Management (Part 1): Fundamentals for Building a Risk Management Class](https://www.mql5.com/en/articles/16820)

In this article, we'll cover the basics of risk management in trading and learn how to create your first functions for calculating the appropriate lot size for a trade, as well as a stop-loss. Additionally, we will go into detail about how these features work, explaining each step. Our goal is to provide a clear understanding of how to apply these concepts in automated trading. Finally, we will put everything into practice by creating a simple script with an include file.

![MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://c.mql5.com/2/175/19890-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

The Stochastic Oscillator and the Fractal Adaptive Moving Average are an indicator pairing that could be used for their ability to compliment each other within an MQL5 Expert Advisor. We introduced this pairing in the last article, and now look to wrap up by considering its 5 last signal patterns. In exploring this, as always, we use the MQL5 wizard to build and test out their potential.

![Introduction to MQL5 (Part 23): Automating Opening Range Breakout Strategy](https://c.mql5.com/2/175/19886-introduction-to-mql5-part-23-logo.png)[Introduction to MQL5 (Part 23): Automating Opening Range Breakout Strategy](https://www.mql5.com/en/articles/19886)

This article explores how to build an Opening Range Breakout (ORB) Expert Advisor in MQL5. It explains how the EA identifies breakouts from the market’s initial range and opens trades accordingly. You’ll also learn how to control the number of positions opened and set a specific cutoff time to stop trading automatically.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/19891&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062547844147094593)

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