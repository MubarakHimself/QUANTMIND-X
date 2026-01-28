---
title: Integrating AI model into already existing MQL5 trading strategy
url: https://www.mql5.com/en/articles/16973
categories: Trading, Integration, Expert Advisors, Machine Learning
relevance_score: 6
scraped_at: 2026-01-22T17:57:07.986225
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/16973&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049508636968987841)

MetaTrader 5 / Trading


### Introduction

In this article, we are going to integrate AI model into already existing MQL5 trading strategy, we will use Order Block coupled with fibonacci from the previous [article](https://www.mql5.com/en/articles/13396). Many existing MQL5 trading strategies rely on fixed indicators, rigid thresholds, or predefined patterns that may not be effective across different market cycles. These strategies lack the ability to learn from past data, recognize complex patterns, or adjust their decisions dynamically based on changing conditions.

Incorporating an AI model into an MQL5 trading strategy can help overcome existing challenges by infusing machine learning-based adaptability and decision-making capabilities. By using techniques like Long Short Term Memory (LSTM) or predictive analytics, AI can analyze extensive historical and real-time datasets to generate smarter trading actions. Unlike rigid, predefined strategies, AI-enhanced systems dynamically adapt and refine their approaches by learning from evolving market conditions. This leads to sharper timing for trades, more effective risk mitigation, and increased profitability over time.

### Getting started

To get started, the first step is to convert the existing MQL5 code into Python. Since we are integrating artificial intelligence into the trading strategy, having a Python-based version of the original MQL5 logic is essential. This conversion ensures that we can seamlessly incorporate AI-driven enhancements without disrupting the core functionality of the strategy. The Python version of the code should replicate the behavior of the MQL5 script precisely, including trade execution logic, indicator calculations, order management, and any risk management rules. This guarantees that the AI model will interact with a system that behaves identically to the one running on MetaTrader 5, allowing for accurate testing and optimization before full integration. Once this step is completed, we can proceed with embedding machine learning models, training the AI on market data, and ultimately creating an intelligent, adaptive trading system that enhances decision-making and performance.

MQL5 code:

```
#include <Trade/Trade.mqh>
#include <Arrays\ArrayObj.mqh>
CTrade trade;

#define BullOB clrLime
#define BearOB clrRed

//+------------------------------------------------------------------+
//|                           Global vars                            |
//+------------------------------------------------------------------+
double Lots = 0.01;
int takeProfit = 170;
int length = 100;
input double stopLoss = 350;
input double Mgtn = 0.85;

bool isBullishOB = false;
bool isBearishOB = false;

input int Time1Hstrt = 3;
input int Time1Hend = 4;

class COrderBlock : public CObject {
public:
   int direction;
   datetime time;
   double high;
   double low;
   bool traded;

   string rectName;
   string tradeRectName;

   COrderBlock(int dir, datetime t, double h, double l) {
      direction = dir;
      time = t;
      high = h;
      low = l;
      traded = false;
      rectName = "";
      tradeRectName = "";

   }

   void draw(datetime tmS, datetime tmE, color clr) {
      rectName = "OB REC" + TimeToString(time);
      ObjectCreate(0, rectName, OBJ_RECTANGLE, 0, time, low, tmS, high);
      ObjectSetInteger(0, rectName, OBJPROP_FILL, true);
      ObjectSetInteger(0, rectName, OBJPROP_COLOR, clr);

      tradeRectName = "OB trade" + TimeToString(time);
      ObjectCreate(0, tradeRectName, OBJ_RECTANGLE, 0, tmS, high, tmE, low);
      ObjectSetInteger(0, tradeRectName, OBJPROP_FILL, true);
      ObjectSetInteger(0, tradeRectName, OBJPROP_COLOR, clr);
   }

   void removeDrawings() {
      if (ObjectFind(0, rectName) != -1) {
         ObjectDelete(0, rectName); // Delete the main rectangle
      }
      if (ObjectFind(0, tradeRectName) != -1) {
         ObjectDelete(0, tradeRectName); // Delete the trade rectangle
      }
   }
};
// Pointer to CArrayObj

// Declare the dynamic array to hold order blocks
CArrayObj *orderBlocks;
color OBClr;
datetime T1;
datetime T2;

int OnInit() {
   orderBlocks = new CArrayObj(); // Allocate memory for the array
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   ObjectsDeleteAll(0, "OB");

   // Clear and free the order blocks
   if (orderBlocks != NULL) {
      orderBlocks.Clear(); // This will delete objects inside
      delete orderBlocks; // Free the array memory
      orderBlocks = NULL;
   }
}

void OnTick() {
   if (isNewBar()) {
      static int prevDay = 0;

      MqlDateTime structTime;
      TimeCurrent(structTime);
      structTime.min = 0;
      structTime.sec = 0;

      structTime.hour = Time1Hstrt;
      datetime timestrt = StructToTime(structTime);

      structTime.hour = Time1Hend;
      datetime timend = StructToTime(structTime);

      getOrderB();
      double Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      for (int i = orderBlocks.Total() - 1; i >= 0; i--) {
         COrderBlock *OB = (COrderBlock *)orderBlocks.At(i);

         if (CheckPointer(OB) != POINTER_INVALID && !OB.traded) {

            if(OB.direction > 0 && Ask < OB.high){
               double entry = Ask;
               double tp = getHigh(iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, iBarShift(_Symbol, PERIOD_CURRENT, OB.time)));
               double sl = NormalizeDouble(OB.low - Mgtn, _Digits);

               T2 = getTime(0);
               OB.draw(T1, T2, BullOB);
               trade.Buy(Lots, _Symbol, entry, sl, tp, "OB buy");
               OB.traded = true;
               //OB.removeDrawings();
               orderBlocks.Delete(i); // Delete from array
               delete OB; // Free memory


            }
         }
         if(CheckPointer(OB) != POINTER_INVALID && !OB.traded){
            if (OB.direction < 0 && Bid > OB.low) {
                  double entry = Bid;
                  double tp = getLow(iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, iBarShift(_Symbol, PERIOD_CURRENT, OB.time)));
                  double sl = NormalizeDouble(OB.high + Mgtn, _Digits);

                  T2 = getTime(0);
                  OB.draw(T1, T2, BearOB);
                  trade.Sell(Lots, _Symbol, entry, sl, tp, "OB sell");
                  OB.traded = true;
                  //OB.removeDrawings();
                  orderBlocks.Delete(i); // Delete from array
                  delete OB; // Free memory


            }
         }
      }
   }
}

void getOrderB(){

   static int prevDay = 0;

   MqlDateTime structTime;
   TimeCurrent(structTime);
   structTime.min = 0;
   structTime.sec = 0;

   structTime.hour = Time1Hstrt;
   datetime timestrt = StructToTime(structTime);

   structTime.hour = Time1Hend;
   datetime timend = StructToTime(structTime);

   int visibleBars = (int)ChartGetInteger(0,CHART_VISIBLE_BARS);

   for(int i = 1; i <= visibleBars; i++){
      if(getOpen(i) < getClose(i)){ // index is i since the loop starts from i which is = 1 "for(int i = 1)..."
         if(getOpen(i + 2) < getClose(i + 2)){
            if(getOpen(i + 3) > getClose(i + 3) && getOpen(i + 3) < getClose(i + 2)){
               Print("Bullish Order Block confirmed at: ", TimeToString(getTime(i + 2), TIME_DATE||TIME_MINUTES));
               //isBullishOB = true;
               //OB = new COrderBlock();
               int direction = 1;
               datetime time = getTime(i + 3);
               double high = getHigh(i + 3);
               double low = getLow(i + 3);
               isBullishOB = true;

               OBClr = isBullishOB ? BullOB : BearOB;

               // specify strt time
               T1 = time;
               // reset BULLOB flag
               isBullishOB = false;
               // crucial
               COrderBlock *newOB = new COrderBlock(direction, time, high, low);
               orderBlocks.Add(newOB);
               break;

               //delete newOB;
            }
         }
      }
      if(getOpen(i) > getClose(i)){
         if(getOpen(i + 2) > getClose(i + 2)){
            if(getOpen(i + 3) < getClose(i + 3) && getOpen(i + 3) < getClose(i + 2)){
               Print("Bearish Order Block confirmed at: ", TimeToString(getTime(i + 2), TIME_DATE||TIME_MINUTES));
               //isBearishOB = true;
               //OB = new COrderBlock();
               int direction = -1;
               datetime time = getTime(i + 3);
               double high = getHigh(i + 3);
               double low = getLow(i + 3);
               isBearishOB = true;

               OBClr = isBearishOB ? BearOB : BullOB;

               T1 = time;

               // reset the BEAROB flag
               isBearishOB = false;
               // crusssial
               COrderBlock *newOB = new COrderBlock(direction, time, high, low);
               orderBlocks.Add(newOB);
               break;

               //delete newOB;
            }
         }
      }
    }

}

double getHigh(int index) {
    return iHigh(_Symbol, _Period, index);
}

double getLow(int index) {
    return iLow(_Symbol, _Period, index);
}

double getOpen(int index){
   return iOpen(_Symbol, _Period, index);
}

double getClose(int index){
   return iClose(_Symbol, _Period, index);
}

datetime getTime(int index) {
    return iTime(_Symbol, _Period, index);
}

bool isNewBar() {
   // Memorize the time of opening of the last bar in the static variable
   static datetime last_time = 0;

   // Get current time
   datetime lastbar_time = (datetime)SeriesInfoInteger(Symbol(), Period(), SERIES_LASTBAR_DATE);

   // First call
   if (last_time == 0) {
      last_time = lastbar_time;
      return false;
   }

   // If the time differs (new bar)
   if (last_time != lastbar_time) {
      last_time = lastbar_time;
      return true;
   }

   // If no new bar, return false
   return false;
}

void deler(){
   static int prevDay = 0;

   MqlDateTime structTime;
   TimeCurrent(structTime);
   structTime.min = 0;
   structTime.sec = 0;

   structTime.hour = Time1Hstrt;
   datetime timestrt = StructToTime(structTime);

   structTime.hour = Time1Hend;
   datetime timend = StructToTime(structTime);

}
```

From our previous article we discussed the core functionality and design of this code, it implements an algorithmic trading strategy centered around identifying and trading "order blocks" – specific candlestick patterns believed to signal potential market reversals. Built as an Expert Advisor (EA), it combines price action analysis with automated trade execution. The code leverages object-oriented programming through the \`COrderBlock\` class to store critical pattern data (timestamps, price boundaries, direction) and manage visual chart annotations. A dynamic array \`CArrayObj\` tracks active order blocks, ensuring efficient memory management and real-time pattern monitoring across multiple chart instances.

The \`getOrderB()\` function drives pattern recognition by scanning historical candlesticks for specific bullish/bearish sequences. Bullish order blocks are identified when three consecutive bullish candles follow a bearish candle, while bearish patterns require three bearish candles after a bullish one. Detected patterns are instantiated as \`COrderBlock\` objects with directional flags (1 for bullish, -1 for bearish) and stored in the array. The code incorporates user-configurable time filters (\`Time1Hstrt\`, \`Time1Hend\`) to focus on specific trading sessions, enhancing pattern relevance.

On each new bar (detected via \`isNewBar()\`), the EA processes active order blocks in the \`OnTick()\` function. For bullish patterns, it enters long positions when price action breaks above the order block's high, setting stop losses below the pattern's low minus a margin (\`Mtgn\`). Bearish trades trigger on breakdowns below pattern lows with stops above pattern highs. The EA uses \`CTrade\`for order management, calculating take-profit levels based on recent swing highs/lows. Executed trades automatically remove their associated order blocks and chart drawings to avoid duplicate signals.

The system employs multiple safeguards: stop-loss distances are normalized to broker digit requirements, position sizing remains fixed at \`Lots\`, and visual indicators (colored rectangles) provide transparency for strategy validation. Users can customize key parameters like stop-loss distance (StopLoss), profit margins (Mtgn), and session timing through input variables. The code balances automation with discretionary elements – while pattern detection is algorithmic, trade execution respects broker-specific constraints like minimum stop distances and spread considerations.

Python version:

Before we get started, as always, we need historical market data. This data serves as the foundation for training our AI model and validating the accuracy of our strategy. I assume that by now you are familiar with obtaining historical data. However, if you are unsure or need a refresher, please refer to the first part of my guide on [integrating MQL5 with data processing packages](https://www.mql5.com/en/articles/16446). Below we simply load the historical data.

```
import pandas as pd

# Load historical data
file_path = '/home/int_junkie/Documents/DataVisuals/AI inside MQL5/XAUUSD_H1.csv'
data = pd.read_csv(file_path)

# Display the first few rows and column names
print(data.head())
print(data.columns)
```

In structuring the Python version of the code, we ensure that it is designed to process historical market data effectively, which serves as the foundation for training our AI model. The first step is to establish a data pipeline that allows the script to ingest, preprocess, and organize historical price data, including key features such as open, high, low, close prices, volume, and any relevant technical indicators. This structured format enables the AI model to learn meaningful patterns from past market behavior.

Additionally, we integrate functionality to train a Long Short-Term Memory (LSTM) model, a specialized type of recurrent neural network (RNN) known for its ability to analyze sequential data and capture long-term dependencies. The training process is aligned with the existing buy and sell logic of the original MQL5 strategy, ensuring that the model learns to make trade decisions based on historical price action. By mapping past data to the corresponding buy and sell signals, the LSTM model gradually refines its predictive capabilities, enabling it to anticipate potential market movements more effectively. This structured approach allows us to bridge the gap between traditional rule-based trading and AI-powered decision-making, ultimately improving the adaptability and accuracy of the trading system.

```
import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

# Constants
LOTS = 0.01
TAKE_PROFIT = 170
STOP_LOSS = 350
MGTN = 0.85
TIME1_HSTRT = 3
TIME1_HEND = 4

# Helper functions
def get_high(data, index):
    return data.iloc[index]['<HIGH>']

def get_low(data, index):
    return data.iloc[index]['<LOW>']

def get_open(data, index):
    return data.iloc[index]['<OPEN>']

def get_close(data, index):
    return data.iloc[index]['<CLOSE>']

def get_time(data, index):
    return data.iloc[index]['DATETIME']  # Combined datetime column

def is_new_bar(current_time, last_time):
    return current_time != last_time

class OrderBlock:
    def __init__(self, direction, time, high, low):
        self.direction = direction
        self.time = time
        self.high = high
        self.low = low
        self.traded = False

def get_order_blocks(data):
    order_blocks = []
    visible_bars = len(data)

    for i in range(1, visible_bars - 3):  # Adjusted to avoid index errors
        if get_open(data, i) < get_close(data, i):  # Bullish condition
            if get_open(data, i + 2) < get_close(data, i + 2):
                if get_open(data, i + 3) > get_close(data, i + 3) and get_open(data, i + 3) < get_close(data, i + 2):
                    print(f"Bullish Order Block confirmed at: {get_time(data, i + 2)}")
                    direction = 1
                    time = get_time(data, i + 3)
                    high = get_high(data, i + 3)
                    low = get_low(data, i + 3)
                    order_blocks.append(OrderBlock(direction, time, high, low))
                    break

        if get_open(data, i) > get_close(data, i):  # Bearish condition
            if get_open(data, i + 2) > get_close(data, i + 2):
                if get_open(data, i + 3) < get_close(data, i + 3) and get_open(data, i + 3) < get_close(data, i + 2):
                    print(f"Bearish Order Block confirmed at: {get_time(data, i + 2)}")
                    direction = -1
                    time = get_time(data, i + 3)
                    high = get_high(data, i + 3)
                    low = get_low(data, i + 3)
                    order_blocks.append(OrderBlock(direction, time, high, low))
                    break

    return order_blocks

def simulate_trading(data, order_blocks):
    trades = []
    last_time = None

    for i, row in data.iterrows():
        current_time = row['DATETIME']
        if is_new_bar(current_time, last_time):
            last_time = current_time

            bid = row['<CLOSE>']  # Assuming bid price is close price
            ask = row['<CLOSE>']  # Assuming ask price is close price

            for ob in order_blocks:
                if not ob.traded:
                    if ob.direction > 0 and ask < ob.high:  # Buy condition
                        entry = ask
                        tp = data.iloc[:i]['<HIGH>'].max()  # Take profit as highest high
                        sl = ob.low - MGTN  # Stop loss
                        trades.append({
                            'time': current_time,
                            'direction': 'buy',
                            'entry': entry,
                            'tp': tp,
                            'sl': sl
                        })
                        ob.traded = True

                    if ob.direction < 0 and bid > ob.low:  # Sell condition
                        entry = bid
                        tp = data.iloc[:i]['<LOW>'].min()  # Take profit as lowest low
                        sl = ob.high + MGTN  # Stop loss
                        trades.append({
                            'time': current_time,
                            'direction': 'sell',
                            'entry': entry,
                            'tp': tp,
                            'sl': sl
                        })
                        ob.traded = True

    return trades
```

Here the code implements a trading strategy based on order block detection and trade simulation using historical market data. It starts by importing essential libraries, including Pandas for data manipulation, NumPy for numerical operations, and Keras/TensorFlow for potential AI integration. The script defines key trading parameters such as lot size (LOTS), take profit (TAKE\_PROFIT), stop loss (STOP\_LOSS), and a risk management factor (MGTN). The helper functions extract key price points (open, high, low, close) and timestamp values from the dataset, ensuring structured access to historical data. The OrderBlock class is introduced to store information about detected bullish or bearish order blocks, which are critical areas in price action trading where institutions may have placed significant orders.

The get\_order\_blocks function scans through the historical price data to detect potential order blocks based on a series of bullish or bearish price movements. It identifies patterns where a price action shift occurs, storing the time, high, and low values of the detected order blocks. Once order blocks are identified, the simulate\_trading function executes simulated trades. It iterates through historical data, checking whether new bars (candles) appear and if price conditions match the criteria for buying or selling based on the detected order blocks. If a buy or sell condition is met, the function records a trade, setting an entry price, take profit, and stop loss dynamically based on past high/low values. This setup allows for back-testing the strategy by analyzing how trades would have performed historically, laying the groundwork for integrating AI to optimize trade execution further.

```
# Columns: ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']
data = pd.read_csv('/home/int_junkie/Documents/DataVisuals/AI inside MQL5/XAUUSD_H1.csv', delimiter='\t')

# Combine DATE and TIME into a single DATETIME column
data['DATETIME'] = pd.to_datetime(data['<DATE>'] + ' ' + data['<TIME>'])

# Drop the original DATE and TIME columns
data.drop(columns=['<DATE>', '<TIME>'], inplace=True)

# Step 1: Detect order blocks
order_blocks = get_order_blocks(data)

# Step 2: Simulate trading based on order blocks
trades = simulate_trading(data, order_blocks)
```

After merging, the original date and time columns are removed to keep the dataset clean. Next, the script calls\`get\_order\_blocks(data)\`, which scans the price data to identify potential bullish and bearish order blocks, key areas where significant price reversals or institutional orders may have been placed. Once the order blocks are detected, \`simulate\_trading(data, order\_blocks)\` runs a back-test by checking whether the market conditions trigger any of these order blocks, executing simulated buy or sell trades based on predefined entry, take profit, and stop loss rules. Basically, we evaluate the effectiveness of the trading strategy using past price action data.

```
# Features: Historical OHLC data
# Labels: Buy (1), Sell (-1), Hold (0)
labels = []
for i, row in data.iterrows():
    label = 0  # Hold by default
    for trade in trades:
        if trade['time'] == row['DATETIME']:
            label = 1 if trade['direction'] == 'buy' else -1
    labels.append(label)

data['label'] = labels

# Step 4: Train LSTM model (example using Keras)
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length][['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values)
        y.append(data.iloc[i + seq_length]['label'])
    return np.array(X), np.array(y)

seq_length = 50  # Sequence length for LSTM
X, y = create_sequences(data, seq_length)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 4)))  # 4 features: open, high, low, close
model.add(Dense(1, activation='tanh'))  # Output: -1 (sell), 0 (hold), 1 (buy)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Save the model
model.save('lstm_trading_model.h5')
```

### Output:

![](https://c.mql5.com/2/128/outLSTM.png)

![](https://c.mql5.com/2/134/outLSTM2.png)

This code segment trains an LSTM model to forecast trading decisions using historical price data. The dataset is labeled with three possible signals: **1** (buy), **-1** (sell), or **0** (hold). Labels are assigned by iterating through historical timestamps—if a trade occurred at a specific time, the label reflects its direction; otherwise, it defaults to **0** (no action). This creates a supervised learning framework where the model learns to associate sequences of past market data with subsequent trading actions.

For the LSTM training, the data is restructured into **50-step sequences** of OHLC (Open, High, Low, Close) values. Each sequence serves as input to predict the label at the next time step. The model architecture includes:

- An **LSTM layer** (50 units) to analyze temporal patterns in the 4-feature input (OHLC).
- A **Dense output layer** with _tanh activation_ to produce predictions between -1 and 1.

The model is trained for **20 epochs** using the Adam optimizer and mean squared error (MSE) loss. After training, it is saved as \`lstm\_trading\_model.h5\`, ready for deployment in trading strategies to generate real-time signals.

### Deploy the Model

With your LSTM model now serialized as a .h5 file, the subsequent phase involves operationalizing it within the MQL5 ecosystem. This requires creating an interoperability layer between Python’s machine learning framework and MQL5’s native trading infrastructure, as direct Python integration is unsupported. A standard solution involves deploying a Python-based microservice (using lightweight frameworks like Flask or FastAPI) to host the trained model. This server acts as a prediction endpoint: it ingests formatted market data from MQL5, executes LSTM-based inferences, and returns trading signals in real time. On the MQL5 side, your Expert Advisor leverages the \`WebRequest()\` API to transmit sequential price data to this endpoint and parse its JSON responses, effectively transforming raw predictions into executable actions.

Following successful implementation of this bridge, the focus shifts to live strategy integration. The Expert Advisor will autonomously stream curated data batches (e.g., 50-bar OHLC sequences) to the inference server at configurable intervals. The Python service preprocesses this input, executes tensor operations through the LSTM architecture, and returns probabilistic trade directives (buy/sell/hold) with confidence metrics. These signals then feed into the EA’s decision pipeline, where they’re combined with predefined risk parameters—dynamic stop-loss thresholds, profit-target ratios, and position-sizing algorithms—to execute managed trades. To mitigate operational risks, implement a phased rollout: validate the system in a sandboxed demo environment, continuously monitor inference latency and model drift, and embed contingency protocols (e.g., reverting to technical indicator-based strategies) to maintain functionality during server outages or prediction anomalies.

### Conclusion

In this project, we took an existing MQL5 trading strategy and enhanced it by integrating an AI-powered decision-making process using a Long Short-Term Memory (LSTM) neural network. We began by translating the core logic of the MQL5 strategy into Python, replicating its behavior using historical OHLC data. From there, we identified key trading signals, such as order blocks, and simulated trades to generate labeled data for training the AI model. We then trained the LSTM model on sequences of historical price data to predict whether the next market move should be a buy, sell, or hold. Finally, the model was saved as a \`.h5\` file, ready to be deployed for live or semi-automated trading.

The integration of AI into a traditional rule-based strategy brings powerful advantages to traders. Unlike static logic that follows fixed conditions, the LSTM model can learn complex price behavior and adapt to changing market dynamics over time. This makes the strategy more flexible, potentially more accurate, and less prone to false signals that rigid systems might fall for. Traders benefit from a system that combines the best of both worlds: the structure and reliability of technical rules with the adaptability and learning capabilities of machine learning.

This hybrid approach improves entry and exit decisions and opens the door for more intelligent, data-driven risk management in real-time trading. Unlike static logic that follows fixed conditions, the LSTM model can learn complex price behavior and adapt to changing market dynamics over time. This makes the strategy more flexible, potentially more accurate, and less prone to false signals that rigid systems might fall for. Traders benefit from a system that combines the best of both worlds: the structure and reliability of technical rules with the adaptability and learning capabilities of machine learning. This hybrid approach not only improves entry and exit decisions but also opens the door for more intelligent, data-driven risk management in real-time trading.

| File Name | Description |
| --- | --- |
| FIB\_OB.mq5 | File containing the original MQL5 strategy |
| FIB\_OB to AI.ipynb | File containing the Notebook to convert strategy logic, train the model, and save it |
| XAUUSD\_H1.csv | File containing XAUUSD historical price data |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16973.zip "Download all attachments in the single ZIP archive")

[Model\_inside\_MQL5.zip](https://www.mql5.com/en/articles/download/16973/model_inside_mql5.zip "Download Model_inside_MQL5.zip")(22.06 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)
- [Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)
- [Black-Scholes Greeks: Gamma and Delta](https://www.mql5.com/en/articles/20054)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/484912)**
(1)


![malky3200](https://c.mql5.com/avatar/avatar_na2.png)

**[malky3200](https://www.mql5.com/en/users/malky3200)**
\|
28 Apr 2025 at 10:25

Question.... Is there any reason you have the epoch set to 20 ?

Is it possible to have the AI learn on an on going basis ?


![From Basic to Intermediate: SWITCH Statement](https://c.mql5.com/2/93/Do_bisico_ao_intermedicrio_Comando_SWITCH___LOGO.png)[From Basic to Intermediate: SWITCH Statement](https://www.mql5.com/en/articles/15391)

In this article, we will learn how to use the SWITCH statement in its simplest and most basic form. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Price Action Analysis Toolkit Development (Part 20): External Flow (IV) — Correlation Pathfinder](https://c.mql5.com/2/134/Price_Action_Analysis_Toolkit_Development_Part_20___LOGO.png)[Price Action Analysis Toolkit Development (Part 20): External Flow (IV) — Correlation Pathfinder](https://www.mql5.com/en/articles/17742)

Correlation Pathfinder offers a fresh approach to understanding currency pair dynamics as part of the Price Action Analysis Toolkit Development Series. This tool automates data collection and analysis, providing insight into how pairs like EUR/USD and GBP/USD interact. Enhance your trading strategy with practical, real-time information that helps you manage risk and spot opportunities more effectively.

![From Novice to Expert: Programming Candlesticks](https://c.mql5.com/2/134/From_Novice_to_Expert_Programming_Candlesticks___LOGO__1.png)[From Novice to Expert: Programming Candlesticks](https://www.mql5.com/en/articles/17525)

In this article, we take the first step in MQL5 programming, even for complete beginners. We'll show you how to transform familiar candlestick patterns into a fully functional custom indicator. Candlestick patterns are valuable as they reflect real price action and signal market shifts. Instead of manually scanning charts—an approach prone to errors and inefficiencies—we'll discuss how to automate the process with an indicator that identifies and labels patterns for you. Along the way, we’ll explore key concepts like indexing, time series, Average True Range (for accuracy in varying market volatility), and the development of a custom reusable Candlestick Pattern library for use in future projects.

![Decoding Opening Range Breakout Intraday Trading Strategies](https://c.mql5.com/2/134/Decoding_Opening_Range_Breakout_Intraday_Trading_Strategies__LOGO.png)[Decoding Opening Range Breakout Intraday Trading Strategies](https://www.mql5.com/en/articles/17745)

Opening Range Breakout (ORB) strategies are built on the idea that the initial trading range established shortly after the market opens reflects significant price levels where buyers and sellers agree on value. By identifying breakouts above or below a certain range, traders can capitalize on the momentum that often follows as the market direction becomes clearer. In this article, we will explore three ORB strategies adapted from the Concretum Group.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=rxxmiempulfutgjjovwvsagxgkugqivi&ssn=1769093826553967397&ssn_dr=0&ssn_sr=0&fv_date=1769093826&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16973&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Integrating%20AI%20model%20into%20already%20existing%20MQL5%20trading%20strategy%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909382627147748&fz_uniq=5049508636968987841&sv=2552)

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