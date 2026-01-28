---
title: Utilizing CatBoost Machine Learning model as a Filter for Trend-Following Strategies
url: https://www.mql5.com/en/articles/16487
categories: Integration, Expert Advisors, Machine Learning
relevance_score: 4
scraped_at: 2026-01-23T17:49:16.082896
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/16487&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6429447454119099498)

MetaTrader 5 / Integration


### Introduction

CatBoost is a powerful tree-based machine learning model that specializes in decision-making based on stationary features. Other tree-based models like XGBoost and Random Forest share similar traits in terms of their robustness, ability to handle complex patterns, and interpretability. These models have a wide range of uses, from feature analysis to risk management.

In this article, we're going to walk through the procedure of utilizing a trained CatBoost model as a filter for a classic moving average cross trend-following strategy. This article is meant to provide insights into the strategy development process while addressing the challenges one may face along the way. I will introduce my workflow of fetching data from MetaTrader 5, training machine learning model in python, and integrating back to MetaTrader 5 expert advisors. By the end of this article, we will validate the strategy through statistical testing and discuss future aspirations extending from the current approach.

### Intuition

The rule of thumb in the industry for developing CTA (Commodity Trading Advisor) strategy is that it's best to have a clear, intuitive explanation behind every strategy idea. This is basically how people think of strategy ideas in the first place, not to mention it avoids overfitting as well. This suggestion is subservient even working with machine learning models. We'll try to explain the intuition behind this idea.

Why this could work:

CatBoost model creates decision trees that take in the feature inputs and output the probability of each outcome. In this case, we're only training on binary outcomes (1 is win,0 is loss). The model will alter rules in the decision trees so that it minimizes the loss function in the training data set. If the model displays a certain level of predictability on the out-of-sample testing outcome, we may consider using it to filter out trades that have little probability of winning, which could in turn boost the overall profitability.

A realistic expectation for retail traders like you and I is that the models we train will not be like oracles, but rather only slightly better than random walk. There are plenty of ways to improve the model precision, which I will discuss later, but nevertheless it's a great endeavor for slight improvement.

### Optimizing Backbone Strategy

We already know from the above section that we can only expect the model to boost the performance slightly, and thus it's crucial for the backbone strategy to already have some sort of profitability.

The strategy also has to be able to generate abundant samples because:

1. The model will filter out a portion of the trades, we want to make sure there are enough samples left to exhibit statistical significance of Laws of Big Numbers.
2. We need enough samples for the model to train on so that it minimizes the loss function for in-sample data effectively.

We use a historically proven trend following strategy which takes trades when two moving averages of different period crosses, and we exit trades when the price turns to the opposite side of the moving average. i.e. following the trend. The following MQL5 code is the expert advisor for this strategy.

```
#include <Trade/Trade.mqh>
//XAU - 1h.
CTrade trade;

input ENUM_TIMEFRAMES TF = PERIOD_CURRENT;
input ENUM_MA_METHOD MaMethod = MODE_SMA;
input ENUM_APPLIED_PRICE MaAppPrice = PRICE_CLOSE;
input int MaPeriodsFast = 15;
input int MaPeriodsSlow = 25;
input int MaPeriods = 200;
input double lott = 0.01;
ulong buypos = 0, sellpos = 0;
input int Magic = 0;
int barsTotal = 0;
int handleMaFast;
int handleMaSlow;
int handleMa;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(Magic);
   handleMaFast =iMA(_Symbol,TF,MaPeriodsFast,0,MaMethod,MaAppPrice);
   handleMaSlow =iMA(_Symbol,TF,MaPeriodsSlow,0,MaMethod,MaAppPrice);
   handleMa = iMA(_Symbol,TF,MaPeriods,0,MaMethod,MaAppPrice);
   return(INIT_SUCCEEDED);
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
  int bars = iBars(_Symbol,PERIOD_CURRENT);
  //Beware, the last element of the buffer list is the most recent data, not [0]
  if (barsTotal!= bars){
     barsTotal = bars;
     double maFast[];
     double maSlow[];
     double ma[];
     CopyBuffer(handleMaFast,BASE_LINE,1,2,maFast);
     CopyBuffer(handleMaSlow,BASE_LINE,1,2,maSlow);
     CopyBuffer(handleMa,0,1,1,ma);
     double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
     double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
     double lastClose = iClose(_Symbol, PERIOD_CURRENT, 1);
     //The order below matters
     if(buypos>0&& lastClose<maSlow[1]) trade.PositionClose(buypos);
     if(sellpos>0 &&lastClose>maSlow[1])trade.PositionClose(sellpos);
     if (maFast[1]>maSlow[1]&&maFast[0]<maSlow[0]&&buypos ==sellpos)executeBuy();
     if(maFast[1]<maSlow[1]&&maFast[0]>maSlow[0]&&sellpos ==buypos) executeSell();
     if(buypos>0&&(!PositionSelectByTicket(buypos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      buypos = 0;
      }
     if(sellpos>0&&(!PositionSelectByTicket(sellpos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      sellpos = 0;
      }
    }
 }

//+------------------------------------------------------------------+
//| Expert trade transaction handling function                       |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans, const MqlTradeRequest& request, const MqlTradeResult& result) {
    if (trans.type == TRADE_TRANSACTION_ORDER_ADD) {
        COrderInfo order;
        if (order.Select(trans.order)) {
            if (order.Magic() == Magic) {
                if (order.OrderType() == ORDER_TYPE_BUY) {
                    buypos = order.Ticket();
                } else if (order.OrderType() == ORDER_TYPE_SELL) {
                    sellpos = order.Ticket();
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Execute sell trade function                                      |
//+------------------------------------------------------------------+
void executeSell() {
       double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
       bid = NormalizeDouble(bid,_Digits);
       trade.Sell(lott,_Symbol,bid);
       sellpos = trade.ResultOrder();
       }

//+------------------------------------------------------------------+
//| Execute buy trade function                                       |
//+------------------------------------------------------------------+
void executeBuy() {
       double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
       ask = NormalizeDouble(ask,_Digits);
       trade.Buy(lott,_Symbol,ask);
       buypos = trade.ResultOrder();
}
```

**For validating your backbone strategy, here are a few things to consider:**

1. Enough sample size (frequency depends on your timeframe and signal restriction, but total samples I suggest 1000-10000. Each trade is a sample.)
2. Already exhibits some sort of profitability but not too much (Profit factor is 1-1.15 I would say is good enough. Because the MetaTrader 5 tester already accounts for spreads, having a profit factor of 1 means it has a statistical edge already. If the profit factor exceeds 1.15, the strategy is most likely good enough to stand on its own, and you probably don't need more filters to increase complexity.)
3. The backbone strategy doesn't have too many parameters. (The backbone strategy is better to be simple, since using a machine learning model as a filter already increases plenty of complexity to your strategy. The less filter, the less chance of overfitting.)

**These are what I did to optimize the strategy:**

1. Finding a good timeframe. After running the code in different timeframe, I found that this strategy works best on higher timeframe, but to generate enough samples, I eventually stuck with 1h timeframe.
2. Optimizing parameters. I optimized the slow MA's period and fast MA's period with step 5 and obtained the settings in the above code.
3. I tried adding a rule where the entry has to be already above a moving average of some period, indicating that it is already trending in the corresponding direction. (Important to note that adding filters also has to have an intuitive explanation, and validate this hypothesis to test without snooping data.) But I eventually found that this didn't improve the performance much, so I discarded this idea to avoid over-complications.

Finally, this is the test result on XAUUSD 1h timeframe, 2004.1.1 – 2024.11.1

![setting](https://c.mql5.com/2/159/TF-setting.png)

![parameters](https://c.mql5.com/2/159/TF-parameters.png)

![curve1](https://c.mql5.com/2/159/backbone_curve.png)

![result1](https://c.mql5.com/2/159/backbone_result.png)

### Fetching Data

For training the model, we need the features values upon each trade, and we need to know the outcome of each trade. My most efficient and reliable way is to write an expert advisor that stores all the corresponding features into a 2-D array, and for the outcome data we simply export the trading report from the backtest.

Firstly, to get the outcome data, we can simply go to the backtest and right click select report and open XML like this.

![excel report](https://c.mql5.com/2/159/ExcelReport__1.png)

Next, to turn a double array into CSV, we'll use the CFileCSV class explained in this [article](https://www.mql5.com/en/articles/12069).

We build on top of our backbone strategy script with the following steps:

**1\.** Include the mqh file and create class object.

```
#include <FileCSV.mqh>

CFileCSV csvFile;
```

**2\.** Declare the file name to be saved and the headers which have "index" and all the other feature names. The "index" here is merely used for updating the array index while running the tester and will get dropped later in python.

```
string fileName = "ML.csv";
string headers[] = {
    "Index",
    "Accelerator Oscillator",
    "Average Directional Movement Index",
    "Average Directional Movement Index by Welles Wilder",
    "Average True Range",
    "Bears Power",
    "Bulls Power",
    "Commodity Channel Index",
    "Chaikin Oscillator",
    "DeMarker",
    "Force Index",
    "Gator",
    "Market Facilitation Index",
    "Momentum",
    "Money Flow Index",
    "Moving Average of Oscillator",
    "MACD",
    "Relative Strength Index",
    "Relative Vigor Index",
    "Standard Deviation",
    "Stochastic Oscillator",
    "Williams' Percent Range",
    "Variable Index Dynamic Average",
    "Volume",
    "Hour",
    "Stationary"
};

string data[10000][26];
int indexx = 0;
vector xx;
```

**3\.** We write a getData() function which calculates all the feature values and stores them into the global array. In this case, we use time, oscillators, and stationary price as features. This function will be called every time there's a trade signal so that it aligns with your trades. The selection of features will be mentioned later.

```
//+------------------------------------------------------------------+
//| Execute get data function                                        |
//+------------------------------------------------------------------+
vector getData(){
//23 oscillators
double ac[];        // Accelerator Oscillator
double adx[];       // Average Directional Movement Index
double wilder[];    // Average Directional Movement Index by Welles Wilder
double atr[];       // Average True Range
double bep[];       // Bears Power
double bup[];       // Bulls Power
double cci[];       // Commodity Channel Index
double ck[];        // Chaikin Oscillator
double dm[];        // DeMarker
double f[];         // Force Index
double g[];         // Gator
double bwmfi[];     // Market Facilitation Index
double m[];         // Momentum
double mfi[];       // Money Flow Index
double oma[];       // Moving Average of Oscillator
double macd[];      // Moving Averages Convergence/Divergence
double rsi[];       // Relative Strength Index
double rvi[];       // Relative Vigor Index
double std[];       // Standard Deviation
double sto[];       // Stochastic Oscillator
double wpr[];       // Williams' Percent Range
double vidya[];     // Variable Index Dynamic Average
double v[];         // Volume

CopyBuffer(handleAc, 0, 1, 1, ac);           // Accelerator Oscillator
CopyBuffer(handleAdx, 0, 1, 1, adx);         // Average Directional Movement Index
CopyBuffer(handleWilder, 0, 1, 1, wilder);   // Average Directional Movement Index by Welles Wilder
CopyBuffer(handleAtr, 0, 1, 1, atr);         // Average True Range
CopyBuffer(handleBep, 0, 1, 1, bep);         // Bears Power
CopyBuffer(handleBup, 0, 1, 1, bup);         // Bulls Power
CopyBuffer(handleCci, 0, 1, 1, cci);         // Commodity Channel Index
CopyBuffer(handleCk, 0, 1, 1, ck);           // Chaikin Oscillator
CopyBuffer(handleDm, 0, 1, 1, dm);           // DeMarker
CopyBuffer(handleF, 0, 1, 1, f);             // Force Index
CopyBuffer(handleG, 0, 1, 1, g);             // Gator
CopyBuffer(handleBwmfi, 0, 1, 1, bwmfi);     // Market Facilitation Index
CopyBuffer(handleM, 0, 1, 1, m);             // Momentum
CopyBuffer(handleMfi, 0, 1, 1, mfi);         // Money Flow Index
CopyBuffer(handleOma, 0, 1, 1, oma);         // Moving Average of Oscillator
CopyBuffer(handleMacd, 0, 1, 1, macd);       // Moving Averages Convergence/Divergence
CopyBuffer(handleRsi, 0, 1, 1, rsi);         // Relative Strength Index
CopyBuffer(handleRvi, 0, 1, 1, rvi);         // Relative Vigor Index
CopyBuffer(handleStd, 0, 1, 1, std);         // Standard Deviation
CopyBuffer(handleSto, 0, 1, 1, sto);         // Stochastic Oscillator
CopyBuffer(handleWpr, 0, 1, 1, wpr);         // Williams' Percent Range
CopyBuffer(handleVidya, 0, 1, 1, vidya);     // Variable Index Dynamic Average
CopyBuffer(handleV, 0, 1, 1, v);             // Volume
//2 means 2 decimal places
data[indexx][0] = IntegerToString(indexx);
data[indexx][1] = DoubleToString(ac[0], 2);       // Accelerator Oscillator
data[indexx][2] = DoubleToString(adx[0], 2);      // Average Directional Movement Index
data[indexx][3] = DoubleToString(wilder[0], 2);   // Average Directional Movement Index by Welles Wilder
data[indexx][4] = DoubleToString(atr[0], 2);      // Average True Range
data[indexx][5] = DoubleToString(bep[0], 2);      // Bears Power
data[indexx][6] = DoubleToString(bup[0], 2);      // Bulls Power
data[indexx][7] = DoubleToString(cci[0], 2);      // Commodity Channel Index
data[indexx][8] = DoubleToString(ck[0], 2);       // Chaikin Oscillator
data[indexx][9] = DoubleToString(dm[0], 2);       // DeMarker
data[indexx][10] = DoubleToString(f[0], 2);       // Force Index
data[indexx][11] = DoubleToString(g[0], 2);       // Gator
data[indexx][12] = DoubleToString(bwmfi[0], 2);   // Market Facilitation Index
data[indexx][13] = DoubleToString(m[0], 2);       // Momentum
data[indexx][14] = DoubleToString(mfi[0], 2);     // Money Flow Index
data[indexx][15] = DoubleToString(oma[0], 2);     // Moving Average of Oscillator
data[indexx][16] = DoubleToString(macd[0], 2);    // Moving Averages Convergence/Divergence
data[indexx][17] = DoubleToString(rsi[0], 2);     // Relative Strength Index
data[indexx][18] = DoubleToString(rvi[0], 2);     // Relative Vigor Index
data[indexx][19] = DoubleToString(std[0], 2);     // Standard Deviation
data[indexx][20] = DoubleToString(sto[0], 2);     // Stochastic Oscillator
data[indexx][21] = DoubleToString(wpr[0], 2);     // Williams' Percent Range
data[indexx][22] = DoubleToString(vidya[0], 2);   // Variable Index Dynamic Average
data[indexx][23] = DoubleToString(v[0], 2);       // Volume

    datetime currentTime = TimeTradeServer();
    MqlDateTime timeStruct;
    TimeToStruct(currentTime, timeStruct);
    int currentHour = timeStruct.hour;
data[indexx][24]= IntegerToString(currentHour);
    double close = iClose(_Symbol,PERIOD_CURRENT,1);
    double open = iOpen(_Symbol,PERIOD_CURRENT,1);
    double stationary = MathAbs((close-open)/close)*100;
data[indexx][25] = DoubleToString(stationary,2);

   vector features(26);
   for(int i = 1; i < 26; i++)
    {
      features[i] = StringToDouble(data[indexx][i]);
    }
    //A lot of the times positions may not open due to error, make sure you don't increase index blindly
    if(PositionsTotal()>0) indexx++;
    return features;
}
```

Note that we added a check here.

```
if(PositionsTotal()>0) indexx++;
```

This is because when your trade signal occurs, it may not result in a trade because the EA is running during market close time, but the tester won't take any trades.

**4\.** We save the file upon OnDeInit() is called, which is when the test is over.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if (!SaveData) return;
   if(csvFile.Open(fileName, FILE_WRITE|FILE_ANSI))
     {
      //Write the header
      csvFile.WriteHeader(headers);
      //Write data rows
      csvFile.WriteLine(data);
      //Close the file
      csvFile.Close();
     }
   else
     {
      Print("File opening error!");
     }

  }
```

Run this expert advisor in the strategy tester, after that, you should be able to see your csv file formed in the  /Tester/Agent-sth000 directory.

### Cleaning and Adjusting Data

Now we have the two data files in the bag, but there remains many underlying problems to solve.

**1\.** The backtest report is messy and is in .xlsx format. We only want whether we won or not for each trade.

First, we extract the rows where it only displays trade outcomes. You may need to scroll down your XLSX file until you see something like this:

![find row](https://c.mql5.com/2/159/find_row__2.png)

Remember the row number and apply it to the following python code:

```
import pandas as pd

# Replace 'your_file.xlsx' with the path to your file
input_file = 'ML2.xlsx'

# Load the Excel file and skip the first {skiprows} rows
df = pd.read_excel(input_file, skiprows=10757)

# Save the extracted content to a CSV file
output_file = 'extracted_content.csv'
df.to_csv(output_file, index=False)

print(f"Content has been saved to {output_file}.")
```

Then we apply this extracted content to the following code to obtain the processed bin. Where winning trades would be 1 and losing trades would be 0.

```
import pandas as pd

# Load the CSV file
file_path = 'extracted_content.csv'  # Update with the correct file path if needed
data = pd.read_csv(file_path)

# Select the 'profit' column (assumed to be 'Unnamed: 10') and filter rows as per your instructions
profit_data = data["Profit"][1:-1]
profit_data = profit_data[profit_data.index % 2 == 0]  # Filter for rows with odd indices
profit_data = profit_data.reset_index(drop=True)  # Reset index
# Convert to float, then apply the condition to set values to 1 if > 0, otherwise to 0
profit_data = pd.to_numeric(profit_data, errors='coerce').fillna(0)  # Convert to float, replacing NaN with 0
profit_data = profit_data.apply(lambda x: 1 if x > 0 else 0)  # Apply condition

# Save the processed data to a new CSV file with index
output_csv_path = 'processed_bin.csv'
profit_data.to_csv(output_csv_path, index=True, header=['bin'])

print(f"Processed data saved to {output_csv_path}")
```

The result file should look something like this

|  | bin |
| --- | --- |
| 0 | 1 |
| 1 | 0 |
| 2 | 1 |
| 3 | 0 |
| 4 | 0 |
| 5 | 1 |

Note that if all the values are 0 it may be because your starting rows are incorrect, make sure to check whether your starting row is now even or odd and change it accordingly with the python code.

**2\.** The feature data is all string due to the CFileCSV class, and they're stuck together in one column, only separated by commas.

The following python code gets the job done.

```
import pandas as pd

# Load the CSV file with semicolon separator
file_path = 'ML.csv'
data = pd.read_csv(file_path, sep=';')

# Drop rows with any missing or incomplete values
data.dropna(inplace=True)

# Drop any duplicate rows if present
data.drop_duplicates(inplace=True)

# Convert non-numeric columns to numerical format
for col in data.columns:
    if data[col].dtype == 'object':
        # Convert categorical to numerical using label encoding
        data[col] = data[col].astype('category').cat.codes

# Ensure all remaining columns are numeric and cleanly formatted for CatBoost
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)  # Drop any rows that might still contain NaNs after conversion

# Save the cleaned data to a new file in CatBoost-friendly format
output_file_path = 'Cleaned.csv'
data.to_csv(output_file_path, index=False)

print(f"Data cleaned and saved to {output_file_path}")
```

Finally, we use this code to merge the two files together so that we can easily access them as a single data frame in the future.

```
import pandas as pd

# Load the two CSV files
file1_path = 'processed_bin.csv'  # Update with the correct file path if needed
file2_path = 'Cleaned.csv'  # Update with the correct file path if needed
data1 = pd.read_csv(file1_path, index_col=0)  # Load first file with index
data2 = pd.read_csv(file2_path, index_col=0)  # Load second file with index

# Merge the two DataFrames on the index
merged_data = pd.merge(data1, data2, left_index=True, right_index=True, how='inner')

# Save the merged data to a new CSV file
output_csv_path = 'merged_data.csv'
merged_data.to_csv(output_csv_path)

print(f"Merged data saved to {output_csv_path}")
```

To confirm that two data is correctly merged, we can check the three CSV files we just produced and see if their final index is the same. If so, we're most likely chillin'.

### Training Model

We won't go too in depth into the technical explanations behind each aspect of machine learning. However, I strongly encourage you to check out _**Advances in Financial Machine Learning**_ by Marcos López de Prado if you're interested in ML trading as a whole.

Our objective for this section is crystal clear.

First, we use the pandas library to read the merged data and split the bin column as y and the rest as X.

```
data = pd.read_csv("merged_data.csv",index_col=0)
XX = data.drop(columns=['bin'])
yy = data['bin']
y = yy.values
X = XX.values
```

Then we split the data into 80% for training and 20% for testing.

We then train. The details of each parameter in the classifier are documented on the [CatBoost website](https://www.mql5.com/go?link=https://catboost.ai/ "https://catboost.ai/").

```
from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier

# Define the CatBoost model with initial parameters
catboost_clf = CatBoostClassifier(
    class_weights=[10, 1],  #more weights to 1 class cuz there's less correct cases
    iterations=20000,             # Number of trees (similar to n_estimators)
    learning_rate=0.02,          # Learning rate
    depth=5,                    # Depth of each tree
    l2_leaf_reg=5,
    bagging_temperature=1,
    early_stopping_rounds=50,
    loss_function='Logloss',    # Use 'MultiClass' if it's a multi-class problem
    random_seed=RANDOM_STATE,
    verbose=1000,                  # Suppress output (set to a positive number if you want to see training progress)
)

fit = catboost_clf.fit(X_train, y_train)
```

We save the .cbm file.

```
catboost_clf.save_model('catboost_test.cbm')
```

Unfortunately, we're not done yet. MetaTrader 5 only supports ONNX format model, so we use the following code from [this article](https://www.mql5.com/en/articles/16017) to transform it into ONNX format.

```
model_onnx = convert_sklearn(
    model,
    "catboost",
    [("input", FloatTensorType([None, X.shape[1]]))],
    target_opset={"": 12, "ai.onnx.ml": 2},
)

# And save.
with open("CatBoost_test.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())
```

### Statistical Testing

After obtaining the .onnx file, we drag it into the MQL5/Files folder. We now build on top of the expert advisor we used to fetch data earlier. Again [this article](https://www.mql5.com/en/articles/16017) already explains the procedure of initializing .onnx model in expert advisors in detail, I would just emphasize on how we change the entry criteria.

```
     if (maFast[1]>maSlow[1]&&maFast[0]<maSlow[0]&&sellpos == buypos){
        xx= getData();
        prob = cat_boost.predict_proba(xx);
        if (prob[1]<max&&prob[1]>min)executeBuy();
     }
     if(maFast[1]<maSlow[1]&&maFast[0]>maSlow[0]&&sellpos == buypos){
        xx= getData();
        prob = cat_boost.predict_proba(xx);
        Print(prob);
        if(prob[1]<max&&prob[1]>min)executeSell();
      }
```

Here we call getData() to store the vector information in variable xx, then we return the probability of success according to the model. We added a print statement so that we get a sense of what range it's going to be. For trend-following strategy, because of its low accuracy and high reward-to-risk ratio per trade, we normally see the model give probability less than 0.5.

We add a threshold for filtering out trades that display low probability of success, and we have finished the coding part. Now let's test.

Remember we split it to 8-2 ratio? Now we're going to do an out-of-sample test on the untrained data, which is approximately 2021.1.1-2024.11.1.

We first run the in-sample test with a 0.05 probability threshold to confirm we trained with the right data. The result should be almost perfect like this.

![in-sample curve](https://c.mql5.com/2/159/in-sample.png)

Then we run an out-of-sample test with no threshold as baseline. We expect, if we scale up the threshold, we should beat this baseline result by quite a margin.

![baseline curve](https://c.mql5.com/2/159/baseline_curve.png)

![baseline result](https://c.mql5.com/2/159/baseline_result.png)

Finally, we conduct out-of-sample tests to analyze the profitability patterns relative to different thresholds.

Results of threshold = 0.05:

![0.05 curve](https://c.mql5.com/2/159/0.05_curve.png)

![0.05 result](https://c.mql5.com/2/159/0.05_result.png)

Results of threshold = 0.1:

![0.1 curve](https://c.mql5.com/2/159/0.1_curve.png)

![0.1 result](https://c.mql5.com/2/159/0.1_result.png)

Results of threshold = 0.2:

![0.2 curve](https://c.mql5.com/2/159/0.2_curve.png)

![0.2 result](https://c.mql5.com/2/159/0.2_result.png)

For a threshold of 0.05, the model filtered out approximately half of the original trades, but this led to a decrease in profitability. This could suggest that the predictor is overfitted, becoming too attuned to the trained patterns and failing to capture the similar patterns shared between the training and testing sets. In financial machine learning, this is a common issue. However, when the threshold is increased to 0.1, the profit factor gradually improves, surpassing that of our baseline.

At a threshold of 0.2, the model filters out about 70% of the original trades, but the overall quality of the remaining trades is significantly more profitable than the original ones. Statistical analysis shows that within this threshold range, overall profitability is positively correlated with the threshold value. This suggests that as the model's confidence in a trade increases, so does its overall performance, which is a favorable outcome.

I ran a ten-fold [cross validation](https://www.mql5.com/go?link=https://scikit-learn.org/1.5/modules/cross_validation.html "https://scikit-learn.org/1.5/modules/cross_validation.html") in python to confirm that the model precision is consistent.

```
{'score': array([-0.97148655, -1.25263677, -1.02043177, -1.06770248, -0.97339545, -0.88611439, -0.83877111, -0.95682533, -1.02443847, -1.1385681 ])}
```

The difference between each cross-validation score is mild, indicating that the model's accuracy remains consistent across different training and testing periods.

Additionally, with an average log-loss score around -1, the model's performance can be considered moderately effective.

To further improve the model precision, one may pick up on the following ideas:

**1\. Feature engineering**

We plot the feature importance like this and remove the ones that have little importance.

For selecting features, anything market related is plausible, but make sure you make the data stationary because tree-based models use fixed value rules to process inputs.

![feature importance](https://c.mql5.com/2/159/feature_importance__2.png)

**2\. Hyperparameter tuning**

Remember the parameters in the classifier function I talked about earlier? We could write a function to loop through a grid of values and test which training parameter would yield the best cross validation scores.

**3\. Model selection**

We can try different machine learning models or different types of values to predict. People have found that while machine learning models are bad at predicting prices, it is rather competent in predicting volatility. Besides, hidden Markov model is widely used to predict hidden trends. Both of these could be potent filters for trend following strategies.

I encourage the readers to try these methods out with my attached code, and let me know if you found any success in improving the performance.

### Conclusion

In this article, we walked through the entire workflow of developing a CatBoost machine learning filter for a trend-following strategy. On the way, we highlighted different aspects to be aware of while researching machine learning strategies. In the end, we validated the strategy through statistical testing and discussed future aspirations extending from the current approach.

**Attached File Table**

| File name | Usage |
| --- | --- |
| ML-Momentum Data.mq5 | The EA to fetch features data |
| ML-Momentum.mq5 | Final execution EA |
| CB2.ipynb | The workflow to train and test CatBoost model |
| handleMql5DealReport.py | extract useful rows from deal report |
| getBinFromMql5.py | get binary outcome from the extracted content |
| clean\_mql5\_csv.py | Clean the features CSV extracted from Mt5 |
| merge\_data2.py | merge features and outcome into one CSV |
| OnnxConvert.ipynb | Convert .cbm model to .onnx format |
| Classic Trend Following.mq5 | The backbone strategy expert advisor |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16487.zip "Download all attachments in the single ZIP archive")

[ML-TF-Project.zip](https://www.mql5.com/en/articles/download/16487/ml-tf-project.zip "Download ML-TF-Project.zip")(186.72 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Decoding Opening Range Breakout Intraday Trading Strategies](https://www.mql5.com/en/articles/17745)
- [Day Trading Larry Connors RSI2 Mean-Reversion Strategies](https://www.mql5.com/en/articles/17636)
- [Exploring Advanced Machine Learning Techniques on the Darvas Box Breakout Strategy](https://www.mql5.com/en/articles/17466)
- [The Kalman Filter for Forex Mean-Reversion Strategies](https://www.mql5.com/en/articles/17273)
- [Robustness Testing on Expert Advisors](https://www.mql5.com/en/articles/16957)
- [Trend Prediction with LSTM for Trend-Following Strategies](https://www.mql5.com/en/articles/16940)
- [The Inverse Fair Value Gap Trading Strategy](https://www.mql5.com/en/articles/16659)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/477915)**
(12)


![johnboy85](https://c.mql5.com/avatar/avatar_na2.png)

**[johnboy85](https://www.mql5.com/en/users/johnboy85)**
\|
7 Jun 2025 at 10:01

Hi. I am playing around with [CatBoost](https://www.mql5.com/en/articles/8642 "Article: Gradient boosting (CatBoost) in the problems of building trading systems. Naive approach") and have got to the point where a strategy trained on (all of) 2024 data will produce >300% returns when backtested (in MetaTrader) on 2024 but performs poorly on other years. Does anyone have experience with this? It intuitively feels like overfitting, but even if I train with much lower iterations (like 1k) I get the same result.

I am training with ~40 - 50 features, on minute data, so something like 250,000 rows per year. The .cbm file size tends to come out at 1000x the number of iterations (e.g. 1000 iterations = 1MB, 10,000 iterations = 10MB, and so on). Backtesting on Metatrader limits me to about 100,000MB before the backtester grinds to a halt. I can backtest with C++ to an arbitrarily high size but my returns in metatrader vs C++ are wildely different.

![Zhuo Kai Chen](https://c.mql5.com/avatar/2024/11/6743e84b-8a3d.jpg)

**[Zhuo Kai Chen](https://www.mql5.com/en/users/sicklemql)**
\|
8 Jun 2025 at 10:23

**johnboy85 [#](https://www.mql5.com/en/forum/477915#comment_56889779):**

Hi. I am playing around with [CatBoost](https://www.mql5.com/en/articles/8642 "Article: Gradient boosting (CatBoost) in the problems of building trading systems. Naive approach") and have got to the point where a strategy trained on (all of) 2024 data will produce >300% returns when backtested (in MetaTrader) on 2024 but performs poorly on other years. Does anyone have experience with this? It intuitively feels like overfitting, but even if I train with much lower iterations (like 1k) I get the same result.

I am training with ~40 - 50 features, on minute data, so something like 250,000 rows per year. The .cbm file size tends to come out at 1000x the number of iterations (e.g. 1000 iterations = 1MB, 10,000 iterations = 10MB, and so on). Backtesting on Metatrader limits me to about 100,000MB before the backtester grinds to a halt. I can backtest with C++ to an arbitrarily high size but my returns in metatrader vs C++ are wildely different.

Hello there. First of all, Metatrader backtester takes into account of spreads and commission, which may explain why it's different from your results in C++. Second of all, in my opinion, machine learning is essentially a process of overfitting. There are plenty of ways to reduce overfitting like ensemble, dropout, and feature engineering. But at the end of the day, in-sample is always way better than out-of-sample. The use of machine learning in predicting financial time series is an age old problem. If you're trying to predict return (I'm assuming cuz you're saying 250k rows), cuz noise is to be expected cuz you and other players have the same prediction objective. Whereas what I introduced in this article is a method of metalabeling where there's there's less noise cuz your prediction objective is narrowed to your own strategy, but it would have less samples to learn from, making complexity constraint even stricter. I would say lower your expectation with ML method and explore ways to reduce overfitting.

![johnboy85](https://c.mql5.com/avatar/avatar_na2.png)

**[johnboy85](https://www.mql5.com/en/users/johnboy85)**
\|
8 Jun 2025 at 11:29

Thanks for replying so quickly on a thread which is > 6 months old. Plenty of think about here. I am getting used to the enormous paramater space and trying to find ways to reduce overfitting.

Thanks again!

![Zhuo Kai Chen](https://c.mql5.com/avatar/2024/11/6743e84b-8a3d.jpg)

**[Zhuo Kai Chen](https://www.mql5.com/en/users/sicklemql)**
\|
8 Jun 2025 at 11:40

**johnboy85 [#](https://www.mql5.com/en/forum/477915#comment_56894486):**

Thanks for replying so quickly on a thread which is > 6 months old. Plenty of think about here. I am getting used to the enormous paramater space and trying to find ways to reduce overfitting.

Thanks again!

Good luck with your research!

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
4 Jul 2025 at 11:19

The hype on MO and the quality of the material is just depressing.


![MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://c.mql5.com/2/104/MQL5_Trading_Toolkit_Part_4____LOGO.png)[MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://www.mql5.com/en/articles/16528)

Learn how to retrieve, process, classify, sort, analyze, and manage closed positions, orders, and deal histories using MQL5 by creating an expansive History Management EX5 Library in a detailed step-by-step approach.

![Trading with the MQL5 Economic Calendar (Part 5): Enhancing the Dashboard with Responsive Controls and Filter Buttons](https://c.mql5.com/2/104/Trading_with_the_MQL5_Economic_Calendar_Part_5___LOGO.png)[Trading with the MQL5 Economic Calendar (Part 5): Enhancing the Dashboard with Responsive Controls and Filter Buttons](https://www.mql5.com/en/articles/16404)

In this article, we create buttons for currency pair filters, importance levels, time filters, and a cancel option to improve dashboard control. These buttons are programmed to respond dynamically to user actions, allowing seamless interaction. We also automate their behavior to reflect real-time changes on the dashboard. This enhances the overall functionality, mobility, and responsiveness of the panel.

![Creating a Trading Administrator Panel in MQL5 (Part VIII): Analytics Panel](https://c.mql5.com/2/104/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_VIII____LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part VIII): Analytics Panel](https://www.mql5.com/en/articles/16356)

Today, we delve into incorporating useful trading metrics within a specialized window integrated into the Admin Panel EA. This discussion focuses on the implementation of MQL5 to develop an Analytics Panel and highlights the value of the data it provides to trading administrators. The impact is largely educational, as valuable lessons are drawn from the development process, benefiting both upcoming and experienced developers. This feature demonstrates the limitless opportunities this development series offers in equipping trade managers with advanced software tools. Additionally, we'll explore the implementation of the PieChart and ChartCanvas classes as part of the continued expansion of the Trading Administrator panel’s capabilities.

![Reimagining Classic Strategies (Part 12): EURUSD Breakout Strategy](https://c.mql5.com/2/104/Reimagining_Classic_Strategies_Part_12___LOGO__1.png)[Reimagining Classic Strategies (Part 12): EURUSD Breakout Strategy](https://www.mql5.com/en/articles/16569)

Join us today as we challenge ourselves to build a profitable break-out trading strategy in MQL5. We selected the EURUSD pair and attempted to trade price breakouts on the hourly timeframe. Our system had difficulty distinguishing between false breakouts and the beginning of true trends. We layered our system with filters intended to minimize our losses whilst increasing our gains. In the end, we successfully made our system profitable and less prone to false breakouts.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/16487&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6429447454119099498)

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