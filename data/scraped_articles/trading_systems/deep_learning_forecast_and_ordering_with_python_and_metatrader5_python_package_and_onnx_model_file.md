---
title: Deep Learning Forecast and ordering with Python and MetaTrader5 python package and ONNX model file
url: https://www.mql5.com/en/articles/13975
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:43:13.720798
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/13975&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062666763201586847)

MetaTrader 5 / Trading systems


### Introduction

Deep learning is a subfield of machine learning that focuses on artificial neural networks, inspired by the structure and function of the human brain. It involves training models to perform tasks without explicit programming but by learning patterns and representations from data. Deep learning has gained significant attention due to its ability to automatically learn hierarchical features and representations, making it effective in various domains such as image and speech recognition, natural language processing, and more.

Key Concepts in Deep Learning:

- **Neural Networks:** Basic units of deep learning, consisting of interconnected nodes or neurons organized into layers.
- **Deep Neural Networks (DNNs):** Neural networks with multiple layers, allowing them to learn complex patterns.
- **Training:** The process of adjusting model parameters using labeled data to minimize the difference between predicted and actual outcomes.
- **Activation Functions:** Functions applied to neurons' output to introduce non-linearity, enabling the model to learn more complex relationships.
- **Backpropagation:** An optimization algorithm used to update the model's weights based on the error in its predictions.

**Python:** Python is a high-level, versatile programming language known for its readability and ease of use. It has a vast ecosystem of libraries and frameworks that make it suitable for various applications, including web development, data science, artificial intelligence, and more.

Key Features of Python:

- **Readability:** Python's syntax is designed to be readable and clean, making it easy for developers to express concepts in fewer lines of code.
- **Extensive Libraries:** Python has a rich set of libraries and frameworks for diverse applications, such as NumPy and pandas for data manipulation, TensorFlow and PyTorch for deep learning, Django and Flask for web development, and more.
- **Community Support:** Python has a large and active community of developers, contributing to its continuous improvement and the availability of numerous resources for learning.

**Using Python for Deep Learning:** Python is a popular language for deep learning due to its extensive libraries and frameworks. Two major libraries for deep learning in Python are TensorFlow and PyTorch. These libraries provide high-level abstractions for building and training neural networks, making it accessible for both beginners and experienced practitioners.

In summary, deep learning is a powerful approach within the broader field of machine learning, and Python serves as an excellent programming language for implementing deep learning models and solutions. The combination of Python and deep learning has led to advancements in various fields, making complex tasks more accessible and automatable.

**Ordering with Python and MQL5 Package:** It appears you are interested in combining Python with the MQL5 package for handling trading orders. MQL5 is a specialized language designed for algorithmic trading in the MetaTrader 5 (MT5) platform. Here's a general outline of how you might approach integrating Python with MQL5 for trading:

- **Python Script for Deep Learning:** Develop a Python script using a deep learning library (e.g., TensorFlow, PyTorch) to create and train a model for your specific trading strategy. This script will handle the analysis and decision-making based on your deep learning model.

- **Communication between Python and MQL5:** Establish a communication bridge between your Python script and the MetaTrader 5 platform. This can be done using various methods, such as sockets, REST APIs, or other communication protocols. The goal is to enable your Python script to send trading signals or orders to the MetaTrader terminal.

- **Execution of Orders in MQL5:** In your MQL5 script or expert advisor, implement the logic to receive signals from the Python script and execute corresponding buy/sell orders in the MetaTrader 5 platform. This involves utilizing the trading functions provided by the MQL5 language.


### ONNX

or Open Neural Network Exchange, is an open-source format designed to facilitate the interoperability of artificial intelligence (AI) models across various frameworks. Developed and supported by Microsoft and Facebook, ONNX allows users to train models using one framework and deploy them using another.

The key advantage of ONNX is its ability to serve as an intermediary format between different deep learning frameworks such as TensorFlow, PyTorch, and others. This interchangeability simplifies the process of integrating machine learning models into different applications, environments, and platforms.

In the context of MQL5 (MetaQuotes Language 5), the integration of ONNX models involves converting a trained machine learning model to the ONNX format. Once in the ONNX format, the model can be loaded into MQL5 scripts or Expert Advisors, enabling the utilization of advanced machine learning capabilities for algorithmic trading strategies within the MetaTrader 5 platform.

### Brief introduction to what will be shown.

In the upcoming article, we will delve into the fascinating world of deep learning applied to automated trading. The focus will be on a program designed for auto trades, utilizing advanced deep learning techniques. We will explore the intricacies of testing the model's performance using key metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2). This exploration aims to provide insights into the effectiveness of the deep learning model in the context of algorithmic trading, shedding light on its predictive accuracy and overall robustness. Join us on this journey as we unravel the intersection of deep learning, automated trading, and comprehensive performance testing.

Additionally, we will explore the process of converting the Python script (.py) into an executable, ensuring a comprehensive and user-friendly package. This step-by-step guide will provide detailed insights, leaving no stone unturned, to ensure a thorough understanding of the entire process. By the end of the article, you'll not only have a deep understanding of implementing deep learning in automated trading but also the practical knowledge of wrapping up your solution into a convenient executable format. Stay tuned for an in-depth walkthrough to seamlessly navigate from coding to a fully functional, executable program, making your experience both educational and practical.

As a concluding step in the article, I will be creating a script to generate output in ONNX format. This script will be instrumental in incorporating the ONNX model into MetaTrader 5 (MT5) using an Expert Advisor (EA). This process enhances the adaptability and integration of machine learning capabilities into the MT5 platform, facilitating the development of more sophisticated and efficient algorithmic trading strategies.

### Before starting to code

What we aim to achieve is the development of a Python bot that undergoes a two-step process: first, it engages in deep learning analysis of the available data, and subsequently, it executes trading orders. Given that MT5 facilitates the download of an extensive amount of tick data, our strategy involves acquiring this data and converting it into price information. The conversion methodology will entail taking the average between tick values, allowing us to create a more manageable representation of the market prices for further analysis and decision-making within our bot.

But before we proceed, we need to utilize Python, and for a convenient Python setup, the recommended approach is to download and install Anaconda. Additionally, installing Visual Studio Code will provide a user-friendly environment for scripting.

[![Anaconda](https://c.mql5.com/2/64/anaconda.png)](https://www.mql5.com/go?link=https://www.anaconda.com/ "https://www.anaconda.com/")

[![VSC](https://c.mql5.com/2/64/vsc.png)](https://www.mql5.com/go?link=https://code.visualstudio.com/ "https://code.visualstudio.com/")

Once you have Anaconda and Visual Studio Code installed, the next step is to install some packages using pip within Visual Studio Code.

You can do this by opening the integrated terminal in Visual Studio Code and using the following commands (If you are using Conda/Anaconda, run these commands in Anaconda Prompt)  :

```
pip install [package1] [package2] ...
```

Replace \[package1\] , \[package2\] , etc., with the names of the specific Python packages you need for your project. This will ensure that your environment is equipped with the necessary tools and libraries to proceed with the development of your Python bot.

### Packages

You need to install specific packages for a Python environment, including MetaTrader 5, TensorFlow, and other libraries that come with Anaconda. Here's a general guide on how to install these libraries:

1. **MetaTrader 5:** MetaTrader 5 (MT5) is a trading platform. Unfortunately, as of my last knowledge update in January 2022, MetaTrader 5 doesn't have a direct Python API available in the official distribution. However, some third-party libraries or wrappers may exist. Please check the official MetaTrader 5 documentation or relevant community forums for the latest information.


```
pip install MetaTrader5
```

2. **TensorFlow:** TensorFlow is a popular machine learning library. You can install it using the following pip command in your terminal or command prompt:


```
pip install tensorflow
```


### Lets start coding

Once you have successfully installed the required libraries, the next step is to begin writing your Python code. To do this, you need to import the libraries that you installed. In Python, you use the import statement to bring in the functionality of a library into your code.

```
import MetaTrader5 as mt5
from MetaTrader5 import *
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
```

```
from MetaTrader5 import *
```

This line imports all functions and classes from the MetaTrader5 library. The use of \* (wildcard) means that you're importing everything, though this practice is generally discouraged due to potential namespace conflicts.

```
import numpy as np
```

This line imports the NumPy library and gives it the alias np . NumPy is widely used for numerical operations in Python.

```
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

These lines import specific functions/classes from the scikit-learn library. It includes utilities for data preprocessing, model evaluation, and model selection.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
```

These lines import components from the TensorFlow library, which is commonly used for deep learning. It brings in the Sequential model, Dense layers, and L2 regularization.

```
from sklearn.model_selection import KFold
```

This line imports the KFold class from scikit-learn, which is often used for cross-validation in machine learning.

In summary, your script is preparing to use the MetaTrader5 library for financial data, NumPy for numerical operations, scikit-learn for machine learning utilities, and TensorFlow for deep learning. It seems like you're setting up an environment for financial data analysis and possibly building a machine learning model for financial predictions. Ensure that you have these libraries installed in your Python environment using the appropriate package manager (e.g., pip or conda)

### Downloading Data

To download ticks, first, we need to connect to the data provider (mt5), and from there, we can download what we need. For that purpose, we will use the following:

```
# You will need to update the values for path, login, pass, and server according to your specific case.
creds = {
    "path": "C:/Program Files/XXX MT5/terminal64.exe",
    "login": account,
    "pass": clave_secreta,
    "server": server_account,
    "timeout": 60000,
    "portable": False
}
# We launch the MT5 platform and connect to the server with our username and password.
if mt5.initialize(path=creds['path'],
                  login=creds['login'],
                  password=creds['pass'],
                  server=creds['server'],
                  timeout=creds['timeout'],
                  portable=creds['portable']):

    print("Plataform MT5 launched correctly")
else:
    print(f"There has been a problem with initialization: {mt5.last_error()}")
```

Once this is done, we proceed to download the data of interest for the symbol we are interested in, in the form of ticks.

```
rates = rates.drop(index=rates.index)
rates = mt5.copy_ticks_from(symbol, utc_from, 1000000000, mt5.COPY_TICKS_ALL)
```

This code first clears any existing data in the 'rates' DataFrame and then fetches ticks data for a particular symbol and time range using the copy\_ticks\_from function from the MetaTrader 5 library.

As we aim to have the data in a Pandas DataFrame (which is commonly used along with NumPy) for better manipulation in Python, we will transfer the data to a Pandas DataFrame.

```
rates_frame=pd.DataFrame()
# Empty DataFrame
rates_frame = rates_frame.drop(index=rates_frame.index)
rates_frame = pd.DataFrame(rates)
```

This code initializes an empty Pandas DataFrame named rates\_frame , clears any existing data in it, and then fills it with the data from the rates variable.

To simplify things, let's add and divide the bid and ask values in the data by two. This way, we obtain the average value, which will be our input for deep learning.

```
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
rates_frame['close']=(rates_frame['ask']+rates_frame['bid'])/2
```

This code converts the 'time' column to datetime format using seconds as the unit and calculates the average of 'ask' and 'bid' values, assigning the result to a new column named 'close' in the rates\_frame DataFrame.

The choice of input data for deep learning in financial applications, such as predicting price movements, depends on various factors, and there isn't a one-size-fits-all solution. However, there are some considerations and general guidelines to keep in mind:

### Raw Data vs. Derived Features (Oscillators and Indicators):

#### 1\. **Raw Data:**

- **Advantages:**
  - Retains all available information without additional assumptions.
  - Allows the neural network to learn patterns directly from the raw input.
- **Considerations:**
  - Might contain noise or irrelevant information.
  - More data preprocessing might be needed.

#### 2\. **Derived Features (Oscillators and Indicators):**

- **Advantages:**
  - Feature engineering can capture specific market dynamics.
  - May provide a more structured representation of data.
- **Considerations:**
  - Introduces assumptions about what features are relevant.
  - The effectiveness depends on the chosen indicators and their parameters.

### Best Practices and Considerations:

#### 1\. **Data Normalization:**

- Scale the data to a consistent range (e.g., between 0 and 1) to improve convergence during training.

#### 2\. **Sequence Length:**

- For time series data, the sequence length matters. Experiment with different lengths to find the optimal trade-off between capturing relevant patterns and computational efficiency.

#### 3\. **Temporal Aspects:**

- Consider incorporating temporal aspects, such as lagged values or sliding windows, to capture time dependencies.

#### 4\. **Feature Selection:**

- Experiment with different subsets of features to identify which are most informative for your specific task.

#### 5\. **Model Architecture:**

- Adjust the neural network architecture based on the nature of your input data. Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs) are effective for sequential data.

#### 6\. **Regularization:**

- Use regularization techniques (e.g., dropout) to prevent overfitting, especially when dealing with high-dimensional data.

#### 7\. **Hyperparameter Tuning:**

- Experiment with different hyperparameter settings to find the optimal configuration for your model.

In practice, it's often beneficial to compare different approaches empirically. Some tasks may benefit from the simplicity of raw data, while others may gain performance from well-crafted derived features. It's crucial to strike a balance between providing enough information for the model to learn and avoiding noise or irrelevant details. Additionally, regularly reassess the chosen approach as market dynamics may change over time.

### Shifting data

We now have the data in a Pandas DataFrame, and we can send it for deep learning to process as input. There are already many interesting articles on deep learning in MQL5, so I won't delve into this much. I'll go straight to the practical aspect in Python. But before sending the data for TensorFlow to work its magic, we need to have an input for the time shift that we will apply to the data for prediction (the shifted time will be the time we aim to predict). That's why we have the following code.

Shifting a DataFrame in the context of deep learning, particularly in time series forecasting, is commonly done to create sequences of input and target variables. Here are the reasons why shifting is used in the context of deep learning for time series prediction:

1. **Temporal Dependencies:** Deep learning models, such as recurrent neural networks (RNNs) or long short-term memory networks (LSTMs), can capture temporal dependencies in sequential data. Shifting the DataFrame allows you to create sequences where each input sequence corresponds to a segment of past data, and the corresponding target sequence represents the future data.

2. **Sequential Learning:** Deep learning models are effective at learning patterns and dependencies in sequential data. By shifting the DataFrame, you ensure that the input and target sequences align temporally, allowing the model to learn from the historical context and make predictions based on that context.

3. **Training Input-Output Pairs:** Shifting helps in creating training examples for the deep learning model. Each row in the shifted DataFrame can be considered an input-output pair, where the input is a sequence of past observations, and the output is the target variable to be predicted in the future.


```
number_of_rows= seconds
empty_rows = pd.DataFrame(np.nan, index=range(number_of_rows), columns=df.columns)
df = df._append(empty_rows, ignore_index=True)
df['target'] = df['close'].shift(-seconds)
print("df modified",df)
```

- number\_of\_rows is a variable representing the number of seconds.
- empty\_rows creates a new DataFrame with NaN values, having the same columns as the original DataFrame ( df ).
- df.append(empty\_rows, ignore\_index=True) appends the empty rows to the original DataFrame ( df ) while ignoring the index to ensure a continuous index.
- df\['target'\] = df\['close'\].shift(-seconds) creates a new column 'target' containing the 'close' values shifted by a negative value of the specified number of seconds. This is commonly done when preparing time series data for predictive modeling.

Now, all that's left is to clean the data so that we can use it as input with TensorFlow.

```
# Drop NaN values
df=df.dropna()
```

The result is a modified DataFrame ( df ) with additional rows filled with NaN values and a new 'target' column for time-shifted 'close' values.

### Deep Learning

Using TensorFlow and Keras to build and train a neural network for time series prediction.

```
# Split the data into features (X) and target variable (y)
X=[]
y=[]
X = df2[['close']]
y = df2['target']

# Split the data into training and testing sets
X_train=[]
X_test=[]
y_train=[]
y_test=[]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the features
X_train_scaled=[]
X_test_scaled=[]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a neural network model
model=None
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(k_reg)))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(1, activation='linear'))

# Compile the model[]
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=int(epoch), batch_size=256, validation_split=0.2, verbose=1)

# Use the model to predict the next 4 instances
X_predict=[]
X_predict_scaled=[]

predictions = pd.DataFrame()
predictions=[]
# Empty DataFrame
#predictions = predictions.drop(index=predictions.index)
X_predict = df2.tail(segundos)[['close']]
X_predict_scaled = scaler.transform(X_predict)
predictions = model.predict(X_predict_scaled)

# Print actual and predicted values for the next
    n  instances
print("Actual Value for the Last Instances:")
print(df.tail(1)['close'].values)

print("\nPredicted Value for the Next Instances:")
print(predictions[:, 0])
predictions=pd.DataFrame(predictions)
```

Preparing and training a neural network using historical financial data to predict future values. It uses the TensorFlow and Keras libraries for building and training the model. The predictions are then printed for the next instances based on the trained model.

Explanation:

- The model is initialized as a sequential model, meaning it's a linear stack of layers.

- The Dense layers represent fully connected layers in the neural network. The parameters include the number of neurons (units), activation function, input shape (applicable only for the first layer), and kernel regularization (in this case, L2 regularization with a strength specified by k\_reg ).

- activation='relu' implies the Rectified Linear Unit (ReLU) activation function, commonly used in hidden layers.

- The last layer has one neuron with a linear activation function, indicating a regression output. If this were a classification task, a different activation function like 'sigmoid' or 'softmax' would be used.


This architecture is a feedforward neural network with multiple hidden layers for regression purposes. Adjust the parameters and layers based on your specific task and dataset characteristics.

### Prediction

How good is the aproximation of this model?

```
# Calculate and print mean squared error
mse = mean_squared_error(y_test, model.predict(X_test_scaled))
print(f"\nMean Squared Error: {mse}")

# Calculate and print mean absolute error
mae = mean_absolute_error(y_test, model.predict(X_test_scaled))
print(f"\nMean Absolute Error: {mae}")

# Calculate and print R2 Score
r2 = r2_score(y_test, model.predict(X_test_scaled))
print(f"\nR2 Score: {r2}")
```

The provided code calculates and prints three evaluation metrics for a trained regression model.

Explanation:

- **Mean Squared Error (MSE):** It measures the average squared difference between the predicted and actual values. The lower the MSE, the better the model.

- **Mean Absolute Error (MAE):** It measures the average absolute difference between the predicted and actual values. Like MSE, lower values indicate better model performance.

- **R2 Score:** Also known as the coefficient of determination, it measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates a perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the target variable. Negative values indicate poor model performance.


These metrics provide insights into how well the trained model is performing on the test set.

In this context, you need to experiment with the amount of data (input delay) and the number of epochs. For instance, I use 900 days of data and 1 epoch for EURUSD for a 2 hours time period (I will explain better this after). However, I also achieve good results with significantly less data and more epochs.

### Orders

Now that we have this, we just need to measure the time it takes for the script to return the data. This allows us to mark on the chart the starting point of the prediction for us (as the prediction begins some time earlier, for example, in my case, it takes around 15-20 minutes). The starting point for us to open orders will be when that time has passed. Additionally, after extensively using this bot, I've noticed that the most accurate predictions are in the first quarter of the entire shift. To enhance accuracy, I'll limit and mark on the chart and execute orders during that specific period. This is easily implemented using \`time.now()\`. We can capture the time at that moment and convert it to seconds since our prediction and the x-axis scale are both in seconds. Given all this, we'll mark where our prediction begins, and orders will be triggered based on that initial point.

This is a typical Python order using MetaTrader 5 (MT5). The function \`open\_trade\_sell2\` represents a common structure for placing a sell order in the MT5 platform. It leverages the \`mt5.order\_send\` method to send a trading request to execute the order. The parameters include the trading action ('buy' or 'sell'), symbol, lot size, a random integer (possibly for magic number identification), take profit (tp), stop loss (sl), and deviation.

```
def open_trade_sell2(action, symbol, lot,random_integer, tp, sl, deviation):
        '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
        '''
        # prepare the buy request structure
        symbol_info = get_info(symbol)

        if action == 'buy':
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        elif action =='sell':
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        point = mt5.symbol_info(symbol).point
        print("el precio mt5 es:", price)

        buy_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": trade_type,
            "price": price,
            "sl":sl,
            "tp":tp,
            "deviation": deviation,
            "magic": random_integer,
            "comment": "python open",
            "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # send a trading request
        result = mt5.order_send(buy_request)
        return result, buy_request
```

This function is part of a script that interacts with the MT5 platform to automate trading strategies. The key steps include determining the trade type, retrieving the current bid or ask price, preparing a trading request structure, and finally, sending the trading request.

### Closing orders

```
def close_position(position,symbol):
    """This function closes the position it receives as an argument."""

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'position': position.ticket,
        'magic': position.magic,
        'symbol': symbol,
        'volume': position.volume,
        'deviation': 50,
        'type': mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
        'type_filling': mt5.ORDER_FILLING_FOK,
        'type_time': mt5.ORDER_TIME_GTC,
        'comment': "mi primera orden desde Python"
    }
    return mt5.order_send(request)

# Now, we define a new function that serves to close ALL open positions.
def close_all_positions(symbol):
    """This function closes ALL open positions and handles potential errors."""

    positions = mt5.positions_get()
    for position in positions:
        if close_position(position,symbol).retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Position {position.ticket} closed correctly.")
        else:
            print(f"An error occurred while closing the position.{position.ticket}: {mt5.last_error()}")
```

We can also close the orders using the EA that I have modified for this occasion. In this case, the .py file creates a document and writes the time in minutes to close orders (if you are using this, remember to leave the file in the File folder of your mt5.

```
#Specify the path for the file and open it in write mode
file_path = "C:/XXX/MetaQuotes/Terminal/XXX/MQL5/Files/python_"+symbol+"_file.txt"
try:
    with open(file_path, "w") as file:
        # Write file parameters
        file.write(str(the_value))
    print(f"File '{file_path}' created.")
except Exception as e:
    print(f"Error creating file: {e}")
```

And this is the EA modified, that I obtained from this [guy](https://www.mql5.com/ru/users/barabashkakvn) (you will have to add the correct paths to the File folder):

```
//+------------------------------------------------------------------+
//|                                               Timer_modified.mq5 |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2021, Vladimir Karputov"
#property link      "https://www.mql5.com/ru/market/product/43516"
#property version   "1.000"

//--- display the window of input parameters when launching the script
#property script_show_inputs

//--- parameters for data reading
input string InpFileName="python_file.txt"; // file name
input string InpDirectoryName="Files"; // directory name
//---
#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>
//#include <File.mqh>
//---
CPositionInfo  m_position;                   // object of CPositionInfo class
CTrade         m_trade;                      // object of CTrade class
//--- input parameters
input uchar    InpAfterHour   = 0; // After: Hour ... (max 255)
input uchar    InpAfterMinutes= 42; // After: Minutes ... (max 255)
input uchar    InpAfterSeconds= 59; // After: Seconds ... (max 255)
//---
int file_handles;
uchar value_uchar;
long     m_after  = 0;
bool         started=false;     // flag of counter relevance
string str1,str2,str3,str4;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

int OnInit()
  {
  str1="C:/XXX/MetaQuotes/Terminal/XXXX/MQL5/Files/python_";
  str3="_file.txt";
  str2=Symbol();
  str4=str1+str2+str3;

   // Open file to read
   string file_path = str4;
   file_handles = FileOpen(file_path, FILE_READ|FILE_TXT);

   if(file_handles != INVALID_HANDLE)
     {
      PrintFormat("File %s opened correctly", file_path);

      // Read uchar value of the file
      if (FileReadInteger(file_handles, INT_VALUE))
        {
         PrintFormat("Uchar value read from file: %u", value_uchar);
         m_after  = 0;
         m_after=value_uchar;
        }
      else
        {
         Print("Error when reading the uchar value from file");
        }

      // Close file after read
      FileClose(file_handles);
     }
   else
     {
      m_after  = 0;
      m_after=InpAfterHour*60*60+InpAfterMinutes*60+InpAfterSeconds;
      PrintFormat("Error when opening file %s, error code = %d", file_path, GetLastError());
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   for(int i=PositionsTotal()-1; i>=0; i--) // returns the number of current positions
      if(m_position.SelectByIndex(i)) // selects the position by index for further access to its properties
        {
         if(TimeCurrent()-m_position.Time()>=m_after)
            m_trade.PositionClose(m_position.Ticket()); // close a position
        }
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| called when a Trade event arrives                                |
//+------------------------------------------------------------------+
void OnTrade()
  {
   if(started) SimpleTradeProcessor();
  }
 void SimpleTradeProcessor()
  {
  str1="C:/Users/XXX/MetaQuotes/Terminal/XXX/MQL5/Files/python_";
  str3="_file.txt";
  str2=Symbol();
  str4=str1+str2+str3;
     // Open file to read it
   string file_path = str4;
   file_handles = FileOpen(file_path, FILE_READ|FILE_TXT);

   if(file_handles != INVALID_HANDLE)
     {
      PrintFormat("File %s is opened correctly", file_path);

      // Reads the uchar value from the file
      if (FileReadInteger(file_handles, INT_VALUE))
        {
         PrintFormat("value_uchar read form file: %u", value_uchar);
         m_after  = 0;
         m_after=value_uchar;
        }
      else
        {
         Print("Error while reading the value uchar from file");
        }

      // Closes the file after reading
      FileClose(file_handles);
     }
   else
     {
      m_after  = 0;
      m_after=InpAfterHour*60*60+InpAfterMinutes*60+InpAfterSeconds;
      PrintFormat("Error when opening the fileo %s, Code error = %d", file_path, GetLastError());
     }
  }
```

I have presented two different ways of closing orders. In Python, the only thing I do is use \`time.sleep(t)\`, where \`t\` is the number of seconds between when the order was placed and when I want to close it. However, this approach is not very practical as it involves waiting, consuming time that could be used for other script operations. I leave it up to you to decide which method to use for closing orders.

We can generate more orders by reducing the use of 'time.sleep(),' for instance, between placing a buy or sell order and closing it. How? We can utilize the modified Expert Advisor (EA) by incorporating the magic number value, allowing it to close only the order with the specified magic number and symbol. This way, instead of the script being paused with sleep, it can remain active and continue working.

### Visualization

To visualize it effectively, we will plot a graph to observe the trends. We will mark the ranges of the most reliable predictions and indicate the maximum and minimum values within those ranges and at the end, the graph will be saved.

```
plt.axvline(x=pt_division2, color='gray', linestyle='--', label='75 %')
plt.axvline(x=center, color='grey', linestyle='--', label='50 %')
plt.axvline(x=pt_division3, color='blue', linestyle='--', label='33 %')
plt.axvline(x=pt_division1, color='gray', linestyle='--', label='25 %')
plt.axvline(x=pt_division6, color='blue', linestyle='--', label='16 %')
plt.axvline(x=pt_division10, color='yellow', linestyle='--', label='10 %')
plt.axvline(x=pt_division14, color='blue', linestyle='--', label='7 %')
plt.axvline(x=pt_division20, color='yellow', linestyle='--', label='5 %')
plt.axvline(x=entry_point, color='orange', linestyle='--', label='entrada')
plt.axvline(x=value_25, color='orange', linestyle='--', label='salida 20%')
plt.axvline(x=maximal_index, color='red', linestyle='--', label='maximal') ##### ni idea de porqué no pinta correctamente la linea
plt.axvline(x=manimal_index, color='red', linestyle='--', label='minimal')# lo mismo aquí

plt.plot(dff5.iloc[:, 0], linestyle='-', label='Predicted')

plt.xlabel('Instances')
plt.ylabel('Prediction Price')
plt.legend()
plt.title(f'Predicted {symbol} y quedan en minutos: ' + str(MyParameter))
plt.savefig('Predicted_for_'+str(symbol)+'_quedan_'+str(MyParameter)+'_minutos_desde_'+str(formatted_now2)+'_.png')
```

### How does the script work?

Now, how does this script work? Firstly, it loads the libraries, then it notes the exact moment to track the current stage of the prediction. After that, it initializes MT5, downloads the data, processes it, and provides the results. We use these results to generate a graph and execute orders at precise moments, including the specific timing for order closures. Additionally, you have the flexibility to choose the method for closing orders.

If you wish to repeat this process for the same currency pair, a simple loop will suffice, making it easy to run the program continuously. There's just one catch – how do you determine the optimal inputs? Well, it's straightforward. A model with an R2 close to one, and MSE and MAE close to zero, indicates good approximations. However, be cautious of overfitting. Once you've identified the optimal input values, you can disregard MSE, MAE, and R2 since their calculations take a significant amount of time, and in this context, what matters most is the speed in obtaining results.

### Making an executable

Since when we start with a currency pair, the first thing we need to know before placing orders is whether we are making accurate predictions. Given that Python is continually evolving, and you might find the need to update libraries, it wouldn't hurt to wrap everything up. This way, you have a consolidated package ready for testing and experimentation.

For this, we will use Tkinter and convert it into an executable using auto-py-to-exe. This way, we won't face further issues in determining the input data, and we can exercise more caution with the .py file responsible for executing orders.

I will demonstrate how to create an executable for testing purposes, leaving the one for order execution for you to customize.

For the executable, I will use Tkinter, which provides a graphical interface to input the values. Remember that the file path should be written with either "/" or "\\". The interface will fetch the accuracy values of the model and display the graph.

Here's the code for the gui (gui\_console)

```
import tkinter as tk
from program_object import execute_program_with
from program_object_without import execute_program_without
import sys

class Aplication:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepLearning Forecast")

        # labels and entries
        label1 = tk.Label(root, text="Input account mt5:")
        label1.pack(pady=5)

        self.value1_entry = tk.Entry(root)
        self.value1_entry.pack(pady=5)

        label2 = tk.Label(root, text="Input password:")
        label2.pack(pady=5)

        self.value2_entry = tk.Entry(root)
        self.value2_entry.pack(pady=5)

        label3 = tk.Label(root, text="Input server of mt5:")
        label3.pack(pady=5)

        self.value3_entry = tk.Entry(root)
        self.value3_entry.pack(pady=5)

        label4 = tk.Label(root, text="Input delay in days (for ticks):")
        label4.pack(pady=5)

        self.value4_entry = tk.Entry(root)
        self.value4_entry.pack(pady=5)

        label5 = tk.Label(root, text="Input timeframe 1 (1h),2,4,1d,1w:")
        label5.pack(pady=5)

        self.value5_entry = tk.Entry(root)
        self.value5_entry.pack(pady=5)

        label6 = tk.Label(root, text="Input epochs:")
        label6.pack(pady=5)

        self.value6_entry = tk.Entry(root)
        self.value6_entry.pack(pady=5)

        label7 = tk.Label(root, text="Input symbol:")
        label7.pack(pady=5)

        self.value7_entry = tk.Entry(root)
        self.value7_entry.pack(pady=5)

        label8 = tk.Label(root, text="Input path for mt5:")
        label8.pack(pady=5)

        self.value8_entry = tk.Entry(root)
        self.value8_entry.pack(pady=5)

        # Radio button to select program to execute
        self.opcion_var = tk.StringVar(value="program_object")
        radio_btn_object = tk.Radiobutton(root, text="With r2 Score, MAE & MSE", variable=self.opcion_var, value="program_object")
        radio_btn_object.pack(pady=5)
        radio_btn_object_without = tk.Radiobutton(root, text="Without", variable=self.opcion_var, value="program_object_without")
        radio_btn_object_without.pack(pady=5)

        # Botón start
        boton_execute = tk.Button(root, text="Run Program", command=self.execute_programa)
        boton_execute.pack(pady=10)

        # Botón close
        boton_quit = tk.Button(root, text="Exit", command=root.destroy)
        boton_quit.pack(pady=10)

    def write(self, text):
        # this method y called when sys.stdout.write is used
        self.salida_text.insert(tk.END, text)
    def flush(self):
        pass
    def execute_program(self):
        # Obteined value of the selected option
        selected_program = self.opcion_var.get()

        # Obteined value of inputs
        value1 = self.value1_entry.get()
        value2 = self.value2_entry.get()
        value3 = self.value3_entry.get()
        value4 = self.value4_entry.get()
        value5 = self.value5_entry.get()
        value6 = self.value6_entry.get()
        value7 = self.value7_entry.get()
        value8 = self.value8_entry.get()

                # Redirects stdout & stderr to the console
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Calls the function to execute a selected program and pass values as arguments
        if selected_program == "program_object":
            execute_program_with(value1, value2, value3, value4, value5, value6, value7, value8)
        elif selected_program == "program_object_without":
            execute_program_without(value1, value2, value3, value4, value5, value6, value7, value8)
        # Restores stdout to a predeterminate value to no have conflicts
        sys.stdout = self
        sys.stderr = self

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplication(root)
    root.mainloop()
```

and this is the code that's called (where you have to paste your code):

```
import sys
import MetaTrader5 as mt5
from MetaTrader5 import *
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for residual plot

def main(value1, value2, value3, value4, value5, value6, value7,value8):
    # Now you can use variables value1, value2, value3, value4, value5, value6, value7 en tu programa
    print("Numero de cuenta mt5 en mt5: ", value1)
    print("Password en mt5: ", value2)
    print("Servidor en mt5: ", value3)
    print("Delay dias ticks: ", value4)
    print("Time frame: ", value5)
    print("Epochs: ", value6)
    print("Symbol: ", value7)
    print("Path a mt5: ", value8)


def execute_program_with(value1, value2, value3, value4, value5, value6, value7, value8):
    main(value1, value2, value3, value4, value5, value6, value7, value8)
```

and it ends with this

```
if __name__ == "__main__":
    value1, value2, value3, value4, value5, value6, value7 = sys.argv[1:]
    main(value1, value2, value3, value4, value5, value6, value7)
```

there are two of this, one with the model predictions scores, and the graph, and the other one just shows the graph.

To convert this to an executable, we'll use auto-py-to-exe. First, install the library.

```
pip install auto-py-to-exe
```

and we make it run, just by typing in the terminal auto-py-to-exe

Just remember to add to the script location gui\_consola, and add files programa\_objetivo.py and programa\_objetivo\_sin.py, hit convert py to exe, and you will have it.

![auto-py-to-exe](https://c.mql5.com/2/64/autopytoexe.png)

You might have to use conda environment, to do this just write

```
conda activate XXX
```

being XXX the name of the environment.

You might encounter issues with Conda. To address this, consider uninstalling Anaconda from the Control Panel (navigate to 'Uninstall a program') and installing Python directly from its official website. You can download it [here](https://www.mql5.com/go?link=https://www.python.org/downloads/release/python-3115/ "https://www.python.org/downloads/release/python-3115/") – make sure to select your operating system.

While installing Python, remember to add the path to the system variables. Additionally, disable length limits.

Additionally, disable length limits. To complete the setup, install matplotlib, seaborn and sklearn by running the following command in your terminal or command prompt:

```
pip install matplotlib
```

```
pip install seaborn
```

```
pip install scikit-learn
```

GUI

The term "Tk GUI" refers to the graphical user interface (GUI) created using the Tkinter toolkit in Python. Tkinter is a built-in module in the Python standard library that facilitates the development of graphical user interfaces.

And it looks like this:

![GUI](https://c.mql5.com/2/64/gui.png)

Remember to add the path for mt5, and change "\\" for "/" as here:

C:/Program Files/XXX MT5/terminal64.exe

GUI

This is more or less what you will see in the GUI when you wrap it into an executable and run it.

![001](https://c.mql5.com/2/64/more_or_less_what_to_see_when_running.png)

The results for the tester look like this:

![tester001](https://c.mql5.com/2/64/tester.png)

This is how the tester looks like, and the values inside the red rectangle, are the important values, obviously those values are not good, and I should look for a r2 score near 1 and a MSE and MAE near to 0, I should try to use more data and/or more epochs, and also change in the script the value for L2 (I promise you that you can get a good fitting and have time to make orders ... its like a game.

Metrics Used:

R2 (R-squared), MSE (Mean Squared Error), and MAE (Mean Absolute Error) are commonly used metrics in statistics and machine learning to evaluate the performance of models:

1. **R2 (R-squared):**

   - R2 is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
   - It is a scale from 0 to 1, where 0 indicates that the model does not explain the variability of the dependent variable, and 1 indicates perfect explanation.
2. **MSE (Mean Squared Error):**

   - MSE is a metric that calculates the average squared difference between the actual and predicted values.
   - It penalizes larger errors more heavily than smaller ones, making it sensitive to outliers.
   - Mathematically, it is the sum of the squared differences divided by the number of observations.
3. **MAE (Mean Absolute Error):**

   - MAE is a metric that calculates the average absolute difference between the actual and predicted values.
   - Unlike MSE, it does not square the differences, making it less sensitive to outliers.
   - It is the sum of the absolute differences divided by the number of observations.

These metrics are commonly used in regression analysis to assess how well a model is able to predict the outcome variable. In general, a higher R2 value and lower MSE or MAE values indicate a better-performing model.

### Overfitting

Overfitting in deep learning for trading involves training a model that performs well on historical data but fails to generalize effectively to new, unseen data. Here are some dangers associated with overfitting in the context of deep learning for trading:

1. Limited Generalization: An overfitted model may capture noise or specific patterns in the historical data that do not represent true market dynamics. As a result, the model may struggle to generalize to new market conditions or unseen data.
2. Poor Performance on Live Data: Since an overfitted model is essentially memorizing the historical data, it may perform poorly when exposed to live market data. The model might make inaccurate predictions or fail to adapt to changing market trends.
3. Market Regime Changes: Financial markets are dynamic, and different market regimes (bull, bear, sideways) may exhibit unique characteristics. An overfitted model trained on a specific market regime may fail when the market undergoes a regime change, as it lacks adaptability.
4. False Signals: Overfitting can lead to the model capturing noise or outliers in the historical data, resulting in false signals. These false signals can misguide trading strategies, leading to financial losses in live trading scenarios.
5. Data Snooping Bias: Overfitting can be exacerbated by data snooping or data leakage. If the model is inadvertently exposed to information from the test set during training, it may learn patterns that do not generalize well to truly unseen data.
6. Hyperparameter Sensitivity: Overfitted models may be overly sensitive to hyperparameters, making them difficult to fine-tune for robust performance across different market conditions.

To mitigate these dangers, practitioners in deep learning for trading often employ strategies such as cross-validation, regularization techniques, and incorporating features that represent a broader understanding of market dynamics. Additionally, a cautious approach is necessary when interpreting model performance on historical data and validating its effectiveness on out-of-sample datasets. Regular monitoring and retraining of models are also essential to adapt to evolving market conditions.

### Cross Validation in Deep Learning:

Cross-validation is a statistical technique used to assess the performance of a machine learning model. In the context of deep learning, cross-validation involves dividing the dataset into multiple subsets or folds. The model is trained on some of these folds and tested on the remaining fold. This process is repeated multiple times, with different folds used for training and testing in each iteration. The performance metrics are then averaged over all iterations to provide a more robust evaluation of the model's performance.

Common types of cross-validation include k-fold cross-validation, where the dataset is divided into k folds, and each fold is used as a testing set exactly once, and leave-one-out cross-validation, where each data point is treated as a single fold.

Cross-validation helps in assessing how well a model generalizes to unseen data and reduces the risk of overfitting or underfitting.

### Kernel Regularizer (kernel\_regularizer=l2):

In the context of deep learning and neural networks, the kernel regularizer term refers to a regularization technique applied to the weights (or kernels) of the neural network layers. The \`kernel\_regularizer=l2\` parameter specifically indicates the use of L2 regularization on the weights.

L2 regularization adds a penalty term to the loss function that is proportional to the squared magnitude of the weights. The goal is to prevent any single weight from becoming too large and dominating the learning process. This helps in preventing overfitting by discouraging the model from fitting the training data too closely.

In summary, \`kernel\_regularizer=l2\` is a regularization technique used to control the complexity of a neural network by penalizing large weights, and it is one of the tools to address overfitting issues in deep learning models.

### Graph

This graph is saved where you run the script.

The number appears at the title is the time in minutes from the first orange line to the end of the graph.

You can change what you need from the graph (for example only show the range between the orange lines, with this line code:

```
plt.plot(dff5.iloc[punto_entrada, valor_25], linestyle='-', label='Predicted')
```

just read the code, and try to understand it.

![Graph](https://c.mql5.com/2/64/Predicted_for_BTCUSD_quedan_55_minutos_desde_2024_01_13_18_52_31_.png)

This is the chart it generates. The red lines indicate where buy/sell and close orders should be placed. The orange lines represent the range where the model is presumed to be more reliable and accurate, specifically in the first 20% interval. In this case, I used a 4-hour timeframe, so the optimal interval is the first hour, showing the maximum and minimum points to enter and exit. Once the script reaches the end, it loops back, and if we add the orders, it will keep running continuously. The rest of the marked lines help to position oneself on the chart when placing manual orders.

The price given in the predictions are only oriented, what's important is knowing where the min and max in the interval are

### How to use all this

What I do is:

First, I experiment with a symbol using the GUI, testing it with R2, MAE, and MSE. I iterate through different values of epochs and days of delay. Second, once I identify promising values for prediction, ensuring careful avoidance of overestimation, I transition to the script. I automate the script to execute orders autonomously, leasing a VPS for continuous order execution throughout the week. It is noteworthy that if high errors are encountered, there's also a parameter that can be adjusted. For instance, in the case of cryptocurrencies, one might opt for an L2 of 0.01 instead of the 0.001 used for forex. Through trial and error, optimal values for data quantity (days delay) and epochs can be determined.

It's important to consider that calculating MSE, R2, and MAE takes approximately one epoch in time. Therefore, it's advisable to conduct experiments initially. Once a good model approximation is achieved, the model can be employed without these parameters. In Python, this is achieved by either commenting out the relevant code or using the symbol #.

What I do with this Python program is use a loop. In case there is ever a failure when closing orders (I have implemented a mechanism to cancel the commands from the symbol), I also utilize the timer.mql5 since I have set an interval for the orders, specifically one hour. This ensures that no order remains open for more than an hour.

### Loop

To loop the DeepLearningForecast, instead of doint a while = True, we can do this (Loop.py):

```
import time
import subprocess

def execute_program():

    subprocess.run(['python', 'DeepLearningForecast.py'])

# Stablish and interval in seconds (for example, 60 seconds)
seconds_interval = 5

while True:
    execute_program()
    print(f"Program executed. Waiting {seconds_interval} seconds")

    # waits the interval before executing the program
    time.sleep(seconds_interval)
```

### Telegram bot

In MT5, we already have various ways to view information, but for this bot, I am sometimes interested in knowing if it is predicting accurately and executing orders correctly. To keep track of these actions and subsequently compare predictions with the actual chart, I send the chart and relevant information to a Telegram bot. I will now explain how to create the Telegram bot and how to send the data via Telegram.

step-by-step guide to create a bot on Telegram and obtain its chat ID and token to send messages through requests:

### Create a Bot on Telegram:

1. **Open Telegram and Search for BotFather:**

   - Open Telegram and search for "BotFather" in the search bar.
   - Start a chat with BotFather.
2. **Create a New Bot:**

   - Use the /newbot command to create a new bot.
   - Follow BotFather's instructions to assign a name and username to your bot.
   - Once completed, BotFather will provide you with a token. This token is required for authenticating your bot.

### Get Chat ID:

1. **Start a Chat with Your Bot:**


   - After creating your bot, initiate a chat with it on Telegram.

3. **Visit the Bot API URL:**


   - Open a web browser and visit the following URL, replacing BOT\_TOKEN with your bot's token:

```
https://api.telegram.org/botBOT_TOKEN/getUpdates
```

1. **Send a Message to the Bot:**

   - Go back to Telegram and send a message to the bot with which you initiated the chat.
2. **Refresh the Browser Page:**

   - Return to the browser page where you entered the bot API URL.
   - Refresh the page.
3. **Find the Chat ID:**

   - In the JSON response, look for the section that contains the message history.
   - Find the object representing your recently sent message.
   - Within that object, locate the chat field, and inside it, the id field will be the Chat ID.

The result of this, will be that you can receive a message like this one (to keep track on the bot)

![Telegram message](https://c.mql5.com/2/64/telegran_image.png)

### Other fixes

As I didn't like how the orders were being closed, I have modified the script to take advantage of the trends. I have added a Simple Moving Average (SMA) and instructed it to close the orders when the price crosses the SMA.

So to fix this:

![SMA](https://c.mql5.com/2/64/sma.png)

I added this:

```
def obtain_sma_pandas(symbol, periodo):
    prices = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, periodo)
    df_sma = pd.DataFrame(prices)
    sma = df_sma['close'].rolling(window=periodo).mean().iloc[-1]
    return sma
```

and this

```
    action="buy"
    price = mt5.symbol_info_tick(symbol).ask
    period_sma=14
    sma = obtain_sma_pandas(symbol, period_sma)
    while True:
        # Obtener el precio actual y la SMA
        actual_price = price
        sma = obtain_sma_pandas(symbol, period_sma)

        # Si el precio está por debajo de la SMA, cerrar la orden
        if actual_price > sma:
            close_all_position(symbol)
            continue

        # Esperar antes de verificar nuevamente
        time.sleep(10)
```

You can also add adx to the script, to search for trades when it's in a trend with this:

```
def calculate_adx(high, low, close, period):

    tr_pos = []
    tr_neg = []

    for i in range(1, len(close)):
        trh = max(high[i] - high[i - 1], 0)
        trl = max(low[i - 1] - low[i], 0)

        if trh > trl:
            tr_pos.append(trh)
            tr_neg.append(0)
        elif trl > trh:
            tr_pos.append(0)
            tr_neg.append(trl)
        else:
            tr_pos.append(0)
            tr_neg.append(0)

    atr = pd.Series(tr_pos).ewm(span=period, adjust=False).mean() + pd.Series(tr_neg).ewm(span=period, adjust=False).mean()
    dx = (pd.Series(tr_pos).ewm(span=period, adjust=False).mean() / atr) * 100
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx.iloc[-1]
```

### ONNX

Now, leveraging the insights from the article ' [How to Use ONNX Models in MQL5](https://www.mql5.com/en/articles/12373)' by MetaQuotes, I am in the process of converting the model to ONNX format. Following the guidelines provided in the same article, I will integrate the resulting ONNX model into the base Expert Advisor (EA) to initiate trading operations. This approach allows for the seamless integration of machine learning models into the MQL5 environment, enhancing the capabilities of the trading algorithm.

Before formatting to ONNX, it is necessary to download the data. To achieve this, we will use the script I have uploaded (ticks\_to\_csv). Simply save it in the MQL5 EA folder, open it in the IDE, and compile it. Once done, add the script to a chart and let it run for some time (as it downloads all ticks for a symbol, it might take a while). In the journal, you will see a completion message when the process is finished. As a reference, I have used it for EUR/USD, and it has occupied several gigabytes.

```
#property script_show_inputs
#property strict
//--- Requesting 100 million ticks to be sure we receive the entire tick history
input long      getticks=100000000000; // The number of required ticks
string fileName = "ticks_data.csv";
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   int     attempts=0;     // Count of attempts
   bool    success=false;  // The flag of a successful copying of ticks
   MqlTick tick_array[];   // Tick receiving array
   MqlTick lasttick;       // To receive last tick data

   SymbolInfoTick(_Symbol,lasttick);
//--- Make 3 attempts to receive ticks
   while(attempts<3)
     {

      //--- Measuring start time before receiving the ticks
      uint start=GetTickCount();
      //--- Requesting the tick history since 1970.01.01 00:00.001 (parameter from=1 ms)
      long received=CopyTicks(_Symbol,tick_array,COPY_TICKS_ALL,1,getticks);

      // Check if ticks were successfully copied
      if(received > 0)
        {
         // Open the CSV file for writing
         int fileHandle = FileOpen(fileName, FILE_WRITE | FILE_CSV);

         // Check if the file was opened successfully
         if(fileHandle != INVALID_HANDLE)
           {
            // Write the CSV header
            FileWrite(fileHandle, "Time,Bid,Ask");

            // Write tick data to the CSV file
            for(long i = 0; i < received; i++)
              {
               string csvLine = StringFormat("%s,%.5f,%.5f", TimeToString(tick_array[i].time), tick_array[i].bid, tick_array[i].ask);
               FileWrite(fileHandle, csvLine);
              }

            // Close the CSV file
            FileClose(fileHandle);

            // Print success message
            Print("Downloaded ", received, " ticks for symbol ", _Symbol, " and period ", Period());
            Print("Ticks data saved to ", fileName);
           }
         else
           {
            // Print an error message if the file could not be opened
            Print("Failed to open the file for writing. Error code: ", GetLastError());
           }
        }
      else
        {
         // Print an error message if no ticks were downloaded
         Print("Failed to download ticks. Error code: ", GetLastError());
        }

      if(received!=-1)
        {
         //--- Showing information about the number of ticks and spent time
         PrintFormat("%s: received %d ticks in %d ms",_Symbol,received,GetTickCount()-start);
         //--- If the tick history is synchronized, the error code is equal to zero
         if(GetLastError()==0)
           {
            success=true;
            break;
           }
         else
            PrintFormat("%s: Ticks are not synchronized yet, %d ticks received for %d ms. Error=%d",
                        _Symbol,received,GetTickCount()-start,_LastError);
        }
      //--- Counting attempts
      attempts++;
      //--- A one-second pause to wait for the end of synchronization of the tick database
      Sleep(1000);
     }
  }
```

And from this data (CSV), we will read and use the segment we need, converting it into a DataFrame (the DeepLearning\_ONNX.py has been defined for the last 1000 days up to today). You can use the entire dataset, but you will need sufficient RAM.

Once we have downloaded data, we can start get the ONNX model from training with the DeepLearningForecast\_ONNX\_training.py. (please save the file in in a folder under the File folder from MQL5.

It imports the data with this:

```
rates = pd.read_csv(file_path, encoding='utf-16le')
```

### creates a model with this:

```
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
```

### Conclusion

In conclusion, this article has outlined a comprehensive approach to developing a Python script for automated trading using deep learning. We explored the integration of MetaTrader 5, data preprocessing, model training, and the execution of buy/sell orders. The utilization of Tkinter and auto-py-to-exe for creating an executable with a user-friendly interface adds practicality to the script.

Additionally, considerations were provided on choosing the appropriate input data, evaluating model performance, and implementing effective closing order strategies. The script's visual output, showcasing predictions, reliable ranges, and order placement, provides a clear and actionable overview for both automated and manual trading.

Last but not least, we will learn how to import all ticks for a currency pair. We will use Python to read this data, create an ONNX model, and (if the next article is interesting) execute the model in MT5 with an Expert Advisor (EA).

However, it's essential to emphasize the dynamic nature of financial markets, warranting continuous monitoring, adaptation, and potential improvements to the model. The outlined framework serves as a solid foundation, and users are encouraged to further tailor and enhance it based on evolving market conditions and their specific trading objectives.

### Disclaimer

Disclaimer: The past performance does not indicate future results. Trading cryptocurrencies and other financial instruments carries risks; no strategy can guarantee profits in every scenario. Always conduct thorough research and seek advice from financial professionals before making investment decisions.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13975.zip "Download all attachments in the single ZIP archive")

[Timer.mq5](https://www.mql5.com/en/articles/download/13975/timer.mq5 "Download Timer.mq5")(4.57 KB)

[Timer\_modified.mq5](https://www.mql5.com/en/articles/download/13975/timer_modified.mq5 "Download Timer_modified.mq5")(9.73 KB)

[Loop.py](https://www.mql5.com/en/articles/download/13975/loop.py "Download Loop.py")(0.4 KB)

[DeepLearningForecast.py](https://www.mql5.com/en/articles/download/13975/deeplearningforecast.py "Download DeepLearningForecast.py")(33.87 KB)

[gui\_console.py](https://www.mql5.com/en/articles/download/13975/gui_console.py "Download gui_console.py")(3.85 KB)

[program\_object.py](https://www.mql5.com/en/articles/download/13975/program_object.py "Download program_object.py")(23.96 KB)

[program\_object\_without.py](https://www.mql5.com/en/articles/download/13975/program_object_without.py "Download program_object_without.py")(28.78 KB)

[ticks\_to\_csv.mq5](https://www.mql5.com/en/articles/download/13975/ticks_to_csv.mq5 "Download ticks_to_csv.mq5")(3.27 KB)

[DeepLearningForecast\_ONNX\_training.py](https://www.mql5.com/en/articles/download/13975/deeplearningforecast_onnx_training.py "Download DeepLearningForecast_ONNX_training.py")(12.16 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Integrating Discord with MetaTrader 5: Building a Trading Bot with Real-Time Notifications](https://www.mql5.com/en/articles/16682)
- [Trading Insights Through Volume: Trend Confirmation](https://www.mql5.com/en/articles/16573)
- [Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)
- [From Python to MQL5: A Journey into Quantum-Inspired Trading Systems](https://www.mql5.com/en/articles/16300)
- [Example of new Indicator and Conditional LSTM](https://www.mql5.com/en/articles/15956)
- [Scalping Orderflow for MQL5](https://www.mql5.com/en/articles/15895)
- [Using PSAR, Heiken Ashi, and Deep Learning Together for Trading](https://www.mql5.com/en/articles/15868)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/460845)**
(4)


![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
20 Jan 2024 at 15:07

Hi, sorry I made an error with the  [DeepLearningForecast\_ONNX\_training.py](https://www.mql5.com/en/articles/download/13975/deeplearningforecast_onnx_training.py "Download DeepLearningForecast_ONNX_training.py") file.

You will have to use this one

![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
20 Jan 2024 at 17:54

**Javier Santiago Gaston De Iriarte Cabrera [#](https://www.mql5.com/en/forum/460845#comment_51835218):**

Hi, sorry I made an error with the  [DeepLearningForecast\_ONNX\_training.py](https://www.mql5.com/en/articles/download/13975/deeplearningforecast_onnx_training.py "Download DeepLearningForecast_ONNX_training.py") file.

You will have to use this one

sorry, this one also has errors, I will make that py tomorrow and leave it here.

I'm also doing a continuation, you will have the correct py also in the continuation.

![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
20 Jan 2024 at 22:20

**Javier Santiago Gaston De Iriarte Cabrera [#](https://www.mql5.com/en/forum/460845#comment_51836095):**

sorry, this one also has errors, I will make that py tomorrow and leave it here.

I'm also doing a continuation, you will have the correct py also in the continuation.

Here is the .py, this is the one I will use in next article.

![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
22 Jan 2024 at 04:15

**Javier Santiago Gaston De Iriarte Cabrera [#](https://www.mql5.com/en/forum/460845#comment_51837416):**

Here is the .py, this is the one I will use in next article.

Sorry for n time.

I forgot to drop some NaN

here is the final file (but, get it from the next article)

![Building and testing Aroon Trading Systems](https://c.mql5.com/2/64/Building_and_testing_Aroon_Trading_Systems___LOGO.png)[Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

In this article, we will learn how we can build an Aroon trading system after learning the basics of the indicators and the needed steps to build a trading system based on the Aroon indicator. After building this trading system, we will test it to see if it can be profitable or needs more optimization.

![Data label for time series mining (Part 6)：Apply and Test in EA Using ONNX](https://c.mql5.com/2/64/Data_label_for_time_series_mining_1Part_60_Apply_and_Test_in_EA_Using_ONNX____LOGO.png)[Data label for time series mining (Part 6)：Apply and Test in EA Using ONNX](https://www.mql5.com/en/articles/13919)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://c.mql5.com/2/64/rj-article-image-60x60.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://www.mql5.com/en/articles/14051)

The multi-currency expert advisor in this article is an expert advisor or trading robot that uses two RSI indicators with crossing lines, the Fast RSI which crosses with the Slow RSI.

![Neural networks made easy (Part 57): Stochastic Marginal Actor-Critic (SMAC)](https://c.mql5.com/2/58/stochastic_marginal_actor_critic_avatar.png)[Neural networks made easy (Part 57): Stochastic Marginal Actor-Critic (SMAC)](https://www.mql5.com/en/articles/13290)

Here I will consider the fairly new Stochastic Marginal Actor-Critic (SMAC) algorithm, which allows building latent variable policies within the framework of entropy maximization.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/13975&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062666763201586847)

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