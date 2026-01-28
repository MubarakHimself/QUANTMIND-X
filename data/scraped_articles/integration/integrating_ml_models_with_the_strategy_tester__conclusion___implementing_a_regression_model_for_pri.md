---
title: Integrating ML models with the Strategy Tester (Conclusion): Implementing a regression model for price prediction
url: https://www.mql5.com/en/articles/12471
categories: Integration, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T21:10:18.175800
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/12471&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071628062140869512)

MetaTrader 5 / Examples


### Introduction:

In the [previous article](https://www.mql5.com/en/articles/12069), we completed the implementation of a CSV file management class for storing and retrieving data related to financial markets. Having created the infrastructure, we are now ready to use this data to build and train a machine learning model.

Our task in this article is to implement a regression model that can predict the closing price of a financial asset within a week. This forecast will allow us to analyze market behavior and make informed decisions when trading financial assets.

Price forecast is a useful tool for developing trading strategies and making decisions in the financial market. The ability to accurately predict price trends can lead to better investment decisions, maximizing profits and minimizing losses. Additionally, price forecasting can help identify trading opportunities and manage risks.

To implement our regression model, we will perform the following steps:

1. **Collect and prepare data**: Using a Python script, we will get historical price data and other required information. We will use this data to train and test our regression model.

2. **Select and train a model**: We will select a suitable regression model for our task and train it using the collected data. There are several regression models such as linear regression, polynomial regression and support vector regression (SVR). The model choice depends on its suitability for solving our problem and the performance obtained during the training process.

3. **Evaluate model performance**: To make sure that our regression model works correctly, we need to evaluate its performance through a series of tests. This evaluation will help us identify potential problems and adjust the model if necessary.


At the end of this article, we will obtain a regression model that can predict the closing price of a financial asset for a week. This forecast will allow us to develop more effective trading strategies and make informed decisions in the financial market.

### Section 1: Selecting a Regression Model

Before applying our regression model to predict the weekly closing price of a financial asset, it is necessary to understand the different types of regression models and their characteristics. This will allow us to choose the most suitable model to solve our problem. In this section, we will discuss some of the most common regression models:

1. **Linear Regression** is one of the simplest and most popular regression models. It assumes a linear relationship between the independent variables and the dependent variable. It aims at finding a straight line that best fits the data while minimizing the sum of the squared error. Although easy to understand and implement, linear regression may not be suitable for problems where the relationship between variables is not linear.

2. **Polynomial Regression** is an extension of linear regression that takes into account non-linear relationships between variables. Uses polynomials of varying degrees to fit a curve to the data. Polynomial regression may provide a better approximation for more complex problems; however, it is important to avoid overfitting, which occurs when the model fits too closely to the training data, reducing its ability to generalize to unseen data.

3. **Decision Tree Regression** is a decision tree-based model that divides the feature space into distinct, non-overlapping regions. In each region, the forecast is made based on the average of the observed values. Decision tree regression is capable of capturing complex nonlinear relationships between variables, but can be subject to overfitting, especially if the tree becomes very large. Pruning and cross-validation techniques can be used to combat overfitting.

4. **Support Vector Regression (SVR)** is an extension of the Support Vector Machine (SVM) algorithm for solving regression problems. SVR attempts to find the best function for the data while maintaining maximum margin between the function and the training points. SVR is capable of modeling nonlinear and complex relationships using kernel functions such as the Radial Basis Function (RBF). However, SVR training can be computationally expensive compared to other regression models.


To select the most appropriate regression model for predicting the closing price of a financial asset for a week, we need to consider the complexity of the problem and the relationship between the variables. In addition, we must consider the balance between model performance and computational complexity. In general, I recommend experimenting with different models and adjusting their settings to achieve the best performance.

When choosing a model, it is important to consider several criteria that affect the quality and applicability of the model. In this section we will look at the main criteria that must be taken into account when choosing a regression model:

1. The **performance** of a regression model is very important to ensure the accuracy and usefulness of predictions. We can evaluate performance using mean square error (MSE) and mean absolute error (MAE), among other metrics. When comparing different models, it is important to choose the one that performs best on these metrics.

2. Model **Interpretability** is the ability to understand the relationships between variables and how they affect the prediction. Simpler models such as linear regression are generally easier to interpret than more complex models such as neural networks. Interpretability is especially important if we want to explain our predictions to others or understand the factors that influence results.

3. The **complexity** of a regression model is related to the number of parameters as well as the structure of the model. More complex models can capture more subtle and nonlinear relationships in the data, but they may also be more prone to overfitting. It is important to find a balance between the complexity of the model and the ability to generalize it to unknown data.

4. **Training time** is an important point to consider, especially when working with large data sets or when training models iteratively. Simpler models, such as linear and polynomial regression, typically require less training time than other, more complex models, such as neural networks or support vector regression. It is important to find a balance between model performance and training time to ensure that the model is applicable.

5. **Robustness** of a regression model is its ability to deal with outliers and noise in the data. Robust models are less sensitive to small changes in data and produce more stable forecasts. It is important to choose a model that can handle outliers and noise in the data.


When choosing the most appropriate regression model for forecasting closing prices, it is important to weigh these criteria and find the right balance between them. It is typically recommended to test different models and fine-tune their parameters to optimize performance. This way you can select the best model for a particular problem.

Based on the above criteria, in this article, I decided to use the Decision Tree Regression model to predict the closing price. The choice of this model is justified for the following reasons:

1. **Performance**: Decision trees typically work well for regression problems because they are able to capture nonlinear relationships and interactions between variables. By properly tuning model hyperparameters, such as tree depth and minimum number of samples per leaf, we can achieve a balance between fitness and generalization.

2. **Interpretability**: One of the main advantages of decision trees is their interpretability. Decision trees are a series of decisions based on attributes and their values, making them easy to understand. This is useful for justifying forecasts and understanding the factors influencing closing prices.

3. **Complexity**: The complexity of decision trees can be controlled by tuning the hyperparameters of the model. With this, we canfind a balance between the ability to model complex relationships and the simplicity of the model, while avoiding overfitting.

4. **Training time**: Decision trees typically train relatively quickly compared to more complex models such as neural networks or SVMs. This fact makes the decision tree regression model suitable for cases where training time is an important factor.

5. **Robustness**: Decision trees are robust to outliers and noise in the data because each decision is based on a set of samples rather than a single observation, and this contributes to the stability of predictions and the reliability of the model.


Given the criteria discussed and the benefits of decision tree regression, I believe this model is suitable for predicting the weekly closing price. However, it is important to remember that the choice of models may vary depending on the specific context and requirements of each problem. Therefore, to select the most appropriate model for your specific problem, you should test and compare different regression models.

### Section 2: Data Preparation

Data preparation and cleaning are important steps in the process of implementing a regression model, since the quality of the input data directly affects the efficiency and performance of the model. These steps are important for the following reasons:

1. **Elimination of outliers and noise**: Raw data may contain outliers, noise and errors that can negatively affect the performance of the model. By identifying and correcting these inconsistencies, you can improve the quality of your data and, therefore, the accuracy of your forecasts.

2. **Filling and removing missing values**: Incomplete data is common in datasets, and missing values can cause the model to perform poorly. To ensure data integrity and reliability, you might consider imputing missing values, deleting records with missing data, or using special techniques to deal with such data. The choice between imputing and deletion depends on the nature of the data, the number of missing values, and the potential impact of those values on model performance. It is important to carefully analyze each situation and choose the most appropriate approach to solving the problem.

3. **Selecting Variables**: Not all variables present in the data set may be important or useful in predicting the closing price. Appropriate variable selection allows the model to focus on the most important features, improving performance and reducing model complexity.

4. **Data Transformation**: Sometimes the original data needs to be transformed to match the assumptions of the regression model or to improve the relationship between independent variables and the dependent variable. Examples of transformations include normalization, standardization, and the use of mathematical functions such as logarithm or square root.

5. **Data division**: Divide the data set into a training subset and a testing subset to properly evaluate the performance of the regression model. This division allows the model to be trained on a subset of data and tested for its ability to generalize to unseen data, which provides an assessment of the model's performance in real-world situations.


The stage of data preparation and cleaning ensures that the model is trained and evaluated from quality data, maximizing its effectiveness and usefulness in predicting closing prices.

We'll look at a basic example of preparing data for a regression model using Python. However, I would like to note that it is important to deepen your knowledge as each specific data set and each problem may require special preparation approaches and methods. Therefore, I strongly recommend that you take the time to learn and understand the various data preparation methods.

To collect data, we will use the **get\_rates\_between** function, is intended for collecting financial data for a specific asset and for a specific period. It uses the MetaTrader 5 library to connect to the trading platform and obtain historical price data for various time intervals.

The function has the following parameters:

- **symbol**: a string that represents the financial symbol (for example, "PETR3", "EURUSD").
- **period**: an integer that specifies the time period for which the data will be collected (for example, mt5.TIMEFRAME\_W1 for weekly data).
- **ini**: a datetime object that represents the start time and date of the time interval for data collection.
- **end**: a datetime object that represents the end date and time of the time interval for data collection.

The function starts by checking MetaTrader 5 boot. If boot failed, the function returns an exception and terminates the program.

Then the function uses **mt5.copy\_rates\_range()** to get financial data of the specified symbol and period. The data is saved in the DataFrame object from pandas, which is a two-dimensional axis-labeled data structure that is suitable for storing financial data.

After receiving the data, the function checks if the DataFrame is empty. If the function is empty, it will return an exception because this indicates that an error occurred while collecting data.

If all goes well, the function will convert the 'time' column of DataFrame to a readable date and time format using the **pd.to\_datetime()** function. The 'time' column is defined as the DataFrame index, which facilitates data access and manipulation.

```
def get_rates_between(symbol:str, period : int, ini : datetime, end : datetime):
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        raise Exception("Error Getting Data")

    rates = mt5.copy_rates_range(symbol, period, ini, end)
    mt5.shutdown()
    rates = pd.DataFrame(rates)

    if rates.empty:
        raise Exception("Error Getting Data")

    rates['time'] = pd.to_datetime(rates['time'], unit='s')
    rates.set_index(['time'], inplace=True)

    return rates
```

In this example, we will use financial data for the EURUSD currency pair for the weekly period from January 1, 2000 to December 31, 2022. For this, we use the get\_rates\_between function. First, import the necessary libraries:

```
import pandas as pd
from datetime import datetime, timezone
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```

Next, determine information about the financial symbol that we want to analyze:

```
symbol = "EURUSD"
date_ini = datetime(2000, 1, 1, tzinfo=timezone.utc)
date_end = datetime(2022, 12, 31, tzinfo=timezone.utc)
period = mt5.TIMEFRAME_W1
```

Now we can call the get\_rates\_between function using the information defined earlier:

```
df = get_rates_between(symbol=symbol, period=period, ini=date_ini, end=date_end)
```

Once we have the data, the next step is to prepare it to create a machine learning model. Preparation includes removing outliers and noise, selecting variables, transforming the data, and dividing it into training and testing sets. Let's consider each step in detail.

**Remove noise and outliers:**

The first step is to remove outliers and noises from the data. Noises are random and unwanted changes in data that can make it difficult to identify patterns. Outliers are values that are significantly different from other values in our dataset. Both can negatively impact the performance of the model.

There are several methods for removing outliers and noise. In this article, we will use the exponential smoothing method, which assigns exponentially decreasing weight to the most recent data, helping to smooth out fluctuations. To do this we will use the **ewm** function from pandas.

```
smoothed_df = df.ewm(alpha=0.1).mean()
```

Actually you may use any method, that suits your model and data. Here we use exponential smoothing just to simplify the example and to demonstrate how to process the data. In real-world scenarios, it is recommended to study and evaluate different outlier and noise removal methods to find the best approach for a specific data set and a specific problem. Other popular methods include moving average filtering, percentile screening, and clustering.

**Selection of variables**

The next step is to select the variables that we will use as features and targets. In this example, we will use the opening price, the Moving Average Convergence Divergence (MACD) indicator, and the exponential moving average (EMA) as features. The target will be the closing price.

```
# Function to calculate MACD
def macd(df, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = df['close'].ewm(span=fast_period).mean()
    ema_slow = df['close'].ewm(span=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    return macd_line, signal_line

# Function to calculate EMA
def ema(df, period=30):
    return df['close'].ewm(span=period).mean()

# Calculating MACD and the signal line
smoothed_df['macd'], smoothed_df['signal'] = macd(smoothed_df)

# Calculating EMA
smoothed_df['ema'] = ema(smoothed_df)

# Selecting the   variables
selected_df = smoothed_df[['open', 'macd', 'ema', 'close']].dropna()
```

**Data Conversion**

Data transformation is important so that the variables are on the same scale and we can compare them correctly. In this example, we use Min-Max normalization, which converts the data to a scale from 0 to 1.

```
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(selected_df), columns=selected_df.columns, index=selected_df.index)
```

**Data division**

Finally, we will divide the data into training and testing sets. We will use the training set to train the model, and the testing one to evaluate its performance.

```
X = normalized_df[['open', 'macd', 'ema']]
y = normalized_df['close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```

In Section 2, we perform several important steps to ensure that the data is ready to be used in the machine learning model.

We start by loading financial symbol data using the MetaTrader 5 API, then convert it into a pandas DataFrame, and after that set up a time column to use as an index. We then clean the data by removing any missing values and checking for outliers and inconsistencies. This action is necessary to ensure that the model is not influenced by invalid or irrelevant values.

For the variables, we have previously chosen Open, MACD and EMA as features and Close as target. However, please note that this selection was made at random to provide a clear example.

We have also normalized the data. This is important to ensure that differences in data scale do not affect the model. Finally, we divide the data into training and testing sets to evaluate the model's ability to generalize to new data.

These data preparation steps are critical to ensuring the model's predictions are accurate and efficient. When the data is ready, we can move on to the next stage, that is, creating and training a machine learning model.

**Section 3: Training and Evaluating a Regression Model Using a Decision Tree**

After creating the training and testing sets and preparing the data accordingly, we are ready to create, train, and evaluate our decision tree regression model. Decision tree is a supervised learning method that we can apply to both classification and regression problems. In this case, we will use a decision tree to predict the closing price of a financial asset.

- Importing libraries and creating a model

Let's start by importing the necessary libraries and creating an instance of the DecisionTreeRegressor model from scikit-learn.

```
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

regressor = DecisionTreeRegressor(random_state=42)
```

- Model training

Next, we train the model using the training dataset (X\_train and y\_train).

```
regressor.fit(X_train, y_train)
```

- Making predictions

With the trained model, we can make predictions on the testing dataset (X\_test) and compare the results with the actual values of the testing dataset (y\_test).

```
y_pred = regressor.predict(X_test)
```

- Evaluating the model performance

Evaluating a model's performance is important to understanding how well it fits the data and makes predictions. In addition to the mean square error (MSE) and coefficient of determination (R²), we can use other metrics to evaluate the performance of our decision tree regression model. Some additional metrics we can study include:

- Mean absolute error (MAE), which is the average of the absolute difference between forecasts and actual values. This is an easy-to-understand metric that shows how far forecasts are from actual values.
- Mean absolute percentage error (MAPE) is the average of the absolute percentage errors between forecasts and actual values. This metric allows evaluates the efficiency of the model as a percentage, which can be useful when comparing models with different value scales.
- Root Mean Square Error (RMSE) is the square root of MSE. The advantage of this metric is that it has the same units as the target variable, making the results easier to interpret.

We can use the Scikit-learn library to calculate these additional metrics. First, import the necessary functions:

```
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

Then, calculate the metrics using predicted and actual values:

```
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

Now we can demonstrate the results:

```
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

- Model setup and optimization

Depending on the results obtained, adjustments and optimization of the model may be necessary. This can be done by tuning decision tree hyperparameters such as maximum depth (max\_depth), minimum number of samples required to split an internal node (min\_samples\_split), minimum number of samples required to be in a leaf node (min\_samples\_leaf), and others.

To optimize hyperparameters, there are methods such as grid search (GridSearchCV) or randomized search (RandomizedSearchCV) from scikit-learn. These methods allow you to test different combinations of hyperparameters and find the best configuration for the model.

```
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=42)

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}

grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Melhores hiperparâmetros: {best_params}")

best_regressor = DecisionTreeRegressor(**best_params, random_state=42)
best_regressor.fit(X_train, y_train)

y_pred_optimized = best_regressor.predict(X_test)

mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mse_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f"MAE otimizado: {mae_optimized:.4f}")
print(f"MSE otimizado: {mse_optimized:.4f}")
print(f"RMSE otimizado: {rmse_optimized:.4f}")
print(f"R² otimizado: {r2_optimized:.4f}")
```

- Denormalizing prices and creating a chart


To denormalize prices and plot a chart, you can use inverse\_transform from MinMaxScaler. First, we denormalize the actual price (y\_test) and the expected price (y\_pred\_optimised), and then build a graph using the matplotlib library. Here's the updated code:

```
import matplotlib.pyplot as plt

# Function for denormalizing prices
def denormalize_price(scaler, normalized_price, column_name):
    dummy_df = pd.DataFrame(np.zeros((len(normalized_price), len(selected_df.columns))), columns=selected_df.columns)
    dummy_df[column_name] = normalized_price
    denormalized_df = scaler.inverse_transform(dummy_df)
    return denormalized_df[:, selected_df.columns.get_loc(column_name)]

# Denormalize actual and forecast prices
y_test_denorm = denormalize_price(scaler, y_test, 'close')
y_pred_optimized_denorm
```

This section discusses the implementation of a regression model with a decision tree to predict the closing price of a financial asset. After training and evaluating the model, we obtained performance metrics and tuned hyperparameters to optimize its performance.

However, it is important to note that this example is just a basic approach, and there are many advanced techniques and strategies available to improve model accuracy and performance. In addition, the choice of regression model and data preparation may vary depending on the context and specific requirements of each task.

Finally, you should always do comprehensive model testing, validations and iterations in different scenarios, using different datasets, to ensure its robustness and reliability in predicting financial market prices.

### Section 4: Integrating a Regression Model into a Strategy Tester

In this section, we will take a detailed look at creating a system that allows you to test a Python model directly in the MetaTrader Strategy Tester. This approach will add an additional level of flexibility in model creation, taking advantage of Python's simplicity. The verification process will take place in MetaTrader itself, since we are exporting the model to the ONNX format for MQL5. This means we can easily build and refine our model while rigorous validation is carried out within the MetaTrader environment, ensuring it is ready for the complexities of the financial market.

**Effective interaction between Python and MQL5**

This system is based on the critical need to establish effective interaction between Python and MQL5. To achieve this, a Python class will be created to simplify the exchange of messages with the MQL5 Strategy Tester. While the Python program is running, it will monitor trades made in the strategy tester, promptly transmitting asset price data to Python once the tester is launched. This continuous flow of information provides the regression model with real-time data, allowing it to make decisions based on that data. Also, you might want to get back to this [article](https://www.mql5.com/en/articles/12069) to refresh information on the implementation of an MQL5 class for working with CSV files.

**Building** **the File class for communication between Python and MQL5**

The File class enables communication between Python and the MQL5 strategy tester. Let's see the main functionality of this class. It's main purpose is to implement file operations.

**Initializing the class**

The File class is designed as a Singleton, which guarantees that there will only be one instance of it throughout the program. This point is important to maintain the integrity of communication between Python and MQL5.

```
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
```

**Verifying and handling files**

The File class provides basic methods for checking for the presence of files, handling file-related errors, and performing read and write operations.

- The \_\_init\_file method checks whether the file exists or not. If the file does not exist, the method will wait one second before returning False. This check allows making sure the file is ready to be accessed.

- The \_\_handle\_error method handles exceptions that may occur when working with files. It identifies common errors such as PermissionError and FileNotFoundError and provides detailed information about these errors.

- The check\_init\_param and check\_open\_file methods are needed to read CSV files. The first one checks if the file exists and, if it does, reads it and returns a specific value from the typerun column in DataFrame. The second method checks for the existence of the file and reads it as a complete DataFrame.


```
from pathlib import Path
from time import sleep
from typing import Tuple
import pandas as pd
from pandas.core.frame import DataFrame
import os
from errno import EACCES, EPERM, ENOENT
import sys

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

CSV_SEPARATOR = ';'

class File(metaclass=Singleton):

    def __init__(self) -> None:
        pass

    def __init_file(self, name_arq: str) -> bool:
        return Path(name_arq).is_file() or sleep(1) or False

    def __handle_error(self, e, name_arq: str):
        ERRORS = {
            EPERM: "PermissionError",
            EACCES: "PermissionError",
            ENOENT: "FileNotFoundError"
        }
        print(f"{ERRORS.get(e.errno, 'Unknown error')} error({e.errno}): {e.strerror} for:\n{name_arq}")

    def check_init_param(self, name_arq : str) -> Tuple[str]:
        while True:
            try:
                if self.__init_file(name_arq):
                    df = pd.read_csv(name_arq, sep=CSV_SEPARATOR)
                    return (df.typerun.values[0])
            except (IOError, OSError) as e:
                self.__handle_error(e, name_arq)
            except:
                print('Unexpected error:', sys.exc_info()[0])

    def check_open_file(self, name_arq: str) -> pd.DataFrame():
        while True:
            try:
                if self.__init_file(name_arq):
                    return pd.read_csv(name_arq, sep=CSV_SEPARATOR)
            except (IOError, OSError) as e:
                self.__handle_error(e, name_arq)
            except:
                print('Unexpected error:', sys.exc_info()[0])

    @staticmethod
    def save_file_csv(name_arq: str, dataset:DataFrame = pd.DataFrame({'col':['ok']})):
        dataset.to_csv(name_arq, sep=CSV_SEPARATOR)

    @staticmethod
    def save(name_arq:str, data:str):
        with open(name_arq, 'w') as f:
            f.write(data)

    @staticmethod
    def delete_file(name_arq: str):
        try:
            os.remove(name_arq)
        except:
            pass
```

**Saving and deleting files**

The File class also manages file saving and deletion.

- The save\_file\_csv method saves DataFrame to a CSV file. This is convenient when we need to save data in an accessible format for later analysis.

- The Save method is used to save data to a text file. The data in a csv file is stored in a convenient readable format.

- The delete\_file method deletes a file from the system.


**Data Processing in Python**

On the Python side, the program receives data on the symbol prices and pre-processes it. The aim of this preprocessing is to make sure that the data is ready for use on a regression model. Let's now take a closer look at how this works:

1. **Data standardization**: At the initial stage, data is normalized. This means setting values within a certain range, usually between 0 and 1. Normalization is necessary to ensure that various characteristics (for example, stock prices and trading volumes) are on the same scale. This avoids errors in the model.

2. **Outlier detection and removal**: Financial data may be unstable and outliers may adversely affect simulation results. The Python program detects and, if necessary, corrects these deviations to ensure the quality of the source data.

3. **Creating additional features**: In many cases, additional features are created based on the original data. These could be, for example, moving averages, technical indicators, or other metrics that the model can use to make more informed decisions.


Only after the data has been fully prepared, we feed it into the regression model.

**Regression model in action**

Our regression model based on the decision tree has been trained on historical data. It is now loaded into a Python program and used to make forecasts based on the latest price data. These forecasts are fundamental to making trading decisions as they provide valuable information about the future behavior of an asset. It should be noted that although in this example we are using a specific regression model, this approach can be applied to various models ones, depending on the specific requirements of our trading strategy. Thus, the flexibility of this system allows it to include a variety of models to meet the individual needs of each strategy.

**ONNX integration to improve MetaTrader 5**

I would like to share an interesting tip that can improve our system in MetaTrader 5: integration with ONNX (Open Neural Network Exchange). ONNX is a very convenient tool for artificial intelligence, as it allows you to easily transfer models to MetaTrader 5.

After thoroughly testing and confirming the effectiveness of our model in MetaTrader 5, we can consider exporting it to the ONNX format. This will not only expand the capabilities of MetaTrader 5, but will also facilitate the use of this model in various strategies and platform systems.

With ONNX's native integration into MetaTrader 5, this can further expand the use of models in our trading strategies.

**Extensive testing before export**

The choice of our current approach, which involves interaction with the MQL5 environment, is due to the need to conduct extensive testing in a controlled environment before exporting the model to the ONNX format. Building and directly integrating a model into MQL5 can be a complex and error-prone process. Thus, this methodology allows us to carefully refine and tune the model to achieve the desired results, even before it is exported to the ONNX format.

### Conclusion

To summarize, we can say that this article discusses an integrated approach to implementing a regression model. In Section 1, we discussed the selection of a regression model, focusing on the importance of choosing the right algorithm to solve the problem at hand. In Section 2, we looked at data preparation, including processing and cleaning for model training.

In Section 3, we trained and evaluated a regression model using a decision tree as an example. In addition, we considered the dividing of data into training and testing datasets, as well as selecting hyperparameters to optimize model performance.

Finally, in Section 4 we explored the integration of a regression model into the MetaTrader strategy tester, interaction between Python and MQL5, and use of ONNX format to improve the implementation in MetaTrader 5.

In summary, this article has provided an overview of the key steps involved in introducing models into trading strategies, highlighting the importance of proper model selection, data preparation, training and evaluation, and effective integration into the trading environment. These steps are the foundation for building more robust, data-driven trading systems.

The above code is available in the [Git repository](https://www.mql5.com/go?link=https://github.com/jowpereira/MetaTrader-Python-Integration "https://github.com/jowpereira/MetaTrader-Python-Integration") for further usage.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12471](https://www.mql5.com/pt/articles/12471)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing an MQL5 RL agent with RestAPI integration (Part 4): Organizing functions in classes in MQL5](https://www.mql5.com/en/articles/13863)
- [Developing an MQL5 RL agent with RestAPI integration (Part 3): Creating automatic moves and test scripts in MQL5](https://www.mql5.com/en/articles/13813)
- [Developing an MQL5 RL agent with RestAPI integration (Part 2): MQL5 functions for HTTP interaction with the tic-tac-toe game REST API](https://www.mql5.com/en/articles/13714)
- [Developing an MQL5 Reinforcement Learning agent with RestAPI integration (Part 1): How to use RestAPIs in MQL5](https://www.mql5.com/en/articles/13661)
- [Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://www.mql5.com/en/articles/12069)
- [Multilayer perceptron and backpropagation algorithm (Part 3): Integration with the Strategy Tester - Overview (I).](https://www.mql5.com/en/articles/9875)

**[Go to discussion](https://www.mql5.com/en/forum/463569)**

![Deep Learning GRU model with Python to ONNX  with EA, and GRU vs LSTM models](https://c.mql5.com/2/70/Deep_Learning_Forecast_and_ordering_with_Python_and_MetaTrader5_python_packag___LOGOe.png)[Deep Learning GRU model with Python to ONNX with EA, and GRU vs LSTM models](https://www.mql5.com/en/articles/14113)

We will guide you through the entire process of DL with python to make a GRU ONNX model, culminating in the creation of an Expert Advisor (EA) designed for trading, and subsequently comparing GRU model with LSTM model.

![Neural networks made easy (Part 61): Optimism issue in offline reinforcement learning](https://c.mql5.com/2/59/NN_easy_61_Logo__V4_.png)[Neural networks made easy (Part 61): Optimism issue in offline reinforcement learning](https://www.mql5.com/en/articles/13639)

During the offline learning, we optimize the Agent's policy based on the training sample data. The resulting strategy gives the Agent confidence in its actions. However, such optimism is not always justified and can cause increased risks during the model operation. Today we will look at one of the methods to reduce these risks.

![Population optimization algorithms: Charged System Search (CSS) algorithm](https://c.mql5.com/2/59/Charged_System_Search_CSS__logo.png)[Population optimization algorithms: Charged System Search (CSS) algorithm](https://www.mql5.com/en/articles/13662)

In this article, we will consider another optimization algorithm inspired by inanimate nature - Charged System Search (CSS) algorithm. The purpose of this article is to present a new optimization algorithm based on the principles of physics and mechanics.

![Creating multi-symbol, multi-period indicators](https://c.mql5.com/2/59/multi-period_indicators_logo.png)[Creating multi-symbol, multi-period indicators](https://www.mql5.com/en/articles/13578)

In this article, we will look at the principles of creating multi-symbol, multi-period indicators. We will also see how to access the data of such indicators from Expert Advisors and other indicators. We will consider the main features of using multi-indicators in Expert Advisors and indicators and will see how to plot them through custom indicator buffers.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/12471&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071628062140869512)

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