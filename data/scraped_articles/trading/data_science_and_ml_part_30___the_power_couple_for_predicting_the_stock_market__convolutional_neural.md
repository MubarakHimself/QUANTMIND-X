---
title: Data Science and ML(Part 30): The Power Couple for Predicting the Stock Market, Convolutional Neural Networks(CNNs) and Recurrent Neural Networks(RNNs)
url: https://www.mql5.com/en/articles/15585
categories: Trading, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:00:27.299222
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pxgrimymtyvzvjtgywsbsaqlbrhrvqnw&ssn=1769180425385996833&ssn_dr=0&ssn_sr=0&fv_date=1769180425&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15585&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Data%20Science%20and%20ML(Part%2030)%3A%20The%20Power%20Couple%20for%20Predicting%20the%20Stock%20Market%2C%20Convolutional%20Neural%20Networks(CNNs)%20and%20Recurrent%20Neural%20Networks(RNNs)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691804257112843&fz_uniq=5068907964863020809&sv=2552)

MetaTrader 5 / Trading


**Contents**

- [Introduction](https://www.mql5.com/en/articles/15585#introduction)
- [Understanding RNNs and CNNs](https://www.mql5.com/en/articles/15585#understanding-rnns-and-cnns)
- [The synergy of CNNs and RNNs](https://www.mql5.com/en/articles/15585#synergy-of-cnns-and-rnns)
- [Feature extraction with CNNs](https://www.mql5.com/en/articles/15585#feature-extraction-with-cnns)
- [Temporal modeling with RNNs](https://www.mql5.com/en/articles/15585#temporal-modelling-with-rnns)
- [Training and making predictions](https://www.mql5.com/en/articles/15585#training-and-getting-predictions)
- [A combination of CNN and LSTM](https://www.mql5.com/en/articles/15585#combination-btn-CNN-and-LSTM)
- [A combination of CNN and GRU](https://www.mql5.com/en/articles/15585#combination-btn-CNN-and-GRU)
- [Conclusion](https://www.mql5.com/en/articles/15585#conclusion)

### Introduction

In the previous articles, we have seen how powerful both [Convolutional Neural Networks (CNNs)](https://www.mql5.com/en/articles/15259/156899#!tab=article) and [Recurrent Neural Networks (RNNs)](https://www.mql5.com/en/articles/15182/156936#!tab=article) are and how they can be deployed to help beat the market by providing us with valuable trading signals.

In this one we are going to attempt combining two of the most powerful techniques CNN and RNN and observe their predictive impact in the stock market. But before that let us briefly understand what CNN and RNN are all about.

### Understanding Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs)

**Convolutional Neural Networks (CNNs)**, are designed to recognize patterns and features in the data, despite originally being developed for image recognition tasks, they perform well in tabular data that is specifically designed for time series forecasting.

As said in the previous articles, they operate by firstly applying filters to the input data then they extract high-level features that can be useful for prediction. In stock market data, these features include trends, seasonal effects, and anomalies.

![convolutional neural network](https://c.mql5.com/2/142/coonvolutional_neural_network__1.png)

> > > CNN architecture

By leveraging the hierarchical nature of CNNs, we can uncover layers of data representations, each providing insights into different aspects of the market.

**Recurrent Neural Networks (RNNs)** are artificial neural networks designed to recognize patterns in sequences of data, such as time series, languages, or in videos.

_Unlike traditional [neural networks](https://www.mql5.com/en/articles/12209), which assume that inputs are independent of each other, RNNs can detect and understand patterns from a sequence of data (information)._

![](https://c.mql5.com/2/142/4586372426720__1.png)

RNNs are explicitly designed for sequential data, Their architecture allows them to maintain a memory of previous inputs, making them very suitable for time series forecasting since are capable of understanding temporal dependencies within the data which is crucial for making accurate predictions in the stock market.

As I explained in [part 25](https://www.mql5.com/en/articles/15114) of this article series, There are three ( _commonly used)_ specific types of RNNs which include, a vanilla Recurrent Neural Network(RNN), Long-Short Term Memory(LSTM), and Gated Recurrent Unit(GRU).

Since CNNs excel at extracting and detecting features from the data, RNNs are exceptional at interpreting these features over time. The idea is simple, to combine these two and see if we can build a powerful and robust model capable of making better predictions in the stock market.

The Synergy of CNNs and RNNs

To integrate these two, we are going to create the models in three steps.

1. Feature Extraction with CNNs

2. Temporal Modeling with RNNs

3. Training and Getting Predictions


Let us go through one step after the other and build this robust model comprised of both RNN and LSTM.

### 01: Feature Extraction with CNNs

This first step involves feeding the time series data into a CNN model, the CNN model processes the data, identifying significant patterns and extracting relevant features.

Using the Tesla stock dataset which consists of Open, High, Low, and Close values. Let us start by preparing the data into a 3D time series format acceptable by CNNs and RNNs.

Let us create the target variable for a classification problem.

Python code

```
target_var = []

open_price = new_df["Open"]
close_price = new_df["Close"]

for i in range(len(open_price)):
    if close_price[i] > open_price[i]: # Closing price is greater than opening price
        target_var.append(1) # buy signal
    else:
        target_var.append(0) # sell signal
```

We normalize the data using the [Standard scaler](https://www.mql5.com/go?link=https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html") to make it robust for ML purposes.

```
X = new_df.iloc[:, :-1]
y = target_var

# Scalling the data

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"x_train = {X_train.shape} - x_test = {X_test.shape}\n\ny_train = {len(y_train)} - y_test = {len(y_test)}")
```

Outputs

```
x_train = (799, 3) - x_test = (200, 3)

y_train = 799 - y_test = 200
```

We can then prepare the data into time series format.

```
# creating the sequence

X_train, y_train = create_sequences(X_train, y_train, time_step)
X_test, y_test = create_sequences(X_test, y_test, time_step)
```

Since this is a classification problem, we [one-hot encode](https://www.mql5.com/en/articles/11858#one-hot-encoding-matrix) the target variable.

```
from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

print(f"One hot encoded\n\ny_train {y_train_encoded.shape}\ny_test {y_test_encoded.shape}")
```

Outputs

```
One hot encoded

y_train (794, 2)
y_test (195, 2)
```

Feature extraction is performed by the CNN model itself, Let's give the model raw data we just prepared it.

```
model = Sequential()
model.add(Conv1D(filters=16,
                 kernel_size=3,
                 activation='relu',
                 strides=2,
                 padding='causal',
                 input_shape=(time_step, X_train.shape[2])
                )
         )

model.add(MaxPooling1D(pool_size=2))
```

### 02: Temporal Modeling with RNNs

The extracted features in the previous step are then passed to the RNN model. The model processes these features, considering the temporal order and dependencies within the data.

Unlike the CNN model architecture we used in [part 27 of this article series](https://www.mql5.com/en/articles/15259#fully-connected-cnn-layers) where we used Fully Connected Neural Network Layers right after the Flatten layer. This time we replace these regular Neural Network(NN) layers with Recurrent Neural Network (RNN) layers.

Without forgetting to remove the "Flatten layer" that is seen in the CNN architecture image.

We remove the Flatten layer in the CNN architecture because this layer is typically used to convert a 3D input into a 2D output meanwhile the RNNs (RNN, LSTM, and GRU) expects a 3D input data in the form of (batch size, time steps, features).

```
model.add(MaxPooling1D(pool_size=2))

model.add(SimpleRNN(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=len(np.unique(y)), activation='softmax'))  # Softmax for binary classification (1 buy, 0 sell signal)
```

### 03: Training and Getting Predictions

Finally, we can proceed to train the model we built in the prior two steps, after that, we validate it, measure its performance then get the predictions out of it.

Python code

```
model.summary()

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train_encoded, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

plt.figure(figsize=(7.5, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig("training loss curve-rnn-cnn-clf.png")
plt.show()

# Evaluating the Trained Model

y_pred = model.predict(X_test)

classes_in_y = np.unique(y)
y_pred_binary = classes_in_y[np.argmax(y_pred, axis=1)]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion-matrix RNN + CNN.png")  # Display the heatmap

print("Classification Report\n",
      classification_report(y_test, y_pred_binary))
```

Outputs

After evaluating the model after 14 epochs, The model was 54% accurate on the test data.

![RNN + LSTM training loss curve](https://c.mql5.com/2/142/RNN_training_loss_curve__2.png)

```
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
Classification Report
               precision    recall  f1-score   support

           0       0.70      0.40      0.51       117
           1       0.45      0.74      0.56        78

    accuracy                           0.54       195
   macro avg       0.58      0.57      0.54       195
weighted avg       0.60      0.54      0.53       195
```

It is worth mentioning that training the final model did take some time when more layers were added, this is due to the complex nature of the two models we combined.

After training, I had to save the final model to ONNX format.

Python code

```
onnx_file_name = "rnn+cnn.TSLA.D1.onnx"

spec = (tf.TensorSpec((None, time_step, X_train.shape[2]), tf.float16, name="input"),)
model.output_names = ['outputs']

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model to a file
with open(onnx_file_name, "wb") as f:
    f.write(onnx_model.SerializeToString())
```

Without forgetting to save the Standardization scaler parameters too.

```
# Save the mean and scale parameters to binary files

scaler.mean_.tofile(f"{onnx_file_name.replace('.onnx','')}.standard_scaler_mean.bin")
scaler.scale_.tofile(f"{onnx_file_name.replace('.onnx','')}.standard_scaler_scale.bin")
```

I opened the saved ONNX model in [Netron](https://www.mql5.com/go?link=https://github.com/lutzroeder/netron "https://github.com/lutzroeder/netron"), It is a massive one.

![CNN + RNN model](https://c.mql5.com/2/142/bandicam_2024-09-24_10-24-10-1281__1.gif)

Similar to how we [deployed the Convolutional Neural Network (CNN) before](https://www.mql5.com/en/articles/15259#creating-cnn-baded-EA), We can use the same library to aid us with the task of reading this massive model effortlessly in MQL5.

```
#include <MALE5\Convolutional Neural Networks(CNNs)\ConvNet.mqh>
#include <MALE5\preprocessing.mqh>

CConvNet cnn;
StandardizationScaler *scaler; //from preprocessing.mqh
```

But, before that, we have to add the ONNX model and Standardization scaler parameters to our Expert Advisor as resources.

```
#resource "\\Files\\rnn+cnn.TSLA.D1.onnx" as uchar onnx_model[]
#resource "\\Files\\rnn+cnn.TSLA.D1.standard_scaler_mean.bin" as double standardization_mean[]
#resource "\\Files\\rnn+cnn.TSLA.D1.standard_scaler_scale.bin" as double standardization_std[]
```

The first thing we have to do inside the OnInit function is to initialize them both (the standardization scaler and the CNN model).

```
int OnInit()
  {
//---

   if (!cnn.Init(onnx_model)) //Initialize the Convolutional neural network
     return INIT_FAILED;

   scaler = new StandardizationScaler(standardization_mean, standardization_std); //Initialize the saved scaler by populating it with values

   ...
   ...

  return (INIT_SUCCEEDED);
 }
```

To get the predictions, we have to normalize the input data using this preloaded scaler, then we apply the normalized data to the CNN model and get the predicted signal and probabilities.

```
   if (NewBar()) //Trade at the opening of a new candle
    {
      CopyRates(Symbol(), PERIOD_D1, 1, time_step, rates);

      for (ulong i=0;  i<x_data.Rows(); i++)
        {
          x_data[i][0] = rates[i].open;
          x_data[i][1] = rates[i].high;
          x_data[i][2] = rates[i].low;
        }

   //---

      x_data = scaler.transform(x_data); //Normalize the data

      int signal = cnn.predict_bin(x_data, classes_in_data_); //getting a trading signal from the RNN model
      vector probabilities = cnn.predict_proba(x_data);  //probability for each class

      Comment("Probability = ",probabilities,"\nSignal = ",signal);
```

Below is how the comment looks on the chart.

![signal and probabilty comments on chart](https://c.mql5.com/2/142/bandicam_2024-09-24_11-00-50-323__1.png)

The probability vector depends on the classes that were present in the target variable of your training data. From the training data, we prepared the target variable to indicate 0 for a sell signal and 1 for a buy signal. **The class identifiers or numbers must be in ascending order.**

```
input int time_step = 5;
input int magic_number = 24092024;
input int slippage = 100;

MqlRates rates[];
matrix x_data(time_step, 3); //3 columns for open, high and low
vector classes_in_data_ = {0, 1}; //unique target variables as they are in the target variable in your training data
int OldNumBars = 0;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
```

The matrix named **x\_data** is the one responsible for the temporary storage of independent variables (features) from the market. This matrix is resized to 3 columns since we trained the model on 3 features (Open, High, and Low), and resized to the number of rows equal to the time step value.

**The time step value must be similar to the one used in creating sequential training data.**

We can make a simple strategy based on signals provided by the model we built.

```
   double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);

      MqlTick ticks;
      SymbolInfoTick(Symbol(), ticks);

      if (signal==1) //if the signal is bullish
       {
          ClosePos(POSITION_TYPE_SELL); //close sell trades when the signal is buy

          if (!PosExists(POSITION_TYPE_BUY)) //There are no buy positions
           {
             if (!m_trade.Buy(min_lot, Symbol(), ticks.ask, 0 , 0)) //Open a buy trade
               printf("Failed to open a buy position err=%d",GetLastError());
           }
       }
      else if (signal==0) //Bearish signal
        {
          ClosePos(POSITION_TYPE_BUY); //close all buy trades when the signal is sell

          if (!PosExists(POSITION_TYPE_SELL)) //There are no Sell positions
           {
            if (!m_trade.Sell(min_lot, Symbol(), ticks.bid, 0 , 0)) //open a sell trade
               printf("Failed to open a sell position err=%d",GetLastError());
           }
        }
      else //There was an error
        return;
```

Now that we have the model loaded up and ready to make predictions, I ran a test from 2020.01.01 to 2024.09.01. Below is the full tester configuration (settings) image.

![](https://c.mql5.com/2/142/tester_configuration__1.png)

Notice I applied the EA on a 4-Hour chart, Instead of the Daily timeframe which the Tesla stock data was collected from. This is because we programmed the strategy and models to kick into action an instant after the new candle is opened but, the daily candle is usually opened when the market is closed hence causing the EA to miss on trading until the next day.

By applying the EA to a lower timeframe (4-Hour timeframe in this case) we ensure that we continuously monitor the market after every 4 hours and perform some trading activities.

_This doesn't affect the data provided to the EA as we applied the CopyRates function to the daily timeframe (trading decisions still depend on the daily chart)_

Below is the Tester's outcome.

![](https://c.mql5.com/2/142/bandicam_2024-09-24_13-46-37-0561__1.gif)

![](https://c.mql5.com/2/142/4143065214049__1.png)

Impressive! The EA produced 90% profitable trades. The AI model was just a Simple RNN.

Now let's see how well LSTM and GRU perform in the same market.

### A combination of Convolutional Neural Network (CNN) and Long-Short Term Memory (LSTM)

Unlike the simple RNN which is incapable when it comes to understanding patterns within long sequences of data or information, the LSTM can understand relationships and patterns in long sequences of information.

LSTMs are often more efficient and accurate than simple RNNs. Let us create a CNN model with LSTM in it and then observe how it fares in the Tesla stock.

Python code

```
from tensorflow.keras.layers import LSTM

# Define the CNN model

model = Sequential()
model.add(Conv1D(filters=16,
                 kernel_size=3,
                 activation='relu',
                 strides=2,
                 padding='causal',
                 input_shape=(time_step, X_train.shape[2])
                )
         )

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=len(np.unique(y)), activation='softmax'))  # For binary classification (e.g., buy/sell signal)

model.summary()
```

Since all RNNs can be implemented the same way, I had to make only one change in the block of code used to create a simple RNN.

After training and validating the model, its accuracy was 53% on the testing data.

```
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step
Classification Report
               precision    recall  f1-score   support

           0       0.67      0.44      0.53       117
           1       0.45      0.68      0.54        78

    accuracy                           0.53       195
   macro avg       0.56      0.56      0.53       195
weighted avg       0.58      0.53      0.53       195
```

In the MQL5 programming language, we can use the same library we used for the simple RNN EA.

```
#resource "\\Files\\lstm+cnn.TSLA.D1.onnx" as uchar onnx_model[]
#resource "\\Files\\lstm+cnn.TSLA.D1.standard_scaler_mean.bin" as double standardization_mean[]
#resource "\\Files\\lstm+cnn.TSLA.D1.standard_scaler_scale.bin" as double standardization_std[]

#include <MALE5\Convolutional Neural Networks(CNNs)\ConvNet.mqh>
#include <MALE5\preprocessing.mqh>

CConvNet cnn;
StandardizationScaler *scaler;
```

The rest of the code is kept the same as in the CNN + RNN EA.

I used the same Tester's settings as before, below was the outcome.

![](https://c.mql5.com/2/142/6045515040757__1.png)

![](https://c.mql5.com/2/142/6045682773543__1.png)

This time the overall trades accuracy is approximately 74%, It is lower than what we got in the previous model but, still outstanding!

### A combination of Convolutional Neural Network (CNN) and Gated Recurrent Unit (GRU)

Just like the LSTM, GRU models are also capable of understanding the relationships between long sequences of information and data despite having a minimalist approach compared to that of the LSTM model.

We can implement it the same as other RNN models, we only make the change in the type of model in the code for building the CNN model architecture.

```
from tensorflow.keras.layers import GRU

# Define the CNN model

model = Sequential()
model.add(Conv1D(filters=16,
                 kernel_size=3,
                 activation='relu',
                 strides=2,
                 padding='causal',
                 input_shape=(time_step, X_train.shape[2])
                )
         )

model.add(MaxPooling1D(pool_size=2))

model.add(GRU(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=len(np.unique(y)), activation='softmax'))  # For binary classification (e.g., buy/sell signal)

model.summary()
```

After training and validating the model, the model achieved an accuracy similar to that of LSTM, 53% accuracy on the testing data.

```
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 41ms/step
Classification Report
               precision    recall  f1-score   support

           0       0.69      0.39      0.50       117
           1       0.45      0.73      0.55        78

    accuracy                           0.53       195
   macro avg       0.57      0.56      0.53       195
weighted avg       0.59      0.53      0.52       195
```

We load the GRU model in ONNX format and its scaler parameters in binary files.

```
#resource "\\Files\\gru+cnn.TSLA.D1.onnx" as uchar onnx_model[]
#resource "\\Files\\gru+cnn.TSLA.D1.standard_scaler_mean.bin" as double standardization_mean[]
#resource "\\Files\\gru+cnn.TSLA.D1.standard_scaler_scale.bin" as double standardization_std[]

#include <MALE5\Convolutional Neural Networks(CNNs)\ConvNet.mqh>
#include <MALE5\preprocessing.mqh>

CConvNet cnn;
StandardizationScaler *scaler;
```

Again the rest of the code is the same as the one used in the simple RNN EA.

After testing the model on the Tester using the same settings, below was the outcome.

![](https://c.mql5.com/2/142/3614567638259__1.png)

![](https://c.mql5.com/2/142/3614622780711__1.png)

The GRU model provided an accuracy of approximately 61%, not as good as the prior two models but, a decent accuracy indeed.

### Final Thoughts

The integration of Convolutional Neural Networks (CNNs) with Recurrent Neural Networks (RNNs) can be a powerful approach to stock market prediction, offering the potential to uncover hidden patterns and temporal dependencies in data. However, this combination is relatively uncommon and comes with certain challenges. One of the key risks is overfitting, especially when applying such sophisticated models to relatively simple problems. Overfitting can cause the model to perform well on training data but fail to generalize to new data.

Additionally, the complexity of combining CNNs and RNNs leads to significant computational costs, particularly if you decide to scale up the model by adding more dense layers or increasing the number of neurons. It is essential to carefully balance model complexity with the resources available and the problem at hand.

Peace out.

Track development of machine learning models and much more discussed in this article series on this [GitHub repo](https://www.mql5.com/go?link=https://github.com/MegaJoctan/MALE5 "https://github.com/MegaJoctan/MALE5").

**Attachments Table**

| File name | File type | Description & Usage |
| --- | --- | --- |
| Experts\\CNN + GRU EA.mq5<br>Experts\\CNN + LSTM EA.mq5<br>Experts\\CNN + RNN EA.mq5 | Expert Advisors | Trading robot for loading the ONNX models and testing the trading strategy in MetaTrader 5. |
| ConvNet.mqh<br>preprocessing.mqh | Include files | \- This file comprises code for loading CNN models saved in ONNX format.<br>\- The Standardization scaler can be found in this file |
| Files\ \*.onnx | ONNX models | Machine learning models discussed in this article in ONNX format |
| Files\\\*.bin | Binary files | Binary files for loading Standardization scaler parameters for each model |
| [Jupyter Notebook\\cnns-rnns.ipynb](https://www.mql5.com/go?link=https://www.kaggle.com/code/omegajoctan/cnns-rnns/notebook%23Convolutional-Neural-Network(CNN)-%2b-Long-Short-Term-Memory(LSTM) "https://www.kaggle.com/code/omegajoctan/cnns-rnns/notebook#Convolutional-Neural-Network(CNN)-+-Long-Short-Term-Memory(LSTM)") | python/Jupyter notebook | All the Python code discussed in this article can be found inside this notebook. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15585.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/15585/attachments.zip "Download Attachments.zip")(342.55 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 04): Tester 101](https://www.mql5.com/en/articles/20917)
- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/473965)**
(1)


![Juan Guirao](https://c.mql5.com/avatar/2023/10/6520fda6-f3b7.jpg)

**[Juan Guirao](https://www.mql5.com/en/users/freenrg)**
\|
13 Oct 2024 at 20:30

These results are surprisingly good. Added to TODO list.


![Gain An Edge Over Any Market (Part V): FRED EURUSD Alternative Data](https://c.mql5.com/2/96/Gain_An_Edge_Over_Any_Market_Part_V___LOGO2.png)[Gain An Edge Over Any Market (Part V): FRED EURUSD Alternative Data](https://www.mql5.com/en/articles/15949)

In today’s discussion, we used alternative Daily data from the St. Louis Federal Reserve on the Broad US-Dollar Index and a collection of other macroeconomic indicators to predict the EURUSD future exchange rate. Unfortunately, while the data appears to have almost perfect correlation, we failed to realize any material gains in our model accuracy, possibly suggesting to us that investors may be better off using ordinary market quotes instead.

![Developing a multi-currency Expert Advisor (Part 11): Automating the optimization (first steps)](https://c.mql5.com/2/78/Developing_a_multi-currency_advisor_4Part_111___LOGO.png)[Developing a multi-currency Expert Advisor (Part 11): Automating the optimization (first steps)](https://www.mql5.com/en/articles/14741)

To get a good EA, we need to select multiple good sets of parameters of trading strategy instances for it. This can be done manually by running optimization on different symbols and then selecting the best results. But it is better to delegate this work to the program and engage in more productive activities.

![MQL5 Wizard Techniques you should know (Part 41): Deep-Q-Networks](https://c.mql5.com/2/96/MQL5_Wizard_Techniques_you_should_know_Part_41__LOGO.png)[MQL5 Wizard Techniques you should know (Part 41): Deep-Q-Networks](https://www.mql5.com/en/articles/16008)

The Deep-Q-Network is a reinforcement learning algorithm that engages neural networks in projecting the next Q-value and ideal action during the training process of a machine learning module. We have already considered an alternative reinforcement learning algorithm, Q-Learning. This article therefore presents another example of how an MLP trained with reinforcement learning, can be used within a custom signal class.

![Risk manager for algorithmic trading](https://c.mql5.com/2/77/Risk_manager_for_algorithmic_trading___LOGO__2.png)[Risk manager for algorithmic trading](https://www.mql5.com/en/articles/14634)

The objectives of this article are to prove the necessity of using a risk manager and to implement the principles of controlled risk in algorithmic trading in a separate class, so that everyone can verify the effectiveness of the risk standardization approach in intraday trading and investing in financial markets. In this article, we will create a risk manager class for algorithmic trading. This is a logical continuation of the previous article in which we discussed the creation of a risk manager for manual trading.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/15585&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068907964863020809)

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