---
title: Data Science and ML (Part 27): Convolutional Neural Networks (CNNs) in MetaTrader 5 Trading Bots — Are They Worth It?
url: https://www.mql5.com/en/articles/15259
categories: Expert Advisors, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T17:28:55.047773
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/15259&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068288454485276580)

MetaTrader 5 / Expert Advisors


_The pooling operation used in convolutional neural networks is a big mistake, and the fact that it works so well is a disaster._

_Geoffrey Hinton_

**Contents**

- [What are Convolutional Neural Networks (CNNS)?](https://www.mql5.com/en/articles/15259#what-are-cnns)

- [Convolutional Layers](https://www.mql5.com/en/articles/15259#what-are-convolutional-layers)

- [Activation Functions](https://www.mql5.com/en/articles/15259#activation-functions-cnn)

- [Pooling Layers](https://www.mql5.com/en/articles/15259#pooling-layers-cnn)

- [Fully Connected Layers](https://www.mql5.com/en/articles/15259#fully-connected-cnn-layers)

- [Dropout Layers](https://www.mql5.com/en/articles/15259#dropout-layers-cnn)

- [Why use Convolutional Neural Networks (CNNs) for Financial Analysis and Trading Applications?](https://www.mql5.com/en/articles/15259#why-use-cnns-for-trading)
- [Making a Convolutional Neural Network (CNN) in Python](https://www.mql5.com/en/articles/15259#making-cnn-in-python)
- [Creating a Convolutional Neural Network (CNN) based Trading Robot](https://www.mql5.com/en/articles/15259#creating-cnn-baded-EA)
- [The Bottom Line](https://www.mql5.com/en/articles/15259#conclusion)

A basic understanding of [Python programming language](https://www.mql5.com/go?link=https://docs.python.org/ "https://docs.python.org/"), [Artificial Neural Networks](https://www.mql5.com/en/articles/11275#:~:text=in%20the%20dataset).-,The%20Output%20layer,-Determining%20the%20size), [Machine learning](https://en.wikipedia.org/wiki/Machine_learning "https://en.wikipedia.org/wiki/Machine_learning") and [ONNX in MQL5](https://www.mql5.com/en/articles/13394) is required to understand the contents of this article fully.

### What are Convolutional Neural Networks (CNNS)?

Convolutional Neural Networks (CNNs) are a class of deep learning algorithms specifically designed to process structured grid-like data, such as images, audio spectrograms, and time-series data. They are particularly well-suited for visual data tasks because they can automatically and adaptively learn spatial hierarchies of features from input data.

CNNs are the extended version of artificial neural networks (ANN). They are predominantly used to extract the feature from the grid-like matrix dataset. For example, visual datasets like images or videos where data patterns play an extensive role.

Convolutional neural networks have several key components such as; Convolutional layers, activation functions, pooling layers, fully connected layers, and dropout layers. To understand CNNs in depth, let us dissect each component and see what it's all about.

> ![convolutional neural network illustration](https://c.mql5.com/2/114/coonvolutional_neural_network.png)

### Convolutional Layers

These are the core building blocks of CNNs, it is where the majority of computation occurs. Convolutional layers are responsible for detecting local patterns in the input data, such as edges in images. This can be achieved through the use of filters (or kernels) that slide over the input data to produce feature maps.

A convolutional layer is a hidden layer that contains several convolution units in a convolutional neural network, that is used for feature extraction.

```
from tensorflow.keras.layers import Conv1D

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X_train.shape[2])))
```

**Filters/Kernels**

Filters (or kernels) are small learnable square matrices (usually of size 3x3, 5x5, etc.) that slide over the input data to detect local patterns.

**How do they work?**

They operate by moving across the input data and then perform element-wise multiplication between the filter values and the input values within the filter's current receptive field, followed by summing the results. This operation is what's called **convolution**.

During training, the network learns the optimal values of the filters. In early layers, filters typically learn to detect simple features like edges and textures meanwhile in deeper layers, filters can detect more complex patterns such as shapes and objects.

Consider a simple 3x3 filter and a 5x5 input image. The filter slides over the image, computing the convolution operation to produce a feature map.

![conv-net kernel](https://c.mql5.com/2/114/iclh-kernel-convolutional-neural-networks.png)

**Stride**

This is another feature found in the convolution layer. The stride is the step size by which the filter moves across the input data. It determines how much the filter shifts at each step during the convolution process.

**How do they work?**

Stride of 1, the filter moves one unit at a time, resulting in a highly overlapping and detailed feature map. This produces a larger output feature map.

Stride of 2 or more, the filter skips units, resulting in a less detailed but smaller output feature map. This reduces the spatial dimensions of the output, effectively downsampling the input.

For example, If you have a 3x3 filter and a 5x5 input image with a stride of 1, the filter will move one pixel at a time, producing a 3x3 output feature map. With a stride of 2, the filter will move two pixels at a time, producing a 2x2 output feature map.

**Padding**

Padding involves adding extra pixels (usually zeros) around the border of the input data. This ensures that the filter fits properly and controls the spatial dimensions of the output feature map.

**Types of padding**

According to [Keras](https://www.mql5.com/go?link=https://keras.io/api/layers/convolution_layers/convolution1d/ "https://keras.io/api/layers/convolution_layers/convolution1d/"), there are three types of padding. ( _case-sensitive_)

1. **valid** \- no padding will be applied,
2. **same** \- pads the input so the output size matches the input size when strides=1.
3. **causal** -  used for temporal data to ensure the output at time step 𝑡 does not depend on future inputs.

Padding helps in preserving the spatial dimensions of the input data. Without padding, the output feature map shrinks with each convolutional layer, which might result in losing important edge information.

By adding padding, the network can learn edge features effectively and maintain the spatial resolution of the input.

Consider a 3x3 filter and a 5x5 input image. With valid padding (no padding), the output feature map will be 3x3. With the same padding, you might add a border of zeros around the input, making it 7x7. The output feature map will then be 5x5, preserving the input dimensions.

Below is the code for a convolution layer in Python.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D

model = Sequential()
model.add(Conv1D(filters=64,
                 kernel_size=3,
                 activation='relu',
                 strides=2,
                 padding='causal',
                 input_shape=(window_size, X_train.shape[2])
                )
         )
```

### Activation Functions

As discussed in the article [Neural Networks Demystified,](https://www.mql5.com/en/articles/11275#:~:text=What%20is%20an%20Activation%20Function%3F) an activation function is a mathematical function that takes an input and processes an output.

The Activation function is applied element-wise to introduce non-linearity into the model. Commonly used activation functions in CNNs include [ReLU](https://www.mql5.com/en/articles/11275#:~:text=01%3A-,RELU,-RELU%20stands%20for) (Rectified Linear Unit), [Sigmoid](https://www.mql5.com/en/articles/11275#:~:text=02%3A-,Sigmoid,-Sounds%20familiar%20right), and [Tan](https://www.mql5.com/en/articles/11275#:~:text=03%3A-,TanH,-The%20Hyperbolic%20Tangent) H.

### Pooling Layers

Also known as desample layers, These layers are an essential part of CNNs as they are responsible for reducing the spatial dimension of the input data in terms of width and height while retaining the most important information.

**How do they work?**

Firstly, they divide the input data into overlapping regions or windows, then they apply an aggregation function such as Max pooling or Average pooling on each window to obtain a single value.

**Max pooling** takes the maximum value from a set of values within a filter region. It reduces the spatial dimensions of the data, which helps in reducing the computational load and the number of parameters.

Python

```
from tensorflow.keras.layers import Conv1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
```

```
MaxPooling1D(pool_size=2)
```

This layer takes the maximum value from each 2-element window.

![max pooling](https://c.mql5.com/2/114/max_pooling__1.png)

[More Information](https://www.mql5.com/go?link=https://arxiv.org/pdf/2009.07485 "https://arxiv.org/pdf/2009.07485") [.](https://www.mql5.com/go?link=https://arxiv.org/pdf/2009.07485 "https://arxiv.org/pdf/2009.07485")

**Average pooling** takes the average value from a set of values within a filter region. Less commonly used than max pooling.

Python

```
from tensorflow.keras.layers import Conv1D, AveragePooling1D

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X_train.shape[2])))
model.add(AveragePooling1D(pool_size=2))
```

```
AveragePooling1D(pool_size=2)
```

This layer takes the average value from each 2-element window.

![average pooling](https://c.mql5.com/2/114/average_pooling__1.png)

[More Information](https://www.mql5.com/go?link=https://arxiv.org/pdf/2009.07485 "https://arxiv.org/pdf/2009.07485") [.](https://www.mql5.com/go?link=https://arxiv.org/pdf/2009.07485 "https://arxiv.org/pdf/2009.07485")

**Why use 1D Convolutional Layer?**

There are Conv1D, Conv2D, and Conv3D layers for CNNs. The 1D convolution layer is one suitable for this type of problem since it is designed for one dimensional data, making it suitable for sequential or Time series data. Other Convolutional layers such as the Conv2D and the Conv3D are too complex for this kind of problem.

### Fully Connected Layers

Neurons in a fully connected layer have connections to all activations in the previous layer. These layers are typically used towards the end of the network to perform classification or regression based on the features extracted by convolutional and pooling layers.

```
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=len(np.unique(y)), activation='sigmoid'))  # For binary classification (e.g., buy/sell signal)

model.summary()
```

The **Flatten Layer** converts the 1D pooled feature map into a 1D vector, so it can be fed into the fully connected (dense) layers.

Dense Layers (Dense) are fully connected layers that are used for making final decisions based on the features extracted by the convolution and pooling layers. **Dense layers are essentially the core component of traditional artificial neural networks (ANNs).**

![dense layer in a CNN highlighted](https://c.mql5.com/2/114/dense_layer_highlighted.png)

### Dropout Layers

The Dropout layer acts as a mask, eliminating some neurons' contributions to the subsequent layer while maintaining the functionality of all other neurons. If we apply a Dropout layer to the input vector, some of its features are eliminated; however, if we apply it to a hidden layer, some hidden neurons are eliminated.

Since they avoid overfitting the training data, dropout layers are crucial in the training of CNNs. If they are absent, the first set of training samples has an excessively large impact on learning. As a result, traits that only show in later samples or batches would not be learned.

```
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=len(np.unique(y)), activation='sigmoid'))  # For binary classification (e.g., buy/sell signal)

model.summary()
```

### Why use Convolutional Neural Networks (CNNs) for Financial Analysis and Trading Appplications?

CNNs are widely used in image and video processing applications since that is what they are designed for. If you look at the above explanations you may be able to notice that I refer to using CNNs when working with image classifications and stuff.

Using Convolutional Neural Networks (CNNs) for tabular data, such as financial analysis, might seem unconventional compared to using other neural network types such as Feed Forward Neural Networks(FFNN), [Recurrent Neural Networks (RNNs)](https://www.mql5.com/en/articles/15114), [Long Short-Term Memory (LSTMs), and Gated Recurrent Units (GRUs)](https://www.mql5.com/en/articles/15182). However, there are several reasons and potential benefits outlined below for employing CNNs in this context.

**01\. CNNs are excellent at automatically extracting local patterns from data**

Financial data often exhibit local temporal patterns, such as trends and seasonalities. By treating the data as a time series. CNNs can learn these local dependencies and interactions between features, which might be missed by traditional methods.

**02\. They can learn hierarchical features**

Multiple layers in CNNs enable them to learn complex feature hierarchies. Early layers can detect simple patterns, while deeper layers can combine these simple patterns into more complex and abstract representations. This hierarchical feature learning can be beneficial in capturing intricate patterns in financial data.

**03\. They can be robust to noise and redundant features**

As we know, financial datasets contain noisy and redundant data. The pooling layers in CNNs help in making the model robust to slight variations and noise as they down-sample the feature maps and reduce the influence of minor fluctuations.

**04: CNNs can handle multivariate time series well**

Financial data often involves multiple correlated time series such as different stock prices and trading volumes. CNNs can effectively capture the interactions and dependencies between these multiple time series, making them suitable for multivariate [time series forecasting](https://www.mql5.com/en/articles/15013).

**05\. They are computationally efficient for high-dimensional data**

Financial data can have a high dimensionality (many features). Through weight sharing and local connectivity CNNs are more computationally efficient than fully connected neural networks, making them scalable to high-dimensional data.

**06\. CNNs enable end-to-end learning**

CNNs can learn directly from raw data and produce predictions without the need for manual [feature engineering](https://www.mql5.com/en/articles/15013#feature-engineering-for-time-series-forecasting). This end-to-end learning approach can simplify the modeling process and potentially yield better performance by letting the model discover the most relevant features.

**07\. CNNs apply convolution operations that can be advantageous for certain types of data.**

Convolution operations can detect and enhance important signals within the data, such as sudden changes or specific patterns. This is particularly useful in financial analysis, where detecting sudden market shifts or patterns can be critical.

Now that we have valid reasons to use CNNs in trading applications, let us create one and train it, then we'll see how we can use a CNN in a Meta Trader 5 Expert Advisor(EA).

### Making a Convolutional Neural Network (CNN) in Python

This involves several steps which are.

1. Collecting the data
2. Preparing data for a CNN model
3. Training a CNN model
4. Saving a CNN model to ONNX format

**01: Collecting the Data**

Using the [data made for Time series forecasting](https://www.mql5.com/go?link=https://www.kaggle.com/datasets/omegajoctan/forex-timeseries-ohlc "https://www.kaggle.com/datasets/omegajoctan/forex-timeseries-ohlc") we used in the previous articles.

![time series forecasting dataset](https://c.mql5.com/2/114/dataset.gif)

Now that we know that Convolutional Neural Networks (CNNs) are good at detecting patterns within high-dimensional data, without complicating the model we can choose some of the features I believe might have plenty of patterns that the CNN model can detect.

Python code

```
open_price = df['TARGET_OPEN']
close_price = df['TARGET_CLOSE']

# making the target variable

target_var = []
for i in range(len(open_price)):
    if close_price[i] > open_price[i]: # if the price closed above where it opened
        target_var.append(1) # bullish signal
    else:
        target_var.append(0) # bearish signal


new_df = pd.DataFrame({
    'OPEN': df['OPEN'],
    'HIGH': df['HIGH'],
    'LOW': df['LOW'],
    'CLOSE': df['CLOSE'],
    'TARGET_VAR': target_var
})

print(new_df.shape)
```

Shortly after preparing the target variable based on the TARGET\_OPEN and TARGET\_CLOSE which are open and close values respectively, collected one bar forward. We created a mini dataset version named **new\_df** which only had 4 independent variables OPEN, HIGH, and LOW values, and one dependent variable named TARGET\_VAR.

**02: Preparing Data for A CNN Model**

Firstly, we have to pre-process the input data by reshaping and aligning it into windows. This is very crucial when working with tabular data in CNNs, here is why.

Since the trading data is sequential, patterns often emerge over a series of time steps rather than a single point in time. By creating overlapping windows of data we can, **capture temporal dependencies and provide context to the CNN model.**

Also, CNNs expect input data to be in a specific shape. For 1D convolutional layers, the **input shape** typically needs to be **(number of windows, window size, number of features).** This shape resembles the one we use in time series analysis using Recurrent Neural Networks (RNNs) in the previous article. The preprocessing procedure we are about to do ensures that the data is in this format, making it suitable for a CNN model input.

```
# Example data preprocessing function

def preprocess_data(df, window_size):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i+window_size, :-1].values)
        y.append(df.iloc[i+window_size, -1])
    return np.array(X), np.array(y)

window_size = 10

X, y = preprocess_data(new_df, window_size)
print(f"x_shape = {X.shape}\ny_shape = {y.shape}")
```

Outputs

```
x_shape = (990, 10, 4)
y_shape = (990,)
```

Since our data was collected on a daily timeframe, the window size of 10 indicates that we will be training the CNN model to understand patterns within 10 days.

Then we have to split the data into training and testing samples.

```
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

print(f"x_train\n{X_train.shape}\nx_test\n{X_test.shape}\n\ny_train {y_train.shape} y_test {y_test.shape}")
```

Outputs

```
x_train
(792, 10, 4)
x_test
(198, 10, 4)

y_train (792,) y_test (198,)
```

Lastly we have to [one-hot-encode](https://www.mql5.com/en/articles/11858#one-hot-encoding-matrix) the target variable for this classification problem task.

```
from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

print(f"One hot encoded\n\ny_train {y_train_encoded.shape}\ny_test {y_test_encoded.shape}")
```

Outputs

```
One hot encoded

y_train (792, 2)
y_test (198, 2)
```

**03: Training a CNN model**

This is where most work gets done.

```
# Defining the CNN model

model = Sequential()
model.add(Conv1D(filters=64,
                 kernel_size=3,
                 activation='relu',
                 strides=2,
                 padding='causal',
                 input_shape=(window_size, X_train.shape[2])
                )
         )

model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=len(np.unique(y)), activation='softmax'))  # For binary classification (buy/sell signal)

model.summary()

# Compiling the model

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train_encoded, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

plt.figure(figsize=(7.5, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig("training loss cuver-cnn-clf.png")
plt.show()
```

Outputs

```
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv1d_2 (Conv1D)               │ (None, 5, 64)          │           832 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d_2 (MaxPooling1D)  │ (None, 2, 64)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_2 (Flatten)             │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 100)            │        12,900 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 100)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (Dense)                 │ (None, 2)              │           202 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

Training stopped at the 34-th epoch.

```
40/40 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5105 - loss: 0.6875 - val_accuracy: 0.4843 - val_loss: 0.6955
Epoch 32/100
40/40 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5099 - loss: 0.6888 - val_accuracy: 0.5283 - val_loss: 0.6933
Epoch 33/100
40/40 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4636 - loss: 0.6933 - val_accuracy: 0.5283 - val_loss: 0.6926
Epoch 34/100
40/40 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5070 - loss: 0.6876 - val_accuracy: 0.5346 - val_loss: 0.6963
```

![CNN training-validation loss curve](https://c.mql5.com/2/114/training_loss_cuver-cnn-clf.png)

The model was approximately 57% of the time accurate on out-of-sample predictions.

```
y_pred = model.predict(X_test)

classes_in_y = np.unique(y)
y_pred_binary = classes_in_y[np.argmax(y_pred, axis=1)]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion-matrix CNN")  # Display the heatmap

print("Classification Report\n",
      classification_report(y_test, y_pred_binary))
```

Outputs

```
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
Classification Report
               precision    recall  f1-score   support

           0       0.53      0.24      0.33        88
           1       0.58      0.83      0.68       110

    accuracy                           0.57       198
   macro avg       0.55      0.53      0.50       198
weighted avg       0.55      0.57      0.52       198
```

Our CNN model is good enough for an Expert Advisor. But, before we can start coding an EA, let us save the CNN model we have trained in ONNX format.

**04: Saving a CNN model to ONNX format.**

The process is fairly simple, We have to save the CNN model in the .onnx format, and the scaling technique parameters in binary files.

```
import tf2onnx

onnx_file_name = "cnn.EURUSD.D1.onnx"

spec = (tf.TensorSpec((None, window_size, X_train.shape[2]), tf.float16, name="input"),)
model.output_names = ['outputs']

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model to a file
with open(onnx_file_name, "wb") as f:
    f.write(onnx_model.SerializeToString())


# Save the mean and scale parameters to binary files
scaler.mean_.tofile(f"{onnx_file_name.replace('.onnx','')}.standard_scaler_mean.bin")
scaler.scale_.tofile(f"{onnx_file_name.replace('.onnx','')}.standard_scaler_scale.bin")
```

### Creating a Convolutional Neural Network (CNN) based Trading Robot

Inside an Expert Advisor, the first thing we have to do is to include the ONNX-formatted model and the [Standard Scaler](https://www.mql5.com/go?link=https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html") binary files as resources.

MQL5 \| ConvNet EA.mq5

```
#resource "\\Files\\cnn.EURUSD.D1.onnx" as uchar onnx_model[]
#resource "\\Files\\cnn.EURUSD.D1.standard_scaler_scale.bin" as double scaler_stddev[]
#resource "\\Files\\cnn.EURUSD.D1.standard_scaler_mean.bin" as double scaler_mean[]
```

We have to initialize them both, the scaler and the onnx model.

```
#include <MALE5\Convolutioal Neural Networks(CNNs)\Convnet.mqh>
#include <MALE5\preprocessing.mqh>

CConvNet cnn;
StandardizationScaler scaler;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

input group "cnn";
input uint cnn_data_window = 10;
//this value must be the same as the one used during training in a python script

vector classes_in_y = {0,1}; //we have to assign the classes manually | it is essential that their order is preserved as they can be seen in python code, HINT: They are usually in ascending order
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   if (!cnn.Init(onnx_model)) //Initialize the ONNX model
     return INIT_FAILED;

//--- Initializing the scaler with values loaded from binary files

   scaler = new StandardizationScaler(scaler_mean, scaler_stddev); //load the scaler

   return(INIT_SUCCEEDED);
  }
```

That is enough to get the model up and running. Let us make the function to extract data similarly to the way the independent variables were used during training. We used four variables OHLC values from the previous closed bar to 10 bars prior which was the **window size** the timeframe must be preserved (A daily timeframe).

```
input group "cnn";
input uint cnn_data_window = 10;
//this value must be the same as the one used during training in a python script

input ENUM_TIMEFRAMES timeframe = PERIOD_D1;
input int magic_number = 1945;
input int slippage = 50;
```

```
matrix GetXVars(int bars, int start_bar=1)
 {
   vector open(bars),
          high(bars),
          low(bars),
          close(bars);

//--- Getting OHLC values

   open.CopyRates(Symbol(), timeframe, COPY_RATES_OPEN, start_bar, bars);
   high.CopyRates(Symbol(), timeframe, COPY_RATES_HIGH, start_bar, bars);
   low.CopyRates(Symbol(), timeframe, COPY_RATES_LOW, start_bar, bars);
   close.CopyRates(Symbol(), timeframe, COPY_RATES_CLOSE, start_bar, bars);

//---

   matrix data(bars, 4); //we have 10 inputs from cnn | this value is fixed

//--- adding the features into a data matrix

   data.Col(open, 0);
   data.Col(high, 1);
   data.Col(low, 2);
   data.Col(close, 3);

   return data;
 }
```

Now that we have a function to collect the independent variables, we can finalize our trading strategy.

```
void OnTick()
  {
//---

   if (NewBar()) //Trade at the opening of a new candle
    {
      matrix input_data_matrix = GetXVars(cnn_data_window); //get data for the past 10 days(default)
      input_data_matrix = scaler.transform(input_data_matrix); //applying StandardSCaler to the input data

      int signal = cnn.predict_bin(input_data_matrix, classes_in_y); //getting trade signal from the RNN model

      Comment("Signal==",signal);

   //---

      MqlTick ticks;
      SymbolInfoTick(Symbol(), ticks);

      if (signal==1) //if the signal is bullish
       {
          if (!PosExists(POSITION_TYPE_BUY)) //There are no buy positions
           {
             if (!m_trade.Buy(lotsize, Symbol(), ticks.ask, 0, 0)) //Open a buy trade
               printf("Failed to open a buy position err=%d",GetLastError());

             ClosePosition(POSITION_TYPE_SELL); //close opposite trade
           }
       }
      else if (signal==0) //Bearish signal
        {
          if (!PosExists(POSITION_TYPE_SELL)) //There are no Sell positions
            if (!m_trade.Sell(lotsize, Symbol(), ticks.bid, 0, 0)) //open a sell trade
               printf("Failed to open a sell position err=%d",GetLastError());

               ClosePosition(POSITION_TYPE_BUY);
        }
      else //There was an error
        return;
    }
  }
```

The strategy is simple. Upon receiving a particular signal, let's say a buy signal we open a buy trade with no stop loss and take profit values, we then close the opposite signal and vice versa for a sell signal.

Finally, I tested this strategy on a symbol it was trained on which is **EURUSD**, for ten years. **From 2014.01.01 to 2024.05.27** On a **4-Hour chart** **on Open Prices** of every bar.

![convNet EA strategy tester config](https://c.mql5.com/2/114/Tester_config.png)

The results from the Strategy tester outcome were outstanding.

![ConvNet EA strategy tester report](https://c.mql5.com/2/114/Tester_report.png)

![ConvNet EA tester graph](https://c.mql5.com/2/114/tester_graph.png)

The EA made accurate predictions 58% of all time as a result the CNN-based EA made $503 net profit.

### The Bottom Line

Despite being made specifically for image and video processing, when adopted to handle tabular data such as the forex data we gave it, Convolutional Neural Networks(CNN) can do a decent job detecting patterns and use them to make predictions in the forex market.

As can be seen from the strategy tester report, the CNN-based EA has made decent predictions. I bet many traditional models designed for tabular data such as Linear regression, Support Vector Machine, Naive Bayes, etc. cannot achieve this predictive accuracy considering the CNN model was given only 4 independent variables (OHLC). In my experience, not many models can become this good given a few variables.

Best regards.

Track development of machine learning models and much more discussed in this article series on this [GitHub repo](https://www.mql5.com/go?link=https://github.com/MegaJoctan/MALE5 "https://github.com/MegaJoctan/MALE5").

**Attachments Table**

| File name | File type | Description & Usage |
| --- | --- | --- |
| ConvNet EA.mq5 | Expert Advisor | Trading robot for loading the CNN model in ONNX format and testing the final trading strategy in MetaTrader 5. |
| cnn.EURUSD.D1.onnx | ONNX | CNN model in ONNX format. |
| cnn.EURUSD.D1.standard\_scaler\_mean.bin  <br> cnn.EURUSD.D1.standard\_scaler\_scale.bin | Binary files | Binary files for the Standardization scaler |
| preprocessing.mqh | An Include file | A library which consists of the Standardization Scaler |
| ConvNet.mqh | An Include file | A library for loading and deploying CNN model in ONNX format |
| [cnn-for-trading-applications-tutorial.ipynb](https://www.mql5.com/go?link=https://www.kaggle.com/code/omegajoctan/cnn-for-trading-applications-tutorial "https://www.kaggle.com/code/omegajoctan/rnns-for-forex-forecasting-tutorial/notebook") | Python Script/Jupyter Notebook | Consists all the python code discussed in this article |

**Sources & References**

- Convolutional Neural Network-based a novel Deep Trend Following Strategy for Stock Market Trading ( [https://ceur-ws.org/Vol-3052/paper2.pdf](https://www.mql5.com/go?link=https://ceur-ws.org/Vol-3052/paper2.pdf "https://ceur-ws.org/Vol-3052/paper2.pdf"))
- What are Convolutional Neural Networks (CNNs)? ( [https://youtu.be/QzY57FaENXg](https://www.mql5.com/go?link=https://youtu.be/QzY57FaENXg "https://youtu.be/QzY57FaENXg"))
- Converting tabular data into images for deep learning with convolutional neural networks ( [https://www.nature.com/articles/s41598-021-90923-y](https://www.mql5.com/go?link=https://www.nature.com/articles/s41598-021-90923-y "https://www.nature.com/articles/s41598-021-90923-y"))
- Image kernels ( [https://setosa.io/ev/image-kernels/](https://www.mql5.com/go?link=https://setosa.io/ev/image-kernels/ "https://setosa.io/ev/image-kernels/"))
- Pooling Methods in Deep Neural Networks, a Review( [https://arxiv.org/pdf/2009.07485](https://www.mql5.com/go?link=https://arxiv.org/pdf/2009.07485 "https://arxiv.org/pdf/2009.07485"))

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15259.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/15259/attachments.zip "Download Attachments.zip")(132.38 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/470161)**
(2)


![npats2007](https://c.mql5.com/avatar/avatar_na2.png)

**[npats2007](https://www.mql5.com/en/users/npats2007)**
\|
29 Jan 2025 at 16:47

**MetaQuotes:**

Published article [Machine Learning and Data Science (Part 27): convolutional neural networks (CNNs) in trading robots for MetaTrader 5](https://www.mql5.com/ru/articles/15259):

Author: [Omega J Msigwa](https://www.mql5.com/ru/users/omegajoctan "omegajoctan")

5.5 trades per year on H4 is not enough. Very little.

![Silk Road Trading LLC](https://c.mql5.com/avatar/2025/5/68239006-fc9d.png)

**[Ryan L Johnson](https://www.mql5.com/en/users/rjo)**
\|
29 Jan 2025 at 21:31

This is the most concise explanation of CNN applied to trading that I've ever seen and for the most part, in plain language and diagrams. Then it's reduced into MQL5 code. Note that the code is not limited to the H4 timeframe.

Well done, Sir!👍

![Combine Fundamental And Technical Analysis Strategies in MQL5 For Beginners](https://c.mql5.com/2/85/Combine_Fundamental_And_Technical_Analysis_Strategies_in_MQL5_For_Beginners___LOGO.png)[Combine Fundamental And Technical Analysis Strategies in MQL5 For Beginners](https://www.mql5.com/en/articles/15293)

In this article, we will discuss how to integrate trend following and fundamental principles seamlessly into one Expert Advisors to build a strategy that is more robust. This article will demonstrate how easy it is for anyone to get up and running building customized trading algorithms using MQL5.

![MQL5 Wizard Techniques you should know (Part 28): GANs Revisited with a Primer on Learning Rates](https://c.mql5.com/2/85/MQL5_Wizard_Techniques_you_should_know_Part_28____LOGO.png)[MQL5 Wizard Techniques you should know (Part 28): GANs Revisited with a Primer on Learning Rates](https://www.mql5.com/en/articles/15349)

The Learning Rate, is a step size towards a training target in many machine learning algorithms’ training processes. We examine the impact its many schedules and formats can have on the performance of a Generative Adversarial Network, a type of neural network that we had examined in an earlier article.

![Hybridization of population algorithms. Sequential and parallel structures](https://c.mql5.com/2/73/Hybridization_of_population_algorithms_Series_and_parallel_circuit___LOGO.png)[Hybridization of population algorithms. Sequential and parallel structures](https://www.mql5.com/en/articles/14389)

Here we will dive into the world of hybridization of optimization algorithms by looking at three key types: strategy mixing, sequential and parallel hybridization. We will conduct a series of experiments combining and testing relevant optimization algorithms.

![Building A Candlestick Trend Constraint Model (Part 6): All in one integration](https://c.mql5.com/2/85/Building_A_Candlestick_Trend_Constraint_Model_Part_6___LOGO__1.png)[Building A Candlestick Trend Constraint Model (Part 6): All in one integration](https://www.mql5.com/en/articles/15143)

One major challenge is managing multiple chart windows of the same pair running the same program with different features. Let's discuss how to consolidate several integrations into one main program. Additionally, we will share insights on configuring the program to print to a journal and commenting on the successful signal broadcast on the chart interface. Find more information in this article as we progress the article series.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/15259&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068288454485276580)

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