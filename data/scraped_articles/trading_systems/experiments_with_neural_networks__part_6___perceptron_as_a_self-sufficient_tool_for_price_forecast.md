---
title: Experiments with neural networks (Part 6): Perceptron as a self-sufficient tool for price forecast
url: https://www.mql5.com/en/articles/12515
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:26:18.793135
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/12515&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070283441024471849)

MetaTrader 5 / Trading systems


### Introduction

Perceptron is a machine learning technique that can be used to predict market prices. It is a useful tool for traders and investors striving to get a price forecast.

### General concepts

Perceptron is a simple neural network consisting of one or more neurons that accept input data, process it and provide output. It was developed by Frank Rosenblatt in 1957 and has since found wide application in various fields, including financial analysis and stock market forecasting.

Perceptron can be used to solve classification and regression problems, including price prediction. In the simplest case, a perceptron consists of a single neuron that takes multiple input variables and produces a single output signal. For example, in order to predict the price on the Forex market, you can use the following input data: exchange rate, trading volume, consumer price index and other factors. After processing this data, the neuron generates an output signal, which is a forecast of the currency pair.

_**Perceptron operation principle**_

Perceptron works based on the supervised learning principle. This means that perceptron is trained on historical data in order to determine the relationships between various market factors and prices. This data is used to tune the weights of the neuron, which determine the importance of each input factor for predicting stock prices.

The perceptron can operate in learning and prediction modes. In the training mode, the perceptron takes historical data and real prices in the Foreх market as input, and then adjusts its weights in such a way as to minimize the prediction error.

**_Benefits of using a perceptron to predict Forex prices_**

1. Using the perceptron to predict Forex prices has several advantages. First, the perceptron is able to adapt to changes in the market and adjust its forecasts according to new data. This makes it more efficient than traditional data analysis methods such as statistical analysis and time series, which cannot always adapt to market changes.
2. Secondly, the perceptron can work with a large number of input data, which makes it possible to take into account many different factors that affect prices. This can lead to more accurate price predictions than traditional data analysis methods.
3. Thirdly, the perceptron can be trained on large amounts of data, which allows it to use a lot of historical data for training and price prediction.

However, using the perceptron to predict prices also has some disadvantages. First, the perceptron may be susceptible to spikes or errors in the data, which can lead to inaccurate price predictions. Secondly, training a perceptron requires a large amount of historical data. If the historical data is not sufficiently representative of the current market situation, then the perceptron predictions may be inaccurate.

Besides, there may be an issue of overfitting when the perceptron becomes too sensitive to historical data and cannot adapt to new market changes. To combat this problem, various regularization techniques can be used, such as L1 and L2 regularization, which help control neuron weights and prevent overfitting.

The Perceptron can be used in combination with other forecasting methods such as Autoregressive Model (ARIMA) or Exponential Smoothing to get more accurate and reliable predictions. For example, you can use a perceptron to predict a long-term price trend, and ARIMA or Exponential Smoothing for short-term forecasts.Keep in mind that the historical data used to train the Perceptron may not match current market conditions. In such cases, the prediction results may be inaccurate. Therefore, the model should be updated regularly so that it can adequately reflect market changes.

### Perceptron parameters for optimization

A perceptron is one of the simplest types of neural networks, which consists of an input layer, hidden layers and an output layer. It can be used for various tasks such as classification, regression or image processing. However, in order for the perceptron to work effectively, it is necessary to choose its parameters correctly.

Perceptron parameters are values that define its structure and behavior. They include the number of hidden layers, the number of neurons in each layer, the activation function, the learning rate and many more. Properly tuned parameters allow the perceptron to get the best results.

**_Here are a few perceptron parameters that can be optimized:_**

_**Number of hidden layers**_

The number of hidden layers determines the complexity of the model. If the model is too simple, then it may not be up to the task, and if it is too complex, then overfitting may occur. Therefore, the number of hidden layers should be chosen optimally, based on the problem to be solved.

**_Number of neurons in each layer_**

The number of neurons in each layer also affects the complexity of the model. A large number of neurons can increase the accuracy of predictions, but at the same time increase the training time. The number of neurons should be optimal for a particular task.

Below is an example of regulating the number of neurons in the input, hidden and output layers. NeuralNets library is used here:

```
int OnInit()
  {

// set the number of neurons in the input, hidden and output layers
int n_inputs = 2;
int n_hidden = 3;
int n_outputs = 1;

// create a perceptron object
CNeuralNet ann;

// add layers
ann.AddLayer(n_inputs);
ann.AddLayer(n_hidden, CNeuralNet::TANH);
ann.AddLayer(n_outputs, CNeuralNet::TANH);

// set learning parameters
ann.SetLearningRate(0.1);
ann.SetMomentum(0.9);
ann.SetMaxEpochs(1000);
ann.SetDesiredAccuracy(90);

// create arrays to store input and target values
double inputs[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
double targets[] = {0, 1, 1, 0};

// train the perceptron
ann.Train((double*)inputs, targets, 4);

// test the perceptron
double output;
ann.Compute((double*)inputs[0], output);
Print("0 XOR 0 = ", output);
ann.Compute((double*)inputs[1], output);
Print("0 XOR 1 = ", output);
ann.Compute((double*)inputs[2], output);
Print("1 XOR 0 = ", output);
ann.Compute((double*)inputs[3], output);
Print("1 XOR 1 = ", output);

}
```

In this example, we create a perceptron with two input neurons, three hidden neurons and one output neuron. We also set training parameters such as training rate, moment and maximum epochs. Next, create arrays to store input and target values, as well as train the perceptron on that data. After testing the perceptron on four different inputs, display the results on the screen.

**_Activation function_**

The activation function determines how the neuron should respond to input. There are many activation functions such as sigmoid, ReLU and hyperbolic tangent. The choice of activation function also depends on the problem to be solved.

Below is an example of using different activation functions:

```
int OnInit()
  {

// set the number of neurons in the input and output layers
int n_inputs = 2;
int n_outputs = 1;

// create a perceptron object
CNeuralNet ann;

// add layers
ann.AddLayer(n_inputs);
ann.AddLayer(3, CNeuralNet::TANH);
ann.AddLayer(n_outputs, CNeuralNet::SIGMOID);

// set learning parameters
ann.SetLearningRate(0.1);
ann.SetMomentum(0.9);
ann.SetMaxEpochs(1000);
ann.SetDesiredAccuracy(90);

// create arrays to store input and target values
double inputs[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
double targets[] = {0, 1, 1, 0};

// train the perceptron
ann.Train((double*)inputs, targets, 4);

// test the perceptron
double output;
ann.Compute((double*)inputs[0], output);
Print("0 XOR 0 = ", output);
ann.Compute((double*)inputs[1], output);
Print("0 XOR 1 = ", output);
ann.Compute((double*)inputs[2], output);
Print("1 XOR 0 = ", output);
ann.Compute((double*)inputs[3], output);
Print("1 XOR 1 = ", output);

}
```

In this example, we add a hidden layer of three neurons and select the "tanh" activation function for a hidden layer and "sigmoid" for an output layer.

**_Training rate_**

The training rate determines how fast the neural network will change its weights. Too high training rate can lead to overflow, while too low value may lead to a lengthy training. You need to choose the training rate that is optimal for a particular task.

**_Regularization_**

Regularization is a method used to prevent overfitting. It consists in adding additional terms to the error function that penalize the model for having too large weights. Regularization reduces the spread of predictions and improves the generalizing ability of the model.

**_Weight initialization_**

Weight initialization is the initial setting of weights for each neuron in the perceptron. Incorrectly initialized weights can cause the model to converge to the local minimum of the error function rather than the global minimum. Therefore, it is necessary to choose the correct method for initializing the weights.

**_Batch size_**

The batch size determines how many data samples will be used in one training iteration. A batch size that is too small can slow down the learning process, while a batch size that is too large can lead to memory overflow. Choose the batch size that is optimal for a particular task.

**_Optimizer_**

An optimizer is an algorithm that is used to update the weights of a model during training. There are many optimizers such as stochastic gradient descent, Adam and RMSprop. Each optimizer has its advantages and disadvantages, and the choice of the optimal one depends on the task.

In general, the optimal parameters of the perceptron depend on the problem to be solved. It is necessary to experiment with different parameter values to find the optimal set for a particular task. Machine learning is the process of iteratively improving a model, and properly tuned parameters are key to achieving better results.

### Passing indicators and prices to a perceptron for market analysis

Indicators are mathematical equations used to analyze the market and help identify trends, entry and exit points, as well as support and resistance levels. Some of the most common indicators that can be used in the perceptron to analyze the Forex market include:

- Moving Average;
- Relative Strength Index (RSI);
- Stochastic Oscillator;
- MACD (Moving Average Convergence Divergence).

Passing the closing price and indicators to the perceptron allows the model to take into account various aspects of market analysis and create more accurate price predictions. For example, a model might use a moving average to determine the overall market trend and then use a stochastic oscillator to determine a market entry point.

However, passing a large number of indicators to a perceptron can lead to a data redundancy problem. Data redundancy can lead to model overfitting and low generalization ability. Therefore, it is necessary to choose the most significant indicators for a specific task of market analysis.

In addition, data transfer to the perceptron requires proper data preprocessing. For example, if the data contains missing values, then you need to solve this problem, for example, fill in the missing values with average values or remove rows with missing values.

It is necessary to choose the optimal parameters for the perceptron so that the model can best train and predict prices. Some of the main parameters that need to be optimized include:

1. The number of neurons in the hidden layer;
2. Neuron activation function;
3. Number of learning epochs;
4. The size of data mini batches for training.

The selection of optimal parameters can be done by trial and error or by using optimization algorithms such as a genetic algorithm or a gradient-based optimization method.

### Examples and practical application

Here we will consider an example of an EA based on a simple perceptron passing the distance between two Moving Average indicators as inputs. Pass the distance between the Moving Average indicator with the value of 1 and the Moving Average indicator with the value of 24. Use exponential Moving Averages with closing by CLOSE, but first normalize these values by converting them to points.

Use the distance on the candles 1, 4, 7, 10 (4 parameters) as inputs. At the perceptron output, we get two values - open a buy position and open a sell position. These conditions are not standard ones. They are given as an example of the perceptron use. Our current example will be as simple as possible.

**_Here I will provide all the parameters for optimization and forward testing, so as not to repeat myself in the text:_**

- Forex market;
- Currency pair EURUSD;
- Timeframe H1;
- StopLoss 300 and TakeProfit 600. TakeProfit in the EA is set as StopLoss multiplied by 2;
- "Open prices only", "Fast (genetic based algorithm)" and "Complex Criterion max" optimization and testing modes. It is very important to use the "Maximum complex criterion" mode, it showed more stable and profitable results compared to "Maximum profitability";
- Optimization range 3 years. From 2019.04.19 to 2022.04.19 . 3 years is not a reliable criterion. You can experiment with this parameter on your own;
- Forward test range is 1 year. From 2022.04.19 to 2023.04.19.
- Initial deposit 10,000 units;
- Leverage 1:500.

**_Optimization:_**

_EA optimization parameters:_

![Optimization](https://c.mql5.com/2/54/Opt3.png)

_EA optimization results:_

![Optimization](https://c.mql5.com/2/54/Opt1.png)

![Optimization](https://c.mql5.com/2/54/Opt2.png)

**Below are the top 5 best forward test results:**

![Test 1](https://c.mql5.com/2/54/1.png)

![Test 2](https://c.mql5.com/2/54/2.png)

![Test 3](https://c.mql5.com/2/54/3.png)

![Test 4](https://c.mql5.com/2/54/4.png)

![Test 5](https://c.mql5.com/2/54/5.png)

### Conclusion

In conclusion, the perceptron is a powerful tool for price prediction in the Forex market. It can be used on its own or in combination with other data analysis methods. However, in order to achieve the best results when using a perceptron to predict Forex prices, one must be aware of its limitations and take into account the context of historical data. It is also necessary to have certain knowledge and experience in Forex trading and understand the high risk associated with Forex trading.

Thank you for your attention!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12515](https://www.mql5.com/ru/articles/12515)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12515.zip "Download all attachments in the single ZIP archive")

[Perceptron\_MA\_4.mq5](https://www.mql5.com/en/articles/download/12515/perceptron_ma_4.mq5 "Download Perceptron_MA_4.mq5")(39.78 KB)

[NeuralNets.mqh](https://www.mql5.com/en/articles/download/12515/neuralnets.mqh "Download NeuralNets.mqh")(7.83 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Experiments with neural networks (Part 7): Passing indicators](https://www.mql5.com/en/articles/13598)
- [Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://www.mql5.com/en/articles/12459)
- [Experiments with neural networks (Part 4): Templates](https://www.mql5.com/en/articles/12202)
- [Testing and optimization of binary options strategies in MetaTrader 5](https://www.mql5.com/en/articles/12103)
- [Experiments with neural networks (Part 3): Practical application](https://www.mql5.com/en/articles/11949)
- [Experiments with neural networks (Part 2): Smart neural network optimization](https://www.mql5.com/en/articles/11186)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/448335)**
(5)


![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
3 Jun 2023 at 14:22

Hi Roman,

Two Great articles! I have just read both for the first time.

As I have not studied the code yet, I am interested to know is the CNeuralNet object a reformulation of your previous Perceptron calculations?  It looks very interesting as the the initial Angle and fan approaches fail miserably in my forward tests.  I am using [EURUSD](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: technical analysis") H4 from 1/1/2020 to 1/1/0203 as my training and using 1/1/2023 to 5/1/2023 as my forward tests.  The angle fails as there are extended trends with pauses that trigger it but do not reverse and stop out and bankrupt the account with the first dip around 1/2/2023 whereas your tests follow this dip perfectly.  The fan approach does not take any trades in the forward test.

Stay safeI', I'm looking forward to your next articles.

CapeCoddah

P.S.

I have just looked at your two source files and have some questions.

It seems like there are missing parts based on source codes from your prior Perceptron articles.

The EA provided seems to be your Optimization EA. However, it does not use the CNeuralNet object which I expected to see.

The forward test EA is missing as the Attached EA does not use the results from the GA optimization run as input for the weights array, e.g. the EURUSD array.

Or dis I miss a a logical change in your Perceptron philosophy?

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
3 Jun 2023 at 15:59

**CapeCoddah [#](https://www.mql5.com/en/forum/448335#comment_47279348):**

Hi Roman,

Two Great articles! I have just read both for the first time.

As I have not studied the code yet, I am interested to know is the CNeuralNet object a reformulation of your previous Perceptron calculations?  It looks very interesting as the the initial Angle and fan approaches fail miserably in my forward tests.  I am using EURUSD H4 from 1/1/2020 to 1/1/0203 as my training and using 1/1/2023 to 5/1/2023 as my forward tests.  The angle fails as there are extended trends with pauses that trigger it but do not reverse and stop out and bankrupt the account with the first dip around 1/2/2023 whereas your tests follow this dip perfectly.  The fan approach does not take any trades in the forward test.

Stay safeI', I'm looking forward to your next articles.

CapeCoddah

P.S.

I have just looked at your two source files and have some questions.

It seems like there are missing parts based on source codes from your prior Perceptron articles.

The EA provided seems to be your Optimization EA. However, it does not use the CNeuralNet object which I expected to see.

The forward test EA is missing as the Attached EA does not use the results from the GA optimization run as input for the weights array, e.g. the EURUSD array.

Or dis I miss a a logical change in your Perceptron philosophy?

Hello, my friend. I do not understand you well, so I ask you to express your thoughts gradually. Optimization depends on the depth of influence of the perceptron used in the settings. Each pair has its own conclusions. It also depends on the number of passes, since their value is infinite.

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
4 Jun 2023 at 11:38

Hi Roman,

I appreciate the speedy response.  I think I understand GA optimization in that the results may vary from run to run using identical time frames, & that the results will be different depending on the starting date, length of test and for each pair.  What I did not expect is that when a 3 year training run produces a 50% profit, that the EA will fail in 5 days by losing the whole starting position or not take any trades during the actual run.

My ultimate objective is to develop a swing trading Perceptron EA that is trained for a fixed length and runs for only one month following the last date of the training period.  It would then be retrained for the same length but the startling period would be one month later, followed by running for the second month for actual data, like a rolling  SMA. My basis for this is my assumption that the Forex market gradually changes direction and that any "trained" Network will be most accurate in the first few months following training and then will gradually lose accuracy as the market conditions continue to change.  I also understand that there can be seminal market changes will have a direct impact on the accuracy of any "trained" network.  This type of change will significantly impact all future changes.

It is my observation that the Angle Perceptron is quite good at sensing the beginning of reversals before they happen.  Unfortunately, it also good at detecting pauses in trends and issuing a trade in anticipation of the reversal which in this case does not happen.  As the trend continues, this leads to a significant loss due to the large SL which at the beginning of the actual test run causes the loss of the starting position.  I think part of the problem is that 100 Perceptron loop requires adaptive adjustment to lower the total number of trades based on the account balance.

**My immediate issues are from the P.S. to my original comment**.

In your former posts, you posted an EA for optimization (opt) and a second EA for trade testing (trade).  In this post, there is only one: EA Perceptron\_MA\_4.  It is my sense that this EA can be used to run GA optimizations directly corresponding to your earlier OPT versions.  But, there is no Trade version included for forward testing.  If this is intentional, I can adapt this EA to load the GA results to produce an EA for forward testing.

In addition to the EA, you posted a class object, NeualNet as an include file, which I have not reviewed.  What surprised me was the Perceptron\_MA\_4 EA does not use this include file.  What I had expected was that there would be an optimizing EA version that included the CNeuralNet class and also used one of the Normalizing techniques from your Part 5 posting. And in addition there would be a separate Trade version for forward testing.  I think the creation of the class objects a very good direction to take.  As an object, it becomes very easy to use multiple different Perceptrons at the same time in an EA, say for trades, stop loss or take profit settings or possibly to even create an adaptive [trading strategy](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies ") using alternative strategies for trending of flat market.  Any yes, I know that multiple objects will be processing hogs for training.

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
4 Jun 2023 at 12:57

**CapeCoddah [#](https://www.mql5.com/en/forum/448335#comment_47288296):**

Hi Roman,

I appreciate the speedy response.  I think I understand GA optimization in that the results may vary from run to run using identical time frames, & that the results will be different depending on the starting date, length of test and for each pair.  What I did not expect is that when a 3 year training run produces a 50% profit, that the EA will fail in 5 days by losing the whole starting position or not take any trades during the actual run.

My ultimate objective is to develop a swing trading Perceptron EA that is trained for a fixed length and runs for only one month following the last date of the training period.  It would then be retrained for the same length but the startling period would be one month later, followed by running for the second month for actual data, like a rolling  SMA. My basis for this is my assumption that the Forex market gradually changes direction and that any "trained" Network will be most accurate in the first few months following training and then will gradually lose accuracy as the market conditions continue to change.  I also understand that there can be seminal market changes will have a direct impact on the accuracy of any "trained" network.  This type of change will significantly impact all future changes.

It is my observation that the Angle Perceptron is quite good at sensing the beginning of reversals before they happen.  Unfortunately, it also good at detecting pauses in trends and issuing a trade in anticipation of the reversal which in this case does not happen.  As the trend continues, this leads to a significant loss due to the large SL which at the beginning of the actual test run causes the loss of the starting position.  I think part of the problem is that 100 Perceptron loop requires adaptive adjustment to lower the total number of trades based on the account balance.

**My immediate issues are from the P.S. to my original comment**.

In your former posts, you posted an EA for optimization (opt) and a second EA for trade testing (trade).  In this post, there is only one: EA Perceptron\_MA\_4.  It is my sense that this EA can be used to run GA optimizations directly corresponding to your earlier OPT versions.  But, there is no Trade version included for forward testing.  If this is intentional, I can adapt this EA to load the GA results to produce an EA for forward testing.

In addition to the EA, you posted a class object, NeualNet as an include file, which I have not reviewed.  What surprised me was the Perceptron\_MA\_4 EA does not use this include file.  What I had expected was that there would be an optimizing EA version that included the CNeuralNet class and also used one of the Normalizing techniques from your Part 5 posting. And in addition there would be a separate Trade version for forward testing.  I think the creation of the class objects a very good direction to take.  As an object, it becomes very easy to use multiple different Perceptrons at the same time in an EA, say for trades, stop loss or take profit settings or possibly to even create an adaptive [trading strategy](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies ") using alternative strategies for trending of flat market.  Any yes, I know that multiple objects will be processing hogs for training.

Hi. You have the right direction. If you need help with a specific task, write to me and I will try to help. Just write in private.

![Enrique Enguix](https://c.mql5.com/avatar/2025/9/68c108f2-b619.jpg)

**[Enrique Enguix](https://www.mql5.com/en/users/envex)**
\|
12 Aug 2023 at 14:47

**MetaQuotes:**

Published article [Experiments with neural networks (Part 6): The perceptron as a self-sufficient price prediction tool](https://www.mql5.com/en/articles/12515):

Author: [Roman Poshtar](https://www.mql5.com/en/users/romanuch "romanuch")

Excellent article: very educational


![How to create a custom Donchian Channel indicator using MQL5](https://c.mql5.com/2/55/donchian_channel_indicator_avatar.png)[How to create a custom Donchian Channel indicator using MQL5](https://www.mql5.com/en/articles/12711)

There are many technical tools that can be used to visualize a channel surrounding prices, One of these tools is the Donchian Channel indicator. In this article, we will learn how to create the Donchian Channel indicator and how we can trade it as a custom indicator using EA.

![Frequency domain representations of time series: The Power Spectrum](https://c.mql5.com/2/54/power_spectrum4_avatar.png)[Frequency domain representations of time series: The Power Spectrum](https://www.mql5.com/en/articles/12701)

In this article we discuss methods related to the analysis of timeseries in the frequency domain. Emphasizing the utility of examining the power spectra of time series when building predictive models. In this article we will discuss some of the useful perspectives to be gained by analyzing time series in the frequency domain using the discrete fourier transform (dft).

![Money management in trading](https://c.mql5.com/2/54/capital_control_avatar.png)[Money management in trading](https://www.mql5.com/en/articles/12550)

We will look at several new ways of building money management systems and define their main features. Today, there are quite a few money management strategies to fit every taste. We will try to consider several ways to manage money based on different mathematical growth models.

![Category Theory in MQL5 (Part 8): Monoids](https://c.mql5.com/2/54/Category-Theory-p8-avatar.png)[Category Theory in MQL5 (Part 8): Monoids](https://www.mql5.com/en/articles/12634)

This article continues the series on category theory implementation in MQL5. Here we introduce monoids as domain (set) that sets category theory apart from other data classification methods by including rules and an identity element.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/12515&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070283441024471849)

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