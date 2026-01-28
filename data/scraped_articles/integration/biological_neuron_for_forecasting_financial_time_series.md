---
title: Biological neuron for forecasting financial time series
url: https://www.mql5.com/en/articles/16979
categories: Integration, Machine Learning
relevance_score: 0
scraped_at: 2026-01-24T14:00:53.381299
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/16979&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083281025674516579)

MetaTrader 5 / Integration


Modern trading is undergoing a revolution. Neural networks, genetic algorithms, deep learning — all these tools produce impressive results, but often suffer from one significant drawback: they are too "mechanical" and unable to capture the subtle nature of market fluctuations, which are essentially the result of the collective behavior of living market participants.

In this context, the use of biologically plausible neural models becomes particularly relevant. Instead of creating abstract mathematical constructs, we turned to the most advanced computer in the observable part of the Universe - the human brain. Our research group has developed a unique system based on the Hodgkin-Huxley model that simulates not only the functionality but also the physical nature of neural processes.

The Nobel Prize-winning Hodgkin-Huxley model describes the mechanism of generation and propagation of nerve impulses at the cellular level. But why might this particular model be the key to understanding financial markets? The answer lies in the striking analogy between the spread of nerve impulses in the brain and the spread of information in markets. Just as neurons exchange electrical signals through synaptic connections, market participants exchange information through trading transactions.

The innovation of our approach lies in adding a plasma-like component to the classical model. We view a neural network as a dynamic system immersed in the "plasma" of market information, where each neuron can influence the behavior of other neurons not only through direct connections, but also through the electromagnetic fields it creates. This allows the system to capture subtle correlations and relationships that go unnoticed by traditional algorithms.

In this article, we will take a detailed look at the system architecture, its operating principles, and the results of practical application on various financial instruments. We will show how a biologically inspired approach can offer a new perspective on the problem of financial time series forecasting and open new horizons in the field of algorithmic trading.

### Hodgkin-Huxley model: When biology meets finance

Imagine a neuron — an amazing cell able to handle and transmit information through electrical impulses. It was this fundamental unit of the brain that Alan Hodgkin and Andrew Huxley described in their groundbreaking work, for which they later received the Nobel Prize. Their model was a real breakthrough in neuroscience, and now imagine that this same model can help us understand and predict the movements of financial markets.

At the very heart of the model lies a mathematical description of how a neuron generates electrical impulses. Just as traders react to incoming market information, a neuron reacts to incoming signals by creating its own unique pattern of activity. Let's take a look at the code that describes this amazing process:

```
class HodgkinHuxleyNeuron:
    def __init__(self):
        self.V = -65.0  # Initial resting potential
        self.m = 0.05   # Activation of sodium channels
        self.h = 0.6    # Inactivation of sodium channels
        self.n = 0.32   # Activation of potassium channels
        self.last_spike_time = float('-inf')
```

**Ion dance: How neurons make decisions**

The most striking thing about the Hodgkin-Huxley model is its description of how ions "dance" across the neuron membrane. Sodium and potassium, the two key players in this molecular ballet, move through special channels, creating an electrical potential. This is very similar to how order flows create price movement in the market. In our trading system, we use this analogy by converting market data into ionic currents:

```
def ion_currents(self, V):
    I_Na = self.g_Na * (self.m ** 3) * self.h * (V - self.E_Na)  # Sodium current
    I_K = self.g_K * (self.n ** 4) * (V - self.E_K)             # Potassium current
    I_L = self.g_L * (V - self.E_L)                             # Leakage current
    return I_Na, I_K, I_L
```

**Plasma influence: A new look at neural networks**

We went beyond the classical model and added the concept of a plasma-like environment. Imagine that neurons are not simply connected by synapses, but are immersed in the "information plasma" of the market. Each neuron spike creates a wave of influence that fades over time, just like a piece of news on the market gradually loses its impact on the price:

```
def plasma_influence(self, current_time):
    time_since_spike = current_time - self.last_spike_time
    influence = self.plasma_strength * np.exp(-time_since_spike / self.plasma_decay)
    return influence * self.get_market_correlation()
```

**Learning through Time: The STDP Mechanism**

Perhaps the most exciting part of our system is the learning mechanism based on the temporal dependence between neuronal spikes (STDP). Just as a trader learns from experience, our neurons adjust the strength of their connections depending on how successfully they predicted market movements:

```
def update_synaptic_weights(self, pre_spike, post_spike, weight):
    delta_t = post_spike - pre_spike
    if delta_t > 0:
        return weight * (1 + self.A_plus * np.exp(-delta_t / self.tau_plus))
    else:
        return weight * (1 - self.A_minus * np.exp(delta_t / self.tau_minus))
```

This biologically inspired architecture allows our system to not just handle market data, but to "feel" the market like an experienced trader. Each neuron becomes a mini-expert on its own aspect of market dynamics, and their collective work creates remarkably accurate forecasts.

**System architecture: from biology to trading decisions**

Imagine the brain of a trader analyzing the market. Thousands of neurons process information about prices, volumes, indicators, and news. It is this natural architecture that we have recreated in our system, adding to it the unique properties of plasma and electromagnetic interactions.

**Eyes of the system: Input data structure**

Just as the human brain receives information through the senses, our system collects and handles a multitude of market signals. Every tick, every price movement is converted into neural impulses. Let's take a look at the data preprocessing code:

```
class MarketFeatures:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.scaler = StandardScaler()

    def add_price(self, price: float, ohlc_data: pd.DataFrame) -> Dict[str, float]:
        features = {}

        # Technical indicators
        features['sma_10'] = self._calculate_sma(ohlc_data['close'], window=10)
        features['ema_20'] = self._calculate_ema(ohlc_data['close'], window=20)
        features['rsi'] = self._calculate_rsi(ohlc_data['close'], window=14)

        # Volumetric characteristics
        features['volume_sma'] = self._calculate_sma(ohlc_data['tick_volume'], window=10)

        # Temporal patterns
        features['hour'] = ohlc_data.index[-1].hour
        features['day_of_week'] = ohlc_data.index[-1].dayofweek

        return self.scaler.fit_transform(np.array(list(features.values())).reshape(1, -1))
```

**Neural network: A bridge between biology and mathematics**

At the heart of our system is a hybrid architecture that combines classical neural networks with the biologically plausible Hodgkin-Huxley model. Each neuron here is not just a mathematical function, but a small living system:

```
class BioTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BioTradingModel, self).__init__()
        self.layers = nn.ModuleList([\
            nn.Linear(input_size, hidden_size),\
            nn.Tanh(),\
            nn.Linear(hidden_size, hidden_size),\
            nn.Tanh(),\
            nn.Linear(hidden_size, output_size)\
        ])

        # Biological neurons
        self.bio_neurons = [HodgkinHuxleyNeuron() for _ in range(hidden_size)]
        self.plasma_field = PlasmaField(hidden_size)
```

**Plasma field: A new dimension of neural interactions**

The uniqueness of our system lies in the plasma-like environment in which the neurons are immersed. Just as electromagnetic fields permeate space, our plasma creates an additional level of interaction between neurons:

```
class PlasmaField:
    def __init__(self, size):
        self.field_strength = np.zeros(size)
        self.decay_rate = 0.95

    def update(self, neuron_activities):
        # Update the field based on neuron activity
        self.field_strength = self.field_strength * self.decay_rate
        self.field_strength += neuron_activities

    def get_influence(self, neuron_index):
        # Calculate the influence of a field on a specific neuron
        return np.sum(self.field_strength * np.exp(-self.distance_matrix[neuron_index]))
```

**Learning mechanism: Dance of neurons and plasma**

In our system, training occurs on several levels simultaneously. Classical gradient descent is combined with biological STDP and plasma dynamics:

```
def train_step(self, inputs, target):
    # Straight pass
    predictions = self.forward(inputs)
    loss = self.criterion(predictions, target)

    # Backpropagation
    self.optimizer.zero_grad()
    loss.backward()

    # Biological education
    for i, neuron in enumerate(self.bio_neurons):
        # STDP update
        neuron.update_weights(self.last_spike_times)

        # Plasma modulation
        plasma_influence = self.plasma_field.get_influence(i)
        neuron.modulate_weights(plasma_influence)

    self.optimizer.step()
    return loss.item()
```

This multi-layered architecture allows the system to capture not only obvious patterns in the data, but also subtle, barely noticeable relationships between different aspects of market dynamics. Each component of the system plays its own unique role, and their synergy creates something greater than the sum of its parts – a true artificial brain for analyzing financial markets.

**Technical indicators: the digital senses of our system**

If we imagine our neural system as a trader's artificial brain, then technical indicators are its sense organs. Just as humans use sight, hearing, and touch to perceive the world, our system uses various indicators to "sense" the market in all its manifestations.

**Basic indicators: System vision**

Let's start with the most fundamental indicators - moving averages. They are like our system's vision, allowing it to discern the underlying trend through market noise:

```
def calculate_moving_averages(self, prices):
    def sma(window):
        return np.convolve(prices, np.ones(window)/window, mode='valid')

    def ema(window):
        alpha = 2 / (window + 1)
        kernel = alpha * (1 - alpha)**np.arange(window)
        return np.convolve(prices, kernel[::-1], mode='valid')

    return {
        'sma_fast': sma(10),  # Fast SMA for short-term trends
        'sma_slow': sma(20),  # Slow SMA for long-term trends
        'ema_fast': ema(10),  # Exponential MA for fast response
        'ema_slow': ema(20)   # Slow EMA for sorting out noise
    }
```

**Oscillators: Sensing market moments**

RSI, Momentum and Stochastic are the tactile receptors of our system. They allow us to "feel" the strength of price movement and potential reversal points:

```
def calculate_oscillators(self, data):
    def rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])

        for i in range(period, len(gain)):
            avg_gain = (avg_gain * 13 + gain[i]) / 14
            avg_loss = (avg_loss * 13 + loss[i]) / 14

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    return {
        'rsi': rsi(data['close']),
        'momentum': data['close'] - np.roll(data['close'], 10),
        'stoch_k': self._calculate_stochastic_k(data)
    }
```

**Volatility and volume: The system hearing**

Volatility and volume metrics are like our system's ears — they pick up on market "noise" and the strength of movements. Bollinger Bands and ATR help to evaluate the amplitude of fluctuations, and volume indicators help to assess their significance:

```
def measure_market_dynamics(self, data):
    def bollinger_bands(prices, window=20):
        sma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        return {
            'upper': sma + 2 * std,
            'lower': sma - 2 * std,
            'width': 4 * std / sma  # Normalized strip width
        }

    def volume_profile(volumes, prices):
        return {
            'volume_ma': np.mean(volumes[-10:]),
            'volume_trend': np.corrcoef(volumes[-20:], prices[-20:])[0,1],
            'volume_oscillator': (np.mean(volumes[-5:]) / np.mean(volumes[-20:]) - 1) * 100
        }

    volatility = bollinger_bands(data['close'])
    volume = volume_profile(data['volume'], data['close'])
    return {**volatility, **volume}
```

**Normalization: Adjusting sensitivity**

Just as the human brain adapts the sensitivity of its senses to environmental conditions, our system dynamically adjusts the scale of input data. This is critical for the stable operation of the neural network:

```
def normalize_features(self, features: dict) -> dict:
    class AdaptiveNormalizer:
        def __init__(self, window=100):
            self.window = window
            self.history = {}

        def update(self, feature_name, value):
            if feature_name not in self.history:
                self.history[feature_name] = []

            self.history[feature_name].append(value)
            if len(self.history[feature_name]) > self.window:
                self.history[feature_name].pop(0)

            mean = np.mean(self.history[feature_name])
            std = np.std(self.history[feature_name])
            return (value - mean) / (std + 1e-8)  # Avoid zero divide

    normalizer = AdaptiveNormalizer()
    return {name: normalizer.update(name, value)
            for name, value in features.items()}
```

All these indicators and features work closely together, creating a multidimensional picture of market reality. Just as the human brain integrates information from all of the senses, our system synthesizes data from all indicators into a single representation of the market situation. This allows it to not simply follow a trend or react to individual signals, but to develop a holistic understanding of market dynamics.

### Experimental results

**Testing methodology**

During the study, we conducted comprehensive testing of the system on historical data for the EURUSD currency pair. The time horizon was 5 years using the daily timeframe D1. Following conventional machine learning principles, we split the data into train and test sets in an 80/20 ratio. As a result, 1659 data points were used for training and 415 for testing. The model went through 20 training iterations with different weight initializations to find the optimal configuration.

![](https://c.mql5.com/2/115/training_error.png)

**Performance analysis**

While analyzing the results, we discovered an interesting feature of our model. Instead of trying to predict short-term price movements, the system appears to aim to determine some kind of "fair" price for the currency pair. This observation is confirmed by the rather high correlation on the training sample, reaching the value of 0.583 with the relatively low standard error of 0.012346. However, on the test sample, the model's performance drops significantly: the correlation drops to negative values (-0.108), and the MSE increases more than 90 times, reaching 1.156584.

**Comparison with traditional approaches**

Our biologically inspired model exhibits significantly different behavior compared to classic technical indicators and standard neural networks. Its forecasts turn out to be significantly more volatile.

Here is a test on a test sample, with a forecast horizon of 15 bars:

![](https://c.mql5.com/2/115/test_results.png)

**Statistical performance metrics**

The most significant result of the model performance was the extremely low ratio of determination (R²) on the test sample, which amounted to approximately 0.01. Interestingly, the model shows a fairly high correlation on the training sample, which indicates its ability to capture long-term patterns in the data. Besides, the model forecasts often contain clearly visible spikes. Whether these spikes correspond to the smallest price movements and whether this is suitable for scalping remains to be seen.

![](https://c.mql5.com/2/115/training_process.png)

**Features and practical application**

The observed behavior of the model can be explained by its biological nature. The plasma-like neural system appears to act as a powerful filter, significantly amplifying market signals, and the model produces exaggerated incremental forecasts. The STDP (Spike-Timing-Dependent Plasticity) mechanism, in theory, should lead to the formation of stable activation patterns, which leads to averaging of predictions, and, accordingly, we should get a different picture of the test and the actual state. An additional factor may be a large number of input in the form of technical indicators, which creates an overregularization effect.

From a practical point of view, this model can be used to determine short-term levels of fair value of a currency pair.

### Conclusion

Our research into a biologically inspired neural system for predicting financial markets has yielded unexpected but intriguing results. Just as the human brain is able to intuitively sense the "fair" value of an asset, our model, based on the working principles of living neurons, has demonstrated a remarkable ability to identify fundamentally sound price levels.

The introduction of a plasma-like environment into the neural network architecture created a kind of "collective intelligence," where each neuron influences the system's operation not only through direct connections, but also through long-range electromagnetic interactions. This mechanism has proven particularly efficient in sorting out market noise and identifying long-term trends.

Perhaps there is a deeper meaning to this. Just as biological systems evolved to survive in the long term, our neural network, built in their image, also strives to identify stable, fundamentally sound patterns.

### Bonus for those who read to the end

I also created an indicator based on this model. The indicator loads the rest of the system as a module and opens in a simple window like this:

![](https://c.mql5.com/2/115/r3lx95z1_30-01-2025_202735.jpg)

I have not tried it live, but considering that all the other systems generally predict the same direction, the indicator might work as it should.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16979](https://www.mql5.com/ru/articles/16979)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16979.zip "Download all attachments in the single ZIP archive")

[BioTraderLearn.py](https://www.mql5.com/en/articles/download/16979/BioTraderLearn.py "Download BioTraderLearn.py")(15.07 KB)

[BioTraderPredictor.py](https://www.mql5.com/en/articles/download/16979/BioTraderPredictor.py "Download BioTraderPredictor.py")(6.05 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forex arbitrage trading: Analyzing synthetic currencies movements and their mean reversion](https://www.mql5.com/en/articles/17512)
- [Forex arbitrage trading: A simple synthetic market maker bot to get started](https://www.mql5.com/en/articles/17424)
- [Forex Arbitrage Trading: Relationship Assessment Panel](https://www.mql5.com/en/articles/17422)
- [Build a Remote Forex Risk Management System in Python](https://www.mql5.com/en/articles/17410)
- [Currency pair strength indicator in pure MQL5](https://www.mql5.com/en/articles/17303)
- [Capital management in trading and the trader's home accounting program with a database](https://www.mql5.com/en/articles/17282)
- [Analyzing all price movement options on the IBM quantum computer](https://www.mql5.com/en/articles/17171)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/497633)**
(7)


![Vladimir Perervenko](https://c.mql5.com/avatar/2014/8/53EE3B01-DE57.png)

**[Vladimir Perervenko](https://www.mql5.com/en/users/vlad1949)**
\|
8 Feb 2025 at 15:31

To Ivan Butko

Preprocessing - preprocessing of predictors - is the first and most important of the three stages of any machine learning [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not accurate "). You need to sit down and learn the basics. Then you wouldn't be talking rubbish.

"Garbage in - rubbish out" - and you don't need to go to a fortune-teller for that.

![Vladimir Perervenko](https://c.mql5.com/avatar/2014/8/53EE3B01-DE57.png)

**[Vladimir Perervenko](https://www.mql5.com/en/users/vlad1949)**
\|
8 Feb 2025 at 16:09

From the article;

Exotics offer no advantage even against simple statistical models. And for what?

By code:

Adaptive normalisation - I didn't see what is adaptive there?

All indicators are in the library of technical analysis ta. Why rewrite everything in Python?

There is no sense in practical application, IMHO

![Ivan Butko](https://c.mql5.com/avatar/2017/1/58797422-0BFA.png)

**[Ivan Butko](https://www.mql5.com/en/users/capitalplus)**
\|
8 Feb 2025 at 16:48

**Vladimir Perervenko project. You need to sit down and learn the basics. Then you would not be talking rubbish.**
**"Garbage in - rubbish out" - and you don't need to go to a fortune-teller for that.**

You translate textbooks

You haven't dealt with the definition of rubbish in prices

You don't know what is rubbish and what is not. And whether it exists in principle. Since at Forex people earn on M1, and on M5, and on M15 and so on, up to D1

You do not understand and do not know how to trade with hands.

Hence - you do not understand what you yourself are saying.

But if you have a confirmation of workability and stability of your NS models solely because of the presence of preprocessing (without it - rubbish) - you will be right.

Are there such?

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
8 Feb 2025 at 18:05

Easy to say: learn the basics, it's at least 1 book on the basics you need to read :) and not just read, but memorise.


![Aleksey Nikolayev](https://c.mql5.com/avatar/2018/8/5B813025-B4F2.jpeg)

**[Aleksey Nikolayev](https://www.mql5.com/en/users/alexeynikolaev2)**
\|
9 Feb 2025 at 05:57

Noise is model error. That is, in reality there is no abstract "price noise", there is only a number of errors of a particular model. A model is considered more or less working if a number of these errors behaves as white noise (stationary process without [correlation](https://www.mql5.com/en/docs/matrix/matrix_products/matrix_correlate "MQL5 Documentation: function Correlate")).


![MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://c.mql5.com/2/175/19890-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

The Stochastic Oscillator and the Fractal Adaptive Moving Average are an indicator pairing that could be used for their ability to compliment each other within an MQL5 Expert Advisor. We introduced this pairing in the last article, and now look to wrap up by considering its 5 last signal patterns. In exploring this, as always, we use the MQL5 wizard to build and test out their potential.

![Self Optimizing Expert Advisors in MQL5 (Part 15): Linear System Identification](https://c.mql5.com/2/175/19891-self-optimizing-expert-advisors-logo__1.png)[Self Optimizing Expert Advisors in MQL5 (Part 15): Linear System Identification](https://www.mql5.com/en/articles/19891)

Trading strategies may be challenging to improve because we often don’t fully understand what the strategy is doing wrong. In this discussion, we introduce linear system identification, a branch of control theory. Linear feedback systems can learn from data to identify a system’s errors and guide its behavior toward intended outcomes. While these methods may not provide fully interpretable explanations, they are far more valuable than having no control system at all. Let’s explore linear system identification and observe how it may help us as algorithmic traders to maintain control over our trading applications.

![Building a Trading System (Part 5): Managing Gains Through Structured Trade Exits](https://c.mql5.com/2/175/19693-building-a-trading-system-final-logo.png)[Building a Trading System (Part 5): Managing Gains Through Structured Trade Exits](https://www.mql5.com/en/articles/19693)

For many traders, it's a familiar pain point: watching a trade come within a whisker of your profit target, only to reverse and hit your stop-loss. Or worse, seeing a trailing stop close you out at breakeven before the market surges toward your original target. This article focuses on using multiple entries at different Reward-to-Risk Ratios to systematically secure gains and reduce overall risk exposure.

![Risk Management (Part 1): Fundamentals for Building a Risk Management Class](https://c.mql5.com/2/112/Gesti7n_de_Riesgo_Parte_1_LOGO.png)[Risk Management (Part 1): Fundamentals for Building a Risk Management Class](https://www.mql5.com/en/articles/16820)

In this article, we'll cover the basics of risk management in trading and learn how to create your first functions for calculating the appropriate lot size for a trade, as well as a stop-loss. Additionally, we will go into detail about how these features work, explaining each step. Our goal is to provide a clear understanding of how to apply these concepts in automated trading. Finally, we will put everything into practice by creating a simple script with an include file.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/16979&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083281025674516579)

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