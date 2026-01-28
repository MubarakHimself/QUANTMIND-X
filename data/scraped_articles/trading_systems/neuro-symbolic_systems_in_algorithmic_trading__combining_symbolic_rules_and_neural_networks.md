---
title: Neuro-symbolic systems in algorithmic trading: Combining symbolic rules and neural networks
url: https://www.mql5.com/en/articles/16894
categories: Trading Systems, Machine Learning
relevance_score: 9
scraped_at: 2026-01-22T17:33:11.504371
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/16894&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049211644275435301)

MetaTrader 5 / Trading systems


### Introduction to neurosymbolic systems: Principles of combining rules and neural networks

Imagine you are trying to explain to a computer how to trade on the stock exchange. On the one hand, we have classic rules and patterns — "head and shoulders," "double bottom," and hundreds of other patterns familiar to any trader. Many of us have written EAs in MQL5, trying to encode these patterns. But the market is a living organism, it is constantly changing, and strict rules often fail.

On the other hand, there are neural networks – fashionable, powerful, but sometimes completely opaque in their decisions. Feed historical data to an LSTM network and it will make predictions with decent accuracy. But the reasoning behind these decisions often remains a mystery. In trading, every wrong step can cost real money.

I remember struggling with this dilemma in my trading algorithm a few years ago. Classic patterns produced false positives, and the neural network sometimes produced incredible predictions without any logic. And then it dawned on me: what if we combine both approaches? What if we use clear rules as the system framework, and the neural network as an adaptive mechanism that takes into account the current state of the market?

This is how the idea of a neurosymbolic system for algorithmic trading was born. Imagine it as an experienced trader who knows all the classic patterns and rules, but also knows how to adapt to the market, taking into account subtle nuances and relationships. Such a system has a "skeleton" of clear rules and "muscles" in the form of a neural network, which adds flexibility and adaptability.

In this article, I will explain how my team and I developed such a system in Python and show how to combine classical pattern analysis with modern machine learning methods. We will walk through the architecture, from basic components to complex decision-making mechanisms, and of course, I will share real code and test results.

Ready to dive into the world where classic trading rules meet neural networks? Then let's go!

### Symbolic rules in trading: Patterns and their statistics

Let's start with the simple thing: what is a market pattern? In classical technical analysis, this is a specific figure on the chart, for example, a "double bottom" or a "flag". But when we talk about programming trading systems, we need to think more abstractly. In our code, a pattern is a sequence of price movements, encoded in binary form: 1 for growth, 0 for decline.

It seems primitive, you might say? Not at all. This representation gives us a powerful tool for analysis. Let's take the sequence \[1, 1, 0, 1, 0\] - this is not just a set of numbers, but an encoded mini-trend. In Python, we can search for such patterns with simple but effective code:

```
pattern = tuple(np.where(data['close'].diff() > 0, 1, 0))
```

But the real magic begins when we start analyzing the statistics. For each pattern we can calculate three key parameters:

1. Frequency - how many times the pattern appeared in history
2. Winrate — how often the price moved in the predicted direction following a pattern
3. Reliability — a complex indicator that takes into account both frequency and win rate

Here is a real example from my practice: the pattern \[1, 1, 1, 0, 0\] on EURUSD H4 shows a win rate of 68% with a frequency of occurrence of more than 200 times per year. Sounds tempting, right? But here it is important not to fall into the trap of over-optimization.

That is why we added a dynamic reliability filter:

```
reliability = frequency * winrate * (1 - abs(0.5 - winrate))
```

This equation is amazing in its simplicity. It not only takes into account frequency and win rate, but also penalizes patterns with suspiciously high efficiency, which often turns out to be a statistical anomaly.

The length of the patterns is a separate story. Short patterns (3-4 bars) are common, but create a lot of noise. Long ones (20-25 bars) are more reliable, but rare. The golden mean is usually in the 5-8 bar range. Although, I admit, for some instruments I have seen excellent results on 12-bar patterns.

An important point is the forecast horizon. In our system, we use the forecast\_horizon parameter, which determines how many bars ahead we try to predict the movement. Empirically, we arrived at the value of 6 – it provides the optimal balance between forecast accuracy and trading opportunities.

But the most interesting thing happens when we start to analyze patterns in different market conditions. The same pattern can behave completely differently with different volatility or at different times of the day. This is why simple statistics are only the first step. This is where neural networks come into play, but we will talk about that in the next section.

### Neural network architecture for market data analysis

Now let's take a look at the "brain" of our system - the neural network. After extensive experimentation, we settled on a hybrid architecture that combines LSTM layers for handling time series and fully connected layers for processing statistical features of patterns.

Why LSTM? The point is that market data is not just a set of numbers, but a sequence where each value is related to the previous ones. LSTM networks are excellent at capturing such long-term dependencies. Here's what the basic structure of our network looks like:

```
model = tf.keras.Sequential([\
    tf.keras.layers.LSTM(256, input_shape=input_shape, return_sequences=True),\
    tf.keras.layers.Dropout(0.4),\
    tf.keras.layers.LSTM(128),\
    tf.keras.layers.Dropout(0.3),\
    tf.keras.layers.Dense(64, activation='relu'),\
    tf.keras.layers.Dense(1, activation='sigmoid')\
])
```

Note the Dropout layers - this is our protection against overfitting. In early versions of the system, we did not use them, and the network worked perfectly on historical data, but failed in the real market. Dropout randomly switches off some neurons during training, forcing the network to search for more robust patterns.

An important point is the dimension of the input data. The input\_shape parameter is determined by three key factors:

1. Analysis window size (in our case it is 10 time steps)
2. Number of basic features (price, volume, technical indicators)
3. Number of features extracted from patterns

The result is a tensor of dimension (batch\_size, 10, features), where 'features' is the total number of all features. This is exactly the data format the first LSTM layer expects.

Note the return\_sequences=True parameter in the first LSTM layer. This means that the layer returns a sequence of outputs for each time step, not just the last one. This allows the second LSTM layer to obtain more detailed information about the temporal dynamics. But the second LSTM only produces the final state - its output goes to fully connected layers.

Fully connected layers (Dense) act as an "interpreter" - they transform the complex patterns found by LSTM into a concrete solution. The first Dense layer with ReLU activation processes nonlinear dependencies, and the final layer with sigmoid activation produces the probability of an upward price movement.

The model compilation process deserves special attention:

```
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
```

We use the Adam optimizer, which has proven itself to be effective for non-stationary data, such as market prices. Binary crossentropy as a loss function is ideal for our binary classification problem (predicting the direction of price movement). A set of metrics helps track not only the accuracy but also the quality of predictions in terms of false positives and false negatives.

During the development, we experimented with different network configurations. We tried adding convolutional layers (CNN) to identify local patterns and experimented with the attention mechanism, but ultimately came to the conclusion that simplicity and transparency of the architecture are more important. The more complex the network, the more difficult it is to interpret its decisions, and in trading, understanding the logic behind the system operation is critically important.

### Pattern integration into neural networks: Input data enrichment

Now comes the most interesting part: how we "cross" classical patterns with a neural network. This is not just a concatenation of features, but a whole system of preliminary data handling and analysis.

Let's start with a basic set of input data. For each time point, we form a multidimensional feature vector, including:

```
base_features = [\
    'close',  # Close price\
    'volume',  # Volume\
    'rsi',    # Relative Strength Index\
    'macd',   # MACD\
    'bb_upper', 'bb_lower'  # Bollinger Bands borders\
]
```

However, this is just the beginning. The main innovation is the addition of pattern statistics. For each pattern, we calculate three key indicators:

```
pattern_stats = {
    'winrate': np.mean(outcomes),  # Percentage of successful triggers
    'frequency': len(outcomes),     # Occurrence frequency
    'reliability': len(outcomes) * np.mean(outcomes) * (1 - abs(0.5 - np.mean(outcomes)))  # Reliability
}
```

Particular attention should be paid to the last metric - reliability. This is our proprietary development, which takes into account not only frequency and win rate, but also the "suspiciousness" of statistics. If the win rate is too close to 100% or too volatile, the reliability indicator decreases.

Integrating this data into a neural network requires special care.

```
def prepare_data(df):
    # We normalize the basic features using MinMaxScaler
    X_base = self.scaler.fit_transform(df[base_features].values)

    # For pattern statistics we use special normalization
    pattern_features = self.pattern_analyzer.extract_pattern_features(
        df, lookback=len(df)
    )

    return np.column_stack((X_base, pattern_features))
```

Solving the problem of different pattern sizes:

```
def extract_pattern_features(self, data, lookback=100):
    features_per_length = 5  # fixed number of features per pattern
    total_features = len(self.pattern_lengths) * features_per_length

    features = np.zeros((len(data) - lookback, total_features))
    # ... filling the feature array
```

Each pattern, regardless of its length, is transformed into a vector of fixed dimension. This solves the problem of a changing number of active patterns and allows the neural network to work with an input of constant dimension.

Taking into account the market context is a separate story. We add special features that characterize the current state of the market:

```
market_features = {
    'volatility': calculate_atr(data),  # Volatility via ATR
    'trend_strength': calculate_adx(data),  # Trend strength via ADX
    'market_phase': identify_market_phase(data)  # Market phase
}
```

This helps the system adapt to different market conditions. For example, during periods of high volatility, we automatically increase the requirements for pattern reliability.

An important point is handling missing data. In real trading, this is a common problem, especially when working with multiple timeframes. We solve it through a combination of methods:

```
# Fill in the blanks, taking into account the specifics of each feature
df['close'] = df['close'].fillna(method='ffill')  # for prices
df['volume'] = df['volume'].fillna(df['volume'].rolling(24).mean())  # for volumes
pattern_features = np.nan_to_num(pattern_features, nan=-1)  # for pattern features
```

As a result, the neural network receives a complete and consistent data set, where classic technical patterns organically complement basic market indicators. This gives the system a unique advantage: it can rely on both time-tested patterns and complex relationships discovered during training.

### Decision-making system: From analysis to signals

Let's talk about how the system actually makes decisions. Forget about neural networks and patterns for a minute - at the end of the day, we need to make a clear decision: to enter the market or not. And if we do enter, then we need to know the volume.

Our basic logic is simple: we take two data streams - a forecast from a neural network and pattern statistics. The neural network gives us the probability of an up/down movement, and the patterns confirm or refute this forecast. But the devil, as usual, is in the details.

Here is what is going on under the hood:

```
def get_trading_decision(self, market_data):
    # Get a forecast from the neural network
    prediction = self.model.predict(market_data)

    # Extract active patterns
    patterns = self.pattern_analyzer.get_active_patterns(market_data)

    # Basic check of market conditions
    if not self._market_conditions_ok():
        return None  # Do not trade if something is wrong

    # Check the consistency of signals
    if not self._signals_aligned(prediction, patterns):
        return None  # No consensus - no deal

    # Calculate the signal confidence
    confidence = self._calculate_confidence(prediction, patterns)

    # Determine the position size
    size = self._get_position_size(confidence)

    return TradingSignal(
        direction='BUY' if prediction > 0.5 else 'SELL',
        size=size,
        confidence=confidence,
        patterns=patterns
    )
```

The first thing we check is the basic market conditions. No rocket science, just common sense:

```
def _market_conditions_ok(self):
    # Check the time
    if not self.is_trading_session():
        return False

    # Look at the spread
    if self.current_spread > self.MAX_ALLOWED_SPREAD:
        return False

    # Check volatility
    if self.current_atr > self.volatility_threshold:
        return False

    return True
```

Next comes the check of signal consistency. The important point here is that we do not require all signals to be perfectly aligned. It is sufficient that the main indicators do not contradict each other:

```
def _signals_aligned(self, ml_prediction, pattern_signals):
    # Define the basic direction
    ml_direction = ml_prediction > 0.5

    # Count how many patterns confirm it
    confirming_patterns = sum(1 for p in pattern_signals
                            if p.predicted_direction == ml_direction)

    # At least 60% of patterns need to be confirmed
    return confirming_patterns / len(pattern_signals) >= 0.6
```

The hardest part is calculating the signal confidence. After numerous experiments and analysis of various approaches, we arrived at the use of a combined metric that takes into account both the statistical reliability of the neural network forecast and the historical performance of the detected patterns:

```
def _calculate_confidence(self, prediction, patterns):
    # Baseline confidence from ML model
    base_confidence = abs(prediction - 0.5) * 2

    # Consider confirming patterns
    pattern_confidence = self._get_pattern_confidence(patterns)

    # Weighted average with empirically selected ratios
    return (base_confidence * 0.7 + pattern_confidence * 0.3)
```

This decision-making architecture demonstrates the efficiency of a hybrid approach, where classical technical analysis methods organically complement the capabilities of machine learning. Each component of the system contributes to the final decision, while a multi-level system of checks ensures the necessary degree of reliability and resilience to various market conditions.

### Conclusion

Combining classic patterns with neural network analysis yields a qualitatively new result: the neural network captures subtle market relationships, while time-tested patterns provide the basic structure of trading decisions. In our tests, this approach has consistently shown better results than both purely technical analysis and the isolated use of machine learning.

An important discovery was the understanding that simplicity and interpretability are crucial. We deliberately avoided more complex architectures in favor of a transparent and understandable system. This allows not only better control over trading decisions, but also the ability to quickly make adjustments as market conditions change. In a world where many chase complexity, simplicity has proven to be our competitive advantage.

I hope our experience will be useful to those who are also exploring the boundaries of what is possible at the intersection of classical trading and artificial intelligence. After all, it is in such interdisciplinary areas that the most interesting and practical solutions are often born. Keep experimenting, but remember that there is no silver bullet in trading. There is only a path of constant development and improvement of your tools.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16894](https://www.mql5.com/ru/articles/16894)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16894.zip "Download all attachments in the single ZIP archive")

[NeuroSymb\_System\_3.py](https://www.mql5.com/en/articles/download/16894/NeuroSymb_System_3.py "Download NeuroSymb_System_3.py")(10.66 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forex arbitrage trading: Analyzing synthetic currencies movements and their mean reversion](https://www.mql5.com/en/articles/17512)
- [Forex arbitrage trading: A simple synthetic market maker bot to get started](https://www.mql5.com/en/articles/17424)
- [Build a Remote Forex Risk Management System in Python](https://www.mql5.com/en/articles/17410)
- [Currency pair strength indicator in pure MQL5](https://www.mql5.com/en/articles/17303)
- [Capital management in trading and the trader's home accounting program with a database](https://www.mql5.com/en/articles/17282)
- [Analyzing all price movement options on the IBM quantum computer](https://www.mql5.com/en/articles/17171)
- [Fibonacci in Forex (Part I): Examining the Price-Time Relationship](https://www.mql5.com/en/articles/17168)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/495951)**
(2)


![Evgeniy Chernish](https://c.mql5.com/avatar/2024/3/65eac9b5-9233.png)

**[Evgeniy Chernish](https://www.mql5.com/en/users/vp999369)**
\|
22 Jan 2025 at 09:25

**MetaQuotes:**

Published article [Neurosymbolic Systems in Algorithm Trading: Combining Symbolic Rules and Neural Networks](https://www.mql5.com/en/articles/16894):

Author: [Yevgeniy Koshtenko](https://www.mql5.com/en/users/Koshtenko "Koshtenko")

The main problem is the stability of the calculated frequency of appearance of a white or black candle after the appearance of a pattern. On small samples it is unreliable, and on large samples it is 50/50.

And I don't understand the logic of first feeding the pattern frequency to neuronka as one of the features, and then using the same frequency to filter neuronka signals built on it.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
22 Jan 2025 at 11:53

Without touching the approach itself, reducing the real ranges of movements to two classes nullifies the useful information that could be extracted by the [neural network](https://www.mql5.com/en/articles/497 "Article: Neural networks - from theory to practice ") (for the sake of which we screw it in) - akin to if we started feeding the colour image recognition system with black and white images. IMHO, it is necessary not to adjust the network to the old methods of binary patterns, but to highlight real, fuzzy ones on full data.


![Price Action Analysis Toolkit Development (Part 41): Building a Statistical Price-Level EA in MQL5](https://c.mql5.com/2/171/19589-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 41): Building a Statistical Price-Level EA in MQL5](https://www.mql5.com/en/articles/19589)

Statistics has always been at the heart of financial analysis. By definition, statistics is the discipline that collects, analyzes, interprets, and presents data in meaningful ways. Now imagine applying that same framework to candlesticks—compressing raw price action into measurable insights. How helpful would it be to know, for a specific period of time, the central tendency, spread, and distribution of market behavior? In this article, we introduce exactly that approach, showing how statistical methods can transform candlestick data into clear, actionable signals.

![Functions for activating neurons during training: The key to fast convergence?](https://c.mql5.com/2/112/Functions_of_neuronal_activation_during_learning___LOGO.png)[Functions for activating neurons during training: The key to fast convergence?](https://www.mql5.com/en/articles/16845)

This article presents a study of the interaction of different activation functions with optimization algorithms in the context of neural network training. Particular attention is paid to the comparison of the classical ADAM and its population version when working with a wide range of activation functions, including the oscillating ACON and Snake functions. Using a minimalistic MLP (1-1-1) architecture and a single training example, the influence of activation functions on the optimization is isolated from other factors. The article proposes an approach to manage network weights through the boundaries of activation functions and a weight reflection mechanism, which allows avoiding problems with saturation and stagnation in training.

![Simplifying Databases in MQL5 (Part 2): Using metaprogramming to create entities](https://c.mql5.com/2/171/19594-simplifying-databases-in-mql5-logo.png)[Simplifying Databases in MQL5 (Part 2): Using metaprogramming to create entities](https://www.mql5.com/en/articles/19594)

We explored the advanced use of #define for metaprogramming in MQL5, creating entities that represent tables and column metadata (type, primary key, auto-increment, nullability, etc.). We centralized these definitions in TickORM.mqh, automating the generation of metadata classes and paving the way for efficient data manipulation by the ORM, without having to write SQL manually.

![From Novice to Expert: Animated News Headline Using MQL5 (XI)—Correlation in News Trading](https://c.mql5.com/2/170/19343-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (XI)—Correlation in News Trading](https://www.mql5.com/en/articles/19343)

In this discussion, we will explore how the concept of Financial Correlation can be applied to improve decision-making efficiency when trading multiple symbols during major economic events announcement. The focus is on addressing the challenge of heightened risk exposure caused by increased volatility during news releases.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mkzpxmxftmbwukiiodhwqwekfarfnjod&ssn=1769092390594527475&ssn_dr=0&ssn_sr=0&fv_date=1769092390&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16894&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neuro-symbolic%20systems%20in%20algorithmic%20trading%3A%20Combining%20symbolic%20rules%20and%20neural%20networks%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909239028817779&fz_uniq=5049211644275435301&sv=2552)

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