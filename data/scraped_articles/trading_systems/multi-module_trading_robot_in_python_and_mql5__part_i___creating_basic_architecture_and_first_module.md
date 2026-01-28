---
title: Multi-module trading robot in Python and MQL5 (Part I): Creating basic architecture and first modules
url: https://www.mql5.com/en/articles/16667
categories: Trading Systems, Integration, Machine Learning
relevance_score: 9
scraped_at: 2026-01-22T17:33:32.546007
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/16667&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049215342242277178)

MetaTrader 5 / Trading systems


### Introduction

One day an idea struck me: trading robots are too simple for the modern market, something more flexible and smart is needed.

The market is constantly changing. One strategy works today becoming useless tomorrow. I watched this for a long time and realized that a completely new approach was needed. The solution came unexpectedly. What if we make a modular system? Imagine a team of professionals: one monitors trends, the second analyzes trading volumes, the third controls risks. This is exactly how a modern trading robot should work!

The choice of technologies was obvious. Python turned out to be perfect for data analysis - you can do wonders with its libraries. MQL5 took over the execution of trades. A great tandem appeared. We started small: first we created a solid foundation - an architecture that could grow and evolve, then we added interaction between Python and MQL5. The data management system turned out to be surprisingly simple and effective.

Asynchrony was a real breakthrough! Now the robot could monitor multiple instruments simultaneously. Productivity skyrocketed.

Do you know what's most interesting? This system really works in the market. It is not just a textbook example, but an actual trading tool. Of course, we will start with the basic version, but even that is impressive. We have a great journey ahead of us. We will create a system capable of learning and adapting. We will improve it step by step. For now, let's start with the most important thing - building a solid foundation.

### Basic architecture of the system. In search of the perfect balance

For three years I struggled to create trading robots. I came to realize that the main thing is not the algorithms themselves, but how they work together. This discovery changed everything.

Imagine a symphony orchestra. Every musician is great, but without a conductor there is no music. In my system, MarketMaker became such a conductor. It controls four modules, each of which knows its own business:

- The first module monitors trading volumes: when and at what prices transactions take place.
- The second module looks for arbitrage opportunities.
- The third module analyzes the economy.
- The fourth module prevents the system from getting carried away and controls risks.

The market waits for no one. It changes at lightning speed, so all modules work simultaneously, constantly communicating with each other. Say, the arbitrage module sees the opportunity. Info from other modules is checked and a decision is made.

At first I thought about making strict rules for entering the market. But practice quickly showed that this was not possible. Sometimes one strong signal is more important than several weak ones. Arranging data took pretty much time. Each module has its own information: quotes, macro indicators, transaction history. All this needs to be stored, updated, shared with others. It was necessary to create a special synchronization system.

Ironically, the more independent the modules were, the better the system worked. The failure of one component did not stop the others. But failures do happen: the connection is lost, or the quotes freeze. The main advantage of this architecture is that it can be expanded. Want to add news analysis? No problem! Create a module, connect it to MarketMaker, and everything works like a charm.

The system lives and develops. It is not perfect, but its foundation of modularity, parallelism, and flexibility allows us to look to the future with confidence. I will tell you more about each component soon.

### Main system class

After much experimentation with different approaches to the architecture of trading robots, I came to the understanding that the success of the system largely depends on how well its core is organized. MarketMaker is the result of this understanding, embodying all the best practices I have accumulated over the years of developing algorithmic systems.

Let's start with the basic structure of the class. This is what its initialization looks like:

```
def __init__(self, pairs: List[str], terminal_path: str,
             volume: float = 1.0, levels: int = 5, spacing: float = 3.0):
    # Main parameters
    self.pairs = pairs
    self.base_volume = volume
    self.levels = levels
    self.spacing = spacing
    self.magic = 12345

    # Trading parameters
    self.portfolio_iterations = 10
    self.leverage = 50
    self.min_profit_pips = 1.0
    self.max_spread_multiplier = 2.0

    # Data warehouses
    self.symbols_info = {}
    self.trading_parameters = {}
    self.optimal_horizons = {}
```

At first glance, everything looks quite simple. But behind each parameter there is a story. Take portfolio\_iterations for example - this parameter was created after I noticed that opening positions too aggressively could lead to liquidity problems. Now the system breaks down the available volume into parts, which makes trading more balanced.

I paid special attention to the initialization of historical data. How it works:

```
def _initialize_history(self, pair: str):
    """Initializing historical data for a pair"""
    try:
        rates = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1,
                                  datetime.now()-timedelta(days=1), 1440)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            self.returns_history[pair] = pd.Series(returns.values,
                                                 index=df.index[1:])
    except Exception as e:
        logger.error(f"Error initializing history for {pair}: {e}")
```

The interesting point here is the use of logarithmic returns instead of simple percentage changes. This is not a random choice. In practice, I have found that logarithmic returns provide more stable results when calculating statistical indicators, especially when it comes to volatility.

One of the most challenging aspects was implementing volume forecasting. After many experiments, the following code was born:

```
async def update_volume_predictions(self):
    """Updating volume predictions for each pair"""
    for pair in self.pairs:
        try:
            df = volume_model.get_volume_data(
                symbol=pair,
                timeframe=mt5.TIMEFRAME_H1,
                n_bars=100
            )

            if pair in self.volume_models:
                feature_columns = [\
                    'volume_sma_5', 'volume_sma_20', 'relative_volume',\
                    'volume_change', 'volume_volatility', 'price_sma_5',\
                    'price_sma_20', 'price_change', 'price_volatility',\
                    'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower'\
                ]

                X = df[feature_columns].iloc[-1:].copy()
                prediction = self.volume_models[pair].predict(X)[0]
                current_price = df['close'].iloc[-1]
                predicted_change = (prediction - current_price) / current_price

                self.volume_predictions[pair] = predicted_change

        except Exception as e:
            logger.error(f"Error updating prediction for {pair}: {e}")
```

Note that the set of features is not just a random set of indicators. Each of them was added gradually, after careful testing. For example, relative\_volume has proven to be particularly useful for identifying abnormal market activity.

And here is the heart of the system - the trading loop:

```
async def trade_cycle(self):
    """Main trading loop"""
    try:
        await self.update_volume_predictions()
        await self.economic_module.update_forecasts()

        all_positions = mt5.positions_get() or []
        open_positions = [pos for pos in all_positions if pos.magic == self.magic]

        if open_positions:
            await self.manage_positions()
            return

        valid_signals = []
        available_volume = self.calculate_available_volume() * len(self.pairs)

        for pair in self.pairs:
            signal = await self.get_combined_signal(pair)
            if signal and self._validate_signal(signal):
                valid_signals.append(signal)

        if valid_signals:
            volume_per_trade = available_volume / len(valid_signals)
            for signal in valid_signals:
                signal['adjusted_volume'] = volume_per_trade
                await self.place_order(signal)

    except Exception as e:
        logger.error(f"Error in trade cycle: {e}")
```

This code is the result of long thoughts about how to properly organize the trading process. The asynchronous nature of the loop allows for efficient handling of multiple pairs simultaneously, and the clear sequence of actions (updating forecasts → checking positions → searching for signals → execution) ensures predictable system behavior.

The signal validation mechanism deserves special attention:

```
def _validate_signal(self, signal: Dict) -> bool:
    """Trading signal check"""
    spread = signal['spread']
    diff_pips = signal['diff_pips']

    # Basic checks
    if spread > self.max_spread_multiplier * diff_pips:
        return False

    if diff_pips < self.min_profit_pips:
        return False

    # Check economic factors
    if signal['economic_volatility'] > self.volatility_threshold:
        return False

    # Check the volume prediction
    if abs(signal['volume_prediction']) < self.min_volume_change:
        return False

    return True
```

Every check here is a result of real trading experience. For example, the economic volatility check was added after I noticed that trading during important news events often resulted in increased losses due to sharp price movements.

In conclusion, I would like to note that MarketMaker is a living system that continues to evolve. Every day of trading brings new ideas and improvements. The modular architecture makes it easy to implement these improvements without disrupting core components.

### Handling data

Data handling has always been one of the most challenging aspects of algorithmic trading. I remember how at the beginning of the development I was faced with a seemingly simple question: how to properly organize the storage and handling of market information? It quickly became clear that a regular database or simple arrays would not be enough.

It all started with creating a basic structure to receive data. After several iterations, the following method was born:

```
def _initialize_history(self, pair: str):
    try:
        rates = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1,
                                  datetime.now()-timedelta(days=1), 1440)
        if rates is None:
            logger.error(f"Failed to get history data for {pair}")
            return

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Calculate logarithmic returns
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()

        # Add new metrics
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_velocity'] = df['close'].diff() / df['time'].diff().dt.total_seconds()
        df['volume_intensity'] = df['tick_volume'] / df['time'].diff().dt.total_seconds()

        self.returns_history[pair] = pd.Series(returns.values, index=df.index[1:])
        self.price_data[pair] = df

    except Exception as e:
        logger.error(f"Error initializing history for {pair}: {e}")
```

The interesting point here is the calculation of the "speed" of price change (price\_velocity) and the intensity of volume (volume\_intensity). These metrics did not appear immediately. Initially, I worked only with regular price data, but quickly realized that the market is not just a sequence of prices, it is a complex dynamic system where not only the magnitude of changes is important, but also the speed of these changes.

Particular attention had to be paid to handling missing data. This is what the validation and cleaning system looks like:

```
def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """Validation and data cleaning"""
    if df.empty:
        raise ValueError("Empty dataset received")

    # Check gaps
    missing_count = df.isnull().sum()
    if missing_count.any():
        logger.warning(f"Found missing values: {missing_count}")

        # Use 'forward fill' for prices
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()

        # Use interpolation for volumes
        df['tick_volume'] = df['tick_volume'].interpolate(method='linear')

    # Check outliers
    for col in ['high', 'low', 'close']:
        zscore = stats.zscore(df[col])
        outliers = abs(zscore) > 3
        if outliers.any():
            logger.warning(f"Found {outliers.sum()} outliers in {col}")

            # Replace extreme outliers
            df.loc[outliers, col] = df[col].mean() + 3 * df[col].std() * np.sign(zscore[outliers])

    return df
```

I remember a case when missing just one tick led to incorrect calculation of indicators and, as a result, to an incorrect trading signal. After this, the data cleaning system was significantly improved.

Here is how we work with volumes, one of the most important characteristics of the market:

```
def analyze_volume_profile(self, pair: str, window: int = 100) -> Dict:
    """Volume profile analysis"""
    try:
        df = self.price_data[pair].copy().last(window)

        # Normalize volumes
        volume_mean = df['tick_volume'].rolling(20).mean()
        volume_std = df['tick_volume'].rolling(20).std()
        df['normalized_volume'] = (df['tick_volume'] - volume_mean) / volume_std

        # Calculate volume clusters
        price_levels = pd.qcut(df['close'], q=10)
        volume_clusters = df.groupby(price_levels)['tick_volume'].sum()

        # Find support/resistance levels by volume
        significant_levels = volume_clusters[volume_clusters > volume_clusters.mean() + volume_clusters.std()]

        # Analyze imbalances
        buy_volume = df[df['close'] > df['open']]['tick_volume'].sum()
        sell_volume = df[df['close'] <= df['open']]['tick_volume'].sum()
        volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)

        return {
            'normalized_profile': volume_clusters.to_dict(),
            'significant_levels': significant_levels.index.to_list(),
            'volume_imbalance': volume_imbalance,
            'current_percentile': stats.percentileofscore(df['tick_volume'], df['tick_volume'].iloc[-1])
        }

    except Exception as e:
        logger.error(f"Error analyzing volume profile: {e}")
        return None
```

This code is the result of a long study of the market microstructure. Of particular interest is the calculation of the imbalance in volumes between purchases and sales. I initially studied this topic on the crypto market. I don't know if the MQL5 administration will give the go-ahead to publish the code with the integration of the crypto exchange, MetaTrader 5, and Python....

But I digress. At first glance, it may seem that simply comparing volumes on rising and falling bars will not provide useful information. But practice has shown that this simple indicator often warns of an impending trend reversal.

Working with economic data is a separate story. Here it was necessary to create a whole synchronization system:

```
async def synchronize_market_data(self):
    """Market data synchronization"""
    while True:
        try:
            # Update basic data
            for pair in self.pairs:
                latest_data = await self._get_latest_ticks(pair)
                if latest_data is not None:
                    self._update_price_data(pair, latest_data)

            # Update derived metrics
            await self._update_derivatives()

            # Check data integrity
            self._verify_data_integrity()

            await asyncio.sleep(1)  # Dynamic delay

        except Exception as e:
            logger.error(f"Error in data synchronization: {e}")
            await asyncio.sleep(5)  # Increased delay on error
```

The key point here is the asynchronicity of data updates. In early versions of the system I used synchronous requests, but this created delays when handling large numbers of pairs. The transition to an asynchronous model has significantly improved productivity.

In conclusion, I would like to note that the correct organization of work with data is not just a technical issue. This is the foundation the entire trading strategy is built on. Clean, well-structured data allows us to see market patterns that remain hidden during superficial analysis.

### First module: Volume analysis

The history of the volume analysis module development began with a simple observation: classic indicators often lag because they work only with prices. But the market is not only about prices, it is also about trading volumes, which often predict the movement of quotes. That is why the first module of our system was the volume analyzer.

Let's start with the basic data retrieval function:

```
def get_volume_data(symbol, timeframe=mt5.TIMEFRAME_H1, n_bars=2000):
    """Getting volume and price data from MT5"""
    try:
        bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if bars is None:
            logger.error(f"Failed to get data for {symbol}")
            return None

        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {e}")
        return None
```

At first glance, the function looks simple. But behind this simplicity lies an important decision: we take exactly 2000 bars of history. Why? I have experimentally found that this is sufficient for building a high-quality model, but at the same time it does not create an excessive load on the server memory in the case of training very large models, with large dataset dimensions and input features as batch sequences.

The most interesting part of the module is creating features for analysis. How it works:

```
def create_features(df, forecast_periods=None):
    """Create features for the forecasting model"""
    try:
        # Basic volume indicators
        df['volume_sma_5'] = df['tick_volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['tick_volume'].rolling(window=20).mean()
        df['relative_volume'] = df['tick_volume'] / df['volume_sma_20']

        # Volume dynamics
        df['volume_change'] = df['tick_volume'].pct_change()
        df['volume_acceleration'] = df['volume_change'].diff()

        # Volume volatility
        df['volume_volatility'] = df['tick_volume'].rolling(window=20).std()
        df['volume_volatility_5'] = df['tick_volume'].rolling(window=5).std()
        df['volume_volatility_ratio'] = df['volume_volatility_5'] / df['volume_volatility']
```

Particular attention should be paid to volume\_volatility\_ratio here. This indicator appeared after I noticed an interesting pattern: before strong movements, short-term volume volatility often begins to grow faster than long-term volatility. This indicator has become one of the key ones in identifying potential entry points.

Calculation of the volume profile is a separate story:

```
# Volume profile
        df['volume_percentile'] = df['tick_volume'].rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        df['volume_density'] = df['tick_volume'] / (df['high'] - df['low'])
        df['volume_density_ma'] = df['volume_density'].rolling(window=20).mean()
        df['cumulative_volume'] = df['tick_volume'].rolling(window=20).sum()
        df['volume_ratio'] = df['tick_volume'] / df['cumulative_volume']
```

The volume\_density indicator did not appear by chance. I have noticed that volume itself can be deceiving - it is important to consider what price range it was collected at. High volume in a narrow price range often indicates the formation of an important support or resistance level.

I developed a special function to predict the direction of price movement:

```
def predict_direction(model, X):
    """Price movement direction forecast"""
    try:
        prediction = model.predict(X)[0]
        current_price = X['close'].iloc[-1] if 'close' in X else None
        if current_price is None:
            return 0

        # Return 1 for rise, -1 for fall, 0 for neutral
        price_change = (prediction - current_price) / current_price
        if abs(price_change) < 0.0001:  # Minimum change threshold
            return 0
        return 1 if price_change > 0 else -1

    except Exception as e:
        logger.error(f"Error predicting direction: {e}")
        return 0
```

Please note the change threshold is 0.0001. This is not a random number - it is chosen based on the analysis of the minimum movement that can be handled taking into account the spread and various types of commissions. For the stock market, the indicator should be selected separately.

The final stage is training the model:

```
def train_model(X_train, X_test, y_train, y_test, model_params=None):
    try:
        if model_params is None:
            model_params = {'n_estimators': 400, 'random_state': 42}

        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)

        # Model evaluation
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)
```

Do you know why I chose RandomForest with 400 trees? Having tried a bunch of things, from simple regression to neural networks of amazing complexity in architecture, I came to the conclusion that this method is the most reliable. Not the most accurate, perhaps, but stable. The market is noisy and twitching, but RandomForest is holding up well.

This is just the beginning, of course. The next questions awaiting us are how to connect all these signals together and how to set up the system so that it learns on the go. But more about that next time.

### Risk management: The art of preserving capital

And now let's consider the most important thing - the risks. It is pretty amusing to listen to everyone discussing cool strategies and neural networks. Over the course of ten years on the market, I have realized the main thing: all these strategies are worthless without risk management. You might have a super trading algorithm, but without proper risk management you will still end up in the red.

Therefore, in our system, capital protection takes the central stage. It is this conservative approach that allows us to earn consistently while others lose money on "perfect" strategies.

```
def calculate_available_volume(self) -> float:
    try:
        account = mt5.account_info()
        if not account:
            return 0.01

        # Use balance and free margin
        balance = account.balance
        free_margin = account.margin_free

        # Take the minimum value for safety
        available_margin = min(balance, free_margin)

        # Calculate the maximum volume taking into account the margin
        margin_ratio = 0.1  # Use only 10% of the available margin
        base_volume = (available_margin * margin_ratio) / 1000

        # Limit to maximum volume
        max_volume = min(base_volume, 1.0)  # max 1 lot
```

Please note that margin\_ratio = 0.1. This is not a random number. After several months of testing, I have come to the conclusion that using more than 10% of available margin significantly increases the risk of a margin call during strong market moves. This is especially critical when trading multiple pairs simultaneously.

The next important point is the calculation of stop losses and take profits:

```
async def calculate_position_limits(self, signal: Dict) -> Tuple[float, float]:
    try:
        pair = signal['pair']
        direction = signal['direction']

        # Get volatility
        volatility = signal['price_volatility']
        economic_volatility = signal['economic_volatility']

        # Base values in pips
        base_sl = 20
        base_tp = 40

        # Adjust for volatility
        volatility_factor = 1 + (volatility * 2)
        sl_points = base_sl * volatility_factor
        tp_points = base_tp * volatility_factor

        # Take economic volatility into account
        if economic_volatility > 0.5:
            sl_points *= 1.5
            tp_points *= 1.2

        # Check minimum distances
        info = self.symbols_info[pair]
        min_stop_level = info.trade_stops_level if hasattr(info, 'trade_stops_level') else 0

        return max(sl_points, min_stop_level), max(tp_points, min_stop_level)

    except Exception as e:
        logger.error(f"Error calculating position limits: {e}")
        return 20, 40  # return base values in case of an error
```

The story with volatility\_factor is particularly interesting. Initially I used fixed stop levels, but quickly noticed that during periods of high volatility they were often triggered too early and too often. Dynamically adjusting stop levels based on current volatility has significantly improved trading results.

And here is what the position management system looks like:

```
async def manage_positions(self):
    """Managing open positions"""
    try:
        positions = mt5.positions_get() or []
        for position in positions:
            if position.magic == self.magic:
                # Check the time in the position
                time_in_trade = datetime.now() - pd.to_datetime(position.time, unit='s')

                # Get current market data
                signal = await self.get_combined_signal(position.symbol)

                # Check the need to modify the position
                if self._should_modify_position(position, signal, time_in_trade):
                    await self._modify_position(position, signal)

                # Check the closing conditions
                if self._should_close_position(position, signal, time_in_trade):
                    await self.close_position(position)

    except Exception as e:
        logger.error(f"Error managing positions: {e}")
```

Particular attention is paid here to the time spent in the position. Experience has shown that the longer a position is open, the higher the requirements for maintaining it should be. This is achieved through dynamic tightening of position holding conditions over time.

An interesting point is related to the partial closing of positions:

```
def calculate_partial_close(self, position, profit_threshold: float = 0.5) -> float:
    """Volume calculation for partial closure"""
    try:
        # Check the current profit
        if position.profit <= 0:
            return 0.0

        profit_ratio = position.profit / (position.volume * 1000)  # approximate ROI estimate

        if profit_ratio >= profit_threshold:
            # Close half of the position when the profit threshold is reached
            return position.volume * 0.5
        return 0.0

    except Exception as e:
        logger.error(f"Error calculating partial close: {e}")
        return 0.0
```

This feature was created after analyzing the transactions. I noticed that partially closing positions when a certain profit level was reached significantly improved overall trading statistics. This allows us to lock in some of our profits while still leaving potential for further growth.

In conclusion, I would like to note that the risk management system is a living organism that is constantly evolving. Every unsuccessful trade, every unexpected market movement is a new experience that we use to improve capital protection algorithms. In the next versions of the system, I decided to add machine learning for dynamic optimization of risk management parameters, as well as a hybrid of the VaR system and Markowitz portfolio theory, but that is a completely different story....

### Economic module: When fundamental analysis meets machine learning

While working on the trading system, I noticed an interesting pattern: even the strongest technical signals can fail if they contradict fundamental factors. It was this observation that led to the creation of the economic module, a component that analyzes macroeconomic indicators and their impact on the movement of currency pairs.

Let's start with the basic structure of the module. This is what the initialization of the main economic indicators looks like:

```
def __init__(self):
    self.indicators = {
        'NY.GDP.MKTP.KD.ZG': 'GDP growth',
        'FP.CPI.TOTL.ZG': 'Inflation',
        'FR.INR.RINR': 'Real interest rate',
        'NE.EXP.GNFS.ZS': 'Exports',
        'NE.IMP.GNFS.ZS': 'Imports',
        'BN.CAB.XOKA.GD.ZS': 'Current account balance',
        'GC.DOD.TOTL.GD.ZS': 'Government debt',
        'SL.UEM.TOTL.ZS': 'Unemployment rate',
        'NY.GNP.PCAP.CD': 'GNI per capita',
        'NY.GDP.PCAP.KD.ZG': 'GDP per capita growth'
    }
```

The choice of these indicators is not random. After analyzing thousands of trades, I have noticed that these are the indicators that have the greatest influence on long-term trends of currency pairs. Of particular interest is the relationship between the real interest rate and currency movements, with changes in this indicator often preceding a trend reversal.

I developed a special method to obtain economic data:

```
def fetch_economic_data(self):
    data_frames = []
    for indicator, name in self.indicators.items():
        try:
            data_frame = wbdata.get_dataframe({indicator: name}, country='all')
            data_frames.append(data_frame)
        except Exception as e:
            logger.error(f"Error fetching data for indicator '{indicator}': {e}")

    if data_frames:
        self.economic_data = pd.concat(data_frames, axis=1)
        return self.economic_data
```

The interesting point here is the use of the wbdata library to get the World Bank data. I chose this source after experimenting with different APIs as it provides the most complete and verified data.

I paid special attention to preparing the data for analysis:

```
def prepare_data(self, symbol_data):
    data = symbol_data.copy()
    data['close_diff'] = data['close'].diff()
    data['close_corr'] = data['close'].rolling(window=30).corr(data['close'].shift(1))

    for indicator in self.indicators.keys():
        if indicator in self.economic_data.columns:
            data[indicator] = self.economic_data[indicator].ffill()

    data.dropna(inplace=True)
    return data
```

Note the use of 'forward fill' for economic indicators. This solution did not come immediately - at first I tried interpolation, but it turned out that for economic data it is more correct to use the last known value.

The heart of the module is the forecasting system:

```
def forecast(self, symbol, symbol_data):
    if len(symbol_data) < 50:
        return None, None

    X = symbol_data.drop(columns=['close'])
    y = symbol_data['close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=8, loss_function='RMSE')
    model.fit(X_train, y_train, verbose=False)
```

The choice of CatBoost as a machine learning algorithm is also not accidental. After experimenting with different models (from simple linear regression to complex neural networks), it turned out that CatBoost copes best with the irregular nature of economic data.

The final stage is interpretation of the results:

```
def interpret_results(self, symbol):
    forecast = self.forecasts.get(symbol)
    importance_df = self.feature_importances.get(symbol)

    if forecast is None or importance_df is None:
        return f"Insufficient data for interpretation of {symbol}"

    trend = "upward" if forecast[-1] > forecast[0] else "downward"
    volatility = "high" if forecast.std() / forecast.mean() > 0.1 else "low"
    top_feature = importance_df.iloc[0]['feature']
```

The calculation of volatility is especially interesting. The threshold of 0.1 for defining high volatility was chosen after analyzing historical data. It turned out that this value distinguishes well between periods of calm market and increased turbulence.

While working on the module, I made an interesting observation: economic factors often act with a delay, but their influence is more stable than the influence of technical factors. This has led to the creation of a weighting system where the importance of economic signals increases on longer timeframes.

Of course, the economic module is not a magic wand and it cannot predict all market movements. But when combined with technical and volume analysis, it provides an additional dimension to understanding market processes. In future versions of the system, I am going to add analysis of news flows and their impact on economic indicators, but this is a topic for a separate discussion.

### Arbitrage module: In search of an actual price

The idea of creating an arbitrage module came to me after long observations of the currency market. I have noticed an interesting pattern: the actual prices of currency pairs often deviate from their theoretical value calculated through cross rates. These deviations create arbitrage opportunities, but more importantly, they can serve as an indicator of future price movement.

Let's start with the basic structure of the module:

```
class ArbitrageModule:
    def __init__(self, terminal_path: str = "C:/Program Files/RannForex MetaTrader 5/terminal64.exe", max_trades: int = 10):
        self.terminal_path = terminal_path
        self.MAX_OPEN_TRADES = max_trades
        self.symbols = [\
            "AUDUSD.ecn", "AUDJPY.ecn", "CADJPY.ecn", "AUDCHF.ecn", "AUDNZD.ecn",\
            "USDCAD.ecn", "USDCHF.ecn", "USDJPY.ecn", "NZDUSD.ecn", "GBPUSD.ecn",\
            "EURUSD.ecn", "CADCHF.ecn", "CHFJPY.ecn", "NZDCAD.ecn", "NZDCHF.ecn",\
            "NZDJPY.ecn", "GBPCAD.ecn", "GBPCHF.ecn", "GBPJPY.ecn", "GBPNZD.ecn",\
            "EURCAD.ecn", "EURCHF.ecn", "EURGBP.ecn", "EURJPY.ecn", "EURNZD.ecn"\
        ]
```

Particular attention should be paid to the structure of base pairs for calculating cross rates:

```
     self.usd_pairs = {
            "EUR": "EURUSD.ecn",
            "GBP": "GBPUSD.ecn",
            "AUD": "AUDUSD.ecn",
            "NZD": "NZDUSD.ecn",
            "USD": None,
            "CAD": ("USDCAD.ecn", True),
            "CHF": ("USDCHF.ecn", True),
            "JPY": ("USDJPY.ecn", True)
        }
```

The interesting thing here is that some pairs are marked as inverse (True). This is no coincidence - for some currencies, such as CAD, CHF and JPY, the base quote is USD/XXX, not XXX/USD. This is an important nuance that is often missed when calculating cross rates.

The heart of the module is the function for calculating synthetic prices:

```
def calculate_synthetic_prices(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculation of synthetic prices through cross rates"""
    synthetic_prices = {}

    try:
        for symbol in self.symbols:
            base = symbol[:3]
            quote = symbol[3:6]

            # Calculate the synthetic price using cross rates
            fair_price = self.calculate_cross_rate(base, quote, data)
            synthetic_prices[f'{symbol}_fair'] = pd.Series([fair_price])
```

I remember struggling to optimize this code. Initially, I tried to calculate all possible conversion routes and choose the optimal one. But it turned out that simple calculation via USD gives more stable results, especially in case of high volatility.

The function for calculating the exchange rate relative to USD is also interesting:

```
def get_usd_rate(self, currency: str, data: dict) -> float:
    """Get exchange rate to USD"""
    if currency == "USD":
        return 1.0

    pair_info = self.usd_pairs[currency]
    if isinstance(pair_info, tuple):
        pair, inverse = pair_info
        rate = data[pair]['close'].iloc[-1]
        return 1 / rate if inverse else rate
    else:
        pair = pair_info
        return data[pair]['close'].iloc[-1]
```

This feature was created after long experiments with different methods of calculating cross rates. The key point here is the correct handling of inverse pairs. An incorrect calculation on even one pair can cause a cascade of errors in synthetic prices.

I developed a special function to handle real data:

```
def get_mt5_data(self, symbol: str, count: int = 1000) -> Optional[pd.DataFrame]:
    try:
        timezone = pytz.timezone("Etc/UTC")
        utc_from = datetime.now(timezone) - timedelta(days=1)

        ticks = mt5.copy_ticks_from(symbol, utc_from, count, mt5.COPY_TICKS_ALL)
        if ticks is None:
            logger.error(f"Failed to fetch data for {symbol}")
            return None

        ticks_frame = pd.DataFrame(ticks)
        ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s')
        return ticks_frame
```

The choice of the number of ticks (1000) is a compromise between the accuracy of calculations and the speed of data handling. In practice, this has proven to be sufficient to reliably determine a fair price.

While working on the module, I made an interesting observation: discrepancies between the real and synthetic price often appear before significant market movements. It is as if smart money starts moving some pairs, creating tension in the cross rate system, which is then discharged by a strong move.

Of course, the arbitrage module is not a magic wand, but when combined with volume analysis and economic indicators, it provides an additional dimension to understanding the market. In future versions, I am going to add the analysis of correlations between deviations in different pairs, and this is a completely different story.

### Conclusion

When I started this project, I had no idea what it would turn into. I thought I would just connect Python with MQL5 and that would be it. And it turned out to be a whole trading platform! Each piece of it is like a detail in a Swiss watch, and this part is only the first of many articles.

During the development, I learned a lot. For example, that there are no easy ways in algorithmic trading. Take, for example, the calculation of the position volume. It is not that difficult, right? But when you start taking into account all the risks and market behavior, your head starts spinning.

And how great the modular architecture works! If one module fails, the others continue to work. You can safely improve each part without fear of breaking the entire system.

The most interesting thing is to observe how the different parts of the system work together. One module looks for arbitrage, another monitors volumes, the third one analyzes the economy, while the fourth one controls risks. Together they see the market in a way that no single analysis can.

Of course, there is still room for growth. I would like to add news analysis, improve machine learning, and develop new risk assessment models. It is especially interesting to work on 3D visualization of the market - to imagine price, volume and time in one space.

The main lesson of this project is that the trading system should be alive. The market does not stand still, and the system must change with it. Learn from mistakes, find new patterns, discard outdated approaches.

I hope my experience will be useful to those who create trading algorithms. And remember - this journey has no finish line. There is only path!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16667](https://www.mql5.com/ru/articles/16667)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16667.zip "Download all attachments in the single ZIP archive")

[economic\_predict.py](https://www.mql5.com/en/articles/download/16667/economic_predict.py "Download economic_predict.py")(7.6 KB)

[arbitrage\_mt5.py](https://www.mql5.com/en/articles/download/16667/arbitrage_mt5.py "Download arbitrage_mt5.py")(5.45 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/494183)**
(1)


![linfo2](https://c.mql5.com/avatar/2023/4/6438c14d-e2f0.png)

**[linfo2](https://www.mql5.com/en/users/neilhazelwood)**
\|
29 Aug 2025 at 04:06

Thank you , Trying to learn  python , your arbitrage\_mt5 does not compile  AttributeError: 'ArbitrageModule' object has no attribute 'run' , what is intended here ?


![Building a Trading System (Part 3): Determining Minimum Risk Levels for Realistic Profit Targets](https://c.mql5.com/2/165/19141-building-a-trading-system-part-logo__1.png)[Building a Trading System (Part 3): Determining Minimum Risk Levels for Realistic Profit Targets](https://www.mql5.com/en/articles/19141)

Every trader's ultimate goal is profitability, which is why many set specific profit targets to achieve within a defined trading period. In this article, we will use Monte Carlo simulations to determine the optimal risk percentage per trade needed to meet trading objectives. The results will help traders assess whether their profit targets are realistic or overly ambitious. Finally, we will discuss which parameters can be adjusted to establish a practical risk percentage per trade that aligns with trading goals.

![MetaTrader Meets Google Sheets with Pythonanywhere: A Guide to Secure Data Flow](https://c.mql5.com/2/165/19175-metatrader-meets-google-sheets-logo.png)[MetaTrader Meets Google Sheets with Pythonanywhere: A Guide to Secure Data Flow](https://www.mql5.com/en/articles/19175)

This article demonstrates a secure way to export MetaTrader data to Google Sheets. Google Sheet is the most valuable solution as it is cloud based and the data saved in there can be accessed anytime and from anywhere. So traders can access trading and related data exported to google sheet and do further analysis for future trading anytime and wherever they are at the moment.

![Automating Trading Strategies in MQL5 (Part 29): Creating a price action Gartley Harmonic Pattern system](https://c.mql5.com/2/165/19111-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 29): Creating a price action Gartley Harmonic Pattern system](https://www.mql5.com/en/articles/19111)

In this article, we develop a Gartley Pattern system in MQL5 that identifies bullish and bearish Gartley harmonic patterns using pivot points and Fibonacci ratios, executing trades with precise entry, stop loss, and take-profit levels. We enhance trader insight with visual feedback through chart objects like triangles, trendlines, and labels to clearly display the XABCD pattern structure.

![From Novice to Expert: Mastering Detailed Trading Reports with Reporting EA](https://c.mql5.com/2/165/19006-from-novice-to-expert-mastering-logo.png)[From Novice to Expert: Mastering Detailed Trading Reports with Reporting EA](https://www.mql5.com/en/articles/19006)

In this article, we delve into enhancing the details of trading reports and delivering the final document via email in PDF format. This marks a progression from our previous work, as we continue exploring how to harness the power of MQL5 and Python to generate and schedule trading reports in the most convenient and professional formats. Join us in this discussion to learn more about optimizing trading report generation within the MQL5 ecosystem.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/16667&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049215342242277178)

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