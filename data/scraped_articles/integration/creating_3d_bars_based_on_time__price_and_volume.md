---
title: Creating 3D bars based on time, price and volume
url: https://www.mql5.com/en/articles/16555
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:06:07.892369
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/16555&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071573430156864142)

MetaTrader 5 / Integration


### Introduction

It has been six months since I started this project. Half a year from the idea, which seemed stupid, so I did not return to it much, only discussing it with familiar traders.

It all started with a simple question - why do traders persist in trying to analyze a 3D market by looking at 2D charts? Price action, technical analysis, wave theory - all of this works with the projection of the market onto a plane. But what if we try to see the real structure of price, volume and time?

In my work on algorithmic systems, I have consistently encountered the fact that traditional indicators miss critical relationships between price and volume.

The idea of 3D bars did not come immediately. First, there was an experiment with 3D visualization of market depth. Then the first sketches of volume-price clusters appeared. And when I added the time component and built the first 3D bar, it became obvious that this was a fundamentally new way of seeing the market.

Today I want to share with you the results of this work. I will show you how Python and MetaTrader 5 allow you to build volume bars in real time. I will talk about math behind the calculations and how to use this information in practical trading.

### What is different about the 3D bar?

As long as we look at the market through the prism of two-dimensional charts, we miss the most important thing - its real structure. Traditional technical analysis works with price-time, volume-time projections, but never shows the full picture of the interaction of these components.

3D analysis is fundamentally different in that it allows us to see the market as a whole. When we construct a volume bar, we are literally creating a "snapshot" of the market state, where each dimension carries critical information:

- the height of the bar shows the amplitude of the price movement
- the width reflects the time scale
- depth visualizes volume distribution

Why is this important? Imagine two identical price movements on a chart. In two dimensions they look identical. But when we add the volume component, the picture changes dramatically - one move can be supported by massive volume, forming a deep and stable bar, while another turns out to be a superficial splash with minimal support for real trades.

An integrated approach using 3D bars solves a classic problem of technical analysis – signal lag. The volumetric structure of the bar begins to form from the first ticks, allowing us to see the emergence of a strong movement long before it appears on a regular chart. In essence, we get a predictive analytics tool based not on historical patterns, but on the real dynamics of current trades.

Multivariate data analysis is more than just pretty visualization; it is a fundamentally new way of understanding market microstructure. Each 3D bar contains information about:

- distribution of volume within the price range
- positions accumulation speed
- imbalances between buyers and sellers
- micro level volatility
- movement momentum

All these components work as a single mechanism, allowing you to see the true nature of price movement. Where classical technical analysis sees just a candle or a bar, 3D analysis shows the complex structure of the interaction of supply and demand.

### Equations for calculating the main metrics. Basic principles of constructing 7D bars. The logic of combining different dimensions into a single system

The mathematical model of 3D bars grew out of the analysis of real market microstructure. Each bar in the system can be represented as a three-dimensional figure, where:

```
class Bar3D:
    def __init__(self):
        self.price_range = None  # Price range
        self.time_period = None  # Time interval
        self.volume_profile = {} # Volume profile by prices
        self.direction = None    # Movement direction
        self.momentum = None     # Impulse
        self.volatility = None   # Volatility
        self.spread = None       # Average spread
```

The key point is the calculation of the volumetric profile inside the bar. Unlike classic bars, we analyze the distribution of volume by price levels.

```
def calculate_volume_profile(self, ticks_data):
    volume_by_price = defaultdict(float)

    for tick in ticks_data:
        price_level = round(tick.price, 5)
        volume_by_price[price_level] += tick.volume

    # Normalize the profile
    total_volume = sum(volume_by_price.values())
    for price in volume_by_price:
        volume_by_price[price] /= total_volume

    return volume_by_price
```

The momentum is calculated as a combination of the rate of change of price and volume:

```
def calculate_momentum(self):
    price_velocity = (self.close - self.open) / self.time_period
    volume_intensity = self.total_volume / self.time_period
    self.momentum = price_velocity * volume_intensity * self.direction
```

Particular attention is paid to the analysis of intra-bar volatility. We use a modified ATR equation that takes into account the microstructure of the move:

```
def calculate_volatility(self, tick_data):
    tick_changes = np.diff([tick.price for tick in tick_data])
    weighted_std = np.std(tick_changes * [tick.volume for tick in tick_data[1:]])
    time_factor = np.sqrt(self.time_period)
    self.volatility = weighted_std * time_factor
```

The fundamental difference from classic bars is that all metrics are calculated in real time, allowing us to see the formation of the bar structure:

```
def update_bar(self, new_tick):
    self.update_price_range(new_tick.price)
    self.update_volume_profile(new_tick)
    self.recalculate_momentum()
    self.update_volatility(new_tick)

    # Recalculate the volumetric center of gravity
    self.volume_poc = self.calculate_poc()
```

All measurements are combined through a system of weighting factors adjusted for a specific instrument:

```
def calculate_bar_strength(self):
    return (self.momentum_weight * self.normalized_momentum +
            self.volatility_weight * self.normalized_volatility +
            self.volume_weight * self.normalized_volume_concentration +
            self.spread_weight * self.normalized_spread_factor)
```

In real trading, this mathematical model allows us to see such aspects of the market as:

- imbalances in volume accumulation
- anomalies in the speed of price formation
- consolidation and breakout zones
- true strength of a trend through volume characteristics

Each 3D bar becomes not just a point on the chart, but a full-fledged indicator of the state of the market at a specific moment in time.

### A detailed analysis of the algorithm for creating 3D bars. Features of working with MetaTrader 5. Data handling specifics

After debugging the main algorithm, I finally got to the most interesting part - the implementation of multidimensional bars in real time. I admit, at first it seemed like a daunting task. MetaTrader 5 is not particularly friendly to external scripts, and the documentation sometimes fails to provide proper understanding. But let me tell you how I managed to overcome this in the end.

I started with a basic structure for storing data. After several iterations, the following class was born:

```
class Bar7D:
    def __init__(self):
        self.time = None
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.tick_volume = 0
        self.volume_profile = {}
        self.direction = 0
        self.trend_count = 0
        self.volatility = 0
        self.momentum = 0
```

The hardest part was figuring out how to correctly calculate the block size. After a lot of experimentation, I settled on this equation:

```
def calculate_brick_size(symbol_info, multiplier=45):
    spread = symbol_info.spread
    point = symbol_info.point
    min_price_brick = spread * multiplier * point

    # Adaptive adjustment for volatility
    atr = calculate_atr(symbol_info.name)
    if atr > min_price_brick * 2:
        min_price_brick = atr / 2

    return min_price_brick
```

I also had a lot of trouble with the volumes. At first, I wanted to use a fixed size volume\_brick, but quickly realized that it does not work. The solution came in the form of an adaptive algorithm:

```
def adaptive_volume_threshold(tick_volume, history_volumes):
    median_volume = np.median(history_volumes)
    std_volume = np.std(history_volumes)

    if tick_volume > median_volume + 2 * std_volume:
        return median_volume + std_volume
    return max(tick_volume, median_volume / 2)
```

But I think I went a bit overboard with the calculation of statistical metrics:

```
def calculate_stats(df):
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['volume_ma_5'] = df['tick_volume'].rolling(5).mean()
    df['price_volatility'] = df['price_change'].rolling(10).std()
    df['volume_volatility'] = df['tick_volume'].rolling(10).std()
    df['trend_strength'] = df['trend_count'] * df['direction']

    # This is probably too much
    df['zscore_price'] = stats.zscore(df['close'], nan_policy='omit')
    df['zscore_volume'] = stats.zscore(df['tick_volume'], nan_policy='omit')
    return df
```

It is funny, but the hardest part was not writing the code, but debugging it in real conditions.

Here is the final result of the function featuring normalization in the range 3-9. Why 3-9? Both Gann and Tesla claimed that there was some kind of magic hidden in these numbers. I have also personally seen a trader on a well known platform who allegedly created a successful reversal script based on these numbers. But let's not delve into conspiracy theories and mysticism. Try this instead:

```
def create_true_3d_renko(symbol, timeframe, min_spread_multiplier=45, volume_brick=500, lookback=20000):
    """
    Creates 3D Renko bars with extended analytics
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
    if rates is None:
        print(f"Error getting data for {symbol}")
        return None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    if df.isnull().any().any():
        print("Missing values detected, cleaning...")
        df = df.dropna()
        if len(df) == 0:
            print("No data for analysis after cleaning")
            return None, None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None, None

    try:
        min_price_brick = symbol_info.spread * min_spread_multiplier * symbol_info.point
        if min_price_brick <= 0:
            print("Invalid block size")
            return None, None
    except AttributeError as e:
        print(f"Error getting symbol parameters: {e}")
        return None, None

    # Convert time to numeric and scale everything
    scaler = MinMaxScaler(feature_range=(3, 9))

    # Convert datetime to numeric (seconds from start)
    df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()

    # Scale all numeric data together
    columns_to_scale = ['time_numeric', 'open', 'high', 'low', 'close', 'tick_volume']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    renko_blocks = []
    current_price = float(df.iloc[0]['close'])
    current_tick_volume = 0
    current_time = df.iloc[0]['time']
    current_time_numeric = float(df.iloc[0]['time_numeric'])
    current_spread = float(symbol_info.spread)
    current_type = 0
    prev_direction = 0
    trend_count = 0

    try:
        for idx, row in df.iterrows():
            if pd.isna(row['tick_volume']) or pd.isna(row['close']):
                continue

            current_tick_volume += float(row['tick_volume'])
            volume_bricks = int(current_tick_volume / volume_brick)

            price_diff = float(row['close']) - current_price
            if pd.isna(price_diff) or pd.isna(min_price_brick):
                continue

            price_bricks = int(price_diff / min_price_brick)

            if volume_bricks > 0 or abs(price_bricks) > 0:
                direction = np.sign(price_bricks) if price_bricks != 0 else 1

                if direction == prev_direction:
                    trend_count += 1
                else:
                    trend_count = 1

                renko_block = {
                    'time': current_time,
                    'time_numeric': float(row['time_numeric']),
                    'open': float(row['open']),
                    'close': float(row['close']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'tick_volume': float(row['tick_volume']),
                    'direction': float(direction),
                    'spread': float(current_spread),
                    'type': float(current_type),
                    'trend_count': trend_count,
                    'price_change': price_diff,
                    'volume_intensity': float(row['tick_volume']) / volume_brick,
                    'price_velocity': price_diff / (volume_bricks if volume_bricks > 0 else 1)
                }

                if volume_bricks > 0:
                    current_tick_volume = current_tick_volume % volume_brick
                if price_bricks != 0:
                    current_price += min_price_brick * price_bricks

                prev_direction = direction
                renko_blocks.append(renko_block)

    except Exception as e:
        print(f"Error processing data: {e}")
        if len(renko_blocks) == 0:
            return None, None

    if len(renko_blocks) == 0:
        print("Failed to create any blocks")
        return None, None

    result_df = pd.DataFrame(renko_blocks)

    # Scale derived metrics to same range
    derived_metrics = ['price_change', 'volume_intensity', 'price_velocity', 'spread']
    result_df[derived_metrics] = scaler.fit_transform(result_df[derived_metrics])

    # Add analytical metrics using scaled data
    result_df['ma_5'] = result_df['close'].rolling(5).mean()
    result_df['ma_20'] = result_df['close'].rolling(20).mean()
    result_df['volume_ma_5'] = result_df['tick_volume'].rolling(5).mean()
    result_df['price_volatility'] = result_df['price_change'].rolling(10).std()
    result_df['volume_volatility'] = result_df['tick_volume'].rolling(10).std()
    result_df['trend_strength'] = result_df['trend_count'] * result_df['direction']

    # Scale moving averages and volatility
    ma_columns = ['ma_5', 'ma_20', 'volume_ma_5', 'price_volatility', 'volume_volatility', 'trend_strength']
    result_df[ma_columns] = scaler.fit_transform(result_df[ma_columns])

    # Add statistical metrics and scale them
    result_df['zscore_price'] = stats.zscore(result_df['close'], nan_policy='omit')
    result_df['zscore_volume'] = stats.zscore(result_df['tick_volume'], nan_policy='omit')
    zscore_columns = ['zscore_price', 'zscore_volume']
    result_df[zscore_columns] = scaler.fit_transform(result_df[zscore_columns])

    return result_df, min_price_brick
```

And this is what the series of bars we obtained on a single scale looks like. Not very stationary, is it?

![](https://c.mql5.com/2/158/EURUSD_3d_main__6.png)

Statistical distributions:

![](https://c.mql5.com/2/158/EURUSD_stats__2.png)

Naturally, I was not satisfied with such a series, because my goal was to create a more or less stationary series - a stationary, time-volume-price series. Here is what I did next:

### Introducing volatility measurement

While implementing the create\_stationary\_4d\_features function, I took a fundamentally different path. Unlike the original 3D bars where we simply scaled the data into the 3-9 range, here I focused on creating truly stationary series.

The key idea of the function is to create a four-dimensional representation of the market through stationary features. Instead of simply scaling, each dimension is transformed in a special way to achieve stationarity:

1. Time dimension: Here I applied the trigonometric transformation, converting the hours into sine and cosine waves. sin(2π \* hour/24) and cos(2π \* hour/24) equations create cyclical features, completely eliminating the problem of daily seasonality.
2. Price measurement: instead of absolute price values, their relative changes are used. In the code, this is implemented by calculating the typical price (high + low + close)/3 and then calculating the returns and their acceleration. This approach makes the series stationary regardless of the price level.
3. Volumetric measurement: here is an interesting point - we take not just changes in volumes, but their relative increments. This is important because volumes are often very unevenly distributed. In the code, this is implemented through the successive application of pct\_change() and diff() .
4. Measuring volatility: Here I have implemented a two-step transformation - first calculating the running volatility through the standard deviation of returns, and then taking the relative changes in that volatility. In effect, we get "volatility of volatility".

Each data block is formed in a sliding window of 20 periods. This is not a random number - it is chosen as a compromise between preserving the local structure of the data and ensuring statistical significance of the calculations.

All calculated features are ultimately scaled to the range 3-9, but this is already a secondary transformation applied to already stationary series. This allows us to maintain compatibility with the original implementation of 3D bars while using a fundamentally different approach to data preprocessing.

A particularly important point is to preserve all the key metrics from the original function - moving averages, volatility, z-scores. This allows the new implementation to be used as a direct replacement for the original function, while obtaining stationary data of higher quality.

As a result, we obtain a set of features that are not only stationary in the statistical sense, but also retain all the important information about the market structure. This approach makes the data much more suitable for applying machine learning and statistical analysis methods, while still maintaining its connection to the original trading context.

Here is the function:

```
def create_true_3d_renko(symbol, timeframe, min_spread_multiplier=45, volume_brick=500, lookback=20000):
    """
    Creates 4D stationary features with same interface as 3D Renko
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
    if rates is None:
        print(f"Error getting data for {symbol}")
        return None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    if df.isnull().any().any():
        print("Missing values detected, cleaning...")
        df = df.dropna()
        if len(df) == 0:
            print("No data for analysis after cleaning")
            return None, None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None, None

    try:
        min_price_brick = symbol_info.spread * min_spread_multiplier * symbol_info.point
        if min_price_brick <= 0:
            print("Invalid block size")
            return None, None
    except AttributeError as e:
        print(f"Error getting symbol parameters: {e}")
        return None, None

    scaler = MinMaxScaler(feature_range=(3, 9))
    df_blocks = []

    try:
        # Time dimension
        df['time_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
        df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()

        # Price dimension
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_return'] = df['typical_price'].pct_change()
        df['price_acceleration'] = df['price_return'].diff()

        # Volume dimension
        df['volume_change'] = df['tick_volume'].pct_change()
        df['volume_acceleration'] = df['volume_change'].diff()

        # Volatility dimension
        df['volatility'] = df['price_return'].rolling(20).std()
        df['volatility_change'] = df['volatility'].pct_change()

        for idx in range(20, len(df)):
            window = df.iloc[idx-20:idx+1]

            block = {
                'time': df.iloc[idx]['time'],
                'time_numeric': scaler.fit_transform([[float(df.iloc[idx]['time_numeric'])]]).item(),
                'open': float(window['price_return'].iloc[-1]),
                'high': float(window['price_acceleration'].iloc[-1]),
                'low': float(window['volume_change'].iloc[-1]),
                'close': float(window['volatility_change'].iloc[-1]),
                'tick_volume': float(window['volume_acceleration'].iloc[-1]),
                'direction': np.sign(window['price_return'].iloc[-1]),
                'spread': float(df.iloc[idx]['time_sin']),
                'type': float(df.iloc[idx]['time_cos']),
                'trend_count': len(window),
                'price_change': float(window['price_return'].mean()),
                'volume_intensity': float(window['volume_change'].mean()),
                'price_velocity': float(window['price_acceleration'].mean())
            }
            df_blocks.append(block)

    except Exception as e:
        print(f"Error processing data: {e}")
        if len(df_blocks) == 0:
            return None, None

    if len(df_blocks) == 0:
        print("Failed to create any blocks")
        return None, None

    result_df = pd.DataFrame(df_blocks)

    # Scale all features
    features_to_scale = [col for col in result_df.columns if col != 'time' and col != 'direction']
    result_df[features_to_scale] = scaler.fit_transform(result_df[features_to_scale])

    # Add same analytical metrics as in original function
    result_df['ma_5'] = result_df['close'].rolling(5).mean()
    result_df['ma_20'] = result_df['close'].rolling(20).mean()
    result_df['volume_ma_5'] = result_df['tick_volume'].rolling(5).mean()
    result_df['price_volatility'] = result_df['price_change'].rolling(10).std()
    result_df['volume_volatility'] = result_df['tick_volume'].rolling(10).std()
    result_df['trend_strength'] = result_df['trend_count'] * result_df['direction']

    # Scale moving averages and volatility
    ma_columns = ['ma_5', 'ma_20', 'volume_ma_5', 'price_volatility', 'volume_volatility', 'trend_strength']
    result_df[ma_columns] = scaler.fit_transform(result_df[ma_columns])

    # Add statistical metrics and scale them
    result_df['zscore_price'] = stats.zscore(result_df['close'], nan_policy='omit')
    result_df['zscore_volume'] = stats.zscore(result_df['tick_volume'], nan_policy='omit')
    zscore_columns = ['zscore_price', 'zscore_volume']
    result_df[zscore_columns] = scaler.fit_transform(result_df[zscore_columns])

    return result_df, min_price_brick
```

This is what it looks like in 2D:

![](https://c.mql5.com/2/158/EURUSD_trends__2.png)

Next, let's try to create an interactive 3D model for 3D prices using plotly. A regular two-dimensional chart should be visible nearby. Here is the code:

```
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_3d(df, symbol, save_dir):
    """
    Creates interactive 3D visualization with smoothed data and original price chart
    """
    try:
        save_dir = Path(save_dir)

        # Smooth all series with MA(100)
        df_smooth = df.copy()
        smooth_columns = ['close', 'tick_volume', 'price_volatility', 'volume_volatility']

        for col in smooth_columns:
            df_smooth[f'{col}_smooth'] = df_smooth[col].rolling(window=100, min_periods=1).mean()

        # Create subplots: 3D view and original chart side by side
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=(f'{symbol} 3D View (MA100)', f'{symbol} Original Price'),
            horizontal_spacing=0.05
        )

        # Add 3D scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=np.arange(len(df_smooth)),
                y=df_smooth['tick_volume_smooth'],
                z=df_smooth['close_smooth'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_smooth['price_volatility_smooth'],
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(x=0.45)
                ),
                hovertemplate=
                "Time: %{x}<br>" +
                "Volume: %{y:.2f}<br>" +
                "Price: %{z:.5f}<br>" +
                "Volatility: %{marker.color:.5f}",
                name='3D View'
            ),
            row=1, col=1
        )

        # Add original price chart
        fig.add_trace(
            go.Candlestick(
                x=np.arange(len(df)),
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            ),
            row=1, col=2
        )

        # Add smoothed price line
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(df_smooth)),
                y=df_smooth['close_smooth'],
                line=dict(color='blue', width=1),
                name='MA100'
            ),
            row=1, col=2
        )

        # Update 3D layout
        fig.update_scenes(
            xaxis_title='Time',
            yaxis_title='Volume',
            zaxis_title='Price',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )

        # Update 2D layout
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Price", row=1, col=2)

        # Update overall layout
        fig.update_layout(
            width=1500,  # Double width to accommodate both plots
            height=750,
            showlegend=True,
            title_text=f"{symbol} Combined Analysis"
        )

        # Save interactive HTML
        fig.write_html(save_dir / f'{symbol}_combined_view.html')

        # Create additional plots with smoothed data (unchanged)
        fig2 = make_subplots(rows=2, cols=2,
                            subplot_titles=('Smoothed Price', 'Smoothed Volume',
                                          'Smoothed Price Volatility', 'Smoothed Volume Volatility'))

        fig2.add_trace(
            go.Scatter(x=np.arange(len(df_smooth)), y=df_smooth['close_smooth'],
                      name='Price MA100'),
            row=1, col=1
        )

        fig2.add_trace(
            go.Scatter(x=np.arange(len(df_smooth)), y=df_smooth['tick_volume_smooth'],
                      name='Volume MA100'),
            row=1, col=2
        )

        fig2.add_trace(
            go.Scatter(x=np.arange(len(df_smooth)), y=df_smooth['price_volatility_smooth'],
                      name='Price Vol MA100'),
            row=2, col=1
        )

        fig2.add_trace(
            go.Scatter(x=np.arange(len(df_smooth)), y=df_smooth['volume_volatility_smooth'],
                      name='Volume Vol MA100'),
            row=2, col=2
        )

        fig2.update_layout(
            height=750,
            width=750,
            showlegend=True,
            title_text=f"{symbol} Smoothed Data Analysis"
        )

        fig2.write_html(save_dir / f'{symbol}_smoothed_analysis.html')

        print(f"Interactive visualizations saved in {save_dir}")

    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        raise
```

This is what our new price range looks like:

YouTube

Overall, it looks very interesting. We see certain sequences of price grouping by time, and outliers in price grouping by volume. So, a feeling is created (and directly confirmed by the experience of leading traders) that when the market is restless, when huge volumes are being pumped out, when volatility is rushing in, we are dealing with a dangerous outburst that goes beyond statistics - the notorious tail risks. Therefore, here we can immediately detect such an "abnormal" exit of the price on such coordinates. I would like to thank the idea of multivariate price charts for this alone!

Please note:

![](https://c.mql5.com/2/158/lo9jda7w_01-12-2024_015955__2.jpg)

### Examining the patient (3D graphics)

Next, I suggest visualizing. But not our bright future under a palm tree, but 3D price charts. Let's break the situations down into four clusters: uptrend, downtrend, reversal from uptrend to downtrend, and reversal from downtrend to uptrend. To do this, we will need to change the code a little: we no longer need the bar indices, we will load data on specific dates. Actually, to do this we just need to go to mt5.copy\_rates\_range.

```
def create_true_3d_renko(symbol, timeframe, start_date, end_date, min_spread_multiplier=45, volume_brick=500):
    """
    Creates 4D stationary features with same interface as 3D Renko
    """
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        print(f"Error getting data for {symbol}")
        return None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    if df.isnull().any().any():
        print("Missing values detected, cleaning...")
        df = df.dropna()
        if len(df) == 0:
            print("No data for analysis after cleaning")
            return None, None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None, None

    try:
        min_price_brick = symbol_info.spread * min_spread_multiplier * symbol_info.point
        if min_price_brick <= 0:
            print("Invalid block size")
            return None, None
    except AttributeError as e:
        print(f"Error getting symbol parameters: {e}")
        return None, None

    scaler = MinMaxScaler(feature_range=(3, 9))
    df_blocks = []

    try:
        # Time dimension
        df['time_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
        df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()

        # Price dimension
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_return'] = df['typical_price'].pct_change()
        df['price_acceleration'] = df['price_return'].diff()

        # Volume dimension
        df['volume_change'] = df['tick_volume'].pct_change()
        df['volume_acceleration'] = df['volume_change'].diff()

        # Volatility dimension
        df['volatility'] = df['price_return'].rolling(20).std()
        df['volatility_change'] = df['volatility'].pct_change()

        for idx in range(20, len(df)):
            window = df.iloc[idx-20:idx+1]

            block = {
                'time': df.iloc[idx]['time'],
                'time_numeric': scaler.fit_transform([[float(df.iloc[idx]['time_numeric'])]]).item(),
                'open': float(window['price_return'].iloc[-1]),
                'high': float(window['price_acceleration'].iloc[-1]),
                'low': float(window['volume_change'].iloc[-1]),
                'close': float(window['volatility_change'].iloc[-1]),
                'tick_volume': float(window['volume_acceleration'].iloc[-1]),
                'direction': np.sign(window['price_return'].iloc[-1]),
                'spread': float(df.iloc[idx]['time_sin']),
                'type': float(df.iloc[idx]['time_cos']),
                'trend_count': len(window),
                'price_change': float(window['price_return'].mean()),
                'volume_intensity': float(window['volume_change'].mean()),
                'price_velocity': float(window['price_acceleration'].mean())
            }
            df_blocks.append(block)

    except Exception as e:
        print(f"Error processing data: {e}")
        if len(df_blocks) == 0:
            return None, None

    if len(df_blocks) == 0:
        print("Failed to create any blocks")
        return None, None

    result_df = pd.DataFrame(df_blocks)

    # Scale all features
    features_to_scale = [col for col in result_df.columns if col != 'time' and col != 'direction']
    result_df[features_to_scale] = scaler.fit_transform(result_df[features_to_scale])

    # Add same analytical metrics as in original function
    result_df['ma_5'] = result_df['close'].rolling(5).mean()
    result_df['ma_20'] = result_df['close'].rolling(20).mean()
    result_df['volume_ma_5'] = result_df['tick_volume'].rolling(5).mean()
    result_df['price_volatility'] = result_df['price_change'].rolling(10).std()
    result_df['volume_volatility'] = result_df['tick_volume'].rolling(10).std()
    result_df['trend_strength'] = result_df['trend_count'] * result_df['direction']

    # Scale moving averages and volatility
    ma_columns = ['ma_5', 'ma_20', 'volume_ma_5', 'price_volatility', 'volume_volatility', 'trend_strength']
    result_df[ma_columns] = scaler.fit_transform(result_df[ma_columns])

    # Add statistical metrics and scale them
    result_df['zscore_price'] = stats.zscore(result_df['close'], nan_policy='omit')
    result_df['zscore_volume'] = stats.zscore(result_df['tick_volume'], nan_policy='omit')
    zscore_columns = ['zscore_price', 'zscore_volume']
    result_df[zscore_columns] = scaler.fit_transform(result_df[zscore_columns])

    return result_df, min_price_brick
```

Here is our modified code:

```
def main():
    try:
        # Initialize MT5
        if not mt5.initialize():
            print("MetaTrader5 initialization error")
            return

        # Analysis parameters
        symbols = ["EURUSD", "GBPUSD"]
        timeframes = {
            "M15": mt5.TIMEFRAME_M15
        }

        # 7D analysis parameters
        params = {
            "min_spread_multiplier": 45,
            "volume_brick": 500
        }

        # Date range for data fetching
        start_date = datetime(2017, 1, 1)
        end_date = datetime(2018, 2, 1)

        # Analysis for each symbol and timeframe
        for symbol in symbols:
            print(f"\nAnalyzing symbol {symbol}")

            # Create symbol directory
            symbol_dir = Path('charts') / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"Failed to get symbol info for {symbol}")
                continue

            print(f"Spread: {symbol_info.spread} points")
            print(f"Tick: {symbol_info.point}")

            # Analysis for each timeframe
            for tf_name, tf in timeframes.items():
                print(f"\nAnalyzing timeframe {tf_name}")

                # Create timeframe directory
                tf_dir = symbol_dir / tf_name
                tf_dir.mkdir(exist_ok=True)

                # Get and analyze data
                print("Getting data...")
                df, brick_size = create_true_3d_renko(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                    min_spread_multiplier=params["min_spread_multiplier"],
                    volume_brick=params["volume_brick"]
                )

                if df is not None and brick_size is not None:
                    print(f"Created {len(df)} 7D bars")
                    print(f"Block size: {brick_size}")

                    # Basic statistics
                    print("\nBasic statistics:")
                    print(f"Average volume: {df['tick_volume'].mean():.2f}")
                    print(f"Average trend length: {df['trend_count'].mean():.2f}")
                    print(f"Max uptrend length: {df[df['direction'] > 0]['trend_count'].max()}")
                    print(f"Max downtrend length: {df[df['direction'] < 0]['trend_count'].max()}")

                    # Create visualizations
                    print("\nCreating visualizations...")
                    create_visualizations(df, symbol, tf_dir)

                    # Save data
                    csv_file = tf_dir / f"{symbol}_{tf_name}_7d_data.csv"
                    df.to_csv(csv_file)
                    print(f"Data saved to {csv_file}")

                    # Results analysis
                    trend_ratio = len(df[df['direction'] > 0]) / len(df[df['direction'] < 0])
                    print(f"\nUp/Down bars ratio: {trend_ratio:.2f}")

                    volume_corr = df['tick_volume'].corr(df['price_change'].abs())
                    print(f"Volume-Price change correlation: {volume_corr:.2f}")

                    # Print warnings if anomalies detected
                    if df['price_volatility'].max() > df['price_volatility'].mean() * 3:
                        print("\nWARNING: High volatility periods detected!")

                    if df['volume_volatility'].max() > df['volume_volatility'].mean() * 3:
                        print("WARNING: Abnormal volume spikes detected!")
                else:
                    print(f"Failed to create 3D bars for {symbol} on {tf_name}")

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        mt5.shutdown()
```

Let's take the first data section - EURUSD, from January 1, 2017 to February 1, 2018. In fact, a very powerful bullish trend. Ready to see what it looks like in 3D bars?

![](https://c.mql5.com/2/158/j9khxtm7r2__2.jpg)

![](https://c.mql5.com/2/158/dhas_ckrea__2.jpg)

Here's what another visualization looks like:

![](https://c.mql5.com/2/158/EURUSD_3d_main__7.png)

Let's pay attention to the beginning of the uptrend:

![](https://c.mql5.com/2/158/70d68z_nn39bv__2.jpg)

And to the end of it:

![](https://c.mql5.com/2/158/2htvk_xz1bp8__2.jpg)

Now let's look at the downtrend. From February 1, 2018 to March 20, 2020:

![](https://c.mql5.com/2/158/EURUSD_3d_main__8.png)

Here is the beginning of the bearish trend:

![](https://c.mql5.com/2/158/nxe312_j64ncnkopr__2.jpg)

And here is its end:

![](https://c.mql5.com/2/158/ciq8q_wh00onb6__2.jpg)

So what we see is that both trends (both bearish and bullish) in the 3D representation started as an area of dots below the 3D dot density. The end of the trend in both cases was marked by a bright yellow color scheme.

To describe this phenomenon, and the behavior of the prices of EURUSD in bullish and bearish trends, the following universal equation can be used:

### P(t) = P\_0 + \\int\_{t\_0}^{t} A \\cdot e^{k(t-u)} \\cdot V(u) \\, du + N(t)

where:

-  P(t) — currency price at a certain time.
-  P\_0 — initial price at a certain time.
-  A — trend amplitude, which characterizes the scale of price changes.
-  k — ratio that determines the rate of change (k > 0 means bullish trend; k < 0 means bearish trend).
-  V(u) — trading volume at a given time, which influences market activity and can increase the significance of price changes.
-  N(t) — random noise that reflects unpredictable market fluctuations.

**Text explanation**

This equation describes how the price of a currency changes over time, depending on a number of factors. The initial price is the starting point, after which the integral takes into account the influence of the trend amplitude and its rate of change, subjecting the price to exponential growth or decline depending on the magnitude. The trading volume represented by the function adds another dimension, showing that market activity also influences price changes.

This model thus allows visualization of price movements under different trends, displaying them in 3D space, where the time axis, price and volume create a rich picture of market activity. The brightness of the color scheme in this pattern can indicate the strength of the trend, with brighter colors corresponding to higher derivative price and trading volume values, signaling strong volume movements in the market.

### Displaying reversal

Here is the period from November 14 to November 28. We will have a reversal in quotes approximately in the middle of this period of time. What does this look like in 3D coordinates? Here it is:

![](https://c.mql5.com/2/158/1__2.jpg)

We see the already familiar yellow color at the moment of reversal and increase in the normalized price coordinate. Now let's look at another price section with a trend reversal, from September 13, 2024 to October 10 of the same year:

![](https://c.mql5.com/2/158/2__4.jpg)

We can see the same picture again, only the yellow color and its accumulation are now at the bottom. Looks interesting.

August 19, 2024 - August 30, 2024, an exact reversal of the trend can be seen in the middle of this date range. Let's look at our coordinates.

![](https://c.mql5.com/2/158/3__4.jpg)

Again, exactly the same picture. Let's consider the period from July 17, 2024 to August 8, 2024. Will the model show signs of a reversal soon?

![](https://c.mql5.com/2/158/4__2.jpg)

The last period is from April 21 to August 10, 2023. The bullish trend ended there.

![](https://c.mql5.com/2/158/5__4.jpg)

We see the familiar yellow color again.

### Yellow clusters

While developing 3D bars, I came across a very interesting feature - yellow volume-volatile clusters. I was captivated by their behavior on the chart! After going through a ton of historical data (more than 400,000 bars for 2022-2024, to be precise), I noticed something surprising.

At first I could not believe my eyes - out of about 100 thousand yellow bars, almost all (97%!) were near price reversals. Moreover, this worked in a range of plus or minus three bars. Interestingly, only 40% of reversals (and there were about 169 thousand of them in total) displayed yellow bars. It turns out that a yellow bar almost guarantees a reversal, although reversals can happen without them.

![](https://c.mql5.com/2/158/reversals_comparison__2.png)

Digging further into the trends, I noticed a clear pattern. At the beginning and during the trend, there are almost no yellow bars, only regular 3D bars in a dense group. But before the reversal, the yellow clusters shine on the chart.

This is especially clearly visible in long trends. Take, for example, the growth of EURUSD from the beginning of 2017 to February 2018, and then the fall until March 2020. In both cases, these yellow clusters appeared before the reversal, and their 3D placement literally indicated where the price would go!

I tested this thing on short periods too - I took several 2-3 week segments in 2024. It worked like clockwork! Each time before a reversal, yellow bars would appear, as if warning: "Hey, man, the trend is about to reverse!"

This is not just some indicator. I think, we have hit on something really important in the market structure itself - the way the volumes are distributed, and the volatility changes before a trend change. Now when I see yellow clusters on a 3D chart, I know it is time to prepare for a reversal!

### Conclusion

As we conclude our exploration of 3D bars, I cannot help but remark on how deeply this dive has changed my understanding of market microstructure. What started as an experiment in visualization has evolved into a fundamentally new way of seeing and understanding the market.

While working on this project, I kept coming across how limited we are by the traditional two-dimensional representation of prices. The move to three-dimensional analysis has opened up entirely new horizons for understanding the relationships between price, volume and time. I was particularly struck by how clearly patterns preceding important market events appeared in three-dimensional space.

The most significant discovery was the ability to detect potential trend reversals early. The characteristic accumulation of volumes and the change in color scheme in the 3D representation have proven to be surprisingly reliable indicators of upcoming trend changes. This is not just a theoretical observation - we have confirmed it with many historical examples.

The mathematical model we have developed allows us not only to visualize, but also to quantitatively evaluate market dynamics. The integration of modern visualization technologies and software tools has made it possible to apply this method in real trading. I use these tools daily and they have made a huge difference in my approach to market analysis.

However, I believe that we are only at the beginning of the journey. This project opened the door to the world of multivariate market microstructure analysis, and I am confident that further research in this direction will yield many more interesting discoveries. Perhaps, the next step will be the integration of machine learning to automatically recognize 3D patterns or the development of new trading strategies based on multivariate analysis.

Ultimately, the real value of this research is not in the pretty charts or complex equations, but in the new market insights it provides. As a researcher, I strongly believe that the future of technical analysis lies in a multivariate approach to analyzing market data.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16555](https://www.mql5.com/ru/articles/16555)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16555.zip "Download all attachments in the single ZIP archive")

[3D\_Bars\_Visual.py](https://www.mql5.com/en/articles/download/16555/3d_bars_visual.py "Download 3D_Bars_Visual.py")(19.48 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/491511)**
(4)


![Bogard_11](https://c.mql5.com/avatar/avatar_na2.png)

**[Bogard\_11](https://www.mql5.com/en/users/bogard_11)**
\|
4 Dec 2024 at 11:39

The question immediately arises - **why**? A flat graph is not enough for accurate analysis? Regular high school [geometry](https://www.mql5.com/en/articles/2742 "Article: Statistical Distributions in MQL5 - Taking the Best of R and Making it Faster ") works there.


![Thibauld Charles Ghislain Robin](https://c.mql5.com/avatar/2022/9/6315d50b-bd00.png)

**[Thibauld Charles Ghislain Robin](https://www.mql5.com/en/users/candlexbomb)**
\|
2 Feb 2025 at 08:28

**Bogard\_11 [#](https://www.mql5.com/ru/forum/477605#comment_55297456):**

The question immediately arises - **why**? A flat graph is not enough for accurate analysis? That's where regular high school geometry works.

Any algorithm essentially explores spatial dimensions. By creating algorithms, we are trying to solve the fundamental problem of combinatorial explosion through multidimensional search. It's our way of navigating an infinite sea of possibilities.

(Apologies if the translation is not perfect )

![Bogard_11](https://c.mql5.com/avatar/avatar_na2.png)

**[Bogard\_11](https://www.mql5.com/en/users/bogard_11)**
\|
2 Feb 2025 at 17:36

**Thibauld Charles Ghislain Robin [#](https://www.mql5.com/ru/forum/477605#comment_55795910):**

Any algorithm essentially explores spatial dimensions. By creating algorithms, we are trying to solve the fundamental problem of combinatorial explosion through multidimensional search. It is our way of navigating an infinite sea of possibilities.

(Apologies if the translation is not perfect )

Understood. If we can't solve trend forecasting through simple school geometric formulas, people start inventing a Lysaped with turbo supercharging, with smartphone control, with smiley faces and other tinsel! Except there are no wheels, and they're not expected to have wheels. And without wheels, you can't go far on one frame.

![BeeXXI Corporation](https://c.mql5.com/avatar/2024/9/66dbee89-a47e.png)

**[Nikolai Semko](https://www.mql5.com/en/users/nikolay7ko)**
\|
2 Feb 2025 at 18:21

**Bogard\_11 [#](https://www.mql5.com/ru/forum/477605#comment_55798520):**

I see. If it is impossible to solve trend forecasting through simple school geometric formulas, people start inventing a lisaped with turbo supercharging, with smartphone control, with smiley faces and other tinsel! Except there are no wheels, and they're not expected to have wheels. And without wheels, you can't go far on one frame.

That's a load of bollocks

I can only sympathise with someone who is born with a 4-dimensional mechanism of perception, but thinks only in two-dimensional concepts.

![Neural Networks in Trading: Enhancing Transformer Efficiency by Reducing Sharpness (SAMformer)](https://c.mql5.com/2/102/Neural_Networks_in_Trading__Improving_Transformer_Efficiency_by_Reducing_Sharpness___LOGO.png)[Neural Networks in Trading: Enhancing Transformer Efficiency by Reducing Sharpness (SAMformer)](https://www.mql5.com/en/articles/16388)

Training Transformer models requires large amounts of data and is often difficult since the models are not good at generalizing to small datasets. The SAMformer framework helps solve this problem by avoiding poor local minima. This improves the efficiency of models even on limited training datasets.

![Data Science and ML (Part 46): Stock Markets Forecasting Using N-BEATS in Python](https://c.mql5.com/2/157/18242-data-science-and-ml-part-46-logo.png)[Data Science and ML (Part 46): Stock Markets Forecasting Using N-BEATS in Python](https://www.mql5.com/en/articles/18242)

N-BEATS is a revolutionary deep learning model designed for time series forecasting. It was released to surpass classical models for time series forecasting such as ARIMA, PROPHET, VAR, etc. In this article, we are going to discuss this model and use it in predicting the stock market.

![MQL5 Trading Tools (Part 5): Creating a Rolling Ticker Tape for Real-Time Symbol Monitoring](https://c.mql5.com/2/158/18844-mql5-trading-tools-part-5-creating-logo.png)[MQL5 Trading Tools (Part 5): Creating a Rolling Ticker Tape for Real-Time Symbol Monitoring](https://www.mql5.com/en/articles/18844)

In this article, we develop a rolling ticker tape in MQL5 for real-time monitoring of multiple symbols, displaying bid prices, spreads, and daily percentage changes with scrolling effects. We implement customizable fonts, colors, and scroll speeds to highlight price movements and trends effectively.

![Price Action Analysis Toolkit Development (Part 32): Python Candlestick Recognition Engine (II) — Detection Using Ta-Lib](https://c.mql5.com/2/157/18824-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 32): Python Candlestick Recognition Engine (II) — Detection Using Ta-Lib](https://www.mql5.com/en/articles/18824)

In this article, we’ve transitioned from manually coding candlestick‑pattern detection in Python to leveraging TA‑Lib, a library that recognizes over sixty distinct patterns. These formations offer valuable insights into potential market reversals and trend continuations. Follow along to learn more.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/16555&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071573430156864142)

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