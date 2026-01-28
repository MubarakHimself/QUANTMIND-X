---
title: Trend strength and direction indicator on 3D bars
url: https://www.mql5.com/en/articles/16719
categories: Trading, Trading Systems, Integration, Machine Learning
relevance_score: 4
scraped_at: 2026-01-23T17:35:22.262295
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/16719&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068415207560116520)

MetaTrader 5 / Trading


### Introduction

One might think nothing new can be found in ordinary candles. Everything has already been discovered, counted and digitized. But as soon as we look at the market from a different angle, it reveals its completely unexpected side.

Imagine that you look at the chart not as a flat picture, but as a living, breathing organism. Each bar is not just a rectangle with shadows, but a volumetric structure pulsating in time with the market heartbeat. That is how [the idea of 3D bars](https://www.mql5.com/en/articles/16555) was born. At first, it was just a visualization experiment - I wanted to look at familiar data in a different way. But the deeper I delved into my research, the more striking patterns emerged.

I remember the moment I first saw the ["yellow" cluster](https://www.mql5.com/en/articles/16580). On the 3D chart it literally glowed, foreshadowing a trend reversal. At first, I thought it was a coincidence. But the pattern repeated itself over and over again, pointing with astonishing accuracy to future price movements. Six months of continuous research, hundreds of sleepless nights, thousands of lines of code - all this gradually formed into a coherent mathematical model.

Now, looking at the test results, I understand that we really hit on something important. This is something that lies in the very structure of the market, deep in its nature. Conventional technical analysis is powerless here - these patterns can only be seen through the prism of tensor analysis - by rising above the plane of the chart into the third dimension.

In this article, I want to share my discovery. I want to show how ordinary market data, when examined from a new angle, can give us amazingly accurate signals about the strength and direction of a trend ahead of time, when there is still time to take a position and wait for a movement to occur. Fasten your seat belts, we are about to embark on a journey through the 3D market.

### Structure of the basic market state tensor

Do you know how we used to solve the Rubik's cube when we were kids? At first it seems like complete chaos. But once you grasp the principle, all the facets begin to come together to form a single picture. It is the same here. I started collecting data into a three-dimensional structure - a tensor. It sounds complicated, but at its core it is just a way to see how price, volume and time influence each other.

The first experiments were not very impressive. Mathematics stubbornly refused to form a beautiful equation, this endless stream of numbers was so infuriating. And then... then I just stopped thinking about them as numbers.

Imagine that each candle is not just a set of open-high-low-close values, but a living organism. It has volume, like the mass of a body. There is an impulse - like movement. And there is an internal structure - like DNA. Once I started looking at the data from this angle, everything fell into place.

The result was a "cube" like this:

- One side is classic price data
- The second is volumes, but not just the number of transactions, but their internal structure.
- The third is the time loops that I could not catch for a long time.

The most amazing thing started when I launched the first test of this model. The chart literally came to life. Where I had previously seen just lines, a clear volumetric structure now appeared. And it moved! It was as if it was pulsating in time with some internal rhythms of the market.

But the main thing is that this structure began to show strange patterns. At first, I mistook them for visualization artifacts. But the more data I ran through the model, the clearer the pattern became. These patterns appeared shortly before strong price movements. As if the market... was warning of its intentions?

Let's move on to how I brought this data to a common denominator. This is a separate story, and it began with a chance discovery in Gann's old works...

### Data normalization using the Gann method

I came across Gann's work completely by accident. I was leafing through archived PDFs in search of completely different material, and suddenly my gaze caught on his strange graphs. Squares, angles, some spirals... My first thought was - this is yet another market mystic. But something made me dig deeper. I wrote an [article](https://www.mql5.com/en/articles/15556) about Gann's methods.

And you know what? Beneath all this geometric frills was a stunningly elegant idea of data normalization. Gann intuitively grasped what I was trying to reach mathematically - the principle of market scale invariance.

I spent three weeks going through his notes. Half of it had to be discarded as outright esotericism. But the rest of it... damn, there was something real there! I was particularly struck by his approach to time loops. I remember jumping out of bed in the middle of the night and running to the computer to check a sudden guess.

It turns out that if you scale the time intervals correctly, the market begins to exhibit an almost crystalline structure. It is like looking at a snowflake under a microscope - each new zoom reveals the same patterns, just at a different size.

I took his basic principles and reworked them to fit my own model. Instead of Gann's "magic" numbers, I used dynamic ratios calculated on the basis of volatility. Each tensor parameter was now normalized not to a fixed scale, but to a "floating" range, which itself adjusted to the current state of the market.

It was like tuning a musical instrument. You know that feeling when the strings finally start to sound in unison? I felt pretty much the same when I saw the first results. The chart no longer fell apart into separate elements - it began to breathe as a single whole.

The most challenging part was finding the right balance between normalization sensitivity and model robustness. Make it too fine-tuned and the system begins to react to market noise. Make it too rough and important signals are lost. I spent two weeks adjusting these parameters until I found the golden mean.

But the real breakthrough came when I applied this normalization to the volume component of the trend. And then the most interesting thing began...

### Calculation of the trend volumetric component

This is where the fun begins. After normalizing the data, I encountered an unexpected problem - classic volume indicators simply "did not see" the key moments of trend reversal. I remember spending a week trying to modify the OBV and MFI. The result was mediocre.

And then, digging through the source code of an ancient indicator (whose author can no longer be found), I came across an interesting approach to calculating a volumetric profile. The idea was simple to the point of genius - to look not at the absolute volume values, but at their relationship to the moving average. Here is the code:

```
def _calculate_components(self, df: pd.DataFrame) -> pd.DataFrame:
    # Basic components
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['momentum'] = df['close'].pct_change(5)

    # Here it is, the key place - the volumetric profile
    df['volume_ma'] = df['tick_volume'].rolling(20).mean()
    df['volume_trend'] = df['tick_volume'] / df['volume_ma']

    # Trend strength as a derivative of three components
    df['trend_force'] = df['volatility'] * df['volume_trend'] * abs(df['momentum'])
```

Look what is happening here. Instead of simply summing the volumes, we create something like "volume acceleration". When the volume increases sharply relative to its average, this is the first warning sign. But the interesting part comes when we add volatility and momentum to the mix.

I tried a lot of periods for moving averages. 10, 15, 25... In the end, 20 bars provided the best balance between sensitivity and signal stability.

But the real "wow" effect happened when I added the trading sessions ratio:

```
# Activity ratios of different sessions
df['session_coef'] = 1.0
hour = df.index.hour

df.loc[(hour >= 0) & (hour < 8), 'session_coef'] = 0.7    # Asian
df.loc[(hour >= 8) & (hour < 16), 'session_coef'] = 1.0   # European
df.loc[(hour >= 16) & (hour < 24), 'session_coef'] = 0.9  # American
```

The chart literally came to life. Now every surge in volume was considered in the context of the current trading session. It is like a tide rising and falling. The same is here - each session has its own character, its own "power of attraction".

But most importantly, this equation began to show what I called "harbingers". A few bars before a strong move, the volume profile began to form a characteristic pattern. It is as if the market is "taking its breath" before jumping.

Then the question arose of how to visualize all this correctly...

### Price dynamics in three-dimensional space

It took me a long time to find the right way to visualize all this madness. The first attempts to build a 3D chart in MatPlotLib looked more like a Christmas tree than something useful for trading. Plotly did not give in right away either - the charts were either too busy or missing important details.

And then chance came to my aid. I was playing with a construction set with my daughter, building something like a bridge, and suddenly I realized that we do not need all this fancy 3D graphics. We only need to arrange the projections correctly! This is what I got:

```
def create_visualization(self, df: pd.DataFrame = None):
    if df is None:
        df = self.analyze_market()

    df = df.reset_index()

    # Three projections of our "bridge"
    fig = make_subplots(rows=3, cols=1,
                       shared_xaxes=True,
                       subplot_titles=('Price', 'Trend Force', 'Trend Direction'),
                       row_heights=[0.5, 0.25, 0.25],
                       vertical_spacing=0.05)

    # Main chart - classic candles
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
```

Look what happens - we take three projections of the same space. The top one shows the usual candles, but this is just the tip of the iceberg. The most interesting part starts in the second window:

```
# The trend force is our main feature
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['trend_force_adjusted'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Trend Force'
        ),
        row=2, col=1
    )

    # Reference levels
    fig.add_hline(y=3, line_dash="dash", line_color="yellow", row=2, col=1)
    fig.add_hline(y=6, line_dash="dash", line_color="green", row=2, col=1)
```

I found these levels (3 and 6) empirically. When the trend strength breaks through level 6, it almost always means a strong move. Level 3 is something like a turbulence zone, where the trend can either strengthen or reverse.

But the real magic happens in the bottom window:

```
# Trend direction as a derivative of force
    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['trend_direction'] * df['trend_force_adjusted'],
            name='Trend Direction',
            marker_color=np.where(df['trend_direction'] > 0, 'green', 'red')
        ),
        row=3, col=1
    )
```

Here we see not just the direction of the trend, but its strength in dynamics. When green bars rise against a high Trend Force value, it is a strong buy signal. Conversely, rising red bars against a strength above 6 are a sure sell signal.

![](https://c.mql5.com/2/167/y3mdc2xy_21-12-2024_083333__1.jpg)

But let's get back to the ground. After visualization, the question of time loops arose...

### Time component and trading sessions

One thing always surprised me about classical technical analysis. It is how easily everyone forgets about time. Traders look at charts, calculate indicators, but miss a simple fact: the market lives in different time zones.

The first warning bell rang when I noticed a strange pattern - my signals were working much better during the European session. At first, I wrote it off as a coincidence. But then I started digging deeper, and this is what I found:

```
# See how easy it is to take into account the impact of sessions
hour = df.index.hour

# Asian session is the calmest
asian_mask = (hour >= 0) & (hour < 8)
df.loc[asian_mask, 'session_coef'] = 0.7

# Europe - peak activity
european_mask = (hour >= 8) & (hour < 16)
df.loc[european_mask, 'session_coef'] = 1.0

# America - still active, but not as strong
american_mask = (hour >= 16) & (hour < 24)
df.loc[american_mask, 'session_coef'] = 0.9
```

I came up with these ratios almost on the fly, while testing different options. I remember sitting up all night running backtests with different values. I just could not tear myself away from the monitor – the results were so exciting.

But the most interesting thing began when I superimposed these ratios on the volumetric profile. Suddenly everything fell into place! It turned out that the same volume has completely different "weight" in different sessions:

- During the Asian session, even a small surge in volume can be significant.
- The European one requires much more significant deviations
- And at the junction of sessions something really interesting happens...

But the real breakthrough came when I added another component - "intersession transitions". I noticed that 30-40 minutes before the opening of a new session the indicator starts to behave... strangely. It is as if the market is preparing for the arrival of new players. And that is where...

### Trend strength integral indicator

Sometimes the most important discoveries come from unfortunate mistakes. In my case, it all started with a bug in the code. I accidentally multiplied the wrong variables and the graph showed some wild anomaly. My first impulse was to rewrite everything, but something made me take a closer look...

It turned out that I had accidentally created what I later called an "integral indicator." Let's have a look:

```
def _calculate_components(self, df: pd.DataFrame) -> pd.DataFrame:
    # Here it is, that very "error" - the multiplication of three components
    df['trend_force'] = df['volatility'] * df['volume_trend'] * abs(df['momentum'])

    # Normalize the result to a range of 3 to 9
    df['trend_force_norm'] = self.scaler.fit_transform(
        df['trend_force'].values.reshape(-1, 1)
    ).flatten()

    # Final adjustments considering sessions
    df['trend_force_adjusted'] = df['trend_force_norm'] * df['session_coef']
```

Do you know what is going on here? Volatility is multiplied by the volume trend and the absolute value of momentum. In theory, this should have resulted in complete chaos. But in practice... In practice, it turned out to be a surprisingly clear indicator of trend strength!

I remember my surprise when I started testing this equation on historical data. The chart showed clear peaks exactly where strong moves began. Not post factum, but several bars before the start of the movement!

The most interesting thing started when I added normalization via MinMaxScaler. I chose the range from 3 to 9 almost at random - it just seemed like the chart would be easier to read that way. And suddenly I discovered that these numbers create almost perfect levels for decision making:

- Below 3 - the market is "sleeping"
- From 3 to 6 - the movement begins
- Above 6 - the trend has gained full strength

And when I put session ratios on it... This is where I was genuinely amazed! The signals became so clear that even my eternally skeptical neighbor trader whistled when looking at the backtests.

But the main discovery was yet to come. It turned out that this indicator does not just measure the strength of a trend - it can predict reversals...

### Determining the direction of future movement

After discovering the integral indicator, I literally became obsessed with searching for reversal patterns. I spent weeks sitting in front of the monitor, moving charts back and forth. My wife was already starting to worry. Sometimes, I even forgot to eat when I found something interesting.

And then one night (why do all important discoveries happen at night?) I noticed a strange pattern. Before strong trend reversals, the strength indicator began... no, not to fall, as one might think. It began to oscillate in a special way:

```
# Determining the trend direction
df['trend_direction'] = np.sign(df['momentum'])

# This is where the magic begins
df['direction_strength'] = df['trend_direction'] * df['trend_force_adjusted']

# Looking for reversal patterns
df['reversal_pattern'] = np.where(
    (df['trend_force_adjusted'] > 6) &  # Strong trend
    (df['direction_strength'].diff().rolling(3).std() > 1.5),  # Directional instability
    1, 0
)
```

Look what happens: when the trend strength exceeds level 6 (remember our normalization?), but the direction becomes unstable, this almost always foretells a reversal!

I spent two weeks re-testing this observation on different timeframes and instruments. It worked everywhere, but the results were especially clear on H1 and H4. It almost seemed as it is on these timeframes that the market "thinks" most rationally.

But the real insight came when I superimposed a volumetric profile on top of it:

```
df['volume_confirmation'] = np.where(
    (df['reversal_pattern'] == 1) &
    (df['volume_trend'] > df['volume_trend'].rolling(20).mean() * 1.5),
    'Strong',
    'Weak'
)
```

And then everything fell into place! It turns out that not all turns are the same. When the pattern coincides with a strong excess of volume above the average, it is almost a guaranteed reversal. And if the volume does not support it, we most likely have a correction.

I remember showing these results to my former statistics teacher. He looked at the equations for a long time, then looked up at me and asked: "Have you checked this on data before 2020?" I nodded. "And after?" I nodded again. "Hm... You know, there's something to it. This goes against random walk, but... it sure seems like the truth!"

Of course, there were some false positives. But I already had a sorting system ready for this...

### Visualization of signals on the chart

When all the components of the indicator were ready, the question arose: how to show this to traders? Not everyone will understand equations and tables. I had a lot of trouble with this issue until I came up with the idea of three-level visualization.

The first attempts in pure MatPlotLib were unsuccessful. Searching for a signal was quite tiresome. I tried about a dozen libraries before settling on Plotly. This is what I ended up with:

```
def create_visualization(self, df: pd.DataFrame = None):
    if df is None:
        df = self.analyze_market()

    fig = make_subplots(rows=3, cols=1,
                       shared_xaxes=True,
                       subplot_titles=('Price', 'Trend Force', 'Trend Direction'),
                       row_heights=[0.5, 0.25, 0.25],
                       vertical_spacing=0.05)

    # Main chart - candles with signal highlighting
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            hoverlabel=dict(
                bgcolor='white',
                font=dict(size=12)
            )
        ),
        row=1, col=1
    )
```

But the main feature is interactivity. After hovering the cursor over the candle, you can immediately see all the parameters: trend force, volumes and direction. And the color scheme... I spent a week choosing shades pleasing to eyes:

```
fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['trend_force_adjusted'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Trend Force',
            hovertemplate="<br>".join([\
                "Time: %{x}",\
                "Force: %{y:.2f}",\
                "<extra></extra>"\
            ])
        ),
        row=2, col=1
    )
```

Visualizing the trend direction is a separate story. I made it in the form of bars, where the height shows the strength, and the color shows the direction:

```
fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['trend_direction'] * df['trend_force_adjusted'],
            name='Trend Direction',
            marker_color=np.where(df['trend_direction'] > 0, 'green', 'red'),
        ),
        row=3, col=1
    )
```

I remember the reaction of my trader friend when I showed him the final version. He clicked on the chart silently for about five minutes, then said: "Listen, now even a newbie can figure out where the market is going!"

But the best part was when reviews from real users started coming in. Some wrote that they finally stopped getting confused by the signals. Some thanked me for the ability to trade without sitting in front of the monitor for hours...

### Let's try to implement the trend strength indicator in MetaTrader 5

After the success with the Python version of the indicator, a logical question arose - how to transfer this to MetaTrader 5?

The task turned out to be... interesting. MQL5 is, of course, a powerful language, but without pandas and numpy I had to do a lot of fiddling. This is what I got:

```
//+------------------------------------------------------------------+
//|                                          TrendForceIndicator.mq5 |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3

// Rendering buffers
double TrendForceBuffer[];
double DirectionBuffer[];
double SignalBuffer[];

// Inputs
input int    InpMAPeriod = 20;     // Smoothing period
input int    InpMomentumPeriod = 5; // Momentum period
input double InpSignalLevel = 6.0;  // Signal level

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
    // Setting the indicator
    SetIndexBuffer(0, TrendForceBuffer, INDICATOR_DATA);
    SetIndexBuffer(1, DirectionBuffer, INDICATOR_DATA);
    SetIndexBuffer(2, SignalBuffer, INDICATOR_DATA);

    // Rendering styles
    PlotIndexSetString(0, PLOT_LABEL, "Trend Force");
    PlotIndexSetInteger(0, PLOT_DRAW_TYPE, DRAW_LINE);
    PlotIndexSetInteger(0, PLOT_LINE_COLOR, clrBlue);

    PlotIndexSetString(1, PLOT_LABEL, "Direction");
    PlotIndexSetInteger(1, PLOT_DRAW_TYPE, DRAW_HISTOGRAM);

    PlotIndexSetString(2, PLOT_LABEL, "Signal");
    PlotIndexSetInteger(2, PLOT_DRAW_TYPE, DRAW_LINE);
    PlotIndexSetInteger(2, PLOT_LINE_COLOR, clrRed);

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
    // Check for data sufficiency
    if(rates_total < InpMAPeriod) return(0);

    // Calculation of components
    int start = (prev_calculated > 0) ? prev_calculated - 1 : 0;

    for(int i = start; i < rates_total; i++)
    {
        // Volatility
        double volatility = 0.0;
        if(i >= InpMAPeriod)
        {
            double sum = 0.0;
            for(int j = 0; j < InpMAPeriod; j++)
            {
                double change = (close[i-j] - close[i-j-1]) / close[i-j-1];
                sum += change * change;
            }
            volatility = MathSqrt(sum / InpMAPeriod);
        }

        // Momentum
        double momentum = 0.0;
        if(i >= InpMomentumPeriod)
        {
            momentum = (close[i] - close[i-InpMomentumPeriod]) / close[i-InpMomentumPeriod];
        }

        // Volume trend
        double volume_ma = 0.0;
        if(i >= InpMAPeriod)
        {
            for(int j = 0; j < InpMAPeriod; j++)
            {
                volume_ma += tick_volume[i-j];
            }
            volume_ma /= InpMAPeriod;
        }

        double volume_trend = volume_ma != 0 ? (double)tick_volume[i] / volume_ma : 0;

        // Session ratio
        MqlDateTime dt;
        TimeToStruct(time[i], dt);
        double session_coef = GetSessionCoefficient(dt.hour);

        // Trend strength calculation
        TrendForceBuffer[i] = NormalizeTrendForce(volatility * MathAbs(momentum) * volume_trend) * session_coef;
        DirectionBuffer[i] = momentum > 0 ? TrendForceBuffer[i] : -TrendForceBuffer[i];

        // Signal line
        SignalBuffer[i] = InpSignalLevel;
    }

    return(rates_total);
}

//+------------------------------------------------------------------+
//| Get session ratio                                                |
//+------------------------------------------------------------------+
double GetSessionCoefficient(int hour)
{
    if(hour >= 0 && hour < 8)   return 0.7;  // Asian session
    if(hour >= 8 && hour < 16)  return 1.0;  // European session
    if(hour >= 16 && hour < 24) return 0.9;  // American session
    return 1.0;
}

//+------------------------------------------------------------------+
//| Normalization of the trend strength indicator                    |
//+------------------------------------------------------------------+
double NormalizeTrendForce(double force)
{
    // Simple normalization to range [3, 9]
    double max_force = 0.01;  // Selected empirically
    return 3.0 + 6.0 * (MathMin(force, max_force) / max_force);
}
```

The hardest part was reproducing the pandas rolling windows behavior. In MQL5, everything has to be calculated in loops, but the performance is higher - the indicator works noticeably faster than the Python version.

![](https://c.mql5.com/2/167/rmhhyx1o_21-12-2024_083943__1.jpg)

### Conclusion

At the end of this long research, I would like to share several important observations. Porting the algorithm from Python to MQL5 opened up unexpected opportunities for optimization. What initially seemed like a disadvantage (lack of familiar libraries) turned into an advantage - the code became faster and more efficient.

The most difficult thing was to find a balance between the accuracy of the signals and the speed of the indicator. Each additional parameter, each new check is an additional load on the system. Eventually, I managed to achieve an optimal ratio: the indicator processes tick data almost in real time, while maintaining high signal accuracy.

The reaction of practical traders is worth mentioning as well. When I started this project, the main goal was to create something new and interesting for myself. But as other traders started using the indicator, it became clear that we had really hit on something important. It was especially pleasant to receive feedback from experienced traders who said that the indicator helps them see the market in a new way.

Of course, this is not the Grail. Like any other technical analysis tool, our indicator requires understanding and proper application. But its main advantage is that it allows us to see those aspects of market dynamics that usually remain invisible in classical analysis.

There is still a lot of work ahead. I would like to add adaptive parameter settings, improve the signal sorting system, and maybe even integrate machine learning elements. But now, looking at the results, I understand that this path was worth it.

Most importantly, I became convinced that even in such a seemingly thoroughly studied tool as technical analysis, there is always room for innovation. We just need to not be afraid to look at familiar things from a new angle and be prepared for unexpected discoveries.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16719](https://www.mql5.com/ru/articles/16719)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16719.zip "Download all attachments in the single ZIP archive")

[Trend\_Force\_Reverse.mq5](https://www.mql5.com/en/articles/download/16719/Trend_Force_Reverse.mq5 "Download Trend_Force_Reverse.mq5")(18.38 KB)

[3D\_Bars\_Trend\_Direction.py](https://www.mql5.com/en/articles/download/16719/3D_Bars_Trend_Direction.py "Download 3D_Bars_Trend_Direction.py")(6.58 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/494738)**
(25)


![Bogard_11](https://c.mql5.com/avatar/avatar_na2.png)

**[Bogard\_11](https://www.mql5.com/en/users/bogard_11)**
\|
11 Jan 2025 at 17:10

**Vitaly Muzichenko [#](https://www.mql5.com/ru/forum/479285/page2#comment_55606376):**

But it helps you a lot, and all the other trading gurus.

Here we enter, and here we exit and earn 100500pp from each trade, and so on during the whole past period.

You know what the problem of all coders is? They have forgotten how to look for patterns.

They all follow the same pattern - now we will put an idea into the code and the machine will find the Grail! And so 100500 times in a circle. ))

But to take and check it by hand, to calculate it on a calculator.... No, it's not the honourable thing to pick a hole.... ))

That's why you don't understand that the market follows the usual laws of physics - the angle of incidence equals the angle of reflection. The graphical system was described 100 years ago. It works the way it works. But to understand it, you need to spend a decent amount of time working with history to find all mathematical connections. And especially by rearranging your thinking.

Even the first impulse carries information about future reversal levels. The error on the pound is usually no more than 2-5 pips.

![Vitaly Muzichenko](https://c.mql5.com/avatar/2025/11/691d3a3a-b70b.png)

**[Vitaly Muzichenko](https://www.mql5.com/en/users/mvs)**
\|
11 Jan 2025 at 17:23

**Bogard\_11 [#](https://www.mql5.com/ru/forum/479285/page3#comment_55607178):**

You know what the problem with all coders is? They've forgotten how to look for patterns.

They all follow the same pattern - now we'll put some idea into the code and the machine will find the Grail! And so 100500 times in a circle. ))

But to take and check it by hand, to calculate it on a calculator.... No, it's not the honourable thing to pick a hole.... ))

That's why you don't understand that the market follows the usual laws of physics - the angle of incidence equals the angle of reflection. The graphical system was described 100 years ago. It works the way it works. But to understand it, you need to spend a decent amount of time working with history to find all mathematical connections. And especially by rearranging your thinking.

Even the first impulse carries information about future reversal levels. The error on the pound is usually no more than 2-5 pips.

Several hundred different systems from different users go through coding, you test them and see all the disadvantages and advantages.

It's not manually picking and missing some patterns. A machine will never miss anything, that's its advantage.

And what worked 100 years ago stopped working as technology improved.

Can you imagine how pips would have worked before, when trades were made by telephone or by telegram to the broker?

![Bogard_11](https://c.mql5.com/avatar/avatar_na2.png)

**[Bogard\_11](https://www.mql5.com/en/users/bogard_11)**
\|
11 Jan 2025 at 18:20

**Vitaly Muzichenko [#](https://www.mql5.com/ru/forum/479285/page3#comment_55607218):**

**And what worked 100 years ago stopped working as technology improved.**

Can you elaborate on why it stopped working? Some broker or market maker cancelled the laws of physics and mathematics?!

Man, how did I miss such a scoop. Can I know when this event happened and who was the author of the cancellation?

![Bogard_11](https://c.mql5.com/avatar/avatar_na2.png)

**[Bogard\_11](https://www.mql5.com/en/users/bogard_11)**
\|
11 Jan 2025 at 18:29

**Vitaly Muzichenko [#](https://www.mql5.com/ru/forum/479285/page3#comment_55607218):**

Several hundred different systems from different users go through the coding, you test them and see all the flaws and advantages.

It's not hand picking, missing some patterns. A machine will never miss anything, that's its advantage.

To code something correctly, you must first find **ALL** patterns by hand. The machine may not be wrong, but it will only look for what you put into it, i.e. what you have found on your own without the machine! I am sometimes amazed at the fanatical belief in the abilities of AI!!!! )))

You coded to search the text for the letters **A**, **B** and **C**. Because you either found them by accident, or someone prompted you. BUT! There are other letters in the alphabet, and you don't know about them.... What will the machine give you?

It's the same in the market, there are general rules, there are mat. models, but every day they have different proportions (although all fit into one general model). Even at the price there will be a backlash, **+-3** points from the calculated value, but it will work out exactly according to the time. Or it will work out exactly at the price, but with a slippage of a couple of bars.

![imcapec](https://c.mql5.com/avatar/avatar_na2.png)

**[imcapec](https://www.mql5.com/en/users/imcapec)**
\|
27 Jan 2025 at 12:09

**Bogard\_11 [#](https://www.mql5.com/ru/forum/479285/page3#comment_55607473):**

In order to code something correctly, you must first find **all** patterns with your hands. The machine may not be wrong, but it will only look for what you put into it, i.e. what you have found on your own without the machine! I am sometimes amazed at the fanatical belief in the abilities of AI!!!! )))

You coded to search the text for the letters **A**, **B** and **C**. Because you either found them by accident, or someone prompted you. BUT! There are other letters in the alphabet, and you don't know about them.... What will the machine give you?

It's the same in the market, there are general rules, there are mat. models, but every day they have different proportions (although all fit into one general model). Even at the price there will be a backlash, **+-3** points from the calculated value, but it will work out exactly according to the time. Or it will work out exactly at the price, but with a slippage of a couple of bars.

The numerical series does not change. The commas in it change.

![Building a Professional Trading System with Heikin Ashi (Part 1): Developing a custom indicator](https://c.mql5.com/2/165/19260-building-a-professional-trading-logo.png)[Building a Professional Trading System with Heikin Ashi (Part 1): Developing a custom indicator](https://www.mql5.com/en/articles/19260)

This article is the first installment in a two-part series designed to impart practical skills and best practices for writing custom indicators in MQL5. Using Heikin Ashi as a working example, the article explores the theory behind Heikin Ashi charts, explains how Heikin Ashi candlesticks are calculated, and demonstrates their application in technical analysis. The centerpiece is a step-by-step guide to developing a fully functional Heikin Ashi indicator from scratch, with clear explanations to help readers understand what to code and why. This foundational knowledge sets the stage for Part Two, where we will build an expert advisor that trades based on Heikin Ashi logic.

![Price Action Analysis Toolkit Development (Part 38): Tick Buffer VWAP and Short-Window Imbalance Engine](https://c.mql5.com/2/166/19290-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 38): Tick Buffer VWAP and Short-Window Imbalance Engine](https://www.mql5.com/en/articles/19290)

In Part 38, we build a production-grade MT5 monitoring panel that converts raw ticks into actionable signals. The EA buffers tick data to compute tick-level VWAP, a short-window imbalance (flow) metric, and ATR-based position sizing. It then visualizes spread, ATR, and flow with low-flicker bars. The system calculates a suggested lot size and a 1R stop, and issues configurable alerts for tight spreads, strong flow, and edge conditions. Auto-trading is intentionally disabled; the focus remains on robust signal generation and a clean user experience.

![Overcoming The Limitation of Machine Learning (Part 3): A Fresh Perspective on Irreducible Error](https://c.mql5.com/2/167/19371-overcoming-the-limitation-of-logo.png)[Overcoming The Limitation of Machine Learning (Part 3): A Fresh Perspective on Irreducible Error](https://www.mql5.com/en/articles/19371)

This article takes a fresh perspective on a hidden, geometric source of error that quietly shapes every prediction your models make. By rethinking how we measure and apply machine learning forecasts in trading, we reveal how this overlooked perspective can unlock sharper decisions, stronger returns, and a more intelligent way to work with models we thought we already understood.

![Black Hole Algorithm (BHA)](https://c.mql5.com/2/107/Black_Hole_Algorithm_LOGO.png)[Black Hole Algorithm (BHA)](https://www.mql5.com/en/articles/16655)

The Black Hole Algorithm (BHA) uses the principles of black hole gravity to optimize solutions. In this article, we will look at how BHA attracts the best solutions while avoiding local extremes, and why this algorithm has become a powerful tool for solving complex problems. Learn how simple ideas can lead to impressive results in the world of optimization.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/16719&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068415207560116520)

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