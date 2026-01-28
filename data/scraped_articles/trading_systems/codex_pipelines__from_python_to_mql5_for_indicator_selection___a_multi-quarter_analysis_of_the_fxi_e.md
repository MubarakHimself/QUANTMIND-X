---
title: Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF
url: https://www.mql5.com/en/articles/20550
categories: Trading Systems, Integration, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:31:47.459621
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/20550&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062531789559342065)

MetaTrader 5 / Trading systems


### Introduction

In the last article, we looked at how various preset indicator pairs can be analyzed and sifted for what works best, when we need to trade the VGT ETF. Our focus then was coming up with two complementary indicators that present an assortment of signal patterns so that they can be put to work as soon as possible. In that article, we did not shed much light on how the pool of 5 pairs of complementary indicators were selected, to begin with. For this article, therefore, we seek to tackle this area.

The choosing of technical indicators, in any trading system, can often be approached without enough methodical discipline, which can result in the end-use of indicators being determined by preference, anecdote, or biased historical interpretations. Such s none structured, almost whimsical approach can introduce survivorship-bias, hindsight-confirmation, structural-overfitting and the overall undermining of analytical integrity. A diligent process is always preferred in order to generate reliable signals across different market regimes.

The case for having such a system is even more poignant when faced with an ETF such as the FXI. Its quarterly behavior reflects shifting volatility conditions with changing momentum structures as well as fluctuations in liquidity. Indicators that show promise in what market regime, can easily falter in another. We therefore seek to remedy this by proposing a rigorous Python-based sequence of steps that culminate in use within a [wizard assembled](https://www.mql5.com/en/articles/171) Expert Advisor.

### The FXI-ETF and its Quarterly Dynamics

This ETF, that tracks the performance of large-cap Chinese equities, often shows some recurring structural behaviours that can be attributed to macroeconomic events, policy interventions, risk-sentiment worldwide, or even liquidity cycles. These behaviors are exhibited in the form of alternating volatility regimes, momentum that is present in a non-uniform structure, as well as periodic gyrations between trending and mean-reversion. When performing analyzes over several quarters, these regime transitions do become relevant. The efficacy of one indicator in one regime, may fizzle in the next, meaning we need a multi-quarter analytical approach that allows the observer to segment FXI’s price evolution into a set of ‘discrete-windows’.

Such a segmentation would help separate differences in trend strength, and size of fluctuations. This can be a foundation for evaluating the ability to generate signals for the indicators, since each is appraised across different market regimes. In order to provide proper context, the pipeline that we consider below, splits the FXI ETF price data, at the 4-hour time frame into quarters, presenting a visual depiction of its behavior by quarters.

Before we delve into that, we present an overall view of the ETF since its 2004 October 5th inception. There has been a stock split since it was issued, which explains the dip in price in 2008, but other than that, the ETF appears to behave like a forex currency pair by hovering around the 40 dollar price point.

![i1](https://c.mql5.com/2/185/i-1.png)

In segmenting FXI price data, retrievable from MetaTrader 5 via the Python MetaTrader 5 module, presented below is a python code snippet that shows how the segmentation can be done. These segments are a new ‘data-type-format’ that is key in the pipeline that we use below.

```
# python libraries
import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize(login=XXXXXXXXX, server="XXXXXXXX", password="XXXXXXXXX"):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# set start and end dates for history data
from datetime import timedelta, datetime
# end_date = datetime.now()
end_date = datetime(2025, 12, 1, 0)
start_date = datetime(2020, 1, 1, 0)

# get rates
fxi_rates = mt5.copy_rates_range("#FXI", mt5.TIMEFRAME_H4, start_date, end_date)

df = pd.DataFrame(fxi_rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df.set_index("time", inplace=True)

# enforce column structure
df = df.rename(columns={
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close"
})

# assign quarterly labels
df["quarter"] = df.index.to_period("Q")

# preview distribution
quarter_counts = df["quarter"].value_counts().sort_index()
print(quarter_counts)
```

Running this short code gives us our quarter boundaries as indicated in the output logs below

```
quarter
2020Q2     68
2020Q3    128
2020Q4    167
2021Q1    170
2021Q2    126

….

2025Q2    127
2025Q3    130
2025Q4    104
Freq: Q-DEC, Name: count, dtype: int64
```

The partitioning we have above sets up a temporal structure, so that all our indicator scores and pattern evaluations are in a framework for easy comparison. The discrete and comparable quarterly windows.

### Preparing the Dataset in Python

In order to put up a framework for evaluating indicators, we need to have the underlying datasets prepared with as much precision as possible, as any seemingly mundane errors can have cascading effects. The FXI ETF’s 4-hour OHLC structure that we are using is our foundation for all subsequent computations. Its integrity, consistency, and temporal alignment need to be verified and also enforced before we can apply any indicator or pattern logic. Preprocessing ensures that the segmentation remains accurate, that the forward return computations are not corrupted by missing or out of order data, and that each indicator gets a clean or relatively uniform input domain.

The dataset preparation stage is made up of four operations:

- Validating the OHLC field consistency
- Re-synchronizing the timestamps in order to maintain a standard 4-hour spacing
- Addressing missing observations or discontinuities; and
- Attaching quarterly labels for analytical segmentation

These operations, all together, bring a stable temporal structure within which our indicator ‘scoring card’ can proceed with minimal contaminations from irregularities in the raw data. We try to realize these formalities in the following python script. It outlines a standard preprocessing workflow for data imported by the MetraTrader 5 module.

```
# python libraries
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from datetime import datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar

...

# ------------------------------------------------------------
# Build base DataFrame
# ------------------------------------------------------------
df = pd.DataFrame(fxi_rates)

# MT5 times are in seconds since epoch
df["time"] = pd.to_datetime(df["time"], unit="s")
df.set_index("time", inplace=True)
df = df.sort_index()

# Confirm essential OHLC columns exist
required_cols = {"open", "high", "low", "close"}
if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    raise ValueError(f"Dataset missing required columns: {missing}")

# ------------------------------------------------------------
# Check interval consistency and resample to uniform 4h grid
# ------------------------------------------------------------
time_diff = df.index.to_series().diff().dropna()
irregularities = time_diff[time_diff != pd.Timedelta(hours=4)]
print("Irregular intervals detected:\n", irregularities.head())

# Use lowercase "h" to avoid FutureWarning
df_resampled = df.resample("4h").ffill()

# ------------------------------------------------------------
# Attach quarterly segmentation
# ------------------------------------------------------------
df_resampled["quarter"] = df_resampled.index.to_period("Q")

# ------------------------------------------------------------
# Market-hours mask + U.S. holidays detection
# ------------------------------------------------------------

# Assume MT5 timestamps are in UTC. If your server is NOT UTC,
# adjust this localize step accordingly.
idx_utc = df_resampled.index.tz_localize("UTC")

# Convert to New York time (U.S. Eastern, with DST handled)
idx_ny = idx_utc.tz_convert("America/New_York")

# Weekday mask (Mon–Fri)
is_us_weekday = idx_ny.weekday < 5  # 0=Mon, 4=Fri

# Session time mask (09:30–16:00 New York time)
ny_times = idx_ny.time
session_start = time(9, 30)
session_end = time(16, 0)
is_session_time = np.array(
    [(t >= session_start) and (t <= session_end) for t in ny_times],
    dtype=bool
)

# U.S. holiday calendar (federal holidays; close approximation)
cal = USFederalHolidayCalendar()
holidays = cal.holidays(
    start=idx_ny.min().date(),
    end=idx_ny.max().date()
)

# Normalize times to dates and check membership in holiday list
is_holiday = pd.Series(idx_ny.normalize()).isin(holidays).to_numpy()

# Final market mask: weekday, in session, not holiday
market_mask = is_us_weekday & is_session_time & (~is_holiday)

# Apply mask
df_market = df_resampled[market_mask].copy()

print("Resampled full data shape:", df_resampled.shape)
print("Market-hours non-holiday data shape:", df_market.shape)
print(df_market.head())
print(df_market.tail())
```

The openness of Python has allowed the development of a multitude of modules and libraries. We therefore, in our code above, also include, A market-hours mask for regular U.S. session (09:30–16:00 New York time); and U.S. holiday detection using USFederalHolidayCalendar. The above snippet gives us the following outputs, which are shortened here to emphasize the detected irregular intervals.

```
Irregular intervals detected:
 time
2020-05-14 12:00:00   0 days 20:00:00
2020-05-15 12:00:00   0 days 20:00:00
2020-05-18 12:00:00   2 days 20:00:00
2020-05-19 12:00:00   0 days 20:00:00
2020-05-20 12:00:00   0 days 20:00:00
Name: time, dtype: timedelta64[ns]
Resampled full data shape: (12153, 8)
Market-hours non-holiday data shape: (2896, 8)
```

An integrity checklist, based on our run above, does therefore look as follows.

| Item | Status | Details |
| --- | --- | --- |
| MetaTrader 5 Data Retrieval | OK | FXI H4 Rates successfully returned |
| Datetime Conversion | OK | Unix Timestamps converted to pandas datetime |
| Index Sorting | OK | Data sorted chronologically |
| Required OHLC Columns Present | OK | Open, High, Low, Close all present |
| Irregular interval Detection | Warning | 20 hour gaps instead of 24, detected on overnight, weekends and holidays |
| Resampling applied | OK | 4h grid enforced and output shape = 12153 |
| Timezone Localization | OK | UTC America/ NY conversion applied |
| Weekday mask applied | OK | Only Monday to Friday retained |
| Session time mask applied | OK | Kept bars corresponding to 9:30 to 16:00 session |
| Holiday detection | OK | US Federal Holidays excluded |
| Final market hours | OK | Resulting data shape = 2896 rows |
| Quarter Segmentation | OK | Quarter column assigned |

With our input data frame ‘sanitized’ we are now ready to proceed to the subsequent steps.

Indicator Recommendation Pipeline

A framework for formally choosing an indicator should ideally be built with some modularity, reproducibility, and a bit of analytical transparency as the main governing tenets. The pipeline introduced here strives to adhere to these requirements. We abstract indicator computation, deriving patterns, evaluation of forward-returns, scoring into discrete parts that are re-usable with different asset classes besides the tested FXI. Our modular approach, on paper, ensures that every component can be independently upgraded/debugged without compromising the integrity of the overall evaluation process.

[![i12](https://c.mql5.com/2/185/i12.png)](https://c.mql5.com/2/185/i12.png "https://c.mql5.com/2/185/i12.png")

The diagram above represents our approach at the concept level. The pipeline is built around three main abstractions. The first of these is the specification of indicators, where we embody the mathematics involved in computing indicator values as well as the deterministic rules that set bullish or bearish or flat signal patterns. The second abstraction is the Quarterly evaluation ‘engine’. Here, we apply these specifications to a given quarterly segment of OHLC data, then compute forward-looking returns over a defined time horizon to generate some form of performance metrics for each pattern. The third abstraction is the ranking and recommendation layer. At this point, we aggregate results from all indicators, then sort them by empirical effectiveness in order to come up with a recommendation set.

The separation of these different stages is important and critical. By isolating compute logic, from evaluation, the analyst avoids the common pitfall of blurring pattern inference with performance measurement, a caveat that usually leads to biases. The structure is in addition, very scalable where new indicators are addable through the specification layer, so that more pattern states and return horizons get added without restructuring the pipeline in major ways. The pipeline is therefore a long-term-analytical-asset, instead of a disposable script.

We develop this in Python, therefore, as shown below. Included is a ‘main-clause’ that demonstrates how this pipeline section could be used in getting performance metrics for an indicator, which in this particular instance is a simple moving average.

```
# python libraries
import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import pandas as pd
import numpy as np

from datetime import datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---------------------------------------------------------------
# 1) Helper functions you referenced but didn't define
# ---------------------------------------------------------------

def compute_forward_returns(df: pd.DataFrame, horizons: Tuple[int, ...]) -> pd.DataFrame:
    """
    Compute forward returns over the given horizons.
    Assumes df has a 'close' column.
    Returns columns like 'fwd_1', 'fwd_3', etc.
    """
    out = {}
    close = df["close"]
    for h in horizons:
        # forward return: (close[t+h] / close[t]) - 1
        out[f"fwd_{h}"] = close.shift(-h) / close - 1.0
    return pd.DataFrame(out, index=df.index)

def score_indicator(pattern_states: pd.DataFrame,
                    forward_returns: pd.DataFrame) -> Dict[str, float]:
    """
    Very simple scoring function:
    For each pattern (bullish/bearish/flat) and each horizon,
    compute the mean forward return where that pattern is True.
    """
    metrics = {}
    for pattern_col in pattern_states.columns:   # bullish, bearish, flat
        mask = pattern_states[pattern_col].astype(bool)

        for fwd_col in forward_returns.columns:  # fwd_1, fwd_3, ...
            key = f"{pattern_col}_{fwd_col}_mean"
            if mask.any():
                metrics[key] = forward_returns.loc[mask, fwd_col].mean()
            else:
                metrics[key] = np.nan
    return metrics

# ---------------------------------------------------------------
# 2) Indicator Specification: Compute function + Pattern rules
# ---------------------------------------------------------------
@dataclass
class IndicatorSpec:
    name: str
    compute: Callable[[pd.DataFrame], pd.Series]
    patterns: Callable[[pd.DataFrame, pd.Series], pd.DataFrame]
    # patterns returns a DataFrame with columns: bullish, bearish, flat

# ---------------------------------------------------------------
# 3) Quarterly Evaluation Engine: Apply indicator to one quarter
# ---------------------------------------------------------------
def evaluate_indicator(df_q: pd.DataFrame,
                       spec: IndicatorSpec,
                       horizons=(1, 3, 6)):
    indicator_series = spec.compute(df_q)
    pattern_states = spec.patterns(df_q, indicator_series)
    forward_returns = compute_forward_returns(df_q, horizons)

    metrics = score_indicator(pattern_states, forward_returns)
    metrics["indicator"] = spec.name
    return metrics

# ---------------------------------------------------------------
# 4) Pipeline Coordinator: Apply all indicators across all quarters
# ---------------------------------------------------------------
class IndicatorPipeline:
    def __init__(self, specs: Dict[str, IndicatorSpec], horizons=(1, 3, 6)):
        self.specs = specs
        self.horizons = horizons

    def run(self, df: pd.DataFrame):
        df = df.copy()
        # df.index must be a DatetimeIndex
        df["quarter"] = df.index.to_period("Q")

        results = []
        for q, df_q in df.groupby("quarter"):
            for spec in self.specs.values():
                res = evaluate_indicator(df_q, spec, self.horizons)
                res["quarter"] = str(q)
                results.append(res)

        return pd.DataFrame(results)

# ---------------------------------------------------------------
# 5) Example Indicator: SMA-20 Trend (bullish / bearish / flat)
# ---------------------------------------------------------------

def compute_sma20(df: pd.DataFrame) -> pd.Series:
    """
    20-period simple moving average of 'close'.
    """
    return df["close"].rolling(window=20, min_periods=1).mean()

def sma20_patterns(df: pd.DataFrame, sma: pd.Series) -> pd.DataFrame:
    """
    Define pattern states based on where price sits vs SMA.
    - bullish: close > sma
    - bearish: close < sma
    - flat:    close == sma (or within tiny epsilon)
    """
    close = df["close"]
    eps = 1e-8
    bullish = close > sma * (1 + eps)
    bearish = close < sma * (1 - eps)
    flat = ~(bullish | bearish)

    return pd.DataFrame(
        {
            "bullish": bullish,
            "bearish": bearish,
            "flat": flat,
        },
        index=df.index,
    )

sma20_spec = IndicatorSpec(
    name="sma20_trend",
    compute=compute_sma20,
    patterns=sma20_patterns,
)

# ---------------------------------------------------------------
# 6) Demo: Make fake OHLC data and run the pipeline
# ---------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    # Initialize MetaTrader 5
    # ------------------------------------------------------------
    if not mt5.initialize(login=XXXXXXXX,
                          server="XXXXXXX",
                          password="XXXXX"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # ------------------------------------------------------------
    # Set start and end dates for history data
    # ------------------------------------------------------------
    end_date = datetime(2025, 12, 1, 0)
    start_date = datetime(2020, 1, 1, 0)

    # Get H4 rates for FXI (CFD symbol "#FXI")
    fxi_rates = mt5.copy_rates_range("#FXI", mt5.TIMEFRAME_H4, start_date, end_date)

    if fxi_rates is None or len(fxi_rates) == 0:
        print("No data returned from MT5.")
        mt5.shutdown()
        quit()

    # ------------------------------------------------------------
    # Build base DataFrame
    # ------------------------------------------------------------
    df = pd.DataFrame(fxi_rates)

    # MT5 times are in seconds since epoch
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.sort_index()

    # Confirm essential OHLC columns exist
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Build pipeline with one or more indicators
    specs = {
        "sma20": sma20_spec,
        # you can add more IndicatorSpec entries here
    }

    pipeline = IndicatorPipeline(specs=specs, horizons=(1, 3, 5))

    results = pipeline.run(df)

    # Show the per-quarter stats
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(results)
```

This is not a complete implementation, but a concept scaffolding on which our next sections can build. The functional components of this pipeline can be summarized as follows, as indicated in this table.

| Component Name | Role in Pipeline | Output Produced |
| --- | --- | --- |
| Indicator Specification | Defines how an indicator is computed and how patterns are derived | Pattern-state data frame |
| Quarterly Evaluation Engine | Scores indicator patterns using forward return metrics | Hit rates, mean returns, indicator scores |
| Pipeline Coordinator | Executes all indicators aross all the quarters | Unified quarterly results dataset |
| Ranking Engine | Sorrts indicators by performance per quarter | Recommended indicator list for each quarter |

With this done, we can now move on to the next step where we formally define the indicators to be used, present some pattern logic and set how to compute the evaluation metrics, all based on the FXI multi-quarter dataset.

### Indicator Specification Layer

In evaluating indicators, we need an explicit mechanism that defines the internal logic of each indicator. This logic would comprise two inseparable components. The mathematical computations that give us the indicator numeric states, and the deterministic pattern rules for classifying each pattern as either bullish or bearish or flat. The accuracy and consistency of this layer determines the quality of all downstream analyses, given that these determined states feed directly into forward-return evaluations. This ultimately influences quarterly indicator rankings.

In order to have a uniform way of analyzing across a diverse set of indicators - whether trending, oscillatory, or structural, -  the specification layer is built as a formalized object that contains both computation and pattern deriving functions. By encapsulating like this, we prevent cross contamination between indicator logic and evaluation procedures, ensuring that the indicators remain ‘mathematically-autonomous’. This also enables expansion, given that extra indicators may be added to the pipeline through defining a single spec object without making changes to the rest of the system.

This specification layer therefore is a foundational ‘contract’, binding compute with interpretation logic. Every indicator is in charge of producing its numeric series and classifying its pattern states.  The pipeline, for its part, is charged with evaluating these states. We implement this in Python as follows.

```
# python libraries
import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import pandas as pd
import numpy as np

from datetime import datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---------------------------------------------------------------
# Your IndicatorSpec + RSI Implementation
# ---------------------------------------------------------------
@dataclass
class IndicatorSpec:
    name: str
    compute: Callable[[pd.DataFrame], pd.Series]
    patterns: Callable[[pd.DataFrame, pd.Series], pd.DataFrame]

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def rsi_patterns(df: pd.DataFrame, rsi: pd.Series) -> pd.DataFrame:
    bullish = (rsi < 30)
    bearish = (rsi > 70)
    flat = ~(bullish | bearish)
    return pd.DataFrame({
        "bullish": bullish.astype(int),
        "bearish": bearish.astype(int),
        "flat": flat.astype(int),
    })

rsi_14_spec = IndicatorSpec(
    name="RSI_14",
    compute=lambda df: compute_rsi(df, 14),
    patterns=rsi_patterns
)

# ---------------------------------------------------------------
# Helper Functions for Returns & Evaluation
# ---------------------------------------------------------------
def compute_forward_returns(df: pd.DataFrame, horizons=(1, 3, 6)):
    out = {}
    close = df["close"]
    for h in horizons:
        out[f"fwd_{h}"] = close.shift(-h) / close - 1
    return pd.DataFrame(out)

def score_indicator(patterns: pd.DataFrame, forward_returns: pd.DataFrame) -> Dict[str, float]:
    metrics = {}
    for p in patterns.columns:
        mask = patterns[p] == 1
        for fwd in forward_returns.columns:
            key = f"{p}_{fwd}_mean"
            if mask.any():
                metrics[key] = forward_returns.loc[mask, fwd].mean()
            else:
                metrics[key] = 0# np.nan
    return metrics

# ---------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------
def evaluate_indicator(df_q: pd.DataFrame, spec: IndicatorSpec, horizons=(1, 3, 6)):
    indicator_series = spec.compute(df_q)
    pattern_states = spec.patterns(df_q, indicator_series)
    forward_returns = compute_forward_returns(df_q, horizons)

    metrics = score_indicator(pattern_states, forward_returns)
    metrics["indicator"] = spec.name
    return metrics

# ---------------------------------------------------------------
# Pipeline Coordinator
# ---------------------------------------------------------------
class IndicatorPipeline:
    def __init__(self, specs: Dict[str, IndicatorSpec], horizons=(1, 3, 6)):
        self.specs = specs
        self.horizons = horizons

    def run(self, df: pd.DataFrame):
        df = df.copy()
        df["quarter"] = df.index.to_period("Q")

        results = []
        for q, df_q in df.groupby("quarter"):
            for spec in self.specs.values():
                res = evaluate_indicator(df_q, spec, self.horizons)
                res["quarter"] = str(q)
                results.append(res)

        return pd.DataFrame(results)

# ---------------------------------------------------------------
# DEMO: Run the pipeline with synthetic data
# ---------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    # Initialize MetaTrader 5
    # ------------------------------------------------------------
    if not mt5.initialize(login=XXXXX,
                          server="XXXXXXXXX",
                          password="XXXXXXX"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # ------------------------------------------------------------
    # Set start and end dates for history data
    # ------------------------------------------------------------
    end_date = datetime(2025, 12, 1, 0)
    start_date = datetime(2020, 1, 1, 0)

    # Get H4 rates for FXI (CFD symbol "#FXI")
    fxi_rates = mt5.copy_rates_range("#FXI", mt5.TIMEFRAME_H4, start_date, end_date)

    if fxi_rates is None or len(fxi_rates) == 0:
        print("No data returned from MT5.")
        mt5.shutdown()
        quit()

    # ------------------------------------------------------------
    # Build base DataFrame
    # ------------------------------------------------------------
    df = pd.DataFrame(fxi_rates)

    # MT5 times are in seconds since epoch
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.sort_index()

    # Confirm essential OHLC columns exist
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    specs = {"RSI14": rsi_14_spec}
    pipeline = IndicatorPipeline(specs=specs, horizons=(1, 3, 5))

    result = pipeline.run(df)
    print(result)
```

Our illustration above that uses the RSI works as a template for other indicators, to allow a consistent, modular and formally defined framework that brings together the broader evaluation pipeline. Below is a tabulation of a summary of the specification requirements required by our pipeline.

| Specification Component | Description | Required Output |
| --- | --- | --- |
| name | unique identifier for indicator | string |
| compute() | function generating the indicator series from OHLC data | pd.series aligned with dataset index |
| patterns() | function deriving bullish, bearish and states for neutrality | pd.Dataframe with 3 binary columns |
| Modularity | Isolated and reusable indicator logic | Consistent across all indicators |
| Extensibility | new indicators require only adding the new object | No pipeline modifications required |

### Pattern Logic for the Indicators

A framework that evaluates indicators needs to set the pattern states in a format that is deterministic as well as consistent across the variety of indicators under consideration. The logic of patterns often acts as a bridge between raw values and actionable signals, which the pipeline evaluates through a forward return analysis. Without defining the patterns in a structured manner the indicator would give outputs that still have meaning numerically, but are operationally ambiguous. This would make evaluation difficult.

Pattern construction has to be indicator-specific. Momentum oscillators usually depend on threshold-linked interpretations; trend indicators on the other hand do necessitate a directional slope assessment or some form of cross comparison; meanwhile volatility based indicators usually depend on bands expansions/contractions. In some instances, particularly in situations where indicators are not originally designed to give direction signals, price action needs to be integrated in order to synthsize a bullish or bearish interpretation given the structural conditions. The aim here would be to produce a cohesive representation of states across all indicators for easy interpretation of the quarterly scoring engine.

For every indicator, the pattern logic should meet three important tenets.

- Explicit Determinism, in order to have no ambiguity in the classification
- Temporal Coherence, meaning the signals need to be in the exact chronological order of the underlying price data
- Compatibility, in order to ensure that every pattern outputs boolean or binary states in a format that indicates bullishness, bearishness or a neutral outlook.

This resulting pattern-state matrix therefore becomes the foundation on which the pipeline evaluates the quality of predictions. We use the following examples covering three broad indicator categories. A momentum oscillator, the RSI; a trend indicator, the MACD; and a volatility indicator, the Bollinger Bands; which we merge with some price action bias. This code is meant to illustrate how varying indicator types can be merged into a common bullish/bearish/flat schema, something we need in order to do our evaluations.

```
# python libraries
import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import pandas as pd
import numpy as np

from datetime import datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar

# -------------------------------------------------------------------------
# Import your full indicator library (CITED)
# -------------------------------------------------------------------------
from IndicatorsAll import (
    RSI, MACD, Bollinger_Bands,
    ADX_Wilder, FRAMA, Parabolic_SAR,
    VIDYA, TRIX, Alligator, Gator_Oscillator,
    Awesome_Oscillator, Bill_Williams_MFI,
    ATR, StdDev, CCI, Accelerator_Oscillator,
    Bill_Williams_Fractals, Williams_R, Ichimoku,
    Envelopes
)

# -------------------------------------------------------------------------
# 1. Pattern Logic Layer (Your Provided Logic)
# -------------------------------------------------------------------------

def rsi_patterns(df: pd.DataFrame, rsi: pd.Series) -> pd.DataFrame:
    bullish = (rsi < 30).astype(int)
    bearish = (rsi > 70).astype(int)
    flat = ((rsi >= 30) & (rsi <= 70)).astype(int)
    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

def macd_patterns(df: pd.DataFrame, macd_hist: pd.Series) -> pd.DataFrame:
    bullish = (macd_hist > 0).astype(int)
    bearish = (macd_hist < 0).astype(int)
    flat = (macd_hist == 0).astype(int)
    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

def bollinger_patterns(df: pd.DataFrame, bb_width: pd.Series) -> pd.DataFrame:
    price = df["close"]
    slope = price.diff()

    contraction = (bb_width < bb_width.rolling(20).mean()).astype(int)
    bullish = ((contraction == 0) & (slope > 0)).astype(int)
    bearish = ((contraction == 0) & (slope < 0)).astype(int)
    flat = (contraction == 1).astype(int)

    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

# -------------------------------------------------------------------------
# 2. Quarterly Evaluation Engine
# -------------------------------------------------------------------------

def compute_forward_returns(df: pd.DataFrame, horizon: int = 3):
    return df["close"].shift(-horizon) / df["close"] - 1

def evaluate_indicator(patterns: pd.DataFrame, forward: pd.Series) -> dict:
    results = {}
    for state in ["bullish", "bearish", "flat"]:
        mask = patterns[state] == 1
        if mask.sum() == 0:
            results[state] = {"count": 0, "avg_return": 0.0}
        else:
            results[state] = {
                "count": int(mask.sum()),
                "avg_return": forward[mask].mean()
            }
    return results

# -------------------------------------------------------------------------
# 3. Ranking Layer
# -------------------------------------------------------------------------

def score_indicator(results: dict) -> float:
    bull = results["bullish"]["avg_return"]
    bear = -results["bearish"]["avg_return"]
    flat = -0.2 * abs(results["flat"]["avg_return"])
    return bull + bear + flat

# -------------------------------------------------------------------------
# 4. Main Pipeline Demonstration
# -------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------
    # Initialize MetaTrader 5
    # ------------------------------------------------------------
    if not mt5.initialize(login=XXXXX,
                          server="XXXXXXXX",
                          password="XXX"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # ------------------------------------------------------------
    # Set start and end dates for history data
    # ------------------------------------------------------------
    end_date = datetime(2025, 12, 1, 0)
    start_date = datetime(2020, 1, 1, 0)

    # Get H4 rates for FXI (CFD symbol "#FXI")
    fxi_rates = mt5.copy_rates_range("#FXI", mt5.TIMEFRAME_H4, start_date, end_date)

    if fxi_rates is None or len(fxi_rates) == 0:
        print("No data returned from MT5.")
        mt5.shutdown()
        quit()

    # ------------------------------------------------------------
    # Build base DataFrame
    # ------------------------------------------------------------
    df = pd.DataFrame(fxi_rates)

    # MT5 times are in seconds since epoch
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.sort_index()

    # Confirm essential OHLC columns exist
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    # ---------------------------------------------------
    # Compute indicators USING YOUR MODULE (cited)
    # ---------------------------------------------------
    df_rsi = RSI(df)
    df_macd = MACD(df)
    df_bb = Bollinger_Bands(df)

    # Create final working copy
    df_all = df.copy()
    df_all["RSI"] = df_rsi["RSI"]
    df_all["MACD_hist"] = df_macd["MACD_hist"]
    df_all["BB_width"] = df_bb["BB_width"]

    # ---------------------------------------------------
    # Pattern Logic from imported indicators
    # ---------------------------------------------------
    pat_rsi = rsi_patterns(df_all, df_all["RSI"])
    pat_macd = macd_patterns(df_all, df_all["MACD_hist"])
    pat_bb = bollinger_patterns(df_all, df_all["BB_width"])

    # ---------------------------------------------------
    # Evaluation Layer
    # ---------------------------------------------------
    forward3 = compute_forward_returns(df_all, horizon=3)

    eval_rsi = evaluate_indicator(pat_rsi, forward3)
    eval_macd = evaluate_indicator(pat_macd, forward3)
    eval_bb = evaluate_indicator(pat_bb, forward3)

    # ---------------------------------------------------
    # Ranking Layer
    # ---------------------------------------------------
    scores = {
        "RSI": score_indicator(eval_rsi),
        "MACD_Hist": score_indicator(eval_macd),
        "BB_Width": score_indicator(eval_bb),
    }

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n========== Indicator Ranking ==========\n")
    for name, score in ranking:
        print(f"{name:12s} → Score: {score:.4f}")

    print("\n========== Detailed Evaluations ==========\n")
    print("RSI:", eval_rsi)
    print("MACD:", eval_macd)
    print("BB Width:", eval_bb)

# -------------------------------------------------------------------------
# Run Script
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
```

### Forward Returns and Evaluation Horizons

Computing the forward returns is the main backbone of this indicator evaluation framework. This pipeline cannot determine if long or short pattern states carry any predictive value unless every signal is directly compared to prior price movements. Forward returns are able to give us this link by giving us by how much the price evolves after a pattern has been triggered. In addition, we also capture different forward-looking horizons so that the system captures both short term and medium term effects, the result of this being that an indicator is not unfairly advantaged or penalized by just one temporal perspective.

![i6](https://c.mql5.com/2/185/i-6.png)

Formally, a forward return at horizon h measures the change in percent between the closing price at a time t and the closing price at a time t + h. These metrics should always be worked out without any look-ahead bias by only relying on future observations relative to the signal. Once they are determined, the forward return matrix will become the reference dataset against which all signal patterns for long/short/neutral are assessed.

The evaluation horizons we have selected to use, in this context, are 1-bar, 3-bar, and 6-bar intervals. They are meant to give us a balanced framework for immediate confirmation, short range continuation, as well as medium range persistence. These horizons ensure that if we have indicators with different behavior frequencies, we can still get a basis for comparing them. To illustrate this, momentum oscillators often perform well at shorter horizons, while trend indicators tend to work well over longer horizons. Volatility indicators on the other hand are bound to be time horizon neutral since they focus on market expansions/ contractions. The forward-return numbers are therefore an important linchpin that allows us to have objective scoring. To compute these forward returns in Python, we use the following code

```
# python libraries
import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import pandas as pd
import numpy as np

from datetime import datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar

# -------------------------------------------------------------------
# Forward return calculator (your function)
# -------------------------------------------------------------------
def compute_forward_returns(df: pd.DataFrame, horizons=(1, 3, 6)) -> pd.DataFrame:
    close = df["close"]
    forward_data = {}

    for h in horizons:
        forward_data[f"fwd_ret_{h}"] = (close.shift(-h) / close) - 1.0

    return pd.DataFrame(forward_data, index=df.index)

# -------------------------------------------------------------------
# Simple pattern definition for demonstration
# (e.g., bullish when price is rising, bearish when falling)
# -------------------------------------------------------------------
def simple_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    slope = df["close"].diff()

    bullish = (slope > 0).astype(int)
    bearish = (slope < 0).astype(int)
    flat = (slope == 0).astype(int)

    return pd.DataFrame({
        "bullish": bullish,
        "bearish": bearish,
        "flat": flat
    })

# -------------------------------------------------------------------
# Evaluation: Compare pattern states with forward returns
# -------------------------------------------------------------------
def evaluate_patterns(patterns: pd.DataFrame, forward: pd.DataFrame) -> dict:
    results = {}

    for state in ["bullish", "bearish", "flat"]:
        mask = patterns[state] == 1

        state_results = {}
        for col in forward.columns:
            if mask.sum() == 0:
                state_results[col] = np.nan
            else:
                state_results[col] = forward.loc[mask, col].mean()

        results[state] = state_results

    return results

# -------------------------------------------------------------------
# MAIN EXAMPLE: Demonstrate usage of forward-return calculator
# -------------------------------------------------------------------
def main():
    # ------------------------------------------------------------
    # Initialize MetaTrader 5
    # ------------------------------------------------------------
    if not mt5.initialize(login=XXXXX,
                          server="XXX",
                          password="XXXXXXXX"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # ------------------------------------------------------------
    # Set start and end dates for history data
    # ------------------------------------------------------------
    end_date = datetime(2025, 12, 1, 0)
    start_date = datetime(2020, 1, 1, 0)

    # Get H4 rates for FXI (CFD symbol "#FXI")
    fxi_rates = mt5.copy_rates_range("#FXI", mt5.TIMEFRAME_H4, start_date, end_date)

    if fxi_rates is None or len(fxi_rates) == 0:
        print("No data returned from MT5.")
        mt5.shutdown()
        quit()

    # ------------------------------------------------------------
    # Build base DataFrame
    # ------------------------------------------------------------
    df = pd.DataFrame(fxi_rates)

    # MT5 times are in seconds since epoch
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.sort_index()

    # Confirm essential OHLC columns exist
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    # -----------------------------------------
    # Compute forward returns (horizons 1,3,6)
    # -----------------------------------------
    fwd = compute_forward_returns(df, horizons=(1, 3, 6))
    print("\n=== Forward Returns (Example) ===\n")
    print(fwd.head())

    # -----------------------------------------
    # Compute simple pattern states
    # (your real pipeline uses indicator patterns)
    # -----------------------------------------
    patterns = simple_price_patterns(df)

    # -----------------------------------------
    # Evaluate patterns against forward returns
    # -----------------------------------------
    evaluation = evaluate_patterns(patterns, fwd)

    print("\n=== Pattern–Forward Return Evaluation ===\n")
    for state, metrics in evaluation.items():
        print(f"{state.upper()}:")
        for horizon, value in metrics.items():
            print(f"  {horizon}: {value:.6f}")
        print()

# -------------------------------------------------------------------
# Run example
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
```

If we run the demonstration script above, we will get the following outputs. They are given below as a table.

| Bullish |  |
| --- | --- |
| fwd-ret-1 | -0.000209 |
| fwd-ret-3 | -0.000181 |
| fwd-ret-6 | -0.000068 |

| Bearish |  |
| --- | --- |
| fwd-ret-1 | 0.000325 |
| fwd-ret-3 | 0.000640 |
| fwd-ret-6 | 0.000998 |

| Neutral |  |
| --- | --- |
| fwd-ret-1 | 0.000896 |
| fwd-ret-3 | 0.001127 |
| fwd-ret-6 | 0.001660 |

This output, which we tabulate above, is meant to show the basic structure that the scoring engine will use to size-up predictive relevance of each pattern state. For example, with a Bullish pattern, we evaluate it by measuring the proportion of forward returns that are positive across the 3 time horizons. Bearish patterns are tested against negative forward returns; while flat patterns are evaluated by considering the absolute magnitude.

### Scoring the Indicators

Once we have the state patterns and their respective forward returns, we then need to build a formal scoring method that quantifies each indicator’s ability to forecast within a set quarter. This mechanism would transform qualitative ideas such as ‘this indicator seems to be reliable’ into more explicit numeric straits, thus allowing us to make objective comparisons over several indicators.

The task of scoring is defined as follows. Firstly, for every indicator, and for each quarter segment the framework has to evaluate how well its bullish, bearish, and flat pattern states align with subsequent price behavior. We do this evaluation by comparing the pattern-state matrix, which is a set of binary signals, with the forward-return matrix, which is a set of realized returns over several time horizons. What we get from this of a collection of performance statistics summarizing every indicator’s accuracy in forecasting direction, average profitability, and how often signals are generated or signal frequency.

We arrive at a composite score, which we get from combining these three metrics in a controlled manner. For instance, our pipeline could give more weight to the hit-rate/ accuracy-score while concurrently allowing for the amount of average return to be factored as a secondary adjustment by having a smaller weight. This can ensure that the indicators are rewarded for both consistency and profitability. This can happen without encouraging overfitting. If we stick to our 3 indicator choice above, and continue to use our already coded module of indicator patterns, labelled [IndicatorsAll.py](https://www.mql5.com/go?link=http://indicatorsall.py/ "http://indicatorsall.py/"), we could implement the scoring as follows.

```
# python libraries
import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import pandas as pd
import numpy as np

from datetime import datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---------------------------------------------------------
# Import indicators from your module
# ---------------------------------------------------------
from IndicatorsAll import RSI, MACD, Bollinger_Bands  # :contentReference[oaicite:1]{index=1}

# ---------------------------------------------------------
# Indicator scoring function (as given)
# ---------------------------------------------------------
def score_indicator(patterns: pd.DataFrame,
                    fwd_rets: pd.DataFrame,
                    min_signals: int = 10) -> dict:
    """
    Compute predictive power metrics for one indicator within a single quarter.
    patterns: DataFrame with columns ['bullish', 'bearish', 'flat'] (0/1 or bool)
    fwd_rets: DataFrame with columns such as ['fwd_ret_1', 'fwd_ret_3', ...]
    """
    results = {}

    for state in ["bullish", "bearish"]:
        mask = patterns[state] == 1
        num_signals = int(mask.sum())
        results[f"{state}_count"] = num_signals

        if num_signals < min_signals:
            results[f"{state}_hit_rate"] = np.nan
            results[f"{state}_mean_ret"] = np.nan
            continue

        expected_sign = 1 if state == "bullish" else -1
        state_hits = []
        state_mean_rets = []

        for col in fwd_rets.columns:
            r = fwd_rets.loc[mask, col].dropna()
            if r.empty:
                continue

            hits = (np.sign(r) == expected_sign).astype(float).mean()
            state_hits.append(hits)
            state_mean_rets.append(r.mean())

        if state_hits:
            results[f"{state}_hit_rate"] = float(np.mean(state_hits))
            results[f"{state}_mean_ret"] = float(np.mean(state_mean_rets))
        else:
            results[f"{state}_hit_rate"] = np.nan
            results[f"{state}_mean_ret"] = np.nan

    bullish_hr = results.get("bullish_hit_rate", np.nan)
    bearish_hr = results.get("bearish_hit_rate", np.nan)
    bullish_ret = results.get("bullish_mean_ret", 0.0)
    bearish_ret = results.get("bearish_mean_ret", 0.0)

    valid_hit_rates = [x for x in [bullish_hr, bearish_hr] if not np.isnan(x)]
    hit_component = float(np.mean(valid_hit_rates)) if valid_hit_rates else 0.0

    magnitude_component = float((abs(bullish_ret) + abs(bearish_ret)) / 2.0)

    final_score = hit_component + 0.5 * magnitude_component
    results["score"] = final_score

    return results

# ---------------------------------------------------------
# Forward returns from CLOSE ONLY (no indicator info)
# ---------------------------------------------------------
def compute_forward_returns(df: pd.DataFrame,
                            horizons=(1, 3, 6)) -> pd.DataFrame:
    close = df["close"]
    forward_data = {}
    for h in horizons:
        forward_data[f"fwd_ret_{h}"] = (close.shift(-h) / close) - 1.0
    return pd.DataFrame(forward_data, index=df.index)

# ---------------------------------------------------------
# Pattern logic USING REAL INDICATOR VALUES
# ---------------------------------------------------------
def rsi_patterns(df: pd.DataFrame, rsi: pd.Series) -> pd.DataFrame:
    bullish = (rsi < 30).astype(int)
    bearish = (rsi > 70).astype(int)
    flat = ((rsi >= 30) & (rsi <= 70)).astype(int)
    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

def macd_patterns(df: pd.DataFrame, macd_hist: pd.Series) -> pd.DataFrame:
    bullish = (macd_hist > 0).astype(int)
    bearish = (macd_hist < 0).astype(int)
    flat = (macd_hist == 0).astype(int)
    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

def bollinger_patterns(df: pd.DataFrame, bb_width: pd.Series) -> pd.DataFrame:
    price = df["close"]
    price_slope = price.diff()
    contraction = (bb_width < bb_width.rolling(20).mean()).astype(int)

    bullish = ((contraction == 0) & (price_slope > 0)).astype(int)
    bearish = ((contraction == 0) & (price_slope < 0)).astype(int)
    flat = (contraction == 1).astype(int)

    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

# ---------------------------------------------------------
# MAIN – Full example with RSI, MACD, Bollinger Bands
# ---------------------------------------------------------
def main():
    # ------------------------------------------------------------
    # Initialize MetaTrader 5
    # ------------------------------------------------------------
    if not mt5.initialize(login=XXXXXXXX,
                          server="XXXXX",
                          password="XXX"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # ------------------------------------------------------------
    # Set start and end dates for history data
    # ------------------------------------------------------------
    end_date = datetime(2025, 12, 1, 0)
    start_date = datetime(2020, 1, 1, 0)

    # Get H4 rates for FXI (CFD symbol "#FXI")
    fxi_rates = mt5.copy_rates_range("#FXI", mt5.TIMEFRAME_H4, start_date, end_date)

    if fxi_rates is None or len(fxi_rates) == 0:
        print("No data returned from MT5.")
        mt5.shutdown()
        quit()

    # ------------------------------------------------------------
    # Build base DataFrame
    # ------------------------------------------------------------
    df = pd.DataFrame(fxi_rates)

    # MT5 times are in seconds since epoch
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.sort_index()

    # Confirm essential OHLC columns exist
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    # ------------------------------------------------------------
    # Check interval consistency and resample to uniform 4h grid
    # ------------------------------------------------------------
    time_diff = df.index.to_series().diff().dropna()
    irregularities = time_diff[time_diff != pd.Timedelta(hours=4)]
    print("Irregular intervals detected:\n", irregularities.head())

    # Use lowercase "h" to avoid FutureWarning
    df_resampled = df.resample("4h").ffill()

    # ------------------------------------------------------------
    # Attach quarterly segmentation
    # ------------------------------------------------------------
    df_resampled["quarter"] = df_resampled.index.to_period("Q")

    # ------------------------------------------------------------
    # Market-hours mask + U.S. holidays detection
    # ------------------------------------------------------------

    # Assume MT5 timestamps are in UTC. If your server is NOT UTC,
    # adjust this localize step accordingly.
    idx_utc = df_resampled.index.tz_localize("UTC")

    # Convert to New York time (U.S. Eastern, with DST handled)
    idx_ny = idx_utc.tz_convert("America/New_York")

    # Weekday mask (Mon–Fri)
    is_us_weekday = idx_ny.weekday < 5  # 0=Mon, 4=Fri

    # Session time mask (09:30–16:00 New York time)
    ny_times = idx_ny.time
    session_start = time(9, 30)
    session_end = time(16, 0)
    is_session_time = np.array(
        [(t >= session_start) and (t <= session_end) for t in ny_times],
        dtype=bool
    )

    # U.S. holiday calendar (federal holidays; close approximation)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=idx_ny.min().date(),
        end=idx_ny.max().date()
    )

    # Normalize times to dates and check membership in holiday list
    is_holiday = pd.Series(idx_ny.normalize()).isin(holidays).to_numpy()

    # Final market mask: weekday, in session, not holiday
    market_mask = is_us_weekday & is_session_time & (~is_holiday)

    # Apply mask
    df_market = df_resampled[market_mask].copy()

    # 2) Compute real indicators from IndicatorsAll
    df_rsi = RSI(df_market)                 # adds 'RSI'
    df_macd = MACD(df_market)               # adds 'MACD_hist'
    df_bb = Bollinger_Bands(df_market)      # adds 'BB_width'

    df_all = df_market.copy()
    df_all["RSI"] = df_rsi["RSI"]
    df_all["MACD_hist"] = df_macd["MACD_hist"]
    df_all["BB_width"] = df_bb["BB_width"]

    # 3) Build pattern states from real indicator values
    pat_rsi = rsi_patterns(df_all, df_all["RSI"])
    pat_macd = macd_patterns(df_all, df_all["MACD_hist"])
    pat_bb = bollinger_patterns(df_all, df_all["BB_width"])

    # 4) Forward returns from close only
    fwd_rets = compute_forward_returns(df_all, horizons=(1, 3, 6))

    # 5) Score each indicator
    rsi_score = score_indicator(pat_rsi, fwd_rets)
    macd_score = score_indicator(pat_macd, fwd_rets)
    bb_score = score_indicator(pat_bb, fwd_rets)

    # 6) Ranking
    scores = {
        "RSI": rsi_score["score"],
        "MACD_Hist": macd_score["score"],
        "BB_Width": bb_score["score"],
    }
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Indicator Ranking (Synthetic Demo) ===\n")
    for name, score in ranking:
        print(f"{name:10s} → Score: {score:.6f}")

    print("\n=== Detailed RSI Score ===")
    print(rsi_score)
    print("\n=== Detailed MACD Score ===")
    print(macd_score)
    print("\n=== Detailed Bollinger Score ===")
    print(bb_score)

if __name__ == "__main__":
    main()
```

Running the above script does provide us with the following output

```
Irregular intervals detected:
 time
2020-05-14 12:00:00   0 days 20:00:00
2020-05-15 12:00:00   0 days 20:00:00
2020-05-18 12:00:00   2 days 20:00:00
2020-05-19 12:00:00   0 days 20:00:00
2020-05-20 12:00:00   0 days 20:00:00
Name: time, dtype: timedelta64[ns]

=== Indicator Ranking (Synthetic Demo) ===

RSI        → Score: 0.485652
MACD_Hist  → Score: 0.427450
BB_Width   → Score: 0.382466

=== Detailed RSI Score ===
{'bullish_count': 271, 'bullish_hit_rate': 0.5006150061500615, 'bullish_mean_ret': 0.003797074603388901, 'bearish_count': 331, 'bearish_hit_rate': 0.46827794561933533, 'bearish_mean_ret': -0.0010230853860337735, 'score': 0.4856515158820541}

=== Detailed MACD Score ===
{'bullish_count': 1434, 'bullish_hit_rate': 0.41668801974563136, 'bullish_mean_ret': 0.00020654619989694508, 'bearish_count': 1460, 'bearish_hit_rate': 0.4378995433789954, 'bearish_mean_ret': 0.000417578535087511, 'score': 0.4274498127460595}

=== Detailed Bollinger Score ===
{'bullish_count': 434, 'bullish_hit_rate': 0.37634408602150543, 'bullish_mean_ret': -0.0006985368250969227, 'bearish_count': 458, 'bearish_hit_rate': 0.38756175169369245, 'bearish_mean_ret': 0.001353899613706402, 'score': 0.3824660279672998}
```

### Ranking and Selecting the Indicators per quarter

We have all our indicators, with their scores, to a quarterly predictive performance, what follow next now is a diagnostic phase where we convert these raw metrics into structured quarterly rankings. Ranking acts as an important role that marks indicator effectiveness that is based on the empirical evaluations we have made above instead of one’s subjective judgment. This hierarchy then sets which indicators deserve more attention and which need to be relegated.

On the whole, the ranking process adheres to three major principles.

1. _Quarter-Specific Independence_.  Every quarter is treated as a self-contained analytical environment. The rankings that are derived from earlier quarters do exert no influence on later ones. This independence means the pipeline detects genuine regime shifts - like transitions from trending to range-bound markets - instead of imposing continuity in cases where there is none.
2. _Composite Score Ordering_.  Indicators are sorted only by the composite score that brings together the accuracy/hit-rate, average forward-return, and amount of generated signals.
3. _Signal Reliability Threshold_. There is a penalization of indicators that have insufficient bullish and bearish signals within a quarter, as these get excluded from ranking. To obtain the rankings from a given pool of indicators, by the quarter, we would use this code snippet that is presented below.

```
# python libraries
import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import pandas as pd
import numpy as np

from datetime import datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---------------------------------------------------------
# Import indicators from your module
# ---------------------------------------------------------
from IndicatorsAll import RSI, MACD, Bollinger_Bands

# ---------------------------------------------------------
# Import quarterly ranking helper from fxi-8
# (make sure the file is named fxi_8.py, not fxi-8.py)
# ---------------------------------------------------------
from fxi_8 import build_quarterly_rankings

# ---------------------------------------------------------
# Indicator scoring function (as given)
# ---------------------------------------------------------
def score_indicator(patterns: pd.DataFrame,
                    fwd_rets: pd.DataFrame,
                    min_signals: int = 10) -> dict:
    """
    Compute predictive power metrics for one indicator within a single quarter.
    patterns: DataFrame with columns ['bullish', 'bearish', 'flat'] (0/1 or bool)
    fwd_rets: DataFrame with columns such as ['fwd_ret_1', 'fwd_ret_3', ...]
    """
    results = {}

    for state in ["bullish", "bearish"]:
        mask = patterns[state] == 1
        num_signals = int(mask.sum())
        results[f"{state}_count"] = num_signals

        if num_signals < min_signals:
            results[f"{state}_hit_rate"] = np.nan
            results[f"{state}_mean_ret"] = np.nan
            continue

        expected_sign = 1 if state == "bullish" else -1
        state_hits = []
        state_mean_rets = []

        for col in fwd_rets.columns:
            r = fwd_rets.loc[mask, col].dropna()
            if r.empty:
                continue

            hits = (np.sign(r) == expected_sign).astype(float).mean()
            state_hits.append(hits)
            state_mean_rets.append(r.mean())

        if state_hits:
            results[f"{state}_hit_rate"] = float(np.mean(state_hits))
            results[f"{state}_mean_ret"] = float(np.mean(state_mean_rets))
        else:
            results[f"{state}_hit_rate"] = np.nan
            results[f"{state}_mean_ret"] = np.nan

    bullish_hr = results.get("bullish_hit_rate", np.nan)
    bearish_hr = results.get("bearish_hit_rate", np.nan)
    bullish_ret = results.get("bullish_mean_ret", 0.0)
    bearish_ret = results.get("bearish_mean_ret", 0.0)

    valid_hit_rates = [x for x in [bullish_hr, bearish_hr] if not np.isnan(x)]
    hit_component = float(np.mean(valid_hit_rates)) if valid_hit_rates else 0.0

    magnitude_component = float((abs(bullish_ret) + abs(bearish_ret)) / 2.0)

    final_score = hit_component + 0.5 * magnitude_component
    results["score"] = final_score

    return results

# ---------------------------------------------------------
# Forward returns from CLOSE ONLY (no indicator info)
# ---------------------------------------------------------
def compute_forward_returns(df: pd.DataFrame,
                            horizons=(1, 3, 6)) -> pd.DataFrame:
    close = df["close"]
    forward_data = {}
    for h in horizons:
        forward_data[f"fwd_ret_{h}"] = (close.shift(-h) / close) - 1.0
    return pd.DataFrame(forward_data, index=df.index)

# ---------------------------------------------------------
# Pattern logic USING REAL INDICATOR VALUES
# ---------------------------------------------------------
def rsi_patterns(df: pd.DataFrame, rsi: pd.Series) -> pd.DataFrame:
    bullish = (rsi < 30).astype(int)
    bearish = (rsi > 70).astype(int)
    flat = ((rsi >= 30) & (rsi <= 70)).astype(int)
    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

def macd_patterns(df: pd.DataFrame, macd_hist: pd.Series) -> pd.DataFrame:
    bullish = (macd_hist > 0).astype(int)
    bearish = (macd_hist < 0).astype(int)
    flat = (macd_hist == 0).astype(int)
    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

def bollinger_patterns(df: pd.DataFrame, bb_width: pd.Series) -> pd.DataFrame:
    price = df["close"]
    price_slope = price.diff()
    contraction = (bb_width < bb_width.rolling(20).mean()).astype(int)

    bullish = ((contraction == 0) & (price_slope > 0)).astype(int)
    bearish = ((contraction == 0) & (price_slope < 0)).astype(int)
    flat = (contraction == 1).astype(int)

    return pd.DataFrame({"bullish": bullish, "bearish": bearish, "flat": flat})

# ---------------------------------------------------------
# MAIN – Full example with RSI, MACD, Bollinger Bands
# ---------------------------------------------------------
def main():
    # ------------------------------------------------------------
    # Initialize MetaTrader 5
    # ------------------------------------------------------------
    if not mt5.initialize(login=XXXXXXXX,
                          server="XXXXX",
                          password="XXX"):
        print("initialize() failed, error code =", mt5.last_error())
        return

    # ------------------------------------------------------------
    # Set start and end dates for history data
    # ------------------------------------------------------------
    end_date = datetime(2025, 12, 1, 0)
    start_date = datetime(2020, 1, 1, 0)

    # Get H4 rates for FXI (CFD symbol "#FXI")
    fxi_rates = mt5.copy_rates_range("#FXI", mt5.TIMEFRAME_H4, start_date, end_date)

    if fxi_rates is None or len(fxi_rates) == 0:
        print("No data returned from MT5.")
        mt5.shutdown()
        return

    # ------------------------------------------------------------
    # Build base DataFrame
    # ------------------------------------------------------------
    df = pd.DataFrame(fxi_rates)

    # MT5 times are in seconds since epoch
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.sort_index()

    # Confirm essential OHLC columns exist
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    # ------------------------------------------------------------
    # Check interval consistency and resample to uniform 4h grid
    # ------------------------------------------------------------
    time_diff = df.index.to_series().diff().dropna()
    irregularities = time_diff[time_diff != pd.Timedelta(hours=4)]
    print("Irregular intervals detected:\n", irregularities.head())

    # Use lowercase "h" to avoid FutureWarning
    df_resampled = df.resample("4h").ffill()

    # ------------------------------------------------------------
    # Attach quarterly segmentation
    # ------------------------------------------------------------
    df_resampled["quarter"] = df_resampled.index.to_period("Q")

    # ------------------------------------------------------------
    # Market-hours mask + U.S. holidays detection
    # ------------------------------------------------------------

    # Assume MT5 timestamps are in UTC. If your server is NOT UTC,
    # adjust this localize step accordingly.
    idx_utc = df_resampled.index.tz_localize("UTC")

    # Convert to New York time (U.S. Eastern, with DST handled)
    idx_ny = idx_utc.tz_convert("America/New_York")

    # Weekday mask (Mon–Fri)
    is_us_weekday = idx_ny.weekday < 5  # 0=Mon, 4=Fri

    # Session time mask (09:30–16:00 New York time)
    ny_times = idx_ny.time
    session_start = time(9, 30)
    session_end = time(16, 0)
    is_session_time = np.array(
        [(t >= session_start) and (t <= session_end) for t in ny_times],
        dtype=bool
    )

    # U.S. holiday calendar (federal holidays; close approximation)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=idx_ny.min().date(),
        end=idx_ny.max().date()
    )

    # Normalize times to dates and check membership in holiday list
    is_holiday = pd.Series(idx_ny.normalize()).isin(holidays).to_numpy()

    # Final market mask: weekday, in session, not holiday
    market_mask = is_us_weekday & is_session_time & (~is_holiday)

    # Apply mask
    df_market = df_resampled[market_mask].copy()

    # ------------------------------------------------------------
    # Compute real indicators from IndicatorsAll
    # ------------------------------------------------------------
    df_rsi = RSI(df_market)                 # expects 'RSI'
    df_macd = MACD(df_market)               # expects 'MACD_hist'
    df_bb = Bollinger_Bands(df_market)      # expects 'BB_width'

    df_all = df_market.copy()
    df_all["RSI"] = df_rsi["RSI"]
    df_all["MACD_hist"] = df_macd["MACD_hist"]
    df_all["BB_width"] = df_bb["BB_width"]

    # ------------------------------------------------------------
    # Per-quarter scoring for each indicator
    # ------------------------------------------------------------
    rows = []

    indicator_specs = {
        "RSI":       ("RSI",        rsi_patterns),
        "MACD_Hist": ("MACD_hist",  macd_patterns),
        "BB_Width":  ("BB_width",   bollinger_patterns),
    }

    for quarter, df_q in df_all.groupby("quarter"):
        fwd_q = compute_forward_returns(df_q, horizons=(1, 3, 6))

        for ind_name, (col_name, pattern_fn) in indicator_specs.items():
            patterns_q = pattern_fn(df_q, df_q[col_name])
            score_dict = score_indicator(patterns_q, fwd_q)

            rows.append({
                "quarter": str(quarter),
                "indicator": ind_name,
                "score": score_dict["score"],
                "bullish_count": score_dict["bullish_count"],
                "bearish_count": score_dict["bearish_count"],
                "bullish_hit_rate": score_dict.get("bullish_hit_rate"),
                "bearish_hit_rate": score_dict.get("bearish_hit_rate"),
                "bullish_mean_ret": score_dict.get("bullish_mean_ret"),
                "bearish_mean_ret": score_dict.get("bearish_mean_ret"),
            })

    results_df = pd.DataFrame(rows)

    print("\n=== Raw per-quarter indicator scores ===\n")
    print(results_df.head())

    # ------------------------------------------------------------
    # Build per-quarter rankings using fxi-8 helper
    # ------------------------------------------------------------
    quarterly_rankings = build_quarterly_rankings(results_df)

    print("\n=== Quarterly Indicator Rankings ===\n")
    for q, ranking in quarterly_rankings.items():
        print(f"{q}:")
        for ind_name, score in ranking:
            print(f"  {ind_name:10s} → {score:.6f}")
        print()

    # Example: top 3 indicators for a specific quarter, if present
    target_q = "2023Q2"
    if target_q in quarterly_rankings:
        print(f"\nTop 3 indicators for {target_q}:")
        for ind_name, score in quarterly_rankings[target_q][:3]:
            print(f"  {ind_name:10s} → {score:.6f}")

    # ------------------------------------------------------------
    # Save per-quarter scoring results for later processing
    # ------------------------------------------------------------
    csv_path = "fxi_quarterly_scores.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved quarterly indicator scores → {csv_path}\n")

    # ------------------------------------------------------------
    # Shutdown MT5
    # ------------------------------------------------------------
    mt5.shutdown()

if __name__ == "__main__":
    main()
```

If we run this code, having got a csv output of the indicator rankings for our three indicators, we get the following matrix of the indicator performance per quarter.

[![i13](https://c.mql5.com/2/185/i13.png)](https://c.mql5.com/2/185/i13.png "https://c.mql5.com/2/185/i13.png")

Thus, with this quarterly ranking mechanism, the pipeline is able to convert raw predictive metrics into recommendations that we can put into action. This ranking list/matrix does serve as an analytical filter, through which only the indicators for which there is empirical evidence, are the ones that get chosen for further development.

### Insights from the Quarterly Scorecards

We performed a dry test run with just 3 indicators, that cover different aspects of the market on the FXI ETF, and the results are arguably disciplined and data-driven. By mapping indicator rank positions on a quarter by quarter basis, our pipeline makes it possible to spot some structural tendencies that could be obscured from analysis over aggregated windows.

Our matrix results though seem to demonstrate that for the FXI ETF, trend indicators are superior across most market regimes. This comes from the MACD histogram repeatedly occupying the top rank, albeit from a very small indicator pool, but outperforming the RSI and Bollinger Bands nonetheless. This could mean that FXI’s mid-quarter structure does favor persistent directional micro-trends in spite of the several periods this ETF has where it seems that the price action is indecisive. So, clearly if we were to ‘wing-it’ and discard trend indicators based on a quick inspection of the FXI chart, we would clearly underperform.

The RSI indicator is able to occupy the top rank only in a few isolated quarters, 2 to be exact, with the other 2 being split between 2nd and 3rd. The RSI also gets ranked only 4 times because we were not able to generate a sufficient number of signals for most of the quarters. This is bound to be expected, since oscillators tend to give predictive benefits only when the market is in reversion-friendly ranges but deteriorate sharply during set trends.

Bollinger Bands width signal seems to settle in the runner-up position to the MACD. It could be argued that its rank improves in quarters where we have volatility compression or transitions, which suggests it could be excelling in structural turning-point conditions. Also, noteworthy is that in quarters where the RSI or the Bollinger Bands have been in pole, the markets were oscillating without a discernable structure. This again makes the case for the MACD and other trend following indicators as the preferred tools for handling the FXI when looking to trade it long term. So, the final ranking of what indicators to use when trading the FXI, based on our 5-year analysis at the 4-hour time frame, would be

1. Trending indicators/MACD histogram;
2. Volatility expansion/contraction indicators such as the Bollinger Bands; and
3. Momentum oscillators such us the RSI, that given the very few signals that we generated, meant this was a selective opportunistic tool.

We now proceed to our final step, where we attempt to crystallize these ideas into a simple Expert Advisor for further testing.

### Conversion to MQL5

The choice of indicators to use when it comes to MQL5, will depend, of course, in the ranking results as we have demonstrated with our three indicator pool size. Most indicators are already coded in MQL5, in fact even custom signal class files, for use in assembling an Expert Advisor via [the MQL5 Wizard](https://www.mql5.com/en/articles/171), are already available ‘out-of-the-box’. Of the three indicators we have tested with, this is true for all except the Bollinger Bands. We can therefore do our own very basic implementation of the Bollinger Bands as follows.

```
//+------------------------------------------------------------------+
//|                                              SignalBollinger.mqh |
//|                             Copyright 2000-2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include <Expert\ExpertSignal.mqh>
// wizard description start

…

class CSignalBollinger : public CExpertSignal
  {
protected:
   CiBands           m_bb;            // object-indicator
   //--- adjusted parameters
   int               m_ma_period;      // the "period of averaging" parameter of the indicator
   int               m_ma_shift;       // the "time shift" parameter of the indicator
   ENUM_MA_METHOD    m_ma_method;      // the "method of averaging" parameter of the indicator
   ENUM_APPLIED_PRICE m_ma_applied;    // the "object of averaging" parameter of the indicator
   double            m_deviation;      // the "deviation" parameter of the indicator
   double            m_limit_in;       // threshold sensitivity of the 'rollback zone'
   double            m_limit_out;      // threshold sensitivity of the 'break through zone'
   //--- "weights" of market models (0-100)
   int               m_pattern_0;      // model 0 "price is near the necessary border of the envelope"
   int               m_pattern_1;      // model 1 "price crossed a border of the envelope"

public:
                     CSignalBollinger(void);
                    ~CSignalBollinger(void);
   //--- methods of setting adjustable parameters
   void              PeriodMA(int value)                 { m_ma_period=value;        }
   void              Shift(int value)                    { m_ma_shift=value;         }
   void              Method(ENUM_MA_METHOD value)        { m_ma_method=value;        }
   void              Applied(ENUM_APPLIED_PRICE value)   { m_ma_applied=value;       }
   void              Deviation(double value)             { m_deviation=value;        }
   void              LimitIn(double value)               { m_limit_in=value;         }
   void              LimitOut(double value)              { m_limit_out=value;        }
   //--- methods of adjusting "weights" of market models
   void              Pattern_0(int value)                { m_pattern_0=value;        }
   void              Pattern_1(int value)                { m_pattern_1=value;        }
   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are formed
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);

protected:
   //--- method of initialization of the indicator
   bool              InitMA(CIndicators *indicators);
   //--- methods of getting data
   double            Upper(int ind)                      { return(m_bb.Upper(ind)); }
   double            Lower(int ind)                      { return(m_bb.Lower(ind)); }
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSignalBollinger::CSignalBollinger(void) : m_ma_period(45),
                                           m_ma_shift(0),
                                           m_ma_method(MODE_SMA),
                                           m_ma_applied(PRICE_CLOSE),
                                           m_deviation(0.15),
                                           m_limit_in(0.2),
                                           m_limit_out(0.2),
                                           m_pattern_0(90),
                                           m_pattern_1(70)
  {
//--- initialization of protected data
   m_used_series=USE_SERIES_OPEN+USE_SERIES_HIGH+USE_SERIES_LOW+USE_SERIES_CLOSE;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSignalBollinger::~CSignalBollinger(void)
  {
  }
//+------------------------------------------------------------------+
//| Validation settings protected data.                              |
//+------------------------------------------------------------------+
bool CSignalBollinger::ValidationSettings(void)
  {
//--- validation settings of additional filters
   if(!CExpertSignal::ValidationSettings())
      return(false);
//--- initial data checks
   if(m_ma_period<=0)
     {
      printf(__FUNCTION__+": period MA must be greater than 0");
      return(false);
     }
//--- ok
   return(true);
  }

…

//+------------------------------------------------------------------+
//| "Voting" that price will grow.                                   |
//+------------------------------------------------------------------+
int CSignalBollinger::LongCondition(void)
  {
   int result=0;
   int idx   =StartIndex();
   double close=Close(idx);
   double upper=Upper(idx);
   double lower=Lower(idx);
   double width=upper-lower;
//--- if the model 0 is used and price is in the rollback zone, then there is a condition for buying
   if(IS_PATTERN_USAGE(0) && close<lower+m_limit_in*width && close>lower-m_limit_out*width)
      result=m_pattern_0;
//--- if the model 1 is used and price is above the rollback zone, then there is a condition for buying
   if(IS_PATTERN_USAGE(1) && close>upper+m_limit_out*width)
      result=m_pattern_1;
//--- return the result
   return(result);
  }
//+------------------------------------------------------------------+
//| "Voting" that price will fall.                                   |
//+------------------------------------------------------------------+
int CSignalBollinger::ShortCondition(void)
  {
   int result  =0;
   int idx     =StartIndex();
   double close=Close(idx);
   double upper=Upper(idx);
   double lower=Lower(idx);
   double width=upper-lower;
//--- if the model 0 is used and price is in the rollback zone, then there is a condition for selling
   if(IS_PATTERN_USAGE(0) && close>upper-m_limit_in*width && close<upper+m_limit_out*width)
      result=m_pattern_0;
//--- if the model 1 is used and price is above the rollback zone, then there is a condition for selling
   if(IS_PATTERN_USAGE(1) && close<lower-m_limit_out*width)
      result=m_pattern_1;
//--- return the result
   return(result);
  }
//+------------------------------------------------------------------+
```

With the Bollinger Bands implemented, one approach we could take in our implementation could simply be to assemble an Expert Advisor in the Wizard that selects all three indicators, where the weighting of each indicator could be in proportion to the number of quarters that indicator has ranked first. Though ‘logical’ this approach could of course run into challenges, where, many indicators are used and trader’s judgment on what to include would be more important. The reader of course can experiment on this on an asset by asset basis to find what works best. For our purposes, though, the wizard assembled Expert Advisor would have its header appearing as follows.

```
//+------------------------------------------------------------------+
//|                                                          FXI.mq5 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include                                                          |
//+------------------------------------------------------------------+
#include <Expert\Expert.mqh>
//--- available signals
#include <Expert\Signal\SignalMACD.mqh>
#include <Expert\Signal\SignalBollinger.mqh>
#include <Expert\Signal\SignalRSI.mqh>
//--- available trailing
#include <Expert\Trailing\TrailingNone.mqh>
//--- available money management
#include <Expert\Money\MoneyFixedMargin.mqh>
```

### Conclusion

The approach that has been presented here in this article, hopefully, has shown a rigorous, reproducible, and extensible method of choosing indicators. Our method has been grounded in multi-quarter empirical analysis. By combining Python’s robustness and speed when it comes to analysis, with MQL5 execution environment, we do present a Codex Pipeline that sets up a unified work-flow for discovering, validating and using technical indicators. A few important attributes set our approach apart as possibly a robust framework for ongoing research and development.

1. _Regime-Aware Evaluation._ By using quarterly segmentation, we set the stage for ensuring that indicator performance is understood within contextually meaningful periods. We are able to capture transitions in volatility, momentum and also evade structural noise.
2. _Formal Pattern Logic._ By adopting the deterministic states of bullish, bearish and neutral, we provide a broadly understood vocabulary for easy comparison across the many different indicator types.
3. _Forward Looking Scoring._ By doing evaluations that are only based on future returns, we eliminate hindsight bias and see to it that indicators are accredited solely for their forecast ability.
4. _Directional Operational Translation._ We wrap up the pipeline, in MQL5, by providing actionable components for algorithmic trading.

With time, as data is updated, and new quarters come into consideration, the Codex Pipeline can always be re-run to update the rankings of a previously considered indicator pool, and then refine its subsequent MQL5 logic. This cycle approach places this method as a long term analytical tool for adapting to not just the FXI ETF evolving market structure, but also any asset for which structured indicator discovery can be insightful. After all, from casually inspecting the FXI price chart, few would have predicted that trend based indicators are the preferred tool.

| name | description |
| --- | --- |
| SignalBollinger.mqh | Custom signal class for Bollinger Bands Indicator |
| FXI.mq5 | Wizard Assembled Expert Advisor Main file |
| fxi-py all.zip | Zipped file of referenced python code |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20550.zip "Download all attachments in the single ZIP archive")

[SignalBollinger.mqh](https://www.mql5.com/en/articles/download/20550/SignalBollinger.mqh "Download SignalBollinger.mqh")(9.41 KB)

[FXI.mq5](https://www.mql5.com/en/articles/download/20550/FXI.mq5 "Download FXI.mq5")(9.06 KB)

[fxi-py\_all.zip](https://www.mql5.com/en/articles/download/20550/fxi-py_all.zip "Download fxi-py_all.zip")(28.11 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)
- [MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

**[Go to discussion](https://www.mql5.com/en/forum/501651)**

![Automated Risk Management for Passing Prop Firm Challenges](https://c.mql5.com/2/185/19655-automated-risk-management-for-logo.png)[Automated Risk Management for Passing Prop Firm Challenges](https://www.mql5.com/en/articles/19655)

This article explains the design of a prop-firm Expert Advisor for GOLD, featuring breakout filters, multi-timeframe analysis, robust risk management, and strict drawdown protection. The EA helps traders pass prop-firm challenges by avoiding rule breaches and stabilizing trade execution under volatile market conditions.

![Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://c.mql5.com/2/185/20414-adaptive-smart-money-architecture-logo.png)[Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)

This topic explores how to build an Adaptive Smart Money Architecture (ASMA)—an intelligent Expert Advisor that merges Smart Money Concepts (Order Blocks, Break of Structure, Fair Value Gaps) with real-time market sentiment to automatically choose the best trading strategy depending on current market conditions.

![Automating Trading Strategies in MQL5 (Part 46): Liquidity Sweep on Break of Structure (BoS)](https://c.mql5.com/2/185/20569-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 46): Liquidity Sweep on Break of Structure (BoS)](https://www.mql5.com/en/articles/20569)

In this article, we build a Liquidity Sweep on Break of Structure (BoS) system in MQL5 that detects swing highs/lows over a user-defined length, labels them as HH/HL/LH/LL to identify BOS (HH in uptrend or LL in downtrend), and spots liquidity sweeps when price wicks beyond the swing but closes back inside on a bullish/bearish candle.

![Overcoming The Limitation of Machine Learning (Part 9): Correlation-Based Feature Learning in Self-Supervised Finance](https://c.mql5.com/2/185/20514-overcoming-the-limitation-of-logo.png)[Overcoming The Limitation of Machine Learning (Part 9): Correlation-Based Feature Learning in Self-Supervised Finance](https://www.mql5.com/en/articles/20514)

Self-supervised learning is a powerful paradigm of statistical learning that searches for supervisory signals generated from the observations themselves. This approach reframes challenging unsupervised learning problems into more familiar supervised ones. This technology has overlooked applications for our objective as a community of algorithmic traders. Our discussion, therefore, aims to give the reader an approachable bridge into the open research area of self-supervised learning and offers practical applications that provide robust and reliable statistical models of financial markets without overfitting to small datasets.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/20550&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062531789559342065)

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