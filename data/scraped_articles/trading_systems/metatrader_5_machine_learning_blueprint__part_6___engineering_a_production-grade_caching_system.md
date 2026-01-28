---
title: MetaTrader 5 Machine Learning Blueprint (Part 6): Engineering a Production-Grade Caching System
url: https://www.mql5.com/en/articles/20302
categories: Trading Systems, Integration, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T11:32:07.203341
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=fxicjfcaetfdbuxdzpssaelanrfhxsfs&ssn=1769157125076373169&ssn_dr=1&ssn_sr=0&fv_date=1769157125&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20302&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MetaTrader%205%20Machine%20Learning%20Blueprint%20(Part%206)%3A%20Engineering%20a%20Production-Grade%20Caching%20System%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915712600174988&fz_uniq=6470814543299650565&sv=2552)

MetaTrader 5 / Trading systems


- [Introduction](https://www.mql5.com/en/articles/20302#para1)
- [Part I: Architectural Foundations](https://www.mql5.com/en/articles/20302#para4)
- [Part II: The Robust Cacheable Decorator](https://www.mql5.com/en/articles/20302#para9)
- [Part III: Advanced Caching Patterns](https://www.mql5.com/en/articles/20302#para10)
- [Part IV: Performance Monitoring and Optimization](https://www.mql5.com/en/articles/20302#para14)
- [Part V: Integration with Existing Projects](https://www.mql5.com/en/articles/20302#para16)
- [Conclusion: From Research Velocity to Execution Speed](https://www.mql5.com/en/articles/20302#para17)
- [Code Repository](https://www.mql5.com/en/articles/20302#para22)
- [Module Files](https://www.mql5.com/en/articles/20302#para23)

### Introduction

In our previous installments of the Machine Learning Blueprint series, we’ve built a robust pipeline for financial machine learning—from ensuring [data integrity](https://www.mql5.com/en/articles/17520) against look-ahead bias to implementing sophisticated labeling methods like [Triple-Barrier](https://www.mql5.com/en/articles/18864) and [Trend-Scanning](https://www.mql5.com/en/articles/19253). However, as our strategies or ML models—as with [sequentially bootstrapped random forests](https://www.mql5.com/en/articles/20059)—grow more complex, we face a critical challenge: computational bottlenecks that prevent rapid iteration.

You've built a promising mean reversion strategy. Your backtest shows a Sharpe ratio of 1.8, consistent profits across market regimes, and clean equity curves. You're ready to optimize parameters, test different lookback periods, and validate with walk-forward analysis.

Then reality hits.

Each parameter combination takes 6 minutes to compute. You want to test 50 variations. That's 5 hours of waiting. Change your feature engineering? Another 5 hours. Add a new indicator? You get the idea.

The real cost isn't just time—it's lost opportunities. While you wait for computations, you can't iterate, can't test new ideas, can't improve your edge. Your development velocity grinds to a halt.

This is the problem that killed my early trading strategies. I would spend entire weekends running backtests, only to realize Monday morning that I'd made a simple mistake in my code. More waiting. More frustration.

There had to be a better way.

This article shows you how to eliminate this bottleneck using intelligent caching. By the end, you'll understand how to:

- Reduce strategy optimization time from hours to minutes
- Test 50+ parameter combinations in the time it used to take for 5
- Iterate on features and models without recomputing everything

Let's start with the problem you're facing right now.

### The Computational Pipeline Visualization

Before implementing caching, it's important to understand how much time each operation in the ML pipeline actually takes. The diagram below shows the full path from loading ticks to the final backtest—and how many seconds are spent on each step without caching.

![WITHOUT CACHING](https://c.mql5.com/2/182/WITHOUT_CACHING.png)

Now let's look at how the same sequence of operations works with AFML caching enabled. Note that most steps are no longer recalculated but loaded instantly. The difference in speed is evident in the diagram below.

![With AFML Caching](https://c.mql5.com/2/182/WITH_AFML_CACHING__2.png)

### Why Generic Caching Fails for Financial ML

Let's see why the obvious solution doesn't work.

It might seem like a no-brainer to just use the standard lru\_cache. Below is a simple example demonstrating that Python's built-in cache completely breaks down when working with financial data structures like Pandas Series and DataFrame.

```
# This is USELESS for financial data. Don't do this.
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

compute_rsi(bb_df.close, period=14)

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[25], line 18
     15     rsi = 100 - (100 / (1 + rs))
     16     return rsi
---> 18 compute_rsi(bb_df.close, period=14)

TypeError: unhashable type: 'Series'
```

While Python offers functools.lru\_cache, it’s fundamentally inadequate for financial ML:

1. Memory-only storage—cache disappears when Python exits.
2. No persistence across sessions—rerun everything after each restart.
3. No automatic invalidation—stale cache when code changes.
4. Poor handling of NumPy/Pandas—hashing issues with arrays.
5. No distributed caching—can’t share cache across processes.
6. No financial data awareness—doesn’t understand timestamps or look-ahead bias.

### Part I: Architectural Foundations

#### The Core Design Principles

Our AFML caching system is built on three fundamental pillars:

![AFML Cache Architecture](https://c.mql5.com/2/182/AFML_CACHE_ARCHITECTURE.png)

Let’s understand why each component exists and how they work together.

#### Challenge \#1: Persistent Storage

Let's start with the first and most obvious limitation—the standard cache lives only in RAM. As soon as you restart Python, all results disappear. Here's a short demonstration of what the "normal way" looks like and why it's useless in real-world ML pipelines.

```
# Traditional approach - memory only
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_features(data):
    # Expensive computation
    return features  # Lost on restart!
```

AFML solves this problem simply: we move the cache to disk. Joblib persists calculation results between Python runs, so even after a restart, you instantly get the previously calculated data. An example implementation is below.

```
# AFML approach - persistent
from joblib import Memory
from appdirs import user_cache_dir

memory = Memory(location=user_cache_dir("afml"), verbose=0)

@memory.cache
def compute_features(data):
    # Expensive computation
    return features  # Saved to disk automatically!
```

#### Challenge \#2: Hashing Complex Financial Data Structures

And then there's an even bigger problem _—_ financial data can't be hashed using standard methods. Pandas DataFrames, NumPy arrays, and dates are all "unhashable", and a regular cache simply won't work. Here's an example of how this breaks down.

```
# This fails!
import pandas as pd
from functools import lru_cache

@lru_cache
def bad_cache_example(df: pd.DataFrame):
    return df.mean()

df = pd.DataFrame({'price': [100, 101, 102]})
bad_cache_example(df)  # TypeError: unhashable type: 'DataFrame'
```

To ensure the cache works correctly, AFML creates its own key generator. It can "understand" DataFrames: their structure, data types, columns, indexes, and, most importantly, time range. Here's what a custom hashing implementation looks like.

```
# afml/cache/robust_cache_keys.py

class CacheKeyGenerator:
    """Generate collision-resistant cache keys for ML data structures."""

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame, name: str) -> str:
        """
        Hash DataFrame with attention to:
        1. Shape and structure
        2. Column names and types
        3. Index (especially DatetimeIndex)
        4. Actual data content
        """
        parts = [\
            f"shape_{df.shape}",\
            f"cols_{hashlib.md5(str(tuple(df.columns)).encode()).hexdigest()[:8]}",\
            f"dtypes_{hashlib.md5(str(tuple(df.dtypes)).encode()).hexdigest()[:8]}",\
        ]

        # Special handling for DatetimeIndex (critical for financial data!)
        if isinstance(df.index, pd.DatetimeIndex):
            # Hash: start date, end date, and length
            # This catches both data changes AND temporal shifts
            parts.append(f"idx_dt_{df.index[0]}_{df.index[-1]}_{len(df.index)}")
        else:
            idx_hash = hashlib.md5(str(tuple(df.index)).encode()).hexdigest()[:8]
            parts.append(f"idx_{idx_hash}")

        # For large DataFrames, sample for performance
        if df.size > 10000:
            # Sample ~100 rows evenly distributed
            sample_rows = df.iloc[::max(1, len(df) // 100)]
            content_hash = hashlib.md5(sample_rows.values.tobytes()).hexdigest()[:8]
        else:
            # Hash full content for small DataFrames
            content_hash = hashlib.md5(df.values.tobytes()).hexdigest()[:8]

        parts.append(f"data_{content_hash}")

        return f"{name}_df_{'_'.join(parts)}"
```

To understand the value of custom hashing, let's compare how standard Python caches data and how AFML does it. Below is an illustration of why the standard approach leads to look-ahead bias, while AFML does not.

![DataFrame Hash Comparison](https://c.mql5.com/2/182/DataFrame_Hash_Comparison.png)

#### Challenge \#3: Automatic Cache Invalidation When Code Changes

Even a perfect cache becomes useless if you edit your code and old, stale results remain in the cache. AFML automatically tracks any changes to functions and flushes only the portion of the cache that is out of date. Let's walk through how this is implemented.

AFML stores a hash of the function's source code and the file's last modification date. If anything changes, the cache for that function is automatically reset. Here's how this mechanism works internally.

```
# afml/cache/selective_cleaner.py
class FunctionTracker:
    """
    Tracks function signatures and source code hashes.
    Automatically detects when functions change.
    """

    def track_function(self, func) -> bool:
        """
        Returns True if function has changed since last tracking.
        """
        func_name = f"{func.__module__}.{func.__qualname__}"

        # Get current function metadata
        current_hash = self._get_function_hash(func)      # Hash source code
        current_mtime = self._get_file_mtime(func)        # File modification time

        # Compare with stored metadata
        stored = self.tracked_functions.get(func_name, {})
        stored_hash = stored.get("hash")
        stored_mtime = stored.get("mtime")

        # Function changed if EITHER hash or mtime differs
        has_changed = (
            current_hash != stored_hash or
            current_mtime != stored_mtime or
            stored_hash is None  # New function
        )

        if has_changed:
            # Update tracking data
            self.tracked_functions[func_name] = {
                "hash": current_hash,
                "mtime": current_mtime,
                "module": func.__module__,
            }
            self._save_tracking_data()

        return has_changed

    def _get_function_hash(self, func) -> Optional[str]:
        """Hash function source code."""
        try:
            source = inspect.getsource(func)
            return hashlib.md5(source.encode()).hexdigest()
        except (OSError, TypeError):
            return None
```

To see this in action, consider a real-world scenario: a function initially contains an error, then we fix it—and AFML automatically detects the change, clearing only the relevant portion of the cache.

![Smart Cacheable: Automatic Invalidation Flow](https://c.mql5.com/2/182/Smart_Cacheable.png)

### Part II: The Robust Cacheable Decorator

All AFML capabilities are combined in a single powerful factory decorator. It creates the desired caching type: regular, temporary, data-tracking, code-tracking, and so on. Here's the source code for this factory.

```
# afml/cache/robust_cache_keys.py

def create_robust_cacheable(
    track_data_access: bool = False,
    dataset_name: Optional[str] = None,
    purpose: Optional[str] = None,
    use_time_awareness: bool = False,
):
    """
    Factory function to create robust cacheable decorators.
    This is where all the magic comes together.
    """
    from functools import wraps
    from . import cache_stats, memory
    from .cache_monitoring import get_cache_monitor

    def decorator(func):
        func_name = f"{func.__module__}.{func.__qualname__}"
        cached_func = memory.cache(func)  # Use joblib for persistence
        seen_signatures = set()           # Track cache hits/misses
        monitor = get_cache_monitor()     # Performance monitoring

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Step 1: Generate cache key
            try:
                if use_time_awareness:
                    cache_key = TimeSeriesCacheKey.generate_key_with_time_range(
                        func, args, kwargs
                    )
                else:
                    cache_key = CacheKeyGenerator.generate_key(func, args, kwargs)

                # Step 2: Track hit/miss
                if cache_key in seen_signatures:
                    cache_stats.record_hit(func_name)
                    is_hit = True
                else:
                    cache_stats.record_miss(func_name)
                    seen_signatures.add(cache_key)
                    is_hit = False

            except Exception as e:
                logger.warning(f"Cache key generation failed: {e}")
                cache_stats.record_miss(func_name)
                cache_key = None
                is_hit = False

            # Step 3: Track data access if requested (prevent look-ahead bias)
            if track_data_access:
                try:
                    from .data_access_tracker import get_data_tracker
                    _track_dataframe_access(
                        get_data_tracker(), args, kwargs, dataset_name, purpose
                    )
                except Exception as e:
                    logger.warning(f"Data tracking failed: {e}")

            # Step 4: Track access time (for monitoring)
            monitor.track_access(func_name)

            # Step 5: Execute function with timing
            start_time = time.time()
            try:
                result = cached_func(*args, **kwargs)

                # Track computation time for misses
                if not is_hit:
                    computation_time = time.time() - start_time
                    monitor.track_computation_time(func_name, computation_time)

                return result

            except (EOFError, pickle.PickleError, OSError) as e:
                # Handle cache corruption gracefully
                logger.warning(f"Cache corruption: {type(e).__name__} - recomputing")

                # Clear corrupted cache
                if cache_key is not None:
                    _clear_corrupted_cache(cached_func, cache_key)

                # Execute function directly
                return func(*args, **kwargs)

        wrapper._afml_cacheable = True
        return wrapper

    return decorator
```

```
# Standard decorators
robust_cacheable = create_robust_cacheable(use_time_awareness=False)
time_aware_cacheable = create_robust_cacheable(use_time_awareness=True)

# Data tracking decorators
data_tracking_cacheable = lambda dataset_name, purpose: create_robust_cacheable(
    track_data_access=True, dataset_name=dataset_name, purpose=purpose, use_time_awareness=False
)

time_aware_data_tracking_cacheable = lambda dataset_name, purpose: create_robust_cacheable(
    track_data_access=True, dataset_name=dataset_name, purpose=purpose, use_time_awareness=True
)
```

### Part III: Advanced Caching Patterns

#### Pattern \#1: Time-Aware Caching for Walk-Forward Analysis

In walk-forward validation, it's critical that the cache distinguishes between different time periods. Without this, collisions and incorrect results can easily occur. The diagram below shows how this works in practice.

```
Time Series Data (2024):
─────────────────────────────────────────────────────────────────────
Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │    │    │    │    │    │
▼────▼────▼────▼────▼────▼────▼────▼────▼────▼────▼────▼────▼

Walk-Forward Splits:
─────────────────────────────────────────────────────────────────────

Split 1:
        Train Period: Jan-Mar              Test Period: Apr
        ├─────────────────────────────┤    ├───────┤

Split 2:
            Train Period: Feb-Apr              Test Period: May
            ├─────────────────────────────┤    ├───────┤

Split 3:
                Train Period: Mar-May              Test Period: Jun
                ├─────────────────────────────┤    ├───────┤
```

The example below illustrates how using regular caching with two different time periods can produce the same hash. This leads to the model being trained on the wrong data, and look-ahead bias occurs.

```
@robust_cacheable  # Generic caching
def train_model(data, params):
    return model

Split 1: data = data['2024-01':'2024-03']
         Cache Key: hash(data.values)
         └──> "a7f3e8d92c1b..."        ◄──┐
                                          │
Split 2: data = data['2024-02':'2024-04'] │
         Cache Key: hash(data.values)     │
         └──> "a7f3e8d92c1b..."        ◄──┘ COLLISION!
              └──> Cache hit on DIFFERENT time period
              └──> Model trained on wrong data!
```

Time-aware cache adds a time range during key generation, completely eliminating such collisions. See how the cache now correctly distinguishes between periods.

```
@time_aware_cacheable  # Includes temporal info
def train_model(data, params):
    return model

Split 1: data = data['2024-01':'2024-03']
         Cache Key: hash(data.values + "2024-01_2024-03")
         └──> "a7f3_time_2024-01_2024-03"

Split 2: data = data['2024-02':'2024-04']
         Cache Key: hash(data.values + "2024-02_2024-04")
         └──> "b9e4_time_2024-02_2024-04"  ← Different!
              └──> Cache miss (correct)
              └──> Train new model for this period ✓

Split 3: data = data['2024-03':'2024-05']
         Cache Key: hash(data.values + "2024-03_2024-05")
         └──> "c7d1_time_2024-03_2024-05"  ← Different!
              └──> Cache miss (correct)
              └──> Train new model for this period ✓
```

Here is an implementation of a class that adds timestamps to cache keys.

```
class TimeSeriesCacheKey(CacheKeyGenerator):
    """Extended cache key generator with time-series awareness."""

    @staticmethod
    def generate_key_with_time_range(
        func,
        args: tuple,
        kwargs: dict,
        time_range: Tuple[pd.Timestamp, pd.Timestamp] = None
    ) -> str:
        """
        Generate cache key that includes time range information.
        Critical for preventing temporal data leakage.
        """
        base_key = CacheKeyGenerator.generate_key(func, args, kwargs)

        if time_range is None:
            # Try to extract time range from data
            time_range = TimeSeriesCacheKey._extract_time_range(args, kwargs)

        if time_range:
            start, end = time_range
            time_hash = f"time_{start}_{end}"
            return f"{base_key}_{time_hash}"

        return base_key
```

```
# Usage in walk-forward validation
@time_aware_cacheable
def train_on_period(data: pd.DataFrame, params: dict) -> Model:
    """
    Train model on specific time period.
    Cache is automatically keyed by time range!
    """
    model = RandomForestClassifier(**params)
    model.fit(data.drop('target', axis=1), data['target'])
    return model

# Walk-forward loop
for train_start, train_end, test_start, test_end in walk_forward_splits:
    # Each period is cached independently
    train_data = data.loc[train_start:train_end]
    model = train_on_period(train_data, params)  # Cached per period

    test_data = data.loc[test_start:test_end]
    predictions = model.predict(test_data)
```

#### Pattern \#2: Cross-Validation Caching with Sklearn Estimators

Cross-validation caching is a separate issue. Sklearn classifiers contain internal state that can't be hashed directly. AFML solves this by hashing only the model type and its parameters. The key function is below.

```
# afml/cache/cv_cache.py

def _hash_classifier(clf: BaseEstimator) -> str:
    """
    Generate stable hash for sklearn classifier.

    KEY INSIGHT: Hash the type + parameters, NOT the trained state!
    """
    try:
        clf_type = type(clf).__name__
        params = clf.get_params(deep=True)

        # Filter out non-serializable params
        serializable_params = {}
        for k, v in params.items():
            try:
                json.dumps(v)  # Test if JSON serializable
                serializable_params[k] = v
            except (TypeError, ValueError):
                # Use type name for non-serializable params
                serializable_params[k] = f"<{type(v).__name__}>"

        # Create stable hash
        param_str = json.dumps(serializable_params, sort_keys=True)
        combined = f"{clf_type}_{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    except Exception as e:
        logger.debug(f"Failed to hash classifier: {e}")
        return f"clf_{type(clf).__name__}_{id(clf)}"
```

Here's what a decorator for caching cross-validation results looks like. It saves a tremendous amount of time when selecting hyperparameters.

```
@cv_cacheable
def ml_cross_val_score(
    classifier: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen,  # PurgedKFold, TimeSeriesSplit, etc.
    sample_weight_train: Optional[np.ndarray] = None,
    scoring: str = 'neg_log_loss'
) -> np.ndarray:
    """
    Cross-validation with proper caching.

    Caches based on:
    - Classifier type and parameters (not trained state)
    - Data content (X, y)
    - CV generator configuration
    - Sample weights
    """
    scores = []

    for train_idx, test_idx in cv_gen.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train fresh model (not from cache)
        model = clone(classifier)

        if sample_weight_train is not None:
            model.fit(X_train, y_train,
                     sample_weight=sample_weight_train[train_idx])
        else:
            model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        scores.append(score)

    return np.array(scores)
```

#### Pattern \#3: Preventing Test Set Contamination

In ML development, one of the most hidden mistakes is accidentally using test data during training. AFML monitors every access to the dataset and alerts you to any leaks.

Here's how AFML records every data access, capturing the time range, purpose (train, test, validate), and source of the call. Based on these logs, the system then generates a report on potential leaks.

```
# afml/cache/data_access_tracker.py

class DataAccessTracker:
    """
    Track every data access to detect test set contamination.

    This is CRITICAL for preventing data snooping bias.
    """

    def log_access(
        self,
        dataset_name: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        purpose: str,  # 'train', 'test', 'validate', 'optimize'
        data_shape: Optional[Tuple[int, int]] = None,
    ):
        """Log a dataset access with full temporal metadata."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "purpose": purpose,
            "data_shape": str(data_shape) if data_shape else None,
            "caller": self._get_caller_info(),  # Stack trace
        }

        self.access_log.append(entry)
        logger.debug(
            f"Logged access: {dataset_name} [{start_date} to {end_date}] "
            f"for {purpose}"
        )
```

```
# Usage with caching decorator
@time_aware_data_tracking_cacheable(dataset_name="eur_usd_2024", purpose="test")
def evaluate_on_test_set(test_data: pd.DataFrame, model) -> dict:
    """
    Evaluate model on test set.
    Access is logged automatically!
    """
    predictions = model.predict(test_data)
    metrics = calculate_metrics(predictions, test_data['target'])
    return metrics
```

To check for contamination during our model development process we call _print\_contamination\_report()_ as shown:

```
# Later, check for contamination
from afml.cache import print_contamination_report

print_contamination_report()
```

### ![Data Contamination Report](https://c.mql5.com/2/182/DATA_CONTAMINATION_REPORT.png)

### Part IV: Performance Monitoring and Optimization

#### Cache Health Monitoring

AFML also provides built-in cache health monitoring tracking:

- which functions are most frequently cached
- where cache misses occur
- what the cache size is
- whether there are any suspicious situations

Below is a sample report:

```
from afml.cache import print_cache_health

print_cache_health()
```

### ![Cache Health Report](https://c.mql5.com/2/182/CACHE_HEALTH_REPORT.png)

### Part V: Integration with Existing Projects

To start using the system, simply import the module and use the desired decorator. No complicated configuration required—here's an example of what it looks like in a real project.

Run the code below to clone from your terminal, or download [cache.zip](https://www.mql5.com/en/articles/download/20302/210299/cache.zip ".zip"):

```
git clone https://github.com/pnjoroge54/Machine-Learning-Blueprint.git
cd Machine-Learning-Blueprint/afml/cache
```

And then copy the cache modules to your own package:

```
my_package/
├── cache/
│   ├── __init__.py
│   ├── backtest_cache.py
│   ├── cache_monitoring.py
│   ├── cv_cache.py
│   ├── data_access_tracker.py
│   ├── robust_cache_keys.py
│   ├── selective_cleaner.py
│   └── mql5_bridge.py
```

For full details on implementing this caching system with your project see [user\_guide.py](https://www.mql5.com/en/articles/download/20302/210299/user_guide.py ".py").

### Conclusion: From Research Velocity to Execution Speed

The AFML caching system fundamentally transforms financial ML development by addressing the two critical bottlenecks that separate research from production: iteration speed and execution latency.

#### Research Velocity: The Foundation of Alpha Generation

In the research phase, our caching architecture enables unprecedented exploration:

- Rapid Feature Engineering: Test dozens of technical indicator combinations, RSI periods, and volatility calculations without recomputing base transformations. The system intelligently caches intermediate results, allowing you to iterate on feature selection rather than waiting for computation.

- Exhaustive Strategy Validation: Run PurgedKFold cross-validation across hundreds of parameter combinations. Each data fold’s features and labels remain cached, enabling rigorous backtesting at scale without the temporal data leakage that plagues traditional approaches.

- Multi-Timeframe Analysis: Experiment with complex feature interactions across 1-minute, 5-minute, and hourly timeframes. The time-aware caching ensures each period’s computations remain independent and reproducible.

- Labeling Strategy Optimization: Compare Triple-Barrier, Trend-Scanning, and Meta-Labeling approaches without recalculating expensive bar sampling operations.


#### The Complete Pipeline: Research to Execution

Our caching architecture creates a seamless transition from experimental research to production trading:

1. Research Phase: Use Python’s rich ecosystem with AFML caching for rapid experimentation and validation
2. Model Export: Convert validated models to ONNX format for dependency-free deployment
3. Feature Pipeline Migration: Implement critical feature computations natively in MQL5 with parallel caching
4. Production Deployment: Execute strategies with microsecond latency while maintaining research integrity

#### Data Integrity: The Unseen Advantage

Beyond performance, the system’s automatic data access tracking prevents the subtle contamination that undermines production ML systems. Complete audit trails ensure you never accidentally optimize on test data, while temporal awareness guarantees walk-forward validation integrity.

#### Looking Ahead: The Final Frontier

In our next installment, we complete the pipeline with:

1. ONNX Model Deployment: Export scikit-learn and custom models to run natively in MQL5 without Python dependencies
2. MQL5 Inference Engine: Build ultra-low-latency prediction systems that operate in microseconds
3. Hybrid Feature Management: Design intelligent systems where complex features compute in Python during research, then migrate to MQL5 for production
4. Real-Time Model Monitoring: Implement drift detection and performance tracking within the MQL5 execution environment

The result is a complete ecosystem where you can research with Python’s flexibility, deploy with MQL5’s performance, and maintain scientific rigor throughout the entire lifecycle. This isn’t just about faster computation—it’s about creating a framework where sophisticated ML strategies can actually work in live markets.

### Code Repository

All code from this article is available in the [Machine-Learning-Blueprint](https://www.mql5.com/go?link=https://github.com/pnjoroge54/Machine-Learning-Blueprint/tree/main/afml/cache "https://github.com/pnjoroge54/Machine-Learning-Blueprint/tree/main/afml/cache") repository. Run the code below to clone from your terminal:

```
git clone https://github.com/pnjoroge54/Machine-Learning-Blueprint.git
cd Machine-Learning-Blueprint/afml/cache
```

### Module Files

| Module File | Purpose | Key Features | When to Use |
| --- | --- | --- | --- |
| \_\_init\_\_.py | Central initialization and coordination module | • Initializes all cache subsystems <br>• Sets up Numba and Joblib caching <br>• Provides unified API for all cache functions • Exports convenience functions <br>• Configures cache directories | Import this to access any cache functionality. It's the single entry point for the entire system. |
| robust\_cache\_keys.py | Advanced cache key generation for financial data | • Hashes NumPy arrays correctly <br>• Handles Pandas DataFrames with DatetimeIndex <br>• Time-series aware key generation <br>• Sklearn estimator hashing <br>• Prevents temporal data leakage | Use for any function that processes financial time-series data, DataFrames, or ML models. |
| selective\_cleaner.py | Intelligent cache invalidation system | • Tracks function source code changes <br>• Automatic cache clearing when code changes <br>• Selective invalidation by module <br>• Size-based and age-based cleanup <br>• smart\_cacheable decorator | Use during active development to avoid stale cache issues. Essential for iterative research. |
| data\_access\_tracker.py | Prevents test set contamination | • Logs every dataset access with timestamps <br>• Tracks train/test/validate usage <br>• Generates contamination reports <br>• Detects data snooping bias <br>• Provides audit trail | Critical for research integrity. Use to track all data access during model development. |
| cv\_cache.py | Cross-validation specialized caching | • Caches CV results efficiently <br>• Handles sklearn estimators correctly <br>• Supports PurgedKFold and custom CV <br>• Separates estimator params from state <br>• Fast CV iterations | Use when running expensive cross-validation experiments. Speeds up hyperparameter optimization dramatically. |
| backtest\_cache.py | Backtesting workflow optimization | • Caches complete backtest runs <br>• Walk-forward analysis support <br>• Parameter optimization tracking <br>• Trade-level caching <br>• Result comparison tools | Essential for strategy development. Cache backtest results to compare parameter variations efficiently. |
| cache\_monitoring.py | Performance analysis and diagnostics | • Hit rate tracking per function <br>• Computation time measurement <br>• Cache size monitoring <br>• Health reports and recommendations <br>• Efficiency analysis | Use to understand cache performance and identify optimization opportunities. |
| mlflow\_integration.py | Experiment tracking integration | • Combines caching with MLflow <br>• Automatic experiment logging <br>• Model versioning <br>• Metric tracking <br>• Result comparison | Use in production research environments to track experiments while benefiting from caching. |
| mql5\_bridge.py | Python–MQL5 communication bridge | • Launches Python scripts from MQL5 <br>• File-based signaling for cross-language execution <br>• Real-time model inference support <br>• Integrates ML pipelines with MetaTrader 5 | Use when deploying Python-based ML models into MQL5 trading environments for automation. |
| startup\_script.py | Environment bootstrap and cache setup | • Initializes cache directories and logging <br>• Loads environment variables <br>• Ensures reproducible startup <br>• Can trigger MLflow or monitoring setup | Use at the beginning of any ML or trading workflow to ensure consistent and reproducible setup. |
| PythonBridgeEA.mq5 | Chart-attached MQL5 bridge to Python | • Acts as a bridge between MQL5 and Python <br>• Uses chart events and file signaling <br>• Triggers Python scripts from MetaTrader <br>• Synchronizes trading logic with Python output | Attach to a chart to enable Python communication from MetaTrader 5. Required for real-time ML integration. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20302.zip "Download all attachments in the single ZIP archive")

[PythonBridgeEA.mq5](https://www.mql5.com/en/articles/download/20302/PythonBridgeEA.mq5 "Download PythonBridgeEA.mq5")(21.22 KB)

[user\_guide.py](https://www.mql5.com/en/articles/download/20302/user_guide.py "Download user_guide.py")(22.37 KB)

[mql5\_integration\_guide.txt](https://www.mql5.com/en/articles/download/20302/mql5_integration_guide.txt "Download mql5_integration_guide.txt")(9.82 KB)

[cache.zip](https://www.mql5.com/en/articles/download/20302/cache.zip "Download cache.zip")(45.84 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MetaTrader 5 Machine Learning Blueprint (Part 5): Sequential Bootstrapping—Debiasing Labels, Improving Returns](https://www.mql5.com/en/articles/20059)
- [Machine Learning Blueprint (Part 4): The Hidden Flaw in Your Financial ML Pipeline — Label Concurrency](https://www.mql5.com/en/articles/19850)
- [MetaTrader 5 Machine Learning Blueprint (Part 3): Trend-Scanning Labeling Method](https://www.mql5.com/en/articles/19253)
- [MetaTrader 5 Machine Learning Blueprint (Part 2): Labeling Financial Data for Machine Learning](https://www.mql5.com/en/articles/18864)
- [MetaTrader 5 Machine Learning Blueprint (Part 1): Data Leakage and Timestamp Fixes](https://www.mql5.com/en/articles/17520)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/500774)**
(1)


![Kgothatso Nkuna](https://c.mql5.com/avatar/avatar_na2.png)

**[Kgothatso Nkuna](https://www.mql5.com/en/users/kgothatsonkuna675)**
\|
27 Nov 2025 at 08:45

I want to build automatically [robot forex](https://www.mql5.com/en/market/mt5/expert "Trading robots  for the MetaTrader 5 and MetaTrader 4") mt5


![Automating Trading Strategies in MQL5 (Part 42): Session-Based Opening Range Breakout (ORB) System](https://c.mql5.com/2/183/20339-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 42): Session-Based Opening Range Breakout (ORB) System](https://www.mql5.com/en/articles/20339)

In this article, we create a fully customizable session-based Opening Range Breakout (ORB) system in MQL5 that lets us set any desired session start time and range duration, automatically calculates the high and low of that opening period, and trades only confirmed breakouts in the direction of the move.

![Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://c.mql5.com/2/182/20271-market-positioning-codex-for-logo.png)[Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)

In this article, we look to explore how a complimentary indicator pairing can be used to analyze the recent 5-year history of Vanguard Information Technology Index Fund ETF. By considering two options of algorithms, Kendall’s Tau and Distance-Correlation, we look to select not just an ideal indicator pair for trading the VGT, but also suitable signal-pattern pairings of these two indicators.

![From Basic to Intermediate: Struct (I)](https://c.mql5.com/2/117/Do_b8sico_ao_intermediario_Struct_I___LOGO.png)[From Basic to Intermediate: Struct (I)](https://www.mql5.com/en/articles/15730)

Today we will begin to study structures in a simpler, more practical, and comfortable way. Structures are among the foundations of programming, whether they are structured or not. I know many people think of structures as just collections of data, but I assure you that they are much more than just structures. And here we will begin to explore this new universe in the most didactic way.

![Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://c.mql5.com/2/122/Developing_a_Multicurrency_Advisor_Part_23___LOGO_2.png)[Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

We aim to create a system for automatic periodic optimization of trading strategies used in one final EA. As the system evolves, it becomes increasingly complex, so it is necessary to look at it as a whole from time to time in order to identify bottlenecks and suboptimal solutions.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mganwscxhaviphaluacsjjysnwxqkdly&ssn=1769157125076373169&ssn_dr=1&ssn_sr=0&fv_date=1769157125&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20302&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MetaTrader%205%20Machine%20Learning%20Blueprint%20(Part%206)%3A%20Engineering%20a%20Production-Grade%20Caching%20System%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915712600168180&fz_uniq=6470814543299650565&sv=2552)

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