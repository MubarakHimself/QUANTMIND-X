---
name: clean_data_anomalies
category: data_skills
description: Handle missing data, outliers, and anomalies in OHLCV datasets
version: 1.0.0
dependencies:
  - fetch_historical_data
tags:
  - data
  - cleaning
  - quality
---

# Clean Data Anomalies Skill

## Description

Identifies and handles data quality issues including missing bars, zero values, spikes, gaps, and outliers in OHLCV datasets. Provides multiple cleaning strategies to ensure data quality for backtesting and analysis.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "description": "Array of OHLCV bars to clean"
    },
    "timeframe": {
      "type": "string",
      "description": "Timeframe of data (e.g., 'H1', 'D1')"
    },
    "cleaning_strategy": {
      "type": "string",
      "enum": ["fill", "interpolate", "forward_fill", "remove", "flag"],
      "description": "How to handle anomalies (default: fill)"
    },
    "outlier_method": {
      "type": "string",
      "enum": ["iqr", "zscore", "isolation_forest", "none"],
      "description": "Outlier detection method (default: iqr)"
    },
    "outlier_threshold": {
      "type": "number",
      "description": "Threshold for outlier detection (default: 3.0)",
      "default": 3.0
    },
    "max_gap_size": {
      "type": "integer",
      "description": "Maximum gap size to fill (default: 10 bars)",
      "default": 10
    },
    "min_volume": {
      "type": "number",
      "description": "Minimum valid volume (default: 0)"
    }
  },
  "required": ["data", "timeframe"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether cleaning was successful"
    },
    "cleaned_data": {
      "type": "array",
      "description": "Cleaned OHLCV data"
    },
    "anomalies": {
      "type": "object",
      "properties": {
        "missing_bars": {
          "type": "integer"
        },
        "zero_values": {
          "type": "integer"
        },
        "negative_values": {
          "type": "integer"
        },
        "outliers": {
          "type": "integer"
        },
        "gaps": {
          "type": "integer"
        },
        "spikes": {
          "type": "integer"
        }
      }
    },
    "cleaning_report": {
      "type": "string",
      "description": "Summary of cleaning actions"
    },
    "data_quality_score": {
      "type": "number",
      "description": "Quality score from 0 to 100"
    },
    "error": {
      "type": "string",
      "description": "Error message if unsuccessful"
    }
  }
}
```

## Code

```python
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


def clean_data_anomalies(
    data: List[Dict[str, Any]],
    timeframe: str,
    cleaning_strategy: str = "fill",
    outlier_method: str = "iqr",
    outlier_threshold: float = 3.0,
    max_gap_size: int = 10,
    min_volume: float = 0
) -> Dict[str, Any]:
    """
    Clean data anomalies in OHLCV dataset.

    Args:
        data: Array of OHLCV bars
        timeframe: Timeframe of data (e.g., 'H1', 'D1')
        cleaning_strategy: How to handle anomalies
        outlier_method: Outlier detection method
        outlier_threshold: Threshold for outlier detection
        max_gap_size: Maximum gap size to fill
        min_volume: Minimum valid volume

    Returns:
        Dictionary containing cleaned data and anomaly report
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        # Track anomalies
        anomalies = {
            "missing_bars": 0,
            "zero_values": 0,
            "negative_values": 0,
            "outliers": 0,
            "gaps": 0,
            "spikes": 0
        }

        # 1. Detect and handle missing bars
        df = _handle_missing_bars(df, timeframe, max_gap_size, cleaning_strategy)
        missing_count = len(data) - len(df)
        anomalies["missing_bars"] = missing_count

        # 2. Detect and handle zero values
        df = _handle_zero_values(df, cleaning_strategy)
        zero_count = _count_zero_values(df)
        anomalies["zero_values"] = zero_count

        # 3. Detect and handle negative values
        df = _handle_negative_values(df, cleaning_strategy)
        neg_count = _count_negative_values(df)
        anomalies["negative_values"] = neg_count

        # 4. Detect and handle outliers
        if outlier_method != "none":
            df, outliers_detected = _handle_outliers(
                df, outlier_method, outlier_threshold, cleaning_strategy
            )
            anomalies["outliers"] = outliers_detected

        # 5. Detect and handle gaps
        df, gaps_detected = _handle_gaps(df, timeframe, cleaning_strategy)
        anomalies["gaps"] = gaps_detected

        # 6. Detect and handle spikes
        df, spikes_detected = _handle_spikes(df, cleaning_strategy)
        anomalies["spikes"] = spikes_detected

        # Calculate data quality score
        quality_score = _calculate_quality_score(df, anomalies, len(data))

        # Convert back to list
        cleaned_data = []
        for idx, row in df.iterrows():
            bar = {
                "time": idx.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "tick_volume": int(row["tick_volume"])
            }
            cleaned_data.append(bar)

        # Generate cleaning report
        report = _generate_cleaning_report(anomalies, cleaning_strategy)

        return {
            "success": True,
            "cleaned_data": cleaned_data,
            "anomalies": anomalies,
            "cleaning_report": report,
            "data_quality_score": quality_score
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Cleaning error: {str(e)}"
        }


def _handle_missing_bars(
    df: pd.DataFrame,
    timeframe: str,
    max_gap_size: int,
    strategy: str
) -> pd.DataFrame:
    """Handle missing bars in dataset."""
    # Generate expected time range
    freq_map = {
        "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
        "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1w"
    }
    freq = freq_map.get(timeframe, "1h")

    # Reindex to include all expected times
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )

    df = df.reindex(full_range)

    # Handle missing values based on strategy
    if strategy == "fill":
        df["open"].fillna(method="ffill", inplace=True)
        df["high"].fillna(method="ffill", inplace=True)
        df["low"].fillna(method="ffill", inplace=True)
        df["close"].fillna(method="ffill", inplace=True)
        df["tick_volume"].fillna(0, inplace=True)
    elif strategy == "interpolate":
        df["open"].interpolate(method="time", inplace=True)
        df["high"].interpolate(method="time", inplace=True)
        df["low"].interpolate(method="time", inplace=True)
        df["close"].interpolate(method="time", inplace=True)
        df["tick_volume"].fillna(0, inplace=True)
    elif strategy == "remove":
        df = df.dropna()

    return df


def _handle_zero_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle zero values in price data."""
    price_cols = ["open", "high", "low", "close"]

    for col in price_cols:
        zero_mask = df[col] == 0

        if strategy == "fill":
            df.loc[zero_mask, col] = df[col].replace(0, np.nan).fillna(method="ffill")
        elif strategy == "interpolate":
            df.loc[zero_mask, col] = df[col].replace(0, np.nan).interpolate()
        elif strategy == "remove":
            df = df[~zero_mask]

    return df


def _handle_negative_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle negative values in price data (shouldn't exist)."""
    price_cols = ["open", "high", "low", "close"]

    for col in price_cols:
        neg_mask = df[col] < 0

        if strategy == "fill":
            df.loc[neg_mask, col] = df[col].clip(lower=0)
        elif strategy == "remove":
            df = df[~neg_mask]

    return df


def _handle_outliers(
    df: pd.DataFrame,
    method: str,
    threshold: float,
    strategy: str
) -> tuple:
    """Detect and handle outliers."""
    outliers_detected = 0

    if method == "iqr":
        # Interquartile Range method
        for col in ["open", "high", "low", "close"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_detected += outlier_mask.sum()

            if strategy == "fill":
                df.loc[outlier_mask, col] = df[col].rolling(5, center=True).mean()
            elif strategy == "remove":
                df = df[~outlier_mask]

    elif method == "zscore":
        # Z-score method
        for col in ["open", "high", "low", "close"]:
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outlier_mask = z_scores > threshold
                outliers_detected += outlier_mask.sum()

                if strategy == "fill":
                    df.loc[outlier_mask, col] = mean
                elif strategy == "remove":
                    df = df[~outlier_mask]

    return df, outliers_detected


def _handle_gaps(df: pd.DataFrame, timeframe: str, strategy: str) -> tuple:
    """Detect and handle time gaps in data."""
    gaps_detected = 0

    # Calculate time differences
    time_diffs = df.index.to_series().diff()

    # Define expected gap based on timeframe
    expected_gaps = {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1)
    }

    expected_gap = expected_gaps.get(timeframe, timedelta(hours=1))

    # Find gaps larger than expected
    large_gaps = time_diffs > expected_gap * 1.5
    gaps_detected = large_gaps.sum()

    if strategy == "fill" and gaps_detected > 0:
        # Already handled in _handle_missing_bars
        pass

    return df, gaps_detected


def _handle_spikes(df: pd.DataFrame, strategy: str) -> tuple:
    """Detect and handle price spikes."""
    spikes_detected = 0

    # Calculate price changes
    for col in ["open", "high", "low", "close"]:
        pct_change = df[col].pct_change()

        # Define spike as >10% change in one bar
        spike_threshold = 0.10
        spike_mask = pct_change.abs() > spike_threshold
        spikes_detected += spike_mask.sum()

        if strategy == "fill":
            # Smooth spikes using rolling mean
            df.loc[spike_mask, col] = df[col].rolling(3, center=True).mean()

    return df, spikes_detected


def _count_zero_values(df: pd.DataFrame) -> int:
    """Count zero values in price columns."""
    price_cols = ["open", "high", "low", "close"]
    return (df[price_cols] == 0).sum().sum()


def _count_negative_values(df: pd.DataFrame) -> int:
    """Count negative values in price columns."""
    price_cols = ["open", "high", "low", "close"]
    return (df[price_cols] < 0).sum().sum()


def _calculate_quality_score(
    df: pd.DataFrame,
    anomalies: Dict[str, int],
    original_count: int
) -> float:
    """Calculate data quality score (0-100)."""
    if len(df) == 0:
        return 0.0

    # Base score
    score = 100.0

    # Deduct points for anomalies
    score -= anomalies["missing_bars"] * 2
    score -= anomalies["zero_values"] * 1
    score -= anomalies["negative_values"] * 5
    score -= anomalies["outliers"] * 3
    score -= anomalies["gaps"] * 2
    score -= anomalies["spikes"] * 1

    # Ensure score is between 0 and 100
    return max(0.0, min(100.0, score))


def _generate_cleaning_report(anomalies: Dict[str, int], strategy: str) -> str:
    """Generate human-readable cleaning report."""
    report_parts = [
        f"Data Cleaning Report (Strategy: {strategy})",
        "="*40
    ]

    for anomaly_type, count in anomalies.items():
        if count > 0:
            report_parts.append(f"- {anomaly_type.replace('_', ' ').title()}: {count}")

    if not any(anomalies.values()):
        report_parts.append("- No anomalies detected. Data is clean!")

    return "\n".join(report_parts)


def detect_data_quality_issues(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze data for quality issues without cleaning.

    Args:
        data: Array of OHLCV bars

    Returns:
        Dictionary with quality analysis
    """
    df = pd.DataFrame(data)

    issues = {
        "duplicate_times": df["time"].duplicated().sum(),
        "null_values": df.isnull().sum().to_dict(),
        "constant_prices": (df["close"].std() == 0),
        "monotonic_prices": df["close"].is_monotonic_increasing or df["close"].is_monotonic_decreasing,
        "volume_zeros": (df["tick_volume"] == 0).sum()
    }

    return issues
```

## Example Usage

```python
# Example 1: Basic data cleaning
data = fetch_historical_data(symbol="EURUSD", timeframe="H1", count=1000)

result = clean_data_anomalies(
    data=data["data"],
    timeframe="H1",
    cleaning_strategy="fill"
)

print(f"Quality score: {result['data_quality_score']}")
print(f"\nCleaning report:\n{result['cleaning_report']}")

# Example 2: Aggressive outlier removal
result = clean_data_anomalies(
    data=data["data"],
    timeframe="H1",
    cleaning_strategy="remove",
    outlier_method="zscore",
    outlier_threshold=2.5
)

print(f"Original bars: {len(data['data'])}")
print(f"Cleaned bars: {len(result['cleaned_data'])}")

# Example 3: Just flag issues, don't modify
result = clean_data_anomalies(
    data=data["data"],
    timeframe="H1",
    cleaning_strategy="flag",
    outlier_method="iqr"
)

# Review anomalies before deciding on cleaning
for anomaly_type, count in result['anomalies'].items():
    if count > 0:
        print(f"{anomaly_type}: {count} issues found")

# Example 4: Analyze quality before cleaning
quality_issues = detect_data_quality_issues(data["data"])

print("Data Quality Analysis:")
for issue, value in quality_issues.items():
    print(f"  {issue}: {value}")

# Example 5: Clean and validate for backtesting
result = clean_data_anomalies(
    data=data["data"],
    timeframe="H1",
    cleaning_strategy="interpolate",
    outlier_method="iqr",
    max_gap_size=5  # Only fill small gaps
)

if result["data_quality_score"] >= 80:
    print("Data quality acceptable for backtesting")
    # Proceed with backtest
else:
    print("Data quality poor. Consider different data source or parameters")

# Example 6: Compare cleaning strategies
strategies = ["fill", "interpolate", "forward_fill"]

for strategy in strategies:
    result = clean_data_anomalies(
        data=data["data"],
        timeframe="H1",
        cleaning_strategy=strategy
    )

    print(f"{strategy}: Quality score {result['data_quality_score']:.1f}")

# Example 7: Clean with custom threshold
result = clean_data_anomalies(
    data=data["data"],
    timeframe="D1",
    cleaning_strategy="fill",
    outlier_method="iqr",
    outlier_threshold=1.5  # More sensitive
)

print(f"Outliers detected: {result['anomalies']['outliers']}")
```

## Notes

- Different strategies suit different use cases
- `fill`: Good for visual charts, but may introduce bias
- `interpolate`: Better for maintaining price continuity
- `remove`: Safest for backtesting, but reduces data
- `flag`: Review issues before deciding on action

## Cleaning Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `fill` | Fill with previous values | Visualization |
| `interpolate` | Interpolate between values | Continuous data |
| `forward_fill` | Forward fill only | Simple gaps |
| `remove` | Remove problematic bars | Backtesting |
| `flag` | Flag issues only | Review before action |

## Outlier Detection Methods

| Method | Description | Sensitivity |
|--------|-------------|-------------|
| `iqr` | Interquartile range | Medium |
| `zscore` | Standard score | High |
| `isolation_forest` | ML-based | Very High |
| `none` | Skip outlier detection | N/A |

## Data Quality Score Interpretation

| Score Range | Quality | Action |
|-------------|---------|--------|
| 90-100 | Excellent | Use as-is |
| 70-89 | Good | Minor cleaning may help |
| 50-69 | Fair | Requires cleaning |
| 0-49 | Poor | Significant issues, consider different data |

## Dependencies

- `fetch_historical_data`: Source data for cleaning
- pandas for DataFrame operations

## See Also

- `fetch_historical_data`: Get source data
- `resample_timeframe`: Convert after cleaning
