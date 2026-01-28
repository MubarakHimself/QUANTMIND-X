---
title: Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator
url: https://www.mql5.com/en/articles/20455
categories: Integration, Strategy Tester
relevance_score: 6
scraped_at: 2026-01-23T11:54:12.777582
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/20455&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062802144865724722)

MetaTrader 5 / Tester


**Contents**

- [Introduction](https://www.mql5.com/en/articles/20455#intro)
- [Handling MetaTrader 5 Historical Ticks](https://www.mql5.com/en/articles/20455#handling-ticks)
- [Handling MetaTrader 5 Historical Bars](https://www.mql5.com/en/articles/20455#handling-bars)
- [Overloading MetaTrader 5 Functions](https://www.mql5.com/en/articles/20455#overlaoding-MT5-functions)

  - [symbol\_info\_tick](https://www.mql5.com/en/articles/20455#symbol_info_tick)
  - [symbol\_info](https://www.mql5.com/en/articles/20455#symbol_info)
  - [copy\_rates\_from](https://www.mql5.com/en/articles/20455#copy_rates_from)
  - [copy\_rates\_from\_pos](https://www.mql5.com/en/articles/20455#copy_rates_from_pos)
  - [copy\_rates\_range](https://www.mql5.com/en/articles/20455#copy_rates_range)
  - [copy\_ticks\_from](https://www.mql5.com/en/articles/20455#copy_ticks_from)
  - [copy\_ticks\_range](https://www.mql5.com/en/articles/20455#copy_ticks_range)
  - [orders\_total](https://www.mql5.com/en/articles/20455#orders_total)
  - [orders\_get](https://www.mql5.com/en/articles/20455#orders_get)
  - [positions\_total](https://www.mql5.com/en/articles/20455#positions_total)
  - [positions\_get](https://www.mql5.com/en/articles/20455#positions_get)
  - [history\_orders\_total](https://www.mql5.com/en/articles/20455#history_orders_total)
  - [history\_orders\_get](https://www.mql5.com/en/articles/20455#history_orders_get)
  - [deals\_total](https://www.mql5.com/en/articles/20455#deals_total)
  - [history\_deals\_get](https://www.mql5.com/en/articles/20455#history_deals_get)
  - [account\_info](https://www.mql5.com/en/articles/20455#account_info)
  - [order\_calc\_profit](https://www.mql5.com/en/articles/20455#order_calc_profit)
  - [order\_calc\_margin](https://www.mql5.com/en/articles/20455#order_calc_margin)

- [Final Thoughts](https://www.mql5.com/en/articles/20455#final%20thoughts)

### Introduction

In the [previous article](https://www.mql5.com/en/articles/18971), we discussed and made a simulator class in Python called TradeSimulator, which relied heavily on information from MetaTrader 5, such as ticks, bar data, symbol information, and much more.

The first article laid the foundation for what's required in imitating the MetaTrader 5 client, and its strategy tester (simulator). In this article, we will introduce ticks and bars data, as well as functions similar to those provided by the Python-MetaTrader 5 module in the simulator, taking a step closer to replicating everything that MetaTrader 5 does and provides.

### Handling MetaTrader 5 Historical Ticks

Ticks are the most granular, real-time price updates for a financial instrument, representing every individual price change, bid/ask movement, and trade volume.

Unlike OHLC bars (Open, High, Low, Close), ticks provide millisecond-level data.

You might be familiar with the function called [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) from the MQL5 programming language. _(The main function for MQL5 bots that gets called upon the arrival of a new tick)._

The MetaTrader 5 terminal relies heavily on tick data when opening, monitoring, and closing trades. No ticks, no operations on this platform.

That being said, we need to be able to get and handle ticks similarly to how the terminal does it.

The Python-MetaTrader 5 module provides various ways of getting ticks; one of the ways is by using a function called  **copy\_ticks\_range**:

```
copy_ticks_range(
   symbol,       // symbol name
   date_from,    // date the ticks are requested from
   date_to,      // date, up to which the ticks are requested
   flags         // combination of flags defining the type of requested ticks
   )
```

Let's attempt to collect ticks data from MetaTrader 5.

```
def fetch_ticks(start_datetime: datetime, end_datetime: datetime, symbol: str):

    ticks = mt5.copy_ticks_range(symbol, start_datetime, end_datetime, mt5.COPY_TICKS_ALL)

    print(f"Fetched {len(ticks)} ticks for {symbol} from {start_datetime} to {end_datetime}")
    print(ticks[:5])  # Print first 5 ticks for inspection

    return ticks
```

Example.

```
import MetaTrader5 as mt5
from datetime import datetime, timezone

if __name__ == "__main__":
    if not mt5.initialize():
        print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
        mt5.shutdown()
        quit()

    symbol = "EURUSD"
    start_dt = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2025, 12, 1, 1, 0, tzinfo=timezone.utc)

    fetch_ticks(start_dt, end_dt, symbol)
```

Outputs.

```
Fetched 2814462 ticks for EURUSD from 2025-01-01 00:00:00+00:00 to 2025-12-01 01:00:00+00:00
[(1758499200, 1.17403, 1.17603, 0., 0, 1758499200161, 134, 0.)\
 (1758499247, 1.17405, 1.17605, 0., 0, 1758499247468, 134, 0.)\
 (1758499500, 1.17346, 1.17546, 0., 0, 1758499500116, 134, 0.)\
 (1758499505, 1.173  , 1.175  , 0., 0, 1758499505869, 134, 0.)\
 (1758499510, 1.17307, 1.17487, 0., 0, 1758499510079, 134, 0.)]
```

As you can see, just for 11 months we were able to obtain 2.8 million tick records. We might as well check its size in megabytes (this should give us a rough estimate of how much memory (RAM) is consumed by this single tick request).

```
    # calculate tick array size in megabytes
    size_in_bytes = ticks.nbytes
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"Tick array size: {size_in_mb:.2f} MB")
```

Outputs.

```
Tick array size: 161.04 MB
```

As you can see, just 11 months of data is worth around 0.1 GB. Now, imagine in our simulator (strategy tester) a user decides to test a multicurrency bot with 12 symbols throughout 20 years, how much would that cost our memory and the overall performance?

We have to find the best approach for handling this much data without consuming too much memory and having a decent overall performance.

_[Polars](https://www.mql5.com/go?link=https://pola.rs/ "https://pola.rs/")  DataFrames are one of the best solutions for situations like this._

Polars is easy to use and very fast; its streaming API allows developers to process large datasets (datasets larger than memory e.g,. 100GB+) in a very efficient way.

As we will no longer use numpy arrays for the entire data storage, we also have to split the data collection process into smaller, less memory-intensive tick data chunks.

```
def ticks_to_polars(ticks):
    return pl.DataFrame({
        "time": ticks["time"],
        "bid": ticks["bid"],
        "ask": ticks["ask"],
        "last": ticks["last"],
        "volume": ticks["volume"],
        "time_msc": ticks["time_msc"],
        "flags": ticks["flags"],
        "volume_real": ticks["volume_real"],
    })

def fetch_historical_ticks(start_datetime: datetime,
                           end_datetime: datetime,
                           symbol: str):

    # first of all, we have to ensure the symbol is valid and can be used for requesting data
    if not utils.ensure_symbol(symbol=symbol):
        print(f"Symbol {symbol} not available")
        return

    current = start_datetime.replace(day=1, hour=0, minute=0, second=0)

    while True:
        month_start, month_end = utils.month_bounds(current)

        # Cap last month to end_date
        if (
            month_start.year == end_datetime.year and
            month_start.month == end_datetime.month
        ):
            month_end = end_datetime

        # Stop condition
        if month_start > end_datetime:
            break

        print(f"Processing ticks {month_start:%Y-%m-%d} -> {month_end:%Y-%m-%d}")

        # --- fetch data here ---
        ticks = mt5.copy_ticks_range(
            symbol,
            month_start,
            month_end,
            mt5.COPY_TICKS_ALL
        )

        if ticks is None or len(ticks) == 0:

            config.simulator_logger.critical(f"Failed to Get ticks. Error = {mt5.last_error()}")
            current = (month_start + timedelta(days=32)).replace(day=1) # Advance to next month safely

            continue

        df = ticks_to_polars(ticks)

        df = df.with_columns([\
            pl.from_epoch("time", time_unit="s").dt.replace_time_zone("utc").alias("time")\
        ])

        df = df.with_columns([\
            pl.col("time").dt.year().alias("year"),\
            pl.col("time").dt.month().alias("month"),\
        ])

        df.write_parquet(
            os.path.join(config.TICKS_HISTORY_DIR, symbol),
            partition_by=["year", "month"],
            mkdir=True
        )

        if config.debug:
           print(df.head(-10))

        # Advance to next month safely
        current = (month_start + timedelta(days=32)).replace(day=1)
```

So, instead of collecting all ticks at once using copy\_ticks\_range, we iteratively collect ticks for every month and store the information into separate files.

```
      df.write_parquet(
            os.path.join(config.TICKS_HISTORY_DIR, symbol),
            partition_by=["year", "month"],
            mkdir=True
        )
```

Let's print to see what the DataFrame object holds.

```
print(df.head(-10))  # optional, see what data looks like
```

Outputs.

```
2025-12-24 16:41:44,138 | CRITICAL | simulator.log20251224 | fetch_historical_ticks 52 --> Failed to Get ticks. Error = (1, 'Success')
Processing ticks 2025-07-01 -> 2025-07-31
2025-12-24 16:41:44,139 | CRITICAL | simulator.log20251224 | fetch_historical_ticks 52 --> Failed to Get ticks. Error = (1, 'Success')
Processing ticks 2025-08-01 -> 2025-08-31
2025-12-24 16:41:44,140 | CRITICAL | simulator.log20251224 | fetch_historical_ticks 52 --> Failed to Get ticks. Error = (1, 'Success')
Processing ticks 2025-09-01 -> 2025-09-30
shape: (434_916, 10)
┌─────────────────────────┬─────────┬─────────┬──────┬───┬───────┬─────────────┬──────┬───────┐
│ time                    ┆ bid     ┆ ask     ┆ last ┆ … ┆ flags ┆ volume_real ┆ year ┆ month │
│ ---                     ┆ ---     ┆ ---     ┆ ---  ┆   ┆ ---   ┆ ---         ┆ ---  ┆ ---   │
│ datetime[μs, UTC]       ┆ f64     ┆ f64     ┆ f64  ┆   ┆ u32   ┆ f64         ┆ i32  ┆ i8    │
╞═════════════════════════╪═════════╪═════════╪══════╪═══╪═══════╪═════════════╪══════╪═══════╡
│ 2025-09-22 00:00:00 UTC ┆ 1.17403 ┆ 1.17603 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 9     │
│ 2025-09-22 00:00:47 UTC ┆ 1.17405 ┆ 1.17605 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 9     │
│ 2025-09-22 00:05:00 UTC ┆ 1.17346 ┆ 1.17546 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 9     │
│ 2025-09-22 00:05:05 UTC ┆ 1.173   ┆ 1.175   ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 9     │
│ 2025-09-22 00:05:10 UTC ┆ 1.17307 ┆ 1.17487 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 9     │
│ …                       ┆ …       ┆ …       ┆ …    ┆ … ┆ …     ┆ …           ┆ …    ┆ …     │
│ 2025-09-30 23:58:44 UTC ┆ 1.17335 ┆ 1.17343 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 9     │
│ 2025-09-30 23:58:45 UTC ┆ 1.17335 ┆ 1.17342 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 9     │
│ 2025-09-30 23:58:46 UTC ┆ 1.17335 ┆ 1.17343 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 9     │
│ 2025-09-30 23:58:47 UTC ┆ 1.17335 ┆ 1.17342 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 9     │
│ 2025-09-30 23:58:50 UTC ┆ 1.17334 ┆ 1.1734  ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 9     │
└─────────────────────────┴─────────┴─────────┴──────┴───┴───────┴─────────────┴──────┴───────┘
Processing ticks 2025-10-01 -> 2025-10-31
shape: (1_401_674, 10)
┌─────────────────────────┬─────────┬─────────┬──────┬───┬───────┬─────────────┬──────┬───────┐
│ time                    ┆ bid     ┆ ask     ┆ last ┆ … ┆ flags ┆ volume_real ┆ year ┆ month │
│ 2025-10-01 00:00:01 UTC ┆ 1.17337 ┆ 1.17506 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 10    │
│ 2025-10-01 00:00:02 UTC ┆ 1.17337 ┆ 1.17402 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 10    │
│ 2025-10-01 00:00:02 UTC ┆ 1.17337 ┆ 1.17389 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 10    │
│ …                       ┆ …       ┆ …       ┆ …    ┆ … ┆ …     ┆ …           ┆ …    ┆ …     │
│ 2025-10-31 23:56:43 UTC ┆ 1.15368 ┆ 1.15368 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 10    │
│ 2025-10-31 23:56:52 UTC ┆ 1.15369 ┆ 1.15369 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 10    │
│ 2025-10-31 23:56:52 UTC ┆ 1.15371 ┆ 1.15371 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 10    │
│ 2025-10-31 23:56:53 UTC ┆ 1.1537  ┆ 1.1537  ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 10    │
│ 2025-10-31 23:56:53 UTC ┆ 1.15371 ┆ 1.15371 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 10    │
└─────────────────────────┴─────────┴─────────┴──────┴───┴───────┴─────────────┴──────┴───────┘
Processing ticks 2025-11-01 -> 2025-11-30
shape: (976_714, 10)
┌─────────────────────────┬─────────┬─────────┬──────┬───┬───────┬─────────────┬──────┬───────┐
│ time                    ┆ bid     ┆ ask     ┆ last ┆ … ┆ flags ┆ volume_real ┆ year ┆ month │
│ ---                     ┆ ---     ┆ ---     ┆ ---  ┆   ┆ ---   ┆ ---         ┆ ---  ┆ ---   │
│ datetime[μs, UTC]       ┆ f64     ┆ f64     ┆ f64  ┆   ┆ u32   ┆ f64         ┆ i32  ┆ i8    │
╞═════════════════════════╪═════════╪═════════╪══════╪═══╪═══════╪═════════════╪══════╪═══════╡
│ 2025-11-03 00:00:00 UTC ┆ 1.1528  ┆ 1.15365 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 11    │
│ 2025-11-03 00:01:00 UTC ┆ 1.1528  ┆ 1.15365 ┆ 0.0  ┆ … ┆ 130   ┆ 0.0         ┆ 2025 ┆ 11    │
│ 2025-11-03 00:01:00 UTC ┆ 1.1528  ┆ 1.15365 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 11    │
│ 2025-11-03 00:01:21 UTC ┆ 1.15295 ┆ 1.15365 ┆ 0.0  ┆ … ┆ 130   ┆ 0.0         ┆ 2025 ┆ 11    │
│ 2025-11-03 00:01:25 UTC ┆ 1.15282 ┆ 1.15365 ┆ 0.0  ┆ … ┆ 130   ┆ 0.0         ┆ 2025 ┆ 11    │
│ …                       ┆ …       ┆ …       ┆ …    ┆ … ┆ …     ┆ …           ┆ …    ┆ …     │
│ 2025-11-28 23:55:12 UTC ┆ 1.15948 ┆ 1.16018 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 11    │
│ 2025-11-28 23:55:13 UTC ┆ 1.15955 ┆ 1.16017 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 11    │
│ 2025-11-28 23:55:36 UTC ┆ 1.15948 ┆ 1.16018 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 11    │
│ 2025-11-28 23:55:37 UTC ┆ 1.15953 ┆ 1.16017 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 11    │
│ 2025-11-28 23:55:54 UTC ┆ 1.15954 ┆ 1.16024 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 11    │
│ time                    ┆ bid     ┆ ask     ┆ last ┆ … ┆ flags ┆ volume_real ┆ year ┆ month │
│ ---                     ┆ ---     ┆ ---     ┆ ---  ┆   ┆ ---   ┆ ---         ┆ ---  ┆ ---   │
│ datetime[μs, UTC]       ┆ f64     ┆ f64     ┆ f64  ┆   ┆ u32   ┆ f64         ┆ i32  ┆ i8    │
╞═════════════════════════╪═════════╪═════════╪══════╪═══╪═══════╪═════════════╪══════╪═══════╡
│ 2025-12-01 00:00:00 UTC ┆ 1.15936 ┆ 1.15969 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 12    │
│ 2025-12-01 00:00:06 UTC ┆ 1.15934 ┆ 1.15962 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 12    │
│ 2025-12-01 00:00:11 UTC ┆ 1.15935 ┆ 1.15997 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 12    │
│ 2025-12-01 00:00:15 UTC ┆ 1.15936 ┆ 1.15979 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 12    │
│ 2025-12-01 00:00:21 UTC ┆ 1.15936 ┆ 1.15964 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 12    │
│ …                       ┆ …       ┆ …       ┆ …    ┆ … ┆ …     ┆ …           ┆ …    ┆ …     │
│ 2025-12-01 00:59:57 UTC ┆ 1.15964 ┆ 1.16005 ┆ 0.0  ┆ … ┆ 4     ┆ 0.0         ┆ 2025 ┆ 12    │
│ 2025-12-01 00:59:57 UTC ┆ 1.15972 ┆ 1.16012 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 12    │
│ 2025-12-01 00:59:57 UTC ┆ 1.15967 ┆ 1.16005 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 12    │
│ 2025-12-01 00:59:57 UTC ┆ 1.15971 ┆ 1.16009 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 12    │
│ 2025-12-01 00:59:57 UTC ┆ 1.15965 ┆ 1.16005 ┆ 0.0  ┆ … ┆ 134   ┆ 0.0         ┆ 2025 ┆ 12    │
└─────────────────────────┴─────────┴─────────┴──────┴───┴───────┴─────────────┴──────┴───────┘
January 2024:
 shape: (0, 10)
┌───────────────────┬─────┬─────┬──────┬───┬───────┬─────────────┬──────┬───────┐
│ time              ┆ bid ┆ ask ┆ last ┆ … ┆ flags ┆ volume_real ┆ year ┆ month │
│ ---               ┆ --- ┆ --- ┆ ---  ┆   ┆ ---   ┆ ---         ┆ ---  ┆ ---   │
│ datetime[μs, UTC] ┆ f64 ┆ f64 ┆ f64  ┆   ┆ u32   ┆ f64         ┆ i32  ┆ i8    │
╞═══════════════════╪═════╪═════╪══════╪═══╪═══════╪═════════════╪══════╪═══════╡
└───────────────────┴─────┴─────┴──────┴───┴───────┴─────────────┴──────┴───────┘
shape: (1, 2)
┌───────────────────┬───────────────────┐
│ time_min          ┆ time_max          │
│ ---               ┆ ---               │
│ datetime[μs, UTC] ┆ datetime[μs, UTC] │
╞═══════════════════╪═══════════════════╡
│ null              ┆ null              │
└───────────────────┴───────────────────┘
```

One of the coolest things about a Polars method named write\_parquet is that, when given some value in the argument partition\_by it uses the received columns as groups and stores data in separate subfolders.

After tick collection from two instruments.

```
if __name__ == "__main__":

    if not mt5.initialize():
        print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
        mt5.shutdown()
        quit()

    symbol = "EURUSD"
    start_dt = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2025, 12, 1, 1, 0, tzinfo=timezone.utc)

    fetch_historical_ticks(start_datetime=start_dt, end_datetime=end_dt, symbol=symbol)
    fetch_historical_ticks(start_datetime=start_dt, end_datetime=end_dt, symbol= "GBPUSD")

    path = os.path.join(config.TICKS_HISTORY_DIR, symbol)
    lf = pl.scan_parquet(path)

    jan_2024 = (
        lf
        .filter(
            (pl.col("year") == 2024) &
            (pl.col("month") == 1)
        )
        .collect(engine="streaming")
    )

    print("January 2024:\n", jan_2024.head(-10))
    print(
        jan_2024.select([\
            pl.col("time").min().alias("time_min"),\
            pl.col("time").max().alias("time_max")\
        ])
    )

    mt5.shutdown()
```

Below is what the output folders look like.

```
(venv) c:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>tree History
Folder PATH listing
Volume serial number is 2CFE-3A78
C:\USERS\OMEGA JOCTAN\ONEDRIVE\DOCUMENTS\PYMETATESTER\HISTORY
├───Bars
│   ├───EURUSD
│   │   └───M5
└───Ticks
    ├───EURUSD
    │   └───year=2025
    │       ├───month=10
    │       ├───month=11
    │       ├───month=12
    │       └───month=9
    └───GBPUSD
        └───year=2025
            ├───month=10
            ├───month=11
            ├───month=12
            └───month=9
```

Unfortunately, I couldn't get all the tick data as I requested (from January 1st to December 1st, in 2025). It appears that you cannot get more ticks than what's available in your MetaTrader 5 terminal. In this case, my broker had only a few months' worth of tick data (and that is what I kept getting).

_From: C:\\Users\\Omega\\AppData\\Roaming\\MetaQuotes\\Terminal\\010E047102812FC0C18890992854220E\\bases\\<broker name>\\ticks\\EURUSD_

![](https://c.mql5.com/2/187/267129752455.png)

### Handling MetaTrader 5 Historical Bars

Unlike ticks, bars are timeframe based. It is easier to work with bars than ticks. Similarly to how we collected ticks, we have to collect bar data similarly.

Firstly, we have to ensure that the symbol is available and select it in the MarketWatch before requesting its bars.

_Inside utils.py_

```
def ensure_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Symbol {symbol} not found")
        return False

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol {symbol}")
            return False
    return True
```

We then collect data starting from the first to the last day of the month.

```
def fetch_historical_bars(symbol: str,
                          timeframe: int,
                          start_datetime: datetime,
                          end_datetime: datetime):
    """
    Fetch historical bar data for a given symbol and timeframe, forward in time.
    Saves data to a single Parquet file in append mode.
    """

    if not utils.ensure_symbol(symbol=symbol):
        print(f"Symbol {symbol} not available")
        return

    current = start_datetime.replace(day=1, hour=0, minute=0, second=0)

    while True:
        month_start, month_end = utils.month_bounds(current)

        # Cap last month to end_date
        if (
            month_start.year == end_datetime.year and
            month_start.month == end_datetime.month
        ):
            month_end = end_datetime

        # Stop condition
        if month_start > end_datetime:
            break

        print(f"Processing {month_start:%Y-%m-%d} -> {month_end:%Y-%m-%d}")

        # --- fetch data here ---
        rates = mt5.copy_rates_range(
            symbol,
            timeframe,
            month_start,
            month_end
        )

        if rates is None and len(rates)==0:
            config.simulator_logger.warning(f"Failed to Get bars from MetaTrader5")
            current = (month_start + timedelta(days=32)).replace(day=1) # Advance to next month safely
            continue

        df = bars_to_polars(rates)
```

We store bars data inside their respective parquet files, separated by months and years (as subfolders).

```
df = df.with_columns([\
    pl.from_epoch("time", time_unit="s").dt.replace_time_zone("utc").alias("time")\
])

df = df.with_columns([\
    pl.col("time").dt.year().alias("year"),\
    pl.col("time").dt.month().alias("month"),\
])

tf_name = utils.TIMEFRAMES_REV[timeframe]
df.write_parquet(
    os.path.join(config.BARS_HISTORY_DIR, symbol, tf_name),
    partition_by=["year", "month"],
    mkdir=True
)

if config.is_debug:
    print(df.head(-10))

# Advance to next month safely
current = (month_start + timedelta(days=32)).replace(day=1)
```

For example, bars collected from three symbols during 10 months.

```
if __name__ == "__main__":

    if not mt5.initialize():
        print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
        mt5.shutdown()
        quit()

    start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 10, tzinfo=timezone.utc)

    fetch_historical_bars("XAUUSD", mt5.TIMEFRAME_M1, start_date, end_date)
    fetch_historical_bars("EURUSD", mt5.TIMEFRAME_H1, start_date, end_date)
    fetch_historical_bars("GBPUSD", mt5.TIMEFRAME_M5, start_date, end_date)

    # read polaris dataframe and print the head for both symbols

    symbol = "GBPUSD"
    timeframe = utils.TIMEFRAMES_REV[mt5.TIMEFRAME_M5]

    path = os.path.join(config.BARS_HISTORY_DIR, symbol, timeframe)

    lf = pl.scan_parquet(path)

    jan_2024 = (
        lf
        .filter(
            (pl.col("year") == 2024) &
            (pl.col("month") == 1)
        )
        .collect(engine="streaming")
    )

    print("January 2024:\n", jan_2024.head(-10))
    print(
        jan_2024.select([\
            pl.col("time").min().alias("time_min"),\
            pl.col("time").max().alias("time_max")\
        ])
    )

    mt5.shutdown()
```

Outputs.

```
(venv) c:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>tree History
Folder PATH listing
Volume serial number is 2CFE-3A78
C:\USERS\OMEGA JOCTAN\ONEDRIVE\DOCUMENTS\PYMETATESTER\HISTORY
├───Bars
│   ├───EURUSD
│   │   ├───H1
│   │   │   ├───year=2022
│   │   │   │   ├───month=1
│   │   │   │   ├───month=10
│   │   │   │   ├───month=11
│   │   │   │   ├───month=12
│   │   │   │   ├───month=2
│   │   │   │   ├───month=3
│   │   │   │   ├───month=4
│   │   │   │   ├───month=5
│   │   │   │   ├───month=6
│   │   │   │   ├───month=7
│   │   │   │   ├───month=8
│   │   │   │   └───month=9
│   │   │   ├───year=2023
│   │   │   │   ├───month=1
│   │   │   │   ├───month=10
│   │   │   │   ├───month=11
│   │   │   │   ├───month=12
│   │   │   │   ├───month=2
│   │   │   │   ├───month=3
│   │   │   │   ├───month=4
│   │   │   │   ├───month=5
│   │   │   │   ├───month=6
│   │   │   │   ├───month=7
│   │   │   │   ├───month=8
│   │   │   │   └───month=9
│   │   │   ├───year=2024
│   │   │   │   ├───month=1
│   │   │   │   ├───month=10
│   │   │   │   ├───month=11
│   │   │   │   ├───month=12
│   │   │   │   ├───month=2
│   │   │   │   ├───month=3
│   │   │   │   ├───month=4
│   │   │   │   ├───month=5
│   │   │   │   ├───month=6
│   │   │   │   ├───month=7
│   │   │   │   ├───month=8
│   │   │   │   └───month=9
│   │   │   └───year=2025
│   │   │       └───month=1
│   │   └───M5
│   │       ├───year=2022
│   │       │   ├───month=1
│   │       │   ├───month=10
│   │       │   ├───month=11
│   │       │   ├───month=12
│   │       │   ├───month=2
│   │       │   ├───month=3
│   │       │   ├───month=4
│   │       │   ├───month=5
│   │       │   ├───month=6
│   │       │   ├───month=7
│   │       │   ├───month=8
│   │       │   └───month=9
│   │       ├───year=2023
│   │       │   ├───month=1
│   │       │   ├───month=10
│   │       │   ├───month=11
│   │       │   ├───month=12
│   │       │   ├───month=2
│   │       │   ├───month=3
│   │       │   ├───month=4
│   │       │   ├───month=5
│   │       │   ├───month=6
│   │       │   ├───month=7
│   │       │   ├───month=8
│   │       │   └───month=9
│   │       ├───year=2024
│   │       │   ├───month=1
│   │       │   ├───month=10
│   │       │   ├───month=11
│   │       │   ├───month=12
│   │       │   ├───month=2
│   │       │   ├───month=3
│   │       │   ├───month=4
│   │       │   ├───month=5
│   │       │   ├───month=6
│   │       │   ├───month=7
│   │       │   ├───month=8
│   │       │   └───month=9
│   │       └───year=2025
│   │           └───month=1
└───Ticks
    ├───EURUSD
    │   └───year=2025
    │       ├───month=10
    │       ├───month=11
    │       ├───month=12
    │       └───month=9
    └───GBPUSD
        └───year=2025
            ├───month=10
            ├───month=11
            ├───month=12
            └───month=9
```

### Overloading MetaTrader 5 Functions

Again in the previous article, we were able to simulate some trading operations despite relying too much on MetaTrader 5 for ticks, rates, and some of the crucial details. This time, we want to have a fully or even close to a completely isolated custom simulator.

Firstly, we're going to add a tester instance, meaning, if a user starts a simulator with an argument IS\_TESTER set to False (the strategy tester mode), instead of extracting crucial information such as rates and ticks directly from MetaTrader 5, we extract such information the custom paths (created in the previous sections).

We do the opposite when IS\_TESTER is set to false, by extracting such data directly from MetaTrader 5.

```
class Simulator:
    def __init__(self, simulator_name: str, mt5_instance: mt5, deposit: float, leverage: str="1:100"):

        #... other variables

        self.IS_RUNNING = True # is the simulator running or stopped
        self.IS_TESTER = True # are we on the strategy tester mode or live trading

        self.symbol_info_cache: dict[str, namedtuple] = {}

    def Start(self, IS_TESTER: bool) -> bool: # simulator start

        self.IS_TESTER = IS_TESTER

    def Stop(self): # simulator stopped
        self.IS_RUNNING = False
        pass
```

**symbol\_info\_tick**

Now that we have our way of storing and reading tick data from a nearby path, we need a way of returning such information to the user, just like the way the MetaTrader 5 client does it.

```
symbol_info_tick(
   symbol      // financial instrument name
)
```

We need a similar function inside the Simulator class. A function has to decide whether to return ticks from MetaTrader 5 or ticks within a simulator.

```
    def symbol_info_tick(self, symbol: str) -> namedtuple:

        if self.IS_TESTER:
            return self.tick_cache[symbol]

        try:
            tick = self.mt5_instance.symbol_info_tick(symbol)
        except Exception as e:
            self.__GetLogger().warning(f"Failed. MT5 Error = {self.mt5_instance.last_error()}")

        return tick
```

Inside a simulator class, we have an array for keeping track of recent ticks.

_Under a class constructor:_

```
self.tick_cache: dict[str, namedtuple] = {}
```

However, this simulator needs to be fed this tick ininformation, we need a function for such a task.

```
def TickUpdate(self, symbol: str, tick: namedtuple):
    self.tick_cache[symbol] = tick
```

**symbol\_info**

This function gets data from the MetaTrader 5 platform on a specified financial instrument.

[Function signature.](https://www.mql5.com/en/docs/python_metatrader5/mt5symbolinfo_py)

```
symbol_info(
   symbol      // financial instrument name
)
```

We need a similar function in our simulator class, but it shouldn't request this data from MetaTrader 5 more than once in the life of a simulator.

After extracting a symbol's data from MetaTrader 5, it must be the values within an array for later usage (this reduces "MetaTrader 5-dependence" and improves the overall performance).

```
def symbol_info(self, symbol: str) -> namedtuple:

    """Gets data on the specified financial instrument."""

    if symbol not in self.symbol_info_cache:
        info = self.mt5_instance.symbol_info(symbol)
        if info is None:
           return None

       self.symbol_info_cache[symbol] = info

   return self.symbol_info_cache[symbol]
```

An array for temporarily storing symbols' data is defined similarly to the one responsible for storing ticks _discussed above_.

```
self.symbol_info_cache: dict[str, namedtuple] = {}
```

**copy\_rates\_from**

This function gets bars from the MetaTrader 5 terminal, starting from the specified date to some given prior bars.

```
copy_rates_from(
   symbol,       // symbol name
   timeframe,    // timeframe
   date_from,    // initial bar open date
   count         // number of bars
   )
```

In a similar function in our class, we start by ensuring that a given starting date is in UTC format.

```
def copy_rates_from(self, symbol: str, timeframe: int, date_from: datetime, count: int) -> np.array:

    date_from = utils.ensure_utc(date_from)
```

If a user has chosen the strategy tester mode (IS\_TESTER=True), we get bars data stored in parquet files.

```
if self.IS_TESTER:

    # instead of getting data from MetaTrader 5, get data stored in our custom directories

    path = os.path.join(config.BARS_HISTORY_DIR, symbol, utils.TIMEFRAMES_REV[timeframe])
    lf = pl.scan_parquet(path)

    try:
        rates = (
            lf
            .filter(pl.col("time") <= date_from) # get data starting at the given date
            .sort("time", descending=True)
            .limit(count) # limit the request to some bars
            .select([\
                pl.col("time").dt.epoch("s").cast(pl.Int64).alias("time"),\
\
                pl.col("open"),\
                pl.col("high"),\
                pl.col("low"),\
                pl.col("close"),\
                pl.col("tick_volume"),\
                pl.col("spread"),\
                pl.col("real_volume"),\
            ]) # return only what's required
            .collect(engine="streaming") # the streming engine, doesn't store data in memory
        ).to_dicts()

        rates = np.array(rates)[::-1] # reverse an array so it becomes oldest -> newest

    except Exception as e:
        config.tester_logger.warn(f"Failed to copy rates {e}")
        return np.array(dict())
else:

    rates = self.mt5_instance.copy_rates_from(symbol, timeframe, date_from, count)
    rates = np.array(self.__mt5_rates_to_dicts(rates))

    if rates is None:
        config.simulator_logger.warn(f"Failed to copy rates. MetaTrader 5 error = {self.mt5_instance.last_error()}")
        return np.array(dict())

return rates
```

_If the variable within a class IS\_TESTER is set to False, we get bars data directly from MetaTrader 5._

Since MetaTrader 5 returns a structured numpy array, let us convert it into a numpy array with dictionaries of data for every element in the array. This makes it consistent with the format received after the conversion is performed on a [Polars DataFrame](https://www.mql5.com/go?link=https://docs.pola.rs/py-polars/html/reference/dataframe/index.html "https://docs.pola.rs/py-polars/html/reference/dataframe/index.html") object.

```
def __mt5_rates_to_dicts(self, rates) -> list[dict]:

    if rates is None or len(rates) == 0:
        return []

    # structured numpy array from MT5
    if rates.dtype.names is not None:
        return [\
            {name: r[name].item() if hasattr(r[name], "item") else r[name]\
            for name in rates.dtype.names}\
            for r in rates\
        ]
```

Example usage:

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

start = datetime(2025, 1, 1)
bars = 10

sim.Start(IS_TESTER=True) # start the simulator in the strategy tester mode
rates = sim.copy_rates_from(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, date_from=start, count=bars)
print("is_tester=true\n", rates)

sim.Start(IS_TESTER=False) # start the simulator in real-time trading
rates = sim.copy_rates_from(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, date_from=start, count=bars)

print("is_tester=false\n",rates)
```

Outputs.

```
is_tester=true
 [{'time': 1735653600, 'open': 1.04104, 'high': 1.04145, 'low': 1.03913, 'close': 1.03928, 'tick_volume': 2543, 'spread': 0, 'real_volume': 0}\
 {'time': 1735657200, 'open': 1.03929, 'high': 1.03973, 'low': 1.03836, 'close': 1.0393, 'tick_volume': 3171, 'spread': 0, 'real_volume': 0}\
 {'time': 1735660800, 'open': 1.03931, 'high': 1.03943, 'low': 1.03748, 'close': 1.03759, 'tick_volume': 4073, 'spread': 0, 'real_volume': 0}\
 {'time': 1735664400, 'open': 1.03759, 'high': 1.03893, 'low': 1.03527, 'close': 1.03548, 'tick_volume': 5531, 'spread': 0, 'real_volume': 0}\
 {'time': 1735668000, 'open': 1.03548, 'high': 1.03614, 'low': 1.0346899999999999, 'close': 1.03504, 'tick_volume': 3918, 'spread': 0, 'real_volume': 0}\
 {'time': 1735671600, 'open': 1.03504, 'high': 1.03551, 'low': 1.03442, 'close': 1.03493, 'tick_volume': 3279, 'spread': 0, 'real_volume': 0}\
 {'time': 1735675200, 'open': 1.0348600000000001, 'high': 1.03569, 'low': 1.03455, 'close': 1.0352999999999999, 'tick_volume': 2693, 'spread': 0, 'real_volume': 0}\
 {'time': 1735678800, 'open': 1.0352999999999999, 'high': 1.03647, 'low': 1.03516, 'close': 1.03548, 'tick_volume': 1840, 'spread': 0, 'real_volume': 0}\
 {'time': 1735682400, 'open': 1.03549, 'high': 1.03633, 'low': 1.03546, 'close': 1.03586, 'tick_volume': 1192, 'spread': 0, 'real_volume': 0}\
 {'time': 1735686000, 'open': 1.03586, 'high': 1.0361, 'low': 1.03527, 'close': 1.03527, 'tick_volume': 975, 'spread': 0, 'real_volume': 0}]
is_tester=false
 [{'time': 1735653600, 'open': 1.04104, 'high': 1.04145, 'low': 1.03913, 'close': 1.03928, 'tick_volume': 2543, 'spread': 0, 'real_volume': 0}\
 {'time': 1735657200, 'open': 1.03929, 'high': 1.03973, 'low': 1.03836, 'close': 1.0393, 'tick_volume': 3171, 'spread': 0, 'real_volume': 0}\
 {'time': 1735660800, 'open': 1.03931, 'high': 1.03943, 'low': 1.03748, 'close': 1.03759, 'tick_volume': 4073, 'spread': 0, 'real_volume': 0}\
 {'time': 1735664400, 'open': 1.03759, 'high': 1.03893, 'low': 1.03527, 'close': 1.03548, 'tick_volume': 5531, 'spread': 0, 'real_volume': 0}\
 {'time': 1735668000, 'open': 1.03548, 'high': 1.03614, 'low': 1.0346899999999999, 'close': 1.03504, 'tick_volume': 3918, 'spread': 0, 'real_volume': 0}\
 {'time': 1735671600, 'open': 1.03504, 'high': 1.03551, 'low': 1.03442, 'close': 1.03493, 'tick_volume': 3279, 'spread': 0, 'real_volume': 0}\
 {'time': 1735675200, 'open': 1.0348600000000001, 'high': 1.03569, 'low': 1.03455, 'close': 1.0352999999999999, 'tick_volume': 2693, 'spread': 0, 'real_volume': 0}\
 {'time': 1735678800, 'open': 1.0352999999999999, 'high': 1.03647, 'low': 1.03516, 'close': 1.03548, 'tick_volume': 1840, 'spread': 0, 'real_volume': 0}\
 {'time': 1735682400, 'open': 1.03549, 'high': 1.03633, 'low': 1.03546, 'close': 1.03586, 'tick_volume': 1192, 'spread': 0, 'real_volume': 0}\
 {'time': 1735686000, 'open': 1.03586, 'high': 1.0361, 'low': 1.03527, 'close': 1.03527, 'tick_volume': 975, 'spread': 0, 'real_volume': 0}]
```

**copy\_rates\_from\_pos**

[According to the docs](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrompos_py), this function gets bars from the MetaTrader 5 terminal starting from a specified index.

_At index 0 lies the current bar, with the bar at the largest index being the oldest bar in the Terminal._

This is the trickiest of all functions that copies bars information from MetaTrader 5, simply because, _it is time-aware_.

Since the bar at index 0 is always the current bar, it means that the current function should be aware of the currenttick time. When running a simulator in the so-called strategy tester, we inherit the function **copy\_rates\_from**, which takes time input for the starting date.

We give it the starting date of:

current time + The current timeframe in seconds \* number of bars requested by the user.

```
    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> np.array:

        if self.tick is None or self.tick.time is None:
            self.__GetLogger().critical("Time information not found in the ticker, call the function 'TickUpdate' giving it the latest tick information")
            now = datetime.now(tz=timezone.utc)
        else:
            now = self.tick.time

        if self.IS_TESTER:
            rates = self.copy_rates_from(symbol=symbol,
                                        timeframe=timeframe,
                                        date_from=now+timedelta(seconds=utils.PeriodSeconds(timeframe)*start_pos),
                                        count=count)

        else:

            rates = self.mt5_instance.copy_rates_from_pos(symbol, timeframe, start_pos, count)
            rates = np.array(self.__mt5_rates_to_dicts(rates))

            if rates is None:
                self.__GetLogger().warning(f"Failed to copy rates. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return np.array(dict())

        return rates

```

Whenever IS\_TESTER=False (running the system in real time), the simulator gets bars directly from the MetaTrader 5 terminal.

Example usage:

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

start = datetime(2025, 1, 1)
bars = 10
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H1

sim.Start(IS_TESTER=True)
rates = sim.copy_rates_from_pos(symbol=symbol, timeframe=timeframe, start_pos=0, count=bars)

print("is_tester=true\n", rates)

sim.Start(IS_TESTER=False) # start the simulator in real-time trading
rates = sim.copy_rates_from_pos(symbol=symbol, timeframe=timeframe, start_pos=0, count=bars)

print("is_tester=false\n",rates)
```

Outputs.

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py
2025-12-25 12:42:33,366 | CRITICAL | tester |  copy_rates_from_pos 221 --> Time information not found in the ticker, call the function 'TickUpdate' giving it the latest tick information
is_tester=true
 [{'time': 1766584800, 'open': 1.17927, 'high': 1.17932, 'low': 1.1784, 'close': 1.17843, 'tick_volume': 1983, 'spread': 0, 'real_volume': 0}\
 {'time': 1766588400, 'open': 1.17843, 'high': 1.17909, 'low': 1.17838, 'close': 1.17853, 'tick_volume': 2783, 'spread': 0, 'real_volume': 0}\
 {'time': 1766592000, 'open': 1.17849, 'high': 1.17869, 'low': 1.17773, 'close': 1.17807, 'tick_volume': 2690, 'spread': 0, 'real_volume': 0}\
 {'time': 1766595600, 'open': 1.17804, 'high': 1.17825, 'low': 1.17754, 'close': 1.17781, 'tick_volume': 2834, 'spread': 0, 'real_volume': 0}\
 {'time': 1766599200, 'open': 1.17781, 'high': 1.1781, 'low': 1.17732, 'close': 1.17795, 'tick_volume': 2354, 'spread': 0, 'real_volume': 0}\
 {'time': 1766602800, 'open': 1.17794, 'high': 1.17832, 'low': 1.17726, 'close': 1.17766, 'tick_volume': 1424, 'spread': 0, 'real_volume': 0}\
 {'time': 1766606400, 'open': 1.17764, 'high': 1.17798, 'low': 1.17744, 'close': 1.17788, 'tick_volume': 1105, 'spread': 0, 'real_volume': 0}\
 {'time': 1766610000, 'open': 1.17788, 'high': 1.1782, 'low': 1.17787, 'close': 1.17817, 'tick_volume': 654, 'spread': 0, 'real_volume': 0}\
 {'time': 1766613600, 'open': 1.17817, 'high': 1.17819, 'low': 1.1779, 'close': 1.1779600000000001, 'tick_volume': 608, 'spread': 0, 'real_volume': 0}\
 {'time': 1766617200, 'open': 1.1779600000000001, 'high': 1.17797, 'low': 1.17761, 'close': 1.17768, 'tick_volume': 1165, 'spread': 0, 'real_volume': 0}]
2025-12-25 12:42:33,394 | CRITICAL | simulator |  copy_rates_from_pos 221 --> Time information not found in the ticker, call the function 'TickUpdate' giving it the latest tick information
is_tester=false
 [{'time': 1766584800, 'open': 1.17927, 'high': 1.17932, 'low': 1.1784, 'close': 1.17843, 'tick_volume': 1983, 'spread': 0, 'real_volume': 0}\
 {'time': 1766588400, 'open': 1.17843, 'high': 1.17909, 'low': 1.17838, 'close': 1.17853, 'tick_volume': 2783, 'spread': 0, 'real_volume': 0}\
 {'time': 1766592000, 'open': 1.17849, 'high': 1.17869, 'low': 1.17773, 'close': 1.17807, 'tick_volume': 2690, 'spread': 0, 'real_volume': 0}\
 {'time': 1766595600, 'open': 1.17804, 'high': 1.17825, 'low': 1.17754, 'close': 1.17781, 'tick_volume': 2834, 'spread': 0, 'real_volume': 0}\
 {'time': 1766599200, 'open': 1.17781, 'high': 1.1781, 'low': 1.17732, 'close': 1.17795, 'tick_volume': 2354, 'spread': 0, 'real_volume': 0}\
 {'time': 1766602800, 'open': 1.17794, 'high': 1.17832, 'low': 1.17726, 'close': 1.17766, 'tick_volume': 1424, 'spread': 0, 'real_volume': 0}\
 {'time': 1766606400, 'open': 1.17764, 'high': 1.17798, 'low': 1.17744, 'close': 1.17788, 'tick_volume': 1105, 'spread': 0, 'real_volume': 0}\
 {'time': 1766610000, 'open': 1.17788, 'high': 1.1782, 'low': 1.17787, 'close': 1.17817, 'tick_volume': 654, 'spread': 0, 'real_volume': 0}\
 {'time': 1766613600, 'open': 1.17817, 'high': 1.17819, 'low': 1.1779, 'close': 1.1779600000000001, 'tick_volume': 608, 'spread': 0, 'real_volume': 0}\
 {'time': 1766617200, 'open': 1.1779600000000001, 'high': 1.17797, 'low': 1.17761, 'close': 1.17768, 'tick_volume': 1165, 'spread': 0, 'real_volume': 0}]
```

**copy\_rates\_range**

This function gets bars in the specified date range from the MetaTrader 5 terminal.

```
copy_rates_range(
   symbol,       // symbol name
   timeframe,    // timeframe
   date_from,    // date the bars are requested from
   date_to       // date, up to which the bars are requested
   )
```

Unlike the prior two, this one returns bars between two dates (date\_from), the starting date, and (date\_to), an end date.

```
    def copy_rates_range(self, symbol: str, timeframe: int, date_from: datetime, date_to: datetime):

        date_from = utils.ensure_utc(date_from)
        date_to = utils.ensure_utc(date_to)

        if self.IS_TESTER:

            # instead of getting data from MetaTrader 5, get data stored in our custom directories

            path = os.path.join(config.BARS_HISTORY_DIR, symbol, utils.TIMEFRAMES_REV[timeframe])
            lf = pl.scan_parquet(path)

            try:
                rates = (
                    lf
                    .filter(
                            (pl.col("time") >= pl.lit(date_from)) &
                            (pl.col("time") <= pl.lit(date_to))
                        ) # get bars between date_from and date_to
                    .sort("time", descending=True)
                    .select([\
                        pl.col("time").dt.epoch("s").cast(pl.Int64).alias("time"),\
\
                        pl.col("open"),\
                        pl.col("high"),\
                        pl.col("low"),\
                        pl.col("close"),\
                        pl.col("tick_volume"),\
                        pl.col("spread"),\
                        pl.col("real_volume"),\
                    ]) # return only what's required
                    .collect(engine="streaming") # the streming engine, doesn't store data in memory
                ).to_dicts()

                rates = np.array(rates)[::-1] # reverse an array so it becomes oldest -> newest

            except Exception as e:
                self.__GetLogger().warning(f"Failed to copy rates {e}")
                return np.array(dict())
        else:

            rates = self.mt5_instance.copy_rates_range(symbol, timeframe, date_from, date_to)
            rates = np.array(self.__mt5_rates_to_dicts(rates))

            if rates is None:
                self.__GetLogger().warning(f"Failed to copy rates. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return np.array(dict())

        return rates
```

**copy\_ticks\_from**

_[According to the documentation](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksfrom_py)_, this function gets ticks from the MetaTrader 5 terminal starting from a specified date.

```
copy_ticks_from(
   symbol,       // symbol name
   date_from,    // date the ticks are requested from
   count,        // number of requested ticks
   flags         // combination of flags defining the type of requested ticks
   )
```

Inside a similar function in our simulator class, we read ticks from our database whenever a user has selected a strategy tester mode (IS\_TESTER=True), and read them directly from MetaTrader 5, in contrast.

```
    def copy_ticks_from(self, symbol: str, date_from: datetime, count: int, flags: int=mt5.COPY_TICKS_ALL) -> np.array:

        date_from = utils.ensure_utc(date_from)
        flag_mask = self.__tick_flag_mask(flags)

        if self.IS_TESTER:

            path = os.path.join(config.TICKS_HISTORY_DIR, symbol)
            lf = pl.scan_parquet(path)

            try:
                ticks = (
                    lf
                    .filter(pl.col("time") >= pl.lit(date_from)) # get data starting at the given date
                    .filter((pl.col("flags") & flag_mask) != 0)
                    .sort(
                        ["time", "time_msc"],
                        descending=[False, False]
                    )
                    .limit(count) # limit the request to a specified number of ticks
                    .select([\
                        pl.col("time").dt.epoch("s").cast(pl.Int64).alias("time"),\
\
                        pl.col("bid"),\
                        pl.col("ask"),\
                        pl.col("last"),\
                        pl.col("volume"),\
                        pl.col("time_msc"),\
                        pl.col("flags"),\
                        pl.col("volume_real"),\
                    ])
                    .collect(engine="streaming") # the streming engine, doesn't store data in memory
                ).to_dicts()

                ticks = np.array(ticks)

            except Exception as e:
                self.__GetLogger().warning(f"Failed to copy ticks {e}")
                return np.array(dict())
        else:

            ticks = self.mt5_instance.copy_ticks_from(symbol, date_from, count, flags)
            ticks = np.array(self.__mt5_data_to_dicts(ticks))

            if ticks is None:
                self.__GetLogger().warning(f"Failed to copy ticks. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return np.array(dict())

        return ticks
```

Since a tick request comes with a flags option to let users decide the kind of ticks they want to get, we need a way to create a flags mask, useful for filtering ticks depending on what a user needs.

_According to the documentation:_

A flag defines the type of the requested ticks.

Flag values are described in the [COPY\_TICKS](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksfrom_py#copy_ticks) enumeration.

| ID | Description |
| --- | --- |
| COPY\_TICKS\_ALL | all ticks |
| COPY\_TICKS\_INFO | ticks containing Bid and/or Ask price changes |
| COPY\_TICKS\_TRADE | ticks containing Last and/or Volume price changes |

TICK\_FLAG defines possible flags for ticks. These flags are used to describe ticks obtained by the [copy\_ticks\_from()](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksfrom_py) and [copy\_ticks\_range()](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py) functions.

| ID | Description |
| --- | --- |
| TICK\_FLAG\_BID | Bid price changed |
| TICK\_FLAG\_ASK | Ask price changed |
| TICK\_FLAG\_LAST | Last price changed |
| TICK\_FLAG\_VOLUME | Volume changed |
| TICK\_FLAG\_BUY | last Buy price changed |
| TICK\_FLAG\_SELL | last Sell price changed |

```
    def __tick_flag_mask(self, flags: int) -> int:
        if flags == mt5.COPY_TICKS_ALL:
            return (
                mt5.TICK_FLAG_BID
                | mt5.TICK_FLAG_ASK
                | mt5.TICK_FLAG_LAST
                | mt5.TICK_FLAG_VOLUME
                | mt5.TICK_FLAG_BUY
                | mt5.TICK_FLAG_SELL
            )

        mask = 0
        if flags & mt5.COPY_TICKS_INFO:
            mask |= mt5.TICK_FLAG_BID | mt5.TICK_FLAG_ASK
        if flags & mt5.COPY_TICKS_TRADE:
            mask |= mt5.TICK_FLAG_LAST | mt5.TICK_FLAG_VOLUME

        return mask
```

Example usage:

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

start = datetime(2025, 1, 1)
end = datetime(2025, 1, 5)

bars = 10
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H1

sim.Start(IS_TESTER=True) # start simulation in the strategy tester

ticks = sim.copy_ticks_from(symbol=symbol, date_from=start.replace(month=12, hour=0, minute=0), count=bars)

print("is_tester=true\n", ticks)

sim.Start(IS_TESTER=False) # start the simulator in real-time trading

ticks = sim.copy_ticks_from(symbol=symbol, date_from=start.replace(month=12, hour=0, minute=0), count=bars)

print("is_tester=false\n", ticks)
```

Outputs.

```
is_tester=true
 [{'time': 1764547200, 'bid': 1.15936, 'ask': 1.1596899999999999, 'last': 0.0, 'volume': 0, 'time_msc': 1764547200174, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547206, 'bid': 1.15934, 'ask': 1.15962, 'last': 0.0, 'volume': 0, 'time_msc': 1764547206476, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547211, 'bid': 1.1593499999999999, 'ask': 1.15997, 'last': 0.0, 'volume': 0, 'time_msc': 1764547211273, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547215, 'bid': 1.15936, 'ask': 1.15979, 'last': 0.0, 'volume': 0, 'time_msc': 1764547215872, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547221, 'bid': 1.15936, 'ask': 1.15964, 'last': 0.0, 'volume': 0, 'time_msc': 1764547221475, 'flags': 4, 'volume_real': 0.0}\
 {'time': 1764547231, 'bid': 1.1593499999999999, 'ask': 1.15997, 'last': 0.0, 'volume': 0, 'time_msc': 1764547231674, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547260, 'bid': 1.1593499999999999, 'ask': 1.15997, 'last': 0.0, 'volume': 0, 'time_msc': 1764547260073, 'flags': 130, 'volume_real': 0.0}\
 {'time': 1764547265, 'bid': 1.15892, 'ask': 1.15998, 'last': 0.0, 'volume': 0, 'time_msc': 1764547265485, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547320, 'bid': 1.15892, 'ask': 1.15998, 'last': 0.0, 'volume': 0, 'time_msc': 1764547320074, 'flags': 130, 'volume_real': 0.0}\
 {'time': 1764547345, 'bid': 1.15894, 'ask': 1.15998, 'last': 0.0, 'volume': 0, 'time_msc': 1764547345872, 'flags': 134, 'volume_real': 0.0}]
is_tester=false
 [{'time': 1764547200, 'bid': 1.15936, 'ask': 1.1596899999999999, 'last': 0.0, 'volume': 0, 'time_msc': 1764547200174, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547206, 'bid': 1.15934, 'ask': 1.15962, 'last': 0.0, 'volume': 0, 'time_msc': 1764547206476, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547211, 'bid': 1.1593499999999999, 'ask': 1.15997, 'last': 0.0, 'volume': 0, 'time_msc': 1764547211273, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547215, 'bid': 1.15936, 'ask': 1.15979, 'last': 0.0, 'volume': 0, 'time_msc': 1764547215872, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547221, 'bid': 1.15936, 'ask': 1.15964, 'last': 0.0, 'volume': 0, 'time_msc': 1764547221475, 'flags': 4, 'volume_real': 0.0}\
 {'time': 1764547231, 'bid': 1.1593499999999999, 'ask': 1.15997, 'last': 0.0, 'volume': 0, 'time_msc': 1764547231674, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547260, 'bid': 1.1593499999999999, 'ask': 1.15997, 'last': 0.0, 'volume': 0, 'time_msc': 1764547260073, 'flags': 130, 'volume_real': 0.0}\
 {'time': 1764547265, 'bid': 1.15892, 'ask': 1.15998, 'last': 0.0, 'volume': 0, 'time_msc': 1764547265485, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547320, 'bid': 1.15892, 'ask': 1.15998, 'last': 0.0, 'volume': 0, 'time_msc': 1764547320074, 'flags': 130, 'volume_real': 0.0}\
 {'time': 1764547345, 'bid': 1.15894, 'ask': 1.15998, 'last': 0.0, 'volume': 0, 'time_msc': 1764547345872, 'flags': 134, 'volume_real': 0.0}]
```

**copy\_ticks\_range**

_[According to the documentation,](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py)_ this function gets ticks for the specified date range from the MetaTrader 5 terminal.

Function signature.

```
copy_ticks_range(
   symbol,       // symbol name
   date_from,    // date the ticks are requested from
   date_to,      // date, up to which the ticks are requested
   flags         // combination of flags defining the type of requested ticks
   )
```

Below is a similar implementation of the function inside the class Simulator.

```
     def copy_ticks_range(self, symbol: str, date_from: datetime, date_to: datetime, flags: int=mt5.COPY_TICKS_ALL) -> np.array:

        date_from = utils.ensure_utc(date_from)
        date_to = utils.ensure_utc(date_to)

        flag_mask = self.__tick_flag_mask(flags)

        if self.IS_TESTER:

            path = os.path.join(config.TICKS_HISTORY_DIR, symbol)
            lf = pl.scan_parquet(path)

            try:
                ticks = (
                    lf
                    .filter(
                            (pl.col("time") >= pl.lit(date_from)) &
                            (pl.col("time") <= pl.lit(date_to))
                        ) # get ticks between date_from and date_to
                    .filter((pl.col("flags") & flag_mask) != 0)
                    .sort(
                        ["time", "time_msc"],
                        descending=[False, False]
                    )
                    .select([\
                        pl.col("time").dt.epoch("s").cast(pl.Int64).alias("time"),\
\
                        pl.col("bid"),\
                        pl.col("ask"),\
                        pl.col("last"),\
                        pl.col("volume"),\
                        pl.col("time_msc"),\
                        pl.col("flags"),\
                        pl.col("volume_real"),\
                    ])
                    .collect(engine="streaming") # the streaming engine, doesn't store data in memory
                ).to_dicts()

                ticks = np.array(ticks)

            except Exception as e:
                self.__GetLogger().warning(f"Failed to copy ticks {e}")
                return np.array(dict())
        else:

            ticks = self.mt5_instance.copy_ticks_range(symbol, date_from, date_to, flags)
            ticks = np.array(self.__mt5_data_to_dicts(ticks))

            if ticks is None:
                self.__GetLogger().warning(f"Failed to copy ticks. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return np.array(dict())

        return ticks
```

Example usage:

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

sim.Start(IS_TESTER=True) # start simulation in the strategy tester

ticks = sim.copy_ticks_range(symbol=symbol,
                             date_from=start.replace(month=12, hour=0, minute=0),
                             date_to=end.replace(month=12, hour=0, minute=5))

print("is_tester=true\n", ticks)

sim.Start(IS_TESTER=False) # start the simulator in real-time trading

ticks = sim.copy_ticks_range(symbol=symbol,
                             date_from=start.replace(month=12, hour=0, minute=0),
                             date_to=end.replace(month=12, hour=0, minute=5))

print("is_tester=false\n", ticks)
```

Outputs.

```
is_tester=true
 [{'time': 1764547200, 'bid': 1.15936, 'ask': 1.1596899999999999, 'last': 0.0, 'volume': 0, 'time_msc': 1764547200174, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547206, 'bid': 1.15934, 'ask': 1.15962, 'last': 0.0, 'volume': 0, 'time_msc': 1764547206476, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547211, 'bid': 1.1593499999999999, 'ask': 1.15997, 'last': 0.0, 'volume': 0, 'time_msc': 1764547211273, 'flags': 134, 'volume_real': 0.0}\
 ...\
 {'time': 1764550799, 'bid': 1.15965, 'ask': 1.16006, 'last': 0.0, 'volume': 0, 'time_msc': 1764550799475, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764550799, 'bid': 1.15971, 'ask': 1.16011, 'last': 0.0, 'volume': 0, 'time_msc': 1764550799669, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764550799, 'bid': 1.15965, 'ask': 1.16006, 'last': 0.0, 'volume': 0, 'time_msc': 1764550799877, 'flags': 134, 'volume_real': 0.0}]
is_tester=false
 [{'time': 1764547200, 'bid': 1.15936, 'ask': 1.1596899999999999, 'last': 0.0, 'volume': 0, 'time_msc': 1764547200174, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547206, 'bid': 1.15934, 'ask': 1.15962, 'last': 0.0, 'volume': 0, 'time_msc': 1764547206476, 'flags': 134, 'volume_real': 0.0}\
 {'time': 1764547211, 'bid': 1.1593499999999999, 'ask': 1.15997, 'last': 0.0, 'volume': 0, 'time_msc': 1764547211273, 'flags': 134, 'volume_real': 0.0}\
 ...\
 {'time': 1764893040, 'bid': 1.16424, 'ask': 1.16479, 'last': 0.0, 'volume': 0, 'time_msc': 1764893040071, 'flags': 130, 'volume_real': 0.0}\
 {'time': 1764893061, 'bid': 1.16424, 'ask': 1.16479, 'last': 0.0, 'volume': 0, 'time_msc': 1764893061887, 'flags': 4, 'volume_real': 0.0}\
 {'time': 1764893096, 'bid': 1.16424, 'ask': 1.16482, 'last': 0.0, 'volume': 0, 'time_msc': 1764893096077, 'flags': 4, 'volume_real': 0.0}]
```

In the previous article, we had custom functions for retrieving information about opened positions, orders, deals, etc. This time, we will overload all of them with the [Python-MetaTrader5 module](https://www.mql5.com/en/docs/python_metatrader5) syntax.

**orders\_total**

[_According to the documentation,_](https://www.mql5.com/en/docs/python_metatrader5/mt5orderstotal_py) this function gets the number of active orders from the MetaTrader 5 terminal.

```
orders_total()
```

It returns an integer value.

If a simulator is running in the strategy tester, the function returns the number of orders stored in a simulated orders container; otherwise, it returns orders from the MetaTrader 5 client.

```
def orders_total(self) -> int:

    """Get the number of active orders.

    Returns (int): The number of active orders in either a simulator or MetaTrader 5
        """

    return len(self.orders_container) if self.IS_TESTER else self.mt5_instance.orders_total()
```

**orders\_get**

[_According to the documentation,_](https://www.mql5.com/en/docs/python_metatrader5/mt5ordersget_py) this function gets active orders with the ability to filter by symbol or ticket. There are three call options.

```
orders_get()
```

Call specifying a symbol active orders should be received for.

```
orders_get(
   symbol="SYMBOL"      // symbol name
)
```

Call specifying a group of symbols active orders should be received for.

```
orders_get(
   group="GROUP"        // filter for selecting orders for symbols
)
```

Call specifying the order ticket.

```
orders_get(
   ticket=TICKET        // ticket
)
```

This function returns info in the form of a named tuple structure (namedtuple). Return None in case of an error. The info on the error can be obtained using [last\_error()](https://www.mql5.com/en/docs/python_metatrader5/mt5lasterror_py).

For our simulator to be as close as the MetaTrader 5 terminal, we have to return a similar data type (namedtuple).

```
from collections import namedtuple
```

We can define the equivalent function in our simulator as follows:

```
def orders_get(self, symbol: Optional[str] = None, group: Optional[str] = None, ticket: Optional[int] = None) -> namedtuple:
    """G et active orders with the ability to filter by symbol or ticket. There are three call options.

    Returns:

        list: Returns info in the form of a named tuple structure (namedtuple). Return None in case of an error. The info on the error can be obtained using last_error().
    """
```

Not only do we have to return a so-called namedtuple, but we also must have similar contents for such a data type.

```
    def __init__(self, simulator_name: str, mt5_instance: mt5, deposit: float, leverage: str="1:100"):

        # ----------------- TradeOrder --------------------------

        self.TradeOrder = namedtuple(
            "TradeOrder",
            [\
                "ticket",\
                "time_setup",\
                "time_setup_msc",\
                "time_done",\
                "time_done_msc",\
                "time_expiration",\
                "type",\
                "type_time",\
                "type_filling",\
                "state",\
                "magic",\
                "position_id",\
                "position_by_id",\
                "reason",\
                "volume_initial",\
                "volume_current",\
                "price_open",\
                "sl",\
                "tp",\
                "price_current",\
                "price_stoplimit",\
                "symbol",\
                "comment",\
                "external_id",\
            ]
        )
```

Below is a similar function in our Simulator class.

```
    def orders_get(self, symbol: Optional[str] = None, group: Optional[str] = None, ticket: Optional[int] = None) -> namedtuple:

        self.__orders_container__.extend([order1, order2])

        if self.IS_TESTER:

            orders = self.__orders_container__

            # no filters → return all orders
            if symbol is None and group is None and ticket is None:
                return tuple(orders)

            # symbol filter (highest priority)
            if symbol is not None:
                return tuple(o for o in orders if o.symbol == symbol)

            # group filter
            if group is not None:
                return tuple(o for o in orders if fnmatch.fnmatch(o.symbol, group))

            # ticket filter
            if ticket is not None:
                return tuple(o for o in orders if o.ticket == ticket)

            return tuple()

        try:
            if symbol is not None:
                return self.mt5_instance.orders_get(symbol=symbol)

            if group is not None:
                return self.mt5_instance.orders_get(group=group)

            if ticket is not None:
                return self.mt5_instance.orders_get(ticket=ticket)

            return self.mt5_instance.orders_get()

        except Exception:
            return None
```

If a user selects the strategy tester mode (IS\_TESTER=true), we obtain orders and their information(s) from within a Simulator; otherwise, we extract them from the MetaTrader 5 terminal.

With two pending orders in my MetaTrader 5 terminal:

![](https://c.mql5.com/2/187/orders_in_MT5.gif)

And two simulated trades:

```
        order1 = self.TradeOrder(
            ticket=123456,
            time_setup=int(datetime.now().timestamp()),
            time_setup_msc=int(datetime.now().timestamp() * 1000),
            time_done=0,
            time_done_msc=0,
            time_expiration=0,
            type=mt5.ORDER_TYPE_BUY_LIMIT,
            type_time=0,
            type_filling=mt5.ORDER_FILLING_RETURN,
            state=mt5.ORDER_STATE_PLACED,
            magic=0,
            position_id=0,
            position_by_id=0,
            reason=0,
            volume_initial=0.01,
            volume_current=0.01,
            price_open=1.1750,
            sl=1.1700,
            tp=1.1800,
            price_current=1.1750,
            price_stoplimit=0.0,
            symbol="GBPUSD",
            comment="",
            external_id="",
        )

        order2 = self.TradeOrder(
            ticket=123457,
            time_setup=int(datetime.now().timestamp()),
            time_setup_msc=int(datetime.now().timestamp() * 1000),
            time_done=0,
            time_done_msc=0,
            time_expiration=0,
            type=mt5.ORDER_TYPE_SELL_LIMIT,
            type_time=0,
            type_filling=mt5.ORDER_FILLING_RETURN,
            state=mt5.ORDER_STATE_PLACED,
            magic=0,
            position_id=0,
            position_by_id=0,
            reason=0,
            volume_initial=0.01,
            volume_current=0.01,
            price_open=1.1800,
            sl=1.1850,
            tp=1.1700,
            price_current=1.1800,
            price_stoplimit=0.0,
            symbol="EURUSD",
            comment="",
            external_id="",
        )

        self.__orders_container__.extend([order1, order2])
```

We then check for the existence of orders in both MetaTrader 5 and the Simulator.

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

sim.Start(IS_TESTER=True) # start simulation in the strategy tester
print("Orders in the simulator:\n", sim.orders_get())

sim.Start(IS_TESTER=False) # start the simulator in real-time trading
print("Orders in MetaTrader 5:\n", sim.orders_get())
```

Outputs:

```
Orders in the simulator:
 (TradeOrder(ticket=123456, time_setup=1766749779, time_setup_msc=1766749779726, time_done=0, time_done_msc=0, time_expiration=0, type=2, type_time=0, type_filling=2, state=1, magic=0, position_id=0, position_by_id=0, reason=0, volume_initial=0.01, volume_current=0.01, price_open=1.175, sl=1.17, tp=1.18, price_current=1.175, price_stoplimit=0.0, symbol='GBPUSD', comment='', external_id=''),
 TradeOrder(ticket=123457, time_setup=1766749779, time_setup_msc=1766749779726, time_done=0, time_done_msc=0, time_expiration=0, type=3, type_time=0, type_filling=2, state=1, magic=0, position_id=0, position_by_id=0, reason=0, volume_initial=0.01, volume_current=0.01, price_open=1.18, sl=1.185, tp=1.17, price_current=1.18, price_stoplimit=0.0, symbol='EURUSD', comment='', external_id=''))
Orders in MetaTrader 5:
 (TradeOrder(ticket=1381968725, time_setup=1766748043, time_setup_msc=1766748043247, time_done=0, time_done_msc=0, time_expiration=0, type=2, type_time=0, type_filling=2, state=1, magic=0, position_id=0, position_by_id=0, reason=0, volume_initial=0.01, volume_current=0.01, price_open=1.17414, sl=0.0, tp=0.0, price_current=1.17769, price_stoplimit=0.0, symbol='EURUSD', comment='', external_id=''),
TradeOrder(ticket=1381968767, time_setup=1766748049, time_setup_msc=1766748049051, time_done=0, time_done_msc=0, time_expiration=0, type=3, type_time=0, type_filling=2, state=1, magic=0, position_id=0, position_by_id=0, reason=0, volume_initial=0.01, volume_current=0.01, price_open=1.17949, sl=0.0, tp=0.0, price_current=1.17769, price_stoplimit=0.0, symbol='EURUSD', comment='', external_id=''))
```

**positions\_total**

[_According to the documentation_](https://www.mql5.com/en/docs/python_metatrader5/mt5positionstotal_py), this function returns the number of open positions in the MetaTrader 5 client.

```
positions_total()
```

Below is a similar method in a Simulator.

```
    def positions_total(self) -> int:
        """Get the number of open positions in MetaTrader 5 client.

        Returns:
            int: number of positions
        """

        if self.IS_TESTER:
            return len(self.__positions_container__)
        try:
            total = self.mt5_instance.positions_total()
        except Exception as e:
            self.__GetLogger().error(f"MetaTrader5 error = {e}")
            return -1

        return total
```

**positions\_get**

This method looks and operates similarly to the method we just discussed above, orders\_get.

_[From the documentation](https://www.mql5.com/en/docs/python_metatrader5/mt5positionsget_py):_

The function gets open positions with the ability to filter by symbol or ticket. It has three call options.

A call without parameters returns open positions for all symbols.

```
positions_get()
```

A call specifying a symbol returns open positions from a specified instrument.

```
positions_get(
   symbol="SYMBOL"      // symbol name
)
```

Call specifying a group of symbols that open positions should be received for.

```
positions_get(
   group="GROUP"        // filter for selecting positions by symbols
)
```

Call specifying a position ticket.

```
positions_get(
   ticket=TICKET        // ticket
)
```

Similarly to the method orders\_get, this method returns data in the form of a namedtuple structure. It returns None in case of an error. The info on the error can be obtained using [last\_error()](https://www.mql5.com/en/docs/python_metatrader5/mt5lasterror_py).

That being said, we need a similar structure for storing position information in our simulator, similarly to the one returned by the module MetaTrader 5-Python.

```
    def positions_get(self, symbol: Optional[str] = None, group: Optional[str] = None, ticket: Optional[int] = None) -> namedtuple:

        if self.IS_TESTER:

            positions = self.__positions_container__

            # no filters → return all positions
            if symbol is None and group is None and ticket is None:
                return tuple(positions)

            # symbol filter (highest priority)
            if symbol is not None:
                return tuple(o for o in positions if o.symbol == symbol)

            # group filter
            if group is not None:
                return tuple(o for o in positions if fnmatch.fnmatch(o.symbol, group))

            # ticket filter
            if ticket is not None:
                return tuple(o for o in positions if o.ticket == ticket)

            return tuple()

        try:
            if symbol is not None:
                return self.mt5_instance.positions_get(symbol=symbol)

            if group is not None:
                return self.mt5_instance.positions_get(group=group)

            if ticket is not None:
                return self.mt5_instance.positions_get(ticket=ticket)

            return self.mt5_instance.positions_get()

        except Exception:
            return None
```

Example usage.

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

sim.Start(IS_TESTER=True) # start simulation in the strategy tester
print("positions total in the Simulator: ",sim.positions_total())
print("positions in the Simulator:\n",sim.positions_get())

sim.Start(IS_TESTER=False) # start the simulator in real-time trading
print("positions total in MetaTrader5: ",sim.positions_total())
print("positions in MetaTraer5:\n",sim.positions_get())
```

Outputs.

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py
positions total in the Simulator:  0
positions in the Simulator:
 ()
positions total in MetaTrader5:  2
positions in MetaTraer5:
 (TradePosition(ticket=1381981938, time=1766748992, time_msc=1766748992425, time_update=1766748992, time_update_msc=1766748992425, type=0, magic=0, identifier=1381981938, reason=0, volume=0.01, price_open=1.17688, sl=0.0, tp=0.0, price_current=1.17755, swap=0.0, profit=0.67, symbol='EURUSD', comment='', external_id=''),
TradePosition(ticket=1381981988, time=1766748994, time_msc=1766748994018, time_update=1766748994, time_update_msc=1766748994018, type=1, magic=0, identifier=1381981988, reason=0, volume=0.01, price_open=1.17688, sl=0.0, tp=0.0, price_current=1.17755, swap=0.0, profit=-0.67, symbol='EURUSD', comment='', external_id=''))
```

**history\_orders\_total**

[According to the documentation,](https://www.mql5.com/en/docs/python_metatrader5/mt5historyorderstotal_py) this method gets the number of orders in trading history within a specific time interval.

```
history_orders_total(
   date_from,    // date the orders are requested from
   date_to       // date, up to which the orders are requested
   )
```

Parameters:

- date\_from: A date which the orders are requested from. Set by the 'datetime' object or as several seconds elapsed since 1970.01.01.
- date\_to: A date, up to which the orders are requested. Set by the 'datetime' object or as several seconds elapsed since 1970.01.01.

A similar function in a simulator can be implemented as follows:

```
    def history_orders_total(self, date_from: datetime, date_to: datetime) -> int:

        # date range is a requirement

        if date_from is None or date_to is None:
            self.__GetLogger().error("date_from and date_to must be specified")
            return None

        date_from = utils.ensure_utc(date_from)
        date_to = utils.ensure_utc(date_to)

        if self.IS_TESTER:

            date_from_ts = int(date_from.timestamp())
            date_to_ts   = int(date_to.timestamp())

            return sum(
                        1
                        for o in self.__orders_history_container__
                        if date_from_ts <= o.time_setup <= date_to_ts
                    )

        try:
            total = self.mt5_instance.history_orders_total(date_from, date_to)
        except Exception as e:
            self.__GetLogger().error(f"MetaTrader5 error = {e}")
            return -1

        return total
```

Example usage:

```
sim.Start(IS_TESTER=True) # start simulation in the strategy tester

date_to = datetime.now()
date_from = date_to - timedelta(days=1)

print(sim.history_orders_total(date_from=date_from,date_to=date_to))

sim.Start(IS_TESTER=False) # start the simulator in real-time trading

print(sim.history_orders_total(date_from=date_from,date_to=date_to))
```

Outputs.

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py
orders in the last 24 hours in the Simulator: 0
orders in the last 24 hours in MetaTrader5: 3
```

**history\_orders\_get**

According to the documentation, this method gets orders from a trading history with the ability to filter by ticket or position.

It returns all orders falling within a specified interval.

It has three call options:

```
history_orders_get(
   date_from,                // date the orders are requested from
   date_to,                  // date, up to which the orders are requested
   group="GROUP"        // filter for selecting orders by symbols
   )
```

Call specifying the order ticket. Returns an order with the specified ticket.

```
history_orders_get(
   ticket=TICKET        // order ticket
)
```

Call specifying the position ticket. Returns all orders with a position ticket specified in the [ORDER\_POSITION\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_integer) property.

```
history_orders_get(
   position=POSITION    // position ticket
)
```

Just like inside the function history\_orders\_total, we read all information from an array named  \_\_orders\_history\_container\_\_  with additional filters for ticket (order ticket), position (ticket of a position stored), and group (the filter for arranging a group of necessary symbols).

```
    def history_orders_get(self,
                           date_from: datetime,
                           date_to: datetime,
                           group: Optional[str] = None,
                           ticket: Optional[int] = None,
                           position: Optional[int] = None
                           ) -> namedtuple:

        if self.IS_TESTER:

            orders = self.__orders_history_container__

            # ticket filter (highest priority)
            if ticket is not None:
                return tuple(o for o in orders if o.ticket == ticket)

            # position filter
            if position is not None:
                return tuple(o for o in orders if o.position_id == position)

            # date range is a requirement
            if date_from is None or date_to is None:
                self.__GetLogger().error("date_from and date_to must be specified")
                return None

            date_from_ts = int(utils.ensure_utc(date_from).timestamp())
            date_to_ts   = int(utils.ensure_utc(date_to).timestamp())

            filtered = (
                o for o in orders
                if date_from_ts <= o.time_setup <= date_to_ts
            ) # obtain orders that fall within this time range

            # optional group filter
            if group is not None:
                filtered = (
                    o for o in filtered
                    if fnmatch.fnmatch(o.symbol, group)
                )

            return tuple(filtered)

        try: # we are not on the strategy tester simulation

            if ticket is not None:
                return self.mt5_instance.history_orders_get(date_from, date_to, ticket=ticket)

            if position is not None:
                return self.mt5_instance.history_orders_get(date_from, date_to, position=position)

            if date_from is None or date_to is None:
                raise ValueError("date_from and date_to are required")

            date_from = utils.ensure_utc(date_from)
            date_to   = utils.ensure_utc(date_to)

            if group is not None:
                return self.mt5_instance.history_orders_get(
                    date_from, date_to, group=group
                )

            return self.mt5_instance.history_orders_get(date_from, date_to)

        except Exception as e:
            self.__GetLogger().error(f"MetaTrader5 error = {e}")
            return None
```

**history\_deals\_total**

[According to the documentation](https://www.mql5.com/en/docs/python_metatrader5/mt5historydealstotal_py), this function gets the number of deals in trading history within a specified interval.

```
history_deals_total(
   date_from,    // date the deals are requested from
   date_to       // date, up to which the deals are requested
   )
```

Parameters:

- date\_from: A date the deals are requested from. Set by the 'datetime' object or as several seconds elapsed since 1970.01.01.
- date\_to: A date, up to which the deals are requested. Set by the 'datetime' object or as several seconds elapsed since 1970.01.01.

```
    def history_deals_total(self, date_from: datetime, date_to: datetime) -> int:
        """
        Get the number of deals in history within the specified date range.

        Args:
            date_from (datetime): Date the orders are requested from. Set by the 'datetime' object or as several seconds elapsed since 1970.01.01.

            date_to (datetime, required): Date, up to which the orders are requested. Set by the 'datetime' object or as several seconds elapsed since 1970.01.01.

        Returns:
            An integer value.
        """

        if date_from is None or date_to is None:
            self.__GetLogger().error("date_from and date_to must be specified")
            return -1

        date_from = utils.ensure_utc(date_from)
        date_to   = utils.ensure_utc(date_to)

        if self.IS_TESTER:

            date_from_ts = int(date_from.timestamp())
            date_to_ts   = int(date_to.timestamp())

            return sum(
                1
                for d in self.__deals_history_container__
                if date_from_ts <= d.time <= date_to_ts
            )

        try:
            return self.mt5_instance.history_deals_total(date_from, date_to)

        except Exception as e:
            self.__GetLogger().error(f"MetaTrader5 error = {e}")
            return -1
```

Example usage.

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

date_to = datetime.now()
date_from = date_to - timedelta(days=1)

print("Total deals in the last 24 hours in MetaTrader5:", sim.history_deals_total(date_from=date_from,date_to=date_to))

sim.Start(IS_TESTER=False) # start the simulator in real-time trading

print("Total deals in the last 24 hours in MetaTrader5:", sim.history_deals_total(date_from=date_from,date_to=date_to))
```

Outputs.

```
Total deals in the last 24 hours in MetaTrader5: 0
Total deals in the last 24 hours in MetaTrader5: 3
```

**history\_deals\_get**

According to the documentation, this method gets deals from a trading history within a specified time interval, with the ability to filter by ticket or position.

The function has three variants.

```
history_deals_get(
   date_from,                // date the deals are requested from
   date_to,                  // date, up to which the deals are requested
   group="GROUP"             // filter for selecting deals for symbols
   )
```

Call specifying the order ticket. Returns all deals having the specified order ticket in the [DEAL\_ORDER](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_property_integer) property.

```
history_deals_get(
   ticket=TICKET        // order ticket
)
```

Call specifying the position ticket. Return all deals having the specified position ticket in the [DEAL\_POSITION\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_property_integer) property.

```
history_deals_get(
   position=POSITION    // position ticket
)
```

In the Simulator class, we'll create a method with the same name. When a user selects the strategy tester model (IS\_TESTER=True) deals history is extracted from an array within a simulator; otherwise, such information is extracted directly from the MetaTrader 5 client.

```
    def history_deals_get(self,
                          date_from: datetime,
                          date_to: datetime,
                          group: Optional[str] = None,
                          ticket: Optional[int] = None,
                          position: Optional[int] = None
                        ) -> namedtuple:

        if self.IS_TESTER:

            deals = self.__deals_history_container__

            # ticket filter (highest priority)
            if ticket is not None:
                return tuple(d for d in deals if d.ticket == ticket)

            # position filter
            if position is not None:
                return tuple(d for d in deals if d.position_id == position)

            # date range is a requirement
            if date_from is None or date_to is None:
                self.__GetLogger().error("date_from and date_to must be specified")
                return None

            date_from_ts = int(utils.ensure_utc(date_from).timestamp())
            date_to_ts   = int(utils.ensure_utc(date_to).timestamp())

            filtered = (
                d for d in deals
                if date_from_ts <= d.time <= date_to_ts
            ) # obtain orders that fall within this time range

            # optional group filter
            if group is not None:
                filtered = (
                    d for d in filtered
                    if fnmatch.fnmatch(d.symbol, group)
                )

            return tuple(filtered)

        try: # we are not on the strategy tester simulation

            if ticket is not None:
                return self.mt5_instance.history_deals_get(date_from, date_to, ticket=ticket)

            if position is not None:
                return self.mt5_instance.history_deals_get(date_from, date_to, position=position)

            if date_from is None or date_to is None:
                raise ValueError("date_from and date_to are required")

            date_from = utils.ensure_utc(date_from)
            date_to   = utils.ensure_utc(date_to)

            if group is not None:
                return self.mt5_instance.history_deals_get(
                    date_from, date_to, group=group
                )

            return self.mt5_instance.history_deals_get(date_from, date_to)

        except Exception as e:
            self.__GetLogger().error(f"MetaTrader5 error = {e}")
            return None
```

Example usage.

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

sim.Start(IS_TESTER=True) # start simulation in the strategy tester

date_to = datetime.now()
date_from = date_to - timedelta(days=1)

print("deals in the last 24 hours in the Simulator:\n", sim.history_deals_get(date_from=date_from,date_to=date_to))

sim.Start(IS_TESTER=False) # start the simulator in real-time trading

print("Deals in the last 24 hours in MetaTrader5:\n", sim.history_deals_get(date_from=date_from,date_to=date_to))
```

Outputs.

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py
deals in the last 24 hours in the Simulator:
 ()
Deals in the last 24 hours in MetaTrader5:
 (TradeDeal(ticket=1134768493, order=1381981938, time=1766748992, time_msc=1766748992425, type=0, entry=0, magic=0, position_id=1381981938, reason=0, volume=0.01, price=1.17688, commission=-0.04, swap=0.0, profit=0.0, fee=0.0, symbol='EURUSD', comment='', external_id=''),
TradeDeal(ticket=1134768532, order=1381981988, time=1766748994, time_msc=1766748994018, type=1, entry=0, magic=0, position_id=1381981988, reason=0, volume=0.01, price=1.17688, commission=-0.04, swap=0.0, profit=0.0, fee=0.0, symbol='EURUSD', comment='', external_id=''),
TradeDeal(ticket=1135016562, order=1381968767, time=1766763381, time_msc=1766763381530, type=1, entry=0, magic=0, position_id=1381968767, reason=0, volume=0.01, price=1.17953, commission=-0.04, swap=0.0, profit=0.0, fee=0.0, symbol='EURUSD', comment='', external_id=''))
```

**account\_info**

It is necessary to have a way of obtaining account information from both the MetaTrader 5 terminal and the simulator. To achieve this in our class, we need a similar way of storing and accessing these accounts' credentials.

If you request account information from MetaTrader 5 using the method account\_info(), you will see a tuple that looks like this:

```
AccountInfo(login=52557820, trade_mode=0, leverage=500, limit_orders=200, margin_so_mode=0, trade_allowed=True, trade_expert=True, margin_mode=2, currency_digits=2, fifo_close=False, balance=941.54, credit=0.0, profit=2.37, equity=943.91, margin=2.36, margin_free=941.55, margin_level=39996.18644067797, margin_so_call=100.0, margin_so_so=0.0, margin_initial=0.0, margin_maintenance=0.0, assets=0.0, liabilities=0.0, commission_blocked=0.0, name='OMEGA MSIGWA', server='ICMarketsSC-Demo', currency='USD', company='Raw Trading Ltd')
```

It is said [in the documentation](https://www.mql5.com/en/docs/python_metatrader5/mt5accountinfo_py) that the function, _returns info in the form of a named tuple structure (namedtuple). It returns None in case of an error. The info on the error can be obtained using [last\_error()](https://www.mql5.com/en/docs/python_metatrader5/mt5lasterror_py)._

We define a similar structure inside the class Simulator.

```
        self.AccountInfo = namedtuple(
            "AccountInfo",
            [\
                "login",\
                "trade_mode",\
                "leverage",\
                "limit_orders",\
                "margin_so_mode",\
                "trade_allowed",\
                "trade_expert",\
                "margin_mode",\
                "currency_digits",\
                "fifo_close",\
                "balance",\
                "credit",\
                "profit",\
                "equity",\
                "margin",\
                "margin_free",\
                "margin_level",\
                "margin_so_call",\
                "margin_so_so",\
                "margin_initial",\
                "margin_maintenance",\
                "assets",\
                "liabilities",\
                "commission_blocked",\
                "name",\
                "server",\
                "currency",\
                "company",\
            ]
        )
```

Since we hope to imitate MetaTrader 5 with this simulator class, we have to populate some of the MetaTrader 5 account's information into a simulated account.

```
        mt5_acc_info = mt5_instance.account_info()

        if mt5_acc_info is None:
            raise RuntimeError("Failed to obtain MT5 account info")

        self.__account_state_update(
            account_info=self.AccountInfo(
                # ---- identity / broker-controlled ----
                login=11223344,
                trade_mode=mt5_acc_info.trade_mode,
                leverage=int(leverage.split(":")[1]),
                limit_orders=mt5_acc_info.limit_orders,
                margin_so_mode=mt5_acc_info.margin_so_mode,
                trade_allowed=mt5_acc_info.trade_allowed,
                trade_expert=mt5_acc_info.trade_expert,
                margin_mode=mt5_acc_info.margin_mode,
                currency_digits=mt5_acc_info.currency_digits,
                fifo_close=mt5_acc_info.fifo_close,

                # ---- simulator-controlled financials ----
                balance=deposit,                # simulator starting balance
                credit=mt5_acc_info.credit,
                profit=0.0,
                equity=deposit,
                margin=0.0,
                margin_free=deposit,
                margin_level=0.0,

                # ---- risk thresholds (copied from broker) ----
                margin_so_call=mt5_acc_info.margin_so_call,
                margin_so_so=mt5_acc_info.margin_so_so,
                margin_initial=mt5_acc_info.margin_initial,
                margin_maintenance=mt5_acc_info.margin_maintenance,

                # ---- rarely used but keep parity ----
                assets=mt5_acc_info.assets,
                liabilities=mt5_acc_info.liabilities,
                commission_blocked=mt5_acc_info.commission_blocked,

                # ---- descriptive ----
                name="John Doe",
                server="MetaTrader5-Simulator",
                currency=mt5_acc_info.currency,
                company=mt5_acc_info.company,
            )
        )
```

We populate every detail except the financial details we can calculate, such as the account balance, equity, margin, margin free, and margin level.

Inside the function account\_info, we check whether a user has selected the strategy tester mode (IS\_TESTER=True); we return the simulator's account information; otherwise, we return information from an account in MetaTrader 5.

```
    def account_info(self) -> namedtuple:

        """Gets info on the current trading account."""

        if self.IS_TESTER:
            return self.AccountInfo

        mt5_ac_info = self.mt5_instance.account_info()
        if  mt5_ac_info is None:
            self.__GetLogger().warning(f"Failed to obtain MT5 account info, MT5 Error = {self.mt5_instance.last_error()}")
            return

        return mt5_ac_info
```

Example usage:

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:200")

sim.Start(IS_TESTER=True) # start simulation in the strategy tester
print("simulator's account info: ", sim.account_info())

sim.Start(IS_TESTER=False) # start the simulator in real-time trading
print("MetaTrader5's account info: ", sim.account_info())
```

Outputs.

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py
simulator's account info:  AccountInfo(login=11223344, trade_mode=0, leverage=200, limit_orders=200, margin_so_mode=0, trade_allowed=True, trade_expert=True, margin_mode=2, currency_digits=2, fifo_close=False, balance=1078.3, credit=0.0, profit=0.0, equity=1078.3, margin=0.0, margin_free=1078.3, margin_level=0.0, margin_so_call=100.0, margin_so_so=0.0, margin_initial=0.0, margin_maintenance=0.0, assets=0.0, liabilities=0.0, commission_blocked=0.0, name='John Doe', server='MetaTrader5-Simulator', currency='USD', company='Raw Trading Ltd')
MetaTrader5's account info:  AccountInfo(login=52557820, trade_mode=0, leverage=500, limit_orders=200, margin_so_mode=0, trade_allowed=True, trade_expert=True, margin_mode=2, currency_digits=2, fifo_close=False, balance=941.54, credit=0.0, profit=2.37, equity=943.91, margin=2.36, margin_free=941.55, margin_level=39996.18644067797, margin_so_call=100.0, margin_so_so=0.0, margin_initial=0.0, margin_maintenance=0.0, assets=0.0, liabilities=0.0, commission_blocked=0.0, name='OMEGA MSIGWA', server='ICMarketsSC-Demo', currency='USD', company='Raw Trading Ltd')
```

**order\_calc\_profit**

This is one of the useful functions in our simulator as it helps in estimating how much is risked or aimed to be gained on a particular position/order.

According to the documentation.

_This function returns profit in the account currency for a specified trading operation._

```
order_calc_profit(
   action,          // order type (ORDER_TYPE_BUY or ORDER_TYPE_SELL)
   symbol,          // symbol name
   volume,          // volume
   price_open,      // open price
   price_close      // close price
   );
```

To make a similar function in MQL5, we have to understand the inner workings of this MetaTrader 5 function.

A detailed description of it is can be found here: [https://www.mql5.com/en/book/automation/experts/experts\_ordercalcprofit](https://www.mql5.com/en/book/automation/experts/experts_ordercalcprofit)

Below is a table containing formulas for estimating the profit of an order in MetaTrader 5.

| Identifier | Formula |
| --- | --- |
| SYMBOL\_CALC\_MODE\_FOREX | (ClosePrice - OpenPrice) \* ContractSize \* Lots |
| SYMBOL\_CALC\_MODE\_FOREX\_NO\_LEVERAGE | (ClosePrice - OpenPrice) \* ContractSize \* Lots |
| SYMBOL\_CALC\_MODE\_CFD | (ClosePrice - OpenPrice) \* ContractSize \* Lots |
| SYMBOL\_CALC\_MODE\_CFDINDEX | (ClosePrice - OpenPrice) \* ContractSize \* Lots |
| SYMBOL\_CALC\_MODE\_CFDLEVERAGE | (ClosePrice - OpenPrice) \* ContractSize \* Lots |
| SYMBOL\_CALC\_MODE\_EXCH\_STOCKS | (ClosePrice - OpenPrice) \* ContractSize \* Lots |
| SYMBOL\_CALC\_MODE\_EXCH\_STOCKS\_MOEX | (ClosePrice - OpenPrice) \* ContractSize \* Lots |
| SYMBOL\_CALC\_MODE\_FUTURES | (ClosePrice - OpenPrice) \* Lots \* TickPrice / TickSize |
| SYMBOL\_CALC\_MODE\_EXCH\_FUTURES | (ClosePrice - OpenPrice) \* Lots \* TickPrice / TickSize |
| SYMBOL\_CALC\_MODE\_EXCH\_FUTURES\_FORTS | (ClosePrice - OpenPrice) \* Lots \* TickPrice / TickSize |
| SYMBOL\_CALC\_MODE\_EXCH\_BONDS | Lots \* ContractSize \* (ClosePrice \* FaceValue + AccruedInterest) |
| SYMBOL\_CALC\_MODE\_EXCH\_BONDS\_MOEX | Lots \* ContractSize \* (ClosePrice \* FaceValue + AccruedInterest) |
| SYMBOL\_CALC\_MODE\_SERV\_COLLATERAL | Lots \* ContractSize \* MarketPrice \* LiqudityRate |

We introduce the same formulas in a similarly named function within our simulator.

```
    def order_calc_profit(self,
                        action: int,
                        symbol: str,
                        volume: float,
                        price_open: float,
                        price_close: float) -> float:
        """
        Return profit in the account currency for a specified trading operation.

        Args:
            action (int): The type of position taken, either 0 (buy) or 1 (sell).
            symbol (str): Financial instrument name.
            volume (float):   Trading operation volume.
            price_open (float): Open Price.
            price_close (float): Close Price.
        """


        sym = self.symbol_info(symbol)

        if self.IS_TESTER:

            contract_size = sym.trade_contract_size

            # --- Determine direction ---
            if action == mt5.ORDER_TYPE_BUY:
                direction = 1
            elif action == mt5.ORDER_TYPE_SELL:
                direction = -1
            else:
                self.__GetLogger().critical("order_calc_profit failed: invalid order type")
                return 0.0

            # --- Core profit calculation ---

            calc_mode = sym.trade_calc_mode
            price_delta = (price_close - price_open) * direction

            try:
                # ------------------ FOREX / CFD / STOCKS -----------------------
                if calc_mode in (
                    mt5.SYMBOL_CALC_MODE_FOREX,
                    mt5.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE,
                    mt5.SYMBOL_CALC_MODE_CFD,
                    mt5.SYMBOL_CALC_MODE_CFDINDEX,
                    mt5.SYMBOL_CALC_MODE_CFDLEVERAGE,
                    mt5.SYMBOL_CALC_MODE_EXCH_STOCKS,
                    mt5.SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX,
                ):
                    profit = price_delta * contract_size * volume

                # ---------------- FUTURES --------------------
                elif calc_mode in (
                    mt5.SYMBOL_CALC_MODE_FUTURES,
                    mt5.SYMBOL_CALC_MODE_EXCH_FUTURES,
                    # mt5.SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS,
                ):
                    tick_value = sym.trade_tick_value
                    tick_size = sym.trade_tick_size

                    if tick_size <= 0:
                        self.__GetLogger().critical("Invalid tick size")
                        return 0.0

                    profit = price_delta * volume * (tick_value / tick_size)

                # ---------- BONDS -------------------

                elif calc_mode in (
                    mt5.SYMBOL_CALC_MODE_EXCH_BONDS,
                    mt5.SYMBOL_CALC_MODE_EXCH_BONDS_MOEX,
                ):
                    face_value = sym.trade_face_value
                    accrued_interest = sym.trade_accrued_interest

                    profit = (
                        volume
                        * contract_size
                        * (price_close * face_value + accrued_interest)
                        - volume
                        * contract_size
                        * (price_open * face_value)
                    )

                # ------ COLLATERAL -------
                elif calc_mode == mt5.SYMBOL_CALC_MODE_SERV_COLLATERAL:
                    liquidity_rate = sym.trade_liquidity_rate
                    market_price = (
                        self.tick.ask if action == mt5.ORDER_TYPE_BUY else self.tick.bid
                    )

                    profit = (
                        volume
                        * contract_size
                        * market_price
                        * liquidity_rate
                    )

                else:
                    self.__GetLogger().critical(
                        f"Unsupported trade calc mode: {calc_mode}"
                    )
                    return 0.0

                return round(profit, 2)

            except Exception as e:
                self.__GetLogger().critical(f"Failed: {e}")
                return 0.0

        # if we are not on the strategy tester

        try:
            profit = self.mt5_instance.order_calc_profit(
                action,
                symbol,
                volume,
                price_open,
                price_close
            )

        except Exception as e:
            self.__GetLogger().critical(f"Failed to calculate profit of a position, MT5 error = {self.mt5_instance.last_error()}")
            return np.nan

        return profit

```

Example usage.

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

sim.Start(IS_TESTER=True) # start simulation in the strategy tester

profit = sim.order_calc_profit(action=mt5.POSITION_TYPE_SELL, symbol=symbol, volume=0.01, price_open=entry, price_close=tp)
print("Simulator profit caclulate: ", profit)

sim.Start(IS_TESTER=False) # start the simulator in real-time trading

profit = sim.order_calc_profit(action=mt5.POSITION_TYPE_SELL, symbol=symbol, volume=0.01, price_open=entry, price_close=tp)
print("MT5 profit caclulate: ", round(profit, 2))
```

Outputs.

```
Simulator profit caclulate:  1.68
MT5 profit caclulate:  1.68
```

**order\_calc\_margin**

This is another useful function in the MetaTrader 5 API, despite its operations being less well-known.

[According to the documentation](https://www.mql5.com/en/book/automation/symbols/symbols_margin), this function calculates margin in the account currency to perform a specified trading operation.

The following table represents the formulas used in making the function order\_calc\_margin the way it is.

| Identifier | Formula |
| --- | --- |
| SYMBOL\_CALC\_MODE\_FOREX<br>Forex | Lots \* ContractSize \* MarginRate / Leverage |
| SYMBOL\_CALC\_MODE\_FOREX\_NO\_LEVERAGE<br>Forex without leverage | Lots \* ContractSize \* MarginRate |
| SYMBOL\_CALC\_MODE\_CFD<br>CFD | Lots \* ContractSize \* MarketPrice \* MarginRate |
| SYMBOL\_CALC\_MODE\_CFDLEVERAGE<br>CFD with leverage | Lots \* ContractSize \* MarketPrice \* MarginRate / Leverage |
| SYMBOL\_CALC\_MODE\_CFDINDEX<br>CFDs on indices | Lots \* ContractSize \* MarketPrice \* TickPrice / TickSize \* MarginRate |
| SYMBOL\_CALC\_MODE\_EXCH\_STOCKS<br>Securities on the stock exchange | Lots \* ContractSize \* LastPrice \* MarginRate |
| SYMBOL\_CALC\_MODE\_EXCH\_STOCKS\_MOEX<br>Securities on MOEX | Lots \* ContractSize \* LastPrice \* MarginRate |
| SYMBOL\_CALC\_MODE\_FUTURES<br>Futures | Lots \* InitialMargin \* MarginRate |
| SYMBOL\_CALC\_MODE\_EXCH\_FUTURES<br>Futures on the stock exchange | Lots \* InitialMargin \* MarginRate              or<br>Lots \* MaintenanceMargin \* MarginRate |
| SYMBOL\_CALC\_MODE\_EXCH\_FUTURES\_FORTS<br>Futures on FORTS | Lots \* InitialMargin \* MarginRate              or<br>Lots \* MaintenanceMargin \* MarginRate |
| SYMBOL\_CALC\_MODE\_EXCH\_BONDS<br>Bonds on the stock exchange | Lots \* ContractSize \* FaceValue \* OpenPrice / 100 |
| SYMBOL\_CALC\_MODE\_EXCH\_BONDS\_MOEX<br>Bonds on MOEX | Lots \* ContractSize \* FaceValue \* OpenPrice / 100 |
| SYMBOL\_CALC\_MODE\_SERV\_COLLATERAL | Non-tradable asset (margin not applicable) |

We'll use the same formulas in estimating the order's margin in our Simulator class.

```
    def order_calc_margin(self, action: int, symbol: str, volume: float, price: float) -> float:
        """
        Return margin in the account currency to perform a specified trading operation.

        """

        if volume <= 0 or price <= 0:
            self.__GetLogger().error("order_calc_margin failed: invalid volume or price")
            return 0.0

        if not self.IS_TESTER:
            try:
                return round(self.mt5_instance.order_calc_margin(action, symbol, volume, price), 2)
            except Exception:
                self.__GetLogger().warning(f"Failed: MT5 Error = {self.mt5_instance.last_error()}")
                return 0.0

        # IS_TESTER = True
        sym = self.symbol_info(symbol)

        contract_size = sym.trade_contract_size
        leverage = max(self.AccountInfo.leverage, 1)

        margin_rate = (
            sym.margin_initial
            if sym.margin_initial > 0
            else sym.margin_maintenance
        )

        if margin_rate <= 0: # if margin rate is zero set it to 1
            margin_rate = 1.0

        mode = sym.trade_calc_mode

        if mode == self.mt5_instance.SYMBOL_CALC_MODE_FOREX:
            margin = (volume * contract_size * price) / leverage

        elif mode == self.mt5_instance.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE:
            margin = volume * contract_size * price

        elif mode in (
            self.mt5_instance.SYMBOL_CALC_MODE_CFD,
            self.mt5_instance.SYMBOL_CALC_MODE_CFDINDEX,
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_STOCKS,
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX,
        ):
            margin = volume * contract_size * price * margin_rate

        elif mode == self.mt5_instance.SYMBOL_CALC_MODE_CFDLEVERAGE:
            margin = (volume * contract_size * price * margin_rate) / leverage

        elif mode in (
            self.mt5_instance.SYMBOL_CALC_MODE_FUTURES,
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_FUTURES,
            # self.mt5_instance.SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS,
        ):
            margin = volume * sym.margin_initial

        elif mode in (
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_BONDS,
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_BONDS_MOEX,
        ):
            margin = (
                volume
                * contract_size
                * sym.trade_face_value
                * price
                / 100
            )

        elif mode == self.mt5_instance.SYMBOL_CALC_MODE_SERV_COLLATERAL:
            margin = 0.0

        else:
            self.__GetLogger().warning(f"Unknown calc mode {mode}, fallback margin formula used")
            margin = (volume * contract_size * price) / leverage

        return round(margin, 2)
```

The margin\_rate part is the trickiest one as we have to ensure the values exist before deciding the right rate value to use.

### Final Thoughts

In the article, we introduced a way of passing tick data in our simulator and implemented almost all necessary functions provided by the MetaTrader 5-python API, this brings us closer to an isolated environment for simulating how MetaTrader 5 works, and in doing so, we will make a custom strategy tester for our Python trading bots.

In the next article, we will implement trading functions and simulate a trading activity for some ticks in the past. More interesting stuff is on the way, so stay tuned!

Peace out.

Share your thoughts and help improve this project on GitHub: [https://github.com/MegaJoctan/PyMetaTester](https://www.mql5.com/go?link=https://github.com/MegaJoctan/StrategyTester5 "https://github.com/MegaJoctan/PyMetaTester")

**Attachments Tab**

| Filename | Description & Usage |
| --- | --- |
| bars.py | It has functions for collecting bars from the MetaTrader 5 client to a custom file and path. |
| ticks.py | It has functions for collecting ticks from the MetaTrader 5 client to a custom file and path. |
| config.py | A Python configuration file where the most useful variables for reusability throughout the project are defined. |
| utils.py | A utility Python file which contains simple functions to help with various tasks (helpers). |
| simulator.py | It has a class named Simulator. Our core simulator logic is in one place. |
| test.py | A file used for testing all the code and functions discussed in this post. |
| error\_description.py | It has functions for converting all MetaTrader 5 error codes into human-readable messages. |
| requirements.txt | Contains all Python dependencies and their versions, used in this project. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20455.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/20455/Attachments.zip "Download Attachments.zip")(18.65 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)
- [Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://www.mql5.com/en/articles/18971)

**[Go to discussion](https://www.mql5.com/en/forum/503481)**

![Larry Williams Market Secrets (Part 5): Automating the Volatility Breakout Strategy in MQL5](https://c.mql5.com/2/190/20745-larry-williams-market-secrets-logo.png)[Larry Williams Market Secrets (Part 5): Automating the Volatility Breakout Strategy in MQL5](https://www.mql5.com/en/articles/20745)

This article demonstrates how to automate Larry Williams’ volatility breakout strategy in MQL5 using a practical, step-by-step approach. You will learn how to calculate daily range expansions, derive buy and sell levels, manage risk with range-based stops and reward-based targets, and structure a professional Expert Advisor for MetaTrader 5. Designed for traders and developers looking to transform Larry Williams’ market concepts into a fully testable and deployable automated trading system.

![From Basic to Intermediate: Events (I)](https://c.mql5.com/2/121/Do_b0sico_ao_intermediyrio_Eventos___LOGO.png)[From Basic to Intermediate: Events (I)](https://www.mql5.com/en/articles/15732)

Given everything that has been shown so far, I think we can now start implementing some kind of application to run some symbol directly on the chart. However, first we need to talk about a concept that can be rather confusing for beginners. Namely, it's the fact that applications developed in MQL5 and intended for display on a chart are not created in the same way as we have seen so far. In this article, we'll begin to understand this a little better.

![Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://c.mql5.com/2/190/20851-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://www.mql5.com/en/articles/20851)

This article explores a method that combines Heikin‑Ashi smoothing with EMA20 High and Low boundaries and an EMA50 trend filter to improve trade clarity and timing. It demonstrates how these tools can help traders identify genuine momentum, filter out noise, and better navigate volatile or trending markets.

![Forex arbitrage trading: A simple synthetic market maker bot to get started](https://c.mql5.com/2/126/Forex_Arbitrage_Trading_Simple_Synthetic_Market_Maker_Bot_to_Get_Started__LOGO.png)[Forex arbitrage trading: A simple synthetic market maker bot to get started](https://www.mql5.com/en/articles/17424)

Today we will take a look at my first arbitrage robot — a liquidity provider (if you can call it that) for synthetic assets. Currently, this bot is successfully operating as a module in a large machine learning system, but I pulled up an old Forex arbitrage robot from the cloud, so let's take a look at it and think about what we can do with it today.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/20455&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062802144865724722)

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