---
title: Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating
url: https://www.mql5.com/en/articles/19428
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:46:21.587999
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/19428&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083098442319795452)

MetaTrader 5 / Trading systems


### Introduction

It’s time to start updating our cointegration model in real-time. More specifically, it’s time to start updating the relative weight of each stock in our trading portfolio with the more recent calculations, while the Expert Advisor is running. Until now, we have been using the same portfolio weights in our explanations and backtests. They were useful as a simplification to ease the understanding of the process and the meaning and workings of the underlying statistical tests. But in real life, these portfolio weights are changing practically every time a new data point arrives, that is, at each new closed bar, in our case.

But first, a bit of context, a quick recap to make sure we are all on the same page.

We are developing a kind of poor man’s statistical arbitrage framework - a stat arb pipeline designed for the average retail trader, with a consumer notebook, and a regular network bandwidth. Here, in this series of articles, we are documenting and describing the statistical tests used for finding highly correlated assets for pairs trading, cointegrated stocks for portfolio building, and also the tests used for assessing the stationarity of their relative spreads, which is required for mean reversion strategies like ours.

[In the previous article of this series (Part 3)](https://www.mql5.com/en/articles/19242), we described the database setup with some notes about our design choices. This article is a direct continuation of that last one. Once our database is set up and being updated regularly, we can decouple our data analysis from our trading activity.

Now we will see:

- how we have modified our Python scripts to incorporate the lessons learned, with all the tests being processed in a single run to ease our asset selection;

- how we totally revamped our Expert Advisor code to make use of the up-to-date database while running, and how we moved almost all functions to an MQL5 header file to make things more manageable and maintainable, keeping only the native event handlers on the main file;

- And, finally, we will see the modifications we made in our database so it can act as an interface between the real-time data analysis and the online trading environment.

Let’s start with the last one, item 3, so the other two items will make sense when described.

### Database as the single source of truth

We started the previous article with a straightforward definition of what kind of answer we want from our database to our Expert Advisor to update the model in real-time. This is a typical “cheat code” commonly used for bottom-up designs. We ask what the immediate need is and work to implement it. This avoids over-engineering and makes route corrections easier. This is the relevant snippet from the previous article.

At the end of the day, our database should answer a very simple question: what should we trade now to get the maximum possible return?

Referring to the last article in this series, where we defined the directions and volume of our orders based on the stock basket portfolio weights, one possible answer could be something like this:

| symbols | weights | timeframe |
| --- | --- | --- |
| "MU", "NVDA", "MPWR", "MCHP" | 2.699439, 1.000000, -1.877447, -2.505294 | D1 (daily timeframe) |

Table 1. Made-up sample of a expected query response for real-time model update (previous article)

To get this kind of answer, we added a new table to our database schema, the ‘strategy’ table. It has four TEXT fields:

- Name PRIMARY KEY
- Symbols
- Weights
- Timeframe

```
-- strategy definition

CREATE TABLE strategy (
        name TEXT CHECK(LENGTH(name) <= 20) NOT NULL,
        symbols TEXT CHECK(LENGTH(symbols) <= 255) NOT NULL,
        weights TEXT CHECK(LENGTH(weights) <= 255),
        timeframe TEXT CHECK (
        timeframe IN (
            'M1',
            'M2',
            'M3',
            'M4',
            'M5',
            'M6',
            'M10',
            'M12',
            'M15',
            'M20',
            'M30',
            'H1',
            'H2',
            'H3',
            'H4',
            'H6',
            'H8',
            'H12',
            'D1',
            'W1',
            'MN1'
        )
    ),CONSTRAINT strategy_pk PRIMARY KEY (name)
) WITHOUT ROWID, STRICT;
```

We defined this table as a WITHOUT ROWID table. [In SQLite, rowid tables](https://www.mql5.com/go?link=https://www.sqlite.org/rowidtable.html "https://www.sqlite.org/rowidtable.html") are all the regular tables except virtual ones and those with the INTEGER PRIMARY KEY constraint. In rowid tables, the primary keys are not real primary keys.

“The PRIMARY KEY constraint for a rowid table (as long as it is not the true primary key or INTEGER PRIMARY KEY) is really the same thing as a UNIQUE constraint. Because it is not a true primary key, columns of the PRIMARY KEY are allowed to be NULL, in violation of all SQL standards.” (emphasis is ours).

Thus, by defining the table as a WITHOUT ROWID table, we avoid having NULL being inserted and, consequently, we avoid having the relational integrity compromised. The drawback is that this clause is SQLite-specific, meaning our schema stops being fully portable to other RDBMS. To use the same schema in other RDBMS, we will have to adapt it. But I think the benefit of not having NULL being inserted in the primary key field is worth the cost.

As you may have noted, ‘symbols’ and ‘weights’ are both arrays. But SQLite does not have the ARRAY data type. Because of this limitation, we will be inserting this data as TEXT, but formatted as a JSON array. This measure will ease our work later if/when we move to another RDBMS that has the ARRAY data type.

| name | symbols | weights | timeframe |
| --- | --- | --- | --- |
| Nasdaq\_NVDA\_Coint | ‘\["MU", "NVDA", "MPWR", "MCHP"\]’ | ‘\[“2.699439”, “1.000000”, “-1.877447”, “-2.505294”\]’ | D1 |

Table 2. Symbols and weights data are inserted in the ‘strategy’ table as the TEXT datatype, formatted as JSON arrays

To query them, we will use the [json\_extract() SQLite function](https://www.mql5.com/go?link=https://www.sqlite.org/json1.html%23the_json_extract_function "https://www.sqlite.org/json1.html#the_json_extract_function").

So our database schema, now upgraded to statarb-0.2, includes the additional ‘strategy’ table, and its Entity-Relationship Diagram looks like this:

![Figure 1 - The statarb-0.2 Database Entity-Relationship Diagram (ERD)](https://c.mql5.com/2/167/statarb-0.2_ER.png)

Figure 1. The statarb-0.2 Database Entity-Relationship Diagram (ERD)

TIP - While developing with SQLite, I’m used to NOT deleting the db files' old versions, as well as the old versions of the schema files. Since we are dealing with a single file, it is easy to duplicate the file, rename the copy to the new version, and maintain both in the Version Control System (VCS). This practice gives us a simple “rollback path”, just in case.

![Figure 2 - Metaeditor Navigator showing different versions of SQLite files (db’s) and respective schema files](https://c.mql5.com/2/167/Capture_metaeditor_sqlite_files_versions.PNG)

Figure 2. Metaeditor Navigator showing different versions of SQLite files (db’s) and respective schema files

As you can see by the ERD pictured above, our new ‘strategy’ table is not connected to any other table yet. This will change as soon as we start backtesting/trading with this new EA version, when it will be connected to the ‘trade’ table. Again, since we are working bottom-up, we will modify our database when, and only when, the need arises. For now, the ‘strategy’ table is nothing more than an interface, a means of communication between our data analysis and our trading environments.

The database will act as an interface between our data analysis and our trading Expert Advisor, and also as an interface between our screening - the application of our data analysis - and the EA. The flow will be something like that:

![Figure 3 - A flow diagram illustrates the circular aspect of our statistical arbitrage pipeline](https://c.mql5.com/2/167/statarb-0.2_pipeline_flow.png)

Figure 3. A flow diagram illustrates the circular aspect of our statistical arbitrage pipeline

### The Python side

We’ve transitioned from individual Jupyter notebooks for each task to a unified Python script that executes a comprehensive statistical arbitrage pipeline for cointegrated stocks. This script will:

Check if the SQLite database exists.

```
def CheckForDb(db_path: str | Path) -> bool:
    """Return True if the SQLite database file exists on disk."""
    exists = Path(db_path).is_file()
    logger.info("Database exists: %s (%s)", exists, db_path)
    return exists
```

It will create one database if one is not found. This is a convenience if you are not following this series and just want to experiment with the stat arb pipeline. All that you need to do is to provide the database schema (attached here).

```
def CreateDbIfMissing(db_path: str | Path, schema_path: Optional[str | Path] = None) -> None:
    """
    If the database does not exist and a schema is provided, create it from SQL.
    Otherwise, create an empty file (SQLite creates DB on first connection).
    """
    db_path = Path(db_path)
    if db_path.exists():
        logger.info("DB already present: %s", db_path)
        return
    logger.info("DB missing; creating: %s", db_path)

    with sqlite3.connect(db_path) as conn:
        if schema_path:
            schema_sql = Path(schema_path).read_text(encoding="utf-8")
            conn.executescript(schema_sql)
            logger.info("Applied schema from: %s", schema_path)
```

Connect to the MetaTrader 5 terminal.

```
def InitMT5(login: Optional[int] = None, password: Optional[str] = None,
            server: Optional[str] = None, path: Optional[str] = None) -> None:

    """Initialize connection to the running MetaTrader 5 terminal."""

    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not available.")

    # if not mt5.initialize(path=path, login=login, password=password, server=server):
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    logger.info("MetaTrader5 initialized.")
```

Download market data from MetaTrader 5.

```
def DownloadMarketData(symbols: List[str], timeframes: List[str], bars: int = 5000
                       ) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Download OHLCV data from MT5 for each (symbol, timeframe).
    Returns dict with keys (symbol, timeframe) -> DataFrame columns:
    ['tstamp','timeframe','open','high','low','close','tick_volume','real_volume','spread']
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not available.")
    out: Dict[Tuple[str, str], pd.DataFrame] = {}

    for sym in symbols:
        for tf in timeframes:
            tf_const = _mt5_timeframe_constant(tf)
            rates = mt5.copy_rates_from_pos(sym, tf_const, 0, bars)

            if rates is None or len(rates) == 0:
                logger.warning("No data for %s %s", sym, tf)
                continue

            df = pd.DataFrame(rates)

            # Rename & select
            df = df.rename(columns={
                "time": "tstamp",
                "open": "price_open",
                "high": "price_high",
                "low": "price_low",
                "close": "price_close",
                "real_volume": "real_volume",
            })

            df["timeframe"] = tf

            cols = ["tstamp","timeframe","price_open","price_high","price_low","price_close",\
                    "tick_volume","real_volume","spread"]
            df = df[cols].copy()
            out[(sym, tf)] = df
            logger.info("Downloaded %d bars for %s %s", len(df), sym, tf)
    return out
```

Store this market data in the SQLite database.

```
def StoreMarketData(db_path: str | Path, data: Dict[Tuple[str, str], pd.DataFrame]) -> None:

    """Upsert market data into SQLite according to the provided schema."""
    with sqlite3.connect(db_path) as conn:
        for (sym, tf), df in data.items():
            symbol_id = _ensure_symbol(conn, sym)

            df = df.copy()
            df["symbol_id"] = symbol_id

            # Use executemany for performance
            sql = """
            INSERT OR REPLACE INTO market_data
            (tstamp, timeframe, price_open, price_high, price_low, price_close,
             tick_volume, real_volume, spread, symbol_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            rows = list(df[["tstamp","timeframe","price_open","price_high","price_low","price_close",\
                            "tick_volume","real_volume","spread","symbol_id"]].itertuples(index=False, name=None))
            conn.executemany(sql, rows)
            conn.commit()

            logger.info("Stored %d rows for %s %s", len(rows), sym, tf)
```

Check the Pearson correlation for the selected symbols.

```
def RunCorrelationTest(prices: pd.DataFrame, plot: bool = True) -> Tuple[pd.DataFrame, Tuple[str, str], float]:

    """Compute Pearson correlation matrix and return the most correlated pair (off-diagonal)."""
    corr = prices.corr(method="pearson")

    # Mask diagonal, find max
    corr_values = corr.where(~np.eye(len(corr), dtype=bool))
    idxmax = np.unravel_index(np.nanargmax(corr_values.values), corr_values.shape)
    symbols = corr.columns.tolist()
    pair = (symbols[idxmax[0]], symbols[idxmax[1]])
    value = corr_values.values[idxmax]

    logger.info("Most correlated pair: %s ~ %s (r=%.4f)", pair[0], pair[1], value)

    if plot:
        plt.figure(figsize=(10, 7))
        sns.heatmap(corr, annot=False, vmin=-1, vmax=1, cmap="coolwarm", square=True)
        plt.title("Pearson Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    print(f"Most correlated pair: {pair[0]} ~ {pair[1]} (r={value:.4f})")

    return corr, pair, float(value)
```

Calculate the Engle-Granger cointegration coefficient for the selected symbols and indicate the most correlated pair.

```
def RunEngleGrangerTest(prices: pd.DataFrame, plot: bool = True
                        ) -> Tuple[Tuple[str, str], float, float, pd.Series]:
    """
    Loop over all symbol pairs, run statsmodels.coint (Engle-Granger). Return:
        best_pair, best_pvalue, hedge_ratio (y_on_x), spread series (y - beta*x)
    """
    best_pair, best_p, best_beta, best_spread = None, 1.0, np.nan, None
    cols = list(prices.columns)
    for y_sym, x_sym in itertools.combinations(cols, 2):
        y = prices[y_sym].dropna()
        x = prices[x_sym].dropna()
        join = pd.concat([y, x], axis=1).dropna()

        if len(join) < 50:
            continue

        score, pvalue, _ = coint(join.iloc[:,0], join.iloc[:,1])

        if pvalue < best_p:
            beta = _ols_hedge_ratio(join.iloc[:,0], join.iloc[:,1])
            spread = join.iloc[:,0] - beta * join.iloc[:,1]
            best_pair, best_p, best_beta, best_spread = (y_sym, x_sym), pvalue, beta, spread

    if best_pair is None:
        raise ValueError("No cointegrated pairs found (Engle-Granger) with sufficient data.")

    print(f"Most cointegrated pair (Engle-Granger): {best_pair[0]} ~ {best_pair[1]} (p={best_p:.6f}, beta={best_beta:.4f})")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prices.index, prices[best_pair[0]], label=best_pair[0])
        ax.plot(prices.index, prices[best_pair[1]]*best_beta, label=f"{best_pair[1]} * beta")
        ax.set_title(f"Best EG Pair Overlay (beta={best_beta:.4f})")
        ax.legend()
        fig.tight_layout()
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(10, 4))
	ax2.plot(best_spread.index, best_spread.values, label="Spread")
	ax2.axhline(0, line)
	ax2.set_title("Spread (y - beta*x)")

        ax2.legend()
	fig2.tight_layout()

        plt.show()

    return best_pair, float(best_p), float(best_beta), best_spread
```

Run the stationarity tests (ADF and KPSS) on the Engle-Granger spread.

```
def RunADFTest(series: pd.Series) -> Tuple[float, float, dict]:
    """Augmented Dickey-Fuller test on a univariate series."""
    series = pd.Series(series).dropna()
    adf_stat, pvalue, usedlag, nobs, crit, icbest = adfuller(series, autolag="AIC")

    print(f"ADF: stat={adf_stat:.4f}, p={pvalue:.6f}, lags={usedlag}, nobs={nobs}, crit={crit}")

    return float(adf_stat), float(pvalue), crit
```

```
def RunKPSSTest(series: pd.Series, regression: str = "c") -> Tuple[float, float, dict]:

    """KPSS stationarity test on a univariate series (regression='c' or 'ct')."""
    series = pd.Series(series).dropna()
    stat, pvalue, lags, crit = kpss(series, regression=regression, nlags="auto")

    print(f"KPSS({regression}): stat={stat:.4f}, p={pvalue:.6f}, lags={lags}, crit={crit}")

    return float(stat), float(pvalue), crit
```

Verify the Johansen cointegration (with portfolio weights from the first eigenvector when rank >= 1).

```
def RunJohansenTest(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1, plot: bool = True
                    ) -> Tuple[int, np.ndarray, np.ndarray, Optional[pd.Series]]:
    """
    Johansen cointegration test on a set of symbols (>=2).
    Returns: (rank, eigenvectors (beta), eigenvalues, spread_from_first_vector or None)
    - Significant rank determined by trace statistics vs 95% critical values.
    - First eigenvector provides portfolio weights (normalized s.t. min(|w|)=1 or w[-1]=1).
    """

    data = prices.dropna().values

    if data.shape[1] < 2:
        raise ValueError("Johansen requires at least two series.")

    joh = coint_johansen(data, det_order, k_ar_diff)

    # Significant rank by trace statistic > crit value (95% = column index 1)
    trace_stats = joh.lr1  # trace stats
    crit_vals = joh.cvt[:, 1]  # 95%
    rank = int(np.sum(trace_stats > crit_vals))
    print("Johansen trace stats:", trace_stats)

    print("Johansen 95% crit vals:", crit_vals)

    print(f"Johansen inferred rank: {rank}")

    weights = None
    spread = None

    if rank >= 1:

        # Beta columns are cointegrating vectors; take the first (associated with largest eigenvalue)
        beta = joh.evec  # shape (k, k)
        eigvals = joh.eig  # shape (k,)

        # Sort by eigenvalue desc to be explicit
        order = np.argsort(eigvals)[::-1]
        beta_sorted = beta[:, order]
        eig_sorted = eigvals[order]
        first_vec = beta_sorted[:, 0]

        # Normalize such that the weight of the last symbol is 1 (common in practice)
        if first_vec[-1] == 0:
            norm = np.max(np.abs(first_vec))
        else:
            norm = first_vec[-1]

        weights = first_vec / norm

        print("First Johansen eigenvector (portfolio weights):")

        for sym, w in zip(prices.columns, weights):
            print(f"  {sym:>12s}: {w:+.6f}")

        # Spread = P * weights
        spread = pd.Series(prices.values @ weights, index=prices.index, name="JohansenSpread")

        if plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(spread.index, spread.values, label="Johansen Spread")
            ax.axhline(0, linestyle="--")
            ax.set_title("Johansen Spread (1st eigenvector)")
            ax.legend()
            fig.tight_layout()

            plt.show()

        return rank, beta_sorted, eig_sorted, spread, weights

    return rank, joh.evec, joh.eig, weights, None
```

If any rank in the Johansen cointegration test is SIGNIFICANT, the script will also run the ADF and the KPSS tests on the Johansen spread, and, finally, save the strategy parameters (symbols, weights, and timeframe) on the table ‘strategy’, as mentioned in the previous section.

```
def StoreJohansenStrategy(db_path: str | Path, name: str, symbols: List[str],
                          weights: np.ndarray, timeframe: str) -> None:
    """
    Store Johansen eigenvector (portfolio weights) and symbols into the strategy table.
    Symbols and weights are stored as JSON arrays (text).
    """

    import json

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO strategy (name, symbols, weights, timeframe)
            VALUES (?, ?, ?, ?)
        """, (
            name,
            json.dumps(symbols),
            json.dumps([float(w) for w in weights]),
            timeframe
        ))
        conn.commit()

    logger.info("Stored Johansen strategy '%s' with symbols=%s timeframe=%s",
                name, symbols, timeframe)
```

For convenience, it will stop the execution and plot the graph at each relevant section (spreads, for example), so you can easily assess if the selected symbols are worth a more detailed evaluation, eventually changing their list order (for Engle-Granger) or the timeframe. The script is a tool for manual screening before we have a fully automated pipeline in the next step. Each task is encapsulated as an independent function, so you can easily import them into other scripts.

WARNING: This script is our system entry-point. Right after a new symbol list is defined here, you should insert it in the db-update Service. This will ensure your database will be updated at a calm pace, even if your EA is not running for any reason. This step must be done manually for now.

To run the script, call it from the command line with parameters. For example:

PS C:\\Users\\your\\path\\to\\JupyterNotebooks\\StatArb\\coint> python py/stat\_arb\_pipeline\_mt5.py --db $env:STATARB\_DB\_PATH --schema sql/schema-0.2.sql --symbols "MU,NVDA,MPWR,MCHP" --timeframes "D1" --bars 1000 --run

As you can see, we are using environment variables for the db path parameter. If everything goes well, you should see something like this when there is NO COINTEGRATION as per the Johansen test.

2025-09-03 15:35:24,588 \[INFO\] Database exists: True (C:\\Users\\your\\path\\to\\MQL5\\Files\\StatArb\\statarb-0.2.db)

2025-09-03 15:35:24,589 \[INFO\] Using existing DB at C:\\Users\\your\\path\\to\\MQL5\\Files\\StatArb\\statarb-0.2.db.

2025-09-03 15:35:24,605 \[INFO\] MetaTrader 5 initialized.

2025-09-03 15:35:24,613 \[INFO\] Downloaded 1000 bars for MU D1

2025-09-03 15:35:24,618 \[INFO\] Downloaded 1000 bars for NVDA D1

2025-09-03 15:35:24,625 \[INFO\] Downloaded 1000 bars for MPWR D1

2025-09-03 15:35:24,633 \[INFO\] Downloaded 1000 bars for MCHP D1

2025-09-03 15:35:24,997 \[INFO\] Stored 1000 rows for MU D1

2025-09-03 15:35:25,261 \[INFO\] Stored 1000 rows for NVDA D1

2025-09-03 15:35:25,514 \[INFO\] Stored 1000 rows for MPWR D1

2025-09-03 15:35:25,870 \[INFO\] Stored 1000 rows for MCHP D1

2025-09-03 15:35:25,871 \[INFO\] MetaTrader 5 shutdown.

2025-09-03 15:35:25,961 \[INFO\] ReadMarketData: D1 timeframe, 1000 rows, 4 symbols

2025-09-03 15:35:25,964 \[INFO\] Most correlated pair: MPWR ~ NVDA (r=0.8335)

**Most correlated pair:** MPWR ~ NVDA (r=0.8335)

**Most cointegrated pair** (Engle-Granger): MPWR ~ MU (p=0.010558, beta=5.4940)

ADF & KPSS on Engle-Granger spread:

ADF: stat=-3.8778, p=0.002204, lags=16, nobs=983, crit={'1%': np.float64(-3.4370198458812156), '5%': np.float64(-2.864484708707697), '10%': np.float64(-2.568337912084273)}

C:\\Users\\your\\path\\to\\JupyterNotebooks\\StatArb\\coint\\py\\stat\_arb\_pipeline\_mt5.py:321: InterpolationWarning: The test statistic is outside of the range of p-values available in the look-up table. The actual p-value is smaller than the p-value returned.

stat, pvalue, lags, crit = kpss(series, regression=regression, nlags="auto")

KPSS(c): stat=0.9279, p=0.010000, lags=19, crit={'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}

Johansen test on all provided symbols:

Johansen trace stats: \[34.26486877 18.91156161  8.04865462  0.11018312\]

Johansen 95% crit vals: \[47.8545 29.7961 15.4943  3.8415\]

**Johansen inferred rank: 0**

As said above, when there is a SIGNIFICANT Johansen rank (>=1), meaning, when there is at least one cointegrated vector between the selected symbols, the script will store the first eigenvector values (portfolio weights) in the database in the ‘strategy table’. It will also run the ADF and KPSS tests on the spread.

PS C:\\Users\\your\\path\\to\\JupyterNotebooks\\StatArb\\coint> python py/stat\_arb\_pipeline\_mt5.py --db $env:STATARB\_DB\_PATH --schema sql/schema-0.2.sql --symbols "MU,NVDA,MPWR,MCHP" --timeframes "H4" --bars 90 --run

In this case, you should see something like this on your Windows PowerShell or Terminal:

2025-09-03 15:50:43,168 \[INFO\] Database exists: True (C:\\Users\\your\\path\\to\\MQL5\\Files\\StatArb\\statarb-0.2.db)

2025-09-03 15:50:43,169 \[INFO\] Using existing DB at C:\\Users\\your\\path\\to\\Files\\StatArb\\statarb-0.2.db.

2025-09-03 15:50:43,188 \[INFO\] MetaTrader 5 initialized.

2025-09-03 15:50:44,297 \[INFO\] Downloaded 90 bars for MU H4

2025-09-03 15:50:44,357 \[INFO\] Downloaded 90 bars for NVDA H4

2025-09-03 15:50:46,322 \[INFO\] Downloaded 90 bars for MPWR H4

2025-09-03 15:50:46,732 \[INFO\] Downloaded 90 bars for MCHP H4

2025-09-03 15:50:49,148 \[INFO\] Stored 90 rows for MU H4

2025-09-03 15:50:49,160 \[INFO\] Stored 90 rows for NVDA H4

2025-09-03 15:50:49,171 \[INFO\] Stored 90 rows for MPWR H4

2025-09-03 15:50:49,276 \[INFO\] Stored 90 rows for MCHP H4

2025-09-03 15:50:49,277 \[INFO\] MetaTrader 5 shutdown.

2025-09-03 15:50:49,617 \[INFO\] ReadMarketData: H4 timeframe, 90 rows, 4 symbols

2025-09-03 15:50:49,622 \[INFO\] Most correlated pair: MPWR ~ NVDA (r=0.6115)

**Most correlated pair**: MPWR ~ NVDA (r=0.6115)

**Most cointegrated pair** (Engle-Granger): MU ~ NVDA (p=0.072055, beta=-0.2391)

ADF & KPSS on Engle-Granger spread:

ADF: stat=-3.1693, p=0.021831, lags=0, nobs=89, crit={'1%': np.float64(-3.506057133647011), '5%': np.float64(-2.8946066061911946), '10%': np.float64(-2.5844100201994697)}

C:\\Users\\your\\path\\to\\JupyterNotebooks\\StatArb\\coint\\py\\stat\_arb\_pipeline\_mt5.py:321: InterpolationWarning: The test statistic is outside of the range of p-values available in the look-up table. The actual p-value is greater than the p-value returned.

stat, pvalue, lags, crit = kpss(series, regression=regression, nlags="auto")

KPSS(c): stat=0.1793, p=0.100000, lags=5, crit={'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}

Johansen test on all provided symbols:

Johansen trace stats: \[63.23695088 32.35462044 16.63904745  5.4295422 \]

Johansen 95% crit vals: \[47.8545 29.7961 15.4943  3.8415\]

**Johansen inferred rank: 4**

First Johansen eigenvector (portfolio weights):

          MCHP: +0.273652

          MPWR: -0.148092

            MU: +2.256498

          NVDA: +1.000000

ADF & KPSS on Johansen spread:

ADF: stat=-4.1427, p=0.000823, lags=1, nobs=88, crit={'1%': np.float64(-3.506944401824286), '5%': np.float64(-2.894989819214876), '10%': np.float64(-2.584614550619835)}

KPSS(c): stat=0.4009, p=0.076784, lags=4, crit={'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}

2025-09-03 15:50:50,051

\[INFO\] **Stored Johansen strategy 'Nasdaq\_NVDA\_Coint' with symbols=\['MCHP', 'MPWR', 'MU', 'NVDA'\] timeframe=H4**

**What about the interpolation warnings above?**

Those interpolation warnings are related to the KPSS stationarity test on the Engle-Granger spread. They are informing us that we got extreme results, outside of the test lookup table. The warnings are simply informing us that our Engle-Granger spread is

- strongly stationary (p-value greater than the p-value returned)
- or strongly non-stationary (a p-value smaller than the p-value returned)

We can safely ignore these warnings for now, while we are dealing with our model parameters' real-time updates. We will see them in detail in the next step, when we will deal with our assets screening.

The MQL5 side

We’ve left only the EA event handlers in the main file. All other functions were moved to the Nasdaq\_NVDA\_Coint.mqh header file.

Previously, the strategy parameters were hard-coded. Now, they are global variables.

```
// Global variables
string symbols[] = {}; // Asset symbols
double weights[] = {}; // Portfolio weights
ENUM_TIMEFRAMES timeframe = PERIOD_CURRENT; // Strategy cointegrated timeframe
```

These global variables are checked right in the OnInit() event handler.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

int OnInit()
  {
   ResetLastError();

// Check if all symbols are available
   for(int i = 0; i < ArraySize(symbols); i++)
     {
      if(!SymbolSelect(symbols[i], true))
        {
         Print("Error: Symbol ", symbols[i], " not found!");
         return(INIT_FAILED);
        }
     }
// Initialize spread buffer
   ArrayResize(spreadBuffer, InpLookbackPeriod);

// Set a timer for spread, mean, and stdev calculations
   EventSetTimer(InpUpdateFreq * 60); // min one minute

// Load strategy parameters from database
   if(!LoadStrategyFromDB(InpDbFilename,
                                InpStrategyName,
                                symbols,
                                weights,
                                timeframe))
     {
      // Handle error - maybe use default values
      printf("Error at " + __FUNCTION__ + " %s ",
             getUninitReasonText(GetLastError()));
      return INIT_FAILED;
     }
   return(INIT_SUCCEEDED);
  }
```

The function LoadStrategyFromDB() is in the header file.

```
//+------------------------------------------------------------------+
//|          Load strategy parameter from database                   |
//+------------------------------------------------------------------+

bool LoadStrategyFromDB(string db_name,
                        string strat_name,
                        string &symbols_arr[],
                        double &weights_arr[],
                        ENUM_TIMEFRAMES &tf)
  {

// Open the database (Metatrader's integrated SQLite)
   int db = DatabaseOpen(db_name, DATABASE_OPEN_READONLY);
   if(db == INVALID_HANDLE)
     {
      Print("Failed to open database: %s", GetLastError());
      return false;
     }
// Prepare the SQL query with json_extract
   string query = StringFormat(
                     "SELECT "
                     "json_extract(symbols, '$') as symbols_json, "
                     "json_extract(weights, '$') as weights_json, "
                     "timeframe "
                     "FROM strategy WHERE name = '%s'",
                     strat_name
                  );

// Execute the query
   int result = DatabasePrepare(db, query);

   if(result <= 0)
     {
      Print("Failed to prepare query: ", GetLastError());
      DatabaseClose(db);
      return false;
     }

// Read the results
   if(DatabaseRead(result))
     {
      // Get symbols_arr JSON array and parse it
      string symbolsJson;
      DatabaseColumnText(result, 0, symbolsJson);
      ParseJsonArray(symbolsJson, symbols_arr);

      // Get weights_arr JSON array and parse it
      string weightsJson;
      DatabaseColumnText(result, 1, weightsJson);
      ParseJsonDoubleArray(weightsJson, weights_arr);

      // Get tf string and convert to ENUM_TIMEFRAMES
      string timeframeStr;
      DatabaseColumnText(result, 2, timeframeStr);
      tf = StringToTimeframe(timeframeStr);

      Print("Successfully loaded strategy: ", strat_name);

      Print("Symbols JSON: ", symbolsJson);
      Print("Weights JSON: ", weightsJson);
      Print("Timeframe: ", timeframeStr);
     }
   else
     {
      Print("Strategy not found: ", strat_name);

      DatabaseFinalize(result);
      DatabaseClose(db);
      return false;
     }
// Clean up

   DatabaseFinalize(result);
   DatabaseClose(db);
   return true;
  }
```

As said above, we will read the symbols and the portfolio weights arrays with SQLite’s json\_extract() function. It has the following signature:

json\_extract(json, path)

Where:

- ‘json’ is the text column or string containing the JSON data
- ‘path’ is a JSON path expression that specifies the location of the value to extract within the JSON. It typically starts with $ to represent the root of the JSON document, followed by dot notation for object keys ($.key) or bracket notation for array indices ($\[index\]). This last form is what we are using here to read the array data.

The two helper functions parse the JSON arrays. These functions handle the JSON format by removing brackets and quotes, then splitting by commas.

ParseJsonArray()

parses JSON string arrays like \["MU", "NVDA", "MPWR", "MCHP"\]

```
// Helper function to parse JSON array of strings
void ParseJsonArray(string json, string &array[])
  {
// Remove brackets and quotes from JSON array
   string cleaned = StringSubstr(json, 1, StringLen(json) - 2); // Remove [ and ]
   StringReplace(cleaned, "\"", ""); // Remove quotes

// Split by commas
   StringSplit(cleaned, ',', array);

// Trim whitespace from each element
   for(int i = 0; i < ArraySize(array); i++)
     {
      array[i] = StringTrim(array[i]);
     }
  }
```

ParseJsonDoubleArray()

parses JSON number arrays like \[2.699439, 1.000000, -1.877447, -2.505294\]

```
// Helper function to parse JSON array of doubles
void ParseJsonDoubleArray(string json, double &array[])
  {
// Remove brackets from JSON array
   string cleaned = StringSubstr(json, 1, StringLen(json) - 2); // Remove [ and ]

// Split by commas
   string stringArray[];
   int count = StringSplit(cleaned, ',', stringArray);

// Convert to double array
   ArrayResize(array, count);

   for(int i = 0; i < count; i++)
     {
      array[i] = StringToDouble(StringTrim(stringArray[i]));
     }
  }
```

Added the

StringTrim()

function to remove any whitespace that might be present in the JSON values.

```
// Helper function to trim whitespace
string StringTrim(string str)
  {

// Trim leading whitespace
   while(StringLen(str) > 0 && str[0] == ' ')
     {
      str = StringSubstr(str, 1);
     }

// Trim trailing whitespace
   while(StringLen(str) > 0 && str[StringLen(str) - 1] == ' ')
     {
      str = StringSubstr(str, 0, StringLen(str) - 1);
     }

   return str;
  }
```

There are JSON libraries on the MQL5 Codebase. If your use of JSON objects gets more intensive, maybe it is worth looking for them and evaluating the benefits. For this simple case, I think the benefits are not enough to carry an external dependency.

The helper function

StringToTimeframe()

converts the database timeframe string to the appropriate MQL5 ENUM\_TIMEFRAMES value.

```
// Helper function to convert tf string to ENUM_TIMEFRAMES
ENUM_TIMEFRAMES StringToTimeframe(string tfStr)
  {
   if(tfStr == "M1")
      return PERIOD_M1;
   if(tfStr == "M2")
      return PERIOD_M2;
.
.
.
   if(tfStr == "MN1")
      return PERIOD_MN1;
   return PERIOD_CURRENT; // Default if not found
  }
```

Besides checking the strategy parameter in the OnInit(), called when the EA is launched or restarted, we also call the same function at a regular time interval in the OnTimer() event handler. This time interval is a user input parameter.

```
// Set a timer for spread, mean, and stdev calculations
// and strategy parameters update (check DB)
   EventSetTimer(InpUpdateFreq * 60); // min one minute

void OnTimer(void)
  {
   ResetLastError();
// Wrapper around LoadStrategyFromDB: for clarity
   if(!UpdateModelParams(InpDbFilename,
                         InpStrategyName,
                         symbols,
                         weights,
                         timeframe))
     {
      printf("%s failed: %s", __FUNCTION__, GetLastError());
     }
```

UpdateModelParams() is just a wrapper around LoadStrategyFromDB().

```
//+------------------------------------------------------------------+
//|          Wrapper to load strategy from db                        |
//+------------------------------------------------------------------+

// Load strategy parameters from database
bool UpdateModelParams(string db_name,
                       string strat_name,
                       string &symbols_arr[],
                       double &weights_arr[],
                       ENUM_TIMEFRAMES tf)
  {
   return LoadStrategyFromDB(db_name, strat_name, symbols_arr, weights_arr, tf);
  }
```

This makes the function’s purpose clearer.

### Conclusion

A well-balanced basket of cointegrated stocks for mean-reversion strategies must have its portfolio weights updated continuously. In this article, we saw that we can always have up-to-date strategy parameters, including the basket symbols themselves.

We described how we can use the MetaTrader 5 integrated SQLite database as a kind of interface between our data analysis and our trading environment, while keeping these two activities decoupled.

To make things easier for those readers who may not be following the series, we provide a unified Python script with a complete statistical arbitrage pipeline for cointegrated stocks. The script runs correlation, cointegration, and stationarity tests in a single run, and lets the database with the up-to-date strategy parameters ready to be read by the Expert Advisor.

References

Daniel P. Palomar (2025). [Portfolio Optimization: Theory and Application.](https://www.mql5.com/go?link=https://portfoliooptimizationbook.com/portfolio-optimization-book.pdf "https://portfoliooptimizationbook.com/portfolio-optimization-book.pdf") Cambridge University Press.

| Filename | Description |
| --- | --- |
| StatArb/db-setup.mq5 | MQL5 Script to create and initialize the SQLite database by reading the schema-0.2.sql schema file. |
| StatArb/db-update-statarb-0.2.mq5 | MQL5 Service to update the SQLite database with the most recent closed price bars. |
| StatArb/Nasdaq\_NVDA\_Coint.mql5 | This file contains the sample Expert Advisor. |
| StatArb/Nasdaq\_NVDA\_Coint.mqh | This file contains the sample Expert Advisor header file. |
| StatArb/schema-0.2.sql | SQL schema file (DDL) to initialize the database (generate tables, fields, and constraints). |
| stat\_arb\_pipeline\_mt5.py | This Python script runs the complete stat arb pipeline for cointegrated stocks based on Pearson correlation test, Johansen cointegration test, and ADF plus KPSS stationarity tests. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19428.zip "Download all attachments in the single ZIP archive")

[mql5-article-files-19428.zip](https://www.mql5.com/en/articles/download/19428/mql5-article-files-19428.zip "Download mql5-article-files-19428.zip")(18.22 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Statistical Arbitrage Through Cointegrated Stocks (Part 10): Detecting Structural Breaks](https://www.mql5.com/en/articles/20946)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://www.mql5.com/en/articles/20657)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 8): Rolling Windows Eigenvector Comparison for Portfolio Rebalancing](https://www.mql5.com/en/articles/20485)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening](https://www.mql5.com/en/articles/19626)

**[Go to discussion](https://www.mql5.com/en/forum/495182)**

![Developing a Custom Market Sentiment Indicator](https://c.mql5.com/2/168/19422-developing-a-custom-market-logo.png)[Developing a Custom Market Sentiment Indicator](https://www.mql5.com/en/articles/19422)

In this article we are developing a custom market sentiment indicator to classify conditions into bullish, bearish, risk-on, risk-off, or neutral. Using multi-timeframe, the indicator can provide traders with a clearer perspective of overall market bias and short-term confirmations.

![Elevate Your Trading With Smart Money Concepts (SMC): OB, BOS, and FVG](https://c.mql5.com/2/168/16340-elevate-your-trading-with-smart-logo.png)[Elevate Your Trading With Smart Money Concepts (SMC): OB, BOS, and FVG](https://www.mql5.com/en/articles/16340)

Elevate your trading with Smart Money Concepts (SMC) by combining Order Blocks (OB), Break of Structure (BOS), and Fair Value Gaps (FVG) into one powerful EA. Choose automatic strategy execution or focus on any individual SMC concept for flexible and precise trading.

![Self Optimizing Expert Advisors in MQL5 (Part 14): Viewing Data Transformations as Tuning Parameters of Our Feedback Controller](https://c.mql5.com/2/168/19382-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 14): Viewing Data Transformations as Tuning Parameters of Our Feedback Controller](https://www.mql5.com/en/articles/19382)

Preprocessing is a powerful yet quickly overlooked tuning parameter. It lives in the shadows of its bigger brothers: optimizers and shiny model architectures. Small percentage improvements here can have disproportionately large, compounding effects on profitability and risk. Too often, this largely unexplored science is boiled down to a simple routine, seen only as a means to an end, when in reality it is where signal can be directly amplified, or just as easily destroyed.

![Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://c.mql5.com/2/130/Moving_to_MQL5_Algo_Forge_Part_LOGO__3.png)[Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)

When working on projects in MetaEditor, developers often face the need to manage code versions. MetaQuotes recently announced migration to GIT and the launch of MQL5 Algo Forge with code versioning and collaboration capabilities. In this article, we will discuss how to use the new and previously existing tools more efficiently.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/19428&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083098442319795452)

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