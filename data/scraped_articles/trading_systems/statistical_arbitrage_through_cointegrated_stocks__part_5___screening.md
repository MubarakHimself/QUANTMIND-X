---
title: Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening
url: https://www.mql5.com/en/articles/19626
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:46:02.190370
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/19626&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083094546784457960)

MetaTrader 5 / Trading systems


### Introduction

We started this series using a heuristic to find some stocks from the semiconductor industry that - at the time of writing - were highly correlated with NVDA (Nvidia Corporation). Then, we filtered down to a couple of cointegrated stocks that we used in our database setup and respective examples. Those stocks were our “portfolio”, and it was good for demonstration purposes and to help in understanding the process.

Now that we have implemented a Metatrader 5 Service to update our database continuously, and our Expert Advisor is also able to update itself in real-time without interruptions, it is time to begin building a real portfolio of cointegrated stocks. It is time to start screening.

“Screening refers to the process of filtering a large universe of stocks or other financial instruments to identify those that meet specific criteria or parameters set by the investor or trader. This process is crucial for narrowing down the vast array of available options and focusing on those that align with a particular investment strategy or trading objective.” ( [Investing.com](https://www.mql5.com/go?link=https://www.investing.com/academy/stocks/stock-screener-health-and-risk-metrics/ "https://www.investing.com/academy/stocks/stock-screener-health-and-risk-metrics/"))

Our strategy is a mean-reversion one, a kind of statistical arbitrage that takes Nasdaq stocks cointegrated with Nvidia, then buys and sells them simultaneously according to their portfolio weights, to seek market neutrality. Our vast array of available options includes all Nasdaq stocks. The specific criteria to be met are:

1. The cointegration strength as indicated by the Engle-Granger and the Johansen tests
2. The stability of the portfolio weights
3. The quality of the spread stationarity, as indicated by the ADF and the KPSS tests
4. The asset liquidity

These four criteria are enough for us to build a scoring system. A scoring system should give us a trading edge that comes from identifying groups of stocks whose prices move together in a stable and predictable way in the long run. We need a scoring system because not every pair or group serves our purposes. If we simply test every possible combination of stocks, with hundreds or thousands of stocks, the number of potential pairs and baskets explodes. We must not forget that we are developing this statistical arbitrage framework for the average retail trader, with a consumer notebook and a regular network bandwidth. If we start testing every possible combination, the process becomes computationally expensive from the beginning. By having a scoring system, we avoid risking our money in pairs or groups that are only correlated in the short run, or that are not tradeable due to low liquidity or transaction costs.

Maybe the better analogy for a screening process with a scoring system is that of a funnel: we start broad and gradually eliminate unsuitable candidates, first excluding by sector and industry similarity, then scoring by correlation, cointegration, and stationarity. Finally, we eliminate those that are not tradeable according to our risk/money management. This process increases the likelihood of finding baskets that are not only statistically significant but also economically meaningful and tradable.

Screening and scoring transform the huge list of all stocks available in our broker into a focused set of opportunities. It saves time and resources for the equally critical steps of backtesting and demo account trading.

Fig. 1 illustrates the flow diagram of our screening process.

![Fig. 1 - Screening process conceptual flow diagram](https://c.mql5.com/2/170/screening.png)

Fig. 1. Screening process conceptual flow diagram

### Defining the Initial Universe

The last of our four criteria above - asset liquidity - is usually the first to be used when starting from scratch. Liquidity will impact not only the transaction costs (e.g., via spread) but the tradeability itself, depending on the strategy. Since our strategy is dependent on the stock availability for short operations, liquidity is an eliminatory criterion.

We could begin with a major index such as the S&P 500 or Nasdaq-100, which already filters for liquidity and market relevance. Or we could build a custom universe using criteria like minimum market capitalization, average daily volume, or narrow bid–ask spreads. Both ways would be valid to define this initial universe. But my main concern here is that you, the reader interested in replicating the process using the sample code attached, can have the easiest and most straightforward experience. So, I’m using the MetaQuotes-Demo server as a data source, because I suppose it is freely available to all Metatrader 5 users. It happens that, as far as I can see, the MetaQuotes-Demo server does not have a separate path (see below) for the stocks listed in the S&P 500 or those listed in the Nasdaq-100. Instead, it offers a path for the whole of the Nasdaq stocks.

![Fig. 2 - Metatrader 5 Symbols window with the Nasdaq/Stock path highlighted](https://c.mql5.com/2/170/Capture_mt5_symbols_dialog_nasdaq_path.PNG)

Fig. 2. Metatrader 5 Symbols window with the Nasdaq/Stock path highlighted

If we choose to start with only the Nasdaq-100 stocks, we would need to build a curated list of stocks that would require maintenance when the index composition is updated and/or its weights are rebalanced. That would make our examples out of date very soon. Thus, for our purposes here, the Nasdaq/Stock path makes an ideal choice as a starting point, our initial universe.

It amounts to more than six thousand symbols!

![Fig. 3 - Number of symbols on the MetaQuotes-Demo server by group](https://c.mql5.com/2/170/Capture_metaquotes_server_full_symbol_tree.PNG)

Fig. 3. Number of symbols on the MetaQuotes-Demo server by group

### Sector and industry filtering

Once we have the initial universe defined, the next step is to apply our market knowledge, experience, and common sense to narrow the field. These will tell us that stocks from the same sector or industry often share common drivers, such as being part of the same supply chain or being dependent on the central bank's interest rates. These common drivers increase the likelihood that their prices move together in a stable way. For example, banks may respond similarly to interest rate changes, while semiconductor companies, which is our case here, may be influenced by global chip demand.

This almost intuitive market knowledge and common sense were the basis of our heuristics when choosing that first basket with Nvidia cointegrated stocks.

We want to have the filtered stocks at hand for our data analysis, so we will store them with the relevant metadata in our ‘symbol’ table. But first, let’s include the column ‘source’ in our table, because both the Metatrader 5 path and

the symbol metadata may be, and probably will be, different when changing brokers/servers. We will need to filter the market data by source when analysing data from different brokers.

```
ALTER TABLE symbol ADD source TEXT CHECK(LENGTH(source) <= 50) DEFAULT ('MetaQuotes') NOT NULL;
```

And an MQL5 Script (SymbolImporter.mql5) to import the symbols from the specific path on the database.

![Fig. 4 - Metatrader 5 SymbolImporter dialog with params](https://c.mql5.com/2/170/Capture_mt5_SymbolImporter_dialog_params.PNG)

Fig. 4. Metatrader 5 SymbolImporter dialog with params

It will fill the gaps we’ve left in the database when starting, namely, asset type, exchange, industry, and sector. Also, it will add the source of the data in the newly created field above.

```
//+------------------------------------------------------------------+
//| Insert a symbol into the database                                |
//+------------------------------------------------------------------+
bool InsertSymbol(int db_handle, string ticker, string source)
  {
   ResetLastError();
// Get symbol information
   string exchange = SymbolInfoString(ticker, SYMBOL_EXCHANGE);
   string asset_type = GetAssetType(ticker);
   string sector = SymbolInfoString(ticker, SYMBOL_SECTOR_NAME);
   string industry = SymbolInfoString(ticker, SYMBOL_INDUSTRY_NAME);
   string currency = SymbolInfoString(ticker, SYMBOL_CURRENCY_BASE);
   if(currency == "")
      currency = SymbolInfoString(ticker, SYMBOL_CURRENCY_PROFIT);
// Prepare SQL insert statement (symbol_id is auto-generated by SQLite)
   string req = StringFormat(
                   "INSERT INTO symbol(ticker, exchange, asset_type, sector, industry, currency, source)"
                   " VALUES( '%s', '%s', '%s', '%s', '%s', '%s', '%s')",
                   ticker, exchange, asset_type, sector, industry, currency, source);
   if(!DatabaseExecute(db_handle, req))
     {
      printf("Failed to insert symbol: %d", GetLastError());
      return false;
     }
   return true;
  }
```

It will try to get the asset type from the path.

```
//+------------------------------------------------------------------+
//| Determine asset type based on symbol characteristics             |
//+------------------------------------------------------------------+

string GetAssetType(string symbol_name)
  {
   ResetLastError();
// Simple asset type detection - you may want to enhance this
   string description = SymbolInfoString(symbol_name, SYMBOL_DESCRIPTION);
   string path = SymbolInfoString(symbol_name, SYMBOL_PATH);
   if(StringFind(path, "Forex") != -1)
      return "Forex";
   if(StringFind(path, "Stock") != -1)
      return "Stock";
   if(StringFind(path, "Index") != -1)
      return "Index";
   if(StringFind(path, "Future") != -1)
      return "Future";
   if(StringFind(path, "CFD") != -1)
      return "CFD";
   if(StringFind(path, "Crypto") != -1)
      return "Cryptocurrency";
// Fallback based on symbol name patterns
   if(StringLen(symbol_name) == 6 && StringFind(symbol_name, "USD") != -1)
      return "Forex";
   if(StringFind(symbol_name, ".", 0) != -1)
      return "Stock";
   return "Other";
  }
```

If everything goes well, you should see something like this in your Experts tab log.

![Fig. 5 - Metatrader 5 Experts tab showing SymbolImport log after successful imports](https://c.mql5.com/2/170/Capture_mt5_experts_log_SymbolImporter_success.PNG)

Fig. 5. Metatrader 5 Experts tab showing SymbolImport log after successful imports

And if you check your database, it should contain all stocks listed on Nasdaq and available on this server (MetaQuotes-Demo, in this case).

![Fig. 6 - Metaeditor database interface showing a sample of Nasdaq stocks metadata](https://c.mql5.com/2/170/Capture_metaeditor_db_nasdaq.PNG)

Fig. 6.  Metaeditor database interface showing a sample of Nasdaq stocks metadata

We can check the total stock symbols imported with a SELECT count() statement.

![Fig. 7 - Metaeditor database interface showing the total of Nasdaq stocks imported into the SQLite database](https://c.mql5.com/2/170/Capture_metaeditor_db_nasdaq_stock_count.PNG)

Fig. 7. Metaeditor database interface showing the total of Nasdaq stocks imported into the SQLite database

Almost seven hundred of them are from the technology sector.

![Fig. 8 - Metaeditor database interface showing the total of Nasdaq stocks from the technology sector](https://c.mql5.com/2/170/Capture_metaeditor_db_nasdaq_stock_count_tech_sector.PNG)

Fig. 8. Metaeditor database interface showing the total of Nasdaq stocks from the technology sector

But only a few more than sixty are from the semiconductor industry.

![Fig. 9 - Metaeditor database interface showing the total of Nasdaq stocks from the semiconductor industry](https://c.mql5.com/2/170/Capture_metaeditor_db_nasdaq_stock_count_semiconductor_industry.PNG)

Fig. 9. Metaeditor database interface showing the total of Nasdaq stocks from the semiconductor industry

These sixty-three Nasdaq stocks from the semiconductor industry are those that deserve our focus. Later, we can expand into new opportunities, exploring sectors and industries' intersections, but now what we need is to focus on those that are more inclined to fluctuate together. They are those who deserve our effort in running continuous statistical tests.

### Correlation analysis

Now that we have narrowed down our initial universe by sector and industry, the next step is to check whether the stocks actually move together in practice. The simplest way is to calculate correlation coefficients. We’ve been using the Pearson correlation since the start of this series, and we will stick with it. The Pearson correlation is fast to compute and easy to interpret, making it a practical tool for screening.

We’ve talked a lot about the Pearson correlation test in the second and third articles of this series. There is no sense in repeating that information here. You are invited to check those articles if you are not familiar with this topic. Only one remark to keep in mind:

Correlation IS NOT the same as cointegration. Stock prices are typically non-stationary time series, and non-stationary time series can be highly correlated in the short term, but not be cointegrated in the long run.

The Spearman correlation test is very common too, and has a clear and specific use case as a complementary check, but we will not deal with it for now. Instead, we’ll see the Spearman correlation later as an improvement for those specific use-cases.

We will store the correlation test results in a dedicated table in our database, a table unsurprisingly called ‘corr\_pearson’.

```
-- corr_pearson definition

CREATE TABLE corr_pearson (
        tstamp INTEGER NOT NULL,
        ref_ticker TEXT CHECK(LENGTH(ref_ticker) <= 10) NOT NULL,
        corr_ticker TEXT CHECK(LENGTH(corr_ticker) <= 10) NOT NULL,
        timeframe TEXT CHECK(LENGTH(timeframe) <= 10) NOT NULL,
        lookback INTEGER NOT NULL,
        coefficient REAL NOT NULL,
CONSTRAINT corr_pearson_pk PRIMARY KEY(tstamp, ref_ticker, corr_ticker)
) STRICT;
```

Technically, storing the results of these preliminary calculations would not be necessary. But, in statistical arbitrage, DATA is the name of the game. We are calculating correlation coefficients between semiconductor stocks and Nvidia Corp., on the H4 timeframe, with a 180-day lookback period. But soon, we will be calculating it for other timeframes and lookback periods. Probably, we will repeat this process for other base symbols beyond NVDA. As our data analysis evolves, we will have tons of processed data for Pearson correlation, for Engle-Granger and Johansen cointegration, maybe for other intermediary calculations.

You got the idea: all of these calculations output invaluable information to be preserved, cross-checked, and reused later. So, if we are serious about statistical arbitrage, we should adopt the habit of not accepting ephemeral data and not discarding every intermediary calculation's results. Let’s store everything that may seem useful in the future. And, if you are not convinced yet, just think about these two words: MACHINE LEARNING. We will arrive there at some point, for sure. 😀

![Fig. 10 - Metaeditor database interface showing the ‘corr_pearson’ table fields](https://c.mql5.com/2/170/Capture_metaeditor_db_corr_pearson_all.PNG)

Fig. 10. Metaeditor database interface showing the ‘corr\_pearson’ table fields

You will note that we have again a composite primary key, now using the insertion timestamp, the reference ticker, and the correlated ticker. This allows us to have the exact point in time when the correlation coefficient was calculated for this specific pair of symbols. This way, we can safely calculate the correlation for all symbols again, but now with a different timeframe.

![Fig. 11 - Metaeditor database interface showing the ‘corr_pearson’ with two different timeframes sorted](https://c.mql5.com/2/170/Capture_metaeditor_db_corr_pearson_all_two_timeframes.PNG)

Fig. 11. Metaeditor database interface showing the ‘corr\_pearson’ with two different timeframes sorted

We can easily see from Fig. 11 above that, in the same lookback period, for some symbols the correlation increases from the H4 to D1 timeframe, while for others the correlation decreases. Expand this to several timeframes and lookback periods, and we start having a powerful analysis tool. Moreover, this data will be, in a sense, unique to you, because it will be you who will choose which symbols to analyse and with which parameters. The conclusions you can arrive at from this data analysis may or may not give you an edge on trading, but this is the idea here.

Later, we can have the evolution of the correlation for each symbol with NVDA over time. Here we have the evolution of the correlation between Broadcom Inc. (AVGO) with NVDA, from one year (252 lookback) to one month (30 lookback), passing through 180, 120, 90, and 60 lookback periods. Since we are ordering by coefficient, the results clearly show that D1 is the preferred timeframe for this pair; that the correlation drops as the timeframe goes shorter; and that the correlation drops dramatically below 90 lookback periods. That is a lot of information, and we are just on the surface.

Note: Except for the year lookback period (252), I’m using rounded numbers for monthly lookback periods for the ease of visualization here.

![Fig. 12 - Metaeditor database interface showing the ‘corr_pearson’ evolution between Broadcom Inc. (AVGO) and Nvidia Corp. (NVDA) from year to month lookback periods in a daily timeframe](https://c.mql5.com/2/170/Capture_metaeditor_db_corr_pearson_evolution_year_to_month.PNG)

Fig. 12. Metaeditor database interface showing the ‘corr\_pearson’ evolution between Broadcom Inc. (AVGO) and Nvidia Corp. (NVDA) from year to month lookback periods in a daily timeframe

Can you spot how valuable this simple correlation table can be? As a simple example, I excluded the H4 timeframe, keeping only daily correlations, and plotted it.

![Fig. 13 - Plot of Pearson correlation evolution between Broadcom Inc. (AVGO) and Nvidia Corp. (NVDA) from year to month lookback periods in a daily timeframe](https://c.mql5.com/2/170/plot_corr_pearson_NVDA_AVGO_year_to_month_D1_H4.png)

Fig. 13. Plot of Pearson correlation evolution between Broadcom Inc. (AVGO) and Nvidia Corp. (NVDA) from year to month lookback periods in a daily timeframe

Although it is common practice to use correlation as a preliminary filter, it says nothing about the cointegration. Thus, we cannot assume that those symbols with low correlation with Nvidia will not be cointegrated. So what is the purpose of testing for correlation? It helps in understanding fluctuation patterns over time. But if correlation is not a guarantee of cointegration, how can we use it in practice? By calculating correlation in short timeframes and short lookback periods, while at the same time testing for cointegration in long timeframes and lookback periods. Note that short and long here are relative to each other. As always in trading, even in statistically based trading, there is no recipe. You should test and experiment, backtest and evaluate, to find the better combination between the correlation test and the cointegration test parameters for each specific pair or group of assets. Use your market knowledge, common sense, and computational power wisely. Rinse and repeat.

The script itself is now structured as a specialized Python class.

![Fig. 14 - Screenshot from one of the Python scripts editor outlines depicting its class methods](https://c.mql5.com/2/170/Capture_python_script_pearson_to_db_functions.PNG)

Fig. 14. Screenshot from one of the Python scripts editor outlines depicting its class methods

Now, let’s see how we can combine the cointegration tests in our screening process.

### Cointegration tests

Engle-Granger

The Engle-Granger test is simple and lightweight. It was the first cointegration test to become popular among quant traders in the 1980s and still is, even after the rise of the Johansen test in the next decade (see below).  As said above, we talked in detail about the cointegration tests in the second and third parts of this series, so we will not repeat ourselves here to save your time. Now we will focus on their implementation and use in the screening process. If you are new to this topic, please refer to those two articles.

Just remember that, for two assets (symbols), the usual approach is the Engle-Granger test. It checks if the residuals of a linear regression between two price series are stationary. But it is limited to only one cointegration relationship, so it only works pairwise.

As we did for Pearson correlation, we will store the Engle-Granger cointegration test results in a dedicated table in our database, ‘coint\_eg’.

```
-- coint_eg definition

CREATE TABLE coint_eg (
        tstamp INTEGER NOT NULL,
        ref_ticker TEXT CHECK(LENGTH(ref_ticker) <= 10) NOT NULL,
        coint_ticker TEXT CHECK(LENGTH(coint_ticker) <= 10) NOT NULL,
        timeframe TEXT CHECK(LENGTH(timeframe) <= 10) NOT NULL,
        lookback INTEGER NOT NULL,
        pvalue REAL NOT NULL,
        test_stat REAL NOT NULL,
        crit_val_1 REAL NOT NULL,
        crit_val_5 REAL NOT NULL,
        crit_val_10 REAL NOT NULL,
        hedge_ratio REAL,
        is_coint INTEGER NOT NULL,
CONSTRAINT coint_eg_pk PRIMARY KEY(tstamp, ref_ticker, coint_ticker)
) STRICT;
```

The ‘is\_coint’ field is a BOOLEAN, but SQLite doesn’t have a separate Boolean data type. Instead, it uses INTEGER (or INT) to store boolean values as 1 or 0, accepting TRUE or FALSE as input.

“Unlike most other SQL implementations, SQLite does not have a separate BOOLEAN data type. Instead, TRUE and FALSE are (normally) represented as integers 1 and 0, respectively.” ( [SQLite docs](https://www.mql5.com/go?link=https://www.sqlite.org/quirks.html%23no_separate_boolean_datatype "https://www.sqlite.org/quirks.html#no_separate_boolean_datatype"))

![Fig. 15 - Metaeditor database interface showing the ‘coint_eg’ table fields](https://c.mql5.com/2/170/Capture_metaeditor_db_coint_eg_all.PNG)

Fig. 15. Metaeditor database interface showing the ‘coint\_eg’ table fields

At the time of writing, among all sixty-two symbols, only Aeluma Inc. (ALMU) is cointegrated with Nvidia Corp. (NVDA) on the daily timeframe for the 365 lookback period.

![Fig. 16 - Metaeditor database interface showing the ‘coint_eg’ table fields with one NVDA cointegrated stock (ALMU)](https://c.mql5.com/2/170/Capture_metaeditor_db_coint_eg_is_coint_true.PNG)

Fig. 16. Metaeditor database interface showing the ‘coint\_eg’ table fields with one NVDA cointegrated stock (ALMU)

As in the ‘corr\_pearson’ above, you will note again that we have a composite primary key, using the insertion timestamp, the reference ticker, and the correlated ticker. It is for the same reason. By having the cointegration test timestamp associated with the reference ticker and the, eventually, cointegrated ticker, we can test the cointegration for all symbols again, but now with a different timeframe and/or lookback period.

This table, along with the ‘coint\_joh’ next below, will be the core of our screening process. As an example, let’s look for opportunities on the H4 timeframe.

![Fig. 17 - Metaeditor database interface showing the ‘coint_eg’ table with seventeen stocks cointegrated with Nvidia Corp. (NVDA) in the H4 timeframe and different lookback periods](https://c.mql5.com/2/170/Capture_metaeditor_db_coint_eg_is_coint_true_H4.PNG)

Fig. 17. Metaeditor database interface showing the ‘coint\_eg’ table with seventeen stocks cointegrated with Nvidia Corp. (NVDA) in the H4 timeframe and different lookback periods

The table in Fig. 17 is sorted by the p-value. At the time of writing, the first one, Silicon Laboratories Inc. (SLAB), is the most cointegrated with NVDA, if we take the 30 lookback period. But any of these stocks is a good candidate for pair trading with NVDA in the H4 timeframe in their respective lookback periods. That is, at this point, we already have a solid portfolio of NVDA cointegrated stocks for pair trading with their respective hedge ratios. If we stop here, we have a portfolio ranked by cointegration strength that only needs to be tested for spread stationarity and validated in backtests.

Johansen

Johansen's method was developed right after the Engle-Granger test, at the end of the 1980s. In the decade of 1990s, it became a standard tool in the Academy, and was gradually adopted by the quant teams on hedge funds around the world at the beginning of this century.

While the Engle-Granger test is designed for two, and only two, time series, or two assets, the Johansen test is designed for multiple time series, including only two time series as well. This makes it especially powerful when screening baskets of two, three, four, or more stocks, where several different long-run relationships may coexist.

Johansen produces two main statistics: trace and maximum eigenvalue. These statistics determine the cointegration rank, that is, the number of independent stationary relationships among the assets.

- Rank = 0: no cointegration.
- Rank = 1: one stable spread. We have been working with this case in all the articles of this series.
- Rank > 1: multiple independent spreads, which gives us more flexibility in portfolio building. We will see this case in detail when dealing with seasonality and anomalies.

The test gives us a set of eigenvectors associated with these stationary relationships. Each of these eigenvectors indicates the weights of a linear combination of the stocks that is stationary over time. It is these weights that we use as hedge ratios, that is, the proportions in which we will combine the assets to form a synthetic, mean-reverting spread. By providing us with these hedge ratios, the test tells us how to build the spread between the cointegrated stocks. We used this feature in [the second article of this series](https://www.mql5.com/en/articles/19052), when we backtested a group of four stocks from the semiconductor industry.

As we did with the two previous tests, we’ll use a dedicated table to store both the parameters and outputs of the Johansen test for later analysis. But now we’ll need two tables:

First table: a main ‘coint\_johansen\_test’ table to store the metadata of each test run.

```
-- coint_johansen_test definition

CREATE TABLE coint_johansen_test (
        test_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tstamp INTEGER NOT NULL,
        timeframe TEXT CHECK(LENGTH(timeframe) <= 10) NOT NULL,
        lookback INTEGER NOT NULL,
        num_assets INTEGER NOT NULL,
        trace_stats_json TEXT NOT NULL,
        trace_crit_vals_json TEXT NOT NULL,
        eigen_stats_json TEXT NOT NULL,
        eigen_crit_vals_json TEXT NOT NULL,
        coint_rank INTEGER NOT NULL,
        CONSTRAINT coint_johansen_test_unique UNIQUE (tstamp, timeframe, lookback)
) STRICT;
```

Instead of using a composite primary key with asset tickers, as we did for the correlation test and the Engle-Granger cointegration test tables, now a single, auto-incrementing test\_id is used as the primary key in the main table. The coint\_johansen\_test\_assets table (below) then stores a row for each asset included in a given test run. This way, we can test any number of assets without changing the table structure. If you have an interest in going deeper to understand this procedure, please take a look at this [Wikipedia article about database normalization](https://en.wikipedia.org/wiki/Database_normalization "https://en.wikipedia.org/wiki/Database_normalization").

The constraint that requires the test timestamp, timeframe, and lookback period to be collectively unique allows us to re-run the test for the same symbols whenever we want, with different timeframes and lookback periods, and each test output will be unique.

The Johansen test outputs multiple values for the trace and max-eigenvalue statistics, along with their critical values. Storing them in single fields like trace\_stat is not an option anymore. Instead, we will be using JSON columns (trace\_stats\_json, trace\_crit\_vals\_json, etc.) to store the entire array of results as a string. We are using JSON-formatted strings and a TEXT field to store the critical values and the trace statistics arrays because SQLite doesn’t have the ARRAY data type natively. Please refer to the previous article to see in detail how we are dealing with this SQLite limitation, and [how we are reading and parsing these JSON strings in MQL5](https://www.mql5.com/en/articles/19428).

We will see how we can use the critical values (trace\_crit\_vals\_json) to our advantage in the next step, when backtesting a group of stocks using this automated screening. For now, just keep in mind that these are the values that we compare against the trace statistic (trace\_stats\_json) to get the number of cointegrating relationships at each significance level (1%, 5%, and 10% respectively).

The is\_coint boolean field is no longer sufficient. The key result of a multi-asset Johansen test is the number of cointegrating relationships, or rank (r). The new coint\_rank field will store the integer value of the rank.

Second table: ‘coint\_johansen\_test\_assets’, to link each test to the specific assets involved.

```
-- coint_johansen_test_assets definition

CREATE TABLE coint_johansen_test_assets (
        test_id INTEGER NOT NULL,
        symbol_id INTEGER NOT NULL,
        CONSTRAINT coint_johansen_test_assets_pk PRIMARY KEY(test_id, symbol_id),
        CONSTRAINT coint_johansen_test_assets_test_fk FOREIGN KEY(test_id) REFERENCES coint_johansen_test(test_id) ON DELETE CASCADE,
        CONSTRAINT coint_johansen_test_assets_symbol_fk FOREIGN KEY(symbol_id) REFERENCES symbol(symbol_id) ON DELETE CASCADE
) STRICT;
```

The coint\_johansen\_test\_assets table uses symbol\_id instead of ticker to establish a foreign key relationship with the ‘symbol’ table. This ensures that the assets we are testing always exist in our system. Besides that, we get some performance gains for free here, because it's generally more efficient for databases to perform joins on integer keys (symbol\_id) than on text fields (ticker). At this point, this is neither relevant nor is it our focus before we have a working system. But it is worth noting, and we are not paying for it. 🙂

Finally, remember that our ‘market\_data’ table is already linked with the ‘symbol’ table. By connecting our Johansen test output to the ‘symbol’ table, we are indirectly connecting it with our (at some point in the future) ‘curated’ market data, and that helps with the consistency of our database.

But, if you are following the reasoning, you may be asking: So, why did we not do the same for the Pearson correlation and the Engle-Granger cointegration test results? Why did we use ‘ticker’ and not ‘symbol\_id’, connecting them to the rest of our system?

There are two reasons for this choice, one of them a bit idiosyncratic. That is that I prefer that you run the Pearson correlation and the Engle-Granger cointegration tests for pairs-trading as easily as possible, without having to worry about the task of previously [importing symbols from Metatrader into the database](https://www.mql5.com/en/articles/19428).

The second reason is that the Johansen test will be our main tool in the automated system, being the correlation and the Engle-Granger tests limited to manual screening, because we will be dealing with baskets of stocks in our portfolio.

![Fig. 18 - Metaeditor database interface showing the ‘coint_johansen_test’ table fields](https://c.mql5.com/2/170/Capture_metaeditor_db_coint_joh_all.PNG)

Fig. 18. Metaeditor database interface showing the ‘coint\_johansen\_test’ table fields

Fig. 18 above shows the Johansen cointegration test table with all fields and one test with the cointegration vectors that we would use as the hedging ratios or portfolio weights. It can be better viewed in Figure 19 below.

![Fig. 19 - Metaeditor database interface showing the ‘coint_johansen_test’ table fields with one positive test for cointegration, along with the recommended hedge ratios (portfolio weights)](https://c.mql5.com/2/170/Capture_metaeditor_db_coint_joh_coint_rank_1_ratios.PNG)

Fig. 19. Metaeditor database interface showing the ‘coint\_johansen\_test’ table fields with one positive test for cointegration, along with the recommended hedge ratios (portfolio weights)

![Fig. 20 - Metaeditor database interface showing the ‘coint_johansen_test’ table with seven positive tests for cointegration with different cointegration ranks in the D1 timeframe and different lookback periods](https://c.mql5.com/2/170/Capture_metaeditor_db_coint_joh_rank_more_than_zero_D1.PNG)

Fig. 20. Metaeditor database interface showing the ‘coint\_johansen\_test’ table with seven positive tests for cointegration with different cointegration ranks in the D1 timeframe and different lookback periods

Here we already have a solid portfolio of cointegrated stocks from the semiconductor industry for all the most commonly used lookback periods, from a month to a year, half-year, a quarter, etc. Each group of stocks from the same lookback period can be tested to be part of a basket of stocks.

These are the data that will be used in our ‘strategy’ table to update our model in real-time, as seen in the previous article. Here lies the base of the backtest that we will run in the next article.

It’s important to understand that the Engle–Granger test wasn’t made obsolete by the Johansen test. They both look for the same thing — whether two or more prices move together in the long run — but they go about it in different ways. Engle–Granger is simpler and good for testing two assets at a time, while Johansen is more powerful when you want to test several assets together. In practice, we should often use whichever fits the situation best, and sometimes check both to be more confident in the result.

Engle-Granger Test

| ✔️Pros | ❌Cons |
| --- | --- |
| - Simple to run and easy to understand.<br>- Works well when testing just two assets (classic pairs trade).<br>- Quick to compute, even with small datasets. | - Sensitive to which asset you treat as “dependent” (order matters).<br>- Limited to two assets — can’t handle baskets.<br>- Less reliable if the data is noisy or the relationship is weak. |

Table 1 -  Bulleted list of the Engle-Granger cointegration test pros and cons

Johansen test

| ✔️Pros | ❌Cons |
| --- | --- |
| - Handles multiple assets at once (pairs, trios, or baskets).<br>- More statistically rigorous — better at finding stable long-run links.<br>- Doesn’t depend on choosing one asset as the “dependent variable.” | - More complex to set up and interpret.<br>- Needs more data (larger sample sizes) to work properly.<br>- Easier to misuse if lags or settings are chosen poorly. |

Table 2 - Bulleted list of the Johansen cointegration test pros and cons

As a rule of thumb, for a quick check of two stocks, use Engle-Granger; for deeper tests for portfolios or more than two assets, use Johansen.

The last step before setting up a ranking system is to validate the stationarity of the spread.

### Stationarity Validation

Once we have identified a cointegrating relationship in a group of stocks for a given timeframe and lookback period, the final step is to validate that the spread is truly stationary. It is the stationary property that guarantees that the spread fluctuates around a stable mean and does not drift over time. This step is required because, without stationarity, we cannot be sure the mean reversion will occur, and the mean reversion is the core of our strategy; it is where the arbitrage opportunities are.

The two tests we’ve been using to validate the spread stationarity are the ADF (Augmented Dickey–Fuller) and the KPSS (Kwiatkowski–Phillips–Schmidt–Shin). Interpreting them together provides robustness: if they agree, we gain strong evidence that the spread is indeed mean-reverting. When the two disagree, it often indicates unstable behavior, and we should be cautious: it is a red flag.

As with all the previous tests, we also have a dedicated table, ‘coint\_adf\_kpss’, to store the results of the stationarity tests.

```
-- coint_adf_kpss definition

CREATE TABLE "coint_adf_kpss" (
        test_id INTEGER NOT NULL,
        symbol_id INTEGER NOT NULL,
        adf_stat REAL NOT NULL,
        adf_pvalue REAL NOT NULL,
        is_adf_stationary INTEGER NOT NULL,
        kpss_stat REAL NOT NULL,
        kpss_pvalue REAL NOT NULL,
        is_kpss_stationary INTEGER NOT NULL,
        CONSTRAINT coint_adf_kpss_pk PRIMARY KEY (test_id, symbol_id),
        CONSTRAINT coint_adf_kpss_test_fk FOREIGN KEY (test_id) REFERENCES coint_johansen_test(test_id) ON DELETE CASCADE,
        CONSTRAINT coint_adf_kpss_symbol_fk FOREIGN KEY (symbol_id) REFERENCES symbol(symbol_id) ON DELETE CASCADE
) STRICT;
```

Again, we use a composite primary key, but now it is not the test run timestamp/timeframe/lookback triplet that makes each test unique, but the connection with the Johansen test ID and symbol (which is made unique by that triplet). This way, we are linking the stationarity tests with the Johansen, but also making the ADF/KPSS test dependent on the previous cointegration test having at least one cointegrated vector.

This dependence of a previous cointegration test having at least one cointegrated vector is explicit in the Python code:

```
def get_coint_groups(self):
        """
        Retrieves information for all Johansen tests with a cointegrating rank > 0.
        Returns a list of dictionaries, each containing test_id, timeframe, lookback, and a list of symbol_ids.
        """
        if not self.conn:
            print("Database connection not established.")
            return []

        try:
            query = """
            SELECT test_id, timeframe, lookback
            FROM coint_johansen_test
            WHERE coint_rank > 0
            """

            df_tests = pd.read_sql_query(query, self.conn)

            if df_tests.empty:
                print("No cointegrated groups found in the database.")
                return []
```

This is a design choice. Theoretically, there is nothing preventing us from running ADF tests as a standalone tool. It is only that it doesn’t make sense in our pipeline. Running it only on a cointegrated group is more resource-friendly in our case, because very soon we will be running them 24/7 on thousands of stocks, and different timeframes/lookback combinations.

![Fig 21 - Metaeditor database interface showing the ‘coint_adf_kpss’ table with one positive result for both ADF and KPSS spread stationarity test](https://c.mql5.com/2/170/Capture_metaeditor_db_coint_adf_kpss_all.PNG)

Fig 21. Metaeditor database interface showing the ‘coint\_adf\_kpss’ table with one positive result for both ADF and KPSS spread stationarity test

From the test ID (20), we get the symbols, timeframe, lookback period used in the Johansen cointegration test, and also the cointegration vectors (portfolio weights) returned by it.

```
SELECT
    t1.test_id,
    t3.ticker,
    t2.timeframe,
    t2.lookback,
    t2.coint_vectors_json
FROM coint_johansen_test_assets AS t1
JOIN coint_johansen_test AS t2
    ON t1.test_id = t2.test_id
JOIN symbol AS t3
    ON t1.symbol_id = t3.symbol_id
WHERE t1.test_id = 123;
```

This is a relatively expensive query with two JOIN operations. But let’s take into account that we will run a small number of these queries, only when we find a group of assets that was qualified in all previous filters. That is, this query is at the end of our pipeline. From there, we already have our cointegrated basket of stocks with a stable spread to be copied to our ‘strategy’ table. The EA reads the ‘strategy’ table to update its own parameters in real-time (see our previous article).

![Fig 22 - Metaeditor database interface showing the ‘coint_johansen_test’ table with details for a specific test ID with positive results for both the ADF and the KPSS test](https://c.mql5.com/2/170/Capture_metaeditor_db_coint_johansen_test_positive_adf_kpss.PNG)

Fig 22. Metaeditor database interface showing the ‘coint\_johansen\_test’ table with details for a specific test ID with positive results for both the ADF and the KPSS test

In this example, with both ADF and KPSS stationarity tests being positive for test id #20, we know that AMD, INTC, LAES, RMBS, TSM, and WOLF can be traded on the daily (D1) timeframe with the respective portfolio weights stored as ‘coint\_vectors\_json’.

At this point, our database schema has changed a lot.

![Fig. 23 - Entity-Relationship Diagram (ERD) depicting the statarb-0.3 schema version.](https://c.mql5.com/2/170/statarb-0.3_ER.png)

Fig. 23. Entity-Relationship Diagram (ERD) depicting the statarb-0.3 schema version.

You may have noted that we have no database indexes yet. This is intentional. Since we chose to design a bottom-up style, for now we can only suppose (with a reasonable level of certainty, it is true) where we will need indexes to speed up our most used and/or more demanding queries (like those on composite primary keys with one pair being a TEXT field). But to avoid overthinking and overengineering, we will only add them when we know for sure which demanding and/or most used queries are.

### Basket Selection - Building a scoring system

The group of stocks depicted in Fig. 22 above represents one single basket of stocks to be part of our portfolio. And our portfolio is expected to have at least a dozen of these baskets. That is because we will need to rotate it; that is, we will need to replace basket #1 with basket #2 at any time the latter proves to be more promising in terms of expected returns than the former. We need to have a sufficiently large portfolio to accommodate the market changes, corporate events, and our own trading decisions (stop trading stock X, for example).

We did all this filtering from a universe of ~10,000 assets, including, for didactic purposes, other asset types than stocks, to a narrow basket of six stocks, with a specific timeframe, and with cointegration valid for a specific lookback period. We explained each detail to ease your understanding, but remember that all this process will run full-time in automated mode, testing many combinations of these three variables (symbol, timeframe, and lookback period). Because of this combination, we can expect to have many more than a single basket. We can expect to be faced with dozens of baskets in the line waiting to be admitted to our trading portfolio. How can we choose which one to be the next and which of them should be sent to the end of the line? The answer is a ranking system.

These are some of the ranking criteria we may consider:

Strength of Cointegration \- We may use the Johansen statistics (trace and maximum eigenvalue) to gauge how strongly the series are cointegrated. Stronger test statistics indicate a more reliable long-run relationship.

Number of Cointegrating Vectors (Rank) \- By considering the number of cointegrating vectors, we may assess the desired complexity of the system we are building.

- Rank = 1 indicates a single stable spread. Easier to monitor.
- Rank > 1 indicates multiple possible spreads. More flexible, but also more complex.

Stability of portfolio weights \- We may avoid highly varying hedge ratios across different sample windows. It can indicate fragile relationships.

Reasonable spreads \- The spread should be economically reasonable. We should avoid extremely large positions in one stock.

Time to reversion \- We may calculate the half-life of mean reversion to assess how quickly the spread reverts to equilibrium.

Liquidity \- Liquidity is an indispensable criterion because spreads involving illiquid stocks may be costly to trade.

Transaction costs \- This is a very individual ranking criterion, because it is relative to each one's objectives and trading system, but we are leaving it here for the sake of completeness.

We may build a scoring system by combining all or some of the above criteria. For example, weighting cointegration strength, stability of portfolio weights, and mean spread. At the end of the day, basket selection is about finding a balance between statistical filtering, market knowledge, and tradability.

However, this scoring system only makes sense in practice. Since our next article is all about backtesting the automated system we are building, we’ll start by developing and testing a scoring system based on some of the criteria above. This way, we’ll be testing the scoring system itself, along with the EA and its different basket compositions.

### Conclusion

We saw how we can build a simplified asset screening for a trading strategy based on statistical arbitrage through cointegrated stocks. We started from a large universe of thousands of symbols available on the trading platform/broker server, and by successive filtering, we arrived at a single basket of six stocks with one cointegration vector and a stationary spread.

We began filtering by economic sector and industry, then we selected the most correlated among those in the semiconductors industry, to build the final basket based on the Johansen cointegration test and the ADF/KPSS stationarity tests.

In the process, we changed our database to accommodate dedicated tables to store each test result for further analysis. Also, we developed modular Python classes to replace the scripts we’ve been using so far.

At this point, we are ready to experiment with our scoring system, which we’ll do in the next step while backtesting our system with the screening method developed here.

A somewhat personal note about the attached code

The attached code, mainly Python, was written with extensive support of the integrated AI Assistants in Metatrader 5, VS Code, and others.

I see AI for code development as the most high-level language of our time. It doesn’t remove the need to know programming. Instead, it requires that the developer have a clear understanding of the system, and also a clear understanding of the so-called user requirements. One needs to know very well what to ask for, how to ask for it, and more than anything, the developer must know how to check if code provided by the assistant fits the requirements, that is, if the code does what it is expected to do. Then, the Assistant will act as a junior developer on steroids, or even as a senior developer, depending on the AI model, the prompts, and the programming language in question.

By analogy, I see AI for code development as the creation of the C programming language: it doesn’t eliminate the need to know programming; it only alleviates the developer's need to write assembly. The AI assistants are already capable of writing better code, in less time, with tests and documentation included for free.

In our case, the relatively “low-level” programming language is Python or MQL5.

| Filename | Description |
| --- | --- |
| Files\\StatArb\\schema-0.3.sql | SQL Schema (DDL) for SQLite database version 0.3 |
| Scripts\\StatArb\\db-setup.mq5 | MQL5 Script to setup the database (reads the above schema file) |
| Scripts\\StatArb\\SymbolImporter.mq5 | MQL5 Script to import symbols metadata from MetaTrader 5 into SQLite DB |
| coint\_adf\_kpss\_to\_db.py | Python class with methods for running the ADF and KPSS stationarity tests |
| coint\_eg\_to\_db.py | Python class with methods for running the Engle-Granger cointegration test |
| coint\_johansen\_to\_db.py | Python class with methods for running the Johansen cointegration test |
| corr\_pearson\_to\_db.py | Python class with methods for running the Pearson correlation test |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19626.zip "Download all attachments in the single ZIP archive")

[mql5-article-files-19626.zip](https://www.mql5.com/en/articles/download/19626/mql5-article-files-19626.zip "Download mql5-article-files-19626.zip")(17.67 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Statistical Arbitrage Through Cointegrated Stocks (Part 10): Detecting Structural Breaks](https://www.mql5.com/en/articles/20946)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://www.mql5.com/en/articles/20657)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 8): Rolling Windows Eigenvector Comparison for Portfolio Rebalancing](https://www.mql5.com/en/articles/20485)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://www.mql5.com/en/articles/19428)

**[Go to discussion](https://www.mql5.com/en/forum/496223)**

![Automating Trading Strategies in MQL5 (Part 34): Trendline Breakout System with R-Squared Goodness of Fit](https://c.mql5.com/2/172/19625-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 34): Trendline Breakout System with R-Squared Goodness of Fit](https://www.mql5.com/en/articles/19625)

In this article, we develop a Trendline Breakout System in MQL5 that identifies support and resistance trendlines using swing points, validated by R-squared goodness of fit and angle constraints, to automate breakout trades. Our plan is to detect swing highs and lows within a specified lookback period, construct trendlines with a minimum number of touch points, and validate them using R-squared metrics and angle constraints to ensure reliability.

![MQL5 Wizard Techniques you should know (Part 80): Using Patterns of Ichimoku and the ADX-Wilder with TD3 Reinforcement Learning](https://c.mql5.com/2/172/19627-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 80): Using Patterns of Ichimoku and the ADX-Wilder with TD3 Reinforcement Learning](https://www.mql5.com/en/articles/19627)

This article follows up ‘Part-74’, where we examined the pairing of Ichimoku and the ADX under a Supervised Learning framework, by moving our focus to Reinforcement Learning. Ichimoku and ADX form a complementary combination of support/resistance mapping and trend strength spotting. In this installment, we indulge in how the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm can be used with this indicator set. As with earlier parts of the series, the implementation is carried out in a custom signal class designed for integration with the MQL5 Wizard, which facilitates seamless Expert Advisor assembly.

![Cyclic Parthenogenesis Algorithm (CPA)](https://c.mql5.com/2/113/Cyclic_Parthenogenesis_Algorithm____LOGO.png)[Cyclic Parthenogenesis Algorithm (CPA)](https://www.mql5.com/en/articles/16877)

The article considers a new population optimization algorithm - Cyclic Parthenogenesis Algorithm (CPA), inspired by the unique reproductive strategy of aphids. The algorithm combines two reproduction mechanisms — parthenogenesis and sexual reproduction — and also utilizes the colonial structure of the population with the possibility of migration between colonies. The key features of the algorithm are adaptive switching between different reproductive strategies and a system of information exchange between colonies through the flight mechanism.

![How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://c.mql5.com/2/171/19547-how-to-build-and-optimize-a-logo.png)[How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)

This article explains how to design and optimise a trading system using the Detrended Price Oscillator (DPO) in MQL5. It outlines the indicator's core logic, demonstrating how it identifies short-term cycles by filtering out long-term trends. Through a series of step-by-step examples and simple strategies, readers will learn how to code it, define entry and exit signals, and conduct backtesting. Finally, the article presents practical optimization methods to enhance performance and adapt the system to changing market conditions.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/19626&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083094546784457960)

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