---
title: Statistical Arbitrage Through Cointegrated Stocks (Part 3): Database Setup
url: https://www.mql5.com/en/articles/19242
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:46:50.619429
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/19242&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083103274158003480)

MetaTrader 5 / Trading systems


### Introduction

In [the previous article of this series (Part 2)](https://www.mql5.com/en/articles/19052), we backtested a statistical arbitrage strategy composed of a basket of cointegrated stocks from the microprocessor sector (Nasdaq stocks). We started filtering among hundreds of stock symbols for those most correlated with Nvidia. Then we tested the filtered group for cointegration using the Johansen test, the stationarity of the spread using ADF and KPSS tests, and finally, we obtained the relative portfolio weights by extracting the Johansen eigenvector for the first rank. The backtest results were promising.

However, two or more assets may have been cointegrated for the last two years and start losing cointegration tomorrow. That is, there is no guarantee that a cointegrated pair or group of assets will remain cointegrated. Changes in company management, the macroeconomic scenario, or sector-specific changes can affect the fundamentals that originally drove the assets’ cointegration. And vice versa. Assets that weren’t cointegrated may start a cointegrated path next minute for the same reasons. The market is “an enigma in a continuous state of change”. We need to cope with this change. \[ [PALOMAR](https://www.mql5.com/go?link=https://bookdown.org/palomar/portfoliooptimizationbook/15.4-discovering-pairs.html%23are-cointegrated-pairs-persistent "https://bookdown.org/palomar/portfoliooptimizationbook/15.4-discovering-pairs.html#are-cointegrated-pairs-persistent"), 2025\]

A basket of cointegrated stocks will change its relative portfolio weights almost continuously, and the portfolio weights determine not only the volume (quantity) of our orders, but also their direction (buy or sell). So, we need to cope with this change too. While cointegration is a more long-term relationship, the portfolio weights change all the time. So we need to check for them more frequently and update our model as soon as they change. When we detect that our model is outdated, we need to act immediately; we want an instantaneous replacement of the outdated model.

Our Expert Advisor must be aware in real-time if the portfolio weights that we’ve been using still apply or have changed. If they changed, the EA must be informed what the new portfolio weights as quickly as possible. Also, our EA must know if the model itself remains valid. If not, the EA should be informed which assets must be replaced, and the rotation must be applied as soon as possible in the active portfolio.

We have been using the Metatrader 5 Python integration and the professionally developed [statistical functions from the statsmodels library](https://www.mql5.com/go?link=https://www.statsmodels.org/ "https://www.statsmodels.org/"), but until now, we have been working with real-time data only, downloading the quotes (the price data) as we need them. This approach is useful in the exploratory phase because of its simplicity. But if we are going to rotate our portfolio, update our models, or the portfolio weights, we might be starting to think about data persistence. That is, we need to start thinking about storing our data in a database because it is not practical to download the data every time we need it. More than that, we may need to look for relationships among different asset classes, among symbols that were not related to our first cointegration tests.

A high-quality, scalable, and metadata-rich database is the core of any serious statistical arbitrage endeavour. By taking into account that database design is a very idiosyncratic task, in the sense that a good database is the one that fits each business's requirements, in this article, we will see one possible approach for building our statistical arbitrage-oriented database.

### What questions must our database answer?

In our quest for a “poor man’s statistical arbitrage framework”, meaning a framework suitable for the average retail trader with a regular consumer notebook and an average network bandwidth, we are facing several challenges related to the lack of specialization in required domains, like statistics and software development. Database design is NOT an exception in this list of required expertise. Database design is a large field, per se. One can write whole books about database design without exhausting the subject. The ideal solution would be to have a specialized professional, possibly more than one individual, to design, implement, and maintain our database.

But since we are developing this stat arb framework for the average retail trader, we need to work with what we have, doing research in books and specialized internet forums and channels; learning what we can with seasoned professionals, learning by trial and error, doing experiments, taking risks and being prepared to modify our design if and when it proves itself inadequate for the task at hand. We need flexibility to change things, and we need to start small, working in a bottom-up style, instead of a top-down approach, to avoid over-engineering.

At the end of the day, our database should answer a very simple question: what should we trade now to get the maximum possible return?

Referring to the last article in this series, where we defined the directions and volume of our orders based on the stock basket portfolio weights, one possible answer could be something like this:

| Symbols | Weigths | Timeframe |
| --- | --- | --- |
| "MU", "NVDA", "MPWR", "MCHP" | 2.699439, 1.000000, -1.877447, -2.505294 | D1 |
| --- | --- | --- |

Table 1 - Made-up sample of expected query response for real-time model update

If our database can provide us with this pretty simple information, updated at a suitable frequency, we have what we need to trade continuously at our most optimized level.

### Database updates as a Service

Until now, we have been conducting our data analysis using real-time quotes from the Metatrader 5 Terminal via Python code (technically, most of the time, we have been utilizing the quotes stored by the underlying Terminal engine). Once the symbols and portfolio weights were defined, we have been updating our Expert Advisor manually with the new symbol and/or the new portfolio weights.

From now on, we will decouple our data analysis from the Terminal and start using the data stored in our database to update our Expert Advisor as soon as we have new portfolio weights, stop the trading if the cointegration relationship is lost, or another group of symbols is deemed more promising. That is, we want to improve our market exposure to each symbol, updating the portfolio weights in real-time and/or rotating our portfolio every time our data analysis recommends.

To update the database, we will implement a Metatrader 5 Service.

From [the Metatrader 5 documentation](https://www.mql5.com/en/docs/runtime/running), we learn that

- Services are not bound to a specific chart.

- Services are loaded right after starting the terminal if they were launched at the moment of the terminal shutdown.

- Services run on their own thread.

We can conclude that a Service is the ideal method to keep our database up-to-date. If it is running when we end a trading session and close the terminal, it will resume as soon as we re-launch the terminal for a new session, independently of any chart/symbol we may be using at the time. Besides that, since it will run on its own thread, it will not impact or be impacted by other services, indicators, scripts, or Expert Advisors we may be running.

So, our workflow will be like this:

1. All data analysis will be made in Python, outside the Metatrader 5 environment. To run the analysis, we will download historical data and insert it into the database.
2. Every time we modify our active portfolio, adding or removing a symbol, we update the Service input parameters with an array of symbols and an array of timeframes.
3. Every time we modify our active portfolio, adding or removing a symbol, we update the Expert Advisor input parameters.

For now, we will do the updates in steps 2 and 3 manually. Later, we will automate it.

### Database setup

While researching for this article, I was reminded that the ideal tool for the job is a time-series specialized columnar-oriented database. There are a lot of products on the market that fit this requirement, both paid and free, proprietary and open-source alternatives. They are meant to support highly specialized workloads, a massive amount of data in the sub-second response time, both for data ingestion and query response in real-time.

But our focus here is not scale. Our primary focus is on simplicity and applicability by an individual, not a team of high-skilled professional database administrators (DBAs) and time-series data management designers. So we will start with the simplest solution, being aware of its limitations, and keeping in mind that the system will evolve in the future as the need arises.

We will start with the Metatrader 5 integrated SQLite database. There is a plethora of information on how to create and use the integrated SQLite database on the Metatrader 5 environment. You can find it:

- On the Metaeditor documentation for how to use the Metaeditor [graphical interface for creating, using, and managing databases](https://www.metatrader5.com/en/metaeditor/help/database "https://www.metatrader5.com/en/metaeditor/help/database")

- Also, on the Metaeditor documentation for the [API functions for working with databases](https://www.mql5.com/en/docs/database)

- In a MetaQuotes primer article for a general introduction about [native handling of SQL databases in MQL5](https://www.mql5.com/en/articles/7463)

- On the MQL5 Book for [advanced use of the SQLite integrated database](https://www.mql5.com/en/book/advanced/sqlite)

If you are planning serious work with the integrated SQLite database, it is strongly recommended that you go deep dive into all these links. What follows is a summary of the steps for our very specific use case in this article, with some notes about the rationale behind the choices.

### The schema

Attached to this article, you will find the _db-setup.mq5_ file, which is an MQL5 script. It will accept as input parameters the SQLite database filename and the database schema filename. These parameters default to s _tatarb-0.1.db_ and _schema-0.1.sql_, respectively.

WARNING: It is strongly recommended that you keep your database schema file under version control, and do not duplicate it in your filesystem. Metatrader 5 offers a robust [version control system already integrated](https://www.metatrader5.com/en/metaeditor/help/mql5storage/mql5storage_working "https://www.metatrader5.com/en/metaeditor/help/mql5storage/mql5storage_working") in the platform.

By launching this script with the default input parameters, you will have a SQLite database created on the MQL5/Files/StatArb folder of your terminal (NOT on the common folder), with all the tables and fields commented below. The script will also generate a file _output.sql_ in your MQL/Files folder for debugging purposes only. In case you have any trouble, you can check this file to see how your system is reading the schema file. If everything goes well, you can safely delete this file.

Alternatively, you can create the database by other means and read the schema file from the database on any SQLite3 client, on the Metaeditor UI, Windows PowerShell, or SQLite3 command-line. I recommend that, at least for the first time, you create the database by running the attached script. You can always customize your process later.

Database schema

Our initial database schema is composed of only four tables, two of them being placeholders for the next steps. That is, at this point, we will be using only the “symbol” and the “market\_data” tables.

Figure 1 shows the Entity-Relationship Diagram (ERD) of this initial schema.

![Fig.1 - Entity-Relationship Diagram (ERD) for the initial database schema](https://c.mql5.com/2/164/statarb-0.1_ER.png)

Fig. 1 - The initial database schema ERD (Entity-Relationship Diagram)

The “corporate\_event” table, without surprise, is meant to store events related to our portfolio companies, like dividend distribution amounts, stock splits, share buybacks, mergers, and others. We will not deal with it for now.

The “trade” table will store our trades (deals). By gathering this data, we will have a unique data collection for aggregation and analysis. We will only use it when we start trading.

The “market\_data” table will store our OHLC data with all the MqlRates fields. It has a composite primary key (symbol\_id, timeframe, and timestamp) to make sure each entry is unique. The “market\_data” table is connected to the “symbol” table by a foreign key.

As you can see, it is the only table that uses a composite primary key. All other tables use as primary key the timestamp stored as an INTEGER. There is a reason for this choice. According to the [SQLite3 documentation](https://www.mql5.com/go?link=https://sqlite.org/lang_createtable.html%23rowid "https://sqlite.org/lang_createtable.html#rowid"),

“The data for rowid tables is stored as a B-Tree structure containing one entry for each table row, using the rowid value as the key. This means that retrieving or sorting records by rowid is fast. Searching for a record with a specific rowid, or for all records with rowids within a specified range, is around twice as fast as a similar search made by specifying any other PRIMARY KEY or indexed value.

(...) if a rowid table has a primary key that consists of a single column and the declared type of that column is "INTEGER" in any mixture of upper and lower case, then the column becomes an alias for the rowid. Such a column is usually referred to as an "integer primary key".

(...) “you can get queries ‘around twice as fast’ if you use an INTEGER as a primary key. Unix epoch timestamps can be inserted as an INTEGER in SQLite3 databases.”

(...) “SQLite stores integer values in the 64-bit twos-complement format¹. This gives a storage range of -9223372036854775808 to +9223372036854775807, inclusive. Integers within this range are exact. ( [SQLite3 docs](https://www.mql5.com/go?link=https://www.sqlite.org/floatingpoint.html "https://www.sqlite.org/floatingpoint.html"))”

So we can replace our date times from string to Unix epoch timestamps and insert them as the primary key to get a boost over light speed. 🙂

The tables below show the full schema documentation (data dictionary) for reference for our team and ourselves in the future.

Table: symbol

Stores metadata about financial instruments traded or tracked.

| Field | Data Type | Null | Key | Description |
| --- | --- | --- | --- | --- |
| symbol\_id | INTEGER | NO | PK | Unique identifier for each financial instrument. |
| ticker | TEXT(≤10) | NO |  | Ticker symbol of the asset (e.g., "AAPL", "MSFT"). |
| exchange | TEXT(≤50) | NO |  | Exchange where the asset is listed (e.g., "NASDAQ", "NYSE"). |
| asset\_type | TEXT(≤50) | YES |  | Type of asset (e.g., "Equity", "ETF", "FX", "Crypto"). |
| sector | TEXT(≤50) | YES |  | Economic sector classification (e.g., "Technology", "Healthcare"). |
| industry | TEXT(≤50) | YES |  | Industry classification within a sector. |
| currency | TEXT(≤50) | YES |  | Currency of denomination for the asset (e.g., "EUR", "USD"). |

Table 2 - ‘symbol’ table data dictionary description (v0.1)

Table: corporate\_event

Captures events affecting assets, such as dividends, stock splits, or earnings announcements.

| Field | Data Type | Null | Key | Description | Example |
| --- | --- | --- | --- | --- | --- |
| tstamp | INTEGER | NO | PK | Unix timestamp when the event takes effect. | 1678905600 |
| event\_type | TEXT ENUM {'dividend', 'split', 'earnings'} | NO |  | Type of corporate action. | "dividend" |
| event\_value | REAL | YES |  | Numeric value of the event:• Dividend amount per share• Split ratio• Earnings per share (EPS). | 0.85, 2.0, 1.35 |
| details | TEXT(≤255) | YES |  | Additional notes or context. | "Q2 dividend payout" |
| symbol\_id | INTEGER | NO | FK | Reference to symbol(symbol\_id); links event to asset. | 1 |

Table 3 - ‘corporate\_event’ table data dictionary description (v0.1)

Table: market\_data

Stores OHLCV (open, high, low, close, volume) and related time-series data for assets.

| Field | Data Type | Null | Key | Description | Example |
| --- | --- | --- | --- | --- | --- |
| tstamp | INTEGER | NO | PK\* | Unix timestamp of bar/candle. | 1678905600 |
| timeframe | TEXT ENUM {M1,M2,M3,M4,M5,M6, M10,M12,M15,M20,M30,H1,H2,H3, H4,H6,H8,H12,D1,W1,MN1} | NO | PK\* | Granularity of the time-series data. | "M5", "D1" |
| price\_open | REAL | NO |  | Open price at beginning of bar. | 145.20 |
| price\_high | REAL | NO |  | Highest price during bar. | 146.00 |
| price\_low | REAL | NO |  | Lowest price during bar. | 144.80 |
| price\_close | REAL | NO |  | Closing price at end of bar. | 145.75 |
| tick\_volume | INTEGER | YES |  | Number of ticks (price updates) within the timeframe. | 200 |
| real\_volume | INTEGER | YES |  | Number of units/contracts traded (if available). | 15000 |
| spread | REAL | YES |  | Average or snapshot bid-ask spread during the bar. | 0.02 |
| symbol\_id | INTEGER | NO | PK\*, FK | Reference to symbol(symbol\_id). | 1 |

Table 4 - ‘market\_data’ table data dictionary description (v0.1)

Table: trade

Tracks live or simulated trades for strategies.

| Field | Data Type | Null | Key | Description | Example |
| --- | --- | --- | --- | --- | --- |
| tstamp | INTEGER | NO | PK | Unix timestamp of trade execution. | 1678905600 |
| ticket | INTEGER | NO |  | Trade ticket/order ID. | 20230001 |
| side | TEXT ENUM {'buy', 'sell'} | NO |  | Trade direction. | "buy" |
| quantity | INTEGER (>0) | NO |  | Number of shares/contracts traded. | 100 |
| price |  | NO |  | Executed price. | 145.50 |
| strategy |  | YES |  | Identifier for trading strategy generating the trade. | "StatArb\_Pairs" |
| symbol\_id | INTEGER | NO | FK | Reference to symbol(symbol\_id). | 1 |

Table 5 - ‘trade’ table data dictionary description (v0.1)

STRICT

By inspecting the schema file, you will see that all tables are STRICT tables.

```
CREATE TABLE symbol(
    symbol_id INTEGER PRIMARY KEY,
    ticker TEXT CHECK(LENGTH(ticker) <= 10) NOT NULL,
    exchange TEXT CHECK(LENGTH(exchange) <= 50) NOT NULL,
    asset_type TEXT CHECK(LENGTH(asset_type) <= 50),
    sector TEXT CHECK(LENGTH(sector) <= 50),
    industry TEXT CHECK(LENGTH(industry) <= 50),
    currency TEXT CHECK(LENGTH(currency) <= 10)
) STRICT;
```

That means we are opting for strict typing in our tables, instead of the convenient SQLite type affinity. It is a choice we think may avoid troubles later.

“In a CREATE TABLE statement, if the "STRICT" table-option keyword is added to the end, after the closing ")", then strict typing rules apply to that table.” ( [SQLite docs](https://www.mql5.com/go?link=https://www.sqlite.org/stricttables.html "https://www.sqlite.org/stricttables.html"))

CHECK LENGTH

Also, we are requiring a check for the length of several TEXT fields. That is because SQLite does not enforce a fixed length or truncate strings based on the (n) specified in CHAR(n) or VARCHAR(n), as is common in other RDBMS.

“Note that numeric arguments in parentheses that follow the type name (ex: "VARCHAR(255)") are ignored by SQLite - SQLite does not impose any length restrictions (other than the large global SQLITE\_MAX\_LENGTH limit) on the length of strings, BLOBs, or numeric values.” ( [SQLite docs](https://www.mql5.com/go?link=https://www.sqlite.org/datatype3.html "https://www.sqlite.org/datatype3.html"))

We are enforcing a limit to avoid troubles with disproportionately long strings from unknown sources.

What about Indexes?

You may be asking why we have created no indexes. We will, as soon as we start making queries, so we can know for sure where they are required.

### Initial data insertion

The database is expected to be filled transparently while executing our data analysis from the MQL5 Python integration. However, for convenience, there is a Python script attached (db\_store\_quotes.ipynb) that will serve as a helper for you to store the quotes from a list of symbols, from a specific timeframe, and from a chosen time interval. From now onwards, we will run our data analysis (correlation, cointegration, and stationarity tests) with this stored data.

![Figure 2 - ‘Symbol’ table after initial data insertion using the Python script](https://c.mql5.com/2/164/Capture_db_symbols_table.PNG)

Figure 2 - ‘Symbol’ table after initial data insertion using the Python script

As you can see, most of the “symbol” metadata is ‘UNKNOWN’. That is because the Python functions for SymbolInfo do not cover all the symbol metadata available in the MQL5 API. We will fill these gaps later.

```
   # Insert new symbol
    cursor.execute("""
    INSERT INTO Symbol (ticker, exchange, asset_type, sector, industry, currency)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        mt5_symbol,
        # some of this data will be filled by the MQL5 DB Update Service
        # because some of them are not provided by the Python MT5 API
        symbol_info.exchange or 'UNKNOWN',
        symbol_info.asset_class or 'UNKNOWN',
        symbol_info.sector or 'UNKNOWN',
        symbol_info.industry or 'UNKNOWN',
        symbol_info.currency_profit or 'UNKNOWN'
    ))
```

The database path should be passed as an environment variable, and we are using the python-dotenv module to load this variable. It should avoid trouble with terminal and/or PowerShell environment variables not being recognized by your editor.

You will find the Jupyter notebook that loads the python-dotenv extension and respective \*.env file right at the start of the script.

```
%load_ext dotenv
%dotenv .env
```

An example \*.env file is also attached to this article.

```
# keep this file at the root of your project
# or in the same folder of the Python script that uses it
STATARB_DB_PATH="your/db/path/here"
```

The db\_store\_quotes.ipynb main call is at the bottom of the script.

```
symbols = ['MPWR', 'AMAT', 'MU']  # Symbols from Market Watch
timeframe = mt5.TIMEFRAME_M5    # 5-minute timeframe
start_date = '2024-02-01'
end_date = '2024-03-31'
db_path = os.getenv('STATARB_DB_PATH')  # Path to your SQLite database

if db_path is None:
        print("Error: STATARB_DB_PATH environment variable is not set.")
else:
        print("db_path: " + db_path)
# Download historical quotes and store them in the database
        download_mt5_historical_quotes(symbols, timeframe, start_date, end_date, db_path)
```

### Updates

The core of our database maintenance for automated model update and portfolio rotation lies on our  “background” running MQL5 Service. As our database evolves, this Service will require updates too.

The Service connects to (or creates) a local SQLite database file.

WARNING: If creating a brand new database using the database update Service, remember to initialize the database with the _db\_setup_ Script, as mentioned above.

Then the Service sets up necessary constraints for data integrity, and then enters an endless loop where it checks for

new market data. For each symbol and timeframe, it

1. retrieves the most recently completed price bar (including open, high, low, close, volume, and spread),
2. verifies if it's already in the database,
3. and inserts the new quote(s) if not

The service wraps the insertion in a database transaction to ensure atomicity. If something goes wrong (eg, a database error), it retries up to a set limit (default 3 times) with one-second pauses. Logging is optional. The loop pauses between updates and stops only when the service is manually halted.

So, let’s take a look at some of its components.

In the inputs, you can choose the database path in the filesystem, the update frequency in minutes, the maximum number of retries if the insertion failed, and whether you want to print the success/failure messages in the EA log. This last parameter can be useful when we reach a stable code after development.

```
//+------------------------------------------------------------------+
//|   Inputs                                                         |
//+------------------------------------------------------------------+
input string   InpDbPath      = "StatArb\\statarb-0.1.db";  // Database filename
input int      InpUpdateFreq  = 1;     // Update frequency in minutes
input int      InpMaxRetries  = 3;     // Max retries
input bool     InpShowLogs    = true; // Enable logging?
```

![Fig. 3 - Metatrader 5 Dialog for Database Update Service input parameters](https://c.mql5.com/2/164/Capture_db_update_input_dialog.PNG)

Figure 3 - Metatrader 5 Dialog for Database Update Service input parameters

You must select the symbols and respective timeframes to be updated. These entries will be automated in the next phase.

```
//+------------------------------------------------------------------+
//|   Global vars                                                    |
//+------------------------------------------------------------------+
string symbols[] = {"EURUSD", "GBPUSD", "USDJPY"};
ENUM_TIMEFRAMES timeframes[] = {PERIOD_M5};
```

Here we initialize the database handle as an INVALID\_HANDLE. It will be checked next when we open the database.

```
// Database handle
int dbHandle = INVALID_HANDLE;
```

OnStart()

In the Service unique event handler OnStart, we only set the infinite loop and call the UpdateMarketData function, where the real work starts. The sleep function is passed the number of milliseconds to wait between each loop (each request for quotes update). We convert it to minutes to be more user-friendly. Also, we are not expecting updates below the one-minute range.

```
//+------------------------------------------------------------------+
//| Main Service function                                            |
//| Parameters:                                                      |
//|   symbols    - Array of symbol names to update                   |
//|   timeframes - Array of timeframes to update                     |
//|   InpMaxRetries - Maximum number of retries for failed operations    |
//+------------------------------------------------------------------+
void OnStart()
  {
   do
     {
      printf("Updating db: %s", InpDbPath);
      UpdateMarketData(symbols, timeframes, InpMaxRetries);
      Sleep(1000 * 60 * InpUpdateFreq); // 60 secs
     }
   while(!IsStopped());
  }
```

UpdateMarketData()

Here we start calling the function that initializes the database. If anything goes wrong, we return ‘false’ and the control back to the main loop. So, if you are facing troubles with the database initialization, you can safely let the Service run while you fix the database initialization issue. In the next loop, it will try again.

```
//+------------------------------------------------------------------+
//| Update market data for multiple symbols and timeframes           |
//+------------------------------------------------------------------+
bool UpdateMarketData(string &symbols_array[], ENUM_TIMEFRAMES &time_frames[], int max_retries = 3)
  {
// Initialize database
   if(!InitializeDatabase())
     {
      LogMessage("Failed to initialize database");
      return false;
     }
   bool allSuccess = true;

If the database is initialized (open), we start processing each symbol and timeframe.

// Process each symbol
   for(int i = 0; i < ArraySize(symbols_array); i++)
     {
      string symbol = symbols_array[i];
      // Process each timeframe
      for(int j = 0; j < ArraySize(time_frames); j++)
        {
         ENUM_TIMEFRAMES timeframe = time_frames[j];
         int retryCount = 0;
         bool success = false;
```

In this while loop, we control the maximum number of retries and effectively call the function to update the database.

```
         // Retry logic
         while(retryCount < max_retries && !success)
           {
            success = UpdateSymbolTimeframeData(symbol, timeframe);
            if(!success)
              {
               retryCount++;
               Sleep(1000); // Wait before retry
              }
           }
         if(!success)
           {
            LogMessage(StringFormat("Failed to update %s %s after %d retries",
                                    symbol, TimeframeToString(timeframe), max_retries));
            allSuccess = false;
           }
        }
     }
   DatabaseClose(dbHandle);
   return allSuccess;
  }
```

UpdateSymbolTimeframeData()

Since our market\_data table requires a symbol\_id as a foreign key, we need to have it first. So we check for its existence or create a new symbol\_id on the ‘symbol’ table if it is a new symbol.

```
//+------------------------------------------------------------------+
//| Update market data for a single symbol and timeframe              |
//+------------------------------------------------------------------+
bool UpdateSymbolTimeframeData(string symbol, ENUM_TIMEFRAMES timeframe)
  {
   ResetLastError();
// Get symbol ID (insert if it doesn't exist)
   long symbol_id = GetOrInsertSymbol(symbol);
   if(symbol_id == -1)
     {
      LogMessage(StringFormat("Failed to get symbol ID for %s", symbol));
      return false;
     }
```

We convert the timeframe from MQL5 ENUM\_TIMEFRAMES to the string (TEXT) type required by our table.

```
   string tfString = TimeframeToString(timeframe);
   if(tfString == "")
     {
      LogMessage(StringFormat("Unsupported timeframe for symbol %s", symbol));
      return false;
     }
```

We copy the last closed bar rates.

```
// Get the latest closed bar
   MqlRates rates[];
   if(CopyRates(symbol, timeframe, 1, 1, rates) != 1)
     {
      LogMessage(StringFormat("Failed to get rates for %s %s: %d", symbol, tfString, GetLastError()));
      return false;
     }
```

We check if this data point already exists. If yes, we log it and return ‘true’.

```
   if(MarketDataExists(symbol_id, rates[0].time, tfString))
     {
      LogMessage(StringFormat("Data already exists for %s %s at %s",
                              symbol, tfString, TimeToString(rates[0].time)));
      return true;
     }
```

If it is a new data point, a new quote, we start a database transaction to ensure atomicity. If anything fails, we rollback it. If everything goes well, we commit the transaction and return ‘true’.

```
// Start transaction
   if(!DatabaseTransactionBegin(dbHandle))
     {
      LogMessage(StringFormat("Failed to start transaction: %d", GetLastError()));
      return false;
     }
// Insert the new data
   if(!InsertMarketData(symbol_id, tfString, rates[0]))
     {
      DatabaseTransactionRollback(dbHandle);
      return false;
     }
// Commit transaction
   if(!DatabaseTransactionCommit(dbHandle))
     {
      LogMessage(StringFormat("Failed to commit transaction: %d", GetLastError()));
      return false;
     }
   LogMessage(StringFormat("Successfully updated %s %s data for %s",
                           symbol, tfString, TimeToString(rates[0].time)));
   return true;
  }
```

InitializeDatabase()

Here, we open the database and validate the handle.

```
//+------------------------------------------------------------------+
//| Initialize database connection                                   |
//+------------------------------------------------------------------+
bool InitializeDatabase()
  {
   ResetLastError();
// Open database (creates if it doesn't exist)
   dbHandle = DatabaseOpen(InpDbPath, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE);
   if(dbHandle == INVALID_HANDLE)
     {
      LogMessage(StringFormat("Failed to open database: %d", GetLastError()));
      return false;
     }
```

This “PRAGMA” directive enabling foreign\_keys on SQLite is not strictly necessary when working with the integrated SQLite database since we know that it has been compiled with this feature enabled. It is only a safety measure in case you would be working with an external database.

```
// Enable foreign key constraints
   if(!DatabaseExecute(dbHandle, "PRAGMA foreign_keys = ON"))
     {
      LogMessage(StringFormat("Failed to enable foreign keys: %d", GetLastError()));
      return false;
     }
   LogMessage("Database initialized successfully");
   return true;
  }
```

TimeframeToString()

The function to convert MQL5 ENUM\_TIMEFRAMES into a string type is a simple switch.

```
//+------------------------------------------------------------------+
//| Convert MQL5 timeframe to SQLite format                          |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES tf)
  {
   switch(tf)
     {
      case PERIOD_M1:
         return "M1";
      case PERIOD_M2:
         return "M2";
      case PERIOD_M3:
         return "M3";
(...)
      case PERIOD_MN1:
         return "MN1";
      default:
         return "";
     }
  }
```

MarketDataExists()

To check if the market data already exists, we execute a simple query over the market\_data table composite primary key.

```
//+------------------------------------------------------------------+
//| Check if market data exists for given timestamp and timeframe     |
//+------------------------------------------------------------------+
bool MarketDataExists(long symbol_id, datetime tstamp, string timeframe)
  {
   ResetLastError();
   int stmt = DatabasePrepare(dbHandle, "SELECT 1 FROM market_data WHERE symbol_id = ? AND tstamp = ? AND timeframe = ? LIMIT 1");
   if(stmt == INVALID_HANDLE)
     {
      LogMessage(StringFormat("Failed to prepare market data existence check: %d", GetLastError()));
      return false;
     }
   if(!DatabaseBind(stmt, 0, symbol_id) ||
      !DatabaseBind(stmt, 1, (long)tstamp) ||
      !DatabaseBind(stmt, 2, timeframe))
     {
      LogMessage(StringFormat("Failed to bind parameters for existence check: %d", GetLastError()));
      DatabaseFinalize(stmt);
      return false;
     }
   bool exists = DatabaseRead(stmt);
   DatabaseFinalize(stmt);
   return exists;
  }
```

InsertMarketData()

Finally, to insert the new market data (the update), we execute an insert query using the MQL5 StringFormat function as suggested by the documentation.

Be careful with the strings in the VALUES replacement. **You need to quote them**. Thank me later. 🙂

```
//+------------------------------------------------------------------+
//| Insert market data into database                                 |
//+------------------------------------------------------------------+
bool InsertMarketData(long symbol_id, string timeframe, MqlRates &rates)
  {
   ResetLastError();
   string req = StringFormat(
                   "INSERT INTO market_data ("
                   "tstamp, timeframe, price_open, price_high, price_low, price_close, "
                   "tick_volume, real_volume, spread, symbol_id) "
                   "VALUES(%d, '%s', %G, %G, %G, %G, %d, %d, %d, %d)",
                   rates.time, timeframe, rates.open, rates.high, rates.low, rates.close,
                   rates.tick_volume, rates.real_volume, rates.spread, symbol_id);
   if(!DatabaseExecute(dbHandle, req))
     {
      LogMessage(StringFormat("Failed to insert market data: %d", GetLastError()));
      return false;
     }
   return true;
  }
```

While the Service is running…

![Fig.5 - Metaeditor Navigator with the DB Update Service running](https://c.mql5.com/2/164/Capture_service_running.PNG)

Figure 5 - Metaeditor Navigator with the DB Update Service running

… you should see something like this in the Experts log tab.

![Fig.7 - Metatrader 5 Experts log tab with Database Update Service output](https://c.mql5.com/2/164/Capture_db_update_log.PNG)

Figure 7 - Metatrader 5 Experts log tab with Database Update Service output

Your ‘market\_data’ table should be like this. Note that we are storing all market data, for all symbols and timeframes, in a single table for now. Later, we will improve this, but only as the need arises. For now, this is more than enough for us to start with our data analysis more sustainably.

![Fig.8 - Metaeditor integrated SQLite tab view with database updates](https://c.mql5.com/2/164/Capture_db_market_data_table.PNG)

Figure 8 - Metaeditor integrated SQLite tab view with database updates

### Conclusion

In this article, we saw how we are moving from ephemeral downloaded price data to the first database version for our statistical arbitrage framework. We saw the rationale behind its initial schema design, how to initialize, and insert initial data.

All tables, fields, and relationships of the schema are now documented, with table descriptions, constraints, and examples.

We also detailed the steps to keep the database updated by building a Metatrader 5 Service, provided one implementation of this service, along with Python scripts to insert initial data for any available symbol (ticker) and timeframe.

With these tools and without the need to write a single line of code, the average retail trader - the intended audience for our statistical arbitrage framework - can get started storing not only immediate market data (price quotes), but also a bit of metadata about the stocks involved in our cointegration strategy and the deals' history.

This initial design will evolve right in the next step, when we will be updating the portfolio weights in real-time and rotating the portfolio if the basket cointegration becomes weak, by replacing and/or adding symbols without manual intervention.

References

Daniel P. Palomar (2025). [Portfolio Optimization: Theory and Application](https://www.mql5.com/go?link=https://portfoliooptimizationbook.com/portfolio-optimization-book.pdf "https://portfoliooptimizationbook.com/portfolio-optimization-book.pdf"). Cambridge University Press.

\* A notable limitation of SQLite in relation to some time-series oriented databases is the absence of “as-of joins”. Since we will be indexing our market data by the timestamp, we almost certainly will not be able to use “joins” between tables in the future because we use timestamps as primary keys and they seldom, possibly never, will be aligned in two or more tables, and “joins” (inner, left, and outer joins) are dependable on this index alignment.So when using “joins” we would receive empty (null) result sets. This [blog post](https://www.mql5.com/go?link=https://duckdb.org/2023/09/15/asof-joins-fuzzy-temporal-lookups.html "https://duckdb.org/2023/09/15/asof-joins-fuzzy-temporal-lookups.html") explains the issue in detail.

| Filename | Description |
| --- | --- |
| StatArb/db-setup.mq5 | MQL5 Script to create and initialize the SQLite database by reading the schema-0.1.sql schema file. |
| StatArb/db-update-statarb-0.1.mq5 | MQL5 Service to update the SQLite database with the most recent closed price bars. |
| StatArb/schema-0.1.sql | SQL schema file (DDL) to initialize the database (generate tables, fields, and constraints). |
| db\_store\_quotes.ipynb | Jupyter notebook containing Python code. Helper file to fill the SQLite database with symbol quotes from a specific time range and timeframe. |
| .env | Sample file with environment variables to be read from the helper Python file above. (optional) |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19242.zip "Download all attachments in the single ZIP archive")

[mql5-article-files-19242.zip](https://www.mql5.com/en/articles/download/19242/mql5-article-files-19242.zip "Download mql5-article-files-19242.zip")(8.43 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Statistical Arbitrage Through Cointegrated Stocks (Part 10): Detecting Structural Breaks](https://www.mql5.com/en/articles/20946)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://www.mql5.com/en/articles/20657)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 8): Rolling Windows Eigenvector Comparison for Portfolio Rebalancing](https://www.mql5.com/en/articles/20485)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening](https://www.mql5.com/en/articles/19626)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://www.mql5.com/en/articles/19428)

**[Go to discussion](https://www.mql5.com/en/forum/493950)**

![Introduction to MQL5 (Part 20): Introduction to Harmonic Patterns](https://c.mql5.com/2/165/19179-introduction-to-mql5-part-20-logo.png)[Introduction to MQL5 (Part 20): Introduction to Harmonic Patterns](https://www.mql5.com/en/articles/19179)

In this article, we explore the fundamentals of harmonic patterns, their structures, and how they are applied in trading. You’ll learn about Fibonacci retracements, extensions, and how to implement harmonic pattern detection in MQL5, setting the foundation for building advanced trading tools and Expert Advisors.

![From Novice to Expert: Animated News Headline Using MQL5 (IX) — Multiple Symbol Management on a single chart for News Trading](https://c.mql5.com/2/165/19008-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (IX) — Multiple Symbol Management on a single chart for News Trading](https://www.mql5.com/en/articles/19008)

News trading often requires managing multiple positions and symbols within a very short time due to heightened volatility. In today’s discussion, we address the challenges of multi-symbol trading by integrating this feature into our News Headline EA. Join us as we explore how algorithmic trading with MQL5 makes multi-symbol trading more efficient and powerful.

![Self Optimizing Expert Advisors in MQL5 (Part 13): A Gentle Introduction To Control Theory Using Matrix Factorization](https://c.mql5.com/2/165/19132-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 13): A Gentle Introduction To Control Theory Using Matrix Factorization](https://www.mql5.com/en/articles/19132)

Financial markets are unpredictable, and trading strategies that look profitable in the past often collapse in real market conditions. This happens because most strategies are fixed once deployed and cannot adapt or learn from their mistakes. By borrowing ideas from control theory, we can use feedback controllers to observe how our strategies interact with markets and adjust their behavior toward profitability. Our results show that adding a feedback controller to a simple moving average strategy improved profits, reduced risk, and increased efficiency, proving that this approach has strong potential for trading applications.

![Reimagining Classic Strategies (Part 15): Daily Breakout Trading Strategy](https://c.mql5.com/2/165/19130-reimagining-classic-strategies-logo__1.png)[Reimagining Classic Strategies (Part 15): Daily Breakout Trading Strategy](https://www.mql5.com/en/articles/19130)

Human traders had long participated in financial markets before the rise of computers, developing rules of thumb that guided their decisions. In this article, we revisit a well-known breakout strategy to test whether such market logic, learned through experience, can hold its own against systematic methods. Our findings show that while the original strategy produced high accuracy, it suffered from instability and poor risk control. By refining the approach, we demonstrate how discretionary insights can be adapted into more robust, algorithmic trading strategies.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hvzczhrmzugjuaisrhuunlscpvfprjgk&ssn=1769251609565736092&ssn_dr=0&ssn_sr=0&fv_date=1769251609&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19242&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Statistical%20Arbitrage%20Through%20Cointegrated%20Stocks%20(Part%203)%3A%20Database%20Setup%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925160923258936&fz_uniq=5083103274158003480&sv=2552)

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