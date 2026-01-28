---
title: Library for easy and quick development of MetaTrader programs (part XV): Collection of symbol objects
url: https://www.mql5.com/en/articles/7041
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:40:04.986423
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/7041&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070467785315784288)

MetaTrader 5 / Examples


### Contents

- [Symbol collection concept](https://www.mql5.com/en/articles/7041#node01)
- [Descendant objects of the "symbol" basic abstract object](https://www.mql5.com/en/articles/7041#node02)
- [Symbol collection class](https://www.mql5.com/en/articles/7041#node03)
- [Symbol collection test](https://www.mql5.com/en/articles/7041#node04)
- [What's next?](https://www.mql5.com/en/articles/7041#node05)


### Symbol collection concept

I have already defined the concept of constructing object collection classes in the [third \\
part of the library description](https://www.mql5.com/en/articles/5687). Here, I am going to adhere to the adopted data storage structure. This means we need to create a list for
the symbol collection. The list is to store descendant objects of the "symbol" class created in the

[previous article](https://www.mql5.com/en/articles/7014). The abstract symbol descendants are to clarify a symbol data and
define the availability of the basic symbol object properties in a program. Such symbol objects are to be distinguished by their
affiliation with groups (symbol status).

- **Forex symbol** — all Forex symbols not falling into the following Forex symbol categories:

- Major Forex symbol — the custom category of the most used forex symbols

- Minor Forex symbol — the custom category of less used Forex symbols
- Exotic Forex symbol — the custom category of rarely used Forex symbols
- Forex symbol/RUB — the custom category of Forex symbols featuring RUB

- Metal — the custom category of metal symbols
- Index — the custom category of index symbols
- Indicative — the custom category of indicative symbols
- Cryptocurrency symbol — the custom category of cryptocurrency symbols
- Commodity symbol — the custom category of commodity symbols
- **Exchange symbol** — all exchange symbols not falling into the following categories of exchange symbols:
- Futures

- CFD

- Security

- Bond
- Option
- **Non-tradable asset**
- **Custom symbol**
- **General category** — symbols that do not fall into the above categories

In order to define a group a symbol belongs to (symbol status), we will create custom data sets — arrays of the names of symbols where we are
to search for a necessary category in the first place. If a symbol is not found in a custom category (its name is not present in any of
user-defined symbol name arrays), then the category is defined by the "margin calculation method" (

[ENUM\_SYMBOL\_CALC\_MODE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_calc_mode)) symbol
property allowing us to define whether the symbol belongs to some of the categories listed above. In other words, we conduct a search as
two checks: the first one is performed in custom categories. If failed to define a category, it is searched for using the margin
calculation method for a symbol. Previously, I planned to use yet another method — defining by a name of a folder where the symbol is
located on the server's symbol directory tree. But this method is too unreliable since folders may have any names and they may be located
on different servers for the same symbol. So, I decided not to use it.



A custom category is to have a greater priority — if a user wants a symbol (for example, USDUSC) to be located in the "major" category,
then nothing can prevent him/her from that despite the fact that this is an indicative.

Now that we have decided on the categories, let's create the necessary abstract symbol descendant classes for them. The abstract symbol
itself was created in the

[previous article](https://www.mql5.com/en/articles/7014).

In order to store all symbol objects, we need to use the
CListObj class inherited from the

[CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) standard library class we have
considered in the

[fifth article](https://www.mql5.com/en/articles/6211#node01) when discussing the re-arrangement of the library
classes. In the library-based programs, there will be an ability to select a symbol list to work with:

1. Only one — the current symbol the program is attached to
2. A predefined symbol set specified in the program
3. Working with a list of symbols located in the Market Watch window
4. Working with a complete list of symbols available on the server

Thus, we will cover most of the necessary programming tasks for accessing working symbols.

Here I should
mention two things:



1. working with a symbol list from the market watch — in this mode, we need to use the search for the Market Watch window events — to
    respond to its changes in a timely manner (adding/removing a symbol from the list and sorting it with the mouse),
2. when working with the complete list of symbols available on the server, we need to arrange a normal handling of a possible great
    number of available symbols since the complete list of symbols on the server can be viewed during the first launch. Besides,
    symbol objects, for which all properties should be received, are created. The process takes some time. In my case, creating
    the collection of all symbols located on the server took about two minutes on my mid-tier laptop.

Initially, when choosing this method of work, it is necessary to at least warn users about the time spent collecting initial information.


We will do that, as well as implement tracking symbol events and Market Watch window events in subsequent articles. For now, let's
create a collection list.

For a start, let's create another include file where we will store all the data needed for the library — arrays of custom symbol
groups, files, images, and other subsequently needed data sets for the library.



In the **\\MQL5\\Include\\DoEasy\** library folder, create a new **Datas.mqh** include file and add some
arrays to it I have already prepared in advance while going past several accounts and collecting data on symbol groups set on
different servers from the Market Watch window tree structure:

```
//+------------------------------------------------------------------+
//|                                                        Datas.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Data sets                                                        |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Major Forex symbols                                              |
//+------------------------------------------------------------------+
string DataSymbolsFXMajors[]=
  {
   "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD","CADCHF","CADJPY","CHFJPY","EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD",
   "EURUSD","GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPNZD","GBPUSD","NZDCAD","NZDCHF","NZDJPY","NZDUSD","USDCAD","USDCHF","USDJPY"
  };
//+------------------------------------------------------------------+
//| Minor Forex symbols                                              |
//+------------------------------------------------------------------+
string DataSymbolsFXMinors[]=
  {
   "EURCZK","EURDKK","EURHKD","EURNOK","EURPLN","EURSEK","EURSGD","EURTRY","EURZAR","GBPSEK","GBPSGD"
      ,"NZDSGD","USDCZK","USDDKK","USDHKD","USDNOK","USDPLN","USDSEK","USDSGD","USDTRY","USDZAR"
  };
//+------------------------------------------------------------------+
//| Exotic Forex symbols                                             |
//+------------------------------------------------------------------+
string DataSymbolsFXExotics[]=
  {
   "EURMXN","USDCNH","USDMXN","EURTRY","USDTRY"
  };
//+------------------------------------------------------------------+
//| Forex RUB symbols                                                |
//+------------------------------------------------------------------+
string DataSymbolsFXRub[]=
  {
   "EURRUB","USDRUB"
  };
//+------------------------------------------------------------------+
//| Indicative Forex symbols                                         |
//+------------------------------------------------------------------+
string DataSymbolsFXIndicatives[]=
  {
   "EUREUC","USDEUC","USDUSC"
  };
//+------------------------------------------------------------------+
//| Metal symbols                                                    |
//+------------------------------------------------------------------+
string DataSymbolsMetalls[]=
  {
   "XAGUSD","XAUUSD"
  };
//+------------------------------------------------------------------+
//| Commodity symbols                                                |
//+------------------------------------------------------------------+
string DataSymbolsCommodities[]=
  {
   "BRN","WTI","NG"
  };
//+------------------------------------------------------------------+
//| Indices                                                          |
//+------------------------------------------------------------------+
string DataSymbolsIndexes[]=
  {
   "CAC40","HSI50","ASX200","STOXX50","NQ100","FTSE100","DAX30","IBEX35","SPX500","NIKK225"
   "Volatility 10 Index","Volatility 25 Index","Volatility 50 Index","Volatility 75 Index","Volatility 100 Index",
   "HF Volatility 10 Index","HF Volatility 50 Index","Crash 1000 Index","Boom 1000 Index","Step Index"
  };
//+------------------------------------------------------------------+
//| Cryptocurrency Symbols                                           |
//+------------------------------------------------------------------+
string DataSymbolsCrypto[]=
  {
   "BCHUSD","BTCEUR","BTCUSD","DSHUSD","EOSUSD","ETHEUR","ETHUSD","LTCUSD","XRPUSD"
  };
//+------------------------------------------------------------------+
//| Options                                                          |
//+------------------------------------------------------------------+
string DataSymbolsOptions[]=
  {
   "BO Volatility 10 Index","BO Volatility 25 Index","BO Volatility 50 Index","BO Volatility 75 Index","BO Volatility 100 Index"
  };
//+------------------------------------------------------------------+
```

As we can see from the listing, this is just an enumeration of symbol names added to required arrays defining a group of symbols located in each
of the arrays. If desired and necessary, you can handle the contents of these symbol name arrays by sending some names to another array,
adding/removing some names, etc.

More "tight" testing of the abstract symbol object behavior provides understanding that using the SYMBOL\_MARGIN\_LONG,
SYMBOL\_MARGIN\_SHORT, SYMBOL\_MARGIN\_STOP, SYMBOL\_MARGIN\_LIMIT and SYMBOL\_MARGIN\_STOPLIMIT constants has not yielded the symbol
properties. Therefore, I have had to implement receiving the properties using the

[SymbolInfoMarginRate()](https://www.mql5.com/en/docs/marketinformation/symbolinfomarginrate) function.


Since an order type is sent to the function, I have had to

**create a custom constant in symbol object real properties in**
**the Defines.mqh file for each of the order types:**

```
//+------------------------------------------------------------------+
//| Symbol real properties                                           |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_PROP_DOUBLE
  {
   SYMBOL_PROP_BID = SYMBOL_PROP_INTEGER_TOTAL,             // Bid - the best price at which a symbol can be sold
   SYMBOL_PROP_BIDHIGH,                                     // The highest Bid price of the day
   SYMBOL_PROP_BIDLOW,                                      // The lowest Bid price of the day
   SYMBOL_PROP_ASK,                                         // Ask - best price, at which an instrument can be bought
   SYMBOL_PROP_ASKHIGH,                                     // The highest Ask price of the day
   SYMBOL_PROP_ASKLOW,                                      // The lowest Ask price of the day
   SYMBOL_PROP_LAST,                                        // The price at which the last deal was executed
   SYMBOL_PROP_LASTHIGH,                                    // The highest Last price of the day
   SYMBOL_PROP_LASTLOW,                                     // The lowest Last price of the day
   SYMBOL_PROP_VOLUME_REAL,                                 // Volume of the day
   SYMBOL_PROP_VOLUMEHIGH_REAL,                             // Maximum Volume within a day
   SYMBOL_PROP_VOLUMELOW_REAL,                              // Minimum Volume within a day
   SYMBOL_PROP_OPTION_STRIKE,                               // Option execution price
   SYMBOL_PROP_POINT,                                       // One point value
   SYMBOL_PROP_TRADE_TICK_VALUE,                            // SYMBOL_TRADE_TICK_VALUE_PROFIT value
   SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT,                     // Calculated tick value for a winning position
   SYMBOL_PROP_TRADE_TICK_VALUE_LOSS,                       // Calculated tick value for a losing position
   SYMBOL_PROP_TRADE_TICK_SIZE,                             // Minimum price change
   SYMBOL_PROP_TRADE_CONTRACT_SIZE,                         // Trade contract size
   SYMBOL_PROP_TRADE_ACCRUED_INTEREST,                      // Accrued interest
   SYMBOL_PROP_TRADE_FACE_VALUE,                            // Face value – initial bond value set by an issuer
   SYMBOL_PROP_TRADE_LIQUIDITY_RATE,                        // Liquidity rate – the share of an asset that can be used for a margin
   SYMBOL_PROP_VOLUME_MIN,                                  // Minimum volume for a deal
   SYMBOL_PROP_VOLUME_MAX,                                  // Maximum volume for a deal
   SYMBOL_PROP_VOLUME_STEP,                                 // Minimum volume change step for deal execution
   SYMBOL_PROP_VOLUME_LIMIT,                                // The maximum allowed total volume of an open position and pending orders in one direction (either buy or sell)
   SYMBOL_PROP_SWAP_LONG,                                   // Long swap value
   SYMBOL_PROP_SWAP_SHORT,                                  // Short swap value
   SYMBOL_PROP_MARGIN_INITIAL,                              // Initial margin
   SYMBOL_PROP_MARGIN_MAINTENANCE,                          // Maintenance margin for an instrument
   SYMBOL_PROP_MARGIN_LONG_INITIAL,                         // Initial margin requirement applicable to long positions
   SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL,                     // Initial margin requirement applicable to BuyStop orders
   SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL,                    // Initial margin requirement applicable to BuyLimit orders
   SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL,                // Initial margin requirement applicable to BuyStopLimit orders
   SYMBOL_PROP_MARGIN_LONG_MAINTENANCE,                     // Maintenance margin requirement applicable to long positions
   SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE,                 // Maintenance margin requirement applicable to BuyStop orders
   SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE,                // Maintenance margin requirement applicable to BuyLimit orders
   SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE,            // Maintenance margin requirement applicable to BuyStopLimit orders
   SYMBOL_PROP_MARGIN_SHORT_INITIAL,                        // Initial margin requirement applicable to short positions
   SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL,                    // Initial margin requirement applicable to SellStop orders
   SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL,                   // Initial margin requirement applicable to SellLimit orders
   SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL,               // Initial margin requirement applicable to SellStopLimit orders
   SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE,                    // Maintenance margin requirement applicable to short positions
   SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE,                // Maintenance margin requirement applicable to SellStop orders
   SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE,               // Maintenance margin requirement applicable to SellLimit orders
   SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE,           // Maintenance margin requirement applicable to SellStopLimit orders
   SYMBOL_PROP_SESSION_VOLUME,                              // The total volume of deals in the current session
   SYMBOL_PROP_SESSION_TURNOVER,                            // The total turnover in the current session
   SYMBOL_PROP_SESSION_INTEREST,                            // The total volume of open positions
   SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME,                   // The total volume of Buy orders at the moment
   SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME,                  // The total volume of Sell orders at the moment
   SYMBOL_PROP_SESSION_OPEN,                                // Open price of the session
   SYMBOL_PROP_SESSION_CLOSE,                               // Close price of the session
   SYMBOL_PROP_SESSION_AW,                                  // The average weighted price of the session
   SYMBOL_PROP_SESSION_PRICE_SETTLEMENT,                    // The settlement price of the current session
   SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN,                     // Minimum allowable price value for the session
   SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX,                     // Maximum allowable price value for the session
   SYMBOL_PROP_MARGIN_HEDGED                                // Size of a contract or margin for one lot of hedged positions (oppositely directed positions at one symbol).
  };
#define SYMBOL_PROP_DOUBLE_TOTAL     (58)                   // Total number of real properties
#define SYMBOL_PROP_DOUBLE_SKIP      (0)                    // Number of real symbol properties not used in sorting
//+------------------------------------------------------------------+
```

Accordingly, the total number of real properties has increased to **58**(instead of the previous 47 ones).

I have also had to add the corresponding constants to the enumeration of possible symbol sorting criteria:

```
//+------------------------------------------------------------------+
//| Possible symbol sorting criteria                                 |
//+------------------------------------------------------------------+
#define FIRST_SYM_DBL_PROP          (SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_INTEGER_SKIP)
#define FIRST_SYM_STR_PROP          (SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_INTEGER_SKIP+SYMBOL_PROP_DOUBLE_TOTAL-SYMBOL_PROP_DOUBLE_SKIP)
enum ENUM_SORT_SYMBOLS_MODE
  {
//--- Sort by integer properties
   SORT_BY_SYMBOL_STATUS = 0,                               // Sort by symbol status
   SORT_BY_SYMBOL_CUSTOM,                                   // Sort by custom symbol property
   SORT_BY_SYMBOL_CHART_MODE,                               // Sort by price type for constructing bars – Bid or Last (from the ENUM_SYMBOL_CHART_MODE enumeration)
   SORT_BY_SYMBOL_EXIST,                                    // Sort by the flag that a symbol with such a name exists
   SORT_BY_SYMBOL_SELECT,                                   // Sort by the flag indicating that a symbol is selected in Market Watch
   SORT_BY_SYMBOL_VISIBLE,                                  // Sort by the flag indicating that a selected symbol is displayed in Market Watch
   SORT_BY_SYMBOL_SESSION_DEALS,                            // Sort by the number of deals in the current session
   SORT_BY_SYMBOL_SESSION_BUY_ORDERS,                       // Sort by the total number of current buy orders
   SORT_BY_SYMBOL_SESSION_SELL_ORDERS,                      // Sort by the total number of current sell orders
   SORT_BY_SYMBOL_VOLUME,                                   // Sort by last deal volume
   SORT_BY_SYMBOL_VOLUMEHIGH,                               // Sort by maximum volume for a day
   SORT_BY_SYMBOL_VOLUMELOW,                                // Sort by minimum volume for a day
   SORT_BY_SYMBOL_TIME,                                     // Sort by the last quote time
   SORT_BY_SYMBOL_DIGITS,                                   // Sort by a number of decimal places
   SORT_BY_SYMBOL_DIGITS_LOT,                               // Sort by a number of decimal places in a lot
   SORT_BY_SYMBOL_SPREAD,                                   // Sort by spread in points
   SORT_BY_SYMBOL_SPREAD_FLOAT,                             // Sort by floating spread
   SORT_BY_SYMBOL_TICKS_BOOKDEPTH,                          // Sort by a maximum number of requests displayed in the market depth
   SORT_BY_SYMBOL_TRADE_CALC_MODE,                          // Sort by contract price calculation method (from the ENUM_SYMBOL_CALC_MODE enumeration)
   SORT_BY_SYMBOL_TRADE_MODE,                               // Sort by order execution type (from the ENUM_SYMBOL_TRADE_MODE enumeration)
   SORT_BY_SYMBOL_START_TIME,                               // Sort by an instrument trading start date (usually used for futures)
   SORT_BY_SYMBOL_EXPIRATION_TIME,                          // Sort by an instrument trading end date (usually used for futures)
   SORT_BY_SYMBOL_TRADE_STOPS_LEVEL,                        // Sort by the minimum indent from the current close price (in points) for setting Stop orders
   SORT_BY_SYMBOL_TRADE_FREEZE_LEVEL,                       // Sort by trade operation freeze distance (in points)
   SORT_BY_SYMBOL_TRADE_EXEMODE,                            // Sort by trade execution mode (from the ENUM_SYMBOL_TRADE_EXECUTION enumeration)
   SORT_BY_SYMBOL_SWAP_MODE,                                // Sort by swap calculation model (from the ENUM_SYMBOL_SWAP_MODE enumeration)
   SORT_BY_SYMBOL_SWAP_ROLLOVER3DAYS,                       // Sort by week day for accruing a triple swap (from the ENUM_DAY_OF_WEEK enumeration)
   SORT_BY_SYMBOL_MARGIN_HEDGED_USE_LEG,                    // Sort by the calculation mode of a hedged margin using the larger leg (Buy or Sell)
   SORT_BY_SYMBOL_EXPIRATION_MODE,                          // Sort by flags of allowed order expiration modes
   SORT_BY_SYMBOL_FILLING_MODE,                             // Sort by flags of allowed order filling modes
   SORT_BY_SYMBOL_ORDER_MODE,                               // Sort by flags of allowed order types
   SORT_BY_SYMBOL_ORDER_GTC_MODE,                           // Sort by StopLoss and TakeProfit orders lifetime
   SORT_BY_SYMBOL_OPTION_MODE,                              // Sort by option type (from the ENUM_SYMBOL_OPTION_MODE enumeration)
   SORT_BY_SYMBOL_OPTION_RIGHT,                             // Sort by option right (Call/Put) (from the ENUM_SYMBOL_OPTION_RIGHT enumeration)
//--- Sort by real properties
   SORT_BY_SYMBOL_BID = FIRST_SYM_DBL_PROP,                 // Sort by Bid
   SORT_BY_SYMBOL_BIDHIGH,                                  // Sort by maximum Bid for a day
   SORT_BY_SYMBOL_BIDLOW,                                   // Sort by minimum Bid for a day
   SORT_BY_SYMBOL_ASK,                                      // Sort by Ask
   SORT_BY_SYMBOL_ASKHIGH,                                  // Sort by maximum Ask for a day
   SORT_BY_SYMBOL_ASKLOW,                                   // Sort by minimum Ask for a day
   SORT_BY_SYMBOL_LAST,                                     // Sort by the last deal price
   SORT_BY_SYMBOL_LASTHIGH,                                 // Sort by maximum Last for a day
   SORT_BY_SYMBOL_LASTLOW,                                  // Sort by minimum Last for a day
   SORT_BY_SYMBOL_VOLUME_REAL,                              // Sort by Volume for a day
   SORT_BY_SYMBOL_VOLUMEHIGH_REAL,                          // Sort by maximum Volume for a day
   SORT_BY_SYMBOL_VOLUMELOW_REAL,                           // Sort by minimum Volume for a day
   SORT_BY_SYMBOL_OPTION_STRIKE,                            // Sort by an option execution price
   SORT_BY_SYMBOL_POINT,                                    // Sort by a single point value
   SORT_BY_SYMBOL_TRADE_TICK_VALUE,                         // Sort by SYMBOL_TRADE_TICK_VALUE_PROFIT value
   SORT_BY_SYMBOL_TRADE_TICK_VALUE_PROFIT,                  // Sort by a calculated tick price for a profitable position
   SORT_BY_SYMBOL_TRADE_TICK_VALUE_LOSS,                    // Sort by a calculated tick price for a loss-making position
   SORT_BY_SYMBOL_TRADE_TICK_SIZE,                          // Sort by a minimum price change
   SORT_BY_SYMBOL_TRADE_CONTRACT_SIZE,                      // Sort by a trading contract size
   SORT_BY_SYMBOL_TRADE_ACCRUED_INTEREST,                   // Sort by accrued interest
   SORT_BY_SYMBOL_TRADE_FACE_VALUE,                         // Sort by face value
   SORT_BY_SYMBOL_TRADE_LIQUIDITY_RATE,                     // Sort by liquidity rate
   SORT_BY_SYMBOL_VOLUME_MIN,                               // Sort by a minimum volume for performing a deal
   SORT_BY_SYMBOL_VOLUME_MAX,                               // Sort by a maximum volume for performing a deal
   SORT_BY_SYMBOL_VOLUME_STEP,                              // Sort by a minimum volume change step for deal execution
   SORT_BY_SYMBOL_VOLUME_LIMIT,                             // Sort by a maximum allowed aggregate volume of an open position and pending orders in one direction
   SORT_BY_SYMBOL_SWAP_LONG,                                // Sort by a long swap value
   SORT_BY_SYMBOL_SWAP_SHORT,                               // Sort by a short swap value
   SORT_BY_SYMBOL_MARGIN_INITIAL,                           // Sort by an initial margin
   SORT_BY_SYMBOL_MARGIN_MAINTENANCE,                       // Sort by a maintenance margin for an instrument
   SORT_BY_SYMBOL_MARGIN_LONG_INITIAL,                      // Sort by initial margin requirement applicable to Long orders
   SORT_BY_SYMBOL_MARGIN_BUY_STOP_INITIAL,                  // Sort by initial margin requirement applicable to BuyStop orders
   SORT_BY_SYMBOL_MARGIN_BUY_LIMIT_INITIAL,                 // Sort by initial margin requirement applicable to BuyLimit orders
   SORT_BY_SYMBOL_MARGIN_BUY_STOPLIMIT_INITIAL,             // Sort by initial margin requirement applicable to BuyStopLimit orders
   SORT_BY_SYMBOL_MARGIN_LONG_MAINTENANCE,                  // Sort by maintenance margin requirement applicable to Long orders
   SORT_BY_SYMBOL_MARGIN_BUY_STOP_MAINTENANCE,              // Sort by maintenance margin requirement applicable to BuyStop orders
   SORT_BY_SYMBOL_MARGIN_BUY_LIMIT_MAINTENANCE,             // Sort by maintenance margin requirement applicable to BuyLimit orders
   SORT_BY_SYMBOL_MARGIN_BUY_STOPLIMIT_MAINTENANCE,         // Sort by maintenance margin requirement applicable to BuyStopLimit orders
   SORT_BY_SYMBOL_MARGIN_SHORT_INITIAL,                     // Sort by initial margin requirement applicable to Short orders
   SORT_BY_SYMBOL_MARGIN_SELL_STOP_INITIAL,                 // Sort by initial margin requirement applicable to SellStop orders
   SORT_BY_SYMBOL_MARGIN_SELL_LIMIT_INITIAL,                // Sort by initial margin requirement applicable to SellLimit orders
   SORT_BY_SYMBOL_MARGIN_SELL_STOPLIMIT_INITIAL,            // Sort by initial margin requirement applicable to SellStopLimit orders
   SORT_BY_SYMBOL_MARGIN_SHORT_MAINTENANCE,                 // Sort by maintenance margin requirement applicable to Short orders
   SORT_BY_SYMBOL_MARGIN_SELL_STOP_MAINTENANCE,             // Sort by maintenance margin requirement applicable to SellStop orders
   SORT_BY_SYMBOL_MARGIN_SELL_LIMIT_MAINTENANCE,            // Sort by maintenance margin requirement applicable to SellLimit orders
   SORT_BY_SYMBOL_MARGIN_SELL_STOPLIMIT_MAINTENANCE,        // Sort by maintenance margin requirement applicable to SellStopLimit orders
   SORT_BY_SYMBOL_SESSION_VOLUME,                           // Sort by summary volume of the current session deals
   SORT_BY_SYMBOL_SESSION_TURNOVER,                         // Sort by the summary turnover of the current session
   SORT_BY_SYMBOL_SESSION_INTEREST,                         // Sort by the summary open interest
   SORT_BY_SYMBOL_SESSION_BUY_ORDERS_VOLUME,                // Sort by the current volume of Buy orders
   SORT_BY_SYMBOL_SESSION_SELL_ORDERS_VOLUME,               // Sort by the current volume of Sell orders
   SORT_BY_SYMBOL_SESSION_OPEN,                             // Sort by a session Open price
   SORT_BY_SYMBOL_SESSION_CLOSE,                            // Sort by a session Close price
   SORT_BY_SYMBOL_SESSION_AW,                               // Sort by an average weighted price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_SETTLEMENT,                 // Sort by a settlement price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_LIMIT_MIN,                  // Sort by a minimum price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_LIMIT_MAX,                  // Sort by a maximum price of the current session
   SORT_BY_SYMBOL_MARGIN_HEDGED,                            // Sort by a contract size or a margin value per one lot of hedged positions
//--- Sort by string properties
   SORT_BY_SYMBOL_NAME = FIRST_SYM_STR_PROP,                // Sort by a symbol name
   SORT_BY_SYMBOL_BASIS,                                    // Sort by an underlying asset of a derivative
   SORT_BY_SYMBOL_CURRENCY_BASE,                            // Sort by a base currency of a symbol
   SORT_BY_SYMBOL_CURRENCY_PROFIT,                          // Sort by a profit currency
   SORT_BY_SYMBOL_CURRENCY_MARGIN,                          // Sort by a margin currency
   SORT_BY_SYMBOL_BANK,                                     // Sort by a feeder of the current quote
   SORT_BY_SYMBOL_DESCRIPTION,                              // Sort by a symbol string description
   SORT_BY_SYMBOL_FORMULA,                                  // Sort by the formula used for custom symbol pricing
   SORT_BY_SYMBOL_ISIN,                                     // Sort by the name of a symbol in the ISIN system
   SORT_BY_SYMBOL_PAGE,                                     // Sort by an address of the web page containing symbol information
   SORT_BY_SYMBOL_PATH                                      // Sort by a path in the symbol tree
  };
//+------------------------------------------------------------------+
```

Since we are editing the Defines.mqh file, let's add all we need to it while explaining why we do that along the way.

We need a timer to update data for all symbols within the collection. Note that we need to update quote data of all symbols along with data that
may change to track them in the symbol events class (more on that in the next article). In the timer, we also need to check the symbol list in the
Market Watch window to respond to its changes and update the collection list in a timely manner.

We need to update the quote data more often than the remaining symbol data and its list in the Market Watch window. This means that we need
two timers for the symbol collection — the quote data timer and the timer for other actions performed with symbol lists.

**Let's add the necessary macro substitutions for two symbol collection timers:**

```
//--- Symbol collection timer 1 parameters
#define COLLECTION_SYM_PAUSE1          (100)                      // Pause of the symbol collection timer 1 in milliseconds (for scanning market watch symbols)
#define COLLECTION_SYM_COUNTER_STEP1   (16)                       // Increment of the symbol timer 1 counter
#define COLLECTION_SYM_COUNTER_ID1     (3)                        // Symbol timer 1 counter ID
//--- Symbol collection timer 2 parameters
#define COLLECTION_SYM_PAUSE2          (300)                      // Pause of the symbol collection timer 2 in milliseconds (for events of the market watch symbol list)
#define COLLECTION_SYM_COUNTER_STEP2   (16)                       // Increment of the symbol timer 2 counter
#define COLLECTION_SYM_COUNTER_ID2     (4)                        // Symbol timer 2 counter ID
```

The difference between these data is only in the pause for
each timer and their

IDs — for the first timer, the pause is 100 milliseconds, while for the
second one, it is 300 milliseconds.

We have a separate ID for each collection. The symbol collection is no exception.

**Let's set a custom ID**
**for its list:**

```
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x7778+1)                 // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x7778+2)                 // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x7778+3)                 // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x7778+4)                 // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x7778+5)                 // Symbol collection list ID
```

[In the previous article](https://www.mql5.com/en/articles/7014), in order to define a background color used to
highlight a symbol in the Market Watch window and display its string description, we

compared the color with [clrWhite](https://www.mql5.com/en/docs/constants/objectconstants/webcolors)
— if the property value exceeds a 'long' value of the color, the background color is considered not set:

```
      property==SYMBOL_PROP_BACKGROUND_COLOR    ?  TextByLanguage("Цвет фона символа в Market Watch","Background color of symbol in Market Watch")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         #ifdef __MQL5__
         (this.GetProperty(property)>clrWhite  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": "+::ColorToString((color)this.GetProperty(property),true))
         #else TextByLanguage(": Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
```

However, according to the developers, this is incorrect — a 'long' value of a symbol background in the Market Watch window may exceed a 'long' value
of the "white" color. This means that such a check returns incorrect results in some cases.

To correctly identify the absence of the background color, we need to compare the property value with the CLR\_DEFAULT and CLR\_NONE
values.

**Let's use the macro substitution to set the "default" color** (the "absent" CLR\_NONE color is already present in MQL5
and MQL4):

```
//--- Symbol parameters
#define CLR_DEFAULT                    (0xFF000000)               // Default color
//+------------------------------------------------------------------+
```

As a result, the section of the Defines file macro substitutions now looks as follows:

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- Describe the function with the error line number
#define DFUN_ERR_LINE                  (__FUNCTION__+(TerminalInfoString(TERMINAL_LANGUAGE)=="Russian" ? ", Page " : ", Line ")+(string)__LINE__+": ")
#define DFUN                           (__FUNCTION__+": ")        // "Function description"
#define COUNTRY_LANG                   ("Russian")                // Country language
#define END_TIME                       (D'31.12.3000 23:59:59')   // End date for account history data requests
#define TIMER_FREQUENCY                (16)                       // Minimal frequency of the library timer in milliseconds
//--- Parameters of the orders and deals collection timer
#define COLLECTION_ORD_PAUSE           (250)                      // Orders and deals collection timer pause in milliseconds
#define COLLECTION_ORD_COUNTER_STEP    (16)                       // Increment of the orders and deals collection timer counter
#define COLLECTION_ORD_COUNTER_ID      (1)                        // Orders and deals collection timer counter ID
//--- Parameters of the account collection timer
#define COLLECTION_ACC_PAUSE           (1000)                     // Account collection timer pause in milliseconds
#define COLLECTION_ACC_COUNTER_STEP    (16)                       // Account timer counter increment
#define COLLECTION_ACC_COUNTER_ID      (2)                        // Account timer counter ID
//--- Symbol collection timer 1 parameters
#define COLLECTION_SYM_PAUSE1          (100)                      // Pause of the symbol collection timer 1 in milliseconds (for scanning market watch symbols)
#define COLLECTION_SYM_COUNTER_STEP1   (16)                       // Increment of the symbol timer 1 counter
#define COLLECTION_SYM_COUNTER_ID1     (3)                        // Symbol timer 1 counter ID
//--- Symbol collection timer 2 parameters
#define COLLECTION_SYM_PAUSE2          (300)                      // Pause of the symbol collection timer 2 in milliseconds (for events of the market watch symbol list)
#define COLLECTION_SYM_COUNTER_STEP2   (16)                       // Increment of the symbol timer 2 counter
#define COLLECTION_SYM_COUNTER_ID2     (4)                        // Symbol timer 2 counter ID
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x7778+1)                 // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x7778+2)                 // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x7778+3)                 // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x7778+4)                 // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x7778+5)                 // Symbol collection list ID
//--- Data parameters for file operations
#define DIRECTORY                      ("DoEasy\\")               // Library directory for storing object folders
//--- Symbol parameters
#define CLR_DEFAULT                    (0xFF000000)               // Default color
//+------------------------------------------------------------------+
```

I have already mentioned the modes of working with the symbol collection: working with the current symbol, working with a list of symbols
pre-defined in the program, working with the Market Watch window and working with the full list of symbols available on the server.

**Let's set all these modes in the enumeration:**

```
//+------------------------------------------------------------------+
//| Data for working with symbols                                    |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Modes of working with symbols                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOLS_MODE
  {
   SYMBOLS_MODE_CURRENT,                                    // Work with the current symbol only
   SYMBOLS_MODE_DEFINES,                                    // Work with the specified symbol list
   SYMBOLS_MODE_MARKET_WATCH,                               // Work with the Market Watch window symbols
   SYMBOLS_MODE_ALL                                         // Work with the full symbol list
  };
//+------------------------------------------------------------------+
```

[In the previous article, we identified the symbol categories](https://www.mql5.com/en/articles/7014#node01)
to be used to sort symbols. At the very beginning of this article, we have examined a slightly expanded list of symbol categories we are going
to use.

**Let's add the enumeration of symbol categories (states):**

```
//+------------------------------------------------------------------+
//| Abstract symbol type (status)                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_STATUS
  {
   SYMBOL_STATUS_FX,                                        // Forex symbol
   SYMBOL_STATUS_FX_MAJOR,                                  // Major Forex symbol
   SYMBOL_STATUS_FX_MINOR,                                  // Minor Forex symbol
   SYMBOL_STATUS_FX_EXOTIC,                                 // Exotic Forex symbol
   SYMBOL_STATUS_FX_RUB,                                    // Forex symbol/RUB
   SYMBOL_STATUS_METAL,                                     // Metal
   SYMBOL_STATUS_INDEX,                                     // Index
   SYMBOL_STATUS_INDICATIVE,                                // Indicative
   SYMBOL_STATUS_CRYPTO,                                    // Cryptocurrency symbol
   SYMBOL_STATUS_COMMODITY,                                 // Commodity symbol
   SYMBOL_STATUS_EXCHANGE,                                  // Exchange symbol
   SYMBOL_STATUS_FUTURES,                                   // Futures
   SYMBOL_STATUS_CFD,                                       // CFD
   SYMBOL_STATUS_STOCKS,                                    // Security
   SYMBOL_STATUS_BONDS,                                     // Bond
   SYMBOL_STATUS_OPTION,                                    // Option
   SYMBOL_STATUS_COLLATERAL,                                // Non-tradable asset
   SYMBOL_STATUS_CUSTOM,                                    // Custom symbol
   SYMBOL_STATUS_COMMON                                     // General category
  };
//+------------------------------------------------------------------+
```

We have prepared data for the symbol collection and made all changes in the Defines.mqh file.

The changes also affected the CSymbol class we created in the [previous \\
article](https://www.mql5.com/en/articles/7014#node01).

Since we now receive data on [margin \\
ratios](https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex#rate "https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex#rate") for various order types using the [SymbolInfoMarginRate()](https://www.mql5.com/en/docs/marketinformation/symbolinfomarginrate)
function, while the variables passed to the function by the link are used to return values requested from it, we now need to create these variables.

Assuming that we have eight orders and two ratio types for each of them (initial and maintenance margin ratios), there should be 16 variables for
receiving these values. Therefore, it would be more convenient to create the structure consisting of two nested structures:

in the first structure, two double variables are defined for storing
the ratios of charging initial and maintenance margins, while

the second one contains the first
declared structures for storing data by order types the ratios should be obtained for.

Let's declare these structures and the class member variable with the
second structure type in the CSymbol class private section of the Symbol.mqh symbol class file:

```
//+------------------------------------------------------------------+
//| Abstract symbol class                                            |
//+------------------------------------------------------------------+
class CSymbol : public CObject
  {
private:
   struct SMarginRate
     {
      double         Initial;          // initial margin rate
      double         Maintenance;      // maintenance margin rate
     };
   struct SMarginRateMode
     {
      SMarginRate    Long;             // MarginRate of long positions
      SMarginRate    Short;            // MarginRate of short positions
      SMarginRate    BuyStop;          // MarginRate of BuyStop orders
      SMarginRate    BuyLimit;         // MarginRate of BuyLimit orders
      SMarginRate    BuyStopLimit;     // MarginRate of BuyStopLimit orders
      SMarginRate    SellStop;         // MarginRate of SellStop orders
      SMarginRate    SellLimit;        // MarginRate of SellLimit orders
      SMarginRate    SellStopLimit;    // MarginRate of SellStopLimit orders
     };
   SMarginRateMode   m_margin_rate;                                  // Margin ratio structure
```

**Let's complement the class private section** with the method
filling in all symbol properties for each margin ratio, the method
initializing the variables of structures storing all margin ratios. Also, let's add two auxiliary methods for
receiving the current day of week and the number of decimal places in a
'double' value:

```
   SMarginRateMode   m_margin_rate;                                  // Margin ratio structure
   MqlTick           m_tick;                                         // Symbol tick structure
   MqlBookInfo       m_book_info_array[];                            // Array of the market depth data structures
   string            m_symbol_name;                                  // Symbol name
   long              m_long_prop[SYMBOL_PROP_INTEGER_TOTAL];         // Integer properties
   double            m_double_prop[SYMBOL_PROP_DOUBLE_TOTAL];        // Real properties
   string            m_string_prop[SYMBOL_PROP_STRING_TOTAL];        // String properties
   int               m_digits_currency;                              // Number of decimal places in an account currency
   int               m_global_error;                                 // Global error code
//--- Return the index of the array the symbol's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_SYMBOL_PROP_DOUBLE property)  const { return(int)property-SYMBOL_PROP_INTEGER_TOTAL;                                    }
   int               IndexProp(ENUM_SYMBOL_PROP_STRING property)  const { return(int)property-SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_DOUBLE_TOTAL;           }
//--- (1) Fill in all the "margin ratio" symbol properties, (2) initialize the ratios
   bool              MarginRates(void);
   void              InitMarginRates(void);
//--- Reset all symbol object data
   void              Reset(void);
//--- Return the current day of the week
   ENUM_DAY_OF_WEEK  CurrentDayOfWeek(void)              const;
//--- Returns the number of decimal places in the 'double' value
   int               GetDigits(const double value)       const;
public:
```

In the same private section of the class, **declare the methods returning data on margin ratios for each order type:**

```
//--- Get and return real properties of a selected symbol from its parameters
   double            SymbolBidHigh(void)                 const;
   double            SymbolBidLow(void)                  const;
   double            SymbolVolumeReal(void)              const;
   double            SymbolVolumeHighReal(void)          const;
   double            SymbolVolumeLowReal(void)           const;
   double            SymbolOptionStrike(void)            const;
   double            SymbolTradeAccruedInterest(void)    const;
   double            SymbolTradeFaceValue(void)          const;
   double            SymbolTradeLiquidityRate(void)      const;
   double            SymbolMarginHedged(void)            const;
   bool              SymbolMarginLong(void);
   bool              SymbolMarginShort(void);
   bool              SymbolMarginBuyStop(void);
   bool              SymbolMarginBuyLimit(void);
   bool              SymbolMarginBuyStopLimit(void);
   bool              SymbolMarginSellStop(void);
   bool              SymbolMarginSellLimit(void);
   bool              SymbolMarginSellStopLimit(void);
//--- Get and return string properties of a selected symbol from its parameters
```

Sometimes, a program needs to know if a symbol exists on the server. This can be done by a symbol name. We already have the Exist()
method returning such data by class symbol. Overload the method so that it can return data by the passed symbol name. To do this, declare
yet another method call form in the private section of the class:

```
//--- Search for a symbol and return the flag indicating its presence on the server
   bool              Exist(void)                         const;
   bool              Exist(const string name)            const;
```

and **declare the overloaded method in the protected section of the class. The**
**method returns the symbol value by its name depending on the MQL5 or MQL4 program type:**

```
protected:
//--- Protected parametric constructor
                     CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name);

//--- Get and return integer properties of a selected symbol from its parameters
   bool              SymbolExists(const string name)     const;
   long              SymbolExists(void)                  const;
```

**Declare the virtual method for displaying a short symbol description**
in the section for the property description methods of the class' public section.

Let's implement this virtual method in the class descendants, in which the clarifying data on the symbol object is set.

```
//+------------------------------------------------------------------+
//| Description of symbol object properties                          |
//+------------------------------------------------------------------+
//--- Get description of a symbol (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_SYMBOL_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_SYMBOL_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_SYMBOL_PROP_STRING property);

//--- Send description of symbol properties to the journal (full_prop=true - all properties, false - only supported ones)
   void              Print(const bool full_prop=false);
//--- Display a short symbol description in the journal (implementation in the descendants)
   virtual void      PrintShort(void) {;}

//--- Compare CSymbol objects by all possible properties (for sorting lists by a specified symbol object property)
```

In the previous article, we added a few service methods when implementing a symbol object.

Let's add a few more
methods to return the start and end times of

quote and trading
sessions, as well as the private methods returning an integer number of hours,
minutes and seconds in a session and the method returning a description of
the session duration in the "HH:MM:SS" format :

```
//--- (1) Add, (2) remove a symbol from the Market Watch window, (3) return the data synchronization flag by a symbol
   bool              SetToMarketWatch(void)                       const { return ::SymbolSelect(this.m_symbol_name,true);                                   }
   bool              RemoveFromMarketWatch(void)                  const { return ::SymbolSelect(this.m_symbol_name,false);                                  }
   bool              IsSynchronized(void)                         const { return ::SymbolIsSynchronized(this.m_symbol_name);                                }
//--- Return the (1) start and (2) end time of the week day's quote session, (3) the start and end time of the required quote session
   long              SessionQuoteTimeFrom(const uint session_index,ENUM_DAY_OF_WEEK day_of_week=WRONG_VALUE)   const;
   long              SessionQuoteTimeTo(const uint session_index,ENUM_DAY_OF_WEEK day_of_week=WRONG_VALUE)     const;
   bool              GetSessionQuote(const uint session_index,ENUM_DAY_OF_WEEK day_of_week,datetime &from,datetime &to);
//--- Return the (1) start and (2) end time of the week day's trading session, (3) the start and end time of the required trading session
   long              SessionTradeTimeFrom(const uint session_index,ENUM_DAY_OF_WEEK day_of_week=WRONG_VALUE)   const;
   long              SessionTradeTimeTo(const uint session_index,ENUM_DAY_OF_WEEK day_of_week=WRONG_VALUE)     const;
   bool              GetSessionTrade(const uint session_index,ENUM_DAY_OF_WEEK day_of_week,datetime &from,datetime &to);
//--- (1) Arrange a (1) subscription to the market depth, (2) close the market depth, (3) fill in the market depth data to the structure array
   bool              BookAdd(void)                                const;
   bool              BookClose(void)                              const;
//--- Return (1) a session duration description in the hh:mm:ss format, number of (1) hours, (2) minutes and (3) seconds in the session duration time
   string            SessionDurationDescription(const ulong duration_sec) const;
private:
   int               SessionHours(const ulong duration_sec)       const;
   int               SessionMinutes(const ulong duration_sec)     const;
   int               SessionSeconds(const ulong duration_sec)     const;
public:
//+------------------------------------------------------------------+
```

In the public section's division of the methods for a simplified access to symbol object properties, **add**
**the second form of calling the method returning the flag of a symbol presence on the server** (previously, we declared the private
overloaded method looking for a symbol on the server by its name and returning the flag with the search result)

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
//+------------------------------------------------------------------+
//--- Integer properties
   long              Status(void)                                 const { return this.GetProperty(SYMBOL_PROP_STATUS);                                      }
   bool              IsCustom(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_CUSTOM);                                }
   color             ColorBackground(void)                        const { return (color)this.GetProperty(SYMBOL_PROP_BACKGROUND_COLOR);                     }
   ENUM_SYMBOL_CHART_MODE ChartMode(void)                         const { return (ENUM_SYMBOL_CHART_MODE)this.GetProperty(SYMBOL_PROP_CHART_MODE);          }
   bool              IsExist(void)                                const { return (bool)this.GetProperty(SYMBOL_PROP_EXIST);                                 }
   bool              IsExist(const string name)                   const { return this.SymbolExists(name);                                                   }
   bool              IsSelect(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_SELECT);                                }
   bool              IsVisible(void)                              const { return (bool)this.GetProperty(SYMBOL_PROP_VISIBLE);                               }
   long              SessionDeals(void)                           const { return this.GetProperty(SYMBOL_PROP_SESSION_DEALS);                               }
```

**In the public section's division of the methods for a simplified access to symbol real properties, add the methods returning all margin**
**ratios:**

```
//--- Real properties
   double            Bid(void)                                    const { return this.GetProperty(SYMBOL_PROP_BID);                                         }
   double            BidHigh(void)                                const { return this.GetProperty(SYMBOL_PROP_BIDHIGH);                                     }
   double            BidLow(void)                                 const { return this.GetProperty(SYMBOL_PROP_BIDLOW);                                      }
   double            Ask(void)                                    const { return this.GetProperty(SYMBOL_PROP_ASK);                                         }
   double            AskHigh(void)                                const { return this.GetProperty(SYMBOL_PROP_ASKHIGH);                                     }
   double            AskLow(void)                                 const { return this.GetProperty(SYMBOL_PROP_ASKLOW);                                      }
   double            Last(void)                                   const { return this.GetProperty(SYMBOL_PROP_LAST);                                        }
   double            LastHigh(void)                               const { return this.GetProperty(SYMBOL_PROP_LASTHIGH);                                    }
   double            LastLow(void)                                const { return this.GetProperty(SYMBOL_PROP_LASTLOW);                                     }
   double            VolumeReal(void)                             const { return this.GetProperty(SYMBOL_PROP_VOLUME_REAL);                                 }
   double            VolumeHighReal(void)                         const { return this.GetProperty(SYMBOL_PROP_VOLUMEHIGH_REAL);                             }
   double            VolumeLowReal(void)                          const { return this.GetProperty(SYMBOL_PROP_VOLUMELOW_REAL);                              }
   double            OptionStrike(void)                           const { return this.GetProperty(SYMBOL_PROP_OPTION_STRIKE);                               }
   double            Point(void)                                  const { return this.GetProperty(SYMBOL_PROP_POINT);                                       }
   double            TradeTickValue(void)                         const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE);                            }
   double            TradeTickValueProfit(void)                   const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT);                     }
   double            TradeTickValueLoss(void)                     const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS);                       }
   double            TradeTickSize(void)                          const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_SIZE);                             }
   double            TradeContractSize(void)                      const { return this.GetProperty(SYMBOL_PROP_TRADE_CONTRACT_SIZE);                         }
   double            TradeAccuredInterest(void)                   const { return this.GetProperty(SYMBOL_PROP_TRADE_ACCRUED_INTEREST);                      }
   double            TradeFaceValue(void)                         const { return this.GetProperty(SYMBOL_PROP_TRADE_FACE_VALUE);                            }
   double            TradeLiquidityRate(void)                     const { return this.GetProperty(SYMBOL_PROP_TRADE_LIQUIDITY_RATE);                        }
   double            LotsMin(void)                                const { return this.GetProperty(SYMBOL_PROP_VOLUME_MIN);                                  }
   double            LotsMax(void)                                const { return this.GetProperty(SYMBOL_PROP_VOLUME_MAX);                                  }
   double            LotsStep(void)                               const { return this.GetProperty(SYMBOL_PROP_VOLUME_STEP);                                 }
   double            VolumeLimit(void)                            const { return this.GetProperty(SYMBOL_PROP_VOLUME_LIMIT);                                }
   double            SwapLong(void)                               const { return this.GetProperty(SYMBOL_PROP_SWAP_LONG);                                   }
   double            SwapShort(void)                              const { return this.GetProperty(SYMBOL_PROP_SWAP_SHORT);                                  }
   double            MarginInitial(void)                          const { return this.GetProperty(SYMBOL_PROP_MARGIN_INITIAL);                              }
   double            MarginMaintenance(void)                      const { return this.GetProperty(SYMBOL_PROP_MARGIN_MAINTENANCE);                          }
   double            MarginLongInitial(void)                      const { return this.GetProperty(SYMBOL_PROP_MARGIN_LONG_INITIAL);                         }
   double            MarginBuyStopInitial(void)                   const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL);                     }
   double            MarginBuyLimitInitial(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL);                    }
   double            MarginBuyStopLimitInitial(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL);                }
   double            MarginLongMaintenance(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE);                     }
   double            MarginBuyStopMaintenance(void)               const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE);                 }
   double            MarginBuyLimitMaintenance(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE);                }
   double            MarginBuyStopLimitMaintenance(void)          const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE);            }
   double            MarginShortInitial(void)                     const { return this.GetProperty(SYMBOL_PROP_MARGIN_SHORT_INITIAL);                        }
   double            MarginSellStopInitial(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL);                    }
   double            MarginSellLimitInitial(void)                 const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL);                   }
   double            MarginSellStopLimitInitial(void)             const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL);               }
   double            MarginShortMaintenance(void)                 const { return this.GetProperty(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE);                    }
   double            MarginSellStopMaintenance(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE);                }
   double            MarginSellLimitMaintenance(void)             const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE);               }
   double            MarginSellStopLimitMaintenance(void)         const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE);           }
   double            SessionVolume(void)                          const { return this.GetProperty(SYMBOL_PROP_SESSION_VOLUME);                              }
   double            SessionTurnover(void)                        const { return this.GetProperty(SYMBOL_PROP_SESSION_TURNOVER);                            }
   double            SessionInterest(void)                        const { return this.GetProperty(SYMBOL_PROP_SESSION_INTEREST);                            }
   double            SessionBuyOrdersVolume(void)                 const { return this.GetProperty(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);                   }
   double            SessionSellOrdersVolume(void)                const { return this.GetProperty(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);                  }
   double            SessionOpen(void)                            const { return this.GetProperty(SYMBOL_PROP_SESSION_OPEN);                                }
   double            SessionClose(void)                           const { return this.GetProperty(SYMBOL_PROP_SESSION_CLOSE);                               }
   double            SessionAW(void)                              const { return this.GetProperty(SYMBOL_PROP_SESSION_AW);                                  }
   double            SessionPriceSettlement(void)                 const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT);                    }
   double            SessionPriceLimitMin(void)                   const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN);                     }
   double            SessionPriceLimitMax(void)                   const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX);                     }
   double            MarginHedged(void)                           const { return this.GetProperty(SYMBOL_PROP_MARGIN_HEDGED);                               }
   double            NormalizedPrice(const double price)          const;
//--- String properties
```

To obtain data on symbol properties, make sure the symbol is selected in the Market Watch window. There may be cases when a symbol is not
selected in the window but its properties are still needed. For such cases, we need to create the

flag indicating whether a
symbol was selected in the Market Watch window before we access its properties. Next, proceed as follows: if
a symbol is not selected, select it, get the properties and hide from
the Market Watch window. If a symbol is already selected, simply get its properties.

Also, we need to initialize margin ratio data and fill
them in for MQL5 in the class constructor. There is no such data for MQL4 and their values remain zero after initialization.


Also,

add calling the methods for saving these properties in the class property fields.

**To achieve this, add the necessary code in the class constructor:**

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CSymbol::CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name) : m_global_error(ERR_SUCCESS)
  {
   this.m_symbol_name=name;
   if(!this.Exist())
     {
      ::Print(DFUN_ERR_LINE,"\"",this.m_symbol_name,"\"",": ",TextByLanguage("Ошибка. Такого символа нет на сервере","Error. There is no such symbol on the server"));
      this.m_global_error=ERR_MARKET_UNKNOWN_SYMBOL;
     }
   bool select=::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SELECT);
   ::ResetLastError();
   if(!select)
     {
      if(!this.SetToMarketWatch())
        {
         this.m_global_error=::GetLastError();
         ::Print(DFUN_ERR_LINE,"\"",this.m_symbol_name,"\": ",TextByLanguage("Не удалось поместить в обзор рынка. Ошибка: ","Failed to put in the market watch. Error: "),this.m_global_error);
        }
     }
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_symbol_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,"\"",this.m_symbol_name,"\": ",TextByLanguage("Не удалось получить текущие цены. Ошибка: ","Could not get current prices. Error: "),this.m_global_error);
     }
//--- Initialize data
   ::ZeroMemory(this.m_tick);
   this.Reset();
   this.m_digits_currency=(#ifdef __MQL5__ (int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif);
   this.InitMarginRates();
   ::ResetLastError();
#ifdef __MQL5__
   if(!this.MarginRates())
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,this.Name(),": ",TextByLanguage("Не удалось получить коэффициенты взимания маржи. Ошибка: ","Failed to get margin rates. Error: "),this.m_global_error);
      return;
     }
#endif

//--- Save integer properties
   this.m_long_prop[SYMBOL_PROP_STATUS]                                       = symbol_status;
   this.m_long_prop[SYMBOL_PROP_VOLUME]                                       = (long)this.m_tick.volume;
   this.m_long_prop[SYMBOL_PROP_TIME]                                         = #ifdef __MQL5__ this.m_tick.time_msc #else this.m_tick.time*1000 #endif ;
   this.m_long_prop[SYMBOL_PROP_SELECT]                                       = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SELECT);
   this.m_long_prop[SYMBOL_PROP_VISIBLE]                                      = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_VISIBLE);
   this.m_long_prop[SYMBOL_PROP_SESSION_DEALS]                                = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SESSION_DEALS);
   this.m_long_prop[SYMBOL_PROP_SESSION_BUY_ORDERS]                           = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SESSION_BUY_ORDERS);
   this.m_long_prop[SYMBOL_PROP_SESSION_SELL_ORDERS]                          = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SESSION_SELL_ORDERS);
   this.m_long_prop[SYMBOL_PROP_VOLUMEHIGH]                                   = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_VOLUMEHIGH);
   this.m_long_prop[SYMBOL_PROP_VOLUMELOW]                                    = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_VOLUMELOW);
   this.m_long_prop[SYMBOL_PROP_DIGITS]                                       = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_DIGITS);
   this.m_long_prop[SYMBOL_PROP_SPREAD]                                       = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SPREAD);
   this.m_long_prop[SYMBOL_PROP_SPREAD_FLOAT]                                 = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SPREAD_FLOAT);
   this.m_long_prop[SYMBOL_PROP_TICKS_BOOKDEPTH]                              = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TICKS_BOOKDEPTH);
   this.m_long_prop[SYMBOL_PROP_TRADE_MODE]                                   = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_MODE);
   this.m_long_prop[SYMBOL_PROP_START_TIME]                                   = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_START_TIME);
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_TIME]                              = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_EXPIRATION_TIME);
   this.m_long_prop[SYMBOL_PROP_TRADE_STOPS_LEVEL]                            = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_STOPS_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_FREEZE_LEVEL]                           = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_FREEZE_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_EXEMODE]                                = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_EXEMODE);
   this.m_long_prop[SYMBOL_PROP_SWAP_ROLLOVER3DAYS]                           = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SWAP_ROLLOVER3DAYS);
   this.m_long_prop[SYMBOL_PROP_EXIST]                                        = this.SymbolExists();
   this.m_long_prop[SYMBOL_PROP_CUSTOM]                                       = this.SymbolCustom();
   this.m_long_prop[SYMBOL_PROP_MARGIN_HEDGED_USE_LEG]                        = this.SymbolMarginHedgedUseLEG();
   this.m_long_prop[SYMBOL_PROP_ORDER_MODE]                                   = this.SymbolOrderMode();
   this.m_long_prop[SYMBOL_PROP_FILLING_MODE]                                 = this.SymbolOrderFillingMode();
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_MODE]                              = this.SymbolExpirationMode();
   this.m_long_prop[SYMBOL_PROP_ORDER_GTC_MODE]                               = this.SymbolOrderGTCMode();
   this.m_long_prop[SYMBOL_PROP_OPTION_MODE]                                  = this.SymbolOptionMode();
   this.m_long_prop[SYMBOL_PROP_OPTION_RIGHT]                                 = this.SymbolOptionRight();
   this.m_long_prop[SYMBOL_PROP_BACKGROUND_COLOR]                             = this.SymbolBackgroundColor();
   this.m_long_prop[SYMBOL_PROP_CHART_MODE]                                   = this.SymbolChartMode();
   this.m_long_prop[SYMBOL_PROP_TRADE_CALC_MODE]                              = this.SymbolCalcMode();
   this.m_long_prop[SYMBOL_PROP_SWAP_MODE]                                    = this.SymbolSwapMode();

//--- Save real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKHIGH)]                    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_ASKHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKLOW)]                     = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_ASKLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTHIGH)]                   = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_LASTHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTLOW)]                    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_LASTLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_POINT)]                      = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_POINT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT)]    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE_PROFIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS)]      = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE_LOSS);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_SIZE)]            = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_CONTRACT_SIZE)]        = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_CONTRACT_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MIN)]                 = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MAX)]                 = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_STEP)]                = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_STEP);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_LIMIT)]               = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_LIMIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_LONG)]                  = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SWAP_LONG);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_SHORT)]                 = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SWAP_SHORT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_INITIAL)]             = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_MAINTENANCE)]         = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_VOLUME)]             = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_TURNOVER)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_TURNOVER);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_INTEREST)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_INTEREST);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME)]  = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_BUY_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME)] = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_SELL_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_OPEN)]               = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_OPEN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_CLOSE)]              = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_CLOSE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_AW)]                 = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_AW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT)]   = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_PRICE_SETTLEMENT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN)]    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_PRICE_LIMIT_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX)]    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_PRICE_LIMIT_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BID)]                        = this.m_tick.bid;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASK)]                        = this.m_tick.ask;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LAST)]                       = this.m_tick.last;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDHIGH)]                    = this.SymbolBidHigh();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDLOW)]                     = this.SymbolBidLow();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_REAL)]                = this.SymbolVolumeReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMEHIGH_REAL)]            = this.SymbolVolumeHighReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMELOW_REAL)]             = this.SymbolVolumeLowReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_OPTION_STRIKE)]              = this.SymbolOptionStrike();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_ACCRUED_INTEREST)]     = this.SymbolTradeAccruedInterest();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_FACE_VALUE)]           = this.SymbolTradeFaceValue();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_LIQUIDITY_RATE)]       = this.SymbolTradeLiquidityRate();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_HEDGED)]              = this.SymbolMarginHedged();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_INITIAL)]              = this.m_margin_rate.Long.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL)]          = this.m_margin_rate.BuyStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL)]         = this.m_margin_rate.BuyLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL)]     = this.m_margin_rate.BuyStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE)]          = this.m_margin_rate.Long.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE)]      = this.m_margin_rate.BuyStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE)]     = this.m_margin_rate.BuyLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE)] = this.m_margin_rate.BuyStopLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_INITIAL)]             = this.m_margin_rate.Short.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL)]         = this.m_margin_rate.SellStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL)]        = this.m_margin_rate.SellLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL)]    = this.m_margin_rate.SellStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE)]         = this.m_margin_rate.Short.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE)]     = this.m_margin_rate.SellStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE)]    = this.m_margin_rate.SellLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE)]= this.m_margin_rate.SellStopLimit.Maintenance;

//--- Save string properties
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_NAME)]                       = this.m_symbol_name;
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_BASE)]              = ::SymbolInfoString(this.m_symbol_name,SYMBOL_CURRENCY_BASE);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_PROFIT)]            = ::SymbolInfoString(this.m_symbol_name,SYMBOL_CURRENCY_PROFIT);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_MARGIN)]            = ::SymbolInfoString(this.m_symbol_name,SYMBOL_CURRENCY_MARGIN);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_DESCRIPTION)]                = ::SymbolInfoString(this.m_symbol_name,SYMBOL_DESCRIPTION);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_PATH)]                       = ::SymbolInfoString(this.m_symbol_name,SYMBOL_PATH);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_BASIS)]                      = this.SymbolBasis();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_BANK)]                       = this.SymbolBank();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_ISIN)]                       = this.SymbolISIN();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_FORMULA)]                    = this.SymbolFormula();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_PAGE)]                       = this.SymbolPage();
//--- Save additional integer properties
   this.m_long_prop[SYMBOL_PROP_DIGITS_LOTS]                                  = this.SymbolDigitsLot();
//---
   if(!select)
      this.RemoveFromMarketWatch();
  }
//+------------------------------------------------------------------+
```

Now we need to write implementations of all declared methods.

**Implement the method filling in all the variables storing the margin ratios** beyond the class body:

```
//+------------------------------------------------------------------+
//| Fill in the margin ratio variables                               |
//+------------------------------------------------------------------+
bool CSymbol::MarginRates(void)
  {
   bool res=true;
   #ifdef __MQL5__
      res &=this.SymbolMarginLong();
      res &=this.SymbolMarginBuyStop();
      res &=this.SymbolMarginBuyLimit();
      res &=this.SymbolMarginBuyStopLimit();
      res &=this.SymbolMarginShort();
      res &=this.SymbolMarginSellStop();
      res &=this.SymbolMarginSellLimit();
      res &=this.SymbolMarginSellStopLimit();
   #else
      this.InitMarginRates();
      res=false;
   #endif
   return res;
  }
//+------------------------------------------------------------------+
```

The MQL5 method simply calls the methods reading the ratio data from
the symbol properties and writing it to the appropriate structure variables. The result of returning all the methods is summed and returned
to the calling program. The methods are to be discussed below.

For MQL4, all structure fields are simply set to zero.

**The method of initializing the fields of the margin ratios property structures:**

```
//+------------------------------------------------------------------+
//| Initialize margin ratios                                         |
//+------------------------------------------------------------------+
void CSymbol::InitMarginRates(void)
  {
   this.m_margin_rate.Long.Initial=0;           this.m_margin_rate.Long.Maintenance=0;
   this.m_margin_rate.BuyStop.Initial=0;        this.m_margin_rate.BuyStop.Maintenance=0;
   this.m_margin_rate.BuyLimit.Initial=0;       this.m_margin_rate.BuyLimit.Maintenance=0;
   this.m_margin_rate.BuyStopLimit.Initial=0;   this.m_margin_rate.BuyStopLimit.Maintenance=0;
   this.m_margin_rate.Short.Initial=0;          this.m_margin_rate.Short.Maintenance=0;
   this.m_margin_rate.SellStop.Initial=0;       this.m_margin_rate.SellStop.Maintenance=0;
   this.m_margin_rate.SellLimit.Initial=0;      this.m_margin_rate.SellLimit.Maintenance=0;
   this.m_margin_rate.SellStopLimit.Initial=0;  this.m_margin_rate.SellStopLimit.Maintenance=0;
  }
//+------------------------------------------------------------------+
```

All fields of the **m\_margin\_rate** structure are simply reset here.

**Implementing the second form of calling the method returning the flag of the symbol presence on the server:**

```
//+------------------------------------------------------------------+
//| Return the symbol existence flag                                 |
//+------------------------------------------------------------------+
long CSymbol::SymbolExists(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_EXIST) #else this.Exist() #endif);
  }
//+------------------------------------------------------------------+
bool CSymbol::SymbolExists(const string name) const
  {
   return(#ifdef __MQL5__ (bool)::SymbolInfoInteger(name,SYMBOL_EXIST) #else this.Exist(name) #endif);
  }
//+------------------------------------------------------------------+
```

Here the [SYMBOL\_EXIST](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants)
symbol property is returned for MQL5, while the search for a symbol on the
server using the second form of calling the Exist(const string name) method is performed for MQL4.

**Implementing the methods filling in the margin ratios for all order types in the structure:**

```
//+------------------------------------------------------------------+
//| Fill in the margin ratios for long positions                     |
//+------------------------------------------------------------------+
bool CSymbol::SymbolMarginLong(void)
  {
   return(#ifdef __MQL5__ ::SymbolInfoMarginRate(this.m_symbol_name,ORDER_TYPE_BUY,this.m_margin_rate.Long.Initial,this.m_margin_rate.Long.Maintenance) #else false #endif);
  }
//+------------------------------------------------------------------+
//| Fill in the margin ratios for short positions                    |
//+------------------------------------------------------------------+
bool CSymbol::SymbolMarginShort(void)
  {
   return(#ifdef __MQL5__ ::SymbolInfoMarginRate(this.m_symbol_name,ORDER_TYPE_SELL,this.m_margin_rate.Short.Initial,this.m_margin_rate.Short.Maintenance) #else false #endif);
  }
//+------------------------------------------------------------------+
//| Fill in the margin ratios for BuyStop orders                     |
//+------------------------------------------------------------------+
bool CSymbol::SymbolMarginBuyStop(void)
  {
   return(#ifdef __MQL5__ ::SymbolInfoMarginRate(this.m_symbol_name,ORDER_TYPE_BUY_STOP,this.m_margin_rate.BuyStop.Initial,this.m_margin_rate.BuyStop.Maintenance) #else false #endif);
  }
//+------------------------------------------------------------------+
//| Fill in the margin ratios for BuyLimit orders                    |
//+------------------------------------------------------------------+
bool CSymbol::SymbolMarginBuyLimit(void)
  {
   return(#ifdef __MQL5__ ::SymbolInfoMarginRate(this.m_symbol_name,ORDER_TYPE_BUY_LIMIT,this.m_margin_rate.BuyLimit.Initial,this.m_margin_rate.BuyLimit.Maintenance) #else false #endif);
  }
//+------------------------------------------------------------------+
//| Fill in the margin ratios for BuyStopLimit orders                |
//+------------------------------------------------------------------+
bool CSymbol::SymbolMarginBuyStopLimit(void)
  {
   return(#ifdef __MQL5__ ::SymbolInfoMarginRate(this.m_symbol_name,ORDER_TYPE_BUY_STOP_LIMIT,this.m_margin_rate.BuyStopLimit.Initial,this.m_margin_rate.BuyStopLimit.Maintenance) #else false #endif);
  }
//+------------------------------------------------------------------+
//| Fill in the margin ratios for SellStop orders                    |
//+------------------------------------------------------------------+
bool CSymbol::SymbolMarginSellStop(void)
  {
   return(#ifdef __MQL5__ ::SymbolInfoMarginRate(this.m_symbol_name,ORDER_TYPE_SELL_STOP,this.m_margin_rate.SellStop.Initial,this.m_margin_rate.SellStop.Maintenance) #else false #endif);
  }
//+------------------------------------------------------------------+
//| Fill in the margin ratios for SellLimit orders                   |
//+------------------------------------------------------------------+
bool CSymbol::SymbolMarginSellLimit(void)
  {
   return(#ifdef __MQL5__ ::SymbolInfoMarginRate(this.m_symbol_name,ORDER_TYPE_SELL_LIMIT,this.m_margin_rate.SellLimit.Initial,this.m_margin_rate.SellLimit.Maintenance) #else false #endif);
  }
//+------------------------------------------------------------------+
//| Fill in the margin ratios for SellStopLimit orders               |
//+------------------------------------------------------------------+
bool CSymbol::SymbolMarginSellStopLimit(void)
  {
   return(#ifdef __MQL5__ ::SymbolInfoMarginRate(this.m_symbol_name,ORDER_TYPE_SELL_STOP_LIMIT,this.m_margin_rate.SellStopLimit.Initial,this.m_margin_rate.SellStopLimit.Maintenance) #else false #endif);
  }
//+------------------------------------------------------------------+
```

Here for MQL5, the [SymbolInfoMarginRate()](https://www.mql5.com/en/docs/marketinformation/symbolinfomarginrate)
function is called, in which the required properties stored in the **m\_margin\_rate** structure are filled in, and the
function operation result is returned.

For MQL4, return false.

Make the changes in the method returning the description of symbol properties in the block returning a description of a symbol background in
the Market Watch window:

```
      property==SYMBOL_PROP_BACKGROUND_COLOR    ?  TextByLanguage("Цвет фона символа в Market Watch","Background color of symbol in Market Watch")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         #ifdef __MQL5__
         (this.GetProperty(property)==CLR_DEFAULT || this.GetProperty(property)==CLR_NONE ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": "+::ColorToString((color)this.GetProperty(property),true))
         #else TextByLanguage(": Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
```

Previously, we compared the color with white (clrWhite), and if the color value exceeds the "white" color value, the color was considered not set. We
have already discussed the drawbacks of this comparison method. Therefore,

compare the color with the "default one" or with
a "missing color" to define the absence of a specified background color for a symbol in the Market Watch window.

**Add display of all margin ratio descriptions to the method returning descriptions of the**
**GetPropertyDescription(ENUM\_SYMBOL\_PROP\_DOUBLE property) real properties of a symbol:**

```
//--- Initial margin requirement of a Long position
      property==SYMBOL_PROP_MARGIN_LONG_INITIAL          ?  TextByLanguage("Коэффициент взимания начальной маржи по длинным позициям","Coefficient of margin initial charging for long positions")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
//--- Initial margin requirement of a Short position
      property==SYMBOL_PROP_MARGIN_SHORT_INITIAL     ?  TextByLanguage("Коэффициент взимания начальной маржи по коротким позициям","Coefficient of margin initial charging for short positions")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
//--- Maintenance margin requirement of a Long position
      property==SYMBOL_PROP_MARGIN_LONG_MAINTENANCE          ?  TextByLanguage("Коэффициент взимания поддерживающей маржи по длинным позициям","Coefficient of margin maintenance charging for long positions")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
//--- Maintenance margin requirement of a Short position
      property==SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE          ?  TextByLanguage("Коэффициент взимания поддерживающей маржи по коротким позициям","Coefficient of margin maintenance charging for short positions")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
//--- Initial margin requirements of Long orders
      property==SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL      ?  TextByLanguage("Коэффициент взимания начальной маржи по BuyStop ордерам","Coefficient of margin initial charging for BuyStop orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL     ?  TextByLanguage("Коэффициент взимания начальной маржи по BuyLimit ордерам","Coefficient of margin initial charging for BuyLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL ?  TextByLanguage("Коэффициент взимания начальной маржи по BuyStopLimit ордерам","Coefficient of margin initial charging for BuyStopLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
//--- Initial margin requirements of Short orders
      property==SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL      ?  TextByLanguage("Коэффициент взимания начальной маржи по SellStop ордерам","Coefficient of margin initial charging for SellStop orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL     ?  TextByLanguage("Коэффициент взимания начальной маржи по SellLimit ордерам","Coefficient of margin initial charging for SellLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL ?  TextByLanguage("Коэффициент взимания начальной маржи по SellStopLimit ордерам","Coefficient of margin initial charging for SellStopLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
//--- Maintenance margin requirements of Long orders
      property==SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE      ?  TextByLanguage("Коэффициент взимания поддерживающей маржи по BuyStop ордерам","Coefficient of margin maintenance charging for BuyStop orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE     ?  TextByLanguage("Коэффициент взимания поддерживающей маржи по BuyLimit ордерам","Coefficient of margin maintenance charging for BuyLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE ?  TextByLanguage("Коэффициент взимания поддерживающей маржи по BuyStopLimit ордерам","Coefficient of margin maintenance charging for BuyStopLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
//--- Maintenance margin requirements of Short orders
      property==SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE      ?  TextByLanguage("Коэффициент взимания поддерживающей маржи по SellStop ордерам","Coefficient of margin maintenance charging for SellStop orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE     ?  TextByLanguage("Коэффициент взимания поддерживающей маржи по SellLimit ордерам","Coefficient of margin maintenance charging for SellLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE ?  TextByLanguage("Коэффициент взимания поддерживающей маржи по SellStopLimit ордерам","Coefficient of margin maintenance charging for SellStopLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+
          #ifdef __MQL5__ (this.GetProperty(property)==0  ?  TextByLanguage(": (Не задан)",": (Not set)") : (::DoubleToString(this.GetProperty(property),8)))
          #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
   //---
```

**Implementing the second form of calling the method searching for a symbol by**
**its name on the server and returning the symbol presence flag:**

```
//+-------------------------------------------------------------------------------+
//| Search for a symbol and return the flag indicating its presence on the server |
//+-------------------------------------------------------------------------------+
bool CSymbol::Exist(void) const
  {
   int total=::SymbolsTotal(false);
   for(int i=0;i<total;i++)
      if(::SymbolName(i,false)==this.m_symbol_name)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
bool CSymbol::Exist(const string name) const
  {
   int total=::SymbolsTotal(false);
   for(int i=0;i<total;i++)
      if(::SymbolName(i,false)==name)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
```

**Implement the method calculating and returning the number of digits in a 'double' value:**

```
//+------------------------------------------------------------------+
//| Return the number of decimal places in the 'double' value        |
//+------------------------------------------------------------------+
int CSymbol::GetDigits(const double value) const
  {
   string val_str=(string)value;
   int len=::StringLen(val_str);
   int n=len-::StringFind(val_str,".",0)-1;
   if(::StringSubstr(val_str,len-1,1)=="0")
      n--;
   return n;
  }
//+------------------------------------------------------------------+
```

We discussed this method in the [previous article](https://www.mql5.com/en/articles/7014). Here it is simply
implemented in a separate method since a repeated calculation for several values is required — for a minimum lot and a lot step.

**Implementing the methods returning the start time of a quote session from the beginning of a day, the end time of a quote session from the beginning of a**
**day and the quote session start and end time:**

```
//+------------------------------------------------------------------+
//| Return the quote session start time                              |
//| in seconds from the beginning of a day                           |
//+------------------------------------------------------------------+
long CSymbol::SessionQuoteTimeFrom(const uint session_index,ENUM_DAY_OF_WEEK day_of_week=WRONG_VALUE) const
  {
   MqlDateTime time={0};
   datetime from=0,to=0;
   ENUM_DAY_OF_WEEK day=(day_of_week<0 || day_of_week>SATURDAY ? this.CurrentDayOfWeek() : day_of_week);
   return(::SymbolInfoSessionQuote(this.m_symbol_name,day,session_index,from,to) ? from : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return the time in seconds since the day start                   |
//| up to the end of a quote session                                 |
//+------------------------------------------------------------------+
long CSymbol::SessionQuoteTimeTo(const uint session_index,ENUM_DAY_OF_WEEK day_of_week=WRONG_VALUE) const
  {
   MqlDateTime time={0};
   datetime from=0,to=0;
   ENUM_DAY_OF_WEEK day=(day_of_week<0 || day_of_week>SATURDAY ? this.CurrentDayOfWeek() : day_of_week);
   return(::SymbolInfoSessionQuote(this.m_symbol_name,day,session_index,from,to) ? to : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return the start and end time of a required quote session        |
//+------------------------------------------------------------------+
bool CSymbol::GetSessionQuote(const uint session_index,ENUM_DAY_OF_WEEK day_of_week,datetime &from,datetime &to)
  {
   ENUM_DAY_OF_WEEK day=(day_of_week<0 || day_of_week>SATURDAY ? this.CurrentDayOfWeek() : day_of_week);
   return ::SymbolInfoSessionQuote(this.m_symbol_name,day,session_index,from,to);
  }
//+------------------------------------------------------------------+
```

The session index and week
day are passed to the very first two methods, while the third one additionally receives 'datetime'
type variables that are to store data on the beginning and end of the required session received using the [SymbolInfoSessionQuote()](https://www.mql5.com/en/docs/marketinformation/symbolinfosessionquote)
function.

For more convenience, if -1 is selected as a week day, the session data is taken for the current week day. The session index should start from
zero. The time is returned as a number of seconds from the beginning of the day defined by the

**day\_of\_week** parameter. Thus, you can always find out the actual requested time by adding the number of seconds received from the
method to the time of the beginning of a day.

**The methods of receiving trade session times are implemented in the same way:**

```
//+------------------------------------------------------------------+
//| Return the trading session start time                            |
//| in seconds from the beginning of a day                           |
//+------------------------------------------------------------------+
long CSymbol::SessionTradeTimeFrom(const uint session_index,ENUM_DAY_OF_WEEK day_of_week=WRONG_VALUE) const
  {
   MqlDateTime time={0};
   datetime from=0,to=0;
   ENUM_DAY_OF_WEEK day=(day_of_week<0 || day_of_week>SATURDAY ? this.CurrentDayOfWeek() : day_of_week);
   return(::SymbolInfoSessionTrade(this.m_symbol_name,day,session_index,from,to) ? from : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return the time in seconds since the day start                   |
//| up to the end of a trading session                               |
//+------------------------------------------------------------------+
long CSymbol::SessionTradeTimeTo(const uint session_index,ENUM_DAY_OF_WEEK day_of_week=WRONG_VALUE) const
  {
   MqlDateTime time={0};
   datetime from=0,to=0;
   ENUM_DAY_OF_WEEK day=(day_of_week<0 || day_of_week>SATURDAY ? this.CurrentDayOfWeek() : day_of_week);
   return(::SymbolInfoSessionTrade(this.m_symbol_name,day,session_index,from,to) ? to : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return the start and end time of a required trading session      |
//+------------------------------------------------------------------+
bool CSymbol::GetSessionTrade(const uint session_index,ENUM_DAY_OF_WEEK day_of_week,datetime &from,datetime &to)
  {
   ENUM_DAY_OF_WEEK day=(day_of_week<0 || day_of_week>SATURDAY ? this.CurrentDayOfWeek() : day_of_week);
   return ::SymbolInfoSessionTrade(this.m_symbol_name,day,session_index,from,to);
  }
//+------------------------------------------------------------------+
```

These methods are similar to the ones described above except that the [SymbolInfoSessionTrade()](https://www.mql5.com/en/docs/marketinformation/symbolinfosessiontrade)
function is used here to receive the necessary data.

**Implementing the method returning the current week day as the [ENUM\_DAY\_OF\_WEEK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_day_of_week)** **enumeration value:**

```
//+------------------------------------------------------------------+
//| Return the current day of the week                               |
//+------------------------------------------------------------------+
ENUM_DAY_OF_WEEK CSymbol::CurrentDayOfWeek(void) const
  {
   MqlDateTime time={0};
   ::TimeCurrent(time);
   return(ENUM_DAY_OF_WEEK)time.day_of_week;
  }
//+------------------------------------------------------------------+
```

Here all is simple: declare the date and time structure,
access the

[TimeCurrent()](https://www.mql5.com/en/docs/dateandtime/timecurrent) function whose second call form fills in the date
and time structure passed to the function and finally return

a week day from the filled
structure.

**Implementing the method returning the number of seconds in duration time of a specified session:**

```
//+------------------------------------------------------------------+
//| Return the number of seconds in a session duration time          |
//+------------------------------------------------------------------+
int CSymbol::SessionSeconds(const ulong duration_sec) const
  {
   return int(duration_sec % 60);
  }
//+------------------------------------------------------------------+
```

The method receives the number of seconds and returns
the residue of dividing by the number of minutes in this time span.

**Implementing the method returning the number of minutes** **in duration time of a specified session:**

```
//+------------------------------------------------------------------+
//| Return the number of minutes in a session duration time          |
//+------------------------------------------------------------------+
int CSymbol::SessionMinutes(const ulong duration_sec) const
  {
   return int((duration_sec-this.SessionSeconds(duration_sec)) % 3600)/60;
  }
//+------------------------------------------------------------------+
```

The method receives the number of seconds and returns the
calculated number of minutes in the time span

except the number of seconds not multiple of one minute.

**Implementing the method returning the number of hours** **in duration time of a specified session:**

```
//+------------------------------------------------------------------+
//| Return the number of hours in a session duration time            |
//+------------------------------------------------------------------+
int CSymbol::SessionHours(const ulong duration_sec) const
  {
   return int(duration_sec-this.SessionSeconds(duration_sec)-this.SessionMinutes(duration_sec))/3600;
  }
//+------------------------------------------------------------------+
```

The method receives the number of seconds and returns the
number of hours in the time span

except the number of seconds not multiple of one minute and the
number of minutes not multiple of one hour.

**Implementing the method returning the description of a session duration in the "HH:MM:SS" format:**

```
//+---------------------------------------------------------------------+
//| Return the description of a session duration in the hh:mm:ss format |
//+---------------------------------------------------------------------+
string CSymbol::SessionDurationDescription(const ulong duration_sec) const
  {
   int sec=this.SessionSeconds(duration_sec);
   int min=this.SessionMinutes(duration_sec);
   int hour=this.SessionHours(duration_sec);
   return ::IntegerToString(hour,2,'0')+":"+::IntegerToString(min,2,'0')+":"+::IntegerToString(sec,2,'0');
  }
//+------------------------------------------------------------------+
```

Here we simply get a session duration in seconds, as
well as a calculated session duration in seconds, minutes and hours,
and display a formatted message in the Hours:Minutes:Seconds format using the

[IntegerToString()](https://www.mql5.com/en/docs/convert/integertostring) function with
the string size for hours, minutes and seconds equal to two digits,
and the

"0" filler in case there is a single digit in the hours, minutes or seconds
values.

For example, if we received 2 hours, it is shown as 02.

Since we slightly reworked the states of symbol objects, correct the method displaying a description of a symbol object status:

```
//+------------------------------------------------------------------+
//| Return the status description                                    |
//+------------------------------------------------------------------+
string CSymbol::GetStatusDescription() const
  {
   return
     (
      this.Status()==SYMBOL_STATUS_FX           ? TextByLanguage("Форекс символ","Forex symbol")                  :
      this.Status()==SYMBOL_STATUS_FX_MAJOR     ? TextByLanguage("Форекс символ-мажор","Forex major symbol")      :
      this.Status()==SYMBOL_STATUS_FX_MINOR     ? TextByLanguage("Форекс символ-минор","Forex minor symbol")      :
      this.Status()==SYMBOL_STATUS_FX_EXOTIC    ? TextByLanguage("Форекс символ-экзотик","Forex Exotic Symbol")   :
      this.Status()==SYMBOL_STATUS_FX_RUB       ? TextByLanguage("Форекс символ/рубль","Forex symbol RUB")        :
      this.Status()==SYMBOL_STATUS_METAL        ? TextByLanguage("Металл","Metal")                                :
      this.Status()==SYMBOL_STATUS_INDEX        ? TextByLanguage("Индекс","Index")                                :
      this.Status()==SYMBOL_STATUS_INDICATIVE   ? TextByLanguage("Индикатив","Indicative")                        :
      this.Status()==SYMBOL_STATUS_CRYPTO       ? TextByLanguage("Криптовалютный символ","Crypto symbol")         :
      this.Status()==SYMBOL_STATUS_COMMODITY    ? TextByLanguage("Товарный символ","Commodity symbol")            :
      this.Status()==SYMBOL_STATUS_EXCHANGE     ? TextByLanguage("Биржевой символ","Exchange symbol")             :
      this.Status()==SYMBOL_STATUS_FUTURES      ? TextByLanguage("Фьючерс","Futures")                             :
      this.Status()==SYMBOL_STATUS_CFD          ? TextByLanguage("Контракт на разницу","Contract For Difference") :
      this.Status()==SYMBOL_STATUS_STOCKS       ? TextByLanguage("Ценная бумага","Stocks")                        :
      this.Status()==SYMBOL_STATUS_BONDS        ? TextByLanguage("Облигация","Bonds")                             :
      this.Status()==SYMBOL_STATUS_OPTION       ? TextByLanguage("Опцион","Option")                               :
      this.Status()==SYMBOL_STATUS_COLLATERAL   ? TextByLanguage("Неторгуемый актив","Collateral")                :
      this.Status()==SYMBOL_STATUS_CUSTOM       ? TextByLanguage("Пользовательский символ","Custom symbol")       :
      this.Status()==SYMBOL_STATUS_COMMON       ? TextByLanguage("Символ общей группы","Common group symbol")     :
      ::EnumToString((ENUM_SYMBOL_STATUS)this.Status())
     );
  }
//+------------------------------------------------------------------+
```

**In the method of updating all symbol data, add receiving margin**
**ratios for MQL5 for all order and position types.** In case of MQL4, their values remain zero after the starting
initialization in the class constructor since they are not used in MQL4:

```
//+------------------------------------------------------------------+
//| Update all symbol data that can change                           |
//+------------------------------------------------------------------+
void CSymbol::Refresh(void)
  {
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_symbol_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();

      return;
     }
#ifdef __MQL5__
   ::ResetLastError();
   if(!this.MarginRates())
     {
      this.m_global_error=::GetLastError();
      return;
     }
#endif
//--- Update integer properties
   this.m_long_prop[SYMBOL_PROP_VOLUME]                                       = (long)this.m_tick.volume;
   this.m_long_prop[SYMBOL_PROP_TIME]                                         = #ifdef __MQL5__ this.m_tick.time_msc #else this.m_tick.time*1000 #endif ;
   this.m_long_prop[SYMBOL_PROP_SELECT]                                       = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SELECT);
   this.m_long_prop[SYMBOL_PROP_VISIBLE]                                      = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_VISIBLE);
   this.m_long_prop[SYMBOL_PROP_SESSION_DEALS]                                = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SESSION_DEALS);
   this.m_long_prop[SYMBOL_PROP_SESSION_BUY_ORDERS]                           = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SESSION_BUY_ORDERS);
   this.m_long_prop[SYMBOL_PROP_SESSION_SELL_ORDERS]                          = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SESSION_SELL_ORDERS);
   this.m_long_prop[SYMBOL_PROP_VOLUMEHIGH]                                   = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_VOLUMEHIGH);
   this.m_long_prop[SYMBOL_PROP_VOLUMELOW]                                    = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_VOLUMELOW);
   this.m_long_prop[SYMBOL_PROP_SPREAD]                                       = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SPREAD);
   this.m_long_prop[SYMBOL_PROP_TICKS_BOOKDEPTH]                              = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TICKS_BOOKDEPTH);
   this.m_long_prop[SYMBOL_PROP_START_TIME]                                   = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_START_TIME);
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_TIME]                              = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_EXPIRATION_TIME);
   this.m_long_prop[SYMBOL_PROP_TRADE_STOPS_LEVEL]                            = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_STOPS_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_FREEZE_LEVEL]                           = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_FREEZE_LEVEL);
   this.m_long_prop[SYMBOL_PROP_BACKGROUND_COLOR]                             = this.SymbolBackgroundColor();
//--- Update real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKHIGH)]                    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_ASKHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKLOW)]                     = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_ASKLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTHIGH)]                   = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_LASTHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTLOW)]                    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_LASTLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT)]    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE_PROFIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS)]      = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE_LOSS);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_SIZE)]            = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_CONTRACT_SIZE)]        = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_CONTRACT_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MIN)]                 = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MAX)]                 = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_STEP)]                = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_STEP);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_LIMIT)]               = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_LIMIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_LONG)]                  = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SWAP_LONG);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_SHORT)]                 = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SWAP_SHORT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_INITIAL)]             = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_MAINTENANCE)]         = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_VOLUME)]             = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_TURNOVER)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_TURNOVER);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_INTEREST)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_INTEREST);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME)]  = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_BUY_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME)] = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_SELL_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_OPEN)]               = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_OPEN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_CLOSE)]              = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_CLOSE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_AW)]                 = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_AW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT)]   = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_PRICE_SETTLEMENT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN)]    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_PRICE_LIMIT_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX)]    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_SESSION_PRICE_LIMIT_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASK)]                        = this.m_tick.ask;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BID)]                        = this.m_tick.bid;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LAST)]                       = this.m_tick.last;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDHIGH)]                    = this.SymbolBidHigh();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDLOW)]                     = this.SymbolBidLow();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_REAL)]                = this.SymbolVolumeReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMEHIGH_REAL)]            = this.SymbolVolumeHighReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMELOW_REAL)]             = this.SymbolVolumeLowReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_OPTION_STRIKE)]              = this.SymbolOptionStrike();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_ACCRUED_INTEREST)]     = this.SymbolTradeAccruedInterest();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_FACE_VALUE)]           = this.SymbolTradeFaceValue();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_LIQUIDITY_RATE)]       = this.SymbolTradeLiquidityRate();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_HEDGED)]              = this.SymbolMarginHedged();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_INITIAL)]              = this.m_margin_rate.Long.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL)]          = this.m_margin_rate.BuyStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL)]         = this.m_margin_rate.BuyLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL)]     = this.m_margin_rate.BuyStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE)]          = this.m_margin_rate.Long.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE)]      = this.m_margin_rate.BuyStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE)]     = this.m_margin_rate.BuyLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE)] = this.m_margin_rate.BuyStopLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_INITIAL)]             = this.m_margin_rate.Short.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL)]         = this.m_margin_rate.SellStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL)]        = this.m_margin_rate.SellLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL)]    = this.m_margin_rate.SellStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE)]         = this.m_margin_rate.Short.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE)]     = this.m_margin_rate.SellStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE)]    = this.m_margin_rate.SellLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE)]= this.m_margin_rate.SellStopLimit.Maintenance;
  }
//+------------------------------------------------------------------+
```

Here for MQL5, we call the method of receiving data on the
MarginRates() margin ratios. If at least one of the ratios is not received (the method returned

false), simply write the error
code to the variable storing the class error code and exit the method without a message.

An error message is not
displayed in the journal since the method works in the timer. In case of an erroneous data receipt, the journal is quickly filled with garbage
messages about the same error. Since this error code can always be received in the CEngine class, let's leave it responsible for receiving
and handling it.

At the very end of the method, all obtained ratio data is written in the
fields of the appropriate symbol object properties.

For the same reason, **remove the string displaying an error message in the**
**journal from the quote data update method:**

```
//+------------------------------------------------------------------+
//| Update quote data by a symbol                                    |
//+------------------------------------------------------------------+
void CSymbol::RefreshRates(void)
  {
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_symbol_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,this.Name(),": ",TextByLanguage("Не удалось получить текущие цены. Ошибка: ","Could not get current prices. Error: "),this.m_global_error);
      return;
     }
//--- Update integer properties
   this.m_long_prop[SYMBOL_PROP_VOLUME]                                       = (long)this.m_tick.volume;
   this.m_long_prop[SYMBOL_PROP_TIME]                                         = #ifdef __MQL5__ this.m_tick.time_msc #else this.m_tick.time*1000 #endif ;
   this.m_long_prop[SYMBOL_PROP_SPREAD]                                       = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SPREAD);
   this.m_long_prop[SYMBOL_PROP_TRADE_STOPS_LEVEL]                            = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_STOPS_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_FREEZE_LEVEL]                           = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_FREEZE_LEVEL);
//--- Update real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKHIGH)]                    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_ASKHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKLOW)]                     = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_ASKLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTHIGH)]                   = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_LASTHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTLOW)]                    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_LASTLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT)]    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE_PROFIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS)]      = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE_LOSS);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASK)]                        = this.m_tick.ask;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BID)]                        = this.m_tick.bid;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LAST)]                       = this.m_tick.last;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDHIGH)]                    = this.SymbolBidHigh();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDLOW)]                     = this.SymbolBidLow();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_REAL)]                = this.SymbolVolumeReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_OPTION_STRIKE)]              = this.SymbolOptionStrike();
  }
//+------------------------------------------------------------------+
```

Now the method will look like as follows:

```
//+------------------------------------------------------------------+
//| Update quote data by a symbol                                    |
//+------------------------------------------------------------------+
void CSymbol::RefreshRates(void)
  {
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_symbol_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();
      return;
     }
//--- Update integer properties
   this.m_long_prop[SYMBOL_PROP_VOLUME]                                       = (long)this.m_tick.volume;
   this.m_long_prop[SYMBOL_PROP_TIME]                                         = #ifdef __MQL5__ this.m_tick.time_msc #else this.m_tick.time*1000 #endif ;
   this.m_long_prop[SYMBOL_PROP_SPREAD]                                       = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SPREAD);
   this.m_long_prop[SYMBOL_PROP_TRADE_STOPS_LEVEL]                            = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_STOPS_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_FREEZE_LEVEL]                           = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_FREEZE_LEVEL);
//--- Update real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKHIGH)]                    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_ASKHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKLOW)]                     = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_ASKLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTHIGH)]                   = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_LASTHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTLOW)]                    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_LASTLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT)]    = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE_PROFIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS)]      = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_TICK_VALUE_LOSS);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASK)]                        = this.m_tick.ask;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BID)]                        = this.m_tick.bid;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LAST)]                       = this.m_tick.last;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDHIGH)]                    = this.SymbolBidHigh();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDLOW)]                     = this.SymbolBidLow();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_REAL)]                = this.SymbolVolumeReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_OPTION_STRIKE)]              = this.SymbolOptionStrike();
  }
//+------------------------------------------------------------------+
```

**This completes the improvement of the abstract CSymbol symbol class.**

We have examined the important and most significant changes to the class methods, with the exception of minor corrections that have
been made but not described here because these are just some spelling and “semantic” errors basically in the property description
methods. You can find them in the attached files.

Now we need to create descendant objects of the abstract symbol base class, divide them by categories and place them to the symbol object
collection.

### Descendant objects of the "symbol" basic abstract object

Let's go back a bit and look at the mentioned symbol categories. Besides, we are going to set the names of the corresponding descendant classes of
the CSymbol base class:

- Forex symbol — CSymbolFX class

- Major Forex symbol — CSymbolFXMajor class
- Minor Forex symbol — CSymbolFXMinor class
- Exotic Forex symbol — CSymbolFXExotic class
- Forex symbol/RUB — CSymbolFXRub class
- Metal — CSymbolMetall class
- Index — CSymbolIndex class
- Indicative — CSymbolIndicative class
- Cryptocurrency symbol — CSymbolCrypto class
- Commodity symbol — CSymbolCommodity class
- Exchange symbol — CSymbolExchange class
- Futures — CSymbolFutures class
- CFD — CSymbolCFD class
- Stock — CSymbolStocks class
- Bond — CSymbolBonds class
- Option — CSymbolOption class
- Non-tradable asset — CSymbolCollateral class
- Custom symbol — CSymbolCustom class
- General category — CSymbolCommon class

In total, we have nineteen derived classes. Let's have a look at creating a class using the Forex symbol class as an example.

In the **\\MQL5\\Include\\DoEasy\\Objects\\Symbols\** library folder, create the new class **CSymbolFX** with the file
named

**SymbolFX.mqh**. The CSymbol abstract symbol class is to be used
as a base class for it.

**Declare all the methods necessary for the class operation:**

```
//+------------------------------------------------------------------+
//|                                                     SymbolFX.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Symbol.mqh"
//+------------------------------------------------------------------+
//| Forex symbol                                                     |
//+------------------------------------------------------------------+
class CSymbolFX : public CSymbol
  {
public:
//--- Constructor
                     CSymbolFX(const string name) : CSymbol(SYMBOL_STATUS_FX,name) {}
//--- Supported integer properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_INTEGER property);
//--- Supported real properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_DOUBLE property);
//--- Supported string properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_STRING property);
//--- Display a short symbol description in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
```

The class constructor is to receive a symbol
name, while the initialization list of the class constructor is used to send
a "Forex symbol" category (its status) to the basic class together with a symbol
name passed to the CSymbolFX class constructor when it is created.

The virtual methods of supporting integer, real
and string properties by an object were declared in the base class,
while their implementation is performed in the descendant classes. The virtual

PrintShort() method, which is also declared in the base class and
implemented in the descendant class, displays brief symbol data in the journal.

Almost all descendant class methods are similar and can be implemented in the base class without the need for descendant classes. However,
in this case, we lose the flexibility of making possible changes of these methods for each symbol group. Therefore, I decided to make a
division by categories via descendant classes to be able to change each descendant class separately, which is much simpler and faster.

**Implementing the method returning the flag of supporting an integer property by a symbol object:**

```
//+------------------------------------------------------------------+
//| Return 'true' if a symbol supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CSymbolFX::SupportProperty(ENUM_SYMBOL_PROP_INTEGER property)
  {
   if(property==SYMBOL_PROP_EXIST
   #ifdef __MQL4__                                 ||
      property==SYMBOL_PROP_CUSTOM                 ||
      property==SYMBOL_PROP_SESSION_DEALS          ||
      property==SYMBOL_PROP_SESSION_BUY_ORDERS     ||
      property==SYMBOL_PROP_SESSION_SELL_ORDERS    ||
      property==SYMBOL_PROP_VOLUME                 ||
      property==SYMBOL_PROP_VOLUMEHIGH             ||
      property==SYMBOL_PROP_VOLUMELOW              ||
      property==SYMBOL_PROP_TICKS_BOOKDEPTH        ||
      property==SYMBOL_PROP_OPTION_MODE            ||
      property==SYMBOL_PROP_OPTION_RIGHT           ||
      property==SYMBOL_PROP_BACKGROUND_COLOR
   #endif
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

A checked integer property is passed to the method. If
the passed property is "Symbol existence", return false — if a symbol is
created, it exists, and we do not need that property neither to be displayed in the journal, nor for searching and sorting. All other checks
are applied

only to MQL4. False
is returned if a knowingly unsupported symbol property is passed to the method in MQL4.

If a passed property was not present among enumerated ones when checking the properties, it is not supported. Return true.

**Implementing the method returning the flag of supporting a real property by a symbol object:**

```
//+------------------------------------------------------------------+
//| Return 'true' if a symbol supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CSymbolFX::SupportProperty(ENUM_SYMBOL_PROP_DOUBLE property)
  {
   if(
     #ifdef __MQL5__
      (this.ChartMode()==SYMBOL_CHART_MODE_BID     &&
        (
         property==SYMBOL_PROP_LAST                ||
         property==SYMBOL_PROP_LASTHIGH            ||
         property==SYMBOL_PROP_LASTLOW
        )
      )                                            ||
      (this.ChartMode()==SYMBOL_CHART_MODE_LAST    &&
        (
         property==SYMBOL_PROP_BID                 ||
         property==SYMBOL_PROP_BIDHIGH             ||
         property==SYMBOL_PROP_BIDLOW              ||
         property==SYMBOL_PROP_ASK                 ||
         property==SYMBOL_PROP_ASKHIGH             ||
         property==SYMBOL_PROP_ASKLOW
        )
      )
     //--- __MQL4__
     #else
      property==SYMBOL_PROP_ASKHIGH                            ||
      property==SYMBOL_PROP_ASKLOW                             ||
      property==SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT            ||
      property==SYMBOL_PROP_TRADE_TICK_VALUE_LOSS              ||
      property==SYMBOL_PROP_LAST                               ||
      property==SYMBOL_PROP_LASTHIGH                           ||
      property==SYMBOL_PROP_LASTLOW                            ||
      property==SYMBOL_PROP_VOLUME_LIMIT                       ||
      property==SYMBOL_PROP_MARGIN_LONG_INITIAL                ||
      property==SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL            ||
      property==SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL           ||
      property==SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL       ||
      property==SYMBOL_PROP_MARGIN_LONG_MAINTENANCE            ||
      property==SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE        ||
      property==SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE       ||
      property==SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE   ||
      property==SYMBOL_PROP_MARGIN_SHORT_INITIAL               ||
      property==SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL           ||
      property==SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL          ||
      property==SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL      ||
      property==SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE           ||
      property==SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE       ||
      property==SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE      ||
      property==SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE  ||
      property==SYMBOL_PROP_SESSION_VOLUME                     ||
      property==SYMBOL_PROP_SESSION_TURNOVER                   ||
      property==SYMBOL_PROP_SESSION_INTEREST                   ||
      property==SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME          ||
      property==SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME         ||
      property==SYMBOL_PROP_SESSION_OPEN                       ||
      property==SYMBOL_PROP_SESSION_CLOSE                      ||
      property==SYMBOL_PROP_SESSION_AW                         ||
      property==SYMBOL_PROP_SESSION_PRICE_SETTLEMENT           ||
      property==SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN            ||
      property==SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX            ||
      property==SYMBOL_PROP_VOLUME_REAL                        ||
      property==SYMBOL_PROP_VOLUMEHIGH_REAL                    ||
      property==SYMBOL_PROP_VOLUMELOW_REAL                     ||
      property==SYMBOL_PROP_OPTION_STRIKE                      ||
      property==SYMBOL_PROP_TRADE_ACCRUED_INTEREST             ||
      property==SYMBOL_PROP_TRADE_FACE_VALUE                   ||
      property==SYMBOL_PROP_TRADE_LIQUIDITY_RATE
     #endif
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

The logic here is the same as in the previous method. But first, check the obtained property for MQL5.
If it is one of the last deal price properties (Last), while

the chart is based on Bid prices, all these properties are equal to
zero and are not supported in this case.

The same is done with Bid price properties in case the chart is based
on Last prices — all Bid price properties are not supported.

In
case of MQL4, do exactly as in the previous method — when passing a knowingly unsupported symbol property to the method, return false.

**Implementing the method returning the flag of supporting a string property by a symbol object:**

```
//+------------------------------------------------------------------+
//| Return 'true' if a symbol supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CSymbolFX::SupportProperty(ENUM_SYMBOL_PROP_STRING property)
  {
   if(
   #ifdef __MQL5__
      property==SYMBOL_PROP_FORMULA && !this.IsCustom()
   #else
      property==SYMBOL_PROP_BASIS                  ||
      property==SYMBOL_PROP_BANK                   ||
      property==SYMBOL_PROP_ISIN                   ||
      property==SYMBOL_PROP_FORMULA                ||
      property==SYMBOL_PROP_PAGE
   #endif
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

Here all is similar: in case of MQL5 — if
the passed property is "Equation for custom symbol calculation", while
a symbol is not custom, return false — the property is not supported. Next, we
check the knowingly unsupported symbol properties

for MQL4 and return false
if a property not supported in MQL4 is passed.

**Method of displaying a brief symbol description to the journal:**

```
//+------------------------------------------------------------------+
//| Display a short symbol description in the journal                |
//+------------------------------------------------------------------+
void CSymbolFX::PrintShort(void)
  {
   ::Print(this.GetStatusDescription()+" "+this.Name());
  }
//+------------------------------------------------------------------+
```

The method simply displays a line consisting of a symbol status string
description and its name.

The remaining descendant classes are constructed in the same way and have the same methods with the same implementation.

The difference
is in the

**custom symbol class** — such symbol type is not present in MQL4, therefore all the checks are applied only to MQL5:

```
//+------------------------------------------------------------------+
//|                                                 SymbolCustom.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Symbol.mqh"
//+------------------------------------------------------------------+
//| Custom symbol                                                    |
//+------------------------------------------------------------------+
class CSymbolCustom : public CSymbol
  {
public:
//--- Constructor
                     CSymbolCustom(const string name) : CSymbol(SYMBOL_STATUS_CUSTOM,name) {}
//--- Supported integer properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_INTEGER property);
//--- Supported real properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_DOUBLE property);
//--- Supported string properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_STRING property);
//--- Display a short symbol description in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if a symbol supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CSymbolCustom::SupportProperty(ENUM_SYMBOL_PROP_INTEGER property)
  {
   if(property==SYMBOL_PROP_EXIST) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if a symbol supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CSymbolCustom::SupportProperty(ENUM_SYMBOL_PROP_DOUBLE property)
  {
   if(
      (this.ChartMode()==SYMBOL_CHART_MODE_BID     &&
        (
         property==SYMBOL_PROP_LAST                ||
         property==SYMBOL_PROP_LASTHIGH            ||
         property==SYMBOL_PROP_LASTLOW
        )
      )                                            ||
      (this.ChartMode()==SYMBOL_CHART_MODE_LAST    &&
        (
         property==SYMBOL_PROP_BID                 ||
         property==SYMBOL_PROP_BIDHIGH             ||
         property==SYMBOL_PROP_BIDLOW              ||
         property==SYMBOL_PROP_ASK                 ||
         property==SYMBOL_PROP_ASKHIGH             ||
         property==SYMBOL_PROP_ASKLOW
        )
      )
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if a symbol supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CSymbolCustom::SupportProperty(ENUM_SYMBOL_PROP_STRING property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a short symbol description in the journal                |
//+------------------------------------------------------------------+
void CSymbolCustom::PrintShort(void)
  {
   ::Print(this.GetStatusDescription()+" "+this.Name());
  }
//+------------------------------------------------------------------+
```

**This concludes the development of the CSymbol descendant classes.**

The implementation of the remaining
descendant classes can be found at the end of the article and in the attached files.



Since we need to search and sort the symbol collection, we should create all the necessary functionality for that. Open the **Select.mqh**
file located in the

**\\MQL5\\Include\\DoEasy\\Services\** library folder and make additions to it.

First, include
the abstract symbol class:

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\Event.mqh"
#include "..\Objects\Accounts\Account.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
//+------------------------------------------------------------------+
```

**Declare all the necessary methods for searching and sorting**
**in the public section of the class:**

```
//+------------------------------------------------------------------+
//| Class for sorting objects meeting the criterion                  |
//+------------------------------------------------------------------+
class CSelect
  {
private:
   //--- Method for comparing two values
   template<typename T>
   static bool       CompareValues(T value1,T value2,ENUM_COMPARER_TYPE mode);
public:
//+------------------------------------------------------------------+
//| Methods of working with orders                                   |
//+------------------------------------------------------------------+
   //--- Return the list of orders with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the order index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property);
   //--- Return the order index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with events                                   |
//+------------------------------------------------------------------+
   //--- Return the list of events with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the event index with the maximum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property);
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property);
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property);
   //--- Return the event index with the minimum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property);
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property);
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with accounts                                 |
//+------------------------------------------------------------------+
   //--- Return the list of accounts with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the event index with the maximum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property);
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property);
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property);
   //--- Return the event index with the minimum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property);
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property);
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with symbols                                  |
//+------------------------------------------------------------------+
   //--- Return the list of symbols with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the symbol index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property);
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property);
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property);
   //--- Return the symbol index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property);
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property);
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

**Let's write their implementation outside the class:**

```
//+------------------------------------------------------------------+
//| Methods of working with symbol lists                             |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of symbols with one integer                      |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CSymbol *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of symbols with one real                         |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CSymbol *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of symbols with one string                       |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CSymbol *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the listed symbol index                                   |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CSymbol *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CSymbol *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed symbol index                                   |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CSymbol *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CSymbol *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed symbol index                                   |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CSymbol *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CSymbol *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed symbol index                                   |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindSymbolMin(CArrayObj* list_source,ENUM_SYMBOL_PROP_INTEGER property)
  {
   int index=0;
   CSymbol *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      CSymbol *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed symbol index                                   |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindSymbolMin(CArrayObj* list_source,ENUM_SYMBOL_PROP_DOUBLE property)
  {
   int index=0;
   CSymbol *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      CSymbol *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed symbol index                                   |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindSymbolMin(CArrayObj* list_source,ENUM_SYMBOL_PROP_STRING property)
  {
   int index=0;
   CSymbol *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      CSymbol *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
```

We have similar methods for each of the collection class. They have been considered in the [third \\
part of the library description](https://www.mql5.com/en/articles/5687#node01) when creating the CSelect class. Therefore, we will not dwell on them here.

**Now everything is ready for creating the symbol collection class.**

The four modes can be used for working with the
symbol collection from the program:

1. Working with a single symbol only,
2. Working with a list of symbols,
3. Working with the Market Watch window,
4. Working with a complete list of symbols available on the server.

In order for the character collection class to “know” what to work with, we will use the following scheme:


Set the operation of methods for working with symbols in the program settings. This may be one of the four announced operation modes.



The program should also have the string array to be filled with the library function following the same principle:

- if working with a single symbol, the array contains only the current symbol,
- if working with a custom symbol list that may also be located in the program settings with the necessary comma-separated symbols
defined, the array is filled with symbols from the string; if only one current symbol is set in the string or the list is empty, then
the current symbol is used for work



- if working with the market watch, "MARKET\_WATCH" instead of a symbol name is set in the only array cell
- if working with a complete list of symbols on the server, "ALL" instead of a symbol name is set in the array

All this is done automatically. A user only needs to arrange selection of the necessary mode for working with a symbol collection and
create at least one string array or a string array and a string of pre-defined symbols in the settings.

### Symbol collection class

In the \\MQL5\\Include\\DoEasy\ **Collections\** library folder, create the new class **CSymbolsCollection** in the **SymbolsCollection.mqh**
file.

The [CObject\\
standard library object class](https://www.mql5.com/en/docs/standardlibrary/cobject) is to be used as a base class for it.

Include all the necessary class files to a newly created file:

```
//+------------------------------------------------------------------+
//|                                            SymbolsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\Symbols\SymbolFX.mqh"
#include "..\Objects\Symbols\SymbolFXMajor.mqh"
#include "..\Objects\Symbols\SymbolFXMinor.mqh"
#include "..\Objects\Symbols\SymbolFXExotic.mqh"
#include "..\Objects\Symbols\SymbolFXRub.mqh"
#include "..\Objects\Symbols\SymbolMetall.mqh"
#include "..\Objects\Symbols\SymbolIndex.mqh"
#include "..\Objects\Symbols\SymbolIndicative.mqh"
#include "..\Objects\Symbols\SymbolCrypto.mqh"
#include "..\Objects\Symbols\SymbolCommodity.mqh"
#include "..\Objects\Symbols\SymbolExchange.mqh"
#include "..\Objects\Symbols\SymbolFutures.mqh"
#include "..\Objects\Symbols\SymbolCFD.mqh"
#include "..\Objects\Symbols\SymbolStocks.mqh"
#include "..\Objects\Symbols\SymbolBonds.mqh"
#include "..\Objects\Symbols\SymbolOption.mqh"
#include "..\Objects\Symbols\SymbolCollateral.mqh"
#include "..\Objects\Symbols\SymbolCustom.mqh"
#include "..\Objects\Symbols\SymbolCommon.mqh"
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CSymbolsCollection : public CObject
  {
private:

public:
//--- Constructor
                     CSymbolsCollection();

  };
//+------------------------------------------------------------------+
```

**Add class member values and methods that have already become standard for the library collection classes:**

```
//+------------------------------------------------------------------+
//|                                            SymbolsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\Symbols\SymbolFX.mqh"
#include "..\Objects\Symbols\SymbolFXMajor.mqh"
#include "..\Objects\Symbols\SymbolFXMinor.mqh"
#include "..\Objects\Symbols\SymbolFXExotic.mqh"
#include "..\Objects\Symbols\SymbolFXRub.mqh"
#include "..\Objects\Symbols\SymbolMetall.mqh"
#include "..\Objects\Symbols\SymbolIndex.mqh"
#include "..\Objects\Symbols\SymbolIndicative.mqh"
#include "..\Objects\Symbols\SymbolCrypto.mqh"
#include "..\Objects\Symbols\SymbolCommodity.mqh"
#include "..\Objects\Symbols\SymbolExchange.mqh"
#include "..\Objects\Symbols\SymbolFutures.mqh"
#include "..\Objects\Symbols\SymbolCFD.mqh"
#include "..\Objects\Symbols\SymbolStocks.mqh"
#include "..\Objects\Symbols\SymbolBonds.mqh"
#include "..\Objects\Symbols\SymbolOption.mqh"
#include "..\Objects\Symbols\SymbolCollateral.mqh"
#include "..\Objects\Symbols\SymbolCustom.mqh"
#include "..\Objects\Symbols\SymbolCommon.mqh"
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CSymbolsCollection : public CObject
  {
private:
   CListObj          m_list_all_symbols;     // The list of all symbol objects

public:
//--- Return the full collection list 'as is'
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_all_symbols;                                      }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)    { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }

//--- Constructor
                     CSymbolsCollection();

  };
//+------------------------------------------------------------------+
```

We have already considered all these variables and methods when creating previous collections. There is no need to discuss them here.

**Add the remaining variables and methods for working with the symbol collection class:**

```
//+------------------------------------------------------------------+
//|                                            SymbolsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\Symbols\SymbolFX.mqh"
#include "..\Objects\Symbols\SymbolFXMajor.mqh"
#include "..\Objects\Symbols\SymbolFXMinor.mqh"
#include "..\Objects\Symbols\SymbolFXExotic.mqh"
#include "..\Objects\Symbols\SymbolFXRub.mqh"
#include "..\Objects\Symbols\SymbolMetall.mqh"
#include "..\Objects\Symbols\SymbolIndex.mqh"
#include "..\Objects\Symbols\SymbolIndicative.mqh"
#include "..\Objects\Symbols\SymbolCrypto.mqh"
#include "..\Objects\Symbols\SymbolCommodity.mqh"
#include "..\Objects\Symbols\SymbolExchange.mqh"
#include "..\Objects\Symbols\SymbolFutures.mqh"
#include "..\Objects\Symbols\SymbolCFD.mqh"
#include "..\Objects\Symbols\SymbolStocks.mqh"
#include "..\Objects\Symbols\SymbolBonds.mqh"
#include "..\Objects\Symbols\SymbolOption.mqh"
#include "..\Objects\Symbols\SymbolCollateral.mqh"
#include "..\Objects\Symbols\SymbolCustom.mqh"
#include "..\Objects\Symbols\SymbolCommon.mqh"
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CSymbolsCollection : public CObject
  {
private:
   CListObj          m_list_all_symbols;     // The list of all symbol objects
   ENUM_SYMBOLS_MODE m_mode_list;            // Mode of working with symbol lists
   int               m_delta_symbol;         // Difference in the number of symbols compared to the previous check
   int               m_last_num_symbol;      // Number of symbols in the Market Watch window during the previous check
   int               m_global_error;         // Global error code
//--- Return the flag of a symbol object presence by its name in the list of all symbols
   bool              IsPresentSymbolInList(const string symbol_name);
//--- Create the symbol object and place it to the list
   bool              CreateNewSymbol(const ENUM_SYMBOL_STATUS symbol_status,const string name);
//--- Return the type of a used symbol list (Market watch/Server)
   ENUM_SYMBOLS_MODE TypeSymbolsList(const string &symbol_used_array[]);

//--- Define a symbol affiliation with a group by name and return it
   ENUM_SYMBOL_STATUS SymbolStatus(const string symbol_name)      const;
//--- Return a symbol affiliation with a category by custom criteria
   ENUM_SYMBOL_STATUS StatusByCustomPredefined(const string symbol_name)  const;
//--- Return a symbol affiliation with categories by margin calculation
   ENUM_SYMBOL_STATUS StatusByCalcMode(const string symbol_name)  const;
//--- Return a symbol affiliation with pre-defined (1) majors, (2) minors, (3) exotics, (4) RUB,
//--- (5) indicatives, (6) metals, (7) commodities, (8) indices, (9) cryptocurrency, (10) options
   bool              IsPredefinedFXMajor(const string name)       const;
   bool              IsPredefinedFXMinor(const string name)       const;
   bool              IsPredefinedFXExotic(const string name)      const;
   bool              IsPredefinedFXRUB(const string name)         const;
   bool              IsPredefinedIndicative(const string name)    const;
   bool              IsPredefinedMetall(const string name)        const;
   bool              IsPredefinedCommodity(const string name)     const;
   bool              IsPredefinedIndex(const string name)         const;
   bool              IsPredefinedCrypto(const string name)        const;
   bool              IsPredefinedOption(const string name)        const;

//--- Search for a symbol and return the flag indicating its presence on the server
   bool              Exist(const string name)                     const;

public:
//--- Return the full collection list 'as is'
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_all_symbols;                                      }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)    { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }
//--- Return the number of new symbols in the Market Watch window
   int               NewSymbols(void)    const                                                              { return this.m_delta_symbol;                                           }
//--- Return the mode of working with symbol lists
   ENUM_SYMBOLS_MODE ModeSymbolsList(void)                        const                                     { return this.m_mode_list;                                              }
//--- Constructor
                     CSymbolsCollection();

//--- Set the list of used symbols
   bool              SetUsedSymbol(const string &symbol_used_array[]);
//--- Update (1) all, (2) quote data of the collection symbols
   void              Refresh(void);
   void              RefreshRates(void);
  };
//+------------------------------------------------------------------+
```

The class should know what symbol class to work with: the current symbol, a specified symbol set, a symbol list located in the Market Watch
window or the complete list of all symbols on the server. The

**m\_mode\_list** class member variable stores one of the listed
modes of working with symbols.

When working with a list of Market Watch symbols, we need to constantly track this list (to be implemented in the next article), know the number
of symbols during the last check and correspondingly be aware of how
much this number has changed when adding/deleting a symbol(s) from the market watch list in order to re-arrange the **m\_list\_all\_symbols** symbol collection list in time and continue to accurately work with symbols.

Like in the previous collections, we
have introduced a new class member variable storing the

error code you can see and handle in the CEngine library base class.


When creating a new symbol object and adding it to the collection, we need to make sure there is no such symbol in the list. This is done by the

IsPresentSymbolInList() method.

The CreateNewSymbol()
method is used to create a new symbol and add it to the collection.

Since working in four symbol list modes is arranged in the
symbol collection, the

TypeSymbolsList() method is used to define the mode to work with.


Before creating a new symbol, we first need to define and set a group to assign it to (or a group a user assigns it to). The

SymbolStatus() method is used to define a symbol group (its status).


When defining a symbol status, its name is first searched in specified custom arrays using the

StatusByCustomPredefined() method.

If the method returns
"general" status, a symbol status is defined by the type of margin calculation by the

StatusByCalcMode() method.

The
additional methods are used to conduct a search by user arrays and defining what custom category a symbol belongs to. These methods return
flags indicating whether a symbol belongs to majors, minors, indices and other groups.

The Exist()
method returns the flag of a symbol existence on the server.

The NewSymbols()
method returns the number of new symbols added or removed from the Market Watch window, while the ModeSymbolsList()
method returns the mode of working with one of the four lists (the current symbol, a pre-defined symbol set, market watch list and the full number of
symbols on the server) to the calling program.

The SetUsedSymbol() method accepts a symbol array or
description of an operation mode with a list of symbols within a passed array and creates a symbol collection list.

The Refresh() method updates all data of all collection symbols
that can change, while the

RefreshRates() method updates only quote data of all collection
symbols. Both methods are called from the CEngine library main object timer.

We have defined all the methods necessary for working with the symbol collection class at this stage. Let's have a look at their structure.

**Implementing the class constructor:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSymbolsCollection::CSymbolsCollection(void) : m_last_num_symbol(0),m_delta_symbol(0),m_mode_list(SYMBOLS_MODE_CURRENT)
  {
   this.m_list_all_symbols.Sort(SORT_BY_SYMBOL_NAME);
   this.m_list_all_symbols.Clear();
   this.m_list_all_symbols.Type(COLLECTION_SYMBOLS_ID);
  }
//+------------------------------------------------------------------+
```

In the constructor initialization list, initialize the number of symbols
during the last check and the difference between the current and
previous number of symbols, as well as set the mode of working with a
symbol list as "working with the current symbol".

In the class body, set
sorting the symbol collection list by name, clear the list
and

assign the "symbol collection list" ID to it.

**The method of updating all data of all collection symbols:**

```
//+------------------------------------------------------------------+
//| Update all collection symbol data                                |
//+------------------------------------------------------------------+
void CSymbolsCollection::Refresh(void)
  {
   int total=this.m_list_all_symbols.Total();
   if(total==0)
      return;
   for(int i=0;i<total;i++)
     {
      CSymbol *symbol=this.m_list_all_symbols.At(i);
      if(symbol==NULL)
         continue;
      symbol.Refresh();
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the number of all collection symbols,get
the next symbol from the collection list and update its data using the
Refresh() method of the CSymbol class considered in the [previous \\
article](https://www.mql5.com/en/articles/7014#node01).

**The method of updating quote data of all collection symbols:**

```
//+------------------------------------------------------------------+
//| Update quote data of the collection symbols                      |
//+------------------------------------------------------------------+
void CSymbolsCollection::RefreshRates(void)
  {
   int total=this.m_list_all_symbols.Total();
   if(total==0)
      return;
   for(int i=0;i<total;i++)
     {
      CSymbol *symbol=this.m_list_all_symbols.At(i);
      if(symbol==NULL)
         continue;
      symbol.RefreshRates();
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the number of all collection symbols,get
the next symbol from the collection list and update its data using the
RefreshRates() method of the CSymbol class.

**The method for creating a new symbol object and placing it in the symbol collection list:**

```
//+------------------------------------------------------------------+
//| Create a symbol object and place it to the list                  |
//+------------------------------------------------------------------+
bool CSymbolsCollection::CreateNewSymbol(const ENUM_SYMBOL_STATUS symbol_status,const string name)
  {
   if(this.IsPresentSymbolInList(name))
     {
      return true;
     }
   if(#ifdef __MQL5__ !::SymbolInfoInteger(name,SYMBOL_EXIST) #else !this.Exist(name) #endif )
     {
      string t1=TextByLanguage("Ошибка входных данных: нет символа ","Input error: no ");
      string t2=TextByLanguage(" на сервере"," symbol on the server");
      ::Print(DFUN,t1,name,t2);
      this.m_global_error=ERR_MARKET_UNKNOWN_SYMBOL;
      return false;
     }
   CSymbol *symbol=NULL;
   switch(symbol_status)
     {
      case SYMBOL_STATUS_FX         :  symbol=new CSymbolFX(name);         break;   // Forex symbol
      case SYMBOL_STATUS_FX_MAJOR   :  symbol=new CSymbolFXMajor(name);    break;   // Major Forex symbol
      case SYMBOL_STATUS_FX_MINOR   :  symbol=new CSymbolFXMinor(name);    break;   // Minor Forex symbol
      case SYMBOL_STATUS_FX_EXOTIC  :  symbol=new CSymbolFXExotic(name);   break;   // Exotic Forex symbol
      case SYMBOL_STATUS_FX_RUB     :  symbol=new CSymbolFXRub(name);      break;   // Forex symbol/RUR
      case SYMBOL_STATUS_METAL      :  symbol=new CSymbolMetall(name);     break;   // Metal
      case SYMBOL_STATUS_INDEX      :  symbol=new CSymbolIndex(name);      break;   // Index
      case SYMBOL_STATUS_INDICATIVE :  symbol=new CSymbolIndicative(name); break;   // Indicative
      case SYMBOL_STATUS_CRYPTO     :  symbol=new CSymbolCrypto(name);     break;   // Cryptocurrency symbol
      case SYMBOL_STATUS_COMMODITY  :  symbol=new CSymbolCommodity(name);  break;   // Commodity
      case SYMBOL_STATUS_EXCHANGE   :  symbol=new CSymbolExchange(name);   break;   // Exchange symbol
      case SYMBOL_STATUS_FUTURES    :  symbol=new CSymbolFutures(name);    break;   // Futures
      case SYMBOL_STATUS_CFD        :  symbol=new CSymbolCFD(name);        break;   // CFD
      case SYMBOL_STATUS_STOCKS     :  symbol=new CSymbolStocks(name);     break;   // Stock
      case SYMBOL_STATUS_BONDS      :  symbol=new CSymbolBonds(name);      break;   // Bond
      case SYMBOL_STATUS_OPTION     :  symbol=new CSymbolOption(name);     break;   // Option
      case SYMBOL_STATUS_COLLATERAL :  symbol=new CSymbolCollateral(name); break;   // Non-tradable asset
      case SYMBOL_STATUS_CUSTOM     :  symbol=new CSymbolCustom(name);     break;   // Custom symbol
      default                       :  symbol=new CSymbolCommon(name);     break;   // The rest
     }
   if(symbol==NULL)
     {
      ::Print(DFUN,TextByLanguage("Не удалось создать объект-символ ","Failed to create symbol object "),name);
      return false;
     }
   if(!this.m_list_all_symbols.Add(symbol))
     {
      string t1=TextByLanguage("Не удалось добавить символ ","Failed to add ");
      string t2=TextByLanguage(" в список"," symbol to the list");
      ::Print(DFUN,t1,name,t2);
      delete symbol;
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method passes the symbol status and name.

If such a symbol is already present in the symbol collection list,

return true"silently" — there is no error, but there is no need to add a symbol since one already exists.

Next, check
if a symbol exists on the server by its name. If there is no such symbol, display the symbol absence message, assign
the "unknown symbol" value to the error code and return false.


If the symbol exists, create a new symbol object

depending on the status passed to the method. To create a symbol
object, use descendant classes of the abstract symbol that correspond to the passed status.

In case of an object creation error, display the appropriate
message and

return false.


If a new symbol object is created successfully,

add it to the symbol collection list and return

truein case of successful
adding, or display an error message and

return false
if unsuccessful.

**Implementing the method returning the symbol presence flag in the collection list:**

```
//+------------------------------------------------------------------+
//| Return the symbol object presence flag                           |
//| by its name in the list of all symbols                           |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPresentSymbolInList(const string symbol_name)
  {
   CArrayObj *list=dynamic_cast<CListObj*>(&this.m_list_all_symbols);
   list.Sort(SORT_BY_SYMBOL_NAME);
   list=CSelect::BySymbolProperty(list,SYMBOL_PROP_NAME,symbol_name,EQUAL);
   return(list==NULL || list.Total()==0 ? false : true);
  }
//+------------------------------------------------------------------+
```

The name of a necessary symbol is passed to the method, then the
list is sorted by a symbol name and a
passed name. If the list is not empty, the symbol with such a name has been found — return true.
Otherwise, return

false — no symbol is present in the list.

**Implementing the method setting the list of collection symbols:**

```
//+------------------------------------------------------------------+
//| Set the list of used symbols                                     |
//+------------------------------------------------------------------+
bool CSymbolsCollection::SetUsedSymbol(const string &symbol_used_array[])
  {
   this.m_mode_list=this.TypeSymbolsList(symbol_used_array);
   this.m_list_all_symbols.Clear();
   this.m_list_all_symbols.Sort(SORT_BY_SYMBOL_NAME);
   //--- Use only the current symbol
   if(this.m_mode_list==SYMBOLS_MODE_CURRENT)
     {
      string name=::Symbol();
      ENUM_SYMBOL_STATUS status=this.SymbolStatus(name);
      return this.CreateNewSymbol(status,name);
     }
   else
     {
      bool res=true;
      //--- Use the pre-defined symbol list
      if(this.m_mode_list==SYMBOLS_MODE_DEFINES)
        {
         int total=::ArraySize(symbol_used_array);
         for(int i=0;i<total;i++)
           {
            string name=symbol_used_array[i];
            ENUM_SYMBOL_STATUS status=this.SymbolStatus(name);
            bool add=this.CreateNewSymbol(status,name);
            res &=add;
            if(!add)
               continue;
           }
         return res;
        }
      //--- Use the full list of the server symbols
      else if(this.m_mode_list==SYMBOLS_MODE_ALL)
        {
         int total=::SymbolsTotal(false);
         for(int i=0;i<total;i++)
           {
            string name=::SymbolName(i,false);
            ENUM_SYMBOL_STATUS status=this.SymbolStatus(name);
            bool add=this.CreateNewSymbol(status,name);
            res &=add;
            if(!add)
               continue;
           }
         return res;
        }
      //--- Use the symbol list from the Market Watch window
      else if(this.m_mode_list==SYMBOLS_MODE_MARKET_WATCH)
        {
         int total=::SymbolsTotal(true);
         for(int i=0;i<total;i++)
           {
            string name=::SymbolName(i,true);
            ENUM_SYMBOL_STATUS status=this.SymbolStatus(name);
            bool add=this.CreateNewSymbol(status,name);
            res &=add;
            if(!add)
               continue;
           }
         return res;
        }
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method creates the symbol collection list depending on the content of the symbol
array passed to it. First, define the method of working with a symbol list
(one symbol/custom set of symbols/market watch/full list), then the list
is cleared and sorted by name.

If only
the current symbol is used,

- define the symbol status by its name and

- return the result of creating a new symbol object and adding it to the symbol
collection list.

If a pre-defined symbol list is used,

- get a symbol name from the array in a loop by a symbol name array passed to the
method
- define the symbol status by its name
- add the result of creating and adding a symbol object to the symbol
collection list to the returned variable
- if failed to create a symbol or add it to the list, move
to the next one
- upon completion of the loop, return the status of the variable the results of
the collection symbols creation were written to


The remaining two modes are handled similar to the mode of working with a pre-defined list. However, instead of the array passed to the method,
the symbols are taken

either from the market watch, or from
the full list of symbols on the server.

**Implementing the method returning the mode of working with symbol lists:**

```
//+------------------------------------------------------------------+
//|Return the type of a used symbol list (Market watch/Server)       |
//+------------------------------------------------------------------+
ENUM_SYMBOLS_MODE CSymbolsCollection::TypeSymbolsList(const string &symbol_used_array[])
  {
   int total=::ArraySize(symbol_used_array);
   if(total<1)
      return SYMBOLS_MODE_CURRENT;
   string type=::StringSubstr(symbol_used_array[0],13);
   return
     (
      type=="MARKET_WATCH" ? SYMBOLS_MODE_MARKET_WATCH   :
      type=="ALL"          ? SYMBOLS_MODE_ALL            :
      (total==1 && symbol_used_array[0]==::Symbol() ? SYMBOLS_MODE_CURRENT : SYMBOLS_MODE_DEFINES)
     );
  }
//+------------------------------------------------------------------+
```

The method receives the array with symbol names or with a
description of modes of working with lists.

If an empty array is passed, return the operation mode only with the current
symbol.

Next, receive the array contents from its zero cell and,

- if starting from the index 13 of the sub-string, there
is the "MARKET\_WATCH" entry, return the mode of working with the Market Watch window.
- If there is the "ALL" string there, return
the mode of working with the full list of symbols.
- Otherwise,
  - if the array contains a single entrycontaining
     the current symbol name, return the mode of working with the current symbol only.

  - in the last of the possible options, return the operation mode
     with a pre-defined list.

**Implementing the method returning a symbol affiliation with a group by its name:**

```
//+------------------------------------------------------------------+
//| Define a symbol affiliation with a group by name and return it   |
//+------------------------------------------------------------------+
ENUM_SYMBOL_STATUS CSymbolsCollection::SymbolStatus(const string symbol_name) const
  {
   ENUM_SYMBOL_STATUS status=this.StatusByCustomPredefined(symbol_name);
   return(status==SYMBOL_STATUS_COMMON ? this.StatusByCalcMode(symbol_name) : status);
  }
//+------------------------------------------------------------------+
```

A symbol name is passed to the method. Next, its
affiliation with specified custom groups is checked. If the "general
group" status is returned, search for a group by the "margin calculation
mode" symbol property. As a result, return the obtained status.
By the way, it may be equated to one of the groups according to the profit and margin calculation mode, or it may remain in the general group of
symbols.

As a result, the symbol has a custom group status. If it is not present in the custom groups, a group by [a \\
margin calculation method](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_calc_mode) may be assigned to it. If failed to define it within that group, it remains in the general group.

**Implementing the methods returning the flag of the symbol affiliation with certain custom groups:**

```
//+------------------------------------------------------------------+
//| Return a symbol affiliation with majors                          |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedFXMajor(const string name) const
  {
   int total=::ArraySize(DataSymbolsFXMajors);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsFXMajors[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with minors                          |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedFXMinor(const string name) const
  {
   int total=::ArraySize(DataSymbolsFXMinors);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsFXMinors[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with exotic symbols                  |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedFXExotic(const string name) const
  {
   int total=::ArraySize(DataSymbolsFXExotics);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsFXExotics[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with RUB symbols                     |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedFXRUB(const string name) const
  {
   int total=::ArraySize(DataSymbolsFXRub);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsFXRub[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with indicative symbols              |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedIndicative(const string name) const
  {
   int total=::ArraySize(DataSymbolsFXIndicatives);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsFXIndicatives[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with metals                          |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedMetall(const string name) const
  {
   int total=::ArraySize(DataSymbolsMetalls);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsMetalls[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with commodities                     |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedCommodity(const string name) const
  {
   int total=::ArraySize(DataSymbolsCommodities);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsCommodities[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with indices                         |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedIndex(const string name) const
  {
   int total=::ArraySize(DataSymbolsIndexes);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsIndexes[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with a cryptocurrency                |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedCrypto(const string name) const
  {
   int total=::ArraySize(DataSymbolsCrypto);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsCrypto[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return a symbol affiliation with options                         |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPredefinedOption(const string name) const
  {
   int total=::ArraySize(DataSymbolsOptions);
   for(int i=0;i<total;i++)
      if(name==DataSymbolsOptions[i])
         return true;
   return false;
  }
//+------------------------------------------------------------------+
```

Depending on a method name (of a checked custom group), the search for a symbol whose name has been passed to the method is performed in the custom array
corresponding to its group.

If such a symbol is found in the array, true
is returned. Otherwise — false.

**The method returning a symbol status by its presence in the custom groups:**

```
//+------------------------------------------------------------------+
//| Return a category by custom criteria                             |
//+------------------------------------------------------------------+
ENUM_SYMBOL_STATUS CSymbolsCollection::StatusByCustomPredefined(const string symbol_name) const
  {
   return
     (
      this.IsPredefinedFXMajor(symbol_name)     ?  SYMBOL_STATUS_FX_MAJOR     :
      this.IsPredefinedFXMinor(symbol_name)     ?  SYMBOL_STATUS_FX_MINOR     :
      this.IsPredefinedFXExotic(symbol_name)    ?  SYMBOL_STATUS_FX_EXOTIC    :
      this.IsPredefinedFXRUB(symbol_name)       ?  SYMBOL_STATUS_FX_RUB       :
      this.IsPredefinedOption(symbol_name)      ?  SYMBOL_STATUS_OPTION       :
      this.IsPredefinedCommodity(symbol_name)   ?  SYMBOL_STATUS_COMMODITY    :
      this.IsPredefinedCrypto(symbol_name)      ?  SYMBOL_STATUS_CRYPTO       :
      this.IsPredefinedMetall(symbol_name)      ?  SYMBOL_STATUS_METAL        :
      this.IsPredefinedIndex(symbol_name)       ?  SYMBOL_STATUS_INDEX        :
      this.IsPredefinedIndicative(symbol_name)  ?  SYMBOL_STATUS_INDICATIVE   :
      SYMBOL_STATUS_COMMON
     );
  }
//+------------------------------------------------------------------+
```

A symbol name is passed to the method and its
presence in each of the custom groups is checked one by one using the above methods. As soon as a symbol appears in any of the groups, a
status corresponding to the group where it has been found is returned.

If the symbol is not found in any of the groups, the "general symbol group" status is
returned.

**The method returning a symbol affiliation with a group by the margin calculation method:**

```
//+------------------------------------------------------------------+
//|Return affiliation with a margin calculation category             |
//+------------------------------------------------------------------+
ENUM_SYMBOL_STATUS CSymbolsCollection::StatusByCalcMode(const string symbol_name) const
  {
   ENUM_SYMBOL_CALC_MODE calc_mode=(ENUM_SYMBOL_CALC_MODE)::SymbolInfoInteger(symbol_name,SYMBOL_TRADE_CALC_MODE);
   return
     (
      calc_mode==SYMBOL_CALC_MODE_EXCH_OPTIONS_MARGIN                                                                               ?  SYMBOL_STATUS_OPTION       :
      calc_mode==SYMBOL_CALC_MODE_SERV_COLLATERAL                                                                                   ?  SYMBOL_STATUS_COLLATERAL   :
      calc_mode==SYMBOL_CALC_MODE_FUTURES                                                                                           ?  SYMBOL_STATUS_FUTURES      :
      calc_mode==SYMBOL_CALC_MODE_CFD           || calc_mode==SYMBOL_CALC_MODE_CFDINDEX || calc_mode==SYMBOL_CALC_MODE_CFDLEVERAGE  ?  SYMBOL_STATUS_CFD          :
      calc_mode==SYMBOL_CALC_MODE_FOREX         || calc_mode==SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE                                    ?  SYMBOL_STATUS_FX           :
      calc_mode==SYMBOL_CALC_MODE_EXCH_STOCKS   || calc_mode==SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX                                     ?  SYMBOL_STATUS_STOCKS       :
      calc_mode==SYMBOL_CALC_MODE_EXCH_BONDS    || calc_mode==SYMBOL_CALC_MODE_EXCH_BONDS_MOEX                                      ?  SYMBOL_STATUS_BONDS        :
      calc_mode==SYMBOL_CALC_MODE_EXCH_FUTURES  || calc_mode==SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS                                   ?  SYMBOL_STATUS_FUTURES      :
      SYMBOL_STATUS_COMMON
     );
  }
//+------------------------------------------------------------------+
```

The method receives a symbol name, next we receive
the margin calculation method for the symbol and return the symbol status
depending on an obtained value. If none of the calculation methods has
been identified, return the "general symbol group" status.

**The method looking for a symbol on the server and returning the flag of its presence:**

```
//+---------------------------------------------------------------------------------+
//| Search for a symbol and return the flag indicating its presence on the server   |
//+---------------------------------------------------------------------------------+
bool CSymbolsCollection::Exist(const string name) const
  {
   int total=::SymbolsTotal(false);
   for(int i=0;i<total;i++)
      if(::SymbolName(i,false)==name)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
```

Symbol collection class is ready. Now we need to launch it. As usual, launching and handling are performed in the CEngine class. Let's make the
necessary changes to it.

**Include the symbol collection class file to the CEngine class file and**
**declare a**

**symbol collection object, as well as the necessary methods:**

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Services\TimerCounter.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CSymbolsCollection   m_symbols;                       // Symbol collection
   CArrayObj            m_list_counters;                 // List of timer counters
   int                  m_global_error;                  // Global error code
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_tester;                     // Flag of working in the tester
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   bool                 m_is_account_event;              // Account change event flag
   ENUM_TRADE_EVENT     m_last_trade_event;              // Last account trading event
   ENUM_ACCOUNT_EVENT   m_last_account_event;            // Last event in the account properties
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return the (1) first launch flag, (2) presence of the flag in the trading event
   bool                 IsFirstStart(void);
//--- Work with (1) order, deal and position, (2) account events
   void                 TradeEventsControl(void);
   void                 AccountEventsControl(void);
//--- (1) Working with a symbol collection and (2) symbol list events in the market watch window
   void                 SymbolEventsControl(void);
   void                 MarketWatchEventsControl(void);
//--- Return the last (1) market pending order, (2) market order, (3) last position, (4) position by ticket
   COrder              *GetLastMarketPending(void);
   COrder              *GetLastMarketOrder(void);
   COrder              *GetLastPosition(void);
   COrder              *GetPosition(const ulong ticket);
//--- Return the last (1) removed pending order, (2) historical market order, (3) historical order (market or pending one) by its ticket
   COrder              *GetLastHistoryPending(void);
   COrder              *GetLastHistoryOrder(void);
   COrder              *GetHistoryOrder(const ulong ticket);
//--- Return the (1) first and the (2) last historical market orders from the list of all position orders, (3) the last deal
   COrder              *GetFirstOrderPosition(const ulong position_id);
   COrder              *GetLastOrderPosition(const ulong position_id);
   COrder              *GetLastDeal(void);
public:
   //--- Return the list of market (1) positions, (2) pending orders and (3) market orders
   CArrayObj           *GetListMarketPosition(void);
   CArrayObj           *GetListMarketPendings(void);
   CArrayObj           *GetListMarketOrders(void);
   //--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals, (4) all position market orders by its id
   CArrayObj           *GetListHistoryOrders(void);
   CArrayObj           *GetListHistoryPendings(void);
   CArrayObj           *GetListDeals(void);
   CArrayObj           *GetListAllOrdersByPosID(const ulong position_id);
//--- Return the list of (1) accounts, (2) account events, (3) account change event by its index in the list
//--- (4) the current account, (5) event description
   CArrayObj           *GetListAllAccounts(void)                        { return this.m_accounts.GetList();                   }
   CArrayInt           *GetListAccountEvents(void)                      { return this.m_accounts.GetListChanges();            }
   ENUM_ACCOUNT_EVENT   GetAccountEventByIndex(const int index)         { return this.m_accounts.GetEvent(index);             }
   CAccount            *GetAccountCurrent(void);
   string               GetAccountEventDescription(ENUM_ACCOUNT_EVENT event);
//--- Return the list of used symbols
   CArrayObj           *GetListAllUsedSymbols(void)                     { return this.m_symbols.GetList();                    }

//--- Return the list of order, deal and position events
   CArrayObj           *GetListAllOrdersEvents(void)                    { return this.m_events.GetList();                     }
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_events.ResetLastTradeEvent(); }
//--- Return the (1) last trading event, (2) the last event in the account properties, (3) hedging account flag, (4) flag of working in the tester
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_last_trade_event;                     }
   ENUM_ACCOUNT_EVENT   LastAccountEvent(void)                    const { return this.m_last_account_event;                   }
   bool                 IsHedge(void)                             const { return this.m_is_hedge;                             }
   bool                 IsTester(void)                            const { return this.m_is_tester;                            }
   bool                 IsAccountsEvent(void)                     const { return this.m_accounts.IsAccountEvent();            }
//--- Return an account event code
   int                  GetAccountEventsCode(void)                const { return this.m_accounts.GetEventCode();              }
//--- Return CEngine global error code
   int                  GetError(void)                            const { return this.m_global_error;                         }
//--- Create the timer counter
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
//--- Set the list of used symbols
   bool                 SetUsedSymbols(const string &array_symbols[])   { return this.m_symbols.SetUsedSymbol(array_symbols); }
//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

The SymbolEventsControl() method is used to update quote data of
all collection symbols, while the

MarketWatchEventsControl() method is used to update the remaining
data of all collection symbols and track events in the Market Watch window (to be considered in the next article's symbol collection events
class).

The GetListAllUsedSymbols() method returns the full list of the
symbol collection using the GetList() method of the CSymbolsCollection class to the calling program.

The SetUsedSymbols() method calls the same-name
SetUsedSymbol() method of the CSymbolsCollection class, which in turn fills the collection list with symbol objects of all symbols used in
the program.

Let's consider the structure of these methods.

**In the class constructor, create the counters of the first**
**and second symbol collection timers.** In the first timer, we
will update quote data of all collection symbols, while in the second one, we will update the remaining symbol data and manage Market Watch
window events.

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),m_last_trade_event(TRADE_EVENT_NO_EVENT),m_last_account_event(ACCOUNT_EVENT_NO_EVENT),m_global_error(ERR_SUCCESS)
  {
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_is_tester=::MQLInfoInteger(MQL_TESTER);

   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_ORD_COUNTER_ID,COLLECTION_ORD_COUNTER_STEP,COLLECTION_ORD_PAUSE);
   this.CreateCounter(COLLECTION_ACC_COUNTER_ID,COLLECTION_ACC_COUNTER_STEP,COLLECTION_ACC_PAUSE);

   this.CreateCounter(COLLECTION_SYM_COUNTER_ID1,COLLECTION_SYM_COUNTER_STEP1,COLLECTION_SYM_PAUSE1);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID2,COLLECTION_SYM_COUNTER_STEP2,COLLECTION_SYM_PAUSE2);

   ::ResetLastError();
   #ifdef __MQL5__
      if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   //---__MQL4__
   #else
      if(!this.IsTester() && !::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   #endif
  }
//+------------------------------------------------------------------+
```

**Add strings for working with two symbol collection timers**
**to the class OnTimer() handler:**

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of the collections of historical orders and deals, as well as of market orders and positions
   int index=this.CounterIndex(COLLECTION_ORD_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the order, deal and position collections events
            if(counter.IsTimeDone())
               this.TradeEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
            this.TradeEventsControl();
        }
     }
//--- Account collection timer
   index=this.CounterIndex(COLLECTION_ACC_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the account collection events
            if(counter.IsTimeDone())
               this.AccountEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
            this.AccountEventsControl();
        }
     }

//--- Timer 1 of the symbol collection (updating symbol quote data in the collection)
   index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID1);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If the pause is over, update quote data of all symbols in the collection
            if(counter.IsTimeDone())
               this.SymbolEventsControl();
           }
         //--- In case of a tester, update quote data of all collection symbols by tick
         else
            this.SymbolEventsControl();
        }
     }
//--- Timer 2 of the symbol collection (updating all data of all symbols in the collection and track symbol search events in the market watch window)
   index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID2);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If the pause is over
            if(counter.IsTimeDone())
              {
               //--- update data of all symbols in the collection
               this.m_symbols.Refresh();
               //--- When workign with the market watch list, check the market watch window events
               if(this.m_symbols.ModeSymbolsList()==SYMBOLS_MODE_MARKET_WATCH)
                  this.MarketWatchEventsControl();
              }
           }
         //--- In case of a tester, update data of all collection symbols by tick
         else
            this.m_symbols.Refresh();
        }
     }
  }
//+------------------------------------------------------------------+
```

The entire logic is provided in the code comments. There is no point in dwelling on it here.

**In the current implementation, the method working with a symbol collection simply updates**
**quote data of all collection symbols using the RefreshRates method of the CSymbolsCollection class:**

```
//+------------------------------------------------------------------+
//| Working with a symbol collection                                 |
//+------------------------------------------------------------------+
void CEngine::SymbolEventsControl(void)
  {
   this.m_symbols.RefreshRates();
  }
//+------------------------------------------------------------------+
```

In the current implementation, **the method of working with the Market Watch window events** only checks
that it has been called from the tester and exits if it works in the tester (there is no point in tracking market watch window events in
the tester):

```
//+------------------------------------------------------------------+
//| Working with symbol list events in the market watch window       |
//+------------------------------------------------------------------+
void CEngine::MarketWatchEventsControl(void)
  {
   if(this.IsTester())
      return;
//--- Tracking Market Watch window events

//---
  }
//+------------------------------------------------------------------+
```

We will do all the work related to symbol collection events in the next article.

These are all the necessary changes to the CEngine class.

Since we need to pass the required mode of working with symbols to the symbol collection class, we need the function defining the operation mode
and correctly fill in the symbol array to be passed to the main object of the CEngine library.

Open the **DELib.mqh** service functions file from \\MQL5\\Include\\DoEasy\ **Services\** and add the necessary
function to it:

```
//+------------------------------------------------------------------+
//| Prepare the symbol array for a symbol collection                 |
//+------------------------------------------------------------------+
bool CreateUsedSymbolsArray(const ENUM_SYMBOLS_MODE mode_used_symbols,string defined_used_symbols,string &used_symbols_array[])
  {
   //--- When working with the current symbol
   if(mode_used_symbols==SYMBOLS_MODE_CURRENT)
     {
      //--- Write the name of the current symbol to the only array cell
      ArrayResize(used_symbols_array,1);
      used_symbols_array[0]=Symbol();
      return true;
     }
   //--- If working with a predefined symbol set (from the defined_used_symbols string)
   else if(mode_used_symbols==SYMBOLS_MODE_DEFINES)
     {
      //--- Set a comma as a separator
      string separator=",";
      //--- Replace erroneous separators with correct ones
      if(StringFind(defined_used_symbols,";")>WRONG_VALUE)  StringReplace(defined_used_symbols,";",separator);
      if(StringFind(defined_used_symbols,":")>WRONG_VALUE)  StringReplace(defined_used_symbols,":",separator);
      if(StringFind(defined_used_symbols,"|")>WRONG_VALUE)  StringReplace(defined_used_symbols,"|",separator);
      if(StringFind(defined_used_symbols,"/")>WRONG_VALUE)  StringReplace(defined_used_symbols,"/",separator);
      if(StringFind(defined_used_symbols,"\\")>WRONG_VALUE) StringReplace(defined_used_symbols,"\\",separator);
      if(StringFind(defined_used_symbols,"'")>WRONG_VALUE)  StringReplace(defined_used_symbols,"'",separator);
      if(StringFind(defined_used_symbols,"-")>WRONG_VALUE)  StringReplace(defined_used_symbols,"-",separator);
      if(StringFind(defined_used_symbols,"`")>WRONG_VALUE)  StringReplace(defined_used_symbols,"`",separator);
      //--- Delete as long as there are spaces
      while(StringFind(defined_used_symbols," ")>WRONG_VALUE && !IsStopped())
         StringReplace(defined_used_symbols," ","");
      //--- As soon as there are double separators (after removing spaces between them), replace them with a separator
      while(StringFind(defined_used_symbols,separator+separator)>WRONG_VALUE && !IsStopped())
         StringReplace(defined_used_symbols,separator+separator,separator);
      //--- If a single separator remains before the first symbol in the string, replace it with a space
      if(StringFind(defined_used_symbols,separator)==0)
         StringSetCharacter(defined_used_symbols,0,32);
      //--- If a single separator remains after the last symbol in the string, replace it with a space
      if(StringFind(defined_used_symbols,separator)==StringLen(defined_used_symbols)-1)
         StringSetCharacter(defined_used_symbols,StringLen(defined_used_symbols)-1,32);
      //--- Remove all redundant things to the left and right
      #ifdef __MQL5__
         StringTrimLeft(defined_used_symbols);
         StringTrimRight(defined_used_symbols);
      //---  __MQL4__
      #else
         defined_used_symbols=StringTrimLeft(defined_used_symbols);
         defined_used_symbols=StringTrimRight(defined_used_symbols);
      #endif
      //--- Prepare the array
      ArrayResize(used_symbols_array,0);
      ResetLastError();
      //--- divide the string by separators (comma) and add all found substrings to the array
      int n=StringSplit(defined_used_symbols,StringGetCharacter(separator,0),used_symbols_array);
      //--- if nothing is found, display the appropriate message (working with the current symbol is selected automatically)
      if(n<1)
        {
         string err=
           (n==0  ?
            DFUN_ERR_LINE+TextByLanguage("Ошибка. Строка предопределённых символов пустая, будет использоваться ","Error. String of predefined symbols empty, symbol will be used: ")+Symbol() :
            DFUN_ERR_LINE+TextByLanguage("Не удалось подготовить массив используемых символов. Ошибка ","Failed to create array of used characters. Error ")+(string)GetLastError()
           );
         Print(err);
         return false;
        }
     }
   //--- If working with the Market Watch window or the full list
   else
     {
      //--- Add the (mode_used_symbols) working mode to the only array cell
      ArrayResize(used_symbols_array,1);
      used_symbols_array[0]=EnumToString(mode_used_symbols);
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives the mode of working with the symbol collection,
which is either to be set in the program settings, or strictly defined (if no mode selection is required),

the comma-separated list of symbols you want to work with (or an
empty string) and the

array the symbol list is to be added to or the working mode
(for working with the market watch and the full list of server symbols), which is to be sent to the CEngine class for setting the library mode of
working with symbols.

All actions of the function directed at the list and the array are described in details directly in the listing. There is no point in
considering them separately.

This concludes the development of the symbol collection class. All is ready for testing.

### Symbol collection test

To test the collection, we will use the EA [from the previous article](https://www.mql5.com/en/articles/7014#node02)
and save it in \\MQL5\\Experts\\TestDoEasy\ under the name

**Part15\\TestDoEasyPart15\_1.mq5**.

In the inputs, add selecting the mode of working with the symbol collection
of the library and a

string variable storing the list of custom symbols you should
work with in case this mode is selected in the settings:

```
//--- input variables
input ulong             InpMagic             =  123;  // Magic number
input double            InpLots              =  0.1;  // Lots
input uint              InpStopLoss          =  50;   // StopLoss in points
input uint              InpTakeProfit        =  50;   // TakeProfit in points
input uint              InpDistance          =  50;   // Pending orders distance (points)
input uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)
input uint              InpSlippage          =  0;    // Slippage in points
input double            InpWithdrawal        =  10;   // Withdrawal funds (in tester)
input uint              InpButtShiftX        =  40;   // Buttons X shift
input uint              InpButtShiftY        =  10;   // Buttons Y shift
input uint              InpTrailingStop      =  50;   // Trailing Stop (points)
input uint              InpTrailingStep      =  20;   // Trailing Step (points)
input uint              InpTrailingStart     =  0;    // Trailing Start (points)
input uint              InpStopLossModify    =  20;   // StopLoss for modification (points)
input uint              InpTakeProfitModify  =  60;   // TakeProfit for modification (points)
input ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;   // Mode of used symbols list
input string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)

//--- global variables
```

**In the list of global variables, add the variable for storing a custom**
**symbol list and a string array for passing the list of symbols to the library:**

```
//--- global variables
CEngine        engine;
#ifdef __MQL5__
CTrade         trade;
#endif
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ulong          magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           slippage;
bool           trailing_on;
double         trailing_stop;
double         trailing_step;
uint           trailing_start;
uint           stoploss_to_modify;
uint           takeprofit_to_modify;
string         used_symbols;
string         array_used_symbols[];
//+------------------------------------------------------------------+
```

**In the EA's OnInit() handler, assign the custom list to the variable for**
**storing it, fill in the array of used symbols and send**
**it to the library:**

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();

//--- Set EA global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      butt_data[i].name=prefix+EnumToString((ENUM_BUTTONS)i);
      butt_data[i].text=EnumToButtText((ENUM_BUTTONS)i);
     }
   lot=NormalizeLot(Symbol(),fmax(InpLots,MinimumLots(Symbol())*2.0));
   magic_number=InpMagic;
   stoploss=InpStopLoss;
   takeprofit=InpTakeProfit;
   distance_pending=InpDistance;
   distance_stoplimit=InpDistanceSL;
   slippage=InpSlippage;
   trailing_stop=InpTrailingStop*Point();
   trailing_step=InpTrailingStep*Point();
   trailing_start=InpTrailingStart;
   stoploss_to_modify=InpStopLossModify;
   takeprofit_to_modify=InpTakeProfitModify;

//--- Fill in the array of used symbols
   used_symbols=InpUsedSymbols;
   CreateUsedSymbolsArray(InpModeUsedSymbols,used_symbols,array_used_symbols);

//--- Set the type of the used symbol list in the symbol collection
   engine.SetUsedSymbols(array_used_symbols);

//--- Check and remove remaining EA graphical objects
```

The SetUsedSymbols() function considered above creates the symbol array to be sent to the symbol collection class. Depending on the
selected mode, the array features either the current symbol, or the custom symbol list, or a string description of the mode of working with
the Market Watch window or with a complete symbol list on the server.

**At the very end of the OnInit() handler, add the code for the fast check of the symbol lists created by the symbol collection class:**

```
//--- Set CTrade trading class parameters
#ifdef __MQL5__
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
#endif
//--- Fast check of the symbol object collection
   CArrayObj *list=engine.GetListAllUsedSymbols();
   CSymbol *symbol=NULL;
   if(list!=NULL)
     {
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         symbol.Refresh();
         symbol.RefreshRates();
         symbol.PrintShort();
         if(InpModeUsedSymbols<SYMBOLS_MODE_MARKET_WATCH)
            symbol.Print();
        }
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Here we obtain the complete list of all collection symbols.
In

the loop by the obtained list, get the next symbol from it, update
all the data and print out the description of the symbol in the journal. At first, the brief
one is displayed. Then, if the "Working with the market watch window" or
"Working with the full list of symbols on the server" mode is not selected, display
the full description of the symbol properties in the journal.

Launch the EA in the terminal window and select the "Working with symbols from the Market Watch window" mode in the settings. As a result, the list
with brief descriptions of all collection symbols created by the symbol collection class is displayed in the journal:

```
2019.06.27 10:01:52.756 Stock ALNU
2019.06.27 10:01:52.756 Stock SU25075RMFS1
2019.06.27 10:01:52.756 Bond SU46022RMFS8
2019.06.27 10:01:52.756 Bond SU26214RMFS5
2019.06.27 10:01:52.756 Stock AESL
2019.06.27 10:01:52.756 Stock 123456.bin
2019.06.27 10:01:52.756 Stock ARMD
2019.06.27 10:01:52.757 Bond SU46018RMFS6
2019.06.27 10:01:52.757 Stock GAZP
2019.06.27 10:01:52.757 Metal XAUUSD
2019.06.27 10:01:52.757 Stock EURRUB_TOD
2019.06.27 10:01:52.757 Stock GBPRUB_TOM
2019.06.27 10:01:52.757 Futures Si-9.19
2019.06.27 10:01:52.757 Futures RTS-3.20
2019.06.27 10:01:52.758 Minor Forex symbol USDNOK
2019.06.27 10:01:52.758 Major Forex symbol USDJPY
2019.06.27 10:01:52.758 Major Forex symbol EURUSD
2019.06.27 10:01:52.758 Minor Forex symbol USDCZK
2019.06.27 10:01:52.758 Major Forex symbol USDCAD
2019.06.27 10:01:52.758 Minor Forex symbol USDZAR
2019.06.27 10:01:52.758 Minor Forex symbol USDSEK
2019.06.27 10:01:52.758 Major Forex symbol AUDUSD
2019.06.27 10:01:52.758 Minor Forex symbol USDDKK
2019.06.27 10:01:52.758 Major Forex symbol NZDUSD
2019.06.27 10:01:52.759 Minor Forex symbol USDPLN
2019.06.27 10:01:52.759 Major Forex symbol GBPUSD
2019.06.27 10:01:52.759 Forex symbol USDRUR
2019.06.27 10:01:52.759 Exotic Forex symbol USDMXN
2019.06.27 10:01:52.759 Forex symbol USDHUF
2019.06.27 10:01:52.759 Minor Forex symbol USDTRY
2019.06.27 10:01:52.759 Minor Forex symbol USDHKD
2019.06.27 10:01:52.760 Major Forex symbol USDCHF
2019.06.27 10:01:52.760 Minor Forex symbol USDSGD
```

**Now let's check the search for specified values in the symbol collection.**

Re-name the EA and save it in
\\MQL5\\Experts\\TestDoEasy\ under the name

**Part15\\TestDoEasyPart15\_2.mq5**.

Change the code of the fast check of the collection symbol list in the OnInit() handler. Move the list of symbols to the journal leaving only the update
of the collection symbol data and add the strings for receiving the
maximum and minimum swaps of a long and short positions, as well as the maximum and minimum spreads of the collection symbols. Enter the
obtained data to the journal:

```
//--- Set CTrade trading class parameters
#ifdef __MQL5__
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
#endif
//--- Fast check of the symbol object collection
   CArrayObj *list=engine.GetListAllUsedSymbols();
   CSymbol *symbol=NULL;
   if(list!=NULL)
     {
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         symbol.Refresh();
         symbol.RefreshRates();
        }
     }
//--- Get the minimum and maximum values
   //--- get the current account properties (we need the number of decimal places for the account currency)
   CAccount *account=engine.GetAccountCurrent();
   if(account!=NULL)
     {
      int index_min=0, index_max=0, dgc=(int)account.CurrencyDigits();
      //--- If working with the Market Watch window, leave only visible symbols in the list
      if(InpModeUsedSymbols==SYMBOLS_MODE_MARKET_WATCH)
         list=CSelect::BySymbolProperty(list,SYMBOL_PROP_VISIBLE,true,EQUAL);

      //--- min/max swap long
      index_min=CSelect::FindSymbolMin(list,SYMBOL_PROP_SWAP_LONG);  // symbol index in the collection list with the minimum swap long
      index_max=CSelect::FindSymbolMax(list,SYMBOL_PROP_SWAP_LONG);  // symbol index in the collection list with the maximum swap long
      if(index_max!=WRONG_VALUE && index_min!=WRONG_VALUE)
        {
         symbol=list.At(index_min);
         if(symbol!=NULL)
            Print("Minimum swap long for a symbol ",symbol.Name()," = ",NormalizeDouble(symbol.SwapLong(),dgc));
         symbol=list.At(index_max);
         if(symbol!=NULL)
            Print("Maximum swap long for a symbol ",symbol.Name()," = ",NormalizeDouble(symbol.SwapLong(),dgc));
        }

      //--- min/max swap short
      index_min=CSelect::FindSymbolMin(list,SYMBOL_PROP_SWAP_SHORT); // symbol index in the collection list with the minimum swap short
      index_max=CSelect::FindSymbolMax(list,SYMBOL_PROP_SWAP_SHORT); // symbol index in the collection list with the maximum swap short
      if(index_max!=WRONG_VALUE && index_min!=WRONG_VALUE)
        {
         symbol=list.At(index_min);
         if(symbol!=NULL)
            Print("Minimum swap short for a symbol ",symbol.Name()," = ",NormalizeDouble(symbol.SwapShort(),dgc));
         symbol=list.At(index_max);
         if(symbol!=NULL)
            Print("Maximum swap short for a symbol ",symbol.Name()," = ",NormalizeDouble(symbol.SwapShort(),dgc));
        }

      //--- min/max spread
      index_min=CSelect::FindSymbolMin(list,SYMBOL_PROP_SPREAD);     // symbol index in the collection list with the minimum spread
      index_max=CSelect::FindSymbolMax(list,SYMBOL_PROP_SPREAD);     // symbol index in the collection list with the maximum spread
      if(index_max!=WRONG_VALUE && index_min!=WRONG_VALUE)
        {
         symbol=list.At(index_min);
         if(symbol!=NULL)
            Print("Minimum symbol spread ",symbol.Name()," = ",symbol.Spread());
         symbol=list.At(index_max);
         if(symbol!=NULL)
            Print("Maximum symbol spread ",symbol.Name()," = ",symbol.Spread());
        }
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Compile and launch the EA on the terminal chart. Select working with the complete list of symbols on the server in the settings. After creating the
collection list of all symbols on the server (which takes some time), data on the maximum and minimum long/short swaps, as well as the maximum
and minimum spreads of all the symbols in the symbol collection list is displayed in the journal:

```
2019.06.27 10:36:28.885 Minimum long position swap for USDZAR = -192.9
2019.06.27 10:36:28.885 Maximum long position swap for USDMXN = 432.7
2019.06.27 10:36:28.886 Minimum short position swap for XAUUSD = -17.8
2019.06.27 10:36:28.886 Maximum short position swap for USDMXN = 200.0
2019.06.27 10:36:28.886 Minimum spread for SU52001RMFS3 = 0
2019.06.27 10:36:28.886 Maximum spread for GBPRUB_TOM = 3975
```

### What's next?

In the next article, we will develop the class of symbol collection events.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7041#node00)

**Previous articles within the series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part 2. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part \\
3\. Collection of market orders and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. \\
Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. \\
Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part 6. Netting account events](https://www.mql5.com/en/articles/6383)

[Part \\
7\. StopLimit order activation events, preparing the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part \\
8\. Order and position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 - \\
Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and \\
activating pending orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure \\
events](https://www.mql5.com/en/articles/6921)

[Part 12. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part 13. Account object events](https://www.mql5.com/en/articles/6995)

[Part \\
14\. Symbol object](https://www.mql5.com/en/articles/7014)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7041](https://www.mql5.com/ru/articles/7041)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7041.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7041/mql5.zip "Download MQL5.zip")(207.05 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7041/mql4.zip "Download MQL4.zip")(206.85 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/324472)**
(11)


![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
29 Oct 2019 at 21:53

**Artyom Trishkin:**

With the translation of articles? That's what MetaQuotes does.

Ok, I thought that was done by the article creator.

So an external translator is still working in the background?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
30 Oct 2019 at 04:00

**Christian :**

Ok, I thought the article creator would do that.

So an external translator is still working in the background?

The MetaQuotes team has its own full-time translators. They deal with the translation of articles.

![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
30 Oct 2019 at 17:00

**Artyom Trishkin:**

The MetaQuotes team has its own full-time translators. They deal with the translation of articles.

Good, thanks for the info

![Colin Mundia](https://c.mql5.com/avatar/2015/9/55E8E544-946A.jpg)

**[Colin Mundia](https://www.mql5.com/en/users/colyske)**
\|
5 Apr 2020 at 15:29

Hi Artyom,

Thank you for sharing your vast knowledge, there is so much to learn in your articles.

However, it seems like even when you select to work with the _**Market Watch Symbols**_, the EA still uses only the current symbol, it does not load the rest of the Market Watch Symbols.. I don't know if its only me experiencing that.. but I would love to see it work on all the Market Watch Symbols if possible.

Otherwise, congrats and thanks again for your selflessness in sharing info.

Kudos!!

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
5 Apr 2020 at 17:42

**Colin Mundia :**

Hi Artyom,

Thank you for sharing your vast knowledge, there is so much to learn in your articles.

However, it seems like even when you select to work with the _**Market Watch Symbols**_ , the EA still uses only the current symbol, it does not load the rest of the Market Watch Symbols.. I don't know if its only me experiencing that.. but I would love to see it work on all the Market Watch Symbols if possible.

Otherwise, congrats and thanks again for your selflessness in sharing info.

Kudos!!

I don’t remember how everything was done there. The fact is that these are only test advisers, and in subsequent articles everything has been done correctly and everything works.

Please see the articles in Russian - there are already 40 of them. You can read with an auto-translator.

Thank you for your feedback and appreciation.

![Parsing HTML with curl](https://c.mql5.com/2/37/logo.png)[Parsing HTML with curl](https://www.mql5.com/en/articles/7144)

The article provides the description of a simple HTML code parsing library using third-party components. In particular, it covers the possibilities of accessing data which cannot be retrieved using GET and POST requests. We will select a website with not too large pages and will try to obtain interesting data from this site.

![A New Approach to Interpreting Classic and Hidden Divergence. Part II](https://c.mql5.com/2/37/new_approach_divergence.png)[A New Approach to Interpreting Classic and Hidden Divergence. Part II](https://www.mql5.com/en/articles/5703)

The article provides a critical examination of regular divergence and efficiency of various indicators. In addition, it contains filtering options for an increased analysis accuracy and features description of non-standard solutions. As a result, we will create a new tool for solving the technical task.

![Developing Pivot Mean Oscillator: a novel Indicator for the Cumulative Moving Average](https://c.mql5.com/2/37/PMO_200x200.png)[Developing Pivot Mean Oscillator: a novel Indicator for the Cumulative Moving Average](https://www.mql5.com/en/articles/7265)

This article presents Pivot Mean Oscillator (PMO), an implementation of the cumulative moving average (CMA) as a trading indicator for the MetaTrader platforms. In particular, we first introduce Pivot Mean (PM) as a normalization index for timeseries that computes the fraction between any data point and the CMA. We then build PMO as the difference between the moving averages applied to two PM signals. Some preliminary experiments carried out on the EURUSD symbol to test the efficacy of the proposed indicator are also reported, leaving ample space for further considerations and improvements.

![Merrill patterns](https://c.mql5.com/2/36/Article_Logo__3.png)[Merrill patterns](https://www.mql5.com/en/articles/7022)

In this article, we will have a look at Merrill patterns' model and try to evaluate their current relevance. To do this, we will develop a tool to test the patterns and apply the model to various data types such as Close, High and Low prices, as well as oscillators.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/7041&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070467785315784288)

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