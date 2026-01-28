---
title: Library for easy and quick development of MetaTrader programs (part XIV): Symbol object
url: https://www.mql5.com/en/articles/7014
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:40:15.730556
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/7014&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070470744548251245)

MetaTrader 5 / Examples


### Contents

- [Symbol object](https://www.mql5.com/en/articles/7014#node01)
- [Testing symbol object](https://www.mql5.com/en/articles/7014#node02)
- [What's next?](https://www.mql5.com/en/articles/7014#node03)


A traded symbol plays an important role in trading. First of all, a program is attached to a a symbol chart (with the exception of services).
Besides, it works and performs trading and other operations based on that symbol. Therefore, it would be reasonable to create conditions
for convenient obtaining of data on necessary symbols with the ability to analyze and compare them. To achieve that, we will develop a symbol
object, a symbol object collection and symbol collection events.

### Symbol object

In this article, we will create a basic object representing a kind of an "abstract" symbol. Then, when
creating a symbol collection, we will create objects derived from the basic symbol object clarifying data on the symbol's belonging to one
or another symbol group and allowing/denying some of the basic object properties.

Let's divide symbol objects into categories:

- Forex symbol
- Major Forex symbol
- Minor Forex symbol
- Exotic Forex symbol
- Forex symbol/RUR
- Metal
- Index
- Indicative
- Cryptocurrency symbol
- Commodity symbol
- Exchange symbol
- Binary option
- Custom symbol

Of course, all these categories are arbitrary. Placing a symbol in a particular category is quite subjective and depends on user's
preferences. Some clarity can be obtained from the "symbol tree path" symbol property. It contains the symbol location path in a
certain symbol tree folder allowing you to assign a category matching a folder name to a symbol. However, each server the terminal
connects to may have its own folders for storing symbols with matching names. The folders' names may vary from server to server.



Based on what we have, the first thing we do is determine the category of a symbol from the custom list of categories, and if the
symbol is not found in the custom category, then we should have a look at its location folder in the symbol tree. If the category cannot be
defined from its location folder, the final choice of the symbol category is its "contract price calculation method" property
allowing us to define and select one of the two symbol categories — exchange or Forex.

However, all this will be done later when developing the symbol object collection. Now we need to start creating the basic symbol object.

Let's start from defining enumerations and macro substitutions for the symbol object properties. Open the **Defines.mqh**
file and add the necessary data for working with a symbol at the end of the listing:

```
//+------------------------------------------------------------------+
//| Data for working with symbols                                    |
//+------------------------------------------------------------------+
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
   SYMBOL_STATUS_FX_METAL,                                  // Metal
   SYMBOL_STATUS_INDEX,                                     // Index
   SYMBOL_STATUS_INDICATIVE,                                // Indicative
   SYMBOL_STATUS_CRYPTO,                                    // Cryptocurrency symbol
   SYMBOL_STATUS_COMMODITY,                                 // Commodity symbol
   SYMBOL_STATUS_EXCHANGE,                                  // Exchange symbol
   SYMBOL_STATUS_BIN_OPTION,                                // Binary option
   SYMBOL_STATUS_CUSTOM,                                    // Custom symbol
  };
//+------------------------------------------------------------------+
```

The symbol status is its belonging to the symbol category that will be selected before creating an object derived from the basic symbol
object. We will do this when describing the symbol collection development.

**Let's add the symbol integer properties:**

```
//+------------------------------------------------------------------+
//| Symbol integer properties                                        |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_PROP_INTEGER
  {
   SYMBOL_PROP_STATUS = 0,                                  // Symbol status
   SYMBOL_PROP_CUSTOM,                                      // Custom symbol flag
   SYMBOL_PROP_CHART_MODE,                                  // The price type used for generating bars – Bid or Last (from the ENUM_SYMBOL_CHART_MODE enumeration)
   SYMBOL_PROP_EXIST,                                       // Flag indicating that the symbol under this name exists
   SYMBOL_PROP_SELECT,                                      // An indication that the symbol is selected in Market Watch
   SYMBOL_PROP_VISIBLE,                                     // An indication that the symbol is displayed in Market Watch
   SYMBOL_PROP_SESSION_DEALS,                               // The number of deals in the current session
   SYMBOL_PROP_SESSION_BUY_ORDERS,                          // The total number of Buy orders at the moment
   SYMBOL_PROP_SESSION_SELL_ORDERS,                         // The total number of Sell orders at the moment
   SYMBOL_PROP_VOLUME,                                      // Last deal volume
   SYMBOL_PROP_VOLUMEHIGH,                                  // Maximum volume within a day
   SYMBOL_PROP_VOLUMELOW,                                   // Minimum volume within a day
   SYMBOL_PROP_TIME,                                        // Latest quote time
   SYMBOL_PROP_DIGITS,                                      // Number of decimal places
   SYMBOL_PROP_DIGITS_LOTS,                                 // Number of decimal places for a lot
   SYMBOL_PROP_SPREAD,                                      // Spread in points
   SYMBOL_PROP_SPREAD_FLOAT,                                // Floating spread flag
   SYMBOL_PROP_TICKS_BOOKDEPTH,                             // Maximum number of orders displayed in the Depth of Market
   SYMBOL_PROP_TRADE_CALC_MODE,                             // Contract price calculation method (from the ENUM_SYMBOL_CALC_MODE enumeration)
   SYMBOL_PROP_TRADE_MODE,                                  // Order execution type (from the ENUM_SYMBOL_TRADE_MODE enumeration)
   SYMBOL_PROP_START_TIME,                                  // Symbol trading start date (usually used for futures)
   SYMBOL_PROP_EXPIRATION_TIME,                             // Symbol trading end date (usually used for futures)
   SYMBOL_PROP_TRADE_STOPS_LEVEL,                           // Minimum distance in points from the current close price for setting Stop orders
   SYMBOL_PROP_TRADE_FREEZE_LEVEL,                          // Freeze distance for trading operations (in points)
   SYMBOL_PROP_TRADE_EXEMODE,                               // Deal execution mode (from the ENUM_SYMBOL_TRADE_EXECUTION enumeration)
   SYMBOL_PROP_SWAP_MODE,                                   // Swap calculation model (from the ENUM_SYMBOL_SWAP_MODE enumeration)
   SYMBOL_PROP_SWAP_ROLLOVER3DAYS,                          // Triple-day swap (from the ENUM_DAY_OF_WEEK enumeration)
   SYMBOL_PROP_MARGIN_HEDGED_USE_LEG,                       // Calculating hedging margin using the larger leg (Buy or Sell)
   SYMBOL_PROP_EXPIRATION_MODE,                             // Flags of allowed order expiration modes
   SYMBOL_PROP_FILLING_MODE,                                // Flags of allowed order filling modes
   SYMBOL_PROP_ORDER_MODE,                                  // Flags of allowed order types
   SYMBOL_PROP_ORDER_GTC_MODE,                              // StopLoss and TakeProfit orders lifetime if SYMBOL_EXPIRATION_MODE=SYMBOL_EXPIRATION_GTC (from the ENUM_SYMBOL_ORDER_GTC_MODE enumeration)
   SYMBOL_PROP_OPTION_MODE,                                 // Option type (from the ENUM_SYMBOL_OPTION_MODE enumeration)
   SYMBOL_PROP_OPTION_RIGHT,                                // Option right (Call/Put) (from the ENUM_SYMBOL_OPTION_RIGHT enumeration)
   SYMBOL_PROP_BACKGROUND_COLOR                             // The color of the background used for the symbol in Market Watch
  };
#define SYMBOL_PROP_INTEGER_TOTAL    (35)                   // Total number of integer properties
#define SYMBOL_PROP_INTEGER_SKIP     (1)                    // Number of symbol integer properties not used in sorting
//+------------------------------------------------------------------+
```

We have already considered arranging enumerations of object properties in the " [Implementing \\
event handling on a netting account](https://www.mql5.com/en/articles/6383#node02)" section of the sixth part of the library description, so we will not dwell on defining the purpose of
macro substitutions of the number of properties and the number of object properties not used in search and sorting.
However, I should note that, apart from standard symbol object integer properties from the

[ENUM\_SYMBOL\_INFO\_INTEGER](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer)
enumeration, two more have been added:

symbol status and number
of decimal places in the symbol lot value.

**Let's add the enumeration of symbol real properties:**

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
   SYMBOL_PROP_VOLUMEHIGH_REAL,                             // Maximum Volume of the day
   SYMBOL_PROP_VOLUMELOW_REAL,                              // Minimum Volume of the day
   SYMBOL_PROP_OPTION_STRIKE,                               // Option execution price
   SYMBOL_PROP_POINT,                                       // Point value
   SYMBOL_PROP_TRADE_TICK_VALUE,                            // SYMBOL_TRADE_TICK_VALUE_PROFIT value
   SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT,                     // Calculated tick value for a winning position
   SYMBOL_PROP_TRADE_TICK_VALUE_LOSS,                       // Calculated tick value for a losing position
   SYMBOL_PROP_TRADE_TICK_SIZE,                             // Minimum price change
   SYMBOL_PROP_TRADE_CONTRACT_SIZE,                         // Trade contract size
   SYMBOL_PROP_TRADE_ACCRUED_INTEREST,                      // Accrued interest
   SYMBOL_PROP_TRADE_FACE_VALUE,                            // Face value – initial bond price set by an issuer
   SYMBOL_PROP_TRADE_LIQUIDITY_RATE,                        // Liquidity rate – share of an asset value that can be used as a collateral
   SYMBOL_PROP_VOLUME_MIN,                                  // Minimum volume for a deal
   SYMBOL_PROP_VOLUME_MAX,                                  // Maximum volume for a deal
   SYMBOL_PROP_VOLUME_STEP,                                 // Minimum volume change step for a deal
   SYMBOL_PROP_VOLUME_LIMIT,                                // Maximum acceptable total volume of an open position and pending orders in one direction (buy or sell)
   SYMBOL_PROP_SWAP_LONG,                                   // Long swap value
   SYMBOL_PROP_SWAP_SHORT,                                  // Short swap value
   SYMBOL_PROP_MARGIN_INITIAL,                              // Initial margin
   SYMBOL_PROP_MARGIN_MAINTENANCE,                          // Maintenance margin for an instrument
   SYMBOL_PROP_MARGIN_LONG,                                 // Margin requirement applicable to long positions
   SYMBOL_PROP_MARGIN_SHORT,                                // Margin requirement applicable to short positions
   SYMBOL_PROP_MARGIN_STOP,                                 // Margin requirement applicable to Stop orders
   SYMBOL_PROP_MARGIN_LIMIT,                                // Margin requirement applicable to Limit orders
   SYMBOL_PROP_MARGIN_STOPLIMIT,                            // Margin requirement applicable to Stop Limit orders
   SYMBOL_PROP_SESSION_VOLUME,                              // The total volume of deals in the current session
   SYMBOL_PROP_SESSION_TURNOVER,                            // The total turnover in the current session
   SYMBOL_PROP_SESSION_INTEREST,                            // The total volume of open positions
   SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME,                   // The total volume of Buy orders at the moment
   SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME,                  // The total volume of Sell orders at the moment
   SYMBOL_PROP_SESSION_OPEN,                                // The open price of the session
   SYMBOL_PROP_SESSION_CLOSE,                               // The close price of the session
   SYMBOL_PROP_SESSION_AW,                                  // The average weighted price of the session
   SYMBOL_PROP_SESSION_PRICE_SETTLEMENT,                    // The settlement price of the current session
   SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN,                     // The minimum allowable price value for the session
   SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX,                     // The maximum allowable price value for the session
   SYMBOL_PROP_MARGIN_HEDGED                                // Size of a contract or margin for one lot of hedged positions (oppositely directed positions at one symbol).
  };
#define SYMBOL_PROP_DOUBLE_TOTAL     (47)                   // Total number of event's real properties
#define SYMBOL_PROP_DOUBLE_SKIP      (0)                    // Number of symbol real properties not used in sorting
//+------------------------------------------------------------------+
```

Apart from the properties described in the [ENUM\_SYMBOL\_INFO\_DOUBLE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double)
enumeration, we added five new symbol properties that for some
reason were not included into the description of the real properties enumeration but they are still present and belong to the enumeration.

**Let's add the enumeration of symbol's string properties**
**and**

**possible symbol sorting criteria:**

```
//+------------------------------------------------------------------+
//| Symbol string properties                                         |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_PROP_STRING
  {
   SYMBOL_PROP_NAME = (SYMBOL_PROP_INTEGER_TOTAL+SYMBOL_PROP_DOUBLE_TOTAL),   // Symbol name
   SYMBOL_PROP_BASIS,                                       // The name of the underlaying asset for a derivative symbol
   SYMBOL_PROP_CURRENCY_BASE,                               // The base currency of an instrument
   SYMBOL_PROP_CURRENCY_PROFIT,                             // Profit currency
   SYMBOL_PROP_CURRENCY_MARGIN,                             // Margin currency
   SYMBOL_PROP_BANK,                                        // The source of the current quote
   SYMBOL_PROP_DESCRIPTION,                                 // The string description of a symbol
   SYMBOL_PROP_FORMULA,                                     // The formula used for custom symbol pricing
   SYMBOL_PROP_ISIN,                                        // The name of a trading symbol in the international system of securities identification numbers (ISIN)
   SYMBOL_PROP_PAGE,                                        // The address of the web page containing symbol information
   SYMBOL_PROP_PATH,                                        // Path in the symbol tree
  };
#define SYMBOL_PROP_STRING_TOTAL     (11)                   // Total number of string properties
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
   SORT_BY_SYMBOL_VOLUME_STEP,                              // Sort by a minimal volume change step for deal execution
   SORT_BY_SYMBOL_VOLUME_LIMIT,                             // Sort by a maximum allowed aggregate volume of an open position and pending orders in one direction
   SORT_BY_SYMBOL_SWAP_LONG,                                // Sort by a long swap value
   SORT_BY_SYMBOL_SWAP_SHORT,                               // Sort by a short swap value
   SORT_BY_SYMBOL_MARGIN_INITIAL,                           // Sort by an initial margin
   SORT_BY_SYMBOL_MARGIN_MAINTENANCE,                       // Sort by a maintenance margin for an instrument
   SORT_BY_SYMBOL_MARGIN_LONG,                              // Sort by coefficient of margin charging for long positions
   SORT_BY_SYMBOL_MARGIN_SHORT,                             // Sort by coefficient of margin charging for short positions
   SORT_BY_SYMBOL_MARGIN_STOP,                              // Sort by coefficient of margin charging for Stop orders
   SORT_BY_SYMBOL_MARGIN_LIMIT,                             // Sort by coefficient of margin charging for Limit orders
   SORT_BY_SYMBOL_MARGIN_STOPLIMIT,                         // Sort by coefficient of margin charging for Stop Limit orders
   SORT_BY_SYMBOL_SESSION_VOLUME,                           // Sort by summary volume of the current session deals
   SORT_BY_SYMBOL_SESSION_TURNOVER,                         // Sort by the summary turnover of the current session
   SORT_BY_SYMBOL_SESSION_INTEREST,                         // Sort by the summary open interest
   SORT_BY_SYMBOL_SESSION_BUY_ORDERS_VOLUME,                // Sort by the current volume of Buy orders
   SORT_BY_SYMBOL_SESSION_SELL_ORDERS_VOLUME,               // Sort by the current volume of Sell orders
   SORT_BY_SYMBOL_SESSION_OPEN,                             // Sort by an Open price of the current session
   SORT_BY_SYMBOL_SESSION_CLOSE,                            // Sort by a Close price of the current session
   SORT_BY_SYMBOL_SESSION_AW,                               // Sort by an average weighted price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_SETTLEMENT,                 // Sort by a settlement price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_LIMIT_MIN,                  // Sort by a minimal price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_LIMIT_MAX,                  // Sort by a maximal price of the current session
   SORT_BY_SYMBOL_MARGIN_HEDGED,                            // Sort by a contract size or a margin value per one lot of hedged positions
//--- Sort by string properties
   SORT_BY_SYMBOL_NAME = FIRST_SYM_STR_PROP,                // Sort by a symbol name
   SORT_BY_SYMBOL_BASIS,                                    // Sort by an underlying asset of a derivative
   SORT_BY_SYMBOL_CURRENCY_BASE,                            // Sort by a basic currency of a symbol
   SORT_BY_SYMBOL_CURRENCY_PROFIT,                          // Sort by a profit currency
   SORT_BY_SYMBOL_CURRENCY_MARGIN,                          // Sort by a margin currency
   SORT_BY_SYMBOL_BANK,                                     // Sort by a feeder of the current quote
   SORT_BY_SYMBOL_DESCRIPTION,                              // Sort by symbol string description
   SORT_BY_SYMBOL_FORMULA,                                  // Sort by the formula used for custom symbol pricing
   SORT_BY_SYMBOL_ISIN,                                     // Sort by the name of a symbol in the ISIN system
   SORT_BY_SYMBOL_PAGE,                                     // Sort by an address of the web page containing symbol information
   SORT_BY_SYMBOL_PATH                                      // Sort by a path in the symbol tree
  };
//+------------------------------------------------------------------+
```

The enumeration of string properties now features the appropriate constants from the [ENUM\_SYMBOL\_INFO\_STRING](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_string)
enumeration, while the list of possible sorting criteria contains all symbol properties excluding symbol background color in the Market Watch window.
There is no reason to sort and compare symbols by their background color in the window belonging to the terminal.

This is all the necessary data for working with symbol objects.

**Now let's develop the symbol object class.**

In \\MQL5\\Include\\DoEasy\ **Objects**\\, create the new file of the **CCymbol** class named **Symbol.mqh**. Fill
the class with inclusions and methods that are standard for the library right away:

```
//+------------------------------------------------------------------+
//|                                                       Symbol.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "..\..\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| Abstract symbol class                                            |
//+------------------------------------------------------------------+
class CSymbol : public CObject
  {
private:
   long              m_long_prop[SYMBOL_PROP_INTEGER_TOTAL];         // Integer properties
   double            m_double_prop[SYMBOL_PROP_DOUBLE_TOTAL];        // Real properties
   string            m_string_prop[SYMBOL_PROP_STRING_TOTAL];        // String properties
//--- Return the index of the array the symbol's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_SYMBOL_PROP_DOUBLE property)  const { return(int)property-SYMBOL_PROP_INTEGER_TOTAL;                                    }
   int               IndexProp(ENUM_SYMBOL_PROP_STRING property)  const { return(int)property-SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_DOUBLE_TOTAL;           }
public:
//--- Default constructor
                     CSymbol(void){;}
protected:
//--- Protected parametric constructor
                     CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name);

public:
//--- Set (1) integer, (2) real and (3) string symbol properties
   void              SetProperty(ENUM_SYMBOL_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                                        }
   void              SetProperty(ENUM_SYMBOL_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;                      }
   void              SetProperty(ENUM_SYMBOL_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;                      }
//--- Return (1) integer, (2) real and (3) string symbol properties from the properties array
   long              GetProperty(ENUM_SYMBOL_PROP_INTEGER property)        const { return this.m_long_prop[property];                                       }
   double            GetProperty(ENUM_SYMBOL_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];                     }
   string            GetProperty(ENUM_SYMBOL_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];                     }

//--- Return the flag of a symbol supporting the property
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_INTEGER property)    { return true; }
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_DOUBLE property)     { return true; }
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_STRING property)     { return true; }

//+------------------------------------------------------------------+
//| Description of symbol object properties                          |
//+------------------------------------------------------------------+
//--- Get description of a symbol (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_SYMBOL_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_SYMBOL_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_SYMBOL_PROP_STRING property);
//--- Return symbol status name
   string            StatusDescription(void)    const;
//--- Send description of symbol properties to the journal (full_prop=true - all properties, false - only supported ones)
   void              Print(const bool full_prop=false);

//--- Compare CSymbol objects by all possible properties (for sorting lists by a specified symbol object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CSymbol objects by all properties (for searching for equal event objects)
   bool              IsEqual(CSymbol* compared_symbol) const;

  };
```

We have already considered all these methods in the [first part of the \\
library description](https://www.mql5.com/en/articles/5654#node04). There is no point in dwelling on them here.

Now let's **add the variables necessary for the class operation to the private section:**

```
//+------------------------------------------------------------------+
//|                                                       Symbol.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "..\..\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| Abstract symbol class                                            |
//+------------------------------------------------------------------+
class CSymbol : public CObject
  {
private:
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
//--- Reset all symbol object data
   void              Reset(void);
public:
//--- Default constructor
                     CSymbol(void){;}
protected:
//--- Protected parametric constructor
                     CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name);
```

We are going to obtain Ask, Bid, Last price and tick time data from the tick
structure — in milliseconds for MQL5 and in seconds for MQL4. Although the millisecond field is present in the tick structure in
MQL4, it is not used. Therefore, for MQL4, we will use second time \* 1000 for converting it to the millisecond format.

We will need the array of market depth data structures
later when obtaining the market depth content (not in this article).

A global error code — there may be cases when a library-based
program cannot operate further due to an error execution of a method or a function. The program should be informed of the error execution of a
method or a function so that it is able to correctly handle the situation in a timely manner. These are the cases we introduce the variable for.
It is to contain an error code, while the CEngine library base object surveys the error code. If the code contains a non-zero value, it is first
handled in the CEngine class. In case it is impossible to "solve the issue" by internal means, the code is sent to a calling program for a timely
response to the error.

Before creating a symbol object, all its fields and structures should be reset.
This is what the Reset() method is used for.

In the protected class constructor, fill in all symbol properties using the standard functions. However, not all functions are suitable
for MQL4, therefore, let's

**create the necessary methods of obtaining data in the protected class section** for the cases where the separation of data
obtained for MQL5 and MQL4 should be introduced:

```
protected:
//--- Protected parametric constructor
                     CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name);

//--- Get and return integer properties of a selected symbol from its parameters
   long              SymbolExists(void)                  const;
   long              SymbolCustom(void)                  const;
   long              SymbolChartMode(void)               const;
   long              SymbolMarginHedgedUseLEG(void)      const;
   long              SymbolOrderFillingMode(void)        const;
   long              SymbolOrderMode(void)               const;
   long              SymbolOrderGTCMode(void)            const;
   long              SymbolOptionMode(void)              const;
   long              SymbolOptionRight(void)             const;
   long              SymbolBackgroundColor(void)         const;
   long              SymbolCalcMode(void)                const;
   long              SymbolSwapMode(void)                const;
   long              SymbolExpirationMode(void)          const;
   long              SymbolDigitsLot(void);
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
//--- Get and return string properties of a selected symbol from its parameters
   string            SymbolBasis(void)                   const;
   string            SymbolBank(void)                    const;
   string            SymbolISIN(void)                    const;
   string            SymbolFormula(void)                 const;
   string            SymbolPage(void)                    const;
//--- Return the number of decimal places of the account currency
   int               DigitsCurrency(void)                const { return this.m_digits_currency; }
//--- Search for a symbol and return the flag indicating its presence on the server
   bool              Exist(void)                         const;

public:
```

**In the public class section, declare the methods returning the states of**
**various flags describing allowed modes of some symbol properties, as well as string descriptions of these modes:**

```
public:
//--- Set (1) integer, (2) real and (3) string symbol properties
   void              SetProperty(ENUM_SYMBOL_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                                        }
   void              SetProperty(ENUM_SYMBOL_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;                      }
   void              SetProperty(ENUM_SYMBOL_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;                      }
//--- Return (1) integer, (2) real and (3) string symbol properties from the properties array
   long              GetProperty(ENUM_SYMBOL_PROP_INTEGER property)        const { return this.m_long_prop[property];                                       }
   double            GetProperty(ENUM_SYMBOL_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];                     }
   string            GetProperty(ENUM_SYMBOL_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];                     }

//--- Return the flag of a symbol supporting the property
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_INTEGER property)    { return true; }
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_DOUBLE property)     { return true; }
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_STRING property)     { return true; }

//--- Return the flag of allowing (1) market, (2) limit, (3) stop (4) and stop limit orders,
//--- the flag of allowing setting (5) StopLoss and (6) TakeProfit orders, (7) as well as closing by an opposite order
   bool              IsMarketOrdersAllowed(void)            const { return((this.OrderModeFlags() & SYMBOL_ORDER_MARKET)==SYMBOL_ORDER_MARKET);             }
   bool              IsLimitOrdersAllowed(void)             const { return((this.OrderModeFlags() & SYMBOL_ORDER_LIMIT)==SYMBOL_ORDER_LIMIT);               }
   bool              IsStopOrdersAllowed(void)              const { return((this.OrderModeFlags() & SYMBOL_ORDER_STOP)==SYMBOL_ORDER_STOP);                 }
   bool              IsStopLimitOrdersAllowed(void)         const { return((this.OrderModeFlags() & SYMBOL_ORDER_STOP_LIMIT)==SYMBOL_ORDER_STOP_LIMIT);     }
   bool              IsStopLossOrdersAllowed(void)          const { return((this.OrderModeFlags() & SYMBOL_ORDER_SL)==SYMBOL_ORDER_SL);                     }
   bool              IsTakeProfitOrdersAllowed(void)        const { return((this.OrderModeFlags() & SYMBOL_ORDER_TP)==SYMBOL_ORDER_TP);                     }
   bool              IsCloseByOrdersAllowed(void)           const;

//--- Return the (1) FOK and (2) IOC filling flag
   bool              IsFillingModeFOK(void)                 const { return((this.FillingModeFlags() & SYMBOL_FILLING_FOK)==SYMBOL_FILLING_FOK);             }
   bool              IsFillingModeIOC(void)                 const { return((this.FillingModeFlags() & SYMBOL_FILLING_IOC)==SYMBOL_FILLING_IOC);             }

//--- Return the flag of order expiration: (1) GTC, (2) DAY, (3) Specified and (4) Specified Day
   bool              IsExipirationModeGTC(void)             const { return((this.ExpirationModeFlags() & SYMBOL_EXPIRATION_GTC)==SYMBOL_EXPIRATION_GTC);    }
   bool              IsExipirationModeDAY(void)             const { return((this.ExpirationModeFlags() & SYMBOL_EXPIRATION_DAY)==SYMBOL_EXPIRATION_DAY);    }
   bool              IsExipirationModeSpecified(void)       const { return((this.ExpirationModeFlags() & SYMBOL_EXPIRATION_SPECIFIED)==SYMBOL_EXPIRATION_SPECIFIED);          }
   bool              IsExipirationModeSpecifiedDay(void)    const { return((this.ExpirationModeFlags() & SYMBOL_EXPIRATION_SPECIFIED_DAY)==SYMBOL_EXPIRATION_SPECIFIED_DAY);  }

//--- Return the description of allowing (1) market, (2) limit, (3) stop and (4) stop limit orders,
//--- the description of allowing (5) StopLoss and (6) TakeProfit orders, (7) as well as closing by an opposite order
   string            GetMarketOrdersAllowedDescription(void)      const;
   string            GetLimitOrdersAllowedDescription(void)       const;
   string            GetStopOrdersAllowedDescription(void)        const;
   string            GetStopLimitOrdersAllowedDescription(void)   const;
   string            GetStopLossOrdersAllowedDescription(void)    const;
   string            GetTakeProfitOrdersAllowedDescription(void)  const;
   string            GetCloseByOrdersAllowedDescription(void)     const;

//--- Return the description of allowing the filling type (1) FOK and (2) IOC, (3) as well as allowed order expiration modes
   string            GetFillingModeFOKAllowedDescrioption(void)   const;
   string            GetFillingModeIOCAllowedDescrioption(void)   const;

//--- Return the description of order expiration: (1) GTC, (2) DAY, (3) Specified and (4) Specified Day
   string            GetExpirationModeGTCDescription(void)        const;
   string            GetExpirationModeDAYDescription(void)        const;
   string            GetExpirationModeSpecifiedDescription(void)  const;
   string            GetExpirationModeSpecDayDescription(void)    const;

//--- Return the description of the (1) status, (2) price type for constructing bars,
//--- (3) method of calculating margin, (4) instrument trading mode,
//--- (5) deal execution mode for a symbol, (6) swap calculation mode,
//--- (7) StopLoss and TakeProfit lifetime, (8) option type, (9) option rights
//--- flags of (10) allowed order types, (11) allowed filling types,
//--- (12) allowed order expiration modes
   string            GetStatusDescription(void)                   const;
   string            GetChartModeDescription(void)                const;
   string            GetCalcModeDescription(void)                 const;
   string            GetTradeModeDescription(void)                const;
   string            GetTradeExecDescription(void)                const;
   string            GetSwapModeDescription(void)                 const;
   string            GetOrderGTCModeDescription(void)             const;
   string            GetOptionTypeDescription(void)               const;
   string            GetOptionRightDescription(void)              const;
   string            GetOrderModeFlagsDescription(void)           const;
   string            GetFillingModeFlagsDescription(void)         const;
   string            GetExpirationModeFlagsDescription(void)      const;

```

Also, **add the following elements to the public section of the class:**

the
error code obtaining method, the method of obtaining all data by a symbol,

the method of updating quote data by a symbol, as well as
additional methods for

adding/removing a
symbol from the Market Watch window, the method returning the flag of data
synchronization by a symbol and the methods for subscribing to the
market depth and unsubscribing from it.

We will arrange the work with the market depth in subsequent articles.

```
//--- Return the global error code
   int               GetError(void)                               const { return this.m_global_error;                                                       }
//--- Update all symbol data that can change
   void              Refresh(void);
//--- Update quote data by a symbol
   void              RefreshRates(void);

//--- (1) Add, (2) remove a symbol from the Market Watch window, (3) return the data synchronization flag by a symbol
   bool              SetToMarketWatch(void)                       const { return ::SymbolSelect(this.m_symbol_name,true);                                   }
   bool              RemoveFromMarketWatch(void)                  const { return ::SymbolSelect(this.m_symbol_name,false);                                  }
   bool              IsSynchronized(void)                         const { return ::SymbolIsSynchronized(this.m_symbol_name);                                }
//--- (1) Arrange a (1) subscription to the market depth, (2) close the market depth
   bool              BookAdd(void)                                const;
   bool              BookClose(void)                              const;
```

Since we have methods returning any property by its name (enumeration constant), we already can obtain data on any of the external properties
but this is not practical from the programming perspective as we need to remember all constant names from the enumerations of the symbol
object properties. Therefore (as in the previous classes and for the same reason), we will introduce additional public methods that return
all the properties of the symbol object, but with more informative names.

**Add them at the very end of the class body:**

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
   bool              IsSelect(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_SELECT);                                }
   bool              IsVisible(void)                              const { return (bool)this.GetProperty(SYMBOL_PROP_VISIBLE);                               }
   long              SessionDeals(void)                           const { return this.GetProperty(SYMBOL_PROP_SESSION_DEALS);                               }
   long              SessionBuyOrders(void)                       const { return this.GetProperty(SYMBOL_PROP_SESSION_BUY_ORDERS);                          }
   long              SessionSellOrders(void)                      const { return this.GetProperty(SYMBOL_PROP_SESSION_SELL_ORDERS);                         }
   long              Volume(void)                                 const { return this.GetProperty(SYMBOL_PROP_VOLUME);                                      }
   long              VolumeHigh(void)                             const { return this.GetProperty(SYMBOL_PROP_VOLUMEHIGH);                                  }
   long              VolumeLow(void)                              const { return this.GetProperty(SYMBOL_PROP_VOLUMELOW);                                   }
   datetime          Time(void)                                   const { return (datetime)this.GetProperty(SYMBOL_PROP_TIME);                              }
   int               Digits(void)                                 const { return (int)this.GetProperty(SYMBOL_PROP_DIGITS);                                 }
   int               DigitsLot(void)                              const { return (int)this.GetProperty(SYMBOL_PROP_DIGITS_LOTS);                            }
   int               Spread(void)                                 const { return (int)this.GetProperty(SYMBOL_PROP_SPREAD);                                 }
   bool              IsSpreadFloat(void)                          const { return (bool)this.GetProperty(SYMBOL_PROP_SPREAD_FLOAT);                          }
   int               TicksBookdepth(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_TICKS_BOOKDEPTH);                        }
   ENUM_SYMBOL_CALC_MODE TradeCalcMode(void)                      const { return (ENUM_SYMBOL_CALC_MODE)this.GetProperty(SYMBOL_PROP_TRADE_CALC_MODE);      }
   ENUM_SYMBOL_TRADE_MODE TradeMode(void)                         const { return (ENUM_SYMBOL_TRADE_MODE)this.GetProperty(SYMBOL_PROP_TRADE_MODE);          }
   datetime          StartTime(void)                              const { return (datetime)this.GetProperty(SYMBOL_PROP_START_TIME);                        }
   datetime          ExpirationTime(void)                         const { return (datetime)this.GetProperty(SYMBOL_PROP_EXPIRATION_TIME);                   }
   int               TradeStopLevel(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_TRADE_STOPS_LEVEL);                      }
   int               TradeFreezeLevel(void)                       const { return (int)this.GetProperty(SYMBOL_PROP_TRADE_FREEZE_LEVEL);                     }
   ENUM_SYMBOL_TRADE_EXECUTION TradeExecutionMode(void)           const { return (ENUM_SYMBOL_TRADE_EXECUTION)this.GetProperty(SYMBOL_PROP_TRADE_EXEMODE);  }
   ENUM_SYMBOL_SWAP_MODE SwapMode(void)                           const { return (ENUM_SYMBOL_SWAP_MODE)this.GetProperty(SYMBOL_PROP_SWAP_MODE);            }
   ENUM_DAY_OF_WEEK  SwapRollover3Days(void)                      const { return (ENUM_DAY_OF_WEEK)this.GetProperty(SYMBOL_PROP_SWAP_ROLLOVER3DAYS);        }
   bool              IsMarginHedgedUseLeg(void)                   const { return (bool)this.GetProperty(SYMBOL_PROP_MARGIN_HEDGED_USE_LEG);                 }
   int               ExpirationModeFlags(void)                    const { return (int)this.GetProperty(SYMBOL_PROP_EXPIRATION_MODE);                        }
   int               FillingModeFlags(void)                       const { return (int)this.GetProperty(SYMBOL_PROP_FILLING_MODE);                           }
   int               OrderModeFlags(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_ORDER_MODE);                             }
   ENUM_SYMBOL_ORDER_GTC_MODE OrderModeGTC(void)                  const { return (ENUM_SYMBOL_ORDER_GTC_MODE)this.GetProperty(SYMBOL_PROP_ORDER_GTC_MODE);  }
   ENUM_SYMBOL_OPTION_MODE OptionMode(void)                       const { return (ENUM_SYMBOL_OPTION_MODE)this.GetProperty(SYMBOL_PROP_OPTION_MODE);        }
   ENUM_SYMBOL_OPTION_RIGHT OptionRight(void)                     const { return (ENUM_SYMBOL_OPTION_RIGHT)this.GetProperty(SYMBOL_PROP_OPTION_RIGHT);      }
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
   double            MarginLong(void)                             const { return this.GetProperty(SYMBOL_PROP_MARGIN_LONG);                                 }
   double            MarginShort(void)                            const { return this.GetProperty(SYMBOL_PROP_MARGIN_SHORT);                                }
   double            MarginStop(void)                             const { return this.GetProperty(SYMBOL_PROP_MARGIN_STOP);                                 }
   double            MarginLimit(void)                            const { return this.GetProperty(SYMBOL_PROP_MARGIN_LIMIT);                                }
   double            MarginStopLimit(void)                        const { return this.GetProperty(SYMBOL_PROP_MARGIN_STOPLIMIT);                            }
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
   string            Name(void)                                   const { return this.GetProperty(SYMBOL_PROP_NAME);                                        }
   string            Basis(void)                                  const { return this.GetProperty(SYMBOL_PROP_BASIS);                                       }
   string            CurrencyBase(void)                           const { return this.GetProperty(SYMBOL_PROP_CURRENCY_BASE);                               }
   string            CurrencyProfit(void)                         const { return this.GetProperty(SYMBOL_PROP_CURRENCY_PROFIT);                             }
   string            CurrencyMargin(void)                         const { return this.GetProperty(SYMBOL_PROP_CURRENCY_MARGIN);                             }
   string            Bank(void)                                   const { return this.GetProperty(SYMBOL_PROP_BANK);                                        }
   string            Description(void)                            const { return this.GetProperty(SYMBOL_PROP_DESCRIPTION);                                 }
   string            Formula(void)                                const { return this.GetProperty(SYMBOL_PROP_FORMULA);                                     }
   string            ISIN(void)                                   const { return this.GetProperty(SYMBOL_PROP_ISIN);                                        }
   string            Page(void)                                   const { return this.GetProperty(SYMBOL_PROP_PAGE);                                        }
   string            Path(void)                                   const { return this.GetProperty(SYMBOL_PROP_PATH);                                        }
//---
  };
//+------------------------------------------------------------------+
```

The methods are declared, and some of them are implemented in the class body right away. Now let's add and analyze implementation of all
declared methods.

**Implementing the protected class constructor:**

```
//+------------------------------------------------------------------+
//| Class methods                                                    |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CSymbol::CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name) : m_global_error(ERR_SUCCESS)
  {
   this.m_symbol_name=name;
   if(!this.Exist())
     {
      ::Print(DFUN_ERR_LINE,"\"",this.m_symbol_name,"\"",": ",TextByLanguage("Ошибка. Такого символа нет на сервере","Error. No such symbol on server"));
      this.m_global_error=ERR_MARKET_UNKNOWN_SYMBOL;
     }
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_symbol_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,"\"",this.m_symbol_name,"\": ",TextByLanguage("Не удалось получить текущие цены. Ошибка: ","Could not get current prices. Error: "),this.m_global_error);
     }
//--- Data initialization
   ::ZeroMemory(this.m_tick);
   this.Reset();
   this.m_digits_currency=(#ifdef __MQL5__ (int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif);
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
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_MODE]                              = ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_EXPIRATION_MODE);
   this.m_long_prop[SYMBOL_PROP_EXIST]                                        = this.SymbolExists();
   this.m_long_prop[SYMBOL_PROP_CUSTOM]                                       = this.SymbolCustom();
   this.m_long_prop[SYMBOL_PROP_MARGIN_HEDGED_USE_LEG]                        = this.SymbolMarginHedgedUseLEG();
   this.m_long_prop[SYMBOL_PROP_ORDER_MODE]                                   = this.SymbolOrderMode();
   this.m_long_prop[SYMBOL_PROP_FILLING_MODE]                                 = this.SymbolOrderFillingMode();
   this.m_long_prop[SYMBOL_PROP_ORDER_GTC_MODE]                               = this.SymbolOrderGTCMode();
   this.m_long_prop[SYMBOL_PROP_OPTION_MODE]                                  = this.SymbolOptionMode();
   this.m_long_prop[SYMBOL_PROP_OPTION_RIGHT]                                 = this.SymbolOptionRight();
   this.m_long_prop[SYMBOL_PROP_BACKGROUND_COLOR]                             = this.SymbolBackgroundColor();
   this.m_long_prop[SYMBOL_PROP_CHART_MODE]                                   = this.SymbolChartMode();
   this.m_long_prop[SYMBOL_PROP_TRADE_CALC_MODE]                              = this.SymbolCalcMode();
   this.m_long_prop[SYMBOL_PROP_SWAP_MODE]                                    = this.SymbolSwapMode();
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_MODE]                              = this.SymbolExpirationMode();
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
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG)]                = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_LONG);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT)]               = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_SHORT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_STOP)]                = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_STOP);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LIMIT)]               = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_LIMIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_STOPLIMIT)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_STOPLIMIT);
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
  }
//+------------------------------------------------------------------+
```

The constructor inputs are symbol status and its
name. When creating the symbol collection, we move along all the necessary symbols in a loop defining whether they belong to a
previously set category (in the ENUM\_SYMBOL\_STATUS enumeration) and create a new object derived from the abstract symbol class. Only the
symbol name is passed to the derived class, while the status is set automatically based on the derived object type and sent to the parent class
constructor when creating the derived object. We have already analyzed that when discussing creation of order objects

[in the second part](https://www.mql5.com/en/articles/5669#node01) of the library description.

Initialize
the error code in the constructor initialization list right away.

In the method body, first check
if such a symbol is present on the server. If not, send the error message to the journal and add the "Unknown symbol" value to the error
code. This check is, in fact, unnecessary, since data on a symbol selected from the list is passed to the object when creating it. But I think,
it should be present since an incorrect symbol may be passed to the newly created object.

Next, we obtain data on the last tick. If none is received,
send the appropriate message to the journal and assign the value of the last error to the error code using

[GetLastError()](https://www.mql5.com/en/docs/check/getlasterror). The object is created in any case, while the error code
allows making a decision in a calling program on whether it should be left intact or removed.

Next, all symbol object data is initialized (reset) and the number
of decimal places for the account currency is set for the correct output of values to the journal.

All object
properties are filled using SymbolInfo functions. In cases where there are differences in MQL5 and MQL4 for receiving these values, the
data is filled with specially created methods that will be described later.

**Below is the method for comparing two symbol objects for searching and sorting:**

```
//+------------------------------------------------------------------+
//|Compare CSymbol objects by all possible properties                |
//+------------------------------------------------------------------+
int CSymbol::Compare(const CObject *node,const int mode=0) const
  {
   const CSymbol *symbol_compared=node;
//--- compare integer properties of two symbols
   if(mode<SYMBOL_PROP_INTEGER_TOTAL)
     {
      long value_compared=symbol_compared.GetProperty((ENUM_SYMBOL_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_SYMBOL_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two symbols
   else if(mode<SYMBOL_PROP_INTEGER_TOTAL+SYMBOL_PROP_DOUBLE_TOTAL)
     {
      double value_compared=symbol_compared.GetProperty((ENUM_SYMBOL_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_SYMBOL_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two symbols
   else if(mode<SYMBOL_PROP_INTEGER_TOTAL+SYMBOL_PROP_DOUBLE_TOTAL+SYMBOL_PROP_STRING_TOTAL)
     {
      string value_compared=symbol_compared.GetProperty((ENUM_SYMBOL_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_SYMBOL_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

**Below is the method for comparing two symbol objects to define whether they are equal to each other:**

```
//+------------------------------------------------------------------+
//| Compare CSymbol objects by all properties                        |
//+------------------------------------------------------------------+
bool CSymbol::IsEqual(CSymbol *compared_symbol) const
  {
   int beg=0, end=SYMBOL_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_INTEGER prop=(ENUM_SYMBOL_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_symbol.GetProperty(prop)) return false;
     }
   beg=end; end+=SYMBOL_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_DOUBLE prop=(ENUM_SYMBOL_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_symbol.GetProperty(prop)) return false;
     }
   beg=end; end+=SYMBOL_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_STRING prop=(ENUM_SYMBOL_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_symbol.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

I have already described both methods in the [first](https://www.mql5.com/en/articles/5654#node04) and [fifth](https://www.mql5.com/en/articles/6211#node01)
parts of the library description.

**Below is the method for initializing all symbol object properties:**

```
//+------------------------------------------------------------------+
//| Reset all symbol object data                                     |
//+------------------------------------------------------------------+
void CSymbol::Reset(void)
  {
   int beg=0, end=SYMBOL_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_INTEGER prop=(ENUM_SYMBOL_PROP_INTEGER)i;
      this.SetProperty(prop,0);
     }
   beg=end; end+=SYMBOL_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_DOUBLE prop=(ENUM_SYMBOL_PROP_DOUBLE)i;
      this.SetProperty(prop,0);
     }
   beg=end; end+=SYMBOL_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_STRING prop=(ENUM_SYMBOL_PROP_STRING)i;
      this.SetProperty(prop,NULL);
     }
  }
//+------------------------------------------------------------------+
```

The method works identically to the previous ones, but instead of comparing to the specimen object, all class fields are reset here in a loop
for each of the three property sets.

**Below are the methods for receiving symbol's integer properties that are either completely or partially absent in MQL4:**

```
//+----------------------------------------------------------------------+
//| Integer properties                                                   |
//+----------------------------------------------------------------------+
//+----------------------------------------------------------------------+
//| Return the symbol existence flag                                     |
//+----------------------------------------------------------------------+
long CSymbol::SymbolExists(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_EXIST) #else this.Exist() #endif);
  }
//+----------------------------------------------------------------------+
//| Return the custom symbol flag                                        |
//+----------------------------------------------------------------------+
long CSymbol::SymbolCustom(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_CUSTOM) #else false #endif);
  }
//+----------------------------------------------------------------------+
//| Return the price type for building bars - Bid or Last                |
//+----------------------------------------------------------------------+
long CSymbol::SymbolChartMode(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_CHART_MODE) #else SYMBOL_CHART_MODE_BID #endif);
  }
//+----------------------------------------------------------------------+
//|Return the calculation mode of a hedging margin using the larger leg  |
//+----------------------------------------------------------------------+
long CSymbol::SymbolMarginHedgedUseLEG(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_MARGIN_HEDGED_USE_LEG) #else false #endif);
  }
//+----------------------------------------------------------------------+
//| Return the order filling policies flags                              |
//+----------------------------------------------------------------------+
long CSymbol::SymbolOrderFillingMode(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_FILLING_MODE) #else 0 #endif );
  }
//+----------------------------------------------------------------------+
//| Return the flag allowing the closure by an opposite position         |
//+----------------------------------------------------------------------+
bool CSymbol::IsCloseByOrdersAllowed(void) const
  {
   return(#ifdef __MQL5__(this.OrderModeFlags() & SYMBOL_ORDER_CLOSEBY)==SYMBOL_ORDER_CLOSEBY #else (bool)::MarketInfo(this.m_symbol_name,MODE_CLOSEBY_ALLOWED)  #endif );
  }
//+----------------------------------------------------------------------+
//| Return the lifetime of pending orders and                            |
//| placed StopLoss/TakeProfit levels                                    |
//+----------------------------------------------------------------------+
long CSymbol::SymbolOrderGTCMode(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_ORDER_GTC_MODE) #else SYMBOL_ORDERS_GTC #endif);
  }
//+----------------------------------------------------------------------+
//| Return the option type                                               |
//+----------------------------------------------------------------------+
long CSymbol::SymbolOptionMode(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_OPTION_MODE) #else SYMBOL_OPTION_MODE_NONE #endif);
  }
//+----------------------------------------------------------------------+
//| Return the option right                                              |
//+----------------------------------------------------------------------+
long CSymbol::SymbolOptionRight(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_OPTION_RIGHT) #else SYMBOL_OPTION_RIGHT_NONE #endif);
  }
//+----------------------------------------------------------------------+
//|Return the background color used to highlight a symbol in Market Watch|
//+----------------------------------------------------------------------+
long CSymbol::SymbolBackgroundColor(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_BACKGROUND_COLOR) #else clrNONE #endif);
  }
//+----------------------------------------------------------------------+
//| Return the margin calculation method                                 |
//+----------------------------------------------------------------------+
long CSymbol::SymbolCalcMode(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_TRADE_CALC_MODE) #else (long)::MarketInfo(this.m_symbol_name,MODE_MARGINCALCMODE) #endif);
  }
//+----------------------------------------------------------------------+
//| Return the swaps calculation method                                  |
//+----------------------------------------------------------------------+
long CSymbol::SymbolSwapMode(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_SWAP_MODE) #else (long)::MarketInfo(this.m_symbol_name,MODE_SWAPTYPE) #endif);
  }
//+----------------------------------------------------------------------+
//| Return the flags of allowed order expiration modes                   |
//+----------------------------------------------------------------------+
long CSymbol::SymbolExpirationMode(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_EXPIRATION_MODE) #else (long)SYMBOL_EXPIRATION_GTC #endif);
  }
//+----------------------------------------------------------------------+
```

Here we use the conditional compilation directives #ifdef
\_\_MQL5\_\_ — to compile for MQL5 and #else—
to compile for MQL4

#endif.

For MQL5, we simply obtain data from the [SymbolInfoInteger()](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger)
function with the required property ID, while for MQL4, we
return either a strictly set value (if we know that this exact value is used in MQL4) or zero (or 'false', or a set macro substitution for the
missing value).

Place the codes of another two methods in the same listing:

**Below is the method returning the flags of allowed order types for a symbol:**

```
//+------------------------------------------------------------------+
//| Return the flags of allowed order types                          |
//+------------------------------------------------------------------+
long CSymbol::SymbolOrderMode(void) const
  {
   return
     (
      #ifdef __MQL5__
         ::SymbolInfoInteger(this.m_symbol_name,SYMBOL_ORDER_MODE)
      #else
         (SYMBOL_ORDER_MARKET+SYMBOL_ORDER_LIMIT+SYMBOL_ORDER_STOP+SYMBOL_ORDER_SL+SYMBOL_ORDER_TP+(this.IsCloseByOrdersAllowed() ? SYMBOL_ORDER_CLOSEBY : 0))
      #endif
     );
  }
//+------------------------------------------------------------------+
```

Since MQL5 provides the ability to receive flags (that indicate the ability to place different order types) for each of the symbols:

- market order (open a position by market),

- limit order,

- stop order,

- stop limit order,

- TakeProfit order,

- StopLoss order,

- close by order (CloseBy)

For MQL5, we need to obtain the flags from the [SymbolInfoInteger()](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger)
function with the [SYMBOL\_ORDER\_MODE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#symbol_order_mode)
property ID returning all flags set for a symbol.

In MQL4, we are able to obtain only data on allowing
closing by an opposite order from the [MarketInfo()](https://docs.mql4.com/en/marketinformation/marketinfo "https://docs.mql4.com/en/marketinformation/marketinfo")
function with the [MODE\_CLOSEBY\_ALLOWED](https://docs.mql4.com/en/constants/environment_state/marketinfoconstants "https://docs.mql4.com/en/constants/environment_state/marketinfoconstants")
request ID (returned by the IsCloseByOrdersAllowed() method
located in the above listing).

Therefore, we need to collect the necessary flags for returning to an MQL4 program:

- market order allowed— add it to a
returned value



- limit order allowed—
add it to a returned value
- stop order allowed—
add it to a returned value
- stop limit order not allowed
- stop loss order allowed—
add it to a returned value
- take profit order allowed—
add it to a returned value
- for a close by order, receive data from the IsCloseByOrdersAllowed()
method and add the SYMBOL\_ORDER\_CLOSEBY constant value if allowedor
0 if not


**Below is the method for obtaining the number of decimal places in a lot value for a symbol:**

```
//+------------------------------------------------------------------+
//| Calculate and return the number of decimal places                |
//| in a symbol lot                                                  |
//+------------------------------------------------------------------+
long CSymbol::SymbolDigitsLot(void)
  {
   if(this.LotsMax()==0 || this.LotsMin()==0 || this.LotsStep()==0)
     {
      ::Print(DFUN_ERR_LINE,TextByLanguage("Не удалось получить данные \"","Failed to get data of \""),this.Name(),"\"");
      this.m_global_error=ERR_MARKET_UNKNOWN_SYMBOL;
      return 2;
     }
   double val=::round(this.LotsMin()/this.LotsStep())*this.LotsStep();
   string val_str=(string)val;
   int len=::StringLen(val_str);
   int n=len-::StringFind(val_str,".",0)-1;
   if(::StringSubstr(val_str,len-1,1)=="0")
      n--;
   return n;
  }
//+------------------------------------------------------------------+
```

The method calculates the number of decimal places in a symbol lot value.

We already have such a function in the
DELib.mqh service functions file:

```
//+------------------------------------------------------------------+
//| Returns the number of decimal places in a symbol lot             |
//+------------------------------------------------------------------+
uint DigitsLots(const string symbol_name)
  {
   return (int)ceil(fabs(log(SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_STEP))/log(10)));
  }
//+------------------------------------------------------------------+
```

However, it has certain flaws. I [was informed of them in one of \\
the article discussion threads](https://www.mql5.com/en/forum/310865#comment_12001686): if a lot step is 0.25, the function does not return a correct value. So I decided to look for a more accurate
method and finally decided that the most accurate method is to simply calculate the number of decimal places in the string value of the lot
reduced to a lot step: lots=MathRound(lots/lotStep)\*lotStep. This decision was prompted by the discussion in the thread (in Russian)
dedicated to this issue where

[a certain search method was suggested](https://www.mql5.com/ru/forum/287618/page9#comment_9355680), which is a
search in a string. But we need to find a necessary number of decimal places only once for each of the used symbols, while a constant value is
used afterwards. This method has no drawbacks of all calculation methods. So, let's use the proposed solution.

**Below are the methods for receiving symbol's real and string**
**properties that are either completely or partially absent in MQL4:**

```
//+------------------------------------------------------------------+
//| Real properties                                                  |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return maximum Bid for a day                                     |
//+------------------------------------------------------------------+
double CSymbol::SymbolBidHigh(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_BIDHIGH) #else ::MarketInfo(this.m_symbol_name,MODE_HIGH) #endif);
  }
//+------------------------------------------------------------------+
//| Return minimum Bid for a day                                     |
//+------------------------------------------------------------------+
double CSymbol::SymbolBidLow(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_BIDLOW) #else ::MarketInfo(this.m_symbol_name,MODE_LOW) #endif);
  }
//+------------------------------------------------------------------+
//| Return real Volume for a day                                     |
//+------------------------------------------------------------------+
double CSymbol::SymbolVolumeReal(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUME_REAL) #else 0 #endif);
  }
//+------------------------------------------------------------------+
//| Return real maximum Volume for a day                             |
//+------------------------------------------------------------------+
double CSymbol::SymbolVolumeHighReal(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUMEHIGH_REAL) #else 0 #endif);
  }
//+------------------------------------------------------------------+
//| Return real minimum Volume for a day                             |
//+------------------------------------------------------------------+
double CSymbol::SymbolVolumeLowReal(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_VOLUMELOW_REAL) #else 0 #endif);
  }
//+------------------------------------------------------------------+
//| Return an option execution price                                 |
//+------------------------------------------------------------------+
double CSymbol::SymbolOptionStrike(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_OPTION_STRIKE) #else 0 #endif);
  }
//+------------------------------------------------------------------+
//| Return accrued interest                                          |
//+------------------------------------------------------------------+
double CSymbol::SymbolTradeAccruedInterest(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_ACCRUED_INTEREST) #else 0 #endif);
  }
//+------------------------------------------------------------------+
//| Return a bond face value                                         |
//+------------------------------------------------------------------+
double CSymbol::SymbolTradeFaceValue(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_FACE_VALUE) #else 0 #endif);
  }
//+------------------------------------------------------------------+
//| Return a liquidity rate                                          |
//+------------------------------------------------------------------+
double CSymbol::SymbolTradeLiquidityRate(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_TRADE_LIQUIDITY_RATE) #else 0 #endif);
  }
//+------------------------------------------------------------------+
//| Return a contract or margin size                                 |
//| for a single lot of covered positions                            |
//+------------------------------------------------------------------+
double CSymbol::SymbolMarginHedged(void) const
  {
   return(#ifdef __MQL5__ ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_HEDGED) #else ::MarketInfo(this.m_symbol_name, MODE_MARGINHEDGED) #endif);
  }
//+------------------------------------------------------------------+
//| String properties                                                |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return a base asset name for a derivative instrument             |
//+------------------------------------------------------------------+
string CSymbol::SymbolBasis(void) const
  {
   return
     (
      #ifdef __MQL5__
         ::SymbolInfoString(this.m_symbol_name,SYMBOL_BASIS)
      #else
         ": "+TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4")
      #endif
     );
  }
//+------------------------------------------------------------------+
//| Return a quote source for a symbol                               |
//+------------------------------------------------------------------+
string CSymbol::SymbolBank(void) const
  {
   return
     (
      #ifdef __MQL5__
         ::SymbolInfoString(this.m_symbol_name,SYMBOL_BANK)
      #else
         ": "+TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4")
      #endif
     );
  }
//+------------------------------------------------------------------+
//| Return a symbol name to ISIN                                     |
//+------------------------------------------------------------------+
string CSymbol::SymbolISIN(void) const
  {
   return
     (
      #ifdef __MQL5__
         ::SymbolInfoString(this.m_symbol_name,SYMBOL_ISIN)
      #else
         ": "+TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4")
      #endif
     );
  }

//+------------------------------------------------------------------+
//| Return a formula for constructing a custom symbol price          |
//+------------------------------------------------------------------+
string CSymbol::SymbolFormula(void) const
  {
   return
     (
      #ifdef __MQL5__
         ::SymbolInfoString(this.m_symbol_name,SYMBOL_FORMULA)
      #else
         ": "+TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4")
      #endif
     );
  }
//+------------------------------------------------------------------+
//| Return an address of a web page with a symbol data               |
//+------------------------------------------------------------------+
string CSymbol::SymbolPage(void) const
  {
   return
     (
      #ifdef __MQL5__
         ::SymbolInfoString(this.m_symbol_name,SYMBOL_PAGE)
      #else
         ": "+TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4")
      #endif
     );
  }
//+------------------------------------------------------------------+
```

Here all is similar to the methods of obtaining integer properties: obtaining a returned value is divided by the conditional compilation
directives, and the property values are either returned with the appropriate MQL5 and MQL4 functions, or 0 is returned for MQL4 (if there is
no MQL5 analogue). The string message indicating that MQL4 does not support this string property can be returned as well.

**Below is the method for sending the full list of all symbol object properties to the journal:**

```
//+------------------------------------------------------------------+
//| Send symbol properties to the journal                            |
//+------------------------------------------------------------------+
void CSymbol::Print(const bool full_prop=false)
  {
   ::Print("============= ",
           TextByLanguage("Начало списка параметров: \"","Beginning of the parameter list: \""),
           this.Name(),"\""," ",(this.Description()!= #ifdef __MQL5__ "" #else NULL #endif  ? "("+this.Description()+")" : ""),
           " =================="
          );
   int beg=0, end=SYMBOL_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_INTEGER prop=(ENUM_SYMBOL_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=SYMBOL_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_DOUBLE prop=(ENUM_SYMBOL_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=SYMBOL_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SYMBOL_PROP_STRING prop=(ENUM_SYMBOL_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("================== ",
           TextByLanguage("Конец списка параметров: ","End of the parameter list: \""),
           this.Name(),"\""," ",(this.Description()!= #ifdef __MQL5__ "" #else NULL #endif  ? "("+this.Description()+")" : ""),
           " ==================\n"
          );
  }
//+------------------------------------------------------------------+
```

The description of each subsequent property is displayed in three loops for all object properties using the **GetPropertyDescription()**
**methods returning an object property description (passed to the method) by its type (**

**integer, real**
**or string):**

```
//+------------------------------------------------------------------+
//| Return the description of the symbol integer property            |
//+------------------------------------------------------------------+
string CSymbol::GetPropertyDescription(ENUM_SYMBOL_PROP_INTEGER property)
  {
   return
     (
   //--- General properties
      property==SYMBOL_PROP_STATUS              ?  TextByLanguage("Статус","Status")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetStatusDescription()
         )  :
      property==SYMBOL_PROP_CUSTOM              ?  TextByLanguage("Пользовательский символ","Custom symbol")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_CHART_MODE          ?  TextByLanguage("Тип цены для построения баров","Price type used for generating symbols bars")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetChartModeDescription()
         )  :
      property==SYMBOL_PROP_EXIST               ?  TextByLanguage("Символ с таким именем существует","Symbol with this name exists")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_SELECT  ?  TextByLanguage("Символ выбран в Market Watch","Symbol selected in Market Watch")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_VISIBLE ?  TextByLanguage("Символ отображается в Market Watch","Symbol visible in Market Watch")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_SESSION_DEALS       ?  TextByLanguage("Количество сделок в текущей сессии","Number of deals in the current session")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property is not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_BUY_ORDERS  ?  TextByLanguage("Общее число ордеров на покупку в текущий момент","Number of Buy orders at the moment")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property is not support") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_SELL_ORDERS ?  TextByLanguage("Общее число ордеров на продажу в текущий момент","Number of Sell orders at the moment")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUME              ?  TextByLanguage("Объем в последней сделке","Volume of the last deal")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUMEHIGH          ?  TextByLanguage("Максимальный объём за день","Maximum day volume")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property is not support") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property is not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUMELOW           ?  TextByLanguage("Минимальный объём за день","Minimum day volume")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_TIME                ?  TextByLanguage("Время последней котировки","Time of last quote")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)==0 ? TextByLanguage("(Ещё не было тиков)","(No ticks yet)") : TimeMSCtoString(this.GetProperty(property)))
         )  :
      property==SYMBOL_PROP_DIGITS              ?  TextByLanguage("Количество знаков после запятой","Digits after decimal point")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_DIGITS_LOTS         ?  TextByLanguage("Количество знаков после запятой в значении лота","Digits after decimal point in lot value")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_SPREAD              ?  TextByLanguage("Размер спреда в пунктах","Spread value in points")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_SPREAD_FLOAT        ?  TextByLanguage("Плавающий спред","Floating spread")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_TICKS_BOOKDEPTH     ?  TextByLanguage("Максимальное количество показываемых заявок в стакане","Maximum number of requests shown in Depth of Market")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_TRADE_CALC_MODE     ?  TextByLanguage("Способ вычисления стоимости контракта","Contract price calculation mode")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetCalcModeDescription()
         )  :
      property==SYMBOL_PROP_TRADE_MODE ?  TextByLanguage("Тип исполнения ордеров","Order execution type")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetTradeModeDescription()
         )  :
      property==SYMBOL_PROP_START_TIME          ?  TextByLanguage("Дата начала торгов по инструменту","Date of symbol trade beginning")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)==0  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": "+TimeMSCtoString(this.GetProperty(property)*1000))
         )  :
      property==SYMBOL_PROP_EXPIRATION_TIME     ?  TextByLanguage("Дата окончания торгов по инструменту","Date of symbol trade end")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)==0  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": "+TimeMSCtoString(this.GetProperty(property)*1000))
         )  :
      property==SYMBOL_PROP_TRADE_STOPS_LEVEL   ?  TextByLanguage("Минимальный отступ от цены закрытия для установки Stop ордеров","Minimum indention from close price to place Stop orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_TRADE_FREEZE_LEVEL  ?  TextByLanguage("Дистанция заморозки торговых операций","Distance to freeze trade operations in points")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_TRADE_EXEMODE       ?  TextByLanguage("Режим заключения сделок","Deal execution mode")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetTradeExecDescription()
         )  :
      property==SYMBOL_PROP_SWAP_MODE           ?  TextByLanguage("Модель расчета свопа","Swap calculation model")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetSwapModeDescription()
         )  :
      property==SYMBOL_PROP_SWAP_ROLLOVER3DAYS  ?  TextByLanguage("День недели для начисления тройного свопа","Day of week to charge 3 days swap rollover")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+DayOfWeekDescription(this.SwapRollover3Days())
         )  :
      property==SYMBOL_PROP_MARGIN_HEDGED_USE_LEG  ?  TextByLanguage("Расчет хеджированной маржи по наибольшей стороне","Calculating hedging margin using larger leg")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_EXPIRATION_MODE     ?  TextByLanguage("Флаги разрешенных режимов истечения ордера","Flags of allowed order expiration modes")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetExpirationModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_FILLING_MODE        ?  TextByLanguage("Флаги разрешенных режимов заливки ордера","Flags of allowed order filling modes")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetFillingModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_ORDER_MODE          ?  TextByLanguage("Флаги разрешенных типов ордера","Flags of allowed order types")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetOrderModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_ORDER_GTC_MODE      ?  TextByLanguage("Срок действия StopLoss и TakeProfit ордеров","Expiration of Stop Loss and Take Profit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetOrderGTCModeDescription()
         )  :
      property==SYMBOL_PROP_OPTION_MODE         ?  TextByLanguage("Тип опциона","Option type")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetOptionTypeDescription()
         )  :
      property==SYMBOL_PROP_OPTION_RIGHT        ?  TextByLanguage("Право опциона","Option right")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetOptionRightDescription()
         )  :
      property==SYMBOL_PROP_BACKGROUND_COLOR    ?  TextByLanguage("Цвет фона символа в Market Watch","Background color of symbol in Market Watch")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         #ifdef __MQL5__
         (this.GetProperty(property)>clrWhite  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": "+::ColorToString((color)this.GetProperty(property),true))
         #else TextByLanguage(": Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the symbol real property               |
//+------------------------------------------------------------------+
string CSymbol::GetPropertyDescription(ENUM_SYMBOL_PROP_DOUBLE property)
  {
   int dg=this.Digits();
   int dgl=this.DigitsLot();
   int dgc=this.DigitsCurrency();
   return
     (
      property==SYMBOL_PROP_BID              ?  TextByLanguage("Цена Bid","Bid price")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)==0 ? TextByLanguage("(Ещё не было тиков)","(No ticks yet)") : ::DoubleToString(this.GetProperty(property),dg))
         )  :
      property==SYMBOL_PROP_BIDHIGH          ?  TextByLanguage("Максимальный Bid за день","Maximum Bid for the day")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==SYMBOL_PROP_BIDLOW           ?  TextByLanguage("Минимальный Bid за день","Minimum Bid for the day")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==SYMBOL_PROP_ASK              ?  TextByLanguage("Цена Ask","Ask price")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)==0 ? TextByLanguage("(Ещё не было тиков)","(No ticks yet)") : ::DoubleToString(this.GetProperty(property),dg))
         )  :
      property==SYMBOL_PROP_ASKHIGH          ?  TextByLanguage("Максимальный Ask за день","Maximum Ask for the day")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_ASKLOW           ?  TextByLanguage("Минимальный Ask за день","Minimum Ask for the day")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_LAST             ?  TextByLanguage("Цена последней сделки","Price of the last deal")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_LASTHIGH         ?  TextByLanguage("Максимальный Last за день","Maximum Last for the day")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_LASTLOW          ?  TextByLanguage("Минимальный Last за день","Minimum Last for the day")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUME_REAL      ?  TextByLanguage("Реальный объём за день","Real volume of the last deal")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgl) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUMEHIGH_REAL  ?  TextByLanguage("Максимальный реальный объём за день","Maximum real volume for the day")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgl) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUMELOW_REAL   ?  TextByLanguage("Минимальный реальный объём за день","Minimum real volume for the day")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgl) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_OPTION_STRIKE    ?  TextByLanguage("Цена исполнения опциона","Option strike price")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_POINT            ?  TextByLanguage("Значение одного пункта","Symbol point value")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==SYMBOL_PROP_TRADE_TICK_VALUE ?  TextByLanguage("Рассчитанная стоимость тика для позиции","Calculated tick price for position")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgc)
         )  :
      property==SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT   ?  TextByLanguage("Рассчитанная стоимость тика для прибыльной позиции","Calculated tick price for profitable position")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_TRADE_TICK_VALUE_LOSS  ?  TextByLanguage("Рассчитанная стоимость тика для убыточной позиции","Calculated tick price for losing position")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_TRADE_TICK_SIZE  ?  TextByLanguage("Минимальное изменение цены","Minimum price change")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==SYMBOL_PROP_TRADE_CONTRACT_SIZE ?  TextByLanguage("Размер торгового контракта","Trade contract size")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgc)
         )  :
      property==SYMBOL_PROP_TRADE_ACCRUED_INTEREST ?  TextByLanguage("Накопленный купонный доход","Accumulated coupon interest")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_TRADE_FACE_VALUE ?  TextByLanguage("Начальная стоимость облигации, установленная эмитентом","Initial bond value set by issuer")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_TRADE_LIQUIDITY_RATE   ?  TextByLanguage("Коэффициент ликвидности","Liquidity rate")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),2) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUME_MIN       ?  TextByLanguage("Минимальный объем для заключения сделки","Minimum volume for a deal")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgl)
         )  :
      property==SYMBOL_PROP_VOLUME_MAX       ?  TextByLanguage("Максимальный объем для заключения сделки","Maximum volume for a deal")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgl)
         )  :
      property==SYMBOL_PROP_VOLUME_STEP      ?  TextByLanguage("Минимальный шаг изменения объема для заключения сделки","Minimum volume change step for deal execution")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgl)
         )  :
      property==SYMBOL_PROP_VOLUME_LIMIT     ?  TextByLanguage("Максимально допустимый общий объем позиции и отложенных ордеров в одном направлении","Maximum allowed aggregate volume of open position and pending orders in one direction")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgl) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SWAP_LONG        ?  TextByLanguage("Значение свопа на покупку","Long swap value")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgc)
         )  :

      property==SYMBOL_PROP_SWAP_SHORT       ?  TextByLanguage("Значение свопа на продажу","Short swap value")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgc)
         )  :
      property==SYMBOL_PROP_MARGIN_INITIAL   ?  TextByLanguage("Начальная (инициирующая) маржа","Initial margin")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgc)
         )  :
      property==SYMBOL_PROP_MARGIN_MAINTENANCE  ?  TextByLanguage("Поддерживающая маржа по инструменту","Maintenance margin")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgc)
         )  :
      property==SYMBOL_PROP_MARGIN_LONG      ?  TextByLanguage("Коэффициент взимания маржи по длинным позициям","Coefficient of margin charging for long positions")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_SHORT     ?  TextByLanguage("Коэффициент взимания маржи по коротким позициям","Coefficient of margin charging for short positions")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_STOP      ?  TextByLanguage("Коэффициент взимания маржи по Stop ордерам","Coefficient of margin charging for Stop orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_LIMIT     ?  TextByLanguage("Коэффициент взимания маржи по Limit ордерам","Coefficient of margin charging for Limit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_STOPLIMIT ?  TextByLanguage("Коэффициент взимания маржи по Stop Limit ордерам","Coefficient of margin charging for StopLimit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_VOLUME   ?  TextByLanguage("Cуммарный объём сделок в текущую сессию","Summary volume of the current session deals")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgl) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_TURNOVER ?  TextByLanguage("Cуммарный оборот в текущую сессию","Summary turnover of the current session")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgc) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_INTEREST ?  TextByLanguage("Cуммарный объём открытых позиций","Summary open interest")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgl) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME ?  TextByLanguage("Общий объём ордеров на покупку в текущий момент","Current volume of Buy orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgl) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME   ?  TextByLanguage("Общий объём ордеров на продажу в текущий момент","Current volume of Sell orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dgl) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_OPEN     ?  TextByLanguage("Цена открытия сессии","Open price of the current session")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_CLOSE    ?  TextByLanguage("Цена закрытия сессии","Close price of the current session")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_AW       ?  TextByLanguage("Средневзвешенная цена сессии","Average weighted price of the current session")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_PRICE_SETTLEMENT  ?  TextByLanguage("Цена поставки на текущую сессию","Settlement price of the current session")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN   ?  TextByLanguage("Минимально допустимое значение цены на сессию","Minimum price of the current session")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX   ?  TextByLanguage("Максимально допустимое значение цены на сессию","Maximum price of the current session")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__::DoubleToString(this.GetProperty(property),dg) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_MARGIN_HEDGED    ?  TextByLanguage("Размер контракта или маржи для одного лота перекрытых позиций","Contract size or margin value per one lot of hedged positions")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::DoubleToString(this.GetProperty(property),dgc)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the symbol string property             |
//+------------------------------------------------------------------+
string CSymbol::GetPropertyDescription(ENUM_SYMBOL_PROP_STRING property)
  {
   return
     (
      property==SYMBOL_PROP_NAME             ?  TextByLanguage("Имя символа","Symbol name")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_BASIS            ?  TextByLanguage("Имя базового актива для производного инструмента","Underlying asset of derivative")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_CURRENCY_BASE    ?  TextByLanguage("Базовая валюта инструмента","Basic currency of a symbol")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_CURRENCY_PROFIT  ?  TextByLanguage("Валюта прибыли","Profit currency")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_CURRENCY_MARGIN  ?  TextByLanguage("Валюта залоговых средств","Margin currency")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_BANK             ?  TextByLanguage("Источник текущей котировки","Feeder of the current quote")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_DESCRIPTION      ?  TextByLanguage("Описание символа","Symbol description")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_FORMULA          ?  TextByLanguage("Формула для построения цены пользовательского символа","Formula used for custom symbol pricing")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_ISIN             ?  TextByLanguage("Имя торгового символа в системе международных идентификационных кодов","Name of a symbol in ISIN system")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_PAGE             ?  TextByLanguage("Адрес интернет страницы с информацией по символу","Address of the web page containing symbol information")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_PATH             ?  TextByLanguage("Путь в дереве символов","Path in symbol tree")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": \""+this.GetProperty(property)+"\"")
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Although the methods are quite bulky, it all boils down to comparing a property passed to the method and returning its string description. I
believe, they are quite easy to understand, and there is no point in dwelling on them.

**Below is the method for searching for a symbol in the list of server symbols and returning the flag of its presence/absence:**

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
```

Compare each subsequent symbol (its name) from the list with the name of the current symbol object in a loop by the full list of all symbols on the
server (in the

[SymbolsTotal()](https://www.mql5.com/en/docs/marketinformation/symbolstotal) and [SymbolName()](https://www.mql5.com/en/docs/marketinformation/symbolname)
functions, flag = false). If
found, return true
— the symbol is present on the server. Otherwise, return false
— no symbol is present on the server.

**Below are the methods for subscribing to the market depth and unsubscribing from it:**

```
//+------------------------------------------------------------------+
//| Subscribe to the market depth                                    |
//+------------------------------------------------------------------+
bool CSymbol::BookAdd(void) const
  {
   return #ifdef __MQL5__ ::MarketBookAdd(this.m_symbol_name) #else false #endif ;
  }
//+------------------------------------------------------------------+
//| Close the market depth                                           |
//+------------------------------------------------------------------+
bool CSymbol::BookClose(void) const
  {
   return #ifdef __MQL5__ ::MarketBookRelease(this.m_symbol_name) #else false #endif ;
  }
//+------------------------------------------------------------------+
```

In MQL5, the methods simply return the result of [MarketBookAdd()](https://www.mql5.com/en/docs/marketinformation/marketbookadd)
and [MarketBookRelease()](https://www.mql5.com/en/docs/marketinformation/marketbookrelease) functions operation,
while

in MQL4, false is
returned right away since there is no way to work with the market depth there. For now, these methods are added to a symbol object. We will
arrange the ability to manage them, as well as add other methods for working with the market depth, in the coming articles.

**Below are the methods returning string descriptions of the symbol object properties:**

```
//+------------------------------------------------------------------+
//| Return the status description                                    |
//+------------------------------------------------------------------+
string CSymbol::GetStatusDescription() const
  {
   return
     (
      this.Status()==SYMBOL_STATUS_FX              ? TextByLanguage("Форекс символ","Forex symbol")                  :
      this.Status()==SYMBOL_STATUS_FX_MAJOR        ? TextByLanguage("Форекс символ-мажор","Forex major symbol")      :
      this.Status()==SYMBOL_STATUS_FX_MINOR        ? TextByLanguage("Форекс символ-минор","Forex minor symbol")      :
      this.Status()==SYMBOL_STATUS_FX_EXOTIC       ? TextByLanguage("Форекс символ-экзотик","Forex Exotic Symbol")   :
      this.Status()==SYMBOL_STATUS_FX_RUB          ? TextByLanguage("Форекс символ/рубль","Forex symbol RUB")        :
      this.Status()==SYMBOL_STATUS_FX_METAL        ? TextByLanguage("Металл","Metal")                                :
      this.Status()==SYMBOL_STATUS_INDEX           ? TextByLanguage("Индекс","Index")                                :
      this.Status()==SYMBOL_STATUS_INDICATIVE      ? TextByLanguage("Индикатив","Indicative")                        :
      this.Status()==SYMBOL_STATUS_CRYPTO          ? TextByLanguage("Криптовалютный символ","Crypto symbol")         :
      this.Status()==SYMBOL_STATUS_COMMODITY       ? TextByLanguage("Товарный символ","Commodity symbol")            :
      this.Status()==SYMBOL_STATUS_EXCHANGE        ? TextByLanguage("Биржевой символ","Exchange symbol")             :
      this.Status()==SYMBOL_STATUS_BIN_OPTION      ? TextByLanguage("Бинарный опцион","Binary option")               :
      this.Status()==SYMBOL_STATUS_CUSTOM          ? TextByLanguage("Пользовательский символ","Custom symbol")       :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the price type description for constructing bars          |
//+------------------------------------------------------------------+
string CSymbol::GetChartModeDescription(void) const
  {
   return
     (
      this.ChartMode()==SYMBOL_CHART_MODE_BID ? TextByLanguage("Бары строятся по ценам Bid","Bars based on Bid prices") :
      TextByLanguage("Бары строятся по ценам Last","Bars based on Last prices")
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the margin calculation method          |
//+------------------------------------------------------------------+
string CSymbol::GetCalcModeDescription(void) const
  {
   return
     (
      this.TradeCalcMode()==SYMBOL_CALC_MODE_FOREX                ?
         TextByLanguage("Расчет прибыли и маржи для Форекс","Forex mode")                                               :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE    ?
         TextByLanguage("Расчет прибыли и маржи для Форекс без учета плеча","Forex No Leverage mode")                   :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_FUTURES              ?
         TextByLanguage("Расчет залога и прибыли для фьючерсов","Futures mode")                                         :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_CFD                  ?
         TextByLanguage("Расчет залога и прибыли для CFD","CFD mode")                                                   :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_CFDINDEX             ?
         TextByLanguage("Расчет залога и прибыли для CFD на индексы","CFD index mode")                                  :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_CFDLEVERAGE          ?
         TextByLanguage("Расчет залога и прибыли для CFD при торговле с плечом","CFD Leverage mode")                    :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_EXCH_STOCKS          ?
         TextByLanguage("Расчет залога и прибыли для торговли ценными бумагами на бирже","Exchange mode")               :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_EXCH_FUTURES         ?
         TextByLanguage("Расчет залога и прибыли для торговли фьючерсными контрактами на бирже","Futures mode")         :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS   ?
         TextByLanguage("Расчет залога и прибыли для торговли фьючерсными контрактами на FORTS","FORTS Futures mode")   :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_EXCH_BONDS           ?
         TextByLanguage("Расчет прибыли и маржи по торговым облигациям на бирже","Exchange Bonds mode")                 :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX     ?
         TextByLanguage("Расчет прибыли и маржи при торговле ценными бумагами на MOEX","Exchange MOEX Stocks mode")     :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_EXCH_BONDS_MOEX      ?
         TextByLanguage("Расчет прибыли и маржи по торговым облигациям на MOEX","Exchange MOEX Bonds mode")             :
      this.TradeCalcMode()==SYMBOL_CALC_MODE_SERV_COLLATERAL      ?
         TextByLanguage("Используется в качестве неторгуемого актива на счете","Collateral mode")                       :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of a symbol trading mode                  |
//+------------------------------------------------------------------+
string CSymbol::GetTradeModeDescription(void) const
  {
   return
     (
      this.TradeMode()==SYMBOL_TRADE_MODE_DISABLED    ? TextByLanguage("Торговля по символу запрещена","Trade disabled for symbol")                     :
      this.TradeMode()==SYMBOL_TRADE_MODE_LONGONLY    ? TextByLanguage("Разрешены только покупки","Only long positions allowed")                               :
      this.TradeMode()==SYMBOL_TRADE_MODE_SHORTONLY   ? TextByLanguage("Разрешены только продажи","Only short positions allowed")                              :
      this.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY   ? TextByLanguage("Разрешены только операции закрытия позиций","Only position close operations allowed")  :
      this.TradeMode()==SYMBOL_TRADE_MODE_FULL        ? TextByLanguage("Нет ограничений на торговые операции","No trade restrictions")                         :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of a symbol trade execution mode          |
//+------------------------------------------------------------------+
string CSymbol::GetTradeExecDescription(void) const
  {
   return
     (
      this.TradeExecutionMode()==SYMBOL_TRADE_EXECUTION_REQUEST   ? TextByLanguage("Торговля по запросу","Execution by request")       :
      this.TradeExecutionMode()==SYMBOL_TRADE_EXECUTION_INSTANT   ? TextByLanguage("Торговля по потоковым ценам","Instant execution")  :
      this.TradeExecutionMode()==SYMBOL_TRADE_EXECUTION_MARKET    ? TextByLanguage("Исполнение ордеров по рынку","Market execution")   :
      this.TradeExecutionMode()==SYMBOL_TRADE_EXECUTION_EXCHANGE  ? TextByLanguage("Биржевое исполнение","Exchange execution")         :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of a swap calculation model               |
//+------------------------------------------------------------------+
string CSymbol::GetSwapModeDescription(void) const
  {
   return
     (
      this.SwapMode()==SYMBOL_SWAP_MODE_DISABLED         ?
         TextByLanguage("Нет свопов","Swaps disabled (no swaps)")                                                                                                                                                    :
      this.SwapMode()==SYMBOL_SWAP_MODE_POINTS           ?
         TextByLanguage("Свопы начисляются в пунктах","Swaps charged in points")                                                                                                                                 :
      this.SwapMode()==SYMBOL_SWAP_MODE_CURRENCY_SYMBOL  ?
         TextByLanguage("Свопы начисляются в деньгах в базовой валюте символа","Swaps charged in money in base currency of symbol")                                                                          :
      this.SwapMode()==SYMBOL_SWAP_MODE_CURRENCY_MARGIN  ?
         TextByLanguage("Свопы начисляются в деньгах в маржинальной валюте символа","Swaps charged in money in margin currency of symbol")                                                                   :
      this.SwapMode()==SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT ?
         TextByLanguage("Свопы начисляются в деньгах в валюте депозита клиента","Swaps charged in money, in client deposit currency")                                                                            :
      this.SwapMode()==SYMBOL_SWAP_MODE_INTEREST_CURRENT ?
         TextByLanguage("Свопы начисляются в годовых процентах от цены инструмента на момент расчета свопа","Swaps charged as specified annual interest from instrument price at calculation of swap")   :
      this.SwapMode()==SYMBOL_SWAP_MODE_INTEREST_OPEN    ?
         TextByLanguage("Свопы начисляются в годовых процентах от цены открытия позиции по символу","Swaps charged as specified annual interest from open price of position")                            :
      this.SwapMode()==SYMBOL_SWAP_MODE_REOPEN_CURRENT   ?
         TextByLanguage("Свопы начисляются переоткрытием позиции по цене закрытия","Swaps charged by reopening positions by close price")                                                                    :
      this.SwapMode()==SYMBOL_SWAP_MODE_REOPEN_BID       ?
         TextByLanguage("Свопы начисляются переоткрытием позиции по текущей цене Bid","Swaps charged by reopening positions by the current Bid price")                                                           :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of StopLoss and TakeProfit order lifetime |
//+------------------------------------------------------------------+
string CSymbol::GetOrderGTCModeDescription(void) const
  {
   return
     (
      this.OrderModeGTC()==SYMBOL_ORDERS_GTC                   ?
         TextByLanguage("Отложенные ордеры и уровни Stop Loss/Take Profit действительны неограниченно по времени до явной отмены","Pending orders and Stop Loss/Take Profit levels are valid for unlimited period until their explicit cancellation") :
      this.OrderModeGTC()==SYMBOL_ORDERS_DAILY                 ?
         TextByLanguage("При смене торгового дня отложенные ордеры и все уровни StopLoss и TakeProfit удаляются","At the end of the day, all Stop Loss and Take Profit levels, as well as pending orders are deleted")                                   :
      this.OrderModeGTC()==SYMBOL_ORDERS_DAILY_EXCLUDING_STOPS ?
         TextByLanguage("При смене торгового дня удаляются только отложенные ордеры, уровни StopLoss и TakeProfit сохраняются","At the end of the day, only pending orders deleted, while Stop Loss and Take Profit levels preserved")           :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the option type description                               |
//+------------------------------------------------------------------+
string CSymbol::GetOptionTypeDescription(void) const
  {
   return
     (
      #ifdef __MQL4__ TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #else
      this.OptionMode()==SYMBOL_OPTION_MODE_EUROPEAN ?
         TextByLanguage("Европейский тип опциона – может быть погашен только в указанную дату","European option may only be exercised on specified date")                               :
      this.OptionMode()==SYMBOL_OPTION_MODE_AMERICAN ?
         TextByLanguage("Американский тип опциона – может быть погашен в любой день до истечения срока опциона","American option may be exercised on any trading day or before expiry")   :
      ""
      #endif
     );
  }
//+------------------------------------------------------------------+
//| Return the option right description                              |
//+------------------------------------------------------------------+
string CSymbol::GetOptionRightDescription(void) const
  {
   return
     (
      #ifdef __MQL4__ TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #else
      this.OptionRight()==SYMBOL_OPTION_RIGHT_CALL ?
         TextByLanguage("Опцион, дающий право купить актив по фиксированной цене","Call option gives you right to buy asset at specified price")    :
      this.OptionRight()==SYMBOL_OPTION_RIGHT_PUT  ?
         TextByLanguage("Опцион, дающий право продать актив по фиксированной цене  ","Put option gives you right to sell asset at specified price") :
      ""
      #endif
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the flags of allowed order types       |
//+------------------------------------------------------------------+
string CSymbol::GetOrderModeFlagsDescription(void) const
  {
   string first=#ifdef __MQL5__ "\n - " #else ""   #endif ;
   string next= #ifdef __MQL5__ "\n - " #else "; " #endif ;
   return
     (
      first+this.GetMarketOrdersAllowedDescription()+
      next+this.GetLimitOrdersAllowedDescription()+
      next+this.GetStopOrdersAllowedDescription()+
      next+this.GetStopLimitOrdersAllowedDescription()+
      next+this.GetStopLossOrdersAllowedDescription()+
      next+this.GetTakeProfitOrdersAllowedDescription()+
      next+this.GetCloseByOrdersAllowedDescription()
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of the flags of allowed filling types         |
//+----------------------------------------------------------------------+
string CSymbol::GetFillingModeFlagsDescription(void) const
  {
   string first=#ifdef __MQL5__ "\n - " #else ""   #endif ;
   string next= #ifdef __MQL5__ "\n - " #else "; " #endif ;
   return
     (
      first+TextByLanguage("Вернуть (Да)","Return (Yes)")+
      next+this.GetFillingModeFOKAllowedDescrioption()+
      next+this.GetFillingModeIOCAllowedDescrioption()
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of the flags of allowed order expiration modes|
//+----------------------------------------------------------------------+
string CSymbol::GetExpirationModeFlagsDescription(void) const
  {
   string first=#ifdef __MQL5__ "\n - " #else ""   #endif ;
   string next= #ifdef __MQL5__ "\n - " #else "; " #endif ;
   return
     (
      first+this.GetExpirationModeGTCDescription()+
      next+this.GetExpirationModeDAYDescription()+
      next+this.GetExpirationModeSpecifiedDescription()+
      next+this.GetExpirationModeSpecDayDescription()
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing to use market orders              |
//+----------------------------------------------------------------------+
string CSymbol::GetMarketOrdersAllowedDescription(void) const
  {
   return
     (this.IsMarketOrdersAllowed() ?
      TextByLanguage("Рыночный ордер (Да)","Market order (Yes)") :
      TextByLanguage("Рыночный ордер (Нет)","Market order (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing to use limit orders               |
//+----------------------------------------------------------------------+
string CSymbol::GetLimitOrdersAllowedDescription(void) const
  {
   return
     (this.IsLimitOrdersAllowed() ?
      TextByLanguage("Лимит ордер (Да)","Limit order (Yes)") :
      TextByLanguage("Лимит ордер (Нет)","Limit order (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing to use stop orders                |
//+----------------------------------------------------------------------+
string CSymbol::GetStopOrdersAllowedDescription(void) const
  {
   return
     (this.IsStopOrdersAllowed() ?
      TextByLanguage("Стоп ордер (Да)","Stop order (Yes)") :
      TextByLanguage("Стоп ордер (Нет)","Stop order (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing to use stop limit orders          |
//+----------------------------------------------------------------------+
string CSymbol::GetStopLimitOrdersAllowedDescription(void) const
  {
   return
     (this.IsStopLimitOrdersAllowed() ?
      TextByLanguage("Стоп-лимит ордер (Да)","StopLimit order (Yes)") :
      TextByLanguage("Стоп-лимит ордер (Нет)","StopLimit order (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing to set StopLoss orders            |
//+----------------------------------------------------------------------+
string CSymbol::GetStopLossOrdersAllowedDescription(void) const
  {
   return
     (this.IsStopLossOrdersAllowed() ?
      TextByLanguage("StopLoss (Да)","StopLoss (Yes)") :
      TextByLanguage("StopLoss (Нет)","StopLoss (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing to set TakeProfit orders          |
//+----------------------------------------------------------------------+
string CSymbol::GetTakeProfitOrdersAllowedDescription(void) const
  {
   return
     (this.IsTakeProfitOrdersAllowed() ?
      TextByLanguage("TakeProfit (Да)","TakeProfit (Yes)") :
      TextByLanguage("TakeProfit (Нет)","TakeProfit (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing to close by an opposite position  |
//+----------------------------------------------------------------------+
string CSymbol::GetCloseByOrdersAllowedDescription(void) const
  {
   return
     (this.IsCloseByOrdersAllowed() ?
      TextByLanguage("Закрытие встречным (Да)","CloseBy order (Yes)") :
      TextByLanguage("Закрытие встречным (Нет)","CloseBy order (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing FOK filling type                  |
//+----------------------------------------------------------------------+
string CSymbol::GetFillingModeFOKAllowedDescrioption(void) const
  {
   return
     (this.IsFillingModeFOK() ?
      TextByLanguage("Всё/Ничего (Да)","Fill or Kill (Yes)") :
      TextByLanguage("Всё/Ничего (Нет)","Fill or Kill (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of allowing IOC filling type                  |
//+----------------------------------------------------------------------+
string CSymbol::GetFillingModeIOCAllowedDescrioption(void) const
  {
   return
     (this.IsFillingModeIOC() ?
      TextByLanguage("Всё/Частично (Да)","Immediate or Cancel order (Yes)") :
      TextByLanguage("Всё/Частично (Нет)","Immediate or Cancel (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of GTC order expiration                       |
//+----------------------------------------------------------------------+
string CSymbol::GetExpirationModeGTCDescription(void) const
  {
   return
     (this.IsExipirationModeGTC() ?
      TextByLanguage("Неограниченно (Да)","Unlimited (Yes)") :
      TextByLanguage("Неограниченно (Нет)","Unlimited (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of DAY order expiration                       |
//+----------------------------------------------------------------------+
string CSymbol::GetExpirationModeDAYDescription(void) const
  {
   return
     (this.IsExipirationModeDAY() ?
      TextByLanguage("До конца дня (Да)","Valid till the end of day (Yes)") :
      TextByLanguage("До конца дня (Нет)","Valid till the end of day (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of Specified order expiration                 |
//+----------------------------------------------------------------------+
string CSymbol::GetExpirationModeSpecifiedDescription(void) const
  {
   return
     (this.IsExipirationModeSpecified() ?
      TextByLanguage("Срок указывается в ордере (Да)","Time specified in order (Yes)") :
      TextByLanguage("Срок указывается в ордере (Нет)","Time specified in order (No)")
     );
  }
//+----------------------------------------------------------------------+
//| Return the description of Specified Day order expiration             |
//+----------------------------------------------------------------------+
string CSymbol::GetExpirationModeSpecDayDescription(void) const
  {
   return
     (this.IsExipirationModeSpecifiedDay() ?
      TextByLanguage("День указывается в ордере (Да)","Date specified in order (Yes)") :
      TextByLanguage("День указывается в ордере (Нет)","Date specified in order (No)")
     );
  }
//+------------------------------------------------------------------+
```

In methods, everything is simple: a property value is checked and its string description is returned.

Some
methods (namely, flag description ones) display string values of other methods within a returned description. In turn, these other
methods return the descriptions of specific flags the value of a checked property consists of. Thus, we obtain a formatted composite
description of all flags of a single property.

In MQL5, flags of a single property are displayed in a column under the name of a described property, for example:

```
The flags of allowed order types:
 - Market order (Yes)
 - Limit order (Yes)
 - Stop order (Yes)
 - Stop limit order (Yes)
 - StopLoss (Yes)
 - TakeProfit (Yes)
 - Close by (Yes)
```

In MQL4, these properties are displayed in a single string:

```
The flags of allowed order types: Market order (Yes); Limit order (Yes); Stop order (Yes); Stop limit order (No); StopLoss (Yes); TakeProfit (Yes); Close by (Yes)
```

This is due to the fact that in MQL4 the [Print ()](https://www.mql5.com/en/docs/common/print) function does not
accept the "\\n" line break codes. Therefore, the methods feature separate formatting for MQL5 and MQL4 using conditional compilation
directives.

**Below is the service method returning a normalized price considering symbol properties:**

```
//+------------------------------------------------------------------+
//| Return a normalized price considering symbol properties          |
//+------------------------------------------------------------------+
double CSymbol::NormalizedPrice(const double price) const
  {
   double tsize=this.TradeTickSize();
   return(tsize!=0 ? ::NormalizeDouble(::round(price/tsize)*tsize,this.Digits()) : ::NormalizeDouble(price,this.Digits()));
  }
//+------------------------------------------------------------------+
```

The method normalizes the price considering the minimum price change.

Finally, there are two methods:

**method of updating all symbol object**
**properties that may change in the future and the method of updating**
**symbol's quote data:**

```
//+------------------------------------------------------------------+
//| Update all symbol data that may change                           |
//+------------------------------------------------------------------+
void CSymbol::Refresh(void)
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
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG)]                = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_LONG);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT)]               = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_SHORT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_STOP)]                = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_STOP);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LIMIT)]               = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_LIMIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_STOPLIMIT)]           = ::SymbolInfoDouble(this.m_symbol_name,SYMBOL_MARGIN_STOPLIMIT);
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
  }
//+------------------------------------------------------------------+
//| Update quote data by symbol                                      |
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

Here all is simple: the required symbol properties are filled in from its data anew.

Both methods are meant for
obtaining relevant data on symbol properties. The Refresh() method should be called right before obtaining the required data, while the
RefreshRates() method is constantly called in the timer for all symbol objects within the collection list we are to deal with later.

This concludes the development of the abstract symbol object methods.

**Now we need to make some additions to the ToMQL4.mqh file where we set all the necessary enumerations and macro substitutions for**
**error-free compilation in MQL4.**

In the symbol object class, we used returning MQL5 error codes.
Let's add them to let MQL4 know about their limitations:

```
//+------------------------------------------------------------------+
//|                                                       ToMQL4.mqh |
//|              Copyright 2017, Artem A. Trishkin, Skype artmedia70 |
//|                         https://www.mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Artem A. Trishkin, Skype artmedia70"
#property link      "https://www.mql5.com/en/users/artmedia70"
#property strict
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| Error codes                                                      |
//+------------------------------------------------------------------+
#define ERR_SUCCESS                       (ERR_NO_ERROR)
#define ERR_MARKET_UNKNOWN_SYMBOL         (ERR_UNKNOWN_SYMBOL)
//+------------------------------------------------------------------+
```

We also used the **flags responsible for various symbol order modes**. Let's specify them as well:

```
//+------------------------------------------------------------------+
//| Flags of allowed order expiration modes                          |
//+------------------------------------------------------------------+
#define SYMBOL_EXPIRATION_GTC             (1)
#define SYMBOL_EXPIRATION_DAY             (2)
#define SYMBOL_EXPIRATION_SPECIFIED       (4)
#define SYMBOL_EXPIRATION_SPECIFIED_DAY   (8)
//+------------------------------------------------------------------+
//| Flags of allowed order filling modes                             |
//+------------------------------------------------------------------+
#define SYMBOL_FILLING_FOK                (1)
#define SYMBOL_FILLING_IOC                (2)
//+------------------------------------------------------------------+
//| Flags of allowed order types                                     |
//+------------------------------------------------------------------+
#define SYMBOL_ORDER_MARKET               (1)
#define SYMBOL_ORDER_LIMIT                (2)
#define SYMBOL_ORDER_STOP                 (4)
#define SYMBOL_ORDER_STOP_LIMIT           (8)
#define SYMBOL_ORDER_SL                   (16)
#define SYMBOL_ORDER_TP                   (32)
#define SYMBOL_ORDER_CLOSEBY              (64)
//+------------------------------------------------------------------+
```

**Add the enumerations absent in MQL4:**

```
//+------------------------------------------------------------------+
//| Prices a symbol chart is based on                                |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_CHART_MODE
  {
   SYMBOL_CHART_MODE_BID,                 // Bars are based on Bid prices
   SYMBOL_CHART_MODE_LAST                 // Bars are based on Last prices
  };
//+------------------------------------------------------------------+
//| Lifetime of pending orders and                                   |
//| placed StopLoss/TakeProfit levels                                |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_ORDER_GTC_MODE
  {
   SYMBOL_ORDERS_GTC,                     // Pending orders and Stop Loss/Take Profit levels are valid for an unlimited period until their explicit cancellation
   SYMBOL_ORDERS_DAILY,                   // At the end of the day, all Stop Loss and Take Profit levels, as well as pending orders are deleted
   SYMBOL_ORDERS_DAILY_EXCLUDING_STOPS    // At the end of the day, only pending orders are deleted, while Stop Loss and Take Profit levels are preserved
  };
//+------------------------------------------------------------------+
//| Option types                                                     |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_OPTION_MODE
  {
   SYMBOL_OPTION_MODE_EUROPEAN,           // European option may only be exercised on a specified date
   SYMBOL_OPTION_MODE_AMERICAN            // American option may be exercised on any trading day or before expiry
  };
#define SYMBOL_OPTION_MODE_NONE     (2)   // Option type absent in MQL4
//+------------------------------------------------------------------+
//| Right provided by an option                                      |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_OPTION_RIGHT
  {
   SYMBOL_OPTION_RIGHT_CALL,              // A call option gives you the right to buy an asset at a specified price
   SYMBOL_OPTION_RIGHT_PUT                // A put option gives you the right to sell an asset at a specified price
  };
#define SYMBOL_OPTION_RIGHT_NONE    (2)   // No option - no right
//+------------------------------------------------------------------+
```

Use macro substitutions to set additional properties for
options. These properties indicate the absence of option type and right. We are going to use them in MQL4 when returning value of these symbol
object properties.

The following two enumerations feature no sequence of constant values in MQL5 and MQL4.

Therefore, I have swapped the sequences of specifying constants included in the enumerations. The appropriate values for MQL5 and MQL4 are set in the
comments to constant values. Their sequence here is set to match the values in MQL4 to return correct values:

```
//+------------------------------------------------------------------+
//| Symbol margin calculation method                                 |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_CALC_MODE
  {
   SYMBOL_CALC_MODE_FOREX,                // (MQL5 - 0, MQL4 - 0) Forex mode
   SYMBOL_CALC_MODE_CFD,                  // (MQL5 - 3, MQL4 - 1) CFD mode
   SYMBOL_CALC_MODE_FUTURES,              // (MQL5 - 2, MQL4 - 2) Futures mode
   SYMBOL_CALC_MODE_CFDINDEX,             // (MQL5 - 4, MQL4 - 3) CFD index mode
   SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE,    // (MQL5 - 1, MQL4 - N) Forex No Leverage mode
   SYMBOL_CALC_MODE_CFDLEVERAGE,          // CFD Leverage mode
   SYMBOL_CALC_MODE_EXCH_STOCKS,          // Exchange mode
   SYMBOL_CALC_MODE_EXCH_FUTURES,         // Futures mode
   SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS,   // FORTS Futures mode
   SYMBOL_CALC_MODE_EXCH_BONDS,           // Exchange Bonds mode
   SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX,     // Exchange MOEX Stocks mode
   SYMBOL_CALC_MODE_EXCH_BONDS_MOEX,      // Exchange MOEX Bonds mode
   SYMBOL_CALC_MODE_SERV_COLLATERAL       // Collateral mode
  };
//+------------------------------------------------------------------+
//| Swap charging methods during a rollover                          |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_SWAP_MODE
  {
   SYMBOL_SWAP_MODE_POINTS,               // (MQL5 - 1, MQL4 - 0) Swaps are charged in points
   SYMBOL_SWAP_MODE_CURRENCY_SYMBOL,      // (MQL5 - 2, MQL4 - 1) Swaps are charged in money in symbol base currency
   SYMBOL_SWAP_MODE_INTEREST_OPEN,        // (MQL5 - 6, MQL4 - 2) Swaps are charged as the specified annual interest from the open price of position
   SYMBOL_SWAP_MODE_CURRENCY_MARGIN,      // (MQL5 - 3, MQL4 - 3) Swaps are charged in money in margin currency of the symbol
   SYMBOL_SWAP_MODE_DISABLED,             // (MQL5 - 0, MQL4 - N) No swaps
   SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT,     // Swaps are charged in money, in client deposit currency
   SYMBOL_SWAP_MODE_INTEREST_CURRENT,     // Swaps are charged as the specified annual interest from the instrument price at calculation of swap
   SYMBOL_SWAP_MODE_REOPEN_CURRENT,       // Swaps are charged by reopening positions by the close price
   SYMBOL_SWAP_MODE_REOPEN_BID            // Swaps are charged by reopening positions by the current Bid price
  };
//+------------------------------------------------------------------+
```

All symbol object data is now created.

Since we introduced the error code returned from the classes to the Engine library main object, we need to add the CEngine class variable member
for storing the error code. Open \\MQL5\\Include\\DoEasy\

**Engine.mqh** and make the necessary changes in it.

**In the private class section, declare**
**the variable for storing error codes:**

```
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CArrayObj            m_list_counters;                 // List of timer counters
   int                  m_global_error;                  // Global error code
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_tester;                     // Flag of working in the tester
```

In the initialization list of the **class constructor**, initialize the
error code. Next, in the blocks for creating the millisecond timer for
MQL5 and MQL4, add
the last error code to the variable:

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),m_last_trade_event(TRADE_EVENT_NO_EVENT),m_last_account_event(ACCOUNT_EVENT_NO_EVENT),m_global_error(ERR_SUCCESS)
  {
   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_ORD_COUNTER_ID,COLLECTION_ORD_COUNTER_STEP,COLLECTION_ORD_PAUSE);
   this.CreateCounter(COLLECTION_ACC_COUNTER_ID,COLLECTION_ACC_COUNTER_STEP,COLLECTION_ACC_PAUSE);

   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_is_tester=::MQLInfoInteger(MQL_TESTER);
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

Now we need to include the file of a newly created class to
the main object of the library (temporarily — only for the current check):

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
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Services\TimerCounter.mqh"
#include "Objects\Symbols\Symbol.mqh"
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
```

Now all is ready for testing the symbol object.

But first, I want to draw your attention to a flaw I left unattended in the [previous \\
article](https://www.mql5.com/en/articles/6995) when creating the method returning an account event by its number in the list — in the **CAccountsCollection** account
collection class.

The current method version looks as follows:

```
//+------------------------------------------------------------------+
//| Return the account event by its number in the list               |
//+------------------------------------------------------------------+
ENUM_ACCOUNT_EVENT CAccountsCollection::GetEvent(const int shift=WRONG_VALUE)
  {
   int total=this.m_list_changes.Total();
   if(total==0)
      return ACCOUNT_EVENT_NO_EVENT;
   int index=(shift<0 || shift>total-1 ? total-1 : total-shift-1);
   int event=this.m_list_changes.At(index);
   return ENUM_ACCOUNT_EVENT(event!=NULL ? event : ACCOUNT_EVENT_NO_EVENT);
  }
//+------------------------------------------------------------------+
```

Here is what I wrote about returning events from the method:

The events in the list of account property changes are located in the order they were added — the very first one is located at index 0, while
the very last one is located at (list\_size-1) index. However, we want to let users to obtain a desired event as in a time series — the zero
index should contain the very last event. To achieve this, the method features the index calculation: index = (list\_size -
desired\_event\_number-1). In this case,

if we pass 0
or -1, the last event in the list is returned; if 1, the last
but one, if a number exceeds the list size, the **last** event
is returned.

Here we have a logical inconsistency: if we pass 0 to the method, we get the last event. If we pass the value exceeding the array size, the logic
dictates we should return the first event instead of the last one. It would be logical to pass either 0 or a negative value (by default) to
receive the last event and then go in order: 1 — last but one, 2 — last but two, etc, like in a timeseries.

If we do not know the array size and pass a knowingly large number, then we should expect receiving the farthest value in time — the first
one, while the method returns the last one. Let's fix this illogical behavior:

```
//+------------------------------------------------------------------+
//| Return the account event by its number in the list               |
//+------------------------------------------------------------------+
ENUM_ACCOUNT_EVENT CAccountsCollection::GetEvent(const int shift=WRONG_VALUE)
  {
   int total=this.m_list_changes.Total();
   if(total==0)
      return ACCOUNT_EVENT_NO_EVENT;
   int index=(shift<=0 ? total-1 : shift>total-1 ? 0 : total-shift-1);
   int event=this.m_list_changes.At(index);
   return ENUM_ACCOUNT_EVENT(event!=NULL ? event : ACCOUNT_EVENT_NO_EVENT);
  }
//+------------------------------------------------------------------+
```

Here: if 0 or -1 is passed, return the last event,
if the value exceeding the array size is passed, return the first event. The
index of a returned event is calculated in other cases.

### Testing symbol object

To test the symbol object, we will take the [EA from the previous \\
article](https://www.mql5.com/en/articles/6995#node03) and save it under the name **TestDoEasyPart14.mq5**.

We will check the symbol object in the OnInit() handler. To do this, simply add the following code
strings to the end of OnInit():

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal,
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

//--- Check and remove remaining EA graphical objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);

//--- Set CTrade trading class parameters
#ifdef __MQL5__
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
#endif
//--- Fast check of a symbol object
   string smb=Symbol();
   CSymbol* sy=new CSymbol(SYMBOL_STATUS_FX,smb);
   if(sy!=NULL)
     {
      sy.Refresh();
      sy.RefreshRates();
      sy.Print();
      delete sy;
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Here we create a new symbol object. Since we are going to assign the symbol to a certain group in the next article, simply pass the "Forex symbol"
and symbol name to the constructor.

If the object is created, update all its data, update quote data, display all symbol object properties in the journal and delete the
object to avoid a memory leak when the test EA completes its work.

If we now try to compile the EA, we will get the error of accessing the protected class constructor:

```
'CSymbol::CSymbol' - cannot access protected member function    TestDoEasyPart14.mq5    131     20
   see declaration of 'CSymbol::CSymbol'        Symbol.mqh      39      22
1 error(s), 0 warning(s)                2       1
```

Go to the Symbol.mqh file and move the protected parametric constructor **temporarily** from the protected section of the class to the public
one:

```
//+------------------------------------------------------------------+
//| Abstract symbol class                                            |
//+------------------------------------------------------------------+
class CSymbol : public CObject
  {
private:
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
//--- Reset all symbol object data
   void              Reset(void);
public:
//--- Default constructor
                     CSymbol(void){;}
                     CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name);
protected:
//--- Protected parametric constructor

//--- Get and return integer properties of a selected symbol from its parameters
```

Now all is compiled with no issues.

Launch the EA on a symbol chart in MetaTrader 5.

**All symbol object properties are sent to the journal:**

```
Account 18222304: Artyom Trishkin (MetaQuotes Software Corp. 10000.00 RUR, 1:100, Demo account MetaTrader 5)
============= Beginning of the parameter list:: "EURUSD" (Euro vs US Dollar) ==================
Status: Forex symbol
Custom symbol: No
The price type used for generating bars: Bars are based on Bid prices
The symbol under this name exists: Yes
The symbol is selected in Market Watch: Yes
The symbol is displayed in Market Watch: Yes
The number of deals in the current session: 0
The total number of Buy orders at the moment: 0
The total number of Sell orders at the moment: 0
Last deal volume: 0
Maximum volume within a day: 0
Minimum volume within a day: 0
Latest quote time: 2019.06.17 15:37:13.016
Number of decimal places: 5
Number of decimal places for a lot: 2
Spread in points: 10
Floating spread: Yes
Maximum number of orders displayed in the Depth of Market: 10
Contract price calculation method: Forex mode
Order execution type: No trade restrictions
Symbol trading start date: (Not set)
Symbol trading end date: (Not set)
Minimum distance in points from the current close price for setting Stop orders: 0
Freeze distance for trading operations: 0
Deal execution mode: Instant execution
Swap calculation model: Swaps are charged in points
Triple-day swap: Wednesday
Calculating hedging margin using the larger leg: No
Flags of allowed order expiration modes:
 - Unlimited (Yes)
 - Valid till the end of the day (Yes)
 - Time is specified in the order (Yes)
 - Date is specified in the order (Yes)
Flags of allowed order filling modes:
 - Return (Yes)
 - Fill or Kill (Yes)
 - Immediate or Cancel (No)
The flags of allowed order types:
 - Market order (Yes)
 - Limit order (Yes)
 - Stop order (Yes)
 - Stop limit order (Yes)
 - StopLoss (Yes)
 - TakeProfit (Yes)
 - CloseBy order (Yes)
StopLoss and TakeProfit orders lifetime: Pending orders and Stop Loss/Take Profit levels are valid for an unlimited period until their explicit cancellation
Option type: European option may only be exercised on a specified date
Option right: A call option gives you the right to buy an asset at a specified price
Background color of the symbol in Market Watch: (Not set)
------
Bid price: 1.12411
The highest Bid price of the day: 1.12467
The lowest Bid price of the day: 1.12033
Ask price: 1.12421
The highest Ask price of the day: 1.12477
The lowest Ask price of the day: 1.12043
The last deal price: 0.00000
The highest Last price of the day: 0.00000
The lowest Last price of the day: 0.00000
Volume of the day: 0.00
Maximum Volume of the day: 0.00
Minimum Volume of the day: 0.00
Option execution price: 0.00000
Point value: 0.00001
Calculated tick value for a position: 64.22
Calculated tick value for a winning position: 64.22
Calculated tick value for a losing position: 64.23
Minimum price change: 0.00001
Trade contract size: 100000.00
Accrued interest: 0.00
Face value: 0.00
Liquidity rate: 0.00
Minimum volume for a deal: 0.01
Maximum volume for a deal: 500.00
Minimum volume change step for a deal: 0.01
Maximum acceptable total volume of an open position and pending orders in one direction: 0.00
Long swap value: -0.70
Short swap value: -1.00
Initial margin: 0.00
Maintenance margin for an instrument: 0.00
Margin requirement applicable to long positions: 0.00
Margin requirement applicable to short positions: 0.00
Margin requirement applicable to Stop orders: 0.00
Margin requirement applicable to Limit orders: 0.00
Margin requirement applicable to Stop Limit orders: 0.00
The total volume of deals in the current session: 0.00
The total turnover in the current session: 0.00
The total volume of open positions: 0.00
The total volume of Buy orders at the moment: 0.00
The total volume of Sell orders at the moment: 0.00
The open price of the session: 0.00000
The close price of the session: 0.00000
The average weighted price of the session: 0.00000
The settlement price of the current session: 0.00000
The minimum allowable price value for the session: 0.00000
The maximum allowable price value for the session: 0.00000
Size of a contract or margin for one lot of hedged positions: 100000.00
------
Symbol name: EURUSD
The name of the underlaying asset for a derivative symbol: (Not set)
The base currency of an instrument: "EUR"
Profit currency: "USD"
Margin currency: "EUR"
The source of the current quote: (Not set)
Symbol description: "Euro vs US Dollar"
The formula used for custom symbol pricing: (Not set)
The name of a trading symbol in the international system of securities identification numbers: (Not set)
The address of the web page containing symbol information: "http://www.google.com/finance?q=EURUSD"
Path in the symbol tree: "Forex\EURUSD"
================== End of the parameter list: EURUSD" (Euro vs US Dollar) ==================
```

Launch the EA on a symbol chart in MetaTrader 4.

**All symbol object properties are sent to the journal:**

```
Account 49610941: Artyom Trishkin (MetaQuotes Software Corp. 5000000.00 USD, 1:100, Hedge, Demo account MetaTrader 4)
============= Beginning of the parameter list: "EURUSD" (Euro vs US Dollar) ==================
Status: Forex symbol
Custom symbol: No
The price type used for generating bars: Bars are based on Bid prices
The symbol under this name exists: Yes
The symbol is selected in Market Watch: Yes
The symbol is displayed in Market Watch: Yes
The number of deals in the current session: Property not supported in MQL4
The total number of Buy orders at the moment: Property not supported in MQL4
The total number of Sell orders at the moment: Property not supported in MQL4
Last deal volume: Property not supported in MQL4
Maximum volume within a day: Property not supported in MQL4
Minimum volume within a day: Property not supported in MQL4
Latest quote time: 2019.06.17 19:40:41.000
Number of decimal places: 5
Number of decimal places for a lot: 2
Spread in points: 20
Floating spread: Yes
Maximum number of orders displayed in the Depth of Market: Property not supported in MQL4
Contract price calculation method: Forex mode
Order execution type: No trade restrictions
Symbol trading start date: (Not set)
Symbol trading end date: (Not set)
Minimum distance in points from the current close price for setting Stop orders: 8
Freeze distance for trading operations: 0
Deal execution mode: Instant execution
Swap calculation model: Swaps are charged in points
Triple-day swap: Wednesday
Calculating hedging margin using the larger leg: No
Flags of allowed order expiration modes: Unlimited (Yes); Valid till the end of the day (No); Time is specified in the order (No); Date is specified in the order (No)
Flags of allowed order filling modes: Return (Yes); Fill or Kill (No); Immediate or Cancel (No)
Flags of allowed order types: Market order (Yes); Limit order (Yes); Stop order (Yes); Stop limit order (No); StopLoss (Yes); TakeProfit (Yes); Close by (Yes)
StopLoss and TakeProfit orders lifetime: Pending orders and Stop Loss/Take Profit levels are valid for an unlimited period until their explicit cancellation
Option type: Property not supported in MQL4
Option right: Property not supported in MQL4
The color of the background used for the symbol in Market Watch: Property not supported in MQL4
------
Bid price: 1.12328
The highest Bid price of the day: 1.12462
The lowest Bid price of the day: 1.12029
Ask price: 1.12348
The highest Ask price of the day: Property not supported in MQL4
The lowest Ask price of the day: Property not supported in MQL4
Last deal price: Property not supported in MQL4
The highest Last price of the day: Property not supported in MQL4
The lowest Last price of the day: Property not supported in MQL4
Volume of the day: Property not supported in MQL4
Maximum Volume of the day: Property not supported in MQL4
Minimum Volume of the day: Property not supported in MQL4
Option execution price: Property not supported in MQL4
Point value: 0.00001
Calculated tick value for a position: 1.00
Calculated tick value for a winning position: Property not supported in MQL4
Calculated tick value for a losing position: Property not supported in MQL4
Minimum price change: 0.00001
Trade contract size: 100000.00
Accrued interest: Property not supported in MQL4
Face value: Property not supported in MQL4
Liquidity rate: Property not supported in MQL4
Minimum volume for a deal: 0.01
Maximum volume for a deal: 100000.00
Minimum volume change step for a deal: 0.01
Maximum acceptable total volume of an open position and pending orders in one direction: Property not supported in MQL4
Long swap value: 0.33
Short swap value: -1.04
Initial margin: 0.00
Maintenance margin for an instrument: 0.00
Margin requirement applicable to long positions: Property not supported in MQL4
Margin requirement applicable to short positions: Property not supported in MQL4
Margin requirement applicable to Stop orders: Property not supported in MQL4
Margin requirement applicable to Limit orders: Property not supported in MQL4
Margin requirement applicable to Stop Limit orders: Property not supported in MQL4
The total volume of deals in the current session: Property not supported in MQL4
The total turnover in the current session: Property not supported in MQL4
The total volume of open positions: Property not supported in MQL4
The total volume of Buy orders at the moment: Property not supported in MQL4
EURUSD,H4: The total volume of Sell orders at the moment: Property not supported in MQL4
The open price of the session: Property not supported in MQL4
The close price of the session: Property not supported in MQL4
The average weighted price of the session: Property not supported in MQL4
The settlement price of the current session: Property not supported in MQL4
The minimum allowable price value for the session: Property not supported in MQL4
The maximum allowable price value for the session: Property not supported in MQL4
Size of a contract or margin for one lot of hedged positions: 50000.00
------
Symbol name: EURUSD
The name of the underlaying asset for a derivative symbol: ": Property not supported in MQL4"
The base currency of an instrument: "EUR"
Profit currency: "USD"
Margin currency: "EUR"
The source of the current quote: ": Property not supported in MQL4"
Symbol description: "Euro vs US Dollar"
The formula used for custom symbol pricing: ": Property not supported in MQL4"
The name of a trading symbol in the international system of securities identification numbers: ": Property not supported in MQL4"
The address of the web page containing symbol information: ": Property not supported in MQL4"
Path in the symbol tree: "Forex\EURUSD"
================== End of the parameter list: EURUSD" (Euro vs US Dollar) ==================
```

Unlike launching in MetaTrader 5, not all properties are supported here. Therefore, there are appropriate messages in the journal. The list of
flags is displayed in a single line, which has already been mentioned above.

In the next article, no description strings for unsupported properties will be displayed when creating derived objects. The flags
indicating the object's support of certain properties will be set for each derived object instead. The flags are to be set for each property.

### What's next?

In the next article, we will start developing the symbol collection class that will allow users to easily search for data, as well as sort and
compare symbols from the collection list.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7014#node00)

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

[Part 12. "Account" object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part \\
13\. Account object events](https://www.mql5.com/en/articles/6995)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7014](https://www.mql5.com/ru/articles/7014)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7014.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7014/mql5.zip "Download MQL5.zip")(153.66 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7014/mql4.zip "Download MQL4.zip")(153.66 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/322733)**
(35)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
7 Jul 2019 at 21:08

**\_SERG\_:**

Commented out the line from Datas.mqh,  recompiled, error: 'CSymbol::CSymbol' - cannot access protected member [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 Documentation: Predefined Macro Substitutions") TestDoEasyPart14.mq413120, downloaded part 15, updated Include DoEasy from the archive with part 15, recompiled, error again and there again.

Compiled the 15th part there everything is normal. By the way it is for MT4.

I don't go into details yet, just observing.

I described above the reason. To check, download the library of this part, but name the folder DoEasyPart14. Then in the EA from this part, in line 10 connect the library from its new location:

```
#include <DoEasyPart14\Engine.mqh>
```

And everything will compile and work as written in this article. Both in MetaTrader4 and MetaTrader5.

The error you mentioned is not present here - I have already explained why. Let me say it again: this part is one step in creating a collection of symbols (already published part 15) and tracking symbol events (which have already been prepared and article #16 is being written). And you, having fully loaded the library from part 15, are trying to compile the intermediate result - creation of one symbol and checking if it works correctly - which is described in this article.

In other words - the test EAs attached to a particular article are compiled and work exactly with the version of the library from the same article.

I am describing the process of library development, not giving you an already polished and finished product from CodeBase. This is educational and creative material, not dry code for self-study.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
21 Sep 2020 at 16:23

Hello Artyom -- is there an easy way to extract or compute the **average spread** for a given symbol using your library, or it is something you recommend I code externally myself?


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
21 Sep 2020 at 18:47

**Dima Diall :**

Hello Artyom -- is there an easy way to extract or compute the **average spread**  for a given symbol using your library, or it is something you recommend I code externally myself?

The average spread of a bar is recorded by the terminal in the parameters of each bar. It can be found [by requesting](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_trade_execution "MQL5 documentation: Symbol properties") bar data from MqlRates. The library contains this data for each bar.

Add up all the spreads of all bars in the sample under study and divide by their number.

I will not do automatic determination of the average spread for a symbol. Because it will slow down the library, and this is not a frequent need. You can easily implement this yourself if you wish.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
22 Sep 2020 at 01:23

**Artyom Trishkin:**

The average spread of a bar is recorded by the terminal in the parameters of each bar. It can be found [by requesting](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_trade_execution "MQL5 documentation: Symbol properties") bar data from MqlRates. The library contains this data for each bar.

Add up all the spreads of all bars in the sample under study and divide by their number.

That's perfect, thank you!

Do you know how reliable is the spread data for each bar when in **testing mode**? Is this consistent across different brokers, or quality of spread data can vary?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
22 Sep 2020 at 01:25

**Dima Diall :**

That's perfect, thank you!

Do you know how reliable is the spread data for each bar when in **testing mode** ? Is this consistent across different brokers, or quality of spread data can vary?

No, unfortunately I do not know.

![Optimization management (Part II): Creating key objects and add-on logic](https://c.mql5.com/2/36/mql5-avatar-opt_control__1.png)[Optimization management (Part II): Creating key objects and add-on logic](https://www.mql5.com/en/articles/7059)

This article is a continuation of the previous publication related to the creation of a graphical interface for optimization management. The article considers the logic of the add-on. A wrapper for the MetaTrader 5 terminal will be created: it will enable the running of the add-on as a managed process via C#. In addition, operation with configuration files and setup files is considered in this article. The application logic is divided into two parts: the first one describes the methods called after pressing a particular key, while the second part covers optimization launch and management.

![Library for easy and quick development of MetaTrader programs (part XIII): Account object events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__8.png)[Library for easy and quick development of MetaTrader programs (part XIII): Account object events](https://www.mql5.com/en/articles/6995)

The article considers working with account events for tracking important changes in account properties affecting the automated trading. We have already implemented some functionality for tracking account events in the previous article when developing the account object collection.

![Merrill patterns](https://c.mql5.com/2/36/Article_Logo__3.png)[Merrill patterns](https://www.mql5.com/en/articles/7022)

In this article, we will have a look at Merrill patterns' model and try to evaluate their current relevance. To do this, we will develop a tool to test the patterns and apply the model to various data types such as Close, High and Low prices, as well as oscillators.

![Library for easy and quick development of MetaTrader programs (part XII): Account object class and collection of account objects](https://c.mql5.com/2/36/MQL5-avatar-doeasy__7.png)[Library for easy and quick development of MetaTrader programs (part XII): Account object class and collection of account objects](https://www.mql5.com/en/articles/6952)

In the previous article, we defined position closure events for MQL4 in the library and got rid of the unused order properties. Here we will consider the creation of the Account object, develop the collection of account objects and prepare the functionality for tracking account events.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=xkwccjhxbayqqatdjtihjdpzvkqqargs&ssn=1769186413847183318&ssn_dr=0&ssn_sr=0&fv_date=1769186413&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7014&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XIV)%3A%20Symbol%20object%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918641334712263&fz_uniq=5070470744548251245&sv=2552)

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