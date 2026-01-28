---
title: Universal Expert Advisor: Accessing Symbol Properties (Part 8)
url: https://www.mql5.com/en/articles/3270
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:17:30.397844
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/3270&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071723410414841144)

MetaTrader 5 / Examples


### Introduction

Some time has passed since the publication of the previous part of the article devoted to the CStrategy trading engine. This time was needed in order to realize the CStrategy development path from a small auxiliary trade library to a full-featured trading complex, which includes the most frequently used tools for creating a full-fledged trading strategy. This time helped to understand ways for the further development of CStrategy. The practical use of CStrategy during this time also helped to reveal some drawbacks in the latest versions of the engine. Correction of these drawbacks gave birth to new articles in the "Universal Trade Expert" series. In the current eighth part, we will discuss work with trading instruments through the object-oriented CStrategy class.

### Overview of previous CStrategy versions

The Expert Advisor trading environment of is diverse. It includes account information, price data, functions for working with time, and information about trading symbols available in the terminal. Most of this information is available in the trading symbol related functions, such as receiving current quotes and working with symbol properties. As a rule, all trading Expert Advisors actively work with price data. They use the latest price data to calculate a pattern or a trading signal, based on which they perform a trade. In order to provide a proper generation of a trade order, they also use information about the properties of the current symbol, such as the minimum volume of a trade or the freeze level, i.e. the range from the current price inside which pending orders cannot be placed.

This data should be easily accessible and always "at hand". Was it so in the previous versions of CStrategy? Let's refer to the history in order to find it out. Below is the description of how the earlier versions of the engine worked with a trading symbol. In the third part of the article, we discussed accessing quotes through a traditional indexer \[\]. Some auxiliary classes were included in CStrategy, such as COpen, CHigh, CLow, CClose, CVolume, CTime. Each of them returned an appropriate value at the requested index. Thus information about the current symbol could be conveniently received in the Expert Advisor code. For example, the current bar closing price could be obtained using the following simple code:

```
...
double close = Close[0];
...
```

However, access to prices in the OHLC format was not enough, and additional Ask(), Bid(), Last() methods were added. However, more methods were required, for example FreezeLevel() to get the basic information on the current instrument. The size of the base CStrategy class began to grow. The large number of methods inside CStrategy started to confuse. The main difficulties began at the attempt to create an Expert Advisor trading multiple symbols. CStrategy is formally a multi-symbol tool. It means that it can be used to create multiple Expert Advisors trading different symbols independently, or one Expert Advisor trading two or more financial instruments. But the latter case was hard to implement, because it required to reconfigure timeseries classes on the fly, by alternately setting different working symbols:

```
string symbol1 = "EURUSD";
string symbol2 = "GBPUSD";
Close.Symbol(symbol1);
double close_eurusd = Close[0];
Close.Symbol(symbol2);
double close_gbpusd = Close[0];
...
```

These difficulties lead to a conclusion that due to a large amount of information on the working symbol, CStrategy directly cannot implement it. The CStrategy class performs complex work while arranging a sequence of trading actions, and any additional functionality can spoil code manageability. So, the method for working with the symbol should better be implemented in a separate **CSymbol** class.

### The first acquaintance with the WS object and the CSymbol class

Now instead of separate Ask(), Bid() and Last() methods and additional classes, such as CHigh and CLow, a CStrategy based trading strategy can access the special **WS** object created based on the CSymbol class. This class belongs to the set of CStrategy libraries and includes a number of methods that make it similar to the standard [СSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class. But the class is different. In addition to working with symbol properties, it can get symbol quotes including information about limit orders (Market Depth). The object name WS is an abbreviation of "Working Symbol". The object is available in the strategy code. The use of a short name is convenient. We often need to access various symbol properties, while the two-character abbreviation allows keeping the code compact and expressive.

It was mentioned in the previous parts of the article that before control is passed to an Expert Advisor, the CStrategy trading engine performs a number of initializations of the internal environment objects. It saves the name of the working symbol and the timeframe, and also creates classes that track the arrival of new events (default events are a new tick and a new bar). It also enables logging and sets operation mode flags. Also, the WS object of the CSymbol class is initialized. Its internal structure is quite simple. It contains two internal fields: the symbol and its timeframe, as well as special objects enabling access to symbol quotes. The WS object is initialized in the InitSeries method. Knowing the working symbol and timeframe of the Expert Advisor, we can easily initialize it:

```
CStrategy::CStrategy(void)
{
   WS.InitSeries(ExpertSymbol(), Timeframe());
}
```

Once the object is initialized, you can use it to get the required property of the symbol. For example, in order to get the high price of the current bar, write the following:

```
double max = WS.High[0];
```

The WS object is provided with a number of additional properties that make it a convenient and self-sufficient tool for a direct use in calculations. Let's consider a common case: you want to place a BuyStop order just above the previous bar's High price. Suppose we trade EURUSD and want to place a stop order at a distance of three five-digit points from the previous bar's High. We need to write the following code:

```
void CMovingAverage::InitBuy(const MarketEvent &event)
{
   ...
   Trade.BuyStop(1.0, WS.High[1] + WS.StepToPrice(3), WS.Name());
   ...
}
```

This single line code includes a lot of actions:

- receiving the extreme value of the previous bar (WS.High\[1\]);
- multiplying the value of one point by three to get the required distance of three points (WS.StepToPrice(3));
- adding the resulting price distance to the high price (WS.High\[1\] + WS.StepToPrice(3));
- sending a BuyStop order with the trigger price at the resulting value, while the traded instrument is equal to the current symbol (WS.Name()).

The StepToPrice method may seem to be quite different from the name system adopted in MetaTrader. In other trading platforms, a price step is a minimum price change. Its equivalent notion in MetaTrader is SYMBOL\_TRADE\_TICK\_SIZE. This name can easily be confused with the tick size or value SYMBOL\_TRADE\_TICK\_VALUE, so CSymbol uses a different name for this parameter. Nevertheless, most other names of CSymbol methods coincide with MQL5 system modifiers and methods, although they are not always identical (e.g. StepToPrice). The main goal of CSymbol is to provide a simple and intuitive set of methods for getting the full information about a trading instrument.

### The structure of the CSymbol class. A comparative table of methods

In MetaTrader, a trading symbol is provided with a large set of properties. First of all, all properties can be conditionally divided into integer, real and string values. Integer properties include bool values, system modifiers in the form of enumerations (enum), date and time (datetime) and integer properties (int and long). Real properties include various fractional values ​​(double). String properties include the properties that return string values, such as the name of the symbol, its string description, etc. In addition, a symbol may have properties that are specific to a certain market segment. For example, additional properties of the current trading session are available for FORTS symbols. Options also have specific unique properties. The CSymbol class defines the properties of the current FORTS trading session in the additional internal SessionInfo class. The rest of the properties are not separated based on their types. They are available "as is" in the form of methods with appropriate names.

Moreover, the CSymbol class contains additional collections that enable access to symbol quotes. For example, the publicly defined COpen, CHigh, CLow, CClose, CVolume are used to access OHLCV series, while the market depth can be accessed using the special CMarketWatch class. The detailed description of CMarketWatch is provided in the article: ["MQL5 Cookbook: Implementing Your Own Depth of Market"](https://www.mql5.com/en/articles/1793). In addition to methods and indexing classes with appropriate names, such as CClose, the CSymbol class contains some methods that do not have analogues in the SymbolInfo class. Let us describe them in more detail.

**Available**: the method returns true if a symbol with the given name exists in the terminal. If such a symbol is not found, false is returned.

**IndexByTime**: returns the index of the bar that corresponds to the specified time. For example, in the following code the value of 1 is assigned to the 'index' variable:

```
int index = WS.IndexByTime(WS.Time[1]);
// index = 1;
```

This method is convenient to use if we know the time and we want to get the index of the bar corresponding to this time. Suppose that the trading Expert Advisor must close a position after holding it during _BarsHold_ bars. A code implementing this function may look like this:

```
//+------------------------------------------------------------------+
//| Managing a long position in accordance with the Moving Average   |
//+------------------------------------------------------------------+
void CImpulse::SupportBuy(const MarketEvent &event,CPosition *pos)
{
   int bar_open = WS.IndexByTime(pos.TimeOpen());
   if(bar_open >= BarsHold)
      pos.CloseAtMarket("Exit by time hold");
}
```

**StepToPrice** is the minimum price change value expressed in symbol points. A description of the method purpose was provided earlier.

The full list of CSymbol methods is provided in a table below. The _Description_ field contains the brief method description. In most cases, it matches the official description of a similar symbol property in the corresponding section of the documentation. Some methods are provided with a more suitable description.

The _Return Type_ field shows the type of value returned by a method or collection.

The _MQL5 Function or System Identifier_ field contains the name of an appropriate MQL5 system identifier or function that is used for similar purposes. If a system function is specified, brackets are added at the end of its name, e.g. CopyOpen() or MarketBookGet(). A system modifier is one of the three modifiers that must be specified when calling the SymbolInfoInteger, SymbolInfoDouble or SymbolInfoString function. The modifier must belong to one of three appropriate system enumerations: ENUM\_SYMBOL\_INFO\_INTEGER, ENUM\_SYMBOL\_INFO\_DOUBLE or ENUM\_SYMBOL\_INFO\_STRING. For example, if the SYMBOL\_TRADE\_STOPS\_LEVEL modifier is specified in "MQL5 Function or System Identifier", this means that you should call SymbolInfoInteger to get this property:

```
int stop_level = SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL);
```

The _CSymbol Method Name_ column contains the name of the method that returns the corresponding property. For example, in order to get the day of the week when triple swap is charged, the following method should be called

```
ENUM_DAY_OF_WEEK day = WS.DayOfSwap3x();
```

Here is the table of methods:

| Description | Return Type | MQL5 Function or System Identifier | CSymbol Method Name |
| --- | --- | --- | --- |
| **ACCESS TO SYMBOL'S HISTORIC QUOTES** |  |  |  |
| Getting the open price for the specified bar index, with a preset symbol timeframe | double | CopyOpen() | Open\[\] |
| Getting the high price for the specified bar index, with a preset symbol timeframe | double | CopyHigh() | High\[\] |
| Getting the low price for the specified bar index, with a preset symbol timeframe | double | CopyLow() | Low\[\] |
| Getting the close price for the specified bar index, with a preset symbol timeframe | double | CopyClose() | Close\[\] |
| Getting the volume of the bar with the specified index | double | CopyVolume() | Volume\[\] |
| Getting the Market Depth properties of the symbol, access to level 2 prices | MqlBookInfo | MarketBookGet() | MarketBook |
| **INTEGER SYMBOL PROPERTIES** |  |  |  |
| An indication that the symbol exists in the terminal | bool | No analogues | Available |
| The number of bars for this symbol and timeframe | int | Bars() | BarsTotal |
| The timeframe of the symbol | ENUM\_TIMEFRAMES | Period() | Period |
| An indication that the symbol is selected in Market Watch | bool | SYMBOL\_SELECT | SelectInMarketWatch |
| An indication of the floating spread | bool | SYMBOL\_SPREAD\_FLOAT | SpreadFloat |
| Spread value in points | int | SYMBOL\_SPREAD | Spread |
| Minimum distance in points from the current close price for setting Stop orders | int | SYMBOL\_TRADE\_STOPS\_LEVEL | StopLevel |
| Freeze distance for trading operations (in points) | int | SYMBOL\_TRADE\_FREEZE\_LEVEL | FreezeLevel |
| Flags of allowed order expiration modes | int | SYMBOL\_EXPIRATION\_MODE | FlagsExpirationOrders |
| Flags of allowed order execution modes | int | SYMBOL\_FILLING\_MODE | FlagsExecutionOrders |
| Flags of allowed order types | int | SYMBOL\_ORDER\_MODE | FlagsAllowedOrders |
| Returns the index of the bar whose open time corresponds to the passed argument | int | No analogues | IndexByTime |
| Contract price calculation mode | ENUM\_SYMBOL\_CALC\_MODE | SYMBOL\_TRADE\_CALC\_MODE | CalcContractType |
| Order execution type | ENUM\_SYMBOL\_TRADE\_MODE | SYMBOL\_TRADE\_MODE | ExecuteOrderType |
| Deal execution mode | ENUM\_SYMBOL\_TRADE\_EXECUTION | SYMBOL\_TRADE\_EXEMODE | ExecuteDealsType |
| Swap calculation model | ENUM\_SYMBOL\_SWAP\_MODE | SYMBOL\_SWAP\_MODE | CalcSwapMode |
| Triple-day swap charging day | ENUM\_DAY\_OF\_WEEK | SYMBOL\_SWAP\_ROLLOVER3DAYS | DayOfSwap3x |
| Option type | ENUM\_SYMBOL\_OPTION\_MODE | SYMBOL\_OPTION\_MODE | OptionType |
| Option right (Call/Put) | ENUM\_SYMBOL\_OPTION\_RIGHT | SYMBOL\_OPTION\_RIGHT | OptionRight |
| Time of the last quote | datetime | SYMBOL\_TIME | TimeOfLastQuote |
| Symbol trading start date (usually used for futures) | datetime | SYMBOL\_START\_TIME | StartDate |
| Symbol trading end date (usually used for futures) | datetime | SYMBOL\_EXPIRATION\_TIME | ExpirationDate |
| **PROPERTIES OF THE CURRENT TRADING SESSION OF MOEX FUTURES SYMBOLS** |  |  |  |
| The number of deals in the current session | long | SYMBOL\_SESSION\_DEALS | SymbolInfo.DealsTotal |
| The total number of Buy orders at the moment | long | SYMBOL\_SESSION\_BUY\_ORDERS | SymbolInfo.BuyOrdersTotal |
| The total number of Sell orders at the moment | long | SYMBOL\_SESSION\_SELL\_ORDERS | SymbolInfo.SellOrdersTotal |
| The highest volume during the current trading session | long | SYMBOL\_VOLUMEHIGH | SymbolInfo.HighVolume |
| The lowest volume during the current trading session | long | SYMBOL\_VOLUMELOW | SymbolInfo.LowVolume |
| The highest Bid price of the day | double | SYMBOL\_BIDHIGH | SymbolInfo.BidHigh |
| The highest Ask price of the day | double | SYMBOL\_ASKHIGH | SymbolInfo.AskHigh |
| The lowest Bid price of the day | double | SYMBOL\_BIDLOW | SymbolInfo.BidLow |
| The lowest Ask price of the day | double | SYMBOL\_ASKLOW | SymbolInfo.AskLow |
| The highest Last price of the day | double | SYMBOL\_LASTHIGH | SymbolInfo.LastHigh |
| The lowest Last price of the day | double | SYMBOL\_LASTLOW | SymbolInfo.LastLow |
| The total volume of deals in the current session | double | SYMBOL\_SESSION\_VOLUME | SymbolInfo.VolumeTotal |
| The total turnover in the current session | double | SYMBOL\_SESSION\_TURNOVER | SymbolInfo.TurnoverTotal |
| The total volume of open positions | double | SYMBOL\_SESSION\_INTEREST | SymbolInfo.OpenInterestTotal |
| The total volume of Buy orders at the moment | double | SYMBOL\_SESSION\_BUY\_ORDERS\_VOLUME | SymbolInfo.BuyOrdersVolume |
| The total volume of Sell orders at the moment | double | SYMBOL\_SESSION\_SELL\_ORDERS\_VOLUME | SymbolInfo.SellOrdersVolume |
| The open price of the session | double | SYMBOL\_SESSION\_OPEN | SymbolInfo.PriceSessionOpen |
| The close price of the session | double | SYMBOL\_SESSION\_CLOSE | SymbolInfo.PriceSessionClose |
| The average weighted price of the session | double | SYMBOL\_SESSION\_AW | SymbolInfo.PriceSessionAverage |
| The settlement price of the current session | double | SYMBOL\_SESSION\_PRICE\_SETTLEMENT | SymbolInfo.PriceSettlement |
| The maximum allowable price value for the session | double | SYMBOL\_SESSION\_PRICE\_LIMIT\_MAX | SymbolInfo.PriceLimitMax |
| The minimum allowable price value for the session | double | SYMBOL\_SESSION\_PRICE\_LIMIT\_MIN | SymbolInfo.PriceLimitMin |
| **REAL SYMBOL PROPERTIES** |  |  |  |
| Ask, the best price at which an instrument can be bought | double | SYMBOL\_ASK | Ask |
| Bid, the best price at which an instrument can be sold | double | SYMBOL\_BID | Bid |
| The price at which the last deal was executed | double | SYMBOL\_LAST | Last |
| The minimum price change value multiplied by the passed number of price steps | double | No analogues | StepToPrice |
| The value of one point (tick) | double | SYMBOL\_POINT | PriceStep |
| The value of one point (tick) expressed in the deposit currency | double | SYMBOL\_TRADE\_TICK\_VALUE | TickValue |
| Option execution price | double | SYMBOL\_OPTION\_STRIKE | OptionStrike |
| Trade contract size | double | SYMBOL\_TRADE\_CONTRACT\_SIZE | ContractSize |
| Minimum volume for deal execution | double | SYMBOL\_VOLUME\_MIN | VolumeContractMin |
| Maximum volume for deal execution | double | SYMBOL\_VOLUME\_MAX | VolumeContractMax |
| The minimum volume change step for deal execution | double | SYMBOL\_VOLUME\_STEP | VolumeContractStep |
| The maximum allowed total volume of an open position and pending orders in one direction (either buy or sell) for this symbol. | double | SYMBOL\_VOLUME\_LIMIT | VolumeContractLimit |
| The value of swap charged for holding a long position with the volume of one contract | double | SYMBOL\_SWAP\_LONG | SwapLong |
| The value of swap charged for holding a short position with the volume of one contract | double | SYMBOL\_SWAP\_SHORT | SwapShort |
| The margin required to open a one-lot position | double | SYMBOL\_MARGIN\_INITIAL | MarginInit |
| The margin required to maintain one lot of an open position | double | SYMBOL\_MARGIN\_MAINTENANCE | MarginMaintenance |
| The margin required to maintain one lot of a hedged position | double | SYMBOL\_MARGIN\_HEDGED | MarginHedged |
| **STRING SYMBOL PROPERTIES** |  |  |  |
| Symbol name | string | Symbol() | Name |
| The name of the underlying asset for a derivative symbol | string | SYMBOL\_BASIS | NameBasisSymbol |
| The base currency of an instrument | string | SYMBOL\_CURRENCY\_BASE | NameBasisCurrency |
| Profit currency | string | SYMBOL\_CURRENCY\_PROFIT | NameCurrencyProfit |
| Margin currency | string | SYMBOL\_CURRENCY\_MARGIN | NameCurrencyMargin |
| The source of the current quote | string | SYMBOL\_BANK | NameBank |
| The string description of a symbol | string | SYMBOL\_DESCRIPTION | Description |
| The name of a trading symbol in the international system of securities identification numbers (ISIN) | string | SYMBOL\_ISIN | NameISIN |
| Path in the symbol tree | string | SYMBOL\_PATH | SymbolPath |

### Using multiple symbols at a time

CSymbol is a regular class, therefore you can create an unlimited number of objects of this class within your Expert Advisor. WS is just one of such objects created by the CStrategy engine, and it indicates the working symbol and timeframe of the Expert Advisor. The EA can also create an additional object that provides access to any other symbol. Suppose our EA trades on Moscow Exchange's Derivatives Market and simultaneously tracks two symbols, Si and Brent. We can use two CSymbol objects in the EA code. Let's call them Si and Brent:

```
//+------------------------------------------------------------------+
//|                                                EventListener.mqh |
//|           Copyright 2017, Vasiliy Sokolov, St-Petersburg, Russia |
//|                                https://www.mql5.com/en/users/c-4 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Vasiliy Sokolov."
#property link      "https://www.mql5.com/en/users/c-4"
#include <Strategy\Strategy.mqh>

//+------------------------------------------------------------------+
//| The template of a strategy working with two symbols at a time    |
//+------------------------------------------------------------------+
class CIntRate : public CStrategy
  {
   CSymbol           Si;         // Ruble-dollar
   CSymbol           Brent;      // Brent oil
public:
   virtual void      OnEvent(const MarketEvent& event);
   virtual bool      OnInit();
  };
//+------------------------------------------------------------------+
//| Initializes the ruble and oil symbols                            |
//+------------------------------------------------------------------+
bool CIntRate::OnInit(void)
  {
   Si.InitSeries("Si Splice", Timeframe());
   Brent.InitSeries("BR Splice", Timeframe());
   return true;
  }

//+------------------------------------------------------------------+
//| The price of Brent expressed in rubles                           |
//+------------------------------------------------------------------+
void CIntRate::OnEvent(const MarketEvent &event)
  {
   double brent_in_rub = Brent.Last()*Si.Last()/Si.ContractSize();
  }

//+------------------------------------------------------------------+
```

The Expert Advisor code receives the last prices of Brent futures and the ruble, and then calculates the Brent price formula expressed in rubles. One Si futures contract is equal to $1000, so we need to divide the result by the size of one contract. It is quite a simple operation, because all symbol properties are available in a single class. The rest of the code is also simple and expressive. The main thing is not to forget to initialize the Si and Brent objects in the OnInit method at the EA launch.

### Building an interest rate profile using CSymbol

The last example of using CSymbol that we are going to consider is a little more complicated, and is also more interesting. Futures contracts are known to be traded with some [contango](https://en.wikipedia.org/wiki/Contango "https://en.wikipedia.org/wiki/Contango") in relation to the underlying asset. It means that the future price of a commodity is higher than the spot price. This difference determines the market interest rate on a particular commodity or asset. Let's consider an example with the ruble/dollar futures. Its spot price as of the moment of writing this article is 56.2875 rubles for 1 dollar, and the price of the nearest Si-6.17 futures contract is 56,682 rubles for 1000$ or 56.682 rubles for 1 dollar. So the difference between the spot price and the future price after 30 days (on 16.05.2017, the expiration of Si-6.17 is 30 days) is 0.395 rubles or 39.5 kopeks. That is, the market expects the ruble to depreciate by 39.5 kopeks, i.e. 0.7% of its spot price. We can easily calculate that the 12-month inflation expected by the market is 8.42%. But this is the level of inflation calculated for the nearest futures. If we use Si-9.17 instead of Si-6.17, the inflation will be lower, about 7.8% per annum. By comparing all Si futures with the underlying asset price, we can obtain the _**interest profile**_. This profile will be displayed as a table showing investors' expectations depending on the time. For example, we will know the interest rate for the next 30, 100, 200, 300, 400 and 500 days.

We will need to actively use various symbol properties and the list of symbols in order to calculate all these values. How the interest profile is calculated:

1. The Expert Advisor is loaded on any futures symbol. It analyzes the symbol name and loads all related futures.
2. Each loaded futures symbol represents a CSymbol object that is placed in the list of symbols.
3. When a new tick is received, the Expert Advisor works with the collection of symbols. It finds an underlying asset for each symbol.
4. Then the EA calculates the difference between the price of the selected symbol and the price of its underlying asset. This difference is converted into interest, which is then converted into an annual interest. For this purpose, the remaining lifetime of the futures contract is taken into account.
5. The resulting difference is displayed on the panel as a table row. Each row is displayed as " _Futures name — Days before expiration — Interest rate_".

As can be seen from the description, the algorithm is actually not as simple as it might seem. However, the CStrategy engine and the CSymbol object help to significantly reduce the calculation complexity for the EA. The below code is implemented in the form of an Expert Advisor, although the EA will not perform any trading action. Instead, it will display interest values on the panel. Here is the resulting code:

```
//+------------------------------------------------------------------+
//|                                                EventListener.mqh |
//|           Copyright 2017, Vasiliy Sokolov, St-Petersburg, Russia |
//|                                https://www.mql5.com/en/users/c-4 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Vasiliy Sokolov."
#property link      "https://www.mql5.com/en/users/c-4"
#include <Strategy\Strategy.mqh>
#include <Arrays\ArrayObj.mqh>
#include "Panel.mqh"

//+------------------------------------------------------------------+
//| Interest rate profile                                            |
//+------------------------------------------------------------------+
class CIntRate : public CStrategy
  {
   CArrayObj         Symbols;    // The list of symbols
   CPercentPanel     Panel;      // Panel for displaying the interest rate
   double            BaseRate(CSymbol* fut);
public:
   virtual void      OnEvent(const MarketEvent& event);
   virtual bool      OnInit();
  };
//+-------------------------------------------------------------------+
//| Adds required futures to calculate the interest rate profile      |
//+-------------------------------------------------------------------+
bool CIntRate::OnInit(void)
  {
   string basis = WS.NameBasisSymbol();
   for(int i = 0; i < SymbolsTotal(false); i++)
   {
      string name = SymbolName(i, false);
      int index = StringFind(name, basis, 0);
      if(index != 0)
         continue;
      CSymbol* Fut = new CSymbol(name, Timeframe());
      if(Fut.ExpirationDate() == 0 || Fut.ExpirationDate() < TimeCurrent())
      {
         delete Fut;
         continue;
      }
      string text = "Add new symbol " + Fut.Name() + " in symbols list";
      CMessage* msg = new CMessage(MESSAGE_INFO, __FUNCTION__, text);
      Log.AddMessage(msg);
      Symbols.Add(Fut);
   }
   string text = "Total add symbols " + (string)Symbols.Total();
   CMessage* msg = new CMessage(MESSAGE_INFO, __FUNCTION__, text);
   Log.AddMessage(msg);
   if(Symbols.Total() > 0)
   {
      Panel.Show();
   }
   return true;
  }

//+------------------------------------------------------------------+
//| Calculates the profile and displays it in a table                |
//+------------------------------------------------------------------+
void CIntRate::OnEvent(const MarketEvent &event)
  {
   double sec_one_day = 60*60*24;   //86 400
   for(int i = 0; i < Symbols.Total(); i++)
   {
      CSymbol* Fut = Symbols.At(i);
      double brate = BaseRate(Fut);
      double days = (Fut.ExpirationDate()-TimeCurrent())/sec_one_day;
      if(Fut.Last() == 0.0)
         continue;
      double per = (Fut.Last() - brate)/brate*100.0;
      double per_in_year = per/days*365;
      Panel.SetLine(i, Fut.NameBasisSymbol() + " " + DoubleToString(days, 0) + " Days:", DoubleToString(per_in_year, 2)+"%");
   }

  }
//+------------------------------------------------------------------+
//| Returns the spot quote of the futures                            |
//+------------------------------------------------------------------+
double CIntRate::BaseRate(CSymbol* fut)
{
   string name = fut.NameBasisSymbol();
   if(StringFind(name, "Si", 0) == 0)
      return SymbolInfoDouble("USDRUB_TOD", SYMBOL_LAST)*fut.ContractSize();
   return SymbolInfoDouble(name, SYMBOL_LAST)*fut.ContractSize();
}
//+------------------------------------------------------------------+
```

The basic functionality is implemented in the OnInit method. This method receives the name of the underlying asset using WS.NameBasisSymbol() and checks all symbols to find all futures corresponding to this underlying asset. Each such futures symbol is converted to a CSymbol object and is added to the CArrayObj list of symbols. Before that, it is checked whether the futures contract is valid. The expiration time of a proper futures contract should be in the future.

The interest rate of each futures from the Symbols collection is calculated in the OnEvent method. The number of days before expiration and the delta between the futures and the spot price are calculated. The price difference is converted to a percentage, which is then normalized in accordance with the annual return. The resulting value is written to the Panel table (using the SetLine method).

The table itself is simple, and is based on a set of graphic classes similar to the CStrategy panel that appears with the Expert Advisor. The code of the EA's graphical component is below:

```
//+------------------------------------------------------------------+
//|                                                        Panel.mqh |
//|                                 Copyright 2017, Vasiliy Sokolov. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Vasiliy Sokolov."
#property link      "https://www.mql5.com"
#include <Panel\ElChart.mqh>

class CPercentPanel : public CElChart
{
private:
   CArrayObj  m_fields;
   CArrayObj  m_values;
public:

   CPercentPanel(void);
   void SetLine(int index, string field, string value);
};
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPercentPanel::CPercentPanel(void) : CElChart(OBJ_RECTANGLE_LABEL)
{
   Width(200);
   Height(200);
}
//+------------------------------------------------------------------+
//| Sets the line                                                    |
//+------------------------------------------------------------------+
void CPercentPanel::SetLine(int index,string field,string value)
{
   if(m_fields.Total() <= index)
   {
      CElChart* sfield = new CElChart(OBJ_LABEL);
      sfield.XCoord(XCoord()+10);
      sfield.YCoord(YCoord()+21*index+10);
      sfield.Text(field);
      m_fields.Add(sfield);
      m_elements.Add(sfield);

      CElChart* svalue = new CElChart(OBJ_LABEL);
      svalue.YCoord(YCoord()+21*index+10);
      svalue.XCoord(XCoord()+132);
      svalue.Text(value);
      svalue.TextColor(clrGreen);
      m_values.Add(svalue);
      m_elements.Add(svalue);
      if(IsShowed())
      {
         sfield.Show();
         svalue.Show();
      }
      Height(m_fields.Total()*20 + m_fields.Total()*2 + 10);
   }
   else
   {
      CElChart* el = m_fields.At(index);
      el.Text(field);
      el = m_values.At(index);
      el.Text(value);
   }
   ChartRedraw();
}
```

Once the Expert Advisor is compiled and launched on a chart of one of Si futures contracts, the following table should appear:

![](https://c.mql5.com/2/28/Expert08.png)

The Ruble/Dollar interest profile as a table

As can be seen from the table, interest rates on virtually all time sections are equal, and they amount to just over 7% per annum. The nearest futures contract shows a slightly higher rate.

Important note: before you launch the Expert Advisor on a chart, make sure that quotes of all required futures contracts are available and have been pre-loaded. Otherwise, the result may be undefined.

### Conclusion

We have reviewed the new CSymbol class included in the CStrategy trading engine. This class simplifies work with trading instruments by providing access to various symbol properties. CSymbol helped us to create a rather interesting and non-trivial indicator of the interest rate profile. This was a very demonstrative example. Many symbol properties were easily obtained from CSymbol objects, and the calculation was not very complicated. The Expert Advisor simultaneously worked with six financial symbols, while this didn't affect the code length. CStrategy is inherited from CObject, and instances can be easily added to standard collections to make data processing scalable and versatile. In addition, specific functionality has been transferred from CStrategy to CSymbol, which has made CStrategy more lightweight and manageable.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/3270](https://www.mql5.com/ru/articles/3270)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3270.zip "Download all attachments in the single ZIP archive")

[article\_16.05.2017.zip](https://www.mql5.com/en/articles/download/3270/article_16.05.2017.zip "Download article_16.05.2017.zip")(146.22 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing graphical interfaces based on .Net Framework and C# (part 2): Additional graphical elements](https://www.mql5.com/en/articles/6549)
- [Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://www.mql5.com/en/articles/5563)
- [Custom Strategy Tester based on fast mathematical calculations](https://www.mql5.com/en/articles/4226)
- [R-squared as an estimation of quality of the strategy balance curve](https://www.mql5.com/en/articles/2358)
- [Universal Expert Advisor: CUnIndicator and Use of Pending Orders (Part 9)](https://www.mql5.com/en/articles/2653)
- [Implementing a Scalping Market Depth Using the CGraphic Library](https://www.mql5.com/en/articles/3336)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/214433)**
(1)


![igorbel](https://c.mql5.com/avatar/avatar_na2.png)

**[igorbel](https://www.mql5.com/en/users/igorbel)**
\|
15 Jun 2017 at 22:20

Why don't you have SL/TP in trading methods of [position opening](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal")? I mean public methods TradeControl Buy/Sell/BuyLimit....

![Naive Bayes classifier for signals of a set of indicators](https://c.mql5.com/2/27/MQL5-avatar-naiveClass-001.png)[Naive Bayes classifier for signals of a set of indicators](https://www.mql5.com/en/articles/3264)

The article analyzes the application of the Bayes' formula for increasing the reliability of trading systems by means of using signals from multiple independent indicators. Theoretical calculations are verified with a simple universal EA, configured to work with arbitrary indicators.

![Testing patterns that arise when trading currency pair baskets. Part I](https://c.mql5.com/2/28/articles_234__1.png)[Testing patterns that arise when trading currency pair baskets. Part I](https://www.mql5.com/en/articles/3339)

We begin testing the patterns and trying the methods described in the articles about trading currency pair baskets. Let's see how oversold/overbought level breakthrough patterns are applied in practice.

![How to conduct a qualitative analysis of trading signals and select the best of them](https://c.mql5.com/2/27/MQL5-avatar-qualityAnalysis-001.png)[How to conduct a qualitative analysis of trading signals and select the best of them](https://www.mql5.com/en/articles/3166)

The article deals with evaluating the performance of Signals Providers. We offer several additional parameters highlighting signal trading results from a slightly different angle than in traditional approaches. The concepts of the proper management and perfect deal are described. We also dwell on the optimal selection using the obtained results and compiling the portfolio of multiple signal sources.

![Graphical Interfaces XI: Rendered controls (build 14.2)](https://c.mql5.com/2/28/av.png)[Graphical Interfaces XI: Rendered controls (build 14.2)](https://www.mql5.com/en/articles/3366)

In the new version of the library, all controls will be drawn on separate graphical objects of the OBJ\_BITMAP\_LABEL type. We will also continue to describe the optimization of code: changes in the core classes of the library will be discussed.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/3270&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071723410414841144)

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