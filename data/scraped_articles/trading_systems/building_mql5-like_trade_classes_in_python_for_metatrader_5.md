---
title: Building MQL5-Like Trade Classes in Python for MetaTrader 5
url: https://www.mql5.com/en/articles/18208
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:34:17.240276
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/18208&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049222489067857765)

MetaTrader 5 / Trading systems


**Contents**

- [Introduction](https://www.mql5.com/en/articles/18208#para1)
- [The CAccountInfo class](https://www.mql5.com/en/articles/18208#CAccountInfo-Python)
- [The CSymbolInfo class](https://www.mql5.com/en/articles/18208#CSymbolInfo-Python)
- [The COrderInfo class](https://www.mql5.com/en/articles/18208#COrderInfo-Python)
- [The CHistoryOrderInfo class](https://www.mql5.com/en/articles/18208#CHistoryOrderInfo-Python)
- [The CPositionInfo class](https://www.mql5.com/en/articles/18208#CPositionInfo-Python)
- [The CDealInfo class](https://www.mql5.com/en/articles/18208#CDealInfo-Python)
- [The CTerminalInfo class](https://www.mql5.com/en/articles/18208#CTerminalInfo-Python)
- [The CTrade class](https://www.mql5.com/en/articles/18208#CTrade-Python)
- [Conclusion](https://www.mql5.com/en/articles/18208#para2)

### Introduction

Building algorithmic trading systems in the MQL5 programming language has been made easier with Standard Libraries that come preloaded in MetaEditor. These modules (libraries) come with functions and variables that simplify the process of opening, validating, closing the trades, etc.

Without these dependencies, it becomes harder to write even a simple program, such as making a simple script for opening a buy position (trade).

**Without the CTrade class**

```
void OnStart()
  {
   MqlTradeRequest request;
   MqlTradeResult result;

   MqlTick ticks;
   SymbolInfoTick(Symbol(), ticks);

//--- setting a trade request

   ZeroMemory(request);
   request.action   =TRADE_ACTION_DEAL;
   request.symbol   =Symbol();
   request.magic    =2025;
   request.volume   =0.01;
   request.type     =ORDER_TYPE_BUY;
   request.price    =ticks.ask;
   request.sl       =0;
   request.tp       =0;
   request.deviation   = 10;  // Max price slippage in points
   request.magic       = 2025;
   request.comment     = "Buy Order";
   request.type_filling= ORDER_FILLING_IOC;   // or ORDER_FILLING_IOC, ORDER_FILLING_RETURN
   request.type_time   = ORDER_TIME_GTC;      // Good till canceled

//--- action and return the result

   if(!OrderSend(request, result))
    {
      Print("OrderSend failed retcode: ", result.retcode);
    }
  }
```

**With the CTrade class**

```
#include <Trade\Trade.mqh>
CTrade m_trade;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
    MqlTick ticks;
    SymbolInfoTick(Symbol(), ticks);

    m_trade.SetTypeFillingBySymbol(Symbol());
    m_trade.SetExpertMagicNumber(2025);
    m_trade.SetDeviationInPoints(10); //Slippage

    m_trade.Buy(0.01, Symbol(), ticks.ask,0,0,"Buy Order");
  }
```

Both these functions open a buy position in MetaTrader 5 but, the first approach is very crude, time-consuming, and it increases the odds of producing bugs due to the large number of lines of code you are supposed to write to get a simple functionality.

Not to mention that it requires you to be more technical (to understand everything that goes into the process of sending a buy position in MetaTrader 5).

There is a [Python package known as MetaTrader 5](https://www.mql5.com/en/docs/python_metatrader5) which gives Python developers access to the platform, an ability to get almost all the information from the platform (symbols, positions opened, etc) and the ability to send some commands for opening, modifying, deleting trades, etc. Similarly to what we can do with the MQL5 programming language.

As useful as this package is, it doesn't come with built-in modules like those present in the MQL5 language to aid us in the development process.

Similarly to the first coding example, writing a simple program in Python, requires you to write more lines of code and what's even worse is that, this MetaTrade5 Python package doesn't cooperate well with most Integrated Development Environments IDE(s) such as [Visual Studio Code](https://www.mql5.com/go?link=https://code.visualstudio.com/ "https://code.visualstudio.com/"), this means that you won't get the Intellisense coding support which is very useful.

Due to the lack of Intellisense in IDE(s) support for this package, you'll often find yourself referring to the documentation for simple concepts you forget instead of figuring them out in the IDE. _This leads to a terrible experience working with this package._

In this article, we are going to implement the Trade Classes in Python on top of the MetaTrader 5 package to help us write programs effectively in Python as in MQL5.

### The CAccountInfo Class

In MQL5 this class is for working with trade account properties. All the information about the trading account on a broker can be accessed using this class, let's make its equivalent in Python.

| Python Custom CAccountInfo trade class | MQL5 Built-in CAccountInfo trade class | Description |
| --- | --- | --- |
| **Integer & String type properties** |  |  |
| ```<br>login()<br>``` | [Login](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfologin) | Gets the account number. |
| ```<br>trade_mode()<br>``` | [TradeMode](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfotrademode) | Gets the trade mode. |
| ```<br>trade_mode_description()<br>``` | [TradeModeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfotrademodedescription) | Gets the trade mode as a string. |
| ```<br>leverage()<br>``` | [Leverage](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfoleverage) | Gets the amount of given leverage. |
| ```<br>stopout_mode()<br>``` | [StopoutMode](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfostopoutmode) | Gets the mode of stop out setting. |
| ```<br>stopout_mode_description()<br>``` | [StopoutModeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfostopoutmodedescription) | Gets the mode of stop out settting as a string. |
| ```<br>margin_mode()<br>``` | [MarginMode](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomarginmode) | Gets the margin calculation mode. |
| ```<br>margin_mode_description()<br>``` | [MarginModeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomarginmodedescription) | Gets the margin calculation mode as a string. |
| ```<br>trade_allowed()<br>``` | [TradeAllowed](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfotradeallowed) | Gets the flag of trade allowance. |
| ```<br>trade_expert()<br>``` | [TradeExpert](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfotradeexpert) | Gets the flag of automated trade allowance. |
| ```<br>limit_orders()<br>``` | [LimitOrders](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfolimitorders) | Gets the maximal number of allowed pending orders. |
| **Double type properties** |  |  |
| ```<br>balance()<br>``` | [Balance](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfobalance) | Gets the balance of MetaTrader 5 account. |
| ```<br>credit()<br>``` | [Credit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfocredit) | Gets the amount of given credit. |
| ```<br>profit()<br>``` | [Profit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfoprofit) | Gets the amount of current profit on an account |
| ```<br>equity()<br>``` | [Equity](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfoequity) | Gets the amount of current equity on account. |
| ```<br>margin()<br>``` | [Margin](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomargin) | Gets the amount of reserved margin. |
| ```<br>free_margin()<br>``` | [FreeMargin](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfofreemargin) | Gets the amount of free margin. |
| ```<br>margin_level()<br>``` | [MarginLevel](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomarginlevel) | Gets the level of margin. |
| ```<br>margin_call()<br>``` | [MarginCall](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomargincall) | Gets the level of margin for deposit. |
| ```<br>margin_stopout()<br>``` | [MarginStopOut](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomarginstopout) | Gets the level of margin for stop out. |
| **Text type properties** |  |  |
| ```<br>name()<br>``` | [Name](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfoname) | Gets the account name |
| ```<br>server()<br>``` | [Server](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfoserver) | Gets the trade server name |
| ```<br>company()<br>``` | [Company](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfocompany) | Gets the company name that serves the account |
| ```<br>currency() <br>``` | [Currency](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfocurrency) | Gets the deposit currency name. |
| **Additional methods** |  |  |
| ```<br>margin_check(self, symbol, order_type, volume, price) <br>``` | [MarginCheck](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomargincheck) | Gets the amount of margin required to execute trade operation. |
| ```<br>free_margin_check(self, symbol, order_type, volume, price)<br>``` | [FreeMarginCheck](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfofreemargincheck) | Gets the amount of free margin left after execution of trade operation. |
| ```<br>order_profit_check(self, symbol, order_type, volume, price_open, price_close)<br>``` | [OrderProfitCheck](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfoorderprofitcheck) | Gets the avaluated profit based on the parameters passed. |
| ```<br>max_lot_check(self, symbol, order_type, price, percent=100) <br>``` | [MaxLotCheck](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomaxlotcheck) | Gets the maximal possible volume of trade operation. |

**Example usage**

```
import MetaTrader5 as mt5
from Trade.AccountInfo import CAccountInfo

if not mt5.initialize(r"c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5\terminal64.exe"):
    print("Failed to initialize Metatrader5 Error = ",mt5.last_error())
    quit()

acc = CAccountInfo()

print(f"""
Account Information
-------------------
Login: {acc.login()}
Name: {acc.name()}
Server: {acc.server()}
Company: {acc.company()}
Currency: {acc.currency()}
Trade Mode: {acc.trade_mode()} ({acc.trade_mode_description()})
Leverage: {acc.leverage()}
Stopout Mode: {acc.stopout_mode()} ({acc.stopout_mode_description()})
Margin Mode: {acc.margin_mode()} ({acc.margin_mode_description()})
Trade Allowed: {acc.trade_allowed()}
Trade Expert: {acc.trade_expert()}
Limit Orders: {acc.limit_orders()}
-------------------
Balance: {acc.balance()}
Credit: {acc.credit()}
Profit: {acc.profit()}
Equity: {acc.equity()}
Margin: {acc.margin()}
Free Margin: {acc.free_margin()}
Margin Level: {acc.margin_level()}
Margin Call: {acc.margin_call()}
Margin StopOut: {acc.margin_stopout()}
-------------------
""")

mt5.shutdown()
```

**Outputs**

```
Account Information
-------------------
Login: 61346344
Name: John Doe
Server: MetaQuotes-Demo
Company: MetaQuotes Software Corp
Currency: USD
Trade Mode: 0 (Demo)
Leverage: 400
Stopout Mode: 0 (Percent)
Margin Mode: 2 (Retail Hedging)
Trade Allowed: True
Trade Expert: True
Limit Orders: 500
-------------------
Balance: 928.42
Credit: 0.0
Profit: -2.21
Equity: 926.21
Margin: 2.81
Free Margin: 923.4
Margin Level: 32961.20996441281
Margin Call: 90.0
Margin StopOut: 20.0
-------------------
```

### The CSymbolInfo Class

This class provides access to the symbol properties.

| **Python custom CSymbolInfo class** | MQL5 built-in CSymbolInfo class | Description |
| --- | --- | --- |
| **Controlling** |  |  |
| ```<br>refresh()<br>``` | [Refresh](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinforefresh) | Refreshes the symbol data. |
| ```<br>refresh_rates()<br>``` | [RefreshRates](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinforefreshrates) | Refreshes the symbol quotes |
| **Properties** |  |  |
| ```<br>name() <br>``` | [Name](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoname) | Gets the symbol name. |
| ```<br>select(self, select=True)<br>``` | [Select](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoselect) | Adds or removes the symbol to and from the "Market Watch" |
| ```<br>is_synchronized()<br>``` | [IsSynchronized](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoissynchronized) | Checks the symbol synchronization with the server. |
| **Volumes** |  |  |
| ```<br>volume()<br>``` | [Volume](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfovolume) | Gets the volumne of the last deal. |
| ```<br>volume_high()<br>``` | [VolumeHigh](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfovolumehigh) | Gets the maximal volume for a day |
| ```<br>volume_low()<br>``` | [VolumeLow](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfovolumelow) | Gets the minimal volume for a day. |
| **Miscellaneous** |  |  |
| ```<br>time()<br>``` | [Time](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotime) | Gets the time of last quote. |
| ```<br>spread()<br>``` | [Spread](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfospread) | Gets the amount of spread (in points). |
| ```<br>spread_float()<br>``` | [SpreadFloat](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfospreadfloat) | Gets the flag of floating spread. |
| ```<br>ticks_book_depth()<br>``` | [TicksBookDepth](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoticksbookdepth) | Gets the depth of ticks saving. |
| **Levels** |  |  |
| ```<br>stops_level()<br>``` | [StopsLevel](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfostopslevel) | Gets the minimal indent for orders (in points). |
| ```<br>freeze_level()<br>``` | [FreezeLevel](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfofreezelevel) | Gets the distance of freezing trade operations (in points). |
| **Bid prices** |  |  |
| ```<br>bid()<br>``` | [Bid](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfobid) | Gets the current bid price. |
| ```<br>bid_high()<br>``` | [BidHigh](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfobidhigh) | Gets the maximal bid price for a day. |
| ```<br>bid_low()<br>``` | [BidLow](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfobidlow) | Gets the minimal bid price for a day. |
| **Ask prices** |  |  |
| ```<br>ask()<br>``` | [Ask](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoask) | Gets the current Ask price |
| ```<br>ask_high()<br>``` | [AskHigh](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoaskhigh) | Gets the maximal Ask price for a day |
| ```<br>ask_low()<br>``` | [AskLow](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoasklow) | Gets the minimal Ask price for a day |
| **Prices** |  |  |
| ```<br>last()<br>``` | [Last](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfolast) | Returns the current last price |
| ```<br>last_high() <br>``` | [LastHigh](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfolasthigh) | Returns the maximal last price for a day |
| ```<br>last_low()<br>``` | [LastLow](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfolastlow) | Returns the minimal last price for a day |
| **Trade modes** |  |  |
| ```<br>trade_calc_mode()<br>``` | [TradeCalcMode](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotradecalcmode) | Gets the mode of contract cost calculation in integer format. |
| ```<br>trade_calc_mode_description()<br>``` | [TradeCalcModeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotradecalcmodedescription) | Gets the mode of contract cost calculation in string format. |
| ```<br>trade_mode()<br>``` | [TradeMode](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotrademode) | Gets the type of order execution in integer format. |
| ```<br>trade_mode_description() <br>``` | [TradeModeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotrademodedescription) | Gets the type of order execution in string format. |
| ```<br>trade_execution()<br>``` | [TradeExecution](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotradeexecution) | Gets the trade execution mode in integer format. |
| ```<br>trade_execution_description()<br>``` | [TradeExecutionDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotradeexecutiondescription) | Gets the trade execution mode in string format |
| **Swaps** |  |  |
| ```<br>swap_mode() <br>``` | [SwapMode](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoswapmode) | Gets the swap calculation mode in integer format |
| ```<br>swap_mode_description()<br>``` | [SwapModeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoswapmodedescription) | Gets the swap calculation mode in string format |
| ```<br>swap_rollover_3days()<br>``` | [SwapRollover3days](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoswaprollover3days) | Gets the day of triple swap charge as an integer |
| ```<br>swap_rollover_3days_description()<br>``` | [SwapRollover3daysDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoswaprollover3daysdescription) | Gets the day of triple swap charge as a string. |
| **Margin** |  |  |
| ```<br>margin_initial()<br>``` | [MarginInitial](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfomargininitial) | Gets the value of initial margin |
| ```<br>margin_maintenance()<br>``` | [MarginMaintenance](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfomarginmaintenance) | Gets the value of maintenance margin |
| ```<br>margin_hedged()<br>``` |  | Returns the hedged margin value for the given symbol. |
| ```<br>margin_hedged_use_leg()<br>``` |  | Returns a boolean that tells whether the hedged margin applies to each leg (position side) individually. |
| **Tick information** |  |  |
| ```<br>digits() <br>``` | [Digits](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfodigits) | Gets the number of digits after period |
| ```<br>point()<br>``` | [Point](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfopoint) | Gets the value of one point |
| ```<br>tick_value()<br>``` | [TickValue](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotickvalue) | Gets the tick value (minimal change of price) |
| ```<br>tick_value_profit()<br>``` | [TickValueProfit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotickvalueprofit) | Gets the calculated tick price of a profitable position |
| ```<br>tick_value_loss()<br>``` | [TickValueLoss](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfotickvalueloss) | Gets the calculated tick price for a losing position |
| ```<br>tick_size()<br>``` | [TickSize](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoticksize) | Gets the minimal change of price |
| **Contracts sizes** |  |  |
| ```<br>contract_size()<br>``` | [ContractSize](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfocontractsize) | Gets the amount of trade contract |
| ```<br>lots_min()<br>``` | [LotsMin](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfolotsmin) | Gets the minimal volume to close a deal |
| ```<br>lots_max()<br>``` | [LotsMax](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfolotsmax) | Gets the maximal volume to close a deal |
| ```<br>lots_step()<br>``` | [LotsStep](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfolotsstep) | Gets the minimal step of volume change to close a deal |
| ```<br>lots_limit()<br>``` | [LotsLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfolotslimit) | Gets the maximal allowed volume of opened position and pending orders in either direction for one symbol |
| **Swap sizes** |  |  |
| ```<br>swap_long()<br>``` | [SwapLong](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoswaplong) | Gets the value of long position swap |
| ```<br>swap_short()<br>``` | [SwapShort](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoswapshort) | Gets the value of short position swap |
| **Symbol/Currency Information** |  |  |
| ```<br>currency_base()<br>``` | [CurrencyBase](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfocurrencybase) | Gets the name of symbol base currency |
| ```<br>currency_profit() <br>``` | [CurrencyProfit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfocurrencyprofit) | Gets the profit currency name |
| ```<br>currency_margin()<br>``` | [CurrencyMargin](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfocurrencymargin) | Gets the margin currency name |
| ```<br>bank()<br>``` | [Bank](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfobank) | Gets the name of the current quote source |
| ```<br>description()<br>``` | [Description](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfodescription) | Gets the string description of a symbol |
| ```<br>path()<br>``` | [Path](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfopath) | Gets the path in symbols tree |
| ```<br>page()<br>``` |  | The address of a webpage containing symbol's information |
| **Session Information** |  |  |
| ```<br>session_deals()<br>``` | [SessionDeals](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessiondeals) | Gets the number of deals in the current session |
| ```<br>session_buy_orders() <br>``` | [SessionBuyOrders](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionbuyorders) | Gets the number of buy orders presently |
| ```<br>session_sell_orders()<br>``` | [SessionSellOrders](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionsellorders) | Gets the number of sell orders presently |
| ```<br>session_turnover() <br>``` | [SessionTurnover](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionturnover) | Gets the summary of turnover of the current session |
| ```<br>session_interest()<br>``` | [SessionInterest](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessioninterest) | Gets the summary of open interest of the current session |
| ```<br>session_buy_orders_volume()<br>``` | [SessionBuyOrdersVolume](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionbuyordersvolume) | Gets the volume of buy orders |
| ```<br>session_sell_orders_volume()<br>``` | [SessionSellOrdersVolume](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionsellordersvolume) | Gets the volume of sell orders |
| ```<br>session_open()<br>``` | [SessionOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionopen) | Gets the open price of the current session |
| ```<br>session_close()<br>``` | [SessionClose](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionclose) | Gets the close price of the current session |
| ```<br>session_aw()<br>``` | [SessionAW](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionaw) | Gets the average weighted price of the current session |
| ```<br>session_price_settlement() <br>``` | [SessionPriceSettlement](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionpricesettlement) | Gets the settlement price of the current session |
| ```<br>session_price_limit_min() <br>``` | [SessionPriceLimitMin](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionpricelimitmin) | Gets the minimal price of the current session |
| ```<br>session_price_limit_max()<br>``` | [SessionPriceLimitMax](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfosessionpricelimitmax) | Gets the maximal price of the current session |

These are some of the methods in the Python class, a full list can be seen inside the file SymbolInfo.py.

**Example Usage**

```
import MetaTrader5 as mt5
from Trade.SymbolInfo import CSymbolInfo

if not mt5.initialize(r"c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5\terminal64.exe"):
    print("Failed to initialize Metatrader5 Error = ",mt5.last_error())
    quit()

m_symbol = CSymbolInfo("EURUSD")

print(f"""
Symbol Information
---------------------
Name: {m_symbol.name()}
Selected: {m_symbol.select()}
Synchronized: {m_symbol.is_synchronized()}

--- Volumes ---
Volume: {m_symbol.volume()}
Volume High: {m_symbol.volume_high()}
Volume Low: {m_symbol.volume_low()}

--- Time & Spread ---
Time: {m_symbol.time()}
Spread: {m_symbol.spread()}
Spread Float: {m_symbol.spread_float()}
Ticks Book Depth: {m_symbol.ticks_book_depth()}

--- Trade Levels ---
Stops Level: {m_symbol.stops_level()}
Freeze Level: {m_symbol.freeze_level()}

--- Bid Parameters ---
Bid: {m_symbol.bid()}
Bid High: {m_symbol.bid_high()}
Bid Low: {m_symbol.bid_low()}

--- Ask Parameters ---
Ask: {m_symbol.ask()}
Ask High: {m_symbol.ask_high()}
Ask Low: {m_symbol.ask_low()}

--- Last Parameters ---
Last: {m_symbol.last()}
Last High: {m_symbol.last_high()}
Last Low: {m_symbol.last_low()}

--- Order & Trade Modes ---
Trade Calc Mode: {m_symbol.trade_calc_mode()} ({m_symbol.trade_calc_mode_description()})
Trade Mode: {m_symbol.trade_mode()} ({m_symbol.trade_mode_description()})
Trade Execution Mode: {m_symbol.trade_execution()}  ({m_symbol.trade_execution_description()})

--- Swap Terms ---
Swap Mode: {m_symbol.swap_mode()} ({m_symbol.swap_mode_description()})
Swap Rollover 3 Days: {m_symbol.swap_rollover_3days()} ({m_symbol.swap_rollover_3days_description()})

--- Futures Dates ---
Start Time: {m_symbol.start_time()}
Expiration Time: {m_symbol.expiration_time()}

--- Margin Parameters ---
Initial Margin: {m_symbol.margin_initial()}
Maintenance Margin: {m_symbol.margin_maintenance()}
Hedged Margin: {m_symbol.margin_hedged()}
Hedged Margin Use Leg: {m_symbol.margin_hedged_use_leg()}

--- Tick Info ---

Digits: {m_symbol.digits()}
Point: {m_symbol.point()}
Tick Value: {m_symbol.tick_value()}
Tick Value Profit: {m_symbol.tick_value_profit()}
Tick Value Loss: {m_symbol.tick_value_loss()}
Tick Size: {m_symbol.tick_size()}

--- Contracts sizes---
Contract Size: {m_symbol.contract_size()}
Lots Min: {m_symbol.lots_min()}
Lots Max: {m_symbol.lots_max()}
Lots Step: {m_symbol.lots_step()}
Lots Limit: {m_symbol.lots_limit()}

--- Swap sizes

Swap Long: {m_symbol.swap_long()}
Swap Short: {m_symbol.swap_short()}

--- Currency Info ---
Currency Base: {m_symbol.currency_base()}
Currency Profit: {m_symbol.currency_profit()}
Currency Margin: {m_symbol.currency_margin()}
Bank: {m_symbol.bank()}
Description: {m_symbol.description()}
Path: {m_symbol.path()}
Page: {m_symbol.page()}

--- Session Info ---
Session Deals: {m_symbol.session_deals()}
Session Buy Orders: {m_symbol.session_buy_orders()}
Session Sell Orders: {m_symbol.session_sell_orders()}
Session Turnover: {m_symbol.session_turnover()}
Session Interest: {m_symbol.session_interest()}
Session Buy Volume: {m_symbol.session_buy_orders_volume()}
Session Sell Volume: {m_symbol.session_sell_orders_volume()}
Session Open: {m_symbol.session_open()}
Session Close: {m_symbol.session_close()}
Session AW: {m_symbol.session_aw()}
Session Price Settlement: {m_symbol.session_price_settlement()}
Session Price Limit Min: {m_symbol.session_price_limit_min()}
Session Price Limit Max: {m_symbol.session_price_limit_max()}
---------------------
""")

mt5.shutdown()
```

**Outputs**

```
Symbol Information
---------------------
Name: EURUSD
Selected: True
Synchronized: True

--- Volumes ---
Volume: 0
Volume High: 0
Volume Low: 0

--- Time & Spread ---
Time: 2025-05-21 20:30:36
Spread: 0
Spread Float: True
Ticks Book Depth: 0

--- Trade Levels ---
Stops Level: 0
Freeze Level: 0

--- Bid Parameters ---
Bid: 1.1335600000000001
Bid High: 1.13623
Bid Low: 1.12784

--- Ask Parameters ---
Ask: 1.1335600000000001
Ask High: 1.13623
Ask Low: 1.12805

--- Last Parameters ---
Last: 0.0
Last High: 0.0
Last Low: 0.0

--- Order & Trade Modes ---
Trade Calc Mode: 0 (Calculation of profit and margin for Forex)
Trade Mode: 4 (No trade restrictions)
Trade Execution Mode: 2  (Market execution)

--- Swap Terms ---
Swap Mode: 1 (Swaps are calculated in points)
Swap Rollover 3 Days: 3 (Wednesday)

--- Futures Dates ---
Start Time: 0
Expiration Time: 0

--- Margin Parameters ---
Initial Margin: 100000.0
Maintenance Margin: 0.0
Hedged Margin: 0.0
Hedged Margin Use Leg: False

--- Tick Info ---

Digits: 5
Point: 1e-05
Tick Value: 1.0
Tick Value Profit: 1.0
Tick Value Loss: 1.0
Tick Size: 1e-05

--- Contracts sizes---
Contract Size: 100000.0
Lots Min: 0.01
Lots Max: 100.0
Lots Step: 0.01
Lots Limit: 0.0

--- Swap sizes

Swap Long: -8.99
Swap Short: 4.5

--- Currency Info ---
Currency Base: EUR
Currency Profit: USD
Currency Margin: EUR
Bank: Pepperstone
Description: Euro vs US Dollar
Path: Markets\Forex\Majors\EURUSD
Page:

--- Session Info ---
Session Deals: 1
Session Buy Orders: 647
Session Sell Orders: 2
Session Turnover: 10.0
Session Interest: 0.0
Session Buy Volume: 3.0
Session Sell Volume: 13.0
Session Open: 1.12817
Session Close: 1.12842
Session AW: 0.0
Session Price Settlement: 0.0
Session Price Limit Min: 0.0
Session Price Limit Max: 0.0
---------------------
```

### The COrderInfo Class

This class provides access to the pending order properties.

| **Python custom COrderInfo class** | MQL5 built-in COrderInfo class | Description |
| --- | --- | --- |
| **Integer & datetime type properties** |  |  |
| ```<br>ticket()<br>``` | [Ticket](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfoticket) | Gets the ticket of an order, previously selected for access. |
| ```<br>type_time()<br>``` | [TypeTime](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotypetime) | Gets the type of order at the time of the expiration. |
| ```<br>type_time_description()<br>``` | [TypeTimeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotypetimedescription) | Gets the order type by expiration as a string |
| ```<br>time_setup()<br>``` | [TimeSetup](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotimesetup) | Gets the time of order placement. |
| ```<br>time_setup_msc()<br>``` | [TimeSetupMsc](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotimesetupmsc) | Receives the time of placing an order in milliseconds since 01.01.1970. |
| ```<br>order_type()<br>``` | [OrderType](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfoordertype) | Gets the order type in integer format. |
| ```<br>order_type_description()<br>``` | [OrderTypeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotypedescription) | Gets the order type as a string |
| ```<br>state()<br>``` | [State](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfostate) | Gets the order state as an integer. |
| ```<br>state_description()<br>``` | [StateDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfostatedescription) | Gets the order state as a string. |
| ```<br>magic()<br>``` | [Magic](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfomagic) | Gets the ID of the expert that placed the order. |
| ```<br>position_id()<br>``` | [PositionId](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfopositionid) | Gets the ID of position. |
| ```<br>type_filling()<br>``` | [TypeFilling](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotypefilling) | Gets the type of order execution by remainder as an integer. |
| ```<br>type_filling_description()<br>``` | [TypeFillingDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotypefillingdescription) | Gets the type of order execution by remainder as a string. |
| ```<br>time_done() <br>``` | [TimeDone](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotimedone) | Gets the time of order execution or cancellation. |
| ```<br>time_done_msc()<br>``` | [TimeDoneMsc](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotimedonemsc) | Receives order execution or cancellation time in milliseconds since since 01.01.1970. |
| ```<br>time_expiration()<br>``` | [TimeExpiration](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotimeexpiration) | Gets the time of order expiration. |
| **Double type properties** |  |  |
| ```<br>volume_initial()<br>``` | [VolumeInitial](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfovolumeinitial) | Gets the initial volume of order. |
| ```<br>volume_current()<br>``` | [VolumeCurrent](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfovolumecurrent) | Gets the unfilled volume of order. |
| ```<br>price_open()<br>``` | [PriceOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfopriceopen) | Gets the order price. |
| ```<br>price_current()<br>``` | [PriceCurrent](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfopricecurrent) | Gets the current price by order symbol. |
| ```<br>stop_loss()<br>``` | [StopLoss](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfostoploss) | Gets the order's Stop loss. |
| ```<br>take_profit()<br>``` | [TakeProfit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfotakeprofit) | Gets the order's Take profit. |
| ```<br>price_stop_limit()<br>``` | [PriceStopLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfopricestoplimit) | Gets the price of a limit order. |
| **Access to text properties** |  |  |
| ```<br>comment()<br>``` | [Symbol](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfosymbol) | Gets the order comment. |
| ```<br>symbol()<br>``` | [Comment](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfocomment) | Gets the name of the order symbol. |
| **Selection** |  |  |
| ```<br>select_order(self, order) -> bool<br>``` |  | Selects an order by its object (dictionary) from list of oders returned by the function [MetaTrader5.orders\_get()](https://www.mql5.com/en/docs/python_metatrader5/mt5positionsget_py) |

**Example Usage**

```
import MetaTrader5 as mt5
from Trade.OrderInfo import COrderInfo

if not mt5.initialize(r"c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5\terminal64.exe"):
    print("Failed to initialize Metatrader5 Error = ",mt5.last_error())
    quit()

# Get all orders from MT5
orders = mt5.orders_get()

# Loop and print info
m_order = COrderInfo()

for i, order in enumerate(orders):
    if m_order.select_order(order=order):
        print(f"""
Order #{i}

--- Integer & datetime type properties ---

Ticket: {m_order.ticket()}
Type Time: {m_order.type_time()} ({m_order.type_time_description()})
Time Setup: {m_order.time_setup()}
Time Setup (ms): {m_order.time_setup_msc()}
State: {m_order.state()} ({m_order.state_description()})
Order Type: {m_order.order_type()} ({m_order.order_type_description()})
Magic Number: {m_order.magic()}
Position ID: {m_order.position_id()}
Type Filling: {m_order.type_filling()} ({m_order.type_filling_description()})
Time Done: {m_order.time_done()}
Time Done (ms): {m_order.time_done_msc()}
Time Expiration: {m_order.time_expiration()}
External ID: {m_order.external_id()}

--- Double type properties ---

Volume Initial: {m_order.volume_initial()}
Volume Current: {m_order.volume_current()}
Price Open: {m_order.price_open()}
Price Current: {m_order.price_current()}
Stop Loss: {m_order.stop_loss()}
Take Profit: {m_order.take_profit()}
Price StopLimit: {m_order.price_stop_limit()}

--- Text type properties ---

Comment: {m_order.comment()}
Symbol: {m_order.symbol()}

""")

mt5.shutdown()
```

**Outputs**

```
Order #0

--- Integer & datetime type properties ---

Ticket: 153201235
Type Time: 2 (ORDER_TIME_SPECIFIED)
Time Setup: 2025-05-21 23:56:16
Time Setup (ms): 1747860976672
State: 1 (Order accepted)
Order Type: 3 (Sell Limit pending order)
Magic Number: 1001
Position ID: 0
Type Filling: 2 (IOC (Immediate or Cancel))
Time Done: 1970-01-01 03:00:00
Time Done (ms): 0
Time Expiration: 2025-05-21 23:57:14.940000
External ID:

--- Double type properties ---

Volume Initial: 0.01
Volume Current: 0.01
Price Open: 1.13594
Price Current: 1.1324
Stop Loss: 0.0
Take Profit: 0.0
Price StopLimit: 0.0

--- Text type properties ---

Comment: Sell Limit Order
Symbol: EURUSD
```

### The CHistoryOrderInfo Class

This class provides easy access to the history order properties.

| **Python custom CHistoryOrderInfo class** | MQL5 built-in CHistoryOrderInfo class | Description |
| --- | --- | --- |
| **Integer, Datetime, and String type properties** |  |  |
| ```<br>time_setup()<br>``` | [TimeSetup](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotimesetup) | Gets the time of order placement. |
| ```<br>time_setup_msc()<br>``` | [TimeSetupMsc](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotimesetupmsc) | Returns the time of placing an order in milliseconds since 01.01.1970 |
| ```<br>time_done()<br>``` | [TimeDone](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotimedone) | Gets the time of order execution or cancellation. |
| ```<br>time_done_msc()<br>``` | [TimeDoneMsc](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotimedonemsc) | Returns order execution or cancellation time in milliseconds since 01.01.1970 |
| ```<br>magic()<br>``` | [Magic](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfomagic) | Gets the ID of an expert advisor that placed a selected order |
| ```<br>ticket()<br>``` |  | Returns the ticket of the selected order. |
| ```<br>order_type()<br>``` | [OrderType](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfoordertype) | Returns the type of the selected order. |
| ```<br>order_type_description()<br>``` | [OrderTypeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotypedescription) | Returns the type of the selected order as a string |
| ```<br>state()<br>``` | [State](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfostate) | Returns the order state as an integer. |
| ```<br>state_description()<br>``` | [StateDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfostatedescription) | Returns the order state as a string. |
| ```<br>time_expiration()<br>``` | [TimeExpiration](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotimeexpiration) | Gets the time of the selected order expiration. |
| ```<br>type_filling()<br>``` | [TypeFilling](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotypefilling) | Gets the type of order execution by remainder in integer format. |
| ```<br>type_filling_description()<br>``` | [TypeFillingDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotypefillingdescription) | Gets the type of order execution by remainder as a string. |
| ```<br>type_time()<br>``` | [TypeTime](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotypetime) | Gets the type of the selected order at the time of the expiration as an integer. |
| ```<br>type_time_description()<br>``` | [TypeTimeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotypetimedescription) | Gets the selected order type at the time of the expiration in string format. |
| ```<br>position_id()<br>``` | [PositionId](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfopositionid) | Gets the position ID |
| **Double type properties** |  |  |
| ```<br>volume_initial()<br>``` | [VolumeInitial](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfovolumeinitial) | Gets the initial volume of the selected order |
| ```<br>volume_current()<br>``` | [VolumeCurrent](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfovolumecurrent) | Gets the unfufilled volume of the selected order. |
| ```<br>price_open()<br>``` | [PriceOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfopriceopen) | Gets the selected order price. |
| ```<br>price_current() <br>``` | [PriceCurrent](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfopricecurrent) | Gets the current price by order symbol. |
| ```<br>stop_loss() <br>``` | [StopLoss](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfostoploss) | Gets the selected order's stop loss. |
| ```<br>take_profit()<br>``` | [TakeProfit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfotakeprofit) | Gets the selected order's take profit. |
| ```<br>price_stop_limit() <br>``` | [PriceStopLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfopricestoplimi) | Gets the price of a selected limit order. |
| **Text properties** |  |  |
| ```<br>symbol() <br>``` | [Symbol](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfosymbol) | Returns the symbol of a selected order. |
| ```<br>comment()<br>``` | [Comment](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo/chistoryorderinfocomment) | Returns the comment of a selected order. |
| **Selection** |  |  |
| ```<br>select_order(self, order) -> bool<br>``` |  | Selects an order by it's object from a list of objects (dictionaries) returned from the function [MetaTrader5.history\_orders\_get](https://www.mql5.com/en/docs/python_metatrader5/mt5historyordersget_py). |

**Example usage**

```
import MetaTrader5 as mt5
from Trade.HistoryOrderInfo import CHistoryOrderInfo
from datetime import datetime, timedelta

if not mt5.initialize(r"c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5\terminal64.exe"):
    print("Failed to initialize Metatrader5 Error = ",mt5.last_error())
    quit()

# The date range

from_date = datetime.now() - timedelta(hours=5)
to_date = datetime.now()

# Get history orders

history_orders = mt5.history_orders_get(from_date, to_date)

if history_orders == None:
    print(f"No deals, error code={mt5.last_error()}")
    exit()

# m_order instance
m_order = CHistoryOrderInfo()

# Loop and print each order
for i, order in enumerate(history_orders):
    if m_order.select_order(order):
        print(f"""
History Order #{i}

--- Integer, Datetime & String type properties ---

Time Setup: {m_order.time_setup()}
Time Setup (ms): {m_order.time_setup_msc()}
Time Done: {m_order.time_done()}
Time Done (ms): {m_order.time_done_msc()}
Magic Number: {m_order.magic()}
Ticket: {m_order.ticket()}
Order Type: {m_order.order_type()} ({m_order.type_description()})
Order State: {m_order.state()} ({m_order.state_description()})
Expiration Time: {m_order.time_expiration()}
Filling Type: {m_order.type_filling()} ({m_order.type_filling_description()})
Time Type: {m_order.type_time()} ({m_order.type_time_description()})
Position ID: {m_order.position_id()}
Position By ID: {m_order.position_by_id()}

--- Double type properties ---

Volume Initial: {m_order.volume_initial()}
Volume Current: {m_order.volume_current()}
Price Open: {m_order.price_open()}
Price Current: {m_order.price_current()}
Stop Loss: {m_order.stop_loss()}
Take Profit: {m_order.take_profit()}
Price Stop Limit: {m_order.price_stop_limit()}

--- Access to text properties ---

Symbol: {m_order.symbol()}
Comment: {m_order.comment()}
""")

mt5.shutdown()
```

**Outputs**

```
History Order #79

--- Integer, Datetime & String type properties ---

Time Setup: 2025-05-21 23:56:17
Time Setup (ms): 1747860977335
Time Done: 2025-05-22 01:57:47
Time Done (ms): 1747868267618
Magic Number: 1001
Ticket: 153201241
Order Type: 5 (Sell Stop pending order)
Order State: 4 (Order fully executed)
Expiration Time: 2025-05-21 23:57:14.940000
Filling Type: 1 (FOK (Fill or Kill))
Time Type: 2 (ORDER_TIME_SPECIFIED)
Position ID: 153201241
Position By ID: 0

--- Double type properties ---

Volume Initial: 0.01
Volume Current: 0.0
Price Open: 1.13194
Price Current: 1.13194
Stop Loss: 0.0
Take Profit: 0.0
Price Stop Limit: 0.0

--- Access to text properties ---

Symbol: EURUSD
Comment: Sell Stop Order
```

### The CPositionInfo Class

This class provides easy access to the open position properties.

| **Python custom CPositionInfo class** | MQL5 built-in CPositionInfo class | Description |
| --- | --- | --- |
| **Integer & datetime type properties** |  |  |
| ```<br>ticket()<br>``` |  | Gets the ticket of a position, previously selected for access. |
| ```<br>time()<br>``` | [Time](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfotime) | Gets the time of position opening. |
| ```<br>time_msc()<br>``` | [TimeMsc](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfotimemsc) | Receives the time of placing a position in milliseconds since 01.01.1970. |
| ```<br>time_update()<br>``` | [TimeUpdate](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfotimeupdate) | Receives the time of position changing in seconds since 01.01.1970. |
| ```<br>time_update_msc()<br>``` | [TimeUpdateMsc](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfotimeupdatemsc) | Receives the time of position changing in milliseconds since 01.01.1970. |
| ```<br>position_type()<br>``` | [PositionType](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfopositiontype) | Gets the position type as an integer. |
| ```<br>position_type_description()<br>``` | [TypeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfotypedescription) | Gets the position type as a string |
| ```<br>magic()<br>``` | [Magic](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfomagic) | Gets the ID of the expert that opened the position. |
| ```<br>position_id()<br>``` | [Identifier](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfoidentifier) | Gets the ID of position. |
| **Double type properties** |  |  |
| ```<br>volume()<br>``` | [Volume](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfovolume) | Gets the volume of position. |
| ```<br>price_open()<br>``` | [PriceOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfopriceopen) | Gets the price of position opening. |
| ```<br>stop_loss()<br>``` | [StopLoss](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfostoploss) | Gets the price of position's Stop loss. |
| ```<br>take_profit()<br>``` | [TakeProfit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfotakeprofit) | Gets the price of position's Take profit. |
| ```<br>price_current()<br>``` | [PriceCurrent](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfopricecurrent) | Gets the current price by position symbol. |
| ```<br>profit()<br>``` | [Profit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfoprofit) | Gets the amount of current profit by position. |
| ```<br>swap() <br>``` | [Swap](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfoswap) | Gets the amount of swap by position. |
| **Access to text properties** |  |  |
| ```<br>comment()<br>``` | [Comment](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfocomment) | Gets the comment of the position. |
| ```<br>symbol()<br>``` | [Symbol](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo/cpositioninfosymbol) | Gets the name of position symbol. |
| **Selection** |  |  |
| ```<br>select_position(self, position) -> bool<br>``` |  | Selects the position object (dictionary) from a list of positions returned by the function [MetaTrader5.positions\_get()](https://www.mql5.com/en/docs/python_metatrader5/mt5positionsget_py) |

**Example Usage**

```
import MetaTrader5 as mt5
from Trade.PositionInfo import CPositionInfo

if not mt5.initialize(r"c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5\terminal64.exe"):
    print("Failed to initialize Metatrader5 Error = ",mt5.last_error())
    quit()

positions = mt5.positions_get()
m_position = CPositionInfo()

# Loop and print each position
for i, position in enumerate(positions):
    if m_position.select_position(position):
        print(f"""
Position #{i}

--- Integer type properties ---

Time Open: {m_position.time()}
Time Open (ms): {m_position.time_msc()}
Time Update: {m_position.time_update()}
Time Update (ms): {m_position.time_update_msc()}
Magic Number: {m_position.magic()}
Ticket: {m_position.ticket()}
Position Type: {m_position.position_type()} ({m_position.position_type_description()})

--- Double type properties ---

Volume: {m_position.volume()}
Price Open: {m_position.price_open()}
Price Current: {m_position.price_current()}
Stop Loss: {m_position.stop_loss()}
Take Profit: {m_position.take_profit()}
Profit: {m_position.profit()}
Swap: {m_position.swap()}

--- Access to text properties ---

Symbol: {m_position.symbol()}
Comment: {m_position.comment()}

""")

mt5.shutdown()
```

**Outputs**

```
Position #1

--- Integer type properties ---

Time Open: 2025-05-22 15:02:06
Time Open (ms): 1747915326225
Time Update: 2025-05-22 15:02:06
Time Update (ms): 1747915326225
Magic Number: 0
Ticket: 153362497
Position Type: 1 (Sell)

--- Double type properties ---

Volume: 0.1
Price Open: 1.12961
Price Current: 1.1296
Stop Loss: 0.0
Take Profit: 0.0
Profit: 0.1
Swap: 0.0

--- Access to text properties ---

Symbol: EURUSD
Comment:
```

### The CDealInfo Class

This class provides access to the deal properties from the MetaTrader 5 program.

| Python custom CDealInfo class | MQL5 built-in CDealInfo class | Description |
| --- | --- | --- |
| **Interger and datetime type properties** |  |  |
| ```<br>ticket()<br>``` |  | Gives the ticket of a selected deal |
| ```<br>time()<br>``` | [Time](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfotime) | Gets the time of deal execution. |
| ```<br>time_msc()<br>``` | [TimeMsc](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfotimemsc) | Receives the time of a deal execution in milliseconds since 01.01.1970 |
| ```<br>deal_type()<br>``` | [DealType](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfodealtype) | Gets the deal type |
| ```<br>type_description()<br>``` | [TypeDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfotypedescription) | Gets the deal type as a string. |
| ```<br>entry()<br>``` | [Entry](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfoentry) | Gets the deal direction |
| ```<br>entry_description()<br>``` | [EntryDescription](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfoentrydescription) | Gets the deal direction as a string. |
| ```<br>magic()<br>``` | [Magic](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfomagic) | Gets the ID of the expert, that executed the deal. |
| ```<br>position_id()<br>``` | [PositionId](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfopositionid) | Gets the ID of the position, in which the deal was involved. |
| **Double type properties** |  |  |
| ```<br>volume()<br>``` | [Volume](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfovolume) | Gets the volume (lot size) of the deal. |
| ```<br>price()<br>``` | [Price](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfoprice) | Gets the deal price. |
| ```<br>commission()<br>``` | [Commision](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfocommision) | Gets the commission of the deal. |
| ```<br>swap()<br>``` | [Swap](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfoswap) | Gets the amount of swap when the position is closed |
| ```<br>profit() <br>``` | [Profit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfoprofit) | Gets the financial result (profit) of the deal |
| **String type properties** |  |  |
| ```<br>symbol() <br>``` | [Symbol](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfosymbol) | Gets the name of the selected deal symbol. |
| ```<br>comment()<br>``` | [Comment](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo/cdealinfocomment) | Gets the comment of the selected deal. |
| **Selection** |  |  |
| ```<br>select_by_index(self, index)<br>``` |  | Selects the deal by index. |
| ```<br>select_deal(self, deal) -> bool<br>``` |  | Selects a deal by its object (dictionary) from list of deals returned by the function [MetaTrader5.history\_deals\_get](https://www.mql5.com/en/docs/python_metatrader5/mt5historydealsget_py#:~:text=history_deals_get-,history_deals_get,-Get%20deals%20from) |

**Example usage**

```
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from Trade.DealInfo import CDealInfo

# The date range
from_date = datetime.now() - timedelta(hours=24)
to_date = datetime.now()

if not mt5.initialize(r"c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5\terminal64.exe"):
    print("Failed to initialize Metatrader5 Error = ",mt5.last_error())
    quit()

m_deal = CDealInfo()

# Get all deals from MT5 history
deals = mt5.history_deals_get(from_date, to_date)

for i, deal in enumerate(deals):

    if (m_deal.select_deal(deal=deal)):
        print(f"""
Deal #{i}

--- integer and dateteime properties ---

Ticket: {m_deal.ticket()}
Time: {m_deal.time()}
Time (ms): {m_deal.time_msc()}
Deal Type: {m_deal.deal_type()} ({m_deal.type_description()})
Entry Type: {m_deal.entry()} ({m_deal.entry_description()})
Order: {m_deal.order()}
Magic Number: {m_deal.magic()}
Position ID: {m_deal.position_id()}

--- double type properties ---

Volume: {m_deal.volume()}
Price: {m_deal.price()}
Commission: {m_deal.commission()}
Swap: {m_deal.swap()}
Profit: {m_deal.profit()}

--- string type properties ---

Comment: {m_deal.comment()}
Symbol: {m_deal.symbol()}

External ID: {m_deal.external_id()}
""")

mt5.shutdown()
```

**Outputs**

```
Deal #53

--- integer and dateteime properties ---

Ticket: 0
Time: 2025-05-22 01:57:47
Time (ms): 1747868267618
Deal Type: 1 (SELL)
Entry Type: 0 (IN)
Order: 153201241
Magic Number: 1001
Position ID: 153201241

--- double type properties ---

Volume: 0.01
Price: 1.13194
Commission: -0.04
Swap: 0.0
Profit: 0.0

--- string type properties ---

Comment: Sell Stop Order
Symbol: EURUSD

External ID:
```

### The CTerminalInfo Class

This class provides access to the properties of the MetaTrader 5 program environment.

| Python custom CTerminalInfo class | MQL5 built-in CTerminalInfo class | Description |
| --- | --- | --- |
| **String type properties** |  |  |
| ```<br>name()<br>``` | [Name](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfoname) | Gets the name of the client terminal. |
| ```<br>company()<br>``` | [Company](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfocompany) | Gets the company name of the client terminal. |
| ```<br>language()<br>``` | [Language](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfolanguage) | Gets the language of the client terminal. |
| ```<br>path()<br>``` | [Path](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfopath) | Gets the folder of the client terminal. |
| ```<br>data_path()<br>``` | [DataPath](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfodatapath) | Gets the data folder for the client terminal. |
| ```<br>common_data_path()<br>``` | [CommonDataPath](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfocommondatapath) | Gets the common data folder of all client terminals (All MetaTrade5 apps installed on the computer. |
| **Integer type properties** |  |  |
| ```<br>build()<br>``` | [Build](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfobuild) | Gets the build number of the client terminal. |
| ```<br>is_connected()<br>``` | [IsConnected](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfoisconnected) | Gets the information about connection to trade server. |
| ```<br>is_dlls_allowed()<br>``` | [IsDLLsAllowed](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfoisdllsallowed) | Gets the information about permission of DLL usage. |
| ```<br>is_trade_allowed()<br>``` | [IsTradeAllowed](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfoistradeallowed) | Gets the information about permission to trade. |
| ```<br>is_email_enabled()<br>``` | [IsEmailEnabled](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfoisemailenabled) | Gets the information about permission to send e-mails to SMTP server and login, specified in the terminal settings. |
| ```<br>is_ftp_enabled()<br>``` | [IsFtpEnabled](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfoisftpenabled) | Gets the information about permission to send trade reports to FTP server and login, specified in the terminal settings. |
| ```<br>are_notifications_enabled()<br>``` |  | Checks whether push notifications are enabled in MetaTrader 5 terminal settings. |
| ```<br>is_community_account()<br>``` |  | Checks if the current terminal is logged into a MetaTrader community in [mql5.com](https://www.mql5.com/en/articles/18208) (this website) |
| ```<br>is_community_connection()<br>``` |  | Checks if the terminal has an active connection to the MQL5 community services. |
| ```<br>is_mqid()<br>``` |  | Checks if the user is signed in using their MQID (MetaQuotes ID). |
| ```<br>is_tradeapi_disabled()<br>``` |  | Checks if the Trade API is disabled in MetaTrader 5 settings. |
| ```<br>max_bars()<br>``` | [MaxBars](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfomaxbars) | Gets the information about the maximum number of bars on chart. |
| ```<br>code_page()<br>``` |  | Returns the integer value representing the current code page (character encoding) used by the MetaTrader 5 terminal. |
| ```<br>ping_last()<br>``` |  | Returns the last recorded ping time (in microseconds) between the MetaTrader terminal and the broker's server. |
| ```<br>community_balance()<br>``` |  | Returns the current balance of the user's MQL5 community account. |
| ```<br>retransmission()<br>``` |  | Returns the rate of data retransmission from the server to the terminal. |

**Example usage**

```
import MetaTrader5 as mt5
from Trade.TerminalInfo import CTerminalInfo

if not mt5.initialize(r"c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5\terminal64.exe"):
    print("Failed to initialize Metatrader5 Error = ",mt5.last_error())
    quit()

terminal = CTerminalInfo()

print(f"""
Terminal Information

--- String type ---

Name: {terminal.name()}
Company: {terminal.company()}
Language: {terminal.language()}
Terminal Path: {terminal.path()}
Data Path: {terminal.data_path()}
Common Data Path: {terminal.common_data_path()}

--- Integers type ---

Build: {terminal.build()}
Connected: {terminal.is_connected()}
DLLs Allowed: {terminal.is_dlls_allowed()}
Trade Allowed: {terminal.is_trade_allowed()}
Email Enabled: {terminal.is_email_enabled()}
FTP Enabled: {terminal.is_ftp_enabled()}
Notifications Enabled: {terminal.are_notifications_enabled()}
Community Account: {terminal.is_community_account()}
Community Connected: {terminal.is_community_connection()}
MQID: {terminal.is_mqid()}
Trade API Disabled: {terminal.is_tradeapi_disabled()}
Max Bars: {terminal.max_bars()}
Code Page: {terminal.code_page()}
Ping Last (μs): {terminal.ping_last()}
Community Balance: {terminal.community_balance()}
Retransmission Rate: {terminal.retransmission()}
""")

mt5.shutdown()
```

**Outputs**

```
Terminal Information

--- String type ---

Name: Pepperstone MetaTrader 5
Company: Pepperstone Group Limited
Language: English
Terminal Path: c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5
Data Path: C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\F4F6C6D7A7155578A6DEA66D12B1D40D
Common Data Path: C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\Common

--- Integers type ---

Build: 4755
Connected: True
DLLs Allowed: True
Trade Allowed: True
Email Enabled: True
FTP Enabled: False
Notifications Enabled: False
Community Account: True
Community Connected: True
MQID: False
Trade API Disabled: False
Max Bars: 100000000
Code Page: 0
Ping Last (μs): 251410
Community Balance: 900.026643
Retransmission Rate: 0.535847326494355
```

### The CTrade Class

This class provides easy access to the trade functions. Unlike the previous classes which return the information about symbols, MetaTrader 5 terminal, historical deals, and the account, this function is what we need for opening trades.

**Setting parameters**

Instead of setting parameters such as Magic Number, filling type, and deviation value in points using seprate functions like in CTrade MQL5 class, in our Python class let's configure all of those in a class constructor.

```
class CTrade:

    def __init__(self, magic_number: int, filling_type_symbol: str, deviation_points: int):
```

This reduces the room for errors as calling seprate functions could be forgotten, hence leading to runtime errors which could occur due to empty and none values.

| Python custom CTrade class | MQL5 built-in CTrade class |  |
| --- | --- | --- |
| **Operation with orders** |  |  |
| ```<br>order_open(self, symbol: str, volume: float, order_type: int, price: float,<br>	   sl: float = 0.0, tp: float = 0.0, type_time: int = mt5.ORDER_TIME_GTC, <br>	   expiration: datetime = None, comment: str = "") -> bool<br>``` | [OrderOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderopen) | Places a pending order with specified parameters. |
| ```<br>order_modify(self, ticket: int, price: float, sl: float, tp: float,<br>	     type_time: int = mt5.ORDER_TIME_GTC, <br>             expiration: datetime = None, stoplimit: float = 0.0) -> bool:<br>``` | [OrderModify](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeordermodify) | modifies the pending order with specified parameters. |
| ```<br>order_delete(self, ticket: int) -> bool<br>``` | [OrderDelete](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderdelete) | Deletes a pending order. |
| **Operations with positions** |  |  |
| ```<br>position_open(self, symbol: str, volume: float, order_type: int, <br>	      price: float, sl: float, tp: float, comment: str="") -> bool<br>``` | [PositionOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionopen) | Opens a position with specified parameters. |
| ```<br>position_modify(self, ticket: int, sl: float, tp: float) -> bool<br>``` | [PositionModify](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionmodify) | Modifies position parameters by the specified ticket. |
| ```<br>position_close(self, ticket: int, deviation: float=float("nan")) -> bool<br>``` | [PositionClose](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionclose) | Closes a position for the specified symbol. |
| **Additional methods** |  |  |
| ```<br>buy(self, volume: float, symbol: str, price: float, <br>    sl: float=0.0, tp: float=0.0, comment: str="") -> bool<br>``` | [Buy](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuy) | Opens a long position with the specified parameters |
| ```<br>sell(self, volume: float, symbol: str, price: float, <br>     sl: float=0.0, tp: float=0.0, comment: str="") -> bool<br>``` | [Sell](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesell) | Opens a short position with the specified parameters |
| ```<br>buy_limit(self, volume: float, price: float, symbol: str, <br>	  sl: float=0.0, tp: float=0.0, type_time: float=mt5.ORDER_TIME_GTC, <br>	  expiration: datetime=None, comment: str="") -> bool<br>``` | [BuyLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuylimit) | Opens a pending order of the Buy Limit type with specified parameters. |
| ```<br>sell_limit(self, volume: float, price: float, symbol: str, <br>	   sl: float=0.0, tp: float=0.0, type_time: float=mt5.ORDER_TIME_GTC, <br>	   expiration: datetime=None, comment: str="") -> bool<br>``` | [SellLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeselllimit) | Opens a pending order of the Sell Limit type with specified parameters. |
| ```<br>buy_stop(self, volume: float, price: float, symbol: str, <br>	 sl: float=0.0, tp: float=0.0, type_time: float=mt5.ORDER_TIME_GTC, <br>	 expiration: datetime=None, comment: str="") -> bool<br>``` | [BuyStop](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuystop) | Places a pending order of the Buy Stop type with specified parameters. |
| ```<br>sell_stop(self, volume: float, price: float, symbol: str, <br>	  sl: float=0.0, tp: float=0.0, type_time: float=mt5.ORDER_TIME_GTC, <br>	  expiration: datetime=None, comment: str="") -> bool<br>``` | [SellStop](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesellstop) | Places a pending order of the Sell Stop type with specified parameters |
| ```<br>buy_stop_limit(self, volume: float, price: float, symbol: str, <br>		sl: float=0.0, tp: float=0.0, type_time: float=mt5.ORDER_TIME_GTC, <br>		expiration: datetime=None, comment: str="") -> bool<br>``` | BuyStopLimi | Places a pending order of the Buy Stop limit type with specified parameters |
| ```<br>sell_stop_limit(self, volume: float, price: float, symbol: str, <br>		sl: float=0.0, tp: float=0.0, type_time: float=mt5.ORDER_TIME_GTC, <br>		expiration: datetime=None, comment: str="") -> bool<br>``` | SellStopLimit | Places a pending order of the Sell Stop limit with specified parameters. |

Now, let's use the CTrade class in Python to open a couple of positions and pending orders.

```
import MetaTrader5 as mt5
from Trade.Trade import CTrade
from Trade.SymbolInfo import CSymbolInfo
from datetime import datetime, timedelta

if not mt5.initialize(r"c:\Users\Omega Joctan\AppData\Roaming\Pepperstone MetaTrader 5\terminal64.exe"):
    print("Failed to initialize Metatrader5 Error = ",mt5.last_error())
    quit()

symbol = "EURUSD"

m_symbol = CSymbolInfo(symbol=symbol)
m_trade = CTrade(magic_number=1001,
                 deviation_points=100,
                 filling_type_symbol=symbol)

m_symbol.refresh_rates()

ask = m_symbol.ask()
bid = m_symbol.bid()

lotsize = m_symbol.lots_min()

# === Market Orders ===

m_trade.buy(volume=lotsize, symbol=symbol, price=ask, sl=0.0, tp=0.0, comment="Market Buy Pos")
m_trade.sell(volume=lotsize, symbol=symbol, price=bid, sl=0.0, tp=0.0, comment="Market Sell Pos")

# expiration time for pending orders
expiration_time = datetime.now() + timedelta(minutes=1)

# === Pending Orders ===

# Buy Limit - price below current ask
m_trade.buy_limit(volume=lotsize, symbol=symbol, price=ask - 0.0020, sl=0.0, tp=0.0, type_time=mt5.ORDER_TIME_SPECIFIED, expiration=expiration_time,
                comment="Buy Limit Order")

# Sell Limit - price above current bid
m_trade.sell_limit(volume=lotsize, symbol=symbol, price=bid + 0.0020, sl=0.0, tp=0.0, type_time=mt5.ORDER_TIME_SPECIFIED, expiration=expiration_time,
                   comment="Sell Limit Order")

# Buy Stop - price above current ask
m_trade.buy_stop(volume=lotsize, symbol=symbol, price=ask + 0.0020, sl=0.0, tp=0.0, type_time=mt5.ORDER_TIME_SPECIFIED, expiration=expiration_time,
                 comment="Buy Stop Order")

# Sell Stop - price below current bid
m_trade.sell_stop(volume=lotsize, symbol=symbol, price=bid - 0.0020, sl=0.0, tp=0.0, type_time=mt5.ORDER_TIME_SPECIFIED, expiration=expiration_time,
                  comment="Sell Stop Order")

# Buy Stop Limit - stop price above ask, limit price slightly lower (near it)
m_trade.buy_stop_limit(volume=lotsize, symbol=symbol, price=ask + 0.0020, sl=0.0, tp=0.0, type_time=mt5.ORDER_TIME_SPECIFIED, expiration=expiration_time,
                       comment="Buy Stop Limit Order")

# Sell Stop Limit - stop price below bid, limit price slightly higher (near it)
m_trade.sell_stop_limit(volume=lotsize, symbol=symbol, price=bid - 0.0020, sl=0.0, tp=0.0, type_time=mt5.ORDER_TIME_SPECIFIED, expiration=expiration_time,
                        comment="Sell Stop Limit Order")

mt5.shutdown()
```

**Outcomes**

> ![](https://c.mql5.com/2/144/bandicam_2025-05-21_20-58-41-422.png)

### Conclusion

The trade classes are one of the good things to happen in MQL5, back in a day we used to write everything from scratch, something which is extremely tiresome and leads to a plenty of bugs as I explained earlier. By extending the MetaTrader 5 python package to libraries (modules) in Python which are very similar in syntax to those in MQL5 it helps developers leverage the knowledge they already have working with MQL5 to Python applications.

These custom Python libraries can help mitigate the Intellisense support issue, by adding the "Docstrings" in your Python functions and classes, text editors such as Visual Studio Code can help document your code and highlight the parameters making the coding process fun and much easier. For example, inside the buy method in the CTrade class there is a short description of the function.

```
class CTrade:
# ....

    def buy(self, volume: float, symbol: str, price: float, sl: float=0.0, tp: float=0.0, comment: str="") -> bool:

        """
        Opens a buy (market) position.

        Args:
            volume: Trade volume (lot size)
            symbol: Trading symbol (e.g., "EURUSD")
            price: Execution price
            sl: Stop loss price (optional, default=0.0)
            tp: Take profit price (optional, default=0.0)
            comment: Position comment (optional, default="")

        Returns:
            bool: True if order was sent successfully, False otherwise
        """
```

This function will now get described in VS Code.

![](https://c.mql5.com/2/144/2186986472452.gif)

Simply put, this article serves as the documentation of the trade classes for MetaTrader5 I made in the Python programming language, please let me know your thoughts in the discussion section.

Best regards.

**Attachments Table**

| Filename & Path | Description & Usage |
| --- | --- |
| **Modules (libraries)** |  |
| Trade\\AccountInfo.py | Contains the CAccountInfo class |
| Trade\\DealInfo.py | Contains the CDealInfo class |
| Trade\\HistoryOrderInfo.py | Contains the CHistoryOrderInfo class |
| Trade\\OrderInfo.py | Contains the COrderInfo class |
| Trade\\PositionInfo.py | Contains the CPositionInfo class |
| Trade\\SymbolInfo.py | Contains the CSymbolInfo class |
| Trade\\TerminalInfo.py | Contains the CTerminalInfo class |
| Trade\\Trade.py | Contains the CTrade class |
| **Test files** |  |
| accountinfo\_test.py | A script for testing methods offered by the CAccountInfo class |
| dealinfo\_test.py | A script for testing methods offered by the CDealInfo class |
| error\_description.py | Contains function to describe error and return codes to human-readable strings |
| historyorderinfo\_test.py | A script for testing methods offfered by the CHistoryOrderInfo class |
| orderinfo\_test.py | A script for testing methods offered by the COrderInfo class |
| positioninfo\_test.py | A script for testing methods offered by the CPositionInfo class |
| symbolinfo\_test.py | A script for testing methods offered by the CSymbolInfo class |
| terminalinfo\_test.py | A script for testing methods offered by the CTerminal class |
| main.py | A script for testing the CTrade class, think of it as a final trading robot in Python |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18208.zip "Download all attachments in the single ZIP archive")

[Trade\_classes\_Python.zip](https://www.mql5.com/en/articles/download/18208/trade_classes_python.zip "Download Trade_classes_Python.zip")(43.78 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)
- [Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://www.mql5.com/en/articles/18971)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/487490)**
(2)


![Yevgeniy Koshtenko](https://c.mql5.com/avatar/2025/3/67e0f73d-60b1.jpg)

**[Yevgeniy Koshtenko](https://www.mql5.com/en/users/koshtenko)**
\|
12 Jun 2025 at 14:30

**MetaQuotes:**

Published article [Creating Python classes for trading in MetaTrader 5, similar to those presented in MQL5](https://www.mql5.com/en/articles/18208):

Author: [Omega J Msigwa](https://www.mql5.com/en/users/omegajoctan "omegajoctan")

Thank you very much. You have a great article


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
12 Jun 2025 at 18:21

thank you very much! you have a very useful article for future [projects](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not for sure ").....

![Trading with the MQL5 Economic Calendar (Part 10): Draggable Dashboard and Interactive Hover Effects for Seamless News Navigation](https://c.mql5.com/2/144/18241-trading-with-the-mql5-economic-logo__2.png)[Trading with the MQL5 Economic Calendar (Part 10): Draggable Dashboard and Interactive Hover Effects for Seamless News Navigation](https://www.mql5.com/en/articles/18241)

In this article, we enhance the MQL5 Economic Calendar by introducing a draggable dashboard that allows us to reposition the interface for better chart visibility. We implement hover effects for buttons to improve interactivity and ensure seamless navigation with a dynamically positioned scrollbar.

![Neural Networks in Trading: Transformer with Relative Encoding](https://c.mql5.com/2/97/Neural_Networks_in_Trading_Transformer_with_Relative_Encoding_____LOGO.png)[Neural Networks in Trading: Transformer with Relative Encoding](https://www.mql5.com/en/articles/16097)

Self-supervised learning can be an effective way to analyze large amounts of unlabeled data. The efficiency is provided by the adaptation of models to the specific features of financial markets, which helps improve the effectiveness of traditional methods. This article introduces an alternative attention mechanism that takes into account the relative dependencies and relationships between inputs.

![From Basic to Intermediate: Array (II)](https://c.mql5.com/2/98/Do_btsico_ao_intermediario_Array_II___LOGO3.png)[From Basic to Intermediate: Array (II)](https://www.mql5.com/en/articles/15472)

In this article, we will look at what a dynamic array and a static array are. Is there a difference between using one or the other? Or are they always the same? When should you use one and when the other type? And what about constant arrays? We will try to understand what they are designed for and consider the risks of not initializing all the values in the array.

![From Novice to Expert: Auto-Geometric Analysis System](https://c.mql5.com/2/144/18183-from-novice-to-expert-auto-logo.png)[From Novice to Expert: Auto-Geometric Analysis System](https://www.mql5.com/en/articles/18183)

Geometric patterns offer traders a concise way to interpret price action. Many analysts draw trend lines, rectangles, and other shapes by hand, and then base trading decisions on the formations they see. In this article, we explore an automated alternative: harnessing MQL5 to detect and analyze the most popular geometric patterns. We’ll break down the methodology, discuss implementation details, and highlight how automated pattern recognition can sharpen a trader's market insights.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=trouvocqrjmzayinuxwvagrfnxwjvhdw&ssn=1769092454028438133&ssn_dr=0&ssn_sr=0&fv_date=1769092454&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18208&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Building%20MQL5-Like%20Trade%20Classes%20in%20Python%20for%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909245460887357&fz_uniq=5049222489067857765&sv=2552)

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