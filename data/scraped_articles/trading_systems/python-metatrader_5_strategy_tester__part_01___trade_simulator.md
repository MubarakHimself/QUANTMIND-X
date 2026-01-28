---
title: Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator
url: https://www.mql5.com/en/articles/18971
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:34:16.623188
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ytzkedtdeqxfbghimxwprxrljcxvzcmt&ssn=1769157254593858419&ssn_dr=0&ssn_sr=0&fv_date=1769157254&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18971&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Python-MetaTrader%205%20Strategy%20Tester%20(Part%2001)%3A%20Trade%20Simulator%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915725455043330&fz_uniq=5062560986747020422&sv=2552)

MetaTrader 5 / Trading systems


**Contents**

- [Introduction](https://www.mql5.com/en/articles/18971#intro)
- [Trading simulator 101](https://www.mql5.com/en/articles/18971#trading-sim-101)
- [Calculating profits/losses made by a position](https://www.mql5.com/en/articles/18971#pos-prof-calc)
- [Simulating a position](https://www.mql5.com/en/articles/18971#simulating-position)
- [Trade validations](https://www.mql5.com/en/articles/18971#trade-validation)
- [Modifying positions](https://www.mql5.com/en/articles/18971#modifying-positions)
- [Monitoring positions](https://www.mql5.com/en/articles/18971#monitoring-positions)
- [Market's pending orders](https://www.mql5.com/en/articles/18971#pending-orders)
- [Deleting pending orders](https://www.mql5.com/en/articles/18971#deleting-pending-orders)
- [Modifying pending orders](https://www.mql5.com/en/articles/18971#modifying-pending-orders)
- [Monitoring pending orders](https://www.mql5.com/en/articles/18971#monitoring-pending-orders)
- [Monitoring the account](https://www.mql5.com/en/articles/18971#monitoring-account)
- [Realtime trade simulation in Python](https://www.mql5.com/en/articles/18971#realtime-trade-simulation-Python)
- [Realtime simulations GUI application](https://www.mql5.com/en/articles/18971#realtime-gui-sim-app)
- [Managing and controlling positions and orders externally](https://www.mql5.com/en/articles/18971#managing-pos-orders-externally)
- [Working with deals](https://www.mql5.com/en/articles/18971#working-w-deals)
- [Conclusion](https://www.mql5.com/en/articles/18971#conclusion)

_It is better to do something than to do nothing while waiting to do everything._

_— Winston Churchill._

### Introduction

The [MetaTrader5-Python package](https://www.mql5.com/en/docs/python_metatrader5) is a beneficial module that enables Python developers to develop their trading applications for the MetaTrader 5 platform. It grants developers access to the trading platform for receiving data, sending, and monitoring trades.

This module revolutionized the way we think of the MetaTrader 5 desktop application, it is not a one-dimensional app restricted to its native programming language for building trading robots known as MQL5. This trading application is flexible enough and capable of receiving trade commands from an external programming language apart from MQL5.

Although, MetaTrader5-module gives us the ability to open trades in the MetaTrader 5 platform using Python, _it is missing_ one crucial capability that all MQL5-based trading apps have — _The ability to test a fully developed trading application in the Strategy Tester._

> ![](https://c.mql5.com/2/159/tester_gif.gif)

_Can you imagine being able to build a trading robot and not being able to test it?_

While there is no shortage of useful modules in Python, as there are plenty of useful modules, libraries, and frameworks for testing the so-called _trading strategies_ such as [Backtrader](https://www.mql5.com/go?link=https://www.backtrader.com/ "https://www.backtrader.com/"), and [Backtesting.py](https://www.mql5.com/go?link=https://pypi.org/project/backtesting/ "https://pypi.org/project/backtesting/"). The issue with these Python-based tools is that they were built to test simple or, sometimes — indicators based trading strategies.

They judge trading performance based on trading signals only. They don't consider everything that goes into trading, such as broker, fees, trading account restrictions, a particular instrument (symbol) credentials, account's leverage, and much more crucial details that the MetaTrader 5 strategy tester considers.

The MetaTrader5-Python module is meant to provide users with the basic ability to get crucial information from the app, and an easy way to get started with the app using the Python programming language.

With our information and the knowledge on how the MetaTrader 5 Strategy tester works, in this short article series, we are going to build and implement a convenient (MetaTrader 5 tester-like) way for testing our Python-based trading robots.

Start by installing all Python dependencies found inside the file named **requirements.txt** attached at the end of this article.

```
pip install -r requirements.txt
```

### Trading Simulator 101

To get the ability to test trading strategies in Python, we have to make a trading simulator. This is similar to what the MetaTrader 5 Strategy Tester does; it simulates the market and runs an application or functions (trading robot or an indicator) in the process.

_Not to be confused, the Strategy tester itself offered by the MetaTrader 5 app is a trading simulator._

We won't be implementing a Graphical User Interface (GUI) like the Strategy Tester (at least for now), Let's implement a Python class for this task.

```
import MetaTrader5 as mt5

class TradeSimulator:
    def __init__(self, simulator_name: str, mt5_instance: mt5, deposit: float, leverage: str="1:100"):

        self.mt5_instance = mt5_instance
        self.simulator_name = simulator_name
```

The goal is to end up with a class constructor similar to MetaTrader 5 Strategy Tester configuration.

![](https://c.mql5.com/2/159/1926123995489.png)

- The variable _mt5\_instance_ is very crucial as it helps in monitoring a selected MetaTrader 5 app instance.
- The variable _simulator\_name_  can be used to create folders and paths that will help in distinguishing trading simulators, think of this variable as a trading robot (Expert or indicator's) name.

In the trading simulator class, we need a way to track the information about all opened orders, positions, and closed positions (deals), similarly to how MetaTrader 5 does it.

![](https://c.mql5.com/2/159/308461511430.png)

```
class TradeSimulator:
    def __init__(self, simulator_name: str, mt5_instance: mt5, deposit: float, leverage: str="1:100"):

        # .... other variables
        # ...
        # ...

        # Position's information

        self.position_info = {
            "time": None,
            "id" : 0,
            "magic": 0,
            "symbol": None,
            "type": None,
            "volume": 0.0,
            "open_price": 0.0,
            "price": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "commission": 0.0,
            "margin_required": 0.0,
            "fee": 0.0,
            "swap": 0.0,
            "profit": 0,
            "comment": 0
        }

        # Order's information

        self.order_info = self.position_info.copy()
        self.order_info["expiry_date"] = datetime
        self.order_info["expiration_mode"] = ""

        # Deal's information

        self.deal_info = self.position_info.copy()

        self.deal_info["reason"] = None # This is used to store the reason why the trade was closed, e.g. "Take Profit", "Stop Loss", etc.
        self.deal_info["direction"] = None # The only difference btn an open trade and a closed one is that the closed one has a direction showing if at that instance it was opened or closed in history

        # Containers for positions, orders, and deals

        self.positions_container = [] # a list for storing all opened trades
        self.deals_container = [] # a list for storing all deals
        self.orders_container = []
```

Table below contains a description of positions, orders, and deal information stored in the simulator class.

| Property | Description |
| --- | --- |
| time | The time of position or order execution. For a deal, this is the time of deal execution (entry or exit). |
| id | A uniquely incremented (identifier) of all orders, positions, or deals. |
| magic | The [magic number](https://www.mql5.com/en/forum/446630) of a position, order, or deal. |
| symbol | An instrument the trade was opened on, eg,. (EURUSD, USDJPY) |
| type | The type of a position for positions, the type of order for orders. |
| volume | Trading [volume (lotsize)](https://www.mql5.com/go?link=https://www.google.com/search?q=mql5+trading+lotsize "https://www.google.com/search?q=mql5+trading+lotsize") applied to a position, order, or deal. |
| open\_price | The opening price of an order or position. It can be either a closing price or an opening price of a deal, _depending on the deal's reason._ |
| price | The current price in the market, it is equal to the ask price for a buy position or _buy related pending orders,_  and bid price for a sell position or _sell related pending orders._ |
| sl | The stop loss value of an order, position, or deal. |
| tp | The take profit value of an order, position, or deal. |
| comission | Gets the amount of commision charged from a position. |
| margin\_required | Stores the required margin for such position or order to execute. |
| fee | Contains broker fees applied to a position. |
| swap | Stores the amount of swap applied to a position. |
| profit | Stores the calculated profit/loss of a position or deal. |
| comment | Stores the comment of a position, order, or deal. |
| expiration\_mode | Stores [SYMBOL\_EXPIRATION\_MODE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_order_gtc_mode:~:text=the%20ENUM_SYMBOL_ORDER_GTC_MODE%20enumeration.-,ENUM_SYMBOL_ORDER_GTC_MODE,-Identifier) for pending orders (self.order\_info). |
| expiry\_date | Stores the expiration time of an order in UTC-time format. |

All information about opened positions, placed pending orders, and executed deals is then stored in their respective arrays in a simulator for easier access.

```
        # Containers for positions, orders, and deals

        self.positions_container = [] # a list for storing all opened trades
        self.deals_container = [] # a list for storing all deals
        self.orders_container = [] # for storing all pending orders placed
```

### Calculating Profits/Loss Made By a Position

The main goal of simulating all trading activity from a trader's standpoint is to determine profits/losses that could be made using a trading robot from a specific time in history.

Below is a universal function for this task.

```
def _calculate_profit(self, action: str, symbol: str, entry_price: float, exit_price: float, lotsize: float) -> float:

    """
    Calculate profit based on entry and exit prices, lot size, tick size, and tick value.

    Args:
        action (str): The action taken, either 'buy' or 'sell'.
        entry_price (float): The price at which the position was opened.
        exit_price (float): The price at which the position was closed.
        lotsize (float): The size of the lot in terms of contract units.
    """

    if action != "buy" and action != "sell":
        print(f"Unknown order type, It can be either 'buy' or 'sell'. Received '{action}' instead.")
        return 0

    order_type = self.mt5_instance.ORDER_TYPE_BUY if action == "buy" else self.mt5_instance.ORDER_TYPE_SELL

    profit = self.mt5_instance.order_calc_profit(
        order_type,
        symbol,
        lotsize,
        entry_price,
        exit_price
    )

    return profit
```

We'll use this function to calculate profits/losses of all market orders (positions) in MetaTrader 5. By giving it an entry and exit price, an instrument (symbol), and the lot size.

```
if not mt5.initialize():
    print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
    mt5.shutdown()
    quit()

sim = TradeSimulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1000, leverage="1:500")
profit = sim._calculate_profit(action="buy",
                            symbol="EURUSD",
                            entry_price=1.17246,
                            exit_price=1.17390,
                            lotsize=0.07)

print("profit: ", profit)
```

Outputs.

```
(pystrategytester) C:\Users\Omega Joctan\OneDrive\Desktop\Python Strategy Tester>conda run --live-stream --name pystrategytester python "c:/Users/Omega Joctan/OneDrive/Desktop/Python Strategy Tester/trade_simulator.py"
profit:  10.08
```

![](https://c.mql5.com/2/159/4559719144166.png)

### Simulating a Position

In a trading simulation, a position is nothing but a bunch of calculated information resembling a trade stored in memory or disk.

Below is the base function for opening positions.

```
    def _open_position(self, pos_type: str, volume: float, symbol: str, price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> bool:

        trade_info = self.trade_info.copy()

        self.m_symbol.name(symbol)

        self.id += 1  # Increment trade ID

        trade_info["time"] = self.time
        trade_info["id"] = self.id
        trade_info["magic"] = self.magic_number
        trade_info["symbol"] = symbol
        trade_info["type"] = pos_type
        trade_info["volume"] = volume
        trade_info["price"] = price
        trade_info["sl"] = sl
        trade_info["tp"] = tp
        trade_info["commission"] = 0.0
        trade_info["fee"] = 0.0
        trade_info["swap"] = 0.0
        trade_info["profit"] = 0.0
        trade_info["comment"] = comment
        trade_info["margin_required"] = self._calculate_margin(symbol=symbol, volume=volume, price=price)

        # Append to open trades
        self.open_trades_container.append(trade_info)
        print("Trade opened successfully: ", trade_info)

        return True
```

_Again, the property, id  which resembles the ticket of a position, is automatically incremented to create a unique ticket number for every position opened in the class instance._

The property named _margin\_required_ has been the most challenging one to craft so far because, whilst the MetaTrader5 module prvodies a function to help in [margin calculation](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercalcmargin_py), it considers the current logged-in account in the MetaTrader 5 app; It uses that account's information including leverage.

Since we want a simulated account in this Python simulator, we need a custom function for calculating the required margin value for every position according to the assigned credentials of the so-called _simulated account_.

```
    def _calculate_margin(self, symbol: str, volume: float, open_price: float, margin_rate=1.0) -> float:

        """
        Calculates margin requirement similar to MetaTrader5 based on the margin mode.
        """
        self.m_symbol.name(symbol)

        if not self.m_symbol.select():
            print(f"Margin calculation failed: MetaTrader5 error = {self.mt5_instance.last_error()}")
            return 0.0

        contract_size = self.m_symbol.contract_size()
        leverage = self.leverage
        margin_mode = self.m_symbol.trade_calc_mode()

        print("Margin calculation mode: ",self.m_symbol.trade_calc_mode_description())

        tick_size = self.m_symbol.tick_size() or 0.0001
        tick_value = self.m_symbol.tick_value() or 0.0
        initial_margin = self.m_symbol.margin_initial() or 0.0
        face_value = self.m_symbol.trade_face_value()


        if margin_mode == self.mt5_instance.SYMBOL_CALC_MODE_FOREX:
            margin = (volume * contract_size * margin_rate) / leverage

        elif margin_mode == self.mt5_instance.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE:
            margin = volume * contract_size * margin_rate

        elif margin_mode == self.mt5_instance.SYMBOL_CALC_MODE_CFD:
            margin = volume * contract_size * open_price * margin_rate

        elif margin_mode == self.mt5_instance.SYMBOL_CALC_MODE_CFDLEVERAGE:
            margin = (volume * contract_size * open_price * margin_rate) / leverage

        elif margin_mode == self.mt5_instance.SYMBOL_CALC_MODE_CFDINDEX:
            margin = volume * contract_size * open_price * tick_value / tick_size * margin_rate

        elif margin_mode in [self.mt5_instance.SYMBOL_CALC_MODE_EXCH_STOCKS, self.mt5_instance.SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX]:
            margin = volume * contract_size * open_price * margin_rate

        elif margin_mode in [self.mt5_instance.SYMBOL_CALC_MODE_FUTURES,\
                             self.mt5_instance.SYMBOL_CALC_MODE_EXCH_FUTURES]:

            margin = volume * initial_margin * margin_rate

        elif margin_mode in [self.mt5_instance.SYMBOL_CALC_MODE_EXCH_BONDS, self.mt5_instance.SYMBOL_CALC_MODE_EXCH_BONDS_MOEX]:
            margin = volume * contract_size * face_value * open_price / 100

        elif margin_mode == self.mt5_instance.SYMBOL_CALC_MODE_SERV_COLLATERAL:
            margin = 0.0

        else:
            print(f"Unknown margin mode: {margin_mode}, falling back to default margin calc.")
            margin = (volume * contract_size * open_price) / leverage

        return margin
```

While the function isn't perfect, considering I couldn't find way of getting the variable _margin\_rate_  from the MetaTrader 5 app, using the MetaTrader5-Python module which seems to take part in [margin calculation formulas](https://www.mql5.com/en/book/automation/symbols/symbols_margin) used in MQL5.

Since the variable isn't available in [symbol\_info](https://www.mql5.com/en/docs/python_metatrader5/mt5symbolinfo_py), an argument named _margin\_rate_ (set to 1.0 by default) provides us a way to manually insert this value.

The process of storing a position in the container assumes nothing is wrong with the given credentials for a position. This is very wrong because, as we know, the MetaTrader 5 app has a way of checking if a trade meets certain account, symbol's, and broker's credentials before accepting it.

For example, the app checks if stop loss and take profit values aren't very tight (close to the market) by rejecting all trades that fall under this condition, it also checks if a trade is given a valid position size (volume/lot size), etc.

That being said, we need a function that returns a boolean for validating all positions. _Only positions with all valid credentials will be accepted; otherwise, rejected._

### Trade Validations

(a) Lotsize Validation

To validate a lot size (volume) of the trade, we check for three conditions.

1. If a given lot size is smaller than the minimum accepted volume for a symbol
2. If a given lot size is larger than the maximum accepted volume for a given symbol
3. If a given lotsize is a multiple of its step size (minimal volume change step for deal execution)

```
    def _position_validation(self,
                       volume: float,
                       symbol: str,
                       pos_type: str,
                       open_price: float,
                       sl: float = 0.0,
                       tp: float = 0.0,
                       expiry_date: datetime = None) -> bool:
        """
        Validates trade parameters similar to MQL5's OrderCheck()

        Returns:
            bool: True if validation passes, False with error message if fails
        """

        self.m_symbol.name(symbol) # Assign the current symbol to the CSymbolInfo class for accessing its properties

        # Get symbol properties
        symbol_info = self.m_symbol.get_info() # Get the information about the current symbol
        if symbol_info is None:
            print(f"Trade validation failed. MetaTrader5 error = {self.mt5_instance.last_error()}")
            return False

        # Validate volume

        if volume < self.m_symbol.lots_min(): # check if the received lotsize is smaller than minimum accepted lot of a symbol
            print(f"Trade validation failed: Volume ({volume}) is less than minimum allowed ({self.m_symbol.lots_min()})")
            return False

        if volume > self.m_symbol.lots_max(): # check if the received lotsize is greater than the maximum accepted lot
            print(f"Trade validation failed: Volume ({volume}) is greater than maximum allowed ({self.m_symbol.lots_max()})")
            return False

        step_count = volume / self.m_symbol.lots_step()

        if abs(step_count - round(step_count)) > 1e-7: # check if the stoploss is a multiple of the step size
            print(f"Trade validation failed: Volume ({volume}) must be a multiple of step size ({self.m_symbol.lots_step()})")
            return False

```

(b): Trade's opening price validation and slippage check

Similarly to MetaTrader 5 strategy tester, we have to ensure that a position has a valid opening price before accepting it, i.e, _It's opening price must be so close or equal to the ask price of a symbol for a buy position, and, it's opening price must be close or equal to the bid price for a sell position._

A slippage value (when given) is used for price comparisons only, _for ensuring a given entry price is close to the bid price._

```
        # Validate the opening price

        self.m_symbol.refresh_rates() # Get recent ticks information

        ask = self.m_symbol.ask()
        bid = self.m_symbol.bid()

        if ask is None or bid is None or ask==0 or bid==0:
            print("Trade Validate: Failed to Get Ask and Bid prices, Call the function market_update() to update the simulator with newly simulated price values")
            return False

        # Slippage check

        actual_price = ask if pos_type == "buy" else bid
        point = self.m_symbol.point()

        # Allowable slippage range (in absolute price)

        max_deviation = self.deviation_points * point
        lower_bound = actual_price - max_deviation
        upper_bound = actual_price + max_deviation

        # Check if requested price is within allowed slippage

        if not (lower_bound <= open_price <= upper_bound):
            print(f"Trade validation failed: {pos_type.capitalize()} price ({open_price}) is out of slippage range: {lower_bound:.5f} - {upper_bound:.5f}")
            return False
```

(c): Stop loss and take profit validation

Not all markets' orders (positions) stop loss and take profit values are acceptable by Metatrader 5 brokers; some stop loss and take profit values might be invalid or too close to the market for a position to open.

We use the same logic for checking both [stops level](https://www.mql5.com/en/articles/2555#invalid_SL_TP_for_position) and [freeze level](https://www.mql5.com/en/articles/2555#modify_in_freeze_level_prohibited).

Firstly, we check if an appropriate stop loss is received in the first place.

Then we ensure that a stop loss value is below the position's opening price and the take profit is above it for a buy trade. We do the opposite for a sell trade (stop loss must be above the opening price and the take profit must be below it).

```
# Validate stop loss and take profit levels

if sl > 0:
    if pos_type == "buy" and sl >= open_price:
        print(f"Trade validation failed: Buy stop loss ({sl}) must be below order opening price ({open_price})")
        return False
    if pos_type == "sell" and sl <= open_price:
        print(f"Trade validation failed: Sell stop loss ({sl}) must be above order opening price ({open_price})")
        return False
    if not self._check_stop_level(symbol, open_price, sl, pos_type):
        return False

if tp > 0:
    if pos_type == "buy" and tp <= open_price:
        print(f"Trade validation failed: Buy take profit ({tp}) must be above order opening price ({open_price})")
        return False
    if pos_type == "sell" and tp >= open_price:
        print(f"Trade validation failed: Sell take profit ({tp}) must be below order opening price ({open_price})")
        return False
    if not self._check_stop_level(symbol, open_price, tp, pos_type):
        return False
```

The above lines of code can be found within the function named _\_check\_stops\_level._

```
    def _check_stop_level(self, symbol: str, price: float, stop_price: float, pos_type: str) -> bool:

        """Check if stop levels comply with broker requirements"""

        self.m_symbol.name(symbol)

        # Validate symbol
        if not self.m_symbol.select():
            print(f"Failed to check stop level: Symbol {symbol}. MetaTrader5 error = {self.mt5_instance.last_error()}")
            return False

        # Check for stops level
        stop_level = self.m_symbol.stops_level()

        if pos_type == "buy":
            if stop_price > price - stop_level * self.m_symbol.point():
                print(f"Trade validation failed: Stop level too close. Must be at least {stop_level} points away")
                return False
        else:  # sell
            if stop_price < price + stop_level * self.m_symbol.point():
                print(f"Trade validation failed: Stop level too close. Must be at least {stop_level} points away")
                return False


        # Check for freeze level

        freeze_level = self.m_symbol.freeze_level()

        if pos_type == "buy":
            if stop_price > price - freeze_level * self.m_symbol.point():
                print(f"Trade validation failed: Stop level too close. Must be at least {freeze_level} points away")
                return False
        else:  # sell
            if stop_price < price + freeze_level * self.m_symbol.point():
                print(f"Trade validation failed: Stop level too close. Must be at least {freeze_level} points away")
                return False

        return True
```

_The above function returns False, if there was an invalid stop loss or take profit value detected from a buy or sell position. Otherwise, it returns True._

Finally, we call the function named _\_position\_validation_ inside the base function for opening positions. It will check the validity of a position before storing it in an array containing all positions.

```
    def _open_position(self, pos_type: str, volume: float, symbol: str, price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> bool:

        trade_info = self.trade_info.copy()

        self.m_symbol.name(symbol)

        if not self._position_validation(volume=volume, symbol=symbol, pos_type=pos_type, price=price, sl=sl, tp=tp):
            return False

        self.id += 1  # Increment trade ID

        trade_info["time"] = self.time
        trade_info["id"] = self.id

        # ... proceeds to store a trade

        # Append to open trades
        self.open_trades_container.append(trade_info)
        print("Trade opened successfully: ", trade_info)

        return True
```

To make it much more convenient to open buy and sell positions, I've created two specific functions named _buy_ and _sell_, for opening buy and sell positions, respectively. These two functions rely on the base function named _\_open\_position_ the only difference between these two and their predecessor is a variable named _pos\_type_ (for setting the type of a position). This value is explicilty applied inside the functions below.

```
    def buy(self, volume: float, symbol: str, price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> bool:
        return self._open_position("buy", volume, symbol, price, sl, tp, comment)

    def sell(self, volume: float, symbol: str, price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> bool:
        return self._open_position("sell", volume, symbol, price, sl, tp, comment)
```

_The above functions were inspired by similar functions available inside the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade)  class offered by [standard trade libraries](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) in MQL5 language._

### Modifying Positions

Being able to modify your positions is crucial for various trading and money management reasons. For example, traders often modify stop loss values in positions by moving them towards an entry or the take profit value to reduce losses or make a profit, this is referred to as trailing stops or breakeven.

Below is a function for aiding Python developers in modifying positions in a simulator.

```
    def position_modify(self, pos: dict, new_sl: float, new_tp) -> bool:

        new_position = pos.copy()

        if pos["type"] == "buy":
            if new_sl >= pos["price"]:
                print("Failed to modify sl, new_sl >= current price")
                return False

        if pos["type"] == "sell":
            if new_sl <= pos["price"]:
                print("Failed to modify sl, new_sl <= current price")
                return False

        if not self._check_stops_level(symbol=pos["symbol"], open_price=pos["open_price"], stop_price=new_sl, pos_type=pos["type"]):
            print("Failed to Modify the Stoploss")

        if not self._check_stops_level(symbol=pos["symbol"], open_price=pos["open_price"], stop_price=new_tp, pos_type=pos["type"]):
            print("Failed to Modify the Takeprofit")

        # new sl and tp values

        new_position["sl"] = new_sl
        new_position["tp"] = new_tp

        # Update the position in a container

        for i, p in enumerate(self.positions_container):
            if p["id"] == pos["id"]:
                self.positions_container[i] = new_position
                print(f"Position with id=[{pos['id']}] modified! new_sl={new_sl} new_tp={new_tp}")
                return True

        print("Failed to modify position: ID not found")

        return True
```

The process of modifying a position in MetaTrader 5 has some similar aspects to opening a new one; the above function ensures two checks are satisfied before confirming position modification.

1. Checking if the newly given stop loss is valid according to the type of position, i.e. new stop loss value must be greater than the position's current price in the market for a buy position and the opposite for a sell position.
2. Ensuring new stop loss or take profit values aren't too close to the market.

Example usage:

Let's open a simple buy position and modify its stop loss. After every one second, we increase the stop loss of such a position by subtracting 0.005.

```
stoploss = 500

ask = m_symbol.ask()
point = m_symbol.point()

sim.buy(volume=0.1, symbol=symbol, open_price=ask, sl=ask-stoploss*point)

while True: # constantly monitor trades and account metrics

    sim.monitor_pending_orders()
    sim.monitor_positions(verbose=False)

    for pos in sim.get_positions(): # go through all positions, same as in MQL5
        if pos["type"] == "buy" and pos["symbol"] == symbol: # select a buy position for the current symbol
            sim.position_modify(pos=pos, new_sl=pos["sl"]-0.005, new_tp=pos["tp"])

    sim.run_toolbox_gui()  # Run the simulator toolbox GUI

    time.sleep(5) # sleep for one second
```

Outputs.

```
Position with id=[1] modified! new_sl=1.1320700000000001 new_tp=0.0
Position with id=[1] modified! new_sl=1.1270700000000002 new_tp=0.0
Position with id=[1] modified! new_sl=1.1220700000000003 new_tp=0.0
Position with id=[1] modified! new_sl=1.1170700000000005 new_tp=0.0
```

### Monitoring Positions

Since a position is nothing but a bunch of information stored temporarily in memory, this information must be updated all the time.

For example, after opening a position, we must update its running profit or loss according to price movements in the market (recent ask and bid prices), not to mention, we need to monitor every position's exit's i.e,. if the current price on the market (bid for a buy position or ask for a sell position) is equal to either the stop loss or take profit of a trade; we close that trade.

(a): Monitoring trade's profit

Using the previously discussed function for calculating profits made by a position, we constantly monitor and update the profits made by every position.

```
    def monitor_positions(self, verbose: bool):

        # monitoring all open trades

        for pos in self.positions_container:

            self.m_symbol.name(pos["symbol"])
            self.m_symbol.refresh_rates()

            # Get ticks information for every symbol

            ask = self.m_symbol.ask()
            bid = self.m_symbol.bid()

            # update price information on all positions

            pos["price"] = ask if pos["type"] == "buy" else bid

            # Monitor and calculate the profit of a position

            pos["profit"] = self._calculate_profit(action=pos["type"], symbol=pos["symbol"], lotsize=pos["volume"], entry_price=pos["open_price"],
                                                    exit_price=(ask if pos["type"]=="buy" else bid))
```

(b): Monitoring positions exits

After _a position is opened in the strategy tester_, with or without given targets (stop loss and take profit values), _it won't close itself_ **.**

We must constantly monitor it by checking whether the current market price _(ask for a sell position and bid for a buy position)_ has reached such a desired target. _If it has reached one of the position's target, such position is closed and a deal is added to deals history._

```
    def monitor_positions(self, verbose: bool):

        # monitoring all open trades

        for pos in self.positions_container:

            self.m_symbol.name(pos["symbol"])
            self.m_symbol.refresh_rates()

            # Get ticks information for every symbol

            ask = self.m_symbol.ask()
            bid = self.m_symbol.bid()


            # ... other monitors


            # Monitor the stoploss and takeprofit situation of positions

            if pos["tp"] > 0 and ((pos["type"] == "buy" and bid >= pos["tp"]) or (pos["type"] == "sell" and ask <= pos["tp"])): # Take profit hit
                self.position_close(pos_id=pos) # close such position

            if pos["sl"] > 0 and ((pos["type"] == "buy" and bid <= pos["sl"]) or (pos["type"] == "sell" and ask >= pos["sl"])): # Stop loss hit
                self.position_close(pos_id=pos) # close such position
```

Finally, we want to print some information about every position as they are updated, similarly to how the MetaTrader 5 terminal toolbox does (it shows us active positions).

_Only when the variable named verbose = True._

```
            # Print the information about all trades (positions and orders (if any))

            if verbose:
                print(f'sim -> ticket | {trade["id"]} | symbol {trade["symbol"]} | time {trade["time"]} | type {trade["type"]} | volume {trade["volume"]} | sl {trade["sl"]} | tp {trade["tp"]} | profit {trade["profit"]:.2f}')
```

For now, we are monitoring buy and sell positions only; we'll discuss monitoring pending orders as well shortly.

### Market's Pending Orders

Unlike Market orders (positions), which are set for instant market execution, pending orders contain an order to commit (a trading operation) under the presence of a certain condition. Pending orders may also contain a time restriction on their actions — the order expiration date.

Pending orders include.

1. [Buy Limit](https://www.mql5.com/en/book/automation/experts/experts_order_type#:~:text=ORDER_TYPE_BUY_LIMIT)
2. [Buy Stop](https://www.mql5.com/en/book/automation/experts/experts_order_type#:~:text=Limit%20pending%20order-,ORDER_TYPE_BUY_STOP,-Buy%20Stop%20pending)
3. [Sell Limit](https://www.mql5.com/en/book/automation/experts/experts_order_type#:~:text=ORDER_TYPE_SELL_LIMIT)
4. [Sell Stop](https://www.mql5.com/en/book/automation/experts/experts_order_type#:~:text=ORDER_TYPE_SELL_STOP)
5. [Buy Stop Limit](https://www.mql5.com/en/book/automation/experts/experts_order_type#:~:text=ORDER_TYPE_BUY_STOP_LIMIT)
6. [Sell Stop Limit](https://www.mql5.com/en/book/automation/experts/experts_order_type#:~:text=ORDER_TYPE_SELL_STOP_LIMIT)

For now, we are going to implement the first four pending order types from the above list, in the trade simulator class, _just to get started._

Starting with the base function for placing pending orders.

_The checks:_

(a): Checking if the order type is correct.

```
    def _place_a_pending_order(self,
                               order_type: str,
                               volume: float,
                               symbol: str,
                               open_price: float,
                               sl: float = 0.0,
                               tp: float = 0.0,
                               comment: str = "",
                               expiry_date: datetime = None,
                               expiration_mode: str="gtc"
                               ):

        order_types = ["buy limit", "buy stop", "sell limit", "sell stop"]

        if order_type not in order_types:
            raise ValueError(f"Invalid pending order type, available order types include: {order_types}")

        expiration_modes = ["gtc", "daily", "daily_excluding_stops"]
        if expiration_mode not in expiration_modes:
            raise ValueError(f"Invalid Expiration mode, available modes include: {expiration_modes}")

```

(b): Ensuring all pending orders aren't so close to the market

1. Ensuring the opening price of a buy-related pending order isn't too close to the bid price.
2. Ensuring the opening price of a sell-related pending order isn't too close to the ask price.

The value of [SYMBOL\_TRADE\_STOPS\_LEVEL](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfostopslevel)  is the one that determines how close a position is to the market.

```
# Get market info

self.m_symbol.name(symbol_name=symbol) # assign symbol's name
self.m_symbol.refresh_rates() # get recent ticks from the market using the current selected symbol

if order_type in ("buy limit", "buy stop"):

    if abs(open_price - self.m_symbol.bid()) < self.m_symbol.stops_level() * self.m_symbol.point():
        print(f"Failed to open a pending order, a '{order_type}' order is too close to the market")

if order_type in ("sell limit", "sell stop"):

    if abs(open_price - self.m_symbol.ask()) < self.m_symbol.stops_level() * self.m_symbol.point():
        print(f"Failed to open a pending order, a '{order_type}' order is too close to the market")
```

(c): Ensuring a valid order expiry date is received

The expiry date or time has to be a time value greater than the current time — _a time in the future._

```
# check if the order has a valid expiry date

if expiry_date is not None: # if an expiry date is given in the first place
    if expiry_date <= self.m_symbol.time(timezone=pytz.UTC):
        print(f"Failed to place a pending order {order_type}, Invalid datetime")
        return
```

Finally, after an order passes the three checks, it gets added to the list of orders stored in the class.

```
order_info = self.order_info.copy()

self.id += 1

order_info["id"] = self.id
order_info["type"] = order_type
order_info["volume"] = volume
order_info["symbol"] = symbol
order_info["open_price"] = open_price
order_info["sl"] = sl
order_info["tp"] = tp
order_info["comment"] = comment
order_info["magic"] = self.magic_number
order_info["margin_required"] = self._calculate_margin(symbol=symbol, volume=volume, open_price=open_price)

order_info["expiry_date"] = expiry_date
order_info["expiration_mode"] = expiration_mode

self.orders_container.append(order_info) # add a valid order to it's container
```

We increment the same **id**(ticket number) used in setting positions' id, in placing pending orders as well because, _a pending order is a position in disguise_, (it is a position waiting to be opened and every position was once an order).

Using the same _id_ helps in preventing duplicate _id_ numbers in the triggered positions.

Using this base function, let's implement convenient/separate functions for placing pending orders.

Placing a Buy Stop order:

```
    def buy_stop(self, volume: float, symbol: str, open_price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "", expiry_date: datetime = None,expiration_mode: str="gtc"):

        # validate an order according to it's type

        self.m_symbol.name(symbol_name=symbol)
        self.m_symbol.refresh_rates()

        if self.m_symbol.bid() >= open_price:
            print("Failed to place a buy stop order, open price <= the bid price")
            return

        self._place_a_pending_order("buy stop", volume, symbol, open_price, sl, tp, comment, expiry_date, expiration_mode)
```

Placing a Buy Limit order:

```
    def buy_limit(self, volume: float, symbol: str, open_price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "", expiry_date: datetime = None, expiration_mode: str="gtc"):

        self.m_symbol.name(symbol_name=symbol)
        self.m_symbol.refresh_rates()

        if self.m_symbol.bid() <= open_price:
            print("Failed to place a buy limit order, open price >= current bid price")
            return

        self._place_a_pending_order("buy limit", volume, symbol, open_price, sl, tp, comment, expiry_date, expiration_mode)
```

Placing a Sell Stop order:

```
    def sell_stop(self, volume: float, symbol: str, open_price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "", expiry_date: datetime = None, expiration_mode: str="gtc"):

        self.m_symbol.name(symbol_name=symbol)
        self.m_symbol.refresh_rates()

        if self.m_symbol.ask() <= open_price:
            print("Failed to place a sell stop order, open price >= current ask price")
            return

        self._place_a_pending_order("sell stop", volume, symbol, open_price, sl, tp, comment, expiry_date, expiration_mode)
```

Placing a Sell Limit order:

```
    def sell_limit(self, volume: float, symbol: str, open_price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "", expiry_date: datetime = None, expiration_mode: str="gtc"):

        self.m_symbol.name(symbol_name=symbol)
        self.m_symbol.refresh_rates()

        if self.m_symbol.ask() >= open_price:
            print("Failed to place a sell limit order, open price <= current ask price")
            return

        self._place_a_pending_order("sell limit", volume, symbol, open_price, sl, tp, comment, expiry_date, expiration_mode)
```

In the above functions, we add conditions to ensure each order is at least placed in the right place, i.e,

1. A buy stop order is placed above the current market's price (ask price)
2. A buy limit order is placed below the current market's price (bid price)
3. A sell stop order is placed below the current market's price (bid price)
4. A sell limit order is placed above the current market's price (ask price)

### Deleting Pending Orders

Having a function responsible for deleting pending orders is as important as having a function for placing pending orders.

No checks are needed in this function, and no deal is stored once an order is deleted.

```
    def order_delete(self, selected_order: dict) -> bool:

        # delete a pending order from the orders container

        if selected_order in self.orders_container:

            self.orders_container.remove(selected_order)
            return True

        else:
            print(f"Warning: An Order with ID {selected_order['id']} not found!")
            return False
```

### Modifying Pending Orders

We need a function for modifying pending orders as well, similarly to how we have a function for modifying positions.

There are three important checks required inside a function for this task.

(a): The check to ensure the new opening price of a position is at the right place according to the type of order.

_In all pending orders, the new opening price must be placed:_

1. Above the current market's price (ask price) for a buy stop order
2. Below the current market's price (bid price) for a buy limit order
3. Below the current market's price (bid price) for a sell stop order
4. Above the current market's price (ask price) for a sell limit order

```
    def order_modify(self, order: dict, new_open_price: float, new_sl: float, new_tp: float, new_expiry: datetime = None, new_expiration_mode: str = None):
        """
         Modify an existing pending order's open price, SL/TP, and optionally its expiration settings.
        """
        new_order = order.copy()

        # Validate order type
        valid_types = ["buy limit", "buy stop", "sell limit", "sell stop"]
        if order["type"] not in valid_types:
            print(f"Invalid order type for modification: {order['type']}")
            return False

        self.m_symbol.name(order["symbol"])
        self.m_symbol.refresh_rates()

        # Ensure open price is placed logically according to type
        ask = self.m_symbol.ask()
        bid = self.m_symbol.bid()

        if order["type"] == "buy stop" and bid >= new_open_price:
            print("Failed to modify Buy Stop: new open price <= current bid price")
            return False
        if order["type"] == "buy limit" and bid <= new_open_price:
            print("Failed to modify Buy Limit: new open price >= current bid price")
            return False
        if order["type"] == "sell stop" and ask <= new_open_price:
            print("Failed to modify Sell Stop: new open price >= current ask price")
            return False
        if order["type"] == "sell limit" and ask >= new_open_price:
            print("Failed to modify Sell Limit: new open price <= current ask price")
            return False
```

(b): The check to ensure the new order's opening price isn't very close to the market.

```
# ensure the order ins't close to the market

order_type = order["type"]
if order_type in ("buy limit", "buy stop"):

    if abs(new_open_price - self.m_symbol.bid()) < self.m_symbol.stops_level() * self.m_symbol.point():
        print(f"Failed to open a pending order, a '{order_type}' order is too close to the market")
        return False

if order_type in ("sell limit", "sell stop"):

    if abs(new_open_price - self.m_symbol.ask()) < self.m_symbol.stops_level() * self.m_symbol.point():
        print(f"Failed to open a pending order, a '{order_type}' order is too close to the market")
        return False
```

(c): The check to ensure that the newly given order's expiry time is appropriate

```
if new_expiry and new_expiry <= self.m_symbol.time(timezone=pytz.UTC):
    print("Invalid Expiry date, new expiry date must be a value in the future")
```

Finally, we modify and update all orders in the container.

```
# Update the order in the container
for i, o in enumerate(self.orders_container):
    if o["id"] == order["id"]:
        self.orders_container[i] = new_order
        print(f"Order with id=[{order['id']}] modified successfully.")
        return True

print("Failed to modify order: ID not found")
return False
```

### Monitoring Pending Orders

Similarly to a position, an order is nothing other than a bunch of information stored in a list of dictionaries in the class. Once an order is stored, it has to be constantly monitored, i.e, there needs to be a code for checking if the current price (ask or bid) has reached a pending order's opening price, once the current market price hits order's opening price it gets triggered and added to a list of open positions instead.

We also need to monitor the expiration time situation in all pending orders with an expiry date and the right expiration mode, [_read more._](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#:~:text=the%20ORDER_TYPE_TIME%20modifier.-,ENUM_ORDER_TYPE_TIME,-Identifier)

```
    def monitor_pending_orders(self):

        now = datetime.now(tz=pytz.UTC)

        expired_orders = []
        triggered_orders = []

        for order in self.orders_container: # loop through all orders

            expiration_mode = order.get("expiration_mode", "gtc")
            expiry_date = order.get("expiry_date")

            # Check for expiration based on mode
            if expiration_mode == "daily" or expiration_mode == "daily_excluding_stops":
                if expiry_date and now >= expiry_date:

                    expired_orders.append(order)
                    continue  # Skip to next order

            self.m_symbol.name(symbol_name=order["symbol"])

            if not self.m_symbol.refresh_rates():
                continue

            ask = self.m_symbol.ask()
            bid = self.m_symbol.bid()
            open_price = order["open_price"]
            order_type = order["type"].lower()

            if order_type in ("buy limit", "buy stop"):
                order["price"] = self.m_symbol.ask()

            if order_type in ("sell limit", "sell stop"):
                order["price"] = self.m_symbol.bid()

            triggered = False # store the triggered condition of an order

            if order_type == "buy limit" and ask <= open_price:
                triggered = self.buy(order["volume"], order["symbol"], ask, order["sl"], order["tp"], order["comment"]) # open a buy position with credentials taken from an order

            elif order_type == "buy stop" and ask >= open_price:
                triggered = self.buy(order["volume"], order["symbol"], ask, order["sl"], order["tp"], order["comment"]) # open a buy position

            elif order_type == "sell limit" and bid >= open_price:
                triggered = self.sell(order["volume"], order["symbol"], bid, order["sl"], order["tp"], order["comment"]) # open a sell position

            elif order_type == "sell stop" and bid <= open_price:
                triggered = self.sell(order["volume"], order["symbol"], bid, order["sl"], order["tp"], order["comment"]) # open a sell position

            if triggered:
                triggered_orders.append(order) # add a triggerd order to the list

        # Clean up expired and triggered orders
        for order in expired_orders + triggered_orders:

            if order in self.orders_container:
                self.orders_container.remove(order)
```

### Monitoring the Account

After monitoring all positions and updating their credentials (including their profit/loss values), we have to update our account credentials as well, i.e, the account balance according to the simulator's deposit, equity, margin, free margin, and margin level) — all these account credentials depends on trading activities.

![](https://c.mql5.com/2/160/bandicam_2025-07-29_07-59-21-693.png)

A simulated account is monitored inside a function named _monitor\_account:_

| Account's property | Calculation | Description |
| --- | --- | --- |
| Running Profit/Loss calculation | ```<br>unrealized_pl = sum(pos['profit'] or 0 for pos in self.positions_container)<br>        <br>self.account_info["profit"] = unrealized_pl<br>``` | Calculates the sum of profits from all the opened positions in the simulator. |
| Updating account's equity | ```<br>self.account_info['equity'] = self.account_info['balance'] + unrealized_pl<br>``` | Account's equity is the result of the sum of profits\\losses from all  positions when subtracted to the account's balance. |
| Used margin | ```<br>self.account_info['margin'] = sum(pos['margin_required'] or 0 for pos in self.positions_container)<br>``` | Total used margin is the sum of the margins consumed by all positions. |
| Free margin | ```<br>self.account_info['free_margin'] = self.account_info['equity'] - self.account_info['margin']<br>``` | Free margin is the difference between the account's equity and the total margin used in the account. |
| Margin level | ```<br>self.account_info['margin_level'] = (self.account_info['equity'] / self.account_info['margin']) * 100 \<br>            if self.account_info['margin'] > 0 else 0.0<br>``` | Equals to account's equity divided by account's margin in percentage, only when the used margin is greater than zero (margin > 0). |

Finally, we print the account's credentials at the end of the function named _monitor\_account._

_Only when the argument named verbose = True._

```
    def monitor_account(self, verbose: bool):

        """Recalculates all account metrics based on current positions"""

        # 1. Calculate unrealized P/L
        unrealized_pl = sum(pos['profit'] or 0 for pos in self.open_trades_container)

        self.account_info["profit"] = unrealized_pl

        # 2. Update Equity (Balance + Floating P/L)
        self.account_info['equity'] = self.account_info['balance'] + unrealized_pl

        # 3. Calculate Used Margin
        self.account_info['margin'] = sum(pos['margin_required'] or 0 for pos in self.open_trades_container)

        # 4. Calculate Free Margin (Equity - Used Margin)
        self.account_info['free_margin'] = self.account_info['equity'] - self.account_info['margin']

        # 5. Calculate Margin Level (Equity / Margin * 100)
        self.account_info['margin_level'] = (self.account_info['equity'] / self.account_info['margin']) * 100 \
            if self.account_info['margin'] > 0 else 0.0

        if verbose:
            print(f"Balance: {self.account_info['balance']:.2f} | Equity: {self.account_info['equity']:.2f} | Profit: {self.account_info['profit']:.2f} | Margin: {self.account_info['margin']:.2f} | Free margin: {self.account_info['free_margin']} | Margin level: {self.account_info['margin_level']:.2f}%")
```

Account's balance is updated only when a trade is closed, this takes us back to the function named _position\_close._

```
    def position_close(self, selected_pos: dict) -> bool:

        # Update deal info

        deal_info = selected_pos.copy()
        deal_info["direction"] = "closed"

        # check if the reason was SL or TP according to recent tick/price information

        self.m_symbol.name(selected_pos["symbol"])
        self.m_symbol.refresh_rates()

        ask = self.m_symbol.ask()
        bid = self.m_symbol.bid()
        digits = self.m_symbol.digits()

        deal_info["reason"] = "Unknown" # Unkown deal reason if the stoploss or takeprofit wasn't hit

        if selected_pos["type"] == "buy":
            if np.isclose(selected_pos["tp"], bid, digits): # check if the current bid price is almost equal to the takeprofit
                deal_info["reason"] = "Take profit"

            elif np.isclose(selected_pos["sl"], bid, digits): # check if the current bid price is almost equal to the stoploss
                deal_info["reason"] = "Stop loss"


        if selected_pos["type"] == "sell":
            if np.isclose(selected_pos["tp"], ask, digits): # check if the current ask price is almost equal to the takeprofit
                deal_info["reason"] = "Take profit"

            elif np.isclose(selected_pos["sl"], ask, digits): # check if the current ask price is almost equal to the stoploss
                deal_info["reason"] = "Stop loss"


        self.deals_container.append(deal_info.copy()) # add the deal to the deals container

        print("Trade closed successfully: ", deal_info)

        # Save closed deal to database
        self._save_closed_deal(deal_info, self.history_db_name)

        # Remove trade from open positions

        if selected_pos in self.open_trades_container:

            # update the account balance
            self.account_info["balance"] += selected_pos["profit"]

            self.open_trades_container.remove(selected_pos)
        else:
            print(f"Warning: Position with ID {selected_pos['id']} not found!")

        return True
```

### RealTime Trade Simulation in Python

Given the ability to open trades and monitor trading activities inside the class, _TradeSimulator_, let's open our very first trades in the simulation as well as the real ones in the MetaTrader 5 desktop app. The goal is to find similarities between the trading activity in two distinct environments.

Before opening the trades, we have to be mindful of the methods used to configure crucial trading parameters in a simulator.

```
class TradeSimulator:
    def __init__(self, simulator_name: str, mt5_instance: mt5, deposit: float, leverage: str="1:100"):

    #... other functions

    def set_magicnumber(self, magic_number: int):

        self.magic_number = magic_number

    def set_deviation_in_points(self, deviation_points: int):

        self.deviation_points = deviation_points
```

The function named _set\_magicnumber_  sets the magic number for all trades in a simulator, while the function named _set\_deviation\_in\_points_  sets the slippage of all trades in the class.

After importing all necessary modules inside the file _simulator\_test.py_, we initialize MetaTrader 5-desktop app using MetaTrader5-module.

```
import MetaTrader5 as mt5
from Trade.SymbolInfo import CSymbolInfo
from Trade.Trade import CTrade
from datetime import datetime
import time
import pytz
from trade_simulator import TradeSimulator

if not mt5.initialize(): # Initialize MetaTrader5 instance
    print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
    mt5.shutdown()
    quit()
```

Followed by initializing the _TradeSimulator_ class.

```
sim = TradeSimulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

magic_number = 123456
slippage = 10

sim.set_magicnumber(magic_number=magic_number) #sets the magic number of a simulator
sim.set_deviation_in_points(deviation_points=slippage) # sets slippage of the simulator
```

We'll use the class [CTrade](https://www.mql5.com/en/articles/18208#CTrade-Python) discussed in [this article](https://www.mql5.com/en/articles/18208) to open the same trades in MetaTrader 5, _we'll compare trades opened in a simulator to those opened in MetaTrader 5._

```
m_trade = CTrade() # Initializing the CTrade class

symbol = "EURUSD"

m_trade.set_magicnumber(magic_number=magic_number) # sets the magic number of the CTrade class
m_trade.set_deviation_in_points(deviation_points=slippage) # sets slippage
m_trade.set_filling_type_by_symbol(symbol=symbol) #set filling type by the given symbol
```

We open the same trades in both a trade simulator and MetaTrader 5.

```
m_symbol = CSymbolInfo(mt5_instance=mt5)
m_symbol.name(symbol_name=symbol) # sets the symbol name for the class CSymbolInfo

if m_symbol.refresh_rates() is None: # Get recent ticks data from MetaTrader 5
    print("failed to get recent ticks data")

sim.monitor_account(verbose=True)  # calculate account credentials initially

# Open trades in a Simulator

lotsize = 0.01

if not sim.buy(volume=lotsize, symbol=symbol, open_price=m_symbol.ask(), sl=0.0, tp=0.0, comment="Test Buy Trade"):
    print("Failed to simulate a trade")

if not sim.sell(volume=lotsize, symbol=symbol, open_price=m_symbol.bid(), sl=0.0, tp=0.0, comment="Test Sell Trade"):
    print("Failed to simulate a trade")

# Open trades in MetaTrader5

if not m_trade.buy(volume=lotsize, symbol=symbol, price=m_symbol.ask(), sl=0.0, tp=0.0, comment="Test Buy Trade"):
    print("Failed to open a trade in MetaTrader5")

if not m_trade.sell(volume=lotsize, symbol=symbol, price=m_symbol.bid(), sl=0.0, tp=0.0, comment="Test Buy Trade"):
    print("Failed to open a trade in MetaTrader5")
```

Inside an infinite loop is where we want to monitor all positions and an account in a simulation.

```
while True: # constantly monitor trades and account metrics

    sim.monitor_account(verbose=True)
    sim.monitor_positions(verbose=True)

    time.sleep(1) # sleep for one second
```

Outputs.

![](https://c.mql5.com/2/160/sim_alongside_mt5.gif)

That's very unpleasant to watch. Let's create a simple GUI app to help in visualizing this trading activity in Python.

### RealTime Simulations GUI Application

For this simple app, we use the tkinter module.

```
import tkinter as tk
from tkinter import ttk
```

```
from datetime import datetime

class SimToolboxGUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trade Simulator Monitor")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        # === ACCOUNT INFO DISPLAY ===
        self.account_label = tk.Label(
            self.root,
            text="",
            font=("Courier", 8),
            anchor="w",
            justify="left",
            bg="#f0f0f0",
            fg="#333",
        )
        self.account_label.pack(fill="x", padx=5, pady=(5, 6))

        # === POSITION TABLE ===
        position_frame = tk.LabelFrame(self.root, text="Open Positions", bg="#f0f0f0")
        position_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.position_columns = [\
            "id", "symbol", "time", "type", "volume", "open_price", "sl", "tp",\
            "swap", "price", "profit", "comment"\
        ]

        self.position_tree = ttk.Treeview(position_frame, columns=self.position_columns, show="headings", height=10)
        for col in self.position_columns:
            self.position_tree.heading(col, text=col)
            self.position_tree.column(col, anchor="center", width=80)
        self.position_tree.pack(fill="both", expand=True, padx=5, pady=5)

        vsb1 = ttk.Scrollbar(position_frame, orient="vertical", command=self.position_tree.yview)
        self.position_tree.configure(yscrollcommand=vsb1.set)
        vsb1.pack(side="right", fill="y")

        # === ORDER TABLE ===
        order_frame = tk.LabelFrame(self.root, text="Pending Orders", bg="#f0f0f0")
        order_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.order_columns = [\
            "id", "symbol", "time", "type", "volume", "open_price", "sl", "tp", "price",\
            "expiry_date", "expiration_mode", "comment"\
        ]

        self.order_tree = ttk.Treeview(order_frame, columns=self.order_columns, show="headings", height=10)
        for col in self.order_columns:
            self.order_tree.heading(col, text=col)
            self.order_tree.column(col, anchor="center", width=100)
        self.order_tree.pack(fill="both", expand=True, padx=5, pady=5)

        vsb2 = ttk.Scrollbar(order_frame, orient="vertical", command=self.order_tree.yview)
        self.order_tree.configure(yscrollcommand=vsb2.set)
        vsb2.pack(side="right", fill="y")

    def update(self, account_info: dict, positions: list, orders: list):
        # === Update account info ===
        acc_text = (
            f"Balance: {account_info['balance']:.2f} | "
            f"Equity: {account_info['equity']:.2f} | "
            f"Profit: {account_info['profit']:.2f} | "
            f"Margin: {account_info['margin']:.2f} | "
            f"Free margin: {account_info['free_margin']:.5f} | "
            f"Margin level: {account_info['margin_level']:.2f}%"
        )
        self.account_label.config(text=acc_text)

        # === Refresh positions ===
        for row in self.position_tree.get_children():
            self.position_tree.delete(row)

        for pos in positions:
            row = [pos.get(col, "") for col in self.position_columns]
            self.position_tree.insert("", "end", values=row)

        # === Refresh orders ===
        for row in self.order_tree.get_children():
            self.order_tree.delete(row)

        for order in orders:
            row = []
            for col in self.order_columns:
                val = order.get(col, "")
                if isinstance(val, datetime):
                    val = val.strftime("%Y-%m-%d %H:%M:%S")
                row.append(val)
            self.order_tree.insert("", "end", values=row)

        self.root.update()

    def run(self):
        self.root.mainloop()
```

The above class creates two tables, one table for displaying orders and the other for positions. On top of the GUI, we add information about the account.

Inside the class _TradeSimulator_, we initialize this _Simulation ToolBox GUI_ in the class constructor.

Inside **trade\_simulator.py**

```
from toolbox_gui import SimToolboxGUI

class TradeSimulator:
    def __init__(self, simulator_name: str, mt5_instance: mt5, deposit: float, leverage: str="1:100"):

        # ... other variables

        self.toolbox_gui = SimToolboxGUI()  # Initialize the GUI
```

We create a separate function for updating data displayed in the GUI application.

```
class TradeSimulator:

    # ... other functions

    def run_toolbox_gui(self):

        """
        Runs the simulator toolbox GUI.
        """

        self.toolbox_gui.update(self.account_info, self.open_trades_container)
```

After calling functions for monitoring and regulating positions, orders, and the account. We call the function for updating the GUI application.

```
while True: # constantly monitor trades and account metrics

    sim.monitor_account(verbose=False)
    sim.monitor_positions(verbose=False)
    sim.monitor_orders()

    sim.run_toolbox_gui()  # Run the simulator toolbox GUI

    time.sleep(1) # sleep for one second
```

Again, let's open a few positions and orders in both MetaTrader 5 and the Python simulator, then observe the outcomes in both.

Filename: simulator\_test.py

```
if not mt5.initialize(): # Initialize MetaTrader5 instance
    print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
    mt5.shutdown()
    quit()

sim = TradeSimulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

magic_number = 123456
slippage = 10

sim.set_magicnumber(magic_number=magic_number) #sets the magic number of a simulator
sim.set_deviation_in_points(deviation_points=slippage) # sets slippage of the simulator

m_trade = CTrade() # Initializing the CTrade class

symbol = "EURUSD"

m_trade.set_magicnumber(magic_number=magic_number) # sets the magic number of the CTrade class
m_trade.set_deviation_in_points(deviation_points=slippage) # sets slippage
m_trade.set_filling_type_by_symbol(symbol=symbol) #set filling type by the given symbol

m_symbol = CSymbolInfo(mt5_instance=mt5)
m_symbol.name(symbol_name=symbol) # sets the symbol name for the class CSymbolInfo

# Open trades in a Simulator

sim.monitor_account(verbose=False)

if m_symbol.refresh_rates() is None: # Get recent ticks data from MetaTrader5
    print("failed to get recent ticks data")

# Market Orders

sim.buy(volume=0.1, symbol=symbol, open_price=m_symbol.ask())
sim.sell(volume=0.1, symbol=symbol, open_price=m_symbol.bid())

m_trade.buy(volume=0.1, symbol=symbol, price=m_symbol.ask())
m_trade.sell(volume=0.1, symbol=symbol, price=m_symbol.bid())

# Pending Orders

expiry = datetime.now(tz=pytz.UTC) + timedelta(days=1) # expiration date for pending orders
price_gap = 0.0005

# Buy Stop: place above current ask
sim.buy_stop(volume=0.1, symbol=symbol, open_price=m_symbol.ask() + price_gap, sl=0.0, tp=0.0,
             comment="Buy Stop Example", expiry_date=expiry, expiration_mode="daily")

m_trade.buy_stop(volume=0.1, symbol=symbol, price=m_symbol.ask() + price_gap)

# Buy Limit: place below current bid
sim.buy_limit(volume=0.1, symbol=symbol, open_price=m_symbol.bid() - price_gap, sl=0.0, tp=0.0,
              comment="Buy Limit Example", expiry_date=expiry, expiration_mode="daily_excluding_stops")

m_trade.buy_limit(volume=0.1, symbol=symbol, price=m_symbol.bid() - price_gap)

# Sell Stop: place below current bid

sim.sell_stop(volume=0.1, symbol=symbol, open_price=m_symbol.bid() - price_gap, sl=0.0, tp=0.0,
              comment="Sell Stop Example", expiry_date=expiry, expiration_mode="gtc")

m_trade.sell_stop(volume=0.1, symbol=symbol, price=m_symbol.ask() - price_gap)

# Sell Limit: place above current ask
sim.sell_limit(volume=0.1, symbol=symbol, open_price=m_symbol.ask() + price_gap, sl=0.0, tp=0.0,
               comment="Sell Limit Example", expiry_date=expiry, expiration_mode="gtc")

m_trade.sell_limit(volume=0.1, symbol=symbol, price=m_symbol.bid() + price_gap)

while True: # constantly monitor trades and account metrics

    sim.monitor_account()
    sim.monitor_pending_orders()
    sim.monitor_positions(verbose=False)
    sim.monitor_orders()

    sim.run_toolbox_gui()  # Run the simulator toolbox GUI

    time.sleep(1) # sleep for one second
```

Outputs.

![](https://c.mql5.com/2/160/closed_deals_table.gif)

Great, our simulated trading outcomes aren't very close to the actual trading outcomes, and they aren't further away either, _that's great progress._

### Managing and Controlling Positions and Orders Externally

Being able to get information about open positions and control them outside the simulator is very crucial, _that's what algorithmic trading is._

For instance, many trading strategies require the knowledge of previously opened positions. For example, a trading strategy might require a robot to open a buy position only if a position in such direction and instrument doesn't exist.

That being said, below is a table containing functions that let you access all orders, positions, and deals outside the class named _TradeSimulator_.

| Function | Returns |
| --- | --- |
| ```<br>def get_positions(self) -> list:<br>``` | Returns all open positions from a container. |
| ```<br>def get_orders(self) -> list:<br>``` | Returns all open orders from a container. |
| ```<br>def get_deals(self, start_time: datetime = None, end_time: datetime = None, from_db: bool = False) -> list: <br>``` | Returns all deals executed between a specific time interval given by the two variables (start\_time and end\_time). <br>_An optional variable named **from\_db**, helps in deciding between selecting deals stored temporarily in memory or from the database._ |

Example usage:

```
sim.buy(volume=0.1, symbol=symbol, open_price=m_symbol.ask())
sim.sell(volume=0.1, symbol=symbol, open_price=m_symbol.bid())

price_gap = 0.0005
# Buy Stop: place above current ask
sim.buy_stop(volume=0.1, symbol=symbol, open_price=m_symbol.ask() + price_gap)

print("Positions total: ",len(sim.get_positions()))
print("Orders total: ",len(sim.get_orders()))

now = m_symbol.time(timezone=pytz.UTC)
start_time = now - timedelta(minutes=5)
end_time = now

print("Deals total: ",len(sim.get_deals(start_time=start_time,
                                        end_time=end_time,
                                        from_db=False
                              )))
```

Outputs.

```
(pystrategytester) C:\Users\Omega Joctan\OneDrive\Desktop\Python Strategy Tester>conda run --live-stream --name pystrategytester python "c:/Users/Omega Joctan/OneDrive/Desktop/Python Strategy
Tester/simulator_test.py"
Trade opened successfully:  {'time': datetime.datetime(2025, 7, 31, 9, 59, 51, tzinfo=<UTC>), 'id': 1, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'buy', 'volume': 0.1, 'open_price': 1.14597, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': ''}
Trade opened successfully:  {'time': datetime.datetime(2025, 7, 31, 9, 59, 51, tzinfo=<UTC>), 'id': 2, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'sell', 'volume': 0.1, 'open_price': 1.14589, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': ''}
Margin calculation mode:   Calculation of profit and margin for Forex
Positions total:  2
Orders total:  1
Deals total:  2
```

When selecting the deals, you should use the symbol's time in UTC, the same we used in opening positions and orders, instead of the current local time to avoid time discrepancies.

These functions will then allow us to introduce specific conditions to our trading strategies.

(a): Checking if a specific trade type exists in a simulation

This is very common for monitoring trades. In some trading strategies, we often want to open certain positions and orders only when they don't exist.

```
if not mt5.initialize(): # Initialize MetaTrader5 instance
    print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
    mt5.shutdown()
    quit()

sim = TradeSimulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")

magic_number = 123456
slippage = 10

sim.set_magicnumber(magic_number=magic_number) #sets the magic number of a simulator
sim.set_deviation_in_points(deviation_points=slippage) # sets slippage of the simulator

symbol = "EURUSD"
m_symbol = CSymbolInfo(mt5_instance=mt5)
m_symbol.name(symbol_name=symbol) # sets the symbol name for the class CSymbolInfo

def is_position_exists(type: str) -> bool:

    for pos in sim.get_positions():
        if pos["magic"] == magic_number and pos["symbol"] == symbol and pos["type"] == type:
            return True # position exists

    return False

while True: #imitating the OnTick function offered in MQL5 language

    sim.monitor_pending_orders()
    sim.monitor_positions(verbose=False)
    sim.monitor_account(verbose=False)

    sim.run_toolbox_gui()  # Run the simulator toolbox GUI

    if m_symbol.refresh_rates() is None: # Get recent ticks data from MetaTrader5
        # print("failed to get recent ticks data")
        continue

    if not is_position_exists("buy"): # open a buy trade in a simulator if it doesn't exist
        sim.buy(volume=0.1, symbol=symbol, open_price=m_symbol.ask())

    if not is_position_exists("sell"): # open a sell trade in a simulator if it doesn't exist
        sim.sell(volume=0.1, symbol=symbol, open_price=m_symbol.bid())

    time.sleep(1) # sleep for one second
```

Outputs.

```
(pystrategytester) C:\Users\Omega Joctan\OneDrive\Desktop\Python Strategy Tester>conda run --live-stream --name pystrategytester python "c:/Users/Omega Joctan/OneDrive/Desktop/Python Strategy
Tester/simulator_test.py"
Trade opened successfully:  {'time': datetime.datetime(2025, 7, 31, 10, 13, 18, tzinfo=<UTC>), 'id': 1, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'buy', 'volume': 0.1, 'open_price': 1.14565, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': ''}
Trade opened successfully:  {'time': datetime.datetime(2025, 7, 31, 10, 13, 18, tzinfo=<UTC>), 'id': 2, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'sell', 'volume': 0.1, 'open_price': 1.14557, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': ''}
```

Only two distinct positions were opened (buy and sell positions).

_This is a familiar interface to the one offered in MQL5, we often use for checking if a certain position exists._

(b): Closing a specific position

```
def close_positions(type: str):

    for pos in sim.get_positions():
        if pos["magic"] == magic_number and pos["symbol"] == symbol and pos["type"] == type:
            sim.position_close(pos)
```

Some strategies might need to close specific trades upon a specific programmed condition; in that regard, the above function or a similar approach becomes handy.

Let's open two positions (buy and sell positions) and close a buy position.

```
while True:

    sim.monitor_pending_orders()
    sim.monitor_positions(verbose=False)
    sim.monitor_account(verbose=False)

    sim.run_toolbox_gui()  # Run the simulator toolbox GUI

    if m_symbol.refresh_rates() is None: # Get recent ticks data from MetaTrader5
        # print("failed to get recent ticks data")
        continue

    if not is_position_exists("buy"): # open a buy trade in a simulator if it doesn't exist
        sim.buy(volume=0.1, symbol=symbol, open_price=m_symbol.ask())

    close_positions("buy") # close all buy positions

    if not is_position_exists("sell"): # open a sell trade in a simulator if it doesn't exist
        sim.sell(volume=0.1, symbol=symbol, open_price=m_symbol.bid())

    time.sleep(1) # sleep for one second
```

Outputs.

```
(pystrategytester) C:\Users\Omega Joctan\OneDrive\Desktop\Python Strategy Tester>conda run --live-stream --name pystrategytester python "c:/Users/Omega Joctan/OneDrive/Desktop/Python Strategy
Tester/simulator_test.py"
Trade opened successfully:  {'time': datetime.datetime(2025, 7, 31, 10, 50, 35, tzinfo=<UTC>), 'id': 1, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'buy', 'volume': 0.1, 'open_price': 1.14447, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': ''}
Trade closed successfully:  {'time': datetime.datetime(2025, 7, 31, 10, 50, 35, tzinfo=<UTC>), 'id': 1, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'buy', 'volume': 0.1, 'open_price': 1.14447, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': '', 'direction': 'closed', 'reason': 'Take profit'}
Trade opened successfully:  {'time': datetime.datetime(2025, 7, 31, 10, 50, 35, tzinfo=<UTC>), 'id': 2, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'sell', 'volume': 0.1, 'open_price': 1.14439, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': ''}
Trade opened successfully:  {'time': datetime.datetime(2025, 7, 31, 10, 50, 37, tzinfo=<UTC>), 'id': 3, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'buy', 'volume': 0.1, 'open_price': 1.14446, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': ''}
Trade closed successfully:  {'time': datetime.datetime(2025, 7, 31, 10, 50, 37, tzinfo=<UTC>), 'id': 3, 'magic': 123456, 'symbol': 'EURUSD', 'type': 'buy', 'volume': 0.1, 'open_price': 1.14446, 'price': 0.0, 'sl': 0.0, 'tp': 0.0, 'commission': 0.0, 'margin_required': 20.0, 'fee': 0.0, 'swap': 0.0, 'profit': 0.0, 'comment': '', 'direction': 'closed', 'reason': 'Take profit'}
```

### Working with Deals

In MetaTrader 5, a deal represents the actual execution of a trade — it is the outcome of an order. Each deal is based on a specific order, but a single order can generate multiple deals (e.g., if the order is filled in parts).

Deals are created when.

1. A position is opened,
2. A position is partially or fully closed,
3. Or an order (like a limit or stop order) is triggered and executed.

In other words, both entry and exit executions are recorded as deals.

Unlike orders and positions, which can be modified temporarily, deals are immutable and are always stored in the trading history. _**They serve as a permanent record of executed trades and cannot be altered or deleted.**_

![](https://c.mql5.com/2/160/5322345185899.png)

At the end of both **_\_position\_open_** and _position\_close_  functions that open and close positions respectively, a deal is added to a list named **deals\_container** found in the class constructor.

```
    def _open_position(self, pos_type: str, volume: float, symbol: str, price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> bool:

        trade_info = self.trade_info.copy()

        # ... other operations
        # ...

        # Append to open trades
        self.open_trades_container.append(trade_info)
        print("Trade opened successfully: ", trade_info)

        # Track deal
        self.deal_info.update(trade_info)
        self.deal_info["direction"] = "opened"
        self.deal_info["reason"] = "Expert"
        self.deals_container.append(self.deal_info.copy())
```

```
    def position_close(self, selected_pos: dict) -> bool:

        # Update deal info

        deal_info = selected_pos.copy()
        deal_info["direction"] = "closed"

        # ... other operations

        deal_info["reason"] = "Unknown" # Unkown deal reason if the stoploss or takeprofit wasn't hit

        if selected_pos["type"] == "buy":
            if np.isclose(selected_pos["tp"], bid, digits): # check if the current bid price is almost equal to the takeprofit
                deal_info["reason"] = "Take profit"

            elif np.isclose(selected_pos["sl"], bid, digits): # check if the current bid price is almost equal to the stoploss
                deal_info["reason"] = "Stop loss"


        if selected_pos["type"] == "sell":
            if np.isclose(selected_pos["tp"], ask, digits): # check if the current ask price is almost equal to the takeprofit
                deal_info["reason"] = "Take profit"

            elif np.isclose(selected_pos["sl"], ask, digits): # check if the current ask price is almost equal to the stoploss
                deal_info["reason"] = "Stop loss"


        self.deals_container.append(deal_info.copy()) # add the deal to the deals container

        print("Trade closed successfully: ", deal_info)
```

However, storing deals opened by the simulator in a list/array isn't ideal, because this information will be lost as soon as the program is closed. Let's store them in a SQLite3 database and make their record permanent unless modified or deleted, _similarly to how Metatrader 5 does it._

```
    def _create_deals_db(self, db_name: str):

        """
         Creates a SQLite database to store trade history and account information.

        Args:
            db_name (str): The name of the database file.
        """

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create tables if they do not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS closed_deals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT,
                magic INTEGER,
                symbol TEXT,
                type TEXT,
                direction TEXT,
                volume REAL,
                price REAL,
                sl REAL,
                tp REAL,
                commission REAL,
                margin_required REAL,
                fee REAL,
                swap REAL,
                profit REAL,
                comment TEXT,
                reason TEXT
            )
        ''')

        conn.commit()
        conn.close()
```

The above function is called inside the class _TradeSimulator_, in the constructor.

```
class TradeSimulator:
    def __init__(self, simulator_name: str, mt5_instance: mt5, deposit: float, leverage: str="1:100"):

        # ... other variables
        # ...

        # Database for trade history

        self.sim_folder = "Simulations"

        os.makedirs(self.sim_folder, exist_ok=True)  # Ensure the simulations path exists

        # Create the database file name

        self.history_db_name = os.path.join(self.sim_folder, self.simulator_name+".db")
        self._create_deals_db(self.history_db_name)
```

After creating a database similar to the simulator's name given by the variable _simulator\_name_ **,**  the function named _\_create\_deals\_db_ creates a table named _closed\_deals_ if it doesn't exist.

We also need a function for saving every single deal into the database.

```
    def _save_deal(self, deal: dict, db_name: str):
        """
            Saves a closed deal to the SQLite database.
        """

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO closed_deals (
                time, magic, symbol, type, direction, volume, price, sl, tp,
                commission, margin_required, fee, swap, profit, comment, reason
            ) VALUES (
                :time, :magic, :symbol, :type, :direction, :volume, :price, :sl, :tp,
                :commission, :margin_required, :fee, :swap, :profit, :comment, :reason
            );
        """, deal)

        conn.commit()
        conn.close()
```

Notice that, we are not adding a column named **id** to the database? This is because the column named id in the database table is set to [AUTOINCREMENT](https://www.mql5.com/go?link=https://www.sqlite.org/autoinc.html%23%3a%7e%3atext%3d3.%2520The%2520AUTOINCREMENT%2520Keyword "https://www.sqlite.org/autoinc.html#:~:text=3.%20The%20AUTOINCREMENT%20Keyword") to ensure every deal is assigned a unique id throughout history from 0 to positive infinity.

We have to save all deals to the database inside the functions responsible for opening and closing positions after storing them to the list named _deals\_container._

Inside the function named _position\_close._

```
    def position_close(self, selected_pos: dict) -> bool:

        # Update deal info

        deal_info = selected_pos.copy()
        deal_info["direction"] = "closed"


        #...
        #...

        print("Trade closed successfully: ", deal_info)

        # Save closed deal to database
        self._save_deal(deal_info, self.history_db_name)
```

Inside the function named _\_open\_position_ **.**

```
    def _open_position(self, pos_type: str, volume: float, symbol: str, price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> bool:

        trade_info = self.trade_info.copy()

        #...
        #...
        #...

        self.deals_container.append(self.deal_info.copy())

        # Log to database
        self._save_deal(self.deal_info, self.history_db_name)

        return True
```

Below is the SQLite database containing all deals made in the past hours and days.

![](https://c.mql5.com/2/160/1908423924023.gif)

### Final Thoughts

While implementing this initial part of a MetaTrader 5 simulator, I can't help but appreciate how sophisticated the MetaTrader 5 strategy tester is. Many things are happening in the background of this tool apart from just executing trades.

To this point, you might be asking yourself Is this simulator necessary? Because we have implemented a trade simulator that opens trades in what looks like a real account, something which isn't different from what the MetaTrader 5 application does when using MetaTrader5-Python module?

The goal of this article was to understand the dynamics of a trade simulator, by simulating some simple trades and ensuring they are very similar to the ones opened in a real account, we can say that we are edging to our goal.

It is also fair to say that this simulator is nowhere near complete/perfect compared to the MetaTrader 5 platform's strategy tester. There are still many things missing or not done properly, _it's challenging to keep track of all details to be honest_, so, if you have thoughts and opinions, or want to collaborate in the project, here is the link to a repository on GitHub -> [https://github.com/MegaJoctan/PyMetaTester](https://www.mql5.com/go?link=https://github.com/MegaJoctan/PyMetaTester "https://github.com/MegaJoctan/PyMetaTester").

**What's next?**

In the current Trade simulator discussed above, we extracted crucial information from the market, such as the current ask and bid price, alongside other crucial information about the selected symbol. In the next article(s), we will discuss about different ways we can extract ticks data and iterate this information within a loop to mimick the strategy tester historical testing behaviour.

Peace out.

**Attachments Table**

| Filename | Description & Usage |
| --- | --- |
| requirements.txt | Contains all Python dependencies used in this project. |
| trade\_simulator.py | It has the TradeSimulator class, which hosts the entire trading simulator. |
| simulator\_test.py | A playground script for testing the trade simulator discussed. |
| toolbox\_gui.py | Contains a simple-MetaTrader 5-like GUI application for displaying trades and account balance information |
| Trade\\SymbolInfo.py | Contains the class named CSymbolInfo, which provides all information from MetaTrader 5 about a particular symbol. |
| Trade\\Trade.py | Contains the class named CTrade, which provides functions for opening positions and orders in MetaTrader 5 using the metatrader5-Python module. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18971.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/18971/attachments.zip "Download Attachments.zip")(17.05 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/492926)**
(1)


![Anton du Plessis](https://c.mql5.com/avatar/2019/11/5DC08949-7A98.png)

**[Anton du Plessis](https://www.mql5.com/en/users/adpatza)**
\|
10 Aug 2025 at 10:19

Thank you for your pioneering work. I'm really looking forward to try this out.


![MQL5 Trading Tools (Part 8): Enhanced Informational Dashboard with Draggable and Minimizable Features](https://c.mql5.com/2/162/19059-mql5-trading-tools-part-8-enhanced-logo__2.png)[MQL5 Trading Tools (Part 8): Enhanced Informational Dashboard with Draggable and Minimizable Features](https://www.mql5.com/en/articles/19059)

In this article, we develop an enhanced informational dashboard that upgrades the previous part by adding draggable and minimizable features for improved user interaction, while maintaining real-time monitoring of multi-symbol positions and account metrics.

![Building a Trading System (Part 2): The Science of Position Sizing](https://c.mql5.com/2/162/18991-building-a-profitable-trading-logo.png)[Building a Trading System (Part 2): The Science of Position Sizing](https://www.mql5.com/en/articles/18991)

Even with a positive-expectancy system, position sizing determines whether you thrive or collapse. It’s the pivot of risk management—translating statistical edges into real-world results while safeguarding your capital.

![Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://c.mql5.com/2/160/19014-mastering-log-records-part-logo.png)[Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://www.mql5.com/en/articles/19014)

We created a log suppression system in the Logify library. It details how the CLogifySuppression class reduces console noise by applying configurable rules to avoid repetitive or irrelevant messages. We also cover the external configuration framework, validation mechanisms, and comprehensive testing to ensure robustness and flexibility in log capture during bot or indicator development.

![Statistical Arbitrage Through Cointegrated Stocks (Part 2): Expert Advisor, Backtests, and Optimization](https://c.mql5.com/2/162/19052-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 2): Expert Advisor, Backtests, and Optimization](https://www.mql5.com/en/articles/19052)

This article presents a sample Expert Advisor implementation for trading a basket of four Nasdaq stocks. The stocks were initially filtered based on Pearson correlation tests. The filtered group was then tested for cointegration with Johansen tests. Finally, the cointegrated spread was tested for stationarity with the ADF and KPSS tests. Here we will see some notes about this process and the results of the backtests after a small optimization.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/18971&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062560986747020422)

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