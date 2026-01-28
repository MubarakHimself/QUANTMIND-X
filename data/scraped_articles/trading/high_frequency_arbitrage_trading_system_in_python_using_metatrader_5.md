---
title: High frequency arbitrage trading system in Python using MetaTrader 5
url: https://www.mql5.com/en/articles/15964
categories: Trading, Trading Systems, Machine Learning
relevance_score: 6
scraped_at: 2026-01-22T17:56:36.771246
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/15964&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049501588927655075)

MetaTrader 5 / Trading


### Introduction

Foreign exchange market. Algorithmic strategies. Python and MetaTrader 5. This came together when I started working on an arbitrage trading system. The idea was simple - create a high-frequency system to find price imbalances. What did all this lead to in the end?

I used MetaTrader 5 API most often during this period. I decided to calculate synthetic cross rates. I decided not to limit myself to ten or a hundred. The number has exceeded one thousand.

Risk management was a separate task. System architecture, algorithms, decision making - we will analyze everything here. I will show the results of backtesting and live trading. And of course, I will share ideas for the future. Who knows, maybe one of you would like to develop this topic further? I hope that my work will be in demand. I would like to believe that it will contribute to the development of algorithmic trading. Maybe someone will take it as a basis and create something even more effective in the world of high-frequency arbitrage. After all, that is the essence of science - moving forward based on the experience of predecessors. Let's get straight to the point.

### Introduction to Forex arbitrage trading

Let's figure out what it really is.

An analogy can be drawn with currency exchange. Let's say you can buy USD for EUR in one place, immediately sell them for GBP in another, and then exchange GBP back for EUR and end up with profit. This is arbitrage in its simplest form.

In fact, it is a little more complicated. Forex is a huge, decentralized market. There are a large number of banks, brokers, funds here. And everyone has their own exchange rates. More often than not, they do not match. This is where we have an opportunity for arbitrage. But don't think that this is easy money. Typically these price discrepancies last for only a few seconds. Or even milliseconds. It is almost impossible to make it in time. This requires powerful computers and fast algorithms.

There are also different types of arbitrage. A simple one is when we profit on the difference in rates in different places. A complex one is when we use cross rates. For example, we calculate how much GBP will cost in USD and EUR, and compare it with the direct GBP/EUR exchange rate.

The list does not end there. There is also time arbitrage. Here we profit on the difference in prices at different points in time. Bought now, sold in a minute. Of course, the process seems simple. But the main problem is that we do not know where the price will go in a minute. These are the main risks. The market may reverse faster than you can activate the desired order. Or your broker may delay executing orders. In general, there are quite a lot of difficulties and risks. Despite all the difficulties, Forex arbitrage is a rather popular system. There are serious financial resources involved here and enough traders who specialize only in this type of trading.

Now, after a short introduction, let's get down to our strategy.

### Overview of technologies used: Python and MetaTrader 5

So, Python and MetaTrader 5.

Python is versatile and easy to understand programming language. It is not for nothing that it is preferred by both novice and experienced developers. And it is best suited for data analysis.

On the other hand, MetaTrader 5. This is a platform familiar to every Forex trader. It is reliable and not complicated. And it is also quite functional – real-time quotes, trading robots, and technical analysis. All in one application. To achieve positive results, we need to combine all of this.

Python takes data from MetaTrader 5, handles it using its libraries, and then sends commands back to MetaTrader 5 to execute trades. Of course, there are difficulties. But together these applications are very efficient.

A [special library from the developers](https://www.mql5.com/en/docs/python_metatrader5) is available for working with MetaTrader 5 from Python. To activate it, you just need to install it. After doing this, we are able to receive quotes, send orders and manage positions. Everything is the same as in the terminal itself, only now Python capabilities are also used.

What features and capabilities are now available to us? There are quite a lot of them now. For example, we are able to automate trading and conduct complex analysis of historical data. We can even create our own trading platform. This is already a task for advanced users, but it is also possible.

### Setting up the environment: installing necessary libraries and connecting to MetaTrader 5

We will start our workflow with Python. If you do not have it yet, visit python.org. You also need to set the ADD TO PATCH consent.

Our next step is libraries. We will need a few of them. The main one is MetaTrader 5. Installation does not require any special skills.

Open the command line and type:

```
pip install MetaTrader5 pandas numpy
```

Press Enter and go drink some coffee. Or tea. Or whatever you prefer.

Is everything set? Now it is time to connect to MetaTrader 5.

The first thing you need to do is install MetaTrader 5 itself. Download it from your broker. Be sure to remember the path to the terminal. Typically, it looks like this: "C:\\ProgramFiles\\MetaTrader 5\\terminal64.exe".

Now open Python and type:

```
import MetaTrader5 as mt5

if not mt5.initialize(path="C:/Program Files/MetaTrader 5/terminal64.exe"):
    print("Alas! Failed to connect :(")
    mt5.shutdown()
else:
    print("Hooray! Connection successful!")
```

If everything starts, proceed to the next part.

### Code structure: main functions and their purpose

Let's start with 'imports'. Here we have imports, such as: MetaTrader5, pandas, datetime, pytz... Next, there are functions.

- The first function is remove\_duplicate\_indices. It makes sure that there are no duplicates in our data.
- Next comes get\_mt5\_data. It accesses MetaTrader 5 functions and extracts the required data for the last 24 hours.
- get\_currency\_data — very interesting function. It calls get\_mt5\_data for a bunch of currency pairs. AUDUSD, EURUSD, GBPJPY and many more pairs.
- The next one is calculate\_synthetic\_prices. This feature is a real achievement. It produces hundreds of synthetic prices while handling currency pairs.
- analyze\_arbitrage looks for arbitrage opportunities by comparing real prices with synthetic ones. All findings are saved in a CSV file.
- open\_test\_limit\_order — another powerful code unit. When an arbitrage opportunity is found, this function opens a test order. But no more than 10 open trades at the same time.

And finally, the 'main' function. It manages this entire process by calling functions in the right order.

It all ends with an endless loop. It runs the entire loop every 5 minutes, but only during working hours. This is the structure we have. It is simple, yet efficient.

### Getting data from MetaTrader 5: get\_mt5\_data function

The first task is to receive data from the terminal.

```
if not mt5.initialize(path=terminal_path):
    print(f"Failed to connect to MetaTrader 5 terminal at {terminal_path}")
    return None
timezone = pytz.timezone("Etc/UTC")
utc_from = datetime.now(timezone) - timedelta(days=1)
```

Note that we use UTC. Because in the world of Forex there is no room for time zone confusion.

Now the most important thing is getting ticks:

```
ticks = mt5.copy_ticks_from(symbol, utc_from, count, mt5.COPY_TICKS_ALL)
```

The data has been received? Great! Now we need to handle it. To do this, we use pandas:

```
ticks_frame = pd.DataFrame(ticks)
ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s')
```

Voila! Now we have our own DataFrame with data. It is already prepared for analysis.

But what if something goes wrong? Don't worry! Our function has this covered too:

```
if ticks is None:
    print(f"Failed to fetch data for {symbol}")
    return None
```

It will simply report a problem and return None.

### Handling multiple currency pairs: get\_currency\_data function

We dive further into the system - the get\_currency\_data function. Let's take a look at the code:

```
def get_currency_data():
    # Define currency pairs and the amount of data
    symbols = ["AUDUSD", "AUDJPY", "CADJPY", "AUDCHF", "AUDNZD", "USDCAD", "USDCHF", "USDJPY", "NZDUSD", "GBPUSD", "EURUSD", "CADCHF", "CHFJPY", "NZDCAD", "NZDCHF", "NZDJPY", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD"]
    count = 1000  # number of data points for each currency pair
    data = {}
    for symbol in symbols:
        df = get_mt5_data(symbol, count, terminal_path)
        if df is not None:
            data[symbol] = df[['time', 'bid', 'ask']].set_index('time')
    return data
```

It all starts with defining currency pairs. The list includes AUDUSD, EURUSD, GBPJPY and other instruments that are well known to us.

Now we move on to the next step. The function creates an empty 'data' dictionary. It will also be filled with the necessary data later.

Now our function starts its work. It will go through the list of currency pairs. For each pair, it calls get\_mt5\_data. If get\_mt5\_data returns data (and not None), our function takes only the most important: time, bid, and ask.

And here, finally, is the grand finale. The function returns a dictionary filled with data.

Now we get get\_currency\_data. It is small, powerful, simple but effective.

### Calculation of 2000 synthetic prices: Strategy and implementation

We dive into the basics of our system - calculate\_synthetic\_prices function. It allows us to get our synthesized data.

Let's take a look at the code:

```
def calculate_synthetic_prices(data):
    synthetic_prices = {}

    # Remove duplicate indices from all DataFrames in the data dictionary
    for key in data:
        data[key] = remove_duplicate_indices(data[key])

    # Calculate synthetic prices for all pairs using multiple methods
    pairs = [('AUDUSD', 'USDCHF'), ('AUDUSD', 'NZDUSD'), ('AUDUSD', 'USDJPY'),\
             ('USDCHF', 'USDCAD'), ('USDCHF', 'NZDCHF'), ('USDCHF', 'CHFJPY'),\
             ('USDJPY', 'USDCAD'), ('USDJPY', 'NZDJPY'), ('USDJPY', 'GBPJPY'),\
             ('NZDUSD', 'NZDCAD'), ('NZDUSD', 'NZDCHF'), ('NZDUSD', 'NZDJPY'),\
             ('GBPUSD', 'GBPCAD'), ('GBPUSD', 'GBPCHF'), ('GBPUSD', 'GBPJPY'),\
             ('EURUSD', 'EURCAD'), ('EURUSD', 'EURCHF'), ('EURUSD', 'EURJPY'),\
             ('CADCHF', 'CADJPY'), ('CADCHF', 'GBPCAD'), ('CADCHF', 'EURCAD'),\
             ('CHFJPY', 'GBPCHF'), ('CHFJPY', 'EURCHF'), ('CHFJPY', 'NZDCHF'),\
             ('NZDCAD', 'NZDJPY'), ('NZDCAD', 'GBPNZD'), ('NZDCAD', 'EURNZD'),\
             ('NZDCHF', 'NZDJPY'), ('NZDCHF', 'GBPNZD'), ('NZDCHF', 'EURNZD'),\
             ('NZDJPY', 'GBPNZD'), ('NZDJPY', 'EURNZD')]

    method_count = 1
    for pair1, pair2 in pairs:
        print(f"Calculating synthetic price for {pair1} and {pair2} using method {method_count}")
        synthetic_prices[f'{pair1}_{method_count}'] = data[pair1]['bid'] / data[pair2]['ask']
        method_count += 1
        print(f"Calculating synthetic price for {pair1} and {pair2} using method {method_count}")
        synthetic_prices[f'{pair1}_{method_count}'] = data[pair1]['bid'] / data[pair2]['bid']
        method_count += 1

    return pd.DataFrame(synthetic_prices)
```

### Analyzing arbitrage opportunities: analyze\_arbitrage function

First, we create an empty dictionary synthetic\_prices. We will also fill it with data. Then we will go through all the data and remove duplicate indices to avoid errors in the future.

The next step is the 'pairs' list. These are our currency pairs that we will use for synthesis. Then another process begins. We run a loop through all pairs. For each pair, we calculate the synthetic price in two ways:

1. Divide bid of the first pair by ask of the second one.
2. Divide bid of the first pair by bid of the second one.

Each time we increase our method\_count. As a result, we get 2000 synthetic pairs!

This is how the calculate\_synthetic\_prices function works. It does not just calculate prices, it actually creates new opportunities. This feature gives great results in the form of arbitrage opportunities!

### Visualizing results: Saving data to CSV

Let's look at the analyze\_arbitrage function. It does not just analyze data, it searches for what it needs in a stream of numbers. Let's take a look at it:

```
def analyze_arbitrage(data, synthetic_prices, method_count):
    # Calculate spreads for each pair
    spreads = {}
    for pair in data.keys():
        for i in range(1, method_count + 1):
            synthetic_pair = f'{pair}_{i}'
            if synthetic_pair in synthetic_prices.columns:
                print(f"Analyzing arbitrage opportunity for {synthetic_pair}")
                spreads[synthetic_pair] = data[pair]['bid'] - synthetic_prices[synthetic_pair]
    # Identify arbitrage opportunities
    arbitrage_opportunities = pd.DataFrame(spreads) > 0.00008
    print("Arbitrage opportunities:")
    print(arbitrage_opportunities)
    # Save the full table of arbitrage opportunities to a CSV file
    arbitrage_opportunities.to_csv('arbitrage_opportunities.csv')
    return arbitrage_opportunities
```

First, our function creates an empty 'spreads' dictionary. We will also fill it with data.

Let's move on to the next step. The function runs through all currency pairs and their synthetic analogues. For each pair, it calculates the spread - the difference between the real bid price and the synthetic price.

```
spreads[synthetic_pair] = data[pair]['bid'] - synthetic_prices[synthetic_pair]
```

This string plays a rather important role. It finds the difference between the real and synthetic price. If this difference is positive, we have an arbitrage opportunity.

To get more serious results, we use the number of 0.00008:

```
arbitrage_opportunities = pd.DataFrame(spreads) > 0.00008
```

This string sorts out all possibilities less than 8 points. This way we will get opportunities with a higher probability of profit.

Here is the next step:

```
arbitrage_opportunities.to_csv('arbitrage_opportunities.csv')
```

Now all our data is saved to a CSV file. Now we can study them, analyze them, plot charts - in general, do productive work. All this is made possible thanks to the following function - analyze\_arbitrage. It does not just analyze, it seeks out, finds and saves arbitrage opportunities.

### Opening test orders: open\_test\_limit\_order function

Next, let's consider the open\_test\_limit\_order function. It will open our orders for us.

Let's take a look:

```
def open_test_limit_order(symbol, order_type, price, volume, take_profit, stop_loss, terminal_path):
    if not mt5.initialize(path=terminal_path):
        print(f"Failed to connect to MetaTrader 5 terminal at {terminal_path}")
        return None
    symbol_info = mt5.symbol_info(symbol)
    positions_total = mt5.positions_total()
    if symbol_info is None:
        print(f"Instrument not found: {symbol}")
        return None
    if positions_total >= MAX_OPEN_TRADES:
        print("MAX POSITIONS TOTAL!")
        return None
    # Check if symbol_info is None before accessing its attributes
    if symbol_info is not None:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 30,
            "magic": 123456,
            "comment": "Stochastic Stupi Sustem",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "tp": price + take_profit * symbol_info.point if order_type == mt5.ORDER_TYPE_BUY else price - take_profit * symbol_info.point,
            "sl": price - stop_loss * symbol_info.point if order_type == mt5.ORDER_TYPE_BUY else price + stop_loss * symbol_info.point,
        }
        result = mt5.order_send(request)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Test limit order placed for {symbol}")
            return result.order
        else:
            print(f"Error: Test limit order not placed for {symbol}, retcode={result.retcode if result is not None else 'None'}")
            return None
    else:
        print(f"Error: Symbol info not found for {symbol}")
        return None
```

The first thing our function does is try to connect to the MetaTrader 5 terminal. Then it checks if the instrument we want to trade even exists.

The following code:

```
if positions_total >= MAX_OPEN_TRADES:
    print("MAX POSITIONS TOTAL!")
    return None
```

This check ensures that we do not open too many positions.

Now the next step is to generate a request to open an order. There are quite a lot of parameters here. Order type, volume, price, deviation, magic number, comment... If everything goes well, the function tells us about it. If not, the message appears.

This is how the open\_test\_limit\_order function works. This is our connection with the market. In a way, it performs the functions of a broker.

### Temporary trading restrictions: work during certain hours

Now let's talk about trading hours.

```
if current_time >= datetime.strptime("23:30", "%H:%M").time() or current_time <= datetime.strptime("05:00", "%H:%M").time():
    print("Current time is between 23:30 and 05:00. Skipping execution.")
    time.sleep(300)  # Wait for 5 minutes before checking again
    continue
```

What is going on here? Our system checks the time. If the clock shows between 11:30 PM and 5:00 AM, it sees that these are not trading hours and goes into standby mode for 5 minutes. Then it activates, checks the time again and, if it is still early, goes into standby mode again.

Why do we need this? There are reasons for this. First, liquidity. At night there is usually less of it. Second, spreads. At night they expand. Third, news. The most important ones usually come out during working hours.

### Runtime loop and error handling

Let's take a look at the 'main' function. It is like a ship captain, but instead of a steering wheel, there is a keyboard. What does it do? All is simple:

1. Collecting data
2. Calculating synthetic prices
3. Looking for arbitrage opportunities
4. Opening orders

There is also a little error handling.

```
def main():
    data = get_currency_data()
    synthetic_prices = calculate_synthetic_prices(data)
    method_count = 2000  # Define the method_count variable here
    arbitrage_opportunities = analyze_arbitrage(data, synthetic_prices, method_count)

    # Trade based on arbitrage opportunities
    for symbol in arbitrage_opportunities.columns:
        if arbitrage_opportunities[symbol].any():
            direction = "BUY" if arbitrage_opportunities[symbol].iloc[0] else "SELL"
            symbol = symbol.split('_')[0]  # Remove the index from the symbol
            symbol_info = mt5.symbol_info_tick(symbol)
            if symbol_info is not None:
                price = symbol_info.bid if direction == "BUY" else symbol_info.ask
                take_profit = 450
                stop_loss = 200
                order = open_test_limit_order(symbol, mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL, price, 0.50, take_profit, stop_loss, terminal_path)
            else:
                print(f"Error: Symbol info tick not found for {symbol}")
```

### System scalability: Adding new currency pairs and methods

Do you want to add a new currency pair? Simply include it to this list:

```
symbols = ["EURUSD", "GBPUSD", "USDJPY", ... , "YOURPAIR"]
```

The system now knows about the new pair. . What about the new calculation methods?

```
def calculate_synthetic_prices(data):
    # ... existing code ...

    # Add a new method
    synthetic_prices[f'{pair1}_{method_count}'] = data[pair1]['ask'] / data[pair2]['bid']
    method_count += 1
```

### Testing and backtesting of the arbitrage system

Let's talk about backtesting. This is a really important point for any trading system. Our arbitrage system is no exception.

What did we do? We ran our strategy through historical data. Why? To understand how efficient it is. Our code starts with get\_historical\_data. This function retrieves old data from MetaTrader 5. Without this data, we will not be able to work productively.

Then comes calculate\_synthetic\_prices. Here we calculate synthetic exchange rates. This is a key part of our arbitrage strategy. Analyze\_arbitrage is our opportunity detector. It compares real prices with synthetic ones and finds the difference, so we can get potential profit. simulate\_trade is almost a trading process. However, it occurs in test mode. This is a very important process: it is better to make a mistake in the simulation than to lose real money.

Finally, backtest\_arbitrage\_system puts it all together and runs our strategy through historical data. Day after day, deal after deal.

```
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

# Path to MetaTrader 5 terminal
terminal_path = "C:/Program Files/ForexBroker - MetaTrader 5/Arima/terminal64.exe"

def remove_duplicate_indices(df):
    """Removes duplicate indices, keeping only the first row with a unique index."""
    return df[~df.index.duplicated(keep='first')]

def get_historical_data(start_date, end_date, terminal_path):
    if not mt5.initialize(path=terminal_path):
        print(f"Failed to connect to MetaTrader 5 terminal at {terminal_path}")
        return None

    symbols = ["AUDUSD", "AUDJPY", "CADJPY", "AUDCHF", "AUDNZD", "USDCAD", "USDCHF", "USDJPY", "NZDUSD", "GBPUSD", "EURUSD", "CADCHF", "CHFJPY", "NZDCAD", "NZDCHF", "NZDJPY", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD"]

    historical_data = {}
    for symbol in symbols:
        timeframe = mt5.TIMEFRAME_M1
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df[['open', 'high', 'low', 'close']]
            df['bid'] = df['close']  # Simplification: use 'close' as 'bid'
            df['ask'] = df['close'] + 0.000001  # Simplification: add spread
            historical_data[symbol] = df

    mt5.shutdown()
    return historical_data

def calculate_synthetic_prices(data):
    synthetic_prices = {}
    pairs = [('AUDUSD', 'USDCHF'), ('AUDUSD', 'NZDUSD'), ('AUDUSD', 'USDJPY'),\
             ('USDCHF', 'USDCAD'), ('USDCHF', 'NZDCHF'), ('USDCHF', 'CHFJPY'),\
             ('USDJPY', 'USDCAD'), ('USDJPY', 'NZDJPY'), ('USDJPY', 'GBPJPY'),\
             ('NZDUSD', 'NZDCAD'), ('NZDUSD', 'NZDCHF'), ('NZDUSD', 'NZDJPY'),\
             ('GBPUSD', 'GBPCAD'), ('GBPUSD', 'GBPCHF'), ('GBPUSD', 'GBPJPY'),\
             ('EURUSD', 'EURCAD'), ('EURUSD', 'EURCHF'), ('EURUSD', 'EURJPY'),\
             ('CADCHF', 'CADJPY'), ('CADCHF', 'GBPCAD'), ('CADCHF', 'EURCAD'),\
             ('CHFJPY', 'GBPCHF'), ('CHFJPY', 'EURCHF'), ('CHFJPY', 'NZDCHF'),\
             ('NZDCAD', 'NZDJPY'), ('NZDCAD', 'GBPNZD'), ('NZDCAD', 'EURNZD'),\
             ('NZDCHF', 'NZDJPY'), ('NZDCHF', 'GBPNZD'), ('NZDCHF', 'EURNZD'),\
             ('NZDJPY', 'GBPNZD'), ('NZDJPY', 'EURNZD')]

    for pair1, pair2 in pairs:
        if pair1 in data and pair2 in data:
            synthetic_prices[f'{pair1}_{pair2}_1'] = data[pair1]['bid'] / data[pair2]['ask']
            synthetic_prices[f'{pair1}_{pair2}_2'] = data[pair1]['bid'] / data[pair2]['bid']

    return pd.DataFrame(synthetic_prices)

def analyze_arbitrage(data, synthetic_prices):
    spreads = {}
    for pair in data.keys():
        for synth_pair in synthetic_prices.columns:
            if pair in synth_pair:
                spreads[synth_pair] = data[pair]['bid'] - synthetic_prices[synth_pair]

    arbitrage_opportunities = pd.DataFrame(spreads) > 0.00008
    return arbitrage_opportunities

def simulate_trade(data, direction, entry_price, take_profit, stop_loss):
    for i, row in data.iterrows():
        current_price = row['bid'] if direction == "BUY" else row['ask']

        if direction == "BUY":
            if current_price >= entry_price + take_profit:
                return {'profit': take_profit * 800, 'duration': i}
            elif current_price <= entry_price - stop_loss:
                return {'profit': -stop_loss * 400, 'duration': i}
        else:  # SELL
            if current_price <= entry_price - take_profit:
                return {'profit': take_profit * 800, 'duration': i}
            elif current_price >= entry_price + stop_loss:
                return {'profit': -stop_loss * 400, 'duration': i}

    # If the loop completes without hitting TP or SL, close at the last price
    last_price = data['bid'].iloc[-1] if direction == "BUY" else data['ask'].iloc[-1]
    profit = (last_price - entry_price) * 100000 if direction == "BUY" else (entry_price - last_price) * 100000
    return {'profit': profit, 'duration': len(data)}

def backtest_arbitrage_system(historical_data, start_date, end_date):
    equity_curve = [10000]  # Starting with $10,000
    trades = []
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    for current_date in dates:
        print(f"Backtesting for date: {current_date.date()}")

        # Get data for the current day
        data = {symbol: df[df.index.date == current_date.date()] for symbol, df in historical_data.items()}

        # Skip if no data for the current day
        if all(df.empty for df in data.values()):
            continue

        synthetic_prices = calculate_synthetic_prices(data)
        arbitrage_opportunities = analyze_arbitrage(data, synthetic_prices)

        # Simulate trades based on arbitrage opportunities
        for symbol in arbitrage_opportunities.columns:
            if arbitrage_opportunities[symbol].any():
                direction = "BUY" if arbitrage_opportunities[symbol].iloc[0] else "SELL"
                base_symbol = symbol.split('_')[0]
                if base_symbol in data and not data[base_symbol].empty:
                    price = data[base_symbol]['bid'].iloc[-1] if direction == "BUY" else data[base_symbol]['ask'].iloc[-1]
                    take_profit = 800 * 0.00001  # Convert to price
                    stop_loss = 400 * 0.00001  # Convert to price

                    # Simulate trade
                    trade_result = simulate_trade(data[base_symbol], direction, price, take_profit, stop_loss)
                    trades.append(trade_result)

                    # Update equity curve
                    equity_curve.append(equity_curve[-1] + trade_result['profit'])

    return equity_curve, trades

def main():
    start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2024, 8, 31, tzinfo=pytz.UTC)  # Backtest for January-August 2024

    print("Fetching historical data...")
    historical_data = get_historical_data(start_date, end_date, terminal_path)

    if historical_data is None:
        print("Failed to fetch historical data. Exiting.")
        return

    print("Starting backtest...")
    equity_curve, trades = backtest_arbitrage_system(historical_data, start_date, end_date)

    total_profit = sum(trade['profit'] for trade in trades)
    win_rate = sum(1 for trade in trades if trade['profit'] > 0) / len(trades) if trades else 0

    print(f"Backtest completed. Results:")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Final Equity: ${equity_curve[-1]:.2f}")

    # Plot equity curve
    plt.figure(figsize=(15, 10))
    plt.plot(equity_curve)
    plt.title('Equity Curve: Backtest Results')
    plt.xlabel('Trade Number')
    plt.ylabel('Account Balance ($)')
    plt.savefig('equity_curve.png')
    plt.close()

    print("Equity curve saved as 'equity_curve.png'.")

if __name__ == "__main__":
    main()
```

Why is this important? Because backtesting shows how efficient our system is. Is it profitable or does it drain your deposit? What is a drawdown? What is the percentage of winning trades? We learn all this from the backtest.

Of course, past results do not guarantee future ones. The market is changing. But without a backtest, we will not get any results. Knowing the result, we know roughly what to expect. Another important point - backtesting helps to optimize the system. We change the parameters and look at the result again and again. So, step by step, we make our system better.

Here is the result of our system backtest:

![](https://c.mql5.com/2/140/equity_curve__3.png)

Here is a test of the system in MetaTrader 5:

![](https://c.mql5.com/2/140/ArbyBot__1.png)

And here is the code of the MQL5 EA for the system:

```
//+------------------------------------------------------------------+
//|                                                 TrissBotDemo.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
// Input parameters
input int MAX_OPEN_TRADES = 10;
input double VOLUME = 0.50;
input int TAKE_PROFIT = 450;
input int STOP_LOSS = 200;
input double MIN_SPREAD = 0.00008;

// Global variables
string symbols[] = {"AUDUSD", "AUDJPY", "CADJPY", "AUDCHF", "AUDNZD", "USDCAD", "USDCHF", "USDJPY", "NZDUSD", "GBPUSD", "EURUSD", "CADCHF", "CHFJPY", "NZDCAD", "NZDCHF", "NZDJPY", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD"};
int symbolsTotal;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    symbolsTotal = ArraySize(symbols);
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Cleanup code here
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if(!IsTradeAllowed()) return;

    datetime currentTime = TimeGMT();
    if(currentTime >= StringToTime("23:30:00") || currentTime <= StringToTime("05:00:00"))
    {
        Print("Current time is between 23:30 and 05:00. Skipping execution.");
        return;
    }

    AnalyzeAndTrade();
}

//+------------------------------------------------------------------+
//| Analyze arbitrage opportunities and trade                        |
//+------------------------------------------------------------------+
void AnalyzeAndTrade()
{
    double synthetic_prices[];
    ArrayResize(synthetic_prices, symbolsTotal);

    for(int i = 0; i < symbolsTotal; i++)
    {
        synthetic_prices[i] = CalculateSyntheticPrice(symbols[i]);
        double currentPrice = SymbolInfoDouble(symbols[i], SYMBOL_BID);

        if(MathAbs(currentPrice - synthetic_prices[i]) > MIN_SPREAD)
        {
            if(currentPrice > synthetic_prices[i])
            {
                OpenOrder(symbols[i], ORDER_TYPE_SELL);
            }
            else
            {
                OpenOrder(symbols[i], ORDER_TYPE_BUY);
            }
        }

    }
}

//+------------------------------------------------------------------+
//| Calculate synthetic price for a symbol                           |
//+------------------------------------------------------------------+
double CalculateSyntheticPrice(string symbol)
{
    // This is a simplified version. You need to implement the logic
    // to calculate synthetic prices based on your specific method
    return SymbolInfoDouble(symbol, SYMBOL_ASK);
}

//+------------------------------------------------------------------+
//| Open a new order                                                 |
//+------------------------------------------------------------------+
void OpenOrder(string symbol, ENUM_ORDER_TYPE orderType)
{
    if(PositionsTotal() >= MAX_OPEN_TRADES)
    {
        Print("MAX POSITIONS TOTAL!");
        return;
    }

    double price = (orderType == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);

    double tp = (orderType == ORDER_TYPE_BUY) ? price + TAKE_PROFIT * point : price - TAKE_PROFIT * point;
    double sl = (orderType == ORDER_TYPE_BUY) ? price - STOP_LOSS * point : price + STOP_LOSS * point;

    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.symbol = symbol;
    request.volume = VOLUME;
    request.type = orderType;
    request.price = price;
    request.deviation = 30;
    request.magic = 123456;
    request.comment = "ArbitrageAdvisor";
    request.type_time = ORDER_TIME_GTC;
    request.type_filling = ORDER_FILLING_IOC;
    request.tp = tp;
    request.sl = sl;

    if(!OrderSend(request, result))
    {
        Print("OrderSend error ", GetLastError());
        return;
    }

    if(result.retcode == TRADE_RETCODE_DONE)
    {
        Print("Order placed successfully");
    }
    else
    {
        Print("Order failed with retcode ", result.retcode);
    }
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool IsTradeAllowed()
{
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
    {
        Print("Trade is not allowed in the terminal");
        return false;
    }

    if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
    {
        Print("Trade is not allowed in the Expert Advisor");
        return false;
    }

    return true;
}
```

### Possible improvements and legality of the system for brokers, or how not to hit a liquidity provider with limit orders

Our system has other potential difficulties. Brokers and liquidity providers often frown upon such systems. Why? Because we are essentially taking the necessary liquidity from the market. They even came up with a special term for this - Toxic Order Flow.

This is a real problem. We literally suck liquidity out of the system with our market orders. Everyone needs it: both large players and small traders. Of course, this has its consequences.

What to do in this situation? There is a compromise - limit orders.

But this does not solve all the problems: the Toxic Order Flow label is placed not so much because of the absorption of the current liquidity from the market, but because of the high loads on servicing such a flow of orders. I have not solved this problem yet. For example, spending, say, USD 100 on servicing a huge flow of arbitrageur transactions, receiving a commission of, say, USD 50 from it, is unprofitable. So perhaps the key here is high turnover and high lot sizes, as well as high turnover speed. Then brokers might also be ready to pay rebates.

Now we get down to the code. How can we improve it? First, we may add a function for handling limit orders. There is also a lot of work here - we need to think through the logic of waiting and canceling unexecuted orders.

Machine learning might be an interesting idea for improving the system. I suggest that it may be possible to train our system to predict which arbitrage opportunities are most likely to work.

### Conclusion

Let's sum it up. We have created a system that looks for arbitrage opportunities. Remember that the system does not solve all your financial problems.

We have sorted out backtesting. It works with time-based data, and even better, it allows us to see how our system would have worked in the past. But remember - past results do not guarantee future ones. The market is a complex mechanism that is constantly changing.

But you know what's most important? Not a code. Not algorithms. But you. Your desire to learn, experiment, make mistakes and try again. This is truly priceless.

So do not stop there. This system is just the beginning of your journey in the world of algorithmic trading. Use it as a starting point for new ideas and new strategies. Just as in life, the main thing in trading is balance. The balance between risk and caution, greed and rationality, complexity and simplicity.

Good luck on this exciting journey, and may your algorithms always be one step ahead of the market!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15964](https://www.mql5.com/ru/articles/15964)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15964.zip "Download all attachments in the single ZIP archive")

[TrissBotDemo.mq5](https://www.mql5.com/en/articles/download/15964/trissbotdemo.mq5 "Download TrissBotDemo.mq5")(5.51 KB)

[Shtenco\_Arbitrage\_Live.py](https://www.mql5.com/en/articles/download/15964/shtenco_arbitrage_live.py "Download Shtenco_Arbitrage_Live.py")(8.68 KB)

[Shtenco\_Arbitrage\_Backtest.py](https://www.mql5.com/en/articles/download/15964/shtenco_arbitrage_backtest.py "Download Shtenco_Arbitrage_Backtest.py")(7.83 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forex arbitrage trading: Analyzing synthetic currencies movements and their mean reversion](https://www.mql5.com/en/articles/17512)
- [Forex arbitrage trading: A simple synthetic market maker bot to get started](https://www.mql5.com/en/articles/17424)
- [Build a Remote Forex Risk Management System in Python](https://www.mql5.com/en/articles/17410)
- [Currency pair strength indicator in pure MQL5](https://www.mql5.com/en/articles/17303)
- [Capital management in trading and the trader's home accounting program with a database](https://www.mql5.com/en/articles/17282)
- [Analyzing all price movement options on the IBM quantum computer](https://www.mql5.com/en/articles/17171)
- [Fibonacci in Forex (Part I): Examining the Price-Time Relationship](https://www.mql5.com/en/articles/17168)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/486095)**
(8)


![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
28 Oct 2024 at 16:09

**pivomoe [#](https://www.mql5.com/ru/forum/475133#comment_54919395):**

What is the Bid of the first pair ? The first pair is:

AUDUSD is also a pair. AUD to USD.

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
28 Oct 2024 at 20:12

**pivomoe [#](https://www.mql5.com/ru/forum/475133#comment_54919395):**

Please explain what this is about:

Here are the pairs:

What is the Bid of the first pair ? The first pair is:

This is how synthetics builds. Not through difference, but division. And not simple - but... read.....


![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
21 Nov 2024 at 10:59

```
ticks = mt5.copy_ticks_from(symbol, utc_from, count, mt5.COPY_TICKS_ALL)
```

All installed. This is what comes up in ticks:

array(\[b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'',\
\
...\
\
b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'',\
\
b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'',\
\
b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'', b''\],

dtype='\|V0')

And here we already get an exit on time:

```
ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s')
```

![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
22 Nov 2024 at 09:11

The code from the example [https://www.mql5.com/ru/docs/python\_metatrader5/mt5copyticksfrom\_py](https://www.mql5.com/ru/docs/python_metatrader5/mt5copyticksfrom_py) doesn't work either

```
>>>  timezone = pytz.timezone("Etc/UTC")
>>>  utc_from = datetime(2020, 1, 10, tzinfo=timezone)
>>>  ticks = mt5.copy_ticks_from("EURUSD", utc_from, 100000, mt5.COPY_TICKS_ALL)
>>>
>>> print("Received ticks:",len(ticks))
Получено тиков: 100000
>>> print("Let's take the resulting ticks as they are.")
Выведем полученные тики как есть
>>>  count = 0
>>> for tick in ticks:
...     count+=1
...     print(tick)
...     if count >= 100:
...         break
...
b''
b''
b''
b''
```

Anyway, what is python like? How to prepare it? It's unclear...

![Yaochi Lin](https://c.mql5.com/avatar/2019/9/5D6D70D6-6167.jpg)

**[Yaochi Lin](https://www.mql5.com/en/users/yuochi)**
\|
28 Aug 2025 at 22:25

**MetaQuotes:**

New article [Python High Frequency Arbitrage Trading System with MetaTrader 5](https://www.mql5.com/en/articles/15964) has been released:

Author: [Yevgeniy Koshtenko](https://www.mql5.com/en/users/Koshtenko "Koshtenko")

Does this really work? It's not supported by regular brokers


![Finding custom currency pair patterns in Python using MetaTrader 5](https://c.mql5.com/2/99/Finding_Custom_Currency_Pair_Patterns_in_Python_Using_MetaTrader_5___LOGO.png)[Finding custom currency pair patterns in Python using MetaTrader 5](https://www.mql5.com/en/articles/15965)

Are there any repeating patterns and regularities in the Forex market? I decided to create my own pattern analysis system using Python and MetaTrader 5. A kind of symbiosis of math and programming for conquering Forex.

![MQL5 Wizard Techniques you should know (Part 63): Using Patterns of DeMarker and Envelope Channels](https://c.mql5.com/2/140/MQL5_Wizard_Techniques_you_should_know_Part_63___LOGO.png)[MQL5 Wizard Techniques you should know (Part 63): Using Patterns of DeMarker and Envelope Channels](https://www.mql5.com/en/articles/17987)

The DeMarker Oscillator and the Envelope indicator are momentum and support/resistance tools that can be paired when developing an Expert Advisor. We therefore examine on a pattern by pattern basis what could be of use and what potentially avoid. We are using, as always, a wizard assembled Expert Advisor together with the Patterns-Usage functions that are built into the Expert Signal Class.

![Creating a Trading Administrator Panel in MQL5 (Part XI): Modern feature communications interface (I)](https://c.mql5.com/2/140/Creating_a_Trading_Administrator_Panel_in_MQL5_8Part_XIl_Modern_feature_communications_interface_lI1.png)[Creating a Trading Administrator Panel in MQL5 (Part XI): Modern feature communications interface (I)](https://www.mql5.com/en/articles/17869)

Today, we are focusing on the enhancement of the Communications Panel messaging interface to align with the standards of modern, high-performing communication applications. This improvement will be achieved by updating the CommunicationsDialog class. Join us in this article and discussion as we explore key insights and outline the next steps in advancing interface programming using MQL5.

![Overcoming The Limitation of Machine Learning (Part 1): Lack of Interoperable Metrics](https://c.mql5.com/2/140/Overcoming_The_Limitation_of_Machine_Learning_Part_1_Lack_of_Interoperable_Metrics__LOGO.png)[Overcoming The Limitation of Machine Learning (Part 1): Lack of Interoperable Metrics](https://www.mql5.com/en/articles/17906)

There is a powerful and pervasive force quietly corrupting the collective efforts of our community to build reliable trading strategies that employ AI in any shape or form. This article establishes that part of the problems we face, are rooted in blind adherence to "best practices". By furnishing the reader with simple real-world market-based evidence, we will reason to the reader why we must refrain from such conduct, and rather adopt domain-bound best practices if our community should stand any chance of recovering the latent potential of AI.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pnoqwgxyronzoxjlxlfqfdxzenrazvhi&ssn=1769093795754925169&ssn_dr=0&ssn_sr=0&fv_date=1769093795&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15964&back_ref=https%3A%2F%2Fwww.google.com%2F&title=High%20frequency%20arbitrage%20trading%20system%20in%20Python%20using%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909379543916916&fz_uniq=5049501588927655075&sv=2552)

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