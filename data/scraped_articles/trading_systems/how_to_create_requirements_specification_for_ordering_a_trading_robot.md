---
title: How to create Requirements Specification for ordering a trading robot
url: https://www.mql5.com/en/articles/4368
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T19:43:55.400385
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/4368&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070515914719303496)

MetaTrader 5 / Examples


### Table of Contents

- [Prerequisites for ordering a trading robot](https://www.mql5.com/en/articles/4368#need)
- [Why is it important to have a well-prepared Requirements Specification?](https://www.mql5.com/en/articles/4368#important)
- [Requirements Specification examples](https://www.mql5.com/en/articles/4368#examples)
- [What is contained in the Requirements Specification?](https://www.mql5.com/en/articles/4368#whatweneed)
- [Where do I get Requirements Specification if I can't create it?](https://www.mql5.com/en/articles/4368#ifyoucant)
- [Terms to use](https://www.mql5.com/en/articles/4368#words)
- [How to write an order description in the Freelance](https://www.mql5.com/en/articles/4368#description)
- [The general idea of ​​a trading strategy](https://www.mql5.com/en/articles/4368#idea)
- [Description of a setup preceding the signal](https://www.mql5.com/en/articles/4368#setup)
- [Signal description](https://www.mql5.com/en/articles/4368#signal)
- [Signal lifetime](https://www.mql5.com/en/articles/4368#time)
- [Placing of orders and opening of positions](https://www.mql5.com/en/articles/4368#ordersend)
- [Management of trading positions/orders](https://www.mql5.com/en/articles/4368#maitenance)
- [Cancellation of orders and closing of positions](https://www.mql5.com/en/articles/4368#close)
- [Order lot calculation](https://www.mql5.com/en/articles/4368#lot)
- [Processing trading errors and environment state](https://www.mql5.com/en/articles/4368#fales)
- [Difference between bar-opening and in-bar trading](https://www.mql5.com/en/articles/4368#trade)
- [Tick/scalping strategies](https://www.mql5.com/en/articles/4368#scalping)
- [Grid, martingale, averaging and the dark side of these techniques](https://www.mql5.com/en/articles/4368#martingale)
- [Important aspects of choosing a developer](https://www.mql5.com/en/articles/4368#casting)
- [What the programmer cannot do for you](https://www.mql5.com/en/articles/4368#onlyyou)


### Prerequisites for ordering a trading robot

Trading robots are programs, which operate according to underlying algorithms. An algorithm is a set of actions that need to be performed in response to certain events. For example, the most common task in algo trading is the identification of the ["New bar" event](https://www.mql5.com/en/articles/159). When the event occurs, the robot checks the emergence of trading signals and acts accordingly.

Before you decide to program or order a trading robot, you need to formulate a trading system with clear rules, based on which favorable moments to perform trading operations will be defined. The development of any trading system, even the most complex one, begins with the basic things, such as the definition of trading signals for buying and selling. Further, you will be able to add various options for managing and closing deals.

You do not need to have many years' experience of working with the trading terminal, to develop a trading strategy. You may choose among hundreds of proven ideas available on the web. Even if you are not sure about your programming skills, this is not an obstacle. The [Freelance](https://www.mql5.com/en/job) service will help you find a suitable developer.

Below are recommended articles, which you may want to read before proceeding with the algo trading techniques:

- [How to Make a Trading Robot in No Time](https://www.mql5.com/en/articles/443)
- [A Few Tips for First-Time Customers](https://www.mql5.com/en/articles/361)
- [How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235)
- [How to prepare Requirements Specification when ordering an indicator](https://www.mql5.com/en/articles/4304)

### Why is it important to have a well-prepared Requirements Specification?

When ordering or developing a trading robot, you need to formulate requirements: tasks to be performed by the robot, conditions under which it will operate, response to incidents and emergency situations, required control methods, etc. Trading robots are programs, which should strictly follow the underlying logic. In order to program the algorithm of actions, you should prepare its detailed description.

Description of a trading strategy is provided in the form of Requirements Specification. The more details you provide, the less misunderstanding will occur between you (the Customer) and the programmer (the order Developer).

The important part of Requirements Specification for an Expert Advisor is presentation of clear formal trading rules. Even if you are not ordering an EA, but want to develop one yourself, you should start with the definition of these rules. Prepare the Requirements Specification and include the EA testing/optimization related points. Add hypotheses, which you will use to check the quality and stability of your trading strategy, describe criteria for selecting the optimal parameters and explain why you consider them important.

Include all EA development stages to the Requirements Specification — this will make the algorithm idea clear for the Developer, and will help you recall the details weeks, months, or even years later. Algo trading is not a hobby, but a thorough research path, all stages of which need to be properly documented. A trading system development diary will be very useful whenever you'll need to test a new idea.

### Requirements Specification examples

Here is an example of Requirements Specification for the development of the [MACD Sample](https://www.mql5.com/en/code/2154) Expert Advisor, which is available in the MetaTrader 5 standard package.

**1\. The idea of the trading system is as follows**: market entries are performed when [MACD's](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main and signal lines intersect in the current trend direction **.**

**2\. Trend** is determined based on the [Exponential Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma#ema "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma#ema") with the specified period (InpMATrendPeriod). If the current EMA value is greater than the previous one, the trend is seen as growing (ema\_current > ema\_previous). Alternatively, if current EMA is below the previous one, the trend is considered to be falling (ema\_current< ema\_previous).

**3\. Trading Signals**:

- Buy signal: the main MACD line crosses the signal line upwards (macd\_current>signal\_current && macd\_previous<signal\_previous).

- Sell signal: the main MACD line crosses the signal line downwards (macd\_current<signal\_current && macd\_previous>signal\_previous). The below figure shows Buy and Sell cases.




![](https://c.mql5.com/2/32/MACD_Sample__2.png)


**4\. Positions are closed** at opposite signals: Buy positions are closed at Sell signals, and Sell positions are closed at Buy signals.

**5\. Positions are opened** at the market price, when a new bar emerges. The Expert Advisor is to be tested using Open prices, so there is no need to add functions for disabling operations inside the bar.

**6\. Additional filters for opening a position**:

The absolute value of MACD's main line will be used to filter out weak signals: the signal is only confirmed if this value is greater than open\_level (in points). Signal confirmation conditions are as follows:

- Confirmation of a buy signal: Abs(macd\_current)>open\_level
- Confirmation of a sell signal: macd\_current>open\_level

**7\. Additional filters for closing a position**:

The absolute value of MACD's main line will also be used to confirm position closure: the signal is confirmed if this value is greater than close\_level (in points). Close signal confirmation conditions are as follows:

- Confirmation to close Buy positions — macd\_current>close\_level
- Confirmation to close Sell positions — Abs(macd\_current)>close\_level

**8\. Close by Take Profit** — during position opening, a Take Profit level is set at a fixed distance from the open price, specified in points. The value is set in the InpTakeProfit input parameter.

**9\. Position management**

[TrailngStop](https://www.metatrader5.com/en/terminal/help/trading/general_concept#trailing_stop "https://www.metatrader5.com/en/terminal/help/trading/general_concept#trailing_stop") is used to protect profit. Stop Loss is set if profit in points exceeds the value specified in the InpTrailingStop parameter. If the price continues to move in the profit direction, Stop Loss should be trailed at the given distance. Stop Loss cannot be moved towards the loss direction, i.e. the Stop Loss value cannot be increased. If none of protective orders (Take Profit or Stop Loss) triggers, the position should be closed by an opposite signal. No other position exit methods are available.

### What is contained in the Requirements Specification?

**Trading idea**

Describe the general underlying idea in the first part of the Requirements Specification. Example: "If the price approaches the resistance level twice and rolls back from it, the next time it is likely to break resistance." Here you can add a chart with the resistance/support lines, indicators and explanatory notes. Exact numbers or calculation algorithms are not required in the idea description. So, in this example we do not need to explain how to determine:

- resistance level,
- level breakout,
- the concept of "is likely to".

Some abstraction at the initial stage will help focus on the idea rather than on technical details. This way you can generate multiple modifications of your trading strategy by replacing or combining strategy blocks, indicators and filters. With the common general idea, you will use different input parameters for your trading robots.

Next, you need to describe all terms and concepts contained in the idea description. If trend is important for your strategy, clearly define what indicator should be used to determine the trend direction and strength. The numerical characteristics of these definitions form the basis of Expert Advisor's [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables), and can be optimized in the Strategy Tester. So, the first section of the Requirements Specification is "The Trading Idea".

**Terms and definitions**

It is recommended to create a separate section in the Requirements Specification for explaining related terms and definitions. Explain terms in separate paragraphs. Use **bold font** to highlight the key concepts of your trading strategy. Where applicable, you may add an image. Input parameters of the desired EA can be written in italics.

**Trading signals**

This is the most important section of Requirements Specification. It provides the description of conditions, market states and indicator values, under which a Buy deal should be performed. To describe each condition required for generating a Buy signal, choose the numeric parameter affecting the emergence of a signal. For example, this may be the smoothing type and period for a [Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma"). These important parameters will be used as your Expert Advisor's _input parameters_.

Provide a separate description of sell conditions, even if they are opposite to buying. This may have specific features, which the programmer may misinterpret. For example, your Buy condition may be set to "Value > 0". Make sure to indicate an exact condition for selling, such as "Value < 0" or "Value <= 0".

Additional conditions and filters are often used for confirming or canceling trading signals. Use screenshots for various market situations with the **visual** explanation of indicators and setups. In future, such visualization may help you to analyze situations, when your EA ignores a seemingly obvious signal or performs a deal at an unfavorable moment.

**Screenshots and flow charts**

You may use any of free programs for creating screenshots and flow charts. Tips on what programs to choose and how to use them are provided in the article [How to create Requirements Specification for ordering an indicator](https://www.mql5.com/en/articles/4304#tools). Also, the article provides recommendations on ordering an arrow indicator generating buy and sell signals. Such an indicator running separately from the Expert Advisor, makes it easier to check and monitor your trading robot both during real-time operation and in visual testing.

**The lifetime of signals/orders/positions**

The second important part of a trading strategy is exiting an open position and deleting pending orders. Trading signals can also be canceled after some time or under certain events. Therefore, you need to describe conditions to close a buy/sell position, remove a pending order or cancel a signal.

**Management of open positions and pending orders**

If your trading strategy implies closing by Stop Loss and Take Profit, describe the calculation algorithm. Optionally, you can request a trailing mechanism for a flexible modification of these levels. In this case, you need to describe Trailing Stop conditions and algorithms. SL/TP modification can be performed at a bar opening or on every tick. Specify the desired option in the Requirements Specification. Also, the on-tick and on-bar options influence strategy testing. Make sure to read the article [Testing trading strategies on real ticks](https://www.mql5.com/en/articles/2612).

### Where do I get Requirements Specification if I can't create it?

A poorly drafted Requirements Specification or its absence often indicates that the trading system rules have not been formulated. So, what the Customer calls a trading system is actually only an idea. All nuances and absence of required logic descriptions will be revealed during the development process. The developer will have to think out possible options, which were not provided by the Customer.

The Developer may program a trading robot at his own risk in this case. But you are likely to lose much time discussing every possible issue. If the robot's behavior then differs from the Customer's expectations due to a lack of a proper description, such an order may be sent for an Arbitration. Customers often accuse the Developer for the incorrect programming of the robot. However, the Arbitration decision will be based on the Requirements Specification. According to [the Freelance Rules](https://www.mql5.com/en/job/rules#part_II), any other correspondence will not be taken into consideration during disputes:

During arbitration, the basis for decision making is only the Requirements Specification.

Sometimes, a customer may have clear trading rules, but may not be able to create the Requirements Specification for some reasons. Problems may arise with the correct description, math formulas, neural network or machine programming related issuers, and other aspects. In this case, the creation of the Requirements Specification can be ordered. This can be done in the "Programming Advice" or "Other" sections of the Freelance service.

Choose one of these categories, create an order named "Creating Requirements Specification for a trading robot order" and specify the initial cost of the work. An experienced developer of trading systems will help you to describe your strategy Rules in a clear and easy-to-understand form. Use screenshots to show setups of your trading signals based on charts, indicators and graphical objects you use.

The programmer will try to understand your trading system and help you prepare a description of the trading algorithm. If you cannot formulate any concepts (for example, "momentum" or "rebound from the level"), the programmer can suggest ready ideas based on his experience. As a rule, any market situation can be described logically (and then programmatically) by some simple model with variation parameters. Such a variation can be expressed by a certain parameter, which you will later optimize in your Expert Advisor.

Perfect patterns do not exist, because the market never repeats. However, similar situations can be found in history. Your cooperation should result in a ready Requirements Specification, which you may use to order a trading robot.

### Terms to use

Often, trading systems contain a number of key concepts or terms describing the market state or price behavior. Even if you think that you use generally accepted and simple concepts in the Requirements Specification, better provide its clear description. Add one description paragraph per each term.

For example, according to Bill Williams, an uptrend occurs when all three Alligator lines appear in the following order from bottom up: Blue, Red, Green.

![](https://c.mql5.com/2/32/2018-05-11_18h38_46.png)

Another classic definition of an uptrend was suggested by [Larry Williams](https://www.mql5.com/en/code/11401): each new peak is higher than the previous peak and each new trough is not lower than the previous trough.

![](https://c.mql5.com/2/32/2018-05-11_18h43_50.png)

You may additionally use chart screenshots in term descriptions. Use bold fonts for the **terms** in the Requirements Specification to help programmer find them in the text whenever necessary.

Do not use place links to other resources (such as websites, publications, forum topics, etc.) instead of explanations. A complete detailed explanation of all points should be available straight in the Requirements Specification. Take your time to describe all terms used - this will save the robot development time.

### How to write an order description in the Freelance

When creating an order, describe the general essence of your trading idea, so that potential developers understand what you need. Do not reveal your trading system rules or details of indicators in the order description.

The description might look like this:

> Develop an Expert Advisor trading trend reversals. Reversal signals will be generated based on Price Action patterns. Trend will be determined based on ADX, Alligator and MACD, while the indicator selection should be available in the EA's input parameters.

### The general idea of ​​a trading strategy

You may mention symbols to be traded, trend identification specifics and other information. For trend following EAs, specify entry methods - during a rollback, at a breakout or other methods.

Generally, there are two large types of financial trading strategies: expecting movement continuation or return to the average value. Your trading idea should relate to one of these two types. Explain how deals should be opened: by market, after a confirmation of a breakout/rollback or at a more appropriate price.

### Description of a setup preceding the signal

Simple signals can be easily described using algorithms. For example, popular simple patterns include "Engulfing" and "Pin bar". However, it is practically impossible to create a profitable strategy based on such simple formations. Such patterns are used for determining trend reversal points. The setup for waiting for the Bearish Engulfing pattern is the presence of an uptrend.

So, in addition to describing the trading signal, you should explain an appropriate setup in the Requirements Specification.

### Signal description

A buy or sell signal emerges when a certain condition is met. For example, a classic Buy signal appears when the price crosses the Moving Average upwards. You should indicate the following parameters in this signal description:

- Moving Average type - SMA, EMA, [VIDYA](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/vida "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/vida"), etc.
- Moving Average period
- Additional parameters which may be required for some Moving Averages, such as for [AMA](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ama "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ama").

Also, the phrase "the price crosses the Moving Average" needs explanation, because it is not as simple as it may seem. A signal can emerge at the very moment of intersection. Alternatively, you may choose to wait for the candlestick to break the MA and close above it. This affects your Expert Advisor code, as well as the [tick generation](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation "https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation") mode to be used during testing.

Therefore, you need to clearly explain the concepts of trend, level, breakout, crossover and similar ones, i.e. choose between operations with ticks, bars and Close prices. Provide formal descriptions and numeric parameters which you will optimize in the Strategy Tester. For example, trend strength can be measured using the [ADX](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/admiw "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/admiw") indicator, while [Ichimoku Kinko Hyo](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ikh "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ikh") is not suitable for this purpose.

The more conditions and filters your trading system needs, the more complex the robot turns out. Also, complex strategies normally have a large number of input parameters, which may require a lot of passes during optimization. Although the MetaTrader 5 Strategy Tester allows speeding up the optimization time through the use of the [genetic algorithm](https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types "https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types") and [MQL5 Cloud Network](https://www.mql5.com/en/articles/341), the volume of resulting data will also be huge.

Therefore, we recommend following some EA developing and debugging rules.

- To enable signal visual testing and debugging, the EA should display emerging signals as labels/objects on the chart. In addition to [debugging on history data](https://www.mql5.com/en/articles/2661), you will be able to view the formation of signals on the chart. Sometimes, it can be hard to understand complex algorithms. The visual display of Signals provides a convenient way to monitor opening of trades.
- Another convenient solution is to start with the creation of an indicator which displays Buy/Sell signals as [arrows](https://www.mql5.com/en/docs/customind/indicators_examples/draw_arrow) on the chart. This is a more convenient solution, which allows you to debug the two applications separately. Let the robot trade and the indicator plot. In this case, the Expert Advisor code will contain only the required functionality. In addition, there can be more Signals than the executed deals. For example, the EA receives a buy signal and enters the market. According to the algorithm, other buy signals are no longer checked. If you are using a separate indicator, it will show all buy signals regardless of the presence of an open position.

- In addition to providing separate descriptions for Buy and Sell signals in the Requirements Specification, it is recommended to debug them separately. Buy and sell signals are often interrelated, i.e. when there is a Buy position, all Sell signals are ignored (unless Sell signals are used for closing Buy positions). Separate testing of Buy and Sell signals allows you to check the correctness of the underlying logic in its pure form.


Also, you can optimize the parameters of the strategy separately for buying and selling, and then combine the algorithms in a single trading robot. This way, search for optimal parameters will be performed faster with fewer errors. However, in this case you will need to pay additionally for the creation of indicators and intermediate Expert Advisors. But a good trading idea is worth it, isn't it?

### Signal lifetime

In some trading systems, a position is not opened immediately after the emergence of a signal. Such system may require confirmation by additional signals. For example, after the breakout of the resistance level, you may want to wait till the price returns to the broken level, in order to enter the market under better conditions. You should define the time parameter here: how long the Level Breakout signal will be valid. Systems may wait for 5 bars or until the trading session end, after which the signal is canceled.

Add the Lifetime parameter to use additional filters, which may improve trading system quality.

### Placing of orders and opening of positions

You may provide additional functions in advance, when developing functions sending trade orders. For example, you may use different [MagicNumbers](https://www.mql5.com/en/articles/112) and comments for further [analysis](https://www.mql5.com/en/articles/3046) of trading and optimization results. You may use MagicNumber based on entry hour and day, a trading pattern number and other details, which enable additional analysis. So, you can implement multiple trading strategies in one Expert Advisor and optimize all of them, to find the best parameters for your trading robot. If you want to request such functionality, describe the MagicNumber calculation algorithm for each pattern/setup/signal.

A trading order is not always executed successfully. It is necessary to provide for situations when the position cannot be opened/closed during the first attempt. How should the EA handle such a situation: should it make a pause or wait for a new tick? How many attempts are allowed? What information should be written to logs? In what format should information be written? Should a notification be sent to a trader? How often should messages be sent to avoid DDoS attack situations?

Use comments to trading orders for quick analysis of trading history. Sometimes, trade servers write specific comments to this field. Therefore, your robot may additionally write its own daily log of trading operations.

If your trading strategy uses protective Stop Loss and Take Profit levels, describe an algorithm for their calculation and placing methods. For example, Stop Loss can be set only when the price moves by a specified number of points in the profit direction. If SL and TP are to be set after a successful position opening, describe the procedure to check the position opening - immediately after sending a trading order or at the next tick.

### Management of trading positions/orders

The basic rule of trader: let your profits run and cut your losses. In algorithmic terms, this means that you should set a protective Stop Loss for each position without limiting potential profits by Take Profit orders.

The stop order size can greatly affect trading results. Often, traders try to find optimal SL/TP distances to maximize profits. Try to find distance calculation algorithms which take into account market volatility, trend direction and support/resistance levels.

You may study existing trading systems to find a suitable SL/TP idea. Many programmers have ready-made libraries, which can be used when creating a trading robot based on your idea.

Consider and describe the following points in the Requirements Specification:

- use of Stop Loss and Take Profit levels, distance calculation algorithm;

- use of Trailing Stop, conditions to trigger, step calculation algorithms;

- if pending orders are used for entries, mention whether they should be trailed and describe the appropriate algorithm;

- the necessity to monitor the floating profit/loss of an open position, to close a position upon reaching the specified profit/loss level;
- etc.


### Cancellation of orders and closing of positions

Another position and order management method is based on time and opposite signals. You may describe additional closure and deletion options, such as:

- based on the floating profit or loss value;
- when the price moves at the specified distance from the current pending order opening level (which may mean that the opportunity was missed);

- at the specified time;
- after the specified number of bars;
- after the specified time interval;
- in case of an opposite signal;
- if the favorable setup/pattern disappears.

### Order lot calculation

Some traders include trading lot calculation algorithms at the first stage of robot creation. However, it is not recommended to include the money management algorithms for calculating the lot at this stage, since additional input parameters can lead to history-based overfitting during EA optimization.

Better test your first EA version using a fixed lot. Only forward testing using history data and several-month real trading will enable you to reveal the weaknesses and strengths of your algorithm, after which you can add money management methods.

Here are some approaches to calculating position lot size:

- fixed volume regardless of profit or loss;

- volume depending on the size of the balance or equity;
- based on profit/loss obtained;
- based on the last N trades (different martingale and anti-martingale techniques);
- depending on risk % with the specified Stop Loss;

- other risk-based calculations, such as [Vince method](https://www.mql5.com/en/articles/3650).

In any case, before adding lot calculation algorithms to the Expert Advisor, make sure that your trading system has an advantage over random trading. Otherwise you will only deceive yourself. A losing system cannot be turned into a profitable one only through money management methods.

### Processing trading errors and environment state

A trading robot is an autonomous program, which operates 24 hours a day. Therefore, provide mechanism to control its operation. Your Expert Advisor actions can be written to the Experts journal using the [Print()](https://www.mql5.com/en/docs/common/print) function. In general, it is recommended to record the emergence of signals, patterns and setups, the current market price and [trade request](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) parameters before [sending an order](https://www.mql5.com/en/docs/trading/ordersend) for execution.

If trade request execution fails, [its results](https://www.mql5.com/en/docs/constants/structures/mqltraderesult) should also be written to the log. Analyze trade server [return codes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes) to understand the reason for the failure and to fix it. Describe the following in the Requirements Specification:

- the situations, in which the EA should write messages to the journal;
- what parameters should be included in the message;
- required entry format, such as specification of time, numbers, separators, etc.


Detailed logs of trade orders and execution results will allow you to quickly identify trade errors and save your money.

An important point often forgotten by beginning algo traders is the restart of the terminal and loss of Internet or server connection. In such cases you may request the possibility of notification via [messaging functions](https://www.mql5.com/en/articles/2355) or e-mail.

### Difference between bar-opening and in-bar trading

With each price change, the robot starts processing the [NewTick](https://www.mql5.com/en/docs/runtime/event_fire#newtick) event by the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) function. A lot of ticks can be received during a bar lifetime, so the EA will execute its logic at each incoming bar. If your strategy produces signals only at the opening of the bar, then you need to decide the following:

1. how to calculate trading signals, get indicator values ​​and the trading environment state only at the first tick, skipping the next ones;
2. what to do if the necessary actions could not be performed at the first tick.


Let's analyze a simple example: a signal at the intersection of moving averages. If your EA checks a signal at each tick, then there may be a situation where the signal appears and then disappears. As a result, the EA will open and close the position several times during one bar. This may cause problems during online trading.

To avoid such problems, test the Expert Advisor in the "Every Tick" or " [Every Tick Based on Real Ticks](https://www.mql5.com/en/articles/2612)" mode. If you see a lot of similar operations within one bar, then revise your robot's code. Make sure to perform the [visual testing](https://www.metatrader5.com/en/terminal/help/algotrading/visualization "https://www.metatrader5.com/en/terminal/help/algotrading/visualization") of the Expert Advisor and used indicators, to check their operation on different history intervals directly on the chart.

### Tick/scalping strategies

If you are a beginner, choose systems operating at new bar opening. Such strategies are easier to develop and debug, while you will only need to provide a proper handling of the [New Bar](https://www.mql5.com/en/articles/159) event. You can check the correctness of the Expert Advisor trading at bar opening: testing results in the "Open Price Only" must match results in the "Every Tick"/"Every Tick Based on Real Ticks" mode.

Trading systems operating inside a bar are more difficult. We recommend reading the article [How to quickly develop and debug a trading strategy in MetaTrader 5](https://www.mql5.com/en/articles/2661), which contains the description of all steps required to create, debug and optimize the code of a strategy based on the analysis of the continuous tick flow.

When developing a scalping robot, note that such strategies are extremely sensitive to spread, commission, network delays, history quality and performance speed. Any worsening in trading conditions can "kill" such a strategy.

Do not confuse scalping strategies which try to quickly enter the market and catch small guaranteed profit, with so-called pipsing strategies. Pipsing strategy can be targeting a few pips and tolerate a drawdown of tens and hundreds of pips. Developers of such systems believe that the price is likely to pass several points in the open position direction than 50-100-300 points in an unfavorable direction. As a result of optimization, they can achieve impressive results on history, with 90-99% of trades winning. When you run this robot on a real account, the strategy may show expected profits for some time. But the market can make a sharp move at some point, and everything earned will be lost.

### Grid, martingale, averaging and the dark side of these techniques

Sometimes, algo traders try to improve results by increasing the number of orders/positions in one direction and manipulating lot value depending on price level/drawdown/loss (technical solutions), rather than improving the quality of signals (strategic solutions).

Additions in the form of order grids, martingale/anti-martingale elements and losing position averaging techniques complicate the code and increase the possibility of a program error. Also, such additional parameters increase the risk of overfitting. The use of such methods does not increase the stability or profitability of a trading system, but can only delay the collapse.

Instead of using such tricks, we recommend choosing another way:

- first, create a portfolio of different uncorrelated trading systems on one symbol;
- then, you can gradually create a set of portfolios on different instruments.

A portfolio of simple trading systems will be more resistant to market changes than one complex system with multiple optimizable input parameters.

### Important aspects of choosing a developer

So, you want to implement your trading system in the form of an Expert Advisor: you have created an order in Freelance and received applications from different developers. How to choose the optimal Developer in terms of cost and quality?

An experienced developer will not tell you about the complexity of previously developed systems or trading system variations, but will ask questions regarding **your** Requirements Specification. In other words, he will not try to impress you. Professionals appreciate time, so they usually do not waste it on philosophical discussions about the nature of trading or difficulties of programming.

The developer may request more details, in addition to the provided short description. If the order is generally clear, the programmer will provide information on the order cost and time.

A responsible Developer will point to unclear points in your Requirements Specification. If your order lacks detail, you can clarify them later with the programmer, as well as pay for consulting services, increasing the time and cost of the order.

A good programmer appreciates time, so he will try to clarify unclear points in order to start working with a clearly and thoroughly prepared Requirements Specification.

### What the programmer cannot do for you

It may happen that a trading robot based on your system will show losses during testing, although you checked it manually and were sure that the strategy would be profitable. As a rule, the reason is that the Customer was able to check the system manually using a short time interval. The Strategy Tester allows getting trading results for any available historic interval. What can be done in this case? A programmer cannot make a profitable strategy out of a losing one, but he can suggest some ideas for improving the quality of entries. For example, you may add some trend, volume or other filters to avoid false signals.

Also, your strategy may work better with other parameters which may suit better for given market condition. Optimize the EA to study dependence of optimal values on the year, volatility and other factors. This will help you identify weaknesses or limitations of your system. But you will have to do it yourself, because the Developer is only a code programmer, but not a trading system analyst.

The last part concerns programmer's errors. It is almost impossible to develop a program without errors. These may include code errors, when a wrong code is written for a correct algorithm, as well as logical errors. In both cases, you will have to find them yourself.

- You should understand [trade server return codes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes "Return codes of a trade server"), as well as [runtime errors](https://www.mql5.com/en/docs/constants/errorswarnings/errorcodes "Runtime Errors").

- Make sure to indicate in your Requirements Specification the need to process the results of each important operation and to log error codes. You may additionally enumerate such important operations: sending of trade requests, calculation of Stop Loss and other operations.

- Such messages should use [predefined macro substitutions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "Predefined macro substitutions"), which will help you to find the reason and exact location of the wrong behavior.


These three rules will help you analyze the situation and communicate with the Developer.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/4368](https://www.mql5.com/ru/articles/4368)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/274219)**
(28)


![Ramdhakal](https://c.mql5.com/avatar/2022/8/630133BF-3F3B.jpg)

**[Ramdhakal](https://www.mql5.com/en/users/ramdhakal)**
\|
21 Aug 2022 at 06:53

Thank you so much


![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
29 Aug 2024 at 11:37

Thank you, great information here.


![FIGOUE Figoue Hapi ](https://c.mql5.com/avatar/avatar_na2.png)

**[FIGOUE Figoue Hapi](https://www.mql5.com/en/users/figoue)**
\|
9 Sep 2024 at 07:49

Bonjour,

je voudrais créer un cahier de charges, faut - il le faire dans un fichier de type Microsoft Word?

quel est le type de fichier accepté pour le cahier de charges SVP ?

Parce que moi je ne parviens pas joindre mon fichier Microsoft Word avec le cahier de charges.

![Patrick Nalletamby](https://c.mql5.com/avatar/2020/8/5F2644C6-4B31.jpg)

**[Patrick Nalletamby](https://www.mql5.com/en/users/pnall3)**
\|
7 Nov 2024 at 15:17

**MetaQuotes:**

New article [How to create Requirements Specification for ordering a trading robot](https://www.mql5.com/en/articles/4368) has been published:

Author: [MetaQuotes Software Corp.](https://www.mql5.com/en/users/MetaQuotes "MetaQuotes")

Very helpful information. Thank you.

![よう子 深美](https://c.mql5.com/avatar/avatar_na2.png)

**[よう子 深美](https://www.mql5.com/en/users/y.akakyabe)**
\|
12 Apr 2025 at 10:46

I have been scammed out of 33.5 million dollars, I used to have some information here, but now I have nothing. Please contact me, my mobile is 09071854610, it's all I have.


![Deep Neural Networks (Part VII). Ensemble of neural networks: stacking](https://c.mql5.com/2/48/Deep_Neural_Networks_07.png)[Deep Neural Networks (Part VII). Ensemble of neural networks: stacking](https://www.mql5.com/en/articles/4228)

We continue to build ensembles. This time, the bagging ensemble created earlier will be supplemented with a trainable combiner — a deep neural network. One neural network combines the 7 best ensemble outputs after pruning. The second one takes all 500 outputs of the ensemble as input, prunes and combines them. The neural networks will be built using the keras/TensorFlow package for Python. The features of the package will be briefly considered. Testing will be performed and the classification quality of bagging and stacking ensembles will be compared.

![Implementing indicator calculations into an Expert Advisor code](https://c.mql5.com/2/32/expert_indicator.png)[Implementing indicator calculations into an Expert Advisor code](https://www.mql5.com/en/articles/4602)

The reasons for moving an indicator code to an Expert Advisor may vary. How to assess the pros and cons of this approach? The article describes implementing an indicator code into an EA. Several experiments are conducted to assess the speed of the EA's operation.

![Testing currency pair patterns: Practical application and real trading perspectives. Part IV](https://c.mql5.com/2/31/LOGO.png)[Testing currency pair patterns: Practical application and real trading perspectives. Part IV](https://www.mql5.com/en/articles/4543)

This article concludes the series devoted to trading currency pair baskets. Here we test the remaining pattern and discuss applying the entire method in real trading. Market entries and exits, searching for patterns and analyzing them, complex use of combined indicators are considered.

![Comparative analysis of 10 flat trading strategies](https://c.mql5.com/2/32/10_flat.png)[Comparative analysis of 10 flat trading strategies](https://www.mql5.com/en/articles/4534)

The article explores the advantages and disadvantages of trading in flat periods. The ten strategies created and tested within this article are based on the tracking of price movements inside a channel. Each strategy is provided with a filtering mechanism, which is aimed at avoiding false market entry signals.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/4368&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070515914719303496)

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