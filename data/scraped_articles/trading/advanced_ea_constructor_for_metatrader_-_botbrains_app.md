---
title: Advanced EA constructor for MetaTrader - botbrains.app
url: https://www.mql5.com/en/articles/9998
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:15:20.087438
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=zmioieeffjlxqeswpwxolsdminlvmmvp&ssn=1769181317287801537&ssn_dr=0&ssn_sr=0&fv_date=1769181317&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9998&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Advanced%20EA%20constructor%20for%20MetaTrader%20-%20botbrains.app%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918131793310367&fz_uniq=5069282348572279429&sv=2552)

MetaTrader 5 / Trading


### Introduction

Generally, trading strategies can be reduced to a specific algorithm that can be automated. Trading robots can trade thousands of times quicker than any human, but not every trader is proficient in programming.

[botbrains.app](https://www.mql5.com/go?link=https://botbrains.app/ "https://botbrains.app/") is a no-code platform for trading robots development. You don't need to program anything to create a trading robot in the BotBrains editor - just drag the necessary blocks onto the scheme, set their parameters, and establish connections between them.

The [botbrains.app](https://www.mql5.com/go?link=https://botbrains.app/ "https://botbrains.app/") has documentation ( [docs.botbrains.app](https://www.mql5.com/go?link=https://docs.botbrains.app/ "https://docs.botbrains.app/")) which describes all the concepts of the editor in detail. This article describes the editor's interface, its key features, as well as the implementation of the "Moving Average Crossing" trading strategy.

![](https://c.mql5.com/2/45/botbrains_mac_mockup.jpg)

### Editor features

In BotBrains you can not only develop basic trading robots (e.g. trading robots based on crossing moving averages), but also implement complex trading algorithms (e.g. trading robots based on spread trading between symbols). BotBrains allows you to link a trading robot to your telegram bot - using special blocks you can send messages on trading results and even send screenshots of charts of a custom size. You can easily create and assign actions to buttons, add input fields to control variable values when the robot is running, and add various other blocks to build your robot interface.

The following table lists the categories of blocks available in the botbrains editor. A complete list of available blocks is presented at the end of the article.

| Category | Description |
| --- | --- |
| **Events** | These blocks are activated when something happens. For example "Start of robot work" event block is activated once the robot is launched. |
| **Condition** | You can perform various checks using the "if" block. |
| **Loop** | Loops have many applications. For example, you can use a loop to go through all available symbols and identify the symbol that made the largest movement during the last trading session. |
| **Indicators** | Indicator is one of the key means of technical analysis. BotBrains provides a wide range of indicators: from simple volumes to Ichimoku clouds |
| **Chart analysis** | Get information on a specific chart bar, get the maximum/minimum price for a certain period of time. Draw horizontal and vertical lines on the chart. |
| **DOM analysis** | Get information on the depth of market spread and its quotes. |
| **Transactions** | Place and remove limit/stop orders. Place market orders, close position. |
| **Variables** | Change variables values. For example, in a variable you can store the number of lots your robot will trade. |
| **Sounds** | Using sound blocks you can play a certain sound when something happens. For example, you can play the "Buy signal" sound when all the conditions for opening a long position are met. |
| **Getting information** | Get information on trading account, trading session, active limit/stop orders, history limit/stop orders, symbol, time and history deals. |
| **Enumerations** | Enumerate symbols, active limit/stop orders, history limit/stop orders, history deal. |
| **Telegram** | Send messages and charts screenshots to your telegram. |
| **Interface** | Build a full-fledged interface of a trading robot. Individual interface elements can be modified using modification blocks. You can bind actions to interface buttons. The "input field" interface block can be used to change variable values while the robot is running. All of these blocks allow creating dynamic user interfaces. |
| **Predefined constants** | Basically, predefined constants are used for comparisons. For example, there are 3 predefined constants for position direction: buy, sell and no direction. You can get the direction of the current open position and compare it with one of the predefined constants. |
| **Debug** | Using the "debug" block you can output debug information to the terminal journal. For example, using this block you can check that variables or constants contain correct values. |
| **Other blocks** | Terminal close, stop robot work, robot work pause. Log to file, notification in the terminal, comment on the chart. |
| **Math operators** | Addition, subtraction, multiplication, division, division remainder, square root, exponentiation. Comparisons: less than, greater than, less than or equal to, greater than or equal to. |
| **Logic operators** | There are logical operators for building conditions: AND, OR, NOT. |
| **Teleports** | There are blocks for entering and exiting from the teleport. Sometimes you need to quickly and conveniently switch to the execution of another section of the scheme - with teleports this can be done in a matter of seconds. |
| **Type conversion blocks** | By default, all user data is stored as a number. With type conversion blocks you can explicitly specify in which format you want the data to be represented. There are 4 types: integer, decimal, string, date and time. |
| **Variable or constant select** | Variable or constant select |
| **Value input** | Value input |

### BotBrains editor interface

BotBrains editor has 3 modes: **Logic**, **Interface**, **Code**.

**_"Logic" mode:_**

In the " logic" mode, you can build the robot's logic scheme. In this mode you can use the majority of blocks: from the "if" block and transaction blocks to blocks for sending messages to telegram and blocks for making robot's logs. In this mode you are developing the logic of your trading robot. Using simple blocks you specify what your robot should do - when to buy, when to sell, when to send messages to telegram, when to stop trading, and so on. There are more than 140 blocks in total in the botbrains editor, using which you can implement almost any trading strategy.

![Logic mode](https://c.mql5.com/2/45/logic_mode__1.jpg)

**_"Interface" mode:_**

In "Interface" mode, as the name suggests, you can build a trading robot's interface. In just a couple of minutes you can build a full-fledged interface for your trading robot. Each interface element has a complete set of settings. Using the "Modify interface element" block, you can change interface element properties on the fly.

![interface mode](https://c.mql5.com/2/45/interface_mode__1.jpg)

**_"Code" mode:_**

In this mode you can see the generated code of your trading robot. To generate the code of a trading robot, just press the ~ key, or click on the corresponding button in the right toolbar.

![Code mode](https://c.mql5.com/2/45/code_mode__1.jpg)

### Launching a trading robot

All code of your trading robot is generated in seconds after you press one key on your keyboard. To run a robot made with BotBrains, you need to make sure that all the necessary dependencies are installed:

1. Library botbrains\_lib.ex5
2. Included file botbrains\_lib.mqh
3. The botbrains\_constants.mqh include file.
4. Sound files

If the above dependencies are not installed, robots made in BotBrains cannot be started or will not work properly.

[You can download all the necessary dependencies](https://www.mql5.com/go?link=https://botbrains.app/ "https://botbrains.app/") [here](https://www.mql5.com/go?link=https://botbrains.app/editor/mql5/1.0.0/dependencies.zip "https://botbrains.app/editor/mql5/1.0.0/dependecies.zip"). The installation of dependencies is discussed in detail in the corresponding documentation article.

### Moving average cross example

Let's look at the implementation of the "Moving Average Intersection" trading strategy in the botbrains editor. You can open the complete scheme directly in the editor by clicking on the [link](https://www.mql5.com/go?link=https://botbrains.app/editor/mql5/1.0.0/editor.php?example=moving_average_cross_en "https://botbrains.app/editor/mql5/1.0.0/editor.php?example=moving_average_cross_en").

Open the editor and create 5 constants:

1. **slow\_ma\_period** \- period of slow moving average (constant value: 60)
2. **fast\_ma\_period** \- period of fast moving average (constant value: 30)
3. **symbol** \- code of symbol to trade (constant value: MGCV21 or any other symbol available in your trading terminal)
4. **admin\_id** \- id of the telegram user, to which our robot will send messages with trade information and chart screenshots (constant value: your Telegram ID)
5. **lot** \- number of traded lots

![constants](https://c.mql5.com/2/45/constants__1.jpg)

In order to detect the fact that two moving averages have crossed, we need to create 4 variables:

1. **ma\_slow** \- current value of slow moving average
2. **ma\_fast** \- current value of fast moving average
3. **ma\_slow\_prev** \- value of slow moving average of the previous bar
4. **ma\_fast\_prev** \- value of fast moving average of the previous bar

![](https://c.mql5.com/2/45/variables__4.jpg)

This way, we can detect the crossover of the slow and fast moving averages simply by comparing the values of these variables. Please note that we do not set initial values to the variables - the values of these variables will be updated as soon as a new tick for the symbol being traded is received.

Let's print the "Robot is launched" message when the robot is launched. To implement this, just drag and drop two blocks onto the robot logic scheme: the "Robot start" event block and the "Journal message" block. And then link the connectors of these blocks:

![robot launch message](https://c.mql5.com/2/45/robot_launch_message__1.jpg)

In the settings of the "Journal message" block specify the message to be displayed:

![journal message block settings](https://c.mql5.com/2/45/journal_message_block_settings__1.jpg)

When a new tick for a symbol being traded is received, our robot should update variable values. In order to do that, let's drag the "New Tick" event block and 4 "Set variable complex value" blocks onto the robot's logic scheme. Drop the "Moving average" block in the body of each "Set variable complex value" block and select the corresponding variable in each of these block. Then establish connections between these blocks:

![variables update on new tick](https://c.mql5.com/2/45/variables_update_on_new_tick__1.jpg)

Using the "Moving average" block we can get the value of the "Moving average" indicator. This block has 6 parameters:

1. **Symbol**
2. **Timeframe**
3. **Period**
4. **Smoothing method**
5. **Applied price**
6. **Shift**

Using the "shift" parameter we can get the value of the "moving average" indicator on a specific bar. For example, shift value of 0 means the value on the last available bar (no shift), but the value of 1 would mean the value on the previous bar. This way we can get the values of the slow and the fast moving averages on the current and on the previous bar - just set the different values of the "shfit" and the "period" parameters.

Let's specify the parameters for the first "MA" block:

![](https://c.mql5.com/2/45/first_ma_block_settings__2.jpg)

Note that the corresponding constants are used as the values of the "symbol" and "period" parameters. You can also use variables as parameter values.

For more information on working with variables and constants, refer to the relevant[documentation article](https://www.mql5.com/go?link=https://docs.botbrains.app/variables-and-constants "https://docs.botbrains.app/variables-and-constants").

This way, the value of the "ma\_slow" variable will be updated each time a new tick for the symbol being traded is recieved.

Let's do the same for the "ma\_fast" variable:

![](https://c.mql5.com/2/45/second_ma_block_settings__1.jpg)

At this stage we update the current values of moving averages, but in order to determine the fact of crossing we need to know the value of moving averages on the previous bar. The current bar has the shift of 0, the previous bar has the shift of 1, and so on. Therefore, in the last 2 blocks of "Set variable complex value" we will do everything similarly, but this time we should set the "shift" parameter of "Moving Average" blocks to 1, so that "ma\_slow\_prev" and "ma\_fast\_prev" variables will contain moving average values from the previous bar.

![](https://c.mql5.com/2/45/third_ma_block_settings__1.jpg)

Do the same for the " ma\_fast\_prev" variable.

Let's make sure that our variables are set to the correct values. Drop the "Timer" block onto the scheme. This block has only 1 parameter - "Interval (sec)". This parameter is set to 1 by default, so this block is executed every second by default. Then drop 4 "Print debug info" blocks onto the scheme - this block just prints the specified message to the terminal journal. Using this block we can see which values our variables contain. In each of the "Print debug info" place 3 blocks in the following order:

1. **Value input** \- this block is a simple input field into which we can enter any text. In our case, we should enter variable names, so that we can understand which variables refer to which values.
2. **"+" operator**\- using this block we can merge 2 strings.
3. **Variable select**\- using this block we can get the value of the specified variable.

At this stage our scheme shoud look like this:

![](https://c.mql5.com/2/45/timer_and_print_debug_info_blocks__1.jpg)

In BotBrains all user values are stored as numbers. In this case we need to explicitly specify that the "value input" blocks contain text, so that the text itself is output and not its numeric representation.

To do this, just place the "Regular string" type conversion block on each of the "value input" blocks:

![](https://c.mql5.com/2/45/timer_and_print_debug_info_blocks_2__1.jpg)

At this point we will be able to compile the robot's code and even run it - the values of the specified variables will be printed to the terminal journal each second. However, a warning **"implicit conversion from 'number' to 'string'"** will appear during compilation:

![](https://c.mql5.com/2/45/ma_cross_12.jpg)

Indeed, in our scheme we are trying to add up a string value ("value input" blocks converted to a string) with a numeric value (variables containing the numeric values of moving averages). To fix this, we simply need to cast all of the variables to strings. We can do this in two ways:

1. Using the "Regular string" type conversion block
2. Using the "Fraction" type conversion block - this block differs from the "Regular string"  block only by the ability to limit the number of digits after the decimal point

Let's use the second way. To do this, place the "Fraction" type conversion block on each of the "variable select" blocks:

![](https://c.mql5.com/2/45/timer_and_print_debug_info_blocks_3__2.jpg)

In the settings of each "Fraction" block, specify 2 as the only parameter - "Decimal places":

![](https://c.mql5.com/2/45/fraction_block_settings__2.jpg)

Working with types is described in detail in the relevant [documentation article](https://www.mql5.com/go?link=https://docs.botbrains.app/type-blocks "https://docs.botbrains.app/type-blocks").

Open the terminal and add 2 "Moving Average" indicators on the chart of the symbol being traded, with the same parameters as used in our robot, so that we can compare the values of the variables with the values of the indicators on the chart:

![](https://c.mql5.com/2/45/moving_averages_on_charts__1.jpg)

Then generate the robot's code using the "~" key or using the corresponding button in the right toolbox.

![](https://c.mql5.com/2/45/generated_code__2.jpg)

Run our robot in the terminal and compare the output values in the termial journal with the actual values of the indicators on the chart:

![](https://c.mql5.com/2/45/moving_averages_and_outputed_values__1.jpg)

All values match, so we can continue our work.

The only thing left to do is to compare variable values when a new tick is received in order to detect a moving average crossover and make a trade if all conditions for making a trade are met.

First, let's determine the conditions for making a trade. There are 3 conditions to be met for opeining a long position:

1. The current value of the fast moving average is greater than the current value of the slow moving average.
2. The value of the fast moving average on the previous bar is less than the value of the slow moving average on the previous bar.
3. The direction of the current position is not "buy"

Please note that if we omit the last condition, our robot may execute a lot of trades in a short amout of time, because the first and the second conditions may be met serval times in one second. As the result of the third condition our robot will not open a new long position on top of the existing long position.

Note that in the BotBrains editor you can specify the maximum and suspicious deals frequency. If the maximum deals frequency is reached, robot will immediately close all of the open positions and remove all of the placed orders, and the robot will also notify you via telegram and/or make a notification directly in the terminal, depending on the parameters set in the robot's security settings. If thesuspicious deals frequency is reached, robot will not close any positions or remove any orders; instead, it will just notify you via telegram and/or notify you directly in the terminal.

The following conditions must be met in order to enter the short:

1. The current value of the fast moving average is lower than the current value of the slow moving average
2. The value of the fast moving average on the previous bar is greater than the value of the slow moving average on the previous bar
3. The direction of the current position is not "sell"

Let's build a condition for opening a long position:

![](https://c.mql5.com/2/45/long_condition__2.jpg)

The "Position info" block has 2 parameters:

1. **Symbol** \- symbol about the position, on which you want to get information
2. **Block value** \- target position parameter. Available options: position volume, positionopen time, open price, positioncurrent profit and position direction

We will use the "symbol" constant as the value for the "symbol" parameter. Choose "Position direction" as the value for the second parameter.

![](https://c.mql5.com/2/45/position_info_block_settings__2.jpg)

This way the value of the "Position info" block will be equal to the direction of the current open position by the specified symbol.

After this block there are three blocks: "Not", "Equals" and "Position direction". The "Position direction" block contains all possible directions: "Buy direction", "Sell direction" and "No direction". We check that the direction of the current position is not "buy".

Then add one "Market order" block onto the scheme and link this block with the "Yes" connector of our "If" block:

![](https://c.mql5.com/2/45/market_order_block_linked_with_the_if_block__2.jpg)

The "Market order" block has three parameters:

1. **Symbol**\- the symbol for which a market order will be placed
2. **Direction** \- order direction (buy/sell)
3. **Volume**\- order volume

![](https://c.mql5.com/2/45/market_order_block_settings__4.jpg)

Note that we use constants as values for many parameters. This allows us to easily configure our robot - we can simply change the value of a constant in one place and the new value will automatically be used in all places where that constant is used.

If a trading robot made in the BotBrains editor has constants, they will be listed at the very top of the generated code:

```
/********** <ROBOT CONSTANTS> **********/

const double __slow_ma_period = user_value("60");
const double __fast_ma_period = user_value("30");
const double __symbol         = user_value("MGCG22");
const double __admin_id       = user_value("744875082");
const double __lot            = user_value("1");

/********** </ROBOT CONSTANTS> **********/
```

You can think of constants as robot settings. For example, if you decide to trade a different symbol, just change the value of the "symbol" constant. Or you can change the value of the constant directly in the BotBrains editor and re-generate the code if you don't want to work with the code directly.

So, at this point the trading robot will open a long when the fast moving average crosses the slow moving average from bottom to top. Let's make a notification when our robot opens a long position:

1. Display it in the terminal journal
2. Create an alert in the terminal
3. Add the record about the deal execution to the robot's log file
4. Play "Buy Signal" sound
5. Send message to telegram
6. Send chartscreenshot to telegram

Please note, if we play the "Buy Signal" sound immediately after making a trade, it is likely that our sound will not be played, because MetaTrader automatically plays its own sound every time a transaction is made. Therefore, we need to play the sound not immediately, but with a certain delay after the transaction is made, e.g. 1 second.

It may seem like it would take a long time to implement all 6 steps; in fact, there are special blocks for all of this. It will literally take 20-30 seconds to implement all 6 steps. Everything we have done before can also be done in a matter of minutes - many times faster than writing code by hand.

![](https://c.mql5.com/2/45/trade_notifications__2.jpg)

In order for telegram blocks to work properly, you need to register your bot in the @BotFather bot, get your bot's token and specify it in your bot's settings in the BotBrains editor. You also need to know your Telegram ID, which you can get from @getmyid\_bot. All this is described in detail in the relevant documentation article. Blocks for working with telegram are available to pro users only.

Similarly, let's implement logic for opening short positions. To do this, jsut select all blocks related to opening a long position, copy (CTRL + C), paste them (CTRL + V) and make the necessary changes.

![](https://c.mql5.com/2/45/selected_blocks_of_long_logic__2.jpg)

![](https://c.mql5.com/2/45/long_and_short_logic__2.jpg)

Generate the code of the robot, compile it and run it in the terminal to check the performance of our trading robot.

Test trading robots on a demo account only! By default, the security settings of all robots made in the BotBrains editor prohibit trading on a real account! Switch to trading on a real account only after your trading robot has been fully tested on a demo account!

Let's set up random moving average periods and run the trading robot on a one-minute timeframe. It is important at this point to test the performance of our trading robot and detect and correct any potential mistakes. After 80 minutes of trading, the trading robot has opened 4 positions. After each position was opened, the robot successfully made a notification in the terminal, played the appropriate sounds and made the appropriate records in the log file of the robot and in the terminal journal. The robot also sent messages and chart screenshots directly to telegram without any problems. And we haven't written a single line of code for all of that. Such a simple robot can be built in just a couple of minutes.

![](https://c.mql5.com/2/45/ma_cross_26__1.jpg)

It may seem like everything is fine. However, if you have a closer look at the trades made by our trading robot, you will find that in every position except the first one, our trading robot has entered in two trades: the first trade to exit the current open position and the second trade to open the next position. It is easy to detect this - just look at the chart, check the log file, see the messages sent by the trading robot to telegram or just listen to the number of times the trading robot generates an audio alert after the trade has been executed. We can also analyze the history of transactions in the terminal.

![](https://c.mql5.com/2/45/telegram_messages__2.jpg)

In the BotBrains editor, you can specify the suspicious and maximum deals frequency in the robot security settings. By default, the maximum deals frequency is 2 trades per 10 seconds. If the maximum frequency of trades is reached, the trading robot will immediately close all the positions it has opened, remove all placed orders, and notify you via live chat and/or create a notification in the terminal.

![](https://c.mql5.com/2/45/deals_maximal_frequency__2.jpg)

The problem is not only that our robot will end up with twice as many trades as necessary and enter most positions at a worse price, but also that at some point our robot will reach its maximum deals frequency and stop trading. This is exactly what happened in this case. Pay attention to the last records in the log file:

![](https://c.mql5.com/2/45/log_file__1.jpg)

Let's fix this issue. Create **full\_lot** variable:

![](https://c.mql5.com/2/45/variables__5.jpg)

When the robot is launched set the **full\_lot** variable to the value of the **lot** constant:

![](https://c.mql5.com/2/45/full_lot_variable_initialization__3.jpg)![](https://c.mql5.com/2/45/full_lot_variable_initialization__4.jpg)

Then in transaction blocks we should use the **full\_lot** variable instead of using the **lot** constant:

![](https://c.mql5.com/2/45/market_order_block_settings__5.jpg)

Then, after entering the position, set the value of the **full\_lot** variable to double the value of the **lot** constant. After the first trade the robot will start trading with double volume. Thus, it will close an existing position and open a new one with just one trade:

![](https://c.mql5.com/2/45/full_lot_variable_update__2.jpg)

The trading robot will now enter all positions with one trade. We will check this by running the robot in the terminal. If we look at the trade history we will see that the first trade was opened with 1 lot, while the following trades were opened with 2 lots:

![](https://c.mql5.com/2/45/ma_cross_39__1.jpg)

![](https://c.mql5.com/2/45/ma_cross_38__1.jpg)

Let's create an interface for the trading robot. To do this, go to the "Interface" mode of the editor and use the special blocks to build the interface:

[![](https://c.mql5.com/2/45/interface__5.jpg)](https://c.mql5.com/2/45/interface__4.jpg "https://c.mql5.com/2/44/interface.jpg")

To build such an interface, it is sufficient to use the following interface blocks:

1. **Rectangle**
2. **Button**
3. **Text**

The current values of fast and slow moving averages should be used instead of dashes. To implement this, use the "Modify interface element" block - this block can be used to modify the properties of the block with the specified ID.

Let's move 4 "Modify interface element" blocks to the robot's logic scheme:

![](https://c.mql5.com/2/45/modify_interface_element_blocks__2.jpg)

To copy the block ID do the following

1. Press CTRL
2. Without releasing CTRL, double-click on the target block

Copy the ID of the first dash:

![](https://c.mql5.com/2/45/copy_id_of_the_interface_element__2.jpg)

Open the settings of the first "Modify interface element" block and set parameters values. The first thing to do is to specify the ID of the block which properties we want to change. In our case it is the copied ID of the first dash. As type of modification we should choose "Text". Specify variable **ma\_fast\_prev** as the new text. If you want, you can also set the number of decimal places in new text (this setting should only be used when a number is used as the text content of an interface text).

![](https://c.mql5.com/2/45/modify_interface_element_settings__2.jpg)

Similarly, let's set parameters of the remaining 3 "Modify interface element" blocks. This way, when a new tick for a symbol being traded is received, the corresponding values of moving averages will be written into variables **ma\_fast**, **ma\_slow**, **ma\_fast\_prev** and **ma\_slow\_prev** and then the values of these variables will be written to the corresponding text elements of the interface.

Now let's assign the corresponding actions to the interface buttons. To do this, let's move three blocks to the robot's logic diagram: 2 "Market order" blocks and 1 "Close position" block:

[![](https://c.mql5.com/2/45/3_transaction_blocks__5.jpg)](https://c.mql5.com/2/45/3_transaction_blocks__4.jpg "https://c.mql5.com/2/44/3_transaction_blocks.jpg")

Then set settings of these blocks:

![](https://c.mql5.com/2/45/3_transaction_blocks_settings__2.jpg)

Copy the ID of the 1st "Market Order" block and specify the copied ID as the value of the "Linked block ID" parameter of the first button:

![](https://c.mql5.com/2/45/linked_block_id__2.jpg)

This way, when the "Buy" button is pressed, the corresponding block will be invoked. Similarly, assign actions to the other buttons. Let's generate the robot code and run it in the terminal to check its functionality:

![](https://c.mql5.com/2/45/robot_in_the_terminal__2.jpg)

Please note that the interface we built in the editor has been fully transferred to the trading terminal. The interface buttons work and instead of dashes we can see the actual values of moving averages.

As a result, the generated code of the trading robot is shown below:

```
//+------------------------------------------------------------------+
//|                                      moving_average_cross_en.mq5 |
//|                                                    botbrains.app |
//+------------------------------------------------------------------+

#include <botbrains_constants.mqh>
#include <botbrains_lib.mqh>

/********** <ROBOT CONSTANTS> **********/

const double __slow_ma_period = user_value("60");
const double __fast_ma_period = user_value("30");
const double __symbol         = user_value("MGCG22");
const double __admin_id       = user_value("744875082");
const double __lot            = user_value("1");

/********** </ROBOT CONSTANTS> **********/

/********** <ROBOT VARIABLES> **********/

double __ma_slow      = user_value("");
double __ma_fast      = user_value("");
double __ma_slow_prev = user_value("");
double __ma_fast_prev = user_value("");
double __full_lot     = user_value("");

/********** </ROBOT VARIABLES> **********/

int OnInit(){

  //Is autotrading allowed:
  if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)){
    MessageBox("Autotrading is not allowed, expert will be removed");
    ExpertRemove();
    return(-1);
  }

  //Is trading on live account allowed:
  if(AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_REAL){
    MessageBox("Expert is not allowd to trade on live account!");
    ExpertRemove();
    return(-1);
  }

  //Set robot name:
  set_robot_name("moving_average_cross_en");

  //Set license key:
  set_license_key("2L5J7K-K986ND-KMPT94-1Q");

  //Set language:
  set_lang("en");

  //Generate robot magic number:
  generate_magic();

  //Set initial trading account balance:
  set_init_account_balance();

  //Set suspicious deals frequency params:
  set_suspicous_deals_frequency(60, 3, false, true);

  //Set maximal deals frequency params:
  set_max_deals_frequency(10, 2, false, true);

  //Set the timer with an interval of 1 second:
  EventSetTimer(1);

  //Blocks executed when the robot is launched:
  block_bYi6ikfde();
  block_bYUS6GLT0();

  //Create interface elements:
  create_rectangle("b1ELCu5iq", 0, 0, CORNER_LEFT_UPPER, 15, 15, 390, 195, C'20,20,20', BORDER_FLAT, STYLE_SOLID, C'10,191,254', 2, 0, false, false, false);
  create_button("b4rh5uKlb", 0, 0, CORNER_LEFT_UPPER, 195, 165, 150, 30, "Sell", "Ubuntu Mono", 8, C'255,255,255', C'20,20,20', C'255,51,0', false, 0, false, false, false);
  create_text("b5BGSldua", 0, 0, CORNER_LEFT_UPPER, 120, 135, "-", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_text("b9EjNpibO", 0, 0, CORNER_LEFT_UPPER, 30, 30, "\"MA Cross\" trading robot", "Ubuntu Mono", 14, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_button("bBTrBQlkS", 0, 0, CORNER_LEFT_UPPER, 30, 165, 150, 30, "Buy", "Ubuntu Mono", 8, C'255,255,255', C'20,20,20', C'51,255,0', false, 0, false, false, false);
  create_text("bEncRhDIR", 0, 0, CORNER_LEFT_UPPER, 285, 75, "Current bar:", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_text("bI0vadsS2", 0, 0, CORNER_LEFT_UPPER, -195, 270, "Text", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_text("bK1cW1i6s", 0, 0, CORNER_LEFT_UPPER, 30, 135, "MA slow:", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_text("bLcIIkYqO", 0, 0, CORNER_LEFT_UPPER, 285, 135, "-", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_text("bSDWBsxbk", 0, 0, CORNER_LEFT_UPPER, 285, 105, "-", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_text("bTK8r1zb1", 0, 0, CORNER_LEFT_UPPER, 30, 105, "MA fast:", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_text("bhKcJ2pwx", 0, 0, CORNER_LEFT_UPPER, 120, 75, "Previous bar:", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_text("bj6MtjhZF", 0, 0, CORNER_LEFT_UPPER, 120, 105, "-", "Ubuntu Mono", 9, C'255,255,255', 0, ANCHOR_LEFT_UPPER, 0, false, false, false);
  create_button("bympSPhAp", 0, 0, CORNER_LEFT_UPPER, 360, 165, 30, 30, "X", "Ubuntu Mono", 8, C'255,255,255', C'20,20,20', C'255,51,0', false, 0, false, false, false);

  ChartRedraw();

  //Robot initialization was successful:
  return(INIT_SUCCEEDED);

}

void OnDeinit(const int reason){

  Comment("");
  PlaySound(NULL);

  //Remove all graphical objects from the chart on which the robot was launched:
  remove_all_objects(0);

}

void OnTimer(){

  //Timer with 1 sec. interval (bgP4YbxaQ):
  block_bTonyumSN();
  block_bO7LNEs4m();
  block_bS1JODjZj();
  block_bM4s77Wvc();

  //Every 10 sec. reset the counter of the performed deals number (max deals frequency):
  if(get_timer_tick_index() % 10 == 0){

    deals_max_frequency_counter_reset();

  }

  //Every 60 sec. reset the counter of the performed deals number (suspicious deals frequency):
  if(get_timer_tick_index() % 60 == 0){

    deals_suspicious_frequency_counter_reset();

  }

  timer_tick_index_increment();

}

void OnTick(void){

  //Blocks executed when a new tick is received on the symbol on the chart of which the robot was launched:
  block_bHxbWWtwW();
  block_bvIvs7SMe();
  block_boHVNlqnO();

}

void OnTrade(){

  //Check deals frequency:
  check_deals_frequency();

}

void OnChartEvent(
  const int      id,      // event identifier
  const long&    lparam,  // event parameter of type long
  const double&  dparam,  // event parameter of type double
  const string&  sparam   // event parameter of type string
){

  if(id == CHARTEVENT_OBJECT_CLICK){

    string object_name = sparam;

    if(ObjectGetInteger(0, object_name, OBJPROP_TYPE) == OBJ_BUTTON){

      if(object_name == "b4rh5uKlb"){

        block_bYGEhpZDM();

      }

      if(object_name == "bBTrBQlkS"){

        block_bCa4uSC92();

      }

      if(object_name == "bympSPhAp"){

        block_bXRrICVna();

      }

      Sleep(100);
      ObjectSetInteger(0, object_name, OBJPROP_STATE, false);
      ChartRedraw();

    }

  }

}

//Function of the block b0Wkfq6OD (set_complex_variable_value):
void block_b0Wkfq6OD(){

  vset(__ma_fast, ( moving_average(to_string(__symbol), PERIOD_CURRENT, (int)__fast_ma_period, MODE_SMA, PRICE_CLOSE, 0) ));

  block_b6LYVQGej();

}

//Function of the block b1VYatoxs (interface_element_modify):
void block_b1VYatoxs(){

  modify_text("b5BGSldua", DoubleToString(__ma_slow_prev, 2) );

}

//Function of the block b3UJfY74N (interface_element_modify):
void block_b3UJfY74N(){

  modify_text("bSDWBsxbk", DoubleToString(__ma_fast, 2) );

}

//Function of the block b3crj4ayK (log):
void block_b3crj4ayK(){

  log_to_file("Open short!");

  block_bzVyOb4tW();

}

//Function of the block b6LYVQGej (set_complex_variable_value):
void block_b6LYVQGej(){

  vset(__ma_slow_prev, ( moving_average(to_string(__symbol), PERIOD_CURRENT, (int)__slow_ma_period, MODE_SMA, PRICE_CLOSE, 1) ));

  block_bQMre45Bd();

}

//Function of the block bCa4uSC92 (place_market_order):
void block_bCa4uSC92(){

  place_market_order(to_string(__symbol), "BUY", __lot);

}

//Function of the block bHxbWWtwW (condition):
void block_bHxbWWtwW(){

  if (( __ma_fast > __ma_slow ) && ( __ma_fast_prev < __ma_slow_prev ) && ( get_position_info(to_string(__symbol), "POSITION_DIRECTION") != BUY_DIRECTION )){

    block_bUySCvh6M();

  }

}

//Function of the block bM4s77Wvc (print_debug_info):
void block_bM4s77Wvc(){

  print_debug_info(( to_string(to_double("ma_fast_prev = ")) + DoubleToString(__ma_fast_prev, 2) ));

}

//Function of the block bMgaVnT74 (pause):
void block_bMgaVnT74(){

  pause(1000);

  block_bT8xv0qGj();

}

//Function of the block bO7LNEs4m (print_debug_info):
void block_bO7LNEs4m(){

  print_debug_info(( to_string(to_double("ma_fast = ")) + DoubleToString(__ma_fast, 2) ));

}

//Function of the block bPlXoF1uA (set_complex_variable_value):
void block_bPlXoF1uA(){

  vset(__full_lot, ( __lot * to_double("2") ));

}

//Function of the block bQ4zsFoIh (log):
void block_bQ4zsFoIh(){

  log_to_file("Open long!");

  block_bMgaVnT74();

}

//Function of the block bQMre45Bd (set_complex_variable_value):
void block_bQMre45Bd(){

  vset(__ma_fast_prev, ( moving_average(to_string(__symbol), PERIOD_CURRENT, (int)__fast_ma_period, MODE_SMA, PRICE_CLOSE, 1) ));

  block_bRUr8MnXh();
  block_b1VYatoxs();
  block_b3UJfY74N();
  block_bofgi9HOT();

}

//Function of the block bRUr8MnXh (interface_element_modify):
void block_bRUr8MnXh(){

  modify_text("bj6MtjhZF", DoubleToString(__ma_fast_prev, 2) );

}

//Function of the block bS1JODjZj (print_debug_info):
void block_bS1JODjZj(){

  print_debug_info(( to_string(to_double("ma_slow_prev = ")) + DoubleToString(__ma_slow_prev, 2) ));

}

//Function of the block bT8xv0qGj (buy_signal_sound):
void block_bT8xv0qGj(){

  play_sound("buy_signal");

  block_bPlXoF1uA();

}

//Function of the block bTonyumSN (print_debug_info):
void block_bTonyumSN(){

  print_debug_info(( to_string(to_double("ma_slow = ")) + DoubleToString(__ma_slow, 2) ));

}

//Function of the block bTs8fZtAO (telegram_send_chart_screenshot):
void block_bTs8fZtAO(){

  telegram_send_chart_screenshot(to_string(__symbol), 2160, 720, (int)__admin_id);

}

//Function of the block bUySCvh6M (place_market_order):
void block_bUySCvh6M(){

  place_market_order(to_string(__symbol), "BUY", __full_lot);

  block_bfrp6ajWk();
  block_bjBTLJMym();
  block_boX0sSwri();
  block_bQ4zsFoIh();

}

//Function of the block bWkG0nSQa (telegram_send_message):
void block_bWkG0nSQa(){

  telegram_send_message("Open short!", (int)__admin_id);

}

//Function of the block bX2tY0y68 (terminal_print):
void block_bX2tY0y68(){

  terminal_print("Open short!");

}

//Function of the block bXRrICVna (close_position):
void block_bXRrICVna(){

  close_position(to_string(__symbol));

}

//Function of the block bYGEhpZDM (place_market_order):
void block_bYGEhpZDM(){

  place_market_order(to_string(__symbol), "SELL", __lot);

}

//Function of the block bYUS6GLT0 (set_complex_variable_value):
void block_bYUS6GLT0(){

  vset(__full_lot, ( __lot ));

}

//Function of the block bYi6ikfde (terminal_print):
void block_bYi6ikfde(){

  terminal_print("Robot started!");

}

//Function of the block bfrp6ajWk (terminal_print):
void block_bfrp6ajWk(){

  terminal_print("Open long!");

}

//Function of the block biLE3RJAD (sell_signal_sound):
void block_biLE3RJAD(){

  play_sound("sell_signal");

  block_bPlXoF1uA();

}

//Function of the block bjBTLJMym (telegram_send_message):
void block_bjBTLJMym(){

  telegram_send_message("Open long!", (int)__admin_id);

}

//Function of the block boHVNlqnO (set_complex_variable_value):
void block_boHVNlqnO(){

  vset(__ma_slow, ( moving_average(to_string(__symbol), PERIOD_CURRENT, (int)__slow_ma_period, MODE_SMA, PRICE_CLOSE, 0) ));

  block_b0Wkfq6OD();

}

//Function of the block boX0sSwri (telegram_send_chart_screenshot):
void block_boX0sSwri(){

  telegram_send_chart_screenshot(to_string(__symbol), 2160, 720, (int)__admin_id);

}

//Function of the block bofgi9HOT (interface_element_modify):
void block_bofgi9HOT(){

  modify_text("bLcIIkYqO", DoubleToString(__ma_slow, 2) );

}

//Function of the block bugmLdNsU (place_market_order):
void block_bugmLdNsU(){

  place_market_order(to_string(__symbol), "SELL", __full_lot);

  block_bX2tY0y68();
  block_bWkG0nSQa();
  block_bTs8fZtAO();
  block_b3crj4ayK();

}

//Function of the block bvIvs7SMe (condition):
void block_bvIvs7SMe(){

  if (( __ma_fast < __ma_slow ) && ( __ma_fast_prev > __ma_slow_prev ) && ( get_position_info(to_string(__symbol), "POSITION_DIRECTION") != SELL_DIRECTION )){

    block_bugmLdNsU();

  }

}

//Function of the block bzVyOb4tW (pause):
void block_bzVyOb4tW(){

  pause(1000);

  block_biLE3RJAD();

}
```

In the BotBrains editor, such trading robot can be built in minutes, while the code generation takes fractions of a second.

### Available blocks

There are more than 140 blocks available in the BotBrains editor. Below is a complete table of all available blocks. Please note that in order to fully understand how to use the editor, it is highly recommended to read the documentation.

_**Event blocks:**_

| Block | Description |
| --- | --- |
| **Start of robot work** | The block is activated once the robot is launched. |
| **End of robot work** | The block is activated once the robot is turned off. |
| **Depth of market change** | The block is activated each time the depth of market of the specified symbol changes. |
| **New tick** | The block is activated each time a new tick is received on the symbol, on the chart of which the robot was launched. |
| **Open volume change** | The block is activated each time the open volume on the specified symbol changes. |
| **Limit order number change** | The block is activated each time the number of active limit orders on the specified symbol changes. |
| **Stop orders number change** | The block is activated each time the number of active stop orders on the specified symbol changes. |
| **Timer** | The block is activated once every specified number of seconds. |
| **Key press** | The block is activated when the key with the specified code is pressed. |

_**Condition:**_

The "if" block is used to check certain conditions.

| Block | Description |
| --- | --- |
| **"If" block** | This block is used to perform various checks. This block has 1 input and 2 outputs. |

_**Loop:**_

The loop block is mainly used to enumerate certain values. For example, the loop block can be used to enumerate the list of active limit orders.

| Block | Description |
| --- | --- |
| **Loop "While"** | The block is activated as long as the specified condition is true. |

**_Trend indicators:_**

| Block |
| --- |
| **Adaptive Moving Average** |
| **Average Directional Movement Index** |
| **Average Directional Movement Index by Welles Wilder** |
| **Bollinger Bands** |
| **Double Exponential Moving Average** |
| **Envelopes** |
| **Fractal Adaptive Moving Average** |
| **Ichimoku** |
| **Moving Average** |
| **Parabolic SAR** |
| **Standard Deviation** |
| **Triple Exponential Moving Average** |
| **Variable Index Dynamic Average** |

**_Oscillators:_**

| Block |
| --- |
| **Average True Range** |
| **Bears Power** |
| **Bulls Power** |
| **Chaikin Oscillator** |
| **Commodity Channel Index** |
| **DeMarker** |
| **Force Index** |
| **MACD** |
| **Momentum** |
| **Moving Average of Oscillator** |
| **RSI (Relative Strength Index)** |
| **Relative Vigor Index** |
| **Stochastic Oscillator** |
| **TRIX (Triple Exponential Moving Averages Oscillator)** |
| **Larry Williams' Percent Range** |
| **Accumulation / Distribution** |
| **Money Flow Index** |
| **On Balance Volume** |
| **Volumes** |

**_Bill Williams:_**

| Block |
| --- |
| **Accelerator Oscillator** |
| **Alligator** |
| **Awesome Oscillator** |
| **Fractals** |
| **Gator** |
| **Market Facilitation Index** |

**_Chart analysis:_**

Getting information about the chart and individual bars. Drawing vertical and horizontal lines on the chart.

| Block | Description |
| --- | --- |
| **Bar information** | Get information on a specific chart bar. |
| **Chart information** | Get information on a specific chart. For example, get the number of available bars, get the time of the first available bar, or get the time of the last bar. |
| **Max price** | Get the maximal price of a symbol for the specified period of time. |
| **Min price** | Get the minimal price of a symbol for the specified period of time. |
| **Average price** | Get the average price of a symbol for the specified period of time. |
| **Draw horizontal line** | Draw a horizontal line on the chart of a symbol. |
| **Draw vertical line** | Draw a vertical line on the chart of a symbol. |
| **Remove all lines** | Remove all horizontal and vertical lines from the chart. |

_**Chart analysis:**_

Getting information on the depth of market and its specific quotes.

| Block | Description |
| --- | --- |
| **Quote info** | Get information on a specific quote of DOM (depth of market) |
| **Spread** | Get spread value (in ticks) of the DOM of the specified symbol. |

**_Transactions:_**

Place market, limit, and stop orders. Remove limit or stop orders. Close positions.

| Block | Description |
| --- | --- |
| **Market order** | Place a market order. |
| **Limit order** | Place a limit order. |
| **Remove limit order** | Remove a limit order |
| **Remove all limit orders** | Remove all limit orders by the specified symbol. |
| **Stop order** | Place a stop order. |
| **Remove stop order** | Remove a stop order. |
| **Remove all stop orders** | Remove all stop orders by the specified symbol. |
| **Close position** | Close position by the specified symbol. |
| **Close all open positions** | Close all positions opened by the robot. |

**_Variables:_**

| Block | Description |
| --- | --- |
| **Set variable simple value** | The new value of the variable is specified by a single input field. That is, with this block you can write something specific to the variable - just a number or some text. |
| **Set variable complex value** | The new value of the variable is determined by the calculated value. For example, using this block you can write the current deposit value or the current price of the traded symbol into a variable. |
| **Variable select** | Variable selection. For example, the "variable select" block can be used within a condition block to check the value of a variable. |

**_Sounds:_**

| Block | Description |
| --- | --- |
| **Smooth sound** | Play "Smooth sound" |
| **Alarm** | Play "Alarm" sound |
| **Buy signal** | Play "Buy signal" sound |
| **Sell signal** | Play "Sell signal" sound |

**_Information:_**

Getting information about the account, positions, active and historical limit/stop orders, historical deals. Getting information about the trading session, symbol specification and time.

| Block |
| --- |
| **Account information** |
| **Position information** |
| **Limit order information** |
| **All limit orders information** |
| **Stop order information** |
| **All stop orders information** |
| **History limit order information** |
| **History stop order information** |
| **History deal information** |
| **Trading session information** |
| **Symbol information** |
| **Time information** |

**_Enumerations:_**

Enum blocks are used to enumerate something. For example, you can use enum blocks to enumerate a list of symbols or a list of active limit/stop orders.

| Block | Description |
| --- | --- |
| **Symbol name** | Get symbol name. |
| **Request active orders list** | Request active orders list. |
| **Active limit order ticket** | Get active limit order ticket. |
| **Active stop order ticket** | Get active stop order ticket. |

Be sure to use "Request list active orders list" block before enumeration of the list of active orders. Otherwise you will work with irrelevant data.

**_History:_**

| Block | Description |
| --- | --- |
| **Request history** | Request history for the specified period of time. |
| **History deal ticket** | Get history deal ticket. |
| **History limit order ticket** | Get history limit order ticket. |
| **History stop order ticket** | Get history stop order ticket. |

Be sure to call the "Request history" block before enumerating the list of historical orders and deals. Otherwise, you will work with irrelevant data.

**_Telegram:_**

Using special blocks your robot can send messages and charts screenshots directly to your telegram.

| Block | Description |
| --- | --- |
| **Send message** | Send a telegram message to the user with the specified ID. |
| **Send chart screenshot** | Send a chart screenshot to the user with the specified ID. |
| **New line** | This block is used to create a line break in a Telegram message. |

**_Other blocks:_**

| Block | Description |
| --- | --- |
| **Journal message** | Print a message to the terminal journal. |
| **Terminal alert** | Make a notification in the terminal. |
| **Chart comment** | Show comment on the chart of the specified symbol. |
| **Log to file** | Make a log into the robot's log file. |
| **Pause** | Pause the robot for the specified number of milliseconds. |
| **Turn the robot off** | Turn the robot off. |
| **Close terminal** | Close the terminal. |

**_Interface elements:_**

| Block | Description |
| --- | --- |
| **Rectangle** | "Rectangle" interface element. |
| **Button** | "Button" interface element. |
| **Text** | "Text" interface element. |
| **Value input** | "Value input" interface element. |

**_Interface elements modifications:_**

| Block | Description |
| --- | --- |
| **Modify interface element** | Change a certain property of an interface element. |

**_Interface elements information:_**

| Block | Description |
| --- | --- |
| **Interface element info** | The block returns the value of the specified property of the interface element block with the specified ID. |

**_Predefined constants:_**

Predefined constants are possible values of certain properties. For example, the "direction" block contains predefined constants of possible directions: buy, sell, no direction.

| Block | Description |
| --- | --- |
| **Direction** | Possible position directions (buy, sell, no direction) |
| **Deal entry** | Possible deal entries (entry in, entry out, reverse, close a position by an opposite one) |
| **Deal type** | Possible deal types (buy, sell, balance, credit, correction, etc.) |

**_Debug:_**

The "print debug info" has many uses. It is mainly used to check the values of variables and the values that blocks return.

| Block | Description |
| --- | --- |
| **Print debug info** | Prints the specified message to the terminal journal. |

**_Mathematical operators:_**

| Block | Description |
| --- | --- |
| **+** | **Addition** |
| **-** | **Subtraction** |
| **/** | **Division** |
| **\*** | **Multiplication** |
| **√** | **Square root** |
| **^** | **Exponentiation** |
| **%** | **Division remainder** |
| **(** | **Opening parenthesis** |
| **)** | **Closing parenthesis** |
| **>** | **Greater** |
| **<** | **Less** |
| **>=** | **Greater or equal** |
| **<=** | **Less or equal** |

**_Logical operators:_**

| Block | Description |
| --- | --- |
| **AND** | Logical "AND" |
| **OR** | Logical "OR" |
| **NOT** | Logical "NOT" |

**_Teleports:_**

| Block |
| --- |
| **Teleport IN** |
| **Teleport OUT** |

**_Variable or constant select:_**

| Block |
| --- |
| **Variable select** |
| **Constant select** |

_**Value input:**_

| Block | Description |
| --- | --- |
| **Value input** | This block can be used to set certain values right in the scheme. |

_**Type conversion:**_

By default, all data is represented as a number. With type conversion blocks you can explicitly specify the format in which a certain value must be represented.

| Block | Description |
| --- | --- |
| **Regular string** | Convert the value to a regular string. |
| **Date and time format string** | Convert the value to a date and time format. |
| **Integer** | Convert the value to an integer. |
| **Fraction** | Convert the value to a fraction. |

### Conclusion

It took months of development to complete this project. If you have any ideas or suggestions regarding the editor's work, please send an email to support@botbrains.app. Your opinion will be taken into account.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9998](https://www.mql5.com/ru/articles/9998)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/387480)**
(29)


![John Winsome Munar](https://c.mql5.com/avatar/2022/6/62A73E8B-F64C.jpg)

**[John Winsome Munar](https://www.mql5.com/en/users/trozovka)**
\|
21 May 2022 at 14:40

Great app, very intuitive.


![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
7 Dec 2022 at 05:50

**Sergey Gubenko [#](https://www.mql5.com/ru/forum/381494/page2#comment_28828103):**

Unfortunately, the BotBrains programme is not completed. For an ordinary person it is not suitable, that to use this programme you need mql5 programming skills, so it is not a visual editor of Expert Advisors.

I agree. Ideally, it is desirable to do without the need to generate and compile code at all. And the interface is too overloaded - it is not much easier to learn it than to write an Expert Advisor.

![Churolnik](https://c.mql5.com/avatar/avatar_na2.png)

**[Churolnik](https://www.mql5.com/en/users/churolnik)**
\|
10 Jan 2023 at 11:24

What about ["custom indicators](https://www.mql5.com/en/articles/5 "Article: Switching to new rails: Custom indicators in MQL5")". Is it possible to create an indicator block independently? Can it be a separate option-order?

![vbymrf](https://c.mql5.com/avatar/avatar_na2.png)

**[vbymrf](https://www.mql5.com/en/users/vbymrf)**
\|
12 Jan 2023 at 15:11

It would be better to see how it's done in TcLab. It is so popular for a reason. It has certain nuances - they have created a completely different approach in visual programming. There are no variables as such, and other. This is the quickest bad thing, as well as labour-intensive in programming. But the principles had to be taken away.

In your case you are creating visual code in its purest form. It's as hard to learn as programming using a language. Because there is no escape from language constructs in principle.

![vbymrf](https://c.mql5.com/avatar/avatar_na2.png)

**[vbymrf](https://www.mql5.com/en/users/vbymrf)**
\|
12 Jan 2023 at 15:19

But the automatic module of interface creation may be interesting for programmers.


![Combinatorics and probability for trading (Part IV): Bernoulli Logic](https://c.mql5.com/2/44/ieu9.png)[Combinatorics and probability for trading (Part IV): Bernoulli Logic](https://www.mql5.com/en/articles/10063)

In this article, I decided to highlight the well-known Bernoulli scheme and to show how it can be used to describe trading-related data arrays. All this will then be used to create a self-adapting trading system. We will also look for a more generic algorithm, a special case of which is the Bernoulli formula, and will find an application for it.

![Universal regression model for market price prediction (Part 2): Natural, technological and social transient functions](https://c.mql5.com/2/43/universal_regression__1.png)[Universal regression model for market price prediction (Part 2): Natural, technological and social transient functions](https://www.mql5.com/en/articles/9868)

This article is a logical continuation of the previous one. It highlights the facts that confirm the conclusions made in the first article. These facts were revealed within ten years after its publication. They are centered around three detected dynamic transient functions describing the patterns in market price changes.

![Learn how to design different Moving Average systems](https://c.mql5.com/2/45/why-and-how.png)[Learn how to design different Moving Average systems](https://www.mql5.com/en/articles/3040)

There are many strategies that can be used to filter generated signals based on any strategy, even by using the moving average itself which is the subject of this article. So, the objective of this article is to share with you some of Moving Average Strategies and how to design an algorithmic trading system.

![Graphics in DoEasy library (Part 89): Programming standard graphical objects. Basic functionality](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__1.png)[Graphics in DoEasy library (Part 89): Programming standard graphical objects. Basic functionality](https://www.mql5.com/en/articles/10119)

Currently, the library is able to track standard graphical objects on the client terminal chart, including their removal and modification of some of their parameters. At the moment, it lacks the ability to create standard graphical objects from custom programs.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/9998&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069282348572279429)

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