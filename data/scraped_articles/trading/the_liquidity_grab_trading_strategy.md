---
title: The Liquidity Grab Trading Strategy
url: https://www.mql5.com/en/articles/16518
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:58:43.275539
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/16518&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068867983012462241)

MetaTrader 5 / Examples


The [liquidity grab](https://www.mql5.com/go?link=https://www.equiruswealth.com/blog/liquidity-grabs-a-crucial-concept-in-the-investment-world "https://www.equiruswealth.com/blog/liquidity-grabs-a-crucial-concept-in-the-investment-world") trading strategy is a key component of Smart Money Concepts (SMC), which seeks to identify and exploit the actions of institutional players in the market. It involves targeting areas of high liquidity, such as support or resistance zones, where large orders can trigger price movements before the market resumes its trend. This article explains the concept of liquidity grab in detail and outlines the development process of the liquidity grab trading strategy Expert Advisor in MQL5.

### Strategy Overview: Key Concepts and Tactics

The liquidity grab strategy focuses on exploiting areas of high liquidity, such as support and resistance levels, where institutional traders often manipulate prices to trigger stop-losses and drive volatility. This strategy leverages common market dynamics to anticipate and capitalize on these movements.

#### Core Concepts

1. **Stop-Loss Hunting**: Driving prices to activate stop-loss orders, causing cascading buying or selling.
2. **Layering and Spoofing**: Using fake orders to mislead traders about market direction.
3. **Iceberg Orders**: Hiding large trades by splitting them into smaller, visible parts.
4. **Momentum Ignition**: Creating artificial momentum to lure other traders before reversing.
5. **Support and Resistance Manipulation**: Exploiting key price levels for predictable reactions.
6. **Psychological Price Points**: Leveraging round numbers to influence behavior.

[Market manipulation](https://www.mql5.com/go?link=https://www.investor.gov/introduction-investing/investing-basics/glossary/market-manipulation%23%3a%7e%3atext%3dMarket%2520manipulation%2520may%2520involve%2520techniques%2csecurity%2520than%2520is%2520the%2520case. "https://www.investor.gov/introduction-investing/investing-basics/glossary/market-manipulation#:~:text=Market%20manipulation%20may%20involve%20techniques,security%20than%20is%20the%20case.") involves intentionally influencing a security’s price or volume to create misleading trading conditions. While most institutional traders adhere to legal and ethical standards, some may engage in manipulative practices to achieve specific strategic goals. Here’s a concise overview of why and how this occurs:

**Motivations**:

- Maximize profits through short-term price moves and arbitrage.
- Conceal trading intentions from competitors.
- Execute large trades with minimal market impact.

**Tactics**:

- Triggering stop-losses to create liquidity or push prices.
- Controlling order flow with iceberg orders or spoofing.
- Targeting support, resistance, or psychological price levels.

Understanding these dynamics allows traders to anticipate institutional behavior and integrate these insights into automated tools for more effective strategies.

We want to capitalize on temporary price movements caused by the accumulation of stop-loss orders and large market participants' actions. By identifying areas of liquidity, traders aim to enter the market at optimal points before the price reverses and continues in the prevailing trend, offering favorable risk-to-reward opportunities.

Here is a rough diagram of what it should look like.

![example diagram](https://c.mql5.com/2/110/example1.png)

### Strategy Development

I propose that we adopt the following order when we code an EA:

1. Consider the necessary functions for this strategy and determine how to encapsulate them into different components.
2. Code each function one by one, declaring related global variables or initializing relevant handles as you proceed.
3. After implementing each function, review it and consider how to connect them, such as by passing parameters or calling functions within other functions.
4. Finally, navigate to OnTick() and develop the logic using the functions from the previous steps.

**Firstly, we try to quantify the rules.** SMC is mainly traded by discretionary traders because there are many unquantifiable nuances involved. Objectively, it is difficult to define the exact characteristics that a manipulation must have. One approach is to analyze the volume change in the order flow, but the volume data provided by brokers is often unreliable. In markets like forex, transactions are not centralized, and for centralized exchanges like futures, most brokers provide data from their liquidity providers rather than centralized data.  A more straightforward and optimizable method is through technical analysis, which is the approach we will use in this article. To quantify the rules, we will simplify the strategy into the following segments:

1.  A rejection candle pattern formed at a key level, where the key level is defined as the highest or lowest point within the look-back period.
2.  Following this rejection candle, the price reverses and breaks through the key level on the opposite side with a shorter look-back period.
3.  Finally, if the overall movement aligns with the broader trend, as indicated by the price's position relative to the moving average, we enter the trade with a fixed stop loss and take profit.

Note that we could add more rules to better mimic market manipulation characteristics, but it is advisable to keep the strategy as simple as possible to avoid overfitting.

**Next, we code the related functions.** These are essential functions to make order executions after calculating take profit and stop loss, and keep track of the order tickets.

```
//+------------------------------------------------------------------+
//| Expert trade transaction handling function                       |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans, const MqlTradeRequest& request, const MqlTradeResult& result) {
    if (trans.type == TRADE_TRANSACTION_ORDER_ADD) {
        COrderInfo order;
        if (order.Select(trans.order)) {
            if (order.Magic() == Magic) {
                if (order.OrderType() == ORDER_TYPE_BUY) {
                    buypos = order.Ticket();
                } else if (order.OrderType() == ORDER_TYPE_SELL) {
                    sellpos = order.Ticket();
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Execute sell trade function                                      |
//+------------------------------------------------------------------+
void executeSell() {
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    bid = NormalizeDouble(bid, _Digits);
    double tp = NormalizeDouble(bid - tpp * _Point, _Digits);
    double sl = NormalizeDouble(bid + slp * _Point, _Digits);
    trade.Sell(lott, _Symbol, bid, sl, tp);
    sellpos = trade.ResultOrder();
}

//+------------------------------------------------------------------+
//| Execute buy trade function                                       |
//+------------------------------------------------------------------+
void executeBuy() {
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    ask = NormalizeDouble(ask, _Digits);
    double tp = NormalizeDouble(ask + tpp * _Point, _Digits);
    double sl = NormalizeDouble(ask - slp * _Point, _Digits);
    trade.Buy(lott, _Symbol, ask, sl, tp);
    buypos = trade.ResultOrder();
}
```

These two functions identify and return the highest or lowest point within a given range, ensuring that this point qualifies as a key level by verifying the presence of support or resistance that triggers a reversal from that level.

```
//+------------------------------------------------------------------+
//| find the key level high given a look-back period                 |
//+------------------------------------------------------------------+
double findhigh(int Range = 0)
{
   double highesthigh = 0;
   for (int i = BarsN; i < Range; i++)
   {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      if (i > BarsN && iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, BarsN * 2 + 1, i - BarsN) == i)
      //used to make sure there's rejection for this high
      {
         if (high > highesthigh)
         {
            return high;
         }
      }
      highesthigh = MathMax(highesthigh, high);
   }
   return 99999;
}

//+------------------------------------------------------------------+
//| find the key level low given a look-back period                  |
//+------------------------------------------------------------------+
double findlow(int Range = 0)
{
   double lowestlow = DBL_MAX;
   for (int i = BarsN; i < Range; i++)
   {
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      if (i > BarsN && iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, BarsN * 2 + 1, i - BarsN) == i)
      {
         if (lowestlow > low)
         {
            return low;
         }
      }
      lowestlow = MathMin(lowestlow, low);
   }
   return -1;
}
```

The findhigh() function ensures there's a rejection at the identified high point by checking if the highest high within a specified range occurs at the current bar (i) and if the highest point within a larger range (twice the look-back period) coincides with that bar. This suggests a rejection, as the price couldn't break higher after reaching that level. If true, it returns the high value as a potential key level. The findlow() function is simply the conversed logic.

These two functions detect whether the last closed candle is a rejection candle on the key level, which can be seen as the liquidity grab behavior.

```
//+------------------------------------------------------------------+
//| Check if the market rejected in the upward direction             |
//+------------------------------------------------------------------+
bool IsRejectionUp(int shift=1)
{
   // Get the values of the last candle (shift = 1)
   double open = iOpen(_Symbol,PERIOD_CURRENT, shift);
   double close = iClose(_Symbol,PERIOD_CURRENT, shift);
   double high = iHigh(_Symbol,PERIOD_CURRENT, shift);
   double low = iLow(_Symbol,PERIOD_CURRENT,shift);

   // Calculate the body size
   double bodySize = MathAbs(close - open);

   // Calculate the lower wick size
   double lowerWickSize = open < close ? open - low : close - low;

   // Check if the lower wick is significantly larger than the body
   if (lowerWickSize >= wickToBodyRatio * bodySize&&low<findlow(DistanceRange)&&high>findlow(DistanceRange))
   {
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Check if the market rejected in the downward direction           |
//+------------------------------------------------------------------+
bool IsRejectionDown(int shift = 1)
{
   // Get the values of the last candle (shift = 1)
   double open = iOpen(_Symbol,PERIOD_CURRENT, shift);
   double close = iClose(_Symbol,PERIOD_CURRENT, shift);
   double high = iHigh(_Symbol,PERIOD_CURRENT, shift);
   double low = iLow(_Symbol,PERIOD_CURRENT,shift);

   // Calculate the body size
   double bodySize = MathAbs(close - open);

   // Calculate the upper wick size
   double upperWickSize = open > close ? high - open : high - close;

   // Check if the upper wick is significantly larger than the body
   if (upperWickSize >= wickToBodyRatio * bodySize&&high>findhigh(DistanceRange)&&low<findhigh(DistanceRange))
   {
      return true;
   }

   return false;
}
```

A candle exhibits a rejection pattern when its wick is significantly larger than its body, and the candle direction reverses from the previous direction.

Utilizing the last two functions, these were created to loop through a specified look-back period to detect any liquidity grab behavior, which will be used to checks whether such behavior occurred before observing a reversal and breakout signal.

```
//+------------------------------------------------------------------+
//| check if there were rejection up for the short look-back period  |
//+------------------------------------------------------------------+
bool WasRejectionUp(){
   for(int i=1; i<CandlesBeforeBreakout;i++){
     if(IsRejectionUp(i))
        return true;
   }
    return false;
}

//+------------------------------------------------------------------+
//| check if there were rejection down for the short look-back period|
//+------------------------------------------------------------------+
bool WasRejectionDown(){
   for(int i=1; i<CandlesBeforeBreakout;i++){
     if(IsRejectionDown(i))
        return true;
   }
    return false;
}
```

To fetch the data for the current moving average value, we first initialize the handle in OnInit() function.

```
int handleMa;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    trade.SetExpertMagicNumber(Magic);
    handleMa = iMA(_Symbol, PERIOD_CURRENT, MaPeriods, 0, MODE_SMA,PRICE_CLOSE);

    if (handleMa == INVALID_HANDLE) {
        Print("Failed to get indicator handles. Error: ", GetLastError());
        return INIT_FAILED;
    }
    return INIT_SUCCEEDED;
}
```

The Moving Average value can then be easily accessed by creating a buffer array and copying the handle value to the buffer array like this:

```
double ma[];
if (CopyBuffer(handleMa, 0, 1, 1, ma) <= 0) {
            Print("Failed to copy MA data. Error: ", GetLastError());
            return;
}
```

**Finally, proceed to OnTick() to apply the trading logic to the program using the functions you have defined.** This ensures that we only calculate the signal when a new bar has formed, by checking if the current bar is different from the last saved closed bar. This conserves computing power and enables smoother transactions.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    int bars = iBars(_Symbol, PERIOD_CURRENT);
    if (barsTotal != bars) {
        barsTotal = bars;
```

Then we simply apply the signal condition like this:

```
if(WasRejectionDown()&&bid<ma[0]&&bid<findlow(CandlesBeforeBreakout))
        executeSell();
else if(WasRejectionUp()&&ask>ma[0]&&ask>findhigh(CandlesBeforeBreakout))
        executeBuy();
```

After this step, try to compile the program and go to the backtest visualizer to check whether the EA works.

In the backtest visualizer, a typical entry would look like this:

![LG example](https://c.mql5.com/2/110/LG_example.jpg)

### Suggestions

Although we have finished the main idea of the strategy, I have a few suggestions for implementing this EA in the live market:

1\. Market manipulations happen rapidly, so it's best to trade intraday with this strategy using timeframes such as 5 minutes or 15 minutes. Lower timeframes may be more susceptible to false signals, while higher timeframes may react too slowly to market manipulations.

2\. Market manipulations typically occur during periods of high volatility, such as the Forex New York/London sessions or stock market open/close times. It's advisable to implement a function that restricts trading to these specific hours, as shown below:

```
//+------------------------------------------------------------------+
//| Check if the current time is within the specified trading hours  |
//+------------------------------------------------------------------+
bool IsWithinTradingHours() {
    datetime currentTime = TimeTradeServer();
    MqlDateTime timeStruct;
    TimeToStruct(currentTime, timeStruct);
    int currentHour = timeStruct.hour;

    if (( currentHour >= startHour1 && currentHour < endHour1) ||
        ( currentHour >= startHour2 && currentHour < endHour2))
         {
        return true;
    }
    return false;
}
```

3.  If the price consolidates around the key levels, it may trigger multiple trades consecutively in both directions. To ensure only one trade is executed at a time, we add another criterion: both position tickets must be set to 0, indicating no open positions for this EA. We reset them to 0 by writing these lines in the OnTick() function.

```
if(buypos>0&&(!PositionSelectByTicket(buypos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
 buypos = 0;
 }
if(sellpos>0&&(!PositionSelectByTicket(sellpos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
 sellpos = 0;
 }
```

We update our original code to incorporate the changes we have just made.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    int bars = iBars(_Symbol, PERIOD_CURRENT);
    if (barsTotal != bars) {
        barsTotal = bars;

        double ma[];

        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

        if (CopyBuffer(handleMa, 0, 1, 1, ma) <= 0) {
            Print("Failed to copy MA data. Error: ", GetLastError());
            return;
        }
        if(WasRejectionDown()&&IsWithinTradingHours()&&sellpos==buypos&&bid<ma[0]&&bid<findlow(CandlesBeforeBreakout))
        executeSell();
        else if(WasRejectionUp()&&IsWithinTradingHours()&&sellpos==buypos&&ask>ma[0]&&ask>findhigh(CandlesBeforeBreakout))
        executeBuy();

     if(buypos>0&&(!PositionSelectByTicket(buypos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      buypos = 0;
      }
     if(sellpos>0&&(!PositionSelectByTicket(sellpos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      sellpos = 0;
      }
    }
}
```

### Backtest

For this article, we'll use this EA on GBPUSD on the 5-minute timeframe.

Here are the parameters settings we decided to use for this expert advisor:

![parameters](https://c.mql5.com/2/110/LG_parameters__1.png)

**Important notes:**

- For take profit and stop loss, we select a reasonable point amount based on intraday volatility. Since this strategy essentially rides with the trend, it is recommended that the reward-to-risk ratio be greater than 1.
- DistanceRange is the look-back period for searching key levels for liquidity grab signals.
- Similarly, CandlesBeforeBreakout is the look-back period for searching recent key levels for breakout signals.
- The wick-to-body ratio can be adjusted to a value that the trader deems sufficient to illustrate a rejection pattern.
- The trading hours are based on your broker's server time. For my broker (GMT+0), the volatile forex New York session period is from 13:00 to 19:00.

**Let's now run the backtest from 2020.11.1 – 2024.11.1:**

![setting](https://c.mql5.com/2/110/LG_setting.png)

![curve](https://c.mql5.com/2/110/LG_curve.png)

![result](https://c.mql5.com/2/110/LG_result.png)

The strategy tested decent result for the past 4 years.

### Conclusion

In this article, we first introduced the concept of liquidity grabs and their underlying motivations. We then provided a step-by-step guide on building the Expert Advisor (EA) for this strategy from the ground up. Next, we offered additional recommendations for optimizing the EA. Finally, we demonstrated its potential profitability through a four-year backtest, featuring over 200 trades.

We hope you find this strategy useful and that it inspires you to build upon it, whether by creating similar strategies or optimizing its settings. The corresponding file for the Expert Advisor is attached below. Feel free to download it and experiment with it.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16518.zip "Download all attachments in the single ZIP archive")

[Liquidity\_Grab\_Breakout\_Example\_Code.mq5](https://www.mql5.com/en/articles/download/16518/liquidity_grab_breakout_example_code.mq5 "Download Liquidity_Grab_Breakout_Example_Code.mq5")(8.89 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Decoding Opening Range Breakout Intraday Trading Strategies](https://www.mql5.com/en/articles/17745)
- [Day Trading Larry Connors RSI2 Mean-Reversion Strategies](https://www.mql5.com/en/articles/17636)
- [Exploring Advanced Machine Learning Techniques on the Darvas Box Breakout Strategy](https://www.mql5.com/en/articles/17466)
- [The Kalman Filter for Forex Mean-Reversion Strategies](https://www.mql5.com/en/articles/17273)
- [Robustness Testing on Expert Advisors](https://www.mql5.com/en/articles/16957)
- [Trend Prediction with LSTM for Trend-Following Strategies](https://www.mql5.com/en/articles/16940)
- [The Inverse Fair Value Gap Trading Strategy](https://www.mql5.com/en/articles/16659)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/479947)**
(4)


![linfo2](https://c.mql5.com/avatar/2023/4/6438c14d-e2f0.png)

**[linfo2](https://www.mql5.com/en/users/neilhazelwood)**
\|
28 Jan 2025 at 18:27

Thanks for the code , very nicely written article and nicely put together, the code is very helpful thank you . Interesting if you see the SMC traders in social media the returns very different . Will review the transactions and try a trailing stop and a trailing tp or some Fibonacci on the external ranges


![Zhuo Kai Chen](https://c.mql5.com/avatar/2024/11/6743e84b-8a3d.jpg)

**[Zhuo Kai Chen](https://www.mql5.com/en/users/sicklemql)**
\|
29 Jan 2025 at 02:14

**linfo2 [#](https://www.mql5.com/en/forum/479947#comment_55758244):**

Thanks for the code , very nicely written article and nicely put together, the code is very helpful thank you . Interesting if you see the SMC traders in social media the returns very different . Will review the transactions and try a trailing stop and a trailing tp or some Fibonacci on the external ranges

Thanks for your comment! Yes, I do see the SMC traders in social media. Generally, I think they don't really agree on the strategy in terms of liquidity grab. Some look for two fakeouts instead of one, and some look at trading volume. Overall their actions involve some discretions from themselves which make them hard to evaluate the validity of their strategies. Nevertheless, I look forward to your result experimenting with trailing sl/tp and Fibonacci ranges.

![rapidace1](https://c.mql5.com/avatar/avatar_na2.png)

**[rapidace1](https://www.mql5.com/en/users/rapidace1)**
\|
2 Mar 2025 at 13:07

This concept looks fresh to me, thanks for sharing.


![Joab jrs](https://c.mql5.com/avatar/2021/4/6075C0C6-DBF1.png)

**[Joab jrs](https://www.mql5.com/en/users/joabjrs)**
\|
28 Dec 2025 at 10:15

Automatic translation was applied by a moderator. Please post in the language of the forum section you selected.

Hi friend, I just finished reading all your articles and they are very interesting.

I was already developing something about three of them and I will send them to you soon for you to validate and add something if necessary!

![Adaptive Social Behavior Optimization (ASBO): Two-phase evolution](https://c.mql5.com/2/85/Adaptive_Social_Behavior_Optimization__Part_2__LOGO.png)[Adaptive Social Behavior Optimization (ASBO): Two-phase evolution](https://www.mql5.com/en/articles/15329)

We continue dwelling on the topic of social behavior of living organisms and its impact on the development of a new mathematical model - ASBO (Adaptive Social Behavior Optimization). We will dive into the two-phase evolution, test the algorithm and draw conclusions. Just as in nature a group of living organisms join their efforts to survive, ASBO uses principles of collective behavior to solve complex optimization problems.

![Neural Networks in Trading: Spatio-Temporal Neural Network (STNN)](https://c.mql5.com/2/84/Neural_networks_in_trading_STNN___LOGO.png)[Neural Networks in Trading: Spatio-Temporal Neural Network (STNN)](https://www.mql5.com/en/articles/15290)

In this article we will talk about using space-time transformations to effectively predict upcoming price movement. To improve the numerical prediction accuracy in STNN, a continuous attention mechanism is proposed that allows the model to better consider important aspects of the data.

![Implementing the SHA-256 Cryptographic Algorithm from Scratch in MQL5](https://c.mql5.com/2/112/Implementing_the_SHA-256_Cryptographic_Algorithm_from_Scratch_in_MQL5__LOGO.png)[Implementing the SHA-256 Cryptographic Algorithm from Scratch in MQL5](https://www.mql5.com/en/articles/16357)

Building DLL-free cryptocurrency exchange integrations has long been a challenge, but this solution provides a complete framework for direct market connectivity.

![Neural Network in Practice: Pseudoinverse (II)](https://c.mql5.com/2/84/Rede_neural_na_prstica__Pseudo_Inversa__LOGO_.png)[Neural Network in Practice: Pseudoinverse (II)](https://www.mql5.com/en/articles/13733)

Since these articles are educational in nature and are not intended to show the implementation of specific functionality, we will do things a little differently in this article. Instead of showing how to apply factorization to obtain the inverse of a matrix, we will focus on factorization of the pseudoinverse. The reason is that there is no point in showing how to get the general coefficient if we can do it in a special way. Even better, the reader can gain a deeper understanding of why things happen the way they do. So, let's now figure out why hardware is replacing software over time.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pzirrrtldfvhdsnpioqyvvfoahlsodko&ssn=1769180321133479700&ssn_dr=0&ssn_sr=0&fv_date=1769180321&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16518&back_ref=https%3A%2F%2Fwww.google.com%2F&title=The%20Liquidity%20Grab%20Trading%20Strategy%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918032196211545&fz_uniq=5068867983012462241&sv=2552)

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