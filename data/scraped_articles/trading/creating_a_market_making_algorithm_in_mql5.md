---
title: Creating a market making algorithm in MQL5
url: https://www.mql5.com/en/articles/13897
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:30:06.255090
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=zpkdyfwzbsglosbgdwlclckvtiropkqn&ssn=1769092205577936941&ssn_dr=0&ssn_sr=0&fv_date=1769092205&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13897&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20market%20making%20algorithm%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909220518156809&fz_uniq=5049176043291518572&sv=2552)

MetaTrader 5 / Trading


### What is liquidity?

Liquidity of financial markets is the "saturation" of the market with money in the form of orders and positions. This allows traders to quickly sell shares (or currencies) for large amounts. The higher the market liquidity, the easier it is to sell or buy an asset for large amounts without significant losses due to slippage.

Slippage is the main evil of large players: the largest funds note that it is not so easy to handle a large position, and often the transaction is closed at a loss only because of order "slippage". Order slippage occurs when a transaction is opened at one price and executed at another price different from the expected one. When a trader only has a couple of hundred dollars, there are usually no problems with liquidity (except for completely illiquid market depths of third-rate cryptocurrencies). But when we deal with hundreds of millions of dollars, then it is difficult to open and close a position at the same time. This is directly related to the market liquidity.

Liquidity in the market is filled thanks to market makers. Their main task is to maintain liquidity. These market participants do everything to make trading as smooth as possible for you, so that there are no sharp gaps in quotes and both a buyer and a seller always receive prices that suit them.

In a market having no market maker, we will very often see sharp price swings in one direction, huge asset fluctuations and quote gaps.

### How does a market maker work, and why is he not a "puppet master"?

Many traders are confident that the market maker is some kind of puppet master - a manipulator who moves prices where he wants, breaks the stop levels, tricks the crowd into stop orders, etc.

In fact, the market maker does not need to make the "crowd" lose at all. The market "crowd" loses on its own due to spreads, commissions and swaps.

As for market shifts in the right direction, this is also not the task of a market maker. All that a market maker is obligated to do under his agreement with the exchange is to provide a buy quote to a buyer, and a sell quote to a seller, as well as fill the empty "market depth" if necessary.

Without market makers, the market would be completely different: we would constantly see price gaps, quote gaps, constant squeezes in both directions, as well as huge price jumps in both directions. All this can still be found today in those markets where it is unprofitable for a market maker to be present, for example in many US penny stocks.

### New AMM technologies in the crypto market

But what if we replace a participant with a smart contract? In other words, what if instead of market makers, we set up an automatic system for adjusting supply and demand, as well as general quotation?

This is roughly how decentralized exchanges (DEX) appeared. They were the first to use the AMM (automated market making) mechanism. The AMM algorithm works through a special liquidity pool using the resource of participants for transactions between them. The price and volume of exchanges are always controlled by the algorithm. This allows bringing all sellers together with all buyers supposedly without losses for the participants. In reality, however, all DEXs have huge price slippage. You are guaranteed to lose a large percentage on the token exchange in case of a large transaction volume.

Besides, this innovation has not eliminated market manipulations. There are plenty of them on a DEX. Even token creators on a DEX can easily pump their tokens and cash out the entire token liquidity pool.

### How do market makers combat price manipulations?

Although this is not the responsibility of market makers, they often extinguish attempts to arrange a pump and dump scheme in the bud when prices are just beginning to be driven up by fraudulent participants. In these initial phases, the market maker throws huge shares of limit orders at the player who is trying to push "market" prices up. This extinguishes demand, so pump scheme newcomers very often break their teeth against the market maker. But if the pump is well planned, the influx of many market orders, powerfully moving the price, forces the market maker to temporarily leave the market.

### When do market makers leave the market?

Most market makers stipulate in their agreements with exchanges that they turn off their algorithms and leave the market during holidays, periods of abnormal activity and periods of important news releases. This is due to MM’s desire to preserve their capital.

We can see the market maker leaving the market immediately by the extended spread. Have you seen how the spread widens even on the ECN on powerful global news release? The usual spread narrowness is achieved through the efforts of market makers. Therefore, without them, we will face very bad trading conditions among other things, including wide spreads, large price slippages, sudden dips and price spikes - all wild market delights.

### What is market maker's inventory risk?

Many people think that a market maker does not bear any risks at all. However, this is not the case. The main risk of a market maker is inventory risk. This risk lies in the fact that a position can sharply move in one direction without the ability to off-load it and make money on the spread. For example, when a frenzied crowd sells an asset, the market maker is forced to buy out the entire supply. As a result, the price goes into the negative driving MM into losses.

Companies try to avoid this risk by using special spread centering equations and determining the optimal price for buying and selling. But this is not always achievable. Even if the price is not optimal, MM's job is to supply liquidity to the market, and they must do this job, even if they are temporarily operating at a loss.

### Analyzing records of the largest market maker on the planet - Kenneth Griffin's company

While analyzing the activity of the largest market maker in the world - Citadel Securities founded by Kenneth Griffin - it becomes clear how important its role is in the financial markets.

The company's reports show an impressive impact: 7 out of 10 trades in the US stock market depend on the liquidity provided by this market maker. This activity demonstrates the significant role of Citadel Securities in maintaining stability and availability of liquidity in this market.

To evaluate the scale of the influence of Griffin's company, it can be mentioned that about 900 million lots of US shares pass through its algorithms every day. This significant trading volume reflects the company's high activity and influence on the US exchange.

By the way, the evolution of Kenneth Griffin moving from directed trading to market making is very interesting. Griffin's company is very actively expanding into global markets actively exploring Asian exchanges and providing liquidity there.

### Preparing a market maker EA

So, we have figured out the theory. It is time to start creating a market maker EA! Of course, our algorithm will be very simple. We will not build spread trading according to special equations.

Instead, we will implement the simplest algorithm that will keep two limit orders constantly open - sell limit and buy limit.

### The simplest implementation of market making in MQL5

Let's analyze the code of our algorithm. Code heading. This section sets the basic parameters of the strategy, such as lot size, profit levels, EA magic number, selected currency pairs for trading, etc. :

```
//+------------------------------------------------------------------+
//|                                                  MarketMaker.mq5 |
//|                                Copyright 2023, Evgeniy Koshtenko |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Evgeniy Koshtenko"
#property link      "https://www.mql5.com/en/users/koshtenko"
#property version   "1.00"

#include <Trade\Trade.mqh>        // Include the CTrade trading class

//--- input parameters
input double Lots       = 0.1;    // lot
input double Profit     = 0.1;    // profit
input double BProfit    = 11;     // buy profit
input double SProfit    = 11;     // sell profit
input int StopLoss      = 0;      // stop loss
input int TakeProfit    = 0;      // take profit
input int    Count      = 5;      // number of orders
input int    Delta      = 55;     // delta
input int    Magic      = 123;    // magic number

input bool   BuyLimit   = 1;      // Buy Limit
input bool   SellLimit  = 1;      // Sell Limit

input string Symbol1    = "EURUSD";
input string Symbol2    = "GBPUSD";
input string Symbol3    = "USDCHF";
input string Symbol4    = "USDJPY";
input string Symbol5    = "USDCAD";
input string Symbol6    = "AUDUSD";
input string Symbol7    = "NZDUSD";
input string Symbol8    = "EURGBP";
input string Symbol9    = "CADCHF";
input int MaxOrders = 20; // Max number of orders
CTrade trade;

datetime t=0;
int delta=0;
```

It includes basic settings such as delta between orders, closing profit (total, buy profit and sell profit), EA magic number, trading library import, as well as selecting currency pairs for trading and limiting the number of orders.

The initialization and deinitialization functions are generally standard. The OnInit() function is called when the EA starts, and OnDeinit() is called when it ends. OnInit() sets the EA magic number and the trading function timer:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   // Set a timer with a resolution of 10000 milliseconds (10 seconds)
   EventSetMillisecondTimer(100000);
   trade.SetExpertMagicNumber(Magic);
//---
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {// Disable timer
   EventKillTimer();
   Comment("");
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
```

Here are the functions for counting open orders and open positions. CountOrders and CountTrades count open orders and positions for a specific symbol taking into account the EA magic number.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CountOrders(string symbol, ENUM_ORDER_TYPE orderType) {
  int count = 0;

  for(int i = OrdersTotal()-1; i >= 0; i--) {

    ulong ticket = OrderGetTicket(i);

    if(!OrderSelect(ticket)) {
      continue;
    }

    if(OrderGetInteger(ORDER_TYPE) != orderType) {
      continue;
    }

    if(PositionGetString(POSITION_SYMBOL) != symbol ||
       PositionGetInteger(POSITION_MAGIC) != Magic) {
      continue;
    }

    count++;
  }

  return count;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CountTrades(string symbol, ENUM_POSITION_TYPE type) {
  int count = 0;

  for(int i=PositionsTotal()-1; i>=0; i--) {

    ulong ticket=PositionGetTicket(i);

    if(!PositionSelectByTicket(ticket)) {
      continue;
    }

    if(PositionGetString(POSITION_SYMBOL)==symbol &&
       PositionGetInteger(POSITION_TYPE)==type) {

      count++;
    }
  }

  return count;
}
```

Here are the functions for deleting orders, calculating profits and closing orders. DelOrder deletes all orders for a specific symbol using a magic number. AllProfit calculates the total profit or profit from buy/sell trades for a specific symbol taking into account the magic number.

```
//+------------------------------------------------------------------+
//|  Position Profit                                                 |
//+------------------------------------------------------------------+
double AllProfit(string symbol, int positionType = -1) {

  double profit = 0;

  for(int i = PositionsTotal()-1; i >= 0; i--) {

    ulong ticket = PositionGetTicket(i);

    if(!PositionSelectByTicket(ticket)) {
      continue;
    }

    if(PositionGetString(POSITION_SYMBOL) != symbol ||
       PositionGetInteger(POSITION_MAGIC) != Magic) {
      continue;
    }

    if(positionType != -1 &&
       PositionGetInteger(POSITION_TYPE) != positionType) {
      continue;
    }

    profit += PositionGetDouble(POSITION_PROFIT);

  }

  return profit;

}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CloseAll(string symbol, int positionType = -1) {

  for(int i = PositionsTotal()-1; i >= 0; i--) {

    ulong ticket = PositionGetTicket(i);

    if(!PositionSelectByTicket(ticket)) {
      continue;
    }

    if(PositionGetString(POSITION_SYMBOL) != symbol ||
       PositionGetInteger(POSITION_MAGIC) != Magic) {
      continue;
    }

    if(positionType != -1 &&
       PositionGetInteger(POSITION_TYPE) != positionType) {
      continue;
    }

    trade.PositionClose(ticket);

  }

}
```

Finally, the two main functions are the trading function and the tick function. Trade is responsible for placing limit buy and sell orders taking into account the specified parameters. OnTimer calls the Trade function to trade the selected symbol and displays profit information for that symbol.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Trade(string symb)
  {
   double sl = 0, tp = 0;
   double pr=0;
   double Bid=SymbolInfoDouble(symb,SYMBOL_BID);

   if(AllProfit(symb)>Profit && Profit>0)
      CloseAll(symb);

   if(AllProfit(symb)>Profit && Profit>0)
      CloseAll(symb);

   if(AllProfit(symb,0)>BProfit && BProfit>0)
      CloseAll(symb,0);

      for(int i=1; i<=Count; i++)
        {
         if(BuyLimit)
           {

            if (StopLoss > 0)
                sl = NormalizeDouble(Bid - (StopLoss) * Point(), _Digits);
            if (TakeProfit > 0)
                tp = NormalizeDouble(Bid + (TakeProfit) * Point(), _Digits);

            pr=NormalizeDouble(Bid-(Delta+Step)*_Point*i,_Digits);
            trade.BuyLimit(Lots,pr,symb,sl, tp,0,0,"");
           }
         if(SellLimit)
           {

            if (StopLoss > 0)
                sl = NormalizeDouble(Bid + (_Point * StopLoss) * Point(), _Digits);
            if (TakeProfit > 0)
                tp = NormalizeDouble(Bid - (_Point * TakeProfit) * Point(), _Digits);

            pr=NormalizeDouble(Bid+(Delta+Step)*_Point*i,_Digits);
            trade.SellLimit(Lots,pr,symb,sl, tp,0,0,"");
           }

        }

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTimer()
  {
   DelOrder();
   Trade(Symbol1);
   Trade(Symbol2);
   Trade(Symbol3);
   Comment("\n All Profit: ",AllProfit(Symbol1),
           "\n Buy Profit: ",AllProfit(Symbol1,0),
           "\n Sell Profit: ",AllProfit(Symbol1,1));
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+--------+
```

This is the entire code of this simple EA.

### Test results

So, let’s launch the EA with default settings in the tester. Here are the EA results for EURUSD, GBPUSD, EURGBP, USDJPY and EURJPY from February 1, 2023 to February 18, 2024:

![EA test](https://c.mql5.com/2/76/2.png)

The drawdowns in relation to profits are very large. The drawdown on equity is generally greater than the annual profit. The EA behaves not much differently than regular grid EAs. Here are the test statistics:

![Test statistics](https://c.mql5.com/2/76/67104384_-_RoboForex-ECN_h0e2-yzi8_-_Hedge_-_RoboForex_Ltd_-_EURUSDnH1.png)

Apparently, this EA does not pay off its risks in any way. Like any algorithm without a stop level, it is a time bomb. Despite it shows no losses, no one can guarantee the market will not experience the collapse of currencies by 10-15% per day. Personally, the last four years have taught me that absolutely anything is possible on the market, and even the most incredible scenarios can come true, so a versatile EA must be prepared for anything. This EA does not meet my evaluation criteria, so I decided to publish it.

### Conclusion

So, we have created an example of the simplest market maker algorithm. Of course, this example is illustrative and very simple. Obviously, not a single market maker has worked like this in the market for a long time. Nowadays, their algorithms keep up with the times, use machine learning and neural networks, apply deep learning based on streaming data from the order book, and take into account many variables and price characteristics. Nobody places orders above and below the price anymore - this is fraught with inventory risk. In the future, it may be reasonable to experiment with creating a market maker using machine learning, which will determine the optimal delta between orders on its own.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/13897](https://www.mql5.com/ru/articles/13897)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13897.zip "Download all attachments in the single ZIP archive")

[Experts.zip](https://www.mql5.com/en/articles/download/13897/experts.zip "Download Experts.zip")(33.42 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/466128)**
(31)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
26 Jan 2024 at 16:46

Does the described algorithm apply only to [hedging accounts](https://www.mql5.com/en/articles/2299 "Article: Hedging Position Accounting Added to MetaTrader 5 ") or is there an option for netting?


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
5 Aug 2024 at 10:10

I assume that this is not a MM simulator.

Where viral market and limit orders of the crowd with different probabilities should be generated, and you all such MM - confirm or refute the myths about market makers by your interaction with this virtuality.

And thus you find out whether you are a "puppeteer" or not. So to speak - on your own, albeit virtual, experience.


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
5 Aug 2024 at 10:15

**JRandomTrader [#](https://www.mql5.com/ru/forum/460403/page3#comment_51721799):**

Once long ago I was a "market maker" in one of illiquid futures - up to a quarter of all deals were mine. Although there were some powerful bids from a real market maker, but far away from the actual prices.

How much did you bank?


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
5 Aug 2024 at 11:54

"Although it is not the market maker's job, they often nip Pump&Dump attempts in the bud when prices are just beginning to be driven up by fraudulent participants. In these initial phases, the market maker throws huge portions of limit orders at the player who is trying to "market" prices upwards. This extinguishes demand, and beginners in the business of pampa very often break their teeth against the market-maker. But if the pump is well planned and executed according to the plan - the influx of many market orders, powerfully moving the price, forces the market-maker to leave the market for a while."

And, here, what will prevent MM himself from organising Pump&Dump, as well as anything else, in order not to be, at least - in losses, and at most - in zero?! In those cases when he needs to bring the liquidity that they poured into the market to +.

After all, if your counterparty is MM, it means that he entered the market.

And if he has an imbalance of supply and demand in the flat, he compensates it with his liquidity for the size of the imbalance - he [enters](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "MetaTrader 5 Help: Opening and Closing Positions in the MetaTrader 5 Trading Terminal") the market.

And guess where the price will go - against MM, or in the direction of his orders?

And in general, if a company, or a bank, or a private person is a MM, then from what motivation of charity?!

Don't laugh at MM's slippers.

MM's function is to make the price attractive for a small speculator by narrowing the spread.

Further - to provide liquidity (which, in turn, reduces volatility), including for small speculator.

Further - price quantisation (price stabilisation) - throwing small speculators out of the market.

Strange as it may seem, "ejection of small speculator" is "price stabilisation". Well, just like in the example with flat and imbalance in it.

MM used, on the basis of "spread narrowing" - "price attractiveness", his liquidity in the flat in order to compensate the imbalance. Had he not used it, the price would have come out of the flat - would have been more volatile. In other words, he restrained volatility with his liquidity. Then he leads the price out of the flat, either with a false move and return to the flat, or with continuation - without return. That demolishes the stops of both those whose counteragent he was and those who stood against his orders. He leaves the market - with a profit, they - with a loss. He, at this stage, does not come out with a big profit, he is a scalper (at this stage). And his takes are the stops of his counterparties.

But he takes the price out of the flat boundaries not as sharply as it would go out without him, but with a return to the flat boundaries and bounces, or with a return to the flat....

Actually... MM organises the flat by injecting its liquidity and restraining the price movement))))))

But the thing is that he trades his liquidity with all types of players.

So, if narrow spreads and liquidity attract a small speculator, who is then thrown out of the market. Why do you need MM?!

What, in your opinion, is its purpose?


![Yevgeniy Koshtenko](https://c.mql5.com/avatar/2025/3/67e0f73d-60b1.jpg)

**[Yevgeniy Koshtenko](https://www.mql5.com/en/users/koshtenko)**
\|
7 Aug 2024 at 12:09

**Уроборос enters the [market.](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "MetaTrader 5 Help: Opening and Closing Positions in the MetaTrader 5 Trading Terminal")**

**And guess where the price will go - against MM, or in the direction of his orders?**

**And in general, if a company, or a bank, or a private person is a MM, then from what motivation of charity?!**

**Don't laugh at MM's slippers.**

**MM's function is to make the price attractive for a small speculator by narrowing the spread.**

**Further - to provide liquidity (which, in turn, reduces volatility), including for small speculator.**

**Further - price quantisation (price stabilisation) - throwing small speculators out of the market.**

**Strange as it may seem, "ejection of small speculator" is "price stabilisation". Well, just like in the example with flat and imbalance in it.**

**MM used, on the basis of "spread narrowing" - "price attractiveness", his liquidity in the flat in order to compensate the imbalance. Had he not used it, the price would have come out of the flat - would have been more volatile. In other words, he restrained volatility with his liquidity. Then he leads the price out of the flat, either with a false movement and return to the flat, or with continuation - without return. That demolishes the stops of both those whose counteragent he was and those who stood against his orders. He leaves the market - with a profit, they - with a loss. He, at this stage, does not come out with a big profit, he is a scalper, so to say (at this stage). And his takes are the stops of his counterparties.**

**But he takes the price out of the flat boundaries not as sharply as it would go out without him, but with a return to the flat boundaries and bounces, or with a return to the flat....**

**Actually... MM organises the flat by injecting its liquidity and restraining the price movement))))))**

**But the thing is that he trades his liquidity with all types of players.**

**So, if narrow spreads and liquidity attract a small speculator, who is then thrown out of the market. Why do you need MM?!**

**What, in your opinion, is its purpose?**

Yes, the price will go in his direction. But he doesn't need sharp pumps and dumps either. Try to be a MM yourself on DEX, and see how the capital sharply jerks on sharp pumps and dumps. It is much more profitable to have a stable flat and cut the spread, isn't it?

![Building A Candlestick Trend Constraint Model (Part 1): For EAs And Technical Indicators](https://c.mql5.com/2/76/Building_A_Candlestick_Trend_Constraint_Model_gPart_1v____LOGO.png)[Building A Candlestick Trend Constraint Model (Part 1): For EAs And Technical Indicators](https://www.mql5.com/en/articles/14347)

This article is aimed at beginners and pro-MQL5 developers. It provides a piece of code to define and constrain signal-generating indicators to trends in higher timeframes. In this way, traders can enhance their strategies by incorporating a broader market perspective, leading to potentially more robust and reliable trading signals.

![A Generic Optimization Formulation (GOF) to Implement Custom Max with Constraints](https://c.mql5.com/2/76/A_Generic_Optimization_Formulation_2GOFt_to_Implement_Custom_Max_with_Constraints____LOGO.png)[A Generic Optimization Formulation (GOF) to Implement Custom Max with Constraints](https://www.mql5.com/en/articles/14365)

In this article we will present a way to implement optimization problems with multiple objectives and constraints when selecting "Custom Max" in the Setting tab of the MetaTrader 5 terminal. As an example, the optimization problem could be: Maximize Profit Factor, Net Profit, and Recovery Factor, such that the Draw Down is less than 10%, the number of consecutive losses is less than 5, and the number of trades per week is more than 5.

![Developing a Replay System (Part 35): Making Adjustments (I)](https://c.mql5.com/2/60/Desenvolvendo_um_sistema_de_Replay_dParte_35a_Logo.png)[Developing a Replay System (Part 35): Making Adjustments (I)](https://www.mql5.com/en/articles/11492)

Before we can move forward, we need to fix a few things. These are not actually the necessary fixes but rather improvements to the way the class is managed and used. The reason is that failures occurred due to some interaction within the system. Despite attempts to find out the cause of such failures in order to eliminate them, all these attempts were unsuccessful. Some of these cases make no sense, for example, when we use pointers or recursion in C/C++, the program crashes.

![Population optimization algorithms: Bacterial Foraging Optimization - Genetic Algorithm (BFO-GA)](https://c.mql5.com/2/64/Bacterial_Foraging_Optimization_-_Genetic_Algorithmz_BFO-GA____LOGO.png)[Population optimization algorithms: Bacterial Foraging Optimization - Genetic Algorithm (BFO-GA)](https://www.mql5.com/en/articles/14011)

The article presents a new approach to solving optimization problems by combining ideas from bacterial foraging optimization (BFO) algorithms and techniques used in the genetic algorithm (GA) into a hybrid BFO-GA algorithm. It uses bacterial swarming to globally search for an optimal solution and genetic operators to refine local optima. Unlike the original BFO, bacteria can now mutate and inherit genes.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=getfeqwomydporzpxpvxkfibwwkpsbnf&ssn=1769092205577936941&ssn_dr=0&ssn_sr=0&fv_date=1769092205&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13897&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20market%20making%20algorithm%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909220518065449&fz_uniq=5049176043291518572&sv=2552)

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