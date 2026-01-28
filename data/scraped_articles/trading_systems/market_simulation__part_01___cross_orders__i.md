---
title: Market Simulation (Part 01): Cross Orders (I)
url: https://www.mql5.com/en/articles/12536
categories: Trading Systems, Strategy Tester
relevance_score: 0
scraped_at: 2026-01-24T13:46:40.431700
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/12536&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083101667840234767)

MetaTrader 5 / Tester


### Introduction

In the previous article " [Developing a Replay System (Part 78): New Chart Trade (V)](https://www.mql5.com/en/articles/12492), I showed how the Expert Advisor is able to interpret the instructions sent by Chart Trade. The information that Chart Trade actually transmits to the Expert Advisor depends on the user's interaction with it. In other words, when the user clicks on the buy, sell, or close position button, a message is sent to the chart. One of the Expert Advisor's tasks, when attached to this chart, is to intercept, decode, and execute the instructions contained in that message.

Although this mechanism is simple and quite reliable, we face a small issue. Well, it is not exactly a problem, more of an inconvenience. And this inconvenience needs to be resolved before we can actually start sending orders to the trading server.

If you are not familiar with what I mean, it may be because you don't trade certain assets, more specifically, futures contracts. These types of assets have an expiration date. Often, two types are traded simultaneously: the full contract, which has higher volume, and the mini contract, which can be seen as a fraction of the full one. The mini allows for strategies that require smaller volume or fewer contracts.

I won't go into the details of these strategies here. The point is that sometimes, to build a strategy, you need fewer contracts. Those interested in this should look into HEDGE strategies. For us, as programmers, the real concern is how to execute a trade in a mini contract when the chart displays the full contract.

But there's another challenge: long-term strategies. Every time a contract expires, which happens on a fixed, well-known date, a new series begins. For traders working with longer horizons, spanning multiple series, this creates a major issue. That's because indicators and moving averages must start their calculations all over again with each new series.

To better understand this, let's look at the B3 (Brazilian Stock Exchange) dollar futures contract. This contract expires monthly. That means every month a series ends, and a new one begins. Considering that each month has around 20 trading days (five trading days per week across four weeks), problems arise when trying to use, for example, a 20-period moving average. By the time the moving average is fully calculated and plotted, the contract expires and resets with a new series. And that's just with a 20-period average. Other indicators, which require even longer periods, suffer even more. In short, this is a big problem.

To overcome this, we can use the historical data of the futures contract. However, using historical data doesn't solve everything. It actually introduces new challenges for us programmers. Remember, the trader doesn't care how the server receives or plots the data. The trader only wants accurate information and reliable execution of their trades. It is up to us, as programmers, to solve the plotting issue and ensure that trade requests from the chart are properly routed to the trading server.

In the article [Developing a trading Expert Advisor from scratch (Part 11): Cross order system](https://www.mql5.com/en/articles/10383), I explained some details about this. But here, since we're also dealing with a replay/simulator, the problem becomes even bigger. Still, we can apply Napoleon's strategy: divide and conquer. By breaking the problem down into smaller parts, we can gradually develop a simulated order system. We'll start with one key aspect: handling indicators in futures contracts. And because the dollar futures contract is the most extreme case I know, we'll focus on it. Keep in mind, however, that the same principles apply to other contracts with similar complexities.

### Beginning the Implementation

In the article mentioned earlier, where we developed the cross order system, adapting it to cover other types of contracts was relatively complex. Here, for practical reasons, we'll take a different approach to make such adjustments simpler. not for the trader, but for us as programmers. The trader will need to adapt to our implementation. but in exchange, they will gain a straightforward option: choosing whether to trade the full contract or the mini.

To make this work (at least initially, when we're still communicating with the live trading server) we'll need to make a few targeted modifications to the existing code. Let's start with a key fact: as noted in the introduction, the best chart to use is the one based on historical data. But this historical chart cannot be traded directly.

To solve this, we need a system that routes the orders placed on the historical chart to the contract the trader actually wants to use. Remember, the trader may want to trade either the full contract or the mini. But we won't worry about that just yet. First, we must understand one simple fact: the content displayed on the chart comes from the historical data of the contract. Period.

On B3, futures contracts have six different naming conventions. This applies to every specific contract. That means six types for full contracts, and six types for minis. This is a seemingly huge complication. But on closer inspection, it's not as bad as it seems. Despite having six variations, they actually boil down to three main types, each with two sub-variations.

This simplification helps us a lot. Still, I recommend studying the differences between these three types. The chart data varies significantly between them, and many traders are completely unaware of this. If you're only programming the solution, make sure to inform the trader. If you're both programming and trading, take this advice even more seriously - you could run into serious problems if you don't understand these distinctions.

So, there are three naming types, each with two variations. Good. But for us programmers, what matters is not the names themselves, as that's more relevant to traders. For us, the key question is: Is there a rule in this naming convention? And if so, how can we use it to build a cross order system?

Fortunately, such a rule does exist. In fact, we've been using it for some time already. Take a look at how this works in the following code snippet.

```
38. //+------------------------------------------------------------------+
39.       void CurrentSymbol(void)
40.          {
41.             MqlDateTime mdt1;
42.             string sz0, sz1;
43.             datetime dt = macroGetDate(TimeCurrent(mdt1));
44.             enum eTypeSymbol {WIN, IND, WDO, DOL, OTHER} eTS = OTHER;
45.
46.             sz0 = StringSubstr(m_Infos.szSymbol = _Symbol, 0, 3);
47.             for (eTypeSymbol c0 = 0; (c0 < OTHER) && (eTS == OTHER); c0++) eTS = (EnumToString(c0) == sz0 ? c0 : eTS);
48.             switch (eTS)
49.             {
50.                case DOL   :
51.                case WDO   : sz1 = "FGHJKMNQUVXZ"; break;
52.                case IND   :
53.                case WIN   : sz1 = "GJMQVZ";       break;
54.                default    : return;
55.             }
56.             for (int i0 = 0, i1 = mdt1.year - 2000, imax = StringLen(sz1);; i0 = ((++i0) < imax ? i0 : 0), i1 += (i0 == 0 ? 1 : 0))
57.                if (dt < macroGetDate(SymbolInfoInteger(m_Infos.szSymbol = StringFormat("%s%s%d", sz0, StringSubstr(sz1, i0, 1), i1), SYMBOL_EXPIRATION_TIME))) break;
58.          }
59. //+------------------------------------------------------------------+
```

Code from the C\_Terminal.mqh file

This code snippet comes from the header file C\_Terminal.mqh. Notice that on line 44 we define the names of the futures contracts that will be supported. You may add other contracts to this list if you wish to work with assets such as corn, cattle, S&P, euro, and so on. Just remember to follow the naming rules for each contract to correctly identify the active one. The procedure described here does not return past contracts, nor does it return contracts two or more expiries ahead. It always resolves to the currently active contract.

To achieve this, line 46 extracts the first three characters of the asset name. Regardless of the asset, it will always capture these first three characters. This is because B3 (the Brazilian Stock Exchange) uses a naming convention where the first three characters identify the asset. After extraction, the asset name is saved in a variable to be used throughout the rest of the code. Please note this fact.

Next, line 47 iterates through the enumeration of contract names we defined. The purpose here is to find the correct match. That's why, in the enumeration at line 44, the names must resemble the contract names themselves. Since B3 uses uppercase characters, the enumeration must also be in uppercase. Once the match is found, or once the list is exhausted, the loop in line 47 ends.

At line 48 we can test the value retrieved. If no match is found, execution jumps to the code at line 54. Otherwise, we proceed to build the full contract name. The final naming takes place at line 57, where the procedure confirms that the contract name being generated corresponds to the currently active contract. In short, the procedure scans through the possible futures contracts until it locates the active one.

However, there's an important limitation here. The procedure uses the base contract name derived from the asset name. This means that you can only map historical data of a given contract to the active version of the same contract. You cannot, with the current code, map historical data from the full contract to the active mini contract. This is the inconvenience we will resolve in this article.

By doing so, we will give the trader the ability to choose whether to trade the full contract or the mini, even while using historical data from either one. Our goal is to make this possible with minimal code changes, because the more code we alter, the greater the chance of introducing errors.

To achieve this, the code snippet above was modified as follows:

```
38. //+------------------------------------------------------------------+
39.       void CurrentSymbol(bool bUsingFull)
40.          {
41.             MqlDateTime mdt1;
42.             string sz0, sz1;
43.             datetime dt = macroGetDate(TimeCurrent(mdt1));
44.             enum eTypeSymbol {WIN, IND, WDO, DOL, OTHER} eTS = OTHER;
45.
46.             sz0 = StringSubstr(m_Infos.szSymbol = _Symbol, 0, 3);
47.             for (eTypeSymbol c0 = 0; (c0 < OTHER) && (eTS == OTHER); c0++) eTS = (EnumToString(c0) == sz0 ? c0 : eTS);
48.             switch (eTS)
49.             {
50.                case DOL   :
51.                case WDO   : sz1 = "FGHJKMNQUVXZ"; break;
52.                case IND   :
53.                case WIN   : sz1 = "GJMQVZ";       break;
54.                default   : return;
55.             }
56.             sz0 = EnumToString((eTypeSymbol)(((eTS & 1) == 1) ? (bUsingFull ? eTS : eTS - 1) : (bUsingFull ? eTS + 1: eTS)));
57.             for (int i0 = 0, i1 = mdt1.year - 2000, imax = StringLen(sz1);; i0 = ((++i0) < imax ? i0 : 0), i1 += (i0 == 0 ? 1 : 0))
58.                if (dt < macroGetDate(SymbolInfoInteger(m_Infos.szSymbol = StringFormat("%s%s%d", sz0, StringSubstr(sz1, i0, 1), i1), SYMBOL_EXPIRATION_TIME))) break;
59.          }
60. //+------------------------------------------------------------------+
```

Code from the C\_Terminal.mqh file

As you can see, the changes were minimal. The first was to add a parameter to the function at line 39. This argument tells the procedure whether the contract name to be generated should be for the full contract or the mini. The choice is up to the trader. Our responsibility as programmers is to give them the flexibility to use whichever chart they prefer, as long as the charted asset data corresponds in some way to the contract being traded. Of course, we could implement more complex behaviors, but let's not overcomplicate what needs to be done.

Beyond this change at line 39, we added a new line. Technically, this addition wasn't strictly necessary as we could have modified the existing code. But it simplifies both the explanation and your understanding of what is happening. This new line, 56, could have been placed directly where the variable sz0 is used at line 58, and the result would be the same. However, that would make the explanation confusing and harder to follow.

So, what does line 56 do with the variable sz0? Let's break it down. Essentially, we are ignoring the asset name itself. Instead, we converte the enumeration defined at line 44 into a string. MQL5 allows this through the EnumToString function.

Now comes a detail that may complicate things if you are working with a futures contract that doesn't have both a full and a mini version. This is fairly common with **COMMODITIES**. But in the cases I want to demonstrate - indices and currencies, particularly the dollar futures - both contract types exist.

Enumerations always start with a value of zero unless you define otherwise. In our case, mini contracts are assigned even values, while full contracts are assigned odd values. Understanding this is important. The values are binary. You should know how to select this or that bit. In binary, the least significant bit (the rightmost one) determines whether a number is even or odd. By applying a bitwise AND to isolate this bit, we can check whether the enumeration value is even or odd. Once again: mini contracts are even, full contracts are odd.

So, if the value is odd, the first part of the ternary operator executes. If it's even, the second part executes. For now, I think everything is clear. Then, inside each branch of this first ternary operator, we use a second ternary operator. This second ternary operator allows us to adjust the variable **eTS**, ensuring it reflects the correct contract name.

For example: if the contract is WDO, then **eTS** equals 2, an even number. This triggers the second part of the first ternary operator. Inside it, the second ternary operator performs the second check. It checks whether the procedure call requested the full or the mini contract.

If the trader requested the full contract, **eTS** is incremented by one. Thus, its value changes from 2 to 3. In the enumeration, position 3 corresponds to DOL. When MQL5 executes EnumToString, the value 3 is converted to the string DOL, thus producing the full contract name, even though the chart is based on mini-dollar historical data.

The reverse works as well. If the chart shows the historical data of the full dollar contract, but the trader requests the mini, the first ternary operator executes its first branch. Inside it, the second ternary operator decrements **eTS** by one, changing its value from 3 to 2. This maps it back to WDO.

In short: the value found at line 47 is adjusted at line 56, so that the contract name matches the trader's choice (mini or full) while still relying on the historical chart of one of them.

So far, so good. But what if there's a contract with no mini version, only the full contract? How do we handle that? You might think there are two possible solutions. In reality, there's only one. If you try duplicating values in the enumeration to artificially create even and odd entries, the compiler will reject it. Instead, the solution is to structure the enumeration in a logical order. Then, when testing a particular value, if no mini contract exists, the variable sz0 remains unchanged. In practice, this requires an extra test in your code. But there's nothing complex there.

With this, we;ve solved the first part of the problem. However, we're not finished yet To continue, we need to adjust another part of the header file C\_Terminal.mqh: the class constructor. The constructor is responsible for calling the procedure we've just modified. So, the original must now be replaced with the new version shown below.

```
72. //+------------------------------------------------------------------+
73.       C_Terminal(const long id = 0, const uchar sub = 0, const bool bUsingFull = false)
74.          {
75.             m_Infos.ID = (id == 0 ? ChartID() : id);
76.             m_Mem.AccountLock = false;
77.             m_Infos.SubWin = (int) sub;
78.             CurrentSymbol(bUsingFull);
79.             m_Mem.Show_Descr = ChartGetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR);
80.             m_Mem.Show_Date  = ChartGetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE);
81.             ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, false);
82.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, true);
83.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, true);
84.             ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, false);
85.             m_Infos.nDigits = (int) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_DIGITS);
86.             m_Infos.Width   = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
87.             m_Infos.Height  = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
88.             m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
89.             m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
90.             m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
91.             m_Infos.AdjustToTrade = m_Infos.ValuePerPoint / m_Infos.PointPerTick;
92.             m_Infos.ChartMode   = (ENUM_SYMBOL_CHART_MODE) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_CHART_MODE);
93.             if(m_Infos.szSymbol != def_SymbolReplay) SetTypeAccount((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE));
94.             ChartChange();
95.          }
96. //+------------------------------------------------------------------+
```

Code from the C\_Terminal.mqh file

Notice that only two very simple changes were made. The first is at line 73, where we add a new parameter. This parameter is then used at line 78, where we call the procedure explained earlier. By default, I've set it to prioritize mini contracts. but the trader is free to choose whichever option suits their strategy best. A few small adjustments in specific parts of the code will be necessary to support this flexibility.

Since we're not yet modifying the Expert Advisor code, the required change must be made in the Chart Trade code. To keep things clear, let's cover this in a separate section.

### Turning Chart Trade into a Cross Order System

The changes needed for Chart Trade to work as a cross order system are quite simple. You might consider adding an object to the interface so that the trader can directly switch the cross order type. However, I will not take that approach here. My goal is to keep changes to a minimum. Adding such an object would require much more code just to support this feature. Instead, we can allow the trader to change the cross order type through the indicator settings. This approach is straightforward and involves very little modification to the existing code. The first step can be seen in the code below.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Chart Trade Base Indicator."
04. #property description "See the articles for more details."
05. #property version   "1.80"
06. #property icon "/Images/Market Replay/Icons/Indicators.ico"
07. #property link "https://www.mql5.com/pt/articles/12536"
08. #property indicator_chart_window
09. #property indicator_plots 0
10. //+------------------------------------------------------------------+
11. #include <Market Replay\Chart Trader\C_ChartFloatingRAD.mqh>
12. //+------------------------------------------------------------------+
13. #define def_ShortName "Indicator Chart Trade"
14. //+------------------------------------------------------------------+
15. C_ChartFloatingRAD *chart = NULL;
16. //+------------------------------------------------------------------+
17. enum eTypeContract {MINI, FULL};
18. //+------------------------------------------------------------------+
19. input ushort         user01 = 1;         //Leverage
20. input double         user02 = 100.1;     //Finance Take
21. input double         user03 = 75.4;      //Finance Stop
22. input eTypeContract  user04 = MINI;      //Cross order in contract
23. //+------------------------------------------------------------------+
24. int OnInit()
25. {
26.    chart = new C_ChartFloatingRAD(def_ShortName, new C_Mouse(0, "Indicator Mouse Study"), user01, user02, user03, (user04 == FULL));
27.
28.    if (_LastError >= ERR_USER_ERROR_FIRST) return INIT_FAILED;
29.
30.    return INIT_SUCCEEDED;
31. }
32. //+------------------------------------------------------------------+
33. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
34. {
35.    return rates_total;
36. }
37. //+------------------------------------------------------------------+
38. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
39. {
40.    if (_LastError < ERR_USER_ERROR_FIRST)
41.       (*chart).DispatchMessage(id, lparam, dparam, sparam);
42. }
43. //+------------------------------------------------------------------+
44. void OnDeinit(const int reason)
45. {
46.    switch (reason)
47.    {
48.       case REASON_INITFAILED:
49.          ChartIndicatorDelete(ChartID(), 0, def_ShortName);
50.          break;
51.       case REASON_CHARTCHANGE:
52.          (*chart).SaveState();
53.          break;
54.    }
55.
56.    delete chart;
57. }
58. //+------------------------------------------------------------------+
```

Chart Trade indicator source code

At line 17, we add an enumeration. It helps the trader (or user) define the contract type. Notice that this enumeration is used at line 22. This is the point where the trader decides whether the Expert Advisor should operate using the full contract or the mini. There is one drawback here: ideally, the selection should be made in the Expert Advisor, not in Chart Trade. But since Chart Trade and the Expert Advisor are still separate entities for now, we'll leave it this way.

The real challenge is not in Chart Trade or even in the Expert Advisor. As I explained in the previous article, Chart Trade can already control the Expert Advisor. The problem lies in another part of the system that will be developed later. That's where the real difficulty will appear, because ultimately everything must pass through the Expert Advisor. Ideally, we would need to provide everything necessary inside it. For now, though, and for demonstration purposes, we'll handle the selection in Chart Trade.

This configured value is then used at line 26. Notice that we are passing a boolean to the constructor, not a numeric value. Why? Because while a boolean may not seem as descriptive for the end user. It's very clear for us programmers. After all, there are only two possible conditions: the trader will either use the full contract or the mini. Therefore, a boolean is the most appropriate choice from a coding standpoint. The boolean is then passed to the class constructor. Let's see how this looks in the following code snippet.

```
213. //+------------------------------------------------------------------+
214.       C_ChartFloatingRAD(string szShortName, C_Mouse *MousePtr, const short Leverage, const double FinanceTake, const double FinanceStop, const bool bUsingFull)
215.          :C_Terminal(0, 0, bUsingFull)
216.          {
217.             m_Mouse = MousePtr;
218.             m_Info.IsSaveState = false;
219.             if (!IndicatorCheckPass(szShortName)) return;
220.             if (!RestoreState())
221.             {
222.                m_Info.Leverage = Leverage;
223.                m_Info.IsDayTrade = true;
224.                m_Info.FinanceTake = FinanceTake;
225.                m_Info.FinanceStop = FinanceStop;
226.                m_Info.IsMaximized = true;
227.                m_Info.minx = m_Info.x = 115;
228.                m_Info.miny = m_Info.y = 64;
229.             }
230.             CreateWindowRAD(170, 210);
231.             AdjustTemplate(true);
232.          }
233. //+------------------------------------------------------------------+
```

Fragment of the file C\_ChartFloatingRAD.mqh

Here, the change is just as simple as in the constructor of the C\_Terminal class. We only add a new parameter to be received by the constructor (line 214), and then pass it to the C\_Terminal constructor (line 215). That simple. Everything is straightforward and self-explanatory.

Still, we need one more small modification. This time, it's an addition to the class C\_ChartFloatingRAD. This change enables Chart Trade to communicate to the Expert Advisor what the trader actually intends to operate. The modification is shown in the snippet below.

```
330.       case MSG_BUY_MARKET:
331.          ev = evChartTradeBuy;
332.       case MSG_SELL_MARKET:
333.          ev = (ev != evChartTradeBuy ? evChartTradeSell : ev);
334.       case MSG_CLOSE_POSITION:
335.          if ((m_Info.IsMaximized) && (sz < 0))
336.          {
337.             string szTmp = StringFormat("%d?%s?%s?%c?%d?%.2f?%.2f", ev, _Symbol, GetInfoTerminal().szSymbol, (m_Info.IsDayTrade ? 'D' : 'S'),
338.                                         m_Info.Leverage, FinanceToPoints(m_Info.FinanceTake, m_Info.Leverage), FinanceToPoints(m_Info.FinanceStop, m_Info.Leverage));
339.             PrintFormat("Send %s - Args ( %s )", EnumToString((EnumEvents) ev), szTmp);
340.             EventChartCustom(GetInfoTerminal().ID, ev, 0, 0, szTmp);
341.          }
342.       break;
```

Fragment of the file C\_ChartFloatingRAD.mqh

This adjustment is so subtle it may go unnoticed. It occurs at line 337, where we add a new value to be sent to the Expert Advisor. This value tells the Expert Advisor which asset - or more precisely, which contract - is being displayed in Chart Trade. Keep in mind that this change will force another update in the Expert Advisor. But we'll deal with that later.

### Final Considerations

What we've done in this article demonstrates the flexibility of MQL5. But it also creates new challenges that we'll need to resolve later. Such things are far from simple and are not always resolved completely and easily. Implementing a cross order system that allows the Chart Trade user to tell the Expert Advisor that the charted asset is not necessarily the one being traded brings significant complexity. And I want to be clear: most of these issues are not caused by Chart Trade or the Expert Advisor themselves.

The real problem arises when we bring in elements that I haven't yet introduced, parts of the code that still need to be developed. Allowing the user to select the contract type within Chart Trade is not the best long-term solution. At least not in my view right now.

It's possible that future changes will make it practical and sustainable to configure this directly in Chart Trade. For the moment, though, I prefer to keep things simple enough to explain clearly. Building a personal solution quickly is easy; building one that others can understand and apply is more demanding. That's why you can expect future updates to this Chart Trade–Expert Advisor system. They will certainly come.

In the video below, you can see how the process looks directly on the chart.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12536](https://www.mql5.com/pt/articles/12536)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12536.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12536/anexo.zip "Download Anexo.zip")(490.53 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)
- [Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

**[Go to discussion](https://www.mql5.com/en/forum/494999)**

![Polynomial models in trading](https://c.mql5.com/2/109/Polynomial_models_in_trading___LOGO.png)[Polynomial models in trading](https://www.mql5.com/en/articles/16779)

This article is about orthogonal polynomials. Their use can become the basis for a more accurate and effective analysis of market information allowing traders to make more informed decisions.

![Big Bang - Big Crunch (BBBC) algorithm](https://c.mql5.com/2/108/16701-logo.png)[Big Bang - Big Crunch (BBBC) algorithm](https://www.mql5.com/en/articles/16701)

The article presents the Big Bang - Big Crunch method, which has two key phases: cyclic generation of random points and their compression to the optimal solution. This approach combines exploration and refinement, allowing us to gradually find better solutions and open up new optimization opportunities.

![Price Action Analysis Toolkit Development (Part 39): Automating BOS and ChoCH Detection in MQL5](https://c.mql5.com/2/168/19365-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 39): Automating BOS and ChoCH Detection in MQL5](https://www.mql5.com/en/articles/19365)

This article presents Fractal Reaction System, a compact MQL5 system that converts fractal pivots into actionable market-structure signals. Using closed-bar logic to avoid repainting, the EA detects Change-of-Character (ChoCH) warnings and confirms Breaks-of-Structure (BOS), draws persistent chart objects, and logs/alerts every confirmed event (desktop, mobile and sound). Read on for the algorithm design, implementation notes, testing results and the full EA code so you can compile, test and deploy the detector yourself.

![Neural Networks in Trading: An Ensemble of Agents with Attention Mechanisms (MASAAT)](https://c.mql5.com/2/105/logo-neural-networks-made-easy-masaat.png)[Neural Networks in Trading: An Ensemble of Agents with Attention Mechanisms (MASAAT)](https://www.mql5.com/en/articles/16599)

We introduce the Multi-Agent Self-Adaptive Portfolio Optimization Framework (MASAAT), which combines attention mechanisms and time series analysis. MASAAT generates a set of agents that analyze price series and directional changes, enabling the identification of significant fluctuations in asset prices at different levels of detail.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=olaznijqdeuiydhmnmnoyyxadcsgbfjc&ssn=1769251599504395814&ssn_dr=0&ssn_sr=0&fv_date=1769251599&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12536&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Market%20Simulation%20(Part%2001)%3A%20Cross%20Orders%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925159955275316&fz_uniq=5083101667840234767&sv=2552)

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