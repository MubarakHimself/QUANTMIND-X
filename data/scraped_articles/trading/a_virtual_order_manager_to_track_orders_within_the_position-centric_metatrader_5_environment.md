---
title: A Virtual Order Manager to track orders within the position-centric MetaTrader 5 environment
url: https://www.mql5.com/en/articles/88
categories: Trading, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:23:47.025938
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=faqeiiiajxhmxtjcuyoeibgedqnwcrpb&ssn=1769181825033622449&ssn_dr=0&ssn_sr=0&fv_date=1769181825&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F88&back_ref=https%3A%2F%2Fwww.google.com%2F&title=A%20Virtual%20Order%20Manager%20to%20track%20orders%20within%20the%20position-centric%20MetaTrader%205%20environment%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918182595931845&fz_uniq=5069402899714344065&sv=2552)

MetaTrader 5 / Trading


### 1\. Introduction

Arguably the biggest change in the transition from [MetaTrader 4](https://www.mql5.com/go?link=http://www.metatrade4.com/ "http://www.metatrade4.com/") to [MetaTrader 5](https://www.metatrader5.com/) is the management of open trades as _positions_. At any one time there can be one position only open for each symbol, and the size of this position adjusts up and down each time orders are processed by the broker. This aligns with the [NFA 2-43(b) FIFO rule](https://www.mql5.com/go?link=http://www.nfa.futures.org/NFA-faqs/compliance-faqs/compliance-rule-2-43-QA.HTML "http://www.nfa.futures.org/NFA-faqs/compliance-faqs/compliance-rule-2-43-QA.HTML") introduced in the US, and also fits with the mode of trading in many other entities such as futures, commodities and CFDs.

A clear example of the difference would be when two EAs running against the same symbol issue orders in opposite directions.This can be a common situation with two EAs working in different timeframes, such as a scalper and a trend-follower.In MetaTrader 4, the open trade list would show buy and sell open orders with zero margin used. In MetaTrader 5, no position would be open at all.

Looking in the EA code itself, functions such as the commonly used MQL4 OpenOrders() below, or similar variant, will not function as expected when migrated to [MQL5](https://www.mql5.com/en/docs).

```
int OpenOrders()  // MetaTrader 4 code to count total open orders for this EA
{
  int nOpenOrders=0;
  for (int i=OrdersTotal()-1; i>=0; i--)
  {
    OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
    if (OrderMagicNumber()==magic)
      if (OrderType()==OP_BUY || OrderType()==OP_SELL)
        if (OrderSymbol()==Symbol())
          nOpenOrders++;
  }
  return(nOpenOrders);
}
```

So the MetaTrader 5 position-centric environment presents unfamiliar challenges for the programmer used to the order processing approach used in MetaTrader 4. What were simple order management functions in MetaTrader 4 become more complex in MetaTrader 5 when multiple orders can get merged into the one position, such as multiple EAs trading the one symbol, or multiple orders from the one EA on a symbol.

### **2\. Ways to work with positions in MetaTrader 5**

There are a number of ways to manage this position-centric environment in MetaTrader 5, depending on the complexity of trading strategies.

Firstly, note that MetaTrader 5’s handling of _pending_ orders is similar to MetaTrader 4, so MQL5 code written for pending orders alone could be a relatively simple migration from MQL4 code.

> **2.1 Straightforward EA; one EA per symbol per account**
>
> The simplest approach is to limit trading on the one account to one straightforward EA per symbol.“Straightforward EA” in this case means one which only issues a single order at a time, which is a common method but excludes strategies such as pyramiding and grid trading.Straightforward EAs can be written in MQL5 in a similar way to MQL4, perhaps using the CTrade library wrapper provided in include\\trade\\trade.mqh.
>
> **2.2 Complex EA; one EA per symbol per account**
>
> For complex EAs, such as those which have a strategy such as pyramiding or grid trading which can require more than one open order for a symbol, some relatively simple order tracking code added to the EA may be all that is necessary to manage the strategy.This will only be possible if the EA will never share positions with another EA trading the same symbol.
>
> **2.3 More than one EA of any type per symbol per account**
>
> This presents the most complex trading and coding requirement, and is the reason for the development of the Virtual Order Manager (VOM) library.This library is intended to simplify greatly the development of robust EA code which is fully sociable with other EAs.

_**The rest of this article describes the Virtual Order Manager library in detail.**_

### 3\. Design goals, benefits and disadvantages of the Virtual Order Manager

The VOM has four main design goals:

1. **Sociability**: the behaviour of EAs written correctly using the VOM trading functions will be isolated from other EA activity
2. **Robustness**: elegant handling of abnormal events such as errors, breaks in client-server communication and incomplete order fills.
3. **Ease of use**: provision of well documented and simple trading functions
4. **Ability to use in the Strategy Tester**

These goals are implemented as follows:

- Use of virtual open orders, pending orders, stoplosses and takeprofits.“Virtual” in this context means that their status is maintained at the client terminal independently of positions at the server.These orders have horizontal lines drawn on the terminal in a similar fashion to positions
- A protective server-based stop maintained a distance away from the virtual stops for disaster protection in the event of PC or internet link failure

The VOM approach allows an MQL5 EA programmer to:

- Code EAs in an “order-centric” fashion, ie similar to the MetaTrader 4 approach
- Implement what many in the Metatrader community refer to as “hedge trading” or, more accurately, simultaneous trades in the opposite direction against a single symbol
- Code other advanced trading strategies relatively easily such as grid trading, pyramiding and money management approaches
- Issue stops and pending orders tighter than the minimum stop level

It should also be noted that a side-effect of the VOM approach is that its virtual stoplosses, takeprofits and pending orders inherently have “stealth” behaviour, ie they cannot be seen at the broker server.Hiding stoploss levels is seen by some as necessary to prevent the broker from being able to engage in stop-hunting.

The VOM also has disadvantages.The amount of equity risk is increased due to the possibility of relying on the more distant protective server stop during an extended PC or internet link failure.Also, slippage when hitting a virtual pending order, stoploss or takeprofit could be much higher than for its server-based equivalent during times of high volatility such as news events.The impact of these disadvantages can be minimised if VOM EAs are traded from a high reliability virtual desktop with a short ping time to the broker’s server.

### 4\. The VOM in practice – a simple EA

Before going further, it’s time to show how a VOM EA can be written.We’ll write a simple MA cross EA, starting with the template EA provided in the distribution package.  We will use the Fractal Moving Average, which has the potential to reduce pointless trades during sideways markets, a notorious problem with MA cross strategies. It should be stressed that this EA has been provided as a simple example and is not recommended for live trading – the backtest is profitable but the low number of trades means that the result is not statistically significant.

The EA is stored in experts\\Virtual Order Manager\\VOM EAs.

```
//+------------------------------------------------------------------+
//|                                           FraMA Cross EA VOM.mq5 |
//+------------------------------------------------------------------+
#property copyright "Paul Hampton-Smith"
#property link      "http://paulsfxrandomwalk.blogspot.com"
#property version   "1.00"

// this is the only include required.  It points to the parent folder
#include "..\VirtualOrderManager.mqh"

input double   Lots=0.1;
input int      Fast_MA_Period=2;
input int      Slow_MA_Period=58;
/*
Because the broker is 3/5 digit, stoplosses and takeprofits should be x10.
It seems likely that all brokers offering MetaTrader 5 will be 3/5 digit brokers,
but if this turns out to be incorrect it will not be a major task to add
digit size detection. */
input int      Stop_Loss=5000;
input int      Take_Profit=0;
/*
We can also change the level of logging.  LOG_VERBOSE is the most prolific
log level.  Once an EA has been fully debugged the level can be reduced to
LOG_MAJOR.  Log files are written under the files\EAlogs folder and are
automatically deleted after 30 days.  */
input ENUM_LOG_LEVEL Log_Level=LOG_VERBOSE;

// The following global variables will store the handles and values for the MAs
double g_FastFrAMA[];
double g_SlowFrAMA[];
int g_hFastFrAMA;
int g_hSlowFrAMA;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   LogFile.LogLevel(Log_Level);

// Need to include this line in all EAs using CVirtualOrderManager
   VOM.Initialise();
   Comment(VOM.m_OpenOrders.SummaryList());

   g_hFastFrAMA = iFrAMA(_Symbol,_Period,Fast_MA_Period,0,PRICE_CLOSE);
   g_hSlowFrAMA = iFrAMA(_Symbol,_Period,Slow_MA_Period,0,PRICE_CLOSE);
   ArraySetAsSeries(g_FastFrAMA,true);
   ArraySetAsSeries(g_SlowFrAMA,true);

   return(0);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Need to include this line in all EAs using CVirtualOrderManager
   VOM.OnTick();
   Comment(VOM.m_OpenOrders.SummaryList());

// We now obtain copies of the most recent two FrAMA values in the
// g_FastFrAMA and  g_SlowFrAMA arrays.
   if(CopyBuffer(g_hFastFrAMA,0,Shift,2,g_FastFrAMA)!=2) ||
      CopyBuffer(g_hSlowFrAMA,0,Shift,2,g_SlowFrAMA)!=2)
     {
      Print("Not enough history loaded");
      return;
     }

// And now we detect a cross of the fast FrAMA over the slow FrAMA,
// close any opposite orders and Buy a single new one
   if(g_FastFrAMA[0]>g_SlowFrAMA[0] && g_FastFrAMA[1]<=g_SlowFrAMA[1])
     {
      VOM.CloseAllOrders(_Symbol,VIRTUAL_ORDER_TYPE_SELL);
      if(VOM.OpenedOrdersInSameBar()<1 && VOM.OpenOrders()==0)
        {
         VOM.Buy(_Symbol,Lots,Stop_Loss,Take_Profit);
        }
     }

// Opposite for Sell
   if(g_FastFrAMA[0]<g_SlowFrAMA[0] && g_FastFrAMA[1]>=g_SlowFrAMA[1])
     {
      VOM.CloseAllOrders(_Symbol,VIRTUAL_ORDER_TYPE_BUY);
      if(VOM.OpenedOrdersInSameBar()<1 && VOM.OpenOrders()==0)
        {
         VOM.Sell(_Symbol,Lots,Stop_Loss,Take_Profit);
        }
     }
  }
//+------------------------------------------------------------------+
```

And now with the release of the Strategy Tester it can be backtested, see Figure 1 below:

![ Figure 1. FrAMA Cross EA backtest](https://c.mql5.com/2/1/tester_.png)

Figure 1. FrAMA Cross EA backtest

The logging section is shown at Figure 2:

![ Figure 2. Strategy test log](https://c.mql5.com/2/1/tester2_.png)

Figure 2. Strategy test log

### 5\. VOM structure

Figure 4 below shows how multiple VOM EAs are configured:

![Figure 3. Multiple VOM EAs](https://c.mql5.com/2/1/v1.png)

Figure 3. Multiple VOM EAs

Then looking inside the VOM, the main components are shown at Figure 4 below:

![Figure 4. VOM internal structure](https://c.mql5.com/2/1/v2.png)

Figure 4. VOM internal structure

**Elements of firgure 4 explained:**

- **Configuration** \- the VOM uses CConfig to store all the main configuration items in one place in a global object Config. To make access simple the member variables are public and no get/set functions are provided.
- **Global variable** s - these are the variables accessed in MQL5 by functions such as GlobalVariableGet(). The VOM uses the global variables to

  - Record and increment the last Virtual Order ticket number using CGlobalVariable
  - Maintain a list of all virtual stoplosses so that disaster protection server stops can be maintained

- **Open trades and history files** \- these are the permanent disk files stored by CVirtualOrderArrays to ensure that order status can be re-established on restart.A pair of these files is created and stored in Files\\\VOM for each EA that uses the VOM.A CVirtualOrder starts life in the VOM.m\_OpenOrders array, and is transferred to the VOM.m\_OrderHistory array when closed or deleted.
- **Activity and debug log** \- most code of any complexity needs the ability to log activity, and this function is encapsulated by the CLog class. This enables logging to be recorded at four different levels of detail and importance, and includes automatic cleanup of old log files to ensure that diskspace is managed.

Expert Advisors that use the VOM interact with the library as shown in Figure 5 below:

![ Figure 5. EA interaction with the VOM library](https://c.mql5.com/2/1/VOM.png)

Figure 5. EA interaction with the VOM library

### 6\. More on the disaster protection stoploss

Virtual stops have been quite common amongst MetaTrader 4 EAs.If a stoploss is maintained at the client end only, the exit level for a trade is invisible to the broker, a strategy often implemented in the belief that some brokers engage in stop hunting.On their own, virtual stops greatly increase trade risk, since a broker-client connection must be always be in place for the stop to be actioned.

The VOM controls this risk by maintaining a server-based stop at a configurable distance away from the tightest virtual stop.This is termed a disaster protection stoploss (DPSL) because it will normally only be actioned if the broker-client connection is broken for some time, as would be the situation with an internet connection break or a PC failure.As virtual orders are opened and closed, and converted to a position at the server, the maintenance of the DPSL at the correct level can be a little complex, as illustrated in the following sequence.

| Virtual order<br>action | Open <br>price | Virtual SL | Position<br>at server | Stoploss<br>at server | Comment |
| --- | --- | --- | --- | --- | --- |
| 0.1 lots BUY #1 | 2.00000 | 1.99000 | 0.1 lots BUY | 1.98500 | DPSL is 50 pips below virtual SL #1 |
| 0.1 lots BUY #2 | 2.00000 | 1.99500 | 0.2 lots BUY | 1.99000 | Virtual order #2 has a tighter SL so DPSL <br>is tightened to 50 pips below virtual SL #2 |
| Close #2 |  |  | 0.1 lots BUY | 1.98500 | Revert to looser DPSL |
| 0.1 lots SELL #3 | 2.00000 | 2.00500 | none | none | Virtual orders #1 and #3 have cancelled each <br>other out at the server |
| Close #1 |  |  | 0.1 lots SELL | 2.01000 | Virtual Order #3 remains open - DPSL is <br>now 50 pips above virtual SL #3 |

### 7\. Testing the Virtual Order Manager

A project of this size takes time to test thoroughly, so I wrote the EA VirtualOrderManaerTester.mq5 to enable virtual orders to be created, modifed, deleted and closed easily with command buttons on the chart.

Figure 6 below shows a virtual buy order at 0.1 lot in the M5 window and a virtual buy order of another 0.1 lot open in the H4 window against EURUSD (see comment lines), with the server status correctly showing one position at 0.2 lots bought. Because the overall position is long, the Distaster Protection Stoploss can been seen below the tighter 20.0 pip stop.

![Figure 6. Two EAs agreeing on direction](https://c.mql5.com/2/1/fig6__3.png)

Figure 6. Two EAs agreeing on direction

Figure 7 now shows the two test EAs with opposing virtual orders, and
no position is open at the broker:

![Figure 7. Two EAs with opposing virtual orders and no position is open at the broker](https://c.mql5.com/2/1/fig7.png)

Figure 7. Two EAs with opposing virtual orders and no position is open at the broker

### 8\. A very simple display of all VOM open orders

Each VOM EA can only see its own orders, so I have written a very simple EA which collates the open orders from all VOMs.  The display is very simple, and when time permits a much better version could be written, perhaps with command buttons to perform modify, delete or close actions as required on each order.  The EA is included in the distribution pack as VOM\_OrderDisplay.mq5.

### 9\. Conclusion

At the time of writing this article, the VOM code is in Beta, just like MetaTrader 5 itself, and time will tell if the VOM concept becomes popular or ends up being regarded as just an interesting piece of MQL5 programming.

Let's go back to the design goals in section 3 and see where we have arrived

1. **Sociability**: the behaviour of EAs written correctly using the VOM trading functions will be isolated from other EA activity

   - **Result** \- yes, the VOM approach achieved this goal

3. **Robustness**: elegant handling of abnormal events such as errors, breaks in client-server communication and incomplete order fills.

   - **Result** \- some robustness is evident but there could be improvement as real trading situations occur and can be analysed

5. **Ease of use**: provision of well documented and simple trading functions

   - **Result** \- As will be seen in the distribution pack of files, a .chm help file is included

7. **Ability to use in the Strategy Tester**
   - ******Result**** -** initial tests in the recently released strategy tester indicate that the VOM does backtest correctly, although the VOM approach slows down the test considerably.  Some work to improve throughput is probably needed

A number of future changes may be desirable

- As with any complex software development, it is likely that there are bugs remaining in the code
- With each MetaTrader 5 Beta build release there may be required VOM changes to maintain compatibility
- VomGetLastError() and VomErrorDescription() functions
- Ability to read configuration from a file
- Trailing stops of various types

### 10\. Files in the zipped distribution pack

The VOM package comes as a number of .mqh files which should be installed in an Experts\\Virtual Order Manager folder,

- ChartObjectsTradeLines.mqh - CEntryPriceLine, CStopLossLine, CTakeProfitLine
- StringUtilities.mqh - global enum descriptors such as ErrorDescription()
- Log.mqh - CLog
- GlobalVirtualStopList.mqh - CGlobalVirtualStopList
- SimpleChartObject.mqh - CButton, CLabel and CEdit
- VirtualOrder.mqh - CVirtualOrder
- GlobalVariable.mqh - CGlobalVariable
- VirtualOrderArray.mqh - CVirtualOrderArray
- **VirtualOrderManager.mqh - CVirtualOrderManager**
- VirtualOrderManagerConfig.mqh - CConfig
- VirtualOrderManagerEnums.mqh - the various enums defined for the VOM
- VOM\_manual.mqh - this page of the manual
- VOM\_doc.chm\*\*\*

Five EA mq5 files are also included under Experts\\Virtual Order Manager\\VOM EAs:

- VOM\_template\_EA.mq5 - clone this to make your own EAs, and store them in Experts\\Virtual Order Manage\\VOM EAs
- VirtualOrderManagerTester.mq5
- Support\_Resistance\_EA\_VOM.mq5
- FrAMA\_Cross\_EA\_VOM.mq5
- VOM\_OrderDisplay.mq5

\*\*\*Note that the VOM\_doc.chm file may need to be unlocked:

![](https://c.mql5.com/2/1/CHM.png)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/88.zip "Download all attachments in the single ZIP archive")

[vom-doc.zip](https://www.mql5.com/en/articles/download/88/vom-doc.zip "Download vom-doc.zip")(727.63 KB)

[vom2\_0.zip](https://www.mql5.com/en/articles/download/88/vom2_0.zip "Download vom2_0.zip")(608.43 KB)

[vom-sources.zip](https://www.mql5.com/en/articles/download/88/vom-sources.zip "Download vom-sources.zip")(40.33 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Auto-Generated Documentation for MQL5 Code](https://www.mql5.com/en/articles/12)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/978)**
(45)


![kenshin71](https://c.mql5.com/avatar/avatar_na2.png)

**[kenshin71](https://www.mql5.com/en/users/kenshin71)**
\|
31 Jan 2018 at 03:18

**Alain Verleyen:**

This library is now mostly obsolete as MT5 is providing hedging account. You can still need it on netting account, but is it really worth it.

My ea is an expanding grid type one operating on one symbol, so I need to track each trade or grid level individually so that I know where to open the next level.  On MT4 I have been doing this by looking at the open trade list and using them to calculate the position where I need to open the next trade in the grid.  On MT5 I find it's dumping all the trades in one big position which completely kills that method.  For example, I can't have two buy trades open in MT5, because it combines them into one position.  How can I calculate where level six of a buy grid is going to be if I don't know where levels one through to five are?  I hope you can see what I mean.  This library was the only way I've found to get around that problem.  On MT4 I am also using unique magic numbers on each trade to identify each grid level. On MT5 I have found that I can't set unique magic numbers either, so I'm going to store them on disk instead.  If there is a better way to manage grid levels like this on MT5, I am eager to learn.

The ea is around 6000 lines long, so I'm not too keen to rewrite it.  I've spent the better part of 2 weeks getting it to compile properly on MT5, and I'm sure it will run fine once I change to storing magic numbers on disk instead of the broker server.

On a positive note, I did find out how to fix the errors in the VOM library, so hopefully this helps someone.  Using version 2 of the library, you need to change line 855 of the virtualordermanager.mqh file from :

```
MtRequest.type_filling=ORDER_FILLING_FOK;
```

to

```
MtRequest.type_filling=SYMBOL_FILLING_FOK;
```

then it will open trades properly with no errors.

![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
31 Jan 2018 at 05:06

**kenshin71:**

My ea is an expanding grid type one operating on one symbol, so I need to track each trade or grid level individually so that I know where to open the next level.  On MT4 I have been doing this by looking at the open trade list and using them to calculate the position where I need to open the next trade in the grid.  On MT5 I find it's dumping all the trades in one big position which completely kills that method.  For example, I can't have two buy trades open in MT5, because it combines them into one position.  How can I calculate where level six of a buy grid is going to be if I don't know where levels one through to five are?  I hope you can see what I mean.  This library was the only way I've found to get around that problem.  On MT4 I am also using unique magic numbers on each trade to identify each grid level. On MT5 I have found that I can't set unique magic numbers either, so I'm going to store them on disk instead.  If there is a better way to manage grid levels like this on MT5, I am eager to learn.

The ea is around 6000 lines long, so I'm not too keen to rewrite it.  I've spent the better part of 2 weeks getting it to compile properly on MT5, and I'm sure it will run fine once I change to storing magic numbers on disk instead of the broker server.

On a positive note, I did find out how to fix the errors in the VOM library, so hopefully this helps someone.  Using version 2 of the library, you need to change line 855 of the virtualordermanager.mqh file from :

to

then it will open trades properly with no errors.

That's why I said you there is now [hedging account](https://www.mql5.com/en/articles/2299), which you should use as you are trading on Forex apparently. The account you are describing is a netting account.


![kenshin71](https://c.mql5.com/avatar/avatar_na2.png)

**[kenshin71](https://www.mql5.com/en/users/kenshin71)**
\|
31 Jan 2018 at 13:43

Thankyou Alain.  I see what you mean now.  I apologise for not paying closer attention to what you said about hedging first.  It was driving me crazy trying to get this VOM working as well as all the other changes required to get my ea working under MT5.  I think I'll take a break from it and come back when I've had some sleep.  I do think I'll need write an MT5 specific version of the ea from the ground up though.  Up till now I have been trying to make a version that works under MT4 and MT5 using conditional compiling, but that idea is not really working out as cleanly as I'd like.  Thanks again for pointing me in the right direction.


![Nikolai Karetnikov](https://c.mql5.com/avatar/2013/4/517CE93C-6F64.jpg)

**[Nikolai Karetnikov](https://www.mql5.com/en/users/ns_k)**
\|
1 Jul 2020 at 23:29

**Alain Verleyen:**

This library is now mostly obsolete as MT5 is providing hedging account. You can still need it on netting account, but is it really worth it.

yeap ) such a common thing in SW development

![Faisal Mahmood](https://c.mql5.com/avatar/2022/2/621BC4C1-E102.jpeg)

**[Faisal Mahmood](https://www.mql5.com/en/users/xbotuk)**
\|
26 Nov 2020 at 12:07

**kenshin71:**

Thankyou Alain.  I see what you mean now.  I apologise for not paying closer attention to what you said about hedging first.  It was driving me crazy trying to get this VOM working as well as all the other changes required to get my ea working under MT5.  I think I'll take a break from it and come back when I've had some sleep.  I do think I'll need write an MT5 specific version of the ea from the ground up though.  Up till now I have been trying to make a version that works under MT4 and MT5 using conditional compiling, but that idea is not really working out as cleanly as I'd like.  Thanks again for pointing me in the right direction.

Did you get this to work? I would like to take this forward from where you left off if its ok to share the fixes you had to do. Unfortunately I need to use netting MT5 account and need to use VOM for virtual hedging.


![Migrating from MQL4 to MQL5](https://c.mql5.com/2/0/logo__4.png)[Migrating from MQL4 to MQL5](https://www.mql5.com/en/articles/81)

This article is a quick guide to MQL4 language functions, it will help you to migrate your programs from MQL4 to MQL5. For each MQL4 function (except trading functions) the description and MQL5 implementation are presented, it allows you to reduce the conversion time significantly. For convenience, the MQL4 functions are divided into groups, similar to MQL4 Reference.

![Creating an Indicator with Graphical Control Options](https://c.mql5.com/2/0/macd__1.png)[Creating an Indicator with Graphical Control Options](https://www.mql5.com/en/articles/42)

Those who are familiar with market sentiments, know the MACD indicator (its full name is Moving Average Convergence/Divergence) - the powerful tool for analyzing the price movement, used by traders from the very first moments of appearance of the computer analysis methods. In this article we'll consider possible modifications of MACD and implement them in one indicator with the possibility to graphically switch between the modifications.

![Practical Application Of Databases For Markets Analysis](https://c.mql5.com/2/0/dar.png)[Practical Application Of Databases For Markets Analysis](https://www.mql5.com/en/articles/69)

Working with data has become the main task for modern software - both for standalone and network applications. To solve this problem a specialized software were created. These are Database Management Systems (DBMS), that can structure, systematize and organize data for their computer storage and processing. As for trading, the most of analysts don't use databases in their work. But there are tasks, where such a solution would have to be handy. This article provides an example of indicators, that can save and load data from databases both with client-server and file-server architectures.

![MetaTrader 5: Publishing trading forecasts and live trading statements via e-mail on blogs, social networks and dedicated websites](https://c.mql5.com/2/0/social-network.png)[MetaTrader 5: Publishing trading forecasts and live trading statements via e-mail on blogs, social networks and dedicated websites](https://www.mql5.com/en/articles/80)

This article aims to present ready-made solutions for publishing forecasts using MetaTrader 5. It covers a range of ideas: from using dedicated websites for publishing MetaTrader statements, through setting up one's own website with virtually no web programming experience needed and finally integration with a social network microblogging service that allows many readers to join and follow the forecasts. All solutions presented here are 100% free and possible to setup by anyone with a basic knowledge of e-mail and ftp services. There are no obstacles to use the same techniques for professional hosting and commercial trading forecast services.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=wxuwyykrqmpyufdfcaobopkbfbslmvbk&ssn=1769181825033622449&ssn_dr=0&ssn_sr=0&fv_date=1769181825&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F88&back_ref=https%3A%2F%2Fwww.google.com%2F&title=A%20Virtual%20Order%20Manager%20to%20track%20orders%20within%20the%20position-centric%20MetaTrader%205%20environment%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918182595997921&fz_uniq=5069402899714344065&sv=2552)

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