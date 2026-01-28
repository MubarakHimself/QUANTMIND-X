---
title: How to copy signals using an EA by your rules?
url: https://www.mql5.com/en/articles/2438
categories: Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:44:15.641922
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/2438&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5072056240315511678)

MetaTrader 5 / Examples


### Table Of Contents

- [Risk Warning](https://www.mql5.com/en/articles/2438#intro)
- [1\. Preparation](https://www.mql5.com/en/articles/2438#chapter1)
- [1.1. The idea of the copier](https://www.mql5.com/en/articles/2438#chapter1)
- [2\. Features of the Signals Service](https://www.mql5.com/en/articles/2438#chapter2)

  - [2.1. Friend or foe](https://www.mql5.com/en/articles/2438#chapter2_1)
  - [2.2. Substitute the Magic Number or not?](https://www.mql5.com/en/articles/2438#chapter2_2)
  - [2.3. What happens to the "foe" position, if provider closes its position?](https://www.mql5.com/en/articles/2438#chapter2_3)

- [3\. How to detect emergence of a deal](https://www.mql5.com/en/articles/2438#chapter3)

  - [3.1. Tracking using the OnTradeTransaction()](https://www.mql5.com/en/articles/2438#chapter3_1)

- [4. When subscription is possible and when it is not](https://www.mql5.com/en/articles/2438#chapter4)
- [5\. Store information about copying](https://www.mql5.com/en/articles/2438#chapter5)
- [Conclusion](https://www.mql5.com/en/articles/2438#exit)

### Risk Warning

Before using this method, you need to rely primarily on common sense, as the increase in the copy ratio entails increased risks.

While copying, the system periodically checks whether positions on the subscriber's account match the provider's ones. If a mismatch is detected (i.e., only some positions have been copied), the system tries to eliminate that and copy the missing ones. Unlike the initial synchronization, the total floating profit of the provider is not checked. If the subscriber started copying, they should follow the provider's trading strategy to the maximum possible extent. It is impossible to copy some positions, while ignoring others.

If the system detects positions not based on signals on the subscriber's account, it offers to close them or closes them automatically depending on the ["Synchronize positions without confirmations"](https://www.metatrader5.com/en/terminal/help/signals/signal_subscriber#settings "https://www.metatrader5.com/en/terminal/help/signals/signal_subscriber#settings") setting.

The method for increasing the copy ratio of the signal can be used when the provider trades minimum lot, the deposit load is set to 95% in the settings (maximum utilization of the deposit) and at the same time the trade copy ratio is still too small for a subscription to be meaningful. You can learn the future trade copy rate before subscribing to a signal you like with the help of the article [Calculator of signals](https://www.mql5.com/en/articles/2329).

The assistant expert for copying trading signals presented in this article can be downloaded from the Market for free:

- [Signals Copier for MetaTrader 5](https://www.mql5.com/en/market/product/16813)

### 1\. Preparation

Before you start, please see the tutorial videos:

1. [Selecting a trading signal](https://www.youtube.com/watch?v=ntu6pZRopq4&list=PLltlMLQ7OLeLZpxDnCMKz1tBlPcUminCT&index=1 "https://www.youtube.com/watch?v=ntu6pZRopq4&amp;list=PLltlMLQ7OLeLZpxDnCMKz1tBlPcUminCT&amp;index=1")
2. [Subscription to Signal](https://www.youtube.com/watch?v=r99S48RiKeA&index=6&list=PLltlMLQ7OLeLZpxDnCMKz1tBlPcUminCT "https://www.youtube.com/watch?v=r99S48RiKeA&amp;list=PLltlMLQ7OLeLZpxDnCMKz1tBlPcUminCT&amp;index=6")

3. [Renting a virtual hosting](https://www.youtube.com/watch?v=yd3ar2y0pCo "https://www.youtube.com/watch?v=yd3ar2y0pCo")
4. [Migrating subscription and Expert Advisor to virtual hosting](https://www.youtube.com/watch?v=NGSHrX-QAOU "https://www.youtube.com/watch?v=NGSHrX-QAOU")


**1.1. The idea of the copier**

The copier operates on a virtual hosting, on a trade account which is subscribed to a [signal](https://www.mql5.com/en/signals). Its main purpose is to increase the positions opened by the Signals service by the specified number of times. Fig. 1 shows the idea of the copier that is attached to a hedging account:

![](https://c.mql5.com/2/24/01-hedging_en.png)

Fig. 1. The idea of the copier on a hedging account

Fig. 2 demonstrates the idea of the copier operation on a trade account with netting position accounting system:

![](https://c.mql5.com/2/24/01-netting_en.png)

Fig. The idea of the copier on a netting account

Thus, as soon as the Signals service successfully performs a deal on the subscriber account, the copier immediately performs a deal with the volume determined by the formula:

( **volume of the deal performed by the Signals service**) \\* " **Deal multiply by**"

The " **Deal multiply by**" parameter is responsible for increase ratio of the deals and can be set in the input parameters:

![Input Parameters ](https://c.mql5.com/2/24/inputs_en__4.png)

Fig. 3. Input Parameters

When working on a netting trade account in MetaTrader 5, the copier actions lead to an increase in the volume of the copied position. On a hedging account in MetaTrader 5, the copier actions cause a new position with an increased volume to be opened. Below is an example of the copier operation for netting and hedging MetaTrader 5 accounts with the " **Deal multiply by**" parameter for increasing the deals set to 5:

| Action of the Signals service | Action of the copier on<br> hedging account | Action of the copier on<br> netting account |
| --- | --- | --- |
| Copied deal BUY EURUSD 0.01 lot | Opened new position BUY EURUSD 0.05 lot — thus, subscriber has two positions BUY EURUSD 0.01 lot and BUY EURUSD 0.06 lot | Opened deal BUY EURUSD 0.05 lot — thus, subscriber has a position BUY EURUSD 0.06 lot |

### 2\. Features of the Signals Service

**2.1. Friend or foe**

As the copier monitors all deals performed by the Signals service, it is necessary to know certain features of the Signals service operation. For example, this service maintains only "friendly" positions — those copied via the service.

When a position is successfully opened at the subscriber, the Signals service writes the name of the signal to the copied deal comment and a unique identifier to the Magic Number field of the deal.

This very identifier is subsequently used for identification of the deal opened via the Signals service in further synchronizations of the subscription.

This is the comment for BUY 0.01 EURUSD deal copied from the "Test3443431" signal. It can be viewed by hovering the mouse over the copied deal in the "History" tab of the "Toolbox" window:

![Comment of the copied deal ("History" tab)](https://c.mql5.com/2/24/subscriber_tab_history_en.png)

Fig. 4. Comment of the copied deal ("History" tab)

Here:

1. "#69801797" — deal ticket;
2. "Test3443431" — name of the signal the deal has been copied from;
3. "361104976048745684" — Magic Number — deal identifier.

In the "Trade" tab it can be seen that the current position of the subscriber has the same description but without the ticket:

![Comment of the copied deal ("Trade" tab)](https://c.mql5.com/2/24/subscriber_tab_trade_en.png)

Fig. 5. Comment of the copied deal ("Trade" tab)

In other words, the Signals service accounts the positions on the subscriber side using the Magic Number field of the deal. The accounted positions will be identified as "friendly". And the Signals service can change the volumes or close only the "friendly" positions.

Unaccounted positions will be identified as "foes". The Signals service has no power over such positions.

**2.2. Substitute the Magic Number or not?**

The MetaTrader 5 terminal allows to connect both hedging and netting trading accounts of the subscriber. Depending on the type of the connected account the behavior of the copier regarding the substitution of the Magic Number in the copied position will differ drastically. To answer this question, let us consider the following figure:

![](https://c.mql5.com/2/24/02_en.png)

Fig. 6. Netting account, without substitution of the Magic Number

As it can be seen, on a netting account the copier increased the position without substituting the Magic Number — this caused the position to become "foe" for the Signals service. That is, operation of the copier without substituting the Magic Number is a mistake. This means that on a netting account the copier must always substitute the Magic Number when increasing the position. But while working on a hedging account everything is the opposite: when increasing the position the copier must substitute the Magic Number with its own.

**2.3. What happens to the "foe" position, if provider closes its position?**

On the example of a hedging account: a position BUY 0.01 EURUSD was copied to the subscriber trading account. Then the subscriber decided to intervene in the work of the service and opened a position BUY 0.01 EURUSD. During one of the nearest synchronizations the Signals service generated this error:

```
Signal  '3447880': local positions do not correspond to signal provider [#85639429 buy 0.04 EURUSD 1.11697]
```

That is, the position BUY 0.01 EURUSD, manually opened by the subscriber, had been identified as "foe" by the Signals service. And now let us see what happens when the provider closes its position: the Signals service similarly closes the "friendly" position, but the manually opened position BUY 0.01 EURUSD remains in the terminal. By the way, inadmissibility of manual intervention in the work of the Signals service is clearly stated in the rules:

### IV. Subscription to Signals

20\. Execution of your own trades in the account which is subscribed to a Signal constitutes an interference and can lead to unpredictable results.

### 3\. How to detect emergence of a deal

The choice of method to identify the emergence of a deal, on the one hand, affects the copying response rate, and on the other - could lead to additional financial costs to cover spread if a deal is opened with an increased volume by mistake.

It should be remembered that we are subscribed to a signal, which means that the trade events of the signal provider are monitored by the [Signals](https://www.mql5.com/en/signals) service. If necessary, this service will open or close the position/positions. Any opening/closing of a position/positions causes a change in the trading account of the subscriber. This very change should be tracked, and the [OnTradeTransaction()](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction) event will inform about it.

**3.1.** **Tracking using the [OnTradeTransaction()](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction)**

Why was the OnTradeTransaction() chosen and not the OnTrade()? Because the OnTradeTransaction() contains very useful information — type of the trade transaction. Out of [all possible transaction types](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_transaction_type) only one is of interest:

TRADE\_TRANSACTION\_DEAL\_ADD — adding a deal to history. It is performed as a result of order execution or making operations with the account balance.

That is, the copier waits for the deal to be added to history (and this guarantees a successful trade operation on the subscriber account), and only after that it starts processing the situation.

### 4\. When subscription is possible and when it is not

A subscription can be successful only if both accounts have the same position accounting system:

| Subscription | Result |
| --- | --- |
| Provider — netting accounting, subscriber — hedging accounting | An attempt to subscribe an account with hedging accounting system to a provider with netting accounting system fails. <br>```<br>2016.05.11 11:15:07.086 Signal  '*******': subscribe to signal [*****] started<br>2016.05.11 11:15:07.141 Signal  '*******': subscription/renewal prohibited<br>```<br>If the provider has netting account and the subscriber has hedging account, the signal cannot be subscribed to. |
| Provider — hedging accounting, subscriber — netting accounting | An attempt to subscribe an account with netting accounting system to a provider with hedging accounting system fails. <br>```<br>2016.05.11 11:39:54.506 Signal  '*******': subscribe to signal [******] started<br>2016.05.11 11:39:54.560 Signal  '*******': subscription/renewal prohibited<br>```<br>If the provider has hedging account and the subscriber has netting account, the signal cannot be subscribed to. |
| Provider — hedging accounting, subscriber — hedging accounting | Subscription successful. |
| Provider — netting accounting, subscriber — netting accounting | Subscription successful. |

### 5\. Store information about copying

What is the best place to store the information about the copying performed? And a more global question — is it necessary to store such information at all? In this version of the copier, the information is stored in a [structure](https://www.mql5.com/en/docs/basis/types/classes). Structure declaration:

```
//+------------------------------------------------------------------+
//| Structure of congruences of positions of the terminal and        |
//| positions of the copier                                          |
//+------------------------------------------------------------------+
struct correlation
  {
   long              POSITION_IDENTIFIER_terminal; // Position identifier (terminal)
   //+------------------------------------------------------------------+
   //| Position identifier is a unique number that is assigned          |
   //| to every newly opened position and doesn't change during         |
   //| the entire lifetime of the position.                             |
   //| Position turnover doesn't change its identifier.                 |
   //+------------------------------------------------------------------+
   double            POSITION_VOLUME_terminal;     // Volume of the position opened by the Signals service
   long              POSITION_IDENTIFIER_copier;   // Position identifier (copier)
   //+------------------------------------------------------------------+
   //| Position identifier is a unique number that is assigned          |
   //| to every newly opened position and doesn't change during         |
   //| the entire lifetime of the position.                             |
   //| Position turnover doesn't change its identifier.                 |
   //+------------------------------------------------------------------+
   ulong             DEAL_ticket;                  // Deal ticket, if the deal is executed.
  };
```

The structure contains only four elements:

1. "POSITION\_IDENTIFIER\_terminal" — this structure stores the identifier of the position opened by the terminal (Signals service)
2. "POSITION\_VOLUME\_terminal" — this stores the volume of the position opened by the terminal (Signals service)
3. "POSITION\_IDENTIFIER\_copier" — this element stores identifier of the position opened by the copier
4. "DEAL\_ticket" — this contains ticket of the deal opened by the copier if that deal succeeded

All actions are performed by the copier only from within the OnTradeTransaction function, and only if the trade transaction type is TRADE\_TRANSACTION\_DEAL\_ADD. When these conditions are met, it always looks for the deal which has generated this transaction, and retrieve its parameters:

```
//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
//--- get transaction type as enumeration value
   ENUM_TRADE_TRANSACTION_TYPE type=trans.type;
//--- if transaction is result of addition of the transaction in history
   if(type==TRADE_TRANSACTION_DEAL_ADD)
     {
      long     deal_entry        =0;
      double   deal_volume       =0;
      string   deal_symbol       ="";
      long     deal_type         =0;
      long     deal_magic        =0;
      long     deal_positions_id =0;
      string   deal_comment      ="";
      if(HistoryDealSelect(trans.deal))
        {
         deal_entry=HistoryDealGetInteger(trans.deal,DEAL_ENTRY);
         deal_volume=HistoryDealGetDouble(trans.deal,DEAL_VOLUME);
         deal_symbol=HistoryDealGetString(trans.deal,DEAL_SYMBOL);
         deal_type=HistoryDealGetInteger(trans.deal,DEAL_TYPE);
         deal_magic=HistoryDealGetInteger(trans.deal,DEAL_MAGIC);
         deal_positions_id=HistoryDealGetInteger(trans.deal,DEAL_POSITION_ID);
         deal_comment=HistoryDealGetString(trans.deal,DEAL_COMMENT);
         //if(deal_magic==0 && SignalInfoGetString(SIGNAL_INFO_NAME)!=HistoryDealGetString(trans.deal,DEAL_COMMENT))
         //   return;
        }
      else
         return;
...
```

The table below shows the operation logic of the copier on a hedging and netting accounts:

the ✔ sign means that a record wad made into this element of the structure

the ✔ sign means that the copier did not make any changes to elements of the structure

| The client terminal | POSITION\_IDENTIFIER terminal | POSITION\_VOLUME terminal | POSITION\_IDENTIFIER copier | deal\_ticket |
| --- | --- | --- | --- | --- |
| **Hedging. Netting.** [DEAL\_ENTRY\_IN](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties)<br> Searching the **deal\_ticket** structure elements for the value of **trans.deal**. |
| If no matches found: increase the structure by 1 ... | 0 | 0 | 0 | 0 |
| ... and regard it as a service deal — therefore, open a position ( [DEAL\_VOLUME](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) \\* coefficient), and if [CTrade.ResultDeal()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctraderesultdeal) != 0, write to the **POSITION\_IDENTIFIER\_terminal** element the value of [DEAL\_POSITION\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties), the value of [CTrade.ResultDeal()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctraderesultdeal) is written to the **deal\_ticket** element, and the deal\_volume — to **POSITION\_VOLUME\_terminal**. | ✔ | ✔ | 0 | ✔ |
| If found – then it is a deal opened by the copier, therefore write its [DEAL\_POSITOIN\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) to the **POSITION\_IDENTIFIER\_copier** element. | ✔ | ✔ | ✔ | ✔ |
|  |
| **Hedging. Netting.** [DEAL\_ENTRY\_OUT](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties)<br> Searching for [DEAL\_POSITION\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) in the elements of the **POSITION\_IDENTIFIER\_terminal** and **POSITION\_IDENTIFIER\_copier** structures. |
| **Hedging.**<br> ... found [DEAL\_POSITION\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties)... |
| ... ... in element **POSITION\_IDENTIFIER\_copier** – do nothing and leave | ✔ | ✔ | ✔ | ✔ |
| ... ... in element **POSITION\_IDENTIFIER\_terminal** –... | ✔ | ✔ | ✔ | ✔ |
| ... ... ... does this position still exist? If it exists, close the position opened by the copier... | ✔ | ✔ | ✔ | ✔ |
| ... ... ... ... open a position (volume of the found position \* coefficient) and if [CTrade.ResultDeal()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctraderesultdeal) != 0, write the value of [CTrade.ResultDeal()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctraderesultdeal) to the **deal\_ticket** element. | ✔ | ✔ | ✔ | ✔ |
| ... ... ... no, this position no longer exists. Close the position opened by the copier. | ✔ | ✔ | ✔ | ✔ |
| **Netting.**<br> ... found [DEAL\_POSITION\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties)... |
| ... ... in element **POSITION\_IDENTIFIER\_terminal**– (the previous volume opened by the Signals service is currently stored in the structure) calculate the new volume, and if [CTrade.ResultDeal()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctraderesultdeal) != 0, then write the value of [CTrade.ResultDeal()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctraderesultdeal) to the **deal\_ticket** element. | ✔ | ✔ | ✔ | ✔ |

### Conclusion

If you subscribed to a signal, rented the built-in virtual hosting, but the provider trades the minimum (or very small) lot — you can send the copier considered in this article to the built-in virtual hosting. That way all deals will be increased in proportion to the copier settings.

Attention: the " **copier.mq5**" file of the copier and the " **languages.mqh**" included file must be located in the same folder.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2438](https://www.mql5.com/ru/articles/2438)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2438.zip "Download all attachments in the single ZIP archive")

[copier.mq5](https://www.mql5.com/en/articles/download/2438/copier.mq5 "Download copier.mq5")(37.81 KB)

[languages.mqh](https://www.mql5.com/en/articles/download/2438/languages.mqh "Download languages.mqh")(6.19 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717)
- [Gap - a profitable strategy or 50/50?](https://www.mql5.com/en/articles/5220)
- [Elder-Ray (Bulls Power and Bears Power)](https://www.mql5.com/en/articles/5014)
- [Improving Panels: Adding transparency, changing background color and inheriting from CAppDialog/CWndClient](https://www.mql5.com/en/articles/4575)
- [How to create a graphical panel of any complexity level](https://www.mql5.com/en/articles/4503)
- [Comparing speeds of self-caching indicators](https://www.mql5.com/en/articles/4388)
- [LifeHack for traders: Blending ForEach with defines (#define)](https://www.mql5.com/en/articles/4332)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/95111)**
(25)


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
10 May 2017 at 10:26

**Александр:**

You can't edit the source code in the Market.

In the article there is a warning about working with netting accounts and in the code there is a complete ban on working with netting (the Expert Advisor will simply crash with an error on netting).

But in the Market, unfortunately, it is impossible to introduce a complete ban - the product will not pass autovalidation, so in the Market on netting will only pop up a warning message [(MessageBox](https://www.mql5.com/en/docs/constants/io_constants/messbconstants "MQL5 Documentation: MessageBox dialogue box constants")).

![highfire](https://c.mql5.com/avatar/2014/6/5399AA13-D5C3.jpg)

**[highfire](https://www.mql5.com/en/users/highfire)**
\|
14 Apr 2021 at 13:49

Opened infinit orders and I lost 1000 dollars.

I cant believe someone could post it here without even test.


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
14 Apr 2021 at 14:03

**highfire :**

Opened infinit orders and I lost 1000 dollars.

I cant believe someone could post it here without even test.

Do you really know how to read and think?

### Risk Warning

Before using this method, you need to rely primarily on common sense, as the increase in the copy ratio entails increased risks.

While copying, the system periodically checks whether positions on the subscriber's account match the provider's ones. If a mismatch is detected (i.e., only some positions have been copied), the system tries to eliminate that and copy the missing ones. Unlike the initial synchronization, the total floating profit of the provider is not checked. If the subscriber started copying, they should follow the provider's trading strategy to the maximum possible extent. It is impossible to copy some positions, while ignoring others.

If the system detects positions not based on signals on the subscriber's account, it offers to close them or closes them automatically depending on the ["Synchronize positions without confirmations"](https://www.metatrader5.com/en/terminal/help/signals/signal_subscriber#settings "https://www.metatrader5.com/en/terminal/help/signals/signal_subscriber#settings") setting.

![Por](https://c.mql5.com/avatar/avatar_na2.png)

**[Por](https://www.mql5.com/en/users/1isuniverse)**
\|
19 May 2021 at 09:22

wrote:

The copier operates on a [virtual hosting](https://www.mql5.com/en/vps "Forex VPS for MetaTrader 4/5")

'a virtual hosting\` means what?

windows vps services?

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
19 May 2021 at 09:28

**Por :**

wrote:

The copier operates on a [virtual hosting](https://www.mql5.com/en/vps "Forex VPS for MetaTrader 4/5")

'a virtual hosting\` means what?

windows vps services?

The site immediately gave you a hint - the site immediately inserted a reference link.

![How to quickly develop and debug a trading strategy in MetaTrader 5](https://c.mql5.com/2/24/avae17.png)[How to quickly develop and debug a trading strategy in MetaTrader 5](https://www.mql5.com/en/articles/2661)

Scalping automatic systems are rightfully regarded the pinnacle of algorithmic trading, but at the same time their code is the most difficult to write. In this article we will show how to build strategies based on analysis of incoming ticks using the built-in debugging tools and visual testing. Developing rules for entry and exit often require years of manual trading. But with the help of MetaTrader 5, you can quickly test any such strategy on real history.

![Graphical Interfaces X: Updates for Easy And Fast Library (Build 2)](https://c.mql5.com/2/23/Graphic-interface_10.png)[Graphical Interfaces X: Updates for Easy And Fast Library (Build 2)](https://www.mql5.com/en/articles/2634)

Since the publication of the previous article in the series, Easy And Fast library has received some new features. The library structure and code have been partially optimized slightly reducing CPU load. Some recurring methods in many control classes have been moved to the CElement base class.

![MQL5 Cookbook - Trading signals of moving channels](https://c.mql5.com/2/24/ava2.png)[MQL5 Cookbook - Trading signals of moving channels](https://www.mql5.com/en/articles/1863)

The article describes the process of developing and implementing a class for sending signals based on the moving channels. Each of the signal version is followed by a trading strategy with testing results. Classes of the Standard Library are used for creating derived classes.

![False trigger protection for Trading Robot](https://c.mql5.com/2/21/avatar.png)[False trigger protection for Trading Robot](https://www.mql5.com/en/articles/2110)

Profitability of trading systems is defined not only by logic and precision of analyzing the financial instrument dynamics, but also by the quality of the performance algorithm of this logic. False trigger is typical for low quality performance of the main logic of a trading robot. Ways of solving the specified problem are considered in this article.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kdpylncnvfoojhdwkreqmyfvrhuxpfbs&ssn=1769193854008073262&ssn_dr=0&ssn_sr=0&fv_date=1769193854&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2438&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20copy%20signals%20using%20an%20EA%20by%20your%20rules%3F%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919385478161145&fz_uniq=5072056240315511678&sv=2552)

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