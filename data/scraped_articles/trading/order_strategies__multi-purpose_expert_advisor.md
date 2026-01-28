---
title: Order Strategies. Multi-Purpose Expert Advisor
url: https://www.mql5.com/en/articles/495
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:21:18.400290
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=mywtxsksqbcuiqmtoqrdnlqzsvqwvkxw&ssn=1769181676607768513&ssn_dr=0&ssn_sr=0&fv_date=1769181676&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F495&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Order%20Strategies.%20Multi-Purpose%20Expert%20Advisor%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918167633140638&fz_uniq=5069368509911204843&sv=2552)

MetaTrader 5 / Trading


### Introduction

The initial key element of any trading strategy is the price analysis and analysis of technical indicators forming the basis for opening a position. We will call it _market analysis_, i.e. everything that happens in the market and is beyond our control.

In addition, strategies may require another type of analysis. We will call it _analysis of the current trading situation_. It comprises analysis of the trading position status and analysis of any available/missing pending orders (if any are used in a strategy). The results of such analysis bring us to decisions of whether certain actions with positions or orders need to be performed, e.g. closing, moving Stop Loss, placing or deleting pending orders, etc. In other words, such analysis includes the study of our market activity, actions according to the situation we (or an Expert Advisor) created and the rules of the strategy in use.

A commonly known Trailing Stop can to a certain extent be considered the second type of elements in a trading strategy. Consider the following analysis: if there is an open position with the profit higher than the set value, while the Stop Loss is not set or is further than the distance from the current price as specified in the settings, the Stop Loss will be moved.

Trailing Stop is a fairly simple function to be of particular interest. Besides, it can be classified as a totally different category of trading strategy elements, being a position management function. Thus, a trading strategy can be comprised of three categories of elements:

1. Market analysis and actions based thereon.

2. Analysis of the trading situation and actions based thereon.

3. Position management.


This article centers around strategies that actively use pending orders (we will call them order strategies for short), a metalanguage that can be created to describe such strategies and the development and use of a multi-purpose tool (Expert Advisor) whose operation is based on those descriptions.

### Examples of Order Strategies

Trading normally starts with opening an initial position. This can be done in several ways:

1. Opening a market position:


   - In the direction suggested by indicators.

   - In the direction selected by the user in the Expert Advisor properties window.

   - Based on the results of closing the last position. Instead of being an initial position, such position can be an intermediary operation phase.


3. Two opposite Stop orders. When either of orders triggers, the second order is deleted.

4. Two opposite Limit orders. When either of orders triggers, the second order is deleted.

5. Limit order and Stop order placed in the same direction. In this case, it is necessary to decide on the direction of the orders, as in 1.


Once the initial position is opened, you can use different order strategies.

**Scaling in Using Limit Orders (Fig. 1.)**

You open an initial position and set one or more Limit orders in the same direction with increasing lot size. As Limit orders trigger, new Limit orders are set until the position is closed at Take Profit. When the position is closed at Take Profit, the remaining pending orders are deleted.

![Fig. 1.  Scaling in Using Limit Orders](https://c.mql5.com/2/4/LimitAdd2.png)

Fig. 1. Scaling in Using Limit Orders

**Stop and Reverse (Fig. 2.)**

You open an initial position and set the opposite Stop order with an increased lot size at the Stop Loss level of the initial position. When the position is closed at Stop Loss, the pending order kicks in a new opposite Stop order is again set at its Stop Loss level, and so on until the position is closed at Take Profit. When the position is closed at Take Profit, the remaining pending order is deleted.

![Fig. 2. Stop and Reverse](https://c.mql5.com/2/4/StopRev2.png)

Fig. 2. Stop and Reverse

**Pyramiding (Fig. 3.)**

You open an initial position and if it appears to be winning, you increase its volume (scale in) and move the Stop Loss to Breakeven. If the position is closed at Take Profit, it is by then expected to have reached a quite large volume, and consequently profit. If however the Stop Loss triggers during the intermediary phase, there will simply be no profit.

![Fig. 3. Pyramiding](https://c.mql5.com/2/4/Piramiding2.png)

Fig. 3. Pyramiding

**Reopening (Fig. 4.)**

You open a market position. Closing at Stop Loss is followed by a new opening with an increased lot size, and so on until the position is closed at Take Profit. This strategy is similar to scaling in using Limit orders.

![Fig. 4. Reopening](https://c.mql5.com/2/4/ReOpen2.png)

Fig. 4. Reopening

It is quite possible to combine all of the above strategies. If a position appears to be winning, pyramiding comes in handy; otherwise, if losses are under way, it may be appropriate to scale in using Limit orders. That said, scaling in using Limit orders does not have to be continuous. For example, you can first scale in three times, then do several Stop and Reverse and switch to scaling in using Limit orders again, etc.

In practice, the development of order strategies can be quite time consuming not only due to the scope of the coding required but also because of the need to use creative thinking in every single case. Let us try to facilitate the programming of such strategies by creating a multi-purpose Expert Advisor that would allow us to implement any order strategy.

### The Basic Principle

The basic principle of order strategy development is the identification of the current strategy operation phase and carrying out actions according to that phase.

Let us have a look at the following example: we need to open a market position, say, this will be a buy position. Once the position is opened, two pending orders are required to be set: a Stop order above and a Limit order below. As we start, there is no position or order in the market thus we identify the phase as the initial operation phase where we need to open a position. The existence of a market position would suggest that this is the next operation phase. So the phases that can be identified are as follows:

1. There is no position or order. A position needs to be opened.

2. There is a position but no order is set. A Stop order is required.

3. There is a position and a Stop order. A Limit order is required.


Work in line with these rules will be reliably implemented however it will require three ticks: the first tick to identify the lack of the position and opening of the position, the next tick to identify the position and lack of orders and the third tick to identify the position and an order. The strategy requires that all three actions be performed in one go.

We should therefore try to perform all actions at once: if there is no position or order, we should open a position. If the position has been successfully opened, we should send a request to set a Stop order and another request to set a Limit order. It may well be that none of the sent requests to set a pending order will be accepted (due to connectivity issues, lack of price information, etc.) but the trading situation has transitioned to another phase where we have an open position. This means that all possible intermediary phases should be covered:

1. There is no position. A position needs to be opened. If the position has successfully been opened, requests for Stop order and Limit order should be sent.

2. There is a position but there is no pending order. Requests for Stop order and Limit order should be sent.

3. There is a position and a Stop order but the Limit order is missing. A request for a Limit order should be sent.

4. There is a position and a Limit order but the Stop order is missing. A request for a Stop order should be sent.


Please note that in order to identify the phase in the given example, the trade situation must be a complete match to the identification rules provided. There should be certain order sets: only one position and no orders or a position and either of the orders - there can be no other way. The description of strategy operation phases following this principle can get very lengthy and make the entire process very time consuming due to the need to account for all possible options which can ultimately turn out to be unfeasible. The operation rules for the above example can be set out in a slightly different way:

1. There is no position. A position needs to be opened. If the position is successfully opened, requests for a Stop order and Limit order should be sent.

2. There is a position. In this case, there should be two pending orders in the market. Check if there is a Stop order in the market and set it if it is missing. Check if there is a Limit order in the market and set it if it is missing.


In this case, we have a minimum set of rules for the identification of the operation phase and a complete description of a trading situation where we should be at that phase.

Application of this principle will require the position itself to be identified in order to distinguish whether it is the initial position or if a certain order has already triggered. In these circumstances, there will be no need to try to place the second order as the system is in the new operation phase. Identification of orders will also be required but we will have a look at position and order identification a bit later. Let us now set up the basic principle of describing order strategies in a more clear and compact way:

1. We need a method for the identification of the current operation phase with the least possible amount of information.

2. Every operation phase must have a complete description of the situation corresponding to that phase.

3. If market actions (opening, closing, scaling in, scaling out, etc.) or pending orders are required at any given phase, such phase should be divided into two sub-phases: before the market action performance and after (so as to enable us to perform all actions in one go and repeat failed pending order attempts).

4. If market actions (opening, closing, scaling in, scaling out, etc.) or pending orders are required at any given phase, pending orders shall be dealt with after the successful completion of the market action.

5. One phase can only correspond to one market action and any number of actions with pending orders.


### Order and Position Identification

Positions and orders can be identified in several ways: using the order comment, magic number or global variables. Let us use comments. The main problems arising out of the use of comments are limited comment size and the fact that the broker can add to the comment something of his own. If there is not enough space for the broker's entry, a part of the comment will be cut off.

You should therefore take the least possible space in the comment and try to find a way to separate it from possible broker's entries. Every order only needs one identifier. In practice, it can be 1 or 2 figures or a combination of a letter and one or two figures. At the end of the identifier we will put a mark, say, "=" (it was never noticed to be used by brokers in their entries). So we have maximum 4 characters. To obtain the identifier from a comment, we can use the following function:

```
//+------------------------------------------------------------------+
//|   Function for obtaining the identifier from the aComment string |
//+------------------------------------------------------------------+
string GetID(string aComment)
  {
   int    p =StringFind(aComment,"=",0); // Determine the position of the separator
   string id=StringSubstr(aComment,0,p); // Get the substring located before the separator
   return(id);
  }
//+------------------------------------------------------------------+
```

If the position or order needs to be checked against any known identifier, this can be done as follows:

```
//+------------------------------------------------------------------+
//|   Checking the comment against the set identifier                |
//+------------------------------------------------------------------+
bool FitsID(string aID,string aComment)
  {
   return(StringFind(aComment,aID+"=",0)==0);
  }
//+------------------------------------------------------------------+
```

### Metalanguage for Description of Order Strategies

Let us now define the language to be used for putting down order strategies. It should be concise, clear and intuitive while at the same time being in line with [MQL5](https://www.mql5.com/en/docs "Documentation on the automated trading language") to ensure fast execution of its commands without unnecessary calculations. I will leave it to the readers to decide whether the outcome has been successful or not.

The description of the strategy is done in a text file which is then connected to the Expert Advisor by specifying its name in the Expert Advisor properties window.

One line of the file corresponds to one operation phase of the system. The line is split into two fields. The first field contains the phase identification rules. The second one covers the list of actions. Fields are separated by the vertical line "\|". Identification rules and action list items are set out, separated by a semicolon ";".

In addition to commands, the right side of each line can contain a comment separated from the rest of the text by "#", e.g.:

```
Nothing | Buy(M1,1,0,0) #If there is no position or order in the market, open a Buy position, mark it with "М1", lot 1, no Stop Loss, no Take Profit.
```

**Phase Identification**

Phase identification may require information on the current market position, pending orders or the last trade. In addition to the position status, some position details may be required, such as price, profit, Stop Loss value, if any is set, etc. Information required on the last trade may include the trade results. For pending orders, it may be necessary to specify their opening price, Stop Loss, Take Profit (which will most likely be required at the execution phase).

This information can be obtained using trade data access commands. Most of these commands will have two parameters: position or order identifier and parameter identifier. If the parameter identifier is not specified, only the existence of the trade object specified by the command and identifier will be subject to check.

For example, the Buy(M1) command suggests that there must be a market position with the "M1" identifier. The Buy() command alone (or simply Buy without the parentheses) means that there must be a Buy position with any identifier. If you specify the parameter identifier, it will indicate the parameter value, e.g. Buy(M1,StopLossInPoints) - Stop Loss value in points set for a Buy position with the "M1" identifier. If no identifier is specified - Buy(,StopLossInPoints), we take it as a Stop Loss of a Buy position with any identifier (as long as there is a Buy position).

The obtained value can be used in the expression for checking conditions, e.g. Buy(M1,StopLossInPoints)>=0 - position is at breakeven. If there is no position or there is a position with a different identifier, the phase expressed that way in the identification rules will not be identified, i.e. there is no need to account for two conditions - to check the position status and Stop Loss value. However, in this case the existence of the Stop Loss will need to be checked in advance - Buy(M1,StopLossExists); Buy(M1,StopLossInPoints)>=0.

When checking values, any comparison expression can be used: ">=", "<=", "==", "!=", ">", "<". The value in the right side of the comparison can be expressed as a number or be represented by special variables: Var1, Var2 ... Var20. "p" added to a number or a variable will suggest that the value will be additionally multiplied by the point value (the [\_Point](https://www.mql5.com/en/docs/predefined/_point "_Point variable") variable).

Alternatively, there may be a more complex arithmetic expression in the right side of the comparison expression. It can be as follows: X1\*X2+X3\*X4 ("+" can certainly be replaced with "-"), where X1, X2, X3 and X4 can be numbers, variables or data access commands. The below example can theoretically be considered correct (if we disregard its practical value):

```
-BuyStop(BS1,StopLossInPoints)*-SellLimit(SL1,StopLossInPoints)+-SellStop(SS1,StopLossInPoints)*-BuyLimit(SL1,StopLossInPoints)
```

Table 1 shows the list of all access commands.

**Table 1. Data access commands**

| Index | Command | Possible parameters | Purpose |
| --- | --- | --- | --- |
| 0 | Nothing | No parameters | There is no position or pending order in the market |
| 1 | NoPos | No parameters | There is no position in the market |
| 2 | Pending | Object identifier, order parameter identifier | Any pending order with the object identifier specified. If the object identifier is not specified, then any pending order regardless of the value of the object identifier |
| 3 | Buy | Object identifier, position parameter identifier | A Buy position with the object identifier specified. If the object identifier is not specified, then just a Buy position |
| 4 | Sell | Object identifier, position parameter identifier | A Sell position with the object identifier specified. If the object identifier is not specified, then just a Sell position |
| 5 | BuyStop | Object identifier, order parameter identifier | A BuyStop order with the object identifier specified. If the object identifier is not specified, then just a BuyStop order |
| 6 | SellStop | Object identifier, order parameter identifier | A SellStop order with the object identifier specified. If the object identified is not specified, then just a SellStop order |
| 7 | BuyLimit | Object identifier, order parameter identifier | A BuyLimit order with the object identifier specified. If the object identifier is not specified, then just a BuyLimit order |
| 8 | SelLimit | Object identifier, order parameter identifier | A SellLimit order with the object identifier specified. If the object identifier is not specified, then just a SellLimit order |
| 9 | BuyStopLimit | Object identifier, order parameter identifier | A BuyStopLimit order with the object identifier specified. If the object identifier is not specified, then just a BuyStopLimit order |
| 10 | SellStopLimit | Object identifier, order parameter identifier | A SellStopLimit order with the object identifier specified. If the object identifier is not specified, then just a SellStopLimit order |
| 11 | LastDeal | Empty, trade parameter identifier | The last trade |
| 12 | LastDealBuy | Empty, trade parameter identifier | The last trade is the Buy trade |
| 13 | LastDealSell | Empty, trade parameter identifier | The last trade is the Sell trade |
| 14 | NoLastDeal | No parameters | There is no data on the trade in the history; this is necessary in case the Expert Advisor has just started operating on the account |
| 15 | SignalOpenBuy | No parameters | Indicator signal to open a Buy position |
| 16 | SignalOpenSell | No parameters | Indicator signal to open a Sell position |
| 17 | SignalCloseBuy | No parameters | Indicator signal to close a Buy position |
| 18 | SignalCloseSell | No parameters | Indicator signal to close a Sell position |
| 19 | UserBuy | No parameters | User command to buy |
| 20 | UserSell | No parameters | User command to sell |
| 21 | Bid | No parameters | Bid price |
| 22 | Ask | No parameters | Ask price |
| 23 | ThisOpenPrice | No parameters | Opening price of the order whose parameter is calculated. It is used in action commands for pending orders, except for orders of the StopLimit type |
| 24 | ThisOpenPrice1 | No parameters | Opening price-1 of the order whose parameter is calculated. It is used in action commands for pending orders of the StopLimit type |
| 25 | ThisOpenPrice2 | No parameters | Opening price-2 of the order whose parameter is calculated. It is used in action commands for pending orders of the StopLimit type |
| 26 | LastEADeal | Object identifier, trade parameter identifier | The last trade executed by the Expert Advisor. The last trade that has "=" in its comment is searched in the history and then checked against the object identifier |
| 27 | LastEADealBuy | Object identifier, trade parameter identifier | The last trade executed by the Expert Advisor is the Buy trade. The last trade that has "=" in its comment is searched in the history and then checked against the object identifier and direction of the trade |
| 28 | LastEADealSell | Object identifier, trade parameter identifier | The last trade executed by the Expert Advisor is the Sell trade. The last trade that has "=" in its comment is searched in the history and then checked against the object identifier and direction of the trade |
| 29 | NoTradeOnBar | No parameters | There are no trades on the last bar |

Commands set out in Table 1 allow you to access the following types of trade objects: positions, orders, trades and orders to be set. Different objects have different parameter sets.

Table 2 features all parameter identifiers along with object types they can be applied to.

**Table 2. Data access identifiers.**

| Index | Identifier | Purpose | Trade object type |
| --- | --- | --- | --- |
| 0 | ProfitInPoints | Profit in points | Position |
| 1 | ProfitInValute | Profit in the deposit currency | Positions, trades |
| 2 | OpenPrice | Opening price | Positions, pending orders (except for StopLimit orders) |
| 3 | LastPrice | Price | Trades |
| 4 | OpenPrice1 | StopLimit-to-Limit transition price | Pending orders of the StopLimit type. OpenPrice identifier applies when the order transitions to Limit |
| 5 | OpenPrice2 | StopLimit-to-position transition price | Pending orders of the StopLimit type. OpenPrice identifier applies when the order transitions to Limit |
| 6 | StopLossValue | StopLoss value | Positions, pending orders |
| 7 | TakeProfitValue | Take Profit value | Positions, pending orders |
| 8 | StopLossInPoints | Stop Loss in points | Positions, pending orders |
| 9 | TakeProfitInPoints | Take Profit in points | Positions, pending orders |
| 10 | StopLossExists | Existence of Stop Loss | Positions, pending orders |
| 11 | TakeProfitExists | Existence of Take Profit | Positions, pending orders |
| 12 | Direction | Direction 1 - Buy, -1 - Sell | Positions, pending orders, trades |

**Description of Actions**

Actions include opening and closing of market positions, setting, modification and deletion of pending orders, execution of management functions: Trailing Stop, Breakeven, Trailing Stop for a pending order (any other position management functions).

Actions of opening a position and setting an order imply the use of parameters required to execute these actions. Those parameters will be specified after the command in parentheses, in the way functions are usually called. Identifier is the first parameter for all commands. When specifying parameters, you can use numeric values, variables, as well as parameters of the existing position or order. You can also use arithmetic expressions like X1\*X2+X3\*X4 touched upon in the Phase Identification section for all parameters of action commands.

Table 3 shows all action commands.

**Table 3. Action commands**

| Index | Command | Purpose |
| --- | --- | --- |
| 0 | Buy(ID,Lot,StopLoss,TakeProfit) | Opening a Buy position |
| 1 | Sell(ID,Lot,StopLoss,TakeProfit) | Opening a Sell position |
| 2 | Close(ID) | Closing a market position |
| 3 | BuyStop(ID,Lot,Price,StopLoss,TakeProfit) | Setting a BuyStop order |
| 4 | SellStop(ID,Lot,Price,StopLoss,TakeProfit) | Setting a SellStop order |
| 5 | BuyLimit(ID,Lot,Price,StopLoss,TakeProfit) | Setting a BuyLimit order |
| 6 | SellLimit(ID,Lot,Price,StopLoss,TakeProfit) | Setting a SellLimit order |
| 7 | BuyStopLimit(ID,Lot,Price1,Price2,StopLoss,TakeProfit) | Setting a BuyStopLimit order |
| 8 | SellStopLimit(ID,Lot,Price1,Price2,StopLoss,TakeProfit) | Setting a SellStopLimit order |
| 9 | Delete(ID) | Deleting a pending order |
| 10 | DeleteAll(ID,BuyStop,SellStop,BuyLimit,SellLimit,BuyStopLimit,SellStopLimit) | Deleting the specified types of pending orders |
| 11 | Modify(ID,Price1,Price2,StopLoss,TakeProfit) | Position or order modification |
| 12 | TrailingStop | Trailing Stop function operation. Function parameters are defined in the Expert Advisor properties window |
| 13 | BreakEven | Breakeven function operation. Function parameters are defined in the Expert Advisor properties window |

Action command parameter descriptions are provided in Table 4.

**Table 4. Action command parameters**

| Parameter | Purpose |
| --- | --- |
| ID | Trade object (position, order) identifier |
| Lot | Lot size in units. The Lots variable that defines the unit value can be found in the Expert Advisor properties window |
| StopLoss | Stop Loss value |
| TakeProfit | Take Profit value |
| Price | Pending order value (except for orders of the StopLimit type) |
| Price1 | StopLimit-to-Limit transition price |
| Price2 | StopLimit-to-position transition price |

Let us now try to put down the order strategies reviewed earlier in our new metalanguage.

### Examples of Order Strategies in Metalanguage

Programs are displayed in tables where all phase identification and action commands are arranged in columns for better comprehension and annotated with comments. Attached to the article are all the programs in text files in the form they should be used in the Expert Advisor.

**Important note:** "+", "-" and "\*" are not allowed in identifiers. It is better to simply use numbers.

**Scaling in Using Limit Orders**

The initial position will be opened according to the direction specified by the user in the properties window. Scaling in is allowed up to 5 times (Limit orders). There can only be three orders in the market at the same time.

**Table 5. Metaprogram for Scaling in Using Limit Orders**

| Phase number | Phase Identification | Actions | Comments |
| --- | --- | --- | --- |
| 1 | Nothing;<br> UserBuy | Buy(1,1,0,Ask+Var1p); <br> BuyLimit(2,2,Buy(1,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p); <br> BuyLimit(3,4,BuyLimit(2,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p);<br> BuyLimit(4,8,BuyLimit(3,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p) | If there is no position or order in the market, the Buy direction is set in the Expert Advisor properties, we open a position with an initial lot. If the position has been successfully opened, we will try to set three Limit orders. That said, every subsequent order will only be set successfully if the previous one has been set because the order price is calculated based on the previous order price. |
| 2 | Buy(1) | BuyLimit(2,2,Buy(1,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p); <br> BuyLimit(3,4,BuyLimit(2,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p);<br> BuyLimit(4,8,BuyLimit(3,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p) | If the position has successfully been opened in phase 1 but not all pending orders have been set, attempts to set all pending orders will continue. |
| 3 | Buy(2) | BuyLimit(3,4,Buy(2,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p);<br> BuyLimit(4,8,BuyLimit(3,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p);<br> BuyLimit(5,16,BuyLimit(4,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p) | If the first Limit order (with identifier 2) has kicked in, attempts to set the other two orders that should have been set (but failed to be set) during previous phases continue and a new order is set so that there is always a total of three Limit orders in the market. |
| 4 | Buy(3) | BuyLimit(4,8,Buy(3,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p);<br> BuyLimit(5,16,BuyLimit(4,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p);<br> BuyLimit(6,32,BuyLimit(4,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p) | Another limit order has triggered and we have to ensure the presence of three Limit orders in the market, like in the previous phase. |
| 5 | Buy(4) | BuyLimit(5,16,Buy(4,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p);<br> BuyLimit(6,32,BuyLimit(5,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p) | This phase ensures the presence of two pending orders only as total orders are close to the maximum number of orders. |
| 6 | Buy(5) | BuyLimit(6,32,Buy(5,OpenPrice)-Var2p,0,ThisOpenPrice+Var3p) | This phase only has one last order. |
| 7 | Buy(6) | Modify(6,,,Buy(6,OpenPrice)-Var4p,) | If the last order has kicked in, a Stop Loss is set for it. |
| 8 | Nothing;<br> UserSell | Sell(1,1,0,Var1p); SellLimit(2,2,Sell(1,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p);<br> SellLimit(3,4,SellLimit(2,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p);<br> SellLimit(4,8,SellLimit(3,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p) | Similar to phase 1 but for the Sell direction. |
| 9 | Sell(1) | SellLimit(2,2,Sell(1,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p);<br> SellLimit(3,4,SellLimit(2,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p);<br> SellLimit(4,8,SellLimit(3,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p) | Similar to phase 2 but for the Sell direction. |
| 10 | Sell(2) | SellLimit(3,4,Sell(2,OpenPrice)+Var2p,0,Var3);<br> SellLimit(4,8,SellLimit(3,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p);<br> SellLimit(5,16,SellLimit(4,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p) | Similar to phase 3 but for the Sell direction. |
| 11 | Sell(3) | SellLimit(4,8,Sell(3,OpenPrice)+Var2p,0,Var3);<br> SellLimit(5,16,SellLimit(4,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p);<br> SellLimit(6,32,SellLimit(4,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p) | Similar to phase 4 but for the Sell direction. |
| 12 | Sell(4) | SellLimit(5,16,Sell(4,OpenPrice)+Var2p,0,Var3);<br> SellLimit(6,32,SellLimit(5,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p) | Similar to phase 5 but for the Sell direction. |
| 13 | Sell(5) | SellLimit(6,32,Sell(5,OpenPrice)+Var2p,0,ThisOpenPrice-Var3p) | Similar to phase 6 but for the Sell direction. |
| 14 | Sell(6) | Modify(6,,,Sell(6,OpenPrice)+Var4p,) | Similar to phase 7 but for the Sell direction. |
| 15 | NoPos;<br> Pending | DeleteAll(,0,0,1,1,0,0) | There are pending orders but no position. This happens when Take Profit of the position is triggered. In this case orders should be deleted. After the deletion of the orders, the system switches to phase 1 or 9. If the user disabled the initial direction during the system operation, there will be no action. |

Use of variables: Var1 - Take Profit of the initial order, Var2 - level at which Limit order are set relative to the opening price of the previous order, Var3 - Stop Loss of the last order.

Fig. 5 is the chart that shows the performance of this metaprogram.

![Fig. 5. Performance of the metaprogram for Scaling in Using Limit Orders](https://c.mql5.com/2/4/LimitAdd.png)

Fig. 5. Performance of the metaprogram for Scaling in Using Limit Orders

Please note: rules for Sell and Buy directions are outlined separately. Every subsequent pending order level is calculated based on the previous order level. If an attempt to set any order fails, the next order will not be set due to the lack of the required parameter. It would be wrong to calculate the level based on the market position price. In such a case some of the orders can be missed.

**Stop and Reverse**

The work starts with two pending Stop orders. It is allowed to have up to five reverses.

**Table 6. Metaprogram for Stop and Reverse**

| Phase number | Phase Identification | Actions | Comments |
| --- | --- | --- | --- |
| 1 | Nothing | BuyStop(1,1,Ask+Var1p,ThisOpenPrice-Var2p,ThisOpenPrice+Var3p);<br> SellStop(1,1,Bid-Var1p,ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | There is no position or order in the market; we try to set two Stop orders with identifier 1. |
| 2 | NoPos; <br> BuyStop(1) | SellStop(1,1,Bid-Var1p,ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | There is no position but there is a BuyStop with identifier 1 which means that there should be a SellStop with identifier 1. |
| 3 | NoPos; <br> SellStop(1) | BuyStop(1,1,Ask+Var1p,ThisOpenPrice-Var2p,ThisOpenPrice+Var3p) | There is no position but there is a SellStop with identifier 1 which means that there should be a BuyStop with identifier 1. |
| 4 | Buy(1) | Delete(1);<br> SellStop(2,2,Buy(1,StopLossValue),ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | There is a Buy position with identifier 1 in the market; in this case there should be no other orders with identifier 1 but there should be a SellStop with identifier 2. |
| 5 | Sell(1) | Delete(1);<br> BuyStop(2,2,Sell(1,StopLossValue),ThisOpenPrice-Var2p,ThisOpenPrice+Var3p) | Similar to phase 4 but the first SellStop has triggered. |
| 6 | Buy(2) | SellStop(3,4,Buy(2,StopLossValue),ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | The second BuyStop has kicked in so the third SellStop should be set. |
| 7 | Sell(2) | BuyStop(3,4,Sell(2,StopLossValue),ThisOpenPrice-Var2p,ThisOpenPrice+Var3p) | The second SellStop has kicked in so the third BuyStop should be set. |
| 8 | Buy(3) | SellStop(4,8,Buy(3,StopLossValue),ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | The third BuyStop has kicked in so the fourth SellStop should be set. |
| 9 | Sell(3) | BuyStop(4,8,Sell(3,StopLossValue),ThisOpenPrice-Var2p,ThisOpenPrice+Var3p) | The third SellStop has kicked in so the fourth BuyStop should be set. |
| 10 | Buy(4) | SellStop(5,16,Buy(4,StopLossValue),ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | The fourth BuyStop has kicked in so the fifth SellStop should be set. |
| 11 | Sell(4) | BuyStop(5,16,Sell(4,StopLossValue),ThisOpenPrice-Var2p,ThisOpenPrice+Var3p) | The fourth SellStop has kicked in so the fifth BuyStop should be set. |
| 12 | Buy(5) | SellStop(6,32,Buy(5,StopLossValue),ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | The fifth BuyStop has kicked in so the sixth SellStop should be set. |
| 13 | Sell(5) | BuyStop(6,32,Sell(5,StopLossValue),ThisOpenPrice-Var2p,ThisOpenPrice+Var3p) | The fifth SellStop has kicked in so the sixth BuyStop should be set. |
| 14 | NoPos; <br> BuyStop(2) | Delete(2) | There is no position but we still have BuyStop; this may happen when the position has closed at Take Profit. In this case the remaining order is deleted and the system switches to phase 1. |
| 15 | NoPos; <br> SellStop(2) | Delete(2) | Similar to phase 14. |
| 16 | NoPos; <br> BuyStop(3) | Delete(3) | Similar to phase 14. |
| 17 | NoPos; <br> SellStop(3) | Delete(3) | Similar to phase 14. |
| 18 | NoPos; <br> BuyStop(4) | Delete(4) | Similar to phase 14. |
| 19 | NoPos; <br> SellStop(4) | Delete(4) | Similar to phase 14. |
| 20 | NoPos; <br> BuyStop(5) | Delete(5) | Similar to phase 14. |
| 21 | NoPos; <br> SellStop(5) | Delete(5) | Similar to phase 14. |
| 22 | NoPos; <br> BuyStop(6) | Delete(6) | Similar to phase 14. |
| 23 | NoPos; <br> SellStop(6) \| | Delete(6) | Similar to phase 14. |

Use of variables: Var1 - level at which initial orders are set from the current market price, Var2 - Stop Loss, Var3 - Take Profit.

Fig. 6 is the chart that shows the performance of this metaprogram.

![Fig. 6. Performance of the metaprogram for Stop and Reverse](https://c.mql5.com/2/4/StopRev.png)

Fig. 6. Performance of the metaprogram for Stop and Reverse

**Pyramiding**

An initial position is opened based on the indicator signal. It is allowed to scale in up to five times.

**Table 7. Metaprogram for Pyramiding"**

| Phase number | Phase Identification | Actions | Comments |
| --- | --- | --- | --- |
| 1 | Nothing; <br> SignalOpenBuy | Buy(1,1,Ask-Var1p,Ask+Var2p\*6) | There is no position or order in the market; we get a signal from the indicators to open a Buy position following which the position is opened. Take Profit is first set at the distance equal to Var2p\*6, at the next step it will be set at Var2p\*5 and so on to ensure that Take Profit is at about the same price level. |
| 2 | Buy(1); <br> Buy(1,ProfitInPoints)>=Var3 | Buy(2,1,Ask-Var1p,Ask+Var2p\*5) | There is a buy position that shows quite a good profit so we scale in. |
| 3 | Buy(2) | Modify(2,,,Buy(2,OpenPrice),) | The position in the market has index 2 suggesting that this position is not initial and must have been scaled in; Stop Loss should be at Breakeven. |
| 4 | Buy(2); <br> Buy(2,ProfitInPoints)>=Var3 | Buy(3,1,Ask-Var1p,Ask+Var2p\*4) | The position is again winning so we scale in. |
| 5 | Buy(3) | Modify(3,,,Buy(3,OpenPrice),) | Stop Loss is moved to Breakeven every time we scale in. |
| 6 | Buy(3); <br> Buy(3,ProfitInPoints)>=Var3 | Buy(4,1,Ask-Var1p,Ask+Var2p\*3) | Similar to phase 4. |
| 7 | Buy(4) | Modify(4,,,Buy(4,OpenPrice),) | Similar to phase 5. |
| 8 | Buy(4); <br> Buy(4,ProfitInPoints)>=Var3 | Buy(5,1,Ask-Var1p,Ask+Var2p\*2) | Similar to phase 4. |
| 9 | Buy(5) | Modify(5,,,Buy(5,OpenPrice),) | Similar to phase 5. |
| 10 | Buy(5); <br> Buy(5,ProfitInPoints)>=Var3 | Buy(6,1,Ask-Var1p,Ask+Var2p) | Similar to phase 4. |
| 11 | Buy(6) | Modify(6,,,Buy(6,OpenPrice),) | Similar to phase 5. |
| 12 | Nothing; <br> SignalOpenSell | Sell(1,1,Bid+Var1p,Bid-Var2p\*6) | Similar to phase 1 but with respect to a Sell position. |
| 13 | Sell(1); <br> Sell(1,ProfitInPoints)>=Var3 | Sell(2,1,Bid+Var1p,Bid-Var2p\*5) | Similar to phase 2 but with respect to a Sell position. |
| 14 | Sell(2) | Modify(2,,,Sell(2,OpenPrice),) | Similar to phase 3 but with respect to a Sell position. |
| 15 | Sell(2); <br> Sell(2,ProfitInPoints)>=Var3 | Sell(3,1,Bid+Var1p,Bid-Var2p\*4) | Similar to phase 4 but with respect to a Sell position. |
| 16 | Sell(3); | Modify(3,,,Sell(3,OpenPrice),) | Similar to phase 5 but with respect to a Sell position. |
| 17 | Sell(3);<br> Sell(3,ProfitInPoints)>=Var3 | Sell(4,1,Bid+Var1p,Bid-Var2p\*3) | Similar to phase 6 but with respect to a Sell position. |
| 18 | Sell(4); | Modify(4,,,Sell(4,OpenPrice),) | Similar to phase 7 but with respect to a Sell position. |
| 19 | Sell(4);<br> Sell(4,ProfitInPoints)>=Var3 | Sell(5,1,Bid+Var1p,Bid-Var2p\*2) | Similar to phase 8 but with respect to a Sell position. |
| 20 | Sell(5); | Modify(5,,,Sell(5,OpenPrice),) | Similar to phase 9 but with respect to a Sell position. |
| 21 | Sell(5);<br> Sell(5,ProfitInPoints)>=Var3 | Sell(6,1,Bid+Var1p,Bid-Var2p) | Similar to phase 10 but with respect to a Sell position. |
| 22 | Sell(6); | Modify(6,,,Sell(6,OpenPrice),) | Similar to phase 11 but with respect to a Sell position. |

Use of variables: Var1 - initial Stop Loss, Var2 - Take Profit of the last order, Var3 - profit in points at which we scale in and move Stop Loss to Breakeven.

Fig. 7 is the chart that shows the performance of this metaprogram.

![Fig. 7. Performance of the metaprogram for Pyramiding](https://c.mql5.com/2/4/Piramiding.png)

Fig. 7. Performance of the metaprogram for Pyramiding

**Reopening**

At first, we set two Limit orders. As soon as one has triggered, the other one is deleted. Then, when Stop Loss kicks in, a new position is opened until it closes at Take Profit or the maximum number of positions (5) is used up.

**Table 8. Metaprogram for Reopening**

| Phase number | Phase Identification | Actions | Comments |
| --- | --- | --- | --- |
| 1 | Nothing; <br> NoLastDeal | BuyLimit(1,1,Ask-Var1p,ThisOpenPrice-Var2p,ThisOpenPrice+Var3p);<br> SellLimit(1,1,Bid+Var1p,ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | There is no position or order in the market and the account history shows no trades on the symbol. It means that this is the very beginning of the system operation. Two Limit orders are set as the initial action. |
| 2 | Nothing; <br> LastDeal(,ProfitInValute)>0 | BuyLimit(1,1,Ask-Var1p,ThisOpenPrice-Var2p,ThisOpenPrice+Var3p);<br> SellLimit(1,1,Bid+Var1p,ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | There is no position or order in the market but the history shows a trade that closed with profit. It suggests that the previous phase is complete and we need to start from the beginning and set two Limit orders as in phase 1. |
| 3 | Nothing; <br> LastEADeal(5) | BuyLimit(1,1,Ask-Var1p,ThisOpenPrice-Var2p,ThisOpenPrice+Var3p);<br> SellLimit(1,1,Bid+Var1p,ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | There is no position or order in the market but the history contains a trade with the last identifier. In this case, profit earned in the trade is of no importance as the phase is considered to be complete anyway; we start from the beginning and set two Limit orders as in phase 1. |
| 4 | NoPos;<br> BuyLimit(1) | SellLimit(1,1,Bid+Var1p,ThisOpenPrice-Var2p,ThisOpenPrice+Var3p) | There is no position in the market but we know that there is one Limit order which means that there should also be the second one. |
| 5 | NoPos;<br> SellLimit(1) | BuyLimit(1,1,Ask-Var1p,ThisOpenPrice+Var2p,ThisOpenPrice-Var3p) | Similar to phase 4. |
| 6 | Buy(1); <br> SellLimit(1) | Delete(1) | There is a position with identifier 1. This means that either of the Limit orders has triggered and the second order should be deleted. |
| 7 | Sell(1); <br> BuyLimit(1) | Delete(1) | Similar to phase 6. |
| 8 | Nothing; <br> LastDeal(1,ProfitInValute)<=0;<br> LastEADeal(1,Direction)==1 | Buy(2,2,Ask-Var2p,Ask+Var3p) | There is no position and the last trade was unprofitable. Check the direction of the last trade executed by the Expert Advisor; if that was a Buy trade, the next position you need to open is a Buy position. |
| 9 | Nothing; <br> LastDeal(1,ProfitInValute)<=0;<br> LastEADeal(1,Direction)==-1 | Sell(2,2,Bid+Var2p,Bid-Var3p) | There is no position and the last trade was unprofitable. Check the direction of the last trade executed by the Expert Advisor; if that was a Sell trade, the next position you need to open is a Sell position. |
| 10 | Nothing; <br> LastDeal(2,ProfitInValute)<=0;<br> LastEADeal(2,Direction)==1 | Buy(3,4,Ask-Var2p,Ask+Var3p) | Similar to phase 8. |
| 11 | Nothing; <br> LastDeal(2,ProfitInValute)<=0;<br> LastEADeal(2,Direction)==-1 | Sell(3,4,Bid+Var2p,Bid-Var3p) | Similar to phase 9. |
| 12 | Nothing; <br> LastDeal(3,ProfitInValute)<=0;<br> LastEADeal(3,Direction)==1 | Buy(4,8,Ask-Var2p,Ask+Var3p) | Similar to phase 8. |
| 13 | Nothing; <br> LastDeal(3,ProfitInValute)<=0;<br> LastEADeal(3,Direction)==-1 | Sell(4,8,Bid+Var2p,Bid-Var3p) | Similar to phase 9. |
| 14 | Nothing; <br> LastDeal(4,ProfitInValute)<=0;<br> LastEADeal(4,Direction)==1 | Buy(5,16,Ask-Var2p,Ask+Var3p) | Similar to phase 8. |
| 15 | Nothing; <br> LastDeal(4,ProfitInValute)<=0;<br> LastEADeal(4,Direction)==-1 | Sell(5,16,Bid+Var2p,Bid-Var3p) | Similar to phase 9. |

Use of variables: Var1 - level from the market price at which Limit orders are set, Var2 - Stop Loss, Var3 - Take Profit.

Fig. 8 is the chart that shows the performance of this metaprogram.

![Fig. 8. Performance of the metaprogram for Reopening](https://c.mql5.com/2/4/Reopen.png)

Fig. 8. Performance of the metaprogram for Reopening

Below you can find a few more simple programs to see the operation of such functions as Trading Signals, Trailing Stop and Breakeven.

**Trading Signals**

Entry and exit are based on trading signals.

**Table 9. Metaprogram for Trading Signals"**

| Phase number | Phase Identification | Actions | Comments |
| --- | --- | --- | --- |
| 1 | Nothing; <br> SignalOpenBuy;<br> NoTradeOnBar | Buy(1,1,0,0) | There is no position or order in the market but we can see a signal to open a Buy position. There are no trades on the current bar and we open a Buy position. |
| 2 | Nothing; <br> SignalOpenSell;<br> NoTradeOnBar | Sell(1,1,0,0) | There is no position or order in the market but we can see a signal to open a Sell position. There are no trades on the current bar and we open a Sell position. |
| 3 | SignalCloseBuy; <br> Buy(1) | Close(1); | There is a Buy position and a signal to close it; the Buy position is being closed. |
| 4 | SignalCloseSell; <br> Sell(1) | Close(1); | There is a Sell position and a signal to close it; the Sell position is being closed. |

Fig. 9 is the chart that shows the performance of this metaprogram.

![Fig. 9. Performance of the metaprogram for Trading Signals](https://c.mql5.com/2/4/TradeSignals.png)

Fig. 9. Performance of the metaprogram for Trading Signals

**Trading Signals with a Trailing Stop**

**Table 10. Metaprogram for Trading Signals with a Trailing Stop**

| Phase number | Phase Identification | Actions | Comments |
| --- | --- | --- | --- |
| 1 | Nothing; <br> SignalOpenBuy;<br> NoTradeOnBar | Buy(1,1,0,0) | There is no position or order in the market but we can see a signal to open a Buy position. There are no trades on the current bar and we open a Buy position. |
| 2 | Nothing; <br> SignalOpenSell;<br> NoTradeOnBar | Sell(1,1,0,0) | There is no position or order in the market but we can see a signal to open a Sell position. There are no trades on the current bar and we open a Sell position. |
| 3 | SignalCloseBuy; <br> Buy(1) | Close(1); | There is a Buy position and a signal to close it; the Buy position is being closed. |
| 4 | SignalCloseSell; <br> Sell(1) | Close(1); | There is a Sell position and a signal to close it; the Sell position is being closed. |
| 5 | Buy(1) | TrailingStop | There is a Buy position in the market; Trailing Stop function should be activated. |
| 6 | Sell(1) | TrailingStop | There is a Sell position in the market; Trailing Stop function should be activated. |

Fig. 10 is the chart that shows the performance of this metaprogram.

![Fig. 10. Performance of the metaprogram for Trading Signals with a Trailing Stop](https://c.mql5.com/2/4/TradeSignalsTR.png)

Fig. 10. Performance of the metaprogram for Trading Signals with a Trailing Stop

**Trading Signals with Breakeven Function**

**Table 11. Metaprogram for Trading Signals with Breakeven Function**

| Phase number | Phase Identification | Actions | Comments |
| --- | --- | --- | --- |
| 1 | Nothing; <br> SignalOpenBuy;<br> NoTradeOnBar | Buy(1,1,0,0) | There is no position or order in the market but we can see a signal to open a Buy position. Since there are no trades on the current bar, the Buy position is being opened. |
| 2 | Nothing; <br> SignalOpenSell;<br> NoTradeOnBar | Sell(1,1,0,0) | There is no position or order in the market but we can see a signal to open a Sell position. Since there are no trades on the current bar, the Sell position is being opened. |
| 3 | SignalCloseBuy; <br> Buy(1) | Close(1); | There is a Buy position and a signal to close it; the Buy position is being closed. |
| 4 | SignalCloseSell; <br> Sell(1) | Close(1); | There is a Sell position and a signal to close it; the Sell position is being closed. |
| 5 | Buy(1) | BreakEven | There is a Buy position in the market; Breakeven function should be activated. |
| 6 | Sell(1) | BreakEven | There is a Sell position in the market; Breakeven function should be activated. |

Fig. 11 is the chart that shows the performance of this metaprogram.

![Fig. 11. Performance of the metaprogram for Trading Signals with Breakeven Function](https://c.mql5.com/2/4/TradeSignalsBE.png)

Fig. 11. Performance of the metaprogram for Trading Signals with Breakeven Function

### Command Interpreter

The above approach to formalizing order strategies allows us to better understand them and work out an algorithm for their further implementation in an Expert Advisor, as well as to directly interpret and follow the elaborated rules. The eInterpretator Expert Advisor has been created with this purpose in mind (see the attached files). Parameters of the Expert Advisor and their descriptions are provided in Table 12.

**Table 12. Parameters of the eInterpretator Expert Advisor**

| Parameters | Purpose |
| --- | --- |
| Lots | Volume of the order when the lot coefficient is equal to 1. |
| UserTradeDir | Direction of the trade specified by the user (it is checked at the phase identification when executing the UserBuy and UserSell commands). |
| ProgramFileName | Metaprogram file name (when working on the account). When testing or optimizing, the metaprogram should be placed into the TesterMetaProgram.txt file |
| DeInterpritate | Reverse interpretation of commands. Upon completion, a file with prefix "De\_" will appear in the Files folder and you will be able to see how the Expert Advisor "understood" the metaprogram from the ProgramFileName file. |

| **User Variables** |
| Var1 - Var20 | User variables. |
| **Trailing Stop** |
| TR\_ON | Activation of the Trailing Stop function. |
| TR\_Start | Profit of the position in points at which the Trailing Stop starts working. |
| TR\_Level | Trailing Stop level. Distance in points from the current market price to the Stop Loss. |
| TR\_Step | Step in points for the modification of the Stop Loss. |
| **Break Even** |
| BE\_ON | Activation of the Breakeven function. |
| BE\_Start | Profit of the position in points that triggers the Breakeven. |
| BE\_Level | Level to which the Stop Loss is moved when the Breakeven is triggered. The BE\_Start-BE\_Level of profit points is fixed. |
| **Open Signals** |
| OS\_ON | Activation of signals for opening. |
| OS\_Shift | Bar on which the indicators are checked: 0 - new, 1 - completed. |
| OS\_TimeFrame | Indicator time frame. |
| OS\_MA2FastPeriod | Fast MA period. |
| OS\_MA2FastShift | Fast MA shift. |
| OS\_MA2FastMethod | Fast MA method. |
| OS\_MA2FastPrice | Fast MA price. |
| OS\_MA2SlowPeriod | Slow MA period. |
| OS\_MA2SlowShift | Slow MA shift. |
| OS\_MA2SlowMethod | Slow MA method. |
| OS\_MA2SlowPrice | Slow MA price. |
| **Close Signals** |
| CS\_ON | Activation of signals for closing. |
| CS\_Shift | Bar on which the indicators are checked: 0 - new, 1 - completed. |
| CS\_TimeFrame | Indicator time frame. |
| CS\_CCIPeriod | CCI period. |
| CS\_CCIPrice | CCI price. |
| CS\_CCILevel | Upper CCI level (for closing a Buy position). A signal for closing a Buy position appears at the downward crossover of the level. It is exactly the opposite for closing a Sell position. |

**How the Expert Advisor Works**

At the beginning, the Expert Advisor loads the metaprogram from the file to review and analyze it. If any gross error is found in the metaprogram, an error alert will pop up. Analyzing the metaprogram, Expert Advisor fills in data structures with numeric values corresponding to text commands to ensure the maximum performance of the Expert Advisor. Upon successful analysis of the metaprogram, the following message is printed to the log: "Initialization of the interpreter complete".

If the DeInterpritate variable is included, the Expert Advisor will run a test reverse interpretation of commands (whereby it will get disconnected from the chart and any testing in the [Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "Strategy Tester in MetaTrader 5") carried out at that time will be aborted). When reverse interpreting, the Expert Advisor transforms numeric values found in the structures into text commands. And although command entries in the file will be different, the reverse interpreted metaprogram will allow you to have a better idea of how the Expert Advisor analyzes commands.

Let us see it using the following string from the metaprogram file:

```
Buy(6) | Modify(6,,,ThisOpenPrice-Var4p,)
```

Following the reverse interpretation this string will look as follows:

```
Buy(6)==1*1+0*0; | Modify(6,,,ThisOpenPrice()*1-0.0025*1,)
```

As we can see, a simple Buy(6) command is transformed into a comparison expression where the right side contains the arithmetic expression X1\*X2+X3\*X4 that gives 1 as a result of calculations. In the action field, the user variable is replaced with the numeric value.

### Tips on Customizing the Expert Advisor

Some of you will probably want to customize this Expert Advisor by adding your own commands to both the analysis phase and command execution and include other position management functions. Due to the structure of the Expert Advisor such customization can be quite straightforward, otherwise the entire work done on the Expert Advisor would have been of no practical value.

**Adding Data Commands**

A list of commands for getting data can be found in the InfoCommand array. Commands are arranged in columns with five commands in a row which allows us to easily count their number and find the index value for the command to be added.

After adding the command to the InfoCommand array, we add a new case corresponding to the new command index to the switch structure in the SetValue() function. To get the value, we need to first select the object from which the value will be obtained and only then get the value. Depending on the type of the object from which data is obtained, different functions are used to select the object. These functions are shown in Table 13.

**Table 13. Functions of the Expert Advisor for selecting trade objects**

| Function | Purpose and Parameters |
| --- | --- |
| Pos.Select(\_Symbol) | Position selection. Standard class method similar to the PositionSelect() function. |
| SelectOrder(long aType,string aID,bool & aSelected) | Function for selecting an order by the symbol of the Expert Advisor, type (aType) and identifier value (aID). If the object is found and selected, the aSeleted variable by reference returns true. |
| bool SelectLastDeal(int aType,bool & aSelected) | Function for selecting the last trade by the symbol of the Expert Advisor and type (aType). If the object is found and selected, the aSeleted variable by reference returns true. |
| SelectLastEADeal(int aType,string aID,bool & aSelected) | Function for selecting the last trade executed by the Expert Advisor by the symbol of the Expert Advisor and type (aType). If the object is found and selected, the aSeleted variable by reference returns true. |

The difference between the last trade and the last trade executed by the Expert Advisor is in that the last trade covers Stop Loss and Take Profit trades. Last trade data may be required to determine the result of closing the last position, while information on the last trade executed by the Expert Advisor may be necessary to identify the last direction of the trade or the Expert Advisor operation phase.

In addition to trade object data, access can be gained to market data, such as price, etc. Important is to make sure that the data can be obtained. Following an attempt to select the object, we have to make sure that the object has really been selected (by checking the value of aSelected), get the required parameter, assign its value to the Val.Value variable and return true.

Table 14 features functions that are used to get parameters of various trade objects.

**Table 14. Functions of the Expert Advisor for getting parameters of the selected trade object**

| Function | Purpose and Parameters |
| --- | --- |
| double SelPosParam(int aIndex) | Getting the position parameter by the set aIndex index. |
| double SelOrdParam(int aIndex) | Getting the order parameter by the set aIndex index. |
| double SelDealParam(int aIndex) | Getting the trade parameter by the set aIndex index. |

The identifier index of the data to be obtained is passed in the function. The index value is contained in the Val.InfoIdentifierIndex variable.

When adding a new access command, you may be required to also add the identifier of data to be obtained or only add the identifier of data to be taken.

**Adding Data Identifiers**

A list of identifiers can be found in the InfoIdentifier array. We need to add the new identifier to the array, find its index and update the SelPosParam(), SelOrdParam() and SelDealParam() functions. Updates may concern some or all functions depending on whether the new identifier can be applied to all trade objects. Function updates consist in adding a new case corresponding to the new identifier index to the switch structure.

**Adding Action Commands**

Action commands are added to the ActCommand array. Commands in the array are arranged in a string, making it a bit more difficult to find the necessary index. Elements represent a string since in addition to adding a command we need to specify the number of its parameters and type. The number of parameters is specified in the ActCmndPrmCnt array and the type is indicated in the ActCmndType array. Possible types include: 0 - market action, 1 - action with a pending order, 2 - position management.

After the command is added to the array, we find the DoAction() function and add another case for the new function call to its switch. New function must be of the bool type and return true if executed successfully or false in case of error. If the function performance check is not necessary, as in Trailing Stop function, it can simply return true.

Keep in mind that functions that deal with pending orders, such as functions for setting of the same, require preliminary checks for the order existence.

**Changes to Trading Signal Functions**

The entire work pertaining to getting trading signals in the Expert Advisor is done in two functions (two functions for signals for closing and two functions for signals for opening).

The CloseSignalsInit() function (initialization of signals for closing) and OpenSignalsInit() function (initialization of signals for opening) are called from the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit "OnInit() function") function of the Expert Advisor. These functions are responsible for loading indicators. The main functions - CloseSignalsMain() (identification of trading signals for closing) and OpenSignalsMain() (identification of trading signals for opening) are called from the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick "OnTick() function") function on every tick.

At the beginning of the function execution, GlobalCloseBuySignal, GlobalCloseSellSignal (signals for closing) and GlobalOpenBuySignal, GlobalOpenSellSignal (signals for opening) should be assigned false and then true upon the corresponding indicator readings.

Further, in the [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit "OnDeinit() function") function of the Expert Advisor, you need to execute [IndicatorRelease()](https://www.mql5.com/en/docs/series/indicatorrelease "IndicatorRelease").

### Attachments

- eInterpretator.mq5 - Expert Advisor that should be placed into MQL5/Experts of the terminal data directory.

- LimitAdd.txt - Metaprogram for Scaling in Using Limit Orders.

- StopRev.txt - Metaprogram for Stop and Reverse.

- Piramiding.txt - Metaprogram for Pyramiding.

- ReOpen.txt.txt - Metaprogram for Reopening.

- TradeSignals.txt - Metaprogram for Trading Signals.

- TradeSignalsTR.txt - Metaprogram for Trading Signals with a Trailing Stop.

- TradeSignalsBE.txt - Metaprogram for Trading Signals with Breakeven Function.

- limitadd.set - file of parameters for Scaling in Using Limit Orders.

- stoprev.set - file of parameters for Stop and Reverse.

- piramiding.set - file of parameters for Pyramiding.

- reopen.set - file of parameters for Reopening.

- tradesignals.set - file of parameters for Trading Signals.

- tradesignalstr.set - file of parameters for Trading Signals with a Trailing Stop.

- tradesignalsbe.set - file of parameters for Trading Signals with Breakeven Function.


**Note.** When using programs with Trading Signals, Trailing Stop and Breakeven, always remember to enable the corresponding functions in the Expert Advisor properties window. When testing strategies in the Strategy Tester, copy metaprogram files into the TesterMetaProgram.txt file (this is necessary to allow for the use of remote testing [agents](https://www.metatrader5.com/en/terminal/help/algotrading/testing "Agents") ). The file should be placed into MQL5/Files of the terminal data directory (you can open it from the terminal: File -> Open Data Folder).

Performance of the programs displayed in the charts that can be found in the Examples of Order Strategies in Metalanguage section is based on the parameters specified in the parameter files. Testing has been performed over the last months (as of 2012.08.29), for EURUSD H1, OHLC model on M1.

### Conclusion

Very often, the first feeling you will probably get when getting started with order strategy development is confusion - what to start with, what to keep in mind, how to ensure stability of the Expert Advisor in real environment, how to combine the execution of a trading strategy algorithm with reliability of its operation?

This article is supposed to, at the very least, help both developers and traders who place orders for the development of EAs with the initial formalization of their strategies and help them understand the stages of the strategy development, what every stage involves and what should be taken into account. The eInterpretator Expert Advisor opens up great possibilities for experimenting with order strategies with minimum time and effort.

Furthermore, I would like to say that I cannot withhold my admiration of the MetaTrader 5 terminal. The operating speed of the eInterpretator Expert Advisor in the Strategy Tester was beyond my expectations!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/495](https://www.mql5.com/ru/articles/495)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/495.zip "Download all attachments in the single ZIP archive")

[files\_\_4.zip](https://www.mql5.com/en/articles/download/495/files__4.zip "Download files__4.zip")(23.39 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Color optimization of trading strategies](https://www.mql5.com/en/articles/5437)
- [Analyzing trading results using HTML reports](https://www.mql5.com/en/articles/5436)
- [Developing the oscillator-based ZigZag indicator. Example of executing a requirements specification](https://www.mql5.com/en/articles/4502)
- [Auto search for divergences and convergences](https://www.mql5.com/en/articles/3460)
- [The Flag Pattern](https://www.mql5.com/en/articles/3229)
- [Wolfe Waves](https://www.mql5.com/en/articles/3131)
- [Universal Trend with the Graphical Interface](https://www.mql5.com/en/articles/3018)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/10197)**
(46)


![ccontipe](https://c.mql5.com/avatar/2020/4/5E8CA72C-8EAB.jpg)

**[ccontipe](https://www.mql5.com/en/users/ccontipe)**
\|
7 Apr 2020 at 13:49

Hi Dmitry!

First of all, I try to understand the metalanguage concepts that you showed here, and then implement the code. The first part that I consider successful, the idea of the code is clear, and I thank you for introducing these concepts in the article above.

The problem for me in the second part ... I did not find a way to implement metalanguage code in MT5. Is it possible to implement this? I'm not an MT5 expert yet, and any help provided here is more than welcome!

My best wishes

C!

![marirbh](https://c.mql5.com/avatar/avatar_na2.png)

**[marirbh](https://www.mql5.com/en/users/marirbh)**
\|
19 Nov 2020 at 03:44

Здравствуйте, Дмитрий!

Я нашел эту статью, потому что искал способ увеличить количество купленных контрактов, если рынок пойдет в направлении, противоположном моей работе.

Мне понравилась ваша статья, но мне не удалось протестировать советник на своем терминале. Я получаю сообщение об ошибке при открытии MetaTesterProgram.txt

Не могли бы вы немного лучше объяснить, как мне действовать, чтобы иметь возможность тестировать (в тестере стратегий).

Заранее большое спасибо!

P.S. Извините за орфографию, я бразилец, пользовался Google Переводчиком ...

![570545](https://c.mql5.com/avatar/avatar_na2.png)

**[570545](https://www.mql5.com/en/users/570545)**
\|
25 Dec 2021 at 05:20

Hello Dimitri I am new and beginner in trading more precisely metals [XAUUSD](https://www.mql5.com/en/quotes/metals/XAUUSD "XAUUSD chart: technical analysis") so I would like to learn trading strategies how to analyze the market, place an order and close my order etc..

So I am coming to you for lessons if possible.

Sincerely

GJB

![Lionel Niquet](https://c.mql5.com/avatar/2016/5/574C137D-49A4.JPG)

**[Lionel Niquet](https://www.mql5.com/en/users/lionelalien)**
\|
9 Jan 2022 at 10:37

Thank you very much. A great example of programming.


![Jose_Henrique](https://c.mql5.com/avatar/2021/9/6153663C-551D.jpg)

**[Jose\_Henrique](https://www.mql5.com/en/users/jose_henrique)**
\|
6 May 2023 at 18:00

Please help me understand, should I create a ''MetaProgram.txt'' in the archive folder with the added content of each >txt sent to the reference sets? this is how I should do it to make the EA work ??

![MetaTrader 4 and MetaTrader 5 Trading Signals Widgets](https://c.mql5.com/2/0/MetaTrader_trading_signal_widget_avatar__1.png)[MetaTrader 4 and MetaTrader 5 Trading Signals Widgets](https://www.mql5.com/en/articles/626)

Recently MetaTrader 4 and MetaTrader 5 user received an opportunity to become a Signals Provider and earn additional profit. Now, you can display your trading success on your web site, blog or social network page using the new widgets. The benefits of using widgets are obvious: they increase the Signals Providers' popularity, establish their reputation as successful traders, as well as attract new Subscribers. All traders placing widgets on other web sites can enjoy these benefits.

![Neural Networks: From Theory to Practice](https://c.mql5.com/2/0/ava_seti.png)[Neural Networks: From Theory to Practice](https://www.mql5.com/en/articles/497)

Nowadays, every trader must have heard of neural networks and knows how cool it is to use them. The majority believes that those who can deal with neural networks are some kind of superhuman. In this article, I will try to explain to you the neural network architecture, describe its applications and show examples of practical use.

![MQL5 Market Turns One Year Old](https://c.mql5.com/2/0/mql5-market-1year-avatar.png)[MQL5 Market Turns One Year Old](https://www.mql5.com/en/articles/632)

One year has passed since the launch of sales in MQL5 Market. It was a year of hard work, which turned the new service into the largest store of trading robots and technical indicators for MetaTrader 5 platform.

![MetaTrader 5 on Linux](https://c.mql5.com/2/0/linux5.png)[MetaTrader 5 on Linux](https://www.mql5.com/en/articles/625)

In this article, we demonstrate an easy way to install MetaTrader 5 on popular Linux versions — Ubuntu and Debian. These systems are widely used on server hardware as well as on traders’ personal computers.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/495&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069368509911204843)

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