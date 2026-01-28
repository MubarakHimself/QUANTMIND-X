---
title: MetaTrader 5 features hedging position accounting system
url: https://www.mql5.com/en/articles/2299
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:18:46.277930
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=fopwhgcwkaevyyxtyxaviudhjgvivvoh&ssn=1769181525027506850&ssn_dr=0&ssn_sr=0&fv_date=1769181525&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2299&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MetaTrader%205%20features%20hedging%20position%20accounting%20system%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918152518465217&fz_uniq=5069333634776761172&sv=2552)

MetaTrader 5 / Trading


The MetaTrader 5 platform was originally designed for trading within the netting position accounting system. The netting system allows having only one position per financial instrument meaning that all further operations at that instrument lead only to closing, reversal or changing the volume of the already existing position. In order to expand possibilities of retail Forex traders, we have added the second accounting system — hedging. Now, it is possible to have multiple positions per symbol, including oppositely directed ones. This paves the way to implementing trading strategies based on the so-called "locking" — if the price moves against a trader, they can open a position in the opposite direction.

Since the new system is similar to the one used in MetaTrader 4, it will be familiar to traders. At the same time, traders will be able to enjoy all the advantages of the fifth platform version — filling orders using multiple deals (including partial fills), multicurrency and multithreaded tester with support for [MQL5 Cloud Network](https://cloud.mql5.com/ "https://cloud.mql5.com/"), and much more.

Now, you can use one account to trade the markets that adhere to the netting system and allow having only one position per instrument, and use another account in the same platform to trade Forex and apply hedging.

This article describes the netting and hedging systems in details, as well as sheds light on the changes related to the implementation of the second accounting system.

### Position accounting depends on a trading account

A position accounting system is set at an account level and displayed in the terminal window header and the Journal:

![Position accounting system on the current account](https://c.mql5.com/2/22/terminal__1.png)

To open a demo account with hedging, enable the appropriate option:

![Opening the demo account with hedging](https://c.mql5.com/2/22/open_acc__3.png)

To open a real account with hedging, contact your broker.

### Netting system

With this system, you can have only one common position for a symbol at the same time:

- If there is an open position for a symbol, executing a deal in the same direction increases the volume of this position.
- If a deal is executed in the opposite direction, the volume of the existing position can be decreased, the position can be closed (when the deal volume is equal to the position volume) or reversed (if the volume of the opposite deal is greater than the current position).

It does not matter, what has caused the opposite deal — an executed market order or a triggered pending order.

The below example shows execution of two EURUSD Buy deals 0.5 lots each:

![Netting](https://c.mql5.com/2/22/netting_positions__2.png)

Execution of both deals resulted in one common position of 1 lot.

### Hedging system

With this system, you can have multiple open positions of one and the same symbol, including opposite positions.

If you have an open position for a symbol, and execute a new deal (or a pending order triggers), a new position is additionally opened. Your current position does not change.

The below example shows execution of two EURUSD Buy deals 0.5 lots each:

![Hedging](https://c.mql5.com/2/22/hedging_positions__2.png)

Execution of these deals resulted in opening two separate positions.

### Impact of the system selected

Depending on the position accounting system, some of the platform functions may have different behavior:

- [Stop Loss and Take Profit inheritance](https://www.metatrader5.com/en/terminal/help/trading/general_concept#sltp_inherit "https://www.metatrader5.com/en/terminal/help/trading/general_concept#sltp_inherit") rules change.
- To close a position in the netting system, you should perform an opposite trading operation for the same symbol and the same volume. To close a position in the hedging system, explicitly select the "Close Position" command in the context menu of the position.
- A position cannot be reversed in the hedging system. In this case, the current position is closed and a new one with the remaining volume is opened.
- In the hedging system, a new condition for margin calculation is available — Hedged margin.

### New trade operation type - Close By

The new trade operation type has been added for hedging accounts — closing a position by an opposite one. This operation allows closing two oppositely directed positions at a single symbol. If the opposite positions have different numbers of lots, only one order of the two remains open. Its volume will be equal to the difference of lots of the closed positions, while the position direction and open price will match (by volume) the greater of the closed positions.

Compared with a single closure of the two positions, the closing by an opposite position allows traders to save one spread:

- In case of a single closing, traders have to pay a spread twice: when closing a buy position at a lower price (Bid) and closing a sell position at a higher one (Ask).
- When using an opposite position, an open price of the second position is used to close the first one, while an open price of the first position is used to close the second one.

![Closing the position by the opposite one](https://c.mql5.com/2/22/closeby_en__1.png)

In the latter case, a "close by" order is placed. Tickets of closed positions are specified in its comment. A pair of opposite positions is closed by two "out by" deals. Total profit/loss resulting from closing the both positions is specified only in one deal.

!["Close by" operation in History](https://c.mql5.com/2/22/history__1.png)

### Margin calculation in the hedging system of position accounting

If the hedging position accounting system is used, the margin is calculated using the same [formulas and principles](https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex "https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex") as described above. However, there are some additional features for multiple positions of the same symbol.

**Positions/orders open in the same direction**

Their volumes are summed up and the weighted average open price is calculated for them. The resulting values are used for calculating margin by the formula corresponding to the symbol type.

For pending orders (if the margin ratio is non-zero), margin is calculated separately.

**Opposite positions/orders**

Oppositely directed open positions of the same symbol are considered hedged or covered. Two margin calculation methods are possible for such positions. The calculation method is determined by the broker.

![Hedged margin calculation settings in the contract specification](https://c.mql5.com/2/22/spec__2.png)

| Basic calculation | Using the larger leg |
| --- | --- |
| Used if "calculate using larger leg" is not specified in the "Hedged margin" field of contract specification.<br>The calculation consists of several steps:<br>- For uncovered volume<br>- For covered volume (if hedged margin size is specified)<br>- For pending orders<br>The resulting margin value is calculated as the sum of margins calculated at each step.<br>**Calculation for uncovered volume**<br>- Calculation of the total volume of all positions and market orders for each of the legs — buy and sell.<br>- Calculation of the weighted average position and market order open price for each leg: (open price of position or order 1 \* volume of position or order 1 + ... + open price of position or order N \* volume of position or order N) / (volume of position or order 1 + ... + volume of position or order N).<br>- Calculation of uncovered volume (smaller leg volume is subtracted from the larger one).<br>- The calculated volume and weighted average price are used then to calculate margin by the appropriate formula corresponding to the symbol type.<br>- The weighted average value of the ratio and rate is used when taking into account the margin ratio and converting margin currency to deposit currency.<br>**Calculation for covered volume**<br>Used if the "Hedged margin" value is specified in a contract specification. In this case margin is charged for hedged, as well as uncovered volume.<br>If the initial margin is specified for a symbol, the hedged margin is specified as an absolute value (in monetary terms).<br>If the initial margin is not specified (equal to 0), the contract size is specified in the "Hedged" field. The margin is calculated by the appropriate formula in accordance with the type of the financial instrument, using the specified contract size. For example, we have two positions Buy EURUSD 1 lot and Sell EURUSD 1 lot, the contract size is 100,000. If the value of 100,000 is specified in the "Hedged field", the margin for the two positions will be calculated as per 1 lot. If you specify 0, no margin is charged for the hedged (covered) volume.<br>Per each hedged lot of a position, the margin is charged in accordance with the value specified in the "Hedged Margin" field in the contract specification:<br>- Calculation of hedged volume for all open positions and market orders (uncovered volume is subtracted from the larger leg).<br>- Calculation of the weighted average position and market order open price: (open price of position or order 1 \* volume of position or order 1 + ... + open price of position or order N \* volume of position or order N) / (volume of position or order 1 + ... + volume of position or order N).<br>- The calculated volume, weighted average price and the hedged margin value are used then to calculate margin by the appropriate formula corresponding to the symbol type.<br>- The weighted average value of the ratio and rate is used when taking into account the margin ratio and converting margin currency to deposit currency.<br>**Calculation for pending orders**<br>- Calculation of margin for each pending order type separately (Buy Limit, Sell Limit, etc.).<br>- The weighted average value of the ratio and rate for each pending order type is used when taking into account the margin ratio and converting margin currency to deposit currency. | Used if "calculate using larger leg" is specified in the "Hedged margin" field of contract specification.<br>- Calculation of margin for shorter and longer legs for all open positions and market orders.<br>- Calculation of margin for each pending order type separately (Buy Limit, Sell Limit, etc.).<br>- Summing up a longer leg margin: long positions and market orders + long pending orders.<br>- Summing up a shorter leg margin: short positions and market orders + short pending orders.<br>- The largest one of all calculated values is used as the final margin value. |

### Changes in MQL5

Now, each position has its unique ticket. It usually corresponds to the ticket of an order used to open the position. A ticket is assigned automatically to all available positions after the terminal update.

When modifying or closing a position in the hedging system, make sure to specify its ticket (MqlTradeRequest::ticket). You can specify a ticket in the netting system as well, however positions are identified by a symbol name.

**MqlTradeRequest**

[MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) features two new fields:

- position — position ticket. Fill it when changing and closing a position for its clear identification. It usually matches the ticket of an order used to open the position.
- position\_by — opposite position ticket. It is used when closing a position by an opposite one (opened at the same symbol but in the opposite direction).


```
struct MqlTradeRequest
  {
   ENUM_TRADE_REQUEST_ACTIONS    action;           // Performed action type
   ulong                         magic;            // Expert Advisor magic number
   ulong                         order;            // Order ticket
   string                        symbol;           // Symbol name
   double                        volume;           // Requested deal volume in lots
   double                        price;            // Price
   double                        stoplimit;        // Stop Limit order level
   double                        sl;               // Stop Loss order level
   double                        tp;               // Take Profit order level
   ulong                         deviation;        // Maximum allowable deviation from the requested price
   ENUM_ORDER_TYPE               type;             // Order type
   ENUM_ORDER_TYPE_FILLING       type_filling;     // Order filling type
   ENUM_ORDER_TYPE_TIME          type_time;        // Order time type
   datetime                      expiration;       // Order expiration date (for ORDER_TIME_SPECIFIED type orders)
   string                        comment;          // Order comment
   ulong                         position;         // Position ticket
   ulong                         position_by;      // Opposite position ticket
  };
```

**MqlTradeTransaction**

[MqlTradeTransaction](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) also features the two similar fields:

- position — ticket of a position affected by transaction. It is filled for transactions related to handling market orders (TRADE\_TRANSACTION\_ORDER\_\* except TRADE\_TRANSACTION\_ORDER\_ADD, where a position ticket is not assigned yet) and order history (TRADE\_TRANSACTION\_HISTORY\_\*).

- position\_by — opposite position ticket. It is used when closing a position by an opposite one (opened at the same symbol but in the opposite direction). It is filled only for orders closing a position by an opposite one (close by) and deals closing by an opposite one (out by).


```
struct MqlTradeTransaction
  {
   ulong                         deal;             // Deal ticket
   ulong                         order;            // Order ticket
   string                        symbol;           // Symbol name
   ENUM_TRADE_TRANSACTION_TYPE   type;             // Transaction type
   ENUM_ORDER_TYPE               order_type;       // Order type
   ENUM_ORDER_STATE              order_state;      // Order state
   ENUM_DEAL_TYPE                deal_type;        // Deal type
   ENUM_ORDER_TYPE_TIME          time_type;        // Order time type
   datetime                      time_expiration;  // Order expiration date
   double                        price;            // Price
   double                        price_trigger;    // Stop limit order trigger price
   double                        price_sl;         // Stop Loss level
   double                        price_tp;         // Take Profit level
   double                        volume;           // Volume in lots
   ulong                         position;         // Position tickets
   ulong                         position_by;      // Opposite position tickets
  };
```

**PositionGetTicket**

The new [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) function returns a position ticket by an index in the list of open positions and automatically selects that position for further work using the [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble), [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger), and [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) functions.

```
ulong  PositionGetTicket(
   int  index      // index in the list of positions
   );
```

**PositionSelectByTicket**

The new [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) function selects an open position for further work by a specified ticket.

```
bool  PositionSelectByTicket(
   ulong   ticket     // position ticket
   );
```

**PositionSelect**

[PositionSelect](https://www.mql5.com/en/docs/trading/positionselect) selects a position by a symbol name for further work using the [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble), [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger), and [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) functions. In the hedging system (where there can be multiple positions at a single symbol), the function selects a position with the lowest ticket.

**ACCOUNT\_MARGIN\_MODE**

The new property [ACCOUNT\_MARGIN\_MODE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation) allows receiving the mode of margin calculation and position accounting on a trading account:

| Identifier | Description |
| --- | --- |
| ACCOUNT\_MARGIN\_MODE\_RETAIL\_NETTING | Used for the over-the-counter market when accounting positions in the netting mode (one position per symbol). Margin calculation is based on a symbol type ( [SYMBOL\_TRADE\_CALC\_MODE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants)). |
| ACCOUNT\_MARGIN\_MODE\_EXCHANGE | Used on the exchange markets. Margin calculation is based on the discounts specified in symbol settings. Discounts are set by the broker, however they cannot be lower than the exchange set values. |
| ACCOUNT\_MARGIN\_MODE\_RETAIL\_HEDGING | Used for the over-the-counter market with independent position accounting (hedging, there can be multiple positions at a single symbol). Margin calculation is based on a symbol type (SYMBOL\_TRADE\_CALC\_MODE). The presence of multiple positions at a single symbol is considered. |

**SYMBOL\_MARGIN\_HEDGED**

The new property [SYMBOL\_MARGIN\_HEDGED](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) allows receiving the value of a hedged margin by a trading symbol. Margin calculation in the hedging system of position accounting has been described [above](https://www.mql5.com/en/articles/2299#margin).

**New trading constants**

Due to the addition of the new Close By operation type, the new trading properties have appeared as well:

- TRADE\_ACTION\_CLOSE\_BY — new [trading operation type](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions) — close a position by an opposite one.
- ORDER\_TYPE\_CLOSE\_BY — new [order type](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#ENUM_ORDER_TYPE) — close a position by an opposite one.
- ORDER\_POSITION\_BY\_ID — new [order propertiy](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties) — ticket of an opposite position used for closing the current one.
- DEAL\_ENTRY\_OUT\_BY — new [deal type](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_type) — close a position by an opposite one.

### Extra bonus — hedging and MQL5 Cloud Network

Now, you can use MetaTrader 5 to trade both stock markets and the popular retail Forex with hedging. Developers of the automated systems applying hedging have received another important advantage. Apart from the multithreaded tester, the entire computing capacity of the [MQL5 Cloud Network](https://www.mql5.com/en/articles/341) is at their disposal now.

Update your platform and try the new features!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2299](https://www.mql5.com/ru/articles/2299)

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
**[Go to discussion](https://www.mql5.com/en/forum/77142)**
(66)


![Vladimir M.](https://c.mql5.com/avatar/2018/8/5B7FEFA5-D317.png)

**[Vladimir M.](https://www.mql5.com/en/users/rosomah)**
\|
4 Feb 2018 at 21:03

**Vladimir Karputov:**

There is a dependence on which trade server you log in to. Connect to MetaQuotes-Demo.

It worked on MetaQuotes-Demo. Thank you.


![Sergiy Riehl](https://c.mql5.com/avatar/2017/12/5A3BEC56-73D2.png)

**[Sergiy Riehl](https://www.mql5.com/en/users/riehl)**
\|
13 Mar 2018 at 12:29

Help who knows, can't find the information on my own. The broker has hedging accounts. My robot is written for non-hedging accounts. How can I [close a position](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal") on a hedging account using MQL5? There is no OrderClose() function in MQL5. The opposite position, as in a netting account, does not close the open position.


![Sergiy Riehl](https://c.mql5.com/avatar/2017/12/5A3BEC56-73D2.png)

**[Sergiy Riehl](https://www.mql5.com/en/users/riehl)**
\|
13 Mar 2018 at 12:38

**Sergiy Riehl:**

Help who knows, can't find the information on my own. The broker has hedging accounts. My robot is written for non-hedging accounts. How can I [close a position](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal") on a hedging account using MQL5? There is no OrderClose() function in MQL5. The opposite position, as in a netting account, does not close the open position.

Maybe TRADE\_ACTION\_CLOSE\_BY  should be set in the trade request on a hedging account ?

or is there a more correct solution?

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
13 Mar 2018 at 12:43

**Sergiy Riehl:**

Help who knows, can't find the information on my own. The broker has hedging accounts. My robot is written for non-hedging accounts. How can I [close a position](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal") on a hedging account using MQL5? There is no OrderClose() function in MQL5. The opposite position, as in a netting account, does not close the open position.

Use the universal code for position traversal - it works on both netting and hedge accounts. Example in the [GalacticExplosion](https://www.mql5.com/en/code/19655) code - CloseAllPositions function

```
//+------------------------------------------------------------------+
//| Close all positions|
//+------------------------------------------------------------------+
void CloseAllPositions()
  {
   for(int i=PositionsTotal()-1;i>=0;i--) // returns the number of current positions
      if(m_position.SelectByIndex(i))     // selects the position by index for further access to its properties
         if(m_position.Symbol()==m_symbol.Name() && m_position.Magic()==m_magic)
            m_trade.PositionClose(m_position.Ticket()); // close a position by the specified symbol
  }
```

![QuickPip](https://c.mql5.com/avatar/2020/1/5E0F75F6-6854.png)

**[QuickPip](https://www.mql5.com/en/users/bromelio)**
\|
26 Mar 2020 at 17:07

**Carl Schreiber:**

What about the commission?

If I have two open positions, one buy, one sell, and close the 'sell' by the 'buy' I have paid twice the commission. But I would have paid only once the commission if I just close the buy, isn't it?

But what if a broker isn't asking for a commission but has increased the spread? Don't I pay the spread twice as well?

What about Carl's concern about paying the commission twice? Any answers from MetaQuotes, any experiences, please?


![Graphical Interfaces II: Setting Up the Event Handlers of the Library (Chapter 3)](https://c.mql5.com/2/22/Graphic-interface-part2__2.png)[Graphical Interfaces II: Setting Up the Event Handlers of the Library (Chapter 3)](https://www.mql5.com/en/articles/2204)

The previous articles contain the implementation of the classes for creating constituent parts of the main menu. Now, it is time to take a close look at the event handlers in the principle base classes and in the classes of the created controls. We will also pay special attention to managing the state of the chart depending on the location of the mouse cursor.

![Universal Expert Advisor: the Event Model and Trading Strategy Prototype (Part 2)](https://c.mql5.com/2/21/smyf67hqftm_kaz2.png)[Universal Expert Advisor: the Event Model and Trading Strategy Prototype (Part 2)](https://www.mql5.com/en/articles/2169)

This article continues the series of publications on a universal Expert Advisor model. This part describes in detail the original event model based on centralized data processing, and considers the structure of the CStrategy base class of the engine.

![Universal Expert Advisor: Custom Strategies and Auxiliary Trade Classes (Part 3)](https://c.mql5.com/2/21/02fe0hhenus_a0y2.png)[Universal Expert Advisor: Custom Strategies and Auxiliary Trade Classes (Part 3)](https://www.mql5.com/en/articles/2170)

In this article, we will continue analyzing the algorithms of the CStrategy trading engine. The third part of the series contains the detailed analysis of examples of how to develop specific trading strategies using this approach. Special attention is paid to auxiliary algorithms — Expert Advisor logging system and data access using a conventional indexer (Close\[1\], Open\[0\] etc.)

![Area method](https://c.mql5.com/2/21/area.png)[Area method](https://www.mql5.com/en/articles/2249)

The "area method" trading system works based on unusual interpretation of the RSI oscillator readings. The indicator that visualizes the area method, and the Expert Advisor that trades using this system are detailed here. The article is also supplemented with detailed findings of testing the Expert Advisor for various symbols, time frames and values of the area.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ctsbtcoylszdhhlopgtowdhpjlnzoihi&ssn=1769181525027506850&ssn_dr=0&ssn_sr=0&fv_date=1769181525&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2299&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MetaTrader%205%20features%20hedging%20position%20accounting%20system%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918152518413525&fz_uniq=5069333634776761172&sv=2552)

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