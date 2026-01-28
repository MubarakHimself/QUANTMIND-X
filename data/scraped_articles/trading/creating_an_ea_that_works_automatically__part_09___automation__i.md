---
title: Creating an EA that works automatically (Part 09): Automation (I)
url: https://www.mql5.com/en/articles/11281
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:08:38.519874
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=mizyziojegrlgmlnyghsofglvkoumdyx&ssn=1769180916942749177&ssn_dr=0&ssn_sr=0&fv_date=1769180916&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11281&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20an%20EA%20that%20works%20automatically%20(Part%2009)%3A%20Automation%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918091689315496&fz_uniq=5069156085123711220&sv=2552)

MetaTrader 5 / Trading


### Introduction

In the previous article [Creating an EA that works automatically (Part 08): OnTradeTransaction](https://www.mql5.com/en/articles/11248), I explained how we can take advantage of the MetaTrader 5 platform by using a rather interesting event handling function. We will start now with building the first level of automation in our EA.

Unlike many existing mechanisms, here we will consider a mechanism that will not overload the EA or the platform. This mechanism can be used for HEDGING accounts, although it is mainly aimed at NETTING accounts.

We will start with a simple system using OCOorders. Later we will expand the functionality to provide an even more reliable and interesting system, especially for those who like to trade in very volatile markets where there is a high risk of missed orders.

### Creating Breakeven and Trailing Stop for OCO orders

If you are not familiar with this, let me explain. The OCO (One-Cancels-the-Other) order system is a system in which the take profit and stop loss are set in the order or position itself. If the order is canceled or the position is closed, then these take profit or stop loss orders are also canceled.

Take profit and stop loss orders can be removed or added at any time. For practical purposes and in order not to complicate the code, we will assume that they will always be created at the moment when the EA sends an order to the trade server, and they will cease to exist when one of the limits is reached and the position is closed.

To create a triggering mechanism, we will work with the C\_Manager class. Here we have almost everything ready to get a trigger system for breakeven and trailing stop. First, let's add a function that will generate a breakeven level for a position. The full code of the function is shown below:

```
inline void TriggerBreakeven(void)
                        {
                                if (PositionSelectByTicket(m_Position.Ticket))
                                        if (PositionGetDouble(POSITION_PROFIT) >= m_Trigger)
                                                m_Position.EnableBreakEven = (ModifyPricePoints(m_Position.Ticket, m_Position.PriceOpen, m_Position.PriceOpen, m_Position.TP) ? false : true);
                        }
```

You probably expected to see a much more complex function than the one shown here, but trust me, this simple function is capable of triggering a position breakeven level. If you do not understand how this is possible, let's analyze how this happens so that you believe that we will not need anything else, except for this simple function.

The first thing we do is call the [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) function. This function downloads all updated information about an open position. We then use the [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) function with the [POSITION\_PROFIT](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties) argument to get the latest **financial value** which was loaded by a call to PositionSelectByTicket. We compare this value with the value entered in the constructor call during class initialization.

If the value is greater than or equal to it ( **this is the trigger**), this suggests that we can reach breakeven. Then we send the price value at which the position was opened to the function in the C\_Orders class. If this interaction is successful, the class will indicate that breakeven has been reached.

This function is pretty simple, isn't it? However, its definition is contained in the private part of the code, which will be accessed by the EA. It is the trailing stop function shown below:

```
inline void TriggerTrailingStop(void)
                        {
                                double price;

                                if ((m_Position.Ticket == 0) || (m_Position.SL == 0)) return;
                                if (m_Position.EnableBreakEven) TriggerBreakeven(); else
                                {
                                        price = SymbolInfoDouble(_Symbol, (GetTerminalInfos().ChartMode == SYMBOL_CHART_MODE_LAST ? SYMBOL_LAST : (m_Position.IsBuy ? SYMBOL_ASK : SYMBOL_BID)));
                                        if (MathAbs(price - m_Position.SL) >= (m_Position.Gap * 2))
                                                ModifyPricePoints(m_Position.Ticket, m_Position.PriceOpen, (m_Position.SL + (m_Position.Gap * (m_Position.IsBuy ? 1 : -1))), m_Position.TP);
                                }
                        }
```

After the breakeven is triggered and executed, the next time the EA calls the TriggerTrailingStop function, we will check the possibility of moving (trailing) the stop loss.

But before that, pay attention to where the breakeven function is actually called. The trailing stop movement has a slightly more complex triggering mechanism than the breakeven movement. In this case, we will do it a little differently. Unlike breakeven, which does not depend on the asset, on the chart mode and on the market type, where only the profit level of the position is important, with the trailing stop we need to know two things: the type of chart and the type of position.

Some EA developers sometimes don't care about the chart plotting method used when creating a trailing stop trigger. This is the right thing to do in some cases. If the asset has a relatively high **spread** variance in history (difference between the best seller and the best buyer), even with the Last price based charting we can ignore the plotting mode. Because if the movement is within the spread, we are very likely to have problems because the stop loss order will be at the wrong point.

That is why it is so important to study and understand the asset well in order to properly develop this kind of trigger. The idea here is to capture the asset price, no matter how we do it. But we really need to have this value in hand. Once this is done, we will subtract this price from the point where the stop loss is located. In this case, it doesn't matter if we are selling or buying — the value will be automatically converted to a value **in points** instead of monetary terms. This value should be at least twice as large as the number of points which we always calculate with each update on the trading server. If it succeeds, we move the stop loss position by the number of gap points. Thus, the interval will always be the same and the cycle will start over.

This trigger mechanism works perfectly well in any situation. There is an important detail: it is incredibly light and very fast, which is very important. Think of these triggers as of a mousetrap. If the mechanism is too complicated or has a long execution time, the mouse will eventually grab the cheese and run away before the trap is triggered.

There is one more thing to understand: what should be the function or the event handler that we need to use in order to call the above functions? Many may think that the event handler should be the OnTick function. Right? Wrong. To explain this, let's move on to the next topic.

### Why shouldn't we use OnTick as a caller for triggers?

I know it can be very tempting to use the OnTick function as a way to call any other function. But this is undoubtedly the biggest mistake you can make. The correct way is to use the OnTime event as shown below:

```
//+------------------------------------------------------------------+
int OnInit()
{
        manager = new C_Manager(def_MAGIC_NUMBER, user03, user02, user01, user04, user08);
        mouse = new C_Mouse(user05, user06, user07, user03, user02, user01);
        (*manager).CheckToleranceLevel();
        EventSetMillisecondTimer(100);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        delete manager;
        delete mouse;
        EventKillTimer();
}
//+------------------------------------------------------------------+
void OnTick() { }
//+------------------------------------------------------------------+
void OnTimer()
{
        (*manager).TriggerTrailingStop();
}
//+------------------------------------------------------------------+
```

Pay attention that in the OnInit event we define a short value so that the OnTime event is generated. We do this using the [EventSetMillisecondTimer](https://www.mql5.com/en/docs/eventfunctions/eventsetmillisecondtimer) function. In most cases, you can set any value, starting from 50 milliseconds. But you must also remember to call the [EventKillTimer](https://www.mql5.com/en/docs/eventfunctions/eventkilltimer) function inside the OnDeInit event. This way the EA will release the OnTime event, not making the MetaTrader 5 platform keep generating the event. Even if the platform manages to solve this problem and will stop generating the event, it is good practice to release the system. So, in this case, every 100 milliseconds, or about 10 per second we will have the verification of our trigger shown in the topic above.

But that does not answer the question: Why shouldn't we use the OnTick event handler function as the trigger caller? In fact, the above explanation does not answer this question. But let's consider it in more detail to understand why we shouldn't do this.

The OnTick event is triggered every tick the server trades. If you look at the bar chart, even on a 1-minute chart, you will not have the real picture of what is happening. For this, it will be necessary to go down to the **HFT** level ( _**High Frequency Trading**_), better known as institutional robots.

Many people don't realize that this is not possible for an EA that runs in a platform which is installed on a computer hundreds of kilometers away from the trading server. They do not have the slightest chance to be compared to an HFT running on a dedicated server a few meters away from the server. This is because of the so called **latency**, which is the time required for the data to reach the server. Those who play online games will understand what I'm talking about, but for those who are not familiar, I will say that the latency is much higher than the frequency of operations performed by a trading server. In other words, **it is impossible to reach a performance** comparable with HFT.

Returning to the topic of the OnTick event, we can say that in just 1 millisecond, the server can trigger more than 10 such events. If so many events reach the platform to trigger all the events that the server has triggered, then as soon as your EA starts working, the platform will crash. This is because it will be totally busy handling many more events than its processor can handle. This means that no matter how everything is organized, it will not be able to execute the required actions due to a large number of calls to the same function, which in this case is the trigger of the trailing stop.

It may happen so that this will not actually occur for a while, and the platform will continue to run smoothly. However, once the system detects that we are in position and starts checking the trigger, we can run into serious problems, as the probability of the platform crashing is very high.

This probability can be even greater if the volatility of an asset grows rapidly. This is especially common for assets such as index futures. So don't try to use OnTick as a way to fire triggers. Forget about this EA event handler. It's not meant to be used by us mere mortals, but it is for those HFTs to be able to monitor the trade server.

So, the most appropriate way to check the trigger system is to use the OnTime event. It is configured so that the checks are done in a low enough time, so that it does not suffer a loss of quality, while providing the possibility to check all the triggers. This way we get a safe, robust, quiet and reliable EA.

There is another issue related to the triggering of trailing stop events. Some EA systems actually do not breakeven before the trailing stop is triggered. They use a trailing stop from the very moment a position is opened. This may seem a little strange, but in fact the whole point is that the operating system or configuration already has this concept running. In this case, the system will be very similar to the one shown above, but with a difference in the trailing distance used. However, the concept itself will remain the same.

In addition to this model, there is another one in which we do not actually have a trailing stop or breakeven. We have another kind of system, which guarantees an exit in case the market starts going against our position. It is a different type of trigger, and we will consider it in another article.

### Trailing Stop and Breakeven using a pending order

The system presented above works very well if you want to have a position based on the OCO order system. However, there is one problem with this OCO order system, which is the volatility. When the volatility is very high, orders may be skipped, which can lead to unfavorable situations.

There is a **"solution"** to this problem, which is to use a pending order for a guaranteed exit. However, this "solution" cannot be implemented in a hedging system, i.e. it cannot be used with the presented EA. If we try to use it on a hedging account, the EA will be kicked off the chart, because it will be making a serious mistake for the C\_Manager class. However, we can make some adjustments in the C\_Manager class to allow the EA to have two open positions in opposite directions on the hedging account. It is possible to do the same thing, which I will show how to do, in case you are using a netting account.

But if you think about it, there is no sense in opening a position in the opposite direction on a netting account. If the order server captures the pending order, it is better to send the server a request to close both positions. This will make the EA on a hedging account behave as it is on a netting account. It will close the open position using a pending order.

Do you like this idea? Before you get into the implementation, there are a few details and questions to consider.

First, a pending order will not necessarily be executed at the point you specified. A pending order will go after the price wherever it is. Another problem is that if there is a serious failure in the EA and the position or order is canceled or closed, the other leg may be left without a counterparty.

But there is also an advantage: if everything goes well and the connection with the server is lost, a pending order in the opposite direction with the same volume will close the position (netting accounts) or will lock the price (hedging accounts).

As you can see, this method has its advantages and disadvantages.

To use it, we need to make some changes to the C\_Manager class. These changes are easy to understand and can be reverted if you don't want to use them. However, there is one important detail here: If you are going to use this method, be careful not to delete pending orders that have been placed by the EA. If you do this, you will be left without the leg responsible for closing.

For this system, it is not recommended to allow it to be modified by the user once the code has been compiled. If it is not suitable, you can disconnect it from the EA. It should remain on all the time or off all the time. To do this, we will create a new definition in the code of the C\_Manager class. See below:

```
//+------------------------------------------------------------------+
#define def_MAX_LEVERAGE                10
#define def_ORDER_FINISH                false
//+------------------------------------------------------------------+
```

When this definition is set to **true**, the system will use the stop method via a pending order. This means that you will no longer have a take profit point, as this will complicate things a lot and may cause the opposite leg to be unavailable. If this parameter is set to **false**, you will use the breakeven and the trailing stop method which we considered in the initial topic of this article. This way you will be using the OCO order system. I will leave it false by default. If you want to use the method explained in this topic, change this value from **false** to **true** and compile the EA.

Now we need to modify the functions that place market orders or create pending orders. They will look like this:

```
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                if ((m_StaticLeverage >= def_MAX_LEVERAGE) || (m_TicketPending > 0) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                m_TicketPending = C_Orders::CreateOrder(type, Price, (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceStop), (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;

                                if ((m_StaticLeverage >= def_MAX_LEVERAGE) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                tmp = C_Orders::ToMarket(type, (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceStop), (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                        }
//+------------------------------------------------------------------+
```

We need to modify these parts so that the take profit and stop loss prices are not actually created. When we do so, the server will understand that these prices will not be created and you will have an order without profit and loss limits. You might be terrified that you are sending a market order which does not actually contain a stop loss or placing a pending order without a stop loss either. But don't worry. To understand what is really going on, you should try the system. Once this is done, we need to9 make the following modification:

```
                void PendingToPosition(void)
                        {
                                ResetLastError();
                                if ((m_bAccountHedging) && (m_Position.Ticket > 0))
                                {
                                        if (def_ORDER_FINISH)
                                        {
                                                if (ClosePosition(m_Position.Ticket)) ZeroMemory(m_Position.Ticket);
                                                ClosePosition(m_TicketPending);
                                        }else SetUserError(ERR_Unknown);
                                }else m_Position.Ticket = (m_Position.Ticket == 0 ? m_TicketPending : m_Position.Ticket);
                                m_TicketPending = 0;
                                if (_LastError != ERR_SUCCESS) UpdatePosition(m_Position.Ticket);
                                CheckToleranceLevel();
                        }
```

Here, when the EA informs that the pending order has turned into a position, we may have problems if we are on a hedging account, and we already have an open position. In this case, if the parameter is set to false, an error message will be generated. If true is set, we send a request to close the original position and reset its memory area. We will also close the newly opened position. Note that if the connection with the server is good, we will be able to successfully close both positions thus zeroing out our exposure in the market.

Now we need the system to create a pending order which will be used as a closing order, i.e. as a stop loss, which will remain in the order book. This point is perhaps the most critical in the entire system, since if it is not well planned, you will have serious problems with the account which will be left without a stop level. Because of this, you have to be aware of what the EA will be doing by checking the following:

![Figure 01](https://c.mql5.com/2/48/001__6.png)

Figure 01. The point where we see pending orders and open positions

The window shown in Figure 01 must always be open so that you can analyze what actions the EA performs on the server. You should not blindly trust an EA, no matter how reliable it may seem.

Always monitor the EA's intentions. So, for the EA to have a stop loss point, the C\_Manager class will have to create a pending order. It is created as follows:

```
                void UpdatePosition(const ulong ticket)
                        {
                                int ret;
                                double price;

                                if ((ticket == 0) || (ticket != m_Position.Ticket)) return;
                                if (PositionSelectByTicket(m_Position.Ticket))
                                {
                                        ret = SetInfoPositions();
                                        if (def_ORDER_FINISH)
                                        {
                                                price = m_Position.PriceOpen + (FinanceToPoints(m_InfosManager.FinanceStop, m_Position.Leverage) * (m_Position.IsBuy ? -1 : 1));
                                                if (m_TicketPending > 0) if (OrderSelect(m_TicketPending))
                                                {
                                                        price = OrderGetDouble(ORDER_PRICE_OPEN);
                                                        C_Orders::RemoveOrderPendent(m_TicketPending);
                                                }
                                                m_TicketPending = C_Orders::CreateOrder(m_Position.IsBuy ? ORDER_TYPE_SELL : ORDER_TYPE_BUY, price, 0, 0, m_Position.Leverage, m_InfosManager.IsDayTrade);
                                        }
                                        m_StaticLeverage += (ret > 0 ? ret : 0);
                                }else
				{
					ZeroMemory(m_Position);
                                	if (def_ORDER_FINISH)
					{
						RemoveOrderPendent(m_TicketPending);
						m_TicketPending = 0;
					}
				}
                                ResetLastError();
                        }
```

First we check if the order system is set to the value of 'true'. If this condition is met, we calculate the price point to start the assembly and use the pending order as a position stop. Next we check if we have an already placed order. If this is true, we capture the price where it is and remove the order from the book. One way or another, we will try to create a new pending order. If everything is correct, we will have a pending order in the order book which will serve as a stop loss order.

Attention: In case the position is closed, the pending order must be canceled. This can be done using the following code lines.

Now we need to change the **SetInfoPosition** function to have a correct indication of whether or not we need to make a breakeven or whether we can start with a trailing stop right away. The new function is as follows:

```
inline int SetInfoPositions(void)
                        {
                                double v1, v2;
                                int tmp = m_Position.Leverage;

                                m_Position.Leverage = (int)(PositionGetDouble(POSITION_VOLUME) / GetTerminalInfos().VolMinimal);
                                m_Position.IsBuy = ((ENUM_POSITION_TYPE) PositionGetInteger(POSITION_TYPE)) == POSITION_TYPE_BUY;
                                m_Position.TP = PositionGetDouble(POSITION_TP);
                                v1 = m_Position.SL = PositionGetDouble(POSITION_SL);
                                v2 = m_Position.PriceOpen = PositionGetDouble(POSITION_PRICE_OPEN);
                                if (def_ORDER_FINISH) if (m_TicketPending > 0) if (OrderSelect(m_TicketPending)) v1 = OrderGetDouble(ORDER_PRICE_OPEN);
                                m_Position.EnableBreakEven = (def_ORDER_FINISH ? m_TicketPending == 0 : m_Position.EnableBreakEven) || (m_Position.IsBuy ? (v1 < v2) : (v1 > v2));
                                m_Position.Gap = FinanceToPoints(m_Trigger, m_Position.Leverage);

                                return m_Position.Leverage - tmp;
                        }
```

Here we make a sequence of tests to capture the price at which the pending order is located. If the pending order ticket has not yet been created for some reason, it will cause the breakeven indicator to be properly started for our purpose.

So far, neither breakeven nor trailing stop can be actually used in the system in which a pending order is used as a stop point. To implement this system, we do not really need the trigger, since it is already configured: we just need to implement a system for moving pending orders. For the breakeven, to implement the movement, we will add the following:

```
inline void TriggerBreakeven(void)
                        {
                                double price;

                                if (PositionSelectByTicket(m_Position.Ticket))
                                        if (PositionGetDouble(POSITION_PROFIT) >= m_Trigger)
                                        {
                                                price = m_Position.PriceOpen + (GetTerminalInfos().PointPerTick * (m_Position.IsBuy ? 1 : -1));
                                                if (def_ORDER_FINISH)
                                                {
                                                        if (m_TicketPending > 0) m_Position.EnableBreakEven = !ModifyPricePoints(m_TicketPending, price, 0, 0);
                                                }else m_Position.EnableBreakEven = !ModifyPricePoints(m_Position.Ticket, m_Position.PriceOpen, price, m_Position.TP);
                                        }
                        }
```

In the previous version of this function responsible for executing breakeven, the stop order price was exactly equal to the opening price of the position. Many traders always like to leave with some profit, even a small one. Therefore, I decided to modify the function so that when the breakeven triggers, the EA places an order shifted by 1 tick from the opening price. Thus, the trader will receive at least 1 tick of profit.

This system works for any asset or market type, as we use data from the asset itself to know where the stop line should be.

Once this is done, we check if we are using the stop model based on a pending order. If this is so, we check if we have any value in the pending order variable. If there is such a value, we send a request to change the point, at which the order was placed, to a new position. If we are using a system based on OCO order, we will use the method described above. These checks may seem pointless, but they prevent us from sending invalid requests to the server.

Now let's see how the trailing stop function works in this case:

```
inline void TriggerTrailingStop(void)
                        {
                                double price, v1;

                                if ((m_Position.Ticket == 0) || (def_ORDER_FINISH ? m_TicketPending == 0 : m_Position.SL == 0)) return;
                                if (m_Position.EnableBreakEven) TriggerBreakeven(); else
                                {
                                        price = SymbolInfoDouble(_Symbol, (GetTerminalInfos().ChartMode == SYMBOL_CHART_MODE_LAST ? SYMBOL_LAST : (m_Position.IsBuy ? SYMBOL_ASK : SYMBOL_BID)));
                                        v1 = m_Position.SL;
                                        if (def_ORDER_FINISH) if (OrderSelect(m_TicketPending)) v1 = OrderGetDouble(ORDER_PRICE_OPEN);
                                        if (v1 > 0) if (MathAbs(price - v1) >= (m_Position.Gap * 2))
                                        {
                                                price = v1 + (m_Position.Gap * (m_Position.IsBuy ? 1 : -1));
                                                if (def_ORDER_FINISH) ModifyPricePoints(m_TicketPending, price, 0, 0);
                                                else ModifyPricePoints(m_Position.Ticket, m_Position.PriceOpen, price, m_Position.TP);
                                        }
                                }
                        }
```

This function may seem strange at first glance, since I used things that are unusual for most people. Let's see what's going on here. First we check if we have any open position, this is the easiest part. But the strange thing here is the ternary operator which is quite unusual to be used here. We separate here whether the check is made by the pending ticket or by the stop loss price value. This comparison will be used in conjunction with position comparison. If the tests indicate that nothing needs to be done, we will simply return to the caller.

If something needs to be done, we test the breakeven. If it has already been executed, we start to capture the current asset price in order to check whether we can activate the trailing stop. Since we may need the price at which the pending order is placed, and the factoring will be the same, we start the temporary variable with the possible value that could be used as the stop loss line. But it may happen so that we are using a pending order as a stop point. In this case, it must move so we capture the price where it is. We carry out the factoring to check whether we should move the stop level. If this is possible, we adjust the price at which the order will move and move the pending order or the stop loss line accordingly. What will actually move will depend on the system we are using.

The video below demonstrates this system in operation. For those who imagine that this is something different or non-functional, watch the video and draw your own conclusions. Although the best thing to do in order to understand what's going on is to compile the EA and do your own tests on a DEMO account. In this way, the understanding of the entire system will be more solid and clear.

Demonstração 09 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11281)

MQL5.community

1.91K subscribers

[Demonstração 09](https://www.youtube.com/watch?v=8zEkavfnU6E)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 2:57

•Live

•

Video 01. Demonstration of the Stop system via pending order.

### Conclusion

In this article, I covered the simplest trigger system that can be built into an EA that many people like or want to have in their EA. However, this system is not suitable for use if you want to use the EA in a portfolio setup. In this case, I would advise using a manual system. But that's just my advice, because I don't know exactly how you're going to use this knowledge.

The attachment contains the full code which you can study and learn more about this type of mechanism. In this code, you will initially have an EA that will use a stop line. To use a pending order for the stop, you will need to modify the EA as described in this article.

Now we have the minimum required automation to implement the rest of the steps. In the next article we will look at how to create a 100% automated EA. Good luck

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11281](https://www.mql5.com/pt/articles/11281)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11281.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_09.zip](https://www.mql5.com/en/articles/download/11281/ea_automatico_-_09.zip "Download EA_Automatico_-_09.zip")(9.27 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/446349)**
(10)


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
22 Jun 2023 at 21:07

**Mackilem take profit orders, are they considered separate orders or are they part of a single "structured position"? In other words, when the position is closed, will the trading server always remove the SL and TP or should I worry about orphaned orders?**
**This question arose when in your article you comment on the risk of the "lame leg" and also when you comment that in order to activate the trailing stop in a hedging account we need to allow 2 open positions.**

**Cheers**

Doubts are part of it. There's no shame in asking. It's shameful to maintain doubts and spread information without knowledge.

But let's go in parts. Let's first understand one situation and then the other.

->When you place an order on the server, or open a position, you can do this in two ways: When you send the order with the stop and take already set on the order or position to be opened. In this case you only send a request to the server. If you haven't put the stop or take in the order, you can do this later by adjusting things. In any case, you will only have one order or position on the server. This is what many people call an OCO order. In other words, when the stop or take is executed, the position will be closed and everything will be as you probably already know when it comes to OCO orders. I don't think you have any doubts about this.

->Now we have a problem, which I've addressed in this same series of articles. An OCO order or position does not indicate that your Take or Stop will not be skipped. They will only be executed if a trade occurs at that specific price. To prevent the order from being skipped, some programmers don't use OCO orders, they do something a little different. It's this something different that tends to generate those lame legs, i.e. you have an order listed in the book, but it's not covered by any other order. In this case, we are using at least two orders to control the position. By doing this, we avoid the skipped stop, because even if the price skips the OCO order, it won't skip the order in the book, which is there precisely to close the open position in MARRA ... However, this doesn't work on the HEDGING type of account, because such accounts allow you to hold a buy and a sell position at the same time, on the same asset. It's almost like a BINARY OPTION ... but when this technique of using two orders is carried out on a NETTING account, the position is closed. But you have to be careful to avoid keeping a loose order in the book.

To understand this, you really need to read all 15 articles in the series and try out the automatic system I demonstrate. But do this on [demo accounts](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 documentation: Account Properties"), both on FOREX and on the stock market. Then you'll really understand what I'm talking about. Don't try to understand it just by imagining how it should or could work. Test the system on demo accounts, both on FOREX, which uses HEDGING accounts, and on BOLSA, which uses NETTING accounts ... 😁👍

![Mackilem](https://c.mql5.com/avatar/avatar_na2.png)

**[Mackilem](https://www.mql5.com/en/users/mackilem)**
\|
22 Jun 2023 at 22:36

**Daniel Jose [#](https://www.mql5.com/pt/forum/437656#comment_47704385):**

Doubts are part of it. There's no shame in asking. It's shameful to maintain doubt and spread information without knowledge.😁

But let's go in parts. Let's first understand one situation and then the other.

->When you place an order on the server, or open a position, you can do this in two ways: When you send the request with the stop and take already set on the order or position to be opened. In this case you only send a request to the server. If you haven't put the stop or take in the order, you can do this later by adjusting things. In any case, you will only have one order or position on the server. This is what many people call an OCO order. In other words, when the stop or take is executed, the position will be closed and everything will be as you probably already know when it comes to OCO orders. I don't think you have any doubts about this.

->Now we have a problem, which I've addressed in this same series of articles. An OCO order or position does not indicate that your Take or Stop will not be skipped. They will only be executed if a trade occurs at that specific price. To prevent the order from being skipped, some programmers don't use OCO orders, they do something a little different. It's this something different that tends to generate those lame legs, i.e. you have an order listed in the book, but it's not covered by any other order. In this case, we are using at least two orders to control the position. By doing this, we avoid the skipped stop, because even if the price skips the OCO order, it won't skip the order in the book, which is there precisely to close the open position in MARRA ... However, this doesn't work on the HEDGING type of account, because such accounts allow you to hold a buy and a sell position at the same time, on the same asset. It's almost like a BINARY OPTION ... but when this technique of using two orders is carried out on a NETTING account, the position is closed. But you have to be careful to avoid keeping a loose order in the book.

To understand this, you really need to read all 15 articles in the series and try out the automatic system I demonstrate. But do this on demo accounts, both on FOREX and on the stock market. Then you'll really understand what I'm talking about. Don't try to understand it just by imagining how it should or could work. Test the system on demo accounts, both on FOREX, which uses HEDGING accounts, and on BOLSA, which uses NETTING accounts ... 😁👍

Thank you very much for your explanation. I've understood it well now.

Yes, I've just finished reading all the articles and I'll start testing soon.

Cheers

![kinghussle](https://c.mql5.com/avatar/avatar_na2.png)

**[kinghussle](https://www.mql5.com/en/users/kinghussle)**
\|
10 Jul 2023 at 22:44

Hello, I've been following along and implementing your EA but ive run into an error I cannot get to the bottom of. Can you assist with the errors. The errors are:

'C\_ManagerAce.mqh'C\_ManagerAce.mqh

'C\_Orders.mqh'C\_Orders.mqh

'C\_Terminal.mqh'C\_Terminal.mqh

'C\_Terminal::GetTerminalInfos' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::GetTerminalInfos'C\_Terminal.mqh

'C\_Terminal::FinanceToPoints' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::FinanceToPoints'C\_Terminal.mqh

'C\_Terminal::GetTerminalInfos' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::GetTerminalInfos'C\_Terminal.mqh

'C\_Terminal::FinanceToPoints' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::FinanceToPoints'C\_Terminal.mqh

'C\_Terminal::GetTerminalInfos' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::GetTerminalInfos'C\_Terminal.mqh

I used the same exact code as you becuase im learning programming once again. I have a bachelors in computer science. I dont understand why I get these errors when I compile my code and I have the code as you and yours compile. Help Please

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
11 Jul 2023 at 16:37

**kinghussle [#](https://www.mql5.com/en/forum/446349#comment_48033783):**

Hello, I've been following along and implementing your EA but ive run into an error I cannot get to the bottom of. Can you assist with the errors. The errors are:

'C\_ManagerAce.mqh'C\_ManagerAce.mqh

'C\_Orders.mqh'C\_Orders.mqh

'C\_Terminal.mqh'C\_Terminal.mqh

'C\_Terminal::GetTerminalInfos' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::GetTerminalInfos'C\_Terminal.mqh

'C\_Terminal::FinanceToPoints' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::FinanceToPoints'C\_Terminal.mqh

'C\_Terminal::GetTerminalInfos' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::GetTerminalInfos'C\_Terminal.mqh

'C\_Terminal::FinanceToPoints' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::FinanceToPoints'C\_Terminal.mqh

'C\_Terminal::GetTerminalInfos' - cannot access private member functionC\_ManagerAce.mqh

see declaration of function 'C\_Terminal::GetTerminalInfos'C\_Terminal.mqh

I used the same exact code as you becuase im learning programming once again. I have a bachelors in computer science. I dont understand why I get these errors when I compile my code and I have the code as you and yours compile. Help Please

The errors you are reporting are due to the attempt to access something private to the class, outside the body of the class. I suggest you start with something a little simpler first. For this, first try to understand what private clauses and public clauses are. But mainly, why use one or the other.

![K Marcos](https://c.mql5.com/avatar/2024/3/65EA0F8A-3465.jpg)

**[K Marcos](https://www.mql5.com/en/users/kmarcoscoder)**
\|
27 Apr 2024 at 04:14

Hi Daniel, good evening! Congratulations on the articles, they're excellent study material. Aren't you thinking of recording these lessons for YouTube? I'd love to watch you.


![Creating an EA that works automatically (Part 10): Automation (II)](https://c.mql5.com/2/50/aprendendo_construindo_010_avatar.png)[Creating an EA that works automatically (Part 10): Automation (II)](https://www.mql5.com/en/articles/11286)

Automation means nothing if you cannot control its schedule. No worker can be efficient working 24 hours a day. However, many believe that an automated system should operate 24 hours a day. But it is always good to have means to set a working time range for the EA. In this article, we will consider how to properly set such a time range.

![Experiments with neural networks (Part 4): Templates](https://c.mql5.com/2/52/neural_network_experiments_004_avatar.png)[Experiments with neural networks (Part 4): Templates](https://www.mql5.com/en/articles/12202)

In this article, I will use experimentation and non-standard approaches to develop a profitable trading system and check whether neural networks can be of any help for traders. MetaTrader 5 as a self-sufficient tool for using neural networks in trading. Simple explanation.

![Population optimization algorithms: Saplings Sowing and Growing up (SSG)](https://c.mql5.com/2/52/growing-tree-avatar.png)[Population optimization algorithms: Saplings Sowing and Growing up (SSG)](https://www.mql5.com/en/articles/12268)

Saplings Sowing and Growing up (SSG) algorithm is inspired by one of the most resilient organisms on the planet demonstrating outstanding capability for survival in a wide variety of conditions.

![How to create a custom indicator (Heiken Ashi) using MQL5](https://c.mql5.com/2/54/heikin_ashi_avatar.png)[How to create a custom indicator (Heiken Ashi) using MQL5](https://www.mql5.com/en/articles/12510)

In this article, we will learn how to create a custom indicator using MQL5 based on our preferences, to be used in MetaTrader 5 to help us read charts or to be used in automated Expert Advisors.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/11281&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069156085123711220)

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