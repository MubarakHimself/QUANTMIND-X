---
title: Creating an EA that works automatically (Part 12): Automation (IV)
url: https://www.mql5.com/en/articles/11305
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:08:06.166080
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/11305&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069146082144878804)

MetaTrader 5 / Trading


### Introduction

In the previous article [Creating an EA that works automatically (Part 11): Automation (III)](https://www.mql5.com/en/articles/11293), we looked at how we can create a robust system, minimizing the failures and loopholes that can affect a program.

Sometimes I see people saying that the chance of something happening is one in 500,000, but even if the percentage is low, the possibility still exists. Since we are aware that this can happen, why not create the means to minimize damage or side effects in case this happens? Why ignore what might happen and not fix it or prevent it in some way just because the chances are low?

If you are following this small series of articles about how to automate an EA, you must have noticed that creating an EA for manual use is something quite fast and simple. But for a 100% automated EA, it's not that simple. You may have noticed that I never expressed the idea that we would have a 100% foolproof system which could be used without any kind of supervision. In fact, I believe that it has become quite clear, exactly the opposite of this premise that many have about automatic EA, thinking that you can turn it on and leave the thing rolling there, without really understanding what it is doing.

When we talk about EA 100% automated, everything gets serious and complicated. Even more so due to the fact that we will always be subject to runtime errors and we have to make a system that will work in real time. These two things together, allied to the fact that we may have some breach or failure in the system, makes the job extremely tiring for those who will be supervising the EA during its operation.

But you as a programmer should always look at some key points that can generate potential problems, even when everything seems to be working in perfect harmony. I do not mean that you should be looking for problems where they may not exist. This is what a careful and attentive professional will actually do — look for flaws in a system that at first sight has no flaws.

Our EA at the current stage of development, intended for manual and semi-automated use (using breakeven and trailing stop) does not have such a destructive bug as **Blockbuster** ( _DC supervillains_). However, if we use it 100% automatically, the situation changes and there is a risk of a potentially dangerous failure.

In the previous article, I raised this question and left it for you to understand where this flaw was, and how it could cause problems, so that we could not automate our EA at 100% yet. Did you manage to understand where the failure was and how it could have been triggered? Well, if the answer is no, it is ok. Not everyone can actually notice the failure just by looking at the code and using it manually, or semi-automatically. But if you try to automate the code, you will have serious problems. This flaw is certainly the simplest one to be observed, but not so simple to be corrected, which would be required for a 100% automated EA.

So, to understand what it's about, let's divide things into topics. I think it will be easier for you to notice something seemingly unimportant that can cause you great annoyances.

### Understanding the problem

The problem starts when we set the maximum volume limit that the EA can trade on a daily basis. Do not confuse this maximum daily volume with the operation volume. Now we are primarily interested in the maximum daily volume.

For clarity, let's assume that the volume is 100 times the minimum volume. That is, the EA will be able to make as many operations as possible until this volume is reached. So, the last rule added in the C\_Manager class is that the volume does not exceed these 100.

Now let's see what actually happens. To do this, we will analyze the code that allows us to trade:

```
inline bool IsPossible(const bool IsPending)
                        {
                                if (!CtrlTimeIsPassed()) return false;
                                if ((m_StaticLeverage >= m_InfosManager.MaxLeverage) || (m_bAccountHedging && (m_Position.Ticket > 0))) return false;
                                if ((IsPending) && (m_TicketPending > 0)) return false;
                                if (m_StaticLeverage + m_InfosManager.Leverage > m_InfosManager.MaxLeverage)
                                {
                                        Print("Request denied, as it would violate the maximum volume allowed for the EA.");
                                        return false;
                                }

                                return true;
                        }
```

The above code prevents the volume from being exceeded. But how does it actually happen?

Let's say the trader launched the EA with the lot multiplier 3 times the minimum required volume, and the maximum volume defined in the EA code is 100 times (this is done during code compilation). This was explained in previous articles. After 33 trades, the EA will reach 99 times the minimum trading volume, which means that we can take one more trade. However, due to the highlighted line in the code above, the trader must change the volume by 1 time in order to reach the maximum limit. Otherwise, the EA will not be able to execute the operation.

The idea is to limit the maximum volume so that the EA does not lose more than the previously specified parameter (_this must always be the main and most important concern_). Since if the EA does not open a position with a volume much higher than the stipulated, we will still have losses, but these may somehow be controlled.

But you might think: I don't see any flaws in this code. In fact, there is no flaw in this code. The functions that use it (shown below) are able to limit the volume traded by the EA to the specified maximum limit.

```
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                if (!IsPossible(true)) return;
                                m_TicketPending = C_Orders::CreateOrder(type, Price, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;

                                if (!IsPossible(false)) return;
                                tmp = C_Orders::ToMarket(type, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                        }
//+------------------------------------------------------------------+
```

But then where is the mistake? Actually, it's not that easy to understand. To understand the problem, we need to think about how the EA might be working not to send orders. Because this code of the C\_Manager class will manage to prevent the EA from triggering. The problem happens when we combine different order types. At this point, we will have a trigger of **block smashing** problem. There are some ways to limit, or rather avoid this type of trigger. They are as follow:

- Choose the order sending type or model. In this case, the EA would rely only on market or pending orders. This type of solution is suitable for some automated trading models, since there are cases in which carrying out operations solely and exclusively on the market is more suitable and more usual. But in this case, we would have limited types of trading systems. But the idea here is to show how to create a system that can cover the largest possible number of cases, without worrying about which order type to use.
- Another way to avoid the trigger is to account in a deeper way, what will already be entered as a position, and what will make changes to the position (in this case, pending orders). This solution is a better choice, however, it has some aggravating factors which make programming complicated.
- Another way (which we are going to implement) is the one that uses the bases of the system already developed, so that we will account for what is being done, but we will not make assumptions about what can be done. But even so, we will try to somehow make the EA predict the volume that will be traded during the entire period of a day. This way we can avoid having a trigger without limiting the types of trading systems the EA can support.

Now, considering the fact that the problem happens when we have the combination of orders, let's understand how things really happen. Returning to the example of lot multiplier of 3, where 33 operations have already taken place while the daily limit is 100 time, everything would be in perfect control if the EA worked only with market orders. But if, for whatever reason, we have a pending order on the trade server, the situation is different. If this pending order adds another 3 units to the minimum volume, then as soon as it is activated, it will have exceeded the maximum volume allowed in the EA.

You can imagine that these 2 units of volume which exceeded the 100 limit is not much, but it's not that simple. Think about the case where trader configures the EA to trade 50 lot. We can execute 2 market orders this exceeding the 100 limit. After sending a market order, the EA sends a pending order to increase the position volume. Now it will have reached volume 100. But for whatever reason this order has not been executed yet, as it is still a pending order. At a certain moment, the EA decides that it can send another market order of the same 50 lot volume.

The C\_Manager class immediately notices that the EA has reached the daily limit, as it will have 100 units. But the C\_Manager class, despite being aware that there is a pending order, does not really know what to do with it. In this case, when the order is executed on the server, the EA will exceed the rule of 100 and will have a volume of 150 units. Do you understand the problem? Some time ago we placed a lock to prevent the EA from hanging too many orders on the book or from opening too many positions with a certain volume. This block was broken by the simple fact that the trigger which automates the EA did not foresee that this could happen. There was an assumption that the C\_Manager class would be able to hold the EA, preventing it from exceeding the defined maximum volume. But the class failed because we combined pending orders with market orders.

Many programmers would simply work around this problem by using only market orders or only pending ones. But this does not make the problem disappear. Each new automated EA will have to go through the same testing and analysis mode in order to prevent the trigger from being fired.

Although the problem described above can be observed even when using the system manually, the probability that a trader will make this mistake is much less. The trader would be to blame, and not the protection system. But for a 100% automated system, this is completely unacceptable. So, for this and for some other reasons, **NEVER** leave a 100% automated EA running without supervision, even if you have programmed it. You may be an excellent programmer, yet it is by no means advisable to leave it alone.

If you think I'm talking nonsense, consider the following. There are autopilot systems in planes, which allow it to take off, travel the route and land without any human intervention. But even so, there is always a qualified pilot in the cabin to operate the plane. Why do you think this happens? Industry would not spend fortunes in order to develop an autopilot only to have to train a pilot to operate the plane. This would make no sense, unless the industry itself did not rely on autopilot. Think a little about this.

### Fixing the crash

In fact, just showing the error does not fix it. It takes a little more than this. But the fact that we know the failure and understand how it is triggered and its consequences, means that we can actually try to generate some kind of solution.

The fact that we trust that the C\_Manager class will be able to avoid violating the volume makes all the difference. To solve the problem, we will need to do a few things in the code. First, we will add a new variable to the system:

```
                struct st01
                {
                        ulong   Ticket;
                        double  SL,
                                TP,
                                PriceOpen,
                                Gap;
                        bool    EnableBreakEven,
                                IsBuy;
                        uint    Leverage;
                }m_Position, m_Pending;
                ulong   m_TicketPending;
```

This new variable will help us to solve the volume problem by generating some kind of forecast. Because of its appearance, another point of the code was deleted.

The addition of the new variable is followed by a series of changes. But I will only emphasize the new parts. You can see all the modifications in more detail in the attached code. The first thing to do is to create a routine to capture pending order data:

```
inline void SetInfoPending(void)
                        {
                                ENUM_ORDER_TYPE eLocal = (ENUM_ORDER_TYPE) OrderGetInteger(ORDER_TYPE);

                                m_Pending.Leverage = (uint)(OrderGetDouble(ORDER_VOLUME_CURRENT) / GetTerminalInfos().VolMinimal);
                                m_Pending.IsBuy = ((eLocal == ORDER_TYPE_BUY) || (eLocal == ORDER_TYPE_BUY_LIMIT) || (eLocal == ORDER_TYPE_BUY_STOP) || (eLocal == ORDER_TYPE_BUY_STOP_LIMIT));
                                m_Pending.PriceOpen = OrderGetDouble(ORDER_PRICE_OPEN);
                                m_Pending.SL = OrderGetDouble(ORDER_SL);
                                m_Pending.TP = OrderGetDouble(ORDER_TP);
                        }
```

This is different from the routine that captures the position data. The main difference is that we check whether we are buying or selling. This is done by checking the possibility of types which indicate a buy order. The rest of the function is self-explanatory.

We need another new function:

```
                void UpdatePending(const ulong ticket)
                        {
                                if ((ticket == 0) || (ticket != m_Pending.Ticket) || (m_Pending.Ticket == 0)) return;
                                if (OrderSelect(m_Pending.Ticket)) SetInfoPending();
                        }
```

It updates the data when the pending order receives any update from the server and passes the information to the EA.

In order for the EA to be able execute the above call, we need to add a new event to the OnTradeTransaction handler:

```
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
{
        switch (trans.type)
        {
                case TRADE_TRANSACTION_POSITION:
                        manager.UpdatePosition(trans.position);
                        break;
                case TRADE_TRANSACTION_ORDER_DELETE:
                        if (trans.order == trans.position) (*manager).PendingToPosition();
                        else (*manager).UpdatePosition(trans.position);
                        break;
                case TRADE_TRANSACTION_ORDER_UPDATE:
                        (*manager).UpdatePending(trans.order);
                        break;
                case TRADE_TRANSACTION_REQUEST: if ((request.symbol == _Symbol) && (result.retcode == TRADE_RETCODE_DONE) && (request.magic == def_MAGIC_NUMBER)) switch (request.action)
                        {
                                case TRADE_ACTION_DEAL:
                                        (*manager).UpdatePosition(request.order);
                                        break;
                                case TRADE_ACTION_SLTP:
                                        (*manager).UpdatePosition(trans.position);
                                        break;
                                case TRADE_ACTION_REMOVE:
                                        (*manager).EraseTicketPending(request.order);
                                        break;
                        }
                        break;
        }
}
```

The highlighted line above will call the C\_Manager class.

So, let's go back to the C\_Manager class to continue implementing the solution to the volume problem.

We need to create a correction system, as described in this section, in order to be able to bring the system to a more adequate level of security. The error that we noticed and that we ignored for a long time, is responsible for updating the open volume. This wouldn't affect the manual system, but it is fatal for an automated system: Therefore, to fix this error, we have to add the following line of code:

```
inline void LoadPositionValid(void)
                        {
                                ulong value;

                                for (int c0 = PositionsTotal() - 1; (c0 >= 0) && (_LastError == ERR_SUCCESS); c0--)
                                {
                                        if ((value = PositionGetTicket(c0)) == 0) continue;
                                        if (PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
                                        if (PositionGetInteger(POSITION_MAGIC) != GetMagicNumber()) continue;
                                        if ((m_bAccountHedging) && (m_TicketPending > 0))
                                        {
                                                C_Orders::ClosePosition(value);
                                                continue;
                                        }
                                        if (m_Position.Ticket > 0) SetUserError(ERR_Unknown); else
                                        {
                                                m_Position.Ticket = value;
                                                SetInfoPositions();
                                                m_StaticLeverage = m_Position.Leverage;
                                        }
                                }
                        }
```

Designing a fully automatic system is a challenging task. For a manual or semi-automatic system, the absence of the above line does not make the slightest difference. But for an automated system, any failure, no matter how small, could mean the possibility of catastrophe. Even more so if the trader does not supervise the EA or does not know what it is actually doing. This will definitely make you lose money in the market.

The next step is to modify the order sending and market requesting functions. We need them to be able to return a value to the caller while also being able to inform the caller about what is happening. Here is the code:

```
//+------------------------------------------------------------------+
                bool CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                bool bRet = false;

                                if (!IsPossible(true)) return bRet;
                                m_Pending.Ticket = C_Orders::CreateOrder(type, Price, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                if (m_Pending.Ticket > 0) bRet = OrderSelect(m_Pending.Ticket);
                                if (bRet) SetInfoPending();

                                return bRet;
                        }
//+------------------------------------------------------------------+
                bool ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;
                                bool bRet = false;

                                if (!IsPossible(false)) return bRet;
                                tmp = C_Orders::ToMarket(type, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                                if (m_Position.Ticket > 0) bRet = PositionSelectByTicket(m_Position.Ticket);
                                if (!bRet) ZeroMemory(m_Position);

                                return bRet;
                        }
//+------------------------------------------------------------------+
```

Pay attention that I'm using a test to check if the ticket is still valid. The reason is that in a NETTING account, you can close a position by sending a volume equal to the position. In this case, if the position was closed, we need to remove data from it for better security and reliability of the system. This is for a 100% automated EA, while for a manual EA such things are unnecessary.

Next, we need to add a way for the EA to know what volume to send. If you want to reverse the position in the current system, you can do this, however it will take two calls, or rather, two request submissions to the server, instead of a single request. Currently you need to close the position, and then send the request to open a new position. Knowing about the open volume, the EA can know which volume to send.

```
const uint GetVolumeInPosition(void) const
                        {
                                return m_Position.Leverage;
                        }
```

This simple code above is enough to implement this. But there is no way in the class code to actually reverse the position.

To implement this, we will again modify the order sending functions. But pay attention to the fact that this is not something that a manual or semi-automated EA needs to have. We are making these changes because we need these means to produce an automated EA. Besides, it's interesting to put some kind of message in some points, so that the trader who is supervising the EA can find out what the EA is actually doing. Even though we are not doing this in the demonstration code, you should seriously consider doing such a thing. Because being blind, just watching the chart, won't actually be enough in order to notice some strange EA behavior.

After implementing all these changes, we reached our key point which will indeed solve the problem we are dealing with. We added a way for the EA to be able to send any volume to the server. This will only be done in two calls, in which the C\_Manager class will give access to the EA. Thus, we fix the problem where the EA could end up using a traded volume higher than indicated in the code. Now we will start to predict the volume, which will possibly enter or exit the position that will be or has already been placed.

The new call code can be seen below:

```
//+------------------------------------------------------------------+
                bool CreateOrder(const ENUM_ORDER_TYPE type, const double Price, const uint LeverageArg = 0)
                        {
                                bool bRet = false;

                                if (!IsPossible(type, (LeverageArg > 0 ? LeverageArg : m_InfosManager.Leverage))) return bRet;
                                m_Pending.Ticket = C_Orders::CreateOrder(type, Price, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                if (m_Pending.Ticket > 0) bRet = OrderSelect(m_Pending.Ticket);
                                if (bRet) SetInfoPending();

                                return bRet;
                        }
//+------------------------------------------------------------------+
                bool ToMarket(const ENUM_ORDER_TYPE type, const uint LeverageArg = 0)
                        {
                                ulong tmp;
                                bool bRet = false;

                                if (!IsPossible(type, (LeverageArg > 0 ? LeverageArg : m_InfosManager.Leverage))) return bRet;
                                tmp = C_Orders::ToMarket(type, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                                if (m_Position.Ticket > 0) bRet = PositionSelectByTicket(m_Position.Ticket);
                                if (!bRet) ZeroMemory(m_Position);

                                return bRet;
                        }
//+------------------------------------------------------------------+
```

In this code, we added that argument, but pay note that its default value is zero. If you do not specify a value when declaring the function, it will use the value specified during the constructor call.

But no handling will be done at this step, since the analysis of what the C\_Manager class should execute or allow, is common for both placing a pending order and for sending a request to execute a market order. To find out which value the EA will expect to be enabled by C\_Manager, we use a small test with a ternary operator to correctly populate the value. Now let's see what we need to add to the function to correctly set up the test:

```
inline bool IsPossible(const ENUM_ORDER_TYPE type, const uint Leverage)
                        {
                                int i0, i1;

                                if (!CtrlTimeIsPassed()) return false;
                                if ((m_StaticLeverage >= m_InfosManager.MaxLeverage) || (m_bAccountHedging && (m_Position.Ticket > 0))) return false;
                                if ((m_Pending.Ticket > 0) || (Leverage > INT_MAX)) return false;
                                i0 = (int)(m_Position.Ticket == 0 ? 0 : m_Position.Leverage) * (m_Position.IsBuy ? 1 : -1);
                                i1 = i0 + ((int)(Leverage * (type == ORDER_TYPE_BUY ? 1 : -1)));
                                if (((i1 < i0) && (i1 >= 0) && (i0 > 0)) || ((i1 > i0) && (i1 <= 0) && (i0 < 0))) return true;
                                if ((m_StaticLeverage + MathAbs(i1)) > m_InfosManager.MaxLeverage)
                                {
                                        Print("Request denied, as it would violate the maximum volume allowed for the EA.");
                                        return false;
                                }

                                return true;
                        }
```

There is something that you should be aware of when programming the firing mechanism, in order to prevent requests from being denied. If there is a pending order, it must be removed before executing any order to modify the volume. If this is not done, the request will be denied. Why? That seems absolutely strange. So, it really deserves an explanation.

The reason is that there is no way to actually know if the EA will allow the pending order to be executed or not. But if the open volume changes, and the system uses a pending order as a way to stop the position, it is important that this order is removed before changing the volume. Anyway, that order will be removed, so that a new one containing the correct volume will be placed.

So, it is clear that the C\_Manager class demands that there is no pending order when changing the open volume. Another reason for this is that if there is a long position and you reverse it, this will be done by placing an order with a volume greater than the open volume. If the pending order, which was initially a stop order, continues to exist, we may have a problem when the order is executed. The volume will increase further. So we have one more reason why C\_Manager will demand to have no pending orders when changing the volume.

I hope now it is clear why the class requires deleting a pending order when changing the volume. Do not remove this line, because the EA will not send requests to the server.

Now we have a rather strange calculation and an even more strange test that can end up giving you a hell of a headache if you try to understand it without actually understanding the calculation. And at the end, there is another test which at first glance doesn't make any sense.

Let's carefully analyze this moment so that everyone can actually understand this complete mathematical insanity. To make it easier, let's see some examples. So be very careful when reading each of the examples, to be able to understand all the highlighted code.

**Example 1:**

Assume that there is no open position, then the **i1** value will be equal to the absolute value contained in the **Leverage** variable; this is the simplest case of all. So, it does not matter whether we open a position or place an order in the order book. If the sum of the i1 with the EA's accumulated value is less than the specified maximum, the request will be sent to the server.

**Example 2:**

Suppose we have a short position with volume X and we have no pending orders. In this situation, we can have several different scenarios:

- If the EA sends an order to sell volume Y, then the **i1** value will be the sum of X and Y. In this case, the order may not be filled because we increase the short position.
- If the EA sends an order to buy volume Y, and it is less than volume X, then the **i1** value will be equal to a difference between X and Y. In this case, the order will pass as the short position will be reduced.
- If the EA's order is to buy Y, while Y is equal to X, then **i1** will be zero. In this case, the order will pass as the short position will be closed.
- If the EA sends an order to buy volume Y, and it is greater than volume X, then the **i1** value will be equal to a difference between X and Y. In this case, the order may not be allowed because we turn the position into a long one.

The same is true for a long position. However, the EA's request will also be modified, so that in the end we will have the same behavior, indicated in the **i1** variable. Note that in the test, in which we check if the sum of **i1** and of the value accumulated by the EA is less than the allowed limit, we call the [MathAbs](https://www.mql5.com/en/docs/math/mathabs) function. We do this because in some cases, we will have the negative **i1**. But we need it to be positive for the test to run properly.

But we still have one last problem to be solved. When we reverse the position, the trading volume update will not occur correctly. So, to solve this, we need to make a small change in the analysis system. This change is shown below:

```
inline int SetInfoPositions(void)
                        {
                                double v1, v2;
                                uint tmp = m_Position.Leverage;
                                bool tBuy = m_Position.IsBuy;

                                m_Position.Leverage = (uint)(PositionGetDouble(POSITION_VOLUME) / GetTerminalInfos().VolMinimal);
                                m_Position.IsBuy = ((ENUM_POSITION_TYPE) PositionGetInteger(POSITION_TYPE)) == POSITION_TYPE_BUY;
                                m_Position.TP = PositionGetDouble(POSITION_TP);
                                v1 = m_Position.SL = PositionGetDouble(POSITION_SL);
                                v2 = m_Position.PriceOpen = PositionGetDouble(POSITION_PRICE_OPEN);
                                if (m_InfosManager.IsOrderFinish) if (m_Pending.Ticket > 0) v1 = m_Pending.PriceOpen;
                                m_Position.EnableBreakEven = (m_InfosManager.IsOrderFinish ? m_Pending.Ticket == 0 : m_Position.EnableBreakEven) || (m_Position.IsBuy ? (v1 < v2) : (v1 > v2));
                                m_Position.Gap = FinanceToPoints(m_Trigger, m_Position.Leverage);

                                return (int)(tBuy == m_Position.IsBuy ? m_Position.Leverage - tmp : m_Position.Leverage);
                        }
```

First, we save the position in which the system was before the update. After that, we do not interfere in anything until in the end, until we check if there was any change in the position direction. The idea is to check whether we remain long or go short. The same applies in the opposite case. If there is a change, we will return the open volume. Otherwise, we perform the calculation normally in order to know which volume is currently open. This way, we can properly know and update the volume that the EA has already traded.

### Final conclusions

Although the current system is much more complicated because we need to ensure that we do not have problems with the automated system, you must have noticed that there are almost no messages printed in the terminal, based on which the trader could monitor the EA's actions. Which is pretty important.

However, since I'm only showing here how you can create a 100% automated system, I don't know at which points it would be more appropriate for you to know what is going on inside the EA. Different information can be more or less important for different people. But I don't advise anyone to simply get the code and launch it directly into a real account, without doing several tests and adjustments, in order to know exactly what will be happening.

In the next article, I will show how to set up and launch the EA so that it is able to operate autonomously. So until the article comes out, in which I provide the explanation, try to study this article. Try to understand how simple measures can solve some issues, which are often much more complicated than it seems.

I'm not trying to show you the ultimate way to create an automated EA. I am just showing one of many possible ways that can be used. I also show the risks associated with using an automated EA, so that you know how to minimize or reduce them to an acceptable level.

One last tip for those who might be trying to use the EA, manually or with some automation. If the system is not allowing to send orders, saying the volume is violating the usage rule, all you need to do is change the following value:

```
int OnInit()
{
        string szInfo;

        manager = new C_Manager(def_MAGIC_NUMBER, user03, user02, user01, user04, user08, false, 10);
        mouse = new C_Mouse(user05, user06, user07, user03, user02, user01);
        for (ENUM_DAY_OF_WEEK c0 = SUNDAY; c0 <= SATURDAY; c0++)
```

The value is 10 by default. But it can happen, especially in the FOREX market, that you need to use a larger value. Many normally use 100. So, you should use a much larger value than 10. For example, if you want to use 10 times the usual volume of 100, enter a value of 1000. So the code will look like this:

```
int OnInit()
{
        string szInfo;

        manager = new C_Manager(def_MAGIC_NUMBER, user03, user02, user01, user04, user08, false, 1000);
        mouse = new C_Mouse(user05, user06, user07, user03, user02, user01);
        for (ENUM_DAY_OF_WEEK c0 = SUNDAY; c0 <= SATURDAY; c0++)
```

Thus, you will be able to send up to 10 orders with the specified volume before the EA blocks the sending of new orders.

In the video below, you can see the system in its current configuration.

YouTube

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11305](https://www.mql5.com/pt/articles/11305)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11305.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_12.zip](https://www.mql5.com/en/articles/download/11305/ea_automatico_-_12.zip "Download EA_Automatico_-_12.zip")(11.29 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/446859)**
(1)


![Pancras Muiruri](https://c.mql5.com/avatar/avatar_na2.png)

**[Pancras Muiruri](https://www.mql5.com/en/users/81f82ead)**
\|
22 May 2023 at 02:38

**MetaQuotes:**

New article [Creating an EA that works automatically (Part 12): Automation (IV)](https://www.mql5.com/en/articles/11305) has been published:

Author: [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831 "DJ_TLoG_831")

It seems the chronology of the Auto EA design is almost complete. Kindly let me know when the last article(s) after Part 12 is out.

Also, a referral on a competent programmer (conversant with the skills displayed in articles 1 to 12) is welcome on an Auto EA project that I have that uses Stochastics and the MA as the EA algo.

![Population optimization algorithms: ElectroMagnetism-like algorithm (ЕМ)](https://c.mql5.com/2/52/Avatar_ElectroMagnetism-like_algorithm_jj.png)[Population optimization algorithms: ElectroMagnetism-like algorithm (ЕМ)](https://www.mql5.com/en/articles/12352)

The article describes the principles, methods and possibilities of using the Electromagnetic Algorithm in various optimization problems. The EM algorithm is an efficient optimization tool capable of working with large amounts of data and multidimensional functions.

![How to create a custom True Strength Index indicator using MQL5](https://c.mql5.com/2/54/true_strength_index_avatar.png)[How to create a custom True Strength Index indicator using MQL5](https://www.mql5.com/en/articles/12570)

Here is a new article about how to create a custom indicator. This time we will work with the True Strength Index (TSI) and will create an Expert Advisor based on it.

![MQL5 Wizard techniques you should know (Part 06): Fourier Transform](https://c.mql5.com/2/54/fourier_transform_avatar.png)[MQL5 Wizard techniques you should know (Part 06): Fourier Transform](https://www.mql5.com/en/articles/12599)

The Fourier transform introduced by Joseph Fourier is a means of deconstructing complex data wave points into simple constituent waves. This feature could be resourceful to traders and this article takes a look at that.

![Category Theory in MQL5 (Part 7): Multi, Relative and Indexed Domains](https://c.mql5.com/2/54/Category-Theory-p7-avatar.png)[Category Theory in MQL5 (Part 7): Multi, Relative and Indexed Domains](https://www.mql5.com/en/articles/12470)

Category Theory is a diverse and expanding branch of Mathematics which is only recently getting some coverage in the MQL5 community. These series of articles look to explore and examine some of its concepts & axioms with the overall goal of establishing an open library that provides insight while also hopefully furthering the use of this remarkable field in Traders' strategy development.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/11305&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069146082144878804)

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