---
title: Creating an EA that works automatically (Part 11): Automation (III)
url: https://www.mql5.com/en/articles/11293
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:08:17.550205
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/11293&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069149200291135710)

MetaTrader 5 / Trading


### Introduction

In the previous article " [Creating an EA that works automatically (Part 10): Automation (II)](https://www.mql5.com/en/articles/11286)", we looked at a way to add EA operation schedule control. While the entire EA system has been built to prioritize autonomy, before moving on to the last phase where we will get a 100% automated EA, we need to make some minor changes to the code.

During the automation phase, we must not modify, create or change any part of the existing code in any way. We only need to remove those points where there was interaction between the trader and the EA and add some kind of automatic trigger instead. Any further changes to the old code should be avoided. If you, however, have to change the old code when implementing automation features, it means that the code was poorly planned, and you have to redo all the planning in order to have a system with the following characteristics.

- Robust: the system should not have primary errors that can violate the integrity of any part of the code;
- Reliable: a reliable system is the one that has been subject to a number of potentially dangerous situations and yet has performed without failure;
- Stable: there is no point in having a system that works well at times, but without explanation crashes the platform;
- Scalable: the system should grow smoothly without a lot of programming involved;
- Encapsulated: only the functions really necessary for use must be visible outside the place where the code is being created;
- Fast: there's no point in having the best model if it's slow because of poorly prepared code.

Some of these features refer to the object-oriented programming model. In fact, this model is by far the most suitable for creating programs requiring a good level of security. Because of this, since the beginning of this series, you must have noticed that all the programming was focused on the use of classes, which means applying object-oriented programming. I know that this type of programming may seem to be confusing and difficult to learn at first, but believe me, you will benefit a lot if you really put in the effort and learn to create your codes as classes.

I'm saying this to show you, aspiring professional programmers, how you can create your code. You must get used to working with 3 folders, in which the code will be kept. No matter how complex or simple the program is, you must always work in 3 steps:

1. In the first phase, which will be in the development folder, we create, modify and test all the code. Any new features or changes must be made only in the code present in this folder.
2. After building and testing the code, it must be moved to the second folder, the working one. In this folder, the code may still contain some errors, but **you should NOT modify it**. If you need to edit the code while being in this folder, move it back to the development folder. One detail: if you edit code in order to correct some found flaw, without making any other, more extreme modifications, the code of this working folder can be kept in it, receiving due corrections.
3. Finally, after you have used the code several times in different situations without any new changes, move it to the third and last folder 'stable'. The code present in this folder has already proven to have no flaws and is very useful and efficient for the task for which it was designed. **Never** add new code in this folder.

If you apply this approach for some time, you'll eventually create a very interesting database of functions and procedures, and you will be able to program things extremely quickly and safely. This type of thing is highly appreciated, especially in an activity such as the financial market, where no one is really interested in using a code that is not suited to the risk present in the market always. Since we are always working in an area where runtime errors are not appreciated and everything happens in the worst type of scenario, which is real time, your code has to be able to resist unexpected events at any moment.

All this has been said in order to reach the following point, which is shown in Figure 01:

![Figure 01](https://c.mql5.com/2/48/001__8.png)

Figure 01. Manual control system

Many people do not fully understand what exactly they are programming or creating. This is because many of them don't really understand what's going on inside the system and end up thinking that the platform should provide exactly what the trader wants to do. However, if you look at Figure 01, you will understand that the platform should not be focused on providing what the trader wants. Instead, it should provide ways to interact with the trader and, at the same time, interact stably, quickly and efficiently with the trade server. At the same time, the platform must be kept operational and must support the elements that will be used to interact with the trader.

Pay attention that none of the arrows go beyond their point, which indicates that this is a manual trading system. Also, note that the EA is serviced by the platform, not the other way around, and that the trader does not communicate directly with the EA. Although it may seem strange, in fact the trader accesses the EA through the platform, and the platform controls the EA by sending it the events that the trader creates or executes. In response, the EA sends requests to the platform, which must be sent to the trade server. When the server responds, it returns those responses back to the platform, which forwards them to the EA. The EA, after analyzing and processing the server's response, sends some information to the platform so that it can show the trader what is happening or what is the result of the request made by the trader.

Many people do not realize things like that. If any glitches happen in the EA, the platform will not have problems. The problem is in the EA, but less experienced traders may mistakenly blame the platform for not doing what they wanted.

If you are not involved in the development and maintenance of the platform as a programmer, then you should not try to influence how it works. Make sure your code responds appropriately to the platform's requirements.

We finally come to the question: Why create a manual EA, use it for a while and only then automate it? The reason is precisely this - we create a way to actually test the code and create exactly what we need, no more and no less.

In order to properly automate the system without affecting a single line of code that will be used as a control point and order system, we need to make some additions and small changes. Thus, the code created up to the previous article will be placed in the working folder, the code created in the previous article will be placed in the stable folder, and the code presented in this article will be moved to the development folder. In this way, the development process expands and evolves, while we have much faster coding. If something goes wrong, we can always go back two versions, where everything worked without problems.

### Implementing the changes

The first thing we will actually do is modify the timing system. This modification is shown in the code below:

```
virtual const bool CtrlTimeIsPassed(void) final
                        {
                                datetime dt;
                                MqlDateTime mdt;

                                TimeToStruct(TimeLocal(), mdt);
                                TimeCurrent(mdt);
                                dt = (mdt.hour * 3600) + (mdt.min * 60);
                                return ((m_InfoCtrl[mdt.day_of_week].Init <= dt) && (m_InfoCtrl[mdt.day_of_week].End >= dt));
                        }
```

The deleted line was replaced by the highlighted line. Why did we make these changes? There are two reasons: first, we replace two calls with one. In the deleted line, we first had a call to get the time and then another one to convert it to a structure. The second reason is that TimeLocal actually returns the computer time, not the time displayed in the market watch element, as shown in Figure 02.

![Figure 02](https://c.mql5.com/2/48/002__3.png)

Figure 02. Time provided by the server in the last update.

The use of computer time is not a problem if it is synchronized via an NTP server (those that keep the official time up to date). However, most often people do not use such servers. Therefore, it may happen that the time control system allows the EA to enter or exit earlier. The change was necessary to avoid this inconvenience.

The change is made not to radically modify the code, but to provide greater stability, which is expected by the trader. Since if the EA enters and exits earlier than expected, the trader might be thinking there is an error in the platform or in the code. But the reason will actually be caused by a lack of time synchronization between the computer and the trading server. The trading server is most likely using an NTP server to maintain the official time, while the computer used for work may not use this server.

The next change was implemented in the order system:

```
                ulong ToServer(void)
                        {
                                MqlTradeCheckResult     TradeCheck;
                                MqlTradeResult          TradeResult;
                                bool bTmp;

                                ResetLastError();
                                ZeroMemory(TradeCheck);
                                ZeroMemory(TradeResult);
                                bTmp = OrderCheck(m_TradeRequest, TradeCheck);
                                if (_LastError == ERR_SUCCESS) bTmp = OrderSend(m_TradeRequest, TradeResult);
                                if (_LastError != ERR_SUCCESS) MessageBox(StringFormat("Error Number: %d", GetLastError()), "Order System", MB_OK);
                                if (_LastError != ERR_SUCCESS) PrintFormat("Order System - Error Number: %d", _LastError);
                                return (_LastError == ERR_SUCCESS ? TradeResult.order : 0);
                        }
```

This is also a necessary change to make the system usable for its intended purpose, so that the EA can be automated without too much trouble. In fact, the deleted string was replaced by the highlighted string not because the code could be faster or more stable, but because it was necessary to handle the occurrence of an error appropriately. When we have an automated EA, some types of failures can be ignored, as we have discussed in previous articles.

The problem is that the deleted line will always launch a message box informing about the error, but in some cases the error can be correctly handled by the code and the box is not needed. In such cases, we can simply print a message in the terminal so that the trader can take appropriate action.

Remember, a 100% automatic EA cannot wait for the trader to make a decision. Even though this could take a while, you can't do things without reporting what kind of problem happened. Once again, the code was modified in order to improve agility. There were no major changes that would require putting the system in a more intense testing phase looking for failures, which may have been caused by the changes.

But unlike these changes made above, now we will have others that will need to be tested in depth, as changes will affect the way the system works.

### Paving the way for automation

The changes that will be made now will allow us to effectively create a 100% automatic system. Without these changes, our hands would be tied for the next article, where I will show how to turn an already tested EA (and I hope you are performing all the required tests to understand how everything actually works) into an autonomous one. In order to implement the necessary changes, we will need to remove, or better say, modify some things, and add others. Let's start with the modification. What will be changed is described in the code below:

```
//+------------------------------------------------------------------+
#include "C_ControlOfTime.mqh"
//+------------------------------------------------------------------+
#define def_MAX_LEVERAGE                10
#define def_ORDER_FINISH                false
//+------------------------------------------------------------------+
class C_Manager : public C_ControlOfTime
```

These two definitions will no longer exist. In their place 2 new variables will appear, which cannot be modified by the trader, but can be defined by you, the programmer. Why make such a change? The reason is that when we make these changes, where the definitions will be replaced by variables, we will lose a little in terms of speed. Even if it is a few machine cycles, in fact there will be a small loss of performance, since it is considerably faster to access a constant value than to access a variable. But in return, we will gain in class reuse, and you will understand this better in the next article. Believe me, the difference in terms of ease and portability compensates for the small loss of performance. So, the above two lines have been replaced by the following:

```
class C_Manager : public C_ControlOfTime
{
        enum eErrUser {ERR_Unknown, ERR_Excommunicate};
        private :
                struct st00
                {
                        double  FinanceStop,
                                FinanceTake;
                        uint    Leverage,
                                MaxLeverage;
                        bool    IsDayTrade,
                                IsOrderFinish;
                }m_InfosManager;
```

Be careful when working with code: you, as a programmer, should not change the value of these two variables outside the place where they will be initialized. Be very careful not to do this. The place where they will be initialized is precisely in the class constructor as shown in the following code:

```
                C_Manager(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage, bool IsDayTrade, double Trigger, const bool IsOrderFinish, const uint MaxLeverage)
                        :C_ControlOfTime(magic),
                        m_bAccountHedging(false),
                        m_TicketPending(0),
                        m_Trigger(Trigger)
                        {
                                string szInfo;

                                ResetLastError();
                                ZeroMemory(m_Position);
                                m_InfosManager.IsOrderFinish    = IsOrderFinish;
                                m_InfosManager.MaxLeverage      = MaxLeverage;
                                m_InfosManager.FinanceStop      = FinanceStop;
                                m_InfosManager.FinanceTake      = FinanceTake;
                                m_InfosManager.Leverage         = Leverage;
                                m_InfosManager.IsDayTrade       = IsDayTrade;
```

The constructor will now receive two new arguments that initialize our variables. After that, we will change the points at which the definitions were instantiated. These changes will be made in the following points:

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
                                if (m_InfosManager.IsOrderFinish) if (m_TicketPending > 0) if (OrderSelect(m_TicketPending)) v1 = OrderGetDouble(ORDER_PRICE_OPEN);
                                m_Position.EnableBreakEven = (m_InfosManager.IsOrderFinish ? m_TicketPending == 0 : m_Position.EnableBreakEven) || (m_Position.IsBuy ? (v1 < v2) : (v1 > v2));
                                m_Position.Gap = FinanceToPoints(m_Trigger, m_Position.Leverage);

                                return m_Position.Leverage - tmp;
                        }

inline void TriggerBreakeven(void)
                        {
                                double price;

                                if (PositionSelectByTicket(m_Position.Ticket))
                                        if (PositionGetDouble(POSITION_PROFIT) >= m_Trigger)
                                        {
                                                price = m_Position.PriceOpen + (GetTerminalInfos().PointPerTick * (m_Position.IsBuy ? 1 : -1));
                                                if (def_ORDER_FINISH)
                                                if (m_InfosManager.IsOrderFinish)
                                                {
                                                        if (m_TicketPending > 0) m_Position.EnableBreakEven = !ModifyPricePoints(m_TicketPending, price, 0, 0);
                                                }else m_Position.EnableBreakEven = !ModifyPricePoints(m_Position.Ticket, m_Position.PriceOpen, price, m_Position.TP);
                                        }
                        }

                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                if (!CtrlTimeIsPassed()) return;
                                if ((m_StaticLeverage >= def_MAX_LEVERAGE) || (m_TicketPending > 0) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                if ((m_StaticLeverage >= m_InfosManager.MaxLeverage) || (m_TicketPending > 0) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                m_TicketPending = C_Orders::CreateOrder(type, Price, (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceStop), (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_TicketPending = C_Orders::CreateOrder(type, Price, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }

                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;

                                if (!CtrlTimeIsPassed()) return;
                                if ((m_StaticLeverage >= def_MAX_LEVERAGE) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                if ((m_StaticLeverage >= m_InfosManager.MaxLeverage) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                tmp = C_Orders::ToMarket(type, (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceStop), (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                tmp = C_Orders::ToMarket(type, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                        }

                void PendingToPosition(void)
                        {
                                ResetLastError();
                                if ((m_bAccountHedging) && (m_Position.Ticket > 0))
                                {
                                        if (def_ORDER_FINISH)
                                        if (m_InfosManager.IsOrderFinish)
                                        {
// ... The rest of the code ...

                void UpdatePosition(const ulong ticket)
                        {
                                int ret;
                                double price;

                                if ((ticket == 0) || (ticket != m_Position.Ticket) || (m_Position.Ticket == 0)) return;
                                if (PositionSelectByTicket(m_Position.Ticket))
                                {
                                        ret = SetInfoPositions();
                                        if (def_ORDER_FINISH)
                                        if (m_InfosManager.IsOrderFinish)
                                        {
                                                price = m_Position.PriceOpen + (FinanceToPoints(m_InfosManager.FinanceStop, m_Position.Leverage) * (m_Position.IsBuy ? -1 : 1));
                                                if (m_TicketPending > 0) if (OrderSelect(m_TicketPending))
                                                {
                                                        price = OrderGetDouble(ORDER_PRICE_OPEN);
                                                        C_Orders::RemoveOrderPendent(m_TicketPending);
                                                }
                                                if (m_Position.Ticket > 0)      m_TicketPending = C_Orders::CreateOrder(m_Position.IsBuy ? ORDER_TYPE_SELL : ORDER_TYPE_BUY, price, 0, 0, m_Position.Leverage, m_InfosManager.IsDayTrade);
                                        }
                                        m_StaticLeverage += (ret > 0 ? ret : 0);
                                }else
                                {
                                        ZeroMemory(m_Position);
                                        if ((def_ORDER_FINISH) && (m_TicketPending > 0))
                                        if ((m_InfosManager.IsOrderFinish) && (m_TicketPending > 0))
                                        {
                                                RemoveOrderPendent(m_TicketPending);
                                                m_TicketPending = 0;
                                        }
                                }
                                ResetLastError();
                        }

inline void TriggerTrailingStop(void)
                        {
                                double price, v1;

                                if ((m_Position.Ticket == 0) || (def_ORDER_FINISH ? m_TicketPending == 0 : m_Position.SL == 0)) return;
                                if ((m_Position.Ticket == 0) || (m_InfosManager.IsOrderFinish ? m_TicketPending == 0 : m_Position.SL == 0)) return;
                                if (m_Position.EnableBreakEven) TriggerBreakeven(); else
                                {
                                        price = SymbolInfoDouble(_Symbol, (GetTerminalInfos().ChartMode == SYMBOL_CHART_MODE_LAST ? SYMBOL_LAST : (m_Position.IsBuy ? SYMBOL_ASK : SYMBOL_BID)));
                                        v1 = m_Position.SL;
                                        if (def_ORDER_FINISH)
                                        if (m_InfosManager.IsOrderFinish)
                                                if (OrderSelect(m_TicketPending)) v1 = OrderGetDouble(ORDER_PRICE_OPEN);
                                        if (v1 > 0) if (MathAbs(price - v1) >= (m_Position.Gap * 2))
                                        {
                                                price = v1 + (m_Position.Gap * (m_Position.IsBuy ? 1 : -1));
                                                if (def_ORDER_FINISH)
                                                if (m_InfosManager.IsOrderFinish)
                                                        ModifyPricePoints(m_TicketPending, price, 0, 0);
                                                else    ModifyPricePoints(m_Position.Ticket, m_Position.PriceOpen, price, m_Position.TP);
                                        }
                                }
                        }
```

Absolutely all the removed parts have been replaced by highlighted fragments. In this way, we managed not only to promote the much-desired class reuse, but also to improve it in terms of usability. Although this is not entirely clear in this article, in the next one you will see how this will be done.

In addition to the changes already made, we still need to make some additions to maximize the access of the automation system to the order sending system and thereby increase the class reuse. To do this, we first add the following function:

```
inline void ClosePosition(void)
	{
		if (m_Position.Ticket > 0)
		{
			C_Orders::ClosePosition(m_Position.Ticket);
			ZeroMemory(m_Position.Ticket);
		}
	}
```

This function is required in some operating models, so we need to include it in the C\_Manager class code. However, after we add this function, the compiler will generate several warnings when trying to compile the code. See Figure 03 below.

![Figure 03](https://c.mql5.com/2/48/003__3.png)

Figure 03. Compiler warnings

Unlike compiler warnings which can be ignored (although it is strongly not recommended), warnings in Figure 03 can be potentially harmful to the program and may cause the generated code to work incorrectly.

Ideally, when you notice that the compiler has generated such warnings, you should try to correct the failure that generates them. Sometimes it's something quite simple to solve, other times it's something a little more complicated. This may be a type of change, where part of the data is lost during the conversion. One way or another, it's always important to look at why the compiler generated such warnings, even if the code is compiled.

The presence of compiler warnings is a sign that something is not going well in the code, since the compiler is having difficulties understanding what you are programming. If it can't understand it, it won't generate 100% reliable code.

Some programming platforms allow you to turn off compiler warnings, but I personally do not recommend doing this. If you want to have 100% reliable code, it's best to leave all warnings enabled. You will realize over time that leaving standard platform settings is the best way to ensure a more reliable code.

We have two options to resolve the above warnings. The first is to replace calls to the ClosePosition function that currently refer to a function present in the C\_Orders class with a new function added in the C\_Manager class. This is the best option as we will check the call present in C\_Manager. The second option is to tell the compiler that the calls will refer to the C\_Orders class.

But I will change the code to use the newly created call. So, the problematic points that generate warnings will be solved, and the compiler will understand what we are trying to do.

```
                ~C_Manager()
                        {
                                if (_LastError == (ERR_USER_ERROR_FIRST + ERR_Excommunicate))
                                {
                                        if (m_TicketPending > 0) RemoveOrderPendent(m_TicketPending);
                                        if (m_Position.Ticket > 0) ClosePosition(m_Position.Ticket);
                                        ClosePosition();
                                        Print("EA was kicked off the chart for making a serious mistake.");
                                }
                        }
```

The easiest way would be to solve this in the destructor, but there is a bit of a tricky part, which is shown below:

```
                void PendingToPosition(void)
                        {
                                ResetLastError();
                                if ((m_bAccountHedging) && (m_Position.Ticket > 0))
                                {
                                        if (m_InfosManager.IsOrderFinish)
                                        {
                                                if (PositionSelectByTicket(m_Position.Ticket)) ClosePosition(m_Position.Ticket);
                                                ClosePosition();
                                                if (PositionSelectByTicket(m_TicketPending)) C_Orders::ClosePosition(m_TicketPending); else RemoveOrderPendent(m_TicketPending);
                                                ZeroMemory(m_Position.Ticket);
                                                m_TicketPending = 0;
                                                ResetLastError();
                                        }else   SetUserError(ERR_Unknown);
                                }else m_Position.Ticket = (m_Position.Ticket == 0 ? m_TicketPending : m_Position.Ticket);
                                m_TicketPending = 0;
                                if (_LastError != ERR_SUCCESS) UpdatePosition(m_Position.Ticket);
                                CheckToleranceLevel();
                        }
```

I deleted some lines and added calls to close a position. But if a pending order turns into a position for any reason, we must remove the position as we did in the original code. But the position has not yet been captured by the C\_Manager class. In this case, we tell the compiler that the call will refer to the C\_Orders class, as shown in the highlighted code.

Below is another change we need to make:

```
inline void EraseTicketPending(const ulong ticket)
                        {
                                if ((m_TicketPending == ticket) && (m_TicketPending > 0))
                                {
                                        if (PositionSelectByTicket(m_TicketPending)) C_Orders::ClosePosition(m_TicketPending);
                                        else RemoveOrderPendent(m_TicketPending);
                                        m_TicketPending = 0;
                                }
                                ResetLastError();
                                m_TicketPending = (ticket == m_TicketPending ? 0 : m_TicketPending);
                        }
```

The crossed out code, which was the original code, has been replaced with a slightly more complex one, but it gives us a greater ability to remove a pending order or, if it has become a position, delete it. Previously, we simply responded to an event that the MetaTrader 5 platform informed us about, causing the value indicated as a pending order ticket to be set to zero so that a new pending order can be sent. But now we are doing a little more than that, as we need such functionality in a 100% automated system.

By implementing these small changes, we get a system-wide bonus, and this will give us an increase in code reuse and testing.

### Final improvements before the final phase

Before the final phase, we can implement a few more improvements that will increase code reuse. The first one is shown in the following code:

```
                void UpdatePosition(const ulong ticket)
                        {
                                int ret;
                                double price;

                                if ((ticket == 0) || (ticket != m_Position.Ticket) || (m_Position.Ticket == 0)) return;
                                if (PositionSelectByTicket(m_Position.Ticket))
                                {
                                        ret = SetInfoPositions();
                                        if (m_InfosManager.IsOrderFinish)
                                        {
                                                price = m_Position.PriceOpen + (FinanceToPoints(m_InfosManager.FinanceStop, m_Position.Leverage) * (m_Position.IsBuy ? -1 : 1));
                                                if (m_TicketPending > 0) if (OrderSelect(m_TicketPending))
                                                {
                                                        price = OrderGetDouble(ORDER_PRICE_OPEN);
                                                        C_Orders::RemoveOrderPendent(m_TicketPending);
                                                        EraseTicketPending(m_TicketPending);
                                                }
                                                if (m_Position.Ticket > 0)      m_TicketPending = C_Orders::CreateOrder(m_Position.IsBuy ? ORDER_TYPE_SELL : ORDER_TYPE_BUY, price, 0, 0, m_Position.Leverage, m_InfosManager.IsDayTrade);
                                        }
                                        m_StaticLeverage += (ret > 0 ? ret : 0);
                                }else
                                {
                                        ZeroMemory(m_Position);
                                        if ((m_InfosManager.IsOrderFinish) && (m_TicketPending > 0))
                                        {
                                                RemoveOrderPendent(m_TicketPending);
                                                m_TicketPending = 0;
                                        }
                                        if (m_InfosManager.IsOrderFinish) EraseTicketPending(m_TicketPending);
                                }
                                ResetLastError();
                        }
```

Another point that can benefit from these improvements is in the following code:

```
                ~C_Manager()
                        {
                                if (_LastError == (ERR_USER_ERROR_FIRST + ERR_Excommunicate))
                                {
                                        if (m_TicketPending > 0) RemoveOrderPendent(m_TicketPending);
                                        EraseTicketPending(m_TicketPending);
                                        ClosePosition();
                                        Print("EA was kicked off the chart for making a serious mistake.");
                                }
                        }
```

And finally, the last point, which also benefits from code reuse:

```
                void PendingToPosition(void)
                        {
                                ResetLastError();
                                if ((m_bAccountHedging) && (m_Position.Ticket > 0))
                                {
                                        if (m_InfosManager.IsOrderFinish)
                                        {
                                                ClosePosition();
                                                EraseTicketPending(m_TicketPending);
                                                if (PositionSelectByTicket(m_TicketPending)) C_Orders::ClosePosition(m_TicketPending); else RemoveOrderPendent(m_TicketPending);
                                                m_TicketPending = 0;
                                                ResetLastError();
                                        }else   SetUserError(ERR_Unknown);
                                }else m_Position.Ticket = (m_Position.Ticket == 0 ? m_TicketPending : m_Position.Ticket);
                                m_TicketPending = 0;
                                if (_LastError != ERR_SUCCESS) UpdatePosition(m_Position.Ticket);
                                CheckToleranceLevel();
                        }
```

To finish the question of improvements and changes, there is a small detail. The EA may, for example, have a maximum definite volume 10 times the minimum volume. But if the trader, while configuring the volume, indicates the 3-time leverage value when performing the third operation, the EA will be very close to the maximum volume allowed. Then, if you send a fourth request, it would be violating the maximum allowed volume.

This may seem like a minor defect, and many may consider it to be of little danger. In a sense, I agree with this idea, since the fifth order will never be accepted. But the indicated volume defined when programming was 10 times. So, when the fourth order was accepted, the EA would have executed the volume 12 times, exceeding 2 times the maximum configured volume. This is because the trader configures a leverage of 3 times, but what if he had indicated a 9-time leverage? The trader would expect the EA to perform only one trade in this case.

So, imagine the surprise and astonishment of the user who sees that the EA opened the second trade, exceeding the maximum volume by 8 times. Such a person may even suffer a heart attack.

As you can see, despite being a failure of little potential, it is still a failure which should not be admitted, especially for an automated EA. This would be no problem for a manual EA, since the trader would check to make sure not create another entry with the same level of leverage. One way or another, we should provide some kind of EA block for such cases. And this should be implemented before the next article. I do not want to worry about such things later.

To fix this, we will add a few lines of code as shown below:

```
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                if (!CtrlTimeIsPassed()) return;
                                if ((m_StaticLeverage >= m_InfosManager.MaxLeverage) || (m_TicketPending > 0) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                if (m_StaticLeverage + m_InfosManager.Leverage > m_InfosManager.MaxLeverage)
                                {
                                        Print("Request denied, as it would violate the maximum volume allowed for the EA.");
                                        return;
                                }
                                m_TicketPending = C_Orders::CreateOrder(type, Price, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;

                                if (!CtrlTimeIsPassed()) return;
                                if ((m_StaticLeverage >= m_InfosManager.MaxLeverage) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                if (m_StaticLeverage + m_InfosManager.Leverage > m_InfosManager.MaxLeverage)
                                {
                                        Print("Request denied, as it would violate the maximum volume allowed for the EA.");
                                        return;
                                }
                                tmp = C_Orders::ToMarket(type, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                        }
//+------------------------------------------------------------------+
```

The code shown in green avoids the event explained above. However, take a closer look at the highlighted code. Have you noticed that it is exactly the same? We can do two things here: create a macro to place all of the code or create a function to replace or rather accumulate all of that common code.

I decided to create a function to keep everything simple and clear, since many readers of these articles are just starting their programming journey and may not have much experience or knowledge. We will place this new function in a private part of the code, while the rest of the EA code does not need to know about its existence. Here is the new function:

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

Let's understand what is happening here. All lines of code that were repeated in order sending codes are now collected in the above code. However, there is little difference between sending a pending order and a market order, and this difference is determined by this particular point. So, we should check whether it is a pending order or a market order. To distinguish these two types of orders, we use the argument which allows you to combine all the code into one. If there is any impossibility to send an order, false is returned. True is returned if the order can be sent.

The new function code is shown below:

```
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                if (!IsPossible(true)) return;
                                if (!CtrlTimeIsPassed()) return;
                                if ((m_StaticLeverage >= m_InfosManager.MaxLeverage) || (m_TicketPending > 0) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                if (m_StaticLeverage + m_InfosManager.Leverage > m_InfosManager.MaxLeverage)
                                {
                                        Print("Request denied, as it would violate the maximum volume allowed for the EA.");
                                        return;
                                }
                                m_TicketPending = C_Orders::CreateOrder(type, Price, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;

                                if (!IsPossible(false)) return;
                                if (!CtrlTimeIsPassed()) return;
                                if ((m_StaticLeverage >= m_InfosManager.MaxLeverage) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                if (m_StaticLeverage + m_InfosManager.Leverage > m_InfosManager.MaxLeverage)
                                {
                                        Print("Request denied, as it would violate the maximum volume allowed for the EA.");
                                        return;
                                }
                                tmp = C_Orders::ToMarket(type, (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceStop), (m_InfosManager.IsOrderFinish ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                        }
//+------------------------------------------------------------------+
```

Calls now depend on each specific case, while all crossed out lines have been removed from the code, since their existence no longer makes sense. This is how we develop programs that become safe, reliable, stable and robust, by eliminating redundant components: by analyzing what can be changed and improved, and by testing and reusing code as much as possible.

When you look at the finished code, you often get the impression that it was born that way. However, it took several steps to reach the final form. These steps include continuous testing and experimentation to minimize failures and potential gaps. This process is gradual and requires dedication and perseverance to bring the code to the desired quality.

### Final conclusions

Despite everything that has been said, spoken and shown so far, there is still a breach which we will have to close, because it is considerably harmful for a 100% automatic system. I know many should be eager to see the EA work automatically. But believe me, you will not want a 100% automated EA which contains a breach or fails to operate. Some less serious flaws may even pass in a manual system, but for a 100% automated system this is not admissible.

Since this topic is difficult to explain, we will cover it in the next article. The code as at its current state is attached below. I leave here a small challenge for you, before reading the next article. You know that there is still a weak part in the EA, which prevents it from being safely automated. Can you spot it? Hint: it is in the way the C\_Manager class analyzes the work of the EA.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11293](https://www.mql5.com/pt/articles/11293)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11293.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_11.zip](https://www.mql5.com/en/articles/download/11293/ea_automatico_-_11.zip "Download EA_Automatico_-_11.zip")(10.87 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/446470)**
(5)


![Luis Antonio Perdomo Martínez](https://c.mql5.com/avatar/avatar_na2.png)

**[Luis Antonio Perdomo Martínez](https://www.mql5.com/en/users/luisantonioperdomomartinez64)**
\|
21 Apr 2023 at 06:15

If automated it is better because only the machine works and even if you are sleeping you are gaining.


![Gunnar Forsgren](https://c.mql5.com/avatar/avatar_na2.png)

**[Gunnar Forsgren](https://www.mql5.com/en/users/gunnarforsgren)**
\|
30 Apr 2023 at 14:50

Small typos seen (red mark followed by correction in green) :

The code present in this folder has already proven to have **now** no flaws and is very useful and efficient for the task for which it was designed. **Never** add **mew** new code in this folder.

And:

You know that there is still a weekweak part in the EA

Figure 01. "Manual control system" has text PlataformPlatform

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
1 May 2023 at 11:26

**Gunnar Forsgren [#](https://www.mql5.com/en/forum/446470#comment_46576460):**

Small typos seen (red mark followed by correction in green) :

The code present in this folder has already proven to have **now** no flaws and is very useful and efficient for the task for which it was designed. **Never** add **mew** new code in this folder.

And:

You know that there is still a weekweak part in the EA

Figure 01. "Manual control system" has text PlataformPlatform

In fact the figure even contains an error at the time of typing. Thank you for reporting and for noticing this.😁👍

![Victor Chukwudumebi Ovorakpor](https://c.mql5.com/avatar/avatar_na2.png)

**[Victor Chukwudumebi Ovorakpor](https://www.mql5.com/en/users/victorovorakpor)**
\|
27 May 2023 at 15:47

I really enjoy reading this


![crt6789](https://c.mql5.com/avatar/avatar_na2.png)

**[crt6789](https://www.mql5.com/en/users/crt6789)**
\|
28 May 2023 at 14:43

When will the moderators finish updating all of them, it's beneficial 🫡


![How to connect MetaTrader 5 to PostgreSQL](https://c.mql5.com/2/53/avatar_How_to_connect_MetaTrader_5_to_PostgreSQL.png)[How to connect MetaTrader 5 to PostgreSQL](https://www.mql5.com/en/articles/12308)

This article describes four methods for connecting MQL5 code to a Postgres database and provides a step-by-step tutorial for setting up a development environment for one of them, a REST API, using the Windows Subsystem For Linux (WSL). A demo app for the API is provided along with the corresponding MQL5 code to insert data and query the respective tables, as well as a demo Expert Advisor to consume this data.

![Population optimization algorithms: Saplings Sowing and Growing up (SSG)](https://c.mql5.com/2/52/growing-tree-avatar.png)[Population optimization algorithms: Saplings Sowing and Growing up (SSG)](https://www.mql5.com/en/articles/12268)

Saplings Sowing and Growing up (SSG) algorithm is inspired by one of the most resilient organisms on the planet demonstrating outstanding capability for survival in a wide variety of conditions.

![Category Theory in MQL5 (Part 7): Multi, Relative and Indexed Domains](https://c.mql5.com/2/54/Category-Theory-p7-avatar.png)[Category Theory in MQL5 (Part 7): Multi, Relative and Indexed Domains](https://www.mql5.com/en/articles/12470)

Category Theory is a diverse and expanding branch of Mathematics which is only recently getting some coverage in the MQL5 community. These series of articles look to explore and examine some of its concepts & axioms with the overall goal of establishing an open library that provides insight while also hopefully furthering the use of this remarkable field in Traders' strategy development.

![Creating an EA that works automatically (Part 10): Automation (II)](https://c.mql5.com/2/50/aprendendo_construindo_010_avatar.png)[Creating an EA that works automatically (Part 10): Automation (II)](https://www.mql5.com/en/articles/11286)

Automation means nothing if you cannot control its schedule. No worker can be efficient working 24 hours a day. However, many believe that an automated system should operate 24 hours a day. But it is always good to have means to set a working time range for the EA. In this article, we will consider how to properly set such a time range.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/11293&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069149200291135710)

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