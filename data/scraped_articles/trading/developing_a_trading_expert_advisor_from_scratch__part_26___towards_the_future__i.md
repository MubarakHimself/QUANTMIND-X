---
title: Developing a trading Expert Advisor from scratch (Part 26): Towards the future (I)
url: https://www.mql5.com/en/articles/10620
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:46:16.606884
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/10620&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051671380635800676)

MetaTrader 5 / Examples


### Introduction

Despite the code fixes and improvements shown in the articles [Part 24](https://www.mql5.com/en/articles/10593) and [Part 25](https://www.mql5.com/en/articles/10606) of the "Developing a trading Expert Advisor from scratch" series, where we have seen how to increase the system robustness, there were still a few details left. But not because they were less relevant, in fact they are really important.

Now we have some questions that are related to how we want to work and what things we do during the trading day. Many traders simply place an order at a certain price and don't move it from that point. Whatever happens, they will assume that this is the perfect entry point and won't move the order. They may shift stop levels or even delete the stop levels, but they do not change the entry point.

Therefore, the remaining flaws in the code will not affect how traders actually work. They may even realize that the order system contains flaws (for example, those we are going to fix in this article). But those who like to be chasing the price, trying to enter a trade anyway, but do not want to enter the market, will witness many errors in the system. Some of them can interfere and make transactions unsafe (to put it mildly) while others will earn, leaving such traders helpless before the market.

### 2.0. Implementation

To begin our journey in this article, let's start by fixing a flaw that makes the EA a real money CRUSHER. Again, if you don't keep changing the entry point all the time, this problem will not affect you. However, I recommend thinking about updating your code, just in case. Even though the fix will already be implemented in the attached code, you might think that this will hurt the EA, because it will lose some performance, which is true. However, which is better: to lose some performance, or to risk losing money on a bad entry?

### 2.0.1. Entry point error

This error is the first thing we're going to fix, though they all need to be fixed one way or another. However, this one is by far the most catastrophic of them all. This happens when we place a pending entry, say BUY STOP, and move the entry point so that the order should now be of the BUY LIMIT type. There seems to be no problem here, but this failure is quite catastrophic, since the EA in the current stage of development will not be able to make the change in the correct way. In fact, many EAs want to make this modification, and if this happens, you will see information on the chart, but the server will have other information. The system will only be correctly updated when the position is opened, until then the data between what the EA shows on the chart and what is on the server will be incoherent.

In some cases, we only have this inconsistency, while in other cases the problem will be a complete disaster. To understand this, read the article carefully.

To eliminate this error, we have a solution that may go through different paths before it is applied. But the principle of operation will always be the same: remove the order from the order book, move it to a new position, change the order type and return it to the order book. This is what should be done, but how it is done will depend on the specific implementation.

Therefore, we are going to implement the most basic solution, but since it is not ideal, we will have to deal with some problems.

The solution is to modify the below function by adding the highlighted lines.

```
void SetPriceSelection(double price)
{
        char Pending;

        if (m_Selection.ticket == 0) return;
        Mouse.Show();
        if (m_Selection.ticket == def_IndicatorTicket0)
        {
                CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price,  price + m_Selection.tp - m_Selection.pr, price + m_Selection.sl - m_Selection.pr, m_Selection.bIsDayTrade);
                RemoveIndicator(def_IndicatorTicket0);
                return;
        }
        if ((Pending = GetInfosTradeServer(m_Selection.ticket)) == 0) return;
        m_TradeLine.SpotLight();
        switch (m_Selection.it)
        {
                case IT_TAKE:
                        if (Pending < 0) ModifyOrderPendent(m_Selection.ticket, m_Selection.pr, price, m_Selection.sl);
                        else ModifyPosition(m_Selection.ticket, price, m_Selection.sl);
                        break;
                case IT_STOP:
                        if (Pending < 0) ModifyOrderPendent(m_Selection.ticket, m_Selection.pr, m_Selection.tp, price);
                        else ModifyPosition(m_Selection.ticket, m_Selection.tp, price);
                        break;
                case IT_PENDING:
                        if (!ModifyOrderPendent(m_Selection.ticket, price, (m_Selection.tp == 0 ? 0 : price + m_Selection.tp - m_Selection.pr), (m_Selection.sl == 0 ? 0 : price + m_Selection.sl - m_Selection.pr)))
                        {
                                MoveSelection(macroGetLinePrice(def_IndicatorGhost, IT_PENDING));
                                m_TradeLine.SpotLight();
                        }
                        break;
        }
        RemoveIndicator(def_IndicatorGhost);
}
```

Although this solution partially solves the problem, it does not completely solve it. For example, for **BUY STOP** and **SELL STOP** orders, the problem is solved by adding these simple lines. But for **BUY LIMIT** and **STOP LIMIT**, the server will immediately fill the order once we click to change the entry point. What's worse here is that we enter a losing position. In case the order is configured as an empty order (with profit or loss levels) and the Stop Loss point is outside the price limits, then in addition to the fact that the server will immediately execute the order, it will also close it immediately after that, which will mean a complete disaster for our trading account. That is why trading systems are so difficult to develop. We carry out several tests on a **demo** account and if everything seems to be working we move on to a **real** account, at which point we start losing money without knowing what is really going on.

I repeat once again: the error does NOT AFFECT the case when an entry point is placed once and is never changed. The problem occurs when the trader moves the point.

Actually, **STOP** orders are working fine. Now we need to solve the problem with **LIMIT** pending orders. Although this problem may seem easy to solve, there is one thing to understand: _**there is NO perfect solution, and the solution that works best for the system developer may not be the one that works for you.**_.

I will show here one of the possible solutions to this problem. The solution will be implemented in the same function shown above. Here is its new code:

```
void SetPriceSelection(double price)
{
        char Pending;
        double last;
        long orderType;

        if (m_Selection.ticket == 0) return;
        Mouse.Show();
        if (m_Selection.ticket == def_IndicatorTicket0)
        {
                CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price,  price + m_Selection.tp - m_Selection.pr, price + m_Selection.sl - m_Selection.pr, m_Selection.bIsDayTrade);
                RemoveIndicator(def_IndicatorTicket0);
                return;
        }
        if ((Pending = GetInfosTradeServer(m_Selection.ticket)) == 0) return;
        m_TradeLine.SpotLight();
        switch (m_Selection.it)
        {
                case IT_TAKE:
                        if (Pending < 0) ModifyOrderPendent(m_Selection.ticket, m_Selection.pr, price, m_Selection.sl);
                        else ModifyPosition(m_Selection.ticket, price, m_Selection.sl);
                        break;
                case IT_STOP:
                        if (Pending < 0) ModifyOrderPendent(m_Selection.ticket, m_Selection.pr, m_Selection.tp, price);
                        else ModifyPosition(m_Selection.ticket, m_Selection.tp, price);
                        break;
                case IT_PENDING:
                        orderType = OrderGetInteger(ORDER_TYPE);
                        if ((orderType == ORDER_TYPE_BUY_LIMIT) || (orderType == ORDER_TYPE_SELL_LIMIT))
                        {
                                last = SymbolInfoDouble(Terminal.GetSymbol(), (m_Selection.bIsBuy ? SYMBOL_ASK : SYMBOL_BID));
                                if (((m_Selection.bIsBuy) && (price > last)) || ((!m_Selection.bIsBuy) && (price < last)))
                                {
                                        RemoveOrderPendent(m_Selection.ticket);
                                        RemoveIndicator(m_Selection.ticket);
                                        CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price, (m_Selection.tp == 0 ? 0 : price + m_Selection.tp - m_Selection.pr), (m_Selection.sl == 0 ? 0 : price + m_Selection.sl - m_Selection.pr), m_Selection.bIsDayTrade);
                                        break;
                                }
                        }
                        if (!ModifyOrderPendent(m_Selection.ticket, price, (m_Selection.tp == 0 ? 0 : price + m_Selection.tp - m_Selection.pr), (m_Selection.sl == 0 ? 0 : price + m_Selection.sl - m_Selection.pr)))
                        {
                                MoveSelection(macroGetLinePrice(def_IndicatorGhost, IT_PENDING));
                                m_TradeLine.SpotLight();
                        }
                        break;
        }
        RemoveIndicator(def_IndicatorGhost);
}
```

This is done as follows. When we are going to change the entry point of a pending order, we check if the order in the order book (Depth of Market) is of the STOP LIMIT or BUY LIMIT type. If it is not, then the execution flow will continue to another point in the code. If it is, then we do an immediate capture of the current asset price and we will use the following criteria: for a BUY order, capture the current ASK value. Respectively, it is BID for Sell orders. This replaces the old method using the LAST value, but since it is not used in some markets, we will not use it as a reference. Then check to see if the order in the order book becomes invalidated or if it is only modified.

If the order is still valid, the system will ignore the validation code and will go to the part where the order will be changed. But if the order in the Depth of Market is invalid, the system will execute the following code:

```
RemoveOrderPendent(m_Selection.ticket);
RemoveIndicator(m_Selection.ticket);
CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price, (m_Selection.tp == 0 ? 0 : price + m_Selection.tp - m_Selection.pr), (m_Selection.sl == 0 ? 0 : price + m_Selection.sl - m_Selection.pr), m_Selection.bIsDayTrade);
break;
```

But the code above will only change the SELL LIMIT and BUY LIMIT orders to SELL STOP and BUY STOP respectively. What if we want to return these types back to the original ones or just prevent such a change?

If we don't want the system to change the type of the executed order, we just have to replace the highlighted fragment with the following code:

```
if ((orderType == ORDER_TYPE_BUY_LIMIT) || (orderType == ORDER_TYPE_SELL_LIMIT))
{
        last = SymbolInfoDouble(Terminal.GetSymbol(), (m_Selection.bIsBuy ? SYMBOL_ASK : SYMBOL_BID));
        if (((m_Selection.bIsBuy) && (price > last)) || ((!m_Selection.bIsBuy) && (price < last)))
        {
                RemoveOrderPendent(m_Selection.ticket);
                RemoveIndicator(m_Selection.ticket);
                CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price, (m_Selection.tp == 0 ? 0 : price + m_Selection.tp - m_Selection.pr), (m_Selection.sl == 0 ? 0 : price + m_Selection.sl - m_Selection.pr), m_Selection.bIsDayTrade);
                MoveSelection(macroGetLinePrice(def_IndicatorGhost, IT_PENDING));
                m_TradeLine.SpotLight();
                break;
        }
}
```

This code will prevent the order type from being changed. You can change the point at which a pending order will be filled, but you cannot change a LIMIT order to a STOP order or vice versa. Now, if you want to keep chasing the price and to force entry at a certain point, use the code shown below. This is the code that will be used in the EA.

```
#define def_AdjustValue(A) (A == 0 ? 0 : price + A - m_Selection.pr)
#define macroForceNewType       {                                                                                                                                               \
                RemoveOrderPendent(m_Selection.ticket);                                                                                                                         \
                RemoveIndicator(m_Selection.ticket);                                                                                                                            \
                CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price, def_AdjustValue(m_Selection.tp), def_AdjustValue(m_Selection.sl), m_Selection.bIsDayTrade);      \
                break;                                                                                                                                                          \
                                }

                void SetPriceSelection(double price)
                        {
                                char Pending;
                                double last;
                                long orderType;

                                if (m_Selection.ticket == 0) return;
                                Mouse.Show();
                                if (m_Selection.ticket == def_IndicatorTicket0)
                                {
                                        CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price,  price + m_Selection.tp - m_Selection.pr, price + m_Selection.sl - m_Selection.pr, m_Selection.bIsDayTrade);
                                        RemoveIndicator(def_IndicatorTicket0);
                                        return;
                                }
                                if (m_Selection.ticket == def_IndicatorFloat)
                                {
                                        CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, m_Selection.pr,  m_Selection.tp, m_Selection.sl, m_Selection.bIsDayTrade);
                                        RemoveIndicator(def_IndicatorFloat);
                                        return;
                                }
                                if ((Pending = GetInfosTradeServer(m_Selection.ticket)) == 0) return;
                                m_TradeLine.SpotLight();
                                switch (m_Selection.it)
                                {
                                        case IT_TAKE:
                                                if (Pending < 0) ModifyOrderPendent(m_Selection.ticket, m_Selection.pr, price, m_Selection.sl);
                                                else ModifyPosition(m_Selection.ticket, price, m_Selection.sl);
                                                break;
                                        case IT_STOP:
                                                if (Pending < 0) ModifyOrderPendent(m_Selection.ticket, m_Selection.pr, m_Selection.tp, price);
                                                else ModifyPosition(m_Selection.ticket, m_Selection.tp, price);
                                                break;
                                        case IT_PENDING:
                                                orderType = OrderGetInteger(ORDER_TYPE);
                                                if ((orderType == ORDER_TYPE_BUY_LIMIT) || (orderType == ORDER_TYPE_SELL_LIMIT))
                                                {
                                                        last = SymbolInfoDouble(Terminal.GetSymbol(), (m_Selection.bIsBuy ? SYMBOL_ASK : SYMBOL_BID));
                                                        if (((m_Selection.bIsBuy) && (price > last)) || ((!m_Selection.bIsBuy) && (price < last))) macroForceNewType;
                                                }
                                                if (!ModifyOrderPendent(m_Selection.ticket, price, def_AdjustValue(m_Selection.tp), def_AdjustValue(m_Selection.sl))) macroForceNewType;
                                }
                                RemoveIndicator(def_IndicatorGhost);
                        }
#undef def_AdjustValue
#undef macroForceNewType
```

Important note: Be careful when working with this code because of the ForceNewType macro. This macro contains a break which, when executed, will cause the code to exit the 'case' block. So, you should be extremely careful when modifying this block.

The system will no longer have an error moving the entry point, but we have other problems to solve. I have shown the way to correct the problem by the modification or keeping the same type of order — you choose the one that best suits you. Remember that each of these solutions has its pros and cons. But I will not go into detail. I only show how to correct and implement the system.

The result of these changes can be seen in the following video:

Demostração 01 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10620)

MQL5.community

1.91K subscribers

[Demostração 01](https://www.youtube.com/watch?v=Jfp6z25DD6M)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=Jfp6z25DD6M&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10620)

0:00

0:00 / 0:47

•Live

•

### 2.0.2. Preparing for the future

The above change solves the problem, but there is something more that can be done. Here I will show the beginning of this change. Looking at the EA's order system, there is still a lot of room for improvement. There are few changes required, and I want to explain them so that you can choose which path suits you best, because each trader has their own way of acting in the market. I don't want you to feel obligated to use the system which I am going to show you. Instead, I want to create a basis so that anyone can develop a custom EA.

So, let's move on to the next fact: starting from [Part 18](https://www.mql5.com/en/articles/10462), I have been showing how to develop an order system that is easy to use for those who trade a particular asset. But in [Part 20](https://www.mql5.com/en/articles/10497), the order system received visual elements, because at some point Chart Trade will become unnecessary for trading, since everything will be indicated by the order system itself, so you will be able to change and configure everything right on the chart. To get to this point, we need to start somewhere, and we will do it right now.

How about changing the traded volume directly within the order, without having to remove the order from the chart, to change the volume in Chart Trade and then to re-place the order on the chart? Interesting, isn't it? We will be implementing this feature right now. It helps a lot in several scenarios, but you should learn and understand how to use the system because you won't find it on any other platform. To be honest, I've never seen an EA that would have such functionality. Let's see what you can do to have this functionality in any EA.

First, define a new indicator index.

```
#define def_IndicatorFloat      3
```

When a pending order receives this value as a ticket, it can be processed in a completely different way. Everything that previously existed will remain in the order system, while we only add a new index.1

After that, we will add a new object to the system:

```
C_Object_BackGround     m_BackGround;
C_Object_TradeLine      m_TradeLine;
C_Object_BtnBitMap      m_BtnClose,
                        m_BtnCheck;
C_Object_Edit           m_EditInfo1,
                        m_EditInfo2;
C_Object_Label          m_BtnMove;
```

This object will always enable a few things while the order is pending.

Now we move on to the C\_Object\_BitMap class to edit it. Add some definitions:

```
#define def_BtnClose            "Images\\NanoEA-SIMD\\Btn_Close.bmp"
#define def_BtnCheckEnabled     "Images\\NanoEA-SIMD\\CheckBoxEnabled.bmp"
#define def_BtnCheckDisabled    "Images\\NanoEA-SIMD\\CheckBoxDisabled.bmp"
//+------------------------------------------------------------------+
#resource "\\" + def_BtnClose
#resource "\\" + def_BtnCheckEnabled
#resource "\\" + def_BtnCheckDisabled
```

We need to know what's going on in this class. So, add the following functions:

```
bool GetStateButton(string szObjectName) const
{
        return (bool) ObjectGetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_STATE);
}
//+------------------------------------------------------------------+
inline void SetStateButton(string szObjectName, bool bState)
{
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_STATE, bState);
}
```

GetStateButton returns the state of the button. MetaTrader 5 changes the state, so we don't need to implement additionalsteps but only find out whether the button value is True or False. But it may happen that the state doesn't reflect what we want. Then use SetStateButton to set the state to reflect the actual state as seen by the trade server and EA.

Another simple modification is in the C\_Object\_Edit class:

```
inline void SetOnlyRead(string szObjectName, bool OnlyRead)
{
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_READONLY, OnlyRead);
}
```

It shows if the value can be edited or not. We want to be able to modify order volumes directly on the chart, without using Chart Trade. Any pending order that is created will always be in read-only mode, but we will create a system that will change this.

So, let's get back to C\_IndicatorTradeView and implement some more changes. We are going to create a new function for the system. It is as follows:

```
#define macroSwapAtFloat(A, B) ObjectSetString(Terminal.Get_ID(), macroMountName(ticket, A, B), OBJPROP_NAME, macroMountName(def_IndicatorFloat, A, B));
                bool PendingAtFloat(ulong ticket)
                        {
                                eIndicatorTrade it;

                                if (macroGetLinePrice(def_IndicatorFloat, IT_PENDING) > 0) return false;
                                macroSwapAtFloat(IT_PENDING, EV_CHECK);
                                for (char c0 = 0; c0 < 3; c0++)
                                {
                                        switch(c0)
                                        {
                                                case 0: it = IT_PENDING;        break;
                                                case 1: it = IT_STOP;           break;
                                                case 2: it = IT_TAKE;           break;
                                                default:
                                                        return false;
                                        }
                                        macroSwapAtFloat(it, EV_CLOSE);
                                        macroSwapAtFloat(it, EV_MOVE);
                                        macroSwapAtFloat(it, EV_EDIT);
                                        macroSwapAtFloat(it, EV_GROUND);
                                        macroSwapAtFloat(it, EV_LINE);
                                        m_EditInfo1.SetOnlyRead(macroMountName(def_IndicatorFloat, IT_PENDING, EV_EDIT), false);
                                }
                                return true;
                        }
#undef macroSwapAtFloat
```

When this function is called, all indicator objects are renamed, i.e. the value pointing to the order ticket will be replaced by another value. In this case, it is the indicator that we considered at the beginning of this topic. We still have one more question. I don't use any structure to maintain the list of indicator objects, I do it in a different way. This way we let MetaTrader 5 take care of this list for us. But because of that, I can't create unlimited floating orders as we will be limited to having only one floating order. This can be checked using The following line:

```
if (macroGetLinePrice(def_IndicatorFloat, IT_PENDING) > 0) return false;
```

The check here is simple: if the indicator line is located somewhere, the macro will return a value different from 0, so we know that there is already an indicator using the reserved ticket. This will be important later, for the EA to restore the data of the indicator for which the request is denied. MetaTrader 5 changes the state of the Bitmap object automatically, so we need to inform the caller about the failure.

The next required change is in the function that creates the indicators:

```
#define macroCreateIndicator(A, B, C, D)        {                                                                               \
                m_TradeLine.Create(ticket, sz0 = macroMountName(ticket, A, EV_LINE), C);                                        \
                m_BackGround.Create(ticket, sz0 = macroMountName(ticket, A, EV_GROUND), B);                                     \
                m_BackGround.Size(sz0, (A == IT_RESULT ? 84 : (A == IT_PENDING ? 108 : 92)), (A == IT_RESULT ? 34 : 22));       \
                m_EditInfo1.Create(ticket, sz0 = macroMountName(ticket, A, EV_EDIT), D, 0.0);                                   \
                m_EditInfo1.Size(sz0, 60, 14);                                                                                  \
                if (A != IT_RESULT)     {                                                                                       \
                        m_BtnMove.Create(ticket, sz0 = macroMountName(ticket, A, EV_MOVE), "Wingdings", "u", 17, C);            \
                        m_BtnMove.Size(sz0, 21, 23);                                                                            \
                                        }else                   {                                                               \
                        m_EditInfo2.Create(ticket, sz0 = macroMountName(ticket, A, EV_PROFIT), clrNONE, 0.0);                   \
                        m_EditInfo2.Size(sz0, 60, 14);  }                                                                       \
                                                }
                void CreateIndicator(ulong ticket, eIndicatorTrade it)
                        {
                                string sz0;

                                switch (it)
                                {
                                        case IT_TAKE    : macroCreateIndicator(it, clrForestGreen, clrDarkGreen, clrNONE); break;
                                        case IT_STOP    : macroCreateIndicator(it, clrFireBrick, clrMaroon, clrNONE); break;
                                        case IT_PENDING:
                                                macroCreateIndicator(it, clrCornflowerBlue, clrDarkGoldenrod, def_ColorVolumeEdit);
                                                m_BtnCheck.Create(ticket, sz0 = macroMountName(ticket, it, EV_CHECK), def_BtnCheckEnabled, def_BtnCheckDisabled);
                                                m_BtnCheck.SetStateButton(sz0, true);
                                                break;
                                        case IT_RESULT  : macroCreateIndicator(it, clrDarkBlue, clrDarkBlue, def_ColorVolumeResult); break;
                                }
                                m_BtnClose.Create(ticket, macroMountName(ticket, it, EV_CLOSE), def_BtnClose);
                        }
#undef macroCreateIndicator
```

All the highlighted parts have been added to support our new system. Basically, we create here a checkbox that will always be set to true, which means that the order will be immediately placed in the order book. I did not want to modify this way of trading, but it is not the simple fact of changing the value of the checkbox from 'true' to 'false' that will prevent the orders from being placed directly. This change would require making other even deeper changes, and the problem is that at some point, you may come to place an order and forget to check the checkbox. The entry point would be missed, and you would think that the EA is defective, when in fact it was all due to forgetfulness. So, to avoid this, by default the pending orders will go directly to the order book, so you will have to change their status explicitly.

The next really important function is shown below:

```
#define def_AdjustValue(A) (A == 0 ? 0 : price + A - m_Selection.pr)
#define macroForceNewType       {                                                                                                                                               \
                RemoveOrderPendent(m_Selection.ticket);                                                                                                                         \
                RemoveIndicator(m_Selection.ticket);                                                                                                                            \
                CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price, def_AdjustValue(m_Selection.tp), def_AdjustValue(m_Selection.sl), m_Selection.bIsDayTrade);      \
                break;                                                                                                                                                          \
                                }

                void SetPriceSelection(double price)
                        {
                                char Pending;
                                double last;
                                long orderType;

                                if (m_Selection.ticket == 0) return;
                                Mouse.Show();
                                if (m_Selection.ticket == def_IndicatorTicket0)
                                {
                                        CreateOrderPendent(m_Selection.vol, m_Selection.bIsBuy, price, def_AdjustValue(m_Selection.tp), def_AdjustValue(m_Selection.sl), m_Selection.bIsDayTrade);
                                        RemoveIndicator(def_IndicatorTicket0);
                                        return;
                                }
                                if (m_Selection.ticket == def_IndicatorFloat)
                                {
                                        switch(m_Selection.it)
                                        {
                                                case IT_STOP   : m_Selection.sl = price; break;
                                                case IT_TAKE   : m_Selection.tp = price; break;
                                                case IT_PENDING:
                                                        m_Selection.sl = def_AdjustValue(m_Selection.sl);
                                                        m_Selection.tp = def_AdjustValue(m_Selection.tp);
                                                        m_Selection.pr = price;
                                                        break;
                                        }
                                        m_Selection.ticket = 0;
                                        m_TradeLine.SpotLight();
                                        return;
                                }
                                if ((Pending = GetInfosTradeServer(m_Selection.ticket)) == 0) return;
                                m_TradeLine.SpotLight();
                                switch (m_Selection.it)
                                {
                                        case IT_TAKE:
                                                if (Pending < 0) ModifyOrderPendent(m_Selection.ticket, m_Selection.pr, price, m_Selection.sl);
                                                else ModifyPosition(m_Selection.ticket, price, m_Selection.sl);
                                                break;
                                        case IT_STOP:
                                                if (Pending < 0) ModifyOrderPendent(m_Selection.ticket, m_Selection.pr, m_Selection.tp, price);
                                                else ModifyPosition(m_Selection.ticket, m_Selection.tp, price);
                                                break;
                                        case IT_PENDING:
                                                orderType = OrderGetInteger(ORDER_TYPE);
                                                if ((orderType == ORDER_TYPE_BUY_LIMIT) || (orderType == ORDER_TYPE_SELL_LIMIT))
                                                {
                                                        last = SymbolInfoDouble(Terminal.GetSymbol(), (m_Selection.bIsBuy ? SYMBOL_ASK : SYMBOL_BID));
                                                        if (((m_Selection.bIsBuy) && (price > last)) || ((!m_Selection.bIsBuy) && (price < last))) macroForceNewType;
                                                }
                                                if (!ModifyOrderPendent(m_Selection.ticket, price, def_AdjustValue(m_Selection.tp), def_AdjustValue(m_Selection.sl))) macroForceNewType;
                                }
                                RemoveIndicator(def_IndicatorGhost);
                        }
#undef def_AdjustValue
#undef macroForceNewType
```

The highlighted code parts do an interesting thing: they only update the values that will be used in the selector, but these values are actually stored in the indicator itself. It may also happen that we are moving the system in a more general way, so we need these values to be specified in the selector so that the function that performs the position calculations specifies the correct values.

There is something in this function that may not make sense. It is responsible for creating and modifying the data of a pending order, but if you look at it, you will not see any point where the pending order will be returned to the order book. You can move, modify and adjust the value volume of the order directly on the chart, but you will not be able to see how it will return to the chart.

It's a fact. The entire system for changing and creating pending orders is implemented in the above function. Strangely enough, this function does not place the order back in the order book just because we want it to, but because it actually makes a request, as shown below. Not to complicate, I will only show the part responsible for a request to place the order in the Depth of Market.

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{

// ... Internal code...

        case CHARTEVENT_OBJECT_CLICK:
                if (GetIndicatorInfos(sparam, ticket, it, ev)) switch (ev)
                {
                        case EV_CLOSE:
                                if (ticket == def_IndicatorFloat) RemoveIndicator(def_IndicatorFloat, it);
                                else if ((cRet = GetInfosTradeServer(ticket)) != 0) switch (it)
                                {
                        case IT_PENDING:
                        case IT_RESULT:
                                if (cRet < 0) RemoveOrderPendent(ticket); else ClosePosition(ticket);
                                break;
                        case IT_TAKE:
                        case IT_STOP:
                                m_Selection.ticket = ticket;
                                m_Selection.it = it;
                                SetPriceSelection(0);
                        break;
                }
                break;
        case EV_MOVE:
                if (ticket == def_IndicatorFloat)
                {
                        m_Selection.ticket = ticket;
                        m_Selection.it = it;
                }else   CreateGhostIndicator(ticket, it);
                break;
        case EV_CHECK:
                if (ticket != def_IndicatorFloat)
                {
                        if (PendingAtFloat(ticket)) RemoveOrderPendent(ticket);
                        else m_BtnCheck.SetStateButton(macroMountName(ticket, IT_PENDING, EV_CHECK), true);
                } else
                {
                        m_Selection.ticket = def_IndicatorTicket0;
                        m_Selection.it = IT_PENDING;
                        m_Selection.pr = macroGetLinePrice(def_IndicatorFloat, IT_PENDING);
                        m_Selection.sl = macroGetLinePrice(def_IndicatorFloat, IT_STOP);
                        m_Selection.tp = macroGetLinePrice(def_IndicatorFloat, IT_TAKE);
                        m_Selection.bIsBuy = (m_Selection.pr < m_Selection.tp) || (m_Selection.sl < m_Selection.pr);
                        m_Selection.bIsDayTrade = true;
                        m_Selection.vol = m_EditInfo1.GetTextValue(macroMountName(def_IndicatorFloat, IT_PENDING, EV_EDIT)) * Terminal.GetVolumeMinimal();
                        SetPriceSelection(m_Selection.pr);
                        RemoveIndicator(def_IndicatorFloat);
                }

// ... Rest of the code...
```

See how the system builds itself: we program less and less as the system grows bigger and bigger.

The highlighted code has something to do with the indicator we created at the beginning of the topic. Although everything seems to work well, we have some things that will be changed later because when the floating order returns to the order book, it will have the disadvantage of being a day trading order, so it will be closed at the end of the day. It will be changed later, but you should be aware of this. Now you might be confused by all this, and still don't understand how the pending order actually enters and leaves the order book, when we click on the checkbox. See the diagram below:

![](https://c.mql5.com/2/45/FluxoGrama.png)

See that all calls come from the same place. We have an order removed from the Depth of Market, but it will continue to be on the chart. All manipulations are performed as shown in the previous articles. But if you try to find a specific time when the order will return to the Depth of Market, you can get a little lost in the code. Now, if you look at the diagram, you can see that the call comes from the DispatchMessage function, because this is the only place that calls the SetPriceSelection function. But if we look at the SetPriceSelection function, there is no reference of creating an order with the index used in the floating system. But pay attention to one thing. We have the order creation by index 0, and this is exactly what we use. We change the order ticket and inform that it will be the index 0 ticket — in this way the order will be created. See the code below to understand how this works.

```
m_Selection.ticket = def_IndicatorTicket0;
m_Selection.it = IT_PENDING;
m_Selection.pr = macroGetLinePrice(def_IndicatorFloat, IT_PENDING);
m_Selection.sl = macroGetLinePrice(def_IndicatorFloat, IT_STOP);
m_Selection.tp = macroGetLinePrice(def_IndicatorFloat, IT_TAKE);
m_Selection.bIsBuy = (m_Selection.pr < m_Selection.tp) || (m_Selection.sl < m_Selection.pr);
m_Selection.bIsDayTrade = true;
m_Selection.vol = m_EditInfo1.GetTextValue(macroMountName(def_IndicatorFloat, IT_PENDING, EV_EDIT)) * Terminal.GetVolumeMinimal();
SetPriceSelection(m_Selection.pr);
RemoveIndicator(def_IndicatorFloat);
```

The code is perfect except for the highlighted line. Currently there is no way to fix this. This will be done in the next article, since we'll have to make some changes to the class itself.

The video below demonstrates the result of the changes. Pay attention to how the volume is modified and how a new order is sent at the specified point. The EA is now much easier to use.

Demostração 02 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10620)

MQL5.community

1.91K subscribers

[Demostração 02](https://www.youtube.com/watch?v=3eYnOlB6Yw0)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=3eYnOlB6Yw0&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10620)

0:00

0:00 / 1:49

•Live

•

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10620](https://www.mql5.com/pt/articles/10620)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10620.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Em\_direooo\_ao\_Futuro\_b\_I\_r.zip](https://www.mql5.com/en/articles/download/10620/ea_-_em_direooo_ao_futuro_b_i_r.zip "Download EA_-_Em_direooo_ao_Futuro_b_I_r.zip")(12033.63 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/434861)**

![DoEasy. Controls (Part 15): TabControl WinForms object — several rows of tab headers, tab handling methods](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__3.png)[DoEasy. Controls (Part 15): TabControl WinForms object — several rows of tab headers, tab handling methods](https://www.mql5.com/en/articles/11316)

In this article, I will continue working on the TabControl WinForm object — I will create a tab field object class, make it possible to arrange tab headers in several rows and add methods for handling object tabs.

![Developing a trading Expert Advisor from scratch (Part 25): Providing system robustness (II)](https://c.mql5.com/2/48/development__1.png)[Developing a trading Expert Advisor from scratch (Part 25): Providing system robustness (II)](https://www.mql5.com/en/articles/10606)

In this article, we will make the final step towards the EA's performance. So, be prepared for a long read. To make our Expert Advisor reliable, we will first remove everything from the code that is not part of the trading system.

![Developing a trading Expert Advisor from scratch (Part 27): Towards the future (II)](https://c.mql5.com/2/48/development__3.png)[Developing a trading Expert Advisor from scratch (Part 27): Towards the future (II)](https://www.mql5.com/en/articles/10630)

Let's move on to a more complete order system directly on the chart. In this article, I will show a way to fix the order system, or rather, to make it more intuitive.

![Developing a trading Expert Advisor from scratch (Part 24): Providing system robustness (I)](https://c.mql5.com/2/48/development.png)[Developing a trading Expert Advisor from scratch (Part 24): Providing system robustness (I)](https://www.mql5.com/en/articles/10593)

In this article, we will make the system more reliable to ensure a robust and secure use. One of the ways to achieve the desired robustness is to try to re-use the code as much as possible so that it is constantly tested in different cases. But this is only one of the ways. Another one is to use OOP.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/10620&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051671380635800676)

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