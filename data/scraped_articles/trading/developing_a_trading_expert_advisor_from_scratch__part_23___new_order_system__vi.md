---
title: Developing a trading Expert Advisor from scratch (Part 23): New order system (VI)
url: https://www.mql5.com/en/articles/10563
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:46:46.715824
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/10563&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051677986295501957)

MetaTrader 5 / Trading


### Introduction

In the previous article [Developing of a trading Expert Advisor from scratch (Part 22)](https://www.mql5.com/en/articles/10516), we have developed a system for moving pending orders and position stop levels. Although this method is relatively safe (because it reflects what is on the trade server), it is not the best way to move the levels quickly.

The problem is that every time we change something using the mouse, this event is sent to the server, and we have then to wait for a response. The issue is connected with the fact that the event is sent on every tick, that is, if we needed to move a level by several ticks at a time, we would have to go through all intermediary values which makes the whole process very slow. This is how we implement changes to the code to make it more flexible and to change the levels faster.

### 1.0. Planning

To implement the changes, we need to do a very simple thing: **_Do not inform the server about all changes_**, only inform about the required change. Even by just doing this will already make everything work fine, although we will not be absolutely sure that everything is exactly the way we are doing it.

Let's see now where we need to modify the code. We use a separate function shown below:

```
#define macroGetPrice(A) StringToDouble(ObjectGetString(Terminal.Get_ID(), MountName(ticket, A, EV_LINE), OBJPROP_TOOLTIP))
                void MoveSelection(double price, uint keys)
                        {
                                static string memStr = NULL;
                                static ulong ticket = 0;
                                static eIndicatorTrade it;
                                eEventType ev;
                                double tp, sl, pr;
                                bool isPending;

                                string sz0 = m_TradeLine.GetObjectSelected();

                                if (sz0 != NULL)
                                {
                                        if (memStr != sz0) GetIndicatorInfos(memStr = sz0, ticket, pr, it, ev);
                                        isPending = OrderSelect(ticket);
                                        switch (it)
                                        {
                                                case IT_TAKE:
                                                        if (isPending) ModifyOrderPendent(ticket, macroGetPrice(IT_PENDING), price, macroGetPrice(IT_STOP));
                                                        else ModifyPosition(ticket, price, macroGetPrice(IT_STOP));
                                                        break;
                                                case IT_STOP:
                                                        if (isPending) ModifyOrderPendent(ticket, macroGetPrice(IT_PENDING), macroGetPrice(IT_TAKE), price);
                                                        else ModifyPosition(ticket, macroGetPrice(IT_TAKE), price);
                                                        break;
                                                case IT_PENDING:
                                                        pr = macroGetPrice(IT_PENDING);
                                                        tp = macroGetPrice(IT_TAKE);
                                                        sl = macroGetPrice(IT_STOP);
                                                        ModifyOrderPendent(ticket, price, (tp == 0 ? 0 : price + tp - pr), (sl == 0 ? 0 : price + sl - pr));
                                                        break;
                                        }
                                };
                        }
#undef macroGetPrice
```

But along with the modifications inside the function, we'll also need to implement some changes related to mouse events. This is what we'll focus on first.

The highlighted segments need to be replaced with something else to represent the new positions that will be used. But all changes should be understandable for the user.

I have found an efficient way to make everything easy to understand and at the same time functional, without making big changes to the entire code structure. This is to create a ghost indication label which will not be visible until needed. Fortunately, MetaTrader 5 provide a very simple way to do this. So, this article will be very easy to understand for those who are already familiar with the material from the earlier articles within this series.

### 2.0. Implementation

To implement a ghost label, we will simply create it with a real one. This will be the exact shadow of a real label until we move on to working with prices in the way we showed in the previous article. At this point, the ghost label will appear on the chart as the real one moves. This will enable you to easily compare what is happening and to understand whether or not to make changes.

### 2.0.1. Creating a ghost indication label

All changes are implemented within the C\_IndicatorTradeView class. Let's start by defining three new directives:

```
#define def_IndicatorGhost      "G"
#define def_IndicatorReal       "R"
#define def_IndicatorGhostColor clrDimGray
```

This will make MetaTrader 5 work for us. The rule is this: first we create a ghost label, and then we create a real one. Thus, MetaTrader 5 will take care not to show the ghost label until the moment when we really need to see it. Since this is done by MetaTrader 5 itself, we will save a lot on programming.

An important detail: if you want to change the color of the ghost, simply change the color specified in the selected part.

Therefore, the next step is to modify the function that allows unique names to be generated.

```
inline string MountName(ulong ticket, eIndicatorTrade it, eEventType ev, bool isGhost = false)
{
        return StringFormat("%s%c%c%c%llu%c%c%c%s", def_NameObjectsTrade, def_SeparatorInfo, (char)it, def_SeparatorInfo, ticket, def_SeparatorInfo, (char)(isGhost ? ev + 32 : ev), def_SeparatorInfo, (isGhost ? def_IndicatorGhost : def_IndicatorReal));
}
```

The highlighted parts have been added or changed from the previous version. We can let MetaTrader 5 generate unique names. So, we don't need to worry about this. Note that I've added values to the events. I did this to prevent the ghost from receiving events and trying to take control.

The next step is obvious - we must create the label itself:

```
inline void CreateIndicatorTrade(ulong ticket, eIndicatorTrade it)
                        {
                                color cor1, cor2, cor3;
                                string sz0, sz1;

                                switch (it)
                                {
                                        case IT_TAKE    :
                                                cor1 = clrForestGreen;
                                                cor2 = clrDarkGreen;
                                                cor3 = clrNONE;
                                                break;
                                        case IT_STOP    :
                                                cor1 = clrFireBrick;
                                                cor2 = clrMaroon;
                                                cor3 = clrNONE;
                                                break;
                                        case IT_PENDING:
                                                cor1 = clrCornflowerBlue;
                                                cor2 = clrDarkGoldenrod;
                                                cor3 = def_ColorVolumeEdit;
                                                break;
                                        case IT_RESULT  :
                                        default:
                                                cor1 = clrDarkBlue;
                                                cor2 = clrDarkBlue;
                                                cor3 = def_ColorVolumeResult;
                                                break;
                                }
                                m_TradeLine.Create(ticket, MountName(ticket, it, EV_LINE, true), def_IndicatorGhostColor);
                                m_TradeLine.Create(ticket, MountName(ticket, it, EV_LINE), cor2);
                                if (ticket == def_IndicatorTicket0) m_TradeLine.SpotLight(MountName(ticket, IT_PENDING, EV_LINE));
                                if (it != IT_RESULT) m_BackGround.Create(ticket, sz0 = MountName(ticket, it, EV_GROUND, true), def_IndicatorGhostColor);
                                m_BackGround.Create(ticket, sz1 = MountName(ticket, it, EV_GROUND), cor1);
                                switch (it)
                                {
                                        case IT_TAKE:
                                        case IT_STOP:
                                        case IT_PENDING:
                                                m_BackGround.Size(sz0, 92, 22);
                                                m_BackGround.Size(sz1, 92, 22);
                                                break;
                                        case IT_RESULT:
                                                m_BackGround.Size(sz1, 84, 34);
                                                break;
                                }
                                m_BtnClose.Create(ticket, MountName(ticket, it, EV_CLOSE), def_BtnClose);
                                m_EditInfo1.Create(ticket, sz0 = MountName(ticket, it, EV_EDIT, true), def_IndicatorGhostColor, 0.0);
                                m_EditInfo1.Create(ticket, sz1 = MountName(ticket, it, EV_EDIT), cor3, 0.0);
                                m_EditInfo1.Size(sz0, 60, 14);
                                m_EditInfo1.Size(sz1, 60, 14);
                                if (it != IT_RESULT)
                                {
                                        m_BtnMove.Create(ticket, sz0 = MountName(ticket, it, EV_MOVE, true), "Wingdings", "u", 17, def_IndicatorGhostColor);
                                        m_BtnMove.Create(ticket, sz1 = MountName(ticket, it, EV_MOVE), "Wingdings", "u", 17, cor2);
                                        m_BtnMove.Size(sz1, 21, 21);
                                }else
                                {
                                        m_EditInfo2.Create(ticket, sz1 = MountName(ticket, it, EV_PROFIT), clrNONE, 0.0);
                                        m_EditInfo2.Size(sz1, 60, 14);
                                }
                        }
```

All highlighted lines create ghosts. One thing may seem strange: why we do not reproduce all the elements. In fact, the ghost is not an exact copy of the real label but only its shadow. Therefore, we do not need to create all the elements. The real label is the one that will define what should happen while the ghost only serves as a reference to what the trade server will see.

Now comes the part that requires a more detailed study, where MetaTrader 5 will really work hard. You might think that arranging objects in the right places would take a lot of work but look what actually changed in the source code.

```
#define macroSetAxleY(A, B)     {                                                                       \
                m_BackGround.PositionAxleY(MountName(ticket, A, EV_GROUND, B), y);                              \
                m_TradeLine.PositionAxleY(MountName(ticket, A, EV_LINE, B), y);                                 \
                m_BtnClose.PositionAxleY(MountName(ticket, A, EV_CLOSE, B), y);                                 \
                m_EditInfo1.PositionAxleY(MountName(ticket, A, EV_EDIT, B), y, (A == IT_RESULT ? -1 : 0));      \
                m_BtnMove.PositionAxleY(MountName(ticket, A, EV_MOVE, B), (A == IT_RESULT ? 9999 : y));         \
                m_EditInfo2.PositionAxleY(MountName(ticket, A, EV_PROFIT, B), (A == IT_RESULT ? y : 9999), 1);  \
                                }

#define macroSetAxleX(A, B, C)  {                                                       \
                m_BackGround.PositionAxleX(MountName(ticket, A, EV_GROUND, C), B);      \
                m_TradeLine.PositionAxleX(MountName(ticket, A, EV_LINE, C), B);         \
                m_BtnClose.PositionAxleX(MountName(ticket, A, EV_CLOSE, C), B + 3);     \
                m_EditInfo1.PositionAxleX(MountName(ticket, A, EV_EDIT, C), B + 21);    \
                m_BtnMove.PositionAxleX(MountName(ticket, A, EV_MOVE, C), B + 80);      \
                m_EditInfo2.PositionAxleX(MountName(ticket, A, EV_PROFIT, C), B + 21);  \
                                }
//+------------------------------------------------------------------+
inline void ReDrawAllsIndicator(void)
                        {
                                int             max = ObjectsTotal(Terminal.Get_ID(), -1, -1);
                                ulong           ticket;
                                double          price;
                                eIndicatorTrade it;
                                eEventType      ev;

                                for (int c0 = 0; c0 <= max; c0++) if (GetIndicatorInfos(ObjectName(Terminal.Get_ID(), c0, -1, -1), ticket, price, it, ev))
                                        PositionAxlePrice(ticket, it, price);
                        }
//+------------------------------------------------------------------+
inline void PositionAxlePrice(ulong ticket, eIndicatorTrade it, double price)
                        {
                                int x, y, desl;

                                ChartTimePriceToXY(Terminal.Get_ID(), 0, 0, price, x, y);
                                ObjectSetString(Terminal.Get_ID(), MountName(ticket, it, EV_LINE), OBJPROP_TOOLTIP, DoubleToString(price));
                                macroSetAxleY(it, true);
                                macroSetAxleY(it, false);
                                switch (it)
                                {
                                        case IT_TAKE: desl = 110; break;
                                        case IT_STOP: desl = 220; break;
                                        default: desl = 0;
                                }
                                macroSetAxleX(it, desl + (int)(ChartGetInteger(Terminal.Get_ID(), CHART_WIDTH_IN_PIXELS) * 0.2), true);
                                macroSetAxleX(it, desl + (int)(ChartGetInteger(Terminal.Get_ID(), CHART_WIDTH_IN_PIXELS) * 0.2), false);
                        }
#undef macroSetAxleX
#undef macroSetAxleY
```

Is that all? Do we simply change the macros? True, there is no need to re-create the entire code, we simply adjust it. All we really need to do is tell MetaTrader 5 the name of the object we are manipulating, while MetaTrader 5 will do the rest for us. We do not need to create a series of functions for this. Simply add the highlighted parts.

Another function to change here is shown below:

```
void SetTextValue(ulong ticket, eIndicatorTrade it, double value0, double value1 = 0.0, double priceOpen = 0.0)
{
        double finance;

        switch (it)
        {
                case IT_RESULT  :
                        PositionAxlePrice(ticket, it, priceOpen);
                        PositionAxlePrice(ticket, IT_PENDING, 0);
                        m_EditInfo2.SetTextValue(MountName(ticket, it, EV_PROFIT), value1);
                case IT_PENDING:
                        value0 = value0 / Terminal.GetVolumeMinimal();
                        m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT), value0, def_ColorVolumeEdit);
                        m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT, true), value0, def_IndicatorGhostColor);
                        break;
                case IT_TAKE    :
                case IT_STOP    :
                        finance = (value1 / Terminal.GetAdjustToTrade()) * value0;
                        m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT), finance);
                        m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT, true), finance, def_IndicatorGhostColor);
                        break;
        }
}
```

Here we have added the selected segments. This is how our ghost is selected. It accurately reflects what is happening to the real label. In fact, it reflects too well, so well that we'll have to implement a few more things to let it work properly. The code is not wrong, but the ghost is too closely related to the real label, which we should actually avoid.

The last function to change here is shown below:

```
inline void RemoveIndicator(ulong ticket, eIndicatorTrade it = IT_NULL)
                        {
#define macroDestroy(A, B)      {                                                                               \
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, A, EV_GROUND, B));                            \
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, A, EV_LINE, B));                              \
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, A, EV_CLOSE, B));                             \
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, A, EV_EDIT, B));                              \
                if (A != IT_RESULT)     ObjectDelete(Terminal.Get_ID(), MountName(ticket, A, EV_MOVE, B));      \
                else ObjectDelete(Terminal.Get_ID(), MountName(ticket, A, EV_PROFIT, B));                       \
                                }

                                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
                                if ((it == IT_NULL) || (it == IT_PENDING) || (it == IT_RESULT))
                                {
                                        macroDestroy(IT_RESULT, true);
                                        macroDestroy(IT_RESULT, false);
                                        macroDestroy(IT_PENDING, true);
                                        macroDestroy(IT_PENDING, false);
                                        macroDestroy(IT_TAKE, true);
                                        macroDestroy(IT_TAKE, false);
                                        macroDestroy(IT_STOP, true);
                                        macroDestroy(IT_STOP, false);
                                } else
                                {
                                        macroDestroy(it, true);
                                        macroDestroy(it, false);
                                }
                                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
#undef macroDestroy
                        }
```

The highlighted fragments have changed, nothing special here.

### 2.0.2. Separate the ghost from the real

The changes made in the previous section create a ghost. But we have a problem — it is too close to the real object. This is something that at first glance will be very difficult to implement. But when we look at the code, we see that we already have a solution. However, this solution is in the wrong place. We need to change the place where the solution is and make it more visible throughout the class.

The solution is shown in the following code:

```
                void DispatchMessage(int id, long lparam, double dparam, string sparam)
                        {
                                ulong   ticket;
                                double  price, tp, sl;
                                bool            isBuy,
                                                        bKeyBuy,
                                                        bKeySell,
                                                        bEClick;
                                long            info;
                                datetime        dt;
                                uint            mKeys;
                                eIndicatorTrade         it;
                                eEventType                      ev;

                                static bool bMounting = false, bIsDT = false, bIsMove = false;
                                static double leverange = 0, valueTp = 0, valueSl = 0, memLocal = 0;

                                switch (id)
                                {
                                        case CHARTEVENT_MOUSE_MOVE:

// ... The rest of the code...
```

The solution is the highlighted code. But how is it possible? Remember that when you are placing a pending order, the system manages to create and manipulate the data so that in the end you have a representation of the labels on the chart indicating where the order will be placed. This is done in the following code:

```
case CHARTEVENT_MOUSE_MOVE:
        Mouse.GetPositionDP(dt, price);
        mKeys   = Mouse.GetButtonStatus();
        bEClick  = (mKeys & 0x01) == 0x01;    //left mouse click
        bKeyBuy  = (mKeys & 0x04) == 0x04;    //SHIFT pressed
        bKeySell = (mKeys & 0x08) == 0x08;    //CTRL pressed
        if (bKeyBuy != bKeySell)
        {
                if (!bMounting)
                {
                        Mouse.Hide();
                        bIsDT = Chart.GetBaseFinance(leverange, valueTp, valueSl);
                        valueTp = Terminal.AdjustPrice(valueTp * Terminal.GetAdjustToTrade() / leverange);
                        valueSl = Terminal.AdjustPrice(valueSl * Terminal.GetAdjustToTrade() / leverange);
                        m_TradeLine.SpotLight(MountName(def_IndicatorTicket0, IT_PENDING, EV_LINE));
                        bMounting = true;
                }
                tp = price + (bKeyBuy ? valueTp : (-valueTp));
                sl = price + (bKeyBuy ? (-valueSl) : valueSl);
                UpdateInfosIndicators(0, def_IndicatorTicket0, price, tp, sl, leverange, bKeyBuy);
                if ((bEClick) && (memLocal == 0)) CreateOrderPendent(leverange, bKeyBuy, memLocal = price, tp, sl, bIsDT);
                }else if (bMounting)
                {
                        UpdateInfosIndicators(0, def_IndicatorTicket0, 0, 0, 0, 0, false);
                        Mouse.Show();
                        memLocal = 0;
                        bMounting = false;

//... Rest of the code...
```

Depending on whether you press SHIFT to buy or CTRL to sell, the system will create a representation of the pending order being created. It will be displayed directly on the chart. The values ​​to be used as Take Profit or Stop Loss are captured within the Chart Trade, then you move the representation to the point where you want to place the order and after that, left mouse button click tells the system that an order must be placed there. As soon as the mouse moves again, the label used to represent the order is removed, and the order indication is left.

So far, there was nothing special. But if you look into the code, you will see the following:

```
// ... CHARTEVENT_MOUSE_MOVE code....

        }else if ((!bMounting) && (bKeyBuy == bKeySell))
        {
                if (bEClick)
                {
                        bIsMove = false;
                        m_TradeLine.SpotLight();
                }
                MoveSelection(price, mKeys);
        }
break;

// ... The rest of the code....
```

So, when we are not creating a pending order, and the keys are released, we send the mouse price position to some label that may be selected. This is done in the highlighted line. As soon as we left click, we end this transfer, the indicator will be deselected, and this will change the status of a variable that is currently local, **_bIsMove_**. But let's change that. This variable has its state modified only by another event inside the DispatchMessage function. This event looks like this:

```
// ... Code ....

        case EV_MOVE:
                if (bIsMove)
                {
                        m_TradeLine.SpotLight();
                        bIsMove = false;
                }else
                {
                        m_TradeLine.SpotLight(MountName(ticket, it, EV_LINE));
                        bIsMove = true;
                }
        break;
```

This code will change the state of the _**bIsMove**_ variable and at the same time it will change the label to show whether it is selected or not.

Thus, if we make this variable visible throughout the class, we can separate the ghost from the real label, and it will be possible to manipulate either only the real or only the ghost — it depends on what you find most interesting. But here we will change the real label, while the ghost will show what the trade server sees.

This way, we don't have to mess around with the code much, just adjust a few details. Once the left click is done and an object has been moved, an order to change a pending order or a stop level will be sent.

Let's see how this is done in practice. First, let's create a private variable.

```
bool    m_bIsMovingSelect;
```

This reflects what I explained above. But it needs to be initialized.

```
C_IndicatorTradeView() : m_bIsMovingSelect(false) {}
```

Now, we go to DispatchMessage and use it instead of bIsMove.

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{
        ulong   ticket;

// ... Internal code...

        switch (id)
        {
                case CHARTEVENT_MOUSE_MOVE:

// ... Internal code...

                                }else if ((!bMounting) && (bKeyBuy == bKeySell))
                                {
                                        if (bEClick)
                                        {
                                                m_bIsMovingSelect = false;
                                                m_TradeLine.SpotLight();
                                        }
                                        MoveSelection(price, mKeys);
                                }
                                break;

// ... Internal code...

                case CHARTEVENT_OBJECT_CLICK:
                        if (GetIndicatorInfos(sparam, ticket, price, it, ev)) switch (ev)
                        {

// ... Internal code....

                                case EV_MOVE:
                                        if (m_bIsMovingSelect)
                                        {
                                                m_TradeLine.SpotLight();
                                                m_bIsMovingSelect = false;
                                        }else
                                        {
                                                m_TradeLine.SpotLight(MountName(ticket, it, EV_LINE));
                                                m_bIsMovingSelect = true;
                                        }
                                        break;
                        }
                        break;
                }
}
```

The changes are highlighted. Thus, now the whole class knows when we work or not with some label chosen by the user. This way we can separate the ghost from the real label and have a more accurate representation.

### 2.0.3. Moving only what matters

In the previous two topics, we created and made some fixes to the system in order to have a ghost indication label. Now we need to move the components, and for this we will have to do some work. The first step is to create a structure to store various information we need to avoid excessive calls between functions. The structure is shown below:

```
struct st00
{
        eIndicatorTrade it;
        bool            bIsMovingSelect,
                        bIsBuy;
        ulong           ticket;
        double          vol,
                        pr,
                        tp,
                        sl;
}m_InfoSelection;
```

The highlighted segment existed in the code previously, but now it is part of the structure. Don't worry. As we go further, it will become clearer why the structure has these elements.

Here we will consider the functions that have been modified and require explanation. The first one is SetTextValue.

```
void SetTextValue(ulong ticket, eIndicatorTrade it, double value0, double value1 = 0.0, double priceOpen = 0.0)
{
        double finance;

        switch (it)
        {
                case IT_RESULT  :
                        PositionAxlePrice(ticket, it, priceOpen);
                        PositionAxlePrice(ticket, IT_PENDING, 0);
                        m_EditInfo2.SetTextValue(MountName(ticket, it, EV_PROFIT), value1);
                case IT_PENDING:
                        value0 = value0 / Terminal.GetVolumeMinimal();
                        m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT), value0, def_ColorVolumeEdit);
                        if (!m_InfoSelection.bIsMovingSelect) m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT, true), value0, def_IndicatorGhostColor);
                        break;
                case IT_TAKE    :
                case IT_STOP    :
                        finance = (value1 / Terminal.GetAdjustToTrade()) * value0;
                        m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT), finance);
                        if (!m_InfoSelection.bIsMovingSelect) m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT, true), finance, def_IndicatorGhostColor);
                        break;
        }
}
```

When we are moving something, we don't want the ghost to follow the new data, we want it to stay still, showing where the label was. This can be easily done by adding the highlighted points. Thus, the ghost stands still while the label freely moves. Here we will not have the movement itself, but the indication of the values ​​that are on the server.

This is followed by the movement code shown below:

```
void MoveSelection(double price)
{
        double tp, sl;

        if (!m_InfoSelection.bIsMovingSelect) return;
        switch (m_InfoSelection.it)
        {
                case IT_TAKE:
                        UpdateInfosIndicators(0, m_InfoSelection.ticket, m_InfoSelection.pr, price, m_InfoSelection.sl, m_InfoSelection.vol, m_InfoSelection.bIsBuy);
                        break;
                case IT_STOP:
                        UpdateInfosIndicators(0, m_InfoSelection.ticket, m_InfoSelection.pr, m_InfoSelection.tp, price, m_InfoSelection.vol, m_InfoSelection.bIsBuy);
                        break;
                case IT_PENDING:
                        tp = (m_InfoSelection.tp == 0 ? 0 : price + m_InfoSelection.tp - m_InfoSelection.pr);
                        sl = (m_InfoSelection.sl == 0 ? 0 : price + m_InfoSelection.sl - m_InfoSelection.pr);
                        UpdateInfosIndicators(0, m_InfoSelection.ticket, price, tp, sl, m_InfoSelection.vol, m_InfoSelection.bIsBuy);
                        break;
        }
}
```

Please pay attention to the highlighted part. It adjusts the movement so that the system correctly indicates the data. This may seem strange, but the UpdateINfosIndicators function has another fix. If we do not do it now, we will have problems later. Other functions are quite simple and do not require explanation.

```
void SetPriceSelection(double price)
{
        bool isPending;
        if (!m_InfoSelection.bIsMovingSelect) return;
        isPending = OrderSelect(m_InfoSelection.ticket);
        m_InfoSelection.bIsMovingSelect = false;
        m_TradeLine.SpotLight();
        switch (m_InfoSelection.it)
        {
                case IT_TAKE:
                        if (isPending) ModifyOrderPendent(m_InfoSelection.ticket, m_InfoSelection.pr, price, m_InfoSelection.sl);
                        else ModifyPosition(m_InfoSelection.ticket, price, m_InfoSelection.sl);
                        break;
                case IT_STOP:
                        if (isPending) ModifyOrderPendent(m_InfoSelection.ticket, m_InfoSelection.pr, m_InfoSelection.tp, price);
                        else ModifyPosition(m_InfoSelection.ticket, m_InfoSelection.tp, price);
                        break;
                case IT_PENDING:
                        ModifyOrderPendent(m_InfoSelection.ticket, price, (m_InfoSelection.tp == 0 ? 0 : price + m_InfoSelection.tp - m_InfoSelection.pr), (m_InfoSelection.sl == 0 ? 0 : price + m_InfoSelection.sl - m_InfoSelection.pr));
                        break;
        }
}
```

The above function will inform the server about what is happening and what the new data is. Please note that some indicator must be selected in the highlighted lines, otherwise the request to the server will not be made.

The last recommended function is shown below:

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{
        ulong   ticket;
        double  price;
        bool    bKeyBuy,
                bKeySell,
                bEClick;
        datetime dt;
        uint    mKeys;
        char    cRet;
        eIndicatorTrade         it;
        eEventType              ev;

        static bool bMounting = false, bIsDT = false;
        static double valueTp = 0, valueSl = 0, memLocal = 0;

        switch (id)
        {
                case CHARTEVENT_MOUSE_MOVE:
                        Mouse.GetPositionDP(dt, price);
                        mKeys   = Mouse.GetButtonStatus();
                        bEClick  = (mKeys & 0x01) == 0x01;    //Left mouse click
                        bKeyBuy  = (mKeys & 0x04) == 0x04;    //Pressed SHIFT
                        bKeySell = (mKeys & 0x08) == 0x08;    //Pressed CTRL
                        if (bKeyBuy != bKeySell)
                        {
                                if (!bMounting)
                                {
                                        Mouse.Hide();
                                        bIsDT = Chart.GetBaseFinance(m_InfoSelection.vol, valueTp, valueSl);
                                        valueTp = Terminal.AdjustPrice(valueTp * Terminal.GetAdjustToTrade() / m_InfoSelection.vol);
                                        valueSl = Terminal.AdjustPrice(valueSl * Terminal.GetAdjustToTrade() / m_InfoSelection.vol);
                                        m_TradeLine.SpotLight(MountName(def_IndicatorTicket0, IT_PENDING, EV_LINE));
                                        m_InfoSelection.it = IT_PENDING;
                                        m_InfoSelection.ticket = def_IndicatorTicket0;
                                        m_InfoSelection.bIsMovingSelect = true;
                                        m_InfoSelection.pr = price;
                                        bMounting = true;
                                }
                                m_InfoSelection.tp = m_InfoSelection.pr + (bKeyBuy ? valueTp : (-valueTp));
                                m_InfoSelection.sl = m_InfoSelection.pr + (bKeyBuy ? (-valueSl) : valueSl);
                                m_InfoSelection.bIsBuy = bKeyBuy;
                                MoveSelection(price);
                                if ((bEClick) && (memLocal == 0))
                                {
                                        MoveSelection(0);
                                        m_InfoSelection.bIsMovingSelect = false;
                                        CreateOrderPendent(m_InfoSelection.vol, bKeyBuy, memLocal = price,  price + m_InfoSelection.tp - m_InfoSelection.pr, price + m_InfoSelection.sl - m_InfoSelection.pr, bIsDT);
                                }
                        }else if (bMounting)
                        {
                                MoveSelection(0);
                                m_InfoSelection.bIsMovingSelect = false;
                                Mouse.Show();
                                memLocal = 0;
                                bMounting = false;
                        }else if ((!bMounting) && (bKeyBuy == bKeySell))
                        {
                                if (bEClick) SetPriceSelection(price); else MoveSelection(price);
                        }
                        break;
                case CHARTEVENT_OBJECT_DELETE:
                        if (GetIndicatorInfos(sparam, ticket, price, it, ev))
                        {
                                CreateIndicatorTrade(ticket, it);
                                GetInfosTradeServer(ticket);
                                m_InfoSelection.bIsMovingSelect = false;
                                UpdateInfosIndicators(0, ticket, m_InfoSelection.pr, m_InfoSelection.tp, m_InfoSelection.sl, m_InfoSelection.vol, m_InfoSelection.bIsBuy);
                        }
                        break;
                case CHARTEVENT_CHART_CHANGE:
                        ChartSetInteger(ChartID(), CHART_SHOW_OBJECT_DESCR, false);
                        ReDrawAllsIndicator();
                        break;
                case CHARTEVENT_OBJECT_CLICK:
                        if (GetIndicatorInfos(sparam, ticket, price, it, ev)) switch (ev)
                        {
                                case EV_CLOSE:
                                        if ((cRet = GetInfosTradeServer(ticket)) != 0) switch (it)
                                        {
                                                case IT_PENDING:
                                                case IT_RESULT:
                                                        if (cRet < 0) RemoveOrderPendent(ticket); else ClosePosition(ticket);
                                                        break;
                                                case IT_TAKE:
                                                case IT_STOP:
                                                        m_InfoSelection.bIsMovingSelect = true;
                                                        SetPriceSelection(0);
                                                        break;
                                        }
                                        break;
                                case EV_MOVE:
                                        if (m_InfoSelection.bIsMovingSelect)
                                        {
                                                m_TradeLine.SpotLight();
                                                m_InfoSelection.bIsMovingSelect = false;
                                        }else
                                        {
                                                m_InfoSelection.ticket = ticket;
                                                m_InfoSelection.it = it;
                                                if (m_InfoSelection.bIsMovingSelect = (GetInfosTradeServer(ticket) != 0))
                                                m_TradeLine.SpotLight(MountName(ticket, it, EV_LINE));
                                        }
                                        break;
                        }
                        break;
        }
}
```

There are a few things to note about this function but pay attention to the difference from the previous version: we're reusing a lot more code here. Thus, if somewhere else we have an error in the code, we can quickly notice and fix it. Previously, this function code was a little unstable, with some fixes and defects in other parts. One of the main points was that when we placed a pending order, there was a separate system for moving indicators, and now we have a single system, the same one that is used to place orders on the chart. It is also used to move objects, i.e., now the same CHARTEVENT\_MOUSE\_MOVE event is used both for placing a pending order and for moving objects. It may seem like a small thing, but it makes any changes to the code visible, and if we have a problem, it will show up as long as we use the mouse event.

### Conclusion

Please watch the video below to have a clearer idea of what's going on with the changes made. You will see that now we only need a few more details to make the EA fully complete in terms of the order system.

YouTube

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10563](https://www.mql5.com/pt/articles/10563)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10563.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Sistema\_de\_ordens\_o\_VI\_v.zip](https://www.mql5.com/en/articles/download/10563/ea_-_sistema_de_ordens_o_vi_v.zip "Download EA_-_Sistema_de_ordens_o_VI_v.zip")(12029.26 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/434066)**
(2)


![Ovm6](https://c.mql5.com/avatar/avatar_na2.png)

**[Ovm6](https://www.mql5.com/en/users/ovm6)**
\|
5 May 2023 at 16:37

Hi Jose,

how can i use this EA on US30 index or S&P500? When i make Shift an left Click, comes an [error 10015](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes "MQL5 Documentation: Invalid Price in Request")? How can i fix that?

Best regards

Florian

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
6 May 2023 at 12:51

**Ovm6 [#](https://www.mql5.com/en/forum/434066#comment_46716354) :**

Hi Jose,

how can i use this EA on US30 index or S&P500? When i make Shift an left Click, comes an [error 10015](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes "MQL5 Documentation: Invalid Price in Request") ? How can i fix that?

Best regards

Florian

I don't know exactly what could be going on. Since this error you are reporting would indicate a failure in the price calculation. But all the calculation is performed in the DispatchMessage procedure in the C\_IndicatorTradeView class. This calculation takes into account the data provided by the server, coming from the C\_Terminal class.

Perhaps what could be happening is that you are using the wrong contract. I don't know if these assets you mention have an expiration date. If this is the case, you will have to add the correct contract search rules in the CurrentSymbol procedure within the C\_Terminal class. Thus, the application will do all the calculations and correctly send the order to the trading server.

You can use newer code. Since this article is completely obsolete. See my most recent articles, as the codes are much better and simpler to understand as I'm focusing much more on explaining them. This makes it simple for you to adapt the system to what you need.

![Experiments with neural networks (Part 2): Smart neural network optimization](https://c.mql5.com/2/51/neural_network_experiments_p2.png)[Experiments with neural networks (Part 2): Smart neural network optimization](https://www.mql5.com/en/articles/11186)

In this article, I will use experimentation and non-standard approaches to develop a profitable trading system and check whether neural networks can be of any help for traders. MetaTrader 5 as a self-sufficient tool for using neural networks in trading.

![DoEasy. Controls (Part 11): WinForms objects — groups, CheckedListBox WinForms object](https://c.mql5.com/2/47/MQL5-avatar-doeasy-library-2__5.png)[DoEasy. Controls (Part 11): WinForms objects — groups, CheckedListBox WinForms object](https://www.mql5.com/en/articles/11194)

The article considers grouping WinForms objects and creation of the CheckBox objects list object.

![Data Science and Machine Learning (Part 07): Polynomial Regression](https://c.mql5.com/2/49/Data_Science_07_Polynomial_Regression_60x60.png)[Data Science and Machine Learning (Part 07): Polynomial Regression](https://www.mql5.com/en/articles/11477)

Unlike linear regression, polynomial regression is a flexible model aimed to perform better at tasks the linear regression model could not handle, Let's find out how to make polynomial models in MQL5 and make something positive out of it.

![Developing a trading Expert Advisor from scratch (Part 22): New order system (V)](https://c.mql5.com/2/47/development__5.png)[Developing a trading Expert Advisor from scratch (Part 22): New order system (V)](https://www.mql5.com/en/articles/10516)

Today we will continue to develop the new order system. It is not that easy to implement a new system as we often encounter problems which greatly complicate the process. When these problems appear, we have to stop and re-analyze the direction in which we are moving.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/10563&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051677986295501957)

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