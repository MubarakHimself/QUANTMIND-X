---
title: Developing a trading Expert Advisor from scratch (Part 28): Towards the future (III)
url: https://www.mql5.com/en/articles/10635
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:45:54.067017
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/10635&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051666995474191439)

MetaTrader 5 / Examples


### Introduction

When starting developing the order system, after the article [Developing a trading Expert Advisor from scratch (Part18)](https://www.mql5.com/en/articles/10462), I had no idea how long it would take to get to this point. We have gone through various moments, changes, amendments, etc. I showed you how to do some specific things, like marking things or making the system more intuitive. But there were also things I couldn't show at this stage because the path wasn't fully prepared. This journey allowed us to build the concept in such a way that everyone can understand the idea and know how the system works.

In all the previous articles, I have set the stage for us to come to this article with the same level of understanding of how the system works. So, I hope this material will not be extremely confusing or complex. There was one question from the very beginning, and I avoided analyzing it in detail. But it is very important for more experienced traders. At first glance, this may seem silly, but when it will come time to trade and we will understand that we are missing something in the EA. Then we will ask ourselves "what is missing here?". I am talking about a way to restore Take Profit and Stop Loss values which were deleted for some reason and which we want to restore on the chart.

If you have ever tried to do this, then you can understand that this is a rather difficult and slow task, since you need to "follow a certain scenario" in order for everything to work out well, otherwise we will end up making mistakes all the time.

The MetaTrader 5 provides a system of tickets which allows creating and correcting order values. The idea is to have an Expert Advisor that would make the same ticket system faster and more efficient. The MetaTrader 5 system is not perfect; sometimes it can be slower and more error prone than using the EA we are developing.

But until now, I have never explained how to generate values for the TP and SL levels (Take Profit and Stop Loss). I think the deletion of stop levels is clear and intuitive enough. But how to implement it, i.e. how should we proceed to set stop or to restore them right on the chart? This is something very intriguing, which sets some questions, and of course, this is the reason for creating this article: to show one of the many ways to create the stop levels right on the chart, without resorting to any external resource, just using the EA's order system.

### 2.0. Getting started: Implementation

First, we need to force the EA to stop checking, which it has been doing for a long time since the first days of its development. To remove this check, you need to remove all the crossed out code and add the highlighted code:

```
inline double SecureChannelPosition(void)
{
        double Res = 0, sl, profit, bid, ask;
        ulong ticket;

        bid = SymbolInfoDouble(Terminal.GetSymbol(), SYMBOL_BID);
        ask = SymbolInfoDouble(Terminal.GetSymbol(), SYMBOL_ASK);
        for (int i0 = PositionsTotal() - 1; i0 >= 0; i0--) if (PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                IndicatorAdd(ticket = PositionGetInteger(POSITION_TICKET));
                SetTextValue(ticket, IT_RESULT, PositionGetDouble(POSITION_VOLUME), Res += PositionGetDouble(POSITION_PROFIT), PositionGetDouble(POSITION_PRICE_OPEN));
                SetTextValue(ticket, IT_RESULT, PositionGetDouble(POSITION_VOLUME), profit = PositionGetDouble(POSITION_PROFIT), PositionGetDouble(POSITION_PRICE_OPEN));
                sl = PositionGetDouble(POSITION_SL);
                if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {
                        if (ask < sl) ClosePosition(ticket);
                }else
                {
                        if ((bid > sl) && (sl > 0)) ClosePosition(ticket);
                }
                Res += profit;
        }
        return Res;
};
```

The crossed out code will not cease to exist but it will be back at another moment. However, at this time it is more of a hindrance than a benefit. Once that's done, we can start thinking about how we're going to implement the TP and SL system right on the chart, without the help of any resource other than the EA.

Each developer will have their own idea for solving this problem: some of them will be easier for a trader to understand, while others will be more difficult; some will be harder to put into practice, while others will be easier. I am not telling that the way I will use and show here is the most appropriate or the easiest one, but it is by far the best adapted to my way of working and using the platform. Also, I will not need to create any new elements. We will only correct some things in the code.

### 2.0.1. Modeling the drag system

The EA code itself at the current stage of development provides some hints on how we should model the system that we are going to create. Look at the following code:

```
#define macroUpdate(A, B) if (B > 0) {                                                                  \
                if (b0 = (macroGetLinePrice(ticket, A) == 0 ? true : b0)) CreateIndicator(ticket, A);   \
                PositionAxlePrice(ticket, A, B);                                                        \
                SetTextValue(ticket, A, vol, (isBuy ? B - pr : pr - B));                                \
                                     } else RemoveIndicator(ticket, A);

                void UpdateIndicators(ulong ticket, double tp, double sl, double vol, bool isBuy)
                        {
                                double pr;
                                bool b0 = false;

                                if (ticket == def_IndicatorGhost) pr = m_Selection.pr; else
                                {
                                        pr = macroGetLinePrice(ticket, IT_RESULT);
                                        if ((pr == 0) && (macroGetLinePrice(ticket, IT_PENDING) == 0))
                                        {
                                                CreateIndicator(ticket, IT_PENDING);
                                                PositionAxlePrice(ticket, IT_PENDING, m_Selection.pr);
                                                ChartRedraw();
                                        }
                                        pr = (pr > 0 ? pr : macroGetLinePrice(ticket, IT_PENDING));
                                        SetTextValue(ticket, IT_PENDING, vol);
                                }
                                if (m_Selection.tp > 0) macroUpdate(IT_TAKE, tp);
                                if (m_Selection.sl > 0) macroUpdate(IT_STOP, sl);
                                if (b0) ChartRedraw();
                        }
#undef macroUpdate
```

The highlighted lines contain a macro that will perform the task. We need to modify it so that it provides the necessary help to implement what we need, namely the stop level indicator. Take a closer look at the macro code. It is shown below:

```
#define macroUpdate(A, B){ if (B > 0) {                                                                 \
                if (b0 = (macroGetLinePrice(ticket, A) == 0 ? true : b0)) CreateIndicator(ticket, A);   \
                PositionAxlePrice(ticket, A, B);                                                        \
                SetTextValue(ticket, A, vol, (isBuy ? B - pr : pr - B));                                \
                                        } else RemoveIndicator(ticket, A); }
```

We do the following: When the B value, which can be Take Profit or Stop Loss, is greater than 0, check if the indicator is on the chart. If not, create it, place it, and set the value it will display. If the B value is equal to 0, then we will completely remove the indicator from the chart, and we will do it at the highlighted point in the macro code. But would it be enough if, instead of completely removing the indicator from the chart, we keep its element, and if this element can be configured to display what we want to do (i.e. create the missing stop(s) for an order or position), the element will turn back to an order or to an OCO position? YES, that would be enough and that's the idea: leave an element, in this case an object that is used to move the stop levels and create the stop level that we are missing. We only need to drag this element and the limit will be created. This is the theoretical framework that we will use to build the system.

But by simply doing this we cannot obtain everything we need. There is one more change required. We will do this before continuing. This change is highlighted in the code below:

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{

//... Internal code...

        m_Selection.tp = (valueTp == 0 ? 0 : m_Selection.pr + (bKeyBuy ? valueTp : (-valueTp)));
        m_Selection.sl = (valueSl == 0 ? 0 : m_Selection.pr + (bKeyBuy ? (-valueSl) : valueSl));

// ... The rest of the code ...

}
```

We check if the initial values entered by Chart Trade are null or not. If so, no indicator will be created and the chart will display only the entry point.

This provides a more linear work in the rest of the system. If no stop level is specified, the entire order model will have the same behavior as when the value is specified from the very beginning, when we are going to place a pending order on the chart.

Thus, not a single trader will wonder: What are these figures hanging on the order or position? Because the trader will know that they represent the elements that can be moved on the chart.

But despite all this, we still have a small problem, for which we will change two macros. See below:

```
#define macroSetLinePrice(ticket, it, price)    ObjectSetDouble(Terminal.Get_ID(), macroMountName(ticket, it, EV_LINE), OBJPROP_PRICE, price)
#define macroGetLinePrice(ticket, it)           ObjectGetDouble(Terminal.Get_ID(), macroMountName(ticket, it, EV_LINE), OBJPROP_PRICE)
#define macroGetPrice(ticket, it, ev)           ObjectGetDouble(Terminal.Get_ID(), macroMountName(ticket, it, ev), OBJPROP_PRICE)
```

This modification is very important for the rest of the system that we will be building. But why am I pulling out the cut lines? The reason is that we need to make the system even more flexible, and in order to achieve this, the crossed out code was removed and a highlighted line appeared in its place. Why do we need macroGetPrice if we don't have a macro to assign a value to the price? In fact, there is only one point that actually makes this price adjustment, writing it on the chart object. This point can be seen in the code below:

```
#define macroSetPrice(ticket, it, ev, price) ObjectSetDouble(Terminal.Get_ID(), macroMountName(ticket, it, ev), OBJPROP_PRICE, price)
//---

// ... Additional code inside the class....

//---
inline void PositionAxlePrice(ulong ticket, eIndicatorTrade it, double price)
                        {
                                int x, y, desl;

                                ChartTimePriceToXY(Terminal.Get_ID(), 0, 0, price, x, y);
                                if (it != IT_RESULT) macroSetPrice(ticket, it, EV_MOVE, price);
                                macroSetPrice(ticket, it, EV_LINE, price);
                                macroSetAxleY(it);
                                switch (it)
                                {
                                        case IT_TAKE: desl = 160; break;
                                        case IT_STOP: desl = 270; break;
                                        default: desl = 0;
                                }
                                macroSetAxleX(it, desl + (int)(ChartGetInteger(Terminal.Get_ID(), CHART_WIDTH_IN_PIXELS) * 0.2));
                        }
```

Because of this there is no need for the macro to adjust the price on objects to be visible to all other places in the code. Actually, it doesn't really even have to be a macro right now, but I'll leave it as it is to reduce the chance of an error later on when this code changes.

Now that our system is up to date, we can move on to the next topic and get everything working as planned.

### 2.0.2. New update feature

I already mentioned this in the previous topic: all we need to do is configure the Update function while the EA will solve all the related problems. Since the main focus is only on the Take Profit and Stop Loss indicators, and they are executed inside the macro, we just need to set up the macro correctly.

But there is an issue that is yet to be resolved. We need to create the move button independently from the rest of the indicator. To do this, let us isolate the button creation code:

```
// ... class code ...

#define def_ColorLineTake       clrDarkGreen
#define def_ColorLineStop       clrMaroon

// ... class code ....

inline void CreateBtnMoveIndicator(ulong ticket, eIndicatorTrade it, color C = clrNONE)
                        {
                                string sz0 = macroMountName(ticket, it, EV_MOVE);

                                ObjectDelete(Terminal.Get_ID(), macroMountName(ticket, it, EV_MOVE));
                                m_BtnMove.Create(ticket, sz0, "Wingdings", "u", 17, (C == clrNONE ? (it == IT_TAKE ? def_ColorLineTake : def_ColorLineStop) : C));
                                m_BtnMove.Size(sz0, 21, 23);
                        }

// ... the rest of the class code ...
```

This code will create only the move button and nothing else. It is very simple and straightforward. I even thought about leaving this code as a macro, but decided to make it as a function. It is declared as a built-in function, so the compiler will treat it the same as it would treat a macro. To make life easier, the same code part has two new definitions, because at some point we will only create a move button and we want it to have the same colors that are used throughout the system. It is not desirable that the system behave or look differently in similar cases. To reduce trouble, we leave the colors as shown above.

Now we can go to the Update function; its full code is shown above. Note that the only difference between the version below and the one presented earlier in the article is the highlighted code. This code is a macro used by the Update function itself.

```
#define macroUpdate(A, B){                                                                                                      \
                if (B == 0) {   if (macroGetPrice(ticket, A, EV_LINE) > 0) RemoveIndicator(ticket, A);                          \
                                if (macroGetPrice(ticket, A, EV_MOVE) == 0) CreateBtnMoveIndicator(ticket, A);                  \
                            } else if (b0 = (macroGetPrice(ticket, A, EV_LINE) == 0 ? true : b0)) CreateIndicator(ticket, A);   \
                PositionAxlePrice(ticket, A, (B == 0 ? pr : B));                                                                \
                SetTextValue(ticket, A, vol, (isBuy ? B - pr : pr - B));                                                        \
                        }

                void UpdateIndicators(ulong ticket, double tp, double sl, double vol, bool isBuy)
                        {
                                double pr;
                                bool b0 = false;

                                if (ticket == def_IndicatorGhost) pr = m_Selection.pr; else
                                {
                                        pr = macroGetPrice(ticket, IT_RESULT, EV_LINE);
                                        if ((pr == 0) && (macroGetPrice(ticket, IT_PENDING, EV_MOVE) == 0))
                                        {
                                                CreateIndicator(ticket, IT_PENDING);
                                                PositionAxlePrice(ticket, IT_PENDING, m_Selection.pr);
                                                ChartRedraw();
                                        }
                                        pr = (pr > 0 ? pr : macroGetPrice(ticket, IT_PENDING, EV_MOVE));
                                        SetTextValue(ticket, IT_PENDING, vol);
                                }
                                macroUpdate(IT_TAKE, tp);
                                macroUpdate(IT_STOP, sl);
                                if (b0) ChartRedraw();
                        }
#undef macroUpdate
```

Let's look at this macro in more detail to understand what is actually happening. Understanding this is necessary in order to be able to solve some of the problems that still arise when using the system. It is not ready yet and we need to make some more changes.

In the first step, we have the following behavior: when removing one of the stop orders (Take Profit or Stop Loss), we will immediately remove the relevant indicator from the symbol chart. To do this, we check whether there is a line that is one of the points indicating the presence of the indicator. Then we check if there is a movement object of this indicator. If it does not exist, create it. So, we will not have the indicator on the chart, but we will have a remainder of it still present on the chart, which is the movement object

The second step occurs in case of creating a stop level: the order on the server will have a stop level that should appear on the chart. In this case the object that represented the movement will be removed and a complete indicator will be created and placed in the appropriate place, indicating where the current stop level (Take Profit or Stop Loss) will be.

In the last step, we position the indicator at the correct point. One interesting detail: if the stop level indicator is only an object representing the possibility of movement, then the point at which it will be placed will be exactly the price of the order or position. In other words, the move object that allows creating Take Profit and Stop Loss will be linked to the price line of the order or position to which it belongs. Thus, it will be easy to notice if an order or position has one of the stop levels missing.

This is basically what we need to do: when we click on the object that indicates the movement, a ghost is created, as usual, and at the same a representation of the complete indicator is also created. This is done without adding or modifying any code. From now on, we can move and adjust the stop levels in the usual way, just as before. But until we click a certain point, the stop order will not exist. This is clearly shown in the demo video at the end of the article, where I show how the system works on a real account.

Although everything seems to be fine, but here we have some inconsistencies here which force us to create or rather change the code at some points. We will see it in the next topic as this step is over by now.

### 2.0.3. Solving the inconvenience of floating indicators

The first of the inconveniences appears when we are in the floating indicator mode: it is on the chart but not on the server. For more information, please see articles [Developing a trading Expert Advisor from scratch (Part 26)](https://www.mql5.com/en/articles/10620) and [(Part 27)](https://www.mql5.com/en/articles/10630), in which I show how the floating indicator works and how it was implemented. These indicators are useful and so they will not be removed from the EA. But they do not suit the system that we have seen above since they their operation is different from the indicators that actually represent orders and positions existing on the trade server. To solve the problems that appear when using a floating indicator, we will have to get to the DispatchMessage function and adjust things there, as shown below.

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{
        ulong   ticket;
        double  price;

// ... Internal code...

        switch (id)
        {

// ... Internal code...

                case CHARTEVENT_OBJECT_CLICK:
                        if (GetIndicatorInfos(sparam, ticket, it, ev)) switch (ev)
                        {
                                case EV_TYPE:
                                        if (ticket == def_IndicatorFloat)
                                        {
                                                macroGetDataIndicatorFloat;
                                                m_Selection.tp = (m_Selection.tp == 0 ? 0 : m_Selection.pr + (MathAbs(m_Selection.tp - m_Selection.pr) * (m_Selection.bIsBuy ? 1 : -1)));
                                                m_Selection.sl = (m_Selection.sl == 0 ? 0 : m_Selection.pr + (MathAbs(m_Selection.sl - m_Selection.pr) * (m_Selection.bIsBuy ? -1 : 1)));
                                                m_Selection.ticket = 0;
                                                UpdateIndicators(def_IndicatorFloat, m_Selection.tp, m_Selection.sl, m_Selection.vol, m_Selection.bIsBuy);
                                        } else m_BtnInfoType.SetStateButton(sparam, !m_BtnInfoType.GetStateButton(sparam));
                                        break;
                                case EV_DS:
                                        if (ticket != def_IndicatorFloat) m_BtnInfo_DS.SetStateButton(sparam, !m_BtnInfo_DS.GetStateButton(sparam));
                                        break;
                                case EV_CLOSE:
                                        if (ticket == def_IndicatorFloat)
                                        {
                                                macroGetDataIndicatorFloat;
                                                RemoveIndicator(def_IndicatorFloat, it);
                                                if (it != IT_PENDING) UpdateIndicators(def_IndicatorFloat, (it == IT_TAKE ? 0 : m_Selection.tp), (it == IT_STOP ? 0 : m_Selection.sl), m_Selection.vol, m_Selection.bIsBuy);
                                        }else if ((cRet = GetInfosTradeServer(ticket)) != 0) switch (it)

// ... The rest of the code...
```

By making the changes highlighted above we practically eliminate any other problem related to the setup of floating indicators, as now we have the same way to adjust floating indicator data and the one that represents data existing on the trading server. But this does not completely solve our problems. We have to solve another inconvenience that was persisting for a long time. To discuss this, let's move on to the next topic, because it deserves a separate discussion.

### 2.0.4. The drawback of the negative Take Profit value

The last drawback we are discussing in this article is that the Take Profit value can often be configured to be negative, and this has been happening for quite a long time. But this makes no sense for the trading system: if you try to send the value to the server, an error message will be returned. Therefore, we have to fix this and also solve another problem, which is that a pending order can have a stop value changed to positive.

The EA allows doing so now, and what's worse, the order system indicates that this value is on the server, while in fact the server returns an error, and the EA simply ignores it. The problem is more complicated in the case of pending orders, because in the case of positions, the behavior should be different, and this bug has not yet been fixed. As soon as we have the ability to define the stop levels directly on the chart, this drawback should disappear.

It should be mentioned here that in case of an open position we can have a stop loss with a positive value, and this indicates that if the stop loss triggers, we will have the relevant value credited to our account. But for a pending order this will be an error which will prevent the server from creating a correct order. To solve this problem, we must check the Take Profit value: when it becomes equal to or less than 0, then we must prevent it from changing to a smaller value. Also, for pending orders, we should not allow Stop Loss to be greater than 0. In fact, we force the EA to use a minimum allowed value when condition 0 is met. In this case, the order or position will make some sense for the trading system, but it makes no sense to open a position with a stop loss or take profit equal to the open price.

To make this as easy as possible, we need to create a variable in the system, which can be seen below:

```
struct st00
{
        eIndicatorTrade it;
        bool            bIsBuy,
                        bIsDayTrade;
        ulong           ticket;
        double          vol,
                        pr,
                        tp,
                        sl,
                        MousePrice;
}m_Selection;
```

Why don't we just change the price point on the mouse line? The reason is that to correctly manipulate the mouse it is necessary to use a system call, that is, it would be necessary to manipulate the mouse position values via the WINDOWS API, and this would force us to enable the use of external dlls, and I don't want to do that. This way it is simpler to assemble a local value inside the EA, and the highlighted data will store this value for us.

This value will be used in three different places. The first place is in the movement function. The code below shows exactly where this is happening:

```
void MoveSelection(double price)
{
        if (m_Selection.ticket == 0) return;
        switch (m_Selection.it)
        {
                case IT_TAKE:
                        UpdateIndicators(m_Selection.ticket, price, m_Selection.sl, m_Selection.vol, m_Selection.bIsBuy);
                        break;
                case IT_STOP:
                        UpdateIndicators(m_Selection.ticket, m_Selection.tp, price, m_Selection.vol, m_Selection.bIsBuy);
                        break;
                case IT_PENDING:
                        PositionAxlePrice(m_Selection.ticket, IT_PENDING, price);
                        UpdateIndicators(m_Selection.ticket, (m_Selection.tp == 0 ? 0 : price + m_Selection.tp - m_Selection.pr), (m_Selection.sl == 0 ? 0 : price + m_Selection.sl - m_Selection.pr), m_Selection.vol, m_Selection.bIsBuy);
                        m_Selection.MousePrice = price;
                        break;
        }
        if (Mouse.IsVisible())
        {
                m_TradeLine.SpotLight(macroMountName(m_Selection.ticket, m_Selection.it, EV_LINE));
                Mouse.Hide();
        }
}
```

Why don't we put everything in the function above, since it is responsible for moving stop order points? The reason is that we need to do some calculations in order to correctly set the take profit or stop loss levels, and it is much easier to do this at a different point. We also need to change this value in another place, so here is the second place where the value is referenced. See the code below:

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{
        ulong   ticket;
        double  price;
        bool    bKeyBuy,

// ... Internal code ....

        switch (id)
        {
                case CHARTEVENT_MOUSE_MOVE:
                        Mouse.GetPositionDP(dt, price);
                        mKeys   = Mouse.GetButtonStatus();
                        bEClick  = (mKeys & 0x01) == 0x01;    //Left mouse button click
                        bKeyBuy  = (mKeys & 0x04) == 0x04;    //SHIFT pressed
                        bKeySell = (mKeys & 0x08) == 0x08;    //CTRL pressed
                        if (bKeyBuy != bKeySell)
                        {
                                if (!bMounting)
                                {
                                        m_Selection.bIsDayTrade = Chart.GetBaseFinance(m_Selection.vol, valueTp, valueSl);
                                        valueTp = Terminal.AdjustPrice(valueTp * Terminal.GetAdjustToTrade() / m_Selection.vol);
                                        valueSl = Terminal.AdjustPrice(valueSl * Terminal.GetAdjustToTrade() / m_Selection.vol);
                                        m_Selection.it = IT_PENDING;
                                        m_Selection.pr = price;
                                }
                                m_Selection.tp = (valueTp == 0 ? 0 : m_Selection.pr + (bKeyBuy ? valueTp : (-valueTp)));
                                m_Selection.sl = (valueSl == 0 ? 0 : m_Selection.pr + (bKeyBuy ? (-valueSl) : valueSl));
                                m_Selection.bIsBuy = bKeyBuy;
                                m_BtnInfoType.SetStateButton(macroMountName(def_IndicatorTicket0, IT_PENDING, EV_TYPE), bKeyBuy);
                                if (!bMounting)
                                {
                                        IndicatorAdd(m_Selection.ticket = def_IndicatorTicket0);
                                        bMounting = true;
                                }
                                MoveSelection(price);
                                if ((bEClick) && (memLocal == 0)) SetPriceSelection(memLocal = price);
                        }else if (bMounting)
                        {
                                RemoveIndicator(def_IndicatorTicket0);
                                memLocal = 0;
                                bMounting = false;
                        }else if ((!bMounting) && (bKeyBuy == bKeySell) && (m_Selection.ticket > def_IndicatorGhost))
                        {
                                if (bEClick) SetPriceSelection(m_Selection.MousePrice); else MoveSelection(price);
                        }
                        break;

// ... Rest of the code...
```

Thanks to this, we have predictable behavior in the system. There is another point where the value is referenced, but due to the complexity I decided to modify the whole thing so that the macro will no longer exist. Now it will be a function. So, the new Update function is shown below:

```
void UpdateIndicators(ulong ticket, double tp, double sl, double vol, bool isBuy)
{
        double pr;
        bool b0, bPen = true;

        if (ticket == def_IndicatorGhost) pr = m_Selection.pr; else
        {
                bPen = (pr = macroGetPrice(ticket, IT_RESULT, EV_LINE)) == 0;
                if (bPen && (macroGetPrice(ticket, IT_PENDING, EV_MOVE) == 0))
                {
                        CreateIndicator(ticket, IT_PENDING);
                        PositionAxlePrice(ticket, IT_PENDING, m_Selection.pr);
                        ChartRedraw();
                }
                pr = (pr > 0 ? pr : macroGetPrice(ticket, IT_PENDING, EV_MOVE));
                SetTextValue(ticket, IT_PENDING, vol);
        }
        b0 = UpdateIndicatorsLimits(ticket, IT_TAKE, tp, vol, pr, isBuy, bPen);
        b0 = (UpdateIndicatorsLimits(ticket, IT_STOP, sl, vol, pr, isBuy, bPen) ? true : b0);
        if (b0) ChartRedraw();
}
```

The highlighted points replace the previous macro, but since I said that the required code was much more complicated, let's see where the third and last point is actually referred. In the code above, there is no difference between take profit and stop loss indicators: they both are handled the same. Take a look at the code below.

```
inline bool UpdateIndicatorsLimits(ulong ticket, eIndicatorTrade it, double price, double vol, double pr, bool isBuy, bool isPen)
{
        bool b0 = false;
        double d1 = Terminal.GetPointPerTick();

        if (
    price  == 0)
        {
                if (macroGetPrice(ticket, it, EV_LINE) > 0) RemoveIndicator(ticket, it);
                if (macroGetPrice(ticket, it, EV_MOVE) == 0) CreateBtnMoveIndicator(ticket, it);
        } else if (b0 = (macroGetPrice(ticket, it, EV_LINE) == 0 ? true : b0)) CreateIndicator(ticket, it);
        switch (it)
        {
                case IT_TAKE:
                        price = (price == 0 ? 0 : (((isBuy ? price - pr : pr - price) > 0) ? price : (isBuy ? pr + d1 : pr - d1)));
                        break;
                case IT_STOP:
                        price = (price == 0 ? 0 : (isPen ? (((isBuy ? price - pr : pr - price) < 0) ? price : (isBuy ? pr - d1 : pr + d1)) : price));
                        break;
        }
        if (m_Selection.it == it) m_Selection.MousePrice = price;
        PositionAxlePrice(ticket, it, (price == 0 ? pr : price));
        SetTextValue(ticket, it, vol, (isBuy ? price - pr : pr - price));

        return b0;
}
```

From now on the take profit value of a pending order cannot be wrong as we have the limit for allowed values. It is no longer possible to place a pending buy order and then to move the take profit to a negative value (i.e. below the entry point), because the calculation of the take profit indicator prevents this. The advantage of writing the code this way is that, no matter if it is an order or a position, the take value can never be negative, because the EA itself will not allow it.

Now, as far as the stop loss is concerned, there is a slightly different calculation in which we check what we are managing - an order or a position. If it is an order, then the stop value will never be positive, and if it is a position, then the EA will simply ignore any other conditions. In this case, the EA must accept the value specified by the trader. So now we can have a positive stop loss, but only in the case of positions, without any harm to the rest of the order system code, and in this way the EA will finally interact with the trade server without the submitted data being rejected.

### Conclusion

Finally, after several articles, we have reached a climax, and now we have an order system that is almost complete and quite adaptable to various situations and market conditions. From now on, we will be able to trade with a fully graphical system using the mouse and keyboard, alongside your market analysis, to enter or exit trades.

For those who have just arrived and want to see how the system behaves or what it looks like at the current stage of development, please watch the video below. And thanks to everyone who has been following this series of articles so far. But the work is not over yet, and we have a lot of work to do until this Expert Advisor becomes something memorable. See you in the next article! 👍

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10635](https://www.mql5.com/pt/articles/10635)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10635.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Em\_diret4o\_ao\_Futuro\_q\_III\_k.zip](https://www.mql5.com/en/articles/download/10635/ea_-_em_diret4o_ao_futuro_q_iii_k.zip "Download EA_-_Em_diret4o_ao_Futuro_q_III_k.zip")(12036.12 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/435553)**
(1)


![Geraldo.Fabricio da Anunciação](https://c.mql5.com/avatar/avatar_na2.png)

**[Geraldo.Fabricio da Anunciação](https://www.mql5.com/en/users/geraldofabricio475)**
\|
22 Aug 2022 at 02:07

**MetaQuotes:**

New article [Developing a trading EA from scratch (Part 28): Towards the future (III)](https://www.mql5.com/en/articles/10635) has been published:

Author: [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831 "DJ_TLoG_831")

![Neural networks made easy (Part 25): Practicing Transfer Learning](https://c.mql5.com/2/48/Neural_networks_made_easy_025.png)[Neural networks made easy (Part 25): Practicing Transfer Learning](https://www.mql5.com/en/articles/11330)

In the last two articles, we developed a tool for creating and editing neural network models. Now it is time to evaluate the potential use of Transfer Learning technology using practical examples.

![DoEasy. Controls (Part 17): Cropping invisible object parts, auxiliary arrow buttons WinForms objects](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__5.png)[DoEasy. Controls (Part 17): Cropping invisible object parts, auxiliary arrow buttons WinForms objects](https://www.mql5.com/en/articles/11408)

In this article, I will create the functionality for hiding object sections located beyond their containers. Besides, I will create auxiliary arrow button objects to be used as part of other WinForms objects.

![DoEasy. Controls (Part 18): Functionality for scrolling tabs in TabControl](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__6.png)[DoEasy. Controls (Part 18): Functionality for scrolling tabs in TabControl](https://www.mql5.com/en/articles/11454)

In this article, I will place header scrolling control buttons in TabControl WinForms object in case the header bar does not fit the size of the control. Besides, I will implement the shift of the header bar when clicking on the cropped tab header.

![Neural networks made easy (Part 24): Improving the tool for Transfer Learning](https://c.mql5.com/2/48/Neural_networks_made_easy_024.png)[Neural networks made easy (Part 24): Improving the tool for Transfer Learning](https://www.mql5.com/en/articles/11306)

In the previous article, we created a tool for creating and editing the architecture of neural networks. Today we will continue working on this tool. We will try to make it more user friendly. This may see, top be a step away form our topic. But don't you think that a well organized workspace plays an important role in achieving the result.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/10635&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051666995474191439)

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