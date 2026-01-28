---
title: Developing a trading Expert Advisor from scratch (Part 24): Providing system robustness (I)
url: https://www.mql5.com/en/articles/10593
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:46:37.081455
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/10593&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051675898941396090)

MetaTrader 5 / Trading


### Introduction

Some things are not so simple, although some people may think so. The order system is one of those things. You can even create a more modest system that serves you perfectly well, as we did in the article [Developing a trading Expert Advisor from scratch](https://www.mql5.com/en/articles/10085), in which we created a basic system that can be useful for many people and not enough for others. Therefore, the moment came when everything began to change — this is when the first part of this series about the new order system was born. This can be seen in the article [Developing a trading Expert Advisor from scratch (Part 18)](https://www.mql5.com/en/articles/10462). This is where we started developing a system that can be managed by the EA while being supported by MetaTrader 5. The idea of the system was to have no limit on orders on the chart. At first, the system seemed rather bold, and I must admit that the very fact of creating a system in which objects would be maintained not by the EA but by MetaTrader 5 seemed rather pointless and inefficient to me.

However, the system was under development and in the article [Developing a trading Expert Advisor from scratch (Part 23)](https://www.mql5.com/en/articles/10563) we developed a ghost system to facilitate the management of orders, positions or stop levels (Take Profit and Stop Loss). It was very interesting to develop, but there was a problem. If you look at the number of objects used and visible compared to the number of objects supported by MetaTrader 5, you will definitely be surprised, because the number of objects supported will always be higher.

In many cases, the problem is not so serious, you can even live with some moments. But there are two problems that made the system not very stable during periods of high market volatility. In some situations, they forced the user to act incorrectly. This is because when the trader adds a pending order, the system sends it to the server and the server sometimes needs more time than usual to respond. And the system indicates at some moments that there is an order and at other moments it shows that there is no order. And when it was done in positions _(see the documentation for the difference between orders and positions)_this turned out to be even more cumbersome because it was unknown whether the server executed the command as expected.

There are several ways to solve this problem. Some of them are simpler, some are more complex. Anyway, we must trust the EA, otherwise we should not use it under any circumstances.

### 1.0. Planning

The big problem here is to design a system that possesses two qualities: **speed** and **reliability**. In some kinds of systems, it is quite difficult or even impossible to achieve both. So, in many cases, we try to balance things out. But since it's about money, OUR money, we don't want to risk it by acquiring a system that doesn't have these qualities. It must be remembered that we are dealing with a system that works in REAL TIME, and this is the most difficult scenario that a developer can get into, since we should always try to have a system that is extremely fast: it must react instantly to events, while showing enough reliability not to collapse when we try to improve it. Thus, it is clear that the task is quite difficult.

Speed can be achieved by ensuring that functions are called and executed in the most appropriate way, avoiding unnecessary calls at even more unnecessary times. This will provide the system as fast as possible within the language. However, if we want something even faster, then we have to go down to the machine language level, in which case, we mean Assembly. But this is often unnecessary, we can use the C language and get equally good results.

One of the ways to achieve the desired robustness is to try to re-use the code as much as possible so that it is constantly tested in different cases. But this is only one way. Another way is to use the OOP (Object Oriented Programming). If this is done correctly and properly so that each object class does not manipulate object class data directly, except in the case of inheritance, then it will be enough to have a very robust system. This sometimes reduces the execution speed, but this reduction is so small that it can be ignored due to the exponential increment generated by the encapsulation provided by the class. This encapsulation gives the robustness we need.

As you can see, it is not that simple to achieve both speed and robustness. But the great thing is that we don't have to sacrifice things so much, as you might think at first glance. We can simply check the system documentation and see what can be changed in order to improve things. The simple fact that we are not trying to reinvent the wheel is already a good start. But remember that programs and systems are constantly improving. So, we should always try to use the available things as much as possible and only then, in the last case, to really reinvent the wheel.

Before some find it unnecessary to present the changes that have been made in this article or think that I'm changing the code a lot without actually moving it, let me explain: When we code something we really have no idea how the final code will work. All we have are the goals to be achieved. Once this goal has been achieved, we start looking at how we achieved this goal and trying to improve things in order to make them better.

In the case of a commercial system, be it an executable nor a library, we make the changes and release it as an update. The user does not really need to know the paths involved to reach the objective, since it is a commercial system. It is it's good that he doesn't actually know. But since it's an open system, I don't want to make you think that you can develop an extremely efficient system right away, so right from the start. Thinking this way is not adequate, it is even an insult, since no matter how much a programmer or developer has knowledge of the language to be used, there will always be things that can be improved over time.

So, don't take this sequence as something that could be summarized in 3 or 4 articles, because if that were the case, it would be better to simply create the code, staying in the way I thought was most appropriate and release it commercially. This is not my intention. I learned to program by watching the code of other more experienced programmers, and I know the value that this has. It is much more important to know how the thing develops over time than simply taking the finished solution and trying to understand how it works.

After these observations, let us move on to development.

### 2.0. Implementation

**2.0.1. New modeling of position indicators**

The first thing to note in the new code format is the change of a function that has become a macro.

```
inline string MountName(ulong ticket, eIndicatorTrade it, eEventType ev, bool isGhost = false)
{
        return StringFormat("%s%c%c%c%llu%c%c%c%s", def_NameObjectsTrade, def_SeparatorInfo, (char)it, def_SeparatorInfo, ticket, def_SeparatorInfo, (char)(isGhost ? ev + 32 : ev), def_SeparatorInfo, (isGhost ? def_IndicatorGhost : def_IndicatorReal));
}
```

Even if the compiler does use this code at every point where it is referenced thanks to the reserved word 'inline', you shouldn't take it for granted, because this function is called many times in the code. We need to make sure it actually runs as fast as possible, so our new code will look like this:

```
#define macroMountName(ticket, it, ev, Ghost) 								 \
		StringFormat("%s%c%llu%c%c%c%c%c%c%c", def_NameObjectsTrade, def_SeparatorInfo,          \
                                                       ticket, def_SeparatorInfo,                        \
                                                       (char)it, def_SeparatorInfo,                      \
                                                       (char)(Ghost ? ev + 32 : ev), def_SeparatorInfo,  \
                                                       (Ghost ? def_IndicatorGhost : def_IndicatorReal))
```

Pay attention that the data in the old version of the macro and the data in this version are different. There is a reason for this change, which we will discuss later in this article.

But because of this modification, we also have to make a small change to the code of another function.

```
inline bool GetIndicatorInfos(const string sparam, ulong &ticket, eIndicatorTrade &it, eEventType &ev)
                        {
                                string szRet[];
                                char szInfo[];

                                if (StringSplit(sparam, def_SeparatorInfo, szRet) < 2) return false;
                                if (szRet[0] != def_NameObjectsTrade) return false;
                                ticket = (ulong) StringToInteger(szRet[1]);
                                StringToCharArray(szRet[2], szInfo);
                                it = (eIndicatorTrade)szInfo[0];
                                StringToCharArray(szRet[3], szInfo);
                                ev = (eEventType)szInfo[0];

                                return true;
                        }
```

The change here was only in the index, which will be used to indicate what is a ticket and what is an indicator. Nothing complicated. Just one simple detail that needs to be done, otherwise we will have inconsistent data when using this function.

You may wonder: "Why do we need these changes? Didn't the system work perfectly?". Yes, it did. But there are things which we cannot control. For example, when the MetaTrader 5 developer improves some functions which are not used in the EA and thus cannot be beneficial for us. **The rule is to avoid reinventing the wheel and to use the available resources instead**. Therefore, we should always try to use the functions provided by the languages, which in our case is MQL5, and avoid creation of our own functions. This may seem absurd, but in fact if you stop and think, you will see that from time to time the platform provides improvements in some functions, and if you are using these same functions you will have better performance and increased security in your programs without having to make any extra effort.

Thus, the end justifies the means. However, will the changes made above help the EA benefit from any improvements in the MQL5 library? The answer to this question is _**NO**._ The above changes are necessary to ensure that object name modeling is correct so that we can effectively use possible future improvements coming from the MQL5 and MetaTrader 5 developers. Below is one of the items that may be useful:

```
inline void RemoveIndicator(ulong ticket, eIndicatorTrade it = IT_NULL)
{
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
        if ((it == IT_NULL) || (it == IT_PENDING) || (it == IT_RESULT))
                ObjectsDeleteAll(Terminal.Get_ID(), StringFormat("%s%c%llu%c", def_NameObjectsTrade, def_SeparatorInfo, ticket, (ticket > 1 ? '*' : def_SeparatorInfo)));
        else ObjectsDeleteAll(Terminal.Get_ID(), StringFormat("%s%c%llu%c%c", def_NameObjectsTrade, def_SeparatorInfo, ticket, def_SeparatorInfo, (char)it));
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
        m_InfoSelection.bIsMovingSelect = false;
        ChartRedraw();
}
```

The previous version of the same code is shown below for those who don't remember it or haven't met it before. The code looks like this:

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

It may seem that the code just became more compact. But it's not only that. Code reduction is an obvious thing, but the truth is much deeper. The old code was replaced by a new one which better uses platform resources. But since the previously used model of object names did not allow for this improvement, we change the modeling so that we can expect to benefit from the MQL5 functions. If this function is ever improved for any reason, the EA will benefit from this modification without the need for us to make any changes to the EA structure. I am talking about the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/objectdeleteall) function. If we use it correctly, MetaTrader 5 will do the cleanup. We don't need to specify too many details, we just specify the name of the object or objects and let MetaTrader 5 do the rest. The points where this function is used are highlighted in the new code. Notice how we did the modeling to inform about the prefix that will be used. This was not possible before the modification of object name modeling.

I would like to draw your attention to one detail in the new code fragment, which is highlighted below.

```
if ((it == IT_NULL) || (it == IT_PENDING) || (it == IT_RESULT))
        ObjectsDeleteAll(Terminal.Get_ID(), StringFormat("%s%c%llu%c", def_NameObjectsTrade, def_SeparatorInfo, ticket, (ticket > 1 ? '*' : def_SeparatorInfo)));
```

Why do you think I added the highlighted part?

This is because if the system creates a ticket starting with a value equal to 1, then as soon as the pending order is placed, all objects will be removed from the screen. Isn't it clear? The input used to place a pending order has a value of 1, i.e. indicator 0 actually has a value of 1, not 0, since 0 is used to perform other tests in the EA. Because of this the initial value is 1. Now we have a problem: suppose that the trading system creates a ticket **1221766803**. Then the object that represents this ticket will have the following value as prefix: **SMD\_OT#** **1221766803**. When the EA executes the ObjectsDeleteAll function to delete indicator 0, the object name will be **SMD\_OT#1** and this will delete all objects starting with this value, including the newly created system. To solve this problem, we'll make a slight adjustment to the name to inform the ObjectsDeleteAll functions by adding an extra character at the end of the name so that the function knows if we're deleting indicator 0 or another one.

Thus, if indicator 0 is to be deleted, the function receives the value **SMD\_OT#1#**. This will avoid the problem. At the same time, in the case of the above example, the function will get the name **SMD\_OT#1221766803\***. It seems to be something simple, but because of this you can be puzzled why the EA keeps deleting indicator objects of a newly placed order.

Now let's talk about one curious detail. At the end of the function there is a call of [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/chartredraw). What is it used here? Doesn't MetaTrader 5 refresh the chart itself? It does. But we don't know exactly when it will happen. There is another problem: all calls to place or delete objects on the chart are synchronous, i.e. they are executed at a certain time, which is not necessarily the time that we expect. However, our order system will use objects to either display or manage orders, and we need to be sure that the object is on the chart. We can't afford to think that MetaTrader 5 has already placed or removed objects from the chart, because we need to be sure of it, which is why we force the platform to make this refresh.

Thus, when we call ChartRedraw, we force the platform to refresh the list of objects on the chart, so we can be sure that a certain object is present or not present on the chart. If this is still not clear, let us move on to the next topic.

### 2.0.2. Less objects — higher speed

The initialization function in the previous version was cumbersome. It had a lot of repetitive checks and some things were duplicated. In addition to some minor issues, the system reused very little of the already existing capacity. Therefore, to take advantage of the new modeling, I decided to reduce the number of objects which are created during initialization. SO, now the system looks like this:

```
void Initilize(void)
{
        ChartSetInteger(Terminal.Get_ID(), CHART_SHOW_OBJECT_DESCR, false);
        ChartSetInteger(Terminal.Get_ID(), CHART_SHOW_TRADE_LEVELS, false);
        ChartSetInteger(Terminal.Get_ID(), CHART_DRAG_TRADE_LEVELS, false);
        for (int c0 = OrdersTotal(); c0 >= 0; c0--) IndicatorInfosAdd(OrderGetTicket(c0));
        for (int c0 = PositionsTotal(); c0 >= 0; c0--) IndicatorInfosAdd(PositionGetTicket(c0));
}
```

It seems that everything was different, and in fact it was. Now we are reusing the function that was not used enough — this is the function that adds indicators to the chart. Let's take a look at this special feature.

```
inline void IndicatorAdd(ulong ticket)
{
        char ret;

        if (ticket == def_IndicatorTicket0) ret = -1; else
        {
                if (ObjectGetDouble(Terminal.Get_ID(), macroMountName(ticket, IT_PENDING, EV_LINE, false), OBJPROP_PRICE) != 0) return;
                if (ObjectGetDouble(Terminal.Get_ID(), macroMountName(ticket, IT_RESULT, EV_LINE, false), OBJPROP_PRICE) != 0) return;
                if ((ret = GetInfosTradeServer(ticket)) == 0) return;
        }
        switch (ret)
        {
                case  1:
                        CreateIndicatorTrade(ticket, IT_RESULT);
                        PositionAxlePrice(ticket, IT_RESULT, m_InfoSelection.pr);
                        break;
                case -1:
                        CreateIndicatorTrade(ticket, IT_PENDING);
                        PositionAxlePrice(ticket, IT_PENDING, m_InfoSelection.pr);
                        break;
        }
        ChartRedraw();
        UpdateIndicators(ticket, m_InfoSelection.tp, m_InfoSelection.sl, m_InfoSelection.vol, m_InfoSelection.bIsBuy);
}
```

Look closely at the above code. It may seem that the code contains unnecessary checks. But they exist for a very simple reason. This function is the only way to actually create a pending order or position indicator. The two highlighted lines will check whether the indicator exists. To do this, it is checked whether any value is stored in the object that is used as a line. Here, it is the value of the price at which the object is located. This value must be non-zero if the indicating object is on the chart. In all other cases, it will be equal to zero, either because the object does not exist, or for any other reason, which does not matter. Is it now clear why we have to force the chart refresh? If this were not done, the EA would add objects unnecessarily, so we cannot wait for the platform to take this action at some unknown time. We must be sure that the chart has been updated. Otherwise, when these checks are done, they will report things that don't actually match the current state of objects, making the system less reliable.

Although it seems that these checks slow down the EA speed, this is a conceptual error. When we do such checks and do not try to force the platform to create an object that may already be in the creation queue, we tell the platform "UPDATE NOW". Then, when we need it, we check to see if the object has already been created, and in case it has already been created, we use it as needed. This is called "programming the right way". Since this way we make the platform work less and avoid unnecessary checks of whether the object is created or not, we make the EA more reliable, because we know that we have data that we want to work with.

Since the checks will show there is no object matching the specified ticket, the object will be created. Pay attention that there is another check at the beginning of whether we are creating indicator 0 or any other one. This ensures that we do not have unnecessary objects supported by MetaTrader 5; we have only those objects that we actually use on the chart. If we create indicator 0, then no further testing is required, since we will create it in very special and specific conditions. The object 0 is used to position orders using SHIFT or CTRL + the mouse. Don't worry, we'll see how it works soon.

There is one important detail in the above code: why are we updating the chart before calling the Update function? It's pointless. To understand this, let's look at the UpdateIndicators function below.

```
void UpdateIndicators(ulong ticket, double tp, double sl, double vol, bool isBuy)
{
        double pr;
        bool b0 = false;

        pr = macroGetLinePrice(ticket, IT_RESULT);
        pr = (pr > 0 ? pr : macroGetLinePrice(ticket, IT_PENDING));
        SetTextValue(ticket, IT_PENDING, vol);
        if (tp > 0)
        {
                if (b0 = (ObjectGetDouble(Terminal.Get_ID(), macroMountName(ticket, IT_TAKE, EV_LINE, false), OBJPROP_PRICE) == 0 ? true : b0))
                        CreateIndicatorTrade(ticket, IT_TAKE);
                PositionAxlePrice(ticket, IT_TAKE, tp);
                SetTextValue(ticket, IT_TAKE, vol, (isBuy ? tp - pr : pr - tp));
        }
        if (sl > 0)
        {
                if (b0 = (ObjectGetDouble(Terminal.Get_ID(), macroMountName(ticket, IT_STOP, EV_LINE, false), OBJPROP_PRICE) == 0 ? true : b0))
                        CreateIndicatorTrade(ticket, IT_STOP);
                PositionAxlePrice(ticket, IT_STOP, sl);
                SetTextValue(ticket, IT_STOP, vol, (isBuy ? sl - pr : pr - sl));
        }
        if (b0) ChartRedraw();
}
```

This function will basically take care of the indicators pointing to the limits. Now take a look at the two highlighted lines: if the chart is not updated, these lines will not trigger, returning a value of 0, and if it does, then the rest of the code will not work, and the limit indicators will not be displayed correctly on the screen.

But before creating the limit indicators, we must conduct some checks to understand whether they really need to be created or they just need to be adjusted. This is done in the same way as when creating the basic object. And even here, when creating the objects, we will also force the chart to be updated so that the chart is always up-to-date.

You may wonder: "Why are there so many forced updates, are they really necessary?" And the answer to this is BIG and SOUND **YES**... and the reason for this is the function below:

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

You might think that there is nothing special about this feature. Are you sure? WRONG! This function contains a key point: we must make sure that the object is on the chart, otherwise all the code to create it will be called several times, creating a large queue to be managed by MetaTrader 5, and some data may be lost or become obsolete. All this will make the system unstable, less secure and therefore unreliable. The call of the function that creates the object is highlighted. If we did not force MetaTrader 5 to update the chart at strategic moments, then we could have problems, since the above function is called by the OnTick event, and during periods of high volatility, the number of calls coming from OnTick is quite large, which can give rise to an excess of objects in the queue, which is not good at all. So, the data is forced to be refreshed via the ChartRedraw call and validated via ObjectGetDouble, thereby reducing the chance that there will be too many objects in the queue.

Even without looking at how the system works, you might think: "It’s good that now, in case of an accidental deletion of the TradeLine object, the EA will notice this, and if the check through ObjectGetDouble fails and the indicator fails, the indicator will be recreated." This is the idea. But it is not recommended for the user to delete objects that are present in the list of objects window without really knowing what the object is, because if you delete any object (except for TradeLine), the EA may not notice there is no indicator, leaving without means of access to it, since it simply has no other way of access other than through the buttons present on it.

The script above would be a real nightmare if it weren't for the function that comes right after it and is responsible for maintaining the entire message flow within the class. However, it is still not the only entry point. I'm talking of the DispatchMessage function, let's take a look at it.

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{
        ulong   ticket;
        double  price;
        bool    bKeyBuy,
                bKeySell,
                bEClick;
        datetime        dt;
        uint            mKeys;
        char            cRet;
        eIndicatorTrade it;
        eEventType      ev;

        static bool bMounting = false, bIsDT = false;
        static double valueTp = 0, valueSl = 0, memLocal = 0;

        switch (id)
        {
                case CHARTEVENT_MOUSE_MOVE:
                        Mouse.GetPositionDP(dt, price);
                        mKeys   = Mouse.GetButtonStatus();
                        bEClick  = (mKeys & 0x01) == 0x01;    //Left mouse click
                        bKeyBuy  = (mKeys & 0x04) == 0x04;    //SHIFT pressed
                        bKeySell = (mKeys & 0x08) == 0x08;    //CTRL pressed
                        if (bKeyBuy != bKeySell)
                        {
                                if (!bMounting)
                                {
                                        Mouse.Hide();
                                        bIsDT = Chart.GetBaseFinance(m_InfoSelection.vol, valueTp, valueSl);
                                        valueTp = Terminal.AdjustPrice(valueTp * Terminal.GetAdjustToTrade() / m_InfoSelection.vol);
                                        valueSl = Terminal.AdjustPrice(valueSl * Terminal.GetAdjustToTrade() / m_InfoSelection.vol);
                                        m_InfoSelection.it = IT_PENDING;
                                        m_InfoSelection.pr = price;
                                }
                                m_InfoSelection.tp = m_InfoSelection.pr + (bKeyBuy ? valueTp : (-valueTp));
                                m_InfoSelection.sl = m_InfoSelection.pr + (bKeyBuy ? (-valueSl) : valueSl);
                                m_InfoSelection.bIsBuy = bKeyBuy;
                                if (!bMounting)
                                {
                                        IndicatorAdd(m_InfoSelection.ticket = def_IndicatorTicket0);
                                        m_TradeLine.SpotLight(macroMountName(def_IndicatorTicket0, IT_PENDING, EV_LINE, false));
                                        m_InfoSelection.bIsMovingSelect = bMounting = true;
                                }
                                MoveSelection(price);
                                if ((bEClick) && (memLocal == 0))
                                {
                                        RemoveIndicator(def_IndicatorTicket0);
                                        CreateOrderPendent(m_InfoSelection.vol, bKeyBuy, memLocal = price,  price + m_InfoSelection.tp - m_InfoSelection.pr, price + m_InfoSelection.sl - m_InfoSelection.pr, bIsDT);
                                }
                        }else if (bMounting)
                        {
                                RemoveIndicator(def_IndicatorTicket0);
                                Mouse.Show();
                                memLocal = 0;
                                bMounting = false;
                        }else if ((!bMounting) && (bKeyBuy == bKeySell))
                        {
                                if (bEClick) SetPriceSelection(price); else MoveSelection(price);
                        }
                        break;
                case CHARTEVENT_OBJECT_DELETE:
                        if (GetIndicatorInfos(sparam, ticket, it, ev))
                        {
                                if (GetInfosTradeServer(ticket) == 0) break;
                                CreateIndicatorTrade(ticket, it);
                                if ((it == IT_PENDING) || (it == IT_RESULT))
                                        PositionAxlePrice(ticket, it, m_InfoSelection.pr);
                                ChartRedraw();
				m_TradeLine.SpotLight();
                                m_InfoSelection.bIsMovingSelect = false;
                                UpdateIndicators(ticket, m_InfoSelection.tp, m_InfoSelection.sl, m_InfoSelection.vol, m_InfoSelection.bIsBuy);
                        }
                        break;
                case CHARTEVENT_CHART_CHANGE:
                        ReDrawAllsIndicator();
                        break;
                case CHARTEVENT_OBJECT_CLICK:
                        if (GetIndicatorInfos(sparam, ticket, it, ev)) switch (ev)
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
							m_InfoSelection.ticket = ticket;
							m_InfoSelection.it = it;
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
                                                m_TradeLine.SpotLight(macroMountName(ticket, it, EV_LINE, false));
                                        }
                                        break;
                        }
                        break;
        }
}
```

This function has gone through so many changes that I'll have to break it down into small parts to explain what's going on inside it. If you already have programming experience, then it won't be difficult for you to understand what it does. However, if you are just an enthusiast or a novice MQL5 programmer, then understanding this function can be a bit difficult, so I will calmly explain it in the next topic.

### 2.0.3. Breaking down the DispatchMessage function

This topic explains what happens in the DispatchMessage function. If you understand how it works by simply looking at the code, then this topic will not give anything new to you.

The first thing we have after local variables is static variables.

```
static bool bMounting = false, bIsDT = false;
static double valueTp = 0, valueSl = 0, memLocal = 0;
```

They could be declared as private variables in the class, but since they will only be used at this point in the code, it makes no sense for other functions in the class to see these variables. They should be declared as static, because they must remember their values when the function is called again. If we do not add the 'static' keyword, they will lose their value as soon as the function ends. Once this is done, we will start processing the events that MetaTrader 5 indicates to the EA.

The first event can be seen below:

```
case CHARTEVENT_MOUSE_MOVE:
        Mouse.GetPositionDP(dt, price);
        mKeys   = Mouse.GetButtonStatus();
        bEClick  = (mKeys & 0x01) == 0x01;    //Left mouse click
        bKeyBuy  = (mKeys & 0x04) == 0x04;    //SHIFT pressed
        bKeySell = (mKeys & 0x08) == 0x08;    //CTRL pressed
```

Here we collect and isolate data from the mouse and some keys (from the keyboard) associated with the mouse. Once we've done that, comes a long code that starts with a test.

```
if (bKeyBuy != bKeySell)
```

If you press the SHIFT or CTRL key, but not both at the same time, this will make the EA understand that you want to place an order at a certain price. If so, check further.

```
if (!bMounting)
{
        Mouse.Hide();
        bIsDT = Chart.GetBaseFinance(m_InfoSelection.vol, valueTp, valueSl);
        valueTp = Terminal.AdjustPrice(valueTp * Terminal.GetAdjustToTrade() / m_InfoSelection.vol);
        valueSl = Terminal.AdjustPrice(valueSl * Terminal.GetAdjustToTrade() / m_InfoSelection.vol);
        m_InfoSelection.it = IT_PENDING;
        m_InfoSelection.pr = price;
}
```

In case indicator 0 has not yet been set, this test will pass. The mouse will be hidden, then the values in in the Chart Trade will be captured. These values are then converted to points based on the levering level that the trader indicates through Chart Trade. The initial value where the order will be placed will be shown. This sequence should only occur once per cycle of use.

The next step is to create the Take Profit and Stop Loss levels and indicate whether we will buy or sell.

```
m_InfoSelection.tp = m_InfoSelection.pr + (bKeyBuy ? valueTp : (-valueTp));
m_InfoSelection.sl = m_InfoSelection.pr + (bKeyBuy ? (-valueSl) : valueSl);
m_InfoSelection.bIsBuy = bKeyBuy;
```

They are created outside of the cycle because when we move the mouse to a different price range, we will also have to move the Take Profit and Stop Loss. But why is this code above not inside the assembly test? The reason is that if you change, release the SHIFT key and press the CTRL key, or vice versa, without moving the mouse, while there are indicators on the screen, the values of the Take Profit and Stop Loss indicators will be exchanged. To avoid this, the fragment has to stay out of the test. But this forces us to do a new assembly test, which is seen below:

```
if (!bMounting)
{
        IndicatorAdd(m_InfoSelection.ticket = def_IndicatorTicket0);
        m_TradeLine.SpotLight(macroMountName(def_IndicatorTicket0, IT_PENDING, EV_LINE, false));
        m_InfoSelection.bIsMovingSelect = bMounting = true;
}
```

Why do we have two tests? Can we have only one? This would be ideal, but the function highlighted in the above code does not allow us to do this. We need to look at [IndicatorAdd](https://www.mql5.com/en/articles/10593#indicator_add) to understand this fact. After creating indication 0, we set it as selected and show that it is already running and built. Therefore, you can move it with the next line.

```
MoveSelection(price);
```

However, even within the same criteria of pressing SHIFT or CTRL to place a pending order, we have a final step.

```
if ((bEClick) && (memLocal == 0))
{
        RemoveIndicator(def_IndicatorTicket0);
        CreateOrderPendent(m_InfoSelection.vol, bKeyBuy, memLocal = price,  price + m_InfoSelection.tp - m_InfoSelection.pr, price + m_InfoSelection.sl - m_InfoSelection.pr, bIsDT);
}
```

This will add a pending order exactly to the point we are targeting. Two conditions must be met. The first is the left mouse button click and the second is that we didn't do it at the same price in one go. That is, to place two or more orders at the same price, we must place this new order with a different call, because this will not happen in the same call.

Simultaneously with the removal of indicator 0 from the chart, an order with properly filled parameters is sent to the trade server.

Now let's move on to the next step...

```
if (bKeyBuy != bKeySell)
{

// ... code described so far ....

}else if (bMounting)
{
        RemoveIndicator(def_IndicatorTicket0);
        Mouse.Show();
        memLocal = 0;
        bMounting = false;
}
```

If indicator 0 was set but the condition was not met because only SHIFT or CTRL was pressed, then the highlighted code executes to remove indicator 0 from the list of objects, simultaneously resetting the mouse and leaving the static variables in their initial state. In other words, the system will be clean.

The next and final step inside the mouse event handling is shown below:

```
if (bKeyBuy != bKeySell)
{

// ... previously described code ...

}else if (bMounting)
{

// ... previously described code ...

}else if ((!bMounting) && (bKeyBuy == bKeySell))
{
        if (bEClick) SetPriceSelection(price); else MoveSelection(price);
}
```

The highlighted code is the last mouse step in message processing. In case we have neither set indicator 0 nor SHIFT or CTRL keys in a different state meaning that they can be pressed or released at the same time, we have the following behavior: if we left click then the price will be sent to the indicator, and if we only move the mouse, the price will be used to move the indicator. But then we have a question: which indicator? Don't worry, we'll soon see which indicator it is, but in case you're wondering, indicator 0 doesn't use this selection. If you don't understand, go back to the beginning of this section and read how this message processing works.

Below is the next message:

```
case CHARTEVENT_OBJECT_DELETE:
        if (GetIndicatorInfos(sparam, ticket, it, ev))
        {
                if (GetInfosTradeServer(ticket) == 0) break;
                CreateIndicatorTrade(ticket, it);
                if ((it == IT_PENDING) || (it == IT_RESULT))
                        PositionAxlePrice(ticket, it, m_InfoSelection.pr);
                ChartRedraw();
		m_TradeLine.SpotLight();
                m_InfoSelection.bIsMovingSelect = false;
                UpdateIndicators(ticket, m_InfoSelection.tp, m_InfoSelection.sl, m_InfoSelection.vol, m_InfoSelection.bIsBuy);
        }
        break;
```

Remember, I said above that the EA has a small security system to prevent incorrect removal of indicators? This system is contained in the code for processing messages about events sent by MetaTrader 5 when an object is deleted.

When this happens, MetaTrader 5 reports, using the sparam parameter, the name of the deleted object against which it is checked whether it was an indicator, and if so, which one. It doesn't matter which object was affected. What we want to know is which indicator was affected, after that we will check if there is any order or position associated with the indicator and if so we will create the whole indicator again. In an extreme case, if the affected indicator was the base indicator, we reposition it immediately and force MetaTrader 5 to place the indicator on the chart immediately, regardless of what the indicator is. We remove the selection indication and place an order for update on the indicator threshold data.

The next event to handle is very simple, it just makes a request to resize all indicators on the screen, its code is shown below.

```
case CHARTEVENT_CHART_CHANGE:
        ReDrawAllsIndicator();
        break;
```

Here is the object click event.

```
case CHARTEVENT_OBJECT_CLICK:
        if (GetIndicatorInfos(sparam, ticket, it, ev)) switch (ev)
        {
//....
        }
        break;
```

It starts as shown above: MetaTrader 5 tells us which object was clicked so that the EA can check what type of event to handle. So far we have 2 events CLOSE and MOVE. Let's first consider the CLOSE event, which will close and define the end of the indicator on the screen.

```
case EV_CLOSE:
        if ((cRet = GetInfosTradeServer(ticket)) != 0) switch (it)
        {
                case IT_PENDING:
                case IT_RESULT:
                        if (cRet < 0) RemoveOrderPendent(ticket); else ClosePosition(ticket);
                        break;
                case IT_TAKE:
                case IT_STOP:
			m_InfoSelection.ticket = ticket;
			m_InfoSelection.it = it;
                        m_InfoSelection.bIsMovingSelect = true;
                        SetPriceSelection(0);
                        break;
        }
        break;
```

The close event will do the following: it will use the ticket to search the server for what should be closed and to check if there is anything to close, because it may happen that by this time the server has already done this but the EA does not yet know about it. Since we have something to close, let's do it correctly so we have the required checks and the right way to inform the class to close or remove an indicator from the chart.

So, we have come to the last step in this topic, which is shown below.

```
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
                m_TradeLine.SpotLight(macroMountName(ticket, it, EV_LINE, false));
        }
        break;
```

MOVE is an event that does exactly this — it selects the indicator to move. So, it only selects, but the movement itself is performed during a mouse movement event. Remember, at the beginning of the topic, I said that there is a condition under which we are not dealing with indicator 0, and even so, something will still move. This something is indicated at this point, in the move event. We check here if anything is selected to move. If it is so, the indicator that was selected will cease to be selected and will not receive the mouse movement events and the new indicator will not be selected. In this case the data from the new indicator to receive the mouse data will be stored in a structure and this indicator will receive a change that will indicate that it is selected. This change is seen in the line thickness.

### 2.0.4. A new Mouse Object class

In addition to the improvements we've covered above, we have others that deserve to be mentioned.

While most traders do not need a mouse-based indicator system implemented in an EA, others may need and want the system to work perfectly. But the trader may delete some of the objects that make up the mouse indicator by mistake, which will lead to its failure. Luckily, we can avoid this by using the EVENT system. Once an object deletion event is detected and sent to the EA, the class that the object belongs to can recreate the object again, giving the system stability. But it is good to keep the list of points as small as possible, create them as they are needed and then delete them when they are no longer needed. This is what we have been doing so far, but the Mouse class was missing.

Let's start by creating some definitions to replace the system of creating constant names.

```
#define def_MousePrefixName "MOUSE "
#define def_NameObjectLineH def_MousePrefixName + "H"
#define def_NameObjectLineV def_MousePrefixName + "TMPV"
#define def_NameObjectLineT def_MousePrefixName + "TMPT"
#define def_NameObjectBitMp def_MousePrefixName + "TMPB"
#define def_NameObjectText  def_MousePrefixName + "TMPI"
```

After that, the new initialization function looks like this:

```
void Init(color c1, color c2, color c3)
{
        m_Infos.cor01 = c1;
        m_Infos.cor02 = c2;
        m_Infos.cor03 = c3;
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_MOUSE_MOVE, true);
        ChartSetInteger(Terminal.Get_ID(), CHART_CROSSHAIR_TOOL, false);
        Show();
}
```

Please note that it is much simpler than the previous version. At this point, we have the call that will show the mouse system. The call is performed at the highlighted point in the previous code. It will call the code that will actually create an indication system on the price axis.

```
inline void Show(void)
{
        if (ObjectGetDouble(Terminal.Get_ID(), def_NameObjectLineH, OBJPROP_PRICE) == 0)
        {
                ObjectCreate(Terminal.Get_ID(), def_NameObjectLineH, OBJ_HLINE, 0, 0, 0);
                ObjectSetString(Terminal.Get_ID(), def_NameObjectLineH, OBJPROP_TOOLTIP, "\n");
                ObjectSetInteger(Terminal.Get_ID(), def_NameObjectLineH, OBJPROP_BACK, false);
        }
        ObjectSetInteger(Terminal.Get_ID(), def_NameObjectLineH, OBJPROP_COLOR, m_Infos.cor01);
}
```

This code is very interesting: it checks if the mouse pointer object exists in price or not. If the check is successful, then it means that there is a line on the chart or something related to the mouse, so all we do is adjust the color of the horizontal line. Why do we perform this check? To understand this, take a look at the function responsible for hiding, or rather removing the objects connected to the mouse. See the function below:

```
inline void Hide(void)
{
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
        ObjectsDeleteAll(Terminal.Get_ID(), def_MousePrefixName + "T");
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
        ObjectSetInteger(Terminal.Get_ID(), def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
}
```

This is an interesting style of operation. All objects connected to the mouse and having the specified name will be deleted from the MetaTrader 5 chart and thus the list of objects will always be small. However, the horizontal line will not be deleted, only its color will change. Therefore, the function showing the mouse performs a check before creating the object, because it is not actually excluded from the list of objects but it is only hidden. But all other objects are deleted from the list of objects. But then how are we going to use these other objects during studies? Since the studies are short moments where we simply want to find out some details, there is no point in keeping the objects in the list only to us them 1-2 times. It's better to create them, do the study, and then remove them from the list, so we get a more reliable system.

This may seem silly, but the order system we show is based on the use of objects, and the more objects in the list, the more work MetaTrader 5 will have to do to search the list when we want to access a certain object. So, we won't leave extra objects on the chart or in the list of objects, let's keep the system as light as possible.

Now, pay attention to the DispatchMessage function which starts as follows:

```
void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        int     w = 0;
        uint    key;
        static int b1 = 0;
        static double memPrice = 0;
```

Right after that, we have the code that will start handling the first event.

```
switch (id)
{
        case CHARTEVENT_MOUSE_MOVE:
                Position.X = (int)lparam;
                Position.Y = (int)dparam;
                ChartXYToTimePrice(Terminal.Get_ID(), Position.X, Position.Y, w, Position.dt, Position.price);
                ObjectMove(Terminal.Get_ID(), def_NameObjectLineH, 0, 0, Position.price = Terminal.AdjustPrice(Position.price));
                if (b1 > 0) ObjectMove(Terminal.Get_ID(), def_NameObjectLineV, 0, Position.dt, 0);
                key = (uint) sparam;
                if ((key & 0x10) == 0x10)    //Middle button....
                {
                        CreateObjectsIntern();
                        b1 = 1;
                }
```

When we press the middle mouse button, we generate a call. But this is not the case now. Then we will see what this function does. Note that we are trying to move an object that does not exist because it is not in the list of objects supported by MetaTrader 5. This call will only happen when the middle mouse button is pressed. Note the **_b1_** variable which controls at what point the trader is inside the set involved in the generation of the study.

As soon as the user clicks the left mouse button and the first step is completed, we will have the following code running:

```
if (((key & 0x01) == 0x01) && (b1 == 1))
{
        ChartSetInteger(Terminal.Get_ID(), CHART_MOUSE_SCROLL, false);
        ObjectMove(Terminal.Get_ID(), def_NameObjectLineT, 0, Position.dt, memPrice = Position.price);
        b1 = 2;
}
```

It will position the trend line and will call the next step in which the value of the b1 variable is changed. At this point we can move on to the next fragment.

```
if (((key & 0x01) == 0x01) && (b1 == 2))
{
        ObjectMove(Terminal.Get_ID(), def_NameObjectLineT, 1, Position.dt, Position.price);
        ObjectSetInteger(Terminal.Get_ID(), def_NameObjectLineT, OBJPROP_COLOR, (memPrice > Position.price ? m_Infos.cor03 : m_Infos.cor02));
        ObjectSetInteger(Terminal.Get_ID(), def_NameObjectText, OBJPROP_COLOR, (memPrice > Position.price ? m_Infos.cor03 : m_Infos.cor02));
        ObjectMove(Terminal.Get_ID(), def_NameObjectBitMp, 0, Position.dt, Position.price);
        ObjectSetInteger(Terminal.Get_ID(), def_NameObjectBitMp, OBJPROP_ANCHOR, (memPrice > Position.price ? ANCHOR_RIGHT_UPPER : ANCHOR_RIGHT_LOWER));
        ObjectSetString(Terminal.Get_ID(), def_NameObjectText, OBJPROP_TEXT, StringFormat("%.2f ", Position.price - memPrice));
        ObjectMove(Terminal.Get_ID(), def_NameObjectText, 0, Position.dt, Position.price);
        ObjectSetInteger(Terminal.Get_ID(), def_NameObjectText, OBJPROP_ANCHOR, (memPrice > Position.price ? ANCHOR_RIGHT_UPPER : ANCHOR_RIGHT_LOWER));
}
```

This fragment above is what will actually show the study on the screen. All these objects that are in this fragment will not exist when the study is over, they will be created and destroyed within this routine. Although this doesn't seem very efficient to do this, I didn't notice any decrease or increase in processing time during the study phase. In fact, I did notice a slight improvement in the order system, something very subtle, which is practically within the margin of error of the comparative estimate. So, I can't say that these changes actually brought improvements in terms of processing.

But note that the study will be performed while the mouse left button is pressed; as soon as we release it, the next fragment will be executed.

```
if (((key & 0x01) != 0x01) && (b1 == 2))
{
        b1 = 0;
        ChartSetInteger(Terminal.Get_ID(), CHART_MOUSE_SCROLL, true);
        Hide();
        Show();
}
Position.ButtonsStatus = (b1 == 0 ? key : 0);
```

Here we remove all objects used to create the study from the list of objects. Let's show the mouse line on the screen again. The highlighted code is a great idea as it prevents any function or subroutine inside the EA from getting false readings when we capture the mouse buttons. If any study is being done, the EA should ignore the button states. For this purpose, we use the highlighted lines. It is not a perfect solution but better than nothing.

We did not consider the code that creates objects to run the study. But since this is a fairly simple function, I will not focus on it in the article.

### Conclusion

Although the changes may seem minor, they all make a big difference to the system itself. There is one thing to remember: our command system is based on graphical objects on the screen, so the more objects the EA processes, the lower its performance will be when we request a particular object. To further complicate the situation, the system operates in real time, i.e. the faster our EA's system, the better its performance will be. Therefore, the less things the EA has to do, the better. Ideally, it should be able to work only with the order system, and we should take everything else to another level, and MetaTrader 5 should take care of it. This we will do, of course, gradually, since we will have to make many small changes, but nothing too complicated. This will be done in the next few articles dedicated solely to improving the reliability of the EA.

I can say one thing for sure: in the future, the EA will be responsible only for the order system. In the next article, we will give the EA a very interesting final look: we will further reduce the number of objects that are present in the list during the operation of the EA, since the order system is a large object generator, and will see how to change this system in such a way as to minimize the load that it creates on MetaTrader 5.

Because of this, I do not attach any modifications to this article as the code itself will still be subject to change. But don't worry, it's worth waiting for the next article. These changes will significantly increase the overall performance of our Expert Advisor. So, see you in the next article in this series.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10593](https://www.mql5.com/pt/articles/10593)

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

**[Go to discussion](https://www.mql5.com/en/forum/434776)**

![Developing a trading Expert Advisor from scratch (Part 25): Providing system robustness (II)](https://c.mql5.com/2/48/development__1.png)[Developing a trading Expert Advisor from scratch (Part 25): Providing system robustness (II)](https://www.mql5.com/en/articles/10606)

In this article, we will make the final step towards the EA's performance. So, be prepared for a long read. To make our Expert Advisor reliable, we will first remove everything from the code that is not part of the trading system.

![Population optimization algorithms](https://c.mql5.com/2/48/logo.png)[Population optimization algorithms](https://www.mql5.com/en/articles/8122)

This is an introductory article on optimization algorithm (OA) classification. The article attempts to create a test stand (a set of functions), which is to be used for comparing OAs and, perhaps, identifying the most universal algorithm out of all widely known ones.

![Developing a trading Expert Advisor from scratch (Part 26): Towards the future (I)](https://c.mql5.com/2/48/development__2.png)[Developing a trading Expert Advisor from scratch (Part 26): Towards the future (I)](https://www.mql5.com/en/articles/10620)

Today we will take our order system to the next level. But before that, we need to solve a few problems. Now we have some questions that are related to how we want to work and what things we do during the trading day.

![Neural networks made easy (Part 23): Building a tool for Transfer Learning](https://c.mql5.com/2/48/Neural_networks_made_easy_023.png)[Neural networks made easy (Part 23): Building a tool for Transfer Learning](https://www.mql5.com/en/articles/11273)

In this series of articles, we have already mentioned Transfer Learning more than once. However, this was only mentioning. in this article, I suggest filling this gap and taking a closer look at Transfer Learning.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/10593&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051675898941396090)

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