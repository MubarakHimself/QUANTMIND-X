---
title: Developing a trading Expert Advisor from scratch (Part 11): Cross order system
url: https://www.mql5.com/en/articles/10383
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:47:56.585320
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/10383&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051691536917320906)

MetaTrader 5 / Trading


### Introduction

There is one type of assets that makes traders' life very difficult for traders — it is the futures contracts. But why do they make life difficult? When a financial instrument contract expires, a new contract is created which we then trade. Actually, upon contract expiration, we need to finish any its analysis, save everything as a template and import this template into a new contract to continue the analysis. This is a common thing for anyone trade this type of asset, but even futures contracts have some history, and using this history, we can analyze them on an ongoing basis.

Professional traders like to analyze certain past information, in which case a second chart is needed. Now, there is no need to have the second chart if we use the appropriate tools. One of such tools is the use of the cross order system.

### Planning

In the first article within this series, we already mentioned this type of orders, but we did not get to the implementation. In that article, we focused on some other things, as we were launching a full system which could work in the MetaTrader 5 platform. In this article, we will show how to implement this functionality.

To understand the reasons for creating this functionality, take a look at the following two images:

![](https://c.mql5.com/2/44/001__2.jpg)![](https://c.mql5.com/2/44/002__2.jpg)

The image on the left is a typical futures contract, in this case it is MINI DOLLAR FUTURE, which started a few days ago, as can be seen from the chart. The chart on the right shows the same contract and contains additional data which actually represents the values of expired contracts, so the chart on the right is a historical chart. The chart on the right is more suitable for analyzing old support and resistance levels. But a problem arises if we need to trade. It is shows below:

![](https://c.mql5.com/2/44/003__2.jpg)![](https://c.mql5.com/2/44/004__1.jpg)

As you can see, the traded symbol is specified in CHART TRADE, and even if we use history CHART TRADE says that we can send an order - this can be seen from the toolbar. In the image on the left, the chart has an order created for the current contract, but in the image of the right the order can only be seen in the message box, while there is nothing visible on the chart.

You might think that this is just a display problem, but no, everything is much more complicated. This is what we are going to deal with in this article.

Important! _**Here we'll see how to create rules to be able to use historical data for work. In our case, these rules will be focused on working with the Mini Dollar (WDO) and Mini Index (WIN), which are traded on the Brazilian Exchange (B3). A correct understanding will allow you to adapt the rules for any type of futures contract, from any exchange in the world.**_

The system is not limited to one asset or another, while it is all about adapting the right parts of the code. If this is done correctly, then we will have an Expert Advisor with which we will not have to worry about whether an asset contract is approaching expiration and what the next contract will be - the EA will do this for us, by replacing the contract with the correct one as needed.

### How to understand the rules of the game

WDO (mini dollar), WIN (mini index), DOL (dollar future) and IND (index future) futures contracts follow very specific rules regarding maturity and contract specification. First, let us see how to find out the contract expiration date:

![](https://c.mql5.com/2/44/005__1.jpg)

Pay attention to the highlighted information: blue shows the contract expiration date and the red one indicates the date when the contract will cease to exist, after which it will no longer be traded. Knowing this is very important.

The contract duration is specified in the contract itself, but no name is specified there. Fortunately, we can easily find the name based on the rules which are strict and are used throughout the market. In the case of dollar and index futures contracts, we have the following:

The first three letters indicate contract type:

| Code | Contract |
| --- | --- |
| WIN | Mini Ibovespa index futures contract |
| IND | Ibovespa index futures contract |
| WDO | Mini dollar futures contract |
| DOL | Dollar futures contract |

This code is followed by a letter that indicates the contract expiration month:

| Expiration month | Letter representing WDO and DOL | Letter representing WIN and IND |
| --- | --- | --- |
| January | F |  |
| February | G | G |
| March | H |  |
| April | J | J |
| May | K |  |
| June | M | M |
| July | N |  |
| August | Q | Q |
| September | U |  |
| October | V | V |
| November | X |  |
| December | Z | Z |

These are followed by two digits representing the contract expiration year. For example, a dollar future contract expiring in April 2022 is indicated as DOLJ22. This is the contract that can be traded until the beginning of May. When May begins, the contract will expire. Since the rule slightly differs for WIN and IND, the contract actually expires on the Wednesday closest to the 15th of the indicated month. So, the rule is more complicated, but the EA can manage this, and it will always provide the correct contract.

### Implementation

Our EA already has the necessary points to receive the rules. Here we will need to implement some settings concerning the order sending system. So, let's get to work. First of all, add the following code to the C\_Terminal class object:

```
void CurrentSymbol(void)
{
        MqlDateTime mdt1;
        string sz0, sz1;
        datetime dt = TimeLocal();

        sz0 = StringSubstr(m_Infos.szSymbol = _Symbol, 0, 3);
        if ((sz0 != "WDO") && (sz0 != "DOL") && (sz0 != "WIN") && (sz0 != "IND")) return;
        sz1 = ((sz0 == "WDO") || (sz0 == "DOL") ? "FGHJKMNQUVXZ" : "GJMQVZ");
        TimeToStruct(TimeLocal(), mdt1);
        for (int i0 = 0, i1 = mdt1.year - 2000;;)
        {
                m_Infos.szSymbol = StringFormat("%s%s%d", sz0, StringSubstr(sz1, i0, 1), i1);
                if (i0 < StringLen(sz1)) i0++; else
                {
                        i0 = 0;
                        i1++;
                }
                if (macroGetDate(dt) < macroGetDate(SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_EXPIRATION_TIME))) break;
        }
}
```

This code uses the rules we've seen above to generate the asset name. To make sure we always use the current contract, we will implement a check shown in the highlighted line, i.e. the asset should be valid for the platform, and the EA will use the generated name. If you want to work with other futures contracts, you should adapt the previous code so that the name is generated correctly, as the name may vary from case to case. But the code is not limited only to assets that are linked to it - it can be used to reflect any type of futures contract, as long as you use a correct rule.

Next comes the part with the order details. If you use the system at this development stage, you will see the following behavior:

![](https://c.mql5.com/2/44/ScreenRecorderProject34.gif)

In other words, you can already have the cross order mode, but it is not yet fully implemented - there is no indication of the order on the chart. It is not as hard to implement as many of you might have imagined, as we need to indicate orders using horizontal lines. But that's not all. When we use cross orders, we miss some things provided by MetaTrader 5/ And thus we need to implement the missing logic so that the order system can work securely, stably and reliably. Otherwise, the use of cross orders can cause problems.

Looking from this point of view, it doesn't seem that simple. In fact, it's not simple, since we will have to create all the logic that the MetaTrader platform originally offers. So, the first thing to do is forget about the internal MetaTrader system - it will not be available to us from the moment we start using the cross order system.

From now on, the order ticket will dictate the rules. But this has some negative consequences. One of the most negative ones is that we don't know how many orders are placed on a chart. Limiting their number would definitely be unpleasing for the trader. Therefore, we need to do something to allow the trader to use the system the same way, as it is normally done with the full MetaTrader logic. This is the first problem to solve.

### Class C\_HLineTrade

To solve this problem, we will create a new class C\_HLineTrade, which will replace the system displaying orders on the chart, provided by MetaTrader 5. So, let's start with the class declaration:

```
class C_HLineTrade
{
#define def_NameHLineTrade "*HLTSMD*"
        protected:
                enum eHLineTrade {HL_PRICE, HL_STOP, HL_TAKE};
        private :
                color   m_corPrice,
                        m_corStop,
                        m_corTake;
                string  m_SelectObj;
```

Note that a few things are defined here - they will be frequently used in the code. Therefore, please be very attentive with further changes - in fact, there will be a lot of changes. Next we declare the class constructor and destructor:

```
C_HLineTrade() : m_SelectObj("")
{
        ChartSetInteger(Terminal.Get_ID(), CHART_SHOW_TRADE_LEVELS, false);
        RemoveAllsLines();
};
//+------------------------------------------------------------------+
~C_HLineTrade()
{
        RemoveAllsLines();
        ChartSetInteger(Terminal.Get_ID(), CHART_SHOW_TRADE_LEVELS, true);
};
```

The constructor will prevent the original lines from being visible, while the destructor will put them back on the chart. Both functions have a common function, which is as follows:

```
void RemoveAllsLines(void)
{
        string sz0;
        int i0 = StringLen(def_NameHLineTrade);

        for (int c0 = ObjectsTotal(Terminal.Get_ID(), -1, -1); c0 >= 0; c0--)
        {
                sz0 = ObjectName(Terminal.Get_ID(), c0, -1, -1);
                if (StringSubstr(sz0, 0, i0) == def_NameHLineTrade) ObjectDelete(Terminal.Get_ID(), sz0);
        }
}
```

The highlighted line checks whether the object (in this case it is a horizontal line), is one of the objects used by the class. If it is, it will delete the object. Note that we don't know how many objects we have, but the system will check object by object, trying to clean up everything that has been created by the class. The next recommended function from this class is shown below:

```
inline void SetLineOrder(ulong ticket, double price, eHLineTrade hl, bool select)
{
        string sz0 = def_NameHLineTrade + (string)hl + (string)ticket, sz1;

        if (price <= 0)
        {
                ObjectDelete(Terminal.Get_ID(), sz0);
                return;
        }
        if (!ObjectGetString(Terminal.Get_ID(), sz0, OBJPROP_TOOLTIP, 0, sz1))
        {
                ObjectCreate(Terminal.Get_ID(), sz0, OBJ_HLINE, 0, 0, 0);
                ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_COLOR, (hl == HL_PRICE ? m_corPrice : (hl == HL_STOP ? m_corStop : m_corTake)));
                ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_WIDTH, 1);
                ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_STYLE, STYLE_DASHDOT);
                ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_SELECTABLE, select);
                ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_SELECTED, false);
                ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_BACK, true);
                ObjectSetString(Terminal.Get_ID(), sz0, OBJPROP_TOOLTIP, (string)ticket + " "+StringSubstr(EnumToString(hl), 3, 10));
        }
        ObjectSetDouble(Terminal.Get_ID(), sz0, OBJPROP_PRICE, price);
}
```

For this function, it does not matter how many objects will be created and whether the object exists by the tome it is called. It ensures that the line is created and placed in the right place. This created line replaces the one originally used in MetaTrader.

Our purpose is to make it functional rather than nice and beautiful. That is why the lines are not selected when they are created - you can change this behavior if needed. But I use the MetaTrader 5 messaging system to position lines. So, to be able to move them around, you will have to indicate this explicitly. To indicate which line is being adjusted, we have another function:

```
inline void Select(const string &sparam)
{
        int i0 = StringLen(def_NameHLineTrade);

        if (m_SelectObj != "") ObjectSetInteger(Terminal.Get_ID(), m_SelectObj, OBJPROP_SELECTED, false);
        m_SelectObj = "";
        if (StringSubstr(sparam, 0, i0) == def_NameHLineTrade)
        {
                if (ObjectGetInteger(Terminal.Get_ID(), sparam, OBJPROP_SELECTABLE))
                {
                        ObjectSetInteger(Terminal.Get_ID(), sparam, OBJPROP_SELECTED, true);
                        m_SelectObj = sparam;
                };
        }
}
```

This function implements the selection of a line. If there is another line selected, it will cancel the previous selection. This is all simple. The function will only manipulate lines actually handled by the class. Another function of this class, which is worth mentioning, is the following:

```
bool GetNewInfosOrder(const string &sparam, ulong &ticket, double &price, eHLineTrade &hl)
{
        int i0 = StringLen(def_NameHLineTrade);

        if (StringSubstr(sparam, 0, i0) == def_NameHLineTrade)
        {
                hl = (eHLineTrade) StringToInteger(StringSubstr(sparam, i0, 1));
                ticket = (ulong)StringToInteger(StringSubstr(sparam, i0 + 1, StringLen(sparam)));
                price = ObjectGetDouble(Terminal.Get_ID(), sparam, OBJPROP_PRICE);
                return true;
        }
        return false;
}
```

This function is perhaps the most important in this class: since we don't know how many lines are on the chart, we need to know which line the user is manipulating. This function does just that - it tells the system which line is being manipulated.

But this is only a small part of what we need to do. The system is still far from being functional. So let us move on to the next step - we will add and modify the functions of the C\_Router class, which is responsible for order routing. This class inherits the functionality we just created in the C\_HLineTrade class. See the following code:

```
#include "C_HLineTrade.mqh"
//+------------------------------------------------------------------+
class C_Router : public C_HLineTrade
```

### New class C\_Router

The source C\_Router class had a limitation allowing to have only one open order. This limitation will be lifted, for which we need to make important changes to the C\_Router class.

The first change is in the class update function, which now looks like this:

```
void UpdatePosition(void)
{
        static int memPositions = 0, memOrder = 0;
        ulong ul;
        int p, o;

        p = PositionsTotal() - 1;
        o = OrdersTotal() - 1;
        if ((memPositions != p) || (memOrder != o))
        {
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
                RemoveAllsLines();
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
                memOrder = o;
                memPositions = p;
        };
        for(int i0 = p; i0 >= 0; i0--) if(PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                ul = PositionGetInteger(POSITION_TICKET);
                SetLineOrder(ul, PositionGetDouble(POSITION_PRICE_OPEN), HL_PRICE, false);
                SetLineOrder(ul, PositionGetDouble(POSITION_TP), HL_TAKE, true);
                SetLineOrder(ul, PositionGetDouble(POSITION_SL), HL_STOP, true);
        }
        for (int i0 = o; i0 >= 0; i0--) if ((ul = OrderGetTicket(i0)) > 0) if (OrderGetString(ORDER_SYMBOL) == Terminal.GetSymbol())
        {
                SetLineOrder(ul, OrderGetDouble(ORDER_PRICE_OPEN), HL_PRICE, true);
                SetLineOrder(ul, OrderGetDouble(ORDER_TP), HL_TAKE, true);
                SetLineOrder(ul, OrderGetDouble(ORDER_SL), HL_STOP, true);
        }
};
```

Previously, this function collected data on only one open position and saved it to its observatory. Now the function will display absolutely all open positions and pending orders on the chart. It is definitely a replacement for the system provided by MetaTrader. As these are serious things, it is important to understand how it works, because if it fails, this will affect the entire cross order system. So, before we trade on a real account, let's test this system on a demo account. Such systems must be properly tested until we are absolutely sure that everything works as it should. First, we need to configure the system because it works a little differently from how MetaTrader 5 works.

See the highlighted lines and answer honestly: Is it clear what they actually do? The reason why these two code lines are here will become clear later, when we talk about the C\_OrderView class later in this article. Without these two lines, the code is very unstable and works strangely. As for the rest code, it is quite simple - it creates each of the lines via the C\_HLineTrade class object. In this case we have only one line which cannot be selected. This is easily indicated, as shown in the code below:

```
SetLineOrder(ul, PositionGetDouble(POSITION_PRICE_OPEN), HL_PRICE, false);
```

In other words, the system has become very simple and straightforward. The function is called by the EA during an event in OnTrade:

```
C_TemplateChart Chart;

// ... Expert Advisor code ...

void OnTrade()
{
        Chart.DispatchMessage(CHARTEVENT_CHART_CHANGE, 0, Chart.UpdateRoof(), C_Chart_IDE::szMsgIDE[C_Chart_IDE::eROOF_DIARY]);
        Chart.UpdatePosition();
}

// ... The rest of the Expert Advisor code ...
```

The highlighted code will enable the update of commands on the screen. Notice that we are using the C\_TemplateChart chart for this - this is because the structure of classes in the system has changed. The new structure is shown below:

![](https://c.mql5.com/2/44/006.jpg)

This structure enables a directed flow of messages in within the EA directional. Whenever you are in doubt about how the message flow gets into a particular class, take a look at this class inheritance graph. The only class considered public is the C\_Terminal object class, while all others are handled by inheritance between classes, and absolutely no variables are public in this system.

Now, since the system does not analyze just a single order, it is necessary to understand something else: How to understand the result of operations? Why is it important? When you only have one open position, the system can easily understand everything, but as the number of open positions increases, you need to figure out what's going on. Here is the function that provide this information:

```
void OnTick()
{
        Chart.DispatchMessage(CHARTEVENT_CHART_CHANGE, 0, Chart.CheckPosition(), C_Chart_IDE::szMsgIDE[C_Chart_IDE::eRESULT]);
}
```

There are not many changes. Take a look at the highlighted function code:

```
inline double CheckPosition(void)
{
        double Res = 0, last, sl;
        ulong ticket;

        last = SymbolInfoDouble(Terminal.GetSymbol(), SYMBOL_LAST);
        for (int i0 = PositionsTotal() - 1; i0 >= 0; i0--) if (PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                ticket = PositionGetInteger(POSITION_TICKET);
                Res += PositionGetDouble(POSITION_PROFIT);
                sl = PositionGetDouble(POSITION_SL);
                if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {
                        if (last < sl) ClosePosition(ticket);
                }else
                {
                        if ((last > sl) && (sl > 0)) ClosePosition(ticket);
                }
        }
        return Res;
};
```

The function consists of three highlighted parts: the yellow part informs about the result of open positions, and green parts check the position in case the stop loss is missed due to high volatility, in which case it is necessary to close the position as soon as possible. Thus, this function does not return the result of o single position, except when you one open position for a specific asset.

There are also other functions that help the system continue working when we use the cross order model. Take a look at them in the code below:

```
bool ModifyOrderPendent(const ulong Ticket, const double Price, const double Take, const double Stop, const bool DayTrade = true)
{
        if (Ticket == 0) return false;
        ZeroMemory(TradeRequest);
        ZeroMemory(TradeResult);
        TradeRequest.action     = TRADE_ACTION_MODIFY;
        TradeRequest.order      = Ticket;
        TradeRequest.price      = NormalizeDouble(Price, Terminal.GetDigits());
        TradeRequest.sl         = NormalizeDouble(Stop, Terminal.GetDigits());
        TradeRequest.tp         = NormalizeDouble(Take, Terminal.GetDigits());
        TradeRequest.type_time  = (DayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
        TradeRequest.expiration = 0;
        return OrderSend(TradeRequest, TradeResult);
};
//+------------------------------------------------------------------+
bool ModifyPosition(const ulong Ticket, const double Take, const double Stop)
{
        ZeroMemory(TradeRequest);
        ZeroMemory(TradeResult);
        if (!PositionSelectByTicket(Ticket)) return false;
        TradeRequest.action     = TRADE_ACTION_SLTP;
        TradeRequest.position   = Ticket;
        TradeRequest.symbol     = PositionGetString(POSITION_SYMBOL);
        TradeRequest.tp         = NormalizeDouble(Take, Terminal.GetDigits());
        TradeRequest.sl         = NormalizeDouble(Stop, Terminal.GetDigits());
        return OrderSend(TradeRequest, TradeResult);
};
```

The first one is responsible for the modification of the order that is still open, and the other one is modifying the open position. Although they seem to be the same, they are not. There is another important function for the system:

```
bool RemoveOrderPendent(ulong Ticket)
{
        ZeroMemory(TradeRequest);
        ZeroMemory(TradeResult);
        TradeRequest.action     = TRADE_ACTION_REMOVE;
        TradeRequest.order      = Ticket;
        return OrderSend(TradeRequest, TradeResult);
};
```

With this last function we complete considering the C\_Router class. We have implemented the basic system which covers the functionality normally supported in MetaTrader - no, because of the system of cross orders, we can no longer count on this support. However, the system is not complete yet. We need to add something else to make the system really work. At the moment, if there is an order, it will look like below. This is required to be able to complete the next step.

![](https://c.mql5.com/2/44/007.jpg)

Take a close look at the image above. The message box shows the open order and the asset for which it has been opened. The traded asset is indicated in CHART TRADE. Note that it is the same asset which is indicated in the message box. Now, let's check the asset displayed on the chart. The name can be checked in the chart window header. But it's completely different - it is not an asset on the chart, but it is the mini-index history which means that now we don't use the MetaTrader 5 internal system but use the cross order system described in this article. Now we only have the functionality that allows showing where the order is located. But it is not enough as we want to have a fully functional system that would allow the operation via the system of cross orders. So, we need something else. As for the event related to order moving, this will be implemented in another class.

### New functionality in the C\_OrderView class

While the C\_OrderView object class can do a few things, it can't handle open or pending order data yet. However, when we add a messaging system to it, we have more possibilities to use it. This is the only addition we will make to the class for now. The full function code is shown below:

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{
        ulong           ticket;
        double          price, pp, pt, ps;
        eHLineTrade     hl;

        switch (id)
        {
                case CHARTEVENT_MOUSE_MOVE:
                        MoveTo((int)lparam, (int)dparam, (uint)sparam);
                        break;
                case CHARTEVENT_OBJECT_DELETE:
                        if (GetNewInfosOrder(sparam, ticket, price, hl))
                        {
                                if (OrderSelect(ticket))
                                {
                                        switch (hl)
                                        {
                                                case HL_PRICE:
                                                        RemoveOrderPendent(ticket);
                                                        break;
                                                case HL_STOP:
                                                        ModifyOrderPendent(ticket, OrderGetDouble(ORDER_PRICE_OPEN), OrderGetDouble(ORDER_TP), 0);
                                                        break;
                                                case HL_TAKE:
                                                        ModifyOrderPendent(ticket, OrderGetDouble(ORDER_PRICE_OPEN), 0, OrderGetDouble(ORDER_SL));
                                                        break;
                                        }
                                }else if (PositionSelectByTicket(ticket))
                                {
                                        switch (hl)
                                        {
                                                case HL_PRICE:
                                                        ClosePosition(ticket);
                                                        break;
                                                case HL_STOP:
                                                        ModifyPosition(ticket, OrderGetDouble(ORDER_TP), 0);
                                                        break;
                                                case HL_TAKE:
                                                        ModifyPosition(ticket, 0, OrderGetDouble(ORDER_SL));
                                                        break;
                                        }
                                }
                        }
                        break;
                case CHARTEVENT_OBJECT_CLICK:
                        C_HLineTrade::Select(sparam);
                        break;
                case CHARTEVENT_OBJECT_DRAG:
                        if (GetNewInfosOrder(sparam, ticket, price, hl))
                        {
                                price = AdjustPrice(price);
                                if (OrderSelect(ticket)) switch(hl)
                                {
                                        case HL_PRICE:
                                                pp = price - OrderGetDouble(ORDER_PRICE_OPEN);
                                                pt = OrderGetDouble(ORDER_TP);
                                                ps = OrderGetDouble(ORDER_SL);
                                                if (!ModifyOrderPendent(ticket, price, (pt > 0 ? pt + pp : 0), (ps > 0 ? ps + pp : 0))) UpdatePosition();
                                                break;
                                        case HL_STOP:
                                                if (!ModifyOrderPendent(ticket, OrderGetDouble(ORDER_PRICE_OPEN), OrderGetDouble(ORDER_TP), price)) UpdatePosition();
                                                break;
                                        case HL_TAKE:
                                                if (!ModifyOrderPendent(ticket, OrderGetDouble(ORDER_PRICE_OPEN), price, OrderGetDouble(ORDER_SL))) UpdatePosition();
                                                break;
                                }
                                if (PositionSelectByTicket(ticket)) switch (hl)
                                {
                                        case HL_PRICE:
                                                UpdatePosition();
                                                break;
                                        case HL_STOP:
                                                ModifyPosition(ticket, PositionGetDouble(POSITION_TP), price);
                                                break;
                                        case HL_TAKE:
                                                ModifyPosition(ticket, price, PositionGetDouble(POSITION_SL));
                                                break;
                                }
                        };
                break;
        }
}
```

This code completes the system of cross orders. Our capabilities have increased so that we can do almost the same thing that were possible without the cross order system. In general, the function should be quite clear. But it has an event type which is not very common - [CHARTEVENT\_OBJECT\_DELETE](https://www.mql5.com/en/docs/constants/chartconstants/charts_samples#chart_event_object_delete). When the user deletes a line, it will be reflected on the chart and in the order system, therefore you should be very careful when you start deleting lines from the chart. We don't need to worry when we remove the EA from the chart, as the orders will remain intact, as shown in the following animation:

![](https://c.mql5.com/2/44/ScreenRecorderProject38.gif)

But if the EA is on the chart, we must be very careful when deleting lines from the chart, especially those that are hidden in the list of objects. Otherwise, you can see below what happens in the order system when we remove the lines created by the cross order system.

![](https://c.mql5.com/2/44/ScreenRecorderProject40.gif)

And to finish the demo of the system, let's see what happens to the order when we drag the price lines. Remember the following: the dragged line must be selected; if it is not selected, then it will be impossible to move it. The price change will occur when the line is released on the chart, before that the price will remain in the same position as before.

![](https://c.mql5.com/2/44/ScreenRecorderProject41.gif)

If it is difficult to know whether a line is selected or not, let's make changes in the selection code. The changes are highlighted below. this change is already implemented in the attached version.

```
inline void Select(const string &sparam)
{
        int i0 = StringLen(def_NameHLineTrade);

        if (m_SelectObj != "")
        {
                ObjectSetInteger(Terminal.Get_ID(), m_SelectObj, OBJPROP_SELECTED, false);
                ObjectSetInteger(Terminal.Get_ID(), m_SelectObj, OBJPROP_WIDTH, 1);
        }
        m_SelectObj = "";
        if (StringSubstr(sparam, 0, i0) == def_NameHLineTrade)
        {
                if (ObjectGetInteger(Terminal.Get_ID(), sparam, OBJPROP_SELECTABLE))
                {
                        ObjectSetInteger(Terminal.Get_ID(), sparam, OBJPROP_SELECTED, true);
                        ObjectSetInteger(Terminal.Get_ID(), sparam, OBJPROP_WIDTH, 2);
                        m_SelectObj = sparam;
                };
        }
}
```

The result of this code modification can be seen in the figure below.

![](https://c.mql5.com/2/44/ScreenRecorderProject42.gif)

### Conclusion

So, here I have shown how to create a cross order system in MetaTrader. I hope that this system will be useful to anyone who will use this knowledge. But please remember the following: before you start trading on a live account with this system, you should test it as thoroughly as possible in many different market scenarios, because although this system is implemented in the MetaTrader platform, it has almost no support from the platform side in terms of error handling, so if they happen by chance, you will have to act quickly so as not to get big losses. But by testing it in different scenarios, you can find out where the problems arise: in movements, maximum number of orders that your computer can handle, the maximum allowable spread for the analysis system, allowable volatility level for open orders, since the greater the number of open orders and information to analyze, the more likely something bad will happen. This is because each of the orders is analyzed on every tick received by the system, and this can be a problem when many orders are open at the same time.

I recommend to not trust this system until you test it on a demo account with a lot of scenarios. Even if though the code seems perfect, it does not have any error analysis.

I am attaching the code of all Expert Advisors as at the moment.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10383](https://www.mql5.com/pt/articles/10383)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10383.zip "Download all attachments in the single ZIP archive")

[EA.zip](https://www.mql5.com/en/articles/download/10383/ea.zip "Download EA.zip")(12013.17 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/427919)**
(3)


![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
25 May 2022 at 20:01

Congratulations on this excellent article Daniel.

I think the only problem will be at the turn of the year, when the "CurrentSymbol" function needs to look up the [name of](https://www.mql5.com/en/docs/predefined/_symbol "MQL5 documentation: string _Symbol") next year's [symbol](https://www.mql5.com/en/docs/predefined/_symbol "MQL5 documentation: string _Symbol"). It seems to me that the value of i1 will always return the number of the current year (22), but in December we already start using the symbol ending in 23.

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
26 May 2022 at 12:52

**Guilherme Mendonca name of next year's [symbol](https://www.mql5.com/en/docs/predefined/_symbol "MQL5 documentation: string _Symbol"). It seems to me that the value of i1 will always return the number of the current year (22), but in December we already start using the symbol ending in 23.**

In reality, this problem won't happen and the reason for this is what makes the Loop end....

```
                                for (int i0 = 0, i1 = mdt1.year - 2000;;)
                                {
                                        m_Infos.szSymbol = StringFormat("%s%s%d", sz0, StringSubstr(sz1, i0, 1), i1);
                                        m_Infos.szFullSymbol = StringFormat("%s%s%d", sz2, StringSubstr(sz1, i0, 1), i1);
                                        if (i0 < StringLen(sz1)) i0++; else
                                        {
                                                i0 = 0;
                                                i1++;
                                        }
                                        if (macroGetDate(dt) < macroGetDate(SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_EXPIRATION_TIME))) break;
                                }
```

Only when this condition is met will the loop end, and the value of i1 will always be incremented... so when the year is changed, the asset will be modified automatically ....

![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
30 May 2022 at 23:10

**Daniel Jose [#](https://www.mql5.com/pt/forum/399478#comment_39814489):**

In fact, this problem won't happen and the reason for it is what causes the TIE to end....

Only when this highlighted condition is reached will the loop end, and the value of i1 will always be incremented... so when the year is changed, the asset will be modified automatically ....

You're right.

I hadn't paid attention to the line incrementing the value of i1.

![Developing a trading Expert Advisor from scratch (Part 12): Times and Trade (I)](https://c.mql5.com/2/46/development__3.png)[Developing a trading Expert Advisor from scratch (Part 12): Times and Trade (I)](https://www.mql5.com/en/articles/10410)

Today we will create Times & Trade with fast interpretation to read the order flow. It is the first part in which we will build the system. In the next article, we will complete the system with the missing information. To implement this new functionality, we will need to add several new things to the code of our Expert Advisor.

![DoEasy. Controls (Part 5): Base WinForms object, Panel control, AutoSize parameter](https://c.mql5.com/2/46/MQL5-avatar-doeasy-library-2__4.png)[DoEasy. Controls (Part 5): Base WinForms object, Panel control, AutoSize parameter](https://www.mql5.com/en/articles/10794)

In the article, I will create the base object of all library WinForms objects and start implementing the AutoSize property of the Panel WinForms object — auto sizing for fitting the object internal content.

![Developing a trading Expert Advisor from scratch (Part 13): Time and Trade (II)](https://c.mql5.com/2/46/development__4.png)[Developing a trading Expert Advisor from scratch (Part 13): Time and Trade (II)](https://www.mql5.com/en/articles/10412)

Today we will construct the second part of the Times & Trade system for market analysis. In the previous article "Times & Trade (I)" we discussed an alternative chart organization system, which would allow having an indicator for the quickest possible interpretation of deals executed in the market.

![Developing a trading Expert Advisor from scratch (Part 10): Accessing custom indicators](https://c.mql5.com/2/46/development.png)[Developing a trading Expert Advisor from scratch (Part 10): Accessing custom indicators](https://www.mql5.com/en/articles/10329)

How to access custom indicators directly in an Expert Advisor? A trading EA can be truly useful only if it can use custom indicators; otherwise, it is just a set of codes and instructions.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=wmklgyhooyhknhauhfszlixjtfljoweg&ssn=1769104075158100772&ssn_dr=0&ssn_sr=0&fv_date=1769104075&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10383&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20(Part%2011)%3A%20Cross%20order%20system%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176910407526244043&fz_uniq=5051691536917320906&sv=2552)

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