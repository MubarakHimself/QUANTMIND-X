---
title: Developing a trading Expert Advisor from scratch (Part 22): New order system (V)
url: https://www.mql5.com/en/articles/10516
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:46:56.990845
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/10516&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051679858901243022)

MetaTrader 5 / Trading


### Introduction

It is not that easy to implement a new system as we often encounter problems which greatly complicate the process. When these problems appear, we have to stop and re-analyze the direction in which we are moving, deciding whether we can leave the things as they are, or we should give it a new look.

Such decisions can be quite frequent while creating a system, especially if we do not have a clear deadline or budget: when we are not under pressure, we can test and adjust things to do our best to ensure stability in terms of development and improvement.

Serious programs, especially commercial ones developed under certain budgets or deadlines, usually contain many errors that need to be corrected later. Often, those who develop the system do not have time to implement or learn some solutions which could be very beneficial to the system. This is how we get programs that, in many cases, fall short of what programmers could create. Sometimes companies release version after version, with minor improvements or bug fixes. This is not because the bugs were detected after program release but because of the programmers were under pressure before the release.

This actually happens throughout the entire production chain. However, while we are in the process of creating the solution, we are looking for the best and preferably the easiest way. Moreover, we can afford to explore all possible and feasible solutions. If necessary, we can stop and go back a little in order to modify and improve the system development process. Very often, a small stop and change of direction can greatly help in developing the desired system.

In the article [Developing a trading system from scratch (Part 21)](https://www.mql5.com/en/articles/10499), the system was almost ready. It only lacked the part responsible for moving orders directly on the chart. To implement this part, I checked and noticed some oddities in the code: there were a lot of repeating parts in it. Although, we were very careful to avoid unnecessary repetitions. What's worse, some things did not work as they should, for example, when using very different assets. When we started to design the EA and when I began to document it in the articles that are published here, I was thinking about using it mostly for trading futures on the B3 exchange. Actually, I ended up developing it to trade dollar futures or index known as WDO and WIN. But as the system got closer and closer to the system that I originally wanted to use, I realized that it could be extended to other markets or assets. This is where the problem arose.

### 1.0. Getting back to clipboard

To understand the problem, pay attention to one important detail that many people overlook when developing an Expert Advisor. How are contracts defined in the trading system? MetaTrader 5 provides this information, while MQL5 allows accessing it. So, we need to understand the data in order to create mathematics to generalize the calculations and get the most complete system.

The order system that we are developing is designed to trade directly from the chart, without the need for any other external resources or anything that may be created in the future. This development is an attempt to create a trading system very similar to what is used in platforms, but completely open source so that you can customize and add the information you need.

The idea is to enable the user to know what's going on with a position just by looking at it. Although the idea seems to be very good, the trading system does not make life easy for those who are trying to create a system for FOREX and stock markets like B3. The level of information available is enough to customize orders for one market or another, but creating something general has become a problem, almost a personal insult. However, I decided to confront the problem and try to create a universal system.

While I'm not sure if I can really do it, I can at least show you how to find the required information, and thus if you need to customize the EA for your local system, you'll have the experience and knowledge of how to do it. Even if the system cannot originally handle the modeling, you will be able to adapt it to your needs.

An observation: the data can be obtained using MQL5 code, but to simplify the work, I will use MetaTrader 5 to show where the data is.

### 1.0.1. Company assets (Shares) traded on B3

Take a look at the following two images:

![](https://c.mql5.com/2/45/001.1__1.png)![](https://c.mql5.com/2/45/002.1.png)

The highlighted parts show the data we need in order to calculate how the position will be placed.

If you create a position based in a financial value, you will need these values in order to calculate the stop loss and take profit values. If you do not use OCO order, things will be even easier, but here we assume that all orders are OCO orders or positions.

Important: if an asset is traded as fractional shares (there are differences between markets on B3), the minimum volume will be 1% of the specified value, that is why instead of trading 100 to 100 we will be trading 1 to 1. The calculation will be different, so don't forget it.

### 1.0.2. Futures contracts in B3

The rules for futures contracts are different from stock trading rules, the volume changes depending on whether you are trading with a full contract or a mini contract, and even from one asset to another. For example, to trade BOI, we should look at how the levering (multiplier) is filled since it is from what we will do here. I will focus on contracts. In some cases, a full contract will equal 25 mini contracts, but this can vary so you should always check volume rules of the exchange.

Now let's take a look at the following images of the mini dollar:

![](https://c.mql5.com/2/45/003.1.png)![](https://c.mql5.com/2/45/004.1.png)

These contracts have an expiration date, but for this article it is not important, because in the article [Development of a trading Expert Advisor from scratch (Part 11)](https://www.mql5.com/en/articles/10383), we considered how to make a cross order system to trade futures contracts directly from your history, and if you use this method you don't have to worry about which contract is currently being traded, as the EA will do it for you. Pay attention to the marked points; they are different from the stock system. This may cause problems sometimes, but the EA has been handling this issue, although this part is not very intuitive. Anyway, over time you will get used to the right values that should be used in the leverage to trade specific financial volumes.

However, I decided to set a higher goal. This is where the difficulties begin.

### 1.0.3. FOREX

My problem was related to the Forex system, and my EA could not solve it. When I managed to handle Forex leverage, I found out that the same code from the B3 EA could not be used. That bothered me because I don't see why you should have two EAs which differ only in this part. To understand what's going on, take a look at the following images:

![](https://c.mql5.com/2/45/005.1.png)![](https://c.mql5.com/2/45/006.1.png)

In B3, we have four values, while having only two values in Forex. The calculations between the two markets did not match because of the two missing values.

Due to this, the calculations performed by the EA made it very difficult to understand what was actually being done: sometimes the calculated values were too high, sometimes the multiplier caused the trading system to reject the order because the volume was too small compared to the minimum expected volume, and sometimes OCO limit points were not accepted due to incorrect positions. So, there was endless confusion.

Having understood this, I decided to modify the EA. So, we will not calculate the value the previous way, no we will use a different method. In fact, this value will now be adjusted in accordance with the asset we work with, regardless of whether we do it in the stock market or in the Forex market. Thus, the EA will adapt to the data and will perform calculations correctly. However, we need to know what we are trading in order to set the multiplier correctly. And although now, for computational reasons, we allow the use of floating point values in the volume, you should avoid this. In fact, you should use integer values which indicate the levering being used.

So, we do not tell the EA how much we want to trade. Instead, we indicate the multiplier to be applied to the minimum allowed volume. This is how I solved the problem. You can test it for yourself and see for yourself how it works.

To make it easier to understand and to reduce work, in the next topic I provide a brief presentation of the results.

### 2.0. Data visualization

First, let us consider the case of the stock market, in particular the Brazilian stock market (B3). This is how the data is presented in the EA in 5.

### 2.0.1. B3 assets

In case of a company share quoted on the Brazilian stock exchange, the minimum trading volume is 100. By indicating 1, we set the EA to trade the minimum allowed volume. This way it is much easier to work with the volume — there is no need to know exactly the minimum lot, we simply use the multiplier, and the EA will to the necessary calculations to create a correct order.

If you use fractional values, just indicates the number of fractions to be used, that is, 50 means that you will use 50 fractions, if you specify 15, 15 fractions will be used and so on.

![](https://c.mql5.com/2/45/ScreenRecorderProject96.gif)

The result is shown below in the MetaTrader 5 toolbox window. It shows how an order with the minimum lot value was created. If you analyze the take and stop values, you will see that they match those indicated on the chart, which means that the EA worked here.

![](https://c.mql5.com/2/45/007__2.png)

### 2.0.2. Mini Dollar

Here the volume is 1 to 1, but in the case of full contracts the value is different. Take a look at the Chart Trade: the values indicated as Take and Stop are the same as those found above. But the EA adjusts the values correctly. So, the values that should be taken into account are the values found in the chart indicator, the Chart Trade values should be ignored, but they will be close to the values indicated on the chart.

![](https://c.mql5.com/2/45/ScreenRecorderProject97.gif)

The following data is shown in the toolbox:

![](https://c.mql5.com/2/45/008__1.png)

Just like in the previous asset, if we run the calculation to check the stop loss and take profit levels, we will see that they match those specified by the EA, that is, the EA also passed this stage without the need to recompile the code, only by changing the asset.

### 2.0.3. Forex

This part is a little more complicated. For those who are not familiar with Forex, the points are rather strange, but still the EA manages to figure them out.

![](https://c.mql5.com/2/45/ScreenRecorderProject98.gif)

MetaTrader 5 will inform us about this pending trade as follows:

![](https://c.mql5.com/2/45/009__1.png)

Remember that in Forex leverage levels are different from those shown above. But if you perform the calculations, you will see that the EA managed to provide the correct points, and the order was created perfectly.

All this is done without any additional changes except those that I will show in the implementation part, although many other changes have been made in addition to this, so that the EA is really suitable for both the stock market and the Forex market, without the need for recompilation. So, here we are completing previous articles. We will see how to move stop loss and take profit levels right on the chart.

### 3.0. Implementation

Let's start by looking at some changes in the code.

Firstly, I removed the limit system implemented 3 or 4 versions ago. This is because the EA sometimes incorrectly adjusted to the lot calculation between stock markets and forex.

I have added a new calculation model to enable the EA to work equally well in the forex and stock markets. This was not possible prior to this version. At the very beginning, the EA was focused on working on stock markets, but I decided to extend its functionality to forex since trading methods do not differ much.

There are details regarding lot calculation issues that the EA could not deal with in previous versions, but with the changes made, it can now be used in both forex and stock market without any major code changes. However, to maintain compatibility with forex and stock markets, I had to make multiple changes.

One of such changes is at the very beginning of the code:

```
int OnInit()
{
        static string   memSzUser01 = "";

        Terminal.Init();
        WallPaper.Init(user10, user12, user11);
        Mouse.Init(user50, user51, user52);
        if (memSzUser01 != user01)
        {
                Chart.ClearTemplateChart();
                Chart.AddThese(memSzUser01 = user01);
        }
        Chart.InitilizeChartTrade(user20 * Terminal.GetVolumeMinimal(), user21, user22, user23);
        VolumeAtPrice.Init(user32, user33, user30, user31);
        TimesAndTrade.Init(user41);
        TradeView.Initilize();

        OnTrade();
        EventSetTimer(1);

        return INIT_SUCCEEDED;
}
```

The highlighted part did not previously exist. There are also many other changes. However, we will not focus on this implementation. Instead, we will see how to use the EA to move the take and stop levels present on the chart directly, without using any other tricks. So, let's move on to this part.

### 3.0.1 Moving orders directly on the chart

This part was very difficult in previous versions, as they had a number of unresolved issues which appeared over time, which made the task difficult. One of the reasons was that the code was very sparse, which made it difficult to effectively implement the order moving system directly on the chart. Originally the system was not supposed to do so.

To give you an idea of the extent of the changes that the EA required (in order to be able to handle order movement, including pending orders and positions, so that you could control limit order movements on the chart using the mouse) let's see what the EA looked like before.

![](https://c.mql5.com/2/45/010.png)

The problem is that when the objects were created, they replaced the C\_HLineTrade class. This was done in the article [Developing a trading EA from scratch (Part 20)](https://www.mql5.com/en/articles/10497). The system now has a much more complex structure, so in order not to show the whole picture above again, we will only look at what happened.

![](https://c.mql5.com/2/45/011.png)

The arrow points to the connection point where the C\_HLineTrade class was removed to give room to new classes. This enabled more implementations which we did in previous articles. But the presence of the C\_OrderView class interfered with development, and eventually had to exclude it. But that's not all. The C\_TradeGraphics class was merged with the old C\_OrderView class and a new class named C\_IndicatorTradeView appeared. So, this class replaced two classes, and this allowed us to develop the order movement system.

What I'm going to present here is the first version of this system. There is another version currently being developed, but it will be presented in another article.

### 3.0.1.1 - Writing the code for the new system

After merging, the new system has the following configuration:

![](https://c.mql5.com/2/45/012.png)

The green area indicates a set of classes that are free, that is, they will be managed by MetaTrader 5 and not by the EA. But how is it done? Let us consider the process in detail. In fact, the EA will create, place and delete only classes and all objects that were created by these classes. Look inside the EA code and you will not find any structures or variables that will refer to the objects created in the green area. This allows the creation of an unlimited number of objects, as long as MetaTrader 5 can allocate memory in the operating system. The number of objects is not limited by structures or variables inside the EA.

You might think that only a crazy person can create such a structure. Ok then, you can call me crazy, because I created it, and it works. Furthermore, surprisingly it does not overload the system that much. People do call me crazy so it's not a big deal... Let's move further. You might notice a certain sluggishness when moving pending or limit orders. This is not because of a code failure or a problem with your computer or in MetaTrader 5, the problem is that the system will move pending orders or limits by moving them on the trading server itself, and there is a latency between the movement and the server's response. But this is the best and safer way to operate in some scenarios, which however doesn't allow us to do something other things that I want the EA to do. So, in the next articles we will fix this by adding a new functionality in the EA, we will also make the system more fluid, but this without modifying the structure above. We will only manipulate the code properly, although this makes the system less secure, but until then, who knows, I might find a good solution for this problem.

Let's take a look at some important points in the current new code. We will start with a little-studied function, its code looks like this:

```
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
{
#define def_IsBuy(A) ((A == ORDER_TYPE_BUY_LIMIT) || (A == ORDER_TYPE_BUY_STOP) || (A == ORDER_TYPE_BUY_STOP_LIMIT) || (A == ORDER_TYPE_BUY))

        ulong ticket;

        if (trans.symbol == Terminal.GetSymbol()) switch (trans.type)
        {
                case TRADE_TRANSACTION_DEAL_ADD:
                case TRADE_TRANSACTION_ORDER_ADD:
                        ticket = trans.order;
                        ticket = (ticket == 0 ? trans.position : ticket);
                        TradeView.IndicatorInfosAdd(ticket);
                        TradeView.UpdateInfosIndicators(0, ticket, trans.price, trans.price_tp, trans.price_sl, trans.volume, (trans.position > 0 ? trans.deal_type == DEAL_TYPE_BUY : def_IsBuy(trans.order_type)));
                        break;
                case TRADE_TRANSACTION_ORDER_DELETE:
                                if (trans.order != trans.position) TradeView.RemoveIndicator(trans.order);
                                else
                                        TradeView.UpdateInfosIndicators(0, trans.position, trans.price, trans.price_tp, trans.price_sl, trans.volume, trans.deal_type == DEAL_TYPE_BUY);
                                if (!PositionSelectByTicket(trans.position))
                                        TradeView.RemoveIndicator(trans.position);
                        break;
                case TRADE_TRANSACTION_ORDER_UPDATE:
                        TradeView.UpdateInfosIndicators(0, trans.order, trans.price, trans.price_tp, trans.price_sl, trans.volume, def_IsBuy(trans.order_type));
                        break;
                case TRADE_TRANSACTION_POSITION:
                        TradeView.UpdateInfosIndicators(0, trans.position, trans.price, trans.price_tp, trans.price_sl, trans.volume, trans.deal_type == DEAL_TYPE_BUY);
                        break;
        }


#undef def_IsBuy
}
```

This code is very interesting because it saves us from having to check every new position that appears or is being modified. Actually, the server itself will inform us about what is happening, so we only need to make sure the EA responds to events correctly. Study well this way of coding and using the [OnTradeTransaction](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction) event, because if I used the model to analyze things the way it is done in previous versions, we would spend a lot of time on the checks. In this case, the server does all the hard work for us, and we can know for sure that the values on the chart really show what the server sees at the moment.

Before we get to the highlights of the code above, let's take a look at another snippet.

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        Mouse.DispatchMessage(id, lparam, dparam, sparam);
        switch (id)
        {
                case CHARTEVENT_CHART_CHANGE:
                        Terminal.Resize();
                        WallPaper.Resize();
                        TimesAndTrade.Resize();
        break;
        }
        Chart.DispatchMessage(id, lparam, dparam, sparam);
        VolumeAtPrice.DispatchMessage(id, sparam);
        TradeView.DispatchMessage(id, lparam, dparam, sparam);
        ChartRedraw();
}
```

So, everything is done in one place. We can go into the class and see what's going inside.

### 3.1. The C\_IndicatorTradeView class

This class is used to present and manipulate data. It basically includes the old C\_OrderView and C\_TradeGraphics classes, as mentioned earlier. But it manipulates data in a totally different way. Let's take a look at some points in this class.

We'll start with the initialization function, the code of which looks like this:

```
void Initilize(void)
{
        int orders = OrdersTotal();
        ulong ticket;
        bool isBuy;
        long info;
        double tp, sl;

        ChartSetInteger(Terminal.Get_ID(), CHART_SHOW_OBJECT_DESCR, false);
        ChartSetInteger(Terminal.Get_ID(), CHART_SHOW_TRADE_LEVELS, false);
        ChartSetInteger(Terminal.Get_ID(), CHART_DRAG_TRADE_LEVELS, false);
        for (int c0 = 0; c0 <= orders; c0++) if ((ticket = OrderGetTicket(c0)) > 0) if (OrderGetString(ORDER_SYMBOL) == Terminal.GetSymbol())
        {
                info = OrderGetInteger(ORDER_TYPE);
                isBuy = ((info == ORDER_TYPE_BUY_LIMIT) || (info == ORDER_TYPE_BUY_STOP) || (info == ORDER_TYPE_BUY_STOP_LIMIT) || (info == ORDER_TYPE_BUY));
                IndicatorInfosAdd(ticket);
                UpdateInfosIndicators(-1, ticket, OrderGetDouble(ORDER_PRICE_OPEN), OrderGetDouble(ORDER_TP), OrderGetDouble(ORDER_SL), OrderGetDouble(ORDER_VOLUME_CURRENT), isBuy);
        }
        orders = PositionsTotal();
        for (int c0 = 0; c0 <= orders; c0++) if (PositionGetSymbol(c0) == Terminal.GetSymbol())
        {
                tp = PositionGetDouble(POSITION_TP);
                sl = PositionGetDouble(POSITION_SL);
                ticket = PositionGetInteger(POSITION_TICKET);
                IndicatorInfosAdd(ticket);
                UpdateInfosIndicators(1, ticket, PositionGetDouble(POSITION_PRICE_OPEN), tp, sl, PositionGetDouble(POSITION_VOLUME), PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
        }
        CreateIndicatorTrade(def_IndicatorTicket0, IT_PENDING);
        CreateIndicatorTrade(def_IndicatorTicket0, IT_TAKE);
        CreateIndicatorTrade(def_IndicatorTicket0, IT_STOP);
}
```

Basically, what we are doing is creating the necessary indicators to work, and presenting anything that currently exists in the account, such as positions, or pending orders. But the highlighted lines are important here, because if you are not using a cross-order system, you will have on the chart the order points (from MetaTrader 5), and if you click and drag these points, the EA will update the new points with the changes in the indicators. This doesn't get in the way too much, but we have to actually use the system we are developing, otherwise what is the point of developing it?

Next, pay attention to the following code:

```
void UpdateInfosIndicators(char test, ulong ticket, double pr, double tp, double sl, double vol, bool isBuy)
{
        bool isPending;

        isPending = (test > 0 ? false : (test < 0 ? true : (ticket == def_IndicatorTicket0 ? true : OrderSelect(ticket))));
        PositionAxlePrice(ticket, (isPending ? IT_RESULT : IT_PENDING), 0);
        PositionAxlePrice(ticket, (isPending ? IT_PENDING : IT_RESULT), pr);
        SetTextValue(ticket, (isPending ? IT_PENDING : IT_RESULT), vol);
        PositionAxlePrice(ticket, IT_TAKE, tp);
        PositionAxlePrice(ticket, IT_STOP, sl);
        SetTextValue(ticket, IT_TAKE, vol, (isBuy ? tp - pr : pr - tp));
        SetTextValue(ticket, IT_STOP, vol, (isBuy ? sl - pr : pr - sl));
}
```

It receives and updates data, presenting the correct values in terms of financial values and the points where the orders are located. Basically, we don't want to worry if there is a pending order or a position — the function will position it so that we can see it correctly on the chart.

Here is the next function.

```
inline double SecureChannelPosition(void)
{
        double Res = 0, sl, profit, bid, ask;
        ulong ticket;

        bid = SymbolInfoDouble(Terminal.GetSymbol(), SYMBOL_BID);
        ask = SymbolInfoDouble(Terminal.GetSymbol(), SYMBOL_ASK);
        for (int i0 = PositionsTotal() - 1; i0 >= 0; i0--) if (PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                ticket = PositionGetInteger(POSITION_TICKET);
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

It is called by the OnTick event, so it is quite critical in terms of speed and system load. The only thing it does except for checks is update the value on the chart, which is implemented by the highlighted code. Please note that the position ticket is very important here.

Let's take a closer look at the function highlighted above.

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
                        m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT), value0 / Terminal.GetVolumeMinimal(), def_ColorVolumeEdit);
                        break;
                case IT_TAKE    :
                case IT_STOP    :
                        finance = (value1 / Terminal.GetAdjustToTrade()) * value0;
                        m_EditInfo1.SetTextValue(MountName(ticket, it, EV_EDIT), finance);
                        break;
        }
}
```

This is how the correct values are displayed. But the question is how to move them? This is done by three other codes. Of course, you could avoid them and use the MetaTrader 5 system itself, which is much faster than the current EA system. But, as I said, I prefer using the EA, as it will receive other improvements soon.

The first function responsible for moving can be seen below, but it only shows the fragments that are necessary to move points, either the limits or the order itself, since the whole code is much more extensive and is not required to understand how to move using mouse movements.

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{
        ulong   ticket;

// ... Code ....

        switch (id)
        {
                case CHARTEVENT_MOUSE_MOVE:
                        Mouse.GetPositionDP(dt, price);
                        mKeys   = Mouse.GetButtonStatus();
                        bEClick  = (mKeys & 0x01) == 0x01;    //Left mouse click
                        bKeyBuy  = (mKeys & 0x04) == 0x04;    //SHIFT press
                        bKeySell = (mKeys & 0x08) == 0x08;    //CTRL press
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

// ... Code ...
                case CHARTEVENT_OBJECT_CLICK:
                        if (GetIndicatorInfos(sparam, ticket, price, it, ev)) switch (ev)
                        {

// ... Code ...

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
                        }
                        break;
        }
}
```

Let's try to understand what's going on. There is a video at the end of the article which shows how to do it and what will actually happen. But first let us try to figure out.

Each indication has an object which allows its selection (except for the result which cannot be moved). A click on this point will change the indication line — it will get thicker. When this happens, the mouse movements will be captured and converted into a new position for this object, until we give a new click outside the selection object that allows the movement of the object. See that it is not necessary to keep holding the mouse button, just click once, drag, and then click again.

But in reality, only part of the work is done here. There are two other functions to help us. One has already been seen above, which is responsible for showing the calculated values; the next is responsible for being the stone that makes the EA looks like a slug when using the order or limit level movement system: It is shown below:

```
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
```

I call it the function a stone because it is responsible for **_making the positioning system to be slow_**. If you do not understand, take a look at the highlighted points. Each of them is a function which is inside the C\_Router class and which will send a request to the trading server, so if the server for one reason or another takes a while to respond (and this will always happen because of latency), the positioning system will be more or less slow, but if the server responds quickly, the system will be fluid, or rather, the thing will go more smoothly. Later we will modify this, because this system does not allow us to do something else. Anyway, you have to keep in mind that this way you will be operating in a little safer way, especially those who like to operate in highly volatile movements, where prices can move very quickly, but even so you run the risk of having the limits jump. There is no way, something has to be sacrificed. For those who agree to operate knowing this is exactly what will be inside the server, the EA is ready at this point. But for those who want to gain fluidity in the EA's functioning, even at the cost of not being accurate to the points, in the next articles things will change and will become more interesting.

The video below shows how the system actually works, so pay attention to the values ​​on the chart and in the toolbox.

Demonstração de ordens Pendentes - YouTube

Tap to unmute

[Demonstração de ordens Pendentes](https://www.youtube.com/watch?v=M7s4yyfejWs) [MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ)

MQL5.community1.91K subscribers

[Watch on](https://www.youtube.com/watch?v=M7s4yyfejWs)

### Conclusion

Although we now have a very interesting Expert Advisor for trading, I advise you to use it on a demo account for a while to get used to how it works. I promise that there will be no more big changes in the way it works. There will only be improvements, and in the next article we will add some things that this EA lacks, at the expense of some security that it provides. Anyway, this will be a great source for learning how the trading system works and how to manipulate the platform to get whatever type of data modeling we need.

Don't forget, if you find that moving the orders or limit levels is too slow, you can remove the points I showed in the article and use MetaTrader 5 itself to move orders or limits and use an EA as a support to help interpret the data. Whether you do it or not is your choice...

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10516](https://www.mql5.com/pt/articles/10516)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10516.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Sistema\_de\_ordens\_z\_V\_f.zip](https://www.mql5.com/en/articles/download/10516/ea_-_sistema_de_ordens_z_v_f.zip "Download EA_-_Sistema_de_ordens_z_V_f.zip")(12028.77 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/433971)**
(1)


![Anatoli Kazharski](https://c.mql5.com/avatar/2022/1/61D72F6B-7C12.jpg)

**[Anatoli Kazharski](https://www.mql5.com/en/users/tol64)**
\|
27 Sep 2022 at 16:56

They forgot that the article should have screenshots in English only. That's always been the rule.

![DoEasy. Controls (Part 11): WinForms objects — groups, CheckedListBox WinForms object](https://c.mql5.com/2/47/MQL5-avatar-doeasy-library-2__5.png)[DoEasy. Controls (Part 11): WinForms objects — groups, CheckedListBox WinForms object](https://www.mql5.com/en/articles/11194)

The article considers grouping WinForms objects and creation of the CheckBox objects list object.

![DoEasy. Controls (Part 10): WinForms objects — Animating the interface](https://c.mql5.com/2/47/MQL5-avatar-doeasy-library-2__4.png)[DoEasy. Controls (Part 10): WinForms objects — Animating the interface](https://www.mql5.com/en/articles/11173)

It is time to animate the graphical interface by implementing the functionality for object interaction with users and objects. The new functionality will also be necessary to let more complex objects work correctly.

![Developing a trading Expert Advisor from scratch (Part 23): New order system (VI)](https://c.mql5.com/2/47/development__6.png)[Developing a trading Expert Advisor from scratch (Part 23): New order system (VI)](https://www.mql5.com/en/articles/10563)

We will make the order system more flexible. Here we will consider changes to the code that will make it more flexible, which will allow us to change position stop levels much faster.

![Risk and capital management using Expert Advisors](https://c.mql5.com/2/49/Risk-and-capital-management-using-Expert-Advisors.png)[Risk and capital management using Expert Advisors](https://www.mql5.com/en/articles/11500)

This article is about what you can not see in a backtest report, what you should expect using automated trading software, how to manage your money if you are using expert advisors, and how to cover a significant loss to remain in the trading activity when you are using automated procedures.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mtolmedyfnvggjparudjmzmalvzdnvii&ssn=1769104015747034301&ssn_dr=0&ssn_sr=0&fv_date=1769104015&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10516&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20(Part%2022)%3A%20New%20order%20system%20(V)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176910401546536856&fz_uniq=5051679858901243022&sv=2552)

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