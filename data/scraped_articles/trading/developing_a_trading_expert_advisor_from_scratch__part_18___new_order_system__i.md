---
title: Developing a trading Expert Advisor from scratch (Part 18): New order system (I)
url: https://www.mql5.com/en/articles/10462
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:47:36.784252
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/10462&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051687503943029942)

MetaTrader 5 / Trading


### Introduction

Since we started documenting this EA in this series from the very first article [Developing a trading Expert Advisor from scratch](https://www.mql5.com/en/articles/10085), it has undergone various changes and improvements while maintaining the same on-chart order system model. It is very simple and functional. However, in many situations, it is not suitable for real trading.

Of course, we could add some things to the original system so that we have some information about orders, both open and pending. But this would turn our code into a Frankenstein which would ultimately make a real nightmare out of the improvement process. Even if you have your own methodology, it will be lost over time as the code gets too big and too complicated.

So, we need to create a completely different system in terms of the order model we use. At the same time, it should be easy for the trader to understand, while providing all the information we need to work safely.

### Important Note

The system that I am describing in this article is designed for netting accounts [ACCOUNT\_MARGIN\_MODE\_RETAIL\_NETTING](https://www.mql5.com/en/docs/constants/environment_state/accountinformation), which allows having only one open position per symbol. If you are using a hedging account [ACCOUNT\_MARGIN\_MODE\_RETAIL\_HEDGING](https://www.mql5.com/en/docs/constants/environment_state/accountinformation), the article will provide nothing to your EA as in this case you can have as many positions as you need, and they will not interfere with each other. So, you can simply reset and delete all modifications form the final code.

The ACCOUNT\_MARGIN\_MODE\_RETAIL\_HEDGING mode is most often used when you want an EA to run automatically, opening and closing positions without your participation, and at the same time you continue trading manually. Furthermore, you trade the same asset which your EA trades. With hedging, your EA's activities will not affect any of your trades, and thus you will have independent positions.

For this reason, I will highlight all code parts being added or modified. We will implement all changes and additions slowly and gradually, so if you need to remove anything from your code, you will be able to easily find the relevant code parts.

Although the modifications can be removed, there is a part where I am checking the system. Even if you save the changes that we will consider here, you can use this EA on any type of trading account, NETTING and HEDGING, because the Expert Advisor will check for the particular model and adjust accordingly.

### 1.0. Planning

In fact, the first thing to do is to understand what happens to orders which are added to the system and filled as they reach the appropriate prices. Many traders may not know this, or rather, that they never thought about it and therefore did not perform any tests in order to understand this idea and then to implement an adequate yet reliable system.

To understand what actually happens, let us analyze a simple example: you have an open position, let's say Buy, for a certain asset with the initial volume of 1 lot. Then you place a new 2-lot Buy order at a slightly higher price. So far so good, nothing special. But as soon as these 2 lots are bought, something will happen, which is where the problem lies.

Once these two lots are bought, you will have 3 lots, but the system will update your initial price and will set it to the average. Now this seems clear, and everyone understands this. But what happens to stop loss and take profit levels of the new position?

Many traders do not know the answer, but they better think about it. If you are using an OCO order system (One Cancel The Other) in all deals, you can notice something interesting every time you open or close a position with a part of the total traded volume.

In the [article about Cross Orders](https://www.mql5.com/en/articles/10383), I presented a way to place take profit and stop loss directly on the chart, without using the standard MetaTrader 5 system. Indeed, this method is almost identical to the MetaTrader 5 system, since its functionality is very close to that of the platform, but with the proper proportions. However, after doing some testing, I found that when we have an open OCO order and a pending order, the OCO order is also captured by the trading system because the price reached the value specified in the order. In addition to a new average price, we also have a change in take profit and stop loss values — they are changed to those specified in the last captured OCO order. So, depending on how it is configured, the EA will close it immediately after the trading system reports new take profit and stop loss values.

This happens due to the following check implemented in the EA:

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

The highlighted lines perform this check and close an order in case the price exits the channel which is formed by the take profit and stop loss values.

It is important to know this because we are not talking about an EA failure, a trading system issue or defect. The real problem is that most of the time we do not give due attention to what is happening and ignore this kind until it is no longer possible to turn a blind eye to this fact.

If you trade in the way that does not average the price, you won't see this sort of thing happen. But there is a very wide range of operations where an average price is appropriate and even necessary. So, in these cases, if you're running the system without knowing the above details, you can exit the position earlier than you would like, even if you have previously changed the OCO order of the open position accordingly. As soon as a pending OCO order is captured, the threshold values ( _every time I talk about threshold I refer to the previous take profit and stop loss_) will be replaced with those specified in the recently captured pending OCO order.

There is a way to fix or to avoid it: do not use OCO orders, at least when you already have an open position. All other orders passed into the system must be of simple type, without setting take profit and stop loss.

Basically, that's it. But when an EA is on the chart, it's there to help us, making our lives easier. So, it wouldn't make sense to program an Expert Advisor if you are not able to use it later.

### 2.0. Implementing a new system

We need to make small changes to the code to implement the system and ensure that it works the way we expect — the EA should help us and assist in avoiding errors.

These is not very difficult, but these changes guarantee that we will never run the risk of having the OCO order coming in at an unwanted moment, causing real confusion in our operations.

Let us start with the following changes:

### 2.0.1. Modification of the C\_Router class

The C\_Router class is responsible for parsing and sending orders to us. Let's add a private variable to it. When an open position is found for the asset which is traded by the EA, this variable will store the relevant information. Every time the EA wants to know if there is an open position, it will tell us about it.

This implementation is shown in the code below. However, this code will further be changed in the article. I just want to show all changes step by step, so that you understand the how the modification process actually looked like.

```
//+------------------------------------------------------------------+
inline bool ExistPosition(void) const { return m_bContainsPosition; }
//+------------------------------------------------------------------+
void UpdatePosition(void)
{
        static int memPositions = 0, memOrder = 0;
        ulong ul;
        int p, o;

        p = PositionsTotal();
        o = OrdersTotal();
        if ((memPositions != p) || (memOrder != o))
        {
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
                RemoveAllsLines();
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
                memOrder = o;
                memPositions = p;
                m_bContainsPosition = false;
        };
        for(int i0 = p; i0 >= 0; i0--) if(PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                ul = PositionGetInteger(POSITION_TICKET);
                m_bContainsPosition = true;
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
//+------------------------------------------------------------------+
```

The highlighted lines show the required additions to run all checks which will be added later in certain places of the EA code.

In a way, we could implement ell checks and adjustments just in the C\_Router class but this will not be enough. I will explain this later. For now, let us continue with the modifications. After creating the above check, let's add a constructor to properly initialize the newly added variable.

```
C_Router() : m_bContainsPosition(false) {}
```

Now we edit the function that places pending orders as follows:

```
ulong CreateOrderPendent(const bool IsBuy, const double Volume, const double Price, const double Take, const double Stop, const bool DayTrade = true)
{
        double last = SymbolInfoDouble(Terminal.GetSymbol(), SYMBOL_LAST);

        ZeroMemory(TradeRequest);
        ZeroMemory(TradeResult);
        TradeRequest.action             = TRADE_ACTION_PENDING;
        TradeRequest.symbol             = Terminal.GetSymbol();
        TradeRequest.volume             = Volume;
        TradeRequest.type               = (IsBuy ? (last >= Price ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_BUY_STOP) : (last < Price ? ORDER_TYPE_SELL_LIMIT : ORDER_TYPE_SELL_STOP));
        TradeRequest.price              = NormalizeDouble(Price, Terminal.GetDigits());
        TradeRequest.sl                 = NormalizeDouble((m_bContainsPosition ? 0 : Stop), Terminal.GetDigits());
        TradeRequest.tp                 = NormalizeDouble((m_bContainsPosition ? 0 : Take), Terminal.GetDigits());
        TradeRequest.type_time          = (DayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
        TradeRequest.stoplimit          = 0;
        TradeRequest.expiration         = 0;
        TradeRequest.type_filling       = ORDER_FILLING_RETURN;
        TradeRequest.deviation          = 1000;
        TradeRequest.comment            = "Order Generated by Experts Advisor.";
        if (!Send()) return 0;

        return TradeResult.order;
};
```

The highlighted parts are the changes to be implemented.

Now let's get back to the updated code to make a new modification. Remember that it is called by the OnTrade function and is called every time the orders are changed. This can be seen in the code below:

```
void UpdatePosition(void)
{
        static int memPositions = 0, memOrder = 0;
        ulong ul;
        int p, o;

        p = PositionsTotal();
        o = OrdersTotal();
        if ((memPositions != p) || (memOrder != o))
        {
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
                RemoveAllsLines();
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
                memOrder = o;
                memPositions = p;
                m_bContainsPosition = false;
        };
        for(int i0 = p; i0 >= 0; i0--) if(PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                ul = PositionGetInteger(POSITION_TICKET);
                m_bContainsPosition = true;
                SetLineOrder(ul, PositionGetDouble(POSITION_PRICE_OPEN), HL_PRICE, false);
                SetLineOrder(ul, PositionGetDouble(POSITION_TP), HL_TAKE, true);
                SetLineOrder(ul, PositionGetDouble(POSITION_SL), HL_STOP, true);
        }
        for (int i0 = o; i0 >= 0; i0--) if ((ul = OrderGetTicket(i0)) > 0) if (OrderGetString(ORDER_SYMBOL) == Terminal.GetSymbol())
        {
                if (m_bContainsPosition)
                {
                        ModifyOrderPendent(ul, OrderGetDouble(ORDER_PRICE_OPEN), 0, 0);
                        (OrderSelect(ul) ? 0 : 0);
                }
                SetLineOrder(ul, OrderGetDouble(ORDER_PRICE_OPEN), HL_PRICE, true);
                SetLineOrder(ul, OrderGetDouble(ORDER_TP), HL_TAKE, true);
                SetLineOrder(ul, OrderGetDouble(ORDER_SL), HL_STOP, true);
        }
};
```

Now we need to make sure that the user does not convert a simple pending order into a pending OCO order when there is already an open position, i.e. if the user opens the toolbox and tries to edit the take profit or stop loss. When the user tries to do this, the trade server will inform us through the OnTrade function, so the EA will immediately know about the change and will cancel the change made by the user, ensuring the reliability of the system.

But there is one more thing that also needs to be modified; it concerns market orders. This is a very simple change as there it does not require any modifications related to checks. The new function code is shown below:

```
ulong ExecuteOrderInMarket(const bool IsBuy, const double Volume, const double Price, const double Take, const double Stop, const bool DayTrade = true)
{
        ZeroMemory(TradeRequest);
        ZeroMemory(TradeResult);
        TradeRequest.action             = TRADE_ACTION_DEAL;
        TradeRequest.symbol             = Terminal.GetSymbol();
        TradeRequest.volume             = Volume;
        TradeRequest.type               = (IsBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
        TradeRequest.price              = NormalizeDouble(Price, Terminal.GetDigits());
        TradeRequest.sl                 = NormalizeDouble((m_bContainsPosition ? 0 : Stop), Terminal.GetDigits());
        TradeRequest.tp                 = NormalizeDouble((m_bContainsPosition ? 0 : Take), Terminal.GetDigits());
        TradeRequest.type_time          = (DayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
        TradeRequest.stoplimit          = 0;
        TradeRequest.expiration         = 0;
        TradeRequest.type_filling       = ORDER_FILLING_RETURN;
        TradeRequest.deviation          = 1000;
        TradeRequest.comment            = "[ Order Market ] Generated by Experts Advisor.";
        if (!Send()) return 0;

        return TradeResult.order;
};
```

Although it may seem strange, these changes already provide a sufficient level of security (at least an acceptable level), which allows not to miss an OCO, pending or market order when there is already an open position for the same asset that the EA is trading. Thus, the EA will actually take care of the order submission system.

Well, it's all too beautiful and too wonderful to be true, isn't it? You might think that this will already guarantee you a good margin of safety, but it is not quite so. These modifications guarantee that an OCO order does not remain pending or does not enter the market when we have an open position. But there is a fatal flaw in these modifications, and, if not properly corrected, this flaw can give you a huge headache and a tremendous loss, which can break your account or the position closed by the broker for lack of margin.

Note that there is no check of whether a pending order is within an open position limits or not, and this is very dangerous because under the current state of the system, when you add a pending OCO order outside the open position limits, the EA will not allow that order to be of the OCO type. In other words: the order will have no limits, this order will be an order without take profit or stop loss, so when you close a position and enter this pending order, you will have to adjust these levels as quickly as possible. If you forget to do this, then you risk having an open position with no limits.

To set the levels, you would have to open the message window, open the order and edit the level values. This issue will be fixed soon. Now, let's fix the current problem.

Therefore, we need to change the way the EA works with pending orders, because if the user wants to create an order without the levels, the EA will treat it as something normal, but if the user ends up creating a limit order, the EA will have to adjust the order accordingly: setting limits if the order is placed outside the open position, or removing the limits of a pending order if the order is placed within the open position.

### 2.0.2. Work within limits

The first thing we're going to do is create verification limits. It is very simple to do. However, it requires a lot of attention to detail, because there are two possible cases which can be extended to more cases. To understand what to do, these two cases are enough.

![](https://c.mql5.com/2/45/001__2.png)

The first case is shown above. It will have a pending order outside the limits (limit in this case is an area in the gradient, i.e we have a position, either buy or sell, and we have an upper limit. When the price reaches or exceeds this limit, the position will be closed. In this case, the pending order can be configured by the user as an OCO order and the EA must accept the way the order is configured by the user, be it a simple order or an OCO order — the EA should not interfere in this case.

The second case is shown below. Here the pending order is within the area limited by the open position. In this case the EA must remove the limits that may be configured by the user.

![](https://c.mql5.com/2/45/002__3.png)

Pay attention that no matter how far the limit goes, whether we buy or sell, of the order enters this area the EA must remove limit values form the pending order. But if it leaves this area, the EA must leave the order the way it is configured by the user.

After defining this, we need to create some variables shown below:

```
class C_Router : public C_HLineTrade
{
        protected:
                MqlTradeRequest TradeRequest;
                MqlTradeResult  TradeResult;
        private  :
                bool            m_bContainsPosition;
                struct st00
                {
                        double  TakeProfit,
                                StopLoss;
                        bool    IsBuy;
                }m_Limits;

// ... Rest of the code
```

Now we have a way to check the limits at the stage when the OnTrade order triggering event occurs. So, once again we modify the update function of the C\_Router class.

```
// Rest of the code....

//+------------------------------------------------------------------+
#define macro_MAX(A, B) (A > B ? A : B)
#define macro_MIN(A, B) (A < B ? A : B)
void UpdatePosition(void)
{
        static int      memPositions = 0, memOrder = 0;
        ulong           ul;
        int             p, o;
        double          price;
        bool            bTest;

        p = PositionsTotal();
        o = OrdersTotal();
        if ((memPositions != p) || (memOrder != o))
        {
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
                RemoveAllsLines();
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
                memOrder = o;
                memPositions = p;
                m_bContainsPosition = false;
                m_Limits.StopLoss = -1;
                m_Limits.TakeProfit = -1;
                m_Limits.IsBuy = false;
        };
        for(int i0 = p; i0 >= 0; i0--) if(PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                ul = PositionGetInteger(POSITION_TICKET);
                m_bContainsPosition = true;
                SetLineOrder(ul, PositionGetDouble(POSITION_PRICE_OPEN), HL_PRICE, false);
                SetLineOrder(ul, m_Limits.TakeProfit = PositionGetDouble(POSITION_TP), HL_TAKE, true);
                SetLineOrder(ul, m_Limits.StopLoss = PositionGetDouble(POSITION_SL), HL_STOP, true);
                m_Limits.IsBuy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
        }
        for (int i0 = o; i0 >= 0; i0--) if ((ul = OrderGetTicket(i0)) > 0) if (OrderGetString(ORDER_SYMBOL) == Terminal.GetSymbol())
        {
                price = OrderGetDouble(ORDER_PRICE_OPEN);
                if ((m_Limits.StopLoss == -1) && (m_Limits.TakeProfit == -1)) bTest = false; else
                {
                        bTest = ((!m_Limits.IsBuy) && (m_Limits.StopLoss > price));
                        bTest = (bTest ? bTest : (m_Limits.IsBuy) && (m_Limits.StopLoss < price));
                        bTest = (bTest ? bTest : ((macro_MAX(m_Limits.TakeProfit, m_Limits.StopLoss) > price) && (macro_MIN(m_Limits.TakeProfit, m_Limits.StopLoss) < price)));
                }
                if ((m_bContainsPosition) && (bTest))
                {
                        ModifyOrderPendent(ul, price, 0, 0);
                        (OrderSelect(ul) ? 0 : 0);
                }
                SetLineOrder(ul, price, HL_PRICE, true);
                SetLineOrder(ul, OrderGetDouble(ORDER_TP), HL_TAKE, true);
                SetLineOrder(ul, OrderGetDouble(ORDER_SL), HL_STOP, true);
        }
};
#undef macro_MAX
#undef macro_MIN
//+------------------------------------------------------------------+

// ... The rest of the code...
```

Now the class will handle pending positions in order to differentiate when they are within an area that cannot have OCO pending orders, or outside it. Note that this function will be called with each status change in the order system. The first thing it will do is initialize the variables properly.

```
m_Limits.StopLoss = -1;
m_Limits.TakeProfit = -1;
m_Limits.IsBuy = false;
```

Once this is done, we will check whether or not there is an open position. This can be done at any time. As soon as we have an open position, it will delimit the region where it will not be possible to have OCO pending orders — this is achieved at this point.

```
SetLineOrder(ul, m_Limits.TakeProfit = PositionGetDouble(POSITION_TP), HL_TAKE, true);
SetLineOrder(ul, m_Limits.StopLoss = PositionGetDouble(POSITION_SL), HL_STOP, true);
m_Limits.IsBuy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
```

Now we can check each of the pending orders to find out whether or not they are within the limit area. An important note: here we need to know if the order is buy or sell, since we may not have a take profit but have a stop loss. So, in this case it is necessary to know the type of the position. To understand this, see the figures below:

![](https://c.mql5.com/2/45/003__5.png)

If it is a sell position, the stop loss marks the upper limit...

![](https://c.mql5.com/2/45/004__2.png)

For buy positions, the stop loss is a lower limit.

In other words, in some cases pending orders can be of the OCO type if they are placed the maximum. In other cases, they should be placed below the minimum. But there can also be another case where the pending orders can also be the OCO type, as shown below:

![](https://c.mql5.com/2/45/005__2.png)

Pending orders outside the limits, which is a typical case.

To check this, we use the following fragment

```
price = OrderGetDouble(ORDER_PRICE_OPEN);
if ((m_Limits.StopLoss == -1) && (m_Limits.TakeProfit == -1)) bTest = false; else
{
        bTest = ((!m_Limits.IsBuy) && (m_Limits.StopLoss > price));
        bTest = (bTest ? bTest : (m_Limits.IsBuy) && (m_Limits.StopLoss < price));
        bTest = (bTest ? bTest : ((macro_MAX(m_Limits.TakeProfit, m_Limits.StopLoss) > price) && (macro_MIN(m_Limits.TakeProfit, m_Limits.StopLoss) < price)));
}
```

It will check if there are any open positions. If not, the EA will have to respect the order settings specified by the user. If a position if found, the following check will be performed in the following order:

1. If we are short, then the price at which a pending order will be placed must be greater than the stop loss value of the open position;
2. If we are long, then the price at which a pending order will be placed must be lower than the stop loss value of the open position;
3. If the system is still accepting the order as an OCO type, we will do one last test to see if the order is out of position.

Once this is done, we will be sure that the pending order can be left or not the way the user configured it, and life goes on... But there is one last addition before we move on to the next step. Actually, it is the last check, that I mentioned at the beginning of the article, and this is in the fragment below:

```
void UpdatePosition(void)
{

// ... Internal code...

        for(int i0 = p; i0 >= 0; i0--) if(PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                ul = PositionGetInteger(POSITION_TICKET);
                m_bContainsPosition = true;
                SetLineOrder(ul, PositionGetDouble(POSITION_PRICE_OPEN), HL_PRICE, false);
                SetLineOrder(ul, m_Limits.TakeProfit = PositionGetDouble(POSITION_TP), HL_TAKE, true);
                SetLineOrder(ul, m_Limits.StopLoss = PositionGetDouble(POSITION_SL), HL_STOP, true);
                m_Limits.IsBuy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
        }
        if (AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
                m_bContainsPosition = false;
        for (int i0 = o; i0 >= 0; i0--) if ((ul = OrderGetTicket(i0)) > 0) if (OrderGetString(ORDER_SYMBOL) == Terminal.GetSymbol())
        {
                price = OrderGetDouble(ORDER_PRICE_OPEN);
                if (m_bContainsPosition)
                {
                        if ((m_Limits.StopLoss == -1) && (m_Limits.TakeProfit == -1)) bTest = false; else
                        {
                                bTest = ((!m_Limits.IsBuy) && (m_Limits.StopLoss > price));
                                bTest = (bTest ? bTest : (m_Limits.IsBuy) && (m_Limits.StopLoss < price));
                                bTest = (bTest ? bTest : ((macro_MAX(m_Limits.TakeProfit, m_Limits.StopLoss) > price) && (macro_MIN(m_Limits.TakeProfit, m_Limits.StopLoss) < price)));
                        }
                        if (bTest)
                        {
                                ModifyOrderPendent(ul, price, 0, 0);
                                (OrderSelect(ul) ? 0 : 0);
                        }
                }
                SetLineOrder(ul, price, HL_PRICE, true);
                SetLineOrder(ul, OrderGetDouble(ORDER_TP), HL_TAKE, true);
                SetLineOrder(ul, OrderGetDouble(ORDER_SL), HL_STOP, true);
        }
};
```

In the highlighted part, we check if the account type is HEDGING. If it is, even if the variable indicated that we would have to work with limits, at this point it will indicate that we do not need to work with them. Therefore, the EA will ignore any restrictions that may arise and will treat orders the way they were configured by the user. This is a very simple check, but it needs to be done at this stage to ensure that our entire system will function properly.

### 2.1.0. Adjusting the positioning system

While most of the issues have been solved by adjusting the C\_Router object class, we still don't have an adequate system. There is another issue to be solved: correcting the system of positioning via mouse, which is another equally important step. It has several implications, since we have to define how the system present in the C\_OrderView class should work.

The big question, which on how you actually want to operate, is whether or not the C\_OrderView class will create limits for pending orders when they leave the limits of an open position.

Although it is tempting to do this every time, there are things that must be taken into account when making this decision. But let's go by parts as Jack the Ripper would say. Basically, the only real change that we will actually have to make in the C\_OrderView class is shown in code below:

```
inline void MoveTo(uint Key)
{
        static double local = 0;
        datetime dt;
        bool    bEClick, bKeyBuy, bKeySell, bCheck;
        double  take = 0, stop = 0, price;

        bEClick  = (Key & 0x01) == 0x01;    //Left click
        bKeyBuy  = (Key & 0x04) == 0x04;    //SHIFT pressed
        bKeySell = (Key & 0x08) == 0x08;    //CTRL pressed
        Mouse.GetPositionDP(dt, price);
        if (bKeyBuy != bKeySell)
        {
                Mouse.Hide();
                bCheck = CheckLimits(price);
        } else Mouse.Show();
        ObjectMove(Terminal.Get_ID(), m_Infos.szHLinePrice, 0, 0, price = (bKeyBuy != bKeySell ? price : 0));
        ObjectMove(Terminal.Get_ID(), m_Infos.szHLineTake, 0, 0, take = (bCheck ? 0 : price + (m_Infos.TakeProfit * (bKeyBuy ? 1 : -1))));
        ObjectMove(Terminal.Get_ID(), m_Infos.szHLineStop, 0, 0, stop = (bCheck ? 0 : price + (m_Infos.StopLoss * (bKeyBuy ? -1 : 1))));
        if((bEClick) && (bKeyBuy != bKeySell) && (local == 0)) CreateOrderPendent(bKeyBuy, m_Infos.Volume, local = price, take, stop, m_Infos.IsDayTrade);
        local = (local != price ? 0 : local);
        ObjectSetInteger(Terminal.Get_ID(), m_Infos.szHLinePrice, OBJPROP_COLOR, (bKeyBuy != bKeySell ? m_Infos.cPrice : clrNONE));
        ObjectSetInteger(Terminal.Get_ID(), m_Infos.szHLineTake, OBJPROP_COLOR, (take > 0 ? m_Infos.cTake : clrNONE));
        ObjectSetInteger(Terminal.Get_ID(), m_Infos.szHLineStop, OBJPROP_COLOR, (stop > 0 ? m_Infos.cStop : clrNONE));
};
```

Is that all? Yes, that's all we need to do. All the rest of the logic is inside the C\_Router class. What hasn't been modified is executed by MetaTrader 5's own messaging system, because when there is a change in the list of orders (pending or positions), the OnTrade routine will be called. When this happens, it will trigger the update routine from within the C\_Router class that will make the necessary adjustments. But there is a code that appears in this routine and can make you go crazy looking for where it is. In fact, it is inside the C\_Router class, it is seen below:

```
#define macro_MAX(A, B) (A > B ? A : B)
#define macro_MIN(A, B) (A < B ? A : B)
inline bool CheckLimits(const double price)
{
        bool bTest = false;

        if ((!m_bContainsPosition) || ((m_Limits.StopLoss == -1) && (m_Limits.TakeProfit == -1))) return bTest;
        bTest = ((macro_MAX(m_Limits.TakeProfit, m_Limits.StopLoss) > price) && (macro_MIN(m_Limits.TakeProfit, m_Limits.StopLoss) < price));
        if (m_Limits.TakeProfit == 0)
        {
                bTest = (bTest ? bTest : (!m_Limits.IsBuy) && (m_Limits.StopLoss > price));
                bTest = (bTest ? bTest : (m_Limits.IsBuy) && (m_Limits.StopLoss < price));
        }
        return bTest;
};
#undef macro_MAX
#undef macro_MIN
```

This code was inside the C\_Router class update function. It has been removed from there and replaced with a call...

### 2.2.0. To limit or not to limit, that is the question

Our work is almost finished, but the last question remains to be solved, which is perhaps the most difficult at the moment. If you have followed and understood the content up to this point, you may have noticed that the system works very well, but whenever a pending order configured as OCO enters the limits of an open position, the order loses the limits that were configured for it. This will always happen.

But if by chance the trader changes open position limits, or if an order that was an OCO and is now a simple order goes beyond those limits, it will still be a simple order. So, we have a potential problem.

Another big detail: how should the EA proceed? Should it notify the trader that a simple order has just appeared for the asset? Or should it simply set limits for the order and turn it into OCO?

This question is extremely relevant if you really want the EA to help you in trading. It would be good for the EA to issue a warning informing us of what has happened. But if you are in an asset, at a time of high volatility it is also good to have the EA automatically create some limit for the order so that it do not stay there loose, which can cause great damage even before we realize what is happening.

So, to solve this problem, the system has gone through one last modification, but as I explained above, you should seriously think about how to actually deal with this problem. Below is how I implemented a possible solution.

First, I added a new variable through which the trader can inform the EA which procedure will be performed. It is shown below:

```
// ... Code ...

input group "Chart Trader"
input int       user20   = 1;              //Leverage factor
input int       user21   = 100;            //Take Profit (financial)
input int       user22   = 75;             //Stop Loss (financial)
input color     user23   = clrBlue;        //Price line color
input color     user24   = clrForestGreen; //Take Profit line color
input color     user25   = clrFireBrick;   //Stop Loss line color
input bool      user26   = true;           //Day Trade?
input bool      user27   = true;           //Always set loose order limits

// ... Rest of the code...

void OnTrade()
{
        Chart.DispatchMessage(CHARTEVENT_CHART_CHANGE, 0, OrderView.UpdateRoof(), C_Chart_IDE::szMsgIDE[C_Chart_IDE::eROOF_DIARY]);
        OrderView.UpdatePosition(user27);
}

// ... Rest of the code...
```

Now we need to go back to the C\_Router class and add 3 new functions to it. They can be seen below:

```
//+------------------------------------------------------------------+
void SetFinance(const int Contracts, const int Take, const int Stop)
{
        m_Limits.Contract = Contracts;
        m_Limits.FinanceTake = Take;
        m_Limits.FinanceStop = Stop;
}
//+------------------------------------------------------------------+
inline double GetDisplacementTake(const bool IsBuy, const double Vol) const
{
        return (Terminal.AdjustPrice(m_Limits.FinanceTake * (Vol / m_Limits.Contract) * Terminal.GetAdjustToTrade() / Vol) * (IsBuy ? 1 : -1));
}
//+------------------------------------------------------------------+
inline double GetDisplacementStop(const bool IsBuy, const double Vol) const
{
        return (Terminal.AdjustPrice(m_Limits.FinanceStop * (Vol / m_Limits.Contract) * Terminal.GetAdjustToTrade() / Vol) * (IsBuy ? -1 : 1));
}
//+------------------------------------------------------------------+
```

The code will maintain the values that are informed in the Chart Trader, as can be seen in the next image, but will also proportionally correct the value that we should use as limits in OCO pending orders.

![](https://c.mql5.com/2/45/006__2.png)

That is, we already have where to get the values that we are going to use so that the EA can minimally configure an OCO order when a pending order is triggered. However, as you may suspect, we will have to make a new change to the C\_Router class update code. The change is shown below:

```
void UpdatePosition(bool bAdjust)
{
        static int      memPositions = 0, memOrder = 0;
        ulong           ul;
        int             p, o;
        long            info;
        double          price, stop, take, vol;
        bool            bIsBuy, bTest;

        p = PositionsTotal();
        o = OrdersTotal();
        if ((memPositions != p) || (memOrder != o))
        {
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
                RemoveAllsLines();
                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
                memOrder = o;
                memPositions = p;
                m_bContainsPosition = false;
                m_Limits.StopLoss = -1;
                m_Limits.TakeProfit = -1;
                m_Limits.IsBuy = false;
        };
        for(int i0 = p; i0 >= 0; i0--) if(PositionGetSymbol(i0) == Terminal.GetSymbol())
        {
                ul = PositionGetInteger(POSITION_TICKET);
                m_bContainsPosition = true;
                SetLineOrder(ul, PositionGetDouble(POSITION_PRICE_OPEN), HL_PRICE, false);
                SetLineOrder(ul, take = PositionGetDouble(POSITION_TP), HL_TAKE, true);
                SetLineOrder(ul, stop = PositionGetDouble(POSITION_SL), HL_STOP, true);
                m_Limits.IsBuy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
                m_Limits.TakeProfit = (m_Limits.TakeProfit < 0 ? take : (m_Limits.IsBuy ? (m_Limits.TakeProfit > take ? m_Limits.TakeProfit : take) : (take > m_Limits.TakeProfit ? m_Limits.TakeProfit : take)));
                m_Limits.StopLoss = (m_Limits.StopLoss < 0 ? stop : (m_Limits.IsBuy ? (m_Limits.StopLoss < stop ? m_Limits.StopLoss : stop) : (stop < m_Limits.StopLoss ? m_Limits.StopLoss : stop)));
        }
        if ((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
                m_bContainsPosition = false;
        for (int i0 = o; i0 >= 0; i0--) if ((ul = OrderGetTicket(i0)) > 0) if (OrderGetString(ORDER_SYMBOL) == Terminal.GetSymbol())
        {
                price = OrderGetDouble(ORDER_PRICE_OPEN);
                take = OrderGetDouble(ORDER_TP);
                stop = OrderGetDouble(ORDER_SL);
		bTest = CheckLimits(price);
                if ((take == 0) && (stop == 0) && (bAdjust) && (!bTest))
                {
                        info = OrderGetInteger(ORDER_TYPE);
                        vol = OrderGetDouble(ORDER_VOLUME_CURRENT);
                        bIsBuy = ((info == ORDER_TYPE_BUY_LIMIT) || (info == ORDER_TYPE_BUY_STOP) || (info == ORDER_TYPE_BUY_STOP_LIMIT) || (info == ORDER_TYPE_BUY));
                        take = price + GetDisplacementTake(bIsBuy, vol);
                        stop = price + GetDisplacementStop(bIsBuy, vol);
                        ModifyOrderPendent(ul, price, take, stop);
                }
                if ((take != 0) && (stop != 0) && (bTest))
                        ModifyOrderPendent(ul, price, take = 0, stop = 0);
                SetLineOrder(ul, price, HL_PRICE, true);
                SetLineOrder(ul, take, HL_TAKE, true);
                SetLineOrder(ul, stop, HL_STOP, true);
        }
};
```

The highlighted lines will check if the order is free and if the EA should intervene in it. If the EA intervenes, the calculation will be made based on the financial value presented in the Chart Trader and based on the pending order volume. The simple pending order will then receive limits calculated based on the collected information. The EA will create lines to inform about that the limits will be created thus turning the simple pending order into an OCO pending order.

### Conclusion

Despite all the attempts to test the system and to see if it recognizes the account as hedging, I have not been successful at this stage. The EA always reported that the account was in the netting mode, even if the MetaTrader 5 platform reported that the account was hedging. Therefore, you should be careful. Although it works as we wanted, pending orders are adjusted even on a hedging account as if it were a netting account...

The video clearly shows everything described above. As you can see, the system is very interesting to use.

Demonstração - YouTube

Tap to unmute

[Demonstração](https://www.youtube.com/watch?v=9Cu9D3QYIYA) [MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ)

MQL5.community1.91K subscribers

[Watch on](https://www.youtube.com/watch?v=9Cu9D3QYIYA)

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10462](https://www.mql5.com/pt/articles/10462)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10462.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Seguranma\_6\_I\_t.zip](https://www.mql5.com/en/articles/download/10462/ea_-_seguranma_6_i_t.zip "Download EA_-_Seguranma_6_I_t.zip")(7151.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/429639)**
(1)


![Gerard William G J B M Dinh Sy](https://c.mql5.com/avatar/2026/1/69609d33-0703.png)

**[Gerard William G J B M Dinh Sy](https://www.mql5.com/en/users/william210)**
\|
20 Jun 2023 at 12:12

**MetaQuotes:**

A new article [Developing an Expert Trading Advisor from scratch (part 18): New order system (I)](https://www.mql5.com/en/articles/10462) has been published:

Author: [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831 "DJ_TLoG_831")

Hello and thank you.

Could you also give us the links to the first episodes?

Thanks

![Neural networks made easy (Part 17): Dimensionality reduction](https://c.mql5.com/2/48/Neural_networks_made_easy_017.png)[Neural networks made easy (Part 17): Dimensionality reduction](https://www.mql5.com/en/articles/11032)

In this part we continue discussing Artificial Intelligence models. Namely, we study unsupervised learning algorithms. We have already discussed one of the clustering algorithms. In this article, I am sharing a variant of solving problems related to dimensionality reduction.

![Learn how to design a trading system by Chaikin Oscillator](https://c.mql5.com/2/48/why-and-how__1.png)[Learn how to design a trading system by Chaikin Oscillator](https://www.mql5.com/en/articles/11242)

Welcome to our new article from our series about learning how to design a trading system by the most popular technical indicator. Through this new article, we will learn how to design a trading system by the Chaikin Oscillator indicator.

![Complex indicators made easy using objects](https://c.mql5.com/2/48/complex-indicators.png)[Complex indicators made easy using objects](https://www.mql5.com/en/articles/11233)

This article provides a method to create complex indicators while also avoiding the problems that arise when dealing with multiple plots, buffers and/or combining data from multiple sources.

![Developing a trading Expert Advisor from scratch (Part 17): Accessing data on the web (III)](https://c.mql5.com/2/47/development.png)[Developing a trading Expert Advisor from scratch (Part 17): Accessing data on the web (III)](https://www.mql5.com/en/articles/10447)

In this article we continue considering how to obtain data from the web and to use it in an Expert Advisor. This time we will proceed to developing an alternative system.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/10462&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051687503943029942)

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