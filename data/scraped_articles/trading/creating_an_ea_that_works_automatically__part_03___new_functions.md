---
title: Creating an EA that works automatically (Part 03): New functions
url: https://www.mql5.com/en/articles/11226
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:09:54.195566
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/11226&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069181129078014270)

MetaTrader 5 / Trading


### Introduction

In the previous article [Creating an EA that works automatically (Part 02): Getting started with the code](https://www.mql5.com/en/articles/11223), we started to develop an order system that we will use in our automated EA. However, we have created only one of the necessary functions.

Usually, the order system needs to have a few additional things in order to work in a fully automated mode. In addition, some people prefer to use several EAs working with different settings but with the same asset.

This is not recommended for accounts with the NETTING position accounting system. The reason is that the trade server creates the so-called average position price and this can cause a situation when one EA tries to sell and the other one tries to buy. Obviously, this will cause both EAs to disappear, losing their deposit in a short period of time. This is not the situation on HEDGING accounts. On such accounts, one EA can sell while the other one can buy. One order does not cancel the other one.

Some people even trade the same asset as the EA — but this must be done on a HEDGING account. When in doubt, never run two EAs trading the same asset (at least those trading with the same broker), and never trade the same asset while an automated EA is running.

Now that you have been warned, we can add other necessary functions to the EA, which will cover more than 90% of the cases.

### Why do we need new functions?

Usually, when an automated EA makes a trade, it basically enters and exits the market, i.e. the EA rarely places an order in the order book. However, there are situations when it becomes necessary to place an order in the order book. The procedure implementing such placement is one of the most difficult. For this reason, the previous article was devoted exclusively to the implementation of such a function. But this is not enough for the automated EA. Because, for the most part, it will be entering and exiting the market. Therefore, we need at least two additional functions:

- One of them will send orders to buy or sell at the market price;
- The other one will enable us to change the order price.

Only these two functions, together with the one discussed in the previous article, are what you really need in an automated EA. Now let's see why we need to add only these two features and nothing more. So, when the EA opens a position, be it a sell or a buy one, it will often do so at the market price with a volume predetermined by the trader.

When you need to close a position, you can do it in two ways: the first is at the market price, by making a deal in the opposite direction and with the same volume. But this will not often lead to position closure. Actually, it will only work on a NETTING account. On a HEDGING account, this operation will not close a position. In this case we need a clear order to close the existing position.

Although opening a trade in the opposite direction with the same volume in the HEDGING account will allow you to lock in the price, it does not actually mean that the position is closed. Locking the price brings neither profit nor loss. For this reason, we will add the ability for the EA to close the position, which will require the implementation of the third function. Note: on a NETTING account, you can simply send a market order with the same volume but in the opposite direction and the position will be closed.

Now we need a position to be able to change the order price. In some operational models the EA works like this: it opens a market position and immediately sends a pending STOP LOSS order. This order is added to the order book and remains there all the time until the position is closed. In this case, the EA will not actually send an order to close a position or any other type of request. It will simply manage the order in the book in such a way that the position closing order is always active.

This works well for NETTING accounts, but for HEDGING accounts, this system will not work as expected: in this case, the order in the order book will simply do what I already explained above about the way the trade is closed. But back to the point. The same procedure that will manage the order in the order book can also be used to move the take profit and stop loss levels in an OCO order ("order cancels the order").

Usually, an automated EA does not use OCO orders, it simply works with a market order and with the one in the order book. But in the case of HEDGING accounts, this mechanism can be done using an OCO order, while only a stop loss level will be set. Or, if the programmer wants to, they can simply enter the market and have the EA watch the market in some way. Once a certain point or price level is reached, the EA will send an order to close a position.

I have explained all this only to show that there are different ways to do the same thing - in our case to **CLOSE A POSITION**. Opening a position is the easiest part of the process. Closing is a tricky part, as you will have to take into account the following:

- Possible moments of high volatility where orders can "jump" (in the case of OCO orders). Orders in the order book excluding orders **STOP LIMIT** will never jump, they can trigger outside the desired point, but they never jump **.**
- Connection problems, when the EA may be unable to access the server for a while;
- Problems involved with liquidity, where an order can sit there, waiting to be executed, but the volume is not enough for it to actually be executed;
- And worst of all: the EA can start to execute orders at random.

All these points must be taken into account and observed when creating an automated EA. There are also other points, for example some programmers add EA operation times. As for this, I am going to be honest. This is completely **useless**. Although another article shows how to do it, I do not recommend doing so and here is the reason why.

Think about the following: you don't know how to trade, you have an automated EA to do this, you set a certain schedule. Now you think you can relax and do something else... **Wrong**. **Never**, I repeat, **never** leave an EA, even an automated one **unsupervised**. **NEVER**. While it is running, you or someone you trust should be near it and observe how it works.

Leaving the EA run unattended means opening the door to trouble. Adding scheduling methods or triggers to start and end the EA is the most stupid thing a person could do when using an automated EA. Please don't do this. If you want the EA to work for you, then turn it on and be there watching. When you need to leave, turn it off and go do what you need to do. Never leave an EA unsupervised, trading on its own. Do not do this, as the results may be very disappointing.

### Implementing required functions

Since the procedure for executing a market order is very similar to that used to submit a pending order, we can create a general procedure — so that all fields that have the same fill type are filled in. Only operation-specific fields will be filled locally. So, let's see this function with common filling:

```
inline void CommonData(const ENUM_ORDER_TYPE type, const double Price, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
                        {
                                double Desloc;

                                ZeroMemory(m_TradeRequest);
				m_TradeRequest.magic		= m_Infos.MagicNumber;
                                m_TradeRequest.symbol           = _Symbol;
                                m_TradeRequest.volume           = NormalizeDouble(m_Infos.VolMinimal + (m_Infos.VolStep * (Leverage - 1)), m_Infos.nDigits);
                                m_TradeRequest.price            = NormalizeDouble(Price, m_Infos.nDigits);
                                Desloc = FinanceToPoints(FinanceStop, Leverage);
                                m_TradeRequest.sl               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? -1 : 1)), m_Infos.nDigits);
                                Desloc = FinanceToPoints(FinanceTake, Leverage);
                                m_TradeRequest.tp               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? 1 : -1)), m_Infos.nDigits);
                                m_TradeRequest.type_time        = (IsDayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
                                m_TradeRequest.stoplimit        = 0;
                                m_TradeRequest.expiration       = 0;
                                m_TradeRequest.type_filling     = ORDER_FILLING_RETURN;
                                m_TradeRequest.deviation        = 1000;
                                m_TradeRequest.comment          = "Order Generated by Experts Advisor.";
                        }
```

Note that all what we have in this common function, was present in the pending order creation function which we considered in the previous article. But I also added a little extra thing, which did not exist before but which can be very useful in case you are working with a HEDGING account or intend to create an EA that will only observe the orders it has created. This is the so called magic number. Normally I do not use this number, but if you are going to do this, you will already have the ready way to support it.

So, let's at this new function that is responsible for sending a pending order:

```
                ulong CreateOrder(const ENUM_ORDER_TYPE type, const double Price, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
                        {
                                double  bid, ask, Desloc;

                                Price = AdjustPrice(Price);
                                bid = SymbolInfoDouble(_Symbol, (m_Infos.PlotLast ? SYMBOL_LAST : SYMBOL_BID));
                                ask = (m_Infos.PlotLast ? bid : SymbolInfoDouble(_Symbol, SYMBOL_ASK));
                                CommonData(type, AdjustPrice(Price), FinanceStop, FinanceTake, Leverage, IsDayTrade);
                                m_TradeRequest.action   = TRADE_ACTION_PENDING;
                                m_TradeRequest.type     = (type == ORDER_TYPE_BUY ? (ask >= Price ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_BUY_STOP) :
                                                                                    (bid < Price ? ORDER_TYPE_SELL_LIMIT : ORDER_TYPE_SELL_STOP));
                                ZeroMemory(m_TradeRequest);
                                m_TradeRequest.action           = TRADE_ACTION_PENDING;
                                m_TradeRequest.symbol           = _Symbol;
                                m_TradeRequest.volume           = NormalizeDouble(m_Infos.VolMinimal + (m_Infos.VolStep * (Leverage - 1)), m_Infos.nDigits);
                                m_TradeRequest.type             = (type == ORDER_TYPE_BUY ? (ask >= Price ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_BUY_STOP) :
                                                                                            (bid < Price ? ORDER_TYPE_SELL_LIMIT : ORDER_TYPE_SELL_STOP));
                                m_TradeRequest.price            = NormalizeDouble(Price, m_Infos.nDigits);
                                Desloc = FinanceToPoints(FinanceStop, Leverage);
                                m_TradeRequest.sl               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? -1 : 1)), m_Infos.nDigits);
                                Desloc = FinanceToPoints(FinanceTake, Leverage);
                                m_TradeRequest.tp               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? 1 : -1)), m_Infos.nDigits);
                                m_TradeRequest.type_time        = (IsDayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
                                m_TradeRequest.type_filling     = ORDER_FILLING_RETURN;
                                m_TradeRequest.deviation        = 1000;
                                m_TradeRequest.comment          = "Order Generated by Experts Advisor.";

                                return (((type == ORDER_TYPE_BUY) || (type == ORDER_TYPE_SELL)) ? ToServer() : 0);
                        };
```

All the crossed out parts have been removed from the code, since these fields are filled in by the common function. What we really need to do is adjust these two values and the order system will create a pending order as shown in the previous article.

Now let's see what we actually need to program to create an order system capable of sending execution requests at the market price. The required code is shown below:

```
                ulong ToMarket(const ENUM_ORDER_TYPE type, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
                        {
                                CommonData(type, SymbolInfoDouble(_Symbol, (type == ORDER_TYPE_BUY ? SYMBOL_ASK : SYMBOL_BID)), FinanceStop, FinanceTake, Leverage, IsDayTrade);
                                m_TradeRequest.action   = TRADE_ACTION_DEAL;
                                m_TradeRequest.type     = type;

                                return (((type == ORDER_TYPE_BUY) || (type == ORDER_TYPE_SELL)) ? ToServer() : 0);
                        };
```

See how easy it is: all you need to change compared to the pending order is just these two things. In this way, we guarantee that the server will always receive compatible data, since the only change will be the request type.

Thus, the operation, both in terms of analyzing the return and the way of calling the procedures for placing a pending order or executing a market transaction, is almost the same. The only real difference for those who call the procedure is that when a market order is executed, you do not need to provide a price, as the class will fill in the value correctly, while a pending order needs a price to be specified. Except for that all the operation will be the same.

Now let's see what has changed in the system. Since we have added a value to be used as the magic number, we need to create a class that will receive this value. This should be done in the class constructor. This is how the constructor looks like now:

```
                C_Orders(const ulong magic = 0)
                        {
                                m_Infos.MagicNumber     = magic;
                                m_Infos.nDigits         = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
                                m_Infos.VolMinimal      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
                                m_Infos.VolStep         = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
                                m_Infos.PointPerTick    = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
                                m_Infos.ValuePerPoint   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
                                m_Infos.AdjustToTrade   = m_Infos.PointPerTick / m_Infos.ValuePerPoint;
                                m_Infos.PlotLast        = (SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE) == SYMBOL_CHART_MODE_LAST);
                        };
```

Let's see what is happening in the above code. When we declare a default value as done here, we don't have to inform about it at the time the class is being created, thus we make this constructor as if it were default (in which no type of argument is received).

But, if you want to force the user of the class (i.e. the programmer), to inform which values should be used during the phase where the class is created, remove the value of the parameter. So, when the compiler tries to generate the code, it will notice that something is missing, and will ask which values to use. But this will only work, if the class does not contain another constructor. This is a small detail, which many people can't understand: why sometimes we have to specify values and sometimes not.

As you can see, programming can be very interesting. In many cases, what we are really doing is trying to create a solution with the least amount of effort so that there are not so many things to code and test. But we have not finished the C\_Orders class yet. It still needs one required function and one optional one, which we will still create due to the difference between HEDGING and NETTING accounts. Let's move on:

```
                bool ModifyPricePoints(const ulong ticket, const double Price, const double PriceStop, const double PriceTake)
                        {
                                ZeroMemory(m_TradeRequest);
                                m_TradeRequest.symbol   = _Symbol;
                                if (OrderSelect(ticket))
                                {
                                        m_TradeRequest.action   = (Price > 0 ? TRADE_ACTION_MODIFY : TRADE_ACTION_REMOVE);
                                        m_TradeRequest.order    = ticket;
                                        if (Price > 0)
                                        {
                                                m_TradeRequest.price      = NormalizeDouble(AdjustPrice(Price), m_Infos.nDigits);
                                                m_TradeRequest.sl         = NormalizeDouble(AdjustPrice(PriceStop), m_Infos.nDigits);
                                                m_TradeRequest.tp         = NormalizeDouble(AdjustPrice(PriceTake), m_Infos.nDigits);
                                                m_TradeRequest.type_time  = (ENUM_ORDER_TYPE_TIME)OrderGetInteger(ORDER_TYPE_TIME) ;
                                                m_TradeRequest.expiration = 0;
                                        }
                                }else if (PositionSelectByTicket(ticket))
                                {
                                        m_TradeRequest.action   = TRADE_ACTION_SLTP;
                                        m_TradeRequest.position = ticket;
                                        m_TradeRequest.tp       = NormalizeDouble(AdjustPrice(PriceTake), m_Infos.nDigits);
                                        m_TradeRequest.sl       = NormalizeDouble(AdjustPrice(PriceStop), m_Infos.nDigits);
                                }else return false;
                                ToServer();

                                return (_LastError == ERR_SUCCESS);
                        };
```

The above procedure is extremely important, it is even more important than the next one, which we will see. The reason is that this procedure is responsible for manipulating price positions, whether they are order book positions in the case of a pending order, or the limits in cased of an open position. The function above is so powerful that it can create or remove order or position limits. To understand how it all works, let's analyze its internal code. This will help you understand how to use this function correctly.

To make the explanation simpler and more understandable, I will break everything down into parts, so be careful not to get lost in the explanation.

Let's start by understanding the following: When you submit an order, whether to place a pending order to the order book or to submit a market order, you get a return value. If no error occurred in the request, this value will be non-null. However, this value should not be ignored, because the value returned by the order or position creation functions should be stored with great care, as it represents the order or position ticket.

This ticket, which serves as a sort of a pass, will give you several possibilities, including the ability to manipulate orders or positions that are on the trade server. So, the value which you get when sending a trade at the market price or when trying to place a pending order, and which is returned by the functions that perform this procedure, essentially serves you (when it is different from zero) as a pass for the EA to communicate with the server. By using the ticket, you get the ability to manipulate prices.

Each order or position has a unique ticket, so take care of this number and don't try to create it randomly. There are ways to get it if you don't know the value or have lost it. However, the ways will take the EA's time, so make sure you don't lose this number.

Let's first assume that we have an order and we want to remove or change the limit values (take profit or stop loss) of this order. Don't confuse order with position. When I say "order" I mean a possible and future position; usually the orders are in the order book, while the position is when the order is actually filled.. In this case, you will provide the order ticket and new price values. Note that these are now price values and you will no longer be indicating financial value (the monetary value associated with buying and selling). What we expect now is the face value which you see on the chart. Therefore, you can't work here randomly, otherwise your request will be rejected.

Now that this has become clear, you can virtually place almost any take profit and stop loss value in an order, but in reality this is not entirely true. If you have a buy order, the stop loss value cannot be greater than the position opening price, and the take profit value cannot be less than the position opening price. If you try to do this, the server will return an error.

Now pay close attention to this: the take profit and stop loss values in the order must meet these criteria, but in the case of a position, the stop loss value of a buy position can be higher than the opening price, in which case the stop loss becomes a stop profit, i.e. you will already have some profit if the stop loss order is hit. But we will talk more about this later. For now, remember that for the buy order, the stop loss should be below the open price, and on the sell order it should be higher. The stop loss of a position can be anywhere.

The above only concerns the stop levels of orders and positions. But if you look at the function above, you will notice that we can also manipulate the opening price of the position, as long as it is still an order. The important detail: when you do this, you will have to move the stop loss and the take profit together. If you don't do this, at some point, the server will deny your request to change the open price.

To understand this part, let's create a small EA program to check these cases. Create a new EA file and then copy and paste the following code into this open file. After that, compile the EA and launch on a chart. Then we will see the explanation:

```
#property copyright "Daniel Jose"
#property description "This one is an automatic Expert Advisor"
#property description "for demonstration. To understand how to"
#property description "develop yours in order to use a particular"
#property description "operational, see the articles where there"
#property description "is an explanation of how to proceed."
#property version   "1.03"
#property link      "https://www.mql5.com/pt/articles/11226"
//+------------------------------------------------------------------+
#include <Generic Auto Trader\C_Orders.mqh>
//+------------------------------------------------------------------+
C_Orders *orders;
//+------------------------------------------------------------------+
input int       user01   = 1;           //Lot value
input int       user02   = 100;         //Take Profit
input int       user03   = 75;          //Stop Loss
input bool      user04   = true;        //Day Trade ?
input double    user05   = 84.00;       //Entry price...
//+------------------------------------------------------------------+
int OnInit()
{
        orders = new C_Orders(1234456789);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        delete orders;
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
#define KEY_UP                  38
#define KEY_DOWN                40
#define KEY_NUM_1               97
#define KEY_NUM_2               98
#define KEY_NUM_3               99
#define KEY_NUM_7               103
#define KEY_NUM_8               104
#define KEY_NUM_9               105

        static ulong sticket = 0;
        int key = (int)lparam;

        switch (id)
        {
                case CHARTEVENT_KEYDOWN:
                        switch (key)
                        {
                                case KEY_UP:
                                        if (sticket == 0)
                                                sticket = (*orders).CreateOrder(ORDER_TYPE_BUY, user05, user03, user02, user01, user04);
                                        break;
                                case KEY_DOWN:
                                        if (sticket == 0)
                                                sticket = (*orders).CreateOrder(ORDER_TYPE_SELL, user05, user03, user02, user01, user04);
                                        break;
                                case KEY_NUM_1:
                                case KEY_NUM_7:
                                        if (sticket > 0) ModifyStop(key == KEY_NUM_7, sticket);
                                        break;
                                case KEY_NUM_2:
                                case KEY_NUM_8:
                                        if (sticket > 0) ModifyPrice(key == KEY_NUM_8, sticket);
                                        break;
                                case KEY_NUM_3:
                                case KEY_NUM_9:
                                        if (sticket > 0) ModifyTake(key == KEY_NUM_9, sticket);
                                        break;
                        }
                        break;
        }
}
//+------------------------------------------------------------------+
void ModifyPrice(bool IsUp, const ulong ticket)
{
        double p, s, t;

        if (!OrderSelect(ticket)) return;
        p = OrderGetDouble(ORDER_PRICE_OPEN) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
        s = OrderGetDouble(ORDER_SL);
        t = OrderGetDouble(ORDER_TP);
        (*orders).ModifyPricePoints(ticket, p, s, t);
}
//+------------------------------------------------------------------+
void ModifyTake(bool IsUp, const ulong ticket)
{
        double p, s, t;

        if (!OrderSelect(ticket)) return;
        p = OrderGetDouble(ORDER_PRICE_OPEN);
        s = OrderGetDouble(ORDER_SL);
        t = OrderGetDouble(ORDER_TP) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
        (*orders).ModifyPricePoints(ticket, p, s, t);
}
//+------------------------------------------------------------------+
void ModifyStop(bool IsUp, const ulong ticket)
{
        double p, s, t;

        if (!OrderSelect(ticket)) return;
        p = OrderGetDouble(ORDER_PRICE_OPEN);
        s = OrderGetDouble(ORDER_SL) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
        t = OrderGetDouble(ORDER_TP);
        (*orders).ModifyPricePoints(ticket, p, s, t);
}
//+------------------------------------------------------------------+
```

Don't worry if this code seems complicated at first glance. It just serves to show one thing that can happen. You should understand it in order to use it to your advantage in the future.

The code itself is very similar to what we saw in the previous article, but here we can do something more: we can manipulate orders by changing the value of the take profit, stop loss and open price. The only inconvenience that is still present in the code is that once an order is placed on the chart, it cannot be simply removed to place another one. You can leave the order on the chart, it doesn't matter, but when using this EA to create a pending order (and at the moment it only works for pending orders), you will be able to change the price point showing where the order will be by using the numerical keyboard, the one in the right part of the physical keyboard.

This is done using this event handler. Please note that each of the keys is used for one thing, for example, to increase the stop loss or decrease the order price. Do it while watching the result in the trade tab of the toolbox. You will learn a lot.

If you are not sure about launching this code in the platform (it is absolutely harmless, but still you should be careful), please watch the video below, which demonstrates what the code actually does.

Demonstração 01 Parte 03 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11226)

MQL5.community

1.91K subscribers

[Demonstração 01 Parte 03](https://www.youtube.com/watch?v=NJsU2I6DwYE)

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

[Watch on](https://www.youtube.com/watch?v=NJsU2I6DwYE&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11226)

0:00

0:00 / 2:05

•Live

•

Demonstration of the above code.

You can see that when moving stop loss and take profit, we have the correct movement, but when we move the open price, stop loss and take profit stay still. Why does this happen? The reason is that for a trade server, you are actually moving an order, which is possibly a stop on another open trade.

Remember that at the beginning I mentioned that one way to close an open trade is to place an order in the order book and move it smoothly. This is exactly what should happen at this stage. That is, for the server, the price to be moved is only the price information about which is provided. It does not treat OCO orders as a whole. An OCO orders appears as different price points. Once one of the limits is reached, the server will send an event that will remove the price where the position is open. Both orders, take profit and stop loss, will cease to exist since the related ticket will be removed from the system.

What we need to do is to make stop loss and take profit move together with the order price in this case. To implement this, let's make the following changes to the above code:

```
void ModifyPrice(bool IsUp, const ulong ticket)
{
        double p, s, t;

        if (!OrderSelect(ticket)) return;
        p = OrderGetDouble(ORDER_PRICE_OPEN) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
        s = OrderGetDouble(ORDER_SL) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
        t = OrderGetDouble(ORDER_TP) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
        (*orders).ModifyPricePoints(ticket, p, s, t);
}
```

Now, as soon as the open price moves, the take profit and the stop loss prices will also move, while always maintaining the same distance from the open point.

It works similarly for the position, but we cannot move the position open price. We will use the same function to move take profit and stop loss with a small difference:

```
                bool ModifyPricePoints(const ulong ticket, const double Price, const double PriceStop, const double PriceTake)
                        {
                                ZeroMemory(m_TradeRequest);
                                m_TradeRequest.symbol   = _Symbol;
                                if (OrderSelect(ticket))
                                {
// ... Code to move orders...
                                }else if (PositionSelectByTicket(ticket))
                                {
                                        m_TradeRequest.action   = TRADE_ACTION_SLTP;
                                        m_TradeRequest.position = ticket;
                                        m_TradeRequest.tp       = NormalizeDouble(AdjustPrice(PriceTake), m_Infos.nDigits);
                                        m_TradeRequest.sl       = NormalizeDouble(AdjustPrice(PriceStop), m_Infos.nDigits);
                                }else return false;
                                ToServer();

                                return (_LastError == ERR_SUCCESS);
                        };
```

Using the code above, we implement the breakeven or trailing stop. The only thing we need to do is check the values that will serve as a trigger to start the movement - this is the breakeven value. Once the trigger is fired, we fix the open price of the position and place it as the stop loss price, then we call the change in the position price, and the result is breakeven.

A trailing stop works exactly the same, only in this case the level will move when the trigger is triggered at a certain price distance or something else. When this happens, we take the new value to be used for the stop loss and call the function above. This is very simple.

We will consider the breakeven and trailing stop triggers later, when I show how to develop these triggers for an EA working automatically. If you are an enthusiast, you may already come up with some ideas regarding these levels. If this is the case - Congratulations! You are on the right path.

Now let's get back to the price change procedure, because there's something that I haven't mentioned yet. It is important that you know how and why it is there. For easier explanation, let's pay attention to the following code:

```
                bool ModifyPricePoints(const ulong ticket, const double Price, const double PriceStop, const double PriceTake)
                        {
                                ZeroMemory(m_TradeRequest);
                                m_TradeRequest.symbol   = _Symbol;
                                if (OrderSelect(ticket))
                                {
                                        m_TradeRequest.action = (Price > 0 ? TRADE_ACTION_MODIFY : TRADE_ACTION_REMOVE);
                                        m_TradeRequest.order  = ticket;
                                        if (Price > 0)
                                        {
                                                m_TradeRequest.price      = NormalizeDouble(AdjustPrice(Price), m_Infos.nDigits);
                                                m_TradeRequest.sl         = NormalizeDouble(AdjustPrice(PriceStop), m_Infos.nDigits);
                                                m_TradeRequest.tp         = NormalizeDouble(AdjustPrice(PriceTake), m_Infos.nDigits);
                                                m_TradeRequest.type_time  = (ENUM_ORDER_TYPE_TIME)OrderGetInteger(ORDER_TYPE_TIME) ;
                                                m_TradeRequest.expiration = 0;
                                        }
                                }else if (PositionSelectByTicket(ticket))
                                {
// Code for working with positions ...
                                }else return false;
                                ToServer();

                                return (_LastError == ERR_SUCCESS);
                        };
```

There are cases when we need to remove an order from the order book, i.e. to cancel or close it. And there is a danger that during execution, you do something which will cause the EA to generate a price equal to zero. Believe me, it happens, and it's something quite common, especially if you are using an automated EA. Then the EA sends an order, so that the order price is modified, but due to an error this price is sent as zero.

In these cases, the trade server will reject the order, but the EA can stand there, almost madly insisting that the opening price must be zero. If you don't do something about this, it can go into a loop, which is extremely unpleasant on a live account. To avoid the EA staying there, insisting, with something that will not be accepted by the server, I included the following idea: If the EA sends a position opening price equal to zero, the order must be closed by the server. That's exactly what this code here does..

It informs the trading server that the order must be closed and removed from the order book. When that happens, the order will no longer be there, and you can be informed about it. But I don't include a code for this here, as there are other equally useful uses for this sort of thing, not just preventing the EA from insisting on something. It can be used to simply remove an order from the order book.

But we haven't finished the article yet. There is one last procedure, although it is optional, but in some cases it can be useful. So, since I'm here opening the black box of how the order system works, let's look at one more procedure, which is shown below:

```
                bool ClosePosition(const ulong ticket, const uint partial = 0)
                        {
                                double v1 = partial * m_Infos.VolMinimal, Vol;
                                bool IsBuy;

                                if (!PositionSelectByTicket(ticket)) return false;
                                IsBuy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
                                Vol = PositionGetDouble(POSITION_VOLUME);
                                ZeroMemory(m_TradeRequest);
                                m_TradeRequest.action    = TRADE_ACTION_DEAL;
                                m_TradeRequest.type      = (IsBuy ? ORDER_TYPE_SELL : ORDER_TYPE_BUY);
                                m_TradeRequest.price     = SymbolInfoDouble(_Symbol, (IsBuy ? SYMBOL_BID : SYMBOL_ASK));
                                m_TradeRequest.position  = ticket;
                                m_TradeRequest.symbol    = _Symbol;
                                m_TradeRequest.volume    = ((v1 == 0) || (v1 > Vol) ? Vol : v1);
                                m_TradeRequest.deviation = 1000;
                                ToServer();

                                return (_LastError == ERR_SUCCESS);
                        };
```

This procedure above makes many people dream big, imagining things, and seeing stars. When looking at this procedure, you must be thinking that it serves to close a position. Why would anyone daydream about it?

Calm down, my dear reader. You still don't understand, you have a preconception when you see the name of the procedure. But let's go a little deeper, analyze the code and understand why many are dreaming big. This procedure has some calculations, but why? This is done to allow partial closure. Let's understand how this is implemented. Suppose you have an open position with a volume of 300. Then, if the minimum tradable volume is 100, you can exit with 100, 200 or 300.

But for that you will have to inform a value, which is by default zero, i.e. it tells the function, that the position will be closed completely, but this will only happen if you keep it as default. But there's a detail: you don't leave it default. You specify the volume value, i.e. the number of lots to be closed. If the minimum lot volume is 100 and you have 300, it means the volume is 3x. To close partially, you can specify 1 or 2. If you specify 0, 3 or a value greater than 3, the position will be fully closed.

However, there are some alternatives. For example, in the case of B3 (Bolsa do Brasil), company shares are traded in lots of 100, but there is the fractional market, where you can trade from 1. In this case, if you have the EA running in fractional mode, the value in the same example of 300 can go from 1 to 299, and even so the position will not be completely closed, leaving one open residue.

I hope now you understand. The specific procedure depends on the market and the interests of the trader. If you work on a HEDGING account, you will definitely need the above function. Without it, positions will simply accumulate, taking time and resources from the EA to analyze what could have already been closed.

To end this article and to cover all question about the order system, let's see how the EA code should be, to remove the following limitation: once an order is created, it cannot place other others and cannot manipulate data. To fix these problems, we'll have to make some changes to the EA code, but by doing this, you'll already be able to have fun, and do a lot more things. Perhaps this will make you a little more excited about what can be done with a relatively simple code, with little programming knowledge.

To partially correct the EA, we'll have to change the code responsible for handling chart events. The new code is shown below:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
#define KEY_UP                  38
#define KEY_DOWN                40
#define KEY_NUM_1               97
#define KEY_NUM_2               98
#define KEY_NUM_3               99
#define KEY_NUM_7               103
#define KEY_NUM_8               104
#define KEY_NUM_9               105

        static ulong sticket = 0;
        ulong ul0;
        int key = (int)lparam;

        switch (id)
        {
                case CHARTEVENT_KEYDOWN:
                        switch (key)
                        {
                                case KEY_UP:
                                        if (sticket == 0)
                                                sticket = (*orders).CreateOrder(ORDER_TYPE_BUY, user05, user03, user02, user01, user04);
                                        ul0 = (*orders).CreateOrder(ORDER_TYPE_BUY, user05, user03, user02, user01, user04);
                                        sticket = (ul0 > 0 ? ul0 : sticket);
                                        break;
                                case KEY_DOWN:
                                        if (sticket == 0)
                                                sticket = (*orders).CreateOrder(ORDER_TYPE_SELL, user05, user03, user02, user01, user04);
                                        ul0 = (*orders).CreateOrder(ORDER_TYPE_SELL, user05, user03, user02, user01, user04);
                                        sticket = (ul0 > 0 ? ul0 : sticket);
                                        break;
                                case KEY_NUM_1:
                                case KEY_NUM_7:
                                        if (sticket > 0) ModifyStop(key == KEY_NUM_7, sticket);
                                        break;
                                case KEY_NUM_2:
                                case KEY_NUM_8:
                                        if (sticket > 0) ModifyPrice(key == KEY_NUM_8, sticket);
                                        break;
                                case KEY_NUM_3:
                                case KEY_NUM_9:
                                        if (sticket > 0) ModifyTake(key == KEY_NUM_9, sticket);
                                        break;
                        }
                        break;
        }
}
```

We have deleted the crossed out parts and added a new variable, which will receive the ticket value returned by the order class. If the value is non-zero, the new ticket will be saved in the static variable which will store the value until a new value is written instead of it. Well, this will already allow you to manage a lot more things, and if an order is accidentally executed and you have not overwritten the value when opening a new order, you will still be able to manage the order limits.

Now, as an extra task, to test if you really have really learned how to work with the order system (before the next article), try to make the order system open a market position and use the ticket value to be able to manage the limits of the open position. Do it without reading further explanation. This will help you understand whether you are able to follow the explanations.

Now let's see the code changing the price. The opening code is the same. We can skip the case when the order is converted into a position. Let's consider the take profit:

```
void ModifyTake(bool IsUp, const ulong ticket)
{
        double p = 0, s, t;

        if (!OrderSelect(ticket)) return;
        if (OrderSelect(ticket))
        {
                p = OrderGetDouble(ORDER_PRICE_OPEN);
                s = OrderGetDouble(ORDER_SL);
                t = OrderGetDouble(ORDER_TP) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
        }else if (PositionSelectByTicket(ticket))
        {
                s = PositionGetDouble(POSITION_SL);
                t = PositionGetDouble(POSITION_TP) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
        }else return;
        (*orders).ModifyPricePoints(ticket, p, s, t);
}
```

The crossed out code no longer exists, and therefore we can use the new code to manage the take profit value of a position. The same applies to the stop loss code shown below:

```
void ModifyStop(bool IsUp, const ulong ticket)
{
        double p = 0, s, t;

        if (!OrderSelect(ticket)) return;
        if (OrderSelect(ticket))
        {
                p = OrderGetDouble(ORDER_PRICE_OPEN);
                s = OrderGetDouble(ORDER_SL) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
                t = OrderGetDouble(ORDER_TP);
        }else if (PositionSelectByTicket(ticket))
        {
                s = PositionGetDouble(POSITION_SL) + (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * (IsUp ? 1 : -1));
                t = PositionGetDouble(POSITION_TP);
        }else return;
        (*orders).ModifyPricePoints(ticket, p, s, t);
}
```

Use this EA as a learning tool, especially on demo accounts - make the most of it; explore in full what is in these first three articles, because at this point I consider the order system complete. In the next article, I'll show how to initialize an EA to capture some information, problems, and possible solutions related to this initialization. But these questions are not part of the order system — we have implemented everything we really need to make the EA automatic.

### Conclusion

Despite what we've seen in these first three articles, we're still a long way from building a complete automated EA. Many often ignore or don't know the details presented here, which is dangerous. We are just beginning to talk about automated systems and the dangers associated with using them without proper knowledge.

All the previously considered code is available in the attachment. Study it and learn how things work. Hope to see you in the next article.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11226](https://www.mql5.com/pt/articles/11226)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11226.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_03.zip](https://www.mql5.com/en/articles/download/11226/ea_automatico_-_03.zip "Download EA_Automatico_-_03.zip")(4.08 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/442199)**
(4)


![Tagli Elliotician](https://c.mql5.com/avatar/2022/12/63ad8be1-30f6.jpg)

**[Tagli Elliotician](https://www.mql5.com/en/users/filipetagli)**
\|
31 Dec 2022 at 11:55

Code three showed errors in my compilation. Code two worked normally. If you have any idea what I'm forgetting to look at, thank you.


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
1 Jan 2023 at 13:42

**filipetagli [#](https://www.mql5.com/pt/forum/435189#comment_44077390):**

Code three showed errors in my compilation. Code two worked normally. If you have any idea what I'm forgetting to look at, I'd appreciate it.

You need to show in detail what's going on, otherwise there's no way of guiding you ...👀👀

![Ilya Prozumentov](https://c.mql5.com/avatar/2015/5/555644C0-DCCB.jpg)

**[Ilya Prozumentov](https://www.mql5.com/en/users/sunnythedreamer)**
\|
18 Feb 2023 at 11:47

Does anyone know why (\*) is used when referring to orders?

```
(*orders).ModifyPricePoints(ticket, p, s, t);
```

![L Kd](https://c.mql5.com/avatar/2023/2/63F146C3-CBD7.jpg)

**[L Kd](https://www.mql5.com/en/users/lorenzo200)**
\|
6 Mar 2023 at 02:25

you have made the subject impossible and unapproachable (signed, a decades long coder). Not sure if that is your intention, or is it to induce maximum caution, so that only the truly committed persist?


![Experiments with neural networks (Part 3): Practical application](https://c.mql5.com/2/51/neural_network_experiments_p3_avatar.png)[Experiments with neural networks (Part 3): Practical application](https://www.mql5.com/en/articles/11949)

In this article series, I use experimentation and non-standard approaches to develop a profitable trading system and check whether neural networks can be of any help for traders. MetaTrader 5 is approached as a self-sufficient tool for using neural networks in trading.

![Population optimization algorithms: Bat algorithm (BA)](https://c.mql5.com/2/51/Bat-algorithm-avatar.png)[Population optimization algorithms: Bat algorithm (BA)](https://www.mql5.com/en/articles/11915)

In this article, I will consider the Bat Algorithm (BA), which shows good convergence on smooth functions.

![Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation](https://c.mql5.com/2/52/Recreating-built-in-OpenCL-API-002-avatar.png)[Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation](https://www.mql5.com/en/articles/12108)

Bulit-in OpenCL support in MetaTrader 5 still has a major problem especially the one about device selection error 5114 resulting from unable to create an OpenCL context using CL\_USE\_GPU\_ONLY, or CL\_USE\_GPU\_DOUBLE\_ONLY although it properly detects GPU. It works fine with directly using of ordinal number of GPU device we found in Journal tab, but that's still considered a bug, and users should not hard-code a device. We will solve it by recreating an OpenCL support as DLL with C++ on Linux. Along the journey, we will get to know OpenCL from concept to best practices in its API usage just enough for us to put into great use later when we deal with DLL implementation in C++ and consume it with MQL5.

![Creating an EA that works automatically (Part 02): Getting started with the code](https://c.mql5.com/2/50/Aprendendo-a-construindo_part_II_avatar.png)[Creating an EA that works automatically (Part 02): Getting started with the code](https://www.mql5.com/en/articles/11223)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. In the previous article, we discussed the first steps that anyone needs to understand before proceeding to creating an Expert Advisor that trades automatically. We considered the concepts and the structure.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11226&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069181129078014270)

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