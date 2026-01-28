---
title: Creating an EA that works automatically (Part 06): Account types (I)
url: https://www.mql5.com/en/articles/11241
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:09:12.397567
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/11241&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069166264196202773)

MetaTrader 5 / Trading


### Introduction

In the previous article [Creating an EA that works automatically (Part 05): Manual triggers (II)](https://www.mql5.com/en/articles/11237), we have developed a fairly simple EA with a great level of robustness and reliability. It can be used to trade any asset, including forex and stock symbols. It does not have any automation and is controlled completely manually.

Our EA in its current state can work in any situation but it is not yet ready for automation. We still have to work on a few points. There is some work to be done before we add break even or trailing stop, because if we add these mechanisms earlier, we will have to cancel some things later. Therefore, we will take a slightly different path and first consider the creation of a universal EA.

### The birth of the C\_Manager class

The C\_Manager class will be the isolation layer between the EA and the order system. At the same time, the class will start promoting some kind of automation for our EA, allowing it to automatically do some things.

Let's now see how the class building begins. Its initial code is shown below:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "C_Orders.mqh"
//+------------------------------------------------------------------+
class C_Manager : private C_Orders
{
        private :
                struct st00
                {
                        double  FinanceStop,
                                FinanceTake;
                        uint    Leverage;
                        bool    IsDayTrade;
                }m_InfosManager;
        public  :
//+------------------------------------------------------------------+
                C_Manager(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage, bool IsDayTrade)
                        :C_Orders(magic)
                        {
                                m_InfosManager.FinanceStop = FinanceStop;
                                m_InfosManager.FinanceTake = FinanceTake;
                                m_InfosManager.Leverage    = Leverage;
                                m_InfosManager.IsDayTrade  = IsDayTrade;
                        }
//+------------------------------------------------------------------+
                ~C_Manager() { }
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                C_Orders::CreateOrder(type, Price, m_InfosManager.FinanceStop, m_InfosManager.FinanceTake, m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                C_Orders::ToMarket(type, m_InfosManager.FinanceStop, m_InfosManager.FinanceTake, m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
};
```

What you see in the above code is the basic structure of what we are going to construct. Please note that the EA will no longer need to provide certain information when sending orders. Everything will be managed in this C\_Manager class. Actually, in the constructor call, we pass all the values that are needed in order to create an order or to send an order to open a market position.

But I want you to pay attention to one fact: the C\_Manager class inherits the C\_Orders class, but this inheritance is private. Why? The reason is security and increased reliability. When placing this class here as a type of 'syndicator', we want it to be the only point of communication between the EA and the class responsible for sending orders.

Since C\_Manager will control access to the order system, being able to send, close or modify orders and positions, we give the EA some kind of means to access the order system. But this access will be limited. Here are the two initial functions which the EA can use to access the order system. As you can see, they are much more limited than those in the C\_Orders class, but they are safer.

To understand the level of things we are implementing here, let's compare the EA's code from the previous article with the current one. We have only created the C\_Manager class. See what happened in two functions present in the EA.

```
int OnInit()
{
        manager = new C_Orders(def_MAGIC_NUMBER);
        manager = new C_Manager(def_MAGIC_NUMBER, user03, user02, user01, user04);
        mouse = new C_Mouse(user05, user06, user07, user03, user02, user01);

        return INIT_SUCCEEDED;
}
```

The previous code has been removed and replaced with a new, however, with a large number of parameters. But this is only a minor detail. The main thing (and, in my opinion, this makes everything more risky) is shown below:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        uint    BtnStatus;
        double  Price;
        static double mem = 0;

        (*mouse).DispatchMessage(id, lparam, dparam, sparam);
        (*mouse).GetStatus(Price, BtnStatus);
        if (TerminalInfoInteger(TERMINAL_KEYSTATE_CONTROL))
        {
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_UP))  (*manager).ToMarket(ORDER_TYPE_BUY, user03, user02, user01, user04);
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_DOWN))(*manager).ToMarket(ORDER_TYPE_SELL, user03, user02, user01, user04);
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_UP))  (*manager).ToMarket(ORDER_TYPE_BUY);
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_DOWN))(*manager).ToMarket(ORDER_TYPE_SELL);
        }
        if ((def_SHIFT_Press(BtnStatus) != def_CTRL_Press(BtnStatus)) && def_BtnLeftClick(BtnStatus))
        {
                if (mem == 0) (*manager).CreateOrder((def_SHIFT_Press(BtnStatus) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL), mem = Price, user03, user02, user01, user04);
                if (mem == 0) (*manager).CreateOrder((def_SHIFT_Press(BtnStatus) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL), mem = Price);
        }else mem = 0;
}
```

There are parts that have been simply removed. In turn, you can see that the new code is much simpler. But not only that. The use of a class to perform the "administrator" work guarantees that the EA will use exactly those parameters that were defined at class initialization. Thus, there is no risk of putting a type of wrong or invalid information in one of the calls. Everything is gathered in one place, in the C\_Manager class which now serves as a communication intermediary between the EA and C\_Orders. This significantly increases the level of security and reliability of the EA code.

### NETTING, EXCHANGE or HEDGING account... that is the question

Although many people ignore or are unaware of this fact, there is a serious problem here. Because of this, the EA may or may not work well — this is the type of account. Most of traders and MetaTrader 5 platform users do not know that there are three types of accounts in the market. But for those who want to develop an Expert Advisor running in a fully automated mode, this knowledge is crucial.

In this article series we will be talking about two account types: NETTING and HEDGING. The reason is simple: the NETTING account works for the EA in the same way as the EXCHANGE account.

Even if an EA has a simple automation, for example breakeven or trailing stop activation, the fact that it is running on a NETTING account makes its operation completely different form the one running on a HEDGING account. The reason is in the way the trading server works. On a NETTING account, the trade server creates an average price as you increase or decrease your position.

This is not done on the server for the HEDGING account. It treats all positions separately, so you can simultaneously have short and open positions for the same asset. This can never happen on a NETTING account. If you try to open an opposite position with the same lot, the server will close the position.

For this reason, we must know if an EA is designed for NETTING or HEDGING accounts, since the operation principle would be completely different. But this only applies to automated EA or EAs with some automation level. This does not matter for a manual EA.

Because of this fact, we cannot create any level of automation without generating some difficulty in terms of programming or usability.

Here, we need to standardize things a bit. In other words, we need to make sure that the EA can work on any account type in a standardized way. It is true that this will reduce the capabilities of the EA. However, an automated EA should have a great degree of freedom. The best way is to make the EA restrained to make it behave well. If it deviates a little, it must be banned or at least get some punishment.

The way to standardize things is to make the HEDGING account work similarly to the NETTING account from the EA's point of view. I know this may seem confusing and complicated, but what we really want is to allow the EA to have only one open position and only one pending order, in other words, it will be extremely limited and will not be able to do anything else.

Thus, we add the following code to the C\_Manager class:

```
class C_Manager : private C_Orders
{
        private :
                struct st00
                {
                        double  FinanceStop,
                                FinanceTake;
                        uint    Leverage;
                        bool    IsDayTrade;
                }m_InfosManager;
//---
                struct st01
                {
                        ulong   Ticket;
                        double  SL,
                                TP,
                                PriceOpen,
                                Gap;
                        bool    EnableBreakEven,
                                IsBuy;
                        int     Leverage;
                }m_Position;
                ulong           m_TicketPending;
                bool            m_bAccountHedging;
		double		m_Trigger;
```

In this structure, we create everything we may need to work with an open position. It already has some things related to the first level of automation, for example breakeven and trailing stop. A pending order will be stored in a simpler way, using a ticket. But if we need more data in the future, we can implement it. This will be enough for now. We also have another variable that tells us whether we are using a HEDGING or NETTING account. It will be particularly useful at certain moments. As usual, another variable has been added which will not be used at this stage, but we will need it later when creating breakeven and trailing stop triggers.

That's how we start to normalize things. After that we can make changes to the class constructor as shown below:

```
                C_Manager(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage, bool IsDayTrade, double Trigger)
                        :C_Orders(magic),
                        m_bAccountHedging(false),
                        m_TicketPending(0),
                        m_Trigger(Trigger)
                        {
                                string szInfo;

                                ZeroMemory(m_Position);
                                m_InfosManager.FinanceStop = FinanceStop;
                                m_InfosManager.FinanceTake = FinanceTake;
                                m_InfosManager.Leverage    = Leverage;
                                m_InfosManager.IsDayTrade  = IsDayTrade;
                                switch ((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE))
                                {
                                        case ACCOUNT_MARGIN_MODE_RETAIL_HEDGING:
                                                m_bAccountHedging = true;
                                                szInfo = "HEDGING";
                                                break;
                                        case ACCOUNT_MARGIN_MODE_RETAIL_NETTING:
                                                szInfo = "NETTING";
                                                break;
                                        case ACCOUNT_MARGIN_MODE_EXCHANGE:
                                                szInfo = "EXCHANGE";
                                                break;
                                }
                                Print("Detected Account ", szInfo);
                        }
```

I will show the code little by little for those who have no programming experience. I hope I am not being too boring about this, since I want everyone to be able to understand what we are doing here. Here are the explanations. These lines inform the compiler that we want these variables to be initialized before the constructor code execution starts. When a variable is created, the compiler usually assigns it a value of zero.

In these lines, we tell the compiler what the variable value will be when we create it. At this moment we reset all the content of the structure. This way we use less code and obtain a faster result. Here we are noting the fact that we will be working with a HEDGING account. If this information becomes required at certain time, we have a variable to tell this. Already here we inform in the terminal which account type was found. This is done to indicate the type in case the user does not know it.

But before we look at these procedures, think about the following: what if the EA finds more than one position (HEDGING account) or more than one pending order? What will happen then? In this case, we will get an error since the EA will not be able to work with more than one position and one order. To handle this, let us create the following enumeration in the code:

```
class C_Manager : private C_Orders
{
        enum eErrUser {ERR_Unknown};
        private :

// ... The rest of the code...

};
```

Here we will use an enumeration as it is easier to add new error codes to it. To do this, we only need to specify a new name, and the compiler will generate a value for the code, while there will be no risk of creating a duplicate value due to inattention. Note that the enumeration is located before private code part so it will be public. But to access it outside the class, we will need to use a small detail to inform the compiler which enumeration is correct. This is especially useful when we want to use enumerations related to a particular class. Now let's look at the procedures that will load the things which could be left on the chart and which the EA must restore before starting work. The first one is as follows:

```
inline void LoadOrderValid(void)
                        {
                                ulong value;

                                for (int c0 = OrdersTotal() - 1; (c0 >= 0) && (_LastError == ERR_SUCCESS); c0--)
                                {
                                        if ((value = OrderGetTicket(c0)) == 0) continue;
                                        if (OrderGetString(ORDER_SYMBOL) != _Symbol) continue;
                                        if (OrderGetInteger(ORDER_MAGIC) != GetMagicNumber()) continue;
                                        if (m_TicketPending > 0) SetUserError(ERR_Unknown); else m_TicketPending = value;
                                }
                        }
```

Let's see how this code works and why it looks so unusual. Here we use a loop to read all pending orders in the order book. The [OrdersTotal](https://www.mql5.com/en/docs/trading/orderstotal) function will return a value greater than zero if orders exist. Indexing always starts from zero. It came from C/C++. But we have two conditions for the loop to end: first, the value of the **c0** variable is less than zero and second - **\_LastError** is different form **ERR\_SUCESS**, which indicates some failure occurred in the EA.

Thus we enter the loop and capture the first order whose index is indicated by variable **c0**, [OrderGetTicket](https://www.mql5.com/en/docs/trading/ordergetticket) will return the ticket value or zero. If it is zero, we will return to the loop, but now we subtract one from the variable **c0**.

Since OrderGetTicket loads order values and the system does not distinguish between them, we will need to filter everything so that the EA only knows about our specific order. So, the first filter we will use is the asset name; for this we compare the asset in the order with the asset on which the EA is running. If they are different, the order will be ignored and we will return for another one, if there is any.

The next filter is the magic number, since the order book cab have orders placed manually or placed by other EAs. We can find out whether the order was placed by our EA based on the magic number which each EA should have. If the magic number differs from the one used by the EA, the order should be ignored. Then we will return to the beginning looking for a new order.

Now we come to crossroads. If the EA has found the order which it placed before it was removed from the chart for some reason (later we will see what the reasons can be), then its memory, i.e. the variable indicating the pending order ticket, will have a value other than zero. Then, if a second order is encountered, it will be considered an error. The function will use an enumeration to show that an error has occurred.

Here I use the common value **ERR\_Unknown**, but you can create a value to specify the error, which will be shown in the **\_LastError** value. The [SetUserError](https://www.mql5.com/en/docs/common/setusererror) function is responsible for setting the error value in the \_LastError variable. But if everything is ok and the variable containing the order ticket is set to zero, the value of the order found after all filters will be saved in the **m\_TicketPending** variable for further use. This is where we finished with the explanation of this procedure. Let's consider the next one which is responsible for searching for any open position. Its code is shown below:

```
inline void LoadPositionValid(void)
                        {
                                ulong value;

                                for (int c0 = PositionsTotal() - 1; (c0 >= 0) && (_LastError == ERR_SUCCESS); c0--)
                                {
                                        if ((value = PositionGetTicket(c0)) == 0) continue;
                                        if (PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
                                        if (PositionGetInteger(POSITION_MAGIC) != GetMagicNumber()) continue;
                                        if (m_Position.Ticket > 0) SetUserError(ERR_Unknown); else
					{
						m_Position.Ticket = value;
						SetInfoPositions();
					}
                                }
                        }
```

Everything I said about the previous code applies to this one as well. The only difference is that previously we manipulated orders and now we manipulate positions. But the logic is the same, only until the next call: **SetInfoPositions** which must store, correct and process the latest position data. To do this, we will use the following code:

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
                                m_Position.EnableBreakEven = (m_Position.IsBuy ? (v1 < v2) : (v1 > v2));
                                m_Position.Gap = FinanceToPoints(m_Trigger, m_Position.Leverage);

                                return m_Position.Leverage - tmp;
                        }
```

This code is especially interesting when it comes to working with the latest position data. But be careful: before calling it, you must update the position data with one of the following calls: [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), [PositionSelect](https://www.mql5.com/en/docs/trading/positionselect), [PositionGetSymbol](https://www.mql5.com/en/docs/trading/positiongetsymbol) or [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket). In general, here we initialize or configure everything as necessary. We had to place this code separately because we will use it at other points to update the position data whenever necessary.

That's basically it, but now we need to make a new modification to the class's constructor, so that the EA can be fully initialized correctly. All we need to do is add the calls shown above. Then the final code of the constructor will be as follows:

```
                C_Manager(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage, bool IsDayTrade, double Trigger)
                        :C_Orders(magic),
                        m_bAccountHedging(false),
                        m_TicketPending(0),
                        m_Trigger(Trigger)
                        {
                                string szInfo;

                                ResetLastError();
                                ZeroMemory(m_Position);
                                m_InfosManager.FinanceStop = FinanceStop;
                                m_InfosManager.FinanceTake = FinanceTake;
                                m_InfosManager.Leverage    = Leverage;
                                m_InfosManager.IsDayTrade  = IsDayTrade;
                                switch ((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE))
                                {
                                        case ACCOUNT_MARGIN_MODE_RETAIL_HEDGING:
                                                m_bAccountHedging = true;
                                                szInfo = "HEDGING";
                                                break;
                                        case ACCOUNT_MARGIN_MODE_RETAIL_NETTING:
                                                szInfo = "NETTING";
                                                break;
                                        case ACCOUNT_MARGIN_MODE_EXCHANGE:
                                                szInfo = "EXCHANGE";
                                                break;
                                }
                                Print("Detected Account ", szInfo);
                                LoadPositionValid();
                                LoadOrderValid();
                                if (_LastError == ERR_SUCCESS)
                                {
                                        szInfo = "Successful upload...";
                                        szInfo += StringFormat("%s", (m_Position.Ticket > 0 ? "\nTicket Position: " + (string)m_Position.Ticket : ""));
                                        szInfo += StringFormat("%s", (m_TicketPending > 0 ? "\nTicket Order: " + (string)m_TicketPending : ""));
                                        Print(szInfo);
                                }
                        }
```

These two lines will make the EA load the open position and the pending order. Now pay attention to the fact that the constructor cannot return any value, as this is an error. We need some way to tell the rest of the code that something went wrong in the constructor.

Every programming system provides or makes us create such means. But MQL5 provides a very practical way, which is using the **\_LastError** variable for this. If everything is fine during initialization, we will see a relevant message in the terminal. If the system has found any positions, we will also see a message indicating which position ticket the EA will observe. If the order was found, we will also see a message informing us about the pending order ticket found by the EA.

The **\_LastError** value will be used as a way to check if the EA has gone offline at some point. So it might be interesting if you add more types message types in the error list, in order to have a more precise indication of what actually happened.

### A problem with HEDGING accounts for an automated EA

Although everything looks beautiful and wonderful, especially for those who are starting to learn programming, we continue the development to achieve a greater level of robustness in an automated EA. We still have a potential problem in the system when used on HEDGING accounts. And we have this even before we proceed to the code responsible for allowing the EA to send orders or requests to the server. The problem lies in the fact that unlike a NETTING account, where the server creates an average price as the position is modified, either by entering new market orders or by executing pending orders, the HEDGING account does not have so much control, which is easy and coming from the server.

The problem with the HEDGING account is that we can have an open position, and if a pending order is executed, it will not change the open position directly. What can happen, and will in fact happen, is that a new position will be opened when the pending order is executed. This new open position can lock the price, so that we have neither profit nor loss. But it can also increase our overall position. This will happen as soon as the order is executed.

This detail, which exists in the HEDGING account, forces us to take another measure. We can prevent the EA from sending orders to the market in case a position is open or a pending order is already in the order book. This can be done easily based on the code I'm showing. But the problem is that during initialization, the EA may find an open position and a pending order on a HEDGING account. This is not a problem for the NETTING account, as I explained above.

What should the EA do in this case? As you remember, the C\_Manager class, which will control the EA, does not allow it to have two open positions or two pending orders. In this case, we will need to remove the pending order or close the open position. In one way or another, something must be done, because we must not allow this situation in an automated EA. I stress this again, an automated EA must never work with more than one open position at the same time, or more than one pending order at the same time. Things are different in a manual EA.

Therefore, you will have to decide which measure should be taken: Close the position or remove the pending order? In case you want to close the position, the C\_Orders class already offers the procedure for this. But if you need to delete the pending order, we don't have any appropriate procedure in the C\_Orders class. So we need to implement a way to do this. Let's start at this point, enabling the system to delete pending orders. To do this, we will add a new code to the system:

```
class C_Orders : protected C_Terminal
{
        protected:
//+------------------------------------------------------------------+
inline const ulong GetMagicNumber(void) const { return m_MagicNumber; }
//+------------------------------------------------------------------+
                void RemoveOrderPendent(const ulong ticket)
                        {
                                ZeroMemory(m_TradeRequest);
                                m_TradeRequest.action   = TRADE_ACTION_REMOVE;
                                m_TradeRequest.order    = ticket;
                                ToServer();
                        };

// ... The rest of the class code

}
```

Pay attention to some details in the code. First, it is located in the protected code part, i.e. even if we try to use the C\_Orders class in the EA directly, we will not have access to this code for the reason that I have already explained earlier. The second thing is that it is used to delete pending orders but is not used to close positions or to modify pending orders.

So, this code is already implemented in the C\_Orders class. We can get back to C\_Manager and implement a system to prevent the automated EA from having a pending order if it is running on a HEDGING account and when it already has an open position. But if you want it to close the position and keep the pending order, it will be enough to make changes in the code to have the desired behavior. The only thing that cannot happen is that the automated EA running on a HEDGING account has both an open position and a pending order. This cannot be allowed.

**Important:** If it is a HEDGING account, you can run more than one EA on the same asset. If this happens, the fact that one EA has an open position and the other has a pending order will not affect the operation of both EAs in any way. In this case, they are independent. So, it may happen that we have more than one open position or more than one pending order on the same asset. This situation is not possible for a single EA. Also, this only concerns automated EAs. I will keep repeating this as this is extremely important to understand and remember.

You might have noticed that in the constructor code, we first capture a position and only then we capture the order. This makes it simpler to remove the order, if necessary. However, if you want to close the position and keep the order, just invert this in the constructor, so that first the orders will be captured and then the position. Then, if there is a need, the position will be closed. Let's see how we'll do it in the case we are considering. Capture the position and then, if necessary, remove any pending orders found. The code for this is seen below:

```
inline void LoadOrderValid(void)
                        {
                                ulong value;

                                for (int c0 = OrdersTotal() - 1; (c0 >= 0) && (_LastError == ERR_SUCCESS); c0--)
                                {
                                        if ((value = OrderGetTicket(c0)) == 0) continue;
                                        if (OrderGetString(ORDER_SYMBOL) != _Symbol) continue;
                                        if (OrderGetInteger(ORDER_MAGIC) != GetMagicNumber()) continue;
                                        if ((m_bAccountHedging) && (m_Position.Ticket > 0))
                                        {
                                                RemoveOrderPendent(value);
                                                continue;
                                        }
                                        if (m_TicketPending > 0) SetUserError(ERR_Unknown); else m_TicketPending = value;
                                }
                        }
```

The change we need to make is to add the highlighted code. Notice how easy it is to delete a pending order, but here all we need to do is remove the order. If we have an open position and the account type is HEDGING, then a situation occurs when the pending order will be deleted. But if we have a NETTING account or there is no open position, then this code will not be executed, which will allow the EA to work smoothly.

However, since you may want to close the position and keep the pending order, let's see what the code should look like in this case. You don't need to change the pending order loading code — use the one shown above. But you will need to make some changes, and the first of them is to add the following code to the procedure that loads the position:

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
                                                ClosePosition(value);
                                                continue;
                                        }
                                        if (m_Position.Ticket > 0) SetUserError(ERR_Unknown); else
					{
						m_Position.Ticket = value;
						SetInfoPositions();
					}
                                }
                        }
```

By adding the highlighted code, you will be able to close the open position while keeping the pending order. But there is one detail here: In order to keep a pending order and close a position on the HEDGING account, we need to change one item in the constructor code as follows:

```
                C_Manager(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage, bool IsDayTrade, double Trigger)
                        :C_Orders(magic),
                        m_bAccountHedging(false),
                        m_TicketPending(0),
                        m_Trigger(Trigger)
                        {
                                string szInfo;

                                ResetLastError();
                                ZeroMemory(m_Position);
                                m_InfosManager.FinanceStop = FinanceStop;
                                m_InfosManager.FinanceTake = FinanceTake;
                                m_InfosManager.Leverage    = Leverage;
                                m_InfosManager.IsDayTrade  = IsDayTrade;
                                switch ((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE))
                                {
                                        case ACCOUNT_MARGIN_MODE_RETAIL_HEDGING:
                                                m_bAccountHedging = true;
                                                szInfo = "HEDGING";
                                                break;
                                        case ACCOUNT_MARGIN_MODE_RETAIL_NETTING:
                                                szInfo = "NETTING";
                                                break;
                                        case ACCOUNT_MARGIN_MODE_EXCHANGE:
                                                szInfo = "EXCHANGE";
                                                break;
                                }
                                Print("Detected Account ", szInfo);
                                LoadOrderValid();
                                LoadPositionValid();
                                if (_LastError == ERR_SUCCESS)
                                {
                                        szInfo = "Successful upload...";
                                        szInfo += StringFormat("%s", (m_Position.Ticket > 0 ? "\nTicket Position: " + (string)m_Position.Ticket + "\n" : ""));
                                        szInfo += StringFormat("%s", (m_TicketPending > 0 ? "\nTicket Order: " + (string)m_TicketPending : ""));
                                        Print(szInfo);
                                }
                        }
```

You may not have noticed any changes. But if you compare it with the code at the end of the previous section, you will see that the highlighted part is different. In this case, the position will be closed while in the previous version we were deleting the order. This is the grace of programming. Sometimes a simple detail makes all the difference. Here we only change the order of code execution, but the result is completely different.

**In theory**, the code considered so far does not have any kind of problem and will work perfectly. But this is in theory. It may turn out that the trading server reports an error, not because there is something wrong with the requests being sent, but due to some kind of interaction which can happen. The **\_LastError** variable will contain a value indicating some kind of failure.

Some failures can be allowed as they are not critical while others cannot be ignored. If you understand this difference and accept this idea, you can add the [ResetLastError](https://www.mql5.com/en/docs/common/resetlasterror) call in certain code parts to prevent the EA from being thrown out of the chart because there was some kind of error which, most likely, was not caused by the EA but by incorrect interaction between the EA and the trade server.

At this early stage, I will not show where you can add these calls. I do this so that you don't get tempted to make these calls indiscriminately at any point or to ignore the values contained in the **\_LastError** variable. This would break the whole thesis of building a strong, robust and reliable automated EA.

### Conclusion

In this article, I have presented the very basics to show you that you should always think about how to automate an EA in a safe, stable and robust way. Programming an EA that will run automatically is not a task that people with little experience can do without problems, but it is an extremely difficult task that requires great care on the part of the programmer.

In the next article, we will consider some more things that need to be implemented to get an automated EA. We will consider how to make it safe to be placed on the chart. We must always act with due care and correct measures so as not to cause any damage to our hard-earned heritage.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11241](https://www.mql5.com/pt/articles/11241)

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
**[Go to discussion](https://www.mql5.com/en/forum/443536)**
(2)


![arikmt5](https://c.mql5.com/avatar/avatar_na2.png)

**[arikmt5](https://www.mql5.com/en/users/arikmt5)**
\|
14 Mar 2023 at 08:39

could you share the [source code](https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development")? many thanks.


![Hope](https://c.mql5.com/avatar/avatar_na2.png)

**[Hope](https://www.mql5.com/en/users/5010664539)**
\|
16 Mar 2023 at 08:12

That was useful.  Thank you Daniel Jose


![MQL5 Cookbook — Macroeconomic events database](https://c.mql5.com/2/51/mql5-recipes-database.png)[MQL5 Cookbook — Macroeconomic events database](https://www.mql5.com/en/articles/11977)

The article discusses the possibilities of handling databases based on the SQLite engine. The CDatabase class has been formed for convenience and efficient use of OOP principles. It is subsequently involved in the creation and management of the database of macroeconomic events. The article provides the examples of using multiple methods of the CDatabase class.

![Data Science and Machine Learning (Part 12): Can Self-Training Neural Networks Help You Outsmart the Stock Market?](https://c.mql5.com/2/52/Self-Training-Neural-Networks-avatar.png)[Data Science and Machine Learning (Part 12): Can Self-Training Neural Networks Help You Outsmart the Stock Market?](https://www.mql5.com/en/articles/12209)

Are you tired of constantly trying to predict the stock market? Do you wish you had a crystal ball to help you make more informed investment decisions? Self-trained neural networks might be the solution you've been looking for. In this article, we explore whether these powerful algorithms can help you "ride the wave" and outsmart the stock market. By analyzing vast amounts of data and identifying patterns, self-trained neural networks can make predictions that are often more accurate than human traders. Discover how you can use this cutting-edge technology to maximize your profits and make smarter investment decisions.

![Neural networks made easy (Part 33): Quantile regression in distributed Q-learning](https://c.mql5.com/2/50/Neural_Networks_Made_Easy_q-learning_avatar.png)[Neural networks made easy (Part 33): Quantile regression in distributed Q-learning](https://www.mql5.com/en/articles/11752)

We continue studying distributed Q-learning. Today we will look at this approach from the other side. We will consider the possibility of using quantile regression to solve price prediction tasks.

![Population optimization algorithms: Bacterial Foraging Optimization (BFO)](https://c.mql5.com/2/51/bacterial-optimization-avatar.png)[Population optimization algorithms: Bacterial Foraging Optimization (BFO)](https://www.mql5.com/en/articles/12031)

E. coli bacterium foraging strategy inspired scientists to create the BFO optimization algorithm. The algorithm contains original ideas and promising approaches to optimization and is worthy of further study.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11241&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069166264196202773)

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