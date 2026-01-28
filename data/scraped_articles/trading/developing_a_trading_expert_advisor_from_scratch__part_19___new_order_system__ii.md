---
title: Developing a trading Expert Advisor from scratch (Part 19): New order system (II)
url: https://www.mql5.com/en/articles/10474
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:47:27.048724
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/10474&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051685442358727853)

MetaTrader 5 / Trading


### Introduction

In the previous article, [Developing a trading Expert Advisor from scratch (Part 18)](https://www.mql5.com/en/articles/10462), we implemented some fixes, changes and adjustments in the order system, aiming at creating a system that would allow different trading on NETTING and HEDGING accounts, since there are differences in the account operations. For the NETTING type, the system generates an average price, and you have only one open position for an asset. On HEDGING accounts, you can have multiple open positions, each of which has individual limits. You can buy and sell the same assets at the same time. This can only be done on HEDGING accounts. This is the foundation, based on which options trading can be understood.

But now it is time to finally make the order system completely visual so that we can eliminate the message box and analyze what values are in each position without it. We can do this just by looking at the new order system. This will allow us to adjust several things at a time. Also, we will be able to easily know the profit and loss limits of an OCO position or a pending OCO order, since the EA will display the relevant information in real time, not requiring any extra calculations.

Although this is the first part of the implementation, we are not starting from scratch: we will modify the existing system by adding more objects and events to the chart of the asset we are trading.

### 1.0. Planning

Planning of the system we are using here is not particularly difficult: we will modify the existing system by changing only the system that represents orders on the chart. This is the main idea which seems quite simple. But practically it requires much creativity, since we are going to manipulate and model data so that the MetaTrader 5 platform does all the hard work for us.

There are several ways to model data, each having its pros and cons.

- The first way is to use a list. It can be a cyclic single, a cyclic double, or even a hashing system. The advantage of using any of these approaches is that the system is easy to implement. However, the disadvantage is that this will prevent data manipulation or will limit the number of orders. Furthermore, in this case, we would have to create all the additional logic just to save the list.
- The second way is to create an array of classes, while the class will contain and maintain all newly created objects. In this case, the array will work like a list, but we have to write less code, because MQL5 already supports a few things that we would have to code in the case of using a list. However, we would have other problems, such as event handling, which in this situation would be quite difficult.
- The third way is the one we are going to use. We will force the code created in MQL5 to support dynamic objects. This seems like something unreal, but if we do the right modeling of the data to be used, then the MQL5 language will enable us to create a system in which there will be no restrictions on the number of objects on the screen. Furthermore, all the objects will be able to generate and receive events. And despite their individuality, the platform will see them all linked as if they were in a list or in an array index.

If you think that this is not easy to implement, take a look at the following code part of the C\_HLineTrade class:

```
inline void SetLineOrder(ulong ticket, double price, eHLineTrade hl, bool select)
{
        string sz0 = def_NameHLineTrade + (string)hl + (string)ticket, sz1;

        ObjectCreate(Terminal.Get_ID(), sz0, OBJ_HLINE, 0, 0, 0);

//... The rest of the code....
```

The highlighted part shows exactly that we can create as many horizontal lines as we want, and they will receive events in a completely independent way. All we need to do is to implement events based on the name that each of the lines will have, since the names will be unique. The MetaTrader 5 platform will take care of the rest. The result will look something like this:

![](https://c.mql5.com/2/45/ScreenRecorderProject42_a13.gif)

Although this already seems like something ideal, this modeling will not be enough to achieve the result we really need. The idea can be implemented. But the data modeling currently available in the EA is not ideal, because we cannot have an unlimited number of objects based on one name. We need to make some changes that require a fairly deep code modification.

We will now start to implement this new data modeling method, but we will only change what is necessary for this, while maintaining the entire code stable, because it should continue to work as steadily as possible. All work will be performed by the MetaTrader 5 platform, we will only indicate how the platform should understand our modeling.

### 2.0. Implementation

The first modification is that we change C\_HLineTrade into the new C\_ObjectsTrade class. This new class will be able to support what we need—a way to link an unlimited number of objects.

Let's start by looking at the original definitions in the following code.

```
class C_ObjectsTrade
{
//+------------------------------------------------------------------+
#define def_NameObjectsTrade 	"SMD_OT"
#define def_SeparatorInfo       '*'
#define def_IndicatorTicket0    1
//+------------------------------------------------------------------+
        protected:
                enum eIndicatorTrade {IT_NULL, IT_STOP= 65, IT_TAKE, IT_PRICE};
//+------------------------------------------------------------------+

// ... The rest of the class code
```

Here we have the initial base which we are going to implement. It will be expanded in the future, but for now I want the system to remain stable despite it is being modified and has new data modeling.

Even within the 'protected' declaration, we have the following functions:

```
inline double GetLimitsTake(void) const { return m_Limits.TakeProfit; }
//+------------------------------------------------------------------+
inline double GetLimitsStop(void) const { return m_Limits.StopLoss; }
//+------------------------------------------------------------------+
inline bool GetLimitsIsBuy(void) const { return m_Limits.IsBuy; }
//+------------------------------------------------------------------+
inline void SetLimits(double take, double stop, bool isbuy)
{
        m_Limits.IsBuy = isbuy;
        m_Limits.TakeProfit = (m_Limits.TakeProfit < 0 ? take : (isbuy ? (m_Limits.TakeProfit > take ? m_Limits.TakeProfit : take) : (take > m_Limits.TakeProfit ? m_Limits.TakeProfit : take)));
        m_Limits.StopLoss = (m_Limits.StopLoss < 0 ? stop : (isbuy ? (m_Limits.StopLoss < stop ? m_Limits.StopLoss : stop) : (stop < m_Limits.StopLoss ? m_Limits.StopLoss : stop)));
}
//+------------------------------------------------------------------+
inline int GetBaseFinanceLeveRange(void) const { return m_BaseFinance.Leverange; }
//+------------------------------------------------------------------+
inline int GetBaseFinanceIsDayTrade(void) const { return m_BaseFinance.IsDayTrade; }
//+------------------------------------------------------------------+
inline int GetBaseFinanceTakeProfit(void) const { return m_BaseFinance.FinanceTake; }
//+------------------------------------------------------------------+
inline int GetBaseFinanceStopLoss(void) const { return m_BaseFinance.FinanceStop; }
```

Currently, these functions just serve as a security measure for another scheme that we will implement in the future. Even though we can implement the data and the parsing performed in another location, it is good to leave some things as low in the inheritance chain as possible. Even if the return values will only be used by derived classes, I do not want to allow this directly: I do not want the derived class to access the values that are inside this C\_ObjectsTrade object class, because that would break the idea of object class encapsulation, making it difficult for future modifications or bug fixes, if the derived class changes the value of the base class without making the relevant changes through a procedure call.

To minimize call overlap as much as possible, all functions are declared inline: this slightly increases the size of the executable, but results in a more secure system.

Now we come to private declarations.

```
//+------------------------------------------------------------------+
        private :
                string  m_SelectObj;
                struct st00
                {
                        double  TakeProfit,
                                StopLoss;
                        bool    IsBuy;
                }m_Limits;
                struct st01
                {
                        int     FinanceTake,
                                FinanceStop,
                                Leverange;
                        bool    IsDayTrade;
                }m_BaseFinance;
//+------------------------------------------------------------------+
                string MountName(ulong ticket, eIndicatorTrade it)
                {
                        return StringFormat("%s%c%c%c%d", def_NameObjectsTrade, def_SeparatorInfo, (char)it, def_SeparatorInfo, ticket);
                }
//+------------------------------------------------------------------+
```

The most important part is the highlighted fragment, which will model the names of the objects. I am keeping the basics which are still available in the system. This is because we first create and modify the modeling, keeping the system stable. Then we will add new objects, while this will be done quite easily, quickly. Furthermore, we will maintain the stability already achieved.

Although the code has undergone many more changes than shown here, I will only focus on new functions as well as the changes that were considerable compared to previous codes.

The first function is shown below:

```
inline string CreateIndicatorTrade(ulong ticket, eIndicatorTrade it, bool select)
{
        string sz0 = MountName(ticket, it);

        ObjectCreate(Terminal.Get_ID(), sz0, OBJ_HLINE, 0, 0, 0);
        ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_COLOR, (it == IT_PRICE ? clrBlue : (it == IT_STOP ? clrFireBrick : clrForestGreen)));
        ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_WIDTH, 1);
        ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_STYLE, STYLE_DASHDOT);
        ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_SELECTABLE, select);
        ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_SELECTED, false);
        ObjectSetInteger(Terminal.Get_ID(), sz0, OBJPROP_BACK, true);
        ObjectSetString(Terminal.Get_ID(), sz0, OBJPROP_TOOLTIP, (string)ticket + " "+StringSubstr(EnumToString(it), 3, 10));

        return sz0;
}
```

For now, it will only create a horizontal line. Pay attention to the name generation code; also note that the colors will now be defined internally by the code and not by the user.

Then we overload the same function, as can be seen below.

```
inline string CreateIndicatorTrade(ulong ticket, double price, eIndicatorTrade it, bool select)
{
        if (price <= 0)
        {
                RemoveIndicatorTrade(ticket, it);
                return NULL;
        }
        string sz0 = CreateIndicatorTrade(ticket, it, select);
        ObjectMove(Terminal.Get_ID(), sz0, 0, 0, price);

        return sz0;
}
```

Do not confuse these two functions, because although they appear to be the same, they are actually different. Overload is quite common: we create a simple function and then add new parameters to it to accumulate a certain type of modeling. If we did not implement it via overloading, we would sometimes have to repeat the same code sequence. This is dangerous, because we can forget to declare something. Also, it is not very practical, so we overload the function to make one call instead of several.

One thing that should be mentioned here is the part that is highlighted in this second version. There is no need to create it here, we could do it in another place. But, as can be seen, when we try to create some object with the zero price, in fact it must be destroyed.

To actually see the moment this happens, take a look at the code below:

```
class C_Router : public C_ObjectsTrade
{

// ... Internal class code ....

                void UpdatePosition(int iAdjust = -1)
                        {

// ... Internal function code ...

                                for(int i0 = p; i0 >= 0; i0--) if(PositionGetSymbol(i0) == Terminal.GetSymbol())
                                {
                                        ul = PositionGetInteger(POSITION_TICKET);
                                        m_bContainsPosition = true;
                                        CreateIndicatorTrade(ul, PositionGetDouble(POSITION_PRICE_OPEN), IT_PRICE, false);
                                        CreateIndicatorTrade(ul, take = PositionGetDouble(POSITION_TP), IT_TAKE, true);
                                        CreateIndicatorTrade(ul, stop = PositionGetDouble(POSITION_SL), IT_STOP, true);

// ... The rest of the code...
```

Every time the EA receives the OnTrade event, it will execute the above function and will try to create an indicator on the selected points, but if the user removes the limit, it will become zero. Therefore, when called, it will actually delete the indicator from the chart, saving us from useless objects in memory. Thus we have gain at some points, since the check will be done right at the moment of creation.

But we still have a problem with overloading, because some people may not fully understand how it is used in real code. To understand this, take a look at the two code parts below:

```
class C_OrderView : public C_Router
{
        private  :
//+------------------------------------------------------------------+
        public   :
//+------------------------------------------------------------------+
                void InitBaseFinance(int nContracts, int FinanceTake, int FinanceStop, bool b1)
                        {
                                SetBaseFinance(nContracts, FinanceTake, FinanceStop, b1);
                                CreateIndicatorTrade(def_IndicatorTicket0, IT_PRICE, false);
                                CreateIndicatorTrade(def_IndicatorTicket0, IT_TAKE, false);
                                CreateIndicatorTrade(def_IndicatorTicket0, IT_STOP, false);
                        }
//+------------------------------------------------------------------+

// ... Rest of the code...
class C_Router : public C_ObjectsTrade
{

// ... Class code ...

                void UpdatePosition(int iAdjust = -1)
                        {
// ... Function code ....
                                for(int i0 = p; i0 >= 0; i0--) if(PositionGetSymbol(i0) == Terminal.GetSymbol())
                                {
                                        ul = PositionGetInteger(POSITION_TICKET);
                                        m_bContainsPosition = true;
                                        CreateIndicatorTrade(ul, PositionGetDouble(POSITION_PRICE_OPEN), IT_PRICE, false);

// ... The rest of the code...
```

Note that in both cases we have the same name of the function being used. Also, they are both part of the same C\_ObjectsTrade class. However, even in this case the compiler can distinguish between them, which is because of the number of parameters. If you look closely, you will see that the only difference is an additional 'price' parameter, but there may also be some others. As you can see, it is much easier to use one call to copy all the code that is present in one of the overloaded versions, so in the end we have cleaner code which is easier to maintain.

Now let us get back to the C\_ObjectsTrade class. The next function we need to understand looks like this:

```
bool GetInfosOrder(const string &sparam, ulong &ticket, double &price, eIndicatorTrade &it)
{
        string szRet[];
        char szInfo[];

        if (StringSplit(sparam, def_SeparatorInfo, szRet) < 2) return false;
        if (szRet[0] != def_NameObjectsTrade) return false;
        StringToCharArray(szRet[1], szInfo);
        it = (eIndicatorTrade)szInfo[0];
        ticket = (ulong) StringToInteger(szRet[2]);
        price = ObjectGetDouble(Terminal.Get_ID(), sparam, OBJPROP_PRICE);

        return true;
}
```

In fact, it is the heart, mind and body of the entire new system. Although it seems quite simple, it does a job that is essential for the entire EA to function as our new modeling system requires it.

Pay close attention to the highlighted code, in particular to the [StringSplit](https://www.mql5.com/en/docs/strings/stringsplit) function. If it did not exist in MQL5, we would have to code it. Fortunately, MQL5 has it, so we will use this function to the fullest. What it does is decompose the name of the object into the required data. When an object name is created, it is modeled in a very specific way, and due to this we can undo this coding model so StringSplit will undo what the [StringFormat](https://www.mql5.com/en/docs/convert/stringformat) function does.

The rest of the function captures the data present in the object name so we can test it and use it later. That is, MetaTrader 5 generates the data for us, we decompose it in order to know what happened and then tell MetaTrader 5 which steps it should take. Our purpose is to make MetaTrader 5 work for us. I do not create a model from scratch; instead, I am modeling the interface and the EA from scratch. Therefore, we should benefit from the support offered by MetaTrader 5 instead of looking for an external solution.

In the code below we will do something very similar to what we did above:

```
inline void RemoveAllsIndicatorTrade(bool bFull)
{
        string sz0, szRet[];
        int i0 = StringLen(def_NameObjectsTrade);

        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
        for (int c0 = ObjectsTotal(Terminal.Get_ID(), -1, -1); c0 >= 0; c0--)
        {
                sz0 = ObjectName(Terminal.Get_ID(), c0, -1, -1);
                if (StringSubstr(sz0, 0, i0) == def_NameObjectsTrade)
                {
                        if (!bFull)
                        {
                                StringSplit(sz0, def_SeparatorInfo, szRet);
                                if (StringToInteger(szRet[2]) == def_IndicatorTicket0) continue;
                        }
                }else continue;
                ObjectDelete(Terminal.Get_ID(), sz0);
        }
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
}
```

Every time we remove a line from the chart, whether it is a position that will be closed or a limit level that will be removed, the corresponding object must be removed, just like when the EA is removed from the chart. We need to delete the objects, but we also have a set of lines that should not be deleted unless absolutely necessary: this is Ticket0, it should not be deleted unless extremely necessary. To avoid the deletion, let us use the highlighted code. Without this, we would need to create this Ticket0 anew every time, because this ticket is very important in another code part which we will discuss later.

In all other times we need to delete something specific. For this, we will use another removal function which is shown below.

```
inline void RemoveIndicatorTrade(ulong ticket, eIndicatorTrade it = IT_NULL)
{
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
        if ((it != NULL) && (it != IT_PRICE))
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, it));
        else
        {
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, IT_PRICE));
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, IT_TAKE));
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, IT_STOP));
        }
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
}
```

The next new routine can be seen below:

```
inline void PositionAxlePrice(double price, ulong ticket, eIndicatorTrade it, int FinanceTake, int FinanceStop, int Leverange, bool isBuy)
{
        double ad = Terminal.GetAdjustToTrade() / (Leverange * Terminal.GetVolumeMinimal());
        ObjectMove(Terminal.Get_ID(), MountName(ticket, it), 0, 0, price);
        if (it == IT_PRICE)
        {
                ObjectMove(Terminal.Get_ID(), MountName(ticket, IT_TAKE), 0, 0, price + Terminal.AdjustPrice(FinanceTake * (isBuy ? ad : (-ad))));
                ObjectMove(Terminal.Get_ID(), MountName(ticket, IT_STOP), 0, 0, price + Terminal.AdjustPrice(FinanceStop * (isBuy ? (-ad) : ad)));
        }
}
```

It will place objects on the price axis. But don't get too attached to it, as it will soon cease to exist for various reasons. Among them is the one which we discussed in another article of this series: [Multiple indicators on one chart (Part 05): Converting MetaTrader 5 into RAD(I) system](https://www.mql5.com/en/articles/10277). This article has a table showing objects that can use Cartesian coordinates for positioning, and these coordinates are X and Y. Price and time coordinates, despite being useful in some cases, are not always convenient: when we want to position elements that have to be positioned at certain points on the screen, although it will be faster to develop things using price and time coordinates, they are much more difficult to work with than the X and Y system.

We will make changes next time, while now our purpose is to create an alternative system to the one used so far.

Next, we have the last important function in the C\_ObjectsTrade class. It is shown in the following code:

```
inline double GetDisplacement(const bool IsBuy, const double Vol, eIndicatorTrade it) const
{
        int i0 = (it == IT_TAKE ? m_BaseFinance.FinanceTake : m_BaseFinance.FinanceStop),
            i1 = (it == IT_TAKE ? (IsBuy ? 1 : -1) : (IsBuy ? -1 : 1));
        return (Terminal.AdjustPrice(i0 * (Vol / m_BaseFinance.Leverange) * Terminal.GetAdjustToTrade() / Vol) * i1);
}
```

This function will make conversion between the values specified in the Chart Trader for a pending order to be placed or a position that will be opened by market.

All these changes have been implemented to transform the C\_HLineTrade function into C\_ObjectsTrade. However, these changes also required some other changes. For example, the class that also has changed considerably is C\_ViewOrder. Some parts of this class simply ceased to exist, because there is no point in their existence, while the remaining functions have been changed. The functions that deserve special attention are highlighted below.

The first one is the function for initializing the data coming from the Chart Trader.

```
void InitBaseFinance(int nContracts, int FinanceTake, int FinanceStop, bool b1)
{
        SetBaseFinance(nContracts, FinanceTake, FinanceStop, b1);
        CreateIndicatorTrade(def_IndicatorTicket0, IT_PRICE, false);
        CreateIndicatorTrade(def_IndicatorTicket0, IT_TAKE, false);
        CreateIndicatorTrade(def_IndicatorTicket0, IT_STOP, false);
}
```

The highlighted parts are where Ticket0 is actually created. This ticket is used to place a pending order using the mouse and keyboard: (SHIFT) to buy, (CTRL) to sell. Previously, lines were created at this point, which were then used to indicate where the order would be located. Now things are much simpler: the same as we see an order to be placed, we will also see a pending order or an open position. It means that we will always be checking the system. It is like if you were to assemble a vehicle and all the time you were checking its brakes so that when you actually have to use them you would know how it would behave.

The big problem with a lengthy code is that when we create function, we can only know that it is working at the time it is actually used. But now the system is always checked — even if we do not use all the functions, they are constantly being checked due to the code reuse in different places.

The last routine that I will mention in this article is shown below. It will place a pending order. Note that it has become extremely compact compared to the same function in previous articles.

```
inline void MoveTo(uint Key)
{
        static double local = 0;
        datetime dt;
        bool    bEClick, bKeyBuy, bKeySell, bCheck;
        double  take = 0, stop = 0, price;

        bEClick  = (Key & 0x01) == 0x01;    //Let mouse button click
        bKeyBuy  = (Key & 0x04) == 0x04;    //Pressed SHIFT
        bKeySell = (Key & 0x08) == 0x08;    //Pressed CTRL
        Mouse.GetPositionDP(dt, price);
        if (bKeyBuy != bKeySell)
        {
                Mouse.Hide();
                bCheck = CheckLimits(price);
        } else Mouse.Show();
        PositionAxlePrice((bKeyBuy != bKeySell ? price : 0), def_IndicatorTicket0, IT_PRICE, (bCheck ? 0 : GetBaseFinanceTakeProfit()), (bCheck ? 0 : GetBaseFinanceStopLoss()), GetBaseFinanceLeveRange(), bKeyBuy);
        if((bEClick) && (bKeyBuy != bKeySell) && (local == 0)) CreateOrderPendent(bKeyBuy, local = price);
        local = (local != price ? 0 : local);
}
```

The reason is that now there will be a new rule in the system, so the function has "lost some weight" and has become more compact.

### Conclusion

I have presented here some changes that will be used in the next article. The purpose of all this is to make them simpler and to show things that can be different at different times. My idea is that everyone follows and learns how to program an EA that will be used to help you with operations, which is why I do not just present a finished and ready-to-use system. I want to show that there are problems to be solved, and to present the path that I took to solve the issues and problems that arise during development. I hope you understand this point. Because if the idea was to create a system and present it in a ready form, I would better do so and sell the idea, but this is not my intention...

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10474](https://www.mql5.com/pt/articles/10474)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10474.zip "Download all attachments in the single ZIP archive")

[EA.zip](https://www.mql5.com/en/articles/download/10474/ea.zip "Download EA.zip")(12023.87 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/430952)**
(1)


![QQ1171513819 微信AT5050](https://c.mql5.com/avatar/2022/9/632149e5-5250.jpg)

**[QQ1171513819 微信AT5050](https://www.mql5.com/en/users/qq1171513819)**
\|
14 Sep 2022 at 03:43

ATFX [gold](https://www.mql5.com/en/quotes/metals/XAUUSD "XAUUSD chart: technical analysis") foreign exchange platform recruiting agents

ATFX free to open personal/company agents, exclusive agent background, commission is returned. UK FCA full licence foreign exchange brokers recruiting agents

Advantage\] 12 foreign offices in 11 countries around the world, across Europe, America, Africa and Asia.

\[Advantage\] platform stability, top liquidity offer. STP mode, a number of well-known liquidity provider LP offer, fast turnover, not chucking not dropped, not slippage, hedging, dial scalp, second single, EA, etc. without any trading restrictions.

Advantage\] UK FCA top foreign exchange licence, Cyprus stp licence. Mauritius financial licence, Abu Dhabi foreign exchange licence, St. Vincent foreign exchange financial licence and other multiple regulation. Capital security, the regular platform to support long-term large capital profit out of the gold.

\[Advantage\] fast and safe a variety of public access to money: wire transfer, UnionPay, public, Alipay, WeChat and so on.

\[Advantage\] variety of full: foreign exchange, gold, crude oil, index, U.S. stocks, Hong Kong stocks and other 200 kinds of trading varieties

\[Advantage\] ultra-low spread gold 0.35, Europe and the United States 0.18. base commission gold / index / crude oil / per hand 14/15/16 U.S. dollars per hand; foreign exchange 8/9/10 U.S. dollars per hand. mt4 operation

ATFX agent consulting: 85292029084 (WeChat) QQ: 1171513819 WeChat: AT5050

![Developing a trading Expert Advisor from scratch (Part 20): New order system (III)](https://c.mql5.com/2/47/development__3.png)[Developing a trading Expert Advisor from scratch (Part 20): New order system (III)](https://www.mql5.com/en/articles/10497)

We continue to implement the new order system. The creation of such a system requires a good command of MQL5, as well as an understanding of how the MetaTrader 5 platform actually works and what resources it provides.

![Metamodels in machine learning and trading: Original timing of trading orders](https://c.mql5.com/2/42/yandex_catboost__4.png)[Metamodels in machine learning and trading: Original timing of trading orders](https://www.mql5.com/en/articles/9138)

Metamodels in machine learning: Auto creation of trading systems with little or no human intervention — The model decides when and how to trade on its own.

![Data Science and Machine Learning — Neural Network (Part 02): Feed forward NN Architectures Design](https://c.mql5.com/2/48/forward_neural_network_design.png)[Data Science and Machine Learning — Neural Network (Part 02): Feed forward NN Architectures Design](https://www.mql5.com/en/articles/11334)

There are minor things to cover on the feed-forward neural network before we are through, the design being one of them. Let's see how we can build and design a flexible neural network to our inputs, the number of hidden layers, and the nodes for each of the network.

![Learn how to design a trading system by Bull's Power](https://c.mql5.com/2/48/why-and-how__5.png)[Learn how to design a trading system by Bull's Power](https://www.mql5.com/en/articles/11327)

Welcome to a new article in our series about learning how to design a trading system by the most popular technical indicator as we will learn in this article about a new technical indicator and how we can design a trading system by it and this indicator is the Bull's Power indicator.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/10474&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051685442358727853)

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