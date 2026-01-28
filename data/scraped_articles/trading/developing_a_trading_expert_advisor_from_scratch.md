---
title: Developing a trading Expert Advisor from scratch
url: https://www.mql5.com/en/articles/10085
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:48:54.922785
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/10085&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051705246452929797)

MetaTrader 5 / Trading


### Introduction

The number of new users in financial market increases. Perhaps many of them even do not know how the order system works. However, there are also the users who really want to know what is happening. They try to understand how it all works in order to control the situation.

Of course, MetaTrader 5 provides a high level of control over trading positions. However, using only the manual ability to place orders can be quite difficult and risky for less experienced users. Furthermore, if someone wants to trade futures contracts, when there is very little time to place an order, such trading can turn into a nightmare, because you have to correctly fill all the fields in a timely manner, but this still takes time and thus you can miss good opportunities or even lose money if something is filled incorrectly.

Now, what if we used an Expert Advisor (EA) to make things easier? In this case, you can specify some details, for example the leverage or how much you are can afford to lose and how much you want to earn (in monetary terms, rather than not quite clear "points" or "pips"). Then use the mouse pointer on the chart to show where to enter into the market and indicate, whether it will be buying or selling...

### Planning

The most difficult part in creating something is to figure out how the things should work. The idea should be formulated very clearly so that we will need to create the minimum required code, as the more complex the code to be created, the greater the possibility of runtime errors. With this in mind, I tried to make code quite simple but still using the maximum of the possibilities provided by MetaTrader 5. The platform is very reliable, it is constantly being tested and thus there cannot be errors on the platform side.

The code will use OOP (object-oriented programming). This approach allows isolating the code and facilitates its maintenance and future development, in case we want to add new features and make improvements.

Although the EA discussed in this article is designed for trading on B3 (Brazilian Exchange) and specifically for trading futures (Mini Index and Mini Dollar), it can be extended into all markets with minimum changes. To make things easier and not to list or check trading assets, we will use the following enumeration:

```
enum eTypeSymbolFast {WIN, WDO, OTHER};
```

If you want to trade another asset, which uses some special feature, add it into the enumeration. This will also require small changes in the code, but with the enumeration this will be much easier as it also reduces the possibility of errors. An interesting part of code is the AdjustPrice function:

```
   double AdjustPrice(const double arg)
     {
      double v0, v1;
      if(m_Infos.TypeSymbol == OTHER)
         return arg;
      v0 = (m_Infos.TypeSymbol == WDO ? round(arg * 10.0) : round(arg));
      v1 = fmod(round(v0), 5.0);
      v0 -= ((v1 != 0) || (v1 != 5) ? v1 : 0);
      return (m_Infos.TypeSymbol == WDO ? v0 / 10.0 : v0);
     };
```

This function will adjust the value to be used in the price, in order to position the lines at accurate points of the chart. Why can't we simply put a line on the chart? This is because some assets have certain steps between prices. For WDO (Mini Dollar) this step is only 0.5 points. For WIN (Mini Index) this step is 5 points, and for stocks it is 0.01 points. In other words, point values differ for different assets. This adjusts the price to the correct tick value so that a proper value is used in the order, otherwise an incorrectly filled order can be rejected by the server.

Without this function, it might be difficult to know the correct values to be used in the order. And thus there is a chance for the server to notify that the order is filled incorrectly and to prevent it from being executed. Now, let's proceed to the function which is the heart of the Expert Advisor: CreateOrderPendent. The function is as follows:

```
   ulong CreateOrderPendent(const bool IsBuy, const double Volume, const double Price, const double Take, const double Stop, const bool DayTrade = true)
     {
      double last = SymbolInfoDouble(m_szSymbol, SYMBOL_LAST);
      ZeroMemory(TradeRequest);
      ZeroMemory(TradeResult);
      TradeRequest.action        = TRADE_ACTION_PENDING;
      TradeRequest.symbol        = m_szSymbol;
      TradeRequest.volume        = Volume;
      TradeRequest.type          = (IsBuy ? (last >= Price ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_BUY_STOP) : (last < Price ? ORDER_TYPE_SELL_LIMIT : ORDER_TYPE_SELL_STOP));
      TradeRequest.price         = NormalizeDouble(Price, m_Infos.nDigits);
      TradeRequest.sl            = NormalizeDouble(Stop, m_Infos.nDigits);
      TradeRequest.tp            = NormalizeDouble(Take, m_Infos.nDigits);
      TradeRequest.type_time     = (DayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
      TradeRequest.stoplimit     = 0;
      TradeRequest.expiration    = 0;
      TradeRequest.type_filling  = ORDER_FILLING_RETURN;
      TradeRequest.deviation     = 1000;
      TradeRequest.comment       = "Order Generated by Experts Advisor.";
      if(!OrderSend(TradeRequest, TradeResult))
        {
         MessageBox(StringFormat("Error Number: %d", TradeResult.retcode), "Nano EA");
         return 0;
        };
      return TradeResult.order;
     };
```

This function is very simple and is designed to be safe. We will create here an OCO order (One Cancels the Other), which will be sent to the trade server. Please note that we are using **_LIMIT_** or _**STOP**_ orders. This is because this type of orders is simpler and its execution is guaranteed even in the event of sudden price movements.

The order type to be used depends on the execution price and the current price of the trading instrument, as well as whether you are entering a buy or a sell position. This is implemented in the following line:

```
TradeRequest.type = (IsBuy ? (last >= Price ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_BUY_STOP) : (last < Price ? ORDER_TYPE_SELL_LIMIT : ORDER_TYPE_SELL_STOP));
```

It is also possible to create a CROSS ORDER, by specifying a trading instrument in the following line:

```
TradeRequest.symbol = m_szSymbol;
```

But when doing so you will also need to add some code in order to handle open or pending orders via the CROSS ORDER system, since you will have a "wrong" chart. Let's view an example. You can be on the full index chart (IND) and trade the Mini Index (WIN), but MetaTrader 5 will not show the open or pending WIN order when you are using it on the IND chart. Therefore, it is necessary to add a code in order to make orders visible. This can be done by reading position values and presenting them as lines on the chart. This is very useful when you trade and track the symbol trading history. When you use, for example, CROSS ORDER, you can trade **WIN** (Mini Index) using the **WIN$** chart (the Mini Index history chart).

Next, please pay attention to the following code lines:

```
      TradeRequest.price         = NormalizeDouble(Price, m_Infos.nDigits);
      TradeRequest.sl            = NormalizeDouble(Stop, m_Infos.nDigits);
      TradeRequest.tp            = NormalizeDouble(Take, m_Infos.nDigits);
```

These 3 lines will create the OCO order stop levels and the position open price. If you trade short-term orders, which may last only a few seconds, it is not advisable to enter the trade without using OCO orders, as volatility can make the price go from one point to another without a clear direction. When you use OCO, the trade server itself will take care of our position. The OCO order will appear as follows.

![](https://c.mql5.com/2/44/MOD_01.jpg)

In the editing window, the same order will look as follows:

![](https://c.mql5.com/2/44/MOD_02.jpg)

Once you fill in all the required fields, the server will manage the order. As soon as it reaches either **Max Profit** or **Max Loss**, the system will close the order. But if you do not specify Max Profit or Max Loss, the order may remain open until another event occurs. If the order type is set to **_Day Trade_**, the system will close it at the end of the trading day. Otherwise, the position will remain open until you close it manually or until there are no more funds to hold the position open.

Some Expert Advisor systems use orders to close positions: once a position is open, an opposite order to close the position at the specified point, with the same volume, is sent. But this may not work from some scenarios because if the asset goes into auction during the session for some reason, the pending order can be canceled and it should be replaced. This would complicate the EA operation, as you would need to add checks for which orders are active and which are not, and if anything is wrong, the EA would be sending orders one after another, without any criteria.

```
   void Initilize(int nContracts, int FinanceTake, int FinanceStop, color cp, color ct, color cs, bool b1)
     {
      string sz0 = StringSubstr(m_szSymbol = _Symbol, 0, 3);
      double v1 = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) / SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      m_Infos.Id = ChartID();
      m_Infos.TypeSymbol = ((sz0 == "WDO") || (sz0 == "DOL") ? WDO : ((sz0 == "WIN") || (sz0 == "IND") ? WIN : OTHER));
      m_Infos.nDigits = (int) SymbolInfoInteger(m_szSymbol, SYMBOL_DIGITS);
      m_Infos.Volume = nContracts * (m_VolMinimal = SymbolInfoDouble(m_szSymbol, SYMBOL_VOLUME_MIN));
      m_Infos.TakeProfit = AdjustPrice(FinanceTake * v1 / m_Infos.Volume);
      m_Infos.StopLoss = AdjustPrice(FinanceStop * v1 / m_Infos.Volume);
      m_Infos.IsDayTrade = b1;
      CreateHLine(m_Infos.szHLinePrice, m_Infos.cPrice = cp);
      CreateHLine(m_Infos.szHLineTake, m_Infos.cTake = ct);
      CreateHLine(m_Infos.szHLineStop, m_Infos.cStop = cs);
      ChartSetInteger(m_Infos.Id, CHART_COLOR_VOLUME, m_Infos.cPrice);
      ChartSetInteger(m_Infos.Id, CHART_COLOR_STOP_LEVEL, m_Infos.cStop);
     };
```

The routine above is responsible for the initiation of EA data as indicated by the user - it creates an OCO order. We only need to make the following change in this routine.

```
m_Infos.TypeSymbol = ((sz0 == "WDO") || (sz0 == "DOL") ? WDO : ((sz0 == "WIN") || (sz0 == "IND") ? WIN : OTHER));
```

Here we add the trading symbol type in addition to the current ones, if you need something specific.

```
      m_Infos.Volume = nContracts * (m_VolMinimal = SymbolInfoDouble(m_szSymbol, SYMBOL_VOLUME_MIN));
      m_Infos.TakeProfit = AdjustPrice(FinanceTake * v1 / m_Infos.Volume);
      m_Infos.StopLoss = AdjustPrice(FinanceStop * v1 / m_Infos.Volume);
```

The three lines above make the required adjustments for the correct order creation. **nContracts** is a leverage factor, use values like 1, 2, 3 etc. In other words, you do not need to know the minimum symbol volume to be traded. All you really need is to indicate the leverage factor of this minimum volume. For example, if the required minimum volume is 5 contracts and you specify the leverage factor of 3, the system will open a 15-contract order. The two other lines set accordingly **_Take Profit_** and _**Stop Loss**_, based on the user specified parameters. The levels are adjusted with the order volume: if the order increases, the level decreases and vice versa. With this code, you will not have to make calculations to create a position - the EA will calculate everything itself: you instruct the EA which financial instrument to trade with which leverage factor, how much money you want to earn and are ready to lose, and the EA will place an appropriate order for you.

```
   inline void MoveTo(int X, int Y, uint Key)
     {
      int w = 0;
      datetime dt;
      bool bEClick, bKeyBuy, bKeySell;
      double take = 0, stop = 0, price;
      bEClick  = (Key & 0x01) == 0x01;    //Left mouse button click
      bKeyBuy  = (Key & 0x04) == 0x04;    //Pressed SHIFT
      bKeySell = (Key & 0x08) == 0x08;    //Pressed CTRL
      ChartXYToTimePrice(m_Infos.Id, X, Y, w, dt, price);
      ObjectMove(m_Infos.Id, m_Infos.szHLinePrice, 0, 0, price = (bKeyBuy != bKeySell ? AdjustPrice(price) : 0));
      ObjectMove(m_Infos.Id, m_Infos.szHLineTake, 0, 0, take = price + (m_Infos.TakeProfit * (bKeyBuy ? 1 : -1)));
      ObjectMove(m_Infos.Id, m_Infos.szHLineStop, 0, 0, stop = price + (m_Infos.StopLoss * (bKeyBuy ? -1 : 1)));
      if((bEClick) && (bKeyBuy != bKeySell))
         CreateOrderPendent(bKeyBuy, m_Infos.Volume, price, take, stop, m_Infos.IsDayTrade);
      ObjectSetInteger(m_Infos.Id, m_Infos.szHLinePrice, OBJPROP_COLOR, (bKeyBuy != bKeySell ? m_Infos.cPrice : clrNONE));
      ObjectSetInteger(m_Infos.Id, m_Infos.szHLineTake, OBJPROP_COLOR, (take > 0 ? m_Infos.cTake : clrNONE));
      ObjectSetInteger(m_Infos.Id, m_Infos.szHLineStop, OBJPROP_COLOR, (stop > 0 ? m_Infos.cStop : clrNONE));
     };
```

The above code will present the order to be created. It uses the mouse movement to show where the order will be placed. But you want to inform the EA whether you want to buy (press and hold SHIFT) or to sell (press and hold CTRL). Once you click the left mouse button, a pending order will be created at that point.

If you need more data to be displayed, for example the break-even point, add the relevant object to the code.

Now we have a whole EA which works and can create OCO orders. But not everything is perfect here...

### Problem with OCO orders

OCO orders have one problem, which is not the fault of the MetaTrader 5 system or of the trade server. It is connected with the volatility itself which is constantly present in the market. Theoretically, the price should move linearly, without roll-backs, but sometimes we have high volatility, which creates _**gaps**_ inside a candlestick. When these gaps occur at the point where the price of the Stop Loss or Take Profit order is, these points will not trigger and, therefore, the position will not be closed. It can also happen that when the user moves these points, the price can be beyond the corridor formed by the stop loss and the take profit. In this case the order will not be closed either. This is a very dangerous situation which is impossible to predict. As a programmer, you have to provide a relevant mechanism to minimize possible damages.

To refresh the price and to try to keep it within the corridor, we will use two subroutines. The first one is as follows:

```
   void UpdatePosition(void)
     {
      for(int i0 = PositionsTotal() - 1; i0 >= 0; i0--)
         if(PositionGetSymbol(i0) == m_szSymbol)
           {
            m_Take      = PositionGetDouble(POSITION_TP);
            m_Stop      = PositionGetDouble(POSITION_SL);
            m_IsBuy     = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
            m_Volume    = PositionGetDouble(POSITION_VOLUME);
            m_Ticket    = PositionGetInteger(POSITION_TICKET);
           }
     };
```

It will be called in **_OnTrade_** \- the function called by MetaTrader 5 on every position change. The next subroutine to be used is called by **_OnTick_**. It checks and makes sure that the price is within the corridor or within the limits of the OCO order. It is as follows:

```
   inline bool CheckPosition(const double price = 0, const int factor = 0)
     {
      double last;
      if(m_Ticket == 0)
         return false;
      last = SymbolInfoDouble(m_szSymbol, SYMBOL_LAST);
      if(m_IsBuy)
        {
         if((last > m_Take) || (last < m_Stop))
            return ClosePosition();
         if((price > 0) && (price >= last))
            return ClosePosition(factor);
        }
      else
        {
         if((last < m_Take) || (last > m_Stop))
            return ClosePosition();
         if((price > 0) && (price <= last))
            return ClosePosition(factor);
        }
      return false;
     };
```

This code fragment is critical as it will be executed on every tick change and thus it must be as simple as possible so that calculations and tests are performed as efficiently as possible. Please note that while we are keeping the price inside the corridor, we also check something interesting, which can be removed if desired. I will explain this additional test in the next section. Inside this subroutine, we have the following function call:

```
   bool ClosePosition(const int arg = 0)
     {
      double v1 = arg * m_VolMinimal;
      if(!PositionSelectByTicket(m_Ticket))
         return false;
      ZeroMemory(TradeRequest);
      ZeroMemory(TradeResult);
      TradeRequest.action     = TRADE_ACTION_DEAL;
      TradeRequest.type       = (m_IsBuy ? ORDER_TYPE_SELL : ORDER_TYPE_BUY);
      TradeRequest.price      = SymbolInfoDouble(m_szSymbol, (m_IsBuy ? SYMBOL_BID : SYMBOL_ASK));
      TradeRequest.position   = m_Ticket;
      TradeRequest.symbol     = m_szSymbol;
      TradeRequest.volume     = ((v1 == 0) || (v1 > m_Volume) ? m_Volume : v1);
      TradeRequest.deviation  = 1000;
      if(!OrderSend(TradeRequest, TradeResult))
        {
         MessageBox(StringFormat("Error Number: %d", TradeResult.retcode), "Nano EA");
         return false;
        }
      else
         m_Ticket = 0;
      return true;
     };
```

The function will close the specified volume and it works as a protection. However, do not forget that you have to be connected, since the function runs in the MetaTrader 5 client terminal - if connection to the server fails, this function will be completely useless.

Looking at these last two codes, we can see that we can finish the given volume at a certain point. By doing that we either do partial closure or reduce our exposure. Let's figure out how to use this function.

### Working with partial orders

Partial orders are something that many traders like and use. The Expert Advisor allows working with partial closure, but I will not show how to implement such code, as partial orders should be the subject of a separate problem. However, if you want to implement work with partial closures, simply call the _**CheckPosition**_ routine and specify the price at which the order will be executed and the volume, while the EA will do the rest.

I say that partial orders are a special case because they are very individual and it is hard to create a generalized solution to satisfy everyone. The use of a dynamic array would not be suitable here, as you might be swinging - it will work for day trading only if you do not close the EA. If you need to close the EA for any reason, the array solution will not work. You will need to use some storage medium, in which data formatting will depend on what you are going to do with this data.

Anyway, you should avoid partial closures using position opening order as much as possible, as the risk of getting a headache is huge. Let me explain: let's assume that you have a 3x leverage buy position, and you want to make a profit with 2x while still having a 1x leveraged position. This can be done by selling 2x leverage. However if your EA sends a market sell order, it may happen so that volatility will cause the price to go and hit your Take Profit before the sell order is actually executed. In this case the EA will open a new short position in the unfavorable direction. Optionally, you can send a **_Sell Limit_** or **_Sell Stop_** to reduce the position by the 2x leverage. This might seem an adequate solution. But what if another order is sent before the price hits the partial point - you can have a very unpleasant surprise: the open position will be stopped and a little later the order will become open again and will increase losses. If the volatility becomes stronger, the situation will become the same as we mentioned above.

So in my view, as a programmer, the best option to make partial orders is to emulate the sending of orders at market price. But you should be very careful not to exceed the open volume. In this EA I have done it exactly this way. If you wish, you can implement other partial closing methods.

### Conclusion

Creating an Expert Advisor for trading is not as trivial as some people think; it's quite simple compared to some of the other problems we often face when programming, however building something stable and reliable enough to risk your money is something that is often a difficult task. In this article, I suggested something that can make life easier for those who are starting to use MetaTrader 5 and do not have the necessary knowledge to program an EA. This is a good start, as this EA does not open orders but only helps to open orders in a more reliable way. Once the order is placed, the EA has nothing else to do, and further MetaTrader 5 starts working, except for the above-mentioned code fragments.

The Expert Advisor presented in this article can be improved in various ways to work with sets of parameters, but this would require more code that will make it more independent from MetaTrader 5.

The great success of this EA is that it uses MetaTrader 5 itself to perform actions that are not in its code, and therefore it is extremely stable and reliable.

![](https://c.mql5.com/2/44/Nano-Video-Demonstrativo.gif)

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10085](https://www.mql5.com/pt/articles/10085)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10085.zip "Download all attachments in the single ZIP archive")

[EA\_Nano\_rvl\_1.1.mq5](https://www.mql5.com/en/articles/download/10085/ea_nano_rvl_1.1.mq5 "Download EA_Nano_rvl_1.1.mq5")(23.44 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/389509)**
(12)


![C4rl1n](https://c.mql5.com/avatar/avatar_na2.png)

**[C4rl1n](https://www.mql5.com/en/users/c4rl1n)**
\|
28 Aug 2023 at 19:08

Good afternoon, where do I set the number of points and number of lots for the mini-index?


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
29 Aug 2023 at 14:03

**C4rl1n [#](https://www.mql5.com/pt/forum/384314#comment_49005632):**

Good afternoon, where do I set the number of points and number of lots for the mini-index?

Actually, in this code the adjustment is automatic. You tell it the financier and the number of contracts to trade and the code makes the adjustment in terms of points... This information is given when you place the Expert Advisor on the chart. 😁👍

![C4rl1n](https://c.mql5.com/avatar/avatar_na2.png)

**[C4rl1n](https://www.mql5.com/en/users/c4rl1n)**
\|
29 Aug 2023 at 16:53

I see, it's because I want to develop my EA, but the number of lots and points don't match up


![László Tugyi](https://c.mql5.com/avatar/2025/1/679BD1B7-439A.jpg)

**[László Tugyi](https://www.mql5.com/en/users/laszlotugyi)**
\|
3 Oct 2025 at 08:43

**MetaIdézetek :**

A new article has been published, ["Developing a Trading Expert Advisor from Scratch":](https://www.mql5.com/en/articles/10085)

Author: [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831 "DJ_TLoG_831")

Where can I download it?


![Vinicius Pereira De Oliveira](https://c.mql5.com/avatar/2025/4/6804f561-0038.png)

**[Vinicius Pereira De Oliveira](https://www.mql5.com/en/users/vinicius-fx)**
\|
3 Oct 2025 at 09:32

**László Tugyi [#](https://www.mql5.com/pt/forum/384314/page2#comment_58179974):** Where can I download it?

[https://www.mql5.com/en/articles/download/10085/ea\_nano\_rvl\_1.1.mq5](https://www.mql5.com/en/articles/download/10085/ea_nano_rvl_1.1.mq5)

![Visual evaluation of optimization results](https://c.mql5.com/2/44/visual-estimation.png)[Visual evaluation of optimization results](https://www.mql5.com/en/articles/9922)

In this article, we will consider how to build graphs of all optimization passes and to select the optimal custom criterion. We will also see how to create a desired solution with little MQL5 knowledge, using the articles published on the website and forum comments.

![Graphics in DoEasy library (Part 92): Standard graphical object memory class. Object property change history](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__4.png)[Graphics in DoEasy library (Part 92): Standard graphical object memory class. Object property change history](https://www.mql5.com/en/articles/10237)

In the article, I will create the class of the standard graphical object memory allowing the object to save its states when its properties are modified. In turn, this allows retracting to the previous graphical object states.

![Graphics in DoEasy library (Part 93): Preparing functionality for creating composite graphical objects](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__5.png)[Graphics in DoEasy library (Part 93): Preparing functionality for creating composite graphical objects](https://www.mql5.com/en/articles/10331)

In this article, I will start developing the functionality for creating composite graphical objects. The library will support creating composite graphical objects allowing those objects have any hierarchy of connections. I will prepare all the necessary classes for subsequent implementation of such objects.

![An Analysis of Why Expert Advisors Fail](https://c.mql5.com/2/45/Why-Expert-Advisors-Fail.png)[An Analysis of Why Expert Advisors Fail](https://www.mql5.com/en/articles/3299)

This article presents an analysis of currency data to better understand why expert advisors can have good performance in some regions of time and poor performance in other regions of time.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=unwfdrdyldcbxbawbdxhkwjdhusmbfdf&ssn=1769104133695752514&ssn_dr=0&ssn_sr=0&fv_date=1769104133&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10085&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176910413372257502&fz_uniq=5051705246452929797&sv=2552)

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