---
title: Developing a trading Expert Advisor from scratch (Part 27): Towards the future (II)
url: https://www.mql5.com/en/articles/10630
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:46:04.826630
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=otswfbkqsvroxtlmerixmpzfuaqosdyc&ssn=1769103962088240065&ssn_dr=0&ssn_sr=0&fv_date=1769103962&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10630&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20(Part%2027)%3A%20Towards%20the%20future%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691039624644277&fz_uniq=5051669220267250778&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the previous article, [Developing a trading Expert Advisor from scratch (Part 26)](https://www.mql5.com/en/articles/10620), we have fixed a catastrophic error which existed in the order system. We have also started to implement changes to enable the operation of the new order system. Although the system that was originally implemented in this article series is quite interesting, it has a flaw that makes it inoperable. This flaw was shown at the end of the previous article. The reason was not knowing how to trade, more specifically how to choose an expiration time for an order or position, in addition to other minor issues. The system was fixed as an order or a position that should be closed at the end of the trading session or the current day. But sometimes we want to make longer-term trades, so leaving everything as it is does not really help.

Therefore, in this article, I will show you a fix. We will see how to make the order system more intuitive so that you can immediately and accurately determine what each order is, how it is processed, and what kind of movement is expected.

This system is so interesting, simple and intuitive that once you see it in action, you won't want to work without it. What I am going to show you in this article is just one of the many possibilities that you can implement in the order system. Perhaps, later I will show other things, but what we are going to see in this article can give an excellent basis for creating other useful and interesting modifications for your particular case. Anyway, I try to keep everything in these articles as general as possible.

### 2.0. Intuitive model

So far, we have been working with the orders as follows:

![](https://c.mql5.com/2/45/001__8.png)

The Take Profit and Stop Loss indications have a recognizable form which is quite intuitive: green shows what will be earned to our account and red shows what will be deducted. It is all clear. If we have a Stop indication as shown below, it still demonstrates that the activation of stop loss will result in this sum to be added to our account. In other words, the stop order levels do not require our effort, at least for now. Perhaps in the future you may want to change something in them, but at the moment they are quite suitable for use.

![](https://c.mql5.com/2/45/002__7.png)

This way of using stop levels is quite simple and intuitive for any trader to analyze. But we have something which is not very clear. The first one is the pending order entry point indication.

![](https://c.mql5.com/2/45/003__11.png)

Can we know whether this pending order is a buy or sell order? And one more thing: is it possible to know if this pending order will be closed at the end of the day or a longer-term position will be opened? It's easy. Now, if we already have open positions, the indicator will look like below:

![](https://c.mql5.com/2/45/004__6.png)

And again, we have the same problems as with the pending order indicators. When you look at the chart and see these indicators, you cannot say for sure if a position will be closed at the end of the day or if it will last longer. If it closes at the end of the day, you wouldn't want the broker to stop it compulsorily, since you will have to pay for this. And it is not very reasonable to close orders without any criteria, since even the MetaTrader toolbox does not show this information, so having it on the chart through the indicator is just great.

Thus, we will have to make changes here, especially in the indicators that show the position entry point, in order to better understand what is happening.

### 2.0.1. How to add new information to indicators

The easiest way to add new information without taking up much space on the chart is to use bitmaps, as they are easy to understand and fairly representative. Thus, without inserting any additional code, we add four new bitmaps to the EA, which can be seen in the C\_IndicatorTradeView class.

```
#define def_BtnClose            "Images\\NanoEA-SIMD\\Btn_Close.bmp"
#define def_BtnCheckEnabled     "Images\\NanoEA-SIMD\\CheckBoxEnabled.bmp"
#define def_BtnCheckDisabled    "Images\\NanoEA-SIMD\\CheckBoxDisabled.bmp"
#define def_BtnDayTrade         "Images\\NanoEA-SIMD\\Inf_DayTrade.bmp"
#define def_BtnSwing            "Images\\NanoEA-SIMD\\Inf_Swing.bmp"
#define def_BtnInfoBuy          "Images\\NanoEA-SIMD\\Inf_Buy.bmp"
#define def_BtnInfoSell         "Images\\NanoEA-SIMD\\Inf_Sell.bmp"
//+------------------------------------------------------------------+
#resource "\\" + def_BtnClose
#resource "\\" + def_BtnCheckEnabled
#resource "\\" + def_BtnCheckDisabled
#resource "\\" + def_BtnDayTrade
#resource "\\" + def_BtnSwing
#resource "\\" + def_BtnInfoBuy
#resource "\\" + def_BtnInfoSell
```

In addition, we only need to implement two new objects in the order system.

```
//+------------------------------------------------------------------+
enum eIndicatorTrade {IT_NULL, IT_STOP= 65, IT_TAKE, IT_PENDING, IT_RESULT};
enum eEventType {EV_NULL, EV_GROUND = 65, EV_LINE, EV_CLOSE, EV_EDIT, EV_PROFIT, EV_MOVE, EV_CHECK, EV_TYPE, EV_DS};
//+------------------------------------------------------------------+
C_Object_BackGround     m_BackGround;
C_Object_TradeLine      m_TradeLine;
C_Object_BtnBitMap      m_BtnClose,
                        m_BtnCheck,
                        m_BtnInfoType,
                        m_BtnInfo_DS;
C_Object_Edit           m_EditInfo1,
                        m_EditInfo2;
C_Object_Label          m_BtnMove;
```

Whenever we are going to add a new object, we will also need to add an EVENT linked to the object, which will ensure that the object has a unique name.

Now comes the most interesting part of programming. The first thing we need to do is to take care about the ghosts. We need to update them so that they keep information located in them. Of course, it could be deleted, but I think it's better to keep the basic data. Take a look at the following code:

```
#define macroSwapName(A, B) ObjectSetString(Terminal.Get_ID(), macroMountName(ticket, A, B), OBJPROP_NAME, macroMountName(def_IndicatorGhost, A, B));
                void CreateGhostIndicator(ulong ticket, eIndicatorTrade it)
                        {
                                if (GetInfosTradeServer(m_Selection.ticket = ticket) != 0)
                                {
                                        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
                                        macroSwapName(it, EV_LINE);
                                        macroSwapName(it, EV_GROUND);
                                        macroSwapName(it, EV_MOVE);
                                        macroSwapName(it, EV_EDIT);
                                        macroSwapName(it, EV_CLOSE);
                                        if (it == IT_PENDING)
                                        {
                                                macroSwapName(it, EV_CHECK);
                                                macroSwapName(it, EV_TYPE);
                                                macroSwapName(it, EV_DS);
                                        }
                                        m_TradeLine.SetColor(macroMountName(def_IndicatorGhost, it, EV_LINE), def_IndicatorGhostColor);
                                        m_BackGround.SetColor(macroMountName(def_IndicatorGhost, it, EV_GROUND), def_IndicatorGhostColor);
                                        m_BtnMove.SetColor(macroMountName(def_IndicatorGhost, it, EV_MOVE), def_IndicatorGhostColor);
                                        ObjectDelete(Terminal.Get_ID(), macroMountName(def_IndicatorGhost, it, EV_CLOSE));
                                        m_TradeLine.SpotLight();
                                        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
                                        m_Selection.it = it;
                                }else m_Selection.ticket = 0;
                        }
#undef macroSwapName
```

The highlighted lines pass the objects to the ghost, it's something very simple and clear. Another simple code transforms indicators from pending to floating.

```
#define macroSwapAtFloat(A, B) ObjectSetString(Terminal.Get_ID(), macroMountName(ticket, A, B), OBJPROP_NAME, macroMountName(def_IndicatorFloat, A, B));
                bool PendingAtFloat(ulong ticket)
                        {
                                eIndicatorTrade it;

                                if (macroGetLinePrice(def_IndicatorFloat, IT_PENDING) > 0) return false;
                                macroSwapAtFloat(IT_PENDING, EV_CHECK);
                                macroSwapAtFloat(IT_PENDING, EV_TYPE);
                                macroSwapAtFloat(IT_PENDING, EV_DS);
                                for (char c0 = 0; c0 < 3; c0++)
                                {
                                        switch(c0)
                                        {
                                                case 0: it = IT_PENDING;        break;
                                                case 1: it = IT_STOP;           break;
                                                case 2: it = IT_TAKE;           break;
                                                default:
                                                        return false;
                                        }
                                        macroSwapAtFloat(it, EV_CLOSE);
                                        macroSwapAtFloat(it, EV_MOVE);
                                        macroSwapAtFloat(it, EV_EDIT);
                                        macroSwapAtFloat(it, EV_GROUND);
                                        macroSwapAtFloat(it, EV_LINE);
                                        m_EditInfo1.SetOnlyRead(macroMountName(def_IndicatorFloat, IT_PENDING, EV_EDIT), false);
                                }
                                return true;
                        }
#undef macroSwapAtFloat
```

The highlighted lines convert the objects to the floating indicator, allowing us to perform required actions later. Now we need to implement some changes in the code that creates the indicator. It is something that you will test and adjust until you like it. Basically, the changes were made in the highlighted points in the code below:

```
#define macroCreateIndicator(A, B, C, D)        {                                                                               \
                m_TradeLine.Create(ticket, sz0 = macroMountName(ticket, A, EV_LINE), C);                                        \
                m_BackGround.Create(ticket, sz0 = macroMountName(ticket, A, EV_GROUND), B);                                     \
                m_BackGround.Size(sz0, (A == IT_RESULT ? 100 : (A == IT_PENDING ? 144 : 92)), (A == IT_RESULT ? 34 : 22));      \
                m_EditInfo1.Create(ticket, sz0 = macroMountName(ticket, A, EV_EDIT), D, 0.0);                                   \
                m_EditInfo1.Size(sz0, 60, 14);                                                                                  \
                if (A != IT_RESULT)     {                                                                                       \
                        m_BtnMove.Create(ticket, sz0 = macroMountName(ticket, A, EV_MOVE), "Wingdings", "u", 17, C);            \
                        m_BtnMove.Size(sz0, 21, 23);                                                                            \
                                        }else                   {                                                               \
                        m_EditInfo2.Create(ticket, sz0 = macroMountName(ticket, A, EV_PROFIT), clrNONE, 0.0);                   \
                        m_EditInfo2.Size(sz0, 60, 14);  }                                                                       \
                                                }

#define macroInfoBase(A)        {                                                                                               \
                m_BtnInfoType.Create(ticket, sz0 = macroMountName(ticket, A, EV_TYPE), def_BtnInfoBuy, def_BtnInfoSell);        \
                m_BtnInfoType.SetStateButton(sz0, m_Selection.bIsBuy);                                                          \
                m_BtnInfo_DS.Create(ticket, sz0 = macroMountName(ticket, A, EV_DS), def_BtnDayTrade, def_BtnSwing);             \
                m_BtnInfo_DS.SetStateButton(sz0, m_Selection.bIsDayTrade);                                                      \
                                }

                void CreateIndicator(ulong ticket, eIndicatorTrade it)
                        {
                                string sz0;

                                switch (it)
                                {
                                        case IT_TAKE    : macroCreateIndicator(it, clrForestGreen, clrDarkGreen, clrNONE); break;
                                        case IT_STOP    : macroCreateIndicator(it, clrFireBrick, clrMaroon, clrNONE); break;
                                        case IT_PENDING:
                                                macroCreateIndicator(it, clrCornflowerBlue, clrDarkGoldenrod, def_ColorVolumeEdit);
                                                m_BtnCheck.Create(ticket, sz0 = macroMountName(ticket, it, EV_CHECK), def_BtnCheckEnabled, def_BtnCheckDisabled);
                                                m_BtnCheck.SetStateButton(sz0, true);
                                                macroInfoBase(IT_PENDING);
                                                break;
                                        case IT_RESULT  :
                                                macroCreateIndicator(it, clrSlateBlue, clrSlateBlue, def_ColorVolumeResult);
                                                macroInfoBase(IT_RESULT);
                                                break;
                                }
                                m_BtnClose.Create(ticket, macroMountName(ticket, it, EV_CLOSE), def_BtnClose);
                        }
#undef macroInfoBase
#undef macroCreateIndicator
```

Note that **macroInfoBase** creates the objects that are used in the indicator, but these objects will be created only in the position opening and position result indicators, while there is no need to create these objects in other indicators. But notice that we do not position the objects in the place where we created them. This is done in another place, which is shown next.

```
#define macroSetAxleY(A)                {                                                                               \
                m_BackGround.PositionAxleY(macroMountName(ticket, A, EV_GROUND), y);                                    \
                m_TradeLine.PositionAxleY(macroMountName(ticket, A, EV_LINE), y);                                       \
                m_BtnClose.PositionAxleY(macroMountName(ticket, A, EV_CLOSE), y);                                       \
                if (A != IT_RESULT)m_BtnMove.PositionAxleY(macroMountName(ticket, A, EV_MOVE), y, 1);                   \
                else m_EditInfo2.PositionAxleY(macroMountName(ticket, A, EV_PROFIT), y, 1);                             \
                m_EditInfo1.PositionAxleY(macroMountName(ticket, A, EV_EDIT), y, (A == IT_RESULT ? -1 : 0));            \
                if (A == IT_PENDING) m_BtnCheck.PositionAxleY(macroMountName(ticket, A, EV_CHECK), y);                  \
                if ((A == IT_PENDING) || (A == IT_RESULT))      {                                                       \
                        m_BtnInfoType.PositionAxleY(macroMountName(ticket, A, EV_TYPE), y + (A == IT_PENDING ? 0 : 8)); \
                        m_BtnInfo_DS.PositionAxleY(macroMountName(ticket, A, EV_DS), y - (A == IT_PENDING ? 0: 8));     \
                                                                }                                                       \
                                        }

#define macroSetAxleX(A, B)             {                                                                                               \
                m_BackGround.PositionAxleX(macroMountName(ticket, A, EV_GROUND), B);                                                    \
                m_TradeLine.PositionAxleX(macroMountName(ticket, A, EV_LINE), B);                                                       \
                m_BtnClose.PositionAxleX(macroMountName(ticket, A, EV_CLOSE), B + 3);                                                   \
                m_EditInfo1.PositionAxleX(macroMountName(ticket, A, EV_EDIT), B + 21);                                                  \
                if (A != IT_RESULT) m_BtnMove.PositionAxleX(macroMountName(ticket, A, EV_MOVE), B + 80 + (A == IT_PENDING ? 52 : 0));   \
                else m_EditInfo2.PositionAxleX(macroMountName(ticket, A, EV_PROFIT), B + 21);                                           \
                if (A == IT_PENDING) m_BtnCheck.PositionAxleX(macroMountName(ticket, A, EV_CHECK), B + 82);                             \
                if ((A == IT_PENDING) || (A == IT_RESULT))      {                                                                       \
                        m_BtnInfoType.PositionAxleX(macroMountName(ticket, A, EV_TYPE), B + (A == IT_PENDING ? 100 : 82));              \
                        m_BtnInfo_DS.PositionAxleX(macroMountName(ticket, A, EV_DS), B + (A == IT_PENDING ? 118 : 82));                 \
                                                                }                                                                       \
                                        }
//---
        void ReDrawAllsIndicator(void)
                        {
                                C_IndicatorTradeView::st00 Local;
                                int             max = ObjectsTotal(Terminal.Get_ID(), -1, OBJ_EDIT);
                                ulong           ticket;
                                eIndicatorTrade it;
                                eEventType ev;

                                Local = m_Selection;
                                m_Selection.ticket = 0;
                                for (int c0 = 0; c0 <= max; c0++)
                                   if (GetIndicatorInfos(ObjectName(Terminal.Get_ID(), c0, -1, OBJ_EDIT), ticket, it, ev))
                                      if ((it == IT_PENDING) || (it == IT_RESULT))
                                      {
                                        PositionAxlePrice(ticket, IT_STOP, macroGetLinePrice(ticket, IT_STOP));
                                        PositionAxlePrice(ticket, IT_TAKE, macroGetLinePrice(ticket, IT_TAKE));
                                        PositionAxlePrice(ticket, it, macroGetLinePrice(ticket, it));
                                        }
                                m_Selection = Local;
                                ChartRedraw();
                        }
//---
inline void PositionAxlePrice(ulong ticket, eIndicatorTrade it, double price)
                        {
                                int x, y, desl;

                                ChartTimePriceToXY(Terminal.Get_ID(), 0, 0, price, x, y);
                                macroSetLinePrice(ticket, it, price);
                                macroSetAxleY(it);
                                switch (it)
                                {
                                        case IT_TAKE: desl = 160; break;
                                        case IT_STOP: desl = 270; break;
                                        default: desl = 0;
                                }
                                macroSetAxleX(it, desl + (int)(ChartGetInteger(Terminal.Get_ID(), CHART_WIDTH_IN_PIXELS) * 0.2));
                        }
#undef macroSetAxleX
#undef macroSetAxleY
```

I would like also to emphasize that I don't like to make drastic and radical changes to the code. Basically, the only changes were highlighted in the code above.

### 2.0.2. Problems in sight

Although everything works perfectly fine, we have a problem. I searched all the MQL5 documentation, but I didn't find any way to solve the thing in a simple way. The problem is how to know if a recently opened position is a Day Trade (short trades within the same day) or Swing Trade (longer trades). In the case of an older position, opened the day before, it is quite simple to do this type of analysis, because it would be enough to compare the current day and the position opening day: if they are different, the position is a Swing Trade. But what if the EA is closed and you initiate it the same day the position was opened? In this case there is no way of knowing if a position is a Day Trade or a Swing Trade.

This problem does not exist for pending orders since there is a way to check this. When calling [OrderGetInteger](https://www.mql5.com/en/docs/trading/ordergetinteger) using the ORDER\_TYPE\_TIME parameter, a value of the [ENUM\_ORDER\_TYPE\_TIME](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_time) enumeration is returned, which indicates whether an order is Day Trade or Swing Trade. But the same does not happen for positions.

For this reason, my solution for this case is to add something to the order or position to let the EA know the duration of the operation, regardless of any other information. But this is not a perfect solution, as it solves the problem in several cases, but not in all. Because the trader can modify the system used by the EA to identify whether a trade is a Swing or Day Trade before the period required for the analysis.

To understand better, let's see how the solution is implemented.

```
inline char GetInfosTradeServer(ulong ticket)
{
        long info;

        if (ticket == 0) return 0;
        if (OrderSelect(ticket))
        {
                if (OrderGetString(ORDER_SYMBOL) != Terminal.GetSymbol()) return 0;
                info = OrderGetInteger(ORDER_TYPE);
                m_Selection.bIsBuy = ((info == ORDER_TYPE_BUY_LIMIT) || (info == ORDER_TYPE_BUY_STOP) || (info == ORDER_TYPE_BUY_STOP_LIMIT) || (info == ORDER_TYPE_BUY));
                m_Selection.pr = OrderGetDouble(ORDER_PRICE_OPEN);
                m_Selection.tp = OrderGetDouble(ORDER_TP);
                m_Selection.sl = OrderGetDouble(ORDER_SL);
                m_Selection.vol = OrderGetDouble(ORDER_VOLUME_CURRENT);
                m_Selection.bIsDayTrade = ((ENUM_ORDER_TYPE_TIME)OrderGetInteger(ORDER_TYPE_TIME) == ORDER_TIME_DAY);

                return -1;
        }
        if (PositionSelectByTicket(ticket))
        {
                if (PositionGetString(POSITION_SYMBOL) != Terminal.GetSymbol()) return 0;
                m_Selection.bIsBuy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
                m_Selection.pr = PositionGetDouble(POSITION_PRICE_OPEN);
                m_Selection.tp = PositionGetDouble(POSITION_TP);
                m_Selection.sl = PositionGetDouble(POSITION_SL);
                m_Selection.vol = PositionGetDouble(POSITION_VOLUME);
                if (macroGetDate(PositionGetInteger(POSITION_TIME)) == macroGetDate(TimeTradeServer()))
                        m_Selection.bIsDayTrade = PositionGetString(POSITION_COMMENT) == def_COMMENT_TO_DAYTRADE;
                else m_Selection.bIsDayTrade = false;

                return 1;
        }
        return 0;
}
```

As mentioned above, in the case of pending orders, it is enough to call OrderGetInteger and get the value we need. It is a little more complicated with positions. It works as follows: check the position opening day and the current day of the trading server. If both are the same, check the comment in the order. If the comment indicates the string that is used in the C\_Router class to indicate that if the position is opened it will be Day Trade, the EA will interpret it and display this in the position indicator. But the comment must not change until the end of the day, because if it changes, then the EA may report that the Day Trade position is actually a Swing Trade, in which case this is not the fault of the EA, but because the trader changed the comment too soon.

This is the downside of this solution, but if anyone has an idea on how to determine if a position is a Day Trade or not just by looking at the position data, please share it in the comments.

The way it looks in the case of pending orders is seen in the video below:

Demonstração Basica - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10630)

MQL5.community

1.91K subscribers

[Demonstração Basica](https://www.youtube.com/watch?v=YCOHe3KznR4)

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

[Watch on](https://www.youtube.com/watch?v=YCOHe3KznR4&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10630)

0:00

0:00 / 0:48

•Live

•

Now we have almost everything ready, we just need to make some more additions to the code so that the EA becomes interesting to use.

### 2.0.3. Responding to platform messages

Our entire order system is based on messages sent by MetaTrader 5 so that the EA can know what should or should not be done. That is why it is so important to know how to implement the message system.

The full messaging related code is shown below:

```
#define macroGetDataIndicatorFloat      {                                                                                                               \
                m_Selection.vol = m_EditInfo1.GetTextValue(macroMountName(def_IndicatorFloat, IT_PENDING, EV_EDIT)) * Terminal.GetVolumeMinimal();      \
                m_Selection.bIsBuy = m_BtnInfoType.GetStateButton(macroMountName(def_IndicatorFloat, IT_PENDING, EV_TYPE));                             \
                m_Selection.pr = macroGetLinePrice(def_IndicatorFloat, IT_PENDING);                                                                     \
                m_Selection.sl = macroGetLinePrice(def_IndicatorFloat, IT_STOP);                                                                        \
                m_Selection.tp = macroGetLinePrice(def_IndicatorFloat, IT_TAKE);                                                                        \
                m_Selection.bIsDayTrade = m_BtnInfo_DS.GetStateButton(macroMountName(def_IndicatorFloat, IT_PENDING, EV_DS));                           \
                                        }

                void DispatchMessage(int id, long lparam, double dparam, string sparam)
                        {
                                ulong   ticket;
                                double  price;
                                bool   	bKeyBuy,
                                        bKeySell,
                                        bEClick;
                                datetime dt;
                                uint     mKeys;
                                char     cRet;
                                eIndicatorTrade  it;
                                eEventType       ev;

                                static bool bMounting = false;
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
                                                                m_Selection.bIsDayTrade = Chart.GetBaseFinance(m_Selection.vol, valueTp, valueSl);
                                                                valueTp = Terminal.AdjustPrice(valueTp * Terminal.GetAdjustToTrade() / m_Selection.vol);
                                                                valueSl = Terminal.AdjustPrice(valueSl * Terminal.GetAdjustToTrade() / m_Selection.vol);
                                                                m_Selection.it = IT_PENDING;
                                                                m_Selection.pr = price;
                                                        }
                                                        m_Selection.tp = m_Selection.pr + (bKeyBuy ? valueTp : (-valueTp));
                                                        m_Selection.sl = m_Selection.pr + (bKeyBuy ? (-valueSl) : valueSl);
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
                                                        if (bEClick) SetPriceSelection(price); else MoveSelection(price);
                                                }
                                                break;
                                        case CHARTEVENT_OBJECT_DELETE:
                                                if (GetIndicatorInfos(sparam, ticket, it, ev))
                                                {
                                                        if (GetInfosTradeServer(ticket) == 0) break;
                                                        CreateIndicator(ticket, it);
                                                        if ((it == IT_PENDING) || (it == IT_RESULT))
                                                                PositionAxlePrice(ticket, it, m_Selection.pr);
                                                        ChartRedraw();
                                                        m_TradeLine.SpotLight();
                                                        m_Selection.ticket = 0;
                                                        UpdateIndicators(ticket, m_Selection.tp, m_Selection.sl, m_Selection.vol, m_Selection.bIsBuy);
                                                }
                                                break;
                                        case CHARTEVENT_OBJECT_ENDEDIT:
                                                macroGetDataIndicatorFloat;
                                                m_Selection.ticket = 0;
                                                UpdateIndicators(def_IndicatorFloat, m_Selection.tp, m_Selection.sl, m_Selection.vol, m_Selection.bIsBuy);
                                                break;
                                        case CHARTEVENT_CHART_CHANGE:
                                                ReDrawAllsIndicator();
                                                break;
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
                                                                if (ticket == def_IndicatorFloat) RemoveIndicator(def_IndicatorFloat, it);
                                                                else if ((cRet = GetInfosTradeServer(ticket)) != 0) switch (it)
                                                                {
                                                                        case IT_PENDING:
                                                                        case IT_RESULT:
                                                                                if (cRet < 0) RemoveOrderPendent(ticket); else ClosePosition(ticket);
                                                                                break;
                                                                        case IT_TAKE:
                                                                        case IT_STOP:
                                                                                m_Selection.ticket = ticket;
                                                                                m_Selection.it = it;
                                                                                SetPriceSelection(0);
                                                                                break;
                                                                }
                                                                break;
                                                        case EV_MOVE:
                                                                        if (ticket == def_IndicatorFloat)
                                                                        {
                                                                                macroGetDataIndicatorFloat;
                                                                                m_Selection.ticket = ticket;
                                                                                m_Selection.it = it;
                                                                        }else   CreateGhostIndicator(ticket, it);
                                                                break;
                                                        case EV_CHECK:
                                                                if (ticket != def_IndicatorFloat)
                                                                {
                                                                        if (PendingAtFloat(ticket)) RemoveOrderPendent(ticket);
                                                                        else m_BtnCheck.SetStateButton(macroMountName(ticket, IT_PENDING, EV_CHECK), true);
                                                                } else
                                                                {
                                                                        macroGetDataIndicatorFloat;
                                                                        m_Selection.ticket = def_IndicatorTicket0;
                                                                        m_Selection.it = IT_PENDING;
                                                                        SetPriceSelection(m_Selection.pr);
                                                                        RemoveIndicator(def_IndicatorFloat);
                                                                }
                                                                break;
                                                }
                                                break;
                                }
                        }
#undef macroGetDataIndicatorFloat
```

Don't be scared by this code. Although it looks big and complicated, it is actually quite simple. I will focus on the highlighted parts to explain what's new in the message processing code.

The first new thing is in the [CHARTEVENT\_OBJECT\_ENDEDIT](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) event handing code. It is triggered by MetaTrader 5 every time we finish editing the content present in the EDIT object. What does it mean? It is very important, because if we do not process this event and try to manipulate the data of stop level indicators after editing the levering value, then we will have a mismatch in the values. Although the EA will force the value to return to its original value, but if we handle this event as shown in the code, this problem will not exist, and we can trade smoothly with the levering data. Bear in mind that when you ask the EA to allow you to make these adjustments, you will actually want to check whether or not it is a good idea to enter the operation more or less leveraged. In this way you can check without taking any risks, as the EA will send the order to the server only when you ask it to do this, and the moment this happens is when the checkbox is activated.

Now let's take a closer look at the [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) event. For this we take the fragment highlighted in the previous code.

```
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

// ... Rest of the code...
```

What is this code actually doing? Do you have any idea? Well, the videos in this article demonstrate this, but can you understand how that kind of thing is done? Many would imagine that it was an extremely complex code, but there it is, just above.

There are two things that we need to do. The first is that when a BitMap object is clicked, its status changes, and we have to check if its ticket is from something that already exists on the server or is it something that is only on the chart. This is done by the points **highlighted in green**. if the ticket exists on the server, the status change must be undone, then the EA will correct this by making the required change.

Now take a look at the section **highlighted in yellow**. The idea is based on the following: Why do should I place another order on the chart, if it already exists on the chart, and I just want to reverse the direction? In other words, if it was Buy, now I want it to be Sell, and vice versa. The yellow fragment does just that: when BitMap responsible for whether we are buying or selling is clicked, the direction changes automatically. One detail: this can only be done for a floating order; it is prohibited for orders already on the server.

With all these changes, the indicators now look like this:

**Types of pending orders**

![](https://c.mql5.com/2/45/003.1__1.png)

**Position indicator:**

![](https://c.mql5.com/2/45/004.1__2.png)

It is now much easier to determine what a pending order or open position is doing, because you can know exactly the expected movement or the position lifetime. The green arrow pointing up indicates a buy position; a red arrow pointing down is for a sell position. Letter D indicates a Day Trade, which will be closed at the end of the day. If it is S, then it is a Swing Trade, and the operation will not necessarily be closed at the end of the day.

The next video shows how the new order system works. I focused on pending orders because they are subject to further modifications while the position indicators cannot be changed. They will only show the data provided by the server regarding the position. Take a closer look at how it all works before trying it out on a live account, because the system is practical, but you need to get familiar with it to get the most out of its functionality.

Demostração Parte 27 01 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10630)

MQL5.community

1.91K subscribers

[Demostração Parte 27 01](https://www.youtube.com/watch?v=hwBMhQnqRvo)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=hwBMhQnqRvo&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10630)

0:00

0:00 / 2:06

•Live

•

### Conclusion

Well, our order system is now quite versatile. It can do several things and helps us a lot, but there is still one important detail missing which will be implemented in the next article. So, see you soon...

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10630](https://www.mql5.com/pt/articles/10630)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10630.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Em\_diretco\_ao\_Futuro\_q\_II\_o.zip](https://www.mql5.com/en/articles/download/10630/ea_-_em_diretco_ao_futuro_q_ii_o.zip "Download EA_-_Em_diretco_ao_Futuro_q_II_o.zip")(12035.84 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/434949)**

![Market math: profit, loss and costs](https://c.mql5.com/2/48/z7jdvip34mo_2022-08-18_235145181.png)[Market math: profit, loss and costs](https://www.mql5.com/en/articles/10211)

In this article, I will show you how to calculate the total profit or loss of any trade, including commission and swap. I will provide the most accurate mathematical model and use it to write the code and compare it with the standard. Besides, I will also try to get on the inside of the main MQL5 function to calculate profit and get to the bottom of all the necessary values from the specification.

![DoEasy. Controls (Part 15): TabControl WinForms object — several rows of tab headers, tab handling methods](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__3.png)[DoEasy. Controls (Part 15): TabControl WinForms object — several rows of tab headers, tab handling methods](https://www.mql5.com/en/articles/11316)

In this article, I will continue working on the TabControl WinForm object — I will create a tab field object class, make it possible to arrange tab headers in several rows and add methods for handling object tabs.

![DIY technical indicator](https://c.mql5.com/2/48/drawing-indicator__1.png)[DIY technical indicator](https://www.mql5.com/en/articles/11348)

In this article, I will consider the algorithms allowing you to create your own technical indicator. You will learn how to obtain pretty complex and interesting results with very simple initial assumptions.

![Developing a trading Expert Advisor from scratch (Part 26): Towards the future (I)](https://c.mql5.com/2/48/development__2.png)[Developing a trading Expert Advisor from scratch (Part 26): Towards the future (I)](https://www.mql5.com/en/articles/10620)

Today we will take our order system to the next level. But before that, we need to solve a few problems. Now we have some questions that are related to how we want to work and what things we do during the trading day.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=elrrakzrfcxhcovlhnyztlwftcdqnoqz&ssn=1769103962088240065&ssn_dr=0&ssn_sr=0&fv_date=1769103962&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10630&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20(Part%2027)%3A%20Towards%20the%20future%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176910396246489394&fz_uniq=5051669220267250778&sv=2552)

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