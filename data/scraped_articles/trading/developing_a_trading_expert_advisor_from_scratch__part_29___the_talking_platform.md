---
title: Developing a trading Expert Advisor from scratch (Part 29): The talking platform
url: https://www.mql5.com/en/articles/10664
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:45:33.486464
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=eeernhcqvjydeqpsmiprsimcpproofhd&ssn=1769103932783507558&ssn_dr=0&ssn_sr=0&fv_date=1769103932&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10664&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20(Part%2029)%3A%20The%20talking%20platform%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176910393219841087&fz_uniq=5051662898075391035&sv=2552)

MetaTrader 5 / Examples


### Introduction

What if we make an Expert Advisor more fun? Financial market trading is often too boring and monotonous, but we can make this job less tiring. Please note that this project can be dangerous for those who experience problems such as addiction. However, in a general case, it just makes things less boring.

Warning: DO NOT use the modifications given in this article if you are considering the market as gambling, because there is a real risk of very large losses.

Although the warning above sounds like a joke, the truth is that some modifications to the EA make it dangerous for those who are addicted to gambling in general.

Some of the changes that will be made here are aimed at improving the overall stability and performance of the EA. If you want to keep some of the things we are deleting in this article, then it's not difficult. Thanks to the order system created and present in the EA, you can remove some things without any damage. Therefore, it is up to you whether to accept and use things or to delete them.

### 2.0. Deleting Chart Trade

Chart Trade is something that still makes sense in a simple order system, which is less complex that the one used on our EA. But for our EA at the current stage of development, it does not make sense to have Chart Trade on the chart, so we can remove it. You can keep it if you wish by simply editing one command. True, I like to keep things simple, so that later I can change (or not change) the things very quickly, so that the changes do not generate any stress, such as problems or catastrophic failures at critical points.

To make the control very fast and at the same time safe, the following definition has been added to the EA code:

```
#define def_INTEGRATION_CHART_TRADER            // Chart trader integration with the EA ...
```

If this definition does not exist or becomes a comment, then CHART TRADE will not be compiled with the EA. Let's look at the points affected by this definition. The first and most obvious one includes the following:

```
#ifdef def_INTEGRATION_CHART_TRADER
        #include <NanoEA-SIMD\SubWindow\C_TemplateChart.mqh>
#endif
```

Although the above code is not in the EA file, but in the C\_IndicatorTradeView.mqh file, the definition will be visible to the compiler everywhere in the code, so we don't need to worry about correcting the code. Here we simply create the definition in an easily accessible place, in this case in the EA code, and use it where necessary.

But let's continue with the C\_IndicatorTradeView.mqh file. Since we can compile the EA without Chart Trade, we need to implement access to the data defined in the EA initialization message box, which can be seen in the image below:

![](https://c.mql5.com/2/46/001.png)

Remember, we need to access this data. Previously we passed it to Chart Trade, and when we needed to know them, we addressed Chart Trade. But now, without Chart Trade, we will have to go a different way to access the same data.

In the C\_IndicatorTradeView.mqh file, these values are used only in one place — when we create indicator 0 which shows where the pending order will be located. This place is inside the DispatchMessage function. It is shown in the code below:

```
// ... Previous code ...

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
#ifdef def_INTEGRATION_CHART_TRADER
                                                                m_Selection.bIsDayTrade = Chart.GetBaseFinance(m_Selection.vol, valueTp, valueSl);
#else
                                                                m_Selection.vol = EA_user20 * Terminal.GetVolumeMinimal();
                                                                valueTp = EA_user21;
                                                                valueSl = EA_user22;
                                                                m_Selection.bIsDayTrade = EA_user23;
#endif
                                                                valueTp = Terminal.AdjustPrice(valueTp * Terminal.GetAdjustToTrade() / m_Selection.vol);
                                                                valueSl = Terminal.AdjustPrice(valueSl * Terminal.GetAdjustToTrade() / m_Selection.vol);
                                                                m_Selection.it = IT_PENDING;
                                                                m_Selection.pr = price;
                                                        }

// ... The rest of the code...
```

Pay attention to the highlighted lines, There is no need to search these **EA\_userXX** values inside the file - they are not there, because they come from the EA code, as follows:

```
#ifdef def_INTEGRATION_CHART_TRADER
        input group "Chart Trader"
#else
        input group "Base Operacional do EA"
#endif
input int       EA_user20   = 1;     //Levering factor
input double    EA_user21   = 100;   //Take Profit (financial)
input double    EA_user22   = 81.74; //Stop Loss (financial)
input bool      EA_user23   = true;  //Day Trade ?
```

This alone already provides control similar to having Chart Trade on the chart. Please note that we practically do not change anything in the code. We simply move the required data defined by the user to the correct place. Some people may find this configuration on the trader part when loading the EA unnecessary, and this is true in a sense, since the order system allows configuring all variables without any difficulty. So, we can just set the minimum value of the levering factor and Stop Loss and Take Profit at 0, and the initial operations as Day Trade — do this in the DispatchMessage function of the C\_IndicatorTradeView class. This will not affect the system at all, since the trader can change the order on the chart and then send it to the server. This type of modification is up to you, as it is something very personal.

### 2.0.1. Some adjustments

Before we get back to the part where we remove the Chart Trade, we need to do one more thing that will improve the stability of the EA as a whole.

Let's do the following. In the C\_IndicatorTradeView class, we define a private data structure, which can be seen below:

```
struct st01
{
        bool    ExistOpenPosition,
                SystemInitilized;
}m_InfoSystem;
```

It must be initialized in the following code:

```
void Initilize(void)
{
        static int ot = 0, pt = 0;

        m_InfoSystem.ExistOpenPosition = false;
        m_InfoSystem.SystemInitilized = false;
        ChartSetInteger(Terminal.Get_ID(), CHART_SHOW_TRADE_LEVELS, false);
        ChartSetInteger(Terminal.Get_ID(), CHART_DRAG_TRADE_LEVELS, false);
        if ((ot != OrdersTotal()) || (pt != PositionsTotal()))
        {
                ObjectsDeleteAll(Terminal.Get_ID(), def_NameObjectsTrade);
                ChartRedraw();
                for (int c0 = ot = OrdersTotal(); c0 >= 0; c0--)  IndicatorAdd(OrderGetTicket(c0));
                for (int c0 = pt = PositionsTotal(); c0 >= 0; c0--) IndicatorAdd(PositionGetTicket(c0));
        }
        m_InfoSystem.SystemInitilized = true;
}
```

Why do we create and initialize this data here? Remember that MetaTrader 5 sends events to the EA, and one of the events is OnTick. There are not many problems with simple systems. But as the system becomes more complex, you need o make sure everything works. It may happen so that MetaTrader 5 will send events to the EA before the EA is ready to process these events. Therefore, we must make sure the EA is ready/ We will create some variables that will indicate the EA's readiness state. If it is not yet ready, the events of MetaTrader 5 should be ignored until the EA can respond appropriately to the events.

The most critical point can be seen in the code below:

```
inline double SecureChannelPosition(void)
                        {
                                static int nPositions = 0;
                                double Res = 0;
                                ulong ticket;
                                int iPos = PositionsTotal();

                                if (!m_InfoSystem.SystemInitilized) return 0;
                                if ((iPos != nPositions) || (m_InfoSystem.ExistOpenPosition))
                                {
                                        m_InfoSystem.ExistOpenPosition = false;
                                        for (int i0 = iPos - 1; i0 >= 0; i0--) if (PositionGetSymbol(i0) == Terminal.GetSymbol())
                                        {
                                                m_InfoSystem.ExistOpenPosition = true;
                                                ticket = PositionGetInteger(POSITION_TICKET);
                                                if (iPos != nPositions) IndicatorAdd(ticket);
                                                SetTextValue(ticket, IT_RESULT, PositionGetDouble(POSITION_VOLUME), Res += PositionGetDouble(POSITION_PROFIT), PositionGetDouble(POSITION_PRICE_OPEN));
                                        }
                                        nPositions = iPos;
                                }
                                return Res;
                        };
```

Highlighted points did not exist before, so strange things could sometimes happen. But now we have the necessary checks to ensure nothing unusual goes unnoticed.

This question of checking everything is something quite complicated, since multiple checks can make a system very stable but can also complicate code maintenance and modification. Some of the checks must be performed in a logical order to actually be effective, which is a rather expensive thing to do.

But note that when we check whether or not there is a position for a symbol monitored by the EA, we can give some agility. Even more so, when trading multiple assets each of which has a certain number of position that will appear in the EA. By filtering this here, we eliminate the loop so that the EA only executes the code inside the loop when is really needed. Otherwise, we reduce the processing time consumed by the EA a bit. It is not long, but in very extreme cases it can make a big difference.

### 2.0.2. Removing the Chart Trade from the EA

Now that we have made changes to the C\_IndicatorTradeView class, we can focus on the EA code and delete Chart Trade from it. The first thing to do is delete it from the OnInit code:

```
int OnInit()
{
        Terminal.Init();

#ifdef def_INTEGRATION_WITH_EA
        WallPaper.Init(user10, user12, user11);
        VolumeAtPrice.Init(user32, user33, user30, user31);
        TimesAndTrade.Init(user41);
        EventSetTimer(1);
#endif

        Mouse.Init(user50, user51, user52);

#ifdef def_INTEGRATION_CHART_TRADER
        static string   memSzUser01 = "";
        if (memSzUser01 != user01)
        {
                Chart.ClearTemplateChart();
                Chart.AddThese(memSzUser01 = user01);
        }
        Chart.InitilizeChartTrade(EA_user20 * Terminal.GetVolumeMinimal(), EA_user21, EA_user22, EA_user23);
        TradeView.Initilize();
        OnTrade();
#else
        TradeView.Initilize();
#endif

        return INIT_SUCCEEDED;
}
```

The green code will be replaced with the blue code. If we do not use Chart Trade, it may seem that the difference is small, that this is only a change in the executable file size. But it's not only that. Please note that in addition to the Chart Trade code we have also removed the OnTrade event. The EA will no longer process this event.

You might think that something is wrong with me. How could I remove the OnTrade event from an EA? How are we going to handle trading events now? These events will be processed by the OnTradeTransaction event. The processing method will be more efficient than OnTrade, which means that the EA will be simpler and more reliable.

There is another moment that is undergoing changes:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        Mouse.DispatchMessage(id, lparam, dparam, sparam);
#ifdef def_INTEGRATION_WITH_EA
        switch (id)
        {
                case CHARTEVENT_CHART_CHANGE:
                        Terminal.Resize();
                        WallPaper.Resize();
                        TimesAndTrade.Resize();
        break;
        }
        VolumeAtPrice.DispatchMessage(id, sparam);
#endif

#ifdef def_INTEGRATION_CHART_TRADER
        Chart.DispatchMessage(id, lparam, dparam, sparam);
#endif

        TradeView.DispatchMessage(id, lparam, dparam, sparam);
}
```

If there is no integration within the EA, the only lines that will be compiled will be the highlighted lines. Because these events tend to be fairly constant (the more efficient the handling of these events, the better), they often compete with another event that MetaTrader 5 triggers for the EA to process. This other event is shown below:

```
void OnTick()
{
#ifdef def_INTEGRATION_CHART_TRADER
        Chart.DispatchMessage(CHARTEVENT_CHART_CHANGE, 0, TradeView.SecureChannelPosition(), C_Chart_IDE::szMsgIDE[C_Chart_IDE::eRESULT]);
#else
        TradeView.SecureChannelPosition();
#endif

#ifdef def_INTEGRATION_WITH_EA
        TimesAndTrade.Update();
#endif
}
```

Such an event is a real nightmare because it is usually called multiple times, and in some cases it can be called can be multiple times in less than 1 second. But thanks to the changes made to the code of the C\_IndicatorTradeView class, the processing of this event has become a little more efficient. Later we will further improve this efficiency, but for now this will be enough.

Well, after all these changes, Chart Trade will not be integrated into the EA. We can turn Chart Trade into an indicator, which will have some benefits while keeping the EA operation focused on the main activity: handling, positioning and supporting the order system. But moving Chart Trade into an indicator involves some additional changes, so I will not show now how to do it, we will see it at another time. But generally yes, we can move Chart Trade into an indicator and still be able to send orders through it.

### 3.0. Adding sounds

Often we do not look at the chart, but still want to know what is happening at the moment. One of the ways to get notified about something is to receive a sound alert. This is one of the best alert types because it really grabs our attention right away. Sometimes we already know how to act just by listening to the alert, without having to check any other message.

So, let's learn to set up some basic sound alerts. In some cases, this will be a sentence that tells something specific. Although what I am showing now and what is available in the attachment provides only basic features, perhaps it may motivate you to increase the amount of existing alerts and warnings, so that you will not have to waste time reading messages. A sound can indicate a specific event giving you a gain in agility at specific trading moments. Believe me, this does a lot difference.

The first thing to do is to create a new file which will contain the new class and which will support and isolate our sound system. Once this is done we can start producing things in a very stable way. The entire class can be seen in the code below:

```
class C_Sounds
{
        protected:
                enum eTypeSound {TRADE_ALLOWED, OPERATION_BEGIN, OPERATION_END};
        public  :
//+------------------------------------------------------------------+
inline bool PlayAlert(const int arg)
                {
                        return PlaySound(StringFormat("NanoEA-SIMD\\RET_CODE\\%d.wav", arg));
                }
//+------------------------------------------------------------------+
inline bool PlayAlert(const eTypeSound arg)
                {
                        string str1;

                        switch (arg)
                        {
                                case TRADE_ALLOWED   : str1 = def_Sound00; break;
                                case OPERATION_BEGIN : str1 = def_Sound01; break;
                                case OPERATION_END   : str1 = def_Sound02; break;
                                default : return false;
                        }
                        PlaySound("::" + str1);

                        return true;
                }
//+------------------------------------------------------------------+
};
```

Despite the extreme simplicity of this code, there is something interesting in it. Note that we are rewriting the **PlayAlert** function, so we have two versions of the same function. What for? The way the sound system will work requires form us to have two variations. In the case of the first version of the function, we will play a sound from a file; in the second version we will play the sound that is part of the EA, i.e. its function. Now there is something that many people may not know how to do: to play sounds directly from audio files. But don't worry, I'll show you how. The reason for the first version is that some people may want to put their own voice or other sound as an alert and change it at any time without having to recompile the EA. In fact you can change these sounds even when the EA is running in MetaTrader 5. From the moment the sound should be played, the EA will use the latest version, so you only need to replace one audio file with another — the EA won't even notice the difference, that helps a lot. But there is another reason that can notice from the EA code.

In fact, the first option is used in a rather specific place, as can be seen in the class code highlighted below:

```
class C_Router
{
        protected:
        private  :
                MqlTradeRequest TradeRequest;
                MqlTradeResult  TradeResult;
//+------------------------------------------------------------------+
inline bool Send(void)
        {
                if (!OrderSend(TradeRequest, TradeResult))
                {
                        if (!Sound.PlayAlert(TradeResult.retcode))Terminal.MsgError(C_Terminal::FAILED_ORDER, StringFormat("Error Number: %d", TradeResult.retcode));

                        return false;
                }
                return true;
        }

// ... The rest of the class code....
```

Just imagine the amount of work that would be required to process all the possible errors returned by the trade server. But if we record an audio file and name it in accordance with the value that the trade server returns, the EA will know which file to play, and this which greatly simplifies our life. Because here we only need to specify which file to use based on the value returned by the server — the EA will find and play this file, giving us a sound warning or a voice message so that we know exactly what happened. Wonderful, isn't it? Now, for example if an order is rejected, the platform will inform us in a very clear way what happened or what is wrong. This will be done in a very adequate way that represents something specific to you, something that can be exclusive and unique to the way you trade and act in the market. See how much agility you will be gaining, because in the same audio file you can make it clear how to solve the problem.

But we also have a second mode of operation, in which the sounds are stored inside the executable EA file. This is the second version of the same function, and it will be used at this stage in 3 different places to indicate 3 different types of events. The first place can be seen in the code below:

```
int OnInit()
{
        if (!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
        {
                Sound.PlayAlert(C_Sounds::TRADE_ALLOWED);
                return INIT_FAILED;
        }

        Terminal.Init();

// ... The rest of the code...
```

This code checks if AlgoTrading is enabled in the platform. If we forgot to enable it, the EA will tell us that it will not be available for trading. To check whether the option is enabled or not, just look at the marker in the platform, as shown in the following image:

![](https://c.mql5.com/2/46/002.png)

The second place where we will use an auxiliary sound is shown below:

```
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
                        Sound.PlayAlert(C_Sounds::OPERATION_BEGIN);
                        m_InfoSystem.ExistOpenPosition = true;
                        break;
        }
        m_BtnClose.Create(ticket, macroMountName(ticket, it, EV_CLOSE), def_BtnClose);
}
```

Each time a position indicator is created, a sound will be played. This will make life much easier for us, as we will know that the pending order has become a position and that we need to start paying attention to it.

The third and final point where we have an auxiliary sound is when a position is closed for whatever reason. This is done in a very specific place:

```
inline void RemoveIndicator(ulong ticket, eIndicatorTrade it = IT_NULL)
{
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
        if ((it == IT_NULL) || (it == IT_PENDING) || (it == IT_RESULT))
        {
                if (macroGetPrice(ticket, IT_RESULT, EV_LINE) > 0) Sound.PlayAlert(C_Sounds::OPERATION_END);
                ObjectsDeleteAll(Terminal.Get_ID(), StringFormat("%s%c%llu%c", def_NameObjectsTrade, def_SeparatorInfo, ticket, (ticket > 1 ? '*' : def_SeparatorInfo)));
        } else ObjectsDeleteAll(Terminal.Get_ID(), StringFormat("%s%c%llu%c%c", def_NameObjectsTrade, def_SeparatorInfo, ticket, def_SeparatorInfo, (char)it));
        ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
        m_Selection.ticket = 0;
        Mouse.Show();
        ChartRedraw();
}
```

You might think that when you remove the EA it will play the position closing sound, but no. This won't happen because the price line will still be present in the system. But when the position is closed something different will happens, and the position price line will be at a price level 0 — at this moment a sound will play indicating that the position has closed.

Since these sounds that are resources of the EA will follow the executable file wherever it goes, and that they cannot be modified without recompiling the code, they are more limited, but at the same time they help to port the EA to other places without we need to take audio files along.

But in the case of the sounds used to alert about failures or errors, the logic is different: they must be moved separately and placed in a predetermined place so that they can work when needed.

The attachment contains a folder called SOUNDS. Do not leave this folder in the same folder where the code will be, because the sounds contained in this folder will not be played. Move it to a different location that can be easily found. If you don't know where it is, don't worry — we will see it later:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
void OnStart()
{
        Print(TerminalInfoString(TERMINAL_PATH));
}
//+------------------------------------------------------------------+
```

When you run this script, you will get information in the toolbar that indicates the location we will be using. An example of the execution result is shown in the image below:

![](https://c.mql5.com/2/46/003.png)

You should do the following:

1. Open the attachment
2. Open File Explorer
3. Go to the folder shown in the figure above
4. Copy the contents of the SOUNDS folder from the attached file to the folder indicated above
5. If you want, you can delete these three files (WARRING, BEGIN, END), because they will be compiled together with the EA
6. If you want, you can change the contents of .WAV to something you like; just make sure not to change the name
7. Use the EA in the MetaTrader 5 platform and be happy!

But remember that in order for the sounds (WARRING, BEGIN, END) to be compiled in the EA, you should have the SOUNDS folder with the same sounds in the MQL5 code directory, otherwise they will not be integrated into the EA code.

### Conclusion

In this article, you have learned how to add custom sounds to the EA system. Here we have used a very simple system to demonstrate how this is done, but you are not limited to just an EA, you can also use it in indicators or even in scripts.

The great thing is that if you use the same concepts and ideas proposed here, you can record voice messages saying, or warning about something. And when the EA, or any other process that is being executed by MetaTrader 5 and that uses the sound system, are activated through the triggers shown in the article, you will receive such a sound message telling you or warning you about some kind of action that you had already foreseen and that you should do.

And this is not a text, but a voice message, which makes it much more effective, as you can quickly explain to anyone using the system what should be done, or what was the cause that generated such a message.

This system is not limited to this scheme, and you can go beyond what has been demonstrated here. The idea is precisely this: to allow the user to have an ally in the platform. By having a voice interaction with the trader, we can convey a message that is perhaps better to understand than plain text. The only thing that limits you here is your creativity.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10664](https://www.mql5.com/pt/articles/10664)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10664.zip "Download all attachments in the single ZIP archive")

[EA\_-\_h\_Parte\_29\_6.zip](https://www.mql5.com/en/articles/download/10664/ea_-_h_parte_29_6.zip "Download EA_-_h_Parte_29_6.zip")(14465.62 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/436189)**
(2)


![Alexey Volchanskiy](https://c.mql5.com/avatar/2018/8/5B70B603-444A.png)

**[Alexey Volchanskiy](https://www.mql5.com/en/users/vdev)**
\|
20 Oct 2022 at 13:31

How wonderful that everything is in Portuguese! I thought the international language was English. Let's have an article with Japanese characters!


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
20 Oct 2022 at 13:39

There are 2 pictures in the article, what exactly do you not understand?


![DoEasy. Controls (Part 20): SplitContainer WinForms object](https://c.mql5.com/2/49/MQL5-avatar-doeasy-library-2__1.png)[DoEasy. Controls (Part 20): SplitContainer WinForms object](https://www.mql5.com/en/articles/11524)

In the current article, I will start developing the SplitContainer control from the MS Visual Studio toolkit. This control consists of two panels separated by a vertical or horizontal movable separator.

![How to deal with lines using MQL5](https://c.mql5.com/2/50/How_to_deal_with_lines_by_MQL5_Avatar.png)[How to deal with lines using MQL5](https://www.mql5.com/en/articles/11538)

In this article, you will find your way to deal with the most important lines like trendlines, support, and resistance by MQL5.

![Data Science and Machine Learning (Part 09): The K-Nearest Neighbors Algorithm (KNN)](https://c.mql5.com/2/50/k_nearest_neighbors_algorithm_knn_avatar.png)[Data Science and Machine Learning (Part 09): The K-Nearest Neighbors Algorithm (KNN)](https://www.mql5.com/en/articles/11678)

This is a lazy algorithm that doesn't learn from the training dataset, it stores the dataset instead and acts immediately when it's given a new sample. As simple as it is, it is used in a variety of real-world applications.

![Neural networks made easy (Part 27): Deep Q-Learning (DQN)](https://c.mql5.com/2/48/Neural_networks_made_easy.png)[Neural networks made easy (Part 27): Deep Q-Learning (DQN)](https://www.mql5.com/en/articles/11369)

We continue to study reinforcement learning. In this article, we will get acquainted with the Deep Q-Learning method. The use of this method has enabled the DeepMind team to create a model that can outperform a human when playing Atari computer games. I think it will be useful to evaluate the possibilities of the technology for solving trading problems.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/10664&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051662898075391035)

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