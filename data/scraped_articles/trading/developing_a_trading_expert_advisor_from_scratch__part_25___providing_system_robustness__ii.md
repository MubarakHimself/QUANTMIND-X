---
title: Developing a trading Expert Advisor from scratch (Part 25): Providing system robustness (II)
url: https://www.mql5.com/en/articles/10606
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:46:26.452959
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/10606&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051673858831930480)

MetaTrader 5 / Examples


### Introduction

In the previous article [Providing system robustness (I)](https://www.mql5.com/en/articles/10593), we have seen how to change some parts of the EA to make the system more reliable and robust.

This was only an introduction to what we are going to do in this article. Forget everything you knew, planned, or wished for. The most difficult thing here is to be able to separate things. Since the beginning of this series, the EA has been evolving almost constantly: we have been adding, changing and even removing some things. This time we will go to the extremes with what we have been doing.

Contrary to what it may seem, there is a big issue: a well designed EA does not and will not contain any kind of indicator inside. It will only observe and ensure that the indicated order positions are respected. The perfect EA is essentially just a wizard that provides real insight into what the price is doing. It does not look at the indicators, while it only looks at the positions or orders that are on the chart.

You might think that I'm talking nonsense and that I don't know what I'm talking about. But have you ever thought why MetaTrader 5 provides different classes for different things? Why does the platform have indicators, services, scripts and Expert Advisors separately and not in one block? So...

This is the point. If things are separated, it is precisely because they are better worked on separately.

Indicators are used for a general purpose, whatever it may be. It is good if the design of the indicators is well thought out so as not to harm the overall performance — I mean not to harm the MetaTrader 5 platform, but other indicators. Because they run on a different thread, they can perform tasks in parallel very efficiently.

Services assist in different ways. For example, in articles [Accessing data on the web (II)](https://www.mql5.com/en/articles/10442) and [Accessing data on the web (III)](https://www.mql5.com/en/articles/10447) within this series, we used services to access data in a very interesting way. In fact, we could do this directly in the EA, but this is not the most suitable method, as I have already explained in other articles.

Scripts help us in a very unique way as they only exist for a certain amount of time, do something very specific and then disappear from the chart. Or they can stay there until we change some chart setting like, for example, the timeframe.

This limits the possibilities a little, but this is part of what we have to accept as it is. Expert Advisors, or EAs, on the contrary, are specific to working with a trading system. Although we can add functions and codes that are not part of the trading system in EAs, this is not very appropriate in high-performance or high-reliability systems. The reason is that everything that is not part of the trading system should not be in the EA: things should be placed in the right places and handled correctly.

Therefore, the first thing to do to improve reliability is to remove absolutely everything from the code that is not part of the trading system, and turn these things into indicators or something like that. The only thing that will remain in the EA code is the parts responsible for managing, analyzing and processing orders or positions. All other things will be removed.

So, let's get started.

### 2.0. Implementation

**2.0.1. Removing the EA background**

While this does not harm the EA or cause any problems, some people sometimes want their screen to be blank with only certain items displayed on it. So, we will remove this part from the EA and turn it into an indicator. It is very easy to implement. We will not touch any of the classes, but will create the following code:

```
#property copyright "Daniel Jose"
#property indicator_chart_window
#property indicator_plots 0
//+------------------------------------------------------------------+
#include <NanoEA-SIMD\Auxiliar\C_Wallpaper.mqh>
//+------------------------------------------------------------------+
input string                    user10 = "Wallpaper_01";        //Used BitMap
input char                      user11 = 60;                    //Transparency (from 0 to 100)
input C_WallPaper::eTypeImage   user12 = C_WallPaper::IMAGEM;   //Background image type
//+------------------------------------------------------------------+
C_Terminal      Terminal;
C_WallPaper WallPaper;
//+------------------------------------------------------------------+
int OnInit()
{
        IndicatorSetString(INDICATOR_SHORTNAME, "WallPaper");
        Terminal.Init();
        WallPaper.Init(user10, user12, user11);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
        return rates_total;
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        switch (id)
        {
                case CHARTEVENT_CHART_CHANGE:
                        Terminal.Resize();
                        WallPaper.Resize();
        break;
        }
        ChartRedraw();
}
//+------------------------------------------------------------------+
```

As you can see, everything is quite natural and understandable. We have simply deleted the code from the EA and converted it into an indicator which can be added to the chart. And any change, be it the background, the level of transparency, or even removing it from the chart, will have no effect on the EA actions.

And now we will start to delete the things that really cause EA performance degradation. These are the things that work from time to time or with every price movement, and therefore can sometimes cause the EA to slow down, which prevents it from doing its real job - watching what is happening with orders or positions on the chart.

### 2.0.2. Converting Volume At Price to an indicator

While it may not seem like this, the Volume At Price system takes time which is often critical for an EA. I mean the moments of high volatility when prices fluctuate wildly without much direction. It is at these times that the EA needs every available machine cycle to complete its task. It would be upsetting to miss a good opportunity because some indicator decides to take over the job. So, let's remove it from the EA and turn it into a real indicator by creating the code below:

```
#property copyright "Daniel Jose"
#property indicator_chart_window
#property indicator_plots 0
//+------------------------------------------------------------------+
#include <NanoEA-SIMD\Tape Reading\C_VolumeAtPrice.mqh>
//+------------------------------------------------------------------+
input color             user0   = clrBlack;                     //Bar color
input   char            user1   = 20;                                   //Transparency (from 0 to 100 )
input color     user2 = clrForestGreen; //Buying
input color     user3 = clrFireBrick;   //Selling
//+------------------------------------------------------------------+
C_Terminal                      Terminal;
C_VolumeAtPrice VolumeAtPrice;
//+------------------------------------------------------------------+
int OnInit()
{
        Terminal.Init();
        VolumeAtPrice.Init(user2, user3, user0, user1);
        EventSetTimer(1);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
        return rates_total;
}
//+------------------------------------------------------------------+
void OnTimer()
{
        VolumeAtPrice.Update();
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        VolumeAtPrice.DispatchMessage(id, sparam);
        ChartRedraw();
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        EventKillTimer();
}
//+------------------------------------------------------------------+
```

This was the easiest part. We removed code from the EA and put it into the indicator. If you want to put the code back into the EA, you just need to copy the indicator code and put it back into the EA.

So, we have started with something simple. But now things become more complicated — we are going to remove Times & Trade from the EA.

### 2.0.3. Transforming Times & Trade into an indicator

This is not that simple if we aim to create the code that can be used both in an EA and in an indicator. Being an indicator that works in a subwindow, it would seem that converting it to an indicator would be easy. But it is not easy exactly because it works in a subwindow. The main problem is that if we just do everything as in the previous cases, then we will have the following result in the indicator window:

![](https://c.mql5.com/2/45/001__7.png)

Placing such things in the indicator window is not recommended, as this will confuse the user if he wants to remove the indicator from the screen. So, this should be done in a different way. And at the end of this path, which may seem quite confusing but is actually a simple set of directives and some editing, we will get the following result in the indicator window.

![](https://c.mql5.com/2/45/002__6.png)

This is exactly what the user expects — not the mess seen in the picture above.

Below is the full code of the Times & Trade indicator:

```
#property copyright "Daniel Jose"
#property version   "1.00"
#property indicator_separate_window
#property indicator_plots 0
//+------------------------------------------------------------------+
#include <NanoEA-SIMD\Tape Reading\C_TimesAndTrade.mqh>
//+------------------------------------------------------------------+
C_Terminal        Terminal;
C_TimesAndTrade   TimesAndTrade;
//+------------------------------------------------------------------+
input int     user1 = 2;      //Scale
//+------------------------------------------------------------------+
bool isConnecting = false;
int SubWin;
//+------------------------------------------------------------------+
int OnInit()
{
        IndicatorSetString(INDICATOR_SHORTNAME, "Times & Trade");
        SubWin = ChartWindowFind();
        Terminal.Init();
        TimesAndTrade.Init(user1);
        EventSetTimer(1);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
        if (isConnecting)
                TimesAndTrade.Update();
        return rates_total;
}
//+------------------------------------------------------------------+
void OnTimer()
{
        if (TimesAndTrade.Connect())
        {
                isConnecting = true;
                EventKillTimer();
        }
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        switch (id)
        {
                case CHARTEVENT_CHART_CHANGE:
                        Terminal.Resize();
                        TimesAndTrade.Resize();
        break;
        }
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        EventKillTimer();
}
//+------------------------------------------------------------------+
```

The code seems similar to the one used in the EA, except for the highlighted line which is not present in the EA code. Then what's the catch? Or there is none? Actually, there is some catch: the code is not exactly the same, there is a difference in it, which is not in the indicator or EA code, but in the class code. But before considering the difference, let's think about the following: how do we tell the compiler what to compile and what not to compile? Perhaps, when programming, you do not worry about this at all — perhaps, you simply create code and if you do not like anything, you simply delete.

Experienced programmers have a rule: only remove something when it definitely doesn't work, otherwise keep fragments even if they aren't actually compiled. But how to do this in a linear code, when we want the written functions to always work? Here is the question: Do you know how to tell the compiler what to compile and what not to compile? If the answer is "No" then it's okay. I personally didn't know how to do it when I started. But it helps a lot. So, let's find out how to do it.

Some languages have compilation directives, which may be also referred to as the [preprocessor](https://www.mql5.com/en/docs/basis/preprosessor), depending on the author. But the idea is the same: tell the compiler what to compile and how to do the compilation. There is a very specific type of directive which can be used to isolate code intentionally so that we can test specific things. These are [conditional compilation](https://www.mql5.com/en/docs/basis/preprosessor/conditional_compilation) directives. When used properly, they allow us to compile the same code in different ways. This is exactly what is done here with the Times & Trade example. We choose who will be responsible for generating the conditional compilation: either the EA or the indicator. After defining this parameter, create the **#define** directive and then use the conditional directive **#ifdef #else #endif** to inform the compiler how the code will be compiled.

This can be difficult to understand, so let's see how it works.

In the EA code, define and add the lines highlighted below:

```
#define def_INTEGRATION_WITH_EA
//+------------------------------------------------------------------+
#include <NanoEA-SIMD\Trade\Control\C_IndicatorTradeView.mqh>
#ifdef def_INTEGRATION_WITH_EA
        #include <NanoEA-SIMD\Auxiliar\C_Wallpaper.mqh>
        #include <NanoEA-SIMD\Tape Reading\C_VolumeAtPrice.mqh>
        #include <NanoEA-SIMD\Tape Reading\C_TimesAndTrade.mqh>
#endif
//+------------------------------------------------------------------+
```

The following happens: If you want to compile an EA with classes in MQH files, leave the **#ifdefine def\_INTEGRATION\_WIT\_EA** directive which is defined in the Expert Advisor. This will make the EA contain all the classes that we take and insert into the indicators. If you want to delete the indicators, there is no need to delete the code, while you should simply comment the definition. This can be done simply by converting the line where the directive is declared into a comment line. The compiler will not see the directive, it will be given as non-existent; since it does not exist, every time the conditional directive **#ifdef def\_INTEGRATION\_WITH\_EA** is found, it will be fully ignored, while code between it and the **#endif** part in the example above will not be compiled.

This is the idea, which we implement in the C\_TimesAndTrade class. Here is how the new class looks like. I will show only one point to grab your attention:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include <NanoEA-SIMD\Auxiliar\C_Canvas.mqh>
#ifdef def_INTEGRATION_WITH_EA

#include <NanoEA-SIMD\Auxiliar\C_FnSubWin.mqh>

class C_TimesAndTrade : private C_FnSubWin

#else

class C_TimesAndTrade

#endif
{
//+------------------------------------------------------------------+
#define def_SizeBuff 2048
#define macro_Limits(A) (A & 0xFF)
#define def_MaxInfos 257
#define def_ObjectName "TimesAndTrade"
//+------------------------------------------------------------------+
        private :
                string  m_szCustomSymbol;

// ... The rest of the class code....

}
```

The code may seem strange to anyone not using compilation directives. The **def\_INTEGRATION\_WITH\_EA** directive is declared in the EA. Then the following happens. When the compiler generates object code from this file, it will assume the following relation: if the file being compiled is an EA and has a declared directive, the compiler will generate object code with parts that are between the conditional directives **#ifdef def\_INTEGRATION\_WITH\_EA** and **#else**. Usually in such cases we use the #else directive. In case another file is compiled, for example, the indicator whose directive **def\_INTEGRATION\_WITH\_EA** is not defined, everything between the directives **#else** and **#endif** will be compiled. That's how it works.

When compiling an EA or an indicator, look at the entire code of the C\_TimesAndTrade class in order to understand each of these tests and the general operation. Thus, the MQL5 compiler will make all the settings, saving us time and effort associated with the need to maintain two different files.

### 2.0.4. Making the EA more agile

As previously mentioned, the EA should work only with the order system. So far, it has possessed the features that have now become indicators. The reason for this is something very personal, which has to do with the things involved in calculations that the EA should do. But this calculation system has been modified and moved to another method. Due to this, I noticed that the order system was harmed by some things that the EA did instead of taking care of orders. The worst of the problems was in the OnTick event:

```
void OnTick()
{
        Chart.DispatchMessage(CHARTEVENT_CHART_CHANGE, 0, TradeView.SecureChannelPosition(), C_Chart_IDE::szMsgIDE[C_Chart_IDE::eRESULT]);
#ifdef def_INTEGRATION_WITH_EA
        TimesAndTrade.Update();
#endif
}
```

This event has now received this conditional directive so that those who do not trade during periods of high volatility can, if desired, have an EA with all the original indicators. But before you think this is a good idea, let me remind you how the Times & Trade update feature works.

```
inline void Update(void)
{
        MqlTick Tick[];
        MqlRates Rates[def_SizeBuff];
        int i0, p1, p2 = 0;
        int iflag;
        long lg1;
        static int nSwap = 0;
        static long lTime = 0;

        if (m_ConnectionStatus < 3) return;
        if ((i0 = CopyTicks(Terminal.GetFullSymbol(), Tick, COPY_TICKS_ALL, m_MemTickTime, def_SizeBuff)) > 0)
        {

// ... The rest of the code...

        }
}
```

The above code is a part of the update function present in the C\_TimesAndTrade class. The problem is in the highlighted part. Each time it is executed, a request is sent to the server to return all trade tickets made since a certain point in time, which, by the way, is not so problematic. The problem is that from time to time this call will coincide with two other events.

The first and most obvious event is the large number of trades that can take place, which causes the OnTick function to receive a large number of calls. In addition to having to run the above code present in the C\_TimesAndTrade class, this function will deal with another problem: calling the SecureChannelPosition function present in the C\_IndicatorTradeView class. So, it is another small problem, but that's not all. I have already said that from time to time, despite the low volatility, we will have the coincidence of two events, the first of which was this event.

The second one is in the OnTime event which has already been updated and looks as follows:

```
#ifdef def_INTEGRATION_WITH_EA
void OnTimer()
{
        VolumeAtPrice.Update();
        TimesAndTrade.Connect();
}
#endif
```

If you are going to use the EA the way it was designed, also given that it receives even more code, then it may sometimes have problems due to the coinciding events. When this happens, the EA will stay (even if for one single second) doing the things that are not related to the order system.

Unlike the function found in C\_TimesAndTrade, this function is present in the C\_VolumeAtPrice class and can really harm the EA performance when managing orders. Here is why this happens:

```
inline virtual void Update(void)
{
        MqlTick Tick[];
        int i1, p1;

        if (macroCheckUsing == false) return;
        if ((i1 = CopyTicksRange(Terminal.GetSymbol(), Tick, COPY_TICKS_TRADE, m_Infos.memTimeTick)) > 0)
        {
                if (m_Infos.CountInfos == 0)
                {
                        macroSetInteger(OBJPROP_TIME, m_Infos.StartTime = macroRemoveSec(Tick[0].time));
                        m_Infos.FirstPrice = Tick[0].last;
                }
                for (p1 = 0; (p1 < i1) && (Tick[p1].time_msc == m_Infos.memTimeTick); p1++);
                for (int c0 = p1; c0 < i1; c0++) SetMatrix(Tick[c0]);
                if (p1 == i1) return;
                m_Infos.memTimeTick = Tick[i1 - 1].time_msc;
                m_Infos.CurrentTime = macroRemoveSec(Tick[i1 - 1].time);
                Redraw();
        };
};
```

The reason is in the highlighted parts, but the worst of them is **REDRAW**. It greatly harms the EA performance because on each received tick with volume _ABOVE_ the specified value, the entire volume at price is removed from the screen, recalculated and set back in place. This happens every 1 second or so. This may coincide with other things, that is why all indicators are being removed from the EA. Although I left them so that you can use them directly in the EA, I still do not recommend doing so for the reasons explained earlier.

These changes were necessary. But there is another one, which is more emblematic and which needs to be done. This time the change concerns the OnTradeTransaction event. The use of this event is an attempt to make the system as flexible as possible. Many of those who program order executing EAs use the OnTrade event where they check which orders are or are not on the server, or which positions are still open. I'm not saying they're doing it wrong. It's just that it's not very efficient as the server informs us about what's going on. But the big problem with the OnTrade event is the fact that we have to keep checking things unnecessarily. If we use the OnTradeTransaction event, we will have a system at least more efficient in terms of movement analysis. But this is not the objective here. Everyone uses the method that best fits their criteria.

When developing this EA, I decided not to use any storage structure and thus not to limit the number of orders or positions that can be worked with. But this fact complicates the situation so much that an alternative to the OnTrade event is needed, which can be found in using the OnTradeTransaction event.

This event is very difficult to implement, which is probably why it is not used by many. But I had no choice. It either works or it doesn't, otherwise things would be complicated. But in the previous version the code for this event was very inefficient and you can see it below:

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
                         else TradeView.UpdateInfosIndicators(0, trans.position, trans.price, trans.price_tp, trans.price_sl, trans.volume, trans.deal_type == DEAL_TYPE_BUY);
                         if (!PositionSelectByTicket(trans.position)) TradeView.RemoveIndicator(trans.position);
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

While the above code works, it is HORRIBLE to say the least. The number of useless calls generated by the above code is insane. Nothing can improve the EA in terms of stability and reliability if the above code cannot be fixed.

Because of this, I did a few things on a demo account to try and find a pattern in the messages, which is actually quite difficult. I didn't find a pattern, but I did find something that avoided the madness of useless calls that were generated, making the code stable, reliable, and at the same time flexible enough to be able to trade at any time in the market. Of course, there are still a few small bugs to fix, but the code is very good:

```
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
{
#define def_IsBuy(A) ((A == ORDER_TYPE_BUY_LIMIT) || (A == ORDER_TYPE_BUY_STOP) || (A == ORDER_TYPE_BUY_STOP_LIMIT) || (A == ORDER_TYPE_BUY))

        if (trans.type == TRADE_TRANSACTION_HISTORY_ADD) if (trans.symbol == Terminal.GetSymbol()) TradeView.RemoveIndicator(trans.position);
        if (trans.type == TRADE_TRANSACTION_REQUEST) if ((request.symbol == Terminal.GetSymbol()) && (result.retcode == TRADE_RETCODE_DONE)) switch (request.action)
        {
                case TRADE_ACTION_PENDING:
                        TradeView.IndicatorAdd(request.order);
                        break;
                case TRADE_ACTION_SLTP:
                        TradeView.UpdateIndicators(request.position, request.tp, request.sl, request.volume, def_IsBuy(request.type));
                        break;
                case TRADE_ACTION_DEAL:
                        TradeView.RemoveIndicator(request.position);
                        break;
                case TRADE_ACTION_REMOVE:
                        TradeView.RemoveIndicator(request.order);
                        break;
                case TRADE_ACTION_MODIFY:
                        TradeView.UpdateIndicators(request.order, request.tp, request.sl, request.volume, def_IsBuy(request.type));
                        break;
        }

#undef def_IsBuy
}
```

Don't try to figure out what's going on right away, just enjoy the beauty of this feature. It's almost living perfection. I say this not because this is done by me, but because of the degree of robustness and agility it has.

Although it may seem complicated, there are two checks in this code. They are highlighted below to better explain what's going on.

```
if (trans.type == TRADE_TRANSACTION_HISTORY_ADD) if (trans.symbol == Terminal.GetSymbol()) TradeView.RemoveIndicator(trans.position);
if (trans.type == TRADE_TRANSACTION_REQUEST) if ((request.symbol == Terminal.GetSymbol()) && (result.retcode == TRADE_RETCODE_DONE)) switch (request.action)
{

//... inner code ...

}
```

The line highlighted in GREEN will check each time a trade occurs in history to see if its asset is the same as the asset observed by the EA. If it is so, the C\_IndicatorTradeView class will receive a command to delete the indicator from the chart. this can happen in two cases: when an order becomes a position and when a position closes. Please note that I only use the NETTING mode and not HEDGING. Thus, no matter what happens, the indicator will be removed from the chart.

One could ask: If the position is being closed, it's ok; but what if the order becomes a position — will I be helpless? No. But the problem is solved not inside the error but inside the C\_IndicatorTradeView class. We will consider it in the next section of the article.

The red line, on the other hand, absurdly reduces the amount of useless messages that were forwarded to the C\_IndicatorTradeView class. This is done by checking the response returned by the server to the request, so we need to get confirmation by raising the request along with the same name of the asset that the EA is tracking. Only then a new round of calls will be sent to the C\_IndicatorTradeView class.

That's all I can say about this system. But the story is not over yet. We have a lot of work ahead of us, and from now on we will focus only on the C\_IndicatorTradeView class. We will start now with some changes that need to be made.

### 2.0.5. Reducing the number of objects created by C\_IndicatorTradeView

In the article [Developing a trading Expert Advisor from scratch (Part 23)](https://www.mql5.com/en/articles/10563) I introduced a rather abstract but very interesting concept of shifting orders or stop levels. The concept was to use position ghosts or shadows. They define and display on the chart what the trade server sees and are used until the actual move occurs. This model has a small problem: it adds objects to be managed by MetaTrader 5, but the added objects are not needed in most cases, so MetaTrader 5 gets a list of objects that is often full of useless or rarely used things.

But we don't want the EA to constantly create objects or to keep unnecessary objects in the list, as this degrades the EA performance. Since we use MetaTrader 5 to manage order, we should eliminate useless objects that interfere with the whole system.

But there is a very simple solution. It's actually not that simple. We'll be making some more changes to the C\_IndicatorTradeView class to improve it. We will keep ghosts on the screen, and we will use a very curious and little used method.

It will be fun and interesting.

First, we will change the selection structure. It will now look as follows:

```
struct st00
{
        eIndicatorTrade it;
        bool            bIsBuy,
			bIsDayTrade;
        ulong           ticket;
        double          vol,
                        pr,
                        tp,
                        sl;
}m_Selection;
```

I won't tell you exactly what has changed — you should understand for yourself. But the changes simplified some moments of the coding logic.

Thus, our ghost indicator will now have its own index:

```
#define def_IndicatorGhost      2
```

Due to this, the name modeling has also changed:

```
#define macroMountName(ticket, it, ev) StringFormat("%s%c%llu%c%c%c%c", def_NameObjectsTrade, def_SeparatorInfo,\
                                                                        ticket, def_SeparatorInfo,              \
                                                                        (char)it, def_SeparatorInfo,            \
                                                                        (char)(ticket <= def_IndicatorGhost ? ev + 32 : ev))
```

It seems like a small thing, but it will change a lot soon. Let's continue.

Now the price position macros are always straight, there are no more duplications, so our code now looks like below:

```
#define macroSetLinePrice(ticket, it, price) ObjectSetDouble(Terminal.Get_ID(), macroMountName(ticket, it, EV_LINE), OBJPROP_PRICE, price)
#define macroGetLinePrice(ticket, it) ObjectGetDouble(Terminal.Get_ID(), macroMountName(ticket, it, EV_LINE), OBJPROP_PRICE)
```

These changes forced us to create two other functions, now I will show one and then another. The first is the replacement of the function that creates the indicators themselves. It literally made clear what really makes one indicator different from another. This can be seen below:

```
#define macroCreateIndicator(A, B, C, D)        {                                                                       \
                m_TradeLine.Create(ticket, sz0 = macroMountName(ticket, A, EV_LINE), C);                                \
                m_BackGround.Create(ticket, sz0 = macroMountName(ticket, A, EV_GROUND), B);                             \
                m_BackGround.Size(sz0, (A == IT_RESULT ? 84 : 92), (A == IT_RESULT ? 34 : 22));                         \
                m_EditInfo1.Create(ticket, sz0 = macroMountName(ticket, A, EV_EDIT), D, 0.0);                           \
                m_EditInfo1.Size(sz0, 60, 14);                                                                          \
                if (A != IT_RESULT)     {                                                                               \
                        m_BtnMove.Create(ticket, sz0 = macroMountName(ticket, A, EV_MOVE), "Wingdings", "u", 17, C);    \
                        m_BtnMove.Size(sz0, 21, 23);                                                                    \
                                        }else                   {                                                       \
                        m_EditInfo2.Create(ticket, sz0 = macroMountName(ticket, A, EV_PROFIT), clrNONE, 0.0);           \
                        m_EditInfo2.Size(sz0, 60, 14);  }                                                               \
                                                }

                void CreateIndicator(ulong ticket, eIndicatorTrade it)
                        {
                                string sz0;

                                switch (it)
                                {
                                        case IT_TAKE    : macroCreateIndicator(it, clrForestGreen, clrDarkGreen, clrNONE); break;
                                        case IT_STOP    : macroCreateIndicator(it, clrFireBrick, clrMaroon, clrNONE); break;
                                        case IT_PENDING : macroCreateIndicator(it, clrCornflowerBlue, clrDarkGoldenrod, def_ColorVolumeEdit); break;
                                        case IT_RESULT  : macroCreateIndicator(it, clrDarkBlue, clrDarkBlue, def_ColorVolumeResult); break;
                                }
                                m_BtnClose.Create(ticket, macroMountName(ticket, it, EV_CLOSE), def_BtnClose);
                        }
#undef macroCreateIndicator
```

You may have noticed that I love using preprocessing directives in my code. I do this almost all the time. However, as you can see, it is now quite easy to differentiate between indicators. If you want to give the indicator the colors you want, change this code. Since they are all almost identical, by using a macro we can make them all work the same and have the same elements. This is an ultimate code re-use.

There is another function with a name very similar to this one. But it does something different, and I will talk about it in detail at the end.

The IndicatorAdd function has been changed — we deleted some of the fragments.

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
				UpdateIndicators(ticket, m_Selection.tp, m_Selection.sl, m_Selection.vol, m_Selection.bIsBuy);
                        }
```

One of the removed fragments is replaced with the highlighted one. Does it mean that pending order and 0 indicators will no longer be created? They are still created but in a different place. So, there is another function to come.

SO here it is — the function that creates pending order indicators and indicator 0. The code of UpdateIndicators is as follows:

```
#define macroUpdate(A, B) { if (B > 0) {                                                                \
                if (b0 = (macroGetLinePrice(ticket, A) == 0 ? true : b0)) CreateIndicator(ticket, A);   \
                PositionAxlePrice(ticket, A, B);                                                        \
                SetTextValue(ticket, A, vol, (isBuy ? B - pr : pr - B));                                \
                                        } else RemoveIndicator(ticket, A); }

                void UpdateIndicators(ulong ticket, double tp, double sl, double vol, bool isBuy)
                        {
                                double pr;
                                bool b0 = false;

                                if (ticket == def_IndicatorGhost) pr = m_Selection.pr; else
                                {
                                        pr = macroGetLinePrice(ticket, IT_RESULT);
                                        if ((pr == 0) && (macroGetLinePrice(ticket, IT_PENDING) == 0))
                                        {
                                                CreateIndicator(ticket, IT_PENDING);
                                                PositionAxlePrice(ticket, IT_PENDING, m_Selection.pr);
                                                ChartRedraw();
                                        }
                                        pr = (pr > 0 ? pr : macroGetLinePrice(ticket, IT_PENDING));
                                        SetTextValue(ticket, IT_PENDING, vol);
                                }
                                if (m_Selection.tp > 0) macroUpdate(IT_TAKE, tp);
                                if (m_Selection.sl > 0) macroUpdate(IT_STOP, sl);
                                if (b0) ChartRedraw();
                        }
#undef macroUpdate
```

The function has a very interesting check that is highlighted in the code. It will help create ghost indicators, so the IndicatorAdd function will no longer be able to create pending order indicators and indicator 0. But just doing this check is not enough to create a ghost indicator.

The DispatchMessage function now includes some details, these are small changes, but they make our lives much easier. I will show the parts that have changed:

```
void DispatchMessage(int id, long lparam, double dparam, string sparam)
{

// ... Code ....

        switch (id)
        {
                case CHARTEVENT_MOUSE_MOVE:

// ... Code ....
                        }else if ((!bMounting) && (bKeyBuy == bKeySell) && (m_Selection.ticket > def_IndicatorGhost))
                        {
                                if (bEClick) SetPriceSelection(price); else MoveSelection(price);
                        }
                        break;

// ... Code ...

                case CHARTEVENT_OBJECT_CLICK:
                        if (GetIndicatorInfos(sparam, ticket, it, ev)) switch (ev)
                        {
                                case EV_CLOSE:

// ... Code ...

                                        break;
                                case EV_MOVE:
                                        CreateGhostIndicator(ticket, it);
                                        break;
                        }
                break;
        }
}
```

CHARTEVENT\_MOUSE\_MOVE has a modified part. This code will check whether we are working with the ghost. If it is a ghost, the fragment is blocked. But if it is not, the movement is possible (provided that the indicator itself can move).

As soon as we click on the new position of the indicator, the ghost with all its components will be removed from the list of objects. I think it should be clear. Now pay attention to the highlighted point — it is the call of the **_CreateGhostndicator_** function. We will discuss this code in the next section.

### 2.0.6. How CreateGhostIndicator works

CreateGhostIndicator seems like a strange function. Let's look at its code below:

### CreateGhostIndicator

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

It is very interesting that nothing is created in this function. However, if the EA is compiled and executed, it will create ghosts that show the order status on the server. To understand this, watch the following video. This is a demonstration of how the system works in reality.

Video demostração - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10606)

MQL5.community

1.91K subscribers

[Video demostração](https://www.youtube.com/watch?v=XpirZTdusyg)

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

[Watch on](https://www.youtube.com/watch?v=XpirZTdusyg&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10606)

0:00

0:00 / 3:11

•Live

•

Ghost indicators are really created on the chart, but how does this actually happen? How did we manage to create indicators without actually creating them somewhere in the code?

These are ghosts. You won't actually see them being created, there's no point in reading the code trying to find the line that says: **_"HERE... I found... ghost indicators are created at this point..."_** The truth is that they are simply already on the chart, but they are not displayed anywhere until we start manipulating the order or position — only then they become visible. How is this possible?

To understand this, let's consider the EA execution thread.

After EA initialization, we see the following execution thread:

### Thread 1

![init_ea](https://c.mql5.com/2/45/003__10.png)  <<<< System initialization thread

The orange area is part of the EA and the green area is part of the C\_IndicatorTradeView class. See what happens before the indicator is created and displayed on the screen. The black arrows are the common path for pending orders and open positions; the blue arrow is the path of positions, and the purple arrows show the path pending orders take to create their indicators. Of course, there are things inside the functions that direct the thread one way or another, but the diagram here is intended to show how everything works in general terms.

The previous scheme is used only once and only during system startup. Now every time we are going to place a pending order on the chart, we will have two different execution threads: the first one is responsible for creating indicator 0 and trying to place the order on the chart. This is shown in the figure below:

### Thread 2

![](https://c.mql5.com/2/45/004__5.png)     <<<< Indicator 0 initialization thread

Please note that it is not really the class that will create the order that appears on the chart. It will only try to do so. If everything goes well, the SetPriceSelection function will be successfully executed, and a new thread will be created, which will present the order on the chart. Thus, we will get the following thread. It will actually place the order in the place that the trade server reports, so there is no point in waiting until the order actually ends up in the place that we originally specified. If the volatility causes the server to fill the order at a different point than the one we indicated, the EA will correct this and will present the order in the correct place. So, you will only have to analyze if the conditions are suitable for your trading model.

### Thread 3

![](https://c.mql5.com/2/45/005__4.png)     <<< Pending order placing thread

This is just the part responsible for placing the order on the chart. Here I'm talking about a full order, that is, it will have an entry point, a take profit and a stop loss. But what will be the thread if one of the limit orders, be it the take profit or stop loss, are removed from the order? These threads do not respond to this. In fact, the thread will be quite different from the ones here, but the elements will be almost the same. Let's see below what the flow will be like if you click on the button to close one of the limit orders.

It may seem strange.

### Thread 4

![](https://c.mql5.com/2/45/006__4.png)     <<< Deleting an order or stop levels

We have 2 threads, one next to the other. The one marked with a purple arrow will be executed first. As soon as it is executed, the OnTradeTransaction event will capture the response from the server and will trigger the system to remove the indicator from the screen. There is only one difference between the deletion of stop orders and the closing of a position or order: in these cases, the SetPriceSelection function will not be executed, but the OnTradeTransaction event flow will remain.

All this is wonderful, but still it does not answer the question of how ghosts appear.

In order to understand how ghosts are created, we need to know how the execution thread occurs: how the EA places a pending order or how the creation of indicator 0 happens in practice. This [flow is shown in the figure above](https://www.mql5.com/en/articles/10606/#fluxo_2). If you understand the execution threads, it will be easier for you to understand ghosts.

Let's finally see how ghosts are created. Look again at the function [CreateGhostIndicator](https://www.mql5.com/en/articles/10606/#create_ghost). It does not create anything, but simply manipulates some data. Why? Because if we try to create an object, it will be overlaid on existing objects and rendered on top of them. Thus, the required objects will be hidden. There are two solutions to this problem. The first one is to create a set that is inferior to all the others. It will be created before any other object representing orders. But this solution has a problem. We will have a lot of useless objects. But we are changing the entire code to avoid this. The second solution is to create a ghost and then delete the pointer we're manipulating, and then create it again. Neither of these solutions is very practical, moreover, both of them are quite expensive.

While studying the documentation, I found information that caught my attention: the [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) function allows you to manipulate the object property that doesn't make sense at first glance — OBJPROP\_NAME. I was intrigued by why this is allowed. It does not make sense. If the object has already been created, then what's the point of changing its name?

The point is this. When we rename an object, the old object ceases to exist and gets a new name. After renaming, the object takes the place of the original object, so the EA can create the original object without problems, and the ghost can appear and disappear without side effects for graphics and without leaving traces. The only object that needs to be removed is the indicator close button. This is done in this line:

```
ObjectDelete(Terminal.Get_ID(), macroMountName(def_IndicatorGhost, it, EV_CLOSE));
```

There is a minor detail here. Looking at the documentation for the ObjectSetString function, we see a warning about its operation:

When renaming a graphical object, two events are generated simultaneously. These events can be processed in the EA or in the indicator using the [OnChartEvent()](https://www.mql5.com/en/docs/event_handlers/onchartevent) function:

- event of deleting an object with an old name
- event of creating an object with a new name

This is important to consider because we don't want the object we're about to rename to just show up if we're not ready for it. So, we add one more thing before and after the name change:

```
ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);

// ... Secure code...

ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
```

Anything inside the code will not trigger object creation and deletion events. We now have the complete code where the ghosts will appear and we will have the correct behavior.

Perhaps, it is not yet clear how the code actually creates ghosts by simply renaming the indicator. I will leave you here. To help you a bit, I will show how the ghost execution thread looks like. This is shown in the image below:

### Thread 5

![](https://c.mql5.com/2/45/007__4.png)    <<<< Ghost thread

Note that this is a near-perfect clone of thread 2, so you can already have fun knowing how ghosts are created and destroyed, but without actually writing any creation code.

### Conclusion

Being an author, I found this article quite interesting and even exciting. Well, we had to change the EA code a lot. But all this is for the better. There are still a few things and steps that need to be taken to make it even more reliable. However, the already implemented changes will greatly benefit the entire system. I would like to emphasize that a well-designed program usually goes through certain steps that have been implemented here: studying the documentation, analyzing execution threads, benchmarking the system to see if it overloads at critical moments, and above all, being calm, so as not to turn the code into a real monster. It is very important to avoid turning our code into a copy of Frankenstein, because this will not make the code better, but will only make future improvements and especially corrections more difficult.

Warm hugs to everyone who follows this series. Hope to see you in the next article, because we are not finished yet and there is still more to do.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10606](https://www.mql5.com/pt/articles/10606)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10606.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Dando\_Robustez\_ao\_sistema.zip](https://www.mql5.com/en/articles/download/10606/ea_-_dando_robustez_ao_sistema.zip "Download EA_-_Dando_Robustez_ao_sistema.zip")(12032.01 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/434849)**

![Developing a trading Expert Advisor from scratch (Part 26): Towards the future (I)](https://c.mql5.com/2/48/development__2.png)[Developing a trading Expert Advisor from scratch (Part 26): Towards the future (I)](https://www.mql5.com/en/articles/10620)

Today we will take our order system to the next level. But before that, we need to solve a few problems. Now we have some questions that are related to how we want to work and what things we do during the trading day.

![Developing a trading Expert Advisor from scratch (Part 24): Providing system robustness (I)](https://c.mql5.com/2/48/development.png)[Developing a trading Expert Advisor from scratch (Part 24): Providing system robustness (I)](https://www.mql5.com/en/articles/10593)

In this article, we will make the system more reliable to ensure a robust and secure use. One of the ways to achieve the desired robustness is to try to re-use the code as much as possible so that it is constantly tested in different cases. But this is only one of the ways. Another one is to use OOP.

![DoEasy. Controls (Part 15): TabControl WinForms object — several rows of tab headers, tab handling methods](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__3.png)[DoEasy. Controls (Part 15): TabControl WinForms object — several rows of tab headers, tab handling methods](https://www.mql5.com/en/articles/11316)

In this article, I will continue working on the TabControl WinForm object — I will create a tab field object class, make it possible to arrange tab headers in several rows and add methods for handling object tabs.

![Population optimization algorithms](https://c.mql5.com/2/48/logo.png)[Population optimization algorithms](https://www.mql5.com/en/articles/8122)

This is an introductory article on optimization algorithm (OA) classification. The article attempts to create a test stand (a set of functions), which is to be used for comparing OAs and, perhaps, identifying the most universal algorithm out of all widely known ones.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/10606&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051673858831930480)

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