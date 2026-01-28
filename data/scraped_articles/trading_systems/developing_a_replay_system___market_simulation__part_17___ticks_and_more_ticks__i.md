---
title: Developing a Replay System — Market simulation (Part 17): Ticks and more ticks (I)
url: https://www.mql5.com/en/articles/11106
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:20:49.424539
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11106&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070212256236507614)

MetaTrader 5 / Examples


### Introduction

In the previous article " [Developing a Replay System — Market simulation (Part 16): New class system](https://www.mql5.com/en/articles/11095)", we have made the necessary changes to the C\_Replay class. These changes are intended to simplify several tasks that we will need to complete. Thus, the C\_Replay class, which was once too large, went through a simplification process in which its complexity was distributed among other classes. This makes it much simpler and easier to implement new functionality and improvements to the replay/simulation system. Starting with this article, these improvements will begin to appear and extend to the next seven articles.

The first question we'll look at is very difficult to model in a way that anyone can understand just by looking at the code. Knowing this, I would like the reader to pay due attention to the explanations that we will consider throughout these articles. If you are attentive enough, you will be able to follow the explanation as it is really rich and complex. I say this because today's material may seem unnecessary to some, while for others it will be of paramount importance. The material will be presented step by step so that you can follow the reasoning.

The big problem is that all of the previous articles focused solely on chart construction, and that chart had to be presented in such a way that the replay/simulation asset behaved very similarly to what is happening in the real market. I know that there are many who trade using some other tools, such as an order book. Although I personally don't think using such a tool is good practice; other traders believe that there is some correlation between what happens on the order book and what is traded. It is ok if every person has their own point of view. But despite this, there is a tool that many people use in their work, which is a tick chart. If you don't know what it is, you can take a look at the image in Figure 01.

![Figure 01](https://c.mql5.com/2/47/001__5.png)

Figure 01 - Tick chart

This chart appears in several places in the MetaTrader 5 platform. To give you an idea of these places, I will mention a few places that are included in the standard version of MetaTrader 5. For example, the Market Watch window, as shown in Figure 01. The Depth of Market (Fig. 02), and the order system (Fig. 03).

Apart from these places, you can also use some kind of indicator to see the same information. An example can be found in the article " [Developing of a trading Expert Advisor from scratch (Part 13): Times And Trade (II)](https://www.mql5.com/en/articles/10412)". For all these systems, the service we develop should be able to report or transmit tick information in an appropriate manner, but this is not exactly the information we see in all these pictures. In fact, we see a change in the ASK and BID price values. This is what is actually shown.

![Figure 02](https://c.mql5.com/2/47/003__2.png)

Figure 02 – Tick chart in the Market Depth

![Figure 03](https://c.mql5.com/2/47/004__3.png)

Figure 03 – Tick chart in the order system

It is important to understand this fact. I do not want this information to be missing from our system. The reason is to provide an experience that is as close to the real market as possible. In addition, the information should be correct and should be there even if the system user does not actually utilize it. I don't want you to think that developing something like this is impossible, even though it is not the easiest task. To be honest, the task is much more difficult than it seems, and you will soon understand the reasons for this. Everything will become clearer as we explain. We will see how complex this task is and how many small details it has, and some of them, let's say, are very peculiar.

Here we will begin to implement this system, but in the simplest possible way. First, we will make it appear in the Market Watch window (Fig. 01). After that, we will try to make it appear in other places. Getting it to appear in the Market Watch window will be a challenge. At the same time, it will be interesting, since when we implement and use the simulation of movements with an interval of 1 minute, the tick chart in the "Market Watch" window will display the RANDOM WALK created by the tester. This is all very interesting.

But first things first. Even though the task seems simple to construct, I haven't found any links that could really help me implement it, make the task easier, or take me to the next step. In fact, the only reference I found after searching in various places was the MQL5 documentation, and even that doesn't clarify some of the details. What I will explain in this series is what I actually learned when implementing the system. I apologize to those who may have a different understanding of the system or who have more experience in this matter. Despite all my attempts, the only real way to make the system work was the one that will be shown. Therefore, I am open to advice or suggestions regarding other possible ways, if they are truly functional.

Let's start implementing the craziest thing of all, given the degree of complexity. In the system that will be implemented, in the first moments we will not use simulated data. Therefore, the attachment to this articles contains **REAL** data for 2 days on 4 different assets so that we have at least a basis for experiments. You don't have to trust me, quite the contrary. I want you to collect real market data yourself and test it in the system yourself. This way we can draw our own conclusions about what is actually happening before we implement the simulation system. Because in reality, everything is much crazier than it might seem at first glance.

### Implementing the first version

In this first version, we will disable some resources because I don't want you to believe that the code is completely correct. Actually, there is a defect in it regarding the timer. This can be seen when testing the attached real data. But for now we can ignore it, since at the moment it does not cause any harm to the process itself. It's just that the time it takes to build 1-minute bars is not quite the same as in the real market.

So let's start with a small change in the service file:

```
#property service
#property icon "\\Images\\Market Replay\\Icon.ico"
#property copyright "Daniel Jose"
#property version   "1.17"
#property description "Replay-simulation system for MT5."
#property description "It is independent from the Market Replay."
#property description "For details see the article:"
#property link "https://www.mql5.com/en/articles/11106"
//+------------------------------------------------------------------+
#define def_Dependence  "\\Indicators\\Market Replay.ex5"
#resource def_Dependence
//+------------------------------------------------------------------+
#include <Market Replay\C_Replay.mqh>
//+------------------------------------------------------------------+
input string            user00 = "Mini Indice.txt";     //"Replay" config file
input ENUM_TIMEFRAMES   user01 = PERIOD_M5;             //Initial timeframe for the chart
//input bool            user02 = false;                 //visual bar construction ( Temporarily blocked )
input bool              user03 = true;                  //Visualize creation metrics
//+------------------------------------------------------------------+
void OnStart()
{
        C_Replay        *pReplay;

        pReplay = new C_Replay(user00);
        if ((*pReplay).ViewReplay(user01))
        {
                Print("Permission received. The replay service can now be used...");
                while ((*pReplay).LoopEventOnTime(false, user03));
        }
        delete pReplay;
}
//+------------------------------------------------------------------+
```

This line was blocked because some details needed to be changed in order for the bar construction visualization to work correctly. For this reason, when fast forwarding, the display process will not be visible. To control this, we pass the corresponding argument as true or false.

This is the first thing we need to do. Now we'll have to make a few more small changes. At this point things may start to get a little confusing for those who see this article before reading the others. If this is the case, then I advise you to stop reading for now and start reading from the first article in this series: " [Developing a Replay System - Market simulation (Part 01): First experiments (I)](https://www.mql5.com/en/articles/10543)", because understanding what has been done will help to understand what will happen now and in the future.

With this advice in mind, let's move on. The first thing we will do now is change the function for reading a file containing real ticks. The original program can be seen below.

```
inline bool ReadAllsTicks(void)
                        {
#define def_LIMIT (INT_MAX - 2)
                                string   szInfo;
                                MqlTick  tick;
                                MqlRates rate;
                                int      i0;

                                Print("Loading ticks for replay. Please wait...");
                                ArrayResize(m_Ticks.Info, def_MaxSizeArray, def_MaxSizeArray);
                                i0 = m_Ticks.nTicks;
                                while ((!FileIsEnding(m_File)) && (m_Ticks.nTicks < def_LIMIT) && (!_StopFlag))
                                {
                                        ArrayResize(m_Ticks.Info, m_Ticks.nTicks + 1, def_MaxSizeArray);
                                        szInfo = FileReadString(m_File) + " " + FileReadString(m_File);
                                        tick.time = StringToTime(StringSubstr(szInfo, 0, 19));
                                        tick.time_msc = (int)StringToInteger(StringSubstr(szInfo, 20, 3));
                                        tick.bid = StringToDouble(FileReadString(m_File));
                                        tick.ask = StringToDouble(FileReadString(m_File));
                                        tick.last = StringToDouble(FileReadString(m_File));
                                        tick.volume_real = StringToDouble(FileReadString(m_File));
                                        tick.flags = (uchar)StringToInteger(FileReadString(m_File));
                                        if ((m_Ticks.Info[i0].last == tick.last) && (m_Ticks.Info[i0].time == tick.time) && (m_Ticks.Info[i0].time_msc == tick.time_msc))
                                                m_Ticks.Info[i0].volume_real += tick.volume_real;
                                        else
                                        {
                                                m_Ticks.Info[m_Ticks.nTicks] = tick;
                                                if (tick.volume_real > 0.0)
                                                {
                                                        ArrayResize(m_Ticks.Rate, (m_Ticks.nRate > 0 ? m_Ticks.nRate + 2 : def_BarsDiary), def_BarsDiary);
                                                        m_Ticks.nRate += (BuiderBar1Min(rate, tick) ? 1 : 0);
                                                        m_Ticks.Rate[m_Ticks.nRate] = rate;
                                                        m_Ticks.nTicks++;
                                                }
                                                i0 = (m_Ticks.nTicks > 0 ? m_Ticks.nTicks - 1 : i0);
                                        }
                                }
                                FileClose(m_File);
                                if (m_Ticks.nTicks == def_LIMIT)
                                {
                                        Print("Too much data in the tick file.\nCannot continue...");
                                        return false;
                                }
                                return (!_StopFlag);
#undef def_LIMIT
                        }
```

You will notice that we have removed some parts of the code. The final code is shown below – this is a new function for reading real ticks.

```
inline bool ReadAllsTicks(const bool ToReplay)
                        {
#define def_LIMIT (INT_MAX - 2)
#define def_Ticks m_Ticks.Info[m_Ticks.nTicks]

                                string   szInfo;
                                MqlRates rate;

                                Print("Loading replay ticks. Please wait...");
                                ArrayResize(m_Ticks.Info, def_MaxSizeArray, def_MaxSizeArray);
                                while ((!FileIsEnding(m_File)) && (m_Ticks.nTicks < def_LIMIT) && (!_StopFlag))
                                {
                                        ArrayResize(m_Ticks.Info, m_Ticks.nTicks + 1, def_MaxSizeArray);
                                        szInfo = FileReadString(m_File) + " " + FileReadString(m_File);
                                        def_Ticks.time = StringToTime(StringSubstr(szInfo, 0, 19));
                                        def_Ticks.time_msc = (int)StringToInteger(StringSubstr(szInfo, 20, 3));
                                        def_Ticks.bid = StringToDouble(FileReadString(m_File));
                                        def_Ticks.ask = StringToDouble(FileReadString(m_File));
                                        def_Ticks.last = StringToDouble(FileReadString(m_File));
                                        def_Ticks.volume_real = StringToDouble(FileReadString(m_File));
                                        def_Ticks.flags = (uchar)StringToInteger(FileReadString(m_File));
                                        if (def_Ticks.volume_real > 0.0)
                                        {
                                                ArrayResize(m_Ticks.Rate, (m_Ticks.nRate > 0 ? m_Ticks.nRate + 2 : def_BarsDiary), def_BarsDiary);
                                                m_Ticks.nRate += (BuiderBar1Min(rate, def_Ticks) ? 1 : 0);
                                                m_Ticks.Rate[m_Ticks.nRate] = rate;
                                        }
                                        m_Ticks.nTicks++;
                                }
                                FileClose(m_File);
                                if (m_Ticks.nTicks == def_LIMIT)
                                {
                                        Print("Too much data in the tick file.\nCannot continue...");
                                        return false;
                                }
                                return (!_StopFlag);
#undef def_Ticks
#undef def_LIMIT
                        }
```

Note that it no longer ignores the values contained in the BID and ASK positions. In addition, it no longer accumulates values if the position allows it, that is, it reads the entire data and stores it in memory, because the changes did not generate any new procedures within the function. Rather, they simplified it. I think you (provided that you've read this series of articles) will have no trouble understanding what's really going on, but the fact that these simplifications are made has consequences elsewhere in the code. Some of these items will suffer greatly, which is why we have to disable some components until the whole code is working reliably again.

We could make changes, stabilize the code, and show the final version right away. But I think showing the changes step by step will be of great value to those who are learning and really want to understand how things work in detail. Moreover, there is another reason to explain these changes. But above all, if you act calmly and systematically, then studying difficult issues will become more accessible. What's worse is that many of these subtleties are poorly explained by those who claim to be professional traders, and I mean those who say they actually make a living in the financial markets. But such questions are beyond the scope of this series of articles. Let's not deviate from our main goal: let's continue to implement things little by little, and it will all make more sense later. Especially if we are talking about another market, which is also very interesting. But I don't want to spoil the surprise. If you continue reading the articles, you will understand what I am talking about.

After making these first changes, we will need to make one more slightly strange, but still necessary change. Now that we have the values at which the volume does not exist (the BID and ASK values), we must start the system at the point where we have some specified volume.

```
class C_ConfigService : protected C_FileTicks
{
        protected:
//+------------------------------------------------------------------+
                datetime m_dtPrevLoading;
                int      m_ReplayCount;
//+------------------------------------------------------------------+
inline void FirstBarNULL(void)
                        {
                                MqlRates rate[1];

                                for(int c0 = 0; m_Ticks.Info[c0].volume_real == 0; c0++)
                                        rate[0].close = m_Ticks.Info[c0].last;
                                rate[0].open = rate[0].high = rate[0].low = rate[0].close;
                                rate[0].tick_volume = 0;
                                rate[0].real_volume = 0;
                                rate[0].time = m_Ticks.Info[0].time - 60;
                                CustomRatesUpdate(def_SymbolReplay, rate);
                                m_ReplayCount = 0;
                        }
//+------------------------------------------------------------------+

//... The rest of the class...

}
```

This function was originally private of the class and did not have points in the spotlight. In addition to it, which is now a protected function, we also have a variable. This is the variable that is used in the replay counter. This variable is intended solely to have its value changed by that specific function only. This loop will cause the initial bar on the far left of the chart to have the appropriate value. Remember: we now have the BID and ASK values along with the price values. As for now, the BID and ASK values do not have any meaning for us.

Up to this point, everything was quite simple and clear. We will now move on to the class responsible for the replay. This part contains quite strange things that at first glance do not make much sense. Let's look at this in the next section.

### Modifying the C\_Replay class

The changes here start in a simpler way and get quite strange. Let's start with the simplest change below:

```
                void AdjustPositionToReplay(const bool bViewBuider)
                        {
                                u_Interprocess Info;
                                MqlRates       Rate[def_BarsDiary];
                                int            iPos,   nCount;

                                Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
                                if (m_ReplayCount == 0)
                                        for (; m_Ticks.Info[m_ReplayCount].volume_real == 0; m_ReplayCount++);
                                if (Info.s_Infos.iPosShift == (int)((m_ReplayCount * def_MaxPosSlider * 1.0) / m_Ticks.nTicks)) return;
                                iPos = (int)(m_Ticks.nTicks * ((Info.s_Infos.iPosShift * 1.0) / (def_MaxPosSlider + 1)));
                                Rate[0].time = macroRemoveSec(m_Ticks.Info[iPos].time);
                                if (iPos < m_ReplayCount)
                                {
                                        CustomRatesDelete(def_SymbolReplay, Rate[0].time, LONG_MAX);
                                        if ((m_dtPrevLoading == 0) && (iPos == 0)) FirstBarNULL(); else
                                        {
                                                for(Rate[0].time -= 60; (m_ReplayCount > 0) && (Rate[0].time <= macroRemoveSec(m_Ticks.Info[m_ReplayCount].time)); m_ReplayCount--);
                                                m_ReplayCount++;
                                        }
                                }else if (iPos > m_ReplayCount)
                                {
                                        if (bViewBuider)
                                        {
                                                Info.s_Infos.isWait = true;
                                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                                        }else
                                        {
                                                for(; Rate[0].time > m_Ticks.Info[m_ReplayCount].time; m_ReplayCount++);
                                                for (nCount = 0; m_Ticks.Rate[nCount].time < macroRemoveSec(m_Ticks.Info[iPos].time); nCount++);
                                                CustomRatesUpdate(def_SymbolReplay, m_Ticks.Rate, nCount);
                                        }
                                }
                                for (iPos = (iPos > 0 ? iPos - 1 : 0); (m_ReplayCount < iPos) && (!_StopFlag);) CreateBarInReplay();
                                Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
                                Info.s_Infos.isWait = false;
                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                        }
```

This function has not yet been finalized. Because of this, we had to block the display of the system that constructs 1-minute bars. Even without being fully finished, we had to add to it extra code. This code does something very similar to what happens when we place a bar on the far left of the chart. Most likely, one of the codes will disappear in future versions. But this code does an even more subtle work. When we start replay/simulation, it prevents the asset leap before the first bar is actually drawn. If we disable this line of code, we will see that there is a leap at the beginning of the chart. This leap is due to another fact, which we will see later.

To explain how this was done and whether it is possible to add ticks to the Market Watch window, we need to look at the original bar creation function. It is shown below:

```
inline void CreateBarInReplay(const bool bViewMetrics = false)
                        {
#define def_Rate m_MountBar.Rate[0]

                                static ulong _mdt = 0;
                                int i;

                                if (m_MountBar.bNew = (m_MountBar.memDT != macroRemoveSec(m_Ticks.Info[m_ReplayCount].time)))
                                {
                                        if (bViewMetrics)
                                        {
                                                _mdt = (_mdt > 0 ? GetTickCount64() - _mdt : _mdt);
                                                i = (int) (_mdt / 1000);
                                                Print(TimeToString(m_Ticks.Info[m_ReplayCount].time, TIME_SECONDS), " - Metrica: ", i / 60, ":", i % 60, ".", (_mdt % 1000));
                                                _mdt = GetTickCount64();
                                        }
                                        m_MountBar.memDT = macroRemoveSec(m_Ticks.Info[m_ReplayCount].time);
                                        def_Rate.real_volume = 0;
                                        def_Rate.tick_volume = 0;
                                }
                                def_Rate.close = m_Ticks.Info[m_ReplayCount].last;
                                def_Rate.open = (m_MountBar.bNew ? def_Rate.close : def_Rate.open);
                                def_Rate.high = (m_MountBar.bNew || (def_Rate.close > def_Rate.high) ? def_Rate.close : def_Rate.high);
                                def_Rate.low = (m_MountBar.bNew || (def_Rate.close < def_Rate.low) ? def_Rate.close : def_Rate.low);
                                def_Rate.real_volume += (long) m_Ticks.Info[m_ReplayCount].volume_real;
                                def_Rate.tick_volume += (m_Ticks.Info[m_ReplayCount].volume_real > 0 ? 1 : 0);
                                def_Rate.time = m_MountBar.memDT;
                                m_MountBar.bNew = false;
                                CustomRatesUpdate(def_SymbolReplay, m_MountBar.Rate, 1);
                                m_ReplayCount++;

#undef def_Rate
                        }
```

This original function is only responsible for creating the bars displayed on the chart. I want you to take a look at the above code and compare it with the following:

```
inline void CreateBarInReplay(const bool bViewMetrics = false)
                        {
#define def_Rate m_MountBar.Rate[0]

                                bool bNew;

                                if (m_MountBar.memDT != macroRemoveSec(m_Ticks.Info[m_ReplayCount].time))
                                {
                                        if (bViewMetrics) Metrics();
                                        m_MountBar.memDT = macroRemoveSec(m_Ticks.Info[m_ReplayCount].time);
                                        def_Rate.real_volume = 0;
                                        def_Rate.tick_volume = 0;
                                }
                                bNew = (def_Rate.tick_volume == 0);
                                def_Rate.close = (m_Ticks.Info[m_ReplayCount].volume_real > 0.0 ? m_Ticks.Info[m_ReplayCount].last : def_Rate.close);
                                def_Rate.open = (bNew ? def_Rate.close : def_Rate.open);
                                def_Rate.high = (bNew || (def_Rate.close > def_Rate.high) ? def_Rate.close : def_Rate.high);
                                def_Rate.low = (bNew || (def_Rate.close < def_Rate.low) ? def_Rate.close : def_Rate.low);
                                def_Rate.real_volume += (long) m_Ticks.Info[m_ReplayCount].volume_real;
                                def_Rate.tick_volume += (m_Ticks.Info[m_ReplayCount].volume_real > 0 ? 1 : 0);
                                def_Rate.time = m_MountBar.memDT;
                                CustomRatesUpdate(def_SymbolReplay, m_MountBar.Rate);
                                ViewTick();
                                m_ReplayCount++;

#undef def_Rate
                        }
```

They seem the same, but they are not: there are differences. And it's not that this second code has two new calls. Well, the first call was added only because I decided to remove the metric code from the function. The metric code can be seen below. This is exactly what was in the original function.

```
inline void Metrics(void)
                        {
                                int i;
                                static ulong _mdt = 0;

                                _mdt = (_mdt > 0 ? GetTickCount64() - _mdt : _mdt);
                                i = (int) (_mdt / 1000);
                                Print(TimeToString(m_Ticks.Info[m_ReplayCount].time, TIME_SECONDS), " - Metrica: ", i / 60, ":", i % 60, ".", (_mdt % 1000));
                                _mdt = GetTickCount64();

                        }
```

In fact, the biggest difference is how the system finds the bar closing price. When there was no influence of the BID and ASK values, it was quite easy to know which value to use as the closing price. But since BID and ASK interfere with the data chain, we need another way to do this. By looking at whether a position has any trading volume, we can know whether or not it is a value that can be used as a closing price.

This is the key point in this new function. We have two new calls. We have already looked at the first one. But in the second case, things really get quite strange.

The code for the second call is shown below:

```
inline void ViewTick(void)
                        {
                                MqlTick tick[1];

                                tick[0] = m_Ticks.Info[m_ReplayCount];
                                tick[0].time_msc = (m_Ticks.Info[m_ReplayCount].time * 1000) + m_Ticks.Info[m_ReplayCount].time_msc;
                                CustomTicksAdd(def_SymbolReplay, tick);
                        }
```

This code may look completely strange, but nevertheless it works. The reason for this can be found in the [CustomTicksAdd](https://www.mql5.com/en/docs/customsymbols/customticksadd) function documentation. I'll use exactly what's in the documentation before explaining why the above function works and why it should be that way.

Below is the content of the documentation:

**Further Note**

The CustomTicksAdd function only works for custom symbols opened in the Market Watch window. If the symbol is not selected in Market Watch, then you should add ticks using CustomTicksReplace.

The CustomTicksAdd function allows feeding quotes as if these quotes were received from a broker's server. Data is sent to the Market Watch window instead of being directly written to the tick database. Then, the terminal saves ticks from the Market Watch to the database. If a large volume of data is passed in one call, the function behavior changes, in order to save resources. If more than 256 ticks are transmitted, data is divided into two parts. The first (larger) part is recorded directly to the tick database (similar to [CustomTicksReplace](https://www.mql5.com/en/docs/customsymbols/customticksreplace)). The second part consisting of the last 128 ticks is sent to the Market Watch, from where the terminal saves the ticks to a database.

The [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) structure has two fields with the time value: time (the tick time in seconds) and time\_msc (the tick time in milliseconds), which are counted from January 1, 1970. These fields in the added ticks are processed in the following order:

1. If ticks\[k\].time\_msc!=0, we use it to fill the ticks\[k\].time field, i.e. ticks\[k\].time=ticks\[k\].time\_msc/1000 (integer division) is set for the tick
2. If ticks\[k\].time\_msc==0 and ticks\[k\].time!=0, time in milliseconds is obtained by multiplying by 1000, i.e. ticks\[k\].time\_msc=ticks\[k\].time\*1000
3. If ticks\[k\].time\_msc==0 and ticks\[k\].time==0, the current [trade server time](https://www.mql5.com/en/docs/dateandtime/timetradeserver) up to a millisecond as of the moment of CustomTicksApply call is written to these fields.

If the value of ticks\[k\].bid, ticks\[k\].ask, ticks\[k\].last or ticks\[k\].volume is greater than zero, a combination of appropriate flags is written to the ticks\[k\].flags field:

- TICK\_FLAG\_BID — the tick has changed the Bid price.
- TICK\_FLAG\_ASK — the tick has changed the Ask price.
- TICK\_FLAG\_LAST — the tick has changed the last trade price.
- TICK\_FLAG\_VOLUME — the tick has changed the volume.

If the value of a field is less than or equal to zero, the corresponding flag is not written to the ticks\[k\].flags field.

Flags TICK\_FLAG\_BUY and TICK\_FLAG\_SELL are not added to the history of a custom symbol.

The important thing about this note is that it may not make much sense to many people, but this is exactly what I use to make things work. Here we specify the conditions under which the time in milliseconds differs from zero; the time in milliseconds is zero, and the tick time is different from zero; and when the time in milliseconds and the tick time are zero. The big problem is that when we use real ticks from a file, for the vast majority these conditions are not so clear, and this will become a problem for us. If someone tries to use real ticks obtained from a file to insert this data into the tick information, **they won't get the desired result**.

For this reason, many people may try to do this modeling, but they fail, and this is simply because they do not understand the documentation. But using exactly this fact (which is implied in the documentation), I created the above code. In this code I force the first of the conditions to be created. This is where the time value in milliseconds is different from zero. However, keep in mind that the value indicating the time in milliseconds must also contain the time value, since MetaTrader 5 will perform the calculation to generate the time value. Therefore, we need to adjust the parameters according to the value specified in the milliseconds field.

This way, the CustomTicksAdd function will be able to insert data into the Market Watch. But it’s not only this: when you enter this data into the system, the BID price lines, the ASK price and the last price line will also appear on the chart being built. In other words, as a bonus for being able to insert ticks into the Market Watch, we also received price lines on the chart. We didn't have it due to the lack of this type of functionality. But don't celebrate yet, as the system is not yet complete. There are still some things that need to be checked, fixed and assembled. That's why we use and provide data from REAL TICKS to test this new phase of the replay/simulation system.

### Final considerations

The article is coming to an end because the required steps may lead to some confusion in the material already presented. So in the next article we will look at how to fix some things that are not working properly in the current system. However, you can use the system without fast forwarding or rewinding. If you do this, the tick data in the Market Watch or the price line information may not match the current situation on the replay/simulation chart.

As you see, I prefer just mini-index type contracts. So, I want you to test the system on other assets. This will clarify how the replay/simulation system will behave in relation to what we put into it. I just want to make one thing clear: there are still some flaws in the fast forwarding system. Therefore, I suggest that you, at least for now, avoid using this feature.

In these tests that I offer you, I want you to pay due attention to both the liquidity and volatility of the asset you choose. Check performances of different assets. Note that on assets with fewer trades in the 1-minute interval, the replay/simulation system seems to have difficulties. In a way, it's good to see this now because this part requires a fix. Although the design of bars seems to be correct. We'll make this fix soon. I want you, dear readers, to understand why the replay/simulator service seems strange before we fix this bug. This understanding is important if you really want to get into programming. Don't stop at creating only simple and easy programs. Real programmers are those who solve problems when they arise, not those who give up at the first sign of difficulty.

However, when observing the time in both the Market Watch window and the in value provided by the metrics system, the replay/simulator service is unable to properly synchronize the system when the time is greater than 1 second. We need to fix this and we will do it soon. In the meantime, study this code, as it will really be useful in terms of studying and working with ticks in the Market Watch window. We will continue in the next article. Everything will become even more interesting.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11106](https://www.mql5.com/pt/articles/11106)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11106.zip "Download all attachments in the single ZIP archive")

[Market\_Replay\_lvw\_17.zip](https://www.mql5.com/en/articles/download/11106/market_replay_lvw_17.zip "Download Market_Replay_lvw_17.zip")(9794.75 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/458282)**
(2)


![wrbjym78](https://c.mql5.com/avatar/avatar_na2.png)

**[wrbjym78](https://www.mql5.com/en/users/wrbjym78)**
\|
18 Feb 2024 at 08:58

Can I program my [trading strategies](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies ") to MT5?

![Sky All](https://c.mql5.com/avatar/2021/3/6040B3D1-9E22.png)

**[Sky All](https://www.mql5.com/en/users/sky-love)**
\|
19 Feb 2024 at 01:33

**wrbjym78 [#](https://www.mql5.com/zh/forum/462550#comment_52322022):**

Can I program my trading strategies to MT5?

Hello, I am a moderator on the official website.

Trading strategies can be applied to MT5, you need to put together the requirements and details.

If your theory is interesting and you have hands-on experience, you can ask for help in the forums and the interested gods will help.

If you need a quick paid development, it is recommended that you recruit specialists at Freelance Recruitment. Here is the relevant post.

[https://www.mql5.com/zh/forum/433240](https://www.mql5.com/zh/forum/433240 "https://www.mql5.com/zh/forum/433240")

![Introduction to MQL5 (Part 1): A Beginner's Guide into Algorithmic Trading](https://c.mql5.com/2/61/Beginnerrs_Guide_into_Algorithmic_Trading_LOGO.png)[Introduction to MQL5 (Part 1): A Beginner's Guide into Algorithmic Trading](https://www.mql5.com/en/articles/13738)

Dive into the fascinating realm of algorithmic trading with our beginner-friendly guide to MQL5 programming. Discover the essentials of MQL5, the language powering MetaTrader 5, as we demystify the world of automated trading. From understanding the basics to taking your first steps in coding, this article is your key to unlocking the potential of algorithmic trading even without a programming background. Join us on a journey where simplicity meets sophistication in the exciting universe of MQL5.

![Developing a Replay System — Market simulation (Part 16): New class system](https://c.mql5.com/2/55/replay-p16-avatar.png)[Developing a Replay System — Market simulation (Part 16): New class system](https://www.mql5.com/en/articles/11095)

We need to organize our work better. The code is growing, and if this is not done now, then it will become impossible. Let's divide and conquer. MQL5 allows the use of classes which will assist in implementing this task, but for this we need to have some knowledge about classes. Probably the thing that confuses beginners the most is inheritance. In this article, we will look at how to use these mechanisms in a practical and simple way.

![Neural networks made easy (Part 53): Reward decomposition](https://c.mql5.com/2/57/decomposition_of_remuneration_053_avatar.png)[Neural networks made easy (Part 53): Reward decomposition](https://www.mql5.com/en/articles/13098)

We have already talked more than once about the importance of correctly selecting the reward function, which we use to stimulate the desired behavior of the Agent by adding rewards or penalties for individual actions. But the question remains open about the decryption of our signals by the Agent. In this article, we will talk about reward decomposition in terms of transmitting individual signals to the trained Agent.

![Neural networks made easy (Part 52): Research with optimism and distribution correction](https://c.mql5.com/2/57/optimistic-actor-critic-avatar.png)[Neural networks made easy (Part 52): Research with optimism and distribution correction](https://www.mql5.com/en/articles/13055)

As the model is trained based on the experience reproduction buffer, the current Actor policy moves further and further away from the stored examples, which reduces the efficiency of training the model as a whole. In this article, we will look at the algorithm of improving the efficiency of using samples in reinforcement learning algorithms.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mcddmkhrumjthhguhdivmvfqggccxvya&ssn=1769185247638262206&ssn_dr=0&ssn_sr=0&fv_date=1769185247&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11106&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20%E2%80%94%20Market%20simulation%20(Part%2017)%3A%20Ticks%20and%20more%20ticks%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918524750992242&fz_uniq=5070212256236507614&sv=2552)

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