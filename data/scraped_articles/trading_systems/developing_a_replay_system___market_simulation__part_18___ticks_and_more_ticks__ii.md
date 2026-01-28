---
title: Developing a Replay System — Market simulation (Part 18): Ticks and more ticks (II)
url: https://www.mql5.com/en/articles/11113
categories: Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T19:20:18.486430
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=uiysjgamdksmgquayvarsclfaaigubld&ssn=1769185216977632082&ssn_dr=0&ssn_sr=0&fv_date=1769185216&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11113&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20%E2%80%94%20Market%20simulation%20(Part%2018)%3A%20Ticks%20and%20more%20ticks%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918521687992616&fz_uniq=5070205264029749696&sv=2552)

MetaTrader 5 / Tester


### Introduction

In the previous article, " [Developing a Replay System — Market simulation (Part 17): Ticks and more ticks (I)](https://www.mql5.com/en/articles/11106)", we have added the capability to display a tick chart in the Market Watch system. This was a very positive development, but in that article I mentioned that our system had some shortcomings. Therefore, I decided to disable some functions of the service until the shortcomings are corrected. Now we will fix many of the errors that arose when we started displaying the tick chart.

One of the most noticeable and perhaps the most annoying bugs for me is related to the simulation time required to create 1-minute bars. Those who have been following and testing the replay/simulation service might have noticed that the timing is far from ideal. This becomes even more obvious when the asset has a certain liquidity, which can cause us to lose real trades for a few seconds. From the very beginning I tried to take care of this, and tried to make the experience of using the replay/simulation service similar to the experience of trading a real asset.

Obviously the current metrics are very far from the ideal time for creating a 1-minute bar. That's the first thing we are going to fix. Fixing the synchronization problem is not difficult. This may seem hard, but it's actually quite simple. We did not make the required correction in the previous article since its purpose was to explain how to transfer the tick data that was used to create the 1-minute bars on the chart into the Market Watch window.

If we decided to fix the timer, it would be difficult to understand for those who wish to know how the actual tick data stored in a file is applied in the Market Watch window. So, by focusing solely on how to enables ticks in the Market Watch window, I think it's now clear how to go about this process. One important detail is that I didn't find any other references on how to do this. The only reference was the documentation itself, and while searching I even found people on a community forum who also wanted to know how to do this. However, they did not find an answer that would really help understand what the process should be. So the previous article seemed to end on a bit of an odd note, making it seem like it wasn't clear how to fix the problems mentioned there.

But here we will really deal with this, although not completely yet, since there are questions that are much more difficult to explain. Although often the implementation is relatively simple. To explain some points that are completely different from each other, but are somehow related, and do everything in one article would make it very confusing. Instead of explaining, it may further complicate the entire process of understanding.

My idea for each article is to explain and encourage people to study and deeply explore the MetaTrader 5 platform and the MQL5 language. This goes far beyond what can be seen in the codes distributed somewhere. I really want each of you to be creative and motivated to explore paths that have never been taken before rather than always doing the same thing, as if MQL5 or MetaTrader 5 brings no benefit other than what everyone does with it help. But let's get back to our article.

### Implementing a correction for the 1-minute bar creation time

Let's start with the timer. To fix it, we will change just one small detail in the entire code. This change is shown in the code below:

```
inline bool ReadAllsTicks(const bool ToReplay)
                        {
#define def_LIMIT (INT_MAX - 2)
#define def_Ticks m_Ticks.Info[m_Ticks.nTicks]

                                string   szInfo;
                                MqlRates rate;

                                Print("Loading ticks for replay. Please wait...");
                                ArrayResize(m_Ticks.Info, def_MaxSizeArray, def_MaxSizeArray);
                                while ((!FileIsEnding(m_File)) && (m_Ticks.nTicks < def_LIMIT) && (!_StopFlag))
                                {
                                        ArrayResize(m_Ticks.Info, m_Ticks.nTicks + 1, def_MaxSizeArray);
                                        szInfo = FileReadString(m_File) + " " + FileReadString(m_File);
                                        def_Ticks.time = StringToTime(StringSubstr(szInfo, 0, 19));
                                        def_Ticks.time_msc = (def_Ticks.time * 1000) + (int)StringToInteger(StringSubstr(szInfo, 20, 3));
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

Everything is ready. Now the timer will work more correctly. You might think: but how can this be? I don't understand 🤔. Simply deleting a line (it's been deleted) and replacing it with a little calculation already completely solves the timer problem, but there's more than that. We could also remove the time value, leaving it at zero.

This wouls save us several machine cycles when adding ticks to the Market Watch chart. But (and this "but" really makes me wonder), we would have to perform an additional calculation when creating the 1-minute bars that would then be plotted in MetaTrader 5. As a result, we would have to spend several machine cycles just for the calculation. They way we do it provides much less costs.

With this change, we can immediately implement another one:

```
inline void ViewTick(void)
                        {
                                MqlTick tick[1];

                                tick[0] = m_Ticks.Info[m_ReplayCount];
                                tick[0].time_msc = (m_Ticks.Info[m_ReplayCount].time * 1000) + m_Ticks.Info[m_ReplayCount].time_msc;
                                CustomTicksAdd(def_SymbolReplay, tick);
                        }
```

We no longer need the previously remote calculation, since it is performed at the moment when we load real ticks from the file. If we had done this in the previous article, many would not have understood why ticks appeared on the Market Watch chart. But, as I already said, it seems to me that this has become much clearer now. The simple fact of the smoother changes makes them much more understandable to everyone.

Now comes a question that may be a little disturbing:

Is it possible that a service can freeze with an asset with low liquidity, where trades can occur in several seconds? Could this prevent from closing it due to which it won't actually stop completely? Could this happen because the timer is on standby for a few seconds?

This is a good question. Let's see why this won't happen.

```
                bool LoopEventOnTime(const bool bViewBuider, const bool bViewMetrics)
                        {

                                u_Interprocess Info;
                                int iPos, iTest;

                                iTest = 0;
                                while ((iTest == 0) && (!_StopFlag))
                                {
                                        iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);
                                        iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value) ? iTest : -1);
                                        iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);
                                        if (iTest == 0) Sleep(100);
                                }
                                if ((iTest < 0) || (_StopFlag)) return false;
                                AdjustPositionToReplay(bViewBuider);
                                m_MountBar.delay = 0;
                                while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))
                                {
                                        CreateBarInReplay(bViewMetrics);
                                        iPos = (int)(m_ReplayCount < m_Ticks.nTicks ? m_Ticks.Info[m_ReplayCount].time_msc - m_Ticks.Info[m_ReplayCount - 1].time_msc : 0);
                                        m_MountBar.delay += (iPos < 0 ? iPos + 1000 : iPos);
                                        if (m_MountBar.delay > 400)
                                        {
                                                if (ChartSymbol(m_IdReplay) == "") break;
                                                GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                                                if (!Info.s_Infos.isPlay) return true;
                                                Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
                                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                                                Sleep(m_MountBar.delay - 20);
                                                m_MountBar.delay = 0;
                                        }
                                }
                                return (m_ReplayCount == m_Ticks.nTicks);
                        }
```

The real problem is that when the Sleep function is reached, the service stays "stopped" for a while, but in no case can the service be terminated. In fact, it can be stopped with a STOP request when the call changes the state of the Stop flag. How do I know this? This is what written in the [Sleep](https://www.mql5.com/en/docs/common/sleep) function documentation. Below is an abstract which makes things clear.

**Note**

The Sleep() function cannot be called from custom indicators, since indicators are executed in the interface thread and should not slow it down. The function has a built-in check of the EA stop flag status every 0.1 seconds.

Therefore, we don't need to constantly check whether the service has been stopped or not. The MetaTrader 5 implementation will do this for us. This is very good. This saves us much work related to the creation of a way to maintain functionality and, at the same time, maintain interactivity with the user.

### Implementing fixes in the quick navigation system

Now we will solve the problem with the navigation system to return everything to its original state. There is one small drawback that we could not solve using only MQL5. And since we are not trying to force the use of a DLL at this stage, a small detail will need to be used in the MetaTrader 5 platform. We are doing this to keep the things right. In fact, what needs to be done is quite simple and in some ways even stupid. However, to understand what will be done, you need to pay attention to what I explain. Because while it may seem almost intuitive at first, you may not be able to truly understand it unless you're very attentive.

But anyway, first let's look at how the code is written.

```
#property service
#property icon "\\Images\\Market Replay\\Icon.ico"
#property copyright "Daniel Jose"
#property version   "1.18"
#property description "Replay-simulation system for MT5."
#property description "It is independent from the Market Replay."
#property description "For details see the article:"
#property link "https://www.mql5.com/en/articles/11113"
//+------------------------------------------------------------------+
#define def_Dependence  "\\Indicators\\Market Replay.ex5"
#resource def_Dependence
//+------------------------------------------------------------------+
#include <Market Replay\C_Replay.mqh>
//+------------------------------------------------------------------+
input string            user00 = "Mini Dolar.txt";      //"Replay" config file.
input ENUM_TIMEFRAMES   user01 = PERIOD_M1;             //Initial timeframe for the chart.
input bool              user02 = true;                  //Visual bar construction.
input bool              user03 = true;                  //Visualize creation metrics
//+------------------------------------------------------------------+
void OnStart()
{
        C_Replay        *pReplay;

        pReplay = new C_Replay(user00);
        if ((*pReplay).ViewReplay(user01))
        {
                Print("Permission received. The replay service can now be used...");
                while ((*pReplay).LoopEventOnTime(user02, user03));
        }
        delete pReplay;
}
//+------------------------------------------------------------------+
```

Now we again have a system with the ability to enable or disable the visualization of bar construction. But the decision is up to the user. If you wish, you can turn off this visualization – from a coding perspective it won't make any difference. The reason is that, one way or another, we will still have to do something in MetaTrader 5 if we want to use Market Watch with the tick chart. This is necessary to ensure that the chart has adequate values. But for the regular chart and price lines, no changes or interventions will be required, since they will be configured correctly. (That's what I thought when I did it this way, but you'll see later that I was wrong. There is a bug that I don't really know how to fix, but that will be seen in another article).

To do this, I had to make some changes to the C\_Replay class. The first change concerned the bar creation routine. Please see the code below:

```
inline void CreateBarInReplay(const bool bViewMetrics, const bool bViewTicks)
                        {
#define def_Rate m_MountBar.Rate[0]

                                bool bNew;
                                MqlTick tick[1];

                                if (m_MountBar.memDT != macroRemoveSec(m_Ticks.Info[m_ReplayCount].time))
                                {
                                        if (bViewMetrics) Metrics();
                                        m_MountBar.memDT = (datetime) macroRemoveSec(m_Ticks.Info[m_ReplayCount].time);
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
                                tick = m_Ticks.Info[m_ReplayCount];
                                if (bViewTicks) CustomTicksAdd(def_SymbolReplay, tick);
                                m_ReplayCount++;

#undef def_Rate
                        }
```

This routine was required to add a new argument. This argument will enable or disable the sending of ticks to the Market Watch window. Why do this? This is because it is impossible to update the tick chart in the Market Watch system and the bar chart at the same time. This happens when we configure the service to start at any time. But with normal use, we can send ticks to both the Market Watch window and the bar chart without any problems, which is quite strange.

Then you might be thinking: So we can't really change the contents of the Market Watch window. We can, but not in any case. All we really can and will do is remove old ticks. But this will not be possible without difficulty, at least until the developers of the MetaTrader 5 platform fix the problem associated with the use of custom symbols in the Market Watch window. This is because the ticks placed in the custom symbol do not disappear from the window where we can see such custom ticks. Oddly enough, they remain there, which makes it difficult to understand when we return to any previous position.

In any case, below you can see the function responsible for managing the position system:

```
                void AdjustPositionToReplay(const bool bViewBuider)
                        {
                                u_Interprocess Info;
                                MqlRates       Rate[def_BarsDiary];
                                int            iPos, nCount;

                                Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
                                if (m_ReplayCount == 0)
                                        for (; m_Ticks.Info[m_ReplayCount].volume_real == 0; m_ReplayCount++);
                                if (Info.s_Infos.iPosShift == (int)((m_ReplayCount * def_MaxPosSlider * 1.0) / m_Ticks.nTicks)) return;
                                iPos = (int)(m_Ticks.nTicks * ((Info.s_Infos.iPosShift * 1.0) / (def_MaxPosSlider + 1)));
                                Rate[0].time = macroRemoveSec(m_Ticks.Info[iPos].time);
                                if (iPos < m_ReplayCount)
                                {
                                        CustomRatesDelete(def_SymbolReplay, Rate[0].time, LONG_MAX);
                                        CustomTicksDelete(def_SymbolReplay, m_Ticks.Info[iPos].time_msc, LONG_MAX);
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
                                                for(; Rate[0].time > (m_Ticks.Info[m_ReplayCount].time); m_ReplayCount++);
                                                for (nCount = 0; m_Ticks.Rate[nCount].time < macroRemoveSec(m_Ticks.Info[iPos].time); nCount++);
                                                CustomRatesUpdate(def_SymbolReplay, m_Ticks.Rate, nCount);
                                        }
                                }
                                for (iPos = (iPos > 0 ? iPos - 1 : 0); (m_ReplayCount < iPos) && (!_StopFlag);) CreateBarInReplay(false, false);
                                Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
                                Info.s_Infos.isWait = false;
                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                        }
```

This is the only change here compared to the previous version: simply a function that removes ticks from a specific point on the tick chart of the Market Watch window. Why didn't we see this implementation before? Because I was still trying to achieve dynamic data updating on both charts (with bars and with ticks). But I couldn't do it without causing errors and problems with the update system. At some point I simply decided that only the bar chart would be updated, and therefore this function now has 2 parameters.

Now that the system is almost the same as it was before implementing the tick chart display and the use of the Market Watch window, I will show one more thing before I end this article. But I hope you understand how replay/modeling works when we work with real data. Now we will add ticks to the Market Watch window only when the data has been simulated. And this is exactly the topic of the next section.

### Using simulated data in Market Watch

Since I don't want to extend this tick topic about ticks in the Market Watch window to another article, let's see how to do it, or more accurately, see my suggestion for this type of situation. As for modeling ticks of 1-minute bars, the question here is much simpler than what has been done so far. If you understand everything that came before, then you will have no problem understanding this.

Unlike what happens when using real trading data, when using simulated data, we initially lack some types of information. Not that it is impossible to create that data, but that this must be done carefully. The information I'm talking about is when the price of the last moves beyond the area between BID and ASK. If you look closely at the replay, at the moment when this happens, you will notice that this breakthrough of the area limited by BID and ASK is always very fast and rare. And in fact, in my experience in the market, these things happen when there is a spike in price volatility. However, as I said, these are rare events.

**NOTE:** So don't ever believe that you can and will always operate within the spread. Sometimes the system can go beyond the spread. It is important that you know this because when developing an order system, this information and its proper understanding will be critical.

Important fact: the price does consist of BID and ASK, but this does not mean that there is a gap or collapse in the system. We simply did not receive the trading server update with new BID and ASK values in time. But if you follow the order book, you'll see that things are a little different than many imagine. Therefore, you need to have a lot of experience with the entire trading system to really know about the problems that exist.

Knowing this, you might even consider incorporating such movements into your simulation system. This will make the situation even more realistic. But remember, such things must be treated with caution. Ideally, you should know very well the asset for which this type of movement will be simulated. Only in this way can you bring what is happening closer to what would actually happen in a real market. To understand how to enable this type of movement in the simulator, first let's look at how the simulator should be implemented so that the price always remains within the BID to ASK range.

First, we need to add a new variable.

```
struct st00
        {
                MqlTick  Info[];
                MqlRates Rate[];
                int      nTicks,
                         nRate;
                bool     bTickReal;
        }m_Ticks;
```

We will use it to know whether the ticks are based on real data or simulated ones. This difference is crucial because we will not actually be simulating the BID and ASK movements. We will build these limits based on the value of the last trading price generated by the simulator. But the main reason is that the BID and ASK values do not participate in the reported trading volume. To keep the simulation function simple, we'll make this setup to generate BID and ASK elsewhere.

Once we have a new variable, we will need to initialize it correctly. There are two places where it will be initialized. The first is when we indicate that we are working with simulated ticks.

```
                bool BarsToTicks(const string szFileNameCSV)
                        {
                                C_FileBars      *pFileBars;
                                int             iMem = m_Ticks.nTicks;
                                MqlRates        rate[1];
                                MqlTick         local[];

                                pFileBars = new C_FileBars(szFileNameCSV);
                                ArrayResize(local, def_MaxSizeArray);
                                Print("Converting bars to ticks. Please wait...");
                                while ((*pFileBars).ReadBar(rate) && (!_StopFlag)) Simulation(rate[0], local);
                                ArrayFree(local);
                                delete pFileBars;
                                m_Ticks.bTickReal = false;

                                return ((!_StopFlag) && (iMem != m_Ticks.nTicks));
                        }
```

Another place where we will initialize this variable is when we indicate that we are working with real ticks.

```
                datetime LoadTicks(const string szFileNameCSV, const bool ToReplay = true)
                        {
                                int      MemNRates,
                                         MemNTicks;
                                datetime dtRet = TimeCurrent();
                                MqlRates RatesLocal[];

                                MemNRates = (m_Ticks.nRate < 0 ? 0 : m_Ticks.nRate);
                                MemNTicks = m_Ticks.nTicks;
                                if (!Open(szFileNameCSV)) return 0;
                                if (!ReadAllsTicks(ToReplay)) return 0;
                                if (!ToReplay)
                                {
                                        ArrayResize(RatesLocal, (m_Ticks.nRate - MemNRates));
                                        ArrayCopy(RatesLocal, m_Ticks.Rate, 0, 0);
                                        CustomRatesUpdate(def_SymbolReplay, RatesLocal, (m_Ticks.nRate - MemNRates));
                                        dtRet = m_Ticks.Rate[m_Ticks.nRate].time;
                                        m_Ticks.nRate = (MemNRates == 0 ? -1 : MemNRates);
                                        m_Ticks.nTicks = MemNTicks;
                                        ArrayFree(RatesLocal);
                                }
                                m_Ticks.bTickReal = true;

                                return dtRet;
                        };
```

Now that we know whether we are dealing with real or simulated ticks, we can get to work. But before we move on to the C\_Replay class and start configuring anything, we need to make some small changes to the simulator itself. When we load real ticks, we adjust the time so that the value in the milliseconds field is corrected to represent a specific point in time. But the simulator does not yet make this adjustment. Therefore, if we try to run the system even after changing the C\_Replay class, we will not get a real idea of the simulated data. This is because the time expressed in the milliseconds field is incorrect.

To fix this issue, we will make the following changes:

```
inline void Simulation(const MqlRates &rate, MqlTick &tick[])
                        {
#define macroRandomLimits(A, B) (int)(MathMin(A, B) + (((rand() & 32767) / 32767.0) * MathAbs(B - A)))

                                long     il0, max, i0, i1;
                                bool     b1 = ((rand() & 1) == 1);
                                double   v0, v1;
                                MqlRates rLocal;

                                ArrayResize(m_Ticks.Rate, (m_Ticks.nRate > 0 ? m_Ticks.nRate + 3 : def_BarsDiary), def_BarsDiary);
                                m_Ticks.Rate[++m_Ticks.nRate] = rate;
                                max = rate.tick_volume - 1;
                                v0 = 4.0;
                                v1 = (60000 - v0) / (max + 1.0);
                                for (int c0 = 0; c0 <= max; c0++, v0 += v1)
                                {
                                        tick[c0].last = 0;
                                        tick[c0].flags = 0;
                                        il0 = (long)v0;
                                        tick[c0].time = rate.time + (datetime) (il0 / 1000);
                                        tick[c0].time_msc = (tick[c0].time * 1000) + (il0 % 1000);
                                        tick[c0].time_msc = il0 % 1000;
                                        tick[c0].volume_real = 1.0;
                                }
                                tick[0].last = rate.open;
                                tick[max].last = rate.close;
                                for (int c0 = (int)(rate.real_volume - rate.tick_volume); c0 > 0; c0--)
                                        tick[macroRandomLimits(0, max)].volume_real += 1.0;
                                i0 = (long)(MathMin(max / 3.0, max * 0.2));
                                i1 = max - i0;
                                rLocal = rate;
                                rLocal.open = rate.open;
                                rLocal.close = (b1 ? rate.high : rate.low);
                                i0 = RandomWalk(1, i0, rLocal, tick, 0);
                                rLocal.open = tick[i0].last;
                                rLocal.close = (b1 ? rate.low : rate.high);
                                RandomWalk(i0, i1, rLocal, tick, 1);
                                rLocal.open = tick[i1].last;
                                rLocal.close = rate.close;
                                RandomWalk(i1, max, rLocal, tick, 2);
                                for (int c0 = 0; c0 <= max; c0++)
                                {
                                        ArrayResize(m_Ticks.Info, (m_Ticks.nTicks + 1), def_MaxSizeArray);
                                        m_Ticks.Info[m_Ticks.nTicks++] = tick[c0];
                                }
#undef macroRandomLimits
                        }
```

We remove some code and replace it with the recommended code. This way the time in milliseconds will be compatible with what the C\_Replay class expects. Now we can go into it and make changes to display the simulated content.

In the C\_Replay class, we will focus on making changes to a single function shown in the following code:

```
inline void CreateBarInReplay(const bool bViewMetrics, const bool bViewTicks)
                        {
#define def_Rate m_MountBar.Rate[0]

                                bool bNew;
                                MqlTick tick[1];

                                if (m_MountBar.memDT != macroRemoveSec(m_Ticks.Info[m_ReplayCount].time))
                                {
                                        if (bViewMetrics) Metrics();
                                        m_MountBar.memDT = (datetime) macroRemoveSec(m_Ticks.Info[m_ReplayCount].time);
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
                                if (bViewTicks)
                                {
                                        tick = m_Ticks.Info[m_ReplayCount];
                                        if (!m_Ticks.bTickReal)
                                        {
                                                tick[0].bid = tick[0].last - m_PointsPerTick;
                                                tick[0].ask = tick[0].last + m_PointsPerTick;
                                        }
                                        CustomTicksAdd(def_SymbolReplay, tick);
                                }
                                m_ReplayCount++;

#undef def_Rate
                        }
```

We have a pretty simple change here. This change is intended just for creating and displaying BID and ASK values. But keep in mind that this creation is based on the value of the last deal price and will only be done when we are dealing with simulated values. By running this code, we will get an internal representation of the graph generated by the RANDOM WALK system, just like we did when we used EXCEL to do it. When, in the article " [Developing a Replay System — Market simulation (Part 15): Birth of the SIMULATOR (V) - RANDOM WALK](https://www.mql5.com/en/articles/11071)", I mentioned that there were other ways to do the same visualization, I was referring to this model. But then it was not the time to say how to do it. It's time now.

If these BID and ASK values are not created, we will only be able to display the value based on the last simulated price. This may be enough for you, but some people really like to look at the BID and ASK value. However, using data in this way is not entirely appropriate. The fact that operations are carried out precisely within the BID and ASK, without actually touching them, indicates that direct operations are performed in the market. So, deals are performed without the order book. In this case, the price should not move, although it does, as we will see in the simulator. Therefore, we only need to fix the part that is highlighted in green. This is done so that the movement is at least consistent with what one would actually expect.

Look at the highlighted segment and the changes shown below:

```
                                if (bViewTicks)
                                {
                                        tick = m_Ticks.Info[m_ReplayCount];
                                        if (!m_Ticks.bTickReal)
                                        {
                                                static double BID, ASK;

                                                if (tick[0].last > ASK)
                                                {
                                                        ASK = tick[0].ask = tick[0].last;
                                                        BID = tick[0].bid = tick[0].last - m_PointsPerTick;
                                                }
                                                if (tick[0].last < BID)
                                                {
                                                        ASK = tick[0].ask = tick[0].last + m_PointsPerTick;
                                                        BID = tick[0].bid = tick[0].last;
                                                }
                                                tick[0].ask = tick[0].last + m_PointsPerTick;
                                                tick[0].bid = tick[0].last - m_PointsPerTick;
                                        }
                                        CustomTicksAdd(def_SymbolReplay, tick);
                                }
```

We have removed some code and added a few more points. It should be noted that the BID and ASK values are static, and this is necessary so that we can build a small indicator. Something very simple, but enough to cause a revolution. Since at system startup it is very likely that these values will be zero, then we will first have a call where the emphasis will first be on ASK and we have a rather narrow channel of only 1 tick. Then, until the last deal price leaves this channel, it will remain there.

This is simple yet functional. Pay attention that the BID value should not be allowed to collide with the ASK value (this is in the exchange market, as in the forex market it is a different story, but we will see this later). What causes the collision is the value of the last executed deal. What if we make a small change to the code shown above? Something very subtle. What happens if the BID or ASK value changes without actually changing the last deal price?

Let's change the code as follows:

```
                                if (bViewTicks)
                                {
                                        tick = m_Ticks.Info[m_ReplayCount];
                                        if (!m_Ticks.bTickReal)
                                        {
                                                static double BID, ASK;

                                                if (tick[0].last > ASK)
                                                {
                                                        ASK = tick[0].ask = tick[0].last;
                                                        BID = tick[0].bid = tick[0].last - (m_PointsPerTick * ((rand() & 1) == 1 ? 2 : 1));
                                                }
                                                if (tick[0].last < BID)
                                                {
                                                        ASK = tick[0].ask = tick[0].last + (m_PointsPerTick * ((rand() & 1) == 1 ? 2 : 1));
                                                        BID = tick[0].bid = tick[0].last;
                                                }
                                        }
                                        CustomTicksAdd(def_SymbolReplay, tick);
                                }
```

Look how interesting it is. By adding a certain level of randomness to the system, we have enabled the inclusion of direct orders. That is, of the orders that will be executed without changing the BID or ASK. In the real market, such orders do not occur very often, and not in the form in which the system will display them. But if we ignore this fact, we will already have a good system in which at times there will be a small spread between BID and ASK included. In other words, the simulator practically adapts to a much more common situation in the real market. However, we should beware of an overabundance of direct orders. To avoid this excess, we can make things a little less random.

To do this, we need to implement the last change. It is shown below:

```
                                if (bViewTicks)
                                {
                                        tick = m_Ticks.Info[m_ReplayCount];
                                        if (!m_Ticks.bTickReal)
                                        {
                                                static double BID, ASK;
                                                double  dSpread;
                                                int     iRand = rand();

                                                dSpread = m_PointsPerTick + ((iRand > 29080) && (iRand < 32767) ? ((iRand & 1) == 1 ? m_PointsPerTick : 0 ) : 0 );
                                                if (tick[0].last > ASK)
                                                {
                                                        ASK = tick[0].ask = tick[0].last;
                                                        BID = tick[0].bid = tick[0].last - dSpread;
                                                        BID = tick[0].bid = tick[0].last - (m_PointsPerTick * ((rand() & 1) == 1 ? 2 : 1));
                                                }
                                                if (tick[0].last < BID)
                                                {
                                                        ASK = tick[0].ask = tick[0].last + (m_PointsPerTick * ((rand() & 1) == 1 ? 2 : 1));
                                                        ASK = tick[0].ask = tick[0].last + dSpread;
                                                        BID = tick[0].bid = tick[0].last;
                                                }
                                        }
                                        CustomTicksAdd(def_SymbolReplay, tick);
                                }
```

Here we control the level of complexity of random generation. This is done to keep everything within a certain degree of contingency. We will have direct orders from time to time. But this will be done in more controlled quantities. To do this, we will simply adjust these values here. By adjusting them, we create a small window in which the spread is likely to be slightly larger than the minimum possible value. As a result, from time to time, we will deal with direct orders generated by the modeling system, something that was not possible in previous articles or until now.

### Conclusion

In this article, I showed you how to implement the system for setting and creating ticks on a chart with Market Watch. We started doing this in the previous article. As a result, we created a simulation system capable of simulating even direct orders. This was not part of our initial goals. We are still quite a long way from this being fully usable in some types of trading systems. But what we did today is just the beginning.

In the next article, we will continue our series on creating the market replay/simulation system. The attachment contains 4 different resources for testing and checking the system operation. Remember that I will provide both real tick data and 1-minute bars so you can see the difference between the simulated and real values. This will allow you to start analyzing things more deeply. To understand everything explained here, you will need to run the replay/simulation service in both modes. First check out the custom symbol when running the simulation, and then look at what it with the replay. But pay attention to the tick window, not the chart itself. You will see that the difference is really noticeable. At least regarding the contents of the Market Watch window.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11113](https://www.mql5.com/pt/articles/11113)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11113.zip "Download all attachments in the single ZIP archive")

[Market\_Replay\_rvt\_18.zip](https://www.mql5.com/en/articles/download/11113/market_replay_rvt_18.zip "Download Market_Replay_rvt_18.zip")(12899.62 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/458656)**

![Modified Grid-Hedge EA in MQL5 (Part I): Making a Simple Hedge EA](https://c.mql5.com/2/62/Modified_Grid-Hedge_EA_in_MQL5_4Part_Ip_Making_a_Simple_Hedge_EA__LOGO.png)[Modified Grid-Hedge EA in MQL5 (Part I): Making a Simple Hedge EA](https://www.mql5.com/en/articles/13845)

We will be creating a simple hedge EA as a base for our more advanced Grid-Hedge EA, which will be a mixture of classic grid and classic hedge strategies. By the end of this article, you will know how to create a simple hedge strategy, and you will also get to know what people say about whether this strategy is truly 100% profitable.

![Design Patterns in software development and MQL5 (Part 3): Behavioral Patterns 1](https://c.mql5.com/2/61/Design_Patterns_yPart_39_Behavioral_Patterns_1__LOGO.png)[Design Patterns in software development and MQL5 (Part 3): Behavioral Patterns 1](https://www.mql5.com/en/articles/13796)

A new article from Design Patterns articles and we will take a look at one of its types which is behavioral patterns to understand how we can build communication methods between created objects effectively. By completing these Behavior patterns we will be able to understand how we can create and build a reusable, extendable, tested software.

![Developing a Replay System — Market simulation (Part 19): Necessary adjustments](https://c.mql5.com/2/56/replay_p19_avatar.png)[Developing a Replay System — Market simulation (Part 19): Necessary adjustments](https://www.mql5.com/en/articles/11125)

Here we will prepare the ground so that if we need to add new functions to the code, this will happen smoothly and easily. The current code cannot yet cover or handle some of the things that will be necessary to make meaningful progress. We need everything to be structured in order to enable the implementation of certain things with the minimal effort. If we do everything correctly, we can get a truly universal system that can very easily adapt to any situation that needs to be handled.

![Data label for time series mining (Part 4)：Interpretability Decomposition Using Label Data](https://c.mql5.com/2/61/Data_label_for_time_series_mining_nPart_45Interpretability_Decomposition_Using_Label_Data_LOGO.png)[Data label for time series mining (Part 4)：Interpretability Decomposition Using Label Data](https://www.mql5.com/en/articles/13218)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=geuiywmjxephnrkvgfqxvmtbmswriqul&ssn=1769185216977632082&ssn_dr=0&ssn_sr=0&fv_date=1769185216&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11113&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20%E2%80%94%20Market%20simulation%20(Part%2018)%3A%20Ticks%20and%20more%20ticks%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918521687949511&fz_uniq=5070205264029749696&sv=2552)

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