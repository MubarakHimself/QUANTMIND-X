---
title: Developing a Replay System (Part 69): Getting the Time Right (II)
url: https://www.mql5.com/en/articles/12317
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:36:50.723290
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/12317&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069595438803257231)

MetaTrader 5 / Examples


### Introduction

In the previous article, " [Developing a Replay System (Part 68): Getting the Time Right (I)](https://www.mql5.com/en/articles/12309)", I explained the portion of code related to the mouse indicator. However, that code has little value unless you also examine the code for the replay/simulator service. In any case, if you haven't read the previous article, I recommend doing so before trying to understand this one. This is because one truly complements the other.

The focus here, for now, is on providing information about the remaining time on the bar when the asset is experiencing low liquidity. This may happen due to the absence of traditional generation of OnCalculate events during such periods. Consequently, the mouse indicator will not receive the correct values corresponding to the elapsed seconds. However, based on what was covered in the previous article, we can indeed pass the necessary values so that the indicator can calculate the remaining seconds.

At this stage, we'll primarily focus on the replay/simulator service. More specifically, our attention will be on the file C\_Replay.mqh. So let's begin by reviewing what we need to modify or add to the code.

### Adjusting the File C\_Replay.mqh

There are not many changes that need to be made. However, they give proper context to the code discussed in the previous article, particularly the section involving the use of the iSpread library function within the OnCalculate event. You may have questioned why I used the iSpread function, especially since it would seem more straightforward to read the spread value directly from the array passed to the OnCalculate function.

Indeed, this is quite an interesting point. But to understand the reasoning, we need to grasp how things actually work under the hood. For this, we need to examine and understand how the replay/simulator service code is operating. And of course, we also need to understand how MetaTrader 5 processes this information.

Let's begin with the simplest part: understanding the code in the C\_Replay.mqh file. This file is responsible for generating the information displayed on the chart. The modified code in full can be seen below:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "C_ConfigService.mqh"
005. #include "C_Controls.mqh"
006. //+------------------------------------------------------------------+
007. #define def_IndicatorControl   "Indicators\\Market Replay.ex5"
008. #resource "\\" + def_IndicatorControl
009. //+------------------------------------------------------------------+
010. #define def_CheckLoopService ((!_StopFlag) && (ChartSymbol(m_Infos.IdReplay) != ""))
011. //+------------------------------------------------------------------+
012. #define def_ShortNameIndControl    "Market Replay Control"
013. #define def_MaxSlider             (def_MaxPosSlider + 1)
014. //+------------------------------------------------------------------+
015. class C_Replay : public C_ConfigService
016. {
017.    private   :
018.       struct st00
019.       {
020.          C_Controls::eObjectControl Mode;
021.          uCast_Double               Memory;
022.          ushort                     Position;
023.          int                        Handle;
024.       }m_IndControl;
025.       struct st01
026.       {
027.          long     IdReplay;
028.          int      CountReplay;
029.          double   PointsPerTick;
030.          MqlTick  tick[1];
031.          MqlRates Rate[1];
032.       }m_Infos;
033.       stInfoTicks m_MemoryData;
034. //+------------------------------------------------------------------+
035. inline bool MsgError(string sz0) { Print(sz0); return false; }
036. //+------------------------------------------------------------------+
037. inline void UpdateIndicatorControl(void)
038.          {
039.             double Buff[];
040.
041.             if (m_IndControl.Handle == INVALID_HANDLE) return;
042.             if (m_IndControl.Memory._16b[C_Controls::eCtrlPosition] == m_IndControl.Position)
043.             {
044.                if (CopyBuffer(m_IndControl.Handle, 0, 0, 1, Buff) == 1)
045.                   m_IndControl.Memory.dValue = Buff[0];
046.                if ((m_IndControl.Mode = (C_Controls::eObjectControl)m_IndControl.Memory._16b[C_Controls::eCtrlStatus]) == C_Controls::ePlay)
047.                   m_IndControl.Position = m_IndControl.Memory._16b[C_Controls::eCtrlPosition];
048.             }else
049.             {
050.                m_IndControl.Memory._16b[C_Controls::eCtrlPosition] = m_IndControl.Position;
051.                m_IndControl.Memory._16b[C_Controls::eCtrlStatus] = (ushort)m_IndControl.Mode;
052.                m_IndControl.Memory._8b[7] = 'D';
053.                m_IndControl.Memory._8b[6] = 'M';
054.                EventChartCustom(m_Infos.IdReplay, evCtrlReplayInit, 0, m_IndControl.Memory.dValue, "");
055.             }
056.          }
057. //+------------------------------------------------------------------+
058.       void SweepAndCloseChart(void)
059.          {
060.             long id;
061.
062.             if ((id = ChartFirst()) > 0) do
063.             {
064.                if (ChartSymbol(id) == def_SymbolReplay)
065.                   ChartClose(id);
066.             }while ((id = ChartNext(id)) > 0);
067.          }
068. //+------------------------------------------------------------------+
069. inline void CreateBarInReplay(bool bViewTick)
070.          {
071.             bool    bNew;
072.             double dSpread;
073.             int    iRand = rand();
074.             static int st_Spread = 0;
075.
076.             if (BuildBar1Min(m_Infos.CountReplay, m_Infos.Rate[0], bNew))
077.             {
078.                m_Infos.tick[0] = m_MemoryData.Info[m_Infos.CountReplay];
079.                if (m_MemoryData.ModePlot == PRICE_EXCHANGE)
080.                {
081.                   dSpread = m_Infos.PointsPerTick + ((iRand > 29080) && (iRand < 32767) ? ((iRand & 1) == 1 ? m_Infos.PointsPerTick : 0 ) : 0 );
082.                   if (m_Infos.tick[0].last > m_Infos.tick[0].ask)
083.                   {
084.                      m_Infos.tick[0].ask = m_Infos.tick[0].last;
085.                      m_Infos.tick[0].bid = m_Infos.tick[0].last - dSpread;
086.                   }else if (m_Infos.tick[0].last < m_Infos.tick[0].bid)
087.                   {
088.                      m_Infos.tick[0].ask = m_Infos.tick[0].last + dSpread;
089.                      m_Infos.tick[0].bid = m_Infos.tick[0].last;
090.                   }
091.                }
092.                if (bViewTick)
093.                {
094.                   CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
095.                   if (bNew) EventChartCustom(m_Infos.IdReplay, evSetServerTime, (long)m_Infos.Rate[0].time, 0, "");
096.                }
097.                st_Spread = (int)macroGetTime(m_MemoryData.Info[m_Infos.CountReplay].time);
098.                m_Infos.Rate[0].spread = (int)macroGetSec(m_MemoryData.Info[m_Infos.CountReplay].time);
099.                CustomRatesUpdate(def_SymbolReplay, m_Infos.Rate);
100.             }
101.             m_Infos.Rate[0].spread = (int)(def_MaskTimeService | st_Spread);
102.             CustomRatesUpdate(def_SymbolReplay, m_Infos.Rate);
103.             m_Infos.CountReplay++;
104.          }
105. //+------------------------------------------------------------------+
106.       void AdjustViewDetails(void)
107.          {
108.             MqlRates rate[1];
109.
110.             ChartSetInteger(m_Infos.IdReplay, CHART_SHOW_ASK_LINE, GetInfoTicks().ModePlot == PRICE_FOREX);
111.             ChartSetInteger(m_Infos.IdReplay, CHART_SHOW_BID_LINE, GetInfoTicks().ModePlot == PRICE_FOREX);
112.             ChartSetInteger(m_Infos.IdReplay, CHART_SHOW_LAST_LINE, GetInfoTicks().ModePlot == PRICE_EXCHANGE);
113.             m_Infos.PointsPerTick = SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE);
114.             CopyRates(def_SymbolReplay, PERIOD_M1, 0, 1, rate);
115.             if ((m_Infos.CountReplay == 0) && (GetInfoTicks().ModePlot == PRICE_EXCHANGE))
116.                for (; GetInfoTicks().Info[m_Infos.CountReplay].volume_real == 0; m_Infos.CountReplay++);
117.             if (rate[0].close > 0)
118.             {
119.                if (GetInfoTicks().ModePlot == PRICE_EXCHANGE)
120.                   m_Infos.tick[0].last = rate[0].close;
121.                else
122.                {
123.                   m_Infos.tick[0].bid = rate[0].close;
124.                   m_Infos.tick[0].ask = rate[0].close + (rate[0].spread * m_Infos.PointsPerTick);
125.                }
126.                m_Infos.tick[0].time = rate[0].time;
127.                m_Infos.tick[0].time_msc = rate[0].time * 1000;
128.             }else
129.                m_Infos.tick[0] = GetInfoTicks().Info[m_Infos.CountReplay];
130.             CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
131.          }
132. //+------------------------------------------------------------------+
133.       void AdjustPositionToReplay(void)
134.          {
135.             int nPos, nCount;
136.
137.             if (m_IndControl.Position == (int)((m_Infos.CountReplay * def_MaxSlider) / m_MemoryData.nTicks)) return;
138.             nPos = (int)((m_MemoryData.nTicks * m_IndControl.Position) / def_MaxSlider);
139.             for (nCount = 0; m_MemoryData.Rate[nCount].spread < nPos; m_Infos.CountReplay = m_MemoryData.Rate[nCount++].spread);
140.             if (nCount > 0) CustomRatesUpdate(def_SymbolReplay, m_MemoryData.Rate, nCount - 1);
141.             while ((nPos > m_Infos.CountReplay) && def_CheckLoopService)
142.                CreateBarInReplay(false);
143.          }
144. //+------------------------------------------------------------------+
145.    public   :
146. //+------------------------------------------------------------------+
147.       C_Replay()
148.          :C_ConfigService()
149.          {
150.             Print("************** Market Replay Service **************");
151.             srand(GetTickCount());
152.             SymbolSelect(def_SymbolReplay, false);
153.             CustomSymbolDelete(def_SymbolReplay);
154.             CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay));
155.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0);
156.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, 0);
157.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, 0);
158.             CustomSymbolSetString(def_SymbolReplay, SYMBOL_DESCRIPTION, "Symbol for replay / simulation");
159.             CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, 8);
160.             SymbolSelect(def_SymbolReplay, true);
161.             m_Infos.CountReplay = 0;
162.             m_IndControl.Handle = INVALID_HANDLE;
163.             m_IndControl.Mode = C_Controls::ePause;
164.             m_IndControl.Position = 0;
165.             m_IndControl.Memory._16b[C_Controls::eCtrlPosition] = C_Controls::eTriState;
166.          }
167. //+------------------------------------------------------------------+
168.       ~C_Replay()
169.          {
170.             SweepAndCloseChart();
171.             IndicatorRelease(m_IndControl.Handle);
172.             SymbolSelect(def_SymbolReplay, false);
173.             CustomSymbolDelete(def_SymbolReplay);
174.             Print("Finished replay service...");
175.          }
176. //+------------------------------------------------------------------+
177.       bool OpenChartReplay(const ENUM_TIMEFRAMES arg1, const string szNameTemplate)
178.          {
179.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
180.                return MsgError("Asset configuration is not complete, it remains to declare the size of the ticket.");
181.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
182.                return MsgError("Asset configuration is not complete, need to declare the ticket value.");
183.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
184.                return MsgError("Asset configuration not complete, need to declare the minimum volume.");
185.             SweepAndCloseChart();
186.             m_Infos.IdReplay = ChartOpen(def_SymbolReplay, arg1);
187.             if (!ChartApplyTemplate(m_Infos.IdReplay, szNameTemplate + ".tpl"))
188.                Print("Failed apply template: ", szNameTemplate, ".tpl Using template default.tpl");
189.             else
190.                Print("Apply template: ", szNameTemplate, ".tpl");
191.
192.             return true;
193.          }
194. //+------------------------------------------------------------------+
195.       bool InitBaseControl(const ushort wait = 1000)
196.          {
197.             Print("Waiting for Mouse Indicator...");
198.             Sleep(wait);
199.             while ((def_CheckLoopService) && (ChartIndicatorGet(m_Infos.IdReplay, 0, "Indicator Mouse Study") == INVALID_HANDLE)) Sleep(200);
200.             if (def_CheckLoopService)
201.             {
202.                AdjustViewDetails();
203.                Print("Waiting for Control Indicator...");
204.                if ((m_IndControl.Handle = iCustom(ChartSymbol(m_Infos.IdReplay), ChartPeriod(m_Infos.IdReplay), "::" + def_IndicatorControl, m_Infos.IdReplay)) == INVALID_HANDLE) return false;
205.                ChartIndicatorAdd(m_Infos.IdReplay, 0, m_IndControl.Handle);
206.                UpdateIndicatorControl();
207.             }
208.
209.             return def_CheckLoopService;
210.          }
211. //+------------------------------------------------------------------+
212.       bool LoopEventOnTime(void)
213.          {
214.             int iPos;
215.
216.             while ((def_CheckLoopService) && (m_IndControl.Mode != C_Controls::ePlay))
217.             {
218.                UpdateIndicatorControl();
219.                Sleep(200);
220.             }
221.             m_MemoryData = GetInfoTicks();
222.             AdjustPositionToReplay();
223.             EventChartCustom(m_Infos.IdReplay, evSetServerTime, (long)macroRemoveSec(m_MemoryData.Info[m_Infos.CountReplay].time), 0, "");
224.             iPos = 0;
225.             while ((m_Infos.CountReplay < m_MemoryData.nTicks) && (def_CheckLoopService))
226.             {
227.                if (m_IndControl.Mode == C_Controls::ePause) return true;
228.                iPos += (int)(m_Infos.CountReplay < (m_MemoryData.nTicks - 1) ? m_MemoryData.Info[m_Infos.CountReplay + 1].time_msc - m_MemoryData.Info[m_Infos.CountReplay].time_msc : 0);
229.                CreateBarInReplay(true);
230.                while ((iPos > 200) && (def_CheckLoopService))
231.                {
232.                   Sleep(195);
233.                   iPos -= 200;
234.                   m_IndControl.Position = (ushort)((m_Infos.CountReplay * def_MaxSlider) / m_MemoryData.nTicks);
235.                   UpdateIndicatorControl();
236.                }
237.             }
238.
239.             return ((m_Infos.CountReplay == m_MemoryData.nTicks) && (def_CheckLoopService));
240.          }
241. };
242. //+------------------------------------------------------------------+
243. #undef def_SymbolReplay
244. #undef def_CheckLoopService
245. #undef def_MaxSlider
246. //+------------------------------------------------------------------+
```

Source code of the C\_Replay.mqh file

In the code above, you may notice that several lines have been struck through. These lines should be removed from the version of the code that existed prior to this article. There aren't many lines to delete, but the impact of their removal will be significant.

The first thing to note is that on line 74, a new variable has been introduced. The purpose of this variable is simple: to count seconds when liquidity drops off or becomes very low. Although this logic is not being executed at this exact moment, it's important to understand what's happening in order to grasp how this will be implemented.

First, observe that on line 223, a custom event has been removed from the original code. Also, note that in each iteration of the loop beginning at line 225, there's a call to CreateBarInReplay. This is done on line 229. Now, pay close attention to the following detail: the CreateBarInReplay function is executed roughly every 195 milliseconds, due to line 232 and the time required to execute the loop beginning at line 225. This results in approximately five calls per second, assuming there are no delays between iterations. You should now forget about scenarios with high liquidity. I'm trying to illustrate how the replay/simulator service actually operates when liquidity is very low. So keep this number in mind: there are approximately five calls per second to the CreateBarInReplay function.

Now let's return to the CreateBarInReplay procedure to understand what happens when liquidity is adequate, that is, when we have at least five calls per second.

In this scenario, the condition in line 76 will evaluate as true. Thus the block of code between lines 77 and 100 will be executed. However, note that within this range, some lines have been removed from the code, as indicated by the strikethroughs. Among those is line 95, which used to trigger a custom event for each new one-minute bar. This particular detail will be crucial in explaining why the iSpread function appears in the OnCalculate procedure. But for now, don't worry about that. Let's focus on understanding the basics. Notice that a new piece of code was added on line 97, which initializes the variable value.

Now, pay close attention to this: lines 98 and 99 were struck through. But the logic they contained has not been discarded, it has merely been relocated. Previously, this code was within the block that executed if the condition on line 76 evaluated as true. Now, it will execute unconditionally, as seen in lines 101 and 102. Now pay attention to the following: while line 101 is different, it performs the same task as line 98. The key difference now is the use of a bitmask. This enables the mouse indicator to recognize that the spread value originated from the replay/simulator service. All we're doing here is using an OR operation to correctly configure the mask. However, this introduces a potential issue: if the value of the st\_Spread variable encroaches upon the bitmask region, the mouse indicator will be unable to interpret the incoming values correctly.

So, if anything appears off or goes wrong, simply verify whether the value of the st\_Spread variable is exceeding the bit boundaries reserved for the mask. Under normal conditions, this shouldn't occur since the replay/simulator is designed for intraday studies and analysis. Only if the replay/simulator service is pushed to its absolute time limit would such a condition potentially arise. For reference, this time limit is nearly 12 days, in terms of seconds, which is far more than sufficient for our intended purposes.

Let's continue understanding how the system works. If you compile and run the replay/simulator service alongside the compiled version of the mouse indicator from the previous article, and if the asset has adequate liquidity (i.e., at least one tick per second), you will receive accurate updates regarding the remaining time for the current bar to close and the next to open.

That's all well and good, but it still doesn't explain why the spread array available in one of the OnCalculate function versions wasn't used, and why the iSpread function was necessary to obtain the spread value being reported by the service, as seen on line 101. To understand that, we need to explore a different concept.

### Understanding Why iSpread Is Used

At the time of writing this article, the most recent version of MetaTrader 5 is shown below:

![Image 01](https://c.mql5.com/2/125/001__2.png)

Even in this version - and it's possible that by the time you, dear reader, are reading this, this behavior remains unchanged - MetaTrader 5 still handles bars, at least for custom assets, in a rather odd way. Maybe not all information related to bars is affected, but since we are transmitting data through the spread, it's clear that this behaves somewhat peculiarly.

To demonstrate this, let's make a few small modifications to the code in the C\_Replay.mqh header file and in the mouse indicator. I believe this will make it much easier to clearly demonstrate what's actually happening, as merely explaining it wouldn't be enough. So, in the file C\_Replay.mqh, we modify the code in the following fragment shown below:

```
068. //+------------------------------------------------------------------+
069. inline void CreateBarInReplay(bool bViewTick)
070.          {
071.             bool    bNew;
072.             double dSpread;
073.             int    iRand = rand();
074.             static int st_Spread = 0;
075.
076.             if (BuildBar1Min(m_Infos.CountReplay, m_Infos.Rate[0], bNew))
077.             {
078.                m_Infos.tick[0] = m_MemoryData.Info[m_Infos.CountReplay];
079.                if (m_MemoryData.ModePlot == PRICE_EXCHANGE)
080.                {
081.                   dSpread = m_Infos.PointsPerTick + ((iRand > 29080) && (iRand < 32767) ? ((iRand & 1) == 1 ? m_Infos.PointsPerTick : 0 ) : 0 );
082.                   if (m_Infos.tick[0].last > m_Infos.tick[0].ask)
083.                   {
084.                      m_Infos.tick[0].ask = m_Infos.tick[0].last;
085.                      m_Infos.tick[0].bid = m_Infos.tick[0].last - dSpread;
086.                   }else if (m_Infos.tick[0].last < m_Infos.tick[0].bid)
087.                   {
088.                      m_Infos.tick[0].ask = m_Infos.tick[0].last + dSpread;
089.                      m_Infos.tick[0].bid = m_Infos.tick[0].last;
090.                   }
091.                }
092.                if (bViewTick)
093.                   CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
094.                st_Spread = (int)macroGetTime(m_MemoryData.Info[m_Infos.CountReplay].time);
095.             }
096.             Print(TimeToString(st_Spread, TIME_SECONDS));
097.             m_Infos.Rate[0].spread = (int)(def_MaskTimeService | st_Spread);
098.             CustomRatesUpdate(def_SymbolReplay, m_Infos.Rate);
099.             m_Infos.CountReplay++;
100.          }
101. //+------------------------------------------------------------------+
```

Code from the C\_Replay.mqh file

Note that the code in this fragment has already been cleaned up, so the line numbering may differ slightly. However, the code itself is identical to what was shown earlier in this article. The only difference is line 96, which was added to display in the terminal the value currently being written into the bar's spread field. As a result of running this modified code, you will see the output shown below:

![Animation 01](https://c.mql5.com/2/125/Animar7o_01__1.gif)

Notice that the value being printed is exactly the same as the one shown on the tick chart as the current time. It's very important to understand this. We now have confirmation that the value being inserted into the bar's spread field is, in fact, the time value displayed on the chart. Now let's move on to something else. We'll make a slight modification to the control indicator (something very subtle) just to analyze how the system behaves. This modification will be made to the code in the header file C\_Study.mqh. You can see the change below:

```
109. //+------------------------------------------------------------------+
110.       void Update(const eStatusMarket arg)
111.          {
112.             int i0;
113.             datetime dt;
114.
115.             switch (m_Info.Status = (m_Info.Status != arg ? arg : m_Info.Status))
116.             {
117.                case eCloseMarket :
118.                   m_Info.szInfo = "Closed Market";
119.                   break;
120.                case eInReplay    :
121.                case eInTrading   :
122.                   i0 = PeriodSeconds();
123.                   dt = (m_Info.Status == eInReplay ? (datetime) GL_TimeAdjust : TimeCurrent());
124.                   m_Info.Rate.time = (m_Info.Rate.time <= dt ? (datetime)(((ulong) dt / i0) * i0) + i0 : m_Info.Rate.time);
125.                   if (dt > 0) m_Info.szInfo = TimeToString((datetime)m_Info.Rate.time/* - dt*/, TIME_SECONDS);
126.                   break;
127.                case eAuction     :
128.                   m_Info.szInfo = "Auction";
129.                   break;
130.                default           :
131.                   m_Info.szInfo = "ERROR";
132.             }
133.             Draw();
134.          }
135. //+------------------------------------------------------------------+
```

Part of the C\_Study.mqh file

Pay close attention here, as the change is quite subtle. On line 125, the dt setting was removed. This means that the information now being displayed is the exact time when a new bar is expected to appear. Take note: it does not represent how much time remains until the next bar, but rather when the next bar is actually expected. With this change made, we recompile the mouse indicator in order to test the output that will be shown. In the animation below, you can observe what actually happens:

![Anime 2](https://c.mql5.com/2/125/Animac7o_02__1.gif)

Note that the chart timeframe used is two minutes. The calculation being performed now indicates the exact moment the next bar will appear. This is what is shown in the mouse indicator. You can see that when the chart time reaches the specified point, the indicator immediately begins reporting when the new bar will emerge. In other words, the system is working as intended. However, these tests do not yet verify the value being provided by the replay/simulation service. What we've done so far is merely confirm the information we expected to be present. Now, we need to verify the actual value that the service is passing along. It's important to ensure that the chart timeframe is not set to one minute, otherwise, the test will be invalid. So let's keep it at two minutes, which is enough for analyzing what's going on.

In order for the test to perform as expected, we need to make a small modification. Once again, pay close attention to the code in the following fragment:

```
109. //+------------------------------------------------------------------+
110.       void Update(const eStatusMarket arg)
111.          {
112.             int i0;
113.             datetime dt;
114.
115.             switch (m_Info.Status = (m_Info.Status != arg ? arg : m_Info.Status))
116.             {
117.                case eCloseMarket :
118.                   m_Info.szInfo = "Closed Market";
119.                   break;
120.                case eInReplay    :
121.                case eInTrading   :
122.                   i0 = PeriodSeconds();
123.                   dt = (m_Info.Status == eInReplay ? (datetime) GL_TimeAdjust : TimeCurrent());
124.                   m_Info.Rate.time = (m_Info.Rate.time <= dt ? (datetime)(((ulong) dt / i0) * i0) + i0 : m_Info.Rate.time);
125.                   if (dt > 0) m_Info.szInfo = TimeToString((datetime)/*m_Info.Rate.time -*/ dt, TIME_SECONDS);
126.                   break;
127.                case eAuction     :
128.                   m_Info.szInfo = "Auction";
129.                   break;
130.                default           :
131.                   m_Info.szInfo = "ERROR";
132.             }
133.             Draw();
134.          }
135. //+------------------------------------------------------------------+
```

Part of the C\_Study.mqh file

Now we select the value provided by the service and display it in the mouse pointer. The result can be seen below:

![Animation 3](https://c.mql5.com/2/125/Animaqvo_03__1.gif)

As you can see, it matches exactly what we expected. At this point, we're not going to make any further changes to the header file. Instead, we'll focus on something else in the mouse indicator. Let's see what happens if we use the spread value obtained during the OnCalculate call. To do this, we need to modify the mouse indicator's code. But keep the following in mind: the value shown in the indicator will be whatever is captured and assigned to the GL\_TimeAdjust variable. Remembering this is crucial. So now, let's modify the indicator code to test whether using the spread value obtained from OnCalculate is actually suitable. The updated code looks like this:

```
46. //+------------------------------------------------------------------+
47. int OnCalculate(const int rates_total, const int prev_calculated, const datetime& time[], const double& open[],
48.                 const double& high[], const double& low[], const double& close[], const long& tick_volume[],
49.                 const long& volume[], const int& spread[])
50. //int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double& price[])
51. {
52.    GL_PriceClose = close[rates_total - 1];
53. //   GL_PriceClose = price[rates_total - 1];
54.    GL_TimeAdjust = (spread[rates_total - 1] & (~def_MaskTimeService);
55. //   if (_Symbol == def_SymbolReplay)
56. //      GL_TimeAdjust = iSpread(NULL, PERIOD_M1, 0) & (~def_MaskTimeService);
57.    m_posBuff = rates_total;
58.    (*Study).Update(m_Status);
59.
60.    return rates_total;
61. }
62. //+------------------------------------------------------------------+
```

Mouse pointer file fragment

Carefully observe what we're doing in the fragment above. We are isolating the code such that the crossed-out lines represent the current version, as seen in the previous article. These lines should be temporarily removed. In their place, we’ve added new lines intended to use the data provided by MetaTrader 5 through the OnCalculate function. In other words, we are no longer relying on the iSpread function call. Instead, we're using the value provided by MetaTrader 5 in the spread array. To ensure compatibility with the service, we need to make a small adjustment, as shown on line 54. You'll notice it performs the same operation that was previously done using iSpread, except now the value used comes directly from the arguments passed to OnCalculate. This can make a significant difference for us, as it removes the need for an extra function call just to retrieve a value that MetaTrader 5 is already providing.

Now, let's take a look at the outcome of running this updated code. It is shown in the animation below:

![Animation 4](https://c.mql5.com/2/125/Animatdo_04__1.gif)

Oops. What just happened here? Why did the value in the mouse indicator freeze? The answer to this isn't simple. Contrary to what many might assume, I don't actually know the answer. Wait, how can that be that I don't know? It's true that I have some suspicions. But rather than speculate, I prefer to simply show you that something you probably didn't expect to happen can happen. This way, you can observe it for yourself and draw your own conclusions.

In any case, the service continues to output the values as before, just like we saw in the previous animations. The indicator still captures the spread value. But why did it freeze? I can't explain. All I know is that when a new bar appears on the chart, and this is why it's important to use a timeframe other than one minute to expose this issue, the value in the spread array will then be updated correctly.

Again, let me remind you: what I'm showing here may not be happening by the time you read this article. That's because it's quite possible MetaTrader 5 will have received an update that corrects this issue. Until that happens, I'm working around the problem by using the iSpread function. Once this minor issue is fixed, we will stop using iSpread and rely instead on the value passed directly to OnCalculate by MetaTrader 5. So don't get too attached to any particular part of the code - everything will be improved as development progresses. With that, I believe you now understand why I use iSpread rather than the spread value passed as an argument to OnCalculate. But we're not done yet. We still need to devise a way for the service to inform us of the remaining time on a bar when low liquidity prevents us from receiving ticks - or more accurately, OnCalculate events - every second. To proceed, we'll now revert the changes made in this section (used to demonstrate the reason for using iSpread in the indicator) and return to working on the service.

### Fixing a Flaw in the Service

Unfortunately, for everything to work properly when the time between ticks exceeds one second, we'll need to take a slightly different approach than originally planned at the start of this article. The issue lies in the fact that I had intended to place the counter within the bar creation routine. However, I overlooked an important detail: TIME. Go back to the beginning of this article and look at the source code for the header file C\_Replay.mqh. In line 230, we have a loop that causes the replay/simulation service to wait until the appropriate time has passed before a new tick should appear. And this is where the problem arises.

During development and testing, I was working with assets that had high liquidity, i.e., historical data where the time between ticks was generally less than one second. Once I began implementing changes to support the possibility of longer tick intervals, a flaw emerged. Not because it suddenly appeared. But because it had been there all along, hidden by the relatively short tick intervals. Now pay close attention to the loop between lines 230 and 236. What's wrong with it? The problem is that it doesn't account for the possibility that the user might pause the system. How does this happen? If the service is in a loop, waiting for the next tick, surely that's fine, right? Not exactly. When the wait time exceeds one second, we run into trouble.

Let's assume that we are replaying Forex data. At the beginning of a daily session, tick intervals can be quite large. If you hit play and the service detects that it needs to wait 40 seconds before the next tick, then even if you press pause, move the control slider to a different point and then press play again, the service won't respond. Because it's stuck in the loop from lines 230 to 236, waiting out the full 40 seconds. So the first thing we need to fix is that. But rather than patching this in isolation, let's go ahead and implement both the fix and the solution for showing remaining bar time during periods of low liquidity, all at once. The updated version of the entire C\_Replay.mqh file is shown below:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "C_ConfigService.mqh"
005. #include "C_Controls.mqh"
006. //+------------------------------------------------------------------+
007. #define def_IndicatorControl   "Indicators\\Market Replay.ex5"
008. #resource "\\" + def_IndicatorControl
009. //+------------------------------------------------------------------+
010. #define def_CheckLoopService ((!_StopFlag) && (ChartSymbol(m_Infos.IdReplay) != ""))
011. //+------------------------------------------------------------------+
012. #define def_ShortNameIndControl   "Market Replay Control"
013. #define def_MaxSlider             (def_MaxPosSlider + 1)
014. //+------------------------------------------------------------------+
015. class C_Replay : public C_ConfigService
016. {
017.    private   :
018.       struct st00
019.       {
020.          C_Controls::eObjectControl Mode;
021.          uCast_Double               Memory;
022.          ushort                     Position;
023.          int                        Handle;
024.       }m_IndControl;
025.       struct st01
026.       {
027.          long     IdReplay;
028.          int      CountReplay;
029.          double   PointsPerTick;
030.          MqlTick  tick[1];
031.          MqlRates Rate[1];
032.       }m_Infos;
033.       stInfoTicks m_MemoryData;
034. //+------------------------------------------------------------------+
035. inline bool MsgError(string sz0) { Print(sz0); return false; }
036. //+------------------------------------------------------------------+
037. inline void UpdateIndicatorControl(void)
038.          {
039.             double Buff[];
040.
041.             if (m_IndControl.Handle == INVALID_HANDLE) return;
042.             if (m_IndControl.Memory._16b[C_Controls::eCtrlPosition] == m_IndControl.Position)
043.             {
044.                if (CopyBuffer(m_IndControl.Handle, 0, 0, 1, Buff) == 1)
045.                   m_IndControl.Memory.dValue = Buff[0];
046.                if ((m_IndControl.Mode = (C_Controls::eObjectControl)m_IndControl.Memory._16b[C_Controls::eCtrlStatus]) == C_Controls::ePlay)
047.                   m_IndControl.Position = m_IndControl.Memory._16b[C_Controls::eCtrlPosition];
048.             }else
049.             {
050.                m_IndControl.Memory._16b[C_Controls::eCtrlPosition] = m_IndControl.Position;
051.                m_IndControl.Memory._16b[C_Controls::eCtrlStatus] = (ushort)m_IndControl.Mode;
052.                m_IndControl.Memory._8b[7] = 'D';
053.                m_IndControl.Memory._8b[6] = 'M';
054.                EventChartCustom(m_Infos.IdReplay, evCtrlReplayInit, 0, m_IndControl.Memory.dValue, "");
055.             }
056.          }
057. //+------------------------------------------------------------------+
058.       void SweepAndCloseChart(void)
059.          {
060.             long id;
061.
062.             if ((id = ChartFirst()) > 0) do
063.             {
064.                if (ChartSymbol(id) == def_SymbolReplay)
065.                   ChartClose(id);
066.             }while ((id = ChartNext(id)) > 0);
067.          }
068. //+------------------------------------------------------------------+
069. inline int RateUpdate(bool bCheck)
070.          {
071.             static int st_Spread = 0;
072.
073.             st_Spread = (bCheck ? (int)macroGetTime(m_MemoryData.Info[m_Infos.CountReplay].time) : st_Spread + 1);
074.             m_Infos.Rate[0].spread = (int)(def_MaskTimeService | st_Spread);
075.             CustomRatesUpdate(def_SymbolReplay, m_Infos.Rate);
076.
077.             return 0;
078.          }
079. //+------------------------------------------------------------------+
080. inline void CreateBarInReplay(bool bViewTick)
081.          {
082.             bool    bNew;
083.             double dSpread;
084.             int    iRand = rand();
085.             static int st_Spread = 0;
086.
087.             if (BuildBar1Min(m_Infos.CountReplay, m_Infos.Rate[0], bNew))
088.             {
089.                m_Infos.tick[0] = m_MemoryData.Info[m_Infos.CountReplay];
090.                if (m_MemoryData.ModePlot == PRICE_EXCHANGE)
091.                {
092.                   dSpread = m_Infos.PointsPerTick + ((iRand > 29080) && (iRand < 32767) ? ((iRand & 1) == 1 ? m_Infos.PointsPerTick : 0 ) : 0 );
093.                   if (m_Infos.tick[0].last > m_Infos.tick[0].ask)
094.                   {
095.                      m_Infos.tick[0].ask = m_Infos.tick[0].last;
096.                      m_Infos.tick[0].bid = m_Infos.tick[0].last - dSpread;
097.                   }else if (m_Infos.tick[0].last < m_Infos.tick[0].bid)
098.                   {
099.                      m_Infos.tick[0].ask = m_Infos.tick[0].last + dSpread;
100.                      m_Infos.tick[0].bid = m_Infos.tick[0].last;
101.                   }
102.                }
103.                if (bViewTick)
104.                   CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
105.                RateUpdate(true);
106.                st_Spread = (int)macroGetTime(m_MemoryData.Info[m_Infos.CountReplay].time);
107.             }
108.             m_Infos.Rate[0].spread = (int)(def_MaskTimeService | st_Spread);
109.             CustomRatesUpdate(def_SymbolReplay, m_Infos.Rate);
110.             m_Infos.CountReplay++;
111.          }
112. //+------------------------------------------------------------------+
113.       void AdjustViewDetails(void)
114.          {
115.             MqlRates rate[1];
116.
117.             ChartSetInteger(m_Infos.IdReplay, CHART_SHOW_ASK_LINE, GetInfoTicks().ModePlot == PRICE_FOREX);
118.             ChartSetInteger(m_Infos.IdReplay, CHART_SHOW_BID_LINE, GetInfoTicks().ModePlot == PRICE_FOREX);
119.             ChartSetInteger(m_Infos.IdReplay, CHART_SHOW_LAST_LINE, GetInfoTicks().ModePlot == PRICE_EXCHANGE);
120.             m_Infos.PointsPerTick = SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE);
121.             CopyRates(def_SymbolReplay, PERIOD_M1, 0, 1, rate);
122.             if ((m_Infos.CountReplay == 0) && (GetInfoTicks().ModePlot == PRICE_EXCHANGE))
123.                for (; GetInfoTicks().Info[m_Infos.CountReplay].volume_real == 0; m_Infos.CountReplay++);
124.             if (rate[0].close > 0)
125.             {
126.                if (GetInfoTicks().ModePlot == PRICE_EXCHANGE)
127.                   m_Infos.tick[0].last = rate[0].close;
128.                else
129.                {
130.                   m_Infos.tick[0].bid = rate[0].close;
131.                   m_Infos.tick[0].ask = rate[0].close + (rate[0].spread * m_Infos.PointsPerTick);
132.                }
133.                m_Infos.tick[0].time = rate[0].time;
134.                m_Infos.tick[0].time_msc = rate[0].time * 1000;
135.             }else
136.                m_Infos.tick[0] = GetInfoTicks().Info[m_Infos.CountReplay];
137.             CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
138.          }
139. //+------------------------------------------------------------------+
140.       void AdjustPositionToReplay(void)
141.          {
142.             int nPos, nCount;
143.
144.             if (m_IndControl.Position == (int)((m_Infos.CountReplay * def_MaxSlider) / m_MemoryData.nTicks)) return;
145.             nPos = (int)((m_MemoryData.nTicks * m_IndControl.Position) / def_MaxSlider);
146.             for (nCount = 0; m_MemoryData.Rate[nCount].spread < nPos; m_Infos.CountReplay = m_MemoryData.Rate[nCount++].spread);
147.             if (nCount > 0) CustomRatesUpdate(def_SymbolReplay, m_MemoryData.Rate, nCount - 1);
148.             while ((nPos > m_Infos.CountReplay) && def_CheckLoopService)
149.                CreateBarInReplay(false);
150.          }
151. //+------------------------------------------------------------------+
152.    public   :
153. //+------------------------------------------------------------------+
154.       C_Replay()
155.          :C_ConfigService()
156.          {
157.             Print("************** Market Replay Service **************");
158.             srand(GetTickCount());
159.             SymbolSelect(def_SymbolReplay, false);
160.             CustomSymbolDelete(def_SymbolReplay);
161.             CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay));
162.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0);
163.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, 0);
164.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, 0);
165.             CustomSymbolSetString(def_SymbolReplay, SYMBOL_DESCRIPTION, "Symbol for replay / simulation");
166.             CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, 8);
167.             SymbolSelect(def_SymbolReplay, true);
168.             m_Infos.CountReplay = 0;
169.             m_IndControl.Handle = INVALID_HANDLE;
170.             m_IndControl.Mode = C_Controls::ePause;
171.             m_IndControl.Position = 0;
172.             m_IndControl.Memory._16b[C_Controls::eCtrlPosition] = C_Controls::eTriState;
173.          }
174. //+------------------------------------------------------------------+
175.       ~C_Replay()
176.          {
177.             SweepAndCloseChart();
178.             IndicatorRelease(m_IndControl.Handle);
179.             SymbolSelect(def_SymbolReplay, false);
180.             CustomSymbolDelete(def_SymbolReplay);
181.             Print("Finished replay service...");
182.          }
183. //+------------------------------------------------------------------+
184.       bool OpenChartReplay(const ENUM_TIMEFRAMES arg1, const string szNameTemplate)
185.          {
186.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
187.                return MsgError("Asset configuration is not complete, it remains to declare the size of the ticket.");
188.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
189.                return MsgError("Asset configuration is not complete, need to declare the ticket value.");
190.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
191.                return MsgError("Asset configuration not complete, need to declare the minimum volume.");
192.             SweepAndCloseChart();
193.             m_Infos.IdReplay = ChartOpen(def_SymbolReplay, arg1);
194.             if (!ChartApplyTemplate(m_Infos.IdReplay, szNameTemplate + ".tpl"))
195.                Print("Failed apply template: ", szNameTemplate, ".tpl Using template default.tpl");
196.             else
197.                Print("Apply template: ", szNameTemplate, ".tpl");
198.
199.             return true;
200.          }
201. //+------------------------------------------------------------------+
202.       bool InitBaseControl(const ushort wait = 1000)
203.          {
204.             Print("Waiting for Mouse Indicator...");
205.             Sleep(wait);
206.             while ((def_CheckLoopService) && (ChartIndicatorGet(m_Infos.IdReplay, 0, "Indicator Mouse Study") == INVALID_HANDLE)) Sleep(200);
207.             if (def_CheckLoopService)
208.             {
209.                AdjustViewDetails();
210.                Print("Waiting for Control Indicator...");
211.                if ((m_IndControl.Handle = iCustom(ChartSymbol(m_Infos.IdReplay), ChartPeriod(m_Infos.IdReplay), "::" + def_IndicatorControl, m_Infos.IdReplay)) == INVALID_HANDLE) return false;
212.                ChartIndicatorAdd(m_Infos.IdReplay, 0, m_IndControl.Handle);
213.                UpdateIndicatorControl();
214.             }
215.
216.             return def_CheckLoopService;
217.          }
218. //+------------------------------------------------------------------+
219.       bool LoopEventOnTime(void)
220.          {
221.             int iPos, iCycles;
222.
223.             while ((def_CheckLoopService) && (m_IndControl.Mode != C_Controls::ePlay))
224.             {
225.                UpdateIndicatorControl();
226.                Sleep(200);
227.             }
228.             m_MemoryData = GetInfoTicks();
229.             AdjustPositionToReplay();
230.             iPos = iCycles = 0;
231.             while ((m_Infos.CountReplay < m_MemoryData.nTicks) && (def_CheckLoopService))
232.             {
233.                if (m_IndControl.Mode == C_Controls::ePause) return true;
234.                iPos += (int)(m_Infos.CountReplay < (m_MemoryData.nTicks - 1) ? m_MemoryData.Info[m_Infos.CountReplay + 1].time_msc - m_MemoryData.Info[m_Infos.CountReplay].time_msc : 0);
235.                CreateBarInReplay(true);
236.                while ((iPos > 200) && (def_CheckLoopService) && (m_IndControl.Mode != C_Controls::ePause))
237.                {
238.                   Sleep(195);
239.                   iPos -= 200;
240.                   m_IndControl.Position = (ushort)((m_Infos.CountReplay * def_MaxSlider) / m_MemoryData.nTicks);
241.                   UpdateIndicatorControl();
242.                   iCycles = (iCycles == 4 ? RateUpdate(false) : iCycles + 1);
243.                }
244.             }
245.
246.             return ((m_Infos.CountReplay == m_MemoryData.nTicks) && (def_CheckLoopService));
247.          }
248. };
249. //+------------------------------------------------------------------+
250. #undef def_SymbolReplay
251. #undef def_CheckLoopService
252. #undef def_MaxSlider
253. //+------------------------------------------------------------------+
```

Source code of the C\_Replay.mqh file

Let's now walk through an explanation of how this updated code works. We'll begin at the end, where the service was previously behaving oddly under low-liquidity conditions. Note that on line 236, a correction was made. The code no longer gets stuck waiting for a long detected delay, which would otherwise make the system unresponsive to the user. All we had to do in this case was add a check to determine whether the user had paused the system. If so, the loop is exited, and when execution reaches line 233, the function terminates, returning to the main control flow. The main logic then makes a new call to the function and waits again. This time, however, it loops on line 223, which allows the user to reposition the control indicator and move to a different point in time. This provides us with a much smoother experience, especially when an asset has low liquidity or enters an auction phase. You might not fully understand what I mean just by looking at this LoopEventOnTime routine. But things will become clearer as the explanation unfolds.

Let's explore the changes made to provide time-remaining feedback even when tick activity is sparse. On line 221, a new variable was added and initialized on line 230. Now pay attention to line 242. We use that same variable to count from 0 to 4. When the value reaches 4, we call the RateUpdate function. But what is RateUpdate? Don't worry, we'll get to that. For now, note that the function is called with the argument false, and its return value is assigned to the variable. This detail is important. Remember earlier in the article, I mentioned we'd have roughly five cycles per second? That's why we have this counter. The idea is to provide the mouse indicator with a sense that a second has passed. But keep in mind: this is only an approximation. We're not timing things with perfect precision. The goal isn't strict accuracy but rather giving the user a general sense of how much time is left before the bar closes.

Now let's go to another part of the code, namely, to the procedure that starts in line 80. Here, lines that have been struck out are replaced with a call to RateUpdate. This time, however, the argument passed is true. If we're adding a new tick, the argument should be true. If we're just updating the time (without receiving a tick), the argument should be false. Interesting, right? Let's now take a look at the RateUpdate procedure itself, which begins on line 69.

The RateUpdate function had to be created because updating time directly carried the risk of accidentally skipping some ticks. This is related to line 110. To avoid this, we moved the time-update logic to its own function. You'll notice the variable previously declared on line 85 has been moved to line 71. Likewise, the work previously done in lines 108 and 109 is now handled on lines 74 and 75. Essentially, this function is almost a copy of what we had before. The difference is on line 73, but notice that this function always returns zero. That's intentional, based on what we expect at line 242.

But let's get back to the question of line 73. What this line does is quite interesting. You see, there's no need for the time to be exact. It just needs to be reasonably close. When a tick comes in from the real data, it's handled by CreateBarInReplay. In this case, the st\_Spread value will reflect the timestamp from that tick. But when the call comes from LoopEventOnTime, st\_Spread is simply incremented by one. This is equivalent to a one-second step. No matter the st\_Spread value, as soon as the next real tick arrives, it will be corrected and brought back in line with the real-time value. So, if liquidity drops and there's a 50-second delay between ticks, the timer might slightly lead or lag. You'll see the mouse indicator show a value, then a slightly different one, not necessarily differing by one second. This isn't a bug. In fact, it offers a small benefit. If liquidity dries up for several seconds, you can simply pause and immediately resume the service. As a result, the system will effectively skip the long waiting period. Interesting, isn't it?

### Final Thoughts

To get a clearer picture of everything I've just explained, you can use simulation/replay ticks on a low-liquidity asset. But even if you don't, the video below demonstrates the pause-play trick in action, allowing you to skip long wait times.

However, there's still one more issue we need to address: how can the mouse indicator tell us when an asset might have entered auction mode? That's a tricky topic, complex enough to deserve its own dedicated article. Yes, the solution is already implemented in the mouse indicator. If you place it on a live chart and follow an asset with real-time data, you'll see that when the asset enters an auction, the indicator shows this clearly. But in the case of our replay/simulator, where we use custom assets, this becomes a challenge. There's a specific issue here that complicates things for us. And that, dear reader, will be the subject of our next article. See you soon!

Demonstramyo e 69 b - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12317)

MQL5.community

1.91K subscribers

[Demonstramyo e 69 b](https://www.youtube.com/watch?v=cUYqpOyN6nY)

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

[Watch on](https://www.youtube.com/watch?v=cUYqpOyN6nY&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12317)

0:00

0:00 / 2:26

•Live

•

Demo video

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12317](https://www.mql5.com/pt/articles/12317)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12317.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12317/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/487090)**

![Neural Networks in Trading: Controlled Segmentation (Final Part)](https://c.mql5.com/2/96/Neural_Networks_in_Trading_Controlled_Segmentation___LOGO__1.png)[Neural Networks in Trading: Controlled Segmentation (Final Part)](https://www.mql5.com/en/articles/16057)

We continue the work started in the previous article on building the RefMask3D framework using MQL5. This framework is designed to comprehensively study multimodal interaction and feature analysis in a point cloud, followed by target object identification based on a description provided in natural language.

![MQL5 Wizard Techniques you should know (Part 66): Using Patterns of FrAMA and the Force Index with the Dot Product Kernel](https://c.mql5.com/2/143/18188-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 66): Using Patterns of FrAMA and the Force Index with the Dot Product Kernel](https://www.mql5.com/en/articles/18188)

The FrAMA Indicator and the Force Index Oscillator are trend and volume tools that could be paired when developing an Expert Advisor. We continue from our last article that introduced this pair by considering machine learning applicability to the pair. We are using a convolution neural network that uses the dot-product kernel in making forecasts with these indicators’ inputs. This is done in a custom signal class file that works with the MQL5 wizard to assemble an Expert Advisor.

![Introduction to MQL5 (Part 16): Building Expert Advisors Using Technical Chart Patterns](https://c.mql5.com/2/144/18147-introduction-to-mql5-part-16-logo.png)[Introduction to MQL5 (Part 16): Building Expert Advisors Using Technical Chart Patterns](https://www.mql5.com/en/articles/18147)

This article introduces beginners to building an MQL5 Expert Advisor that identifies and trades a classic technical chart pattern — the Head and Shoulders. It covers how to detect the pattern using price action, draw it on the chart, set entry, stop loss, and take profit levels, and automate trade execution based on the pattern.

![Data Science and ML (Part 41): Forex and Stock Markets Pattern Detection using YOLOv8](https://c.mql5.com/2/143/18143-data-science-and-ml-part-41-logo.png)[Data Science and ML (Part 41): Forex and Stock Markets Pattern Detection using YOLOv8](https://www.mql5.com/en/articles/18143)

Detecting patterns in financial markets is challenging because it involves seeing what's on the chart, something that's difficult to undertake in MQL5 due to image limitations. In this article, we are going to discuss a decent model made in Python that helps us detect patterns present on the chart with minimal effort.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=tfwvizydwpersmnddzexqxixemyungcn&ssn=1769182608223685971&ssn_dr=0&ssn_sr=0&fv_date=1769182608&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12317&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2069)%3A%20Getting%20the%20Time%20Right%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918260866333132&fz_uniq=5069595438803257231&sv=2552)

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