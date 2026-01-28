---
title: Developing a Replay System (Part 50): Things Get Complicated (II)
url: https://www.mql5.com/en/articles/11871
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:07:33.008610
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/11871&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070028255542578882)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 49): Things Get Complicated (I)](https://www.mql5.com/en/articles/11820), we started to complicate things even more within our replay/simulator system. This complication, although unintentional, is intended to make the system more stable and secure: this concerns the possibility of using it in a model that is completely modular.

While these changes may seem completely unnecessary at first, they prevent users from abusing some of the things we are developing by protecting specific parts that are only needed for the replay/simulator system. This way we prevent an inexperienced user from accidentally deleting or misconfiguring parts of the chart that are needed by the replay/simulator service. This happens while the service is running.

At the end of the previous article, I reported that there is a problem that causes the control indicator to be unstable. Due to this, some things can happen that shouldn't actually happen. These issues are not serious and will not crash the platform, but they can sometimes cause unexpected service failures. Because of this, I did not attach any files to the previous article, otherwise you would access the system at the current stage of development.

In this article, we will fix or try to fix existing problems. At least one of them.

### Solving the first problem

The first problem is that the replay/simulator service was implemented to use a template. The template was expected to contain all the data needed to open the chart and start the service.

While this approach is functional and can be considered attractive, it is limiting to the user. To put it more simply: the fact that a service uses a certain template means that the user can no longer use their own methods or a desired chart configuration. It is much more convenient to configure a chart using a user-created template that contains everything you need than having to manually set it up each time you are going to trade a particular asset.

If you are new to the market, this may not make much sense, but more experienced traders have pre-set templates to conduct a standard analysis of a particular asset at a particular point in time. They develop these templates over several months or even years to prepare everything in advance. This way, the trader simply saves the template and applies it to the chart when needed.

Something similar was shown in the article [Developing a Replay System (Part 48): Understanding the concept of a sevice](https://www.mql5.com/en/articles/11781), which is where all this work on modifying the replay/simulator system began. In that article I showed how you can set up a chart using not a template but a service or a standard chart, so that regardless of the template used, certain elements are present on the chart. Such a setup is always supported by the service running on MetaTrader 5 to standardize the work.

However, in order to place the required graphic objects on the chart so that the control indicator can manage the service, important information is needed, including the chart ID that will receive these objects.

You might think that it would be easy to make the indicator know this by simply using the **ChartID()** function. Some say ignorance is bliss. Don't get me wrong, I've had headaches myself every time I tried to figure out why something wasn't working properly at a certain time. So I don't think I'm wrong.

In fact, using the ChartID() function will definitely return the chart ID so that we can place objects on it. Remember: We need the ID to inform MetaTrader 5 on which chart the object will be attached.

However, the ChartID function will not work when the chart is opened through a service. That is, when the service uses the C\_Replay.mqh class and executes the code in line 183, another ID will be created. We have seen this code in the previous article. In the same line 183, we call ChartOpen to create a chart of the symbol on which the replay/simulation will run.

If you compare the values returned by the ChartOpen service with the ChartID value in the control indicator, you can see that they are different. This means that the MetaTrader 5 platform will not know which ID to use. If you use the ID returned by ChartID, you will place objects in the wrong or even non-existent window, but if you use the ID generated inside the service, then as soon as **ChartOpen** creates the ID, we will be able to use objects.

Now the problem arises: What is the best way to solve the Chart ID problem? You might be thinking: Why not use the ID value you get from calling ChartOpen? But this is where the problem lies. In the previous article, we removed the global terminal variable that was responsible for passing the chart ID generated in ChartOpen to the control indicator.

After that, the task for getting the chart ID was transfered to the C\_Terminal class code. This is done using the ChartID function. If you've been following this series and updating the codes accordingly, your C\_Terminal class code should look like this:

```
59. //+------------------------------------------------------------------+
60.             C_Terminal(const long id = 0)
61.                     {
62.                             m_Infos.ID = (id == 0 ? ChartID() : id);
63.                             m_Mem.AccountLock = false;
64.                             CurrentSymbol();
65.                             m_Mem.Show_Descr = ChartGetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR);
66.                             m_Mem.Show_Date  = ChartGetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE);
67.                             ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, false);
68.                             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, 0, true);
69.                             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, 0, true);
70.                             ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, false);
71.                             m_Infos.nDigits = (int) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_DIGITS);
72.                             m_Infos.Width   = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
73.                             m_Infos.Height  = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
74.                             m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
75.                             m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
76.                             m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
77.                             m_Infos.AdjustToTrade = m_Infos.ValuePerPoint / m_Infos.PointPerTick;
78.                             m_Infos.ChartMode       = (ENUM_SYMBOL_CHART_MODE) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_CHART_MODE);
79.                             if(m_Infos.szSymbol != def_SymbolReplay) SetTypeAccount((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE));
80.                             ResetLastError();
81.                     }
82. //+------------------------------------------------------------------+
83.
```

C\_Terminal.mqh source coded fragment

In this code, you can see that line 60 contains the constructor of the C\_Terminal class. It gets a default value of NULL, so the constructor acts like a normal constructor. The actual problem occurs in line 62, where when checking the value passed to the constructor, we determine which chart ID to use. If this value is used by default, the C\_Terminal class will ask MetaTrader 5 to provide the chart ID using the value returned by ChartID. This value will be incorrect if the call is made because the service created the chart and started the indicator that in turn calls C\_Terminal to find out the ID value.

So we can pass to the C\_Terminal class constructor the ID value, and if we do, the ChartID call will be ignored and the ID passed to the constructor will be the one used by the indicator.

Again, remember that we will no longer be using the global terminal variable to pass this value to the indicator. We can do it as something temporary, but the solution will be different. We will pass the ID value through the indicator call parameter.

### Implementing the solution

You might be very surprised at what we are going to do, but that is because in the previous article I did not explain how to implement the solution and how it works before applying it. Please watch Video 01 that shows how the system behaved. This was before the ID was passed as a parameter to the control indicator.

Demonstração Parte 50 1 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11871)

MQL5.community

1.91K subscribers

[Demonstração Parte 50 1](https://www.youtube.com/watch?v=Zh6cDGmn_KU)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 0:48

•Live

•

Video 01

You may notice that an error message appears. This message appears because the indicator cannot know the chart ID, so the ChartID call returns an error and the indicator code checks for this (see line 25 in the indicator code above). But when you look at this code, you will notice that there is a difference between the code and the video: indeed, some things are different. But don't worry, you will soon have access to the code shown in the video, so everything will be more reliable. The difference between what can be seen in the video and the code given above is that at the time I did not understand why the indicator was listed as present on the chart but was not displayed on it.

I had to change the code to figure out why it wasn't working properly. That's why I asked you not to be offended if I said that ignorance is bliss. I also did not understand why the code behaved this way.

However, I will not claim that everything is completely fixed, as that would be dishonest on my part. Passing an ID from a chart to an indicator causes the indicator to be displayed. But... Before we look into that "but", let's look at how we changed the code to get everything working again. At least now the control indicator is presented on the chart.

This did not require changing the source code of the service, but it did require changing the code of the C\_Replay.mqh header file. The full text of the modified file is shown below:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "C_ConfigService.mqh"
005. //+------------------------------------------------------------------+
006. class C_Replay : private C_ConfigService
007. {
008.    private :
009.            long    m_IdReplay;
010.            struct st01
011.            {
012.                    MqlRates Rate[1];
013.                    datetime memDT;
014.            }m_MountBar;
015.            struct st02
016.            {
017.                    bool    bInit;
018.                    double  PointsPerTick;
019.                    MqlTick tick[1];
020.            }m_Infos;
021. //+------------------------------------------------------------------+
022.            void AdjustPositionToReplay(const bool bViewBuider)
023.                    {
024.                            u_Interprocess Info;
025.                            MqlRates       Rate[def_BarsDiary];
026.                            int            iPos, nCount;
027.
028.                            Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
029.                            if (Info.s_Infos.iPosShift == (int)((m_ReplayCount * def_MaxPosSlider * 1.0) / m_Ticks.nTicks)) return;
030.                            iPos = (int)(m_Ticks.nTicks * ((Info.s_Infos.iPosShift * 1.0) / (def_MaxPosSlider + 1)));
031.                            Rate[0].time = macroRemoveSec(m_Ticks.Info[iPos].time);
032.                            CreateBarInReplay(true);
033.                            if (bViewBuider)
034.                            {
035.                                    Info.s_Infos.isWait = true;
036.                                    GlobalVariableSet(def_GlobalVariableReplay, Info.df_Value);
037.                            }else
038.                            {
039.                                    for(; Rate[0].time > (m_Ticks.Info[m_ReplayCount].time); m_ReplayCount++);
040.                                    for (nCount = 0; m_Ticks.Rate[nCount].time < macroRemoveSec(m_Ticks.Info[iPos].time); nCount++);
041.                                    nCount = CustomRatesUpdate(def_SymbolReplay, m_Ticks.Rate, nCount);
042.                            }
043.                            for (iPos = (iPos > 0 ? iPos - 1 : 0); (m_ReplayCount < iPos) && (!_StopFlag);) CreateBarInReplay(false);
044.                            CustomTicksAdd(def_SymbolReplay, m_Ticks.Info, m_ReplayCount);
045.                            Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
046.                            Info.s_Infos.isWait = false;
047.                            GlobalVariableSet(def_GlobalVariableReplay, Info.df_Value);
048.                    }
049. //+------------------------------------------------------------------+
050. inline void CreateBarInReplay(const bool bViewTicks)
051.                    {
052. #define def_Rate m_MountBar.Rate[0]
053.
054.                            bool    bNew;
055.                            double  dSpread;
056.                            int     iRand = rand();
057.
058.                            if (BuildBar1Min(m_ReplayCount, def_Rate, bNew))
059.                            {
060.                                    m_Infos.tick[0] = m_Ticks.Info[m_ReplayCount];
061.                                    if ((!m_Ticks.bTickReal) && (m_Ticks.ModePlot == PRICE_EXCHANGE))
062.                                    {
063.                                            dSpread = m_Infos.PointsPerTick + ((iRand > 29080) && (iRand < 32767) ? ((iRand & 1) == 1 ? m_Infos.PointsPerTick : 0 ) : 0 );
064.                                            if (m_Infos.tick[0].last > m_Infos.tick[0].ask)
065.                                            {
066.                                                    m_Infos.tick[0].ask = m_Infos.tick[0].last;
067.                                                    m_Infos.tick[0].bid = m_Infos.tick[0].last - dSpread;
068.                                            }else   if (m_Infos.tick[0].last < m_Infos.tick[0].bid)
069.                                            {
070.                                                    m_Infos.tick[0].ask = m_Infos.tick[0].last + dSpread;
071.                                                    m_Infos.tick[0].bid = m_Infos.tick[0].last;
072.                                            }
073.                                    }
074.                                    if (bViewTicks) CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
075.                                    CustomRatesUpdate(def_SymbolReplay, m_MountBar.Rate);
076.                            }
077.                            m_ReplayCount++;
078. #undef def_Rate
079.                    }
080. //+------------------------------------------------------------------+
081.            void ViewInfos(void)
082.                    {
083.                            MqlRates Rate[1];
084.
085.                            ChartSetInteger(m_IdReplay, CHART_SHOW_ASK_LINE, m_Ticks.ModePlot == PRICE_FOREX);
086.                            ChartSetInteger(m_IdReplay, CHART_SHOW_BID_LINE, m_Ticks.ModePlot == PRICE_FOREX);
087.                            ChartSetInteger(m_IdReplay, CHART_SHOW_LAST_LINE, m_Ticks.ModePlot == PRICE_EXCHANGE);
088.                            m_Infos.PointsPerTick = SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE);
089.                            m_MountBar.Rate[0].time = 0;
090.                            m_Infos.bInit = true;
091.                            CopyRates(def_SymbolReplay, PERIOD_M1, 0, 1, Rate);
092.                            if ((m_ReplayCount == 0) && (m_Ticks.ModePlot == PRICE_EXCHANGE))
093.                                    for (; m_Ticks.Info[m_ReplayCount].volume_real == 0; m_ReplayCount++);
094.                            if (Rate[0].close > 0)
095.                            {
096.                                    if (m_Ticks.ModePlot == PRICE_EXCHANGE) m_Infos.tick[0].last = Rate[0].close; else
097.                                    {
098.                                            m_Infos.tick[0].bid = Rate[0].close;
099.                                            m_Infos.tick[0].ask = Rate[0].close + (Rate[0].spread * m_Infos.PointsPerTick);
100.                                    }
101.                                    m_Infos.tick[0].time = Rate[0].time;
102.                                    m_Infos.tick[0].time_msc = Rate[0].time * 1000;
103.                            }else
104.                                    m_Infos.tick[0] = m_Ticks.Info[m_ReplayCount];
105.                            CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
106.                            ChartRedraw(m_IdReplay);
107.                    }
108. //+------------------------------------------------------------------+
109.            void CreateGlobalVariable(const string szName, const double value)
110.                    {
111.                            GlobalVariableDel(szName);
112.                            GlobalVariableTemp(szName);
113.                            GlobalVariableSet(szName, value);
114.                    }
115. //+------------------------------------------------------------------+
116.            void AddIndicatorControl(void)
117.                    {
118.                            int handle;
119.
120.                            handle = iCustom(ChartSymbol(m_IdReplay), ChartPeriod(m_IdReplay), "::" + def_IndicatorControl, m_IdReplay);
121.                            ChartIndicatorAdd(m_IdReplay, 0, handle);
122.                            IndicatorRelease(handle);
123.                    }
124. //+------------------------------------------------------------------+
125.    public  :
126. //+------------------------------------------------------------------+
127.            C_Replay(const string szFileConfig)
128.                    {
129.                            m_ReplayCount = 0;
130.                            m_dtPrevLoading = 0;
131.                            m_Ticks.nTicks = 0;
132.                            m_Infos.bInit = false;
133.                            Print("************** Market Replay Service **************");
134.                            srand(GetTickCount());
135.                            GlobalVariableDel(def_GlobalVariableReplay);
136.                            SymbolSelect(def_SymbolReplay, false);
137.                            CustomSymbolDelete(def_SymbolReplay);
138.                            CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay), _Symbol);
139.                            CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
140.                            CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
141.                            CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0);
142.                            CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, 0);
143.                            CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, 0);
144.                            CustomSymbolSetString(def_SymbolReplay, SYMBOL_DESCRIPTION, "Symbol for replay / simulation");
145.                            CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, 8);
146.                            m_IdReplay = (SetSymbolReplay(szFileConfig) ? 0 : -1);
147.                            SymbolSelect(def_SymbolReplay, true);
148.                    }
149. //+------------------------------------------------------------------+
150.            ~C_Replay()
151.                    {
152.                            ArrayFree(m_Ticks.Info);
153.                            ArrayFree(m_Ticks.Rate);
154.                            m_IdReplay = ChartFirst();
155.                            do
156.                            {
157.                                    if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
158.                                            ChartClose(m_IdReplay);
159.                            }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
160.                            for (int c0 = 0; (c0 < 2) && (!SymbolSelect(def_SymbolReplay, false)); c0++);
161.                            CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
162.                            CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
163.                            CustomSymbolDelete(def_SymbolReplay);
164.                            GlobalVariableDel(def_GlobalVariableReplay);
165.                            GlobalVariableDel(def_GlobalVariableServerTime);
166.                            Print("Finished replay service...");
167.                    }
168. //+------------------------------------------------------------------+
169.            bool ViewReplay(ENUM_TIMEFRAMES arg1)
170.                    {
171. #define macroError(A) { Print(A); return false; }
172.                            u_Interprocess info;
173.
174.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
175.                                    macroError("Asset configuration is not complete, it remains to declare the size of the ticket.");
176.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
177.                                    macroError("Asset configuration is not complete, need to declare the ticket value.");
178.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
179.                                    macroError("Asset configuration not complete, need to declare the minimum volume.");
180.                            if (m_IdReplay == -1) return false;
181.                            if ((m_IdReplay = ChartFirst()) > 0) do
182.                            {
183.                                    if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
184.                                    {
185.                                            ChartClose(m_IdReplay);
186.                                            ChartRedraw();
187.                                    }
188.                            }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
189.                            Print("Waiting for [Market Replay] indicator permission to start replay ...");
190.                            info.ServerTime = ULONG_MAX;
191.                            CreateGlobalVariable(def_GlobalVariableServerTime, info.df_Value);
192.                            m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
193.                            AddIndicatorControl();
194.                            while ((!GlobalVariableGet(def_GlobalVariableReplay, info.df_Value)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);
195.                            info.s_Infos.isHedging = TypeAccountIsHedging();
196.                            info.s_Infos.isSync = true;
197.                            GlobalVariableSet(def_GlobalVariableReplay, info.df_Value);
198.
199.                            return ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""));
200. #undef macroError
201.                    }
202. //+------------------------------------------------------------------+
203.            bool LoopEventOnTime(const bool bViewBuider)
204.                    {
205.                            u_Interprocess Info;
206.                            int iPos, iTest, iCount;
207.
208.                            if (!m_Infos.bInit) ViewInfos();
209.                            iTest = 0;
210.                            while ((iTest == 0) && (!_StopFlag))
211.                            {
212.                                    iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);
213.                                    iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.df_Value) ? iTest : -1);
214.                                    iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);
215.                                    if (iTest == 0) Sleep(100);
216.                            }
217.                            if ((iTest < 0) || (_StopFlag)) return false;
218.                            AdjustPositionToReplay(bViewBuider);
219.                            Info.ServerTime = m_Ticks.Info[m_ReplayCount].time;
220.                            GlobalVariableSet(def_GlobalVariableServerTime, Info.df_Value);
221.                            iPos = iCount = 0;
222.                            while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))
223.                            {
224.                                    iPos += (int)(m_ReplayCount < (m_Ticks.nTicks - 1) ? m_Ticks.Info[m_ReplayCount + 1].time_msc - m_Ticks.Info[m_ReplayCount].time_msc : 0);
225.                                    CreateBarInReplay(true);
226.                                    while ((iPos > 200) && (!_StopFlag))
227.                                    {
228.                                            if (ChartSymbol(m_IdReplay) == "") return false;
229.                                            GlobalVariableGet(def_GlobalVariableReplay, Info.df_Value);
230.                                            if (!Info.s_Infos.isPlay) return true;
231.                                            Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
232.                                            GlobalVariableSet(def_GlobalVariableReplay, Info.df_Value);
233.                                            Sleep(195);
234.                                            iPos -= 200;
235.                                            iCount++;
236.                                            if (iCount > 4)
237.                                            {
238.                                                    iCount = 0;
239.                                                    GlobalVariableGet(def_GlobalVariableServerTime, Info.df_Value);
240.                                                    if ((m_Ticks.Info[m_ReplayCount].time - m_Ticks.Info[m_ReplayCount - 1].time) > 60) Info.ServerTime = ULONG_MAX; else
241.                                                    {
242.                                                            Info.ServerTime += 1;
243.                                                            Info.ServerTime = ((Info.ServerTime + 1) < m_Ticks.Info[m_ReplayCount].time ? Info.ServerTime : m_Ticks.Info[m_ReplayCount].time);
244.                                                    };
245.                                                    GlobalVariableSet(def_GlobalVariableServerTime, Info.df_Value);
246.                                            }
247.                                    }
248.                            }
249.                            return (m_ReplayCount == m_Ticks.nTicks);
250.                    }
251. //+------------------------------------------------------------------+
252. };
253. //+------------------------------------------------------------------+
254. #undef macroRemoveSec
255. #undef def_SymbolReplay
256. //+------------------------------------------------------------------+
```

C\_Replay.mqh class source code

While we've modified this code to support what we need, it will allow us to do much more in the near future. Let's first see what's been added. You can see that most of the code remains the same as in the previous article. Since I want you to have a good understanding of how this is implemented, I have included the full code so you know for sure where to place the functions used.

In line 116, I added a new procedure to add the control indicator to the chart. This procedure is called in line 193, that is, immediately after the chart has been opened by the service and displayed by MetaTrader 5. But let's go back to line 116. The first thing we do is create a handle in line 120 that will reference the indicator present in the service code. Remember that the indicator is embedded into the service executable file as a resource. After we have told MetaTrader 5 where the indicator is located, we need to provide it with some information. It represents m\_IdReplay, which is the chart ID created by the ChartOpen call.

Thus, the indicator will know which of the chart IDs is correct. Please pay attention to this. Even if you open another chart associated with the replay symbol, the indicator will not appear and will only be displayed on the chart that was created by the service. This is imposed and executed in line 121. Then, in line 122, we free the created handle since we don't need it anymore.

But what we have just seen is only part of the solution. The other part is located in the source code of the control indicator. You can see it below:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property description "Control indicator for the Replay-Simulator service."
05. #property description "This one doesn't work without the service loaded."
06. #property version   "1.50"
07. #property link "https://www.mql5.com/en/articles/11871"
08. #property indicator_chart_window
09. #property indicator_plots 0
10. //+------------------------------------------------------------------+
11. #include <Market Replay\Service Graphics\C_Controls.mqh>
12. //+------------------------------------------------------------------+
13. C_Terminal *terminal = NULL;
14. C_Controls *control = NULL;
15. //+------------------------------------------------------------------+
16. input long user00 = 0;   //ID
17. //+------------------------------------------------------------------+
18. int OnInit()
19. {
20.     u_Interprocess Info;
21.
22.     ResetLastError();
23.     if (CheckPointer(control = new C_Controls(terminal = new C_Terminal(user00))) == POINTER_INVALID)
24.             SetUserError(C_Terminal::ERR_PointerInvalid);
25.     if ((!(*terminal).IndicatorCheckPass("Market Replay Control")) || (_LastError != ERR_SUCCESS))
26.     {
27.             Print("Control indicator failed on initialization.");
28.             return INIT_FAILED;
29.     }
30.     if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.df_Value = 0;
31.     EventChartCustom(user00, C_Controls::ev_WaitOff, 1, Info.df_Value, "");
32.     (*control).Init(Info.s_Infos.isPlay);
33.
34.     return INIT_SUCCEEDED;
35. }
36. //+------------------------------------------------------------------+
37. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
38. {
39.     static bool bWait = false;
40.     u_Interprocess Info;
41.
42.     Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
43.     if (!bWait)
44.     {
45.             if (Info.s_Infos.isWait)
46.             {
47.                     EventChartCustom(user00, C_Controls::ev_WaitOn, 1, 0, "");
48.                     bWait = true;
49.             }
50.     }else if (!Info.s_Infos.isWait)
51.     {
52.             EventChartCustom(user00, C_Controls::ev_WaitOff, 1, Info.df_Value, "");
53.             bWait = false;
54.     }
55.
56.     return rates_total;
57. }
58. //+------------------------------------------------------------------+
59. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
60. {
61.     (*control).DispatchMessage(id, lparam, dparam, sparam);
62. }
63. //+------------------------------------------------------------------+
64. void OnDeinit(const int reason)
65. {
66.     switch (reason)
67.     {
68.             case REASON_REMOVE:
69.             case REASON_CHARTCLOSE:
70.                     if (ChartSymbol(user00) != def_SymbolReplay) break;
71.                     GlobalVariableDel(def_GlobalVariableReplay);
72.                     ChartClose(user00);
73.                     break;
74.     }
75.     delete control;
76.     delete terminal;
77. }
78. //+------------------------------------------------------------------+
```

Source code of the control indicator

Notice that we now have an input in line 16, meaning the indicator will receive a parameter. This is one of the reasons why the user is not given direct access to this indicator and cannot place it manually on the chart. This parameter, which the indicator will receive in line 16, indicates the ID of the chart on which the objects will be placed. This value must be filled in correctly. By default it will be NULL, i.e. if the service tries to place an indicator but does not provide a chart, an error will be generated. The error message appears in line 27. This explains the difference between what was expected and what was presented in Video 01.

Now notice where the parameter value entered in line 16 is used. It is used in several places, but the main one is line 23, where we tell the C\_Terminal class that it should not use the generated value when looking up the chart ID. The C\_Terminal class should use the value reported by the service that created the chart. As you can see, other parts also use the value specified in line 16. However, after recompiling the service file and running it in MetaTrader 5, we get the result shown in video 02.

Demonstração Parte 50 2 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11871)

MQL5.community

1.91K subscribers

[Demonstração Parte 50 2](https://www.youtube.com/watch?v=unjD9GJhpSY)

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

[Watch on](https://www.youtube.com/watch?v=unjD9GJhpSY&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11871)

0:00

0:00 / 1:28

•Live

•

Video 02

Watch video 02 carefully. As you can see, the system behaves quite strangely when we try to interact with it. The question is why this happens. We have not made any changes that would cause this behavior.

This is the second problem we have to solve. However, the solution to this problem is much more complicated. The reason here is not in the service, not in the control indicator, and not in the MetaTrader 5 platform. This is the interaction or lack of interaction of the mouse indicator with objects on the chart.

At this point you're probably thinking, "How are we going to fix a bug that didn't exist before we started changing seemingly perfectly working code?" That was before the fateful decision was made to allow the user to use a custom template rather than the one the replay/simulator system had long been using. Fine. This is programming: It solves problems that arise when introducing new elements, and it solves future problems.

But before we address this issue and solve the problem of the mouse indicator interacting with the chart objects created by the control indicator, let's implement a function that allows the user to use a custom template. This will allow the user to apply a pre-set template intended to be used with the specific symbol on a demo or real account. But now we will allow the user to use the same template in the replay/simulator system.

It's not difficult to do this. However, users might tend to do things in an illogical manner, so we need to make sure that the control indicator remains on the chart. We should protect it even if the user insists on using the replay/simulator system in a completely unintended way.

In this case, the first thing to do is to enable the user to instruct the service to open the chart according to a certain template. This can be easily done with some small changes. To avoid repeating the entire code, I will post only the part of the function that needs to be changed. First, let's look at the service code. It is located just below:

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property copyright "Daniel Jose"
05. #property version   "1.50"
06. #property description "Replay-Simulator service for MT5 platform."
07. #property description "This is dependent on the Market Replay indicator."
08. #property description "For more details on this version see the article."
09. #property link "https://www.mql5.com/en/articles/11871"
10. //+------------------------------------------------------------------+
11. #define def_IndicatorControl        "Indicators\\Replay\\Market Replay.ex5"
12. #resource "\\" + def_IndicatorControl
13. //+------------------------------------------------------------------+
14. #include <Market Replay\Service Graphics\C_Replay.mqh>
15. //+------------------------------------------------------------------+
16. input string             user00 = "Forex - EURUSD.txt";  //Replay Configuration File.
17. input ENUM_TIMEFRAMES    user01 = PERIOD_M5;             //Initial Graphic Time.
18. input string             user02 = "Default";             //Template File Name
19. //+------------------------------------------------------------------+
20. void OnStart()
21. {
22.     C_Replay  *pReplay;
23.
24.     pReplay = new C_Replay(user00);
25.     if ((*pReplay).ViewReplay(user01, user02))
26.     {
27.             Print("Permission granted. Replay service can now be used...");
28.             while ((*pReplay).LoopEventOnTime(false));
29.     }
30.     delete pReplay;
31. }
32. //+------------------------------------------------------------------+
```

Source code of the service

We have added a new line 18, which gives the user the option to specify which template should be used when the service opens a symbol chart for use as a replay or simulator. This value is passed to the class in line 25, where we make the necessary configurations. To understand what is happening, look at a fragment of this procedure. I provide only this fragment, since repeating the entire code would be useless.

```
175. //+------------------------------------------------------------------+
176.            bool ViewReplay(ENUM_TIMEFRAMES arg1, const string szNameTemplate)
177.                    {
178. #define macroError(A) { Print(A); return false; }
179.                            u_Interprocess info;
180.
181.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
182.                                    macroError("Asset configuration is not complete, it remains to declare the size of the ticket.");
183.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
184.                                    macroError("Asset configuration is not complete, need to declare the ticket value.");
185.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
186.                                    macroError("Asset configuration not complete, need to declare the minimum volume.");
187.                            if (m_IdReplay == -1) return false;
188.                            if ((m_IdReplay = ChartFirst()) > 0) do
189.                            {
190.                                    if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
191.                                    {
192.                                            ChartClose(m_IdReplay);
193.                                            ChartRedraw();
194.                                    }
195.                            }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
196.                            Print("Waiting for [Market Replay] indicator permission to start replay ...");
197.                            info.ServerTime = ULONG_MAX;
198.                            CreateGlobalVariable(def_GlobalVariableServerTime, info.df_Value);
199.                            m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
200.                            if (!ChartApplyTemplate(m_IdReplay, szNameTemplate + ".tpl"))
201.                                    Print("Failed apply template: ", szNameTemplate, ".tpl Using template default.tpl");
202.                            AddIndicatorControl();
203.                            while ((!GlobalVariableGet(def_GlobalVariableReplay, info.df_Value)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);
204.                            info.s_Infos.isHedging = TypeAccountIsHedging();
205.                            info.s_Infos.isSync = true;
206.                            GlobalVariableSet(def_GlobalVariableReplay, info.df_Value);
207.
208.                            return ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""));
209. #undef macroError
210.                    }
211. //+------------------------------------------------------------------+
```

C\_Replay.mqh source code fragment (Update)

Note that in line 176, we add an extra parameter, which informs the class which template will be loaded. Then in line 200 we try to apply the template provided by the user. If the attempt fails, in line 201 we inform the user about this and switch to the default template. Let's now make a pause to explain something. By default, MetaTrader 5 will always use a predefined template, so there is no need for a new call to load and apply it to the chart.

Something interesting to clarify at this point: If you are working with multiple symbols and want to run targeted studies for each symbol using a specific template, it might make sense to add the name of the desired template to the modeling configuration file. It will be pre-configured and there will be no need to notify about the service launch. But this is just an idea for those who want to do it. I'm not going to do that in this system. Moreover, this story with the template has another aspect. This is precisely the aspect that is a headache for programmers. The aspect is the user.

Why is the user a problem for us? It may not make much sense. But this is a big headache, because we can tell MetaTrader 5 to start the service and at the right moment the chart template for the replay/simulator can be changed. This happens without restarting the service to allow it to apply changes. Perhaps you did not understand the essence of the problem. In this case, the template settings will override the chart settings, and the problem is that the control indicator will be removed from the chart.

The user will not be able to manually place the control indicator on the chart. There are ways to do this though. But let's take into account the fact that if the user does not know how to work in MetaTrader 5 and MQL5, then they will not be able to replace the control indicator on the chart. This type of situation is exactly the kind of failure for which the user is responsible. But the one who is responsible for its correction is the programmer.

When a template is changed, MetaTrader 5 will notify the programs on the chart that a template change has occurred so that they can take action. This is how MetaTrader 5 does this:

```
void OnDeinit(const int reason)
{
        switch (reason)
        {
                case REASON_TEMPLATE:
                        Print("Template change ...");
                        break;
```

When a template is changed, a DeInit event occurs in MetaTrader 5, which will call the procedure shown above. We can check the condition that caused MetaTrader 5 to call the DeInit event, and if it is a template change, then the relevant message will be displayed in the terminal.

In other words, we can know if the template has changed, but this knowledge does not immediately cause the indicator to reset. Here we have to make a decision: force the service to terminate or force the service to re-place the control indicator on the chart. In my subjective opinion, we should forcibly terminate the service. The reason is simple. If the user is allowed to customize the template that will be used on the replay/simulator chart, why should we allow the user to manually change the template? It's pointless. So, in my opinion, the best thing to do is just force the service to terminate and ask the user to provide a template when the service starts. Otherwise, why would there be a need to allow the user to provide a template to the service?

So, we need to make another simple update in the indicator code, which can be seen below:

```
64. //+------------------------------------------------------------------+
65. void OnDeinit(const int reason)
66. {
67.     switch (reason)
68.     {
69.             case REASON_TEMPLATE:
70.                     Print("Modified template. Replay/simulation system shutting down.");
71.             case REASON_PARAMETERS:
72.             case REASON_REMOVE:
73.             case REASON_CHARTCLOSE:
74.                     if (ChartSymbol(user00) != def_SymbolReplay) break;
75.                     GlobalVariableDel(def_GlobalVariableReplay);
76.                     ChartClose(user00);
77.                     break;
78.     }
79.     delete control;
80.     delete terminal;
81. }
82. //+------------------------------------------------------------------+
```

Fragment of the control indicator source code (update)

Note that in line 69 we add a test to check why the indicator is removed from the chart. If the reason is a template change, line 70 will print a message to the terminal and we will get the same result as if the user closed the chart or the indicator was deleted by the user. That is, the service will terminate. This decision may seem too radical, but as I have already explained, there is no point in doing otherwise.

We have fixed one bug and now we have another problem: the user can simply change any parameter used by the control indicator and passed to it by the service. We are still configuring the system, so new parameters may appear. It is much more difficult to cope with such a situation, but we will take a radical approach to solving this issue. This solution has already been implemented in the fragment above.

Pay attention to line 71. There was no such line before. It has been added to prevent user from changing any of the parameters that the service passed to the control indicator. If this happens, MetaTrader 5 will generate a DeInit event with the argument being the parameter change. We will not report any failure at this moment as the failure will occur when MetaTrader 5 re-starts the indicator. But since some more astute user can provide the ID of the actual chart, we close the replay/simulator chart at line 76. So when the service checks if the chart is open, it will get an error, which means that the service should be terminated. Thus, we correct this issue as well.

### Conclusion

In this article, we fixed several primary bugs that were caused by the fact that the control indicator became unavailable to the user. While it is still visible in the indicators window via the CTRL+I shortcut, it is no longer included in the list of indicators that can be used on any chart. This type of change involves a lot of modifications to the code to make it more stable and consistent and to prevent the user from doing something we don't want or expect.

However, we still have an issue that makes it difficult for the user to interact with the indicator, adjust and manipulate the control indicator using the mouse indicator. But this issue is related to something that we will fix soon, which will make the system even more free and fully modular.

In video 03, located just below, you can see how the system now behaves with the updates described in this article. However, since the code is still unstable, I will not attach any files to this article.

Demonstração Parte 50 3 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11871)

MQL5.community

1.91K subscribers

[Demonstração Parte 50 3](https://www.youtube.com/watch?v=NLKXMYsVIV0)

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

[Watch on](https://www.youtube.com/watch?v=NLKXMYsVIV0&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11871)

0:00

0:00 / 2:51

•Live

•

Video 03

In the next article, we will continue to address issues and problems related to the interaction between the user and the replay/simulator service.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11871](https://www.mql5.com/pt/articles/11871)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11871.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11871/anexo.zip "Download Anexo.zip")(420.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/475588)**
(1)


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
25 May 2024 at 21:29

Will there be playback speed [control](https://www.mql5.com/en/articles/310 "Article: Customised graphic controls Part 1: Creating a simple control")?


![Exploring Cryptography in MQL5: A Step-by-Step Approach](https://c.mql5.com/2/99/Exploring_Cryptography_in_MQL5__LOGO.png)[Exploring Cryptography in MQL5: A Step-by-Step Approach](https://www.mql5.com/en/articles/16238)

This article explores the integration of cryptography within MQL5, enhancing the security and functionality of trading algorithms. We’ll cover key cryptographic methods and their practical implementation in automated trading.

![Trading with the MQL5 Economic Calendar (Part 1): Mastering the Functions of the MQL5 Economic Calendar](https://c.mql5.com/2/99/Trading_with_the_MQL5_Economic_Calendar_Part_1___LOGO.png)[Trading with the MQL5 Economic Calendar (Part 1): Mastering the Functions of the MQL5 Economic Calendar](https://www.mql5.com/en/articles/16223)

In this article, we explore how to use the MQL5 Economic Calendar for trading by first understanding its core functionalities. We then implement key functions of the Economic Calendar in MQL5 to extract relevant news data for trading decisions. Finally, we conclude by showcasing how to utilize this information to enhance trading strategies effectively.

![Building A Candlestick Trend Constraint Model (Part 9): Multiple Strategies Expert Advisor (II)](https://c.mql5.com/2/99/Building_A_Candlestick_Trend_Constraint_Model_Part_9__P2___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 9): Multiple Strategies Expert Advisor (II)](https://www.mql5.com/en/articles/16137)

The number of strategies that can be integrated into an Expert Advisor is virtually limitless. However, each additional strategy increases the complexity of the algorithm. By incorporating multiple strategies, an Expert Advisor can better adapt to varying market conditions, potentially enhancing its profitability. Today, we will explore how to implement MQL5 for one of the prominent strategies developed by Richard Donchian, as we continue to enhance the functionality of our Trend Constraint Expert.

![News Trading Made Easy (Part 4): Performance Enhancement](https://c.mql5.com/2/99/News_Trading_Made_Easy_Part_4__LOGO__2.png)[News Trading Made Easy (Part 4): Performance Enhancement](https://www.mql5.com/en/articles/15878)

This article will dive into methods to improve the expert's runtime in the strategy tester, the code will be written to divide news event times into hourly categories. These news event times will be accessed within their specified hour. This ensures that the EA can efficiently manage event-driven trades in both high and low-volatility environments.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/11871&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070028255542578882)

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