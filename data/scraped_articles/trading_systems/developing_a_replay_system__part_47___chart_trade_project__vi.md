---
title: Developing a Replay System (Part 47): Chart Trade Project (VI)
url: https://www.mql5.com/en/articles/11760
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:08:23.798435
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=tefikgmaafwpoljfwbfseshhfhtlrjjy&ssn=1769184502442538032&ssn_dr=0&ssn_sr=0&fv_date=1769184502&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11760&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2047)%3A%20Chart%20Trade%20Project%20(VI)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918450210697159&fz_uniq=5070040560623881973&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 46): Chart Trade Project (V)](https://www.mql5.com/en/articles/11737), I showed how you can add data to an executable file so that it doesn't have to be transferred separately. This knowledge is very important for what will be done in the near future. But for now, we will continue to develop what needs to be implemented first.

In this article, we will improve the Chart Trade indicator, making it functional enough to be used with some EAs. This will allow us to access the Chart Trade indicator and work with it as if it were actually connected with an EA. But let's make it much more interesting than in one of the past articles [Developing a trading Expert Advisor from scratch (Part 30): CHART TRADE as an indicator?!](https://www.mql5.com/en/articles/10653). In that article, we used Chart Trade as a conditional indicator. This time it will be a real indicator.

To do this, we will make it function in a very specific way, just like when working with any other type of indicator. For this, we will create a corresponding data buffer. This process has been described previously in other articles, information can be found here:

- [Developing a Replay System (Part 37): Paving the Path (I)](https://www.mql5.com/en/articles/11585)
- [Developing a Replay System (Part 38): Paving the Path (II)](https://www.mql5.com/en/articles/11591)
- [Developing a Replay System (Part 39): Paving the Path (III)](https://www.mql5.com/en/articles/11599)

These 3 articles contain the basis of what we are actually going to do. If you haven't read them, I encourage you to do so. Otherwise, you risk finding yourself confused while reading this article and having difficulty understanding it due to the lack of the very deeper knowledge on which it is based. So, you should read the mentioned articles and understand their contents well.

Before we start making changes, there are a few small modifications that need to be made. They are all included in the existing C\_ChartFloatingRAD class code to provide easy and adequate access to the data we need. So, we are moving on to the final stage of work on the Chart Trade indicator.

### Small Changes, Big Results

The changes that will be made are few and simple. Of course, provided that you have been following this series of articles. Now we are in the second stage of articles on the replay/simulator system. Below is the full code of the Chart Trade indicator. It is important to review this code first to make it easier to explain the class code.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Chart Trade base indicator."
04. #property description "This version communicates via buffer with the EA."
05. #property description "See the articles for more details."
06. #property version   "1.47"
07. #property icon "/Images/Market Replay/Icons/Indicators.ico"
08. #property link "https://www.mql5.com/es/articles/11760"
09. #property indicator_chart_window
10. #property indicator_plots 0
11. #property indicator_buffers 1
12. //+------------------------------------------------------------------+
13. #include <Market Replay\Chart Trader\C_ChartFloatingRAD.mqh>
14. //+------------------------------------------------------------------+
15. C_ChartFloatingRAD *chart = NULL;
16. //+------------------------------------------------------------------+
17. input int      user01 = 1;       //Leverage
18. input double   user02 = 100.1;   //Finance Take
19. input double   user03 = 75.4;    //Finance Stop
20. //+------------------------------------------------------------------+
21. double m_Buff[];
22. //+------------------------------------------------------------------+
23. int OnInit()
24. {
25.     bool bErr;
26.
27.     chart = new C_ChartFloatingRAD("Indicator Chart Trade", new C_Mouse("Indicator Mouse Study"), user01, user02, user03);
28.
29.     if (bErr = (_LastError != ERR_SUCCESS)) Print(__FILE__, " - [Error]: ", _LastError);
30.
31.     SetIndexBuffer(0, m_Buff, INDICATOR_DATA);
32.     ArrayInitialize(m_Buff, EMPTY_VALUE);
33.
34.     return (bErr ? INIT_FAILED : INIT_SUCCEEDED);
35. }
36. //+------------------------------------------------------------------+
37. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
38. {
39.     (*chart).MountBuffer(m_Buff, rates_total);
40.
41.     return rates_total;
42. }
43. //+------------------------------------------------------------------+
44. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
45. {
46.     (*chart).DispatchMessage(id, lparam, dparam, sparam);
47.     (*chart).MountBuffer(m_Buff);
48.
49.     ChartRedraw();
50. }
51. //+------------------------------------------------------------------+
52. void OnDeinit(const int reason)
53. {
54.     if (reason == REASON_CHARTCHANGE) (*chart).SaveState();
55.
56.     delete chart;
57. }
58. //+------------------------------------------------------------------+
```

Chart Trade indicator source code

Please note that all the code above contains everything needed for the Chart Trade indicator to function. However, for practical reasons, most of the code was moved to a class that will be discussed later. But how does this code work? How does it allow us to send commands to the EA to perform operations? Wait, let's first figure out what's going on here in the indicator code.

Line 11 contains the first of the stages we need. In this line, we define that we will use 1 buffer. We could use more buffers, but one will be enough.

The buffer we will use is declared in line 21. But we only define how to use it in line 31. Since we don't want the buffer to be filled with "garbage", we initialize it in line 32 to contain only zero values.

As you can see, the indicator code has not undergone any major changes compared to what was in the previous articles. But it now has two new lines: 39 and 47. Both lines call the same function inside the class, which we'll look at shortly. You could assume that these are different functions due to the difference in number and parameters. However, you will soon realize that they are both the same. To understand this, let's look at the full code of the class, which is given below.

```
001.//+------------------------------------------------------------------+
002.#property copyright "Daniel Jose"
003.//+------------------------------------------------------------------+
004.#include "../Auxiliar/C_Mouse.mqh"
005.#include "../Auxiliar/Interprocess.mqh"
006.#include "C_AdjustTemplate.mqh"
007.//+------------------------------------------------------------------+
008.#define macro_NameGlobalVariable(A) StringFormat("ChartTrade_%u%s", GetInfoTerminal().ID, A)
009.//+------------------------------------------------------------------+
010.class C_ChartFloatingRAD : private C_Terminal
011.{
012.    public          :
013.            enum eObjectsIDE {MSG_LEVERAGE_VALUE, MSG_TAKE_VALUE, MSG_STOP_VALUE, MSG_MAX_MIN, MSG_TITLE_IDE, MSG_DAY_TRADE, MSG_BUY_MARKET, MSG_SELL_MARKET, MSG_CLOSE_POSITION, MSG_NULL};
014.            struct stData
015.            {
016.                    int             Leverage;
017.                    double          PointsTake,
018.                                    PointsStop;
019.                    bool            IsDayTrade;
020.                    union u01
021.                    {
022.                            ulong   TickCount;
023.                            double  dValue;
024.                    }uCount;
025.                    eObjectsIDE Msg;
026.            };
027.    private :
028.            struct st00
029.            {
030.                    int     x, y, minx, miny;
031.                    string  szObj_Chart,
032.                            szObj_Editable,
033.                            szFileNameTemplate;
034.                    long    WinHandle;
035.                    double  FinanceTake,
036.                            FinanceStop;
037.                    bool    IsMaximized;
038.                    stData  ConfigChartTrade;
039.                    struct st01
040.                    {
041.                            int    x, y, w, h;
042.                            color  bgcolor;
043.                            int    FontSize;
044.                            string FontName;
045.                    }Regions[MSG_NULL];
046.            }m_Info;
047.//+------------------------------------------------------------------+
048.            C_Mouse   *m_Mouse;
049.            string    m_szShortName;
050.//+------------------------------------------------------------------+
051.            void CreateWindowRAD(int w, int h)
052.                    {
053.                            m_Info.szObj_Chart = "Chart Trade IDE";
054.                            m_Info.szObj_Editable = m_Info.szObj_Chart + " > Edit";
055.                            ObjectCreate(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJ_CHART, 0, 0, 0);
056.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x);
057.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y);
058.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XSIZE, w);
059.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, h);
060.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_DATE_SCALE, false);
061.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_PRICE_SCALE, false);
062.                            m_Info.WinHandle = ObjectGetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_CHART_ID);
063.                    };
064.//+------------------------------------------------------------------+
065.            void AdjustEditabled(C_AdjustTemplate &Template, bool bArg)
066.                    {
067.                            for (eObjectsIDE c0 = 0; c0 <= MSG_STOP_VALUE; c0++)
068.                                    if (bArg)
069.                                    {
070.                                            Template.Add(EnumToString(c0), "bgcolor", NULL);
071.                                            Template.Add(EnumToString(c0), "fontsz", NULL);
072.                                            Template.Add(EnumToString(c0), "fontnm", NULL);
073.                                    }
074.                                    else
075.                                    {
076.                                            m_Info.Regions[c0].bgcolor = (color) StringToInteger(Template.Get(EnumToString(c0), "bgcolor"));
077.                                            m_Info.Regions[c0].FontSize = (int) StringToInteger(Template.Get(EnumToString(c0), "fontsz"));
078.                                            m_Info.Regions[c0].FontName = Template.Get(EnumToString(c0), "fontnm");
079.                                    }
080.                    }
081.//+------------------------------------------------------------------+
082.inline void AdjustTemplate(const bool bFirst = false)
083.                    {
084.#define macro_AddAdjust(A) {                         \
085.             (*Template).Add(A, "size_x", NULL);     \
086.             (*Template).Add(A, "size_y", NULL);     \
087.             (*Template).Add(A, "pos_x", NULL);      \
088.             (*Template).Add(A, "pos_y", NULL);      \
089.                           }
090.#define macro_GetAdjust(A) {                                                                                                                                                        \
091.             m_Info.Regions[A].x = (int) StringToInteger((*Template).Get(EnumToString(A), "pos_x"));   \
092.             m_Info.Regions[A].y = (int) StringToInteger((*Template).Get(EnumToString(A), "pos_y"));   \
093.             m_Info.Regions[A].w = (int) StringToInteger((*Template).Get(EnumToString(A), "size_x"));  \
094.             m_Info.Regions[A].h = (int) StringToInteger((*Template).Get(EnumToString(A), "size_y"));  \
095.                           }
096.#define macro_PointsToFinance(A) A * (GetInfoTerminal().VolumeMinimal + (GetInfoTerminal().VolumeMinimal * (m_Info.ConfigChartTrade.Leverage - 1))) * GetInfoTerminal().AdjustToTrade
097.
098.                            C_AdjustTemplate  *Template;
099.
100.                            if (bFirst)
101.                            {
102.                                    Template = new C_AdjustTemplate(m_Info.szFileNameTemplate = IntegerToString(GetInfoTerminal().ID) + ".tpl", true);
103.                                    for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++) macro_AddAdjust(EnumToString(c0));
104.                                    AdjustEditabled(Template, true);
105.                            }else Template = new C_AdjustTemplate(m_Info.szFileNameTemplate);
106.                            m_Info.ConfigChartTrade.Leverage = (m_Info.ConfigChartTrade.Leverage <= 0 ? 1 : m_Info.ConfigChartTrade.Leverage);
107.                            m_Info.FinanceTake = macro_PointsToFinance(FinanceToPoints(MathAbs(m_Info.FinanceTake), m_Info.ConfigChartTrade.Leverage));
108.                            m_Info.FinanceStop = macro_PointsToFinance(FinanceToPoints(MathAbs(m_Info.FinanceStop), m_Info.ConfigChartTrade.Leverage));
109.                            m_Info.ConfigChartTrade.PointsTake = FinanceToPoints(m_Info.FinanceTake, m_Info.ConfigChartTrade.Leverage);
110.                            m_Info.ConfigChartTrade.PointsStop = FinanceToPoints(m_Info.FinanceStop, m_Info.ConfigChartTrade.Leverage);
111.                            (*Template).Add("MSG_NAME_SYMBOL", "descr", GetInfoTerminal().szSymbol);
112.                            (*Template).Add("MSG_LEVERAGE_VALUE", "descr", IntegerToString(m_Info.ConfigChartTrade.Leverage));
113.                            (*Template).Add("MSG_TAKE_VALUE", "descr", DoubleToString(m_Info.FinanceTake, 2));
114.                            (*Template).Add("MSG_STOP_VALUE", "descr", DoubleToString(m_Info.FinanceStop, 2));
115.                            (*Template).Add("MSG_DAY_TRADE", "state", (m_Info.ConfigChartTrade.IsDayTrade ? "1" : "0"));
116.                            (*Template).Add("MSG_MAX_MIN", "state", (m_Info.IsMaximized ? "1" : "0"));
117.                            (*Template).Execute();
118.                            if (bFirst)
119.                            {
120.                                    for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++) macro_GetAdjust(c0);
121.                                    m_Info.Regions[MSG_TITLE_IDE].w = m_Info.Regions[MSG_MAX_MIN].x;
122.                                    AdjustEditabled(Template, false);
123.                            };
124.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, (m_Info.IsMaximized ? 210 : m_Info.Regions[MSG_TITLE_IDE].h + 6));
125.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, (m_Info.IsMaximized ? m_Info.x : m_Info.minx));
126.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, (m_Info.IsMaximized ? m_Info.y : m_Info.miny));
127.
128.                            delete Template;
129.
130.                            ChartApplyTemplate(m_Info.WinHandle, "/Files/" + m_Info.szFileNameTemplate);
131.                            ChartRedraw(m_Info.WinHandle);
132.
133.#undef macro_PointsToFinance
134.#undef macro_GetAdjust
135.#undef macro_AddAdjust
136.                    }
137.//+------------------------------------------------------------------+
138.            eObjectsIDE CheckMousePosition(const int x, const int y)
139.                    {
140.                            int xi, yi, xf, yf;
141.
142.                            for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++)
143.                            {
144.                                    xi = (m_Info.IsMaximized ? m_Info.x : m_Info.minx) + m_Info.Regions[c0].x;
145.                                    yi = (m_Info.IsMaximized ? m_Info.y : m_Info.miny) + m_Info.Regions[c0].y;
146.                                    xf = xi + m_Info.Regions[c0].w;
147.                                    yf = yi + m_Info.Regions[c0].h;
148.                                    if ((x > xi) && (y > yi) && (x < xf) && (y < yf)) return c0;
149.                            }
150.                            return MSG_NULL;
151.                    }
152.//+------------------------------------------------------------------+
153.inline void DeleteObjectEdit(void)
154.                    {
155.                            ChartRedraw();
156.                            ObjectsDeleteAll(GetInfoTerminal().ID, m_Info.szObj_Editable);
157.                            m_Info.ConfigChartTrade.Msg = MSG_NULL;
158.                    }
159.//+------------------------------------------------------------------+
160.            template <typename T >
161.            void CreateObjectEditable(eObjectsIDE arg, T value)
162.                    {
163.                            long id = GetInfoTerminal().ID;
164.
165.                            DeleteObjectEdit();
166.                            CreateObjectGraphics(m_Info.szObj_Editable, OBJ_EDIT, clrBlack, 0);
167.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_XDISTANCE, m_Info.Regions[arg].x + m_Info.x + 3);
168.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_YDISTANCE, m_Info.Regions[arg].y + m_Info.y + 3);
169.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_XSIZE, m_Info.Regions[arg].w);
170.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_YSIZE, m_Info.Regions[arg].h);
171.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_BGCOLOR, m_Info.Regions[arg].bgcolor);
172.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_ALIGN, ALIGN_CENTER);
173.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_FONTSIZE, m_Info.Regions[arg].FontSize - 1);
174.                            ObjectSetString(id, m_Info.szObj_Editable, OBJPROP_FONT, m_Info.Regions[arg].FontName);
175.                            ObjectSetString(id, m_Info.szObj_Editable, OBJPROP_TEXT, (typename(T) == "double" ? DoubleToString(value, 2) : (string) value));
176.                            ChartRedraw();
177.                            m_Info.ConfigChartTrade.Msg = MSG_NULL;
178.                    }
179.//+------------------------------------------------------------------+
180.            bool RestoreState(void)
181.                    {
182.                            uCast_Double info;
183.                            bool bRet;
184.
185.                            if (bRet = GlobalVariableGet(macro_NameGlobalVariable("P"), info.dValue))
186.                            {
187.                                    m_Info.x = info._int[0];
188.                                    m_Info.y = info._int[1];
189.                            }
190.                            if (bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("M"), info.dValue) : bRet))
191.                            {
192.                                    m_Info.minx = info._int[0];
193.                                    m_Info.miny = info._int[1];
194.                            }
195.                            if (bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("B"), info.dValue) : bRet))
196.                            {
197.                                    m_Info.ConfigChartTrade.IsDayTrade = info._char[0];
198.                                    m_Info.IsMaximized = info._char[1];
199.                                    m_Info.ConfigChartTrade.Msg = (eObjectsIDE)info._char[2];
200.                            }
201.                            if (bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("L"), info.dValue) : bRet))
202.                                    m_Info.ConfigChartTrade.Leverage = info._int[0];
203.                            bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("Y"), m_Info.ConfigChartTrade.uCount.dValue) : bRet);
204.                            bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("T"), m_Info.FinanceTake) : bRet);
205.                            bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("S"), m_Info.FinanceStop) : bRet);
206.
207.
208.                            GlobalVariablesDeleteAll(macro_NameGlobalVariable(""));
209.
210.                            return bRet;
211.                    }
212.//+------------------------------------------------------------------+
213.    public  :
214.//+------------------------------------------------------------------+
215.            C_ChartFloatingRAD(const string szShortName)
216.                    :m_Mouse(NULL),
217.                     m_szShortName(szShortName)
218.                    {
219.                    }
220.//+------------------------------------------------------------------+
221.            C_ChartFloatingRAD(string szShortName, C_Mouse *MousePtr, const int Leverage, const double FinanceTake, const double FinanceStop)
222.                    :C_Terminal(),
223.                     m_szShortName(NULL)
224.                    {
225.                            if (!IndicatorCheckPass(szShortName)) SetUserError(C_Terminal::ERR_Unknown);
226.                            m_Mouse = MousePtr;
227.                            if (!RestoreState())
228.                            {
229.                                    m_Info.ConfigChartTrade.Leverage = Leverage;
230.                                    m_Info.FinanceTake = FinanceTake;
231.                                    m_Info.FinanceStop = FinanceStop;
232.                                    m_Info.ConfigChartTrade.IsDayTrade = true;
233.                                    m_Info.ConfigChartTrade.uCount.TickCount = 0;
234.                                    m_Info.ConfigChartTrade.Msg = MSG_NULL;
235.                                    m_Info.IsMaximized = true;
236.                                    m_Info.minx = m_Info.x = 115;
237.                                    m_Info.miny = m_Info.y = 64;
238.                            }
239.                            CreateWindowRAD(170, 210);
240.                            AdjustTemplate(true);
241.                    }
242.//+------------------------------------------------------------------+
243.            ~C_ChartFloatingRAD()
244.                    {
245.                            if (m_Mouse == NULL) return;
246.                            ChartRedraw();
247.                            ObjectsDeleteAll(GetInfoTerminal().ID, m_Info.szObj_Chart);
248.                            FileDelete(m_Info.szFileNameTemplate);
249.
250.                            delete m_Mouse;
251.                    }
252.//+------------------------------------------------------------------+
253.            void SaveState(void)
254.                    {
255.#define macro_GlobalVariable(A, B) if (GlobalVariableTemp(A)) GlobalVariableSet(A, B);
256.
257.                            uCast_Double info;
258.
259.                            if (m_Mouse == NULL) return;
260.                            info._int[0] = m_Info.x;
261.                            info._int[1] = m_Info.y;
262.                            macro_GlobalVariable(macro_NameGlobalVariable("P"), info.dValue);
263.                            info._int[0] = m_Info.minx;
264.                            info._int[1] = m_Info.miny;
265.                            macro_GlobalVariable(macro_NameGlobalVariable("M"), info.dValue);
266.                            info._char[0] = m_Info.ConfigChartTrade.IsDayTrade;
267.                            info._char[1] = m_Info.IsMaximized;
268.                            info._char[2] = (char)m_Info.ConfigChartTrade.Msg;
269.                            macro_GlobalVariable(macro_NameGlobalVariable("B"), info.dValue);
270.                            info._int[0] = m_Info.ConfigChartTrade.Leverage;
271.                            macro_GlobalVariable(macro_NameGlobalVariable("L"), info.dValue);
272.                            macro_GlobalVariable(macro_NameGlobalVariable("T"), m_Info.FinanceTake);
273.                            macro_GlobalVariable(macro_NameGlobalVariable("S"), m_Info.FinanceStop);
274.                            macro_GlobalVariable(macro_NameGlobalVariable("Y"), m_Info.ConfigChartTrade.uCount.dValue);
275.
276.#undef macro_GlobalVariable
277.                    }
278.//+------------------------------------------------------------------+
279.inline void MountBuffer(double &Buff[], const int iPos = -1)
280.                    {
281.                            static int posBuff = 0;
282.                            uCast_Double info;
283.
284.                            if ((m_szShortName != NULL) || (m_Info.ConfigChartTrade.Msg == MSG_NULL)) return;
285.                            posBuff = (iPos > 5 ? iPos - 5 : posBuff);
286.                            Buff[posBuff + 0] = m_Info.ConfigChartTrade.uCount.dValue;
287.                            info._char[0] = (char)m_Info.ConfigChartTrade.IsDayTrade;
288.                            info._char[1] = (char)m_Info.ConfigChartTrade.Msg;
289.                            Buff[posBuff + 1] = info.dValue;
290.                            info._int[0] = m_Info.ConfigChartTrade.Leverage;
291.                            Buff[posBuff + 2] = info.dValue;
292.                            Buff[posBuff + 3] = m_Info.ConfigChartTrade.PointsTake;
293.                            Buff[posBuff + 4] = m_Info.ConfigChartTrade.PointsStop;
294.                    }
295.//+------------------------------------------------------------------+
296.inline const stData GetDataBuffer(void)
297.                    {
298.                            double Buff[];
299.                            int handle;
300.                            uCast_Double info;
301.                            stData data;
302.
303.                            ZeroMemory(data);
304.                            if (m_szShortName == NULL) return data;
305.                            if ((handle = ChartIndicatorGet(ChartID(), 0, m_szShortName)) == INVALID_HANDLE) return data;
306.                            if (CopyBuffer(handle, 0, 0, 5, Buff) == 5)
307.                            {
308.                                    data.uCount.dValue = Buff[0];
309.                                    info.dValue = Buff[1];
310.                                    data.IsDayTrade = (bool)info._char[0];
311.                                    data.Msg = (C_ChartFloatingRAD::eObjectsIDE) info._char[1];
312.                                    info.dValue = Buff[2];
313.                                    data.Leverage = info._int[0];
314.                                    data.PointsTake = Buff[3];
315.                                    data.PointsStop = Buff[4];
316.                            }
317.                            if (handle != INVALID_HANDLE) IndicatorRelease(handle);
318.
319.                            return data;
320.                    };
321.//+------------------------------------------------------------------+
322.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
323.                    {
324.#define macro_AdjustMinX(A, B)  {                            \
325.             B = (A + m_Info.Regions[MSG_TITLE_IDE].w) > x;  \
326.             mx = x - m_Info.Regions[MSG_TITLE_IDE].w;       \
327.             A = (B ? (mx > 0 ? mx : 0) : A);                \
328.                                }
329.#define macro_AdjustMinY(A, B)  {                            \
330.             B = (A + m_Info.Regions[MSG_TITLE_IDE].h) > y;  \
331.             my = y - m_Info.Regions[MSG_TITLE_IDE].h;       \
332.             A = (B ? (my > 0 ? my : 0) : A);                \
333.                                }
334.
335.                            static int sx = -1, sy = -1;
336.                            int x, y, mx, my;
337.                            static eObjectsIDE obj = MSG_NULL;
338.                            double dvalue;
339.                            bool b1, b2, b3, b4;
340.                            eObjectsIDE tmp;
341.
342.                            if (m_szShortName == NULL) switch (id)
343.                            {
344.                                    case CHARTEVENT_CHART_CHANGE:
345.                                            x = (int)ChartGetInteger(GetInfoTerminal().ID, CHART_WIDTH_IN_PIXELS);
346.                                            y = (int)ChartGetInteger(GetInfoTerminal().ID, CHART_HEIGHT_IN_PIXELS);
347.                                            macro_AdjustMinX(m_Info.x, b1);
348.                                            macro_AdjustMinY(m_Info.y, b2);
349.                                            macro_AdjustMinX(m_Info.minx, b3);
350.                                            macro_AdjustMinY(m_Info.miny, b4);
351.                                            if (b1 || b2 || b3 || b4) AdjustTemplate();
352.                                            break;
353.                                    case CHARTEVENT_MOUSE_MOVE:
354.                                            if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft)) switch (tmp = CheckMousePosition(x = (int)lparam, y = (int)dparam))
355.                                            {
356.                                                    case MSG_TITLE_IDE:
357.                                                            if (sx < 0)
358.                                                            {
359.                                                                    DeleteObjectEdit();
360.                                                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
361.                                                                    sx = x - (m_Info.IsMaximized ? m_Info.x : m_Info.minx);
362.                                                                    sy = y - (m_Info.IsMaximized ? m_Info.y : m_Info.miny);
363.                                                            }
364.                                                            if ((mx = x - sx) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, mx);
365.                                                            if ((my = y - sy) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, my);
366.                                                            if (m_Info.IsMaximized)
367.                                                            {
368.                                                                    m_Info.x = (mx > 0 ? mx : m_Info.x);
369.                                                                    m_Info.y = (my > 0 ? my : m_Info.y);
370.                                                            }else
371.                                                            {
372.                                                                    m_Info.minx = (mx > 0 ? mx : m_Info.minx);
373.                                                                    m_Info.miny = (my > 0 ? my : m_Info.miny);
374.                                                            }
375.                                                            break;
376.                                                    case MSG_BUY_MARKET:
377.                                                    case MSG_SELL_MARKET:
378.                                                    case MSG_CLOSE_POSITION:
379.                                                            DeleteObjectEdit();
380.                                                            m_Info.ConfigChartTrade.Msg = tmp;
381.                                                            m_Info.ConfigChartTrade.uCount.TickCount = GetTickCount64();
382.                                                            break;
383.                                            }else if (sx > 0)
384.                                            {
385.                                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
386.                                                    sx = sy = -1;
387.                                            }
388.                                            break;
389.                                    case CHARTEVENT_OBJECT_ENDEDIT:
390.                                            switch (obj)
391.                                            {
392.                                                    case MSG_LEVERAGE_VALUE:
393.                                                    case MSG_TAKE_VALUE:
394.                                                    case MSG_STOP_VALUE:
395.                                                            dvalue = StringToDouble(ObjectGetString(GetInfoTerminal().ID, m_Info.szObj_Editable, OBJPROP_TEXT));
396.                                                            if (obj == MSG_TAKE_VALUE)
397.                                                                    m_Info.FinanceTake = (dvalue <= 0 ? m_Info.FinanceTake : dvalue);
398.                                                            else if (obj == MSG_STOP_VALUE)
399.                                                                    m_Info.FinanceStop = (dvalue <= 0 ? m_Info.FinanceStop : dvalue);
400.                                                            else
401.                                                                    m_Info.ConfigChartTrade.Leverage = (dvalue <= 0 ? m_Info.ConfigChartTrade.Leverage : (int)MathFloor(dvalue));
402.                                                            AdjustTemplate();
403.                                                            obj = MSG_NULL;
404.                                                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
405.                                                            break;
406.                                            }
407.                                            break;
408.                                    case CHARTEVENT_OBJECT_CLICK:
409.                                            if (sparam == m_Info.szObj_Chart) switch (obj = CheckMousePosition(x = (int)lparam, y = (int)dparam))
410.                                            {
411.                                                    case MSG_DAY_TRADE:
412.                                                            m_Info.ConfigChartTrade.IsDayTrade = (m_Info.ConfigChartTrade.IsDayTrade ? false : true);
413.                                                            DeleteObjectEdit();
414.                                                            break;
415.                                                    case MSG_MAX_MIN:
416.                                                            m_Info.IsMaximized = (m_Info.IsMaximized ? false : true);
417.                                                            DeleteObjectEdit();
418.                                                            break;
419.                                                    case MSG_LEVERAGE_VALUE:
420.                                                            CreateObjectEditable(obj, m_Info.ConfigChartTrade.Leverage);
421.                                                            break;
422.                                                    case MSG_TAKE_VALUE:
423.                                                            CreateObjectEditable(obj, m_Info.FinanceTake);
424.                                                            break;
425.                                                    case MSG_STOP_VALUE:
426.                                                            CreateObjectEditable(obj, m_Info.FinanceStop);
427.                                                            break;
428.                                            }
429.                                            if (obj != MSG_NULL) AdjustTemplate();
430.                                            break;
431.                            }
432.                    }
433.//+------------------------------------------------------------------+
434.};
435.//+------------------------------------------------------------------+
436.#undef macro_NameGlobalVariable
437.//+------------------------------------------------------------------+
```

C\_ChartFloatingRAD class source code

It may seem like I'm joking by publishing the source code in the article, but it's not a joke. Actually, I want to explain in as much detail as possible what is happening. This is not done so that you can actually use this code, but so that you can understand it and create something BETTER.

So let's figure out what's going on here. If you do not understand how this class works, you will not understand what will be developed next.

We have some changes right at the very beginning. On line 12 we declare a public clause. We use it to make the data to be accessible outside the class. You might think that we could place the same data elsewhere. Well, here I declare only one structure and one enumeration. The enumeration is already familiar to us, but the structure is new. Between lines 14 and 26 we find the declaration of a structure that is used to organize the required data.

Most of this structure is fairly easy to understand and requires little explanation. However, there are some things here that look quite strange. What is the purpose of this union between lines 20 and 24? And what is line 25 for? The union will allow us to transfer things more easily. Also, line 25 will be used at a very specific moment. We'll get to that a bit later.

The same structure is mentioned in line 38. Remember this: data present in a class cannot be directly accessed outside the class. So we need some means to access such data. We won't actually be addressing it directly. There is no such access as can be seen from the indicator's source code. Next we have line 49, which does something interesting. It will serve as a sort of selector, but that will become clearer later.

In line 106, we collect and adapt the first of the variables declared in the structure in line 38. Note that this is essentially the same thing we did before, but we are now working with a slightly different model.

So, in line 109 we adjust one of the variables. This is the variable of the number of points of the take profit value. Please pay attention to this, I do **NOT** I speak in terms of financial values, I speak in terms of **POINTS**. Don't confuse these two things. I also don't mean the number of points based on the current trading price. I'm talking about the points in general. The price doesn't matter, what matters is how many shift points we have. It is also important to note that this value in points will be then adjusted in financial terms. This adjustment is done in line 107.

In line 110, we have something similar, only this time we do it for the stop loss value. Just as was done in line 107 to adjust the financial values before taking into account the points in line 109. In line 108 we adjust the stop financials before taking into account the stop loss points. It is very important that you understand what is happening at this moment. If you cannot understand this, you will have difficulty understanding the adjustment that will be made later when placing orders.

To make your life a little easier, the attachment contains the Mouse and Chart Trade indicators and a fairly simple Expert Advisor so you can understand what's going on. For security reasons, the EA in the attachment will not place orders, but will only print the entered values. This can help you understand how the system actually works.

Now in lines 157 and 177, also for practical reasons, we adjust the value of one more variable. No access has been made so far. All we do is set and adjust the values of variables. But looking at line 203 – something seems strange. Why do we save this value in a global terminal variable? Do we really need to do this or will it be just a waste of time? In fact, it is necessary to do this. The reason is that when you delete the Chart Trade indicator and reload it on the chart, all values in memory are lost. But this value is very important to us.

So, here we restore the values that were previously saved. So, this same value that we restore in line 203 was actually saved in line 274. You may have noticed that I have skipped some functions in this explanation. So, let's get bask to them. Let's start with the constructor. Yes, now we have not one, but two class constructors. The reason for this is the same as in the earlier article in which we discussed the mouse indicator. We need a method to transfer data from the buffer.

To be honest, we don't really need it. We could do this directly in the EA or in the indicator. However, for convenience, I prefer to put everything together. This way, if I need to make any changes later, all I have to do is modify the C\_ChartFloatingRAD class. I won't have to configure each program individually to achieve standardization of modules. Getting back to constructors, we have an old constructor that starts at line 221. Basically, it gets the data from line 223, where we initialize the indicator name. **Note:** This name is not used for writing to the buffer, but for reading from the buffer. Additionally, we have lines 232 through 234 where we introduce new variables. Not all, as the rest are configured according to the template setup procedure.

The second constructor is quite simple and starts at line 215. There we mostly assign default values. This happens during the translation phase. As with the mouse indicator, there are also two phases here: one in which we write data to the buffer, and one in which we read the buffer. But this is not because the programming is going wrong. If you see here something wrong, then you might have not understood what we're doing. In fact, when we use the indicator, we write data to the buffer. This data is read via CopyBuffer by some other program, usually an EA. That's why we have two phases, and that's why we have two constructors.

Unlike constructors, we can only have one destructor, and it starts at line 243. The only thing we have added here is line 245. If we use this class in an indicator, when the destructor is called, the check on line 245 will pass, allowing the created objects to be deleted. But if the class is used to read the indicator buffer, the test in line 245 will fail and prevent any object from being deleted. Simple but functional mechanism.

That was the easy part. Now we come to the critical part of our explanation. Let's see how interaction occurs and how data is saved to and read from the buffer. This part may seem a little confusing for those just starting out. This is why it is so important to understand the concepts discussed in the previous articles. To make it a little easier, let's take a look at Figure 01.

![Figure 01](https://c.mql5.com/2/50/001__1.png)

Figure 01 – Interaction diagram

In Figure 01, we see a communication system for transmitting information from the indicator to the EA. Note that the buffer is not actually part of the indicator. Even though the buffer is declared in the indicator, it should not be considered as part of the indicator's memory. In fact, it is supported by MetaTrader 5, and when we remove the indicator from the chart, the latter frees up the memory that was used for the buffer. But herein lies one of the dangers. The data is not actually destroyed, just the memory is freed. What is the full danger of this. I will try to explain it in simple language.

If the memory where the buffer was located in MetaTrader 5 is freed because the indicator was removed from the chart, this may happen after writing to the buffer. If the indicator is placed back on the chart, it is possible that the buffer still contains some data. This may happen with the latest update. Therefore, when the EA reads this data using CopyBuffer, it may read it incorrectly.

The matter is so serious that we must be extremely careful when accessing the buffer. To provide some synchronization between writing and reading, we need to modify the DispatchMessage method in the C\_ChartFloatingRAD class. This synchronization between writing to the buffer and reading it is very important. If not done correctly, there will be a delay between what the user has submitted to be executed and what is actually executed. That is, a user can submit a buy order and it will not be executed. However, if you send a sell order immediately after this, a buy will be executed. These types of failures are not caused by the MetaTrader 5 platform, but rather by a misunderstanding of how to write code to ensure proper event synchronization.

In fact, there is no connection in terms of code between the EA and the Chart Trade indicator. This communication is implemented through the use of a buffer. Thus, the EA doesn't know what Chart Trade is doing, just like Chart Trade doesn't know what the EA is actually doing. However, MetaTrader 5 knows what both are doing. And since the common point between the indicator and the EA is MetaTrader 5, we use it to make things happen. It is like magic when you or the user gets the impression that the EA and the Chart Trade indicator are the same program.

There is another, equally important point. You should know that without the mouse indicator mentioned in previous articles, the Chart Trade indicator will **NOT** work. You need to have all three applications on the chart: the mouse indicator, the Chart Trade indicator and the EA, as well as other things that we will talk about in the future. Without this, the entire system will do absolutely nothing.

How does it all work? All this is guaranteed by the **CHARTEVENT\_MOUSE\_MOVE** event, which is located in line 353 of the C\_ChartFloatingRAD class. If the mouse indicator is not on the chart, you will be able to click, edit and modify the values of the Chart Trade indicator. This is because such events are not necessarily related to the mouse indicator. However, you will not be able to send buy, sell, close orders, or move the Chart Trade indicator unless the mouse indicator is placed on the chart.

But wait. Do I need the mouse indicator to be on the chart for interacting with Chart Trade? Yes. It is possible to remove this dependency, but it will lead to other problems later if you want to use the other methods I will talk about. These are the kind of costs you have to live with if you want to work like I do. This is why it is important that you have a good understanding of how the system works, otherwise you will end up with a system that you cannot trust.

But let's get back to the code to understand how I managed to synchronize the system. To fully understand this, you need to understand what the indicator's source code does, but if you've read the articles I linked to at the beginning of this article, you shouldn't have any trouble with this. So, let's assume that you already understand how the indicator code functions. This will allow me to focus on the class code.

Every time you click or move your mouse, MetaTrader 5 generates an event. Click events are typically handled using **CHARTEVENT\_OBJECT\_CLICK**. However, using this handler does not allow us to maintain synchronization. The reason for this is quite complex to explain; it has to do with the order in which operations are performed. Therefore, in order for an event to be generated in the ea at the same time when one of the three Chart Trade buttons (buy button, sell button and close position button) is clicked, we do it a little differently.

That's why if you compare this code present in the DispatchMessage method with the same code from the previous article, you can see that they are slightly different. The difference lies in the way clicks on the mentioned buttons are processed. In the previous version, this click was handled in the **CHARTEVENT\_OBJECT\_CLICK** event, and now we are processing it in **CHARTEVENT\_MOUSE\_MOVE**. Since the code itself tells us which object was clicked, I have created a new variable to make things more organized. This variable is declared in line 340 and its value is set in line 354.

Now pay close attention to what I am about to explain. In lines 376 to 378 we place the button codes. So, when the mouse indicator sends data to us, we can send a command to the EA to execute. But there is one small detail here. How can I inform the EA about which button was pressed? It's very easy. We send the button code to the EA. This is done in line 380. Now, in line 381, we record the number of ticks to generate a unique number. This will be necessary for the EA. You'll soon understand how exactly it works.

So for each event generated, in addition to this call to the DispatchMessage function that prepares the data to be sent to the buffer, we will have another call. The one that actually places the data into the buffer. You can see its code starting from line 279 in the class. Now let's look at line 281 which has a static variable. It will store the value passed by the OnCalculate event. That is, we need to save the **rates\_total** value, the reason for which has already been explained earlier. Read the articles listed at the beginning to understand why. This way, when the OnChartEvent handler is called, we know where to put the data in the buffer.

Notice that in line 284, we perform a check to ensure that only the Chart Trade indicator will write data to memory. In addition, recording into memory will only occur when one of the required buttons is pressed. All this is very good. However, none of this will be of any value if the EA cannot interpret the data sent to it by the Chart Trade indicator. To understand how the EA will be able to interpret the data, we need to consider some other the code, which will be as basic as possible.

### Using the Test Expert Advisor

Since everything needs to be tested, we need to make sure that the tests are done properly, clearly demonstrating what is actually happening. In this case, to test the interaction between the Chart Trade indicator, the mouse indicator and the EA, we need to use a very simple system. But at the same time, we need this system, in addition to being simple, to work in a way that we will actually use in the future.

Based on this criteria, we will use the following code:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Demo version between interaction of Chart Trade and EA"
04. #property version   "1.47"
05. #property link "https://www.mql5.com/es/articles/11760"
06. //+------------------------------------------------------------------+
07. #include <Market Replay\Chart Trader\C_ChartFloatingRAD.mqh>
08. //+------------------------------------------------------------------+
09. C_ChartFloatingRAD *chart = NULL;
10. //+------------------------------------------------------------------+
11. int OnInit()
12. {
13.     chart = new C_ChartFloatingRAD("Indicator Chart Trade");
14.
15.     return (CheckPointer(chart) != POINTER_INVALID ? INIT_SUCCEEDED : INIT_FAILED);
16. }
17. //+------------------------------------------------------------------+
18. void OnDeinit(const int reason)
19. {
20.     delete chart;
21. }
22. //+------------------------------------------------------------------+
23. void OnTick() {}
24. //+------------------------------------------------------------------+
25. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
26. {
27.     static ulong st_uTime = 0;
28.     C_ChartFloatingRAD::stData info;
29.
30.     switch (id)
31.     {
32.             case CHARTEVENT_OBJECT_CLICK:
33.                     info = (*chart).GetDataBuffer();
34.                     if (st_uTime != info.uCount.TickCount)
35.                     {
36.                             st_uTime = info.uCount.TickCount;
37.                             PrintFormat("%u -- %s [%s] %d : %f <> %f", info.uCount.TickCount, info.IsDayTrade ? "DT" : "SW", EnumToString(info.Msg), info.Leverage, info.PointsTake, info.PointsStop);
38.                     }else Print("IGNORADO...");
39.                     break;
40.     }
41. }
42. //+------------------------------------------------------------------+
```

Source code of the test Expert Advisor

Note that this code is very simple, compact, and mostly self-explanatory. However, some readers, especially beginners, may not be able to really understand how it works. Don't be upset about this. All beginners face similar difficulties. But if you try, study, invest time, stay disciplined and always strive to improve, in the future you will definitely become a great professional. Giving in to difficulties is unprofessional.

Now let's quickly look at the code, since it's only a few lines and should be easy to understand.

All the code that was presented in the previous topic and present in the C\_ChartFloatingRAD class is summarized when used here in the EA. Although we include the class completely in line 7, this is not exactly what the compiler thinks. So, in line 9 we declare the pointer globally. It will be used to access the C\_ChartFloatingRAD class. This may seem confusing. But only because you imagine that the class will be accessed in the same way as the indicator code.

In fact, it would be possible to do it this way, it's just not practical. The reason is that the class is not intended to be used without an indicator. The same goes for the C\_Mouse class, which is used in the mouse indicator and should not be used in code that is not an indicator. I know many people might be tempted to do this. But you should NOT do this. Because all the expected safety, modeling and performance are not intended for using classes in code other than the original one. That is, if you put the same coding method into the EA as you do in the indicator, you can actually achieve the point where there is no need to use the Chart Trade indicator. It is a fact.

However, if you transfer the indicator code to the EA, you may have problems with the stability and security of the entire system, since it is not designed to provide such stability. Everything has its place. So when we create the pointer in line 13 by calling the constructor, notice that I use the same name that was specified as the name of the Chart Trade indicator. This way both the indicator and the EA will do their jobs. Everyone is in their place. If something goes wrong in one of them, all you have to do is restart it on the chart.

Now notice that almost all the code boils down to exactly this. Specify the indicator name, and then in the OnChartEvent event handler, capture the **CHARTEVENT\_OBJECT\_CLICK** event and analyze what happened. There is one important point to make here. Every time you click the mouse, a click event will be generated on the object, even if you think you didn't click on anything important. The reason for this is that the mouse indicator must always be on the chart. This indicator has a horizontal line, which is the object. So whenever you click the mouse, this horizontal line will generate an event.

But how then will the system be able to distinguish a click on this line from a click on another object? This is quite an interesting question, which will soon be the subject of another article. However, this is something to keep in mind when using different objects on the chart, mainly because we will be adding other indicators soon. But this is a matter for the future.

The problem is that in line 33, the object click event will try to read the contents of the Chart Trade indicator buffer. If this succeeds, we will get the returned data. Then, in line 34, we'll check if this is a Chart Trade related event or something else that can be ignored.

If indeed an event was generated in Chart Trade, and it was caused by a click on one of the order system interaction buttons, we will update the value of the static variable as we did in line 36. After that, we will output a message to the terminal so that we can analyze what happened. If for some reason the event needs to be ignored, we will execute line 38.

### Conclusion

Video 01 shows how the system works in practice. However, nothing can replace the opportunity to see the system in action in person. Therefore, the attachment to the article contains the system in its current state.

Demonstração Parte 47 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11760)

MQL5.community

1.91K subscribers

[Demonstração Parte 47](https://www.youtube.com/watch?v=CFSj81SHuDU)

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

0:00 / 0:54

•Live

•

Video 01 – Demonstration

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11760](https://www.mql5.com/pt/articles/11760)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11760.zip "Download all attachments in the single ZIP archive")

[Anexo\_47.zip](https://www.mql5.com/en/articles/download/11760/anexo_47.zip "Download Anexo_47.zip")(175.7 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/474737)**

![Body in Connexus (Part 4): Adding HTTP body support](https://c.mql5.com/2/99/http60x60__4.png)[Body in Connexus (Part 4): Adding HTTP body support](https://www.mql5.com/en/articles/16098)

In this article, we explored the concept of body in HTTP requests, which is essential for sending data such as JSON and plain text. We discussed and explained how to use it correctly with the appropriate headers. We also introduced the ChttpBody class, part of the Connexus library, which will simplify working with the body of requests.

![Data Science and ML (Part 31): Using CatBoost AI Models for Trading](https://c.mql5.com/2/97/Data_Science_and_ML_Part_31___LOGO.png)[Data Science and ML (Part 31): Using CatBoost AI Models for Trading](https://www.mql5.com/en/articles/16017)

CatBoost AI models have gained massive popularity recently among machine learning communities due to their predictive accuracy, efficiency, and robustness to scattered and difficult datasets. In this article, we are going to discuss in detail how to implement these types of models in an attempt to beat the forex market.

![How to Create an Interactive MQL5 Dashboard/Panel Using the Controls Class (Part 1): Setting Up the Panel](https://c.mql5.com/2/97/How_to_Create_an_Interactive_MQL5_Dashboard___LOGO.png)[How to Create an Interactive MQL5 Dashboard/Panel Using the Controls Class (Part 1): Setting Up the Panel](https://www.mql5.com/en/articles/16084)

In this article, we create an interactive trading dashboard using the Controls class in MQL5, designed to streamline trading operations. The panel features a title, navigation buttons for Trade, Close, and Information, and specialized action buttons for executing trades and managing positions. By the end of the article, you will have a foundational panel ready for further enhancements in future installments.

![Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs (II)-LoRA-Tuning](https://c.mql5.com/2/97/Integrate_Your_Own_LLM_into_EA_Part_5___LOGO__1.png)[Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs (II)-LoRA-Tuning](https://www.mql5.com/en/articles/13499)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/11760&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070040560623881973)

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