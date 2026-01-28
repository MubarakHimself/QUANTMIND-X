---
title: Developing a Replay System (Part 45): Chart Trade Project (IV)
url: https://www.mql5.com/en/articles/11701
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:10:24.120387
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/11701&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070069732041756525)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 44): Chart Trade Project (III)](https://www.mql5.com/en/articles/11690), I showed how you can add some interactivity to the Chart Trade window so that it behaves as if there were objects in it. Even though the only real object represented on the chart was **OBJ\_CHART**.

But despite the existing interaction, which is quite pleasant, it cannot be called ideal. There are still some details that will finally be resolved in this article. What we end up with is some pretty interesting code, which will do what you can see in video 01 below:

Demonstração Parte 45 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11701)

MQL5.community

1.91K subscribers

[Demonstração Parte 45](https://www.youtube.com/watch?v=cqWcW4QleFI)

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

[Watch on](https://www.youtube.com/watch?v=cqWcW4QleFI&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11701)

0:00

0:00 / 3:31

•Live

•

Video 01. Demonstration of the capabilities of this version

This video demonstrates exactly what we will be able to do at this stage of development. Despite everything, we still won't have an order system. Not yet, because we still have a lot of things to create before the Chart Trade indicator can actually send orders or close positions.

### New Indicator

Despite the title of this topic, which makes it clear that we will be creating a new indicator, this is not exactly what we will be doing. We will add some elements that will make the Chart Trade indicator actually a new construction model. Below is the full source code of the indicator:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Base version for Chart Trade (DEMO version)"
04. #property version   "1.45"
05. #property icon "/Images/Market Replay/Icons/Indicators.ico"
06. #property link "https://www.mql5.com/en/articles/11701"
07. #property indicator_chart_window
08. #property indicator_plots 0
09. //+------------------------------------------------------------------+
10. #include <Market Replay\Chart Trader\C_ChartFloatingRAD.mqh>
11. //+------------------------------------------------------------------+
12. C_ChartFloatingRAD *chart = NULL;
13. //+------------------------------------------------------------------+
14. input int           user01 = 1;             //Leverage
15. input double        user02 = 100.1;         //Finance Take
16. input double        user03 = 75.4;          //Finance Stop
17. //+------------------------------------------------------------------+
18. #define macro_ERROR(A) if (_LastError != ERR_SUCCESS) { Print(__FILE__, " - [Error]: ", _LastError); if (A) ResetLastError(); }
19. //+------------------------------------------------------------------+
20. int OnInit()
21. {
22.     chart = new C_ChartFloatingRAD("Indicator Chart Trade", new C_Mouse("Indicator Mouse Study"), user01, user02, user03);
23.
24.     macro_ERROR(false);
25.
26.     return (_LastError == ERR_SUCCESS ? INIT_SUCCEEDED : INIT_FAILED);
27. }
28. //+------------------------------------------------------------------+
29. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
30. {
31.     return rates_total;
32. }
33. //+------------------------------------------------------------------+
34. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
35. {
36.     (*chart).DispatchMessage(id, lparam, dparam, sparam);
37.
38.     macro_ERROR(true);
39.
40.     ChartRedraw();
41. }
42. //+------------------------------------------------------------------+
43. void OnDeinit(const int reason)
44. {
45.     if (reason == REASON_CHARTCHANGE) (*chart).SaveState();
46.
47.     delete chart;
48. }
49. //+------------------------------------------------------------------+
```

**Chart Trade indicator source code**

As you can see, the code has hardly changed much since the last demonstration. But the changes that are taking place radically change the indicator's operating principle.

First, we've added a macro. It is in line 18 of the source code. This macro standardizes the error message displayed in the terminal. Now notice that it takes a parameter whose purpose is to specify whether the macro should reset the error constant. You can see this can in the points where the macro is used. The first point is in line 24, just after the attempt to initialize the indicator. In this case, we don't want or need to reset the constant, so the argument is false. The second point is in line 38. Here it may turn out that the error is acceptable to some extent, so the argument is true to reset the constant value. Therefore, it is important to monitor messages that appear in the terminal to follow what is happening.

There is another rather interesting point, which is in line 45. This is a security measure. A better understanding will be provided in the context of explaining the C\_ChartFloatingRAD class code. Basically, the reason is the need to somehow maintain the functionality of the Chart Trade indicator. Pay attention that I am using the chart update call. This event occurs whenever we change the chart timeframe. I would like to point out that, in addition to other things, our main problem is the timeframe change.

When you switch timeframes, all indicators are removed from the chart and then relaunched. At this point, data edited directly on the chart is lost. There are several ways to prevent this data loss. One of them is the one we are going to use. So, there is nothing more to say about the source code of the indicator. Since the C\_AdjustTemplate class has not undergone any changes, we can move on to explaining the code for the C\_ChartFloatingRAD class.

### **Making the C\_ChartFloatingRAD class almost fully functional**

The main purpose of this article is to introduce and explain the C\_ChartFloatingRAD class. As you may have seen in the video I presented at the beginning of the article, we have a Chart Trade indicator that works in a rather interesting way. As you may have noticed, we still have a fairly small number of objects on the chart, and yet we get the expected functionality. The values present in the indicator can be edited. The question is, how is this possible?

To answer this and other questions, we need to look at the source code of the class. This code is shown in full just below. Please note that there will be no attached files, but you will still be able to use the system as shown in the video. If you've been following this series of articles, you shouldn't have any problems.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "../Auxiliar/C_Mouse.mqh"
005. #include "../Auxiliar/Interprocess.mqh"
006. #include "C_AdjustTemplate.mqh"
007. //+------------------------------------------------------------------+
008. #define macro_NameGlobalVariable(A) StringFormat("ChartTrade_%u%s", GetInfoTerminal().ID, A)
009. //+------------------------------------------------------------------+
010. class C_ChartFloatingRAD : private C_Terminal
011. {
012.    private :
013.            enum eObjectsIDE {MSG_LEVERAGE_VALUE, MSG_TAKE_VALUE, MSG_STOP_VALUE, MSG_MAX_MIN, MSG_TITLE_IDE, MSG_DAY_TRADE, MSG_BUY_MARKET, MSG_SELL_MARKET, MSG_CLOSE_POSITION, MSG_NULL};
014.            struct st00
015.            {
016.                    int     x, y, minx, miny;
017.                    string  szObj_Chart,
018.                            szObj_Editable,
019.                            szFileNameTemplate;
020.                    long    WinHandle;
021.                    double  FinanceTake,
022.                            FinanceStop;
023.                    int     Leverage;
024.                    bool    IsDayTrade,
025.                            IsMaximized;
026.                    struct st01
027.                    {
028.                            int    x, y, w, h;
029.                            color  bgcolor;
030.                            int    FontSize;
031.                            string FontName;
032.                    }Regions[MSG_NULL];
033.            }m_Info;
034. //+------------------------------------------------------------------+
035.            C_Mouse *m_Mouse;
036. //+------------------------------------------------------------------+
037.            void CreateWindowRAD(int w, int h)
038.                    {
039.                            m_Info.szObj_Chart = "Chart Trade IDE";
040.                            m_Info.szObj_Editable = m_Info.szObj_Chart + " > Edit";
041.                            ObjectCreate(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJ_CHART, 0, 0, 0);
042.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x);
043.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y);
044.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XSIZE, w);
045.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, h);
046.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_DATE_SCALE, false);
047.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_PRICE_SCALE, false);
048.                            m_Info.WinHandle = ObjectGetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_CHART_ID);
049.                    };
050. //+------------------------------------------------------------------+
051.            void AdjustEditabled(C_AdjustTemplate &Template, bool bArg)
052.                    {
053.                            for (eObjectsIDE c0 = 0; c0 <= MSG_STOP_VALUE; c0++)
054.                                    if (bArg)
055.                                    {
056.                                            Template.Add(EnumToString(c0), "bgcolor", NULL);
057.                                            Template.Add(EnumToString(c0), "fontsz", NULL);
058.                                            Template.Add(EnumToString(c0), "fontnm", NULL);
059.                                    }
060.                                    else
061.                                    {
062.                                            m_Info.Regions[c0].bgcolor = (color) StringToInteger(Template.Get(EnumToString(c0), "bgcolor"));
063.                                            m_Info.Regions[c0].FontSize = (int) StringToInteger(Template.Get(EnumToString(c0), "fontsz"));
064.                                            m_Info.Regions[c0].FontName = Template.Get(EnumToString(c0), "fontnm");
065.                                    }
066.                    }
067. //+------------------------------------------------------------------+
068. inline void AdjustTemplate(const bool bFirst = false)
069.                    {
070. #define macro_AddAdjust(A) {                     \
071.              (*Template).Add(A, "size_x", NULL); \
072.              (*Template).Add(A, "size_y", NULL); \
073.              (*Template).Add(A, "pos_x", NULL);  \
074.              (*Template).Add(A, "pos_y", NULL);  \
075.                            }
076. #define macro_GetAdjust(A) {                                                                                                                                                                                                                                               \
077.              m_Info.Regions[A].x = (int) StringToInteger((*Template).Get(EnumToString(A), "pos_x"));  \
078.              m_Info.Regions[A].y = (int) StringToInteger((*Template).Get(EnumToString(A), "pos_y"));  \
079.              m_Info.Regions[A].w = (int) StringToInteger((*Template).Get(EnumToString(A), "size_x")); \
080.              m_Info.Regions[A].h = (int) StringToInteger((*Template).Get(EnumToString(A), "size_y")); \
081.                            }
082. #define macro_PointsToFinance(A) A * (GetInfoTerminal().VolumeMinimal + (GetInfoTerminal().VolumeMinimal * (m_Info.Leverage - 1))) * GetInfoTerminal().AdjustToTrade
083.
084.                            C_AdjustTemplate *Template;
085.
086.                            if (bFirst)
087.                            {
088.                                    Template = new C_AdjustTemplate("Chart Trade/IDE_RAD.tpl", m_Info.szFileNameTemplate = StringFormat("Chart Trade/%u.tpl", GetInfoTerminal().ID));
089.                                    for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++) macro_AddAdjust(EnumToString(c0));
090.                                    AdjustEditabled(Template, true);
091.                            }else Template = new C_AdjustTemplate(m_Info.szFileNameTemplate);
092.                            m_Info.Leverage = (m_Info.Leverage <= 0 ? 1 : m_Info.Leverage);
093.                            m_Info.FinanceTake = macro_PointsToFinance(FinanceToPoints(MathAbs(m_Info.FinanceTake), m_Info.Leverage));
094.                            m_Info.FinanceStop = macro_PointsToFinance(FinanceToPoints(MathAbs(m_Info.FinanceStop), m_Info.Leverage));
095.                            (*Template).Add("MSG_NAME_SYMBOL", "descr", GetInfoTerminal().szSymbol);
096.                            (*Template).Add("MSG_LEVERAGE_VALUE", "descr", (string)m_Info.Leverage);
097.                            (*Template).Add("MSG_TAKE_VALUE", "descr", (string)m_Info.FinanceTake);
098.                            (*Template).Add("MSG_STOP_VALUE", "descr", (string)m_Info.FinanceStop);
099.                            (*Template).Add("MSG_DAY_TRADE", "state", (m_Info.IsDayTrade ? "1" : "0"));
100.                            (*Template).Add("MSG_MAX_MIN", "state", (m_Info.IsMaximized ? "1" : "0"));
101.                            (*Template).Execute();
102.                            if (bFirst)
103.                            {
104.                                    for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++) macro_GetAdjust(c0);
105.                                    m_Info.Regions[MSG_TITLE_IDE].w = m_Info.Regions[MSG_MAX_MIN].x;
106.                                    AdjustEditabled(Template, false);
107.                            };
108.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, (m_Info.IsMaximized ? 210 : m_Info.Regions[MSG_TITLE_IDE].h + 6));
109.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, (m_Info.IsMaximized ? m_Info.x : m_Info.minx));
110.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, (m_Info.IsMaximized ? m_Info.y : m_Info.miny));
111.
112.                            delete Template;
113.
114.                            ChartApplyTemplate(m_Info.WinHandle, "/Files/" + m_Info.szFileNameTemplate);
115.                            ChartRedraw(m_Info.WinHandle);
116.
117. #undef macro_PointsToFinance
118. #undef macro_GetAdjust
119. #undef macro_AddAdjust
120.                    }
121. //+------------------------------------------------------------------+
122.            eObjectsIDE CheckMousePosition(const int x, const int y)
123.                    {
124.                            int xi, yi, xf, yf;
125.
126.                            for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++)
127.                            {
128.                                    xi = (m_Info.IsMaximized ? m_Info.x : m_Info.minx) + m_Info.Regions[c0].x;
129.                                    yi = (m_Info.IsMaximized ? m_Info.y : m_Info.miny) + m_Info.Regions[c0].y;
130.                                    xf = xi + m_Info.Regions[c0].w;
131.                                    yf = yi + m_Info.Regions[c0].h;
132.                                    if ((x > xi) && (y > yi) && (x < xf) && (y < yf)) return c0;
133.                            }
134.                            return MSG_NULL;
135.                    }
136. //+------------------------------------------------------------------+
137.            template <typename T >
138.            void CreateObjectEditable(eObjectsIDE arg, T value)
139.                    {
140.                            long id = GetInfoTerminal().ID;
141.                            ObjectDelete(id, m_Info.szObj_Editable);
142.                            CreateObjectGraphics(m_Info.szObj_Editable, OBJ_EDIT, clrBlack, 0);
143.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_XDISTANCE, m_Info.Regions[arg].x + m_Info.x + 3);
144.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_YDISTANCE, m_Info.Regions[arg].y + m_Info.y + 3);
145.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_XSIZE, m_Info.Regions[arg].w);
146.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_YSIZE, m_Info.Regions[arg].h);
147.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_BGCOLOR, m_Info.Regions[arg].bgcolor);
148.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_ALIGN, ALIGN_CENTER);
149.                            ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_FONTSIZE, m_Info.Regions[arg].FontSize - 1);
150.                            ObjectSetString(id, m_Info.szObj_Editable, OBJPROP_FONT, m_Info.Regions[arg].FontName);
151.                            ObjectSetString(id, m_Info.szObj_Editable, OBJPROP_TEXT, (string)value);
152.                            ChartRedraw();
153.                    }
154. //+------------------------------------------------------------------+
155.            bool RestoreState(void)
156.                    {
157.                            uCast_Double info;
158.                            bool bRet;
159.
160.                            if (bRet = GlobalVariableGet(macro_NameGlobalVariable("P"), info.dValue))
161.                            {
162.                                    m_Info.x = info._int[0];
163.                                    m_Info.y = info._int[1];
164.                            }
165.                            if (bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("M"), info.dValue) : bRet))
166.                            {
167.                                    m_Info.minx = info._int[0];
168.                                    m_Info.miny = info._int[1];
169.                            }
170.                            if (bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("B"), info.dValue) : bRet))
171.                            {
172.                                    m_Info.IsDayTrade = info._char[0];
173.                                    m_Info.IsMaximized = info._char[1];
174.                            }
175.                            if (bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("L"), info.dValue) : bRet))
176.                                    m_Info.Leverage = info._int[0];
177.                            bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("T"), m_Info.FinanceTake) : bRet);
178.                            bRet = (bRet ? GlobalVariableGet(macro_NameGlobalVariable("S"), m_Info.FinanceStop) : bRet);
179.
180.                            GlobalVariablesDeleteAll(macro_NameGlobalVariable(""));
181.
182.                            return bRet;
183.                    }
184. //+------------------------------------------------------------------+
185.    public  :
186. //+------------------------------------------------------------------+
187.            C_ChartFloatingRAD(string szShortName, C_Mouse *MousePtr, const int Leverage, const double FinanceTake, const double FinanceStop)
188.                    :C_Terminal()
189.                    {
190.                            if (!IndicatorCheckPass(szShortName)) SetUserError(C_Terminal::ERR_Unknown);
191.                            m_Mouse = MousePtr;
192.                            if (!RestoreState())
193.                            {
194.                                    m_Info.Leverage = Leverage;
195.                                    m_Info.FinanceTake = FinanceTake;
196.                                    m_Info.FinanceStop = FinanceStop;
197.                                    m_Info.IsDayTrade = true;
198.                                    m_Info.IsMaximized = true;
199.                                    m_Info.minx = m_Info.x = 115;
200.                                    m_Info.miny = m_Info.y = 64;
201.                            }
202.                            CreateWindowRAD(170, 210);
203.                            AdjustTemplate(true);
204.                    }
205. //+------------------------------------------------------------------+
206.            ~C_ChartFloatingRAD()
207.                    {
208.                            ObjectsDeleteAll(GetInfoTerminal().ID, m_Info.szObj_Chart);
209.                            FileDelete(m_Info.szFileNameTemplate);
210.
211.                            delete m_Mouse;
212.                    }
213. //+------------------------------------------------------------------+
214.            void SaveState(void)
215.                    {
216. #define macro_GlobalVariable(A, B) if (GlobalVariableTemp(A)) GlobalVariableSet(A, B);
217.
218.                            uCast_Double info;
219.
220.                            info._int[0] = m_Info.x;
221.                            info._int[1] = m_Info.y;
222.                            macro_GlobalVariable(macro_NameGlobalVariable("P"), info.dValue);
223.                            info._int[0] = m_Info.minx;
224.                            info._int[1] = m_Info.miny;
225.                            macro_GlobalVariable(macro_NameGlobalVariable("M"), info.dValue);
226.                            info._char[0] = m_Info.IsDayTrade;
227.                            info._char[1] = m_Info.IsMaximized;
228.                            macro_GlobalVariable(macro_NameGlobalVariable("B"), info.dValue);
229.                            info._int[0] = m_Info.Leverage;
230.                            macro_GlobalVariable(macro_NameGlobalVariable("L"), info.dValue);
231.                            macro_GlobalVariable(macro_NameGlobalVariable("T"), m_Info.FinanceTake);
232.                            macro_GlobalVariable(macro_NameGlobalVariable("S"), m_Info.FinanceStop);
233.
234. #undef macro_GlobalVariable
235.                    }
236. //+------------------------------------------------------------------+
237.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
238.                    {
239. #define macro_AdjustMinX(A, B) {                             \
240.              B = (A + m_Info.Regions[MSG_TITLE_IDE].w) > x;  \
241.              mx = x - m_Info.Regions[MSG_TITLE_IDE].w;       \
242.              A = (B ? (mx > 0 ? mx : 0) : A);                \
243.                                }
244. #define macro_AdjustMinY(A, B) {                             \
245.              B = (A + m_Info.Regions[MSG_TITLE_IDE].h) > y;  \
246.              my = y - m_Info.Regions[MSG_TITLE_IDE].h;       \
247.              A = (B ? (my > 0 ? my : 0) : A);                \
248.                                }
249.
250.                            static int sx = -1, sy = -1;
251.                            int x, y, mx, my;
252.                            static eObjectsIDE obj = MSG_NULL;
253.                            double dvalue;
254.                            bool b1, b2, b3, b4;
255.
256.                            switch (id)
257.                            {
258.                                    case CHARTEVENT_CHART_CHANGE:
259.                                            x = (int)ChartGetInteger(GetInfoTerminal().ID, CHART_WIDTH_IN_PIXELS);
260.                                            y = (int)ChartGetInteger(GetInfoTerminal().ID, CHART_HEIGHT_IN_PIXELS);
261.                                            macro_AdjustMinX(m_Info.x, b1);
262.                                            macro_AdjustMinY(m_Info.y, b2);
263.                                            macro_AdjustMinX(m_Info.minx, b3);
264.                                            macro_AdjustMinY(m_Info.miny, b4);
265.                                            if (b1 || b2 || b3 || b4) AdjustTemplate();
266.                                            break;
267.                                    case CHARTEVENT_MOUSE_MOVE:
268.                                            if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft)) switch (CheckMousePosition(x = (int)lparam, y = (int)dparam))
269.                                            {
270.                                                    case MSG_TITLE_IDE:
271.                                                            if (sx < 0)
272.                                                            {
273.                                                                    ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
274.                                                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
275.                                                                    sx = x - (m_Info.IsMaximized ? m_Info.x : m_Info.minx);
276.                                                                    sy = y - (m_Info.IsMaximized ? m_Info.y : m_Info.miny);
277.                                                            }
278.                                                            if ((mx = x - sx) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, mx);
279.                                                            if ((my = y - sy) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, my);
280.                                                            if (m_Info.IsMaximized)
281.                                                            {
282.                                                                    m_Info.x = (mx > 0 ? mx : m_Info.x);
283.                                                                    m_Info.y = (my > 0 ? my : m_Info.y);
284.                                                            }else
285.                                                            {
286.                                                                    m_Info.minx = (mx > 0 ? mx : m_Info.minx);
287.                                                                    m_Info.miny = (my > 0 ? my : m_Info.miny);
288.                                                            }
289.                                                            break;
290.                                            }else if (sx > 0)
291.                                            {
292.                                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
293.                                                    sx = sy = -1;
294.                                            }
295.                                            break;
296.                                    case CHARTEVENT_OBJECT_ENDEDIT:
297.                                            switch (obj)
298.                                            {
299.                                                    case MSG_LEVERAGE_VALUE:
300.                                                    case MSG_TAKE_VALUE:
301.                                                    case MSG_STOP_VALUE:
302.                                                            dvalue = StringToDouble(ObjectGetString(GetInfoTerminal().ID, m_Info.szObj_Editable, OBJPROP_TEXT));
303.                                                            if (obj == MSG_TAKE_VALUE)
304.                                                                    m_Info.FinanceTake = (dvalue <= 0 ? m_Info.FinanceTake : dvalue);
305.                                                            else if (obj == MSG_STOP_VALUE)
306.                                                                    m_Info.FinanceStop = (dvalue <= 0 ? m_Info.FinanceStop : dvalue);
307.                                                            else
308.                                                                    m_Info.Leverage = (dvalue <= 0 ? m_Info.Leverage : (int)MathFloor(dvalue));
309.                                                            AdjustTemplate();
310.                                                            obj = MSG_NULL;
311.                                                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
312.                                                            break;
313.                                            }
314.                                            break;
315.                                    case CHARTEVENT_OBJECT_CLICK:
316.                                            if (sparam == m_Info.szObj_Chart) switch (obj = CheckMousePosition(x = (int)lparam, y = (int)dparam))
317.                                            {
318.                                                    case MSG_DAY_TRADE:
319.                                                            m_Info.IsDayTrade = (m_Info.IsDayTrade ? false : true);
320.                                                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
321.                                                            break;
322.                                                    case MSG_MAX_MIN:
323.                                                            m_Info.IsMaximized = (m_Info.IsMaximized ? false : true);
324.                                                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
325.                                                            break;
326.                                                    case MSG_LEVERAGE_VALUE:
327.                                                            CreateObjectEditable(obj, m_Info.Leverage);
328.                                                            break;
329.                                                    case MSG_TAKE_VALUE:
330.                                                            CreateObjectEditable(obj, m_Info.FinanceTake);
331.                                                            break;
332.                                                    case MSG_STOP_VALUE:
333.                                                            CreateObjectEditable(obj, m_Info.FinanceStop);
334.                                                            break;
335.                                                    case MSG_BUY_MARKET:
336.                                                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
337.                                                            break;
338.                                                    case MSG_SELL_MARKET:
339.                                                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
340.                                                            break;
341.                                                    case MSG_CLOSE_POSITION:
342.                                                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
343.                                                            break;
344.                                            }
345.                                            if (obj != MSG_NULL) AdjustTemplate();
346.                                            break;
347.                            }
348.                    }
349. //+------------------------------------------------------------------+
350. };
351. //+------------------------------------------------------------------+
352. #undef macro_NameGlobalVariable
353. //+------------------------------------------------------------------+
354.
```

**C\_ChartFloatingRAD class source code**

While the code seems extensive, it's actually not that much if you've been following this series. The reason is that it undergoes gradual changes, which I try to introduce little by little, one after another. This allows me to demonstrate and explain code in a way that everyone can understand what's going on.

The first thing you'll notice in the class code is line 8, where we have the macro definition for creating the name. It will be used to define the name of global terminal variables. We'll look at how these variables will be used in more detail shortly. For now, you just need to know that we are defining a macro that will be used in the near future.

There are some minor changes in the code, but since there is no point in talking about them, let's focus on what is really important. Let's move to line 51 where we have a procedure that will be used quite often. It wasn't there before, but it's important to what we're going to do.

We will be using a **for** loop to reduce the number of lines and declarations a little. This is because they will be quite repetitive and the chances of making a mistake are very high. This loop includes the **if else** statement, which will allow us to both search and identify things.

When line 54 is executed, we can look up the information. In this case, lines 56 – 58 define what parameters we will look for in the template. This aspect has already been explained in the previous article. But now we will look for properties of objects that are defined in the template. Such properties are necessary so that the objects that will be created in the indicator work the same way as they were intended in the template.

In the second case, when we are not looking for information from the template, we will save such values locally. This will significantly speed up subsequent code execution. This data is saved in lines 62 and 64 and can be easily accessed later, as you will see during the explanation.

Next, look at line 68. This is where things get interesting because now we have other information that will be adjusted. Previously, everything that was displayed on the indicator happened at the moment the user placed the indicator on the chart, and no action was required. But now that we can interact directly with the indicator, we need to ensure that the data is somewhat representative.

For this reason, we have added a new macro, which can be seen on line 82. The very existence of this macro means that the old function no longer exists. Besides this macro, you may notice that the code has undergone some changes, including the appearance of line 90. The procedure described above is used here. Also note that we instruct the indicator to fix the parameters. This will be done only once, and exactly at the moment when the indicator starts. Detail: There is a second problem with this circumstance, and we will get back to it later in this article.

After that, in lines 92 to 94, we set up and adjust the values that will be displayed on the Chart Trade indicator. Previously this was done in the class constructor, but there was no user interaction. Now we have this interaction, so we need to ensure that the values are representative. So we configure the values here at the time of template update.

It is important to always remember: we do not edit the values directly in the objects in any way, since only **OBJ\_CHART** will be on our chart. So we need to make sure that the template is updated and displayed on the chart. For this, the update must occur exactly at this point.

The rest of the lines have already been explained in the previous article [Developing a Replay System (Part 44): Chart Trade Project (III)](https://www.mql5.com/en/articles/11690). Now, in line 105, we do something that wasn't there before. In this line we fix a small bug where it was possible to move the floating window by clicking and dragging the maximize/minimize button. This line removes it, and immediately after that, in line 106, we get the required values from the template. Please note that we now report the call as false. This way the values will be saved in the correct places for future use.

There's something interesting in line 108: we give the floating window the ability to maximize or minimize. Previously, this was done elsewhere in the code, but for practicality reasons I decided to place this control here. This makes the modeling process much easier since everything related to the window is in the same place.

Similarly, lines 109 and 110 allow us to work with the floating window more conveniently. Very often users want a floating window to change its position depending on its state. That is, it is in one position when maximized and in another one when minimized. And that's exactly what lines 109 and 110 do: they position the floating window at the last point it was in, depending on whether it was maximized or minimized.

Line 112 was already discussed in the previous article, so let's look at lines 114 and 115, which were previously in a separate function. So, in line 114 we run the already modified template in the **OBJ\_CHART** object. So, when line 115 is executed, we will get the template displayed and updated on the chart. The point is that now the entire procedure is concentrated in this subroutine. So we don't have to worry about communicating additional data to ensure that the information is presented to the user correctly.

It is possible, but unlikely for now, that this system could be placed in a separate class. Since it is only used in Chart Trade, I will leave it here. Implementing it in a class might be interesting because I decided to transform other things into a template. But for now I'll leave the code as it is.

Now we have something a little different. Line 137 contains a rather unusual type of code. This type of code is quite common when we have the same procedures but for different types. And what does this mean? 🤔 Well, first you need to understand this: why create one subroutine to display a double value, another for int values, and a third for strings? Wouldn't it be much easier to create one subroutine, since the code presented in them would basically always be the same? The only difference would be that in one case the value would be of one type, and in the other case it would be of another. This is exactly what line 137 does.

But wait, if the idea is to represent the value, couldn't we just pass it directly as a string? Yes, we could, but there is something we did not take into account. What if we need the value to be represented in a specific way and called from different points in the code? Think about the work that would have to be done. But this way we can just pass the value in its original type, let the compiler create a subroutine for us, and represent it the way we want. If we change the view, all we have to do is change just one point in the code. The compiler will take care of setting everything up.

**_Work less and produce more._**

Otherwise, the amount of work would only increase. My advice: whenever possible, make the compiler do the work for you. You'll notice that your codes will become much easier to maintain and your productivity will increase exponentially.

Line 137 is used in line 138. In no other place beyond this point do we use line 137. However, in line 151, we use the value passed as a parameter. Note that I do an explicit conversion to string type. We can perform this conversion during this step or after it, it will not make any difference. In this particular case.

Now, notice pay attention that here, in this procedure, we create an additional object **OBJ\_EDIT**. Why are we doing this? To make it easier to use the Chart Trade indicator. In fact, creating such an object is not necessary. But without it the indicator would be more difficult to use. The point is not that it is difficult to program the necessary logic, but that the user will have difficulties with the operation and use of the indicator. For this reason, we turn to MetaTrader 5 for help and ask it to create an editing object.

We need this object to be in the right place, with the right format and style. This is done as follows: when calling this procedure, we delete the created editing object, if it exists. This is done in line 141. But there is a nuance: this object will exist only when it is needed. So, in lines 142 to 150, we will use the values that are defined in the template. This way, the created object will be the same as the one used in the template.

There is a detail that is present in lines 143 and 144. This is a small adjustment where we add 3 to the dimensions. This meaning is not accidental, because **OBJ\_CHART** uses 3 pixels on the edges, and the **OBJ\_EDIT** object must be shifted by exactly these 3 pixels. This way it will be located exactly where the template is located on the chart.

Line 155 contains a function that will help us when changing the position of the indicator on the chart. **Attention:** This function does not work on its own, it works in combination with another one that we will see later. The function does the following: all sensitive indicator data is saved and then restored. Here we restore this data. There are several ways to do this, including the one used here. The fact that I do it this way is because I don't want to use DLLs unless I really have to. That's why I use global terminal variables so that MetaTrader 5 can help with the transaction.

Lines 160, 165, 170, 175, 177 and 178 will restore the data present in the terminal's global variables. Such variables are of type double, but we can store different values in them. I have already explained many times how this is done. But here we do it in a very specific way. So, if in any of these given lines the terminal global variable cannot be accessed or read, we will return a false value to the caller. The only point where this function will actually be called is the constructor, and we'll come back to that shortly.

On each call made in the specified lines, we restore the previously saved value. Thus, when changing the timeframe, you can continue working with the Chart Trade indicator as if there were no changes. There is one question that comes up here that is more personal than practical, but we'll talk about that when I explain how to store data.

Regardless of whether data could be read successfully or not, in line 180 we remove all global terminal variables that are linked to the Chart Trade indicator. However, please note that MetaTrader 5 may have more such variables. To know which ones to remove, we use a macro defined at the beginning of the class code.

Now let's move on to the class constructor. It starts at line 187. What needs to be explained is the interaction that occurs in line 192. It calls the procedure described above. If it fails, we execute lines 194 – 200, which will generate default values for the Chart Trade indicator. However, since the process of removing and reinstalling the indicator happens very quickly due to the change of timeframe, it is very unlikely that the preset values will be used. But it can happen, so it's helpful to be prepared for it.

Note that, unlike what happened before, the default values are now initialized without any adjustments. This is due to the fact that now such adjustments will be carried out by the procedure that updates the template.

Let's now see what happens in line 214. Here we temporarily save the state of Chart Trade. Why do we do so? Why do we save this state in global terminal variables? Is there no other way to do this? Let's look at it step by step.

First of all, yes, we could have done it differently. Basically, there are several possible ways. The question is not how to save, but how to restore saved data. The reason we use global terminal variables is because they are much easier to access. Considering that in some cases it would be more difficult to restore the data rather than save it, I seriously considered putting the data directly into the template. In fact, they would already be there, except for the data related to the position. Because we have one positioning for the maximized window and another for the minimized one.

The difference between the expanded and collapsed positions makes the template difficult to use. We could also use other methods, but in any case, this would unnecessarily complicate the system and would not be worth the effort. Let me repeat: the data is always already present in the template. However, when line 209 is executed, the template is deleted, causing the data to disappear. Even if you don't use different positions for the maximized and minimized window, you would have problems in relation to line 209.

One solution would be to place the template removal call in the indicator's source code. If we did this, the indicator code would look like this:

```
43. void OnDeinit(const int reason)
44. {
45.     if (reason != REASON_CHARTCHANGE) (*chart).RemoveTemplate();
46.
47.     delete chart;
48. }
```

This RemoveTemplate function would contain one call that corresponded to what is in line 209 of the class code. While this would work (relatively well), we would have other problems. One of them would be that if the indicator returned a more serious error, the corresponding file would not be deleted, but would remain on the disk. If you tried to place the indicator on the chart again, the data would be incorrect, which could lead to the indicator being removed again. This would continue until the defective file was deleted.

For these and other reasons, I prefer to use terminal global variables. But note that I don't use them directly. To use the variables, I use a macro. Why? The reason is the lifetime of the global terminal variable.

Look at what's happening in the macro that appears in line 216. Notice that we first try to create the variable as a temporary terminal global variable. This means that when you close the MetaTrader 5 terminal, the variable is destroyed along with it. This way we guarantee the integrity of the Chart Trade indicator.

Note that each global terminal variable will store one value. The order in which these variables are executed is not important, only the name and value really matter. We remember that the name cannot contain more than 64 characters. For this reason, we use a macro to create a name, which gives us a certain advantage in creating names.

There is nothing special to highlight in the procedure for saving template data. The fact is that without it, every time the timeframe changed, you would have to worry about reconfiguring the data present in the indicator. Considering that many users tend to change the timeframe several times during a trading period, it would be a big problem to constantly adjust the values of the Chart Trade indicator. By using programming and MetaTrader 5 capabilities, we can put that aside and focus on other things. To do this, we use the procedure in line 214.

There is another way to save data in "memory", but I will not go into details now, since it involves working with graphical objects. We'll talk about this another time.

We're almost done with this article. But first, we need to consider something else. It is the message handling function. It starts at line 237, and contrary to what it may seem, it is much simpler and friendlier than many people imagine. However, you may be wondering: why does this message handling function include 4 types of events if we are actually only going to use the mouse indicator?

I would like to emphasize once again that MetaTrader 5 is an event-based platform. Therefore, you need to understand how to work in this style. In the previous article, I mentioned that we could use other events to simplify our logic. Although the code is a bit confusing in some aspects, it is still functional. However, we can leave most of the checks discussed in the previous article that really should be present in this code. These checks will be performed by MetaTrader 5. So, if you compare both class codes, you will see that this one contains fewer checks. Why?

Because MetaTrader 5 will perform them. The object click events in Chart Trade are now replaced by a version where we analyze the object click, not just the mouse click. This makes coding much easier. This way we can include more events in more readable code. You can see the thing with the clicks by looking at the code in lines 315 to 343. In these lines, we handle clicks on all objects present in the template. All of them, even those that don't yet have any functions associated with them, such as the buy, sell, and close position buttons.

One of the things worth noting here in the message handler is the **CHARTEVENT\_CHART\_CHANGE** event, present in line 258. There is one detail when the terminal undergoes some changes in its dimensions. When this happens, MetaTrader 5 triggers an event, informing our programs about it. This event is handled via **CHARTEVENT\_CHART\_CHANGE**, so we can check if the floating window remains visible on the chart. If you do not handle this event, it may happen that the window remains hidden, but the indicator continues to be active. Since this handling is the same for both minimized and maximized mode, I use a macro to make the necessary adjustments. So, if any change occurs that requires the window position to change accordingly, line 265 will do it for us.

Another event that should also be mentioned is **CHARTEVENT\_OBJECT\_ENDEDIT**. In this case, whenever MetaTrader 5 detects that the **OBJ\_EDIT** object finished editing, it triggers an event. This way we can update data directly in the template. This is done in line 309. But please note: this update adjusts the data if necessary. If you try to enter a value or quantity that does not correspond to the asset, the code will adjust that value. In this way we avoid future problems.

### Conclusion

Despite all the complexity that Chart Trade can present in creating, compared to what we have seen so far, this version is significantly more stable and scalable than the old version. While it retains much of the concept presented previously, we tried to create a more modular system. Now, in addition to the real market and demo account, we have a system for simulating and replaying the market. This requires creating the system in a completely different way. If this is not done properly, we will have big problems working with the same instruments in such diverse systems and markets.

Although the Chart Trade indicator is not yet fully functional, since it has no functionality of the buy, sell and close position buttons, the core of the code is already correctly directed. We will come back to this indicator soon to get these buttons working. At the moment, the indicator already meets expectations.

I acknowledge that many may feel frustrated since there are no attachments to the article. But I have my reasons. I want you to see, read and understand the code and the system. I have noticed that many people do not actually read the articles and do not understand what they are using. This is dangerous and not the best way to use anything. Although it is not obvious, all the code is published and placed in the article, the task is only to understand it and edit it in MetaEditor. This ensures that the code won't be used by someone who doesn't know what it's about.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11701](https://www.mql5.com/pt/articles/11701)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11701.zip "Download all attachments in the single ZIP archive")

[Indicators.zip](https://www.mql5.com/en/articles/download/11701/indicators.zip "Download Indicators.zip")(149.46 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/472238)**
(1)


![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
29 Aug 2024 at 23:21

I loved the incorporation of a picture as the background on the chart. Thank you, @ [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831).


![Reimagining Classic Strategies (Part VII) : Forex Markets And Sovereign Debt Analysis on the USDJPY](https://c.mql5.com/2/91/Reimagining_Classic_Strategies_Part_VII___LOGO.png)[Reimagining Classic Strategies (Part VII) : Forex Markets And Sovereign Debt Analysis on the USDJPY](https://www.mql5.com/en/articles/15719)

In today's article, we will analyze the relationship between future exchange rates and government bonds. Bonds are among the most popular forms of fixed income securities and will be the focus of our discussion.Join us as we explore whether we can improve a classic strategy using AI.

![Implementing a Rapid-Fire Trading Strategy Algorithm with Parabolic SAR and Simple Moving Average (SMA) in MQL5](https://c.mql5.com/2/91/Implementing_a_Rapid_Fire_Trading_Strategy_Algorithm_with_Parabolic_SAR_and_Simple_Moving_Average___.png)[Implementing a Rapid-Fire Trading Strategy Algorithm with Parabolic SAR and Simple Moving Average (SMA) in MQL5](https://www.mql5.com/en/articles/15698)

In this article, we develop a Rapid-Fire Trading Expert Advisor in MQL5, leveraging the Parabolic SAR and Simple Moving Average (SMA) indicators to create a responsive trading strategy. We detail the strategy’s implementation, including indicator usage, signal generation, and the testing and optimization process.

![Example of Causality Network Analysis (CNA) and Vector Auto-Regression Model for Market Event Prediction](https://c.mql5.com/2/91/Vector_Auto-Regression_Model_for_Market_Event_Prediction___LOGO.png)[Example of Causality Network Analysis (CNA) and Vector Auto-Regression Model for Market Event Prediction](https://www.mql5.com/en/articles/15665)

This article presents a comprehensive guide to implementing a sophisticated trading system using Causality Network Analysis (CNA) and Vector Autoregression (VAR) in MQL5. It covers the theoretical background of these methods, provides detailed explanations of key functions in the trading algorithm, and includes example code for implementation.

![MQL5 Wizard Techniques you should know (Part 35): Support Vector Regression](https://c.mql5.com/2/91/MQL5_Wizard_Techniques_you_should_know_Part_35__LOGO.png)[MQL5 Wizard Techniques you should know (Part 35): Support Vector Regression](https://www.mql5.com/en/articles/15692)

Support Vector Regression is an idealistic way of finding a function or ‘hyper-plane’ that best describes the relationship between two sets of data. We attempt to exploit this in time series forecasting within custom classes of the MQL5 wizard.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/11701&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070069732041756525)

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