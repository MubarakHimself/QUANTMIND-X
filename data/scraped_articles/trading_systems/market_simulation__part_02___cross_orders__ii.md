---
title: Market Simulation (Part 02): Cross Orders (II)
url: https://www.mql5.com/en/articles/12537
categories: Trading Systems, Strategy Tester
relevance_score: 0
scraped_at: 2026-01-24T13:45:43.210001
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/12537&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083090908947158229)

MetaTrader 5 / Tester


### Introduction

In the previous article, [Market Simulation (Part 01): Cross Order (I)](https://www.mql5.com/en/articles/12536), I demonstrated and explained an alternative solution to a rather common problem, especially for those who trade futures contracts in some way. Although I did not present the final solution - since the entire article focused solely on the Chart Trade indicator - the content covered there is of utmost importance. It gives us the possibility for the C\_Terminal class to provide a proper name for trading these futures contracts.

However, the problem persists. Not because of the Chart Trade or the Expert Advisor, but because of the pending orders system. Although we have not yet started discussing it in these articles, we must prepare for it in some way. Part of the information used by this system comes from the Chart Trade, and all communication with the server occurs through the Expert Advisor. In other words, we have a triangle, where each vertex represents one of the applications to be developed. The edge that communicates with the server originates from the Expert Advisor vertex.

Thus, it is entirely reasonable to consider having the Expert Advisor control the choice of futures contract. This would allow us to decide whether to trade mini contracts or full contracts at a given time. However, this choice cannot be replicated across all applications. The decision on which type of contract to trade must occur in a single location to avoid ambiguities in both selection and information. Otherwise, the system would become highly confusing for the user if the contract type could be chosen in multiple places.

By making the selection in Chart Trade, as shown in the previous article, we can control the process. However, since the Expert Advisor is the only component that communicates directly with the server to send orders, it is also logical to consider placing the choice there. This introduces another type of problem, and demonstrating one way to solve it will be the focus of this article. But before you assume we are presenting a definitive solution here, I want to emphasize that what follows is a proposed solution. The final solution will be addressed later, as I have not yet decided on the ultimate approach.

### Understanding the Problems

Since the proposed solution will indeed use the Expert Advisor to select the type of contract, we need to modify some aspects of the Chart Trade. This is because a problem arises when the contract type selection is placed under the Expert Advisor's control. The problem lies in the information displayed in Chart Trade, as shown in the image below.

![Figure 1](https://c.mql5.com/2/147/001__1.png)

Notice that I am highlighting the asset or contract name. One of the key issues is that this name is used to calculate the financial values displayed in the Chart Trade interface. The problem becomes more significant because Chart Trade sends the values already converted into ticks to the Expert Advisor, leaving the Expert Advisor responsible only for sending market orders.

Now consider the challenge: if the Expert Advisor changes the contract type to a mini contract or a full contract, Chart Trade must replicate this information so the user or operator does not make a mistake. Replicating this information is not inherently difficult. We can achieve this by linking Chart Trade to the Expert Advisor. Unlike the current setup, where the user must add Chart Trade to the chart, the Expert Advisor would handle this. The user would place the Expert Advisor on the chart, and it would subsequently add the Chart Trade indicator, automatically adjusting the contract information.

This would be the simplest and most obvious solution. However, it would require placing the Chart Trade indicator in every Expert Advisor created - a completely unfeasible approach. Not because it cannot be done, but because any improvement to Chart Trade would require recompiling all Expert Advisors. This is why Chart Trade was designed to remain separate from the Expert Advisor.

So, we will keep the components separate. How, then, do we solve the problem in this case? The solution is already being suggested: use message exchanges between the Expert Advisor and the Chart Trade indicator. Simple, isn't it? In reality, it is not that simple. This is why I decided to explain the implementation in this article.

The first problem concerns the order in which the applications are added to the chart. How so? We could force the user to add the Expert Advisor before Chart Trade. Then, when Chart Trade is loaded, it would query the Expert Advisor for the contract type. Sounds good, but there is another problem at this very point. While we can instruct the user on the correct sequence, we cannot force MetaTrader 5 to comply. Perhaps you did not consider the following issue, which is problem number two: when MetaTrader 5 changes the chart timeframe, it destroys and reloads the chart. This applies to indicators and Expert Advisors. Now the question is: who loads first - the Chart Trade or the Expert Advisor? If the Expert Advisor loads first, problem one solves problem two. But what if Chart Trade loads first?

As you can see, the situation is more complicated than it appears. Additionally, there is a third problem: if the user changes the contract parameter in the Expert Advisor, or any other parameter, Chart Trade will not be correctly updated. All prior planning only addressed the first two problems. This issue occurs because the Expert Advisor would be removed and re-added to the chart by MetaTrader 5. Moreover, Chart Trade must somehow be aware of the Expert Advisor's presence. Otherwise, further problems arise.

Fortunately, the awareness issue can be temporarily set aside. The system is designed to provide feedback to the user about its status. Therefore, this awareness issue will become a real problem at another point. This is not a critical problem in the current context of Chart Trade and Expert Advisor interaction. So, to avoid large code changes, we will make certain concessions. The separation of components will be addressed in a new section.

### Making Some Concessions

The changes we implement here will be permanent, meaning the code will continue to evolve rather than regress. These concessions will allow the use of certain capabilities that were previously unavailable. Therefore, these new possibilities should be used carefully and attentively.

The first change can be seen in the file C\_Terminal.mqh. Below, you can find the full updated code to be used.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "Macros.mqh"
005. #include "..\Defines.mqh"
006. //+------------------------------------------------------------------+
007. class C_Terminal
008. {
009. //+------------------------------------------------------------------+
010.    protected:
011.       enum eErrUser {ERR_Unknown, ERR_FileAcess, ERR_PointerInvalid, ERR_NoMoreInstance};
012. //+------------------------------------------------------------------+
013.       struct st_Terminal
014.       {
015.          ENUM_SYMBOL_CHART_MODE   ChartMode;
016.          ENUM_ACCOUNT_MARGIN_MODE TypeAccount;
017.          long           ID;
018.          string         szSymbol;
019.          int            Width,
020.                         Height,
021.                         nDigits,
022.                         SubWin,
023.                         HeightBar;
024.          double         PointPerTick,
025.                         ValuePerPoint,
026.                         VolumeMinimal,
027.                         AdjustToTrade;
028.       };
029. //+------------------------------------------------------------------+
030.       void CurrentSymbol(bool bUsingFull)
031.          {
032.             MqlDateTime mdt1;
033.             string sz0, sz1;
034.             datetime dt = macroGetDate(TimeCurrent(mdt1));
035.             enum eTypeSymbol {WIN, IND, WDO, DOL, OTHER} eTS = OTHER;
036.
037.             sz0 = StringSubstr(m_Infos.szSymbol = _Symbol, 0, 3);
038.             for (eTypeSymbol c0 = 0; (c0 < OTHER) && (eTS == OTHER); c0++) eTS = (EnumToString(c0) == sz0 ? c0 : eTS);
039.             switch (eTS)
040.             {
041.                case DOL   :
042.                case WDO   : sz1 = "FGHJKMNQUVXZ"; break;
043.                case IND   :
044.                case WIN   : sz1 = "GJMQVZ";       break;
045.                default    : return;
046.             }
047.             sz0 = EnumToString((eTypeSymbol)(((eTS & 1) == 1) ? (bUsingFull ? eTS : eTS - 1) : (bUsingFull ? eTS + 1: eTS)));
048.             for (int i0 = 0, i1 = mdt1.year - 2000, imax = StringLen(sz1);; i0 = ((++i0) < imax ? i0 : 0), i1 += (i0 == 0 ? 1 : 0))
049.                if (dt < macroGetDate(SymbolInfoInteger(m_Infos.szSymbol = StringFormat("%s%s%d", sz0, StringSubstr(sz1, i0, 1), i1), SYMBOL_EXPIRATION_TIME))) break;
050.          }
051. //+------------------------------------------------------------------+
052.    private   :
053.       st_Terminal m_Infos;
054.       struct mem
055.       {
056.          long    Show_Descr,
057.                Show_Date;
058.          bool   AccountLock;
059.       }m_Mem;
060. //+------------------------------------------------------------------+
061. inline void ChartChange(void)
062.          {
063.             int x, y, t;
064.             m_Infos.Width  = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
065.             m_Infos.Height = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
066.             ChartTimePriceToXY(m_Infos.ID, 0, 0, 0, x, t);
067.             ChartTimePriceToXY(m_Infos.ID, 0, 0, m_Infos.PointPerTick * 100, x, y);
068.             m_Infos.HeightBar = (int)((t - y) / 100);
069.          }
070. //+------------------------------------------------------------------+
071.    public   :
072. //+------------------------------------------------------------------+
073.       C_Terminal(const long id = 0, const uchar sub = 0)
074.          {
075.             m_Infos.ID = (id == 0 ? ChartID() : id);
076.             m_Mem.AccountLock = false;
077.             m_Infos.SubWin = (int) sub;
078.             CurrentSymbol(false);
079.             m_Mem.Show_Descr = ChartGetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR);
080.             m_Mem.Show_Date  = ChartGetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE);
081.             ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, false);
082.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, true);
083.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, true);
084.             ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, false);
085.             m_Infos.nDigits = (int) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_DIGITS);
086.             m_Infos.Width   = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
087.             m_Infos.Height  = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
088.             m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
089.             m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
090.             m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
091.             m_Infos.AdjustToTrade = m_Infos.ValuePerPoint / m_Infos.PointPerTick;
092.             m_Infos.ChartMode   = (ENUM_SYMBOL_CHART_MODE) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_CHART_MODE);
093.             if(m_Infos.szSymbol != def_SymbolReplay) SetTypeAccount((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE));
094.             ChartChange();
095.          }
096. //+------------------------------------------------------------------+
097.       ~C_Terminal()
098.          {
099.             ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, m_Mem.Show_Date);
100.             ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, m_Mem.Show_Descr);
101.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, false);
102.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, false);
103.          }
104. //+------------------------------------------------------------------+
105. inline void SetTypeAccount(const ENUM_ACCOUNT_MARGIN_MODE arg)
106.          {
107.             if (m_Mem.AccountLock) return; else m_Mem.AccountLock = true;
108.             m_Infos.TypeAccount = (arg == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING ? arg : ACCOUNT_MARGIN_MODE_RETAIL_NETTING);
109.          }
110. //+------------------------------------------------------------------+
111. inline const st_Terminal GetInfoTerminal(void) const
112.          {
113.             return m_Infos;
114.          }
115. //+------------------------------------------------------------------+
116. const double AdjustPrice(const double arg) const
117.          {
118.             return NormalizeDouble(round(arg / m_Infos.PointPerTick) * m_Infos.PointPerTick, m_Infos.nDigits);
119.          }
120. //+------------------------------------------------------------------+
121. inline datetime AdjustTime(const datetime arg)
122.          {
123.             int nSeconds= PeriodSeconds();
124.             datetime   dt = iTime(m_Infos.szSymbol, PERIOD_CURRENT, 0);
125.
126.             return (dt < arg ? ((datetime)(arg / nSeconds) * nSeconds) : iTime(m_Infos.szSymbol, PERIOD_CURRENT, Bars(m_Infos.szSymbol, PERIOD_CURRENT, arg, dt)));
127.          }
128. //+------------------------------------------------------------------+
129. inline double FinanceToPoints(const double Finance, const uint Leverage)
130.          {
131.             double volume = m_Infos.VolumeMinimal + (m_Infos.VolumeMinimal * (Leverage - 1));
132.
133.             return AdjustPrice(MathAbs(((Finance / volume) / m_Infos.AdjustToTrade)));
134.          };
135. //+------------------------------------------------------------------+
136.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
137.          {
138.             static string st_str = "";
139.
140.             switch (id)
141.             {
142.                case CHARTEVENT_CHART_CHANGE:
143.                   m_Infos.Width  = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
144.                   m_Infos.Height = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
145.                   ChartChange();
146.                   break;
147.                case CHARTEVENT_OBJECT_CLICK:
148.                   if (st_str != sparam) ObjectSetInteger(m_Infos.ID, st_str, OBJPROP_SELECTED, false);
149.                   if (ObjectGetInteger(m_Infos.ID, sparam, OBJPROP_SELECTABLE) == true)
150.                      ObjectSetInteger(m_Infos.ID, st_str = sparam, OBJPROP_SELECTED, true);
151.                   break;
152.                case CHARTEVENT_OBJECT_CREATE:
153.                   if (st_str != sparam) ObjectSetInteger(m_Infos.ID, st_str, OBJPROP_SELECTED, false);
154.                   st_str = sparam;
155.                   break;
156.             }
157.          }
158. //+------------------------------------------------------------------+
159. inline void CreateObjectGraphics(const string szName, const ENUM_OBJECT obj, const color cor = clrNONE, const int zOrder = -1) const
160.          {
161.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, 0, false);
162.             ObjectCreate(m_Infos.ID, szName, obj, m_Infos.SubWin, 0, 0);
163.             ObjectSetString(m_Infos.ID, szName, OBJPROP_TOOLTIP, "\n");
164.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_BACK, false);
165.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_COLOR, cor);
166.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_SELECTABLE, false);
167.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_SELECTED, false);
168.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_ZORDER, zOrder);
169.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, 0, true);
170.          }
171. //+------------------------------------------------------------------+
172.       bool IndicatorCheckPass(const string szShortName)
173.          {
174.             string szTmp = szShortName + "_TMP";
175.
176.             IndicatorSetString(INDICATOR_SHORTNAME, szTmp);
177.             m_Infos.SubWin = ((m_Infos.SubWin = ChartWindowFind(m_Infos.ID, szTmp)) < 0 ? 0 : m_Infos.SubWin);
178.             if (ChartIndicatorGet(m_Infos.ID, m_Infos.SubWin, szShortName) != INVALID_HANDLE)
179.             {
180.                ChartIndicatorDelete(m_Infos.ID, 0, szTmp);
181.                Print("Only one instance is allowed...");
182.                SetUserError(C_Terminal::ERR_NoMoreInstance);
183.
184.                return false;
185.             }
186.             IndicatorSetString(INDICATOR_SHORTNAME, szShortName);
187.
188.             return true;
189.          }
190. //+------------------------------------------------------------------+
191. };
```

Source code of C\_Terminal.mqh

Note that the changes are minimal. In fact, compared to the code in the previous article, we have only opened the possibility of calling the CurrentSymbol procedure within other classes that inherit from the C\_Terminal class. Previously, this procedure was private to the class, but I have now changed it to protected. Many might make it public outright, but I prefer not to make such radical changes. I like to grant the minimum privilege necessary until the program demonstrates a real need for more access. As a protected procedure, it cannot be accessed indiscriminately.

In addition to this small change, another modification was made. Now, in the constructor, note that on line 73, I removed the parameter added in the previous article. However, on line 78, I forced the initial use of the mini contract. This kind of adjustment sets a precedent for further modifications. The reason for doing it this way is that if we had maintained the old structure, we would have been forced to close Chart Trade to update the contract displayed in the indicator. Now, this is no longer necessary. We can allow the contract to be switched via a message. However, as a consequence of this change, other modifications from the previous article remain relevant to the Chart Trade code, but we will not focus on them for now. Something else must be addressed first.

Speaking of messages, we need to implement new messages to cover the solution to these problems. But before examining the new messages, let’s pause for a moment and reflect. Look at the Figure below:

![Figure 2](https://c.mql5.com/2/147/002__1.png)

Here we see the points where message exchanges will actually occur. Notice one thing: in the case of the Expert Advisor, there are two points where messages will be sent to Chart Trade. For Chart Trade, there is only one point.

To understand how this exchange will take place, it is important to recognize that when something is added to the chart, after OnInit, OnChartEvent is executed. But do you know which event triggers OnChartEvent when an object is placed on the chart? If you investigate, you will find that when any object - indicator or Expert Advisor - is added to the chart, MetaTrader 5 triggers a CHARTEVENT\_CHART\_CHANGE event. This is crucial to understand because we will use this event to make the system function properly.

Before proceeding, consider this: why do we need buttons to send orders if the Expert Advisor that executes them may not even be on the chart? It does not make sense. To demonstrate that the message exchange has actually occurred, and that the information in Chart Trade can and should be used by the user or operator, we will modify a small but important detail in Chart Trade. With this, we now have the conceptual foundation needed to implement the solution to our problem.

### Implementing the Solution

The first step is to add three new events to our system. These can be seen in the Defines.mqh file, fully presented below:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #define def_VERSION_DEBUG
05. //+------------------------------------------------------------------+
06. #ifdef def_VERSION_DEBUG
07.    #define macro_DEBUG_MODE(A) \
08.                Print(__FILE__, " ", __LINE__, " ", __FUNCTION__ + " " + #A + " = " + (string)(A));
09. #else
10.    #define macro_DEBUG_MODE(A)
11. #endif
12. //+------------------------------------------------------------------+
13. #define def_SymbolReplay         "RePlay"
14. #define def_MaxPosSlider          400
15. #define def_MaskTimeService      0xFED00000
16. #define def_IndicatorTimeFrame   (_Period < 60 ? _Period : (_Period < PERIOD_D1 ? _Period - 16325 : (_Period == PERIOD_D1 ? 84 : (_Period == PERIOD_W1 ? 91 : 96))))
17. #define def_IndexTimeFrame         4
18. //+------------------------------------------------------------------+
19. union uCast_Double
20. {
21.    double    dValue;
22.    long      _long;                                 // 1 Information
23.    datetime _datetime;                              // 1 Information
24.    uint     _32b[sizeof(double) / sizeof(uint)];    // 2 Informations
25.    ushort   _16b[sizeof(double) / sizeof(ushort)];  // 4 Informations
26.    uchar    _8b [sizeof(double) / sizeof(uchar)];   // 8 Informations
27. };
28. //+------------------------------------------------------------------+
29. enum EnumEvents    {
30.          evHideMouse,                  //Hide mouse price line
31.          evShowMouse,                  //Show mouse price line
32.          evHideBarTime,                //Hide bar time
33.          evShowBarTime,                //Show bar time
34.          evHideDailyVar,               //Hide daily variation
35.          evShowDailyVar,               //Show daily variation
36.          evHidePriceVar,               //Hide instantaneous variation
37.          evShowPriceVar,               //Show instantaneous variation
38.          evCtrlReplayInit,             //Initialize replay control
39.          evChartTradeBuy,              //Market buy event
40.          evChartTradeSell,             //Market sales event
41.          evChartTradeCloseAll,         //Event to close positions
42.          evChartTrade_At_EA,           //Event to communication
43.          evEA_At_ChartTrade            //Event to communication
44.                   };
45. //+------------------------------------------------------------------+
```

Defines.mqh file source code

What has been added are lines 42 and 43. These lines are almost self-explanatory, as the key lies in the direction of communication. Line 42 concerns communication from Chart Trade to the Expert Advisor. Important: do not confuse this with trading events. This event is intended for a different type of communication - essentially a dedicated communication channel.

Line 43 specifies the name of the event that serves as the response to the request initiated by Chart Trade to the Expert Advisor. For now, that's all - just these two new lines. But their impact will be significant. Next, let's examine the full code of the Chart Trade indicator. It is shown just below. Note that it differs slightly from the version in the previous article. Specifically, the parameter that the user or trader could previously adjust is no longer present.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Chart Trade Base Indicator."
04. #property description "See the articles for more details."
05. #property version   "1.81"
06. #property icon "/Images/Market Replay/Icons/Indicators.ico"
07. #property link "https://www.mql5.com/en/articles/12537"
08. #property indicator_chart_window
09. #property indicator_plots 0
10. //+------------------------------------------------------------------+
11. #include <Market Replay\Chart Trader\C_ChartFloatingRAD.mqh>
12. //+------------------------------------------------------------------+
13. #define def_ShortName "Indicator Chart Trade"
14. //+------------------------------------------------------------------+
15. C_ChartFloatingRAD *chart = NULL;
16. //+------------------------------------------------------------------+
17. input ushort         user01 = 1;         //Leverage
18. input double         user02 = 100.1;     //Finance Take
19. input double         user03 = 75.4;      //Finance Stop
20. //+------------------------------------------------------------------+
21. int OnInit()
22. {
23.    chart = new C_ChartFloatingRAD(def_ShortName, new C_Mouse(0, "Indicator Mouse Study"), user01, user02, user03);
24.
25.    if (_LastError >= ERR_USER_ERROR_FIRST) return INIT_FAILED;
26.
27.    return INIT_SUCCEEDED;
28. }
29. //+------------------------------------------------------------------+
30. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
31. {
32.    return rates_total;
33. }
34. //+------------------------------------------------------------------+
35. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
36. {
37.    if (_LastError < ERR_USER_ERROR_FIRST)
38.       (*chart).DispatchMessage(id, lparam, dparam, sparam);
39. }
40. //+------------------------------------------------------------------+
41. void OnDeinit(const int reason)
42. {
43.    switch (reason)
44.    {
45.       case REASON_INITFAILED:
46.          ChartIndicatorDelete(ChartID(), 0, def_ShortName);
47.          break;
48.       case REASON_CHARTCHANGE:
49.          (*chart).SaveState();
50.          break;
51.    }
52.
53.    delete chart;
54. }
55. //+------------------------------------------------------------------+
```

Chart Trade indicator source code

Иut where are the new events? One would expect to see them in the OnChartEvent procedure. In fact, they are there, but to simplify things, everything is considered in one place, that is, in the DispatchMessage procedure. And this procedure is located in the C\_ChartFloatingRAD class, which can be seen in full below.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "../Auxiliar/C_Mouse.mqh"
005. #include "C_AdjustTemplate.mqh"
006. //+------------------------------------------------------------------+
007. #define macro_NameGlobalVariable(A) StringFormat("ChartTrade_%u%s", GetInfoTerminal().ID, A)
008. #define macro_CloseIndicator(A)   {           \
009.                OnDeinit(REASON_INITFAILED);   \
010.                SetUserError(A);               \
011.                return;                        \
012.                                  }
013. //+------------------------------------------------------------------+
014. class C_ChartFloatingRAD : private C_Terminal
015. {
016.    private   :
017.       enum eObjectsIDE {MSG_LEVERAGE_VALUE, MSG_TAKE_VALUE, MSG_STOP_VALUE, MSG_MAX_MIN, MSG_TITLE_IDE, MSG_DAY_TRADE, MSG_BUY_MARKET, MSG_SELL_MARKET, MSG_CLOSE_POSITION, MSG_NULL};
018.       struct st00
019.       {
020.          short    x, y, minx, miny,
021.                   Leverage;
022.          string   szObj_Chart,
023.                   szObj_Editable,
024.                   szFileNameTemplate;
025.          long     WinHandle;
026.          double   FinanceTake,
027.                   FinanceStop;
028.          bool     IsMaximized,
029.                   IsDayTrade,
030.                   IsSaveState;
031.          struct st01
032.          {
033.             short  x, y, w, h;
034.             color  bgcolor;
035.             int    FontSize;
036.             string FontName;
037.          }Regions[MSG_NULL];
038.       }m_Info;
039.       struct st01
040.       {
041.          short     y[2];
042.          bool      bOk;
043.       }m_Init;
044.       C_Mouse       *m_Mouse;
045. //+------------------------------------------------------------------+
046.       void CreateWindowRAD(int w, int h)
047.          {
048.             m_Info.szObj_Chart = "Chart Trade IDE";
049.             m_Info.szObj_Editable = m_Info.szObj_Chart + " > Edit";
050.             ObjectCreate(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJ_CHART, 0, 0, 0);
051.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x);
052.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y);
053.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XSIZE, w);
054.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, h);
055.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_DATE_SCALE, false);
056.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_PRICE_SCALE, false);
057.             m_Info.WinHandle = ObjectGetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_CHART_ID);
058.          };
059. //+------------------------------------------------------------------+
060.       void AdjustEditabled(C_AdjustTemplate &Template, bool bArg)
061.          {
062.             for (eObjectsIDE c0 = MSG_LEVERAGE_VALUE; c0 <= MSG_STOP_VALUE; c0++)
063.                if (bArg)
064.                {
065.                   Template.Add(EnumToString(c0), "bgcolor", NULL);
066.                   Template.Add(EnumToString(c0), "fontsz", NULL);
067.                   Template.Add(EnumToString(c0), "fontnm", NULL);
068.                }
069.                else
070.                {
071.                   m_Info.Regions[c0].bgcolor = (color) StringToInteger(Template.Get(EnumToString(c0), "bgcolor"));
072.                   m_Info.Regions[c0].FontSize = (int) StringToInteger(Template.Get(EnumToString(c0), "fontsz"));
073.                   m_Info.Regions[c0].FontName = Template.Get(EnumToString(c0), "fontnm");
074.                }
075.          }
076. //+------------------------------------------------------------------+
077. inline void AdjustTemplate(const bool bFirst = false)
078.          {
079. #define macro_PointsToFinance(A) A * (GetInfoTerminal().VolumeMinimal + (GetInfoTerminal().VolumeMinimal * (m_Info.Leverage - 1))) * GetInfoTerminal().AdjustToTrade
080.
081.             C_AdjustTemplate    *Template;
082.
083.             if (bFirst)
084.             {
085.                Template = new C_AdjustTemplate(m_Info.szFileNameTemplate = IntegerToString(GetInfoTerminal().ID) + ".tpl", true);
086.                for (eObjectsIDE c0 = MSG_LEVERAGE_VALUE; c0 <= MSG_CLOSE_POSITION; c0++)
087.                {
088.                   (*Template).Add(EnumToString(c0), "size_x", NULL);
089.                   (*Template).Add(EnumToString(c0), "size_y", NULL);
090.                   (*Template).Add(EnumToString(c0), "pos_x", NULL);
091.                   (*Template).Add(EnumToString(c0), "pos_y", NULL);
092.                }
093.                AdjustEditabled(Template, true);
094.             }else Template = new C_AdjustTemplate(m_Info.szFileNameTemplate);
095.             if (_LastError >= ERR_USER_ERROR_FIRST)
096.             {
097.                delete Template;
098.
099.                return;
100.             }
101.             m_Info.Leverage = (m_Info.Leverage <= 0 ? 1 : m_Info.Leverage);
102.             m_Info.FinanceTake = macro_PointsToFinance(FinanceToPoints(MathAbs(m_Info.FinanceTake), m_Info.Leverage));
103.             m_Info.FinanceStop = macro_PointsToFinance(FinanceToPoints(MathAbs(m_Info.FinanceStop), m_Info.Leverage));
104.             (*Template).Add("MSG_NAME_SYMBOL", "descr", GetInfoTerminal().szSymbol);
105.             (*Template).Add("MSG_LEVERAGE_VALUE", "descr", IntegerToString(m_Info.Leverage));
106.             (*Template).Add("MSG_TAKE_VALUE", "descr", DoubleToString(m_Info.FinanceTake, 2));
107.             (*Template).Add("MSG_STOP_VALUE", "descr", DoubleToString(m_Info.FinanceStop, 2));
108.             (*Template).Add("MSG_DAY_TRADE", "state", (m_Info.IsDayTrade ? "1" : "0"));
109.             (*Template).Add("MSG_MAX_MIN", "state", (m_Info.IsMaximized ? "1" : "0"));
110.             if (!(*Template).Execute())
111.             {
112.                delete Template;
113.
114.                macro_CloseIndicator(C_Terminal::ERR_FileAcess);
115.             };
116.             if (bFirst)
117.             {
118.                for (eObjectsIDE c0 = MSG_LEVERAGE_VALUE; c0 <= MSG_CLOSE_POSITION; c0++)
119.                {
120.                   m_Info.Regions[c0].x = (short) StringToInteger((*Template).Get(EnumToString(c0), "pos_x"));
121.                   m_Info.Regions[c0].y = (short) StringToInteger((*Template).Get(EnumToString(c0), "pos_y"));
122.                   m_Info.Regions[c0].w = (short) StringToInteger((*Template).Get(EnumToString(c0), "size_x"));
123.                   m_Info.Regions[c0].h = (short) StringToInteger((*Template).Get(EnumToString(c0), "size_y"));
124.                }
125.                m_Info.Regions[MSG_TITLE_IDE].w = m_Info.Regions[MSG_MAX_MIN].x;
126.                AdjustEditabled(Template, false);
127.             };
128.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, (m_Info.IsMaximized ? m_Init.y[m_Init.bOk] : m_Info.Regions[MSG_TITLE_IDE].h + 6));
129.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, (m_Info.IsMaximized ? m_Info.x : m_Info.minx));
130.             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, (m_Info.IsMaximized ? m_Info.y : m_Info.miny));
131.
132.             delete Template;
133.
134.             ChartApplyTemplate(m_Info.WinHandle, "/Files/" + m_Info.szFileNameTemplate);
135.             ChartRedraw(m_Info.WinHandle);
136.
137. #undef macro_PointsToFinance
138.          }
139. //+------------------------------------------------------------------+
140.       eObjectsIDE CheckMousePosition(const short x, const short y)
141.          {
142.             int xi, yi, xf, yf;
143.
144.             for (eObjectsIDE c0 = MSG_LEVERAGE_VALUE; c0 <= MSG_CLOSE_POSITION; c0++)
145.             {
146.                xi = (m_Info.IsMaximized ? m_Info.x : m_Info.minx) + m_Info.Regions[c0].x;
147.                yi = (m_Info.IsMaximized ? m_Info.y : m_Info.miny) + m_Info.Regions[c0].y;
148.                xf = xi + m_Info.Regions[c0].w;
149.                yf = yi + m_Info.Regions[c0].h;
150.                if ((x > xi) && (y > yi) && (x < xf) && (y < yf)) return c0;
151.             }
152.             return MSG_NULL;
153.          }
154. //+------------------------------------------------------------------+
155. inline void DeleteObjectEdit(void)
156.          {
157.             ChartRedraw();
158.             ObjectsDeleteAll(GetInfoTerminal().ID, m_Info.szObj_Editable);
159.          }
160. //+------------------------------------------------------------------+
161.       template <typename T >
162.       void CreateObjectEditable(eObjectsIDE arg, T value)
163.          {
164.             long id = GetInfoTerminal().ID;
165.
166.             DeleteObjectEdit();
167.             CreateObjectGraphics(m_Info.szObj_Editable, OBJ_EDIT, clrBlack, 0);
168.             ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_XDISTANCE, m_Info.Regions[arg].x + m_Info.x + 3);
169.             ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_YDISTANCE, m_Info.Regions[arg].y + m_Info.y + 3);
170.             ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_XSIZE, m_Info.Regions[arg].w);
171.             ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_YSIZE, m_Info.Regions[arg].h);
172.             ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_BGCOLOR, m_Info.Regions[arg].bgcolor);
173.             ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_ALIGN, ALIGN_CENTER);
174.             ObjectSetInteger(id, m_Info.szObj_Editable, OBJPROP_FONTSIZE, m_Info.Regions[arg].FontSize - 1);
175.             ObjectSetString(id, m_Info.szObj_Editable, OBJPROP_FONT, m_Info.Regions[arg].FontName);
176.             ObjectSetString(id, m_Info.szObj_Editable, OBJPROP_TEXT, (typename(T) == "double" ? DoubleToString(value, 2) : (string) value));
177.             ChartRedraw();
178.          }
179. //+------------------------------------------------------------------+
180.       bool RestoreState(void)
181.          {
182.             uCast_Double info;
183.             bool bRet;
184.             C_AdjustTemplate *Template;
185.
186.             if (bRet = GlobalVariableGet(macro_NameGlobalVariable("POST"), info.dValue))
187.             {
188.                m_Info.x = (short) info._16b[0];
189.                m_Info.y = (short) info._16b[1];
190.                m_Info.minx = (short) info._16b[2];
191.                m_Info.miny = (short) info._16b[3];
192.                Template = new C_AdjustTemplate(m_Info.szFileNameTemplate = IntegerToString(GetInfoTerminal().ID) + ".tpl");
193.                if (_LastError >= ERR_USER_ERROR_FIRST) bRet = false; else
194.                {
195.                   (*Template).Add("MSG_LEVERAGE_VALUE", "descr", NULL);
196.                   (*Template).Add("MSG_TAKE_VALUE", "descr", NULL);
197.                   (*Template).Add("MSG_STOP_VALUE", "descr", NULL);
198.                   (*Template).Add("MSG_DAY_TRADE", "state", NULL);
199.                   (*Template).Add("MSG_MAX_MIN", "state", NULL);
200.                   if (!(*Template).Execute()) bRet = false; else
201.                   {
202.                      m_Info.IsDayTrade = (bool) StringToInteger((*Template).Get("MSG_DAY_TRADE", "state")) == 1;
203.                      m_Info.IsMaximized = (bool) StringToInteger((*Template).Get("MSG_MAX_MIN", "state")) == 1;
204.                      m_Info.Leverage = (short)StringToInteger((*Template).Get("MSG_LEVERAGE_VALUE", "descr"));
205.                      m_Info.FinanceTake = (double) StringToDouble((*Template).Get("MSG_TAKE_VALUE", "descr"));
206.                      m_Info.FinanceStop = (double) StringToDouble((*Template).Get("MSG_STOP_VALUE", "descr"));
207.                   }
208.                };
209.                delete Template;
210.             };
211.
212.             GlobalVariablesDeleteAll(macro_NameGlobalVariable(""));
213.
214.             return bRet;
215.          }
216. //+------------------------------------------------------------------+
217.    public   :
218. //+------------------------------------------------------------------+
219.       C_ChartFloatingRAD(string szShortName, C_Mouse *MousePtr, const short Leverage, const double FinanceTake, const double FinanceStop)
220.          :C_Terminal(0)
221.          {
222.             m_Mouse = MousePtr;
223.             m_Info.IsSaveState = false;
224.             if (!IndicatorCheckPass(szShortName)) return;
225.             if (!RestoreState())
226.             {
227.                m_Info.Leverage = Leverage;
228.                m_Info.IsDayTrade = true;
229.                m_Info.FinanceTake = FinanceTake;
230.                m_Info.FinanceStop = FinanceStop;
231.                m_Info.IsMaximized = true;
232.                m_Info.minx = m_Info.x = 115;
233.                m_Info.miny = m_Info.y = 64;
234.             }
235.             m_Init.y[false] = 150;
236.             m_Init.y[true] = 210;
237.             CreateWindowRAD(170, m_Init.y[m_Init.bOk = false]);
238.             AdjustTemplate(true);
239.          }
240. //+------------------------------------------------------------------+
241.       ~C_ChartFloatingRAD()
242.          {
243.             ChartRedraw();
244.             ObjectsDeleteAll(GetInfoTerminal().ID, m_Info.szObj_Chart);
245.             if (!m_Info.IsSaveState)
246.                FileDelete(m_Info.szFileNameTemplate);
247.
248.             delete m_Mouse;
249.          }
250. //+------------------------------------------------------------------+
251.       void SaveState(void)
252.          {
253. #define macro_GlobalVariable(A, B) if (GlobalVariableTemp(A)) GlobalVariableSet(A, B);
254.
255.             uCast_Double info;
256.
257.             info._16b[0] = m_Info.x;
258.             info._16b[1] = m_Info.y;
259.             info._16b[2] = m_Info.minx;
260.             info._16b[3] = m_Info.miny;
261.             macro_GlobalVariable(macro_NameGlobalVariable("POST"), info.dValue);
262.             m_Info.IsSaveState = true;
263.
264. #undef macro_GlobalVariable
265.          }
266. //+------------------------------------------------------------------+
267.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
268.          {
269. #define macro_AdjustMinX(A, B)    {                          \
270.             B = (A + m_Info.Regions[MSG_TITLE_IDE].w) > x;   \
271.             mx = x - m_Info.Regions[MSG_TITLE_IDE].w;        \
272.             A = (B ? (mx > 0 ? mx : 0) : A);                 \
273.                                  }
274. #define macro_AdjustMinY(A, B)   {                           \
275.             B = (A + m_Info.Regions[MSG_TITLE_IDE].h) > y;   \
276.             my = y - m_Info.Regions[MSG_TITLE_IDE].h;        \
277.             A = (B ? (my > 0 ? my : 0) : A);                 \
278.                                  }
279.
280.             static short sx = -1, sy = -1, sz = -1;
281.             static eObjectsIDE obj = MSG_NULL;
282.             short   x, y, mx, my;
283.             double dvalue;
284.             bool b1, b2, b3, b4;
285.             ushort ev = evChartTradeCloseAll;
286.
287.             switch (id)
288.             {
289.                case CHARTEVENT_CUSTOM + evEA_At_ChartTrade:
290.                   if (m_Init.bOk = ((lparam >= 0) && (lparam < 2)))
291.                      CurrentSymbol((bool)lparam);
292.                   AdjustTemplate(true);
293.                   break;
294.                case CHARTEVENT_CHART_CHANGE:
295.                   if (!m_Init.bOk)
296.                      EventChartCustom(GetInfoTerminal().ID, evChartTrade_At_EA, 0, 0, "");
297.                   x = (short)ChartGetInteger(GetInfoTerminal().ID, CHART_WIDTH_IN_PIXELS);
298.                   y = (short)ChartGetInteger(GetInfoTerminal().ID, CHART_HEIGHT_IN_PIXELS);
299.                   macro_AdjustMinX(m_Info.x, b1);
300.                   macro_AdjustMinY(m_Info.y, b2);
301.                   macro_AdjustMinX(m_Info.minx, b3);
302.                   macro_AdjustMinY(m_Info.miny, b4);
303.                   if (b1 || b2 || b3 || b4) AdjustTemplate();
304.                   break;
305.                case CHARTEVENT_MOUSE_MOVE:
306.                   if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft))
307.                   {
308.                      switch (CheckMousePosition(x = (short)lparam, y = (short)dparam))
309.                      {
310.                         case MSG_MAX_MIN:
311.                            if (sz < 0) m_Info.IsMaximized = (m_Info.IsMaximized ? false : true);
312.                            break;
313.                         case MSG_DAY_TRADE:
314.                            if ((m_Info.IsMaximized) && (sz < 0)) m_Info.IsDayTrade = (m_Info.IsDayTrade ? false : true);
315.                            break;
316.                         case MSG_LEVERAGE_VALUE:
317.                            if ((m_Info.IsMaximized) && (sz < 0)) CreateObjectEditable(obj = MSG_LEVERAGE_VALUE, m_Info.Leverage);
318.                            break;
319.                         case MSG_TAKE_VALUE:
320.                            if ((m_Info.IsMaximized) && (sz < 0)) CreateObjectEditable(obj = MSG_TAKE_VALUE, m_Info.FinanceTake);
321.                            break;
322.                         case MSG_STOP_VALUE:
323.                            if ((m_Info.IsMaximized) && (sz < 0)) CreateObjectEditable(obj = MSG_STOP_VALUE, m_Info.FinanceStop);
324.                            break;
325.                         case MSG_TITLE_IDE:
326.                            if (sx < 0)
327.                            {
328.                               ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
329.                               sx = x - (m_Info.IsMaximized ? m_Info.x : m_Info.minx);
330.                               sy = y - (m_Info.IsMaximized ? m_Info.y : m_Info.miny);
331.                            }
332.                            if ((mx = x - sx) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, mx);
333.                            if ((my = y - sy) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, my);
334.                            if (m_Info.IsMaximized)
335.                            {
336.                               m_Info.x = (mx > 0 ? mx : m_Info.x);
337.                               m_Info.y = (my > 0 ? my : m_Info.y);
338.                            }else
339.                            {
340.                               m_Info.minx = (mx > 0 ? mx : m_Info.minx);
341.                               m_Info.miny = (my > 0 ? my : m_Info.miny);
342.                            }
343.                            break;
344.                         case MSG_BUY_MARKET:
345.                            ev = evChartTradeBuy;
346.                         case MSG_SELL_MARKET:
347.                            ev = (ev != evChartTradeBuy ? evChartTradeSell : ev);
348.                         case MSG_CLOSE_POSITION:
349.                            if ((m_Info.IsMaximized) && (sz < 0) && (m_Init.bOk)) //<<
350.                            {
351.                               string szTmp = StringFormat("%d?%s?%s?%c?%d?%.2f?%.2f", ev, _Symbol, GetInfoTerminal().szSymbol, (m_Info.IsDayTrade ? 'D' : 'S'),
352.                                                           m_Info.Leverage, FinanceToPoints(m_Info.FinanceTake, m_Info.Leverage), FinanceToPoints(m_Info.FinanceStop, m_Info.Leverage));
353.                               PrintFormat("Send %s - Args ( %s )", EnumToString((EnumEvents) ev), szTmp);
354.                               EventChartCustom(GetInfoTerminal().ID, ev, 0, 0, szTmp);
355.                            }
356.                            break;
357.                      }
358.                      if (sz < 0)
359.                      {
360.                         sz = x;
361.                         AdjustTemplate();
362.                         if (obj == MSG_NULL) DeleteObjectEdit();
363.                      }
364.                   }else
365.                   {
366.                      sz = -1;
367.                      if (sx > 0)
368.                      {
369.                         ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
370.                         sx = sy = -1;
371.                      }
372.                   }
373.                   break;
374.                case CHARTEVENT_OBJECT_ENDEDIT:
375.                   switch (obj)
376.                   {
377.                      case MSG_LEVERAGE_VALUE:
378.                      case MSG_TAKE_VALUE:
379.                      case MSG_STOP_VALUE:
380.                         dvalue = StringToDouble(ObjectGetString(GetInfoTerminal().ID, m_Info.szObj_Editable, OBJPROP_TEXT));
381.                         if (obj == MSG_TAKE_VALUE)
382.                            m_Info.FinanceTake = (dvalue <= 0 ? m_Info.FinanceTake : dvalue);
383.                         else if (obj == MSG_STOP_VALUE)
384.                            m_Info.FinanceStop = (dvalue <= 0 ? m_Info.FinanceStop : dvalue);
385.                         else
386.                            m_Info.Leverage = (dvalue <= 0 ? m_Info.Leverage : (short)MathFloor(dvalue));
387.                         AdjustTemplate();
388.                         obj = MSG_NULL;
389.                         ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
390.                         break;
391.                   }
392.                   break;
393.                case CHARTEVENT_OBJECT_DELETE:
394.                   if (sparam == m_Info.szObj_Chart) macro_CloseIndicator(C_Terminal::ERR_Unknown);
395.                   break;
396.             }
397.             ChartRedraw();
398.          }
399. //+------------------------------------------------------------------+
400. };
401. //+------------------------------------------------------------------+
402. #undef macro_NameGlobalVariable
403. #undef macro_CloseIndicator
404. //+------------------------------------------------------------------+
```

Source code of the file C\_ChartFloatingRAD.mqh

Although the code in the header file C\_ChartFloatingRAD.mqh may appear overwhelming to many, and it is not strictly necessary to view it in full, I decided to include it in its entirety in this article. This is due to the changes made compared to the previous article. However, the only significant change is in the class constructor. Since line numbers have shifted, I did not want to leave you, the reader, confused while trying to locate the correct lines. Hence, the full code is provided.

Let's focus on what has actually changed. Not in comparison to the previous article, but rather to support what will be seen in the video at the end of this article. The first change is a small structure appearing on line 39. Pay close attention to my explanation, or you will not understand what is shown in the video. Within this structure, there is line 41. It contains two values representing the window size, which is actually an OBJ\_CHART object. Don't worry - you will understand this shortly. Line 42 contains a variable that, together with this two-element array, enables the functionality seen in the video.

Next, let's examine the class constructor, starting on line 219. Notice that the extra parameter from the previous article is no longer present. Line 220 calls the constructor of the C\_Terminal class, returning to the state prior to the last article. Now, pay attention: the Chart Trade window is an OBJ\_CHART object, which can be minimized or maximized. This functionality was already implemented. However, when the Expert Advisor is not present on the chart where Chart Trade resides, we want the buttons to be hidden.

To achieve this, we must adjust the Y-dimension of the coordinate system. These dimensions are set on lines 235 and 236. Pay very close attention: if the variable m\_Init.bOk is false, it indicates that something is not aligned. In this case, interaction buttons are hidden to prevent the user or operator from mistakenly thinking orders are being sent to the server. For the buttons to be hidden, the Y-coordinate is set to 150. When m\_Init.bOk is true, the user or operator can send orders via the Expert Advisor. So ,the Y-coordinate is set to 210. This value of 210 was the default used previously, as seen on line 237.

Notice that line 237, which creates the OBJ\_CHART object to contain Chart Trade, has changed. Instead of using the fixed value 210 for the dimension, we now access the value defined in the array. At the same time, m\_Init.bOk is initialized as false. Essentially, the OBJ\_CHART object is initially created with a Y-coordinate of 150. Why not just use this value directly? The reason is to test and confirm that the code works as intended. Passing 150 directly would not provide the same assurance that the model is functioning correctly. This approach ensures the implementation behaves as expected.

A new detail appears in the AdjustTemplate procedure, called shortly afterward, with its code starting at line 77. The only change in this procedure is on line 128. The reason is as follows: when Chart Trade is maximized or minimized, we change the Y-coordinate. To ensure Chart Trade remains consistent with the rules established in the constructor, we must adjust the Y-coordinate accordingly. Thus, even if you try to maximize or minimize Chart Trade to access the buttons, they will not be visible until the conditions for display are satisfied.

This was an easy part. The changes are few and simple. Now let's examine the messaging functionality, which also has a few minor modifications. The DispatchMessage procedure begins on line 267. Before reviewing the messages, note line 349, where a small but important difference was introduced. To simplify the mouse-checking function, a new test value was added on this line. If m\_Init.bOk is false and the user clicks in the region where the buttons would appear, the check prevents any event from being triggered. This level of simplicity is what makes programming elegant. Many developers would otherwise try to implement this check elsewhere, complicating the procedure unnecessarily.

Now, let's return to the messaging system. When the indicator is added to the chart, the first event received is **CHARTEVENT\_CHART\_CHANGE**. On line 295, we check whether our variable is false. If so, a custom event is triggered on line 296. Regardless of whether the Expert Advisor captured the triggered event or was just added to the chart, another event will occur, which is captured by Chart Trade on line 289.

Now the fun part begins. When the value tested on line 290 is greater than or equal to zero, line 291 is executed. Pay attention: if the value is zero, it represents false; if nonzero, it represents true. However, values greater than one are invalid. Therefore, the Expert Advisor should only send a value of zero or one, indicating whether a mini contract or a full contract is in use. Finally, line 292 requests an update of Chart Trade. This may sound complex, especially because a value other than zero or one might arrive. In that case, the buttons are hidden. This indicates that the Expert Advisor has been removed from the chart or an unexpected event has occurred.

At this point, you may think I am being overly cautious. How could Chart Trade know that the Expert Advisor was removed or that something unusual happened? To understand this, we need to examine the Expert Advisor itself. This is presented next.

### How the Expert Advisor Functions Now

The Expert Advisor code for demonstration purposes can be seen in full below.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Virtual Test..."
04. #property description "Demo version between interaction"
05. #property description "of Chart Trade and Expert Advisor"
06. #property version   "1.81"
07. #property link "https://www.mql5.com/en/articles/12537"
08. //+------------------------------------------------------------------+
09. #include <Market Replay\Defines.mqh>
10. //+------------------------------------------------------------------+
11. class C_Decode
12. {
13.    private   :
14.       struct stInfoEvent
15.       {
16.          EnumEvents  ev;
17.          string      szSymbol,
18.                      szContract;
19.          bool        IsDayTrade;
20.          ushort      Leverange;
21.          double      PointsTake,
22.                      PointsStop;
23.       }info[1];
24.    public   :
25. //+------------------------------------------------------------------+
26.       C_Decode()
27.          {
28.             info[0].szSymbol = _Symbol;
29.          }
30. //+------------------------------------------------------------------+
31.       bool Decode(const int id, const string sparam)
32.       {
33.          string Res[];
34.
35.          if (StringSplit(sparam, '?', Res) != 7) return false;
36.          stInfoEvent loc = {(EnumEvents) StringToInteger(Res[0]), Res[1], Res[2], (bool)(Res[3] == "D"), (ushort) StringToInteger(Res[4]), StringToDouble(Res[5]), StringToDouble(Res[6])};
37.          if ((id == loc.ev) && (loc.szSymbol == info[0].szSymbol)) info[0] = loc;
38.
39.          ArrayPrint(info, 2);
40.
41.          return true;
42.       }
43. }*GL_Decode;
44. //+------------------------------------------------------------------+
45. enum eTypeContract {MINI, FULL};
46. //+------------------------------------------------------------------+
47. input eTypeContract user00 = MINI;       //Cross order in contract
48. //+------------------------------------------------------------------+
49. bool bOk;
50. //+------------------------------------------------------------------+
51. int OnInit()
52. {
53.    bOk = false;
54.    GL_Decode = new C_Decode;
55.
56.    return INIT_SUCCEEDED;
57. }
58. //+------------------------------------------------------------------+
59. void OnTick() {}
60. //+------------------------------------------------------------------+
61. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
62. {
63.    switch (id)
64.    {
65.       case CHARTEVENT_CUSTOM + evChartTradeBuy     :
66.       case CHARTEVENT_CUSTOM + evChartTradeSell    :
67.       case CHARTEVENT_CUSTOM + evChartTradeCloseAll:
68.          GL_Decode.Decode(id - CHARTEVENT_CUSTOM, sparam);
69.          break;
70.       case CHARTEVENT_CHART_CHANGE:
71.          if (bOk)   break;
72.       case CHARTEVENT_CUSTOM + evChartTrade_At_EA:
73.          bOk = true;
74.          EventChartCustom(ChartID(), evEA_At_ChartTrade, user00, 0, "");
75.          break;
76.    }
77. }
78. //+------------------------------------------------------------------+
79. void OnDeinit(const int reason)
80. {
81.    switch (reason)
82.    {
83.       case REASON_REMOVE:
84.       case REASON_INITFAILED:
85.          EventChartCustom(ChartID(), evEA_At_ChartTrade, -1, 0, "");
86.          break;
87.    }
88.    delete GL_Decode;
89. }
90. //+------------------------------------------------------------------+
```

EA source code

Now comes the question: how does this code function? Before answering, let's note a small change that occurred in the previous article, when Chart Trade began indicating which contract it was displaying. If you look at the Chart Trade code - more specifically, the C\_ChartFloatingRAD class - you'll see that on line 351, a new piece of information was added. This information concerns the contract name and is decoded on line 36 of the Expert Advisor code. This is a minor detail, but it is important to mention. Now, let's examine how the Expert Advisor code works.

Notice that the code allowing the user or operator to select the contract type, previously located in Chart Trade, is now on lines 45 and 47 of the Expert Advisor. However, nowhere in this Expert Advisor do we access the C\_Terminal class. Why? Because we are not yet connected to the server. The Expert Advisor is still in demonstration mode. In any case, observe line 49, which declares a variable used to prevent any chart changes from triggering an event to Chart Trade unnecessarily. On line 53, this variable is initialized as false. On line 73, it is set to true. And on line 71, it is checked to avoid sending unnecessary messages.

Pay attention: like the indicator, the Expert Advisor's first event is a CHARTEVENT\_CHART\_CHANGE, declared on line 70. The variable is false. Therefore, the same process that would occur if Chart Trade requested something is executed. This brings us to line 72. As a result, on line 74, an event is triggered, with a value of either zero or one. This is why the enumeration on line 45 must match the order shown. Changing the order will produce incorrect results in Chart Trade. Only modify these values if you understand the consequences. Otherwise, Chart Trade will display the wrong contract.

The second part involves the Expert Advisor signaling to Chart Trade that it is no longer available. This requires Chart Trade to hide the order buttons. This occurs when the routine on line 79 is executed. Under normal circumstances, MetaTrader 5 triggers a DeInit event to remove an object from the chart. The variable reason contains the cause of this call. On line 81, we check this reason. If it matches the specified criteria, an event is triggered on line 85. Notice that the value following the event type is -1. This tells Chart Trade that the Expert Advisor is no longer available, causing the order controls to be hidden automatically.

The rest of the code remains unchanged from what was explained in the article on Chart Trade communication. Thus, I consider this stage complete.

### Final Thoughts

Although I have shown how to easily implement communication to initialize both the Expert Advisor and the Chart Trade indicator, I am not certain at this stage whether the code presented in this article will actually be used. This is due to the pending orders system, which will prove to be our "thorn in the side". Up to this point, the user or operator can send orders to the server using either a mini contract or a full contract, simply by changing one system parameter. This part was relatively straightforward. The necessary steps were outlined in previous articles of this series. However, unlike simple inter-application messaging, managing pending order information complicates the process significantly.

In any case, the results of the changes described so far can be seen in the video. Additionally, the executable files used in the video are attached below, allowing you to review and understand exactly what has been implemented.

Narrado 81 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12537)

MQL5.community

1.91K subscribers

[Narrado 81](https://www.youtube.com/watch?v=mwI2buLOvCo)

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

0:00 / 2:23

•Live

•

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12537](https://www.mql5.com/pt/articles/12537)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12537.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12537/anexo.zip "Download Anexo.zip")(490.53 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/496951)**

![Post-Factum trading analysis: Selecting trailing stops and new stop levels in the strategy tester](https://c.mql5.com/2/115/Post-hoc_trading_analysis___LOGO3.png)[Post-Factum trading analysis: Selecting trailing stops and new stop levels in the strategy tester](https://www.mql5.com/en/articles/16991)

We continue the topic of analyzing completed deals in the strategy tester to improve the quality of trading. Let's see how using different trailing stops can change our existing trading results.

![MetaTrader 5 Machine Learning Blueprint (Part 3): Trend-Scanning Labeling Method](https://c.mql5.com/2/172/19253-metatrader-5-machine-learning-logo.png)[MetaTrader 5 Machine Learning Blueprint (Part 3): Trend-Scanning Labeling Method](https://www.mql5.com/en/articles/19253)

We have built a robust feature engineering pipeline using proper tick-based bars to eliminate data leakage and solved the critical problem of labeling with meta-labeled triple-barrier signals. This installment covers the advanced labeling technique, trend-scanning, for adaptive horizons. After covering the theory, an example shows how trend-scanning labels can be used with meta-labeling to improve on the classic moving average crossover strategy.

![From Novice to Expert: Demystifying Hidden Fibonacci Retracement Levels](https://c.mql5.com/2/173/19780-from-novice-to-expert-demystifying-logo.png)[From Novice to Expert: Demystifying Hidden Fibonacci Retracement Levels](https://www.mql5.com/en/articles/19780)

In this article, we explore a data-driven approach to discovering and validating non-standard Fibonacci retracement levels that markets may respect. We present a complete workflow tailored for implementation in MQL5, beginning with data collection and bar or swing detection, and extending through clustering, statistical hypothesis testing, backtesting, and integration into an MetaTrader 5 Fibonacci tool. The goal is to create a reproducible pipeline that transforms anecdotal observations into statistically defensible trading signals.

![MQL5 Wizard Techniques you should know (Part 81):  Using Patterns of Ichimoku and the ADX-Wilder with Beta VAE Inference Learning](https://c.mql5.com/2/173/19781-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 81): Using Patterns of Ichimoku and the ADX-Wilder with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19781)

This piece follows up ‘Part-80’, where we examined the pairing of Ichimoku and the ADX under a Reinforcement Learning framework. We now shift focus to Inference Learning. Ichimoku and ADX are complimentary as already covered, however we are going to revisit the conclusions of the last article related to pipeline use. For our inference learning, we are using the Beta algorithm of a Variational Auto Encoder. We also stick with the implementation of a custom signal class designed for integration with the MQL5 Wizard.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ihlhdbyerzgnofeclkvmvixxrvlzdesz&ssn=1769251541830748103&ssn_dr=0&ssn_sr=0&fv_date=1769251541&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12537&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Market%20Simulation%20(Part%2002)%3A%20Cross%20Orders%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925154141283007&fz_uniq=5083090908947158229&sv=2552)

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