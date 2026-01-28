---
title: Developing a Replay System (Part 59): A New Future
url: https://www.mql5.com/en/articles/12075
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:41:27.800215
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/12075&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069660022226487461)

MetaTrader 5 / Examples


### Introduction

In the previous article " [Developing a Replay System (Part 58): Returning to Work on the Service](https://www.mql5.com/en/articles/12039)", I mentioned that the system has undergone some changes, and that there is a reason to include a small delay between applying a template to a schedule and updating the schedule by the service. If this does not happen, the modules will force the service to close prematurely. Some of you may have been left wondering how this could happen. In this article, we will explore this in more detail.

To properly explain this and ensure you fully understand the concept, please watch Video 01 below.

Demonstração 59 1 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12075)

MQL5.community

1.91K subscribers

[Demonstração 59 1](https://www.youtube.com/watch?v=qwixkhRqNNE)

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

[Watch on](https://www.youtube.com/watch?v=qwixkhRqNNE&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12075)

0:00

0:00 / 2:41

•Live

•

Demonstration Video: How Things Can Fail

This video provides an unedited, unaltered demonstration of what actually happens. But why does this issue occur? To understand the explanation, it's essential to revisit the key points from the previous article. I will briefly summarize them here.

### Understanding Why the Service Shuts Down

There is a specific reason why the service shuts down prematurely as soon as MetaTrader 5 applies the template. The explanation is simple: the indicator - or more precisely, the control module - is causing the chart to close. Still unclear? Let's examine the relevant section of the control indicator's code, which is responsible for this behavior. You can find the code snippet below.

```
53. void OnDeinit(const int reason)
54. {
55.    switch (reason)
56.    {
57.       case REASON_TEMPLATE:
58.          Print("Modified template. Replay // simulation system shutting down.");
59.       case REASON_INITFAILED:
60.       case REASON_PARAMETERS:
61.       case REASON_REMOVE:
62.       case REASON_CHARTCLOSE:
63.          ChartClose(user00);
64.          break;
65.    }
66.    delete control;
67. }
```

Control module code fragment

Observe line 63, where there is a call instructing MetaTrader 5 to close the chart. But who is actually making this call? If it were triggered solely by the template change, the message in line 58 would appear in the MetaTrader 5 message box. However, as you may have noticed, this message is not printed. This suggests that the chart closure is not directly caused by the template change.

I understand why you might question this and even doubt the explanation. However, the truth is that applying the template in MetaTrader 5 does, in fact, remove the control indicator from the chart, which ultimately causes the chart to close. The key point is that this process occurs asynchronously, meaning events do not unfold in the exact sequence you might expect.

What actually happens is that at some point, MetaTrader 5 completes the application of the template. When this occurs, MetaTrader 5 sends a Deinit event to the indicator, which is handled by the OnDeinit function. However, contrary to what you might expect, the reason variable does not hold the value REASON\_TEMPLATE but rather REASON\_REMOVE. This is because MetaTrader 5 is actively removing the control indicator from the chart as part of the template application process. That is why the message from line 58 is never printed in the terminal's message window.

A natural question arises: Why doesn't MetaTrader 5 apply templates synchronously? In other words, why doesn't executing the ChartApplyTemplate function, found in line 87 of the C\_Replay.mqh header file, immediately and fully apply the template to the chart? The answer lies in performance. A template may contain multiple indicators or even an Expert Advisor that requires a certain number of bars to be present on the chart before providing useful data.

If MetaTrader 5 were to wait for ChartApplyTemplate to complete synchronously, the platform could freeze for several seconds, or even crash if something in the template caused a critical failure. By handling this process asynchronously, MetaTrader 5 improves performance but also introduces challenges for less experienced programmers. One such challenge is understanding which functions execute asynchronously and which execute immediately.

Since forcing an immediate chart update using ChartRedraw does not ensure that the template is applied instantly, we need an alternative approach. The demonstration video illustrates the implemented solution.

You might ask: why not resolve this issue by including the control indicator directly in the template? While this would prevent the premature closure problem, it would create another issue. In previous versions, we used a global terminal variable to prevent the control indicator from appearing on unintended charts. However, we no longer rely on this method. Instead, we now use a different blocking mechanism. This means we cannot allow the template to include the control indicator, as doing so would introduce complications that are difficult to resolve.

As mentioned in the previous article, both the control and mouse modules have undergone changes. This was a result of the adjustments made when we refocused on the replay/simulation service. Now, let's take a look at the updated code for both indicators, focusing on the key modifications.

### Updated Code for the Replay/Simulation Service

Below, you will find the latest version of the control module's source code.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property description "Control indicator for the Replay-Simulator service."
05. #property description "This one doesn't work without the service loaded."
06. #property version   "1.59"
07. #property link "https://www.mql5.com/en/articles/12075"
08. #property indicator_chart_window
09. #property indicator_plots 0
10. #property indicator_buffers 1
11. //+------------------------------------------------------------------+
12. #include <Market Replay\Service Graphics\C_Controls.mqh>
13. //+------------------------------------------------------------------+
14. C_Controls *control = NULL;
15. //+------------------------------------------------------------------+
16. input long user00 = 0;      //ID
17. //+------------------------------------------------------------------+
18. double m_Buff[];
19. int    m_RatesTotal = 0;
20. //+------------------------------------------------------------------+
21. int OnInit()
22. {
23.    ResetLastError();
24.    if (CheckPointer(control = new C_Controls(user00, "Market Replay Control", new C_Mouse(user00, "Indicator Mouse Study"))) == POINTER_INVALID)
25.       SetUserError(C_Terminal::ERR_PointerInvalid);
26.    if ((_LastError != ERR_SUCCESS) || (user00 == 0))
27.    {
28.       Print("Control indicator failed on initialization.");
29.       return INIT_FAILED;
30.    }
31.    SetIndexBuffer(0, m_Buff, INDICATOR_DATA);
32.    ArrayInitialize(m_Buff, EMPTY_VALUE);
33.
34.    return INIT_SUCCEEDED;
35. }
36. //+------------------------------------------------------------------+
37. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
38. {
39.    return m_RatesTotal = rates_total;
40. }
41. //+------------------------------------------------------------------+
42. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
43. {
44.    (*control).DispatchMessage(id, lparam, dparam, sparam);
45.    if (_LastError >= ERR_USER_ERROR_FIRST + C_Terminal::ERR_Unknown)
46.    {
47.       Print("Internal failure in the messaging system...");
48.       ChartClose(user00);
49.    }
50.    (*control).SetBuffer(m_RatesTotal, m_Buff);
51. }
52. //+------------------------------------------------------------------+
53. void OnDeinit(const int reason)
54. {
55.    switch (reason)
56.    {
57.       case REASON_TEMPLATE:
58.          Print("Modified template. Replay // simulation system shutting down.");
59.       case REASON_INITFAILED:
60.       case REASON_PARAMETERS:
61.       case REASON_REMOVE:
62.       case REASON_CHARTCLOSE:
63.          ChartClose(user00);
64.          break;
65.    }
66.    delete control;
67. }
68. //+------------------------------------------------------------------+
69.
```

Source code of the control indicator

You may have noticed that I don't show the code in the header file because it hasn't been changed. While we can look at the entire control module code above, we will focus on the modified parts. The only difference is in line 26. In accordance with this, we check two conditions. One of them is the \_LastError value and the other is the user00 value. This user00 is defined in line 16 and is responsible for receiving from the replay/simulation service the ID of the chart on which the control module should be running.

Now please be careful. Typically, we as users cannot determine the value that should be assigned at the beginning of line 16. It happens naturally. But what if we try to trick the system and save the entire configuration as a template file? This could make everything work and save us the hassle of using a template, right? No. This won't work as you expect.

If after the service has loaded everything and MetaTrader 5 has stabilized the chart, we save this finished chart as a template, then when we open the template file we will notice something. This can be seen from the fragment below.

```
01. <indicator>
02. name=Custom Indicator
03. path=Services\Market Replay.ex5::Indicators\Market Replay.ex5
04. apply=0
05. show_data=1
06. scale_inherit=0
07. scale_line=0
08. scale_line_percent=50
09. scale_line_value=0.000000
10. scale_fix_min=0
11. scale_fix_min_val=0.000000
12. scale_fix_max=0
13. scale_fix_max_val=0.000000
14. expertmode=1610613824
15. fixed_height=-1
16.
17. <graph>
18. name=
19. draw=0
20. style=0
21. width=1
22. color=
23. </graph>
24. <inputs>
25. user00=130652731570824061
26. </inputs>
27. </indicator>
```

Template file fragment

The exact position of this code fragment within the file is not relevant here. The line numbers are merely references to help us explain things more clearly.

Observe that line 1 contains an opening tag, while line 27 closes the structure. Between these two tags, various details are specified, including the location of the indicator to be applied to the chart, which is indicated on line 3. Great. Now, pay close attention to line 24. Here, a tag opens the section for input parameters expected by the indicator defined in line 3. Similarly, line 26 marks the closing of this section.

On line 25, we reference an input parameter that, in the indicator's source code, is declared on line 16 to receive the chart ID. This line assigns a value to the control indicator so that the condition tested on line 26 of the indicator’s source code evaluates to false, allowing the indicator to load and function properly. Correct? Unfortunately, no. When the replay/simulation service correctly applies the control indicator to the chart, the indicator code will detect that another instance is already running. When this happens, line 24 in the control indicator code will produce a result that triggers line 25. Consequently, \_LastError will no longer hold the expected value at line 26, leading to a system failure. As a result, MetaTrader 5 will generate a Deinit event, which will be handled by the OnDeinit procedure found in line 53. At this point, the execution will proceed with REASON\_INITFAILED, closing the chart and terminating the replay/simulation service.

This demonstrates how everything works seamlessly when we fully utilize MetaTrader 5 platform's capabilities. But that was the issue with the control indicator. What about the mouse indicator?

The situation with the mouse indicator is slightly more intriguing. It was decided that this module would be permitted to perform a specific action, but not exclusively. Instead, it would be the first to initiate this process. Because of this, some additions and removals were necessary within the mouse indicator module. To properly define these adjustments, let's cover them in the next section.

### Small Changes for Significant Improvements

Although the mouse module was originally intended for placement in the main window, MQL5 allows us to do much more than that. Honestly, I hadn't initially considered implementing certain functionalities. However, since MetaTrader 5 provides configuration and customization options that enable us to execute specific actions more efficiently, I decided to introduce several modifications. The following section presents the updates and additions made, starting with the C\_Terminal class, shown below.

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
015.          ENUM_SYMBOL_CHART_MODE    ChartMode;
016.          ENUM_ACCOUNT_MARGIN_MODE  TypeAccount;
017.          long           ID;
018.          string         szSymbol;
019.          int            Width,
020.                         Height,
021.                         nDigits,
022.                         SubWin;
023.          double         PointPerTick,
024.                         ValuePerPoint,
025.                         VolumeMinimal,
026.                         AdjustToTrade;
027.       };
028. //+------------------------------------------------------------------+
029.    private   :
030.       st_Terminal m_Infos;
031.       struct mem
032.       {
033.          long   Show_Descr,
034.                 Show_Date;
035.          bool   AccountLock;
036.       }m_Mem;
037. //+------------------------------------------------------------------+
038.       void CurrentSymbol(void)
039.          {
040.             MqlDateTime mdt1;
041.             string sz0, sz1;
042.             datetime dt = macroGetDate(TimeCurrent(mdt1));
043.             enum eTypeSymbol {WIN, IND, WDO, DOL, OTHER} eTS = OTHER;
044.
045.             sz0 = StringSubstr(m_Infos.szSymbol = _Symbol, 0, 3);
046.             for (eTypeSymbol c0 = 0; (c0 < OTHER) && (eTS == OTHER); c0++) eTS = (EnumToString(c0) == sz0 ? c0 : eTS);
047.             switch (eTS)
048.             {
049.                case DOL   :
050.                case WDO   : sz1 = "FGHJKMNQUVXZ"; break;
051.                case IND   :
052.                case WIN   : sz1 = "GJMQVZ";       break;
053.                default    : return;
054.             }
055.             for (int i0 = 0, i1 = mdt1.year - 2000, imax = StringLen(sz1);; i0 = ((++i0) < imax ? i0 : 0), i1 += (i0 == 0 ? 1 : 0))
056.                if (dt < macroGetDate(SymbolInfoInteger(m_Infos.szSymbol = StringFormat("%s%s%d", sz0, StringSubstr(sz1, i0, 1), i1), SYMBOL_EXPIRATION_TIME))) break;
057.          }
058. //+------------------------------------------------------------------+
059.    public   :
060. //+------------------------------------------------------------------+
061.       C_Terminal(const long id = 0, const uchar sub = 0)
062.          {
063.             m_Infos.ID = (id == 0 ? ChartID() : id);
064.             m_Mem.AccountLock = false;
065.             m_Infos.SubWin = (int) sub;
066.             CurrentSymbol();
067.             m_Mem.Show_Descr = ChartGetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR);
068.             m_Mem.Show_Date  = ChartGetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE);
069.             ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, false);
070.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, true);
071.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, true);
072.             ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, false);
073.             m_Infos.nDigits = (int) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_DIGITS);
074.             m_Infos.Width   = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
075.             m_Infos.Height  = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
076.             m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
077.             m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
078.             m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
079.             m_Infos.AdjustToTrade = m_Infos.ValuePerPoint / m_Infos.PointPerTick;
080.             m_Infos.ChartMode   = (ENUM_SYMBOL_CHART_MODE) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_CHART_MODE);
081.             if(m_Infos.szSymbol != def_SymbolReplay) SetTypeAccount((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE));
082.             ResetLastError();
083.          }
084. //+------------------------------------------------------------------+
085.       ~C_Terminal()
086.          {
087.             ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, m_Mem.Show_Date);
088.             ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, m_Mem.Show_Descr);
089.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, false);
090.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, false);
091.          }
092. //+------------------------------------------------------------------+
093. inline void SetTypeAccount(const ENUM_ACCOUNT_MARGIN_MODE arg)
094.          {
095.             if (m_Mem.AccountLock) return; else m_Mem.AccountLock = true;
096.             m_Infos.TypeAccount = (arg == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING ? arg : ACCOUNT_MARGIN_MODE_RETAIL_NETTING);
097.          }
098. //+------------------------------------------------------------------+
099. inline const st_Terminal GetInfoTerminal(void) const
100.          {
101.             return m_Infos;
102.          }
103. //+------------------------------------------------------------------+
104. const double AdjustPrice(const double arg) const
105.          {
106.             return NormalizeDouble(round(arg / m_Infos.PointPerTick) * m_Infos.PointPerTick, m_Infos.nDigits);
107.          }
108. //+------------------------------------------------------------------+
109. inline datetime AdjustTime(const datetime arg)
110.          {
111.             int nSeconds= PeriodSeconds();
112.             datetime   dt = iTime(m_Infos.szSymbol, PERIOD_CURRENT, 0);
113.
114.             return (dt < arg ? ((datetime)(arg / nSeconds) * nSeconds) : iTime(m_Infos.szSymbol, PERIOD_CURRENT, Bars(m_Infos.szSymbol, PERIOD_CURRENT, arg, dt)));
115.          }
116. //+------------------------------------------------------------------+
117. inline double FinanceToPoints(const double Finance, const uint Leverage)
118.          {
119.             double volume = m_Infos.VolumeMinimal + (m_Infos.VolumeMinimal * (Leverage - 1));
120.
121.             return AdjustPrice(MathAbs(((Finance / volume) / m_Infos.AdjustToTrade)));
122.          };
123. //+------------------------------------------------------------------+
124.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
125.          {
126.             static string st_str = "";
127.
128.             switch (id)
129.             {
130.                case CHARTEVENT_CHART_CHANGE:
131.                   m_Infos.Width  = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
132.                   m_Infos.Height = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
133.                   break;
134.                case CHARTEVENT_OBJECT_CLICK:
135.                   if (st_str != sparam) ObjectSetInteger(m_Infos.ID, st_str, OBJPROP_SELECTED, false);
136.                   if (ObjectGetInteger(m_Infos.ID, sparam, OBJPROP_SELECTABLE) == true)
137.                      ObjectSetInteger(m_Infos.ID, st_str = sparam, OBJPROP_SELECTED, true);
138.                   break;
139.                case CHARTEVENT_OBJECT_CREATE:
140.                   if (st_str != sparam) ObjectSetInteger(m_Infos.ID, st_str, OBJPROP_SELECTED, false);
141.                   st_str = sparam;
142.                   break;
143.             }
144.          }
145. //+------------------------------------------------------------------+
146. inline void CreateObjectGraphics(const string szName, const ENUM_OBJECT obj, const color cor = clrNONE, const int zOrder = -1) const
147.          {
148.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, 0, false);
149.             ObjectCreate(m_Infos.ID, szName, obj, m_Infos.SubWin, 0, 0);
150.             ObjectSetString(m_Infos.ID, szName, OBJPROP_TOOLTIP, "\n");
151.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_BACK, false);
152.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_COLOR, cor);
153.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_SELECTABLE, false);
154.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_SELECTED, false);
155.             ObjectSetInteger(m_Infos.ID, szName, OBJPROP_ZORDER, zOrder);
156.             ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, 0, true);
157.          }
158. //+------------------------------------------------------------------+
159.       bool IndicatorCheckPass(const string szShortName)
160.          {
161.             string szTmp = szShortName + "_TMP";
162.
163.             if (_LastError != ERR_SUCCESS) return false;
164.             IndicatorSetString(INDICATOR_SHORTNAME, szTmp);
165.             m_Infos.SubWin = ((m_Infos.SubWin = ChartWindowFind(m_Infos.ID, szTmp)) < 0 ? 0 : m_Infos.SubWin);
166.             if (ChartIndicatorGet(m_Infos.ID, m_Infos.SubWin, szShortName) != INVALID_HANDLE)
167.             {
168.                ChartIndicatorDelete(m_Infos.ID, 0, szTmp);
169.                Print("Only one instance is allowed...");
170.                SetUserError(C_Terminal::ERR_NoMoreInstance);
171.
172.                return false;
173.             }
174.             IndicatorSetString(INDICATOR_SHORTNAME, szShortName);
175.             ResetLastError();
176.
177.             return true;
178.          }
179. //+------------------------------------------------------------------+
180. };
```

Source code of the C\_Terminal.mqh header file

When reviewing the code in the header file containing the C\_Terminal class, you might not immediately notice the subtle changes introduced. These modifications are minor but enable us to accomplish a lot, especially tasks we will need to perform in the near future.

All the changes stem from line 22, where a new variable has been introduced. Previously, this variable did not exist, but it has now been added to allow us to direct objects to a specific subwindow.

Since this variable can be accessed outside the class, we must ensure it has a proper initial value. To achieve this, we made a modification to the class constructor it line 61. Now, the constructor receives an additional parameter, which can hold values between 0 and 255. This range is more than sufficient, considering that charts rarely contain more than two or three subwindows. However, there's an important detail to address. Looking at line 65, we handle this with an explicit type conversion. But why not declare the variable as uchar from the start instead of converting it here? The answer lies in backward compatibility. MQL5 expects a signed integer type, and keeping it this way simplifies our life later. At the same time, using an uchar for the parameter conveniently limits the number of subwindows to 255. This approach ensures both compatibility and flexibility.

Moving to line 149, we see how this value is being used. It informs MetaTrader 5 which chart window should display the object. For those familiar with MQL5, this is a common technique. However, things become more interesting when we examine the IndicadorCheckPass function, which begins at line 159.

Typically, when placing an indicator on a chart, we do not need to specify in which window it will be plotted. However, when working with graphical objects, things become more complex because we must explicitly determine which window will contain the object. Without this information, the function call at line 149 would always place the object in the wrong window. So, how do we determine the correct window?

To do this efficiently, we use a simple trick on line 164: assigning a temporary name to the indicator. Then, at line 165, we use the ChartWindowFind function in MQL5. This function allows MetaTrader 5 to tell us precisely which window contains the indicator. A crucial point here is that no other indicator should have a similar temporary name, as this could lead to false positives. If ChartWindowFind does not return a valid window index, we default to the main window (index 0). Finally, on line 166, we ensure that only a single instance of the indicator is present on the chart. The rest of the function remains unchanged and does not require further discussion.

With these adjustments, we can now position the mouse indicator module in any subwindow. However, for it to function correctly, we need to modify some aspects of the class code. Next, we will quickly review the changes made to the C\_Mouse.mqh header file. You can see it below.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "C_Terminal.mqh"
005. //+------------------------------------------------------------------+
006. #define def_MousePrefixName "MouseBase" + (string)GetInfoTerminal().SubWin + "_"
007. #define macro_NameObjectStudy (def_MousePrefixName + "T" + (string)ObjectsTotal(0))
008. //+------------------------------------------------------------------+
009. class C_Mouse : public C_Terminal
010. {
011.    public   :
012.       enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
013.       enum eBtnMouse {eKeyNull = 0x00, eClickLeft = 0x01, eClickRight = 0x02, eSHIFT_Press = 0x04, eCTRL_Press = 0x08, eClickMiddle = 0x10};
014.       struct st_Mouse
015.       {
016.          struct st00
017.          {
018.             short    X_Adjusted,
019.                      Y_Adjusted,
020.                      X_Graphics,
021.                      Y_Graphics;
022.             double   Price;
023.             datetime dt;
024.          }Position;
025.          uchar      ButtonStatus;
026.          bool       ExecStudy;
027.       };
028. //+------------------------------------------------------------------+
029.    protected:
030. //+------------------------------------------------------------------+
031.       void CreateObjToStudy(int x, int w, string szName, color backColor = clrNONE) const
032.          {
033.             if (!m_OK) return;
034.             CreateObjectGraphics(szName, OBJ_BUTTON);
035.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_STATE, true);
036.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BORDER_COLOR, clrBlack);
037.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_COLOR, clrBlack);
038.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BGCOLOR, backColor);
039.             ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_FONT, "Lucida Console");
040.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_FONTSIZE, 10);
041.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
042.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XDISTANCE, x);
043.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YDISTANCE, TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT) + 1);
044.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XSIZE, w);
045.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YSIZE, 18);
046.          }
047. //+------------------------------------------------------------------+
048.    private   :
049.       enum eStudy {eStudyNull, eStudyCreate, eStudyExecute};
050.       struct st01
051.       {
052.          st_Mouse Data;
053.          color    corLineH,
054.                   corTrendP,
055.                   corTrendN;
056.          eStudy   Study;
057.       }m_Info;
058.       struct st_Mem
059.       {
060.          bool     CrossHair,
061.                   IsFull;
062.          datetime dt;
063.          string   szShortName,
064.                   szLineH,
065.                   szLineV,
066.                   szLineT,
067.                   szBtnS;
068.       }m_Mem;
069.       bool m_OK;
070. //+------------------------------------------------------------------+
071.       void GetDimensionText(const string szArg, int &w, int &h)
072.          {
073.             TextSetFont("Lucida Console", -100, FW_NORMAL);
074.             TextGetSize(szArg, w, h);
075.             h += 5;
076.             w += 5;
077.          }
078. //+------------------------------------------------------------------+
079.       void CreateStudy(void)
080.          {
081.             if (m_Mem.IsFull)
082.             {
083.                CreateObjectGraphics(m_Mem.szLineV = macro_NameObjectStudy, OBJ_VLINE, m_Info.corLineH);
084.                CreateObjectGraphics(m_Mem.szLineT = macro_NameObjectStudy, OBJ_TREND, m_Info.corLineH);
085.                ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szLineT, OBJPROP_WIDTH, 2);
086.                CreateObjToStudy(0, 0, m_Mem.szBtnS = macro_NameObjectStudy);
087.             }
088.             m_Info.Study = eStudyCreate;
089.          }
090. //+------------------------------------------------------------------+
091.       void ExecuteStudy(const double memPrice)
092.          {
093.             double v1 = GetInfoMouse().Position.Price - memPrice;
094.             int w, h;
095.
096.             if (!CheckClick(eClickLeft))
097.             {
098.                m_Info.Study = eStudyNull;
099.                ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
100.                if (m_Mem.IsFull)   ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName + "T");
101.             }else if (m_Mem.IsFull)
102.             {
103.                string sz1 = StringFormat(" %." + (string)GetInfoTerminal().nDigits + "f [ %d ] %02.02f%% ",
104.                   MathAbs(v1), Bars(GetInfoTerminal().szSymbol, PERIOD_CURRENT, m_Mem.dt, GetInfoMouse().Position.dt) - 1, MathAbs((v1 / memPrice) * 100.0));
105.                GetDimensionText(sz1, w, h);
106.                ObjectSetString(GetInfoTerminal().ID, m_Mem.szBtnS, OBJPROP_TEXT, sz1);
107.                ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szBtnS, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corTrendN : m_Info.corTrendP));
108.                ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szBtnS, OBJPROP_XSIZE, w);
109.                ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szBtnS, OBJPROP_YSIZE, h);
110.                ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szBtnS, OBJPROP_XDISTANCE, GetInfoMouse().Position.X_Adjusted - w);
111.                ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szBtnS, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - (v1 < 0 ? 1 : h));
112.                ObjectMove(GetInfoTerminal().ID, m_Mem.szLineT, 1, GetInfoMouse().Position.dt, GetInfoMouse().Position.Price);
113.                ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szLineT, OBJPROP_COLOR, (memPrice > GetInfoMouse().Position.Price ? m_Info.corTrendN : m_Info.corTrendP));
114.             }
115.             m_Info.Data.ButtonStatus = eKeyNull;
116.          }
117. //+------------------------------------------------------------------+
118. inline void DecodeAlls(int xi, int yi)
119.          {
120.             int w = 0;
121.
122.             xi = (xi > 0 ? xi : 0);
123.             yi = (yi > 0 ? yi : 0);
124.             ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X_Graphics = (short)xi, m_Info.Data.Position.Y_Graphics = (short)yi, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
125.             m_Info.Data.Position.dt = AdjustTime(m_Info.Data.Position.dt);
126.             m_Info.Data.Position.Price = AdjustPrice(m_Info.Data.Position.Price);
127.             ChartTimePriceToXY(GetInfoTerminal().ID, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price, xi, yi);
128.             yi -= (int)ChartGetInteger(GetInfoTerminal().ID, CHART_WINDOW_YDISTANCE, GetInfoTerminal().SubWin);
129.             m_Info.Data.Position.X_Adjusted = (short) xi;
130.             m_Info.Data.Position.Y_Adjusted = (short) yi;
131.          }
132. //+------------------------------------------------------------------+
133.    public   :
134. //+------------------------------------------------------------------+
135.       C_Mouse(const long id, const string szShortName)
136.          :C_Terminal(id),
137.          m_OK(false)
138.          {
139.             m_Mem.szShortName = szShortName;
140.          }
141. //+------------------------------------------------------------------+
142.       C_Mouse(const long id, const string szShortName, color corH, color corP, color corN)
143.          :C_Terminal(id)
144.          {
145.             if (!(m_OK = IndicatorCheckPass(m_Mem.szShortName = szShortName))) SetUserError(C_Terminal::ERR_Unknown);
146.             if (_LastError != ERR_SUCCESS) return;
147.             m_Mem.CrossHair = (bool)ChartGetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL);
148.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, true);
149.              ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, false);
150.             ZeroMemory(m_Info);
151.             m_Info.corLineH  = corH;
152.             m_Info.corTrendP = corP;
153.             m_Info.corTrendN = corN;
154.             m_Info.Study = eStudyNull;
155.             if (m_Mem.IsFull = (corP != clrNONE) && (corH != clrNONE) && (corN != clrNONE))
156.                CreateObjectGraphics(m_Mem.szLineH = (def_MousePrefixName + (string)ObjectsTotal(0)), OBJ_HLINE, m_Info.corLineH);
157.             ChartRedraw(GetInfoTerminal().ID);
158.          }
159. //+------------------------------------------------------------------+
160.       ~C_Mouse()
161.          {
162.             if (!m_OK) return;
163.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
164.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, ChartWindowFind(GetInfoTerminal().ID, m_Mem.szShortName) != -1);
165.              ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, m_Mem.CrossHair);
166.             ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName);
167.          }
168. //+------------------------------------------------------------------+
169. inline bool CheckClick(const eBtnMouse value)
170.          {
171.             return (GetInfoMouse().ButtonStatus & value) == value;
172.          }
173. //+------------------------------------------------------------------+
174. inline const st_Mouse GetInfoMouse(void)
175.          {
176.             if (!m_OK)
177.             {
178.                double Buff[];
179.                uCast_Double loc;
180.                int handle = ChartIndicatorGet(GetInfoTerminal().ID, 0, m_Mem.szShortName);
181.
182.                ZeroMemory(m_Info.Data);
183.                if (CopyBuffer(handle, 0, 0, 1, Buff) == 1)
184.                {
185.                   loc.dValue = Buff[0];
186.                   m_Info.Data.ButtonStatus = loc._8b[0];
187.                   DecodeAlls((int)loc._16b[1], (int)loc._16b[2]);
188.                }
189.                IndicatorRelease(handle);
190.             }
191.
192.             return m_Info.Data;
193.          }
194. //+------------------------------------------------------------------+
195. inline void SetBuffer(const int rates_total, double &Buff[])
196.          {
197.             uCast_Double info;
198.
199.             info._8b[0] = (uchar)(m_Info.Study == C_Mouse::eStudyNull ? m_Info.Data.ButtonStatus : 0);
200.             info._16b[1] = (ushort) m_Info.Data.Position.X_Graphics;
201.             info._16b[2] = (ushort) m_Info.Data.Position.Y_Graphics;
202.             Buff[rates_total - 1] = info.dValue;
203.          }
204. //+------------------------------------------------------------------+
205.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
206.          {
207.             int w = 0;
208.             static double memPrice = 0;
209.
210.             if (m_OK)
211.             {
212.                C_Terminal::DispatchMessage(id, lparam, dparam, sparam);
213.                switch (id)
214.                {
215.                   case (CHARTEVENT_CUSTOM + evHideMouse):
216.                      if (m_Mem.IsFull)   ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szLineH, OBJPROP_COLOR, clrNONE);
217.                      break;
218.                   case (CHARTEVENT_CUSTOM + evShowMouse):
219.                      if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, m_Mem.szLineH, OBJPROP_COLOR, m_Info.corLineH);
220.                      break;
221.                   case CHARTEVENT_MOUSE_MOVE:
222.                      DecodeAlls((int)lparam, (int)dparam);
223.                      if (m_Mem.IsFull) ObjectMove(GetInfoTerminal().ID, m_Mem.szLineH, 0, 0, m_Info.Data.Position.Price);
224.                      if ((m_Info.Study != eStudyNull) && (m_Mem.IsFull)) ObjectMove(GetInfoTerminal().ID, m_Mem.szLineV, 0, m_Info.Data.Position.dt, 0);
225.                      m_Info.Data.ButtonStatus = (uchar) sparam;
226.                      if (CheckClick(eClickMiddle))
227.                         if ((!m_Mem.IsFull) || ((color)ObjectGetInteger(GetInfoTerminal().ID, m_Mem.szLineH, OBJPROP_COLOR) != clrNONE)) CreateStudy();
228.                      if (CheckClick(eClickLeft) && (m_Info.Study == eStudyCreate))
229.                      {
230.                         ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
231.                         if (m_Mem.IsFull)   ObjectMove(GetInfoTerminal().ID, m_Mem.szLineT, 0, m_Mem.dt = GetInfoMouse().Position.dt, memPrice = GetInfoMouse().Position.Price);
232.                         m_Info.Study = eStudyExecute;
233.                      }
234.                      if (m_Info.Study == eStudyExecute) ExecuteStudy(memPrice);
235.                      m_Info.Data.ExecStudy = m_Info.Study == eStudyExecute;
236.                      break;
237.                   case CHARTEVENT_OBJECT_DELETE:
238.                      if ((m_Mem.IsFull) && (sparam == m_Mem.szLineH))
239.                         CreateObjectGraphics(m_Mem.szLineH, OBJ_HLINE, m_Info.corLineH);
240.                      break;
241.                }
242.             }
243.          }
244. //+------------------------------------------------------------------+
245. };
246. //+------------------------------------------------------------------+
247. #undef macro_NameObjectStudy
248. //+------------------------------------------------------------------+
```

Source code of the C\_Mouse.mqh header file

The first thing that immediately catches our attention can be seen on lines 6 and 7. Notice that, unlike before, we have implemented a slightly different naming convention. This adjustment is necessary to avoid conflicts with existing object names on the chart. As a result, the name now becomes dependent on both the window and the number of objects present on the chart. In other words, each name is now unique.

There are some minor differences in the code, but nothing that truly deserves significant attention. However, between lines 64 and 67, new variables are declared. These variables are used to store the names of the objects to be created. To help you understand how this works, look at line 83, where an example of naming one of the objects is provided. Here, it is clear how the macro is used to generate and assign a name to one of the variables.

Although much of the code doesn't require special emphasis, as the modifications were subtle and intended to improve support for our needs, there is a section that does require some explanation. While it is not perfect, it is sufficient for achieving our goals. I am referring to the procedure that begins at line 118, DecodeAlls.

This procedure works wonderfully when the mouse indicator module is located in the main window (i.e., window with zero index). However, when we place the mouse indicator in a different window, problems start to arise. While we have resolved many of them, a few remain, as you will see in the video at the end of this article.

The significant point, which may leave you, dear reader, utterly perplexed and disoriented, lies in line 128. Why does this line exist, and why was it not there before? To understand this, it is essential to understand another thing. The mouse indicator module was originally intended to be displayed only in the main window. This window has its initial Y position at the top, meaning Y always starts at zero. However, when we add extra windows, the Y position of the main window remains unchanged, while those of the extra windows are offset by a specific value. But for the operating system (Windows), this does not matter, and it informs MetaTrader 5 of the exact position of the mouse.

MetaTrader 5 then adjusts the mouse position so that it remains within the chart window. Consequently, values outside the window may be either negative or positive. A negative value occurs if the mouse pointer is above the window's workspace area. If you are unfamiliar with what constitutes the workspace area of the window, refer to Figure 01, where the entire region containing the bitmap is considered the workspace.

![Figure 01](https://c.mql5.com/2/113/001__1.png)

Figure 01 - Understanding the workspace area

Note that the borders and title bar are not part of the window's workspace. Therefore, if the mouse pointer enters the title bar area, it is MetaTrader 5, not the operating system, that corrects the value to make it negative. It is crucial to understand that the Y value becomes negative when the pointer enters the title bar area, not because the operating system did this, but because MetaTrader 5 corrected the value to keep it within the workspace area.

When you add an element that creates a region, such as a strip where the chart no longer belongs, within the main window, MetaTrader 5 does not recognize this as a separate window. Even if we define it as such, MetaTrader 5 continues to view the entire window as a single entity, and all of the workspace remains within the borders of the main window. Fortunately, MetaTrader 5 allows us to know where the region begins, though it does not provide information on where it ends. It is up to us to find a way to determine this. Nevertheless, simply knowing where the region begins is already beneficial.

To identify where the region starts, we call ChartGetInteger, passing the constant CHART\_WINDOW\_YDISTANCE and the subwindow number. While we use the term "subwindow" to simplify matters, this terminology isn't entirely accurate.

The value returned by the call in line 128 is subtracted from the converted value. Remember, the converted value represents what MetaTrader 5 reports, so it remains within the window. Without this correction in line 128, when the mouse pointer enters a position that MetaTrader 5 interprets as part of the subwindow region, we would have a false indication of the horizontal line's position. However, with this correction in place, that issue is avoided. At least, not in the way it would have normally occurred.

Another important point that deserves mention is found in line 145, where we store the indicator's short name. But why do we do this? The reason becomes clear in line 164. Without knowing the mouse indicator name, we wouldn't be able to check and determine whether we can tell MetaTrader 5 to stop sending mouse events. Some might question whether we could simply keep the event always enabled, but that would be unnecessary. Every time the mouse moved, MetaTrader 5 would trigger a mouse event, and if no one is using it, this would just add something useless to the event queue. To avoid this, we disable the event as soon as we no longer need it. But without knowing the mouse indicator name, how could we tell if there was an indicator that still required the event? There would be no way. So, to make things easier, we store the indicator name and use it to check whether or not we can turn off the mouse event.

In addition to the changes mentioned above, there are others, but since they are relatively simpler, I will not go into detail. To find them, simply compare this code with the previous versions of the C\_Mouse.mqh header file. This will be your homework to learn more about how to code and modify things without introducing major issues in the final code.

As expected, the C\_Study.mqh file also underwent some changes. However, just as with the changes in the C\_Mouse.mqh header file, I will not go into detail here. Instead, I will provide the full code of C\_Study.mqh so you can see how it was modified.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\C_Mouse.mqh"
005. //+------------------------------------------------------------------+
006. #define def_ExpansionPrefix def_MousePrefixName + "Expansion_"
007. //+------------------------------------------------------------------+
008. class C_Study : public C_Mouse
009. {
010.    private   :
011. //+------------------------------------------------------------------+
012.       struct st00
013.       {
014.          eStatusMarket  Status;
015.          MqlRates       Rate;
016.          string         szInfo,
017.                         szBtn1,
018.                         szBtn2,
019.                         szBtn3;
020.          color          corP,
021.                         corN;
022.          int            HeightText;
023.          bool           bvT, bvD, bvP;
024.          datetime       TimeDevice;
025.       }m_Info;
026. //+------------------------------------------------------------------+
027.       const datetime GetBarTime(void)
028.          {
029.             datetime dt;
030.             int i0 = PeriodSeconds();
031.
032.             if (m_Info.Status == eInReplay)
033.             {
034.                if ((dt = m_Info.TimeDevice) == ULONG_MAX) return ULONG_MAX;
035.             }else dt = TimeCurrent();
036.             if (m_Info.Rate.time <= dt)
037.                m_Info.Rate.time = (datetime)(((ulong) dt / i0) * i0) + i0;
038.
039.             return m_Info.Rate.time - dt;
040.          }
041. //+------------------------------------------------------------------+
042.       void Draw(void)
043.          {
044.             double v1;
045.
046.             if (m_Info.bvT)
047.             {
048.                ObjectSetInteger(GetInfoTerminal().ID, m_Info.szBtn1, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 18);
049.                ObjectSetString(GetInfoTerminal().ID, m_Info.szBtn1, OBJPROP_TEXT, m_Info.szInfo);
050.             }
051.             if (m_Info.bvD)
052.             {
053.                v1 = NormalizeDouble((((GetInfoMouse().Position.Price - m_Info.Rate.close) / m_Info.Rate.close) * 100.0), 2);
054.                ObjectSetInteger(GetInfoTerminal().ID, m_Info.szBtn2, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 1);
055.                ObjectSetInteger(GetInfoTerminal().ID, m_Info.szBtn2, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
056.                ObjectSetString(GetInfoTerminal().ID, m_Info.szBtn2, OBJPROP_TEXT, StringFormat("%.2f%%", MathAbs(v1)));
057.             }
058.             if (m_Info.bvP)
059.             {
060.                v1 = NormalizeDouble((((iClose(GetInfoTerminal().szSymbol, PERIOD_D1, 0) - m_Info.Rate.close) / m_Info.Rate.close) * 100.0), 2);
061.                ObjectSetInteger(GetInfoTerminal().ID, m_Info.szBtn3, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 1);
062.                ObjectSetInteger(GetInfoTerminal().ID, m_Info.szBtn3, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
063.                ObjectSetString(GetInfoTerminal().ID, m_Info.szBtn3, OBJPROP_TEXT, StringFormat("%.2f%%", MathAbs(v1)));
064.             }
065.          }
066. //+------------------------------------------------------------------+
067. inline void CreateObjInfo(EnumEvents arg)
068.          {
069.             switch (arg)
070.             {
071.                case evShowBarTime:
072.                   C_Mouse::CreateObjToStudy(2, 110, m_Info.szBtn1 = (def_ExpansionPrefix + (string)ObjectsTotal(0)), clrPaleTurquoise);
073.                   m_Info.bvT = true;
074.                   break;
075.                case evShowDailyVar:
076.                   C_Mouse::CreateObjToStudy(2, 53, m_Info.szBtn2 = (def_ExpansionPrefix + (string)ObjectsTotal(0)));
077.                   m_Info.bvD = true;
078.                   break;
079.                case evShowPriceVar:
080.                   C_Mouse::CreateObjToStudy(58, 53, m_Info.szBtn3 = (def_ExpansionPrefix + (string)ObjectsTotal(0)));
081.                   m_Info.bvP = true;
082.                   break;
083.             }
084.          }
085. //+------------------------------------------------------------------+
086. inline void RemoveObjInfo(EnumEvents arg)
087.          {
088.             string sz;
089.
090.             switch (arg)
091.             {
092.                case evHideBarTime:
093.                   sz = m_Info.szBtn1;
094.                   m_Info.bvT = false;
095.                   break;
096.                case evHideDailyVar:
097.                   sz = m_Info.szBtn2;
098.                   m_Info.bvD   = false;
099.                   break;
100.                case evHidePriceVar:
101.                   sz = m_Info.szBtn3;
102.                   m_Info.bvP = false;
103.                   break;
104.             }
105.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
106.             ObjectDelete(GetInfoTerminal().ID, sz);
107.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
108.          }
109. //+------------------------------------------------------------------+
110.    public   :
111. //+------------------------------------------------------------------+
112.       C_Study(long IdParam, string szShortName, color corH, color corP, color corN)
113.          :C_Mouse(IdParam, szShortName, corH, corP, corN)
114.          {
115.             if (_LastError != ERR_SUCCESS) return;
116.             ZeroMemory(m_Info);
117.             m_Info.Status = eCloseMarket;
118.             m_Info.Rate.close = iClose(GetInfoTerminal().szSymbol, PERIOD_D1, ((GetInfoTerminal().szSymbol == def_SymbolReplay) || (macroGetDate(TimeCurrent()) != macroGetDate(iTime(GetInfoTerminal().szSymbol, PERIOD_D1, 0))) ? 0 : 1));
119.             m_Info.corP = corP;
120.             m_Info.corN = corN;
121.             CreateObjInfo(evShowBarTime);
122.             CreateObjInfo(evShowDailyVar);
123.             CreateObjInfo(evShowPriceVar);
124.          }
125. //+------------------------------------------------------------------+
126.       void Update(const eStatusMarket arg)
127.          {
128.             datetime dt;
129.
130.             switch (m_Info.Status = (m_Info.Status != arg ? arg : m_Info.Status))
131.             {
132.                case eCloseMarket   :
133.                   m_Info.szInfo = "Closed Market";
134.                   break;
135.                case eInReplay      :
136.                case eInTrading   :
137.                   if ((dt = GetBarTime()) < ULONG_MAX)
138.                   {
139.                      m_Info.szInfo = TimeToString(dt, TIME_SECONDS);
140.                      break;
141.                   }
142.                case eAuction      :
143.                   m_Info.szInfo = "Auction";
144.                   break;
145.                default            :
146.                   m_Info.szInfo = "ERROR";
147.             }
148.             Draw();
149.          }
150. //+------------------------------------------------------------------+
151. virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
152.          {
153.             C_Mouse::DispatchMessage(id, lparam, dparam, sparam);
154.             switch (id)
155.             {
156.                case CHARTEVENT_CUSTOM + evHideBarTime:
157.                   RemoveObjInfo(evHideBarTime);
158.                   break;
159.                case CHARTEVENT_CUSTOM + evShowBarTime:
160.                   CreateObjInfo(evShowBarTime);
161.                   break;
162.                case CHARTEVENT_CUSTOM + evHideDailyVar:
163.                   RemoveObjInfo(evHideDailyVar);
164.                   break;
165.                case CHARTEVENT_CUSTOM + evShowDailyVar:
166.                   CreateObjInfo(evShowDailyVar);
167.                   break;
168.                case CHARTEVENT_CUSTOM + evHidePriceVar:
169.                   RemoveObjInfo(evHidePriceVar);
170.                   break;
171.                case CHARTEVENT_CUSTOM + evShowPriceVar:
172.                   CreateObjInfo(evShowPriceVar);
173.                   break;
174.                case (CHARTEVENT_CUSTOM + evSetServerTime):
175.                   m_Info.TimeDevice = (datetime)dparam;
176.                   break;
177.                case CHARTEVENT_MOUSE_MOVE:
178.                   Draw();
179.                   break;
180.             }
181.             ChartRedraw(GetInfoTerminal().ID);
182.          }
183. //+------------------------------------------------------------------+
184. };
185. //+------------------------------------------------------------------+
186. #undef def_ExpansionPrefix
187. #undef def_MousePrefixName
188. //+------------------------------------------------------------------+
```

Source code of the C\_Study.mqh class

However, in this article, I want to show the following code. This is the source code of the indicator.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "This is an indicator for graphical studies using the mouse."
04. #property description "This is an integral part of the Replay / Simulator system."
05. #property description "However it can be used in the real market."
06. #property version "1.59"
07. #property icon "/Images/Market Replay/Icons/Indicators.ico"
08. #property link "https://www.mql5.com/pt/articles/12075"
09. #property indicator_chart_window
10. #property indicator_plots 0
11. #property indicator_buffers 1
12. //+------------------------------------------------------------------+
13. #include <Market Replay\Auxiliar\Study\C_Study.mqh>
14. //+------------------------------------------------------------------+
15. C_Study *Study       = NULL;
16. //+------------------------------------------------------------------+
17. input color user02   = clrBlack;                       //Price Line
18. input color user03   = clrPaleGreen;                   //Positive Study
19. input color user04   = clrLightCoral;                  //Negative Study
20. //+------------------------------------------------------------------+
21. C_Study::eStatusMarket m_Status;
22. int m_posBuff = 0;
23. double m_Buff[];
24. //+------------------------------------------------------------------+
25. int OnInit()
26. {
27.    ResetLastError();
28.    Study = new C_Study(0, "Indicator Mouse Study", user02, user03, user04);
29.    if (_LastError != ERR_SUCCESS) return INIT_FAILED;
30.    if ((*Study).GetInfoTerminal().szSymbol != def_SymbolReplay)
31.    {
32.       MarketBookAdd((*Study).GetInfoTerminal().szSymbol);
33.       OnBookEvent((*Study).GetInfoTerminal().szSymbol);
34.       m_Status = C_Study::eCloseMarket;
35.    }else
36.       m_Status = C_Study::eInReplay;
37.    SetIndexBuffer(0, m_Buff, INDICATOR_DATA);
38.    ArrayInitialize(m_Buff, EMPTY_VALUE);
39.
40.    return INIT_SUCCEEDED;
41. }
42. //+------------------------------------------------------------------+
43. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
44. {
45.    m_posBuff = rates_total;
46.    (*Study).Update(m_Status);
47.
48.    return rates_total;
49. }
50. //+------------------------------------------------------------------+
51. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
52. {
53.    (*Study).DispatchMessage(id, lparam, dparam, sparam);
54.    (*Study).SetBuffer(m_posBuff, m_Buff);
55.
56.    ChartRedraw((*Study).GetInfoTerminal().ID);
57. }
58. //+------------------------------------------------------------------+
59. void OnBookEvent(const string &symbol)
60. {
61.    MqlBookInfo book[];
62.    C_Study::eStatusMarket loc = m_Status;
63.
64.    if (symbol != (*Study).GetInfoTerminal().szSymbol) return;
65.    MarketBookGet((*Study).GetInfoTerminal().szSymbol, book);
66.    m_Status = (ArraySize(book) == 0 ? C_Study::eCloseMarket : C_Study::eInTrading);
67.    for (int c0 = 0; (c0 < ArraySize(book)) && (m_Status != C_Study::eAuction); c0++)
68.       if ((book[c0].type == BOOK_TYPE_BUY_MARKET) || (book[c0].type == BOOK_TYPE_SELL_MARKET)) m_Status = C_Study::eAuction;
69.    if (loc != m_Status) (*Study).Update(m_Status);
70. }
71. //+------------------------------------------------------------------+
72. void OnDeinit(const int reason)
73. {
74.    if (reason != REASON_INITFAILED)
75.    {
76.       if ((*Study).GetInfoTerminal().szSymbol != def_SymbolReplay)
77.          MarketBookRelease((*Study).GetInfoTerminal().szSymbol);
78.    }
79.    delete Study;
80. }
81. //+------------------------------------------------------------------+
82.
```

Mouse Pointer source code

In this code, right from the start, you can notice that we are no longer using certain inputs. This means that the way we approach this indicator has been modified. The reason for this change is to give you the freedom to place the indicator wherever you see fit. It could be used for purposes beyond what I initially envisioned. Therefore, I will no longer indicate a specific location for it. Consequently, the inputs that previously required the user to set the chart ID and asset status are no longer necessary.

However, I would like you to pay attention to a small change made шn line 36. Now, if this module is added to the chart of the asset being used by the replay/simulator service, line 36 will automatically adjust things. This means that the user will no longer have the option to make changes that were previously required, such as specifying the chart ID. A note: Although the chart ID no longer needs to be provided, everything discussed in the previous articles regarding the function calls still applies, at least up to the point I'm writing this article.

### Conclusion

With a bit of work and leveraging prior knowledge, I've been able to demonstrate how certain features in MetaTrader 5 function. It's true that if you've been following the articles, you might have the impression that we haven't made much progress. In reality, we have been making progress, albeit at a slower pace. This is because I've had to come up with and test things that I wasn't sure could actually be done in MetaTrader 5.

Many people who call themselves programmers simply claim that certain things cannot be done in MetaTrader 5 – that the platform lacks this or that feature. However, I've found that these individuals are often misinformed.

To conclude this article, I'll leave you with a video showing how the mouse indicator module is functioning. Pay close attention to the video and note that there is a flaw. I am aware of it, but since it is not critical, I'll save the fix for another time, once I've gained a better understanding of how MetaTrader 5 truly works, particularly in aspects that I haven't fully understood yet.

I want you, dear reader, to be a part of this learning process. So, I will continue to demonstrate how the system designed for replay/simulation is also being developed for use in the real market and demo accounts.

Demonstração 59 2 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12075)

MQL5.community

1.91K subscribers

[Demonstração 59 2](https://www.youtube.com/watch?v=oaus2WLTIhY)

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

[Watch on](https://www.youtube.com/watch?v=oaus2WLTIhY&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12075)

0:00

0:00 / 3:06

•Live

•

Demonstration Video: The New Mouse Indicator in Action

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12075](https://www.mql5.com/pt/articles/12075)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12075.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12075/anexo.zip "Download Anexo.zip")(420.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/481432)**
(1)


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
5 Aug 2024 at 12:42

I have a suggestion.

Based on your market replication, do something a little different that can be useful and interesting not only to you, but to many other people.

I would do it myself, but I am not a professional programmer. And I study all your codes thoroughly. And I cannot implement my idea on their basis due to lack of time.

And if you find my proposal interesting and/or consider making a commercial [product](https://www.metatrader5.com/en/terminal/help/fundamental/economic_indicators_usa/usa_productivity "Productivity") based on this idea, then I would like to get free access to it in payment for the idea.

And if you agree, then I am ready to voice it.


![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (II): Modularization](https://c.mql5.com/2/119/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_IX___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (II): Modularization](https://www.mql5.com/en/articles/16562)

In this discussion, we take a step further in breaking down our MQL5 program into smaller, more manageable modules. These modular components will then be integrated into the main program, enhancing its organization and maintainability. This approach simplifies the structure of our main program and makes the individual components reusable in other Expert Advisors (EAs) and indicator developments. By adopting this modular design, we create a solid foundation for future enhancements, benefiting both our project and the broader developer community.

![From Basic to Intermediate: Variables (III)](https://c.mql5.com/2/87/Do_b9sico_ao_intermediwrio_Varicveis_III____LOGO.png)[From Basic to Intermediate: Variables (III)](https://www.mql5.com/en/articles/15304)

Today we will look at how to use predefined MQL5 language variables and constants. In addition, we will analyze another special type of variables: functions. Knowing how to properly work with these variables can mean the difference between an application that works and one that doesn't. In order to understand what is presented here, it is necessary to understand the material that was discussed in previous articles.

![Price Action Analysis Toolkit Development (Part 13): RSI Sentinel Tool](https://c.mql5.com/2/119/Price_Action_Analysis_Toolkit_Development_Part_13___LOGO.png)[Price Action Analysis Toolkit Development (Part 13): RSI Sentinel Tool](https://www.mql5.com/en/articles/17198)

Price action can be effectively analyzed by identifying divergences, with technical indicators such as the RSI providing crucial confirmation signals. In the article below, we explain how automated RSI divergence analysis can identify trend continuations and reversals, thereby offering valuable insights into market sentiment.

![Deconstructing examples of trading strategies in the client terminal](https://c.mql5.com/2/88/logo-examples-of-trading-strategies_15479_387_3725.png)[Deconstructing examples of trading strategies in the client terminal](https://www.mql5.com/en/articles/15479)

The article uses block diagrams to examine the logic of the candlestick-based training EAs located in the Experts\\Free Robots folder of the terminal.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/12075&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069660022226487461)

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