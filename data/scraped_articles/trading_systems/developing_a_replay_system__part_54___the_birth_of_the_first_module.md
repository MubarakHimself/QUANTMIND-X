---
title: Developing a Replay System (Part 54): The Birth of the First Module
url: https://www.mql5.com/en/articles/11971
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:05:16.813337
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/11971&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069997009655500346)

MetaTrader 5 / Examples


### Introduction

In the previous article " [Developing a Replay System (Part 53): Things Get Complicated (V)](https://www.mql5.com/en/articles/11932)", I explained some concepts that will become part of our programming from now on, and that we will use in MQL5 and the MetaTrader 5 platform. I realize that many of these concepts are new to most readers, and I also know that anyone with systems programming experience (such as those who program Windows systems) will be familiar with these concepts.

So if you really want to dive in and understand why and how what I'm about to show you works, I suggest you learn a little Windows programming to know how messages are exchanged between programs. This information in the article would take us far away from what I really want to show: how to develop and work in MQL5 and MetaTrader 5 at a more advanced level.

You won't have any trouble finding programs that use this messaging to communicate in the Windows environment, but understanding them properly requires some prior knowledge and a solid foundation in C programming. If you don't have this knowledge, my advice is to start by learning C programming, and then learn how messages are exchanged between programs in Windows. In this way, it will be possible to create a broad and solid basis for understanding our further work.

### Making things possible

If you read the previous article carefully, you probably noticed that I was trying to do something for a long period of time. However, even though all these elements worked partially, they could not coexist in the larger system, or at least not the way things developed.

Perhaps my biggest mistake was ignoring for several weeks that our applications respond to events in the MetaTrader 5 platform, but the mistake was different: I thought of MetaTrader 5 not as a platform, but as a simple program in which other processes would run.

This flaw in the way I цыф seeing MetaTrader 5 is due to the fact that other platforms do not give us the same flexibility as MetaTrader 5. Because of this, I lost some time and speed when developing applications in a more advanced way. However, since this replay/simulator system has proven to be more appropriate to be developed in a modular way (different from what many typically do), I ran into some problems. But it wasn't exactly the problems, it was something that I was ignoring.

You can see in the videos presented in the previous article that MetaTrader 5 offers much more than many of have explored. But today we're going to turn what was once just a dream into something achievable. We'll start programming less and building more. Starting with this article, we will explore the exchange of messages in order to make MetaTrader 5 work harder for us, focusing only on making the application work harmoniously with other things present on the chart.

The first thing we will do is adapt the mouse indicator to start this new phase in MetaTrader 5 application development.

Since the code will undergo quite significant changes, many parts of it will have to be changed. However, if you have been following all the steps, you will have no trouble making these changes.

Therefore, we will immediately create a common file for all applications that will appear in the future. This will allow us to address messages that can be handled by any code we create from now on. We will have control over what happens when two applications that have a message handler are on the chart. This way, every application will be able to process the message correctly.

The initial contents of this file are shown below. It should be saved under the name Defines.mqh. Its location will be shown soon. So if you know absolutely nothing about programming, I'm sorry, but from now on you won't be able to follow what I'm going to implement. From this moment on, there is a barrier in front of you that prevents you from continuing. If you really want to use what we'll cover today, you'll have to have some basic programming knowledge.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #define def_VERSION_DEBUG
05. //+------------------------------------------------------------------+
06. #ifdef def_VERSION_DEBUG
07.     #define macro_DEBUG_MODE(A) \
08.                             Print(__FILE__, " ", __LINE__, " ", __FUNCTION__ + " " + #A + " = " + (string)(A));
09. #else
10.     #define macro_DEBUG_MODE(A)
11. #endif
12. //+------------------------------------------------------------------+
13. #define def_SymbolReplay            "RePlay"
14. #define def_MaxPosSlider            400
15. //+------------------------------------------------------------------+
16. union uCast_Double
17. {
18.     double   dValue;
19.     long     _long;                                  // 1 Information
20.     datetime _datetime;                              // 1 Information
21.     int      _int[sizeof(double) / sizeof(int)];     // 2 Informations
22.     char     _char[sizeof(double) / sizeof(char)];   // 8 Informations
23. };
24. //+------------------------------------------------------------------+
25. enum EnumEvents    {
26.                     evHideMouse,        //Hide mouse price line
27.                     evShowMouse,        //Show mouse price line
28.                     evHideBarTime,      //Hide bar time
29.                     evShowBarTime,      //Show bar time
30.                     evHideDailyVar,     //Hide daily variation
31.                     evShowDailyVar,     //Show daily variation
32.                     evHidePriceVar,     //Hide instantaneous variation
33.                     evShowPriceVar,     //Show instantaneous variation
34.                     evSetServerTime,    //Replay/simulation system timer
35.                     evCtrlReplayInit    //Initialize replay control
36.                    };
37. //+------------------------------------------------------------------+
```

Defines.mqh file source code

In line 4 of this file you can see that we have defined something. So, if you want to debug your code at runtime, add the \_DEBUG\_MODE macro to it, passing the variable you want to debug as a parameter. Then you can quickly add and remove the system to analyze at runtime what is happening. Just check if the compiler uses line 4.

In line 13, we define the name that will be used in our custom symbol and that will be used by the replay/simulator service, as we have done since the beginning of this article series. In line 14, we declare something that will be used only by the control indicator.

These two lines, as well as what is between lines 16 and 23, were already part of the code that had been written previously. However, since the InterProcess.mqh header file will no longer exist, it was necessary to move this information into the definition file.

What we're really interested in in this file starts from line 25. We are announcing a list here that will grow as new events are introduced and added. There is a small danger, but it is not that serious if you take the appropriate precautions: always add an enumeration to the end of the list. If you do this, then you won't have to recompile old codes, but if you add something in the middle of the list, then you will have to recompile all the old codes. This will help avoid problems.

Note that in each line we define a value and give a short comment about the corresponding event.

You will see that as the codes are displayed, they will indicate where and how each of these events will occur. But the main thing will be something that will be visible only in the code. So, if you plan to use any of what I am showing, take a look at the message handlers.

And with this we start making the whole system completely modular. From now on, pay close attention to the code and what will be present on the chart: it's time to use MetaTrader 5 like a real professional.

As you already know, it was necessary to change the code of the mouse indicator classes again, but these changes are necessary in order to facilitate the correct use and implementation of what we will do. Before changing the mouse indicator code, we need to make a small change to the C\_Terminal class code. Therefore, in the header file C\_Terminal.mqh, we need to do as shown in the following fragment.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #include "Macros.mqh"
05. #include "..\Defines.mqh"
06. #include "Interprocess.mqh"
07. //+------------------------------------------------------------------+
08. class C_Terminal
09. {
10. //+------------------------------------------------------------------+
11.     protected:
12.             enum eErrUser {ERR_Unknown, ERR_FileAcess, ERR_PointerInvalid, ERR_NoMoreInstance};
13. //+------------------------------------------------------------------+
14.             struct st_Terminal
```

Code from the C\_Terminal.mqh file

Look closely at line 5, which specifies the location of the Defines.mqh header file mentioned above relative to the C\_Terminal.mqh header file. But that's not all I want you to pay attention to. Note that line 6 has been removed from the code, meaning you can now remove the InterProcess.mqh header file from the project as it will no longer be used.

Since this change is fairly simple and no other changes were made to the C\_Terminal.mqh file code, I don't see any need to duplicate the entire file. While there are no major changes, one needs to be mentioned otherwise you will have problems when trying to compile the codes that will be displayed.

Line 12 contains an enumeration that has been assigned a new value. It should be used during testing to check if the same indicator is already present on the chart. Therefore, it is necessary to make one more small change to the same header file C\_Terminal.mqh. This change is shown in the following code.

```
157. //+------------------------------------------------------------------+
158.            bool IndicatorCheckPass(const string szShortName)
159.                    {
160.                            string szTmp = szShortName + "_TMP";
161.
162.                            if (_LastError != ERR_SUCCESS) return false;
163.                            IndicatorSetString(INDICATOR_SHORTNAME, szTmp);
164.                            if (ChartWindowFind(m_Infos.ID, szShortName) != -1)
165.                            {
166.                                    ChartIndicatorDelete(m_Infos.ID, 0, szTmp);
167.                                    Print("Only one instance is allowed...");
168.                                    SetUserError(C_Terminal::ERR_NoMoreInstance);
169.
170.                                    return false;
171.                            }
172.                            IndicatorSetString(INDICATOR_SHORTNAME, szShortName);
173.                            ResetLastError();
174.
175.                            return true;
176.                    }
177. //+------------------------------------------------------------------+
```

The pat that needs to be changed in the C\_Terminal.mqh file

We will need to replace change the original function present in the C\_Terminal.mqh file with the code shown in fragment 2. Thus, during testing, the system will correctly set the \_LastError constant, indicating that an error occurred. This error was caused by the presence of another instance of the same indicator on the chart. Other than these two simple changes, no other modifications were made, so we can move on and start developing the mouse indicator classes.

Below is the complete code for the C\_Mouse.mqh header file. Since understanding the code properly can be difficult for some, I'll briefly explain what's going on. Then you can better see how everything starts to become modular.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "C_Terminal.mqh"
005. //+------------------------------------------------------------------+
006. #define def_MousePrefixName "MouseBase_"
007. #define def_NameObjectLineH def_MousePrefixName + "H"
008. #define def_NameObjectLineV def_MousePrefixName + "TV"
009. #define def_NameObjectLineT def_MousePrefixName + "TT"
010. #define def_NameObjectStudy def_MousePrefixName + "TB"
011. //+------------------------------------------------------------------+
012. class C_Mouse : public C_Terminal
013. {
014.    public  :
015.            enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
016.            enum eBtnMouse {eKeyNull = 0x00, eClickLeft = 0x01, eClickRight = 0x02, eSHIFT_Press = 0x04, eCTRL_Press = 0x08, eClickMiddle = 0x10};
017.            struct st_Mouse
018.            {
019.                    struct st00
020.                    {
021.                            int      X_Adjusted,
022.                                     Y_Adjusted,
023.                                     X_Graphics,
024.                                     Y_Graphics;
025.                            double   Price;
026.                            datetime dt;
027.                    }Position;
028.                    uint     ButtonStatus;
029.                    bool     ExecStudy;
030.                    datetime TimeDevice;
031.            };
032. //+------------------------------------------------------------------+
033.    protected:
034. //+------------------------------------------------------------------+
035.            void CreateObjToStudy(int x, int w, string szName, color backColor = clrNONE) const
036.                    {
037.                            if (m_Mem.szShortName != NULL) return;
038.                            CreateObjectGraphics(szName, OBJ_BUTTON, clrNONE);
039.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_STATE, true);
040.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BORDER_COLOR, clrBlack);
041.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_COLOR, clrBlack);
042.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BGCOLOR, backColor);
043.                            ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_FONT, "Lucida Console");
044.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_FONTSIZE, 10);
045.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
046.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XDISTANCE, x);
047.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YDISTANCE, TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT) + 1);
048.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XSIZE, w);
049.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YSIZE, 18);
050.                    }
051. //+------------------------------------------------------------------+
052.    private :
053.            enum eStudy {eStudyNull, eStudyCreate, eStudyExecute};
054.            struct st01
055.            {
056.                    st_Mouse Data;
057.                    color    corLineH,
058.                             corTrendP,
059.                             corTrendN;
060.                    eStudy   Study;
061.            }m_Info;
062.            struct st_Mem
063.            {
064.                    bool     CrossHair,
065.                             IsFull;
066.                    datetime dt;
067.                    string   szShortName;
068.            }m_Mem;
069.            bool m_OK;
070. //+------------------------------------------------------------------+
071.            void GetDimensionText(const string szArg, int &w, int &h)
072.                    {
073.                            TextSetFont("Lucida Console", -100, FW_NORMAL);
074.                            TextGetSize(szArg, w, h);
075.                            h += 5;
076.                            w += 5;
077.                    }
078. //+------------------------------------------------------------------+
079.            void CreateStudy(void)
080.                    {
081.                            if (m_Mem.IsFull)
082.                            {
083.                                    CreateObjectGraphics(def_NameObjectLineV, OBJ_VLINE, m_Info.corLineH);
084.                                    CreateObjectGraphics(def_NameObjectLineT, OBJ_TREND, m_Info.corLineH);
085.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_WIDTH, 2);
086.                                    CreateObjToStudy(0, 0, def_NameObjectStudy);
087.                            }
088.                            m_Info.Study = eStudyCreate;
089.                    }
090. //+------------------------------------------------------------------+
091.            void ExecuteStudy(const double memPrice)
092.                    {
093.                            double v1 = GetInfoMouse().Position.Price - memPrice;
094.                            int w, h;
095.
096.                            if (!CheckClick(eClickLeft))
097.                            {
098.                                    m_Info.Study = eStudyNull;
099.                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
100.                                    if (m_Mem.IsFull) ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName + "T");
101.                            }else if (m_Mem.IsFull)
102.                            {
103.                                    string sz1 = StringFormat(" %." + (string)GetInfoTerminal().nDigits + "f [ %d ] %02.02f%% ",
104.                                            MathAbs(v1), Bars(GetInfoTerminal().szSymbol, PERIOD_CURRENT, m_Mem.dt, GetInfoMouse().Position.dt) - 1, MathAbs((v1 / memPrice) * 100.0)));
105.                                    GetDimensionText(sz1, w, h);
106.                                    ObjectSetString(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_TEXT, sz1);
107.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corTrendN : m_Info.corTrendP));
108.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_XSIZE, w);
109.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_YSIZE, h);
110.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_XDISTANCE, GetInfoMouse().Position.X_Adjusted - w);
111.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - (v1 < 0 ? 1 : h));
112.                                    ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 1, GetInfoMouse().Position.dt, GetInfoMouse().Position.Price);
113.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_COLOR, (memPrice > GetInfoMouse().Position.Price ? m_Info.corTrendN : m_Info.corTrendP));
114.                            }
115.                            m_Info.Data.ButtonStatus = eKeyNull;
116.                    }
117. //+------------------------------------------------------------------+
118.    public  :
119. //+------------------------------------------------------------------+
120.            C_Mouse(const long id, const string szShortName)
121.                    :C_Terminal(id),
122.                    m_OK(false)
123.                    {
124.                            m_Mem.szShortName = szShortName;
125.                    }
126. //+------------------------------------------------------------------+
127.            C_Mouse(const long id, const string szShortName, color corH, color corP, color corN)
128.                    :C_Terminal(id)
129.                    {
130.                            if (!(m_OK = IndicatorCheckPass(szShortName))) SetUserError(C_Terminal::ERR_Unknown);
131.                            if (_LastError != ERR_SUCCESS) return;
132.                            m_Mem.szShortName = NULL;
133.                            m_Mem.CrossHair = (bool)ChartGetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL);
134.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, true);
135.                            ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, false);
136.                            ZeroMemory(m_Info);
137.                            m_Info.corLineH  = corH;
138.                            m_Info.corTrendP = corP;
139.                            m_Info.corTrendN = corN;
140.                            m_Info.Study = eStudyNull;
141.                            if (m_Mem.IsFull = (corP != clrNONE) && (corH != clrNONE) && (corN != clrNONE))
142.                                    CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
143.                    }
144. //+------------------------------------------------------------------+
145.            ~C_Mouse()
146.                    {
147.                            if (!m_OK) return;
148.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
149.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, false);
150.                            ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, m_Mem.CrossHair);
151.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName);
152.                    }
153. //+------------------------------------------------------------------+
154. inline bool CheckClick(const eBtnMouse value)
155.                    {
156.                            return (GetInfoMouse().ButtonStatus & value) == value;
157.                    }
158. //+------------------------------------------------------------------+
159. inline const st_Mouse GetInfoMouse(void)
160.                    {
161.                            if (m_Mem.szShortName != NULL)
162.                            {
163.                                    double Buff[];
164.                                    uCast_Double loc;
165.                                    int handle = ChartIndicatorGet(GetInfoTerminal().ID, 0, m_Mem.szShortName);
166.
167.                                    ZeroMemory(m_Info.Data);
168.                                    if (CopyBuffer(handle, 0, 0, 6, Buff) == 6)
169.                                    {
170.                                            m_Info.Data.Position.Price = Buff[0];
171.                                            loc.dValue = Buff[1];
172.                                            m_Info.Data.Position.dt = loc._datetime;
173.                                            loc.dValue = Buff[2];
174.                                            m_Info.Data.Position.X_Adjusted = loc._int[0];
175.                                            m_Info.Data.Position.Y_Adjusted = loc._int[1];
176.                                            loc.dValue = Buff[3];
177.                                            m_Info.Data.Position.X_Graphics = loc._int[0];
178.                                            m_Info.Data.Position.Y_Graphics = loc._int[1];
179.                                            loc.dValue = Buff[4];
180.                                            m_Info.Data.ButtonStatus = loc._char[0];
181.                                            m_Info.Data.TimeDevice = (datetime)Buff[5];
182.                                            IndicatorRelease(handle);
183.                                    }
184.                            }
185.
186.                            return m_Info.Data;
187.                    }
188. //+------------------------------------------------------------------+
189.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
190.                    {
191.                            int w = 0;
192.                            static double memPrice = 0;
193.
194.                            if (m_Mem.szShortName == NULL)
195.                            {
196.                                    C_Terminal::DispatchMessage(id, lparam, dparam, sparam);
197.                                    switch (id)
198.                                    {
199.                                            case (CHARTEVENT_CUSTOM + evHideMouse):
200.                                                    if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
201.                                                    break;
202.                                            case (CHARTEVENT_CUSTOM + evShowMouse):
203.                                                    if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, m_Info.corLineH);
204.                                                    break;
205.                                            case (CHARTEVENT_CUSTOM + evSetServerTime):
206.                                               m_Info.Data.TimeDevice = (datetime)dparam;
207.                                               break;
208.                                            case CHARTEVENT_MOUSE_MOVE:
209.                                                    ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X_Graphics = (int)lparam, m_Info.Data.Position.Y_Graphics = (int)dparam, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
210.                                                    if (m_Mem.IsFull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price = AdjustPrice(m_Info.Data.Position.Price));
211.                                                    m_Info.Data.Position.dt = AdjustTime(m_Info.Data.Position.dt);
212.                                                    ChartTimePriceToXY(GetInfoTerminal().ID, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price, m_Info.Data.Position.X_Adjusted, m_Info.Data.Position.Y_Adjusted);
213.                                                    if ((m_Info.Study != eStudyNull) && (m_Mem.IsFull)) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineV, 0, m_Info.Data.Position.dt, 0);
214.                                                    m_Info.Data.ButtonStatus = (uint) sparam;
215.                                                    if (CheckClick(eClickMiddle))
216.                                                            if ((!m_Mem.IsFull) || ((color)ObjectGetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR) != clrNONE)) CreateStudy();
217.                                                    if (CheckClick(eClickLeft) && (m_Info.Study == eStudyCreate))
218.                                                    {
219.                                                            ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
220.                                                            if (m_Mem.IsFull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 0, m_Mem.dt = GetInfoMouse().Position.dt, memPrice = GetInfoMouse().Position.Price);
221.                                                            m_Info.Study = eStudyExecute;
222.                                                    }
223.                                                    if (m_Info.Study == eStudyExecute) ExecuteStudy(memPrice);
224.                                                    m_Info.Data.ExecStudy = m_Info.Study == eStudyExecute;
225.                                                    break;
226.                                            case CHARTEVENT_OBJECT_DELETE:
227.                                                    if ((m_Mem.IsFull) && (sparam == def_NameObjectLineH)) CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
228.                                                    break;
229.                                    }
230.                            }
231.                    }
232. //+------------------------------------------------------------------+
233. };
234. //+------------------------------------------------------------------+
235. #undef def_NameObjectLineV
236. #undef def_NameObjectLineH
237. #undef def_NameObjectLineT
238. #undef def_NameObjectStudy
239. //+------------------------------------------------------------------+
```

Source code of the C\_Mouse.mqh file

Unlike previous versions, now only the C\_Terminal.mqh file is considered necessary. However, you should pay attention to other details that have been changed in this class, as shown above.

If you notice, we now have a new variable in line 30. It is only useful if the replay/simulator service is used. This variable gives us access to the value provided by the replay/simulator service, which was previously obtained through the global terminal variable. Although this variable is declared in line 30, it is used elsewhere. But because of another problem that we will explain later, we had to add it here, in the C\_Mouse class.

As you can see, most of the code remains identical to the previous one, until line 168, where we find something different. Now we will use six buffer positions. The sixth position should be occupied by the variable declared in line 30. So in line 181, we fill in the value when we need to read it and find out what the value of that variable is. Here I must stop for a second and explain something. The only procedure that will actualy use the contents of the variable declared in line 30 will be the replay/simulator service and the mouse indicator. The latter will use this value to inform the user about the time remaining until the opening of the next bar. However, if we don't want to use it, we can remove it from the mouse indicator or the replay/simulator service. There will be no problems if you do this. If you want to use this information, you should be aware that unless we use a global terminal variable or other means to store the time value in the replay/simulator service, that service will not know when this data should be updated again.

For this reason, we place the value in the mouse indicator buffer. But then you think: "Won't this value be deleted and reset when MetaTrader 5 restores the indicator on the chart after the user changes the chart timeframe?" Then storing it in the indicator buffer makes no sense. In fact, this is exactly what we want. When the service detects that the value in the buffer is no longer valid, it will send an event to the mouse indicator to update the value again. This way we will keep everything as expected.

That is, the fact that MetaTrader 5 forces the indicator to zero the buffer makes the replay/simulator service understand that the user has made some change that needs to be re-evaluated by the replay/simulator service. This way, our application will easily detect that something has happened, and MetaTrader 5 will do everything for us.

But the question becomes even more interesting if we look at the code starting from line 189, where the C\_Mouse class message handler is located.

In this message handling procedure, we have the first three messages that our modular system will start processing. So pay attention to what is happening. First, to make sure that the message handling is done by the indicator and not by some other code that uses the class, we must check whether we are calling exactly the indicator. This check is performed in line 194. If the check is successful, we consider that we are dealing with an indicator, so we only need to have one mouse indicator on the chart to avoid potential conflict of interest. I've covered this in other articles before, so let's move further.

In line 199, we handle the event where the mouse indicator should hide the line used as the price line.

In line 202, we handle the event that tells the mouse indicator that the price line should be displayed on the chart again. This way, any application will be able to tell the mouse indicator when to show or hide the price line it is using.

In both events there is no need to specify any additional parameters, so any application that wants to show or hide the mouse line just needs to generate a custom event with the specified values and direct this event to the chart where the mouse indicator is located. I'll explain more about this later. For now, we can understand it this way: if at some point a mouse indicator appears on the chart and you want to hide the mouse line, then in order to hide or show the mouse line, you need to generate a custom event with the desired value. This event can be triggered, for example, by an Expert Advisor.

The third event, which we will also implement here, is shown in line 205. In this case, we specify the value that should be placed as the service running time, i.e. it is the replay/simulator service that will actually do this. Because only the service can take some advantage in generating such an event. But there is one important point here: when this event is fired, it must specify a time value. This value must be in the 'dparam' parameter, which is a double value.

Again, you need to look at things from a broader perspective. A double value consists of 8 bytes, just like a datetime value, which also consists of 8 bytes. So we perform type conversion so that the compiler understands what we are doing. But to the processor, all we do is put 8 bytes into a variable. The contents of these bytes do not matter.

It is important to understand this, because there are situations when we need to pass entire strings of values, sometimes entire structures, and since in MetaTrader 5, or more precisely in MQL5, we cannot use pointers like in C/C++, we will need some trick to pass this data between our applications running in MetaTrader 5.

Okay, the first part of the job is done. But we are still going to make things better. If you've been paying attention, you've probably noticed that there are other events associated with the mouse indicator. However, these events are not in the header file C\_Mouse.mqh, but in the file C\_Study.mqh. To see these events and understand what will happen, let's look at this code. The full code is shown below:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\C_Mouse.mqh"
005. //+------------------------------------------------------------------+
006. #define def_ExpansionPrefix def_MousePrefixName + "Expansion_"
007. #define def_ExpansionBtn1 def_ExpansionPrefix + "B1"
008. #define def_ExpansionBtn2 def_ExpansionPrefix + "B2"
009. #define def_ExpansionBtn3 def_ExpansionPrefix + "B3"
010. //+------------------------------------------------------------------+
011. class C_Study : public C_Mouse
012. {
013.    private :
014. //+------------------------------------------------------------------+
015.            struct st00
016.            {
017.                    eStatusMarket  Status;
018.                    MqlRates       Rate;
019.                    string         szInfo;
020.                    color          corP,
021.                                   corN;
022.                    int            HeightText;
023.                    bool           bvT, bvD, bvP;
024.            }m_Info;
025. //+------------------------------------------------------------------+
026.            const datetime GetBarTime(void)
027.                    {
028.                            datetime dt;
029.                            int i0 = PeriodSeconds();
030.
031.                            if (m_Info.Status == eInReplay)
032.                            {
033.                                    if ((dt = GetInfoMouse().TimeDevice) == ULONG_MAX) return ULONG_MAX;
034.                            }else dt = TimeCurrent();
035.                            if (m_Info.Rate.time <= dt)
036.                                    m_Info.Rate.time = (datetime)(((ulong) dt / i0) * i0) + i0;
037.
038.                            return m_Info.Rate.time - dt;
039.                    }
040. //+------------------------------------------------------------------+
041.            void Draw(void)
042.                    {
043.                            double v1;
044.
045.                            if (m_Info.bvT)
046.                            {
047.                                    ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn1, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 18);
048.                                    ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn1, OBJPROP_TEXT, m_Info.szInfo);
049.                            }
050.                            if (m_Info.bvD)
051.                            {
052.                                    v1 = NormalizeDouble(100.0 - ((m_Info.Rate.close / GetInfoMouse().Position.Price) * 100.0), 2);
053.                                    ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 1);
054.                                    ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
055.                                    ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_TEXT, StringFormat("%.2f%%", MathAbs(v1)));
056.                            }
057.                            if (m_Info.bvP)
058.                            {
059.                                    v1 = NormalizeDouble(100.0 - ((m_Info.Rate.close / iClose(GetInfoTerminal().szSymbol, PERIOD_D1, 0)) * 100.0), 2);
060.                                    ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 1);
061.                                    ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
062.                                    ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_TEXT, StringFormat("%.2f%%", MathAbs(v1)));
063.                            }
064.                    }
065. //+------------------------------------------------------------------+
066. inline void CreateObjInfo(EnumEvents arg)
067.                    {
068.                            switch (arg)
069.                            {
070.                                    case evShowBarTime:
071.                                            C_Mouse::CreateObjToStudy(2, 110, def_ExpansionBtn1, clrPaleTurquoise);
072.                                            m_Info.bvT = true;
073.                                            break;
074.                                    case evShowDailyVar:
075.                                            C_Mouse::CreateObjToStudy(2, 53, def_ExpansionBtn2);
076.                                            m_Info.bvD = true;
077.                                            break;
078.                                    case evShowPriceVar:
079.                                            C_Mouse::CreateObjToStudy(58, 53, def_ExpansionBtn3);
080.                                            m_Info.bvP = true;
081.                                            break;
082.                            }
083.                    }
084. //+------------------------------------------------------------------+
085. inline void RemoveObjInfo(EnumEvents arg)
086.                    {
087.                            string sz;
088.
089.                            switch (arg)
090.                            {
091.                                    case evHideBarTime:
092.                                            sz = def_ExpansionBtn1;
093.                                            m_Info.bvT = false;
094.                                            break;
095.                                    case evHideDailyVar:
096.                                            sz = def_ExpansionBtn2;
097.                                            m_Info.bvD      = false;
098.                                            break;
099.                                    case evHidePriceVar:
100.                                            sz = def_ExpansionBtn3;
101.                                            m_Info.bvP = false;
102.                                            break;
103.                            }
104.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
105.                            ObjectDelete(GetInfoTerminal().ID, sz);
106.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
107.                    }
108. //+------------------------------------------------------------------+
109.    public  :
110. //+------------------------------------------------------------------+
111.            C_Study(long IdParam, string szShortName, color corH, color corP, color corN)
112.                    :C_Mouse(IdParam, szShortName, corH, corP, corN)
113.                    {
114.                            if (_LastError != ERR_SUCCESS) return;
115.                            ZeroMemory(m_Info);
116.                            m_Info.Status = eCloseMarket;
117.                            m_Info.Rate.close = iClose(GetInfoTerminal().szSymbol, PERIOD_D1, ((GetInfoTerminal().szSymbol == def_SymbolReplay) || (macroGetDate(TimeCurrent()) != macroGetDate(iTime(GetInfoTerminal().szSymbol, PERIOD_D1, 0))) ? 0 : 1));
118.                            m_Info.corP = corP;
119.                            m_Info.corN = corN;
120.                            CreateObjInfo(evShowBarTime);
121.                            CreateObjInfo(evShowDailyVar);
122.                            CreateObjInfo(evShowPriceVar);
123.                    }
124. //+------------------------------------------------------------------+
125.            void Update(const eStatusMarket arg)
126.                    {
127.                            datetime dt;
128.
129.                            switch (m_Info.Status = (m_Info.Status != arg ? arg : m_Info.Status))
130.                            {
131.                                    case eCloseMarket :
132.                                            m_Info.szInfo = "Closed Market";
133.                                            break;
134.                                    case eInReplay    :
135.                                    case eInTrading   :
136.                                            if ((dt = GetBarTime()) < ULONG_MAX)
137.                                            {
138.                                                    m_Info.szInfo = TimeToString(dt, TIME_SECONDS);
139.                                                    break;
140.                                            }
141.                                    case eAuction     :
142.                                            m_Info.szInfo = "Auction";
143.                                            break;
144.                                    default           :
145.                                            m_Info.szInfo = "ERROR";
146.                            }
147.                            Draw();
148.                    }
149. //+------------------------------------------------------------------+
150. virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
151.                    {
152.                            C_Mouse::DispatchMessage(id, lparam, dparam, sparam);
153.                            switch (id)
154.                            {
155.                                    case CHARTEVENT_CUSTOM + evHideBarTime:
156.                                            RemoveObjInfo(evHideBarTime);
157.                                            break;
158.                                    case CHARTEVENT_CUSTOM + evShowBarTime:
159.                                            CreateObjInfo(evShowBarTime);
160.                                            break;
161.                                    case CHARTEVENT_CUSTOM + evHideDailyVar:
162.                                            RemoveObjInfo(evHideDailyVar);
163.                                            break;
164.                                    case CHARTEVENT_CUSTOM + evShowDailyVar:
165.                                            CreateObjInfo(evShowDailyVar);
166.                                            break;
167.                                    case CHARTEVENT_CUSTOM + evHidePriceVar:
168.                                            RemoveObjInfo(evHidePriceVar);
169.                                            break;
170.                                    case CHARTEVENT_CUSTOM + evShowPriceVar:
171.                                            CreateObjInfo(evShowPriceVar);
172.                                            break;
173.                                    case CHARTEVENT_MOUSE_MOVE:
174.                                            Draw();
175.                                            break;
176.                            }
177.                            ChartRedraw(GetInfoTerminal().ID);
178.                    }
179. //+------------------------------------------------------------------+
180. };
181. //+------------------------------------------------------------------+
182. #undef def_ExpansionBtn3
183. #undef def_ExpansionBtn2
184. #undef def_ExpansionBtn1
185. #undef def_ExpansionPrefix
186. #undef def_MousePrefixName
187. //+------------------------------------------------------------------+
```

C\_Study.mqh file source code

Unlike C\_Mouse.mqh, there are many differences here. Let's start with the fact that we have new variables, checks and other things that are now being implemented. But since most of them are simple, we will only dwell on a few of the most "unusual" ones. Among them is line 33, where we see that the function to access the global variable on the terminal is no longer used. Now we ask the mouse indicator for something that was previously searched for in the terminal's global variable. This is exactly how we want to do things: we want to do things without using something that the user can control and without resorting to external programming. The idea is to implement everything in pure MQL5.

Looking at the code, it may not be entirely clear that our implementation has a lot more settings than what was before. We did this to make our mouse indicator a kind of standard that can be used in other applications on the platform. There is one important point here: as a programmer, you can send messages to the mouse indicator to turn on or off something already defined in it.

To do this, we need to isolate certain parts of the code so that when there is an event that requires something to be turned on or off, the object we are accessing undergoes the changes we want to make. In this case, we are talking about something simple, like how to make something appear or not appear on the chart. Well, there could have been something even more complicated. The degree of freedom we gradually introduce makes many things possible. To implement them, you will just need to make minor modifications of what is already ready. This also means that the level of security and reliability of the application will always remain the highest possible.

Most of the code you see in the C\_Study.mqh file does exactly this. But now we're really interested in what happens from line 150 onwards, where we'll implement what's described in this header file. Note that the outstanding parts we are not interested in handling here are passed to the C\_Mouse class so that events can be handled there. This can be seen from line 152.

Now pay attention to one thing. Each of the user events that can be seen in this handler will turn something on or off in the mouse indicator, making it look a little different, but it will do so at runtime, without the need for the user to recompile the code or configure a large list of options. Please note this fact. You can simply create small script files that will send custom events to the mouse indicator so that its appearance changes during use.

If you are creative enough, you will already be thinking and planning different tricks just by looking at this code and how it works. But I advise you not to rush, because at this stage it is not yet entirely clear in which direction the replay/simulator system will develop. This is because turning different elements into modules opens up a wide range of possibilities. And the way the system can grow forces us to take some care when we think about new possibilities. But for now I'm focused on getting the system to start working without using terminal global variables, while still maintaining the same qualities it had when the modules were first created.

All right. Now that all of this is shown, we can move on to the code that actually creates the mouse indicator. You can see it below.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. #property description "This is an indicator for graphical studies using the mouse."
004. #property description "This is an integral part of the Replay / Simulator system."
005. #property description "However it can be used in the real market."
006. #property version "1.54"
007. #property icon "/Images/Market Replay/Icons/Indicators.ico"
008. #property link "https://www.mql5.com/pt/articles/11971"
009. #property indicator_chart_window
010. #property indicator_plots 0
011. #property indicator_buffers 1
012. //+------------------------------------------------------------------+
013. #include <Market Replay\Auxiliar\Study\C_Study.mqh>
014. //+------------------------------------------------------------------+
015. C_Study *Study     = NULL;
016. //+------------------------------------------------------------------+
017. input long user00  = 0;                                    //ID
018. input C_Study::eStatusMarket user01 = C_Study::eAuction;   //Market Status
019. input color user02 = clrBlack;                             //Price Line
020. input color user03 = clrPaleGreen;                         //Positive Study
021. input color user04 = clrLightCoral;                        //Negative Study
022. //+------------------------------------------------------------------+
023. C_Study::eStatusMarket m_Status;
024. int m_posBuff = 0;
025. double m_Buff[];
026. //+------------------------------------------------------------------+
027. int OnInit()
028. {
029.    ResetLastError();
030.    Study = new C_Study(user00, "Indicator Mouse Study", user02, user03, user04);
031.    if (_LastError != ERR_SUCCESS) return INIT_FAILED;
032.    if ((*Study).GetInfoTerminal().szSymbol != def_SymbolReplay)
033.    {
034.            MarketBookAdd((*Study).GetInfoTerminal().szSymbol);
035.            OnBookEvent((*Study).GetInfoTerminal().szSymbol);
036.            m_Status = C_Study::eCloseMarket;
037.    }else
038.            m_Status = user01;
039.    SetIndexBuffer(0, m_Buff, INDICATOR_DATA);
040.    ArrayInitialize(m_Buff, EMPTY_VALUE);
041.
042.    return INIT_SUCCEEDED;
043. }
044. //+------------------------------------------------------------------+
045. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
046. {
047.    m_posBuff = rates_total - 6;
048.    (*Study).Update(m_Status);
049.
050.    return rates_total;
051. }
052. //+------------------------------------------------------------------+
053. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
054. {
055.    (*Study).DispatchMessage(id, lparam, dparam, sparam);
056.    SetBuffer();
057.
058.    ChartRedraw((*Study).GetInfoTerminal().ID);
059. }
060. //+------------------------------------------------------------------+
061. void OnBookEvent(const string &symbol)
062. {
063.    MqlBookInfo book[];
064.    C_Study::eStatusMarket loc = m_Status;
065.
066.    if (symbol != (*Study).GetInfoTerminal().szSymbol) return;
067.    MarketBookGet((*Study).GetInfoTerminal().szSymbol, book);
068.    m_Status = (ArraySize(book) == 0 ? C_Study::eCloseMarket : C_Study::eInTrading);
069.    for (int c0 = 0; (c0 < ArraySize(book)) && (m_Status != C_Study::eAuction); c0++)
070.            if ((book[c0].type == BOOK_TYPE_BUY_MARKET) || (book[c0].type == BOOK_TYPE_SELL_MARKET)) m_Status = C_Study::eAuction;
071.    if (loc != m_Status) (*Study).Update(m_Status);
072. }
073. //+------------------------------------------------------------------+
074. void OnDeinit(const int reason)
075. {
076.    if (reason != REASON_INITFAILED)
077.    {
078.            if ((*Study).GetInfoTerminal().szSymbol != def_SymbolReplay)
079.                    MarketBookRelease((*Study).GetInfoTerminal().szSymbol);
080.    }
081.    delete Study;
082. }
083. //+------------------------------------------------------------------+
084. inline void SetBuffer(void)
085. {
086.    uCast_Double Info;
087.
088.    m_posBuff = (m_posBuff < 0 ? 0 : m_posBuff);
089.    m_Buff[m_posBuff + 0] = (*Study).GetInfoMouse().Position.Price;
090.    Info._datetime = (*Study).GetInfoMouse().Position.dt;
091.    m_Buff[m_posBuff + 1] = Info.dValue;
092.    Info._int[0] = (*Study).GetInfoMouse().Position.X_Adjusted;
093.    Info._int[1] = (*Study).GetInfoMouse().Position.Y_Adjusted;
094.    m_Buff[m_posBuff + 2] = Info.dValue;
095.    Info._int[0] = (*Study).GetInfoMouse().Position.X_Graphics;
096.    Info._int[1] = (*Study).GetInfoMouse().Position.Y_Graphics;
097.    m_Buff[m_posBuff + 3] = Info.dValue;
098.    Info._char[0] = ((*Study).GetInfoMouse().ExecStudy == C_Mouse::eStudyNull ? (char)(*Study).GetInfoMouse().ButtonStatus : 0);
099.    m_Buff[m_posBuff + 4] = Info.dValue;
100.    m_Buff[m_posBuff + 5] = (double)(*Study).GetInfoMouse().TimeDevice;
101. }
102. //+------------------------------------------------------------------+
```

Source code of the mouse indicator

The code has not undergone significant changes. There is only one new addition, which can be seen in line 100, where the data is placed into the indicator buffer. Again, this data is only useful in the early stages for the replay/simulator service. It has no other use. At least I didn't notice any others.

### Conclusion

To help you understand what's going on here, I'll leave the indicator executable attached, as well as a few scripts that will be used to modify the indicator when it's on the chart. But there are also those who do not want to run an already compiled program on their platform. Well, I understand why. In the video at the end of this article you can see what happens when scripts are executed.

Please note that this is only a small part of what we are capable of. Those who take a narrower view will say that this is nonsense and could have been done differently, but I will not pay attention to them. I prefer to see the sparkle in the eyes of those who think about the possibilities of what I am showing: what this modular system can become and how everything will develop further. It is for them that I write these articles. In them, I show that many are only scratching the surface of what we can actually do with MQL5, and that MetaTrader 5 is actually a great platform, and in the right hands it can do incredible things.

However, I apologize again to those who don't actually have programming skills. I'm sorry, but without proper knowledge of what has been shown here, access to the source code of this replay/simulator service becomes something dangerous. But I promise not to leave you alone with the problem, I will make the executable available to you as soon as the module version is finished and becomes stable, and will provide the executable here in the articles. However, if you want to compile the source code, it will always be available in the article. However, I know that without the proper knowledge you will not be able to create the directory structure needed to compile the system.

This is exactly what I wanted to do, because I have seen from my own experience that in previous articles people were trying to compile something without understanding what they were talking about. Such things are not only risky but also dangerous. You shouldn't use things without understanding their purpose.

YouTube

Video 01 - Demonstration of the operation of the module

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11971](https://www.mql5.com/pt/articles/11971)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11971.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11971/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/478188)**

![Ensemble methods to enhance numerical predictions in MQL5](https://c.mql5.com/2/105/logo-ensemble_methods_to_enhance_numerical_predictions-2.png)[Ensemble methods to enhance numerical predictions in MQL5](https://www.mql5.com/en/articles/16630)

In this article, we present the implementation of several ensemble learning methods in MQL5 and examine their effectiveness across different scenarios.

![Portfolio Risk Model using Kelly Criterion and Monte Carlo Simulation](https://c.mql5.com/2/103/banner3_resized.png)[Portfolio Risk Model using Kelly Criterion and Monte Carlo Simulation](https://www.mql5.com/en/articles/16500)

For decades, traders have been using the Kelly Criterion formula to determine the optimal proportion of capital to allocate to an investment or bet to maximize long-term growth while minimizing the risk of ruin. However, blindly following Kelly Criterion using the result of a single backtest is often dangerous for individual traders, as in live trading, trading edge diminishes over time, and past performance is no predictor of future result. In this article, I will present a realistic approach to applying the Kelly Criterion for one or more EA's risk allocation in MetaTrader 5, incorporating Monte Carlo simulation results from Python.

![Price Action Analysis Toolkit Development (Part 5): Volatility Navigator EA](https://c.mql5.com/2/105/Price_Action_Analysis_Toolkit_Development_Part_5___LOGO.png)[Price Action Analysis Toolkit Development (Part 5): Volatility Navigator EA](https://www.mql5.com/en/articles/16560)

Determining market direction can be straightforward, but knowing when to enter can be challenging. As part of the series titled "Price Action Analysis Toolkit Development", I am excited to introduce another tool that provides entry points, take profit levels, and stop loss placements. To achieve this, we have utilized the MQL5 programming language. Let’s delve into each step in this article.

![Neural Network in Practice: Pseudoinverse (I)](https://c.mql5.com/2/81/Rede_neural_na_prztica__Pseudo_Inversa___LOGO.png)[Neural Network in Practice: Pseudoinverse (I)](https://www.mql5.com/en/articles/13710)

Today we will begin to consider how to implement the calculation of pseudo-inverse in pure MQL5 language. The code we are going to look at will be much more complex for beginners than I expected, and I'm still figuring out how to explain it in a simple way. So for now, consider this an opportunity to learn some unusual code. Calmly and attentively. Although it is not aimed at efficient or quick application, its goal is to be as didactic as possible.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=tjrnayltgvyigcpdgxvcahhtiuuatewj&ssn=1769184315772049292&ssn_dr=0&ssn_sr=0&fv_date=1769184314&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11971&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2054)%3A%20The%20Birth%20of%20the%20First%20Module%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918431504526481&fz_uniq=5069997009655500346&sv=2552)

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