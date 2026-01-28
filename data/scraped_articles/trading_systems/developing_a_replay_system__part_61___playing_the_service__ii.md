---
title: Developing a Replay System (Part 61): Playing the service (II)
url: https://www.mql5.com/en/articles/12121
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:39:54.346998
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/12121&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069637576727398471)

MetaTrader 5 / Examples


### Introduction

In the previous article, [Developing a Replay System (Part 60): Playing the service (I)](https://www.mql5.com/en/articles/12086), we made some adjustments to enable the replay/simulator service to start generating new data on the chart. Although we made minimal changes to allow the system to begin launching data, it quickly became apparent that something unusual had occurred. Despite the absence of major modifications, the system seemed to have suffered a significant setback. This situation gives the impression that the system has become unviable, as it suddenly slowed down drastically. Is it really so? And if so, how can we resolve this issue? It is important to remember that we aim to keep everything aligned with object-oriented programming principles.

Although there was indeed a drop in performance, we can address most of this issue simply by understanding and properly adjusting certain aspects of the code. In this article, I may begin demonstrating how to use some of the tools available in MetaEditor, which greatly facilitate the process of refining and improving the code. In hindsight, I should have introduced this topic a few articles ago. However, I did not see the same level of necessity as I do now, when it is crucial to understand how the code operates and why its performance has degraded so significantly.

### Implementing the Most Evident and Direct Improvements

Misunderstandings or a lack of in-depth explanations regarding how MetaTrader 5 and MQL5 function often create significant obstacles in certain implementations. Fortunately, within the community, we can consolidate knowledge and share it effectively, even if it does not provide an immediate solution to our current implementation challenges. Regardless, having accurate and high-quality knowledge is always beneficial.

One of these key aspects is precisely what I will attempt to explain. Much of what I discuss is easier to grasp when you actively use MQL5, allowing you to accomplish far more within MetaTrader 5 than most developers typically achieve or attempt.

Perhaps one of the most misunderstood aspects for many MQL5 programmers is graphical objects. Many believe that these objects can only be accessed, manipulated, and adjusted through something that is directly present on the chart: an indicator, script, or even an Expert Advisor. However, this is far from the truth.

Up until now, we have worked in a way that avoids creating dependencies between what appears in the custom asset chart window and what is executed within MetaTrader 5. However, beyond the methods we are currently using to transfer information between applications running in MetaTrader 5, there is also the possibility of implementing a more sophisticated (yet riskier) approach. Do not misunderstand me; when dependencies are introduced between what is being executed and what we expect to be executed, unexpected issues can arise.

Even though this approach may work in many cases, it can lead us down a complex and problematic path, potentially resulting in wasted time that could be better spent elsewhere. The reason is that such changes often make it impossible to make further improvements or implement new features. To grasp what I am proposing, it is essential to have a solid understanding of how the system functions as a whole.

The first critical point to note is that the control indicator module will only be present on the chart if, and only if, the replay/simulation service is running. You should not attempt to manually add the control module to the chart, as doing so will disrupt everything we are about to implement.

The second key point is that all graphical objects created by the control module must follow a strict and consistent naming convention; otherwise, we will have serious issues later.

In addition to these two points, we will also implement changes that significantly enhance code readability. It is essential to avoid using symbols or markers that lack clear meaning. However, these readability improvements are primarily intended to make it easier to understand specific adjustments rather than to improve code execution speed. This will become clearer once we examine the source code.

The first modifications we will make will be to the C\_Controls.mqh header file. However, before delving into why these changes are necessary, let's first examine the modifications that have been made to this file. The new code is shown below:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\Auxiliar\C_DrawImage.mqh"
005. #include "..\Defines.mqh"
006. //+------------------------------------------------------------------+
007. #define def_PathBMP           "Images\\Market Replay\\Control\\"
008. #define def_ButtonPlay        def_PathBMP + "Play.bmp"
009. #define def_ButtonPause       def_PathBMP + "Pause.bmp"
010. #define def_ButtonLeft        def_PathBMP + "Left.bmp"
011. #define def_ButtonLeftBlock   def_PathBMP + "Left_Block.bmp"
012. #define def_ButtonRight       def_PathBMP + "Right.bmp"
013. #define def_ButtonRightBlock  def_PathBMP + "Right_Block.bmp"
014. #define def_ButtonPin         def_PathBMP + "Pin.bmp"
015. #resource "\\" + def_ButtonPlay
016. #resource "\\" + def_ButtonPause
017. #resource "\\" + def_ButtonLeft
018. #resource "\\" + def_ButtonLeftBlock
019. #resource "\\" + def_ButtonRight
020. #resource "\\" + def_ButtonRightBlock
021. #resource "\\" + def_ButtonPin
022. //+------------------------------------------------------------------+
023. #define def_ObjectCtrlName(A)   "MarketReplayCTRL_" + (typename(A) == "enum eObjectControl" ? EnumToString((C_Controls::eObjectControl)(A)) : (string)(A))
024. #define def_PosXObjects         120
025. //+------------------------------------------------------------------+
026. #define def_SizeButtons         32
027. #define def_ColorFilter         0xFF00FF
028. //+------------------------------------------------------------------+
029. #include "..\Auxiliar\C_Terminal.mqh"
030. #include "..\Auxiliar\C_Mouse.mqh"
031. //+------------------------------------------------------------------+
032. class C_Controls : private C_Terminal
033. {
034.    protected:
035.    private   :
036. //+------------------------------------------------------------------+
037.       enum eMatrixControl {eCtrlPosition, eCtrlStatus};
038.       enum eObjectControl {ePause, ePlay, eLeft, eRight, ePin, eNull, eTriState = (def_MaxPosSlider + 1)};
039. //+------------------------------------------------------------------+
040.       struct st_00
041.       {
042.          string  szBarSlider,
043.                  szBarSliderBlock;
044.          ushort  Minimal;
045.       }m_Slider;
046.       struct st_01
047.       {
048.          C_DrawImage *Btn;
049.          bool        state;
050.          short       x, y, w, h;
051.       }m_Section[eObjectControl::eNull];
052.       C_Mouse   *m_MousePtr;
053. //+------------------------------------------------------------------+
054. inline void CreteBarSlider(short x, short size)
055.          {
056.             ObjectCreate(GetInfoTerminal().ID, m_Slider.szBarSlider = def_ObjectCtrlName("B1"), OBJ_RECTANGLE_LABEL, 0, 0, 0);
057.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_XDISTANCE, def_PosXObjects + x);
058.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_YDISTANCE, m_Section[ePin].y + 11);
059.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_XSIZE, size);
060.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_YSIZE, 9);
061.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BGCOLOR, clrLightSkyBlue);
062.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BORDER_COLOR, clrBlack);
063.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_WIDTH, 3);
064.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BORDER_TYPE, BORDER_FLAT);
065.             ObjectCreate(GetInfoTerminal().ID, m_Slider.szBarSliderBlock = def_ObjectCtrlName("B2"), OBJ_RECTANGLE_LABEL, 0, 0, 0);
066.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XDISTANCE, def_PosXObjects + x);
067.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_YDISTANCE, m_Section[ePin].y + 6);
068.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_YSIZE, 19);
069.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_BGCOLOR, clrRosyBrown);
070.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_BORDER_TYPE, BORDER_RAISED);
071.          }
072. //+------------------------------------------------------------------+
073.       void SetPlay(bool state)
074.          {
075.             if (m_Section[ePlay].Btn == NULL)
076.                m_Section[ePlay].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_ObjectCtrlName(ePlay), def_ColorFilter, "::" + def_ButtonPause, "::" + def_ButtonPlay);
077.             m_Section[ePlay].Btn.Paint(m_Section[ePlay].x, m_Section[ePlay].y, m_Section[ePlay].w, m_Section[ePlay].h, 20, ((m_Section[ePlay].state = state) ? 1 : 0));
078.             if (!state) CreateCtrlSlider();
079.          }
080. //+------------------------------------------------------------------+
081.       void CreateCtrlSlider(void)
082.          {
083.             CreteBarSlider(77, 436);
084.             m_Section[eLeft].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_ObjectCtrlName(eLeft), def_ColorFilter, "::" + def_ButtonLeft, "::" + def_ButtonLeftBlock);
085.             m_Section[eRight].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_ObjectCtrlName(eRight), def_ColorFilter, "::" + def_ButtonRight, "::" + def_ButtonRightBlock);
086.             m_Section[ePin].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_ObjectCtrlName(ePin), def_ColorFilter, "::" + def_ButtonPin);
087.             PositionPinSlider(m_Slider.Minimal);
088.          }
089. //+------------------------------------------------------------------+
090. inline void RemoveCtrlSlider(void)
091.          {
092.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
093.             for (eObjectControl c0 = ePlay + 1; c0 < eNull; c0++)
094.             {
095.                delete m_Section[c0].Btn;
096.                m_Section[c0].Btn = NULL;
097.             }
098.             ObjectsDeleteAll(GetInfoTerminal().ID, def_ObjectCtrlName("B"));
099.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
100.          }
101. //+------------------------------------------------------------------+
102. inline void PositionPinSlider(ushort p)
103.          {
104.             int iL, iR;
105.
106.             m_Section[ePin].x = (short)(p < m_Slider.Minimal ? m_Slider.Minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
107.             iL = (m_Section[ePin].x != m_Slider.Minimal ? 0 : 1);
108.             iR = (m_Section[ePin].x < def_MaxPosSlider ? 0 : 1);
109.             m_Section[ePin].x += def_PosXObjects;
110.             m_Section[ePin].x += 95 - (def_SizeButtons / 2);
111.             for (eObjectControl c0 = ePlay + 1; c0 < eNull; c0++)
112.                m_Section[c0].Btn.Paint(m_Section[c0].x, m_Section[c0].y, m_Section[c0].w, m_Section[c0].h, 20, (c0 == eLeft ? iL : (c0 == eRight ? iR : 0)));
113.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, m_Slider.Minimal + 2);
114.          }
115. //+------------------------------------------------------------------+
116. inline eObjectControl CheckPositionMouseClick(short &x, short &y)
117.          {
118.             C_Mouse::st_Mouse InfoMouse;
119.
120.             InfoMouse = (*m_MousePtr).GetInfoMouse();
121.             x = (short) InfoMouse.Position.X_Graphics;
122.             y = (short) InfoMouse.Position.Y_Graphics;
123.             for (eObjectControl c0 = ePlay; c0 < eNull; c0++)
124.             {
125.                if ((m_Section[c0].Btn != NULL) && (m_Section[c0].x <= x) && (m_Section[c0].y <= y) && ((m_Section[c0].x + m_Section[c0].w) >= x) && ((m_Section[c0].y + m_Section[c0].h) >= y))
126.                   return c0;
127.             }
128.
129.             return eNull;
130.          }
131. //+------------------------------------------------------------------+
132.    public   :
133. //+------------------------------------------------------------------+
134.       C_Controls(const long Arg0, const string szShortName, C_Mouse *MousePtr)
135.          :C_Terminal(Arg0),
136.           m_MousePtr(MousePtr)
137.          {
138.             if ((!IndicatorCheckPass(szShortName)) || (CheckPointer(m_MousePtr) == POINTER_INVALID)) SetUserError(C_Terminal::ERR_Unknown);
139.             if (_LastError != ERR_SUCCESS) return;
140.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
141.             ObjectsDeleteAll(GetInfoTerminal().ID, def_ObjectCtrlName(""));
142.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
143.             for (eObjectControl c0 = ePlay; c0 < eNull; c0++)
144.             {
145.                m_Section[c0].h = m_Section[c0].w = def_SizeButtons;
146.                m_Section[c0].y = 25;
147.                m_Section[c0].Btn = NULL;
148.             }
149.             m_Section[ePlay].x = def_PosXObjects;
150.             m_Section[eLeft].x = m_Section[ePlay].x + 47;
151.             m_Section[eRight].x = m_Section[ePlay].x + 511;
152.             m_Slider.Minimal = eTriState;
153.          }
154. //+------------------------------------------------------------------+
155.       ~C_Controls()
156.          {
157.             for (eObjectControl c0 = ePlay; c0 < eNull; c0++) delete m_Section[c0].Btn;
158.             ObjectsDeleteAll(GetInfoTerminal().ID, def_ObjectCtrlName(""));
159.             delete m_MousePtr;
160.          }
161. //+------------------------------------------------------------------+
162.       void SetBuffer(const int rates_total, double &Buff[])
163.          {
164.             uCast_Double info;
165.
166.             info._16b[eCtrlPosition] = m_Slider.Minimal;
167.             info._16b[eCtrlStatus] = (ushort)(m_Slider.Minimal > def_MaxPosSlider ? m_Slider.Minimal : (m_Section[ePlay].state ? ePlay : ePause));//SHORT_MAX : SHORT_MIN);
168.             if (rates_total > 0)
169.                Buff[rates_total - 1] = info.dValue;
170.          }
171. //+------------------------------------------------------------------+
172.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
173.          {
174.             short x, y;
175.             static ushort iPinPosX = 0;
176.             static short six = -1, sps;
177.             uCast_Double info;
178.
179.             switch (id)
180.             {
181.                case (CHARTEVENT_CUSTOM + evCtrlReplayInit):
182.                   info.dValue = dparam;
183.                   if ((info._8b[7] != 'D') || (info._8b[6] != 'M')) break;
184.                   x = (short) info._16b[eCtrlPosition];
185.                   iPinPosX = m_Slider.Minimal = (info._16b[eCtrlPosition] > def_MaxPosSlider ? def_MaxPosSlider : (info._16b[eCtrlPosition] < iPinPosX ? iPinPosX : info._16b[eCtrlPosition]));
186.                   SetPlay((eObjectControl)(info._16b[eCtrlStatus]) == ePlay);
187.                   break;
188.                case CHARTEVENT_OBJECT_DELETE:
189.                   if (StringSubstr(sparam, 0, StringLen(def_ObjectCtrlName(""))) == def_ObjectCtrlName(""))
190.                   {
191.                      if (sparam == def_ObjectCtrlName(ePlay))
192.                      {
193.                         delete m_Section[ePlay].Btn;
194.                         m_Section[ePlay].Btn = NULL;
195.                         SetPlay(m_Section[ePlay].state);
196.                      }else
197.                      {
198.                         RemoveCtrlSlider();
199.                         CreateCtrlSlider();
200.                      }
201.                   }
202.                   break;
203.                case CHARTEVENT_MOUSE_MOVE:
204.                   if ((*m_MousePtr).CheckClick(C_Mouse::eClickLeft))   switch (CheckPositionMouseClick(x, y))
205.                   {
206.                      case ePlay:
207.                         SetPlay(!m_Section[ePlay].state);
208.                         if (m_Section[ePlay].state)
209.                         {
210.                            RemoveCtrlSlider();
211.                            m_Slider.Minimal = iPinPosX;
212.                         }else CreateCtrlSlider();
213.                         break;
214.                      case eLeft:
215.                         PositionPinSlider(iPinPosX = (iPinPosX > m_Slider.Minimal ? iPinPosX - 1 : m_Slider.Minimal));
216.                         break;
217.                      case eRight:
218.                         PositionPinSlider(iPinPosX = (iPinPosX < def_MaxPosSlider ? iPinPosX + 1 : def_MaxPosSlider));
219.                         break;
220.                      case ePin:
221.                         if (six == -1)
222.                         {
223.                            six = x;
224.                            sps = (short)iPinPosX;
225.                            ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
226.                         }
227.                         iPinPosX = sps + x - six;
228.                         PositionPinSlider(iPinPosX = (iPinPosX < m_Slider.Minimal ? m_Slider.Minimal : (iPinPosX > def_MaxPosSlider ? def_MaxPosSlider : iPinPosX)));
229.                         break;
230.                   }else if (six > 0)
231.                   {
232.                      six = -1;
233.                      ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
234.                   }
235.                   break;
236.             }
237.             ChartRedraw(GetInfoTerminal().ID);
238.          }
239. //+------------------------------------------------------------------+
240. };
241. //+------------------------------------------------------------------+
242. #undef def_PosXObjects
243. #undef def_ButtonPlay
244. #undef def_ButtonPause
245. #undef def_ButtonLeft
246. #undef def_ButtonRight
247. #undef def_ButtonPin
248. #undef def_PathBMP
249. //+------------------------------------------------------------------+
```

Source code of the C\_Controls.mqh file

The method for ensuring that control objects follow a strict format while remaining easy to declare is defined on line 23. This line may appear extremely complex at first glance, but do not be misled by its unusual appearance. If in doubt, test it in isolation to understand why it works.

Now, pay close attention to an important detail. Notice that on lines 37 and 38, we have two enumerations. The enumeration on line 37 did not previously exist but was created to simplify access to data in the buffer. You can see how this works by examining the SetBuffer procedure, located on line 162. A similar approach was applied to the message-handling procedure, though in this case, the implementation is slightly different. Review this between lines 182 and 186. But take special note of line 184: it was removed from the original code.

Returning to the topic of enumerations, observe that the enumeration on line 38 has been modified compared to its previous version. This change was made to improve code readability. For instance, the variable on line 44, which was previously a signed type, is now an unsigned type. This adjustment enables minor modifications, such as those seen on line 152, or something similar to what is observed on line 186.

All these refinements contribute to making the code more readable, as the goal is to introduce a slightly different approach than before.

Now, let's examine what exactly will be done. These modifications could potentially save a few CPU cycles over time. But first, let's understand a key aspect. n line 77, we request a change to the image displayed on the graphical object. This object is a button that indicates whether we are in play or pause mode. However, the service constantly monitors the control indicator buffer, though we can take a different approach to determine whether we are in play or pause mode. This approach directly involves the object manipulated on line 77.

### Quickly Accessing the Button Status

As mentioned in the previous section, these simple changes do not provide a performance boost significant enough to justify their implementation on their own. However, when we analyze the areas where the service most needs performance improvements, the situation changes.

In the previous article, I demonstrated where this optimization is necessary. To refresh your memory, the key point lies within LoopEventOnTime. This function periodically calls another function to check the status of the control indicator button and determine whether we are in pause or play mode.

Initially, this verification is performed by examining the data stored in the control indicator buffer. However, a slightly more elegant approach exists (though it introduces additional complexities): directly inspecting the control object itself. Keep in mind that the control object is an OBJ\_BITMAP\_LABEL, a type of object with two possible states that we can check by examining a specific variable within it.

By checking the value of a particular variable within the OBJ\_BITMAP\_LABEL object displayed on the chart, we can enable the service to bypass reading from the buffer when determining whether to play or pause data transmission to the chart.

However, if you review the C\_DrawImage.mqh header file, you will not find any modifications to the variable we need within the OBJ\_BITMAP\_LABEL object. This is the case even before attempting any changes to the service. However, by analyzing the C\_Controls.mqh file, you will notice that on line 77, a request is made to update the object. This provides us with a starting point for implementing the necessary modifications so the service can take advantage of them. In theory, this should result in some CPU cycle savings per call.

Since these changes are minimal, I will not include the entire header file here. Instead, open the C\_DrawImage.mqh file and modify it as shown in the code snippet below:

```
174. //+------------------------------------------------------------------+
175.       void Paint(const int x, const int y, const int w, const int h, const uchar cView, const int what)
176.          {
177.
178.             if ((m_szRecName == NULL) || (what < 0) || (what >= def_MaxImages)) return;
179.             ReSizeImage(w, h, cView, what);
180.             ObjectSetInteger(GetInfoTerminal().ID, m_szObjName, OBJPROP_XDISTANCE, x);
181.             ObjectSetInteger(GetInfoTerminal().ID, m_szObjName, OBJPROP_YDISTANCE, y);
182.             if (ResourceCreate(m_szRecName, m_Pixels, w, h, 0, 0, 0, COLOR_FORMAT_ARGB_NORMALIZE))
183.             {
184.                ObjectSetString(GetInfoTerminal().ID, m_szObjName, OBJPROP_BMPFILE, m_szRecName);
185.                ObjectSetString(GetInfoTerminal().ID, m_szObjName, OBJPROP_BMPFILE, what, m_szRecName);
186.                ObjectSetInteger(GetInfoTerminal().ID, m_szObjName, OBJPROP_STATE, what == 1);
187.                ChartRedraw(GetInfoTerminal().ID);
188.             }
189.          }
190. //+------------------------------------------------------------------+
```

C\_DrawImage.mqh source code snippet

Note that line 184 has been replaced by line 185 because it contains a parameter that specifies the image index. However, what we are really interested in is line 186, which updates the state of the OBJ\_BITMAP\_LABEL object variable. Now the object's OBJPROP\_STATE variable will directly reflect its state. Let's remember that there are only two possible states: playing or paused.

After that, we can look at the code in the C\_Replay.mqh header file, where the service is allowed to directly access the object and determine whether we are in play or pause mode.

For the service to understand what's going on, it first needs to add something new. The first change is just below:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #include "C_ConfigService.mqh"
05. #include "C_Controls.mqh"
06. //+------------------------------------------------------------------+
07. #define def_IndicatorControl   "Indicators\\Market Replay.ex5"
08. #resource "\\" + def_IndicatorControl
09. //+------------------------------------------------------------------+
10. #define def_CheckLoopService ((!_StopFlag) && (ChartSymbol(m_Infos.IdReplay) != ""))
11. //+------------------------------------------------------------------+
12. #define def_ShortNameIndControl "Market Replay Control"
13. //+------------------------------------------------------------------+
14. class C_Replay : public C_ConfigService
15. {
16.    private   :
17.       struct st00
18.       {
19.          C_Controls::eObjectControl Mode;
20.          uCast_Double               Memory;
21.          ushort                     Position;
22.          int                        Handle;
23.       }m_IndControl;
```

C\_Replay.mqh source code snippet

Notice that on line 5, we added a reference to the control indicator header file. We won't be implementing anything that directly uses the control class, but we do need access to the definitions that are in this file. The main one is the one that allows us to identify the names of objects created by the class. Don't worry, we'll get to that.

There are other changes in this same code part. For example, in line 19 the variable has a different type, which improves the readability of the code. Additionally, on line 20 we added a new variable. It is used to store certain values from the control indicator buffer. However, it will not be used exactly as we would like. This will become clearer later. After making these changes, we must immediately fix the constructor of the C\_Replay class. The changes can be seen below:

```
131. //+------------------------------------------------------------------+
132.       C_Replay()
133.          :C_ConfigService()
134.          {
135.             Print("************** Market Replay Service **************");
136.             srand(GetTickCount());
137.             SymbolSelect(def_SymbolReplay, false);
138.             CustomSymbolDelete(def_SymbolReplay);
139.             CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay));
140.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0);
141.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, 0);
142.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, 0);
143.             CustomSymbolSetString(def_SymbolReplay, SYMBOL_DESCRIPTION, "Symbol for replay / simulation");
144.             CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, 8);
145.             SymbolSelect(def_SymbolReplay, true);
146.             m_Infos.CountReplay = 0;
147.             m_IndControl.Handle = INVALID_HANDLE;
148.             m_IndControl.Mode = C_Controls::ePause;
149.             m_IndControl.Position = 0;
150.             m_IndControl.Memory._16b[C_Controls::eCtrlPosition] = C_Controls::eTriState;
151.          }
152. //+------------------------------------------------------------------+
```

C\_Replay.mqh source code snippet

Note how the values of the m\_IndControl structure are initialized. It is important to understand how this initialization is performed and, most importantly, why these particular values are used. Although the cause may not be clear at this stage, it will soon become apparent. The idea is to access an object present on the chart, created and maintained by the control indicator module.

To actually take advantage of this functionality and access the OBJ\_BITMAP\_LABEL object directly from the graph via the service, we need to slightly modify the UpdateIndicatorControl code that already exists in the C\_Replay class. The modification can be seen in the following fragment:

```
34. //+------------------------------------------------------------------+
35. inline void UpdateIndicatorControl(void)
36.          {
37.             static bool bTest = false;
38.             double Buff[];
39.
40.             if (m_IndControl.Handle == INVALID_HANDLE) return;
41.             if (m_IndControl.Memory._16b[C_Controls::eCtrlPosition] == m_IndControl.Position)
42.             {
43.                if (bTest)
44.                   m_IndControl.Mode = (ObjectGetInteger(m_Infos.IdReplay, def_ObjectCtrlName((C_Controls::eObjectControl)C_Controls::ePlay), OBJPROP_STATE) == 1  ? C_Controls::ePause : C_Controls::ePlay);
45.                else
46.                {
47.                   if (CopyBuffer(m_IndControl.Handle, 0, 0, 1, Buff) == 1)
48.                      m_IndControl.Memory.dValue = Buff[0];
49.                   if ((C_Controls::eObjectControl)m_IndControl.Memory._16b[C_Controls::eCtrlStatus] != C_Controls::eTriState)
50.                      if (bTest = ((m_IndControl.Mode = (C_Controls::eObjectControl)m_IndControl.Memory._16b[C_Controls::eCtrlStatus]) == C_Controls::ePlay))
51.                         m_IndControl.Position = m_IndControl.Memory._16b[C_Controls::eCtrlPosition];
52.                }
53.             }else
54.             {
55.                m_IndControl.Memory._16b[C_Controls::eCtrlPosition] = m_IndControl.Position;
56.                m_IndControl.Memory._16b[C_Controls::eCtrlStatus] = (ushort)m_IndControl.Mode;
57.                m_IndControl.Memory._8b[7] = 'D';
58.                m_IndControl.Memory._8b[6] = 'M';
59.                EventChartCustom(m_Infos.IdReplay, evCtrlReplayInit, 0, m_IndControl.Memory.dValue, "");
60.                bTest = false;
61.             }
62.          }
63. //+------------------------------------------------------------------+
```

C\_Replay.mqh source code snippet

You may have noticed that this code is significantly different from what we reviewed earlier. The main reason for these changes is to implement a safer approach to access elements related to the control indicator module.

To fully understand how this fragment functions, you need to recall a few key points:

1) Values are initialized in the constructor; 2) the first function to call this routine is the one that initializes the control indicator module; 3) at regular intervals, the loop procedure verifies the state of the button in the control indicator.

These three steps follow this sequence, but the third step is the most problematic. This is because it is responsible for injecting new ticks into the chart and continuously monitoring the control indicator. However, the ability to read data directly from the object on the chart, rather than accessing the buffer and sending events to the control indicator only in specific cases, may prevent the UpdateIndicatorControl procedure from causing a performance drop during critical phases of the replay/simulation service.

Let's see how this fragment works. First, on line 40, we check whether we have a valid handler. If true, the process continues. The next step is to verify if the memory value matches the position, which is done on line 41. If yes, we then check on line 43 whether the static variable is true. If so, we use an object access function to determine the current value of OBJ\_BITMAP\_LABEL. Pay close attention to how this is being done. It may seem unusual since we are referring to an element from the C\_Controls.mqh header file. Nevertheless, the access is indeed occurring.

If the static variable is false, it indicates that there is no issue with performing a slightly slower data read. In this case, we retrieve the data from the control indicator buffer. Important note: This does not mean that buffer reading is inherently slower, but when comparing the number of operations involved, reading a direct property of the graphical object is a simpler task.

Once the buffer is read, line 49 checks if we are not in TriState mode. If this condition is satisfied, we execute a set of operations on line 50 before determining whether we are in play mode, which will set the static variable to true or false. These operations, which are actually variable assignments, are structured in a way that might make the command appear more complex than it actually is. However, since this does not matter to the compiler and the values are assigned as expected, we can structure it this way. If line 50 evaluates to true, we save the previous value of the buffer in the internal position variable. This happens on line 51.

This sequence of operations occurs in only one situation: when the user interacts with the slider and changes the position where the replay/simulator should start. In other words, when we are in play mode, this code is not executed. However, when transitioning from pause mode to play mode, this code will be triggered, which will be important later.

If the condition in line 41 evaluates to false, the instructions between lines 55 and 60 are executed. This triggers a custom event to update the control indicator module. Moving forward, this is how the system will function.

Reading the object directly from the chart may not necessarily improve performance. However, it opens new possibilities for those who want to manipulate chart objects efficiently. This approach allows for the development of more sophisticated tools without overloading the MetaTrader 5 platform, eliminating the need to clutter charts with unnecessary indicators solely for object manipulation.

### Achieving Real Performance Gains

Despite all the improvements discussed so far, they do not provide a substantial performance boost for the replay/simulation service. At least, no major changes are observed. However, these changes help streamline certain sections of the code, making them more efficient. The most significant benefit is the improved readability. This is due to the definitions made in the C\_Controls.mqh file, which were then applied in the C\_Replay.mqh header file. Even if you have not yet reviewed the entire code, you can likely already anticipate where modifications should be made to improve the readability of the C\_Replay class.

Now, let's explore a modification that genuinely enhances performance. The goal is to restore the functionality of generating a one-minute bar within the expected period.

At first, the proposed change may seem bizarre and entirely counterintuitive. However, believe me, you can test it yourself. This simple modification results in a substantial performance gain. To see it in action, let's examine the complete code of the C\_Replay.mqh header file. Below is its full version. The line numbering will differ slightly from the previous code snippets, but don't focus on that. Let's study the code.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12121](https://www.mql5.com/pt/articles/12121)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12121.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12121/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/483323)**

![From Basic to Intermediate: IF ELSE](https://c.mql5.com/2/90/logo-midjourney_image_15365_401_3870__8.png)[From Basic to Intermediate: IF ELSE](https://www.mql5.com/en/articles/15365)

In this article we will discuss how to work with the IF operator and its companion ELSE. This statement is the most important and significant of those existing in any programming language. However, despite its ease of use, it can sometimes be confusing if we have no experience with its use and the concepts associated with it. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Price Action Analysis Toolkit Development (Part 18): Introducing Quarters Theory (III) — Quarters Board](https://c.mql5.com/2/126/Price_Action_Toolkit_Development_Part_18__LOGO.png)[Price Action Analysis Toolkit Development (Part 18): Introducing Quarters Theory (III) — Quarters Board](https://www.mql5.com/en/articles/17442)

In this article, we enhance the original Quarters Script by introducing the Quarters Board, a tool that lets you toggle quarter levels directly on the chart without needing to revisit the code. You can easily activate or deactivate specific levels, and the EA also provides trend direction commentary to help you better understand market movements.

![Data Science and ML (Part 35): NumPy in MQL5 – The Art of Making Complex Algorithms with Less Code](https://c.mql5.com/2/126/Data_Science_and_ML_Part_35__LOGO.png)[Data Science and ML (Part 35): NumPy in MQL5 – The Art of Making Complex Algorithms with Less Code](https://www.mql5.com/en/articles/17469)

NumPy library is powering almost all the machine learning algorithms to the core in Python programming language, In this article we are going to implement a similar module which has a collection of all the complex code to aid us in building sophisticated models and algorithms of any kind.

![Exploring Advanced Machine Learning Techniques on the Darvas Box Breakout Strategy](https://c.mql5.com/2/126/Exploring_Advanced_Machine_Learning_Techniques_on_the_Darvas_Box_Breakout_Strategy___LOGO.png)[Exploring Advanced Machine Learning Techniques on the Darvas Box Breakout Strategy](https://www.mql5.com/en/articles/17466)

The Darvas Box Breakout Strategy, created by Nicolas Darvas, is a technical trading approach that spots potential buy signals when a stock’s price rises above a set "box" range, suggesting strong upward momentum. In this article, we will apply this strategy concept as an example to explore three advanced machine learning techniques. These include using a machine learning model to generate signals rather than to filter trades, employing continuous signals rather than discrete ones, and using models trained on different timeframes to confirm trades.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/12121&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069637576727398471)

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