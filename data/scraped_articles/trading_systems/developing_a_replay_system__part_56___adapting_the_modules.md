---
title: Developing a Replay System (Part 56): Adapting the Modules
url: https://www.mql5.com/en/articles/12000
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:42:58.738696
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/12000&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069681793415710976)

MetaTrader 5 / Examples


### Introduction

In the previous article " [Developing a Replay System (Part 55): Control Module](https://www.mql5.com/en/articles/11988)" we implemented some changes that allowed us to create a control indicator without using terminal global variables. At least in terms of storing information and user-specified settings.

While everything worked fine and was fairly stable, when the system was placed on a chart with a certain number of bars, it kept crashing due to range limits being exceeded when using data related to a custom symbol.

The reason is not that the custom symbol is actually broken or causing problems. This is because the database is often insufficient for the correct functioning of the entire system. The main problem factor is the buffer.

You may be wondering now, "How is it possible that the buffer is the problem? The system works when we use a real symbol, but when we use a custom symbol, it crashes, and the reason for this crash is the indicator buffer?!"

Yes, the reason is the buffer. But not in the way you might imagine. That's because when we create a custom symbol, the buffer may not have the size required to store the data we want to put into it.

You might think that all you need to do is allocate more memory and the problem will be solved. However, when it comes to MQL5, things are not so simple. Allocating memory for data in the indicator buffer does not work as you probably imagine. Memory allocation depends on the number of bars present on the chart. So there is no point in using any function to allocate memory since it will not actually be used in the way you expect.

### Understanding the problem

The real problem is not the control indicator, but the mouse indicator. While fixing this anomaly, we will create a solution that will also affect the control indicator. The changes will become visible later in this article. But first, let's understand the nature of the failure that occurs and how it occurs.

If you use the mouse indicator presented in the previous articles and place it on a custom symbol chart that has, for example, 60 one-minute bars, you will have no problem on timeframes equal to or less than 10 minutes. However, if you try to use a timeframe greater than 10 minutes, you will receive a message from MetaTrader 5: "Mouse indicator: Range error".

Why does this happen? The reason is that the mouse indicator presented in the previous article requires 6 positions to store data in the indicator buffer. Therefore, mathematics will answer the question of what actually happened. With 60 bars per minute you can change the timeframe to 10 minutes which will produce 6 bars on the chart. These six bars will provide the required six buffer positions to accommodate the data. However, if you choose a higher timeframe, the number of bars on the chart will be less than six.

At this point, the mouse indicator will return a range error because it will try to write data to a memory position that MetaTrader 5 has not allocated.

This is where the error lies, and there are two ways to fix it. The first one is to place a sufficient number of bars on the custom symbol so that there are at least six of them on any timeframe. This is not the most suitable solution, since in order to access the monthly timeframe, we would need to load at least six months of 1-minute bars onto the custom symbol chart. This is only necessary to prevent a range error from occurring.

Personally, I think, and I'm sure many would agree, that this is far from the best solution, especially when it comes to the replay/simulator system. If the system was purely replay oriented, perhaps this solution could work, provided that the analysis was only done on one timeframe or lower timeframes. But since we can use the system to model market movements, this solution is completely unacceptable and something more elegant is needed.

This is what we will be doing now. We will modify the mouse indicator so that the information will fit compactly into one position inside the buffer. Thus, we will need just one bar on the chart for the indicator to perform its function.

### Starting the solution implementation

I decided to implement the compact placement of information in one position because it will be much easier for the replay/simulator system to add and maintain at least one bar on the chart than to do anything else.

However, the main reason is that we may want to create market simulation without using a huge number of bars. We will be able to use only the amount that we want, and the replay/simulator service itself will provide the system with the necessary stability. This will allow us to use the same means both on a real account and on a demo account.

Summarizing: We will make the mouse indicator buffer much smaller. It will only use one position, but we will get the same amount of information that will be returned when we request from the indicator to read the buffer. One point worth noting: If you read directly from the mouse indicator buffer, you will only get data at one position in that buffer. Such data must be translated so that it can be really useful. With the C\_Mouse class, we can use a certain function to ensure that this translation is done correctly.

With that said, we move on to the implementation phase. The first thing we need to change is the header file. Its code is given below.

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
13. #define def_SymbolReplay      "RePlay"
14. #define def_MaxPosSlider       400
15. //+------------------------------------------------------------------+
16. union uCast_Double
17. {
18.    double   dValue;
19.    long     _long;                                  // 1 Information
20.    datetime _datetime;                              // 1 Information
21.    uint     _32b[sizeof(double) / sizeof(uint)];    // 2 Informations
22.    ushort   _16b[sizeof(double) / sizeof(ushort)];  // 4 Informations
23.    uchar    _8b [sizeof(double) / sizeof(uchar)];   // 8 Informations
24. };
25. //+------------------------------------------------------------------+
26. enum EnumEvents    {
27.          evHideMouse,               //Hide mouse price line
28.          evShowMouse,               //Show mouse price line
29.          evHideBarTime,             //Hide bar time
30.          evShowBarTime,             //Show bar time
31.          evHideDailyVar,            //Hide daily variation
32.          evShowDailyVar,            //Show daily variation
33.          evHidePriceVar,            //Hide instantaneous variation
34.          evShowPriceVar,            //Show instantaneous variation
35.          evSetServerTime,           //Replay/simulation system timer
36.          evCtrlReplayInit           //Initialize replay control
37.                   };
38. //+------------------------------------------------------------------+
```

Defines.mqh file source code

There are virtually no significant differences here, at least in general terms. However, if you examine closely, you'll notice changes in lines 21 and 23. These changes were implemented specifically to enable more efficient use of bits. To make it easier to identify the type of information we'll be working with, I used a simple notation: \_32b represents 32 bits, \_16b represents 16 bits, and \_8b represents 8 bits. This way, during any type of access, we'll know exactly how many bits will be used. It's worth noting that a double value represents 64 bits, so each of the data packets will have a length limit within these 64 bits.

Thus, \_32b can contain 2 values, \_16b can contain 4 values, and \_8b can contain 8 values. However, pay attention to the fact that we are using a union, which means we can combine these sets. To do this correctly, you must understand that every array in MQL5 is based on the numbering system used in C/C++. In other words, arrays always start at zero, and each subsequent position is incremented by one unit.

At this point, many might start to feel confused if they lack a foundational understanding of how things work in C/C++. This is because the term "unit" here may not adequately convey what is actually required when incrementing the index value to correctly place data within the packet. Misunderstanding this concept can leave you completely lost or, at the very least, unable to comprehend how the information is actually being compacted.

Before we dive into the changes made to the mouse indicator, let's take a quick look at the control indicator. The reason for this is straightforward: in the control indicator, we just needed to adjust the functions, procedures, and variables to align with the new types introduced in the header file Defines.mqh. Let's begin by examining the class code, which can be seen below.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\Auxiliar\C_DrawImage.mqh"
005. #include "..\Defines.mqh"
006. //+------------------------------------------------------------------+
007. #define def_PathBMP            "Images\\Market Replay\\Control\\"
008. #define def_ButtonPlay         def_PathBMP + "Play.bmp"
009. #define def_ButtonPause        def_PathBMP + "Pause.bmp"
010. #define def_ButtonLeft         def_PathBMP + "Left.bmp"
011. #define def_ButtonLeftBlock    def_PathBMP + "Left_Block.bmp"
012. #define def_ButtonRight        def_PathBMP + "Right.bmp"
013. #define def_ButtonRightBlock   def_PathBMP + "Right_Block.bmp"
014. #define def_ButtonPin          def_PathBMP + "Pin.bmp"
015. #resource "\\" + def_ButtonPlay
016. #resource "\\" + def_ButtonPause
017. #resource "\\" + def_ButtonLeft
018. #resource "\\" + def_ButtonLeftBlock
019. #resource "\\" + def_ButtonRight
020. #resource "\\" + def_ButtonRightBlock
021. #resource "\\" + def_ButtonPin
022. //+------------------------------------------------------------------+
023. #define def_PrefixCtrlName    "MarketReplayCTRL_"
024. #define def_PosXObjects       120
025. //+------------------------------------------------------------------+
026. #define def_SizeButtons       32
027. #define def_ColorFilter       0xFF00FF
028. //+------------------------------------------------------------------+
029. #include "..\Auxiliar\C_Terminal.mqh"
030. #include "..\Auxiliar\C_Mouse.mqh"
031. //+------------------------------------------------------------------+
032. class C_Controls : private C_Terminal
033. {
034.    protected:
035.    private   :
036. //+------------------------------------------------------------------+
037.       enum eObjectControl {ePlay, eLeft, eRight, ePin, eNull};
038. //+------------------------------------------------------------------+
039.       struct st_00
040.       {
041.          string   szBarSlider,
042.                   szBarSliderBlock;
043.          short    Minimal;
044.       }m_Slider;
045.       struct st_01
046.       {
047.          C_DrawImage *Btn;
048.          bool         state;
049.          short        x, y, w, h;
050.       }m_Section[eObjectControl::eNull];
051.       C_Mouse   *m_MousePtr;
052. //+------------------------------------------------------------------+
053. inline void CreteBarSlider(short x, short size)
054.          {
055.             ObjectCreate(GetInfoTerminal().ID, m_Slider.szBarSlider = def_PrefixCtrlName + "B1", OBJ_RECTANGLE_LABEL, 0, 0, 0);
056.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_XDISTANCE, def_PosXObjects + x);
057.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_YDISTANCE, m_Section[ePin].y + 11);
058.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_XSIZE, size);
059.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_YSIZE, 9);
060.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BGCOLOR, clrLightSkyBlue);
061.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BORDER_COLOR, clrBlack);
062.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_WIDTH, 3);
063.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BORDER_TYPE, BORDER_FLAT);
064.             ObjectCreate(GetInfoTerminal().ID, m_Slider.szBarSliderBlock = def_PrefixCtrlName + "B2", OBJ_RECTANGLE_LABEL, 0, 0, 0);
065.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XDISTANCE, def_PosXObjects + x);
066.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_YDISTANCE, m_Section[ePin].y + 6);
067.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_YSIZE, 19);
068.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_BGCOLOR, clrRosyBrown);
069.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_BORDER_TYPE, BORDER_RAISED);
070.          }
071. //+------------------------------------------------------------------+
072.       void SetPlay(bool state)
073.          {
074.             if (m_Section[ePlay].Btn == NULL)
075.                m_Section[ePlay].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_PrefixCtrlName + EnumToString(ePlay), def_ColorFilter, "::" + def_ButtonPlay, "::" + def_ButtonPause);
076.             m_Section[ePlay].Btn.Paint(m_Section[ePlay].x, m_Section[ePlay].y, m_Section[ePlay].w, m_Section[ePlay].h, 20, ((m_Section[ePlay].state = state) ? 0 : 1));
077.          }
078. //+------------------------------------------------------------------+
079.       void CreateCtrlSlider(void)
080.          {
081.             CreteBarSlider(77, 436);
082.             m_Section[eLeft].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_PrefixCtrlName + EnumToString(eLeft), def_ColorFilter, "::" + def_ButtonLeft, "::" + def_ButtonLeftBlock);
083.             m_Section[eRight].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_PrefixCtrlName + EnumToString(eRight), def_ColorFilter, "::" + def_ButtonRight, "::" + def_ButtonRightBlock);
084.             m_Section[ePin].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_PrefixCtrlName + EnumToString(ePin), def_ColorFilter, "::" + def_ButtonPin);
085.             PositionPinSlider(m_Slider.Minimal);
086.          }
087. //+------------------------------------------------------------------+
088. inline void RemoveCtrlSlider(void)
089.          {
090.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
091.             for (eObjectControl c0 = ePlay + 1; c0 < eNull; c0++)
092.             {
093.                delete m_Section[c0].Btn;
094.                m_Section[c0].Btn = NULL;
095.             }
096.             ObjectsDeleteAll(GetInfoTerminal().ID, def_PrefixCtrlName + "B");
097.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
098.          }
099. //+------------------------------------------------------------------+
100. inline void PositionPinSlider(short p)
101.          {
102.             int iL, iR;
103.
104.             m_Section[ePin].x = (p < m_Slider.Minimal ? m_Slider.Minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
105.             iL = (m_Section[ePin].x != m_Slider.Minimal ? 0 : 1);
106.             iR = (m_Section[ePin].x < def_MaxPosSlider ? 0 : 1);
107.             m_Section[ePin].x += def_PosXObjects;
108.              m_Section[ePin].x += 95 - (def_SizeButtons / 2);
109.              for (eObjectControl c0 = ePlay + 1; c0 < eNull; c0++)
110.                m_Section[c0].Btn.Paint(m_Section[c0].x, m_Section[c0].y, m_Section[c0].w, m_Section[c0].h, 20, (c0 == eLeft ? iL : (c0 == eRight ? iR : 0)));
111.             ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, m_Slider.Minimal + 2);
112.          }
113. //+------------------------------------------------------------------+
114. inline eObjectControl CheckPositionMouseClick(short &x, short &y)
115.          {
116.             C_Mouse::st_Mouse InfoMouse;
117.
118.             InfoMouse = (*m_MousePtr).GetInfoMouse();
119.             x = (short) InfoMouse.Position.X_Graphics;
120.             y = (short) InfoMouse.Position.Y_Graphics;
121.             for (eObjectControl c0 = ePlay; c0 < eNull; c0++)
122.             {
123.                if ((m_Section[c0].Btn != NULL) && (m_Section[c0].x <= x) && (m_Section[c0].y <= y) && ((m_Section[c0].x + m_Section[c0].w) >= x) && ((m_Section[c0].y + m_Section[c0].h) >= y))
124.                   return c0;
125.             }
126.
127.             return eNull;
128.          }
129. //+------------------------------------------------------------------+
130.    public   :
131. //+------------------------------------------------------------------+
132.       C_Controls(const long Arg0, const string szShortName, C_Mouse *MousePtr)
133.          :C_Terminal(Arg0),
134.           m_MousePtr(MousePtr)
135.          {
136.             if ((!IndicatorCheckPass(szShortName)) || (CheckPointer(m_MousePtr) == POINTER_INVALID)) SetUserError(C_Terminal::ERR_Unknown);
137.             if (_LastError != ERR_SUCCESS) return;
138.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
139.             ObjectsDeleteAll(GetInfoTerminal().ID, def_PrefixCtrlName);
140.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
141.             for (eObjectControl c0 = ePlay; c0 < eNull; c0++)
142.             {
143.                m_Section[c0].h = m_Section[c0].w = def_SizeButtons;
144.                m_Section[c0].y = 25;
145.                m_Section[c0].Btn = NULL;
146.             }
147.             m_Section[ePlay].x = def_PosXObjects;
148.             m_Section[eLeft].x = m_Section[ePlay].x + 47;
149.             m_Section[eRight].x = m_Section[ePlay].x + 511;
150.             m_Slider.Minimal = SHORT_MIN;
151.          }
152. //+------------------------------------------------------------------+
153.       ~C_Controls()
154.          {
155.             for (eObjectControl c0 = ePlay; c0 < eNull; c0++) delete m_Section[c0].Btn;
156.             ObjectsDeleteAll(GetInfoTerminal().ID, def_PrefixCtrlName);
157.             delete m_MousePtr;
158.          }
159. //+------------------------------------------------------------------+
160.       void SetBuffer(const int rates_total, double &Buff[])
161.          {
162.             uCast_Double info;
163.
164.             info._16b[0] = (ushort) m_Slider.Minimal;
165.             info._16b[1] = (ushort) (m_Section[ePlay].state ? SHORT_MAX : SHORT_MIN);
166.             if (rates_total > 0)
167.                Buff[rates_total - 1] = info.dValue;
168.          }
169. //+------------------------------------------------------------------+
170.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
171.          {
172.             short x, y;
173.             static short iPinPosX = -1, six = -1, sps;
174.             uCast_Double info;
175.
176.             switch (id)
177.             {
178.                case (CHARTEVENT_CUSTOM + evCtrlReplayInit):
179.                   info.dValue = dparam;
180.                   iPinPosX = m_Slider.Minimal = (short) info._16b[0];
181.                   if (info._16b[1] == 0) SetUserError(C_Terminal::ERR_Unknown); else
182.                   {
183.                      SetPlay((short)(info._16b[1]) == SHORT_MAX);
184.                      if ((short)(info._16b[1]) == SHORT_MIN) CreateCtrlSlider();
185.                   }
186.                   break;
187.                case CHARTEVENT_OBJECT_DELETE:
188.                   if (StringSubstr(sparam, 0, StringLen(def_PrefixCtrlName)) == def_PrefixCtrlName)
189.                   {
190.                      if (sparam == (def_PrefixCtrlName + EnumToString(ePlay)))
191.                      {
192.                         delete m_Section[ePlay].Btn;
193.                         m_Section[ePlay].Btn = NULL;
194.                         SetPlay(m_Section[ePlay].state);
195.                      }else
196.                      {
197.                         RemoveCtrlSlider();
198.                         CreateCtrlSlider();
199.                      }
200.                   }
201.                   break;
202.                case CHARTEVENT_MOUSE_MOVE:
203.                   if ((*m_MousePtr).CheckClick(C_Mouse::eClickLeft))   switch (CheckPositionMouseClick(x, y))
204.                   {
205.                      case ePlay:
206.                         SetPlay(!m_Section[ePlay].state);
207.                         if (m_Section[ePlay].state)
208.                         {
209.                            RemoveCtrlSlider();
210.                            m_Slider.Minimal = iPinPosX;
211.                         }else CreateCtrlSlider();
212.                         break;
213.                      case eLeft:
214.                         PositionPinSlider(iPinPosX = (iPinPosX > m_Slider.Minimal ? iPinPosX - 1 : m_Slider.Minimal));
215.                         break;
216.                      case eRight:
217.                         PositionPinSlider(iPinPosX = (iPinPosX < def_MaxPosSlider ? iPinPosX + 1 : def_MaxPosSlider));
218.                         break;
219.                      case ePin:
220.                         if (six == -1)
221.                         {
222.                            six = x;
223.                            sps = iPinPosX;
224.                            ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
225.                         }
226.                         iPinPosX = sps + x - six;
227.                         PositionPinSlider(iPinPosX = (iPinPosX < m_Slider.Minimal ? m_Slider.Minimal : (iPinPosX > def_MaxPosSlider ? def_MaxPosSlider : iPinPosX)));
228.                         break;
229.                   }else if (six > 0)
230.                   {
231.                      six = -1;
232.                      ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
233.                   }
234.                   break;
235.             }
236.             ChartRedraw(GetInfoTerminal().ID);
237.          }
238. //+------------------------------------------------------------------+
239. };
240. //+------------------------------------------------------------------+
241. #undef def_PosXObjects
242. #undef def_ButtonPlay
243. #undef def_ButtonPause
244. #undef def_ButtonLeft
245. #undef def_ButtonRight
246. #undef def_ButtonPin
247. #undef def_PrefixCtrlName
248. #undef def_PathBMP
249. //+------------------------------------------------------------------+
```

Source code of C\_Control.mqh

When examining the code, you'll notice there are no visible differences, at least not at first glance, between the code shown above and the same code presented in the last article where this header file was discussed. However, due to changes made to the definition file, certain parts of the code underwent minor modifications. One such change involves the explicit use of typecasting, as can be observed in lines 164 and 165, where we instruct the compiler to explicitly use a specific data type.

Note the change in line 165. Previously, we used integer constants, but now we use short constants. Although these are signed constants, meaning they can represent negative values, the values placed in the array are unsigned. At this point, you might think this could lead to an incorrect interpretation of the values. If so, then you might not be understanding correctly how values are represented in binary form. I recommend studying binary value representation to understand how negative values can be represented in a system that doesn't explicitly handle them while still transferring information without quality loss.

In the same code, you'll see another explicit type conversion in line 180. Here a value stored in an unsigned variable is assigned to a signed variable. This type conversion ensures that negative values are appropriately represented. Throughout the custom event handling code and during the initialization of the control indicator, this type of conversion is heavily utilized. Take your time to carefully study the segment between lines 178 and 186, as it demonstrates intensive use of this conversion method.

It's worth mentioning that the compression performed in this control indicator was not as deep as it could have been to maximize the efficient use of bits within a double. This is because our case doesn't demand such advanced optimization. Lastly, regarding the control indicator code, the only changes made were related to the version number and a link. Therefore, I will not repeat the entire code here. All you need to do is replace the header file from the previous article with the one provided in this article. The rest of the indicator code remains unchanged and can be used without issue.

Now, when it comes to the mouse indicator, things get a bit more complex. As a result, we will need all three files required to create the indicator. Let's discuss these files win the next section.

### Implementing the solution for the mouse indicator

As you may have seen in the previous section, minor adjustments were made to the control indicator code, and these modifications were limited to the header file. However, with the mouse indicator, the situation is quite different and significantly more complex.

Let's view the changes. First, let's examine the new C\_Mouse class, which is presented below.

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
014.    public   :
015.       enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
016.       enum eBtnMouse {eKeyNull = 0x00, eClickLeft = 0x01, eClickRight = 0x02, eSHIFT_Press = 0x04, eCTRL_Press = 0x08, eClickMiddle = 0x10};
017.       struct st_Mouse
018.       {
019.          struct st00
020.          {
021.             short    X_Adjusted,
022.                      Y_Adjusted,
023.                      X_Graphics,
024.                      Y_Graphics;
025.             double   Price;
026.             datetime dt;
027.          }Position;
028.          uchar      ButtonStatus;
029.          bool      ExecStudy;
030.       };
031. //+------------------------------------------------------------------+
032.    protected:
033. //+------------------------------------------------------------------+
034.       void CreateObjToStudy(int x, int w, string szName, color backColor = clrNONE) const
035.          {
036.             if (m_Mem.szShortName != NULL) return;
037.             CreateObjectGraphics(szName, OBJ_BUTTON, clrNONE);
038.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_STATE, true);
039.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BORDER_COLOR, clrBlack);
040.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_COLOR, clrBlack);
041.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BGCOLOR, backColor);
042.             ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_FONT, "Lucida Console");
043.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_FONTSIZE, 10);
044.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
045.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XDISTANCE, x);
046.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YDISTANCE, TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT) + 1);
047.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XSIZE, w);
048.             ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YSIZE, 18);
049.          }
050. //+------------------------------------------------------------------+
051.    private   :
052.       enum eStudy {eStudyNull, eStudyCreate, eStudyExecute};
053.       struct st01
054.       {
055.          st_Mouse Data;
056.          color    corLineH,
057.                   corTrendP,
058.                   corTrendN;
059.          eStudy   Study;
060.       }m_Info;
061.       struct st_Mem
062.       {
063.          bool     CrossHair,
064.                   IsFull;
065.          datetime dt;
066.          string   szShortName;
067.       }m_Mem;
068.       bool m_OK;
069. //+------------------------------------------------------------------+
070.       void GetDimensionText(const string szArg, int &w, int &h)
071.          {
072.             TextSetFont("Lucida Console", -100, FW_NORMAL);
073.             TextGetSize(szArg, w, h);
074.             h += 5;
075.             w += 5;
076.          }
077. //+------------------------------------------------------------------+
078.       void CreateStudy(void)
079.          {
080.             if (m_Mem.IsFull)
081.             {
082.                CreateObjectGraphics(def_NameObjectLineV, OBJ_VLINE, m_Info.corLineH);
083.                CreateObjectGraphics(def_NameObjectLineT, OBJ_TREND, m_Info.corLineH);
084.                ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_WIDTH, 2);
085.                CreateObjToStudy(0, 0, def_NameObjectStudy);
086.             }
087.             m_Info.Study = eStudyCreate;
088.          }
089. //+------------------------------------------------------------------+
090.       void ExecuteStudy(const double memPrice)
091.          {
092.             double v1 = GetInfoMouse().Position.Price - memPrice;
093.             int w, h;
094.
095.             if (!CheckClick(eClickLeft))
096.             {
097.                m_Info.Study = eStudyNull;
098.                ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
099.                if (m_Mem.IsFull)   ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName + "T");
100.             }else if (m_Mem.IsFull)
101.             {
102.                string sz1 = StringFormat(" %." + (string)GetInfoTerminal().nDigits + "f [ %d ] %02.02f%% ",
103.                   MathAbs(v1), Bars(GetInfoTerminal().szSymbol, PERIOD_CURRENT, m_Mem.dt, GetInfoMouse().Position.dt) - 1, MathAbs((v1 / memPrice) * 100.0));
104.                GetDimensionText(sz1, w, h);
105.                ObjectSetString(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_TEXT, sz1);
106.                ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corTrendN : m_Info.corTrendP));
107.                ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_XSIZE, w);
108.                ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_YSIZE, h);
109.                ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_XDISTANCE, GetInfoMouse().Position.X_Adjusted - w);
110.                ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - (v1 < 0 ? 1 : h));
111.                ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 1, GetInfoMouse().Position.dt, GetInfoMouse().Position.Price);
112.                ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_COLOR, (memPrice > GetInfoMouse().Position.Price ? m_Info.corTrendN : m_Info.corTrendP));
113.             }
114.             m_Info.Data.ButtonStatus = eKeyNull;
115.          }
116. //+------------------------------------------------------------------+
117. inline void DecodeAlls(int xi, int yi)
118.          {
119.             int w = 0;
120.
121.             ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X_Graphics = (short) xi, m_Info.Data.Position.Y_Graphics = (short)yi, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
122.             m_Info.Data.Position.dt = AdjustTime(m_Info.Data.Position.dt);
123.             m_Info.Data.Position.Price = AdjustPrice(m_Info.Data.Position.Price);
124.             ChartTimePriceToXY(GetInfoTerminal().ID, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price, xi, yi);
125.             m_Info.Data.Position.X_Adjusted = (short) xi;
126.             m_Info.Data.Position.Y_Adjusted = (short) yi;
127.          }
128. //+------------------------------------------------------------------+
129.    public   :
130. //+------------------------------------------------------------------+
131.       C_Mouse(const long id, const string szShortName)
132.          :C_Terminal(id),
133.          m_OK(false)
134.          {
135.             m_Mem.szShortName = szShortName;
136.          }
137. //+------------------------------------------------------------------+
138.       C_Mouse(const long id, const string szShortName, color corH, color corP, color corN)
139.          :C_Terminal(id)
140.          {
141.             if (!(m_OK = IndicatorCheckPass(szShortName))) SetUserError(C_Terminal::ERR_Unknown);
142.             if (_LastError != ERR_SUCCESS) return;
143.             m_Mem.szShortName = NULL;
144.             m_Mem.CrossHair = (bool)ChartGetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL);
145.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, true);
146.              ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, false);
147.             ZeroMemory(m_Info);
148.             m_Info.corLineH  = corH;
149.             m_Info.corTrendP = corP;
150.             m_Info.corTrendN = corN;
151.             m_Info.Study = eStudyNull;
152.             if (m_Mem.IsFull = (corP != clrNONE) && (corH != clrNONE) && (corN != clrNONE))
153.                CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
154.          }
155. //+------------------------------------------------------------------+
156.       ~C_Mouse()
157.          {
158.             if (!m_OK) return;
159.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
160.             ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, false);
161.              ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, m_Mem.CrossHair);
162.             ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName);
163.          }
164. //+------------------------------------------------------------------+
165. inline bool CheckClick(const eBtnMouse value)
166.          {
167.             return (GetInfoMouse().ButtonStatus & value) == value;
168.          }
169. //+------------------------------------------------------------------+
170. inline const st_Mouse GetInfoMouse(void)
171.          {
172.             if (m_Mem.szShortName != NULL)
173.             {
174.                double Buff[];
175.                uCast_Double loc;
176.                int handle = ChartIndicatorGet(GetInfoTerminal().ID, 0, m_Mem.szShortName);
177.
178.                ZeroMemory(m_Info.Data);
179.                if (CopyBuffer(handle, 0, 0, 1, Buff) == 1)
180.                {
181.                   loc.dValue = Buff[0];
182.                   m_Info.Data.ButtonStatus = loc._8b[0];
183.                   DecodeAlls((int)loc._16b[1], (int)loc._16b[2]);
184.                }
185.                IndicatorRelease(handle);
186.             }
187.
188.             return m_Info.Data;
189.          }
190. //+------------------------------------------------------------------+
191. inline void SetBuffer(const int rates_total, double &Buff[])
192.          {
193.             uCast_Double info;
194.
195.             info._8b[0] = (uchar)(m_Info.Study == C_Mouse::eStudyNull ? m_Info.Data.ButtonStatus : 0);
196.             info._16b[1] = (ushort) m_Info.Data.Position.X_Graphics;
197.             info._16b[2] = (ushort) m_Info.Data.Position.Y_Graphics;
198.             Buff[rates_total - 1] = info.dValue;
199.          }
200. //+------------------------------------------------------------------+
201.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
202.          {
203.             int w = 0;
204.             static double memPrice = 0;
205.
206.             if (m_Mem.szShortName == NULL)
207.             {
208.                C_Terminal::DispatchMessage(id, lparam, dparam, sparam);
209.                switch (id)
210.                {
211.                   case (CHARTEVENT_CUSTOM + evHideMouse):
212.                      if (m_Mem.IsFull)   ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
213.                      break;
214.                   case (CHARTEVENT_CUSTOM + evShowMouse):
215.                      if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, m_Info.corLineH);
216.                      break;
217.                   case CHARTEVENT_MOUSE_MOVE:
218.                      DecodeAlls((int)lparam, (int)dparam);
219.                      if (m_Mem.IsFull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price = AdjustPrice(m_Info.Data.Position.Price));
220.                      if ((m_Info.Study != eStudyNull) && (m_Mem.IsFull)) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineV, 0, m_Info.Data.Position.dt, 0);
221.                      m_Info.Data.ButtonStatus = (uchar) sparam; //Mudança no tipo ...
222.                      if (CheckClick(eClickMiddle))
223.                         if ((!m_Mem.IsFull) || ((color)ObjectGetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR) != clrNONE)) CreateStudy();
224.                      if (CheckClick(eClickLeft) && (m_Info.Study == eStudyCreate))
225.                      {
226.                         ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
227.                         if (m_Mem.IsFull)   ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 0, m_Mem.dt = GetInfoMouse().Position.dt, memPrice = GetInfoMouse().Position.Price);
228.                         m_Info.Study = eStudyExecute;
229.                      }
230.                      if (m_Info.Study == eStudyExecute) ExecuteStudy(memPrice);
231.                      m_Info.Data.ExecStudy = m_Info.Study == eStudyExecute;
232.                      break;
233.                   case CHARTEVENT_OBJECT_DELETE:
234.                      if ((m_Mem.IsFull) && (sparam == def_NameObjectLineH)) CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
235.                      break;
236.                }
237.             }
238.          }
239. //+------------------------------------------------------------------+
240. };
241. //+------------------------------------------------------------------+
242. #undef def_NameObjectLineV
243. #undef def_NameObjectLineH
244. #undef def_NameObjectLineT
245. #undef def_NameObjectStudy
246. //+------------------------------------------------------------------+
```

Source code of the C\_Mouse.mqh file

You might not immediately notice the changes that were made, primarily because most of them concern variable types. Nonetheless, they are worth mentioning, and a proper understanding will help you identify the limitations of this mouse indicator. Yes, it has limitations, but once you understand them, you'll be able to use it appropriately, ensuring these constraints do not impact your usage.

First, you'll notice in lines 21 to 24 that we are now using the SHORT type, whereas previously, it was INT. Similarly, the variable type was modified in line 28.

Before you panic thinking it's chaos caused by these type changes, let me remind you of some characteristics and peculiarities of the SHORT and INT types.

In MQL5, the INT type is a 32-bit data type, meaning its values range from -2,147,483,648 to 2,147,483,647 when signed. If it is unsigned, that is, positive only, the value can range from 0 to 4,294,967,295. In contrast, the SHORT type is a 16-bit data type, with a range from -32,768 to 32,767 when signed and from 0 to 65,535 when unsigned.

Now, consider this: what are your monitor's pixel dimensions? And why am I asking this? Because using an INT to represent Cartesian dimensions (X and Y) on a monitor is simply inefficient. Don't get me wrong, but you're effectively wasting a lot of space. For example, an 8K monitor, which is an exceptionally high-resolution display, has 7,680 horizontal pixels and 4,320 vertical pixels. This resolution is less than 2 to the power of 13 positions both horizontally and vertically, which fits comfortably within a signed SHORT type's 2 to the power of 6 positions. As a result, there are still 3 unused bits for other purposes.

This simple optimization is more than sufficient. Using an INT would allow for just two values, but by using SHORT, we can store four within the same 64-bit double. Since we need only two SHORT values to represent the mouse position (X and Y), we have two additional SHORT slots for other purposes. Moreover, even for an 8K display, we still have 3 unused bits within each SHORT used to store the mouse coordinates.

As you can see, the limitation isn't related to positional data. The actual limitation lies elsewhere. If you've been following this series of articles, you'll have noticed that we aim to allow the mouse indicator to provide information for simpler and faster analysis. Herein lies the limitation: price and time values cannot be directly stored in the mouse indicator's buffer. These values require 64 bits each, and we cannot afford to allocate two positions for them. To address this, we need a different approach, leading to changes in the C\_Mouse class.

The first noticeable change is in line 117. Note: to avoid making extensive modifications to all the types in the mouse indicator, I decided to keep positions as INT within the class. However, outside the class, these values are adjusted as needed.

The procedure in line 117 translates the screen coordinates provided by the operating system into coordinates adjusted by MQL5, aligning them with what is displayed on the chart. Pay special attention to this. This procedure is private to the class, meaning no code outside the class can access it. However, it ensures data translation, preserving functionality. If you've been using this indicator to translate data via the class, you won't encounter any issues.

To understand this, look at line 170. This function translates the indicator's data. Note that while the internal code has been modified, its interface remains unchanged. Previously, six values were returned, but now we return just one. Pay close attention to how we handle this. In line 179, we retrieve a value from the indicator buffer. If a value is returned, line 181 assigns it to the translation system. In line 182, we capture and translate the value to determine the button status. Notice the index used and the bit length. Then, in line 183, the data is sent to a procedure that converts screen coordinates X and Y into types expected for use in MetaTrader 5 or other applications using the mouse indicator as a helper. Despite the changes, they remain confined to the function, leaving everything outside the class unaffected.

Now for a more complex but essential aspect: line 191 introduces a procedure that writes data into the indicator buffer. Previously, this procedure was part of the indicator code. However, for practical reasons, I moved it into the class, primarily to restrict access to the class data.

Let's examine this step-by-step. In line 193, we declare the compression variable. Line 195 starts the data compression process. Pay close attention to understand how indices are used when accessing arrays.

The \_8b array contains 8 positions, while the \_16b array contains 4. Both start at index zero. However, note that index zero of the \_16b array corresponds to indices zero and one of the \_8b array. For example, in line 195, when using index zero in \_8b, we occupy index zero of \_16b. Since the required information only needs \_8b index zero, \_8b index one remains vacant. Nevertheless, \_16b index zero cannot be reused, as it requires two \_8b indices for its construction.

Thus, \_8b index one remains vacant, while values for the mouse position start from \_8b index two. As \_8b index two corresponds to \_16b index one, line 196 references this index, logically and functionally separating the data. Line 197 follows the same indexing principle. Despite appearances, not all bits are occupied. Bits in \_16b index three, \_8b index one, and six bits in the two SHORT values for X and Y positions remain unused. These unused bits can be repurposed for other data if desired.

Please see the image below for clarity.

![Image](https://c.mql5.com/2/110/Image_01__5.png)

The image illustrates the buffer contents byte-by-byte, with each byte representing 8 bits. Blue areas indicate occupied bits, while white areas are free for future data. X represents the graphical X-coordinate, and Y represents the Y-coordinate. Hopefully by looking at this image you will have a better understanding of what I am actually doing.

As for the rest of the mouse indicator code, no detailed explanation is necessary since it remains relatively straightforward. However, because the code has undergone modifications, the updated version is provided below. It will ensure everything functions as intended.

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
013.    private   :
014. //+------------------------------------------------------------------+
015.       struct st00
016.       {
017.          eStatusMarket  Status;
018.          MqlRates       Rate;
019.          string         szInfo;
020.          color          corP,
021.                         corN;
022.          int            HeightText;
023.          bool           bvT, bvD, bvP;
024.          datetime       TimeDevice;
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
048.                ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn1, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 18);
049.                ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn1, OBJPROP_TEXT, m_Info.szInfo);
050.             }
051.             if (m_Info.bvD)
052.             {
053.                v1 = NormalizeDouble((((GetInfoMouse().Position.Price - m_Info.Rate.close) / m_Info.Rate.close) * 100.0), 2);
054.                ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 1);
055.                ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
056.                ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_TEXT, StringFormat("%.2f%%", MathAbs(v1)));
057.             }
058.             if (m_Info.bvP)
059.             {
060.                v1 = NormalizeDouble((((iClose(GetInfoTerminal().szSymbol, PERIOD_D1, 0) - m_Info.Rate.close) / m_Info.Rate.close) * 100.0), 2);
061.                ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 1);
062.                ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
063.                ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_TEXT, StringFormat("%.2f%%", MathAbs(v1)));
064.             }
065.          }
066. //+------------------------------------------------------------------+
067. inline void CreateObjInfo(EnumEvents arg)
068.          {
069.             switch (arg)
070.             {
071.                case evShowBarTime:
072.                   C_Mouse::CreateObjToStudy(2, 110, def_ExpansionBtn1, clrPaleTurquoise);
073.                   m_Info.bvT = true;
074.                   break;
075.                case evShowDailyVar:
076.                   C_Mouse::CreateObjToStudy(2, 53, def_ExpansionBtn2);
077.                   m_Info.bvD = true;
078.                   break;
079.                case evShowPriceVar:
080.                   C_Mouse::CreateObjToStudy(58, 53, def_ExpansionBtn3);
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
093.                   sz = def_ExpansionBtn1;
094.                   m_Info.bvT = false;
095.                   break;
096.                case evHideDailyVar:
097.                   sz = def_ExpansionBtn2;
098.                   m_Info.bvD   = false;
099.                   break;
100.                case evHidePriceVar:
101.                   sz = def_ExpansionBtn3;
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
132.                case eCloseMarket :
133.                   m_Info.szInfo = "Closed Market";
134.                   break;
135.                case eInReplay    :
136.                case eInTrading   :
137.                   if ((dt = GetBarTime()) < ULONG_MAX)
138.                   {
139.                      m_Info.szInfo = TimeToString(dt, TIME_SECONDS);
140.                      break;
141.                   }
142.                case eAuction     :
143.                   m_Info.szInfo = "Auction";
144.                   break;
145.                default           :
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
186. #undef def_ExpansionBtn3
187. #undef def_ExpansionBtn2
188. #undef def_ExpansionBtn1
189. #undef def_ExpansionPrefix
190. #undef def_MousePrefixName
191. //+------------------------------------------------------------------+
```

C\_Study.mqh file source code

The C\_Study class remains virtually unchanged. Therefore, I no further explanation is required for it. Let's now look at the indicator code shown below.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "This is an indicator for graphical studies using the mouse."
04. #property description "This is an integral part of the Replay / Simulator system."
05. #property description "However it can be used in the real market."
06. #property version "1.56"
07. #property icon "/Images/Market Replay/Icons/Indicators.ico"
08. #property link "https://www.mql5.com/pt/articles/12000"
09. #property indicator_chart_window
10. #property indicator_plots 0
11. #property indicator_buffers 1
12. //+------------------------------------------------------------------+
13. #include <Market Replay\Auxiliar\Study\C_Study.mqh>
14. //+------------------------------------------------------------------+
15. C_Study *Study       = NULL;
16. //+------------------------------------------------------------------+
17. input long   user00   = 0;                                //ID
18. input C_Study::eStatusMarket user01 = C_Study::eAuction;  //Market Status
19. input color user02   = clrBlack;                          //Price Line
20. input color user03   = clrPaleGreen;                      //Positive Study
21. input color user04   = clrLightCoral;                     //Negative Study
22. //+------------------------------------------------------------------+
23. C_Study::eStatusMarket m_Status;
24. int m_posBuff = 0;
25. double m_Buff[];
26. //+------------------------------------------------------------------+
27. int OnInit()
28. {
29.    ResetLastError();
30.    Study = new C_Study(user00, "Indicator Mouse Study", user02, user03, user04);
31.    if (_LastError != ERR_SUCCESS) return INIT_FAILED;
32.    if ((*Study).GetInfoTerminal().szSymbol != def_SymbolReplay)
33.    {
34.       MarketBookAdd((*Study).GetInfoTerminal().szSymbol);
35.       OnBookEvent((*Study).GetInfoTerminal().szSymbol);
36.       m_Status = C_Study::eCloseMarket;
37.    }else
38.       m_Status = user01;
39.    SetIndexBuffer(0, m_Buff, INDICATOR_DATA);
40.    ArrayInitialize(m_Buff, EMPTY_VALUE);
41.
42.    return INIT_SUCCEEDED;
43. }
44. //+------------------------------------------------------------------+
45. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
46. {
47.    m_posBuff = rates_total;
48.    (*Study).Update(m_Status);
49.
50.    return rates_total;
51. }
52. //+------------------------------------------------------------------+
53. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
54. {
55.    (*Study).DispatchMessage(id, lparam, dparam, sparam);
56.    (*Study).SetBuffer(m_posBuff, m_Buff);
57.
58.    ChartRedraw((*Study).GetInfoTerminal().ID);
59. }
60. //+------------------------------------------------------------------+
61. void OnBookEvent(const string &symbol)
62. {
63.    MqlBookInfo book[];
64.    C_Study::eStatusMarket loc = m_Status;
65.
66.    if (symbol != (*Study).GetInfoTerminal().szSymbol) return;
67.    MarketBookGet((*Study).GetInfoTerminal().szSymbol, book);
68.    m_Status = (ArraySize(book) == 0 ? C_Study::eCloseMarket : C_Study::eInTrading);
69.    for (int c0 = 0; (c0 < ArraySize(book)) && (m_Status != C_Study::eAuction); c0++)
70.       if ((book[c0].type == BOOK_TYPE_BUY_MARKET) || (book[c0].type == BOOK_TYPE_SELL_MARKET)) m_Status = C_Study::eAuction;
71.    if (loc != m_Status) (*Study).Update(m_Status);
72. }
73. //+------------------------------------------------------------------+
74. void OnDeinit(const int reason)
75. {
76.    if (reason != REASON_INITFAILED)
77.    {
78.       if ((*Study).GetInfoTerminal().szSymbol != def_SymbolReplay)
79.          MarketBookRelease((*Study).GetInfoTerminal().szSymbol);
80.    }
81.    delete Study;
82. }
83. //+------------------------------------------------------------------+
```

Source code of the mouse indicator

Please note that there are some differences here from the code that was there before. But they are not so significant that you cannot understand the code. There are, however, two points that I think deserve brief comment. They are in lines 47 and 56.

In line 47 you can see that, unlike before, we now only save the value that MetaTrader 5 provides and do not make any changes to it, as that will be done elsewhere.

Then, in line 56, we pass information to the C\_Mouse class, where we will correct the situation and write it to the indicator buffer. Notice that things have become much simpler here precisely because we have moved all the complexity into the class.

### Conclusion

After all the modifications and adaptations of the code, we now have the ability to practically use the mouse indicator in the replay/simulator system we are developing. We can now use the message exchange system between the involved applications and, at the same time, use buffer reading to pass information between them. The only limitation we have is that we must always use one single position in the buffer. This is necessary in order not to overload the event system. But this will be a next topic for discussion in the future.

In the video below, you can see a demonstration of the system in operation. The attachment provides the compiled code so you can run tests and understand what is actually happening. Before I forget to mention it, you can test the range error by replacing the mouse indicator with the one from the previous article and running the service from this article. You will see the mentioned error appear.

One last point: although this article doesn't mention the service code that will be used, I'll leave its explanation for the next one, since we'll be using the same code as a springboard for what we'll actually be developing. So be patient and wait for the next article, because every day everything becomes more and more interesting.

Demonstração Parte 56 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12000)

MQL5.community

1.91K subscribers

[Demonstração Parte 56](https://www.youtube.com/watch?v=xxXPxiRrTBs)

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

0:00 / 1:14

•Live

•

Demo video

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12000](https://www.mql5.com/pt/articles/12000)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12000.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12000/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/479723)**

![Mastering Log Records (Part 2): Formatting Logs](https://c.mql5.com/2/108/logify60x60.png)[Mastering Log Records (Part 2): Formatting Logs](https://www.mql5.com/en/articles/16833)

In this article, we will explore how to create and apply log formatters in the library. We will see everything from the basic structure of a formatter to practical implementation examples. By the end, you will have the necessary knowledge to format logs within the library, and understand how everything works behind the scenes.

![MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://c.mql5.com/2/110/MQL5_Trading_Toolkit_Part_6___LOGO.png)[MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)

Learn how to create an EX5 module of exportable functions that seamlessly query and save data for the most recently filled pending order. In this comprehensive step-by-step guide, we will enhance the History Management EX5 library by developing dedicated and compartmentalized functions to retrieve essential properties of the last filled pending order. These properties include the order type, setup time, execution time, filling type, and other critical details necessary for effective pending orders trade history management and analysis.

![Price Action Analysis Toolkit Development (Part 7): Signal Pulse EA](https://c.mql5.com/2/110/Price_Action_Analysis_Toolkit_Development_Part_7____LOGO.png)[Price Action Analysis Toolkit Development (Part 7): Signal Pulse EA](https://www.mql5.com/en/articles/16861)

Unlock the potential of multi-timeframe analysis with 'Signal Pulse,' an MQL5 Expert Advisor that integrates Bollinger Bands and the Stochastic Oscillator to deliver accurate, high-probability trading signals. Discover how to implement this strategy and effectively visualize buy and sell opportunities using custom arrows. Ideal for traders seeking to enhance their judgment through automated analysis across multiple timeframes.

![MetaTrader 5 on macOS](https://c.mql5.com/2/0/1045_13.png)[MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)

We provide a special installer for the MetaTrader 5 trading platform on macOS. It is a full-fledged wizard that allows you to install the application natively. The installer performs all the required steps: it identifies your system, downloads and installs the latest Wine version, configures it, and then installs MetaTrader within it. All steps are completed in the automated mode, and you can start using the platform immediately after installation.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/12000&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069681793415710976)

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