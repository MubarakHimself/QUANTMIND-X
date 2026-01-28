---
title: Developing a Replay System (Part 55): Control Module
url: https://www.mql5.com/en/articles/11988
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:43:29.042918
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/11988&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069689206529263903)

MetaTrader 5 / Examples


### Introduction

In the previous article " [Developing a Replay System (Part 54): The Birth of the First Module](https://www.mql5.com/en/articles/11971)", we have assembled the first real module of our entire new replay/simulator system. In addition to the possibility of using them in the system we are developing, we will also be able to apply the modules individually and in a personalized manner to avoid the need for a large amount of programming to create such a system. Either way, once the module is built, we will be able to easily customize it without the need for recompilation.

To do this, you will only need to send a message to the module to change its appearance or the way it works, which can be easily done with a simple script.

So, based on what we have looked at in the last articles, we have the opportunity to create a system that can be used on both a real account and a demo account. But that's not the only thing we can do; among other things, we can create a replay/simulator system that will behave very similar to what you would see on a real or demo account.

However, the main advantage of this new model that we will start using is that you will be able to use the same tools and applications both in the replay/simulator system and in your daily work in MetaTrader 5 for training on a demo account or for making trades on a real one.

Now that our mouse indicator is ready, we can start creating, or rather adapting, our control indicator so that it starts working in a modular manner. A brief explanation is needed regarding what has just been mentioned.

Until recently, the replay/simulator system used global terminal variables to provide some level of communication between programs needed to interact, control, and access the replay/simulator service.

Since we start using a module system where messaging is done through user events, we will no longer need to use global terminal variables. With this in mind, we can now remove all global terminal variables that were used previously. However, in this case, we will need to adapt the system so that information continues to flow between programs.

Modelling an information transmission system is a task that requires great attention and caution, since there is simply no possibility of reading the information later. If a program or application is not on the chart when information is received via a custom event, that information will be lost. Therefore, additional mechanisms are needed to resend the same information until we have confirmation that it was received by the intended application or program.

Based on this criterion, it was decided that the replay/simulator system should include three core programs required for minimal functioning. Of these programs, only two will be visible to the user: the program responsible for the service itself, and the mouse indicator. The control indicator will be treated as a service program resource, so it cannot be used without providing services.

So, with that brief explanation in mind, let's take a look at how the control indicator has been modified so that it can start doing its job, that is manage the replay/simulator service.

### Modifying the Control Indicator

There weren't many changes that needed to be made to the control indicator, since many of them had already been made in the previous step when we started removing the terminal's global variables. However, without a clear understanding of how exactly the message exchange will take place, it is also impossible to understand how the system will perform its tasks.

So, let's try to get it all straight from the start so you don't get lost when reading future articles.

When the indicator is placed on the chart, the user can configure several parameters. These parameters refer to the indicator input variables. However, at certain moments, these same variables are more of a hindrance than a help. Don't get me wrong, I'm not trying to push for any radical changes. But when we allow the user to access a variable to configure the indicator (in this case) in advance, we open the door to potential problems.

If you put aside the fact that the user might accidentally change something they shouldn't, these same variables turn out to be very useful. So much so that we use them to pass the chart ID to the indicator. Not that the indicator really needs this information, but it is worth remembering that at the moment the indicator is placed on the chart, its ID may differ from the expected one for the objects. This was already discussed in one of the previous articles in this series.

Although we can use the messaging system to pass the ID to the indicator, since the chart is opened by the service and the latter knows its ID, this would simply unnecessarily complicate the code of both the service and the indicator. For this reason, I will leave everything as it has been done so far. However, we will need to make some small changes to the control indicator code since we will no longer be using global terminal variables to pass data.

Below is the full code for the C\_Control.mqh file. Since most of the code has already been explained in previous articles, we will only focus on the new parts that require explanation.

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
023. #define def_PrefixCtrlName    "MarketReplayCTRL_"
024. #define def_PosXObjects       120
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
037.       enum eObjectControl {ePlay, eLeft, eRight, ePin, eNull};
038. //+------------------------------------------------------------------+
039.       struct st_00
040.       {
041.          string  szBarSlider,
042.                  szBarSliderBlock;
043.          int     Minimal;
044.       }m_Slider;
045.       struct st_01
046.       {
047.          C_DrawImage *Btn;
048.          bool         state;
049.          int          x, y, w, h;
050.       }m_Section[eObjectControl::eNull];
051.       C_Mouse   *m_MousePtr;
052. //+------------------------------------------------------------------+
053. inline void CreteBarSlider(int x, int size)
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
100. inline void PositionPinSlider(int p)
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
114. inline eObjectControl CheckPositionMouseClick(int &x, int &y)
115.          {
116.             C_Mouse::st_Mouse InfoMouse;
117.
118.             InfoMouse = (*m_MousePtr).GetInfoMouse();
119.             x = InfoMouse.Position.X_Graphics;
120.             y = InfoMouse.Position.Y_Graphics;
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
150.             m_Slider.Minimal = INT_MIN;
151.          }
152. //+------------------------------------------------------------------+
153.       ~C_Controls()
154.          {
155.             for (eObjectControl c0 = ePlay; c0 < eNull; c0++) delete m_Section[c0].Btn;
156.             ObjectsDeleteAll(GetInfoTerminal().ID, def_PrefixCtrlName);
157.             delete m_MousePtr;
158.          }
159. //+------------------------------------------------------------------+
160.       void SetBuff(const int rates_total, double &Buff[])
161.          {
162.             uCast_Double info;
163.
164.             info._int[0] = m_Slider.Minimal;
165.             info._int[1] = (m_Section[ePlay].state ? INT_MAX : INT_MIN);
166.             Buff[rates_total - 1] = info.dValue;
167.          }
168. //+------------------------------------------------------------------+
169.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
170.          {
171.             int x, y;
172.             static int iPinPosX = -1, six = -1, sps;
173.             uCast_Double info;
174.
175.             switch (id)
176.             {
177.                case (CHARTEVENT_CUSTOM + evCtrlReplayInit):
178.                   info.dValue = dparam;
179.                   iPinPosX = m_Slider.Minimal = info._int[0];
180.                   if (info._int[1] == 0) SetUserError(C_Terminal::ERR_Unknown); else
181.                   {
182.                      SetPlay(info._int[1] == INT_MAX);
183.                      if (info._int[1] == INT_MIN) CreateCtrlSlider();
184.                   }
185.                   break;
186.                case CHARTEVENT_OBJECT_DELETE:
187.                   if (StringSubstr(sparam, 0, StringLen(def_PrefixCtrlName)) == def_PrefixCtrlName)
188.                   {
189.                      if (sparam == (def_PrefixCtrlName + EnumToString(ePlay)))
190.                      {
191.                         delete m_Section[ePlay].Btn;
192.                         m_Section[ePlay].Btn = NULL;
193.                         SetPlay(m_Section[ePlay].state);
194.                      }else
195.                      {
196.                         RemoveCtrlSlider();
197.                         CreateCtrlSlider();
198.                      }
199.                   }
200.                   break;
201.                case CHARTEVENT_MOUSE_MOVE:
202.                   if ((*m_MousePtr).CheckClick(C_Mouse::eClickLeft))   switch (CheckPositionMouseClick(x, y))
203.                   {
204.                      case ePlay:
205.                         SetPlay(!m_Section[ePlay].state);
206.                         if (m_Section[ePlay].state)
207.                         {
208.                            RemoveCtrlSlider();
209.                            m_Slider.Minimal = iPinPosX;
210.                         }else CreateCtrlSlider();
211.                         break;
212.                      case eLeft:
213.                         PositionPinSlider(iPinPosX = (iPinPosX > m_Slider.Minimal ? iPinPosX - 1 : m_Slider.Minimal));
214.                         break;
215.                      case eRight:
216.                         PositionPinSlider(iPinPosX = (iPinPosX < def_MaxPosSlider ? iPinPosX + 1 : def_MaxPosSlider));
217.                         break;
218.                      case ePin:
219.                         if (six == -1)
220.                         {
221.                            six = x;
222.                            sps = iPinPosX;
223.                            ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
224.                         }
225.                         iPinPosX = sps + x - six;
226.                         PositionPinSlider(iPinPosX = (iPinPosX < m_Slider.Minimal ? m_Slider.Minimal : (iPinPosX > def_MaxPosSlider ? def_MaxPosSlider : iPinPosX)));
227.                         break;
228.                   }else if (six > 0)
229.                   {
230.                      six = -1;
231.                      ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
232.                   }
233.                   break;
234.             }
235.             ChartRedraw(GetInfoTerminal().ID);
236.          }
237. //+------------------------------------------------------------------+
238. };
239. //+------------------------------------------------------------------+
240. #undef def_PosXObjects
241. #undef def_ButtonPlay
242. #undef def_ButtonPause
243. #undef def_ButtonLeft
244. #undef def_ButtonRight
245. #undef def_ButtonPin
246. #undef def_PrefixCtrlName
247. #undef def_PathBMP
248. //+------------------------------------------------------------------+
```

Source code of C\_Control.mqh

There are some seemingly strange things in the code. The first one is in line 150, where we specify the minimum slider offset value using a constant defined in MQL5, INT\_MIN. This is a negative value, which is the minimum possible for an integer variable. Why did I do it? It's a bit difficult to understand the reason now, so please be patient as there are other things that need to be sorted out to fully understand line 150.

The next thing to pay attention to is in line 160, where we have a routine for writing data to the control indicator buffer. At this stage, we are recording only two values. I don't know if we'll need to write more values in the future, but for now these two values will be compressed into a single double value so that it will only take one position in the buffer.

For this compression we use a union declared in line 162. Then, in line 164, we place the value that will be adjusted by the user when manipulating the slider. Attention: we will be storing the slider position changed by the user. In line 165, we specify the status of the control indicator, that is, whether it is in play or pause mode.

There is something important here. When we indicate that the user is in play mode, i.e. that the user pressed the play button, a certain value will be saved. Another value will be used for the pause mode. The values we will use are the extreme ends of the range possible for the integer data type. In this way, we avoid the ambiguity of using the null value and, at the same time, guarantee the integrity of the information, which facilitates its subsequent testing.

This point is extremely important: if you do not understand this at the stage of constructing the control indicator, you may have difficulties with future explanations. Remember: the information will not come directly from the control indicator to the service, it will have to go through the channel we will use, which is the buffer. I will return to this point shortly, as some details will need to be clarified.

Finally, in line 166, we save the compressed value at a specific position in the indicator buffer. In another article within this series, I already explained the reason for saving data in this particular position, so if you have any doubts, I recommend reading previous articles for clarification.

The next piece of code that really requires explanation is the message handler, which starts at line 169. It has two specific points that deserve attention. Let's start with the one that involves fewer details, but still relates to what was explained above. So, let's jump to line 209.

There's something interesting about this line: we save the value the user sets when moving the slider in the m\_Slider.Minimal variable. The reason is to simplify the process by allowing everything to be concentrated in key points of the code. If line 209 didn't exist, we would have to perform a check somewhere in the code to pass the user-set position to the buffer. Or, even worse, we would have to find some way to directly pass the user-specified value to the service without using terminal global variables. Previously we used a global variable for this, but now we will work with a buffer. Therefore, to avoid repeated checking and adjustments, we place the value in an easily accessible place. Please note that this value will only be saved here after the user clicks the play button.

Now, let's go back to the previous code content and look at line 177 where we have a custom event whose purpose is to initialize the control indicator so that the slider can be positioned properly.

This custom event will be fired from time to time, but note that the value containing the data will be present in a double value and in a packed form. Therefore, we need to translate this information while ensuring its safety and security. The information received follows the same principle as the information we save in the buffer. However, there is a nuance here: we check its integrity.

Note that there is a small test in line 180. It checks if the value indicating whether the system is in play or pause mode is zero. If it is zero, something is wrong, the control indicator is receiving incorrect data and an error has occurred. This is why SetUserError is called. Usually this function is not performed, but if it does, appropriate measures will have to be taken. We will these measures discussed later in the indicator code.

If everything is ok, we will perform two more actions. The first is in line 182, where we call a function responsible for displaying the play or pause button. The second is the check shown in line 183. If the value is minimum, it means we are in pause mode, so we need to recreate the slider so the user can adjust it.

That's basically it. Once the indicator is added to the chart, it will not work until it is initialized by a custom event. But how this will be done, we will consider later. The interaction between the mouse pointer and the control indicator will generate the entire message cycle necessary to effectively control the replay/simulator service.

Now let's look at the control indicator code below in full. Pay close attention to the explanation of this code, as at this point we are laying the foundation for all future modules of this system.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property description "Control indicator for the Replay-Simulator service."
05. #property description "This one doesn't work without the service loaded."
06. #property version   "1.55"
07. #property link "https://www.mql5.com/pt/articles/11988"
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
19. int    m_RatesTotal;
20. //+------------------------------------------------------------------+
21. int OnInit()
22. {
23.    ResetLastError();
24.    if (CheckPointer(control = new C_Controls(user00, "Market Replay Control", new C_Mouse(user00, "Indicator Mouse Study"))) == POINTER_INVALID)
25.       SetUserError(C_Terminal::ERR_PointerInvalid);
26.    if (_LastError != ERR_SUCCESS)
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
50.    (*control).SetBuff(m_RatesTotal, m_Buff);
51. }
52. //+------------------------------------------------------------------+
53. void OnDeinit(const int reason)
54. {
55.    switch (reason)
56.    {
57.       case REASON_TEMPLATE:
58.          Print("Modified template. Replay/simulation system shutting down.");
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
```

Source code of the control indicator

Pay attention to line 10 where we tell MQL5 that we will need a buffer, and to line 9 where we specify that we will not display any information on the chart. This means that the buffer will be internal to the indicator and invisible to the user, but accessible to any code that knows how to use it.

Since we will be using a buffer, we need to declare it. We do this first in line 18, and then in line 31 we declare the buffer so that it is accessible outside the indicator. In line 32 we make sure that the buffer is completely empty. Now please pay close attention to the following, because it is important.

When a user interacts with the MetaTrader 5 platform, for example by changing the timeframe, MetaTrader 5 clears everything from the chart and then reloads it. In this case, all code is reset. For any indicator this means a new call to the OnInit event, which performs the entire process of its re-initialization. Our control indicator does not actually do anything, or rather, it does not perform any specific calculations. Its only function is to provide the user with the means to interact with and control the service that is used to display bars on a chart in a custom symbol.

So when the user changes the timeframe, all the values inside the indicator will be lost. We need to ensure everything is done in an appropriate manner. The service that is controlled by the indicator has no idea what is happening on the chart, just as the indicator does not know what the service is doing. The way to inform the service and the indicator what each is doing is to exchange messages between them.

Previously, this was done through a global terminal variable. But now we are creating a different way and we need to ensure that regardless of the user's actions on the chart, both the indicator and the service remain consistent and aware of what is happening. Therefore, to inform the service about the indicator state, we use the indicator buffer, where we place all the information related to the processes occurring in it.

The indicator knows what the service does through the input parameters, namely those declared in line 16, and through the user events that are handled in the indicator's OnChartEvent call.

Passing data to the indicator via input parameters after it is already on the chart is not practical. I'm not saying it can't be done, but it's impractical and we're not going to do it. Once the service places the indicator on the chart, you will lose the ability to interact with it through input parameters. That's why use here custom events.

Now in more detail. When the user changes the timeframe, the indicator loses information about the previous status and the shift position. However, the service have this information. But how can we ensure that the service can pass this data to the indicator so that it maintains its integrity? The service does not know when MetaTrader 5 restores the indicator on the chart, but the service can see the indicator buffer. This is the main trick.

When the indicator is placed on a chart, its buffer is initially set to zero. When the constructor of the C\_Controls class is executed, a specific value is initialized in line 150 (to understand this, look at the class code). But this value will not be written to the buffer until OnChartEvent is called, and then, in line 50 of the indicator code, the buffer is updated.

So when the service reads the buffer after MetaTrader 5 restores the control indicator on the chart, it will see zero or abnormal values. At this point, a special event will be launched for MetaTrader 5 and the service will inform the indicator about the updated values so that they are correctly displayed on the screen. This will ensure that the buttons and sliders are displayed correctly.

If we tried to do this any other way, we would have to come up with ways to restore the lost data. We would end up creating different solutions with the same result: indicator initialization. However, some of these solutions could be vulnerable to manipulation by the user, thus complicating the control over communication between the service and the indicator. Our solution creates an additional layer of security while ensuring that the information is only accessible to those parts that actually need to read and use it.

What I just explained may seem very confusing and exotic to most people, especially those who are just starting to program. The idea of message exchange and controlled initializationis something that many people have probably never heard of. So how can we demonstrate that this actually works in practice? For this, let's use the control indicator and the mouse indicator. But before we can see the real system working, we need to create something to demonstrate and understand the idea itself.

For this, we will use much simpler code but effective enough to clarify the main concept. This code is presented below.

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property copyright "Daniel Jose"
04. #property description "Data synchronization demo service."
05. #property version   "1.00"
06. //+------------------------------------------------------------------+
07. #include <Market Replay\Defines.mqh>
08. //+------------------------------------------------------------------+
09. #define def_IndicatorControl   "Indicators\\Market Replay.ex5"
10. #resource "\\" + def_IndicatorControl
11. //+------------------------------------------------------------------+
12. input string user00 = "BOVA11";    //Symbol
13. //+------------------------------------------------------------------+
14. #define def_Loop ((!_StopFlag) && (ChartSymbol(id) != ""))
15. //+------------------------------------------------------------------+
16. void OnStart()
17. {
18.    uCast_Double info;
19.    long id;
20.    int handle, iPos, iMode;
21.    double Buff[];
22.
23.    SymbolSelect(user00, true);
24.    id = ChartOpen(user00, PERIOD_H1);
25.    if ((handle = iCustom(ChartSymbol(id), ChartPeriod(id), "::" + def_IndicatorControl, id)) != INVALID_HANDLE)
26.       ChartIndicatorAdd(id, 0, handle);
27.    IndicatorRelease(handle);
28.    if ((handle = iCustom(ChartSymbol(id), ChartPeriod(id), "\\Indicators\\Mouse Study.ex5", id)) != INVALID_HANDLE)
29.       ChartIndicatorAdd(id, 0, handle);
30.    IndicatorRelease(handle);
31.    Print("Service maintaining sync state...");
32.    iPos = 0;
33.    iMode = INT_MIN;
34.    while (def_Loop)
35.    {
36.       while (def_Loop && ((handle = ChartIndicatorGet(id, 0, "Market Replay Control")) == INVALID_HANDLE)) Sleep(50);
37.       info.dValue = 0;
38.       if (CopyBuffer(handle, 0, 0, 1, Buff) == 1) info.dValue = Buff[0];
39.       IndicatorRelease(handle);
40.       if (info._int[0] == INT_MIN)
41.       {
42.          info._int[0] = iPos;
43.          info._int[1] = iMode;
44.          EventChartCustom(id, evCtrlReplayInit, 0, info.dValue, "");
45.       }else if (info._int[1] != 0)
46.       {
47.          iPos = info._int[0];
48.          iMode = info._int[1];
49.       }
50.       Sleep(250);
51.    }
52.    ChartClose(id);
53.    Print("Finished service...");
54. }
55. //+------------------------------------------------------------------+
```

Source code of the demonstration service

Notice line 10 where we convert the control indicator into a service resource. Thus, there is no need to include it in the list of indicators, as it is of no use for anything other than our system. In line 12, we indicate the symbol to test the system. Make sure to use a valid character as its validity will not be checked later. In line 14, we have a definition that will serve to check some conditions to be able to shut down the service properly.

In line 23, we place the symbol in the market watch window in case it is not there. In line 24, we open a graphics window with the symbol specified by the user (in this case you). Once this is done, we will have a chart ID and we can use it to place indicators on the chart.

So, in line 25 we place the control indicator on the newly opened chart. Then, in line 29, we add the mouse indicator. Please note that the mouse indicator will be in the list of indicators, but the control indicator will not. However, we need both for full testing.

In line 31, we inform the terminal that the service will be active and will monitor what is happening on the chart.

By this point the mouse indicator should already be visible on the chart. However, the control indicator will not be visible, although it will be listed among the indicators present on the chart. I have explained earlier how the control indicator is initialized. At this point, its buffer will contain random values that have no meaning for us. For this reason, we cannot interact with it. But if the control indicator was initialized correctly, and the buffer was already written, we will get a certain value. So, in line 34, we enter a loop that will continue as long as the conditions defined in line 14 are met.

Please note that in line 36 we check whether the control indicator is actually added to the chart. But why do we perform this check and wait for it to complete? The reason is that the code can execute much faster than what actually happens. So we need to somehow let MetaTrader 5 stabilize the situation, and for that we use this loop in line 36.

Once everything is ok, we try to read the control indicator buffer. Once again, I would like to remind you that the indicator will not be visible on the chart.

If the buffer is read successfully, some non-zero value will be added to the info.dValue variable. At this point we can check the status of the control indicator. In line 40, we check if it has already been initialized. Since this is the first time, we see that the indicator has not yet been initialized. In lines 42 and 43, we generate a value to pass to the indicator and send a request to MetaTrader 5 to generate a custom event on the chart. This event is displayed in line 44 where we pass a message to the control indicator to initialize it.

At any other moment we will check whether the indicator is in pause or play status. If it is in one of these states, line 45 will allow the user-specified values to be saved in the indicator. Thus, when the user changes the timeframe, the check in line 40 will return a true value, and the values stored in the service will be passed back to the indicator, allowing it to initialize correctly even when the timeframe changes.

Finally, we close the chart using line 52 and print a message about the service termination in line 53.

In the video below, you can see the system running if you don't want to try it yet. I also provide the executable files attached so you can see how it all works in practice.

YouTube

Demo video

### Conclusion

In this article, we have begun to lay the foundation for what will be covered in future articles. I understand that the material is quite intensive, so study it calmly because further on everything will get more complicated.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11988](https://www.mql5.com/pt/articles/11988)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11988.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11988/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/479590)**

![Adaptive Social Behavior Optimization (ASBO): Schwefel, Box-Muller Method](https://c.mql5.com/2/84/Adaptive_Social_Behavior_Optimization___LOGO.png)[Adaptive Social Behavior Optimization (ASBO): Schwefel, Box-Muller Method](https://www.mql5.com/en/articles/15283)

This article provides a fascinating insight into the world of social behavior in living organisms and its influence on the creation of a new mathematical model - ASBO (Adaptive Social Behavior Optimization). We will examine how the principles of leadership, neighborhood, and cooperation observed in living societies inspire the development of innovative optimization algorithms.

![Developing A Swing Entries Monitoring (EA)](https://c.mql5.com/2/109/Developing_A_Swing_Entries_Monitoring___LOGO.png)[Developing A Swing Entries Monitoring (EA)](https://www.mql5.com/en/articles/16563)

As the year approaches its end, long-term traders often reflect on market history to analyze its behavior and trends, aiming to project potential future movements. In this article, we will explore the development of a long-term entry monitoring Expert Advisor (EA) using MQL5. The objective is to address the challenge of missed long-term trading opportunities caused by manual trading and the absence of automated monitoring systems. We'll use one of the most prominently traded pairs as an example to strategize and develop our solution effectively.

![Neural Networks in Trading: Piecewise Linear Representation of Time Series](https://c.mql5.com/2/82/Neural_networks_are_simple_Piecewise_linear_representation_of_time_series__LOGO.png)[Neural Networks in Trading: Piecewise Linear Representation of Time Series](https://www.mql5.com/en/articles/15217)

This article is somewhat different from my earlier publications. In this article, we will talk about an alternative representation of time series. Piecewise linear representation of time series is a method of approximating a time series using linear functions over small intervals.

![Build Self Optimizing Expert Advisors in MQL5  (Part 3): Dynamic Trend Following and Mean Reversion Strategies](https://c.mql5.com/2/109/Build_Self_Optimizing_Expert_Advisors_in_MQL5_Part_3__LOGO.png)[Build Self Optimizing Expert Advisors in MQL5 (Part 3): Dynamic Trend Following and Mean Reversion Strategies](https://www.mql5.com/en/articles/16856)

Financial markets are typically classified as either in a range mode or a trending mode. This static view of the market may make it easier for us to trade in the short run. However, it is disconnected from the reality of the market. In this article, we look to better understand how exactly financial markets move between these 2 possible modes and how we can use our new understanding of market behavior to gain confidence in our algorithmic trading strategies.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/11988&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069689206529263903)

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