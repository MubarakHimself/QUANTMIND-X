---
title: Developing a Replay System (Part 51): Things Get Complicated (III)
url: https://www.mql5.com/en/articles/11877
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:07:11.913738
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/11877&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070023689992343213)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 50): Things Get Complicated (II)](https://www.mql5.com/en/articles/11871), we started modifying the control indicator further to ensure it stays within the chart limits. Not within any chart, but the one that was opened by the replay/simulator service. The main thing that was implemented there was the ability to use a custom template, not the one required by the system.

Allowing this kind of thing makes using the whole replay/simulator system much more enjoyable and suitable for those who actually want to use the system to do some research. You can create a template, use it in the replay/simulator system, and then use the same template on a live account. For this reason, we are changing the system now.

However, the interaction between the control indicator and the user was not very good. This is due to the fact that there are several duplicate points in the system. But the main problem is that the system was not as secure and stable as it seemed. This happened because the development stage, which was the first stage, did not take into account the fact that some users could try to use the system in a way that was not intended. But with the current change in direction and positioning, the level of security and stability will start to appropriate.

While we are still dealing with an unstable system in terms of user experience at the moment, this will be fixed soon. Once this is done, the entire system will benefit from it.

If you read the previous article (I encourage you to), you know that we've started using the control indicator directly through the service. After that, the indicator was no longer freely available for placement on any chart.

After calmly analyzing the code, I noticed that we could make some modifications to make it even better. However, some changes will need to be made to other modules that will also be used by the replay/simulator system. For this reason, in this article we will focus on explaining the changes that will be implemented. This is because some of these modules will be available to the user. Therefore, it is important that you understand what can and cannot be manipulated, since you are the user of this system and the programmer who monitors its development. This is necessary to avoid instability in the use of any of the modules. Please remember that some modules can be used on LIVE and DEMO accounts, not just in the replay/simulator system.

You may not think that the changes you see here are that big, but before we push further modifications we need to make the code stable. For this reason, the article may seem vague. But I repeat: you must understand how the system develops. If you do not understand this, you will not be able to use the system properly.

Without further ado, let's get to the modifications.

### Extensive use of modules

The first change we need to make is to the source code of the control indicator. The code discussed before the previous article did not use some modules that had already been developed and created. This caused some inconsistency between what was happening on the control indicator and in other modules of the system.

One of such modules is the mouse indicator. This module was developed to accumulate everything related to the mouse. This way, any other new code will not require any specific testing and analysis. All this will be done by the mouse indicator module.

This indicator was developed some time ago in within this series of articles about the replay/simulator system. However, it was originally designed to work with a real or demo account and is not well suited to work in the current form, but we will talk about this in more detail later.

In the code below you can see how to integrate the mouse indicator module with the control indicator, which already has some fixes and changes to allow for such integration.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property description "Control indicator for the Replay-Simulator service."
05. #property description "This one doesn't work without the service loaded."
06. #property version   "1.51"
07. #property link "https://www.mql5.com/en/articles/11877"
08. #property indicator_chart_window
09. #property indicator_plots 0
10. //+------------------------------------------------------------------+
11. #include <Market Replay\Service Graphics\C_Controls.mqh>
12. //+------------------------------------------------------------------+
13. C_Controls *control = NULL;
14. //+------------------------------------------------------------------+
15. input long user00 = 0;   //ID
16. //+------------------------------------------------------------------+
17. int OnInit()
18. {
19.     u_Interprocess Info;
20.
21.     ResetLastError();
22.     if (CheckPointer(control = new C_Controls(user00, "Market Replay Control", "Indicator Mouse Study")) == POINTER_INVALID)
23.             SetUserError(C_Terminal::ERR_PointerInvalid);
24.     if (_LastError != ERR_SUCCESS)
25.     {
26.             Print("Control indicator failed on initialization.");
27.             return INIT_FAILED;
28.     }
29.     if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.df_Value = 0;
30.     EventChartCustom(user00, C_Controls::ev_WaitOff, 1, Info.df_Value, "");
31.     (*control).Init(Info.s_Infos.isPlay);
32.
33.     return INIT_SUCCEEDED;
34. }
35. //+------------------------------------------------------------------+
36. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
37. {
38.     static bool bWait = false;
39.     u_Interprocess Info;
40.
41.     Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
42.     if (!bWait)
43.     {
44.             if (Info.s_Infos.isWait)
45.             {
46.                     EventChartCustom(user00, C_Controls::ev_WaitOn, 1, 0, "");
47.                     bWait = true;
48.             }
49.     }else if (!Info.s_Infos.isWait)
50.     {
51.             EventChartCustom(user00, C_Controls::ev_WaitOff, 1, Info.df_Value, "");
52.             bWait = false;
53.     }
54.
55.     return rates_total;
56. }
57. //+------------------------------------------------------------------+
58. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
59. {
60.     (*control).DispatchMessage(id, lparam, dparam, sparam);
61. }
62. //+------------------------------------------------------------------+
63. void OnDeinit(const int reason)
64. {
65.     switch (reason)
66.     {
67.             case REASON_TEMPLATE:
68.                     Print("Modified template. Replay/simulation system shutting down.");
69.             case REASON_PARAMETERS:
70.             case REASON_REMOVE:
71.             case REASON_CHARTCLOSE:
72.                     if (ChartSymbol(user00) != def_SymbolReplay) break;
73.                     GlobalVariableDel(def_GlobalVariableReplay);
74.                     ChartClose(user00);
75.                     break;
76.     }
77.     delete control;
78. }
79. //+------------------------------------------------------------------+
```

Source code of the control indicator

You may have noticed that this code is very different from what existed before. But the most noticeable difference is in the OnInit function, where in line 22 we have a completely different declaration from the previous one for the reference to the control class.

You might feel a little lost if you look and try to figure out what line 22 does. But in this line, we do two things:

- First, we convert the control indicator into a module; this module can be used by any other module that needs to know what this indicator does.
- Second, we tell the control indicator that we will no longer use coding to analyze the mouse. In other words, the control indicator will now ask the mouse indicator what the user is doing or about to do. Based on this information, the control indicator will perform the appropriate procedure.

This change or decision to make the mouse indicator responsible for the interaction between the user, the mouse and the graphics is certainly the best way to accomplish this. Duplicating things doesn't make sense because we fix the problem in the mouse indicator, but create problems in the control indicator, and when we put them together, one interferes with the other. This is not what we want. We want both of them to work harmoniously. This way we don't waste time on corrections, as it's enough to edit how the interaction is performed.

Now if you pay attention, you'll notice that the control indicator no longer mentions the C\_Terminal class like it used to. Why? The reason is inheritance. I decided to make the control class a child of the C\_Terminal class. The reason for this action is a little difficult to explain now, but it was necessary because of what we will do later.

With this first explanation of the indicator code in mind, let's look at the control class code.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\Auxiliar\Interprocess.mqh"
005. //+------------------------------------------------------------------+
006. #define def_PathBMP           "Images\\Market Replay\\Control\\"
007. #define def_ButtonPlay        def_PathBMP + "Play.bmp"
008. #define def_ButtonPause       def_PathBMP + "Pause.bmp"
009. #define def_ButtonLeft        def_PathBMP + "Left.bmp"
010. #define def_ButtonLeftBlock   def_PathBMP + "Left_Block.bmp"
011. #define def_ButtonRight       def_PathBMP + "Right.bmp"
012. #define def_ButtonRightBlock  def_PathBMP + "Right_Block.bmp"
013. #define def_ButtonPin         def_PathBMP + "Pin.bmp"
014. #define def_ButtonWait        def_PathBMP + "Wait.bmp"
015. #resource "\\" + def_ButtonPlay
016. #resource "\\" + def_ButtonPause
017. #resource "\\" + def_ButtonLeft
018. #resource "\\" + def_ButtonLeftBlock
019. #resource "\\" + def_ButtonRight
020. #resource "\\" + def_ButtonRightBlock
021. #resource "\\" + def_ButtonPin
022. #resource "\\" + def_ButtonWait
023. //+------------------------------------------------------------------+
024. #define def_PrefixObjectName    "Market Replay _ "
025. #define def_NameObjectsSlider   def_PrefixObjectName + "Slider"
026. #define def_PosXObjects         120
027. //+------------------------------------------------------------------+
028. #include "..\Auxiliar\C_Terminal.mqh"
029. #include "..\Auxiliar\C_Mouse.mqh"
030. //+------------------------------------------------------------------+
031. class C_Controls : private C_Terminal
032. {
033.    protected:
034.            enum EventCustom {ev_WaitOn, ev_WaitOff};
035.    private :
036. //+------------------------------------------------------------------+
037.            string  m_szBtnPlay;
038.            bool    m_bWait;
039.            struct st_00
040.            {
041.                    string  szBtnLeft,
042.                            szBtnRight,
043.                            szBtnPin,
044.                            szBarSlider,
045.                            szBarSliderBlock;
046.                    int     posPinSlider,
047.                            posY,
048.                            Minimal;
049.            }m_Slider;
050.            C_Mouse *m_MousePtr;
051. //+------------------------------------------------------------------+
052. inline void CreateObjectBitMap(int x, int y, string szName, string Resource1, string Resource2 = NULL)
053.                    {
054.                            ObjectCreate(GetInfoTerminal().ID, szName, OBJ_BITMAP_LABEL, 0, 0, 0);
055.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XDISTANCE, def_PosXObjects + x);
056.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YDISTANCE, y);
057.                            ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_BMPFILE, 0, "::" + Resource1);
058.                            ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_BMPFILE, 1, "::" + (Resource2 == NULL ? Resource1 : Resource2));
059.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_ZORDER, 1);
060.                    }
061. //+------------------------------------------------------------------+
062. inline void CreteBarSlider(int x, int size)
063.                    {
064.                            ObjectCreate(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJ_RECTANGLE_LABEL, 0, 0, 0);
065.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_XDISTANCE, def_PosXObjects + x);
066.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_YDISTANCE, m_Slider.posY - 4);
067.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_XSIZE, size);
068.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_YSIZE, 9);
069.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BGCOLOR, clrLightSkyBlue);
070.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BORDER_COLOR, clrBlack);
071.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_WIDTH, 3);
072.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BORDER_TYPE, BORDER_FLAT);
073.                            ObjectCreate(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJ_RECTANGLE_LABEL, 0, 0, 0);
074.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XDISTANCE, def_PosXObjects + x);
075.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_YDISTANCE, m_Slider.posY - 9);
076.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_YSIZE, 19);
077.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_BGCOLOR, clrRosyBrown);
078.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_BORDER_TYPE, BORDER_RAISED);
079.                    }
080. //+------------------------------------------------------------------+
081.            void CreateBtnPlayPause(bool state)
082.                    {
083.                            m_szBtnPlay = def_PrefixObjectName + "Play";
084.                            CreateObjectBitMap(0, 25, m_szBtnPlay, (m_bWait ? def_ButtonWait : def_ButtonPause), (m_bWait ? def_ButtonWait : def_ButtonPlay));
085.                            ObjectSetInteger(GetInfoTerminal().ID, m_szBtnPlay, OBJPROP_STATE, state);
086.                    }
087. //+------------------------------------------------------------------+
088.            void CreteCtrlSlider(void)
089.                    {
090.                            u_Interprocess Info;
091.
092.                            m_Slider.szBarSlider       = def_NameObjectsSlider + " Bar";
093.                            m_Slider.szBarSliderBlock  = def_NameObjectsSlider + " Bar Block";
094.                            m_Slider.szBtnLeft         = def_NameObjectsSlider + " BtnL";
095.                            m_Slider.szBtnRight        = def_NameObjectsSlider + " BtnR";
096.                            m_Slider.szBtnPin          = def_NameObjectsSlider + " BtnP";
097.                            m_Slider.posY = 40;
098.                            CreteBarSlider(77, 436);
099.                            CreateObjectBitMap(47, 25, m_Slider.szBtnLeft, def_ButtonLeft, def_ButtonLeftBlock);
100.                            CreateObjectBitMap(511, 25, m_Slider.szBtnRight, def_ButtonRight, def_ButtonRightBlock);
101.                            CreateObjectBitMap(0, m_Slider.posY, m_Slider.szBtnPin, def_ButtonPin);
102.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBtnPin, OBJPROP_ANCHOR, ANCHOR_CENTER);
103.                            if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.df_Value = 0;
104.                            m_Slider.Minimal = Info.s_Infos.iPosShift;
105.                            PositionPinSlider(Info.s_Infos.iPosShift);
106.                    }
107. //+------------------------------------------------------------------+
108. inline void RemoveCtrlSlider(void)
109.                    {
110.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
111.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_NameObjectsSlider);
112.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
113.                    }
114. //+------------------------------------------------------------------+
115. inline void PositionPinSlider(int p, const int minimal = 0)
116.                    {
117.                            m_Slider.posPinSlider = (p < minimal ? minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
118.                            m_Slider.posPinSlider = (p < m_Slider.Minimal ? m_Slider.Minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
119.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBtnPin, OBJPROP_XDISTANCE, m_Slider.posPinSlider + def_PosXObjects + 95);
120.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != minimal);
121.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != m_Slider.Minimal);
122.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBtnRight, OBJPROP_STATE, m_Slider.posPinSlider < def_MaxPosSlider);
123.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, minimal + 2);
124.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, m_Slider.Minimal + 2);
125.                            ChartRedraw(GetInfoTerminal().ID);
126.                    }
127. //+------------------------------------------------------------------+
128.    public  :
129. //+------------------------------------------------------------------+
130.            C_Controls(const long Arg0, const string szShortName, C_Mouse *MousePtr)
131.                    :C_Terminal(Arg0),
132.                     m_bWait(false),
133.                     m_MousePtr(MousePtr)
134.                    {
135.                            if (!IndicatorCheckPass(szShortName)) SetUserError(C_Terminal::ERR_Unknown);
136.                            m_szBtnPlay          = NULL;
137.                            m_Slider.szBarSlider = NULL;
138.                            m_Slider.szBtnPin    = NULL;
139.                            m_Slider.szBtnLeft   = NULL;
140.                            m_Slider.szBtnRight  = NULL;
141.                    }
142. //+------------------------------------------------------------------+
143.            ~C_Controls()
144.                    {
145.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
146.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_PrefixObjectName);
147.                    }
148. //+------------------------------------------------------------------+
149.            void Init(const bool state)
150.                    {
151.                            CreateBtnPlayPause(state);
152.                            GlobalVariableTemp(def_GlobalVariableReplay);
153.                            if (!state) CreteCtrlSlider();
154.                            ChartRedraw(GetInfoTerminal().ID);
155.                    }
156. //+------------------------------------------------------------------+
157.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
158.                    {
159.                            u_Interprocess Info;
160.
161.                            switch (id)
162.                            {
163.                                    case (CHARTEVENT_CUSTOM + C_Controls::ev_WaitOn):
164.                                            if (lparam == 0) break;
165.                                            m_bWait = true;
166.                                            CreateBtnPlayPause(true);
167.                                            break;
168.                                    case (CHARTEVENT_CUSTOM + C_Controls::ev_WaitOff):
169.                                            if (lparam == 0) break;
170.                                            m_bWait = false;
171.                                            Info.df_Value = dparam;
172.                                            CreateBtnPlayPause(Info.s_Infos.isPlay);
173.                                            break;
174.                                    case CHARTEVENT_OBJECT_DELETE:
175.                                            if (StringSubstr(sparam, 0, StringLen(def_PrefixObjectName)) == def_PrefixObjectName)
176.                                            {
177.                                                    if (StringSubstr(sparam, 0, StringLen(def_NameObjectsSlider)) == def_NameObjectsSlider)
178.                                                    {
179.                                                            RemoveCtrlSlider();
180.                                                            CreteCtrlSlider();
181.                                                    }else
182.                                                    {
183.                                                            Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
184.                                                            CreateBtnPlayPause(Info.s_Infos.isPlay);
185.                                                    }
186.                                                    ChartRedraw(GetInfoTerminal().ID);
187.                                            }
188.                                            break;
189.                            }
190.                    }
191. //+------------------------------------------------------------------+
192. };
193. //+------------------------------------------------------------------+
194. #undef def_PosXObjects
195. #undef def_ButtonPlay
196. #undef def_ButtonPause
197. #undef def_ButtonLeft
198. #undef def_ButtonRight
199. #undef def_ButtonPin
200. #undef def_NameObjectsSlider
201. #undef def_PrefixObjectName
202. #undef def_PathBMP
203. //+------------------------------------------------------------------+
```

Source code of the C\_Control class

You may not notice any significant changes in the code. In fact, they are quite subtle. First, the control class is a private inheritor of the C\_Terminal class. This can be seen in line 31. This way we no longer need to use a pointer to access the C\_Terminal class. In other words, the bulk access to the C\_Terminal class can be executed directly, and in some cases a little faster, due to certain compilation details, but this is no longer important.

In line 130, where the class constructor is declared, we have something very interesting. Please note that at this early stage, only the service will have access to this control indicator. Look at line 135. Actually, this is not really necessary, but since code reuse is always appropriate, we ensure that the indicator follows the assumptions we want. In other words, there should be only one single indicator per chart. Again, this is not necessary as at this early stage only the service has access to this indicator and can add it to the chart. There are other ways to do this, but I won't go into detail because I don't want to encourage anyone to use the resource without understanding how it works.

Another thing worth mentioning in this constructor is that in line 133 we store in a private global class variable the pointer to access the mouse indicator. But this will become clear later.

If you pay attention, in this code you can see a fix that has been made, it's a very subtle change but it makes a big difference. Not for the code, but for MetaTrader 5. The point is that ChartRedraw calls receive a value. We usually don't pass any values to this call. So why are we doing this now? The reason is that the Chart ID value is different.

I explained this difference in the previous article. However, this may not be clear enough, so let's consolidate the knowledge gained. The main problem with the Chart ID is not who will open the chart, but who will place objects on it. Let's take a moment to understand why sometimes we need to pass a value into the ChartRedraw call and other times we don't.

When the service opens a chart (although it can be anything), this chart gets an ID from MetaTrader 5. If you check this identifier, you will see some value when parsing the value returned by ChartOpen. Fine. Now, to place, for example, an indicator on this chart, the program must use the same identifier that will be returned by the ChartOpen function. So far everything is clear, but it is at this point that a problem arises.

When you as a user place the same indicator on the chart, you will not get the same Chart ID that the program got using ChartOpen. Now everything seems confusing. It may even seem like I'm crazy or don't understand what I'm talking about. When the service opens the chart, we will get the ID value via ChartOpen, and we should use this value to create the indicator handle to place the indicator on the desired chart. If the indicator, as is the case with the control indicator, uses functions to place objects on the chart using an ObjectCreate call, we will need to specify an identifier so that MetaTrader 5 knows which chart is the correct one.

The ID provided must NOT be the ID obtained via ChartID if the chart was opened via ChartOpen. If we use the Chart ID specified in ChartID, then with the ObjectCreate call on a chart opened with ChartOpen, the object will not be displayed on the chart. But if the same code is placed on the chart by the user or a template using the ObjectCreate function, then the value provided by ChartID must be used as the ID.

I know it seems very confusing, but that's exactly what it takes. For this reason, a few articles ago, the C\_Terminal class was updated specifically to handle this issue.

But since our code doesn't know who exactly executed it on the chart, we use a call to the C\_Terminal class to return the correct ID. This way the ChartRedraw function will update the chart correctly as expected.

Despite all these complexities, in line 157, where the message handler begins, you can see that this code is responsible for handling the events reported by MetaTrader 5. We removed events that parse mouse movement and click events. This is because we are going to handle this kind of event a little differently, although within the same function.

However, before we begin processing, we need to make some additional changes. But now it is not in this indicator code or in the service code. We'll have to go back to the mouse indicator code and make some changes to it.

### Updating the mouse indicator code

This update we are about to make does not conflict with what is already in use in the mouse indicator. We need to make this update for the very reason mentioned above. Once we run the replay/simulator service, it will launch a mouse indicator on the chart so that we can interact with the replay/simulator system. But you don't have to worry about some things, because the code itself will make the necessary changes so that the mouse indicator works the same as if you, the user, had placed it on the chart manually or using a template.

The big problem comes when we use the tools we have to properly initialize the replay/simulator system. So, let's look at the first thing that has been updated, which is the mouse indicator source code. Here is its full code.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "This is an indicator for graphical studies using the mouse."
04. #property description "This is an integral part of the Replay / Simulator system."
05. #property description "However it can be used in the real market."
06. #property version "1.51"
07. #property icon "/Images/Market Replay/Icons/Indicators.ico"
08. #property link "https://www.mql5.com/en/articles/11877"
09. #property indicator_chart_window
10. #property indicator_plots 0
11. #property indicator_buffers 1
12. //+------------------------------------------------------------------+
13. #include <Market Replay\Auxiliar\Study\C_Study.mqh>
14. //+------------------------------------------------------------------+
15. C_Study *Study      = NULL;
16. //+------------------------------------------------------------------+
17. input long  user00  = 0;                                    //ID
18. input C_Study::eStatusMarket user01 = C_Study::eAuction;    //Market Status
19. input color user02  = clrBlack;                             //Price Line
20. input color user03  = clrPaleGreen;                         //Positive Study
21. input color user04  = clrLightCoral;                        //Negative Study
22. //+------------------------------------------------------------------+
23. C_Study::eStatusMarket m_Status;
24. int m_posBuff = 0;
25. double m_Buff[];
26. //+------------------------------------------------------------------+
27. int OnInit()
28. {
29.     ResetLastError();
30.     Study = new C_Study(user00, "Indicator Mouse Study", user02, user03, user04);
31.     if (_LastError != ERR_SUCCESS) return INIT_FAILED;
32.     if ((*Study).GetInfoTerminal().szSymbol != def_SymbolReplay)
33.     {
34.             MarketBookAdd((*Study).GetInfoTerminal().szSymbol);
35.             OnBookEvent((*Study).GetInfoTerminal().szSymbol);
36.             m_Status = C_Study::eCloseMarket;
37.     }else
38.             m_Status = user01;
39.     SetIndexBuffer(0, m_Buff, INDICATOR_DATA);
40.     ArrayInitialize(m_Buff, EMPTY_VALUE);
41.
42.     return INIT_SUCCEEDED;
43. }
44. //+------------------------------------------------------------------+
45. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
46. {
47.     m_posBuff = rates_total - 4;
48.     (*Study).Update(m_Status);
49.
50.     return rates_total;
51. }
52. //+------------------------------------------------------------------+
53. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
54. {
55.     (*Study).DispatchMessage(id, lparam, dparam, sparam);
56.     SetBuffer();
57.
58.     ChartRedraw((*Study).GetInfoTerminal().ID);
59. }
60. //+------------------------------------------------------------------+
61. void OnBookEvent(const string &symbol)
62. {
63.     MqlBookInfo book[];
64.     C_Study::eStatusMarket loc = m_Status;
65.
66.     if (symbol != (*Study).GetInfoTerminal().szSymbol) return;
67.     MarketBookGet((*Study).GetInfoTerminal().szSymbol, book);
68.     m_Status = (ArraySize(book) == 0 ? C_Study::eCloseMarket : C_Study::eInTrading);
69.     for (int c0 = 0; (c0 < ArraySize(book)) && (m_Status != C_Study::eAuction); c0++)
70.             if ((book[c0].type == BOOK_TYPE_BUY_MARKET) || (book[c0].type == BOOK_TYPE_SELL_MARKET)) m_Status = C_Study::eAuction;
71.     if (loc != m_Status) (*Study).Update(m_Status);
72. }
73. //+------------------------------------------------------------------+
74. void OnDeinit(const int reason)
75. {
76.     if (reason != REASON_INITFAILED)
77.     {
78.             if ((*Study).GetInfoTerminal().szSymbol != def_SymbolReplay)
79.                     MarketBookRelease((*Study).GetInfoTerminal().szSymbol);
80.             delete Study;
81.     }
82. }
83. //+------------------------------------------------------------------+
84. inline void SetBuffer(void)
85. {
86.     uCast_Double Info;
87.
88.     m_posBuff = (m_posBuff < 0 ? 0 : m_posBuff);
89.     m_Buff[m_posBuff + 0] = (*Study).GetInfoMouse().Position.Price;
90.     Info._datetime = (*Study).GetInfoMouse().Position.dt;
91.     m_Buff[m_posBuff + 1] = Info.dValue;
92.     Info._int[0] = (*Study).GetInfoMouse().Position.X;
93.     Info._int[1] = (*Study).GetInfoMouse().Position.Y;
94.     m_Buff[m_posBuff + 2] = Info.dValue;
95.     Info._char[0] = ((*Study).GetInfoMouse().ExecStudy == C_Mouse::eStudyNull ? (char)(*Study).GetInfoMouse().ButtonStatus : 0);
96.     m_Buff[m_posBuff + 3] = Info.dValue;
97. }
98. //+------------------------------------------------------------------+
```

Source code of the mouse indicator

**Attention:** You might think there is no difference between this code and its previous version. But there are differences. You have to be extremely careful with them.

For the user, the main difference is in line 17. This is probably the most complex line in the entire code. Ideally, the user should NEVER be able to change this line or see it in the indicator interface. This is because any new user will be tempted to change this line as it is an input. However, please remember that you must NEVER change this value. This value must be set either by the program calling the indicator or by the indicator itself. But you, as a user, should NEVER change this value. The other values can be configured and changed by the user without any problems, but the ID value can never be changed.

Besides this, the content of line 30 is interesting. Here we initialize the study class pointer. Previously the constructor received 3 values which were mainly the colors used in the mouse indicator, but now the constructor will receive two additional values. The first value is the chart ID and the second is the indicator name. This name will be used for subsequent access to the indicator buffer. So whenever the mouse indicator is on the chart and we want to read its buffer, we use the name specified here.

There are a few more differences in the code, but since they are not difficult to understand, we will not discuss them. Then we can move on to the C\_Study class code. In this code, we won't use a pointer to access the C\_Terminal class, but we'll use inheritance instead.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\C_Mouse.mqh"
005. //+------------------------------------------------------------------+
006. #define def_ExpansionPrefix "MouseExpansion_"
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
017.                    eStatusMarket   Status;
018.                    MqlRates        Rate;
019.                    string          szInfo;
020.                    color           corP,
021.                                    corN;
022.                    int             HeightText;
023.            }m_Info;
024. //+------------------------------------------------------------------+
025.            const datetime GetBarTime(void)
026.                    {
027.                            datetime dt;
028.                            u_Interprocess info;
029.                            int i0 = PeriodSeconds();
030.
031.                            if (m_Info.Status == eInReplay)
032.                            {
033.                                    if (!GlobalVariableGet(def_GlobalVariableServerTime, info.df_Value)) return ULONG_MAX;
034.                                    if ((dt = info.ServerTime) == ULONG_MAX) return ULONG_MAX;
035.                            }else dt = TimeCurrent();
036.                            if (m_Info.Rate.time <= dt)
037.                                    m_Info.Rate.time = (datetime)(((ulong) dt / i0) * i0) + i0;
038.
039.                            return m_Info.Rate.time - dt;
040.                    }
041. //+------------------------------------------------------------------+
042.            void Draw(void)
043.                    {
044.                            double v1;
045.
046.                            ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn1, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y - 18);
047.                            ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y - 1);
048.                            ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y - 1);
049.                            ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn1, OBJPROP_TEXT, m_Info.szInfo);
050.                            v1 = NormalizeDouble(100.0 - ((m_Info.Rate.close / GetInfoMouse().Position.Price) * 100.0), 2);
051.                            ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
052.                            ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_TEXT, StringFormat("%.2f%%", MathAbs(v1)));
053.                            v1 = NormalizeDouble(100.0 - ((m_Info.Rate.close / iClose(GetInfoTerminal().szSymbol, PERIOD_D1, 0)) * 100.0), 2);
054.                            ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
055.                            ObjectSetString(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_TEXT, StringFormat("%.2f%%", MathAbs(v1)));
056.                    }
057. //+------------------------------------------------------------------+
058.    public  :
059. //+------------------------------------------------------------------+
060.            C_Study(long IdParam, string szShortName, color corH, color corP, color corN)
061.                    :C_Mouse(IdParam, szShortName, corH, corP, corN)
062.                    {
063.                            if (_LastError != ERR_SUCCESS) return;
064.                            ZeroMemory(m_Info);
065.                            m_Info.Status = eCloseMarket;
066.                            m_Info.Rate.close = iClose(GetInfoTerminal().szSymbol, PERIOD_D1, ((GetInfoTerminal().szSymbol == def_SymbolReplay) || (macroGetDate(TimeCurrent()) != macroGetDate(iTime(GetInfoTerminal().szSymbol, PERIOD_D1, 0))) ? 0 : 1));
067.                            m_Info.corP = corP;
068.                            m_Info.corN = corN;
069.                            CreateObjectInfo(2, 110, def_ExpansionBtn1, clrPaleTurquoise);
070.                            CreateObjectInfo(2, 53, def_ExpansionBtn2);
071.                            CreateObjectInfo(58, 53, def_ExpansionBtn3);
072.                    }
073. //+------------------------------------------------------------------+
074.            ~C_Study()
075.                    {
076.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_ExpansionPrefix);
077.                    }
078. //+------------------------------------------------------------------+
079.            void Update(const eStatusMarket arg)
080.                    {
081.                            datetime dt;
082.
083.                            switch (m_Info.Status = (m_Info.Status != arg ? arg : m_Info.Status))
084.                            {
085.                                    case eCloseMarket : m_Info.szInfo = "Closed Market";
086.                                            break;
087.                                    case eInReplay    :
088.                                    case eInTrading   :
089.                                            if ((dt = GetBarTime()) < ULONG_MAX)
090.                                            {
091.                                                    m_Info.szInfo = TimeToString(dt, TIME_SECONDS);
092.                                                    break;
093.                                            }
094.                                    case eAuction     : m_Info.szInfo = "Auction";
095.                                            break;
096.                                    default           : m_Info.szInfo = "ERROR";
097.                            }
098.                            Draw();
099.                    }
100. //+------------------------------------------------------------------+
101. virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
102.                    {
103.                            C_Mouse::DispatchMessage(id, lparam, dparam, sparam);
104.                            if (id == CHARTEVENT_MOUSE_MOVE) Draw();
105.                    }
106. //+------------------------------------------------------------------+
107. };
108. //+------------------------------------------------------------------+
109. #undef def_ExpansionBtn3
110. #undef def_ExpansionBtn2
111. #undef def_ExpansionBtn1
112. #undef def_ExpansionPrefix
113. //+------------------------------------------------------------------+
```

Source code of the C\_Study class

There is practically no difference between this code and the old one. The only difference is that we look for the graph identifier directly in the C\_Terminal class. Previously we used a pointer and now we use inheritance, but where does this inheritance come from? It comes from the C\_Mouse class. Let's look at the code of the C\_Mouse class to better understand this inheritance.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "C_Terminal.mqh"
005. #include "Interprocess.mqh"
006. //+------------------------------------------------------------------+
007. #define def_MousePrefixName "MouseBase_"
008. #define def_NameObjectLineH def_MousePrefixName + "H"
009. #define def_NameObjectLineV def_MousePrefixName + "TV"
010. #define def_NameObjectLineT def_MousePrefixName + "TT"
011. #define def_NameObjectStudy def_MousePrefixName + "TB"
012. //+------------------------------------------------------------------+
013. class C_Mouse : public C_Terminal
014. {
015.    public  :
016.            enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
017.            enum eBtnMouse {eKeyNull = 0x00, eClickLeft = 0x01, eClickRight = 0x02, eSHIFT_Press = 0x04, eCTRL_Press = 0x08, eClickMiddle = 0x10};
018.            struct st_Mouse
019.            {
020.                    struct st00
021.                    {
022.                            int      X,
023.                                     Y;
024.                            double   Price;
025.                            datetime dt;
026.                    }Position;
027.                    uint    ButtonStatus;
028.                    bool    ExecStudy;
029.            };
030. //+------------------------------------------------------------------+
031.    protected:
032.            enum eEventsMouse {ev_HideMouse, ev_ShowMouse};
033. //+------------------------------------------------------------------+
034.            void CreateObjectInfo(int x, int w, string szName, color backColor = clrNONE) const
035.                    {
036.                            if (m_Mem.szShortName != NULL) return;
037.                            CreateObjectGraphics(szName, OBJ_BUTTON, clrNONE);
038.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_STATE, true);
039.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BORDER_COLOR, clrBlack);
040.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_COLOR, clrBlack);
041.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BGCOLOR, backColor);
042.                            ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_FONT, "Lucida Console");
043.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_FONTSIZE, 10);
044.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
045.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XDISTANCE, x);
046.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YDISTANCE, TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT) + 1);
047.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XSIZE, w);
048.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YSIZE, 18);
049.                    }
050. //+------------------------------------------------------------------+
051.    private :
052.            enum eStudy {eStudyNull, eStudyCreate, eStudyExecute};
053.            struct st01
054.            {
055.                    st_Mouse Data;
056.                    color    corLineH,
057.                             corTrendP,
058.                             corTrendN;
059.                    eStudy   Study;
060.            }m_Info;
061.            struct st_Mem
062.            {
063.                    bool     CrossHair,
064.                             IsFull;
065.                    datetime dt;
066.                    string   szShortName;
067.            }m_Mem;
068. //+------------------------------------------------------------------+
069.            void GetDimensionText(const string szArg, int &w, int &h)
070.                    {
071.                            TextSetFont("Lucida Console", -100, FW_NORMAL);
072.                            TextGetSize(szArg, w, h);
073.                            h += 5;
074.                            w += 5;
075.                    }
076. //+------------------------------------------------------------------+
077.            void CreateStudy(void)
078.                    {
079.                            if (m_Mem.IsFull)
080.                            {
081.                                    CreateObjectGraphics(def_NameObjectLineV, OBJ_VLINE, m_Info.corLineH);
082.                                    CreateObjectGraphics(def_NameObjectLineT, OBJ_TREND, m_Info.corLineH);
083.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_WIDTH, 2);
084.                                    CreateObjectInfo(0, 0, def_NameObjectStudy);
085.                            }
086.                            m_Info.Study = eStudyCreate;
087.                    }
088. //+------------------------------------------------------------------+
089.            void ExecuteStudy(const double memPrice)
090.                    {
091.                            double v1 = GetInfoMouse().Position.Price - memPrice;
092.                            int w, h;
093.
094.                            if (!CheckClick(eClickLeft))
095.                            {
096.                                    m_Info.Study = eStudyNull;
097.                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
098.                                    if (m_Mem.IsFull) ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName + "T");
099.                            }else if (m_Mem.IsFull)
100.                            {
101.                                    string sz1 = StringFormat(" %." + (string)GetInfoTerminal().nDigits + "f [ %d ] %02.02f%% ",
102.                                            MathAbs(v1), Bars(GetInfoTerminal().szSymbol, PERIOD_CURRENT, m_Mem.dt, GetInfoMouse().Position.dt) - 1, MathAbs((v1 /memPrice) * 100.0)));
103.                                    GetDimensionText(sz1, w, h);
104.                                    ObjectSetString(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_TEXT, sz1);
105.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corTrendN : m_Info.corTrendP));
106.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_XSIZE, w);
107.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_YSIZE, h);
108.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_XDISTANCE, GetInfoMouse().Position.X - w);
109.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y - (v1 < 0 ? 1 : h));
110.                                    ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 1, GetInfoMouse().Position.dt, GetInfoMouse().Position.Price);
111.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_COLOR, (memPrice > GetInfoMouse().Position.Price ? m_Info.corTrendN : m_Info.corTrendP));
112.                            }
113.                            m_Info.Data.ButtonStatus = eKeyNull;
114.                    }
115. //+------------------------------------------------------------------+
116.    public  :
117. //+------------------------------------------------------------------+
118.            C_Mouse(const long id, const string szShortName)
119.                    :C_Terminal(id)
120.                    {
121.                            m_Mem.szShortName = szShortName;
122.                    }
123. //+------------------------------------------------------------------+
124.            C_Mouse(const long id, const string szShortName, color corH, color corP, color corN)
125.                    :C_Terminal(id)
126.                    {
127.                            if (!IndicatorCheckPass(szShortName)) SetUserError(C_Terminal::ERR_Unknown);
128.                            m_Mem.szShortName = NULL;
129.                            if (_LastError != ERR_SUCCESS) return;
130.                            m_Mem.CrossHair = (bool)ChartGetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL);
131.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, true);
132.                            ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, false);
133.                            ZeroMemory(m_Info);
134.                            m_Info.corLineH  = corH;
135.                            m_Info.corTrendP = corP;
136.                            m_Info.corTrendN = corN;
137.                            m_Info.Study = eStudyNull;
138.                            if (m_Mem.IsFull = (corP != clrNONE) && (corH != clrNONE) && (corN != clrNONE))
139.                                    CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
140.                    }
141. //+------------------------------------------------------------------+
142.            ~C_Mouse()
143.                    {
144.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, 0, false);
145.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, false);
146.                            ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, m_Mem.CrossHair);
147.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName);
148.                    }
149. //+------------------------------------------------------------------+
150. inline bool CheckClick(const eBtnMouse value)
151.                    {
152.                            return (GetInfoMouse().ButtonStatus & value) == value;
153.                    }
154. //+------------------------------------------------------------------+
155. inline const st_Mouse GetInfoMouse(void)
156.                    {
157.                            if (m_Mem.szShortName != NULL)
158.                            {
159.                                    double Buff[];
160.                                    uCast_Double loc;
161.                                    int handle = ChartIndicatorGet(GetInfoTerminal().ID, 0, m_Mem.szShortName);
162.
163.                                    ZeroMemory(m_Info.Data);
164.                                    if (CopyBuffer(handle, 0, 0, 4, Buff) == 4)
165.                                    {
166.                                            m_Info.Data.Position.Price = Buff[0];
167.                                            loc.dValue = Buff[1];
168.                                            m_Info.Data.Position.dt = loc._datetime;
169.                                            loc.dValue = Buff[2];
170.                                            m_Info.Data.Position.X = loc._int[0];
171.                                            m_Info.Data.Position.Y = loc._int[1];
172.                                            loc.dValue = Buff[3];
173.                                            m_Info.Data.ButtonStatus = loc._char[0];
174.                                            IndicatorRelease(handle);
175.                                    }
176.                            }
177.
178.                            return m_Info.Data;
179.                    }
180. //+------------------------------------------------------------------+
181. virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
182.                    {
183.                            int w = 0;
184.                            static double memPrice = 0;
185.
186.                            if (m_Mem.szShortName == NULL) switch (id)
187.                            {
188.                                    case (CHARTEVENT_CUSTOM + ev_HideMouse):
189.                                            if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
190.                                            break;
191.                                    case (CHARTEVENT_CUSTOM + ev_ShowMouse):
192.                                            if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, m_Info.corLineH);
193.                                            break;
194.                                    case CHARTEVENT_MOUSE_MOVE:
195.                                            ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X = (int)lparam, m_Info.Data.Position.Y = (int)dparam, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
196.                                            if (m_Mem.IsFull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price = AdjustPrice(m_Info.Data.Position.Price));
197.                                            m_Info.Data.Position.dt = AdjustTime(m_Info.Data.Position.dt);
198.                                            ChartTimePriceToXY(GetInfoTerminal().ID, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price, m_Info.Data.Position.X, m_Info.Data.Position.Y);
199.                                            if ((m_Info.Study != eStudyNull) && (m_Mem.IsFull)) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineV, 0, m_Info.Data.Position.dt, 0);
200.                                            m_Info.Data.ButtonStatus = (uint) sparam;
201.                                            if (CheckClick(eClickMiddle))
202.                                                    if ((!m_Mem.IsFull) || ((color)ObjectGetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR) != clrNONE)) CreateStudy();
203.                                            if (CheckClick(eClickLeft) && (m_Info.Study == eStudyCreate))
204.                                            {
205.                                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
206.                                                    if (m_Mem.IsFull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 0, m_Mem.dt = GetInfoMouse().Position.dt, memPrice = GetInfoMouse().Position.Price);
207.                                                    m_Info.Study = eStudyExecute;
208.                                            }
209.                                            if (m_Info.Study == eStudyExecute) ExecuteStudy(memPrice);
210.                                            m_Info.Data.ExecStudy = m_Info.Study == eStudyExecute;
211.                                            break;
212.                                    case CHARTEVENT_OBJECT_DELETE:
213.                                            if ((m_Mem.IsFull) && (sparam == def_NameObjectLineH)) CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
214.                                            break;
215.                            }
216.                    }
217. //+------------------------------------------------------------------+
218. };
219. //+------------------------------------------------------------------+
220. #undef def_MousePrefixName
221. #undef def_NameObjectLineV
222. #undef def_NameObjectLineH
223. #undef def_NameObjectLineT
224. #undef def_NameObjectStudy
225. //+------------------------------------------------------------------+
```

Source code of the C\_Mouse class

As before, the major difference between this code and the old one is that a pointer is no longer used to access the C\_Terminal class. The responsibility for this procedure can be seen in line 13, where we inherit from the C\_Terminal class to the C\_Mouse class. Thus, all procedures and public functions of class C\_Terminal now extend class C\_Mouse. All the code is mostly already explained in the articles related to the mouse indicator.

The last article on this topic was [Developing a Replay System (Part 31): Expert Advisor Project - C\_Mouse class (V)](https://www.mql5.com/en/articles/11378), you can use this article to jump to the first article about this mouse indicator, as there is a link at the beginning of the article, as in all my articles, so you can go back and follow the evolution of the system development.

If you have any doubts about how the mouse indicator works, or you want to adapt it to your personal needs, you can read the articles for the explanation of how the indicator was developed. Try to understand how it works, then come back to this place and add the modifications shown. You can also place other things which you need in the mouse indicator. If you do everything correctly, you will be able to use your own mouse indicator in this replay/simulator system.

Here's a tip for those who really want to learn programming: try adding new functionality to the mouse indicator and use it in the replay/simulator system. But don't forget to indicate the source of your knowledge, which will bring me great joy. Not only does it motivate me, but I also enjoy showing how I achieve solutions to problems that many people consider insurmountable.

But let's get back to our explanation of the code. In the constructor of the C\_Mouse class, you can see that on line 125 we initialize the C\_Terminal class. The value to be used during initialization is provided in the indicator code, in the input named user00. If the user leaves this input untouched, the C\_Terminal class will determine the chart ID and, if necessary, will switch to using this ID to place objects on the desired chart.

Additionally, line 127 locks the indicator so that it cannot be placed on the chart more than once. This will prevent the user from placing another mouse indicator on the chart. The indicator will be present in the list of MetaTrader 5 indicators.

The rest of the code works the same as before. This mouse indicator will be responsible for telling MetaTrader 5 to receive mouse events and send them to the chart where the indicator is located. Please note that in order to receive mouse events in MetaTrader 5, the mouse indicator must be on the chart; trying to access the indicator from another chart does not make sense. Even if we have the ability to read what a mouse indicator does on another chart, MetaTrader 5 will not generate mouse events for a chart that does not contain such an indicator.

Line 131 is responsible for this. Thus, any Expert Advisor or other indicator will receive mouse events as long as the indicator is on the chart. Remember this fact, because the following codes will require the mouse indicator to be present on the chart so that the mouse events will be passed on by MetaTrader 5 to other codes that will deal with them in a specific way.

Before I move on to the next phase, where we will modify the control indicator, I think it is necessary to give a more precise explanation of what is actually happening. This way you can correctly understand the entire content of this article.

### Understanding and assimilating the knowledge

You may think that everything I am talking about in this article is nonsense, because many of you must have a lot of experience with MQL5 or know someone who is an expert in the coding system. But to make everything clear, I will attach the executable file of the mouse indicator to this article. This is done for simplicity, however you can use the materials in this series and you will get the same result as I place the full code inside the articles. You can try it out and see what happens using the following code, which is much simpler than the replay/simulator system, but works on the same principles.

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property copyright "Daniel Jose"
04. #property version   "1.00"
05. //+------------------------------------------------------------------+
06. input string user00 = "EURUSD"; // Symbol
07. //+------------------------------------------------------------------+
08. void OnStart()
09. {
10.    long id;
11.    int handle;
12.
13.    SymbolSelect(user00, true);
14.    id = ChartOpen(user00, PERIOD_M5);
15.    handle = iCustom(ChartSymbol(id), ChartPeriod(id), "\\Indicators\\Replay\\Mouse Study.ex5", id);
16.    ChartIndicatorAdd(id, 0, handle);
17.    IndicatorRelease(handle);
18.
19.    Print("ID: ", id);
20.
21.    while ((!_StopFlag) && (ChartSymbol(id) == user00)) Sleep(300);
22.
23.    ChartClose(id);
24.    SymbolSelect(user00, false);
25. }
26. //+------------------------------------------------------------------+
```

Source code of the test service

In line 06 we give the user the option to specify the asset. It can be used to open a chart, but the symbol must be in the Market Watch window. For this we use line 13. In line 14 we ask MetaTrader 5 to open a chart for us. In line 15 we create a handle that will add the indicator to the chart; in this case, we will force MetaTrader 5 to load the mouse indicator. In line 16 we add the indicator to the chart. In line 17 we free the handle since we don't need it anymore. In line 19 we print a message to the terminal indicating the ID of the chart that was opened by the service. Now, in line 21, we wait for the user to close the chart or terminate the service.

If the user closes the chart, line 23 will do nothing. However, if the user terminates the service, line 23 will close the previously opened chart. Line 24 removes the symbol from the Market Watch window. Another point: for a symbol to be removed, it must not have any elements associated with it. In this case we use the chart just for testing, so there will be nothing associated with it.

You might think that this code doesn't make much sense. Why would bother creating something like this and trying to explain how it works? This is exactly what everyone should do: strive to explore beyond the boundaries that many are comfortable with. Only then will you truly understand how everything works.

I want you to understand the idea and essence, as it is of great importance for the following articles. You can watch video 01 and see the tests. You can also try to test it yourself using your own criteria. I'd like to emphasize: Don't accept someone's "truth" just because they seem better or more competent to you. Test, question authorities, and only then will true knowledge become clear.

Demonstração Parte 51 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11877)

MQL5.community

1.91K subscribers

[Demonstração Parte 51](https://www.youtube.com/watch?v=cGY3RHE-9UE)

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

[Watch on](https://www.youtube.com/watch?v=cGY3RHE-9UE&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11877)

0:00

0:00 / 3:13

•Live

•

Video 01

### Conclusion

This article was one of the most difficult ones to date. This is caused by the level of complexity of the information presented. Although the programming itself seems quite simple to me, explaining what is happening is very difficult. The reason is that it can be read not only by people with a much higher level of knowledge, but also by enthusiasts who are making their first attempts to understand this **PROGRAMMING** world.

In the next article, I will show how to create interactions between the mouse indicator, the control indicator, the replay/simulator system and you, dear users of the system. See you soon

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11877](https://www.mql5.com/pt/articles/11877)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11877.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11877/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/475829)**

![Requesting in Connexus (Part 6): Creating an HTTP Request and Response](https://c.mql5.com/2/100/http60x60__6.png)[Requesting in Connexus (Part 6): Creating an HTTP Request and Response](https://www.mql5.com/en/articles/16182)

In this sixth article of the Connexus library series, we will focus on a complete HTTP request, covering each component that makes up a request. We will create a class that represents the request as a whole, which will help us bring together the previously created classes.

![Self Optimizing Expert Advisor With MQL5 And Python (Part VI): Taking Advantage of Deep Double Descent](https://c.mql5.com/2/100/Self_Optimizing_Expert_Advisor_With_MQL5_And_Python_Part_VI__LOGO.png)[Self Optimizing Expert Advisor With MQL5 And Python (Part VI): Taking Advantage of Deep Double Descent](https://www.mql5.com/en/articles/15971)

Traditional machine learning teaches practitioners to be vigilant not to overfit their models. However, this ideology is being challenged by new insights published by diligent researches from Harvard, who have discovered that what appears to be overfitting may in some circumstances be the results of terminating your training procedures prematurely. We will demonstrate how we can use the ideas published in the research paper, to improve our use of AI in forecasting market returns.

![How to view deals directly on the chart without weltering in trading history](https://c.mql5.com/2/80/How_to_avoid_drowning_in_trading_history_and_easily_glide_right_along_the_chart____LOGO.png)[How to view deals directly on the chart without weltering in trading history](https://www.mql5.com/en/articles/15026)

In this article, we will create a simple tool for convenient viewing of positions and deals directly on the chart with key navigation. This will allow traders to visually examine individual deals and receive all the information about trading results right on the spot.

![Elements of correlation analysis in MQL5: Pearson chi-square test of independence and correlation ratio](https://c.mql5.com/2/80/Pearson_chi-square_independence_test_and_correlation_ratio____LOGO.png)[Elements of correlation analysis in MQL5: Pearson chi-square test of independence and correlation ratio](https://www.mql5.com/en/articles/15042)

The article observes classical tools of correlation analysis. An emphasis is made on brief theoretical background, as well as on the practical implementation of the Pearson chi-square test of independence and the correlation ratio.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11877&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070023689992343213)

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