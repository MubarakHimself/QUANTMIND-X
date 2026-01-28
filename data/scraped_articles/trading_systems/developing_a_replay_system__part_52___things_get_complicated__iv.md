---
title: Developing a Replay System (Part 52): Things Get Complicated (IV)
url: https://www.mql5.com/en/articles/11925
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:06:19.272144
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/11925&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070012342688747129)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 51): Things Get Complicated (III)](https://www.mql5.com/en/articles/11877), we made some changes to our old mouse pointer so that it would work properly in our Replay/Simulator system. In the same article, I demonstrated in practice the difference between getting a chart ID using ChartOpen, a function used by programs to tell MetaTrader 5 to open a chart, and getting the same ID of an already open chart, but using the ChartID function.

I think it was quite clear that the difference and long-term possibilities will allow you to use MQL5 programming much more widely.

Despite all the changes made to the mouse pointer and control indicator, there were some difficulties with integration, or more precisely, with getting the mouse pointer and control indicator to interact correctly. This was due to my mistake, because during the process of developing the mouse pointer I missed some details, due to which the interaction did not work properly.

In this article we will fix this: don't worry, it's a pretty simple but very necessary thing. After making this change, you will be able to use the same mouse pointer in your personal projects. This way you will have a personalized learning system and a secure way to interact with the graphics. When the mouse pointer is in study mode, it does not allow you to interact with other objects on the chart. But to do this, we will need to integrate it into our code locally.

To illustrate how this integration should be done, since we will be doing the same with other parts of the replay/simulator system, we will use a control indicator. However, this control indicator has been modified to allow the use of a more suitable system, as previously we were using the basic MQL5 system. We will dive into MQL5 a little deeper, and we will have some features that we could not get before.

The interaction we will achieve in this article will be basic, but it will be enough for you to understand how to integrate the mouse pointer into your programs and thus obtain a customized study system that at the same time will not interact unstably with objects placed on the chart. So let's start this article with the first topic, in which we will look at the changes that need to be made to the mouse pointer.

### Mouse pointer improvements

All the code (and I'm leaving it as is for now) will be posted in full in the article. You might think that it is split into too many parts. However, this way it's easier to explain all the details. So if the code references something that isn't here, you will have to look for references in previous articles.

This way I can have complete confidence that you are actually following the explanations and understanding how I develop the code. With this brief explanation in mind, let's take a look at the mouse pointer code. It is shown below:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. #property description "This is an indicator for graphical studies using the mouse."
004. #property description "This is an integral part of the Replay / Simulator system."
005. #property description "However it can be used in the real market."
006. #property version "1.52"
007. #property icon "/Images/Market Replay/Icons/Indicators.ico"
008. #property link "https://www.mql5.com/en/articles/11925"
009. #property indicator_chart_window
010. #property indicator_plots 0
011. #property indicator_buffers 1
012. //+------------------------------------------------------------------+
013. #include <Market Replay\Auxiliar\Study\C_Study.mqh>
014. //+------------------------------------------------------------------+
015. C_Study *Study     = NULL;
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
047.    m_posBuff = rates_total - 5;
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
100. }
101. //+------------------------------------------------------------------+
```

Mouse Pointer source code

Most likely, you will not notice any difference between this code and the previous ones. As I said, the changes are minor, but they are there.

The first of these, and perhaps the only one, is the index of the buffer into which the indicator will be written. In line 47 of the code, you can see that the offset value, which was previously four, is now five. What is the reason for this change? The increase by one unit is necessary to provide wider coverage, but for a better understanding, let's look at the function responsible for setting the indicator buffer on line 84.

You probably won't notice any differences between this feature and the ones you had in the past. The changes are subtle. You should understand them in order to use them in your personal projects where you may want to integrate this indicator into your specific model.

In particular, from line 92 onwards, things start to look different, but not that much. Why do I declare two X variables and two Y variables? And why is one called Adjusted and the other Graphics? This is the point. Previously the indicator returned only a graphical variable adjusted by time and price, but this is not suitable in all cases. There are situations when we actually need the value of a graphic coordinate, not a given value. To cover both situations, the indicator returns these two values so that we can use them in the best possible way.

Please note that the zero position of the buffer will contain the price value, and the first position will contain the time value. I decided to keep it for backwards compatibility with other things I already use personally. However, there is no need to read the buffer as it was created. Remember that at a higher level there is an easier way which is to use the Mouse class, but if you still want to read data from the buffer directly, you must understand how it was built. Therefore, understanding this function shown in line 84 is of utmost importance.

The changes don't end there. It is important to understand that making changes to the code may affect a large part of the rest of the code. However, in the class responsible for creating and conducting study, the changes were not as radical, but still deserve some explanation. The study code is shown below.

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
017.                    eStatusMarket Status;
018.                    MqlRates      Rate;
019.                    string        szInfo;
020.                    color         corP,
021.                                  corN;
022.                    int           HeightText;
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
046.                            ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn1, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 18);
047.                            ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn2, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 1);
048.                            ObjectSetInteger(GetInfoTerminal().ID, def_ExpansionBtn3, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - 1);
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
074.            void Update(const eStatusMarket arg)
075.                    {
076.                            datetime dt;
077.
078.                            switch (m_Info.Status = (m_Info.Status != arg ? arg : m_Info.Status))
079.                            {
080.                                    case eCloseMarket : m_Info.szInfo = "Closed Market";
081.                                            break;
082.                                    case eInReplay    :
083.                                    case eInTrading   :
084.                                            if ((dt = GetBarTime()) < ULONG_MAX)
085.                                            {
086.                                                    m_Info.szInfo = TimeToString(dt, TIME_SECONDS);
087.                                                    break;
088.                                            }
089.                                    case eAuction     : m_Info.szInfo = "Auction";
090.                                            break;
091.                                    default           : m_Info.szInfo = "ERROR";
092.                            }
093.                            Draw();
094.                    }
095. //+------------------------------------------------------------------+
096. virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
097.                    {
098.                            C_Mouse::DispatchMessage(id, lparam, dparam, sparam);
099.                            if (id == CHARTEVENT_MOUSE_MOVE) Draw();
100.                    }
101. //+------------------------------------------------------------------+
102. };
103. //+------------------------------------------------------------------+
104. #undef def_ExpansionBtn3
105. #undef def_ExpansionBtn2
106. #undef def_ExpansionBtn1
107. #undef def_ExpansionPrefix
108. #undef def_MousePrefixName
109. //+------------------------------------------------------------------+
```

Source code of the C\_Study class

The only real changes in this class were made to the Draw function, which is on line 42. These changes are aimed at maintaining the logic of time- and price-based research. However, objects whose coordinate system is graphical will cause this error. We have seen the explanation of this type of object when creating this mouse indicator a few articles ago. I explained why and how to work with price-time coordinates and XY-type graphic coordinates.

To maintain backward compatibility, we use adjusted graphic coordinates. This can be seen in lines 46-48. However, if we use other objects that require graphic coordinates and want them to follow the Price-Time coordinates, we will simply need to use the adjusted graphic coordinates as shown here. If you replace this adjusted system with a graphic one that is not adjusted, the study may look somewhat different.

Perhaps you should try this for yourself.

To finish the mouse indicator part, let's look at the class responsible for supporting the base system. This will allow us to better understand the difference between adjusted and unadjusted graphic coordinates. We will then move on to the control indicator code to understand the reason for all these changes. The full code for the mouse class can be seen below. I hope that we will not need to make any further amendments to it.

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
022.                            int      X_Adjusted,
023.                                     Y_Adjusted,
024.                                     X_Graphics,
025.                                     Y_Graphics;
026.                            double   Price;
027.                            datetime dt;
028.                    }Position;
029.                    uint    ButtonStatus;
030.                    bool    ExecStudy;
031.            };
032. //+------------------------------------------------------------------+
033.    protected:
034.            enum eEventsMouse {ev_HideMouse, ev_ShowMouse};
035. //+------------------------------------------------------------------+
036.            void CreateObjectInfo(int x, int w, string szName, color backColor = clrNONE) const
037.                    {
038.                            if (m_Mem.szShortName != NULL) return;
039.                            CreateObjectGraphics(szName, OBJ_BUTTON, clrNONE);
040.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_STATE, true);
041.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BORDER_COLOR, clrBlack);
042.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_COLOR, clrBlack);
043.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BGCOLOR, backColor);
044.                            ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_FONT, "Lucida Console");
045.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_FONTSIZE, 10);
046.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
047.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XDISTANCE, x);
048.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YDISTANCE, TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT) + 1);
049.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_XSIZE, w);
050.                            ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_YSIZE, 18);
051.                    }
052. //+------------------------------------------------------------------+
053.    private :
054.            enum eStudy {eStudyNull, eStudyCreate, eStudyExecute};
055.            struct st01
056.            {
057.                    st_Mouse Data;
058.                    color    corLineH,
059.                             corTrendP,
060.                             corTrendN;
061.                    eStudy   Study;
062.            }m_Info;
063.            struct st_Mem
064.            {
065.                    bool     CrossHair,
066.                             IsFull;
067.                    datetime dt;
068.                    string   szShortName;
069.            }m_Mem;
070.            bool m_OK;
071. //+------------------------------------------------------------------+
072.            void GetDimensionText(const string szArg, int &w, int &h)
073.                    {
074.                            TextSetFont("Lucida Console", -100, FW_NORMAL);
075.                            TextGetSize(szArg, w, h);
076.                            h += 5;
077.                            w += 5;
078.                    }
079. //+------------------------------------------------------------------+
080.            void CreateStudy(void)
081.                    {
082.                            if (m_Mem.IsFull)
083.                            {
084.                                    CreateObjectGraphics(def_NameObjectLineV, OBJ_VLINE, m_Info.corLineH);
085.                                    CreateObjectGraphics(def_NameObjectLineT, OBJ_TREND, m_Info.corLineH);
086.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_WIDTH, 2);
087.                                    CreateObjectInfo(0, 0, def_NameObjectStudy);
088.                            }
089.                            m_Info.Study = eStudyCreate;
090.                    }
091. //+------------------------------------------------------------------+
092.            void ExecuteStudy(const double memPrice)
093.                    {
094.                            double v1 = GetInfoMouse().Position.Price - memPrice;
095.                            int w, h;
096.
097.                            if (!CheckClick(eClickLeft))
098.                            {
099.                                    m_Info.Study = eStudyNull;
100.                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
101.                                    if (m_Mem.IsFull) ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName + "T");
102.                            }else if (m_Mem.IsFull)
103.                            {
104.                                    string sz1 = StringFormat(" %." + (string)GetInfoTerminal().nDigits + "f [ %d ] %02.02f%% ",
105.                                            MathAbs(v1), Bars(GetInfoTerminal().szSymbol, PERIOD_CURRENT, m_Mem.dt, GetInfoMouse().Position.dt) - 1, MathAbs((v1 / memPrice) * 100.0)));
106.                                    GetDimensionText(sz1, w, h);
107.                                    ObjectSetString(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_TEXT, sz1);
108.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corTrendN : m_Info.corTrendP));
109.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_XSIZE, w);
110.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_YSIZE, h);
111.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_XDISTANCE, GetInfoMouse().Position.X_Adjusted - w);
112.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectStudy, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y_Adjusted - (v1 < 0 ? 1 : h));
113.                                    ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 1, GetInfoMouse().Position.dt, GetInfoMouse().Position.Price);
114.                                    ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_COLOR, (memPrice > GetInfoMouse().Position.Price ? m_Info.corTrendN : m_Info.corTrendP));
115.                            }
116.                            m_Info.Data.ButtonStatus = eKeyNull;
117.                    }
118. //+------------------------------------------------------------------+
119.    public  :
120. //+------------------------------------------------------------------+
121.            C_Mouse(const long id, const string szShortName)
122.                    :C_Terminal(id),
123.                    m_OK(false)
124.                    {
125.                            m_Mem.szShortName = szShortName;
126.                    }
127. //+------------------------------------------------------------------+
128.            C_Mouse(const long id, const string szShortName, color corH, color corP, color corN)
129.                    :C_Terminal(id)
130.                    {
131.                            if (!(m_OK = IndicatorCheckPass(szShortName))) SetUserError(C_Terminal::ERR_Unknown);
132.                            if (_LastError != ERR_SUCCESS) return;
133.                            m_Mem.szShortName = NULL;
134.                            m_Mem.CrossHair = (bool)ChartGetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL);
135.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, true);
136.                            ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, false);
137.                            ZeroMemory(m_Info);
138.                            m_Info.corLineH  = corH;
139.                            m_Info.corTrendP = corP;
140.                            m_Info.corTrendN = corN;
141.                            m_Info.Study = eStudyNull;
142.                            if (m_Mem.IsFull = (corP != clrNONE) && (corH != clrNONE) && (corN != clrNONE))
143.                                    CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
144.                    }
145. //+------------------------------------------------------------------+
146.            ~C_Mouse()
147.                    {
148.                            if (!m_OK) return;
149.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, 0, false);
150.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, false);
151.                            ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, m_Mem.CrossHair);
152.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName);
153.                    }
154. //+------------------------------------------------------------------+
155. inline bool CheckClick(const eBtnMouse value)
156.                    {
157.                            return (GetInfoMouse().ButtonStatus & value) == value;
158.                    }
159. //+------------------------------------------------------------------+
160. inline const st_Mouse GetInfoMouse(void)
161.                    {
162.                            if (m_Mem.szShortName != NULL)
163.                            {
164.                                    double Buff[];
165.                                    uCast_Double loc;
166.                                    int handle = ChartIndicatorGet(GetInfoTerminal().ID, 0, m_Mem.szShortName);
167.
168.                                    ZeroMemory(m_Info.Data);
169.                                    if (CopyBuffer(handle, 0, 0, 5, Buff) == 5)
170.                                    {
171.                                            m_Info.Data.Position.Price = Buff[0];
172.                                            loc.dValue = Buff[1];
173.                                            m_Info.Data.Position.dt = loc._datetime;
174.                                            loc.dValue = Buff[2];
175.                                            m_Info.Data.Position.X_Adjusted = loc._int[0];
176.                                            m_Info.Data.Position.Y_Adjusted = loc._int[1];
177.                                            loc.dValue = Buff[3];
178.                                            m_Info.Data.Position.X_Graphics = loc._int[0];
179.                                            m_Info.Data.Position.Y_Graphics = loc._int[1];
180.                                            loc.dValue = Buff[4];
181.                                            m_Info.Data.ButtonStatus = loc._char[0];
182.                                            IndicatorRelease(handle);
183.                                    }
184.                            }
185.
186.                            return m_Info.Data;
187.                    }
188. //+------------------------------------------------------------------+
189. virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
190.                    {
191.                            int w = 0;
192.                            static double memPrice = 0;
193.
194.                            if (m_Mem.szShortName == NULL) switch (id)
195.                            {
196.                                    case (CHARTEVENT_CUSTOM + ev_HideMouse):
197.                                            if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
198.                                            break;
199.                                    case (CHARTEVENT_CUSTOM + ev_ShowMouse):
200.                                            if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, m_Info.corLineH);
201.                                            break;
202.                                    case CHARTEVENT_MOUSE_MOVE:
203.                                            ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X_Graphics = (int)lparam, m_Info.Data.Position.Y_Graphics = (int)dparam, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
204.                                            if (m_Mem.IsFull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price = AdjustPrice(m_Info.Data.Position.Price));
205.                                            m_Info.Data.Position.dt = AdjustTime(m_Info.Data.Position.dt);
206.                                            ChartTimePriceToXY(GetInfoTerminal().ID, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price, m_Info.Data.Position.X_Adjusted, m_Info.Data.Position.Y_Adjusted);
207.                                            if ((m_Info.Study != eStudyNull) && (m_Mem.IsFull)) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineV, 0, m_Info.Data.Position.dt, 0);
208.                                            m_Info.Data.ButtonStatus = (uint) sparam;
209.                                            if (CheckClick(eClickMiddle))
210.                                                    if ((!m_Mem.IsFull) || ((color)ObjectGetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR) != clrNONE)) CreateStudy();
211.                                            if (CheckClick(eClickLeft) && (m_Info.Study == eStudyCreate))
212.                                            {
213.                                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
214.                                                    if (m_Mem.IsFull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 0, m_Mem.dt = GetInfoMouse().Position.dt, memPrice = GetInfoMouse().Position.Price);
215.                                                    m_Info.Study = eStudyExecute;
216.                                            }
217.                                            if (m_Info.Study == eStudyExecute) ExecuteStudy(memPrice);
218.                                            m_Info.Data.ExecStudy = m_Info.Study == eStudyExecute;
219.                                            break;
220.                                    case CHARTEVENT_OBJECT_DELETE:
221.                                            if ((m_Mem.IsFull) && (sparam == def_NameObjectLineH)) CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
222.                                            break;
223.                            }
224.                    }
225. //+------------------------------------------------------------------+
226. };
227. //+------------------------------------------------------------------+
228. #undef def_NameObjectLineV
229. #undef def_NameObjectLineH
230. #undef def_NameObjectLineT
231. #undef def_NameObjectStudy
232. //+------------------------------------------------------------------+
```

Source code of the C\_Mouse class

You may notice that the situation here seems tense, but don't let yourself be fooled by first impressions. Let's move on to the initial details. The includes present in lines 4 and 5 can be obtained from previous articles. They haven't changed, so there's no point in repeating them here.

If we scroll down a little, in lines 22 through 25 we will find the variable declarations that we have seen in the previous codes. They are part of a structure that is public. Thus, we can use the same structure in the future. This makes the code more readable and of higher quality. These same variables are used again in lines 110 and 111 just to ensure that some objects are positioned correctly.

But it's line 157 where things get really interesting. As we have seen before, you don't need to know how the buffer was mounted to be able to use the mouse class to get the values we need. It is through the use of this function in line 157 that this becomes possible. This function is able to adapt and respond correctly to a buffer read request, returning information through the structure mentioned above. This makes programming much easier.

But note that just as the buffer was created, here we have to work in reverse to get the information that is present in the buffer. Then, if everything goes correctly, the GetInfoMouse function will return data related to mouse positioning and clicks. This data can be used in any program you develop or in the mouse indicator itself. The interface is the same, which makes it much easier to maintain and understand the code.

After this mouse GetInfoMouse function, we have another one that is perhaps the most important of all, as it will respond to interactions that occur with the mouse. This function, DispatchMessage, is on line 186, but it's essentially the same as before, just with a few improvements that do exactly what we need.

There are a few things to understand: MetaTrader 5 uses a message system very similar to Windows, and for this reason messages are passed to our program in much the same way as if it were running on Windows. Knowing Windows programming is a great help in these cases, but even if the information is reviewed as needed, it often ends up in the wrong form or in a format that is foreign to those not familiar with programming.

So when MetaTrader 5 sends a mouse event to our program, it compiles the mouse data into the message itself. This can be seen in line 200, where MetaTrader 5 checks exactly what Windows tells it. Thus, MetaTrader 5 does not report the mouse coordinates based on price and time, but does so in graphical form. At this stage we have the necessary graphic coordinates, which are not adjusted and represent the position of the mouse in the window. Not on the chart, but in the window. However, there is something that needs to be clarified here. The word "window" here refers to graphic windows. To understand this, you need to know Windows programming, which is not the purpose of this article.

Once we have this position, we can use MQL5 to convert it into price and time positions. For this reason, objects that use these coordinates for positioning may lie slightly away from the point, having no direct connection with the bars present on the chart. To fix this, we use line 201, which adjusts the price, and line 202, which adjusts the time. This establishes a correspondence between the position and the chart bar.

But since at some points we want the X and Y values to be adjusted for price and time, we use line 203. This way we get adjusted values. Previously, no distinction was made between the graphically adjusted and unadjusted value. However, when trying to manipulate objects and integrate the mouse indicator with the control indicator, it was necessary to make this distinction. This closes the mouse indicator and we can switch our attention to the control indicator.

### New control indicator

As we mentioned at the beginning of the article, we are going to use MQL5 more intensively in order to push some improvements to the control indicator. These improvements will help us make life a little more enjoyable. In the application, we will have access to new shapes that will allow us to achieve transparency. To make it clear how to achieve this, we will use new raster images.

You may be wondering: will I have to create a new class for this? Don't worry, I promise I'll show you the main interactions by the end of the article.

The source code for the class is as follows.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #include "C_Terminal.mqh"
05. //+------------------------------------------------------------------+
06. #define def_MaxImages 2
07. //+------------------------------------------------------------------+
08. class C_DrawImage : protected C_Terminal
09. {
10. //+------------------------------------------------------------------+
11.     private :
12.             struct st_00
13.             {
14.                     int  widthMap,
15.                          heightMap;
16.                     uint Map[];
17.             }m_InfoImage[def_MaxImages];
18.             uint    m_Pixels[];
19.             string  m_szObjName,
20.                     m_szRecName;
21. //+------------------------------------------------------------------+
22.             void ReSizeImage(const int w, const int h, const uchar v, const int what)
23.                     {
24. #define _Transparency(A) (((A > 100 ? 100 : (100 - A)) * 2.55) / 255.0)
25.                             double fx = (w * 1.0) / m_InfoImage[what].widthMap;
26.                             double fy = (h * 1.0) / m_InfoImage[what].heightMap;
27.                             uint pyi, pyf, pxi, pxf, tmp;
28.                             uint uc;
29.
30.                             ArrayResize(m_Pixels, w * h);
31.                             for (int cy = 0, y = 0; cy < m_InfoImage[what].heightMap; cy++, y += m_InfoImage[what].widthMap)
32.                             {
33.                                     pyf = (uint)(fy * cy) * w;
34.                                     tmp = pyi = (uint)(fy * (cy - 1)) * w;
35.                                     for (int x = 0; x < m_InfoImage[what].widthMap; x++)
36.                                     {
37.                                             pxf = (uint)(fx * x);
38.                                             pxi = (uint)(fx * (x - 1));
39.                                             uc = (uchar(double((uc = m_InfoImage[what].Map[x + y]) >> 24) * _Transparency(v)) << 24) | uc & 0x00FFFFFF;
40.                                             m_Pixels[pxf + pyf] = uc;
41.                                             for (pxi++; pxi < pxf; pxi++) m_Pixels[pxi + pyf] = uc;
42.                                     }
43.                                     for (pyi += w; pyi < pyf; pyi += w)
44.                                             for (int x = 0; x < w; x++)
45.                                                     m_Pixels[x + pyi] = m_Pixels[x + tmp];
46.                             }
47. #undef _Transparency
48.                     }
49. //+------------------------------------------------------------------+
50.     public  :
51. //+------------------------------------------------------------------+
52.             C_DrawImage(long id, int sub, string szObjName, const color cFilter, const string szFile1, const string szFile2 = NULL)
53.                     :C_Terminal(id),
54.                     m_szObjName(NULL),
55.                     m_szRecName(NULL)
56.                     {
57.                             if (!ObjectCreate(GetInfoTerminal().ID, m_szObjName = szObjName, OBJ_BITMAP_LABEL, sub, 0, 0)) SetUserError(C_Terminal::ERR_Unknown);
58.                             m_szRecName = "::" + m_szObjName;
59.                             for (int c0 = 0; (c0 < def_MaxImages) && (_LastError == ERR_SUCCESS); c0++)
60.                             {
61.                                     ResourceReadImage((c0 == 0 ? szFile1 : (szFile2 == NULL ? szFile1 : szFile2)), m_InfoImage[c0].Map, m_InfoImage[c0].widthMap, m_InfoImage[c0].heightMap);
62.                                     ArrayResize(m_Pixels, m_InfoImage[c0].heightMap * m_InfoImage[c0].widthMap);
63.                                     ArrayInitialize(m_Pixels, 0);
64.                                     for (int c1 = (m_InfoImage[c0].heightMap * m_InfoImage[c0].widthMap) - 1; c1 >= 0; c1--)
65.                                             if ((m_InfoImage[c0].Map[c1] & 0x00FFFFFF) != cFilter) m_Pixels[c1] = m_InfoImage[c0].Map[c1];
66.                                     ArraySwap(m_InfoImage[c0].Map, m_Pixels);
67.                             }
68.                             ArrayResize(m_Pixels, 1);
69.                     }
70. //+------------------------------------------------------------------+
71.             ~C_DrawImage()
72.                     {
73.                             for (int c0 = 0; c0 < def_MaxImages; c0++)
74.                                     ArrayFree(m_InfoImage[c0].Map);
75.                             ArrayFree(m_Pixels);
76.                             ObjectDelete(GetInfoTerminal().ID, m_szObjName);
77.                             ResourceFree(m_szRecName);
78.                     }
79. //+------------------------------------------------------------------+
80.             void Paint(const int x, const int y, const int w, const int h, const uchar cView, const int what)
81.                     {
82.
83.                             if ((m_szRecName == NULL) || (what < 0) || (what >= def_MaxImages)) return;
84.                             ReSizeImage(w, h, cView, what);
85.                             ObjectSetInteger(GetInfoTerminal().ID, m_szObjName, OBJPROP_XDISTANCE, x);
86.                             ObjectSetInteger(GetInfoTerminal().ID, m_szObjName, OBJPROP_YDISTANCE, y);
87.                             if (ResourceCreate(m_szRecName, m_Pixels, w, h, 0, 0, 0, COLOR_FORMAT_ARGB_NORMALIZE))
88.                             {
89.                                     ObjectSetString(GetInfoTerminal().ID, m_szObjName, OBJPROP_BMPFILE, m_szRecName);
90.                                     ChartRedraw(GetInfoTerminal().ID);
91.                             }
92.                     }
93. //+------------------------------------------------------------------+
94. };
95. //+------------------------------------------------------------------+
96. #undef def_MaxImages
97. //+------------------------------------------------------------------+
```

Source code of the C\_DrawImage class

Note that the code is very compact. This is because we will not be using external resources. We have at our disposal MQL5 and everything it can offer to help us. Essentially, this class is an alternative way to use OBJ\_BITMAP\_LABEL or OBJ\_BITMAP objects. Both types are part of the MQL5 standard library, but they do not have the ability to use transparency or invisible dots on the image. So, the above class can be extended to incorporate the things we need.

This class is so simple that we only use four straightforward procedures. Let me briefly explain how everything works here.

In line 6 we specify the maximum number of images we will use. It is possible to increase this number, but some code modification will be required. Soon I'll show you where you need to make changes. Then we'll move into the class code in line 8. Note that it inherits the terminal class. This was explained in detail in previous articles.

Immediately after this, in line 11, we use the private part of the code to indicate that the declared data will be private to the inner code of the class. Actually we don't need to change this part where such data is declared. However, in line 22 we enter the pre-rendering procedure where we set the image size and transparency level. This level varies from 0% to 100% and changes in integer steps, i.e. one step at a time. The definition of line 24 ensures that the image transparency transformation is correct.

Here we increase or decrease the image resolution. Since these are small images, I didn't see the need to add anti-aliasing. Therefore, if you zoom in too much, the image may appear very pixelated. If you want to enlarge the image for some reason, you should consider adding anti-aliasing, but the current system is quite sufficient for our purposes.

Line 52 contains the class constructor. At this point, you will have to increase the number of parameters if you want to work with more images, but again, the idea is to extend the capabilities of MQL5. Therefore, two images will be enough.

In this constructor, in line 61, we will access the images that we will reference as resources. Remember that our system does not need external resources, so only executable files can be transported. Line 63 ensures that the image is fully transparent. Also, in line 64, we introduce a loop that will iterate over the loaded image, and on each interaction in line 65 we will check if the specified color is found as transparent. If true, the color will not be added, if false, it will be added.

In line 66, we use the MQL5 function to change the original image to the modified one, so we do everything as quickly as possible. From line 68 we are clearing the system.

The destructor at the beginning of line 71 is used to return all resources (in this case, memory) to the system, freeing them up for use by other programs. No special details or explanations are required.

Line 80 contains the drawing procedure. This procedure will cause the image to be reproduced on the chart at the specified position and dimensions using the first four parameters as guidelines. The fifth call parameter must be a value between 0 and 100, where 0 is no transparency and 100 is full transparency. The sixth parameter specifies which image will actually be rendered. The value should always start with 0 - this is the index of the first image, and the maximum value is the highest number minus 1. That is, we use the same system as for the OBJ\_BITMAP\_LABEL and OBJ\_BITMAP objects, in which we specify the image index, only here there can be more than 2 if we change the points I specified. This drawing function does not require any explanation or additional changes if we want to expand the capabilities.

The only places that will need to be changed are the constructor and the definition on line 6. Now, if we want to load an image directly from a file, we will need to implement that as well. For what we want to do, this class is already perfect.

Now we can move on to the code for the indicator and control class. But before we look at the control class code, let's take a quick look at the indicator code. You can see it in full below.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property description "Control indicator for the Replay-Simulator service."
05. #property description "This one doesn't work without the service loaded."
06. #property version   "1.52"
07. #property link "https://www.mql5.com/en/articles/11925"
08. #property indicator_chart_window
09. #property indicator_plots 0
10. //+------------------------------------------------------------------+
11. #include <Market Replay\Service Graphics\C_Controls.mqh>
12. //+------------------------------------------------------------------+
13. C_Controls *control = NULL;
14. //+------------------------------------------------------------------+
15. input long user00 = 0;    //ID
16. //+------------------------------------------------------------------+
17. int OnInit()
18. {
19.     u_Interprocess Info;
20.
21.     ResetLastError();
22.     if (CheckPointer(control = new C_Controls(user00, "Market Replay Control", new C_Mouse(user00, "Indicator Mouse Study"))) == POINTER_INVALID)
23.             SetUserError(C_Terminal::ERR_PointerInvalid);
24.     if (_LastError != ERR_SUCCESS)
25.     {
26.             Print("Control indicator failed on initialization.");
27.             return INIT_FAILED;
28.     }
29.     if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.df_Value = 0;
30.     EventChartCustom(user00, C_Controls::evInit, Info.s_Infos.iPosShift, Info.df_Value, "");
31.     GlobalVariableTemp(def_GlobalVariableReplay);
32.
33.     return INIT_SUCCEEDED;
34. }
35. //+------------------------------------------------------------------+
36. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
37. {
38.     return rates_total;
39. }
40. //+------------------------------------------------------------------+
41. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
42. {
43.     (*control).DispatchMessage(id, lparam, dparam, sparam);
44. }
45. //+------------------------------------------------------------------+
46. void OnDeinit(const int reason)
47. {
48.     switch (reason)
49.     {
50.             case REASON_TEMPLATE:
51.                     Print("Modified template. Replay/simulation system shutting down.");
52.             case REASON_PARAMETERS:
53.             case REASON_REMOVE:
54.             case REASON_CHARTCLOSE:
55.                     if (ChartSymbol(user00) != def_SymbolReplay) break;
56.                     GlobalVariableDel(def_GlobalVariableReplay);
57.                     ChartClose(user00);
58.                     break;
59.     }
60.     delete control;
61. }
62. //+------------------------------------------------------------------+
```

Source code of the control indicator

You can see that this code has lost some weight (volume). But this is temporary as we need lighter code to test it properly. You can see that we have much fewer calls and references to the control class. Why? The reason is the change in how the process is organized. We should have as few access points as possible. Note that the OnInit code is very different. Next, there is something very unusual in line 30. In this step, we initialize the control class data after the constructor. However, as for the constructor mentioned in line 22, we decided that it would only deal with initializing the pointers used. Event handler will take care of all other tasks. For this reason we have line 30. To make things clearer, let's take a look at the basic code of the control class.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\Auxiliar\Interprocess.mqh"
005. #include "..\Auxiliar\C_DrawImage.mqh"
006. //+------------------------------------------------------------------+
007. #define def_PathBMP                "Images\\Market Replay\\Control\\"
008. #define def_ButtonPlay             def_PathBMP + "Play.bmp"
009. #define def_ButtonPause            def_PathBMP + "Pause.bmp"
010. #define def_ButtonLeft             def_PathBMP + "Left.bmp"
011. #define def_ButtonLeftBlock        def_PathBMP + "Left_Block.bmp"
012. #define def_ButtonRight            def_PathBMP + "Right.bmp"
013. #define def_ButtonRightBlock       def_PathBMP + "Right_Block.bmp"
014. #define def_ButtonPin              def_PathBMP + "Pin.bmp"
015. #resource "\\" + def_ButtonPlay
016. #resource "\\" + def_ButtonPause
017. #resource "\\" + def_ButtonLeft
018. #resource "\\" + def_ButtonLeftBlock
019. #resource "\\" + def_ButtonRight
020. #resource "\\" + def_ButtonRightBlock
021. #resource "\\" + def_ButtonPin
022. //+------------------------------------------------------------------+
023. #define def_PrefixCtrlName         "MarketReplayCTRL_"
024. #define def_PosXObjects            120
025. //+------------------------------------------------------------------+
026. #define def_SizeButtons            32
027. #define def_ColorFilter            0xFF00FF
028. //+------------------------------------------------------------------+
029. #include "..\Auxiliar\C_Terminal.mqh"
030. #include "..\Auxiliar\C_Mouse.mqh"
031. //+------------------------------------------------------------------+
032. class C_Controls : private C_Terminal
033. {
034.    protected:
035.            enum EventCustom {evInit};
036.    private :
037. //+------------------------------------------------------------------+
038.            enum eObjectControl {ePlay, eLeft, eRight, ePin, eNull};
039. //+------------------------------------------------------------------+
040.            struct st_00
041.            {
042.                    string  szBarSlider,
043.                            szBarSliderBlock;
044.                    int     Minimal;
045.            }m_Slider;
046.            struct st_01
047.            {
048.                    C_DrawImage *Btn;
049.                    bool         state;
050.                    int          x, y, w, h;
051.            }m_Section[eObjectControl::eNull];
052.            C_Mouse *m_MousePtr;
053. //+------------------------------------------------------------------+
054. inline void CreteBarSlider(int x, int size)
055.                    {
056.                            ObjectCreate(GetInfoTerminal().ID, m_Slider.szBarSlider = def_PrefixCtrlName + "B1", OBJ_RECTANGLE_LABEL, 0, 0, 0);
057.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_XDISTANCE, def_PosXObjects + x);
058.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_YDISTANCE, m_Section[ePin].y + 11);
059.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_XSIZE, size);
060.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_YSIZE, 9);
061.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BGCOLOR, clrLightSkyBlue);
062.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BORDER_COLOR, clrBlack);
063.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_WIDTH, 3);
064.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSlider, OBJPROP_BORDER_TYPE, BORDER_FLAT);
065.                            ObjectCreate(GetInfoTerminal().ID, m_Slider.szBarSliderBlock = def_PrefixCtrlName + "B2", OBJ_RECTANGLE_LABEL, 0, 0, 0);
066.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XDISTANCE, def_PosXObjects + x);
067.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_YDISTANCE, m_Section[ePin].y + 6);
068.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_YSIZE, 19);
069.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_BGCOLOR, clrRosyBrown);
070.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_BORDER_TYPE, BORDER_RAISED);
071.                    }
072. //+------------------------------------------------------------------+
073.            void SetPlay(bool state)
074.                    {
075.                            if (m_Section[ePlay].Btn == NULL)
076.                                    m_Section[ePlay].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_PrefixCtrlName + EnumToString(ePlay), def_ColorFilter, "::" + def_ButtonPlay, "::" + def_ButtonPause);
077.                            m_Section[ePlay].Btn.Paint(m_Section[ePlay].x, m_Section[ePlay].y, m_Section[ePlay].w, m_Section[ePlay].h, 20, ((m_Section[ePlay].state = state) ? 0 : 1));
078.                            ChartRedraw(GetInfoTerminal().ID);
079.                    }
080. //+------------------------------------------------------------------+
081.            void CreateCtrlSlider(void)
082.                    {
083.                            CreteBarSlider(77, 436);
084.                            m_Section[eLeft].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_PrefixCtrlName + EnumToString(eLeft), def_ColorFilter, "::" + def_ButtonLeft, "::" + def_ButtonLeftBlock);
085.                            m_Section[eRight].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_PrefixCtrlName + EnumToString(eRight), def_ColorFilter, "::" + def_ButtonRight, "::" + def_ButtonRightBlock);
086.                            m_Section[ePin].Btn = new C_DrawImage(GetInfoTerminal().ID, 0, def_PrefixCtrlName + EnumToString(ePin), def_ColorFilter, "::" + def_ButtonPin);
087.                            PositionPinSlider(m_Slider.Minimal);
088.                    }
089. //+------------------------------------------------------------------+
090. inline void RemoveCtrlSlider(void)
091.                    {
092.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
093.                            for (eObjectControl c0 = ePlay + 1; c0 < eNull; c0++)
094.                            {
095.                                    delete m_Section[c0].Btn;
096.                                    m_Section[c0].Btn = NULL;
097.                            }
098.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_PrefixCtrlName + "B");
099.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
100.                    }
101. //+------------------------------------------------------------------+
102. inline void PositionPinSlider(int p)
103.                    {
104.                            int iL, iR;
105.
106.                            m_Section[ePin].x = (p < m_Slider.Minimal ? m_Slider.Minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
107.                            iL = (m_Section[ePin].x != m_Slider.Minimal ? 0 : 1);
108.                            iR = (m_Section[ePin].x < def_MaxPosSlider ? 0 : 1);
109.                            m_Section[ePin].x += def_PosXObjects;
110.                            m_Section[ePin].x += 95 - (m_Section[ePin].w / 2);
111.                            for (eObjectControl c0 = ePlay + 1; c0 < eNull; c0++)
112.                                    m_Section[c0].Btn.Paint(m_Section[c0].x, m_Section[c0].y, m_Section[c0].w, m_Section[c0].h, 20, (c0 == eLeft ? iL : (c0 == eRight ? iR : 0)));
113.                            ObjectSetInteger(GetInfoTerminal().ID, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, m_Slider.Minimal + 2);
114.                    }
115. //+------------------------------------------------------------------+
116. inline eObjectControl CheckPositionMouseClick(void)
117.                    {
118.                            C_Mouse::st_Mouse InfoMouse;
119.                            int x, y;
120.
121.                            InfoMouse = (*m_MousePtr).GetInfoMouse();
122.                            x = InfoMouse.Position.X_Graphics;
123.                            y = InfoMouse.Position.Y_Graphics;
124.                            for (eObjectControl c0 = ePlay; c0 < eNull; c0++)
125.                            {
126.                                    if ((m_Section[c0].x <= x) && (m_Section[c0].y <= y) && ((m_Section[c0].x + m_Section[c0].w) >= x) && ((m_Section[c0].y + m_Section[c0].h) >= y))
127.                                            return c0;
128.                            }
129.
130.                            return eNull;
131.                    }
132. //+------------------------------------------------------------------+
133.    public  :
134. //+------------------------------------------------------------------+
135.            C_Controls(const long Arg0, const string szShortName, C_Mouse *MousePtr)
136.                    :C_Terminal(Arg0),
137.                     m_MousePtr(MousePtr)
138.                    {
139.                            if (!IndicatorCheckPass(szShortName)) SetUserError(C_Terminal::ERR_Unknown);
140.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, false);
141.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_PrefixCtrlName);
142.                            ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, true);
143.                            for (eObjectControl c0 = ePlay; c0 < eNull; c0++)
144.                            {
145.                                    m_Section[c0].h = m_Section[c0].w = def_SizeButtons;
146.                                    m_Section[c0].y = 25;
147.                                    m_Section[c0].Btn = NULL;
148.                            }
149.                            m_Section[ePlay].x = def_PosXObjects;
150.                            m_Section[eLeft].x = m_Section[ePlay].x + 47;
151.                            m_Section[eRight].x = m_Section[ePlay].x + 511;
152.                    }
153. //+------------------------------------------------------------------+
154.            ~C_Controls()
155.                    {
156.                            delete m_MousePtr;
157.                            for (eObjectControl c0 = ePlay; c0 < eNull; c0++) delete m_Section[c0].Btn;
158.                            ObjectsDeleteAll(GetInfoTerminal().ID, def_PrefixCtrlName);
159.                    }
160. //+------------------------------------------------------------------+
161.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
162.                    {
163.                            u_Interprocess Info;
164.
165.                            switch (id)
166.                            {
167.                                    case CHARTEVENT_CUSTOM + C_Controls::evInit:
168.                                            Info.df_Value = dparam;
169.                                            m_Slider.Minimal = Info.s_Infos.iPosShift;
170.                                            SetPlay(Info.s_Infos.isPlay);
171.                                            if (!Info.s_Infos.isPlay) CreateCtrlSlider();
172.                                            break;
173.                                    case CHARTEVENT_OBJECT_DELETE:
174.                                            if (StringSubstr(sparam, 0, StringLen(def_PrefixCtrlName)) == def_PrefixCtrlName)
175.                                            {
176.                                                    if (sparam == (def_PrefixCtrlName + EnumToString(ePlay)))
177.                                                    {
178.                                                            delete m_Section[ePlay].Btn;
179.                                                            m_Section[ePlay].Btn = NULL;
180.                                                            SetPlay(false);
181.                                                    }else
182.                                                    {
183.                                                            RemoveCtrlSlider();
184.                                                            CreateCtrlSlider();
185.                                                    }
186.                                            }
187.                                            break;
188.                                    case CHARTEVENT_MOUSE_MOVE:
189.                                            if ((*m_MousePtr).CheckClick(C_Mouse::eClickLeft)) switch (CheckPositionMouseClick())
190.                                            {
191.                                                    case ePlay:
192.                                                            SetPlay(!m_Section[ePlay].state);
193.                                                            if (m_Section[ePlay].state) RemoveCtrlSlider();
194.                                                            else CreateCtrlSlider();
195.                                                            break;
196.                                                    case eLeft:
197.                                                            break;
198.                                                    case eRight:
199.                                                            break;
200.                                                    case ePin:
201.                                                            break;
202.                                            }
203.                                            break;
204.                            }
205.                            ChartRedraw(GetInfoTerminal().ID);
206.                    }
207. //+------------------------------------------------------------------+
208. };
209. //+------------------------------------------------------------------+
210. #undef def_PosXObjects
211. #undef def_ButtonPlay
212. #undef def_ButtonPause
213. #undef def_ButtonLeft
214. #undef def_ButtonRight
215. #undef def_ButtonPin
216. #undef def_PrefixCtrlName
217. #undef def_PathBMP
218. //+------------------------------------------------------------------+
```

Source code of the C\_Controls class

We say "basis code" because it doesn't actually do anything useful. It just allows us to demonstrate the use of the drawing class mentioned above and some very simple interaction with the mouse pointer. Let's see what we have. In line 26, we define the size of the interaction buttons. In line 27, we specify the color we will use to indicate that an area of the image is fully transparent. You can see this color in the attached images, which will be embedded into the executable file as resources.

In line 38, we specify the types of the main controls, and between in 46 to 51, we have a structure that will be located in the array that holds the controls. Pay attention to this, as the ability to understand this type of construction will be important for the next steps.

If you look at the control class code, you can see that we no longer use all the old calls. In fact, the code has undergone quite a lot of changes. In this article, I will not go into all the details, since it is not finished yet and allows us to have full control over the situation. Now, let's look at the SetPlay procedure, which is located in line 73. I'm not sure what the final name of the procedure will be, but that's what it is for now.

In this procedure, in line 75, we check if the pointer for the play and pause button has been created. If it has not been created, the drawing class constructor will be executed in line 76. This way we specify which images to use. Just like if we used the ObjectCreate function from the MQL5 library. Then, in line 77, we play the image at the position specified by the control button and tell it which image to track. Finally, in line 78, we ask the chart to be updated to draw the image.

I will not explain the other events for now, as we will look at them another time. But I want you to pay attention to line 188. In this line, we intercept the mouse events that MetaTrader 5 tracks. I'll briefly explain what's going on so you can understand the basics. Later, when the code becomes more complex, we'll explain in detail what's actually going on here.

When MetaTrader 5 sends us a mouse event, we ask the mouse indicator if a left click occurred. This will only be confirmed if the mouse indicator is not in study mode. If a click occurs, the control class will check which one was clicked. There was an error that we will fix later, but when it is confirmed that the play/pause button was pressed, the code in lines 191-195 will be executed, so we get an idea of the interaction that is happening between the user and the entire system.

You can see how this happens in the video below. Note that we don't have anything functional yet, but the idea was to try to set things up to provide just such an interaction between the mouse indicator and the control indicator.

### Conclusion

In this article, we looked at how to change the system to make it more enjoyable and useful. In the long run, this means that we will stop writing programs so often, and the system will become more and more stable and reliable. However, we also saw that in many cases we have to delve deeper into using MQL5 to get a better result.

I have not included executable files in the application because they are not working. In the attachment, you will find the images you need to replace the old ones with if you want to get the same system I am developing while keeping the same code shown in the articles. Feel free to change and correct what you need.

In the next article, we will take a closer look at this control indicator. Now everything will start to expand exponentially.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11925](https://www.mql5.com/pt/articles/11925)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11925.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11925/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/476816)**

![Visualizing deals on a chart (Part 2): Data graphical display](https://c.mql5.com/2/80/Visualization_of_trades_on_a_chart_Part_2_____LOGO.png)[Visualizing deals on a chart (Part 2): Data graphical display](https://www.mql5.com/en/articles/14961)

Here we are going to develop a script from scratch that simplifies unloading print screens of deals for analyzing trading entries. All the necessary information on a single deal is to be conveniently displayed on one chart with the ability to draw different timeframes.

![MQL5 Wizard Techniques you should know (Part 48): Bill Williams Alligator](https://c.mql5.com/2/101/MQL5_Wizard_Techniques_you_should_know_Part_48__LOGO.png)[MQL5 Wizard Techniques you should know (Part 48): Bill Williams Alligator](https://www.mql5.com/en/articles/16329)

The Alligator Indicator, which was the brain child of Bill Williams, is a versatile trend identification indicator that yields clear signals and is often combined with other indicators. The MQL5 wizard classes and assembly allow us to test a variety of signals on a pattern basis, and so we consider this indicator as well.

![Chemical reaction optimization (CRO) algorithm (Part I): Process chemistry in optimization](https://c.mql5.com/2/81/Algorithm_for_optimization_by_chemical_reactions__LOGO___2.png)[Chemical reaction optimization (CRO) algorithm (Part I): Process chemistry in optimization](https://www.mql5.com/en/articles/15041)

In the first part of this article, we will dive into the world of chemical reactions and discover a new approach to optimization! Chemical reaction optimization (CRO) uses principles derived from the laws of thermodynamics to achieve efficient results. We will reveal the secrets of decomposition, synthesis and other chemical processes that became the basis of this innovative method.

![Developing a multi-currency Expert Advisor (Part 13): Automating the second stage — selection into groups](https://c.mql5.com/2/80/Developing_a_multi-currency_advisor_Part_13__LOGO.png)[Developing a multi-currency Expert Advisor (Part 13): Automating the second stage — selection into groups](https://www.mql5.com/en/articles/14892)

We have already implemented the first stage of the automated optimization. We perform optimization for different symbols and timeframes according to several criteria and store information about the results of each pass in the database. Now we are going to select the best groups of parameter sets from those found at the first stage.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iuqfjrhayqoacdgatzyhjwtwqntjpghr&ssn=1769184377673517054&ssn_dr=0&ssn_sr=0&fv_date=1769184377&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11925&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2052)%3A%20Things%20Get%20Complicated%20(IV)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918437759881269&fz_uniq=5070012342688747129&sv=2552)

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