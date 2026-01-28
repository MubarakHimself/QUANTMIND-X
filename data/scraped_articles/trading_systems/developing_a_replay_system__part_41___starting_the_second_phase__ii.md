---
title: Developing a Replay System (Part 41): Starting the second phase (II)
url: https://www.mql5.com/en/articles/11607
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:13:29.143426
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/11607&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070112393951907879)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 40): Starting the second phase (I)](https://www.mql5.com/en/articles/11624), we started creating an indicator to support the study system and the mouse. But before we start, I would like to ask you a question: can you imagine why we are creating this indicator? What's behind our intention to demonstrate how to create something like this? What am I trying to do by creating an indicator for the mouse?

Well, perhaps these questions may not make much sense, but have you ever stopped at some points to observe your own code? Have you ever noticed how many times you repeat the same things in different codes? However, there is no consistency or stability in different codes. Please don't get me wrong. I should not be telling people with many years of programming experience what to do or what not to do. And I will not say that you are somehow incorrectly using the MQL5 language or any other language in which you specialize.

I just want to somehow get the reader, who has years of programming experience, to stop doing the same thing over and over again. Perhaps, it's time to look carefully at your own work. Stop doing the same thing over and over again due to lack of thoughtful planning.

In fact, there is a hidden reason behind the start of a new phase in the replay/simulator system. We have already talked about this in another article. I'm tired of doing everything the old way in MQL5. My personal codes follow certain rules. However, the codes attached to the articles were different. I always try to keep them as simple and clear as possible, but I constantly see people saying something about MQL5 or MetaTrader 5.

I want to show that it is possible to do much more than what others have achieved. So get ready, no more jokes. From now on we will see what my code actually looks like. I'm not going to demonstrate anything specific about certain assets or indicators, but I will show you that MQL5, like MetaTrader 5, is capable of much more than you think.

In the previous article, I mentioned that there was one unresolved issue to be resolved. So, let's start by solving this problem.

### Modification of the C\_Mouse class

The question here is how to interact with the mouse pointer. The code shown in the previous article is finalized. The indicator will not undergo any changes, at least for quite a long time. However, there is an issue that makes things much more complicated when we need to use our data.

For you to understand in depth what I am going to so, you will have to abandon the ideas and concepts that you knew about MetaTrader 5 and the MQL5 language. Forget everything you knew about working with them. Let's think in a totally different way. If you can understand this, we will be able to do a type of coding that will make your code more and more productive and stable.

In previous articles, we talked about the fact that processes running in MetaTrader 5 should be looked at not as indicators, Expert Advisors, scripts or services, but as functions. The idea of functions may not be the best term for this. However, there is one definition that fits best: DLLs. Yes, think about all the processes running in MetaTrader 5, especially indicators, as being DLLs.

Why am I saying this? Because the same concepts that are involved in DLLs can be applied to what I am going to show here. At the moment, the only indicator we have is the mouse. This way you will not have a complete and comprehensive understanding of what will happen. However, any great building always begins by laying the cornerstone. We will do it right now. But due to some problems, we will have to simplify the work entirely so as not to create complete chaos.

If you read the article [Developing a Replay System (Part 31): Expert Advisor project — C\_Mouse class (V)](https://www.mql5.com/en/articles/11378), you will notice that the C\_Mouse class, like the C\_Study class, works in a certain way. But the way they work **DOES NOT** allow us to use simple programming to the extent that we can take advantage of an indicator using these classes. **Attention**: I'm not saying we can't make the most of it, only that it would require extremely complex programming.

The idea is to simplify this program. To do this, we will remove some elements from the C\_Study class and move them to the C\_Mouse class. However, we are also going to make additions to the C\_Mouse class in terms of programming and structuring to make the indicator easier to use, as we will see in the next articles.

What we are going to remove in the C\_Study class can be seen in the following code.

**Part of C\_Study class code**:

```
01. #property copyright "Daniel Jose"
02. //+------------------------------------------------------------------+
03. #include "..\C_Mouse.mqh"
04. #include "..\..\Auxiliar\C_Mouse.mqh"
05. #include "..\..\Auxiliar\Interprocess.mqh"
06. //+------------------------------------------------------------------+
07. #define def_ExpansionPrefix "MouseExpansion_"
08. #define def_ExpansionBtn1 def_ExpansionPrefix + "B1"
09. #define def_ExpansionBtn2 def_ExpansionPrefix + "B2"
10. #define def_ExpansionBtn3 def_ExpansionPrefix + "B3"
11. //+------------------------------------------------------------------+
12. #define def_AcessTerminal (*Terminal)
13. #define def_InfoTerminal def_AcessTerminal.GetInfoTerminal()
14. //+------------------------------------------------------------------+
15. class C_Study : public C_Mouse
16. {
17.     protected:
18.     private :
19. //+------------------------------------------------------------------+
20.             enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
21. //+------------------------------------------------------------------+
22.             struct st00
23.             {
24.                     eStatusMarket   Status;
25.                     MqlRates        Rate;
26.                     string          szInfo;
27.                     color           corP,
28.                                     corN;
29.                     int             HeightText;
30.             }m_Info;
```

The location of the header file C\_Study.mqh has changed. We have replaced line 04 with line 03, and since we are no longer referencing the InterProcess.mqh header file, line 05 was also removed.

We also removed line 20 and included it in the C\_Mouse class. The purpose for this relocation is to make programming easier at a later phase. Now that we've seen the changes made to the C\_Study class, let's move on to the C\_Mouse class, where the changes are much more extensive.

Below you can see the entire code of the new C\_Mouse class.

**C\_Mouse.mqh file code:**

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
013. #define def_AcessTerminal (*Terminal)
014. #define def_InfoTerminal def_AcessTerminal.GetInfoTerminal()
015. //+------------------------------------------------------------------+
016. class C_Mouse
017. {
018.    public  :
019.            enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
020.            enum eBtnMouse {eKeyNull = 0x00, eClickLeft = 0x01, eClickRight = 0x02, eSHIFT_Press = 0x04, eCTRL_Press = 0x08, eClickMiddle = 0x10};
021.            struct st_Mouse
022.            {
023.                    struct st00
024.                    {
025.                            int       X,
026.                                      Y;
027.                            double    Price;
028.                            datetime  dt;
029.                    }Position;
030.                    uint    ButtonStatus;
031.                    bool    ExecStudy;
032.            };
033. //+------------------------------------------------------------------+
034.    protected:
035.            enum eEventsMouse {ev_HideMouse, ev_ShowMouse};
036. //+------------------------------------------------------------------+
037.            void CreateObjectInfo(int x, int w, string szName, color backColor = clrNONE) const
038.                    {
039.                            if (m_Mem.IsTranslator) return;
040.                            def_AcessTerminal.CreateObjectGraphics(szName, OBJ_BUTTON, clrNONE);
041.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_STATE, true);
042.                    	ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_BORDER_COLOR, clrBlack);
043.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_COLOR, clrBlack);
044.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_BGCOLOR, backColor);
045.                            ObjectSetString(def_InfoTerminal.ID, szName, OBJPROP_FONT, "Lucida Console");
046.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_FONTSIZE, 10);
047.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
048.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_XDISTANCE, x);
049.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_YDISTANCE, TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT) + 1);
050.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_XSIZE, w);
051.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_YSIZE, 18);
052.                    }
053. //+------------------------------------------------------------------+
054.    private :
055.            enum eStudy {eStudyNull, eStudyCreate, eStudyExecute};
056.            struct st01
057.            {
058.                    st_Mouse Data;
059.                    color    corLineH,
060.                             corTrendP,
061.                             corTrendN;
062.                    eStudy   Study;
063.            }m_Info;
064.            struct st_Mem
065.            {
066.                    bool     CrossHair,
067.                             IsFull,
068.                             IsTranslator;
069.                    datetime dt;
070.                    string   szShortName;
071.            }m_Mem;
072. //+------------------------------------------------------------------+
073.            C_Terminal *Terminal;
074. //+------------------------------------------------------------------+
075.            void GetDimensionText(const string szArg, int &w, int &h)
076.                    {
077.                            TextSetFont("Lucida Console", -100, FW_NORMAL);
078.                            TextGetSize(szArg, w, h);
079.                            h += 5;
080.                            w += 5;
081.                    }
082. //+------------------------------------------------------------------+
083.            void CreateStudy(void)
084.                    {
085.                            if (m_Mem.IsFull)
086.                            {
087.                                    def_AcessTerminal.CreateObjectGraphics(def_NameObjectLineV, OBJ_VLINE, m_Info.corLineH);
088.                                    def_AcessTerminal.CreateObjectGraphics(def_NameObjectLineT, OBJ_TREND, m_Info.corLineH);
089.                                    ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectLineT, OBJPROP_WIDTH, 2);
090.                                    CreateObjectInfo(0, 0, def_NameObjectStudy);
091.                            }
092.                            m_Info.Study = eStudyCreate;
093.                    }
094. //+------------------------------------------------------------------+
095.            void ExecuteStudy(const double memPrice)
096.                    {
097.                            double v1 = GetInfoMouse().Position.Price - memPrice;
098.                            int w, h;
099.
100.                            if (!CheckClick(eClickLeft))
101.                            {
102.                                    m_Info.Study = eStudyNull;
103.                                    ChartSetInteger(def_InfoTerminal.ID, CHART_MOUSE_SCROLL, true);
104.                                    if (m_Mem.IsFull)       ObjectsDeleteAll(def_InfoTerminal.ID, def_MousePrefixName + "T");
105.                            }else if (m_Mem.IsFull)
106.                            {
107.                                    string sz1 = StringFormat(" %." + (string)def_InfoTerminal.nDigits + "f [ %d ] %02.02f%% ",
108.                                            MathAbs(v1), Bars(def_InfoTerminal.szSymbol, PERIOD_CURRENT, m_Mem.dt, GetInfoMouse().Position.dt) - 1, MathAbs((v1 / memPrice) * 100.0)));
109.                                    GetDimensionText(sz1, w, h);
110.                                    ObjectSetString(def_InfoTerminal.ID, def_NameObjectStudy, OBJPROP_TEXT, sz1);
111.                                    ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectStudy, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corTrendN : m_Info.corTrendP));
112.                                    ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectStudy, OBJPROP_XSIZE, w);
113.                                    ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectStudy, OBJPROP_YSIZE, h);
114.                                    ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectStudy, OBJPROP_XDISTANCE, GetInfoMouse().Position.X - w);
115.                                    ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectStudy, OBJPROP_YDISTANCE, GetInfoMouse().Position.Y - (v1 < 0 ? 1 : h));
116.                                    ObjectMove(def_InfoTerminal.ID, def_NameObjectLineT, 1, GetInfoMouse().Position.dt, GetInfoMouse().Position.Price);
117.                                    ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectLineT, OBJPROP_COLOR, (memPrice > GetInfoMouse().Position.Price ? m_Info.corTrendN : m_Info.corTrendP));
118.                            }
119.                            m_Info.Data.ButtonStatus = eKeyNull;
120.                    }
121. //+------------------------------------------------------------------+
122.    public  :
123. //+------------------------------------------------------------------+
124.            C_Mouse(const string szShortName)
125.                    {
126.                            Terminal = NULL;
127.                            m_Mem.IsTranslator = true;
128.                            m_Mem.szShortName = szShortName;
129.                    }
130. //+------------------------------------------------------------------+
131.            C_Mouse(C_Terminal *arg, color corH = clrNONE, color corP = clrNONE, color corN = clrNONE)
132.                    {
133.                            m_Mem.IsTranslator = false;
134.                            Terminal = arg;
135.                            if (CheckPointer(Terminal) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
136.                            if (_LastError != ERR_SUCCESS) return;
137.                            m_Mem.CrossHair = (bool)ChartGetInteger(def_InfoTerminal.ID, CHART_CROSSHAIR_TOOL);
138.                            ChartSetInteger(def_InfoTerminal.ID, CHART_EVENT_MOUSE_MOVE, true);
139.                            ChartSetInteger(def_InfoTerminal.ID, CHART_CROSSHAIR_TOOL, false);
140.                            ZeroMemory(m_Info);
141.                            m_Info.corLineH  = corH;
142.                            m_Info.corTrendP = corP;
143.                            m_Info.corTrendN = corN;
144.                            m_Info.Study = eStudyNull;
145.                            if (m_Mem.IsFull = (corP != clrNONE) && (corH != clrNONE) && (corN != clrNONE))
146.                                    def_AcessTerminal.CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
147.                    }
148. //+------------------------------------------------------------------+
149.            ~C_Mouse()
150.                    {
151.                            if (CheckPointer(Terminal) == POINTER_INVALID) return;
152.                            ChartSetInteger(def_InfoTerminal.ID, CHART_EVENT_OBJECT_DELETE, 0, false);
153.                            ChartSetInteger(def_InfoTerminal.ID, CHART_EVENT_MOUSE_MOVE, false);
154.                            ChartSetInteger(def_InfoTerminal.ID, CHART_CROSSHAIR_TOOL, m_Mem.CrossHair);
155.                            ObjectsDeleteAll(def_InfoTerminal.ID, def_MousePrefixName);
156.                    }
157. //+------------------------------------------------------------------+
158. inline bool CheckClick(const eBtnMouse value)
159.                    {
160.                            return (GetInfoMouse().ButtonStatus & value) == value;
161.                    }
162. //+------------------------------------------------------------------+
163. inline const st_Mouse GetInfoMouse(void)
164.                    {
165.                            if (m_Mem.IsTranslator)
166.                            {
167.                                    double Buff[];
168.                                    uCast_Double loc;
169.                                    int handle = ChartIndicatorGet(ChartID(), 0, m_Mem.szShortName);
170.
171.                                    ZeroMemory(m_Info.Data);
172.                                    if (CopyBuffer(handle, 0, 0, 4, Buff) == 4)
173.                                    {
174.                                            m_Info.Data.Position.Price = Buff[0];
175.                                            loc.dValue = Buff[1];
176.                                            m_Info.Data.Position.dt = loc._datetime;
177.                                            loc.dValue = Buff[2];
178.                                            m_Info.Data.Position.X = loc._int[0];
179.                                            m_Info.Data.Position.Y = loc._int[1];
180.                                            loc.dValue = Buff[3];
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
194.                            if (!m_Mem.IsTranslator) switch (id)
195.                            {
196.                                    case (CHARTEVENT_CUSTOM + ev_HideMouse):
197.                                            if (m_Mem.IsFull)       ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
198.                                            break;
199.                                    case (CHARTEVENT_CUSTOM + ev_ShowMouse):
200.                                            if (m_Mem.IsFull) ObjectSetInteger(def_InfoTerminal.ID, def_NameObjectLineH, OBJPROP_COLOR, m_Info.corLineH);
201.                                            break;
202.                                    case CHARTEVENT_MOUSE_MOVE:
203.                                            ChartXYToTimePrice(def_InfoTerminal.ID, m_Info.Data.Position.X = (int)lparam, m_Info.Data.Position.Y = (int)dparam, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
204.                                            if (m_Mem.IsFull) ObjectMove(def_InfoTerminal.ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price = def_AcessTerminal.AdjustPrice(m_Info.Data.Position.Price));
205.                                            m_Info.Data.Position.dt = def_AcessTerminal.AdjustTime(m_Info.Data.Position.dt);
206.                                            ChartTimePriceToXY(def_InfoTerminal.ID, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price, m_Info.Data.Position.X, m_Info.Data.Position.Y);
207.                                            if ((m_Info.Study != eStudyNull) && (m_Mem.IsFull)) ObjectMove(def_InfoTerminal.ID, def_NameObjectLineV, 0, m_Info.Data.Position.dt, 0);
208.                                            m_Info.Data.ButtonStatus = (uint) sparam;
209.                                            if (CheckClick(eClickMiddle))
210.                                                    if ((!m_Mem.IsFull) || ((color)ObjectGetInteger(def_InfoTerminal.ID, def_NameObjectLineH, OBJPROP_COLOR) != clrNONE)) CreateStudy();
211.                                            if (CheckClick(eClickLeft) && (m_Info.Study == eStudyCreate))
212.                                            {
213.                                                    ChartSetInteger(def_InfoTerminal.ID, CHART_MOUSE_SCROLL, false);
214.                                                    if (m_Mem.IsFull)       ObjectMove(def_InfoTerminal.ID, def_NameObjectLineT, 0, m_Mem.dt = GetInfoMouse().Position.dt, memPrice = GetInfoMouse().Position.Price);
215.                                                    m_Info.Study = eStudyExecute;
216.                                            }
217.                                            if (m_Info.Study == eStudyExecute) ExecuteStudy(memPrice);
218.                                            m_Info.Data.ExecStudy = m_Info.Study == eStudyExecute;
219.                                            break;
220.                                    case CHARTEVENT_OBJECT_DELETE:
221.                                            if ((m_Mem.IsFull) && (sparam == def_NameObjectLineH)) def_AcessTerminal.CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
222.                                            break;
223.                            }
224.                    }
225. //+------------------------------------------------------------------+
226. };
227. //+------------------------------------------------------------------+
228. #undef def_AcessTerminal
229. #undef def_InfoTerminal
230. //+------------------------------------------------------------------+
231. #undef def_MousePrefixName
232. #undef def_NameObjectLineV
233. #undef def_NameObjectLineH
234. #undef def_NameObjectLineT
235. #undef def_NameObjectStudy
236. //+------------------------------------------------------------------+
237.
```

The reason why I provide here the full code is that there won't be any files in the application (at least for now). Therefore, anyone who wants to take advantage of the improvements can do so by using the given codes. When the system is at a more advanced stage, we will return the investment again. But at the moment they will not be available.

If you look at the code, you will see that there are highlighted points. I will not dwell on them in detail, although they have a certain significance. The reason is that most of them are self-explanatory.

The code removed from line 20 of the C\_Study class is now on line 19 of the C\_Mouse class. Now, look at line 18 of the C\_Mouse.mqh file. I am declaring here the public part. But why do we need it? The reason is that without it, any information will not have the level of access we need. We really need this data to be public. Normally we only use this part once, but because of my habit of running protected data first, private data second, and public data last, you will see this same part on line 122. But this is due to my programming style, which I am used to.

On line 20, we also have another enumeration that already existed in the C\_Mouse class but was not public, as well as a structure that starts on line 21 and goes to line 32. In this particular case, the structure is not used in any public global variable. Declaring public global variables in classes is not a good practice. Any global class variable must always be private or protected, and **never public**.

Starting from line 34, the data is no longer publicly available. Here we have the protected part, which ensures the protection of data, functions and procedures. We have already seen this in previous articles, but that is not the point. The problem is in the check on line 39. This check did not exist before. If the variable being checked has the value 'true', then it will not be possible to create objects that are created in the **CreateObjectInfo** method. Why do we do so? To understand this, we need to take a closer look at the code.

Next we find two lines in which two variables are declared: lines 68 and 70. Initially, they were not in the published code. But in this case, these variables are necessary. One of the reasons was shown in line 39: to allow or disable the execution of a method or function. The same applies to lines 165 and 194. In the case of line 165, there is a stronger motive, which we will look at later. However, in line 194 the reason is the same as in line 39: to avoid executing a function or method because it will not work or execute correctly when we are in translation mode.

As one might expect, these variables are initialized in the class constructor, that's a fact. But in the meantime, we no longer have a class constructor. Now we have two constructors for the C\_Mouse class 😱😵😲. If you're new to OOP programming, this will probably terrify you. However, this is completely normal for OOP. One of the constructors is written in code between lines 124 and 129. Another constructor is between lines 131 and 147. Although we have two constructors, we only have one destructor.

Since one of the constructors will serve for a certain purpose, and the second one will be used for a different one, we must somehow separate them. Private global variables are used for this. Without them, we can hardly make such a separation.

Now let's look at the reasons why we have two constructors. Firstly, there is no point in creating a second class just to satisfy the demand we need. Secondly, we can control the flow of execution in such a way that we can reuse existing code. All this can be done without having to create inheritance or manipulate data, functions and procedures.

If you look closely, you can see that all we have done so far is to isolate two codes: one public and the other protected. So if we use the C\_Mouse class to fulfill a certain purpose, we will have a usage model. However, when it comes to fulfilling other types of purpose, we will have a different model. But the way you program using a class will always be the same, whenever we use it in our functions.

Look at the constructor written between lines 131 and 147. You probably noticed that it is practically no different from the one presented in " [Developing a Replay System (Part 31): Expert Advisor project — C\_Mouse class (V)](https://www.mql5.com/en/articles/11378)". We did this on purpose, precisely because this constructor will be responsible for fulfilling the purposes of the class when the code uses the objects present in the class. The difference between this constructor and the original one is line 133, which initializes a new variable to indicate that we are going to use the class in its original form.

The fact that this happens means that most of the code remains the same, and any explanation provided in this particular article will still be relevant. Next we can focus on the second way to use the class. To do this, first look at the constructor code between lines 124 and 129. Despite its simplicity, it should be noted that we are initializing all the variables that are actually needed.

This makes it easier to use the class as a translator. Maybe you didn't understand how or why to do this. But we will use the C\_Mouse class as a mouse pointer translator. This probably seems quite complicated and difficult, doesn't it? But it's not difficult at all. There is a reason for all this.

To understand this, you need to think a little: when we create programs, be they indicators, Expert Advisors, scripts or anything else, we tend to add things that are repeated frequently. One of these things is mouse handling procedures. If in every EA we have to add functions to generate studies or mouse analysis, our code will never be truly reliable. However, when we create something that will last a long time, it can be improved independently of the rest of the code. Then we will have a new kind of program. Naturally, the code becomes much more reliable and efficient.

In this class, we will always assume that the mouse pointer will be present on the chart. Remember: we should never assume anything, but here we will assume that the indicator will be on the chart. Those who use the C\_Mouse class as a translator must know this fact. **The class will assume that the indicator is on the chart**.

If everything is clear with this fact, then you can move on to the next question. In fact, we do not need the C\_Mouse class to translate the indicator for us, because this can be done directly in the created program. However, it is much easier and more convenient to let the class do this translation for us. The reason is that if we don't want to use a translator, we just need to change the class constructor and add an event handling call to our program.

Then you will understand how much easier it will be for you to cope with your tasks. But I still want you to use the translation system. To understand how the translation system works, just look at the **GetInfoMouse** function on line 163. At first, it was a constant function, but is not any more. Although the data cannot be changed outside the function, we need it to change inside.

Let's get back to what we discussed in the previous article [Developing a Replay System (Part 40): Starting the second phase (I)](https://www.mql5.com/en/articles/11624). You will see that it is very difficult to interpret, or rather maintain the coding standard to interpret indicator buffer data. That's why I'll show you how to modify the C\_Mouse class to create this standardization. To imagine how difficult this is, think that every time we need to use the mouse pointer, we will have to write the same thing that we see in functions **CheckClick** and **GetInfoMouse**. This is quite a hassle.

Let's see what's going on here. Let's start with the **CheckClick** function. In this function, we simply load and check the value into the mouse data, either as a pointer or as a translator. We do this in any case, be it to analyze the indicator or to use the indicator itself. On line 160, we check whether the button being analyzed has been activated, i.e. whether it was pressed or not.

This is the question to understand. Regardless of whether we use the C\_Mouse class as a translator or as a pointer, we will always receive some kind of response from the **CheckClick** and **GetInfoMouse** functions. Always. This answer will always represent what the mouse is doing, no matter where we use the information.

To understand the answer given by **CheckClick**, we must follow and understand how the **GetInfoMouse** function works. This function can be seen starting from line 163 onwards.

On line 165 of the function, we check whether we are using the class as a translator or as a pointer. If the check passes, the class goes into the translator mode, and then we will need to access the pointer buffer. This question is both complex and simple. Let's look at the easy part first.

On line 171, we reset the return structure data. It is actually a private global class variable that is declared on line 58. After this we ask MetaTrader 5 to read the indicator buffer. If reading from line 172 is correct, we begin converting the data into the return structure.

Data conversion follows the same logic as data writing. The relevant code can be seen in the SetBuffer function form the previous article. To make things easier, I provide the same function below.

**Indicator code fragment**:

```
102. inline void SetBuffer(void)
103. {
104.    uCast_Double Info;
105.
106.    m_posBuff = (m_posBuff < 0 ? 0 : m_posBuff);
107.    m_Buff[m_posBuff + 0] = (*Study).GetInfoMouse().Position.Price;
108.    Info._datetime = (*Study).GetInfoMouse().Position.dt;
109.    m_Buff[m_posBuff + 1] = Info.dValue;
110.    Info._int[0] = (*Study).GetInfoMouse().Position.X;
111.    Info._int[1] = (*Study).GetInfoMouse().Position.Y;
112.    m_Buff[m_posBuff + 2] = Info.dValue;
113.    Info._char[0] = ((*Study).GetInfoMouse().ExecStudy == C_Mouse::eStudyNull ? (char)(*Study).GetInfoMouse().ButtonStatus : 0);
114.    m_Buff[m_posBuff + 3] = Info.dValue;
115. }
```

Pay attention that the information in line 107 of the indicator fragment is decoded in line 174 of the class. Most often the translation is very simple, but you must adhere to the rules made during coding. This can be better understood if you save the data in line 112 of the indicator fragment. Note that in this case we are working with two values compressed into a double. When it comes to translation, we have to do the opposite. This is done on line 177 of the class where we capture the value, while on lines 178 and 179 we put the values in the right place for later use.

The same thing happens in line 113 of the indicator fragment, where we store the mouse click value. Then we translate it in line 181 of the class. But let's now look again at line 113 of the indicator fragment. Note that the ternary operator will retain the value zero if we are in the study mode. It's important that you understand this. If we're doing the study via the indicator and the class is used to translate it, then when we check for a click through the CheckClick function it will return false. This will always happen, if, of course, we use an indicator and a class as a translator.

This was the simplest and most understandable part, but, as mentioned above, there is another part: complex and difficult.

It appears when we use a class as a translator. However, we do not have access to the indicator buffer. This will usually happen when the indicator is removed from the chart. When this happens, line 169 will generate a null handler and we will not have a buffer to read. We will still have to run line 171, which will restore the data.

This can lead to various violations and failures when trying to do something with the indicator data. Although the system always reports zero, we won't actually have any positive evidence of a click or movement, we still have problems with this. Not in this particular case, but in other cases that will also pose a problem for us. As soon as this happens, we will return to this issue.

### Using the C\_Mouse class as a translator

The below part is provided mainly for demonstration purposes as we will expand on this topic later.

Let's look at the Expert Advisor that will work the same way as in the article [Developing a Replay System (Part 31): Expert Advisor project — C\_Mouse class (V)](https://www.mql5.com/en/articles/11378/117130#!tab=article), but its code will be written in a different way.

**Expert Advisor Code**:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Generic EA for use on Demo account, replay system/simulator and Real account."
04. #property description "This system has means of sending orders using the mouse and keyboard combination."
05. #property description "For more information see the article about the system."
06. #property version   "1.41"
07. #property icon "/Images/Market Replay/Icons/Replay - EA.ico"
08. #property link "https://www.mql5.com/en/articles/11607"
09. //+------------------------------------------------------------------+
10. #include <Market Replay\Auxiliar\C_Mouse.mqh>
11. //+------------------------------------------------------------------+
12. C_Mouse *mouse = NULL;
13. //+------------------------------------------------------------------+
14. int OnInit()
15. {
16.     mouse = new C_Mouse("Indicator Mouse Study");
17.
18.     return INIT_SUCCEEDED;
19. }
20. //+------------------------------------------------------------------+
21. void OnTick()
22. { }
23. //+------------------------------------------------------------------+
24. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
25. {
26.     C_Mouse::st_Mouse Infos;
27.
28.     switch (id)
29.     {
30.             case CHARTEVENT_MOUSE_MOVE:
31.                     Infos = (*mouse).GetInfoMouse();
32.                     Comment((string)Infos.Position.Price + " :: [" + TimeToString(Infos.Position.dt), "]");
33.                     break;
34.     }
35. }
36. //+------------------------------------------------------------------+
37. void OnDeinit(const int reason)
38. {
39.     delete mouse;
40. }
41. //+------------------------------------------------------------------+
```

Although it doesn't look so, the code above has the same behavior as the code in the article mentioned, but is much simpler, more practical, and more reliable. The reason for this is not easy to understand now, but you will understand it gradually as we move forward in our coding journey.

You probably understand that in fact we are not doing anything super complex. We just assume here that everything indicated will be on the chart. On line 16, we indicate the short name of the indicator, and on line 39 we remove the C\_Mouse class.

Now I want you to look at lines 31 and 32, which will inform the mouse about the chart data so we can look at it.

The best part about this is that if we declare on line 16 that we are using the class as an indicator, we just need to add a call to the DispatchMessage function in the OnChatEvent function to get the same effect as using a graphical indicator. In other words, programming will not change. Instead, this will be just an adaptation to what we need.

If you want to use the function as a translator, the EA will analyze to such an extent that it will know where and what the mouse is doing, always reporting things appropriately.

### Conclusion

It is important that you understand how things work. Otherwise, you will be completely lost in the next articles, where we will no longer act as before. We will do everything in a much more complex form. Although what was shown in the last two articles seems complex, it is all intended for those who do not have much programming knowledge. Everything we've covered in the last two articles is just a little preparation for adding the Chart Trader app. In the upcoming articles, we will start adding and developing Chart Trader in our system. This app will allow us to trade directly on the market, which is very useful and important, especially since we cannot rely on the existing market order system. Therefore, we will have to create our own solution, that is, Chart Trader.

Despite this, we are still working at the basic level of MQL5. Even if you think this material is difficult, it is still at a very basic level. But feel bad about finding it too difficult, it's all natural. This may happen because you haven't studied MQL5 deeply. When you walk on any surface, you think that you are on a high hill, until someone shows you that all this time you have been walking at sea level. Believe me, soon you will think that everything you saw in the latest articles is children's games that any junior could do. So, get ready. Tough things are coming. When one of such moments comes, we'll come back to these mouse pointer issues.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11607](https://www.mql5.com/pt/articles/11607)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11607.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11607/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/469544)**

![Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part III)](https://c.mql5.com/2/83/Building_A_Candlestick_Trend_Constraint_Model__Part_5___CONT___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part III)](https://www.mql5.com/en/articles/14969)

This part of the article series is dedicated to integrating WhatsApp with MetaTrader 5 for notifications. We have included a flow chart to simplify understanding and will discuss the importance of security measures in integration. The primary purpose of indicators is to simplify analysis through automation, and they should include notification methods for alerting users when specific conditions are met. Discover more in this article.

![MQL5 Wizard Techniques you should know (Part 25): Multi-Timeframe Testing and Trading](https://c.mql5.com/2/82/MQL5_Wizard_Techniques_you_should_know_Part_25__LOGO.png)[MQL5 Wizard Techniques you should know (Part 25): Multi-Timeframe Testing and Trading](https://www.mql5.com/en/articles/15185)

Strategies that are based on multiple time frames cannot be tested in wizard assembled Expert Advisors by default because of the MQL5 code architecture used in the assembly classes. We explore a possible work around this limitation for strategies that look to use multiple time frames in a case study with the quadratic moving average.

![Neural networks made easy (Part 77): Cross-Covariance Transformer (XCiT)](https://c.mql5.com/2/70/Neural_networks_made_easy_pPart_77c__Cross-Covariance_Transformer_tXCiTl____LOGO.png)[Neural networks made easy (Part 77): Cross-Covariance Transformer (XCiT)](https://www.mql5.com/en/articles/14276)

In our models, we often use various attention algorithms. And, probably, most often we use Transformers. Their main disadvantage is the resource requirement. In this article, we will consider a new algorithm that can help reduce computing costs without losing quality.

![Data Science and Machine Learning (Part 25): Forex Timeseries Forecasting Using a Recurrent Neural Network (RNN)](https://c.mql5.com/2/82/Data_Science_and_ML_Part_25__LOGO.png)[Data Science and Machine Learning (Part 25): Forex Timeseries Forecasting Using a Recurrent Neural Network (RNN)](https://www.mql5.com/en/articles/15114)

Recurrent neural networks (RNNs) excel at leveraging past information to predict future events. Their remarkable predictive capabilities have been applied across various domains with great success. In this article, we will deploy RNN models to predict trends in the forex market, demonstrating their potential to enhance forecasting accuracy in forex trading.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/11607&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070112393951907879)

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