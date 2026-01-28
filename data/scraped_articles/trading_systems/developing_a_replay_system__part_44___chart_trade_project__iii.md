---
title: Developing a Replay System (Part 44): Chart Trade Project (III)
url: https://www.mql5.com/en/articles/11690
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:10:35.150734
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=lqdrbaoydmfsrzkvhirjibuuewcfxtwf&ssn=1769184633807749135&ssn_dr=0&ssn_sr=0&fv_date=1769184633&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11690&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2044)%3A%20Chart%20Trade%20Project%20(III)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918463332474095&fz_uniq=5070072184468082552&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 43): Chart Trade Project (II)](https://www.mql5.com/en/articles/11664), I explained how you can manipulate template data to use it in **OBJ\_CHART**. In that article, I only outlined the topic without going into details, since in that version the work was done in a very simplified way. This was done to make it easier to explain the content, because despite the apparent simplicity of many things, some of them were not so obvious, and without understanding the simplest and most basic part, you would not be able to truly understand the entire picture.

So, even though this code works (as we've already seen), it still doesn't allow us to do some things. In other words, performing certain tasks will be much more difficult unless some improvement is made in data modeling. The improvement mentioned involves a bit more complex coding, but the concept used will be the same. It's just that the code will be a little more complicated.

Besides this small fact, we will solve one more issue. If you noticed, and I also mentioned this in the article, this code is not very efficient as it contains, in my opinion, too many calls to set things up. To solve this problem, we will make some small changes to the code that will dramatically reduce the number of calls while simultaneously allowing for more adequate data modeling.

### Birth of a new class: C\_AdjustTemplate

To implement the necessary improvements, we will have to create a new class. Its full code is shown below:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #include "../Auxiliar/C_Terminal.mqh"
05. //+------------------------------------------------------------------+
06. class C_AdjustTemplate
07. {
08.     private :
09.             string m_szName[];
10.             string m_szFind[];
11.             string m_szReplace[];
12.             string m_szFileName;
13.             int m_maxIndex;
14.             int m_FileIn;
15.             int m_FileOut;
16.             bool m_bSimple;
17.     public  :
18. //+------------------------------------------------------------------+
19.             C_AdjustTemplate(const string szFileNameIn, string szFileNameOut = NULL)
20.                     :m_maxIndex(0),
21.                      m_szFileName(szFileNameIn),
22.                      m_FileIn(INVALID_HANDLE),
23.                      m_FileOut(INVALID_HANDLE)
24.                     {
25.                             ResetLastError();
26.                             if ((m_FileIn = FileOpen(szFileNameIn, FILE_TXT | FILE_READ)) == INVALID_HANDLE) SetUserError(C_Terminal::ERR_FileAcess);
27.                             if (_LastError == ERR_SUCCESS)
28.                             {
29.                                     if (!(m_bSimple = (StringLen(szFileNameOut) > 0))) szFileNameOut = szFileNameIn + "_T";
30.                                     if ((m_FileOut = FileOpen(szFileNameOut, FILE_TXT | FILE_WRITE)) == INVALID_HANDLE) SetUserError(C_Terminal::ERR_FileAcess);
31.                             }
32.                     }
33. //+------------------------------------------------------------------+
34.             ~C_AdjustTemplate()
35.                     {
36.                             FileClose(m_FileIn);
37.                             if (m_FileOut != INVALID_HANDLE)
38.                             {
39.                                     FileClose(m_FileOut);
40.                                     if ((!m_bSimple) && (_LastError == ERR_SUCCESS)) FileMove(m_szFileName + "_T", 0, m_szFileName, FILE_REWRITE);
41.                                     if ((!m_bSimple) && (_LastError != ERR_SUCCESS)) FileDelete(m_szFileName + "_T");
42.                             }
43.                             ArrayResize(m_szName, 0);
44.                             ArrayResize(m_szFind, 0);
45.                             ArrayResize(m_szReplace, 0);
46.                     }
47. //+------------------------------------------------------------------+
48.             void Add(const string szName, const string szFind, const string szReplace)
49.                     {
50.                             m_maxIndex++;
51.                             ArrayResize(m_szName, m_maxIndex);
52.                             ArrayResize(m_szFind, m_maxIndex);
53.                             ArrayResize(m_szReplace, m_maxIndex);
54.                             m_szName[m_maxIndex - 1] = szName;
55.                             m_szFind[m_maxIndex - 1] = szFind;
56.                             m_szReplace[m_maxIndex - 1] = szReplace;
57.                     }
58. //+------------------------------------------------------------------+
59.             string Get(const string szName, const string szFind)
60.                     {
61.                             for (int c0 = 0; c0 < m_maxIndex; c0++) if ((m_szName[c0] == szName) && (m_szFind[c0] == szFind)) return m_szReplace[c0];
62.
63.                             return NULL;
64.                     }
65. //+------------------------------------------------------------------+
66.             void Execute(void)
67.                     {
68.                             string sz0, tmp, res[];
69.                             int count0 = 0, i0;
70.
71.                             if ((m_FileIn != INVALID_HANDLE) && (m_FileOut != INVALID_HANDLE)) while ((!FileIsEnding(m_FileIn)) && (_LastError == ERR_SUCCESS))
72.                             {
73.                                     sz0 = FileReadString(m_FileIn);
74.                                     if (sz0 == "<object>") count0 = 1;
75.                                     if (sz0 == "</object>") count0 = 0;
76.                                     if (count0 > 0) if (StringSplit(sz0, '=', res) > 1)
77.                                     {
78.                                             i0 = (count0 == 1 ? 0 : i0);
79.                                             for (int c0 = 0; (c0 < m_maxIndex) && (count0 == 1); i0 = c0, c0++) count0 = (res[1] == (tmp = m_szName[c0]) ? 2 : count0);
80.                                             for (int c0 = i0; (c0 < m_maxIndex) && (count0 == 2); c0++) if ((res[0] == m_szFind[c0]) && (tmp == m_szName[c0]))
81.                                             {
82.                                                     if (StringLen(m_szReplace[c0])) sz0 =  m_szFind[c0] + "=" + m_szReplace[c0];
83.                                                     else m_szReplace[c0] = res[1];
84.                                             }
85.                                     }
86.                                     FileWriteString(m_FileOut, sz0 + "\r\n");
87.                             };
88.                     }
89. //+------------------------------------------------------------------+
90. };
91. //+------------------------------------------------------------------+
```

**Source code: C\_AdjustTemplate**

This code contains exactly what we need. If you look at this code here and look at the C\_ChartFloatingRAD class code from the previous article, you will notice that the content present between lines 38 and 90 of the C\_ChartFloatingRAD class is also here, but it looks different. This is because the data modeling in this C\_AdjustTemplate class is done in a way that promotes more efficient execution. You will see this when the new C\_ChartFloatingRAD class code is shown later in this article. So, let's leave that for later. First, let's understand what's going on in this C\_AdjustTemplate class.

Although it may seem complex and difficult to understand, the code for the C\_AdjustTemplate class is actually quite simple. However, it is designed to be executed as a single function despite having multiple functions. To really understand the whole idea, forget that you are working with code, with MetaTrader or MQL5. If you imagine that you are dealing with parts of one machine, it will be easier to understand everything. The C\_AdjustTemplate class should be thought of as a template. ,So, consider it a template file which we discussed in the previous article.

If you think about it this way, it will become clear what is really going on and why we should work with this class the way we will in the future. So when you use a class constructor, you are basically opening up the template to operate on what's inside it. When you use a destructor, it's like you're saying, "Okay, MetaTrader, you can use the template now." The other functions serve as tools to adjust the template.

With that in mind, let's take a look at how each part of this class works. Let's start with the constructor. It starts at line 19 where we see that we must provide a sting, but we can optionally provide a second string. Why is it done this way? The reason is simple: overload. If overloading were not possible, we would have to write two constructors, but since it is possible, we will use it. However, in fact, this overload is not normal, it is intended to be used in this way.

Once that's done, between lines 20 and 23 we pre-initialize some variables. This is important for us, although the compiler does implicit initialization, it is always better to do it explicitly. This way we will know exactly what the value of each variable is.

Now notice the following fact: in line 25 we "reset" the **\_LastError** constant. So if there is any error before calling the constructor, you should check it; otherwise you will lose the value specified in the error constant. I have already explained in previous articles why you need to do it this way, read them for more detailed information.

On line 26, we try to open the source file, which must be specified. If this attempt fails, we will report this in the **\_LastError** constant. If we manage to open the file, the **\_LastError** constant will contain the **ERR\_SUCCESS** value, and the check performed in line 27 will succeed, allowing us to move on to the next step.

At this point, we check on line 29 to see if a name has been specified for the destination file. If it is not specified, then we will work with a temporary file whose name will be based on the name of the original file. Having a name for the destination file, we can execute line 30, which will attempt to create the destination file. If this attempt fails, we will report this in the **\_LastError** constant. Upon successful execution of all these steps, the **\_LastError** constant will contain the **ERR\_SUCCESS** value, and we will have file identifiers. They will be used to manipulate files, so you should not try to do anything external to the files until the handles are closed. Think about the fact that the machine is open and if you turn it on, problems may occur.

So, let's follow the code in the order in which it should be edited. This brings us to line 34, where the class destructor code begins. The first thing we do on line 36 is close the input file. Note: This file will only be closed if its handle is valid. That is, the file must be open. On line 37, we check if the output file is open. We do this to avoid unnecessary execution of the following lines.

So, if the target file is open, on line 40 we check if a name was specified for it and if there were any errors during the setup process. If everything is ok, we will rename the file to match the file expected by the rest of the indicator. Anyway, on line 41, we delete the temporary file we use when something goes wrong.

Between lines 43 and 45 we free the allocated memory. Things like this are quite important, yet many people forget to do them. According to good programming practice, if you allocate memory, you should always free it. This way, MetaTrader 5 will not consume resources excessively and unnecessarily.

Next we have a very basic and simple procedure that starts at line 50 where we increment the counter and immediately after that we allocate memory to store the data we will use later. This allocation is done between lines 51 and 53. Also pay attention to lines 54 - 56, the way we will store the information. Since this process is simple, I will not go into details. Next, line 59 has one interesting function.

This function, starting at line 59, although also very simple, is quite interesting. Not so much because of what it does, but because of how it does it. Look at line 60, which is actually the only line in this function. There is a loop where we loop through all the positions that were added during the function presented in line 50. The question is, why would a programmer want to read the information written to the class using the function on line 50? It seems pointless. Indeed, if you look only at the class code, this does not make sense, but there is a nuance. A small detail that starts at line 66.

The **Execute** function that starts at line 66 is extremely delicate. The reason for this is that a lot of things can go wrong. Basically, we may encounter the following errors:

- The input file could not be read;
- The output file may not be available;
- The **StringSplit** function may fail.

Any of these problems will cause the error constant to change. If this happens, the **while** loop on line 71 will terminate prematurely, causing the entire indicator to not work correctly when placed on a chart. Remember the following: if the **\_LastError** constant contains a value other than **ERR\_SUCCESS**, the template will fail to update at the time of execution of the destructor. And if it is the first call, it will not be created. Then, if it is not there, the indicator will not be placed on the chart. That's why **Execute** is so important.

However, suppose everything is functioning perfectly. Let's see what happens inside the **while** loop.

On line 73, we read a string from the source file. This string will consist of one complete line. Reading this way will make it much easier to check the rest. Next, on line 74, we check whether a definition of some object is being entered, and on line 75 we check whether it has completed.

These checks are important to speed up the process of reading and customizing the template. In case we are not dealing with any object, we can simply write the information to the output file. This information is written in line 86, please note this. Because now we will see how the original template will be adjusted and customized to create what we need.

Once we are inside the object, we have a check on line 76 that allows us to "break" the string. This "break" will take place on the equal sign (=), which means that we are defining some parameter for some property of the object. Well, if it is true that we are defining a property, these checks will pass, allowing us to execute line 78. This line will simply adjust the temporary memory. But a question arises in the following lines.

In line 79, we loop through all the data that was added during the **Add** calls (method from line 48). If by chance we find the value of the parameter in the template, this indicates that we are dealing with the desired object. We temporarily save the name of the object and indicate that we will perform the second level of analysis. Because we are doing a second level, line 79 will not be executed again within the same object. Due to this, we must ensure that the template will have the same structure as in the previous article [Developing a Replay System (Part 43): Chart Trade Project (II)](https://www.mql5.com/en/articles/11664). The file must be exactly the same. If you change it, make sure it stays as close as possible to the one you provided previously.

Well, since we are already in the second level, in line 80, we will have another loop. Important: the loops in lines 79 and 80 are never executed together. One or the other will always be executed, but never both at once. The only exception is the first call. It may seem strange that both lines 79 and 80 are not executed, but that is actually what happens. But if line 80 does get executed, inside the loop we will have a whether the desired object property exists. Note that the object name is important, so we temporarily write it during the loop on line 79.

If this check shows that the object property is found, we perform a second check shown on line 82. At this point we will get a justification for the existence of the function found in line 59. If, during the programming process that will be discussed later, you tell the C\_AdjustTemplate class that you don't know what parameter to use in the object's property, the check on line 82 will cause line 83 to be executed, thus fixing the value present in the object's property. If you specify the value you want to use, it will be written to the template.

It is this type of functionality that makes the C\_AdjustTemplate class so interesting, since you can ask it to report the value of a property, or to specify which value to use.

This concludes the explanation of the C\_AdjustTemplate class. Now let's see what the C\_ChartFloatingRAD class turned out to be. As you can imagine, it has also been modified and has become even more interesting.

### New look for the C\_ChartFloatingRAD class

Although I won't show the final code of this class in this article (the reason is that I want to explain every detail slowly), you will notice that it will now be significantly more complex than the code presented in the previous one. Nevertheless, most of the code remains the same. This is why it is advisable to follow the articles in order, otherwise you may miss some details that may affect your overall understanding.

Below you can see the complete code for the C\_ChartFloatingRAD class, up to the current point.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "../Auxiliar/C_Mouse.mqh"
005. #include "C_AdjustTemplate.mqh"
006. //+------------------------------------------------------------------+
007. class C_ChartFloatingRAD : private C_Terminal
008. {
009.    private :
010.            enum eObjectsIDE {MSG_MAX_MIN, MSG_TITLE_IDE, MSG_DAY_TRADE, MSG_LEVERAGE_VALUE, MSG_TAKE_VALUE, MSG_STOP_VALUE, MSG_BUY_MARKET, MSG_SELL_MARKET, MSG_CLOSE_POSITION, MSG_NULL};
011.            struct st00
012.            {
013.                    int     x, y;
014.                    string  szObj_Chart,
015.                            szFileNameTemplate;
016.                    long    WinHandle;
017.                    double  FinanceTake,
018.                            FinanceStop;
019.                    int     Leverage;
020.                    bool    IsDayTrade,
021.                            IsMaximized;
022.                    struct st01
023.                    {
024.                            int x, y, w, h;
025.                    }Regions[MSG_NULL];
026.            }m_Info;
027. //+------------------------------------------------------------------+
028.            C_Mouse  *m_Mouse;
029. //+------------------------------------------------------------------+
030.            void CreateWindowRAD(int x, int y, int w, int h)
031.                    {
032.                            m_Info.szObj_Chart = (string)ObjectsTotal(GetInfoTerminal().ID);
033.                            ObjectCreate(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJ_CHART, 0, 0, 0);
034.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x = x);
035.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y = y);
036.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XSIZE, w);
037.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, h);
038.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_DATE_SCALE, false);
039.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_PRICE_SCALE, false);
040.                            m_Info.WinHandle = ObjectGetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_CHART_ID);
041.                    };
042. //+------------------------------------------------------------------+
043. inline void UpdateChartTemplate(void)
044.                    {
045.                            ChartApplyTemplate(m_Info.WinHandle, "/Files/" + m_Info.szFileNameTemplate);
046.                            ChartRedraw(m_Info.WinHandle);
047.                    }
048. //+------------------------------------------------------------------+
049. inline double PointsToFinance(const double Points)
050.                    {
051.                            return Points * (GetInfoTerminal().VolumeMinimal + (GetInfoTerminal().VolumeMinimal * (m_Info.Leverage - 1))) * GetInfoTerminal().AdjustToTrade;
052.                    };
053. //+------------------------------------------------------------------+
054. inline void AdjustTemplate(const bool bFirst = false)
055.                    {
056. #define macro_AddAdjust(A) {                         \
057.              (*Template).Add(A, "size_x", NULL);     \
058.              (*Template).Add(A, "size_y", NULL);     \
059.              (*Template).Add(A, "pos_x", NULL);      \
060.              (*Template).Add(A, "pos_y", NULL);      \
061.                            }
062. #define macro_GetAdjust(A) {                                                                          \
063.              m_Info.Regions[A].x = (int) StringToInteger((*Template).Get(EnumToString(A), "pos_x"));  \
064.              m_Info.Regions[A].y = (int) StringToInteger((*Template).Get(EnumToString(A), "pos_y"));  \
065.              m_Info.Regions[A].w = (int) StringToInteger((*Template).Get(EnumToString(A), "size_x")); \
066.              m_Info.Regions[A].h = (int) StringToInteger((*Template).Get(EnumToString(A), "size_y")); \
067.                            }
068.
069.                            C_AdjustTemplate *Template;
070.
071.                            if (bFirst)
072.                            {
073.                                    Template = new C_AdjustTemplate("Chart Trade/IDE_RAD.tpl", m_Info.szFileNameTemplate = StringFormat("Chart Trade/%u.tpl", GetInfoTerminal().ID));
074.                                    for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++) macro_AddAdjust(EnumToString(c0));
075.                            }else Template = new C_AdjustTemplate(m_Info.szFileNameTemplate);
076.                            Template.Add("MSG_NAME_SYMBOL", "descr", GetInfoTerminal().szSymbol);
077.                            Template.Add("MSG_LEVERAGE_VALUE", "descr", (string)m_Info.Leverage);
078.                            Template.Add("MSG_TAKE_VALUE", "descr", (string)m_Info.FinanceTake);
079.                            Template.Add("MSG_STOP_VALUE", "descr", (string)m_Info.FinanceStop);
080.                            Template.Add("MSG_DAY_TRADE", "state", (m_Info.IsDayTrade ? "1" : "0"));
081.                            Template.Add("MSG_MAX_MIN", "state", (m_Info.IsMaximized ? "1" : "0"));
082.                            Template.Execute();
083.                            if (bFirst) for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++) macro_GetAdjust(c0);
084.
085.                            delete Template;
086.
087.                            UpdateChartTemplate();
088.
089. #undef macro_AddAdjust
090. #undef macro_GetAdjust
091.                    }
092. //+------------------------------------------------------------------+
093.            eObjectsIDE CheckMousePosition(const int x, const int y)
094.                    {
095.                            int xi, yi, xf, yf;
096.
097.                            for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++)
098.                            {
099.                                    xi = m_Info.x + m_Info.Regions[c0].x;
100.                                    yi = m_Info.y + m_Info.Regions[c0].y;
101.                                    xf = xi + m_Info.Regions[c0].w;
102.                                    yf = yi + m_Info.Regions[c0].h;
103.                                    if ((x > xi) && (y > yi) && (x < xf) && (y < yf) && ((c0 == MSG_MAX_MIN) || (c0 == MSG_TITLE_IDE) || (m_Info.IsMaximized))) return c0;
104.                            }
105.                            return MSG_NULL;
106.                    }
107. //+------------------------------------------------------------------+
108.    public  :
109. //+------------------------------------------------------------------+
110.            C_ChartFloatingRAD(string szShortName, C_Mouse *MousePtr, const int Leverage, const double FinanceTake, const double FinanceStop)
111.                    :C_Terminal()
112.                    {
113.                            m_Mouse = MousePtr;
114.                            m_Info.Leverage = (Leverage < 0 ? 1 : Leverage);
115.                            m_Info.FinanceTake = PointsToFinance(FinanceToPoints(MathAbs(FinanceTake), m_Info.Leverage));
116.                            m_Info.FinanceStop = PointsToFinance(FinanceToPoints(MathAbs(FinanceStop), m_Info.Leverage));
117.                            m_Info.IsDayTrade = true;
118.                            m_Info.IsMaximized = true;
119.                            if (!IndicatorCheckPass(szShortName)) SetUserError(C_Terminal::ERR_Unknown);
120.                            CreateWindowRAD(115, 64, 170, 210);
121.                            AdjustTemplate(true);
122.                    }
123. //+------------------------------------------------------------------+
124.            ~C_ChartFloatingRAD()
125.                    {
126.                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Chart);
127.                            FileDelete(m_Info.szFileNameTemplate);
128.
129.                            delete m_Mouse;
130.                    }
131. //+------------------------------------------------------------------+
132.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
133.                    {
134.                            static int sx = -1, sy = -1;
135.                            int x, y, mx, my;
136.                            static eObjectsIDE it = MSG_NULL;
137.
138.                            switch (id)
139.                            {
140.                                    case CHARTEVENT_MOUSE_MOVE:
141.                                            if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft))
142.                                            {
143.                                                    switch (it = CheckMousePosition(x = (int)lparam, y = (int)dparam))
144.                                                    {
145.                                                            case MSG_TITLE_IDE:
146.                                                                    if (sx < 0)
147.                                                                    {
148.                                                                            ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
149.                                                                            sx = x - m_Info.x;
150.                                                                            sy = y - m_Info.y;
151.                                                                    }
152.                                                                    if ((mx = x - sx) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x = mx);
153.                                                                    if ((my = y - sy) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y = my);
154.                                                                    break;
155.                                                    }
156.                                            }else
157.                                            {
158.                                                    if (it != MSG_NULL)
159.                                                    {
160.                                                            switch (it)
161.                                                            {
162.                                                                    case MSG_MAX_MIN:
163.                                                                            m_Info.IsMaximized = (m_Info.IsMaximized ? false : true);
164.                                                                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, (m_Info.IsMaximized ? 210 : m_Info.Regions[MSG_TITLE_IDE].h + 6));
165.                                                                            break;
166.                                                                    case MSG_DAY_TRADE:
167.                                                                            m_Info.IsDayTrade = (m_Info.IsDayTrade ? false : true);
168.                                                                            break;
169.                                                            }
170.                                                            it = MSG_NULL;
171.                                                            AdjustTemplate();
172.                                                    }
173.                                                    if (sx > 0)
174.                                                    {
175.                                                            ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
176.                                                            sx = sy = -1;
177.                                                    }
178.                                            }
179.                                            break;
180.                            }
181.                    }
182. //+------------------------------------------------------------------+
183. };
184. //+------------------------------------------------------------------+
```

**C\_ChartFloatingRAD class source code**

I know this code seems very confusing and complicated, especially for beginners. But if you've been following this series from the beginning, you should have learned a lot about MQL5 programming by now. Anyway, this code is much more complex than what many people typically create.

If you compare this code with the one presented in the previous article, you will see that it has become more complicated. However, most of the complexity has been moved to the C\_AdjustTemplate class, which was described in the previous section. Let's figure out what this code does, because this is where the magic of the Chart Trade indicator lies. This is because the indicator code shown in the previous article remains the same. But this code shown here has changed, and changed enough to add new features to the indicator.

To start the explanation, let's look at line 10, where we have an enumeration that will make it easier for us to access the objects present in the template. But the same enumeration has the value **MSG\_NULL**, which is used for control. This will become clear in the course of further explanations.

Looking at the code, we see between lines 22 and 25 a structure that will be used by a variable that is an array. But if you look at the number of elements in the array, you may wonder what it is. We don't have a value, it's something else. You might be thinking, "I can't find anywhere what this thing means". Calm down, no need to panic. This data, which denotes the number of elements in the array, is just the last data in the enumeration done in line 10. But there is one detail here. This last value does not actually represent an element or object. If it did, the declaration should have been different.

The next line that deserves some explanation is 54. This is where we actually get access to the template. But before we get into that explanation, let's look at one more thing. This function is accessed in two location. The first is on line 121, which is the constructor, and the second is on line 171, which is in the message handling part. Why is it important to know this? The reason is what we do and what we want to do in the template.

In the first case, which occurs in the constructor, we want to customize the template so that it works in a certain way. In the second case, we will work with what we already have; we will not change the template, but we will make it fit precisely what we want.

This explanation might seem a bit confusing, but let's look at how the method on line 54 works. Maybe this will help you understand the situation better. Between lines 56 and 67 we have the definition of two macros to help us and make programming easier. Just like lines 89 and 90 serve to eliminate such macros. I usually like to use macros when I repeat the same code or some parameter multiple times. In this particular case, a parameter is repeated. The macro code is quite simple.

The first one, which is between lines 56 and 61, will add the elements that the C\_AdjustTemplate class will return to us. In the second macro, which is between lines 62 and 67, we take the values that the C\_AdjustTemplate class tells us and convert them into a value that is stored in the array declared on line 25. Pay attention to this. We don't simply guess where objects are, but we ask the template where they are.

In line 71 we check whether we are starting to adjust the template. If this is not the case, we execute the call present in line 75. But if this is the first call, we tell the C\_AdjustTemplate class what names will be used. This is done in line 73. Now pay special attention to line 74. You can see that in this line we use an enumeration to tell the C\_AdjustTemplate class what objects we need to get the data from. For this we use a loop. This way the C\_AdjustTemplate class will know which properties need to be fixed.

In any case, in lines 76 and 81, we pass to the template the values that should be used in the object properties. Each line specifies an object, a property to be changed, and a value to be used.

Finally, in line 82, we tell the C\_AdjustTemplate class that it can customize the template according to the information provided. This is done as shown in the previous topic. Once all the work is done, we check in line 83 whether this is the first call. If so, we adjust the values of the array declared on line 25. This is done using a loop that tells the C\_AdjustTemplate class what objects and properties we want to know about.

Once this work is complete, we use line 85 to close the C\_Template class. And finally, in line 87, we ask the **OBJ\_CHART** object to update. This way we will see the magic in action as shown in the demo video at the end of the article.

Please note: we do not check for any errors here, it is assumed that everything is fine and works as expected. But we will check for errors in the indicator code. Therefore, if anything fails, it will be handled not here, but in the indicator code.

Now let's look at something else: line 93 runs a rather interesting function that could be placed right where it will be used. It is done so to make the code more readable. This function has a loop that starts at line 97 and goes through each of the objects present in **OBJ\_CHART**. Remember the following: the **OBJ\_CHART** object contains a template, and this template contains the objects that we will be checking. During the check, we will create a rectangle that is the click area for each object. This is done in lines 99 and 102.

Once we have this click area, we can compare it with the area specified as call parameters. This comparison is made in line 103. In addition to the area, there are some additional conditions. If everything is ok, the index of the objects will be returned; otherwise, **MSG\_NULL** is returned. This is the reason why we need the enumeration defined at the beginning to include this value. Without this value, it would be impossible to report that a click was made on an invalid object in the Chart Trade indicator.

The next thing to explain is located on line 132. It is the event handler. Now it contains some new parts. But it's these new parts that make what you see in the demo video possible. So let's take a very close look at what's going on. And notice that up to this point, we haven't created any other object except **OBJ\_CHART**. And yet, we have the expected functioning.

Most of the code looks very similar to what was shown in the previous article. However, there are some small differences that are worth commenting on so that those less experienced can understand what is going on. In lines 134–136 we define some variables. The variable defined in line 136 is of particular interest in the framework of this article, since the others have already been explained.

We will use this variable in line 136 as memory. This is because we cannot rely on additional assistance from MetaTrader 5 to resolve click-related issues. Typically, when there are objects on the chart, MetaTrader 5 will tell us the name of the object that was clicked. This is done through the **CHARTEVENT\_OBJECT\_CLICK** event. But in this case we have no real objects other than **OBJ\_CHART**. Thus, any click in the Chart Trade indicator area will be interpreted by MetaTrader 5 as a click on **OBJ\_CHART**.

The only event we are dealing with is **CHARTEVENT\_MOUSE\_MOVE**, which is enough for us. However, clicks will only be processed if the mouse indicator is not in the study state. This is checked in line 141. If the mouse indicator is in the study state, or something else has happened, we move to line 156. And here is a question. If the variable declared in line 136 has a different value, something must happen. But first, let's look at when and where this variable gets its value.

When the mouse indicator is free and a click occurs, the check performed in line 141 will determine where and what exactly the click was made on. This is done in line 143. At this point, we tell the analysis function where the mouse is at the time of the click. However, there is a small flaw here, which I will not dwell on now, since it will be corrected in the next article, as well as other little things that we still have to do. Simultaneously with the check, the function returns the name of the object that received the click. This name is written to a static variable.

For practical reasons we are only checking the title object now. If it was clicked, line 145 will allow the drag code to execute. We could place other objects here, but that would complicate the check logic, at least at this stage. Since the mouse button is pressed, the object will continue to receive click messages.

As mentioned, we can improve this. But for now I want to keep the code as simple as possible, making changes gradually, since the concept shown here is very different from what many people usually program.

But let's return to line 156. When this line is executed, we will perform two checks within this condition. The first one will determine if any object present in **OBJ\_CHART** was clicked. If this happened, then we will have the same condition that would have been met if the object actually existed on the chart and MetaTrader 5 would have generated the **CHARTEVENT\_OBJECT\_CLICK** event. That is, we "emulate" the operation of an existing system. We do this to ensure adequate behavior of the entire indicator.

If the check presented in line 158 passes successfully, we will first perform the appropriate event handling. This is done in lines 160 and 169, and then in line 170 we remove the object reference and in line 171 we update the current state of the template. This way, the whole indicator will be updated, creating the illusion of objects being present on the chart, although the only real object is **OBJ\_CHART**.

All the rest of the code in the C\_ChartFloatingRAD class has already been explained in the previous article, so I don't see the need to comment on it again here.

Demonstração 44 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11690)

MQL5.community

1.91K subscribers

[Demonstração 44](https://www.youtube.com/watch?v=uur1i3d06FA)

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

[Watch on](https://www.youtube.com/watch?v=uur1i3d06FA&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11690)

0:00

0:00 / 2:26

•Live

•

Demo video

### Conclusion

As you can see, in this article, I have presented a way to use templates in **OBJ\_CHART**, which allows you to get behavior that is very similar to what would happen if there were real objects on the chart. Perhaps the biggest advantage of what I am showing is the ability to quickly create an interface from elements present in MetaTrader 5 itself, without using complex MQL5 programming.

Although what I am demonstrating seems quite confusing and complicated, it is only because it is something completely new to you, dear reader. Over time, as you begin to practice, you will notice that this knowledge can be widely used in a variety of scenarios and situations. However, the system is not yet complete. Besides, it has one small flaw. But it will be fixed in the next article where we will finally allow the user to directly enter values into Chart Trade. This part will be very interesting to implement. So don't miss the next article in this series.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11690](https://www.mql5.com/pt/articles/11690)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11690.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11690/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/472086)**

![MQL5 Wizard Techniques you should know (Part 35): Support Vector Regression](https://c.mql5.com/2/91/MQL5_Wizard_Techniques_you_should_know_Part_35__LOGO.png)[MQL5 Wizard Techniques you should know (Part 35): Support Vector Regression](https://www.mql5.com/en/articles/15692)

Support Vector Regression is an idealistic way of finding a function or ‘hyper-plane’ that best describes the relationship between two sets of data. We attempt to exploit this in time series forecasting within custom classes of the MQL5 wizard.

![Brain Storm Optimization algorithm (Part I): Clustering](https://c.mql5.com/2/75/Brain_Storm_Optimization_hPart_I4_____LOGO_2.png)[Brain Storm Optimization algorithm (Part I): Clustering](https://www.mql5.com/en/articles/14707)

In this article, we will look at an innovative optimization method called BSO (Brain Storm Optimization) inspired by a natural phenomenon called "brainstorming". We will also discuss a new approach to solving multimodal optimization problems the BSO method applies. It allows finding multiple optimal solutions without the need to pre-determine the number of subpopulations. We will also consider the K-Means and K-Means++ clustering methods.

![Implementing a Rapid-Fire Trading Strategy Algorithm with Parabolic SAR and Simple Moving Average (SMA) in MQL5](https://c.mql5.com/2/91/Implementing_a_Rapid_Fire_Trading_Strategy_Algorithm_with_Parabolic_SAR_and_Simple_Moving_Average___.png)[Implementing a Rapid-Fire Trading Strategy Algorithm with Parabolic SAR and Simple Moving Average (SMA) in MQL5](https://www.mql5.com/en/articles/15698)

In this article, we develop a Rapid-Fire Trading Expert Advisor in MQL5, leveraging the Parabolic SAR and Simple Moving Average (SMA) indicators to create a responsive trading strategy. We detail the strategy’s implementation, including indicator usage, signal generation, and the testing and optimization process.

![Matrix Factorization: The Basics](https://c.mql5.com/2/72/Fatorando_Matrizes_q_O_Bgsico____LOGO.png)[Matrix Factorization: The Basics](https://www.mql5.com/en/articles/13646)

Since the goal here is didactic, we will proceed as simply as possible. That is, we will implement only what we need: matrix multiplication. You will see today that this is enough to simulate matrix-scalar multiplication. The most significant difficulty that many people encounter when implementing code using matrix factorization is this: unlike scalar factorization, where in almost all cases the order of the factors does not change the result, this is not the case when using matrices.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ptahlylqprpewzvtrlkebxeareqqawtp&ssn=1769184633807749135&ssn_dr=0&ssn_sr=0&fv_date=1769184633&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11690&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2044)%3A%20Chart%20Trade%20Project%20(III)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691846333237822&fz_uniq=5070072184468082552&sv=2552)

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