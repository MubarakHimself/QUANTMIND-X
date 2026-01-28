---
title: Developing a Replay System (Part 49): Things Get Complicated (I)
url: https://www.mql5.com/en/articles/11820
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:07:54.455434
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/11820&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070033143215361752)

MetaTrader 5 / Examples


### Introduction

This article will use what was discussed in the article [Developing a Replay System (Part 48): Concepts to Understand and Think About](https://www.mql5.com/en/articles/11781). So, if you haven't read it yet, please do, because the content of this article is very important to understanding what we're going to do here.

One of the things that bothered me the most while writing the previous articles was that the replay/simulator system contained an indicator that was visible to the MetaTrader 5 user in the area where indicators are listed, from which it could be placed on the chart.

Even though the articles had a lock feature that prevented a user from trying to place such an indicator on the wrong chart, i.e. on the one different from the symbol used by the replay/simulator service, the very presence of this indicator in that list, among the others, was very disconcerting to me.

During all these months I have been trying and analyzing how everything could be organized what in the most appropriate way. Luckily, I recently managed to find a solution that improves the situation. In this case, the control indicator will no longer be present among other indicators and will become an integral part of the replay/simulator service.

By doing this, we will have a greater degree of freedom regarding some factors. However, I will be making changes gradually, as I will also be refining the indicator to reduce the load on MetaTrader 5. In other words, we will stop using some resources and start using other platform capabilities. This will improve the stability, security and reliability of the replay/simulator system.

Let's see how this will be implemented, as the changes will be very interesting to implement and will give us great knowledge on how to work more professionally with MQL5. Anyway, this does not imply a delay in the development of our replay/simulator system. In fact, the system needs some things, including those we are going to discuss in this article, in order to program something else in the future.

So, let's continue our epic journey to implement a more advanced replay/simulator service.

### Starting the changes

In order to clearly explain what will be done, I will move gradually so that you, dear reader, can follow the changes. Many may consider the method I chose for demonstration to be excessive and think that it would be possible to immediately go to the final code and that's it. But if this was exactly what was needed, then why would all the previously written articles exist? This doesn't make sense, does it? But since the purpose of this material is to motivate and encourage people to create their own programs and solutions, I must show how to clean up existing code and make it suitable for use in a more complex model. Furthermore, we should the code should not turn into something strange and inefficient.

So let's not rush, and let's continue, always thinking about those who have less experience.

First, we will remove the chart ID binding system. This system prevents the control indicator from being placed on more than one chart. At the same time, it does not allow the indicator to be placed on the wrong chart. Although we will remove this system now, it will be put back into operation later. However, when this happens, the indicator will remain on the correct chart and the replay/simulator service will perform the operation, not the indicator itself.

To ensure this removal is done correctly and safely, we will always use the same point of interaction between the service and the control indicator. This general code is shown below:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #define def_SymbolReplay               "RePlay"
05. #define def_GlobalVariableReplay       def_SymbolReplay + "_Infos"
06. #define def_GlobalVariableIdGraphics   def_SymbolReplay + "_ID"
07. #define def_GlobalVariableServerTime   def_SymbolReplay + "_Time"
08. #define def_MaxPosSlider               400
09. //+------------------------------------------------------------------+
10. union u_Interprocess
11. {
12.     union u_0
13.     {
14.             double  df_Value;       // Value of the terminal global variable...
15.             ulong   IdGraphic;      // Contains the Graph ID of the asset...
16.     }u_Value;
17.     struct st_0
18.     {
19.             bool    isPlay;         // Indicates whether we are in Play or Pause mode...
20.             bool    isWait;         // Tells the user to wait...
21.             bool    isHedging;      // If true we are in a Hedging account, if false the account is Netting...
22.             bool    isSync;         // If true indicates that the service is synchronized...
23.             ushort  iPosShift;      // Value between 0 and 400...
24.     }s_Infos;
25.     datetime        ServerTime;
26. };
27. //+------------------------------------------------------------------+
28. union uCast_Double
29. {
30.     double   dValue;
31.     long     _long;                  // 1 Information
32.     datetime _datetime;              // 1 Information
33.     int      _int[sizeof(double)];   // 2 Informations
34.     char     _char[sizeof(double)];  // 8 Informations
35. };
36. //+------------------------------------------------------------------+
```

Source code in Interprocess.mqh

Line 06 contains a definition that tells both the service and the indicator what global terminal variable name will be used to pass the chart ID. If we remove this line, we will get a series if errors when trying to compile the indicator and service. This is exactly what we need. In fact, every error generated during compilation points to a place where we should intervene to eliminate this connection that exists between the service and the indicator via the global terminal variable.

To keep this from being too tedious and repetitive, I will show this in code fragments. I will always specify where and how exactly you need to act in the source code so that the system continues to function.

So, let's start by considering the following:

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property copyright "Daniel Jose"
05. #property version   "1.49"
06. #property description "Replay-Simulator service for MT5 platform."
07. #property description "This is dependent on the Market Replay indicator."
08. #property description "For more details on this version see the article."
09. #property link "https://www.mql5.com/ru/articles/11820"
10. //+------------------------------------------------------------------+
11. #define def_Dependence_01   "Indicators\\Replay\\Market Replay.ex5"
12. #resource "\\" + def_Dependence_01
13. //+------------------------------------------------------------------+
14. #include <Market Replay\Service Graphics\C_Replay.mqh>
15. //+------------------------------------------------------------------+
16. input string           user00 = "Forex - EURUSD.txt";  //Replay configuration file.
17. input ENUM_TIMEFRAMES  user01 = PERIOD_M5;             //Initial graphic time.
18. //+------------------------------------------------------------------+
19. void OnStart()
20. {
21.     C_Replay  *pReplay;
22.
23.     pReplay = new C_Replay(user00);
24.     if ((*pReplay).ViewReplay(user01))
25.     {
26.             Print("Permission granted. Replay service can now be used...");
27.             while ((*pReplay).LoopEventOnTime(false));
28.     }
29.     delete pReplay;
30. }
31. //+------------------------------------------------------------------+
```

Source code: replay.mq5 service

In the above code, which is the service code itself, you can see that in lines 11 and 12, we have some differences from what we had in the previous codes. Although the difference is minor, it will contribute to what we want to do. We'll get back to this later. The question here is whether the control indicator was compiled recently or not, after we changed the header code by removing the definition of the global terminal variable name.

Well, at the time of writing this article, we don't yet have a MAKE compilation system in MetaEditor. But who knows, maybe in the future, those who create the MQL platform and language will implement this opportunity. But until then, we need to take certain precautions.

Before I continue the explanation, let me pause for a moment and explain what the MAKE compilation system is. For many this will be something new, but for those who have been professionally involved in programming for many years, this system is very familiar. It works like this: you build your source code in the usual way, but sometimes you may need to build multiple executables at once. It can also be used to create a single executable, although this does not necessarily have to be done via MAKE.

So, in fact, before you compile something, you create another file. This is a MakeFile file that contains steps, definitions, configurations and settings needed to allow the compiler and LinkEditor to generate all the executables in one go. The big advantage of this system is that if you change one file, be it a header or a library, MAKE will detect it and compile only those files that are really needed. This way you will end up with all executables updated as needed.

Thus, the developer does not have to worry about what to update when creating the final executable as this is done by MAKE. Thus you avoid the risk of creating something or fixing a bug, applying it in one executable and skipping in another. So, MAKE can do all the work for us.

However, at the time of writing, such a tool does not yet exist in MetaEditor. It is possible to create something similar using batch files from the command line, but it would only be a temporary solution and not ideal. So I won't go into details of how this is done. But since line 12 of the above code tells the MQL5 compiler that the control indicator executable should be added to the service, you can simply ask it to compile the service code, and the indicator code will be compiled along with it. I have already mentioned this before, but I will emphasize it again here: in order for the indicator to be compiled correctly, each time you compile the service code, you will need to delete the indicator executable file. If this is not done, the indicator code will not be recompiled when the service code is compiled. Please pay attention to this.

When you try to compile the service code, you will get the following result:

![Figure 01](https://c.mql5.com/2/50/001__3.png)

Figure 01. Result of trying to compile new code

You may notice that we have 2 errors and a warning. Warning is not, and should not be, your first concern. First, the errors need to be corrected. If you look at image 01, you can see where the errors are. Click on them to go to the exact location where they happened.

The error shown in Image 01 and present on line 12 appeared because the compiler could not find an executable for the service resource.

Now the error reported in line 156 is part of the service code, and we need to fix it to continue compiling.

Below is the code related to that error:

```
141. ~C_Replay()
142.    {
143.            ArrayFree(m_Ticks.Info);
144.            ArrayFree(m_Ticks.Rate);
145.            m_IdReplay = ChartFirst();
146.            do
147.            {
148.                    if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
149.                            ChartClose(m_IdReplay);
150.            }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
151.            for (int c0 = 0; (c0 < 2) && (!SymbolSelect(def_SymbolReplay, false)); c0++);
152.            CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
153.            CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
154.            CustomSymbolDelete(def_SymbolReplay);
155.            GlobalVariableDel(def_GlobalVariableReplay);
156.            GlobalVariableDel(def_GlobalVariableIdGraphics);
157.            GlobalVariableDel(def_GlobalVariableServerTime);
158.            Print("Finished replay service...");
159.    }
```

Code from C\_Replay.mqh

You can comment out or simply remove line 156 from the code. Since the change will be permanent, let's remove this line right now. This way the service will no longer see this global terminal variable. However, after making the fix in the service, when we try to recompile the code, we get the result as in image 02.

![Figure 02](https://c.mql5.com/2/50/002__1.png)

Image 02. New compilation attempt

Now we only have an indication of one error, and no more warnings. At this point, many less experienced readers are wondering how to fix this displayed error. However, they forget to read the compiler message.

Look at figure 02. Just above the error message, we see some other information. NEVER ignore what the compiler tells you. You should pay attention to absolutely EVERYTHING. The message immediately preceding the error indication contains the following information from the compiler:

**_compiling '\\Indicators\\Replay\\Market Replay.mq5' failed_**

This message is key because it tells us that the compiler is unable to create an executable indicator file for some reason. Since the service code is partially compiled, we need to focus on the indicator code. We open the given code in MetaEditor and ask the compiler to try to generate an executable file. And before anyone thinks, "But why do that when we know the code has errors?", yes, we know they are there, but we want the compiler to tell us where exactly. Searching for errors manually is unproductive. Let the compiler show them.

So when you open the indicator code and ask the compiler to generate an executable file, you will see figure 03.

![Figure 03](https://c.mql5.com/2/50/003__1.png)

Image 03. Trying to compile the control indicator

This time we see indication of five errors and five warnings. Handle errors first, then warnings. The first time we click on the first error message, we will be redirected to the code at the exact location of the error. Important point: many people think that they can click on errors randomly. But according to the rules, you should act as follows: find the first error in the list. Don't pay attention to the rest, always start with the first one. Make the necessary corrections and try compiling the code again. Repeat this until the code compiles without errors or warnings. Very often, especially for beginners, panic occurs when they see 100 or more compilation errors.

However, in many cases, after fixing the first error, the entire code compiles without problems. So follow this advice: look for the first error the compiler points to. See how to fix it properly and try compiling the code again. If a new error appears, go to the first one listed and fix it, and so on until there are no errors left. The same should be done in case of warnings. Always start with the first one on the list.

Since all the errors reported by the compiler (shown in Figure 03) are in the same code, it's easier to show the entire code so you know where the changes will be made.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property description "Control indicator for the Replay-Simulator service."
05. #property description "This one doesn't work without the service loaded."
06. #property version   "1.49"
07. #property link "https://www.mql5.com/ru/articles/11820"
08. #property indicator_chart_window
09. #property indicator_plots 0
10. //+------------------------------------------------------------------+
11. #include <Market Replay\Service Graphics\C_Controls.mqh>
12. //+------------------------------------------------------------------+
13. #define def_BitShift ((sizeof(ulong) * 8) - 1)
14. //+------------------------------------------------------------------+
15. C_Terminal *terminal = NULL;
16. C_Controls *control = NULL;
17. //+------------------------------------------------------------------+
18. #define def_InfoTerminal (*terminal).GetInfoTerminal()
19. #define def_ShortName       "Market_" + def_SymbolReplay
20. //+------------------------------------------------------------------+
21. int OnInit()
22. {
23. #define macro_INIT_FAILED { ChartIndicatorDelete(def_InfoTerminal.ID, 0, def_ShortName); return INIT_FAILED; }
24.     u_Interprocess Info;
25.     ulong ul = 1;
26.
27.     ResetLastError();
28.     if (CheckPointer(control = new C_Controls(terminal = new C_Terminal())) == POINTER_INVALID) return INIT_FAILED;
29.     if (_LastError != ERR_SUCCESS) return INIT_FAILED;
30.     ul <<= def_BitShift;
31.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
32.     if ((def_InfoTerminal.szSymbol != def_SymbolReplay) || (!GlobalVariableCheck(def_GlobalVariableIdGraphics))) macro_INIT_FAILED;
33.     Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableIdGraphics);
34.     if (Info.u_Value.IdGraphic != def_InfoTerminal.ID) macro_INIT_FAILED;
35.     if ((Info.u_Value.IdGraphic >> def_BitShift) == 1) macro_INIT_FAILED;
36.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName + "Device");
37.     Info.u_Value.IdGraphic |= ul;
38.     GlobalVariableSet(def_GlobalVariableIdGraphics, Info.u_Value.df_Value);
39.     if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.u_Value.df_Value = 0;
40.     EventChartCustom(def_InfoTerminal.ID, C_Controls::ev_WaitOff, 1, Info.u_Value.df_Value, "");
41.     (*control).Init(Info.s_Infos.isPlay);
42.
43.     return INIT_SUCCEEDED;
44.
45. #undef macro_INIT_FAILED
46. }
47. //+------------------------------------------------------------------+
48. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
49. {
50.     static bool bWait = false;
51.     u_Interprocess Info;
52.
53.     Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
54.     if (!bWait)
55.     {
56.             if (Info.s_Infos.isWait)
57.             {
58.                     EventChartCustom(def_InfoTerminal.ID, C_Controls::ev_WaitOn, 1, 0, "");
59.                     bWait = true;
60.             }
61.     }else if (!Info.s_Infos.isWait)
62.     {
63.             EventChartCustom(def_InfoTerminal.ID, C_Controls::ev_WaitOff, 1, Info.u_Value.df_Value, "");
64.             bWait = false;
65.     }
66.
67.     return rates_total;
68. }
69. //+------------------------------------------------------------------+
70. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
71. {
72.     (*control).DispatchMessage(id, lparam, dparam, sparam);
73. }
74. //+------------------------------------------------------------------+
75. void OnDeinit(const int reason)
76. {
77.     u_Interprocess Info;
78.     ulong ul = 1;
79.
80.     switch (reason)
81.     {
82.             case REASON_CHARTCHANGE:
83.                     ul <<= def_BitShift;
84.                     Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableIdGraphics);
85.                     Info.u_Value.IdGraphic ^= ul;
86.                     GlobalVariableSet(def_GlobalVariableIdGraphics, Info.u_Value.df_Value);
87.                     break;
88.             case REASON_REMOVE:
89.             case REASON_CHARTCLOSE:
90.                     if (def_InfoTerminal.szSymbol != def_SymbolReplay) break;
91.                     GlobalVariableDel(def_GlobalVariableReplay);
92.                     ChartClose(def_InfoTerminal.ID);
93.                     break;
94.     }
95.     delete control;
96.     delete terminal;
97. }
98. //+------------------------------------------------------------------+
```

Source code of the indicator: Market replay.mq5

All crossed out lines in the code no longer exist. So, when you try to compile again, you will get the following message as shown in Image 04.

![Figure 04](https://c.mql5.com/2/50/004__1.png)

Figure 04

Please note that there is a warning in this image 04. It doesn't actually interfere with the code, but it can be annoying. So go to line 77 and delete it, then try compiling the code again. You will get the message shown in image 05 and this is exactly what we need. Note: Line 77 can be removed only because the compiler told us that the variable is not used.

![Figure 05](https://c.mql5.com/2/50/005__1.png)

Figure 05. Compilation completed successfully

Very good. Our code has been partially cleaned up. However, if you look back at the Interprocess.mqh header file code shown at the beginning of this topic, you'll notice that there's still some stuff in the code that we won't need anymore. This is because we no longer use the global terminal variable to pass the chart ID to the indicator. So, line 15 of the Interprocess.mqh file should be removed. But the question arises: "Why didn't I delete this line earlier?" The reason is that the service file should be treated with special care. But there is another reason, and to understand it, let's move on to the next topic.

### Radicalizing the decisions

When the replay/simulator service was designed, one of the things I tried to implement was to eliminate the possibility of user intervention in what should or should not be present in the chart.

I found a way to solve this problem by adding a global terminal variable to tell the indicator on which chart it should be present. The user would not be able to place it on another chart. In fact, this solution was adequate and quite interesting to implement. You can also use something like this if you want to make sure that an indicator, script or EA is not placed on the chart. But that is not what we are interested in right now.

The moment we use a service to control what should or should not be on the chart, such a system becomes completely unnecessary. Or better said: since the indicator will be present as a service resource and will not be accessible to the user, there is no point in continuing to support that code. The level of security has increased significantly and access control is becoming completely different.

So, since you need to be careful when deleting an ID variable from the system, I didn't delete it in the previous topic. And soon you will understand why. If the deletion had been done ahead of time, it would have become very difficult to make the appropriate changes and we would have made many mistakes.

The greatest asset of a great programmer lies precisely in this: DON'T RUSH. Solve problem after problem, gradually correcting and changing things in such a way as to preserve all the existing capabilities of the program, and, if necessary or interesting, expand and multiply them. Therefore, the motto is:

**_Divide and conquer._**

Let's see how the code will be cleaned up. The entire new code of Interprocess.mqh is shown below:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #define def_SymbolReplay             "RePlay"
05. #define def_GlobalVariableReplay     def_SymbolReplay + "_Infos"
06. #define def_GlobalVariableServerTime def_SymbolReplay + "_Time"
07. #define def_MaxPosSlider             400
08. //+------------------------------------------------------------------+
09. union u_Interprocess
10. {
11.     double  df_Value; // Value of the terminal global variable...
12.     struct st_0
13.     {
14.             bool    isPlay;     // Indicates whether we are in Play or Pause mode...
15.             bool    isWait;     // Tells the user to wait...
16.             bool    isHedging;  // If true we are in a Hedging account, if false the account is Netting...
17.             bool    isSync;     // If true indicates that the service is synchronized...
18.             ushort  iPosShift;  // Value between 0 and 400...
19.     }s_Infos;
20.     datetime ServerTime;
21. };
22. //+------------------------------------------------------------------+
23. union uCast_Double
24. {
25.     double  dValue;
26.     long     _long;                  // 1 Information
27.     datetime _datetime;              // 1 Information
28.     int      _int[sizeof(double)];   // 2 Informations
29.     char     _char[sizeof(double)];  // 8 Informations
30. };
31. //+------------------------------------------------------------------+
```

Source code: Interprocess.mqh

Note that this code, although it doesn't look much different, will make a big difference to the system we just modified in the previous section. Therefore, when you try to compile the system again, do not be surprised or frightened by the number of errors that appear. The only thing we will need to do is to configure everything to achieve the same performance as before. But now the control indicator will no longer be available to the user. The service will be responsible for its placement and support on the correct chart.

When you try to compile the service code, you will get the following output from the compiler. It is shown in figure 06.

![Figure 06](https://c.mql5.com/2/50/006__2.png)

Figure 06. Multiple errors. But are there really that many of them?

The number of errors is quite high, as is the number of warnings. As I mentioned above, we'll start with the first error on this list. Therefore, the changes start from line 28 of the C\_Replay.mqh header file. Not to make this too boring, let's take a look at the code below, since most of what we'll need to do is remove **u\_Value**. The code without this reference is here:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "C_ConfigService.mqh"
005. //+------------------------------------------------------------------+
006. class C_Replay : private C_ConfigService
007. {
008.    private :
009.            long   m_IdReplay;
010.            struct st01
011.            {
012.                   MqlRates Rate[1];
013.                   datetime memDT;
014.            }m_MountBar;
015.            struct st02
016.            {
017.                   bool    bInit;
018.                   double  PointsPerTick;
019.                   MqlTick tick[1];
020.            }m_Infos;
021. //+------------------------------------------------------------------+
022.            void AdjustPositionToReplay(const bool bViewBuider)
023.                    {
024.                            u_Interprocess Info;
025.                            MqlRates       Rate[def_BarsDiary];
026.                            int            iPos, nCount;
027.
028.                            Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
029.                            if (Info.s_Infos.iPosShift == (int)((m_ReplayCount * def_MaxPosSlider * 1.0) / m_Ticks.nTicks)) return;
030.                            iPos = (int)(m_Ticks.nTicks * ((Info.s_Infos.iPosShift * 1.0) / (def_MaxPosSlider + 1)));
031.                            Rate[0].time = macroRemoveSec(m_Ticks.Info[iPos].time);
032.                            CreateBarInReplay(true);
033.                            if (bViewBuider)
034.                            {
035.                                    Info.s_Infos.isWait = true;
036.                                    GlobalVariableSet(def_GlobalVariableReplay, Info.df_Value);
037.                            }else
038.                            {
039.                                    for(; Rate[0].time > (m_Ticks.Info[m_ReplayCount].time); m_ReplayCount++);
040.                                    for (nCount = 0; m_Ticks.Rate[nCount].time < macroRemoveSec(m_Ticks.Info[iPos].time); nCount++);
041.                                    nCount = CustomRatesUpdate(def_SymbolReplay, m_Ticks.Rate, nCount);
042.                            }
043.                            for (iPos = (iPos > 0 ? iPos - 1 : 0); (m_ReplayCount < iPos) && (!_StopFlag);) CreateBarInReplay(false);
044.                            CustomTicksAdd(def_SymbolReplay, m_Ticks.Info, m_ReplayCount);
045.                            Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
046.                            Info.s_Infos.isWait = false;
047.                            GlobalVariableSet(def_GlobalVariableReplay, Info.df_Value);
048.                    }
049. //+------------------------------------------------------------------+
050. inline void CreateBarInReplay(const bool bViewTicks)
051.                    {
052. #define def_Rate m_MountBar.Rate[0]
053.
054.                            bool    bNew;
055.                            double  dSpread;
056.                            int     iRand = rand();
057.
058.                            if (BuildBar1Min(m_ReplayCount, def_Rate, bNew))
059.                            {
060.                                    m_Infos.tick[0] = m_Ticks.Info[m_ReplayCount];
061.                                    if ((!m_Ticks.bTickReal) && (m_Ticks.ModePlot == PRICE_EXCHANGE))
062.                                    {
063.                                            dSpread = m_Infos.PointsPerTick + ((iRand > 29080) && (iRand < 32767) ? ((iRand & 1) == 1 ? m_Infos.PointsPerTick : 0 ) : 0 );
064.                                            if (m_Infos.tick[0].last > m_Infos.tick[0].ask)
065.                                            {
066.                                                    m_Infos.tick[0].ask = m_Infos.tick[0].last;
067.                                                    m_Infos.tick[0].bid = m_Infos.tick[0].last - dSpread;
068.                                            }else   if (m_Infos.tick[0].last < m_Infos.tick[0].bid)
069.                                            {
070.                                                    m_Infos.tick[0].ask = m_Infos.tick[0].last + dSpread;
071.                                                    m_Infos.tick[0].bid = m_Infos.tick[0].last;
072.                                            }
073.                                    }
074.                                    if (bViewTicks) CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
075.                                    CustomRatesUpdate(def_SymbolReplay, m_MountBar.Rate);
076.                            }
077.                            m_ReplayCount++;
078. #undef def_Rate
079.                    }
080. //+------------------------------------------------------------------+
081.            void ViewInfos(void)
082.                    {
083.                            MqlRates Rate[1];
084.
085.                            ChartSetInteger(m_IdReplay, CHART_SHOW_ASK_LINE, m_Ticks.ModePlot == PRICE_FOREX);
086.                            ChartSetInteger(m_IdReplay, CHART_SHOW_BID_LINE, m_Ticks.ModePlot == PRICE_FOREX);
087.                            ChartSetInteger(m_IdReplay, CHART_SHOW_LAST_LINE, m_Ticks.ModePlot == PRICE_EXCHANGE);
088.                            m_Infos.PointsPerTick = SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE);
089.                            m_MountBar.Rate[0].time = 0;
090.                            m_Infos.bInit = true;
091.                            CopyRates(def_SymbolReplay, PERIOD_M1, 0, 1, Rate);
092.                            if ((m_ReplayCount == 0) && (m_Ticks.ModePlot == PRICE_EXCHANGE))
093.                                    for (; m_Ticks.Info[m_ReplayCount].volume_real == 0; m_ReplayCount++);
094.                            if (Rate[0].close > 0)
095.                            {
096.                                    if (m_Ticks.ModePlot == PRICE_EXCHANGE) m_Infos.tick[0].last = Rate[0].close; else
097.                                    {
098.                                            m_Infos.tick[0].bid = Rate[0].close;
099.                                            m_Infos.tick[0].ask = Rate[0].close + (Rate[0].spread * m_Infos.PointsPerTick);
100.                                    }
101.                                    m_Infos.tick[0].time = Rate[0].time;
102.                                    m_Infos.tick[0].time_msc = Rate[0].time * 1000;
103.                            }else
104.                                    m_Infos.tick[0] = m_Ticks.Info[m_ReplayCount];
105.                            CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
106.                            ChartRedraw(m_IdReplay);
107.                    }
108. //+------------------------------------------------------------------+
109.            void CreateGlobalVariable(const string szName, const double value)
110.                    {
111.                            GlobalVariableDel(szName);
112.                            GlobalVariableTemp(szName);
113.                            GlobalVariableSet(szName, value);
114.                    }
115. //+------------------------------------------------------------------+
116.    public  :
117. //+------------------------------------------------------------------+
118.            C_Replay(const string szFileConfig)
119.                    {
120.                            m_ReplayCount = 0;
121.                            m_dtPrevLoading = 0;
122.                            m_Ticks.nTicks = 0;
123.                            m_Infos.bInit = false;
124.                            Print("************** Market Replay Service **************");
125.                            srand(GetTickCount());
126.                            GlobalVariableDel(def_GlobalVariableReplay);
127.                            SymbolSelect(def_SymbolReplay, false);
128.                            CustomSymbolDelete(def_SymbolReplay);
129.                            CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay), _Symbol);
130.                            CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
131.                            CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
132.                            CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0);
133.                            CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, 0);
134.                            CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, 0);
135.                            CustomSymbolSetString(def_SymbolReplay, SYMBOL_DESCRIPTION, "Symbol for replay / simulation");
136.                            CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, 8);
137.                            m_IdReplay = (SetSymbolReplay(szFileConfig) ? 0 : -1);
138.                            SymbolSelect(def_SymbolReplay, true);
139.                    }
140. //+------------------------------------------------------------------+
141.            ~C_Replay()
142.                    {
143.                            ArrayFree(m_Ticks.Info);
144.                            ArrayFree(m_Ticks.Rate);
145.                            m_IdReplay = ChartFirst();
146.                            do
147.                            {
148.                                    if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
149.                                            ChartClose(m_IdReplay);
150.                            }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
151.                            for (int c0 = 0; (c0 < 2) && (!SymbolSelect(def_SymbolReplay, false)); c0++);
152.                            CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
153.                            CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
154.                            CustomSymbolDelete(def_SymbolReplay);
155.                            GlobalVariableDel(def_GlobalVariableReplay);
156.                            GlobalVariableDel(def_GlobalVariableServerTime);
157.                            Print("Finished replay service...");
158.                    }
159. //+------------------------------------------------------------------+
160.            bool ViewReplay(ENUM_TIMEFRAMES arg1)
161.                    {
162. #define macroError(A) { Print(A); return false; }
163.                            u_Interprocess info;
164.
165.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
166.                                    macroError("Asset configuration is not complete, it remains to declare the size of the ticket.");
167.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
168.                                    macroError("Asset configuration is not complete, need to declare the ticket value.");
169.                            if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
170.                                    macroError("Asset configuration not complete, need to declare the minimum volume.");
171.                            if (m_IdReplay == -1) return false;
172.                            if ((m_IdReplay = ChartFirst()) > 0) do
173.                            {
174.                                    if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
175.                                    {
176.                                            ChartClose(m_IdReplay);
177.                                            ChartRedraw();
178.                                    }
179.                            }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
180.                            Print("Waiting for [Market Replay] indicator permission to start replay ...");
181.                            info.ServerTime = ULONG_MAX;
182.                            CreateGlobalVariable(def_GlobalVariableServerTime, info.df_Value);
183.                            m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
184.                            ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");
185.                            while ((!GlobalVariableGet(def_GlobalVariableReplay, info.df_Value)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);
186.                            info.s_Infos.isHedging = TypeAccountIsHedging();
187.                            info.s_Infos.isSync = true;
188.                            GlobalVariableSet(def_GlobalVariableReplay, info.df_Value);
189.
190.                            return ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""));
191. #undef macroError
192.                    }
193. //+------------------------------------------------------------------+
194.            bool LoopEventOnTime(const bool bViewBuider)
195.                    {
196.                            u_Interprocess Info;
197.                            int iPos, iTest, iCount;
198.
199.                            if (!m_Infos.bInit) ViewInfos();
200.                            iTest = 0;
201.                            while ((iTest == 0) && (!_StopFlag))
202.                            {
203.                                    iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);
204.                                    iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.df_Value) ? iTest : -1);
205.                                    iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);
206.                                    if (iTest == 0) Sleep(100);
207.                            }
208.                            if ((iTest < 0) || (_StopFlag)) return false;
209.                            AdjustPositionToReplay(bViewBuider);
210.                            Info.ServerTime = m_Ticks.Info[m_ReplayCount].time;
211.                            GlobalVariableSet(def_GlobalVariableServerTime, Info.df_Value);
212.                            iPos = iCount = 0;
213.                            while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))
214.                            {
215.                                    iPos += (int)(m_ReplayCount < (m_Ticks.nTicks - 1) ? m_Ticks.Info[m_ReplayCount + 1].time_msc - m_Ticks.Info[m_ReplayCount].time_msc : 0);
216.                                    CreateBarInReplay(true);
217.                                    while ((iPos > 200) && (!_StopFlag))
218.                                    {
219.                                            if (ChartSymbol(m_IdReplay) == "") return false;
220.                                            GlobalVariableGet(def_GlobalVariableReplay, Info.df_Value);
221.                                            if (!Info.s_Infos.isPlay) return true;
222.                                            Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
223.                                            GlobalVariableSet(def_GlobalVariableReplay, Info.df_Value);
224.                                            Sleep(195);
225.                                            iPos -= 200;
226.                                            iCount++;
227.                                            if (iCount > 4)
228.                                            {
229.                                                    iCount = 0;
230.                                                    GlobalVariableGet(def_GlobalVariableServerTime, Info.df_Value);
231.                                                    if ((m_Ticks.Info[m_ReplayCount].time - m_Ticks.Info[m_ReplayCount - 1].time) > 60) Info.ServerTime = ULONG_MAX; else
232.                                                    {
233.                                                            Info.ServerTime += 1;
234.                                                            Info.ServerTime = ((Info.ServerTime + 1) < m_Ticks.Info[m_ReplayCount].time ? Info.ServerTime : m_Ticks.Info[m_ReplayCount].time);
235.                                                    };
236.                                                    GlobalVariableSet(def_GlobalVariableServerTime, Info.df_Value);
237.                                            }
238.                                    }
239.                            }
240.                            return (m_ReplayCount == m_Ticks.nTicks);
241.                    }
242. //+------------------------------------------------------------------+
243. };
244. //+------------------------------------------------------------------+
245. #undef macroRemoveSec
246. #undef def_SymbolReplay
247. //+------------------------------------------------------------------+
```

Source code of the file C\_Replay.mqh

The code above has been corrected to suit what we are going to do in this article. But the indicator code is still missing, and if you compile the service code again with the changes made to the C\_Replay.mqh header file, the result will be as shown in Figure 07.

![Figure 07](https://c.mql5.com/2/50/007__2.png)

Figure 07. Errors still present in the Indicator

So, now we need to fix the indicator code. When you try to do this, you will get the result as in figure 08.

![Figure 08](https://c.mql5.com/2/50/008__2.png)

Figure 08. Errors that occur for the same reason

Again, we continue step by step. Always start with the first error.

Then, if the indicator file is modified correctly, you will get the following code:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property description "Control indicator for the Replay-Simulator service."
05. #property description "This one doesn't work without the service loaded."
06. #property version   "1.49"
07. #property link "https://www.mql5.com/ru/articles/11820"
08. #property indicator_chart_window
09. #property indicator_plots 0
10. //+------------------------------------------------------------------+
11. #include <Market Replay\Service Graphics\C_Controls.mqh>
12. //+------------------------------------------------------------------+
13. C_Terminal *terminal = NULL;
14. C_Controls *control = NULL;
15. //+------------------------------------------------------------------+
16. #define def_InfoTerminal (*terminal).GetInfoTerminal()
17. #define def_ShortName       "Market_" + def_SymbolReplay
18. //+------------------------------------------------------------------+
19. int OnInit()
20. {
21.     u_Interprocess Info;
22.
23.     ResetLastError();
24.     if (CheckPointer(control = new C_Controls(terminal = new C_Terminal())) == POINTER_INVALID) return INIT_FAILED;
25.     if (_LastError != ERR_SUCCESS) return INIT_FAILED;
26.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
27.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName + "Device");
28.     if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.df_Value = 0;
29.     EventChartCustom(def_InfoTerminal.ID, C_Controls::ev_WaitOff, 1, Info.df_Value, "");
30.     (*control).Init(Info.s_Infos.isPlay);
31.
32.     return INIT_SUCCEEDED;
33. }
34. //+------------------------------------------------------------------+
35. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
36. {
37.     static bool bWait = false;
38.     u_Interprocess Info;
39.
40.     Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
41.     if (!bWait)
42.     {
43.             if (Info.s_Infos.isWait)
44.             {
45.                     EventChartCustom(def_InfoTerminal.ID, C_Controls::ev_WaitOn, 1, 0, "");
46.                     bWait = true;
47.             }
48.     }else if (!Info.s_Infos.isWait)
49.     {
50.             EventChartCustom(def_InfoTerminal.ID, C_Controls::ev_WaitOff, 1, Info.df_Value, "");
51.             bWait = false;
52.     }
53.
54.     return rates_total;
55. }
56. //+------------------------------------------------------------------+
57. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
58. {
59.     (*control).DispatchMessage(id, lparam, dparam, sparam);
60. }
61. //+------------------------------------------------------------------+
62. void OnDeinit(const int reason)
63. {
64.     switch (reason)
65.     {
66.             case REASON_REMOVE:
67.             case REASON_CHARTCLOSE:
68.                     if (def_InfoTerminal.szSymbol != def_SymbolReplay) break;
69.                     GlobalVariableDel(def_GlobalVariableReplay);
70.                     ChartClose(def_InfoTerminal.ID);
71.                     break;
72.     }
73.     delete control;
74.     delete terminal;
75. }
76. //+------------------------------------------------------------------+
```

Replay indicator source code

This code is now completely fixed, but when you try to compile it, you will still get some errors from the compiler, which can be seen in Figure 09.

![Figure 09](https://c.mql5.com/2/50/009__2.png)

Figure 09. Trying to compile the control indicator

Now we need to go to the C\_Controls.mqh header file and make some corrections. But these fixes are fairly simple. All you'll need to do is remove any references to **u\_Value**. So, you end up with the following code:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\Auxiliar\Interprocess.mqh"
005. //+------------------------------------------------------------------+
006. #define def_PathBMP                "Images\\Market Replay\\Control\\"
007. #define def_ButtonPlay             def_PathBMP + "Play.bmp"
008. #define def_ButtonPause            def_PathBMP + "Pause.bmp"
009. #define def_ButtonLeft             def_PathBMP + "Left.bmp"
010. #define def_ButtonLeftBlock        def_PathBMP + "Left_Block.bmp"
011. #define def_ButtonRight            def_PathBMP + "Right.bmp"
012. #define def_ButtonRightBlock       def_PathBMP + "Right_Block.bmp"
013. #define def_ButtonPin              def_PathBMP + "Pin.bmp"
014. #define def_ButtonWait             def_PathBMP + "Wait.bmp"
015. #resource "\\" + def_ButtonPlay
016. #resource "\\" + def_ButtonPause
017. #resource "\\" + def_ButtonLeft
018. #resource "\\" + def_ButtonLeftBlock
019. #resource "\\" + def_ButtonRight
020. #resource "\\" + def_ButtonRightBlock
021. #resource "\\" + def_ButtonPin
022. #resource "\\" + def_ButtonWait
023. //+------------------------------------------------------------------+
024. #define def_PrefixObjectName       "Market Replay _ "
025. #define def_NameObjectsSlider      def_PrefixObjectName + "Slider"
026. #define def_PosXObjects            120
027. //+------------------------------------------------------------------+
028. #include "..\Auxiliar\C_Terminal.mqh"
029. #include "..\Auxiliar\C_Mouse.mqh"
030. //+------------------------------------------------------------------+
031. #define def_AcessTerminal (*Terminal)
032. #define def_InfoTerminal def_AcessTerminal.GetInfoTerminal()
033. //+------------------------------------------------------------------+
034. class C_Controls : protected C_Mouse
035. {
036.    protected:
037.            enum EventCustom {ev_WaitOn, ev_WaitOff};
038.    private :
039. //+------------------------------------------------------------------+
040.            string  m_szBtnPlay;
041.            bool            m_bWait;
042.            struct st_00
043.            {
044.                    string  szBtnLeft,
045.                            szBtnRight,
046.                            szBtnPin,
047.                            szBarSlider,
048.                            szBarSliderBlock;
049.                    int     posPinSlider,
050.                            posY,
051.                            Minimal;
052.            }m_Slider;
053.            C_Terminal *Terminal;
054. //+------------------------------------------------------------------+
055. inline void CreateObjectBitMap(int x, int y, string szName, string Resource1, string Resource2 = NULL)
056.                    {
057.                            ObjectCreate(def_InfoTerminal.ID, szName, OBJ_BITMAP_LABEL, 0, 0, 0);
058.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_XDISTANCE, def_PosXObjects + x);
059.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_YDISTANCE, y);
060.                            ObjectSetString(def_InfoTerminal.ID, szName, OBJPROP_BMPFILE, 0, "::" + Resource1);
061.                            ObjectSetString(def_InfoTerminal.ID, szName, OBJPROP_BMPFILE, 1, "::" + (Resource2 == NULL ? Resource1 : Resource2));
062.                            ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_ZORDER, 1);
063.                    }
064. //+------------------------------------------------------------------+
065. inline void CreteBarSlider(int x, int size)
066.                    {
067.                            ObjectCreate(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJ_RECTANGLE_LABEL, 0, 0, 0);
068.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJPROP_XDISTANCE, def_PosXObjects + x);
069.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJPROP_YDISTANCE, m_Slider.posY - 4);
070.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJPROP_XSIZE, size);
071.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJPROP_YSIZE, 9);
072.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJPROP_BGCOLOR, clrLightSkyBlue);
073.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJPROP_BORDER_COLOR, clrBlack);
074.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJPROP_WIDTH, 3);
075.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSlider, OBJPROP_BORDER_TYPE, BORDER_FLAT);
076. //---
077.                            ObjectCreate(def_InfoTerminal.ID, m_Slider.szBarSliderBlock, OBJ_RECTANGLE_LABEL, 0, 0, 0);
078.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSliderBlock, OBJPROP_XDISTANCE, def_PosXObjects + x);
079.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSliderBlock, OBJPROP_YDISTANCE, m_Slider.posY - 9);
080.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSliderBlock, OBJPROP_YSIZE, 19);
081.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSliderBlock, OBJPROP_BGCOLOR, clrRosyBrown);
082.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSliderBlock, OBJPROP_BORDER_TYPE, BORDER_RAISED);
083.                    }
084. //+------------------------------------------------------------------+
085.            void CreateBtnPlayPause(bool state)
086.                    {
087.                            m_szBtnPlay = def_PrefixObjectName + "Play";
088.                            CreateObjectBitMap(0, 25, m_szBtnPlay, (m_bWait ? def_ButtonWait : def_ButtonPause), (m_bWait ? def_ButtonWait : def_ButtonPlay));
089.                            ObjectSetInteger(def_InfoTerminal.ID, m_szBtnPlay, OBJPROP_STATE, state);
090.                    }
091. //+------------------------------------------------------------------+
092.            void CreteCtrlSlider(void)
093.                    {
094.                            u_Interprocess Info;
095.
096.                            m_Slider.szBarSlider      = def_NameObjectsSlider + " Bar";
097.                            m_Slider.szBarSliderBlock = def_NameObjectsSlider + " Bar Block";
098.                            m_Slider.szBtnLeft        = def_NameObjectsSlider + " BtnL";
099.                            m_Slider.szBtnRight       = def_NameObjectsSlider + " BtnR";
100.                            m_Slider.szBtnPin         = def_NameObjectsSlider + " BtnP";
101.                            m_Slider.posY = 40;
102.                            CreteBarSlider(77, 436);
103.                            CreateObjectBitMap(47, 25, m_Slider.szBtnLeft, def_ButtonLeft, def_ButtonLeftBlock);
104.                            CreateObjectBitMap(511, 25, m_Slider.szBtnRight, def_ButtonRight, def_ButtonRightBlock);
105.                            CreateObjectBitMap(0, m_Slider.posY, m_Slider.szBtnPin, def_ButtonPin);
106.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBtnPin, OBJPROP_ANCHOR, ANCHOR_CENTER);
107.                            if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.df_Value = 0;
108.                            m_Slider.Minimal = Info.s_Infos.iPosShift;
109.                            PositionPinSlider(Info.s_Infos.iPosShift);
110.                    }
111. //+------------------------------------------------------------------+
112. inline void RemoveCtrlSlider(void)
113.                    {
114.                            ChartSetInteger(def_InfoTerminal.ID, CHART_EVENT_OBJECT_DELETE, false);
115.                            ObjectsDeleteAll(def_InfoTerminal.ID, def_NameObjectsSlider);
116.                            ChartSetInteger(def_InfoTerminal.ID, CHART_EVENT_OBJECT_DELETE, true);
117.                    }
118. //+------------------------------------------------------------------+
119. inline void PositionPinSlider(int p, const int minimal = 0)
120.                    {
121.                            m_Slider.posPinSlider = (p < minimal ? minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
122.                            m_Slider.posPinSlider = (p < m_Slider.Minimal ? m_Slider.Minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
123.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBtnPin, OBJPROP_XDISTANCE, m_Slider.posPinSlider + def_PosXObjects + 95);
124.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != minimal);
125.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != m_Slider.Minimal);
126.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBtnRight, OBJPROP_STATE, m_Slider.posPinSlider < def_MaxPosSlider);
127.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, minimal + 2);
128.                            ObjectSetInteger(def_InfoTerminal.ID, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, m_Slider.Minimal + 2);
129.                            ChartRedraw();
130.                    }
131. //+------------------------------------------------------------------+
132.    public  :
133. //+------------------------------------------------------------------+
134.            C_Controls(C_Terminal *arg)
135.                    :C_Mouse(arg),
136.                     m_bWait(false)
137.                    {
138.                            if (CheckPointer(Terminal = arg) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
139.                            m_szBtnPlay          = NULL;
140.                            m_Slider.szBarSlider = NULL;
141.                            m_Slider.szBtnPin    = NULL;
142.                            m_Slider.szBtnLeft   = NULL;
143.                            m_Slider.szBtnRight  = NULL;
144.                    }
145. //+------------------------------------------------------------------+
146.            ~C_Controls()
147.                    {
148.                            if (CheckPointer(Terminal) == POINTER_INVALID) return;
149.                            ChartSetInteger(def_InfoTerminal.ID, CHART_EVENT_OBJECT_DELETE, false);
150.                            ObjectsDeleteAll(def_InfoTerminal.ID, def_PrefixObjectName);
151.                    }
152. //+------------------------------------------------------------------+
153.            void Init(const bool state)
154.                    {
155.                            CreateBtnPlayPause(state);
156.                            GlobalVariableTemp(def_GlobalVariableReplay);
157.                            if (!state) CreteCtrlSlider();
158.                            ChartRedraw();
159.                    }
160. //+------------------------------------------------------------------+
161.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
162.                    {
163.                            u_Interprocess Info;
164.                            static int six = -1, sps;
165.                            int x, y, px1, px2;
166.
167.                            C_Mouse::DispatchMessage(id, lparam, dparam, sparam);
168.                            switch (id)
169.                            {
170.                                    case (CHARTEVENT_CUSTOM + C_Controls::ev_WaitOn):
171.                                            if (lparam == 0) break;
172.                                            m_bWait = true;
173.                                            CreateBtnPlayPause(true);
174.                                            break;
175.                                    case (CHARTEVENT_CUSTOM + C_Controls::ev_WaitOff):
176.                                            if (lparam == 0) break;
177.                                            m_bWait = false;
178.                                            Info.df_Value = dparam;
179.                                            CreateBtnPlayPause(Info.s_Infos.isPlay);
180.                                            break;
181.                                    case CHARTEVENT_OBJECT_DELETE:
182.                                            if (StringSubstr(sparam, 0, StringLen(def_PrefixObjectName)) == def_PrefixObjectName)
183.                                            {
184.                                                    if (StringSubstr(sparam, 0, StringLen(def_NameObjectsSlider)) == def_NameObjectsSlider)
185.                                                    {
186.                                                            RemoveCtrlSlider();
187.                                                            CreteCtrlSlider();
188.                                                    }else
189.                                                    {
190.                                                            Info.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
191.                                                            CreateBtnPlayPause(Info.s_Infos.isPlay);
192.                                                    }
193.                                                    ChartRedraw();
194.                                            }
195.                                            break;
196.                                    case CHARTEVENT_OBJECT_CLICK:
197.                                            if (m_bWait) break;
198.                                            if (sparam == m_szBtnPlay)
199.                                            {
200.                                                    Info.s_Infos.isPlay = (bool) ObjectGetInteger(def_InfoTerminal.ID, m_szBtnPlay, OBJPROP_STATE);
201.                                                    if (!Info.s_Infos.isPlay) CreteCtrlSlider(); else
202.                                                    {
203.                                                            RemoveCtrlSlider();
204.                                                            m_Slider.szBtnPin = NULL;
205.                                                    }
206.                                                    Info.s_Infos.iPosShift = (ushort) m_Slider.posPinSlider;
207.                                                    GlobalVariableSet(def_GlobalVariableReplay, Info.df_Value);
208.                                                    ChartRedraw();
209.                                            }else   if (sparam == m_Slider.szBtnLeft) PositionPinSlider(m_Slider.posPinSlider - 1);
210.                                            else if (sparam == m_Slider.szBtnRight) PositionPinSlider(m_Slider.posPinSlider + 1);
211.                                            break;
212.                                    case CHARTEVENT_MOUSE_MOVE:
213.                                            if (GetInfoMouse().ExecStudy) return;
214.                                            if ((CheckClick(C_Mouse::eClickLeft)) && (m_Slider.szBtnPin != NULL))
215.                                            {
216.                                                    x = GetInfoMouse().Position.X;
217.                                                    y = GetInfoMouse().Position.Y;
218.                                                    px1 = m_Slider.posPinSlider + def_PosXObjects + 86;
219.                                                    px2 = m_Slider.posPinSlider + def_PosXObjects + 114;
220.                                                    if ((y >= (m_Slider.posY - 14)) && (y <= (m_Slider.posY + 14)) && (x >= px1) && (x <= px2) && (six == -1))
221.                                                    {
222.                                                            six = x;
223.                                                            sps = m_Slider.posPinSlider;
224.                                                            ChartSetInteger(def_InfoTerminal.ID, CHART_MOUSE_SCROLL, false);
225.                                                    }
226.                                                    if (six > 0) PositionPinSlider(sps + x - six);
227.                                            }else if (six > 0)
228.                                            {
229.                                                    six = -1;
230.                                                    ChartSetInteger(def_InfoTerminal.ID, CHART_MOUSE_SCROLL, true);
231.                                            }
232.                                            break;
233.                            }
234.                    }
235. //+------------------------------------------------------------------+
236. };
237. //+------------------------------------------------------------------+
238. #undef def_InfoTerminal
239. #undef def_AcessTerminal
240. #undef def_PosXObjects
241. #undef def_ButtonPlay
242. #undef def_ButtonPause
243. #undef def_ButtonLeft
244. #undef def_ButtonRight
245. #undef def_ButtonPin
246. #undef def_NameObjectsSlider
247. #undef def_PrefixObjectName
248. #undef def_PathBMP
249. //+------------------------------------------------------------------+
```

Source code of the C\_Controls.mqh file

After making these fixes, changes and adjustments, we can try to compile the replay/simulator service again. And as a result, we get the following:

![Figure 10](https://c.mql5.com/2/50/010.png)

Figure 10. Final compilation

This means that the service was compiled successfully, and note that the indicator was also compiled. This in turn means that you can now simply delete the indicator executable using file explorer, as it will now become part of the replay/simulator service executable. However, if you delete the indicator executable file and try to run the replay/simulator system in MetaTrader 5, you will see that the chart opens but the control indicator does not appear. Why?

The reason is that the control indicator is actually triggered by the template, not the service. There is nothing in the template that indicates that the control indicator is part of the service executable. You can notice this by looking at the contents of the template file. But since the idea is not to use a template but to use a service to run a control indicator on the chart, I will not go into details of how to solve this problem. Let's focus on what we really want to do.

Since what we need to do is not that simple and I don't want to go into too much detail in this article, I'll leave the changes for the next one. This is because we will need to change many things in order to keep the control indicator functional, while at the same time keeping the replay/simulator service free, lightweight and open so that you can use your indicators, strategies or personal models. The idea behind making these changes is precisely to encourage this. To allow you to use the system with your own concepts and ideas.

But the main reason we are leaving the demonstration of everything we need to do to make the control indicator appear on the chart, being a service resource and without using a template, for the next article is the fact that we will have to remove some of the things that are duplicated now. Such duplication makes the control indicator extremely unstable if used directly through the service.

### Conclusion

Although in this article we have made changes to the control indicator and service to the point that we can remove the indicator executable file from the list of those available to the user, the system itself has proven to be unstable due to unresolved issues. These issues concern the way the control indicator and the user interact. This instability is caused by the need for other elements that need to be present on the chart. These are elements that are part of the template file that I don't want to use when using the relay/simulator service. I want and am going to show how we can do this to make the service self-sufficient so that the user can use their own settings or templates.

At the same time, we create a suitable way to promote the replay/simulator service so that you can practice your analysis strategy or model.

We still have a lot to do before this system can get the long-awaited functionality for working with orders and positions in the replay and simulator mode. I hope, dear reader, that you understand the level and degree of complexity we have reached. And all this is done only with the help of MQL5, without external programming. Indeed, this system turned out to be quite complex. But I like challenges, and this one continues to inspire me.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11820](https://www.mql5.com/pt/articles/11820)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11820.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11820/anexo.zip "Download Anexo.zip")(420.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/475191)**
(1)


![Sergei Lebedev](https://c.mql5.com/avatar/2019/7/5D3E1392-839A.jpg)

**[Sergei Lebedev](https://www.mql5.com/en/users/salebedev)**
\|
11 May 2024 at 12:10

Thank you a lot for such interesting project! However reading all 49 articles to get the clue for every component is really hard work for any traders.

I kindly ask you to write next one #50 article under the name "User Guide" and lay out in it on each component of this [project](https://www.mql5.com/en/articles/7863 "Article: Projects let you create profitable trading robots! But it's not exactly") \- Replay System, EA and indicators.

It will be great to add some practical examples of using it, like following:

\- "Replaying and re-trading Brexit case on GBPUSD in June 2016 for educational purpose";

\- "Replaying and re-trading Gold-rush case on XAUUSD in March 2020 for educational purpose";

It is expected that in these practical example should be used to retrieve historical data from real symbols 'GBPUSD'/'XAUUSD' (from any connected forex account), and to feed custom symbols like 'repGBPUSD'/'repXAUUSd' using extracted data in replay mode, to add some generic indicators (RSI(14), MA(50) etc) and to provide users with real-time experience of re-trading these historical events.

Such a User Guide with practical examples of real time re-trading Brexit and Gold-rush will be really great finalisation of this project!

![Feature selection and dimensionality reduction using principal components](https://c.mql5.com/2/98/Feature_selection_and_dimensionality_reduction_using_principal_components____LOGO.png)[Feature selection and dimensionality reduction using principal components](https://www.mql5.com/en/articles/16190)

The article delves into the implementation of a modified Forward Selection Component Analysis algorithm, drawing inspiration from the research presented in “Forward Selection Component Analysis: Algorithms and Applications” by Luca Puggini and Sean McLoone.

![Neural networks made easy (Part 89): Frequency Enhanced Decomposition Transformer (FEDformer)](https://c.mql5.com/2/77/Neural_networks_are_easy_cPart_89q___LOGO.png)[Neural networks made easy (Part 89): Frequency Enhanced Decomposition Transformer (FEDformer)](https://www.mql5.com/en/articles/14858)

All the models we have considered so far analyze the state of the environment as a time sequence. However, the time series can also be represented in the form of frequency features. In this article, I introduce you to an algorithm that uses frequency components of a time sequence to predict future states.

![Creating a Trading Administrator Panel in MQL5 (Part V): Two-Factor Authentication (2FA)](https://c.mql5.com/2/99/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_V__LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part V): Two-Factor Authentication (2FA)](https://www.mql5.com/en/articles/16142)

Today, we will discuss enhancing security for the Trading Administrator Panel currently under development. We will explore how to implement MQL5 in a new security strategy, integrating the Telegram API for two-factor authentication (2FA). This discussion will provide valuable insights into the application of MQL5 in reinforcing security measures. Additionally, we will examine the MathRand function, focusing on its functionality and how it can be effectively utilized within our security framework. Continue reading to discover more!

![How to Create an Interactive MQL5 Dashboard/Panel Using the Controls Class (Part 2): Adding Button Responsiveness](https://c.mql5.com/2/98/How_to_Create_an_Interactive_MQL5_Dashboard___LOGO__1.png)[How to Create an Interactive MQL5 Dashboard/Panel Using the Controls Class (Part 2): Adding Button Responsiveness](https://www.mql5.com/en/articles/16146)

In this article, we focus on transforming our static MQL5 dashboard panel into an interactive tool by enabling button responsiveness. We explore how to automate the functionality of the GUI components, ensuring they react appropriately to user clicks. By the end of the article, we establish a dynamic interface that enhances user engagement and trading experience.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/11820&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070033143215361752)

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