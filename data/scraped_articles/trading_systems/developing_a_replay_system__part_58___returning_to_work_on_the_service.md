---
title: Developing a Replay System (Part 58): Returning to Work on the Service
url: https://www.mql5.com/en/articles/12039
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:41:57.707751
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=utpkqhgjqednenleuzglggkwdbtiufsl&ssn=1769182916673457078&ssn_dr=0&ssn_sr=0&fv_date=1769182916&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12039&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2058)%3A%20Returning%20to%20Work%20on%20the%20Service%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918291634539722&fz_uniq=5069666791094945987&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 57): Breaking Down the Testing Service](https://www.mql5.com/en/articles/12005), I have explained in detail the source code needed to demonstrate a possible way of interaction between the modules we will use in our replay/simulator system.

While this code gives us an idea of what we will actually need to implement, it still lacks an important detail that could be really useful for our system - the ability to use templates. You might think that this is not that important if you do not use or do not fully understand the benefits that templates give us, both in terms of coding and MetaTrader 5 settings.

However, knowing, understanding and applying patterns significantly reduces our workload. There are things that are very easy to do with templates, but become extremely complex and difficult to implement if you try to program them directly. Maybe in the future I'll show you how to do some things using only templates, but for now we have other, more pressing tasks.

To be honest, I thought I had achieved the point where the control and mouse modules did not need any improvement. However, due to some details that we will see in the following articles, both modules will still have to undergo minor changes. We will see this later, but for now, in this article, we will figure out how to turn the knowledge gained in the previous article into something feasible and functional. To do this, let's move on to a new topic.

### Modifying the old replay/simulator service

Although it has been some time since we last made any modifications or improvements to the replay/simulator code, certain header files involved in building the replay/simulator executable have undergone changes. Perhaps the most notable change is the removal of the InterProcess.mqh header file, which has been replaced by Defines.mqh, a file with a much broader purpose.

Since we have already made adjustments to the control and mouse modules to accommodate this new header file, we must now apply the same changes to the replay/simulation service. As a result, attempting to compile the replay/simulation service with the updated header file structure will lead to compilation errors, as illustrated in Figure 01.

![Figure 01](https://c.mql5.com/2/110/01__1.png)

Figure 01. Attempt to compile replication/modeling service

Among the various errors that may appear, you should first address the two highlighted ones. To resolve them, open the C\_Simulation.mqh header file and modify the code as shown in the snippet below. The required change is minimal – simply remove line 04 and replace it with the adjustment shown in line 05. This modification ensures that C\_Simulation.mqh conforms to the new framework we are implementing.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #include "..\..\Auxiliar\Interprocess.mqh"
05. #include "..\..\Defines.mqh"
06. //+------------------------------------------------------------------+
07. #define def_MaxSizeArray    16777216 // 16 Mbytes of positions
08. //+------------------------------------------------------------------+
09. class C_Simulation
10. {
11.    private   :
12. //+------------------------------------------------------------------+
13.       int       m_NDigits;
14.       bool       m_IsPriceBID;
```

A fragment of the source code of the C\_Simulation.mqh file

Just like we did in the C\_Simulation.mqh header file, we will need to do something similar in the C\_FilesBars.mqh file. To do this, open the C\_FilesBars.mqh header file and change the code as shown below.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #include "..\..\Auxiliar\Interprocess.mqh"
05. #include "..\..\Defines.mqh"
06. //+------------------------------------------------------------------+
07. #define def_BarsDiary   1440
08. //+------------------------------------------------------------------+
09. class C_FileBars
10. {
11.    private   :
12.       int      m_file;
```

A fragment of the source code of the C\_FilesBars.mqh file

In both code fragments, we have removed the InterProcess.mqh header file and replaced it with Defines.mqh. With these two modifications, most of the code will align with the expected structure of the replay/simulator service. However, there is an issue. If you compare the contents of InterProcess.mqh and Defines.mqh, you will notice that Defines.mqh does not reference terminal global variables. Despite this, the replay/simulator system still refers to these variables.

More specifically, these variables are used within the C\_Replay.mqh file. However, this is not our only concern. In the future, I may decide to restructure the code further to improve its organization, stability, and flexibility. For now, however, I will focus on adapting the existing structure rather than making drastic changes to the entire system just for a minor improvement in flexibility and stability – although both are always worth enhancing.

To keep things clear, let's break this explanation into sections. The first issue we will address is a flaw that, while not critical, violates one of the core principles of object-oriented programming: **encapsulation**.

### Reviewing Code Encapsulation

One of the most serious issues in any codebase is failing to adhere to fundamental object-oriented programming principles that ensure security and maintainability. For a long time, I have overlooked and misused a specific part of the code to facilitate direct access to certain data required for replay/simulator functionality.

However, from this point forward, this practice will no longer be used. Specifically, I am referring to the encapsulation breach present in the C\_ConfigService class.

If you examine the header file for this class (C\_ConfigService.mqh), you will notice a protected clause containing several variables. The existence of these variables in this section breaks encapsulation, even though they are only used within C\_ConfigService and its derived class, C\_Replay. It is not appropriate for these variables to be accessible outside C\_ConfigService in their current form. If you review the C\_Replay class, you will see that it modifies these variables, which is precisely what makes this approach problematic. In C++, there are ways to make class variables private while still allowing controlled access and manipulation outside the base class. However, these techniques often result in overly complex and difficult-to-maintain code. Additionally, they make future improvements significantly more challenging.

Since MQL5 is derived from C++, it avoids incorporating certain risky practices that C++ allows. Therefore, it is more appropriate to adhere strictly to the three fundamental principles of object-oriented programming, including proper encapsulation.

By modifying the C\_ConfigService.mqh header file, we will restore proper encapsulation within our system. However, this change will require adjustments at higher levels of the codebase. Specifically, the C\_Replay class, located in the C\_Replay.mqh file, will undergo significant modifications. At the same time, we will take this opportunity to improve the code structure, making the replay/simulator service less nested. By implementing smaller, incremental changes, we can simplify maintenance and improve control over what is happening at each step. This will be particularly beneficial for future updates, as we will soon need to implement even more complex functionality that involves multiple interconnected components.

Let's see what needs to be done to make things more suitable. To begin improving encapsulation, open the C\_ConfigService.mqh header file and modify the code as shown in the following fragment. The rest of the code will remain unchanged, but the changes in this fragment will ensure that encapsulation is properly enforced.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #include "Support\C_FileBars.mqh"
05. #include "Support\C_FileTicks.mqh"
06. #include "Support\C_Array.mqh"
07. //+------------------------------------------------------------------+
08. class C_ConfigService : protected C_FileTicks
09. {
10.    protected:
11.         datetime m_dtPrevLoading;
12.         int      m_ReplayCount,
13.                  m_ModelLoading;
14. //+------------------------------------------------------------------+
15. inline void FirstBarNULL(void)
16.          {
17.             MqlRates rate[1];
18.             int c0 = 0;
19.
20.             for(; (m_Ticks.ModePlot == PRICE_EXCHANGE) && (m_Ticks.Info[c0].volume_real == 0); c0++);
21.             rate[0].close = (m_Ticks.ModePlot == PRICE_EXCHANGE ? m_Ticks.Info[c0].last : m_Ticks.Info[c0].bid);
22.             rate[0].open = rate[0].high = rate[0].low = rate[0].close;
23.             rate[0].tick_volume = 0;
24.             rate[0].real_volume = 0;
25.             rate[0].time = macroRemoveSec(m_Ticks.Info[c0].time) - 86400;
26.             CustomRatesUpdate(def_SymbolReplay, rate);
27.             m_ReplayCount = 0;
28.          }
29. //+------------------------------------------------------------------+
30.    private   :
31.       enum eWhatExec {eTickReplay, eBarToTick, eTickToBar, eBarPrev};
32.       enum eTranscriptionDefine {Transcription_INFO, Transcription_DEFINE};
33.       struct st001
34.       {
35.          C_Array *pTicksToReplay, *pBarsToTicks, *pTicksToBars, *pBarsToPrev;
36.          int      Line;
37.       }m_GlPrivate;
38.       string    m_szPath;
39.       bool      m_AccountHedging;
40.       datetime  m_dtPrevLoading;
41.       int       m_ReplayCount,
42.                 m_ModelLoading;
43. //+------------------------------------------------------------------+
44. inline void FirstBarNULL(void)
45.          {
46.             MqlRates rate[1];
47.             int c0 = 0;
48.
49.             for(; (m_Ticks.ModePlot == PRICE_EXCHANGE) && (m_Ticks.Info[c0].volume_real == 0); c0++);
50.             rate[0].close = (m_Ticks.ModePlot == PRICE_EXCHANGE ? m_Ticks.Info[c0].last : m_Ticks.Info[c0].bid);
51.             rate[0].open = rate[0].high = rate[0].low = rate[0].close;
52.             rate[0].tick_volume = 0;
53.             rate[0].real_volume = 0;
54.             rate[0].time = macroRemoveSec(m_Ticks.Info[c0].time) - 86400;
55.             CustomRatesUpdate(def_SymbolReplay, rate);
56.             m_ReplayCount = 0;
57.          }
58. //+------------------------------------------------------------------+
59. inline eTranscriptionDefine GetDefinition(const string &In, string &Out)
```

A fragment of the source code of the C\_ConfigService.mqh file

Note that the contents of lines 11–13 have been moved to lines 40 and 42. This means that it will now be impossible to access these variables outside the body of the C\_ConfigService class. Besides this, one more change was made. This change could have been ignored, but since some things won't be used outside the class, I decided to make the FirstBarNULL procedure private. So the content that was between lines 15 and 28 has been moved to lines 44 to 57.

It is clear that when you make these changes to the actual file, the line numbers will be different because the removed code will no longer be part of the class code. However, I decided to leave everything in the fragment as is, for clarity. I think this way it will be clearer and easier to understand what has been changed.

Great. Now, after making these changes, we will have to radically change the code present in the C\_Replay.mqh file. But let's continue to separate things one from the other and look at this in the next topic.

### Restarting the implementation of the C\_Replay class

Although the title of this section may seem discouraging, implying that we are reinventing something that was already built, this is not the case. I want to emphasize that, while we do need to rework a large portion of the C\_Replay class, the knowledge gained throughout this series of articles remains valuable. What we are doing is adapting to a new structure and methodology, as certain things can no longer be implemented the way they were before.

The complete revised code for the C\_Replay class is provided below.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "C_ConfigService.mqh"
005. //+------------------------------------------------------------------+
006. #define def_IndicatorControl   "Indicators\\Market Replay.ex5"
007. #resource "\\" + def_IndicatorControl
008. //+------------------------------------------------------------------+
009. #define def_CheckLoopService ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""))
010. //+------------------------------------------------------------------+
011. #define def_ShortNameIndControl "Market Replay Control"
012. //+------------------------------------------------------------------+
013. class C_Replay : public C_ConfigService
014. {
015.    private   :
016.       long      m_IdReplay;
017.       struct st00
018.       {
019.          ushort Position;
020.          short  Mode;
021.       }m_IndControl;
022. //+------------------------------------------------------------------+
023. inline bool MsgError(string sz0) { Print(sz0); return false; }
024. //+------------------------------------------------------------------+
025. inline void UpdateIndicatorControl(void)
026.          {
027.             uCast_Double info;
028.             int handle;
029.             double Buff[];
030.
031.             if ((handle = ChartIndicatorGet(m_IdReplay, 0, def_ShortNameIndControl)) == INVALID_HANDLE) return;
032.             info.dValue = 0;
033.             if (CopyBuffer(handle, 0, 0, 1, Buff) == 1)
034.                info.dValue = Buff[0];
035.             IndicatorRelease(handle);
036.             if ((short)(info._16b[0]) != SHORT_MIN)
037.                m_IndControl.Mode = (short)info._16b[1];
038.             if (info._16b[0] != m_IndControl.Position)
039.             {
040.                if (((short)(info._16b[0]) != SHORT_MIN) && ((short)(info._16b[1]) == SHORT_MAX))
041.                   m_IndControl.Position = info._16b[0];
042.                info._16b[0] = m_IndControl.Position;
043.                info._16b[1] = (ushort)m_IndControl.Mode;
044.                EventChartCustom(m_IdReplay, evCtrlReplayInit, 0, info.dValue, "");
045.             }
046.          }
047. //+------------------------------------------------------------------+
048.       void SweepAndCloseChart(void)
049.          {
050.             long id;
051.
052.             if ((id = ChartFirst()) > 0) do
053.             {
054.                if (ChartSymbol(id) == def_SymbolReplay)
055.                   ChartClose(id);
056.             }while ((id = ChartNext(id)) > 0);
057.          }
058. //+------------------------------------------------------------------+
059.    public   :
060. //+------------------------------------------------------------------+
061.       C_Replay()
062.          :C_ConfigService()
063.          {
064.             Print("************** Market Replay Service **************");
065.             srand(GetTickCount());
066.             SymbolSelect(def_SymbolReplay, false);
067.             CustomSymbolDelete(def_SymbolReplay);
068.             CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay));
069.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0);
070.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, 0);
071.             CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, 0);
072.             CustomSymbolSetString(def_SymbolReplay, SYMBOL_DESCRIPTION, "Symbol for replay / simulation");
073.             CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, 8);
074.             SymbolSelect(def_SymbolReplay, true);
075.          }
076. //+------------------------------------------------------------------+
077.       bool OpenChartReplay(const ENUM_TIMEFRAMES arg1, const string szNameTemplate)
078.          {
079.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
080.                return MsgError("Asset configuration is not complete, it remains to declare the size of the ticket.");
081.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
082.                return MsgError("Asset configuration is not complete, need to declare the ticket value.");
083.             if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
084.                return MsgError("Asset configuration not complete, need to declare the minimum volume.");
085.             SweepAndCloseChart();
086.             m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
087.             if (!ChartApplyTemplate(m_IdReplay, szNameTemplate + ".tpl"))
088.                Print("Failed apply template: ", szNameTemplate, ".tpl Using template default.tpl");
089.             else
090.                Print("Apply template: ", szNameTemplate, ".tpl");
091.
092.             return true;
093.          }
094. //+------------------------------------------------------------------+
095.       bool InitBaseControl(const ushort wait = 1000)
096.          {
097.             int handle;
098.
099.             Print("Waiting for Mouse Indicator...");
100.             Sleep(wait);
101.             while ((def_CheckLoopService) && (ChartIndicatorGet(m_IdReplay, 0, "Indicator Mouse Study") == INVALID_HANDLE)) Sleep(200);
102.             if (def_CheckLoopService)
103.             {
104.                Print("Waiting for Control Indicator...");
105.                if ((handle = iCustom(ChartSymbol(m_IdReplay), ChartPeriod(m_IdReplay), "::" + def_IndicatorControl, m_IdReplay)) == INVALID_HANDLE) return false;
106.                ChartIndicatorAdd(m_IdReplay, 0, handle);
107.                IndicatorRelease(handle);
108.                m_IndControl.Position = 0;
109.                m_IndControl.Mode = SHORT_MIN;
110.                UpdateIndicatorControl();
111.             }
112.
113.             return def_CheckLoopService;
114.          }
115. //+------------------------------------------------------------------+
116.       bool LoopEventOnTime(void)
117.          {
118.
119.             while (def_CheckLoopService)
120.             {
121.                UpdateIndicatorControl();
122.                Sleep(250);
123.             }
124.
125.             return false;
126.          }
127. //+------------------------------------------------------------------+
128.       ~C_Replay()
129.          {
130.             SweepAndCloseChart();
131.             SymbolSelect(def_SymbolReplay, false);
132.             CustomSymbolDelete(def_SymbolReplay);
133.             Print("Finished replay service...");
134.          }
135. //+------------------------------------------------------------------+
```

Source code of the C\_Replay.mqh file

Although this code does not yet perform replay/simulation as it did previously, since certain components are still missing, its purpose is to enable the replay/simulation service to utilize elements not covered in the previous article. Among these elements is the ability to load previous bars, just as before, as well as the bars required for both replay and simulation. However, in this article, we will not yet be able to fully utilize these replay or simulation bars. Instead, they will be loaded and made available for when the system is capable of properly displaying them on the custom asset chart.

There are several aspects of the code above that warrant further explanation. Many of its components may not be immediately clear, even to those with solid experience in MQL5. However, the explanations provided here will be aimed at those who genuinely want to understand why this code is being structured in this particular way.

At the beginning of the code, in lines 5 to 11, we define certain parameters and include the compiled indicator file within the service executable. The reasoning behind this has been extensively discussed in previous articles in this series on replay/simulation. Therefore, I simply highlight this to remind you that it is not necessary to manually transfer the control indicator file.

Then, in line 13, we establish a public inheritance from the C\_ConfigService class. This is done to ensure that the workload is not concentrated solely in the C\_Replay class; rather, it is distributed between C\_Replay and C\_ConfigService. This reinforces the importance of the changes made in the previous section, where we discussed the necessary modifications to properly encapsulate data and variables.

The private section of the C\_Replay class begins on line 15 and extends until line 58, where the public section begins. Let's first examine how the private section functions. It includes a small set of global variables, declared between in 16 to 21. Pay particular attention to line 21, where a variable is declared as a structure, meaning it contains additional nested data.

In line 23, we define a small function whose sole purpose is to print an error message to the terminal and return false. But why return false here? Without this return value, we would need an additional line of code every time we print an error message to the terminal. For clarity, look at line 79, where we check a certain condition. If an error is detected, we would typically need two separate lines: one to print the error message and another to return an error indication. This would create unnecessary redundancy. However, by using the function declared in line 23, we can print the message and return a failure indication in a single step. This seen on line 80, simplifying the implementation. We combine things in such a way as to reduce our coding work.

Perhaps the most important section of the code is between lines 25 and 46. This code does some very important work for us. It manages and adjusts data from the control indicator. Before attempting to understand this section, ensure you fully comprehend how all related components interact. If in doubt, refer to previous articles explaining how the control indicator communicates with external components.

Line 31 attempts to capture a handle for accessing the control indicator. If this fails, it is not a critical error. The function simply returns, skipping the rest of the procedure. If a handle is successfully captured, we reset the testing value, as shown in line 32. This is crucial and must be done correctly. Line 33 checks whether the indicator buffer is readable. If so, line 34 assigns the value to a test and adjustment variable. This section might undergo minor refinements in future articles, but the core logic will remain the same.

Once the handle is no longer needed, line 35 releases it, and we enter the phase of testing and adjusting the retrieved information. Line 36 checks if the control indicator contains valid data. If so, we save information on whether the system is in paused mode or active play mode (replay/simulation). This saving is performed in line 37. This must be done before any modifications occur; otherwise, the retrieved data might be altered prematurely, compromising the integrity of information. The goal here is to ensure the service provides the latest form of control indicator – something that was previously done using a global terminal variable.

Now pay attention to line 38. It compares the indicator buffer contents with the global positioning system. If a discrepancy is found, line 40 performs a secondary check to see if the control indicator has been initialized and if the system is in play mode. If both conditions are met, line 41 saves the buffer value. This is critical because, during pause mode, we do not want to update the data automatically. We want to allow the user to manually adjust the control indicator as needed.

Finally, in lines 42 and 43, we assemble the information to be passed to the control indicator. This is transmitted via a custom event triggered in line 44. Once triggered, MetaTrader 5 takes over, executing its tasks while the service continues running in parallel.

The code present in this procedure should be analyzed very carefully until it is really clear what is going on. Compared to the approach from the previous article, this version is more complex, despite performing essentially the same function. Once the control indicator is placed on the chart by MetaTrader 5, this code initializes it. From then on, it monitors its state. If the user changes the time frame, the service preserves the last known state of the indicator, ensuring that it is reinitialized with its previous settings upon returning to the chart.

Now let's look at code that was created with reusability in mind. It is located in line 48. The procedure simply closes all MetaTrader 5 chart windows containing symbols to be replicated. As you can see, there is nothing complicated. But since we need to do this at least twice, I decided to create this procedure to avoid duplicating code.

So, from this point we move on to the public procedures of the C\_Replay class. Basically, you can see that the code is not much different from what was before, at least with regard to the class constructor and destructor. Therefore, I will not make any further comments about them, since they have already been properly covered in previous articles where I explained the functioning of the C\_Replay class. There are, however, three functions here that do deserve some explanation. Let's look at them in the order they appear in the code.

The first function is called OpenChartReplay, which starts in line 77 and ends in line 93. It checks the integrity of the information collected by the boot system. This is necessary so that replay or simulator can actually be performed. However, it is in this function that we find something quite complex that, together with the InitBaseControl function, which we will talk about later, allows us to use the template.

The issue of using a template is of great importance to us. It is necessary that it is used correctly and launched in the appropriate manner. But doing this is not as easy as many, including me, might have thought at first. In line 87 we try to add the template to the chart after previously opening it in line 86. The template to use is specified as one of the function arguments. In any case, the template will be placed on the chart, whether it is a user-specified template or a standard MetaTrader 5 template. But there is a detail here that is rarely mentioned: the template is not placed immediately. The ChartApplyTemplate function does not apply the template immediately. This function is asynchronous, meaning it can be executed within a few milliseconds of being called. And this is a problem for us.

To understand the scale of the problem, we'll take a short break from the C\_Replay class and look at the service code below.

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property icon "/Images/Market Replay/Icons/Replay - Device.ico"
04. #property copyright "Daniel Jose"
05. #property version   "1.58"
06. #property description "Replay-Simulator service for MetaTrade 5 platform."
07. #property description "This is dependent on the Market Replay indicator."
08. #property description "For more details on this version see the article."
09. #property link "https://www.mql5.com/pt/articles/"
10. //+------------------------------------------------------------------+
11. #include <Market Replay\Service Graphics\C_Replay.mqh>
12. //+------------------------------------------------------------------+
13. input string            user00 = "Mini Dolar.txt";   //Replay Configuration File.
14. input ENUM_TIMEFRAMES   user01 = PERIOD_M5;          //Initial Graphic Time.
15. input string            user02 = "Default";          //Template File Name
16. //+------------------------------------------------------------------+
17. C_Replay *pReplay;
18. //+------------------------------------------------------------------+
19. void OnStart()
20. {
21.    pReplay = new C_Replay();
22.
23.    UsingReplay();
24.
25.    delete pReplay;
26. }
27. //+------------------------------------------------------------------+
28. void UsingReplay(void)
29. {
30.    if (!(*pReplay).SetSymbolReplay(user00)) return;
31.    if (!(*pReplay).OpenChartReplay(user01, user02)) return;
32.    if (!(*pReplay).InitBaseControl()) return;
33.    Print("Permission granted. Replay service can now be used...");
34.    while ((*pReplay).LoopEventOnTime());
35. }
36. //+------------------------------------------------------------------+
```

Source code of the replay/simulation service

Notice that we perform tasks in a specific sequence, as seen between lines 30 and 34. After initializing via the constructor in line 21, we proceed to line 30 to verify that everything is correct with the loading process. Then, in line 31, we attempt to open the chart, and only after that, in line 32, do we load the necessary elements to control the service. If everything goes smoothly, in line 33, we print a message to the terminal, and in line 34, we enter the execution loop.

At a glance, it seems like nothing unusual happens between opening the chart in line 31 and adding the controls in line 32. However, due to the use of the template loaded in the C\_Replay class, some unforeseen issues may arise. To better understand the potential problem, let's revisit the class to examine the real complication of using a template.

After instructing MetaTrader 5 to apply a template, as seen in line 87 of the C\_Replay class, the code can execute much faster than it ideally should. As a result, in line 99, we inform the user that the service is waiting for the mouse indicator. If the mouse indicator is present in the template, it will load automatically; if not, the user will need to add it manually.

This presents a problem because the function responsible for applying the template runs asynchronously. To mitigate the potential issues, we use line 100, where we pause the service for a short time to allow the chart to stabilize and the template application function to properly execute. Only after this wait do we verify on line 101 if the mouse indicator is present. This loop will continue until the mouse indicator appears on the chart or the chart is closed by the user.

Once the mouse indicator is detected or the chart is closed, the code continues. If everything is as expected, we try to add the control indicator to the chart on line 105. While this works beautifully, there is an important detail: the control indicator will not be accepted if it is already part of the template. This is one of the modifications I'll show later, which prevents the control indicator from appearing in the template. A slight change will also be required for the mouse indicator, but that will come later. Without line 100, the chart would be closed shortly after opening, which is precisely what we aim to prevent.

### Conclusion

Although there is a feeling that this is not the end, it is necessary to explain in detail why the chart closes immediately after applying the template. It's quite complicated, and other things need to be shown for you to really understand how this is possible, and why simply having line 100 prevents it. Therefore, I will reserve a more detailed discussion on the template and the necessary modifications in the indicator modules for the next article. This will help you fully grasp how these changes ensure the replay/simulation service works as expected.

As you can see, this system is distinct from the testing service discussed in the previous article. Before I leave you, I will share a video showing the result of executing this system. Since it is not yet in the state shown in the video, there will be no attachment in this article.

YouTube

Demo video

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12039](https://www.mql5.com/pt/articles/12039)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12039.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12039/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/481075)**

![Feature Engineering With Python And MQL5 (Part III): Angle Of Price (2) Polar Coordinates](https://c.mql5.com/2/117/Feature_Engineering_With_Python_And_MQL5_Part_III_Angle_Of_Price_2__LOGO.png)[Feature Engineering With Python And MQL5 (Part III): Angle Of Price (2) Polar Coordinates](https://www.mql5.com/en/articles/17085)

In this article, we take our second attempt to convert the changes in price levels on any market, into a corresponding change in angle. This time around, we selected a more mathematically sophisticated approach than we selected in our first attempt, and the results we obtained suggest that our change in approach may have been the right decision. Join us today, as we discuss how we can use Polar coordinates to calculate the angle formed by changes in price levels, in a meaningful way, regardless of which market you are analyzing.

![Price Action Analysis Toolkit Development (Part 11): Heikin Ashi Signal EA](https://c.mql5.com/2/117/Price_Action_Analysis_Toolkit_Development_Part_11___LOGO__2.png)[Price Action Analysis Toolkit Development (Part 11): Heikin Ashi Signal EA](https://www.mql5.com/en/articles/17021)

MQL5 offers endless opportunities to develop automated trading systems tailored to your preferences. Did you know it can even perform complex mathematical calculations? In this article, we introduce the Japanese Heikin-Ashi technique as an automated trading strategy.

![Artificial Bee Hive Algorithm (ABHA): Tests and results](https://c.mql5.com/2/88/Artificial_Bee_Hive_Algorithm_ABHA__Final__LOGO.png)[Artificial Bee Hive Algorithm (ABHA): Tests and results](https://www.mql5.com/en/articles/15486)

In this article, we will continue exploring the Artificial Bee Hive Algorithm (ABHA) by diving into the code and considering the remaining methods. As you might remember, each bee in the model is represented as an individual agent whose behavior depends on internal and external information, as well as motivational state. We will test the algorithm on various functions and summarize the results by presenting them in the rating table.

![Custom Indicator: Plotting Partial Entry, Exit and Reversal Deals for Netting Accounts](https://c.mql5.com/2/85/Tra7ar_os_Pontos_de_Entradas_Parciais_em_contas_Netting___LOGO.png)[Custom Indicator: Plotting Partial Entry, Exit and Reversal Deals for Netting Accounts](https://www.mql5.com/en/articles/12576)

In this article, we will look at a non-standard way of creating an indicator in MQL5. Instead of focusing on a trend or chart pattern, our goal will be to manage our own positions, including partial entries and exits. We will make extensive use of dynamic matrices and some trading functions related to trade history and open positions to indicate on the chart where these trades were made.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/12039&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069666791094945987)

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