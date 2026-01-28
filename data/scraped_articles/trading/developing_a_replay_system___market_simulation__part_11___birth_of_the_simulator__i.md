---
title: Developing a Replay System — Market simulation (Part 11): Birth of the SIMULATOR (I)
url: https://www.mql5.com/en/articles/10973
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:05:34.870795
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/10973&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069066779868725308)

MetaTrader 5 / Examples


### Introduction

So far, including the previous article [Developing a Replay System — Market simulation (Part 10): Using only real data for Replay](https://www.mql5.com/en/articles/10932), everything we did involved real data, meaning we used actually traded tickets. This makes movements precise and easy to create since we don't have to worry about collecting information. All we had to do was convert traded tickets into 1-minute bars, and the MetaTrader 5 platform took care of the rest for us.

However, we now face another, more difficult task.

### Planning

Many people might think that planning is easy, especially since it involves converting bars, which should always be 1 minute long (we'll explain why later) into tickets. However, simulation is much more complex than it seems at first glance. The main problem is that we do not have a clear understanding of the actual behavior of the tickets to create a 1-minute bar. We only have the bar and some information about it, but we don't know how the bar formed. We will use 1 minute bars because they offer the least amount of difficulty. If you can create a complex movement that is very similar to the real thing, then you will be able to reproduce something very close to the real thing.

This detail may not seem that important since we usually see a zig-zag type of movement in the market. Regardless of the complexity of the movement, it all comes down to creating a zigzag between the OHCL points. It starts at the opening point of the bar and does no less than 9 movements to create this inner zigzag. It always ends at the close of the bar and repeats the process on the next bar. The MetaTrader 5 strategy tester uses the same logic. For more details see [Real and generated ticks: Algorithmic trading](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation "https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation"). We will start with this strategy. Although not ideal for our purposes, it will provide a starting point for developing more suitable approaches.

I say that the tester strategy is not the most suitable for the replay/simulation system because in a trading strategy tester, time concerns are not of paramount importance. That is, it is not necessary to create and represent a 1-minute bar in such a way that its length is actually 1 minute. In fact, it is even more convenient that it does not correspond to this time in reality. If this were the case, then testing a strategy would become impossible. Imagine running a test with bars spanning several days or even years, if each bar represented a different actual time. This would be an impossible task. However, for a replay/simulation system we are looking for a different dynamic. We want a 1 minute bar to be created at 1 minute intervals, getting as close to that as possible.

### Preparing the ground

Our focus will be solely on the replay/simulation service code. There is no need to worry about other aspects at this time. Thus, we will begin modifying the code of the C\_Replay class, trying to optimize as much as possible what we have already developed and tested. Here is the first procedure that appears in the class:

```
inline bool CheckFileIsBar(int &file, const string szFileName)
                        {
                                string  szInfo = "";
                                bool    bRet;

                                for (int c0 = 0; (c0 < 9) && (!FileIsEnding(file)); c0++) szInfo += FileReadString(file);
                                if ((bRet = (szInfo == def_Header_Bar)) == false)
                                {
                                        Print("File ", szFileName, ".csv is not a file with bars.");
                                        FileClose(file);
                                }

                                return bRet;
                        }
```

The goal here is to remove from the bar reading function those tests that determine whether the specified file is a file of preview bars or not. This is necessary to avoid repetition of code when it is important to use the same set to determine whether the bars file is ours. In this situation, these bars will not be used as preview bars. They will be converted into simulated tickets for use in the trading system. Based on this, we introduce another function:

```
inline void FileReadBars(int &file, MqlRates &rate[])
                        {
                                rate[0].time = StringToTime(FileReadString(file) + " " + FileReadString(file));
                                rate[0].open = StringToDouble(FileReadString(file));
                                rate[0].high = StringToDouble(FileReadString(file));
                                rate[0].low = StringToDouble(FileReadString(file));
                                rate[0].close = StringToDouble(FileReadString(file));
                                rate[0].tick_volume = StringToInteger(FileReadString(file));
                                rate[0].real_volume = StringToInteger(FileReadString(file));
                                rate[0].spread = (int) StringToInteger(FileReadString(file));
                        }
```

It will read data line by line from the bars present in the specified file. I think you will not encounter any difficulties in understanding this code. Continuing this preparation phase, here is another function:

```
inline bool OpenFileBars(int &file, const string szFileName)
                        {
                                if ((file = FileOpen("Market Replay\\Bars\\" + szFileName + ".csv", FILE_CSV | FILE_READ | FILE_ANSI)) != INVALID_HANDLE)
                                {
                                        if (!CheckFileIsBar(file, szFileName))
                                                return false;
                                        return true;
                                }
                                Print("Falha ao acessar ", szFileName, ".csv de barras.");

                                return false;
                        }
```

We have now completely centralized our system to provide standard access to bars: both when we use them as preview bars, and when we use them as bars that will be simulated and converted into tickets for presentation. Therefore, the previous function for loading preview bars also had to be changed, making it as shown below:

```
bool LoadPrevBars(const string szFileNameCSV)
        {
                int     file,
                        iAdjust = 0;
                datetime dt = 0;
                MqlRates Rate[1];

                if (OpenFileBars(file, szFileNameCSV))
                {
                        Print("Loading preview bars for Replay. Please wait....");
                        while ((!FileIsEnding(file)) && (!_StopFlag))
                        {
                                FileReadBars(file, Rate);
                                iAdjust = ((dt != 0) && (iAdjust == 0) ? (int)(Rate[0].time - dt) : iAdjust);
                                dt = (dt == 0 ? Rate[0].time : dt);
                                CustomRatesUpdate(def_SymbolReplay, Rate, 1);
                        }
                        m_dtPrevLoading = Rate[0].time + iAdjust;
                        FileClose(file);

                        return (!_StopFlag);
                }
                m_dtPrevLoading = 0;

                return false;
        }
```

The way this download function works has not changed, although there are now more calls. Extracting from the previous function the parts to be used in the new point gives us greater security since all the code has already been previously tested. This way we will only have to worry about new functions. Now that the ground is ready, we need to implement a new addition in the configuration file. The function aims to determine which bar files should be simulated in terms of tickets. To do this we need to add a new definition:

```
#define def_STR_FilesBar        "[BARS]"
#define def_STR_FilesTicks      "[TICKS]"
#define def_STR_TicksToBars     "[TICKS->BARS]"
#define def_STR_BarsToTicks     "[BARS->TICKS]"
```

This already allows us to run a simple test, which is exactly what we need to start working on simulation.

```
                bool SetSymbolReplay(const string szFileConfig)
                        {
#define macroERROR(MSG) { FileClose(file); MessageBox((MSG != "" ? MSG : StringFormat("An error occurred in line %d", iLine)), "Market Replay", MB_OK); return false; }
                                int     file,
                                        iLine;
                                string  szInfo;
                                char    iStage;

                                if ((file = FileOpen("Market Replay\\" + szFileConfig, FILE_CSV | FILE_READ | FILE_ANSI)) == INVALID_HANDLE)
                                {
                                        MessageBox("Failed to open the\nconfiguration file.", "Market Replay", MB_OK);
                                        return false;
                                }
                                Print("Loading data for replay. Please wait....");
                                ArrayResize(m_Ticks.Rate, def_BarsDiary);
                                m_Ticks.nRate = -1;
                                m_Ticks.Rate[0].time = 0;
                                iStage = 0;
                                iLine = 1;
                                while ((!FileIsEnding(file)) && (!_StopFlag))
                                {
                                        switch (GetDefinition(FileReadString(file), szInfo))
                                        {
                                                case Transcription_DEFINE:
                                                        if (szInfo == def_STR_FilesBar) iStage = 1; else
                                                        if (szInfo == def_STR_FilesTicks) iStage = 2; else
                                                        if (szInfo == def_STR_TicksToBars) iStage = 3; else
                                                        if (szInfo == def_STR_BarsToTicks) iStage = 4; else
                                                                macroERROR(StringFormat("%s is not recognized in the system\nin line %d.", szInfo, iLine));
                                                        break;
                                                case Transcription_INFO:
                                                        if (szInfo != "") switch (iStage)
                                                        {
                                                                case 0:
                                                                        macroERROR(StringFormat("Command not recognized in line %d\nof the configuration file.", iLine));
                                                                        break;
                                                                case 1:
                                                                        if (!LoadPrevBars(szInfo)) macroERROR("");
                                                                        break;
                                                                case 2:
                                                                        if (!LoadTicksReplay(szInfo)) macroERROR("");
                                                                        break;
                                                                case 3:
                                                                        if (!LoadTicksReplay(szInfo, false)) macroERROR("");
                                                                        break;
                                                                case 4:
                                                                        if (!LoadBarsToTicksReplay(szInfo)) macroERROR("");
                                                                        break;
                                                        }
                                                        break;
                                        };
                                        iLine++;
                                }
                                FileClose(file);

                                return (!_StopFlag);
#undef macroERROR
                        }
```

See how easy it is to add new functions to the code. The very fact of adding this test already gives us the opportunity to analyze even more aspects. From here we can look at the same type of behavior that we observed in other situations. Any file defined at this stage will be treated as a bar file to be converted to tickets, and this is done using this call.

Again, whenever possible, we should avoid writing unnecessary code. It is advisable to reuse previously tested codes. This is exactly how we have done it until now. However, we will soon start a new topic, although this will be a topic for another article. But before that, it is important to understand one essential aspect.

### A few thoughts before implementation

Before implementing the conversion system, there is one point to consider. Do you know how many different types of bar configurations really exist? Although many people believe that there are many types, we can actually boil down all possible configurations to just four. They are shown in the figure below:

![](https://c.mql5.com/2/46/001__11.png)

Why is this relevant to us? This is relevant because it determines how many options we will have to implement. If we do not understand the fact that there are only these four options, then we run the risk of missing some options or, conversely, creating more types of cases than necessary. Once again, I want to emphasize that there is no way to create a perfect simulated model to recreate bars. The most that can be achieved is a more or less accurate estimate of the actual movement that led to the formation of this particular bar.

There are some details regarding the second type, where the bar body can be placed only on top as shown in the image. However, this fact does not affect the system that we will implement. Likewise, it doesn't matter whether the bar represents a sell or buy trade; the implementation will remain the same. The only nuance is in which initial direction we should go. This way we minimize the number of cases we need to implement. But besides the cases presented in the figure, we still need to understand one more thing: How many minimum tickets do we really need to create? This may be confusing for some, but for someone implementing a replay/simulation system or even a strategy tester, it will all make sense.

Let's think about it: It is not practical to use only 1 ticket in any system as this will only represent a buy or sell trade and not the movement itself. Therefore we can rule out this possibility. We could come up with at least two tickets that would symbolize the opening point and the closing point. Although this seems logical, we will not have any real movement either, since we will only need to generate one tick to open the bar and a second to close it.

NOTE: We are not trying to generate just tickets, in fact we want to create a movement that simulates a bar. We'll further explore these topics in future articles, but first we need to develop a basic system.

Thus, the minimum number of tickets will be 3. Therefore, the 1-minute bar may reflect some of the configurations observed in the previous figure. However, note that the presence of at least 3 tickets does not mean that the price has moved exactly 1 tick up and one tick down, or 3 ticks up or down. The movement may differ from this 1 tick due to the lack of liquidity at the bar creation time.

Important: The reader may be confused by some of the terms used here. Let's clarify this to avoid misunderstandings: When I mention the term **_TICKET_**, I actually mean a trading event, that is, an event of buying or selling an asset at a specified price. Regarding the term _TICK_, I mean the smallest deviation relative to the trading price. To understand this difference, you need to consider the following: 1 tick in the stock market costs 0.01 points, 1 tick in dollar futures costs 0.5 points, and 1 tick in index futures costs 5 points.

While this can make things more difficult in some aspects since the movement simulation no longer reflects exact reality but rather an idealized movement, it is important to mention this fact to keep in mind that in many cases a simulation system using 1 minute bars will not accurately reproduces what actually happened or is happening in the market. Therefore, it will always be better to use the shortest timeframes. Since this timeframe is 1-minute, you should always use it.

Maybe you don't yet understand the real problem with using bars as a method of creating tickets. But think about keep in mind the following: if in a real market a 1-minute bar opens at a certain price. Due to lack of liquidity the price jumps by 3 ticks, and after some time falls by 1 tick, then when it closes at that last position, the final situation will be the following:

![](https://c.mql5.com/2/46/002__5.png)

The image above may be confusing and you may not have understood it correctly. It represents the following information: In the left corner we look at the actual movement in ticks. Small horizontal bars represent each tick. The circles represent the prices at which the asset actually stopped between tickets. The green line indicates a price jump. Please note that there have been cases where there were no trades on certain ticks, but when analyzing the OHCL value we do not see any obvious tick spikes. By simulating the movement using only the bar candlesticks, we see what is shown in the following image.

![](https://c.mql5.com/2/46/003__3.png)

The blue line represents the simulated movement. In this case, we are going to go through all the ticks, regardless of what actually happened during the live trade. So always keep in mind that modeling is not the same as using real data. No matter how complex a modeling system is, it will never accurately reflect reality.

### Implementing the basic conversion system

As already mentioned, the first thing to do is determine the size of the price tick. To do this, we will have to include some additional elements in the configuration file. They must be recognized by the C\_Replay class. Therefore, we will need to add some definitions and additional code to this class. We will start with the following lines.

```
#define def_STR_FilesBar        "[BARS]"
#define def_STR_FilesTicks      "[TICKS]"
#define def_STR_TicksToBars     "[TICKS->BARS]"
#define def_STR_BarsToTicks     "[BARS->TICKS]"
#define def_STR_ConfigSymbol    "[CONFIG]"
#define def_STR_PointsPerTicks  "POINTSPERTICK"
#define def_Header_Bar          "<DATE><TIME><OPEN><HIGH><LOW><CLOSE><TICKVOL><VOL><SPREAD>"
#define def_Header_Ticks        "<DATE><TIME><BID><ASK><LAST><VOLUME><FLAGS>"
#define def_BarsDiary           540
```

This line defines a string that will recognize the configuration data that we will process. It will also provide the first of the configurations we can define from now on. Again, we will need to add a few lines of code to the system so that these settings can be interpreted and applied, but the additions are relatively simple. Next, let's look at what needs to be done:

```
                bool SetSymbolReplay(const string szFileConfig)
                        {
#define macroERROR(MSG) { FileClose(file); MessageBox((MSG != "" ? MSG : StringFormat("An error occurred in line %d", iLine)), "Market Replay", MB_OK); return false; }
                                int     file,
                                        iLine;
                                string  szInfo;
                                char    iStage;

                                if ((file = FileOpen("Market Replay\\" + szFileConfig, FILE_CSV | FILE_READ | FILE_ANSI)) == INVALID_HANDLE)
                                {
                                        MessageBox("Failed to open the\nconfiguration file.", "Market Replay", MB_OK);
                                        return false;
                                }
                                Print("Loading data for replay. Please wait....");
                                ArrayResize(m_Ticks.Rate, def_BarsDiary);
                                m_Ticks.nRate = -1;
                                m_Ticks.Rate[0].time = 0;
                                iStage = 0;
                                iLine = 1;
                                while ((!FileIsEnding(file)) && (!_StopFlag))
                                {
                                        switch (GetDefinition(FileReadString(file), szInfo))
                                        {
                                                case Transcription_DEFINE:
                                                        if (szInfo == def_STR_FilesBar) iStage = 1; else
                                                        if (szInfo == def_STR_FilesTicks) iStage = 2; else
                                                        if (szInfo == def_STR_TicksToBars) iStage = 3; else
                                                        if (szInfo == def_STR_BarsToTicks) iStage = 4; else
                                                        if (szInfo == def_STR_ConfigSymbol) iStage = 5; else
                                                                macroERROR(StringFormat("%s is not recognized in the system\nin line %d.", szInfo, iLine));
                                                        break;
                                                case Transcription_INFO:
                                                        if (szInfo != "") switch (iStage)
                                                        {
                                                                case 0:
                                                                        macroERROR(StringFormat("Command not recognized in line %d\nof the configuration file.", iLine));
                                                                        break;
                                                                case 1:
                                                                        if (!LoadPrevBars(szInfo)) macroERROR("");
                                                                        break;
                                                                case 2:
                                                                        if (!LoadTicksReplay(szInfo)) macroERROR("");
                                                                        break;
                                                                case 3:
                                                                        if (!LoadTicksReplay(szInfo, false)) macroERROR("");
                                                                        break;
                                                                case 4:
                                                                        if (!LoadBarsToTicksReplay(szInfo)) macroERROR("");
                                                                        break;
                                                                case 5:
                                                                        if (!Configs(szInfo)) macroERROR("");
                                                                        break;
                                                        }
                                                        break;
                                        };
                                        iLine++;
                                }
                                FileClose(file);

                                return (!_StopFlag);
#undef macroERROR
                        }
```

Here we indicate that all information from the next line will be processed by the system at the 5th stage of analysis. When the information entry is triggered and captured in step 5, the procedure is called. To simplify the description of this procedure, below we describe the procedure called at the 5th stage.

```
inline bool Configs(const string szInfo)
                        {
                                string szRet[];

                                if (StringSplit(szInfo, '=', szRet) == 2)
                                {
                                        StringTrimRight(szRet[0]);
                                        StringTrimLeft(szRet[1]);
                                        if (szRet[0] == def_STR_PointsPerTicks) m_PointsPerTick = StringToDouble(szRet[1]); else
                                        {
                                                Print("Variable >>", szRet[0], "<< not defined.");
                                                return false;
                                        }
                                        return true;
                                }
                                Print("Definition of configuration >>", szInfo, "<< is invalid.");
                                return false;
                        }
```

First we capture and isolate the name of the internal repeating variable from the value that will be used. This was defined by the user in the replay/simulation configuration file. The result of this operation will give us two pieces of information: the first is the name of the variable being defined, and the second is its value. This value may vary in type depending on the variable, but it will be completely transparent to the user. You don't have to worry about whether the type is string, double or integer. The type is selected here in the code.

Before using this data, you must remove everything that is not relevant to the basic information. Typically this element is some type of internal format that can be used to facilitate the user's reading or writing of configuration files. Anything that is not understood and applied is considered an error. This causes us to return false, which, in turn, causes the service to terminate. Later, of course, the reason is reported in the MetaTrader 5 platform.

So our configuration file will be as follows. Remember that this is just an example configuration file:

```
[Config]
PointsPerTick = 5

[Bars]
WIN$N_M1_202112060900_202112061824
WIN$N_M1_202112070900_202112071824

[ Ticks -> Bars]

[Ticks]

[ Bars -> Ticks ]
WIN$N_M1_202112080900_202112081824

#End of the configuration file...
```

Before you continue, you need to do some final setup on your system. It is crucial. We need to debug the system to ensure that the bar mechanism translates into a simulated ticket model. The required change is highlighted in the following code:

```
inline int Event_OnTime(void)
        {
                bool    bNew;
                int     mili, iPos;
                u_Interprocess Info;
                static MqlRates Rate[1];
                static datetime _dt = 0;
                datetime tmpDT = macroRemoveSec(m_Ticks.Info[m_ReplayCount].time);

                if (m_ReplayCount >= m_Ticks.nTicks) return -1;
                if (bNew = (_dt != tmpDT))
                {
                        _dt = tmpDT;
                        Rate[0].real_volume = 0;
                        Rate[0].tick_volume = 0;
                }
                mili = (int) m_Ticks.Info[m_ReplayCount].time_msc;
                do
                {
                        while (mili == m_Ticks.Info[m_ReplayCount].time_msc)
                        {
                                Rate[0].close = m_Ticks.Info[m_ReplayCount].last;
                                Rate[0].open = (bNew ? Rate[0].close : Rate[0].open);
                                Rate[0].high = (bNew || (Rate[0].close > Rate[0].high) ? Rate[0].close : Rate[0].high);
                                Rate[0].low = (bNew || (Rate[0].close < Rate[0].low) ? Rate[0].close : Rate[0].low);
                                Rate[0].real_volume += (long) m_Ticks.Info[m_ReplayCount].volume_real;
                                bNew = false;
                                m_ReplayCount++;
                        }
                        mili++;
                }while (mili == m_Ticks.Info[m_ReplayCount].time_msc);
                Rate[0].time = _dt;
                CustomRatesUpdate(def_SymbolReplay, Rate, 1);
                iPos = (int)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
                GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                if (Info.s_Infos.iPosShift != iPos)
                {
                        Info.s_Infos.iPosShift = (ushort) iPos;
                        GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                }
                return (int)(m_Ticks.Info[m_ReplayCount].time_msc < mili ? m_Ticks.Info[m_ReplayCount].time_msc + (1000 - mili) : m_Ticks.Info[m_ReplayCount].time_msc - mili);
        }
```

There is an interesting point here: you can turn off visualization to see how the movement occurs on the chart. We are talking about disabling the function that excludes the second values that are present in the simulated tickets. In this case, each simulated ticket will be represented on the chart as a bar.

While this seems a little unusual, it will help us understand what's going on without having to write to a log file. It is for this entry that we will have to remove the value corresponding to seconds. If we don't do this, then every simulated ticket will generate a bar, and ultimately we don't need that. We want the 1 minute bars to appear as if they were real.

### Finally we come to the implementation phase

Before we proceed to the code, let's see the movement I'm going to introduce in this article. In the future I will show the reader how to make it more complex, but first it is important that it works correctly. Below you can see how this movement will occur:

![](https://c.mql5.com/2/46/004.png)

While this may seem simple, we will convert the bar into simulated ticket movement. Because the movement is **never** linear but is a kind of zigzag, I decided to compose a movement with three such oscillations. You can increase this number if you wish. In future articles, I'll show you how to turn this basic technique into something much more complex.

Now that we know what this movement will be, we can move on to the code. The first function required for creating the conversion system is shown in the following code:

```
                bool LoadBarsToTicksReplay(const string szFileNameCSV)
                        {
//#define DEBUG_SERVICE_CONVERT
                                int file, max;
                                MqlRates rate[1];
                                MqlTick tick[];

                                if (OpenFileBars(file, szFileNameCSV))
                                {
                                        Print("Converting bars to ticks. Please wait...");
                                        ArrayResize(m_Ticks.Info, def_MaxSizeArray, def_MaxSizeArray);
                                        ArrayResize(m_Ticks.Rate, def_BarsDiary);
                                        ArrayResize(tick, def_MaxSizeArray);
                                        while ((!FileIsEnding(file)) && (!_StopFlag))
                                        {
                                                FileReadBars(file, rate);
                                                max = SimuleBarToTicks(rate[0], tick);
                                                for (int c0 = 0; c0 <= max; c0++)
                                                {
                                                        ArrayResize(m_Ticks.Info, (m_Ticks.nTicks + 1), def_MaxSizeArray);
                                                        m_Ticks.Info[m_Ticks.nTicks++] = tick[c0];
                                                }
                                        }
                                        FileClose(file);
                                        ArrayFree(tick);

#ifdef DEBUG_SERVICE_CONVERT
        file = FileOpen("Infos.txt", FILE_ANSI | FILE_WRITE);
        for (long c0 = 0; c0 < m_Ticks.nTicks; c0++)
                FileWriteString(file, StringFormat("%s.%03d %f --> %f\n", TimeToString(m_Ticks.Info[c0].time, TIME_DATE | TIME_SECONDS), m_Ticks.Info[c0].time_msc, m_Ticks.Info[c0].last, m_Ticks.Info[c0].volume_real));
        FileClose(file);
#endif

                                        return (!_StopFlag);
                                }

                                return false;
                        }
```

Although it seems simple, this function is the first step responsible for creating a simulation. Seeing it, we can understand at least a little how the system performs the simulation. It is important to note that, unlike the strategy tester, here the simulation of each minute bar will take approximately one minute while using the service until it is fully displayed on the screen. This is done intentionally. Therefore, if the goal here is, say, to test a strategy, I recommend that you use the tool available on the MetaTrader 5 platform, and use this tool (the development of which we show) only for manual training and testing.

There is an aspect that will no longer exist in the future, but at this moment it is crucial. This refers to the definition we are currently discussing, which allows you to generate a file of simulated ticket data to analyze the level of complexity present in the simulated system. It is important to note that the created file will contain only the data necessary for analysis, without additional or unnecessary information.

The rest of the function is pretty intuitive since the code already existed before. Now we have a call that we will look at in more detail later. After this call, which simulates the tickets, a small loop is executed to save the tickets in the database. It's easy to understand and doesn't require any further explanation.

```
inline int SimuleBarToTicks(const MqlRates &rate, MqlTick &tick[])
                        {
                                int t0 = 0;
                                long v0, v1, v2, msc;

                                m_Ticks.Rate[++m_Ticks.nRate] = rate;
                                Pivot(rate.open, rate.low, t0, tick);
                                Pivot(rate.low, rate.high, t0, tick);
                                Pivot(rate.high, rate.close, t0, tick, true);
                                v0 = (long)(rate.real_volume / (t0 + 1));
                                v1 = 0;
                                msc = 5;
                                v2 = ((60000 - msc) / (t0 + 1));
                                for (int c0 = 0; c0 <= t0; c0++, v1 += v0)
                                {
                                        tick[c0].volume_real = (v0 * 1.0);
                                        tick[c0].time = rate.time + (datetime)(msc / 1000);
                                        tick[c0].time_msc = msc % 1000;
                                        msc += v2;
                                }
                                tick[t0].volume_real = ((rate.real_volume - v1) * 1.0);

                                return t0;
                        }
```

The above feature makes the task a little more difficult and hence makes it more interesting. At the same time, it helps us put together all the necessary structure in a more organized and manageable manner. In fact, to reduce the complexity of this function, I created another procedure that is called three times. If you read the documentation [Real and generated ticks - Algorithmic trading](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation "https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation"), you will notice that in it the system is called not three, but four times. You can add more calls if you so desire, but as stated, I will show a way to increase the complexity of this system without having to add additional calls to the Pivot procedure.

Regarding the previously mentioned procedure, after making three calls to Pivot we will have a number of simulated tickets that will depend on how the division was performed. Thanks to this, we can now make small corrections to the 1-minute bar data, allowing us to use the original data in some way. The first step is to perform a simple division of the real volume so that each simulated tick contains a fraction of the total volume. We then make a small adjustment to the timing of each simulated tick. After we have defined what fractions to use, we can enter a loop and ensure that each fraction is stored in the appropriate ticket. At the moment, we will stick to the fact that the system must actually work. Although the above functions can be improved a lot, so it makes things more interesting. Unlike with the time, the volume must be corrected in order to remain identical to the original. Because of this detail, we have the last calculation in this procedure, and this makes the correction so that the final volume of the 1-minute bar, is the same as the initial volume.

Now let's look at the last function in this article, which will create a pivot point based on the values and parameters provided by the above code. It's important to note that the values can be adjusted to suit your interests, but care must be taken to ensure the subsequent function works correctly.

```
//+------------------------------------------------------------------+
#define macroCreateLeg(A, B, C) if (A < B)      {               \
                while (A < B)   {                               \
                        tick[C++].last = A;                     \
                        A += m_PointsPerTick;                   \
                                }                               \
                                                } else {        \
                while (A > B)   {                               \
                        tick[C++].last = A;                     \
                        A -= m_PointsPerTick;                   \
                                }               }

inline void Pivot(const double p1, const double p2, int &t0, MqlTick &tick[], bool b0 = false)
                        {
                                double v0, v1, v2;

                                v0 = (p1 > p2 ? p1 - p2 : p2 - p1);
                                v1 = p1 + (MathFloor((v0 * 0.382) / m_PointsPerTick) * m_PointsPerTick * (p1 > p2 ? -1 : 1));
                                v2 = p1 + (MathFloor((v0 * 0.618) / m_PointsPerTick) * m_PointsPerTick * (p1 > p2 ? -1 : 1));
                                v0 = p1;
                                macroCreateLeg(v0, v2, t0);
                                macroCreateLeg(v0, v1, t0);
                                macroCreateLeg(v0, p2, t0);
                                if (b0) tick[t0].last = v0;
                        }
#undef macroCreateLeg
//+------------------------------------------------------------------+
```

The above feature is simple in terms of operation. Although its calculations may seem strange, when looking at the values used to create the pivot point, you will notice that we always try to set the pivot using the first and third Fibonacci lines. First, it's important to note that it doesn't matter whether the pivot is up or down; the function will perform the calculations accordingly. Then comes an aspect that might be confusing to those with little programming knowledge: MACRO. The reason for using the macro is that it is easier to create part of a pivot using a macro. But you can also create a function for this. In fact, if we were using pure C++, this macro would probably have completely different code. But here, using MQL5 as it was created, it works as a workaround.

It is much more efficient to use this macro than to embed code inside the areas where it is declared.

### Conclusion

We should always prefer code that is easy to read and understand over code that requires corrections or modifications and forces us to spend hours figuring out our actions. This concludes this article. The video below shows the result at the current stage of development, that was created using what is attached to the article.

However, I would like to note a problem that arose when using simulated tickets. This refers to a system for moving or searching for a position other than the one in which the replay service is located. This problem will only occur if you use simulated tickets. In the next article, we will address and fix this issue and make other improvements.

Narrado 11 - YouTube

Tap to unmute

[Narrado 11](https://www.youtube.com/watch?v=J4BZ_Gq4OZ0) [MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ)

MQL5.community1.91K subscribers

[Watch on](https://www.youtube.com/watch?v=J4BZ_Gq4OZ0)

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10973](https://www.mql5.com/pt/articles/10973)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10973.zip "Download all attachments in the single ZIP archive")

[Market\_Replay.zip](https://www.mql5.com/en/articles/download/10973/market_replay.zip "Download Market_Replay.zip")(50.42 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/457330)**

![Developing a Replay System — Market simulation (Part 12): Birth of the SIMULATOR (II)](https://c.mql5.com/2/54/replay-p12-avatar.png)[Developing a Replay System — Market simulation (Part 12): Birth of the SIMULATOR (II)](https://www.mql5.com/en/articles/10987)

Developing a simulator can be much more interesting than it seems. Today we'll take a few more steps in this direction because things are getting more interesting.

![Developing a Replay System — Market simulation (Part 10): Using only real data for Replay](https://c.mql5.com/2/54/replay-p10.png)[Developing a Replay System — Market simulation (Part 10): Using only real data for Replay](https://www.mql5.com/en/articles/10932)

Here we will look at how we can use more reliable data (traded ticks) in the replay system without worrying about whether it is adjusted or not.

![Data Science and Machine Learning (Part 15): SVM, A Must-Have Tool in Every Trader's Toolbox](https://c.mql5.com/2/60/Data_Science_and_Machine_LearningdPart_15g__Logo.png)[Data Science and Machine Learning (Part 15): SVM, A Must-Have Tool in Every Trader's Toolbox](https://www.mql5.com/en/articles/13395)

Discover the indispensable role of Support Vector Machines (SVM) in shaping the future of trading. This comprehensive guide explores how SVM can elevate your trading strategies, enhance decision-making, and unlock new opportunities in the financial markets. Dive into the world of SVM with real-world applications, step-by-step tutorials, and expert insights. Equip yourself with the essential tool that can help you navigate the complexities of modern trading. Elevate your trading game with SVM—a must-have for every trader's toolbox.

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 3): Added symbols prefixes and/or suffixes and Trading Time Session](https://c.mql5.com/2/60/FXSAR_MTF_MCEA_icon.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 3): Added symbols prefixes and/or suffixes and Trading Time Session](https://www.mql5.com/en/articles/13705)

Several fellow traders sent emails or commented about how to use this Multi-Currency EA on brokers with symbol names that have prefixes and/or suffixes, and also how to implement trading time zones or trading time sessions on this Multi-Currency EA.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/10973&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069066779868725308)

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