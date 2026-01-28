---
title: Developing a Replay System — Market simulation (Part 19): Necessary adjustments
url: https://www.mql5.com/en/articles/11125
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:04:30.398507
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/11125&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069026621924507644)

MetaTrader 5 / Tester


### Introduction

I think it is clear from the previous articles within this series that we need to implement some additional points. This is absolutely necessary to better organize the work, especially with some future improvements. If you plan to use the replay/simulation system only to work with one asset, then you won't need many of the things we are going to implement now. You can leave them aside – I mean they do not necessarily have to be present in the configuration file.

However, it is very likely that you will use not only one asset, but several different ones or even a fairly large database. In this case, we need to organize things, and therefore will need to implement additional code to achieve this goal, although in some very specific cases we could simply use what we already have available in the source code, but in an implicit way. This just needs to be brought out into the light.

I always like to keep things very well organized, and I guess many people think and try to do the same. It will be good to know and understand how to implement this functionality. In addition, you will learn how to add new parameters to the system if you need a specific parameter for a specific asset that you want to use for study or analysis.

Here we will prepare the ground so that if we need to add new functions to the code, this will happen smoothly and easily. The current code cannot yet cover or handle some of the things that will be necessary to make meaningful progress. We need everything to be structured in order to enable the implementation of certain things with the minimal effort. If we do everything correctly, we can get a truly universal system that can very easily adapt to any situation that needs to be handled. One of these aspects will be the topic of the next article. Luckily, thanks to the last two articles showing how to add ticks to the Market Watch window, things are generally going according to plan. If you have missed these articles, you can access them using the following links: [Developing a Replay System — Market simulation (Part 17): Ticks and more ticks (I)](https://www.mql5.com/en/articles/11106) and [Developing a Replay System — Market simulation (Part 18): Ticks and more ticks (II)](https://www.mql5.com/en/articles/11113). These two articles provide valuable information about what we will be doing in future articles.

However, some very specific details are still missing, and these will be implemented in this article. In addition, there are other quite complex issues that will require separate articles explaining how to work on them and solve the problems. Let's now start implementing the system that we will see in this article. We'll start by improving the organization of the data we use.

### Implementing a directory system

The question here is not whether we really need to implement this system, but why we should implement it. At the current stage of development, we can use the directory system. However, we will have to do a lot more work on implementing things for the replay/simulation service. I mean that there's a lot more to the job than simply adding a new variable to a config file. To understand what I am talking about, take a look at the following images:

![Figure 01](https://c.mql5.com/2/47/001__6.png)

Figure 01 – Way to access directories in the current system.

![Figure 02](https://c.mql5.com/2/47/002__2.png)

Figure 02 - An alternative way to access directories.

Even though Figure 01 has the same behavior as Figure 02 from the replay/simulation system perspective, you will soon notice that it is much more practical to configure things using is shown in Figure 02. This is because we only need to specify the directory where the data will be located once, and the replay/modeling system will take care of the rest. While using the system shown in Figure 02, we are saved from forgetting or incorrectly specifying where to look for data if we are using a very large database, for example to add the use of a moving average. If you make this write error, two situations can occur:

- In the first case the system will simply issue a warning that the data cannot be accessed.
- In the second case, which is more serious, incorrect data will be used.

However, by being able to set up a directory in one place, these types of errors are much less likely. It's not that they won't happen at all, but they will be more unlikely. Remember that we can organize objects into even more specific directories, thus combining Figure 01 and Figure 02. However, here I will leave everything at a simpler level. Feel free to implement things in a way that suits your data processing and fits your organizational style.

We've seen the theory, now it's time to see how to do it in practice. The process is relatively simple and straightforward, at least compared to what we have yet to do. First we create a new private variable for the class, as shown in the code below:

```
private :
    enum eTranscriptionDefine {Transcription_INFO, Transcription_DEFINE};
    string m_szPath;
```

When we add this variable to this position, it will become visible to all internal procedures of the class. However, it will not be accessible from outside the class. This prevents it from being overly modified. This is because in some internal procedure of a class, we can change the value of a variable without even realizing it. And we may have difficulty understanding why the code is not working as expected.

Once this is done, we need to tell our class to start recognizing the new command in the configuration file. This procedure is done at a very specific point, but can vary depending on what we add. In our case, we will do this in the order shown below:

```
inline bool Configs(const string szInfo)
    {
        const string szList[] = {
                                "POINTSPERTICK",
                                "PATH"
                                };
        string  szRet[];
        char    cWho;

        if (StringSplit(szInfo, '=', szRet) == 2)
        {
            StringTrimRight(szRet[0]);
            StringTrimLeft(szRet[1]);
            for (cWho = 0; cWho < ArraySize(szList); cWho++) if (szList[cWho] == szRet[0]) break;
            switch (cWho)
            {
                case 0:
                    m_PointsPerTick = StringToDouble(szRet[1]);
                    return true;
                case 1:
                    m_szPath = szRet[1];
                    return true;
            }
            Print("Variable >>", szRet[0], "<< undefined.");
        }else
            Print("Definition of configuratoin >>", szInfo, "<< invalid.");

        return false;
    }
```

Notice how much easier this is when all the code is structured to receive improvements. However, we must be careful. If we take precautions, we will have no problem adding everything we need to the code.

The first thing we do is add the name or label of the command to be used in the configuration file inside the serial data array. Note that all this must be written in capital letters. We could make it case sensitive, but that would make it more difficult for the user to type and place the command in the configuration file. If you are the only one who uses the system and want to use the same label but with different values, perhaps using a case-sensitive system is a good idea. Otherwise, this idea would complicate the whole job. Personally, I think that using the same label for different meanings only makes our lives more difficult. That's why I don't do that.

Once the label has been added to the command matrix, we need to implement its functionality. This is done right at this point. Simple as that. Since it is second in the chain and the chain starts from zero, we use the value 1 to indicate that we are implementing that particular functionality. The idea is to only specify the directory name, so the command is quite simple. Finally, we will return true to the caller, thereby indicating that the command was recognized and successfully implemented.

The sequence of making any additions to the system is exactly as shown. Once we do this, we can use the data provided in the configuration file. However, there is one point that I forgot to mention, it is quite simple, but deserves attention. In some cases, it may appear that a new resource that has been added is causing problems, when in fact it may simply be because it was not initialized correctly. In this case, whenever we add a private global variable, we will need to make sure that it is properly initialized in the class constructor. You can see this in the code below where we are initializing a new variable.

```
C_ConfigService()
	:m_szPath(NULL)
	{
	}
```

By doing this, we ensure that we have a known value for the variable that has not yet been assigned a value. In some situations this detail may seem insignificant, but in others it can avoid serious problems and save time, and is considered good programming practice. After this work has been done, the variable has been initialized in the class constructor, and we've established how to assign a value to it based on what's specified in the configuration file, it's time to use this value. This value will be used in only one function that is responsible for controlling the database loading.

Let's see how to implement this:

```
bool SetSymbolReplay(const string szFileConfig)
    {
        #define macroFileName ((m_szPath != NULL ? m_szPath + "\\" : "") + szInfo)
        int        file,
                iLine;
        char    cError,
                cStage;
        string  szInfo;
        bool    bBarsPrev;
        C_FileBars *pFileBars;

        if ((file = FileOpen("Market Replay\\" + szFileConfig, FILE_CSV | FILE_READ | FILE_ANSI)) == INVALID_HANDLE)
        {
            Print("Failed to open the configuration file [", szFileConfig, "]. Closing the service...");
            return false;
        }
        Print("Loading ticks for replay. Please wait....");
        ArrayResize(m_Ticks.Rate, def_BarsDiary);
        m_Ticks.nRate = -1;
        m_Ticks.Rate[0].time = 0;
        iLine = 1;
        cError = cStage = 0;
        bBarsPrev = false;
        while ((!FileIsEnding(file)) && (!_StopFlag) && (cError == 0))
        {
            switch (GetDefinition(FileReadString(file), szInfo))
            {
                case Transcription_DEFINE:
                    cError = (WhatDefine(szInfo, cStage) ? 0 : 1);
                    break;
                case Transcription_INFO:
                    if (szInfo != "") switch (cStage)
                    {
                        case 0:
                            cError = 2;
                            break;
                        case 1:
                            pFileBars = new C_FileBars(macroFileName);
                            if ((m_dtPrevLoading = (*pFileBars).LoadPreView()) == 0) cError = 3; else bBarsPrev = true;
                            delete pFileBars;
                            break;
                        case 2:
                            if (LoadTicks(macroFileName) == 0) cError = 4;
                            break;
                        case 3:
                            if ((m_dtPrevLoading = LoadTicks(macroFileName, false)) == 0) cError = 5; else bBarsPrev = true;
                            break;
                        case 4:
                            if (!BarsToTicks(macroFileName)) cError = 6;
                            break;
                        case 5:
                            if (!Configs(szInfo)) cError = 7;
                            break;
                    }
                break;
            };
            iLine += (cError > 0 ? 0 : 1);
        }
        FileClose(file);
        switch(cError)
        {
            case 0:
                if (m_Ticks.nTicks <= 0)
                {
                    Print("No ticks to use. Closing the service...");
                    cError = -1;
                }else if (!bBarsPrev) FirstBarNULL();
                break;
            case 1  : Print("Command in line ", iLine, " cannot be recognized by the system...");    break;
            case 2  : Print("The system did not expect the content of the line ", iLine);                  break;
            default : Print("Error in line ", iLine);
        }

        return (cError == 0 ? !_StopFlag : false);
#undef macroFileName
    }
```

Since we will be using this in a unique way and in several different places at the same time, I chose to use a macro to make coding easier. All points marked in yellow will receive exactly the code contained in the macro. This greatly simplifies the task, since there is no need to write the same thing several times. This also avoids possible errors which could occur in the case of maintaining or changing code that is used in several different places. Now let's take a closer look at what the macro does.

```
#define macroFileName ((m_szPath != NULL ? m_szPath + "\\" : "") + szInfo)
```

Remember we initialized a variable with a specific value? The moment we try to use the variable, we will check exactly what value it contains. If this is the same path that we initialized in the constructor, we will have a defined path. If it is the one found in the configuration file, we will have a different path, but one way or another, we will end up with a name by which we can access the file.

This system is so universal that you can change the directory at any time without changing anything else in the already completed and compiled system. This way we won't have to recompile all the code when changing the config file. The only thing we need to do to change the directory we want to work in is to use the following fragment inside the configuration file:

```
[Config]
Path = < NEW PATH >
```

Where <NEW PATH> will contain the new address, which will be used in the configuration file from now on. This is quite nice because the work will be greatly reduced when working with databases that may contain a directory structure. Remember that you should systematize and organize the data in the directory to make it easier to find the file you need.

Once this is done, we can move on to the next step, where we will finalize some things that need to be implemented. This is discussed in the next topic.

### Adjusting the custom symbol data

To implement our order system, we will initially need three basic values: **minimum volume**, **tick value** and **tick size**. Only one of these value types is currently implemented, and its implementation is not exactly what it should be, since it may happen that the value is not set in the configuration file. This complicates our job of creating a synthetic symbol that is only intended to simulate probable market movements.

Without the required correction, we may have inconsistent data in the system when working with the order system later. We need to make sure that this data is configured correctly. This will save us from problems when trying to implement something in our system, which already has quite a lengthy code. Therefore, we will begin to correct the wrong points to avoid problems in the next stage of our work. If we have problems, let them be of a different nature. The order system will not actually interact with the service that is used to create the market replay/simulation service. The only information we will encounter is the chart and the symbol name, nothing more. At least that's my intention for now. I don't know if we will really succeed.

For this scenario, the first thing we need to do is initialize the three values that we absolutely need. However, they will be set to zero data. Let's consider this step by step. First, we need to fix our problem. This is done in the below code:

```
C_Replay(const string szFileConfig)
    {
        m_ReplayCount = 0;
        m_dtPrevLoading = 0;
        m_Ticks.nTicks = 0;
        Print("************** Market Replay Service **************");
        srand(GetTickCount());
        GlobalVariableDel(def_GlobalVariableReplay);
        SymbolSelect(def_SymbolReplay, false);
        CustomSymbolDelete(def_SymbolReplay);
        CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay), _Symbol);
        CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
        CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
        SymbolSelect(def_SymbolReplay, true);
        CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0.0);
        CustomSymbolSetString(def_SymbolReplay, SYMBOL_DESCRIPTION, "Symbol for replay / simulation");
        m_IdReplay = (SetSymbolReplay(szFileConfig) ? 0 : -1);
    }
```

Here we set the initial value to zero, but as a bonus we'll also provide a description of our custom symbol. This is not necessary, but it can be interesting if you open a window with a list of symbols and see a symbol with a unique name. You probably already noticed that we will no longer use the variable that existed previously. The variable that was declared at a specific location, as shown in the code below:

```
class C_FileTicks
{
    protected:
        struct st00
        {
            MqlTick  Info[];
            MqlRates Rate[];
            int      nTicks,
                     nRate;
            bool     bTickReal;
        }m_Ticks;
        double       m_PointsPerTick;
    private :
        int          m_File;
```

All points where this variable appears should now refer to the value contained and defined in the symbol. Now we have new code, but essentially this value was mentioned in two places throughout the replay/simulation system. The first one is shown below:

```
inline long RandomWalk(long pIn, long pOut, const MqlRates &rate, MqlTick &tick[], int iMode)
    {
        double vStep, vNext, price, vHigh, vLow, PpT;
        char i0 = 0;

        PpT = SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE);
        vNext = vStep = (pOut - pIn) / ((rate.high - rate.low) / PpT);
        vHigh = rate.high;
        vLow = rate.low;
        for (long c0 = pIn, c1 = 0, c2 = 0; c0 < pOut; c0++, c1++)
        {
            price = tick[c0 - 1].last + (PpT * ((rand() & 1) == 1 ? -1 : 1));
            price = tick[c0].last = (price > vHigh ? price - PpT : (price < vLow ? price + PpT : price));
            switch (iMode)
            {
                case 0:
                    if (price == rate.close)
                        return c0;
                        break;
                case 1:
                    i0 |= (price == rate.high ? 0x01 : 0);
                        i0 |= (price == rate.low ? 0x02 : 0);
                        vHigh = (i0 == 3 ? rate.high : vHigh);
                        vLow = (i0 ==3 ? rate.low : vLow);
                        break;
                case 2:
                    break;
            }
            if ((int)floor(vNext) < c1)
            {
                if ((++c2) <= 3) continue;
                vNext += vStep;
                if (iMode == 2)
                {
                    if ((c2 & 1) == 1)
                    {
                        if (rate.close > vLow) vLow += PpT; else vHigh -= PpT;
                    }else
                    {
                        if (rate.close < vHigh) vHigh -= PpT; else vLow += PpT;
                    }
                } else
                {
                    if (rate.close > vLow) vLow = (i0 == 3 ? vLow : vLow + PpT); else vHigh = (i0 == 3 ? vHigh : vHigh - PpT);
                }
            }
        }

        return pOut;
    }
```

Since we don't want to repeat the same code in multiple places, we use a local variable to help us. However, the principle is the same: we refer to the value defined inside the symbol. The second point this value refers to is in the C\_Replay class. However, for practical reasons, we will do something slightly different from what was shown above, as opposed to when we created the random walk. Presenting and using information in the charts tends to degrade performance due to too many unnecessary calls. This is due to the fact that during the creation of a random walk, each bar will generate three accesses to it.

But once it has been created, it can contain thousands of ticks, all of which were created in just three calls. This tends to slightly slow down performance during presentation and plotting, but let's see how this plays out in practice. When we use a real tick file, i. e. we replay, such slowdown will not occur. This is because when using real data, the system will not require any additional information to plot 1-minute bars and transfer the information to the tick chart in the Market Watch window. We looked at this in two previous articles.

But when we are going to use 1-minute bars to generate ticks, that is, to perform a simulation, we need to know the tick size, so this information can help the service create a suitable movement model. This movement will be visible in the Market Watch window. But this information is not required to create bars, since the conversion was performed in the C\_FileTicks class.

Knowing this detail, we must consider the function that generates the specified chart, and thus check how many calls will be received during execution. Below is a function used during simulation:

```
inline void CreateBarInReplay(const bool bViewMetrics, const bool bViewTicks)
        {
#define def_Rate m_MountBar.Rate[0]

                bool bNew;
        MqlTick tick[1];
        static double PointsPerTick = 0.0;

        if (m_MountBar.memDT != macroRemoveSec(m_Ticks.Info[m_ReplayCount].time))
        {
                        PointsPerTick = (PointsPerTick == 0.0 ? SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) : PointsPerTick);
                        if (bViewMetrics) Metrics();
                        m_MountBar.memDT = (datetime) macroRemoveSec(m_Ticks.Info[m_ReplayCount].time);
                        def_Rate.real_volume = 0;
                        def_Rate.tick_volume = 0;
        }
        bNew = (def_Rate.tick_volume == 0);
        def_Rate.close = (m_Ticks.Info[m_ReplayCount].volume_real > 0.0 ? m_Ticks.Info[m_ReplayCount].last : def_Rate.close);
        def_Rate.open = (bNew ? def_Rate.close : def_Rate.open);
        def_Rate.high = (bNew || (def_Rate.close > def_Rate.high) ? def_Rate.close : def_Rate.high);
        def_Rate.low = (bNew || (def_Rate.close < def_Rate.low) ? def_Rate.close : def_Rate.low);
        def_Rate.real_volume += (long) m_Ticks.Info[m_ReplayCount].volume_real;
        def_Rate.tick_volume += (m_Ticks.Info[m_ReplayCount].volume_real > 0 ? 1 : 0);
        def_Rate.time = m_MountBar.memDT;
        CustomRatesUpdate(def_SymbolReplay, m_MountBar.Rate);
        if (bViewTicks)
        {
                        tick = m_Ticks.Info[m_ReplayCount];
                        if (!m_Ticks.bTickReal)
                        {
                                static double BID, ASK;
                                double  dSpread;
                                int     iRand = rand();

                                dSpread = PointsPerTick + ((iRand > 29080) && (iRand < 32767) ? ((iRand & 1) == 1 ? PointsPerTick : 0 ) : 0 );
                                if (tick[0].last > ASK)
                                {
                                        ASK = tick[0].ask = tick[0].last;
                                        BID = tick[0].bid = tick[0].last - dSpread;
                                }
                                if (tick[0].last < BID)
                                {
                                        ASK = tick[0].ask = tick[0].last + dSpread;
                                        BID = tick[0].bid = tick[0].last;
                                }
                        }
                        CustomTicksAdd(def_SymbolReplay, tick);
        }
                m_ReplayCount++;

#undef def_Rate
        }
```

Here we declare a static local variable. This will be used to avoid unnecessary calls to the function that fixes the tick size. This capture will only happen once during the service lifetime and runtime, while the variable will only be used in the specified places. So there is no point in extending it beyond this function. But note that this place where the variable is actually used will only be available if we are using the simulation mode. In the replay mode, this variable has no practical use.

This has also solved the problem with the tick size. There are two more problems left to solve. However, the problem with ticks has not yet been completely resolved. There is an issue with initialization. We will solve this while solving the other two problems, since the approach will be the same.

### Last details to be created

The question is what we should actually adjust. It's true that we can adjust several things in the custom symbol. But most of them are not needed for our purposes. We need to focus only on what we really need. We also need to and set it up so that when we need this information, we have it in a simple and universal way. I say this because we will start creating an order system soon, but I'm not sure it will actually happen that quickly. In any case, I want our EA to be replay/simulation compatible and to be suitable for use in the real market, both with a demo account and a real one. To do this, we need the components to have the same level of required information that exist in the real market.

In this case, we need to initialize them with zero values. This ensures that the custom symbol will have these values consistent with the values in the real symbol. Additionally, initializing the values to zero means we can test them later, and it makes the job of implementing and testing possible errors in the symbol configuration much easier.

```
C_Replay(const string szFileConfig)
    {
        m_ReplayCount = 0;
        m_dtPrevLoading = 0;
        m_Ticks.nTicks = 0;
        Print("************** Market Replay Service **************");
        srand(GetTickCount());
        GlobalVariableDel(def_GlobalVariableReplay);
        SymbolSelect(def_SymbolReplay, false);
        CustomSymbolDelete(def_SymbolReplay);
        CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay), _Symbol);
        CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
        CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
        SymbolSelect(def_SymbolReplay, true);
        CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0);
        CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, 0);
        CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, 0);
        CustomSymbolSetString(def_SymbolReplay, SYMBOL_DESCRIPTION, "Symbol for replay / simulation");
        m_IdReplay = (SetSymbolReplay(szFileConfig) ? 0 : -1);
    }
```

Here we run those values that we really need in the future, at the very near moment. These values are tick size, tick value, and volume (in this case, the step). But since the step very often corresponds to the minimum volume that should be used, I do not see any problem in setting it instead of the minimum volume. Also because at the next stage this step is even more important for us. There is another reason for that: I tried to adjust the minimum volume, but for some reason I was unable to do so. MetaTrader 5 simply ignored the fact that we needed to set a minimum volume.

Once this is done, we will need to do something else and check if these values have actually been initialized. This is done in the next code:

```
bool ViewReplay(ENUM_TIMEFRAMES arg1)
   {
#define macroError(A) { Print(A); return false; }
   u_Interprocess info;

   if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
        macroError("Configuração do ativo não esta completa, falta declarar o tamanho do ticket.");
   if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
        macroError("Configuração do ativo não esta completa, falta declarar o valor do ticket.");
   if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
        macroError("Configuração do ativo não esta completa, falta declarar o volume mínimo.");
   if (m_IdReplay == -1) return false;
   if ((m_IdReplay = ChartFirst()) > 0) do
   {
        if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
        {
            ChartClose(m_IdReplay);
            ChartRedraw();
        }
   }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
   Print("Aguardando permissão do indicador [Market Replay] para iniciar replay ...");
   info.u_Value.IdGraphic = m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
   ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");
   ChartRedraw(m_IdReplay);
   GlobalVariableDel(def_GlobalVariableIdGraphics);
   GlobalVariableTemp(def_GlobalVariableIdGraphics);
   GlobalVariableSet(def_GlobalVariableIdGraphics, info.u_Value.df_Value);
   while ((!GlobalVariableCheck(def_GlobalVariableReplay)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);

   return ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""));
#undef macroError
  }
```

To avoid unnecessary repetition, we will use a macro. It will display an error message and end with a system failure message. Here we check one by one the values that must be declared and initialized in the configuration file. If any of these have not been initialized, the user will be notified to configure the custom symbol correctly. Without this, the replay/simulation service will not continue to function. From this point on, it will be considered functional and capable of providing the order system, in this case the EA being created, with the necessary data to properly model or replay the market. This will allow us to simulate sending orders.

This is good, but in order to initialize these values, we need to make some additions to the system as shown below:

```
inline bool Configs(const string szInfo)
    {
        const string szList[] = {
                                "PATH",
                                "POINTSPERTICK",
                                "VALUEPERPOINTS",
                                "VOLUMEMINIMAL"
                                };
        string  szRet[];
        char    cWho;

        if (StringSplit(szInfo, '=', szRet) == 2)
        {
            StringTrimRight(szRet[0]);
            StringTrimLeft(szRet[1]);
            for (cWho = 0; cWho < ArraySize(szList); cWho++) if (szList[cWho] == szRet[0]) break;
            switch (cWho)
            {
                case 0:
                    m_szPath = szRet[1];
                    return true;
                case 1:
                    CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, StringToDouble(szRet[1]));
                    return true;
                case 2:
                    CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, StringToDouble(szRet[1]));
                    return true;
                case 3:
                    CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, StringToDouble(szRet[1]));
                    return true;
            }
            Print("Variable >>", szRet[0], "<< undefined.");
        }else
            Print("Definition of configuratoin >>", szInfo, "<< invalid.");

        return false;
    }
```

Adding things to the system is quite simple. Here we have added two new values that can be configured by simply editing the file that configures the replay or simulation of any symbol. This is a tick value that will generate a call to a function that expects the corresponding value, and a step size value that will also call an internal function that adjusts that value. Any other additions will be in the next steps.

### Final considerations

I haven't yet tested whether the values are suitable or not. Therefore, be careful when editing the configuration file to avoid errors when using the order system.

Anyway, you can check how things are going using the attached custom symbols.

**Important note:** _Despite the fact that the system is practically functional, this is not entirely true. Since at the moment it is not possible to perform replay or simulation using Forex data. Because the Forex market uses some things that the system cannot handle yet. Attempting to do this will result in range errors in the system arrays, whether in replay or simulation modes. But I'm working on fixes to be able to work with Forex market data._

In the next article we will begin considering this topic: FOREX.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11125](https://www.mql5.com/pt/articles/11125)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11125.zip "Download all attachments in the single ZIP archive")

[Market\_Replay\_4vv\_19.zip](https://www.mql5.com/en/articles/download/11125/market_replay_4vv_19.zip "Download Market_Replay_4vv_19.zip")(12901.8 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/458690)**

![Neural networks made easy (Part 54): Using random encoder for efficient research (RE3)](https://c.mql5.com/2/57/random_encoder_for_efficient_exploration_054_avatar.png)[Neural networks made easy (Part 54): Using random encoder for efficient research (RE3)](https://www.mql5.com/en/articles/13158)

Whenever we consider reinforcement learning methods, we are faced with the issue of efficiently exploring the environment. Solving this issue often leads to complication of the algorithm and training of additional models. In this article, we will look at an alternative approach to solving this problem.

![Modified Grid-Hedge EA in MQL5 (Part I): Making a Simple Hedge EA](https://c.mql5.com/2/62/Modified_Grid-Hedge_EA_in_MQL5_4Part_Ip_Making_a_Simple_Hedge_EA__LOGO.png)[Modified Grid-Hedge EA in MQL5 (Part I): Making a Simple Hedge EA](https://www.mql5.com/en/articles/13845)

We will be creating a simple hedge EA as a base for our more advanced Grid-Hedge EA, which will be a mixture of classic grid and classic hedge strategies. By the end of this article, you will know how to create a simple hedge strategy, and you will also get to know what people say about whether this strategy is truly 100% profitable.

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5):  Bollinger Bands On Keltner Channel — Indicators Signal](https://c.mql5.com/2/61/rj-article-image-60x60.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5): Bollinger Bands On Keltner Channel — Indicators Signal](https://www.mql5.com/en/articles/13861)

The Multi-Currency Expert Advisor in this article is an Expert Advisor or Trading Robot that can trade (open orders, close orders and manage orders for example: Trailing Stop Loss and Trailing Profit) for more than one symbol pair from only one symbol chart. In this article we will use signals from two indicators, in this case Bollinger Bands® on Keltner Channel.

![Developing a Replay System — Market simulation (Part 18): Ticks and more ticks (II)](https://c.mql5.com/2/56/replay-p18-avatar.png)[Developing a Replay System — Market simulation (Part 18): Ticks and more ticks (II)](https://www.mql5.com/en/articles/11113)

Obviously the current metrics are very far from the ideal time for creating a 1-minute bar. That's the first thing we are going to fix. Fixing the synchronization problem is not difficult. This may seem hard, but it's actually quite simple. We did not make the required correction in the previous article since its purpose was to explain how to transfer the tick data that was used to create the 1-minute bars on the chart into the Market Watch window.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/11125&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069026621924507644)

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