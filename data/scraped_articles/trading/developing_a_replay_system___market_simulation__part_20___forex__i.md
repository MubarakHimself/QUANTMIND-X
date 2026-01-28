---
title: Developing a Replay System — Market simulation (Part 20): FOREX (I)
url: https://www.mql5.com/en/articles/11144
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:04:10.747231
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/11144&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069014127864643560)

MetaTrader 5 / Tester


### Introduction

In the previous article " [Developing a Replay System — Market simulation (Part 19): Necessary adjustments](https://www.mql5.com/en/articles/11125)", we implemented certain things the presence of which was more urgent. However, while the focus of this series has been on the stock market since the beginning, I would also like to try to cover the Forex market. The reason for my initial lack of interest in Forex is due to the fact that trades constantly occur in this market, so there is no point in having replay/simulation for testing or training.

You can simply use a demo account for this. However, there are issues unique to this market that are not replicated in the stock market. For this reason, it becomes interesting to show how to make the necessary corrections to the system to adapt the system to other types of markets, e.g. _crypto assets_.

Thus, it will become clear how versatile and suitable the MetaTrader 5 platform can be for many more applications than its creators originally proposed. Only your imagination and knowledge of a specific market will be the limitations to your capabilities.

### Learning some things about Forex

The initial goal of this article is not to cover all the possibilities of Forex trading, but rather to adapt the system so that you can perform at least one market replay. We'll leave simulation for another moment. However, if we don't have ticks and only bars, with a little effort we can simulate possible trades that could happen in the Forex market. This will be the case until we look at how to adapt the simulator. An attempt to work with Forex data inside the system without modifying it leads to a range of errors. Although we try to avoid such errors, they will always happen. However, it is possible to overcome them and thereby create a replay system for the Forex market. But for this we will have to make some adjustments and change some of the concepts that we have been working on so far. I think it's worth it because it will make the system much more flexible to handle much more exotic data.

The article attachment contains a Forex symbol (currency pair). These will be real tics so we can visualize the situation. Without a doubt, Forex is not an easy market to work with in terms of simulation and replay. Although the we see the same basic type of information, Forex has its specific characteristics. Therefore, it is interesting to observe and analyze it.

This market has certain specific features. In order to implement the replay system, I will have to explain some of these features so that you understand what we are talking about and how interesting it may be to know other markets.

### How trades are performed

In the Forex market, trading usually occurs without a real spread between the BID and ASK values. In most cases, these two values may be the same. But how is this possible? How can they be the same? Unlike the stock market, where there is always a spread between BID and ASK, this is not the case in the Forex market. Although sometimes there is a spread and sometimes it is much higher, usually the BID and ASK values can be the same. This can be confusing to those who come from the stock market and want to Forex, as trading strategies often require significant changes.

It should also be noted that the main players in the Forex market are central banks; those who work on the B3 (Brazilian Stock Exchange) have already seen and know very well what the Central Bank sometimes does with the dollar. For this reason, many avoid trading this asset due to fears of possible Central Bank intervention in the currency, as this could quickly turn a previously winning position into a big loser. Many inexperienced traders often go bankrupt during this time, and in some cases even face lawsuits from the stock exchange and stock broker. This could happen through one of the interventions that the central bank may carry out without prior warning and without mercy on those who hold positions.

However, for us this does not matter: we are interested in the program itself. Thus, in the Forex market, the display of prices is based on the BID price value, as seen in Figure 01.

![Figure 01](https://c.mql5.com/2/47/002__3.png)

Figure 01: Chart display on the Forex market

How this differs from, for example, the B3 display system, which uses the price of the last trade executed, can be seen in Figure 02 below, where we have data for a dollar futures contract (at the time of writing this article).

![Figure 02](https://c.mql5.com/2/47/003__3.png)

Figure 02: Mini dollar contract traded on the Brazilian Stock Exchange (B3).

The replay/simulation system was developed to promote the use of this type of analysis. That is, when we use the last traded price, we will have a difference in how the data is laid out in the traded tick file. Not only that, but we can even have a big difference in the kind of information that will actually be available in the tick file or in the 1 minute bars. Because of these variations, we will now focus solely on looking at how to perform replay, since the simulation involves other, even more complex problems. However, as mentioned at the beginning of this article: it is possible to use 1-minute bar data to simulate what likely happened during the trade. Without talking only about theory, let's study the difference in information between Forex and the stock market in the case of B3, that is, the Brazilian Stock Exchange, for which the replay /simulation system was originally developed. In Figure 03 we have information about one of the Forex currency pairs.

![Figure 03](https://c.mql5.com/2/47/001__7.png)

Figure 03: Information about real trades on the Forex market

Figure 04 shows the same type of information, but this time from one of the mini dollar futures contracts that are traded on the B3 (Brazilian Stock Exchange).

![Figure 04](https://c.mql5.com/2/47/004__4.png)

Figure 04: Real tick information in B3

It's completely different. There is no last price or trading volume in the Forex market. On B3 these values are available, and many trading models use trading volume and last deal price. Everything that has been said so far is simply to show that the way the system is built does not allow it to serve other types of markets without making some significant changes. I considered splitting the issue from a market perspective, but for some reason that wouldn't be practical. Not from a programming point of view, since such a separation would greatly simplify programming, but from a usability point of view, since we will always have to adapt to one market or another. What we can do is try to find a middle ground, but this won't without effort. I will try to minimize these difficulties as much as possible, since I do not want and do not intend to recreate the entire system from scratch.

### Starting implementation to cover Forex

The first thing we need to do is fix the floating point numbering system. But in the previous article we already made some changes concerning the floating point system. And it is not suitable for Forex. This is because precision is limited to four decimal places, and we need to tell the system that we are going to use a set with more decimal places. We need to fix this to avoid other problems in the future. This fix is done in the following code:

```
C_Replay(const string szFileConfig)
    {
        m_ReplayCount = 0;
        m_dtPrevLoading = 0;
        m_Ticks.nTicks = 0;
        Print("************** Serviço Market Replay **************");
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
        CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, 8);
        m_IdReplay = (SetSymbolReplay(szFileConfig) ? 0 : -1);
    }
```

Right here we will inform MetaTrader 5 that we need more decimal places in our floating point system. In this case we will use 8 decimal places, which is more than enough to cover a wide range of conditions. One important detail: B3 handles 4 decimal places well, but to work in Forex we need 5 decimal places. By using 8, we make the system free. However, this will not always be the case. We'll have to change it later because of a detail I can't explain yet, but for now it's enough.

Once this is done, we will begin to make our lives easier in some way. We will start by considering the following scenario: the preview bars that will be on our chart are 1-minute bars. As for the ticks, they will be real ticks present in another file. Thus, we will get back to the most basic system, although we will quickly work our way up to a more comprehensive system.

### Working with the Basics

In order not to force the user to select the type of market to analyze, i.e., the type of market from which the data for replay comes, we will take advantage of the fact that in some cases we will not have the value of the last price or trading volume, and in other cases we will have it. The enable the system to check this for us, we will have to add something to the code.

First, we add the following:

```
class C_FileTicks
{
    protected:
        enum ePlotType {PRICE_EXCHANGE, PRICE_FOREX};
        struct st00
        {
            MqlTick   Info[];
            MqlRates  Rate[];
            int       nTicks,
                      nRate;
            bool      bTickReal;
            ePlotType ModePlot;
        }m_Ticks;

//... The rest of the class code....
```

This enumeration will help us avoid confusion in some areas. Essentially, we narrow it down to two types of markets. You will soon understand the reason for this. To avoid unnecessary function calls, let's add a new variable to the system. Now things are starting to take shape, but we need the system to be able to recognize when we are using a particular display method. I don't want to complicate the user's life with such problems, so let's make a small change to the code below:

```
inline bool ReadAllsTicks(const bool ToReplay)
        {
#define def_LIMIT (INT_MAX - 2)
#define def_Ticks m_Ticks.Info[m_Ticks.nTicks]

                string   szInfo;
                MqlRates rate;

                Print("Loading ticks for replay. Please wait...");
                ArrayResize(m_Ticks.Info, def_MaxSizeArray, def_MaxSizeArray);
                m_Ticks.ModePlot = PRICE_FOREX;
                while ((!FileIsEnding(m_File)) && (m_Ticks.nTicks < def_LIMIT) && (!_StopFlag))
                {
                        ArrayResize(m_Ticks.Info, m_Ticks.nTicks + 1, def_MaxSizeArray);
                        szInfo = FileReadString(m_File) + " " + FileReadString(m_File);
                        def_Ticks.time = StringToTime(StringSubstr(szInfo, 0, 19));
                        def_Ticks.time_msc = (def_Ticks.time * 1000) + (int)StringToInteger(StringSubstr(szInfo, 20, 3));
                        def_Ticks.bid = StringToDouble(FileReadString(m_File));
                        def_Ticks.ask = StringToDouble(FileReadString(m_File));
                        def_Ticks.last = StringToDouble(FileReadString(m_File));
                        def_Ticks.volume_real = StringToDouble(FileReadString(m_File));
                        def_Ticks.flags = (uchar)StringToInteger(FileReadString(m_File));
                        m_Ticks.ModePlot = (def_Ticks.volume_real > 0.0 ? PRICE_EXCHANGE : m_Ticks.ModePlot);
                        if (def_Ticks.volume_real > 0.0)
                        {
                                ArrayResize(m_Ticks.Rate, (m_Ticks.nRate > 0 ? m_Ticks.nRate + 2 : def_BarsDiary), def_BarsDiary);
                                m_Ticks.nRate += (BuiderBar1Min(rate, def_Ticks) ? 1 : 0);
                                m_Ticks.Rate[m_Ticks.nRate] = rate;
                        }
                        m_Ticks.nTicks++;
                }
                FileClose(m_File);
                if (m_Ticks.nTicks == def_LIMIT)
                {
                        Print("Too much data in the tick file.\nCannot continue...");
                        return false;
                }
                return (!_StopFlag);
#undef def_Ticks
#undef def_LIMIT
        }
```

Let's start by specifying that the display type will be a FOREX model. However, if a tick containing traded volume is found while reading the tick file, this model will be changed to EXCHANGE type. It is important to understand that this happens without any user intervention. But here an important point arises: this system will only work for those cases when reading is performed during replay startup. In the case of simulation, the situation will be different. We won't deal with the simulation for now.

For this reason, until we create the simulation code, **DO NOT** use only bar files. It is imperative to use tick files, either real or simulated. There are ways to create simulated tick files, but I won't go into detail as it is beyond the scope of this article. However, we must not leave users completely in the dark. Even though the system can analyze the data, we can show the user what kind of display is being used. Thus, by opening the Symbol window, we can check the display form. Exactly as shown in Figure 01 and Figure 02.

To make this possible, we need to add a few more things to our code. These include the lines shown below:

```
datetime LoadTicks(const string szFileNameCSV, const bool ToReplay = true)
    {
        int      MemNRates,
                 MemNTicks;
        datetime dtRet = TimeCurrent();
        MqlRates RatesLocal[];

        MemNRates = (m_Ticks.nRate < 0 ? 0 : m_Ticks.nRate);
        MemNTicks = m_Ticks.nTicks;
        if (!Open(szFileNameCSV)) return 0;
        if (!ReadAllsTicks(ToReplay)) return 0;
        if (!ToReplay)
        {
            ArrayResize(RatesLocal, (m_Ticks.nRate - MemNRates));
            ArrayCopy(RatesLocal, m_Ticks.Rate, 0, 0);
            CustomRatesUpdate(def_SymbolReplay, RatesLocal, (m_Ticks.nRate - MemNRates));
            dtRet = m_Ticks.Rate[m_Ticks.nRate].time;
            m_Ticks.nRate = (MemNRates == 0 ? -1 : MemNRates);
            m_Ticks.nTicks = MemNTicks;
            ArrayFree(RatesLocal);
        }else
        {
            CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_TRADE_CALC_MODE, m_Ticks.ModePlot == PRICE_EXCHANGE ? SYMBOL_CALC_MODE_EXCH_STOCKS : SYMBOL_CALC_MODE_FOREX);
            CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_CHART_MODE, m_Ticks.ModePlot == PRICE_EXCHANGE ? SYMBOL_CHART_MODE_LAST : SYMBOL_CHART_MODE_BID);
        }
        m_Ticks.bTickReal = true;

        return dtRet;
    };
```

With the addition of these highlighted lines, we will have more adequate information about the symbol. But I want you to pay attention to what I'm about to explain, because if you don't understand any of the concepts presented here, you'll think the system is playing a trick on you. Figure 05 shows what the above code does. Try this with other symbols to make things clearer. The first point relates to this particular line. It will show what type of calculation we can or will use for the symbol. It's true that there are more ways to calculate symbols, but since the idea here is to keep it as simple as possible and still let it work, we've boiled it down to just these two types of calculations.

If you want to get more details or implement other types of calculations, you can refer to the documentation and view [SYMBOL\_TRADE\_CALC\_MODE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants). There you will find a detailed description of each calculation mode. Here we will work only in the simplest way. One thing that might drive you crazy is that second highlighted line. In this line we simply indicate the type of display mode. Basically, there are only these two types. The problem here is not in this line itself, but in the configuration file, or more precisely in how the configuration file is read at the current stage of development.

Currently the system is written in such a way that if we read the bar file and then the tick file, we will have problems. It's not because we did something wrong, on the contrary, we are following the correct logic. However, the fact that this line is executed after the bar file has been loaded into the chart causes any content present in the chart to be removed. This issue is caused by the data not being buffered as it comes directly to the chart, so we'll have to look into that. However, this solution will be discussed later in the article.

In my personal opinion, if the system was only for my personal use, everything would be done differently. We would simply generate some type of alert so that the tick file is read before the bars file. But since the system will often be used by people who do not have programming knowledge, it seems to me appropriate to solve such a problem. This is even good, since you will learn how to do a very interesting and quite useful trick.

![Figure 05](https://c.mql5.com/2/47/005__1.png)

Figure 05: Display of automatic tick reading recognition

Now that the system can identify some things, we need it to somehow adapt to our display system.

```
class C_Replay : private C_ConfigService
{
    private :
        long         m_IdReplay;
        struct st01
        {
            MqlRates Rate[1];
            datetime memDT;
            int      delay;
        }m_MountBar;
        struct st02
        {
            bool     bInit;
        }m_Infos;

// ... The rest of the class code....
```

To configure everything and save the settings, we will first add this variable. It will show whether the system has been fully initialized. But pay attention to how we will use it. First we initialize it with the appropriate value. It is done in the following fragment of the code:

```
C_Replay(const string szFileConfig)
{
    m_ReplayCount = 0;
    m_dtPrevLoading = 0;
    m_Ticks.nTicks = 0;
    m_Infos.bInit = false;
    Print("************** Serviço Market Replay **************");
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
    CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, 8);
    m_IdReplay = (SetSymbolReplay(szFileConfig) ? 0 : -1);
}
```

Why do we initialize the variable to false? The reason is that the system is not initialized yet, but once we load the chart on the screen, it will be initialized. You might think that we would specify this in the function that initializes the chart. Right? No. We can't do this in the function that initializes the chart. We have to wait for this initialization function to complete and once the next function is called, we can indicate that the system is initialized. But why use this variable? Does the system really not know whether it has already been initialized or not? Why do we have to use the variable? The system knows that it is initialized, but we need this variable for another reason. To be clear, let's look at a function that changes its state.

```
bool LoopEventOnTime(const bool bViewBuider, const bool bViewMetrics)
        {
                u_Interprocess Info;
                int iPos, iTest;

                if (!m_Infos.bInit)
                {
                        ChartSetInteger(m_IdReplay, CHART_SHOW_ASK_LINE, m_Ticks.ModePlot == PRICE_FOREX);
                        ChartSetInteger(m_IdReplay, CHART_SHOW_BID_LINE, m_Ticks.ModePlot == PRICE_FOREX);
                        ChartSetInteger(m_IdReplay, CHART_SHOW_LAST_LINE, m_Ticks.ModePlot == PRICE_EXCHANGE);
                        m_Infos.bInit = true;
                }
                iTest = 0;
                while ((iTest == 0) && (!_StopFlag))
                {
                        iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);
                        iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value) ? iTest : -1);
                        iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);
                        if (iTest == 0) Sleep(100);
                }
                if ((iTest < 0) || (_StopFlag)) return false;
                AdjustPositionToReplay(bViewBuider);
                m_MountBar.delay = 0;
                while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))
                {
                        CreateBarInReplay(bViewMetrics, true);
                        iPos = (int)(m_ReplayCount < m_Ticks.nTicks ? m_Ticks.Info[m_ReplayCount].time_msc - m_Ticks.Info[m_ReplayCount - 1].time_msc : 0);
                        m_MountBar.delay += (iPos < 0 ? iPos + 1000 : iPos);
                        if (m_MountBar.delay > 400)
                        {
                                if (ChartSymbol(m_IdReplay) == "") break;
                                GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                                if (!Info.s_Infos.isPlay) return true;
                                Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                                Sleep(m_MountBar.delay - 20);
                                m_MountBar.delay = 0;
                        }
                }
                return (m_ReplayCount == m_Ticks.nTicks);
        }
```

This function described above is causing us some headaches. This is why we need a variable that indicates whether the system is initialized or not. Please note: the first time you run this function, the variable will still indicate that the system has **NOT** been fully initialized. At this time, its initialization will be completed. Here we make sure that the correct price lines are displayed on the screen. If the system defines the display mode as FOREX, the BID and ASK price lines will be displayed, and the last price line will be hidden. The opposite will happen if the display mode is EXCHANGE. In this case, the BID and ASK price lines will be hidden and the latest price line will be displayed.

Everything would be very nice and good if it weren't for the fact that some users prefer to set a different setting. Even if they work in the EXCHANGE display style, they like to display the BID or ASK lines, and in some cases both lines. Therefore, if the user pauses the system after configuring it to their liking and then restarts the system, the system will ignore the user's settings and revert to its internal settings. However, by specifying that the system has already been initialized (and using a variable for this), it will not return to the internal settings, but will remain as the user just configured it.

But then the question arises: why not make this setting in the ViewReplay function? The reason is that the chart did not actually receive such a setting for the lines. We will also have to solve other rather unpleasant problems. We need additional variables to help us. Simple programming will not solve all problems.

### Displaying bars

We have finally reached the point where we can display bars on the chart. However, if you try to do this at this stage, you will encounter range errors in the arrays. Therefore, before actually presenting the bars on the chart, we need to make some corrections to the system.

The first fix is in the function below:

```
void AdjustPositionToReplay(const bool bViewBuider)
        {
                u_Interprocess Info;
                MqlRates       Rate[def_BarsDiary];
                int            iPos, nCount;

                Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
                if ((m_ReplayCount == 0) && (m_Ticks.ModePlot == PRICE_EXCHANGE))
                for (; m_Ticks.Info[m_ReplayCount].volume_real == 0; m_ReplayCount++);
                if (Info.s_Infos.iPosShift == (int)((m_ReplayCount * def_MaxPosSlider * 1.0) / m_Ticks.nTicks)) return;
                iPos = (int)(m_Ticks.nTicks * ((Info.s_Infos.iPosShift * 1.0) / (def_MaxPosSlider + 1)));
                Rate[0].time = macroRemoveSec(m_Ticks.Info[iPos].time);
                if (iPos < m_ReplayCount)
                {
                        CustomRatesDelete(def_SymbolReplay, Rate[0].time, LONG_MAX);
                        CustomTicksDelete(def_SymbolReplay, m_Ticks.Info[iPos].time_msc, LONG_MAX);
                        if ((m_dtPrevLoading == 0) && (iPos == 0)) FirstBarNULL(); else
                        {
                                for(Rate[0].time -= 60; (m_ReplayCount > 0) && (Rate[0].time <= macroRemoveSec(m_Ticks.Info[m_ReplayCount].time)); m_ReplayCount--);
                                m_ReplayCount++;
                        }
                }else if (iPos > m_ReplayCount)
                {
                        if (bViewBuider)
                        {
                                Info.s_Infos.isWait = true;
                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                        }else
                        {
                                for(; Rate[0].time > (m_Ticks.Info[m_ReplayCount].time); m_ReplayCount++);
                                for (nCount = 0; m_Ticks.Rate[nCount].time < macroRemoveSec(m_Ticks.Info[iPos].time); nCount++);
                                CustomRatesUpdate(def_SymbolReplay, m_Ticks.Rate, nCount);
                        }
                }
                for (iPos = (iPos > 0 ? iPos - 1 : 0); (m_ReplayCount < iPos) && (!_StopFlag);) CreateBarInReplay(false, false);
                Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
                Info.s_Infos.isWait = false;
                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
        }
```

This check allows the system to determine whether to skip the first few ticks. The problem is that if we do not check whether we are working with a display similar to the stock exchange, then the loop included in the check will fail. And this will result in a range error. However, by adding a second check, we will pass this stage. If the display mode matches the type used in FOREX, the loop will not be executed. So let's be ready for the next stage.

In the next step we actually insert ticks into the chart. Here, the only thing we have to worry about is telling the system what the bar closing price will be; the rest is handled by the bar simulation function. In this case, it will be the same for both the exchange mode display and the FOREX display. The code for its implementation is shown below:

```
inline void CreateBarInReplay(const bool bViewMetrics, const bool bViewTicks)
    {
#define def_Rate m_MountBar.Rate[0]

        bool bNew;
        MqlTick tick[1];
        static double PointsPerTick = 0.0;

        if (bNew = (m_MountBar.memDT != macroRemoveSec(m_Ticks.Info[m_ReplayCount].time)))
        {
            PointsPerTick = (PointsPerTick == 0.0 ? SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) : PointsPerTick);
            if (bViewMetrics) Metrics();
            m_MountBar.memDT = (datetime) macroRemoveSec(m_Ticks.Info[m_ReplayCount].time);
            def_Rate.real_volume = 0;
            def_Rate.tick_volume = 0;
        }
        def_Rate.close = (m_Ticks.ModePlot == PRICE_EXCHANGE ? (m_Ticks.Info[m_ReplayCount].volume_real > 0.0 ? m_Ticks.Info[m_ReplayCount].last : def_Rate.close) :
                                                               (m_Ticks.Info[m_ReplayCount].bid > 0.0 ? m_Ticks.Info[m_ReplayCount].bid : def_Rate.close));
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

This line does exactly that. It will generate the bar closing price depending on the type of display used. Otherwise the function remains the same as before. In this way, we will be able to cover the FOREX mapping system and perform replay with the data presented in the tick file. We don't have the ability to do simulations yet.

You might think that the system is already finished, but it is not. We still have two problems and they are very relevant before we even think about FOREX simulation.

The first problem is that we need the replay/simulation configuration file to have creation logic. In many cases, this will force the user to adapt to the system without real need. Besides this, we have another problem. Timer system. The reason for the second problem is that we may be dealing with a symbol or trading time where the symbol may remain inactive for hours, or because it is auctioned, or suspended, or for some other reason. But that doesn't matter, we also need to fix this timer issue.

Since the second problem is more pressing and urgent, let's start with it.

### Fixing the timer

The biggest problem with the system and timer is that the system cannot handle the conditions that sometimes occur in some symbols. This condition may be extremely low liquidity, suspension of transactions, auctions or other reasons. If for some reason the tick file tells the timer that the symbol should sleep for, say, 15 minutes, the system will be completely locked during that time.

In the real market this is solved in a special way. Typically the platform will notify us if a symbol is not trading, but even if the platform does not provide us with this information, we will still receive notification from the market. More experienced traders, looking at a symbol, will notice that something has happened and nothing needs to be done during this period. However, if the market replay is used, this situation can be problematic. We must allow the user to close or attempt to change the position in which the replay or simulation is running.

This type of solution has been used before, in fact, a control indicator can be used for this. However, we had no precedents that would force us to take such a drastic measure, even to the point of repeating the situation where a symbol was put up for auction, with very low liquidity, or even suspended due to a relevant event. All this creates the so-called liquidity risk, but in the replay/simulation system we can easily avoid this risk and continue our research. However, to do this effectively, we need to change the way the timer works.

Next we have a new display system loop. I know the code may seem confusing at first glance.

```
bool LoopEventOnTime(const bool bViewBuider, const bool bViewMetrics)
    {
        u_Interprocess Info;
        int iPos, iTest;

        iTest = 0;
        while ((iTest == 0) && (!_StopFlag))
        {
            iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);
            iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value) ? iTest : -1);
            iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);
            if (iTest == 0) Sleep(100);
        }
        if ((iTest < 0) || (_StopFlag)) return false;
        AdjustPositionToReplay(bViewBuider);
        iPos = 0;
        while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))
        {
            iPos = (int)(m_ReplayCount < m_Ticks.nTicks ? m_Ticks.Info[m_ReplayCount].time_msc - m_Ticks.Info[m_ReplayCount - 1].time_msc : 0);
            m_MountBar.delay += (iPos < 0 ? iPos + 1000 : iPos);
            iPos += (int)(m_ReplayCount < (m_Ticks.nTicks - 1) ? m_Ticks.Info[m_ReplayCount + 1].time_msc - m_Ticks.Info[m_ReplayCount].time_msc : 0);
            CreateBarInReplay(bViewMetrics, true);
            if (m_MountBar.delay > 400)
            while ((iPos > 200) && (!_StopFlag))
            {
                if (ChartSymbol(m_IdReplay) == "") break;
                if (ChartSymbol(m_IdReplay) == "") return false;
                GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                if (!Info.s_Infos.isPlay) return true;
                Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
                Sleep(195);
                iPos -= 200;
                Sleep(m_MountBar.delay - 20);
                m_MountBar.delay = 0;
            }
        }
        return (m_ReplayCount == m_Ticks.nTicks);
    }
```

All removed parts have been replaced with other codes. Thus, we can solve the first problem related to the display of bars. Previously we used backward calculation, but now we are going to use forward calculation. This prevents some weird things happening on the chart while we are in idle mode. Please note that when we start the system, it always waits for some time before displaying the tick. Previously it was done exactly the opposite ( _**MY FAULT**_). Since the time can now be very long, up to several hours, we have another way to control the timer. To better understand, you should know that previously the system remained in standby mode until the time had completely expired. If we tried to make any changes, such as closing the chart or trying to change the execution point, the system simply did not respond as expected. This happened because in those files that were attached earlier, there was no risk of using a set where the asset remained "non-trading" for a long time. But when I started writing this article, the system showed this error. Therefore, corrections have been made.

The timer will now run until the period is greater than or equal to 200 milliseconds. You can change this value, but be careful to change it at other points as well. We have started to improve the situation, but still, we need to do one more thing before we finish. If you close the chart, the system will exit the loop. Now go back to the calling program. This ensures that everything will work, at least in theory. This is because the user may interact with the system again during the idle period. The remaining functions have remained virtually untouched, so everything continues to work as before. However, if during a period of time you ask the control indicator to change its position, then this will now become possible, but before this was not possible. This is very important because some assets can go dormant and remain there for quite a long period. This is acceptable in real trading, but **NOT** in a replay/simulation system.

### Conclusion

Despite all the failures, you can now start experimenting with using FOREX data in the system. Let's start with this version of the replay/simulation system. To support this, I provide to some FOREX data in the attachment. The system still needs improvements in some areas. Since I don't want to go into detail just yet (as it may require radical changes to some of the things shown in this article), I'll end the modifications here.

In the next article I will address some additional unresolved issues. While these issues do not prevent you from using the system, if you intend to use it with FOREX market data, you will notice that some things are not represented correctly. We need to tweak them slightly, but that will be seen in the next article.

The configuration file issue, as well as other related issues, still needs to be resolved. Since the system works properly, allowing you to replicate data obtained from the foreign exchange market, the solution to these lingering problems will be postponed until the next article.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11144](https://www.mql5.com/pt/articles/11144)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11144.zip "Download all attachments in the single ZIP archive")

[Market\_Replay\_yvg20.zip](https://www.mql5.com/en/articles/download/11144/market_replay_yvg20.zip "Download Market_Replay_yvg20.zip")(14386.04 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/458974)**

![Filtering and feature extraction in the frequency domain](https://c.mql5.com/2/62/power_spectrumf_avatar.png)[Filtering and feature extraction in the frequency domain](https://www.mql5.com/en/articles/13881)

In this article we explore the application of digital filters on time series represented in the frequency domain so as to extract unique features that may be useful to prediction models.

![Data Science and Machine Learning (Part 16): A Refreshing Look at Decision Trees](https://c.mql5.com/2/62/1midjourney_image_13862_46_406__3_logo.png)[Data Science and Machine Learning (Part 16): A Refreshing Look at Decision Trees](https://www.mql5.com/en/articles/13862)

Dive into the intricate world of decision trees in the latest installment of our Data Science and Machine Learning series. Tailored for traders seeking strategic insights, this article serves as a comprehensive recap, shedding light on the powerful role decision trees play in the analysis of market trends. Explore the roots and branches of these algorithmic trees, unlocking their potential to enhance your trading decisions. Join us for a refreshing perspective on decision trees and discover how they can be your allies in navigating the complexities of financial markets.

![MQL5 Wizard Techniques you should know (Part 09): Pairing K-Means Clustering with Fractal Waves](https://c.mql5.com/2/62/midjourney_image_13915_50_439__5-logo.png)[MQL5 Wizard Techniques you should know (Part 09): Pairing K-Means Clustering with Fractal Waves](https://www.mql5.com/en/articles/13915)

K-Means clustering takes the approach to grouping data points as a process that’s initially focused on the macro view of a data set that uses random generated cluster centroids before zooming in and adjusting these centroids to accurately represent the data set. We will look at this and exploit a few of its use cases.

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5):  Bollinger Bands On Keltner Channel — Indicators Signal](https://c.mql5.com/2/61/rj-article-image-60x60.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5): Bollinger Bands On Keltner Channel — Indicators Signal](https://www.mql5.com/en/articles/13861)

The Multi-Currency Expert Advisor in this article is an Expert Advisor or Trading Robot that can trade (open orders, close orders and manage orders for example: Trailing Stop Loss and Trailing Profit) for more than one symbol pair from only one symbol chart. In this article we will use signals from two indicators, in this case Bollinger Bands® on Keltner Channel.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/11144&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069014127864643560)

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