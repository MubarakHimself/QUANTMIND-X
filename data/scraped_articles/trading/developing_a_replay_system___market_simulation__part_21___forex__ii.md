---
title: Developing a Replay System — Market simulation (Part 21): FOREX (II)
url: https://www.mql5.com/en/articles/11153
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:03:20.332381
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=jgtkkshvbnsirsmmmchsjmuvhpcubmpo&ssn=1769180599542057094&ssn_dr=0&ssn_sr=0&fv_date=1769180599&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11153&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20%E2%80%94%20Market%20simulation%20(Part%2021)%3A%20FOREX%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918059915814357&fz_uniq=5068991961538428854&sv=2552)

MetaTrader 5 / Tester


### Introduction

In the previous article, " [Developing a Replay System — Market simulation (Part 20): FOREX (I)](https://www.mql5.com/en/articles/11144), we started to assemble, or rather, to adapt the replay/simulation system. This is done to allow the use of market data, for example, FOREX, in order to at least be able to make a replay for this market.

To develop this potential, it was necessary to make a number of changes and adjustments to the system, but in this article we are considering not only the forex market. It can be seen that we have been able to cover a wide range of possible market types, as we can now not only see the last traded price, but also use a BID-based display system, which is quite common among some specific types of assets or markets.

But that's not all. We have also implemented a way to prevent the replay/simulation system from getting "locked" when tick data shows that there were no transactions for an asset for a certain period. The system was in timed mode and was released after a certain time to be used again. We currently do not encounter such problems, although new challenges and problems still remain, awaiting solution and thus further improvement of our replay/simulation system.

### Solving the problem with the configuration file

Let's start by making the configuration file more user-friendly. You may have noticed that it has become a little more difficult to work with. I don't want the user to be upset when trying to set up replay or simulation if the chart is left completely blank and there is no access to control indicators while there is no message indicating there was an error when starting the service.

As a result, when attempting to use the replay/simulation service, a situation similar to the one shown in Figure may occur. 01:

![Figure 01](https://c.mql5.com/2/47/003__4.png)

Figure 01: Result of failure in boot sequence

This error is not due to misuse or misunderstanding of how to operate the system. It appeared because all data placed in the custom asset has been completely removed. This could occur either as a result of a change in the configuration of the asset, or as a result of the destruction of data existing in it.

But in any case, this failure will occur more often if we load tick trades after loading 1-minute bars. In this case, the system will understand that it needs to change the way it displays the price, and by making this change, all bars will be deleted, as shown in the previous article.

In order to solve this problem, we must first declare the loading of ticks before loading the previous bars. This solves the problem, but at the same time forces the user to follow some structure in the configuration file, which, personally, does not make much sense to me. The reason is that by designing a program that is responsible for analyzing and executing what is in the configuration file, we can allow the user to declare things in any order, as long as certain syntax is respected.

basically, there is only one way to solve this problem. The way we create this path allows us to generate slight variations, but basically it is always the same. But this is from a programming point of view. In short, we need to read the entire configuration file and then access the necessary elements in the order in which they should be available so that everything works in perfect harmony.

One variation of this technique is the use of functions [FileSeek](https://www.mql5.com/en/docs/files/fileseek) and [FileTell](https://www.mql5.com/en/docs/files/filetell) to be able to read in the required sequence. We can access the file just once and then access the desired elements directly in memory, or we can read the file piece by piece to do the same work that would be done if it were already in memory. But there is also another option.

Personally, I prefer to load it completely into memory and then read it in parts so that I have the specific sequence needed to load it and not get the result shown in Fig. 01. So we will use the following technique: We will load the configuration file into memory and then load the data so that the asset can be used in replay or simulation.

To create the ability for programming to override a sequence in a configuration file, we must first create a set of variables. But I'm not going to program everything from scratch. Instead, I will take advantage of the functions already included in MQL5.

More precisely, we will use the standard MQL5 library. Well, we will use only a small part of what is presented in this library. The big advantage of this approach is that the features it contains have already been thoroughly tested. If at some point in the future you make improvements to your program, whatever they may be, the system will automatically accept them. This way, the programming process becomes much easier since we will need less time during the testing and optimization phase.

Let's now see what we need. This is shown in the code below:

```
#include "C_FileBars.mqh"
#include "C_FileTicks.mqh"
//+------------------------------------------------------------------+
#include <Arrays\ArrayString.mqh>
//+------------------------------------------------------------------+
class C_ConfigService : protected C_FileTicks
{
// ... Internal class code ....
}
```

Here we will use the include file that is provided in MQL5. It is quite sufficient for our purposes.

After this, we will need private global variables for the class:

```
    private :
//+------------------------------------------------------------------+
        enum eTranscriptionDefine {Transcription_INFO, Transcription_DEFINE};
        string m_szPath;
        struct st001
        {
            CArrayString *pTicksToReplay, *pBarsToTicks, *pTicksToBars, *pBarsToPrev;
        }m_GlPrivate;
```

These variables, located inside the structure, are actually pointers to the class that we are going to use. It is important to note that working with pointers is different from working with variables, because pointers are structures that point to a location in memory, more precisely, an address, while variables contain only values.

An important detail in this whole story is that by default and to make life easier for many programmers (especially those who are just starting to learn programming), MQL5 does not actually use exactly the same concept of pointers as in C/C++. Those who have programmed in C/C++ know how useful but at the same time dangerous and confusing pointers can be. However, in MQL5, the developers have tried to eliminate much of the confusion and dangers associated with the use of pointers. For practical purposes, it should be noted that we are actually going to use pointers to access the CArrayString class.

If you want to know more about the methods of the CArrayString class, please refer to its documentation. Just click on [CArrayString](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraystring) and see what is already available for our use. It is quite suitable for our purposes.

Since we use pointers to access objects, they need to be initialized first, which is what we did in the following code.

```
bool SetSymbolReplay(const string szFileConfig)
   {

// ... Internal code ...

      m_GlPrivate.pTicksToReplay = m_GlPrivate.pTicksToBars = m_GlPrivate.pBarsToTicks = m_GlPrivate.pBarsToPrev = NULL;
// ... The rest of the code ...

   }
```

Perhaps the biggest mistake programmers make when using pointers, especially in C/C++, is that they try to use the pointer data before it has been initialized. The danger with pointers is that they never point to an area of memory which we can use before they are initialized. Trying to read, and especially write, in an unfamiliar area of memory most likely means that we are writing in a critical part or reading the wrong information. In this case, the system often crashes, leading to various problems. Therefore, be very careful when using pointers in your programs.

With all precautions in mind, let's see how it gets in out replay/simulation system configuration function. The entire function is shown in the code below:

```
bool SetSymbolReplay(const string szFileConfig)
   {
#define macroFileName ((m_szPath != NULL ? m_szPath + "\\" : "") + szInfo)
      int    file,
             iLine;
      char   cError,
             cStage;

          string szInfo;

          bool   bBarsPrev;


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
      m_GlPrivate.pTicksToReplay = m_GlPrivate.pTicksToBars = m_GlPrivate.pBarsToTicks = m_GlPrivate.pBarsToPrev = NULL;

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

                         if (m_GlPrivate.pBarsToPrev == NULL) m_GlPrivate.pBarsToPrev = new CArrayString();
                     (*m_GlPrivate.pBarsToPrev).Add(macroFileName);
                     pFileBars = new C_FileBars(macroFileName);
                     if ((m_dtPrevLoading = (*pFileBars).LoadPreView()) == 0) cError = 3; else bBarsPrev = true;
                     delete pFileBars;
                     break;
                  case 2:
                     if (m_GlPrivate.pTicksToReplay == NULL) m_GlPrivate.pTicksToReplay = new CArrayString();
                     (*m_GlPrivate.pTicksToReplay).Add(macroFileName);
                     if (LoadTicks(macroFileName) == 0) cError = 4;
                     break;
                  case 3:
                     if (m_GlPrivate.pTicksToBars == NULL) m_GlPrivate.pTicksToBars = new CArrayString();
                     (*m_GlPrivate.pTicksToBars).Add(macroFileName);
                     if ((m_dtPrevLoading = LoadTicks(macroFileName, false)) == 0) cError = 5; else bBarsPrev = true;
                     break;
                  case 4:
                     if (m_GlPrivate.pBarsToTicks == NULL) m_GlPrivate.pBarsToTicks = new CArrayString();
                     (*m_GlPrivate.pBarsToTicks).Add(macroFileName);
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
         Cmd_TicksToReplay(cError);
         Cmd_BarsToTicks(cError);
         bBarsPrev = (Cmd_TicksToBars(cError) ? true : bBarsPrev);
         bBarsPrev = (Cmd_BarsToPrev(cError) ? true : bBarsPrev);
         switch(cError)
         {
            case 0:
               if (m_Ticks.nTicks <= 0)
               {
                  Print("No ticks to use. Closing the service...");
                  cError = -1;
               }else if (!bBarsPrev) FirstBarNULL();
               break;
            case 1  : Print("Command in line ", iLine, " could not be recognized by the system...");    break;
            case 2  : Print("The contents of the line are unexpected for the system ", iLine);                  break;
            default : Print("Error occurred while accessing one of the specified files...");
         }

         return (cError == 0 ? !_StopFlag : false);
#undef macroFileName
      }
```

The crossed out parts in the previous code were removed because we no longer need them here. They were placed in another place. But before looking at their new location, pay attention to how the code works at this stage.

You can see that there's some pretty repetitive code here, even though we'll always be accessing different pointers.

This code is intended for the **new** operator to create an area of memory in which the class will exist. At the same time, this operator initializes the class. Since there is no initialization constructor, the class is created and initialized in all cases with default values.

After the class is initialized, the pointer value will no longer be **NULL**. The pointer will only refer to the first element, which would look like a list if we were not talking about arrays here. Now that the pointer is pointing to the correct location in memory, we can use one of the class methods to append a string to it.

Note that this is done almost transparently. We don't need to know where and how the information we add is organized in the class. What's really important is that every time we call the method, the class store data for us.

Please note that we have never referenced the files that will be listed in the configuration file. The only thing done in real-time reading is setting up the basic operating information. Reading the data, be it histograms or tick data, will be done later.

Once the system has finished reading the configuration file, we move on to the next step. Now comes the main question that you should definitely think about when adding or using this system. So pay close attention to what I'm about to explain, as it may be important for your specific system. For us it is not important.

Here we have 4 functions, and the order in which they appear affects the final result. For now, don't worry about the code of each separate function. Here it is necessary to take into account the order of their appearance in the code, because depending on the order of their appearance, the system will display the graph in one way or another. This may seem confusing to you, but let's try to understand how the situation will develop. We need to know which order is most appropriate depending on the type of database we are going to use.

In the order used in the code, the system performs the following actions. First, it reads the actual data ticks. Next, it converts data into ticks using a modeling process. Then it creates bars, sticking to the initial conversion from ticks to bars, and finally loads 1-minute bars.

The data is read in exactly the order described above. If we want to change this order, we will have to change the order in which these functions are called. However, you should not read previous bars before reading the ticks that we will model. This is due to the fact that reading ticks for use in replay or reading bars for use in a tester results in the removal of everything that is on the chart. This is related to the details discussed in the previous article. But the fact that the order of use or reading is determined in the system, and not in the configuration file, makes it possible to declare bars before ticks and at the same time allow the system to cope with possible problems.

However, the matter is not that simple. By doing so, we allow the system to determine the order of events, or rather, the order in which they are read. However, the way everything is done when we read all the data in memory and then interpret it, makes it difficult for the system to report the exact line where reading has failed. We still have the name of the file data from which could not be accessed.

To solve this inconvenience of the system not being able to report the exact line where the error occurred, we will have to make a small and fairly simple addition to the code. Remember: We do not want to reduce the functionality of our system. Instead, we want to increase and expand it to cover as many cases as possible.

Before we implement the solution, I want you to understand why we made this decision.

To solve this definition problem, during the loading phase we must store both the name of the file and the line on which it was declared. This solution is actually quite simple. All we will need to do is declare another set of variables using the CArrayInt class. This class will store strings corresponding to each of the file names.

While at first glance this seems like a pretty nice solution since we'll be using the standard library, it's a bit expensive because it forces us to add many more variables than would be necessary if we were to develop our own solution. We will use the same principle as the classes used in the standard library, but with the ability to simultaneously work with a large amount of data.

You might think that in this way we are complicating our project, because we can use an already tested and implemented solution available in the standard library, and personally I agree with this idea in many cases. But here it does not make much sense, since we do not need all these methods available in the standard library. You only need two of them. Thus, the costs of implementation compensate for the additional labor costs. So a new class appears in our project: C\_Array.

Its full code is shown below:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
class C_Array
{
    private :
//+------------------------------------------------------------------+
        string  m_Info[];
        int     m_nLine[];
        int     m_maxIndex;
//+------------------------------------------------------------------+
    public  :
        C_Array()
            :m_maxIndex(0)
        {};
//+------------------------------------------------------------------+
        ~C_Array()
        {
            if (m_maxIndex > 0)
            {
                ArrayResize(m_nLine, 0);
                ArrayResize(m_Info, 0);
            }
        };
//+------------------------------------------------------------------+
        bool Add(const string Info, const int nLine)
        {
            m_maxIndex++;
            ArrayResize(m_Info, m_maxIndex);
            ArrayResize(m_nLine, m_maxIndex);
            m_Info[m_maxIndex - 1] = Info;
            m_nLine[m_maxIndex - 1] = nLine;

            return true;
        }
//+------------------------------------------------------------------+
        string At(const int Index, int &nLine) const
        {
            if (Index >= m_maxIndex)
            {
                nLine = -1;
                return "";
            }
            nLine = m_nLine[Index];
            return m_Info[Index];
        }
//+------------------------------------------------------------------+
};
//+------------------------------------------------------------------+
```

Pay attention to how simple and compact it is, but at the same time it fits perfectly the purpose we need. Using this class, we will be able to store both the line and the name of the file containing information needed for replay or simulation. All this data is declared in the configuration file. Thus, by adding this class to our project, we maintain the same level of functionality, or rather, increase it, since now we can also replay data coming from a market similar to the Forex market.

Some changes need to be made to the code of the class responsible for setting up the replay/simulation. These changes begin in the following code:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "C_FileBars.mqh"
#include "C_FileTicks.mqh"
#include "C_Array.mqh"
//+------------------------------------------------------------------+
class C_ConfigService : protected C_FileTicks
{
        protected:
//+------------------------------------------------------------------+
                datetime m_dtPrevLoading;
                int      m_ReplayCount;
//+------------------------------------------------------------------+
inline void FirstBarNULL(void)
                {
                        MqlRates rate[1];

                        for(int c0 = 0; m_Ticks.Info[c0].volume_real == 0; c0++)
                                rate[0].close = m_Ticks.Info[c0].last;
                        rate[0].open = rate[0].high = rate[0].low = rate[0].close;
                        rate[0].tick_volume = 0;
                        rate[0].real_volume = 0;
                        rate[0].time = m_Ticks.Info[0].time - 60;
                        CustomRatesUpdate(def_SymbolReplay, rate);
                        m_ReplayCount = 0;
                }
//+------------------------------------------------------------------+
        private :
//+------------------------------------------------------------------+
                enum eTranscriptionDefine {Transcription_INFO, Transcription_DEFINE};
                string m_szPath;
                struct st001
                {
                        C_Array *pTicksToReplay, *pBarsToTicks, *pTicksToBars, *pBarsToPrev;
                        int     Line;
                }m_GlPrivate;
```

It was necessary to add only these items to the file. Also, we need to make corrections to the configuration function. It will look like this:

```
bool SetSymbolReplay(const string szFileConfig)
    {
#define macroFileName ((m_szPath != NULL ? m_szPath + "\\" : "") + szInfo)
        int     file;
        char    cError,
                cStage;
        string  szInfo;
        bool    bBarsPrev;

        if ((file = FileOpen("Market Replay\\" + szFileConfig, FILE_CSV | FILE_READ | FILE_ANSI)) == INVALID_HANDLE)
        {
            Print("Failed to open configuration file [", szFileConfig, "]. Closing the service...");
            return false;
        }
        Print("Loading ticks for replay. Please wait....");
        ArrayResize(m_Ticks.Rate, def_BarsDiary);
        m_Ticks.nRate = -1;
        m_Ticks.Rate[0].time = 0;
        cError = cStage = 0;
        bBarsPrev = false;
        m_GlPrivate.Line = 1;
        m_GlPrivate.pTicksToReplay = m_GlPrivate.pTicksToBars = m_GlPrivate.pBarsToTicks = m_GlPrivate.pBarsToPrev = NULL;
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
                        if (m_GlPrivate.pBarsToPrev == NULL) m_GlPrivate.pBarsToPrev = new C_Array();
                        (*m_GlPrivate.pBarsToPrev).Add(macroFileName, m_GlPrivate.Line);
                        break;
                    case 2:
                        if (m_GlPrivate.pTicksToReplay == NULL) m_GlPrivate.pTicksToReplay = new C_Array();
                        (*m_GlPrivate.pTicksToReplay).Add(macroFileName, m_GlPrivate.Line);
                        break;
                    case 3:
                        if (m_GlPrivate.pTicksToBars == NULL) m_GlPrivate.pTicksToBars = new C_Array();
                        (*m_GlPrivate.pTicksToBars).Add(macroFileName, m_GlPrivate.Line);
                        break;
                    case 4:
                        if (m_GlPrivate.pBarsToTicks == NULL) m_GlPrivate.pBarsToTicks = new C_Array();
                        (*m_GlPrivate.pBarsToTicks).Add(macroFileName, m_GlPrivate.Line);
                        break;
                    case 5:
                        if (!Configs(szInfo)) cError = 7;
                        break;
                }
                break;
            };
            m_GlPrivate.Line += (cError > 0 ? 0 : 1);
        }
        FileClose(file);
        Cmd_TicksToReplay(cError);
        Cmd_BarsToTicks(cError);
        bBarsPrev = (Cmd_TicksToBars(cError) ? true : bBarsPrev);
        bBarsPrev = (Cmd_BarsToPrev(cError) ? true : bBarsPrev);
        switch(cError)
        {
            case 0:
                if (m_Ticks.nTicks <= 0)
                {
                    Print("No ticks to use. Closing the service...");
                    cError = -1;
                }else if (!bBarsPrev) FirstBarNULL();
                break;
            case 1  : Print("Command in line ", m_GlPrivate.Line, " could not be recognized by the system..."); break;
            case 2  : Print("The contents of the line is unexpected for the system: ", m_GlPrivate.Line);              break;
            default : Print("Access error occurred in the files indicated in line: ", m_GlPrivate.Line);
        }

        return (cError == 0 ? !_StopFlag : false);
#undef macroFileName

    }
```

Now we can see that we can tell the user that an error occurred and on which line it occurred. The cost of this is practically zero, since, as you can see, the functions remain almost the same as before, only with the addition of a new element. This is very good considering how much work we would have to do if we used the standard library. Please understand me correctly: I'm not saying that you shouldn't use the standard library, I'm just showing that there are times when you need to create your own solution, because in those cases the cost outweighs the work.

We can now look at new functions added to the system, namely the four functions shown above. They are all quite simple, their codes are shown below, and since they all work on the same principle, I will explain them right away so as not to make the process too tedious.

```
inline void Cmd_TicksToReplay(char &cError)
    {
        string szInfo;

        if (m_GlPrivate.pTicksToReplay != NULL)
        {
            for (int c0 = 0; (c0 < INT_MAX) && (cError == 0); c0++)
            {
                if ((szInfo = (*m_GlPrivate.pTicksToReplay).At(c0, m_GlPrivate.Line)) == "") break;
                if (LoadTicks(szInfo) == 0) cError = 4;
            }
            delete m_GlPrivate.pTicksToReplay;
        }
    }
//+------------------------------------------------------------------+
inline void Cmd_BarsToTicks(char &cError)
    {
        string szInfo;

        if (m_GlPrivate.pBarsToTicks != NULL)
        {
            for (int c0 = 0; (c0 < INT_MAX) && (cError == 0); c0++)
            {
                if ((szInfo = (*m_GlPrivate.pBarsToTicks).At(c0, m_GlPrivate.Line)) == "") break;
                if (!BarsToTicks(szInfo)) cError = 6;
            }
            delete m_GlPrivate.pBarsToTicks;
        }
    }
//+------------------------------------------------------------------+
inline bool Cmd_TicksToBars(char &cError)
    {
        bool bBarsPrev = false;
        string szInfo;

        if (m_GlPrivate.pTicksToBars != NULL)
        {
            for (int c0 = 0; (c0 < INT_MAX) && (cError == 0); c0++)
            {
                if ((szInfo = (*m_GlPrivate.pTicksToBars).At(c0, m_GlPrivate.Line)) == "") break;
                if ((m_dtPrevLoading = LoadTicks(szInfo, false)) == 0) cError = 5; else bBarsPrev = true;
            }
            delete m_GlPrivate.pTicksToBars;
        }
        return bBarsPrev;
    }
//+------------------------------------------------------------------+
inline bool Cmd_BarsToPrev(char &cError)
    {
        bool bBarsPrev = false;
        string szInfo;
        C_FileBars      *pFileBars;

        if (m_GlPrivate.pBarsToPrev != NULL)
        {
            for (int c0 = 0; (c0 < INT_MAX) && (cError == 0); c0++)
            {
                if ((szInfo = (*m_GlPrivate.pBarsToPrev).At(c0, m_GlPrivate.Line)) == "") break;
                pFileBars = new C_FileBars(szInfo);
                if ((m_dtPrevLoading = (*pFileBars).LoadPreView()) == 0) cError = 3; else bBarsPrev = true;
                delete pFileBars;
            }
            delete m_GlPrivate.pBarsToPrev;
        }

        return bBarsPrev;
    }
//+------------------------------------------------------------------+
```

But wait... look closely at all four functions above... there is a lot of repeated code... What do we do when we have a lot of repeated code? We try to bring the function down to a more basic level to enable code reuse. I know this may sound stupid or careless on my part, but in this series of articles I not only show how programs are created from scratch, but also how good programmers improve their programs by reducing the amount of maintenance work and creating them.

It's not about how many more years of programming experience you have, but the fact that you, even if you're just starting out in the world of programming, can understand that you can often improve functions and reduce code and therefore make development easier and faster. Optimizing code for reusability may seem like a waste of time at first. However, it will help us throughout development because when we need to improve something, we'll have fewer things to repeat in multiple places. Ask yourself this question when programming: Can I reduce the amount of code that is used to execute this task?

This is exactly what the function below does: it replaces the previous 4 functions, so that if we need to improve the system, we will only need to change one function, not four.

```
inline bool CMD_Array(char &cError, eWhatExec e1)
    {
        bool        bBarsPrev = false;
        string      szInfo;
        C_FileBars    *pFileBars;
        C_Array     *ptr = NULL;

        switch (e1)
        {
            case eTickReplay: ptr = m_GlPrivate.pTicksToReplay; break;
            case eTickToBar : ptr = m_GlPrivate.pTicksToBars;   break;
            case eBarToTick : ptr = m_GlPrivate.pBarsToTicks;   break;
            case eBarPrev   : ptr = m_GlPrivate.pBarsToPrev;    break;
        }
        if (ptr != NULL)
        {
            for (int c0 = 0; (c0 < INT_MAX) && (cError == 0); c0++)
            {
                if ((szInfo = ptr.At(c0, m_GlPrivate.Line)) == "") break;
                switch (e1)
                {
                    case eTickReplay:
                        if (LoadTicks(szInfo) == 0) cError = 4;
                        break;
                    case eTickToBar :
                        if ((m_dtPrevLoading = LoadTicks(szInfo, false)) == 0) cError = 5; else bBarsPrev = true;
                        break;
                    case eBarToTick :
                        if (!BarsToTicks(szInfo)) cError = 6;
                        break;
                    case eBarPrev   :
                        pFileBars = new C_FileBars(szInfo);
                        if ((m_dtPrevLoading = (*pFileBars).LoadPreView()) == 0) cError = 3; else bBarsPrev = true;
                        delete pFileBars;
                        break;
                }
            }
            delete ptr;
        }

        return bBarsPrev;
    }
```

It is very important to note that this function is actually divided into two segments, although it is within the same function. In the first segment, we initialize a pointer that will be used to access the data stored when reading the configuration file. This segment is quite simple and I don't think anyone will have any difficulty understanding it. After this, we move on to the second segment, where we will read the contents of user-specified files. Here, each of the pointers will read the contents stored in memory and do its job. At the end, the pointer is destroyed, freeing the used memory.

But before we get back to the setup function, I want to show you that there's even more you can do with this system. At the beginning of this topic, I said that you need to compile a program, thinking about how the reading will happen, but in the process of working on the article, I thought about it and decided that it was possible to expand the scope so that you do not need to constantly recompile the program.

The idea is to first load the simulated data and then load previous bars if needed. But we have a problem: both the ticks that will be used in the replay/simulation and the previous bars can come from files of type "Tick" or type "Bar", but we need to indicate this somehow. So I suggest using a variable that the user can define. Let's be very specific to keep things simple and make things viable in the long run. To do this, let's use the following table:

| Value | Reading mode |
| --- | --- |
| 1 | Tick - Tick mode |
| 2 | Tick - Bar mode |
| 3 | Bar - Tick mode |
| 4 | Bar - Bar mode |

Table 01 - Data for internal reading model

The table above shows how the reading will occur. We always start by reading the data that will be used in the replay or simulation, and then read the value that will be used as a previous data on the chart. Then, if the user reports a value of 1, the system will work as follows: File of real ticks - File of bars converted to ticks - File of ticks converted to previous data - File of previous bars. This would be option 1, but suppose that for some reason we want to modify the system so that it uses the same database but performs a different type of analysis. In this case, you can report, for example, the value 4, and the system will use the same database, but the result will be slightly different, since reading will be done in the following order: File of bars converted into ticks - File of real ticks - File of previous bars - File of ticks converted to previous data... If we try this, we will see that indicators, like average or otherwise, will generate small changes from one mode to another, even using the same database.

This implementation requires minimal programming effort, so I think it makes sense to offer such a resource.

Let's now see how to implement such a system. First, we add a global but private variable to the class so that we can continue to work.

```
class C_ConfigService : protected C_FileTicks
{

       protected:
//+------------------------------------------------------------------+

          datetime m_dtPrevLoading;

          int      m_ReplayCount,
               m_ModelLoading;
```

This variable will store the model. In case the user does not define it in the configuration file, we will set an initial value for it.

```
C_ConfigService()
   :m_szPath(NULL), m_ModelLoading(1)
   {
   }
```

In other words, we will always run and use Tick - Tick mode by default. After this, you need to give the user the opportunity to specify the value to use. This is also simple:

```
inline bool Configs(const string szInfo)
    {
        const string szList[] =
        {
            "PATH",
            "POINTSPERTICK",
            "VALUEPERPOINTS",
            "VOLUMEMINIMAL",
            "LOADMODEL"
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
                case 4:
                    m_ModelLoading = StringInit(szRet[1]);
                    m_ModelLoading = ((m_ModelLoading < 1) && (m_ModelLoading > 4) ? 1 : m_ModelLoading);
                    return true;
            }
            Print("Variable >>", szRet[0], "<< undefined.");
        }else
        Print("Set-up configuration >>", szInfo, "<< invalid.");

        return false;
    }
```

Here we specify the name that the user will use when working with the variable and also allow us to define the value to be used, as shown in Table 01. There is a small point here: we need to make sure that the value is within the limits expected by the system. If the user specifies a value other than what is expected, no error will occur, but the system will start using the default value.

With all this done, we can see what the final load and setup function looks like. It is shown below:

```
bool SetSymbolReplay(const string szFileConfig)
    {
#define macroFileName ((m_szPath != NULL ? m_szPath + "\\" : "") + szInfo)
        int     file;
        char    cError,
                cStage;
        string  szInfo;
        bool    bBarsPrev;

        if ((file = FileOpen("Market Replay\\" + szFileConfig, FILE_CSV | FILE_READ | FILE_ANSI)) == INVALID_HANDLE)
        {
            Print("Failed to open configuration file [", szFileConfig, "]. Closing the service...");
            return false;
        }
        Print("Loading ticks for replay. Please wait....");
        ArrayResize(m_Ticks.Rate, def_BarsDiary);
        m_Ticks.nRate = -1;
        m_Ticks.Rate[0].time = 0;
        cError = cStage = 0;
        bBarsPrev = false;
        m_GlPrivate.Line = 1;
        m_GlPrivate.pTicksToReplay = m_GlPrivate.pTicksToBars = m_GlPrivate.pBarsToTicks = m_GlPrivate.pBarsToPrev = NULL;
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
                            if (m_GlPrivate.pBarsToPrev == NULL) m_GlPrivate.pBarsToPrev = new C_Array();
                            (*m_GlPrivate.pBarsToPrev).Add(macroFileName, m_GlPrivate.Line);
                            break;
                        case 2:
                            if (m_GlPrivate.pTicksToReplay == NULL) m_GlPrivate.pTicksToReplay = new C_Array();
                            (*m_GlPrivate.pTicksToReplay).Add(macroFileName, m_GlPrivate.Line);
                            break;
                        case 3:
                            if (m_GlPrivate.pTicksToBars == NULL) m_GlPrivate.pTicksToBars = new C_Array();
                            (*m_GlPrivate.pTicksToBars).Add(macroFileName, m_GlPrivate.Line);
                            break;
                        case 4:
                            if (m_GlPrivate.pBarsToTicks == NULL) m_GlPrivate.pBarsToTicks = new C_Array();
                            (*m_GlPrivate.pBarsToTicks).Add(macroFileName, m_GlPrivate.Line);
                            break;
                        case 5:
                            if (!Configs(szInfo)) cError = 7;
                            break;
                    }
                break;
            };
            m_GlPrivate.Line += (cError > 0 ? 0 : 1);
        }
        FileClose(file);
        CMD_Array(cError, (m_ModelLoading <= 2 ? eTickReplay : eBarToTick));
        CMD_Array(cError, (m_ModelLoading <= 2 ? eBarToTick : eTickReplay));
        bBarsPrev = (CMD_Array(cError, ((m_ModelLoading & 1) == 1 ? eTickToBar : eBarPrev)) ? true : bBarsPrev);
        bBarsPrev = (CMD_Array(cError, ((m_ModelLoading & 1) == 1 ? eBarPrev : eTickToBar)) ? true : bBarsPrev);
        switch(cError)
        {
            case 0:
                if (m_Ticks.nTicks <= 0)
                {
                    Print("No ticks to use. Closing the service...");
                    cError = -1;
                }else if (!bBarsPrev) FirstBarNULL();
                break;
            case 1  : Print("Command in line ", m_GlPrivate.Line, " could not be recognized by the system..."); break;
            case 2  : Print("The contents of the line is unexpected for the system: ", m_GlPrivate.Line);              break;
            default : Print("Access error occurred in the files indicated in line: ", m_GlPrivate.Line);
        }

        return (cError == 0 ? !_StopFlag : false);
#undef macroFileName
    }
```

Now we have a small system that allows us to choose which model will be used for loading. However, remember that we always start by loading what will be used in replay or simulation, and then load the data that will be used as previous bars.

### Final considerations

This concludes this stage of work on the configuration file. Now (at least for now) the user will be able to specify everything needed at this early stage.

This article took quite a lot of time, but, in my opinion, it was worth it, since now we can simply not worry about problems with the configuration file for a while. It will always work as expected, no matter how it's built.

In the next article, we will continue to adapt the replay/simulation system to further cover forex and similar markets, since the stock market is already at a much more advanced stage. I will focus on the forex market so that the system can work there in the same way as the stock market.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11153](https://www.mql5.com/pt/articles/11153)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11153.zip "Download all attachments in the single ZIP archive")

[Market\_Replay\_ev221.zip](https://www.mql5.com/en/articles/download/11153/market_replay_ev221.zip "Download Market_Replay_ev221.zip")(14387.15 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/462424)**

![Population optimization algorithms: Stochastic Diffusion Search (SDS)](https://c.mql5.com/2/59/SDS_avatar.png)[Population optimization algorithms: Stochastic Diffusion Search (SDS)](https://www.mql5.com/en/articles/13540)

The article discusses Stochastic Diffusion Search (SDS), which is a very powerful and efficient optimization algorithm based on the principles of random walk. The algorithm allows finding optimal solutions in complex multidimensional spaces, while featuring a high speed of convergence and the ability to avoid local extrema.

![Neural networks made easy (Part 58): Decision Transformer (DT)](https://c.mql5.com/2/58/decision-transformer-avatar.png)[Neural networks made easy (Part 58): Decision Transformer (DT)](https://www.mql5.com/en/articles/13347)

We continue to explore reinforcement learning methods. In this article, I will focus on a slightly different algorithm that considers the Agent’s policy in the paradigm of constructing a sequence of actions.

![Introduction to MQL5 (Part 4): Mastering Structures, Classes, and Time Functions](https://c.mql5.com/2/70/Introduction_to_MQL5_xPart_44_Mastering_Structureso_Classesi_and_Time_Functions____LOGO.png)[Introduction to MQL5 (Part 4): Mastering Structures, Classes, and Time Functions](https://www.mql5.com/en/articles/14232)

Unlock the secrets of MQL5 programming in our latest article! Delve into the essentials of structures, classes, and time functions, empowering your coding journey. Whether you're a beginner or an experienced developer, our guide simplifies complex concepts, providing valuable insights for mastering MQL5. Elevate your programming skills and stay ahead in the world of algorithmic trading!

![Building and testing Keltner Channel trading systems](https://c.mql5.com/2/69/Building_and_testing_Keltner_Channel_trading_systems____LOGO__1.png)[Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)

In this article, we will try to provide trading systems using a very important concept in the financial market which is volatility. We will provide a trading system based on the Keltner Channel indicator after understanding it and how we can code it and how we can create a trading system based on a simple trading strategy and then test it on different assets.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=auzrlfsydxpulwdmfsmfrtdtmaffkrjy&ssn=1769180599542057094&ssn_dr=0&ssn_sr=0&fv_date=1769180599&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11153&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20%E2%80%94%20Market%20simulation%20(Part%2021)%3A%20FOREX%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918059915874792&fz_uniq=5068991961538428854&sv=2552)

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