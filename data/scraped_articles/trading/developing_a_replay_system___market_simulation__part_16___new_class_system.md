---
title: Developing a Replay System — Market simulation (Part 16): New class system
url: https://www.mql5.com/en/articles/11095
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:04:41.974908
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/11095&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069033180339568646)

MetaTrader 5 / Tester


### Introduction

In the previous article " [Developing a Replay System — Market simulation (Part 15): Birth of the SIMULATOR (V) - RANDOM WALK](https://www.mql5.com/en/articles/11071)", we have developed a method to obtain randomized data. This was done to ensure completely appropriate results. But even though the system is indeed capable of plausibly simulating orders, it still lacks certain information, not only from the simulator, but also from the replay system itself. The truth is that some of the things we need to implement are quite complex, especially from a system-building perspective. So we need to make some changes to at least get a better model and not get lost in the next steps.

Trust me, if you found RANDOM WALK challenging, it's only because you haven't seen what we need to developed to make the simulation/replay experience the best it can be. One of the most challenging aspects is that we must know at least the basics about the asset which will be used for simulation or replay. At this stage of development, I'm not worried about automatically solving some problems, since there are more practical ways to do it.

If you carefully study the previous article, you will notice that programs such as EXCEL can be used to create possible market scenarios. But we can also use information from other assets and combine them to create more complex simulations. This makes it difficult to create a fully automated system that could look for missing information about the asset itself. We have one more small problem. Therefore, here we will cut and distribute the contents of the C\_Replay.mqh header file to facilitate maintenance and improvements. I don't like working with a very large class as this seems inconvenient to me. The reason is that in many cases the code may not be fully optimized. So to make things easier, we will make both changes, i.e., we will distribute the contents of the C\_Replay class across several classes and at the same time implement the most relevant parts.

### Implementing the service with a new class system

Although the code discussed in previous articles has generally undergone very few changes, this new implementation contains a new model. This model is easier to extend and also easier to understand for those without extensive programming knowledge because the elements are connected in a simpler way. We will not have long and tiring functions to read. There are some details that may seem a little strange, but, oddly enough, they make the code more secure, stable, and structurally sound. One of these details is the use of pointers in a same way different from that used in C++ or C legacy languages. However, the behavior will very similar to those languages.

The use of pointers allows us to do things that would otherwise be impossible. Although many of the features introduced in C++ are not available in MQL5, the simple fact of using pointers in their simplest form allows us to implement modeling in a more flexible and enjoyable way, at least when it comes to programming and syntax.

Therefore, the new service file in the current version now looks like this:

```
#property service
#property icon "\\Images\\Market Replay\\Icon.ico"
#property copyright "Daniel Jose"
#property version   "1.16"
#property description "Replay-simulation system for MT5."
#property description "It is independent from the Market Replay."
#property description "For details see the article:"
#property link "https://www.mql5.com/ru/articles/11095"
//+------------------------------------------------------------------+
#define def_Dependence  "\\Indicators\\Market Replay.ex5"
#resource def_Dependence
//+------------------------------------------------------------------+
#include <Market Replay\C_Replay.mqh>
//+------------------------------------------------------------------+
input string            user00 = "Config.txt";  //"Replay" config file.
input ENUM_TIMEFRAMES   user01 = PERIOD_M1;     //Initial timeframe for the chart.
input bool              user02 = true;          //Visual bar construction.
input bool              user03 = true;          //Visualize creation metrics.
//+------------------------------------------------------------------+
void OnStart()
{
        C_Replay        *pReplay;

        pReplay = new C_Replay(user00);
        if (pReplay.ViewReplay(user01))
        {
                Print("Permission received. The replay service can now be used...");
                while (pReplay.LoopEventOnTime(user02, user03));
        }
        delete pReplay;
}
//+------------------------------------------------------------------+
```

It may seem that it has not undergone any major changes, but in reality everything is completely different. For the end user it will remain the same and have exactly the same behavior as previous versions. But for the platform and especially for the OS, the generated executable will look different and have different behavior. The reason is that we are using the class not as a variable, but as a pointer to memory. And that's why the whole system behaves completely differently. If you program carefully, classes will be even more secure and stable than classes that are used only as variables. Although they show the same behavior, they will differ in terms of memory usage.

Now, since classes are used as pointers, we will exclude some things and start using others. The first thing is that the class will always be launched and closed explicitly. This is done using the **_new_** and **_delete_** operators. When we use the " **new**" operator to create a class, we must always call the class constructor. They never return a value, so we cannot check the return value directly. We'll have to do that at another time. The same thing happens when using the " **delete**" operator, and the class destructor will be called. Like the class constructor, the destructor never returns a value. But unlike the constructor, the destructor **does not receive** any arguments.

We will always have to do it this way: we create a class using the "new" operator and destroy the class using the "delete" operator. This will be the only work we actually have to do. The rest will be done by the OS: it allocates enough memory for the program to run in a specific memory area, making it as secure as possible throughout its existence. But here lies the danger for those who are used to using pointers in C++/C: we are talking about notation. In C++ and C, whenever we refer to pointers, we use a very specific notation. Typically we use an arrow ( **->**). For a C++/C programmer, this means that a pointer is used. But we can also use a different notation, which can be seen in my codes when accessing a pointer.

In addition, of course, the variable name is used, which usually begins with the letter " **p**" or combinations like " **ptr**" (although this is not a strict rule, so don't stick to it). Although MQL5 accepts both the notation shown in the code above and the notation shown below, I personally find code that uses pointers to be easier to read when it actually uses the correct declaration. Therefore, in our codes the notation will be as shown below, thanks to my knowledge of the C++/C language:

```
void OnStart()
{
        C_Replay        *pReplay;

        pReplay = new C_Replay(user00);
        if ((*pReplay).ViewReplay(user01))
        {
                Print("Permission received. The replay service can now be used...");
                while ((*pReplay).LoopEventOnTime(user02, user03));
        }
        delete pReplay;
}
```

There is actually extra work related to writing the code. But as a C++/C programmer with years of experience, it's easier to understand that I am referring to a pointer when I look at code like the one shown above. And since MQL5 understands this in the same way as C++/C, I see no problems in using this notation. Whenever we see code with a notation like the one shown above, you shouldn't worry because it's just a pointer.

We can continue exploring the new class system. If you think that only these changes have happened, you are being quite optimistic. The very fact of making these changes, where we explicitly guarantee that the class will be created and destroyed at a certain time, requires making several more changes to the code. The constructor and destructor do not return values. But we must somehow know whether the class is created correctly or not.

To understand how to do this, you need to look inside the black box of the C\_Replay class. It is located in the header file C\_Replay.mqh. Its internal structure is shown in the image below:

![Figure 01 – C_Replay.mqh](https://c.mql5.com/2/47/001__4.png)

Figure 01 - Replay class connection system

Figure 01 shows how the classes are connected to each other to perform the desired work by the replay/simulation service. The green arrow indicates that the class is imported, so some internal members of the class will be public. The red arrow indicates that the data will be imported but they will no longer be visible in the senior classes. The C\_FilesBars class is a loose class. It will not be actually inherited by any other class, but its methods will be used by other classes.

To truly understand how these connections are created, you need to see what's going inside. To do this, we will need to look at how each of the classes was created and how they are located inside the corresponding files. These files always have the same name as the class. This is not necessary, but is still a good practice since the code has undergone some changes. We won't go into detail about this, but in a few minutes we'll look at what changes have occurred and why. This can be very useful for those who are starting their journey in programming, since it is much easier to learn when there is functional code and we modify it to generate something different (and at the same time what we want), maintaining at the same time the operability of the code.

Let's look at how the structure was implemented.

### Loose class: C\_FileBars

This class is quite simple and contains everything needed to read the bars present in the file. It's not a very big class. Its entire code is shown below:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "Interprocess.mqh"
//+------------------------------------------------------------------+
#define def_BarsDiary   1440
//+------------------------------------------------------------------+
class C_FileBars
{
        private :
                int     m_file;
                string  m_szFileName;
//+------------------------------------------------------------------+
inline void CheckFileIsBar(void)
                        {
                                string  szInfo = "";

                                for (int c0 = 0; (c0 < 9) && (!FileIsEnding(m_file)); c0++) szInfo += FileReadString(m_file);
                                if (szInfo != "<DATE><TIME><OPEN><HIGH><LOW><CLOSE><TICKVOL><VOL><SPREAD>")
                                {
                                        Print("Файл ", m_szFileName, ".csv не является файлом баров.");
                                        FileClose(m_file);
                                        m_file = INVALID_HANDLE;
                                }
                        }
//+------------------------------------------------------------------+
        public  :
//+------------------------------------------------------------------+
                C_FileBars(const string szFileNameCSV)
                        :m_szFileName(szFileNameCSV)
                        {
                                if ((m_file = FileOpen("Market Replay\\Bars\\" + m_szFileName + ".csv", FILE_CSV | FILE_READ | FILE_ANSI)) == INVALID_HANDLE)
                                        Print("Could not access ", m_szFileName, ".csv with bars.");
                                else
                                        CheckFileIsBar();
                        }
//+------------------------------------------------------------------+
                ~C_FileBars()
                        {
                                if (m_file != INVALID_HANDLE) FileClose(m_file);
                        }
//+------------------------------------------------------------------+
                bool ReadBar(MqlRates &rate[])
                        {
                                if (m_file == INVALID_HANDLE) return false;
                                if (FileIsEnding(m_file)) return false;
                                rate[0].time = StringToTime(FileReadString(m_file) + " " + FileReadString(m_file));
                                rate[0].open = StringToDouble(FileReadString(m_file));
                                rate[0].high = StringToDouble(FileReadString(m_file));
                                rate[0].low = StringToDouble(FileReadString(m_file));
                                rate[0].close = StringToDouble(FileReadString(m_file));
                                rate[0].tick_volume = StringToInteger(FileReadString(m_file));
                                rate[0].real_volume = StringToInteger(FileReadString(m_file));
                                rate[0].spread = (int) StringToInteger(FileReadString(m_file));

                                return true;
                        }
//+------------------------------------------------------------------+
                datetime LoadPreView(const string szFileNameCSV)
                        {
                                int      iAdjust = 0;
                                datetime dt = 0;
                                MqlRates Rate[1];

                                Print("Loading bars for Replay. Please wait....");
                                while (ReadBar(Rate) && (!_StopFlag))
                                {
                                        iAdjust = ((dt != 0) && (iAdjust == 0) ? (int)(Rate[0].time - dt) : iAdjust);
                                        dt = (dt == 0 ? Rate[0].time : dt);
                                        CustomRatesUpdate(def_SymbolReplay, Rate, 1);
                                }
                                return ((_StopFlag) || (m_file == INVALID_HANDLE) ? 0 : Rate[0].time + iAdjust);
                        }
//+------------------------------------------------------------------+
};
```

You might think this class doesn't make much sense, but it contains everything needed to open, read, build a custom resource, and close a panel file. All of this is implemented in very specific steps, so the changes made here do not matter. If the reader wants to read the bar file, this can be done here.

This class is designed for continuous use by operators **NEW** and **DELETE**. Thus, it will remain in memory only long enough to do its job. As much as possible, we should avoid using this class without using the operators mentioned above. Otherwise we may have stability problems. Not that this is really going to happen. But the fact that it was designed to use the operators does not make it suitable for being used by other means.

In this class we have 2 global and private variables. They are initialized precisely in the class constructor, where we will try to open and check whether the specified file is a bar file or not. However, please do not forger that the constructor does not return a value of any type and this we have no way to return anything at all here, but we can indicate that a file does not meet expectations by marking it as an invalid file. It can become clear after it is closed. Any attempt to read will result in an error, which will be reported accordingly. The return can then be processed by the caller. As a consequence, the class will act as if it were a file that already contained bars. Or a large object that already contains the contents of the file. The callers here will read the contents of this large object. But once the destructor is called, the file will be closed and the class will be destroyed by the calling program.

This type of simulation may not seem as secure and stable, but trust me, it is much more secure and stable than it seems. If it were possible to access some of the other operators present in C++, things would get even more interesting. Of course, for this you need to write the code correctly, otherwise everything would be a complete disaster. But since MQL5 is not C++, let's study and make the most of the capabilities of this language. This way we will have a system that will use something very close to the limits that language allows us to achieve.

### Deep class: C\_FileTicks

The next class we are going to look at is the C\_FileTicks class. It's much more complex than the C\_FileBars class because we have public elements, private elements, and elements that fall somewhere in between. They have a special name: **_PROTECTED_**. The term "protected" has a special level when it comes to inheritance between classes. In the case of C++, everything is quite complicated, at least at the beginning of learning. This is due to some operators present in C++. Fortunately, MQL5 solves the problem in a much simpler way. So, it will be much easier to understand how elements declared as protected are inherited, and whether they can be accessed or not, depending of course on how the inheritance occurs. See the table below:

| Definition in the base class | Base class inheritance type | Access within a derived class | Access by calling a derived class |
| --- | --- | --- | --- |
| private | public | Access is denied | Unable to access base class data or procedures |
| public | public | Access is allowed | Able to access base class data or procedures |
| protected | public | Access is allowed | Unable to access base class data or procedures |
| private | private | Access is denied | Unable to access base class data or procedures |
| public | private | Access is allowed | Unable to access base class data or procedures |
| protected | private | Access is allowed | Unable to access base class data or procedures |
| private | protected | Access is denied | Unable to access base class data or procedures |
| public | protected | Access is allowed | Unable to access base class data or procedures |
| protected | protected | Access is allowed | Unable to access base class data or procedures |

Table of access levels to class elements and functions

Only in one case can we access data or procedures within a class, using an inheritance or access definition system. And that’s when everything is declared public. In all other cases, this is more or less possible at the class inheritance level, but it is impossible to access any procedure or data located inside the base class. This does not depend on the access clause.

**Important**: If you declare something as "protected" and try to directly access that data or procedures without using class inheritance, then you will not be able to access such data or procedures. This is because without using inheritance, the data or procedures declared as protected are considered private and therefore will not be accessible.

This seems quite complicated, doesn't it? However, in practice it is much easier. However, we will have to experiment with this mechanism in action several times to truly understand how it functions. But believe me, this is much easier to do in MQL5 than in C++; things are much more complicated there. The reason is that we have ways to change the level of access to data or procedures declared as protected, in some cases even private, in the process of class inheritance. That's really crazy. However, in MQL5 everything works smoothly.

Based on this, we can finally see the C\_FileTicks class, which, despite being theoretically more complex, has relatively simple code. Let's start looking at the first elements inside the class, and start with its declaration:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "C_FileBars.mqh"
//+------------------------------------------------------------------+
#define def_MaxSizeArray        16777216 // 16 Mbytes of positions
//+------------------------------------------------------------------+
#define macroRemoveSec(A) (A - (A % 60))
//+------------------------------------------------------------------+
class C_FileTicks
{
        protected:
                struct st00
                {
                        MqlTick  Info[];
                        MqlRates Rate[];
                        int      nTicks,
                                 nRate;
                }m_Ticks;
                double          m_PointsPerTick;
```

Look, it's quite simple, but be careful because this structure and this variable can be accessed by following the access level table. Once this is done, we will have a number of private procedures for the class. They cannot be accessed outside the class and serve to support public procedures. There are two of them, as can be seen below:

```
        public  :
//+------------------------------------------------------------------+
                bool BarsToTicks(const string szFileNameCSV)
                        {
                                C_FileBars      *pFileBars;
                                int             iMem = m_Ticks.nTicks;
                                MqlRates        rate[1];
                                MqlTick         local[];

                                pFileBars = new C_FileBars(szFileNameCSV);
                                ArrayResize(local, def_MaxSizeArray);
                                Print("Converting bars to ticks. Please wait...");
                                while ((*pFileBars).ReadBar(rate) && (!_StopFlag)) Simulation(rate[0], local);
                                ArrayFree(local);
                                delete pFileBars;

                                return ((!_StopFlag) && (iMem != m_Ticks.nTicks));
                        }
//+------------------------------------------------------------------+
                datetime LoadTicks(const string szFileNameCSV, const bool ToReplay = true)
                        {
                                int             MemNRates,
                                                MemNTicks;
                                datetime dtRet = TimeCurrent();
                                MqlRates RatesLocal[];

                                MemNRates = (m_Ticks.nRate < 0 ? 0 : m_Ticks.nRate);
                                MemNTicks = m_Ticks.nTicks;
                                if (!Open(szFileNameCSV)) return 0;
                                if (!ReadAllsTicks()) return 0;
                                if (!ToReplay)
                                {
                                        ArrayResize(RatesLocal, (m_Ticks.nRate - MemNRates));
                                        ArrayCopy(RatesLocal, m_Ticks.Rate, 0, 0);
                                        CustomRatesUpdate(def_SymbolReplay, RatesLocal, (m_Ticks.nRate - MemNRates));
                                        dtRet = m_Ticks.Rate[m_Ticks.nRate].time;
                                        m_Ticks.nRate = (MemNRates == 0 ? -1 : MemNRates);
                                        m_Ticks.nTicks = MemNTicks;
                                        ArrayFree(RatesLocal);
                                }
                                return dtRet;
                        };
//+------------------------------------------------------------------+
```

All private procedures are procedures that have already been explained and analyzed in previous articles in this series. We won't go into detail here as the procedures have remained virtually unchanged from what has been explained. However, there is something worthy of our attention regarding these two public procedures. This something is in **BarsToTicks**. In the previous topic, I mentioned that the C\_FileBars class is a loose class that is not actually inherited. This is still true. But I also mentioned that it would be used at certain points and should be accessed in a rather specific way.

And here is one of those moments. First we declare the class in a very specific way. Now we call the class constructor with the name of the file from which we want to get the bar values. Remember that this call will not return any value. We use the **_NEW_** operator so that the class has memory space reserved just for it. This space will contain the class, since MetaTrader 5 does not actually control where this class can be located. Only the operating system has this information.

But more conveniently, we get a value from the NEW operator, and this value is a "pointer" that we can use to directly reference our class (NOTE: The word "pointer" is in quotes because it is not actually a pointer, it is simply a variable that can refer to a class as if that class were another variable or constant. In the future, I'll show how to use this to create a constant class in which we can only access data, but not perform calculations. Since it has a very specific use, we will leave it for another time.). Once we have this "pointer", we can do work in our class, but again, we are still not sure that the file is open and can be read. Therefore, we will need to perform some kind of validation before attempting to read and use any data. Fortunately, this is not necessary because during the read attempt we can check whether the call completed correctly or not. That is, we can check whether the data has been read or not here.

If for some reason the reading fails, the **WHILE** loop will just close. Therefore, no further checks are necessary. The very attempt to read the data will serve to check whether the read was successful or not. This way we can manipulate things outside of the C\_FileBars class, but we must terminate the class explicitly so that whatever memory it was in is returned to the OS. This is done by calling the destructor via the **_DELETE_** operator. This ensures that the class has been properly deleted and is no longer referenced.

Failure to do so may result in inconsistent data and even junk in our programs. But by following the above procedure, we will know exactly when, where and how the class is used. This can help us in several scenarios where the modeling can be quite complex.

### One class, multiple functions: C\_ConfigService

This class is very interesting. Even though this is a class that acts as a bridge between the C\_FileTicks class and the C\_Replay class, it somehow ensures that everything remains as we expect and that improvements or changes in the configuration system are reflected only in places where they really should be visible. This is not a very extensive or complex class, but just an intermediate class with fairly simple code. The idea is to put everything related to setting up the replay/simulation service in this class. Essentially, its purpose is to read the configuration file and apply its contents to the service so that it works as configured by the user.

The class should do the following: read and create ticks for modeling, apply bars to the replay asset, and, in some cases, adjust the variables of the replay asset. Thus, the asset will behave very closely to the real asset. Below is the full code of the class at the current stage of development:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "C_FileBars.mqh"
#include "C_FileTicks.mqh"
//+------------------------------------------------------------------+
class C_ConfigService : protected C_FileTicks
{
        protected:
//+------------------------------------------------------------------+
                datetime m_dtPrevLoading;
//+------------------------------------------------------------------+
        private :
//+------------------------------------------------------------------+
                enum eTranscriptionDefine {Transcription_INFO, Transcription_DEFINE};
//+------------------------------------------------------------------+
inline eTranscriptionDefine GetDefinition(const string &In, string &Out)
                        {
                                string szInfo;

                                szInfo = In;
                                Out = "";
                                StringToUpper(szInfo);
                                StringTrimLeft(szInfo);
                                StringTrimRight(szInfo);
                                if (StringSubstr(szInfo, 0, 1) == "#") return Transcription_INFO;
                                if (StringSubstr(szInfo, 0, 1) != "[")\
                                {\
                                        Out = szInfo;\
                                        return Transcription_INFO;\
                                }\
                                for (int c0 = 0; c0 < StringLen(szInfo); c0++)\
                                        if (StringGetCharacter(szInfo, c0) > ' ')\
                                                StringAdd(Out, StringSubstr(szInfo, c0, 1));\
\
                                return Transcription_DEFINE;\
                        }\
//+------------------------------------------------------------------+\
inline bool Configs(const string szInfo)\
                        {\
                                const string szList[] = {\
\
						"POINTSPERTICK"\
                                                        };\
                                string  szRet[];\
                                char    cWho;\
\
                                if (StringSplit(szInfo, '=', szRet) == 2)\
                                {\
                                        StringTrimRight(szRet[0]);\
                                        StringTrimLeft(szRet[1]);\
                                        for (cWho = 0; cWho < ArraySize(szList); cWho++) if (szList[cWho] == szRet[0]) break;\
                                        switch (cWho)\
                                        {\
                                                case 0:\
                                                        m_PointsPerTick = StringToDouble(szRet[1]);\
                                                        return true;\
                                        }\
                                        Print("Variable >>", szRet[0], "<< undefined.");\
                                }else\
                                        Print("Configuration >>", szInfo, "<< invalid.");\
\
                                return false;\
                        }\
//+------------------------------------------------------------------+\
inline void FirstBarNULL(void)\
                        {\
                                MqlRates rate[1];\
\
                                rate[0].close = rate[0].open =  rate[0].high = rate[0].low = m_Ticks.Info[0].last;\
                                rate[0].tick_volume = 0;\
                                rate[0].real_volume = 0;\
                                rate[0].time = m_Ticks.Info[0].time - 60;\
                                CustomRatesUpdate(def_SymbolReplay, rate, 1);\
                        }\
//+------------------------------------------------------------------+\
inline bool WhatDefine(const string szArg, char &cStage)\
                        {\
                                const string szList[] = {\
                                        "[BARS]",\
                                        "[TICKS]",\
                                        "[TICKS->BARS]",\
                                        "[BARS->TICKS]",\
                                        "[CONFIG]"\
                                                        };\
\
                                cStage = 1;\
                                for (char c0 = 0; c0 < ArraySize(szList); c0++, cStage++)\
                                        if (szList[c0] == szArg) return true;\
\
                                return false;\
                        }\
//+------------------------------------------------------------------+\
        public  :\
//+------------------------------------------------------------------+\
                bool SetSymbolReplay(const string szFileConfig)\
                        {\
                                int             file,\
                                                iLine;\
                                char            cError,\
                                                cStage;\
                                string          szInfo;\
                                bool            bBarPrev;\
                                C_FileBars      *pFileBars;\
\
                                if ((file = FileOpen("Market Replay\\" + szFileConfig, FILE_CSV | FILE_READ | FILE_ANSI)) == INVALID_HANDLE)\
                                {\
                                        MessageBox("Failed to open the\nconfiguration file.", "Market Replay", MB_OK);\
                                        return false;\
                                }\
                                Print("Loading data for replay. Please wait....");\
                                ArrayResize(m_Ticks.Rate, def_BarsDiary);\
                                m_Ticks.nRate = -1;\
                                m_Ticks.Rate[0].time = 0;\
                                bBarPrev = false;\
\
                                iLine = 1;\
                                cError = cStage = 0;\
                                while ((!FileIsEnding(file)) && (!_StopFlag) && (cError == 0))\
                                {\
                                        switch (GetDefinition(FileReadString(file), szInfo))\
                                        {\
                                                case Transcription_DEFINE:\
                                                        cError = (WhatDefine(szInfo, cStage) ? 0 : 1);\
                                                        break;\
                                                case Transcription_INFO:\
                                                        if (szInfo != "") switch (cStage)\
                                                        {\
                                                                case 0:\
                                                                        cError = 2;\
                                                                        break;\
                                                                case 1:\
                                                                        pFileBars = new C_FileBars(szInfo);\
                                                                        if ((m_dtPrevLoading = (*pFileBars).LoadPreView(szInfo)) == 0) cError = 3; else bBarPrev = true;\
                                                                        delete pFileBars;\
                                                                        break;\
                                                                case 2:\
                                                                        if (LoadTicks(szInfo) == 0) cError = 4;\
                                                                        break;\
                                                                case 3:\
                                                                        if ((m_dtPrevLoading = LoadTicks(szInfo, false)) == 0) cError = 5; else bBarPrev = true;\
                                                                        break;\
                                                                case 4:\
                                                                        if (!BarsToTicks(szInfo)) cError = 6;\
                                                                        break;\
                                                                case 5:\
                                                                        if (!Configs(szInfo)) cError = 7;\
                                                                        break;\
                                                        }\
                                                        break;\
                                        };\
                                        iLine += (cError > 0 ? 0 : 1);\
                                }\
                                FileClose(file);\
                                switch(cError)\
                                {\
                                        case 0:\
                                                if (m_Ticks.nTicks <= 0)\
                                                {\
                                                        Print("No ticks to use. Closing the service...");\
                                                        cError = -1;\
                                                }else   if (!bBarPrev) FirstBarNULL();\
                                                break;\
                                        case 1  : Print("Command in line ", iLine, " cannot be recognized by the system...");    break;\
                                        case 2  : Print("The system did not expect the content of the line ", iLine);                  break;\
                                        default : Print("Error in line ", iLine);\
                                }\
\
                                return (cError == 0 ? !_StopFlag : false);\
                        }\
//+------------------------------------------------------------------+\
};\
```\
\
Here we start using this inheritance table. All these items are inherited from the C\_FileTicks class. Thus, we are actually extending the functionality of the C\_ConfigService class itself. But this is not the only thing, because if you look closely, there is a very specific situation when we need to load data from bars. To do this we need to use the C\_FileBars class. So, we used the same method as in the C\_FileTicks class, where we needed to load data from the bars file to convert it into ticks. The explanation we gave there applies here too.\
\
In a sense, this class will be responsible for translating the data contained in the configuration file. Now, all we have to do is to define things at the right points so that they point to, or rather cause, the right state. We will do this to ensure that the values are filled in correctly or the data is loaded correctly. This is done in two places.\
\
In the first place we indicate in which state or, more precisely, which key we are capturing or adjusting. While it's not very complicated, what we have here is this: a list of things that will serve as a key to what we'll be working on in the next few lines of the configuration file. Here you just need to pay attention to the fact that this list must follow a certain logical order. Otherwise, we will have problems translating the values. To find out what position the data element should be in, simply look at the SetSymbolReplay function and see what exactly each value does here.\
\
The second place is responsible for decoding the values contained in the replay/simulation configuration file into constants that will be used within the service. Here we will do almost the same thing as before, but this time each of the values contained in the array will indicate the name of a variable within the class. So, all you need to do is add the name that the variable will have to the list of the configuration file. Then we add its position in the list to the call. This way we change the value of the desired variable. If you don't understand what I an saying, don't worry. Soon I'll show a real example of how to add new variables. Before that, we will need to define some additional things here at this specific point.\
\
Although everything looks very beautiful, we have yet to see the last class.\
\
### Class C\_Replay - I don’t understand anything... Where are the things?!\
\
This is the only class that the replay/simulation service will actually deal with. We need to think of this class as a library, but a library whose only function is to promote behavior that is similar to what would happen if, instead of replay or simulation, we interacted with a physical market or demo account. That is, all we need to do is to implement things exclusively within this class so that the MetaTrader 5 platform can perform all the replay simulation as if it were coming from a real server.\
\
However, if you look closely at the class code and start looking for something, you may wonder: where are the variables, structures and functions to be called? I can't find them anywhere! This kind of thinking can occur at the initial stage. It just means that you are not yet very familiar with inheritance between classes. Don't worry, study the code calmly and soon you will begin to understand how this inheritance works. It is better to start now, because soon I'll show you something even more complicated and it might confuse you a lot. One of these things is **POLYMORPHISM**. This is something very useful, but it also creates a lot of confusion for those who do not understand the problems associated with how inheritance works. So I recommend you to study this code properly.\
\
For now, let's leave the topic of polymorphism for the future. See the code below:\
\
```\
#property copyright "Daniel Jose"\
//+------------------------------------------------------------------+\
#include "C_ConfigService.mqh"\
//+------------------------------------------------------------------+\
class C_Replay : private C_ConfigService\
{\
        private :\
                int             m_ReplayCount;\
                long            m_IdReplay;\
                struct st01\
                {\
                        MqlRates Rate[1];\
                        bool     bNew;\
                        datetime memDT;\
                        int      delay;\
                }m_MountBar;\
//+------------------------------------------------------------------+\
                void AdjustPositionToReplay(const bool bViewBuider)\
                        {\
                                u_Interprocess  Info;\
                                MqlRates        Rate[def_BarsDiary];\
                                int             iPos, nCount;\
\
                                Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);\
                                if (Info.s_Infos.iPosShift == (int)((m_ReplayCount * def_MaxPosSlider * 1.0) / m_Ticks.nTicks)) return;\
                                iPos = (int)(m_Ticks.nTicks * ((Info.s_Infos.iPosShift * 1.0) / (def_MaxPosSlider + 1)));\
                                Rate[0].time = macroRemoveSec(m_Ticks.Info[iPos].time);\
                                if (iPos < m_ReplayCount)\
                                {\
                                        CustomRatesDelete(def_SymbolReplay, Rate[0].time, LONG_MAX);\
                                        if ((m_dtPrevLoading == 0) && (iPos == 0))\
                                        {\
                                                m_ReplayCount = 0;\
                                                Rate[m_ReplayCount].close = Rate[m_ReplayCount].open = Rate[m_ReplayCount].high = Rate[m_ReplayCount].low = m_Ticks.Info[iPos].last;\
                                                Rate[m_ReplayCount].tick_volume = Rate[m_ReplayCount].real_volume = 0;\
                                                CustomRatesUpdate(def_SymbolReplay, Rate, 1);\
                                        }else\
                                        {\
                                                for(Rate[0].time -= 60; (m_ReplayCount > 0) && (Rate[0].time <= macroRemoveSec(m_Ticks.Info[m_ReplayCount].time)); m_ReplayCount--);\
                                                m_ReplayCount++;\
                                        }\
                                }else if (iPos > m_ReplayCount)\
                                {\
                                        if (bViewBuider)\
                                        {\
                                                Info.s_Infos.isWait = true;\
                                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);\
                                        }else\
                                        {\
                                                for(; Rate[0].time > m_Ticks.Info[m_ReplayCount].time; m_ReplayCount++);\
                                                for (nCount = 0; m_Ticks.Rate[nCount].time < macroRemoveSec(m_Ticks.Info[iPos].time); nCount++);\
                                                CustomRatesUpdate(def_SymbolReplay, m_Ticks.Rate, nCount);\
                                        }\
                                }\
                                for (iPos = (iPos > 0 ? iPos - 1 : 0); (m_ReplayCount < iPos) && (!_StopFlag);) CreateBarInReplay();\
                                Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);\
                                Info.s_Infos.isWait = false;\
                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);\
                        }\
//+------------------------------------------------------------------+\
inline void CreateBarInReplay(const bool bViewMetrics = false)\
                        {\
#define def_Rate m_MountBar.Rate[0]\
\
                                static ulong _mdt = 0;\
                                int i;\
\
                                if (m_MountBar.bNew = (m_MountBar.memDT != macroRemoveSec(m_Ticks.Info[m_ReplayCount].time)))\
                                {\
                                        if (bViewMetrics)\
                                        {\
                                                _mdt = (_mdt > 0 ? GetTickCount64() - _mdt : _mdt);\
                                                i = (int) (_mdt / 1000);\
                                                Print(TimeToString(m_Ticks.Info[m_ReplayCount].time, TIME_SECONDS), " - Metrica: ", i / 60, ":", i % 60, ".", (_mdt % 1000));\
                                                _mdt = GetTickCount64();\
                                        }\
                                        m_MountBar.memDT = macroRemoveSec(m_Ticks.Info[m_ReplayCount].time);\
                                        def_Rate.real_volume = 0;\
                                        def_Rate.tick_volume = 0;\
                                }\
                                def_Rate.close = m_Ticks.Info[m_ReplayCount].last;\
                                def_Rate.open = (m_MountBar.bNew ? def_Rate.close : def_Rate.open);\
                                def_Rate.high = (m_MountBar.bNew || (def_Rate.close > def_Rate.high) ? def_Rate.close : def_Rate.high);\
                                def_Rate.low = (m_MountBar.bNew || (def_Rate.close < def_Rate.low) ? def_Rate.close : def_Rate.low);\
                                def_Rate.real_volume += (long) m_Ticks.Info[m_ReplayCount].volume_real;\
                                def_Rate.tick_volume += (m_Ticks.Info[m_ReplayCount].volume_real > 0 ? 1 : 0);\
                                def_Rate.time = m_MountBar.memDT;\
                                m_MountBar.bNew = false;\
                                CustomRatesUpdate(def_SymbolReplay, m_MountBar.Rate, 1);\
                                m_ReplayCount++;\
\
#undef def_Rate\
                        }\
//+------------------------------------------------------------------+\
        public  :\
//+------------------------------------------------------------------+\
                C_Replay(const string szFileConfig)\
                        {\
                                m_ReplayCount = 0;\
                                m_dtPrevLoading = 0;\
                                m_Ticks.nTicks = 0;\
                                m_PointsPerTick = 0;\
                                Print("************** Market Replay Service **************");\
                                srand(GetTickCount());\
                                GlobalVariableDel(def_GlobalVariableReplay);\
                                SymbolSelect(def_SymbolReplay, false);\
                                CustomSymbolDelete(def_SymbolReplay);\
                                CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay), _Symbol);\
                                CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);\
                                CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);\
                                SymbolSelect(def_SymbolReplay, true);\
                                m_IdReplay = (SetSymbolReplay(szFileConfig) ? 0 : -1);\
                        }\
//+------------------------------------------------------------------+\
                ~C_Replay()\
                        {\
                                ArrayFree(m_Ticks.Info);\
                                ArrayFree(m_Ticks.Rate);\
                                m_IdReplay = ChartFirst();\
                                do\
                                {\
                                        if (ChartSymbol(m_IdReplay) == def_SymbolReplay)\
                                                ChartClose(m_IdReplay);\
                                }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);\
                                for (int c0 = 0; (c0 < 2) && (!SymbolSelect(def_SymbolReplay, false)); c0++);\
                                CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);\
                                CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);\
                                CustomSymbolDelete(def_SymbolReplay);\
                                GlobalVariableDel(def_GlobalVariableReplay);\
                                GlobalVariableDel(def_GlobalVariableIdGraphics);\
                                Print("Replay service completed...");\
                        }\
//+------------------------------------------------------------------+\
                bool ViewReplay(ENUM_TIMEFRAMES arg1)\
                        {\
                                u_Interprocess info;\
\
                                if (m_IdReplay == -1) return false;\
                                if ((m_IdReplay = ChartFirst()) > 0) do\
                                {\
                                        if (ChartSymbol(m_IdReplay) == def_SymbolReplay)\
                                        {\
                                                ChartClose(m_IdReplay);\
                                                ChartRedraw();\
                                        }\
                                }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);\
                                Print("Wait for permission from [Market Replay] indicator to start replay ...");\
                                info.u_Value.IdGraphic = m_IdReplay = ChartOpen(def_SymbolReplay, arg1);\
                                ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");\
                                ChartRedraw(m_IdReplay);\
                                GlobalVariableDel(def_GlobalVariableIdGraphics);\
                                GlobalVariableTemp(def_GlobalVariableIdGraphics);\
                                GlobalVariableSet(def_GlobalVariableIdGraphics, info.u_Value.df_Value);\
                                while ((!GlobalVariableCheck(def_GlobalVariableReplay)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);\
\
                                return ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""));\
                        }\
//+------------------------------------------------------------------+\
                bool LoopEventOnTime(const bool bViewBuider, const bool bViewMetrics)\
                        {\
\
                                u_Interprocess Info;\
                                int iPos, iTest;\
\
                                iTest = 0;\
                                while ((iTest == 0) && (!_StopFlag))\
                                {\
                                        iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);\
                                        iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value) ? iTest : -1);\
                                        iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);\
                                        if (iTest == 0) Sleep(100);\
                                }\
                                if ((iTest < 0) || (_StopFlag)) return false;\
                                AdjustPositionToReplay(bViewBuider);\
                                m_MountBar.delay = 0;\
                                while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))\
                                {\
                                        CreateBarInReplay(bViewMetrics);\
                                        iPos = (int)(m_ReplayCount < m_Ticks.nTicks ? m_Ticks.Info[m_ReplayCount].time_msc - m_Ticks.Info[m_ReplayCount - 1].time_msc : 0);\
                                        m_MountBar.delay += (iPos < 0 ? iPos + 1000 : iPos);\
                                        if (m_MountBar.delay > 400)\
                                        {\
                                                if (ChartSymbol(m_IdReplay) == "") break;\
                                                GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value);\
                                                if (!Info.s_Infos.isPlay) return true;\
                                                Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);\
                                                GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);\
                                                Sleep(m_MountBar.delay - 20);\
                                                m_MountBar.delay = 0;\
                                        }\
                                }\
                                return (m_ReplayCount == m_Ticks.nTicks);\
                        }\
//+------------------------------------------------------------------+\
};\
//+------------------------------------------------------------------+\
#undef macroRemoveSec\
#undef def_SymbolReplay\
//+------------------------------------------------------------------+\
```\
\
As you can see, there is nothing unusual about this. However, it is much larger than the previous version. Yet, we managed to do everything that we implemented before. But I want to emphasize that all the points noted are not actually part of the C\_Replay class. These points are inherited. This happens because **I don't want them** to be accessible outside the C\_Replay class. To achieve this, we inherit things privately. In this way, we guarantee the integrity of the inherited information. This class only has two functions that can actually be accessed externally, since the constructor and destructor are not taken into account.\
\
But before we talk about the class constructor and destructor, let's take a look at two functions that can be accessed from outside the class. At some point I decided to keep only one of them, but for practical reasons I left both functions. It is simpler this way. We considered the **LoopEventOnTime** function in previous articles. And since it has not undergone any modifications here, there is no point in giving additional explanations. We can skip it and focus on the one that has undergone changes: **ViewReplay**.\
\
The ViewReplay function has only one change, namely the check. Here we check if the class constructor was able to successfully initialize the class. If it fails, the function will return a value that should cause the replay service to terminate. This is the only modification compared to what was in previous articles.\
\
### Final considerations\
\
Given all the changes, I suggest that you study the new materials, and also compare the attached code with other codes presented in previous articles. In the next article we will begin a completely different and rather interesting topic. See you!\
\
Translated from Portuguese by MetaQuotes Ltd.\
\
Original article: [https://www.mql5.com/pt/articles/11095](https://www.mql5.com/pt/articles/11095)\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/11095.zip "Download all attachments in the single ZIP archive")\
\
[Market\_Replay\_6vi\_16\_.zip](https://www.mql5.com/en/articles/download/11095/market_replay_6vi_16_.zip "Download Market_Replay_6vi_16_.zip")(10360 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)\
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)\
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)\
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)\
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)\
- [Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)\
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)\
\
**[Go to discussion](https://www.mql5.com/en/forum/458251)**\
\
![Developing a Replay System — Market simulation (Part 17): Ticks and more ticks (I)](https://c.mql5.com/2/55/replay-p17-avatar.png)[Developing a Replay System — Market simulation (Part 17): Ticks and more ticks (I)](https://www.mql5.com/en/articles/11106)\
\
Here we will see how to implement something really interesting, but at the same time very difficult due to certain points that can be very confusing. The worst thing that can happen is that some traders who consider themselves professionals do not know anything about the importance of these concepts in the capital market. Well, although we focus here on programming, understanding some of the issues involved in market trading is paramount to what we are going to implement.\
\
![Neural networks made easy (Part 52): Research with optimism and distribution correction](https://c.mql5.com/2/57/optimistic-actor-critic-avatar.png)[Neural networks made easy (Part 52): Research with optimism and distribution correction](https://www.mql5.com/en/articles/13055)\
\
As the model is trained based on the experience reproduction buffer, the current Actor policy moves further and further away from the stored examples, which reduces the efficiency of training the model as a whole. In this article, we will look at the algorithm of improving the efficiency of using samples in reinforcement learning algorithms.\
\
![Introduction to MQL5 (Part 1): A Beginner's Guide into Algorithmic Trading](https://c.mql5.com/2/61/Beginnerrs_Guide_into_Algorithmic_Trading_LOGO.png)[Introduction to MQL5 (Part 1): A Beginner's Guide into Algorithmic Trading](https://www.mql5.com/en/articles/13738)\
\
Dive into the fascinating realm of algorithmic trading with our beginner-friendly guide to MQL5 programming. Discover the essentials of MQL5, the language powering MetaTrader 5, as we demystify the world of automated trading. From understanding the basics to taking your first steps in coding, this article is your key to unlocking the potential of algorithmic trading even without a programming background. Join us on a journey where simplicity meets sophistication in the exciting universe of MQL5.\
\
![The case for using a Composite Data Set this Q4 in weighing SPDR XLY's next performance](https://c.mql5.com/2/61/Composite_Data_Set_this_Q4_in_weighing_SPDR_XLY_LOGO.png)[The case for using a Composite Data Set this Q4 in weighing SPDR XLY's next performance](https://www.mql5.com/en/articles/13775)\
\
We consider XLY, SPDR’s consumer discretionary spending ETF and see if with tools in MetaTrader’s IDE we can sift through an array of data sets in selecting what could work with a forecasting model with a forward outlook of not more than a year.\
\
[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mgwmtdgkxgrtgbyatmtueoxffwxrncza&ssn=1769180679265007228&ssn_dr=0&ssn_sr=0&fv_date=1769180679&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11095&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20%E2%80%94%20Market%20simulation%20(Part%2016)%3A%20New%20class%20system%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918067979985077&fz_uniq=5069033180339568646&sv=2552)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).