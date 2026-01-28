---
title: Developing a Replay System — Market simulation (Part 23): FOREX (IV)
url: https://www.mql5.com/en/articles/11177
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:03:00.322435
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/11177&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068984101748277154)

MetaTrader 5 / Tester


### Introduction

In the previous article, " [Developing a Replay System — Market simulation (Part 22): FOREX (III)](https://www.mql5.com/en/articles/11174), we made some changes to the system to enable the simulator to generate information based on the Bid price, and not just based on Last. But these modifications did not satisfy me, and the reason is simple: we are duplicating the code, and this does not suit me at all.

There is a point in that article where I made my dissatisfaction clear:

"... Don't ask me why. But for some strange reason that I personally have no idea about, we have to add this line here. If you don't add it, the value indicated in the tick volume will be incorrect. Pay attention that there is a condition in the function. This avoids problems when using the fast positioning system, and prevents the appearance of a strange bar that would be out of time on the system's chart. Although this is a very strange reason, everything else works as expected. This will be a new calculation where we will count ticks in the same way - both when working with a Bid-based asset and when working with the Last-based instrument.

However, since the code for the article was ready and the article was almost completed, I left everything as is, but this really bothered me. It makes no sense for code to work in some situations and not in others. Even debugging the code and trying to find the cause of the error, I could not find it. But after leaving the code alone for a moment and looking at the system flowchart (yes, you should always try to use a flowchart to speed up coding), I noticed that I could make some changes to avoid code duplication. And to make matters worse, the code was actually duplicated. This caused a problem that I could not solve. But there is a solution, and we will start this article with a solution to this problem, since its presence can make it impossible to correctly write simulator code to work with forex market data.

### Solving the problem with tick volume

In this topic I will show how the problem causing the tick volume to fail was resolved. First, I had to change the tick reading code, as shown below:

```
datetime LoadTicks(const string szFileNameCSV, const bool ToReplay = true)
    {
        int      MemNRates,
                 MemNTicks;
        datetime dtRet = TimeCurrent();
        MqlRates RatesLocal[],
                 rate;
        bool     bNew;

        MemNRates = (m_Ticks.nRate < 0 ? 0 : m_Ticks.nRate);
        MemNTicks = m_Ticks.nTicks;
        if (!Open(szFileNameCSV)) return 0;
        if (!ReadAllsTicks(ToReplay)) return 0;
        rate.time = 0;
        for (int c0 = MemNTicks; c0 < m_Ticks.nTicks; c0++)
        {
            if (!BuildBar1Min(c0, rate, bNew)) continue;
            if (bNew) ArrayResize(m_Ticks.Rate, (m_Ticks.nRate > 0 ? m_Ticks.nRate + 2 : def_BarsDiary), def_BarsDiary);
            m_Ticks.Rate[(m_Ticks.nRate += (bNew ? 1 : 0))] = rate;
        }
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

Previously, this code was part of the code that converts ticks to 1-minute bars, but now we will use different code. The reason is that now this call will serve more than one purpose, and the work it does will also be used to create repeating bars. This will avoid duplicating the code for creating bars in classes.

Let's look at the conversion code:

```
inline bool BuildBar1Min(const int iArg, MqlRates &rate, bool &bNew)
inline void BuiderBar1Min(const int iFirst)
   {
      MqlRates rate;
      double   dClose = 0;
      bool     bNew;

      rate.time = 0;
      for (int c0 = iFirst; c0 < m_Ticks.nTicks; c0++)
      {
         switch (m_Ticks.ModePlot)
         {
            case PRICE_EXCHANGE:
               if (m_Ticks.Info[c0].last == 0.0) continue;
               if (m_Ticks.Info[iArg].last == 0.0) return false;
               dClose = m_Ticks.Info[c0].last;
               break;
            case PRICE_FOREX:
               dClose = (m_Ticks.Info[c0].bid > 0.0 ? m_Ticks.Info[c0].bid : dClose);
               if ((dClose == 0.0) || (m_Ticks.Info[c0].bid == 0.0)) continue;
               if ((dClose == 0.0) || (m_Ticks.Info[iArg].bid == 0.0)) return false;
               break;
         }
         if (bNew = (rate.time != macroRemoveSec(m_Ticks.Info[c0].time)))
         {
            ArrayResize(m_Ticks.Rate, (m_Ticks.nRate > 0 ? m_Ticks.nRate + 2 : def_BarsDiary), def_BarsDiary);
            rate.time = macroRemoveSec(m_Ticks.Info[c0].time);
            rate.real_volume = 0;
            rate.tick_volume = (m_Ticks.ModePlot == PRICE_FOREX ? 1 : 0);
            rate.open = rate.low = rate.high = rate.close = dClose;
         }else
         {
            rate.close = dClose;
            rate.high = (rate.close > rate.high ? rate.close : rate.high);
            rate.low = (rate.close < rate.low ? rate.close : rate.low);
            rate.real_volume += (long) m_Ticks.Info[c0].volume_real;
            rate.tick_volume++;
         }
         m_Ticks.Rate[(m_Ticks.nRate += (bNew ? 1 : 0))] = rate;
      }
      return true;
   }
```

All crossed-out elements in the code were removed as they prevented correct creation of elements for use in the C\_Replay class. But on the other hand, I had to add these points, to inform the caller what happened in the conversion.

Note that initially this function was private in the C\_FileTicks class. I changed its access level so it can be used in the C\_Replay class. Despite of this, I don't want it to go too far beyond these limits, so it will be not public but protected. This way we can limit access to the maximum level allowed by the C\_Replay class. As you remember, the highest level is the C\_Replay class. Therefore, only procedures and functions declared as public in the C\_Replay class can be accessed outside the class. The internal design of the system must be completely hidden within this C\_Replay class.

Now let's look at the new bar creation function.

```
inline void CreateBarInReplay(const bool bViewTicks)
   {
#define def_Rate m_MountBar.Rate[0]

      bool    bNew;
      double  dSpread;
      int     iRand = rand();

      if (BuildBar1Min(m_ReplayCount, def_Rate, bNew))
      {
         m_Infos.tick[0] = m_Ticks.Info[m_ReplayCount];
         if ((!m_Ticks.bTickReal) && (m_Ticks.ModePlot == PRICE_EXCHANGE))
         {
            dSpread = m_Infos.PointsPerTick + ((iRand > 29080) && (iRand < 32767) ? ((iRand & 1) == 1 ? m_Infos.PointsPerTick : 0 ) : 0 );
            if (m_Infos.tick[0].last > m_Infos.tick[0].ask)
            {
               m_Infos.tick[0].ask = m_Infos.tick[0].last;
               m_Infos.tick[0].bid = m_Infos.tick[0].last - dSpread;
            }else   if (m_Infos.tick[0].last < m_Infos.tick[0].bid)
            {
               m_Infos.tick[0].ask = m_Infos.tick[0].last + dSpread;
               m_Infos.tick[0].bid = m_Infos.tick[0].last;
            }
         }
         if (bViewTicks) CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
         CustomRatesUpdate(def_SymbolReplay, m_MountBar.Rate);
      }
      m_ReplayCount++;
#undef def_Rate
   }
```

Now the creation occurs at the same point where we converted ticks into bars. This way, if something goes wrong during the conversion process, we will immediately notice the error. This is because the same code that places 1-minute bars on the chart during fast forwarding is also used for the positioning system to place bars during normal performance. In other words, the code that is responsible for this task is not duplicated anywhere else. This way we get a much better system for both maintenance and improvement. But I also want you to notice something important that we added to the above code. Simulation of Bid and Ask prices will only happen if we are in a simulated system and the simulated data is of the stock market type. That is, if the plotting is based on Bid, this simulation will no longer be performed. This is important for what we'll start designing in the next topic.

### Let's start the simulation of the Bid-based presentation (forex mode).

In what follows, we will consider exclusively the C\_Simulation class. We will do this in order to model data that is not covered by the current implementation of the system. But first we need to do one small thing:

```
bool BarsToTicks(const string szFileNameCSV)
   {
      C_FileBars *pFileBars;
      int         iMem = m_Ticks.nTicks,
                  iRet;
      MqlRates    rate[1];
      MqlTick     local[];

      pFileBars = new C_FileBars(szFileNameCSV);
      ArrayResize(local, def_MaxSizeArray);
      Print("Converting bars to ticks. Please wait...");
      while ((*pFileBars).ReadBar(rate) && (!_StopFlag))
      {
         ArrayResize(m_Ticks.Rate, (m_Ticks.nRate > 0 ? m_Ticks.nRate + 3 : def_BarsDiary), def_BarsDiary);
         m_Ticks.Rate[++m_Ticks.nRate] = rate[0];
         if ((iRet = Simulation(rate[0], local)) < 0)
         {
            ArrayFree(local);
            delete pFileBars;
            return false;
         }
         for (int c0 = 0; c0 <= iRet; c0++)
         {
            ArrayResize(m_Ticks.Info, (m_Ticks.nTicks + 1), def_MaxSizeArray);
            m_Ticks.Info[m_Ticks.nTicks++] = local[c0];
         }
      }
      ArrayFree(local);
      delete pFileBars;
      m_Ticks.bTickReal = false;

      return ((!_StopFlag) && (iMem != m_Ticks.nTicks));
   }
```

If something goes wrong and we want to shut down the system completely, we will need a way to tell other classes that the simulation failed. This is the easiest way to do this. However, I don't really like the way we created this function. Although it works, it is missing some things that we need to tell the C\_Simulation class. After analyzing the code, I decided to change the way the function works. It needs to be changed to avoid code duplication. So forget about the previous function. Although it works, we will actually use the following one:

```
int SetSymbolInfos(void)
   {
      int iRet;

      CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_DIGITS, iRet = (m_Ticks.ModePlot == PRICE_EXCHANGE ? 4 : 5));
      CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_TRADE_CALC_MODE, m_Ticks.ModePlot == PRICE_EXCHANGE ? SYMBOL_CALC_MODE_EXCH_STOCKS : SYMBOL_CALC_MODE_FOREX);
      CustomSymbolSetInteger(def_SymbolReplay, SYMBOL_CHART_MODE, m_Ticks.ModePlot == PRICE_EXCHANGE ? SYMBOL_CHART_MODE_LAST : SYMBOL_CHART_MODE_BID);

      return iRet;
   }
//+------------------------------------------------------------------+
   public  :
//+------------------------------------------------------------------+
      bool BarsToTicks(const string szFileNameCSV)
      {
         C_FileBars      *pFileBars;
         C_Simulation    *pSimulator = NULL;
         int             iMem = m_Ticks.nTicks,
                         iRet = -1;
         MqlRates        rate[1];
         MqlTick         local[];
         bool            bInit = false;

         pFileBars = new C_FileBars(szFileNameCSV);
         ArrayResize(local, def_MaxSizeArray);
         Print("Converting bars to ticks. Please wait...");
         while ((*pFileBars).ReadBar(rate) && (!_StopFlag))
         {
            if (!bInit)
            {
               m_Ticks.ModePlot = (rate[0].real_volume > 0 ? PRICE_EXCHANGE : PRICE_FOREX);
               pSimulator = new C_Simulation(SetSymbolInfos());
               bInit = true;
            }
            ArrayResize(m_Ticks.Rate, (m_Ticks.nRate > 0 ? m_Ticks.nRate + 3 : def_BarsDiary), def_BarsDiary);
            m_Ticks.Rate[++m_Ticks.nRate] = rate[0];
            if (pSimulator == NULL) iRet = -1; else iRet = (*pSimulator).Simulation(rate[0], local);
            if (iRet < 0) break;
            for (int c0 = 0; c0 <= iRet; c0++)
            {
               ArrayResize(m_Ticks.Info, (m_Ticks.nTicks + 1), def_MaxSizeArray);
               m_Ticks.Info[m_Ticks.nTicks++] = local[c0];
            }
         }
         ArrayFree(local);
         delete pFileBars;
         delete pSimulator;
         m_Ticks.bTickReal = false;

         return ((!_StopFlag) && (iMem != m_Ticks.nTicks) && (iRet > 0));
      }
```

The second option is much more effective from the point of view of our goals. In addition, we avoid duplicating the code, mainly because by using it we will get the following advantages:

- Eliminate inheritance of the C\_Simulation class. This will make the system even more flexible.
- Initialization of asset data, which was previously performed only when using real ticks.
- The appropriate width of symbols used in the graphic display.
- Using the C\_Simulation class as a pointer. That is, more efficient use of system memory, since after the class has completed its work, the memory it occupied will be freed.
- Guarantee that there is only one entry point and one exit point from the function.

Some things will change compared to the previous article. But let's continue with the implementation of the C\_Simulation class. The main detail for developing the C\_Simulation class is that we can have any number of ticks in the system. While this is not a problem (at least for now), the difficulty is that in many cases the range we will have to cover between the high and low will already be much larger than the number of ticks reported or which can be created. This is not counting the section that starts from the open price and goes to one of the extremes, and the section that starts from one of the extremes and goes up to the close price. If we implement this calculation using a RANDOM WALK, then in a huge number of cases this will be impossible. Therefore, we will have to eliminate the random walk that we created in previous articles and develop a new method for generating ticks. As I said, the problem with forex is not so clear-cut.

The problem with this approach is that you often have to create and make two different methods work as harmoniously as possible. The worst part is this: in some cases, the random walk simulation is much closer to what happens in the real asset. But when we are dealing with low trading volume (less than 500 trades per minute), then random walk is completely inappropriate. In this situation, we can use a more exotic approach to cover all possible cases. The first thing we will do (since we need to initialize the class) is to define a constructor for the class, the code for which can be seen below:

```
C_Simulation(const int nDigits)
   {
      m_NDigits       = nDigits;
      m_IsPriceBID    = (SymbolInfoInteger(def_SymbolReplay, SYMBOL_CHART_MODE) == SYMBOL_CHART_MODE_BID);
      m_TickSize      = SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE);
   }
```

Here we simply initialize the private data of the class so as not to look for it elsewhere. Therefore, ensure that all settings are set correctly in the configuration file of the asset being simulated, including the plotting type. Otherwise, strange errors may occur in the system.

Now we can start moving forward since we have done some basic initialization of the class. Let's start looking at the problems that need to be solved. First, we need to generate a random time value, but this time must be able to handle all the ticks that will be generated on 1-minute bars. This is actually the simplest part of the implementation. But before we start creating functions, we need to create a special type of procedure shown below:

```
template < typename T >
inline T RandomLimit(const T Limit01, const T Limit02)
   {
      T a = (Limit01 > Limit02 ? Limit01 - Limit02 : Limit02 - Limit01);
      return (Limit01 >= Limit02 ? Limit02 : Limit01) + ((T)(((rand() & 32767) / 32737.0) * a));
   }
```

What exactly does this procedure give us? It can be surprising to see this feature without understanding what's going on. So I'll try to explain as simply as possible what this function actually does and why it looks so strange.

In the new code, we need a type of function that is capable of generating a random value between two extremes. In some cases, we will need this value to be formed as a Double data type, while in other cases we will need integer values. Creating two virtually identical procedures to perform the same type of factorization would require considerable effort. To avoid this, we force, or rather, tell the compiler that we need to use the same factorization and overload it so that in the code we can use the same function, but in the executable form we will actually have two different functions. We use this declaration for this purpose – this defines the type, which in this case is the letter T. This needs to be repeated wherever we need the compiler to set the type. Therefore, you should be careful not to mix anything up. Let the compiler make corrections to avoid casting problems.

Thus, we will always perform the same calculation, but it will be adjusted depending on the type of variable used. The compiler will do this, since it will be the one who decides which type is correct. This way we will be able to generate a pseudo-random number in each call, regardless of the type used, but note that the types of both boundaries should be the same. In other words, you cannot mix double with integer or long integer with short integer. This won't work. This is the only limitation of this approach when we use type overloading.

But we're not done yet. We have created the above function to avoid generating macros in the code of the C\_Simulation class. Let's now move on to the next step - generating the simulation timing system. This generation can be seen in the code below:

```
inline void Simulation_Time(int imax, const MqlRates &rate, MqlTick &tick[])
   {
      for (int c0 = 0, iPos, v0 = (int)(60000 / rate.tick_volume), v1 = 0, v2 = v0; c0 <= imax; c0++, v1 = v2, v2 += v0)
      {
         iPos = RandomLimit(v1, v2);
         tick[c0].time = rate.time + (iPos / 1000);
         tick[c0].time_msc = iPos % 1000;
      }
   }
```

Here we simulate time to be slightly random. At first glance this might look quite confusing. But believe me, the time here is random, although it still does not correspond to the logic expected by the C\_Replay class. This is because the value in milliseconds is set incorrectly. This adjustment will be made later. Here we just want the time to be generated randomly, but within a 1 minute bar. How can we do this? First, we divide the time of 60 seconds, which is actually 60,000 milliseconds, by the number of ticks that need to be generated. This value is important to us as it will tell us what limit range we will use. After that, in each iteration of the loop, we will perform several simple assignments. Now the secret to generating a random timer lies in these three lines inside the loop. In the first line, we ask the compiler to generate a call in which we will use integer data. This call will return a value in the specified range. We will then perform two very simple calculations. We first fit the generated value to the minute bar time, and then use the same generated value to fit the time in milliseconds. Thus, each tick will have a completely random time value. Remember that at this early stage we are only correcting the time. The purpose of this setting is to avoid excessive predictability.

Continuing, let's simulate prices. Let me remind you again that we will only focus on the Bid-based plotting system. We will then link the simulation system so that we have a much more general way of doing such simulation that covers both Bid and Last. Here we focus on Bid. To create simulation in this first step, we will always keep the spread at the same distance. We won't complicate the code before we test whether it actually works. This first simulation is performed using several fairly short functions. We'll use short functions to make everything as modular as possible. Later you will see the reason for this.

Let's now look at the first of the calls which will be made to create the Bid-based simulation:

```
inline void Simulation_BID(int imax, const MqlRates &rate, MqlTick &tick[])
   {
      bool    bHigh  = (rate.open == rate.high) || (rate.close == rate.high),
      bLow = (rate.open == rate.low) || (rate.close == rate.low);

      Mount_BID(0, rate.open, rate.spread, tick);
      for (int c0 = 1; c0 < imax; c0++)
      {
         Mount_BID(c0, NormalizeDouble(RandomLimit(rate.high, rate.low), m_NDigits), rate.spread, tick);
         bHigh = (rate.high == tick[c0].bid) || bHigh;
         bLow = (rate.low == tick[c0].bid) || bLow;
      }
      if (!bLow) Mount_BID(Unique(imax, rate.high, tick), rate.low, rate.spread, tick);
      if (!bHigh) Mount_BID(Unique(imax, rate.low, tick), rate.high, rate.spread, tick);
      Mount_BID(imax, rate.close, rate.spread, tick);
   }
```

The above function is quite simple to understand. Although, it would seem that the most difficult part is the random construction of the Bid value. But even in this case, everything is quite simple. We will generate pseudo-random values in the range between the maximum and minimum values of the bar. But notice that I'm normalizing the value. This is because the value generated is usually outside the price range. That is why we need to normalize it. But I think the rest of the function should be clear.

If you look closely, you will notice that we have two functions that are often mentioned in the modeling part: MOUNT\_BID and UNIQUE. Each of them serves a specific purpose. Lets start with **Unique**. Its code is shown below:

```
inline int Unique(const int imax, const double price, const MqlTick &tick[])
   {
      int iPos = 1;

      do
      {
         iPos = (imax > 20 ? RandomLimit(1, imax - 1) : iPos + 1);
      }while ((m_IsPriceBID ? tick[iPos].bid : tick[iPos].last) == price);

      return iPos;
   }
```

This function prevents the deletion of the value of one of the limits or any other price when generating a random position. For now, we will use it only for the limits. Note that we can use either the simulated Bid value or the simulated Last value. Now we work only with Bid. This is the sole purpose of this function: to ensure that we do not overwrite the limit value.

Now let's look at the Mount\_BID function, the code for which is given below:

```
inline void Mount_BID(const int iPos, const double price, const int spread, MqlTick &tick[])
   {
      tick[iPos].bid = price;
      tick[iPos].ask = NormalizeDouble(price + (m_TickSize * spread), m_NDigits);
   }
```

Although at this early stage this code is quite simple and doesn't approach the beauty of pure programming, it makes our life a lot easier. It allows you to avoid repeating the code in several places and, most importantly, helps you remember to normalize the value that should be placed in the Ask price position. If this normalization is not performed, then problems will arise when using this Ask value further. The ASK price value will always be offset by the spread value. However, for now this offset is always constant. This is because this is the first implementation, and if we implemented the randomization system now, it would be completely unclear why and how the spread value is made arbitrary.

The spread value shown here is actually the value shown on the specific 1 minute bar. Each bar may have a different spread, but there is something else you need to understand. If you are running a simulation to obtain a system that resembles what would happen in a real market (i.e. the data contained in a real tick file), then you will notice that the spread used is the smaller of the values present in the formation of the 1 minute bar. But if you are running a random simulation in which the data may or may not resemble what would happen in the real market, then that spread can have any value. Here we will stick to the idea of constructing what might happen in the market. Therefore, the spread value will always be the one specified in the bars file.

There is one more function required for the system to work. It should be responsible for setting up the timing so that the C\_Replay class has the correct timing values. This code can be seen below:

```
inline void CorretTime(int imax, MqlTick &tick[])
   {
      for (int c0 = 0; c0 <= imax; c0++)
         tick[c0].time_msc += (tick[c0].time * 1000);
   }
```

This function simply adjusts the specified time in milliseconds accordingly. If you look closely, you can see that the calculations are the same as those used in the function that loads the actual ticks from a file. The reason for this modular approach is that it can be interesting to keep records of each of the functions performed. If all the code were interconnected, then creating such records would be more difficult. However, in this way it is possible to create records and study them, and therefore check what should or should not be improved to meet specific needs.

**Important note**: at this early stage I will block the use of the Last-based system. We will modify it in some places to make it work with assets during periods of low liquidity. This is currently not possible, but we will fix this later. If you try to run a simulation based on the Last prices now, the system won't let you do it. We will fix this later.

To make sure of this, we will use one of the programming techniques. It will be something very complex and well managed. See the code below:

```
inline int Simulation(const MqlRates &rate, MqlTick &tick[])
   {
      int imax;

      imax = (int) rate.tick_volume - 1;
      Simulation_Time(imax, rate, tick);
      if (m_IsPriceBID) Simulation_BID(imax, rate, tick); else return -1;
      CorretTime(imax, tick);

      return imax;
   }
```

Every time the system uses the Last plotting mode, it will throw an error. This is because we will need to improve the Last-based simulation. Therefore, I had to add this complex and sophisticated trick. If you try to run a Last-based simulation, you will get a negative value. Isn't it a complicated method?

But before we conclude this article, we will once again dwell on the issue of Bid plotting modeling. As a result, we will have a slightly improved way of randomization. Basically, we need to change one moment so that it has a random spread value. This can be done in the Mount\_Bid or Simulation\_Bid function. In some ways this is not a big deal, but in order to ensure the minimum spread value specified in the 1 minute bar file, we will make a modification to the function shown below:

```
inline void Simulation_BID(int imax, const MqlRates &rate, MqlTick &tick[])
   {
      bool    bHigh  = (rate.open == rate.high) || (rate.close == rate.high),
      bLow = (rate.open == rate.low) || (rate.close == rate.low);

      Mount_BID(0, rate.open, rate.spread, tick);
      for (int c0 = 1; c0 < imax; c0++)
      {
         Mount_BID(c0, NormalizeDouble(RandomLimit(rate.high, rate.low), m_NDigits), (rate.spread + RandomLimit((int)(rate.spread | (imax & 0xF)), 0)), tick);
         bHigh = (rate.high == tick[c0].bid) || bHigh;
         bLow = (rate.low == tick[c0].bid) || bLow;
      }
      if (!bLow) Mount_BID(Unique(imax, rate.high, tick), rate.low, rate.spread, tick);
      if (!bHigh) Mount_BID(Unique(imax, rate.low, tick), rate.high, rate.spread, tick);
      Mount_BID(imax, rate.close, rate.spread, tick);
   }
```

Here we provide randomization of the spread value, however, this randomization is only for demonstration purposes. If you wish, you can do things a little differently in terms of limits. We'll just need to tweak things a little. Now you should understand that I'm using this randomization, which seems a little strange to some, but here's what I'm actually doing: I'm making sure that the greatest possible value can be used to randomize the spread. This value is based on a calculation where we bitwise combine the spread value with a value that can range from 1 to 16 since we are only using a portion of all bits. Note that if the spread is zero (and at some points it will actually be zero), we will still get a value that will be at least 3, since values 1 and 2 do not actually create randomization of the spread. This is because a value of 1 only indicates the open price equal to close, while a value of 2 indicates that the open can be either equal or different from the close. But in this case, it is the value 2 that will actually create the value. In all other cases, we will be dealing with the creation of randomization in the spread.

I hope now it is clear why I didn't put randomization to the Mount\_Bid function. If I did this, there would be some points where the minimum spread reported by the bars file would not be true. But, as I already said, you can freely experiment and adapt the system to your taste and style.

### Conclusion

In this article, we solved the problems associated with code duplication. I think it's now clear what problems arise when using duplicate code. In very large projects you always need to be careful with this. Even this code, which is not that big, can have serious problems because of this carelessness.

One last detail that also deserves mention is that in a real tick file there are times when we actually have some kind of "false" movement. But this does not happen here; such "false" movements occur when variations occur in only one of the prices, either Bid or ASK. However, for the sake of simplicity, I left such situations without attention. In my opinion, this does not make much sense for the the of a system that simulates the market. This would not bring operational improvements. For every change to Bid without Ask, we would have to do Ask without Bid. This is necessary to maintain the balance required by the real market.

This closes the question of Bid-based modeling, at least for this first attempt. In the future, I may make changes to this system to make it work differently. But when using it with the forex data, I noticed that it works quite well, although it may not be sufficient for other markets.

The attached file will give you access to the system in its current state of development. However, as I already said in this article, you should not try to carry out modeling with stock market assets, you can only do it with forex instruments. Although you can replay any instruments, simulation is disabled for exchange-traded assets. In the next article, we will fix this by improving the stock market replay system so that it can work in low liquidity environments. This concludes our consideration of simulation. See you in the next article!

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11177](https://www.mql5.com/pt/articles/11177)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11177.zip "Download all attachments in the single ZIP archive")

[Market\_Replay\_7vx23.zip](https://www.mql5.com/en/articles/download/11177/market_replay_7vx23.zip "Download Market_Replay_7vx23.zip")(14388.45 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/462716)**
(5)


![Philip Tweens](https://c.mql5.com/avatar/2022/8/62FBFAAA-F47E.png)

**[Philip Tweens](https://www.mql5.com/en/users/alimehr882657)**
\|
17 Aug 2023 at 19:39

Hello dear Daniel,

Congratulations on this great system you have designed.

I have encountered some problems while testing your system and I need your help.

The first thing is that I saved the tick data for a month and put it for replay. However, in the 1 minute TimeFrame for the replay in a month with the change of the pin on the slider, the number of candles shown does not match the position of the pin on the slider, I have shown this in the attached video. I took the pin all the way to the end, but the candles only repeat for a small [number of](https://www.mql5.com/en/docs/series/bars "MQL5 documentation: Bars function") 1-minute [bars](https://www.mql5.com/en/docs/series/bars "MQL5 documentation: Bars function") (about 20 bars).

The second thing is that I need to change this system so that the way the bars move is like Strategy Tester, i.e. the position of the pin represents the speed at which the bars are displayed, or I have the possibility of moving bar by bar like the TradingView website. Your system is in such a way that it can be changed like this????

I would be grateful if you could guide me.

Yours sincerely,

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
17 Aug 2023 at 21:32

**Philip Tweens number of 1-minute [bars](https://www.mql5.com/en/docs/series/bars "MQL5 documentation: Bars function") (about 20 bars).**
**The second thing is that I need to change this system so that the way the bars move is like Strategy Tester, i.e. the position of the pin represents the speed at which the bars are displayed, or I have the possibility of moving bar by bar like the TradingView website. Your system is in such a way that it can be changed like this????**

**I would be grateful if you could guide me.**

**Yours sincerely,**

OK, let's go in parts, as JACK would say...😁👍

You may be very confused about this application, or rather, you may be expecting this application to be used for something that it wasn't intended to be used for in the first place. I'm not saying that it can't be used for something in particular, such as a strategy tester. But that wasn't the initial objective for implementing it.

On the first question: You may not have really understood how the replay/simulation will take place. Forget about the slider for a moment. When you play the system, it will retrieve the data that has been loaded, either as ticks or bars, and display it as bars on the graph, based on a time of 1 minute. This is regardless of the chart time you want to use. That's why the data in the file should be thought of as 1-minute bars. You shouldn't look at the data in the file as individual data. This application doesn't see it that way. It will always interpret bars, even two-hour bars, as 1-minute bars. **Always**.

If you are using bars, the application will automatically notice this and create a simulation so that each bar is approximately 1 minute long. By creating as many ticks as necessary for the values to be plotted correctly on the graph. If the data in the file is ticks, the system will launch each tick at the approximate interval defined between them, see previous articles to understand this. This interval can vary from a few milliseconds to several hours. But by doing this, anything in the interval will be treated as an auction or a trading halt. So if you use data with an interval of more than a day, or 24 hours, the application will most likely not be able to recognise the bars properly. This is the case if you use the slider to search for a new study point. For this reason, you should avoid using data with a time span of more than one day.

Remember, the application is designed to be used in a time equivalent to real time. In other words, short periods. To enter long periods in the study. If you need to use an average or indicator that requires many bars to be plotted. You **MUST NOT** use the data in the replay or simulator. You should use them as previous bars. This is the first point you should try to understand.

As for the second question: You imagine that the slider will search for a specific point. Indeed it does, but not in the way you want or imagine. To understand this better, take a look at the previous articles where the slider was implemented. There you'll see in detail how it actually searches for a particular position. But in this very question, you're confusing the use of the control. Since you also raise the idea that it might be used to modify the speed at which the bars are plotted. This doesn't happen at all. The plotting that you notice, when you drag the control and then press the play button, happens at a higher speed. It's actually an illusion created by the application. To show how the bars were created up to the point where you indicated for the simulation or replay to start, so that you can carry out your study.

My suggestion is: Read the previous articles carefully, and if you have any questions, post them as comments. This will make it much easier for you to understand what is really going on and how you can use the application with a good user experience. If you have any questions you can ask in the comments ... 😁👍

![Philip Tweens](https://c.mql5.com/avatar/2022/8/62FBFAAA-F47E.png)

**[Philip Tweens](https://www.mql5.com/en/users/alimehr882657)**
\|
18 Aug 2023 at 06:15

**daniel jose [#](https://www.mql5.com/pt/forum/452110#comment_48810834) :**

Okay. Let's break it down as JACK would say...😁👍

Perhaps you are very confused about this application, or rather, perhaps you are hoping that this application will come to serve something, which in fact it was not intended to be used in principle. I'm not saying that it can't be used for something in particular, like for example a strategy tester. But this was not the initial objective for it to be implemented.

About the first question: You might not have really understood how the replay / simulation will happen. Forget about the slider for a moment. When you press play on the system, it will fetch the data that has been loaded, either as ticks or bars, and will display them as bars on the chart, based on a time of 1 minute. This is independent of the timeframe you want to use. For this reason, the data, which must be in the file, must be thought of as bars of 1 minute. You should not look at file data as individual data. Because this application doesn't see them that way. It will always interpret bars even two hour bars as being 1 minute bars. **Always**.

If you are using bars, the application will automatically notice this, and will create a simulation so that each of the bars is approximately 1 minute long. Creating as many ticks as necessary for the values to be correctly plotted on the chart. If the data present in the file are ticks, the system will record each of the ticks in the approximate interval that is defined between them. See previous articles to understand this. Such an interval can vary from a few milliseconds to several hours. But by doing this, anything that is in range will be treated as either an auction or a trading hold. Thus, if you use data with an interval of more than one day, that is, 24 hours, the application most likely will not be able to recognise the bars properly. This is if you use the slider to look for a new study point. Therefore, one should avoid using data with a time greater than one day.

Remember the application was thought to be used in a time equivalent to real time. In other words, short periods. To enter with long periods in the study. In case you need to use some average or indicator that needs many bars to be plotted. You **MUST NOT** use the data in the replay or simulator. You must place them as prebars. This is the first point you should try to understand.

Now about the second question: You imagine that the slider will look for a specific point. Indeed it does, but not in the way you might want or imagine. To better understand, see the previous articles, where the control was implemented. There you will see in detail how he actually does to seek a certain position. But in this same question, you are confusing the use of the control. Since you also raise the idea that it maybe serves to modify the speed in plotting the bars. This actually doesn't happen at all. Such a plot that you notice, when dragging the control and then pressing the play button, in this at a higher speed. It is actually an illusion created by the application. To show how the bars were created up to the point where you indicated that the simulation or replay should start,

My suggestion is: Read the previous articles calmly, and if in doubt, post them as a comment. Because it will be much easier for you to understand what is actually happening and how you can use the application having a good user experience. Any questions you can ask in the comments... 😁👍

I don't think you understood what I meant and I probably expressed myself badly.

I understand the function of the slider. I put the data in Replay for a month (about 20 days). However, I moved the pin closer to the end of the slider, but only a few bars were drawn on the first day, when it should have been at least 15 days before reaching the desired point. Have I got it wrong? I imagine it's because of what you said about not using more than one day of data.

Regarding the speed at which the bars are displayed, I would like you to advise me on how to change the system in this way.

Thank you for your reply.

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
18 Aug 2023 at 10:31

**Philip Tweens [#](https://www.mql5.com/pt/forum/452110#comment_48817123):**

I don't think you understood what I meant and I probably expressed myself badly.

I understand the function of the slider. I put the data in Replay for a month (about 20 days). However, I moved the pin closer to the end of the slider, but only a few bars were drawn on the first day, when it should have been at least 15 days before reaching the desired point. Have I got it wrong? I imagine it's because of what you said about not using more than one day of data.

Regarding the speed at which the bars are displayed, I'd like you to advise me on how to alter the system in this way.

Thank you for your reply.

Changing the speed is very simple. Just go to the **C\_Replay** class and look for the **LoopEventOnTime** function. There is a **Sleep** call there. This is where we control the plotting speed when we're in play mode. But I believe this has been adequately explained in previous articles.

![Philip Tweens](https://c.mql5.com/avatar/2022/8/62FBFAAA-F47E.png)

**[Philip Tweens](https://www.mql5.com/en/users/alimehr882657)**
\|
18 Aug 2023 at 13:30

**Daniel Jose [#](https://www.mql5.com/pt/forum/452110#comment_48820395):**

Changing the speed is very simple. Just go to the **C\_Replay** class and look for the **LoopEventOnTime** function. There is a **Sleep** call there. This is where we control the plotting speed when we are in play mode. But I believe this has been adequately explained in previous articles.

No, I want to change the way the slider behaves. Instead of the position of the pin being equal to the position of a specific point, it represents the speed at which the bars are displayed for the replay in long time. Similar to what we see in the strategy tester.

Thank you.

![Benefiting from Forex market seasonality](https://c.mql5.com/2/59/Seasonal_analysis_logo_UP.png)[Benefiting from Forex market seasonality](https://www.mql5.com/en/articles/12996)

We are all familiar with the concept of seasonality, for example, we are all accustomed to rising prices for fresh vegetables in winter or rising fuel prices during severe frosts, but few people know that similar patterns exist in the Forex market.

![Neural networks are easy (Part 59): Dichotomy of Control (DoC)](https://c.mql5.com/2/58/logo__1.png)[Neural networks are easy (Part 59): Dichotomy of Control (DoC)](https://www.mql5.com/en/articles/13551)

In the previous article, we got acquainted with the Decision Transformer. But the complex stochastic environment of the foreign exchange market did not allow us to fully implement the potential of the presented method. In this article, I will introduce an algorithm that is aimed at improving the performance of algorithms in stochastic environments.

![Developing a Replay System — Market simulation (Part 24): FOREX (V)](https://c.mql5.com/2/57/replay_p24_avatar.png)[Developing a Replay System — Market simulation (Part 24): FOREX (V)](https://www.mql5.com/en/articles/11189)

Today we will remove a limitation that has been preventing simulations based on the Last price and will introduce a new entry point specifically for this type of simulation. The entire operating mechanism will be based on the principles of the forex market. The main difference in this procedure is the separation of Bid and Last simulations. However, it is important to note that the methodology used to randomize the time and adjust it to be compatible with the C\_Replay class remains identical in both simulations. This is good because changes in one mode lead to automatic improvements in the other, especially when it comes to handling time between ticks.

![Developing a Replay System — Market simulation (Part 22): FOREX (III)](https://c.mql5.com/2/57/replay_p22_avatar.png)[Developing a Replay System — Market simulation (Part 22): FOREX (III)](https://www.mql5.com/en/articles/11174)

Although this is the third article on this topic, I must explain for those who have not yet understood the difference between the stock market and the foreign exchange market: the big difference is that in the Forex there is no, or rather, we are not given information about some points that actually occurred during the course of trading.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/11177&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068984101748277154)

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