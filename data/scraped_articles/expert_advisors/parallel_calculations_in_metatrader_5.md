---
title: Parallel Calculations in MetaTrader 5
url: https://www.mql5.com/en/articles/197
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:29:54.505438
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/197&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071876422919729184)

MetaTrader 5 / Examples


### Introduction to the processor parallelism

Almost all modern PCs are able to perform multiple tasks simultaneously - due to the presence of several processor cores. Their number is growing every year - 2, 3, 4, 6 cores ... Intel has recently [demonstrated](https://www.mql5.com/go?link=https://www.cnet.com/news/intel-shows-off-80-core-processor/ "http://news.cnet.com/2100-1006_3-6158181.html") a working experimental 80-core processor (yes, this is not a typo - eighty cores - unfortunately, this computer will not appear in stores, since this processor was created solely for the purpose of studying the potential capabilities of technology).

Not all computer users (and not even all novice programmers) understand how it works. Therefore, someone will surely ask the question: why do we need a processor with so many cores, when even before (with a single core), the computer could run many programs simultaneously and they all worked? The trust is, this is not so. Let's look at the following diagram.

![Figure 1. Parallel execution of applications](https://c.mql5.com/2/2/figure1.gif)

Figure 1. Parallel execution of applications

**Case A** on the diagram shows what happens when a single program is run on a single-core processor. The processor dedicates all of its time to its implementation, and the program performs some amount of work over time T.

**Case B** \- 2 programs launched. But the processor is arranged in such a way that physically, at any one point of time, one of its cores can execute only one command, so it has to constantly switch between the two programs: it will execute some of the first, then the second, etc. This happens very quickly, many times per second, so it looks like the processor executes both programs simultaneously. In reality however, their execution will take twice as much longer than if each program was executed on the processor separately.

**Case C** shows that this problem is effectively solved if the number of cores in a processor corresponds to the number of running programs. Each program has at its disposal a separate core, and the speed of its execution increases, just like in case A.

**Case D** is a response to the common delusion of many users. They believe that if a program is running on a multi-core processor, then it is executed several times faster. In general, this can not be true because the processor is not able to independently divide the program into separate parts, and execute them all simultaneously.

For example, if the program first asks for a password, and then its verifications is performed, it would be unacceptable to perform the password prompt on one core, and the verification on another, at the same time. The verification simply will never succeed, because the password, at the time of its inception, has not yet been inputed.

The processor doesn't know all of the designs the programmer has implemented, nor the entire logic of the program's work, and therefore it can not independently separate the program amongst the cores. So if we run a single program in a multi-core system, it will use only one core, and will be executed with the same speed as if it was run on a single-core processor.

**Case E** explains what needs to be done to make the program use all of its cores and be executed quicker. Since the programmer knows the logic of the program, he should, during its development, somehow mark those parts of the program, which can be executed at the same time. The program, during its execution, will communicate this information to the processor, and the processor will then allocate the program to the required number of cores.

### Parallelism in MetaTrader

In the previous chapter, we figured out what needs to be done in order to use all of the CPU cores and to accelerate the execution of programs: we need to somehow allocate the parallelizable code of the program into separate threads. In many programming languages there are special classes or operators for this. But there is no such built-in instrument in the MQL5 language. So what can we do?

There are two ways to solve this problem:

| 1\. Use DLL | 2\. Use non-language resources of MetaTrader |
| --- | --- |
| By creating a DLL in a language that has a built-in tool for parallelization, we will obtain the parallelization in the MQL5-EA as well. | According to the information from MetaTrader developers, the architecture of the client terminal is multi-threaded. Hence, under certain conditions, the incoming market data is processed in separate threads. Thus, if we can find a way to separate the code of our program to a number of EAs or indicators, then MetaTrader will be able to use to a number of CPU cores for its execution. |

The first method we won't discuss in this article. It is clear that in DLL we can implement anything we want. We will try to find a solution, which will involve only the standard means of MetaTrader and which will not require the use of any languages other than MQL5.

And so, more about the second method. We will need to perform a series of experiments to find out exactly how multiple cores are supported in MetaTrader. To do this, let's create a test indicator and a test EAs, which will perform any ongoing work, which heavily loads the CPU.

I wrote the following **i-flood** indicator:

```
//+------------------------------------------------------------------+
//|                                                      i-flood.mq5 |
//+------------------------------------------------------------------+
#property indicator_chart_window

input string id;
//+------------------------------------------------------------------+
void OnInit()
  {
   Print(id,": OnInit");
  }
//+------------------------------------------------------------------+
int OnCalculate(const int rt,const int pc,const int b,const double &p[])
  {
   Print(id,": OnCalculate Begin");

   for (int i=0; i<1e9; i++)
     for (int j=0; j<1e1; j++);

   Print(id,": OnCalculate End");
   return(0);
  }
//+------------------------------------------------------------------+
```

And the **e-flood** EA analogous to it:

```
//+------------------------------------------------------------------+
//|                                                      e-flood.mq5 |
//+------------------------------------------------------------------+
input string id;
//+------------------------------------------------------------------+
void OnInit()
  {
   Print(id,": OnInit");
  }
//+------------------------------------------------------------------+
void OnTick()
  {
   Print(id,": OnTick Begin");

   for (int i=0; i<1e9; i++)
     for (int j=0; j<1e1; j++);

   Print(id,": OnTick End");
  }
//+------------------------------------------------------------------+
```

Further, by opening various combinations of chart windows (one chart, two charts with the same symbol, two charts with different symbols), and by placing on them one or two copies of this indicator or EA, we can observe how the terminal uses CPU cores.

These indicators and EA also send messages to the log, and it is interesting to observe the sequence of their appearance. I won't provide these logs since you can generate them yourself, but in this article, we are interested in finding out how many cores and in what combinations of charts are used by the terminal.

We can measure the number of working cores through the Windows "Task Manager":

![Figure 2. CPU Cores](https://c.mql5.com/2/2/figure2.png)

Figure 2. CPU Cores

The results of all of the measurements are gathered in the table below:

| №<br> combination | The contents of the terminal | CPU usage |
| --- | --- | --- |
| 1 | 2 indicators on one charts | 1 core |
| 2 | 2 indicators on different charts, the same pair | 1 core |
| 3 | 2 indicators on different charts, different pairs | **2 cores** |
| 4 | 2 EAs on the same chart - this situation is impossible | - |
| 5 | 2 EAs on different charts, the same pair | **2 cores** |
| 6 | 2 EAs on different charts, different pairs | **2 cores** |
| 7 | 2 indicators on different pairs, created from the EA | **2 cores** |

The 7th combination is a common way to create an indicator, used in many trading strategies.

The only special feature is that I created two indicators on two different currency pairs, since combinations 1 and 2 make it clear that it makes no sense to place the indicators on the same pair. For this combination I used the EA **e-flood-starter**, which produced two copies of i-flood:

```
//+------------------------------------------------------------------+
//|                                              e-flood-starter.mq5 |
//+------------------------------------------------------------------+
void OnInit()
  {
   string s="EURUSD";
   for(int i=1; i<=2; i++)
     {
      Print("Indicator is created, handle=",
            iCustom(s,_Period,"i-flood",IntegerToString(i)));
      s="GBPUSD";
     }
  }
//+------------------------------------------------------------------+
```

Thus, all of the calculations of cores have been carried out, and now we know for which combinations MetaTrader uses multiple cores. Next, we will try to apply this knowledge to implement the ideas of parallel computations.

### We design a parallel system

With regard to the trading terminal for the parallel system, we mean a group of indicators or EAs (or a mixture of both) that together perform some common task, for example, conduct trade or draw on the chart. Meaning this group works as one big indicator or as one big EA. But at the same time distributes the computational load across all available processor cores.

Such a system consists of two types of software components:

- CM - computational module. Their number can be from 2 and up to the number of processor cores. It is in the CM that all of the code that needs to be parallelized is placed. As we found out from the previous chapter, the CM can be implemented as an indicator, as well as an EA - for any form of implementation, there is a combination, which uses all the processor cores;
- MM - the main module. Performs the main functions of the system. So if the MM is an indicator, then it performs drawing on the chart, and if the MM is an Ea, then it performs the trading functions. The MM also manages all of the CMs.

For example, for a MM EA and a 2-core processor, the scheme of the system's work will look like this:

![Figure 3. Scheme of the system with 2 CPU cores.](https://c.mql5.com/2/2/figure3.gif)

Figure 3. Scheme of the system with 2 CPU cores.

It should be understood that the developed by us system is not a traditional program, where you can just call the needed at the given moment procedure. The MM and CM are EAs or indicators, ie this is are independent and standalone programs. There is no direct connection between them, they operate independently, and can not communicate directly with each other.

The execution of any of these programs begins only with the appearance in the terminal of any [event](https://www.mql5.com/en/docs/runtime/event_fire) (for example, the arrival of quotations or a timer tick). And between the events, all of the data that these programs want to convey to each other, must be stored somewhere outside of the, in a publicly accessed place (let's call it "Data Exchange Buffer"). So the above scheme is implemented in the terminal in the following way:

![Figure 4. Implementation details](https://c.mql5.com/2/2/figure4.gif)

Figure 4. Implementation details

For the implementation of this system, we need to answer the following questions:

- which of the multiple cores combinations, found in the previous chapter, we will use in our system?
- since the system consists of several EAs or indicators, how can we better organize the exchange of data (two-sided) between them (ie, what will the clipboard data physically be like)?
- how can we organize the coordination and synchronization of their actions?

For each of these questions there is more than one answer, and they are all provided below. In practice, the specific options should be selected based on the particular situation. We will do this in the next chapter. In the meantime, let's consider all of the possible answers.

**_Combination_**

The **Combination 7** is the most convenient for regular practical use (all other combinations are listed in the previous chapter), because there is no need to open additional windows in the terminal and place on them EAs or indicators. The entire system is located in a single window, and all of the indicators (CM-1 and CM-2) are created by the EA (MM) automatically. The lack of extra windows and manual actions eliminated the confusion for the trader, and thus, the related to such confusion errors.

In some trading strategies, other combinations may be more helpful. For example, on the basis of any of them, we can create entire software systems, operating on the  "client-server" principle. Where the same CMs will be common for several MMs. Such common CMs can perform, not only a secondary role of a "computers", but be a "server" that stores some kind of unified for all strategies information, or even the coordinators of their collective work. A CM-server could, for example, centrally control the distribution of means in some portfolio of strategies and currency pairs, while maintaining the desired overall level of risk.

_**Data exchange**_

We can transmit the information between the MM and CM using any of the 3 ways:

1. [global variables of the terminal](https://www.mql5.com/en/docs/globals);
2. files;
3. Indicator [buffers](https://www.mql5.com/en/docs/customind/propertiesandfunctions).

The 1st method is optimal when there is a small number of numeric variables being transferred. If there is a need to transfer text data, it will have to somehow be coded into numbers, because global variables only have the type double.

The alternative is the 2nd method, because anything can be written into file(s). And this is a convenient (and possibly faster than the 1-st) method for the circumstance in which you need to transfer a large amounts of data.

The third method is suitable if the MM and CM are [indicators](https://www.mql5.com/en/docs/customind). Only the data of type double can be transferred, but it is more convenient to transfer large numeric arrays. But there is a drawback: during the formation of a new bar, the numbering of elements in the buffers is shifted. Because the MM and CM are on different currency pairs, the new bars will not appear simultaneously. We must take into account these shifts.

_**Synchronization**_

When the terminal receives a quote for the MM, and it begins to process it, it can not immediately transfer control to the CM. It can only (as shown in the diagram above) form a task (placing it in the global variables, a file, or an indicator buffer), and wait for the CM to be executed. Since all of the CMs are located on different currency pairs, the waiting may take a time. This is because one pair may receive the quote, while the other has not yet received it, and it will only come in a few seconds or even minutes (for example, this may occur during the night time on non-liquid pairs).

Hence, for the CM to obtain control, we should not use the [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick) and [OnCalculate](https://www.mql5.com/en/docs/basis/function/events#oncalculate) events, which depend on quotes. Instead of them, we need to use the [OnTimer](https://www.mql5.com/en/docs/basis/function/events#ontimer) event (innovation of MQL5), which is executed with a specified frequency (for example, 1 second). In this case the delays in the system will be severely limited.

Also, instead of OnTimer, we can use the cycling technique: meaning placing an infinite cycle for the CM in the [OnInit](https://www.mql5.com/en/docs/basis/function/events#oninit) or [OnCalculate](https://www.mql5.com/en/docs/basis/function/events#oncalculate). Each of its iterations is an analog of a timer tick.

**Warning.** I performed some experiments and found that when using Combination 7, the OnTimer event does not work in the indicators (for some reason), although the timers were successfully created.

You must also be careful with the infinite loops in the OnInit and OnCalculate: if even one CM-indicator is located on the same currency pair as the MM-EA, then the price stops moving on the chart, and the EA stops working (it ceases to generate the OnTick events). The developers of the terminal explained the reasons of this behaviour.

From the developers: scripts and EAs work in their own separate threads, while all of the indicators, on a single symbol, work in the same thread. In the same stream as the indicators, all other actions on this symbol are also consecutively executed: the processing of ticks, the synchronization of the history, and the calculation of indicators. So if the indicator performs an infinite action, all other events for its symbol will never be executed.

| Program | Execution | Note |
| Script | In its own thread, there are as many threads of execution as there are scripts | A cycling script can not disrupt the work of other programs |
| Expert Advisor | In its own thread, there are as many threads of execution as there are EAs | A cycling script can not disrupt the work of other programs |
| Indicator | One thread of execution for all indicators on one symbol. As many symbols with indicators as there are threads of execution for them | An infinite cycle in one indicator will stop the work of all other indicators on that symbol |

### Creating a test Expert Advisor

Let's select some trading strategy, which would make sense to parallelize, and an algorithm that would be suitable for this.

For example, this can be a simple strategy: compiling the sequence from N of last bars, and finding the most similar sequence to this in the history. Knowing to where the price has shifted in the history, we open the relevant deal.

If the length of the sequence is relatively small, this strategy will work very quickly in MetaTrader 5 - within seconds. But if we take a large length - for example, all of the bars of the time-frame M1 for the last 24 hours (which will be 1440 bars), - and if we search back in history as far as one year ago (about 375,000 bars), this will require a significant amount of time. And yet, this search can be easily parallelized: it is enough to divide the history into equal parts over the number of available processor cores, and assign each core to search a specific location.

The parameters of the parallel system will be as following:

- MM - is the EA that implements the patterned trading strategy;
- parallel computation is done in the CM-indicators, automatically generated from the Ea (ie, using Combination 7);
- the computing code in the CM-indicators is placed inside an infinite cycle in the OnInit;
- data exchange between the MM-EA and CM-indicators - done through the [global variables of the terminal](https://www.mql5.com/en/docs/globals).

For the convenience of development and subsequent use, we will create the EA in such a way so that, depending on the settings, it could operate as a parallel (with calculations in the indicators), and as a usual (ie without the use of indicators) EA. The code of the obtained **e-MultiThread** Expert Advisor:

```
//+------------------------------------------------------------------+
//|                                                e-MultiThread.mq5 |
//+------------------------------------------------------------------+
input int Threads=1; // How many cores should be used
input int MagicNumber=0;

// Strategy parameters
input int PatternLen  = 1440;   // The length of the sequence to analyze (pattern)
input int PrognozeLen = 60;     // Forecast length (bars)
input int HistoryLen  = 375000; // History length to search

input double Lots=0.1;
//+------------------------------------------------------------------+
class IndData
  {
public:
   int               ts,te;
   datetime          start_time;
   double            prognoze,rating;
  };

IndData Calc[];
double CurPattern[];
double Prognoze;
int  HistPatternBarStart;
int  ExistsPrognozeLen;
uint TicksStart,TicksEnd;
//+------------------------------------------------------------------+
#include <ThreadCalc.mqh>
#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
int OnInit()
  {

   double rates[];

//--- Make sure there is enough history
   int HistNeed=HistoryLen+Threads+PatternLen+PatternLen+PrognozeLen-1;
   if(TerminalInfoInteger(TERMINAL_MAXBARS)<HistNeed)
     {
      Print("Change the terminal setting \"Max. bars in chart\" to the value, not lesser than ",
            HistNeed," and restart the terminal");
      return(1);
     }
   while(Bars(_Symbol,_Period)<HistNeed)
     {
      Print("Insufficient history length (",Bars(_Symbol,_Period),") in the terminal, upload...");
      CopyClose(_Symbol,_Period,0,HistNeed,rates);
     }
   Print("History length in the terminal: ",Bars(_Symbol,_Period));

//--- For a multi-core mode create computational indicators
   if(Threads>1)
     {
      GlobalVarPrefix="MultiThread_"+IntegerToString(MagicNumber)+"_";
      GlobalVariablesDeleteAll(GlobalVarPrefix);

      ArrayResize(Calc,Threads);

      // Length of history for each core
      int HistPartLen=MathCeil(HistoryLen/Threads);
      // Including the boundary sequences
      int HistPartLenPlus=HistPartLen+PatternLen+PrognozeLen-1;

      string s;
      int snum=0;
      // Create all computational indicators
      for(int t=0; t<Threads; t++)
        {
         // For each indicator - its own currency pair,
         // it should not be the same as for the EA
         do
            s=SymbolName(snum++,false);
         while(s==_Symbol);

         int handle=iCustom(s,_Period,"i-Thread",
                            GlobalVarPrefix,t,_Symbol,PatternLen,
                            PatternLen+t*HistPartLen,HistPartLenPlus);

         if(handle==INVALID_HANDLE) return(1);
         Print("Indicator created, pair ",s,", handle ",handle);
        }
     }

   return(0);
  }
//+------------------------------------------------------------------+
void OnTick()
  {
   TicksStart=GetTickCount();

   // Fill in the sequence with the last bars
   while(CopyClose(_Symbol,_Period,0,PatternLen,CurPattern)<PatternLen) Sleep(1000);

   // If there is an open position, measure its "age"
   // and modify the forecast range for the remaining
   // planned life time of the deal
   CalcPrognozeLen();

   // Find the most similar sequence in the history
   // and the forecast of the movement of its price on its basis
   FindHistoryPrognoze();

   // Perform the necessary trade actions
   Trade();

   TicksEnd=GetTickCount();
   // Debugging information in
   PrintReport();
  }
//+------------------------------------------------------------------+
void FindHistoryPrognoze()
  {
   Prognoze=0;
   double MaxRating;

   if(Threads>1)
     {
      //--------------------------------------
      // USE COMPUTATIONAL INDICATORS
      //--------------------------------------
      // Look through all of the computational indicators
      for(int t=0; t<Threads; t++)
        {
         // Send the parameters of the computational task
         SetParam(t,"PrognozeLen",ExistsPrognozeLen);
         // "Begin computations" signal
         SetParam(t,"Query");
        }

      for(int t=0; t<Threads; t++)
        {
         // Wait for results
         while(!ParamExists(t,"Answer"))
            Sleep(100);
         DelParam(t,"Answer");

         // Obtain results
         double progn        = GetParam(t, "Prognoze");
         double rating       = GetParam(t, "Rating");
         datetime time[];
         int start=GetParam(t,"PatternStart");
         CopyTime(_Symbol,_Period,start,1,time);
         Calc [t].prognoze   = progn;
         Calc [t].rating     = rating;
         Calc [t].start_time = time[0];
         Calc [t].ts         = GetParam(t, "TS");
         Calc [t].te         = GetParam(t, "TE");

         // Select the best result
         if((t==0) || (rating>MaxRating))
           {
            MaxRating = rating;
            Prognoze  = progn;
           }
        }
     }
   else
     {
      //----------------------------
      // INDICATORS ARE NOT USED
      //----------------------------
      // Calculate everything in the EA, into one stream
      FindPrognoze(_Symbol,CurPattern,0,HistoryLen,ExistsPrognozeLen,
                   Prognoze,MaxRating,HistPatternBarStart);
     }
  }
//+------------------------------------------------------------------+
void CalcPrognozeLen()
  {
   ExistsPrognozeLen=PrognozeLen;

   // If there is an opened position, determine
   // how many bars have passed since its opening
   if(PositionSelect(_Symbol))
     {
      datetime postime=PositionGetInteger(POSITION_TIME);
      datetime curtime,time[];
      CopyTime(_Symbol,_Period,0,1,time);
      curtime=time[0];
      CopyTime(_Symbol,_Period,curtime,postime,time);
      int poslen=ArraySize(time);
      if(poslen<PrognozeLen)
         ExistsPrognozeLen=PrognozeLen-poslen;
      else
         ExistsPrognozeLen=0;
     }
  }
//+------------------------------------------------------------------+
void Trade()
  {

   // Close the open position, if it is against the forecast
   if(PositionSelect(_Symbol))
     {
      long type=PositionGetInteger(POSITION_TYPE);
      bool close=false;
      if((type == POSITION_TYPE_BUY)  && (Prognoze <= 0)) close = true;
      if((type == POSITION_TYPE_SELL) && (Prognoze >= 0)) close = true;
      if(close)
        {
         CTrade trade;
         trade.PositionClose(_Symbol);
        }
     }

   // If there are no position, open one according to the forecast
   if((Prognoze!=0) && (!PositionSelect(_Symbol)))
     {
      CTrade trade;
      if(Prognoze > 0) trade.Buy (Lots);
      if(Prognoze < 0) trade.Sell(Lots);
     }
  }
//+------------------------------------------------------------------+
void PrintReport()
  {
   Print("------------");
   Print("EA: started ",TicksStart,
         ", finished ",TicksEnd,
         ", duration (ms) ",TicksEnd-TicksStart);
   Print("EA: Forecast on ",ExistsPrognozeLen," bars");

   if(Threads>1)
     {
      for(int t=0; t<Threads; t++)
        {
         Print("Indicator ",t+1,
               ": Forecast ", Calc[t].prognoze,
               ", Rating ", Calc[t].rating,
               ", sequence from ",TimeToString(Calc[t].start_time)," in the past");
         Print("Indicator ",t+1,
               ": started ",  Calc[t].ts,
               ", finished ",   Calc[t].te,
               ", duration (ms) ",Calc[t].te-Calc[t].ts);
        }
     }
   else
     {
      Print("Indicators were not used");
      datetime time[];
      CopyTime(_Symbol,_Period,HistPatternBarStart,1,time);
      Print("EA: sequence from ",TimeToString(time[0])," in the past");
     }

   Print("EA: Forecast ",Prognoze);
   Print("------------");
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Send the "finish" command to the indicators
   if(Threads>1)
      for(int t=0; t<Threads; t++)
         SetParam(t,"End");
  }
//+------------------------------------------------------------------+
```

The code of the computational indicator **i-Thread**, used by the Expert Advisor:

```
//+------------------------------------------------------------------+
//|                                                     i-Thread.mq5 |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_chart_window
#property indicator_buffers 1
#property indicator_plots   1

//--- input parameters
input string VarPrefix;  // Prefix for global variables (analog to MagicNumber)
input int    ThreadNum;  // Core number (so that indicators on different cores could
                        // differentiate their tasks from the tasks of the "neighboring" cores)
input string DataSymbol; // On what pair is the MM-EA working
input int    PatternLen; // Length of the sequence for analysis
input int    BarStart;   // From which bar in the history the search for a similar sequence began
input int    BarCount;   // How many bars of the history to perform a search on

//--- indicator buffers
double Buffer[];
//---
double CurPattern[];

//+------------------------------------------------------------------+
#include <ThreadCalc.mqh>
//+------------------------------------------------------------------+
void OnInit()
  {
   SetIndexBuffer(0,Buffer,INDICATOR_DATA);

   GlobalVarPrefix=VarPrefix;

   // Infinite loop - so that the indicator always "listening",
   // for new commands from the EA
   while(true)
     {
      // Finish the work of the indicator, if there is a command to finish
      if(ParamExists(ThreadNum,"End"))
         break;

      // Wait for the signal to begin calculations
      if(!ParamExists(ThreadNum,"Query"))
        {
         Sleep(100);
         continue;
        }
      DelParam(ThreadNum,"Query");

      uint TicksStart=GetTickCount();

      // Obtain the parameters of the task
      int PrognozeLen=GetParam(ThreadNum,"PrognozeLen");

      // Fill the sequence from the last bars
      while(CopyClose(DataSymbol,_Period,0,PatternLen,CurPattern)
            <PatternLen) Sleep(1000);

      // Perform calculations
      int HistPatternBarStart;
      double Prognoze,Rating;
      FindPrognoze(DataSymbol,CurPattern,BarStart,BarCount,PrognozeLen,
                   Prognoze,Rating,HistPatternBarStart);

      // Send the results of calculations
      SetParam(ThreadNum,"Prognoze",Prognoze);
      SetParam(ThreadNum,"Rating",Rating);
      SetParam(ThreadNum,"PatternStart",HistPatternBarStart);
      SetParam(ThreadNum,"TS",TicksStart);
      SetParam(ThreadNum,"TE",GetTickCount());
      // Signal "everything is ready"
      SetParam(ThreadNum,"Answer");
     }
  }
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
  {
   // The handler of this event is required
   return(0);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   SetParam(ThreadNum,"End");
  }
//+------------------------------------------------------------------+
```

The Expert Advisor and the indicator uses a common **ThreadCalc.mqh** library.

Here is its code:

```
//+------------------------------------------------------------------+
//|                                                   ThreadCalc.mqh |
//+------------------------------------------------------------------+
string GlobalVarPrefix;
//+------------------------------------------------------------------+
// It finds the price sequence, most similar to the assigned one.
// in the specified range of the history
// Returns the estimation of similarity and the direction
// of the further changes of prices in history.
//+------------------------------------------------------------------+
void FindPrognoze(
                  string DataSymbol,    // symbol
                  double  &CurPattern[],// current pattern
                  int BarStart,         // start bar
                  int BarCount,         // bars to search
                  int PrognozeLen,      // forecast length

                  // RESULT
                  double  &Prognoze,        // forecast (-,0,+)
                  double  &Rating,          // rating
                  int  &HistPatternBarStart // starting bar of the found sequence
                  )
  {

   int PatternLen=ArraySize(CurPattern);

   Prognoze=0;
   if(PrognozeLen<=0) return;

   double rates[];
   while(CopyClose(DataSymbol,_Period,BarStart,BarCount,rates)
         <BarCount) Sleep(1000);

   double rmin=-1;
   // Shifting by one bar, go through all of the price sequences in the history
   for(int bar=BarCount-PatternLen-PrognozeLen; bar>=0; bar--)
     {
      // Update to eliminate the differences in the levels of price in the sequences
      double dr=CurPattern[0]-rates[bar];

      // Calculate the level of differences between the sequences - as a sum
      // of squares of price deviations from the sample values
      double r=0;
      for(int i=0; i<PatternLen; i++)
         r+=MathPow(MathAbs(rates[bar+i]+dr-CurPattern[i]),2);

      // Find the sequence with the least difference level
      if((r<rmin) || (rmin<0))
        {
         rmin=r;
         HistPatternBarStart   = bar;
         int HistPatternBarEnd = bar + PatternLen-1;
         Prognoze=rates[HistPatternBarEnd+PrognozeLen]-rates[HistPatternBarEnd];
        }
     }
   // Convert the bar number into an indicator system of coordinates
   HistPatternBarStart=BarStart+BarCount-HistPatternBarStart-PatternLen;

   // Convert the difference into the rating of similarity
   Rating=-rmin;
  }
//====================================================================
// A set of functions for easing the work with global variables.
// As a parameter contain the number of computational threads
// and the names of the variables, automatically converted into unique
// global names.
//====================================================================
//+------------------------------------------------------------------+
string GlobalParamName(int ThreadNum,string ParamName)
  {
   return GlobalVarPrefix+IntegerToString(ThreadNum)+"_"+ParamName;
  }
//+------------------------------------------------------------------+
bool ParamExists(int ThreadNum,string ParamName)
  {
   return GlobalVariableCheck(GlobalParamName(ThreadNum,ParamName));
  }
//+------------------------------------------------------------------+
void SetParam(int ThreadNum,string ParamName,double ParamValue=0)
  {
   string VarName=GlobalParamName(ThreadNum,ParamName);
   GlobalVariableTemp(VarName);
   GlobalVariableSet(VarName,ParamValue);
  }
//+------------------------------------------------------------------+
double GetParam(int ThreadNum,string ParamName)
  {
   return GlobalVariableGet(GlobalParamName(ThreadNum,ParamName));
  }
//+------------------------------------------------------------------+
double DelParam(int ThreadNum,string ParamName)
  {
   return GlobalVariableDel(GlobalParamName(ThreadNum,ParamName));
  }
//+------------------------------------------------------------------+
```

Our trading system, which is able to use more than one processor core in its work, is ready!

When using it, you should remember that in this example we used CM-indicators with infinite loops.

If you are planning to run other programs, along with this system, in the terminal, then you should make sure that you are using them on the currency pairs that are not used by the CM-indicators. A good way to avoid such a conflict is to modify the system so that in the input parameters of the MM-EA you could directly specify the currency pairs for the CM-indicators.

### Measuring the speed of the EA's work

**_Regular mode_**

Open the EURUSD M1 chart and launch our Expert Advisor, created in the previous chapter. In the settings, specify the length of the patterns as 24 hours (1440 min bars), and the depth of the search in history - as 1 year (375 000 bars).

![Figure 4. Expert Advisor input parameters](https://c.mql5.com/2/2/figure5.png)

Figure 4. Expert Advisor input parameters

Parameter "Threads" set equal to 1. This means that all of the calculations of the EA will be produced into one thread (on a single core). Meanwhile, it will not use computational indicators, but will calculate everything itself. Basically, by the principle of work of a regular EA.

Log of its execution:

![Figure 6. Expert Advisor Log](https://c.mql5.com/2/2/figure6.png)

Figure 6. Expert Advisor Log (1 thread)

**_Parallel mode_**

Now let's delete this EA and the position, opened by it. Add the EA again, but this time, with the parameter "Threads" equal to 2.

Now, the EA will need to create and use in its work 2 computational indicators, occupying two processor cores. Log of its execution:

![Figure 7. Expert Advisor Log (2 threads)](https://c.mql5.com/2/2/figure7.png)

Figure 7. Expert Advisor Log (2 threads)

**_Speed Comparison_**

Analyzing both of these logs, we conclude that the approximate time of execution of the EA is:

- in regular mode - 52 seconds;
- in a 2-core mode - 27 seconds.

So by performing parallelization on a 2-core CPU, we were able to increase the speed of an EA **by 1.9 times**. It can be assumed that when using a processor with a _large amount_ of cores, the speed of execution will increase even more, in proportion to the number of cores.

**_Control of the correctness of the work_**

In addition to the execution time, the logs provide additional information, which allow us to verify that all of the measurements were carried out correctly. The lines _EA: Beginning work ... ending work ... "_ and _"Indicator ...: Beginning work ... ending work ..."_ show that the indicators have started their calculations not a second before the EA gave them that command.

Let's also verify that there are no violations of the trading strategy during the launch of the EA in the parallel mode. According to the logs, it is clear that the launch of the EA in the parallel mode was made almost immediately after its launch in the regular mode. Meaning the market situations, in both cases, was similar. The logs show that the dates, found in the history of patterns, in both cases were also very similar. So everything is good: the algorithm of the strategy works in both cases equally well.

Here are the patterns, described in the logs situations. Such was the current market situation (length - 1440-minute bars) at the time of running the EA in the regular mode:

![Figure 8. Current market situation](https://c.mql5.com/2/2/figure8.gif)

Figure 8. Current market situation

The EA found in the history the following similar pattern:

![Figure 9. Similar market situation](https://c.mql5.com/2/2/figure9.gif)

Figure 9. Similar market situation

When running the EA in the parallel mode, the same pattern was found by "Indicator 1". "Indicator 2", as follows from the log, was looking for patterns in the other half year of the history, thereforit found a different similar pattern:

![Figure 10. Similar market situation](https://c.mql5.com/2/2/figure10.gif)

Figure 10. Similar market situation

And this is what [global variables](https://www.mql5.com/en/docs/globals) in MetaTrader 5 look like during the EA's work in the parallel mode:

![Figure 11. Global variables](https://c.mql5.com/2/2/figure11.png)

Figure 11. Global variables

The data exchange between the EA and the indicators through [global variables](https://www.mql5.com/en/docs/globals) was implemented successfully.

### Conclusion

In this study, we found that it is possible to parallelize resourceful algorithms by the standard means of MetaTrader 5. And the discovered solution to this problem is suitable for convenient use in real-world trading strategies.

This program, in a multi-core system, really works in a proportionally faster way. The number of cores in a processors grows with every year, and it is good that traders, who use MetaTrader, have the opportunity to effectively use these hardware resources. We can safely create more resourceful trading strategies, which will still be able to analyze the market in real time.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/197](https://www.mql5.com/ru/articles/197)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/197.zip "Download all attachments in the single ZIP archive")

[i-flood.mq5](https://www.mql5.com/en/articles/download/197/i-flood.mq5 "Download i-flood.mq5")(0.7 KB)

[e-flood.mq5](https://www.mql5.com/en/articles/download/197/e-flood.mq5 "Download e-flood.mq5")(0.58 KB)

[e-flood-starter.mq5](https://www.mql5.com/en/articles/download/197/e-flood-starter.mq5 "Download e-flood-starter.mq5")(0.43 KB)

[i-thread.mq5](https://www.mql5.com/en/articles/download/197/i-thread.mq5 "Download i-thread.mq5")(3.1 KB)

[e-multithread.mq5](https://www.mql5.com/en/articles/download/197/e-multithread.mq5 "Download e-multithread.mq5")(8.1 KB)

[threadcalc.mqh](https://www.mql5.com/en/articles/download/197/threadcalc.mqh "Download threadcalc.mqh")(3.93 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Trading strategy based on the improved Doji candlestick pattern recognition indicator](https://www.mql5.com/en/articles/12355)
- [Improved candlestick pattern recognition illustrated by the example of Doji](https://www.mql5.com/en/articles/9801)
- [Neural Networks Cheap and Cheerful - Link NeuroPro with MetaTrader 5](https://www.mql5.com/en/articles/830)
- [3 Methods of Indicators Acceleration by the Example of the Linear Regression](https://www.mql5.com/en/articles/270)
- [Decreasing Memory Consumption by Auxiliary Indicators](https://www.mql5.com/en/articles/259)
- [Connecting NeuroSolutions Neuronets](https://www.mql5.com/en/articles/236)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3303)**
(22)


![autotrader5](https://c.mql5.com/avatar/avatar_na2.png)

**[autotrader5](https://www.mql5.com/en/users/autotrader5)**
\|
27 Dec 2017 at 14:08

Does MT4 support that?


![Ali Erfanbeigi](https://c.mql5.com/avatar/avatar_na2.png)

**[Ali Erfanbeigi](https://www.mql5.com/en/users/jalalahmadi22)**
\|
23 Oct 2019 at 17:50

hi every one.

i am trying to develop a [backtest](https://www.mql5.com/en/articles/2612 "Article \"Testing trading strategies on real ticks\"") platform using python .(an integration between python and meta5) .

in one of its steps , i need to know more about that how cores(workers) in optimization phase complete a process and what is the exact flow of the contribution between cores.

is there any useful doc about this? c

thanks! c

![Konstantin Efremov](https://c.mql5.com/avatar/2020/5/5EAC9E40-7E56.jpg)

**[Konstantin Efremov](https://www.mql5.com/en/users/leonardo4)**
\|
15 Feb 2020 at 20:25

Thanks for the cool article, and especially for the mini lesson on linking EA and indicator via global variables.

My EA-indicator link requires transferring a decent amount of data to the EA, I used to do it through indicator buffers, but your method is more universal and simple, and also less load the processor with polling of [indicator buffers](https://www.mql5.com/en/articles/180 "Article: Averaging price series without additional buffers for intermediate calculations").

I just did it, everything works perfectly.

Also, the Expert Advisor is multi-currency and your article helped me to understand how to distribute the computational load.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
16 Feb 2020 at 12:36

There is an alternative of [parallel computing on chart objects with expert calculators and sharing data across resources](https://www.mql5.com/en/code/27644).


![Zeke Yaeger](https://c.mql5.com/avatar/2022/6/629E37C1-8BFC.jpg)

**[Zeke Yaeger](https://www.mql5.com/en/users/ozymandias_vr12)**
\|
31 Aug 2020 at 01:16

Wonderful!

Thank you a lot, it works well.


![Implementation of Indicators as Classes by Examples of Zigzag and ATR](https://c.mql5.com/2/0/indicator_boxed.png)[Implementation of Indicators as Classes by Examples of Zigzag and ATR](https://www.mql5.com/en/articles/247)

Debate about an optimal way of calculating indicators is endless. Where should we calculate the indicator values - in the indicator itself or embed the entire logic in a Expert Advisor that uses it? The article describes one of the variants of moving the source code of a custom indicator iCustom right in the code of an Expert Advisor or script with optimization of calculations and modeling the prev\_calculated value.

![Building a Spectrum Analyzer](https://c.mql5.com/2/0/spectrum_MQL5__1.png)[Building a Spectrum Analyzer](https://www.mql5.com/en/articles/185)

This article is intended to get its readers acquainted with a possible variant of using graphical objects of the MQL5 language. It analyses an indicator, which implements a panel of managing a simple spectrum analyzer using the graphical objects. The article is meant for readers acquianted with basics of MQL5.

![Filtering Signals Based on Statistical Data of Price Correlation](https://c.mql5.com/2/0/fa_title01.png)[Filtering Signals Based on Statistical Data of Price Correlation](https://www.mql5.com/en/articles/269)

Is there any correlation between the past price behavior and its future trends? Why does the price repeat today the character of its previous day movement? Can the statistics be used to forecast the price dynamics? There is an answer, and it is positive. If you have any doubt, then this article is for you. I'll tell how to create a working filter for a trading system in MQL5, revealing an interesting pattern in price changes.

![Finding Errors and Logging](https://c.mql5.com/2/0/filter_Log_MQL5.png)[Finding Errors and Logging](https://www.mql5.com/en/articles/150)

MetaEditor 5 has the debugging feature. But when you write your MQL5 programs, you often want to display not the individual values, but all messages that appear during testing and online work. When the log file contents have large size, it is obvious to automate quick and easy retrieval of required message. In this article we will consider ways of finding errors in MQL5 programs and methods of logging. Also we will simplify logging into files and will get to know a simple program LogMon for comfortable viewing of logs.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/197&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071876422919729184)

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