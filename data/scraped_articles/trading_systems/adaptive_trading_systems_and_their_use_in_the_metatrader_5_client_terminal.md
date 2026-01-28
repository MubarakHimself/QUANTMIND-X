---
title: Adaptive Trading Systems and Their Use in the MetaTrader 5 Client Terminal
url: https://www.mql5.com/en/articles/143
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:52:23.106261
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/143&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062781327159240900)

MetaTrader 5 / Trading systems


### Introduction

Hundreds of thousands of traders all over the world use the trading platforms developed by [MetaQuotes Software Corp](https://www.metaquotes.net/). The key factor leading to success is the technological superiority based on the experience of many years and the best software solutions.

Many people have already estimated new opportunities that have become available with the new [MQL5 language](https://www.mql5.com/en/docs). Its key features are high performance and the possibility of using object-oriented programming. In addition to it, with appearing of the multi-currency strategy tester in the MetaTrader 5 client terminal, many traders have acquired unique tools for developing, learning and using complex trading systems.

[Automated Trading Championship 2010](https://championship.mql5.com/2010/en) starts this autumn; thousands of trading robots written in MQL5 are going to participate in it. An Expert Advisor that earns the maximum profit during the competition will win. But what strategy will appear the most effective one?

The strategy tester of the MetaTrader 5 terminal allows finding the best set of parameters, using which the system earns the maximum amount of profit during a specified time period. But can it be done in the real time? The idea of the virtual trading using several strategies in an Expert Advisor was considered in the ["Contest of Expert Advisors inside an Expert Advisor"](https://www.mql5.com/en/articles/1578) article, which contains its implementation in MQL4.

In this article, we are going to show that the creation and analysis of adaptive strategies has become significantly easier in MQL5 due to the usage of [object-oriented programming](https://www.mql5.com/en/docs/basis/oop), [classes for working with data](https://www.mql5.com/en/docs/standardlibrary/datastructures) and [trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) of the Standard Library.

### 1\. Adaptive Trading Strategies

Markets change constantly. Trade strategies need their adaptation to the current market conditions.

The values of parameters that give the maximal profitability of the strategy can be found without using the optimization through sequential change of parameters and analysis of testing results.

Figure 1 demonstrates the equity curves for ten Expert Advisors (MA\_3,...MA\_93); each of them traded by the strategy of moving averages but with different periods (3,13,..93). The testing has been performed at EURUSD H1, the testing period is 4.01.2010-20.08.2010.

![Figure 1. Diagrams of equity curves of ten Expert Advisors at the account](https://c.mql5.com/2/2/Image1.png)

Figure 1. Diagrams of equity curves of ten Expert Advisors at the account

As you can see at the Figure 1,  the Expert Advisors had nearly the same results during the first two weeks of working, but further their profits started diverging significantly. At the end of testing period the best trading results were shown by the Expert Advisors with periods 63, 53 and 43.

The market has chosen the best ones. Why shouldn't we follow its choice? What if we combine all ten strategies in a single Expert Advisor, provide the possibility of "virtual" trading for each strategy, and periodically (for example, at the beginning of each new bar) determine the best strategy for the real trading and trade in accordance with its signals?

The results of obtained adaptive strategy are shown in the Figure 2. The equity curve of the account with adaptive trading is shown with the red color. Note, that during more than a half of period the form of equity curve for the adaptive strategy is the same as the one of the MA\_63 strategy, which has appeared to be the winner finally.

![Figure 2. Equity curves at the account with the adaptive strategy that uses signals of 10 trade systems](https://c.mql5.com/2/2/Image2.png)

Figure 2. Equity curves at the account with the adaptive strategy that uses signals of 10 trade systems

The balance curves have the similar dynamics (Fig. 3):

![Figure 3. Balance curves of the adaptive strategy that uses signals of 10 trade systems](https://c.mql5.com/2/2/Image3.png)

Figure 3. Balance curves of the adaptive strategy that uses signals of 10 trade systems

If none of the strategies is profitable at the moment, the adaptive systems shouldn't perform trade operations. The example of such case is shown in the fig. 4 (period from 4-th to 22-nd of January 2010).

![Figure 4. The time period when the adaptive strategy stopped opening new positions due to the absence of profitable strategies](https://c.mql5.com/2/2/Image4.png)

Figure 4. The time period when the adaptive strategy stopped opening new positions due to the absence of profitable strategies

Starting from the January 2010 the best effectiveness is shown by the MA\_3 strategy. Since the MA\_3 (blue) had the maximum amount of money earned at that moment, the adaptive strategy (red) followed its signals. In the period from 8-th to 20th of January all the considered strategies had a negative result, that's why the adaptive strategy didn't open new trade positions.

If all the strategies have a negative result, it's better to stay away from trading. This is the significant thing that allows stopping unprofitable trading and keeping your money save.

### 2\. Implementation of the Adaptive Trading Strategy

In this section, we are going to consider the structure of the adaptive strategy that performs the "virtual" trading using several trade strategies simultaneously, and chooses the most profitable one for real trading according to its signals. Note that the use of the [object-oriented approach](https://www.mql5.com/en/docs/basis/oop) makes the solution of this problem significantly easier.

First of all we are going to investigate the code of the adaptive Expert Advisor, then we're going to take a detailed look into the **CAdaptiveStrategy** where the functionality of the adaptive system is implemented, and then we will show the structure of the **CSampleStrategy** class - the base class of the trade strategies where the functionality of virtual trading is implemented.

Further, we're going to consider the code of two of its children - the **CStrategyMA** and **CStrategyStoch** classes that represent the strategies of trading by moving averages and the stochastic oscillator. After analyzing their structure you'll be able to easily write and add you own classes that realize your strategies.

**2.1. Code of the Expert Advisor**

The code of the Expert Advisor looks very simple:

```
//+------------------------------------------------------------------+
//|                                       Adaptive_Expert_Sample.mq5 |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <CAdaptiveStrategy.mqh>

CAdaptiveStrategy Adaptive_Expert;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(Adaptive_Expert.Expert_OnInit());
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Adaptive_Expert.Expert_OnDeInit(reason);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   Adaptive_Expert.Expert_OnTick();
  }
//+------------------------------------------------------------------+
```

The first three lines define the [properties of the program](https://www.mql5.com/en/docs/basis/preprosessor/compilation), then comes the [#include directive](https://www.mql5.com/en/docs/basis/preprosessor/include) that tells the preprocessor to include the CAdaptiveStrategy.mqh file. Angle brackets specify that the file should be taken from the standard directory (usually, it is terminal\_folder\\MQL5\\Include).

The next line contains the declaration of the Adaptive\_Expert object (instance of the CAdaptiveStrategy class); and the code of the [OnInit](https://www.mql5.com/en/docs/basis/function/events#oninit), [OnDeinit](https://www.mql5.com/en/docs/basis/function/events#ondeinit) and [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick) functions of the Expert Advisor consists of the calls of corresponding functions Expert\_OnInit,  Expert\_OnDeInit and Expert\_OnTick and the Adaptive\_Expert object.

**2.2. The CAdaptiveStrategy class**

The class of thr adaptive Expert Advisor (CAdaptiveStrategy class) is located in the CAdaptiveStrategy.mqh file. Let's start with the include files:

```
#include <Arrays\ArrayObj.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>
#include <CStrategyMA.mqh>
#include <CStrategyStoch.mqh>
```

The reason why we include the ArrayObj.mqh file is the convenience of working with classes of different strategies using the object of the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class, which represents a dynamic array of pointers to the class instances spawned by the base class [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) and its children. This object will the **m\_all\_strategies** array, it will be used a "container" of trade strategies.

Each strategy is represented as a class. In this case, we have included the files that contain the **CStrategyMA** and **CStrategyStoch** classes, which represent the strategies of trading by moving averages and trading by the stochastic oscillator.

For requesting properties of current positions and for performing trade operations, we will use the [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) and [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) classes of the Standard library, that's why we include the PositionInfo.mqh and Trade.mqh files.

Let's take a look into the structure of the **CAdaptiveStrategy** class.

```
//+------------------------------------------------------------------+
//| Class CAdaptiveStrategy                                          |
//+------------------------------------------------------------------+
class CAdaptiveStrategy
  {
protected:
   CArrayObj        *m_all_strategies;   // objects of trade strategies

   void              ProceedSignalReal(int state,double trade_volume);
   int               RealPositionDirection();

public:
   // initialization of the adaptive strategy
   int               Expert_OnInit();
   // deinitialization of the adaptive strategy
   int               Expert_OnDeInit(const int reason);
   // check of trade conditions and opening of virtual positions
   void              Expert_OnTick();
  };
```

To implement a united approach to the objects of different classes, the trade strategies (or rather the instances of their classes) are stored in the dynamic array **m\_all\_strategies** (of the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) type), which is used as a "container" of classes of the strategies. This is the reason why the class of trade strategies SampleStrategy is spawned from the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) class.

The **ProceedSignalReal** function implements the "synchronization" of the direction and volume of a real position with the given direction and volume:

```
//+------------------------------------------------------------------+
//| This method is intended for "synchronization" of current         |
//| real trade position with the value of the 'state' state          |
//+------------------------------------------------------------------+
void CAdaptiveStrategy::ProceedSignalReal(int state,double trade_volume)
  {
   CPositionInfo posinfo;
   CTrade trade;

   bool buy_opened=false;
   bool sell_opened=false;

   if(posinfo.Select(_Symbol)) // if there are open positions
     {
      if(posinfo.Type()==POSITION_TYPE_BUY) buy_opened=true;    // a buy position is opened
      if(posinfo.Type()==POSITION_TYPE_SELL) sell_opened=true;  // a sell position is opened

      // if state = 0, then we need to close open positions
      if((state==POSITION_NEUTRAL) && (buy_opened || sell_opened))
        {
         if(!trade.PositionClose(_Symbol,200))
            Print(trade.ResultRetcodeDescription());
        }
      //reverse: closing buy position and opening sell position
      if((state==POSITION_SHORT) && (buy_opened))
        {
         if(!trade.PositionClose(_Symbol,200))
            Print(trade.ResultRetcodeDescription());
         if(!trade.PositionOpen(_Symbol,ORDER_TYPE_SELL,trade_volume,SymbolInfoDouble(_Symbol,SYMBOL_BID),0,0))
            Print(trade.ResultRetcodeDescription());
        }
      //reverse: close sell position and open buy position
      if(((state==POSITION_LONG) && (sell_opened)))
        {
         if(!trade.PositionClose(_Symbol,200))
            Print(trade.ResultRetcodeDescription());
         if(!trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,trade_volume,SymbolInfoDouble(_Symbol,SYMBOL_ASK),0,0))
            Print(trade.ResultRetcodeDescription());
        }
     }
   else // if there are no open positions
     {
      // open a buy position
      if(state==POSITION_LONG)
        {
         if(!trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,0.1,SymbolInfoDouble(_Symbol,SYMBOL_ASK),0,0))
            Print(trade.ResultRetcodeDescription());
        }
      // open a sell position
      if(state==POSITION_SHORT)
        {
         if(!trade.PositionOpen(_Symbol,ORDER_TYPE_SELL,0.1,SymbolInfoDouble(_Symbol,SYMBOL_BID),0,0))
            Print(trade.ResultRetcodeDescription());
        }
     }
  }
```

Note that it's easier to work with the trade position using the [trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses). We used the objects of the [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) and [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) classes for requesting the properties of market position and for performing trade operations respectively.

The **RealPositionDirection** function requests the parameters of the real open position and returns its direction:

```
//+------------------------------------------------------------------+
//| Returns direction (0,+1,-1) of the current real position         |
//+------------------------------------------------------------------+
int CAdaptiveStrategy::RealPositionDirection()
  {
   int direction=POSITION_NEUTRAL;
   CPositionInfo posinfo;

   if(posinfo.Select(_Symbol)) // if there are open positions
     {
      if(posinfo.Type()==POSITION_TYPE_BUY) direction=POSITION_LONG;    // a buy position is opened
      if(posinfo.Type()==POSITION_TYPE_SELL) direction=POSITION_SHORT;  // a short position is opened
     }
   return(direction);
  }
```

Now we're going to take a look into the main functions of the СAdaptiveStrategy class.

Let's start with the **Expert\_OnInit:** function

```
//+------------------------------------------------------------------+
//| Function of initialization of the Adaptive Expert Advisor        |
//+------------------------------------------------------------------+
int CAdaptiveStrategy::Expert_OnInit()
  {
//--- Create array of objects m_all_strategies
//--- we will put our object with strategies in it
   m_all_strategies=new CArrayObj;
   if(m_all_strategies==NULL)
     {
      Print("Error of creation of the object m_all_strategies"); return(-1);
     }

// create 10 trading strategies CStrategyMA (trading by moving averages)
// initialize them, set parameters
// and add to the m_all_strategies container
   for(int i=0; i<10; i++)
     {
      CStrategyMA *t_StrategyMA;
      t_StrategyMA=new CStrategyMA;
      if(t_StrategyMA==NULL)
        {
         delete m_all_strategies;
         Print("Error of creation of object of the CStrategyMA type");
         return(-1);
        }
      //set period for each strategy
      int period=3+i*10;
      // initialize strategy
      t_StrategyMA.Initialization(period,true);
      // set details of the strategy
      t_StrategyMA.SetStrategyInfo(_Symbol,"[MA_"+IntegerToString(period)+"]",period,"Moving Averages "+IntegerToString(period));
      //t_StrategyMA.Set_Stops(3500,1000);

      //add the object of the strategy to the array of objects m_all_strategies
      m_all_strategies.Add(t_StrategyMA);
     }

   for(int i=0; i<m_all_strategies.Total(); i++)
     {
      CSampleStrategy *t_SampleStrategy;
      t_SampleStrategy=m_all_strategies.At(i);
      Print(i," Strategy name:",t_SampleStrategy.StrategyName(),
              " Strategy ID:",t_SampleStrategy.StrategyID(),
              " Virtual trading:",t_SampleStrategy.IsVirtualTradeAllowed());
     }
//---
   return(0);
  }
```

The set of trading strategies is prepared in the **Expert\_OnInit** function. First of all, the object of the m\_all\_strategies dynamic array is created.

In this case, we created ten instances of the **CStrategyMA** class. Each of them was initialized (in this case, we set different periods and allowed "virtual" trading) using the **Initialization** function.

Then, using the **SetStrategyInfo** function we set the financial instrument, strategy name and comment.

If necessary, using the **Set\_Stops(TP,SL)** function we can specify a value (in points) of Take Profit and Stop Loss, that will be executed during the "virtual" trading. We have this line commented.

Once the strategy class is created and adjusted, we add it to the m\_all\_strategies container.

All classes of trade strategies should have the **CheckTradeConditions()** function that performs the checks of trading conditions. In the class of the adaptive strategy this function is called at the beginning of each new bar, thus we give the strategies a possibility to check the values of indicators and to make the "virtual" trade operations.

Instead of ten specified moving averages (3, 13, 23...93) we can add hundreds of moving averages (instances if the **CStrategyMA** class):

```
  for(int i=0; i<100; i++)
     {
      CStrategyMA *t_StrategyMA;
      t_StrategyMA=new CStrategyMA;
      if(t_StrategyMA==NULL)
        {
         delete m_all_strategies;
         Print("Error of creation of object of the CStrategyMA type");
         return(-1);
        }
      //set period for each strategy
      int period=3+i*10;
      // initialization of strategy
      t_StrategyMA.Initialization(period,true);
      // set details of the strategy
      t_StrategyMA.SetStrategyInfo(_Symbol,"[MA_"+IntegerToString(period)+"]",period,"Moving Averages "+IntegerToString(period));
      //add the object of the strategy to the array of objects m_all_strategies
      m_all_strategies.Add(t_StrategyMA);
     }
```

Or we can add the classes of strategy that works by the signals of the stochastic oscillator (instances of the **CStrategyStoch** class):

```
  for(int i=0; i<5; i++)
     {
      CStrategyStoch *t_StrategyStoch;
      t_StrategyStoch=new CStrategyStoch;
      if(t_StrategyStoch==NULL)
        {
         delete m_all_strategies;
         printf("Error of creation of object of the CStrategyStoch type");
         return(-1);
        }
      //set period for each strategy
      int Kperiod=2+i*5;
      int Dperiod=2+i*5;
      int Slowing=3+i;
      // initialization of strategy
      t_StrategyStoch.Initialization(Kperiod,Dperiod,Slowing,true);
      // set details of the strategy
      string s=IntegerToString(Kperiod)+"/"+IntegerToString(Dperiod)+"/"+IntegerToString(Slowing);
      t_StrategyStoch.SetStrategyInfo(_Symbol,"[Stoch_"+s+"]",100+i," Stochastic "+s);
      //add the object of the strategy to the array of objects m_all_strategies
      m_all_strategies.Add(t_StrategyStoch);
     }
```

In this case the container includes 10 strategies of moving averages and 5 strategies of the stochastic oscillator.

The instances of classes of trading strategies should be the children of the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) class and should contain the **CheckTradeConditions()** function. It is better to inherit them from the **CSampleStrategy** class. Classes that implement trade strategies can be different and their number is not limited.

The Expert\_OnInit function ends with the list of strategies that are present in the m\_all\_strategies container. Note that all the strategies in the container are considered as the children of the **CSampleStrategy** class. The classes of trade strategies CStrategyMA and CStrategyStoch are also its children.

The same trick is used in the **Expert\_OnDeInit** function. In the container, we call the **SaveVirtualDeals** function for each strategy; it stores the history of performed virtual deals.

We use the name of strategy for the file name that is passed as a parameter. Then we deinitialize the strategies by calling the **Deinitialization()** function and deleting the m\_all\_strategies container:

```
//+------------------------------------------------------------------+
//| Function of deinitialization the adaptive Expert Advisor         |
//+------------------------------------------------------------------+
int CAdaptiveStrategy::Expert_OnDeInit(const int reason)
  {
   // deinitialize all strategies
   for(int i=0; i<m_all_strategies.Total(); i++)
     {
      CSampleStrategy *t_Strategy;
      t_Strategy=m_all_strategies.At(i);
      t_Strategy.SaveVirtualDeals(t_Strategy.StrategyName()+"_deals.txt");
      t_Strategy.Deinitialization();
     }
   //delete the array of object with strategies
   delete m_all_strategies;
   return(0);
  }
```

If you don't need to know about the virtual deals performed by the strategies, remove the line where tStrategy.SaveVirtualDeals is called. Note that when using the strategy tester the files are save to the /tester\_directory/Files/ directory.

Let's consider the **Expert\_OnTick** function of the CAdaptiveStrategy class that is called each time a new tick comes:

```
//+------------------------------------------------------------------+
//| Function of processing ticks of the adaptive strategy            |
//+------------------------------------------------------------------+
void CAdaptiveStrategy::Expert_OnTick()
  {
   CSampleStrategy *t_Strategy;

   // recalculate the information about positions for all strategies
   for(int i=0; i<m_all_strategies.Total(); i++)
     {
      t_Strategy=m_all_strategies.At(i);
      t_Strategy.UpdatePositionData();
     }

   // the expert advisor should check the conditions of making trade operations only when a new bar comes
   if(IsNewBar()==false) { return; }

   // check trading conditions for all strategies
   for(int i=0; i<m_all_strategies.Total(); i++)
     {
      t_Strategy=m_all_strategies.At(i);
      t_Strategy.CheckTradeConditions();
     }

   //search for the best position
   //prepare the array performance[]
   double performance[];
   ArrayResize(performance,m_all_strategies.Total());

   //request the current effectiveness for each strategy,
   //each strategy returns it in the Strategyperformance() function
   for(int i=0; i<m_all_strategies.Total(); i++)
     {
      t_Strategy=m_all_strategies.At(i);
      performance[i]=t_Strategy.StrategyPerformance();
     }
   //find the strategy (or rather its index in the m_all_strategies container)
   //with maximum value of Strategyperformance()
   int best_strategy_index=ArrayMaximum(performance,0,WHOLE_ARRAY);

   //this strategy is - t_Strategy
   t_Strategy=m_all_strategies.At(best_strategy_index);
   //request the direction of its current position
   int best_direction=t_Strategy.PositionDirection();

   string s=s+" "+t_Strategy.StrategyName()+" "+DoubleToString(t_Strategy.GetVirtualEquity())+" "+IntegerToString(best_direction);
   Print(TimeCurrent()," TOTAL=",m_all_strategies.Total(),
                       " BEST IND=",best_strategy_index,
                       " BEST STRATEGY="," ",t_Strategy.StrategyName(),
                       " BEST=",performance[best_strategy_index],"  =",
                       " BEST DIRECTION=",best_direction,
                       " Performance=",t_Strategy.StrategyPerformance());

   //if the best strategy has a negative result and doesn't have open positions, it's better to stay away from trading
   if((performance[best_strategy_index]<0) && (RealPositionDirection()==POSITION_NEUTRAL)) {return;}

   if(best_direction!=RealPositionDirection())
     {
      ProceedSignalReal(best_direction,t_Strategy.GetCurrentLotSize());
     }
  }
```

The code is very simple. Each strategy, located in the container must be able to recalculate the current financial result of its virtual positions using the current prices. It is done by calling the **UpdatePositionData()** function. Here, once again we call the strategies as the heirs of the CSampleStrategy class.

All trade operations are performed at the beginning of a new bar (the IsNewBar() function allows determining this moment as well as the [other methods](https://www.mql5.com/en/articles/22#check_new_bar) of checking new bar). In this case, the end of forming of a bar means that all the data of the previous bar (prices and indicator values) won't change anymore, so it can be analyzed on the correspondence to the trading conditions. To all the strategies we give the opportunity to perform this check and to perform their virtual trade operations by calling their **CheckTradeConditions** function.

Now we should find the most successful strategy among all strategies in the m\_all\_strategies array. To get it done, we used the Performance\[\] array, values that are returned by the **StrategyPerformance()** function of each strategy are placed into it. The base class CSampleStrategy contains this function as the difference between the current values of "virtual" Equity and Balance.

The search of index of the most successful strategy is performed using the [ArrayMaximum](https://www.mql5.com/en/docs/array/arraymaximum) function. If the best strategy has a negative profit at the moment and it doesn't have real open positions, then it's better not to trade, that's the reason why we exit from the function (see section 1).

Further, we request the direction of the virtual position of this strategy (best\_direction). If it differs from the current direction of the real position, then the current direction of the real position will be corrected (using the **ProceedSignalReal** function) according to the best\_direction direction.

**2.3. Class CSampleStrategy**

Strategies placed in the m\_all\_strategies container were considered as the heirs of the CSampleStrategy class.

This class is the base one for the trade strategies; it contains the implementation of virtual trading. In this article we will consider a simplified case of virtual trading implementation, the swaps aren't aken into consideration. The classes of trade strategies should be inherited from the **CSampleStrategy** class.

Let's show the structure of this class.

```
//+------------------------------------------------------------------+
//|                                              CSampleStrategy.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#include <Object.mqh>

#define POSITION_NEUTRAL   0     // no position
#define POSITION_LONG      1     // long position
#define POSITION_SHORT    -1     // short position

#define SIGNAL_OPEN_LONG    10   // signal to open a long position
#define SIGNAL_OPEN_SHORT  -10   // signal to open a short position
#define SIGNAL_CLOSE_LONG   -1   // signal to close a long position
#define SIGNAL_CLOSE_SHORT   1   // signal to close a short position
//+------------------------------------------------------------------+
//| Structure for storing the parameters of virtual position         |
//+------------------------------------------------------------------+
struct virtual_position
  {
   string            symbol;            // symbol
   int               direction;         // direction of the virtual position (0-no open position,+1 long,-1 short)
   double            volume;            // volume of the position in lots
   double            profit;            // current profit of the virtual position on points
   double            stop_loss;         // Stop Loss of the virtual position
   double            take_profit;       // Take Profit of the virtual position
   datetime          time_open;         // date and time of opening the virtual position
   datetime          time_close;        // date and time of closing the virtual position
   double            price_open;        // open price of the virtual position
   double            price_close;       // close price of the virtual position
   double            price_highest;     // maximum price during the life of the position
   double            price_lowest;      // minimal price during the lift of the position
   double            entry_eff;         // effectiveness of entering
   double            exit_eff;          // effectiveness of exiting
   double            trade_eff;         // effectiveness of deal
  };
//+------------------------------------------------------------------+
//| Class CSampleStrategy                                            |
//+------------------------------------------------------------------+
class CSampleStrategy: public CObject
  {
protected:
   int               m_strategy_id;            // Strategy ID
   string            m_strategy_symbol;        // Symbol
   string            m_strategy_name;          // Strategy name
   string            m_strategy_comment;       // Comment

   MqlTick           m_price_last;             // Last price
   MqlRates          m_rates[];                // Array for current quotes
   bool              m_virtual_trade_allowed;  // Flag of allowing virtual trading
   int               m_current_signal_state;   // Current state of strategy
   double            m_current_trade_volume;   // Number of lots for trading
   double            m_initial_balance;        // Initial balance (set in the constructor, default value is 10000)
   int               m_sl_points;              // Stop Loss
   int               m_tp_points;              // Take Profit

   virtual_position  m_position;               // Virtual position
   virtual_position  m_deals_history[];        // Array of deals
   int               m_virtual_deals_total;    // Total number of deals

   double            m_virtual_balance;           // "Virtual" balance
   double            m_virtual_equity;            // "Virtual" equity
   double            m_virtual_cumulative_profit; // cumulative "virtual" profit
   double            m_virtual_profit;            // profit of the current open "virtual" position

   //checks and closes the virtual position by stop levels if it is necessary
   bool              CheckVirtual_Stops(virtual_position &position);
   // recalculation of position and balance
   void              RecalcPositionProperties(virtual_position &position);
   // recalculation of open virtual position in accordance with the current prices
   void              Position_RefreshInfo(virtual_position &position);
   // open virtual short position
   void              Position_OpenShort(virtual_position &position);
   // closes virtual short position
   void              Position_CloseShort(virtual_position &position);
   // opens virtual long position
   void              Position_OpenLong(virtual_position &position);
   // closes the virtual long position
   void              Position_CloseLong(virtual_position &position);
   // closes open virtual position
   void              Position_CloseOpenedPosition(virtual_position &position);
   // adds closed position to the m_deals_history[] array (history of deals)
   void              AddDealToHistory(virtual_position &position);
   //calculates and returns the recommended volume that will be used in trading
   virtual double    MoneyManagement_CalculateLots(double trade_volume);
public:
   // constructor
   void              CSampleStrategy();
   // destructor
   void             ~CSampleStrategy();

   //returns the current size of virtual balance
   double            GetVirtualBalance() { return(m_virtual_balance); }
   //returns the current size of virtual equity
   double            GetVirtualEquity() { return(m_virtual_equity); }
   //returns the current size of virtual profit of open position
   double            GetVirtualProfit() { return(m_virtual_profit); }

   //sets Stop Loss and Take Profit in points
   void              Set_Stops(int tp,int sl) {m_tp_points=tp; m_sl_points=sl;};
   //sets the current volume in lots
   void              SetLots(double trade_volume) {m_current_trade_volume=trade_volume;};
   //returns the current volume in lots
   double            GetCurrentLots() { return(m_current_trade_volume); }

   // returns strategy name
   string            StrategyName() { return(m_strategy_name); }
   // returns strategy ID
   int               StrategyID() { return(m_strategy_id); }
   // returns the comment of strategy
   string            StrategyComment() { return(m_strategy_comment); }
   // sets the details of strategy (symbol, name and ID of strategy)
   void              SetStrategyInfo(string symbol,string name,int id,string comment);

   // set the flag of virtual trading (allowed or not)
   void              SetVirtualTradeFlag(bool pFlag) { m_virtual_trade_allowed=pFlag; };
   // returns flag of allowing virtual trading
   bool              IsVirtualTradeAllowed() { return(m_virtual_trade_allowed); };

   // returns the current state of strategy
   int               GetSignalState();
   // sets the current state of strategy (changes virtual position if necessary)
   void              SetSignalState(int state);
   // changes virtual position in accordance with the current state
   void              ProceedSignalState(virtual_position &position);

   // sets the value of cumulative "virtual" profit
   void              SetVirtualCumulativeProfit(double cumulative_profit) { m_virtual_cumulative_profit=cumulative_profit; };

   //returns the effectiveness of strategy ()
   double            StrategyPerformance();

   //updates position data
   void              UpdatePositionData();
   //closes open virtual position
   void              CloseVirtualPosition();
   //returns the direction of the current virtual position
   int               PositionDirection();
   //virtual function of initialization
   virtual int       Initialization() {return(0);};
   //virtual function of checking trade conditions
   virtual bool      CheckTradeConditions() {return(false);};
   //virtual function of deinitialization
   virtual int       Deinitialization() {return(0);};

   //saves virtual deals to a file
   void              SaveVirtualDeals(string file_name);
  };
```

We won't analyze its detailed description, additional information can be found in the CSampleStrategy.mqh file. There you can also find the function of checking new bar - IsNewBar.

### 3\. Classes of Trade Strategies

This section is devoted to the structure of classes of trade strategies that are used in the adaptive Expert Advisor.

**3.1. Class CStrategyMA - Strategy of Trading by Moving Averages**

```
//+------------------------------------------------------------------+
//|                                                  CStrategyMA.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#include <CSampleStrategy.mqh>
//+------------------------------------------------------------------+
//| Class CStrategyMA for implementation of virtual trading          |
//| by the strategy based on moving average                          |
//+------------------------------------------------------------------+
class CStrategyMA : public CSampleStrategy
  {
protected:
   int               m_handle;     // handle of the Moving Average (iMA) indicator
   int               m_period;     // period of the Moving Average indicator
   double            m_values[];   // array for storing values of the indicator
public:
   // initialization of the strategy
   int               Initialization(int period,bool virtual_trade_flag);
   // deinitialization of the strategy
   int               Deinitialization();
   // checking trading conditions and opening virtual positions
   bool              CheckTradeConditions();
  };
```

The CStrategyMA class is a child of the **CSampleStrategy** class where the entire functionality of virtual trading is implemented.

The protected section contains internal variables that will be used in the class of the strategy. These are: m\_handle - handle of the [iMA](https://www.mql5.com/en/docs/indicators/ima) indicator, m\_period - period of the moving average, m\_values\[\] - array that will be used in the **CheckTradeConditions function** for getting current values of the indicator.

The public section contains three functions that provide the implementation of the trade strategy.

- **Function Initialization.** The strategy is initialized here. If you need to create indicators, create them here.

- **Function** **Deinitialization.** The strategy is deinitialized here. The handles of indicators are released here.

- **Function** **СheckTradeConditions.** Here, the strategy checks the trading conditions and generates trade signals that are used for the virtual trading. To perform virtual trade operations, the SetSignalState function of the CStrategy parent class is called; one of four of the following trade signals are is passed to it:

> 1. The signal for opening a long position (SIGNAL\_OPEN\_LONG)
>
> 2. The signal for opening a short position (SIGNAL\_OPEN\_SHORT)
>
> 3. The signal for closing a long position (SIGNAL\_CLOSE\_LONG)
>
> 4. The signal for closing a short position (SIGNAL\_CLOSE\_SHORT)

```
//+------------------------------------------------------------------+
//| Strategy Initialization Method                                   |
//+------------------------------------------------------------------+
int CStrategyMA::Initialization(int period,bool virtual_trade_flag)
  {
   // set period of the moving average
   m_period=period;
   // set specified flag of virtual trading
   SetVirtualTradeFlag(virtual_trade_flag);

   //set indexation of arrays like the one of timeseries
   ArraySetAsSeries(m_rates,true);
   ArraySetAsSeries(m_values,true);

   //create handle of the indicator
   m_handle=iMA(_Symbol,_Period,m_period,0,MODE_EMA,PRICE_CLOSE);
   if(m_handle<0)
     {
      Alert("Error of creation of the MA indicator - error number: ",GetLastError(),"!!");
      return(-1);
     }

   return(0);
  }
//+------------------------------------------------------------------+
//| Strategy Deinitialization Method                                 |
//+------------------------------------------------------------------+
int CStrategyMA::Deinitialization()
  {
   Position_CloseOpenedPosition(m_position);
   IndicatorRelease(m_handle);
   return(0);
  };
//+------------------------------------------------------------------+
//| Checking trading conditions and opening virtual positions        |
//+------------------------------------------------------------------+
bool CStrategyMA::CheckTradeConditions()
  {
   RecalcPositionProperties(m_position);
   double p_close;

   // get history data of the last three bars
   if(CopyRates(_Symbol,_Period,0,3,m_rates)<0)
     {
      Alert("Error of copying history data - error:",GetLastError(),"!!");
      return(false);
     }
   // Copy the current price of closing of the previous bar (it is bar 1)
   p_close=m_rates[1].close;  // close price of the previous bar

   if(CopyBuffer(m_handle,0,0,3,m_values)<0)
     {
      Alert("Error of copying buffers of the Moving Average indicator - error number:",GetLastError());
      return(false);
     }

   // buy condition 1: MA rises
   bool buy_condition_1=(m_values[0]>m_values[1]) && (m_values[1]>m_values[2]);
   // buy condition 2: previous price is greater than the MA
   bool buy_condition_2=(p_close>m_values[1]);

   // sell condition 1: // MA falls
   bool sell_condition_1=(m_values[0]<m_values[1]) && (m_values[1]<m_values[2]);
   // sell condition 2: // previous price is lower than the MA
   bool sell_condition_2=(p_close<m_values[1]);

   int new_state=0;

   if(buy_condition_1  &&  buy_condition_2) new_state=SIGNAL_OPEN_LONG;
   if(sell_condition_1 && sell_condition_2) new_state=SIGNAL_OPEN_SHORT;

   if((GetSignalState()==SIGNAL_OPEN_SHORT) && (buy_condition_1 || buy_condition_2)) new_state=SIGNAL_CLOSE_SHORT;
   if((GetSignalState()==SIGNAL_OPEN_LONG) && (sell_condition_1 || sell_condition_2)) new_state=SIGNAL_CLOSE_LONG;

   if(GetSignalState()!=new_state)
     {
      SetSignalState(new_state);
     }

   return(true);
  };
```

The concept is simple - on the basis of indicator states and prices, the signal type (new\_state) is determined, then the current state of the virtual trading is requested (using the **GetSignalState** function); and if they're not the same, the **SetSignalState** function is called for "correcting" the virtual position.

**3.2. Class CStrategyStoch - the Strategy of Trading by Stochastic**

The code of the class that performs trading on the basis of intersection of the main and signal lines of the [iStochastic](https://www.mql5.com/en/docs/indicators/istochastic) oscillator is given below:

```
//+------------------------------------------------------------------+
//|                                               CStrategyStoch.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#include <CSampleStrategy.mqh>
//+------------------------------------------------------------------+
//| Class CStrategyStoch for implementation of virtual trading by    |
//| the strategy of intersection of lines of stochastic oscillator   |
//+------------------------------------------------------------------+
class CStrategyStoch : public CSampleStrategy
  {
protected:
   int               m_handle;          // handle of the Stochastic Oscillator (iStochastic)
   int               m_period_k;        // K-period (number of bars for calculations)
   int               m_period_d;        // D-period (period of primary smoothing)
   int               m_period_slowing;  // final smoothing
   double            m_main_line[];     // array for storing indicator values
   double            m_signal_line[];   // array for storing indicator values
public:
   // initialization of strategy
   int               Initialization(int period_k,int period_d,int period_slowing,bool virtual_trade_flag);
   // deinitialization of strategy
   int               Deinitialization();
   // checking trading conditions and opening virtual positions
   bool              CheckTradeConditions();
  };
//+------------------------------------------------------------------+
//| Strategy Initialization Method                                   |
//+------------------------------------------------------------------+
int CStrategyStoch::Initialization(int period_k,int period_d,int period_slowing,bool virtual_trade_flag)
  {
   // Set period of the oscillator
   m_period_k=period_k;
   m_period_d=period_d;
   m_period_slowing=period_slowing;

   // set specified flag of the virtual trading
   SetVirtualTradeFlag(virtual_trade_flag);

   // set indexation of arrays like the one of timeseries
   ArraySetAsSeries(m_rates,true);
   ArraySetAsSeries(m_main_line,true);
   ArraySetAsSeries(m_signal_line,true);

   // create handle of the indicator
   m_handle=iStochastic(_Symbol,_Period,m_period_k,m_period_d,m_period_slowing,MODE_SMA,STO_LOWHIGH);
   if(m_handle<0)
     {
      Alert("Error of creating the Stochastic indicator - error number: ",GetLastError(),"!!");
      return(-1);
     }

   return(0);
  }
//+------------------------------------------------------------------+
//| Strategy Deinitialization Method                                 |
//+------------------------------------------------------------------+
int CStrategyStoch::Deinitialization()
  {
   // close all open positions
   Position_CloseOpenedPosition(m_position);
   // release handle of the indicator
   IndicatorRelease(m_handle);
   return(0);
  };
//+------------------------------------------------------------------+
//| Checking Trading Conditions and Opening Virtual Positions        |
//+------------------------------------------------------------------+
bool CStrategyStoch::CheckTradeConditions()
  {
   // call the functions of recalculation of position parameters
   RecalcPositionProperties(m_position);
   double p_close;

   // get history  data of the last 3 bars
   if(CopyRates(_Symbol,_Period,0,3,m_rates)<0)
     {
      Alert("Error of copying history data - error:",GetLastError(),"!!");
      return(false);
     }
   // copy the current close price of the previous bar (it is bar 1)
   p_close=m_rates[1].close;  // close price of the previous bar

   if((CopyBuffer(m_handle,0,0,3,m_main_line)<3) || (CopyBuffer(m_handle,1,0,3,m_signal_line)<3))
     {
      Alert("Error of copying buffers of the Stochastic indicator - error number:",GetLastError());
      return(false);
     }

   // buy condition: crossing the signal line by the main one from bottom up
   bool buy_condition=((m_signal_line[2]<m_main_line[2]) && (m_signal_line[1]>m_main_line[1]));
   // sell condition: crossing the signal line by the main one from top downwards
   bool sell_condition=((m_signal_line[2]>m_main_line[2]) && (m_signal_line[1]<m_main_line[1]));

   int new_state=0;

   if(buy_condition) new_state=SIGNAL_OPEN_LONG;
   if(sell_condition) new_state=SIGNAL_OPEN_SHORT;

   if((GetSignalState()==SIGNAL_OPEN_SHORT) && (buy_condition)) new_state=SIGNAL_CLOSE_SHORT;
   if((GetSignalState()==SIGNAL_OPEN_LONG) && (sell_condition)) new_state=SIGNAL_CLOSE_LONG;

   if(GetSignalState()!=new_state)
     {
      SetSignalState(new_state);
     }

   return(true);
  };
```

As you see,  the only differences between the structure of the **CStrategyStoch** class and the one of **CStrategyMA** are the initialization function (different parameters), the type of indicator used and the trade signals.

Thus, to use your strategies in the adaptive Expert Advisor, you should rewrite them in the form of classes of such type and load them to the m\_all\_strategies container.

### **4\. Results of Analysis of the Adaptive Trade Strategies**

In this section, we're going to discuss several aspects of practical use of the adaptive strategies and the methods of improving them.

**4.1. Enhancing the System with Strategies that Use Inversed Signals**

Moving Averages are not good when there are no trends. We've already met this kind of situation - in the figure 3, you can see that there was no trend within the period from 8-th to 20-th of January; so all 10 strategies that use moving averages in trading had a virtual loss. The adaptive system stopped trading as a result of absence of a strategy with positive amount of money earned. Is there any way to avoid such negative effect?

Let's add to our 10 strategies (MA\_3, MA\_13, ... MA\_93) another 10 classes **CStrategyMAinv**, whose trade signals are reversed (the conditions are the same but SIGNAL\_OPEN\_LONG/SIGNAL\_OPEN\_SHORT and SIGNAL\_CLOSE\_LONG/SIGNAL\_CLOSE\_SHORT exchanged their places). Thus, in addition to ten trend strategies (instances of the **CStrategyMA** class), we have another ten counter-trend strategies (instances of the **CStrategyMAinv** class).

The result of using the adaptive system that consists of twenty strategies is shown in the figure 5.

![Figure 5. Diagrams of equity at the account of the adaptive strategy that uses 20 trade signals: 10 moving averages CAdaptiveMA and 10 "mirrored" ones CAdaptiveMAinv](https://c.mql5.com/2/2/Image5.png)

Figure 5. Diagrams of equity at the account of the adaptive strategy that uses 20 trade signals: 10 moving averages CAdaptiveMA and 10 "mirrored" ones CAdaptiveMAinv

As you can see at the figure 5, during the period when all the **CAdaptiveMA** strategies had a negative result, following the **CAdaptiveMAinv** strategies allowed the Expert Advisor to avoid unwanted drawdowns at the very beginning of trading.

![Figure 6. Time period when the adaptive strategy used the signals of "counter-trend" CAdaptiveMAinv strategies](https://c.mql5.com/2/2/Image6.png)

Figure 6. Time period when the adaptive strategy used the signals of "counter-trend"CAdaptiveMAinv strategies

This kind of approach may seem unacceptable, since losing the deposit is just a question of time when using a counter-trend strategy. However in our case, we're not limited with a single strategy. The market knows better which strategies are effective at the moment.

The strong side of adaptive systems is the market suggests by itself which strategy should be used and when it should be used.

It gives a possibility to abstract from the logic of strategies - if a strategy is effective, then the way it works is of no significance. The adaptive approach uses the only criterion of success of a strategy - its effectiveness.

**4.2. Is It Worth to Invert the Signals of the Worst Strategy?**

The trick with inversion shown above leads to a thought about the potential possibility of using the signals of the worst strategy. If a strategy is unprofitable (and the worst one at that), then can we get a profit by acting in reverse?

Can we turn a losing strategy into a profitable one by a simple change of its signals? To answer this question, we need to change [ArrayMaximum](https://www.mql5.com/en/docs/array/arraymaximum) with [ArrayMinimum](https://www.mql5.com/en/docs/array/arrayminimum) in the Expert\_OnTick() function of the CAdaptiveStrategy class, as well as to implement the change of directions by multiplying value of the BestDirection variable by -1.

In addition, we need to comment the limitation of virtual trading in case of negative effectiveness (since we are going to analyze the result of the worst strategy):

```
//if((Performance[BestStrategyIndex]<0) && (RealPositionDirection()==0)) {return;}
```

Diagram of equity of the adaptive Expert Advisor that uses the reversed signals of the worst strategy is shown in the figure 7:

![Figure 7. Diagrams of equity at the accounts of ten strategies and the adaptive system that uses the reversed signals of the worst system](https://c.mql5.com/2/2/Image7.png)

Figure 7. Diagrams of equity at the accounts of ten strategies and the adaptive system that uses the reversed signals of the worst system

In this case, the least successful strategy for most of the time was the one based on intersection of moving averages with period 3 (MA\_3). As you can see at the figure 7, the reverse correlation between MA\_3 (blue colored) and the adaptive strategy (red colored) exists , but the financial result of the adaptive system doesn't impress.

Copying (and reversing) the signals of the worst strategy doesn't lead to improving the effectiveness of trading.

**4.2. Why the Bunch of Moving Averages is not so Effective as it Seems?**

Instead of 10 moving averages you can use lots of them by adding another hundred of CStrategyMA strategies with different periods to the m\_all\_strategies container.

To do it, slightly change the code in the CAdaptiveStrategy class:

```
   for(int i=0; i<100; i++)
     {
      CStrategyMA *t_StrategyMA;
      t_StrategyMA=new CStrategyMA;
      if(t_StrategyMA==NULL)
        {
         delete m_all_strategies;
         Print("Error of creation of object of the CStrategyMA type");
         return(-1);
        }
      //set period for each strategy
      int period=3+i*10;
      // initialization of strategy
      t_StrategyMA.Initialization(period,true);
      // set details of the strategy
      t_StrategyMA.SetStrategyInfo(_Symbol,"[MA_"+IntegerToString(period)+"]",period,"Moving Averages "+IntegerToString(period));
      //add the object of the strategy to the array of objects m_all_strategies
      m_all_strategies.Add(t_StrategyMA);
     }
```

However, you should understand that close moving averages will inevitably intersect; the leader will constantly change; and the adaptive system will switch its states and open/close positions more frequently than it is necessary. As a result, the characteristics of the adaptive system will become worse. You can make sure in it on your own by comparing the statistical characteristics of the system (the "Results" tab of the strategy tester).

It's better not to make adaptive systems based on many strategies with close parameters.

### **5\.** What Should Be Considered

The m\_all\_strategies container can have thousands of instances of suggested strategies included, you can even add all the strategies with different parameters; however, to win the [Automated Trading Championship 2010](https://championship.mql5.com/2010/en), you need to develop the advanced money management system. Note that we have used the trading volume equal 0.1 lots for testing on history data (and in the code of classes) .

**5.1 How to Increase the Profitability of the Adaptive Expert Advisor?**

The CSampleStrategy class has the virtual function **MoneyManagement\_CalculateLots**:

```
//+------------------------------------------------------------------+
//| The function returns the recommended volume for a strategy       |
//| Current volume is passed to it as a parameter                    |
//| Volume can be set depending on:                                  |
//| current m_virtual_balance and m_virtual_equity                   |
//| current statistics of deals (located in m_deals_history)         |
//| or any other thing you want                                      |
//| If a strategy doesn't require change of volume                   |
//| you can return the passed value of volume:  return(trade_volume);|
//+------------------------------------------------------------------+
double CSampleStrategy::MoneyManagement_CalculateLots(double trade_volume)
  {
   //return what we've got
   return(trade_volume);
  }
```

To manage the volume for trading, you can use the statistical information about the results and characteristics of virtual deals that is recorded in the m\_deals\_history\[\] array.

If you need to increase the volume (for example, to double it if the last virtual deals in m\_deals\_history\[\] are profitable; or to decrease it), you should change the returned value in the corresponding way.

**5.2 Using the Deals Statistics for Calculation of Strategy Performance**

The StrategyPerformance() function, implemented in the **CSampleStrategy** class is intended for the calculation of the strategy performance,

```
//+-----------------------------------------------------------------------+
//| Function StrategyPerformance - the function of strategy effectiveness |
//+-----------------------------------------------------------------------+
double CSampleStrategy::StrategyPerformance()
  {
   //returns effectiveness of a strategy
   //in this case it's the difference between the amount
   //of equity at the moment and the initial balance,
   //i.e. the amount of assets earned by the strategy
   double performance=(m_virtual_equity-m_initial_balance);
   return(performance);
  }
```

The formula of effectiveness of a strategy can be more complex and, for example, include the effectiveness of entering, exiting, the effectiveness of deals, profits, drawdowns, etc.

The calculation of the effectiveness of entering, exiting and the effectiveness of deals (the entry\_eff, exit\_eff and trade\_eff fields of structures of the m\_deals\_history\[\] array) is performed automatically during the virtual trading (see the CSampeStrategy class). This statistical information can be used for making your own, more complex rates of the effectiveness of strategy.

For example, as a characteristics of effectiveness you can use the profit of last three deals (use the pos\_Profit field from the archive of deals m\_deals\_history\[\]):

```
double CSampleStrategy::StrategyPerformance()
  {
  //if there are deals, multiply this value by the result of three last deals
   if(m_virtual_deals_total>0)
     {
      int avdeals=MathRound(MathMin(3,m_virtual_deals_total));
      double sumprofit=0;
      for(int j=0; j<avdeals; j++)
        {
         sumprofit+=m_deals_history[m_virtual_deals_total-1-j].profit;
        }
      double performance=sumprofit/avdeals;
     }
     return(performance);

  }
```

If you want to change this function, change it only in the CSampleStrategy class, it must be the same for all trade strategies of the adaptive system. However, you should remember that the difference between Equity and Balance is also a good factor of effectiveness.

**5.3 Using Take Profit and Stop Loss**

You can change the effectiveness of trading systems by setting fixed stop levels (it can be done by calling the Set\_Stops function; it allows setting the stop levels in points for virtual trading). If the levels are specified, closing of virtual positions will be performed automatically; this functionality is implemented in the **CSampleStrategy** class.

In our example (see 2.2, the function of classes of moving averages), the function of setting stop levels is commented.

**5.4. Periodic Zeroizing of Cumulative Virtual Profit**

The adaptive approach has the same disadvantage as common strategies have. If the leading strategy starts losing, the adaptive system starts losing as well. That is the reason why sometimes you need to "zeroize" the results of working of all strategies and to close all their virtual positions.

To do it, the following functions are implemented in the **CSampleStrategy** class:

```
// sets a value for cumulative "virtual profit"
 void              SetVirtualCumulativeProfit(double cumulative_profit) { m_virtual_cumulative_profit=cumulative_perofit; };
//closes an open virtual position
 void              CloseVirtualPosition();
```

CheckPoint of this kind can be used from time to time, for example after each N bars.

**5.5. No Miracles**

You should remember that the adaptive system is not a grail (USDJPY H1, 4.01.2010-20.08.2010):

![Figure 8. Balance and equity curves of the adaptive system that uses the signals of the best of 10 strategies (USDJPY H1)](https://c.mql5.com/2/2/Image8__1.png)

Figure 8. Balance and equity curves of the adaptive system that uses the signals of the best of 10 strategies (USDJPY H1)

Equity curves of all the strategies are shown in the figure 9.

![Figure 9. Equity curves at the account with the adaptive system based on 10 strategies (USDJPY H1)](https://c.mql5.com/2/2/Image9.png)

Figure 9. Equity curves at the account with the adaptive system based on 10 strategies (USDJPY H1)

If there are no profitable strategies in the adaptive system, using them is not effective. Use profitable strategies.

We should consider another important and interesting thing. Pay attention to the behavior of the adaptive strategy at the very beginning of trading:

![Figure 10. Equity curves at the account with 10 strategies of the adaptive strategy](https://c.mql5.com/2/2/Image10.png)

Figure 10. Equity curves at the account with 10 strategies of the adaptive strategy

At first, all the strategies had negative results and the adaptive strategy stopped trading; then it started switching between strategies that had a positive result; and then all the strategies became unprofitable again.

All the strategies have the same balance in the beginning. And only after a while, one or another strategy becomes a leader; thus it's recommended to set a limitation in the adaptive strategy to avoid trading at first bars. To do it, supplement the Expert\_OnTick function of the **CAdaptiveStrategy** class with a variable, which value is increased each time a new bar comes.

In the beginning, until the market chooses the best strategy, you should stay away from real trading.

### Conclusions

In this article, we have considered an example of the adaptive system that consists of many strategies, each of which makes its own "virtual" trade operations. Real trading is performed in accordance with the signals of a most profitable strategy at the moment.

Thanks to using the [object-oriented](https://www.mql5.com/en/docs/basis/oop) approach, [classes for working with data](https://www.mql5.com/en/docs/standardlibrary/datastructures) and [trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) of the Standard library, the architecture of the system appeared to be simple and scalable; now you can easily create and analyze the adaptive systems that include hundreds of trade strategies.

P.S. For the convenience analysis of behavior of adaptive systems, the debug version of the CSampleStrategy class is attached (the adaptive-systems-mql5-sources-debug-en.zip archive). The difference of this version is creation of text files during its working; they contain the summary reports on the dynamics of changing of virtual balance/equity of the strategies included in the system.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/143](https://www.mql5.com/ru/articles/143)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/143.zip "Download all attachments in the single ZIP archive")

[adaptive-systems-mql5-doc-en.zip](https://www.mql5.com/en/articles/download/143/adaptive-systems-mql5-doc-en.zip "Download adaptive-systems-mql5-doc-en.zip")(274.09 KB)

[adaptive-systems-mql5-sources-debug-en.zip](https://www.mql5.com/en/articles/download/143/adaptive-systems-mql5-sources-debug-en.zip "Download adaptive-systems-mql5-sources-debug-en.zip")(12.78 KB)

[adaptive-systems-mql5-sources-en.zip](https://www.mql5.com/en/articles/download/143/adaptive-systems-mql5-sources-en.zip "Download adaptive-systems-mql5-sources-en.zip")(12.38 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/2038)**
(39)


![Alireza](https://c.mql5.com/avatar/2021/3/605E016E-D43E.png)

**[Alireza](https://www.mql5.com/en/users/rozen1977)**
\|
18 Sep 2022 at 16:52

Wondering has anyone succeeded in this mind blowing idea. This literally allows to optimize the EA on-the-fly.


![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
18 Sep 2022 at 17:45

**Alireza [#](https://www.mql5.com/en/forum/2038/page3#comment_42134166):** Wondering has anyone succeeded in this mind blowing idea. This literally allows to optimize the EA on-the-fly.

Yes, I use a similar approach on several of my EAs, usually based on EMAs because they can be calculated incrementally and use very little CPU and RAM.

![Alireza](https://c.mql5.com/avatar/2021/3/605E016E-D43E.png)

**[Alireza](https://www.mql5.com/en/users/rozen1977)**
\|
19 Sep 2022 at 17:47

**Fernando Carreiro [#](https://www.mql5.com/en/forum/2038/page3#comment_42134502):**

Yes, I use a similar approach on several of my EAs, usually based on EMAs because they can be calculated incrementally and use very little CPU and RAM.

Thank you very much Fernando for reply.


![gardee005](https://c.mql5.com/avatar/avatar_na2.png)

**[gardee005](https://www.mql5.com/en/users/gardee005)**
\|
27 Oct 2024 at 22:34

is there anyway to know the current number of open virtual positions for each virtual strategy at any time?


![Mashinetrud](https://c.mql5.com/avatar/avatar_na2.png)

**[Mashinetrud](https://www.mql5.com/en/users/mashinetrud)**
\|
31 Jul 2025 at 08:38

Since 2015, I have reached a similar approach myself. However, my selection is not made on each candle, but on a shifting window. The results of adaptation so far are worse than presented here. From the constructive additions: you need to initially consider everything with the commission that some branches will bend strongly downwards. And most importantly, there will be three classes of outcomes of one deal: positive, negative because of the commission and negative more than the commission. So only the last ones can be reversed.


![Contest of Expert Advisors inside an Expert Advisor](https://c.mql5.com/2/17/922_20.jpg)[Contest of Expert Advisors inside an Expert Advisor](https://www.mql5.com/en/articles/1578)

Using virtual trading, you can create an adaptive Expert Advisor, which will turn on and off trades at the real market. Combine several strategies in a single Expert Advisor! Your multisystem Expert Advisor will automatically choose a trade strategy, which is the best to trade with at the real market, on the basis of profitability of virtual trades. This kind of approach allows decreasing drawdown and increasing profitability of your work at the market. Experiment and share your results with others! I think many people will be interested to know about your portfolio of strategies.

![Several Ways of Finding a Trend in MQL5](https://c.mql5.com/2/0/Determine_Trend_MQL5.png)[Several Ways of Finding a Trend in MQL5](https://www.mql5.com/en/articles/136)

Any trader would give a lot for opportunity to accurately detect a trend at any given time. Perhaps, this is the Holy Grail that everyone is looking for. In this article we will consider several ways to detect a trend. To be more precise - how to program several classical ways to detect a trend by means of MQL5.

![Protect Yourselves, Developers!](https://c.mql5.com/2/17/846_12.gif)[Protect Yourselves, Developers!](https://www.mql5.com/en/articles/1572)

Protection of intellectual property is still a big problem. This article describes the basic principles of MQL4-programs protection. Using these principles you can ensure that results of your developments are not stolen by a thief, or at least to complicate his "work" so much that he will just refuse to do it.

![Interview with Alexander Topchylo (ATC 2010)](https://c.mql5.com/2/0/35.png)[Interview with Alexander Topchylo (ATC 2010)](https://www.mql5.com/en/articles/527)

Alexander Topchylo (Better) is the winner of the Automated Trading Championship 2007. Alexander is an expert in neural networks - his Expert Advisor based on a neural network was on top of best EAs of year 2007. In this interview Alexander tells us about his life after the Championships, his own business and new algorithms for trading systems.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/143&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062781327159240900)

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