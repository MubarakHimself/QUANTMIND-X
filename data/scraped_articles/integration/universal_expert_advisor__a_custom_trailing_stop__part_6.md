---
title: Universal Expert Advisor: A Custom Trailing Stop (Part 6)
url: https://www.mql5.com/en/articles/2411
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:18:52.106374
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/2411&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071739469297560969)

MetaTrader 5 / Examples


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/2411#intro)
- [Trailing function implementation options](https://www.mql5.com/en/articles/2411#c1)
- [Standardization of Trailing Functions. The CTrailing Class](https://www.mql5.com/en/articles/2411#c2)
- [Interaction of CTrailing with other modules of the strategy](https://www.mql5.com/en/articles/2411#c3)
- [Practical use of a trailing stop. An example of a classical trailing function](https://www.mql5.com/en/articles/2411#c4)
- [Adding a trailing stop to the CImpulse strategy](https://www.mql5.com/en/articles/2411#c5)
- [Adding trailing parameters to the Expert Advisor settings](https://www.mql5.com/en/articles/2411#c6)
- [Moving Average based trailing stop](https://www.mql5.com/en/articles/2411#c7)
- [Individual trailing stop for a position](https://www.mql5.com/en/articles/2411#c8)
- [A short list of fixes in the latest version](https://www.mql5.com/en/articles/2411#c9)
- [Conclusion](https://www.mql5.com/en/articles/2411#exit)

### Introduction

This material continues a series of articles about the so-called 'Universal Expert Advisor' — a specific set of classes that constitute a trading engine allowing users to create their own strategies. In the previous parts of the article, we discussed various modules of the engine, which allow expanding the basic functionality of any strategy created based on the engine. However, most of these modules are either part of the CStrategy class, or they implement objects with which this class operates directly.

In this part we continue to develop the functionality of the CStrategy trading engine and consider a new opportunity: _**support for a trailing stop**_. Unlike other CStrategy modules, trailing stop algorithms are _external_ in relation to the trading engine. This means that their presence, as well as their absence in no way should affect the functionality of CStrategy. This feature can be implemented by using a special programming technique called _**composition**_. This technique will be described later in this article. We also continue to add new functions using additional modules or classes, strictly following the principles of object-oriented programming philosophy.

### Trailing function implementation options

A Trailing Stop is a kind of algorithm, and its only task is to move a stop-loss order to a certain price level in order to protect a position from excessive loss. Obviously, there can be a lot of algorithms for managing stop-loss of a position. A trailing stop management algorithm could be implemented as a separate method within the CStrategy class. For example, it could receive the current position as a parameter and then return the trade level to move the current stop-loss order to:

```
class CStrategy
   {
public:
   double TrailingMode1(CPosition* pos);
   };
```

Then position management as per the trailing function would be possible straight inside the strategy:

```
void CMovingAverage::SupportBuy(const MarketEvent &event,CPosition *pos)
  {
   double new_sl = TrailingMode1(pos);
   pos.StopLossValue(new_sl);
  }
```

However, since there can be a lot of trailing stop management functions, it is highly undesirable to place it inside the CStrategy class. A trailing stop is an _external_ with respect to the basic algorithm of the Expert Advisor. It is not a necessary function for the trading engine operation, and its task is only to simplify the trading process. Therefore absence of trailing functions should not affect the functioning of CStrategy or of a strategy that does not use any trailing stop. On the other hand, the existence of stop management algorithms should not complicate the code readability or increase the code of the base classes. That is why all trailing stop functions should be placed in separate classes and files that are connectible to a trading strategy on demand.

Alternatively, we could implement a trailing function in the CPosition class. In this case, operation of a trailing stop would look something like this:

```
void CMovingAverage::SupportBuy(const MarketEvent &event,CPosition *pos)
  {
   pos.TrailingMode1();
  }
```

However, this would only transfer the problem from the CStrategy class to CPosition. In this case, the CPosition class would contain odd, though useful position management algorithms, the number of which could be quite large.

In addition, the trailing algorithms require the configuration of certain parameters. For example, for a classic trailing stop we need to specify the distance in pips between the reached extreme price values and the stop-loss level, which we want to be managed. Therefore, in addition to the algorithms themselves, we need to store their operation parameters somewhere. If you store this data in infrastructure classes, such as CPosition or CStrategy, this would confuse internal variables of these classes with a lot of trailing variables, and thus it would significantly complicate operation of these classes.

### Standardization of trailing functions. The CTrailing class

Very often, the most effective solution is the most simple and proven one. The case of a trailing stop is no exception. If we imagine that a trailing stop is a special class that stores its operation parameters and the stop moving algorithm as a method, then all of the above problems with the location of these functions would be solved. Well, if we create the trailing stop as an independent class, its data and methods will not be confused with the data and methods of the base CStrategy class and other important infrastructure objects, such as CPosition.

When developing this class, we need to solve two questions.

1. The inside of the trailing stop class. Standardization of its operation.

2. Interaction of the class with other modules of the CStrategy engine.

Let us consider the first of these questions. Obviously, any trailing stop has its unique set of parameters. Hence, any standardization of this set is not possible. On the other hand, all trailing algorithms have one required parameter, which is the position, which stop-loss order you want to modify. The position will be represented by the CPosition class, which is already familiar to us. Also, each type of trailing stop should have a method, during the call of which the stop-loss order of a position would be modified. This method is a kind of a "button" that launches the trailing algorithm, so the name of this method should be the same for all types of trailing.

We have identified two general entities for all types of trailing, which can be conveniently presented in the form of a special base class, call it **CTrailing**. It will contain methods for setting the current position, and the virtual Modify method, by which the stop-loss will be modified, as well as a special virtual Copy method, the purpose of which will be explained later:

```
//+------------------------------------------------------------------+
//|                                                     Trailing.mqh |
//|                                 Copyright 2016, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include <Object.mqh>
#include "..\PositionMT5.mqh"
#include "..\Logs.mqh"
class CPosition;
//+------------------------------------------------------------------+
//| The base class of a trailing stop                                |
//+------------------------------------------------------------------+
class CTrailing : public CObject
  {
protected:
   CPosition         *m_position;     // The position whose trailing stop you want to modify.
   CLog              *Log;
public:
                      CTrailing(void);
   void               SetPosition(CPosition *position);
   CPosition         *GetPosition(void);
   virtual bool       Modify(void);
   virtual CTrailing* Copy(void);
  };
//+------------------------------------------------------------------+
//| Constructor. Receives a logger module                            |
//+------------------------------------------------------------------+
CTrailing::CTrailing(void)
  {
   Log=CLog::GetLog();
  }
//+------------------------------------------------------------------+
//| Trailing stop modification method, which should be               |
//| overridden in the derived trailing class                         |
//+------------------------------------------------------------------+
bool CTrailing::Modify(void)
  {
   return false;
  }
//+------------------------------------------------------------------+
//| Returns a copy of the instance                                   |
//+------------------------------------------------------------------+
CTrailing* CTrailing::Copy(void)
{
   return new CTrailing();
}
//+------------------------------------------------------------------+
//| Sets a position, the stop loss of which should be modified       |
//+------------------------------------------------------------------+
void CTrailing::SetPosition(CPosition *position)
  {
   m_position=position;
  }
//+------------------------------------------------------------------+
//| Returns a position, the stop loss of which should be modified    |
//+------------------------------------------------------------------+
CPosition *CTrailing::GetPosition(void)
  {
   return m_position;
  }
//+------------------------------------------------------------------+
```

Any trailing stop will be derived from this class. The base trailing class does not contain parameters of specific stop-loss moving algorithms. This helps to achieve the maximum flexibility of the class operation. Despite the fact that trailing functions require various parameters for operation, the Modify method does not accept any of them. All parameters are set directly in the child trailing classes by using special methods. Thus, by the time of Modify call, all the required parameters are already known.

### Interaction of CTrailing with other modules of the strategy

We now only have a base CTrailing class, but this is enough in order to include it in the overall structure of the trading engine. We will add the base class inside the position class CPosition:

```
class CPosition
  {
public:
   CTrailing* Trailing;    // Trailing stop module
  };
```

This arrangement seems intuitive and natural. In this case, the position can be controlled the same as if it is controlled by means of the position itself:

```
void CMovingAverage::SupportBuy(const MarketEvent &event,CPosition *pos)
  {
   pos.Trailing.Modify();
  }
```

This is also possible due to the standardized Modify method, i.e. it is also known exactly what should be done in order to modify the position.

However, the integration of the trailing stop module does not end there. The above example still requires positions management at the level of the user strategy. We need to override the BuySupport and SellSupport methods and manage each position within the Expert Advisor logic. In order to further simplify position management, we cann add a trailing stop module directly into the CStrategy class:

```
class CStrategy
  {
public:
   CTrailing* Trailing;   // A trailing stop module for all positions
  };
```

We also need an additional CallSupport method that belongs to the CStrategy class:

```
//+------------------------------------------------------------------+
//| Calls the position management logic, provided that the trading   |
//| state isn't equal to TRADE_WAIT.                                 |
//+------------------------------------------------------------------+
void CStrategy::CallSupport(const MarketEvent &event)
  {
   m_trade_state=m_state.GetTradeState();
   if(m_trade_state == TRADE_WAIT)return;
   SpyEnvironment();
   for(int i=ActivePositions.Total()-1; i>=0; i--)
     {
      CPosition *pos=ActivePositions.At(i);
      if(pos.ExpertMagic()!=m_expert_magic)continue;
      if(pos.Symbol()!=ExpertSymbol())continue;
      if(CheckPointer(Trailing)!=POINTER_INVALID)
        {
         if(CheckPointer(Trailing)==POINTER_INVALID)
            pos.Trailing=Trailing.Copy();
         pos.Trailing.Modify();
         if(!pos.IsActive())
            continue;
        }
      if(pos.Direction()==POSITION_TYPE_BUY)
         SupportBuy(event,pos);
      else
         SupportSell(event,pos);
      if(m_trade_state==TRADE_STOP && pos.IsActive())
         ExitByStopRegim(pos);
     }
  }
```

New features are highlighted in yellow. They are very simple: if a trailing stop is placed by default, but the current position has none, the trailing stop is then placed for the current position, and then its stop-loss order is modified as per the trailing logic. However, we should take into account a very important feature. An _instance_ of a default trailing stop is assigned to each position, not the default trailing stop itself. This solution helps to avoid confusion inside the trailing stop logic. Imagine that the same instance of the trailing stop manages multiple positions. If it calculates any variables for one position inside its logic and stores them, they will be no longer valid during the next call, because the position passed for management will be different. This fact would cause very strange errors. In order to avoid this problem, an individual instance of a trailing stop is assigned to each position. This instance does not change during the entire lifetime of the position. In order to assign an individual trailing stop, it is necessary to _copy the default trailing stop_. Since CStrategy does not know what data and internal variables need to be copied, the copy procedure is performed by the end trailing class. It should override the virtual Copy() method of the base CTrailing class, and then return a reference that points to the created copy of itself in the form of a generic CTrailing class. Here is an example of implementation of a copying method for the classic trailing stop CTrailingClassic:

```
//+------------------------------------------------------------------+
//| Returns a copy of the instance                                   |
//+------------------------------------------------------------------+
CTrailing *CTrailingClassic::Copy(void)
  {
   CTrailingClassic *tral=new CTrailingClassic();
   tral.SetDiffExtremum(m_diff_extremum);
   tral.SetStepModify(m_step_modify);
   tral.SetPosition(m_position);
   return tral;
  }
```

The method creates an instance of the CTrailingClassic type, sets its parameters equal to the parameters of the current instance, and then returns an object in the form of a pointer to the CTrailing type.

Remember a simple rule when developing a custom trailing class:

In order to set a trailing stop by default, it is necessary to override the Copy method of the base CTrailing class. Otherwise, CStrategy would be unable to automatically manage open positions. If you plan to only use the trailing stop in the BuySupport and SellSupport methods, it is not necessary to override the virtual Copy method.

The requirement to override the Copy method complicates the development of a custom trailing stop, but makes the module behavior logic more safe, preventing the appearance of common data processing errors.

Now CStrategy can manage positions using the passed trailing stop as the managing logic. If we link the CStrategy::Trailing pointer to any trailing stop algorithm in a custom strategy constructor, it will become a _default trailing stop_ for all positions that will belong to the current Expert Advisor. So there is not need to override the BuySupport and SellSupport methods for the strategies that only use trailing to manage positions. The positions will be automatically managed on the CStrategy level.

Note that in the CallSupport code, the call of CTrailing::Modify is followed by a check of whether the position is active or not. This means that if a position were closed during the trailing stop modification process, it would not be a problem, while the search cycle would interrupt the call of overridden methods and would continue search with the next position. This feature causes an interesting result:

Any position management algorithm can actually be used as a trailing stop function. Modification of its stop-loss order is not necessary. Position management algorithm can close the position under certain conditions. This action is expected, and CStrategy would process normally.

### Practical use of a trailing stop. An example of a classical trailing stop

Now, when the base class is defined, and we have taught the basic strategy engine to interact with it, we can create a specific implementation of trailing stops. Let us begin with a classical trailing stop algorithm. Its operation is quite simple. A trailing stop moves the stop-loss order of a position following new highs (for a long position) and lows (for a short position). If the price goes back, the stop-loss remains at the same level. Thus, the stop-loss is trailed following the price at a certain distance from it. This distance is defined by the corresponding parameter.

Also, the trailing stop has one more parameter. It is optional. To avoid a too often modification of stop-loss, we will introduce an additional limitation: the minimum difference of a new stop-loss level from the previous level should be equal to **_StepModify_** points. The StepModify value is set in a separate parameter. This parameter is important for trading on FORTS. According to FORTS trading rules, the Exchange charges an additional fee for the so-called "inefficient transactions". If there are many modifications of stop loss, while there are few actual trades — the Exchange will require an additional fee from the trader. So the algorithms must take this feature into account.

Here is the code of our first trailing stop. It is based on the CTrailing class and overrides the Modify method:

```
//+------------------------------------------------------------------+
//|                                              TrailingClassic.mqh |
//|                                 Copyright 2016, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include "Trailing.mqh"
//+------------------------------------------------------------------+
//| Integrate trailing stop parameters in the list of parameters of  |
//| the Expert Advisor                                               |
//+------------------------------------------------------------------+
#ifdef SHOW_TRAILING_CLASSIC_PARAMS
input double PointsModify=0.00200;
input double StepModify=0.00005;
#endif
//+------------------------------------------------------------------+
//| A classical trailing stop                                        |
//+------------------------------------------------------------------+
class CTrailingClassic : public CTrailing
  {
private:
   double            m_diff_extremum;  // The distance in points from a reached extreme to the position's stop-loss
   double            m_step_modify;    // The minimum number of points to modify stop loss
   double            FindExtremum(CPosition *pos);
public:
                     CTrailingClassic(void);
   void              SetDiffExtremum(double points);
   double            GetDiffExtremum(void);
   void              SetStepModify(double points_step);
   double            GetStepModify(void);
   virtual bool      Modify(void);
   virtual CTrailing *Copy(void);
  };
//+------------------------------------------------------------------+
//| Constructor. Initializes default parameters                      |
//+------------------------------------------------------------------+
CTrailingClassic::CTrailingClassic(void) : m_diff_extremum(0.0),
                                           m_step_modify(0.0)
  {
#ifdef SHOW_TRAILING_CLASSIC_PARAMS
   m_diff_extremum=PointsModify;
   m_step_modify=StepModify;
#endif
  }
//+------------------------------------------------------------------+
//| Returns a copy of the instance                                   |
//+------------------------------------------------------------------+
CTrailing *CTrailingClassic::Copy(void)
  {
   CTrailingClassic *tral=new CTrailingClassic();
   tral.SetDiffExtremum(m_diff_extremum);
   tral.SetStepModify(m_step_modify);
   tral.SetPosition(m_position);
   return tral;
  }
//+------------------------------------------------------------------+
//| Sets the number of points from a reached extremum                |
//+------------------------------------------------------------------+
void CTrailingClassic::SetDiffExtremum(double points)
  {
   m_diff_extremum=points;
  }
//+------------------------------------------------------------------+
//| Sets the value of a minimal modification in points               |
//+------------------------------------------------------------------+
void CTrailingClassic::SetStepModify(double points_step)
  {
   m_step_modify=points_step;
  }
//+------------------------------------------------------------------+
//| Returns the number of points from a reached extremum             |
//+------------------------------------------------------------------+
double CTrailingClassic::GetDiffExtremum(void)
  {
   return m_diff_extremum;
  }
//+------------------------------------------------------------------+
//| Returns the value of a minimal modification in points            |
//+------------------------------------------------------------------+
double CTrailingClassic::GetStepModify(void)
  {
   return m_step_modify;
  }
//+------------------------------------------------------------------+
//| Modifies trailing stop in accordance with logic of a classical   |
//| trailing stop                                                    |
//+------------------------------------------------------------------+
bool CTrailingClassic::Modify(void)
  {

   if(CheckPointer(m_position)==POINTER_INVALID)
     {
      string text="Invalid position for current trailing-stop. Set position with 'SetPosition' method";
      CMessage *msg=new CMessage(MESSAGE_WARNING,__FUNCTION__,text);
      Log.AddMessage(msg);
      return false;
     }
   if(m_diff_extremum<=0.0)
     {
      string text="Set points trailing-stop with 'SetDiffExtremum' method";
      CMessage *msg=new CMessage(MESSAGE_WARNING,__FUNCTION__,text);
      Log.AddMessage(msg);
      return false;
     }
   double extremum=FindExtremum(m_position);
   if(extremum == 0.0)return false;
   double n_sl = 0.0;
   if(m_position.Direction()==POSITION_TYPE_BUY)
      n_sl=extremum-m_diff_extremum;
   else
      n_sl=extremum+m_diff_extremum;
   if(n_sl!=m_position.StopLossValue())
      return m_position.StopLossValue(n_sl);
   return false;
  }
//+------------------------------------------------------------------+
//| Returns the extreme price reached while holding the              |
//| position. For a long position, it will return the highest        |
//| reached price. For a short one - the lowest price that was       |
//| reached.                                                         |
//+------------------------------------------------------------------+
double CTrailingClassic::FindExtremum(CPosition *pos)
  {
   double prices[];
   if(pos.Direction()==POSITION_TYPE_BUY)
     {
      if(CopyHigh(pos.Symbol(),PERIOD_M1,pos.TimeOpen(),TimeCurrent(),prices)>1)
         return prices[ArrayMaximum(prices)];
     }
   else
     {
      if(CopyLow(pos.Symbol(),PERIOD_M1,pos.TimeOpen(),TimeCurrent(),prices)>1)
         return prices[ArrayMinimum(prices)];
     }
   return 0.0;
  }
//+------------------------------------------------------------------+
```

The basic code of the class is contained in the Modify and FindExtremum methods. The EA searches for the high or low of a price (depending on the position type) from the history using the FindExtremum method. Due to this, even after restarting the strategy or after its idle time, the stop-loss will be calculated properly.

Our trailing stop class contains additional yet obscure programming constructions in the form of the SHOW\_TRAILING\_CLASSIC\_PARAMS macro and a few input variables 'input'. We will discuss these constructions later in a separate section "Adding trailing parameters to the Expert Advisor settings".

### Adding a trailing stop to the CImpulse strategy

In the previous article " [Universal Expert Advisor: Use of Pending Orders and Hedging Support](https://www.mql5.com/en/articles/2404#c6)" we first introduced the CImpulse strategy. Its simple trading strategy is based on entries during sharp price movements. The proposed strategy applies position management based on the Moving Average values. The strategy closes a long position when a bar opens below the moving average. The strategy closes a short position when a bar opens above the moving average. Once again here is the code as described in the previous article, which implements this logic:

```
//+------------------------------------------------------------------+
//| Managing a long position in accordance with the Moving Average   |
//+------------------------------------------------------------------+
void CImpulse::SupportBuy(const MarketEvent &event,CPosition *pos)
{
   if(!IsTrackEvents(event))return;
   ENUM_ACCOUNT_MARGIN_MODE mode = (ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mode != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
   {
      double target = Bid() - Bid()*(m_percent/100.0);
      if(target < Moving.OutValue(0))
         pos.StopLossValue(target);
      else
         pos.StopLossValue(0.0);
   }
   if(Bid() < Moving.OutValue(0))
      pos.CloseAtMarket();
}
//+------------------------------------------------------------------+
//| Managing a short position in accordance with the Moving Average  |
//+------------------------------------------------------------------+
void CImpulse::SupportSell(const MarketEvent &event,CPosition *pos)
{
   if(!IsTrackEvents(event))return;
   ENUM_ACCOUNT_MARGIN_MODE mode = (ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mode != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
   {
      double target = Ask() + Ask()*(m_percent/100.0);
      if(target > Moving.OutValue(0))
         pos.StopLossValue(target);
      else
         pos.StopLossValue(0.0);
   }
   if(Ask() > Moving.OutValue(0))
      pos.CloseAtMarket();
}
```

Let us simplify it by using the classical trailing stop logic instead of the positions management rules. We will delete the methods from the code of the strategy, and will add a classical trailing stop as a default trailing function in the constructor of the strategy. Let's rename the strategy: CImpulseTrailingAuto:

```
//+------------------------------------------------------------------+
//| Strategy initialization and trailing stop configuration          |
//| at startup                                                       |
//+------------------------------------------------------------------+
CImpulseTrailing::CImpulseTrailing(void)
{
   CTrailingClassic* classic = new CTrailingClassic();
   classic.SetDiffExtremum(0.00100);
   Trailing = classic;
}
```

Now, according to the new logic, the position exit is based on the trailing stop, which is at a distance of 0.00100 points from a reached extreme price.

The full source code of the CImpulse strategy with an automatic trailing stop is available in _**ImpulseTrailingAuto.mqh**_.

### Adding trailing parameters to the Expert Advisor settings

The trailing stop operation method that we have created is very convenient. But we still need to configure the parameters of trailing stop within the custom strategy. We need to simplify the procedure: for example, somehow add trailing stop parameters to the Expert Advisor settings. The problem is that if the trailing stop is not used, the parameters still exist inside the Expert Advisor settings, which may cause ambiguity. In order to avoid this problem, we can use a _conditional compilation_. Let us add the trailing stop parameters and the special _conditional compilation macros_ SHOW\_TRAILING\_CLASSIC\_PARAMS to the module of a classical trailing stop:

```
//+------------------------------------------------------------------+
//|                                              TrailingClassic.mqh |
//|                                 Copyright 2016, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include "Trailing.mqh"
//+------------------------------------------------------------------+
//| Integrate trailing stop parameters in the list of parameters of  |
//| the Expert Advisor                                               |
//+------------------------------------------------------------------+
#ifdef SHOW_TRAILING_CLASSIC_PARAMS
input double PointsModify = 0.00100;
input double StepModify =   0.00005;
#endif
//+------------------------------------------------------------------+
//| A classical trailing stop                                        |
//+------------------------------------------------------------------+
class CTrailingClassic : public CTrailing
  {
   ...
public:
                     CTrailingClassic(void);
   ...
  };
//+------------------------------------------------------------------+
//| Constructor. Initializes default parameters                      |
//+------------------------------------------------------------------+
CTrailingClassic::CTrailingClassic(void) : m_diff_extremum(0.0),
                                           m_step_modify(0.0)
  {
   #ifdef SHOW_TRAILING_CLASSIC_PARAMS
   m_diff_extremum = PointsModify;
   m_step_modify = StepModify;
   #endif
  }
```

Now, if the SHOW\_TRAILING\_CLASSIC\_PARAMS macro is defined, trailing parameters will be integrated to the Expert Advisor settings during compilation:

![](https://c.mql5.com/2/24/2016-09-02_10h35_43.png)

Fig. 1. Dynamically linked parameters PointsModify and StepModify.

When the SHOW\_TRAILING\_CLASSIC\_PARAMS macro is commented or it is not available, trailing stop settings disappear from the EA parameters:

![](https://c.mql5.com/2/24/2016-09-02_10h26_58.png)

Fig 2. Disabled parameters of a trailing stop

The SHOW\_TRAILING\_CLASSIC\_PARAMS macro adds trailing parameters to the EA settings and additionally configures CTrailingClassic so that the configuration parameters are automatically added to it at the time of creation. Thus, when the strategy creates the macro, it already contains the parameters entered by the user through the Expert Advisor setup window.

### Moving Average based trailing stop

In the previous part of this article, the CImpulse strategy closed its positions if the price moved above or below the moving average. The moving average was presented in the form of the CIndMovingAverage indicator class. The CIndMovingAverage class is very similar to the trailing stop class. It calculates the values of the moving average and allows for the flexible configuration of the indicator parameters. The only difference from the trailing stop is the absence of a position management algorithm. The CIndMovingAverage class does not have the Modify() method. On the other hand, the base CTrailing class already includes all the required methods, but it has algorithms for working with the moving average. The composition method makes it possible to combine the advantages of each of these classes, and create a new type of trailing stop based on these classes: _a trailing stop based on the moving average_. The operation of the algorithm is very simple: it sets a stop-loss level of a position equal to the moving average. Let us also add an additional check to the Modify method: if the current price is below (when buying) or above (when selling) the calculated stop loss level, the position should be closed at current market prices. The entire class is available below:

```
//+------------------------------------------------------------------+
//|                                               TrailingMoving.mqh |
//|                                 Copyright 2016, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include "Trailing.mqh"
#include "..\Indicators\MovingAverage.mqh"

//+------------------------------------------------------------------+
//| Trailing stop based on MovingAverage.  Sets a stop loss of       |
//| a position equal to the MA level                                 |
//+------------------------------------------------------------------+
class CTrailingMoving : public CTrailing
{
public:
   virtual bool       Modify(void);
   CIndMovingAverage* Moving;
   virtual CTrailing* Copy(void);
};
//+------------------------------------------------------------------+
//| Sets the stop-loss of a position equal to the MA level           |
//+------------------------------------------------------------------+
bool CTrailingMoving::Modify(void)
{
   if(CheckPointer(Moving) == POINTER_INVALID)
      return false;
   double value = Moving.OutValue(1);
   if(m_position.Direction() == POSITION_TYPE_BUY &&
      value > m_position.CurrentPrice())
      m_position.CloseAtMarket();
   else if(m_position.Direction() == POSITION_TYPE_SELL &&
      value < m_position.CurrentPrice())
      m_position.CloseAtMarket();
   else if(m_position.StopLossValue() != value)
      return m_position.StopLossValue(value);
   return false;
}
//+------------------------------------------------------------------+
//| Returns an exact copy of the CTrailingMoving instance            |
//+------------------------------------------------------------------+
CTrailing* CTrailingMoving::Copy(void)
{
   CTrailingMoving* mov = new CTrailingMoving();
   mov.Moving = Moving;
   return mov;
}
```

The Modify function compares the moving average level with the current stop level. If the levels are not equal, the function places a new stop level for the position. The moving average value of the previous completed bar is used, because the current bar is always in the process of formation. Also note that the MovingAverage indicator is declared as a pointer. This feature allows the user to connect any object of the CIndMovingAverage type to this trailing stop.

Now let us test the operation of the class in the Strategy Tester. Here is a short video showing its operation:

Trailing stop by moving average - YouTube

[Photo image of Василий Соколов](https://www.youtube.com/channel/UCxbtFmFmwZRe-0q3SP9kgLw?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2411)

Василий Соколов

42 subscribers

[Trailing stop by moving average](https://www.youtube.com/watch?v=yrrH6HdLgyw)

Василий Соколов

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=yrrH6HdLgyw&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2411)

0:00

0:00 / 1:29

•Live

•

### Individual trailing stop for a position

We have analyzed the mechanism of trailing stop operation. Through the use of the unified virtual Modify method, the CStrategy trading engine can automatically set a trailing stop for each position and call its calculation algorithm. Usually this is enough, but in some cases we may need to manage each position individually. It means we may need to apply one type of trailing to one position and a different type to another position. Such trailing features cannot be unified or implemented on the trading engine side, therefore control of such trailing stops should be implemented inside the strategy. This can be done by overriding the BuySupport and SellSupport methods. Moreover, in this case there is no need to initialize a default trailing stop the way we did in the constructor of a custom strategy.

Suppose that the long positions of the CImpulse strategy should be managed using a trailing stop based on the Moving Average. A classical trailing stop should be applied to short positions. Both of these types of trailing stop have been described earlier. Let us override the BuySupport and SellSupport methods the following way:

```
//+------------------------------------------------------------------+
//| Managing a long position in accordance with the Moving Average   |
//+------------------------------------------------------------------+
void CImpulseTrailing::SupportBuy(const MarketEvent &event,CPosition *pos)
{
   if(!IsTrackEvents(event))
      return;
   if(pos.Trailing == NULL)
   {
      CTrailingMoving* trailing = new CTrailingMoving();
      trailing.Moving = GetPointer(this.Moving);
      pos.Trailing = trailing;
   }
   pos.Trailing.Modify();
}
//+------------------------------------------------------------------+
//| Managing a short position in accordance with the Moving Average  |
//+------------------------------------------------------------------+
void CImpulseTrailing::SupportSell(const MarketEvent &event,CPosition *pos)
{
   if(!IsTrackEvents(event))
      return;
   if(pos.Trailing == NULL)
   {
      CTrailingClassic* trailing = new CTrailingClassic();
      trailing.SetDiffExtremum(0.00100);
      pos.Trailing = trailing;
   }
   pos.Trailing.Modify();
}
```

Note that for the trailing stop based on the moving average, the CIndMovingAverage class is set as a parameter. This class is available in the strategy as a Moving object. In one line of code, we have instructed the trailing stop which object to use for the calculation of the stop-loss level.

In the SupportSell method, another trailing type is applied for new positions, and this trailing stop has its own set of parameters. The trailing uses a channel of 0.00100 pips form the extreme price.

The full code of the CImpulse strategy with individual trailing methods for each position type is available in the **ImpulseTrailingManual.mqh** file.

### A short list of fixes in the latest version

The CStrategy trading engine has been changed greatly since its first publication in the first part of the article. We have added new functions and modules that expand trading possibilities. Also, several compiler versions with various changes have been released since the first publication. Some of the changes were incompatible with the old versions of CStrategy, so we had to modify the trading engine. These corrections and extensions caused the appearance of inevitable errors. In this section, we will cover the bug fixes as of the latest version of the trading engine.

- A trading strategy panel included into the project. Due to a bug in the previous version of the compiler, the panel was disabled in parts 3 and 4 because of compatibility issues. After compiler fixes, the panel was added again in part 5, but it did not work properly. In the sixth edition of the trading engine, the operation of the panel is completely restored.
- The trading panel contained an error: instead of displaying "SellOnly" it displayed the "BuyOnly" mode twice. This bug is fixed.
- In previous versions, changing a trading mode on the panel would not change the actual trading mode of the strategy. The bug is fixed in the sixth version.
- A new behavior is added for the process of mode changing: in the SellOnly mode, all Buy positions are closed and additionally all pending Buy orders belonging to the strategy are deleted. The same applies to BuyOnly: all pending Sell orders are canceled. When you select the "Stop" mode, all pending orders of both direction are also deleted.

Please report any bugs found inn the engine operation. Detected bugs will be fixed.

### Conclusion

New functions have been added to the trading engine in the sixth part of the article. Now the trading engine supports trailing stops. Each trailing stop is a special class that contains the standardized Modify method that modifies the stop-loss level. It also contains specific data and parameters that configure the trailing moving algorithm. Trailing stop can be used in two ways.

- _**The trailing stop operation can be implemented in a trading engine of a strategy or the Autopilot mode can be enabled**_. In the latter case, a default trailing stop will be automatically applied to a position. No control from the custom strategy side is required in this case.
- **_Trailing stop management on the custom strategy side._** In this case the custom strategy manages its own positions using the trailing stop function. This mode allows implementing complex control logic — e.g. use different trailing algorithms for different position types.

Each trailing stop contains complete information about the position that it manages. Also, the position modification method can close this position at any time. This method provides for a high degree of flexibility for the trailing algorithms. Formally, any position control algorithm can act as the trailing stop algorithm. For example, instead of changing the stop-loss level, the algorithm can change the level of the take-profit order. Such changes do not break the trading logic, while the trading engine still operates properly.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2411](https://www.mql5.com/ru/articles/2411)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2411.zip "Download all attachments in the single ZIP archive")

[article\_13.05.2016.zip](https://www.mql5.com/en/articles/download/2411/article_13.05.2016.zip "Download article_13.05.2016.zip")(113.88 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing graphical interfaces based on .Net Framework and C# (part 2): Additional graphical elements](https://www.mql5.com/en/articles/6549)
- [Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://www.mql5.com/en/articles/5563)
- [Custom Strategy Tester based on fast mathematical calculations](https://www.mql5.com/en/articles/4226)
- [R-squared as an estimation of quality of the strategy balance curve](https://www.mql5.com/en/articles/2358)
- [Universal Expert Advisor: CUnIndicator and Use of Pending Orders (Part 9)](https://www.mql5.com/en/articles/2653)
- [Implementing a Scalping Market Depth Using the CGraphic Library](https://www.mql5.com/en/articles/3336)
- [Universal Expert Advisor: Accessing Symbol Properties (Part 8)](https://www.mql5.com/en/articles/3270)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/88636)**
(10)


![igorbel](https://c.mql5.com/avatar/avatar_na2.png)

**[igorbel](https://www.mql5.com/en/users/igorbel)**
\|
4 Jun 2016 at 21:26

I didn't change anything, but when compiling any of the trailing modules, including the base class, I get errors:

```
'CTrailing' - declaration without type  PositionMT5.mqh 48      4
'Trailing' - undeclared identifier      PositionMT5.mqh 73      20
'Trailing' - object pointer expected    PositionMT5.mqh 73      20
'Trailing' - object pointer expected    PositionMT5.mqh 74      14
```

![igorbel](https://c.mql5.com/avatar/avatar_na2.png)

**[igorbel](https://www.mql5.com/en/users/igorbel)**
\|
19 Sep 2016 at 19:19

```
bool CTrailingClassic::Modify(void)
  {

   if(CheckPointer(m_position)==POINTER_INVALID)
     {
      string text="Invalid position for current trailing-stop. Set position with 'SetPosition' method";
      CMessage *msg=new CMessage(MESSAGE_WARNING,__FUNCTION__,text);
      Log.AddMessage(msg);
      return false;
     }
   if(m_diff_extremum<=0.0)
     {
      string text="Set points trailing-stop with 'SetDiffExtremum' method";
      CMessage *msg=new CMessage(MESSAGE_WARNING,__FUNCTION__,text);
      Log.AddMessage(msg);
      return false;
     }
   double extremum=FindExtremum(m_position);
   if(extremum == 0.0)return false;
   double n_sl = 0.0;
   if(m_position.Direction()==POSITION_TYPE_BUY)
      n_sl=extremum-m_diff_extremum;
   else
      n_sl=extremum+m_diff_extremum;
   if(n_sl!=m_position.StopLossValue())
      return m_position.StopLossValue(n_sl);
   return false;
  }
```

It wouldn't hurt to check that the new sl is below the [current price](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_double "MQL5 Documentation: Order Properties") for a long position and above the current price for a short position.

![Treblatina](https://c.mql5.com/avatar/avatar_na2.png)

**[Treblatina](https://www.mql5.com/en/users/treblatina)**
\|
29 Dec 2020 at 15:53

Hello. Thanks for the article.

How can I adjust the lot size for each trade? I see it buys one lot only

Thanks in advance

![AK377MC](https://c.mql5.com/avatar/avatar_na2.png)

**[AK377MC](https://www.mql5.com/en/users/ak377mc)**
\|
9 Jun 2022 at 12:32

Dear Mr Sokolov,

Very interesting article, but unfortunately I couldn't install a single one of your codes without tons of compiller errors. I tried all 9 parts but in vain.

Hence my question: Is there a specific approach to install your code?

Thank You

![Johann Kern](https://c.mql5.com/avatar/2023/11/6544a3f7-8f2d.png)

**[Johann Kern](https://www.mql5.com/en/users/joosy)**
\|
17 Mar 2023 at 05:08

Brevity is the soul of wit.

Acceptance, understanding, correct interpretation, there are hardly any more. Unfortunately

![Creating an assistant in manual trading](https://c.mql5.com/2/23/panel__1.png)[Creating an assistant in manual trading](https://www.mql5.com/en/articles/2281)

The number of trading robots used on the currency markets has significantly increased recently. They employ various concepts and strategies, however, none of them has yet succeeded to create a win-win sample of artificial intelligence. Therefore, many traders remain committed to manual trading. But even for such specialists, robotic assistants or, so called, trading panels, are created. This article is yet another example of creating a trading panel from scratch.

![Graphical Interfaces VI: the Slider and the Dual Slider Controls (Chapter 2)](https://c.mql5.com/2/23/avad1j__1.png)[Graphical Interfaces VI: the Slider and the Dual Slider Controls (Chapter 2)](https://www.mql5.com/en/articles/2468)

In the previous article, we have enriched our library with four controls frequently used in graphical interfaces: checkbox, edit, edit with checkbox and check combobox. The second chapter of the sixth part will be dedicated to the slider and the dual slider controls.

![How to create bots for Telegram in MQL5](https://c.mql5.com/2/22/telegram-avatar.png)[How to create bots for Telegram in MQL5](https://www.mql5.com/en/articles/2355)

This article contains step-by-step instructions for creating bots for Telegram in MQL5. This information may prove useful for users who wish to synchronize their trading robot with a mobile device. There are samples of bots in the article that provide trading signals, search for information on websites, send information about the account balance, quotes and screenshots of charts to you smart phone.

![Self-optimization of EA: Evolutionary and genetic algorithms](https://c.mql5.com/2/22/images__2.png)[Self-optimization of EA: Evolutionary and genetic algorithms](https://www.mql5.com/en/articles/2225)

This article covers the main principles set fourth in evolutionary algorithms, their variety and features. We will conduct an experiment with a simple Expert Advisor used as an example to show how our trading system benefits from optimization. We will consider software programs that implement genetic, evolutionary and other types of optimization, and provide examples of application when optimizing a predictor set and parameters of the trading system.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=cmqpgwsanbfozxdsuvqyolfivpzttqyh&ssn=1769192329508698626&ssn_dr=0&ssn_sr=0&fv_date=1769192329&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2411&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Universal%20Expert%20Advisor%3A%20A%20Custom%20Trailing%20Stop%20(Part%206)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691923299451234&fz_uniq=5071739469297560969&sv=2552)

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