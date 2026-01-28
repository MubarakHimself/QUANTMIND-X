---
title: Developing multi-module Expert Advisors
url: https://www.mql5.com/en/articles/3133
categories: Trading, Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:28:41.791261
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/3133&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062496570827514679)

MetaTrader 5 / Trading


### Introduction

Currently, there are several approaches to programming: modular, object-oriented and structured. In this article, we will discuss modular programming with respect to trading robots.

Modular programming is a program development method that involves splitting the program into independent modules.

The main principle of the modular programming is " _divide and rule_". The convenience of using the modular architecture is the ability to update (replace) the module without the need to change the rest of the system.

Three basic concepts lie at the heart of the modular programming.

- **Parnas' principle of withholding information**. The module is used to hide information and form the algorithm for solving a specific problem. The module can be replaced by another one later.
- **Cohen's modularity axiom**. The module is an independent program unit for performing a certain function of the program.
- **Tseytin's assembly programming**. Modules are "bricks" the program consists of.

The only alternative to modularity is a monolithic program. It is not very convenient though. If you need to change or supplement some of the program functions, you need to edit the code of the Expert Advisor (EA), which in most cases can be done either by the code author or by another experienced programmer. Besides, if the monolithic program is compiled, it can be edited only by the copyright owner. It would be much more convenient to change important program functions on your own or hire third-party developers.

![Fig. 1. Modular trading robot](https://c.mql5.com/2/33/diagram-1__2.png)

Fig. 1. Abstract diagram of a modular trading robot

### Multi-modularity principle

Modular programming is the art of splitting a task into a number of sub-tasks implemented as separate modules (files). In general, a program module is a separate program, or a functionally complete and autonomously compiled program unit somehow identified and combined with the called module. In other words, a module is a functionally finished fragment of the program, designed as a separate compiled file developed for use in other programs.

When determining the set of modules that implement the functions of a particular algorithm, the following should be considered:

- each module is called for execution by a parent module and, upon completion of its work, returns control to the module that called it;
- main decisions in the algorithm are made at the highest level in the hierarchy;
- modules are independent from each other regarding the data;
- modules do not depend on the history of accessing them.

Summarizing all of the above, a modular program is a program, in which any part of the logical structure can be changed without causing changes in other parts.

Main module parameters:

- **one input and one output** — at the input, the program module receives a certain set of initial data, processes them and returns one set of resulting data, thus implementing the IPO principle (Input-Process-Output);
- **functional completeness** — to perform a separate function, the module performs a complete list of regulated operations, sufficient to complete the processing that has been started;
- **logical independence** — result of the program module depends only on the source data. It does not depend on the operation of other modules;
- **weak data links with other program modules** — data exchange between the modules should be as minimized as possible.

MQL5 language allows developing three types of programs: an EA, an indicator or a script. An EA managing all modules and featuring trading functions is best for the main module. Other modules can be implemented, for example, as indicators. The indicators are perfect for forming a module: the data calculated with the given algorithm can be stored in the indicator buffers and passed to the multi-module EA if necessary. In turn, the EA can use or ignore these data depending on a task. In some projects, the use of EAs as external modules is justified, but at the same time, it is necessary to think over the mechanism of data exchange in details.

Many of you have surely applied modular technologies in your EAs: for example, [custom indicators](https://www.mql5.com/en/docs/indicators/icustom) as modules for generating and sorting out trading signals.

The most rational solution, in my opinion, looks like this: all the basic functionality is concentrated in the main module and does not require the participation of external ones. In turn, external modules are needed to adapt to different market conditions and improve the trading strategy. The set of program functions is determined by a user rather than a code or strategy developer. It is important to note that no one violates the each other's legitimate rights.

### Main module — EA

The main module, where the management of the entire project is located, is the most important one in the EA hierarchy. It should contain trading functions. Without them, any trading strategy is meaningless.

Let's consider developing a multi-module EA using a specific [example](https://www.mql5.com/en/code/17992) from CodeBase. The initial EA trades a fixed lot in the [iBands](https://www.mql5.com/en/docs/indicators/ibands) indicator channel with a position reversal on the channel borders. The EA is completely self-sufficient and does not require any external programs.

Not every EA can be multi-module.

What should we add to the code to turn it into a module project?

1. Declare external modules (indicators) a user can subsequently apply.
2. Add the necessary functionality for their integration.
3. Prepare documentation for external module developers (enable the documentation generation function in a separate file). Info on data structure that can be correctly used by the main module may be needed for developing external modules. For example, in this example, the money management module should pass the lot size to the EA, while the position tracking module should pass the distance from the current price in points.

As a result of the transformation, we obtain a modular EA allowing the integration of up to seven external modules.

- Module 1 — money management module. It provides a lot size.
- Module 2 — tracking positions and placing SL. It provides a distance to SL in points from the Open price.
- Module 3 — tracking positions and placing TP. It provides a distance to TP in points from the Open price.
- Module 4 — tracking positions and placing a trailing stop. It provides a distance to SL in points from the current price.
- Module 5 — trading signals generation. It provides a signal value.
- Module 6 — module for sorting out trading signals. It provides the filter value.
- Module 7 — tracking positions and placing a breakeven level. It provides a distance to SL from the Open price.


![Fig. 2. OnInit() function and initializing external modules](https://c.mql5.com/2/33/diagram-2__1.png)

Fig. 2. OnInit() function and initializing external modules

![Fig. 3. OnTick() function and reading data from external modules](https://c.mql5.com/2/33/diagram-3__1.png)

Fig. 3. OnTick() function and reading data from external modules

![Fig. 4. OnTrade() function and reading data from external modules](https://c.mql5.com/2/33/diagram-4__1.png)

Fig. 4. OnTrade() function and reading data from external modules

![Fig. 5. The function for generating trading signals and reading data from external modules](https://c.mql5.com/2/33/diagram-5.png)

Fig. 5. The function for generating trading signals and reading data from external modules

```
//****** project (module expert): test_module_exp.mq5
//+------------------------------------------------------------------+
//|          The program code is generated Modular project generator |
//|                      Copyright 2010-2017, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010-2017, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project TEST Main module."
#property link      "The project uses 7 external modules."
//---
#include <Trade\Trade.mqh>
//---
MqlTick    last_tick;
CTrade     trade;
//---
input int                  e_bands_period=80;            // Moving average period
int                        e_bands_shift=0;              // shift
input double               e_deviation=3.0;              // Number of standard deviations
input ENUM_APPLIED_PRICE   e_applied_price=PRICE_CLOSE;  // Price type
input bool                 on_module=false;              // whether or not to use plug-ins
//---
double lot=0.01;           // Fixed lot
double min_lot=0.01;       // Minimum allowable lot
bool   on_trade=false;     // Trade function flag
//--- Variable for storing the indicator iBands handle
int    handle_Bands;
//--- module 1
bool   on_lot=false;
int    handle_m1;
//--- module 2
bool   on_SL=false;
int    handle_m2;
//--- module 3
bool   on_TP=false;
int    handle_m3;
//--- module 4
bool   on_Trail=false;
int    handle_m4;
//--- module 5
bool   on_signals=false;
int    handle_m5;
//--- module 6
bool   on_Filter=false;
int    handle_m6;
//--- module 7
bool   on_Breakeven=false;
int    handle_m7;
//+------------------------------------------------------------------+
//| Structure of trading signals                                     |
//+------------------------------------------------------------------+
struct sSignal
  {
   bool              Buy;    // Buy signal
   bool              Sell;   // Sell signal
  };
//+------------------------------------------------------------------+
//| Trading signals generator                                        |
//+------------------------------------------------------------------+
sSignal Buy_or_Sell()
  {
   sSignal res={false,false};
//--- MODULE 5
   if(on_signals)
     { // If there is an additional module
      double buffer_m5[];
      ArraySetAsSeries(buffer_m5,true);
      if(CopyBuffer(handle_m5,0,0,1,buffer_m5)<0) return(res);
      if(buffer_m5[0]<-1) res.Sell=true;
      if(buffer_m5[0]>1) res.Buy=true;
     }
//--- MODULE 6
   if(on_Filter)
     { // If there is an additional module
      double buffer_m6[];
      ArraySetAsSeries(buffer_m6,true);
      if(CopyBuffer(handle_m6,0,0,1,buffer_m6)<0) return(res);
      lot=buffer_m6[0];
      if(buffer_m6[0]<1) res.Buy=false;
      if(buffer_m6[0]>-1) res.Sell=false;
     }
//---
//--- Indicator buffers
   double         UpperBuffer[];
   double         LowerBuffer[];
   double         MiddleBuffer[];
   ArraySetAsSeries(MiddleBuffer,true); CopyBuffer(handle_Bands,0,0,1,MiddleBuffer);
   ArraySetAsSeries(UpperBuffer,true);  CopyBuffer(handle_Bands,1,0,1,UpperBuffer);
   ArraySetAsSeries(LowerBuffer,true);  CopyBuffer(handle_Bands,2,0,1,LowerBuffer);
//--- Timeseries
   double L[];
   double H[];
   ArraySetAsSeries(L,true); CopyLow(_Symbol,_Period,0,1,L);
   ArraySetAsSeries(H,true); CopyHigh(_Symbol,_Period,0,1,H);
   if(H[0]>UpperBuffer[0]&& L[0]>MiddleBuffer[0]) res.Sell=true;
   if(L[0]<LowerBuffer[0] && H[0]<MiddleBuffer[0]) res.Buy=true;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Create the indicator handle
   handle_Bands=iBands(_Symbol,_Period,e_bands_period,e_bands_shift,e_deviation,e_applied_price);
   if(handle_Bands==INVALID_HANDLE)
      return(INIT_FAILED);
   else
      on_trade=true;
   if(on_module)
     {
      //--- MODULE 1
      //--- check: whether there is an external module?
      handle_m1=iCustom(NULL,0,"Market\\test_module_MM");
      if(handle_m1!=INVALID_HANDLE)
         on_lot=true;
      //--- MODULE 2
      //--- check: whether there is an external module?
      handle_m2=iCustom(NULL,0,"Market\\test_module_SL");
      if(handle_m2!=INVALID_HANDLE)
         on_SL=true;
      //--- MODULE 3
      //--- check: whether there is an external moduleь?
      handle_m3=iCustom(NULL,0,"Market\\test_module_TP");
      if(handle_m3!=INVALID_HANDLE)
         on_TP=true;
      //--- MODULE 4
      //--- check: whether there is an external module?
      handle_m4=iCustom(NULL,0,"Market\\test_module_Trail");
      if(handle_m4!=INVALID_HANDLE)
         on_Trail=true;
      //--- MODULE 5
      //--- check: whether there is an external module?
      handle_m5=iCustom(NULL,0,"Market\\test_module_signals");
      if(handle_m5!=INVALID_HANDLE)
         on_signals=true;
      //--- MODULE 6
      //--- check: whether there is an external module?
      handle_m6=iCustom(NULL,0,"Market\\test_module_Filter");
      if(handle_m6!=INVALID_HANDLE)
         on_Filter=true;
      //--- MODULE 7
      //--- check: whether there is an external module?
      handle_m7=iCustom(NULL,0,"Market\\test_module_Breakeven");
      if(handle_m7!=INVALID_HANDLE)
         on_Breakeven=true;
     }
//--- Minimum allowable volume for trading operationsn
   min_lot=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   double Equity=AccountInfoDouble(ACCOUNT_EQUITY);
//--- MODULE 1
   if(on_lot)
     { // If there is an additional module
      double buffer_m1[];
      ArraySetAsSeries(buffer_m1,true);
      if(CopyBuffer(handle_m1,0,0,1,buffer_m1)<0) return;
      lot=buffer_m1[0];
     }
//--- MODULE 4
   if(on_Trail)
      if(PositionSelect(_Symbol))
         if(PositionGetDouble(POSITION_PROFIT)>0)
           { // If there is an additional module
            double buffer_m4[];
            ArraySetAsSeries(buffer_m4,true);
            if(CopyBuffer(handle_m4,0,0,1,buffer_m4)<0) return;
            double TR=buffer_m4[0];
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
               if(PositionGetDouble(POSITION_PRICE_CURRENT)-TR*_Point>PositionGetDouble(POSITION_PRICE_OPEN))
                 {
                  double price_SL=PositionGetDouble(POSITION_PRICE_CURRENT)-TR*_Point;
                  if(price_SL>PositionGetDouble(POSITION_SL))
                     trade.PositionModify(_Symbol,NormalizeDouble(price_SL,Digits()),0);
                 }
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
               if(PositionGetDouble(POSITION_PRICE_CURRENT)+TR*_Point<PositionGetDouble(POSITION_PRICE_OPEN))
                 {
                  double price_SL=PositionGetDouble(POSITION_PRICE_CURRENT)+TR*_Point;
                  if(price_SL<PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL)==NULL)
                     trade.PositionModify(_Symbol,NormalizeDouble(price_SL,Digits()),0);
                 }
           }
//--- MODULE 7
   if(on_Breakeven)
      if(PositionSelect(_Symbol))
         if(PositionGetDouble(POSITION_PROFIT)>0)
           { // If there is an additional module
            double buffer_m7[];
            ArraySetAsSeries(buffer_m7,true);
            if(CopyBuffer(handle_m7,0,0,1,buffer_m7)<0) return;
            double TRB=buffer_m7[0];
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
               if(PositionGetDouble(POSITION_PRICE_CURRENT)-TRB*_Point>PositionGetDouble(POSITION_PRICE_OPEN))
                 {
                  double price_SL=PositionGetDouble(POSITION_PRICE_CURRENT)-5*_Point;
                  if(price_SL>PositionGetDouble(POSITION_SL))
                     trade.PositionModify(_Symbol,NormalizeDouble(price_SL,Digits()),PositionGetDouble(POSITION_TP));
                 }
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
               if(PositionGetDouble(POSITION_PRICE_CURRENT)+TRB*_Point<PositionGetDouble(POSITION_PRICE_OPEN))
                 {
                  double price_SL=PositionGetDouble(POSITION_PRICE_CURRENT)+5*_Point;
                  if(price_SL<PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL)==NULL)
                     trade.PositionModify(_Symbol,NormalizeDouble(price_SL,Digits()),PositionGetDouble(POSITION_TP));
                 }
           }
//---
   if(lot<min_lot) lot=min_lot;
//---
   if(on_trade)
     {
      sSignal signal=Buy_or_Sell();
      //--- The value of the required and free margin
      double margin,free_margin=AccountInfoDouble(ACCOUNT_MARGIN_FREE);
      //--- BUY
      if(signal.Buy)
        {
         if(!PositionSelect(_Symbol))
           {
            SymbolInfoTick(_Symbol,last_tick);
            if(OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,NormalizeDouble(lot,2),last_tick.ask,margin))
               if(margin<Equity)
                  trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,NormalizeDouble(lot,2),last_tick.ask,0,0,"BUY: new position");
           }
         else
           {
            if(PositionGetDouble(POSITION_PROFIT)<0) return;
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
              {
               trade.PositionClose(_Symbol);
               SymbolInfoTick(_Symbol,last_tick);
               if(OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,NormalizeDouble(lot,2),last_tick.ask,margin))
                  if(margin<Equity)
                     trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,NormalizeDouble(lot,2),last_tick.ask,0,0,"BUY: reversal");
              }
           }
        }
      //--- SELL
      if(signal.Sell)
        {
         if(!PositionSelect(_Symbol))
           {
            SymbolInfoTick(_Symbol,last_tick);
            if(OrderCalcMargin(ORDER_TYPE_SELL,_Symbol,NormalizeDouble(lot,2),last_tick.bid,margin))
               if(margin<Equity)
                  trade.PositionOpen(_Symbol,ORDER_TYPE_SELL,NormalizeDouble(lot,2),last_tick.bid,0,0,"SELL: new position");
           }
         else
           {
            if(PositionGetDouble(POSITION_PROFIT)<0) return;
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
              {
               trade.PositionClose(_Symbol);
               SymbolInfoTick(_Symbol,last_tick);
               if(OrderCalcMargin(ORDER_TYPE_SELL,_Symbol,NormalizeDouble(lot,2),last_tick.bid,margin))
                  if(margin<Equity)
                     trade.PositionOpen(_Symbol,ORDER_TYPE_SELL,NormalizeDouble(lot,2),last_tick.bid,0,0,"SELL: reversal");
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
   if(on_SL && on_TP) // If there is an additional module
     {
      //--- MODULE 2
      double buffer_m2[];
      ArraySetAsSeries(buffer_m2,true);
      if(CopyBuffer(handle_m2,0,0,1,buffer_m2)<0) return;
      double SL=buffer_m2[0];
      //--- MODULE 3
      double buffer_m3[];
      ArraySetAsSeries(buffer_m3,true);
      if(CopyBuffer(handle_m3,0,0,1,buffer_m3)<0) return;
      double TP=buffer_m3[0];
      //--- Position modification
      if(PositionSelect(_Symbol))
         if(PositionGetDouble(POSITION_SL)==0)
           {
            //--- BUY
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
              {
               double priceTP=PositionGetDouble(POSITION_PRICE_OPEN)+TP*_Point;
               double priceSL=PositionGetDouble(POSITION_PRICE_OPEN)-SL*_Point;
               trade.PositionModify(_Symbol,NormalizeDouble(priceSL,Digits()),NormalizeDouble(priceTP,Digits()));
              }
            //--- SELL
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
              {
               double priceTP=PositionGetDouble(POSITION_PRICE_OPEN)-TP*_Point;
               double priceSL=PositionGetDouble(POSITION_PRICE_OPEN)+SL*_Point;
               trade.PositionModify(_Symbol,NormalizeDouble(priceSL,Digits()),NormalizeDouble(priceTP,Digits()));
              }
           }
     }
  }
```

If the directory set in the program contains no necessary modules (files), a trading robot applies the default functionality. Thus, the absence of external modules does not have a critical impact on the EA performance.

### The most important modules

How to make a multi-module EA from a monolith program? The modular project begins with the analysis of a common task and the defining of functionally closed fragments, which can later be formalized as compiled modules. In this case, you need to select the most typical functions that can significantly change the work of the Expert Advisor and are based on various algorithms. It is well-known that most EAs apply the same procedures:

- money (risk) management module;
- position tracking module (SL and TP);
- trailing stop module;
- trading signals generation module;
- signals filtration module.

There are many options for implementing each of the listed modules. In this article, we will show you the most simple solutions, since the method of modular programming is more important for us here than the multi-string functionality.

### Developing auxiliary modules

An auxiliary (external) module is an indicator that performs a certain function and places input data to indicator buffers. If necessary, the main module uses these data. Thus, the EA adapts to the requirements of traders applying this trading strategy. The same initial EA can be re-assembled for each specific financial instrument or broker. In fact, this is a toolkit allowing traders to assemble an unlimited number of trading robots.

Programming is a labor-consuming process. Despite its creative nature, it contains a lot of mundane operations you might want to automate. Among other things, automation boosts productivity and reduces errors.

The module generator attached below allows you to form up to eight files connected into one multi-module project in a few seconds. This greatly simplifies and speeds up the development and assembly (see the video).

![The panel for managing the multi-module projects generator](https://c.mql5.com/2/27/Demo_1_0.gif)

Video 1. The panel for managing the multi-module projects generator

The panel allows you to set specific modules to be generated for a project. In our example, a "test" project is created. Depending on the selected combination of modules, the generator automatically generates code without unnecessary blocks and files.

The generated files are placed to the Files folder (see Fig. 2). The name of the "test\_module\_exp.mq5" main module consists of the project name ("test") and the "\_module\_exp.mq5" prefix. Place it to the Experts folder, while the remaining files of the external modules should be located in Indicators\\Market.

![Fig. 6. The generated files of the "test" project](https://c.mql5.com/2/27/1__2.png)

Fig. 6. The generated files of the "test" project

After that, compile all the files and proceed testing the multi-module project.

![Video 2. Compiling generated files of the 'test' project](https://c.mql5.com/2/27/Demo_1_4.gif)

Video 2. Compiling generated files of the "test" project

Creating a similar project manually takes much time. Obviously, we start from the main module. After defining the external modules that can later be connected to the project, proceed to development and programming. The main thing to track is the output data of the auxiliary modules that the main module is waiting for. Since the modules are indicators, and [indicator buffers](https://www.mql5.com/en/docs/series/copybuffer) contain values ​​of exclusively real type, then in the main module, it is necessary to provide the conversion of variables from the real type to the type corresponding to the algorithm.

External modules should be designed so that they can be called in the main module without inputs, i.e. by default. Such a call mechanism simplifies the development of the external data management system.

Let us consider in more detail which external modules are the most important in trading strategies.

### Example 1: Money management module

This external module calculates the lot volume to open orders. Below you can see the simplest method of implementing a trading volume calculation (in % of available funds on the deposit):

```
//****** project (module MM): test_module_MM_ind.mq5
//+------------------------------------------------------------------+
//|          The program code is generated Modular project generator |
//|                      Copyright 2010-2017, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010-2017, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project test module MM"
//--- Display indicator in the chart window
#property indicator_chart_window
//--- Number of buffers to calculate the indicator
#property indicator_buffers 1
//--- Number of graphic series in the indicator
#property indicator_plots   1
//---
input double   lot_perc=0.1;  // Percentage of Equity value
double         Buffer1[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ArraySetAsSeries(Buffer1,true);
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   return(INIT_SUCCEEDED);
  };
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate (const int rates_total,
                 const int prev_calculated,
                 const datetime& time[],
                 const double& open[],
                 const double& high[],
                 const double& low[],
                 const double& close[],
                 const long& tick_volume[],
                 const long& volume[],
                 const int &spread[])
  {
   double Equity=AccountInfoDouble(ACCOUNT_EQUITY);
//--- calculation of the lot of equity
   Buffer1[0]=NormalizeDouble(Equity*lot_perc/1000.0,2); // Lot size determination function
   if(Buffer1[0]<0.01) Buffer1[0]=0.01;
   return(rates_total);
  };
```

If we call this module by default (without setting the values of input parameters), the main module gets the size of the allowed lot to perform a trade in the amount of 0.1% of the available funds. Example of calling this module from the main program:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   double Equity=AccountInfoDouble(ACCOUNT_EQUITY);
//--- MODULE 1
   if(on_lot)
     { // If there is an additional module
      double buffer_m1[];
      ArraySetAsSeries(buffer_m1,true);
      if(CopyBuffer(handle_m1,0,0,1,buffer_m1)<0) return;
      lot=buffer_m1[0];
     }

  ...

  }
```

### Example 2: Position tracking module (SL, TP and trailing)

Placing a stop loss (SL) and take profit (TP) is one of the ways to track an open position. Since different trading strategies apply various combinations of calculating and placing SL and TP, the option of splitting into two modules (for SL and TP) seems to be the most practical here. But if we still decide to combine SL and TP in a single module, their values should be placed in different [indicator buffers](https://www.mql5.com/en/docs/series/copybuffer).

**SL placement module**:

```
//****** project (module SL): test_module_SL_ind.mq5
//+------------------------------------------------------------------+
//|          The program code is generated Modular project generator |
//|                      Copyright 2010-2017, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010-2017, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project test module SL"
//--- Display indicator in the chart window
#property indicator_chart_window
//--- Number of buffers to calculate the indicator
#property indicator_buffers 1
//--- Number of graphic series in the indicator
#property indicator_plots   1
//---
double      Buffer1[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ArraySetAsSeries(Buffer1,true);
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   return(INIT_SUCCEEDED);
  };
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate (const int rates_total,
                 const int prev_calculated,
                 const datetime& time[],
                 const double& open[],
                 const double& high[],
                 const double& low[],
                 const double& close[],
                 const long& tick_volume[],
                 const long& volume[],
                 const int &spread[])
  {
   double SL=100; // SL in points
//--- calculation of the SL
   Buffer1[0]=SL;
   return(rates_total);
  };
```

**TP placement module**:

```
//****** project (module TP): test_module_TP_ind.mq5
//+------------------------------------------------------------------+
//|          The program code is generated Modular project generator |
//|                      Copyright 2010-2017, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010-2017, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project test module TP"
//--- Display indicator in the chart window
#property indicator_chart_window
//--- Number of buffers to calculate the indicator
#property indicator_buffers 1
//--- Number of graphic series in the indicator
#property indicator_plots   1
//---
double      Buffer1[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ArraySetAsSeries(Buffer1,true);
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   return(INIT_SUCCEEDED);
  };
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate (const int rates_total,
                 const int prev_calculated,
                 const datetime& time[],
                 const double& open[],
                 const double& high[],
                 const double& low[],
                 const double& close[],
                 const long& tick_volume[],
                 const long& volume[],
                 const int &spread[])
  {
   double TP=100; // TP in points
//--- calculation of the TP
   Buffer1[0]=TP;
   return(rates_total);
  };
```

The codes show the most obvious option for calculating SL and TP values — in points. In fact, they are not calculated but are rather set by a constant. The values are specified directly in the program rather than in the inputs. This is done to demonstrate the implementation of external modules without inputs. Any novice programmer can write such a "rough" code.

I believe, the [OnTrade](https://www.mql5.com/en/docs/basis/function/events#ontrade) function is the best place for storing the call of the modules described above. This approximately looks as follows:

```
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
   if(on_SL && on_TP) // If there is an additional module
     {
      //--- MODULE 2
      double buffer_m2[];
      ArraySetAsSeries(buffer_m2,true);
      if(CopyBuffer(handle_m2,0,0,1,buffer_m2)<0) return;
      double SL=buffer_m2[0];
      //--- MODULE 3
      double buffer_m3[];
      ArraySetAsSeries(buffer_m3,true);
      if(CopyBuffer(handle_m3,0,0,1,buffer_m3)<0) return;
      double TP=buffer_m3[0];
      //--- Position modification
      if(PositionSelect(_Symbol))
         if(PositionGetDouble(POSITION_SL)==0)
           {
            //--- BUY
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
              {
               double priceTP=PositionGetDouble(POSITION_PRICE_OPEN)+TP*_Point;
               double priceSL=PositionGetDouble(POSITION_PRICE_OPEN)-SL*_Point;
               trade.PositionModify(_Symbol,NormalizeDouble(priceSL,Digits()),NormalizeDouble(priceTP,Digits()));
              }
            //--- SELL
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
              {
               double priceTP=PositionGetDouble(POSITION_PRICE_OPEN)-TP*_Point;
               double priceSL=PositionGetDouble(POSITION_PRICE_OPEN)+SL*_Point;
               trade.PositionModify(_Symbol,NormalizeDouble(priceSL,Digits()),NormalizeDouble(priceTP,Digits()));
              }
           }
     }
  }
```

Apart from static SL and TP values set right after opening a position, a trailing stop, or a floating SL, is often applied. Most often, it is set after a position becomes profitable. Let's have a look at the most obvious implementation: we set the distance of SL from the current price in points.

```
//****** project (module Trail): test_module_Trail_ind.mq5
//+------------------------------------------------------------------+
//|          The program code is generated Modular project generator |
//|                      Copyright 2010-2017, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010-2017, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project test module Trail"
//--- Display indicator in the chart window
#property indicator_chart_window
//--- Number of buffers to calculate the indicator
#property indicator_buffers 1
//--- Number of graphic series in the indicator
#property indicator_plots   1
//---
double      Buffer1[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ArraySetAsSeries(Buffer1,true);
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   return(INIT_SUCCEEDED);
  };
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate (const int rates_total,
                 const int prev_calculated,
                 const datetime& time[],
                 const double& open[],
                 const double& high[],
                 const double& low[],
                 const double& close[],
                 const long& tick_volume[],
                 const long& volume[],
                 const int &spread[])
  {
   double TR=50;  // Trail in points
//--- calculation of Trail
   Buffer1[0]=TR;
   return(rates_total);
  };
```

Like in previous SL and TP codes, the distance for a trailing stop calculation is set as a constant for simplifying the program and reading.

The call of the trailing stop module should be implemented in the [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick) function, since the current price changes on the current tick and a stop level should be tracked continuously. The main module decides whether it should be changed or not. After receiving a distance in points, the EA modifies the position and moves the SL in the profit growth direction.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

   ...

//--- MODULE 4
   if(on_Trail)
      if(PositionSelect(_Symbol))
         if(PositionGetDouble(POSITION_PROFIT)>0)
           { // If there is an additional module
            double buffer_m4[];
            ArraySetAsSeries(buffer_m4,true);
            if(CopyBuffer(handle_m4,0,0,1,buffer_m4)<0) return;
            double TR=buffer_m4[0];
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
               if(PositionGetDouble(POSITION_PRICE_CURRENT)-TR*_Point>PositionGetDouble(POSITION_PRICE_OPEN))
                 {
                  double price_SL=PositionGetDouble(POSITION_PRICE_CURRENT)-TR*_Point;
                  if(price_SL>PositionGetDouble(POSITION_SL))
                     trade.PositionModify(_Symbol,NormalizeDouble(price_SL,Digits()),0);
                 }
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
               if(PositionGetDouble(POSITION_PRICE_CURRENT)+TR*_Point<PositionGetDouble(POSITION_PRICE_OPEN))
                 {
                  double price_SL=PositionGetDouble(POSITION_PRICE_CURRENT)+TR*_Point;
                  if(price_SL<PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL)==NULL)
                     trade.PositionModify(_Symbol,NormalizeDouble(price_SL,Digits()),0);
                 }
           }

   ...

  }
```

There is yet another position tracking method — placing SL in a breakeven point. When SL is activated, a position is closed with a zero result or a predetermined profit. The module might look something like this:

```
//****** project (module Breakeven): test_module_Breakeven_ind.mq5
//+------------------------------------------------------------------+
//|          The program code is generated Modular project generator |
//|                      Copyright 2010-2017, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010-2017, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project test module Breakeven"
//--- Display indicator in the chart window
#property indicator_chart_window
//--- Number of buffers to calculate the indicator
#property indicator_buffers 1
//--- Number of graphic series in the indicator
#property indicator_plots   1
//---
double      Buffer1[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ArraySetAsSeries(Buffer1,true);
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   return(INIT_SUCCEEDED);
  };
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate (const int rates_total,
                 const int prev_calculated,
                 const datetime& time[],
                 const double& open[],
                 const double& high[],
                 const double& low[],
                 const double& close[],
                 const long& tick_volume[],
                 const long& volume[],
                 const int &spread[])
  {
   double Breakeven=100; // Breakeven in points
//--- calculation of the Breakeven
   Buffer1[0]=Breakeven;
   return(rates_total);
  };
```

This module sets the distance of the current price from the position open price in points for setting SL to breakeven point. Calling the breakeven module should also be located in the [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick) function.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

   ...

//--- MODULE 7
   if(on_Breakeven)
      if(PositionSelect(_Symbol))
         if(PositionGetDouble(POSITION_PROFIT)>0)
           { // If there is an additional module
            double buffer_m7[];
            ArraySetAsSeries(buffer_m7,true);
            if(CopyBuffer(handle_m7,0,0,1,buffer_m7)<0) return;
            double TRB=buffer_m7[0];
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
               if(PositionGetDouble(POSITION_PRICE_CURRENT)-TRB*_Point>PositionGetDouble(POSITION_PRICE_OPEN))
                 {
                  double price_SL=PositionGetDouble(POSITION_PRICE_CURRENT)-5*_Point;
                  if(price_SL>PositionGetDouble(POSITION_SL))
                     trade.PositionModify(_Symbol,NormalizeDouble(price_SL,Digits()),PositionGetDouble(POSITION_TP));
                 }
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
               if(PositionGetDouble(POSITION_PRICE_CURRENT)+TRB*_Point<PositionGetDouble(POSITION_PRICE_OPEN))
                 {
                  double price_SL=PositionGetDouble(POSITION_PRICE_CURRENT)+5*_Point;
                  if(price_SL<PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL)==NULL)
                     trade.PositionModify(_Symbol,NormalizeDouble(price_SL,Digits()),PositionGetDouble(POSITION_TP));
                 }
           }

   ...

  }
```

### Example 3: Trading signals generation module

Probably, this is the most complex module in terms of implementation. It generates signals for performing trades: placing orders, closing positions, etc. Its development complexity stems from the fact that almost all indicators should be adapted to trading conditions. There are no indicators having the same inputs and generating working signals for different financial instruments.

Obviously, the main program should not configure the signal modules on its own, since too many signal modules may completely disrupt the module project operation. Therefore, the indicators generating trading signals should be prepared in advance before connecting to the general project. We will talk about this a bit later, in the section devoted to the optimization of multi-module EAs. Now, let's have a look at the code of the trading signals module:

```
//****** project (module signals): test_module_signals_ind.mq5
//+------------------------------------------------------------------+
//|          The program code is generated Modular project generator |
//|                      Copyright 2010-2017, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010-2017, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project test module Trading signals"
//--- Display indicator in the chart window
#property indicator_chart_window
//--- Number of buffers to calculate the indicator
#property indicator_buffers 1
//--- Number of graphic series in the indicator
#property indicator_plots   1
//---
double      Buffer1[];
//--- Variable for storing the indicator iBands handle
int    handle_Bands;
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   handle_Bands=iBands(_Symbol,_Period,20,0,3.5,PRICE_CLOSE);
//---
   ArraySetAsSeries(Buffer1,true);
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   return(INIT_SUCCEEDED);
  };
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate (const int rates_total,
                 const int prev_calculated,
                 const datetime& time[],
                 const double& open[],
                 const double& high[],
                 const double& low[],
                 const double& close[],
                 const long& tick_volume[],
                 const long& volume[],
                 const int &spread[])
  {
   double signal=0.0;
   ArraySetAsSeries(high,true);
   ArraySetAsSeries(low,true);
//--- Indicator buffers
   double         UpperBuffer[];
   double         LowerBuffer[];
   ArraySetAsSeries(UpperBuffer,true);  CopyBuffer(handle_Bands,1,0,1,UpperBuffer);
   ArraySetAsSeries(LowerBuffer,true);  CopyBuffer(handle_Bands,2,0,1,LowerBuffer);
//--- calculation of the Trading signals
   if(high[0]>UpperBuffer[0]) signal=-2.0;
   if(low[0]<LowerBuffer[0]) signal=2.0;
   Buffer1[0]=signal;
   return(rates_total);
  };
```

The following values are written to the signal module's indicator buffer:

- 2.0 — if BUY signal is formed;
- 0.0 — if there are no trading signals;
- -2.0 — if SELL signal is formed.

It is better to use the obtained values of the trading signals module in the special function of the main module - for example, like this:

```
//+------------------------------------------------------------------+
//| Trading signals generator                                        |
//+------------------------------------------------------------------+
sSignal Buy_or_Sell()
  {
   sSignal res={false,false};
//--- MODULE 5
   if(on_signals)
     { // If there is an additional module
      double buffer_m5[];
      ArraySetAsSeries(buffer_m5,true);
      if(CopyBuffer(handle_m5,0,0,1,buffer_m5)<0) return(res);
      if(buffer_m5[0]<-1) res.Sell=true;
      if(buffer_m5[0]>1) res.Buy=true;
     }
//--- MODULE 6
   if(on_Filter)
     { // If there is an additional module
      double buffer_m6[];
      ArraySetAsSeries(buffer_m6,true);
      if(CopyBuffer(handle_m6,0,0,1,buffer_m6)<0) return(res);
      lot=buffer_m6[0];
      if(buffer_m6[0]<1) res.Buy=false;
      if(buffer_m6[0]>-1) res.Sell=false;
     }
//---
//--- Indicator buffers
   double         UpperBuffer[];
   double         LowerBuffer[];
   double         MiddleBuffer[];
   ArraySetAsSeries(MiddleBuffer,true); CopyBuffer(handle_Bands,0,0,1,MiddleBuffer);
   ArraySetAsSeries(UpperBuffer,true);  CopyBuffer(handle_Bands,1,0,1,UpperBuffer);
   ArraySetAsSeries(LowerBuffer,true);  CopyBuffer(handle_Bands,2,0,1,LowerBuffer);
//--- Timeseries
   double L[];
   double H[];
   ArraySetAsSeries(L,true); CopyLow(_Symbol,_Period,0,1,L);
   ArraySetAsSeries(H,true); CopyHigh(_Symbol,_Period,0,1,H);
   if(H[0]>UpperBuffer[0]&& L[0]>MiddleBuffer[0]) res.Sell=true;
   if(L[0]<LowerBuffer[0] && H[0]<MiddleBuffer[0]) res.Buy=true;
//---
   return(res);
  }
```

There are a lot of trading strategies, and each of them has its own signals. Therefore, it is necessary to organize the work of the trading signals module, so that they fit into the given strategy. The trading strategy should be described in the EA documentation, so that signal module developers act in accordance with the technical requirements of the module project.

### Example 4: Signals filter module

Traders often apply trading signal filters to increase the profitability of trading robots. They may consider various things, including trend, trading time, news, additional signal indicators, etc.

```
//****** project (module Filter): test_module_Filter_ind.mq5
//+------------------------------------------------------------------+
//|          The program code is generated Modular project generator |
//|                      Copyright 2010-2017, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010-2017, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project test module Filter"
//--- Display indicator in the chart window
#property indicator_chart_window
//--- Number of buffers to calculate the indicator
#property indicator_buffers 1
//--- Number of graphic series in the indicator
#property indicator_plots   1
//---
double      Buffer1[];
//--- Variable for storing the indicator iBands handle
int    handle_Bands;
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   handle_Bands=iBands(_Symbol,_Period,35,0,4.1,PRICE_CLOSE);
//---
   ArraySetAsSeries(Buffer1,true);
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   return(INIT_SUCCEEDED);
  };
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate (const int rates_total,
                 const int prev_calculated,
                 const datetime& time[],
                 const double& open[],
                 const double& high[],
                 const double& low[],
                 const double& close[],
                 const long& tick_volume[],
                 const long& volume[],
                 const int &spread[])
  {
   double filtr=0.0;
   ArraySetAsSeries(high,true);
   ArraySetAsSeries(low,true);
//--- Indicator buffers
   double         MiddleBuffer[];
   ArraySetAsSeries(MiddleBuffer,true);  CopyBuffer(handle_Bands,0,0,1,MiddleBuffer);
//--- calculation of the filter
   if(high[0]<MiddleBuffer[0]) filtr=2.0;
   if(low[0]>MiddleBuffer[0]) filtr=-2.0;
   Buffer1[0]=filtr;
   return(rates_total);
  };
```

Thus, we have considered the options for implementing external modules and the principle of their integration into the module EA.

### Optimizing multi-module EAs

Optimizing multi-module EAs is, perhaps, one of the most critical issues. Indeed, how to optimize the input parameters of external modules in the strategy tester? If they are not set in the main module, our options appear to be limited. We can try to specify the input data of external modules discretely and then test the EA. However, this is a tedious and, most probably, pointless work. What can we do?

One of the possible options is using self-optimizing indicators as external modules. Many articles have been written about auto-optimization. I am also going to contribute to this topic. Let's use the ideas from the article " [Visual testing of profitability of indicators and alerts](https://www.mql5.com/en/articles/1557)". The author suggests using a candle value as a virtual trade execution price: the maximum candle value should be used for BUY, while the minimum one — for SELL. Consequently, the worst trading conditions are selected, and the inputs are optimized with such an approach to prices. It is assumed that with the obtained optimal values, the result will not be worse in real trade (on the same historical data interval). In real trading, no profit can be guaranteed after any optimization.

The strategy of our EA is based on trading inside the Bollinger indicator with a position reversal on its borders. Let's replace this indicator, while the channel will be based on the Envelope indicator. The borders equidistant from MA to a fixed distance will be formed from the MA indicator. The new signal indicator will automatically self-optimize before use. The optimal values showing maximum profit will be used as inputs. The two parameters — МА period and borders distance from the MA — are selected for optimization.

The algorithm of developing a signal indicator with the auto-optimization function:

1. Define the parameters and optimization criterion. In our case, the inputs are МА period and border shift distance, while the maximum profit is used as a criterion.
2. Create the optimization block in the indicator. The proposed example implements a complete search of input data in a specified range with a fixed step. The MA period is 10-100 with a step of 10. The shift values are searched within the interval 1000-10 000 with a step of 1000.


```
//+------------------------------------------------------------------+
//|                           Copyright 2018, Sergey Pavlov (DC2008) |
//|                              http://www.mql5.com/en/users/dc2008 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, Sergey Pavlov (DC2008)"
#property link      "http://www.mql5.com/en/users/dc2008"
#property link      "1.00"
#property link      "Example of a multimodule expert: project test module Trading signals"
//---
#include <MovingAverages.mqh>
//--- Display indicator in the chart window
#property indicator_chart_window
//--- Number of buffers to calculate the indicator
#property indicator_buffers 1
//--- Number of graphic series in the indicator
#property indicator_plots   1
//+------------------------------------------------------------------+
//| Structure of optimization results                                |
//+------------------------------------------------------------------+
struct Opt
  {
   int               var1;          // optimal value of parameter 1
   int               var2;          // optimal value of parameter 2
   double            profit;        // profit
  };
//---
double      Buffer1[];
bool        optimum=false;
Opt         test={NULL,NULL,NULL};
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   ArraySetAsSeries(Buffer1,true);
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   optimum=false;
   return(INIT_SUCCEEDED);
  };
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   double signal=0.0;
   Buffer1[0]=signal;
//--- optimization of input parameters
   if(!optimum)
     {
      ArraySetAsSeries(close,false);
      int count=rates_total;
      int total=0;
      int total_profit=0;
      for(int d=1000;d<10001;d+=1000)
         for(int j=10;j<101;j+=10)
           {
            double shift=d*_Point;
            bool open_buy=false;
            bool open_sell=false;
            double price_buy=0;
            double price_sell=0;
            double profit=0;
            int order=0;
            for(int i=j+1;i<count;i++)
              {
               double ma=SimpleMA(i,j,close);
               double sell=ma+shift;
               double buy=ma-shift;
               //--- BUY
               if(buy>close[i] && !open_buy)
                 {
                  price_buy=high[i]+spread[i]*_Point;
                  if(order==0) profit=0;
                  else profit+=price_sell-price_buy;
                  order++;
                  open_buy=true;
                  open_sell=false;
                 }
               //--- SELL
               if(sell<close[i] && !open_sell)
                 {
                  price_sell=low[i]-spread[i]*_Point;
                  if(order==0) profit=0;
                  else profit+=price_sell-price_buy;
                  order++;
                  open_sell=true;
                  open_buy=false;
                 }
               //---
              }
            if(profit>0)
               if(profit>test.profit)
                 {
                  test.var1=j;
                  test.var2=d;
                  test.profit=profit;
                  total_profit++;
                 }
            //---
            Comment("Optimizing inputs..."," passes=",total," // profitable ones =",total_profit);
            total++;
           }
      //---
      Print(" Optimization complete: ",test.var1," ",test.var2);
      Comment("Optimization complete");
      optimum=true;
     }
//---
   if(optimum)
      if(test.profit>0)
        {
         ArraySetAsSeries(close,true);
         double ma=SimpleMA(0,test.var1,close);
         double sell=ma+test.var2*_Period;
         double buy=ma-test.var2*_Period;
         //--- calculation of the Trading signals
         if(buy>close[0]) signal=2.0;
         if(sell<close[0]) signal=-2.0;
        }
//--- Indicator buffers
   Buffer1[0]=signal;
   return(rates_total);
  };
```

The optimization will take some time, during which the EA will not be able to trade. If the module trading robot works around the clock, then the auto-optimization delay should not have a significant impact on the total trading time.

### Conclusion

1. As it turns out, developing a multi-module EA is not only possible but sometimes useful and commercially beneficial as well.
2. The article demonstrates a primitive concept of a trading robot with external modules. Nevertheless, the technology of module programming allows the development of fairly complex projects involving third-party developers. When creating modules, developers are free not to disclose their code and thus preserve their copyrights to the algorithms.
3. The issue of optimizing module projects remains open. Self-optimization of signal indicators used as signal modules or filters is a topic that needs to be developed further.


Note: The attached file allows generating the source codes of the module project in the required configuration.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/3133](https://www.mql5.com/ru/articles/3133)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3133.zip "Download all attachments in the single ZIP archive")

[Modular\_project\_generator.ex5](https://www.mql5.com/en/articles/download/3133/modular_project_generator.ex5 "Download Modular_project_generator.ex5")(389.16 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [3D Modeling in MQL5](https://www.mql5.com/en/articles/2828)
- [Statistical distributions in the form of histograms without indicator buffers and arrays](https://www.mql5.com/en/articles/2714)
- [The ZigZag Indicator: Fresh Approach and New Solutions](https://www.mql5.com/en/articles/646)
- [Calculation of Integral Characteristics of Indicator Emissions](https://www.mql5.com/en/articles/610)
- [Testing Performance of Moving Averages Calculation in MQL5](https://www.mql5.com/en/articles/106)
- [Migrating from MQL4 to MQL5](https://www.mql5.com/en/articles/81)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/263299)**
(18)


![Cleverson Santos](https://c.mql5.com/avatar/2018/12/5C1AE2A0-51DD.jpg)

**[Cleverson Santos](https://www.mql5.com/en/users/cleverson_br)**
\|
26 Feb 2019 at 01:13

Congratulations! Great study material,


![panika1979](https://c.mql5.com/avatar/avatar_na2.png)

**[panika1979](https://www.mql5.com/en/users/panika1979)**
\|
27 Nov 2019 at 05:20

I would like to express my gratitude to the author of this article. Thank you for such labours. For me as a beginner in OOP and mql5 specifics, this article helps me to master the language in general. Gentlemen who see the shortcomings of the implementation of the concept in the above article, I would like to say that there is no limit to perfection anywhere, to improve perhaps I think and your work is still where .....

This article is rather oriented for beginners in language learning...

![Marcus Vinicius Coutinho Giugni Santos](https://c.mql5.com/avatar/2022/11/6370f110-76c0.png)

**[Marcus Vinicius Coutinho Giugni Santos](https://www.mql5.com/en/users/mvgiugni)**
\|
18 Jul 2021 at 14:15

![Haili Lv](https://c.mql5.com/avatar/avatar_na2.png)

**[Haili Lv](https://www.mql5.com/en/users/sweven)**
\|
14 Dec 2021 at 11:41

**MetaQuotes:**

New article [Developing a multi-module smart trading system](https://www.mql5.com/en/articles/3133) has been released:

By [Sergey Pavl](https://www.mql5.com/en/users/DC2008 "DC2008")

Hello Is there a source code?

![leida265 liao](https://c.mql5.com/avatar/2020/11/5FC1A061-F2FD.jpg)

**[leida265 liao](https://www.mql5.com/en/users/leida265)**
\|
15 Dec 2021 at 01:08

MT5 hedge ea in xm broker back [test normal](https://www.mql5.com/en/docs/matrix/matrix_characteristics/matrix_norm "MQL5 Documentation: function Norm") open order hedge! But when I test with ic, it doesn't open long orders and doesn't hedge? Does anyone know why this happens?


![Random Decision Forest in Reinforcement learning](https://c.mql5.com/2/31/family-eco.png)[Random Decision Forest in Reinforcement learning](https://www.mql5.com/en/articles/3856)

Random Forest (RF) with the use of bagging is one of the most powerful machine learning methods, which is slightly inferior to gradient boosting. This article attempts to develop a self-learning trading system that makes decisions based on the experience gained from interaction with the market.

![ZUP - Universal ZigZag with Pesavento patterns. Search for patterns](https://c.mql5.com/2/31/MQL5_ZUP.png)[ZUP - Universal ZigZag with Pesavento patterns. Search for patterns](https://www.mql5.com/en/articles/2990)

The ZUP indicator platform allows searching for multiple known patterns, parameters for which have already been set. These parameters can be edited to suit your requirements. You can also create new patterns using the ZUP graphical interfaces and save their parameters to a file. After that you can quickly check, whether these new patterns can be found on charts.

![Processing optimization results using the graphical interface](https://c.mql5.com/2/31/Frame_Mode.png)[Processing optimization results using the graphical interface](https://www.mql5.com/en/articles/4562)

This is a continuation of the idea of processing and analysis of optimization results. This time, our purpose is to select the 100 best optimization results and display them in a GUI table. The user will be able to select a row in the optimization results table and receive a multi-symbol balance and drawdown graph on separate charts.

![Synchronizing several same-symbol charts on different timeframes](https://c.mql5.com/2/31/6cd68idtz6fac-lu770iwbwo-3ndzmpk7.png)[Synchronizing several same-symbol charts on different timeframes](https://www.mql5.com/en/articles/4465)

When making trading decisions, we often have to analyze charts on several timeframes. At the same time, these charts often contain graphical objects. Applying the same objects to all charts is inconvenient. In this article, I propose to automate cloning of objects to be displayed on charts.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=urwxculfcnqitaftzroxvarfworpkgzx&ssn=1769156920701230599&ssn_dr=0&ssn_sr=0&fv_date=1769156920&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F3133&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20multi-module%20Expert%20Advisors%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915692012489877&fz_uniq=5062496570827514679&sv=2552)

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