---
title: MVC design pattern and its possible application
url: https://www.mql5.com/en/articles/9168
categories: Trading Systems
relevance_score: 1
scraped_at: 2026-01-23T21:38:38.083708
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/9168&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071987129996751404)

MetaTrader 5 / Trading systems


### Introduction

I suppose that many developers have gone through a stage when the project grows, becomes more complex and acquires new functionality, and thus the code starts resembling some kind of spaghetti. The project has not come to an end yet, and it is already very difficult to remember the places where this or that method is called, why this call is located exactly here, and how it all works.

Over time, it becomes more difficult to understand the code even for the code author. It is even worse when another developer tries to understand this code. The task becomes practically insoluble if the code author is unavailable for any reason at this time. An unstructured code is very difficult to maintain and to modify for any code more difficult than "Hello, world". This is one of the reasons for the emergence of design patterns. They bring a certain structure to the project, make it clearer and visually more understandable.

### The MVC pattern and its purpose

This pattern appeared quite a long time ago (in 1978), but its first description appeared much later, in 1988. Since then, the template has been developing further, giving rise to new approaches.

In this article we will consider the "classical MVC", without any complications or additional functionality. The idea is to split an existing code into three separate components: Model, View and Controller. According to the MVC pattern, these three components can be developed and maintained independently. Each component can be developed by a separate group of developers, who undertake to create new versions and to fix errors. Obviously this can make management of the overall project much easier. Furthermore, it can assist other people in understanding the code.

Let us take a look at each component.

1. **View**. View is responsible for visual representation of information. In a general case it sends data to the user. There can be different methods for presenting the same data to the user. For example, data can be represented by a table, graph or chart at the same time. In other words, an MVC-based application can contain multiple views. Views receive data from the Model without knowing what is happening inside the Model.

2. **Model**. The model contains data. It manages connections with data bases, sends requests and communicates with different resources. It modifies the data, verifies it, stores and deletes if necessary. The Model does not know anything about how the View works and how many Views exist, but it has the necessary interfaces through which the Views can request data. There is nothing else the Views can do, i.e. they cannot force the Model to change its state. This part is performed by the Controller. Internally, a Model can be composed of several other Models arranged in a hierarchy or working equally. The Model is not limited in this respect, except for the previously mentioned restriction — the Model keeps its internal structure in secret from the View and the Controller.

3. **Controller**. The Controller implements communication between the user and the Model. The Controller does not know what the Model is doing with the data, but it can tell the Model that it is time to update the content. In general, the Controller works with the Model by its interface, without trying to understand what is happening inside it.

The relationship between the individual components of the MVC pattern can be visually represented as follows:

![](https://c.mql5.com/2/42/MVC-Process.png)

Still, there are no particularly strict rules and restrictions on the use of MVC. The developer should be careful not to add Model operating logic into the Controller and not to interfere with the View. The Controller itself should be made lighter; you should not overload it. The MVC scheme is also used for other design patterns, such as, for example, Observer and Strategy.

Now, let us view how the MVC template can be used in MQL and whether there is a need do to use it.

### The simplest indicator from the MVC point of view

Let us create a simple indicator which can draw a line using the simplest calculations. The indicator is very small and its code fits in one file. This is how it might look like:

```
.......
#property indicator_chart_window
#property indicator_buffers 1
#property indicator_plots   1
//--- plot Label1
#property indicator_label1  "Label1"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDarkSlateBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2
//--- indicator buffers
double         lb[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, lb, INDICATOR_DATA);
   ArraySetAsSeries(lb, true);
   IndicatorSetString(INDICATOR_SHORTNAME, "Primitive1");
   IndicatorSetInteger(INDICATOR_DIGITS, _Digits);

   return(INIT_SUCCEEDED);
  }
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
   if(rates_total <= 4)
      return 0;

   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);

   int limit = rates_total - prev_calculated;

   if(limit == 0)
     {
     }
   else
      if(limit == 1)
        {

         lb[1] = (open[1] + close[1]) / 2;
         return(rates_total);

        }
      else
         if(limit > 1)
           {

            ArrayInitialize(lb, EMPTY_VALUE);

            limit = rates_total - 4;
            for(int i = limit; i >= 1 && !IsStopped(); i--)
              {
               lb[i] = (open[i] + close[i]) / 2;
              }
            return(rates_total);

           }

   lb[0] = (open[0] + close[0]) / 2;

   return(rates_total);
  }
//+------------------------------------------------------------------+
```

The indicator calculates the average value of open\[i\] + close\[i\]. The source code is provided in the attached zip-archive **MVC\_primitive\_1.zip**.

The indicator is very poorly written, which experienced developers can easily notice. Suppose there is a need to change the method of calculation: use only close\[i\] instead of open\[i\] + close\[i\]. This indicator has three places where we need to make changes. What if we need to make even more changes or to make calculations more complex? Obviously, it is better to implement calculations in a separate function. So, when necessary, we can make the relevant logic corrections only in this function.

Here is how the handler and the function look like now:

```
double Prepare(const datetime &t[], const double &o[], const double &h[], const double &l[], const double &c[], int shift) {

   ArraySetAsSeries(c, true);
   ArraySetAsSeries(o, true);

   return (o[shift] + c[shift]) / 2;
}
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
                const int &spread[]) {

   if(rates_total <= 4) return 0;

   int limit = rates_total - prev_calculated;

   if (limit == 0)        {
   } else if (limit == 1) {

      lb[1] = Prepare(time, open, high, low, close, 1);
      return(rates_total);

   } else if (limit > 1)  {

      ArrayInitialize(lb, EMPTY_VALUE);

      limit = rates_total - 4;
      for(int i = limit; i >= 1 && !IsStopped(); i--) {
         lb[i] = Prepare(time, open, high, low, close, i);
      }
      return(rates_total);

   }
   lb[0] = Prepare(time, open, high, low, close, 0);

   return(rates_total);
}
```

Please note that almost all timeseries are passed into the new function. Why? It is not necessary, as only two timeseries are used then: **open and close**. However, we expect that in the future there can be a lot of changes and improvements in the indicator, where the rest of the timeseries can be used. Actually, we implement a solid basis for potential versions.

Now let us consider the current code from the point of view of the MVC template.

- **View**. Since this component presents data to the user, then it should contain code related to indicator buffers. This should also include code from **OnInit()** — the whole code in our case.

- **Model**. Our indicator has a very simple one-line model, in which we calculate the average between **open and close**. Then the View is updated without our participation. Therefore, the Model component will only contain the **Prepare** function, which is written with the potential future development in mind.
- **Controller**. This component is responsible for communication between two other components and for user interaction. According to this, the component will include event handlers and indicator input parameters. Also, the Controller calls the Prepare function which serves as an input for the Model. Such a call will force the Model to change its state when new ticks arrive and when the symbol price history changes.

Let us try to rebuild our indicator based on the above explanation. Let us implement the code of the components not only in different files, but also in different folders. This is a reasonable solution because there can be multiple Views, the Model can contain other Models and the Controller can be very complex. Here is how the main indicator file looks like now:

```
//+------------------------------------------------------------------+
//|                                              MVC_primitive_2.mq5 |
//|                                Copyright 2021, Andrei Novichkov. |
//|                    https://www.mql5.com/en/users/andreifx60/news |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, Andrei Novichkov."
#property link      "https://www.mql5.com/en/users/andreifx60/news"

#property version   "1.00"

#property indicator_chart_window

#property indicator_buffers 1
#property indicator_plots   1

#include "View\MVC_View.mqh"
#include "Model\MVC_Model.mqh"

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit() {

   return Initialize();
}

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
                const int &spread[]) {

   if(rates_total <= 4) return 0;

   int limit = rates_total - prev_calculated;

   if (limit == 0)        {
   } else if (limit == 1) {

      lb[1] = Prepare(time, open, high, low, close, 1);
      return(rates_total);

   } else if (limit > 1)  {

      ArrayInitialize(lb, EMPTY_VALUE);

      limit = rates_total - 4;
      for(int i = limit; i >= 1 && !IsStopped(); i--) {
         lb[i] = Prepare(time, open, high, low, close, i);
      }
      return(rates_total);

   }
   lb[0] = Prepare(time, open, high, low, close, 0);

   return(rates_total);
}
//+------------------------------------------------------------------+
```

A few words about the indicator properties:

```
#property indicator_buffers 1
#property indicator_plots   1
```

These two lines could be moved to the View (the **MVC\_View.mqh** file). However, this would cause the generation of a compiler warning:

no indicator plot defined for indicator

Therefore, these two lines are left in the main file containing the Controller code. The source code of the indicator is in the attached zip-archive **MVC\_primitive\_2.zip**.

Now, please pay attention to the communications between separate components of the pattern. Currently there are no communications. We simply connect two include files and everything works. In particular, the View includes an indicator buffer in the form of a global variable and a function in which initialization is performed. Let us rewrite this part in a more correct and safer way. Let us combine the buffer, its initialization and access to it in one object. This produces a convenient and compact code, easy to debug and to maintain. Moreover, this approach provides all the opportunities for further program improvement. The developer can move a part of the code to a base class or interface and create, for example, an array of Views. Here is how the new View might look like:

```
class CView
  {
   public:
      void CView();
      void ResetBuffers();
      int  Initialize();
      void SetData(double value, int shift = 0);

   private:
      double _lb[];
      int    _Width;
      string _Name;
      string _Label;

  };// class CView

void CView::CView()
  {
      _Width = 2;
      _Name  = "Primitive" ;
      _Label = "Label1";
  }// void CView::CView()

void CView::ResetBuffers()
  {
   ArrayInitialize(_lb, EMPTY_VALUE);
  }

int CView::Initialize()
  {
      SetIndexBuffer     (0,   _lb, INDICATOR_DATA);
      ArraySetAsSeries   (_lb, true);

      IndicatorSetString (INDICATOR_SHORTNAME, _Name);
      IndicatorSetInteger(INDICATOR_DIGITS,    _Digits);

      PlotIndexSetString (0, PLOT_LABEL,      _Label);
      PlotIndexSetInteger(0, PLOT_DRAW_TYPE,  DRAW_LINE);
      PlotIndexSetInteger(0, PLOT_LINE_COLOR, clrDarkSlateBlue);
      PlotIndexSetInteger(0, PLOT_LINE_STYLE, STYLE_SOLID);
      PlotIndexSetInteger(0, PLOT_LINE_WIDTH, _Width);

      return(INIT_SUCCEEDED);
  }

void CView::SetData(double value,int shift)
  {
   _lb[shift] = value;
  }
```

Pay attention to the last method **SetData**. We prohibit uncontrolled access to the indicator buffer and implement a special method for access, to which additional checks can be added. Optionally, the method can be declared virtual in the base class. There are also some minor changes in the Controller file, but here they are not presented here. Obviously, we need here another constructor in which we can pass buffer initialization parameters, such as color, style, and others.

Also, the calls in the object of the first View do not look appropriate:

```
      IndicatorSetString (INDICATOR_SHORTNAME, _Name);
      IndicatorSetInteger(INDICATOR_DIGITS, _Digits);
```

In real life they should be removed from the CView class, keeping in mind that:

- There can be many Views.

- These two strings are not related to the View! It is the initialization of the indicator as a whole, so leave them in the Controller file, in the **OnInit** handler.

The source code of this indicator is provided in the attached zip-archive **MVC\_primitive\_3.zip**.

So, the main indicator file — the file with the Controller code — has become significantly shorter. The overall code is now more secure, and it is ready for future changes and debugging. But is it now clearer to other developers? It is quite doubtful. In this case, it could be more reasonable to leave the indicator code in one file combining the Controller, Model and View. The way it was at the very beginning.

This point of view seems logical. But this is only applicable for this specific indicator. Imagine an indicator consisting of a dozen files, having a graphical panel and requesting data on the web. In this case, the MVC model would be very useful. MVC would make it easy-to-understand for other people, allowing convenient detection of errors as well as providing basis for potential logic modifications and other changes. Do you want to attract a specialist for a specific development part? This will be much easier to do. Do you need to add another initialization scheme? This can also be done. The conclusion from the above is quite obvious: the more complex the project, the more useful the MVC pattern.

Does it apply only to indicators? Let us view the structure of Expert Advisors in terms of the possibility of using the MVC pattern.

### MVC in Expert Advisors

Let us create a very simple pseudo-Expert Advisor. It should open a buy position if the previous candle is bullish, and a sell position if it is bearish. For simplicity, the EA will not open real positions, while it will only simulate entry and exit. Only one position will exist in the market. In this case the EA code ( **Ea\_primitive.mq5** in the attached zip-archive) can look like this:

```
datetime dtNow;

int iBuy, iSell;

int OnInit()
  {
   iBuy  = iSell = 0;

   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {

  }

void OnTick()
  {
      if (IsNewCandle() )
        {
         double o = iOpen(NULL,PERIOD_CURRENT,1);
         double c = iClose(NULL,PERIOD_CURRENT,1);
         if (c < o)
           { // Enter Sell
            if (GetSell() == 1) return;
            if (GetBuy()  == 1) CloseBuy();
            EnterSell();
           }
         else
           {      // Enter Buy
            if (GetBuy()  == 1) return;
            if (GetSell() == 1) CloseSell();
            EnterBuy();
           }
        }// if (IsNewCandle() )
  }// void OnTick()

bool IsNewCandle()
  {
   datetime d = iTime(NULL, PERIOD_CURRENT, 0);
   if (dtNow == -1 || dtNow != d)
     {
      dtNow = d;
      return true;
     }
   return false;
  }// bool IsNewCandle()

void CloseBuy()  {iBuy = 0;}

void CloseSell() {iSell = 0;}

void EnterBuy()  {iBuy = 1;}

void EnterSell() {iSell = 1;}

int GetBuy()     {return iBuy;}

int GetSell()    {return iSell;}
```

When considering indicators, we already concluded that the **OnInit, OnDeinit** and other handlers relate to the Controller. The same applies to Expert Advisors. But what should be attributed to the View? It does not plot any graphical objects or charts. As you know, the View is responsible for data presentation to the user. For Expert Advisors, data presentation is the display of open positions. So, the View is anything related to the positions. These include orders, trailing stop, virtual Stop Loss and Take Profit, weighted average prices, etc.

Then the Model will include the logic of decision on position opening, lot selection, defining Take Profit and Stop Loss. Money management should also be implemented in the Model. This starts to look like a closed system consisting of several submodels: price analysis, volume calculation, account state check (possibly checking the state of other submodels) and the resulting market entry decision.

Let us change the structure of the pseudo-EA in accordance with the above considerations. We do not have lot calculation or work with the account, so let us perform the steps which we can — move functions related to different components to their subfolders and edit some of them. This is how the **OnTick** pseudocode handler will change:

```
void OnTick()
  {
      if (IsNewCandle() )
        {
         double o = iOpen(NULL,PERIOD_CURRENT,1);
         double c = iClose(NULL,PERIOD_CURRENT,1);
         if (MaySell(o, c) ) EnterSell();
         if (MayBuy(o, c)  ) EnterBuy();
        }// if (IsNewCandle() )
  }// void OnTick()
```

Even in this section we can see that the code has become shorter. Has it become clearer to a third-party developer? Here the assumption that we earlier considered for indicators are also applicable:

\- The more complex the EA, the more useful is the MVC pattern.

The whole EA is in the attached **MVC\_EA\_primitive.zip** archive. Now let us try to apply the MVC pattern to a "real" code.

For these purposes, let us use a simple Expert Advisor, it should not necessarily be a working one or well-written one. On the contrary, the Expert Advisor should poorly written — this way we can evaluate the effect of using the pattern.

For this purpose, I found an old draft of the **$OrdersInTheMorning** EA created in 2013. Its strategy was as follows:

- On Monday, at a specified time, the EA opened two pending buy and sell orders at a certain distance from the market. When one order triggered, the second was deleted. The order was closed on Friday evening. It worked only with the specified list of currency pairs.

Since the EA was developed for MetaTrader 4, I had to be remade it for MetaTrader 5 (which was though done very carelessly). Here are the main EA functions in their original form:

```
#property copyright "Copyright 2013, MetaQuotes Software Corp."
#property link      "http://www.metaquotes.net"

#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
//| script program start function                                    |
//+------------------------------------------------------------------+
input double delta = 200;
input double volumes = 0.03;
input double sTopLossKoeff = 1;
input double tAkeProfitKoeff = 2;
input int iTHour = 0;
input bool bHLprocess = true;
input bool oNlyMondeyOrders = false;
input string sTimeToCloseOrders = "22:00";
input string sTimeToOpenOrders  = "05:05";
input double iTimeIntervalForWork = 0.5;
input int iSlippage = 15;
input int iTradeCount = 3;
input int iTimeOut = 2000;

int dg;
bool bflag;

string smb[] = {"AUDJPY","CADJPY","EURJPY","NZDJPY","GBPJPY","CHFJPY"};

int init ()
{
   if ( (iTimeIntervalForWork < 0) || (iTimeIntervalForWork > 24) )
   {
      Alert ("... ",iTimeIntervalForWork);
   }
   return (0);
}

void OnTick()
{
   if ((oNlyMondeyOrders == true) && (DayOfWeek() != 1) )
   {
   }
   else
   {
         int count=ArraySize(smb);
         bool br = true;
         for (int i=0; i<count;i++)
         {
            if (!WeekOrderParam(smb[i], PERIOD_H4, delta*SymbolInfoDouble(smb[i],SYMBOL_POINT) ) )
               br = false;
         }
         if (!br)
            Alert("...");
         bflag = true;
    }//end if if ((oNlyMondeyOrders == true) && (DayOfWeek() != 1) )  else...

   if ((oNlyMondeyOrders == true) && (DayOfWeek() != 5) )
   {
   }
   else
   {
         if (OrdersTotal() != 0)
            Alert ("...");
   }//end if ((oNlyMondeyOrders == true) && (DayOfWeek() != 5) )  else...
}

  bool WeekOrderParam(string symbol,int tf, double dlt)
  {
   int j = -1;
   datetime mtime = 0;
   int k = 3;
   Alert(symbol);
   if (iTHour >= 0)
   {
      if (oNlyMondeyOrders == true)
      {
         for (int i = 0; i < k; i++)
         {
            mtime = iTime(symbol,0,i);
            if (TimeDayOfWeek(mtime) == 1)
            {
               if (TimeHour(mtime) == iTHour)
               {
                  j = i;
                  break;
               }
            }
         }
      }
      else
      {
         for (int i = 0; i < k; i++)
         {
            mtime = iTime(symbol,0,i);
            if (TimeHour(mtime) == iTHour)
            {
               j = i;
               break;
            }
         }
      }
      if (j == -1)
      {
         Print("tf?");
         return (false);
      }
   }//end if (iTHour >= 0)
   else
      j = 0;
   Alert(j);
   double bsp,ssp;
   if (bHLprocess)
   {
      bsp = NormalizeDouble(iHigh(symbol,0,j) + dlt, dg);
      ssp = NormalizeDouble(iLow(symbol,0,j) - dlt, dg);
   }
   else
   {
      bsp = NormalizeDouble(MathMax(iOpen(symbol,0,j),iClose(symbol,0,j)) + dlt, dg);
      ssp = NormalizeDouble(MathMin(iOpen(symbol,0,j),iClose(symbol,0,j)) - dlt, dg);
   }
   double slsize = NormalizeDouble(sTopLossKoeff * (bsp - ssp), dg);
   double tpb = NormalizeDouble(bsp + tAkeProfitKoeff*slsize, dg);
   double tps = NormalizeDouble(ssp - tAkeProfitKoeff*slsize, dg);
   datetime expr = 0;
   return (mOrderSend(symbol,ORDER_TYPE_BUY_STOP,volumes,bsp,iSlippage,ssp,tpb,NULL,0,expr,CLR_NONE) && mOrderSend(symbol,ORDER_TYPE_SELL_STOP,volumes,ssp,iSlippage,bsp,tps,NULL,0,expr,CLR_NONE) );
  }

 int mOrderSend( string symbol, int cmd, double volume, double price, int slippage, double stoploss, double takeprofit, string comment = "", int magic=0, datetime expiration=0, color arrow_color=CLR_NONE)
 {
   int ticket = -1;
      for (int i = 0; i < iTradeCount; i++)
      {
//         ticket=OrderSend(symbol,cmd,volume,price,slippage,stoploss,takeprofit,comment,magic,expiration,arrow_color);
         if(ticket<0)
            Print(symbol,": ",GetNameOP(cmd), GetLastError() ,iTimeOut);
         else
            break;
      }
   return (ticket);
 }

```

It has an initialization block, the **OnTick** handler and helper functions. Handlers will be left in the Controller. The outdated **init** call should be corrected. Now pay attention to **OnTick** . Inside the handler, there are some checks and a loop in which the **WeekOrderParam** helper function is called. This function refers to decision-making relating to market entry and position opening. This approach is absolutely wrong. As you can see, the function is long; it has multi-nested conditions and loops. This function should be split at least into two parts. The last function mOrderSend is quite good — it refers to the View, based on the ideas expressed above. In addition to the changes in the EA structure in accordance with the pattern, the code itself needs to be corrected. Comments will be given along with the relevant changes.

Let us start by moving the list of currency pairs to input parameters. Remove the garbage from the OnInit handler. Create the EA\_Init.mqh file which will contain the initialization details; connect this file to the main one. In this new file, create a class and perform all initialization in it:

```
class CInit {
public:
   void CInit(){}
   void Initialize(string pair);
   string names[];
   double points[];
   int iCount;
};

void CInit::Initialize(string pair) {

   iCount = StringSplit(pair, StringGetCharacter(",", 0), names);
   ArrayResize(points, iCount);
   for (int i = 0; i < iCount; i++) {
      points[i] = SymbolInfoDouble(names[i], SYMBOL_POINT);
   }
}
```

The code is very simple. I will explain a few points:

- All members of the class are public, which is not very correct. This is done as an exception, so as not to clutter up the code with multiple methods for accessing private members.
- There is only one method in the class and thus we do not necessarily need a class. But in this case, all the data would have global access, which should better be avoided.
- This class implements interaction with the user and is a part of the Controller.

Let us create an object of the created class type in the main EA file and call its initialization method in the **OnInit** handler.

Now let us continue with the Model. Delete all contents from the **OnTick** handler. Create the Model folder and the Model.mqh file in it. Create a CModel class in the new file. The class should contain two methods for checking the market entry and exit conditions. Also, in this class we need to save the flag indicating that positions have been opened or closed. Note that if it were not for the need to store this flag, then the existence of the entire class would not be necessary. A couple of functions would be enough. When trading in real conditions, we would need to implement additional checks, such as volume, funds and others. All of them should be implemented in the Model. Now, the file containing the Model looks like this:

```
class CModel {
public:
         void CModel(): bFlag(false) {}
         bool TimeToOpen();
         bool TimeToClose();
private:
   bool bFlag;
};

bool CModel::TimeToOpen() {

   if (bFlag) return false;

   MqlDateTime tm;
   TimeCurrent(tm);
   if (tm.day_of_week != 1) return false;
   if (tm.hour < iHourOpen) return false;

   bFlag = true;

   return true;
}

bool CModel::TimeToClose() {

   if (!bFlag) return false;

   MqlDateTime tm;
   TimeCurrent(tm);
   if (tm.day_of_week != 5)  return false;
   if (tm.hour < iHourClose) return false;

   bFlag = false;

   return true;
}
```

As in the previous case, create an object of this class type in the main EA file and add its method calls in the **OnInit** handler.

Now let us continue with the View. Create a View folder and a View.mqh file in it. This file contains elements for opening/closing orders and positions. It also has components for managing virtual levels, trailing stop and various graphical objects. In this case, the primary goal is to make code clear and simple. As an option, let us try to implement the View component without using classes. The View component will have three functions: one for market entry, the second one for closing all positions and the third one for closing orders. Each of the three functions uses a CTrade type object, which should be created anew every time it is used. This is not optimal:

```
void Enter() {

   CTrade trade;

   trade.SetExpertMagicNumber(Magic);
   trade.SetMarginMode();
   trade.SetDeviationInPoints(iSlippage);

   double dEnterBuy, dEnterSell;
   double dTpBuy,    dTpSell;
   double dSlBuy,    dSlSell;
   double dSlSize;

   for (int i = 0; i < init.iCount; i++) {
      dEnterBuy  = NormalizeDouble(iHigh(init.names[i],0,1) + delta * init.points[i], _Digits);
      dEnterSell = NormalizeDouble(iLow(init.names[i],0,1)  - delta * init.points[i], _Digits);
      dSlSell    = dEnterBuy;
      dSlBuy     = dEnterSell;
      dSlSize    = (dEnterBuy - dEnterSell) * tAkeProfitKoeff;
      dTpBuy     = NormalizeDouble(dEnterBuy + dSlSize, _Digits);
      dTpSell    = NormalizeDouble(dEnterSell - dSlSize, _Digits);

      trade.SetTypeFillingBySymbol(init.names[i]);

      trade.BuyStop(volumes,  dEnterBuy,  init.names[i], dSlBuy,  dTpBuy);
      trade.SellStop(volumes, dEnterSell, init.names[i], dSlSell, dTpSell);
   }
}

void ClosePositions() {

   CTrade trade;

   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      trade.PositionClose(PositionGetTicket(i) );
   }
}

void CloseOrder(string pair) {

   CTrade trade;

   ulong ticket;
   for (int i = OrdersTotal() - 1; i >= 0; i--) {
      ticket = OrderGetTicket(i);
      if (StringCompare(OrderGetString(ORDER_SYMBOL), pair) == 0) {
         trade.OrderDelete(ticket);
         break;
      }
   }
}
```

Let us change the code by creating the CView class. Move the already created functions to the new class and create one more component initialization method for a private field of the CTrade type. As in other cases, create an object of the created class type in the main file and add its initialization method call to the **OnInit** handler.

Now we need to implement the removal of the untriggered orders. To do this, add the **OnTrade** handler to the Controller. In the handler, check the change in the number of orders: if it has changed, then delete the corresponding untriggered order. This handler is the only tricky part of the EA. Create a method in the CView class and call it from the **OnTrade** handler of the Controller. This is how the View will look like:

```
#include <Trade\Trade.mqh>

class CView {

public:
   void CView() {}
   void Initialize();
   void Enter();
   void ClosePositions();
   void CloseAllOrder();
   void OnTrade();
private:
   void InitTicketArray() {
      ArrayInitialize(bTicket, 0);
      ArrayInitialize(sTicket, 0);
      iOrders = 0;
   }
   CTrade trade;
   int    iOrders;
   ulong  bTicket[], sTicket[];

};

void CView::OnTrade() {

   if (OrdersTotal() == iOrders) return;

   for (int i = 0; i < init.iCount; i++) {
      if (bTicket[i] != 0 && !OrderSelect(bTicket[i]) ) {
         bTicket[i] = 0; iOrders--;
         if (sTicket[i] != 0) {
            trade.OrderDelete(sTicket[i]);
            sTicket[i] = 0; iOrders--;
         }
         continue;
      }

      if (sTicket[i] != 0 && !OrderSelect(sTicket[i]) ) {
         sTicket[i] = 0; iOrders--;
         if (bTicket[i] != 0) {
            trade.OrderDelete(bTicket[i]);
            bTicket[i] = 0; iOrders--;
         }
      }
   }
}

void CView::Initialize() {

   trade.SetExpertMagicNumber(Magic);
   trade.SetMarginMode();
   trade.SetDeviationInPoints(iSlippage);

   ArrayResize(bTicket, init.iCount);
   ArrayResize(sTicket, init.iCount);

   InitTicketArray();
}

void CView::Enter() {

   double dEnterBuy, dEnterSell;
   double dTpBuy,    dTpSell;
   double dSlBuy,    dSlSell;
   double dSlSize;

   for (int i = 0; i < init.iCount; i++) {
      dEnterBuy  = NormalizeDouble(iHigh(init.names[i],0,1) + delta * init.points[i], _Digits);
      dEnterSell = NormalizeDouble(iLow(init.names[i],0,1)  - delta * init.points[i], _Digits);
      dSlSell    = dEnterBuy;
      dSlBuy     = dEnterSell;
      dSlSize    = (dEnterBuy - dEnterSell) * tAkeProfitKoeff;
      dTpBuy     = NormalizeDouble(dEnterBuy + dSlSize, _Digits);
      dTpSell    = NormalizeDouble(dEnterSell - dSlSize, _Digits);

      trade.SetTypeFillingBySymbol(init.names[i]);

      trade.BuyStop(volumes,  dEnterBuy,  init.names[i], dSlBuy,  dTpBuy);
      bTicket[i] = trade.ResultOrder();

      trade.SellStop(volumes, dEnterSell, init.names[i], dSlSell, dTpSell);
      sTicket[i] = trade.ResultOrder();

      iOrders +=2;
   }
}

void CView::ClosePositions() {

   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      trade.PositionClose(PositionGetTicket(i) );
   }

   InitTicketArray();
}

void CView::CloseAllOrder() {

   for (int i = OrdersTotal() - 1; i >= 0; i--) {
      trade.OrderDelete(OrderGetTicket(i));
   }
}
```

As you can see, the entire original code has been rewritten. Has it become better? Undoubtedly! The result of the entire work is in the attached **EA\_Real.zip** archive. Now the main file of the Expert Advisor (Controller) looks like this:

```
input string smb             = "AUDJPY, CADJPY, EURJPY, NZDJPY, GBPJPY, CHFJPY";
input double delta           = 200;
input double volumes         = 0.03;
input double tAkeProfitKoeff = 2;
input int    iHourOpen       = 5;
input int    iHourClose      = 22;
input int    iSlippage       = 15;
input int    Magic           = 12345;

#include "EA_Init.mqh"
#include "View\View.mqh"
#include "Model\Model.mqh"

CInit  init;
CModel model;
CView  view;


int OnInit()
{
   init.Initialize(smb);
   view.Initialize();

   return INIT_SUCCEEDED;
}

void OnTick() {
   if (model.TimeToOpen() ) {
      view.Enter();
      return;
   }
   if (model.TimeToClose() ) {
      view.CloseAllOrder();
      view.ClosePositions();
   }
}

void OnTrade() {
   view.OnTrade();
}
```

Now, if there is a need to change, add or fix something, we can simply work with an appropriate component of the Expert Advisor. The component location is easily determined. We can further develop the EA, implement new functionality, add Models and extend the View. We can even completely change one of the components, without affecting the other two.

In the considered MVC application, there is one aspect which was mentioned at the beginning of the article. It is about the interaction of pattern components with each other. From the user's point of view, there is no problem: we have a Controller, to which we can add a dialog box and a trade panel. We also have input parameters as part of the Controller. But how should Model and View interact? In our Expert Advisor they do not interact with each other directly, while they only interact through the Controller in the **OnTick** handler. In addition, the View communicates with the Controller in a similar way — by calling CInint object methods "directly". In this case, the interaction of components is organized through their global objects. This is because our Expert Advisor is very simple, and I did not want to complicate the code.

However, despite the coed simplicity, the View has eleven calls to the Controller. If we develop the EA further, the number of interactions may increase many times, spoiling the positive effect of the MVC pattern. This issue can be solved by refusing global objects and by accessing components and methods by reference. An example of this kind of interaction is MFC and its Document and View components.

From the point of view of the pattern, the interaction between the pattern components is not regulated in any way. Therefore, we will not go deep into this topic.

### Conclusion

In conclusion, let us see how we could further develop the structure of the indicator or the Expert Advisor, to which we originally applied the MVC pattern. Suppose there are two more Models. And one more View. The Controller has become much more complex. What should we do to stick to the MVC pattern? The solution here is the use of separate modules. It is very simple. We have three components, each of which provides a method for accessing it. Each component consists of separate modules. This method was mentioned [here](https://www.mql5.com/en/articles/7318). The same article discussed ways of interaction and management at the module level.

Programs used in the article:

| # | Issued to | Type | Description |
| --- | --- | --- | --- |
| 1 | MVC\_primitive\_1.zip | Archive | The first and worst indicator variant. |
| 2 | MVC\_primitive\_2.zip | Archive | The second indicator version with division into components. |
| 3 | MVC\_primitive\_3.zip | Archive | The third indicator version with objects. |
| 4 | EA\_primitive.zip | Archive | Pseudo-Expert Advisor |
| 5 | MVC\_EA\_primitive.zip | Archive | MVC-based pseudo-Expert Advisor. |
| 6 | EA\_Real.zip | Archive | MVC-based Expert Advisor. |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9168](https://www.mql5.com/ru/articles/9168)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9168.zip "Download all attachments in the single ZIP archive")

[MVC\_primitive\_1.zip](https://www.mql5.com/en/articles/download/9168/mvc_primitive_1.zip "Download MVC_primitive_1.zip")(0.9 KB)

[MVC\_primitive\_2.zip](https://www.mql5.com/en/articles/download/9168/mvc_primitive_2.zip "Download MVC_primitive_2.zip")(1.76 KB)

[MVC\_primitive\_3.zip](https://www.mql5.com/en/articles/download/9168/mvc_primitive_3.zip "Download MVC_primitive_3.zip")(2.18 KB)

[Ea\_primitive.zip](https://www.mql5.com/en/articles/download/9168/ea_primitive.zip "Download Ea_primitive.zip")(0.71 KB)

[MVC\_EA\_primitive.zip](https://www.mql5.com/en/articles/download/9168/mvc_ea_primitive.zip "Download MVC_EA_primitive.zip")(1.47 KB)

[EA\_Real.zip](https://www.mql5.com/en/articles/download/9168/ea_real.zip "Download EA_Real.zip")(71.7 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MVC design pattern and its application (Part 2): Diagram of interaction between the three components](https://www.mql5.com/en/articles/10249)
- [Using cryptography with external applications](https://www.mql5.com/en/articles/8093)
- [Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)
- [Parsing HTML with curl](https://www.mql5.com/en/articles/7144)
- [Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)
- [A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017](https://www.mql5.com/en/articles/5798)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/369756)**
(33)


![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
31 Mar 2021 at 22:14

There may be many parameters in the input parameters, among them Magik. Scatter these parameters among different components? In my opinion, this is not the best solution. But you can try your idea. See how it will look like.


![Andriy Konovalov](https://c.mql5.com/avatar/2016/2/56D125AB-A47E.jpg)

**[Andriy Konovalov](https://www.mql5.com/en/users/kanua)**
\|
31 Mar 2021 at 22:39

Ok, thanks for the article and answering the questions.


![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
22 Jul 2021 at 21:22

I am an amateur at AI, so I would like to withdraw my remaining MT4 money. I would like to invest again, so please allow me to withdraw as soon as possible. Thank you for your cooperation.

![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
22 Jul 2021 at 21:22

**ゆうじ 保坂:**

I am an amateur at AI, so I would like to withdraw my remaining MT4 money. I would like to invest again, so please allow me to withdraw as soon as possible. Thank you for your cooperation.

![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
22 Jul 2021 at 21:22

**ゆうじ 保坂:**

![Combination scalping: analyzing trades from the past to increase the performance of future trades](https://c.mql5.com/2/42/logo_01.png)[Combination scalping: analyzing trades from the past to increase the performance of future trades](https://www.mql5.com/en/articles/9231)

The article provides the description of the technology aimed at increasing the effectiveness of any automated trading system. It provides a brief explanation of the idea, as well as its underlying basics, possibilities and disadvantages.

![Other classes in DoEasy library (Part 69): Chart object collection class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__7.png)[Other classes in DoEasy library (Part 69): Chart object collection class](https://www.mql5.com/en/articles/9260)

With this article, I start the development of the chart object collection class. The class will store the collection list of chart objects with their subwindows and indicators providing the ability to work with any selected charts and their subwindows or with a list of several charts at once.

![Tips from a professional programmer (Part I): Code storing, debugging and compiling. Working with projects and logs](https://c.mql5.com/2/42/tipstricks.png)[Tips from a professional programmer (Part I): Code storing, debugging and compiling. Working with projects and logs](https://www.mql5.com/en/articles/9266)

These are some tips from a professional programmer about methods, techniques and auxiliary tools which can make programming easier.

![Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__6.png)[Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window](https://www.mql5.com/en/articles/9236)

In this article, I will continue the development of the chart object class. I will add the list of chart window objects featuring the lists of available indicators.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/9168&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071987129996751404)

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