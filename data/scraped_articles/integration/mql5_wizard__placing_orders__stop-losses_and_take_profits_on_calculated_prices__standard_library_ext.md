---
title: MQL5 Wizard: Placing Orders, Stop-Losses and Take Profits on Calculated Prices. Standard Library Extension
url: https://www.mql5.com/en/articles/987
categories: Integration, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:50:28.356866
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ipdozvflllnenpxdxnplwrllymejozpm&ssn=1769179826587588322&ssn_dr=0&ssn_sr=0&fv_date=1769179826&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F987&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%3A%20Placing%20Orders%2C%20Stop-Losses%20and%20Take%20Profits%20on%20Calculated%20Prices.%20Standard%20Library%20Extension%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691798266179652&fz_uniq=5068710108604595378&sv=2552)

MetaTrader 5 / Examples


### Introduction

[The MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) is a useful aid in developing large projects that require a strict architecture. [The MQL5 Wizard](https://www.mql5.com/en/articles/171) allows assembling ready made parts into an extensive scheme in the dialog mode within a few minutes, which cannot be overestimated. The MQL5 Wizard automates gathering all parts of the Expert together and automatically declares module parameters in the Expert according to their handles. When there is a great number of various modules involved, such automation saves a lot of time and routine operations.

It sounds good however there is an obvious disadvantage - capabilities of trading systems created with the Wizard based on standard classes are limited. This article considers a universal method allowing to significantly extend functionality of created Experts. When this method is implemented, compatibility with the Wizard and standard modules remains the same.

The idea of the method is using mechanisms of inheritance and polymorphism in [Object-Oriented Programming](https://www.mql5.com/en/docs/basis/oop) or creating classes to substitute standard classes in the code of generated Experts. This way all advantages of the Wizard and the Standard Library are utilized, which results in developing an Expert with required capabilities. To get there though, code has to be reduced a little, only by four strings.

The practical purpose of this article is adding to generated Experts a capability to place orders, Stop Losses and Take Profits at required price levels, not only at the specified distance from the current price.

Similar idea was discussed in the article " [MQL5 Wizard: How to Teach an EA to Open Pending Orders at Any Price](https://www.mql5.com/en/articles/723)". The significant drawback of the suggested solution is the "forced" change of the parameter of the trading signal module from the subordinate filter. This approach does not abet working with a lot of modules and using the Wizard for process optimization makes no sense.

Implementation of placing orders as well as Stop Losses and Take Profits at any prices in classes inherited from the standard ones is considered in detail further down. That said, any conflicts between modules are impossible. I hope that this article serves an example and inspires readers for writing own improvements of the standard framework and also allows users to implement the developed Library extension.

### 1\. Standard Algorithm of Decision Making

Experts, generated in the MQL5 Wizard are based on the **[CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert)** class instance. The pointer to the object of the **[CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal)** class is declared in this class. Further in the article this object is called the main signal for brevity's sake. The main signal contains pointers to the subordinate filters (signal modules are the inheritors of the **CExpertSignal** class).

If there are no open positions and orders, the Expert refers to the main signal to check for an opportunity to open a position on a new tick. The main signal inquires subordinate filters one by one and calculates weighted average forecast (direction) based on the obtained forecast. If its value exceeds the threshold (value of the **m\_threshold\_open** parameter in the main signal), order parameters and results of check for conditions of the **bool** type are passed to the Expert. If these conditions are met, either a position on the market price gets opened or a pending order at a certain distance from it (see Fig. 1). Stop Losses can be placed only at a fixed distance. The opening price indentions, Stop Loss and Take Profit from the market price are specified in the Expert settings and stored in the main signal, in the **m\_price\_level**, **m\_stop\_level** and **m\_take\_level** variables respectively.

So, currently two conditions have to be met for an order to be placed:

1. No open positions for the current symbol;
2. Absolute value of weighted average forecast exceeds the threshold value, which means that a trend is rather strong).

![Fig. 1. Pattern of decision making on entering the market](https://c.mql5.com/2/12/Fig1_Scheme1.png)

Fig. 1. Pattern of decision making on entering the market

The current pattern of decision making on Fig. 1 significantly limits the application area of the MQL5 Wizard. Strategies with fixed value of Stop Loss are rarely efficient in long term trading due to changeable volatility. Systems employing pending orders usually require placing them on dynamically calculated levels.

### 2\. Modified Algorithm of Decision Making

From the perspective of calculating levels and placing orders, this is a dead-end situation as signal modules cannot generate anything other than a forecast value and the main signal was not designed for work with levels. In this regard it is suggested:

- To introduce a new type of signal modules (we are going to call them price modules) able to generate order parameters;
- Train the main signal to handle those parameters, i.e. select the best ones and pass them on to the Expert.

Modified algorithm (Fig. 2) allows working with other requests besides pending orders. The essence of this algorithm is that it separates the entry point (price) from defining a trend (weighted average forecast). That means that to make a decision, the preferred direction of trading with filters is defined and a set of order parameters with suitable direction obtained from price modules is selected. If there are several similar sets available, the set received from the module with the greatest weight (parameter value **m\_weight** is chosen). If the direction has been determined but there are no entry points currently available, the Expert is inactive.

![Fig. 2. Modified pattern of decision making on entering the market](https://c.mql5.com/2/12/Fig2_Scheme2.png)

Fig. 2. Modified pattern of decision making on entering the market

To place a new order, the following requirements have to be met:

1. No open positions and orders for the current symbol;
2. Absolute value of weighted average forecast exceeds the threshold value;
3. At least one opening price of the order has been found.

The algorithm of Fig. 2 enables handling many possible entry points, filtering them by direction and choosing the best one in one Expert.

### 3\. Development of Modified Classes of the Expert and the Signal Module

The extension of the Library is based on two classes: **CExpertSignalAdvanced** and **CExpertAdvanced**, inherited from **[CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal)** and **[CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert)** respectively.

All measures on adding new capabilities aimed at changing interfaces employed for data exchange between different blocks of the Expert. For instance, to implement the algorithm by the pattern on Fig.2 it is required to organize interaction of the main signal (class **CExpertSignalAdvanced**) with price modules (descendants of that class). Updating already placed orders when data change, implies interaction of the Expert (class **CExpertAdvanced**) with the main signal.

So at this stage we are going to implement the pattern on Fig. 2 for opening positions and organize updating of already placed orders when parameters change (for example, when a more favorable entry point appears). Let us consider the **CExpertSignalAdvanced** class.

### 3.1. CExpertSignalAdvanced

This class is going to substitute its ancestor in the role of the main signal and become the basic one for price modules the same way as its ancestor is basic for signal modules.

```
class CExpertSignalAdvanced : public CExpertSignal
  {
protected:
   //---data members for storing parameters of the orders being placed
   double            m_order_open_long;         //opening price of the order to buy
   double            m_order_stop_long;         //Stop Loss of the order to buy
   double            m_order_take_long;         //Take Profit of the order to buy
   datetime          m_order_expiration_long;   //expiry time of the order to buy
   double            m_order_open_short;        //opening price of the order to sell
   double            m_order_stop_short;        //Stop Loss of the order to sell
   double            m_order_take_short;        //Take Profit of the order to sell
   datetime          m_order_expiration_short;  //expiry time of the order to sell
   //---
   int               m_price_module;            //index of the first price module in the m_filters array
public:
                     CExpertSignalAdvanced();
                    ~CExpertSignalAdvanced();
   virtual void      CalcPriceModuleIndex() {m_price_module=m_filters.Total();}
   virtual bool      CheckOpenLong(double &price,double &sl,double &tp,datetime &expiration);
   virtual bool      CheckOpenShort(double &price,double &sl,double &tp,datetime &expiration);
   virtual double    Direction(void);		//calculating weighted average forecast based on the data received from signal modules
   virtual double    Prices(void);		//updating of parameters of the orders being placed according to the data received from price modules
   virtual bool      OpenLongParams(double &price,double &sl,double &tp,datetime &expiration);
   virtual bool      OpenShortParams(double &price,double &sl,double &tp,datetime &expiration);
   virtual bool      CheckUpdateOrderLong(COrderInfo *order_ptr,double &open,double &sl,double &tp,datetime &ex);
   virtual bool      CheckUpdateOrderShort(COrderInfo *order_ptr,double &open,double &sl,double &tp,datetime &ex);
   double            getOpenLong()              { return m_order_open_long;         }
   double            getOpenShort()             { return m_order_open_short;        }
   double            getStopLong()              { return m_order_stop_long;         }
   double            getStopShort()             { return m_order_stop_short;        }
   double            getTakeLong()              { return m_order_take_long;         }
   double            getTakeShort()             { return m_order_take_short;        }
   datetime          getExpLong()               { return m_order_expiration_long;   }
   datetime          getExpShort()              { return m_order_expiration_short;  }
   double            getWeight()                { return m_weight;                  }
  };
```

Data members for storing order parameters have been declared in the **CExpertSignalAdvanced** class. The values of these variables get updated in the **Prices()** method. These variables act as a buffer.

Then the parameter **m\_price\_module** gets declared. It stores the index of the first price module in the **m\_filters** array declared in **CExpertSignal**. This array contains the pointers to included signal modules. Pointers to standard modules (filters) are stored in the beginning of the array. Then, starting from the **m\_price\_module** index, price modules come. To avoid the necessity of changing initialization methods of indicators and timeseries, it was decided to store everything in one array. Moreover, there is a possibility to include 64 modules through one array and usually it is sufficient.

In addition, helper methods were declared in the **CExpertSignalAdvanced** class for obtaining values of protected data members. Their names start with **get** (see class declaration).

**3.1.1. Constructor**

The **constructor** **CExpertSignalAdvanced** initializes variables declared inside the class:

```
CExpertSignalAdvanced::CExpertSignalAdvanced()
  {
   m_order_open_long=EMPTY_VALUE;
   m_order_stop_long=EMPTY_VALUE;
   m_order_take_long=EMPTY_VALUE;
   m_order_expiration_long=0;
   m_order_open_short=EMPTY_VALUE;
   m_order_stop_short=EMPTY_VALUE;
   m_order_take_short=EMPTY_VALUE;
   m_order_expiration_short=0;
   m_price_module=-1;
  }
```

**3.1.2. CalcPriceModuleIndex()**

The method **CalcPriceModuleIndex()** assigns in **m\_price\_module** current number of array elements, which is equal to the index of the following added module. This method is called before adding the first price module. The function body is in the class declaration.

```
virtual void      CalcPriceModuleIndex() {m_price_module=m_filters.Total();}
```

**3.1.3. CheckOpenLong(...) and CheckOpenShort(...)**

The method **CheckOpenLong(...)** is called from the **CExpert** class instance and works as described below:

1. Check for included price modules. If there are none, then call the eponymous method of the parent class;
2. Receive weighted average forecast (direction) from the **Direction() method;**
3. Verify if entry conditions are met by comparing the direction with EMPTY\_VALUE and the threshold value of **m\_threshold\_open**;
4. Renew values of order parameters by the **Prices()**, method and pass them on to the Expert with the **OpenLongParams(...)** function. Save the result of this function;
5. Return the saved result.

```
bool CExpertSignalAdvanced::CheckOpenLong(double &price,double &sl,double &tp,datetime &expiration)
  {
//--- if price modules were not found, call the method of the basic class CExpertSignal
   if(m_price_module<0)
      return(CExpertSignal::CheckOpenLong(price,sl,tp,expiration));

   bool   result   =false;
   double direction=Direction();
//--- prohibitive signal
   if(direction==EMPTY_VALUE)
      return(false);
//--- check for exceeding the threshold
   if(direction>=m_threshold_open)
     {
      Prices();
      result=OpenLongParams(price,sl,tp,expiration);//there's a signal if m_order_open_long!=EMPTY_VALUE
     }
//--- return the result
   return(result);
  }
```

**CheckOpenShort(...)** has the same operation principle and is used the same way, that is why we are not going to consider it.

**3.1.4 Direction()**

The **Direction()** method queries filters and calculates weighted average forecast. This method is very similar to an eponymous method of the parent class **CExpertSignal** with an exception that in the loop we do not refer to all elements of the **m\_filters** array but only to those having an index that varies from 0 to the one less than **m\_price\_module**. Everything else is similar to **CExpertSignal::Direction()**.

```
double CExpertSignalAdvanced::Direction(void)
  {
   long   mask;
   double direction;
   double result=m_weight*(LongCondition()-ShortCondition());
   int    number=(result==0.0)? 0 : 1;      // number of queried modules
//--- loop by filters
   for(int i=0;i<m_price_module;i++)
     {
      //--- mask for bitmaps (variables, containing flags)
      mask=((long)1)<<i;
      //--- checking for a flag of ignoring a filter signal
      if((m_ignore&mask)!=0)
         continue;
      CExpertSignal *filter=m_filters.At(i);
      //--- checking for a pointer
      if(filter==NULL)
         continue;
      direction=filter.Direction();
      //--- prohibitive signal
      if(direction==EMPTY_VALUE)
         return(EMPTY_VALUE);
      if((m_invert&mask)!=0)
         result-=direction;
      else
         result+=direction;
      number++;
     }
//--- averaging the sum of weighted forecasts
   if(number!=0)
      result/=number;
//--- return the result
   return(result);
  }
```

**3.1.5. Prices()**

The method **Prices()** iterates over the second part of the **m\_filters** array starting from the **m\_price\_module** index up to the end. This queries price modules and renews the values of the class variables with the functions **OpenLongParams(...)** and **OpenShortParams(...)**. Before the cycle, the parameter values get cleared.

During the cycle, parameter values get overwritten if the weight of the current price module ( **m\_weight**) is greater than the one of the previously queried modules, which provided the values. As a result, either empty parameters are left (if nothing was found) or parameters with the best weight available at the time of calling a method.

```
double CExpertSignalAdvanced::Prices(void)
  {
   m_order_open_long=EMPTY_VALUE;
   m_order_stop_long=EMPTY_VALUE;
   m_order_take_long=EMPTY_VALUE;
   m_order_expiration_long=0;
   m_order_open_short=EMPTY_VALUE;
   m_order_stop_short=EMPTY_VALUE;
   m_order_take_short=EMPTY_VALUE;
   m_order_expiration_short=0;
   int    total=m_filters.Total();
   double last_weight_long=0;
   double last_weight_short=0;
//--- cycle for price modules
   for(int i=m_price_module;i<total;i++)
     {
      CExpertSignalAdvanced *prm=m_filters.At(i);
      if(prm==NULL)
         continue;
//--- ignore the current module if it has returned EMPTY_VALUE
      if(prm.Prices()==EMPTY_VALUE)continue;
      double weight=prm.getWeight();
      if(weight==0.0)continue;
//--- select non-empty values from modules with the greatest weight
      if(weight>last_weight_long && prm.getExpLong()>TimeCurrent())
         if(prm.OpenLongParams(m_order_open_long,m_order_stop_long,m_order_take_long,m_order_expiration_long))
            last_weight_long=weight;
      if(weight>last_weight_short && prm.getExpShort()>TimeCurrent())
         if(prm.OpenShortParams(m_order_open_short,m_order_stop_short,m_order_take_short,m_order_expiration_short))
            last_weight_short=weight;
     }
   return(0);
  }
```

**3.1.6.** **OpenLongParams(...) and** **OpenShortParams(...)**

Within the **CExpertSignalAdvanced** class, the method **OpenLongParams(...)** passes parameter values of the order to buy by reference from the class variables to the input parameters.

The role of this method in the parent class was slightly different. This was calculating required parameters based on the market price and specified indentations in the main signal. Now this only passes ready parameters. If the opening price is correct (not equal to EMPTY\_VALUE), then the method returns true, otherwise false.

```
bool CExpertSignalAdvanced::OpenLongParams(double &price,double &sl,double &tp,datetime &expiration)
  {
   if(m_order_open_long!=EMPTY_VALUE)
     {
      price=m_order_open_long;
      sl=m_order_stop_long;
      tp=m_order_take_long;
      expiration=m_order_expiration_long;
      return(true);
     }
   return(false);
  }
```

We are not going to consider **OpenShortParams(...)** as its operation principle is the same and it is used similarly.

**3.1.7. CheckUpdateOrderLong(...) and CheckUpdateOrderShort(...)**

The methods **CheckUpdateOrderLong(...)** and **CheckUpdateOrderShort(...)** are called the **CExpertAdvanced** class. They are used for updating already placed pending order according to the last price levels.

We are going to examine the method **CheckUpdateOrderLong(...)** more closely. At first price levels get updated when calling the method **Prices(...)**, then a check for data updates is carried out to exclude possible modification errors. Finally, the method **OpenLongParams(...)** is called for passing updated data and returning the result.

```
bool CExpertSignalAdvanced::CheckUpdateOrderLong(COrderInfo *order_ptr,double &open,double &sl,double &tp,datetime &ex)
  {
   Prices();   //update prices
//--- check for changes
   double point=m_symbol.Point();
   if(   MathAbs(order_ptr.PriceOpen() - m_order_open_long)>point
      || MathAbs(order_ptr.StopLoss()  - m_order_stop_long)>point
      || MathAbs(order_ptr.TakeProfit()- m_order_take_long)>point
      || order_ptr.TimeExpiration()!=m_order_expiration_long)
      return(OpenLongParams(open,sl,tp,ex));
//--- update is not required
   return (false);
  }
```

**CheckUpdateOrderShort(...)** is not going to be considered as it operates similarly and applied the same way.

### 3.2. CExpertAdvanced

Changes in the class of Expert concern only modifying already placed orders according to the updated data on the prices in the main signal. The **CExpertAdvanced** class declaration is presented below.

```
class CExpertAdvanced : public CExpert
  {
protected:
   virtual bool      CheckTrailingOrderLong();
   virtual bool      CheckTrailingOrderShort();
   virtual bool      UpdateOrder(double price,double sl,double tp,datetime ex);
public:
                     CExpertAdvanced();
                    ~CExpertAdvanced();
  };
```

As we can see, the methods are few, the constructor and destructor are empty.

**3.2.1. CheckTrailingOrderLong() and CheckTrailingOrderShort()**

The method **CheckTrailingOrderLong()** overrides an eponymous method of the basic class and calls the method **CheckUpdateOrderLong(...)** of the main signal to figure out the necessity of modifying the order. If modification is required, the method **UpdateOrder(...)** is called and result gets returned.

```
bool CExpertAdvanced::CheckTrailingOrderLong(void)
  {
   CExpertSignalAdvanced *signal_ptr=m_signal;
//--- check for the opportunity to modify the order to buy
   double price,sl,tp;
   datetime ex;
   if(signal_ptr.CheckUpdateOrderLong(GetPointer(m_order),price,sl,tp,ex))
      return(UpdateOrder(price,sl,tp,ex));
//--- return with no actions taken
   return(false);
  }
```

The method **CheckTrailingOrderShort()** is similar and is used the same way.

**3.2.2. UpdateOrder()**

The **UpdateOrder()** function starts with the check for a relevant price (not EMPTY\_VALUE). If there is none, the order gets deleted, else the order is modified according to received parameters.

```
bool CExpertAdvanced::UpdateOrder(double price,double sl,double tp,datetime ex)
  {
   ulong  ticket=m_order.Ticket();
   if(price==EMPTY_VALUE)
      return(m_trade.OrderDelete(ticket));
//--- modify the order, return the result
   return(m_trade.OrderModify(ticket,price,sl,tp,m_order.TypeTime(),ex));
  }
```

Development of inheritors of standard classes is complete.

### 4\. Developing Price Modules

We already have the base for creating Experts that place orders and Stop Losses at calculated levels. To be precise, we have classes ready to use for work with price levels. Now only the modules to generate those levels are left to be written.

The process of developing price modules is similar to writing modules of trading signals. The only difference between them is that it is **Prices()**, the method responsible for price updating within the module, has to be overridden, not **LongCondition()**, **ShortCondition()** or **Direction()** like in signal modules. Ideally, the reader should have a clear idea about signal modules development. The articles " [Create Your Own Trading Robot in 6 Steps!](https://www.mql5.com/en/articles/367)" and " [Trading Signal Generator Based on a Custom Indicator](https://www.mql5.com/en/articles/691)" may be useful in this.

Code of several price modules is going to serve an example.

### 4.1. Price Module Based on the "Delta ZigZag" Indicator

The indicator [Delta ZigZag](https://www.mql5.com/en/code/1321) draws levels by the specified number of several latest peaks. If the price crosses those levels, it means a probable trend reversal.

The goal of the price module is to take the entry level from the indicator buffer, find the nearest local extremum for placing Stop Loss, calculate Take Profit by multiplying Stop Loss by the coefficient specified in the settings.

![Fig. 3. Illustration of the price module operation on the Delta ZigZag indicator using the order to buy as an example](https://c.mql5.com/2/10/Article_Module_Example_DZZ_2.png)

Fig. 3. Illustration of the price module operation on the Delta ZigZag indicator using the order to buy as an example

Fig 3. represents the levels generated by the price module. Before the order triggers, Stop Loss and Take Profit change accordingly following the updates of the minimum.

**4.1.1. Module Descriptor**

```
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=DeltaZZ Price Module                                       |
//| Type=SignalAdvanced                                              |
//| Name=DeltaZZ Price Module                                        |
//| ShortName=DeltaZZ_PM                                             |
//| Class=CPriceDeltaZZ                                              |
//| Page=not used                                                    |
//| Parameter=setAppPrice,int,1, Applied price: 0 - Close, 1 - H/L   |
//| Parameter=setRevMode,int,0, Reversal mode: 0 - Pips, 1 - Percent |
//| Parameter=setPips,int,300,Reverse in pips                        |
//| Parameter=setPercent,double,0.5,Reverse in percent               |
//| Parameter=setLevels,int,2,Peaks number                           |
//| Parameter=setTpRatio,double,1.6,TP:SL ratio                      |
//| Parameter=setExpBars,int,10,Expiration after bars number         |
//+------------------------------------------------------------------+
// wizard description end
```

The first five parameters in the descriptor are required for setting up the indicator in use. Then come the coefficient for calculating Take Profit based on Stop Loss and expiration time in the bars for orders from the current price module.

**4.1.2. Class Declaration**

```
class CPriceDeltaZZ : public CExpertSignalAdvanced
  {
protected:
   CiCustom          m_deltazz;           //object of the DeltaZZ indicator
   //--- module settings
   int               m_app_price;
   int               m_rev_mode;
   int               m_pips;
   double            m_percent;
   int               m_levels;
   double            m_tp_ratio;          //tp:sl ratio
   int               m_exp_bars;          //lifetime of the orders in bars
   //--- method of indicator initialization
   bool              InitDeltaZZ(CIndicators *indicators);
   //--- helper methods
   datetime          calcExpiration() { return(TimeCurrent()+m_exp_bars*PeriodSeconds(m_period)); }
   double            getBuySL();          //function for searching latest minimum of ZZ for buy SL
   double            getSellSL();         //function for searching latest maximum of ZZ for sell SL
public:
                     CPriceDeltaZZ();
                    ~CPriceDeltaZZ();
   //--- methods of changing module settings
   void              setAppPrice(int ap)           { m_app_price=ap; }
   void              setRevMode(int rm)            { m_rev_mode=rm;  }
   void              setPips(int pips)             { m_pips=pips;    }
   void              setPercent(double perc)       { m_percent=perc; }
   void              setLevels(int rnum)           { m_levels=rnum;  }
   void              setTpRatio(double tpr)        { m_tp_ratio=tpr; }
   void              setExpBars(int bars)          { m_exp_bars=bars;}
   //--- method of checking correctness of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating indicators
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- main method of price module updating the output data
   virtual double    Prices();
  };
```

Protected data - object of the indicator of the **CiCustom** type and parameters according to the types specified in the descriptor are declared in the module.

**4.1.3. Constructor**

Constructor initializes class parameters with default values. Later, these values get initialized once again when the module gets included into the main signal according to the input parameters of the Expert.

```
CPriceDeltaZZ::CPriceDeltaZZ() : m_app_price(1),
                                 m_rev_mode(0),
                                 m_pips(300),
                                 m_percent(0.5),
                                 m_levels(2),
                                 m_tp_ratio(1),
                                 m_exp_bars(10)
  {
  }
```

**4.1.4. ValidationSettings()**

**ValidationSettings()** is an important method for checking input parameters. If the values of the module parameters are invalid, the result false is returned and an error message is printed in the journal.

```
bool CPriceDeltaZZ::ValidationSettings(void)
  {
//--- checking for settings of additional filters
   if(!CExpertSignal::ValidationSettings())
      return(false);
//--- initial data check
   if(m_app_price<0 || m_app_price>1)
     {
      printf(__FUNCTION__+": Applied price must be 0 or 1");
      return(false);
     }
   if(m_rev_mode<0 || m_rev_mode>1)
     {
      printf(__FUNCTION__+": Reversal mode must be 0 or 1");
      return(false);
     }
   if(m_pips<10)
     {
      printf(__FUNCTION__+": Number of pips in a ray must be at least 10");
      return(false);
     }
   if(m_percent<=0)
     {
      printf(__FUNCTION__+": Percent must be greater than 0");
      return(false);
     }
   if(m_levels<1)
     {
      printf(__FUNCTION__+": Ray Number must be at least 1");
      return(false);
     }
   if(m_tp_ratio<=0)
     {
      printf(__FUNCTION__+": TP Ratio must be greater than zero");
      return(false);
     }
   if(m_exp_bars<0)
     {
      printf(__FUNCTION__+": Expiration must be zero or positive value");
      return(false);
     }
//--- parameter check passed
   return(true);
  }
```

**4.1.5. InitIndicators(...)**

The **InitIndicators(...)** method calls an eponymous method of the basic class and initializes the indicator of the current module by the **InitDeltaZZ(...)** method.

```
bool CPriceDeltaZZ::InitIndicators(CIndicators *indicators)
  {
//--- initialization of indicator filters
   if(!CExpertSignal::InitIndicators(indicators))
      return(false);
//--- creating and initializing of custom indicator
   if(!InitDeltaZZ(indicators))
      return(false);
//--- success
   return(true);
  }
```

**4.1.6. InitDeltaZZ(...)**

The **InitDeltaZZ(...)** method adds the object of custom indicator to the collection and creates the new indicator "Delta ZigZag".

```
bool CPriceDeltaZZ::InitDeltaZZ(CIndicators *indicators)
  {
//--- adds to collection
   if(!indicators.Add(GetPointer(m_deltazz)))
     {
      printf(__FUNCTION__+": error adding object");
      return(false);
     }
//--- specifies indicator parameters
   MqlParam parameters[6];
//---
   parameters[0].type=TYPE_STRING;
   parameters[0].string_value="deltazigzag.ex5";
   parameters[1].type=TYPE_INT;
   parameters[1].integer_value=m_app_price;
   parameters[2].type=TYPE_INT;
   parameters[2].integer_value=m_rev_mode;
   parameters[3].type=TYPE_INT;
   parameters[3].integer_value=m_pips;
   parameters[4].type=TYPE_DOUBLE;
   parameters[4].double_value=m_percent;
   parameters[5].type=TYPE_INT;
   parameters[5].integer_value=m_levels;
//--- object initialization
   if(!m_deltazz.Create(m_symbol.Name(),m_period,IND_CUSTOM,6,parameters))
     {
      printf(__FUNCTION__+": error initializing object");
      return(false);
     }
//--- number of the indicator buffers
   if(!m_deltazz.NumBuffers(5)) return(false);
//--- ок
   return(true);
  }
```

After successful completion of the methods **ValidationSettings()**, **InitDeltaZZ(...)** and **InitIndicators(...)**, the module is initialized and is ready for work.

**4.1.7. Prices()**

This method is basic for the price module. This is the very place where order parameters get updated. Their values are passed on to the main signal. This method returns the operation result of the **double** type. This was implemented mainly for the future development. The result of the method **Prices()** can code some peculiar situations and events so the main signal could handle them accordingly. Currently only handling of the returned value EMPTY\_VALUE is intended. Upon receiving this result, the main signal will ignore the parameters suggested by the module.

Operation algorithm of the **Prices()** method in this module:

1. Take the opening prices of the orders from the indicator buffers 3 and 4 for buying and selling respectively;
2. Zeroize final parameters of the orders;
3. Check for the price to buy. If there is one, identify the level to place Stop Loss by the getBuySL() method, calculate Take Profit level by the Stop Loss value and the opening price and calculate the order expiration time;
4. Check for the price to sell. If detected, find the level to place Stop Loss by the getSellSL()method, calculate Take Profit level by Stop Loss value and opening price and calculate the order expiration time;
5. Exit.

Conditions of presence of the prices to buy and to sell are mutually exclusive due to some operational aspects of the "Delta ZigZag" indicator. Buffers 3 and 4 are drawn as dots by default (see Fig. 3).

```
double CPriceDeltaZZ::Prices(void)
  {
   double openbuy =m_deltazz.GetData(3,0);//receive the last value from buffer 3
   double opensell=m_deltazz.GetData(4,0);//receive the last value from buffer 4
//--- clear parameter values
   m_order_open_long=EMPTY_VALUE;
   m_order_stop_long=EMPTY_VALUE;
   m_order_take_long=EMPTY_VALUE;
   m_order_expiration_long=0;
   m_order_open_short=EMPTY_VALUE;
   m_order_stop_short=EMPTY_VALUE;
   m_order_take_short=EMPTY_VALUE;
   m_order_expiration_short=0;
   int digits=m_symbol.Digits();
//--- check for the prices to buy
   if(openbuy>0)//if buffer 3 is not empty
     {
      m_order_open_long=NormalizeDouble(openbuy,digits);
      m_order_stop_long=NormalizeDouble(getBuySL(),digits);
      m_order_take_long=NormalizeDouble(m_order_open_long + m_tp_ratio*(m_order_open_long - m_order_stop_long),digits);
      m_order_expiration_long=calcExpiration();
     }
//--- check for the prices to sell
   if(opensell>0)//if buffer 4 is not empty
     {
      m_order_open_short=NormalizeDouble(opensell,digits);
      m_order_stop_short=NormalizeDouble(getSellSL(),digits);
      m_order_take_short=NormalizeDouble(m_order_open_short - m_tp_ratio*(m_order_stop_short - m_order_open_short),digits);
      m_order_expiration_short=calcExpiration();
     }
   return(0);
  }
```

**4.1.8. getBuySL() and getSellSL()**

Methods **getBuySL()** and **getSellSL()** are looking for local respective minimums and maximums for placing Stop Losses. The last non-zero value in a relevant buffer – the price level of the last local extremum is sought in every method.

```
double CPriceDeltaZZ::getBuySL(void)
  {
   int i=0;
   double sl=0.0;
   while(sl==0.0)
     {
      sl=m_deltazz.GetData(0,i);
      i++;
     }
   return(sl);
  }
double CPriceDeltaZZ::getSellSL(void)
  {
   int i=0;
   double sl=0.0;
   while(sl==0.0)
     {
      sl=m_deltazz.GetData(1,i);
      i++;
     }
   return(sl);
  }
```

### 4.2. Price Module Based on Inside Bar

Inside bar is one of the widely used models or patterns in the trading with no indicators called [Price action](https://en.wikipedia.org/wiki/Price_action_trading "https://en.wikipedia.org/wiki/Price_action_trading"). An inside bar is a bar having High and Low within the extrema of the previous bar. An inside bar indicates the beginning of consolidation or possible reversal of the price movement.

On Fig. 4 inside bars are inside red ellipses. When an inside bar is detected, the price module generates opening prices of the buystop and sellstop orders on the extrema of the bar preceding the inside bar.

Opening prices are the Stop Loss levels for opposite orders. Take Profit is calculated the same way as in the module discussed above - by multiplying the Stop Loss level by the coefficient specified in the settings. Opening prices and Stop Losses are marked with red horizontal lines and Take Profits with green ones.

![Fig. 4. Illustration of inside bars and levels of the price module](https://c.mql5.com/2/10/EURUSDH1_IB_EXAMPLE__2.png)

Fig. 4. Illustration of inside bars and levels of the price module

On Fig. 4 orders to sell are not placed as the trend is upcoming. Nevertheless, price module generates entry levels in both directions. Some Take Profit levels are outside the plot.

Unlike the previous module, algorithm of this module is simple and does not require initialization of indicators. There is little code in the module, it is presented below in full. We are not going to analyze every function separately though.

```
#include <Expert\ExpertSignalAdvanced.mqh>
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Inside Bar Price Module                                    |
//| Type=SignalAdvanced                                              |
//| Name=Inside Bar Price Module                                     |
//| ShortName=IB_PM                                                  |
//| Class=CPriceInsideBar                                            |
//| Page=not used                                                    |
//| Parameter=setTpRatio,double,2,TP:SL ratio                        |
//| Parameter=setExpBars,int,10,Expiration after bars number         |
//| Parameter=setOrderOffset,int,5,Offset for open and stop loss     |
//+------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//| Class CPriceInsideBar                                            |
//| Purpose: Class of the generator of price levels for orders based |
//|          on the "inside bar" pattern.                            |
//| Is derived from the CExpertSignalAdvanced class.                 |
//+------------------------------------------------------------------+
class CPriceInsideBar : public CExpertSignalAdvanced
  {
protected:
   double            m_tp_ratio;          //tp:sl ratio
   int               m_exp_bars;          //lifetime of the orders in bars
   double            m_order_offset;      //shift of the opening and Stop Loss levels
   datetime          calcExpiration()  { return(TimeCurrent()+m_exp_bars*PeriodSeconds(m_period)); }
public:
                     CPriceInsideBar();
                    ~CPriceInsideBar();
   void              setTpRatio(double ratio){ m_tp_ratio=ratio; }
   void              setExpBars(int bars)    { m_exp_bars=bars;}
   void              setOrderOffset(int pips){ m_order_offset=m_symbol.Point()*pips;}
   bool              ValidationSettings();
   double            Prices();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPriceInsideBar::CPriceInsideBar()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPriceInsideBar::~CPriceInsideBar()
  {
  }
//+------------------------------------------------------------------+
//| Validation of protected settings                                 |
//+------------------------------------------------------------------+
bool CPriceInsideBar::ValidationSettings(void)
  {
//--- verification of the filter parameters
   if(!CExpertSignal::ValidationSettings())
      return(false);
//--- initial data check
  if(m_tp_ratio<=0)
     {
      printf(__FUNCTION__+": TP Ratio must be greater than zero");
      return(false);
     }
   if(m_exp_bars<0)
     {
      printf(__FUNCTION__+": Expiration must be zero or positive value");
      return(false);
     }
//--- check passed
   return(true);
  }
//+------------------------------------------------------------------+
//| Price levels refreshing                                          |
//+------------------------------------------------------------------+
double CPriceInsideBar::Prices(void)
  {
   double h[2],l[2];
   if(CopyHigh(m_symbol.Name(),m_period,1,2,h)!=2 || CopyLow(m_symbol.Name(),m_period,1,2,l)!=2)
      return(EMPTY_VALUE);
//--- check for inside bar
   if(h[0] >= h[1] && l[0] <= l[1])
     {
      m_order_open_long=h[0]+m_order_offset;
      m_order_stop_long=l[0]-m_order_offset;
      m_order_take_long=m_order_open_long+(m_order_open_long-m_order_stop_long)*m_tp_ratio;
      m_order_expiration_long=calcExpiration();

      m_order_open_short=m_order_stop_long;
      m_order_stop_short=m_order_open_long;
      m_order_take_short=m_order_open_short-(m_order_stop_short-m_order_open_short)*m_tp_ratio;
      m_order_expiration_short=m_order_expiration_long;
      return(0);
     }
   return(EMPTY_VALUE);
  }
//+------------------------------------------------------------------+
```

In the method **Prices()** of this module, the return of the EMPTY\_VALUE result is used to demonstrate to the main signal that there are no available price levels.

### 4.3. Price Module Based on Outside Bar

Outside bar is another popular pattern of Price Action also called "absorption". Outside bar is called so because its High and Low overlap High and Low of the previous bar. Appearance of an outside bar is an indication of the volatility increase followed by a directed price movement.

On Fig. 5 outside bars are marked by red ellipses. Upon detecting an outside bar, the price module generates opening prices of the buystop and sellstop orders on its extrema.

Opening prices are the Stop Loss levels for opposite orders. Take Profit is calculated the same way as in the module discussed above - by multiplying the Stop Loss level by the coefficient specified in the settings. Opening prices and Stop Losses are marked with red horizontal lines and Take Profits with green ones.

![Fig. 5. Illustration of outside bars and price module levels](https://c.mql5.com/2/10/EURUSDH1_OB_EXAMPLE2.png)

Fig. 5. Illustration of outside bars and price module levels

On Fig. 5 outside bars are marked with red ellipses. Horizontal lines reflect the opening prices of pending orders generated by the price module.

In this case, only orders to sell are opened because there is a downtrend. The operation of this module is essentially similar to the previous one. The only difference between its code is only the **Prices()** method, other parts are exactly the same and have the same names. Below is code of **Prices()**.

```
double CPriceOutsideBar::Prices(void)
{
   double h[2],l[2];
   if(CopyHigh(m_symbol.Name(),m_period,1,2,h)!=2 || CopyLow(m_symbol.Name(),m_period,1,2,l)!=2)
      return(EMPTY_VALUE);
//--- check of outside bar
   if(h[0] <= h[1] && l[0] >= l[1])
   {
      m_order_open_long=h[1]+m_order_offset;
      m_order_stop_long=l[1]-m_order_offset;
      m_order_take_long=m_order_open_long+(m_order_open_long-m_order_stop_long)*m_tp_ratio;
      m_order_expiration_long=calcExpiration();

      m_order_open_short=m_order_stop_long;
      m_order_stop_short=m_order_open_long;
      m_order_take_short=m_order_open_short-(m_order_stop_short-m_order_open_short)*m_tp_ratio;
      m_order_expiration_short=m_order_expiration_long;
      return(0);
   }
   return(EMPTY_VALUE);
}
```

### 4.4. Module Performance Test

In the next section three examples of modules are tested to check their individual and then joined work in one Expert and make sure that the system works as designed. As a result, four Experts are generated. [Trading signal module based on the indicator Awesome Oscillator](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ao) working on the H12 timeframe is used as a filter in each of the generated Experts. Price modules are working on H6.

Money management: fixed lot and Trailing Stop are not used. All tests are carried out on the EURUSD quotes from the demo account of the MetaQuotes-Demo server for the period from 2013.01.01 till 2014.06.01. Terminal version: 5.00, build 965 (June, 27 2014).

To keep it short, test results are represented only by balance plots. They are enough to get the Expert's behavior in every particular case.

**4.4.1. Testing the Module Based on the Delta ZigZag Indicator**

![Fig. 6. Balance plot in testing the price module based on Delta ZigZag](https://c.mql5.com/2/10/TesterGraphReport2014_06_26_DDZ__2.png)

Fig. 6. Balance plot in testing the price module based on the "Delta ZigZag" indicator

**4.4.2. Testing the Module Based on the "Inside Bar" Pattern**

![Fig. 7. Balance plot in testing the price module based on the inside bar](https://c.mql5.com/2/10/TesterGraphReport2014_06_26_IB_H6__2.png)

Fig. 7. Balance plot in testing the price module based on the inside bar

Looking at Fig. 7., it should be kept in mind that the purpose of the tests at this stage is to check the code performance, not to get a winning strategy.

**4.4.3. Testing the Module Based on the "Outside Bar" Pattern**

![Fig. 8. Balance plot in testing the price module based on the outside bar](https://c.mql5.com/2/10/TesterGraphReport2014_07_03_OB_H6__3.png)

Fig. 8. Balance plot in testing the price module based on the outside bar

**4.4.4. Testing Joint Work of Developed Price Modules**

Taking into consideration previous test results, price modules based on DeltaZigZag, inside bar and outside bar received weight coefficients of 1, 0.5 and 0.7 respectively. Weight coefficients define priorities of the price modules.

![Fig. 9. Balance plot in testing the Expert with three price modules](https://c.mql5.com/2/10/TesterGraphReport2014_07_03_ALL__2.png)

Fig. 9. Balance plot in testing the Expert with three price modules

At every stage of testing, all performed operations on the price plots were carefully analyzed. Strategy errors and deviations were not elicited. All the suggested programs are attached to this article and you can test them yourself.

### 5\. Instruction for Use

We are going to consider the stages of the Expert development using the developed extension on the example of Expert with three price modules and Awesome Oscillator as a filter.

Before the generation starts, make sure that the header files "CExpertAdvanced.mqh" and "CExpertSignalAdvanced.mqh" are in the catalog of the MetaTrader 5 terminal in the folder MQL5/Include/Expert and files of the price modules are in the folder MQL5/Include/Expert/MySignal. Before launching the Expert, ensure that compiled indicators are in the MQL5/Indicators folder. In this case it is the file "DeltaZigZag.ex5". In the attached archive, all files are in their places. All it takes is unpacking this archive into a folder with MetaTrader 5 terminal and confirm merging catalogs.

**5.1. Expert Generation**

To start Expert generation, select "File"->"New" in the MetaEditor.

![Fig. 10. Creating a new Expert using the MQL5 Wizard](https://c.mql5.com/2/12/Fig10.png)

Fig. 10. Creating a new Expert using the MQL5 Wizard

In the new window select "Expert Advisor (generate)" and then press "Next".

![Fig. 11. Generating an Expert using the MQL5 WIzard](https://c.mql5.com/2/12/Fig11.png)

Fig. 11. Generating an Expert using the MQL5 WIzard

It will bring up a window where you can specify the name. In this particular example, the name of the Expert is "TEST\_EA\_AO\_DZZ\_IB\_OB", parameters Symbol and TimeFrame have default values. Then click "Next".

![Fig. 12. Common parameters of the Expert Advisor](https://c.mql5.com/2/12/Fig12.png)

Fig. 12. General properties of the Expert Advisor

In the appeared window add include modules one by one pressing "Add". The whole process is presented below.

**PLEASE NOTE!** When you are adding modules, at first include all signal modules (inheritors **CExpertSignal**) and then price modules (inheritors **CExpertSignalAdvanced**). The result of disrupting this order of including modules is going to be unpredictable.

So, we start with including modules of signals from Awesome Oscillator. It is the only filter in this example.

![Fig. 13. Including the module of signals from Awesome Oscillator](https://c.mql5.com/2/12/Fig13.png)

Fig. 13. Including the module of signals from Awesome Oscillator

Include required price modules in a random order after all signal modules have been included. There are three of them in this example.

![Fig. 14. Including the module of signals from DeltaZZ Price Module](https://c.mql5.com/2/12/Fig14.png)

Fig. 14. Including the module of signals from DeltaZZ Price Module

![Fig. 15. Including the module of signals from Inside Bar Price Module](https://c.mql5.com/2/12/Fig15.png)

Fig. 15. Including the module of signals from Inside Bar Price Module

![Fig. 16. Including the module of signals from Outside Bar Price Module](https://c.mql5.com/2/12/Fig16.png)

Fig. 16. Including the module of signals from Outside Bar Price Module

After all modules were added, the window will look as follows:

![Fig. 17. List of included modules of trading signals](https://c.mql5.com/2/12/Fig17.png)

Fig. 17. List of included modules of trading signals

Value of the "Weight" parameter for the price module defines its priority.

After the "Next" button has been pressed, the Wizard suggests selecting money management modules and Trailing Stop modules. Select one of those or leave as it is, pressing "Next" or "Ready".

Pressing "Ready" we shall receive generated code of Advisor.

**5.2. Editing Generated Code**

So, we have code.

1\. Find string in the very beginning

```
#include <Expert\Expert.mqh>
```

and edit it so it looks like

```
#include <Expert\ExpertAdvanced.mqh>
```

It is required to include the files of developed classes.

2\. Then find the string

```
CExpert ExtExpert;
```

and change for

```
CExpertAdvanced ExtExpert;
```

This will change the Expert standard class for its descendant with required functions.

3\. Now find the string

```
CExpertSignal *signal=new CExpertSignal;
```

and change for

```
CExpertSignalAdvanced *signal=new CExpertSignalAdvanced;
```

This is the way to change the standard class of the main signal for its descendant with required functions.

4\. Find the string implementing the addition of the first price module to the **m\_filters** array of the main signal. In this example it looks like:

```
signal.AddFilter(filter1);
```

Before this we insert the string

```
signal.CalcPriceModuleIndex();
```

This is required for the main signal to recognize up to which index in the **m\_filters** array there are signal module pointers and starting from which there are price module pointers.

Finding the right position for inserting the specified string may cause difficulty. Use the number after the word "filter" as a reference point. It will simplify the search and allow not to miss the right position. The MQL5 Wizard names included modules automatically in their order. The first module is called **filter0**, the second one – **filter1**, the third – **filter2** etc. In our case there is only one signal module. Therefore, the first included price module is number two and we need to search for the string "signal.AddFilter(filter1);" for adding the filter as numbering in the code starts with zero. The illustration is on Fig. 18:

![Fig. 18. Module names in code according to the order of inclusion](https://c.mql5.com/2/12/Fig18.png)

Fig. 18. Module names in code according to the order of inclusion

5\. This part is not compulsory. Through introduced changes, Expert parameters responsible for opening price indentations, Stop Losses, Take Profits and order expiration time lost their use. To make code more compact, you can delete the following strings:

```
input double Signal_PriceLevel            =0.0;                    // Price level to execute a deal
input double Signal_StopLevel             =50.0;                   // Stop Loss level (in points)
input double Signal_TakeLevel             =50.0;                   // Take Profit level (in points)
input int    Signal_Expiration            =4;                      // Expiration of pending orders (in bars)
```

Compilation error encountered after deleting the above strings takes us to the next group of strings to be deleted:

```
   signal.PriceLevel(Signal_PriceLevel);
   signal.StopLevel(Signal_StopLevel);
   signal.TakeLevel(Signal_TakeLevel);
   signal.Expiration(Signal_Expiration);
```

After the latter have been deleted, compilation is going to be successful.

Comments and explanations of editing can be found in the attached code of the Expert "TEST\_EA\_AO\_DZZ\_IB\_OB". Strings of the code that could be deleted have accompanying comments.

### Conclusion

In this article we significantly expanded the application area of the MQL5 Wizard. Now it can be used for development optimization of automatic trading systems that require placing orders, Stop Losses and Take Profits at different price levels irrespective to the current price.

Generated Experts may include a set of price modules calculating parameters of the orders being sent. The most suitable parameter set is selected out of the available ones. Preferences are specified in the settings. It allows using many various entry points with maximum efficiency. This approach makes Experts selective. If the direction is known but the entry point is undefined, then the Expert will wait for it to appear.

Introducing standard compatible modules responsible for search of price levels is a significant advantage and is meant to simplify development of Experts. Though at the moment there are only three modules, their number will no doubt increase in the future. If you have found this article useful, please suggest algorithms of price modules operation in the comments. Interesting ideas will be implemented in code.

This article also features a method for further extension of Expert capabilities developed in the Wizard. Inheritance is the optimal way of introducing changes.

Users with no programming skills though capable to edit code following instructions, will have an opportunity to create more advanced Experts based on the available models.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/987](https://www.mql5.com/ru/articles/987)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/987.zip "Download all attachments in the single ZIP archive")

[mql5\_library\_extension.zip](https://www.mql5.com/en/articles/download/987/mql5_library_extension.zip "Download mql5_library_extension.zip")(715.33 KB)

[expertsignaladvanced.mqh](https://www.mql5.com/en/articles/download/987/expertsignaladvanced.mqh "Download expertsignaladvanced.mqh")(22.09 KB)

[expertadvanced.mqh](https://www.mql5.com/en/articles/download/987/expertadvanced.mqh "Download expertadvanced.mqh")(6.61 KB)

[pricedeltazz.mqh](https://www.mql5.com/en/articles/download/987/pricedeltazz.mqh "Download pricedeltazz.mqh")(22.93 KB)

[priceinsidebar.mqh](https://www.mql5.com/en/articles/download/987/priceinsidebar.mqh "Download priceinsidebar.mqh")(9.45 KB)

[priceoutsidebar.mqh](https://www.mql5.com/en/articles/download/987/priceoutsidebar.mqh "Download priceoutsidebar.mqh")(9.47 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39002)**
(3)


![Jinping Ou](https://c.mql5.com/avatar/2015/2/54EBBF95-9D3F.png)

**[Jinping Ou](https://www.mql5.com/en/users/xiaoping)**
\|
26 Feb 2015 at 03:14

Up and at 'em!


![Konstantin Katulkin](https://c.mql5.com/avatar/avatar_na2.png)

**[Konstantin Katulkin](https://www.mql5.com/en/users/globax8)**
\|
16 Feb 2019 at 20:06

Here is the following error: failed instant buy 0.20 [EURUSD](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: technical analysis") at 1.07971 sl: 1.07959 tp: 1.08053 \[Invalid stops\]

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
19 Apr 2022 at 08:45

Hi, thank you for your code and perspective on employing a different logic to the CSignal class. What would the opposite for this be?

```
   double direction=Direction();
//--- prohibitive signal
   if(direction==EMPTY_VALUE)
      return(false);
```

To allow signals to be taken on each opportunity regardless of running positions?

![Random Forests Predict Trends](https://c.mql5.com/2/11/Random_Forest_MetaTrader5.png)[Random Forests Predict Trends](https://www.mql5.com/en/articles/1165)

This article considers using the Rattle package for automatic search of patterns for predicting long and short positions of currency pairs on Forex. This article can be useful both for novice and experienced traders.

![Liquid Chart](https://c.mql5.com/2/11/800px-Wiki.png)[Liquid Chart](https://www.mql5.com/en/articles/1208)

Would you like to see an hourly chart with bars opening from the second and the fifth minute of the hour? What does a redrawn chart look like when the opening time of bars is changing every minute? What advantages does trading on such charts have? You will find answers to these questions in this article.

![Programming EA's Modes Using Object-Oriented Approach](https://c.mql5.com/2/12/Expert_Advisor_modes_programming_img.png)[Programming EA's Modes Using Object-Oriented Approach](https://www.mql5.com/en/articles/1246)

This article explains the idea of multi-mode trading robot programming in MQL5. Every mode is implemented with the object-oriented approach. Instances of both mode classes hierarchy and classes for testing are provided. Multi-mode programming of trading robots is supposed to take into account all peculiarities of every operational mode of an EA written in MQL5. Functions and enumeration are created for identifying the mode.

![MQL5 Programming Basics: Global Variables of the Terminal](https://c.mql5.com/2/12/MQL5_Basics_Global_variables_terminal_MetaTrader5.png)[MQL5 Programming Basics: Global Variables of the Terminal](https://www.mql5.com/en/articles/1210)

This article highlights object-oriented capabilities of the MQL5 language for creating objects facilitating work with global variables of the terminal. As a practical example I consider a case when global variables are used as control points for implementation of program stages.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/987&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068710108604595378)

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