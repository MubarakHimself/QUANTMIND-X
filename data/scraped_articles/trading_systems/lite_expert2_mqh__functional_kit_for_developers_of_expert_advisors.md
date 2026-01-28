---
title: Lite_EXPERT2.mqh: Functional Kit for Developers of Expert Advisors
url: https://www.mql5.com/en/articles/1380
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:46:44.254402
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/1380&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070553349654255602)

MetaTrader 4 / Trading systems


### Introduction

In the articles of the series "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization" [1](https://www.mql5.com/en/articles/1516 "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization"), [2](https://www.mql5.com/en/articles/1517), [3](https://www.mql5.com/en/articles/1521), [4](https://www.mql5.com/en/articles/1523), [5](https://www.mql5.com/en/articles/1525), [6](https://www.mql5.com/en/articles/1535), [7](https://www.mql5.com/en/articles/1535), I introduced the novice EA developers with my approach to writing an Expert Advisor that allows you to easily transform trading strategies into a very simple and quickly implemented program code using ready-made custom trade functions of the Lite\_EXPERT1.mqh. It is quite natural that in order to ensure maximum simplicity of information, the number of custom functions used in this file was minimal, being sufficient to, so to say, get the grasp of it without any trouble.

However, the functionality offered by that file is not quite sufficient for working with the program code at broader scale. Lite\_EXPERT2.mqh was written with this objective in view and contains a more universal set of custom functions which is yet more complex for a first-time user. It is assumed that the reader is already familiar with the information provided in the article ["Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization 1"](https://www.mql5.com/en/articles/1516 "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization") and considers this new article as the next step in further development and improvement of his skills necessary to use such custom functions.

### The Lite\_EXPERT2.mqh File Content

Generally speaking, all the functions available in Lite\_EXPERT2.mqh can be shown in the form of the flowchart as follows:

![Description of functions](https://c.mql5.com/2/13/lite_expert2.png)

In addition to functions, Lite\_EXPERT2.mqh contains the integer variable int LastTime declared in global scope and used in all trading functions, as well as the integer variable extern int Slippage\_ for slippage in points, which will be visible from the external parameters of the Expert Advisor.

### 1\. Position Opening Functions

All functions of this block can be divided into two large groups. Functions of the first group have "Open" at the beginning of their names, while function names of the second group start with dOpen. Functions of the first group use relative distances from the position opening price expressed in points as the external variables of the Stop Loss and Take Profit, i.e. they are represented by integer variables.

```
bool OpenBuyOrder1_
        (bool& BUY_Signal, int MagicNumber, datetime TimeLevel,
                           double Money_Management, int Margin_Mode,
                                         int STOPLOSS, int TAKEPROFIT)

bool OpenSellOrder1_
        (bool& SELL_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                          int STOPLOSS, int TAKEPROFIT)

bool OpenBuyOrder2_
        (bool& BUY_Signal, int MagicNumber, datetime TimeLevel,
                           double Money_Management, int Margin_Mode,
                                          int STOPLOSS, int TAKEPROFIT)

bool OpenSellOrder2_
        (bool& SELL_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                          int STOPLOSS, int TAKEPROFIT)
```

The Stop Loss and Take Profit in the functions of the second group are represented by floating-point variables. In this case, values of those variables are absolute values of the corresponding orders as obtained from the price chart. This makes writing the code of an Expert Advisor that serves a specific purpose much more convenient.

```
bool dOpenBuyOrder1_
        (bool& BUY_Signal, int MagicNumber, datetime TimeLevel,
                           double Money_Management, int Margin_Mode,
                                  double dSTOPLOSS, double dTAKEPROFIT)

bool dpenSellOrder1_
        (bool& SELL_Signal, int MagicNumber, datetime TimeLevel,
                           double Money_Management, int Margin_Mode,
                                  double dSTOPLOSS, double dTAKEPROFIT)

bool dOpenBuyOrder2_
        (bool& BUY_Signal, int MagicNumber, datetime TimeLevel,
                           double Money_Management, int Margin_Mode,
                                   double dSTOPLOSS, double dTAKEPROFIT)

bool dOpenSellOrder2_
        (bool& SELL_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                   double dSTOPLOSS, double dTAKEPROFIT)
```

All functions whose names end in "1\_" are intended for brokers that allow setting Stop Loss and Take Profit right when executing a deal. Functions whose names contain "2\_" are used for brokers who allow setting these orders for positions that are already open.

In contrast to the functions featured in Lite\_EXPERT1.mqh, these eight functions have two new external variables (datetime TimeLevel and int Margin\_Mode). So the first thing we should do is have a closer look at them. The value of the TimeLevel variable represents a certain time limit after the execution of the current deal. All trading functions of this file will not open any new positions or orders with the current magic number till the specified time limit is reached. The value of this external variable when executing a deal is saved in a global variable on the hard drive of the computer and it will therefore not be lost when restarting the trading terminal or the Windows operating system. The most basic use of this variable is to block the reopening of a position or setting a pending order on the same bar.

```
//----
   static datetime TimeLevel;
   TimeLevel = Time[0] + Period() * 60;

   //----
   if (!OpenBuyOrder1_
        (BUY_Signal, MagicNumber, TimeLevel,
                           Money_Management, Margin_Mode,
                                         STOPLOSS, TAKEPROFIT))
                                                          return(-1);
```

It is assumed that the TimeLevel static variable will be initialized in the same block as the BUY\_Signal variable. Now, if the OpenBuyOrder1\_() function opens a position, it will save the TimeLevel variable value in the global variable on the hard drive. And the functions of the Lite\_EXPERT2.mqh file that open positions or pending orders will not open any order with that magic number until the last quote time becomes greater than or equal to this value.

So, basically, the trading functions of this file do not zero out the external variable of the bool& BUY\_Signal type by reference as this is not necessary. Zeroing out will only be performed if the TimeLevel variable is initialized to "-1"! If we set the value of this variable equal to zero, the position opening function will not zero out or save anything and will open positions based on signals of the BUY\_Signal external variable at any time when there are no orders with the magic number equal to the value of the int MagicNumber external variable.

The string name that is used for a global variable saved on the hard drive is generated for testing and optimization using the following formula:

```
string G_Name_ = "TimeLevel", "_", AccountNumber(), "_",
                                "_Test_", OrderSymbol(), "_", OrderMagicNumber());
```

In other cases:

```
string G_Name_ = "TimeLevel", "_", AccountNumber(),
                                     "_", OrderSymbol(), "_", OrderMagicNumber());
```

This name must not be used for other global variables!

Trading functions available in Lite\_EXPERT1.mqh use only one method of lot calculation (MM based on free margin). This method of lot size calculation may not be suitable for every strategy. This issue has been taken into account in the trading functions provided in Lite\_EXPERT2.mqh and the lot size calculation method is determined using the int Margin\_Mode external variable. In determining the lot size calculation method, the Margin\_Mode external variable may take on values from zero to five:

- 0 - MM based on free margin
- 1 - MM based on the account balance
- 2 - MM based on losses on free margin
- 3 - MM based on losses on the account balance
- 4 - minimum lot between 0 and 2
- 5 - minimum lot between 1 and 3
- by default - MM based on free margin

Please bear in mind that if you use the second or third lot size calculation method and your Stop Loss is dynamic, i.e. it varies from deal to deal, you should take into account the boundary values of the Stop Loss and MM. E.g. if the variables have the following values: Money\_Management = 0.1, Margin\_Mode = 3 and int STOPLOSS = 100 (five decimal places), the Expert Advisor will use the entire deposit in opening a position!

Just in case, I will once again explain these two variants for values of the Margin\_Mode variable (2 and 3). Here, the lot size calculation function takes either the free margin or account balance value and multiplies it by the value of the Money\_Management variable. The resulting value represents the amount of losses that may arise if the position will be closed at the Stop Loss level! Those losses do not depend on the Stop Loss size. Thus, the lot size calculation function determines the position volume based on the Stop Loss size so that losses from the Stop Loss are hard-set!

Where the Margin\_Mode variable value is equal to 4 or 5, the trading functions can calculate the lot size using two variants at the same time and choose the smallest value. For example, if Margin\_Mode = 5, a trading function will calculate the lot size based on the account balance and based on losses on the account balance and will then use the smallest value.

Once again, I would like to mention here that if the value of the Money\_Management variable is negative, all these functions will disregard the Margin\_Mode variable values and use the values of the Money\_Management variable as the lot size. When using such values, the minus sign will be discarded and the value itself will be rounded to the nearest standard value which cannot be greater than the available one. In calculating the lot size, these eight functions always check the free margin to be sufficient for execution of a certain deal and reduce the calculated lot size to the permissible value, if necessary.

### 2\. Functions for Placing Pending Orders

This is the second, large group of functions divided into two large groups, similarly to the previous ones.

```
bool OpenBuyLimitOrder1_
        (bool& Order_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                           int STOPLOSS, int TAKEPROFIT,
                                              int LEVEL, datetime Expiration)
bool OpenBuyStopOrder1_
        (bool& Order_Signal, int MagicNumber, datetime TimeLevel,
                             double Money_Management, int Margin_Mode,
                                           int STOPLOSS, int TAKEPROFIT,
                                               int LEVEL, datetime Expiration)
bool OpenSellLimitOrder1_
        (bool& Order_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                           int STOPLOSS, int TAKEPROFIT,
                                               int LEVEL, datetime Expiration)
bool OpenSellStopOrder1_
        (bool& Order_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                           int STOPLOSS, int TAKEPROFIT,
                                               int LEVEL, datetime Expiration)
```

```
bool dOpenBuyLimitOrder1_
        (bool& Order_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                    double dSTOPLOSS, double dTAKEPROFIT,
                                           double dLEVEL, datetime Expiration)
bool dOpenBuyStopOrder1_
        (bool& Order_Signal, int MagicNumber, datetime TimeLevel,
                             double Money_Management, int Margin_Mode,
                                    double dSTOPLOSS, double dTAKEPROFIT,
                                            double dLEVEL, datetime Expiration)
bool dOpenSellLimitOrder1_
        (bool& Order_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                    double dSTOPLOSS, double dTAKEPROFIT,
                                            double dLEVEL, datetime Expiration)
bool dOpenSellStopOrder1_
        (bool& Order_Signal, int MagicNumber, datetime TimeLevel,
                            double Money_Management, int Margin_Mode,
                                    double dSTOPLOSS, double dTAKEPROFIT,
                                            double dLEVEL, datetime Expiration)
```

Everything that was said earlier regarding the previous functions is also true for these eight functions. There are only two facts that can be considered as exceptions. It is basically impossible to calculate the lot size based on what will be happening when the order is being placed. And if the lot size is calculated at the point of placing a pending order, which is how it is done, one can hardly expect the calculation to be in strict accordance with the Money\_Management variable value. For the same reason, these functions do not check if there are sufficient funds for the lot size.

### 3\. Functions for Closing Open Positions

This block only consists of four functions

```
bool CloseBuyOrder1_(bool& CloseStop, int MagicNumber)

bool CloseSellOrder1_(bool& CloseStop, int MagicNumber)

bool CloseAllBuyOrders1_(bool CloseStop)

bool CloseAllSellOrders1_(bool CloseStop)
```

These functions are quite simple and do not require any additional explanations. The first two functions close positions with the specified magic numbers by the symbol from the chart on which the Expert Advisor is running. The other two functions close all open positions available.

### 4\. Functions for Deleting Pending Orders

The list of functions in this section consists of only two basic functions

```
bool CloseOrder1_(bool& CloseStop, int cmd, int MagicNumber)
bool CloseAllOrders1_(bool CloseStop, int cmd)
```

the external variables of which contain the new variable - int cmd. Its values are provided below:

- OP\_BUYLIMIT 2 Pending order BUY LIMIT
- OP\_SELLLIMIT 3 Pending order SELL LIMIT
- OP\_BUYSTOP 4 Pending order BUY STOP
- OP\_SELLSTOP 5 Pending order SELL STOP

### 5\. Position Modification and Trailing Stop Functions

This block includes three function groups:

1) position modifiers

```
bool dModifyOpenBuyOrder_
       (bool& Modify_Signal, int MagicNumber,
                     datetime ModifyTimeLevel_, double dSTOPLOSS, double dTAKEPROFIT)

bool dModifyOpenSellOrder_
       (bool& Modify_Signal, int MagicNumber,
                     datetime ModifyTimeLevel_, double dSTOPLOSS, double dTAKEPROFIT)

bool dModifyOpenBuyOrderS (bool& Modify_Signal, double dSTOPLOSS, double dTAKEPROFIT)

bool dModifyOpenSellOrderS(bool& Modify_Signal, double dSTOPLOSS, double dTAKEPROFIT)
```

All four functions of this group use absolute values of the Stop Loss and Take Profit on a price chart scale and are represented by floating point variables. Everything written with respect to the variables of the first and second group is true and applies to the variables of this group, as well.

Please note that the datetime ModifyTimeLevel\_ variable value is only used in the first two functions of this group and nowhere else! This variable is absent altogether from the last two functions. To be able to call one of the last two functions in the code of the Expert Advisor, you need to first select an order of OP\_BUY or OP\_SELL type to continue working with them, using the OrderSelect() function! These functions are intended for working with open positions that do not have magic numbers. The formulas for the string name of the global variable for the purpose of saving the ModifyTimeLevel\_ variable value on the hard drive are as follows:

```
string G_Name_ = "ModifyTimeLevel_", "_", AccountNumber(),
                                          "_", "_Test_", Symbol(), "_", MagicNumber;
```

```
string G_Name_ = "ModifyTimeLevel_", "_", AccountNumber(),
                                                    "_", Symbol(), "_", MagicNumber;
```

2) Trailing Stops

```
bool Make_BuyTrailingStop_
             (bool& TreilSignal, int MagicNumber, datetime TrailTimeLevel, int TRAILINGSTOP)

bool Make_SellTrailingStop_
             (bool& TreilSignal, int MagicNumber, datetime TrailTimeLevel, int TRAILINGSTOP)

bool dMake_BuyTrailingStop_
        (bool& TreilSignal, int MagicNumber, datetime TrailTimeLevel_, double dTRAILINGSTOP)

bool dMake_SellTrailingStop_
        (bool& TreilSignal, int MagicNumber, datetime TrailTimeLevel_, double dTRAILINGSTOP)
```

This group contains four classical Trailing Stops for forcing the Stop Loss against the current price. Two of them have Trailing Stop value in points relative to the current price as the external parameter, while the other two use the Trailing Stop absolute value for the same purpose. Like the previous functions, these four functions use time limits in the form of the TrailTimeLevel and TrailTimeLevel\_ variables with the similar value. If you intend to change Trailing Stops on every tick, these variables should be set to zero.

3) and one more group of four position modifiers

```
bool BuyStoplossCorrect
           (int MagicNumber, int ExtPointProfit, int StoplossProfit)

bool SellStoplossCorrect
           (int MagicNumber, int ExtPointProfit, int StoplossProfit)

bool AllBuyStoplossCorrects (int ExtPointProfit, int StoplossProfit)

bool AllSellStoplossCorrects(int ExtPointProfit, int StoplossProfit)
```

that perform one-time changes of Stop Loss levels. The first two functions check for profit in points with respect to an open position with the fixed magic number and if it is not less than the value of the ExtPointProfit variable, the Stop Loss is moved to the current price at the StoplossProfit distance from the position opening price.

The last two functions of this group track the profit of all open positions for the current symbol with any magic number and perform one-time changes of their corresponding Stop Loss levels.

### 6\. Additional Functions

This is the last and probably the largest group of functions in the Lite\_EXPERT2.mqh file. Many functions of this group are used as additional ones within the code of other functions described above and therefore are, in many cases, of little practical interest. So I will limit myself here to the review of the most essential of them.

First of all, I would like to turn your attention to the TimeLevelGlobalVariableDel() function:

```
void TimeLevelGlobalVariableDel(string symbol, int MagicNumber)
```

After testing and optimization, this function deletes all global variables that were generated by the trading functions featured in the file and saved on the hard drive of your computer. This function should be called in the Expert Advisor deinitialization block, e.g. as follows:

```
//+X================================================================X+
//| Custom Expert deinitialization function                          |
//+X================================================================X+
int deinit()
  {
//----+
    TimeLevelGlobalVariableDel(Symbol(), 1);
    TimeLevelGlobalVariableDel(Symbol(), 2);
    TimeLevelGlobalVariableDel(Symbol(), 3);
    TimeLevelGlobalVariableDel(Symbol(), 4);
    //---- Completing deinitialization of the Expert Advisor
    return(0);
//----+
  }
```

If this is not done, the global variables with the latest time values remaining after testing or optimization will block the testing and optimization of the Expert Advisor!!!

It is often the case that chart time frames which can only be selected from the standard series of values are used as external variables of the Expert Advisor. One can always make a mistake and set the wrong value when doing it manually. In such cases, you can do the check using the TimeframeCheck() function,

```
void TimeframeCheck(string TimeframeName, int Timeframe)
```

that checks time frame values in the external variables when initializing the Expert Advisor.

All calculations that are made in the Expert Advisor for any price level are initially based on price chart values. And it may not always be quite clear what prices are shown in the charts - ASK or BID. So the values obtained from price charts need to be adjusted for that difference prior to be used in trading functions. This can be done using the dGhartVelueCorrect() function:

```
double dGhartVelueCorrect(int cmd, double& GhartVelue)
```

The purpose of the cmd variable is absolutely identical to the description provided in the MetaEditor Help. When calling this function, the external parameter GhartVelue can only be represented by a variable whose value is changed to the adjusted one by reference. For example:

```
//---- get the GhartVelue_ price level
   double GhartVelue_ = Low[0];
   //---- correct the GhartVelue_ price level
   dGhartVelueCorrect(OP_BUY, GhartVelue_);
```

There are two more functions for selecting orders to be able to continue using them further:

```
bool OrderSelect_(string symbol, int cmd, int MagicNumber, int pool)
bool Order_Select(string symbol, int MagicNumber)
```

The pool variable can take on two values only: MODE\_TRADES - the order is selected from open and pending orders, and MODE\_HISTORY - the order is selected from closed and deleted orders. If successful, the functions return true, otherwise false.

In certain cases, you may need to use MarginCheck(), the function that checks the lot size against the funds available in the account and, if necessary, decreases the lot size to be adequate for the available funds:

```
bool MarginCheck(string symbol, int Cmd, double& Lot)
```

The Cmd parameter in this function can only take on two values: OP\_BUY and OP\_SELL. If the calculation is successful, the function will return true. In case of errors in the function operation, it returns false.

Sometimes you may want to calculate the lot size. For this purpose, we have two more functions:

```
double BuyLotCount(double Money_Management, int Margin_Mode, int STOPLOSS)
double SellLotCount(double Money_Management, int Margin_Mode, int STOPLOSS)
```

If there are no errors, the functions return the lot value, otherwise -1.

And there are three more functions for those who like to wrap an indicator code in the code of Expert Advisors:

```
int IndicatorCounted_(int Number, string symbol, int timeframe)

int ReSetAsIndexBuffer(string symbol, int timeframe, double& array[])

int ReSetAsIndexBuffer_(string symbol, int timeframe, double& Array[], int ArrSize)
```

The first function is equivalent to the IndicatorCounted() function operating in EAs. The external variable Number represents the function call number in the code of the Expert Advisor (one number per indicator).

The purpose of the second function is to convert a data array declared in global scope to an indicator buffer analog. So the function synchronizes the elements of the array whose name represents the variable double& array\[\] with the corresponding time series arrays. This function must be called in the start() function, outside the range of loop operators iterating over chart bars.

The third function is a complete analog of the second function but the number of elements in the Array\[\] array is derived from the limited value of the ArrSize variable. In other words, in this case the array contains the ArrSize number of only the last, most recent bars. This function may be more useful than the previous one in many cases where in the indicator code values are only added to such array and only the ArrSize number of the last values is used in the code of the Expert Advisor.

And finally, the last two functions:

```
bool IsNewBar(int Number, string symbol, int timeframe)

bool MinBarCheck(string symbol, int timeframe, int MinBarTotal)
```

The IsNewBar() function returns True at the point of time when the bar is changed in the corresponding time series arrays. In all other cases, this function returns False. The Number variable represents the function call number in the code of the Expert Advisor. The MinBarCheck() function compares the number of bars of the relevant chart against the MinBarTotal variable value. If the number of bars is smaller, it returns False. This function is used in order not to allow an Expert Advisor to trade if the number of bars required for the calculation is not sufficient.

### Conclusion

That is basically the entire list of the most needed Lite\_EXPERT2.mqh functions which represent the sufficient minimum for convenient and efficient strategy tester writing in MQL4. In the next articles of this series, I will provide specific examples illustrating the use of the above file functions in EAs.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1380](https://www.mql5.com/ru/articles/1380)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1380.zip "Download all attachments in the single ZIP archive")

[Lite\_EXPERT2.mqh](https://www.mql5.com/en/articles/download/1380/Lite_EXPERT2.mqh "Download Lite_EXPERT2.mqh")(186.06 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Simple Trading Systems Using Semaphore Indicators](https://www.mql5.com/en/articles/358)
- [Averaging Price Series for Intermediate Calculations Without Using Additional Buffers](https://www.mql5.com/en/articles/180)
- [Creating an Indicator with Multiple Indicator Buffers for Newbies](https://www.mql5.com/en/articles/48)
- [Creating an Expert Advisor, which Trades on a Number of Instruments](https://www.mql5.com/en/articles/105)
- [The Principles of Economic Calculation of Indicators](https://www.mql5.com/en/articles/109)
- [Practical Implementation of Digital Filters in MQL5 for Beginners](https://www.mql5.com/en/articles/32)
- [Custom Indicators in MQL5 for Newbies](https://www.mql5.com/en/articles/37)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39147)**
(2)


![Rodrigo Olivares](https://c.mql5.com/avatar/avatar_na2.png)

**[Rodrigo Olivares](https://www.mql5.com/en/users/rodlivar)**
\|
24 Nov 2013 at 11:40

Nikolay I wanted to ask you if you have this [functional](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") kit for MQL5?. Im a beginner in MQL5 and when I I first start I was using your functions to program expert advisors in MQL4.

It was very convenient. It is possible that you published an article with this functions in MQL5 or give a way to construct those functions in MQL5. Please I would appreciated any

kind of Help. Thanks a lot for your help in MQL4.

PD Is possible to construct a reverse function or is better to close an order and open another to reverse?

![Josep Marsa](https://c.mql5.com/avatar/2014/12/548CACC2-C53B.jpg)

**[Josep Marsa](https://www.mql5.com/en/users/daviunx)**
\|
30 Jan 2015 at 20:11

Great job! I'll test it.


![MQL5 Cookbook: Saving Optimization Results of an Expert Advisor Based on Specified Criteria](https://c.mql5.com/2/0/avatar__7.png)[MQL5 Cookbook: Saving Optimization Results of an Expert Advisor Based on Specified Criteria](https://www.mql5.com/en/articles/746)

We continue the series of articles on MQL5 programming. This time we will see how to get results of each optimization pass right during the Expert Advisor parameter optimization. The implementation will be done so as to ensure that if the conditions specified in the external parameters are met, the corresponding pass values will be written to a file. In addition to test values, we will also save the parameters that brought about such results.

![MetaTrader AppStore Results for Q3 2013](https://c.mql5.com/2/0/avatar3.png)[MetaTrader AppStore Results for Q3 2013](https://www.mql5.com/en/articles/769)

Another quarter of the year has passed and we have decided to sum up its results for MetaTrader AppStore - the largest store of trading robots and technical indicators for MetaTrader platforms. More than 500 developers have placed over 1 200 products in the Market by the end of the reported quarter.

![Marvel Your MQL5 Customers with a Usable Cocktail of Technologies!](https://c.mql5.com/2/0/cocktails.png)[Marvel Your MQL5 Customers with a Usable Cocktail of Technologies!](https://www.mql5.com/en/articles/728)

MQL5 provides programmers with a very complete set of functions and object-oriented API thanks to which they can do everything they want within the MetaTrader environment. However, Web Technology is an extremely versatile tool nowadays that may come to the rescue in some situations when you need to do something very specific, want to marvel your customers with something different or simply you do not have enough time to master a specific part of MT5 Standard Library. Today's exercise walks you through a practical example about how you can manage your development time at the same time as you also create an amazing tech cocktail.

![MQL5 Wizard: How to Teach an EA to Open Pending Orders at Any Price](https://c.mql5.com/2/0/Pending_Orders_Trading_MQ5_Wizard_signals.png)[MQL5 Wizard: How to Teach an EA to Open Pending Orders at Any Price](https://www.mql5.com/en/articles/723)

The article describes a method of modifying the code of a trading signal module for the implementation of the functionality allowing you to set pending orders at any distance from the current price: it may be the Close or Open price of the previous bar or the value of the moving average. There are plenty of options. Important is that you can set any opening price for a pending order. This article will be useful to traders who trade with pending orders.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/1380&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070553349654255602)

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