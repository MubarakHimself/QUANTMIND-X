---
title: Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part IV)
url: https://www.mql5.com/en/articles/1523
categories: Trading Systems
relevance_score: 12
scraped_at: 2026-01-22T17:15:56.014495
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/1523&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049005515909997338)

MetaTrader 4 / Trading systems


### Introduction

In my previous articles No. [2](https://www.mql5.com/en/articles/1517) and [3](https://www.mql5.com/en/articles/1521) I described the basics of backtesting. In my opinion, the main purpose of backesting is a qualitative analysis of an EA behavior within a certain period of time provided that the EA parameters are changed from time to time. The strategy of using optimization results in this process will be some strict rule of selecting parameters form all variants received as a result of each optimization. For example, a variant with maximal profitability and minimal drawdown.

Such a variant is selected among all variants of a separate optimization regardless of what we had in previous optimizations. If you detect that such a strategy is profitable in some time periods and in some periods is near a break-even level, we can conclude that your trading system with such a strategy of using optimized parameters is likely to be profitable in future in a certain time period.

This is a perfect case. But what should we do if an EA with a seemingly very rational operation logics appears to be hopelessly lossmaking in half of all backtesting cases? Actually, strategies based on EA parameters selection inside each optimization separately is not the only possible method. It is more interesting to conduct statistical analysis and compare EA parameters from different optimizations.

Moreover, optimizations in backtesting take much time and it is not reasonable to conduct these optimizations only for the backtesting itself. For example, together with optimizations we can save results of all profitable optimization runs into one file in the form of a table that can be easily processed by means of statistic analyzing of tables available in programs like Microsoft Excel. MetaTrader 4 terminal offers the possibility of saving optimization results in the form of HTML table:

![](https://c.mql5.com/2/16/1_1.png)

that can be easily uploaded to Microsoft Excel. But for processing only final reports of each optimization run are presented in the form of a table. External parameters of an Expert Advisor are not included into the table:

![](https://c.mql5.com/2/16/2_2.png)

Moreover it is the result of only one optimization. And for analysis we need results of all optimizations recorded into one file. Actually all this can be implemented by MQL4 means, so in this article I would like to offer my own variant of the solution.

### Recording EA Optimization Results in the Form of HTML Table in One File

To implement such recording into a file, I have written the following function:

```
void Write_Param_htm
(
bool ToResolveWrite_Up, int Timeframe_Up, string ExtrVarblsNemes_Up, string ExtrVarblsVelues_Up,
bool ToResolveWrite_Dn, int Timeframe_Dn, string ExtrVarblsNemes_Dn, string ExtrVarblsVelues_Dn
)
```

This function allows to record results of profitable runs of all optimizations performed for one pair, on one timeframe and in one trade direction. This function call is placed in the EA deinitialization block (inside the deinit() function):

```
int deinit()
{
//----+
//---- Recording EA parameters into a text file
if (IsOptimization())
{
//---- +------------------------------------------+
//Here is the code of external variables initialization of the Write_Param_htm() function
//---- +------------------------------------------+
//---- RECORDING STRINGS INTO A HTML FILE
Write_Param_htm(Test_Up, Timeframe_Up, ExtVarNemes_Up, ExtVarVelues_Up, Test_Dn, Timeframe_Dn,
ExtVarNemes_Dn, ExtVarVelues_Dn);
//----+ +-------------------------------------------------------+
}
//---- End of the EA deinitialzation
return(0);
//----+
}
```

This function writes three files in C:\ directory: two files of optimization results (for long and short positions correspondingly) and a log-file (MetaTraderTester.log), in which paths and names of these two files are written. The names of the two files look like this (OptReport\_Exp\_5\_2\_GBPUSD\_240\_Long.htm):

![](https://c.mql5.com/2/16/falesname.png)

Write\_Param\_htm() function is included into the file by the directive:

```
#include <TestReport.mqh>
```

which should be better done before the declaration of EA external variables. Besides this function, TestReport.mqh file containes other functions that are used in the code of the function Write\_Param\_htm(); I will explain their meaning later. The file also contains the already known function IsBackTestingTime() with external variables for backtesting (the function was described in the [previous article](https://www.mql5.com/en/articles/1521)). The function call should be made in the block of start() function:

```
//----+ Execution of backtesting conditions
if (!IsBackTestingTime())
return(0);
```

So the simplest use of the TestReport.mqh file is absolutely analogous to the use of IsBackTestingTime.mqh from the previous article, but in this case during the EA compilation all other EA functions will not be used. As external parameters in the Write\_Param\_htm() functions eight variables are used. These variables can be divided into two groups: for recording into a file EA optimization results with long positions ending in \_Up and for recording into a file with short positions ending in \_Dn.

Let's analyze one group, the second one is analogous. The ToResolveWrite\_Up parameter passes into the function allowance or prohibition to record into the file. The Timeframe\_Up parameter passes the timeframe, on which the EA's long algorithm works. The ExtrVarblsNemes\_Up parameter passes into the function a string consisting of names of the EA external parameters for long positions. The same way the ExtrVarblsVelues\_Up parameter is used to pass a string consisting of values of the EA external parameters for long positions.

Let's discuss the initialization of the latter two variables. Suppose, we have the following external variables for long positions:

```
n_ExternParam_1, n_ExternParam_2, d_ExternParam_3, n_ExternParam_4, b_ExternParam_5, d_ExternParam_6,
d_ExternParam_7, n_ExternParam_8
```

The preffix "n\_" denotes that the parameter is of integer type,  "d\_" means that a parameter is of double type,  "b\_" - of bool type. Before the initialization of ExtrVarblsNemes\_Up andExtrVarblsVelues\_Up, a pair of additional string variables should be declared and initialized by html-code parts:

```
string n_Width = "</td><td><center>";
string d_Width = "</td><td class=mspt><center>";
```

After that we may assemble necessary strings. For the external variable ExtrVarblsNemes\_Up this code will look like this:

```
ExtVarNemes_Up =
StringConcatenate
(
n_Width, "n_ExternParam_1",
n_Width, "n_ExternParam_2",
n_Width, "d_ExternParam_3",
n_Width, "n_ExternParam_4",
n_Width, "b_ExternParam_5",
n_Width, "d_ExternParam_6",
n_Width, "d_ExternParam_7",
n_Width, "n_ExternParam_8"
);
```

The initialization of the external variable ExtVarVelues\_Up is a little more difficult:

```
ExtVarVelues_Up =
StringConcatenate
(
n_Width, n_ExternParam_1,
n_Width, n_ExternParam_2,
d_Width, d_ExternParam_3,
n_Width, n_ExternParam_4,
n_Width, b_ExternParam_5,
d_Width, d_ExternParam_6,
d_Width, d_ExternParam_7,
n_Width, n_ExternParam_8
);
```

In this code before external EA parameters, the d\_Width variable is always placed, before all other parameters n\_Width is written. In such an arrangement of the last string all parameters of the double type will be passed into the Write\_Param\_htm() function with four signs after a decimal point. If another amount of signs is needed, use the following function:

```
DoubleToStr( double value, int digits)
```

Please note, that these strings cannot contain more than 255 symbols. If the length is larger than that, excessive symbols are omitted and the compiler shows a corresponding alert. There is only one way out in such a situation: in the ExtVarNemes\_Up names of external variables should be shortened, inExtVarVelues\_Up the number of variables for string assembling should be cut down. But for most tasks 255 symbols is more than enough>

Here is the variant of implementing recording into a HTML file on the example of Exp\_5\_2.mq4:

```
//+==================================================================+
//| expert deinitialization function |
//+==================================================================+
int deinit()
{
//----+
//---- Recording EA parameters into a text file
if (IsOptimization())
{
string ExtVarNemes_Up, ExtVarVelues_Up;
string ExtVarNemes_Dn, ExtVarVelues_Dn, n_Width, d_Width;

//---- INITIALIZATION OF STRINGS FOR THE Write_Param_htm FUNCTION
//----+ +-------------------------------------------------------+
n_Width = "</td><td><center>";
d_Width = "</td><td class=mspt><center>";
//----
ExtVarNemes_Up =
StringConcatenate(
n_Width, "IndLevel_Up",
n_Width, "FastEMA_Up",
n_Width, "SlowEMA_Up",
n_Width, "SignalSMA_Up",
n_Width, "STOPLOSS_Up",
n_Width, "TAKEPROFIT_Up",
n_Width, "TRAILINGSTOP_Up",
n_Width, "PriceLevel_Up",
n_Width, "ClosePos_Up");

ExtVarVelues_Up =
StringConcatenate(
d_Width, DoubleToStr(IndLevel_Up, Digits), // 9
n_Width, FastEMA_Up, // 10
n_Width, SlowEMA_Up, // 11
n_Width, SignalSMA_Up, // 12
n_Width, STOPLOSS_Up, // 13
n_Width, TAKEPROFIT_Up, // 14
n_Width, TRAILINGSTOP_Up, // 15
n_Width, PriceLevel_Up, // 16
n_Width, ClosePos_Up); // 17
//----+ +-------------------------------------------------------+
ExtVarNemes_Dn =
StringConcatenate(
n_Width, "IndLevel_Dn",
n_Width, "FastEMA_Dn",
n_Width, "SlowEMA_Dn",
n_Width, "SignalSMA_Dn",
n_Width, "STOPLOSS_Dn",
n_Width, "TAKEPROFIT_Dn",
n_Width, "TRAILINGSTOP_Dn",
n_Width, "PriceLevel_Dn",
n_Width, "ClosePos_Dn");

ExtVarVelues_Dn =
StringConcatenate(
d_Width, DoubleToStr(IndLevel_Dn, Digits), // 9
n_Width, FastEMA_Dn, // 10
n_Width, SlowEMA_Dn, // 11
n_Width, SignalSMA_Dn, // 12
n_Width, STOPLOSS_Dn, // 13
n_Width, TAKEPROFIT_Dn, // 14
n_Width, TRAILINGSTOP_Dn, // 15
n_Width, PriceLevel_Dn, // 16
n_Width, ClosePos_Dn); // 17

//---- RECORDING STRINGS INTO HTML FILE
Write_Param_htm
(Test_Up, Timeframe_Up, ExtVarNemes_Up, ExtVarVelues_Up,
Test_Dn, Timeframe_Dn, ExtVarNemes_Dn, ExtVarVelues_Dn);
//----+ +-------------------------------------------------------+
}
//---- End of the EA deinitialization
return(0);
//----+
}
```

Here is the resulting HTML table of parameters for this EA opened in Microsoft Excel:

![](https://c.mql5.com/2/26/fig4__1.png)

The table contains results of all conducted optimizations. The table itself has the following form (the first eight columns contain the EA trade results):

![](https://c.mql5.com/2/16/face1_3.png)

Note, parameters Drawdown $ and Drawdown % are calculated only for closed trades, that is why they will differ from those available in a strategy tester. Also take into account that the 'Profit factor' parameter makes sense only if there was at least one loss trade, it is senseless if there were no loss trades at all. Initially the table does not contain lossmaking trades, that is why for situations with no losses at all I made this parameter equal to zero.

To the right of the EA trade results there are optimized EA parameters:

![](https://c.mql5.com/2/16/face2.png)

Thus we have in the form of a table all necessary values of each optimization run and all optimizations at the same time. This allows to leave for some time a tester and a terminal to conduct statistical analysis on the basis of the information contained in one html file using very efficient means available in up-to-date program analyzers of table contents.

One more thing. The described html file recorded this way will have an incomplete html code because of the constant adding of data into the file. If you are not going to add data into the file later, the following lines should be written at the end of the html code (open it in some text editor):

```
</table>
</body></html>
```

### Some Explanations of TestReport.mqh Contents

This file that I used for writing a MQL4 code providing the export of data into HTML tables contains many universal user-defined functions. These functions in a ready form can be useful for any EA writer, so I would like to consider these functions in details:

The file starts with the declaration of external variables for backtesting and IsBackTestingTime() described in the [previous article](https://www.mql5.com/en/articles/1521 "ксперты на основе популярных торговых систем и алхимия оптимизации торгового робота (Продолжение1)"). After that goes the declaration of a specialized CountTime() function which is used in some functions of the file, but will hardly be used by a reader.

The directive

```
#include <Read_Write_File.mqh>
```

includes into the TestReport.mqh file the contents of the Read\_Write\_File.mqh file: import of functions for recording and reading files in any directory from modules of the operating system Windows:

```
#include <WinUser32.mqh>
#import "kernel32.dll"
int _lopen (string path, int of);
int _lcreat (string path, int attrib);
int _llseek (int handle, int offset, int origin);
int _lread (int handle, int& buffer[], int bytes);
int _lwrite (int handle, string buffer, int bytes);
int _lclose (int handle);
#import
```

The meaning of use of this function can be understood from the code of four user-defined universal functions contained in this file:

```
int _FileSize
(
string path
)

bool WriteFile
(
string path, string buffer
)

bool ReadFile
(
string path, string&amp;amp; StrBuffer[]
)

SpellFile
(
string path, string&amp; SpellBuffer[]
)
```

These functions contain the variable path that introduces a string containing a path to a text file and its name.

The \_FileSize() function returns the file size in bytes.

The WriteFile() function records into the file end the contents of the 'buffer' string. If you need each string to be written from a new line add "\\n" at the end of each line:

```
buffer = buffer + "\n";
```

The ReadFile() function changes the size of the string buffer StrBuffer\[\] and uploads into it the file contents by strings consisting of four letters into one buffer cell.

The SpellFile() function changes the size of the string buffer and uploads the file content letter by letter.

After the #include <Read\_Write\_File.mqh> directive in the TestReport.mqh file there is a group of universal user-defined functions used to calculate the so called optimization parameters,

```
CountProfit(int cmd)
CountMaxDrawdownPrs(int cmd)
CountMaxDrawdown(int cmd)
CountAbsDrawdown(int cmd)
CountProfitFactor(int cmd)
CountProfitTrades(int cmd)
CountTotalTrades(int cmd)
CountExpectedPayoff(int cmd)
```

which are analogous to those that appear in "Optimization Results" tab of a Strategy Tester. These functions perform calculations only based on data of closed trades, that's why functions CountMaxDrawdownPrs(), CountMaxDrawdown(), CountAbsDrawdown() return their values different from those obtained in a strategy tester. If the 'cmd' variable accepts value OP\_BUY, calculations in all functions are conducted only for closed long positions, if OP\_SELL - for closed short positions. Any other values of this variable provide calculations for all closed trade operations.

After these functions four more functions are declared in the file - for recording optimization results and optimized parameters into files:

```
void Write_Param_htm
(
bool ToResolveWrite_Up, int Timeframe_Up, string ExtrVarblsNemes_Up, string ExtrVarblsVelues_Up,
bool ToResolveWrite_Dn, int Timeframe_Dn, string ExtrVarblsNemes_Dn, string ExtrVarblsVelues_Dn
)

void Write_Param_1htm
(
int Timeframe, string ExtrVarblsNemes, string ExtrVarblsVelues
)

void Write_Param_txt
(
bool ToResolveWrite_Up, int Timeframe_Up, string ExtrVarblsVelues_Up,
bool ToResolveWrite_Dn, int Timeframe_Dn, string ExtrVarblsVelues_Dn
)

void Write_Param_1txt
(
int Timeframe, string ExtrVarblsVelues
)
```

As for Write\_Param\_htm(), it is described above.

The Write\_Param\_1htm() function is a full analogue of the previous function, but it is created for Expert Advisors, in which external parameters are not divided into those for long and short positions. The ToResolveWrite function is not needed here and that's why deleted.

The Write\_Param\_txt() function is the analogue of Write\_Param\_htm(), it records the same data but without html indication. The additional variable for assembling a string consisting of optimization results and optimized parameters is initialized in the following way:

```
Width = " ";
```

Each new string is recorded into the file from a new line, at the end of each string in the text file ";" is written. Between values of external variables of one string spaces are written in the text file.

The Write\_Param\_1txt() function is a full analogue of the Write\_Param\_txt() function, but it is created for Expert Advisors, in which external parameters are not divided into those for long and short positions.

And the last two functions

```
void Write_Param_htm_B
(
bool ToResolveWrite_Up, int Timeframe_Up, string ExtrVarblsNemes_Up, string ExtrVarblsVelues_Up,
bool ToResolveWrite_Dn, int Timeframe_Dn, string ExtrVarblsNemes_Dn, string ExtrVarblsVelues_Dn
)

void Write_Param_1htm_B
(
int Timeframe, string ExtrVarblsNemes, string ExtrVarblsVelues
)
```

are full analogues of Write\_Param\_htm() and Write\_Param\_1htm(), but html files received with their help have inverse coloring. here is an example:

![](https://c.mql5.com/2/16/exel_b1.png)

Probably, there are people that prefer to work with such files. I personally like this kind of information presentation.

### Parabolic Trading System

The working indicator of this system is included in all popular trading platforms for working in the Forex and stock markets.

![](https://c.mql5.com/2/16/sar0_2.gif)

Here I will introduce a system with pending orders of BuyLimit and SellLimit types.

Here is the variant of algorithm implementation for BuyLimit:

![](https://c.mql5.com/2/16/sar1_1.gif)

The algorithm variant for SellLimit orders will be analogous:

![](https://c.mql5.com/2/16/sar2_1.gif)

Here is the implementation of the program code for an expert Advisor:

```
//+==================================================================+
//|                                                        Exp_8.mq4 |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern int    Timeframe_Up = 240;
extern double Money_Management_Up = 0.1;
extern double Step_Up = 0.02;
extern double Maximum_Up = 0.2;
extern int    STOPLOSS_Up = 50;  // stop loss
extern int    TAKEPROFIT_Up = 100; // take profit
extern int    TRAILINGSTOP_Up = 0; // trailing stop
extern int    PriceLevel_Up =40; // difference between the current price and
                          // pending order triggering price
extern bool   ClosePos_Up = true; // forced position closing
                                              //is allowed
//----+ +------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern int    Timeframe_Dn = 240;
extern double Money_Management_Dn = 0.1;
extern double Step_Dn = 0.02;
extern double Maximum_Dn = 0.2;
extern int    STOPLOSS_Dn = 50;  // stop loss
extern int    TAKEPROFIT_Dn = 100; // take profit
extern int    TRAILINGSTOP_Dn = 0; // trailing stop
extern int    PriceLevel_Dn = 40; // difference between the current price and
                          // pending order triggering price
extern bool   ClosePos_Dn = true; // forced position closing
                                              //is allowed
//----+ +------------------------------------------------------------+
//---- Integer variables for the minimum of calculation bars
int MinBar_Up, MinBar_Dn;
//+==================================================================+
//| TimeframeCheck() functions                                       |
//+==================================================================+
void TimeframeCheck(string Name, int Timeframe)
  {
//----+
   //---- Checking the correctness of Timeframe variable value
   if (Timeframe != 1)
    if (Timeframe != 5)
     if (Timeframe != 15)
      if (Timeframe != 30)
       if (Timeframe != 60)
        if (Timeframe != 240)
         if (Timeframe != 1440)
           Print(StringConcatenate("Parameter ",Name,
                     " cannot ", "be equal to ", Timeframe, "!!!"));
//----+
  }
//+==================================================================+
//| Custom Expert functions                                          |
//+==================================================================+
#include <Lite_EXPERT1.mqh>
//+==================================================================+
//| Custom Expert initialization function                            |
//+==================================================================+
int init()
  {
//---- Checking the correctness of Timeframe_Up variable value
   TimeframeCheck("Timeframe_Up", Timeframe_Up);
//---- Checking the correctness of Timeframe_Dn variable value
   TimeframeCheck("Timeframe_Dn", Timeframe_Dn);
//---- Initialization of variables
   MinBar_Up  = 5;
   MinBar_Dn  = 5;
//---- end of initialization
   return(0);
  }
//+==================================================================+
//| expert deinitialization function                                 |
//+==================================================================+
int deinit()
  {
//----+

    //---- End of the EA deinitialization
    return(0);
//----+
  }
//+==================================================================+
//| Custom Expert iteration function                                 |
//+==================================================================+
int start()
  {
   //----+ Declaring local variables
   double SAR1, SAR2, CLOSE1, CLOSE2;
   //----+ Declaring static variables
   //----+ +---------------------------------------------------------------+
   static datetime StopTime_Up, StopTime_Dn;
   static int LastBars_Up, LastBars_Dn;
   static bool BUY_Sign, BUY_Stop, SELL_Sign, SELL_Stop;
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS 1
   if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);

      if (IBARS_Up >= MinBar_Up)
       {
         if (LastBars_Up != IBARS_Up)
          {
           //----+ Initialization of variables
           BUY_Sign = false;
           BUY_Stop = false;
           LastBars_Up = IBARS_Up;
           StopTime_Up = iTime(NULL, Timeframe_Up, 0)
                                           + 60 * Timeframe_Up;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           SAR1 = iSAR(NULL, Timeframe_Up, Step_Up, Maximum_Up, 1);
           SAR2 = iSAR(NULL, Timeframe_Up, Step_Up, Maximum_Up, 2);
           //---
           CLOSE1 = iClose(NULL, Timeframe_Up, 1);
           CLOSE2 = iClose(NULL, Timeframe_Up, 2);

           //----+ DEFINING SIGNALS FOR TRADES
           if (SAR2 > CLOSE2)
             if (SAR1 < CLOSE1)
                          BUY_Sign = true;

           if (SAR1 > CLOSE1)
                          BUY_Stop = true;
          }
          //----+ EXECUTION OF TRADES
          if (PriceLevel_Up == 0)
           {
            if (!OpenBuyOrder1(BUY_Sign, 1,
                Money_Management_Up, STOPLOSS_Up, TAKEPROFIT_Up))
                                                                return(-1);
           }
          else
           {
            if (!OpenBuyLimitOrder1(BUY_Sign, 1,
                Money_Management_Up, STOPLOSS_Up, TAKEPROFIT_Up,
                                            PriceLevel_Up, StopTime_Up))
                                                                return(-1);
           }

          if (ClosePos_Up)
                if (!CloseOrder1(BUY_Stop, 1))
                                        return(-1);

          if (!Make_TreilingStop(1, TRAILINGSTOP_Up))
                                                  return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR SHORT POSITIONS 1
   if (Test_Dn)
    {
      int IBARS_Dn = iBars(NULL, Timeframe_Dn);

      if (IBARS_Dn >= MinBar_Dn)
       {
         if (LastBars_Dn != IBARS_Dn)
          {
           //----+ Initialization of variables
           SELL_Sign = false;
           SELL_Stop = false;
           LastBars_Dn = IBARS_Dn;
           StopTime_Dn = iTime(NULL, Timeframe_Dn, 0)
                                           + 60 * Timeframe_Dn;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           SAR1 = iSAR(NULL, Timeframe_Dn, Step_Dn, Maximum_Dn, 1);
           SAR2 = iSAR(NULL, Timeframe_Dn, Step_Dn, Maximum_Dn, 2);
           //---
           CLOSE1 = iClose(NULL, Timeframe_Dn, 1);
           CLOSE2 = iClose(NULL, Timeframe_Dn, 2);

           //----+ DEFINING SIGNALS FOR TRADES
           if (SAR2 < CLOSE2)
             if (SAR1 > CLOSE1)
                          SELL_Sign = true;

           if (SAR1 < CLOSE1)
                          SELL_Stop = true;
          }
          //----+ EXECUTION OF TRADES
          if (PriceLevel_Dn == 0)
           {
            if (!OpenSellOrder1(SELL_Sign, 2,
                Money_Management_Dn, STOPLOSS_Dn, TAKEPROFIT_Dn))
                                                                return(-1);
           }
          else
           {
            if (!OpenSellLimitOrder1(SELL_Sign, 2,
                Money_Management_Dn, STOPLOSS_Dn, TAKEPROFIT_Dn,
                                            PriceLevel_Dn, StopTime_Dn))
                                                                return(-1);
           }

          if (ClosePos_Dn)
                if (!CloseOrder1(SELL_Stop, 2))
                                        return(-1);

          if (!Make_TreilingStop(2, TRAILINGSTOP_Dn))
                                                  return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
//----+

    return(0);
  }
//+------------------------------------------------------------------+
```

The block of trade execution is more complicated in this EA. Here functions for entering market OpenBuyOrder1() and OpenSellOrder1() and functions for placing pending orders OpenBuyLimitOrder1() and OpenSellLimitOrder1() are used together. This is done for the case when external variables PriceLevel\_Up and PriceLevel\_Dn are equal to zero. In such a situation the EA will open trades from the market, while an EA working only with pending orders will place these orders at the minimal distance from the market.

### Breakout Trading System

As for trading systems based on breakout levels, there are a great many of them and you can always invent one more breakout system. In this article I will describe a system, in which breakout levels are defined by the sides of a channel built on the basis of a moving:

![](https://c.mql5.com/2/16/movchnl0.gif)

The breakout will be defined on closed bars. Besides, I will use a breakout trading system with some rebound following the breakout. Here is the variant of this trading systems with BuyLimit orders:

![](https://c.mql5.com/2/16/movchnl1.gif)

The same for SellLimit:

![](https://c.mql5.com/2/16/movchnl2.gif)

The EA code does not have any construction peculiarities, that is why I do not include it into the article. You can find the code attached to the EA (Exp\_9.mq4). There is no forced position closing in the EA. New global variables are initialized in the EA initialization block.

```
dMovLevel_Up = MovLevel_Up * Point;
dMovLevel_Dn = MovLevel_Dn * Point;
```

these variables are used to calculate values of UpMovLeve1 and DownMovLevel1 levels.

### Trading on News Releases

Very often a situation happens in the market, when strong movements occur invariably at one and the same time. It is hard to say beforehand for sure in what direction the movement will appear. But one can state beforehand that there will be such a movement. For example, such strong movements can result from Non-farm Payrolls. Many traders like to catch such strong movements. It would be interesting to check whether such a trading system is reasonable. The idea of the system consists in placing of two pending orders of BuyStop and SellStop types and wait until at a strong movement one of them opens a trade and is closed by Take Profit:

![](https://c.mql5.com/2/16/newsstop__1.gif)

This is one more variant of a breakout system. The implementation of such a system in a program code is not very difficult:

```
//+==================================================================+
//|                                                       Exp_10.mq4 |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR PLACING PENDING ORDERS
extern int    NewsWeekDay = 5;  // Weekday for news release
// if parameter is less than 1 or more than 5 any working day is used
extern int    NewsHour = 15;  // hour of news release
extern int    NewsMinute = 0;  // minute of news release
extern int    OrderLife = 30;  // Number of minutes of pending order validity
//----+ +------------------------------------------------------------+
//---- ea INPUT PARAMETERS FOR CLOSING POSITIONS
extern int    PosLife = 120;  // Number of minutes of an opened position validity
                                              //from news release time
//----+ +------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern double Money_Management_Up = 0.1;
extern int    STOPLOSS_Up = 50;  // stop loss
extern int    TAKEPROFIT_Up = 100; // take profit
extern int    TRAILINGSTOP_Up = 0; // trailing stop
extern int    PriceLevel_Up =40; // difference between the current price and
                          // the price of pending order triggering
extern bool   ClosePos_Up = true; // forced position closing
                                              //is allowed
//----+ +------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern double Money_Management_Dn = 0.1;
extern int    STOPLOSS_Dn = 50;  // stop loss
extern int    TAKEPROFIT_Dn = 100; // take profit
extern int    TRAILINGSTOP_Dn = 0; // trailing stop
extern int    PriceLevel_Dn = 40; // difference between the current price and
                          // the price of pending order triggering
extern bool   ClosePos_Dn = true; // forced position closing
                                              //is allowed
//+==================================================================+
//| Custom Expert functions                                          |
//+==================================================================+
#include <Lite_EXPERT1.mqh>
//+==================================================================+
//| Custom Expert initialization function                            |
//+==================================================================+
int init()
  {
//----
    if (NewsHour > 23)
             NewsHour = 23;
    if (NewsMinute > 59)
           NewsMinute = 59;
//---- end of initialization
   return(0);
  }
//+==================================================================+
//| expert deinitialization function                                 |
//+==================================================================+
int deinit()
  {
//----+

    //---- End of the EA deinitialization
    return(0);
//----+
  }
//+==================================================================+
//| Custom Expert iteration function                                 |
//+==================================================================+
int start()
  {
   //----+ Declaring static variables
   //----+ +---------------------------------------------------------------+
   static datetime StopTime, PosStopTime;
   static bool BUY_Sign, SELL_Sign, BUY_Stop, SELL_Stop;
   //----+ +---------------------------------------------------------------+

   //----+ DEFINING SIGNALS FOR TRADES
   if (DayOfWeek() == NewsWeekDay || NewsWeekDay < 1 || NewsWeekDay > 5)
     if (Hour() == NewsHour)
        if (Minute() == NewsMinute)
    {
     StopTime = TimeCurrent() + OrderLife * 60;
     PosStopTime = TimeCurrent() + PosLife * 60;
     BUY_Sign = true;
     SELL_Sign = true;
    }



   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS 1
   if (Test_Up)
    {
      //----+ EXECUTION OF TRADES
      if (!OpenBuyStopOrder1(BUY_Sign, 1,
                Money_Management_Up, STOPLOSS_Up, TAKEPROFIT_Up,
                                            PriceLevel_Up, StopTime))
                                                                return(-1);
      if (TimeCurrent() >= PosStopTime)
          if (ClosePos_Up)
                if (!CloseOrder1(BUY_Stop, 1))
                                           return(-1);

      if (!Make_TreilingStop(1, TRAILINGSTOP_Up))
                                           return(-1);
     }
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR SHORT POSITIONS 1
   if (Test_Dn)
     {
      //----+ EXECUTION OF TRADES
      if (!OpenSellStopOrder1(SELL_Sign, 2,
                Money_Management_Dn, STOPLOSS_Dn, TAKEPROFIT_Dn,
                                            PriceLevel_Dn, StopTime))
                                                                return(-1);
      if (TimeCurrent() >= PosStopTime)
          if (ClosePos_Dn)
                if (!CloseOrder1(SELL_Stop, 2))
                                           return(-1);

      if (!Make_TreilingStop(2, TRAILINGSTOP_Dn))
                                           return(-1);
     }
   //----+ +---------------------------------------------------------------+
//----+

    return(0);
  }
//+------------------------------------------------------------------+
```

### Conclusion

In this article I have offered you a universal approach to exporting various information about optimization results into the format of spreadsheets, which opens wide opportunities for deep analysis of this information and serious improvement of trading results used by Expert Advisors in automated trading. I hope my approach will be useful to any EA writer.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1523](https://www.mql5.com/ru/articles/1523)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1523.zip "Download all attachments in the single ZIP archive")

[EXPERTS.zip](https://www.mql5.com/en/articles/download/1523/EXPERTS.zip "Download EXPERTS.zip")(13.19 KB)

[INCLUDE.zip](https://www.mql5.com/en/articles/download/1523/INCLUDE.zip "Download INCLUDE.zip")(22.68 KB)

[indicators.zip](https://www.mql5.com/en/articles/download/1523/indicators.zip "Download indicators.zip")(5.2 KB)

[TESTER.zip](https://www.mql5.com/en/articles/download/1523/TESTER.zip "Download TESTER.zip")(7.93 KB)

[xtjmytfz9oqg.zip](https://www.mql5.com/en/articles/download/1523/xtjmytfz9oqg.zip "Download xtjmytfz9oqg.zip")(161.41 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/39447)**
(2)


![Hendrick Stam](https://c.mql5.com/avatar/avatar_na2.png)

**[Hendrick Stam](https://www.mql5.com/en/users/hendrick)**
\|
10 Jun 2008 at 18:43

Nikolay,

Many thanks for your absolutely brilliant four articles. It gave me a completely different look on how to program and test EA's. Many thanks again!!

Hendrick.

![Mehmet Bastem](https://c.mql5.com/avatar/avatar_na2.png)

**[Mehmet Bastem](https://www.mql5.com/en/users/mehmet)**
\|
11 Jun 2008 at 00:24

my tested GBPUSD 240 Timeframe.

1.2008.06.11 01:19:42 2008.06.09 23:58  Exp\_10 GBPUSD,H4: [OrderSend](https://docs.mql4.com/trading/ordersend "OrderSend") error 3

2\. first balance is 10.000 last balance is 629

3\. Please traslate English

(sample //----+ ÎÏĞÅÄÅËÅÍÈÅ ÑÈÃÍÀËÎÂ ÄËß ÑÄÅËÎÊ

if (DayOfWeek() == NewsWeekDay \|\| NewsWeekDay < 1 \|\| NewsWeekDay > 5)

)

![Comfortable Scalping](https://c.mql5.com/2/15/553_7.gif)[Comfortable Scalping](https://www.mql5.com/en/articles/1509)

The article describes the method of creating a tool for comfortable scalping. However, such an approach to trade opening can be applied in any trading.

![A Non-Trading EA Testing Indicators](https://c.mql5.com/2/16/627_11.gif)[A Non-Trading EA Testing Indicators](https://www.mql5.com/en/articles/1534)

All indicators can be divided into two groups: static indicators, the displaying of which, once shown, always remains the same in history and does not change with new incoming quotes, and dynamic indicators that display their status for the current moment only and are fully redrawn when a new price comes. The efficiency of a static indicator is directly visible on the chart. But how can we check whether a dynamic indicator works ok? This is the question the article is devoted to.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part  V)](https://c.mql5.com/2/15/600_99.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part V)](https://www.mql5.com/en/articles/1525)

In this article the author offers ways to improve trading systems described in his previous articles. The article will be interesting for traders that already have some experience of writing Expert Advisors.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part III)](https://c.mql5.com/2/15/584_49.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part III)](https://www.mql5.com/en/articles/1521)

In this article the author continues to analyze implementation algorithms of simplest trading systems and introduces backtesting automation. The article will be useful for beginning traders and EA writers.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dldwdodfphhvmbqzyymwzpautpnpehsh&ssn=1769091354779679008&ssn_dr=0&ssn_sr=0&fv_date=1769091354&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1523&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Expert%20Advisors%20Based%20on%20Popular%20Trading%20Systems%20and%20Alchemy%20of%20Trading%20Robot%20Optimization%20(Part%20IV)%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176909135459581541&fz_uniq=5049005515909997338&sv=2552)

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