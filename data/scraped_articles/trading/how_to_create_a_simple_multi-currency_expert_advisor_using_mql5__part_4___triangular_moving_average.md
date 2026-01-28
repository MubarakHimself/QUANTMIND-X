---
title: How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals
url: https://www.mql5.com/en/articles/13770
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 16
scraped_at: 2026-01-22T17:08:57.499667
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/13770&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048823533850697566)

MetaTrader 5 / Trading


### Introduction

The definition of a Multi-Currency Expert Advisor in this article is one Expert Advisor or trading robot that can trade (open orders, close orders and manage orders for example: Trailing Stop Loss and Trailing Profit) for more than one symbol pair from only one symbol chart, where in this article Expert Advisor will trade for 30 pairs.

This time we will use only 1 indicator, namely Triangular moving average in multi-timeframes or single timeframe mode.

In this article, Expert Advisor signal calculations can be selected whether to activate multi-timeframes or activate single-timeframes.

[Triangular moving average](https://www.mql5.com/en/code/23163) is a custom indicator for MT5 from author [Mladen Rakic](https://www.mql5.com/en/users/mladen), and I have received permission from the author to use his indicator as a signal on the Multi-Currency Expert Advisor TriangularMA\_MTF\_MCEA.

Salute and thank to the author Mladen Rakic.

We all know that multi-currency trading, both on the trading terminal and on the strategy tester, is all possible with the power, capabilities and facilities provided by MQL5.

Therefore, the aim is to fulfill the essential needs of traders who want efficient and effective trading robots, so by relying on the strengths, capabilities and facilities provided by the highly reliable MQL5, we can create a simple Multi-Currency Expert Advisor which in this article uses Indicator Signals: Triangular moving average Indicator.

Note: The creation of the Multi-Currency Expert Advisor TriangularMA\_MTF\_MCEA was at the suggestion and request of traders.

### Plans and Features

**1\. Trading Currency Pairs.**

This Multi-Currency Expert Advisor is planned to trade on a Symbol or Pair as follows:

EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURGBP,EURAUD, EURNZD, EURCAD, EURCHF, EURJPY, GBPAUD, GBPNZD, GBPCAD,GBPCHF,GBPJPY,AUDNZD,AUDCAD,AUDCHF,AUDJPY,NZDCAD,NZDCHF,NZDJPY, CADCHF, CADJPY, CHFJPY = 28 pairs

Plus 2 Metal pairs: XAUUSD (Gold) and XAGUSD (Silver)

Total is 30 pairs.

In the [previous article](https://www.mql5.com/en/articles/13705), for pairs on brokers with special pair names with prefixes and/or suffixes, we used an automatic function to detect pair names with prefixes and/or suffixes.

But in this article we make it simple by adding special input properties for the pair name prefix and pair name suffix.

Then with a simple function, we handle the prefix and/or suffix pair names combined with the 30 registered pair names, so that if an EA is used on MetaTrader5 from a broker with special symbol names like that, everything will run smoothly.

The weakness of the function for detecting symbol names that have prefixes and suffixes is that this function only works on forex and metal symbol pairs or names in MT5, but will not work on special symbols and indices.

Apart from that, another weakness of this method is if the trader makes a typo (must type in case sensitive) in the name of the pair's prefix and/or suffix.

As in the previous article, in this expert advisor we also added 10 options for the pairs that will be traded at this time.

One of the 10 option pairs that will be traded is "Trader Wishes Pairs", where the pairs that will be traded must be input manually by the trader on the expert input property. But must always remember that the name of the pair entered must already be in the list of 30 pairs.

Still the same as the previous article, in this version of the expert advisor we also added an option for Trading Session (Time Zone), so the pairs that will be trade which may correspond to the time for the trading session.

**2\. Signal indicator.**

In the description of the [Triangular moving average](https://www.mql5.com/en/code/23163) indicator, the author states:

"Usage:

You can use the color change as signal..."

By default, the colors of the Triangular moving average indicator are:

- 0-DarkGray = Unknown signal
- 1-DeepPink = Sell Signal
- 2-MediumSeaGreen = Buy Signal.

In this version of the expert advisor, we created 2 options for using timeframes in calculating the Triangular moving average indicator signal.

1\. Signal calculation based on multi-timeframe.

In a multi-timeframe calculation system, traders must select from an enumeration list the desired timeframe series.

The selected timeframe series provided range from M5 to D1 (11 timeframes).

And traders can choose a timeframe series, choose start: for example M15, and select end: for example H4.

So, the expert advisor will calculate the Triangular moving average indicator signal starting from Timeframe M15 to Timeframe H4.

Triangular moving average signal calculations on multi-timeframes are:

- Buy signal, if on all selected timeframes the indicator is colored MediumSeaGreen, and
- Sell signal, if on all selected timeframes the indicator is DeepPink colored.

2\. Signal calculation based on single-timeframe.

In the single-timeframe signal calculation system, traders must choose one timeframe from 11 timeframes, starting from the M5 timeframe to the D1 timeframe.

So, the expert advisor will calculate the Triangular moving average indicator signal from Timeframe selected.

Meanwhile, the calculation of the trianguler moving average signal on a single timeframe is:

- Buy signal, if the previous 2 bars are DeepPink, and the previous 1 bar is MediumSeaGreen and the current bar is MediumSeaGreen.
- Sell signal, if the previous 2 bars are MediumSeaGreen, and the previous 1 bar is DeepPink and the current bar is DeepPink.

An illustration of the Triangular moving average indicator for BUY or SELL signal, can be seen in Figure 1 and Figure 2.

![H4_TriangularMA01](https://c.mql5.com/2/60/H4_TriangularMA01.png)

![H4_TriangularMA02](https://c.mql5.com/2/60/H4_TriangularMA02.png)

**3\. Trade & Order Management.**

Trading management on this Multi-Currency Expert Advisor is given several options:

1\. Stop Loss Orders

- Options: Use Order Stop Loss (Yes) or (No)

            If the Use Order Stop Loss (No) option is selected, then all orders will be opened without a stop loss.

            If the option Use Order Stop Loss (Yes):

            Again given the option: Use Automatic Calculation Stop Loss (Yes) or (No)

            If the option Automatic Calculation Stop Loss (Yes),  then the Stop Loss calculation will be performed automatically by the Expert.

            If the option Automatic Calculation Stop Loss (No), then the trader must Input Stop Loss value in Pips.

            If the option Use Order Stop Loss (No):

            Then the Expert will check for each order opened, whether the signal condition is still good and order

            may be maintained in a profit or condition the signal has weakened and the order needs to be closed to save

            profit or signal condition has reversed direction and order must be closed in a loss condition.

            Note:

             Especially for Close Trade and Save profit due to weak signal, an option is given, whether to activate it or not.

             If it is not activated (No), even though the signal has weakened, the order will still be maintained or will not be closed to save profit.

2\. Take Profit orders

- Options: Use Order Take Profit (Yes) or (No)

            If the Use Order Take Profit (No) option is selected, then all orders will be opened without take profit.

            If the option Use Order Take Profit (Yes):

            Again given the option: Use Automatic Calculation Order Take Profit (Yes) or (No)

            If the option Automatic Calculation Order Take Profit (Yes), then the calculation of the Take Profit Order will be carried out automatically by the Expert.

            If the option Automatic Calculation Order Take Profit (No), then the trader must Input Order Take Profit value in Pips.

3\. Trailing Stop and Trailing Take Profit

- Options: Use Trailing SL/TP (Yes) or (No)

            If the Use Trailing SL/TP option is (No), then the Expert will not do trailing stop loss and trailing take profit.

            If the option Use Trailing SL/TP (Yes):

            Again given the option: Use Automatic Trailing (Yes) or (No)

            If the option Use Automatic Trailing (Yes), then the trailing stop will be executed by the Expert using

            Triangular moving average buffer 0 (Indicator Data) on the timeframe wich automatically selected by the expert advisor, and at the same time

            by making trailing profit based on the variable value TPmin (minimum trailing profit value).

            If the option Use Automatic Trailing (No), then the trailing stop will be performed by the Expert using the value in the input property.

            Note: The Expert will carry out a trailing take profit simultaneously with a trailing stop.

Trailing Stop Price function:

```
double MCEA::TSPrice(const string xsymb,ENUM_POSITION_TYPE ptype,int TS_type)
  {
//---
    int br=2;
    double pval=0.0;
    int x=PairsIdxArray(xsymb);
    Pips(xsymb);
    //--
    switch(TS_type)
      {
        case 0:
          {
            RefreshTick(xsymb);
            if(ptype==POSITION_TYPE_BUY)  pval=mc_symbol.NormalizePrice(mc_symbol.Bid()-TSval*pip);
            if(ptype==POSITION_TYPE_SELL) pval=mc_symbol.NormalizePrice(mc_symbol.Ask()+TSval*pip);
            break;
          }
        case 1:
          {
            double TriMAID[];
            //--
            ArrayResize(TriMAID,br,br);
            ArraySetAsSeries(TriMAID,true);
            CopyBuffer(hTriMAt[x],0,0,br,TriMAID); // Copy buffer 0 from the hTriMAt indicator handle
            //--
            RefreshTick(xsymb);
            if(ptype==POSITION_TYPE_BUY  && (mc_symbol.Bid()>mc_symbol.NormalizePrice(TriMAID[0]+TSval*pip))) pval=TriMAID[0];
            if(ptype==POSITION_TYPE_SELL && (mc_symbol.Ask()<mc_symbol.NormalizePrice(TriMAID[0]-TSval*pip))) pval=TriMAID[0];
            break;
          }
      }
    //--
    return(pval);
//---
  } //-end TSPrice()
//---------//
```

Modify SL/TP Function:

```
bool MCEA::ModifySLTP(const string symbx,int TS_type)
  {
//---
   ResetLastError();
   MqlTradeRequest req={};
   MqlTradeResult  res={};
   MqlTradeCheckResult check={};
   //--
   int TRSP=TS_type;
   bool modist=false;
   int x=PairsIdxArray(symbx);
   Pips(symbx);
   //--
   int total=PositionsTotal();
   //--
   for(int i=total-1; i>=0; i--)
     {
       string symbol=PositionGetSymbol(i);
       if(symbol==symbx && mc_position.Magic()==magicEA)
         {
           ENUM_POSITION_TYPE opstype = mc_position.PositionType();
           if(opstype==POSITION_TYPE_BUY)
             {
               RefreshTick(symbol);
               double price = mc_position.PriceCurrent();
               double vtrsb = mc_symbol.NormalizePrice(TSPrice(symbx,opstype,TRSP));
               double pos_open   = mc_position.PriceOpen();
               double pos_stop   = mc_position.StopLoss();
               double pos_profit = mc_position.Profit();
               double pos_swap   = mc_position.Swap();
               double pos_comm   = mc_position.Commission();
               double netp=pos_profit+pos_swap+pos_comm;
               double modstart=mc_symbol.NormalizePrice(pos_open+TSmin*pip);
               double modminsl=mc_symbol.NormalizePrice(vtrsb+TSmin*pip);
               double modbuysl=vtrsb;
               double modbuytp=mc_symbol.NormalizePrice(price+TPmin*pip);
               bool modbuy = (price>modminsl && modbuysl>modstart && (pos_stop==0.0||modbuysl>pos_stop));
               //--
               if(modbuy && netp>0.05)
                 {
                   modist=mc_trade.PositionModify(symbol,modbuysl,modbuytp);
                 }
             }
           if(opstype==POSITION_TYPE_SELL)
             {
               RefreshTick(symbol);
               double price = mc_position.PriceCurrent();
               double vtrss = mc_symbol.NormalizePrice(TSPrice(symbx,opstype,TRSP));
               double pos_open   = mc_position.PriceOpen();
               double pos_stop   = mc_position.StopLoss();
               double pos_profit = mc_position.Profit();
               double pos_swap   = mc_position.Swap();
               double pos_comm   = mc_position.Commission();
               double netp=pos_profit+pos_swap+pos_comm;
               double modstart=mc_symbol.NormalizePrice(pos_open-TSmin*pip);
               double modminsl=mc_symbol.NormalizePrice(vtrss-TSmin*pip);
               double modselsl=vtrss;
               double modseltp=mc_symbol.NormalizePrice(price-TPmin*pip);
               bool modsel = (price<modminsl && modselsl<modstart && (pos_stop==0.0||modselsl<pos_stop));
               //--
               if(modsel && netp>0.05)
                 {
                   modist=mc_trade.PositionModify(symbol,modselsl,modseltp);
                 }
             }
         }
     }
    //--
    return(modist);
//---
  } //-end ModifySLTP()
//---------//
```

**4\. Manual Order Management.**

To support efficiency in this Multi-Currency Expert Advisor, several manual click buttons will be added.

> 1\. Set SL / TP All Orders

> When the trader input parameter sets Use Order Stop Loss (No) and/or Use Order Take Profit (No)
>
> but then the trader intends to use stop loss or take profit on all orders, then with just single click of the button
>
> "Set SL / TP All Orders" all orders will be modified and a stop loss will be applied and/or take profits.

> 2\. Close All Orders
>
> If a trader wants to close all orders, then with just single click of the button "Close All Orders" all open orders will be closed.
>
> 3\. Close All Orders Profit
>
> If a trader wants to close all orders that are already profitable, then with only single click of the button
>
> "Close All Orders Profit" all open orders that are already profitable will be closed.

**5\. Management Orders and Chart Symbols.**

For Multi-Currency Expert Advisors who will trade 30 pairs from only one chart symbol, it will be very effective and efficient if a button panel is provided for all symbols, so traders can change charts or symbols with just one click.

### Implementation of planning in the MQL5 program

**1\. Program header and input properties**

Include Header file MQL5

```
//+------------------------------------------------------------------+
//|                             Include                              |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>
//--
CTrade              mc_trade;
CSymbolInfo         mc_symbol;
CPositionInfo       mc_position;
CAccountInfo        mc_account;
//---
```

Enumeration to use Time Zone

```
//--
enum tm_zone
 {
   Cus_Session,        // Trading on Custom Session
   New_Zealand,        // Trading on New Zealand Session
   Australia,          // Trading on Autralia Sydney Session
   Asia_Tokyo,         // Trading on Asia Tokyo Session
   Europe_London,      // Trading on Europe London Session
   US_New_York         // Trading on US New York Session
 };
//--
```

Enumeration to select time hour

```
//--
enum swhour
  {
    hr_00=0,   // 00:00
    hr_01=1,   // 01:00
    hr_02=2,   // 02:00
    hr_03=3,   // 03:00
    hr_04=4,   // 04:00
    hr_05=5,   // 05:00
    hr_06=6,   // 06:00
    hr_07=7,   // 07:00
    hr_08=8,   // 08:00
    hr_09=9,   // 09:00
    hr_10=10,  // 10:00
    hr_11=11,  // 11:00
    hr_12=12,  // 12:00
    hr_13=13,  // 13:00
    hr_14=14,  // 14:00
    hr_15=15,  // 15:00
    hr_16=16,  // 16:00
    hr_17=17,  // 17:00
    hr_18=18,  // 18:00
    hr_19=19,  // 19:00
    hr_20=20,  // 20:00
    hr_21=21,  // 21:00
    hr_22=22,  // 22:00
    hr_23=23   // 23:00
  };
//--
```

Enumeration to select time minutes

```
//--
enum inmnt
  {
    mn_00=0,   // Minute 0
    mn_05=5,   // Minute 5
    mn_10=10,  // Minute 10
    mn_15=15,  // Minute 15
    mn_20=20,  // Minute 20
    mn_25=25,  // Minute 25
    mn_30=30,  // Minute 30
    mn_35=35,  // Minute 35
    mn_40=40,  // Minute 40
    mn_45=45,  // Minute 45
    mn_50=50,  // Minute 50
    mn_55=55   // Minute 55
  };
//--
```

Enumeration to select option pairs to be traded

```
//--
enum PairsTrade
 {
   All30,  // All Forex 30 Pairs
   TrdWi,  // Trader Wishes Pairs
   Usds,   // Forex USD Pairs
   Eurs,   // Forex EUR Pairs
   Gbps,   // Forex GBP Pairs
   Auds,   // Forex AUD Pairs
   Nzds,   // Forex NZD Pairs
   Cads,   // Forex CDD Pairs
   Chfs,   // Forex CHF Pairs
   Jpys    // Forex JPY Pairs
 };
//--
```

Enumeration YN is Used for options (Yes) or (No) in expert input property

```
//--
enum YN
  {
   No,
   Yes
  };
//--
```

Enumeration to use Money Management Lot size

```
//--
enum mmt
  {
   FixedLot,   // Fixed Lot Size
   DynamLot    // Dynamic Lot Size
  };
//--
```

Enumeration to select the timeframe that will be used in multi-timeframe and single-timeframe

```
//--
enum TFMTF
  {
   TFM5,     // PERIOD_M5
   TFM15,    // PERIOD_M15
   TFM30,    // PERIOD_M30
   TFH1,     // PERIOD_H1
   TFH2,     // PERIOD_H2
   TFH3,     // PERIOD_H3
   TFH4,     // PERIOD_H4
   TFH6,     // PERIOD_H6
   TFH8,     // PERIOD_H8
   TFH12,    // PERIOD_H12
   TFD1      // PERIOD_D1
  };
//--
```

Enumeration to use Multi-Timeframe or Single-Timeframe

```
//--
enum SMTF
  {
    MTF,   // Use Multi-Timeframe
    STF    // Use Single-Timeframe
  };
//--
```

Expert input properties

```
//---
input group               "=== Global Strategy EA Parameter ==="; // Global Strategy EA Parameter
input SMTF                tfinuse = MTF;              // Select Calculation in Multi or Single Timeframe
input TFMTF              singletf = TFH1;             // Select Single Calculation TimeFrame, default PERIOD_H1
input TFMTF               tfstart = TFM15;            // Select Multi Timeframe calculation start
input TFMTF               tfclose = TFH4;             // Select Multi Timeframe calculation end
input int              Trmaperiod = 14;               // Input Triangular MA Indicator period, default 14
input ENUM_APPLIED_PRICE  Trprice = PRICE_CLOSE;      // Select Triangular MA Applied Price, default Price Close
//---
input group               "=== Select Pairs to Trade ===";  // Selected Pairs to trading
input PairsTrade         usepairs = All30;           // Select Pairs to Use
input string         traderwishes = "eg. eurusd,usdchf"; // If Use Trader Wishes Pairs, input pair name here, separate by comma
input string           sym_prefix = "";              // Input the symbol prefix in case sensitive (if any)
input string           sym_suffix = "";              // Input the symbol suffix in case sensitive (if any)
//--
input group               "=== Money Management Lot Size Parameter ==="; // Money Management Lot Size Parameter
input mmt                  mmlot = DynamLot;         // Money Management Type
input double                Risk = 10.0;             // Percent Equity Risk per Trade (Min=1.0% / Max=10.0%)
input double                Lots = 0.01;             // Input Manual Lot Size FixedLot
//--Trade on Specific Time
input group               "=== Trade on Specific Time ==="; // Trade on Specific Time
input YN           trd_time_zone = Yes;              // Select If You Like to Trade on Specific Time Zone
input tm_zone            session = Cus_Session;      // Select Trading Time Zone
input swhour            stsescuh = hr_00;            // Time Hour to Start Trading Custom Session (0-23)
input inmnt             stsescum = mn_15;            // Time Minute to Start Trading Custom Session (0-55)
input swhour            clsescuh = hr_23;            // Time Hour to Stop Trading Custom Session (0-23)
input inmnt             clsescum = mn_55;            // Time Minute to Stop Trading Custom Session (0-55)
//--Day Trading On/Off
input group               "=== Day Trading On/Off ==="; // Day Trading On/Off
input YN                    ttd0 = No;               // Select Trading on Sunday (Yes) or (No)
input YN                    ttd1 = Yes;              // Select Trading on Monday (Yes) or (No)
input YN                    ttd2 = Yes;              // Select Trading on Tuesday (Yes) or (No)
input YN                    ttd3 = Yes;              // Select Trading on Wednesday (Yes) or (No)
input YN                    ttd4 = Yes;              // Select Trading on Thursday (Yes) or (No)
input YN                    ttd5 = Yes;              // Select Trading on Friday (Yes) or (No)
input YN                    ttd6 = No;               // Select Trading on Saturday (Yes) or (No)
//--Trade & Order management Parameter
input group               "=== Trade & Order management Parameter ==="; // Trade & Order management Parameter
input YN                  use_sl = No;               // Use Order Stop Loss (Yes) or (No)
input YN                  autosl = Yes;              // Use Automatic Calculation Stop Loss (Yes) or (No)
input double               SLval = 30;               // If Not Use Automatic SL - Input SL value in Pips
input YN                  use_tp = Yes;               // Use Order Take Profit (Yes) or (No)
input YN                  autotp = Yes;              // Use Automatic Calculation Take Profit (Yes) or (No)
input double               TPval = 10;               // If Not Use Automatic TP - Input TP value in Pips
input YN            TrailingSLTP = Yes;              // Use Trailing SL/TP (Yes) or (No)
input YN                 autotrl = Yes;              // Use Automatic Trailing (Yes) or (No)
input double               TSval = 5;                // If Not Use Automatic Trailing Input Trailing value in Pips
input double               TSmin = 5;                // Minimum Pips to start Trailing Stop
input double               TPmin = 25;               // Input Trailing Profit Value in Pips
input YN           Close_by_Opps = Yes;              // Close Trade By Opposite Signal (Yes) or (No)
input YN               SaveOnRev = Yes;              // Close Trade and Save profit due to weak signal (Yes) or (No)
//--Others Expert Advisor Parameter
input group               "=== Others Expert Advisor Parameter ==="; // Others EA Parameter
input YN                  alerts = Yes;              // Display Alerts / Messages (Yes) or (No)
input YN           UseEmailAlert = No;               // Email Alert (Yes) or (No)
input YN           UseSendnotify = No;               // Send Notification (Yes) or (No)
input YN      trade_info_display = Yes;              // Select Display Trading Info on Chart (Yes) or (No)
input ulong               magicEA = 2023111;          // Expert ID (Magic Number)
//---
```

In the expert input property group Global Strategy EA Parameter, the traders must choose whether to use multi-timeframe or single-timeframe signal calculations.

If the trader choose a single-timeframe (STF), then the trader must determine the timeframe that will be used.

In the expert input property, you must Select Single Calculation timeframe, default PERIOD\_H1.

If the trader choose multi-timeframe (MTF), then the trader must determine the timeframe series that will be used.

In the expert input property, you are instructed to Select Multi Timeframe calculation start and Select Multi Timeframe calculation end.

In the TriangularMA\_MTF\_MCEA\_Config() function lines 478 to 518, it is explained how to handle multi-timeframe and single-timeframe.

```
    ENUM_TIMEFRAMES TFs[]={PERIOD_M5,PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H2,PERIOD_H3,PERIOD_H4,PERIOD_H6,PERIOD_H8,PERIOD_H12,PERIOD_D1};
    int arTFs=ArraySize(TFs);
    //--
    for(int x=0; x<arTFs; x++)
      {
        if(singletf==x) TFt=TFs[x]; // TF for single-timeframe
        if(tfstart==x)  arstr=x;    // multi-timeframe start calculation timeframe
        if(tfclose==x)  arend=x;    // multi-timeframe end calculation timeframe
      }
    //--
    if(arstr>=arend)
      {
        Alert("Error selecting Start and End Timeframe, Start Timeframe must be smaller than End Timeframe");
        Alert("-- "+expname+" -- ",Symbol()," -- expert advisor will be Remove from the chart.");
        ExpertRemove();
      }
    //--
    switch(tfinuse)
      {
        case MTF:
          {
            TFArrays=arend-arstr+1;
            ArrayResize(TFTri,TFArrays,TFArrays);
            ArrayCopy(TFTri,TFs,0,0,WHOLE_ARRAY);
            tfcinws=arstr+1;
            tftrlst=(int)TFArrays/2;
            TFts=TFs[tftrlst+arstr-1];   // TF for Trailing Stop
            TFCWS=TFs[tfcinws];          // TF for Close Order in weak signal
            break;
          }
        case STF:
          {
            TFArrays=arTFs;
            ArrayResize(TFTri,TFArrays,TFArrays);
            ArrayCopy(TFTri,TFs,0,0,WHOLE_ARRAY);
            tfcinws=TFIndexArray(TFt)-2 <=0 ? 1 : TFIndexArray(TFt)-2;
            TFts=TFt;            // TF for Trailing Stop
            TFCWS=TFs[tfcinws];  // TF for Close Order in weak signal
            break;
          }
      }
```

The variable ENUM\_TIMEFRAMES TFs\[\], must be inherent to the enumeration option enum TFMTF

```
ENUM_TIMEFRAMES TFs[]={PERIOD_M5,PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H2,PERIOD_H3,PERIOD_H4,PERIOD_H6,PERIOD_H8,PERIOD_H12,PERIOD_D1};

//--
enum TFMTF
  {
   TFM5,     // PERIOD_M5
   TFM15,    // PERIOD_M15
   TFM30,    // PERIOD_M30
   TFH1,     // PERIOD_H1
   TFH2,     // PERIOD_H2
   TFH3,     // PERIOD_H3
   TFH4,     // PERIOD_H4
   TFH6,     // PERIOD_H6
   TFH8,     // PERIOD_H8
   TFH12,    // PERIOD_H12
   TFD1      // PERIOD_D1
  };
//--
```

Then the trader must determine the Triangular MA Indicator period, default period 14.

In addition, traders must also specify Triangular MA Applied Price, default PRICE\_CLOSE.

In the expert input property group Select Pairs to Trade, the traders must choose the pair to trade from the 10 options provided, by default All Forex 30 Pairs is determined.

To configure the pair to be traded, we will call the HandlingSymbolArrays() function.

With HandlingSymbolArrays() function we will handle all pairs that will be traded.

```
void MCEA::HandlingSymbolArrays(void)
  {
//---
    string All30[]={"EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY","EURGBP",
                    "EURAUD","EURNZD","EURCAD","EURCHF","EURJPY","GBPAUD","GBPNZD","GBPCAD",
                    "GBPCHF","GBPJPY","AUDNZD","AUDCAD","AUDCHF","AUDJPY","NZDCAD","NZDCHF",
                    "NZDJPY","CADCHF","CADJPY","CHFJPY","XAUUSD","XAGUSD"}; // 30 pairs
    string USDs[]={"USDCAD","USDCHF","USDJPY","AUDUSD","EURUSD","GBPUSD","NZDUSD","XAUUSD","XAGUSD"}; // USD pairs
    string EURs[]={"EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD"}; // EUR pairs
    string GBPs[]={"GBPAUD","GBPCAD","GBPCHF","EURGBP","GBPJPY","GBPNZD","GBPUSD"}; // GBP pairs
    string AUDs[]={"AUDCAD","AUDCHF","EURAUD","GBPAUD","AUDJPY","AUDNZD","AUDUSD"}; // AUD pairs
    string NZDs[]={"AUDNZD","NZDCAD","NZDCHF","EURNZD","GBPNZD","NZDJPY","NZDUSD"}; // NZD pairs
    string CADs[]={"AUDCAD","CADCHF","EURCAD","GBPCAD","CADJPY","NZDCAD","USDCAD"}; // CAD pairs
    string CHFs[]={"AUDCHF","CADCHF","EURCHF","GBPCHF","NZDCHF","CHFJPY","USDCHF"}; // CHF pairs
    string JPYs[]={"AUDJPY","CADJPY","CHFJPY","EURJPY","GBPJPY","NZDJPY","USDJPY"}; // JPY pairs
    //--
    sall=ArraySize(All30);
    arusd=ArraySize(USDs);
    aretc=ArraySize(EURs);
    ArrayResize(VSym,sall,sall);
    ArrayCopy(VSym,All30,0,0,WHOLE_ARRAY);
    //--
    if(usepairs==TrdWi && StringFind(traderwishes,"eg.",0)<0)
      {
        string to_split=traderwishes; // A string to split into substrings pairs name
        string sep=",";               // A separator as a character
        ushort u_sep;                 // The code of the separator character
        //--- Get the separator code
        u_sep=StringGetCharacter(sep,0);
        //--- Split the string to substrings
        int p=StringSplit(to_split,u_sep,SPC);
        if(p>0)
          {
            for(int i=0; i<p; i++) StringToUpper(SPC[i]);
            //--
            for(int i=0; i<p; i++)
              {
                if(ValidatePairs(SPC[i])<0) ArrayRemove(SPC,i,1);
              }
          }
        arspc=ArraySize(SPC);
      }
    //--
    SetSymbolNamePS();      // With this function we will detect whether the Symbol Name has a prefix and/or suffix
    //--
    if(inpre>0 || insuf>0)
      {
        if(usepairs==TrdWi && arspc>0)
          {
            for(int t=0; t<arspc; t++)
              {
                SPC[t]=pre+SPC[t]+suf;
              }
          }
        //--
        for(int t=0; t<sall; t++)
          {
            All30[t]=pre+All30[t]+suf;
          }
        for(int t=0; t<arusd; t++)
          {
            USDs[t]=pre+USDs[t]+suf;
          }
        for(int t=0; t<aretc; t++)
          {
            EURs[t]=pre+EURs[t]+suf;
          }
        for(int t=0; t<aretc; t++)
          {
            GBPs[t]=pre+GBPs[t]+suf;
          }
        for(int t=0; t<aretc; t++)
          {
            AUDs[t]=pre+AUDs[t]+suf;
          }
        for(int t=0; t<aretc; t++)
          {
            NZDs[t]=pre+NZDs[t]+suf;
          }
        for(int t=0; t<aretc; t++)
          {
            CADs[t]=pre+CADs[t]+suf;
          }
        for(int t=0; t<aretc; t++)
          {
            CHFs[t]=pre+CHFs[t]+suf;
          }
        for(int t=0; t<aretc; t++)
          {
            JPYs[t]=pre+JPYs[t]+suf;
          }
      }
    //--
    ArrayCopy(VSym,All30,0,0,WHOLE_ARRAY);
    ArrayResize(AS30,sall,sall);
    ArrayCopy(AS30,All30,0,0,WHOLE_ARRAY);
    for(int x=0; x<sall; x++) {SymbolSelect(AS30[x],true);}
    if(ValidatePairs(Symbol())>=0) symbfix=true;
    if(!symbfix)
      {
        Alert("Expert Advisors will not trade on pairs "+Symbol());
        Alert("-- "+expname+" -- ",Symbol()," -- expert advisor will be Remove from the chart.");
        ExpertRemove();
      }
    //--
    switch(usepairs)
      {
        case 0: // All Forex 30 Pairs
          {
            ArrayResize(DIRI,sall,sall);
            arrsymbx=sall;
            ArraySymbolResize();
            ArrayCopy(DIRI,All30,0,0,WHOLE_ARRAY);
            pairs="Multi Currency 30 Pairs";
            //--
            break;
          }
        case 1: // Trader wishes pairs
          {
            ArrayResize(DIRI,arspc,arspc);
            arrsymbx=arspc;
            ArraySymbolResize();
            ArrayCopy(DIRI,SPC,0,0,WHOLE_ARRAY);
            pairs="("+string(arspc)+") Trader Wishes Pairs";
            //--
            break;
          }
        case 2: // USD pairs
          {
            ArrayResize(DIRI,arusd,arusd);
            arrsymbx=arusd;
            ArraySymbolResize();
            ArrayCopy(DIRI,USDs,0,0,WHOLE_ARRAY);
            pairs="("+string(arusd)+") Multi Currency USD Pairs";
            //--
            break;
          }
        case 3: // EUR pairs
          {
            ArrayResize(DIRI,aretc,aretc);
            arrsymbx=aretc;
            ArraySymbolResize();
            ArrayCopy(DIRI,EURs,0,0,WHOLE_ARRAY);
            pairs="("+string(aretc)+") Forex EUR Pairs";
            //--
            break;
          }
        case 4: // GBP pairs
          {
            ArrayResize(DIRI,aretc,aretc);
            arrsymbx=aretc;
            ArraySymbolResize();
            ArrayCopy(DIRI,GBPs,0,0,WHOLE_ARRAY);
            pairs="("+string(aretc)+") Forex GBP Pairs";
            //--
            break;
          }
        case 5: // AUD pairs
          {
            ArrayResize(DIRI,aretc,aretc);
            arrsymbx=aretc;
            ArraySymbolResize();
            ArrayCopy(DIRI,AUDs,0,0,WHOLE_ARRAY);
            pairs="("+string(aretc)+") Forex AUD Pairs";
            //--
            break;
          }
        case 6: // NZD pairs
          {
            ArrayResize(DIRI,aretc,aretc);
            arrsymbx=aretc;
            ArraySymbolResize();
            ArrayCopy(DIRI,NZDs,0,0,WHOLE_ARRAY);
            pairs="("+string(aretc)+") Forex NZD Pairs";
            //--
            break;
          }
        case 7: // CAD pairs
          {
            ArrayResize(DIRI,aretc,aretc);
            arrsymbx=aretc;
            ArraySymbolResize();
            ArrayCopy(DIRI,CADs,0,0,WHOLE_ARRAY);
            pairs="("+string(aretc)+") Forex CAD Pairs";
            //--
            break;
          }
        case 8: // CHF pairs
          {
            ArrayResize(DIRI,aretc,aretc);
            arrsymbx=aretc;
            ArraySymbolResize();
            ArrayCopy(DIRI,CHFs,0,0,WHOLE_ARRAY);
            pairs="("+string(aretc)+") Forex CHF Pairs";
            //--
            break;
          }
        case 9: // JPY pairs
          {
            ArrayResize(DIRI,aretc,aretc);
            arrsymbx=aretc;
            ArraySymbolResize();
            ArrayCopy(DIRI,JPYs,0,0,WHOLE_ARRAY);
            pairs="("+string(aretc)+") Forex JPY Pairs";
            //--
            break;
          }
      }
    //--
    return;
//---
  } //-end HandlingSymbolArrays()
//---------//
```

Inside the HandlingSymbolArrays() function we will call the SetSymbolNamePS() function.

With the SetSymbolNamePS() function, we will be able to handle symbol names that have prefixes and/or suffixes.

```
void MCEA::SetSymbolNamePS(void)
  {
//---
   symbfix=false;
   int ptriml;
   int ptrimr;
   string insymbol=Symbol();
   int sym_Lenpre=StringLen(prefix);
   int sym_Lensuf=StringLen(suffix);
   if(sym_Lenpre>0)
     {
       ptriml=StringTrimLeft(suffix);
       ptriml=StringTrimRight(suffix);
     }
   if(sym_Lensuf>0)
     {
       ptrimr=StringTrimLeft(suffix);
       ptrimr=StringTrimRight(suffix);
     }
   string sym_pre=prefix;
   string sym_suf=suffix;
   //--
   pre=sym_pre;
   suf=sym_suf;
   inpre=StringLen(pre);
   insuf=StringLen(suf);
   posCur1=inpre;
   posCur2=posCur1+3;
   //--
   return;
//---
  } //-end SetSymbolNamePS()
//---------//
```

Note:

The expert will validate the pairs.

If the trader makes a mistake in entering the pair name or pair prefix name and/or pair suffix name (typos) or

if pair validation fails, the expert will give a warning and the expert advisor will be removed from the chart.

In the expert input property group Trade on Specific Time, here the trader will choose to Trade on Specific Time Zone (Yes) or (No)

and If Yes, select the enumeration options:

- Trading on Custom Session
- Trading on New Zealand Session
- Trading on Australia Sydney Session
- Trading on Asia Tokyo Session
- Trading on Europe London Session
- Trading on America New York Session

Trading on Custom Session: In this session, traders must set the time or hours and minutes to start trading and the hours and minutes to close trading.

So the EA will only carry out activities during the specified time from start to close.

In Trading on New Zealand Session to Trading on US New York Session, the time from start of trading to close of trading is calculated by the EA.

To declare all variables, objects and functions needed in this Multi-Currency Expert Advisor, we will create a Class to specify the construction and configuration in the expert advisor workflow.

In particular, the variables used in the function for handle prefix symbol names and/or suffix symbol names as well as time zone calculations we made it in the MCEA Class.

```
//+------------------------------------------------------------------+
//| Class for working Expert Advisor                                 |
//+------------------------------------------------------------------+
class MCEA
  {
//---
    private:
    //----
    int              x_year;       // Year
    int              x_mon;        // Month
    int              x_day;        // Day of the month
    int              x_hour;       // Hour in a day
    int              x_min;        // Minutes
    int              x_sec;        // Seconds
    //--
    int              oBm,
                     oSm,
                     ldig;
    //--- Variables used in prefix and suffix symbols
    int              posCur1,
                     posCur2;
    int              inpre,
                     insuf;
    bool             symbfix;
    string           pre,suf;
    string           prefix,suffix;
    //--- Variables are used in Trading Time Zone
    int              ishour,
                     onhour;
    int              tftrlst,
                     tfcinws;
    datetime         rem,
                     znop,
                     zncl,
                     zntm;
    datetime         SesCuOp,
                     SesCuCl,
                     Ses01Op,
                     Ses01Cl,
                     Ses02Op,
                     Ses02Cl,
                     Ses03Op,
                     Ses03Cl,
                     Ses04Op,
                     Ses04Cl,
                     Ses05Op,
                     Ses05Cl,
                     SesNoOp,
                     SesNoCl;
    //--
    string           tz_ses,
                     tz_opn,
                     tz_cls;
    //--
    string           tmopcu,
                     tmclcu,
                     tmop01,
                     tmcl01,
                     tmop02,
                     tmcl02,
                     tmop03,
                     tmcl03,
                     tmop04,
                     tmcl04,
                     tmop05,
                     tmcl05,
                     tmopno,
                     tmclno;
    //----------------------
    //--
    double           LotPS;
    double           slv,
                     tpv,
                     pip,
                     xpip;
    double           floatprofit,
                     fixclprofit;
    //--
    string           pairs,
                     hariini,
                     daytrade,
                     trade_mode;
    //--
    double           OPEN[],
                     HIGH[],
                     LOW[],
                     CLOSE[];
    datetime         TIME[];
    datetime         closetime;
    //--
    //------------

    //------------
    void             SetSymbolNamePS(void);
    void             HandlingSymbolArrays(void);
    void             Set_Time_Zone(void);
    void             Time_Zone(void);
    bool             Trade_session(void);
    string           PosTimeZone(void);
    int              ThisTime(const int reqmode);
    int              ReqTime(datetime reqtime,const int reqmode);
    //--
    int              DirectionMove(const string symbol,const ENUM_TIMEFRAMES stf);
    int              TriaMASMTF(const string symbol,ENUM_TIMEFRAMES mtf);
    int              GetTriaMASignalMTF(string symbol);
    int              TriaMASignalSTF(const string symbol);
    int              LotDig(const string symbol);
    //--
    double           MLots(const string symbx);
    double           NonZeroDiv(double val1,double val2);
    double           OrderSLSet(const string xsymb,ENUM_ORDER_TYPE type,double atprice);
    double           OrderTPSet(const string xsymb,ENUM_ORDER_TYPE type,double atprice);
    double           SetOrderSL(const string xsymb,ENUM_POSITION_TYPE type,double atprice);
    double           SetOrderTP(const string xsymb,ENUM_POSITION_TYPE type,double atprice);
    double           TSPrice(const string xsymb,ENUM_POSITION_TYPE ptype,int TS_type);
    //--
    string           ReqDate(int d,int h,int m);
    string           TF2Str(ENUM_TIMEFRAMES period);
    string           timehr(int hr,int mn);
    string           TradingDay(void);
    string           AccountMode();
    string           GetCommentForOrder(void)             { return(expname); }
    //------------

    public:
    //---

    //-- TriangularMA_MTF_MCEA Config --
    string           DIRI[],
                     AS30[],
                     VSym[];
    string           SPC[];
    string           USD[];
    string           EUR[];
    string           GBP[];
    string           AUD[];
    string           NZD[];
    string           CAD[];
    string           CHF[];
    string           JPY[];
    //--
    string           expname;
    string           indiname;
    //--
    int              hTriMAt[];
    int              hTriMAs[];
    int              hTriMAm[];
    int              hTriMAb[][11];
    int              ALO,
                     dgts,
                     arrsar,
                     arrsymbx;
    int              sall,
                     arusd,
                     aretc,
                     arspc,
                     arper;
    ulong            slip;
    //--
    double           profitb[],
                     profits[];
    //--
    int              Buy,
                     Sell;
    int              ccur,
                     psec,
                     xtto,
                     TFArrays,
                     checktml;
    int              OpOr[],xob[],xos[];
    //--
    int              year,  // Year
                     mon,   // Month
                     day,   // Day
                     hour,  // Hour
                     min,   // Minutes
                     sec,   // Seconds
                     dow,   // Day of week (0-Sunday, 1-Monday, ... ,6-Saturday)
                     doy;   // Day number of the year (January 1st is assigned the number value of zero)
    //--
    ENUM_TIMEFRAMES  TFt,
                     TFts,
                     TFT05,
                     TFCWS;
    ENUM_TIMEFRAMES  TFTri[];
    //--
    bool             PanelExtra;
    //------------
                     MCEA(void);
                     ~MCEA(void);
    //------------
    //--
    virtual void     TriangularMA_MTF_MCEA_Config(void);
    virtual void     ExpertActionTrade(void);
    //--
    void             ArraySymbolResize(void);
    void             CurrentSymbolSet(const string symbol);
    void             Pips(const string symbol);
    void             TradeInfo(void);
    void             Do_Alerts(const string symbx,string msgText);
    void             CheckOpenPMx(const string symbx);
    void             SetSLTPOrders(void);
    void             CloseBuyPositions(const string symbol);
    void             CloseSellPositions(const string symbol);
    void             CloseAllOrders(void);
    void             CheckClose(const string symbx);
    void             TodayOrders(void);
    void             UpdatePrice(const string symbol,ENUM_TIMEFRAMES xtf);
    void             RefreshPrice(const string symbx,ENUM_TIMEFRAMES xtf,int bars);
    //--
    bool             RefreshTick(const string symbx);
    bool             TradingToday(void);
    bool             OpenBuy(const string symbol);
    bool             OpenSell(const string symbol);
    bool             ModifyOrderSLTP(double mStop,double ordtp);
    bool             ModifySLTP(const string symbx,int TS_type);
    bool             CloseAllProfit(void);
    bool             ManualCloseAllProfit(void);
    //--
    int              PairsIdxArray(const string symbol);
    int              ValidatePairs(const string symbol);
    int              TFIndexArray(ENUM_TIMEFRAMES TF);
    int              GetOpenPosition(const string symbol);
    int              GetSignalMidTF(const string symbol);
    int              GetCloseInWeakSignal(const string symbol,int exis);
    //--
    string           getUninitReasonText(int reasonCode);
    //--
    //------------
//---
  }; //-end class MCEA
//---------//
```

The very first and foremost function in the Multi-Currency Expert Advisor work process that is called from OnInit() is TriangularMA\_MTF\_MCEA\_Config().

In the TriangularMA\_MTF\_MCEA\_Config() function, all symbols to be used are configured, all handle indicators used and some important functions of the include file header for the expert advisor workflow.

```
void MCEA::TriangularMA_MTF_MCEA_Config(void)
  {
//---
    //--
    HandlingSymbolArrays(); // With this function we will handle all pairs that will be traded
    //--
    int arstr=0,
        arend=0;
    TFT05=PERIOD_M5;
    ENUM_TIMEFRAMES TFs[]={PERIOD_M5,PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H2,PERIOD_H3,PERIOD_H4,PERIOD_H6,PERIOD_H8,PERIOD_H12,PERIOD_D1};
    int arTFs=ArraySize(TFs);
    //--
    for(int x=0; x<arTFs; x++)
      {
        if(singletf==x) TFt=TFs[x]; // TF for single-timeframe
        if(tfstart==x)  arstr=x;    // multi-timeframe start calculation timeframe
        if(tfclose==x)  arend=x;    // multi-timeframe end calculation timeframe
      }
    //--
    if(arstr>=arend)
      {
        Alert("Error selecting Start and End Timeframe, Start Timeframe must be smaller than End Timeframe");
        Alert("-- "+expname+" -- ",Symbol()," -- expert advisor will be Remove from the chart.");
        ExpertRemove();
      }
    //--
    switch(tfinuse)
      {
        case MTF:
          {
            TFArrays=arend-arstr+1;
            ArrayResize(TFTri,TFArrays,TFArrays);
            ArrayCopy(TFTri,TFs,0,0,WHOLE_ARRAY);
            tfcinws=arstr+1;
            tftrlst=(int)TFArrays/2;
            TFts=TFs[tftrlst+arstr-1];   // TF for Trailing Stop
            TFCWS=TFs[tfcinws];          // TF for Close Order in weak signal
            break;
          }
        case STF:
          {
            TFArrays=arTFs;
            ArrayResize(TFTri,TFArrays,TFArrays);
            ArrayCopy(TFTri,TFs,0,0,WHOLE_ARRAY);
            tfcinws=TFIndexArray(TFt)-2 <=0 ? 1 : TFIndexArray(TFt)-2;
            TFts=TFt;            // TF for Trailing Stop
            TFCWS=TFs[tfcinws];  // TF for Close Order in weak signal
            break;
          }
      }
    //--
    //-- Triangular MA Indicators handle for all symbol
    for(int x=0; x<arrsymbx; x++)
      {
        hTriMAs[x]=iCustom(DIRI[x],TFT05,indiname,Trmaperiod,Trprice);
        hTriMAm[x]=iCustom(DIRI[x],TFCWS,indiname,Trmaperiod,Trprice);
        hTriMAt[x]=iCustom(DIRI[x],TFts,indiname,Trmaperiod,Trprice);
        //--
        for(int i=0; i<TFArrays; i++)
          {
            if(tfinuse==MTF) // MTF indicator handle
              {
                hTriMAb[x][i]=iCustom(DIRI[x],TFTri[i],indiname,Trmaperiod,Trprice);
              }
            if(tfinuse==STF)
              {
                if(TFs[i]==TFt) // Single-TF indicator handle
                  {
                    hTriMAb[x][i]=iCustom(DIRI[x],TFs[i],indiname,Trmaperiod,Trprice);
                    break;
                  }
              }
          }
      }
    //--
    ALO=(int)mc_account.LimitOrders()>sall ? sall : (int)mc_account.LimitOrders();
    //--
    LotPS=(double)ALO;
    //--
    mc_trade.SetExpertMagicNumber(magicEA);
    mc_trade.SetDeviationInPoints(slip);
    mc_trade.SetMarginMode();
    Set_Time_Zone();
    //--
    return;
//---
  } //-end TriangularMA_MTF_MCEA_Config()
//---------//
```

**2\. Expert tick function**

Inside the Expert tick function (OnTick() function) we will call one of the main functions in a multi-currency expert advisor namely function ExpertActionTrade().

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(void)
  {
//---
    mc.ExpertActionTrade();
    //--
    return;
//---
  } //-end OnTick()
//---------//
```

The sequence of the EA work process inside this function.

The ExpertActionTrade() function will carry out all activities and manage automatic trading, starting from Open Orders, Close Orders, Trailing Stop or Trading Profits and other additional activities.

```
void MCEA::ExpertActionTrade(void)
  {
//---
    //--Check Trading Terminal
    ResetLastError();
    //--
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED) && mc.checktml==0) //-- Check whether MT5 Algorithmic trading is Allow or Prohibit
      {
        mc.Do_Alerts(Symbol(),"Trading Expert at "+Symbol()+" are NOT Allowed by Setting.");
        mc.checktml=1;  //-- Variable checktml is given a value of 1, so that the alert is only done once.
        return;
      }
    //--
    if(!DisplayManualButton("M","C","R")) DisplayManualButton(); //-- Show the expert manual button panel
    //--
    if(trade_info_display==Yes) mc.TradeInfo(); //-- Displayed Trading Info on Chart
    //---
    //--
    int mcsec=mc.ThisTime(mc.sec);
    //--
    if(fmod((double)mcsec,5.0)==0) mc.ccur=mcsec;
    //--
    if(mc.ccur!=mc.psec)
      {
        string symbol;
        //-- Here we start with the rotation of the name of all symbol or pairs to be traded
        for(int x=0; x<mc.arrsymbx && !IsStopped(); x++)
          {
            //--
            if(mc.DIRI[x]==Symbol()) symbol=Symbol();
            else symbol=mc.DIRI[x];
            //--
            mc.CurrentSymbolSet(symbol);
            //--
            if(mc.TradingToday() && mc.Trade_session())
              {
                //--
                mc.OpOr[x]=mc.GetOpenPosition(symbol); //-- Get trading signals to open positions
                //--                                   //-- and store in the variable OpOr[x]
                if(mc.OpOr[x]==mc.Buy) //-- If variable OpOr[x] get result of GetOpenPosition(symbol) as "Buy" (value=1)
                  {
                    //--
                    mc.CheckOpenPMx(symbol);
                    //--
                    if(Close_by_Opps==Yes && mc.xos[x]>0) mc.CloseSellPositions(symbol);
                    //--
                    if(mc.xob[x]==0 && mc.xtto<mc.ALO) mc.OpenBuy(symbol);
                    else
                    if(mc.xtto>=mc.ALO)
                      {
                        //--
                        mc.Do_Alerts(symbol,"Maximum amount of open positions and active pending orders has reached"+
                                            "\n the limit = "+string(mc.ALO)+" Orders ");
                        //--
                        mc.CheckOpenPMx(symbol);
                        //--
                        if(mc.xos[x]>0 && mc.profits[x]<-1.02 && mc.xob[x]==0) {mc.CloseSellPositions(symbol); mc.OpenBuy(symbol);}
                        else
                        if(SaveOnRev==Yes) mc.CloseAllProfit();
                      }
                  }
                if(mc.OpOr[x]==mc.Sell) //-- If variable OpOr[x] get result of GetOpenPosition(symbol) as "Sell" (value=-1)
                  {
                    //--
                    mc.CheckOpenPMx(symbol);
                    //--
                    if(Close_by_Opps==Yes && mc.xob[x]>0) mc.CloseBuyPositions(symbol);
                    //--
                    if(mc.xos[x]==0 && mc.xtto<mc.ALO) mc.OpenSell(symbol);
                    else
                    if(mc.xtto>=mc.ALO)
                      {
                        //--
                        mc.Do_Alerts(symbol,"Maximum amount of open positions and active pending orders has reached"+
                                            "\n the limit = "+string(mc.ALO)+" Orders ");
                        //--
                        mc.CheckOpenPMx(symbol);
                        //--
                        if(mc.xob[x]>0 && mc.profitb[x]<-1.02 && mc.xos[x]==0) {mc.CloseBuyPositions(symbol); mc.OpenSell(symbol);}
                        else
                        if(SaveOnRev==Yes) mc.CloseAllProfit();
                      }
                  }
              }
            //--
            mc.CheckOpenPMx(symbol);
            //--
            if(mc.xtto>0)
              {
                //--
                if(SaveOnRev==Yes) //-- Close Trade and Save profit due to weak signal (Yes)
                  {
                    mc.CheckOpenPMx(symbol);
                    if(mc.profitb[x]>0.02 && mc.xob[x]>0 && mc.GetCloseInWeakSignal(symbol,mc.Buy)==mc.Sell)
                      {
                        mc.CloseBuyPositions(symbol);
                        mc.Do_Alerts(symbol,"Close BUY order "+symbol+" to save profit due to weak signal.");
                      }
                    if(mc.profits[x]>0.02 && mc.xos[x]>0 && mc.GetCloseInWeakSignal(symbol,mc.Sell)==mc.Buy)
                      {
                        mc.CloseSellPositions(symbol);
                        mc.Do_Alerts(symbol,"Close SELL order "+symbol+" to save profit due to weak signal.");
                      }
                  }
                //--
                if(TrailingSLTP==Yes) //-- Use Trailing SL/TP (Yes)
                  {
                    if(autotrl==Yes) mc.ModifySLTP(symbol,1); //-- If Use Automatic Trailing (Yes)
                    if(autotrl==No)  mc.ModifySLTP(symbol,0); //-- Use Automatic Trailing (No)
                  }
              }
            //--
            mc.CheckClose(symbol);
          }
        //--
        mc.psec=mc.ccur;
      }
    //--
    return;
//---
  } //-end ExpertActionTrade()
//---------//
```

Specifically for Time Zone Trading, in the ExpertActionTrade() function, a call to the boolean Trade\_session() function is added.

If Trade\_session() is true, then the EA work process will continue until it is finished, but if it is false, then the EA will only carry out the tasks "Close Trade and Save profit due to weak signal (Yes)" and "Trailing Stop (Yes)".

```
bool MCEA::Trade_session(void)
  {
//---
   bool trd_ses=false;
   ishour=ThisTime(hour);
   if(ishour!=onhour) Set_Time_Zone();
   datetime tcurr=TimeCurrent(); // Server Time
   //--
   switch(session)
     {
       case Cus_Session:
         {
           if(tcurr>=SesCuOp && tcurr<=SesCuCl) trd_ses=true;
           break;
         }
       case New_Zealand:
         {
           if(tcurr>=Ses01Op && tcurr<=Ses01Cl) trd_ses=true;
           break;
         }
       case Australia:
         {
           if(tcurr>=Ses02Op && tcurr<=Ses02Cl) trd_ses=true;
           break;
         }
       case Asia_Tokyo:
         {
           if(tcurr>=Ses03Op && tcurr<=Ses03Cl) trd_ses=true;
           break;
         }
       case Europe_London:
         {
           if(tcurr>=Ses04Op && tcurr<=Ses04Cl) trd_ses=true;
           break;
         }
       case US_New_York:
         {
           if(tcurr>=Ses05Op && tcurr<=Ses05Cl) trd_ses=true;
           break;
         }
     }
   //--
   if(trd_time_zone==No)
     {
      if(tcurr>=SesNoOp && tcurr<=SesNoCl) trd_ses=true;
     }
   //--
   onhour=ishour;
   //--
   return(trd_ses);
//---
  } //-end Trade_session()
//---------//
```

**3\. How to get trading signals for open positions?**

To get a signal, the ExpertActionTrade() function will call the GetOpenPosition() function.

```
int MCEA::GetOpenPosition(const string symbol) // Signal Open Position
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    int trimOp=GetTriaMASignalMTF(symbol);
    int getmid=GetSignalMidTF(symbol);
    if(trimOp==rise && getmid==rise) ret=rise;
    if(trimOp==down && getmid==down) ret=down;
    //--
    return(ret);
//---
  } //-end GetOpenPosition()
//---------//
```

GetOpenPosition() function will call 2 functions that perform signal calculations:

1. GetSignalMidTF(const string symbol);          //-- Function to get signals on the middle timeframe and price movement positions

```
int MCEA::GetSignalMidTF(const string symbol) // Signal Indicator Position Close in profit
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    int br=2;
    //--
    double TriMACI[];
    //--
    ArrayResize(TriMACI,br,br);
    ArraySetAsSeries(TriMACI,true);
    int xx=PairsIdxArray(symbol);
    CopyBuffer(hTriMAm[xx],1,0,br,TriMACI);
    //#property indicator_color1  clrDarkGray,clrDeepPink,clrMediumSeaGreen
    //                                 0          1             2
    //--
    int dirmove=DirectionMove(symbol,TFCWS);
    //--
    if(TriMACI[0]==2.0 && dirmove==rise) ret=rise;
    if(TriMACI[0]==1.0 && dirmove==down) ret=down;
    //--
    return(ret);
//---
  } //-end GetSignalMidTF()
//---------//
```

2\. GetTriaMASignalMTF(const string symbol);    //-- Function to calculation formula of Triangular moving average.

The GetTriaMASignalMTF() function will call a function TriaMASMTF() that calculates the Triangular moving average signal according to the requested timeframe.

```
int MCEA::GetTriaMASignalMTF(string symbol)
  {
//---
    int mv=0;
    int rise=1,
        down=-1;
    int tfloop=tfinuse==MTF ? TFArrays : 1;
    //--
    int trimup=0,
        trimdw=0;
    //--
    for(int x=0; x<tfloop; x++)
      {
        if(TriaMASMTF(symbol,TFTri[x])>0) trimup++;
        if(TriaMASMTF(symbol,TFTri[x])<0) trimdw++;
      }
    //--
    if(trimup==tfloop) mv=rise;
    if(trimdw==tfloop) mv=down;
    //--
    return(mv);
//---
  } //- end GetTriaMASignalMTF()
//---------//
```

```
int MCEA::TriaMASMTF(const string symbol,const ENUM_TIMEFRAMES mtf) // formula Triangular MA on the requested Timeframe
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    int br=3;
    ENUM_TIMEFRAMES TFUse=tfinuse==MTF ? mtf : TFt;
    //--
    double TriMACI[];
    ArrayResize(TriMACI,br,br);
    ArraySetAsSeries(TriMACI,true);
    int xx=PairsIdxArray(symbol);
    int tx=TFIndexArray(TFUse);
    CopyBuffer(hTriMAb[xx][tx],1,0,br,TriMACI);
    //#property indicator_color1  clrDarkGray,clrDeepPink,clrMediumSeaGreen
    //                                 0          1             2
    //Print("Symbol = "+symbol+" TF = "+EnumToString(mtf)+" TriMACI[0] = "+string(TriMACI[0]));
    //--
    switch(tfinuse)
      {
        case MTF:
          {
            if(TriMACI[0]==2.0) ret=rise;
            if(TriMACI[0]==1.0) ret=down;
            //--
            break;
          }
        case STF:
          {
            if(TriMACI[2]==1.0 && TriMACI[1]==2.0 && TriMACI[0]==2.0) ret=rise;
            if(TriMACI[2]==2.0 && TriMACI[1]==1.0 && TriMACI[0]==1.0) ret=down;
            //--
            break;
          }
      }
    //--
    return(ret);
//---
  } //-end TriaMASMTF()
//---------//
```

As you can see, inside the TriaMASMTF() function, we use and call 2 functions:

- 1\. int xx= PairsIdxArray(symbol)
- 2\. int tx=TFIndexArray(mtf).

The PairsIdxArray() function is used to get the name of the requested symbol, and the TFIndexArray() function is used to get the timeframe array sequence of the requested timeframe.

Then the appropriate indicator handle is called to get the buffer value of the Triangular moving average signal from that Timeframe.

As the author of the Triangular moving average indicator says:

"Usage:

You can use the color change as signal..."

So, how do you take the Triangular moving average indicator signal?

In the Triangular moving average indicator property:

```
#property indicator_color1  clrDarkGray,clrDeepPink,clrMediumSeaGreen
//                               0            1             2
```

```
SetIndexBuffer(1,valc,INDICATOR_COLOR_INDEX);
```

```
valc[i] = (i>0) ?(val[i]>val[i-1]) ? 2 :(val[i]<val[i-1]) ? 1 : valc[i-1]: 0;
```

So, we know that:

- 0-DarkGray = Unknown signal
- 1-DeepPink = Sell Signal
- 2-MediumSeaGreen = Buy Signal.

So, we can take the value of buffer 1 of the Triangular moving average indicator as a signal by CopyBuffer function as in TriaMASMTF() function.

```
    double TriMACI[];
    ArrayResize(TriMACI,br,br);
    ArraySetAsSeries(TriMACI,true);
    int xx=PairsIdxArray(symbol);
    int tx=TFIndexArray(TFUse);
    CopyBuffer(hTriMAb[xx][tx],1,0,br,TriMACI);
    //#property indicator_color1  clrDarkGray,clrDeepPink,clrMediumSeaGreen
    //                                 0           1             2
```

**4\. ChartEvent Function**

To support effectiveness and efficiency in the use of Multi-Currency Expert Advisors, it is deemed necessary to create one several manual buttons in managing orders

and changing charts or symbols.

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
//--- handling CHARTEVENT_CLICK event ("Clicking the chart")
   ResetLastError();
   //--
   ENUM_TIMEFRAMES CCS=mc.TFt;
   //--
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
       int lensymbol=StringLen(Symbol());
       int lensparam=StringLen(sparam);
       //--
       //--- if "Set SL All Orders" button is click
       if(sparam=="Set SL/TP All Orders")
         {
           mc.SetSLTPOrders();
           Alert("-- "+mc.expname+" -- ",Symbol()," -- Set SL/TP All Orders");
           //--- unpress the button
           ObjectSetInteger(0,"Set SL/TP All Orders",OBJPROP_STATE,false);
           ObjectSetInteger(0,"Set SL/TP All Orders",OBJPROP_ZORDER,0);
           CreateManualPanel();
         }
       //--- if "Close All Order" button is click
       if(sparam=="Close All Order")
         {
           mc.CloseAllOrders();
           Alert("-- "+mc.expname+" -- ",Symbol()," -- Close All Orders");
           //--- unpress the button
           ObjectSetInteger(0,"Close All Order",OBJPROP_STATE,false);
           ObjectSetInteger(0,"Close All Order",OBJPROP_ZORDER,0);
           CreateManualPanel();
         }
       //--- if "Close All Profit" button is click
       if(sparam=="Close All Profit")
         {
           mc.ManualCloseAllProfit();
           Alert("-- "+mc.expname+" -- ",Symbol()," -- Close All Profit");
           //--- unpress the button
           ObjectSetInteger(0,"Close All Profit",OBJPROP_STATE,false);
           ObjectSetInteger(0,"Close All Profit",OBJPROP_ZORDER,0);
           CreateManualPanel();
         }
       //--- if "X" button is click
       if(sparam=="X")
         {
           ObjectsDeleteAll(0,0,OBJ_BUTTON);
           ObjectsDeleteAll(0,0,OBJ_LABEL);
           ObjectsDeleteAll(0,0,OBJ_RECTANGLE_LABEL);
           //--- unpress the button
           ObjectSetInteger(0,"X",OBJPROP_STATE,false);
           ObjectSetInteger(0,"X",OBJPROP_ZORDER,0);
           //--
           DeleteButtonX();
           mc.PanelExtra=false;
           DisplayManualButton();
         }
       //--- if "M" button is click
       if(sparam=="M")
         {
           //--- unpress the button
           ObjectSetInteger(0,"M",OBJPROP_STATE,false);
           ObjectSetInteger(0,"M",OBJPROP_ZORDER,0);
           mc.PanelExtra=true;
           CreateManualPanel();
         }
       //--- if "C" button is click
       if(sparam=="C")
         {
           //--- unpress the button
           ObjectSetInteger(0,"C",OBJPROP_STATE,false);
           ObjectSetInteger(0,"C",OBJPROP_ZORDER,0);
           mc.PanelExtra=true;
           CreateSymbolPanel();
         }
       //--- if "R" button is click
       if(sparam=="R")
         {
           Alert("-- "+mc.expname+" -- ",Symbol()," -- expert advisor will be Remove from the chart.");
           ExpertRemove();
           //--- unpress the button
           ObjectSetInteger(0,"R",OBJPROP_STATE,false);
           ObjectSetInteger(0,"R",OBJPROP_ZORDER,0);
           if(!ChartSetSymbolPeriod(0,Symbol(),Period()))
             ChartSetSymbolPeriod(0,Symbol(),Period());
           DeletePanelButton();
           ChartRedraw(0);
         }
       //--- if Symbol button is click
       if(lensparam==lensymbol)
         {
           int sx=mc.ValidatePairs(sparam);
           ChangeChartSymbol(mc.AS30[sx],CCS);
           mc.PanelExtra=false;
         }
       //--
     }
    //--
    return;
//---
  } //-end OnChartEvent()
//---------//
```

To change chart symbols with one click, when one of the symbol names is clicked, the OnChartEvent() will be called function ChangeChartSymbol().

```
void ChangeChartSymbol(string c_symbol,ENUM_TIMEFRAMES cstf)
  {
//---
   //--- unpress the button
   ObjectSetInteger(0,c_symbol,OBJPROP_STATE,false);
   ObjectSetInteger(0,c_symbol,OBJPROP_ZORDER,0);
   ObjectsDeleteAll(0,0,OBJ_BUTTON);
   ObjectsDeleteAll(0,0,OBJ_LABEL);
   ObjectsDeleteAll(0,0,OBJ_RECTANGLE_LABEL);
   //--
   ChartSetSymbolPeriod(0,c_symbol,cstf);
   //--
   ChartRedraw(0);
   //--
   return;
//---
  } //-end ChangeChartSymbol()
//---------//
```

Because we are adding an expert trade mode multi-timeframe or single timeframe and trade session or trading time zone and options for the pairs that will be traded, we need additional information in the Displayed Trading Info on Chart.

To add information to the Displayed Trading Info on Chart, we made changes to the TradeInfo() function.

```
void MCEA::TradeInfo(void) // function: write comments on the chart
  {
//----
   Pips(Symbol());
   double spread=SymbolInfoInteger(Symbol(),SYMBOL_SPREAD)/xpip;
   rem=zntm-TimeCurrent();
   string postime=PosTimeZone();
   string eawait=" - Waiting for active time..!";
   //--
   string tradetf=tfinuse==MTF ? EnumToString(Period()) : EnumToString(TFts);
   string eamode=tfinuse==MTF ? "Multi-Timeframe" : "Single-Timeframe";
   string comm="";
   TodayOrders();
   //--
   comm="\n     :: Server Date Time : "+string(ThisTime(year))+"."+string(ThisTime(mon))+"."+string(ThisTime(day))+ "   "+TimeToString(TimeCurrent(),TIME_SECONDS)+
        "\n     ------------------------------------------------------------"+
        "\n      :: Broker               :  "+ TerminalInfoString(TERMINAL_COMPANY)+
        "\n      :: Expert Name      :  "+ expname+
        "\n      :: Acc. Name         :  "+ mc_account.Name()+
        "\n      :: Acc. Number      :  "+ (string)mc_account.Login()+
        "\n      :: Acc. TradeMode :  "+ AccountMode()+
        "\n      :: Acc. Leverage    :  1 : "+ (string)mc_account.Leverage()+
        "\n      :: Acc. Equity       :  "+ DoubleToString(mc_account.Equity(),2)+
        "\n      :: Margin Mode     :  "+ (string)mc_account.MarginModeDescription()+
        "\n      :: Magic Number   :  "+ string(magicEA)+
        "\n      :: Trade on TF      :  "+ tradetf+
        "\n      :: Trade Mode      :  "+ eamode+
        "\n      :: Today Trading   :  "+ TradingDay()+" : "+hariini+
        "\n      :: Trading Session :  "+ tz_ses+
        "\n      :: Trading Time    :  "+ postime;
        if(TimeCurrent()<zntm)
          {
            comm=comm+
            "\n      :: Time Remaining :  "+(string)ReqTime(rem,hour)+":"+(string)ReqTime(rem,min)+":"+(string)ReqTime(rem,sec) + eawait;
          }
        comm=comm+
        "\n     ------------------------------------------------------------"+
        "\n      :: Trading Pairs     :  "+pairs+
        "\n      :: BUY Market      :  "+string(oBm)+
        "\n      :: SELL Market     :  "+string(oSm)+
        "\n      :: Total Order       :  "+string(oBm+oSm)+
        "\n      :: Order Profit      :  "+DoubleToString(floatprofit,2)+
        "\n      :: Fixed Profit       :  "+DoubleToString(fixclprofit,2)+
        "\n      :: Float Money     :  "+DoubleToString(floatprofit,2)+
        "\n      :: Nett Profit        :  "+DoubleToString(floatprofit+fixclprofit,2);
   //--
   Comment(comm);
   ChartRedraw(0);
   return;
//----
  } //-end TradeInfo()
//---------//
```

We also added a function to describe time according to trading time zone conditions as part of the TradeInfo() function.

```
string MCEA::PosTimeZone(void)
  {
//---
    string tzpos="";
    //--
    if(ReqTime(zntm,day)>ThisTime(day))
     {
       tzpos=tz_opn+ " Next day to " +tz_cls + " Next day";
     }
    else
    if(TimeCurrent()<znop)
      {
        if(ThisTime(day)==ReqTime(znop,day) && ThisTime(day)==ReqTime(zncl,day))
          tzpos=tz_opn+" to " +tz_cls+ " Today";
        //else
        if(ThisTime(day)==ReqTime(znop,day) && ThisTime(day)<ReqTime(zncl,day))
          tzpos=tz_opn+ " Today to " +tz_cls+ " Next day";
      }
    else
    if(TimeCurrent()>=znop && TimeCurrent()<zncl)
      {
        if(ThisTime(day)<ReqTime(zncl,day))
          tzpos=tz_opn+ " Today to " +tz_cls+ " Next day";
        else
        if(ThisTime(day)==ReqTime(zncl,day))
          tzpos=tz_opn+" to " +tz_cls+ " Today";
      }
    else
    if(ThisTime(day)==ReqTime(znop,day) && ThisTime(day)<ReqTime(zncl,day))
      {
        tzpos=tz_opn+" Today to " +tz_cls+ " Next day";
      }
    //--
    return(tzpos);
//----
  } //-end PosTimeZone()
//---------//
```

The Multi-Currency Expert Advisor TriangularMA\_MTF\_MCEA interface looks like the following figure.

![TradeInfo](https://c.mql5.com/2/60/TradeInfo.png)

Under the Expert Advisor name TriangularMA\_MTF\_MCEA as you can see there are buttons "M", "C" and "R"

![Expert_manual_button](https://c.mql5.com/2/60/Expert_manual_button.png)

If the M button is clicked, a manual click button panel will be displayed as shown below

![Expert_manual_button_01](https://c.mql5.com/2/60/Expert_manual_button_01.png)

Then the trader can manage orders as explained in the Manual Order Management.

- 1\. Set SL/TP All Orders
- 2\. Close All Orders
- 3\. Close All Profits

If the C button is clicked, a panel button of 30 symbol names or pairs will be displayed and traders can click on one of the pair names or symbol names.

If one of the pair names or symbols is clicked, the chart symbol will immediately be replaced with the symbol whose name was clicked.

![Expert_manual_button_02](https://c.mql5.com/2/60/Expert_manual_button_02.png)

If the R button is clicked, the Multi-Currency Expert Advisor TriangularMA\_MTF\_MCEA will be removed from the chart,

so traders don't need to detach experts manually.

### Strategy Tester

As is known, the MetaTrader5 terminal Strategy Tester supports and allows us to [test strategies](https://www.mql5.com/en/docs/runtime/testing), trade on multiple symbols or test automatic trading for all available symbols and on all available timeframes.

So on this occasion we will test a TriangularMA\_MTF\_MCEA as Multi-Timeframe and Single-Timeframe in Multi-Currency Expert Advisor on the MetaTrader5 Strategy Tester platform.

1. Test TriangularMA\_MTF\_MCEA with Multi-Timeframe mode.

![st-mtf](https://c.mql5.com/2/60/st-mtf.png)

![st-mtf-result](https://c.mql5.com/2/60/st-mtf-result.png)

2\. Test TriangularMA\_MTF\_MCEA with Single-Timeframe mode.

![st-stf](https://c.mql5.com/2/60/st-stf.png)

![st-stf-result](https://c.mql5.com/2/60/st-stf-result.png)

### Conclusion

The conclusion in creating a Multi-Currency Expert Advisor in both Multi-Timeframe mode and Single-Timeframe mode using MQL5 is as follows:

- It turns out that creating a Multi-Currency Expert Advisor in MQL5 is very simple and not much different from a Single-Currency Expert Advisor. But especially for Multi-Currency Expert Advisors with Multi Timeframes, it is a bit more complicated than with single timeframes.
- Creating a Multi-Currency Expert Advisor will increase the efficiency and effectiveness of traders, because traders do not need to open many chart symbols for trading.
- By applying the right trading strategy, the probability of profit will increase when compared to using a Single-Currency Expert Advisor. Because the losses that occur in one pair will be covered by profits in other pairs.
- This TriangularMA\_MTF\_MCEA Multi-Currency Expert Advisor is just an example to learn and develop ideas.
- The test results on the Strategy Tester are still not good. Therefore, if a better strategy with more accurate signal calculations is implemented and adds some better timeframes, I believe the results will be better than the current strategy.
- From the test results on the Strategy Tester of the TriangularMA\_MTF\_MCEA, it turns out that the results from Single-Timeframe are still better than Multi-Timeframe.

Note:

If you have an idea for creating a simple Multi-Currency Expert Advisor based on built-in MQL5 standard indicator signals, please suggest it in the comments.

Hopefully this article and the MQL5 Multi-Currency Expert Advisor program will be useful for traders in learning and developing ideas. Thanks for reading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13770.zip "Download all attachments in the single ZIP archive")

[TriangularMA\_MTF\_MCEA.mq5](https://www.mql5.com/en/articles/download/13770/triangularma_mtf_mcea.mq5 "Download TriangularMA_MTF_MCEA.mq5")(109.8 KB)

[Triangular\_moving\_average.mq5](https://www.mql5.com/en/articles/download/13770/triangular_moving_average.mq5 "Download Triangular_moving_average.mq5")(10.48 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 7): ZigZag with Awesome Oscillator Indicators Signal](https://www.mql5.com/en/articles/14329)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://www.mql5.com/en/articles/14051)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5): Bollinger Bands On Keltner Channel — Indicators Signal](https://www.mql5.com/en/articles/13861)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 3): Added symbols prefixes and/or suffixes and Trading Time Session](https://www.mql5.com/en/articles/13705)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 2): Indicator Signals: Multi Timeframe Parabolic SAR Indicator](https://www.mql5.com/en/articles/13470)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 1): Indicator Signals based on ADX in combination with Parabolic SAR](https://www.mql5.com/en/articles/13008)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/457852)**
(6)


![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
22 Nov 2023 at 15:03

**[@Mohammed Yousif](https://www.mql5.com/en/users/henryyousif9) [#](https://www.mql5.com/en/forum/457852#comment_50680463):** How do I access  the multi currency  expert trader

By reading the article, learning from it and downloading the sample code at the the end.

![Usman Akram](https://c.mql5.com/avatar/avatar_na2.png)

**[Usman Akram](https://www.mql5.com/en/users/usmakr10)**
\|
7 Dec 2023 at 07:37

**Fernando Carreiro [#](https://www.mql5.com/en/forum/457852#comment_50680769):**

By reading the article, learning from it and downloading the sample code at the the end.

Kindly tell me where sample code download


![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
7 Dec 2023 at 11:15

**Usman Akram [#](https://www.mql5.com/en/forum/457852#comment_50994791):**

Kindly tell me where sample code download

As explained by moderator **[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**, look at the very bottom of the article labeled Attached files and click on the file name.

![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
24 Mar 2024 at 04:54

**Usman Akram [#](https://www.mql5.com/ru/forum/464438#comment_52808234):**

Can you please tell me where I can download the sample code?

You can only download it on your computer, if you are reading the article on your phone, the EA file will not be visible.


![Khaled Ali E Msmly](https://c.mql5.com/avatar/2020/12/5FE5FF28-4741.jpg)

**[Khaled Ali E Msmly](https://www.mql5.com/en/users/kamforex9496)**
\|
17 Oct 2024 at 20:28

Thank you for your excellent effort, how long is the back test?


![Developing a Replay System — Market simulation (Part 14): Birth of the SIMULATOR (IV)](https://c.mql5.com/2/55/Desenvolvendo_um_sistema_de_Replay_Parte_14_avatar.png)[Developing a Replay System — Market simulation (Part 14): Birth of the SIMULATOR (IV)](https://www.mql5.com/en/articles/11058)

In this article we will continue the simulator development stage. this time we will see how to effectively create a RANDOM WALK type movement. This type of movement is very intriguing because it forms the basis of everything that happens in the capital market. In addition, we will begin to understand some concepts that are fundamental to those conducting market analysis.

![Design Patterns in software development and MQL5 (Part 2): Structural Patterns](https://c.mql5.com/2/61/Design_Patterns_2Part_2i_Structural_Patterns_Logo.png)[Design Patterns in software development and MQL5 (Part 2): Structural Patterns](https://www.mql5.com/en/articles/13724)

In this article, we will continue our articles about Design Patterns after learning how much this topic is more important for us as developers to develop extendable, reliable applications not only by the MQL5 programming language but others as well. We will learn about another type of Design Patterns which is the structural one to learn how to design systems by using what we have as classes to form larger structures.

![Trade transactions. Request and response structures, description and logging](https://c.mql5.com/2/57/printformat_trading_transactions_avatar.png)[Trade transactions. Request and response structures, description and logging](https://www.mql5.com/en/articles/13052)

The article considers handling trade request structures, namely creating a request, its preliminary verification before sending it to the server, the server's response to a trade request and the structure of trade transactions. We will create simple and convenient functions for sending trading orders to the server and, based on everything discussed, create an EA informing of trade transactions.

![Developing a quality factor for Expert Advisors](https://c.mql5.com/2/55/Desenvolvendo_um_fator_de_qualidade_para_os_EAs_Avatar.png)[Developing a quality factor for Expert Advisors](https://www.mql5.com/en/articles/11373)

In this article, we will see how to develop a quality score that your Expert Advisor can display in the strategy tester. We will look at two well-known calculation methods – Van Tharp and Sunny Harris.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=teoqaurpfnzuasvyhdjvanspdblrggyk&ssn=1769090399568631377&ssn_dr=0&ssn_sr=0&fv_date=1769090399&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13770&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20create%20a%20simple%20Multi-Currency%20Expert%20Advisor%20using%20MQL5%20(Part%204)%3A%20Triangular%20moving%20average%20%E2%80%94%20Indicator%20Signals%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909039939947816&fz_uniq=5048823533850697566&sv=2552)

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