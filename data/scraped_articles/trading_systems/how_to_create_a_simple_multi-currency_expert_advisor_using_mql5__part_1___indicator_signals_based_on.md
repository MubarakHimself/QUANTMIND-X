---
title: How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 1): Indicator Signals based on ADX in combination with Parabolic SAR
url: https://www.mql5.com/en/articles/13008
categories: Trading Systems, Expert Advisors
relevance_score: 13
scraped_at: 2026-01-22T17:11:29.397591
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/13008&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048836049385398174)

MetaTrader 5 / Examples


### Introduction

The definition of a Multi-Currency Expert Advisor in this article is one Expert Advisor or trading robot that can trade (open orders, close orders and manage orders an more) for more than 1 symbol pair from only one symbol chart.

The need for and interest in trading automation systems or trading robots with multi-currency systems is currently very large, but we see that the implementation of multi-currency system programs on MQL5 automated trading robots has not been widely published or may still be kept secret by many programmers.

Therefore, the aim is to fulfill the essential needs of traders who want efficient and effective trading robots, so by relying on the strengths, capabilities and facilities provided by the highly reliable MQL5, we can create a simple Multi-Currency Expert Advisor which in this article uses Indicator Signals: Average Directional Movement in   combination with Parabolic SAR Indicator.

### Plans and Features

**1\. Trading Currency Pairs.**

This Multi-Currency Expert Advisor is planned to trade on a Symbol or Pair as follows:

EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURGBP,EURAUD, EURNZD, EURCAD, EURCHF, EURJPY, GBPAUD, GBPNZD, GBPCAD,GBPCHF,GBPJPY,AUDNZD,AUDCAD,AUDCHF,AUDJPY,NZDCAD,NZDCHF,NZDJPY, CADCHF, CADJPY, CHFJPY = 28 pairs

Plus 2 Metal pairs: XAUUSD (Gold) and XAGUSD (Silver)

Total is 30 pairs.

All of these symbols or pairs are symbols or pairs commonly used by brokers. So, this Multi-Currency Expert Advisor will not work with brokers with symbol or pair names that have prefixes or suffixes.

**2. Signal Indicators.**

The Multi-Currency Expert Advisor will use 2 indicator signals: 1. Average Directional Movement (ADX) indicator with period 7, as the main signal; And 2. Parabolic SAR indicator, as trend signal confirmation.

The two main indicators will use the same Timeframe as specified in the expert property.In addition to detecting the strength or weakness of a trend, the Parabolic SAR indicator is also used with the M15 and M5 Timeframes.

ADX Signals Condition Strategy Formula:  iADX

> UP   = (+DI\[2\] <= -DI\[2\]) && (+DI\[1\] > -DI\[1\]+difference) && (+DI\[0\] > +DI\[1\]) && ((+DI\[0\]/-DI\[0\]) > (+DI\[1\]/-DI\[1\]))
>
> DOWN = (+DI\[2\] >= -DI\[2\]) && (+DI\[1\] < -DI\[1\]-difference) && (+DI\[0\] < +DI\[1\]) && ((+DI\[0\]/-DI\[0\]) < (+DI\[1\]/-DI\[1\]))

Where difference = 0.34 :

> (+DI\[1\] > -DI\[1\]+0.34) = Signal Buy true;
>
> (+DI\[1\] < -DI\[1\]-0.34) = Signal Sell true;

The percentage of PLUSDI\_LINE's Current Bar divided by MINUSDI\_LINE's Current Bar

compared to

The percentage of the Previous Bar PLUSDI\_LINE divided by the Previous Bar MINUSDI\_LINE

> (+DI\[0\]/-DI\[0\]) = V0 = (Current Bar PLUSDI\_LINE / Current Bar MINUSDI\_LINE x 100) - 100;
>
> (+DI\[1\]/-DI\[1\]) = V1 = (Previous Bar PLUSDI\_LINE / Previous Bar MINUSDI\_LINE x 100) - 100;

Then:

> IF V0 > V1 = ADX condition percentage value = Rise
>
> IF V0 < V1 = ADX condition percentage value = Down

Parabolic SAR Signals Condition Strategy: Parabolic Stop And Reverse System (iSAR)

> iSAR UP      = PRICE\_LOW\[0\] > iSAR\[0\]
>
> iSAR DOWN = PRICE\_HIGH\[0\] < iSAR\[0\]

An illustration of the iADX signal combined with iSAR, can be seen in Figure 1.

![Figure 1. Indicators Signal](https://c.mql5.com/2/57/AUDCHFH2_01_art-signal-indicators.png)

**3\. Trade & Order Management**

Trading management on this Multi-Currency Expert Advisor is given several options:

1\. Stop Loss Orders

- Options: Use Order Stop Loss (Yes) or (No)

                If the Use Order Stop Loss (No) option is selected, then all orders will be opened without a stop loss.

                If the option Use Order Stop Loss (Yes):  again given the option: Use Automatic Calculation Stop Loss (Yes) or (No)

                If the option Automatic Calculation Stop Loss (Yes), then the Stop Loss calculation will be performed automatically by the Expert.

                If the option Automatic Calculation Stop Loss (No), then the trader must Input Stop Loss value in Pips.

                If the option Use Order Stop Loss (No):  then the Expert will check for each order opened, whether the signal condition is still good and order

                may be maintained in a profit OR condition  the signal has weakened and the order needs to be closed to save profit

                or signal condition has reversed direction and order must be closed in a loss position.

2\. Take Profit orders

Options: Use Order Take Profit (Yes) or (No)

                If the Use Order Take Profit (No) option is selected, then all orders will be opened without take profit.

                If the option Use Order Take Profit (Yes):  again given the option: Use Automatic Calculation Order Take Profit (Yes) or (No)

                If the option Automatic Calculation Order Take Profit (Yes), then the calculation of the Take Profit Order will be carried out automatically by the Expert.

                If the option Automatic Calculation Order Take Profit (No), then the trader must Input Order Take Profit value in Pips.

3\. Trailing Stop and Trailing Take Profit

Options: Use Trailing SL/TP (Yes) or (No)

                If the Use Trailing SL/TP option is (No), then the Expert will not do trailing stop loss and trailing take profit.

                If the option Use Trailing SL/TP (Yes): again given the option: Use Automatic Trailing (Yes) or (No)

                If the option Use Automatic Trailing (Yes), then the trailing stop will be executed by the Expert using the Parabolic SAR value.

                If the option Use Automatic Trailing (No), then the trailing stop will be performed by the Expert using the value in the input property.

                Note: The Expert will carry out a trailing take profit simultaneously with a trailing stop.

**4\. Manual Order Management.**

To support efficiency in this Multi-Currency Expert Advisor, several manual click buttons will be added

1\. Set SL / TP All Orders

       When the trader input parameter sets Use Order Stop Loss (No) and/or Use Order Take Profit (No)

       but then the trader intends to use stop loss or take profit on all orders, then with just single click of the button "Set SL / TP All Orders" all orders will be

       modified and a stop loss will be applied and/or take profits.

2\. Close All Orders

       If a trader wants to close all orders, then with just single click of the button "Close All Orders" all open orders will be closed.

3\. Close All Orders Profit

       If a trader wants to close all orders that are already profitable, then with only single click of the button "Close All Orders Profit" then

       all open orders that are already profitable will be closed.

**5\. Management Orders and Symbols Chart.**

For Multi-Currency Expert Advisors who will trade 30 pairs from only one chart symbol, it will be very effective and efficient if a button panel is provided for all symbols, so traders can change charts or symbols with just one click.

### Implementation of planning in the MQL5 program

**1. Program header and expert input properties.**

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

Enumeration YN is used for options (Yes) or (No) in expert input property

```
enum YN
  {
   No,
   Yes
  };
//--
```

Enumeration to use Money Management Lot size

```
enum mmt
  {
   FixedLot,     // Fixed Lot Size
   DynamLot    // Dynamic Lot Size
  };
//--
```

Enumeration Timeframe for expert signal calculation

```
enum TFX
  {
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

Expert input properties

```
//---
input group               "=== Global Strategy EA Parameter ==="; // Global Strategy EA Parameter
input TFX             TimeFrames = TFH4;             // Select Expert TimeFrame, default PERIOD_H4
input int              ADXPeriod = 7;                // Input ADX Period
input group               "=== Money Management Lot Size Parameter ==="; // Money Management Lot Size Parameter
input mmt                  mmlot = DynamLot;         // Money Management Type
input double                Risk = 10.0;             // Percent Equity Risk per Trade (Min=1.0% / Max=10.0%)
input double                Lots = 0.01;             // Input Manual Lot Size FixedLot
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
input YN                  use_tp = No;               // Use Order Take Profit (Yes) or (No)
input YN                  autotp = Yes;              // Use Automatic Calculation Take Profit (Yes) or (No)
input double               TPval = 50;               // If Not Use Automatic TP - Input TP value in Pips
input YN            TrailingSLTP = Yes;              // Use Trailing SL/TP (Yes) or (No)
input YN                 autotrl = No;               // Use Automatic Trailing (Yes) or (No)
input double               TSval = 5;                // If Not Use Automatic Trailing Input Trailing value in Pips
input double               TSmin = 5;                // Minimum Pips to start Trailing Stop
input YN           Close_by_Opps = Yes;              // Close Trade By Opposite Signal (Yes) or (No)
input YN               SaveOnRev = Yes;              // Close Trade and Save profit due to weak signal (Yes) or (No)
//--Others Expert Advisor Parameter
input group               "=== Others Expert Advisor Parameter ==="; // Others EA Parameter
input YN                  alerts = Yes;              // Display Alerts / Messages (Yes) or (No)
input YN           UseEmailAlert = No;               // Email Alert (Yes) or (No)
input YN           UseSendnotify = No;               // Send Notification (Yes) or (No)
input YN      trade_info_display = Yes;              // Select Display Trading Info on Chart (Yes) or (No)
input ulong               magicEA = 202307;          // Expert ID (Magic Number)
//---
```

To declare all variables, objects and functions needed in this Multi-Currency Expert Advisor, we will create a Class to specify the construction and configuration in the expert advisor workflow.

```
//+------------------------------------------------------------------+
//| Class for working Expert Advisor                                 |
//+------------------------------------------------------------------+
class MCEA
  {
   //---
private:
   //----
   int               x_year;       // Year
   int               x_mon;        // Month
   int               x_day;        // Day of the month
   int               x_hour;       // Hour in a day
   int               x_min;        // Minutes
   int               x_sec;        // Seconds
   //--
   int               oBm,
                     oSm,
                     ldig;
   int               posCur1,
                     posCur2;
   //--
   double            LotPS,
                     difDi;
   double            slv,
                     tpv,
                     pip,
                     xpip;
   double            floatprofit,
                     fixclprofit;
   double            ADXDIp[];
   double            ADXDIm[];
   //--
   string            pairs,
                     hariini,
                     daytrade,
                     trade_mode;
   //--
   double            OPEN[],
                     HIGH[],
                     LOW[],
                     CLOSE[];
   datetime          TIME[];
   datetime          closetime;
   //--
   //------------

   //------------
   int               iADXCross(const string symbol);
   int               iADXpct(const string symbol,const int index);
   int               PARSAR05(const string symbol);
   int               PARSAR15(const string symbol);
   int               PARSAROp(const string symbol);
   int               LotDig(const string symbol);
   //--
   double            MLots(const string symbx);
   double            NonZeroDiv(double val1,double val2);
   double            OrderSLSet(const string xsymb,ENUM_ORDER_TYPE type,double atprice);
   double            OrderTPSet(const string xsymb,ENUM_ORDER_TYPE type,double atprice);
   double            SetOrderSL(const string xsymb,ENUM_POSITION_TYPE type,double atprice);
   double            SetOrderTP(const string xsymb,ENUM_POSITION_TYPE type,double atprice);
   double            TSPrice(const string xsymb,ENUM_POSITION_TYPE ptype,int TS_type);
   //--
   string            ReqDate(int d,int h,int m);
   string            TF2Str(ENUM_TIMEFRAMES period);
   string            timehr(int hr,int mn);
   string            TradingDay(void);
   string            AccountMode();
   string            GetCommentForOrder(void)             { return(expname); }
   //------------

public:
   //---

   //-- ADXPSAR_MCEA Config --
   string            DIRI[],
                     AS30[];
   string            expname;
   int               handADX[];
   int               hParOp[],
                     hPar15[],
                     hPar05[];
   int               ALO,
                     dgts,
                     arrsymbx;
   int               sall,
                     arper;
   ulong             slip;
   ENUM_TIMEFRAMES   TFt,
                     TFT15,
                     TFT05;
   //--
   double            SARstep,
                     SARmaxi;
   double            profitb[],
                     profits[];
   //--
   int               Buy,
                     Sell;
   int               ccur,
                     psec,
                     xtto,
                     checktml;
   int               OpOr[],xob[],xos[];
   //--
   int               year,  // Year
                     mon,   // Month
                     day,   // Day
                     hour,  // Hour
                     min,   // Minutes
                     sec,   // Seconds
                     dow,   // Day of week (0-Sunday, 1-Monday, ... ,6-Saturday)
                     doy;   // Day number of the year (January 1st is assigned the number value of zero)
   //------------
                     MCEA(void);
                    ~MCEA(void);
   //------------
   //--
   virtual void      ADXPSAR_MCEA_Config(void);
   virtual void      ExpertActionTrade(void);
   //--
   void              ArraySymbolResize(void);
   void              CurrentSymbolSet(const string symbol);
   void              Pips(const string symbol);
   void              TradeInfo(void);
   void              Do_Alerts(const string symbx,string msgText);
   void              CheckOpenPMx(const string symbx);
   void              SetSLTPOrders(void);
   void              CloseBuyPositions(const string symbol);
   void              CloseSellPositions(const string symbol);
   void              CloseAllOrders(void);
   void              CheckClose(const string symbx);
   void              TodayOrders(void);
   void              UpdatePrice(const string symbol,ENUM_TIMEFRAMES xtf);
   void              RefreshPrice(const string symbx,ENUM_TIMEFRAMES xtf,int bars);
   //--
   bool              RefreshTick(const string symbx);
   bool              TradingToday(void);
   bool              OpenBuy(const string symbol);
   bool              OpenSell(const string symbol);
   bool              ModifyOrderSLTP(double mStop,double ordtp);
   bool              ModifySLTP(const string symbx,int TS_type);
   bool              CloseAllProfit(void);
   bool              ManualCloseAllProfit(void);
   //--
   int               PairsIdxArray(const string symbol);
   int               GetOpenPosition(const string symbol);
   int               DirectionMove(const string symbol);
   int               GetCloseInWeakSignal(const string symbol,int exis);
   int               CheckToCloseInWeakSignal(const string symbol,int exis);
   int               ThisTime(const int reqmode);
   //--
   string            getUninitReasonText(int reasonCode);
   //--
   //------------
   //---
  }; //-end class MCEA
//---------//
```

The very first and foremost function in the Multi-Currency Expert Advisor work process that is called from OnInit() is ADXPSAR\_MCEA\_Config().

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//---
   mc.ADXPSAR_MCEA_Config();
//--
   return(INIT_SUCCEEDED);
//---
  } //-end OnInit()
//---------//
```

In the ADXPSAR\_MCEA\_Config() function, all symbols to be used are configured, all handle indicators used and some important functions of the include file header for the expert advisor workflow.

```
//+------------------------------------------------------------------+
//| Expert Configuration                                             |
//+------------------------------------------------------------------+
void MCEA::ADXPSAR_MCEA_Config(void)
  {
//---
//-- Here we will register all the symbols or pairs that will be used on the Multi-Currency Expert Advisor
//--
   string All30[]= {"EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY","EURGBP",
                    "EURAUD","EURNZD","EURCAD","EURCHF","EURJPY","GBPAUD","GBPNZD","GBPCAD",
                    "GBPCHF","GBPJPY","AUDNZD","AUDCAD","AUDCHF","AUDJPY","NZDCAD","NZDCHF",
                    "NZDJPY","CADCHF","CADJPY","CHFJPY","XAUUSD","XAGUSD"
                   }; // 30 pairs
//--
   sall=ArraySize(All30);
   ArrayResize(AS30,sall,sall);
//-- These AS30[] arrays will be used in the symbol list panel and for the buttons to change symbols and charts
   ArrayCopy(AS30,All30,0,0,WHOLE_ARRAY);
//--
   arrsymbx=sall;
   ArraySymbolResize();
   ArrayCopy(DIRI,All30,0,0,WHOLE_ARRAY); //-- The "DIRI[]" array containing the symbol or pair name will be used
//-- in all trading activities of the multi-currency expert
//--
//-- This function is for Select all symbol in the Market Watch window
   for(int x=0; x<arrsymbx; x++)
     {
      SymbolSelect(DIRI[x],true);
     }
   pairs="Multi Currency 30 Pairs";
//---
//-- Here we will provide a Period Timeframe value which will be used for signal calculations according
//-- to the Timeframe option on the expert input property.
   ENUM_TIMEFRAMES TFs[]= {PERIOD_H1,PERIOD_H2,PERIOD_H3,PERIOD_H4,PERIOD_H6,PERIOD_H8,PERIOD_H12,PERIOD_D1};
   int arTFs=ArraySize(TFs);
   for(int x=0; x<arTFs; x++)
     {
      if(x==TimeFrames)
        {
         TFt=TFs[x];
         break;
        }
     }
//--
//-- Indicators handle for all symbol
   for(int x=0; x<arrsymbx; x++)
     {
      handADX[x]=iADX(DIRI[x],TFt,ADXPeriod);         //-- Handle for the iADX indicator according to the selected Timeframe
      hParOp[x]=iSAR(DIRI[x],TFt,SARstep,SARmaxi);    //-- Handle for the iSAR indicator according to the selected Timeframe
      hPar15[x]=iSAR(DIRI[x],TFT15,SARstep,SARmaxi);  //-- Handle for the iSAR indicator for M15 Timeframe
      hPar05[x]=iSAR(DIRI[x],TFT05,SARstep,SARmaxi);  //-- Handle for the iSAR indicator for M5 Timeframe
     }
//--
//-- Since this expert advisor is a multi-currency expert, we must check the maximum number
//-- of account limit orders allowed by the broker.
//-- This needs to be checked, so that when the expert opens an order there will be
//-- no return codes of the trade server error 10040 = TRADE_RETCODE_LIMIT_POSITIONS
   ALO=(int)mc_account.LimitOrders()>arrsymbx ? arrsymbx : (int)mc_account.LimitOrders();
//--
//-- The LotPS variable will later be used for the proportional distribution of Lot sizes for each symbol
   LotPS=(double)ALO;
//--
   mc_trade.SetExpertMagicNumber(magicEA); //-- Set Magic Number as expert ID
   mc_trade.SetDeviationInPoints(slip);    //-- Set expert deviation with slip variable value
   mc_trade.SetMarginMode();               //-- Set the Margin Mode expert to the value of Account Margin Mode
//--
   return;
//---
  } //-end ADXPSAR_MCEA_Config()
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

The ExpertActionTrade() function will carry out all activities and manage automatic trading, starting from Open Orders, Close Orders, Trailing Stop or Trailing Profits and other additional activities.

The sequence of the work process is as follows, as I explained on the sidelines of the program.

```
void MCEA::ExpertActionTrade(void)
  {
//---
    //Check Trading Terminal
    ResetLastError();
    //--
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED) && mc.checktml==0) //-- Check whether MT5 Algorithmic trading is Allow or Prohibit
      {
        mc.Do_Alerts(Symbol(),"Trading Expert at "+Symbol()+" are NOT Allowed by Setting.");
        mc.checktml=1;  //-- Variable checktml is given a value of 1, so that the alert is only done once.
        return;
      }
    //--
    if(!DisplayManualButton("M","C","R"))
      DisplayManualButton(); //-- Show the expert manual button panel
    //--
    //-- The functions below will be displayed on the expert chart according to
    //-- the Select Display Trading Info on Chart (Yes) or (No) option on expert property.
    //--
    if(trade_info_display==Yes)
       mc.TradeInfo(); //-- Displayed Trading Info on Chart
//--
    //--
    if(trade_info_display==Yes) mc.TradeInfo(); //-- Displayed Trading Info on Chart
    //---
    //-- Because the current prices of a specified symbol (SymbolInfoTick) will occur differently
    //-- for each symbol, we reduce the tick update frequency to only every 5 seconds.
    //-- So, looping to check the signal for all trading activity of all symbols will only be done every 5 seconds.
    //--
    int mcsec=mc.ThisTime(mc.sec);
    //-- With the function ThisTime(mc.sec), we will retrieve the current seconds value to mcsec variable.
    //--
    //-- MathMod is a formula that gives the (modulus) real remainder after the division of two numbers.
    //-- By dividing the value of seconds with the value of 5.0, if the result is 0, it means that 5 seconds
    //-- have been reached from the previous psec variable seconds value.
    //--
    if(fmod((double)mcsec,5.0)==0)
       mc.ccur=mcsec;
    //--
    if(mc.ccur!=mc.psec) //-- So, if the seconds value in the ccur variable is not the same as the psec variable value
        {
         //-- (then the psec variable value already 5 seconds before)
         string symbol;
         //-- Here we start with the rotation of the name of all symbol or pairs to be traded
         //-- This is the basic framework for the automated trading workflow of this Multi-Currency Expert Advisor
         //-- Here we start with the rotation of the name of all symbol or pairs to be traded
         //-- This is the basic framework for the automated trading workflow of this Multi-Currency Expert Advisor
         for(int x=0; x<mc.arrsymbx && !IsStopped(); x++)
           {
             //--
             if(mc.DIRI[x]==Symbol())
                symbol=Symbol();
             else
                symbol=mc.DIRI[x];
             //-- After the symbol or pair name is set, we declare or notify the symbol to MarketWatch
             //-- and the trade server by calling the CurrentSymbolSet(symbol) function.
             mc.CurrentSymbolSet(symbol);
             //--
             if(mc.TradingToday()) //-- The TradingToday() function checks whether today is allowed for trading
               {                   //-- If today is not allowed for trading, then the Expert will only perform management
                                   //-- orders such as trailing stops or trailing profits and closing orders.
                 //-- according to the expert input property Day Trading On/Off group
                 //-- If TradingToday() == Yes, then the next process is to call the function ThisTime(mc.sec)
                 //--
                 mc.OpOr[x]=mc.GetOpenPosition(symbol); //-- Get trading signals to open positions
                 //--                                   //-- and store in the variable OpOr[x]
                 if(mc.OpOr[x]==mc.Buy) //-- If variable OpOr[x] get result of GetOpenPosition(symbol) as "Buy" (value=1)
                   {
                     //--
                     mc.CheckOpenPMx(symbol);
                     //--
                     //-- If it turns out that the "Sell Order" has been opened,
                     //-- and Close Trade By Opposite Signal according to the input property is (Yes),
                     //-- then call the function CloseSellPositions(symbol) to close the sell order on that symbol.
                     //--
                     if(Close_by_Opps==Yes && mc.xos[x]>0) mc.CloseSellPositions(symbol);
                     //--
                     if(mc.xob[x]==0 && mc.xtto<mc.ALO) mc.OpenBuy(symbol); //-- Open BUY order for this symbol
                     else
                     //-- OR
                     //-- If Close Trade and Save profit due to weak signal according to the input property is (Yes)
                     //-- then call the CloseAllProfit() function to close all orders
                     //-- who are already in profit.
                     //--
                     if(mc.xtto>=mc.ALO)
                       {
                         //-- If the total number of orders is greater than or equal
                         //-- to the account limit orders allowed by the broker, then turn on alerts
                         //--
                         mc.Do_Alerts(symbol,"Maximum amount of open positions and active pending orders has reached"+
                                      "\n the limit = "+string(mc.ALO)+" Orders ");
                         //--
                         mc.CheckOpenPMx(symbol); //-- Call the CheckOpenPMx(symbol) function
                         //--
                         //-- If it turns out that the "Sell Order" has been opened,
                         //-- and the condition of the Sell order has lost more than 1.02 USD,
                         //-- and the "Buy Order" has not been opened, then call CloseSellPositions(symbol)
                         //-- to close "Sell order" and open "Buy order".
                         if(mc.xos[x]>0 && mc.profits[x]<-1.02 && mc.xob[x]==0)
                           {
                            mc.CloseSellPositions(symbol);
                            mc.OpenBuy(symbol);
                           }
                         else
                         //-- OR
                         //-- If Close Trade and Save profit due to weak signal according to the input property is (Yes)
                         //-- then call the CloseAllProfit() function to close all orders
                         //-- who are already in profit.
                         //--
                         if(SaveOnRev==Yes)
                            mc.CloseAllProfit();
                       }
                   }
                 if(mc.OpOr[x]==mc.Sell) //-- If variable OpOr[x] get result of GetOpenPosition(symbol) as "Sell" (value=-1)
                   {
                     //--
                     //-- Call the CheckOpenPMx(symbol) function to check whether there are
                     //-- already open "Buy" or "Sell" orders or no open orders.
                     //--
                     mc.CheckOpenPMx(symbol);
                     //--
                     //-- If it turns out that the "Buy Order" has been opened,
                     //-- and Close Trade By Opposite Signal according to the input property is (Yes),
                     //-- then call the function CloseBuyPositions(symbol) to close the buy order on that symbol.
                     //--
                     if(Close_by_Opps==Yes && mc.xob[x]>0)
                        mc.CloseBuyPositions(symbol);
                     //--
                     //-- The algorithm below means that the expert will only open 1 order per symbol,
                     //-- provided that the total number of orders is still less than
                     //-- the account limit orders allowed by the broker.
                     //--
                     if(mc.xos[x]==0 && mc.xtto<mc.ALO) mc.OpenSell(symbol);  //-- Open SELL order for this symbol
                     else
                     if(mc.xtto>=mc.ALO)
                       {
                         //-- If the total number of orders is greater than or equal
                         //-- to the account limit orders allowed by the broker, then turn on alerts
                         //--
                         mc.Do_Alerts(symbol,"Maximum amount of open positions and active pending orders has reached"+
                                      "\n the limit = "+string(mc.ALO)+" Orders ");
                         //--
                         mc.CheckOpenPMx(symbol); //-- Call the CheckOpenPMx(symbol) function
                         //--
                         //-- If it turns out that the "Buy Order" has been opened,
                         //-- and the condition of the Buy order has lost more than 1.02 USD,
                         //-- and the "Sell Order" has not been opened, then call CloseBuyPositions(symbol)
                         //-- to close "Buy order" and open "Sell order".
                         if(mc.xob[x]>0 && mc.profitb[x]<-1.02 && mc.xos[x]==0)
                           {
                             mc.CloseBuyPositions(symbol);
                             mc.OpenSell(symbol);
                           }
                         else
                         //-- OR
                         //-- If Close Trade and Save profit due to weak signal according to the input property is (Yes)
                         //-- then call the CloseAllProfit() function to close all orders
                         //-- who are already in profit.
                         //--
                         if(SaveOnRev==Yes)
                            mc.CloseAllProfit();
                       }
                   }
                 //--
                 mc.CheckOpenPMx(symbol);
                 //-- The algorithm block below will check whether there is a weakening of the signal on Buy or Sell positions.
                 //-- If it is true that there is a weakening signal and the iSAR indicator has reversed direction,
                 //-- then close the losing order and immediately opened an order in the opposite direction.
                 //--
                 if(mc.xob[x]>0 && mc.CheckToCloseInWeakSignal(symbol,mc.Buy)==mc.Sell) {mc.CloseBuyPositions(symbol); mc.OpenSell(symbol);}
                 if(mc.xos[x]>0 && mc.CheckToCloseInWeakSignal(symbol,mc.Sell)==mc.Buy) {mc.CloseSellPositions(symbol); mc.OpenBuy(symbol);}
               }
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
                     //--
                     if(mc.xob[x]>0 && mc.CheckToCloseInWeakSignal(symbol,mc.Buy)==mc.Sell) mc.CloseBuyPositions(symbol);
                     if(mc.xos[x]>0 && mc.CheckToCloseInWeakSignal(symbol,mc.Sell)==mc.Buy) mc.CloseSellPositions(symbol);
                   }
                 //--
                 if(TrailingSLTP==Yes) //-- Use Trailing SL/TP (Yes)
                   {
                     if(autotrl==Yes) mc.ModifySLTP(symbol,1); //-- If Use Automatic Trailing (Yes)
                     if(autotrl==No)  mc.ModifySLTP(symbol,0); //-- Use Automatic Trailing (No)
                   }
               }
             //--
             //-- Check if there are orders that were closed in the last 6 seconds.
             //-- If there are give alerts.
             mc.CheckClose(symbol);
           }
        //-- replace the value of the psec variable with the value of the ccur variable.
        mc.psec=mc.ccur;
      }
    //--
    return;
//---
  } //-end ExpertActionTrade()
//---------//
```

**3\. How to get trading signals for open or close position?**

To get the indicator signal, we have to call the function GetOpenPosition(symbol) to get a trading signal for open position

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int MCEA::GetOpenPosition(const string symbol) // Signal Open Position
  {
//---
   int ret=0;
   int rise=1,
       down=-1;
//--
   int dirmov=DirectionMove(symbol);
   int pars15=PARSAR15(symbol);
   int parsOp=PARSAROp(symbol);
   int sigADX=iADXCross(symbol);
//--
   if(sigADX==rise && parsOp==rise && dirmov==rise && pars15==rise)
      ret=rise;
   if(sigADX==down && parsOp==down && dirmov==down && pars15==down)
      ret=down;
//--
   return(ret);
//---
  } //-end GetOpenPosition()
//---------//
```

GetOpenPosition() function will call 4 signal functions and store in the variable OpOr\[\].

> 1\. DirectionMove(symbol); //-- Function to check whether the price  on the candlestick bar in the expert period
>
> 2\. PARSAR15(symbol);        //-- Function to check whether the condition of the iSAR indicator rises or falls in Period M15
>
> 3\. PARSAROp(symbol);       //-- Function to check whether the condition of the iSAR indicator rises or falls in the expert period
>
> 4\. iADXCross(symbol);       //-- Function to check whether the condition of the iADX indicator rises or falls in the expert period

Then in the iADXCross(symbol) function, the iADXpct() function will be called to check the percentage  movement between +DI and -DI, as described in the Signal Indicators section.

To get the condition of the indicator, in the 4 functions PARSAR15(symbol), PARSAROP(symbol), iADXCross(symbol) and iADXpct() we have to get the index number of each required indicator handle.

To get the indicator handle index number, we call the function PairsIdxArray(symbol) by means of

```
int x=PairsIdxArray(symbol);
```

The x value is the arrays of indicator handle index number of the symbol in question.

In the example of the PARSAR15() function we can see how to call the iSAR indicator handle for the symbol in question.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int MCEA::PARSAR15(const string symbol) // formula Parabolic SAR M15
  {
//---
   int ret=0;
   int rise=1,
       down=-1;
   int br=2;
//--
   double PSAR[];
   ArrayResize(PSAR,br,br);
   ArraySetAsSeries(PSAR,true);
   int xx=PairsIdxArray(symbol);
   CopyBuffer(hPar15[xx],0,0,br,PSAR);
//--
   RefreshPrice(symbol,TFT15,br);
   double HIG0=iHigh(symbol,TFT15,0);
   double LOW0=iLow(symbol,TFT15,0);
//--
   if(PSAR[0]<LOW0)
      ret=rise;
   if(PSAR[0]>HIG0)
      ret=down;
//--
   return(ret);
//---
  } //-end PARSAR15()
//---------//
```

**4\. ChartEvent Function**

To support effectiveness and efficiency in the use of Multi-Currency Expert Advisors, it is deemed necessary to create several manual buttons in managing orders and changing charts or symbols.

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
      //--- if "Set SL/TP All Orders" button is click
      if(sparam=="Set SL/TP All Orders")
        {
         mc.SetSLTPOrders();
         Alert("-- "+mc.expname+" -- ",Symbol()," -- Set SL/TP All Orders");
         //--- unpress the button
         ObjectSetInteger(0,"Set SL/TP All Orders",OBJPROP_STATE,false); //-- Button state (depressed button)
         ObjectSetInteger(0,"Set SL/TP All Orders",OBJPROP_ZORDER,0);    //-- Priority of a graphical object for receiving events
         CreateManualPanel();
        }
      //--- if "Close All Order" button is click
      if(sparam=="Close All Order")
        {
         mc.CloseAllOrders();
         Alert("-- "+mc.expname+" -- ",Symbol()," -- Close All Orders");
         //--- unpress the button
         ObjectSetInteger(0,"Close All Order",OBJPROP_STATE,false); //-- Button state (depressed button)
         ObjectSetInteger(0,"Close All Order",OBJPROP_ZORDER,0);    //-- Priority of a graphical object for receiving events
         CreateManualPanel();
        }
      //--- if "Close All Profit" button is click
      if(sparam=="Close All Profit")
        {
         mc.ManualCloseAllProfit();
         Alert("-- "+mc.expname+" -- ",Symbol()," -- Close All Profit");
         //--- unpress the button
         ObjectSetInteger(0,"Close All Profit",OBJPROP_STATE,false); //-- Button state (depressed button)
         ObjectSetInteger(0,"Close All Profit",OBJPROP_ZORDER,0);    //-- Priority of a graphical object for receiving events
         CreateManualPanel();
        }
      //--- if "X" button is click
      if(sparam=="X")
        {
         ObjectsDeleteAll(0,0,OBJ_BUTTON);
         ObjectsDeleteAll(0,0,OBJ_LABEL);
         ObjectsDeleteAll(0,0,OBJ_RECTANGLE_LABEL);
         //--- unpress the button
         ObjectSetInteger(0,"X",OBJPROP_STATE,false); //-- Button state (depressed button)
         ObjectSetInteger(0,"X",OBJPROP_ZORDER,0);    //-- Priority of a graphical object for receiving events
         //--
         DeleteButtonX();
         mc.PanelExtra=false;
         DisplayManualButton();
        }
      //--- if "M" button is click
      if(sparam=="M")
        {
         //--- unpress the button
         ObjectSetInteger(0,"M",OBJPROP_STATE,false); //-- Button state (depressed button)
         ObjectSetInteger(0,"M",OBJPROP_ZORDER,0);    //-- Priority of a graphical object for receiving events
         mc.PanelExtra=true;
         CreateManualPanel();
        }
      //--- if "C" button is click
      if(sparam=="C")
        {
         //--- unpress the button
         ObjectSetInteger(0,"C",OBJPROP_STATE,false); //-- Button state (depressed button)
         ObjectSetInteger(0,"C",OBJPROP_ZORDER,0);    //-- Priority of a graphical object for receiving events
         mc.PanelExtra=true;
         CreateSymbolPanel();
        }
      //--- if "R" button is click
      if(sparam=="R")
        {
         Alert("-- "+mc.expname+" -- ",Symbol()," -- expert advisor will be Remove from the chart.");
         ExpertRemove();
         //--- unpress the button
         ObjectSetInteger(0,"R",OBJPROP_STATE,false); //-- Button state (depressed button)
         ObjectSetInteger(0,"R",OBJPROP_ZORDER,0);    //-- Priority of a graphical object for receiving events
         if(!ChartSetSymbolPeriod(0,Symbol(),Period()))
            ChartSetSymbolPeriod(0,Symbol(),Period());
         DeletePanelButton();
         ChartRedraw(0);
        }
      //--- if Symbol name button is click
      if(lensparam==lensymbol)
        {
         int sx=mc.PairsIdxArray(sparam);
         ChangeChartSymbol(mc.AS30[sx],CCS);
        }
      //--
     }
//--
   return;
//---
  } //-end OnChartEvent()
//---------//
```

The Multi-Currency Expert Advisor interface looks like the following figure.

![ADXPSAR_MCEA](https://c.mql5.com/2/57/ADXPSAR_MCEA__1.png)

Expert Button Manual

![Expert Manual Button](https://c.mql5.com/2/57/Expert_manual_button.png)

If the M button is clicked, a manual click button panel will be displayed as shown below

![Expert_manual_button_01](https://c.mql5.com/2/57/Expert_manual_button_01.png)

Then the trader can manage orders:

> 1\. Set SL/TP All Orders
>
> 2\. Close All Orders
>
> 3\. Close All Profits

If the C button is clicked, a panel button of 30 symbol names or pairs will be displayed as shown below

![Expert_manual_button_02](https://c.mql5.com/2/57/Expert_manual_button_02.png)

If one of the pair names or symbols is clicked, the chart symbol will immediately be replaced with the symbol whose name was clicked.

If the R button is clicked, the Multi-Currency Expert Advisor ADXPSAR\_MCEA will be removed from the chart.

### Strategy Tester

As it is known that the Strategy Tester of the MetaTrader 5 terminal already supports and allows us to perform a testing of strategies, trading on multiple symbols or testing automated trading for all of the available symbols.

Testing Trading Strategies [Multi-Currency Testing](https://www.mql5.com/en/docs/runtime/testing#multicurrency)

So on this occasion we will test the ADXPSAR\_MCEA Multi-Currency Expert Advisor on the Strategy Tester MetaTrader 5 platforms.

![Strategy Tester Result](https://c.mql5.com/2/57/Strategy_Tester_Result.png)

![Balance Equity](https://c.mql5.com/2/57/Balance_Equity.png)

![Entries Profit](https://c.mql5.com/2/57/Entries_Profit.png)

![Correlation](https://c.mql5.com/2/57/Correlation.png)

![Minimal Position](https://c.mql5.com/2/57/Minimal_Position.png)

### Conclusion

As I wrote above, then I came to the conclusion that creating a Multi-Currency Expert Advisor using MQL5:

- It turns out that creating a Multi-Currency Expert Advisor in MQL5 is very simple and not much different from a Single-Currency Expert Advisor.
- Creating a Multi-Currency Expert Advisor will increase the efficiency and effectiveness of traders, because traders do not need to open many chart symbols for trading.
- By applying the right trading strategy and calculating better indicator signals, the probability of profit will increase when compared to using a Single-Currency Expert Advisor. Because the losses that occur in one pair will be covered by profits in other pairs.
- This ADXPSAR\_MCEA Multi-Currency Expert Advisor is just an example to learn and develop ideas. And the test results on the Strategy Tester are still not good. Therefore, if a better strategy with more accurate signal calculation is implemented, I believe the result will be better than the current one.

Hopefully this article and the MQL5 Multi-Currency Expert Advisor program will be useful for traders in learning and developing ideas.

Thanks for reading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13008.zip "Download all attachments in the single ZIP archive")

[ADXPSAR\_MCEA.mq5](https://www.mql5.com/en/articles/download/13008/adxpsar_mcea.mq5 "Download ADXPSAR_MCEA.mq5")(84.17 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 7): ZigZag with Awesome Oscillator Indicators Signal](https://www.mql5.com/en/articles/14329)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://www.mql5.com/en/articles/14051)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5): Bollinger Bands On Keltner Channel — Indicators Signal](https://www.mql5.com/en/articles/13861)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://www.mql5.com/en/articles/13770)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 3): Added symbols prefixes and/or suffixes and Trading Time Session](https://www.mql5.com/en/articles/13705)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 2): Indicator Signals: Multi Timeframe Parabolic SAR Indicator](https://www.mql5.com/en/articles/13470)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/453259)**
(30)


![Alexey Viktorov](https://c.mql5.com/avatar/2017/4/58E3DFDD-D3B2.jpg)

**[Alexey Viktorov](https://www.mql5.com/en/users/alexeyvik)**
\|
4 Dec 2023 at 17:37

**Sergey Pavlov [#](https://www.mql5.com/ru/forum/458463#comment_50934075):**

... curious, but it is not clear on which part of the history the testing was done, because a hundred trades on 30 currency pairs is confusing.

This was not written to sell to suckers. There are no fish here...

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
4 Dec 2023 at 18:37

Maybe I'm missing something, but it's not [a multicurrency trading system](https://www.mql5.com/en/articles/234 "Article: The Implementation of a Multi-currency Mode in MetaTrader 5 "). It doesn't correlate signals from different instruments with each other.

It's a rather complicated way to combine autonomous trading on different symbols into one programme. It's like running a much simpler owl on several tabs.

![Juan Luis De Frutos Blanco](https://c.mql5.com/avatar/2023/2/63df76f5-9ce7.jpg)

**[Juan Luis De Frutos Blanco](https://www.mql5.com/en/users/febrero59)**
\|
5 Feb 2024 at 08:48

Hello Roberto,

This is the first of your articles I read and I have the following curiosity in:

```
    ...
    difDi(0.34),
    ...
    bool gadxrise=(ADXDIp[2]<=ADXDIm[2] && ADXDIp[1]>ADXDIm[1]+difDi && ADXDIp[0]>ADXDIp[1] && GiADXc==rise);
    bool gadxdown=(ADXDIp[2]>=ADXDIm[2] && ADXDIp[1]<ADXDIm[1]-difDi && ADXDIp[0]<ADXDIp[1] && GiADXc==down);
```

about the 0.34 value: what does it mean, what is its purpose, what does it intend to compensate, how did you arrive at it?

Thanks in advance,

Juan Luis

![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
5 Feb 2024 at 09:40

**Juan Luis De Frutos Blanco [#](https://www.mql5.com/en/forum/453259/page3#comment_52168811):**

about the 0.34 value: what does it mean, what is its purpose, what does it intend to compensate, how did you arrive at it?

There must be a difference between PLUSDI\_LINE and MINUSDI\_LINE to get the value that there is a significant movement between PLUSDI\_LINE and MINUSDI\_LINE.

The value is 0.34 based only on observations and estimates.

If you want to replace it, it's up to you.

![Juan Luis De Frutos Blanco](https://c.mql5.com/avatar/2023/2/63df76f5-9ce7.jpg)

**[Juan Luis De Frutos Blanco](https://www.mql5.com/en/users/febrero59)**
\|
5 Feb 2024 at 10:21

**Roberto Jacobs [#](https://www.mql5.com/en/forum/453259/page3#comment_52169120):**

There must be a difference between PLUSDI\_LINE and MINUSDI\_LINE to get the value that there is a significant movement between PLUSDI\_LINE and MINUSDI\_LINE.

The value is 0.34 based only on observations and estimates.

If you want to replace it, it's up to you.

Thanks. 👍

![Data label for time series  mining(Part 1)：Make a dataset with trend markers through the EA operation chart](https://c.mql5.com/2/57/data-label-for-time-series-mining-avatar.png)[Data label for time series mining(Part 1)：Make a dataset with trend markers through the EA operation chart](https://www.mql5.com/en/articles/13225)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

![DoEasy. Controls (Part 32): Horizontal ScrollBar, mouse wheel scrolling](https://c.mql5.com/2/55/MQL5-avatar-doeasy-library-2.png)[DoEasy. Controls (Part 32): Horizontal ScrollBar, mouse wheel scrolling](https://www.mql5.com/en/articles/12849)

In the article, we will complete the development of the horizontal scrollbar object functionality. We will also make it possible to scroll the contents of the container by moving the scrollbar slider and rotating the mouse wheel, as well as make additions to the library, taking into account the new order execution policy and new runtime error codes in MQL5.

![Category Theory in MQL5 (Part 19): Naturality Square Induction](https://c.mql5.com/2/58/Category-Theory-p19-avatar.png)[Category Theory in MQL5 (Part 19): Naturality Square Induction](https://www.mql5.com/en/articles/13273)

We continue our look at natural transformations by considering naturality square induction. Slight restraints on multicurrency implementation for experts assembled with the MQL5 wizard mean we are showcasing our data classification abilities with a script. Principle applications considered are price change classification and thus its forecasting.

![Category Theory in MQL5 (Part 18): Naturality Square](https://c.mql5.com/2/57/category-theory-p18-avatar.png)[Category Theory in MQL5 (Part 18): Naturality Square](https://www.mql5.com/en/articles/13200)

This article continues our series into category theory by introducing natural transformations, a key pillar within the subject. We look at the seemingly complex definition, then delve into examples and applications with this series’ ‘bread and butter’; volatility forecasting.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/13008&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048836049385398174)

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