---
title: How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 2): Indicator Signals: Multi Timeframe Parabolic SAR Indicator
url: https://www.mql5.com/en/articles/13470
categories: Trading Systems, Expert Advisors
relevance_score: 13
scraped_at: 2026-01-22T17:11:20.349107
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/13470&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048835207571808153)

MetaTrader 5 / Examples


### Introduction

The definition of a Multi-Currency Expert Advisor in this article is one Expert Advisor or trading robot that can trade (open orders, close orders and manage orders for example: Trailing Stop Loss and Trailing Profit) for more than 1 symbol pair from only one symbol chart, where in this article Expert Advisor will trade for 30 pairs. This time we will use only 1 indicator, namely Parabolic SAR or iSAR in multi-timeframes starting from PERIOD\_M15 to PERIOD\_D1

We all know that multi-currency trading, both on the trading terminal and on the Strategy tester, is all possible with the power, capabilities and facilities provided by MQL5.

Therefore, the aim is to fulfill the essential needs of traders who want efficient and effective trading robots, so by relying on the strengths, capabilities and facilities provided by the highly reliable MQL5, we can create a simple Multi-Currency Expert Advisor which in this article uses Indicator Signals: Multi Timeframe Parabolic SAR or iSAR Indicator

### Plans and Features

**1\. Trading Currency Pairs.**

This Multi-Currency Expert Advisor is planned to trade on a Symbol or Pair as follows:

For Forex:

> EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURGBP,
>
> EURAUD, EURNZD, EURCAD, EURCHF, EURJPY, GBPAUD, GBPNZD, GBPCAD,
>
> GBPCHF,GBPJPY,AUDNZD,AUDCAD,AUDCHF,AUDJPY,NZDCAD,NZDCHF,
>
> NZDJPY, CADCHF, CADJPY, CHFJPY = 28 pairs

Plus 2 Metal pairs: XAUUSD (Gold) and XAGUSD (Silver)

Total is 30 pairs.

Note: All of these symbols or pairs are symbols or pairs commonly used by brokers. So, this Multi-Currency Expert Advisor will not work with brokers with symbol or pair names that have prefixes or suffixes.

**2\. Signal indicators.**

The Multi-Currency Expert Advisor will use 1 indicator signals but with 5 Timeframes, starting from PERIOD\_M15, PERIOD\_M30, PERIOD\_H1, PERIOD\_H4 and PERIOD\_D1

In this Expert Advisor, it does not use a fixed timeframe to calculate indicator signals, so there is no need to determine the signal calculation timeframe.

This means that the FXSAR\_MTF\_MCEA Expert Advisor can be used on any timeframe from PERIOD\_M1 to PERIOD\_MN1, and FXSAR\_MTF\_MCEA will still calculate signals based on iSAR PERIOD\_M15, PERIOD\_M30, PERIOD\_H1, PERIOD\_H4 and PERIOD\_D1

These five Parabolic SAR Timeframes will determine the signal for open orders.

Meanwhile, to close orders when the signal weakens, use the iSAR indicator PERIOD\_M15 provided the order is in profit condition.

And to do Trailing stop and Trailing profit, use the iSAR indicator PERIOD\_H1.

iSAR Signals Condition Strategy Formula:

> UP   = (PRICE\_LOW\[0\] is greater than iSAR Line) or PRICE\_LOW\[0\] > iSAR\[0\]
>
> DOWN = (PRICE-HIGH\[0\] is smaller than iSAR Line) or PRICE-HIGH\[0\] < iSAR\[0\]

Where to get a BUY signal or SELL signal:

> The five iSAR indicator timeframes must total 5 x UP for BUY and 5 x DOWN for SELL.

An illustration of the iSAR indikator untuk BUY atau SELL , can be seen in Figure 1

![iSAR_Signal_Buy and Sell](https://c.mql5.com/2/58/iSAR_Signal_Buy_and_Sell.png)

**3\. Trade & Order Management**

Trading management on this Multi-Currency Expert Advisor is given several options:

1\. Stop Loss Orders.

- Options: Use Order Stop Loss (Yes) or (No)

> If the Use Order Stop Loss (No) option is selected, then all orders will be opened without a stop loss.
>
> If the option Use Order Stop Loss (Yes):  again given the option: Use Automatic Calculation Stop Loss (Yes) or (No)
>
> If the option Automatic Calculation Stop Loss (Yes),  then the Stop Loss calculation will be performed automatically by the Expert.

> If the option Automatic Calculation Stop Loss (No),  then the trader must Input Stop Loss value in Pips.

> If the option Use Order Stop Loss (No):  then the Expert will check for each order opened, whether the signal condition is still good and order
>
> may be maintained in a profit OR condition the signal has weakened and the order needs to be closed to save
>
> profit or signal condition has reversed direction and order must be closed in a loss position.

2\. Take Profit orders.

- Options: Use Order Take Profit (Yes) or (No)

> If the Use Order Take Profit (No) option is selected, then all orders will be opened without take profit.

> If the option Use Order Take Profit (Yes):  again given the option: Use Automatic Calculation Order Take Profit (Yes) or (No)
>
> If the option Automatic Calculation Order Take Profit (Yes),  then the calculation of the Take Profit Order will be carried out automatically by the Expert.
>
> If the option Automatic Calculation Order Take Profit (No), then the trader must Input Order Take Profit value in Pips.

3\. Trailing Stop and Trailing Take Profit

- Options: Use Trailing SL/TP (Yes) or (No)

> If the Use Trailing SL/TP option is (No), then the Expert will not do trailing stop loss and trailing take profit. If the option Use Trailing SL/TP (Yes):   again given the option: Use Automatic Trailing (Yes) or (No) If the option Use Automatic Trailing (Yes), then the trailing stop will be executed by the Expert using the Parabolic SAR value PERIOD\_H1, at the same time by making trailing profit based on the variable value TPmin (Trailing Profit Value). If the option Use Automatic Trailing (No), then the trailing stop will be performed by the Expert using the value in the input property.
>
>  Note: The Expert will carry out a trailing take profit simultaneously with a trailing stop.

4\. Manual Order Management.

To support efficiency in this Multi-Currency Expert Advisor, several manual click buttons will be added.

> 1\. Set SL / TP All Orders

> When the trader input parameter sets Use Order Stop Loss (No) and/or Use Order Take Profit (No),
>
> but then the trader intends to use stop loss or take profit on all orders, then with just single click of the button "Set SL / TP All Orders" all orders will be modified and a stop loss will be applied and/or take profits.

> 2\. Close All Orders

> If a trader wants to close all orders, then with just single click of the button "Close All Orders" all open orders will be closed.

> 3\. Close All Orders Profit

> If a trader wants to close all orders that are already profitable, then with only single click of the button "Close All Orders Profit"
>
> then all open orders that are already profitable will be closed.

5\. Management Orders and Symbols Chart.

For Multi-Currency Expert Advisors who will trade 30 pairs from only one chart symbol, it will be very effective and efficient if a button panel is provided for all symbols, so traders can change charts or symbols with just one click.

### Implementation of planning in the MQL5 program

**1\. Program header and properties input.**

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

Enumeration YN is used for options (Yes) or (No) in expert input properties.

```
enum YN
 {
   No,
   Yes
 };
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

Expert input properties:

```
//---
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
input ulong               magicEA = 2023102;          // Expert ID (Magic Number)
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
    int              posCur1,
                     posCur2;
    //--
    double           LotPS;
    double           slv,
                     tpv,
                     pip,
                     xpip;
    double           differ;
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

    //------------
    int              DirectionMove(const string symbol);
    int              GetPSARSignalMTF(string symbol);
    int              PARSAR05(const string symbol);
    int              PARSARMTF(const string symbol,ENUM_TIMEFRAMES mtf);
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

    //-- FXSAR_MTF_MCEA Config --
    string           DIRI[],
                     AS30[];
    string           expname;
    int              hPar05[];   // Handle for the iSAR indicator for M5 Timeframe
    int              hPSAR[][5]; // Handle Indicator, where each Symbol has 5 arrays for Timeframe starting from TF_M15 to TF_D1
    int              ALO,
                     dgts,
                     arrsar,
                     arrsymbx;
    int              sall,
                     arper;
    ulong            slip;
    //--
    double           SARstep,
                     SARmaxi;
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
                     TFT05,
                     TFSAR[];
    //--
    bool             PanelExtra;
    //------------
                     MCEA(void);
                     ~MCEA(void);
    //------------
    virtual void     FXSAR_MTF_MCEA_Config(void);
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
    int              TFIndexArray(ENUM_TIMEFRAMES TF);
    int              GetOpenPosition(const string symbol);
    int              GetCloseInWeakSignal(const string symbol,int exis);
    int              ThisTime(const int reqmode);
    //--
    string           getUninitReasonText(int reasonCode);
    //------------
//---
  }; //-end class MCEA
```

The very first and foremost function in the Multi-Currency Expert Advisor work process that is called from OnInit() is FXSAR\_MTF\_MCEA\_Config().

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//---
   mc.FXSAR_MTF_MCEA_Config();
   //--
   return(INIT_SUCCEEDED);
//---
  } //-end OnInit()
```

In the FXSAR\_MTF\_MCEA\_Config() function, all symbols to be used are configured, all handle indicators used and some important functions of the include file header for the expert advisor workflow.

```
//+------------------------------------------------------------------+
//| Expert Configuration                                             |
//+------------------------------------------------------------------+
void MCEA::FXSAR_MTF_MCEA_Config(void)
  {
//---
    //--
    string All30[]={"EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY","EURGBP",
                    "EURAUD","EURNZD","EURCAD","EURCHF","EURJPY","GBPAUD","GBPNZD","GBPCAD",
                    "GBPCHF","GBPJPY","AUDNZD","AUDCAD","AUDCHF","AUDJPY","NZDCAD","NZDCHF",
                    "NZDJPY","CADCHF","CADJPY","CHFJPY","XAUUSD","XAGUSD"}; // 30 pairs
    //--
    sall=ArraySize(All30);
    ArrayResize(AS30,sall,sall);
    ArrayCopy(AS30,All30,0,0,WHOLE_ARRAY);
    //--
    arrsymbx=sall;
    ArraySymbolResize();
    ArrayCopy(DIRI,All30,0,0,WHOLE_ARRAY);
    for(int x=0; x<arrsymbx; x++) {SymbolSelect(DIRI[x],true);}
    pairs="Multi Currency 30 Pairs";
    //--
    TFT05=PERIOD_M5;
    ENUM_TIMEFRAMES TFA[]={PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H4,PERIOD_D1};
    TFArrays=ArraySize(TFA);
    ArrayResize(TFSAR,TFArrays,TFArrays);
    ArrayCopy(TFSAR,TFA,0,0,WHOLE_ARRAY);
    //--
    TFt=TFSAR[2];
    //--
    //-- iSAR Indicators handle for all symbol
    for(int x=0; x<arrsymbx; x++)
      {
        hPar05[x]=iSAR(DIRI[x],TFT05,SARstep,SARmaxi);  //-- Handle for the iSAR indicator for M5 Timeframe
        //--
        for(int i=0; i<TFArrays; i++)
          {
            hPSAR[x][i]=iSAR(DIRI[x],TFSAR[i],SARstep,SARmaxi); // Handle for iSAR Indicator array sequence of the requested timeframe
          }
      }
    //--
    ALO=(int)mc_account.LimitOrders()>arrsymbx ? arrsymbx : (int)mc_account.LimitOrders();
    //--
    LotPS=(double)ALO;
    //--
    mc_trade.SetExpertMagicNumber(magicEA);
    mc_trade.SetDeviationInPoints(slip);
    mc_trade.SetMarginMode();
    //--
    return;
//---
  } //-end FXSAR_MTF_MCEA_Config()
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
            if(mc.TradingToday())
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
```

**3\. How to get trading signals for open or close position?**

To get the indicator signal, we have to call the function GetOpenPosition(symbol) to get a trading signal for open position.

```
int MCEA::GetOpenPosition(const string symbol) // Signal Open Position
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    int dirmov=DirectionMove(symbol);
    int parsOp=GetPSARSignalMTF(symbol);
    //--
    if(parsOp==rise && dirmov==rise) ret=rise;
    if(parsOp==down && dirmov==down) ret=down;
    //--
    return(ret);
//---
  } //-end GetOpenPosition()
```

GetOpenPosition() function will call 2 signal functions and store in the variable OpOr\[\].

> 1\. DirectionMove(symbol);         //-- Function to check whether the price  move on the candlestick bar in the expert period.

> 2\. GetPSARSignalMTF(symbol);   //-- Function to calculation formula Parabolic iSAR on the requested Timeframe.

```
int MCEA::GetPSARSignalMTF(string symbol) // iSAR MTF signal calculation
  {
//---
    int mv=0;
    int rise=1,
        down=-1;
    //--
    int sarup=0,
        sardw=0;
    //--
    for(int x=0; x<TFArrays; x++) // The TFArrays variable has a value of 5 which is taken from the number of time frames from TF_M1 to TF_H1.
      {
        if(PARSARMTF(symbol,TFSAR[x])>0) sarup++;
        if(PARSARMTF(symbol,TFSAR[x])<0) sardw++;
      }
    //--
    if(sarup==TFArrays) mv=rise;
    if(sardw==TFArrays) mv=down;
    //--
    return(mv);
//---
  } //- end GetPSARSignalMTF()
```

The GetPSARSignalMTF() function will call a function PARSARMTF() that calculates the iSAR signal according to the requested timeframe.

As you can see, inside the PARSARMTF() function, we use and call 2 functions:

> 1\. int xx= PairsIdxArray(symbol)

> 2\. int tx=TFIndexArray(mtf).

The PairsIdxArray() function is used to get the name of the requested symbol, and the TFIndexArray() function is used to get the timeframe array sequence of the requested timeframe.

Then the appropriate indicator handle is called to get the buffers value of the iSAR indicator from that requested Timeframe.

```
int MCEA::PARSARMTF(const string symbol,ENUM_TIMEFRAMES mtf) // formula Parabolic iSAR on the requested Timeframe
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    int br=2;
    //--
    double PSAR[];
    ArrayResize(PSAR,br,br);
    ArraySetAsSeries(PSAR,true);
    int xx=PairsIdxArray(symbol);
    int tx=TFIndexArray(mtf);
    CopyBuffer(hPSAR[xx][tx],0,0,br,PSAR);
    //--
    double OPN0=iOpen(symbol,TFSAR[tx],0);
    double HIG0=iHigh(symbol,TFSAR[tx],0);
    double LOW0=iLow(symbol,TFSAR[tx],0);
    double CLS0=iClose(symbol,TFSAR[tx],0);
    //--
    if(PSAR[0]<LOW0 && CLS0>OPN0) ret=rise;
    if(PSAR[0]>HIG0 && CLS0<OPN0) ret=down;
    //--
    return(ret);
//---
  } //-end PARSARMTF()
```

```
int MCEA::PairsIdxArray(const string symbol)
  {
//---
    int pidx=0;
    //--
    for(int x=0; x<arrsymbx; x++)
      {
        if(DIRI[x]==symbol)
          {
            pidx=x;
            break;
          }
      }
    //--
    return(pidx);
//---
  } //-end PairsIdxArray()
```

```
int MCEA::TFIndexArray(ENUM_TIMEFRAMES TF)
  {
//---
    int res=-1;
    //--
    for(int x=0; x<TFArrays; x++)
      {
        if(TF==TFSAR[x])
          {
            res=x;
            break;
          }
      }
    //--
    return(res);
//---
  } //-end TFIndexArray()
```

**4\. ChartEvent Function**

To support effectiveness and efficiency in the use of Multi-Currency Expert Advisors, it is deemed necessary to create oneseveral manual buttons in managing orders and changing charts or symbols.

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
           int sx=mc.PairsIdxArray(sparam);
           ChangeChartSymbol(mc.AS30[sx],CCS);
           mc.PanelExtra=false;
         }
       //--
     }
    //--
    return;
//---
  } //-end OnChartEvent()
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
```

If the option on the expert property Display Trading Info on Chart is selected "Yes", then on the chart where the expert advisor is placed trading info will be displayed by calling the TradeInfo() function.

```
void MCEA::TradeInfo(void) // function: write comments on the chart
  {
//----
   Pips(Symbol());
   double spread=SymbolInfoInteger(Symbol(),SYMBOL_SPREAD)/xpip;
   //--
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
      "\n      :: Trade on TF      :  "+ EnumToString(TFt)+
      "\n      :: Today Trading   :  "+ TradingDay()+" : "+hariini+
      "\n     ------------------------------------------------------------"+
      "\n      :: Trading Pairs     :  "+pairs+
      "\n      :: BUY Market      :  "+string(oBm)+
      "\n      :: SELL Market     :  "+string(oSm)+
      "\n      :: Total Order       :  "+string(oBm+oSm)+
      "\n      :: Order Profit      :  "+DoubleToString(floatprofit,2)+
      "\n      :: Fixed Profit       :  "+DoubleToString(fixclprofit,2)+
      "\n      :: Float Money     :  "+DoubleToString(floatprofit,2)+
      "\n      :: Nett Profit        :  "+DoubleToString(floatprofit+fixclprofit,2);
   //---
   Comment(comm);
   ChartRedraw(0);
   return;
//----
  } //-end TradeInfo()
```

The Multi-Currency Expert Advisor FXSAR\_MTF\_MCEA interface looks like the following figure.

![FXSAR_MTF_MCEA_look](https://c.mql5.com/2/58/FXSAR_MTF_MCEA_look.png)

Under the Expert Advisor name FXSAR\_MTF\_MCEA as you can see there are buttons "M", "C" and "R"

If the "M" or "C" button is clicked, a manual click button panel will be displayed as shown below

![MCR_Combine](https://c.mql5.com/2/58/MCR_Combine.png)

If the M button is clicked, a manual click button panel will be displayed, then the trader can manage orders:

> 1\. Set SL/TP All Orders
>
> 2\. Close All Orders
>
> 3\. Close All Profits

If the C button is clicked, a panel button of 30 symbol names or pairs will be displayed and traders can click on one of the pair names or symbol names. If one of the pair names or symbols is clicked, the chart symbol will immediately be replaced with the symbol whose name was clicked.

If the R button is clicked, the Multi-Currency Expert Advisor FXSAR\_MTF\_MCEA will be removed from the chart

### Strategy Tester

As is known, the MetaTrader 5 terminal Strategy Tester supports and allows us to test strategies, trade on multiple symbols or test automatic trading for all available symbols and on all available timeframes.

So on this occasion we will test FXSAR\_MTF\_MCEA as a Multi-Timeframe and Multi-Currency Expert Advisor on the MetaTrader 5 Strategy Tester platform.

![](https://c.mql5.com/2/58/Strategy_Tester_FXSAR_MTF_MCEA.png)

### Conclusion

The conclusion in creating a Multi-Currency and Multi-Timeframe Expert Advisor using MQL5 is as follows:

1. It turns out that creating a Multi-Currency Expert Advisor in MQL5 is very simple and not much different from a Single-Currency Expert Advisor. But especially for Multi-Currency Expert Advisors with Multi Timeframes, it is a bit more complicated than with single timeframes.
2. Creating a Multi-Currency Expert Advisor will increase the efficiency and effectiveness of traders, because traders do not need to open many chart symbols for trading.
3. By applying the right trading strategy and calculating better indicator signals, the probability of profit will increase when compared to using a Single-Currency Expert Advisor. Because the losses that occur in one pair will be covered by profits in other pairs.
4. This FXSAR\_MTF\_MCEA Multi-Currency Expert Advisor is just an example to learn and develop ideas.
5. The test results on the Strategy Tester are still not good. Therefore, if a better strategy with more accurate signal calculations is implemented and adds some better timeframes, I believe the results will be better than the current strategy.

Note:

If you have an idea for creating a simple Multi-Currency Expert Advisor based on built-in MQL5 standard indicator signals, please suggest it in the comments.

Hopefully this article and the MQL5 Multi-Currency Expert Advisor program will be useful for traders in learning and developing ideas. Thanks for reading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13470.zip "Download all attachments in the single ZIP archive")

[FXSAR\_MTF\_MCEA.mq5](https://www.mql5.com/en/articles/download/13470/fxsar_mtf_mcea.mq5 "Download FXSAR_MTF_MCEA.mq5")(80.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 7): ZigZag with Awesome Oscillator Indicators Signal](https://www.mql5.com/en/articles/14329)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://www.mql5.com/en/articles/14051)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5): Bollinger Bands On Keltner Channel — Indicators Signal](https://www.mql5.com/en/articles/13861)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://www.mql5.com/en/articles/13770)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 3): Added symbols prefixes and/or suffixes and Trading Time Session](https://www.mql5.com/en/articles/13705)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 1): Indicator Signals based on ADX in combination with Parabolic SAR](https://www.mql5.com/en/articles/13008)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/455220)**
(10)


![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
4 Nov 2023 at 11:20

Of course it's slow!  It is doing 30 times the calculations of the "Fast EAs", try running 30 Fast EAs simultaneously and see what happens.  I'll bet this EA is much much faster.  If the winning percentages in the test run of 75+% hold up, who cares about speed when you are winning 3 out of 4 trades? just buy faster machines.

With Multi-currency EAs. manual optimization of the code is a **Necessity**.Look at loops to move static assignments, use local variables in loops and functions to reduce calculation, make sure that there are not multiple calls of the same function, do as much work as possible in the OnInit function by moving one time calls and static calculations into global variables etc, etc, etc.

To get around the symbol prefix suffix problem, consider using 2 variables for each symbol, pair for the 6 chr name and quoted for the full name with prefix and or suffix.  Either examine the name with a string function to set the two variables.

You may want to create an adaptive [Parabolic](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/psar "MetaTrader 5 Help: Parabolic SAR Indicator") Stop Loss that tracks the bars more closely, I think there are several adaptive PSAR indicators to use as a guide.

The work that Roberto has put into this EA should not be underestimated, it is very substantial.

![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
4 Nov 2023 at 18:41

**CapeCoddah [#](https://www.mql5.com/en/forum/455220#comment_50330471):**

Of course it's slow!  It is doing 30 times the calculations of the "Fast EAs", try running 30 Fast EAs simultaneously and see what happens.  I'll bet this EA is much much faster.  If the winning percentages in the test run of 75+% hold up, who cares about speed when you are winning 3 out of 4 trades? just buy faster machines.

With Multi-currency EAs. manual optimization of the code is a **Necessity**.Look at loops to move static assignments, use local variables in loops and functions to reduce calculation, make sure that there are not multiple calls of the same function, do as much work as possible in the OnInit function by moving one time calls and static calculations into global variables etc, etc, etc.

To get around the symbol prefix suffix problem, consider using 2 variables for each symbol, pair for the 6 chr name and quoted for the full name with prefix and or suffix.  Either examine the name with a string function to set the two variables.

You may want to create an adaptive [Parabolic](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/psar "MetaTrader 5 Help: Parabolic SAR Indicator") Stop Loss that tracks the bars more closely, I think there are several adaptive PSAR indicators to use as a guide.

The work that Roberto has put into this EA should not be underestimated, it is very substantial.

Thank you for your support. I will create an article to add automatic detection and handling of brokers with special symbol names, prefixes and/or suffixes.

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
7 Nov 2023 at 11:52

Roberto,

Bad News, I ran your EA on EURUSD H4 from 1/1/2023 to 11/1/2023 with $1,000 initial balance.  The EA bankrupted the account in less than 3 months.  With $10,000, it ran completely but lost  $8,250.  The graph shows consistent losses from start to end with no sharp peaks or valleys.

First Don't despair!  FX trading is tough and it is tougher still to design a multi-currency EA.  I know, I am in the midst of transforming one from MQ4 to MQ5.

It may be time to implement a variable pair capability to enable the specification of pairs to provide you with the ability to test on only one pair.  The easiest way is to have your pair string be an input item and use STRSPLIT to separate each pair in the string to enable loading of your pairs.  A better approach is to use your 30 pair display to allow the user to select the pairs for the run by clocking on them and changing color.  There are two recent GUI articles, GUI: Tips and Tricks...... 10/5/2023 and another set of Articles on Moveable GUIs.  I use the latter but I think the Tips and Tricks may be better and more complete.  You should also use the GUIs to display your data, which I think is excellent, instead of using the [Comment function](https://www.mql5.com/en/docs/common/comment "MQL5 documentation: Comment function").

I am a firm believer in Pareto's Law: 80% of a characteristic comes from 20% of the elements.  This means that 80% of the overall profits come from 6 pairs and correspondingly 6 pairs contribute to 80% of the losses.

Enhanced Strategy Tester  statistics for individual pairs in a multi-currency test are mandatory to enable identification of problem areas and Pareto's Law.  Elements of the BackTest tab are needed on a pair level i.e Net Profit, Gross Profit Gross Loss etc. etc. etc.

I still think an adaptive process for the SAR would provide a improvement in your profits.  If you look at your Buy/Sell chart in your textabove, an adaptive function  that increases the acceleration speed of the SAR based on the Bar size increase would flex the SAR into giving you increased profits in the first 4 Buy/Sell illustrations on the chart.  This adaptive flex would provide two benefits:

it would provide an increase in profits of perhaps $5-$10 by closing the trade sooner.  **More Importantly, it would allow the next trade to open $5-$10 earlier.  Thus, the impact of the flex could be $10-$20 overall for each trade.**  However, it may also cause a lot of additional losing trades to be placed with a corresponding decrease in overall profits.

Concentrate on these targets and optimal time frames and your profitability will substantially increase.  I admit I haven't figured out a dynamic evaluation process yet.

![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
7 Nov 2023 at 15:22

**CapeCoddah [#](https://www.mql5.com/en/forum/455220#comment_50382707):**

I am a firm believer in Pareto's Law: 80% of a characteristic comes from 20% of the elements.  This means that 80% of the overall profits come from 6 pairs and correspondingly 6 pairs contribute to 80% of the losses.

Thank you for your input.

As I said in conclusions 4 and 5:

This FXSAR\_MTF\_MCEA Multi-Currency Expert Advisor is just an example to learn and develop ideas.

The test results on the Strategy Tester are still not good. Therefore, if a better strategy with more accurate signal calculations is implemented and adds some better timeframes, I believe the results will be better than the current strategy.

So, it's up to you to upgrade using what you say is an adaptive function to get better results.

![CFXfinanceuk Mottana](https://c.mql5.com/avatar/avatar_na2.png)

**[CFXfinanceuk Mottana](https://www.mql5.com/en/users/cfxfinanceuk)**
\|
8 Aug 2024 at 09:50

Hello Roberto,

very interesting, i like multi timeframe systems.

Sorry but I do not understand how I can change the time frames of the single SAR and if the SAR has a fixed value calculation.

Is there a way to Buy and Sell every timeframe (instead of wating for all being on one side)?

In this case I could have a sell on 1 minute and a buy on 5 min etc, supposing 0.1 each I will have a variable quantity long and short.

I tried the testing on [GOLD](https://www.mql5.com/en/quotes/metals/XAUUSD "XAUUSD chart: technical analysis") since 1.1.24 but nothing happens, no trades.

Any suggestion? You can write me in private too.

Thanks so much.

Marco

![Studying PrintFormat() and applying ready-made examples](https://c.mql5.com/2/56/printformat-avatar.png)[Studying PrintFormat() and applying ready-made examples](https://www.mql5.com/en/articles/12905)

The article will be useful for both beginners and experienced developers. We will look at the PrintFormat() function, analyze examples of string formatting and write templates for displaying various information in the terminal log.

![GUI: Tips and Tricks for creating your own Graphic Library in MQL](https://c.mql5.com/2/58/gui_tips_and_tricks_avatar.png)[GUI: Tips and Tricks for creating your own Graphic Library in MQL](https://www.mql5.com/en/articles/13169)

We'll go through the basics of GUI libraries so that you can understand how they work or even start making your own.

![Classification models in the Scikit-Learn library and their export to ONNX](https://c.mql5.com/2/58/Scikit_learn_to-ONNX_avatar.png)[Classification models in the Scikit-Learn library and their export to ONNX](https://www.mql5.com/en/articles/13451)

In this article, we will explore the application of all classification models available in the Scikit-Learn library to solve the classification task of Fisher's Iris dataset. We will attempt to convert these models into ONNX format and utilize the resulting models in MQL5 programs. Additionally, we will compare the accuracy of the original models with their ONNX versions on the full Iris dataset.

![Category Theory in MQL5 (Part 22): A different look at Moving Averages](https://c.mql5.com/2/58/Category-Theory-p22-avatar.png)[Category Theory in MQL5 (Part 22): A different look at Moving Averages](https://www.mql5.com/en/articles/13416)

In this article we attempt to simplify our illustration of concepts covered in these series by dwelling on just one indicator, the most common and probably the easiest to understand. The moving average. In doing so we consider significance and possible applications of vertical natural transformations.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/13470&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048835207571808153)

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